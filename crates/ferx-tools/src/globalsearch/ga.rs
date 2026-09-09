//! The genetic algorithm (#1185), written over an abstract fitness oracle.
//!
//! Nothing here knows what a gene *is*: a genome is one allele index per
//! axis, the [`Oracle`] turns a batch of genomes into fitnesses (lower is
//! better), and the search is the same whether the axes are absorption
//! routes and covariate forms or a test function. That is what lets the
//! algorithm be tested against exhaustive enumeration on a known landscape
//! without a single fit.
//!
//! # The algorithm
//!
//! pyDarwin's GA (DEAP underneath), integer-coded rather than bit-coded —
//! one gene per axis holding an allele index, which is the same search
//! space without the bit strings that encode nothing between two alleles.
//!
//! 1. An initial population of distinct random genomes.
//! 2. Each generation: evaluate, keep the `elites` best, then fill the
//!    population by **tournament selection** on the *shared* fitness,
//!    **one-point crossover** with probability `crossover_rate`, and
//!    **mutation** with probability `mutation_rate` per child — each gene
//!    replaced by another allele with probability
//!    `gene_mutation_probability`.
//! 3. Every `downhill_period` generations, and once more at the end when
//!    `final_downhill` is set, the best `niches` individuals that lie at
//!    least `niche_radius` apart (Hamming distance) are polished by a
//!    **one-gene downhill search**: every single-gene neighbour is
//!    evaluated, the best replaces the individual if it improves, until
//!    none does. The polished individuals rejoin the population.
//!
//! **Fitness sharing** is what keeps the population from collapsing onto
//! one basin: for selection only, an individual's fitness is raised by
//! `niche_penalty` times its crowding, Goldberg–Richardson's
//! `Σ 1 − (d/r)^α` over the other individuals within `niche_radius`. Ranking
//! and the report use the raw fitness.
//!
//! Every fitness the oracle returns is cached by genome inside the oracle
//! (that is the caller's contract), so a genome the GA proposes twice costs
//! one fit; the counts in [`GaOutcome`] are of *proposals*, the oracle knows
//! how many were new.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::Deserialize;

/// One allele index per axis.
pub type Genome = Vec<usize>;

/// The GA's knobs — pyDarwin's option names where the option exists there.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GaOptions {
    /// Individuals per generation. Capped at the size of the space, since a
    /// population cannot hold more distinct genomes than there are.
    pub population_size: usize,
    /// Generations after the initial one. `0` evaluates the initial
    /// population only.
    pub generations: usize,
    /// Probability that a pair of parents is crossed rather than copied.
    pub crossover_rate: f64,
    /// Probability that a child is mutated at all.
    pub mutation_rate: f64,
    /// Given a mutated child, the probability each gene is replaced by
    /// another allele of its axis.
    pub gene_mutation_probability: f64,
    /// The best individuals carried unchanged into the next generation.
    pub elites: usize,
    /// Tournament size for parent selection.
    pub tournament_size: usize,
    /// Run the downhill search every this many generations; `0` never.
    pub downhill_period: usize,
    /// How many individuals the downhill search polishes each time.
    pub niches: usize,
    /// Hamming distance below which two individuals crowd each other, for
    /// fitness sharing and for picking distinct downhill seeds.
    pub niche_radius: usize,
    /// The sharing charge per unit of crowding, on the fitness scale.
    pub niche_penalty: f64,
    /// The sharing exponent `α`.
    pub sharing_alpha: f64,
    /// Polish the best individuals once more after the last generation.
    pub final_downhill: bool,
    /// The RNG seed: the same seed on the same space proposes the same
    /// genomes, which is what makes a `--resume` reuse its journal.
    pub seed: u64,
}

impl Default for GaOptions {
    fn default() -> Self {
        Self {
            population_size: 20,
            generations: 10,
            crossover_rate: 0.95,
            mutation_rate: 0.95,
            gene_mutation_probability: 0.1,
            elites: 2,
            tournament_size: 2,
            downhill_period: 5,
            niches: 2,
            niche_radius: 2,
            niche_penalty: 20.0,
            sharing_alpha: 0.1,
            final_downhill: true,
            seed: 12345,
        }
    }
}

impl GaOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.population_size < 2 {
            return Err("[globalsearch] population_size must be at least 2".into());
        }
        for (name, p) in [
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
            ("gene_mutation_probability", self.gene_mutation_probability),
        ] {
            if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
                return Err(format!(
                    "[globalsearch] {name} = {p}: must be a probability in [0, 1]"
                ));
            }
        }
        if self.tournament_size == 0 {
            return Err("[globalsearch] tournament_size must be at least 1".into());
        }
        if self.elites >= self.population_size {
            return Err(format!(
                "[globalsearch] elites = {} must be smaller than population_size = {}",
                self.elites, self.population_size
            ));
        }
        if !(self.niche_penalty.is_finite() && self.niche_penalty >= 0.0) {
            return Err(format!(
                "[globalsearch] niche_penalty = {}: must be finite and non-negative",
                self.niche_penalty
            ));
        }
        if !(self.sharing_alpha.is_finite() && self.sharing_alpha > 0.0) {
            return Err(format!(
                "[globalsearch] sharing_alpha = {}: must be finite and positive",
                self.sharing_alpha
            ));
        }
        if self.downhill_period > 0 && self.niches == 0 {
            return Err(
                "[globalsearch] niches must be at least 1 when a downhill search runs".into(),
            );
        }
        Ok(())
    }
}

/// What turns genomes into fitnesses.
///
/// The oracle owns the cache: a genome it has seen is answered without a
/// fit, and [`cancelled`](Self::cancelled) reports a cancellation the
/// search must stop on. `evaluate` returns one fitness per genome, in
/// order, every one finite (a crashed candidate is the crash value, not
/// `NaN`).
pub trait Oracle {
    /// `what` names the batch for the journal: `generation-3`,
    /// `downhill-3-1`.
    fn evaluate(&mut self, what: &str, genomes: &[Genome]) -> Result<Vec<f64>, String>;
    fn cancelled(&self) -> bool;
}

/// One generation's summary, for the progress line and the report.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// `0` is the initial population.
    pub index: usize,
    pub best: Genome,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    /// Individuals the downhill search moved after this generation.
    pub polished: usize,
}

/// What the GA found.
#[derive(Debug, Clone, PartialEq)]
pub struct GaOutcome {
    pub best: Genome,
    pub best_fitness: f64,
    pub generations: Vec<Generation>,
    /// Stopped on the oracle's cancellation; `best` is the best so far.
    pub cancelled: bool,
}

/// The size of the space: the product of the allele counts. `None` when it
/// overflows — no search runs over such a space.
pub fn space_size(alleles: &[usize]) -> Option<usize> {
    alleles
        .iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(*n))
}

/// Run the GA over a space whose axis `i` has `alleles[i]` alleles.
///
/// Every axis must have at least one allele and the space at least two
/// points; a one-point space has nothing to search.
pub fn run(
    alleles: &[usize],
    options: &GaOptions,
    oracle: &mut dyn Oracle,
) -> Result<GaOutcome, String> {
    options.validate()?;
    if alleles.contains(&0) {
        return Err("the search space has an axis with no allele".into());
    }
    let size = space_size(alleles).ok_or("the search space is too large to count")?;
    if size < 2 {
        return Err("the search space has a single point; there is nothing to search".into());
    }
    let mut rng = StdRng::seed_from_u64(options.seed);
    let pop_size = options.population_size.min(size);
    let elites = options.elites.min(pop_size - 1);

    // ── the initial population: distinct random genomes ─────────────────
    let mut population: Vec<Genome> = Vec::with_capacity(pop_size);
    // Bounded: a space only slightly larger than the population makes a
    // distinct draw rare, and enumeration is the right fallback then.
    let mut attempts = 0usize;
    while population.len() < pop_size && attempts < 50 * pop_size {
        attempts += 1;
        let g = random_genome(alleles, &mut rng);
        if !population.contains(&g) {
            population.push(g);
        }
    }
    if population.len() < pop_size {
        population = fill_distinct(alleles, population, pop_size, &mut rng);
    }

    let mut generations: Vec<Generation> = Vec::new();
    let mut best: Option<(Genome, f64)> = None;
    let mut cancelled = false;

    for index in 0..=options.generations {
        let fitness = oracle.evaluate(&format!("generation-{index}"), &population)?;
        let mut fitness = fitness;
        if oracle.cancelled() {
            cancelled = true;
        }
        record_best(&mut best, &population, &fitness);

        // ── downhill ────────────────────────────────────────────────────
        let mut polished = 0usize;
        let last = index == options.generations;
        let due = options.downhill_period > 0 && index > 0 && index % options.downhill_period == 0;
        if !cancelled && (due || (last && options.final_downhill)) {
            for (k, i) in niche_seeds(&population, &fitness, options)
                .into_iter()
                .enumerate()
            {
                let (g, f, moved) = downhill(
                    alleles,
                    population[i].clone(),
                    fitness[i],
                    &format!("downhill-{index}-{}", k + 1),
                    oracle,
                )?;
                if moved {
                    polished += 1;
                    population[i] = g;
                    fitness[i] = f;
                }
                if oracle.cancelled() {
                    cancelled = true;
                    break;
                }
            }
            record_best(&mut best, &population, &fitness);
        }

        let best_i = argmin(&fitness);
        generations.push(Generation {
            index,
            best: population[best_i].clone(),
            best_fitness: fitness[best_i],
            mean_fitness: fitness.iter().sum::<f64>() / fitness.len() as f64,
            polished,
        });
        if cancelled || last {
            break;
        }

        // ── the next generation ─────────────────────────────────────────
        let shared = shared_fitness(&population, &fitness, options);
        let mut order: Vec<usize> = (0..population.len()).collect();
        order.sort_by(|a, b| fitness[*a].total_cmp(&fitness[*b]));
        let mut next: Vec<Genome> = order[..elites]
            .iter()
            .map(|i| population[*i].clone())
            .collect();
        let mut stalls = 0usize;
        while next.len() < pop_size {
            let a = tournament(&shared, options.tournament_size, &mut rng);
            let b = tournament(&shared, options.tournament_size, &mut rng);
            let (mut c1, mut c2) = if rng.random::<f64>() < options.crossover_rate {
                crossover(&population[a], &population[b], &mut rng)
            } else {
                (population[a].clone(), population[b].clone())
            };
            for child in [&mut c1, &mut c2] {
                if rng.random::<f64>() < options.mutation_rate {
                    mutate(child, alleles, options.gene_mutation_probability, &mut rng);
                }
            }
            for child in [c1, c2] {
                if next.len() < pop_size {
                    if next.contains(&child) {
                        stalls += 1;
                    } else {
                        next.push(child);
                        stalls = 0;
                    }
                }
            }
            // A population that cannot find a new genome by breeding — a
            // small space, or one it has converged on — is topped up with
            // fresh distinct draws rather than looping.
            if stalls > 20 * pop_size {
                next = fill_distinct(alleles, next, pop_size, &mut rng);
            }
        }
        population = next;
    }

    let (best, best_fitness) = best.expect("at least one generation was evaluated");
    Ok(GaOutcome {
        best,
        best_fitness,
        generations,
        cancelled,
    })
}

fn random_genome(alleles: &[usize], rng: &mut StdRng) -> Genome {
    alleles.iter().map(|n| rng.random_range(0..*n)).collect()
}

/// Top `population` up to `n` distinct genomes by walking the space from a
/// random offset, so a space barely larger than the population is still
/// filled without an unbounded rejection loop.
fn fill_distinct(
    alleles: &[usize],
    mut population: Vec<Genome>,
    n: usize,
    rng: &mut StdRng,
) -> Vec<Genome> {
    let size = space_size(alleles).unwrap_or(usize::MAX);
    let start = rng.random_range(0..size.max(1));
    let mut k = 0usize;
    while population.len() < n && k < size {
        let g = unrank((start + k) % size, alleles);
        if !population.contains(&g) {
            population.push(g);
        }
        k += 1;
    }
    population
}

/// The `k`-th genome of the space in mixed-radix order, last axis fastest.
pub fn unrank(mut k: usize, alleles: &[usize]) -> Genome {
    let mut g = vec![0; alleles.len()];
    for (i, n) in alleles.iter().enumerate().rev() {
        g[i] = k % n;
        k /= n;
    }
    g
}

/// Every genome of the space, in mixed-radix order.
pub fn enumerate(alleles: &[usize]) -> Vec<Genome> {
    match space_size(alleles) {
        Some(size) => (0..size).map(|k| unrank(k, alleles)).collect(),
        None => Vec::new(),
    }
}

fn argmin(fitness: &[f64]) -> usize {
    fitness
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("a non-empty population")
}

fn record_best(best: &mut Option<(Genome, f64)>, population: &[Genome], fitness: &[f64]) {
    let i = argmin(fitness);
    // Strictly better only, so the first genome to reach a fitness keeps
    // the title and the outcome is deterministic across tie orders.
    if best.as_ref().is_none_or(|(_, f)| fitness[i] < *f) {
        *best = Some((population[i].clone(), fitness[i]));
    }
}

pub fn hamming(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Goldberg–Richardson sharing on the Hamming distance: each individual's
/// fitness plus `niche_penalty · Σ_j (1 − (d_ij / r)^α)` over the *other*
/// individuals closer than `r`. With `niche_radius = 0` nothing crowds.
fn shared_fitness(population: &[Genome], fitness: &[f64], options: &GaOptions) -> Vec<f64> {
    let r = options.niche_radius as f64;
    (0..population.len())
        .map(|i| {
            if options.niche_radius == 0 || options.niche_penalty == 0.0 {
                return fitness[i];
            }
            let crowd: f64 = population
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, g)| hamming(&population[i], g))
                .filter(|d| (*d as f64) < r)
                .map(|d| 1.0 - (d as f64 / r).powf(options.sharing_alpha))
                .sum();
            fitness[i] + options.niche_penalty * crowd
        })
        .collect()
}

fn tournament(shared: &[f64], size: usize, rng: &mut StdRng) -> usize {
    let mut best = rng.random_range(0..shared.len());
    for _ in 1..size {
        let i = rng.random_range(0..shared.len());
        if shared[i] < shared[best] {
            best = i;
        }
    }
    best
}

/// One-point crossover. A one-axis genome has no interior point and the
/// parents come back as they are.
fn crossover(a: &Genome, b: &Genome, rng: &mut StdRng) -> (Genome, Genome) {
    if a.len() < 2 {
        return (a.clone(), b.clone());
    }
    let point = rng.random_range(1..a.len());
    let mut c1 = a[..point].to_vec();
    c1.extend_from_slice(&b[point..]);
    let mut c2 = b[..point].to_vec();
    c2.extend_from_slice(&a[point..]);
    (c1, c2)
}

/// Replace each gene, with probability `p`, by a *different* allele of its
/// axis (an axis with one allele cannot mutate).
fn mutate(g: &mut Genome, alleles: &[usize], p: f64, rng: &mut StdRng) {
    for (i, n) in alleles.iter().enumerate() {
        if *n < 2 || rng.random::<f64>() >= p {
            continue;
        }
        let shift = rng.random_range(1..*n);
        g[i] = (g[i] + shift) % n;
    }
}

/// The indices of the best `niches` individuals that are at least
/// `niche_radius` apart, greedily by fitness — the downhill seeds.
fn niche_seeds(population: &[Genome], fitness: &[f64], options: &GaOptions) -> Vec<usize> {
    let mut order: Vec<usize> = (0..population.len()).collect();
    order.sort_by(|a, b| fitness[*a].total_cmp(&fitness[*b]));
    let mut seeds: Vec<usize> = Vec::new();
    for i in order {
        if seeds.len() >= options.niches {
            break;
        }
        let far = seeds
            .iter()
            .all(|s| hamming(&population[*s], &population[i]) >= options.niche_radius.max(1));
        if far {
            seeds.push(i);
        }
    }
    seeds
}

/// Every single-gene neighbour of `g`, in axis order then allele order.
pub fn neighbours(g: &Genome, alleles: &[usize]) -> Vec<Genome> {
    let mut out = Vec::new();
    for (i, n) in alleles.iter().enumerate() {
        for a in 0..*n {
            if a != g[i] {
                let mut h = g.clone();
                h[i] = a;
                out.push(h);
            }
        }
    }
    out
}

/// Steepest one-gene descent from `g`: evaluate every neighbour, move to
/// the best if it improves on the current fitness, repeat. Bounded by the
/// number of points a descent can visit without cycling — every move is a
/// strict improvement, so that bound is the space size, but a cap keeps a
/// pathological oracle from running forever.
fn downhill(
    alleles: &[usize],
    mut g: Genome,
    mut f: f64,
    what: &str,
    oracle: &mut dyn Oracle,
) -> Result<(Genome, f64, bool), String> {
    let mut moved = false;
    for round in 1..=64 {
        let candidates = neighbours(&g, alleles);
        if candidates.is_empty() {
            break;
        }
        let fitness = oracle.evaluate(&format!("{what}-{round}"), &candidates)?;
        let i = argmin(&fitness);
        if fitness[i] < f {
            g = candidates[i].clone();
            f = fitness[i];
            moved = true;
        } else {
            break;
        }
        if oracle.cancelled() {
            break;
        }
    }
    Ok((g, f, moved))
}

#[cfg(test)]
#[path = "ga_tests.rs"]
mod tests;
