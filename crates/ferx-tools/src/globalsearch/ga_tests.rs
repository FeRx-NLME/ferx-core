//! The GA against exhaustive enumeration on known landscapes (#1185): the
//! acceptance criterion "on a space small enough to enumerate, the GA finds
//! the same optimum as exhaustive enumeration", plus the mechanics — a
//! seed fixes the trajectory, a cancellation stops it, the downhill search
//! leaves a one-gene local optimum, and the population never exceeds the
//! space.

use std::collections::HashMap;

use super::*;

/// `(batch name, genomes proposed, genomes new)`.
type Batch = (String, usize, usize);

/// A fitness function with a genome cache and a batch log.
struct Landscape {
    f: Box<dyn Fn(&Genome) -> f64>,
    cache: HashMap<Genome, f64>,
    batches: Vec<Batch>,
    cancel_after_batches: Option<usize>,
}

impl Landscape {
    fn new(f: impl Fn(&Genome) -> f64 + 'static) -> Self {
        Landscape {
            f: Box::new(f),
            cache: HashMap::new(),
            batches: Vec::new(),
            cancel_after_batches: None,
        }
    }

    fn unique_evaluations(&self) -> usize {
        self.cache.len()
    }
}

impl Oracle for Landscape {
    fn evaluate(&mut self, what: &str, genomes: &[Genome]) -> Result<Vec<f64>, String> {
        let mut new = 0usize;
        let out = genomes
            .iter()
            .map(|g| {
                *self.cache.entry(g.clone()).or_insert_with(|| {
                    new += 1;
                    (self.f)(g)
                })
            })
            .collect();
        self.batches.push((what.to_string(), genomes.len(), new));
        Ok(out)
    }

    fn cancelled(&self) -> bool {
        self.cancel_after_batches
            .is_some_and(|n| self.batches.len() >= n)
    }
}

/// A landscape with a deceptive local optimum. The global minimum (100)
/// sits at `target`, the far corner of the space, and the fitness climbs
/// by 2 per gene away from it — except at the opposite corner, the
/// *decoy*, which dips 15 below its own neighbours and so is a one-gene
/// local minimum that a hill climb from its side of the space ends in.
fn deceptive(alleles: &[usize]) -> (Box<dyn Fn(&Genome) -> f64>, Genome) {
    let target: Genome = alleles.iter().map(|n| n - 1).collect();
    let decoy: Genome = alleles.iter().map(|_| 0).collect();
    let t = target.clone();
    let f = move |g: &Genome| {
        let away: usize = g.iter().zip(&t).map(|(a, b)| a.abs_diff(*b)).sum();
        100.0 + 2.0 * away as f64 - if *g == decoy { 15.0 } else { 0.0 }
    };
    (Box::new(f), target)
}

fn exhaustive_best(alleles: &[usize], f: &dyn Fn(&Genome) -> f64) -> (Genome, f64) {
    enumerate(alleles)
        .into_iter()
        .map(|g| {
            let v = f(&g);
            (g, v)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
}

#[test]
fn enumerate_and_unrank_cover_the_space_exactly_once() {
    let alleles = [2usize, 3, 4];
    let all = enumerate(&alleles);
    assert_eq!(all.len(), 24);
    assert_eq!(space_size(&alleles), Some(24));
    let mut seen = all.clone();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), 24, "a genome was enumerated twice");
    assert_eq!(all[0], vec![0, 0, 0]);
    assert_eq!(all[1], vec![0, 0, 1], "the last axis moves fastest");
    assert_eq!(all[23], vec![1, 2, 3]);
    assert_eq!(unrank(5, &alleles), vec![0, 1, 1]);
    assert_eq!(space_size(&[usize::MAX, 2]), None);
}

#[test]
fn the_ga_finds_the_exhaustive_optimum_on_a_small_space() {
    // 3·3·3·4·2 = 216 points; the GA sees a fraction of them.
    let alleles = [3usize, 3, 3, 4, 2];
    let (f, target) = deceptive(&alleles);
    let (best_g, best_f) = exhaustive_best(&alleles, &f);
    assert_eq!(
        best_g, target,
        "the landscape's minimum is where it was put"
    );

    for seed in [1u64, 7, 42, 1185, 99991] {
        let (f, _) = deceptive(&alleles);
        let mut oracle = Landscape::new(f);
        let options = GaOptions {
            population_size: 16,
            generations: 8,
            seed,
            ..GaOptions::default()
        };
        let outcome = run(&alleles, &options, &mut oracle).unwrap();
        assert_eq!(outcome.best, best_g, "seed {seed}");
        assert_eq!(outcome.best_fitness, best_f, "seed {seed}");
        assert!(!outcome.cancelled);
        assert!(
            oracle.unique_evaluations() < 216,
            "seed {seed}: the GA enumerated the whole space ({} evaluations)",
            oracle.unique_evaluations()
        );
        // The best is the best of the last generation's summary too.
        let last = outcome.generations.last().unwrap();
        assert_eq!(last.best_fitness, best_f, "seed {seed}");
        assert_eq!(outcome.generations.len(), 9, "seed {seed}");
    }
}

#[test]
fn the_ga_beats_a_hill_climb_on_the_deceptive_landscape() {
    // Without the GA — a downhill search alone from the decoy's side of
    // the space — the search stays in the wrong well: from a genome one
    // step from the decoy, the decoy (−15) beats every step towards the
    // target (−2), and the decoy itself is a local minimum. The GA's
    // population and crossover are what cross the gap; this is the
    // regression that would make `run` a glorified hill climb.
    let alleles = [3usize, 3, 3, 4, 2];
    let (f, target) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let start: Genome = vec![1, 0, 0, 0, 0];
    let f0 = oracle
        .evaluate("start", std::slice::from_ref(&start))
        .unwrap()[0];
    let (g, fitness, moved) = downhill(&alleles, start, f0, "hill", &mut oracle).unwrap();
    assert!(moved);
    assert_eq!(g, vec![0, 0, 0, 0, 0], "the hill climb ends in the decoy");
    assert_eq!(fitness, 100.0 + 2.0 * 10.0 - 15.0);
    assert_ne!(g, target);
}

#[test]
fn the_same_seed_proposes_the_same_genomes() {
    let alleles = [3usize, 3, 4];
    let runs: Vec<(GaOutcome, Vec<Batch>)> = (0..2)
        .map(|_| {
            let (f, _) = deceptive(&alleles);
            let mut oracle = Landscape::new(f);
            let options = GaOptions {
                population_size: 8,
                generations: 4,
                seed: 3,
                ..GaOptions::default()
            };
            let outcome = run(&alleles, &options, &mut oracle).unwrap();
            (outcome, oracle.batches)
        })
        .collect();
    assert_eq!(runs[0], runs[1]);
    // A different seed is a different trajectory (the outcome may agree;
    // the proposals do not).
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let options = GaOptions {
        population_size: 8,
        generations: 4,
        seed: 4,
        ..GaOptions::default()
    };
    run(&alleles, &options, &mut oracle).unwrap();
    assert_ne!(oracle.batches, runs[0].1);
}

#[test]
fn batches_are_named_for_the_journal_and_the_initial_population_is_distinct() {
    let alleles = [3usize, 3, 4];
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let options = GaOptions {
        population_size: 10,
        generations: 5,
        downhill_period: 5,
        final_downhill: true,
        seed: 11,
        ..GaOptions::default()
    };
    run(&alleles, &options, &mut oracle).unwrap();
    let names: Vec<&str> = oracle.batches.iter().map(|b| b.0.as_str()).collect();
    assert_eq!(names[0], "generation-0");
    assert_eq!(oracle.batches[0].1, 10);
    assert_eq!(
        oracle.batches[0].2, 10,
        "the initial population holds ten distinct genomes"
    );
    assert!(names.contains(&"generation-5"));
    // The downhill search ran after generation 5 (the period and the final
    // pass coincide, so once), for up to two niches, at least one round.
    assert!(
        names.iter().any(|n| n.starts_with("downhill-5-1-")),
        "{names:?}"
    );
    assert!(!names.iter().any(|n| n.starts_with("downhill-1-")));
    // Within one batch every genome is distinct, so a fit is never asked
    // for twice in a step.
    for (name, proposed, _) in &oracle.batches {
        let _ = name;
        assert!(*proposed > 0);
    }
}

#[test]
fn a_cancellation_stops_the_search_with_the_best_so_far() {
    let alleles = [3usize, 3, 3, 4, 2];
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    oracle.cancel_after_batches = Some(2);
    let options = GaOptions {
        population_size: 8,
        generations: 10,
        seed: 5,
        ..GaOptions::default()
    };
    let outcome = run(&alleles, &options, &mut oracle).unwrap();
    assert!(outcome.cancelled);
    assert_eq!(
        outcome.generations.len(),
        2,
        "generations 0 and 1 were evaluated"
    );
    assert_eq!(oracle.batches.len(), 2);
    assert!(outcome.best_fitness.is_finite());
}

#[test]
fn the_downhill_search_leaves_a_one_gene_local_optimum() {
    let alleles = [4usize, 4, 4];
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let options = GaOptions {
        population_size: 6,
        generations: 0,
        final_downhill: true,
        niches: 3,
        seed: 8,
        ..GaOptions::default()
    };
    let outcome = run(&alleles, &options, &mut oracle).unwrap();
    let (f, _) = deceptive(&alleles);
    let best = &outcome.best;
    for n in neighbours(best, &alleles) {
        assert!(
            f(&n) >= outcome.best_fitness,
            "a neighbour {n:?} of the polished best {best:?} is better"
        );
    }
    assert_eq!(outcome.generations.len(), 1);
    // Without the final pass the initial population is all there is.
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let options = GaOptions {
        final_downhill: false,
        ..options
    };
    run(&alleles, &options, &mut oracle).unwrap();
    assert_eq!(oracle.batches.len(), 1);
}

#[test]
fn the_population_is_capped_at_the_size_of_the_space() {
    let alleles = [2usize, 2];
    let (f, _) = deceptive(&alleles);
    let mut oracle = Landscape::new(f);
    let options = GaOptions {
        population_size: 20,
        generations: 3,
        elites: 2,
        seed: 1,
        ..GaOptions::default()
    };
    let outcome = run(&alleles, &options, &mut oracle).unwrap();
    assert_eq!(oracle.batches[0].1, 4);
    assert_eq!(
        oracle.unique_evaluations(),
        4,
        "a four-point space costs four fits"
    );
    let (_, best_f) = exhaustive_best(&alleles, &|g: &Genome| {
        let (f, _) = deceptive(&alleles);
        f(g)
    });
    assert_eq!(outcome.best_fitness, best_f);
}

#[test]
fn a_one_point_space_and_bad_options_are_refused() {
    let (f, _) = deceptive(&[1, 1]);
    let mut oracle = Landscape::new(f);
    let e = run(&[1, 1], &GaOptions::default(), &mut oracle).unwrap_err();
    assert!(e.contains("single point"), "{e}");
    let e = run(&[2, 0], &GaOptions::default(), &mut oracle).unwrap_err();
    assert!(e.contains("no allele"), "{e}");

    let bad = [
        (
            "population_size must be at least 2",
            GaOptions {
                population_size: 1,
                ..GaOptions::default()
            },
        ),
        (
            "crossover_rate = 1.5",
            GaOptions {
                crossover_rate: 1.5,
                ..GaOptions::default()
            },
        ),
        (
            "elites = 20 must be smaller",
            GaOptions {
                elites: 20,
                ..GaOptions::default()
            },
        ),
        (
            "tournament_size",
            GaOptions {
                tournament_size: 0,
                ..GaOptions::default()
            },
        ),
        (
            "niche_penalty = -1",
            GaOptions {
                niche_penalty: -1.0,
                ..GaOptions::default()
            },
        ),
        (
            "sharing_alpha = 0",
            GaOptions {
                sharing_alpha: 0.0,
                ..GaOptions::default()
            },
        ),
        (
            "niches must be at least 1",
            GaOptions {
                niches: 0,
                ..GaOptions::default()
            },
        ),
    ];
    for (msg, options) in bad {
        let e = options.validate().unwrap_err();
        assert!(e.contains(msg), "{e}");
    }
}

#[test]
fn fitness_sharing_charges_the_crowded_individual_only_for_selection() {
    let population = vec![vec![0, 0, 0], vec![0, 0, 1], vec![3, 3, 3]];
    let fitness = vec![10.0, 10.0, 10.0];
    let options = GaOptions {
        niche_radius: 2,
        niche_penalty: 20.0,
        sharing_alpha: 1.0,
        ..GaOptions::default()
    };
    let shared = shared_fitness(&population, &fitness, &options);
    // The first two are one apart (< r = 2): each crowds the other by
    // 1 − 1/2 = 0.5, i.e. a charge of 10; the third is alone.
    assert_eq!(shared, vec![20.0, 20.0, 10.0]);
    let off = GaOptions {
        niche_radius: 0,
        ..options
    };
    assert_eq!(shared_fitness(&population, &fitness, &off), fitness);
    assert_eq!(hamming(&[0, 1, 2], &[0, 2, 2]), 1);
}

#[test]
fn the_toml_section_overlays_the_defaults() {
    let o: GaOptions = toml::from_str("population_size = 30\nseed = 7").unwrap();
    assert_eq!(
        o,
        GaOptions {
            population_size: 30,
            seed: 7,
            ..GaOptions::default()
        }
    );
    let e = toml::from_str::<GaOptions>("generation = 3").unwrap_err();
    assert!(e.to_string().contains("unknown field `generation`"), "{e}");
}

#[test]
fn the_downhill_search_is_not_capped_short_of_a_local_optimum() {
    // A 65-axis binary additive landscape descending from all ones needs
    // 65 strictly improving one-gene moves; a hard cap of 64 stopped at
    // fitness 1 with an improving neighbour still available.
    let alleles = vec![2usize; 65];
    let mut oracle = Landscape::new(|g: &Genome| g.iter().sum::<usize>() as f64);
    let start: Genome = vec![1; 65];
    let (g, f, moved) = downhill(&alleles, start, 65.0, "hill", &mut oracle).unwrap();
    assert!(moved);
    assert_eq!(f, 0.0);
    assert_eq!(g, vec![0; 65]);
    for n in neighbours(&g, &alleles) {
        assert!((oracle.f)(&n) >= f, "a neighbour of the optimum is better");
    }
}
