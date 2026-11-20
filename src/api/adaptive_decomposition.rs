//! Nested variance decomposition of the adaptive outcome metrics (#946 PR3, epic #391).
//!
//! For a metric `Y` of one population row (subject) `i`, the nested law of total variance
//! over the run's random inputs:
//!
//! ```text
//! Var(Y | i) = Var_Θ E[Y|Θ] + E_Θ Var_η E[Y|Θ,η] + E_{Θ,η} Var_W(Y|Θ,η)
//!              parameter      BSV                   within (κ and assay ε)
//! ```
//!
//! Interactions go to the inner layer. Each layer is estimated by the balanced nested
//! ANOVA over the run's `D` draws × `S` sims × `R` assay replicates, with a delete-one
//! jackknife standard error over the outermost independent level (draws, or sims when
//! `D = 1`). A negative estimate is reported as it is.
//!
//! A pure function over a finished [`AdaptiveSimulationResult`], like
//! [`super::summarize_adaptive`], and reading its values through the same producer.

use super::adaptive_summary::{run_values, RunKey};
use super::{AdaptiveSimulationResult, AdaptiveSummaryMetric};
use crate::diagnostics::EngineError;
use std::collections::{BTreeMap, BTreeSet};

/// The variance decomposition of an adaptive run: see [`decompose_adaptive`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveDecomposition {
    /// Parameter draws `D` (1 without parameter uncertainty).
    pub n_draws: usize,
    /// Sims per draw `S` (the run's `n_sim`).
    pub n_sims: usize,
    /// Assay replicates per `(draw, sim)` `R` (the run's `n_reps`).
    pub n_reps: usize,
    /// One row per `(metric, scope, layer)`, long form.
    pub components: Vec<AdaptiveVarianceComponent>,
    /// The `(metric, subject)` pairs left out because the metric is missing in some of the
    /// subject's runs, each with how many. Imputing them would unbalance the design.
    pub skipped: Vec<AdaptiveDecompositionSkip>,
}

/// One cell of [`AdaptiveDecomposition::components`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveVarianceComponent {
    /// The decomposed metric (the [`super::summarize_adaptive`] vocabulary).
    pub metric: AdaptiveSummaryMetric,
    /// One subject, or the population average.
    pub scope: AdaptiveVarianceScope,
    /// Which layer of the nested split.
    pub layer: AdaptiveVarianceLayer,
    /// The estimate, or why there is none.
    pub status: AdaptiveComponentStatus,
    /// Subjects the cell is computed over: 1 for a subject row; for a population row, the
    /// subjects whose metric is complete (and, for `Bsv`, whose η was drawn).
    pub n_subjects: usize,
}

/// The scope of an [`AdaptiveVarianceComponent`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum AdaptiveVarianceScope {
    /// The average of the subject components over the subjects.
    Population,
    /// One subject (population row), by id.
    Subject(String),
}

/// A layer of the nested split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum AdaptiveVarianceLayer {
    /// `Var_Θ E[Y|Θ]`: parameter uncertainty.
    Parameter,
    /// `E_Θ Var_η E[Y|Θ,η]`: between-subject variability (and the mixture class).
    Bsv,
    /// `E_{Θ,η} Var_W(Y|Θ,η)`: IOV κ and assay ε, redrawn per assay replicate.
    Within,
    /// `Var_i E[Y|i]` over the subjects (population scope only): covariates and design.
    BetweenRow,
    /// `Var(Y | i)` for a subject: the sum of its estimated layers. For the population,
    /// the pooled variance `mean_i Var(Y | i) + Var_i E[Y | i]`, which is the sum of the
    /// population lines when no subject's η was supplied (the population `Bsv` averages
    /// the drawn subjects only).
    Total,
}

/// What an [`AdaptiveVarianceComponent`] holds.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveComponentStatus {
    /// The estimate and its jackknife Monte Carlo standard error; `mc_se` is `None` when
    /// a delete-one replicate cannot estimate the layer (two draws for `Parameter`, two
    /// sims for `Bsv` without parameter uncertainty), and for the `Total` that includes it.
    Estimated {
        /// The estimate, as computed: a negative value is not truncated.
        value: f64,
        /// The delete-one jackknife standard error of `value`.
        mc_se: Option<f64>,
    },
    /// The subject's η was supplied (`AdaptiveInputs::with_etas`), so there is no BSV to
    /// estimate.
    Supplied,
    /// The run drew no parameter uncertainty, so there is no parameter layer.
    NotDrawn,
}

/// A `(metric, subject)` pair left out of the decomposition: see
/// [`AdaptiveDecomposition::skipped`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveDecompositionSkip {
    /// The metric left out for this subject.
    pub metric: AdaptiveSummaryMetric,
    /// The subject (population row), by id.
    pub subject: String,
    /// Runs of the subject that do not carry the metric, out of `D × S × R`.
    pub n_missing: usize,
}

/// One outer unit of a subject's `D × S × R` design, summarised once so a delete-one
/// replicate costs O(units), not O(D·S·R): a draw when `D > 1`, else a sim. `sse` is the
/// within-cell sum of squares, `sss` the between-sim sum of squares inside the unit
/// (`R·Σ_s (m_s − m)²`; 0 for a sim unit, which is one cell), `mean` the unit mean.
#[derive(Clone, Copy)]
struct Unit {
    sse: f64,
    sss: f64,
    mean: f64,
}

/// One subject's metric: its outer units, and whether its η was supplied.
struct Cube {
    units: Vec<Unit>,
    supplied: bool,
}

/// The outer units of `y`, stored `y[(d·S + s)·R + r]`.
fn units_of(y: &[f64], (d, s, r): (usize, usize, usize)) -> Vec<Unit> {
    let rf = r as f64;
    let cell = |di: usize, si: usize| &y[(di * s + si) * r..(di * s + si + 1) * r];
    let unit = |cells: Vec<&[f64]>| {
        let means: Vec<f64> = cells.iter().map(|c| c.iter().sum::<f64>() / rf).collect();
        let sse = cells
            .iter()
            .zip(&means)
            .map(|(c, m)| c.iter().map(|v| (v - m).powi(2)).sum::<f64>())
            .sum::<f64>();
        let mean = means.iter().sum::<f64>() / means.len() as f64;
        let sss = rf * means.iter().map(|m| (m - mean).powi(2)).sum::<f64>();
        Unit { sse, sss, mean }
    };
    if d > 1 {
        (0..d).map(|di| unit((0..s).map(|si| cell(di, si)).collect())).collect()
    } else {
        (0..s).map(|si| unit(vec![cell(0, si)])).collect()
    }
}

/// The estimated layers of one cube, or of the population. `None` where the kept design
/// cannot estimate the layer.
#[derive(Clone, Copy)]
struct Layers {
    param: Option<f64>,
    bsv: Option<f64>,
    within: Option<f64>,
    mean: f64,
}

/// The balanced nested ANOVA over `units`, leaving out unit `drop`:
/// `(MS_D − MS_S)/(S R)`, `(MS_S − MS_E)/R`, `MS_E`. With one draw the units are the sims
/// and there is no parameter layer.
fn layers(units: &[Unit], (d, s, r): (usize, usize, usize), drop: Option<usize>) -> Layers {
    let kept: Vec<Unit> =
        units.iter().enumerate().filter(|&(u, _)| drop != Some(u)).map(|(_, u)| *u).collect();
    let n = kept.len() as f64;
    let rf = r as f64;
    let mean = kept.iter().map(|u| u.mean).sum::<f64>() / n;
    let sse = kept.iter().map(|u| u.sse).sum::<f64>();
    let between_units = kept.iter().map(|u| (u.mean - mean).powi(2)).sum::<f64>();
    let ms = |ss: f64, df: f64| (df > 0.0).then(|| ss / df);
    let (sk, mse, mss, msd) = if d > 1 {
        let sk = s as f64;
        let sss = kept.iter().map(|u| u.sss).sum::<f64>();
        (
            sk,
            ms(sse, n * sk * (rf - 1.0)),
            ms(sss, n * (sk - 1.0)),
            ms(sk * rf * between_units, n - 1.0),
        )
    } else {
        (n, ms(sse, n * (rf - 1.0)), ms(rf * between_units, n - 1.0), None)
    };
    Layers {
        param: msd.zip(mss).map(|(a, b)| (a - b) / (sk * rf)),
        bsv: mss.zip(mse).map(|(a, b)| (a - b) / rf),
        within: mse,
        mean,
    }
}

/// The delete-one jackknife standard error of each of `full`'s estimates over `n` outer
/// units; `None` for an estimate some replicate cannot form.
fn jackknife(n: usize, est: impl Fn(Option<usize>) -> Vec<Option<f64>>) -> Vec<Option<f64>> {
    let reps: Vec<Vec<Option<f64>>> = (0..n).map(|u| est(Some(u))).collect();
    let k = est(None).len();
    (0..k)
        .map(|j| {
            let vals: Option<Vec<f64>> = reps.iter().map(|rep| rep[j]).collect();
            vals.map(|v| {
                let nf = v.len() as f64;
                let m = v.iter().sum::<f64>() / nf;
                ((nf - 1.0) / nf * v.iter().map(|x| (x - m).powi(2)).sum::<f64>()).sqrt()
            })
        })
        .collect()
}

/// The sum of the lines a scope reports, `(estimate, reported)`: `None` when a reported
/// line is missing (a delete-one replicate that cannot form it), so a jackknife replicate
/// of the total is never the sum of fewer lines than the total itself.
fn total_of(lines: &[(Option<f64>, bool)]) -> Option<f64> {
    lines.iter().filter(|(_, reported)| *reported).map(|(v, _)| *v).sum()
}

/// A subject's `Var(Y | i)`: the sum of its reported layers (BSV only when its η was
/// drawn, the parameter layer only under uncertainty). One definition for the subject's
/// `Total` and for the population's pooled one.
fn subject_total(l: &Layers, supplied: bool, d: usize) -> Option<f64> {
    total_of(&[(l.param, d > 1), (l.bsv, !supplied), (l.within, true)])
}

/// Which layers a run of this design identifies; `Err` names the one it cannot.
fn identifiability(d: usize, s: usize, r: usize, uncertain: bool) -> Result<(), String> {
    if r < 2 {
        return Err(format!(
            "n_reps = {r} leaves the within-subject layer confounded with BSV: run with \
             `AdaptiveSimulateOptions::n_reps` >= 2"
        ));
    }
    if s < 2 {
        return Err(format!(
            "n_sim = {s} leaves BSV confounded with the parameter layer: run with n_sim >= 2"
        ));
    }
    if uncertain && d < 2 {
        return Err("one draw cannot estimate the parameter layer: run with n_draws >= 2 in \
                    `UncertaintySpec`"
            .to_string());
    }
    Ok(())
}

/// The nested variance decomposition of `result`'s outcome metrics into parameter / BSV /
/// within-subject layers per subject, their averages over the population plus a
/// between-subject line, and a jackknife Monte Carlo standard error for each (#946).
///
/// The metrics are those of [`super::summarize_adaptive`]: the scalar metrics of each run
/// and each analyte's latent signal at each decision. A metric missing in any run of a
/// subject is skipped for that subject and counted in
/// [`AdaptiveDecomposition::skipped`], never imputed. Without parameter uncertainty the
/// parameter layer is [`AdaptiveComponentStatus::NotDrawn`]; a subject whose η was
/// supplied has a `Supplied` BSV, and the population BSV averages the drawn subjects only.
/// A non-finite value propagates into every estimate it enters.
///
/// # Errors
///
/// `E_ADAPTIVE_OPTIONS` when the design cannot separate a layer (`n_reps < 2`,
/// `n_sim < 2`, or one parameter draw), or when the rows are not the balanced
/// `D × S × R` set a run returns.
pub fn decompose_adaptive(
    result: &AdaptiveSimulationResult,
) -> Result<AdaptiveDecomposition, EngineError> {
    let refuse =
        |msg: String| crate::entry_refusals::adaptive_options(format!("decompose_adaptive: {msg}"));
    if result.metrics.is_empty() {
        return Err(refuse("the result has no metrics rows to decompose".to_string()));
    }
    let uncertain = !result.parameters.is_empty();
    let d = result.parameters.len().max(1);
    let s = result.metrics.iter().map(|m| m.sim).max().unwrap_or(0);
    let r = result.metrics.iter().map(|m| m.rep).max().unwrap_or(0);
    identifiability(d, s, r, uncertain).map_err(refuse)?;
    let dims = (d, s, r);
    let n_runs = d * s * r;
    let unbalanced = |detail: String| {
        refuse(format!(
            "the rows are not one balanced set of {d} draw(s) × {s} sim(s) × {r} rep(s) per \
             subject: {detail}"
        ))
    };
    let index = |k: &RunKey<'_>| ((k.draw - 1) * s + (k.sim - 1)) * r + (k.rep - 1);

    // Subjects in first-seen order, each with whether its η was supplied, and a balance
    // check: every subject has each (draw, sim, rep) exactly once.
    let mut subjects: Vec<(&str, bool)> = Vec::new();
    let mut runs_of: BTreeMap<&str, usize> = BTreeMap::new();
    let mut seen: BTreeSet<RunKey<'_>> = BTreeSet::new();
    for m in &result.metrics {
        let k = RunKey { subject: &m.subject, draw: m.draw, sim: m.sim, rep: m.rep };
        if k.draw == 0 || k.draw > d || k.sim == 0 || k.rep == 0 || !seen.insert(k) {
            return Err(unbalanced(format!(
                "subject '{}' (draw {}, sim {}, rep {}) is out of range or repeated",
                k.subject, k.draw, k.sim, k.rep
            )));
        }
        let runs = runs_of.entry(&m.subject).or_insert(0);
        if *runs == 0 {
            subjects.push((&m.subject, m.eta_supplied));
        }
        *runs += 1;
    }
    if let Some((id, runs)) =
        subjects.iter().map(|(id, _)| (id, runs_of[id])).find(|&(_, n)| n != n_runs)
    {
        return Err(unbalanced(format!("subject '{id}' has {runs} of {n_runs} runs")));
    }

    // Each (metric, subject)'s values in cube order; `None` marks a run without the metric.
    let mut values: BTreeMap<AdaptiveSummaryMetric, BTreeMap<&str, Vec<Option<f64>>>> =
        BTreeMap::new();
    for (k, metric, v) in run_values(result) {
        if !seen.contains(&k) {
            return Err(unbalanced(format!(
                "a decision of subject '{}' (draw {}, sim {}, rep {}) has no metrics row",
                k.subject, k.draw, k.sim, k.rep
            )));
        }
        let cube = values
            .entry(metric)
            .or_default()
            .entry(k.subject)
            .or_insert_with(|| vec![None; n_runs]);
        cube[index(&k)] = Some(v);
    }

    let mut out = AdaptiveDecomposition {
        n_draws: d,
        n_sims: s,
        n_reps: r,
        components: Vec::new(),
        skipped: Vec::new(),
    };
    for (metric, by_subject) in &values {
        let mut cubes: Vec<(&str, Cube)> = Vec::new();
        for &(id, supplied) in &subjects {
            let slots = by_subject.get(id).map_or(&[][..], Vec::as_slice);
            let present: Option<Vec<f64>> = slots.iter().copied().collect();
            match present.filter(|y| y.len() == n_runs) {
                Some(y) => cubes.push((id, Cube { units: units_of(&y, dims), supplied })),
                None => out.skipped.push(AdaptiveDecompositionSkip {
                    metric: metric.clone(),
                    subject: id.to_string(),
                    n_missing: n_runs - slots.iter().flatten().count(),
                }),
            }
        }
        push_metric(&mut out.components, metric, &cubes, dims);
    }
    Ok(out)
}

/// Every component row of one metric: each subject's layers, then the population's.
fn push_metric(
    rows: &mut Vec<AdaptiveVarianceComponent>, metric: &AdaptiveSummaryMetric,
    cubes: &[(&str, Cube)], dims: (usize, usize, usize),
) {
    let (d, s, _) = dims;
    let n_units = if d > 1 { d } else { s };
    let status = |value: Option<f64>, mc_se: Option<f64>| match value {
        Some(value) => AdaptiveComponentStatus::Estimated { value, mc_se },
        None => AdaptiveComponentStatus::NotDrawn,
    };
    let row = |scope: AdaptiveVarianceScope, layer, status, n_subjects| AdaptiveVarianceComponent {
        metric: metric.clone(),
        scope,
        layer,
        status,
        n_subjects,
    };
    use AdaptiveVarianceLayer::*;

    // A subject: parameter, BSV, within, and their sum.
    for (id, cube) in cubes {
        let est = |drop: Option<usize>| {
            let l = layers(&cube.units, dims, drop);
            // A supplied subject's BSV is computed but never reported: the `Supplied`
            // status below and `total_of`'s `reported` flag are the one gate.
            vec![l.param, l.bsv, l.within, subject_total(&l, cube.supplied, d)]
        };
        let (full, se) = (est(None), jackknife(n_units, est));
        let scope = || AdaptiveVarianceScope::Subject(id.to_string());
        rows.push(row(scope(), Parameter, status(full[0], se[0]), 1));
        let bsv =
            if cube.supplied { AdaptiveComponentStatus::Supplied } else { status(full[1], se[1]) };
        rows.push(row(scope(), Bsv, bsv, 1));
        rows.push(row(scope(), Within, status(full[2], se[2]), 1));
        rows.push(row(scope(), Total, status(full[3], se[3]), 1));
    }
    if cubes.is_empty() {
        return;
    }

    // The population: each layer averaged over the subjects (BSV over the drawn ones), the
    // between-subject variance of the subject means, and the pooled variance
    // `mean_i Var(Y | i) + Var_i E[Y | i]` — the sum of the lines when no subject is supplied.
    // Leaving out a draw leaves it out for every subject, which share it.
    let n_drawn = cubes.iter().filter(|(_, c)| !c.supplied).count();
    let est = |drop: Option<usize>| {
        let ls: Vec<(Layers, bool)> =
            cubes.iter().map(|(_, c)| (layers(&c.units, dims, drop), c.supplied)).collect();
        let avg = |f: &dyn Fn(&(Layers, bool)) -> Option<Option<f64>>| {
            let v: Vec<Option<f64>> = ls.iter().filter_map(f).collect();
            let v: Option<Vec<f64>> = v.into_iter().collect();
            v.filter(|v| !v.is_empty()).map(|v| v.iter().sum::<f64>() / v.len() as f64)
        };
        let param = avg(&|(l, _)| Some(l.param));
        let bsv = avg(&|(l, sup)| (!sup).then_some(l.bsv));
        let within = avg(&|(l, _)| Some(l.within));
        let n = ls.len() as f64;
        let grand = ls.iter().map(|(l, _)| l.mean).sum::<f64>() / n;
        let between = ls.iter().map(|(l, _)| (l.mean - grand).powi(2)).sum::<f64>() / n;
        let totals: Option<Vec<f64>> =
            ls.iter().map(|(l, sup)| subject_total(l, *sup, d)).collect();
        let total = totals.map(|t| t.iter().sum::<f64>() / n + between);
        vec![param, bsv, within, Some(between), total]
    };
    let (full, se) = (est(None), jackknife(n_units, est));
    let pop = || AdaptiveVarianceScope::Population;
    let n = cubes.len();
    rows.push(row(pop(), Parameter, status(full[0], se[0]), n));
    let bsv = if n_drawn == 0 { AdaptiveComponentStatus::Supplied } else { status(full[1], se[1]) };
    rows.push(row(pop(), Bsv, bsv, n_drawn));
    rows.push(row(pop(), Within, status(full[2], se[2]), n));
    rows.push(row(pop(), BetweenRow, status(full[3], se[3]), n));
    rows.push(row(pop(), Total, status(full[4], se[4]), n));
}
