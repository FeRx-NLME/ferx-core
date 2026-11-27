//! The controller effect as a contrast against a reference regimen (#946 PR4, epic #391).
//!
//! The controller is a deterministic map from the random inputs to doses, so it has no
//! variance layer of its own. Its effect is a contrast: the same model and population run
//! twice with the same seed, once under the controller and once under a reference regimen,
//! and each outcome summary is reported for both arms and as their difference
//! (adaptive − reference).
//!
//! **Common random numbers.** With one seed, parameters and decision grid, the two runs draw
//! the same parameter draws, η, mixture classes, κ and assay ε, whatever the controllers do:
//! η is drawn once per `(draw, sim, subject)` before the controller runs, and κ and ε are
//! keyed by the grid index ([`super::AdaptiveRunManifest`]). Both functions refuse a pair
//! that does not share them, and name what differs.
//!
//! **Paired.** The arms share their outer units (draws, or sims when `D = 1`), so the
//! difference's jackknife leaves the same unit out of both arms at once. Combining the arms'
//! own standard errors as `√(se_A² + se_R²)` ignores their correlation and overstates the
//! difference's.
//!
//! Pure functions over two finished results, like [`super::summarize_adaptive`] and
//! [`super::decompose_adaptive`]: they cannot change a trajectory, a ledger row or a decision.

use super::adaptive_decomposition::{
    population_estimates, push_rows, subject_estimates, Design, Unit,
};
use super::adaptive_inputs::first_value_difference;
use super::adaptive_summary::{
    check_summary_options, run_values, runs_per_draw, summarize_values, RunKey,
};
use super::{
    summarize_adaptive, AdaptiveDecomposition, AdaptiveDecompositionSkip,
    AdaptivePopulationSummary, AdaptiveSimulationResult, AdaptiveSummaryMetric,
    AdaptiveSummaryOptions,
};
use crate::diagnostics::EngineError;
use std::collections::{BTreeSet, HashMap};

/// What each arm of a contrast is, in the user's words: the reference is never implicit
/// (#946). For a reference built with
/// [`crate::sim::adaptive::AdaptiveDosingSpec::fixed_reference`], say so, e.g. "start dose
/// every due point".
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdaptiveArmLabels {
    /// The arm run under the controller.
    pub adaptive: String,
    /// The arm run under the reference regimen.
    pub reference: String,
}

impl AdaptiveArmLabels {
    /// Both labels. Neither may be empty.
    pub fn new(adaptive: impl Into<String>, reference: impl Into<String>) -> Self {
        Self { adaptive: adaptive.into(), reference: reference.into() }
    }
}

/// One outcome summary for both arms of a contrast and their difference: see
/// [`decompose_contrast`] and [`summarize_contrast`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveContrast<T> {
    /// What each arm is.
    pub labels: AdaptiveArmLabels,
    /// The adaptive arm's summary, exactly what the arm alone gives.
    pub adaptive: T,
    /// The reference arm's summary, exactly what the arm alone gives.
    pub reference: T,
    /// Adaptive − reference.
    pub difference: T,
}

/// The variance decomposition ([`super::decompose_adaptive`]) of both arms and of the
/// controller's effect on each layer (#946): `difference` holds adaptive − reference per
/// `(metric, scope, layer)`, with a **paired** jackknife standard error (the same outer
/// unit left out of both arms).
///
/// Each arm's decomposition is the one [`super::decompose_adaptive`] returns for it. The
/// difference is formed over the subjects whose metric is complete in both arms; a subject
/// missing it in either is listed in `difference.skipped`, with the runs missing in either
/// arm. A `Supplied` or `NotDrawn` status carries over.
///
/// The run does not record the model, the covariates, the base regimen in the data or the
/// supplied η values (`AdaptiveInputs::with_etas`), so the pair check cannot see them: pass
/// the same ones to both runs.
///
/// # Errors
///
/// `E_ADAPTIVE_OPTIONS` when a label is empty, the arms are not a common-random-numbers
/// pair (a seed missing or different, other parameters or parameter draws, another
/// decision grid, `n_sim`, `n_reps`, subjects or supplied η), or an arm's design cannot be
/// decomposed.
pub fn decompose_contrast(
    adaptive: &AdaptiveSimulationResult, reference: &AdaptiveSimulationResult,
    labels: &AdaptiveArmLabels,
) -> Result<AdaptiveContrast<AdaptiveDecomposition>, EngineError> {
    const WHO: &str = "decompose_contrast";
    check_pair(adaptive, reference, labels).map_err(|m| refusal(WHO, m))?;
    let a = Design::of(adaptive, "decompose_contrast: the adaptive arm")?;
    let r = Design::of(reference, "decompose_contrast: the reference arm")?;
    Ok(AdaptiveContrast {
        labels: labels.clone(),
        adaptive: a.decompose(),
        reference: r.decompose(),
        difference: difference_of(&a, &r),
    })
}

/// The population summary ([`super::summarize_adaptive`]) of both arms and of the
/// controller's effect (#946): `difference` is the summary of the per-run paired
/// differences adaptive − reference, so its `Mean` row is the mean effect per draw, its
/// band the parameter uncertainty of that effect, and its prediction interval the spread
/// of one patient's effect. A run whose metric is missing in either arm is missing in the
/// difference.
///
/// A threshold in `opts` applies to each summary's own values. On `difference` its
/// `FractionAtLeast(a)` row is therefore the fraction of runs whose effect is at least `a`,
/// P(A − R ≥ a); the change in attainment, P(A ≥ a) − P(R ≥ a), is the `adaptive` row minus
/// the `reference` row.
///
/// # Errors
///
/// `E_ADAPTIVE_OPTIONS` for `opts` [`super::summarize_adaptive`] refuses, an empty label,
/// or arms that are not a common-random-numbers pair (see [`decompose_contrast`]).
pub fn summarize_contrast(
    adaptive: &AdaptiveSimulationResult, reference: &AdaptiveSimulationResult,
    labels: &AdaptiveArmLabels, opts: &AdaptiveSummaryOptions,
) -> Result<AdaptiveContrast<AdaptivePopulationSummary>, EngineError> {
    const WHO: &str = "summarize_contrast";
    check_summary_options(opts, WHO)?;
    check_pair(adaptive, reference, labels).map_err(|m| refusal(WHO, m))?;
    let reference_values: HashMap<(RunKey<'_>, AdaptiveSummaryMetric), f64> =
        run_values(reference).map(|(run, metric, v)| ((run, metric), v)).collect();
    let paired = run_values(adaptive).filter_map(|(run, metric, v)| {
        let w = *reference_values.get(&(run, metric.clone()))?;
        Some((run.draw, metric, v - w))
    });
    Ok(AdaptiveContrast {
        labels: labels.clone(),
        adaptive: summarize_adaptive(adaptive, opts)?,
        reference: summarize_adaptive(reference, opts)?,
        difference: summarize_values(&runs_per_draw(adaptive), paired, opts),
    })
}

fn refusal(who: &str, message: String) -> EngineError {
    crate::entry_refusals::adaptive_options(format!("{who}: {message}"))
}

/// Adaptive − reference, per layer, over the subjects complete in both arms. Each
/// estimator is the difference of the two arms' estimators at the same left-out unit, so
/// [`push_rows`]' jackknife is the paired one.
fn difference_of(a: &Design<'_>, r: &Design<'_>) -> AdaptiveDecomposition {
    let dims = a.dims;
    let (n_draws, n_sims, n_reps) = dims;
    let mut out = AdaptiveDecomposition {
        n_draws,
        n_sims,
        n_reps,
        components: Vec::new(),
        skipped: Vec::new(),
    };
    let metrics: BTreeSet<&AdaptiveSummaryMetric> = a.metrics().chain(r.metrics()).collect();
    for metric in metrics {
        let mut kept: Vec<(&str, bool)> = Vec::new();
        let (mut cubes_a, mut cubes_r): (Vec<(Vec<Unit>, bool)>, Vec<(Vec<Unit>, bool)>) =
            (Vec::new(), Vec::new());
        for &(id, supplied) in &a.subjects {
            match (a.units(metric, id), r.units(metric, id)) {
                (Some(ua), Some(ur)) => {
                    kept.push((id, supplied));
                    cubes_a.push((ua, supplied));
                    cubes_r.push((ur, supplied));
                }
                _ => {
                    let (sa, sr) = (a.slots(metric, id), r.slots(metric, id));
                    let present =
                        |s: &[Option<f64>], k: usize| s.get(k).is_some_and(Option::is_some);
                    let n_missing =
                        (0..a.n_runs()).filter(|&k| !(present(sa, k) && present(sr, k))).count();
                    out.skipped.push(AdaptiveDecompositionSkip {
                        metric: metric.clone(),
                        subject: id.to_string(),
                        n_missing,
                    });
                }
            }
        }
        push_rows(
            &mut out.components,
            metric,
            dims,
            &kept,
            |i, drop| {
                minus(
                    subject_estimates(&cubes_a[i].0, cubes_a[i].1, dims, drop),
                    subject_estimates(&cubes_r[i].0, cubes_r[i].1, dims, drop),
                )
            },
            |drop| {
                minus(
                    population_estimates(&cubes_a, dims, drop),
                    population_estimates(&cubes_r, dims, drop),
                )
            },
        );
    }
    out
}

/// `a − b` per estimate; `None` where either side cannot form it.
fn minus(a: Vec<Option<f64>>, b: Vec<Option<f64>>) -> Vec<Option<f64>> {
    a.into_iter().zip(b).map(|(x, y)| x.zip(y).map(|(x, y)| x - y)).collect()
}

/// Subjects in population order (the order η is drawn in), each with whether its η was
/// supplied.
fn subjects_of(result: &AdaptiveSimulationResult) -> Vec<(&str, bool)> {
    let mut out: Vec<(&str, bool)> = Vec::new();
    for m in &result.metrics {
        if !out.iter().any(|(id, _)| *id == m.subject) {
            out.push((&m.subject, m.eta_supplied));
        }
    }
    out
}

/// Why `adaptive` and `reference` are not a common-random-numbers pair, if they are not:
/// the first input that fixes the random numbers and differs between them. One function,
/// so every refusal sentence lives here.
fn check_pair(
    a: &AdaptiveSimulationResult, r: &AdaptiveSimulationResult, labels: &AdaptiveArmLabels,
) -> Result<(), String> {
    if labels.adaptive.trim().is_empty() || labels.reference.trim().is_empty() {
        return Err(format!(
            "each arm needs a label that says what it is; got adaptive '{}' and reference '{}'",
            labels.adaptive, labels.reference
        ));
    }
    let (ma, mr) = (&a.manifest, &r.manifest);
    for (arm, seed) in [("adaptive", ma.seed), ("reference", mr.seed)] {
        if seed.is_none() {
            return Err(format!(
                "the {arm} arm was run without a seed, so its random numbers came from entropy \
                 and cannot be paired. Set `AdaptiveSimulateOptions::seed` to the same value \
                 in both arms"
            ));
        }
    }
    if ma.seed != mr.seed {
        return Err(format!(
            "the arms were run with different seeds (adaptive {}, reference {}); common \
             random numbers need one seed",
            ma.seed.unwrap_or_default(),
            mr.seed.unwrap_or_default()
        ));
    }
    let sides = ("adaptive arm", "reference arm");
    if let Some(what) = first_value_difference(&ma.point_parameters, &mr.point_parameters, sides) {
        return Err(format!("the arms were run at different parameters: {what}"));
    }
    if a.parameters.len() != r.parameters.len() {
        return Err(format!(
            "the adaptive arm has {} parameter draw(s) and the reference arm {}",
            a.parameters.len(),
            r.parameters.len()
        ));
    }
    for (pa, pr) in a.parameters.iter().zip(&r.parameters) {
        if let Some(what) = first_value_difference(&pa.parameters, &pr.parameters, sides) {
            return Err(format!(
                "parameter draw {} differs: {what}. Run both arms with the same \
                 `with_uncertainty` fit and spec",
                pa.draw
            ));
        }
    }
    let bits = |t: &[f64]| t.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&ma.decision_times) != bits(&mr.decision_times) {
        return Err(format!(
            "the decision grids differ (adaptive {:?}, reference {:?}); IOV occasions and \
             assay draws are keyed by the grid index, so both arms need one grid",
            ma.decision_times, mr.decision_times
        ));
    }
    let largest = |res: &AdaptiveSimulationResult, rep: bool| {
        res.metrics.iter().map(|m| if rep { m.rep } else { m.sim }).max().unwrap_or(0)
    };
    for (name, rep) in [("n_sim", false), ("n_reps", true)] {
        let (x, y) = (largest(a, rep), largest(r, rep));
        if x != y {
            return Err(format!("the adaptive arm has {name} = {x} and the reference arm {y}"));
        }
    }
    let (sa, sr) = (subjects_of(a), subjects_of(r));
    if sa.len() != sr.len() {
        return Err(format!(
            "the adaptive arm has {} subject(s) and the reference arm {}",
            sa.len(),
            sr.len()
        ));
    }
    for (k, ((ida, supa), (idr, supr))) in sa.iter().zip(&sr).enumerate() {
        if ida != idr {
            return Err(format!(
                "subject {} is '{ida}' in the adaptive arm and '{idr}' in the reference arm; η \
                 is drawn in population order, so both arms need the same population",
                k + 1
            ));
        }
        if supa != supr {
            let arm = if *supa { "adaptive" } else { "reference" };
            return Err(format!("subject '{ida}' has a supplied η in the {arm} arm only"));
        }
    }
    Ok(())
}
