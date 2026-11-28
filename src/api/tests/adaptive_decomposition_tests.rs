//! #946 PR3: `decompose_adaptive`, the nested variance decomposition into parameter / BSV /
//! within-subject layers, and the `n_reps` assay replicates it needs.
//!
//! The adaptive exception (AGENTS.md, epic #391): no NONMEM run. The kernel is pinned on
//! hand-computed arrays (exact binary arithmetic); the engine on exact zeros, structural
//! rep-keying tests, and an analytic conditional oracle computed by quadrature here.

use super::adaptive_sim_tests::{fixed_bolus, population, subj};
use super::*;
use crate::estimation::uncertainty_samples::{fitted_params_from_result, UncertaintyMethod};
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DecisionOutcome, DoseAction,
    MonitorSpec, ObserveMode, ObservedSignal,
};
use AdaptiveComponentStatus as St;
use AdaptiveSummaryMetric as M;
use AdaptiveVarianceLayer as L;
use AdaptiveVarianceScope as Sc;

// ── hand-built rows ────────────────────────────────────────────────────────────────────

pub(super) fn metrics_row(
    id: &str, (draw, sim, rep): (usize, usize, usize), y: f64,
) -> AdaptiveSubjectMetrics {
    AdaptiveSubjectMetrics {
        subject: id.to_string(),
        draw,
        sim,
        rep,
        cumulative_dose: 100.0,
        n_doses: 1,
        n_increases: 0,
        n_decreases: 0,
        n_holds: 0,
        n_not_due: 0,
        discontinued: false,
        time_to_discontinuation: None,
        signal_min: None,
        signal_max: None,
        signal_mean: Some(y),
        pct_readings_in_window: None,
        auc_target_attainment: None,
        eta_supplied: false,
        dose_oracle: false,
    }
}

/// The balanced rows of `cubes` (`(id, y[d][s][r])`), with `n_draws` parameter sets when
/// `uncertain` (their values are not read by the kernel).
pub(super) fn result_of(
    cubes: &[(&str, &[[[f64; 2]; 2]])], uncertain: bool,
) -> AdaptiveSimulationResult {
    let mut metrics = Vec::new();
    for (id, y) in cubes {
        for (d, sims) in y.iter().enumerate() {
            for (s, reps) in sims.iter().enumerate() {
                for (r, &v) in reps.iter().enumerate() {
                    metrics.push(metrics_row(id, (d + 1, s + 1, r + 1), v));
                }
            }
        }
    }
    let n_draws = cubes[0].1.len();
    let template = model("omega ETA_CL ~ 0.09", "").default_params;
    let parameters = if uncertain {
        (1..=n_draws).map(|draw| DrawnParameters { draw, parameters: template.clone() }).collect()
    } else {
        vec![]
    };
    AdaptiveSimulationResult {
        trajectories: vec![],
        ledger: vec![],
        decisions: vec![],
        metrics,
        parameters,
        manifest: AdaptiveRunManifest {
            seed: Some(946),
            point_parameters: template,
            decision_times: DECISIONS.to_vec(),
        },
        warnings: vec![],
    }
}

fn decision(id: &str, (draw, sim, rep): (usize, usize, usize)) -> DecisionLogEntry {
    DecisionLogEntry {
        subject: id.to_string(),
        draw,
        sim,
        rep,
        decision_idx: 0,
        time: 0.0,
        observed_signals: vec![ObservedSignal {
            name: "A".into(),
            value: 1.0,
            latent: 1.0,
            mode: ObserveMode::Ipred,
        }],
        outcome: DecisionOutcome::Hold,
        interval: None,
    }
}

fn cell<'a>(
    d: &'a AdaptiveDecomposition, metric: &M, scope: &Sc, layer: L,
) -> &'a AdaptiveVarianceComponent {
    d.components
        .iter()
        .find(|c| &c.metric == metric && &c.scope == scope && c.layer == layer)
        .unwrap_or_else(|| panic!("no {metric:?} {scope:?} {layer:?}"))
}

pub(super) fn est(d: &AdaptiveDecomposition, scope: &Sc, layer: L) -> (f64, Option<f64>) {
    match cell(d, &M::SignalMean, scope, layer).status {
        St::Estimated { value, mc_se } => (value, mc_se),
        other => panic!("{scope:?} {layer:?} is {other:?}"),
    }
}

pub(super) fn subject(id: &str) -> Sc {
    Sc::Subject(id.into())
}

/// draw 1: sim 1 [1, 3], sim 2 [4, 8]; draw 2: sim 1 [2, 2], sim 2 [9, 5].
pub(super) const CUBE: [[[f64; 2]; 2]; 2] = [[[1.0, 3.0], [4.0, 8.0]], [[2.0, 2.0], [9.0, 5.0]]];

// ── the kernel on hand-computed arrays ─────────────────────────────────────────────────

/// The balanced nested ANOVA on `CUBE`, by hand (every value exact in binary):
/// cell means 2, 6, 2, 7; draw means 4, 4.5; grand 4.25.
/// SSE = 2 + 8 + 0 + 8 = 18 on 4 df → MS_E = 4.5.
/// SSS = 2·(4 + 4 + 6.25 + 6.25) = 41 on 2 df → MS_S = 20.5.
/// SSD = 4·(0.0625 + 0.0625) = 0.5 on 1 df → MS_D = 0.5.
/// param = (0.5 − 20.5)/4 = −5 (negative, reported as it is), BSV = (20.5 − 4.5)/2 = 8,
/// within = 4.5, total 7.5.
/// Jackknife over the two draws: without draw 1, MS_E 4, MS_S 25, BSV 10.5; without draw 2,
/// MS_E 5, MS_S 16, BSV 5.5 → SE(BSV) = √(½·(2.5² + 2.5²)) = 2.5, SE(within) = 0.5. One draw
/// left cannot form the parameter layer, so its SE (and the total's) is `None`.
///
/// Mutations: swap two layers; drop R from the BSV divisor (BSV 16); a denominator off by
/// one df; truncating a negative estimate to 0 (param 0); the jackknife's (n−1)/n factor.
#[test]
fn the_nested_anova_matches_a_hand_computed_cube() {
    let d = decompose_adaptive(&result_of(&[("1", &CUBE)], true)).expect("decomposes");
    assert_eq!((d.n_draws, d.n_sims, d.n_reps), (2, 2, 2));
    let s = subject("1");
    assert_eq!(est(&d, &s, L::Parameter), (-5.0, None));
    assert_eq!(est(&d, &s, L::Bsv), (8.0, Some(2.5)));
    assert_eq!(est(&d, &s, L::Within), (4.5, Some(0.5)));
    assert_eq!(est(&d, &s, L::Total), (7.5, None));
    // One subject: the population is that subject, and the between-row line is 0.
    let p = Sc::Population;
    assert_eq!(est(&d, &p, L::Parameter).0, -5.0);
    // The population jackknife drops each draw for the subjects that share it: here the
    // one subject's SE (review r1, mutation 3).
    assert_eq!(est(&d, &p, L::Bsv), (8.0, Some(2.5)));
    assert_eq!(est(&d, &p, L::BetweenRow), (0.0, Some(0.0)));
    assert_eq!(est(&d, &p, L::Total), (7.5, None));
}

/// Without uncertainty (D = 1): no parameter layer, the jackknife runs over sims. Subject A
/// = `CUBE[0]`, B = `CUBE[1]`, each a 1 × 2 × 2 cube.
/// A: MS_E 5, MS_S 16 → BSV 5.5, within 5, mean 4. B: MS_E 4, MS_S 25 → BSV 10.5, within
/// 4, mean 4.5. Population: BSV 8, within 4.5, between ((−¼)² + (¼)²)/2 = 0.0625, total
/// 12.5625: the between-row line enters the total once.
/// Two sims leave one per jackknife replicate, so BSV has no SE; within's is over sims:
/// A without sim 1 → 8, without sim 2 → 2, so SE = √(½·(3² + 3²)) = 3.
///
/// Mutations: the parameter layer reported as 0 instead of `NotDrawn`; the between-row
/// line added to the total twice (12.625), or left out (12.5).
#[test]
fn without_uncertainty_the_parameter_layer_is_not_drawn_and_the_lines_sum_to_the_total() {
    let a: [[[f64; 2]; 2]; 1] = [CUBE[0]];
    let b: [[[f64; 2]; 2]; 1] = [CUBE[1]];
    let d = decompose_adaptive(&result_of(&[("A", &a), ("B", &b)], false)).expect("decomposes");
    assert_eq!(d.n_draws, 1);
    for scope in [subject("A"), subject("B"), Sc::Population] {
        assert_eq!(cell(&d, &M::SignalMean, &scope, L::Parameter).status, St::NotDrawn);
    }
    assert_eq!(est(&d, &subject("A"), L::Bsv), (5.5, None));
    assert_eq!(est(&d, &subject("A"), L::Within), (5.0, Some(3.0)));
    assert_eq!(est(&d, &subject("B"), L::Bsv), (10.5, None));
    let p = Sc::Population;
    assert_eq!(est(&d, &p, L::Bsv).0, 8.0);
    assert_eq!(est(&d, &p, L::Within).0, 4.5);
    assert_eq!(est(&d, &p, L::BetweenRow).0, 0.0625);
    assert_eq!(est(&d, &p, L::Total).0, 12.5625);
    assert_eq!(cell(&d, &M::SignalMean, &p, L::Total).n_subjects, 2);
}

/// A supplied-η subject has a `Supplied` BSV (never 0, never estimated), and the
/// population BSV averages the drawn subjects only (A's 5.5), while within and the
/// between-row line still pool both. The population `Total` is the pooled variance
/// `mean_i Var(Y | i) + Var_i E[Y | i]` = (A 10.5 + B 4)/2 + 0.0625 = 7.3125, not the sum
/// of the lines (5.5 + 4.5 + 0.0625 = 10.0625, which counts A's BSV for B too; review r1
/// row 3).
///
/// Mutations: the supplied subject's BSV estimated (10.5) or entered in the population
/// average (8); `n_subjects` of the population BSV counting it; the population `Total` as
/// the sum of the lines (10.0625).
#[test]
fn a_supplied_subject_has_no_bsv_and_the_population_bsv_pools_the_drawn_ones() {
    let a: [[[f64; 2]; 2]; 1] = [CUBE[0]];
    let b: [[[f64; 2]; 2]; 1] = [CUBE[1]];
    let mut r = result_of(&[("A", &a), ("B", &b)], false);
    r.metrics.iter_mut().filter(|m| m.subject == "B").for_each(|m| m.eta_supplied = true);
    let d = decompose_adaptive(&r).expect("decomposes");
    assert_eq!(cell(&d, &M::SignalMean, &subject("B"), L::Bsv).status, St::Supplied);
    assert_eq!(est(&d, &subject("B"), L::Total).0, 4.0, "B's total is its within alone");
    let p = Sc::Population;
    assert_eq!(est(&d, &p, L::Bsv).0, 5.5);
    assert_eq!(cell(&d, &M::SignalMean, &p, L::Bsv).n_subjects, 1);
    assert_eq!(est(&d, &p, L::Within).0, 4.5);
    assert_eq!(est(&d, &p, L::Total).0, 7.3125);

    // Every subject supplied: the population BSV is `Supplied` too, not an empty average.
    r.metrics.iter_mut().for_each(|m| m.eta_supplied = true);
    let d = decompose_adaptive(&r).expect("decomposes");
    assert_eq!(cell(&d, &M::SignalMean, &p, L::Bsv).status, St::Supplied);
}

/// A metric missing in one run of a subject is skipped for that subject and counted, never
/// imputed; the other subject is decomposed, and the population is that subject alone.
///
/// Mutation: imputing `None` as 0 (subject B decomposed; the population BSV changes).
#[test]
fn a_metric_missing_in_one_run_is_skipped_for_that_subject_and_counted() {
    let a: [[[f64; 2]; 2]; 1] = [CUBE[0]];
    let b: [[[f64; 2]; 2]; 1] = [CUBE[1]];
    let mut r = result_of(&[("A", &a), ("B", &b)], false);
    let hole = r.metrics.iter_mut().find(|m| m.subject == "B" && m.rep == 2).expect("row");
    hole.signal_mean = None;
    let d = decompose_adaptive(&r).expect("decomposes");
    let skip = d.skipped.iter().find(|k| k.metric == M::SignalMean).expect("skipped");
    assert_eq!((skip.subject.as_str(), skip.n_missing), ("B", 1));
    assert!(!d.components.iter().any(|c| c.metric == M::SignalMean && c.scope == subject("B")));
    assert_eq!(est(&d, &Sc::Population, L::Bsv).0, 5.5);
    assert_eq!(cell(&d, &M::SignalMean, &Sc::Population, L::Bsv).n_subjects, 1);
    // A metric every run carries is not skipped.
    assert!(!d.skipped.iter().any(|k| k.metric == M::CumulativeDose));
}

/// A non-finite value propagates into every estimate it enters: a sum carries it, and no
/// fold drops it.
///
/// Mutation: a NaN-dropping fold (`f64::max`-style, or filtering non-finite values).
#[test]
fn a_non_finite_value_propagates_into_every_estimate() {
    let mut y = CUBE;
    y[1][0][1] = f64::NAN;
    let d = decompose_adaptive(&result_of(&[("1", &y)], true)).expect("decomposes");
    for layer in [L::Parameter, L::Bsv, L::Within, L::Total] {
        assert!(est(&d, &subject("1"), layer).0.is_nan(), "{layer:?}");
    }
    // The other metrics of the same runs are unaffected.
    let dose = cell(&d, &M::CumulativeDose, &subject("1"), L::Within);
    assert_eq!(dose.status, St::Estimated { value: 0.0, mc_se: Some(0.0) });
}

// ── refusals ───────────────────────────────────────────────────────────────────────────

fn refusal(r: &AdaptiveSimulationResult) -> String {
    let e = decompose_adaptive(r).expect_err("refused");
    let code = e.diagnostic().map(|d| d.code.clone()).unwrap_or_default();
    assert_eq!(code, "E_ADAPTIVE_OPTIONS", "{e}");
    e.to_string()
}

/// Input space: R ∈ {1, ≥2} × S ∈ {1, ≥2} × uncertainty ∈ {off, on with D = 1, on with
/// D ≥ 2} × rows ∈ {balanced, a run repeated or out of range, a run missing, a decision
/// without a metrics row}. Each refusal names the layer it cannot identify, and is checked
/// whole, so deleting any sentence reddens it. R is checked first, then S, then D.
#[test]
fn designs_that_cannot_separate_a_layer_are_refused_naming_the_layer() {
    let one_rep: Vec<_> = (1..=2).map(|s| metrics_row("1", (1, s, 1), s as f64)).collect();
    let r = AdaptiveSimulationResult { metrics: one_rep, ..result_of(&[("1", &[CUBE[0]])], false) };
    assert!(refusal(&r).ends_with(
        "decompose_adaptive: n_reps = 1 leaves the within-subject layer confounded with BSV: \
         run with `AdaptiveSimulateOptions::n_reps` >= 2"
    ));

    let one_sim: Vec<_> = (1..=2).map(|k| metrics_row("1", (1, 1, k), k as f64)).collect();
    let r = AdaptiveSimulationResult { metrics: one_sim, ..result_of(&[("1", &[CUBE[0]])], false) };
    assert!(refusal(&r).ends_with(
        "decompose_adaptive: n_sim = 1 leaves BSV confounded with the parameter layer: run \
         with n_sim >= 2"
    ));

    let one_draw = result_of(&[("1", &[CUBE[0]])], true);
    assert!(refusal(&one_draw).ends_with(
        "decompose_adaptive: one draw cannot estimate the parameter layer: run with n_draws \
         >= 2 in `UncertaintySpec`"
    ));
    // The straddle: the same one-draw rows without uncertainty decompose.
    assert!(decompose_adaptive(&result_of(&[("1", &[CUBE[0]])], false)).is_ok());
}

/// A result with no metrics rows (a caller's filter can leave none) is refused as such,
/// not as an `n_reps = 0` design (review r1 row 7).
#[test]
fn a_result_without_runs_is_refused_as_empty() {
    let mut r = result_of(&[("1", &CUBE)], true);
    r.metrics.clear();
    assert!(
        refusal(&r).ends_with("decompose_adaptive: the result has no metrics rows to decompose")
    );
}

/// Rows that are not one balanced D × S × R set per subject are refused, never decomposed
/// on a ragged design. The message names the set and the first offending subject.
#[test]
fn rows_that_are_not_a_balanced_set_are_refused() {
    let head = "decompose_adaptive: the rows are not one balanced set of 2 draw(s) × 2 sim(s) \
                × 2 rep(s) per subject: ";
    let mut r = result_of(&[("1", &CUBE)], true);
    r.metrics.push(r.metrics[0].clone());
    assert!(refusal(&r).ends_with(&format!(
        "{head}subject '1' (draw 1, sim 1, rep 1) is out of range or repeated"
    )));

    let mut r = result_of(&[("1", &CUBE)], true);
    r.metrics.remove(3);
    assert!(refusal(&r).ends_with(&format!("{head}subject '1' has 7 of 8 runs")));

    let mut r = result_of(&[("1", &CUBE)], true);
    r.decisions.push(decision("1", (3, 1, 1)));
    assert!(refusal(&r).ends_with(&format!(
        "{head}a decision of subject '1' (draw 3, sim 1, rep 1) has no metrics row"
    )));
}

// ── engine fixtures ────────────────────────────────────────────────────────────────────

/// A 1-cpt IV ODE, CL = TVCL·exp(η [+ κ]), V fixed; the cmt-1 monitor reads the central
/// **amount**. `kappa` is an extra `[parameters]` line (an IOV κ on CL) or empty.
pub(super) fn model(omega: &str, kappa: &str) -> CompiledModel {
    let cl = if kappa.is_empty() { "TVCL * exp(ETA_CL)" } else { "TVCL * exp(ETA_CL + KAPPA_CL)" };
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  {omega}
  {kappa}
  sigma PROP ~ 0.04
[individual_parameters]
  CL = {cl}
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#
    ))
    .expect("parse")
}

pub(super) const DECISIONS: [f64; 3] = [0.0, 24.0, 48.0];

pub(super) fn pop(n: usize) -> Population {
    population((1..=n).map(|i| subj(&i.to_string(), vec![6.0, 30.0, 54.0], vec![])).collect())
}

pub(super) fn opts(mode: ObserveMode, n_reps: usize) -> AdaptiveSimulateOptions {
    AdaptiveSimulateOptions {
        seed: Some(946),
        decision_times: DECISIONS.to_vec(),
        monitors: vec![MonitorSpec::new("A", 1, mode)],
        n_reps,
        ..Default::default()
    }
}

/// A controller that reads the (noised, under `Dv`) signal: the assay noise moves doses.
fn reading_controller() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        let amt = if ctx.signal("A").unwrap_or(0.0) < 9.0 { 100.0 } else { 50.0 };
        vec![DoseAction::Bolus { amt, cmt: 1 }]
    }
}

fn run<F, C>(
    m: &CompiledModel, n_sub: usize, n_sim: usize, o: &AdaptiveSimulateOptions, make: F,
) -> AdaptiveSimulationResult
where
    F: Fn() -> C,
    C: FnMut(&ControllerCtx) -> Vec<DoseAction>,
{
    simulate_adaptive(m, &pop(n_sub), &m.default_params, n_sim, make, o).expect("runs")
}

fn rep_rows<T: Clone>(rows: &[T], rep_of: impl Fn(&T) -> usize, rep: usize) -> Vec<T> {
    rows.iter().filter(|r| rep_of(r) == rep).cloned().collect()
}

// ── the replicates in the run ──────────────────────────────────────────────────────────

/// R = 1 is the run without replicates, and rep 1 of an R = 3 run is that run bit for bit:
/// trajectories, ledger, decisions and metrics, on IOV κ + a `Dv` monitor read by the
/// controller, so both keyed substreams reach the doses. The default-on frozen-replay and
/// snapshot verifiers ran on every rep (they would `Err` the run).
/// The straddle: reps 2 and 3 differ from rep 1 (κ and ε are redrawn).
///
/// Mutations: κ / assay keyed by sim only (reps identical: the straddle dies); the rep
/// loop changing the rep-1 key or the η stream (rep 1 differs from the R = 1 run).
#[test]
fn rep_one_of_a_replicated_run_is_the_run_without_replicates() {
    let m = model("omega ETA_CL ~ 0.09", "kappa KAPPA_CL ~ 0.04");
    let base = run(&m, 3, 2, &opts(ObserveMode::Dv, 1), reading_controller);
    let reps = run(&m, 3, 2, &opts(ObserveMode::Dv, 3), reading_controller);
    assert_eq!(format!("{:?}", base.trajectories), format!("{:?}", reps.trajectories));
    assert_eq!(base.ledger, rep_rows(&reps.ledger, |e| e.rep, 1));
    assert_eq!(base.decisions, rep_rows(&reps.decisions, |d| d.rep, 1));
    assert_eq!(base.metrics, rep_rows(&reps.metrics, |r| r.rep, 1));
    assert!(base.metrics.iter().all(|r| r.rep == 1));
    assert_eq!(reps.metrics.len(), 3 * base.metrics.len());
    for k in 2..=3 {
        let differs =
            base.decisions.iter().zip(rep_rows(&reps.decisions, |d| d.rep, k)).any(|(a, b)| {
                a.observed_signals[0].value.to_bits() != b.observed_signals[0].value.to_bits()
            });
        assert!(differs, "rep {k} redrew no κ or assay noise");
    }
}

/// The `Dv` assay noise is redrawn per rep on its own, without an IOV κ to carry the
/// difference (review r1 row 1): the controller reads the noised value, so rep 2 doses
/// differently from rep 1 and the within layer is positive. The straddle is the
/// `Ipred` case of the test below, where the same model has within = 0 exactly.
///
/// Mutation: the assay key without the rep (κ keeping it) — reps identical, within 0.
#[test]
fn the_assay_noise_is_redrawn_per_rep_without_iov() {
    let m = model("omega ETA_CL ~ 0.09", "");
    let r = run(&m, 2, 3, &opts(ObserveMode::Dv, 3), reading_controller);
    let rep1 = rep_rows(&r.decisions, |d| d.rep, 1);
    for k in 2..=3 {
        let differs = rep1.iter().zip(rep_rows(&r.decisions, |d| d.rep, k)).any(|(a, b)| {
            a.observed_signals[0].value.to_bits() != b.observed_signals[0].value.to_bits()
        });
        assert!(differs, "rep {k} redrew no assay noise");
    }
    let d = decompose_adaptive(&r).expect("decomposes");
    let (within, _) = est(&d, &Sc::Population, L::Within);
    assert!(within > 0.0, "{within}");
}

/// η (and the class) stay fixed across reps: without IOV and with an `Ipred` monitor there
/// is no within-subject randomness, so every rep's metrics equal rep 1's bit for bit, and
/// the within layer is exactly 0 (MS_E = 0, no tolerance). The straddle: the same run with
/// an IOV κ has a positive within layer.
///
/// Mutations: η redrawn per rep (reps differ; within > 0); κ keyed by sim only (the IOV
/// within layer is exactly 0).
#[test]
fn within_is_exactly_zero_without_within_randomness_and_live_with_iov() {
    let o = opts(ObserveMode::Ipred, 2);
    let no_iov = run(&model("omega ETA_CL ~ 0.09", ""), 2, 3, &o, fixed_bolus);
    let (r1, r2) =
        (rep_rows(&no_iov.metrics, |r| r.rep, 1), rep_rows(&no_iov.metrics, |r| r.rep, 2));
    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(&r2) {
        assert_eq!((&a.subject, a.draw, a.sim), (&b.subject, b.draw, b.sim));
        assert_eq!(a.signal_mean.map(f64::to_bits), b.signal_mean.map(f64::to_bits));
    }
    let d = decompose_adaptive(&no_iov).expect("decomposes");
    for scope in [subject("1"), subject("2"), Sc::Population] {
        assert_eq!(est(&d, &scope, L::Within).0, 0.0, "{scope:?}");
    }
    let iov = run(&model("omega ETA_CL ~ 0.09", "kappa KAPPA_CL ~ 0.04"), 2, 3, &o, fixed_bolus);
    let d = decompose_adaptive(&iov).expect("decomposes");
    assert!(est(&d, &Sc::Population, L::Within).0 > 0.0);
}

/// With Ω = 0 and no within-subject randomness every run of a subject is the same, so BSV
/// is exactly 0 (S = 2, R = 2: every mean is of two equal values, exact). The straddle:
/// Ω > 0 gives a positive BSV.
///
/// Mutation: BSV read from the between-draw or within mean square.
#[test]
fn bsv_is_exactly_zero_with_zero_omega_and_live_with_omega() {
    let o = opts(ObserveMode::Ipred, 2);
    let zero = run(&model("omega ETA_CL ~ 0.0 FIX", ""), 2, 2, &o, fixed_bolus);
    let d = decompose_adaptive(&zero).expect("decomposes");
    for scope in [subject("1"), Sc::Population] {
        assert_eq!(est(&d, &scope, L::Bsv).0, 0.0, "{scope:?}");
        assert_eq!(est(&d, &scope, L::Within).0, 0.0, "{scope:?}");
    }
    let live = run(&model("omega ETA_CL ~ 0.09", ""), 2, 2, &o, fixed_bolus);
    let d = decompose_adaptive(&live).expect("decomposes");
    assert!(est(&d, &Sc::Population, L::Bsv).0 > 0.0);
}

/// `n_reps = 0` is refused before any subject runs, whole text.
#[test]
fn zero_reps_are_refused() {
    let m = model("omega ETA_CL ~ 0.09", "");
    let o = opts(ObserveMode::Ipred, 0);
    let e =
        simulate_adaptive(&m, &pop(1), &m.default_params, 1, fixed_bolus, &o).expect_err("refused");
    assert_eq!(e.diagnostic().map(|d| d.code.clone()).unwrap_or_default(), "E_ADAPTIVE_OPTIONS");
    let want = "n_reps must be >= 1 (the number of assay replicates per subject run); 1 is a \
                run without replicates";
    assert!(e.to_string().contains(want), "{e}");
}

// ── the analytic conditional oracle ────────────────────────────────────────────────────

/// Probabilists' Gauss–Hermite nodes and normalised weights (Golub–Welsch).
pub(super) fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = nalgebra::DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        j[(k - 1, k)] = (k as f64).sqrt();
        j[(k, k - 1)] = (k as f64).sqrt();
    }
    let e = j.symmetric_eigen();
    let w: Vec<f64> = (0..n).map(|k| e.eigenvectors[(0, k)].powi(2)).collect();
    let total: f64 = w.iter().sum();
    (e.eigenvalues.iter().copied().collect(), w.iter().map(|v| v / total).collect())
}

/// `E[Y | θ]`, `Var_η E[Y | θ, η]` and `E_η Var_κ(Y | θ, η)` of `Y = signal_mean` on the
/// fixture, by quadrature outside the engine: three pre-dose amount readings 0, `A₁`, `A₂`.
/// The interval ending at a record runs under that record's η (#2895), so the observation
/// at t = 6 (occasion 0) closes `[0, 6]` under κ₀ and the decision at 24 closes `[6, 24]`
/// under κ₁: `A₁ = 100·e^{−(6 r₀ + 18 r₁)}`, `A₂ = (A₁ + 100)·e^{−(6 r₁ + 18 r₂)}`,
/// `r_g = TVCL·e^{η + κ_g}/V`, η ~ N(0, ω²), κ₀..κ₂ ~ N(0, ω²_IOV) independent. (Measured:
/// the κ implied by the decision-1 reading has variance 0.0258, against
/// (6² + 18²)/24² · 0.04 = 0.025 for this split and 0.04 for a single κ per interval.)
fn conditional_moments(tvcl: f64, v: f64, omega2: f64, iov2: f64) -> [f64; 3] {
    let (xe, we) = gauss_hermite(40);
    let (xk, wk) = gauss_hermite(12);
    let sk = iov2.sqrt();
    let y = |l: f64, k: [f64; 3]| {
        let r = |kg: f64| (l + kg).exp() / v;
        let a1 = 100.0 * (-(6.0 * r(k[0]) + 18.0 * r(k[1]))).exp();
        let a2 = (a1 + 100.0) * (-(6.0 * r(k[1]) + 18.0 * r(k[2]))).exp();
        (a1 + a2) / 3.0
    };
    let (mut m, mut m2, mut w) = (0.0, 0.0, 0.0);
    for (&xa, &wa) in xe.iter().zip(&we) {
        let l = tvcl.ln() + omega2.sqrt() * xa;
        let (mut me, mut me2) = (0.0, 0.0);
        for (&x0, &w0) in xk.iter().zip(&wk) {
            for (&x1, &w1) in xk.iter().zip(&wk) {
                for (&x2, &w2) in xk.iter().zip(&wk) {
                    let v = y(l, [sk * x0, sk * x1, sk * x2]);
                    me += w0 * w1 * w2 * v;
                    me2 += w0 * w1 * w2 * v * v;
                }
            }
        }
        m += wa * me;
        m2 += wa * me * me;
        w += wa * (me2 - me * me);
    }
    [m, m2 - m * m, w]
}

/// The fixture's fit: packed variance 0.04 on ln TVCL only (log-normal θ); every other
/// coordinate gets the sampler's floor, and the oracle reads each draw's θ, ω and ω_IOV
/// back from `result.parameters`, so the floor's movements are accounted for.
pub(super) fn iov_fit(m: &CompiledModel) -> FitResult {
    let mut fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
    fit.omega_iov = m.default_params.omega_iov.as_ref().map(|o| o.matrix.clone());
    fit.kappa_names = m.default_params.omega_iov.as_ref().map_or(vec![], |o| o.eta_names.clone());
    fit.kappa_fixed = m.default_params.kappa_fixed.clone();
    let cov = fit.covariance_matrix.as_mut().expect("synthetic fit has a matrix");
    cov.fill(0.0);
    cov[(0, 0)] = 0.04;
    fit
}

/// The plan's analytic decomposition, re-derived after PR1.5: metrics are scored on the
/// latent signal, so with a fixed controller the assay ε cannot reach `signal_mean` and the
/// within layer comes from the IOV κ alone (`~/ferx-probes/946/pr3/decomp_oracle_iov.R`:
/// param 10.60 (25.1%), BSV 26.48 (62.7%), within 5.16 (12.2%) amount², averaged over θ;
/// without κ the within layer is exactly 0, which is the exact-zero test above).
///
/// Conditional oracle: each layer is compared with its quadrature at the run's **own**
/// draws θ_d (`result.parameters`) — `Var_d E[Y|θ_d]` (sample variance), `mean_d Var_η`
/// and `mean_d E_η Var_κ` — which cancels the D-sampling error of the parameter layer.
/// D 100 × S 20 × R 2 on one subject. Tolerances are 4 SD of the gap, measured over 300
/// seeds in `cond_gap_sd.R`: SD 8.65% (param), 5.15% (BSV), 4.51% (within).
/// Realised gaps (macOS arm64): param −4.1% (0.48 SD), BSV +0.46% (0.09 SD), within −2.6%
/// (0.57 SD); 4000 runs in 0.7 s (debug).
///
/// Mutations: swap two layers (≥ 150% gaps); drop R from the BSV divisor (BSV ×2); η keyed
/// by rep (BSV → ~0, within absorbs it).
#[test]
fn the_layers_match_the_conditional_quadrature_at_the_runs_own_draws() {
    let m = model("omega ETA_CL ~ 0.09", "kappa KAPPA_CL ~ 0.04");
    let fit = iov_fit(&m);
    let params = fitted_params_from_result(&fit, &m).expect("template");
    let start = std::time::Instant::now();
    let r = simulate_adaptive_with_inputs(
        &m,
        &pop(1),
        &AdaptiveInputs::new(&params).with_uncertainty(
            &fit,
            UncertaintySpec { method: UncertaintyMethod::Asymptotic, n_draws: 100 },
        ),
        20,
        fixed_bolus,
        &opts(ObserveMode::Ipred, 2),
    )
    .expect("runs");
    eprintln!("4000 runs: {:.2} s", start.elapsed().as_secs_f64());
    let d = decompose_adaptive(&r).expect("decomposes");
    assert_eq!((d.n_draws, d.n_sims, d.n_reps), (100, 20, 2));

    let moments: Vec<[f64; 3]> = r
        .parameters
        .iter()
        .map(|p| {
            let q = &p.parameters;
            let iov = q.omega_iov.as_ref().expect("Ω_IOV").matrix[(0, 0)];
            conditional_moments(q.theta[0], q.theta[1], q.omega.matrix[(0, 0)], iov)
        })
        .collect();
    let n = moments.len() as f64;
    let mean_of = |k: usize| moments.iter().map(|q| q[k]).sum::<f64>() / n;
    let m0 = mean_of(0);
    let want = [
        moments.iter().map(|q| (q[0] - m0).powi(2)).sum::<f64>() / (n - 1.0),
        mean_of(1),
        mean_of(2),
    ];
    let s = subject("1");
    for ((layer, want), tol) in
        [L::Parameter, L::Bsv, L::Within].into_iter().zip(want).zip([0.35, 0.21, 0.18])
    {
        let (got, _) = est(&d, &s, layer);
        let gap = (got - want) / want;
        eprintln!("{layer:?}: engine {got:.5} quadrature {want:.5} gap {gap:+.4}");
        assert!(gap.is_finite() && gap.abs() < tol, "{layer:?}: {got} vs {want} ({gap:+.4})");
    }
}

/// The jackknife over three sims (D = 1), where `(n − 1)/n` and `1/n` differ (with two
/// units they coincide, so the 2 × 2 × 2 cube above cannot see that factor). Sims
/// [1, 3], [4, 8], [2, 2]: SSE per sim 2, 8, 0, so within = 10/6·2 = 10/3. Delete-one
/// within: (8 + 0)/2 = 4, (2 + 0)/2 = 1, (2 + 8)/2 = 5; mean 10/3, squared deviations
/// sum 78/9, SE = √(2/3 · 78/9) = √52 / 3.
///
/// Mutation: the jackknife's `(n − 1)/n` replaced by `1/n` (SE √78 / 9).
#[test]
fn the_jackknife_over_three_units_carries_the_n_minus_one_factor() {
    let y: [[[f64; 2]; 3]; 1] = [[[1.0, 3.0], [4.0, 8.0], [2.0, 2.0]]];
    let mut metrics = Vec::new();
    for (s, reps) in y[0].iter().enumerate() {
        for (r, &v) in reps.iter().enumerate() {
            metrics.push(metrics_row("1", (1, s + 1, r + 1), v));
        }
    }
    let res = AdaptiveSimulationResult { metrics, ..result_of(&[("1", &[CUBE[0]])], false) };
    let d = decompose_adaptive(&res).expect("decomposes");
    let (within, se) = est(&d, &subject("1"), L::Within);
    assert!((within - 10.0 / 3.0).abs() < 1e-12, "{within}");
    let se = se.expect("three sims form every delete-one replicate");
    assert!((se - 52f64.sqrt() / 3.0).abs() < 1e-12, "{se}");
}

// ── replicate labels in messages ───────────────────────────────────────────────────────

/// The rep is named only with replicates (`n_reps > 1`), after the draw and sim, so every
/// message of a run without replicates keeps its text.
///
/// Mutations: always printing the rep (the `None` side reddens); dropping it; printing it
/// before the sim.
#[test]
fn replicate_tag_names_the_rep_only_with_replicates() {
    use crate::sim::adaptive::ReplicateTag;
    let t = |draw, rep| ReplicateTag { draw, sim: 3, rep };
    assert_eq!(t(None, None).sim_label(), "sim 3");
    assert_eq!(t(None, Some(2)).sim_label(), "sim 3, rep 2");
    assert_eq!(t(Some(4), Some(2)).sim_label(), "draw 4, sim 3, rep 2");
    assert_eq!(t(None, Some(2)).replicate_label(), "replicate 3, rep 2");
    assert_eq!(t(Some(4), Some(2)).replicate_label(), "draw 4, replicate 3, rep 2");
}

/// `W_ADAPTIVE_STEP_ABSORBED` with replicates and no uncertainty names the first
/// replicate by its sim **and rep** (a bare sim number would not say which rep), and
/// counts over `n_sim × n_reps`.
///
/// Mutation: keeping the bare-sim branch for `(draw: None, rep: Some)` ("first in 1").
#[test]
fn absorbed_step_warning_names_the_rep_of_the_first_replicate() {
    use super::adaptive::{absorbed_step_warning, AbsorbedPlan};
    use crate::sim::adaptive::{AbsorbedIncrease::FromStart, ReplicateTag};
    let p = crate::parser::model_parser::parse_full_model(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [0, 24]
  start_dose = 500
  route = bolus(cmt=1)
  dose_bounds = [250, 4000]
  round_to = 250
  when signal < 0 : increase 10%
"#,
    )
    .expect("parse");
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let plan = |sim, rep| AbsorbedPlan {
        subject: 0,
        replicate: Some(ReplicateTag { draw: None, sim, rep: Some(rep) }),
        found: FromStart(500.0),
    };
    let w = absorbed_step_warning(spec, &pop(1), 4, vec![plan(2, 1), plan(1, 2)]).expect("a line");
    assert!(
        w.contains(
            "From the start dose on: '1' in 2 of 4 replicate(s), first in sim 1, rep 2 at 500."
        ),
        "{w}"
    );
}

/// The labels are ferx-r's tokens, one per variant, unique per enum (ferx-r#779 review
/// finding 4). The `match`es without a wildcard stop compiling when a variant is added.
///
/// Mutation: any label changed or duplicated.
#[test]
fn every_scope_layer_and_status_has_a_unique_stable_label() {
    let _exhaustive = |s: &Sc| match s {
        Sc::Population | Sc::Subject(_) => (),
    };
    assert_eq!(
        [Sc::Population, Sc::Subject("1".into())].map(|s| s.label()),
        ["population", "subject"]
    );

    let _exhaustive = |l: &L| match l {
        L::Parameter | L::Bsv | L::Within | L::BetweenRow | L::Total => (),
    };
    assert_eq!(
        [L::Parameter, L::Bsv, L::Within, L::BetweenRow, L::Total].map(|l| l.label()),
        ["parameter", "bsv", "within", "between_subject", "total"]
    );

    let _exhaustive = |s: &St| match s {
        St::Estimated { .. } | St::Supplied | St::NotDrawn => (),
    };
    assert_eq!(
        [St::Estimated { value: 1.0, mc_se: None }, St::Supplied, St::NotDrawn].map(|s| s.label()),
        ["estimated", "supplied", "not_drawn"]
    );
}
