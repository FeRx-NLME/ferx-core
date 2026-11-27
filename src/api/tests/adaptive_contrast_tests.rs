//! #946 PR4: the controller effect as a contrast against a reference regimen under common
//! random numbers (`decompose_contrast`, `summarize_contrast`, `fixed_reference`).
//!
//! The adaptive exception (AGENTS.md, epic #391): no NONMEM run. The kernel is pinned on
//! hand-computed arrays (exact binary arithmetic), the pairing on exact zeros through the
//! engine, and the difference on an analytic conditional oracle computed by quadrature here.

use super::adaptive_decomposition_tests::{
    est, gauss_hermite, iov_fit, model, opts, pop, result_of, subject, CUBE, DECISIONS,
};
use super::*;
use crate::estimation::uncertainty_samples::{fitted_params_from_result, UncertaintyMethod};
use crate::parser::model_parser::parse_full_model;
use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
use AdaptiveComponentStatus as St;
use AdaptiveSummaryMetric as M;
use AdaptiveVarianceLayer as L;
use AdaptiveVarianceScope as Sc;

fn labels() -> AdaptiveArmLabels {
    AdaptiveArmLabels::new("TDM rule", "start dose every due point")
}

/// `y[d][s][r]` for subject "1", scaled by `k`.
fn scaled(k: f64) -> [[[f64; 2]; 2]; 2] {
    CUBE.map(|d| d.map(|s| s.map(|v| k * v)))
}

/// Every estimated value and standard error of a decomposition, in row order.
fn estimates(d: &AdaptiveDecomposition) -> Vec<(f64, Option<f64>)> {
    d.components
        .iter()
        .filter_map(|c| match c.status {
            St::Estimated { value, mc_se } => Some((value, mc_se)),
            _ => None,
        })
        .collect()
}

// ── the kernel on hand-built pairs ─────────────────────────────────────────────────────

/// The reference is the adaptive arm doubled: every layer ×4, so adaptive − reference =
/// −3 × the adaptive layer, and the paired jackknife (one draw left out of both arms at
/// once) gives exactly 3 × the adaptive SE. On `CUBE` (see PR3's hand computation): BSV 8
/// (SE 2.5) and within 4.5 (SE 0.5) against 32 (SE 10) and 18 (SE 2), so the difference
/// is BSV −24 (SE 7.5) and within −13.5 (SE 1.5). Combining the arms' SEs unpaired,
/// √(se_A² + se_R²), would give 10.31 and 2.06.
///
/// Mutations: unpaired SE (10.31 / 2.06); Difference = reference − adaptive (+24); the arms
/// swapped in the output (`adaptive` ≠ `decompose_adaptive(adaptive)`).
#[test]
fn the_difference_is_adaptive_minus_reference_with_a_paired_standard_error() {
    let a = result_of(&[("1", &CUBE)], true);
    let r = result_of(&[("1", &scaled(2.0))], true);
    let c = decompose_contrast(&a, &r, &labels()).expect("contrasts");
    assert_eq!(c.adaptive, decompose_adaptive(&a).unwrap());
    assert_eq!(c.reference, decompose_adaptive(&r).unwrap());
    let s = subject("1");
    assert_eq!(est(&c.difference, &s, L::Bsv), (-24.0, Some(7.5)));
    assert_eq!(est(&c.difference, &s, L::Within), (-13.5, Some(1.5)));
    let (_, se_a) = est(&c.adaptive, &s, L::Bsv);
    let (_, se_r) = est(&c.reference, &s, L::Bsv);
    let unpaired = (se_a.unwrap().powi(2) + se_r.unwrap().powi(2)).sqrt();
    assert!((unpaired - 10.307764064044152).abs() < 1e-12, "{unpaired}");
    // The population lines are differences too (one subject: the same values; between-row 0).
    let p = Sc::Population;
    assert_eq!(est(&c.difference, &p, L::Bsv), (-24.0, Some(7.5)));
    assert_eq!(est(&c.difference, &p, L::BetweenRow), (0.0, Some(0.0)));
    // Two draws: one left cannot form the parameter layer, so its SE is `None` in each arm
    // and in the difference.
    assert_eq!(est(&c.difference, &s, L::Parameter), (-3.0 * -5.0, None));
    assert_eq!(c.labels, labels());
}

/// A metric missing in some run of either arm leaves that subject out of the difference
/// and lists it with the runs missing in **either** arm; each arm alone skips only its own.
/// The summary difference counts a run missing in either arm as missing.
///
/// Mutations: a difference formed when only one arm is complete; `n_missing` counting one
/// arm only (1 instead of 2); imputing a missing paired value.
#[test]
fn a_run_missing_in_either_arm_is_missing_in_the_difference_and_counted() {
    let mut a = result_of(&[("1", &CUBE), ("2", &CUBE)], true);
    let mut r = result_of(&[("1", &CUBE), ("2", &scaled(2.0))], true);
    a.metrics
        .iter_mut()
        .find(|m| m.subject == "2" && m.sim == 1 && m.draw == 1)
        .unwrap()
        .signal_mean = None;
    r.metrics
        .iter_mut()
        .find(|m| m.subject == "2" && m.sim == 2 && m.draw == 2)
        .unwrap()
        .signal_mean = None;
    let c = decompose_contrast(&a, &r, &labels()).expect("contrasts");
    let skip = |d: &AdaptiveDecomposition| -> Vec<(String, usize)> {
        d.skipped
            .iter()
            .filter(|s| s.metric == M::SignalMean)
            .map(|s| (s.subject.clone(), s.n_missing))
            .collect()
    };
    assert_eq!(skip(&c.adaptive), vec![("2".to_string(), 1)]);
    assert_eq!(skip(&c.reference), vec![("2".to_string(), 1)]);
    assert_eq!(skip(&c.difference), vec![("2".to_string(), 2)]);
    // Subject 1 is identical in both arms: its difference is exactly 0, and the population
    // difference is over subject 1 alone.
    assert_eq!(est(&c.difference, &subject("1"), L::Bsv), (0.0, Some(0.0)));
    let pop_bsv = c
        .difference
        .components
        .iter()
        .find(|x| x.metric == M::SignalMean && x.scope == Sc::Population && x.layer == L::Bsv);
    assert_eq!(pop_bsv.map(|x| x.n_subjects), Some(1));

    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    let mean = |d: usize| {
        s.difference
            .rows
            .iter()
            .find(|x| {
                x.metric == M::SignalMean
                    && x.statistic == AdaptiveSummaryStatistic::Mean
                    && x.draw == d
            })
            .map(|x| (x.value, x.n_runs, x.n_missing))
    };
    // Draw 1: subject 2's sim-1 runs (2 reps, 1 missing in the adaptive arm) → 1 missing of 8.
    assert_eq!(mean(1).map(|m| (m.1, m.2)), Some((8, 1)));
    assert_eq!(mean(2).map(|m| (m.1, m.2)), Some((8, 1)));
}

/// A non-finite value in one arm makes every difference it enters `NaN`; it is never
/// folded away.
///
/// Mutation: a NaN-dropping fold in the difference (`filter_map`/`max` over the pair).
#[test]
fn a_non_finite_value_propagates_into_the_difference() {
    let a = result_of(&[("1", &CUBE)], true);
    let mut r = result_of(&[("1", &CUBE)], true);
    r.metrics[3].signal_mean = Some(f64::NAN);
    let c = decompose_contrast(&a, &r, &labels()).expect("contrasts");
    let (bsv, _) = est(&c.difference, &subject("1"), L::Bsv);
    assert!(bsv.is_nan(), "{bsv}");
    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    let v = s.difference.rows.iter().find(|x| {
        x.metric == M::SignalMean && x.statistic == AdaptiveSummaryStatistic::Mean && x.draw == 1
    });
    assert!(v.and_then(|x| x.value).is_some_and(f64::is_nan), "{v:?}");
}

/// The summary difference is the summary of the per-run differences adaptive − reference,
/// and each arm's summary is what the arm alone gives. With the reference = 2 × adaptive,
/// draw 1's runs y = [1, 3, 4, 8] give differences −y: mean −4, median −3.5.
///
/// Mutations: the difference summary formed as reference − adaptive (+4 / +3.5); the arms
/// swapped in the output.
#[test]
fn the_summary_difference_summarises_the_paired_runs() {
    let a = result_of(&[("1", &CUBE)], true);
    let r = result_of(&[("1", &scaled(2.0))], true);
    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    assert_eq!(s.adaptive, summarize_adaptive(&a, &AdaptiveSummaryOptions::default()).unwrap());
    assert_eq!(s.reference, summarize_adaptive(&r, &AdaptiveSummaryOptions::default()).unwrap());
    let stat = |st: AdaptiveSummaryStatistic, d: usize| {
        s.difference
            .rows
            .iter()
            .find(|x| x.metric == M::SignalMean && x.statistic == st && x.draw == d)
            .and_then(|x| x.value)
    };
    // Draw 1: y = [1, 3, 4, 8], differences −y.
    assert_eq!(stat(AdaptiveSummaryStatistic::Mean, 1), Some(-4.0));
    assert_eq!(stat(AdaptiveSummaryStatistic::Median, 1), Some(-3.5));
    assert_eq!(s.labels, labels());
}

/// The difference's `mc_se` is paired: the jackknife over sims of the per-run differences,
/// leaving one sim out of both arms at once (#3215). Draw 1, differences by sim
/// [−1, −3], [−4, −8]: replicates −6 and −2, SE = √((1/2)·8) = 2, exactly. The arms' own
/// SEs are 2 (adaptive) and 4 (reference), so combining them unpaired gives √20.
///
/// Mutation: the difference's SE formed as `√(se_A² + se_R²)` (√20 ≠ 2).
#[test]
fn the_summary_difference_has_a_paired_mc_se() {
    let a = result_of(&[("1", &CUBE)], true);
    let r = result_of(&[("1", &scaled(2.0))], true);
    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    let se = |sum: &AdaptivePopulationSummary| {
        sum.rows
            .iter()
            .find(|x| {
                x.metric == M::SignalMean
                    && x.statistic == AdaptiveSummaryStatistic::Mean
                    && x.draw == 1
            })
            .and_then(|x| x.mc_se)
            .expect("an SE")
    };
    assert_eq!((se(&s.adaptive), se(&s.reference)), (2.0, 4.0));
    assert_eq!(se(&s.difference), 2.0);
    assert!(20f64.sqrt() - se(&s.difference) > 2.0);
}

// ── refusals ───────────────────────────────────────────────────────────────────────────

fn refusal(a: &AdaptiveSimulationResult, r: &AdaptiveSimulationResult) -> String {
    let e = decompose_contrast(a, r, &labels()).expect_err("refused");
    let code = e.diagnostic().map(|d| d.code.clone()).unwrap_or_default();
    assert_eq!(code, "E_ADAPTIVE_OPTIONS", "{e}");
    e.to_string()
}

fn pair() -> (AdaptiveSimulationResult, AdaptiveSimulationResult) {
    (result_of(&[("1", &CUBE), ("2", &CUBE)], true), result_of(&[("1", &CUBE), ("2", &CUBE)], true))
}

/// Input space, in check order: labels {both set, one blank} × seed {Some, None} per arm ×
/// seeds {equal, different} × point parameters {equal, a value differs} × draws {same
/// count, another count} × each draw's parameters {equal, differ} × grid {equal, differs} ×
/// `n_sim`, `n_reps` {equal, differ} × subjects {same, another count, another order} ×
/// supplied η {same, in one arm only, either arm}. Each message is asserted whole, so
/// deleting any sentence or clause reddens its cell, and the unchanged pair is accepted
/// (the straddle).
///
/// Mutation: accepting a pair with different seeds (the seed cell passes).
#[test]
fn a_pair_that_does_not_share_its_random_numbers_is_refused_naming_what_differs() {
    let (a, r) = pair();
    assert!(decompose_contrast(&a, &r, &labels()).is_ok());
    let pre = "decompose_contrast: ";
    let cell = |f: &dyn Fn(&mut AdaptiveSimulationResult, &mut AdaptiveSimulationResult)| {
        let (mut a, mut r) = pair();
        f(&mut a, &mut r);
        refusal(&a, &r)
    };

    let e = decompose_contrast(&a, &r, &AdaptiveArmLabels::new("TDM", " ")).unwrap_err();
    assert!(
        e.to_string().ends_with(&format!(
        "{pre}each arm needs a label that says what it is; got adaptive 'TDM' and reference ' '"
    )),
        "{e}"
    );
    let e = decompose_contrast(&a, &r, &AdaptiveArmLabels::new("", "ref")).unwrap_err();
    assert!(e.to_string().ends_with("got adaptive '' and reference 'ref'"), "{e}");

    for arm in ["adaptive", "reference"] {
        let e = cell(&|a, r| {
            if arm == "adaptive" {
                a.manifest.seed = None
            } else {
                r.manifest.seed = None
            }
        });
        assert!(
            e.ends_with(&format!(
                "{pre}the {arm} arm was run without a seed, so its random numbers came from \
             entropy and cannot be paired. Set `AdaptiveSimulateOptions::seed` to the same \
             value in both arms"
            )),
            "{e}"
        );
    }
    let e = cell(&|_, r| r.manifest.seed = Some(7));
    assert!(
        e.ends_with(&format!(
            "{pre}the arms were run with different seeds (adaptive 946, reference 7); common \
         random numbers need one seed"
        )),
        "{e}"
    );

    let e = cell(&|_, r| r.manifest.point_parameters.theta[0] = 6.0);
    assert!(
        e.ends_with(&format!(
            "{pre}the arms were run at different parameters: θ TVCL is 5, the reference arm's is 6"
        )),
        "{e}"
    );

    let e = cell(&|_, r| {
        r.parameters.pop();
    });
    assert!(
        e.ends_with(&format!(
            "{pre}the adaptive arm has 2 parameter draw(s) and the reference arm 1"
        )),
        "{e}"
    );
    let e = cell(&|_, r| r.parameters[1].parameters.omega.matrix[(0, 0)] = 0.1);
    assert!(
        e.ends_with(&format!(
            "{pre}parameter draw 2 differs: Ω[1, 1] is 0.09, the reference arm's is 0.1. Run both \
         arms with the same `with_uncertainty` fit and spec"
        )),
        "{e}"
    );

    let e = cell(&|_, r| r.manifest.decision_times = vec![0.0, 24.0]);
    assert!(
        e.ends_with(&format!(
            "{pre}the decision grids differ (adaptive [0.0, 24.0, 48.0], reference [0.0, 24.0]); \
         IOV occasions and assay draws are keyed by the grid index, so both arms need one grid"
        )),
        "{e}"
    );

    let e = cell(&|_, r| r.metrics.retain(|m| m.sim == 1));
    assert!(
        e.ends_with(&format!("{pre}the adaptive arm has n_sim = 2 and the reference arm 1")),
        "{e}"
    );
    let e = cell(&|_, r| r.metrics.retain(|m| m.rep == 1));
    assert!(
        e.ends_with(&format!("{pre}the adaptive arm has n_reps = 2 and the reference arm 1")),
        "{e}"
    );

    let e = cell(&|_, r| r.metrics.retain(|m| m.subject == "1"));
    assert!(
        e.ends_with(&format!("{pre}the adaptive arm has 2 subject(s) and the reference arm 1")),
        "{e}"
    );
    let e = cell(&|_, r| r.metrics.sort_by(|x, y| y.subject.cmp(&x.subject)));
    assert!(
        e.ends_with(&format!(
        "{pre}subject 1 is '1' in the adaptive arm and '2' in the reference arm; η is drawn in \
         population order, so both arms need the same population"
    )),
        "{e}"
    );

    let supply = |res: &mut AdaptiveSimulationResult| {
        res.metrics.iter_mut().filter(|m| m.subject == "2").for_each(|m| m.eta_supplied = true)
    };
    let e = cell(&|a, _| supply(a));
    assert!(
        e.ends_with(&format!("{pre}subject '2' has a supplied η in the adaptive arm only")),
        "{e}"
    );
    let e = cell(&|_, r| supply(r));
    assert!(
        e.ends_with(&format!("{pre}subject '2' has a supplied η in the reference arm only")),
        "{e}"
    );
    // Supplied in both arms is a pair: subject 2's BSV is `Supplied` in the difference too.
    let (mut a, mut r) = pair();
    supply(&mut a);
    supply(&mut r);
    let c = decompose_contrast(&a, &r, &labels()).expect("a supplied subject in both arms pairs");
    let bsv = c.difference.components.iter().find(|x| x.scope == subject("2") && x.layer == L::Bsv);
    assert_eq!(bsv.map(|x| x.status), Some(St::Supplied));
}

/// `summarize_contrast` runs the same pair check (its own prefix), and its own options
/// check first; an arm whose design cannot be decomposed is refused naming the arm.
///
/// Mutations: `summarize_contrast` skipping the pair check; an arm's design refusal without
/// the arm.
#[test]
fn both_entries_check_the_pair_and_name_the_arm_whose_design_fails() {
    let (a, mut r) = pair();
    r.manifest.seed = Some(7);
    let e = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap_err();
    assert!(
        e.to_string().contains("summarize_contrast: the arms were run with different seeds"),
        "{e}"
    );
    let bad = AdaptiveSummaryOptions { probs: [0.5, 0.5, 0.9], ..Default::default() };
    let e = summarize_contrast(&a, &a, &labels(), &bad).unwrap_err();
    assert!(e.to_string().contains("summarize_contrast: probabilities"), "{e}");

    // The reference arm misses one run: same n_sim / n_reps / subjects, an unbalanced design.
    let (a, mut r) = pair();
    r.metrics.remove(0);
    let e = refusal(&a, &r);
    assert!(e.ends_with(
        "decompose_contrast: the reference arm: the rows are not one balanced set of 2 draw(s) \
         × 2 sim(s) × 2 rep(s) per subject: subject '1' has 7 of 8 runs"
    ), "{e}");
    let (mut a, r) = pair();
    a.metrics.remove(0);
    assert!(refusal(&a, &r).contains("decompose_contrast: the adaptive arm: the rows"));
}

// ── through the engine ─────────────────────────────────────────────────────────────────

/// The run records what fixes its random numbers: the seed, the parameters it was given
/// and the grid, on both entries (the spec entry's grid is the block's `at`).
#[test]
fn the_manifest_records_the_seed_the_parameters_and_the_grid() {
    let m = model("omega ETA_CL ~ 0.09", "");
    let o = AdaptiveSimulateOptions { seed: Some(11), ..opts(ObserveMode::Ipred, 1) };
    let res = simulate_adaptive(&m, &pop(1), &m.default_params, 1, fixed_bolus_100, &o).unwrap();
    assert_eq!(res.manifest.seed, Some(11));
    assert_eq!(res.manifest.decision_times, DECISIONS.to_vec());
    assert_eq!(format!("{:?}", res.manifest.point_parameters), format!("{:?}", m.default_params));

    let (pm, spec) = spec_model();
    let o = AdaptiveSimulateOptions { seed: None, ..Default::default() };
    let res =
        simulate_adaptive_from_spec(&pm.model, &pop(1), &pm.model.default_params, 1, &spec, &o)
            .unwrap();
    assert_eq!(res.manifest.seed, None);
    assert_eq!(res.manifest.decision_times, spec.at);
}

fn fixed_bolus_100() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
}

/// The proportional controller of the oracle: `A0 − G·reading` at every decision, from the
/// (assay-noised under `Dv`) reading. The reading at t = 0 is 0, so its first dose is A0,
/// the fixed reference's dose.
const A0: f64 = 100.0;
const G: f64 = 0.5;
fn proportional() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        let amt = A0 - G * ctx.signal("A").expect("monitor A");
        assert!(amt > 0.0, "the closed form assumes a positive dose; got {amt}");
        vec![DoseAction::Bolus { amt, cmt: 1 }]
    }
}

/// The controller contrasted with itself: both runs bit-identical (ledger, decisions,
/// metrics, trajectories), and every difference exactly 0 with a jackknife SE of exactly
/// 0, under parameter uncertainty, IOV κ and a `Dv` monitor the controller reads, so every
/// random stream reaches the doses. The summary difference is exactly 0 in every row.
/// This pins that κ and ε are keyed, not consumed in order: nothing about the run but the
/// controller differs, and here not even that.
///
/// Mutations: any consumption-order dependence of a stream on the controller (none exists
/// to mutate in the kernel: the guard is the bit-identity); the paired SE replaced by the
/// unpaired one (SE > 0 on a zero difference).
#[test]
fn a_controller_contrasted_with_itself_is_exactly_zero_in_every_layer() {
    let m = model("omega ETA_CL ~ 0.09", "kappa KAPPA_CL ~ 0.04");
    let fit = iov_fit(&m);
    let params = fitted_params_from_result(&fit, &m).expect("template");
    let inputs = AdaptiveInputs::new(&params).with_uncertainty(
        &fit,
        UncertaintySpec { method: UncertaintyMethod::Asymptotic, n_draws: 3 },
    );
    let run = || {
        simulate_adaptive_with_inputs(
            &m,
            &pop(2),
            &inputs,
            3,
            proportional,
            &opts(ObserveMode::Dv, 2),
        )
        .expect("runs")
    };
    let (a, r) = (run(), run());
    assert_eq!(format!("{:?}", a.ledger), format!("{:?}", r.ledger));
    assert_eq!(format!("{:?}", a.decisions), format!("{:?}", r.decisions));
    assert_eq!(format!("{:?}", a.trajectories), format!("{:?}", r.trajectories));
    let before = format!("{a:?}");
    let c = decompose_contrast(&a, &r, &AdaptiveArmLabels::new("rule", "the same rule")).unwrap();
    // Post-processing only: the contrast borrows the results (`&`), so they are unchanged.
    // A record of that, not a check that could fail on its own.
    assert_eq!(format!("{a:?}"), before);
    let diffs = estimates(&c.difference);
    assert!(!diffs.is_empty());
    for (value, se) in &diffs {
        assert_eq!(value.to_bits(), 0f64.to_bits(), "{diffs:?}");
        assert!(se.is_none_or(|s| s.to_bits() == 0f64.to_bits()), "{diffs:?}");
    }
    // The arms themselves are live, so the zero is not vacuous.
    assert!(est(&c.adaptive, &subject("1"), L::Within).0 > 0.0);
    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    assert!(s.difference.rows.iter().all(|x| x.value.is_none_or(|v| v == 0.0)));
    assert!(s.difference.rows.iter().any(|x| x.value == Some(0.0)));
    // The paired summary SE is exactly 0 too (#3215); the arms' own SEs are not.
    assert!(s.difference.rows.iter().all(|x| x.mc_se.is_none_or(|v| v.to_bits() == 0)));
    assert!(s.difference.rows.iter().any(|x| x.mc_se == Some(0.0)));
    assert!(s.adaptive.rows.iter().any(|x| x.mc_se.is_some_and(|v| v > 0.0)));
}

/// The fixture's spec: a `with_assay_error` block on the PR3 model, so its single `signal`
/// monitor is a `Dv` monitor on cmt 1, which the programmatic entry can declare too.
fn spec_model() -> (crate::types::ParsedModel, crate::sim::adaptive::AdaptiveDosingSpec) {
    let base = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  with_assay_error = true
  assay_cmt = 1
  at = [0, 24, 48]
  start_dose = 100
  route = bolus(cmt = 1)
  dose_bounds = [0, 400]
  when signal < 50 : increase 25%
  when signal > 80 : decrease 25%
"#;
    let pm = parse_full_model(base).expect("parses");
    let spec = pm.adaptive_dosing.clone().expect("block");
    (pm, spec)
}

/// A fixed regimen run on both entries is the same run: the spec's `fixed_reference()`
/// (start dose 100 at every due point) through `simulate_adaptive_from_spec`, and a
/// programmatic controller emitting 100 every decision with the same `Dv` monitor. Their
/// metrics are bit-identical and the contrast is exactly 0. Against the spec's own rule
/// the difference is live.
///
/// Mutations: `fixed_reference` keeping a rule (the doses move: metrics differ); the spec
/// entry seeding the assay off another key than the programmatic entry.
#[test]
fn a_fixed_controller_and_the_spec_fixed_reference_are_the_same_regimen() {
    let (pm, spec) = spec_model();
    let m = &pm.model;
    let o = AdaptiveSimulateOptions { seed: Some(5), n_reps: 2, ..Default::default() };
    let p = &m.default_params;
    let reference =
        simulate_adaptive_from_spec(m, &pop(2), p, 3, &spec.fixed_reference(), &o).unwrap();
    let po = AdaptiveSimulateOptions {
        decision_times: spec.at.clone(),
        monitors: vec![MonitorSpec::new("signal", 1, ObserveMode::Dv)],
        ..o.clone()
    };
    let fixed = simulate_adaptive(m, &pop(2), p, 3, fixed_bolus_100, &po).unwrap();
    assert_eq!(format!("{:?}", reference.metrics), format!("{:?}", fixed.metrics));
    assert_eq!(format!("{:?}", reference.ledger), format!("{:?}", fixed.ledger));
    let c = decompose_contrast(&fixed, &reference, &labels()).unwrap();
    assert!(estimates(&c.difference).iter().all(|(v, _)| *v == 0.0));

    let adaptive = simulate_adaptive_from_spec(m, &pop(2), p, 3, &spec, &o).unwrap();
    let c = decompose_contrast(&adaptive, &reference, &labels()).unwrap();
    assert!(estimates(&c.difference).iter().any(|(v, _)| *v != 0.0));
}

// ── the analytic conditional oracle ────────────────────────────────────────────────────

/// Per arm, `[E m, Var_η m, E_η Var_W]` at one draw `(θ_CL, V, ω², σ)`, by quadrature over
/// η outside the engine. Readings are the pre-dose amounts at 0, 24, 48 (0, `A₁`, `A₂`),
/// `x = exp(−24·θ e^η / V)`, `Y = signal_mean` scored on the latent signal.
/// Reference (100 every decision): `A₁ = 100x`, `A₂ = x(A₁ + 100)`, so
/// `Y = 100(2x + x²)/3` and no within variance (doses ignore the reading).
/// Proportional (`A0 − G·reading`, the reading `A₁(1 + σε)`): the dose at 24 is
/// `A0 − G·A₁(1 + σε)`, so `Y = A0(2x + (1 − G)x²)/3 − (A0·G·σ·x²/3)·ε`: mean
/// `A0(2x + (1 − G)x²)/3`, within variance `(A0·G·σ·x²/3)²`. The dose at 48 reaches no
/// reading.
fn arm_moments(adaptive: bool, cl: f64, v: f64, omega2: f64, sigma: f64) -> [f64; 3] {
    let (z, w) = gauss_hermite(60);
    let (mut m, mut m2, mut within) = (0.0, 0.0, 0.0);
    for (&zk, &wk) in z.iter().zip(&w) {
        let x = (-24.0 * cl * (omega2.sqrt() * zk).exp() / v).exp();
        let (mean, var_w) = if adaptive {
            (A0 * (2.0 * x + (1.0 - G) * x * x) / 3.0, (A0 * G * sigma * x * x / 3.0).powi(2))
        } else {
            (A0 * (2.0 * x + x * x) / 3.0, 0.0)
        };
        m += wk * mean;
        m2 += wk * mean * mean;
        within += wk * var_w;
    }
    [m, m2 - m * m, within]
}

/// The conditional layers `[param, BSV, within]` of one arm at the run's own draws:
/// `Var_d E[Y|θ_d]` (sample variance), `mean_d Var_η`, `mean_d E_η Var_W`.
fn conditional_layers(r: &AdaptiveSimulationResult, adaptive: bool) -> [f64; 3] {
    let q: Vec<[f64; 3]> = r
        .parameters
        .iter()
        .map(|p| {
            let p = &p.parameters;
            arm_moments(adaptive, p.theta[0], p.theta[1], p.omega.matrix[(0, 0)], p.sigma.values[0])
        })
        .collect();
    let n = q.len() as f64;
    let mean_of = |k: usize| q.iter().map(|x| x[k]).sum::<f64>() / n;
    let m0 = mean_of(0);
    [q.iter().map(|x| (x[0] - m0).powi(2)).sum::<f64>() / (n - 1.0), mean_of(1), mean_of(2)]
}

/// The difference against its conditional quadrature: a proportional controller with gain
/// G = 0.8 on a `Dv` reading against the fixed 100 it starts from, one subject, D 100 × S 20
/// × R 2, the real asymptotic sampler (log-normal TVCL). Each arm's layers and the
/// difference are computed outside the engine at the run's own draws.
///
/// Expected sign pattern (plan §3, a sanity print, not an assertion): the parameter and BSV
/// layers shrink under the compensating controller, the within layer grows from exactly 0.
///
/// Mutations: arms swapped or Difference = reference − adaptive (the sign of every
/// difference flips); the difference formed from one arm only.
#[test]
fn the_difference_matches_the_conditional_quadrature_at_the_runs_own_draws() {
    let m = oracle_model();
    let fit = iov_fit(&m);
    let params = fitted_params_from_result(&fit, &m).expect("template");
    let inputs = AdaptiveInputs::new(&params).with_uncertainty(
        &fit,
        UncertaintySpec { method: UncertaintyMethod::Asymptotic, n_draws: 100 },
    );
    let o = opts(ObserveMode::Dv, 2);
    let a = simulate_adaptive_with_inputs(&m, &pop(1), &inputs, 20, proportional, &o).unwrap();
    let r = simulate_adaptive_with_inputs(&m, &pop(1), &inputs, 20, fixed_bolus_100, &o).unwrap();
    let c = decompose_contrast(&a, &r, &labels()).expect("contrasts");
    let (qa, qr) = (conditional_layers(&a, true), conditional_layers(&r, false));
    let s = subject("1");
    for (k, (layer, tol)) in
        [(L::Parameter, TOL[0]), (L::Bsv, TOL[1]), (L::Within, TOL[2])].into_iter().enumerate()
    {
        let want = qa[k] - qr[k];
        let (got, se) = est(&c.difference, &s, layer);
        let gap = (got - want) / want.abs();
        eprintln!(
            "{layer:?}: adaptive {:.4} (q {:.4}) reference {:.4} (q {:.4}) difference {got:.4} \
             (q {want:.4}, se {se:?}) gap {gap:+.4}",
            est(&c.adaptive, &s, layer).0,
            qa[k],
            est(&c.reference, &s, layer).0,
            qr[k]
        );
        assert!(gap.is_finite() && gap.abs() < tol, "{layer:?}: {got} vs {want} ({gap:+.4})");
    }
}

/// Per-run moments of the difference `D = Y_a − Y_r` at one draw, by quadrature over η
/// (from `arm_moments`' closed forms: the reference has no reading term):
/// `D = −(A0·G/3)·x² − (A0·G·σ/3)·x²·ε`, so `[Var_η E[D|η], E_η Var_W(D|η)]`.
fn difference_moments(cl: f64, v: f64, omega2: f64, sigma: f64) -> [f64; 2] {
    let (z, w) = gauss_hermite(60);
    let (mut m, mut m2, mut within) = (0.0, 0.0, 0.0);
    for (&zk, &wk) in z.iter().zip(&w) {
        let x = (-24.0 * cl * (omega2.sqrt() * zk).exp() / v).exp();
        let mean = -A0 * G * x * x / 3.0;
        m += wk * mean;
        m2 += wk * mean * mean;
        within += wk * (A0 * G * sigma * x * x / 3.0).powi(2);
    }
    [m2 - m * m, within]
}

/// The summary difference's `mc_se` against the analytic Monte Carlo SD of the mean
/// effect, without parameter uncertainty (#3215): one subject, S 200 × R 2, so the row's
/// mean has SD `√((Var_η E[D|η] + E Var_W / R) / S)` (the two reps of a sim share η).
///
/// Band: over 300 seeds at this design (`~/ferx-probes/946/mc_se/`) the SE's RMS was 0.995
/// of this SD and its own coefficient of variation 0.062; the band is ±4 of that, ±25 %.
/// Realised at seed 946 (macOS arm64): mc_se 0.0995, SD 0.1013, ratio 0.982.
///
/// Mutation: the run as the unit instead of the sim (the reps share η). Measured at seed
/// 946: mc_se 0.0738, ratio 0.728, outside the band by only 0.022 (0.35 cv), so this test
/// is a weak kill; the hand test
/// (`the_mc_se_is_the_jackknife_over_sims_by_hand_and_none_for_the_median`) is the robust
/// one.
#[test]
fn the_mean_effect_mc_se_matches_the_analytic_monte_carlo_sd_without_uncertainty() {
    let m = oracle_model();
    let params = fitted_params_from_result(&iov_fit(&m), &m).expect("template");
    let inputs = AdaptiveInputs::new(&params);
    let (n_sim, n_reps) = (200, 2);
    let o = opts(ObserveMode::Dv, n_reps);
    let a = simulate_adaptive_with_inputs(&m, &pop(1), &inputs, n_sim, proportional, &o).unwrap();
    let r =
        simulate_adaptive_with_inputs(&m, &pop(1), &inputs, n_sim, fixed_bolus_100, &o).unwrap();
    let s = summarize_contrast(&a, &r, &labels(), &AdaptiveSummaryOptions::default()).unwrap();
    let row = s
        .difference
        .rows
        .iter()
        .find(|x| x.metric == M::SignalMean && x.statistic == AdaptiveSummaryStatistic::Mean)
        .expect("the mean effect");
    let p = &params;
    let [var_eta, within] =
        difference_moments(p.theta[0], p.theta[1], p.omega.matrix[(0, 0)], p.sigma.values[0]);
    let want = ((var_eta + within / n_reps as f64) / n_sim as f64).sqrt();
    let got = row.mc_se.expect("an SE");
    let ratio = got / want;
    eprintln!("mean effect {:?} mc_se {got:.5} analytic SD {want:.5} ratio {ratio:.3}", row.value);
    assert!(ratio.is_finite() && (ratio - 1.0).abs() < 0.25, "{got} vs {want} ({ratio:.3})");
}

/// The PR3 model at TVCL = 2 (`x = e^{−0.96}` ≈ 0.38 at the typical value), so the
/// `x²` terms the controller moves are not negligible next to `x`.
fn oracle_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(ORACLE_MODEL).expect("parse")
}

const ORACLE_MODEL: &str = r#"
[parameters]
  theta TVCL(2.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

/// Tolerances of the oracle above, relative to |difference|: 4 SD of the gap, measured over
/// 300 seeds at this design (`~/ferx-probes/946/pr4/probe_se_and_gap.rs`): gap SD 7.0%
/// (param), 3.2% (BSV), 5.0% (within), mean gap < 0.5% in each. Realised gaps at seed 946
/// (macOS arm64): param +16.1% (2.3 SD), BSV −2.5% (0.8 SD), within −7.9% (1.6 SD).
const TOL: [f64; 3] = [0.28, 0.13, 0.20];
