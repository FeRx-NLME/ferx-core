//! FREM covariate rows on the adaptive path (#2156), and the `W_FREM_DRAWN_ETA` warning every
//! simulator raises when it draws a FREM model's covariate η (#2245, at the end).
//!
//! `simulate_adaptive` returned the ODE readout on a `FREMTYPE > 0` row, where every
//! other prediction path returns the covariate pseudo-observation `θ_k + η_m` (the
//! `pk` sink's last step, `apply_frem_prediction_override`): 2.0 where `simulate()`
//! gave 69.99. Validation follows the adaptive-dosing exception in AGENTS.md (no
//! NONMEM): the degenerate oracle (a holding controller equals `simulate()` row for
//! row), a closed form (a supplied η makes the FREM row exactly `θ_k + η_m`), and the
//! row conversion pinned bit for bit against the full `pk` sink, with and without IOV.
use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveMonitor, AdaptiveRoute, AdaptiveRule, Comparison,
    ControllerCtx, DoseAction, DoseStep, MonitorSpec, ObserveMode,
};
use std::collections::HashMap;

/// 1-cpt ODE plus an independent second state (so Form B has two observed CMTs), with
/// one FREM covariate `WT`: `TV_WT` + `ETA_WT` predict the `FREMTYPE = 100` rows. With
/// `iov`, `V` carries a κ, so the run is an IOV run.
fn model_src(readout: &str, iov: bool, error: &str) -> String {
    let (kappa, v) = if iov {
        ("  kappa KAPPA_V ~ 0.04", "TVV * exp(ETA_V + KAPPA_V)")
    } else {
        ("", "TVV * exp(ETA_V)")
    };
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TV_WT(70.0) FIX
  omega ETA_V ~ 0.09
  omega ETA_WT ~ 100.0
{kappa}
  sigma S ~ 0.04
  sigma EPSCOV ~ 0.0001 FIX
[individual_parameters]
  CL = TVCL
  V  = {v}
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL / V) * central
  d/dt(periph) = -0.2 * periph
[scaling]
{readout}
[error_model]
{error}
[fit_options]
  frem_predictions = TV_WT/ETA_WT:100
  frem_sigma = EPSCOV
"#
    )
}

const PROP: &str = "  DV ~ proportional(S)";
const PER_CMT_ERR: &str = "  CMT=1: DV ~ proportional(S)\n  CMT=2: DV ~ proportional(S)";

/// The three readout forms, each with the CMTs its observations sit on and its error
/// model. Form A divides by an η-dependent `V`, Form B per CMT, Form C is the twin.
const FORMS: [(&str, &str, [usize; 6]); 3] = [
    ("A: obs_scale = V", "  obs_scale = V", [1; 6]),
    (
        "B: obs_scale[CMT=n]",
        "  obs_scale[CMT=1] = V\n  obs_scale[CMT=2] = 10",
        [1, 1, 2, 1, 1, 2],
    ),
    ("C: y = central / V", "  y = central / V", [1; 6]),
];

fn model(readout: &str, iov: bool, cmts: [usize; 6]) -> CompiledModel {
    let err = if cmts == [1; 6] { PROP } else { PER_CMT_ERR };
    let m = parse_model_string(&model_src(readout, iov, err)).expect("parse");
    assert!(m.frem_config.is_some(), "fixture is a FREM model");
    assert_eq!(m.n_kappa, usize::from(iov), "fixture IOV flag");
    m
}

/// Observation grid. Rows 0 and 4 are `FREMTYPE = 100` covariate rows: one before the
/// first decision and one inside the last decision window, so under IOV both the κ = 0
/// stretch and a κ-carrying window hold a FREM row.
const OBS: [f64; 6] = [0.5, 1.0, 4.0, 13.0, 14.0, 20.0];
const FREMTYPE: [u16; 6] = [100, 0, 0, 0, 100, 0];
const FREM_ROWS: [usize; 2] = [0, 4];
const DECISIONS: [f64; 2] = [4.0, 12.0];

/// Occasion label per row, parallel to `OBS`: the decision window a row falls in
/// (`occasion_of`), `0` before the first decision. Used only by the sink parity test,
/// so `predict_iov` reads each row under the κ the adaptive driver reads it with.
fn window_occasions() -> Vec<u32> {
    OBS.iter()
        .map(|&t| crate::pk::occasion_of(&DECISIONS, t).map_or(0, |g| g as u32 + 1))
        .collect()
}

fn subject(id: &str, cmts: [usize; 6], occasions: Vec<u32>) -> Subject {
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(0.0, 80.0, 2, 0.0, false, 0.0),
    ];
    let n = OBS.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.to_string(),
        dose_occasions: vec![occasions[0]; doses.len()],
        doses,
        obs_times: OBS.to_vec(),
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: cmts.to_vec(),
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions,
        obs_l2: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: FREMTYPE.to_vec(),
        obs_records: vec![],
    }
}

fn population(cmts: [usize; 6]) -> Population {
    Population {
        subjects: vec![subject("1", cmts, vec![1u32; OBS.len()])],
        covariate_names: vec![],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn opts() -> AdaptiveSimulateOptions {
    let mut o = AdaptiveSimulateOptions::default();
    o.seed = Some(7);
    o.decision_times = DECISIONS.to_vec();
    o.monitors = vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)];
    assert!(o.verify, "the frozen-replay verifier runs on every case");
    o
}

fn hold() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_c: &ControllerCtx| vec![DoseAction::Hold]
}

/// A from-spec block that re-issues a zero dose at each decision (the bounds pin it and
/// its one rule never fires), so the realized regimen is the base regimen.
fn spec_hold() -> AdaptiveDosingSpec {
    AdaptiveDosingSpec {
        observe: Some("central / V".to_string()),
        observe_declared_covariates: Vec::new(),
        with_assay_error: false,
        assay_cmt: None,
        at: DECISIONS.to_vec(),
        start_dose: 0.0,
        route: AdaptiveRoute::Bolus { cmt: 1 },
        dose_bounds: (0.0, 0.0),
        confirm: 1,
        levels: None,
        target_window: None,
        auc_target: None,
        rules: vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: -1e9,
            action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
        }],
    }
}

fn spec_opts() -> AdaptiveSimulateOptions {
    let mut o = AdaptiveSimulateOptions::default();
    o.seed = Some(7);
    o
}

fn ipreds(r: &AdaptiveSimulationResult) -> Vec<f64> {
    r.trajectories.iter().map(|t| t.ipred).collect()
}

/// The trajectory value must carry the same number as `ipred` (no residual draw).
fn assert_values_are_ipreds(r: &AdaptiveSimulationResult, label: &str) {
    for t in &r.trajectories {
        match t.outcome {
            SimOutcome::Continuous { value } => assert_eq!(
                value.to_bits(),
                t.ipred.to_bits(),
                "{label}: value vs ipred at t={}",
                t.time
            ),
            ref o => panic!("{label}: continuous row expected, got {o:?}"),
        }
    }
}

/// A FREM row sits at `TV_WT + η_WT`, two orders above the concentration (2·e^{-0.05}
/// at t = 0.5 for η_V = 0). This straddle keeps every FREM assertion live: a FREM row
/// left on the readout cannot be within 1 of 70 unless η_WT ≈ -68, i.e. 6.8 SD.
fn assert_frem_rows_are_not_the_readout(rows: &[f64], label: &str) {
    for j in FREM_ROWS {
        assert!(
            rows[j] > 20.0,
            "{label}: FREM row {j} = {} is the readout, not θ + η",
            rows[j]
        );
    }
}

/// The degenerate oracle on FREM rows: a holding controller reproduces `simulate()`
/// row for row — FREM covariate rows included — under Form A, B and C, through both
/// non-supplied entry points, with the frozen-replay verifier on.
///
/// Same seed ⇒ same η: both entry points draw η first from one seeded stream. Every
/// row is compared bit for bit: the realised difference is exactly 0 on all 24 PK rows
/// (the conversion runs the sink's own `apply_scaling` on the same readout), and the
/// FREM rows are `θ + η` on both sides.
///
/// Regression: drop `apply_frem_prediction_override` from
/// `adaptive_rows_to_model_scale` → each FREM row is the readout (Form A row 0: 2.5545
/// against `simulate()`'s 64.2325, measured), and the parity and the straddle fail.
#[test]
fn holding_controller_matches_simulate_on_frem_rows() {
    for (label, readout, cmts) in FORMS {
        let m = model(readout, false, cmts);
        let pop = population(cmts);
        let reference = simulate_with_seed(&m, &pop, &m.default_params, 1, 7).expect("sim");
        let want: Vec<f64> = reference.iter().map(|r| r.ipred).collect();
        assert_frem_rows_are_not_the_readout(&want, &format!("{label}: simulate()"));

        let programmatic =
            simulate_adaptive(&m, &pop, &m.default_params, 1, hold, &opts()).expect("adaptive");
        let declarative =
            simulate_adaptive_from_spec(&m, &pop, &m.default_params, 1, &spec_hold(), &spec_opts())
                .expect("from spec");
        for (entry, run) in [
            ("simulate_adaptive", &programmatic),
            ("simulate_adaptive_from_spec", &declarative),
        ] {
            let label = format!("{label}, {entry}");
            let got = ipreds(run);
            assert_eq!(got.len(), want.len(), "{label}: row count");
            assert_values_are_ipreds(run, &label);
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(g.is_finite(), "{label}: row {j} = {g}");
                assert_eq!(g.to_bits(), w.to_bits(), "{label}: row {j}: {g} vs {w}");
            }
            assert_frem_rows_are_not_the_readout(&got, &label);
        }
    }
}

/// The closed form: with η supplied, a FREM row is exactly `TV_WT + η_WT` — plain and
/// under IOV (where `simulate()` draws its κ per data occasion and so is no reference),
/// through both supplied-η entry points, under every readout form.
///
/// Under IOV the FREM row in the last decision window (row 4) is the one that sees the
/// window's `[η | κ]`: the override must read the BSV slot, never a κ slot.
///
/// Regression: drop the FREM step → Form A row 0 reads 1.5718 against 81.5 (measured).
#[test]
fn supplied_eta_frem_rows_are_theta_plus_eta() {
    const ETA_V: f64 = 0.2;
    const ETA_WT: f64 = 11.5;
    let e = SuppliedEtas::from_table(
        vec!["1".to_string()],
        vec!["ETA_V".to_string(), "ETA_WT".to_string()],
        vec![vec![ETA_V, ETA_WT]],
    )
    .expect("valid table");
    for iov in [false, true] {
        for (label, readout, cmts) in FORMS {
            let m = model(readout, iov, cmts);
            let want = m.default_params.theta[2] + ETA_WT;
            assert_eq!(want, 81.5, "TV_WT is 70");
            let pop = population(cmts);
            let inputs = AdaptiveInputs::new(&m.default_params).with_etas(&e);
            let programmatic = simulate_adaptive_with_inputs(&m, &pop, &inputs, 1, hold, &opts())
                .expect("adaptive with inputs");
            let declarative = simulate_adaptive_from_spec_with_inputs(
                &m,
                &pop,
                &inputs,
                1,
                &spec_hold(),
                &spec_opts(),
            )
            .expect("from spec with inputs");
            for (entry, run) in [
                ("simulate_adaptive_with_inputs", &programmatic),
                ("simulate_adaptive_from_spec_with_inputs", &declarative),
            ] {
                let label = format!("{label}, IOV {iov}, {entry}");
                let got = ipreds(run);
                assert_values_are_ipreds(run, &label);
                for j in FREM_ROWS {
                    assert_eq!(got[j], want, "{label}: FREM row {j}");
                }
                for (j, g) in got.iter().enumerate() {
                    if FREMTYPE[j] == 0 {
                        assert!(
                            g.is_finite() && *g > 0.0 && *g < 20.0,
                            "{label}: PK row {j} = {g} is not a concentration"
                        );
                    }
                }
            }
        }
    }
}

/// `adaptive_rows_to_model_scale` against the **full** `pk` sink, bit for bit — not
/// against a list of the sink's steps, so a step the sink gains later fails here.
///
/// - raw: the sink on the same model with `[scaling]` and FREM stripped (the engine
///   readout, from whichever route the sink takes);
/// - expected: the sink on the full model — `compute_predictions_with_tv`, or under
///   IOV `predict_iov` with `subject.occasions` set to the decision windows
///   (`occasion_of`), so every row is read under the κ the adaptive driver uses.
///
/// η is non-zero (it moves Form A's `V` and the FREM row) and each window's κ differs,
/// so a κ-blind or η-blind conversion cannot pass.
///
/// Regressions, each measured, each failing on the first form (Form A): drop the FREM
/// step → row 0 is 1.5718 against 81.5. Under IOV, divide every row under the baseline
/// `eta_slice` instead of its window's (the conversion side) → row 2 is 1.2577 against
/// the sink's 0.9318; feed the sink one occasion for all rows instead of the decision
/// windows (the sink side) → row 1 is 1.5411 against 1.1417.
#[test]
fn row_conversion_matches_the_full_pk_sink() {
    let eta_bsv = [0.2, 11.5];
    let kappa = [[0.0], [0.3], [-0.25]];
    for iov in [false, true] {
        for (label, readout, cmts) in FORMS {
            let label = format!("{label}, IOV {iov}");
            let m = model(readout, iov, cmts);
            let theta = &m.default_params.theta;
            // `CompiledModel` is not `Clone`: parse the fixture again and strip it.
            let mut stripped = model(readout, iov, cmts);
            stripped.scaling = crate::types::ScalingSpec::None;
            stripped.frem_config = None;

            let occasions = if iov {
                window_occasions()
            } else {
                vec![1u32; OBS.len()]
            };
            let s = subject("1", cmts, occasions);
            let mut eta_slice = eta_bsv.to_vec();
            eta_slice.resize(m.n_eta + m.n_kappa, 0.0);

            let (mut got, want, eta_occ) = if iov {
                // κ per occasion id, in `predict_iov`'s group order.
                let kappas: Vec<Vec<f64>> = crate::stats::likelihood::iov_occasion_groups(&s)
                    .iter()
                    .map(|(occ, _)| kappa[*occ as usize].to_vec())
                    .collect();
                let raw = crate::pk::predict_iov(&stripped, &s, theta, &eta_bsv, &kappas);
                let want = crate::pk::predict_iov(&m, &s, theta, &eta_bsv, &kappas);
                // The adaptive driver's per-window `[η | κ_g]`.
                let eta_occ: Vec<Vec<f64>> = (0..DECISIONS.len())
                    .map(|g| [eta_bsv.as_slice(), &kappa[g + 1]].concat())
                    .collect();
                (raw, want, Some(eta_occ))
            } else {
                let raw = crate::pk::compute_predictions_with_tv(&stripped, &s, theta, &eta_slice);
                let want = crate::pk::compute_predictions_with_tv(&m, &s, theta, &eta_slice);
                (raw, want, None)
            };
            // The fixture must exercise what it pins: the raw FREM rows are the readout,
            // not the sink's θ + η, and (Form A/B) the raw PK rows are amounts.
            assert_frem_rows_are_not_the_readout(&want, &label);
            for j in FREM_ROWS {
                assert!(
                    (got[j] - want[j]).abs() > 0.01 * want[j],
                    "{label}: raw FREM row {j} {} already equals the sink's {}",
                    got[j],
                    want[j]
                );
            }
            if !readout.contains("y =") {
                assert!(
                    (got[1] - want[1]).abs() > 0.01 * want[1],
                    "{label}: [scaling] is live on row 1: {} vs {}",
                    got[1],
                    want[1]
                );
            }

            super::adaptive::adaptive_rows_to_model_scale(
                &m,
                &s,
                theta,
                &eta_slice,
                eta_occ.as_deref(),
                &DECISIONS,
                &mut got,
            );
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(w.is_finite(), "{label}: sink row {j} = {w}");
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{label}: row {j}: conversion {g} vs sink {w}"
                );
            }
        }
    }
}

/// A monitor with a `[scaling]` divisor of 2 on CMT 1, for the window-AUC divide tests.
fn auc_divide(m: &CompiledModel, s: &Subject, eta_occ: Option<&[Vec<f64>]>) -> Vec<f64> {
    let spec = MonitorSpec::new("A", 1, ObserveMode::Ipred);
    let two = |_eta: &[f64]| 2.0;
    let mon = AdaptiveMonitor {
        spec: &spec,
        observe: None,
        obs_scale: Some(&two),
    };
    super::adaptive::adaptive_window_aucs_to_model_scale(
        m,
        s,
        &mon,
        &[0.0, 0.0],
        eta_occ,
        vec![4.0, 6.0],
    )
}

/// The control for the two `#[should_panic]` tests below: a subject with no
/// decision-window κ and no per-event PK divides each window AUC by its one divisor and
/// trips neither assertion.
#[test]
fn window_auc_divide_on_a_constant_divisor() {
    let m = model("  obs_scale = V", false, [1; 6]);
    let s = subject("1", [1; 6], vec![1u32; OBS.len()]);
    assert!(!crate::pk::subject_needs_per_event_pk(&m, &s));
    assert_eq!(auc_divide(&m, &s, None), vec![2.0, 3.0]);
}

/// The `auc_target` divide states its first condition (#2156): handed a decision-window
/// κ, it fails in debug instead of dividing every window by the κ = 0 divisor. The
/// fixture's `V` reads `KAPPA_V`, so κ really moves the `obs_scale = V` divisor. Today
/// the `auc_target` refusal keeps IOV out before this point; the assertion is what fails
/// if that refusal is ever widened without the per-grid-point conversion (#943).
///
/// Regression: drop the `eta_occ.is_none()` assertion → this returns.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "has a decision-window κ, so its [scaling] divisor can change")]
fn window_auc_divide_asserts_no_decision_kappa() {
    let m = model("  obs_scale = V", true, [1; 6]);
    let s = subject("1", [1; 6], vec![1u32; OBS.len()]);
    let eta_occ = vec![vec![0.0, 0.0, 0.3], vec![0.0, 0.0, -0.25]];
    auc_divide(&m, &s, Some(&eta_occ));
}

/// The second condition: a subject that needs per-event PK. Here `CL` reads a `WT` that
/// moves from 70 to 90 across the horizon. The `obs_scale = V` divisor does not move (it
/// never follows the covariates), but the window pass would integrate the AUC from one
/// frozen PK snapshot, so the integrand itself is wrong.
///
/// Regression: drop the `!subject_needs_per_event_pk` assertion → this returns.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "needs per-event PK, but the window AUCs integrate one frozen PK")]
fn window_auc_divide_asserts_no_per_event_pk() {
    let src = model_src("  obs_scale = V", false, PROP).replace("CL = TVCL", "CL = TVCL * WT / 70");
    let m = parse_model_string(&src).expect("parse");
    let mut s = subject("1", [1; 6], vec![1u32; OBS.len()]);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    s.covariates = wt(70.0);
    s.obs_covariates = OBS
        .iter()
        .map(|&t| wt(if t < 5.0 { 70.0 } else { 90.0 }))
        .collect();
    assert!(
        crate::pk::subject_needs_per_event_pk(&m, &s),
        "fixture moves the PK"
    );
    auc_divide(&m, &s, None);
}

// ---- W_FREM_DRAWN_ETA (#2245) ---------------------------------------------------------
//
// A FREM model links its covariates to the PK only through Ω, so a subject simulated at a
// drawn η has PK that follows the drawn covariate η, not its own covariate values. Every
// path that draws η says so; every path that does not (propensity matching, supplied η)
// stays silent. Each test asserts the whole message, so deleting any sentence of it — the
// shared head, the gated FREMTYPE sentence, or an entry point's remedy — reddens one.

const DRAWN_CODE: &str = "W_FREM_DRAWN_ETA";

/// The message head for `n` of `of` drawn subjects; `rows` adds the FREMTYPE sentence.
fn drawn_head(n: usize, of: usize, rows: bool) -> String {
    let rows = if rows {
        " Their FREMTYPE rows report the drawn covariate values, not the ones in the data."
    } else {
        ""
    };
    format!(
        "W_FREM_DRAWN_ETA: {n} of {of} subject(s) of this FREM model were simulated with an \
         eta drawn from Omega, covariate etas included. A FREM model links its covariates to \
         the PK only through the correlations in Omega, so the PK of each of these subjects \
         follows its drawn covariate etas, not its own covariate values.{rows} "
    )
}

const SIMULATE_REMEDY: &str = "To keep each subject's PK with its own covariates, set \
     `SimulateOptions::match_method`, which reassigns the drawn etas by matching them to each \
     subject's posthoc eta (approximately), or simulate the fitted subjects from the FREM \
     fit's EBEs with `simulate_adaptive_with_inputs` and `SuppliedEtas`.";
const UNCERTAINTY_REMEDY: &str = "`simulate_with_uncertainty` has no matching option: to keep \
     each subject's PK with its own covariates, use `simulate_with_options` with \
     `SimulateOptions::match_method`, or `simulate_adaptive_with_inputs` with the FREM fit's \
     EBEs as `SuppliedEtas`.";
const ADAPTIVE_REMEDY: &str = "A controller reading `ControllerCtx::covariates` doses these \
     subjects on covariate values their PK does not follow. Supply the FREM fit's EBEs as \
     `SuppliedEtas` (`AdaptiveInputs::with_etas`), which are conditioned on each subject's \
     covariate rows.";

fn drawn_warnings(warnings: &[String]) -> Vec<&String> {
    warnings
        .iter()
        .filter(|w| w.starts_with(DRAWN_CODE))
        .collect()
}

/// A FREM fixture subject with no `FREMTYPE` column at all.
fn subject_without_frem_rows(id: &str) -> Subject {
    let mut s = subject(id, [1; 6], vec![1u32; OBS.len()]);
    s.fremtype = Vec::new();
    s
}

/// The helper's gates, both sides of each in one test: FREM vs not, a drawn subject vs
/// none, and the FREMTYPE sentence on a subject with covariate rows vs one without.
///
/// Regressions: drop the `frem_config` gate → the non-FREM case warns; drop the
/// `n_drawn == 0` arm → the empty case warns; make the FREMTYPE sentence unconditional
/// (or drop it) → one of the two text assertions fails.
#[test]
fn frem_drawn_eta_warning_gates_and_text() {
    let m = model("  obs_scale = V", false, [1; 6]);
    let with_rows = subject("1", [1; 6], vec![1u32; OBS.len()]);
    let without_rows = subject_without_frem_rows("2");

    let got = crate::api::frem_drawn_eta_warning(&m, 3, [&with_rows, &without_rows], "R.");
    assert_eq!(
        got.as_deref(),
        Some(format!("{}R.", drawn_head(2, 3, true)).as_str())
    );
    let got = crate::api::frem_drawn_eta_warning(&m, 3, [&without_rows], "R.");
    assert_eq!(
        got.as_deref(),
        Some(format!("{}R.", drawn_head(1, 3, false)).as_str())
    );

    assert_eq!(
        crate::api::frem_drawn_eta_warning(&m, 3, std::iter::empty(), "R."),
        None,
        "no drawn subject: silent"
    );
    let mut plain = model("  obs_scale = V", false, [1; 6]);
    plain.frem_config = None;
    assert_eq!(
        crate::api::frem_drawn_eta_warning(&plain, 3, [&with_rows], "R."),
        None,
        "a non-FREM model: silent"
    );

    // Classified as a simulation finding, not by a prose arm keyed on "omega".
    let w = crate::types::classify_warning(&drawn_head(1, 1, true));
    assert_eq!(w.category.as_str(), "simulation");
    assert_eq!(w.severity, crate::types::WarningSeverity::Warning);
}

/// `simulate_with_options_diag` warns on its drawn branch, with its own remedy, and is
/// silent under `match_method`, the route that remedy names.
///
/// Regressions: drop the call → the drawn case has no warning; move it below the branch
/// split so it also runs on the matched branch → the matched case warns.
#[test]
fn simulate_with_options_diag_reports_a_drawn_frem_eta() {
    let m = model("  obs_scale = V", false, [1; 6]);
    let mut pop = population([1; 6]);
    // Finite, positive observations so the matched branch's posthoc pass has data.
    pop.subjects[0].observations = vec![70.0, 1.8, 1.0, 0.2, 70.0, 0.1];
    let drawn = simulate_with_options_diag(
        &m,
        &pop,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .expect("simulate");
    let want = format!("{}{SIMULATE_REMEDY}", drawn_head(1, 1, true));
    assert_eq!(drawn_warnings(&drawn.warnings), vec![&want]);

    let matched = simulate_with_options_diag(
        &m,
        &pop,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            match_method: Some(crate::propensity_match::MatchMethod::Optimal),
            ..Default::default()
        },
    )
    .expect("matched simulate");
    assert!(
        drawn_warnings(&matched.warnings).is_empty(),
        "propensity matching is the named remedy: {:?}",
        matched.warnings
    );
}

/// `simulate_with_uncertainty_diag` always draws η, so it always warns on a FREM model,
/// with a remedy that does not point at an option it lacks.
///
/// Regression: drop the call → no warning.
#[test]
fn simulate_with_uncertainty_diag_reports_a_drawn_frem_eta() {
    let m = model("  obs_scale = V", false, [1; 6]);
    let pop = population([1; 6]);
    let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
    let out = simulate_with_uncertainty_diag(
        &m,
        &pop,
        &fit,
        &SimulateUncertaintyOptions {
            n_uncertainty_draws: 2,
            n_sim_per_draw: 1,
            method: crate::estimation::uncertainty_samples::UncertaintyMethod::Asymptotic,
            seed: Some(7),
        },
    )
    .expect("uncertainty");
    let want = format!("{}{UNCERTAINTY_REMEDY}", drawn_head(1, 1, true));
    assert_eq!(drawn_warnings(&out.warnings), vec![&want]);
}

/// The adaptive path counts only the subjects whose η was drawn: all of them with no
/// supplied η, none when every subject is supplied, and the unmatched one when the run
/// mixes supplied and drawn subjects (`UnmatchedSubjects::Draw`).
///
/// Regressions: drop the call → the drawn case has no warning; drop the `is_none` filter
/// → the all-supplied case warns and the mixed case says 2 of 2.
#[test]
fn adaptive_reports_only_the_subjects_whose_frem_eta_was_drawn() {
    let m = model("  obs_scale = V", false, [1; 6]);
    let pop = population([1; 6]);
    let drawn = simulate_adaptive(&m, &pop, &m.default_params, 1, hold, &opts()).expect("drawn");
    let want = format!("{}{ADAPTIVE_REMEDY}", drawn_head(1, 1, true));
    assert_eq!(drawn_warnings(&drawn.warnings), vec![&want]);

    let e = SuppliedEtas::from_table(
        vec!["1".to_string()],
        vec!["ETA_V".to_string(), "ETA_WT".to_string()],
        vec![vec![0.2, 11.5]],
    )
    .expect("valid table");
    let inputs = AdaptiveInputs::new(&m.default_params).with_etas(&e);
    let supplied =
        simulate_adaptive_with_inputs(&m, &pop, &inputs, 1, hold, &opts()).expect("supplied");
    assert!(
        drawn_warnings(&supplied.warnings).is_empty(),
        "every η supplied: {:?}",
        supplied.warnings
    );

    // Subject "2" has no supplied η and no FREMTYPE rows; "1" is supplied and has them,
    // so the FREMTYPE sentence must also follow the drawn subjects only.
    let mut mixed_pop = population([1; 6]);
    mixed_pop.subjects.push(subject_without_frem_rows("2"));
    let inputs = AdaptiveInputs::new(&m.default_params)
        .with_etas(&e)
        .with_unmatched(UnmatchedSubjects::Draw);
    let mixed =
        simulate_adaptive_with_inputs(&m, &mixed_pop, &inputs, 1, hold, &opts()).expect("mixed");
    let want = format!("{}{ADAPTIVE_REMEDY}", drawn_head(1, 2, false));
    assert_eq!(drawn_warnings(&mixed.warnings), vec![&want]);
}
