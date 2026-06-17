//! Tier-2 integration tests for **modeled infusion duration** (`RATE=-2` →
//! `D{cmt}`) on ODE models (#324).
//!
//! NONMEM's `RATE=-2` makes the infusion *duration* a `$PK` parameter `D{n}`
//! (the rate is then `AMT / D{n}`). These tests exercise the public
//! `predict()` / `check_model_data()` boundaries and assert:
//!
//!   * the **core invariant**: a `RATE=-2` dose with `D{cmt} = d` is identical to
//!     an explicit `RATE = AMT/d` infusion (the cleanest correctness proof);
//!   * **composition** with bioavailability `F{cmt}` (applied exactly once — no
//!     double-counting) and with absorption lag `ALAG{cmt}` (shifts the window,
//!     `D` sets its length);
//!   * **steady state** (`SS=1`) equilibrates with the modeled duration;
//!   * **per-compartment** binding (`D1` vs `D2`);
//!   * **loud rejection** of the unsupported / misconfigured cases (no `D{cmt}`
//!     parameter; `RATE=-2` on an analytical model);
//!   * a **NONMEM-anchored closed form** for a one-compartment infusion.
//!
//! All return immediately (`predict` with fixed params / a `check_model_data`
//! pass — no convergence loop), so they need no `slow-tests` gate.

use ferx_core::api::{check_model_data, check_model_data_warnings};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{
    predict, read_nonmem_csv, simulate, simulate_with_options, CompiledModel, Population, Severity,
    SimulateOptions,
};
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

/// One-compartment IV model whose infusion *duration* is the modeled parameter
/// `D1` (NONMEM `RATE=-2`). `central` is an amount; the read-out is `central/V`
/// (Form-C scaling), so an infusion into CMT=1 injects `rate = AMT/D1`. `D1`
/// defaults to 5.0 → with `AMT=100` the rate is 20 over a 5 h window.
const ODE_D1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

/// `ODE_D1` plus per-compartment bioavailability `F1 = 0.5`: the modeled-duration
/// infusion must deliver `F1 * AMT` over `D1` (F applied exactly once).
const ODE_D1_F1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVF1(0.5, 0.01, 1.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1
  F1 = TVF1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

/// `ODE_D1` plus absorption lag `ALAG1 = 2`: the infusion window starts at
/// `time + 2` and runs for `D1`.
const ODE_D1_LAG1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(5.0, 0.1, 24.0)
  theta TVLAG1(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  D1    = TVD1
  ALAG1 = TVLAG1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

/// `ODE_D1` but the modeled duration `D1` is non-positive at the initial typical
/// value (`TVD1 = -1` with a `D1 = TVD1` identity link). `check_model_data` still
/// accepts it (the `D1` parameter exists on an ODE model), but
/// `check_model_data_warnings` must flag `W_MODELED_DURATION_NONPOSITIVE` (#324
/// review #3): a `D ≤ 0` is clamped to a near-bolus spike, so the fit can converge
/// wrong with no other diagnostic.
const ODE_D1_NEG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(-1.0, -10.0, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

fn write_csv(contents: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("create temp csv");
    f.write_all(contents.as_bytes()).expect("write temp csv");
    f.flush().expect("flush temp csv");
    f
}

fn model_of(src: &str) -> CompiledModel {
    parse_full_model(src).expect("model parses").model
}

fn pop_of(csv: &str) -> Population {
    let f = write_csv(csv);
    read_nonmem_csv(f.path(), None, None).expect("dataset loads")
}

/// Predicted values for a CSV dataset under `model` at its default parameters.
fn preds_of(model: &CompiledModel, csv: &str) -> Vec<f64> {
    let pop = pop_of(csv);
    predict(model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{ctx}: row {i}: {x} vs {y} (|Δ| {:.3e} > {tol:.0e})",
            (x - y).abs()
        );
    }
}

// Observation grid spanning the 5 h infusion and the decay tail. DV is a
// placeholder (predict() recomputes the prediction); rows are observations
// (EVID=0, MDV=0, AMT=0) on the observed compartment (CMT=1).
const OBS_ROWS: &str = "1,1,0,0,0,1,0,0\n\
                        1,3,0,0,0,1,0,0\n\
                        1,5,0,0,0,1,0,0\n\
                        1,8,0,0,0,1,0,0\n\
                        1,12,0,0,0,1,0,0\n\
                        1,18,0,0,0,1,0,0\n\
                        1,24,0,0,0,1,0,0\n";

fn coded_csv() -> String {
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,-2,1\n{OBS_ROWS}")
}

fn explicit_csv() -> String {
    // RATE = AMT / D1 = 100 / 5 = 20 (the concrete infusion D1=5 resolves to).
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,20,1\n{OBS_ROWS}")
}

#[test]
fn modeled_duration_matches_explicit_infusion() {
    // Core #324 invariant: `RATE=-2` with `D1=5` is bit-equal to an explicit
    // `RATE = AMT/5 = 20` infusion. A regression in resolve/threading would make
    // these diverge.
    let model = model_of(ODE_D1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(&coded, &explicit, 1e-9, "RATE=-2 D1=5 vs explicit RATE=20");
    // And the predictions are non-trivial (a plateau-then-decay infusion, not all
    // zero) — guards against "both happen to be empty/zero".
    assert!(
        coded.iter().any(|&c| c > 0.1),
        "predictions should be nonzero"
    );
}

#[test]
fn modeled_duration_composes_with_bioavailability_once() {
    // F1 must scale the resolved rate exactly ONCE: `RATE=-2` (D1=5) with F1=0.5
    // equals explicit `RATE=20` with the same F1=0.5. A double-application of F
    // in `resolve_rate` would scale the coded case by F again (0.25 vs 0.5) and
    // the two would diverge by a factor of F.
    let model = model_of(ODE_D1_F1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(&coded, &explicit, 1e-9, "F1 + RATE=-2 vs F1 + explicit");
    // Sanity: F1=0.5 halves exposure vs the no-F model (so F is actually applied).
    let no_f = preds_of(&model_of(ODE_D1), &coded_csv());
    assert!(
        coded[2] < 0.75 * no_f[2],
        "F1=0.5 must reduce exposure: {} vs {}",
        coded[2],
        no_f[2]
    );
}

#[test]
fn modeled_duration_composes_with_lagtime() {
    // ALAG1 shifts the infusion window start; D1 sets its length. `RATE=-2`
    // (D1=5) + ALAG1=2 equals explicit `RATE=20` + ALAG1=2.
    let model = model_of(ODE_D1_LAG1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(
        &coded,
        &explicit,
        1e-9,
        "ALAG1 + RATE=-2 vs ALAG1 + explicit",
    );
    // The lag delays uptake: at t=1 (< lag 2) the central compartment is still
    // empty, unlike the no-lag model where the infusion is already running.
    let no_lag = preds_of(&model_of(ODE_D1), &coded_csv());
    assert!(
        coded[0] < 1e-9,
        "pre-lag prediction must be ~0, got {}",
        coded[0]
    );
    assert!(no_lag[0] > 1e-3, "no-lag model should have uptake by t=1");
}

#[test]
fn modeled_duration_steady_state_matches_explicit() {
    // SS=1 equilibration must use the resolved duration: a steady-state `RATE=-2`
    // (D1=5, II=12) infusion equals the explicit `RATE=20` SS infusion.
    let coded = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS\n\
                 1,0,.,1,100,1,-2,1,12,1\n\
                 1,1,0,0,0,1,0,0,0,0\n\
                 1,6,0,0,0,1,0,0,0,0\n\
                 1,11,0,0,0,1,0,0,0,0\n";
    let explicit = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS\n\
                    1,0,.,1,100,1,20,1,12,1\n\
                    1,1,0,0,0,1,0,0,0,0\n\
                    1,6,0,0,0,1,0,0,0,0\n\
                    1,11,0,0,0,1,0,0,0,0\n";
    let model = model_of(ODE_D1);
    assert_close(
        &preds_of(&model, coded),
        &preds_of(&model, explicit),
        1e-9,
        "SS RATE=-2 vs SS explicit",
    );
}

#[test]
fn modeled_duration_with_reset_matches_explicit() {
    // A system reset (EVID=3) forces the subject onto the *event-driven* ODE path
    // (per-dose resolution), distinct from the plain segment loop the other tests
    // hit. The RATE=-2 / explicit invariant must hold there too — with a modeled
    // dose both before and after the reset.
    let coded = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                 1,0,.,1,100,1,-2,1\n\
                 1,2,0,0,0,1,0,0\n\
                 1,5,.,3,.,1,.,1\n\
                 1,6,.,1,100,1,-2,1\n\
                 1,8,0,0,0,1,0,0\n\
                 1,12,0,0,0,1,0,0\n";
    let explicit = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
                    1,0,.,1,100,1,20,1\n\
                    1,2,0,0,0,1,0,0\n\
                    1,5,.,3,.,1,.,1\n\
                    1,6,.,1,100,1,20,1\n\
                    1,8,0,0,0,1,0,0\n\
                    1,12,0,0,0,1,0,0\n";
    let model = model_of(ODE_D1);
    let coded_p = preds_of(&model, coded);
    assert_close(
        &coded_p,
        &preds_of(&model, explicit),
        1e-9,
        "reset: RATE=-2 vs explicit",
    );
    // Post-reset uptake is nonzero (the t=8 sample is mid second infusion).
    assert!(
        coded_p.last().is_some_and(|&c| c > 0.01),
        "post-reset uptake expected"
    );
}

#[test]
fn modeled_duration_resolves_per_compartment() {
    // D1 and D2 bind independently: a 2-compartment model dosed RATE=-2 into
    // CMT=1 uses D1, and into CMT=2 uses D2. With different D1/D2 the two single-
    // dose runs must differ, and each must match its explicit-RATE equivalent.
    let two_cmt = r#"
[parameters]
  theta TVK(0.1, 0.001, 5.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVD1(4.0, 0.1, 24.0)
  theta TVD2(8.0, 0.1, 24.0)
  omega ETA ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  K  = TVK * exp(ETA)
  V  = TVV
  D1 = TVD1
  D2 = TVD2

[structural_model]
  ode(states=[a, b])

[odes]
  d/dt(a) = -K * a
  d/dt(b) = -K * b

[scaling]
  y = a + b

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = model_of(two_cmt);
    // Dose into CMT=1 (D1=4) -> explicit RATE = 100/4 = 25.
    let coded1 =
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,-2,1\n1,2,0,0,0,1,0,0\n1,6,0,0,0,1,0,0\n";
    let expl1 =
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,25,1\n1,2,0,0,0,1,0,0\n1,6,0,0,0,1,0,0\n";
    // Dose into CMT=2 (D2=8) -> explicit RATE = 100/8 = 12.5.
    let coded2 =
        "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,2,-2,1\n1,2,0,0,0,1,0,0\n1,6,0,0,0,1,0,0\n";
    let expl2 = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,2,12.5,1\n1,2,0,0,0,1,0,0\n1,6,0,0,0,1,0,0\n";

    assert_close(
        &preds_of(&model, coded1),
        &preds_of(&model, expl1),
        1e-9,
        "CMT=1 -> D1",
    );
    assert_close(
        &preds_of(&model, coded2),
        &preds_of(&model, expl2),
        1e-9,
        "CMT=2 -> D2",
    );
    // D1 != D2 so the two compartments' single-dose curves differ.
    assert!(
        (preds_of(&model, coded1)[0] - preds_of(&model, coded2)[0]).abs() > 1e-6,
        "distinct D1/D2 must give distinct predictions"
    );
}

#[test]
fn modeled_duration_without_matching_param_is_rejected() {
    // A `RATE=-2` dose into a compartment with no `D{cmt}` parameter is a loud
    // model+data join error — never a silent fall-through to a bolus.
    let no_d1 = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = model_of(no_d1);
    let pop = pop_of(&coded_csv());
    let diags = check_model_data(&model, &pop);
    let d = diags
        .iter()
        .find(|d| d.code == "E_MODELED_DURATION_NO_PARAM")
        .expect("RATE=-2 with no D1 must be rejected");
    assert_eq!(d.severity, Severity::Error);
    assert!(
        d.message.contains("D1") && d.message.contains("compartment 1"),
        "{}",
        d.message
    );
}

#[test]
fn modeled_duration_on_analytical_model_is_rejected() {
    // Modeled duration is ODE-only in this release; a `RATE=-2` dose on an
    // analytical model is rejected with a pointer to the follow-up, not silently
    // mis-modeled.
    let analytical = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;
    let model = model_of(analytical);
    assert!(model.ode_spec.is_none(), "model must be analytical");
    let pop = pop_of(&coded_csv());
    let diags = check_model_data(&model, &pop);
    let d = diags
        .iter()
        .find(|d| d.code == "E_MODELED_DURATION_ANALYTICAL")
        .expect("RATE=-2 on an analytical model must be rejected");
    assert_eq!(d.severity, Severity::Error);
    assert!(
        d.message.contains("ODE") && d.message.contains("#324"),
        "{}",
        d.message
    );
}

// ── NONMEM-anchored closed form ─────────────────────────────────────────────
//
// For a one-compartment IV infusion of rate `R = AMT/D1` over `T = D1` into a
// compartment with elimination `k = CL/V`, the concentration is the exact
// ADVAN1 solution NONMEM computes:
//   t <= T:  C(t) = R/(V·k) · (1 − e^{−k t})
//   t  > T:  C(t) = C(T) · e^{−k (t−T)}
// With CL=5, V=50 (k=0.1), AMT=100, D1=5 → R=20.
//
// NONMEM run: `nmfe75 modeled_duration.ctl modeled_duration.lst`
// (ADVAN1 TRANS2, `$PK D1=THETA(3)=5 FIX`, MAXEVAL=0, η=0 → IPRED=PRED).
// NONMEM IPRED values from sdtab1 (S1PE11.4):
//   t=0.5:  1.9508E-01
//   t=1.0:  3.8065E-01
//   t=2.0:  7.2508E-01
//   t=5.0:  1.5739E+00
//   t=8.0:  1.1660E+00
//   t=12.0: 7.8156E-01
//   t=18.0: 4.2893E-01
//   t=24.0: 2.3540E-01
// These agree with the closed form to 5 s.f. (NONMEM's output precision).
// The committed control file is `tests/nonmem/modeled_duration.ctl`.
fn one_cpt_infusion_closed_form(t: f64) -> f64 {
    let (cl, v, amt, d1) = (5.0_f64, 50.0_f64, 100.0_f64, 5.0_f64);
    let k = cl / v;
    let r = amt / d1;
    let plateau = r / (v * k);
    if t <= d1 {
        plateau * (1.0 - (-k * t).exp())
    } else {
        plateau * (1.0 - (-k * d1).exp()) * (-k * (t - d1)).exp()
    }
}

// One-compartment analytical (non-ODE) model: modeled duration is unsupported
// here, so a `RATE=-2` dose must be rejected at the public boundaries too.
const ANALYTICAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;

// ODE model with NO `D1` parameter — a `RATE=-2` dose into CMT=1 has no slot to
// resolve against, the join error `E_MODELED_DURATION_NO_PARAM`.
const ODE_NO_D1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

#[test]
#[should_panic(expected = "model cannot honour")]
fn predict_on_analytical_model_with_modeled_dose_panics() {
    // `predict()` runs no `check_model_data`, so before #384's entrypoint guard a
    // RATE=-2 dose on an analytical model reached the predictor and silently
    // degraded to a 0-rate "infusion" in release (the `debug_assert` is a no-op).
    // The guard now turns it into a loud panic carrying the diagnostic.
    let model = model_of(ANALYTICAL);
    assert!(model.ode_spec.is_none(), "model must be analytical");
    let pop = pop_of(&coded_csv());
    let _ = predict(&model, &pop, &model.default_params);
}

#[test]
#[should_panic(expected = "model cannot honour")]
fn predict_on_ode_missing_param_panics() {
    // RATE=-2 into a compartment with no `D{cmt}` would hit `resolve_rate`'s
    // slot `.expect` deep in the ODE path; the entrypoint guard intercepts it
    // first with the actionable `E_MODELED_DURATION_NO_PARAM` message.
    let model = model_of(ODE_NO_D1);
    let pop = pop_of(&coded_csv());
    let _ = predict(&model, &pop, &model.default_params);
}

#[test]
#[should_panic(expected = "model cannot honour")]
fn simulate_on_analytical_model_with_modeled_dose_panics() {
    // The same guard covers every `simulate*` variant via the shared
    // `simulate_inner_with_draw` chokepoint.
    let model = model_of(ANALYTICAL);
    let pop = pop_of(&coded_csv());
    let _ = simulate(&model, &pop, &model.default_params, 1);
}

#[test]
#[should_panic(expected = "model cannot honour")]
fn simulate_propensity_on_analytical_model_with_modeled_dose_panics() {
    // Review #1: the propensity-match branch of `simulate_with_options` runs a
    // full inner EBE pass (`run_inner_loop_warm`) — integrating every subject —
    // BEFORE control reaches the `simulate_inner_with_draw` chokepoint guard. On
    // an unsupported config that pass would degrade silently (analytical, release)
    // or hit an opaque `.expect` first. The guard now also runs at the top of
    // `simulate_with_options`, so the propensity path fails fast with the same
    // actionable diagnostic as every other entrypoint.
    let model = model_of(ANALYTICAL);
    assert!(model.ode_spec.is_none(), "model must be analytical");
    let pop = pop_of(&coded_csv());
    let opts = SimulateOptions {
        seed: Some(1),
        propensity_match: true,
    };
    let _ = simulate_with_options(&model, &pop, &model.default_params, 1, &opts);
}

#[test]
fn valid_modeled_dose_predicts_without_panicking() {
    // The guard is a no-op on a supported config: a RATE=-2 dose on an ODE model
    // with the matching `D1` predicts normally (the all-`Fixed` Ok path of the
    // entrypoint guard, and a regression guard that the guard isn't over-eager).
    let model = model_of(ODE_D1);
    let preds = predict(&model, &pop_of(&coded_csv()), &model.default_params);
    assert!(preds.iter().any(|p| p.pred > 0.1), "expected real uptake");
}

#[test]
fn modeled_duration_steady_state_overlap_warns() {
    // W_STEADY_STATE_INFUSION (T_inf > II) must fire for a *modeled* SS infusion
    // too: the warning's effective-duration check resolves `D{cmt}` at init
    // params (#384), since `dose.duration` is 0 until `resolve_rate`. Here
    // D1=5 > II=4, so the overlapping-pulse warning is expected.
    let model = model_of(ODE_D1);
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS\n\
               1,0,.,1,100,1,-2,1,4,1\n\
               1,2,0,0,0,1,0,0,0,0\n";
    let pop = pop_of(csv);
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        diags.iter().any(|d| d.code == "W_STEADY_STATE_INFUSION"),
        "modeled SS infusion with D1=5 > II=4 must warn; got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn modeled_duration_nonpositive_at_init_warns() {
    // Review #3: a modeled duration `D1` that is ≤ 0 at the initial typical value
    // (here TVD1 = -1 via a `D1 = TVD1` identity link) is accepted by
    // `check_model_data` (the parameter exists) but flagged by
    // `check_model_data_warnings` — `resolve_rate` would clamp it to a near-bolus
    // spike, so the fit can converge wrong with no other diagnostic.
    let model = model_of(ODE_D1_NEG);
    let pop = pop_of(&coded_csv());
    // No hard error — the parameter exists on an ODE model.
    assert!(
        check_model_data(&model, &pop)
            .iter()
            .all(|d| d.severity != Severity::Error),
        "non-positive D is a warning, not an error"
    );
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        diags
            .iter()
            .any(|d| d.code == "W_MODELED_DURATION_NONPOSITIVE"),
        "TVD1=-1 must warn; got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn modeled_duration_positive_at_init_does_not_warn() {
    // Converse: the default `ODE_D1` (TVD1 = 5 > 0) must NOT raise the
    // non-positive-duration warning — a regression guard against a false positive.
    let model = model_of(ODE_D1);
    let pop = pop_of(&coded_csv());
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        !diags
            .iter()
            .any(|d| d.code == "W_MODELED_DURATION_NONPOSITIVE"),
        "TVD1=5 must not warn; got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn modeled_duration_warnings_are_panic_free_when_param_missing() {
    // An ODE model with a modeled SS dose but no matching `D{cmt}` is a hard
    // error from `check_model_data` (E_MODELED_DURATION_NO_PARAM), but the
    // separate `check_model_data_warnings` pass can still be called on it and
    // must stay panic-free — it must NOT hit `resolve_rate`'s slot `.expect`.
    // This exercises the slot-absent fallbacks in both modeled-dose warnings:
    // the SS-overlap check's `effective_duration` (`_ => 0.0`) and the
    // non-positive-duration loop's `indexed_slot(..) == None` skip. With no slot,
    // the effective duration is 0 (no overlap) and neither warning fires.
    let model = model_of(ODE_NO_D1);
    assert!(model.ode_spec.is_some(), "model must be ODE");
    // The data is genuinely unsupported (the join would reject it)…
    assert!(
        check_model_data(&model, &pop_of(&coded_csv()))
            .iter()
            .any(|d| d.code == "E_MODELED_DURATION_NO_PARAM"),
        "missing D{{cmt}} must be a join error"
    );
    // …but the warnings pass over an SS modeled dose must not panic.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS\n\
               1,0,.,1,100,1,-2,1,4,1\n\
               1,2,0,0,0,1,0,0,0,0\n";
    let diags = check_model_data_warnings(&model, &pop_of(csv), &model.default_params);
    assert!(
        !diags
            .iter()
            .any(|d| d.code == "W_STEADY_STATE_INFUSION"
                || d.code == "W_MODELED_DURATION_NONPOSITIVE"),
        "slot-absent modeled dose raises no modeled-dose warning; got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn modeled_duration_steady_state_no_overlap_does_not_warn() {
    // Converse: D1=5 <= II=6 is a non-overlapping SS infusion — no warning. This
    // pins that the effective-duration resolution compares the *resolved* D, not
    // the unresolved 0 (which would never warn) nor a false positive.
    let model = model_of(ODE_D1);
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS\n\
               1,0,.,1,100,1,-2,1,6,1\n\
               1,2,0,0,0,1,0,0,0,0\n";
    let pop = pop_of(csv);
    let diags = check_model_data_warnings(&model, &pop, &model.default_params);
    assert!(
        !diags.iter().any(|d| d.code == "W_STEADY_STATE_INFUSION"),
        "D1=5 <= II=6 must not warn; got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn modeled_duration_matches_nonmem_closed_form() {
    // NONMEM IPRED values from sdtab1 (nmfe75 run, MAXEVAL=0, η=0 → IPRED=PRED).
    // Times match data/modeled_duration_ref.csv observation rows.
    let nonmem_ipred: &[(f64, f64)] = &[
        (0.5, 1.9508e-1),
        (1.0, 3.8065e-1),
        (2.0, 7.2508e-1),
        (5.0, 1.5739e0),
        (8.0, 1.1660e0),
        (12.0, 7.8156e-1),
        (18.0, 4.2893e-1),
        (24.0, 2.3540e-1),
    ];

    let model = model_of(ODE_D1);
    let population = read_nonmem_csv(Path::new("data/modeled_duration_ref.csv"), None, None)
        .expect("anchor dataset loads");
    let preds = predict(&model, &population, &model.default_params);
    assert_eq!(preds.len(), nonmem_ipred.len(), "prediction count mismatch");

    for (p, &(t_ref, nm)) in preds.iter().zip(nonmem_ipred) {
        assert!(
            (p.time - t_ref).abs() < 1e-9,
            "time mismatch: got {}, expected {}",
            p.time,
            t_ref
        );
        // Compare against NONMEM IPRED (5 s.f. precision from sdtab1).
        let rel_nm = (p.pred - nm).abs() / nm.max(1e-12);
        assert!(
            rel_nm < 1e-4,
            "t={t_ref}: ferx {:.6} vs NONMEM IPRED {:.6} (rel {:.2e})",
            p.pred,
            nm,
            rel_nm
        );
        // Also compare against closed form (exact; agreement tighter than NONMEM's
        // 5 s.f. output).
        let cf = one_cpt_infusion_closed_form(p.time);
        let rel_cf = (p.pred - cf).abs() / cf.max(1e-12);
        assert!(
            rel_cf < 1e-4,
            "t={t_ref}: ferx {:.6} vs closed form {:.6} (rel {:.2e})",
            p.pred,
            cf,
            rel_cf
        );
    }
}
