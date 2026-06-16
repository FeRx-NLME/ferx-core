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

use ferx_core::api::check_model_data;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv, CompiledModel, Population, Severity};
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
// With CL=5, V=50 (k=0.1), AMT=100, D1=5 → R=20. The committed control file
// `tests/nonmem/modeled_duration.ctl` (ADVAN1 TRANS2, `$PK D1=THETA(3)`,
// `RATE=-2` in the data) reproduces this to ADVAN1's precision; running it is a
// follow-up (no NONMEM in CI), so the closed form is the in-test reference.
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

#[test]
fn modeled_duration_matches_nonmem_closed_form() {
    let model = model_of(ODE_D1);
    let population = read_nonmem_csv(Path::new("data/modeled_duration_ref.csv"), None, None)
        .expect("anchor dataset loads");
    let preds = predict(&model, &population, &model.default_params);
    assert!(!preds.is_empty(), "anchor dataset must yield predictions");
    for p in &preds {
        let expected = one_cpt_infusion_closed_form(p.time);
        let rel = (p.pred - expected).abs() / expected.max(1e-12);
        assert!(
            rel < 1e-4,
            "t={}: ferx ODE PRED {:.6} vs ADVAN1 closed form {:.6} (rel {:.2e})",
            p.time,
            p.pred,
            expected,
            rel
        );
    }
}
