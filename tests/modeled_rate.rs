//! Tier-2 integration tests for **modeled infusion rate** (`RATE=-1` → `R{cmt}`)
//! on ODE and analytical PK models (#324).
//!
//! NONMEM's `RATE=-1` makes the infusion *rate* a `$PK` parameter `R{n}` (the
//! duration is then `AMT / R{n}`). `resolve_rate` turns a `RATE=-1 R=r` dose into
//! exactly the explicit `RATE=r` infusion, so these tests assert:
//!
//!   * the **core invariant**: a `RATE=-1` dose with `R{cmt} = r` is identical to
//!     an explicit `RATE = r` infusion (the cleanest correctness proof);
//!   * **composition** with bioavailability `F{cmt}` — applied exactly once;
//!   * the **F!=1 behaviour** (see below): under `F`, `RATE=-1 R=r` holds the rate
//!     and scales the *duration* to `F·AMT/r` (NONMEM-faithful, #419), so it now
//!     DIFFERS from the `RATE=-2 D=AMT/r` dose (which holds the duration and scales
//!     the rate);
//!   * **loud rejection** of the misconfigured case (`RATE=-1` with no matching
//!     `R{cmt}` — on either engine);
//!   * the **analytical engine** honours `RATE=-1` identically (coded vs explicit
//!     `RATE = r`, plus the NONMEM-anchored closed form), given an `R{cmt}`;
//!   * the **non-positive-rate warning** at the typical-value point.
//!
//! ## F!=1 and NONMEM faithfulness (read before changing the F tests)
//!
//! ferx applies bioavailability to an infusion the NONMEM way (#419): for a
//! *rate-defined* infusion (`RATE>0` data **and** `RATE=-1`) it holds the rate at
//! `R` and scales the **duration** to `F·AMT/R`; for a *duration-defined* infusion
//! (`RATE=-2`) it holds the duration at `D` and scales the **rate** to `F·AMT/D`.
//! Total exposure is `F·AMT` either way; the infusion *shape* differs between the
//! two modes only when `F != 1`. So under `F != 1` a `RATE=-1 R=r` dose now
//! DIFFERS from the `RATE=-2 D=AMT/r` dose (it used to match, when ferx scaled the
//! rate for both - the bug #419 fixed). `modeled_rate_under_f_scales_duration`
//! anchors the rate-defined case against the duration-scaled ADVAN1 closed form
//! (= NONMEM IPRED) and pins the divergence from `RATE=-2`. At `F=1` both modes
//! coincide, anchored by `analytical_modeled_rate_matches_nonmem_closed_form`.
//!
//! All return immediately (`predict` with fixed params / a `check_model_data`
//! pass — no convergence loop), so they need no `slow-tests` gate.

use ferx_core::api::{check_model_data, check_model_data_warnings};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv, CompiledModel, Population, Severity};
use std::io::Write;
use tempfile::NamedTempFile;

/// One-compartment IV model whose infusion *rate* is the modeled parameter `R1`
/// (NONMEM `RATE=-1`). `central` is an amount; the read-out is `central/V`, so an
/// infusion into CMT=1 runs at `rate = R1`. `R1` defaults to 20.0 → with `AMT=100`
/// the duration is `AMT/R1 = 5` h.
const ODE_R1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 100.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

/// `ODE_R1` plus per-compartment bioavailability `F1 = 0.5`.
const ODE_R1_F1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 100.0)
  theta TVF1(0.5, 0.01, 1.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1
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

/// `ODE` one-compartment model with `RATE=-2 → D1` and `F1=0.5`, used to pin that
/// (under `F`) `RATE=-1 R1=20` is identical to `RATE=-2 D1=5` in ferx (both are
/// the same resolved infusion: rate 20, duration 5, then `F` scales the rate).
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

/// `ODE_R1` plus absorption lag `ALAG1 = 2`: the infusion window starts at
/// `time + 2` and runs for `AMT/R1`.
const ODE_R1_LAG1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 100.0)
  theta TVLAG1(2.0, 0.0, 12.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  R1    = TVR1
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

/// `ODE_R1` but the modeled rate `R1` is non-positive at the initial typical value
/// (`TVR1 = -1` with an `R1 = TVR1` identity link). `check_model_data` accepts it
/// (the `R1` parameter exists), but `check_model_data_warnings` must flag
/// `W_MODELED_RATE_NONPOSITIVE`: a non-positive rate is clamped, delivering the
/// dose as a near-zero trickle, so the fit can converge wrong with no other signal.
const ODE_R1_NEG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(-1.0, -10.0, 10.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
"#;

/// One-compartment **analytical** (closed-form) IV model with a modeled infusion
/// rate `R1`. Same parameters as `ODE_R1` but using `one_cpt_iv` instead of an
/// `ode(...)` block, so `RATE=-1` resolves to `rate = R1` and feeds the analytical
/// infusion solution.
const ANALYTICAL_R1: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVR1(20.0, 0.1, 100.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  R1 = TVR1

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
"#;

/// One-compartment analytical IV model with NO `R1`: a `RATE=-1` dose has no slot
/// to resolve against and must be rejected (`E_MODELED_RATE_NO_PARAM`).
const ANALYTICAL_NO_R1: &str = r#"
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

// Observation grid spanning the 5 h infusion and the decay tail (CMT=1).
const OBS_ROWS: &str = "1,1,0,0,0,1,0,0\n\
                        1,3,0,0,0,1,0,0\n\
                        1,5,0,0,0,1,0,0\n\
                        1,8,0,0,0,1,0,0\n\
                        1,12,0,0,0,1,0,0\n\
                        1,18,0,0,0,1,0,0\n\
                        1,24,0,0,0,1,0,0\n";

fn coded_csv() -> String {
    // RATE=-1 → the rate is the modeled R1.
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,-1,1\n{OBS_ROWS}")
}

fn explicit_csv() -> String {
    // The concrete infusion R1=20 resolves to: an explicit RATE = 20.
    format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,20,1\n{OBS_ROWS}")
}

#[test]
fn modeled_rate_matches_explicit_infusion() {
    // Core #324 invariant: `RATE=-1` with `R1=20` is bit-equal to an explicit
    // `RATE = 20` infusion. A regression in resolve/threading would diverge.
    let model = model_of(ODE_R1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(&coded, &explicit, 1e-9, "RATE=-1 R1=20 vs explicit RATE=20");
    assert!(
        coded.iter().any(|&c| c > 0.1),
        "predictions should be nonzero"
    );
}

#[test]
fn modeled_rate_composes_with_bioavailability_once() {
    // F1 must scale the resolved rate exactly ONCE: `RATE=-1` (R1=20) with F1=0.5
    // equals explicit `RATE=20` with the same F1=0.5. A double-application of F in
    // `resolve_rate` would scale the coded case again (factor of F) and diverge.
    let model = model_of(ODE_R1_F1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(&coded, &explicit, 1e-9, "F1 + RATE=-1 vs F1 + explicit");
    // Sanity: F1=0.5 halves exposure vs the no-F model (so F is actually applied).
    let no_f = preds_of(&model_of(ODE_R1), &coded_csv());
    assert!(
        coded[2] < 0.75 * no_f[2],
        "F1=0.5 must reduce exposure: {} vs {}",
        coded[2],
        no_f[2]
    );
}

#[test]
fn modeled_rate_under_f_scales_duration() {
    // #419: under F1=0.5 the modeled-RATE dose `RATE=-1 R1=20` is rate-defined, so
    // NONMEM (and now ferx) hold the rate at 20 and scale the DURATION to
    // F·AMT/R = 0.5·100/20 = 2.5 h. This is the duration-scaled ADVAN1 closed form
    // (= NONMEM IPRED) and DIFFERS from the duration-defined `RATE=-2 D1=5` dose
    // (which holds D=5 h and scales the rate to F·AMT/D = 10). Total exposure
    // F·AMT is identical; only the infusion shape differs (it used to match, when
    // ferx scaled the rate for both - the bug #419 fixed).
    let rate_model = model_of(ODE_R1_F1);
    let rate_preds = preds_of(&rate_model, &coded_csv());

    // (a) Anchor against the duration-scaled closed form. CL=5, V=50 (k=0.1),
    //     AMT=100, R1=20, F1=0.5 -> rate 20 over T_F = 2.5 h, plateau R/(V·k)=4.
    let (cl, v, amt, r1, f1) = (5.0_f64, 50.0, 100.0, 20.0, 0.5);
    let k = cl / v;
    let t_f = f1 * amt / r1; // F-scaled duration
    let plateau = r1 / (v * k);
    let times = [1.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0];
    let expected: Vec<f64> = times
        .iter()
        .map(|&t| {
            if t <= t_f {
                plateau * (1.0 - (-k * t).exp())
            } else {
                plateau * (1.0 - (-k * t_f).exp()) * (-k * (t - t_f)).exp()
            }
        })
        .collect();
    // ODE-engine vs the exact closed form: 1e-4 covers the solver tolerance and
    // still decisively rejects the old rate-scaling (which differs by >0.1 here).
    assert_close(
        &rate_preds,
        &expected,
        1e-4,
        "RATE=-1 R1=20 + F1=0.5 holds rate, scales duration to 2.5 h (NONMEM)",
    );

    // (b) Must now DIFFER from the duration-defined RATE=-2 dose under the same F.
    let dur_model = model_of(ODE_D1_F1);
    let dur_coded = format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n1,0,.,1,100,1,-2,1\n{OBS_ROWS}");
    let dur_preds = preds_of(&dur_model, &dur_coded);
    assert!(
        rate_preds
            .iter()
            .zip(dur_preds.iter())
            .any(|(r, d)| (r - d).abs() > 1e-3),
        "RATE=-1 (rate-defined) and RATE=-2 (duration-defined) must differ under F!=1 (#419)",
    );
}

#[test]
fn modeled_rate_composes_with_lagtime() {
    // ALAG1 shifts the infusion window start; R1 sets the rate (duration AMT/R1).
    let model = model_of(ODE_R1_LAG1);
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(
        &coded,
        &explicit,
        1e-9,
        "ALAG1 + RATE=-1 vs ALAG1 + explicit",
    );
    // The lag delays uptake: at t=1 (< lag 2) the compartment is still empty.
    let no_lag = preds_of(&model_of(ODE_R1), &coded_csv());
    assert!(
        coded[0] < 1e-9,
        "pre-lag prediction must be ~0, got {}",
        coded[0]
    );
    assert!(no_lag[0] > 1e-3, "no-lag model should have uptake by t=1");
}

#[test]
fn modeled_rate_without_matching_param_is_rejected() {
    // A `RATE=-1` dose into a compartment with no `R{cmt}` parameter is a loud
    // model+data join error — never a silent fall-through to a bolus.
    let no_r1 = r#"
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
    let model = model_of(no_r1);
    let pop = pop_of(&coded_csv());
    let d = check_model_data(&model, &pop)
        .into_iter()
        .find(|d| d.code == "E_MODELED_RATE_NO_PARAM")
        .expect("RATE=-1 with no R1 must be rejected");
    assert_eq!(d.severity, Severity::Error);
    assert!(
        d.message.contains("R1") && d.message.contains("compartment 1"),
        "{}",
        d.message
    );
}

#[test]
fn analytical_modeled_rate_matches_explicit_infusion() {
    // Core invariant on the ANALYTICAL engine: `RATE=-1` with `R1=20` equals an
    // explicit `RATE=20` infusion through the closed-form `one_cpt_iv` solution —
    // proving the resolution step is wired into the analytical predict path too.
    let model = model_of(ANALYTICAL_R1);
    assert!(model.ode_spec.is_none(), "model must be analytical");
    let coded = preds_of(&model, &coded_csv());
    let explicit = preds_of(&model, &explicit_csv());
    assert_close(
        &coded,
        &explicit,
        1e-9,
        "analytical RATE=-1 R1=20 vs explicit RATE=20",
    );
}

// NONMEM-anchored closed form. For a one-compartment IV infusion of rate `R = R1`
// over `T = AMT/R1` into a compartment with elimination `k = CL/V`, the
// concentration is the exact ADVAN1 solution:
//   t <= T:  C(t) = R/(V·k) · (1 − e^{−k t})
//   t  > T:  C(t) = C(T) · e^{−k (t−T)}
// With CL=5, V=50 (k=0.1), AMT=100, R1=20 → rate 20 over T=5 h. This is the SAME
// physical infusion as the NONMEM-verified RATE=-2 D1=5 anchor in
// `tests/modeled_duration.rs` (NONMEM IPRED from `modeled_duration.lst`), so its
// committed IPRED values apply unchanged. The equivalent control file using
// `$PK R1=20 FIX` with `RATE=-1` data is `tests/nonmem/modeled_rate.ctl`.
fn one_cpt_infusion_closed_form(t: f64) -> f64 {
    let (cl, v, amt, r1) = (5.0_f64, 50.0_f64, 100.0_f64, 20.0_f64);
    let k = cl / v;
    let t_inf = amt / r1;
    let plateau = r1 / (v * k);
    if t <= t_inf {
        plateau * (1.0 - (-k * t).exp())
    } else {
        plateau * (1.0 - (-k * t_inf).exp()) * (-k * (t - t_inf)).exp()
    }
}

#[test]
fn analytical_modeled_rate_matches_nonmem_closed_form() {
    // F=1, so ferx's rate-scaling and NONMEM's duration-scaling coincide: anchor
    // the analytical RATE=-1 predictions against the exact ADVAN1 closed form
    // (= NONMEM IPRED, verified in the RATE=-2 anchor for the identical infusion).
    let model = model_of(ANALYTICAL_R1);
    let pop = pop_of(&coded_csv());
    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect();
    let times = [1.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0];
    let expected: Vec<f64> = times
        .iter()
        .map(|&t| one_cpt_infusion_closed_form(t))
        .collect();
    assert_close(
        &preds,
        &expected,
        1e-6,
        "analytical RATE=-1 vs ADVAN1 closed form",
    );
}

#[test]
fn analytical_modeled_rate_on_model_without_param_is_rejected() {
    // Analytical models support `RATE=-1` only with a matching `R{cmt}`. With no
    // `R1` there is no slot to resolve against, so it is rejected with the same
    // actionable `E_MODELED_RATE_NO_PARAM` error — never a silent bolus.
    let model = model_of(ANALYTICAL_NO_R1);
    assert!(model.ode_spec.is_none(), "model must be analytical");
    let pop = pop_of(&coded_csv());
    let d = check_model_data(&model, &pop)
        .into_iter()
        .find(|d| d.code == "E_MODELED_RATE_NO_PARAM")
        .expect("RATE=-1 with no R1 on an analytical model must be rejected");
    assert_eq!(d.severity, Severity::Error);
    assert!(
        d.message.contains("R1") && d.message.contains("[individual_parameters]"),
        "{}",
        d.message
    );
}

#[test]
fn modeled_rate_nonpositive_at_init_warns() {
    // A modeled rate `R1 ≤ 0` at the initial typical value must raise
    // `W_MODELED_RATE_NONPOSITIVE` (it is clamped to a near-zero trickle).
    let model = model_of(ODE_R1_NEG);
    let pop = pop_of(&coded_csv());
    // Still accepted by the data check (R1 exists); the warning is the signal.
    assert!(
        check_model_data(&model, &pop)
            .iter()
            .all(|d| d.severity != Severity::Error),
        "non-positive R1 is a warning, not a data error"
    );
    assert!(
        check_model_data_warnings(&model, &pop, &model.default_params)
            .iter()
            .any(|d| d.code == "W_MODELED_RATE_NONPOSITIVE"),
        "R1 ≤ 0 at init must warn"
    );
}

#[test]
fn modeled_rate_positive_at_init_does_not_warn() {
    // The positive control: a well-specified R1 > 0 must NOT raise the warning.
    let model = model_of(ODE_R1);
    let pop = pop_of(&coded_csv());
    assert!(
        check_model_data_warnings(&model, &pop, &model.default_params)
            .iter()
            .all(|d| d.code != "W_MODELED_RATE_NONPOSITIVE"),
        "a positive R1 must not warn"
    );
}
