//! Tier-1 tests for `W_ZERO_DV_MAGNITUDE_ONLY` (#2459): `DV = 0` rows scored by a residual
//! error model whose variance is zero at a zero prediction. A child of `validation`, so
//! `super::check_zero_dv_magnitude_only` resolves directly.
//!
//! The input space, enumerated before the message was written: error model (proportional,
//! power, combined with additive start > 0 / = 0, additive, per-CMT with the zero on each
//! endpoint, an unregistered CMT) × row kind (plain, `CENS != 0` under `drop` / `m3`, FREM
//! covariate row). Every skip and every gate has a test whose two halves straddle it, so
//! forcing the gate either way reddens one assertion.

use super::{check_zero_dv_magnitude_only, zero_dv_magnitude_only_under};
use crate::diagnostics::Diagnostic;
use crate::parser::model_parser::parse_model_string;
use crate::types::{
    classify_warning, BloqMethod, CompiledModel, DoseEvent, Population, Subject, WarningCode,
    WarningSeverity,
};

fn model(error_block: &str, sigmas: &str) -> CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         theta POW(0.8, 0.1, 2.0)\n  omega ETA_CL ~ 0.09\n{sigmas}\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n{error_block}"
    ))
    .expect("fixture parses")
}

fn proportional() -> CompiledModel {
    model("  DV ~ proportional(PROP)\n", "  sigma PROP ~ 0.1 (sd)\n")
}

/// One subject per entry of `dvs`, observed at 0.5, 2, 8, 24 h on `CMT = cmt`.
fn pop(dvs: &[[f64; 4]], cmt: usize) -> Population {
    let subjects = dvs
        .iter()
        .enumerate()
        .map(|(i, dv)| Subject {
            id: format!("{}", i + 1),
            obs_times: vec![0.5, 2.0, 8.0, 24.0],
            observations: dv.to_vec(),
            obs_cmts: vec![cmt; 4],
            cens: vec![0; 4],
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Three subjects, two of them with `DV = 0` rows (three between them), the first at ID 1,
/// 0.5 h. The third has none, so the subject count is not the population size.
fn with_zeros(cmt: usize) -> Population {
    pop(
        &[
            [0.0, 20.0, 5.0, 1.0],
            [0.0, 0.0, 5.0, 1.0],
            [50.0, 20.0, 5.0, 1.0],
        ],
        cmt,
    )
}

/// The same design with the zeros replaced by positive values.
fn without_zeros(cmt: usize) -> Population {
    pop(&[[50.0, 20.0, 5.0, 1.0], [50.0, 20.0, 5.0, 1.0]], cmt)
}

fn run(m: &CompiledModel, p: &Population, bloq: BloqMethod) -> Option<Diagnostic> {
    zero_dv_magnitude_only_under(m, p, &m.default_params, bloq)
}

/// The public entry reads the censoring method from `bloq_in_force` (#1824), so an armed
/// override decides, not the model's own field: the same `CENS = 1` zeros on a `drop` model
/// are skipped once `m3` is armed, and counted with nothing armed.
#[test]
fn the_entry_point_follows_the_armed_override() {
    let m = proportional();
    assert_eq!(m.bloq_method, BloqMethod::Drop);
    let mut p = with_zeros(1);
    for s in &mut p.subjects {
        s.cens = s.observations.iter().map(|&y| i8::from(y == 0.0)).collect();
    }
    assert!(check_zero_dv_magnitude_only(&m, &p, &m.default_params).is_some());
    let _armed = crate::types::arm_bloq_override(Some(BloqMethod::M3));
    assert!(check_zero_dv_magnitude_only(&m, &p, &m.default_params).is_none());
}

/// The finding on its own fixture, sentence by sentence: each `contains` below names one
/// sentence of the message, so deleting that sentence reddens exactly this test. The silent
/// half is the same model on the same design without the zeros — the gate on `DV == 0`.
#[test]
fn proportional_with_zero_dv_warns_and_says_why_and_what_to_do() {
    let m = proportional();
    let d = run(&m, &with_zeros(1), BloqMethod::Drop).expect("DV = 0 under proportional warns");
    assert_eq!(d.code, "W_ZERO_DV_MAGNITUDE_ONLY");
    assert!(!d.is_error());
    let msg = &d.message;
    for needle in [
        // The head the classifier keys on.
        "W_ZERO_DV_MAGNITUDE_ONLY: ",
        // Sentence 1: what was found, counted, and where it starts.
        "3 observation(s) in 2 subject(s) have DV = 0 (first: ID 1 at time 0.5)",
        "variance is zero at a zero prediction (proportional, power, or an additive SD that \
         starts at 0)",
        // Sentence 2: why it matters.
        "These rows have no stable likelihood",
        "down to the subnormal guard (about -708 per row in -2LL)",
        "stop at a wrong optimum that still reports converged",
        // Sentence 3: what to do.
        "Add an additive component (combined(...))",
        "remove these rows (e.g. pre-dose or pre-lag zeros)",
        "a below-LLOQ row belongs in CENS = 1 with DV = LLOQ and bloq_method = m3",
    ] {
        assert!(msg.contains(needle), "missing {needle:?} in: {msg}");
    }
    assert!(run(&m, &without_zeros(1), BloqMethod::Drop).is_none());
}

/// The −708 in the message is `ln(f64::MIN_POSITIVE)`, measured rather than recalled: the
/// smallest normal variance, below which the guard substitutes `1e-12`.
#[test]
fn the_stated_floor_is_ln_min_positive() {
    let floor = f64::MIN_POSITIVE.ln();
    assert!(
        (floor - (-708.4)).abs() < 0.05,
        "ln(MIN_POSITIVE) = {floor}"
    );
}

/// The gate is the engine's variance at `f = 0`, not the error model's name: a `power(...)`
/// magnitude warns like `proportional`, and an additive model does not.
#[test]
fn power_warns_and_additive_does_not() {
    let pow = model("  DV ~ power(PROP, POW)\n", "  sigma PROP ~ 0.1 (sd)\n");
    assert!(run(&pow, &with_zeros(1), BloqMethod::Drop).is_some());
    let add = model("  DV ~ additive(ADD)\n", "  sigma ADD ~ 0.5 (sd)\n");
    assert!(run(&add, &with_zeros(1), BloqMethod::Drop).is_none());
}

/// `combined` straddles the gate on its additive start alone: a positive additive SD gives a
/// positive variance at `f = 0`, a zero one does not.
#[test]
fn combined_warns_only_when_its_additive_sd_starts_at_zero() {
    let mut m = model(
        "  DV ~ combined(PROP, ADD)\n",
        "  sigma PROP ~ 0.1 (sd)\n  sigma ADD ~ 0.5 (sd)\n",
    );
    assert!(run(&m, &with_zeros(1), BloqMethod::Drop).is_none());
    m.default_params.sigma.values[1] = 0.0;
    assert!(run(&m, &with_zeros(1), BloqMethod::Drop).is_some());
}

/// A censored row is scored by its CDF under `m3` and as an ordinary `DV` under `drop`
/// (`BloqMethod::is_censored_row`), so the same `CENS = 1` zeros are silent under one and
/// counted under the other.
#[test]
fn censored_zeros_are_skipped_under_m3_and_counted_under_drop() {
    let m = proportional();
    let mut p = with_zeros(1);
    for s in &mut p.subjects {
        s.cens = s.observations.iter().map(|&y| i8::from(y == 0.0)).collect();
    }
    assert!(run(&m, &p, BloqMethod::M3).is_none());
    let d = run(&m, &p, BloqMethod::Drop).expect("drop fits CENS rows at their DV");
    assert!(d.message.contains("3 observation(s)"), "{}", d.message);
}

/// A FREM covariate row (`fremtype > 0`) is scored with its own EPSCOV variance, not the PK
/// error model, so its zero is not counted; the same row with `fremtype = 0` is.
#[test]
fn frem_covariate_rows_are_skipped() {
    let m = proportional();
    let mut p = with_zeros(1);
    for s in &mut p.subjects {
        s.fremtype = s
            .observations
            .iter()
            .map(|&y| if y == 0.0 { 1 } else { 0 })
            .collect();
    }
    assert!(run(&m, &p, BloqMethod::Drop).is_none());
    for s in &mut p.subjects {
        s.fremtype = vec![0; 4];
    }
    assert!(run(&m, &p, BloqMethod::Drop).is_some());
}

/// Per-CMT: the zero counts only on the endpoint whose variance vanishes. A zero on an
/// unregistered CMT reads a `NaN` variance, which is not zero, and stays silent.
#[test]
fn per_cmt_warns_on_the_proportional_endpoint_only() {
    // Per-CMT error models need an ODE structural model.
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma ADD ~ 0.5 (sd)\n  sigma PROP ~ 0.1 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
         [structural_model]\n  ode(states=[central, peri])\n\
         [odes]\n  d/dt(central) = -CL/V*central - 0.1*central + 0.1*peri\n  \
         d/dt(peri) = 0.1*central - 0.1*peri\n\
         [scaling]\n  y[CMT=1] = central / V\n  y[CMT=2] = peri\n\
         [error_model]\n  CMT=1: DV ~ additive(ADD)\n  CMT=2: DV ~ proportional(PROP)\n",
    )
    .expect("per-CMT fixture parses");
    assert!(run(&m, &with_zeros(1), BloqMethod::Drop).is_none());
    assert!(run(&m, &with_zeros(2), BloqMethod::Drop).is_some());
    assert!(run(&m, &with_zeros(7), BloqMethod::Drop).is_none());
}

/// The fit's structured channel files it as data quality, not as a convergence failure —
/// the message says "still reports converged", which a prose convergence arm could claim.
#[test]
fn classified_as_a_data_quality_warning() {
    let m = proportional();
    let d = run(&m, &with_zeros(1), BloqMethod::Drop).unwrap();
    let e = classify_warning(&d.message);
    assert_eq!(e.category, WarningCode::DataQuality);
    assert_eq!(e.severity, WarningSeverity::Warning);
}

/// Not in the bundle `simulate()` / `predict()` share: on a simulation template `DV = 0` is a
/// placeholder. The control half shows the fixture does trip the fit-only check.
#[test]
fn not_in_the_shared_model_data_bundle() {
    let m = proportional();
    let p = with_zeros(1);
    assert!(run(&m, &p, BloqMethod::Drop).is_some());
    let bundle = crate::api::check_model_data_warnings(&m, &p, &m.default_params);
    assert!(
        !bundle.iter().any(|d| d.code == "W_ZERO_DV_MAGNITUDE_ONLY"),
        "{bundle:?}"
    );
    let served = crate::api::predict_diag(&m, &p, &m.default_params).expect("predicts");
    assert!(
        !served
            .warnings
            .iter()
            .any(|w| w.contains("W_ZERO_DV_MAGNITUDE_ONLY")),
        "{:?}",
        served.warnings
    );
}
