//! End-to-end tests for built-in **inverse-Gaussian (Freijer & Post)
//! absorption** (#322, #347). Exercises the `igd()` input-rate forcing through
//! the *public* API — parse → `predict()` → ODE integration → readout — the
//! `igd` counterpart of `tests/transit_absorption.rs`.
//!
//! The centrepiece is the same model-independent **mass-balance invariant**:
//! for a one-compartment model with `d/dt(A) = R_in(t) − ke·A`, integrating the
//! state gives `∫₀^∞ A dt = (∫ R_in dt) / ke = F·Dose / ke`, *regardless of the
//! absorption shape*. Recovering `Dose·V/CL` therefore confirms the `igd()`
//! forcing delivers exactly the dose mass (not zero — forcing missing; not
//! 2×Dose — bolus double-counted; not the wrong total — density mis-normalised)
//! through the whole pipeline, including the parser's argument-slot wiring.
//!
//! Unlike the Savic transit model (depot → central, central in concentration
//! units), `igd()` cannot be scaled (`igd(...)/V` is rejected), so the central
//! state here carries the drug **amount**; the invariant is on the amount AUC.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, DoseEvent, Population};

/// One-compartment model with built-in inverse-Gaussian absorption straight
/// into central (no first-order `ka`). central (CMT 1) holds the drug AMOUNT
/// (mg) and receives `R_in(tad)`; η fixed at 0 so `predict()` returns the
/// typical-value curve. CL = 5, V = 50 ⇒ ke = 0.1 ⇒ amount AUC∞ = Dose/ke =
/// 100/0.1 = 1000 mg·h. F defaults to 1.0 (no `f=` mapping).
const IGD_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,   0.1, 100.0)
  theta TVV(50.0,   5.0, 500.0)
  theta TVMAT(2.0, 0.05,  24.0)
  theta TVCV2(0.3, 0.001, 10.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MAT = TVMAT
  CV2 = TVCV2

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = igd(mat=MAT, cv2=CV2) - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single oral bolus of 100 mg into the `igd()` compartment (central, CMT 1),
/// observed on central over the supplied grid.
fn pop_single_igd(obs_times: Vec<f64>) -> Population {
    let n = obs_times.len();
    let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![dose],
            obs_times,
            vec![0.0; n],
            vec![1; n],
        )],
    }
}

#[test]
fn igd_curve_recovers_dose_auc_and_has_delayed_peak() {
    let model = parse_full_model(IGD_MODEL)
        .expect("inverse-Gaussian model parses")
        .model;

    // 0, 0.25, …, 72.0 — fine enough for trapezoidal AUC, long enough that the
    // truncated tail is negligible (ke = CL/V = 0.1 ⇒ t½ ≈ 6.9 h).
    let obs_times: Vec<f64> = (0..=288).map(|i| i as f64 * 0.25).collect();
    let pop = pop_single_igd(obs_times);
    let preds = predict(&model, &pop, &model.default_params);

    // (1) No instantaneous bolus jump: the dose enters as R_in over time, and
    //     the IG density vanishes at tad → 0, so central starts at exactly 0.
    assert_eq!(preds[0].time, 0.0);
    assert!(
        preds[0].pred.abs() < 1e-12,
        "igd dose leaked in as a bolus: central(0) = {}",
        preds[0].pred
    );

    // (2) Delayed, interior peak — the hallmark of inverse-Gaussian absorption
    //     (the amount rises while R_in dominates, then falls as elimination
    //     wins). The maximum is neither the first nor the last sample.
    let max_idx = (0..preds.len())
        .max_by(|&a, &b| preds[a].pred.partial_cmp(&preds[b].pred).unwrap())
        .unwrap();
    assert!(
        max_idx > 1 && max_idx < preds.len() - 1,
        "expected an interior Tmax, got index {} (t = {})",
        max_idx,
        preds[max_idx].time
    );

    // (3) Mass balance via the absorption-independent invariant ∫A dt = Dose/ke.
    //     Catches a missing forcing (AUC → 0), a double-counted bolus (AUC →
    //     2·Dose/ke), or a mis-normalised IG density (∫R_in ≠ Dose).
    let auc: f64 = preds
        .windows(2)
        .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
        .sum();
    let auc_inf = 100.0 * 50.0 / 5.0; // F·Dose/ke = Dose·V/CL with F = 1
    let rel = (auc - auc_inf).abs() / auc_inf;
    assert!(
        rel < 0.02,
        "igd amount AUC {:.4} vs Dose·V/CL {:.4} (rel err {:.2e})",
        auc,
        auc_inf,
        rel
    );
}

#[test]
fn igd_normal_dosing_passes_data_checks() {
    // Positive control: ordinary (non-SS, bolus) dosing into the igd()
    // compartment raises no absorption diagnostic.
    use ferx_core::check_model_data;
    let model = parse_full_model(IGD_MODEL)
        .expect("inverse-Gaussian model parses")
        .model;
    let pop = pop_single_igd(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code.starts_with("E_ABSORPTION")),
        "unexpected absorption diagnostic: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

// ── Biphasic inverse-Gaussian (Freijer & Post): the shared fraction mechanism (#388) ──

/// Two IG pathways into central, split by declared fractions (`FR1*igd + FR2*igd`).
/// `odes` is the `d/dt(central) = …` line and `fracs` the `[individual_parameters]`
/// fraction definitions, so each test varies just those two while reusing the same
/// disposition (CL = 5, V = 50 ⇒ ke = 0.1) and IG parameters.
fn biphasic_model(odes: &str, fracs: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0,    0.1, 100.0)
  theta TVV(50.0,    5.0, 500.0)
  theta TVMAT1(1.0, 0.05,  24.0)
  theta TVMAT2(4.0, 0.05,  24.0)
  theta TVCV2_1(0.3, 0.001, 10.0)
  theta TVCV2_2(0.5, 0.001, 10.0)
  theta TVFR1(0.6, 0.001, 2.0)
  theta TVFR2(0.4, 0.001, 2.0)

  omega ETA_CL ~ 0.0
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  MAT1  = TVMAT1
  MAT2  = TVMAT2
  CV2_1 = TVCV2_1
  CV2_2 = TVCV2_2
{fracs}

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  {odes}

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#
    )
}

const BIPHASIC_ODES: &str =
    "d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + FR2*igd(mat=MAT2, cv2=CV2_2) - CL/V*central";

#[test]
fn biphasic_igd_recovers_dose_auc() {
    // Mass balance through the fraction mechanism: with FR1 + FR2 = 1 the two
    // pathways together deliver exactly the dose, so the absorption-independent
    // invariant ∫A dt = F·Dose/ke holds just as for the single IG — a wrong
    // fraction scaling (e.g. both pathways delivering the full dose) would inflate
    // the AUC. FR2 = 1 − FR1 is the declared-complement pattern (#388).
    let src = biphasic_model(BIPHASIC_ODES, "  FR1 = TVFR1\n  FR2 = 1 - TVFR1");
    let model = parse_full_model(&src)
        .expect("biphasic inverse-Gaussian model parses")
        .model;
    let obs_times: Vec<f64> = (0..=288).map(|i| i as f64 * 0.25).collect();
    let pop = pop_single_igd(obs_times);
    let preds = predict(&model, &pop, &model.default_params);

    assert!(
        preds[0].pred.abs() < 1e-12,
        "biphasic dose leaked in as a bolus"
    );
    let auc: f64 = preds
        .windows(2)
        .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
        .sum();
    let auc_inf = 100.0 * 50.0 / 5.0; // F·Dose/ke
    let rel = (auc - auc_inf).abs() / auc_inf;
    assert!(
        rel < 0.02,
        "biphasic amount AUC {auc:.4} vs Dose·V/CL {auc_inf:.4} (rel {rel:.2e})"
    );
}

#[test]
fn biphasic_igd_fraction_validation() {
    use ferx_core::check_model_data;
    let pop = pop_single_igd(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    let has_fraction_err = |src: &str| {
        let model = parse_full_model(src).expect("model parses").model;
        check_model_data(&model, &pop)
            .iter()
            .any(|d| d.code == "E_ABSORPTION_FRACTION")
    };

    // Good: complementary fractions summing to 1 → no fraction diagnostic.
    assert!(!has_fraction_err(&biphasic_model(
        BIPHASIC_ODES,
        "  FR1 = TVFR1\n  FR2 = 1 - TVFR1"
    )));
    // (c) Fractions that don't sum to 1 (both 0.6 → 1.2) are rejected.
    assert!(has_fraction_err(&biphasic_model(
        BIPHASIC_ODES,
        "  FR1 = TVFR1\n  FR2 = TVFR1"
    )));
    // (b) A fraction outside (0, 1] (FR1 = 1.5) is rejected.
    assert!(has_fraction_err(&biphasic_model(
        BIPHASIC_ODES,
        "  FR1 = TVFR1 * 2.5\n  FR2 = TVFR2"
    )));
    // (a) Structural: a bare term alongside a fractioned one on the same compartment
    //     (the dose would be over-delivered) is rejected.
    assert!(has_fraction_err(&biphasic_model(
        "d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) + igd(mat=MAT2, cv2=CV2_2) - CL/V*central",
        "  FR1 = TVFR1",
    )));
    // (d) A *lone* fractioned term (a single pathway carrying a fraction) is rejected:
    //     a fraction only partitions a dose across ≥2 terms, so a single pathway must
    //     be written bare (review #1). Caught structurally, before the sum-check.
    assert!(has_fraction_err(&biphasic_model(
        "d/dt(central) = FR1*igd(mat=MAT1, cv2=CV2_1) - CL/V*central",
        "  FR1 = TVFR1",
    )));
}
