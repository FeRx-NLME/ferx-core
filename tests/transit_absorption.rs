//! End-to-end tests for built-in **transit-compartment absorption** (#322,
//! Phase 0). Exercises the `transit()` input-rate forcing through the *public*
//! API — parse → `predict()` → ODE integration → readout — plus the fit-time
//! guards that reject the not-yet-supported dosing combinations.
//!
//! The centrepiece is an **AUC invariant**: for a one-compartment model the
//! steady-state exposure is `AUC∞ = F·Dose / CL` *regardless of the absorption
//! model*. Integrating the predicted transit curve must therefore recover
//! `Dose/CL` — a model-independent check that the forcing delivers exactly the
//! dose mass (not zero — forcing missing; not 2×Dose — bolus double-counted)
//! through the whole pipeline, including the parser's argument-slot wiring.

mod common;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{check_model_data, predict, DoseEvent, Population};

/// One-compartment oral model with built-in Savic transit absorption.
/// depot (CMT 1, mg) receives `R_in(tad)`; central (CMT 2, mg/L) is the readout.
/// η fixed at 0 so `predict()` returns the typical-value curve. CL = 5 ⇒
/// AUC∞ = 100/5 = 20 mg·h/L. F defaults to 1.0 (no `f=` mapping).
const TRANSIT_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.05,  24.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0,   0.1,  30.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  MTT = TVMTT
  NTR = TVN

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Same disposition + transit absorption, but with a `[diffusion]` block — the
/// SDE/EKF path, which cannot carry the input-rate forcing (rejected by guard).
const TRANSIT_DIFFUSION_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.05,  24.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0,   0.1,  30.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  MTT = TVMTT
  NTR = TVN

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[diffusion]
  central ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = foce
"#;

/// Transit absorption whose mean transit time is out of domain at typical
/// values: `MTT = TVMTT` with a *negative* typical `TVMTT`, so `mtt ≤ 0`. Without
/// the fit-time domain guard this would produce `ktr.ln() = NaN` and propagate a
/// NaN through the ODE RHS; with it, the model is rejected loudly.
const TRANSIT_BAD_DOMAIN_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.0, 0.05,  24.0)
  theta TVMTT(-1.0, -10.0, 24.0)
  theta TVN(3.0,   0.1,  30.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KA  = TVKA
  MTT = TVMTT
  NTR = TVN

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"#;

/// Single oral bolus of 100 mg into the depot (CMT 1), observed on central
/// (CMT 2) over a fine grid out to 72 h (~10 elimination half-lives).
fn pop_single_oral(obs_times: Vec<f64>) -> Population {
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
            vec![2; n],
        )],
    }
}

#[test]
fn transit_curve_recovers_dose_auc_and_has_delayed_peak() {
    let model = parse_full_model(TRANSIT_MODEL)
        .expect("transit model parses")
        .model;

    // 0, 0.25, …, 72.0 — fine enough for trapezoidal AUC, long enough that the
    // truncated tail is negligible (ke = CL/V = 0.1 ⇒ t½ ≈ 6.9 h).
    let obs_times: Vec<f64> = (0..=288).map(|i| i as f64 * 0.25).collect();
    let pop = pop_single_oral(obs_times);
    let preds = predict(&model, &pop, &model.default_params);

    // (1) No instantaneous bolus jump: the dose enters as R_in over time, so
    //     central starts at exactly 0 (initial state) at the dose time.
    assert_eq!(preds[0].time, 0.0);
    assert!(
        preds[0].pred.abs() < 1e-12,
        "transit dose leaked in as a bolus: central(0) = {}",
        preds[0].pred
    );

    // (2) Delayed, interior peak — the hallmark of transit absorption (Tmax is
    //     pushed out vs first-order). The maximum is neither the first nor the
    //     last sample.
    let max_idx = (0..preds.len())
        .max_by(|&a, &b| preds[a].pred.partial_cmp(&preds[b].pred).unwrap())
        .unwrap();
    assert!(
        max_idx > 1 && max_idx < preds.len() - 1,
        "expected an interior Tmax, got index {} (t = {})",
        max_idx,
        preds[max_idx].time
    );

    // (3) Mass balance via the absorption-independent invariant AUC∞ = Dose/CL.
    //     Catches a missing forcing (AUC → 0) or a double-counted bolus
    //     (AUC → 2·Dose/CL).
    let auc: f64 = preds
        .windows(2)
        .map(|w| 0.5 * (w[0].pred + w[1].pred) * (w[1].time - w[0].time))
        .sum();
    let auc_inf = 100.0 / 5.0; // F·Dose/CL with F = 1, CL = 5
    let rel = (auc - auc_inf).abs() / auc_inf;
    assert!(
        rel < 0.02,
        "transit AUC {:.4} vs Dose/CL {:.4} (rel err {:.2e})",
        auc,
        auc_inf,
        rel
    );
}

#[test]
fn transit_normal_dosing_passes_data_checks() {
    // Positive control: ordinary (non-SS) dosing into the transit compartment
    // raises no absorption diagnostic.
    let model = parse_full_model(TRANSIT_MODEL)
        .expect("transit model parses")
        .model;
    let pop = pop_single_oral(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    let diags = check_model_data(&model, &pop);
    assert!(
        !diags.iter().any(|d| d.code.starts_with("E_ABSORPTION")),
        "unexpected absorption diagnostic: {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn ss_dose_into_transit_compartment_is_rejected() {
    // Steady-state dosing into a transit compartment is not yet supported and
    // must be rejected loudly rather than silently mis-modeled as a bolus train.
    let model = parse_full_model(TRANSIT_MODEL)
        .expect("transit model parses")
        .model;
    let ss_dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0); // SS=1, II=12
    let n = 2;
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![ss_dose],
            vec![1.0, 6.0],
            vec![0.0; n],
            vec![2; n],
        )],
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_SS"),
        "expected E_ABSORPTION_SS, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn infusion_into_transit_compartment_is_rejected() {
    // An infusion (RATE>0) into a transit compartment would be delivered twice —
    // once as the `+rate` infusion injection in the ODE RHS wrapper, once as
    // R_in(tad) superposed by the forcing — silently ~doubling exposure. The
    // transit chain already defines the input rate from the dose amount, so an
    // infusion rate on that record is undefined and must be rejected loudly.
    let model = parse_full_model(TRANSIT_MODEL)
        .expect("transit model parses")
        .model;
    let inf_dose = DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0); // RATE=10>0 into depot (CMT 1)
    let n = 2;
    let pop = Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![inf_dose],
            vec![1.0, 6.0],
            vec![0.0; n],
            vec![2; n],
        )],
    };
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_RATE"),
        "expected E_ABSORPTION_RATE, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn transit_with_diffusion_block_is_rejected() {
    // A built-in input-rate model + a [diffusion] block (SDE/EKF) is rejected:
    // the EKF propagation does not carry the R_in forcing.
    let model = parse_full_model(TRANSIT_DIFFUSION_MODEL)
        .expect("transit+diffusion model parses")
        .model;
    let pop = pop_single_oral(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_DIFFUSION"),
        "expected E_ABSORPTION_DIFFUSION, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}

#[test]
fn transit_model_with_undeclared_param_fails_to_parse() {
    // The `extract_input_rate_terms` error must propagate all the way out of
    // `build_ode_spec` / `parse_full_model` (the `?` at the call site), not only
    // from the unit-level helper. `mtt=NOPE` references an undeclared parameter.
    let bad = TRANSIT_MODEL.replace("mtt=MTT", "mtt=NOPE");
    let err = parse_full_model(&bad)
        .err()
        .expect("expected a parse error for an undeclared transit parameter");
    assert!(
        err.contains("not a declared individual parameter"),
        "unexpected parse error: {err}"
    );
}

#[test]
fn out_of_domain_transit_parameter_is_rejected() {
    // A transit `mtt ≤ 0` at typical values must be rejected at fit time rather
    // than silently propagating a NaN through the ODE RHS (the `validate_transit`
    // domain guard, evaluated on η = 0 per subject).
    let model = parse_full_model(TRANSIT_BAD_DOMAIN_MODEL)
        .expect("bad-domain transit model still parses")
        .model;
    let pop = pop_single_oral(vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    let diags = check_model_data(&model, &pop);
    assert!(
        diags.iter().any(|d| d.code == "E_ABSORPTION_DOMAIN"),
        "expected E_ABSORPTION_DOMAIN, got {:?}",
        diags.iter().map(|d| &d.code).collect::<Vec<_>>()
    );
}
