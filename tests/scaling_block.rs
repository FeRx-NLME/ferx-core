//! Integration test for the `[scaling]` block (Phase 1).
//!
//! Tier 2: exercises the full parser → compiled model → prediction pipeline
//! via the public API without running a fit to convergence. Verifies that:
//!
//! - Form A (`obs_scale = 1000`) yields predictions that are 1/1000 of the
//!   unscaled baseline.
//! - Form C (ODE `y = central / V`) on an amount-based ODE produces the same
//!   per-observation values as a concentration-baked ODE with `obs_cmt=central`.
//!
//! These tests run on every PR (`cargo check --tests`) and exit fast — no
//! `fit()` calls — so they don't need to be gated behind `slow-tests`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{predict, ScalingSpec};
use std::collections::HashMap;

fn one_subject_pop() -> Population {
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
    let n_obs = obs_times.len();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }],
    }
}

const ANALYTICAL_BASE: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv_bolus(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

#[test]
fn form_a_scalar_scale_divides_predictions_by_exactly_k() {
    let baseline = parse_model_string(ANALYTICAL_BASE).expect("baseline parses");
    let mut scaled_src = String::from(ANALYTICAL_BASE);
    scaled_src.push_str("\n[scaling]\n  obs_scale = 1000\n");
    let scaled = parse_model_string(&scaled_src).expect("scaled parses");

    assert!(matches!(baseline.scaling, ScalingSpec::None));
    assert!(matches!(scaled.scaling, ScalingSpec::ScalarScale(k) if (k - 1000.0).abs() < 1e-12));

    let pop = one_subject_pop();
    let pop_preds = predict(&baseline, &pop, &baseline.default_params);
    let scaled_preds = predict(&scaled, &pop, &scaled.default_params);

    assert_eq!(pop_preds.len(), scaled_preds.len());
    assert!(!pop_preds.is_empty(), "must have predictions to compare");

    for (a, b) in pop_preds.iter().zip(scaled_preds.iter()) {
        assert!(a.pred > 0.0, "baseline pred must be positive");
        let ratio = a.pred / b.pred;
        assert!(
            (ratio - 1000.0).abs() < 1e-9,
            "scalar scaling must divide by exactly 1000: baseline={} scaled={} ratio={}",
            a.pred,
            b.pred,
            ratio
        );
    }
}

const ODE_CONCENTRATION_FORM: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - CL/V * central

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

const ODE_AMOUNT_FORM_C: &str = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 5
  gradient = fd
";

#[test]
fn form_c_amount_ode_matches_concentration_ode() {
    let conc = parse_model_string(ODE_CONCENTRATION_FORM).expect("concentration form parses");
    let amt = parse_model_string(ODE_AMOUNT_FORM_C).expect("Form C parses");

    // ODE spec sanity: concentration form keeps obs_cmt_idx, Form C clears it
    // and sets output_fn instead.
    let ode_conc = conc.ode_spec.as_ref().expect("conc ODE present");
    let ode_amt = amt.ode_spec.as_ref().expect("amt ODE present");
    assert!(ode_conc.obs_cmt_idx.is_some());
    assert!(ode_conc.output_fn.is_none());
    assert!(ode_amt.obs_cmt_idx.is_none());
    assert!(ode_amt.output_fn.is_some());

    let pop = one_subject_pop();
    let conc_preds = predict(&conc, &pop, &conc.default_params);
    let amt_preds = predict(&amt, &pop, &amt.default_params);

    assert_eq!(conc_preds.len(), amt_preds.len());
    assert!(!conc_preds.is_empty(), "must have predictions to compare");

    // Same physical system written two ways → identical predictions to
    // ODE-solver tolerance. Use a tight relative tolerance; the absolute
    // floor handles observations very near zero.
    for (a, b) in conc_preds.iter().zip(amt_preds.iter()) {
        let denom = a.pred.abs().max(1e-12);
        let rel = (a.pred - b.pred).abs() / denom;
        assert!(
            rel < 1e-4,
            "Form C must match concentration-baked ODE: t={} conc={} form_c={} rel={}",
            a.time,
            a.pred,
            b.pred,
            rel
        );
    }
}
