//! Integration tests for multi-endpoint (per-CMT) residual error models —
//! issue #14, simultaneous PK/PD fitting.
//!
//! These exercise the public parse / `predict()` / `fit()` boundary. The
//! per-CMT residual dispatch and the SAEM M-step score derivatives are covered
//! by fast unit tests in `src/` (`ErrorSpec::{variance_at, dvar_df,
//! dvar_dlogsigma}`, `compute_r_diag`/`compute_iwres`, and the parser). Here we
//! add a fast prediction regression test (co-temporal multi-CMT obs must all be
//! finite) and gate the full ODE fits as slow (heavy in the unoptimized test
//! profile). The canonical nonlinear Emax showcase lives in
//! `examples/emax_pkpd.ferx`.

use ferx_core::parser::model_parser::{parse_model_file, parse_model_string};
use ferx_core::types::{DoseEvent, ErrorSpec, Population, SigmaType, Subject};
use ferx_core::{fit, predict, EstimationMethod, FitOptions};
use std::collections::HashMap;
use std::path::Path;

/// Small linear-ODE PK/PD model: central compartment (PK, CMT=1, proportional
/// error) plus a biophase effect compartment (PD, CMT=2, additive error).
const LINEAR_PKPD: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKE0(0.5, 0.05, 5.0)

  omega ETA_CL ~ 0.04

  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 0.50 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)

[fit_options]
  method   = focei
  gradient = fd
";

/// One subject observed on CMT=1 (PK) and CMT=2 (PD) at three times each.
fn pkpd_pop() -> Population {
    let times = vec![1.0, 1.0, 2.0, 2.0, 4.0, 4.0];
    let cmts = vec![1, 2, 1, 2, 1, 2];
    let n = times.len();
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: times,
            observations: vec![8.0, 1.0, 6.0, 2.0, 4.0, 3.0],
            obs_cmts: cmts,
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }],
    }
}

#[test]
fn linear_pkpd_parses_to_per_cmt_error_spec() {
    let model = parse_model_string(LINEAR_PKPD).expect("linear PK/PD model must parse");
    match &model.error_spec {
        ErrorSpec::PerCmt(map) => {
            assert_eq!(map.len(), 2);
            assert!(map.contains_key(&1) && map.contains_key(&2));
        }
        other => panic!("expected PerCmt error spec, got {other:?}"),
    }
    // sigma_types maps each global sigma slot to its endpoint's type.
    let types = model.error_spec.sigma_types(2);
    assert!(types.contains(&SigmaType::Proportional));
    assert!(types.contains(&SigmaType::Additive));
}

/// Regression test for the co-temporal multi-CMT recorder bug: PK and PD
/// observations sampled at the *same* timepoints must both receive finite
/// predictions. Before the fix, the ODE recorder keyed observations by time
/// alone, so all but one observation per timepoint stayed at the initial NaN.
#[test]
fn per_cmt_predictions_finite_for_cotemporal_obs() {
    let model = parse_model_string(LINEAR_PKPD).expect("model parses");
    let pop = pkpd_pop(); // obs on CMT 1 and 2 share each timepoint
    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .iter()
        .map(|p| p.pred)
        .collect();
    assert!(
        preds.iter().all(|v| v.is_finite()),
        "co-temporal multi-CMT predictions must all be finite, got {preds:?}"
    );
}

/// SAEM now supports per-CMT error models (the M-step score dispatches the
/// residual-variance derivatives per endpoint). Gated as slow: a full ODE
/// SAEM fit is too heavy for the per-PR job. The per-CMT M-step gradient is
/// unit-tested against finite differences in `estimation::saem`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn per_cmt_saem_fit_runs() {
    let model = parse_model_string(LINEAR_PKPD).expect("model parses");
    let pop = pkpd_pop();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 5;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("per-CMT SAEM fit must run");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert_eq!(result.sigma.len(), 2);
    for (name, s) in result.sigma_names.iter().zip(result.sigma.iter()) {
        assert!(
            s.is_finite() && *s > 0.0,
            "endpoint sigma {name} must be positive/finite, got {s}"
        );
    }
}

/// A model observing a CMT with no matching `CMT=N:` error entry is rejected
/// at `fit()` entry (coverage validation), naming the missing compartment.
#[test]
fn per_cmt_error_missing_endpoint_is_rejected_at_fit() {
    let model = parse_model_string(LINEAR_PKPD).expect("model parses");
    // Population observes CMT=3, which has no endpoint (model covers 1 and 2).
    let mut pop = pkpd_pop();
    pop.subjects[0].obs_cmts = vec![1, 2, 1, 2, 1, 3];

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let err = fit(&model, &pop, &model.default_params, &opts)
        .expect_err("observed CMT with no error endpoint must be rejected");
    assert!(err.contains('3'), "error should name missing CMT 3: {err}");
}

/// Per-CMT error models cannot be combined with a `[diffusion]` (SDE/EKF)
/// block: observing multiple compartments needs a Form C `y[CMT=N]` readout,
/// which the parser rejects on SDE models. So the multi-endpoint + SDE
/// combination is unreachable at parse time (and the EKF path can soundly
/// assume a single error model).
#[test]
fn per_cmt_readout_with_sde_is_rejected_at_parse() {
    let sde_pkpd = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 0.50 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect)  =  central/V - effect

[diffusion]
  central ~ 0.5

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_PK)
  CMT=2: DV ~ additive(ADD_ERR_PD)
";
    let err = parse_model_string(sde_pkpd)
        .err()
        .expect("multi-CMT (Form C) readout on an SDE model must be rejected");
    assert!(
        err.contains("SDE") || err.to_lowercase().contains("diffusion"),
        "error should cite the SDE restriction: {err}"
    );
}

/// The shipped Emax PK/PD showcase parses into a two-endpoint per-CMT error
/// spec (proportional PK on CMT=2 + additive PD on CMT=3).
#[test]
fn emax_example_parses_to_per_cmt_error_spec() {
    let model = parse_model_file(Path::new("examples/emax_pkpd.ferx"))
        .expect("emax_pkpd example must parse");
    match &model.error_spec {
        ErrorSpec::PerCmt(map) => {
            assert!(map.contains_key(&2) && map.contains_key(&3));
        }
        other => panic!("expected PerCmt error spec, got {other:?}"),
    }
}
