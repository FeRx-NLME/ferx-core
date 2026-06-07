//! Tier 2 integration smoke test for `omega_dist = vine`.
//!
//! Calls `fit()` with `omega_dist = vine` on a tiny synthetic population and
//! confirms it returns `Ok` with `vine_params` populated. Does not converge —
//! just a few SAEM iterations to exercise the public API boundary.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaDist, Population, Subject};
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::collections::HashMap;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.16
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn tiny_population() -> Population {
    let obs_times = vec![0.5_f64, 2.0, 8.0];
    let subjects = (1u32..=20)
        .map(|i| {
            // Deterministic synthetic observations based on a nominal 1-cpt profile.
            let ke = 5.0 / 50.0;
            let obs: Vec<f64> = obs_times
                .iter()
                .map(|&t| (100.0 / 50.0) * (-ke * t).exp() * (1.0 + 0.02 * (i as f64 - 10.0)))
                .collect();
            Subject {
                id: format!("{i}"),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                observations: obs,
                obs_cmts: vec![1; obs_times.len()],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; obs_times.len()],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            }
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

#[test]
fn vine_omega_dist_fit_returns_ok() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let population = tiny_population();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 5;
    opts.saem_n_mh_steps = 3;
    opts.saem_omega_burnin = 0;
    opts.saem_seed = Some(42);
    opts.saem_omega_dist = OmegaDist::VineCopula;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(&model, &population, &model.default_params, &opts).expect("vine SAEM must return Ok");

    // vine_params should be populated after a successful vine run.
    assert!(
        result.vine_params.is_some(),
        "vine_params should be Some after omega_dist = vine fit"
    );

    let vp = result.vine_params.as_ref().unwrap();
    assert_eq!(
        vp.marginals.len(),
        2,
        "expected 2 marginals (ETA_CL, ETA_V)"
    );
    assert!(
        !vp.trees.is_empty(),
        "vine_params.trees should be non-empty"
    );
}
