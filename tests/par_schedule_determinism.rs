//! Tier-2 determinism: every fitted quantity (OFV, theta, omega, sigma) is bit-identical at
//! 1, 4 and 8 workers, on a population whose subjects differ in cost (3 to 20 observations),
//! so work stealing hands the subjects to different workers in different runs. Complements
//! `thread_count_determinism.rs` (#703), which pins only the OFV on warfarin.
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, CompiledModel, DoseEvent, EstimationMethod, FitOptions, Population, Subject};
use std::collections::HashMap;

fn model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  omega ETA_V ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse")
}

fn subject(id: &str, scale: f64, n_obs: usize) -> Subject {
    let obs_times: Vec<f64> = (1..=n_obs).map(|k| 0.5 + 2.0 * k as f64).collect();
    let observations = obs_times
        .iter()
        .map(|t| scale * 10.0 * (-0.1 * t).exp())
        .collect();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n_obs],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cmt_toggles: Vec::new(),
        cens: vec![0; n_obs],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

fn population() -> Population {
    Population {
        subjects: (0..48)
            .map(|i| {
                subject(
                    &format!("{i}"),
                    1.0 + 0.07 * ((i * 7) % 13) as f64,
                    3 + (i * 5) % 18,
                )
            })
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn bits(threads: usize) -> Vec<u64> {
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 8,
        run_covariance_step: false,
        threads: Some(threads),
        ..Default::default()
    };
    let model = model();
    let r = fit(&model, &population(), &model.default_params, &opts).expect("fit");
    let mut v = vec![r.ofv.to_bits()];
    v.extend(r.theta.iter().map(|x| x.to_bits()));
    v.extend(r.omega.iter().map(|x| x.to_bits()));
    v.extend(r.sigma.iter().map(|x| x.to_bits()));
    v
}

/// A reduction in completion order, or per-worker partial sums, would break this comparison.
#[test]
fn fit_is_bit_identical_at_1_4_and_8_threads() {
    let one = bits(1);
    assert!(f64::from_bits(one[0]).is_finite());
    assert_eq!(one, bits(4), "1 vs 4 threads");
    assert_eq!(one, bits(8), "1 vs 8 threads");
}
