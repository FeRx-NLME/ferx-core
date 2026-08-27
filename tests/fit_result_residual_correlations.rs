use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

const MODEL: &str = "\
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  theta TVV(10.0, 0.1, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
";

#[test]
fn fit_result_carries_fixed_block_sigma_correlations() {
    let model = parse_model_string(MODEL).expect("block_sigma model must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };

    let result = fit(&model, &population, &model.default_params, &options)
        .expect("initial correlated-residual evaluation must succeed");

    assert_eq!(result.residual_correlations, model.residual_correlations);
    let corr = result.residual_correlations[0];
    assert_eq!(result.sigma_names[corr.sigma_i], "ADD_ERR");
    assert_eq!(result.sigma_names[corr.sigma_j], "PROP_ERR");
    assert!((corr.rho - 0.5).abs() < 1e-12);
    let covariance = corr.rho * result.sigma[corr.sigma_i] * result.sigma[corr.sigma_j];
    assert!((covariance - 0.1).abs() < 1e-12);
}
