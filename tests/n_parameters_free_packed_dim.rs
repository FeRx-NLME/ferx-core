//! `FitResult::n_parameters` and the BIC tally on a real `fit()` (#1177).
//!
//! `fit()` counts `n_parameters` from `packed_held_mask` — the coordinates the
//! outer optimizer searches — so a mixed `block_omega` + diagonal `omega`'s
//! structural zeros are not parameters. The unit test on the tally helper
//! never calls `fit()`, so reverting the call site to `packed_fixed_mask`
//! (which counts the zeros) left the whole suite green. This one goes through
//! the public entry point and asserts against `free_packed_dim()`.
//!
//! Tier 2: two outer iterations, no covariance step — returns immediately.

use ferx_core::{bic, fit, parse_model_string, read_nonmem_csv, BicType, FitOptions};
use std::path::Path;

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

#[test]
fn n_parameters_excludes_block_omega_structural_zeros_on_a_fit() {
    let model = parse_model_string(MODEL).expect("model parses");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.outer_maxiter = 2;
    let result = fit(&model, &population, &model.default_params, &opts).expect("fit runs");

    // 3 θ + 3 Cholesky entries of the 2×2 block + 1 diagonal Ω + 1 σ. The full
    // 3×3 lower triangle would be 6 Ω entries (10 in total) — the two
    // cross-block zeros that `packed_fixed_mask` counts.
    assert_eq!(model.free_packed_dim(), 8);
    assert_eq!(result.n_parameters, model.free_packed_dim());
    assert_eq!(result.bic_inputs.n_free(), result.n_parameters);
    assert_eq!(result.bic_inputs.omega, 4);
    assert_eq!(result.bic_inputs.theta_random, 3);
    assert!(bic(&result, BicType::Mixed).is_finite());
    assert_eq!(
        bic(&result, BicType::Fixed).to_bits(),
        result.bic.to_bits(),
        "Fixed reproduces FitResult::bic on a live fit"
    );
    // The NLopt outer loop records its own init-escape verdict.
    assert!(result.left_init.is_some());
}
