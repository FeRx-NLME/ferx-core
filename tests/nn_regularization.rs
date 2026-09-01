//! Public-API integration tests for the covariate-NN (DCM) regularization
//! penalties: L2 (`nn_l2`) and smoothness/curvature (`nn_smooth`).
//!
//! Scope: this file drives the feature the way ferx-r does — set the two
//! `FitOptions` fields, call [`fit`], read the result. Nothing here touches a
//! crate-private item, so it doubles as the check that the feature is usable
//! from outside the crate at all.
//!
//! The mechanism-level coverage lives in `src/nn/mod.rs`, because it needs
//! `pub(crate)` internals (`NnRegularizer`, `CurvatureGrid`,
//! `MlpMapper::weight_param_indices`) that the workspace boundary rule keeps out
//! of the public API:
//!   - `nn::regularization_tests` — per-kernel L2 / curvature values and their
//!     analytic-vs-FD gradient parity.
//!   - `nn::regularizer_fit_tests` — the whole-`NnRegularizer` λ = 0 no-op, its
//!     FD gradient parity, and the fitted weight-norm / modulator-variance
//!     shrinkage (Tier 3, `slow-tests`).
//!
//! Run via:
//!   cargo test --features nn --test nn_regularization                 # fast
//!   cargo test --features nn,slow-tests --test nn_regularization      # + slow

#![cfg(feature = "nn")]

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{CompiledModel, FitOptions, Population};
use ferx_core::{fit, read_nonmem_csv};

/// Two-cpt oral DCM with a small MLP mapping (WT, CRCL) → the five PK params.
///
/// Kept in step with the copy in `src/nn/mod.rs`'s `regularizer_fit_tests`; the
/// two cannot share a fixture because an integration test cannot see
/// `#[cfg(test)]` items. `center` / `scale` z-score the inputs — see the note on
/// that copy for why raw covariates saturate this network.
const MODEL: &str = r#"
[parameters]
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [WT, CRCL]
  outputs    = [CL, V1, Q, V2, KA]
  layers     = [4]
  activation = tanh
  output     = softplus
  center     = [70, 86]
  scale      = [12, 23]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  Q  = TYPICAL_PK.Q  * exp(ETA_Q)
  V2 = TYPICAL_PK.V2 * exp(ETA_V2)
  KA = TYPICAL_PK.KA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 50
  covariance = false
"#;

fn load() -> (CompiledModel, FitOptions, Population) {
    let parsed = parse_full_model(MODEL).expect("model parses with --features nn");
    let population = read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads");
    (parsed.model, parsed.fit_options, population)
}

/// A couple of outer iterations with both penalties on — proves the optimizer
/// wiring (penalized objective value + matching analytic gradient) runs end to
/// end from a public-API caller without panicking and returns a finite OFV.
#[test]
fn fit_with_regularization_runs() {
    let (model, mut options, population) = load();
    options.outer_maxiter = 2;
    options.nn_l2_lambda = 5e-2;
    options.nn_smooth_lambda = 5e-2;
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("regularized fit returns Ok");
    assert!(
        result.ofv.is_finite(),
        "regularized fit OFV must be finite, got {}",
        result.ofv
    );
}

/// λ = 0 must reproduce the unregularized fit byte-for-byte (the no-op
/// guarantee). Runs two short fits and asserts identical OFV and theta.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn zero_lambda_fit_matches_baseline() {
    let (model, options, population) = load();

    let mut baseline = options.clone();
    baseline.outer_maxiter = 8;
    let base_fit =
        fit(&model, &population, &model.default_params, &baseline).expect("baseline fit");

    let mut zero_reg = baseline.clone();
    zero_reg.nn_l2_lambda = 0.0;
    zero_reg.nn_smooth_lambda = 0.0;
    let zr_fit = fit(&model, &population, &model.default_params, &zero_reg).expect("λ=0 fit");

    assert_eq!(
        base_fit.ofv, zr_fit.ofv,
        "λ = 0 must be byte-identical to the unregularized baseline"
    );
    for (a, b) in base_fit.theta.iter().zip(zr_fit.theta.iter()) {
        assert_eq!(a, b, "λ = 0 theta must be byte-identical to baseline");
    }
}
