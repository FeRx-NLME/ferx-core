//! End-to-end smoke test: fit an NN-bearing model (DCM-style) via FOCEI
//! and verify it runs without panicking and returns a sensible fit result.
//!
//! This is the first integration test that actually exercises the fitting
//! pipeline against a `[covariate_nn]`-bearing model. It validates that
//! the plumbing from PRs #38 (parser), #39 (simulate-path dispatch + mu-ref
//! detection + tv_fn parity) hangs together:
//!
//! - parser registers NN-weight thetas in the optimizer vector
//! - `pk_param_fn` dispatches through the NN forward pass per call
//! - `tv_fn` (eta=0 typical values) returns NN outputs, not zeros
//! - FOCEI inner loop (BFGS on etas) finds finite EBEs
//! - FOCEI outer loop returns a result struct, with NN-weight thetas
//!   updated in the optimizer vector
//!
//! The classical "DCM with mixed effects" workflow lands fully in M2.
//! For now we run a short fit (`maxiter = 5`) — enough to prove the
//! pipeline doesn't crash and that NN-weight thetas move.
//!
//! Gated with `slow-tests` per AGENTS.md: any test that calls `fit()` to
//! convergence is opt-in. This particular test has `maxiter = 5` so it's
//! fast (~1–2 s) but still calls into the optimizer.
//!
//! Run via: `RUSTFLAGS="-Z autodiff=Enable" cargo test --release \
//!   --features nn,slow-tests --test nn_fit_smoke`.

#![cfg(feature = "nn")]

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv};

const NN_DCM_MODEL: &str = r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V1 ~ 0.09
  omega ETA_Q  ~ 0.09
  omega ETA_V2 ~ 0.09
  omega ETA_KA ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V1, Q, V2, KA]
  layers = [3]
  activation = tanh
  output = softplus

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
  maxiter = 5
  covariance = false
"#;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn fit_runs_without_panicking_on_nn_bearing_model() {
    let parsed = parse_full_model(NN_DCM_MODEL).expect("model parses with --features nn");
    let model = parsed.model;
    let options = parsed.fit_options;

    // Load the covariate-bearing dataset (WT + CRCL columns).
    let population = read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads against the NN-bearing model schema");

    let initial_theta = model.default_params.theta.clone();

    // Sanity: NN-weight thetas should sit after the user-declared TVKA.
    // 2 inputs -> 3 hidden -> 5 outputs:
    //   W_1: 3*2 = 6, b_1: 3, W_2: 5*3 = 15, b_2: 5 → 29 weights.
    assert_eq!(model.covariate_nns.len(), 1);
    assert_eq!(model.n_theta, 1 + 29);
    assert_eq!(model.covariate_nns[0].weights_offset, 1);

    // Run the fit. The point is to verify nothing panics and that fit
    // produces a finite OFV.
    let result = fit(&model, &population, &model.default_params, &options).expect("fit returns Ok");

    // OFV must be finite.
    assert!(
        result.ofv.is_finite(),
        "fit returned non-finite OFV: {}",
        result.ofv
    );

    // The optimizer should have nudged at least one NN weight.
    let final_theta = &result.theta;
    let n_w = ferx_core::nn::CovariateMapper::n_weights(&model.covariate_nns[0].mapper);
    let mut moved = false;
    for i in 0..n_w {
        let idx = model.covariate_nns[0].weights_offset + i;
        if (final_theta[idx] - initial_theta[idx]).abs() > 1e-9 {
            moved = true;
            break;
        }
    }
    assert!(
        moved,
        "no NN weight moved during 5 iterations of FOCEI — \
         optimizer is likely not seeing them as live parameters"
    );

    // Per-subject EBEs must be finite (proxy for the inner BFGS being well-behaved).
    for sr in &result.subjects {
        for (j, v) in sr.eta.iter().enumerate() {
            assert!(
                v.is_finite(),
                "subject {} eta[{}] = {} is not finite — inner-loop divergence",
                sr.id,
                j,
                v
            );
        }
    }
}
