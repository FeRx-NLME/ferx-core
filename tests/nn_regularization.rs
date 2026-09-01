//! Integration tests for the covariate-NN (DCM) regularization penalties:
//! L2 (weight-decay, `nn_l2`) and smoothness/curvature (`nn_smooth`).
//!
//! Coverage of the task's three validation requirements:
//!   (a) λ = 0 is a strict no-op — the [`NnRegularizer`] is inactive, its
//!       penalty and gradient contribution are exactly 0, and a λ = 0 fit is
//!       byte-identical to the unregularized baseline.
//!   (b) The analytic penalty gradient matches central finite differences at
//!       λ > 0 (checked here at the whole-`NnRegularizer` level; the per-kernel
//!       L2 / curvature FD parity lives in `src/nn/mod.rs` unit tests).
//!   (c) On a fitted model, the learned covariate→modulator variance shrinks
//!       toward 0 as the L2 strength grows.
//!
//! The `NnRegularizer`-level tests (a)/(b) call no fit and stay fast (Tier 2);
//! the fit-based shrinkage / byte-identity checks that run to convergence are
//! gated behind `slow-tests` per CLAUDE.md.
//!
//! Run via:
//!   cargo test --features nn --test nn_regularization                 # fast
//!   cargo test --features nn,slow-tests --test nn_regularization      # + slow

#![cfg(feature = "nn")]

use ferx_core::nn::{CovariateMapper, NnRegularizer};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitOptions, Population};
use ferx_core::{fit, read_nonmem_csv, CompiledModel};

/// Two-cpt oral DCM with a small MLP mapping (WT, CRCL) → the five PK params.
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
  # Normalization is load-bearing, not cosmetic. `WT` runs 45–90 and `CRCL`
  # 46–82 on this dataset, so on raw inputs the Glorot-initialised first layer
  # produces pre-activations spanning ±170 — `tanh` is clipped at ±1 for every
  # subject and the network is constant *by saturation* before the fit starts.
  # The λ = 0 baseline then looks flat for the wrong reason (see the shrinkage
  # test below, which needs a non-degenerate unregularized fit to compare
  # against). z-scoring puts layer 1 in tanh's responsive band, and matches the
  # space the `nn_smooth` curvature grid already works in.
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

/// Variance of the NN's CL modulator (output 0) across subjects, evaluated at a
/// given theta. Higher = more covariate-driven spread in the learned curve.
fn cl_modulator_variance(model: &CompiledModel, population: &Population, theta: &[f64]) -> f64 {
    let nn = &model.covariate_nns[0];
    let n_w = nn.mapper.n_weights();
    let w = &theta[nn.weights_offset..nn.weights_offset + n_w];
    let cls: Vec<f64> = population
        .subjects
        .iter()
        .map(|s| nn.mapper.forward_raw(w, &s.covariates).expect("forward")[0])
        .collect();
    let n = cls.len() as f64;
    let mean = cls.iter().sum::<f64>() / n;
    cls.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / n
}

/// L2 norm of the NN's *weight matrices* (biases excluded) at a given theta —
/// the quantity `nn_l2` directly shrinks. A flatter covariate→modulator curve
/// (and thus smaller modulator variance) follows from smaller weights, so this
/// is the robust, monotone signal for the shrinkage effect.
fn weight_block_sq_norm(model: &CompiledModel, theta: &[f64]) -> f64 {
    let nn = &model.covariate_nns[0];
    let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
    nn.mapper
        .mlp()
        .weight_param_indices()
        .iter()
        .map(|&i| w[i] * w[i])
        .sum()
}

// ---------------------------------------------------------------------------
// (a)/(b): NnRegularizer-level, no fit — fast (Tier 2)
// ---------------------------------------------------------------------------

#[test]
fn regularizer_lambda_zero_is_noop() {
    let (model, mut options, population) = load();
    options.nn_l2_lambda = 0.0;
    options.nn_smooth_lambda = 0.0;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(!reg.is_active(), "λ = 0 regularizer must be inactive");

    let theta = model.default_params.theta.clone();
    assert_eq!(reg.penalty_value(&theta), 0.0, "λ = 0 penalty must be 0");

    let mut grad = vec![0.0; theta.len()];
    reg.add_packed_gradient(&theta, &mut grad);
    assert!(
        grad.iter().all(|&g| g == 0.0),
        "λ = 0 gradient contribution must be exactly 0"
    );
}

#[test]
fn regularizer_penalty_gradient_matches_fd() {
    let (model, mut options, population) = load();
    options.nn_l2_lambda = 1e-2;
    options.nn_smooth_lambda = 1e-1;
    let reg = NnRegularizer::build(&model, &population, &options);
    assert!(reg.is_active());

    let theta = model.default_params.theta.clone();
    let mut grad = vec![0.0; theta.len()];
    reg.add_packed_gradient(&theta, &mut grad);

    // Only the NN-weight coordinates carry a penalty gradient; everything else
    // (omegas, sigma) must be untouched.
    let nn = &model.covariate_nns[0];
    let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
    for (k, &g) in grad.iter().enumerate() {
        if k < w_lo || k >= w_hi {
            assert_eq!(g, 0.0, "non-NN coordinate {k} must have zero penalty grad");
        }
    }

    // Central FD of penalty_value w.r.t. each NN weight theta.
    let eps = 1e-6;
    let mut p = theta.clone();
    for k in w_lo..w_hi {
        let saved = p[k];
        p[k] = saved + eps;
        let vp = reg.penalty_value(&p);
        p[k] = saved - eps;
        let vm = reg.penalty_value(&p);
        p[k] = saved;
        let fd = (vp - vm) / (2.0 * eps);
        let tol = 1e-6 + 1e-5 * fd.abs();
        assert!(
            (grad[k] - fd).abs() <= tol,
            "penalty grad mismatch at weight {k}: analytic {}, fd {}",
            grad[k],
            fd
        );
    }
}

/// A couple of outer iterations with both penalties on — proves the optimizer
/// wiring (penalized objective value + matching analytic gradient) runs end to
/// end without panicking and returns a finite OFV. Fast (Tier 2).
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

// ---------------------------------------------------------------------------
// (a)/(c): fit-based — slow (Tier 3)
// ---------------------------------------------------------------------------

/// λ = 0 must reproduce the unregularized fit byte-for-byte (the no-op
/// guarantee). Runs two short fits and asserts identical OFV.
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

/// (c) Growing the L2 strength shrinks the NN weights, flattening the learned
/// covariate→modulator curve so the across-subject modulator variance is driven
/// toward 0.
///
/// The robust, deterministic signal is the **fitted weight-block norm**: L2 adds
/// `2λw` to the weight gradient, so a heavier λ pulls the optimum's weights
/// closer to 0 (monotonically).
///
/// The modulator variance is the *effect* being claimed, and it is only a
/// meaningful check when the λ = 0 fit actually produces spread to remove. On
/// this null-covariate dataset the unregularized fit invents a large spurious
/// CL modulator variance (~480 across subjects) — precisely the overfitting
/// `nn_l2` exists to suppress — and L2 collapses it to ~0. Asserting both ends
/// keeps the oracle non-degenerate in the sense CLAUDE.md requires: an assertion
/// that the regularized modulator is flat is worthless if the unregularized one
/// was flat too. It was, on raw inputs, because the network was saturated rather
/// than because it had learned nothing — see the `center`/`scale` note on the
/// fixture.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn l2_shrinks_weights_and_modulator_variation() {
    let (model, options, population) = load();

    let fit_at = |lambda: f64| -> Vec<f64> {
        let mut o = options.clone();
        o.nn_l2_lambda = lambda;
        fit(&model, &population, &model.default_params, &o)
            .unwrap_or_else(|e| panic!("fit at λ={lambda} failed: {e}"))
            .theta
    };

    let t0 = fit_at(0.0);
    let t_mid = fit_at(5.0);
    let t_big = fit_at(100.0);

    let (n0, n_mid, n_big) = (
        weight_block_sq_norm(&model, &t0),
        weight_block_sq_norm(&model, &t_mid),
        weight_block_sq_norm(&model, &t_big),
    );
    let (v0, v_mid, v_big) = (
        cl_modulator_variance(&model, &population, &t0),
        cl_modulator_variance(&model, &population, &t_mid),
        cl_modulator_variance(&model, &population, &t_big),
    );
    eprintln!(
        "weight ‖W‖²: λ=0 {n0:.5}, λ=5 {n_mid:.5}, λ=100 {n_big:.5}\n\
         CL modulator var: λ=0 {v0:.6}, λ=5 {v_mid:.6}, λ=100 {v_big:.6}"
    );

    // Decisive signal: the fitted weight norm shrinks strongly and monotonically
    // with λ (observed here ~2159 → ~0.004 → ~0.003). This is the guaranteed
    // mechanism by which L2 flattens the covariate→modulator map.
    assert!(
        n_mid <= n0 + 1e-9 && n_big <= n_mid + 1e-9,
        "weight norm must be non-increasing in λ (‖W‖²: {n0:.5} → {n_mid:.5} → {n_big:.5})"
    );
    assert!(
        n_big < n0 * 0.5,
        "heavy L2 (λ=100) must more than halve the fitted weight norm ({n_big:.5} vs λ=0 {n0:.5})"
    );

    // The unregularized fit must actually overfit — otherwise the flatness check
    // below passes against a baseline that was already flat and proves nothing.
    assert!(
        v0 > 1.0,
        "the λ=0 fit must invent real spurious CL spread for this test to have a \
         baseline to remove (var {v0:.6}); a near-zero unregularized variance means \
         the fixture is degenerate, not that L2 worked"
    );

    // The effect: L2 collapses that spurious spread toward a constant map. Both
    // regularized fits must be flat; their ordering *relative to each other* is
    // not asserted, because at ~1e-9 the difference between them is float noise
    // rather than an effect of λ.
    assert!(
        v_mid < 1e-3 && v_big < 1e-3,
        "L2 must collapse the spurious CL modulator spread ({v0:.6} → {v_mid:.6} → {v_big:.6})"
    );
}
