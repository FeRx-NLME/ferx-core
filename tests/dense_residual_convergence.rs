//! Convergence cross-check for the analytic correlated-residual (`block_sigma`,
//! dense-R) FOCEI/FOCE gradients (issue #627).
//!
//! Commit (b) of #627 wires an analytic outer + inner gradient for `block_sigma`
//! models — previously both loops ran finite differences (#620 scoped the dense
//! `R` to the marginal objective only). Per-coordinate agreement with central FD
//! is pinned by the fast unit tests
//! (`population_packed_gradient_block_sigma_matches_fd`,
//! `dense_residual_inner_grad_matches_fd`, and the FOCE / ExpressionScale
//! variants). This slow test pins the *fit*: the analytic-gradient path must
//! converge to the **same** optimum (OFV + estimates) as the finite-difference
//! path, i.e. swapping in the analytic gradient does not move the minimum.
//!
//! No new NONMEM run is needed: the `block_sigma` OFV itself is already
//! NONMEM-anchored (`examples/correlated_residual_combined.ferx`, OFV 18.722087,
//! see `docs/model-file/error-model.qmd`); this test anchors the *gradient* by
//! self-consistency against the FD fit that was validated there.
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test dense_residual_convergence

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, FitOptions, GradientMethod, Optimizer};
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
[fit_options]
  method = focei
";

fn fit_with(gradient: GradientMethod) -> ferx_core::FitResult {
    let mut model = parse_model_string(MODEL).expect("block_sigma model must parse");
    assert!(
        !model.residual_correlations.is_empty(),
        "model must carry a residual correlation"
    );
    model.gradient_method = gradient;
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("correlated residual data must load");

    let mut opts = FitOptions::default();
    opts.optimizer = Optimizer::Lbfgs;
    opts.inner_tol = 1e-9;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &population, &model.default_params, &opts).expect("block_sigma fit must succeed")
}

/// The analytic dense-R FOCEI gradient converges to the same basin as the
/// finite-difference gradient — and, being noise-free, reaches an optimum at least
/// as good. Swapping the gradient in does not move the minimum to a different
/// region (per-coordinate gradient equality is pinned by the fast FD unit tests).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn dense_residual_analytic_and_fd_fits_agree() {
    let analytic = fit_with(GradientMethod::Auto);
    let fd = fit_with(GradientMethod::Fd);

    assert!(
        analytic.ofv.is_finite() && fd.ofv.is_finite(),
        "both OFVs must be finite: analytic {}, fd {}",
        analytic.ofv,
        fd.ofv
    );
    // Same basin (the tiny 2-subject surface is flat, so the two optimizers stop at
    // slightly different points), and the noise-free analytic gradient reaches an
    // optimum no worse than the FD one.
    assert!(
        analytic.ofv <= fd.ofv + 1e-2,
        "analytic OFV {} should be no worse than FD OFV {}",
        analytic.ofv,
        fd.ofv
    );
    assert!(
        (analytic.ofv - fd.ofv).abs() < 0.1,
        "analytic OFV {} vs FD OFV {} land in different basins",
        analytic.ofv,
        fd.ofv
    );
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for k in 0..analytic.theta.len() {
        assert!(
            rel(analytic.theta[k], fd.theta[k]) < 5e-2,
            "theta[{k}] analytic {} vs FD {}",
            analytic.theta[k],
            fd.theta[k]
        );
    }
}
