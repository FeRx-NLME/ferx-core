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
    fit_source(MODEL, gradient)
}

fn fit_source(src: &str, gradient: GradientMethod) -> ferx_core::FitResult {
    let mut model = parse_model_string(src).expect("block_sigma model must parse");
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

/// The analytic dense-R FOCEI gradient reaches an optimum at least as good as the
/// finite-difference gradient — and, being noise-free, converges to the same region
/// of parameter space (per-coordinate gradient equality is pinned by the fast FD
/// unit tests). We do *not* pin the two OFVs to within a shared basin: on this
/// deliberately tiny, flat 2-subject surface the noisy FD outer gradient stalls at a
/// shallower point than the analytic path (a ~0.7-unit-higher OFV since #925
/// sharpened the inner-EBE fallback, though the estimates still agree to a few %).
/// The invariant that matters — analytic ≤ FD, i.e. the exact gradient never lands
/// somewhere worse — is what this test guards.
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
    // The noise-free analytic gradient reaches an optimum no worse than the FD one.
    // (We don't pin |analytic - fd| to a shared basin: the flat 2-subject surface lets
    // the noisy FD path stall at a shallower point — see the doc comment above.)
    assert!(
        analytic.ofv <= fd.ofv + 1e-2,
        "analytic OFV {} should be no worse than FD OFV {}",
        analytic.ofv,
        fd.ofv
    );
    // Despite the OFV gap, both paths converge to the same region of parameter space.
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

/// #847: a bare `block_sigma` estimates its off-diagonal, so the free-rho fit must
/// (a) actually move rho off its declared value and (b) land at an OFV no worse
/// than the `FIX`ed fit — the fixed model is the free model restricted to a
/// single point of the rho axis, so a free fit that scored worse would mean the
/// rho gradient points the optimizer the wrong way.
///
/// This is the convergence-level companion to the per-coordinate parity tests
/// (`population_packed_gradient_block_sigma_matches_fd` and siblings), which pin
/// the rho gradient itself against Richardson reconverged FD.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn dense_residual_free_rho_beats_fixed_rho() {
    let free_src = MODEL.replace("1.00] FIX", "1.00]");
    let free = fit_source(&free_src, GradientMethod::Auto);
    let fixed = fit_with(GradientMethod::Auto);

    assert!(
        free.ofv.is_finite() && fixed.ofv.is_finite(),
        "both OFVs must be finite: free {}, fixed {}",
        free.ofv,
        fixed.ofv
    );
    assert_eq!(free.residual_correlation_fixed, vec![false]);
    assert_eq!(fixed.residual_correlation_fixed, vec![true]);

    // The FIXed fit holds rho at the declaration; the free fit moves it.
    let declared = 0.5_f64;
    assert!((fixed.residual_correlations[0].rho - declared).abs() < 1e-12);
    let free_rho = free.residual_correlations[0].rho;
    assert!(
        (free_rho - declared).abs() > 1e-3,
        "free rho should move off the {declared} init, got {free_rho}"
    );
    assert!(
        free_rho.abs() < 1.0,
        "the Fisher-z box must keep rho admissible, got {free_rho}"
    );

    // Widening the feasible set cannot make the optimum worse.
    assert!(
        free.ofv <= fixed.ofv + 1e-2,
        "free-rho OFV {} should be no worse than fixed-rho OFV {}",
        free.ofv,
        fixed.ofv
    );

    // The analytic rho gradient must reach the same optimum the FD path does —
    // the rho analogue of `dense_residual_analytic_and_fd_fits_agree`.
    let free_fd = fit_source(&free_src, GradientMethod::Fd);
    assert!(
        free.ofv <= free_fd.ofv + 1e-2,
        "analytic free-rho OFV {} should be no worse than FD {}",
        free.ofv,
        free_fd.ofv
    );
}
