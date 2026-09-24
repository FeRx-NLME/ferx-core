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
    fit_source_reconverging(src, gradient, 0)
}

/// `fit_source` with `reconverge_gradient_interval = interval`. `1` re-solves every
/// subject's EBE at each FD perturbation on every outer evaluation — the reconverged
/// FD gradient, i.e. the exact marginal gradient the analytic one must equal.
fn fit_source_reconverging(
    src: &str,
    gradient: GradientMethod,
    interval: usize,
) -> ferx_core::FitResult {
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
    opts.reconverge_gradient_interval = interval;
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
/// (a) actually move rho off its declared value, (b) land at an OFV no worse
/// than the `FIX`ed fit — `FIX` holds the whole block (both sigmas and rho) at its
/// declaration, a point the free model contains — and (c) reach the OFV the
/// reconverged-FD gradient reaches from the same start (#1552).
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

    // The analytic rho gradient must reach the optimum the *reconverged* FD gradient
    // reaches — the rho analogue of `dense_residual_analytic_and_fd_fits_agree`, but
    // against the exact oracle rather than the fixed-EBE one (#1552).
    //
    // Why not the default `GradientMethod::Fd` fit: without reconvergence that fit takes
    // the held-EBE gradient, which omits the EBE response — a different, inexact
    // gradient, so which point it wanders to on this surface says nothing about the
    // analytic one. The surface is degenerate: omega collapses onto its guard (99%
    // shrinkage) in every fit, and the objective has several boundary optima. Measured
    // on this fixture (`main@293ac058`, #1552):
    //
    //   analytic (auto)             -14.857433  rho -0.48, PROP_ERR on its lower guard
    //   reconverged FD (interval 1) -14.857829  rho -0.44, PROP_ERR on its lower guard
    //   held-EBE FD (default fd)    -15.302520  rho +0.995 on the Fisher-z guard, not converged
    //   held-EBE FD before #1531    -14.865891  rho frozen at exactly 0.5
    //
    // The held-EBE reference was never an oracle here: until #1531 its closed form built
    // a diagonal `R` and left the packed rho coordinate at zero, so that fit could not
    // move rho at all, and the old one-sided `analytic <= fd + 1e-2` passed by 1.5e-3.
    // Once #1531 sent `block_sigma` to the FD fallback its rho moved, and ran to rho -> 1.
    //
    // Two-sided, because a broken analytic rho block can land *lower*. Measured with the
    // analytic rho component of `subject_packed_gradient` mutated (#1552):
    //
    //   sign-flipped  -15.294935  gap 4.37e-1  -- passed the old one-sided check
    //   zeroed        -15.289427  gap 4.32e-1  (also caught by "rho should move" above)
    //
    // The realised gap is 3.96e-4: the two paths park rho at different points of the
    // flat PROP_ERR -> 0 ridge, where rho is unidentified. The bound gives it 5x headroom
    // and sits two orders of magnitude under either mutation.
    let free_reconverged = fit_source_reconverging(&free_src, GradientMethod::Fd, 1);
    assert!(
        free_reconverged.ofv.is_finite(),
        "reconverged-FD OFV must be finite, got {}",
        free_reconverged.ofv
    );
    let gap = (free.ofv - free_reconverged.ofv).abs();
    assert!(
        gap < 2e-3,
        "analytic free-rho OFV {} should match the reconverged-FD fit's {} (gap {gap:.3e})",
        free.ofv,
        free_reconverged.ofv
    );
}
