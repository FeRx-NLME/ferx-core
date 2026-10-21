//! Convergence cross-check for the analytic correlated-residual (`block_sigma`,
//! dense-R) FOCEI/FOCE gradients (issue #627).
//!
//! Commit (b) of #627 wires an analytic outer + inner gradient for `block_sigma`
//! models — previously both loops ran finite differences (#620 scoped the dense
//! `R` to the marginal objective only). Per-coordinate agreement with central FD
//! is pinned by the fast unit tests
//! (`kernel_block_sigma_matches_reconverged_fd` and
//! `kernel_serial_residual_correlation_matches_reconverged_fd` in
//! `sens/ode_adjoint_tests.rs`, `dense_residual_inner_grad_matches_fd`, and the FOCE /
//! ExpressionScale variants). This slow test pins the *fit*: the analytic-gradient path must
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

/// The analytic dense-R FOCEI gradient and the **reconverged** finite-difference gradient
/// (`reconverge_gradient_interval = 1`: every subject's EBE re-solved at every FD
/// perturbation — the exact marginal gradient the analytic one must equal) must drive
/// L-BFGS to the same optimum. The check is two-sided: a one-sided "analytic ≤ FD" bound
/// cannot see an analytic gradient that lands *lower* in the wrong place, and the held-EBE
/// FD fit (`reconverge_gradient_interval = 0`) is not a valid reference — it omits the
/// EBE-response term (#1552; Codex review on #1556).
///
/// Measured on this fixture (Linux): analytic OFV 17.630655436615, reconverged
/// 17.630655456710, |Δ| = 2.0e-8 with the analytic fit the lower; worst θ relative
/// difference 2.3e-5 (θ[1], along a flat ridge); ω² on the 6.1e-6 guard in both. Before
/// #2093's inner-solver changes the pair stopped at 17.630655452624 / 17.630655451731 and
/// 1.5e-7 in θ — the analytic fit now goes 1.6e-8 lower than either did. The held-EBE fit
/// lands at 17.630655604168. Bounds: 1e-7 on the OFV (5× headroom) and 1e-4 on θ (4.3×).
/// The FD reference's stopping point sets that gap, not the analytic gradient: NLopt
/// L-BFGS ignores `outer_gtol` (1e-6, 1e-7 and 1e-8 give the same bits, measured), so it
/// cannot be tightened from here. The sweep below was measured against the earlier 1e-5 θ
/// bound; at 1e-4 the θ₁ × 1.5 row still fails both bounds (θ 1.5e-4, |Δ| 1.7e-6).
///
/// Mutation sweep on `subject_packed_gradient`, each against these bounds:
/// - θ₀ gradient sign-flipped → |Δ| = 0.56, θ 7.0e-2 (both bounds fail; so did the old check);
/// - ω gradient zeroed → |Δ| = 0.75, θ 2.2e-2 (both fail; so did the old check);
/// - θ₁ gradient × 1.5 → |Δ| = 1.7e-6, θ 1.5e-4 — **passed the old one-sided check**, fails
///   both bounds here;
/// - ω gradient × 0.5 → |Δ| = 2.7e-10, θ 2.9e-5 — only the 1e-5 θ bound saw it, and the
///   1e-4 bound does **not**: ω² has collapsed onto its guard, so its gradient is nearly
///   unobservable on this 3-subject fixture, and the ridge gap (2.3e-5) is now the same size
///   as the mutation's. The dense-R gradient is pinned coordinate by coordinate against the
///   reconverged FD gradient at Tier 1 (e.g. `kernel_block_sigma_matches_reconverged_fd`
///   in `sens/ode_adjoint_tests.rs`); this convergence fixture cannot see an ω error.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn dense_residual_analytic_and_fd_fits_agree() {
    let analytic = fit_with(GradientMethod::Auto);
    let reconverged = fit_source_reconverging(MODEL, GradientMethod::Fd, 1);

    assert!(
        analytic.ofv.is_finite() && reconverged.ofv.is_finite(),
        "both OFVs must be finite: analytic {}, reconverged FD {}",
        analytic.ofv,
        reconverged.ofv
    );
    assert!(analytic.converged, "analytic fit did not converge");
    assert!(reconverged.converged, "reconverged-FD fit did not converge");
    // Two-sided: measured |Δ| = 2.0e-8 (#2093; 8.9e-10 before), bound 1e-7 (any comparison against NaN is
    // false, so a NaN OFV fails here too).
    assert!(
        (analytic.ofv - reconverged.ofv).abs() < 1e-7,
        "analytic OFV {} vs reconverged-FD OFV {}: |Δ| = {:e} exceeds 1e-7",
        analytic.ofv,
        reconverged.ofv,
        (analytic.ofv - reconverged.ofv).abs()
    );
    // The analytic fit is at least as low as the FD reference, to the reference's own
    // resolution: measured analytic − FD = −2.0e-8 (#2093; before it, on #2094's tree, +8.9e-10).
    assert!(
        analytic.ofv <= reconverged.ofv + 1e-9,
        "analytic OFV {} stops above the reconverged-FD OFV {}",
        analytic.ofv,
        reconverged.ofv
    );
    // Same optimum, not just the same objective value. The optimum sits on a flat ridge
    // along θ[1]: the two fits stop 2.0e-8 OFV apart and 2.3e-5 apart in θ[1] (measured on
    // #2093, worst θ; the FD reference is the higher of the two, so the gap is its stopping
    // point, not the analytic gradient's). A 2.0e-8 objective gap resolves θ only to
    // √(2·2e-8 / curvature) along a ridge this flat, so the bound is 1e-4 — 4.3× the measured
    // worst.
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for k in 0..analytic.theta.len() {
        assert!(
            rel(analytic.theta[k], reconverged.theta[k]) < 1e-4,
            "theta[{k}] analytic {} vs reconverged FD {}",
            analytic.theta[k],
            reconverged.theta[k]
        );
    }
    assert!(
        rel(analytic.omega[(0, 0)], reconverged.omega[(0, 0)]) < 1e-5,
        "omega^2 analytic {} vs reconverged FD {}",
        analytic.omega[(0, 0)],
        reconverged.omega[(0, 0)]
    );
}

/// #847: a bare `block_sigma` estimates its off-diagonal, so the free-rho fit must
/// (a) actually move rho off its declared value, (b) land at an OFV no worse
/// than the `FIX`ed fit — `FIX` holds the whole block (both sigmas and rho) at its
/// declaration, a point the free model contains — and (c) reach the OFV the
/// reconverged-FD gradient reaches from the same start (#1552).
///
/// This is the convergence-level companion to the per-coordinate parity tests
/// (`kernel_block_sigma_matches_reconverged_fd` and siblings in
/// `sens/ode_adjoint_tests.rs`), which pin the rho gradient itself against Richardson
/// reconverged FD.
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
