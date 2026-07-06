//! Convergence + NONMEM cross-check for the M3 BLOQ likelihood with the analytic
//! M3 inner EBE gradient (issue #367).
//!
//! `analytic_eta_nll_gradient` now has a closed-form η-gradient for the M3
//! censored data term `−logΦ((LLOQ−f)/√V)` (inverse-Mills-ratio coefficient), so
//! the inner EBE loop on `bloq_method = m3` fits runs without finite differencing.
//! The converged EBEs — and therefore the fit — are unchanged versus the FD inner
//! gradient (verified by the `analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq`
//! unit test); this slow test pins the full M3 fit end-to-end and is the NONMEM
//! cross-check hand-off.
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test bloq_convergence
//!
//! ## NONMEM cross-check
//!
//! ferx model: `examples/warfarin_bloq.ferx` (one_cpt_oral, `DV ~
//! proportional(PROP_ERR)`, `bloq_method = m3`, LLOQ = 2.0). NONMEM reproduces M3
//! with the F_FLAG mixed-likelihood pattern under LAPLACE — censored rows return
//! `Φ((LLOQ−IPRED)/SD)` as a likelihood; the proportional SD lives in a THETA so
//! `EPS(1) ~ N(0,1)` (SIGMA fixed). Control stream + data:
//!
//!   tests/nonmem/warfarin_bloq.ctl   (F_FLAG / PHI, METHOD=1 LAPLACE INTER)
//!   tests/nonmem/warfarin_bloq.csv   (copy of data/warfarin_bloq.csv; DV on
//!                                     CENS=1 rows carries the LLOQ)
//!
//! NONMEM 7.5.1 (MINIMIZATION SUCCESSFUL, `tests/nonmem/warfarin_bloq.lst`)
//! reaches TVCL 0.132801 / TVV 7.73139 / TVKA 0.809824 / PROP(SD) 0.010760 /
//! ω 0.028849 / 0.009544 / 0.335772, which ferx matches to ~4 significant figures:
//!
//! | Parameter   | ferx (analytic, reconverged) | NONMEM 7.5.1 |
//! |-------------|------------------------------|--------------|
//! | TVCL        | 0.132810                     | 0.132801     |
//! | TVV         | 7.731954                     | 7.73139      |
//! | TVKA        | 0.809961                     | 0.809824     |
//! | PROP_ERR(SD)| 0.010764                     | 0.010760     |
//! | ω²(CL/V/KA) | 0.02885 / 0.00954 / 0.33577  | 0.02885 / 0.00954 / 0.33577 |
//!
//! The OFV is *not* compared directly: NONMEM's F_FLAG likelihood objective for
//! the censored rows carries a different additive constant than ferx's M3 term
//! (ferx −217.18 vs NONMEM −216.79), so the cross-check pins the MLE, not the OFV.
//!
//! Convergence note: both FOCEI-M3 (`prepare`'s M3 branch) and FOCE-M3
//! (`subject_packed_gradient_foce`, censored rows excluded from R̃) now have exact
//! closed-form censored outer gradients, so a gradient optimizer reaches the true
//! minimum directly. Without that, the fixed-EBE FD fallback was biased and a
//! gradient optimizer stalled on this flat KA ridge at TVKA ≈ 1.10 / OFV ≈ −213.8;
//! BOBYQA reconverges anyway.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

/// The M3 analytic inner-gradient path (with the auto-reconverged outer gradient)
/// converges on the real warfarin BLOQ fit and recovers NONMEM's MLE. Exercises
/// `analytic_eta_nll_gradient`'s M3 branch in a full gradient-based fit.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored M3 cross-check: opt in with --features slow-tests"
)]
fn bloq_m3_analytic_lbfgs_matches_nonmem() {
    let model = parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
        .expect("warfarin BLOQ model must parse");
    assert!(
        matches!(model.bloq_method, ferx_core::BloqMethod::M3),
        "model must be M3"
    );
    let population = read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
        .expect("warfarin BLOQ data must load");
    assert!(
        population
            .subjects
            .iter()
            .any(|s| s.cens.iter().any(|&c| c != 0)),
        "data must contain censored rows"
    );

    // Gradient-based path: built-in L-BFGS outer on the analytic FOCEI-M3 outer
    // gradient (+ analytic inner M3 gradient). Inner solver stays at the default
    // Auto/BFGS — the choice doesn't change the EBE/gradient, and pinning it
    // mutates a process-global that races sibling tests under parallel execution.
    let mut opts = FitOptions::default();
    opts.optimizer = Optimizer::Lbfgs;
    opts.inner_tol = 1e-8;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("analytic M3 fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );

    // NONMEM 7.5.1 LAPLACE M3 MLE (tests/nonmem/warfarin_bloq.lst). The OFV is not
    // compared (F_FLAG likelihood constant offset); the estimates are pinned.
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    const NM_TVCL: f64 = 0.132801;
    const NM_TVV: f64 = 7.73139;
    const NM_TVKA: f64 = 0.809824;
    const NM_PROP_SD: f64 = 0.0107600;
    const NM_OM_CL: f64 = 0.0288494;
    const NM_OM_V: f64 = 0.00954401;
    const NM_OM_KA: f64 = 0.335772;
    assert!(
        rel(result.theta[0], NM_TVCL) < 0.01,
        "TVCL {} vs NM {NM_TVCL}",
        result.theta[0]
    );
    assert!(
        rel(result.theta[1], NM_TVV) < 0.01,
        "TVV {} vs NM {NM_TVV}",
        result.theta[1]
    );
    assert!(
        rel(result.theta[2], NM_TVKA) < 0.02,
        "TVKA {} vs NM {NM_TVKA}",
        result.theta[2]
    );
    assert!(
        rel(result.sigma[0], NM_PROP_SD) < 0.02,
        "PROP {} vs NM {NM_PROP_SD}",
        result.sigma[0]
    );
    let om: Vec<f64> = (0..3).map(|i| result.omega[(i, i)]).collect();
    assert!(
        rel(om[0], NM_OM_CL) < 0.05,
        "ω²(CL) {} vs NM {NM_OM_CL}",
        om[0]
    );
    assert!(
        rel(om[1], NM_OM_V) < 0.05,
        "ω²(V) {} vs NM {NM_OM_V}",
        om[1]
    );
    assert!(
        rel(om[2], NM_OM_KA) < 0.05,
        "ω²(KA) {} vs NM {NM_OM_KA}",
        om[2]
    );
}

/// FOCE-M3 (no interaction) on the analytic FOCE censored gradient.
///
/// **There is no NONMEM anchor for this estimator, by construction.** NONMEM's M3 BLOQ
/// is the F_FLAG mixed likelihood, which *requires* the LAPLACE objective (`$EST METHOD=1
/// LAPLACE`); NONMEM has no genuine first-order, non-interaction "FOCE M3" method to
/// compare against. NONMEM's LAPLACE-M3 corresponds to ferx's **FOCEI**-M3, which is what
/// `bloq_m3_analytic_lbfgs_matches_nonmem` cross-checks against `warfarin_bloq.lst`.
///
/// ferx additionally offers a Sheiner–Beal **FOCE**-M3 (no interaction): the censored
/// rows enter the marginal as `−logΦ((LLOQ−f̂)/√R⁰)` (excluded from R̃, the population
/// variance). It is a genuinely different, ferx-specific optimum from FOCEI-M3
/// (interaction shifts TVKA up ~0.71 → ~0.81), so it is validated by **self-consistency**
/// rather than a NONMEM cross-check: the analytic FOCE-M3 outer gradient must drive the
/// optimizer to the same MLE as the finite-difference gradient of the *identical* marginal.
/// (The censored inner-EBE gradient is separately FD-checked by the
/// `analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq` unit test.)
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bloq_m3_foce_analytic_matches_fd() {
    let model = parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
        .expect("warfarin BLOQ model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
        .expect("warfarin BLOQ data must load");

    // FOCE (no interaction) M3, driven by the analytic vs the finite-difference outer
    // gradient of the same Sheiner–Beal marginal. Both legs share optimizer/inits, so the
    // only difference is how the gradient is formed — they must land on the same optimum.
    let run = |gm: ferx_core::GradientMethod| -> ferx_core::FitResult {
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::Foce;
        opts.optimizer = Optimizer::Lbfgs;
        opts.gradient_method = gm;
        opts.inner_tol = 1e-8;
        opts.outer_maxiter = 300;
        opts.run_covariance_step = false;
        opts.verbose = false;
        fit(&model, &population, &model.default_params, &opts)
            .expect("analytic/FD FOCE-M3 fit must succeed")
    };
    let analytic = run(ferx_core::GradientMethod::Auto);
    let fd = run(ferx_core::GradientMethod::Fd);

    assert!(
        analytic.ofv.is_finite() && fd.ofv.is_finite(),
        "OFV must be finite, got analytic {} / FD {}",
        analytic.ofv,
        fd.ofv
    );
    // Same marginal, two gradient methods ⇒ same optimum (allowing for optimizer-path
    // noise on the flat KA ridge).
    assert!(
        (analytic.ofv - fd.ofv).abs() < 1.0,
        "analytic OFV {:.4} vs FD OFV {:.4} should agree (same FOCE-M3 marginal)",
        analytic.ofv,
        fd.ofv
    );
    for k in 0..analytic.theta.len() {
        let (a, f) = (analytic.theta[k], fd.theta[k]);
        assert!(
            (a - f).abs() <= 0.03 * f.abs().max(1e-3),
            "theta[{k}]: analytic {a:.5} vs FD {f:.5} diverge beyond 3%"
        );
    }
    // FOCE-M3 must be a distinct optimum from FOCEI-M3 (interaction shifts TVKA to ~0.81).
    assert!(
        analytic.theta[2] < 0.76,
        "FOCE TVKA {} should be well below FOCEI ~0.81",
        analytic.theta[2]
    );
}
