//! Convergence + NONMEM cross-check for constant output scaling (`[scaling]
//! obs_scale = k`) on the analytic sensitivity provider (issue #367).
//!
//! The provider applies the constant divisor to the prediction *and* its η/θ jet
//! in closed form (`pk::apply_scaling` does `pred /= k`; the provider divides
//! `f` and every derivative by `k`). This test fits warfarin with `obs_scale =
//! 10` and **additive** error — additive on purpose, because a proportional
//! error model is invariant to a constant output scale and so could not detect a
//! scaling bug.
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test scaling_convergence
//!
//! ## NONMEM cross-check
//!
//! ferx `obs_scale = 10` divides the concentration prediction by 10. NONMEM
//! reproduces that with `S2 = V*10` (ADVAN2 TRANS2: `F = A2/S2 = (A2/V)/10`).
//! The control stream and the scaled dataset live next to this file:
//!
//!   tests/nonmem/warfarin_scaled.ctl   (S2 = V*10, additive error)
//!   tests/nonmem/warfarin_scaled.csv   (warfarin DV / 10, so the scaled
//!                                       prediction and the data share a scale)
//!
//! Run on the NONMEM host:  `nmfe75 warfarin_scaled.ctl warfarin_scaled.lst`
//!
//! ferx (this engine) converges as below — the default gradient-free BOBYQA and
//! the gradient-based analytic L-BFGS path agree (the L-BFGS path is the one the
//! scaling jet transform feeds; both see the same scaled objective):
//!
//! | Parameter   | ferx (analytic L-BFGS) | ferx (default BOBYQA) |
//! |-------------|------------------------|-----------------------|
//! | OFV         | −740.838               | −740.762              |
//! | TVCL        | 0.132956               | 0.133164              |
//! | TVV         | 7.72748                | 7.70177               |
//! | TVKA        | 0.810476               | 0.826152              |
//! | ADD (SD)    | 0.008687               | 0.008727              |
//! | ω²(CL/V/KA) | 0.02876 / 0.00947 / 0.33549 |  0.02726 / 0.01031 / 0.33202 |
//!
//! NONMEM 7.5.1 (MINIMIZATION SUCCESSFUL) reaches OFV −740.8376 / TVCL 0.132943
//! / TVV 7.72787 / TVKA 0.810919 / SIGMA(1,1) 7.54671e-5 (SD 0.008687) — matching
//! ferx's analytic L-BFGS to ~5 significant figures. The cross-check assertions
//! at the end of the test pin this (`tests/nonmem/warfarin_scaled.lst`).

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions, Optimizer};
use std::path::Path;

/// Expression `obs_scale` (`obs_scale = TVSCALE`, TVSCALE fixed at 10) routed
/// through the differentiable scale program must reproduce the constant-scale
/// NONMEM reference (`tests/nonmem/warfarin_scaled.lst`). This exercises the
/// `ExpressionScale` analytic path — the scale's `∂/∂(θ,η)` jet is evaluated by a
/// `Dual2`-differentiable bytecode program (not a constant divisor) — and, with
/// TVSCALE fixed, the optimum is identical to the scalar case (#367).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored expression-scaling cross-check: opt in with --features slow-tests"
)]
fn expression_scale_matches_nonmem() {
    let model = parse_model_file(Path::new("examples/warfarin_exprscale_additive.ferx"))
        .expect("expression-scaled model must parse");
    assert!(
        matches!(
            model.scaling,
            ferx_core::ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ),
        "model must carry a differentiable ExpressionScale program"
    );
    let population = read_nonmem_csv(Path::new("tests/nonmem/warfarin_scaled.csv"), None, None)
        .expect("scaled warfarin data must load");

    let mut opt = FitOptions::default();
    opt.optimizer = Optimizer::Lbfgs;
    opt.inner_tol = 1e-8;
    opt.outer_maxiter = 300;
    opt.run_covariance_step = false;
    opt.verbose = false;
    let r = fit(&model, &population, &model.default_params, &opt)
        .expect("analytic expression-scale fit must succeed");
    assert!(r.ofv.is_finite(), "OFV must be finite, got {}", r.ofv);

    // Same NONMEM 7.5.1 reference as the constant-scale case (S2 = V*10):
    // the differentiable scale program reproduces it to ~5 significant figures.
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    const NM_OFV: f64 = -740.83761;
    const NM_TVCL: f64 = 0.132943;
    const NM_TVV: f64 = 7.72787;
    const NM_TVKA: f64 = 0.810919;
    const NM_ADD_SD: f64 = 0.0086872;
    assert!(
        (r.ofv - NM_OFV).abs() < 0.1,
        "OFV {:.4} vs NONMEM {NM_OFV:.4}",
        r.ofv
    );
    assert!(rel(r.theta[0], NM_TVCL) < 0.01, "TVCL {}", r.theta[0]);
    assert!(rel(r.theta[1], NM_TVV) < 0.01, "TVV {}", r.theta[1]);
    assert!(rel(r.theta[2], NM_TVKA) < 0.02, "TVKA {}", r.theta[2]);
    assert!(rel(r.sigma[0], NM_ADD_SD) < 0.02, "ADD {}", r.sigma[0]);
    // TVSCALE stays fixed at its declared value.
    assert!(
        (r.theta[3] - 10.0).abs() < 1e-9,
        "TVSCALE must stay fixed at 10, got {}",
        r.theta[3]
    );
}

/// The analytic scaling path converges and the gradient-based (analytic
/// L-BFGS) and gradient-free (default BOBYQA) optimizers agree — the scaling jet
/// transform is self-consistent with the scaled production objective. Pins ferx
/// estimates as a regression guard; the NONMEM cross-check is the documented
/// hand-off above.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn scaling_obs_scale_additive_converges_and_agrees() {
    let model = parse_model_file(Path::new("examples/warfarin_scaled_additive.ferx"))
        .expect("scaled additive model must parse");
    assert!(
        matches!(
            model.scaling,
            ferx_core::ScalingSpec::ScalarScale(k) if (k - 10.0).abs() < 1e-9
        ),
        "model must carry obs_scale = 10"
    );
    let population = read_nonmem_csv(Path::new("tests/nonmem/warfarin_scaled.csv"), None, None)
        .expect("scaled warfarin data must load");

    // Gradient-based analytic path (built-in L-BFGS outer + analytic inner). The
    // inner solver stays at the default Auto/BFGS — the choice doesn't change the
    // EBE/gradient, and pinning it mutates a process-global that races sibling
    // tests under parallel execution.
    let mut opt_lbfgs = FitOptions::default();
    opt_lbfgs.optimizer = Optimizer::Lbfgs;
    opt_lbfgs.inner_tol = 1e-8;
    opt_lbfgs.outer_maxiter = 300;
    opt_lbfgs.run_covariance_step = false;
    opt_lbfgs.verbose = false;
    let lbfgs = fit(&model, &population, &model.default_params, &opt_lbfgs)
        .expect("analytic L-BFGS scaled fit must succeed");
    // The `converged` flag is the outer gradient-norm criterion; additive error
    // leaves it just above tolerance on warfarin's flat KA ridge. Substantive
    // convergence is asserted below via the OFV/estimate match (anchored to ferx
    // and, independently, to the default gradient-free optimizer).
    assert!(
        lbfgs.ofv.is_finite(),
        "OFV must be finite, got {}",
        lbfgs.ofv
    );

    // ferx self-consistency anchor (analytic L-BFGS path).
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    assert!(rel(lbfgs.ofv, -740.838).abs() < 0.01, "OFV {}", lbfgs.ofv);
    assert!(
        rel(lbfgs.theta[0], 0.132956) < 0.02,
        "TVCL {}",
        lbfgs.theta[0]
    );
    assert!(
        rel(lbfgs.theta[1], 7.72748) < 0.02,
        "TVV {}",
        lbfgs.theta[1]
    );
    assert!(
        rel(lbfgs.theta[2], 0.810476) < 0.05,
        "TVKA {}",
        lbfgs.theta[2]
    );
    assert!(
        rel(lbfgs.sigma[0], 0.008687) < 0.05,
        "ADD {}",
        lbfgs.sigma[0]
    );

    // Default gradient-free path sees the same scaled objective: OFV agrees and
    // the well-determined θ (CL, V) match; TVKA sits on a flat ridge (≈58% CV),
    // so it gets a wider band.
    let mut opt_def = FitOptions::default();
    opt_def.outer_maxiter = 300;
    opt_def.run_covariance_step = false;
    opt_def.verbose = false;
    let def = fit(&model, &population, &model.default_params, &opt_def)
        .expect("default scaled fit must succeed");
    assert!(
        (lbfgs.ofv - def.ofv).abs() < 0.5,
        "OFV L-BFGS {} vs default {}",
        lbfgs.ofv,
        def.ofv
    );
    assert!(
        rel(lbfgs.theta[0], def.theta[0]) < 0.02,
        "TVCL L-BFGS {} vs default {}",
        lbfgs.theta[0],
        def.theta[0]
    );
    assert!(
        rel(lbfgs.theta[1], def.theta[1]) < 0.02,
        "TVV L-BFGS {} vs default {}",
        lbfgs.theta[1],
        def.theta[1]
    );

    // ── NONMEM 7.5.1 FOCEI cross-check (tests/nonmem/warfarin_scaled.{ctl,lst},
    //    S2 = V*10, MINIMIZATION SUCCESSFUL) — the analytic obs_scale path matches
    //    NONMEM's MLE to ~5 significant figures.
    const NM_OFV: f64 = -740.83761;
    const NM_TVCL: f64 = 0.132943;
    const NM_TVV: f64 = 7.72787;
    const NM_TVKA: f64 = 0.810919;
    const NM_ADD_SD: f64 = 0.0086872; // sqrt(SIGMA(1,1) = 7.54671e-5)
    const NM_OM_CL: f64 = 0.0287566;
    const NM_OM_V: f64 = 0.00947183;
    const NM_OM_KA: f64 = 0.335508;

    assert!(
        (lbfgs.ofv - NM_OFV).abs() < 0.1,
        "OFV {:.4} vs NONMEM {NM_OFV:.4}",
        lbfgs.ofv
    );
    assert!(
        rel(lbfgs.theta[0], NM_TVCL) < 0.01,
        "TVCL {} vs NM {NM_TVCL}",
        lbfgs.theta[0]
    );
    assert!(
        rel(lbfgs.theta[1], NM_TVV) < 0.01,
        "TVV {} vs NM {NM_TVV}",
        lbfgs.theta[1]
    );
    assert!(
        rel(lbfgs.theta[2], NM_TVKA) < 0.02,
        "TVKA {} vs NM {NM_TVKA}",
        lbfgs.theta[2]
    );
    assert!(
        rel(lbfgs.sigma[0], NM_ADD_SD) < 0.02,
        "ADD {} vs NM {NM_ADD_SD}",
        lbfgs.sigma[0]
    );
    let om: Vec<f64> = (0..3).map(|i| lbfgs.omega[(i, i)]).collect();
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
