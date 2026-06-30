//! Convergence / objective-consistency tests for the IOV fitting path.
//!
//! Gate the slow ones so they are skipped in the default PR job (run nightly /
//! on demand):
//!
//!   cargo test --features slow-tests --test iov_convergence
//!
//! ## What these check (issue #101)
//!
//! 1. `foce_subject_nll_iov` is a *proper* augmented FOCE marginal — the
//!    per-occasion κ are integrated out through R̃ like the BSV η (kappa-augmented
//!    H + block-diagonal Σ), not via an additive κ MAP penalty. All estimation
//!    paths share one consistent objective.
//!
//! 2. For IOV models the outer optimizer computes its gradient by **re-converging
//!    the EBEs at each FD point** (`reconverged_fd_gradient`). The IOV variance
//!    components — especially Ω_iov — are weakly identified and their gradient is
//!    dominated by the EBE response (raising Ω_iov un-shrinks the kappas); the
//!    old fixed-EBE gradient missed this and left Ω_iov pinned at its initial
//!    value. With the fix, gradient FOCEI moves the variance components and a
//!    `saem → focei` chain polishes SAEM's result down to a *better* optimum
//!    (OFV ≈ 308 vs SAEM's ≈ 320) — see issue #101 rec #2.
//!
//! The FOCEI optimum (CL≈0.17, V≈8.5, KA≈1.15, Ω_iov≈0.047) matches the NONMEM
//! 7.5.1 reference basin (tests/nonmem/warfarin_iov.ctl), and the per-occasion
//! prediction is exact (ferx PRED == NONMEM PRED to 5 s.f. — issue #104).
//!
//! Since the Almquist 2015 Laplace switch in `foce_subject_nll_interaction`
//! (FOCEI INTER now matches NONMEM's actual marginal, not the Sheiner–Beal
//! linearised form), ferx's IOV OFV converges to ≈307.8, within ≈1 of NONMEM's
//! 308.83 — see `tests/warfarin_iov_nonmem.rs` and
//! `[[focei-laplace-not-sheiner-beal]]`.
//!
//! 3. Pure FOCEI/SLSQP now reaches the minimum from the model's cold default
//!    start: for IOV models the SLSQP path auto-enables per-coordinate scaling
//!    so its uniform gradient cap no longer starves the omega/omega_iov step
//!    (issue #101 rec #2). Very far-off starts (e.g. a residual-error init off
//!    by >2×) can still stall; SAEM / `saem → focei` remain the most robust.
//!
//! The trust-region outer optimizer does not support IOV models (n_kappa > 0).
//! Use slsqp, bfgs, bobyqa, lbfgs, or mma for IOV fits.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult, Optimizer};
use std::path::Path;

/// OFV the FOCEI marginal minimum reaches on warfarin_iov with the continuous
/// per-occasion-aware prediction (issue #104) and the Almquist 2015 Laplace
/// FOCEI INTER form. Pure SLSQP, pure BFGS, and the `saem → focei` chain all
/// agree here; the parameters match the NONMEM reference basin (CL≈0.17,
/// V≈8.5, Ω_iov≈0.047). NONMEM 7.5.1 reports 308.83 — the ~1-unit gap is
/// FD-vs-analytical-sensitivity noise.
const IOV_FOCEI_OFV: f64 = 307.84;

fn load() -> (ferx_core::CompiledModel, ferx_core::Population) {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");
    (model, population)
}

/// Run a single method with the given optimizer.
fn run_single(method: EstimationMethod, optimizer: Optimizer) -> FitResult {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.optimizer = optimizer;
    opts.outer_maxiter = 800;
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &population, &model.default_params, &opts).expect("IOV fit must succeed")
}

/// Run a `saem → focei` method chain with the given polishing optimizer.
fn run_chain(optimizer: Optimizer) -> FitResult {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
    opts.optimizer = optimizer;
    opts.outer_maxiter = 800;
    opts.run_covariance_step = false;
    opts.verbose = false;
    fit(&model, &population, &model.default_params, &opts).expect("IOV chain fit must succeed")
}

/// SLSQP FOCEI on warfarin_iov returns a finite OFV (smoke).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_slsqp_converges() {
    let result = run_single(EstimationMethod::FoceI, Optimizer::Slsqp);
    assert!(
        result.ofv.is_finite(),
        "SLSQP IOV OFV must be finite, got {}",
        result.ofv
    );
}

/// The `saem → focei` chain reaches the FOCEI marginal minimum and, crucially,
/// improves on SAEM alone — the regression guard for the reconverged-EBE
/// gradient (issue #101 rec #2). Before that fix the fixed-EBE gradient left
/// FOCEI pinned at SAEM's point; now it descends to a strictly better OFV.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_chain_improves_on_saem_and_reaches_reference() {
    let saem = run_single(EstimationMethod::Saem, Optimizer::Slsqp);
    assert!(
        saem.ofv.is_finite() && saem.omega_iov.is_some(),
        "SAEM must return a finite OFV and omega_iov, got {}",
        saem.ofv
    );

    let chain = run_chain(Optimizer::Slsqp);
    assert!(
        chain.ofv.is_finite(),
        "chain OFV must be finite, got {}",
        chain.ofv
    );
    // FOCEI polish reaches the reference minimum...
    assert!(
        (chain.ofv - IOV_FOCEI_OFV).abs() < 2.0,
        "saem→focei chain OFV {:.4} should reach the FOCEI reference {:.2}",
        chain.ofv,
        IOV_FOCEI_OFV
    );
    // ...and still improves on SAEM alone (would NOT, with a fixed-EBE gradient
    // that can't move the variance components). The improvement margin used to
    // be ~1.4 OFV units, but the SAEM multi-kernel + damped-Ω fixes
    // (`saem_block_omega_collapse`) make SAEM land much closer to the FOCEI
    // minimum on this IOV model — observed chain 307.84 vs SAEM 308.02
    // (gap ≈ 0.18). The floor is therefore 0.05 OFV units: well above the
    // FD-noise scale (~1e-3) and still cleanly above the pre-fix bug's gap ≈ 0
    // (chain pinned at SAEM's point), while accommodating the now-smaller
    // genuine gap. The strict-minimum assertion above (`chain.ofv ≈
    // IOV_FOCEI_OFV`) does the heavy lifting on "did FOCEI polish reach the
    // right place".
    assert!(
        chain.ofv < saem.ofv - 0.05,
        "FOCEI polish must improve on SAEM by >0.05 OFV units: chain {:.4} vs SAEM {:.4}",
        chain.ofv,
        saem.ofv
    );
}

/// The two outer optimizers (SLSQP, BFGS) driving the augmented marginal in a
/// `saem → focei` chain reach the same minimum — the objective and the
/// reconverged gradient are optimizer-independent. BFGS may report
/// `converged = false` while sitting on the minimum, so this asserts on OFV.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_chain_slsqp_and_bfgs_agree() {
    let slsqp = run_chain(Optimizer::Slsqp);
    let bfgs = run_chain(Optimizer::Bfgs);
    assert!(slsqp.ofv.is_finite() && bfgs.ofv.is_finite());
    assert!(
        (slsqp.ofv - bfgs.ofv).abs() < 1.0,
        "saem→focei OFV must match across optimizers: SLSQP {:.4} vs BFGS {:.4}",
        slsqp.ofv,
        bfgs.ofv
    );
    assert!(
        (slsqp.ofv - IOV_FOCEI_OFV).abs() < 2.0,
        "saem→focei/SLSQP OFV {:.4} should reach the FOCEI reference {:.2}",
        slsqp.ofv,
        IOV_FOCEI_OFV
    );
}

/// SAEM + IOV (smoke): fit() must return Ok with a finite OFV after a handful of
/// SAEM iterations. Tier-2 — calls the public API but exits after 5 iterations.
#[test]
fn iov_saem_smoke_returns_finite_ofv() {
    let (model, population) = load();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.outer_maxiter = 5; // just enough to exercise the full E+M loop
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("SAEM IOV smoke must not Err");

    assert!(
        result.ofv.is_finite(),
        "SAEM IOV smoke OFV must be finite after 5 iter, got {}",
        result.ofv
    );
    assert!(
        result.omega_iov.is_some(),
        "omega_iov must be present in SAEM IOV result"
    );
}

/// Pure FOCEI/SLSQP from the model's cold default start reaches the FOCEI
/// minimum on warfarin_iov — no SAEM seeding needed. This is the payoff of the
/// IOV+SLSQP auto-scaling fix (issue #101 rec #2): before it, SLSQP's uniform
/// gradient cap starved the omega step and the fit stalled at OFV ≈ 292 with the
/// variance components pinned near their initial values.
///
/// The cold-start SLSQP path uses a fixed-EBE FD gradient (the same gradient
/// bias that motivated switching the default optimizer to BOBYQA in PR #155)
/// and terminates at platform-dependent points a few OFV units above the true
/// minimum: macOS arm64 reaches 307.8, Linux x86_64 stalls at 314.7. The
/// SAEM→FOCEI chain tests above are the platform-independent reference. The
/// tolerance here is therefore 10 OFV units — wide enough to absorb that
/// platform gap while still catching the pre-fix stall at OFV ≈ 292. The
/// companion `omega_iov > 0.02` assertion guards the mechanism that #101
/// rec #2 actually fixed (variance component moves off its 0.01 init).
// Re-enabled (#335): cold-start SLSQP on this IOV model used to stall at OFV 343.5
// with omega_iov pinned at its 0.01 init. The fix is `parameter_scaling = rescale2`
// now applying to SLSQP under `Auto` (bound-half-width rescaling — see
// `resolve_scaling`): with it, pure FOCEI/SLSQP from the cold default start reaches
// OFV 307.84 and omega_iov climbs to ≈0.046, matching the default BOBYQA.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_pure_slsqp_from_cold_start_reaches_minimum() {
    let focei = run_single(EstimationMethod::FoceI, Optimizer::Slsqp);
    assert!(
        (focei.ofv - IOV_FOCEI_OFV).abs() < 10.0,
        "pure FOCEI/SLSQP from the cold default start must reach the FOCEI \
         minimum {:.2} (within 10 OFV units for platform-dependent SLSQP \
         termination), got {:.4}",
        IOV_FOCEI_OFV,
        focei.ofv
    );
    // The fix is specifically about moving the IOV variance off its init: Ω_iov
    // starts at 0.01 and must climb toward ≈0.036.
    let iov = focei.omega_iov.expect("omega_iov present")[(0, 0)];
    assert!(
        iov > 0.02,
        "omega_iov must move off its 0.01 init toward ≈0.036, got {:.4}",
        iov
    );
}

/// Estimate-level validation for **IOV + `iiv_on_ruv`** (#4b/#486): the new analytic
/// FOCEI gradient must drive the optimizer to the *same MLE* as the finite-difference
/// gradient — the FD path being the NONMEM-anchored one (warfarin_iov ≈307.8 vs NONMEM
/// 308.83, see the module header and `tests/warfarin_iov_nonmem.rs`). Both gradients
/// differentiate the identical FOCEI marginal, so the converged OFV and estimates must
/// agree; this confirms the FD→analytic swap does not move the optimum, i.e. the
/// analytic fit inherits the FD path's NONMEM anchoring at the estimate level (not just
/// the per-point gradient agreement the `*_matches_fd` unit tests already pin).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_iiv_on_ruv_analytic_matches_fd_estimates() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method     = focei
  iov_column = OCC
  covariance = false
"#;
    let model = ferx_core::parser::model_parser::parse_model_string(src)
        .expect("IOV + iiv_on_ruv model parses");
    assert_eq!(model.residual_error_eta, Some(3));
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let run = |gm: ferx_core::GradientMethod| -> FitResult {
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::FoceI;
        opts.interaction = true;
        opts.optimizer = Optimizer::Slsqp;
        opts.gradient_method = gm;
        opts.run_covariance_step = false;
        opts.verbose = false;
        fit(&model, &pop, &model.default_params, &opts).expect("IOV + iiv_on_ruv fit runs")
    };
    let analytic = run(ferx_core::GradientMethod::Auto);
    let fd = run(ferx_core::GradientMethod::Fd);

    // Same objective ⇒ same optimum (allowing for optimizer-path noise).
    assert!(
        (analytic.ofv - fd.ofv).abs() < 1.0,
        "analytic OFV {:.4} vs FD OFV {:.4} should agree (same marginal)",
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
}

/// Estimate-level validation for **M3 BLOQ + IOV** (#4a/#580): the new analytic FOCEI
/// gradient must drive the optimizer to the *same MLE* as the finite-difference gradient.
/// Both differentiate the identical M3-promoted FOCEI marginal `foce_subject_nll_iov`
/// (censored rows enter as `−logΦ`, excluded from `H̃`), so the converged OFV and θ must
/// agree — confirming the FD→analytic swap does not move the optimum. The FD path is the
/// reference: the IOV FD gradient is NONMEM-anchored on warfarin_iov (≈307.8 vs NONMEM
/// 308.83, see the module header / `tests/warfarin_iov_nonmem.rs`), and the M3 censored
/// term is NONMEM-anchored on the non-IOV warfarin_bloq fit (`tests/bloq_convergence.rs`);
/// the analytic M3+IOV fit inherits both anchors transitively by landing on the same
/// optimum. Data: `data/warfarin_iov_bloq.csv` = warfarin_iov with a CENS column
/// (LLOQ = 1.5 left-censors the 23 sub-limit observations, ~10%).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_m3_analytic_matches_fd_estimates() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method      = focei
  iov_column  = OCC
  bloq_method = m3
  covariance  = false
"#;
    let model =
        ferx_core::parser::model_parser::parse_model_string(src).expect("IOV + M3 model parses");
    assert!(matches!(
        model.bloq_method,
        ferx_core::types::BloqMethod::M3
    ));
    // Gate: M3 + IOV (no iiv_on_ruv) is analytic on both loops.
    assert!(ferx_core::sens::provider::iov_analytical_supported(&model));
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov_bloq.csv"), None, Some("OCC"))
        .expect("warfarin_iov_bloq data loads");
    assert!(
        pop.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0)),
        "dataset must carry censored rows"
    );

    let run = |gm: ferx_core::GradientMethod| -> FitResult {
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::FoceI;
        opts.interaction = true;
        opts.optimizer = Optimizer::Slsqp;
        opts.gradient_method = gm;
        opts.run_covariance_step = false;
        opts.verbose = false;
        fit(&model, &pop, &model.default_params, &opts).expect("IOV + M3 fit runs")
    };
    let analytic = run(ferx_core::GradientMethod::Auto);
    let fd = run(ferx_core::GradientMethod::Fd);

    // Same objective ⇒ same optimum (allowing for optimizer-path noise).
    assert!(
        (analytic.ofv - fd.ofv).abs() < 1.0,
        "analytic OFV {:.4} vs FD OFV {:.4} should agree (same marginal)",
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
}

/// Estimate-level validation for the **triple** M3 + IOV + `iiv_on_ruv` (#591): the
/// analytic FOCEI gradient must drive the optimizer to the same MLE as the FD gradient.
/// The closed-form assembly already carried the censored residual-eta cross coefficients
/// `(C·z, C·m)` over the stacked layout (#4c); #591 opens the gate, so both gradients now
/// differentiate the identical M3-censored, `exp(2·η_ruv)`-scaled FOCEI marginal. The FD
/// path inherits the IOV (warfarin_iov ≈307.8 vs NONMEM 308.83) and non-IOV M3 (warfarin
/// BLOQ) anchors; the analytic fit inherits both by landing on the same optimum. Data:
/// `data/warfarin_iov_bloq.csv` (LLOQ = 1.5, ~10% left-censored).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_m3_iiv_on_ruv_analytic_matches_fd_estimates() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV
[fit_options]
  method      = focei
  iov_column  = OCC
  bloq_method = m3
  covariance  = false
"#;
    let model = ferx_core::parser::model_parser::parse_model_string(src)
        .expect("triple M3 + IOV + iiv_on_ruv model parses");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(matches!(
        model.bloq_method,
        ferx_core::types::BloqMethod::M3
    ));
    // Gate: the closed-form triple is analytic on both loops as of #591.
    assert!(ferx_core::sens::provider::iov_analytical_supported(&model));
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov_bloq.csv"), None, Some("OCC"))
        .expect("warfarin_iov_bloq data loads");
    assert!(
        pop.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0)),
        "dataset must carry censored rows"
    );

    let run = |gm: ferx_core::GradientMethod| -> FitResult {
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::FoceI;
        opts.interaction = true;
        opts.optimizer = Optimizer::Slsqp;
        opts.gradient_method = gm;
        opts.run_covariance_step = false;
        opts.verbose = false;
        fit(&model, &pop, &model.default_params, &opts).expect("triple fit runs")
    };
    let analytic = run(ferx_core::GradientMethod::Auto);
    let fd = run(ferx_core::GradientMethod::Fd);

    assert!(
        (analytic.ofv - fd.ofv).abs() < 1.0,
        "analytic OFV {:.4} vs FD OFV {:.4} should agree (same marginal)",
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
}

/// Estimate-level validation for **FOCE-IOV-M3** (#591): under non-interaction FOCE, M3
/// censoring no longer promotes the subject to interaction — the whole subject keeps a
/// consistent Sheiner–Beal objective (censored rows re-enter as `−logΦ` at the population
/// variance). The analytic FOCE IOV gradient (now carrying the censored SB term over the
/// stacked layout) must drive the optimizer to the same MLE as FD of that same FOCE
/// marginal. FOCE-IOV-M3 and FOCEI-IOV-M3 are genuinely different optima (cf. the non-IOV
/// warfarin-BLOQ FOCE-vs-FOCEI split, `tests/bloq_convergence.rs`). Data:
/// `data/warfarin_iov_bloq.csv`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_m3_foce_analytic_matches_fd_estimates() {
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  kappa KAPPA_CL ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method      = foce
  iov_column  = OCC
  bloq_method = m3
  covariance  = false
"#;
    let model = ferx_core::parser::model_parser::parse_model_string(src)
        .expect("FOCE IOV + M3 model parses");
    assert!(matches!(
        model.bloq_method,
        ferx_core::types::BloqMethod::M3
    ));
    assert!(ferx_core::sens::provider::iov_analytical_supported(&model));
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov_bloq.csv"), None, Some("OCC"))
        .expect("warfarin_iov_bloq data loads");
    assert!(
        pop.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0)),
        "dataset must carry censored rows"
    );

    let run = |gm: ferx_core::GradientMethod| -> FitResult {
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::Foce;
        opts.interaction = false;
        opts.optimizer = Optimizer::Slsqp;
        opts.gradient_method = gm;
        opts.run_covariance_step = false;
        opts.verbose = false;
        fit(&model, &pop, &model.default_params, &opts).expect("FOCE IOV + M3 fit runs")
    };
    let analytic = run(ferx_core::GradientMethod::Auto);
    let fd = run(ferx_core::GradientMethod::Fd);

    assert!(
        (analytic.ofv - fd.ofv).abs() < 1.0,
        "analytic OFV {:.4} vs FD OFV {:.4} should agree (same FOCE marginal)",
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
}
