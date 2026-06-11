//! Slow convergence test for the log-transform-both-sides (LTBS) error model
//! (issue #120).
//!
//! Fits the log-scale warfarin model (`examples/warfarin_ltbs.ferx`,
//! `log(DV) ~ additive(ADD_LOG)`) on the warfarin dataset and checks that the
//! full FOCEI LTBS pipeline — DV log-transform at load, log-wrapped predictions,
//! additive-on-log likelihood — converges to a sensible minimum and recovers
//! plausible PK parameters.
//!
//! Gate: skipped in the default PR job.
//!
//!   cargo test --features slow-tests --test ltbs_convergence
//!
//! ## NONMEM cross-check
//!
//! The log-scale model corresponds to NONMEM's
//!
//! ```text
//! $ERROR
//!   IPRED = LOG(F)
//!   Y     = IPRED + EPS(1)
//! ```
//!
//! with the warfarin DV column log-transformed. A NONMEM 7.5.1 FOCEI reference
//! fit (`tests/nonmem/warfarin_ltbs.{ctl,lst}`, run over `data/warfarin.csv`
//! with DV log-transformed) gives an essentially exact cross-engine match —
//! ferx fitting the natural-scale data with `log(DV) ~ additive` recovers the
//! same MLE as NONMEM fitting log-DV data with `Y = LOG(F) + EPS(1)`:
//!
//! | Parameter   | ferx       | NONMEM 7.5.1 |
//! |-------------|------------|--------------|
//! | OFV         | −675.302   | −675.302     |
//! | TVCL        | 0.132698   | 0.132697     |
//! | TVV         | 7.7381     | 7.7383       |
//! | TVKA        | 0.81085    | 0.81096      |
//! | ADD_LOG (SD)| 0.010564   | 0.010564     |
//! | ω²(CL)      | 0.028588   | 0.028594     |
//! | ω²(V)       | 0.009601   | 0.009604     |
//! | ω²(KA)      | 0.335861   | 0.335945     |
//!
//! The assertions below pin ferx to NONMEM's MLE so a regression in the LTBS
//! wiring (log-wrap, DV transform, or the additive-on-log likelihood) is caught.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ltbs_warfarin_fit_converges_and_recovers_pk() {
    let model = parse_model_file(Path::new("examples/warfarin_ltbs.ferx"))
        .expect("LTBS warfarin model must parse");
    assert!(model.log_transform, "model must be flagged LTBS");
    assert!(
        !model.dv_pre_logged,
        "log(DV) ~ additive logs DV in-engine (case 2)"
    );

    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("LTBS FOCEI fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(result.converged, "LTBS fit should converge");

    // Cross-check against the NONMEM 7.5.1 FOCEI reference (see module docs and
    // tests/nonmem/warfarin_ltbs.lst). Tolerances absorb optimizer/engine noise.
    const NM_OFV: f64 = -675.3016;
    const NM_TVCL: f64 = 0.132697;
    const NM_TVV: f64 = 7.73826;
    const NM_TVKA: f64 = 0.810965;
    const NM_ADD_LOG_SD: f64 = 0.010564; // sqrt(SIGMA(1,1) = 1.11601e-4)
    const NM_OM_CL: f64 = 0.0285942;
    const NM_OM_V: f64 = 0.00960426;
    const NM_OM_KA: f64 = 0.335945;

    assert!(
        (result.ofv - NM_OFV).abs() < 0.5,
        "OFV {:.4} differs from NONMEM {NM_OFV:.4} by > 0.5",
        result.ofv
    );

    let rel = |got: f64, nm: f64| (got - nm).abs() / nm.abs();
    assert!(
        rel(result.theta[0], NM_TVCL) < 0.02,
        "TVCL {} vs NM {NM_TVCL}",
        result.theta[0]
    );
    assert!(
        rel(result.theta[1], NM_TVV) < 0.02,
        "TVV {} vs NM {NM_TVV}",
        result.theta[1]
    );
    assert!(
        rel(result.theta[2], NM_TVKA) < 0.02,
        "TVKA {} vs NM {NM_TVKA}",
        result.theta[2]
    );
    assert!(
        rel(result.sigma[0], NM_ADD_LOG_SD) < 0.05,
        "ADD_LOG SD {} vs NM {NM_ADD_LOG_SD}",
        result.sigma[0]
    );

    // OMEGA diagonal (variances), parallel to eta order CL, V, KA.
    let om: Vec<f64> = (0..3).map(|i| result.omega[(i, i)]).collect();
    assert!(
        rel(om[0], NM_OM_CL) < 0.10,
        "ω²(CL) {} vs NM {NM_OM_CL}",
        om[0]
    );
    assert!(
        rel(om[1], NM_OM_V) < 0.10,
        "ω²(V) {} vs NM {NM_OM_V}",
        om[1]
    );
    assert!(
        rel(om[2], NM_OM_KA) < 0.10,
        "ω²(KA) {} vs NM {NM_OM_KA}",
        om[2]
    );
}

/// NONMEM `$COVARIANCE MATRIX=R` cross-check for the LTBS (additive-on-log)
/// covariance step. The convergence test above pins the MLE; this pins the
/// standard errors, exercising the covariance step on an error model
/// (`log(DV) ~ additive`) not covered by the proportional/combined-error and IOV
/// cross-checks in `warfarin_covariance_nonmem.rs`.
///
/// Reference SEs from `tests/nonmem/warfarin_ltbs.lst`, `STANDARD ERROR OF
/// ESTIMATE` block (`$COVARIANCE UNCONDITIONAL` = `MATRIX=R`). NONMEM reports
/// `SIGMA(1,1)` on the variance scale; ferx's `ADD_LOG` is an SD, so the
/// variance-scale SE `1.69e-5` is converted to the SD scale via the delta method
/// `SE(σ) = SE(σ²) / (2σ)` with `σ = sqrt(1.11601e-4) = 0.010564`.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored LTBS covariance SE cross-check (#120/#223): opt in with --features slow-tests"
)]
fn ltbs_covariance_se_matches_nonmem() {
    let model = parse_model_file(Path::new("examples/warfarin_ltbs.ferx"))
        .expect("LTBS warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("LTBS FOCEI fit must succeed");

    assert!(
        result.covariance_matrix.is_some(),
        "LTBS covariance step must produce a matrix"
    );
    let se_theta = result.se_theta.as_ref().expect("theta SEs present");
    let se_omega = result.se_omega.as_ref().expect("omega SEs present");
    let se_sigma = result.se_sigma.as_ref().expect("sigma SEs present");

    // NONMEM 7.5.1 FOCEI $COVARIANCE MATRIX=R SEs (warfarin_ltbs.lst).
    // (name, ferx SE, NONMEM SE, relative band)
    const NM_SE_SIGMA_SD: f64 = 1.69e-5 / (2.0 * 0.010564); // var-scale → SD-scale
    let checks = [
        ("TVCL", se_theta[0], 7.10e-3, 0.20),
        ("TVV", se_theta[1], 2.40e-1, 0.20),
        ("TVKA", se_theta[2], 1.49e-1, 0.20),
        ("ADD_LOG (SD)", se_sigma[0], NM_SE_SIGMA_SD, 0.25),
        // ω SEs get a 25% band: still catches the factor-of-2 regression (29%)
        // and indefinite-Hessian blow-up, but absorbs the larger FD-vs-autodiff
        // spread on the weakly-determined ω²(KA) (≈42% RSE).
        ("omega_CL", se_omega[0], 1.09e-2, 0.25),
        ("omega_V", se_omega[1], 3.98e-3, 0.25),
        ("omega_KA", se_omega[2], 1.40e-1, 0.25),
    ];
    for (name, ferx_se, nm_se, tol) in checks {
        let rd = (ferx_se - nm_se).abs() / nm_se;
        assert!(
            ferx_se.is_finite() && rd < tol,
            "SE({name}) = {ferx_se:.6} vs NONMEM {nm_se:.6} — relative diff {:.1}% exceeds {:.0}% band",
            rd * 100.0,
            tol * 100.0
        );
    }
}
