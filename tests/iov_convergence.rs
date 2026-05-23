//! Slow convergence tests for the IOV fitting path.
//!
//! These run SLSQP and BFGS to convergence on a real IOV model and verify
//! both optimizers find the same minimum — confirming that the IOV objective
//! (kappa EBEs + IOV prior) is computed consistently.  Gate them so they
//! are skipped in the default PR job (only run nightly / on demand):
//!
//!   cargo test --features slow-tests --test iov_convergence
//!
//! The trust-region outer optimizer does not support IOV models (n_kappa > 0).
//! Use slsqp, bfgs, bobyqa, lbfgs, or mma for IOV fits.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

/// SLSQP FOCEI on warfarin_iov converges to a finite OFV.
///
/// The OCC column must be passed as iov_column so that subject.occasions is
/// populated; without it every subject falls through to the non-IOV EBE path
/// and panics when KAPPA_CL (eta index 3) is evaluated against a 3-element
/// BSV eta slice.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_slsqp_converges() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 500;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("SLSQP IOV fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "SLSQP IOV OFV must be finite, got {}",
        result.ofv
    );
}

/// BFGS FOCEI on warfarin_iov must reach the same OFV as SLSQP.
///
/// Both optimizers drive the same FOCE objective with the same IOV kappa
/// EBEs — they should find the same local minimum within 1.0 OFV unit.
/// BFGS is the natural cross-check for SLSQP on IOV models (trust-region
/// does not support IOV).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_bfgs_matches_slsqp() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

    let mut opts_ref = FitOptions::default();
    opts_ref.method = EstimationMethod::FoceI;
    opts_ref.optimizer = Optimizer::Slsqp;
    opts_ref.outer_maxiter = 500;
    opts_ref.run_covariance_step = false;
    opts_ref.verbose = false;
    let ref_result = fit(&model, &population, &model.default_params, &opts_ref)
        .expect("SLSQP IOV reference fit must succeed");
    assert!(
        ref_result.ofv.is_finite(),
        "SLSQP IOV OFV must be finite, got {}",
        ref_result.ofv
    );

    let mut opts_bfgs = FitOptions::default();
    opts_bfgs.method = EstimationMethod::FoceI;
    opts_bfgs.optimizer = Optimizer::Bfgs;
    opts_bfgs.outer_maxiter = 500;
    opts_bfgs.run_covariance_step = false;
    opts_bfgs.verbose = false;
    let bfgs_result = fit(&model, &population, &model.default_params, &opts_bfgs)
        .expect("BFGS IOV fit must succeed");

    assert!(
        bfgs_result.ofv.is_finite(),
        "BFGS IOV OFV must be finite, got {}",
        bfgs_result.ofv
    );
    // One-sided: BFGS may find a better minimum than SLSQP, but must not
    // regress more than 1.0 unit above the SLSQP baseline.
    assert!(
        bfgs_result.ofv <= ref_result.ofv + 1.0,
        "BFGS IOV OFV {:.4} is more than 1.0 above SLSQP OFV {:.4}",
        bfgs_result.ofv,
        ref_result.ofv,
    );
}

/// SAEM + IOV (Step 11): smoke test — fit() must return Ok with a finite OFV
/// after a handful of SAEM iterations.  Does NOT assert convergence; just
/// confirms the SAEM IOV path (kappa MH + omega_iov analytic update) compiles
/// and runs without panicking.
///
/// This is a Tier-2 test: it calls the public API but exits after 5 iterations.
#[test]
fn iov_saem_smoke_returns_finite_ofv() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

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

/// SAEM + IOV convergence: run to convergence and compare OFV to the SLSQP
/// FOCEI baseline.  SAEM and FOCE are different approximations so OFVs can
/// differ; we allow up to 2.0 OFV units of slack.
///
/// This is a Tier-3 slow test — gated on the `slow-tests` feature.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_saem_converges_within_tolerance_of_focei() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");

    // FOCEI reference (SLSQP)
    let mut opts_ref = FitOptions::default();
    opts_ref.method = EstimationMethod::FoceI;
    opts_ref.optimizer = Optimizer::Slsqp;
    opts_ref.outer_maxiter = 500;
    opts_ref.run_covariance_step = false;
    opts_ref.verbose = false;
    let focei_result = fit(&model, &population, &model.default_params, &opts_ref)
        .expect("FOCEI IOV reference fit must succeed");
    assert!(
        focei_result.ofv.is_finite(),
        "FOCEI IOV OFV must be finite, got {}",
        focei_result.ofv
    );

    // SAEM
    let mut opts_saem = FitOptions::default();
    opts_saem.method = EstimationMethod::Saem;
    opts_saem.outer_maxiter = 800; // saem_n_exploration + saem_n_convergence default
    opts_saem.run_covariance_step = false;
    opts_saem.verbose = false;
    let saem_result = fit(&model, &population, &model.default_params, &opts_saem)
        .expect("SAEM IOV fit must succeed");

    assert!(
        saem_result.ofv.is_finite(),
        "SAEM IOV OFV must be finite, got {}",
        saem_result.ofv
    );
    assert!(
        saem_result.omega_iov.is_some(),
        "omega_iov must be present in SAEM IOV result"
    );
    // SAEM and FOCEI are different approximations; allow 2.0 OFV units of slack.
    // TODO: tighten after nightly baseline is established.
    assert!(
        (saem_result.ofv - focei_result.ofv).abs() < 2.0,
        "SAEM IOV OFV {:.4} differs from FOCEI OFV {:.4} by more than 2.0 units",
        saem_result.ofv,
        focei_result.ofv
    );
}
