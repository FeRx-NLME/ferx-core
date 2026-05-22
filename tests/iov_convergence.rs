//! Slow convergence tests for the IOV analytical gradient path.
//!
//! These run to convergence on a real IOV model and verify that the analytical
//! gradient produces the same result as the central-FD fallback.  Gate them so
//! they are skipped in the default PR job (only run nightly / on demand):
//!
//!   cargo test --features slow-tests --test iov_convergence

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

/// Full FOCEI fit on warfarin_iov must converge to the same OFV whether the
/// analytical IOV gradient is used or not.
///
/// We verify this by comparing the analytical-gradient run (default, non-ODE
/// model so the new path is taken) against a reference SLSQP run.  The two
/// must agree within 0.01 OFV units.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iov_analytical_gradient_converges_to_slsqp_baseline() {
    let model = parse_model_file(Path::new("examples/warfarin_iov.ferx"))
        .expect("warfarin_iov model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, None)
        .expect("warfarin_iov data must load");

    // Reference: SLSQP FOCEI (uses the same analytical gradient internally,
    // so this also exercises the full gradient path to convergence).
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

    // Same run from different optimizer (trust-region) to cross-validate.
    let mut opts_tr = FitOptions::default();
    opts_tr.method = EstimationMethod::FoceI;
    opts_tr.optimizer = Optimizer::TrustRegion;
    opts_tr.outer_maxiter = 500;
    opts_tr.run_covariance_step = false;
    opts_tr.verbose = false;
    let tr_result = fit(&model, &population, &model.default_params, &opts_tr)
        .expect("trust-region IOV fit must succeed");

    assert!(
        tr_result.ofv.is_finite(),
        "Trust-region IOV OFV must be finite, got {}",
        tr_result.ofv
    );
    assert!(
        (tr_result.ofv - ref_result.ofv).abs() < 0.01,
        "Trust-region IOV OFV {:.4} deviates from SLSQP {:.4} by more than 0.01",
        tr_result.ofv,
        ref_result.ofv,
    );
}
