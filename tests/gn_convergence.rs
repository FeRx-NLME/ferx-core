//! Slow convergence tests for the Gauss-Newton trust-region optimizer.
//!
//! These run to convergence and verify that the TR-based GN optimizer finds
//! the same minimum as the reference SLSQP/FOCEI run. Gate them so they are
//! skipped in the default PR job (only run nightly / on demand):
//!
//!   cargo test --features slow-tests --test gn_convergence

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

fn warfarin_data_and_model() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

/// GN trust-region on warfarin must converge to an OFV within 1.0 of SLSQP.
///
/// Both optimizers start from the same initial parameters and are solving the
/// same FOCE objective — they should find the same local minimum. A 1.0 OFV
/// slack absorbs minor differences in convergence point due to the different
/// path each optimizer takes.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_tr_warfarin_ofv_matches_slsqp_baseline() {
    let (model, population) = warfarin_data_and_model();

    // Reference: SLSQP FOCE
    let mut opts_ref = FitOptions::default();
    opts_ref.method = EstimationMethod::Foce;
    opts_ref.optimizer = Optimizer::Slsqp;
    opts_ref.outer_maxiter = 500;
    opts_ref.run_covariance_step = false;
    opts_ref.verbose = false;
    let ref_result = fit(&model, &population, &model.default_params, &opts_ref)
        .expect("SLSQP reference fit must succeed");
    assert!(ref_result.ofv.is_finite(), "SLSQP OFV must be finite");

    // GN trust-region
    let mut opts_gn = FitOptions::default();
    opts_gn.method = EstimationMethod::FoceGn;
    opts_gn.outer_maxiter = 200;
    opts_gn.run_covariance_step = false;
    opts_gn.verbose = false;
    let gn_result = fit(&model, &population, &model.default_params, &opts_gn)
        .expect("GN trust-region fit must succeed");

    assert!(
        gn_result.ofv.is_finite(),
        "GN-TR OFV must be finite, got {}",
        gn_result.ofv
    );
    assert!(
        (gn_result.ofv - ref_result.ofv).abs() < 1.0,
        "GN-TR OFV {:.4} deviates from SLSQP OFV {:.4} by more than 1.0 unit",
        gn_result.ofv,
        ref_result.ofv,
    );
}

/// GN-hybrid trust-region on warfarin: the GN phase followed by FOCEI polish
/// must converge to the same OFV as pure SLSQP within 0.1 units.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_hybrid_tr_warfarin_ofv_matches_slsqp() {
    let (model, population) = warfarin_data_and_model();

    let mut opts_ref = FitOptions::default();
    opts_ref.method = EstimationMethod::Foce;
    opts_ref.optimizer = Optimizer::Slsqp;
    opts_ref.outer_maxiter = 500;
    opts_ref.run_covariance_step = false;
    opts_ref.verbose = false;
    let ref_result = fit(&model, &population, &model.default_params, &opts_ref)
        .expect("SLSQP reference fit must succeed");

    let mut opts_hybrid = FitOptions::default();
    opts_hybrid.method = EstimationMethod::FoceGnHybrid;
    opts_hybrid.outer_maxiter = 200;
    opts_hybrid.run_covariance_step = false;
    opts_hybrid.verbose = false;
    let hybrid_result = fit(&model, &population, &model.default_params, &opts_hybrid)
        .expect("GN-hybrid fit must succeed");

    assert!(
        hybrid_result.ofv.is_finite(),
        "GN-hybrid OFV must be finite, got {}",
        hybrid_result.ofv
    );
    assert!(
        (hybrid_result.ofv - ref_result.ofv).abs() < 0.1,
        "GN-hybrid OFV {:.4} deviates from SLSQP OFV {:.4} by more than 0.1",
        hybrid_result.ofv,
        ref_result.ofv,
    );
}
