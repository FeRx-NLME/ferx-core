//! Slow convergence tests for the Gauss-Newton trust-region optimizer.
//!
//! These run to convergence and verify that the TR-based GN optimizer finds
//! a good minimum. Gate them so they are skipped in the default PR job
//! (only run nightly / on demand):
//!
//!   cargo test --features slow-tests --test gn_convergence
//!
//! Baselines are hard-coded to the best observed OFV for each dataset.
//! The assertion is one-sided: finding a *better* (lower) OFV is never a
//! failure; regressing above `baseline + tolerance` is.  Update the constant
//! if a deliberate algorithmic improvement raises the bar.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
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

/// GN trust-region on warfarin must reach the known-good minimum.
///
/// Baseline: best observed OFV from GN-TR (-285.7163).  SLSQP reaches only
/// -280.1837 on this dataset — GN-TR consistently finds a better minimum.
/// Tolerance of 0.5 absorbs minor run-to-run numerical variation.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_tr_warfarin_ofv_matches_slsqp_baseline() {
    const KNOWN_GOOD_OFV: f64 = -285.7163;
    const TOLERANCE: f64 = 0.5;

    let (model, population) = warfarin_data_and_model();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 200;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("GN trust-region fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "GN-TR OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        result.ofv <= KNOWN_GOOD_OFV + TOLERANCE,
        "GN-TR OFV {:.4} regressed above known-good {:.4} (tolerance {:.1})",
        result.ofv,
        KNOWN_GOOD_OFV,
        TOLERANCE,
    );
}

/// GN-hybrid trust-region on warfarin: the GN phase followed by FOCEI polish
/// must reach the known-good minimum.
///
/// Same baseline as pure GN-TR (-285.7163); the hybrid should be at least as
/// good since it polishes the GN solution with FOCEI.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_hybrid_tr_warfarin_ofv_matches_slsqp() {
    const KNOWN_GOOD_OFV: f64 = -285.7163;
    const TOLERANCE: f64 = 0.5;

    let (model, population) = warfarin_data_and_model();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceGnHybrid;
    opts.outer_maxiter = 200;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("GN-hybrid fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "GN-hybrid OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        result.ofv <= KNOWN_GOOD_OFV + TOLERANCE,
        "GN-hybrid OFV {:.4} regressed above known-good {:.4} (tolerance {:.1})",
        result.ofv,
        KNOWN_GOOD_OFV,
        TOLERANCE,
    );
}

/// GN trust-region on the covariate model must reach the known-good minimum.
///
/// This model has weakly-identified covariate exponents alongside well-identified
/// PK parameters — the TR per-direction step scaling handles the mixed-curvature
/// case.  Baseline: best observed OFV from GN-TR (-1198.3229); SLSQP reaches
/// only -1116.0357, suggesting it gets stuck in a shallow region of the
/// likelihood surface driven by the flat covariate directions.
/// Tolerance of 1.0 absorbs minor numerical variation; a jump of more than
/// ~80 units above the baseline would indicate a real regression.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_tr_two_cpt_oral_cov_ofv_matches_slsqp_baseline() {
    const KNOWN_GOOD_OFV: f64 = -1198.3229;
    const TOLERANCE: f64 = 1.0;

    let model = parse_model_file(Path::new("examples/two_cpt_oral_cov.ferx"))
        .expect("two_cpt_oral_cov model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
        .expect("two_cpt_oral_cov data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("GN trust-region fit must succeed");

    assert!(
        result.ofv.is_finite(),
        "GN-TR OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        result.ofv <= KNOWN_GOOD_OFV + TOLERANCE,
        "GN-TR OFV {:.4} regressed above known-good {:.4} (tolerance {:.1})",
        result.ofv,
        KNOWN_GOOD_OFV,
        TOLERANCE,
    );
}
