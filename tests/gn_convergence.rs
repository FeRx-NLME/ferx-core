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
//!
//! ## Baseline history
//!
//! The original baselines (`-285.7163` warfarin, `-1198.3229` two_cpt_oral_cov)
//! were captured against the Sheiner–Beal linearised marginal NLL.  PR #130
//! ([`fix(focei): switch FOCEI INTER marginal to Almquist 2015 Laplace form`])
//! replaced that NLL with the Almquist Laplace form for `interaction=true`.
//! These tests run with the `FitOptions::default()` (which sets
//! `interaction: true`), so the GN optimiser now targets the Laplace NLL —
//! a different objective whose minimum sits at a different OFV value.  The
//! baselines below reflect the Almquist Laplace minimum and were re-anchored
//! at HEAD on the issue #144 fix.  GN-TR still finds a better minimum than
//! SLSQP on both datasets (the original ordering the tests were written to
//! enforce):
//!
//! | Dataset           | SLSQP    | GN-TR     | GN-hybrid |
//! |-------------------|----------|-----------|-----------|
//! | warfarin          | -278.734 | -279.114  | -279.124  |
//! | two_cpt_oral_cov  |-1144.948 |-1153.071  |   —       |

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
/// Baseline: best observed OFV from GN-TR under the Almquist Laplace NLL
/// (-279.1136).  SLSQP reaches -278.7336 on this dataset — GN-TR
/// consistently finds a marginally better minimum (~0.38 OFV).  Tolerance
/// of 0.25 sits comfortably below that gap, so a regression to SLSQP-level
/// performance still fails the assert (enforcing the "GN-TR > SLSQP"
/// invariant), while leaving ~0.24 OFV of headroom for run-to-run
/// numerical variation (observed variation between GN-TR and GN-hybrid on
/// this dataset is ~0.01 OFV, so 0.25 is generous).  See the
/// module-level "Baseline history" comment for why this differs from the
/// pre-#130 Sheiner-Beal baseline (-285.7163).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_tr_warfarin_ofv_matches_slsqp_baseline() {
    const KNOWN_GOOD_OFV: f64 = -279.1136;
    const TOLERANCE: f64 = 0.25;

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
/// Baseline -279.1243: the FOCEI polish (run with the default `bobyqa`
/// optimizer since #155 — see `FitOptions::default`) improves on pure GN-TR by
/// ~0.01 OFV.  SLSQP-driven polish only reaches -278.7336 (gap ~0.39 OFV),
/// so the pass threshold `KNOWN_GOOD_OFV + TOLERANCE` = -278.8743 is
/// hardcoded below the SLSQP reference: any polish stage that regresses to
/// SLSQP-level performance fails the assert.  Name says "beats_slsqp"
/// because of that threshold placement, not because of any direct
/// `ofv < slsqp_ofv` comparison in the code.  See the module-level
/// "Baseline history" comment for the Almquist Laplace shift from the
/// pre-#130 SB baseline.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_hybrid_tr_warfarin_ofv_beats_slsqp_baseline() {
    const KNOWN_GOOD_OFV: f64 = -279.1243;
    const TOLERANCE: f64 = 0.25;

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
/// case.  Baseline: best observed OFV from GN-TR under the Almquist Laplace
/// NLL (-1153.0708); SLSQP reaches only -1144.9481, so GN-TR still finds a
/// meaningfully better minimum on the flat covariate directions.  Tolerance
/// of 1.0 absorbs minor numerical variation; a jump of more than ~8 OFV
/// units (back toward the SLSQP basin) would indicate a real regression.
/// See the module-level "Baseline history" comment for the shift from the
/// pre-#130 SB baseline (-1198.3229).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_tr_two_cpt_oral_cov_ofv_matches_slsqp_baseline() {
    const KNOWN_GOOD_OFV: f64 = -1153.0708;
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
