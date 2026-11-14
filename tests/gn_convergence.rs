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
//! | warfarin          | -278.734 | -279.114  | -286.004 (#1540; was -279.124) |
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
/// Baseline -286.0042 (measured on #1540, release, macOS). The FOCEI polish
/// reaches the same minimum under every outer optimizer measured: `auto` (which
/// resolves to `nlopt_lbfgs` on warfarin since #490) -286.0042, `slsqp`
/// -286.0042, `bobyqa` -286.0040. So this test pins the minimum the hybrid
/// reaches, not which optimizer polishes it. The older -279.1243 (BOBYQA polish)
/// and -278.7336 (SLSQP polish) baselines predate later engine changes and no
/// longer describe either optimizer. The previous bound (-278.87) sat 7 OFV above
/// the current minimum, so a regression to the old one would have passed; the
/// 0.25 tolerance is ~1000x the measured spread between optimizers (2.4e-4)
/// and rejects it. It was measured on macOS only; the nightly Linux run is
/// the cross-platform check. Name says "beats_slsqp"
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
    const KNOWN_GOOD_OFV: f64 = -286.0042;
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

/// #2740: `method = gn` on an M3 BLOQ FOCEI model must recover NONMEM's MLE.
///
/// M3 takes the kernel fixed-EBE score, which had no EBE-response correction (`tᵢ = 0`),
/// so GN stepped along a gradient that disagreed with its own objective (θ: 49.86 vs a
/// reconverged central difference of 6.86). The trust region collapsed and GN stopped
/// 10.6 OFV above FOCEI. It now takes each subject's exact total derivative. The
/// reference is the NONMEM 7.5.1 LAPLACE M3 fit `bloq_convergence.rs` anchors L-BFGS
/// to (`tests/nonmem/warfarin_bloq.lst`), at the same tolerances.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_warfarin_bloq_m3_matches_nonmem() {
    let model = parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
        .expect("warfarin BLOQ model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
        .expect("warfarin BLOQ data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 200;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("GN fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    eprintln!(
        "[#2740] GN warfarin_bloq: OFV {:.6}, θ {:?}, σ {:?}, converged {}",
        result.ofv, result.theta, result.sigma, result.converged
    );
    // #2738: before #2740 this fit stopped at −206.196 (+10.6 above FOCEI's −216.8187,
    // max |grad| 34.7) on a micro-step at a collapsed radius and reported converged. A
    // `converged: true` must now sit at the optimum; 0.1 OFV is far below that gap.
    assert!(
        !result.converged || (result.ofv - (-216.8187)).abs() < 0.1,
        "GN reports converged at OFV {} away from FOCEI's -216.8187",
        result.ofv
    );

    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    for (name, got, want, tol) in [
        ("TVCL", result.theta[0], 0.132801, 0.01),
        ("TVV", result.theta[1], 7.73139, 0.01),
        ("TVKA", result.theta[2], 0.809824, 0.02),
        ("PROP", result.sigma[0], 0.0107600, 0.02),
        ("ω²(CL)", result.omega[(0, 0)], 0.0288494, 0.05),
        ("ω²(V)", result.omega[(1, 1)], 0.00954401, 0.05),
        ("ω²(KA)", result.omega[(2, 2)], 0.335772, 0.05),
    ] {
        assert!(
            rel(got, want) < tol,
            "{name} {got} vs NONMEM {want} (rel {:.3e})",
            rel(got, want)
        );
    }
}

/// #2777: on `examples/one_cpt_transit.ferx` GN drives `TVV` onto its upper bound (500)
/// with the gradient still pushing outward. Before the fix, rho was measured against the
/// unclamped step's predicted reduction, which counted the reduction from the coordinate
/// the clamp holds still, so rho was a fixed 0.214 at every radius and the fit stopped on
/// "trust radius collapsed" at 1513.94. With the bound-active coordinate held and rho on
/// the clamped step it ends at 1492.1729 (Linux x86_64). The bound stays active, and the
/// gap to FOCEI (1215.98, in a different basin with `TVMTT` on its upper bound) is not
/// this test's object.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn gn_one_cpt_transit_does_not_collapse_on_the_volume_bound() {
    let model = parse_model_file(Path::new("examples/one_cpt_transit.ferx"))
        .expect("one_cpt_transit model must parse");
    let population = read_nonmem_csv(Path::new("data/datsim_oral.csv"), None, None)
        .expect("datsim_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceGn;
    opts.outer_maxiter = 500;
    opts.run_covariance_step = false;
    opts.verbose = false;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("GN fit must return a result");

    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("trust radius collapsed")),
        "GN collapsed its trust region: {:?}",
        result.warnings
    );
    assert!(result.ofv.is_finite(), "OFV must be finite: {}", result.ofv);
    // Measured 1492.1729 after the fix against 1513.94 before; 5 OFV of headroom.
    assert!(
        result.ofv < 1497.0,
        "GN OFV {} regressed toward the collapsed 1513.94",
        result.ofv
    );
}
