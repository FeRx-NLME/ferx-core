//! #1118: a fit whose estimate is pinned to an *internal* packed-space guard is
//! not an interior optimum, so `converged` must not come back `true`.
//!
//! The unit tests in `src/api/postfit.rs`'s sibling own the side/severity rules;
//! this one pins the wiring — that `fit()` actually applies them to the
//! `FitResult` every consumer keys off, for every estimator.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::{WarningCode, WarningSeverity};
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// An evaluation-only IMP stage reports the likelihood at the parameters it is
/// handed without moving them, which is what lets this exercise the fit-end
/// check on a guard-pinned point without a convergence loop (Tier 2).
#[test]
fn fit_demotes_converged_when_sigma_sits_on_the_internal_ceiling() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.method = EstimationMethod::Imp;
    opts.imp_eval_only = true;

    // Baseline: the same evaluation at the model's own inits is interior, so a
    // demotion below is attributable to the guard and not to the eval-only path.
    let interior = fit(&model, &population, &model.default_params, &opts)
        .expect("eval-only IMP must succeed at the inits");
    assert!(
        interior.converged,
        "control: an interior evaluation must not be demoted"
    );

    // exp(5) is the literal packed upper guard on a log-packed sigma.
    let mut params = model.default_params.clone();
    params.sigma.values[0] = 5.0_f64.exp();
    let pinned =
        fit(&model, &population, &params, &opts).expect("eval-only IMP must succeed at the guard");

    assert!(
        !pinned.converged,
        "an estimate held at an implementation ceiling is not a converged fit"
    );
    let entry = pinned
        .warnings_structured
        .iter()
        .find(|w| w.category == WarningCode::ParameterAtRunawayGuard)
        .expect("the guard hit must be reported");
    assert_eq!(entry.severity, WarningSeverity::Critical);
    assert_eq!(
        entry.details.as_ref().unwrap()["parameters"][0]["side"],
        "upper"
    );
}

/// #1328: the coordinate that ran away on the busulfan DCM subset was an Ω
/// **off-diagonal** at the ±10 rail, not a Σ ceiling, and it reaches this check
/// by a different path. The Σ arm above hands `fit()` a number it stores
/// verbatim; a block Ω's rail lives on the raw Cholesky element `L[i,j]`, which
/// `fit()` packs, clamps in scaled space, unpacks into a covariance matrix and
/// re-reads at fit end. Nothing at Tier 1 crosses that round trip — the unit
/// tests in `api/postfit`'s sibling build an `OmegaMatrix` and re-pack it in the
/// same breath.
///
/// Both rails are asserted because the side does not decide the verdict here
/// (#1205): `L[i,j]` is bounded symmetrically, so −10 is the same runaway as
/// +10 and must demote too.
///
/// The demotion is *attributed*, not merely observed. `ω²_V ≥ L₂₁² = 100` at
/// either rail — the same 100–450 #1328 reports — so the evaluation is a bad
/// one, and a test that asserted only `!converged` would pass on #1303's
/// objective gate firing instead. Hence the two premise assertions: the
/// objective is finite and nowhere near the `DIVERGENCE_OFV` = 1e14 cutoff, and
/// no `W_NONFINITE_OBJECTIVE` warning is present.
#[test]
fn fit_demotes_converged_when_a_block_omega_off_diagonal_sits_on_either_rail() {
    use ferx_core::OmegaMatrix;

    let model = parse_model_file(Path::new("examples/warfarin_block_omega.ferx"))
        .expect("the block-omega example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.method = EstimationMethod::Imp;
    opts.imp_eval_only = true;

    // Control, on this model rather than the Σ one: the declared block
    // (`[0.09, 0.02, 0.04]`, i.e. L₂₁ = 0.067) is interior, so a demotion below
    // is attributable to the rail and not to the block or to the eval-only path.
    let interior = fit(&model, &population, &model.default_params, &opts)
        .expect("eval-only IMP must succeed at the inits");
    assert!(
        interior.converged,
        "control: an interior block Ω must not be demoted"
    );
    assert!(
        !interior
            .warnings_structured
            .iter()
            .any(|w| w.category == WarningCode::ParameterAtRunawayGuard),
        "control: no guard hit at the declared block"
    );

    for rail in [10.0_f64, -10.0_f64] {
        let mut params = model.default_params.clone();
        let mut chol = params.omega.chol.clone();
        // Row 1, column 0 is L₂₁ — the (ETA_CL, ETA_V) block's off-diagonal.
        // ETA_KA is a separate diagonal omega; its structural zeros are held by
        // `free_mask` and are untouched here.
        chol[(1, 0)] = rail;
        params.omega = OmegaMatrix::from_chol_factor(
            chol,
            params.omega.eta_names.clone(),
            params.omega.diagonal,
            params.omega.free_mask.clone(),
        );

        let pinned =
            fit(&model, &population, &params, &opts).expect("eval-only IMP must succeed at a rail");

        // Premise: this is a *guard* demotion, not #1303's objective demotion.
        assert!(
            pinned.ofv.is_finite() && pinned.ofv.abs() < 1e14,
            "rail {rail}: the objective must stay reportable so the demotion is \
             attributable to the guard, got {}",
            pinned.ofv
        );
        assert!(
            !pinned
                .warnings
                .iter()
                .any(|w| w.contains("W_NONFINITE_OBJECTIVE")),
            "rail {rail}: #1303's gate must not be the one firing: {:#?}",
            pinned.warnings
        );

        assert!(
            !pinned.converged,
            "rail {rail}: a block Ω off-diagonal held at an implementation rail \
             is not a converged fit"
        );
        let entry = pinned
            .warnings_structured
            .iter()
            .find(|w| w.category == WarningCode::ParameterAtRunawayGuard)
            .unwrap_or_else(|| panic!("rail {rail}: the guard hit must be reported"));
        assert_eq!(entry.severity, WarningSeverity::Critical, "rail {rail}");
        let hit = &entry.details.as_ref().unwrap()["parameters"][0];
        assert_eq!(hit["parameter"], "ETA_V~ETA_CL", "rail {rail}");
        assert_eq!(hit["verdict"], "runaway", "rail {rail}: either rail is one");
        assert_eq!(
            hit["side"],
            if rail > 0.0 { "upper" } else { "lower" },
            "rail {rail}"
        );
        // The round trip this test exists for: the coordinate `fit()` re-reads
        // at the end is the literal rail, bit for bit, and not a Cholesky
        // re-factorisation a few ULPs off it.
        assert_eq!(
            hit["packed_estimate"].as_f64().unwrap(),
            rail,
            "rail {rail}: the packed coordinate must survive the fit round trip exactly"
        );
    }
}
