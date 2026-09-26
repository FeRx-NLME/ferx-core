//! What it costs a fit when a free Ω Cholesky diagonal **starts on one of its
//! own rails** (#1242).
//!
//! The two rails behave differently, and the difference is measured rather than
//! reasoned:
//!
//! * **Lower (`ln L = -6`, variance 6.144e-6).** Absorbing. With the
//!   `E_OMEGA_INIT_AT_RAIL` gate bypassed, `omega ETA_CL ~ 6.144212353328210e-6`
//!   on `examples/warfarin.ferx` + `data/warfarin.csv` leaves the coordinate
//!   **bit-identical for the whole run** — 300+ evaluations whose gradient points
//!   inward the entire time — ending at the rail with TVV 10× high and σ at 42%.
//!   Moving *only* the rail to `-6.0000001`, identical start, recovers the base
//!   optimum exactly. So the trap is the exact equality `packed_start == lower`.
//!   It has no test here because it has no fixture: since #1246 every free
//!   Ω / Ω_IOV / mixture-Ω diagonal at or below the rail is a hard
//!   `E_OMEGA_INIT_AT_RAIL` error before any optimizer runs, so no accepted model
//!   can reach the state. `api/tests/variance_init_rail_tests.rs` owns that gate.
//! * **Upper (`ln L = +6`, variance 1.628e5).** Not absorbing — and nothing
//!   rejects a start there, so unlike the lower rail this one is live. That is
//!   what the test below pins.
//!
//! Tier 3: a full population fit run to convergence, gated behind `slow-tests`.
//! The fixture's *premise* — that the declared decimal packs bit-exactly onto
//! the rail — is Tier 1 and lives in `estimation::parameterization`'s
//! `upper_rail_variance_fixture_packs_onto_the_rail`, where the rail constant
//! can be named rather than restated (PR #1408 review).

use ferx_core::{fit, parse_model_file, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// `exp(2 · 6)` — the variance whose packed coordinate `ln(√v)` is bit-exactly
/// the engine's upper Ω Cholesky-diagonal rail.
///
/// Written as the decimal the packer sees, because `OMEGA_CHOL_PACKED_UPPER` is
/// `pub(crate)` and cannot be named from an integration test. The premise that
/// this decimal really lands *on* the rail is not asserted here: it is pure
/// arithmetic over a crate-private constant, so it belongs on the fast PR path
/// and lives in `parameterization.rs`'s Tier-1
/// `upper_rail_variance_fixture_packs_onto_the_rail`, which names the constant
/// directly instead of restating it (PR #1408 review).
const UPPER_RAIL_VARIANCE: f64 = 162_754.791_419_003_92;

/// The warfarin base model, with `omega ETA_CL` spliced in.
///
/// Deliberately no `[fit_options]` block: the method, the iteration budget and
/// the covariance switch are set on the `FitOptions` handed to `fit`, which is
/// the only place they are guaranteed to arrive.
fn warfarin_model(omega_cl: &str) -> String {
    format!(
        "[parameters]\n\
         \x20 theta TVCL(0.2, 0.001, 10.0)\n\
         \x20 theta TVV(10.0, 0.1, 500.0)\n\
         \x20 theta TVKA(1.5, 0.01, 50.0)\n\
         \n\
         \x20 omega ETA_CL ~ {omega_cl}\n\
         \x20 omega ETA_V  ~ 0.04\n\
         \x20 omega ETA_KA ~ 0.30\n\
         \n\
         \x20 sigma PROP_ERR ~ 0.02 (sd)\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL)\n\
         V  = TVV  * exp(ETA_V)\n\
         KA = TVKA * exp(ETA_KA)\n\
         \n\
         [structural_model]\n\
         pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n"
    )
}

/// Fit the warfarin dataset with the given `omega ETA_CL` start.
fn fit_warfarin(dir: &Path, omega_cl: &str) -> FitResult {
    let model_path = dir.join(format!("warfarin_{}.ferx", omega_cl.replace('.', "_")));
    std::fs::write(&model_path, warfarin_model(omega_cl)).unwrap();
    let model = parse_model_file(&model_path).expect("model must parse");

    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/warfarin.csv");
    let population = read_nonmem_csv(&data_path, None, None).expect("warfarin.csv must read");

    let options = FitOptions {
        method: EstimationMethod::Foce,
        outer_maxiter: 300,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let init = model.default_params.clone();
    fit(&model, &population, &init, &options).expect("fit must reach an Ok")
}

/// Regression: an upper-rail twin of #1229's lower-rail gate, written on the
/// assumption that the two rails are symmetric. They are not — measured here —
/// and rejecting a start on the `+6` rail would refuse a model that fits
/// perfectly well.
///
/// Also the regression for the rail itself becoming absorbing at the top, which
/// is what the lower rail does: at `ln L = -6` the coordinate never moves again
/// (see the module docs). Nothing rejects a start on `+6`, so if that ever
/// happened the fit would come back silently wrong.
///
/// **Measured worst error** between the two arms on this fixture (10 subjects,
/// FOCE, macOS/arm64 debug), printed from the run rather than assumed:
///
/// | quantity | realised |
/// |---|---|
/// | OFV, absolute | `5.35e-10` |
/// | θ, relative (worst of three, `TVV`) | `1.37e-9` |
/// | ω², relative (worst of three) | `7.30e-11` |
///
/// Re-measured after #1389 introduced the warm-started inner BFGS Hessian seed
/// (2026-09, Windows/x86_64): the seed perturbs the two arms' outer trajectories,
/// and on the weakly-identified `TVKA` direction they now stop ~2e-6 apart while
/// still agreeing on the OFV to ~9e-10 — both arms at the documented optimum
/// (`-280.363962`), `converged = true`. The scatter is the derivative-free outer
/// optimizer's parameter tolerance on a flat direction, not a wrong basin; it is
/// intrinsic to any change of inner trajectory, so the θ/ω² bounds are re-derived
/// from the realised seeded error rather than the pre-seed one:
///
/// | quantity | realised (seeded, worst of both arms) | bound | headroom |
/// |---|---|---|---|
/// | OFV, absolute | `9e-10` | `1e-6` (unchanged) | ~1 100× |
/// | θ, relative (worst of three, `TVKA`) | `2.4e-6` (Linux CI: `4.8e-6`) | `1e-4` | ~20–40× |
/// | ω², relative (worst of three) | `1.4e-5` | `1e-4` | ~7× |
///
/// The OFV assertions stay three orders of magnitude tighter than the parameter
/// ones and remain the primary discriminator: a fit that was repelled rather than
/// solved (a rail failure) moves the OFV by orders of magnitude, and the hard
/// `DIVERGENCE_OFV` / `BASE_OPTIMUM_OFV` checks below still separate a solved fit
/// from a repelled one by eleven orders of magnitude.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn upper_omega_rail_start_still_reaches_the_base_optimum() {
    let dir = tempfile::tempdir().unwrap();

    let base = fit_warfarin(dir.path(), "0.09");
    let railed = fit_warfarin(dir.path(), &format!("{UPPER_RAIL_VARIANCE}"));

    // Every comparand checked *before* anything folds it, and `is_finite()` is
    // not the check. The inner objective clamps a blown-up value to a `1e20`
    // sentinel and the outer objective doubles it, so a fit that was **repelled**
    // rather than solved comes back as a perfectly finite `2e20` — and the
    // differential below would then compare two repelled arms to each other and
    // pass. Verified on PR #1408's review by overwriting both arms' `ofv` with
    // `2e20`: the test stayed green. So each arm must have converged, and each
    // OFV must sit below `DIVERGENCE_OFV` (`1e14`, the engine's own
    // repelled-vs-solved cutoff, `estimation::outer_optimizer`) *and* on the
    // real optimum.
    //
    // Realised margins on this fixture: the arms score -280.363962, which is
    // 3.6e11 x below the 1e14 cutoff and 7.1e17 x below the 2e20 sentinel, so
    // the bound separates a solved fit from a repelled one by eleven orders of
    // magnitude rather than by a tolerance.
    const DIVERGENCE_OFV: f64 = 1e14;
    const BASE_OPTIMUM_OFV: f64 = -280.363_962;
    for (label, arm) in [("base", &base), ("upper rail", &railed)] {
        assert!(
            arm.converged,
            "{label} arm did not converge (OFV {})",
            arm.ofv
        );
        assert!(
            arm.ofv.is_finite() && arm.ofv < DIVERGENCE_OFV,
            "{label} arm OFV {} is not a solved objective — at or above the \
             {DIVERGENCE_OFV:e} repelled cutoff, or non-finite",
            arm.ofv
        );
        assert!(
            (arm.ofv - BASE_OPTIMUM_OFV).abs() < 1e-3,
            "{label} arm OFV {} is not the base optimum {BASE_OPTIMUM_OFV}",
            arm.ofv
        );
        for (name, v) in arm.theta_names.iter().zip(&arm.theta) {
            assert!(v.is_finite(), "{label} arm {name} is not finite: {v}");
        }
        for (i, v) in arm.omega.diagonal().iter().enumerate() {
            assert!(
                v.is_finite(),
                "{label} arm omega[{i},{i}] is not finite: {v}"
            );
        }
    }

    assert!(
        (railed.ofv - base.ofv).abs() < 1e-6,
        "a start on the +6 rail must reach the base optimum: OFV {} vs base {}",
        railed.ofv,
        base.ofv
    );

    for (i, name) in base.theta_names.iter().enumerate() {
        let (b, r) = (base.theta[i], railed.theta[i]);
        assert!(
            (r - b).abs() <= 1e-4 * b.abs().max(1.0),
            "{name}: {r} from the +6 rail vs {b} from the base start"
        );
    }

    for i in 0..base.omega.nrows() {
        let (b, r) = (base.omega[(i, i)], railed.omega[(i, i)]);
        assert!(
            (r - b).abs() <= 1e-4 * b.abs().max(1e-8),
            "omega[{i},{i}]: {r} from the +6 rail vs {b} from the base start"
        );
    }

    // The straddle, asserted rather than assumed: ETA_CL really did start far
    // from where the base arm started, so the agreement above is two
    // trajectories meeting and not one fixture written twice.
    // (measured: ln(√0.09) = -1.204 against the rail's +6.0, a gap of 7.204 out
    // of the box's 12 units of width).
    let base_start = 0.09_f64.sqrt().ln();
    assert!(
        UPPER_RAIL_VARIANCE.sqrt().ln() - base_start > 7.0,
        "the two arms must start far apart in the box: base packs at {base_start}, \
         the rail arm at {}",
        UPPER_RAIL_VARIANCE.sqrt().ln()
    );
}
