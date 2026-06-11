//! Integration coverage for the `covariance_fallback = sir` fit option (#223).
//!
//! The SIR fallback only *fires* when the FD Hessian is negative-semidefinite at
//! convergence (every free-block eigenvalue ≤ 0). That condition cannot be
//! produced deterministically from a real PK fit without a contrived concave
//! model, so the branch that maps a fired fallback to
//! `CovarianceStatus::SirFallback` is unit-tested directly in `api.rs`
//! (`resolve_covariance_status`) and the proposal construction is unit-tested in
//! `outer_optimizer.rs` (`build_non_pd_fallback_proposal`).
//!
//! What this Tier-2 test guards is the *wiring*: enabling the option on a
//! well-identified model must (a) run through `fit()` without panicking, and
//! (b) stay inert — it must not flip the status to `sir_fallback`, fabricate a
//! fallback warning, or otherwise change the outcome — because the covariance
//! step succeeds and the api.rs gate only runs the fallback when no covariance
//! matrix was produced. Warfarin is well-identified, so its Hessian is never
//! negative-semidefinite and the fallback must not trigger.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, CovarianceFallback, CovarianceStatus, FitOptions};
use std::path::Path;

fn warfarin_setup() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    FitOptions,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = true;
    opts.outer_maxiter = 60;
    (model, population, opts)
}

/// Enabling `covariance_fallback = sir` on a well-identified model runs through
/// `fit()` without panicking and does not fire the fallback: the status reaches
/// a real decision (not `NotRequested`, since we asked for it) and is never
/// `SirFallback`, and no SIR-fallback warning is emitted. This guards the api.rs
/// gate that only runs the fallback when `covariance_matrix.is_none()` for a
/// genuinely non-PD reason.
#[test]
fn covariance_fallback_sir_does_not_fire_on_well_identified_model() {
    let (model, population, mut opts) = warfarin_setup();
    opts.covariance_fallback = CovarianceFallback::Sir;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("warfarin fit with covariance_fallback = sir must succeed");

    // We requested the covariance step, so it must have reached a decision.
    assert_ne!(
        result.covariance_status,
        CovarianceStatus::NotRequested,
        "covariance was requested; status must not be NotRequested"
    );
    // Warfarin is well-identified: the Hessian is never negative-semidefinite, so
    // the fallback must not fire regardless of how far the fit converged.
    assert_ne!(
        result.covariance_status,
        CovarianceStatus::SirFallback,
        "SIR fallback must not fire on a well-identified model"
    );
    // A Computed status must carry a covariance matrix; a non-Computed status
    // must not (the fallback didn't run, so there is no SIR-derived matrix).
    if result.covariance_status == CovarianceStatus::Computed {
        assert!(
            result.covariance_matrix.is_some(),
            "Computed status must carry a covariance matrix"
        );
    }
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.to_lowercase().contains("sir fallback")),
        "no SIR-fallback warning should appear when the fallback did not fire: {:?}",
        result.warnings
    );
}

/// Toggling the fallback option must not change the reported status on a model
/// whose Hessian is never negative-semidefinite: whether the covariance step
/// succeeds (`Computed`) or fails for a non-`FailedNonPd` reason (`Failed`), the
/// fallback path is never taken, so `None` and `Sir` agree. This is the core
/// no-op invariant of the option on the success path.
#[test]
fn covariance_fallback_does_not_change_outcome_on_well_identified_model() {
    let (model, population, mut opts) = warfarin_setup();

    opts.covariance_fallback = CovarianceFallback::None;
    let baseline = fit(&model, &population, &model.default_params, &opts)
        .expect("warfarin fit (no fallback) must succeed");

    opts.covariance_fallback = CovarianceFallback::Sir;
    let with_fallback = fit(&model, &population, &model.default_params, &opts)
        .expect("warfarin fit (sir fallback) must succeed");

    assert_eq!(
        baseline.covariance_status, with_fallback.covariance_status,
        "fallback option must not alter the status when the Hessian is not non-PD"
    );
}
