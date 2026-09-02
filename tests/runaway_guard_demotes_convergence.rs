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
