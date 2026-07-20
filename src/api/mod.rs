use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::io::datareader::{ERR_COV_MISSING_COLUMNS, ERR_COV_NON_NUMERIC};
use crate::pk;
use crate::types::*;
#[cfg(test)]
use nalgebra::{DMatrix, DVector};
#[cfg(test)]
use rand::RngExt;
use rayon::prelude::*;
use std::path::Path;

// ── validation subsystem (extracted verbatim; see src/api/validation.rs) ──
mod validation;
pub(crate) use validation::{
    apply_iov_occasion_rule, assert_absorption_closed_form_support,
    assert_absorption_dosing_supported, assert_absorption_flip_flop_no_twin,
    assert_analytic_readout_support, assert_modeled_doses_supported,
    check_absorption_closed_form_support, check_absorption_dosing,
    check_absorption_flip_flop_no_twin, check_analytic_readout_support, check_covariates,
    check_modeled_dose_rates,
};
#[cfg(feature = "survival")]
pub(crate) use validation::{
    assert_survival_tv_covariates, check_rtte_records, check_survival_tv_covariates,
};
pub use validation::{
    check_experimental_features, check_model_data, check_model_data_rule,
    check_model_data_warnings, check_model_options, validate_model_file, validate_output_columns,
};

// ── production submodules (peeled from this file) ──
mod adaptive;
mod fit;
mod output_columns;
mod pool;
mod postfit;
mod predict;
mod run;
mod simulate;

pub use adaptive::{
    simulate_adaptive, simulate_adaptive_from_spec, AdaptiveSimulateOptions,
    AdaptiveSimulationResult,
};
pub use fit::{fit, fit_from_files};
pub use output_columns::tafd_tad_for_subject;
pub(crate) use output_columns::{compute_extra_output_columns, trapezoid};
pub use pool::configure_global_thread_pool;
pub(crate) use pool::{build_fit_pool, default_fit_pool};
pub(crate) use postfit::{
    absorption_flip_flop_ebe_warning, boundary_estimate_warning, compute_eps_shrinkage,
    compute_eta_shrinkage, compute_kappa_shrinkage, compute_kappa_shrinkage_by_occ,
    compute_param_corr, compute_subject_results, cov_diagnostics, eps_shrinkage_warning,
    eta_shrinkage_warning, extract_standard_errors, high_correlation_warning, inflated_rse_warning,
    is_last_estimating_stage, probe_nlopt_algorithms, rebuild_warnings_structured,
    resolve_covariance_status, resolve_sir_fallback,
};
pub use predict::{predict, PredictionResult};
#[cfg(feature = "survival")]
pub use predict::{predict_categorical, predict_survival, SurvivalPredictionResult};
pub(crate) use run::{build_selection_filter_merged, log_transform_observations};
pub use run::{
    read_population_for, resolve_data_path, run_from_file, run_model_simulate, run_model_with_data,
    run_model_with_data_inits,
};
pub(crate) use simulate::obs_row_time;
pub use simulate::{
    simulate, simulate_with_options, simulate_with_options_diag, simulate_with_seed,
    simulate_with_uncertainty, SimulateOptions, SimulateUncertaintyOptions, SimulationOutput,
    SimulationResult,
};

// ── test-support re-exports (reached via `super::` from the relocated
//    #[cfg(test)] test siblings; production code uses these items in place) ──
#[cfg(test)]
pub(crate) use adaptive::{
    pk_bits_eq, reject_selected_error_for_adaptive, reject_unsupported_adaptive,
    verify_adaptive_snapshots,
};
#[cfg(test)]
pub(crate) use fit::{
    multistart_prefers, perturb_init, saem_non_mu_referenced_individual_params_warning,
};
#[cfg(test)]
pub(crate) use output_columns::build_indiv_map;
#[cfg(test)]
pub(crate) use pool::{cap_default_threads, default_thread_count, FIT_RAYON_STACK_SIZE};
#[cfg(test)]
pub(crate) use postfit::{
    diagnostic_details, high_correlation_pairs, should_run_sir_fallback, theta_boundary_side,
    DiagStats,
};
#[cfg(all(test, feature = "survival"))]
pub(crate) use predict::grid_median_from_cumhaz;
#[cfg(test)]
pub(crate) use run::derive_output_occasions;
#[cfg(test)]
pub(crate) use simulate::emit_correlated_residual_rows;

/// Route predictions through analytical PK or ODE solver, then apply
/// `model.scaling` so simulate / predict / post-fit IPRED see the same
/// scaled output as the estimation dispatcher in `pk::compute_predictions_with_tv_into_with_schedule`.
///
/// `theta` and `eta` are required so that `ScalingSpec::ExpressionScale`
/// can evaluate its `scale_fn(theta, eta, covariates)`. Callers that don't
/// have a separate eta vector (population predictions) pass an all-zero eta.
///
/// Production code routes through [`pk::compute_predictions_with_tv`] (the
/// TV-covariate-aware dispatcher) instead; this baseline-only helper now only
/// backs the TV-vs-no-TV gap assertions in the regression tests.
#[cfg(test)]
pub(crate) fn model_preds(
    model: &CompiledModel,
    subject: &Subject,
    pk_params: &PkParams,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    let mut preds = if let Some(ref ode_spec) = model.ode_spec {
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values, theta, eta)
    } else {
        // Resolve any modeled-`RATE` doses (#324/#394, e.g. `RATE=-2` → `D{cmt}`)
        // to a concrete duration/rate before the analytical closed form — mirrors
        // the ODE `resolve_subject_doses` step inside `compute_predictions_ode`.
        // Borrowed (no allocation) for the all-`Fixed` common case.
        let resolved = crate::dosing::resolve_subject_doses(
            subject,
            model.active_dose_attr_map(),
            &pk_params.values,
        );
        pk::compute_predictions(model.pk_model, &resolved, pk_params)
    };
    // Analytic Form C readout (#650): replaces the built-in concentration. No-op
    // for ODE models (handled inside `compute_predictions_ode`) and when unset.
    pk::apply_analytic_readout(model, subject, theta, eta, &mut preds);
    pk::apply_scaling(model, subject, theta, eta, &mut preds);
    pk::apply_log_transform(model, &mut preds);
    preds
}

// ── Step 7: [output] validation and TAFD/TAD helpers ────────────────────────

/// Mandatory sdtab column names that are always written — declaring them in
/// [output] is allowed but produces a W_OUTPUT_DUPLICATE warning.
const OUTPUT_MANDATORY: &[&str] = &[
    "ID", "TIME", "DV", "CENS", "OCC", "CMT", "PRED", "IPRED", "CWRES", "IWRES", "NPDE", "NPD",
    "EBE_OFV", "N_OBS", "TAFD", "TAD",
];

// ── Step 8: post-fit extra column computation ────────────────────────────────

#[cfg(test)]
#[path = "tests/multistart_prefers_tests.rs"]
mod multistart_prefers_tests;

#[cfg(test)]
#[path = "tests/build_indiv_map_tests.rs"]
mod build_indiv_map_tests;

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

// ======================================================================
// Adaptive (state-reactive / feedback) dosing — epic #391, beta.
//
// Two public entry points wrap the `pub(crate)` reactive driver
// (`ode_predictions_adaptive_impl`) with shared per-(subject, replicate)
// orchestration (`run_adaptive_population`), the tagged-row output schema
// (Part D), and the frozen-schedule replay verifier (Part E backbone,
// default-on): `simulate_adaptive` takes a hand-written controller closure, and
// `simulate_adaptive_from_spec` compiles a declarative `[adaptive_dosing]` block
// into the same engine. Both support Ipred and `Dv` (assay-noised) monitors on
// the S1.5 controller-assay substreams.
// ======================================================================

// ── TTE / survival prediction ─────────────────────────────────────────────────

#[cfg(all(test, feature = "survival"))]
#[path = "tests/survival_predict_tests.rs"]
mod survival_predict_tests;

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci`.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests/iov_integration.rs"]
mod iov_integration;

#[cfg(test)]
#[path = "tests/extract_se_tests.rs"]
mod extract_se_tests;

#[cfg(test)]
#[path = "tests/tests_cov_diagnostics.rs"]
mod tests_cov_diagnostics;

#[cfg(test)]
#[path = "tests/tests_sir_fallback.rs"]
mod tests_sir_fallback;

#[cfg(test)]
#[path = "tests/tests_param_corr.rs"]
mod tests_param_corr;

#[cfg(test)]
#[path = "tests/simulate_with_uncertainty_tests.rs"]
mod simulate_with_uncertainty_tests;

// ── SDE end-to-end integration ───────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/sde_integration.rs"]
mod sde_integration;

#[cfg(test)]
#[path = "tests/multi_start_tests.rs"]
mod multi_start_tests;

#[cfg(test)]
#[path = "tests/tests_sdtab_tv_cov.rs"]
mod tests_sdtab_tv_cov;

#[cfg(test)]
#[path = "tests/tests_derived_session_clock.rs"]
mod tests_derived_session_clock;

#[cfg(test)]
#[path = "tests/tests_derived_iov_kappa.rs"]
mod tests_derived_iov_kappa;

// ── Tests: adaptive (state-reactive) dosing — simulate_adaptive (#391 S1.4b) ──
#[cfg(test)]
#[path = "tests/adaptive_sim_tests.rs"]
mod adaptive_sim_tests;

// ── Tests: #748 independent adaptive-snapshot correctness check ──────────────
//
// The frozen-replay verifier reuses the precomputed snapshots (`eta_occ`,
// `decision_pk`, `event_pk`) the driver was handed, so it validates their
// *consumption*, not their *correctness* — a build-loop that feeds a leaf the
// wrong covariate / occasion / κ passes it bit-exact. These tests pin the teeth of
// `verify_adaptive_snapshots`, the independent re-derivation that closes that gap:
// a canonical build is accepted, and each deliberately-corrupted snapshot class
// (the twice-fixed #732 / #739 "decision covariate frozen at t=0" defect included)
// is rejected. Every corruption is constructed to differ from the canonical build,
// so the tests fail if the check ever regresses to a no-op.
#[cfg(test)]
#[path = "tests/adaptive_snapshot_verify_tests.rs"]
mod adaptive_snapshot_verify_tests;
