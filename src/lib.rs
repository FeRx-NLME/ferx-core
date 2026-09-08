pub mod api;
pub mod build_info;
pub mod cancel;
#[cfg(feature = "survival")]
pub mod categorical;
pub mod diagnostics;
pub(crate) mod dosing;
pub mod edit;
pub mod environment;
pub mod estimation;
pub mod frem;
pub mod io;
#[cfg(feature = "markov")]
pub mod markov;
pub mod model_selection;
#[cfg(feature = "nn")]
pub mod nn;
pub mod ode;
pub mod parser;
pub mod pk;
pub mod propensity_match;
pub mod sens;
pub(crate) mod serde_nalgebra;
pub mod sim;
pub mod stats;
pub mod suggest_start;
#[cfg(feature = "survival")]
pub mod survival;
pub mod types;

pub use api::{
    bind_theta_levels, check_model_data, check_model_data_warnings, check_model_options,
    configure_global_thread_pool, fit, fit_from_files, predict, prepare_run,
    prepare_run_with_inits, resolve_data_path, run_from_file, run_model_simulate,
    run_model_with_data, run_model_with_data_inits, simulate, simulate_adaptive,
    simulate_adaptive_from_spec, simulate_with_options, simulate_with_options_diag,
    simulate_with_seed, simulate_with_uncertainty, theta_level_map, validate_model_file,
    AdaptiveSimulateOptions, AdaptiveSimulationResult, PoolPlan, PredictionResult, PreparedRun,
    SimulateOptions, SimulateUncertaintyOptions, SimulationOutput, SimulationResult,
    FIT_RAYON_STACK_SIZE,
};
pub use cancel::CancelFlag;
pub use diagnostics::{CheckReport, Diagnostic, Severity};
pub use environment::EnvironmentInfo;
pub use estimation::run_covariance::run_covariance;
pub use estimation::run_sir::run_sir;
pub use estimation::uncertainty_samples::UncertaintyMethod;
pub use frem::{prepare_frem, FremDataInfo, FremFitInit, FremPrepareResult};
pub use io::datareader::{read_nonmem_csv, read_nonmem_csv_with_covariates};
pub use model_selection::{
    bic, check_strictness, estimate_near_boundary, max_abs_correlation, natural_scale_covariance,
    stalled_at_init, BicType, Strictness, StrictnessVerdict,
};
pub use parser::model_parser::{
    known_block_names, parse_full_model_file, parse_model_file, parse_model_string,
    ODE_INIT_REJECTED_BUILTINS, ODE_INIT_SCOPE_BUILTINS,
};
pub use propensity_match::MatchMethod;
// Adaptive (feedback) dosing vocabulary (#391). Re-exported at the crate root so
// the public `simulate_adaptive` API — its controller, monitors, and the fields
// of `AdaptiveSimulationResult` (`ledger` / `decisions`) — is usable without
// reaching into the `sim::adaptive` module path.
pub use sim::adaptive::{
    AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, AdaptiveSubjectMetrics,
    Comparison, ControllerCtx, DecisionLogEntry, DecisionOutcome, DoseAction, DoseLedgerEntry,
    DoseStep, MonitorSpec, ObserveMode, ObservedSignal,
};
pub use suggest_start::{inits_from_nca, NcaInit, SuggestedStart};
pub use types::*;

#[cfg(feature = "survival")]
pub use api::{predict_categorical, predict_survival, SurvivalPredictionResult};

/// Positive proof that the `debug_assert!` guards in this crate are LIVE
/// (#344, #1248).
///
/// `release`, `ci-test` and `ci-fast` all leave `debug-assertions` off, so all ~180
/// `debug_assert!`s compile to nothing there. `[profile.ci-cov]` turns them on, and
/// the two per-PR coverage jobs build under it — which both runs the guards and
/// stops their condition lines being regions that can never execute, the
/// permanently-missed patch lines of #1248.
///
/// That is an invariant held by *absence*, which is the fragile kind. Nothing in a
/// test result distinguishes a build with the guards live from one without: a
/// `[profile.ci-cov] debug-assertions = false` in `Cargo.toml`, a
/// `CARGO_PROFILE_CI_COV_DEBUG_ASSERTIONS=false` in the workflow environment, or a
/// `--profile` that quietly reverted to `ci-fast` would neuter the whole thing while
/// all 4366 tests and every preflight-contract test stayed green. The gate would
/// then be exactly the no-op #344 was filed to replace.
///
/// So each caller arms this canary explicitly — `FERX_REQUIRE_DEBUG_ASSERTIONS=1`,
/// on the coverage steps in `ci.yml` and in the argument vector of
/// `tools/preflight.sh`'s `debug-assertions` group (visible in `--list`, and
/// asserted by `tests/preflight_owns_the_fast_gates.rs`) — and the canary fails the
/// run when the guards turn out to be dead.
#[cfg(test)]
#[path = "lib_tests.rs"]
mod debug_assertion_canary;
