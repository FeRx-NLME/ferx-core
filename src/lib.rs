pub mod api;
pub mod build_info;
pub mod cancel;
pub mod diagnostics;
pub mod environment;
pub mod estimation;
pub mod frem;
pub mod io;
#[cfg(feature = "nn")]
pub mod nn;
pub mod ode;
pub mod parser;
pub mod pk;
pub mod propensity_match;
pub mod sens;
pub mod sim;
pub mod stats;
pub mod suggest_start;
#[cfg(feature = "survival")]
pub mod survival;
pub mod types;

pub use api::{
    check_model_data, check_model_data_warnings, check_model_options, configure_global_thread_pool,
    fit, fit_from_files, predict, resolve_data_path, run_from_file, run_model_simulate,
    run_model_with_data, run_model_with_data_inits, simulate, simulate_adaptive,
    simulate_adaptive_from_spec, simulate_with_options, simulate_with_seed,
    simulate_with_uncertainty, validate_model_file, AdaptiveSimulateOptions,
    AdaptiveSimulationResult, PredictionResult, SimulateOptions, SimulateUncertaintyOptions,
    SimulationResult,
};
pub use cancel::CancelFlag;
pub use diagnostics::{CheckReport, Diagnostic, Severity};
pub use environment::EnvironmentInfo;
pub use estimation::run_covariance::run_covariance;
pub use estimation::run_sir::run_sir;
pub use estimation::uncertainty_samples::UncertaintyMethod;
pub use frem::{prepare_frem, FremDataInfo, FremFitInit, FremPrepareResult};
pub use io::datareader::{read_nonmem_csv, read_nonmem_csv_with_covariates};
pub use parser::model_parser::{parse_full_model_file, parse_model_file, parse_model_string};
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
pub use api::{predict_survival, SurvivalPredictionResult};
