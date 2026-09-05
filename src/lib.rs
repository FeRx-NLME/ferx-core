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

/// Positive proof that the `debug_assert!` guards in this crate are LIVE (#344).
///
/// Every other test job builds with `ci-test`/`ci-fast`, both of which
/// `inherits = "release"`, so all ~180 `debug_assert!`s compile to nothing. The
/// `Tests (debug-assertions)` job exists to run them, and it does that by simply
/// omitting `--profile` — the dev default has them on.
///
/// That is an invariant held by *absence*, which is the fragile kind. Nothing in
/// the command string distinguishes a build with the guards live from one without:
/// a `[profile.dev] debug-assertions = false` in `Cargo.toml`, or a
/// `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false` in the workflow environment, would
/// neuter the whole job while all 4366 tests, all nine preflight-contract tests and
/// both `cargo test` commands stayed green. The gate would then be exactly the
/// no-op it was filed to replace.
///
/// So the group arms this canary through its own argument vector
/// (`env FERX_REQUIRE_DEBUG_ASSERTIONS=1 cargo test …`, visible in
/// `tools/preflight.sh --list` and asserted by
/// `tests/preflight_owns_the_fast_gates.rs`), and the canary fails the run when the
/// guards turn out to be dead.
#[cfg(test)]
mod debug_assertion_canary {
    /// The env var `tools/preflight.sh`'s `debug-assertions` group sets. Deliberately
    /// opt-*in*: every other job legitimately runs with the guards off, so an
    /// unconditional assertion here would fail `Tests + coverage (core)` and both
    /// coverage jobs for doing exactly what they are supposed to do.
    const DEMAND: &str = "FERX_REQUIRE_DEBUG_ASSERTIONS";

    #[test]
    fn debug_assert_guards_run_when_the_gate_demands_them() {
        let demanded = std::env::var_os(DEMAND).is_some();

        // Observe the MACRO, not `cfg!(debug_assertions)`. The property the job exists
        // to establish is "the condition of a `debug_assert!` is evaluated"; reading
        // the cfg flag is one indirection away from it.
        //
        // `Cell` rather than `let mut`: with the guards off the assignment is compiled
        // away, and a `mut` binding that is then never mutated warns.
        let evaluated = std::cell::Cell::new(false);
        debug_assert!({
            evaluated.set(true);
            true
        });
        let evaluated = evaluated.get();

        assert_eq!(
            evaluated,
            cfg!(debug_assertions),
            "`debug_assert!` and `cfg!(debug_assertions)` disagree, which should be \
             impossible — the macro is defined in terms of that cfg. Read this as the \
             canary itself being broken rather than the profile being wrong."
        );

        // The gate. Note the shape: `!demanded || evaluated`, so the only run this can
        // fail is one that asked for the guards and did not get them. A single
        // expression rather than an `if`, so both profiles execute every line here and
        // the canary cannot read as uncovered on the patch gate.
        assert!(
            !demanded || evaluated,
            "{DEMAND} is set — this run came from `tools/preflight.sh debug-assertions` \
             or the `Tests (debug-assertions)` CI job — but `debug_assert!` compiled to \
             nothing, so every guard in the crate is dead and the job is green for no \
             reason. Something switched debug-assertions off for the dev profile: a \
             `[profile.dev] debug-assertions = false` in Cargo.toml, or a \
             `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS` in the environment (#344)."
        );
    }
}
