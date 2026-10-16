//! The codes of the refusals only an entry point raises (#1774).
//!
//! `ferx check` cannot reach these: it has no `SimulateOptions`, no adaptive controller
//! and no adaptive spec, and it does not ask whether a model can be predicted or
//! simulated (`fit()` supports CTMM, so a CTMM model is valid). Each code is assigned
//! here and nowhere else, one constructor per cause, so two call sites refusing for the
//! same reason cannot give two codes. Every constructor keeps the caller's message as
//! is, so `to_string()` is the text the refusal had before it carried a code.
//!
//! The two refusals that reuse a code `ferx check` or another entry point already has
//! (`E_PARAM_SHAPE`, `E_MISSING_COVARIATE`) are built here too, by name.
//!
//! So are the input refusals of the post-hoc steps on a fit, `run_sir` and
//! `run_covariance` (#1774 part b): `ferx check` has no fit, so it cannot reach an
//! input missing from one (`E_FIT_INPUT_MISSING`), a file changed since it
//! (`E_FIT_INPUT_CHANGED`) or a model that is not its own (`E_FIT_INPUT_MODEL`,
//! and `E_THETA_LENGTH` for the θ count), and it runs no SIR, so it cannot refuse
//! SIR's settings (`E_SIR_OPTIONS`) or a fit with nothing free to sample
//! (`E_SIR_NOTHING_TO_SAMPLE`). Their re-reads of the model and data files are
//! not here: those failures are `ferx check`'s own, and are coded by its read
//! classifiers in `validation.rs`.

use crate::diagnostics::{Diagnostic, EngineError};

/// The one builder: an error diagnostic with an optional block.
fn diag(code: &str, block: Option<&str>, message: String) -> Diagnostic {
    let d = Diagnostic::error(code, message);
    match block {
        Some(b) => d.with_block(b),
        None => d,
    }
}

fn coded(code: &str, block: Option<&str>, message: String) -> EngineError {
    EngineError::from_diagnostic(diag(code, block, message))
}

/// `E_PARAM_SHAPE`: an Ω, σ or Ω_IOV whose shape is not the model's (#1764), including
/// a missing Ω_IOV on a model with κ, which `simulate*` refuses with #1019's message.
pub(crate) fn param_shape(message: impl Into<String>) -> EngineError {
    coded("E_PARAM_SHAPE", None, message.into())
}

/// `E_MISSING_COVARIATE` on block `adaptive_dosing`: an `[adaptive_dosing] observe`
/// expression reads a covariate the data lack, the cause `ferx check` gives that code for
/// a model expression. `simulate_adaptive_from_spec` adds its context.
pub(crate) fn observe_covariate_missing(message: impl Into<String>) -> EngineError {
    coded(
        "E_MISSING_COVARIATE",
        Some("adaptive_dosing"),
        message.into(),
    )
}

/// `E_NESTED_LEVEL_SIMULATE_UNSUPPORTED`: a model with nested levels (more than one
/// `level`, #1082) reached a `simulate*` or adaptive entry point, which draw one κ per
/// occasion and have no per-level draw yet.
pub(crate) fn nested_level_simulate_unsupported() -> EngineError {
    coded(
        "E_NESTED_LEVEL_SIMULATE_UNSUPPORTED",
        Some("parameters"),
        "simulation does not yet support nested levels (more than one `level`): it draws \
         one kappa per occasion, and a kappa at an outer level must be drawn once per group \
         and shared by the groups inside it. Fit the model; simulate from it in a later \
         release."
            .to_string(),
    )
}

/// `E_CTMM_UNSUPPORTED`: a `[markov_model]` endpoint reached `predict*` / `simulate*`,
/// which have no state-occupancy path yet (#759, #820).
#[cfg(feature = "markov")]
pub(crate) fn ctmm_unsupported(message: impl Into<String>) -> EngineError {
    coded("E_CTMM_UNSUPPORTED", Some("markov_model"), message.into())
}

/// `E_TTE_NOT_SIMULATABLE`: a TTE endpoint and population the event-time simulator
/// cannot draw from — more than one recurrent stream on a subject, a recurrent cause
/// beside a competing single-event one, an EVID=3/4 reset under a recurrent or an
/// ODE-accumulated hazard, or left truncation under an ODE-accumulated hazard.
#[cfg(feature = "survival")]
pub(crate) fn tte_not_simulatable(message: impl Into<String>) -> EngineError {
    coded("E_TTE_NOT_SIMULATABLE", Some("event_model"), message.into())
}

/// `E_SIMULATE_HORIZON`: the simulation's administrative horizon is missing where a
/// recurrent or ODE-accumulated TTE endpoint needs one, is not finite and positive, or
/// lies below a subject's entry time.
pub(crate) fn simulate_horizon(message: impl Into<String>) -> EngineError {
    coded("E_SIMULATE_HORIZON", None, message.into())
}

/// `E_PROPENSITY_INPUT`: propensity-score matching was requested on a population it
/// cannot compute posthoc etas from — empty, a subject with no observations, or a
/// subject with a non-finite `DV`.
pub(crate) fn propensity_input(message: impl Into<String>) -> EngineError {
    coded("E_PROPENSITY_INPUT", None, message.into())
}

/// `E_ADAPTIVE_UNSUPPORTED`: a model the adaptive-dosing driver cannot simulate
/// faithfully — no `[odes]` block, an SDE, a covariate-selected `[error_model]`, a
/// residual `correlation` line — or an `auc_target` on a subject whose exposure metric
/// it cannot integrate exactly. Once the driver runs (#1983): a base regimen it cannot
/// replay under a time-varying covariate or IOV, a non-finite break time, or a dose
/// clock the RHS reads before any dose. `block` is the block at fault, when one is.
pub(crate) fn adaptive_unsupported(block: Option<&str>, message: impl Into<String>) -> EngineError {
    EngineError::from_diagnostic(adaptive_unsupported_diag(block, message))
}

/// [`adaptive_unsupported`] as a [`Diagnostic`], for `compile_adaptive`'s ODE gate.
pub(crate) fn adaptive_unsupported_diag(
    block: Option<&str>,
    message: impl Into<String>,
) -> Diagnostic {
    diag("E_ADAPTIVE_UNSUPPORTED", block, message.into())
}

/// `E_ADAPTIVE_OPTIONS`: `AdaptiveSimulateOptions` that give no usable schedule — an
/// empty or not finite and strictly increasing `decision_times`, more decisions than
/// `max_decisions`, or one the driver cannot key against a subject's own break times
/// (#1983) — monitors it cannot read, or that set `decision_times` / `monitors` where
/// the `[adaptive_dosing]` spec already owns them.
pub(crate) fn adaptive_options(message: impl Into<String>) -> EngineError {
    coded("E_ADAPTIVE_OPTIONS", None, message.into())
}

/// `E_ADAPTIVE_CONTROLLER_ACTION`: a controller returned an action the driver refuses —
/// a dose with a non-finite or negative amount, a rate that is not positive and finite,
/// compartment 0 or one past the model's states, a compartment fed by an input-rate
/// function or carrying a lag, or a `Stop` with actions after it (#1983). The
/// controller is the caller's code, so this is an input refusal, although it depends on
/// the simulated state and `ferx check` cannot see it. `simulate_adaptive` only: on
/// `simulate_adaptive_from_spec` the controller is compiled from the spec's `route`, so
/// the same refusal is [`adaptive_spec`].
pub(crate) fn adaptive_controller_action(message: impl Into<String>) -> EngineError {
    coded("E_ADAPTIVE_CONTROLLER_ACTION", None, message.into())
}

/// A refusal of the reactive driver (`ode_predictions_adaptive_impl`) as the code its
/// kind names (#1983). Its runtime failures carry no code and do not come here.
pub(crate) fn adaptive_driver_refusal(
    kind: crate::ode::AdaptiveRefusalKind,
    message: String,
) -> EngineError {
    use crate::ode::AdaptiveRefusalKind as K;
    match kind {
        K::Options => adaptive_options(message),
        K::Unsupported => adaptive_unsupported(None, message),
        K::ControllerAction => adaptive_controller_action(message),
    }
}

/// `E_ADAPTIVE_SPEC`: an `[adaptive_dosing]` spec that does not compile against the
/// model — an invariant the block parser checks, an `assay_cmt` out of range or with no
/// `[error_model]` to draw its assay from (#1983), a `route` compartment out of range or
/// fed by an input-rate function, a `route` dose the driver refuses (#1983), or an
/// `observe` expression that does not compile.
pub(crate) fn adaptive_spec(message: impl Into<String>) -> Diagnostic {
    diag("E_ADAPTIVE_SPEC", Some("adaptive_dosing"), message.into())
}

/// `E_THETA_LENGTH`: a θ whose length is not the model's (#1614, #1615) — a
/// parameter vector handed to a `simulate*` / `predict*` entry point, or the fit's θ
/// against the model `run_sir` / `run_covariance` run on (#1774).
pub(crate) fn theta_length(message: impl Into<String>) -> EngineError {
    coded("E_THETA_LENGTH", None, message.into())
}

/// `E_FIT_INPUT_MISSING`: a post-hoc step on a fit (`run_sir`, `run_covariance`)
/// needs an input that neither the caller passed nor the fit records — the model
/// file's path, the data file's path, SIR's covariance matrix, the population of an
/// IOV model passed as `model = Some` on a fit with no recorded reader settings, the
/// occasion rule of an IOV fit whose population carries no occasion labels, or the
/// fitted values of a `[mixture]` fit's per-class overrides (which
/// `simulate_with_uncertainty` needs too).
pub(crate) fn fit_input_missing(message: impl Into<String>) -> EngineError {
    coded("E_FIT_INPUT_MISSING", None, message.into())
}

/// `E_FIT_INPUT_CHANGED`: a file a post-hoc step re-reads (`fit.model_path`,
/// `fit.data_path`) no longer has the SHA-256 the fit recorded.
pub(crate) fn fit_input_changed(message: impl Into<String>) -> EngineError {
    EngineError::from_diagnostic(fit_input_changed_diag(message))
}

/// [`fit_input_changed`] as a [`Diagnostic`], for the model-read classifier
/// (`validation::model_source_diagnostic`), which codes the model file's hash
/// failure beside its read and parse failures.
pub(crate) fn fit_input_changed_diag(message: impl Into<String>) -> Diagnostic {
    diag("E_FIT_INPUT_CHANGED", None, message.into())
}

/// `E_FIT_INPUT_MODEL`: the model a post-hoc step would run on is not the one the
/// fit was made with — a supplied model bound with other data-derived bindings than
/// the fit's, or (on a fit that records none) with covariate statistics the
/// population does not reproduce, a model whose η count is not the width of the
/// fit's per-subject η̂, a `[mixture]` fit whose packed estimate has another layout
/// or no longer reproduces its reported estimates, or a covariance matrix that is
/// not the size of the model's packed parameters (SIR's proposal).
pub(crate) fn fit_input_model(message: impl Into<String>) -> EngineError {
    coded("E_FIT_INPUT_MODEL", None, message.into())
}

/// `E_SIR_OPTIONS`: SIR settings the importance resampler cannot run with —
/// `sir_resamples` above `sir_samples`, a `sir_df` that is not a valid Student-t
/// degrees of freedom, or `sir_scale = natural` on a model with `prior(...)`.
pub(crate) fn sir_options(message: impl Into<String>) -> EngineError {
    coded("E_SIR_OPTIONS", None, message.into())
}

/// `E_ODE_SOLVER_OPTIONS`: ODE solver settings on a caller's `FitOptions` that no
/// solver can run with — an `ode_reltol` / `ode_abstol` that is not finite and
/// positive, or `ode_max_steps = 0` (#2044). A model file's `[fit_options]` never
/// gets here: the parser refuses the same values as `E_PARSE`, so only a hand-built
/// `FitOptions` handed to `run_sir` / `run_covariance` / `run_sir_core` reaches it.
pub(crate) fn ode_solver_options(message: impl Into<String>) -> EngineError {
    coded("E_ODE_SOLVER_OPTIONS", None, message.into())
}

/// `E_LOGIT_THETA_OUT_OF_RANGE` on block `parameters`: a free θ used as
/// `inv_logit(logit(θ) + ETA)` whose estimate is not strictly inside (0, 1), so it has
/// no logit-scale draw (#1548, #2044). Reachable only with a declared upper bound
/// above 1; the suggestion is that fix, and `Display` folds it in after the message,
/// so `to_string()` is the text the refusal had before it carried a code.
pub(crate) fn logit_theta_out_of_range(name: &str, estimate: f64) -> Diagnostic {
    diag(
        "E_LOGIT_THETA_OUT_OF_RANGE",
        Some("parameters"),
        format!(
            "Cannot draw parameter uncertainty for {name}: it is used as \
             inv_logit(logit({name}) + ETA), so it must lie strictly inside \
             (0, 1), but its estimate is {estimate}."
        ),
    )
    .with_suggestion("declare its upper bound below 1 (e.g. 0.999) and refit")
}

/// `E_SIR_NOTHING_TO_SAMPLE`: every packed parameter of the fit is FIX, so SIR has
/// no free direction to propose in. The model's own `[parameters]`, not the SIR
/// settings, which is why it is not `E_SIR_OPTIONS`.
pub(crate) fn sir_nothing_to_sample(message: impl Into<String>) -> EngineError {
    coded(
        "E_SIR_NOTHING_TO_SAMPLE",
        Some("parameters"),
        message.into(),
    )
}
