//! The codes of the refusals only an entry point raises (#1774).
//!
//! `ferx check` cannot reach these: it has no `SimulateOptions`, no adaptive controller
//! and no adaptive spec, and it does not ask whether a model can be predicted or
//! simulated (`fit()` supports CTMM, so a CTMM model is valid). Each code is assigned
//! here and nowhere else, one constructor per cause, so two call sites refusing for the
//! same reason cannot give two codes. Every constructor keeps the caller's message as
//! is, so `to_string()` is the text the refusal had before it carried a code.

use crate::diagnostics::{Diagnostic, EngineError};

fn coded(code: &str, block: Option<&str>, message: String) -> EngineError {
    let d = Diagnostic::error(code, message);
    EngineError::from_diagnostic(match block {
        Some(b) => d.with_block(b),
        None => d,
    })
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
/// it cannot integrate exactly. `block` is the block at fault, when one is.
pub(crate) fn adaptive_unsupported(block: Option<&str>, message: impl Into<String>) -> EngineError {
    EngineError::from_diagnostic(adaptive_unsupported_diag(block, message))
}

/// [`adaptive_unsupported`] as a [`Diagnostic`], for `compile_adaptive`'s ODE gate.
pub(crate) fn adaptive_unsupported_diag(
    block: Option<&str>,
    message: impl Into<String>,
) -> Diagnostic {
    let d = Diagnostic::error("E_ADAPTIVE_UNSUPPORTED", message);
    match block {
        Some(b) => d.with_block(b),
        None => d,
    }
}

/// `E_ADAPTIVE_OPTIONS`: `AdaptiveSimulateOptions` that give no usable schedule — an
/// empty or not finite and strictly increasing `decision_times` — or that set
/// `decision_times` / `monitors` where the `[adaptive_dosing]` spec already owns them.
pub(crate) fn adaptive_options(message: impl Into<String>) -> EngineError {
    coded("E_ADAPTIVE_OPTIONS", None, message.into())
}

/// `E_ADAPTIVE_SPEC`: an `[adaptive_dosing]` spec that does not compile against the
/// model — an invariant the block parser checks, an `assay_cmt` out of range, or an
/// `observe` expression that does not compile.
pub(crate) fn adaptive_spec(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error("E_ADAPTIVE_SPEC", message).with_block("adaptive_dosing")
}
