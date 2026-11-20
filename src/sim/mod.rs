//! Simulation back-ends beyond the basic forward pass.
//!
//! Houses the state-reactive ("adaptive" / feedback) dosing machinery for issue
//! #391. The [`adaptive`] module holds the controller vocabulary; [`adaptive_control`]
//! compiles a declarative `[adaptive_dosing]` block into a controller; the reactive
//! driver lives in [`crate::ode::predictions`] and the public
//! [`crate::api::simulate_adaptive`] entry point wraps it.

pub mod adaptive {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::sim::adaptive::*;
    pub use ferx_model::sim::adaptive::{
        AdaptiveAction, AdaptiveDosingSpec, AdaptiveRoute, AdaptiveRule, AdaptiveRun,
        AdaptiveSubjectMetrics, Comparison, ControllerCtx, DecisionCovariates, DecisionLogEntry,
        DecisionOutcome, DoseAction, DoseKnowledge, DoseLedgerEntry, DoseStep, DoseValue,
        IntervalStep, MonitorSpec, ObserveMode, ObservedSignal,
    };
}
pub mod adaptive_control;
