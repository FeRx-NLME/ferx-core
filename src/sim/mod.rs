//! Simulation back-ends beyond the basic forward pass.
//!
//! Houses the state-reactive ("adaptive" / feedback) dosing machinery for issue
//! #391. The [`adaptive`] module holds the controller vocabulary; the reactive
//! driver lives in [`crate::ode::predictions`] and the public
//! [`crate::api::simulate_adaptive`] entry point wraps it.

pub mod adaptive;
