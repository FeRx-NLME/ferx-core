//! Simulation back-ends beyond the basic forward pass.
//!
//! Houses the state-reactive ("adaptive" / feedback) dosing machinery for issue
//! #391. Step S1.1 lands the [`adaptive`] vocabulary; the reactive driver and the
//! public `simulate_adaptive()` entry point follow in later steps.

pub mod adaptive;
