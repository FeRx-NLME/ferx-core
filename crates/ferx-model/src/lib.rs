//! The model and numerics layer of ferx-core (#3041): the data types, the model parser
//! and its JIT, the closed-form and ODE engines, the sensitivities, the likelihood
//! kernels, and the data reader / output writer. `ferx-core` re-exports it; nothing here
//! depends on the estimators or the public API layer.
#![allow(clippy::too_many_arguments)]

pub mod cancel;
#[cfg(feature = "survival")]
pub mod categorical;
pub mod cold_sort;
pub mod construct;
pub mod covariate_stats;
pub mod diagnostics;
pub mod dosing;
pub mod edit;
pub mod entry_refusals;
pub mod environment;
pub mod estimation;
pub mod io;
pub mod levels;
#[cfg(feature = "markov")]
pub mod markov;
pub mod model_checks;
pub mod nn;
pub mod occasions;
pub mod ode;
pub mod par;
pub mod parser;
pub mod pk;
pub mod rng;
pub mod scratch;
pub mod sens;
pub mod serde_nalgebra;
pub mod sim;
pub mod sim_result;
pub mod stats;
#[cfg(feature = "survival")]
pub mod survival;
#[cfg(any(test, feature = "test-support"))]
pub mod test_models;
pub mod types;
