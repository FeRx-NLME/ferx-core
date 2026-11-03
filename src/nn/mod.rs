//! Neural-network-based covariate models (DCM) and dynamics (low-dim NODE).
//!
//! The network evaluation is gated behind the `nn` cargo feature. It provides:
//!
//! - [`MlpMapper`] — a pure-math multilayer perceptron with `forward` and
//!   analytical `jacobian` (full output-vs-weights Jacobian). Built on
//!   `nalgebra`; no new runtime dependencies.
//! - [`CovariateMapper`] — a trait the rest of the engine talks to. The
//!   higher-level [`NamedMlpMapper`] adapts `MlpMapper` to the engine's
//!   `(HashMap<String, f64>, &[f64]) -> PkParams` interface used by
//!   `pk_param_fn` on `CompiledModel`.
//!
//! See `plans/dcm-and-low-dim-node.md` (Phase A M1) for the design rationale
//! and the role of this module in the larger plan. The parser hookup
//! (`[covariate_nn NAME]` block → auto-generated weight thetas → NN-aware
//! `pk_param_fn` closure) lands in a follow-up PR; this module is callable
//! today via direct construction in Rust, which is what the integration
//! tests exercise.
//!
//! ## Differentiability
//!
//! All activation functions and their derivatives use explicit `if`/`else`
//! comparisons instead of `f64::max`/`f64::min`, so a future generic
//! `PkNum`/`Dual2` instantiation (mixed-effects DCM via FOCEI, Phase A M2)
//! differentiates cleanly without branch-on-`max` ambiguity.
//!
//! Two submodules are compiled in every build, because neither evaluates a
//! network: [`Activation`] (shared with the expression builtins `tanh` /
//! `softplus`) and [`export`], which writes a fitted network out as plain
//! `[individual_parameters]` equations (#1331) — a text transform that has to
//! run in the default build, where `[covariate_nn]` itself does not parse.

pub(crate) mod activation;
pub mod export;

pub use activation::Activation;

#[cfg(feature = "nn")]
mod mlp;
#[cfg(feature = "nn")]
pub use mlp::*;
