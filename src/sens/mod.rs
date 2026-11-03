//! Analytic sensitivities: the generic number types the closed forms, event walks and ODE
//! solvers are written over (`*_g<T: PkNum>`), and the adjoint that differentiates them.
//!
//! The PK solutions are written once, generic in [`num::PkNum`]; `T = f64` is the prediction.
//! The adjoint (`ode_provider::adjoint`) records them over its own dual types — `vtape::DualV`
//! (θ on a reverse tape, η-jets forward), `vfree::DualF` (tape-free η-jets for the inner loop),
//! `rtape::{DualT, DualH}` (the covariance step's nested duals) — and gives FOCE, FOCEI,
//! Laplace and AGQ their exact outer gradient, inner gradient, `H` and R-matrix from them.

pub mod dual1;
pub mod dual2;
pub mod dual_mixed;
#[cfg(feature = "survival")]
pub(crate) mod eta_jets;
pub(crate) mod focei_kernel;
pub(crate) mod linsolve;
pub mod num;
pub mod ode_provider;
pub mod one_cpt;
pub mod propagate;
pub mod provider;
pub(crate) mod rtape;
pub(crate) mod taylor;
pub mod three_cpt;
pub mod two_cpt;
pub(crate) mod vfree;
pub(crate) mod vtape;

#[cfg(test)]
#[path = "nested_levels_tests.rs"]
mod nested_levels_tests;
