//! Analytic sensitivities: the generic number types the closed forms, event walks and ODE
//! solvers are written over (`*_g<T: PkNum>`), and the adjoint that differentiates them.
//!
//! The PK solutions are written once, generic in [`num::PkNum`]; `T = f64` is the prediction.
//! The adjoint (`ode_provider::adjoint`) records them into compiled per-subject plans
//! (`parser::kernel_jit`) and gives FOCE, FOCEI, Laplace and AGQ their exact outer gradient,
//! inner gradient, `H` and R-matrix from them. [`dual2::Dual2`] is the independent
//! second-order forward dual the tests check those plans against.

pub mod dual2;
pub(crate) mod focei_kernel;
pub(crate) mod jet;
pub(crate) mod linsolve;
pub mod num;
pub mod ode_provider;
pub mod one_cpt;
pub mod propagate;
pub mod provider;
pub(crate) mod taylor;
pub mod three_cpt;
pub mod two_cpt;

#[cfg(test)]
#[path = "closed_form_reference_tests.rs"]
mod closed_form_reference_tests;
#[cfg(test)]
#[path = "nested_levels_tests.rs"]
mod nested_levels_tests;
