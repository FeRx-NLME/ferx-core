//! Covariate-NN (DCM) regularizer as seen by the outer optimizers.
//!
//! With the `nn` feature this is [`crate::nn::NnRegularizer`]. Without it there
//! are no `[covariate_nn]` blocks to regularize, so a unit-struct stub with the
//! same signatures stands in: `build` returns it, `is_active` is `false`,
//! `penalty_value` is `0.0`, and the gradient / Hessian adders touch nothing.
//! The optimizers therefore carry **no feature gate at their call sites** —
//! the same arrangement `NnGradPlan::build` uses in
//! [`super::nn_theta_gradient`] — instead of a `#[cfg(feature = "nn")]` per
//! objective, gradient, and Hessian site plus the `#[allow(unused_mut)]` shims
//! those need to stay warning-free.

#[cfg(feature = "nn")]
pub(crate) use crate::nn::NnRegularizer;

#[cfg(not(feature = "nn"))]
#[derive(Debug, Clone)]
pub(crate) struct NnRegularizer;

#[cfg(not(feature = "nn"))]
impl NnRegularizer {
    pub(crate) fn build(
        _model: &crate::types::CompiledModel,
        _population: &crate::types::Population,
        _options: &crate::types::FitOptions,
    ) -> Self {
        Self
    }

    pub(crate) fn is_active(&self) -> bool {
        false
    }

    pub(crate) fn penalty_value(&self, _theta: &[f64]) -> f64 {
        0.0
    }

    pub(crate) fn add_packed_gradient(&self, _theta: &[f64], _grad: &mut [f64]) {}

    pub(crate) fn penalty_and_gradient(&self, _theta: &[f64], _grad: &mut [f64]) -> f64 {
        0.0
    }

    pub(crate) fn add_packed_hessian(
        &self,
        _theta: &[f64],
        _add: &mut dyn FnMut(usize, usize, f64),
    ) {
    }
}

#[cfg(all(test, not(feature = "nn")))]
mod tests {
    use super::NnRegularizer;

    /// The stub must be a strict no-op with the same surface the real type has,
    /// so an optimizer compiled without `nn` behaves exactly as if λ = 0.
    #[test]
    fn stub_is_a_strict_noop() {
        let reg = NnRegularizer;
        assert!(!reg.is_active());
        let theta = [1.0, 2.0, 3.0];
        assert_eq!(reg.penalty_value(&theta), 0.0);
        let mut grad = [0.5; 3];
        reg.add_packed_gradient(&theta, &mut grad);
        assert_eq!(reg.penalty_and_gradient(&theta, &mut grad), 0.0);
        assert_eq!(grad, [0.5; 3]);
        let mut touched = false;
        reg.add_packed_hessian(&theta, &mut |_, _, _| touched = true);
        assert!(!touched);
    }
}
