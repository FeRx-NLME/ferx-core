//! Neural-network-based covariate models (DCM) and dynamics (low-dim NODE).
//!
//! This module is gated behind the `nn` cargo feature. It provides:
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

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

use crate::types::{CompiledModel, FitOptions, PkParams, Population};

/// Uniform grid step (in z-scored input units) for the smoothness/curvature
/// penalty's marginal partial-dependence sweep. A quarter of a standard
/// deviation resolves the wiggles the penalty targets without over-sampling.
pub(crate) const NN_SMOOTH_GRID_STEP_Z: f64 = 0.25;

/// Cap on the number of grid nodes swept per NN input, bounding the curvature
/// penalty's cost for covariates with a very wide observed range.
pub(crate) const NN_SMOOTH_GRID_MAX_NODES: usize = 65;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum NnError {
    #[error("layers must have at least an input and an output dimension; got {0:?}")]
    InvalidLayers(Vec<usize>),
    #[error("layer dimension must be > 0 at index {index}; got {value}")]
    ZeroLayerDimension { index: usize, value: usize },
    #[error("expected {expected} weights, got {actual}")]
    WeightCountMismatch { expected: usize, actual: usize },
    #[error("expected {expected} inputs, got {actual}")]
    InputCountMismatch { expected: usize, actual: usize },
    #[error("covariate '{0}' missing from input map")]
    MissingCovariate(String),
    #[error("output name '{0}' is not a recognised PK parameter (see PkParams::name_to_index)")]
    UnknownPkOutput(String),
    #[error("duplicate output name '{0}'")]
    DuplicateOutput(String),
    #[error("`{field}` must have one entry per input: expected {expected}, got {actual}")]
    NormalizationLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("`scale` for input '{input}' must be finite and non-zero; got {value}")]
    InvalidScale { input: String, value: f64 },
    #[error("`center` for input '{input}' must be finite; got {value}")]
    InvalidCenter { input: String, value: f64 },
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// Element-wise activation functions for hidden / output layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// f(x) = x. The output layer of a DCM model usually uses this and
    /// wraps a positivity head (`Softplus` / `Exp`) separately.
    Identity,
    /// f(x) = x if x > 0 else 0. Implemented with `if`/`else` (not
    /// `f64::max`) for AD safety; see module docs.
    Relu,
    /// f(x) = ln(1 + exp(x)). Smooth positive-output head. Numerically
    /// stable for large `x` (falls back to `x` past a threshold).
    Softplus,
    /// f(x) = tanh(x). Bounded (-1, 1); useful for hidden layers.
    Tanh,
    /// f(x) = 1 / (1 + exp(-x)). Bounded (0, 1); useful for gating / bounded
    /// outputs (e.g. `F` bioavailability).
    Sigmoid,
    /// f(x) = exp(x). Strictly positive head, but unbounded; prefer
    /// `Softplus` for stability unless you need the multiplicative behavior.
    Exp,
}

impl Activation {
    /// Lowercase identifier as used in the `.ferx` DSL
    /// (`activation = tanh`). Symmetric round-trip with the parser.
    pub fn as_str(self) -> &'static str {
        match self {
            Activation::Identity => "identity",
            Activation::Relu => "relu",
            Activation::Softplus => "softplus",
            Activation::Tanh => "tanh",
            Activation::Sigmoid => "sigmoid",
            Activation::Exp => "exp",
        }
    }

    /// Apply elementwise. Uses `if`/`else` for AD safety.
    #[inline]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::Relu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::Softplus => {
                // ln(1+exp(x)) ≈ x for large x; use threshold to avoid overflow.
                if x > 20.0 {
                    x
                } else if x < -20.0 {
                    x.exp()
                } else {
                    (1.0 + x.exp()).ln()
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => sigmoid(x),
            Activation::Exp => x.exp(),
        }
    }

    /// Solve `f(z) = y` for `z`, or `None` when `y` lies outside the
    /// activation's range.
    ///
    /// Used to turn a declared output value into the output-layer bias that
    /// produces it (`[covariate_nn] init`). The inverse is exact for every
    /// activation here, so a declared `init` is realised exactly rather than
    /// approached.
    pub fn invert(self, y: f64) -> Option<f64> {
        if !y.is_finite() {
            return None;
        }
        match self {
            Activation::Identity => Some(y),
            // 0 has no unique preimage under ReLU (every z ≤ 0 maps to it), so only
            // the strictly positive branch is invertible.
            Activation::Relu => (y > 0.0).then_some(y),
            Activation::Softplus => {
                if y <= 0.0 {
                    None
                } else if y > 20.0 {
                    // `apply` returns `x` unchanged past this threshold, so the exact
                    // inverse of the implemented function is the identity here too.
                    Some(y)
                } else {
                    // `exp_m1` keeps the small-`y` end accurate, where `exp(y) - 1`
                    // would lose most of its significant digits to cancellation.
                    Some(y.exp_m1().ln())
                }
            }
            Activation::Tanh => (y.abs() < 1.0).then(|| y.atanh()),
            Activation::Sigmoid => (y > 0.0 && y < 1.0).then(|| (y / (1.0 - y)).ln()),
            Activation::Exp => (y > 0.0).then(|| y.ln()),
        }
    }

    /// Human-readable description of the activation's range, for error messages
    /// when [`invert`](Self::invert) declines.
    pub fn range_description(self) -> &'static str {
        match self {
            Activation::Identity => "any finite value",
            Activation::Relu => "a value > 0",
            Activation::Softplus => "a value > 0",
            Activation::Tanh => "a value strictly between -1 and 1",
            Activation::Sigmoid => "a value strictly between 0 and 1",
            Activation::Exp => "a value > 0",
        }
    }

    /// Derivative f'(x). For ReLU at x=0 we return 0 (left-derivative
    /// convention, also what FOCEI implementations typically use).
    #[inline]
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Activation::Identity => 1.0,
            Activation::Relu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Softplus => sigmoid(x),
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            Activation::Exp => x.exp(),
        }
    }
}

/// Numerically stable sigmoid. Uses `if`/`else` for AD safety.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// MlpMapper — the math
// ---------------------------------------------------------------------------

/// A fully-connected feedforward MLP with a single hidden-layer activation
/// and an optional output-layer activation.
///
/// Layout of the flat weight vector:
///
/// ```text
/// [W_1.row_major, b_1, W_2.row_major, b_2, ..., W_L.row_major, b_L]
/// ```
///
/// where layer `l` has `W_l` of shape `(layers[l] × layers[l-1])` and bias
/// `b_l` of length `layers[l]`. Row-major storage means `W_l[i, j]` lives at
/// offset `i * layers[l-1] + j` within the block.
#[derive(Debug, Clone)]
pub struct MlpMapper {
    layers: Vec<usize>,
    hidden_activation: Activation,
    output_activation: Activation,
    /// Cached total parameter count.
    n_params: usize,
    /// Cached weight-block / bias-block offsets per layer. `offsets[l]` is
    /// the start of layer `l+1`'s W block in the flat weight vector
    /// (`offsets[0] = 0`, `offsets[L] = n_params`).
    offsets: Vec<usize>,
}

impl MlpMapper {
    /// Construct an MLP. `layers` must be `[n_input, n_hidden_1, ..., n_output]`
    /// (length ≥ 2).
    pub fn new(
        layers: Vec<usize>,
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Result<Self, NnError> {
        if layers.len() < 2 {
            return Err(NnError::InvalidLayers(layers));
        }
        for (i, &v) in layers.iter().enumerate() {
            if v == 0 {
                return Err(NnError::ZeroLayerDimension { index: i, value: v });
            }
        }

        let mut offsets = Vec::with_capacity(layers.len());
        offsets.push(0);
        let mut acc = 0usize;
        for l in 1..layers.len() {
            acc += layers[l] * layers[l - 1] + layers[l];
            offsets.push(acc);
        }
        let n_params = acc;

        Ok(Self {
            layers,
            hidden_activation,
            output_activation,
            n_params,
            offsets,
        })
    }

    /// Total number of weights + biases.
    pub fn n_weights(&self) -> usize {
        self.n_params
    }

    /// Full layer shape: `[n_input, n_hidden_1, ..., n_output]`.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layers
    }

    /// Hidden-layer activation (applied between every adjacent layer
    /// except the output).
    pub fn hidden_activation(&self) -> Activation {
        self.hidden_activation
    }

    /// Output-layer activation (applied at the output).
    pub fn output_activation(&self) -> Activation {
        self.output_activation
    }

    /// Number of input features.
    pub fn n_inputs(&self) -> usize {
        self.layers[0]
    }

    /// Number of output features.
    pub fn n_outputs(&self) -> usize {
        *self
            .layers
            .last()
            .expect("layers non-empty by construction")
    }

    /// Index, within the flat weight vector, of the output layer's bias for
    /// output `k`.
    ///
    /// This bias is special: it enters the network only through the single
    /// pre-activation `z_k`, with `∂z_k/∂b_k = 1` and `∂z_j/∂b_k = 0` for
    /// `j ≠ k`. That makes it the one coordinate whose finite difference
    /// recovers `∂·/∂z_k` for free — see
    /// [`jacobian_preactivation`](Self::jacobian_preactivation).
    ///
    /// Panics if `k >= n_outputs()`.
    pub fn output_bias_index(&self, k: usize) -> usize {
        let n_out = self.n_outputs();
        assert!(
            k < n_out,
            "output index {k} out of range (n_outputs {n_out})"
        );
        let l = self.layers.len() - 1;
        let n_lm1 = self.layers[l - 1];
        self.offsets[l - 1] + n_out * n_lm1 + k
    }

    /// Index range of the output layer's **weight** block (`W_L`, excluding its
    /// biases) within the flat weight vector.
    ///
    /// Zeroing this block makes the output layer's pre-activation equal its bias
    /// for every input, which is how a declared `[covariate_nn] init` is made
    /// exact rather than approximate — see the parser's `init` handling.
    pub fn output_weight_range(&self) -> std::ops::Range<usize> {
        let l = self.layers.len() - 1;
        let start = self.offsets[l - 1];
        start..self.output_bias_index(0)
    }

    /// Forward pass.
    ///
    /// Errors:
    /// - [`NnError::InputCountMismatch`] if `x.len() != n_inputs()`.
    /// - [`NnError::WeightCountMismatch`] if `weights.len() != n_weights()`.
    pub fn forward(&self, x: &[f64], weights: &[f64]) -> Result<Vec<f64>, NnError> {
        self.check_shapes(x, weights)?;
        let (_pre, post) = self.forward_cache(x, weights);
        let last = post
            .into_iter()
            .last()
            .expect("at least one layer activation");
        Ok(last.iter().copied().collect())
    }

    /// Full Jacobian dy/dθ, shape `(n_outputs × n_weights)`, as a dense
    /// matrix. Computed via reverse-mode backpropagation, one output row at
    /// a time.
    ///
    /// For a 5-output / 200-weight MLP this matrix is ~8 kB and computed in
    /// 5 backward passes; the cost is `O(n_outputs · n_weights)`, fine for
    /// paper-scale networks. For larger architectures use vector-Jacobian
    /// products instead (deferred to Phase A M3).
    pub fn jacobian(&self, x: &[f64], weights: &[f64]) -> Result<DMatrix<f64>, NnError> {
        self.jacobian_impl(x, weights, false)
    }

    /// Jacobian of the output layer's **pre-activations** `z_L` vs the flat
    /// weight vector, shape `(n_outputs × n_weights)`. Identical to
    /// [`jacobian`](Self::jacobian) except the backward sweep is seeded at
    /// `z_L` rather than at `a_L = f(z_L)`, so the output activation's
    /// derivative is not applied. When `output_activation` is
    /// [`Activation::Identity`] the two agree exactly.
    ///
    /// # Why this variant exists
    ///
    /// It is the exact chain-rule bridge used by
    /// `crate::estimation::fixed_eta_gradient` to get `∂NLL/∂w` for every NN
    /// weight from just `n_outputs` finite-difference evaluations. Because
    /// `z_k = (W_L a_{L-1} + b_L)_k` depends on the output-layer bias `b_k`
    /// with `∂z_k/∂b_k = 1` exactly (and `∂z_j/∂b_k = 0` for `j ≠ k`), a
    /// finite difference of the objective in `b_k` *is* `∂NLL/∂z_k`. Then
    ///
    /// ```text
    /// ∂NLL/∂w_j = Σ_k (∂NLL/∂z_k) · (∂z_k/∂w_j)
    /// ```
    ///
    /// with the second factor read straight off this matrix.
    ///
    /// Seeding at `z_L` instead of `a_L` is what keeps that identity
    /// unconditional. The post-activation form would need a division by
    /// `f'(z_k)`, which is unusable exactly where it matters — a saturated
    /// `Softplus`/`Sigmoid` head drives `f'(z_k) → 0`, and the recovered
    /// derivative would be `0/0` in floating point. Here the factor never
    /// appears: it cancels analytically between the seed and the chain.
    pub fn jacobian_preactivation(
        &self,
        x: &[f64],
        weights: &[f64],
    ) -> Result<DMatrix<f64>, NnError> {
        self.jacobian_impl(x, weights, true)
    }

    /// Shared backprop for [`jacobian`](Self::jacobian) and
    /// [`jacobian_preactivation`](Self::jacobian_preactivation). `preactivation`
    /// skips the output layer's activation derivative in the seed.
    fn jacobian_impl(
        &self,
        x: &[f64],
        weights: &[f64],
        preactivation: bool,
    ) -> Result<DMatrix<f64>, NnError> {
        self.jacobian_with_output(x, weights, preactivation)
            .map(|(_, jac)| jac)
    }

    /// [`jacobian_impl`](Self::jacobian_impl) plus the network output `a_L` at
    /// the same point. The backward sweep already runs the forward pass, so a
    /// caller that needs both (the curvature penalty evaluates `f` and `∂f/∂w`
    /// at every grid node) gets the output for free instead of paying a second
    /// forward pass.
    fn jacobian_with_output(
        &self,
        x: &[f64],
        weights: &[f64],
        preactivation: bool,
    ) -> Result<(Vec<f64>, DMatrix<f64>), NnError> {
        self.check_shapes(x, weights)?;
        let (pre, post) = self.forward_cache(x, weights);
        let output: Vec<f64> = post
            .last()
            .expect("at least one layer activation")
            .iter()
            .copied()
            .collect();

        let n_out = self.n_outputs();
        let mut jac = DMatrix::<f64>::zeros(n_out, self.n_params);

        // Backprop one output at a time. For each output dimension k:
        //   - seed the adjoint da_L = e_k
        //   - propagate backward through the activation derivative at each
        //     layer, accumulating grad_W_l and grad_b_l into `jac` row k.
        for k in 0..n_out {
            let mut adjoint = DVector::<f64>::zeros(n_out);
            adjoint[k] = 1.0;

            // Walk layers L, L-1, ..., 1.
            for l in (1..self.layers.len()).rev() {
                let is_output_layer = l == self.layers.len() - 1;
                let activation = if is_output_layer {
                    self.output_activation
                } else {
                    self.hidden_activation
                };

                // dz_l = da_l ⊙ activation'(z_l). Under `preactivation` the
                // output layer is seeded directly at z_L, so its activation
                // derivative is skipped — every deeper layer is unaffected.
                let z_l = &pre[l - 1]; // pre-activation of layer l (indexed from 1)
                let dz_l = if preactivation && is_output_layer {
                    adjoint.clone()
                } else {
                    let d: DVector<f64> = DVector::from_iterator(
                        self.layers[l],
                        z_l.iter().map(|&z| activation.derivative(z)),
                    );
                    adjoint.component_mul(&d)
                };

                // grad_W_l[i,j] = dz_l[i] * a_{l-1}[j].
                // We unflatten into the row-major W block within `jac` row k.
                let a_prev = if l == 1 {
                    DVector::<f64>::from_column_slice(x)
                } else {
                    post[l - 2].clone()
                };

                let w_start = self.offsets[l - 1];
                let n_l = self.layers[l];
                let n_lm1 = self.layers[l - 1];
                for i in 0..n_l {
                    let dz_i = dz_l[i];
                    let row_offset = w_start + i * n_lm1;
                    for j in 0..n_lm1 {
                        jac[(k, row_offset + j)] = dz_i * a_prev[j];
                    }
                    // bias gradient
                    jac[(k, w_start + n_l * n_lm1 + i)] = dz_i;
                }

                // Propagate to layer l-1: da_{l-1} = W_l^T · dz_l.
                if l > 1 {
                    let w_l = self.weight_matrix(weights, l);
                    adjoint = w_l.transpose() * dz_l;
                }
            }
        }

        Ok((output, jac))
    }

    /// Local flat-vector indices of the **weight-matrix** parameters, excluding
    /// the bias entries. Within each layer `l`'s block
    /// (`offsets[l-1]..offsets[l]`) the first `layers[l]·layers[l-1]` entries are
    /// weights and the trailing `layers[l]` are biases (matches the row-major
    /// `[W_l, b_l]` layout and the parser's `W_…`/`B_…` theta naming). Used by
    /// the L2 (weight-decay) penalty, which shrinks weights but leaves biases
    /// free.
    #[cfg(test)]
    pub(crate) fn weight_param_indices(&self) -> Vec<usize> {
        let mut idx = Vec::new();
        for l in 1..self.layers.len() {
            let start = self.offsets[l - 1];
            let n_w = self.layers[l] * self.layers[l - 1];
            idx.extend(start..start + n_w);
        }
        idx
    }

    /// L2 (weight-decay) penalty value `lambda · Σ wᵢ²` over the **weight**
    /// blocks only (biases excluded). `lambda == 0.0` returns `0.0` (no-op).
    /// Test-facing wrapper; production goes through [`NnRegularizer`].
    #[cfg(test)]
    pub(crate) fn l2_weight_penalty_value(&self, weights: &[f64], lambda: f64) -> f64 {
        self.l2_weight_penalty_accum(weights, lambda, &mut PenaltySink::none())
    }

    /// L2 penalty value and its gradient w.r.t. the flat weight vector.
    /// `grad[i] = 2·lambda·wᵢ` on weight entries, `0` on bias entries.
    /// `lambda == 0.0` returns `(0.0, zeros)` (strict no-op).
    #[cfg(test)]
    pub(crate) fn l2_weight_penalty(&self, weights: &[f64], lambda: f64) -> (f64, Vec<f64>) {
        let mut grad = vec![0.0f64; self.n_params];
        let v = self.l2_weight_penalty_accum(weights, lambda, &mut PenaltySink::grad(&mut grad));
        (v, grad)
    }

    /// One-pass L2 kernel behind [`l2_weight_penalty_value`](Self::l2_weight_penalty_value)
    /// / [`l2_weight_penalty`](Self::l2_weight_penalty): returns the value and
    /// accumulates whatever derivative pieces `sink` asks for. The gradient is
    /// `2·lambda·wᵢ` on weight entries; the Hessian is exact, `2·lambda` on the
    /// weight diagonal and nothing on biases. `lambda == 0.0` touches nothing.
    pub(crate) fn l2_weight_penalty_accum(
        &self,
        weights: &[f64],
        lambda: f64,
        sink: &mut PenaltySink<'_>,
    ) -> f64 {
        if lambda == 0.0 {
            return 0.0;
        }
        let mut s = 0.0;
        for l in 1..self.layers.len() {
            let start = self.offsets[l - 1];
            let n_w = self.layers[l] * self.layers[l - 1];
            for i in start..start + n_w {
                s += weights[i] * weights[i];
                if let Some(g) = sink.grad.as_deref_mut() {
                    g[i] += 2.0 * lambda * weights[i];
                }
                if let Some(h) = sink.hess.as_deref_mut() {
                    h(i, i, 2.0 * lambda);
                }
            }
        }
        lambda * s
    }

    /// Smoothness (curvature) penalty value `lambda · Σ_grid Σ_output C²`, where
    /// `C = f(x+h) − 2·f(x) + f(x−h)` is the finite-difference 2nd derivative of
    /// each output along a marginal partial-dependence curve (see
    /// [`CurvatureGrid`]). Forward passes only; `lambda == 0.0` or an empty grid
    /// returns `0.0` (no-op).
    #[cfg(test)]
    pub(crate) fn curvature_penalty_value(
        &self,
        weights: &[f64],
        grid: &CurvatureGrid,
        lambda: f64,
    ) -> f64 {
        self.curvature_penalty_accum(weights, grid, lambda, &mut PenaltySink::none())
    }

    /// Smoothness (curvature) penalty value and its gradient w.r.t. the flat
    /// weight vector. `lambda == 0.0` or an empty grid returns `(0.0, zeros)`
    /// (strict no-op). See [`curvature_penalty_accum`](Self::curvature_penalty_accum).
    #[cfg(test)]
    pub(crate) fn curvature_penalty(
        &self,
        weights: &[f64],
        grid: &CurvatureGrid,
        lambda: f64,
    ) -> (f64, Vec<f64>) {
        let mut grad = vec![0.0f64; self.n_params];
        let v =
            self.curvature_penalty_accum(weights, grid, lambda, &mut PenaltySink::grad(&mut grad));
        (v, grad)
    }

    /// One-pass curvature kernel: value, plus whatever `sink` asks for.
    ///
    /// Each grid axis is a uniform sweep of nodes `x_0 … x_{n−1}` in the
    /// network's own input space (see [`CurvatureGrid`]), and every interior
    /// node `k` contributes `C_k = f(x_{k+1}) − 2·f(x_k) + f(x_{k−1})` per
    /// output. The network is evaluated **once per node** — adjacent stencils
    /// share nodes, so this is `n` evaluations per axis rather than `3(n−2)` —
    /// and when a derivative is wanted the output is read off the same backward
    /// sweep that produces the Jacobian ([`jacobian_with_output`](Self::jacobian_with_output)),
    /// so no separate forward pass runs.
    ///
    /// Gradient: `∂C/∂w = J(x_{k+1}) − 2·J(x_k) + J(x_{k−1})`, so
    /// `∂pen/∂w = Σ 2·lambda·C·∂C/∂w`. Touches bias entries too (biases shift
    /// the curve), unlike the L2 term. Hessian: the Gauss–Newton form
    /// `2·lambda·Σ (∂C/∂w)(∂C/∂w)ᵀ`, which drops the `C·∂²C/∂w²` term — positive
    /// semi-definite by construction and exact at `C = 0`, i.e. at the flat map
    /// the penalty pulls toward, which is the regime a trust-region / BHHH
    /// consumer needs it in. `lambda == 0.0` or an empty grid touches nothing.
    pub(crate) fn curvature_penalty_accum(
        &self,
        weights: &[f64],
        grid: &CurvatureGrid,
        lambda: f64,
        sink: &mut PenaltySink<'_>,
    ) -> f64 {
        if lambda == 0.0 || grid.is_empty() {
            return 0.0;
        }
        let want_deriv = sink.grad.is_some() || sink.hess.is_some();
        let n_p = self.n_params;
        let mut s = 0.0;
        let mut dc = vec![0.0f64; n_p];
        for nodes in grid.axes() {
            // Evaluate every node once.
            let mut ys: Vec<Vec<f64>> = Vec::with_capacity(nodes.len());
            let mut js: Vec<DMatrix<f64>> = Vec::new();
            for x in nodes {
                if want_deriv {
                    let (y, j) = self
                        .jacobian_with_output(x, weights, false)
                        .expect("grid input shape ok");
                    ys.push(y);
                    js.push(j);
                } else {
                    ys.push(self.forward(x, weights).expect("grid input shape ok"));
                }
            }
            let n_out = ys[0].len();
            for k in 1..nodes.len() - 1 {
                for o in 0..n_out {
                    let c = ys[k + 1][o] - 2.0 * ys[k][o] + ys[k - 1][o];
                    s += c * c;
                    if !want_deriv {
                        continue;
                    }
                    let (jm, jc, jp) = (&js[k - 1], &js[k], &js[k + 1]);
                    for w in 0..n_p {
                        dc[w] = jp[(o, w)] - 2.0 * jc[(o, w)] + jm[(o, w)];
                    }
                    if let Some(g) = sink.grad.as_deref_mut() {
                        let two_lc = 2.0 * lambda * c;
                        for w in 0..n_p {
                            g[w] += two_lc * dc[w];
                        }
                    }
                    if let Some(h) = sink.hess.as_deref_mut() {
                        let two_l = 2.0 * lambda;
                        for i in 0..n_p {
                            if dc[i] == 0.0 {
                                continue;
                            }
                            for j in 0..n_p {
                                h(i, j, two_l * dc[i] * dc[j]);
                            }
                        }
                    }
                }
            }
        }
        lambda * s
    }

    /// Build an `(n_l × n_{l-1})` `DMatrix` from the layer-`l` weight block
    /// (1-indexed, `1..=L`). The flat weight vector is row-major while
    /// nalgebra's `DMatrix` is column-major, so this is a copy, not a
    /// zero-cost view. For the paper-scale networks this module targets
    /// (≤300 weights for DCM, ≤62 for low-dim NODE) the per-call alloc is
    /// negligible; a zero-copy variant via column-major storage is tracked
    /// against Phase A M3 in `plans/dcm-and-low-dim-node.md`.
    fn weight_matrix(&self, weights: &[f64], l: usize) -> DMatrix<f64> {
        let n_l = self.layers[l];
        let n_lm1 = self.layers[l - 1];
        let start = self.offsets[l - 1];
        DMatrix::<f64>::from_row_slice(n_l, n_lm1, &weights[start..start + n_l * n_lm1])
    }

    fn bias_slice<'a>(&self, weights: &'a [f64], l: usize) -> &'a [f64] {
        let n_l = self.layers[l];
        let n_lm1 = self.layers[l - 1];
        let bias_start = self.offsets[l - 1] + n_l * n_lm1;
        &weights[bias_start..bias_start + n_l]
    }

    /// Forward pass returning pre-activations and post-activations per layer.
    /// `pre[l-1]` is the pre-activation z_l (length `layers[l]`);
    /// `post[l-1]` is the post-activation a_l (length `layers[l]`).
    fn forward_cache(&self, x: &[f64], weights: &[f64]) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        let l_max = self.layers.len() - 1;
        let mut pre = Vec::with_capacity(l_max);
        let mut post = Vec::with_capacity(l_max);

        let mut a_prev = DVector::<f64>::from_column_slice(x);
        for l in 1..=l_max {
            let w = self.weight_matrix(weights, l);
            let b = DVector::<f64>::from_column_slice(self.bias_slice(weights, l));
            let z = &w * &a_prev + b;
            let activation = if l == l_max {
                self.output_activation
            } else {
                self.hidden_activation
            };
            let a: DVector<f64> =
                DVector::from_iterator(z.len(), z.iter().map(|&v| activation.apply(v)));
            pre.push(z);
            post.push(a.clone());
            a_prev = a;
        }

        (pre, post)
    }

    fn check_shapes(&self, x: &[f64], weights: &[f64]) -> Result<(), NnError> {
        if x.len() != self.n_inputs() {
            return Err(NnError::InputCountMismatch {
                expected: self.n_inputs(),
                actual: x.len(),
            });
        }
        if weights.len() != self.n_params {
            return Err(NnError::WeightCountMismatch {
                expected: self.n_params,
                actual: weights.len(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CovariateMapper trait + NamedMlpMapper
// ---------------------------------------------------------------------------

/// The interface the rest of the engine talks to when a `[covariate_nn]`
/// block is active in a model. Implementors translate
/// `(covariates, weights) → PkParams`.
///
/// The trait is intentionally narrow: any custom mapping (analytical hybrid,
/// alternative architectures via `candle`/`burn`, …) can implement this
/// without leaking the implementation details upwards.
pub trait CovariateMapper: Send + Sync {
    /// Number of weights (i.e. extra thetas to pack into the optimizer
    /// vector).
    fn n_weights(&self) -> usize;

    /// Number of PK parameters the network outputs.
    fn n_outputs(&self) -> usize;

    /// Write the NN-derived PK parameters into `out`. Caller owns `out` and
    /// is responsible for initialising it (typically via
    /// `PkParams::default()`, which sets `F=1` and leaves the rest at 0 —
    /// matching analytical-model conventions).
    fn forward(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
        out: &mut PkParams,
    ) -> Result<(), NnError>;

    /// Jacobian of the (named) PK outputs vs the flat weight vector.
    /// Row order matches the order returned by `output_names()`.
    fn jacobian(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Result<DMatrix<f64>, NnError>;

    /// PK parameter names written into `out` by `forward`, in row order
    /// matching `jacobian`. Used by the parser to wire up mu-ref detection
    /// and the eta-composition syntax (`TYPICAL_PK.CL * exp(ETA_CL)`).
    fn output_names(&self) -> &[String];
}

/// Adapt an [`MlpMapper`] to the [`CovariateMapper`] interface using named
/// input covariates and named PK output slots.
#[derive(Debug, Clone)]
pub struct NamedMlpMapper {
    mlp: MlpMapper,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Indices into `PkParams::values` for each output (one per
    /// `output_names`).
    output_pk_indices: Vec<usize>,
    /// Per-input location subtracted before the forward pass (`center` in the
    /// `[covariate_nn]` block). All zeros unless the model declares otherwise.
    input_center: Vec<f64>,
    /// Per-input scale divided out after centering (`scale` in the block). All ones
    /// unless the model declares otherwise; entries are validated non-zero and finite at
    /// parse time.
    ///
    /// # Why this is declared rather than estimated from the data
    ///
    /// A network fed raw covariates is badly conditioned: `WT ≈ 70` and `CRCL ≈ 86`
    /// saturate a `tanh` layer at Glorot initialisation, and the optimizer's only escape
    /// is to drive the first-layer weights to tiny values while later layers grow to
    /// compensate. Measured on a two-covariate DCM, unnormalised inputs pushed weights to
    /// ~1e11 and the residual error to 15× its true value; the same model with
    /// standardised inputs stayed inside `[-1.3, 1.6]`.
    ///
    /// The constants live in the model file, not in fitted state, for two reasons.
    /// `fit()` takes `&CompiledModel`, so statistics derived from the estimation data
    /// could not be written back for `predict()` to reuse — and recomputing them on new
    /// data would silently change the model between fitting and prediction, the classic
    /// train/serve skew. Declaring them also matches how population PK already writes
    /// normalisation down (`(WT/70)^0.75` names its reference weight), and keeps the
    /// model file a complete description of the transform.
    input_scale: Vec<f64>,
}

impl NamedMlpMapper {
    /// Construct a NamedMlpMapper.
    ///
    /// `output_names` must all resolve via [`PkParams::name_to_index`]
    /// (case-insensitive — names are lower-cased internally to match the
    /// analytical path).
    pub fn new(
        mlp: MlpMapper,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> Result<Self, NnError> {
        if mlp.n_inputs() != input_names.len() {
            return Err(NnError::InputCountMismatch {
                expected: mlp.n_inputs(),
                actual: input_names.len(),
            });
        }
        if mlp.n_outputs() != output_names.len() {
            return Err(NnError::InputCountMismatch {
                expected: mlp.n_outputs(),
                actual: output_names.len(),
            });
        }

        let mut seen = std::collections::HashSet::new();
        let mut output_pk_indices = Vec::with_capacity(output_names.len());
        for name in &output_names {
            let lower = name.to_ascii_lowercase();
            if !seen.insert(lower.clone()) {
                return Err(NnError::DuplicateOutput(name.clone()));
            }
            let idx = PkParams::name_to_index(&lower)
                .ok_or_else(|| NnError::UnknownPkOutput(name.clone()))?;
            output_pk_indices.push(idx);
        }

        let n_in = input_names.len();
        Ok(Self {
            mlp,
            input_names,
            output_names,
            output_pk_indices,
            input_center: vec![0.0; n_in],
            input_scale: vec![1.0; n_in],
        })
    }

    /// Attach per-input normalisation: the forward pass sees `(x - center) / scale`.
    ///
    /// Lengths must match `inputs`, and every `scale` entry must be finite and non-zero.
    /// Callers that want centering only (or scaling only) pass the identity for the other.
    pub fn with_normalization(
        mut self,
        center: Vec<f64>,
        scale: Vec<f64>,
    ) -> Result<Self, NnError> {
        let n_in = self.input_names.len();
        for (label, v) in [("center", &center), ("scale", &scale)] {
            if v.len() != n_in {
                return Err(NnError::NormalizationLengthMismatch {
                    field: label,
                    expected: n_in,
                    actual: v.len(),
                });
            }
        }
        for (i, s) in scale.iter().enumerate() {
            if !s.is_finite() || *s == 0.0 {
                return Err(NnError::InvalidScale {
                    input: self.input_names[i].clone(),
                    value: *s,
                });
            }
        }
        // Mirror the `scale` loop rather than reporting a single fabricated key: the user
        // needs the offending input's real name and the value they actually wrote.
        for (i, c) in center.iter().enumerate() {
            if !c.is_finite() {
                return Err(NnError::InvalidCenter {
                    input: self.input_names[i].clone(),
                    value: *c,
                });
            }
        }
        self.input_center = center;
        self.input_scale = scale;
        Ok(self)
    }

    /// Per-input `center`, in `inputs` order. All-zero unless the model declares
    /// normalisation. Reported on `NeuralNetworkInfo` because the fitted weights
    /// are only meaningful alongside the transform they were fitted under.
    pub fn input_center(&self) -> &[f64] {
        &self.input_center
    }

    /// Per-input `scale`, in `inputs` order. All-one unless the model declares
    /// normalisation. See [`NamedMlpMapper::input_center`].
    pub fn input_scale(&self) -> &[f64] {
        &self.input_scale
    }

    /// `(x - center) / scale` for input `i`. Identity unless the model declares
    /// normalisation, and cheap enough to apply unconditionally.
    #[inline]
    fn normalize(&self, i: usize, x: f64) -> f64 {
        (x - self.input_center[i]) / self.input_scale[i]
    }

    /// Direct access to the underlying MLP (for testing or weight
    /// inspection).
    pub fn mlp(&self) -> &MlpMapper {
        &self.mlp
    }

    /// Names of the inputs in `inputs` order — i.e. the covariate keys this
    /// mapper reads from a `&HashMap<String, f64>` on every forward pass.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Forward pass returning the raw output vector in *declaration order*
    /// (the order of `output_names`), without routing through `PkParams`.
    ///
    /// `forward` writes results into PK slots via `name_to_index`, which is
    /// what the fit / predict / simulate paths ultimately want. The parser,
    /// however, needs to look up outputs by their position in the
    /// `[covariate_nn]` block's `outputs` list (so the AST can carry a tiny
    /// `output_idx` rather than a string slot name). This method is the
    /// parser-facing variant.
    ///
    /// Missing covariates are substituted with `0.0` to match the rest of the
    /// parser's expression evaluator (which uses `unwrap_or(0.0)` for missing
    /// covariate lookups). The remaining error variants — `WeightCountMismatch`
    /// / `InputCountMismatch` — only fire on genuine wiring bugs, so callers
    /// can typically `.expect(...)` the result.
    ///
    /// **The zero-fill is not the guard against a bad input name.** Silently
    /// substituting `0.0` degenerates the network to a constant — a typo'd input, or
    /// `inputs = [TIME]` (a reserved column, not a covariate), would otherwise produce a
    /// plausible-looking fit that learned nothing. What prevents that is
    /// `api::check_covariates` at fit time, which sees `[covariate_nn]` inputs only
    /// because the parser registers them in `referenced_covariates` — that registration
    /// exists for this reason as much as for time-varying covariates. Keep the two
    /// together: dropping the registration re-opens *both* failure modes silently.
    pub fn forward_raw(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Result<Vec<f64>, NnError> {
        let x = self.build_input_vec_zero_fill(covariates);
        self.mlp.forward(&x, weights)
    }

    /// Pre-activation Jacobian `∂z_L/∂weights` at this subject's covariates,
    /// built with the **same zero-fill input construction as
    /// [`forward_raw`](Self::forward_raw)**.
    ///
    /// The pairing matters: `forward_raw` is what `pk_param_fn` calls on every
    /// prediction, so a gradient assembled from the strict
    /// [`CovariateMapper::jacobian`] would be differentiating a slightly
    /// different function than the one being evaluated whenever a covariate is
    /// absent. Use this variant anywhere the Jacobian has to agree with the
    /// production forward pass.
    pub fn jacobian_preactivation_raw(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Result<DMatrix<f64>, NnError> {
        let x = self.build_input_vec_zero_fill(covariates);
        self.mlp.jacobian_preactivation(&x, weights)
    }

    /// Strict variant used by [`CovariateMapper::forward`] / `jacobian`: errors
    /// out with `MissingCovariate` if any input name is absent.
    fn build_input_vec(&self, covariates: &HashMap<String, f64>) -> Result<Vec<f64>, NnError> {
        self.input_names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                covariates
                    .get(n)
                    .copied()
                    .map(|x| self.normalize(i, x))
                    .ok_or_else(|| NnError::MissingCovariate(n.clone()))
            })
            .collect()
    }

    /// Zero-fill variant used by [`Self::forward_raw`]: substitutes `0.0` for
    /// any missing covariate, matching the parser's expression evaluator.
    fn build_input_vec_zero_fill(&self, covariates: &HashMap<String, f64>) -> Vec<f64> {
        self.input_names
            .iter()
            .enumerate()
            .map(|(i, n)| match covariates.get(n).copied() {
                Some(x) => self.normalize(i, x),
                // Absent: feed the network the *centered* origin rather than a raw 0.0,
                // so the fallback does not depend on where the user put `center`. This
                // path is guarded by `api::check_covariates` regardless (see
                // `forward_raw`).
                None => self.normalize(i, self.input_center[i]),
            })
            .collect()
    }
}

impl CovariateMapper for NamedMlpMapper {
    fn n_weights(&self) -> usize {
        self.mlp.n_weights()
    }

    fn n_outputs(&self) -> usize {
        self.mlp.n_outputs()
    }

    fn forward(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
        out: &mut PkParams,
    ) -> Result<(), NnError> {
        let x = self.build_input_vec(covariates)?;
        let y = self.mlp.forward(&x, weights)?;
        for (i, &v) in y.iter().enumerate() {
            out.values[self.output_pk_indices[i]] = v;
        }
        Ok(())
    }

    fn jacobian(
        &self,
        weights: &[f64],
        covariates: &HashMap<String, f64>,
    ) -> Result<DMatrix<f64>, NnError> {
        let x = self.build_input_vec(covariates)?;
        self.mlp.jacobian(&x, weights)
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

// ---------------------------------------------------------------------------
// CovariateNn — a parsed `[covariate_nn NAME]` block, ready to be
// consumed by the fitting pipeline.
// ---------------------------------------------------------------------------

/// One instance of a `[covariate_nn NAME]` block, stored on `CompiledModel`.
///
/// The parser builds this and:
///
/// 1. registers the mapper's `n_weights()` weights as plain thetas in the
///    optimizer parameter vector, with names of the form
///    `W_<NAME>_<l>_<i>_<j>` / `B_<NAME>_<l>_<i>` (uppercased), starting
///    at index [`weights_offset`](Self::weights_offset);
/// 2. stores the resulting handle here so the fit / predict / simulate
///    paths can slice out the relevant weights at runtime via
///    `&theta[weights_offset..weights_offset + n_weights]`.
///
/// Multiple `[covariate_nn]` blocks per model are syntactically allowed by
/// the parser (the `named` block map keys them by `NAME`), though Phase A M1
/// only exercises the single-block case end-to-end.
#[derive(Debug, Clone)]
pub struct CovariateNn {
    /// User-visible identifier from the block header (e.g. `TYPICAL_PK`).
    /// Used in `[individual_parameters]` dot-access (`TYPICAL_PK.CL`).
    pub name: String,
    /// The mapper that translates `(covariates, weights) → PkParams`.
    pub mapper: NamedMlpMapper,
    /// Index into `ModelParameters::theta` where this NN's weight block
    /// starts. The block has `mapper.n_weights()` contiguous entries.
    pub weights_offset: usize,
}

// ---------------------------------------------------------------------------
// Regularization — L2 (weight-decay) + smoothness (curvature)
// ---------------------------------------------------------------------------

/// Derivative sinks for a penalty kernel. A kernel always returns the penalty
/// value; on top of that it accumulates `+= ∂pen/∂w` into `grad` (indexed by the
/// network's local flat weight vector) and calls `hess(i, j, v)` for every
/// Hessian contribution `v` at local `(i, j)`. Either may be absent, and the
/// kernels skip the corresponding work entirely, so value-only callers pay for
/// forward passes alone.
pub(crate) struct PenaltySink<'a> {
    pub(crate) grad: Option<&'a mut [f64]>,
    pub(crate) hess: Option<&'a mut dyn FnMut(usize, usize, f64)>,
}

#[cfg(test)]
impl<'a> PenaltySink<'a> {
    /// Value only.
    pub(crate) fn none() -> Self {
        Self {
            grad: None,
            hess: None,
        }
    }

    /// Value + gradient.
    pub(crate) fn grad(grad: &'a mut [f64]) -> Self {
        Self {
            grad: Some(grad),
            hess: None,
        }
    }
}

/// A marginal partial-dependence grid for the smoothness/curvature penalty.
///
/// For each NN input in turn, the input is swept across its observed range in
/// **z-scored** space (uniform step `NN_SMOOTH_GRID_STEP_Z`, in units of that
/// input's observed sd) while every other input is held at its median. Every
/// interior node `k` of a sweep contributes one stencil
/// `C = f(x_{k+1}) − 2·f(x_k) + f(x_{k−1})`. The grid mirrors the
/// partial-dependence curves used in downstream diagnostics, so the penalty
/// smooths exactly what those plots show.
///
/// **The grid lives in the network's input space, not in raw covariate
/// space.** [`NamedMlpMapper`] feeds its MLP `(x − center) / scale` on every
/// prediction (`build_input_vec_zero_fill`), so the values handed to
/// [`build`](Self::build) are already normalised the same way
/// ([`collect_input_values`]), and the node vectors go straight into
/// [`MlpMapper::forward`] / [`MlpMapper::jacobian`] — the same function the fit
/// evaluates, at the same inputs. A grid built on raw covariates would probe the
/// network where the fit never runs it: with `center = [70, 86]`, `scale = [12,
/// 23]` the fit sees `z ∈ ≈[−2, 2]` while raw `WT ∈ [45, 90]` saturates every
/// `tanh` unit, so `C ≈ 0` and the penalty is a silent no-op on exactly the
/// normalised models it targets.
///
/// Node count per axis is capped at `NN_SMOOTH_GRID_MAX_NODES`; an axis whose
/// observed z-span would need more nodes at the default step gets a **coarser
/// step** over the full span rather than a truncated sweep, so the upper tail of
/// a wide-range covariate is never silently left unpenalised.
#[derive(Debug, Clone, Default)]
pub(crate) struct CurvatureGrid {
    /// One sweep per usable input axis: the node input vectors (length
    /// `n_inputs`, in the network's input space), uniformly spaced along that
    /// axis, at least three per sweep.
    axes: Vec<Vec<Vec<f64>>>,
}

impl CurvatureGrid {
    /// Build the grid from per-input observed values. `input_values[i]` is the
    /// vector of observed values of NN input `i` (in the mapper's input order and
    /// **in the network's input space**, i.e. already `(x − center) / scale`),
    /// and `step_z` is the uniform grid step in units of each input's observed
    /// sd.
    ///
    /// Non-finite values are ignored — the reader stores `NaN` for a subject with
    /// no finite value in a covariate column, and one such subject must not
    /// poison the whole axis' mean/sd and silently drop it. An input with zero
    /// spread (constant covariate) or too narrow a range to fit an interior
    /// stencil contributes nothing. Returns an empty grid (a no-op penalty) when
    /// no input yields a usable sweep.
    pub(crate) fn build(input_values: &[Vec<f64>], step_z: f64) -> Self {
        let n_in = input_values.len();
        let mut axes = Vec::new();
        if step_z <= 0.0 {
            return Self { axes };
        }
        // Held-input centers (median) and per-axis mean/sd/min/max.
        let mut medians = vec![0.0; n_in];
        let mut means = vec![0.0; n_in];
        let mut sds = vec![0.0; n_in];
        let mut mins = vec![0.0; n_in];
        let mut maxs = vec![0.0; n_in];
        for (i, vals) in input_values.iter().enumerate() {
            let mut sorted: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
            if sorted.is_empty() {
                continue;
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values are ordered"));
            medians[i] = median_of_sorted(&sorted);
            let n = sorted.len() as f64;
            let mean = sorted.iter().sum::<f64>() / n;
            let var = sorted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            means[i] = mean;
            sds[i] = var.sqrt();
            mins[i] = *sorted.first().expect("non-empty");
            maxs[i] = *sorted.last().expect("non-empty");
        }
        for i in 0..n_in {
            let sd = sds[i];
            if sd <= 0.0 {
                continue; // constant covariate — no curvature axis
            }
            let z_lo = (mins[i] - means[i]) / sd;
            let z_hi = (maxs[i] - means[i]) / sd;
            let span = z_hi - z_lo;
            if span < 2.0 * step_z {
                continue; // too narrow to form an interior stencil
            }
            // `span >= 2·step_z` gives `floor(span / step_z) >= 2`, so at least
            // three nodes. Past the cap, widen the step to keep the whole span.
            let n_at_step = (span / step_z).floor() as usize + 1;
            let (n_nodes, step) = if n_at_step > NN_SMOOTH_GRID_MAX_NODES {
                (
                    NN_SMOOTH_GRID_MAX_NODES,
                    span / (NN_SMOOTH_GRID_MAX_NODES - 1) as f64,
                )
            } else {
                (n_at_step, step_z)
            };
            debug_assert!(n_nodes >= 3);
            let nodes: Vec<Vec<f64>> = (0..n_nodes)
                .map(|g| {
                    let mut x = medians.clone();
                    x[i] = (z_lo + step * g as f64) * sd + means[i];
                    x
                })
                .collect();
            axes.push(nodes);
        }
        Self { axes }
    }

    /// The per-axis node sweeps (see the struct docs).
    fn axes(&self) -> &[Vec<Vec<f64>>] {
        &self.axes
    }

    /// The `(x−h, x, x+h)` stencils the penalty evaluates, materialised.
    #[cfg(test)]
    fn triples(&self) -> Vec<[Vec<f64>; 3]> {
        self.axes
            .iter()
            .flat_map(|nodes| {
                (1..nodes.len() - 1)
                    .map(|k| [nodes[k - 1].clone(), nodes[k].clone(), nodes[k + 1].clone()])
            })
            .collect()
    }

    /// Whether the grid contributes any curvature terms.
    pub(crate) fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }
}

fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Collect the observed values of each of a mapper's named inputs across the
/// population's subject-static covariates, in the mapper's input order and
/// **normalised the way the mapper feeds them to its MLP** (`(x − center) /
/// scale`, [`NamedMlpMapper::normalize`]) — so the grid built from them lives in
/// the space the fit actually evaluates the network in. Missing covariates are
/// skipped (they never contribute a value); a subject's `NaN` placeholder for a
/// column it has no finite value in passes through and is dropped by
/// [`CurvatureGrid::build`].
///
/// This reads the subject-static covariate snapshot (the first finite row per
/// subject). A time-varying NN input is therefore swept over its *baseline*
/// range; later-occasion values only enter the grid through the subjects whose
/// baseline lands there.
fn collect_input_values(mapper: &NamedMlpMapper, population: &Population) -> Vec<Vec<f64>> {
    mapper
        .input_names()
        .iter()
        .enumerate()
        .map(|(i, name)| {
            population
                .subjects
                .iter()
                .filter_map(|s| s.covariates.get(name).copied())
                .map(|x| mapper.normalize(i, x))
                .collect()
        })
        .collect()
}

/// Per-network regularization state, built once before the optimize loop and
/// shared (by reference) across every objective/gradient evaluation.
///
/// Holds a cheap clone of each NN's inner [`MlpMapper`], the theta offset of
/// its weight block, and (when smoothness is on) its prebuilt
/// [`CurvatureGrid`]. Built with no specs — so every method is a no-op that
/// touches nothing — unless at least one λ is positive, and each kernel is
/// itself a no-op at `λ == 0.0`, so an unregularized fit adds nothing.
///
/// The penalty is applied to the **optimizer's** objective only. `theta` is the
/// full natural-space theta vector; NN weights are identity-packed (the parser
/// gives them `theta_lower = −∞`), so a natural-space weight coordinate *is* a
/// packed coordinate and the raw-weight gradient / Hessian map one-to-one onto
/// packed indices. The outer optimizers' per-coordinate `scale[k]` is applied
/// by the caller on the assembled packed gradient, exactly as it is to the
/// likelihood part — the penalty needs no separate Jacobian-of-scaling factor.
#[derive(Debug, Clone)]
pub(crate) struct NnRegularizer {
    l2_lambda: f64,
    smooth_lambda: f64,
    specs: Vec<NnRegSpec>,
}

#[derive(Debug, Clone)]
struct NnRegSpec {
    weights_offset: usize,
    mapper: MlpMapper,
    grid: CurvatureGrid,
}

impl NnRegularizer {
    /// Build the regularizer from the model's `[covariate_nn]` blocks and the
    /// fit options' `nn_l2` / `nn_smooth` strengths. When both λ are `0.0` the
    /// spec list is empty and every method is a no-op.
    pub(crate) fn build(
        model: &CompiledModel,
        population: &Population,
        options: &FitOptions,
    ) -> Self {
        let l2_lambda = options.nn_l2_lambda;
        let smooth_lambda = options.nn_smooth_lambda;
        let mut specs = Vec::new();
        if l2_lambda > 0.0 || smooth_lambda > 0.0 {
            for nn in &model.covariate_nns {
                let grid = if smooth_lambda > 0.0 {
                    CurvatureGrid::build(
                        &collect_input_values(&nn.mapper, population),
                        NN_SMOOTH_GRID_STEP_Z,
                    )
                } else {
                    CurvatureGrid::default()
                };
                specs.push(NnRegSpec {
                    weights_offset: nn.weights_offset,
                    mapper: nn.mapper.mlp().clone(),
                    grid,
                });
            }
        }
        Self {
            l2_lambda,
            smooth_lambda,
            specs,
        }
    }

    /// Whether any penalty term is active (a positive λ and at least one NN).
    pub(crate) fn is_active(&self) -> bool {
        !self.specs.is_empty()
    }

    /// Additive penalty on the population objective: `Σ_nn (L2 + smoothness)`.
    /// Returns `0.0` when inactive.
    pub(crate) fn penalty_value(&self, theta: &[f64]) -> f64 {
        self.accumulate(theta, None, None)
    }

    /// Add the penalty gradient into `grad` at the NN-weight (packed)
    /// coordinates. No-op when inactive.
    pub(crate) fn add_packed_gradient(&self, theta: &[f64], grad: &mut [f64]) {
        self.accumulate(theta, Some(grad), None);
    }

    /// Penalty value **and** gradient in one pass — the gradient-path callers
    /// use this instead of `penalty_value` + `add_packed_gradient`, which would
    /// evaluate every curvature node twice.
    pub(crate) fn penalty_and_gradient(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        self.accumulate(theta, Some(grad), None)
    }

    /// Add the penalty's Hessian contributions through `add(i, j, v)` at packed
    /// coordinates: exact `2·λ_l2` on the weight diagonal for L2, and the
    /// Gauss–Newton `2·λ_smooth·Σ ∂C/∂w ∂C/∂wᵀ` for the curvature term (see
    /// [`MlpMapper::curvature_penalty_accum`]). Storage-agnostic so the
    /// trust-region `Vec<Vec<f64>>` and the Gauss–Newton `DMatrix` both feed
    /// from it. No-op when inactive.
    pub(crate) fn add_packed_hessian(&self, theta: &[f64], add: &mut dyn FnMut(usize, usize, f64)) {
        self.accumulate(theta, None, Some(add));
    }

    fn accumulate(
        &self,
        theta: &[f64],
        mut grad: Option<&mut [f64]>,
        mut hess: Option<&mut dyn FnMut(usize, usize, f64)>,
    ) -> f64 {
        let mut p = 0.0;
        for spec in &self.specs {
            let off = spec.weights_offset;
            let n_w = spec.mapper.n_weights();
            let w = &theta[off..off + n_w];
            let mut hess_local = hess
                .as_deref_mut()
                .map(|h| move |i: usize, j: usize, v: f64| h(off + i, off + j, v));
            let mut sink = PenaltySink {
                grad: grad.as_deref_mut().map(|g| &mut g[off..off + n_w]),
                hess: hess_local
                    .as_mut()
                    .map(|h| h as &mut dyn FnMut(usize, usize, f64)),
            };
            p += spec
                .mapper
                .l2_weight_penalty_accum(w, self.l2_lambda, &mut sink);
            p += spec
                .mapper
                .curvature_penalty_accum(w, &spec.grid, self.smooth_lambda, &mut sink);
        }
        p
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Construct a 2→3→1 MLP with identity activations everywhere. Forward
    /// pass then reduces to two linear maps composed.
    #[test]
    fn forward_identity_matches_hand_computed() {
        let mlp = MlpMapper::new(vec![2, 3, 1], Activation::Identity, Activation::Identity)
            .expect("valid layers");
        // Layer 1: W_1 (3×2) + b_1 (3) = 9; Layer 2: W_2 (1×3) + b_2 (1) = 4.
        assert_eq!(mlp.n_weights(), 13);

        // Layer 1: W_1 = [[1, 2], [3, 4], [5, 6]], b_1 = [0.1, 0.2, 0.3]
        // Layer 2: W_2 = [[1, 1, 1]],            b_2 = [0.0]
        let weights = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // W_1 row-major
            0.1, 0.2, 0.3, // b_1
            1.0, 1.0, 1.0, // W_2
            0.0, // b_2
        ];
        let x = vec![1.0, 2.0];
        let y = mlp.forward(&x, &weights).expect("forward ok");

        // h = W_1 x + b_1 = [1+4+0.1, 3+8+0.2, 5+12+0.3] = [5.1, 11.2, 17.3]
        // y = W_2 h + b_2 = 5.1 + 11.2 + 17.3 = 33.6
        assert_eq!(y.len(), 1);
        assert_relative_eq!(y[0], 33.6, epsilon = 1e-12);
    }

    /// ReLU activation in the hidden layer with a known mix of active /
    /// inactive units.
    #[test]
    fn forward_relu_handles_inactive_units() {
        let mlp = MlpMapper::new(vec![1, 3, 1], Activation::Relu, Activation::Identity).unwrap();
        // W_1 = [[1], [-1], [2]], b_1 = [0, 0, -3]. For x = 1:
        //   z = [1, -1, -1]. ReLU(z) = [1, 0, 0].
        // W_2 = [[1, 1, 1]], b_2 = [0]. y = 1.
        let weights = vec![1.0, -1.0, 2.0, 0.0, 0.0, -3.0, 1.0, 1.0, 1.0, 0.0];
        let y = mlp.forward(&[1.0], &weights).unwrap();
        assert_relative_eq!(y[0], 1.0, epsilon = 1e-12);
    }

    /// Deterministic, non-degenerate weights for FD comparisons. Small
    /// magnitudes keep bounded activations in their responsive range so the
    /// finite differences have clean signal.
    fn probe_weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.1 * (i as f64).sin() + 0.05 * ((i * 7) as f64).cos())
            .collect()
    }

    /// Forward pass stopped at the output layer's pre-activation `z_L`, i.e.
    /// `forward` with the output activation peeled off. Test-side reference
    /// for what `jacobian_preactivation` claims to differentiate.
    fn forward_preactivation(mlp: &MlpMapper, x: &[f64], weights: &[f64]) -> Vec<f64> {
        let (pre, _post) = mlp.forward_cache(x, weights);
        pre.last()
            .expect("at least one layer")
            .iter()
            .copied()
            .collect()
    }

    /// `jacobian_preactivation` must be the central FD of `z_L`, not of the
    /// activated output. Uses `Softplus` on the output head so the two
    /// genuinely differ — under `Identity` the test would pass even if the
    /// `preactivation` flag were ignored.
    #[test]
    fn jacobian_preactivation_matches_central_fd_of_z() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Softplus).unwrap();
        let n_w = mlp.n_weights();
        let weights = probe_weights(n_w);
        let x = vec![0.3, -0.7];

        let jac = mlp.jacobian_preactivation(&x, &weights).unwrap();
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), n_w);

        let eps = 1e-7;
        let mut perturbed = weights.clone();
        for j in 0..n_w {
            let saved = perturbed[j];
            perturbed[j] = saved + eps;
            let z_plus = forward_preactivation(&mlp, &x, &perturbed);
            perturbed[j] = saved - eps;
            let z_minus = forward_preactivation(&mlp, &x, &perturbed);
            perturbed[j] = saved;
            for i in 0..3 {
                let fd = (z_plus[i] - z_minus[i]) / (2.0 * eps);
                assert_relative_eq!(jac[(i, j)], fd, epsilon = 1e-6, max_relative = 1e-5);
            }
        }
    }

    /// Under an identity output head the two Jacobians describe the same
    /// function, so they must agree exactly — a cheap guard that the shared
    /// `jacobian_impl` refactor did not perturb the original path.
    #[test]
    fn jacobian_preactivation_equals_jacobian_under_identity_head() {
        let mlp = MlpMapper::new(vec![3, 5, 2], Activation::Tanh, Activation::Identity).unwrap();
        let weights = probe_weights(mlp.n_weights());
        let x = vec![0.2, -0.4, 0.9];

        let post = mlp.jacobian(&x, &weights).unwrap();
        let pre = mlp.jacobian_preactivation(&x, &weights).unwrap();
        for i in 0..post.nrows() {
            for j in 0..post.ncols() {
                assert_relative_eq!(pre[(i, j)], post[(i, j)], epsilon = 1e-15);
            }
        }
    }

    /// The identity the hybrid NN gradient rests on: the output-layer bias
    /// `b_k` moves `z_k` one-for-one and moves no other output's
    /// pre-activation at all.
    #[test]
    fn output_bias_moves_only_its_own_preactivation() {
        let mlp = MlpMapper::new(vec![2, 3, 4], Activation::Tanh, Activation::Softplus).unwrap();
        let weights = probe_weights(mlp.n_weights());
        let x = vec![0.5, -0.2];
        let jac = mlp.jacobian_preactivation(&x, &weights).unwrap();

        for k in 0..mlp.n_outputs() {
            let b_k = mlp.output_bias_index(k);
            for i in 0..mlp.n_outputs() {
                let expected = if i == k { 1.0 } else { 0.0 };
                assert_relative_eq!(jac[(i, b_k)], expected, epsilon = 1e-15);
            }
        }
    }

    /// A saturated `Softplus` head drives the *post*-activation Jacobian's
    /// bias entry to ~0 while the pre-activation entry stays exactly 1. This
    /// is the case that makes dividing by `f'(z_k)` unusable and the
    /// pre-activation seed necessary — not a stylistic preference.
    #[test]
    fn saturated_head_kills_post_activation_bias_but_not_preactivation() {
        let mlp =
            MlpMapper::new(vec![1, 1, 1], Activation::Identity, Activation::Softplus).unwrap();
        // W_1 = [1], b_1 = [0], W_2 = [1], b_2 = [-60]. For x = 1: z_2 = -59,
        // so softplus'(z_2) = sigmoid(-59) ≈ 2e-26.
        let weights = vec![1.0, 0.0, 1.0, -60.0];
        let x = vec![1.0];

        let b_k = mlp.output_bias_index(0);
        let post = mlp.jacobian(&x, &weights).unwrap();
        let pre = mlp.jacobian_preactivation(&x, &weights).unwrap();

        assert!(
            post[(0, b_k)].abs() < 1e-20,
            "expected a saturated post-activation entry, got {}",
            post[(0, b_k)]
        );
        assert_relative_eq!(pre[(0, b_k)], 1.0, epsilon = 1e-15);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn output_bias_index_rejects_out_of_range_output() {
        let mlp = MlpMapper::new(vec![2, 3, 2], Activation::Tanh, Activation::Identity).unwrap();
        let _ = mlp.output_bias_index(2);
    }

    /// The Jacobian computed analytically must match central FD to high
    /// precision. This is the strongest correctness check on the backprop.
    #[test]
    fn jacobian_matches_central_fd() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Identity).unwrap();
        let n_w = mlp.n_weights();
        // Deterministic non-trivial weights — small magnitudes keep tanh in
        // its linear regime so FD has clean signal.
        let weights: Vec<f64> = (0..n_w)
            .map(|i| 0.1 * (i as f64).sin() + 0.05 * ((i * 7) as f64).cos())
            .collect();
        let x = vec![0.3, -0.7];

        let jac = mlp.jacobian(&x, &weights).unwrap();
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), n_w);

        let eps = 1e-7;
        let mut perturbed = weights.clone();
        for j in 0..n_w {
            let saved = perturbed[j];
            perturbed[j] = saved + eps;
            let y_plus = mlp.forward(&x, &perturbed).unwrap();
            perturbed[j] = saved - eps;
            let y_minus = mlp.forward(&x, &perturbed).unwrap();
            perturbed[j] = saved;
            for i in 0..3 {
                let fd = (y_plus[i] - y_minus[i]) / (2.0 * eps);
                assert_relative_eq!(jac[(i, j)], fd, epsilon = 1e-6, max_relative = 1e-5);
            }
        }
    }

    /// ReLU has a kink at 0; verify the jacobian matches FD on the smooth
    /// side and is exactly zero on the inactive side.
    #[test]
    fn jacobian_relu_zeros_inactive_paths() {
        let mlp = MlpMapper::new(vec![1, 2, 1], Activation::Relu, Activation::Identity).unwrap();
        // W_1 = [[1], [-1]], b_1 = [0, 0]. For x = 1: z = [1, -1].
        //   ReLU(z) = [1, 0]. Unit 2 is inactive — its weights and bias
        //   should have zero gradient.
        let weights = vec![1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let jac = mlp.jacobian(&[1.0], &weights).unwrap();
        // Layer 1 W: [w_11, w_21] at indices 0, 1. Bias at 2, 3.
        // Layer 2 W: indices 4, 5. Bias at 6.
        // Output y = ReLU(w_11*x+b_11) * w_21_out + ReLU(w_21*x+b_21) * w_22_out + b_2.
        // dy/dw_21 = 0 (unit 2 inactive). dy/db_21 = 0. dy/dw_22_out = 0
        // (multiplied by inactive ReLU output).
        assert_relative_eq!(jac[(0, 1)], 0.0, epsilon = 1e-12); // w_21
        assert_relative_eq!(jac[(0, 3)], 0.0, epsilon = 1e-12); // b_21
        assert_relative_eq!(jac[(0, 5)], 0.0, epsilon = 1e-12); // w_22_out
                                                                // Active path: dy/dw_11 = x * w_21_out = 1 * 1 = 1.
        assert_relative_eq!(jac[(0, 0)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn constructor_rejects_invalid_layers() {
        assert!(matches!(
            MlpMapper::new(vec![3], Activation::Identity, Activation::Identity),
            Err(NnError::InvalidLayers(_))
        ));
        assert!(matches!(
            MlpMapper::new(vec![2, 0, 1], Activation::Identity, Activation::Identity),
            Err(NnError::ZeroLayerDimension { index: 1, value: 0 })
        ));
    }

    #[test]
    fn forward_rejects_mismatched_shapes() {
        let mlp =
            MlpMapper::new(vec![2, 3, 1], Activation::Identity, Activation::Identity).unwrap();
        assert!(matches!(
            mlp.forward(&[1.0], &[0.0; 13]),
            Err(NnError::InputCountMismatch { .. })
        ));
        assert!(matches!(
            mlp.forward(&[1.0, 2.0], &[0.0; 5]),
            Err(NnError::WeightCountMismatch { .. })
        ));
    }

    /// Activation derivatives match central finite differences at random
    /// points; covers the AD-safe `if`/`else` branches.
    #[test]
    fn activation_derivatives_match_fd() {
        let xs = [-3.0, -1.0, -0.001, 0.0, 0.001, 1.0, 3.0, 10.0];
        let eps = 1e-6;
        for activation in [
            Activation::Identity,
            Activation::Softplus,
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::Exp,
        ] {
            for &x in &xs {
                let fd = (activation.apply(x + eps) - activation.apply(x - eps)) / (2.0 * eps);
                assert_relative_eq!(
                    activation.derivative(x),
                    fd,
                    epsilon = 1e-5,
                    max_relative = 1e-4
                );
            }
        }
        // ReLU separately — skip the kink at 0.
        for &x in &[-3.0, -1.0, -0.001, 0.001, 1.0, 3.0] {
            let fd =
                (Activation::Relu.apply(x + eps) - Activation::Relu.apply(x - eps)) / (2.0 * eps);
            assert_relative_eq!(Activation::Relu.derivative(x), fd, epsilon = 1e-9);
        }
    }

    // -----------------------------------------------------------------
    // NamedMlpMapper / CovariateMapper integration
    // -----------------------------------------------------------------

    fn five_param_mapper() -> NamedMlpMapper {
        let mlp = MlpMapper::new(vec![2, 4, 5], Activation::Tanh, Activation::Softplus).unwrap();
        NamedMlpMapper::new(
            mlp,
            vec!["WT".into(), "CRCL".into()],
            vec![
                "CL".into(),
                "V1".into(),
                "Q".into(),
                "V2".into(),
                "KA".into(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn named_mapper_writes_into_correct_pk_slots() {
        use crate::types::{PK_IDX_CL, PK_IDX_KA, PK_IDX_Q, PK_IDX_V, PK_IDX_V2};

        let mapper = five_param_mapper();
        let n_w = mapper.n_weights();
        let weights: Vec<f64> = (0..n_w).map(|i| 0.1 * (i as f64).sin()).collect();
        let mut covariates = HashMap::new();
        covariates.insert("WT".to_string(), 70.0);
        covariates.insert("CRCL".to_string(), 100.0);

        let mut out = PkParams::default();
        // Sanity: F starts at 1.0 and must remain 1.0 after forward — NN
        // does not touch unmapped slots.
        assert_relative_eq!(out.f_bio(), 1.0);

        mapper.forward(&weights, &covariates, &mut out).unwrap();

        // Softplus output → all PK params must be strictly positive.
        for idx in [PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_KA] {
            assert!(
                out.values[idx] > 0.0,
                "expected positive PK param at index {}, got {}",
                idx,
                out.values[idx]
            );
        }
        assert_relative_eq!(out.f_bio(), 1.0); // F untouched.
    }

    #[test]
    fn named_mapper_reports_missing_covariate() {
        let mapper = five_param_mapper();
        let weights = vec![0.0; mapper.n_weights()];
        let mut covariates = HashMap::new();
        covariates.insert("WT".to_string(), 70.0); // missing CRCL
        let mut out = PkParams::default();
        let err = mapper.forward(&weights, &covariates, &mut out).unwrap_err();
        assert!(matches!(err, NnError::MissingCovariate(ref n) if n == "CRCL"));
    }

    #[test]
    fn forward_raw_substitutes_zero_for_missing_covariates() {
        // `forward_raw` is the parser-facing entrypoint and must match the
        // expression evaluator's `unwrap_or(0.0)` semantics — missing
        // covariates become 0.0 inputs, not errors. Reference: this commit's
        // fix to silent error-swallowing at NN dispatch sites.
        let mapper = five_param_mapper();
        let weights = vec![0.1; mapper.n_weights()];

        let mut both = HashMap::new();
        both.insert("WT".to_string(), 70.0);
        both.insert("CRCL".to_string(), 0.0); // explicit zero for CRCL
        let y_explicit = mapper.forward_raw(&weights, &both).unwrap();

        let mut wt_only = HashMap::new();
        wt_only.insert("WT".to_string(), 70.0);
        let y_implicit = mapper.forward_raw(&weights, &wt_only).unwrap();

        // Missing CRCL must produce identical outputs to CRCL = 0.0.
        assert_eq!(y_explicit.len(), y_implicit.len());
        for (a, b) in y_explicit.iter().zip(y_implicit.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }
    }

    #[test]
    fn forward_raw_surfaces_weight_count_mismatch() {
        // `MissingCovariate` is no longer reachable via `forward_raw`, but
        // genuine wiring bugs (wrong weight slice length) must still surface
        // as errors so callers can `.expect(...)` them loudly.
        let mapper = five_param_mapper();
        let bad_weights = vec![0.0; mapper.n_weights() - 1];
        let mut covariates = HashMap::new();
        covariates.insert("WT".to_string(), 70.0);
        covariates.insert("CRCL".to_string(), 100.0);
        let err = mapper.forward_raw(&bad_weights, &covariates).unwrap_err();
        assert!(matches!(err, NnError::WeightCountMismatch { .. }));
    }

    #[test]
    fn named_mapper_rejects_unknown_pk_output() {
        let mlp = MlpMapper::new(vec![1, 2, 1], Activation::Tanh, Activation::Identity).unwrap();
        let err =
            NamedMlpMapper::new(mlp, vec!["WT".into()], vec!["NOT_A_PK_PARAM".into()]).unwrap_err();
        assert!(matches!(err, NnError::UnknownPkOutput(ref n) if n == "NOT_A_PK_PARAM"));
    }

    #[test]
    fn named_mapper_rejects_duplicate_outputs() {
        let mlp = MlpMapper::new(vec![1, 2, 2], Activation::Tanh, Activation::Identity).unwrap();
        let err = NamedMlpMapper::new(mlp, vec!["WT".into()], vec!["CL".into(), "CL".into()])
            .unwrap_err();
        assert!(matches!(err, NnError::DuplicateOutput(_)));
    }

    /// Higher-level integration: build the kind of pk_param_fn closure the
    /// parser will eventually produce, and confirm it matches the analytical
    /// `tv * exp(eta)` shape when eta = 0. This is a forward-looking sanity
    /// check for the M2 mu-ref composition.
    #[test]
    fn composing_named_mapper_with_eta_recovers_typical_value() {
        use crate::types::PK_IDX_CL;

        let mapper = five_param_mapper();
        let n_w = mapper.n_weights();
        let weights: Vec<f64> = (0..n_w).map(|i| 0.05 * ((i * 3) as f64).sin()).collect();
        let mut cov = HashMap::new();
        cov.insert("WT".into(), 65.0);
        cov.insert("CRCL".into(), 90.0);

        // Typical value (eta=0)
        let mut tv = PkParams::default();
        mapper.forward(&weights, &cov, &mut tv).unwrap();
        let tv_cl = tv.values[PK_IDX_CL];

        // Mixed-effects composition: CL = tv * exp(eta_cl)
        let eta_cl: f64 = 0.3;
        let cl_indiv = tv_cl * eta_cl.exp();
        assert!(cl_indiv > tv_cl); // positive eta increases CL
        assert_relative_eq!(cl_indiv / tv_cl, eta_cl.exp(), epsilon = 1e-12);
    }

    /// A rejected `center` or `scale` entry must name the **offending input** and report
    /// the value the user actually wrote.
    ///
    /// The center check used to reuse `InvalidScale` with a hardcoded `input: "center"`
    /// and a fabricated `NaN`, so `center = [inf, 0]` was reported as a *scale* error, on
    /// an input named "center", carrying a value that never appeared in the model file —
    /// three wrong facts in one message.
    #[test]
    fn normalization_errors_name_the_offending_input_and_value() {
        let mapper = || {
            NamedMlpMapper::new(
                MlpMapper::new(vec![2, 2, 1], Activation::Tanh, Activation::Identity)
                    .expect("valid layers"),
                vec!["WT".to_string(), "CRCL".to_string()],
                vec!["CL".to_string()],
            )
            .expect("valid mapper")
        };

        // Second input's scale is zero — a division the forward pass cannot survive.
        match mapper().with_normalization(vec![0.0, 0.0], vec![1.0, 0.0]) {
            Err(NnError::InvalidScale { input, value }) => {
                assert_eq!(input, "CRCL");
                assert_eq!(value, 0.0);
            }
            other => panic!("expected InvalidScale on CRCL, got {other:?}"),
        }

        // Second input's center is non-finite. Distinct variant, real name, real value.
        match mapper().with_normalization(vec![0.0, f64::INFINITY], vec![1.0, 1.0]) {
            Err(NnError::InvalidCenter { input, value }) => {
                assert_eq!(input, "CRCL");
                assert!(value.is_infinite());
                assert!(
                    format!("{}", NnError::InvalidCenter { input, value }).contains("`center`"),
                    "the message must say which key is at fault"
                );
            }
            other => panic!("expected InvalidCenter on CRCL, got {other:?}"),
        }
    }

    /// The transform a network was fitted under is readable back off the mapper, which is
    /// what lets `NeuralNetworkInfo` report it. Identity by default, so a model that
    /// declares no normalisation reports vectors that change nothing.
    #[test]
    fn normalization_vectors_are_readable_and_default_to_identity() {
        let base = NamedMlpMapper::new(
            MlpMapper::new(vec![2, 2, 1], Activation::Tanh, Activation::Identity)
                .expect("valid layers"),
            vec!["WT".to_string(), "CRCL".to_string()],
            vec!["CL".to_string()],
        )
        .expect("valid mapper");
        assert_eq!(base.input_center(), &[0.0, 0.0]);
        assert_eq!(base.input_scale(), &[1.0, 1.0]);

        let normed = base
            .with_normalization(vec![70.0, 90.0], vec![15.0, 30.0])
            .expect("valid normalisation");
        assert_eq!(normed.input_center(), &[70.0, 90.0]);
        assert_eq!(normed.input_scale(), &[15.0, 30.0]);
    }
}

#[cfg(test)]
mod invert_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `invert` must be the exact inverse of `apply` across each activation's
    /// range — including the piecewise branches `apply` uses for numerical
    /// stability, which are the easy place for an inverse to disagree.
    #[test]
    fn invert_round_trips_through_apply() {
        let cases: &[(Activation, &[f64])] = &[
            (Activation::Identity, &[-3.0, 0.0, 7.5]),
            (Activation::Relu, &[0.25, 1.0, 40.0]),
            // Spans the `y > 20` branch of `apply` and the small-`y` end where
            // `exp(y) - 1` would lose precision to cancellation. `LN_2` is the
            // deliberate middle point: `softplus(0) = ln 2`, so it is the one input
            // whose inverse is exactly `0`. Spelled as the constant rather than a
            // truncated `0.693_147` so it lands on that zero exactly (and so clippy's
            // `approx_constant` does not have to be silenced for a value that really
            // is ln 2).
            (
                Activation::Softplus,
                &[1e-8, 1e-3, std::f64::consts::LN_2, 1.0, 10.0, 25.0, 400.0],
            ),
            (Activation::Tanh, &[-0.95, 0.0, 0.5]),
            (Activation::Sigmoid, &[0.01, 0.5, 0.99]),
            (Activation::Exp, &[1e-6, 1.0, 500.0]),
        ];
        for (act, ys) in cases {
            for &y in *ys {
                let z = act
                    .invert(y)
                    .unwrap_or_else(|| panic!("{:?} should invert {y}", act));
                assert_relative_eq!(act.apply(z), y, max_relative = 1e-9);
            }
        }
    }

    /// Out-of-range values must decline rather than return a NaN that would
    /// silently become a NaN weight.
    #[test]
    fn invert_declines_outside_the_range() {
        assert!(Activation::Softplus.invert(0.0).is_none());
        assert!(Activation::Softplus.invert(-1.0).is_none());
        assert!(Activation::Exp.invert(0.0).is_none());
        assert!(Activation::Relu.invert(0.0).is_none());
        assert!(Activation::Tanh.invert(1.0).is_none());
        assert!(Activation::Sigmoid.invert(1.0).is_none());
        assert!(Activation::Sigmoid.invert(0.0).is_none());
        assert!(Activation::Identity.invert(f64::NAN).is_none());
        assert!(Activation::Identity.invert(f64::INFINITY).is_none());
    }

    /// The range blurb is user-facing error text (`[covariate_nn] init` quotes it
    /// when a declared output value cannot be inverted), so it has to *describe
    /// the range `invert` actually enforces*. Pinning the string alone would let
    /// the two drift apart silently — a widened `invert` with a stale blurb tells
    /// the user their value is out of range when it is not. So each arm is paired
    /// with a probe the text excludes, and `invert` must agree by declining it.
    #[test]
    fn range_description_describes_the_range_invert_enforces() {
        let cases = [
            (Activation::Identity, "any finite value", f64::NAN),
            (Activation::Relu, "a value > 0", 0.0),
            (Activation::Softplus, "a value > 0", 0.0),
            (Activation::Tanh, "a value strictly between -1 and 1", 1.0),
            (Activation::Sigmoid, "a value strictly between 0 and 1", 1.0),
            (Activation::Exp, "a value > 0", 0.0),
        ];
        for (act, expected, excluded) in cases {
            assert_eq!(
                act.range_description(),
                expected,
                "{act:?} reports an unexpected range description"
            );
            assert!(
                act.invert(excluded).is_none(),
                "{act:?} says its range is `{expected}` but inverts the excluded value {excluded}"
            );
        }
    }

    /// The output-layer weight block must be exactly the entries between the
    /// last layer's start and its first bias — the range `init` zeroes.
    #[test]
    fn output_weight_range_covers_the_last_layer_weights_only() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Softplus).unwrap();
        let r = mlp.output_weight_range();
        assert_eq!(r.len(), 3 * 4, "W_L is n_out x n_hidden");
        assert_eq!(r.end, mlp.output_bias_index(0));
        // And it must not overlap any bias.
        for k in 0..mlp.n_outputs() {
            assert!(!r.contains(&mlp.output_bias_index(k)));
        }
    }

    /// The property `init` exists to provide: with `W_L` zeroed and the biases
    /// set to `f^{-1}(v)`, the network emits exactly `v` — for *any* input, so
    /// every subject starts at the same declared value.
    #[test]
    fn zeroed_output_weights_make_the_bias_the_whole_output() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Softplus).unwrap();
        let mut w: Vec<f64> = (0..mlp.n_weights())
            .map(|i| 0.3 * (i as f64).sin())
            .collect();
        for i in mlp.output_weight_range() {
            w[i] = 0.0;
        }
        let targets = [0.75f64, 10.0, 3.5];
        for (k, &v) in targets.iter().enumerate() {
            w[mlp.output_bias_index(k)] = Activation::Softplus.invert(v).unwrap();
        }
        // Two very different input vectors must give the same output.
        for x in [[0.0, 0.0], [4.0, -9.0]] {
            let y = mlp.forward(&x, &w).unwrap();
            for (k, &v) in targets.iter().enumerate() {
                assert_relative_eq!(y[k], v, max_relative = 1e-12);
            }
        }
    }
}

#[cfg(test)]
mod regularization_tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------

    /// Deterministic non-trivial weights, small magnitudes so tanh stays in a
    /// smooth regime (clean finite-difference signal).
    fn det_weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 0.2 * (i as f64 * 0.7).sin() + 0.1 * (i as f64 * 1.3).cos())
            .collect()
    }

    #[test]
    fn weight_param_indices_excludes_biases() {
        // 2→3→1: layer1 W=6 (0..6) b=3 (6..9); layer2 W=3 (9..12) b=1 (12).
        let mlp = MlpMapper::new(vec![2, 3, 1], Activation::Tanh, Activation::Identity).unwrap();
        assert_eq!(mlp.n_weights(), 13);
        assert_eq!(
            mlp.weight_param_indices(),
            vec![0, 1, 2, 3, 4, 5, 9, 10, 11]
        );
    }

    #[test]
    fn l2_penalty_zero_lambda_is_noop() {
        let mlp = MlpMapper::new(vec![2, 3, 1], Activation::Tanh, Activation::Softplus).unwrap();
        let w = det_weights(mlp.n_weights());
        assert_eq!(mlp.l2_weight_penalty_value(&w, 0.0), 0.0);
        let (v, g) = mlp.l2_weight_penalty(&w, 0.0);
        assert_eq!(v, 0.0);
        assert!(g.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn l2_penalty_value_sums_weights_not_biases() {
        let mlp = MlpMapper::new(vec![2, 3, 1], Activation::Tanh, Activation::Identity).unwrap();
        let w = det_weights(mlp.n_weights());
        let lambda = 0.37;
        // Hand-sum over the weight indices only.
        let expected: f64 = lambda
            * mlp
                .weight_param_indices()
                .iter()
                .map(|&i| w[i] * w[i])
                .sum::<f64>();
        assert_relative_eq!(
            mlp.l2_weight_penalty_value(&w, lambda),
            expected,
            epsilon = 1e-12
        );
        // A large bias must not change the penalty (biases excluded).
        let mut w_big_bias = w.clone();
        w_big_bias[6] += 100.0; // a layer-1 bias index
        assert_relative_eq!(
            mlp.l2_weight_penalty_value(&w_big_bias, lambda),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn l2_penalty_gradient_matches_central_fd() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Softplus).unwrap();
        let w = det_weights(mlp.n_weights());
        let lambda = 0.53;
        let (_v, g) = mlp.l2_weight_penalty(&w, lambda);
        let eps = 1e-6;
        let mut p = w.clone();
        for j in 0..w.len() {
            let saved = p[j];
            p[j] = saved + eps;
            let vp = mlp.l2_weight_penalty_value(&p, lambda);
            p[j] = saved - eps;
            let vm = mlp.l2_weight_penalty_value(&p, lambda);
            p[j] = saved;
            let fd = (vp - vm) / (2.0 * eps);
            assert_relative_eq!(g[j], fd, epsilon = 1e-6, max_relative = 1e-5);
        }
    }

    #[test]
    fn curvature_grid_skips_constant_and_narrow_inputs() {
        // Input 0 varies widely; input 1 is constant (sd=0 → no axis).
        let vals = vec![
            vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let grid = CurvatureGrid::build(&vals, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty(), "wide input 0 must yield curvature nodes");
        // Every triple must hold input 1 fixed at its median (1.0) and vary only
        // input 0 across its three stencil points.
        for t in grid.triples() {
            assert_eq!(t[0].len(), 2);
            assert_relative_eq!(t[0][1], 1.0, epsilon = 1e-12);
            assert_relative_eq!(t[1][1], 1.0, epsilon = 1e-12);
            assert_relative_eq!(t[2][1], 1.0, epsilon = 1e-12);
            // Uniform stencil in raw space: center is the midpoint of ±h.
            assert_relative_eq!(t[1][0], 0.5 * (t[0][0] + t[2][0]), epsilon = 1e-9);
        }
        // Both inputs constant → no curvature axis at all → empty grid (no-op).
        let both_constant = vec![vec![5.0; 6], vec![1.0; 6]];
        assert!(CurvatureGrid::build(&both_constant, NN_SMOOTH_GRID_STEP_Z).is_empty());
        // A non-positive step is also a no-op.
        assert!(CurvatureGrid::build(&vals, 0.0).is_empty());
    }

    #[test]
    fn curvature_penalty_zero_lambda_is_noop() {
        let mlp = MlpMapper::new(vec![2, 3, 2], Activation::Tanh, Activation::Softplus).unwrap();
        let w = det_weights(mlp.n_weights());
        let vals = vec![
            vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            vec![30.0, 50.0, 70.0, 90.0, 110.0, 130.0],
        ];
        let grid = CurvatureGrid::build(&vals, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty());
        assert_eq!(mlp.curvature_penalty_value(&w, &grid, 0.0), 0.0);
        let (v, g) = mlp.curvature_penalty(&w, &grid, 0.0);
        assert_eq!(v, 0.0);
        assert!(g.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn curvature_penalty_zero_for_affine_network() {
        // A single-layer identity net is affine in its inputs, so the 2nd
        // difference C = f(x+h) − 2f(x) + f(x−h) is exactly 0 everywhere.
        let mlp = MlpMapper::new(vec![2, 1], Activation::Identity, Activation::Identity).unwrap();
        let w = det_weights(mlp.n_weights());
        let vals = vec![
            vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            vec![30.0, 50.0, 70.0, 90.0, 110.0, 130.0],
        ];
        let grid = CurvatureGrid::build(&vals, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty());
        assert_relative_eq!(
            mlp.curvature_penalty_value(&w, &grid, 1.0),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn curvature_penalty_gradient_matches_central_fd() {
        let mlp = MlpMapper::new(vec![2, 4, 3], Activation::Tanh, Activation::Softplus).unwrap();
        let w = det_weights(mlp.n_weights());
        let vals = vec![
            vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            vec![30.0, 50.0, 70.0, 90.0, 110.0, 130.0],
        ];
        let grid = CurvatureGrid::build(&vals, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty());
        let lambda = 0.8;
        let (_v, g) = mlp.curvature_penalty(&w, &grid, lambda);
        let eps = 1e-6;
        let mut p = w.clone();
        for j in 0..w.len() {
            let saved = p[j];
            p[j] = saved + eps;
            let vp = mlp.curvature_penalty_value(&p, &grid, lambda);
            p[j] = saved - eps;
            let vm = mlp.curvature_penalty_value(&p, &grid, lambda);
            p[j] = saved;
            let fd = (vp - vm) / (2.0 * eps);
            assert_relative_eq!(g[j], fd, epsilon = 1e-6, max_relative = 1e-5);
        }
    }

    /// An input carrying no observed values at all contributes no curvature
    /// axis, and one whose z-span cannot fit an interior triple is skipped.
    ///
    /// The empty case is reachable in production: `collect_input_values` skips
    /// covariates a subject does not carry, so a covariate absent from every
    /// subject arrives here as an empty vector.
    #[test]
    fn curvature_grid_skips_empty_and_too_narrow_inputs() {
        // Input 0 has no observed values; input 1 varies widely.
        let with_empty = vec![vec![], vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0]];
        let grid = CurvatureGrid::build(&with_empty, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty(), "the varying input still yields nodes");
        // Every triple holds the empty input at its default 0.0 and sweeps only
        // input 1 — the empty axis contributed nothing of its own.
        for t in grid.triples() {
            assert_eq!(t[0][0], 0.0);
            assert_eq!(t[1][0], 0.0);
            assert_eq!(t[2][0], 0.0);
        }

        // Two distinct values always give a z-span of exactly 2.0 (±1 sd), so a
        // step wider than 1.0 leaves no room for an interior triple.
        let two_point = vec![vec![1.0, 2.0]];
        assert!(
            CurvatureGrid::build(&two_point, 1.5).is_empty(),
            "z-span 2.0 < 2·step 3.0 must yield no nodes"
        );
        // The same data at a step that does fit is not empty — so the emptiness
        // above is the span check, not the data being unusable.
        assert!(!CurvatureGrid::build(&two_point, 0.5).is_empty());
    }

    /// `median_of_sorted` on the three shapes it can meet: empty (the neutral
    /// 0.0 used for an input with no observed values), odd length (the middle
    /// element), and even length (the midpoint of the two central elements).
    #[test]
    fn median_of_sorted_handles_empty_odd_and_even() {
        assert_eq!(median_of_sorted(&[]), 0.0);
        assert_relative_eq!(median_of_sorted(&[1.0, 2.0, 9.0]), 2.0, epsilon = 1e-12);
        assert_relative_eq!(
            median_of_sorted(&[1.0, 2.0, 4.0, 9.0]),
            3.0,
            epsilon = 1e-12
        );
        // Single element is its own median.
        assert_relative_eq!(median_of_sorted(&[7.5]), 7.5, epsilon = 1e-12);
    }

    /// Past the node cap the sweep gets a **coarser step over the full span**,
    /// not a truncated one: the last node must still sit at the observed
    /// maximum. Regression — the cap once dropped the upper tail of any input
    /// whose z-span exceeded `(cap − 1) · step` (16 sd at the defaults).
    #[test]
    fn curvature_grid_coarsens_instead_of_truncating_past_the_node_cap() {
        // A wide input: 200 evenly spaced values plus one far outlier gives a
        // z-span of ~14 sd, well past 16/… — use a step small enough that the
        // default cap binds.
        let mut wide: Vec<f64> = (0..200).map(|i| i as f64).collect();
        wide.push(2000.0);
        let vals = vec![wide.clone()];
        let n = wide.len() as f64;
        let mean = wide.iter().sum::<f64>() / n;
        let sd = (wide.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let span_z = (2000.0 - mean) / sd - (0.0 - mean) / sd;
        let step = 0.05;
        assert!(
            span_z / step + 1.0 > NN_SMOOTH_GRID_MAX_NODES as f64,
            "fixture must exceed the cap at this step ({})",
            span_z / step
        );
        let grid = CurvatureGrid::build(&vals, step);
        let nodes = &grid.axes()[0];
        assert_eq!(
            nodes.len(),
            NN_SMOOTH_GRID_MAX_NODES,
            "capped at the max node count"
        );
        assert_relative_eq!(nodes[0][0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(nodes[nodes.len() - 1][0], 2000.0, epsilon = 1e-6);
        // Uniform in between.
        let h = nodes[1][0] - nodes[0][0];
        for k in 1..nodes.len() {
            assert_relative_eq!(nodes[k][0] - nodes[k - 1][0], h, epsilon = 1e-6);
        }
        // Below the cap the requested step is honoured exactly.
        let narrow = vec![vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0]];
        let g = CurvatureGrid::build(&narrow, NN_SMOOTH_GRID_STEP_Z);
        let nodes = &g.axes()[0];
        let sd = (narrow[0].iter().map(|v| (v - 75.0).powi(2)).sum::<f64>() / 6.0).sqrt();
        assert!(nodes.len() < NN_SMOOTH_GRID_MAX_NODES);
        assert_relative_eq!(
            nodes[1][0] - nodes[0][0],
            NN_SMOOTH_GRID_STEP_Z * sd,
            epsilon = 1e-9
        );
    }

    /// A subject with no finite value for an input arrives as `NaN` (the reader's
    /// placeholder) and must be dropped from that axis — not poison its mean/sd
    /// and silently delete the whole axis, which is what `NaN` did through
    /// `sd <= 0.0` (false), `span` (NaN) and `(NaN).floor() as usize` (0).
    #[test]
    fn curvature_grid_ignores_non_finite_values() {
        let clean = vec![vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0]];
        let mut dirty = clean.clone();
        dirty[0].push(f64::NAN);
        dirty[0].insert(2, f64::INFINITY);
        let g_clean = CurvatureGrid::build(&clean, NN_SMOOTH_GRID_STEP_Z);
        let g_dirty = CurvatureGrid::build(&dirty, NN_SMOOTH_GRID_STEP_Z);
        assert!(!g_clean.is_empty());
        assert_eq!(
            g_dirty.axes().len(),
            g_clean.axes().len(),
            "axis must survive"
        );
        for (a, b) in g_dirty.axes()[0].iter().zip(&g_clean.axes()[0]) {
            assert_relative_eq!(a[0], b[0], epsilon = 1e-12);
        }
        // All-NaN input: nothing usable, no axis, no panic.
        assert!(CurvatureGrid::build(&[vec![f64::NAN; 4]], NN_SMOOTH_GRID_STEP_Z).is_empty());
    }

    /// The curvature penalty must evaluate the network **where the fit does**:
    /// a `NamedMlpMapper` with `center`/`scale` feeds its MLP `(x − center) /
    /// scale`, so the grid built from its observed covariates must equal, node
    /// for node, the grid a mapper *without* normalisation builds from the
    /// pre-normalised covariates — and the penalty values must agree exactly.
    /// Regression: the grid was built on raw covariates and handed to the inner
    /// MLP, so with `center = [70, 86]`, `scale = [12, 23]` it probed `tanh` at
    /// pre-activations of ±100 where `C ≈ 0` and `nn_smooth` was a silent no-op.
    #[test]
    fn curvature_grid_lives_in_the_mappers_normalised_input_space() {
        use std::collections::HashMap;

        let names = vec!["WT".to_string(), "CRCL".to_string()];
        let outs = vec!["CL".to_string()];
        let center = vec![70.0, 86.0];
        let scale = vec![12.0, 23.0];
        let raw: Vec<(f64, f64)> = vec![
            (45.0, 46.0),
            (52.0, 60.0),
            (61.0, 70.0),
            (70.0, 75.0),
            (78.0, 82.0),
            (90.0, 66.0),
        ];

        let mlp = || MlpMapper::new(vec![2, 4, 1], Activation::Tanh, Activation::Softplus).unwrap();
        let normed = NamedMlpMapper::new(mlp(), names.clone(), outs.clone())
            .unwrap()
            .with_normalization(center.clone(), scale.clone())
            .unwrap();
        let plain = NamedMlpMapper::new(mlp(), names, outs).unwrap();

        let subj = |wt: f64, crcl: f64| {
            let mut c = HashMap::new();
            c.insert("WT".to_string(), wt);
            c.insert("CRCL".to_string(), crcl);
            c
        };
        let pop_raw = raw.iter().map(|&(w, c)| subj(w, c)).collect::<Vec<_>>();
        let pop_z = raw
            .iter()
            .map(|&(w, c)| subj((w - center[0]) / scale[0], (c - center[1]) / scale[1]))
            .collect::<Vec<_>>();
        let values = |mapper: &NamedMlpMapper, covs: &[HashMap<String, f64>]| -> Vec<Vec<f64>> {
            mapper
                .input_names()
                .iter()
                .enumerate()
                .map(|(i, n)| covs.iter().map(|c| mapper.normalize(i, c[n])).collect())
                .collect()
        };
        let g_normed = CurvatureGrid::build(&values(&normed, &pop_raw), NN_SMOOTH_GRID_STEP_Z);
        let g_plain = CurvatureGrid::build(&values(&plain, &pop_z), NN_SMOOTH_GRID_STEP_Z);
        assert!(!g_normed.is_empty());
        assert_eq!(g_normed.axes().len(), g_plain.axes().len());
        for (a, b) in g_normed.axes().iter().zip(g_plain.axes()) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                for (p, q) in x.iter().zip(y) {
                    assert_relative_eq!(p, q, epsilon = 1e-12);
                }
            }
            // The nodes are in z-space: nothing should look like a raw kilogram.
            for x in a {
                assert!(
                    x[0].abs() < 5.0 && x[1].abs() < 5.0,
                    "raw-looking node {x:?}"
                );
            }
        }
        // Same weights, same inputs, same function → same penalty, and a
        // non-trivial one: the un-saturated tanh has curvature here. Weights
        // of order 1 so that z ∈ [−2, 2] lands in tanh's curved band (the tiny
        // `det_weights` keep the net nearly affine, which is the *other* way to
        // read zero curvature).
        let w: Vec<f64> = det_weights(normed.mlp().n_weights())
            .into_iter()
            .map(|v| 5.0 * v)
            .collect();
        let p_normed = normed.mlp().curvature_penalty_value(&w, &g_normed, 1.0);
        let p_plain = plain.mlp().curvature_penalty_value(&w, &g_plain, 1.0);
        assert_relative_eq!(p_normed, p_plain, max_relative = 1e-12);
        assert!(
            p_normed > 1e-6,
            "penalty must be live in the responsive band: {p_normed}"
        );
        // …whereas the old raw-space grid, fed to the same network, sits in the
        // saturated tail: orders of magnitude smaller. This is the failure the
        // fix removes.
        let raw_vals: Vec<Vec<f64>> = (0..2)
            .map(|i| {
                raw.iter()
                    .map(|&(w, c)| if i == 0 { w } else { c })
                    .collect()
            })
            .collect();
        let g_raw = CurvatureGrid::build(&raw_vals, NN_SMOOTH_GRID_STEP_Z);
        let p_raw = normed.mlp().curvature_penalty_value(&w, &g_raw, 1.0);
        assert!(
            p_raw < 1e-3 * p_normed,
            "raw-space grid must be (nearly) blind: raw {p_raw} vs normalised {p_normed}"
        );
    }

    /// One-pass kernel: value with a gradient sink must equal the value-only
    /// call, and the Gauss–Newton Hessian sink must match central FD of the
    /// gradient on the affine-in-weights (last-layer) coordinates exactly and
    /// the full block to GN tolerance.
    #[test]
    fn curvature_penalty_hessian_matches_fd_of_gradient() {
        let mlp = MlpMapper::new(vec![2, 3, 2], Activation::Tanh, Activation::Identity).unwrap();
        let w = det_weights(mlp.n_weights());
        let vals = vec![
            vec![-1.5, -0.8, -0.2, 0.3, 0.9, 1.6],
            vec![-1.2, -0.5, 0.0, 0.4, 1.1, 1.7],
        ];
        let grid = CurvatureGrid::build(&vals, NN_SMOOTH_GRID_STEP_Z);
        assert!(!grid.is_empty());
        let lambda = 0.7;
        let n = mlp.n_weights();

        let mut grad = vec![0.0; n];
        let mut hess = vec![vec![0.0; n]; n];
        let v = {
            let mut add = |i: usize, j: usize, x: f64| hess[i][j] += x;
            let mut sink = PenaltySink {
                grad: Some(&mut grad),
                hess: Some(&mut add),
            };
            mlp.curvature_penalty_accum(&w, &grid, lambda, &mut sink)
        };
        assert_relative_eq!(
            v,
            mlp.curvature_penalty_value(&w, &grid, lambda),
            max_relative = 1e-12
        );
        let (_v2, g2) = mlp.curvature_penalty(&w, &grid, lambda);
        for (a, b) in grad.iter().zip(&g2) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
        // Symmetric PSD by construction.
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(hess[i][j], hess[j][i], epsilon = 1e-12);
            }
            assert!(hess[i][i] >= 0.0);
        }
        // Against FD of the gradient. On the output layer (identity head, so
        // the output is affine in its weights) `∂²C/∂w²` vanishes and GN is
        // exact; elsewhere GN drops `C·∂²C/∂w²`, so compare to a loose bound.
        let eps = 1e-5;
        let out_start = mlp.offsets[mlp.layers.len() - 2];
        let mut p = w.clone();
        for k in 0..n {
            let saved = p[k];
            p[k] = saved + eps;
            let (_, gp) = mlp.curvature_penalty(&p, &grid, lambda);
            p[k] = saved - eps;
            let (_, gm) = mlp.curvature_penalty(&p, &grid, lambda);
            p[k] = saved;
            for j in 0..n {
                let fd = (gp[j] - gm[j]) / (2.0 * eps);
                if k >= out_start && j >= out_start {
                    assert_relative_eq!(hess[k][j], fd, epsilon = 1e-6, max_relative = 1e-5);
                } else {
                    let scale = fd.abs().max(hess[k][j].abs()).max(1e-3);
                    assert!(
                        (hess[k][j] - fd).abs() / scale < 0.5,
                        "({k},{j}): GN {} vs FD {}",
                        hess[k][j],
                        fd
                    );
                }
            }
        }
    }
}

/// Fit-driven tests for the covariate-NN regularizers.
///
/// These live in `src/` rather than `tests/` because they reach for
/// [`NnRegularizer`] and [`MlpMapper::weight_param_indices`], which are
/// `pub(crate)`: the estimation layer is their only caller, and the workspace
/// boundary rule (CLAUDE.md) says an item is either public API — documented, in
/// `api/ferx-core-public-api.txt`, reachable from ferx-r — or it stays
/// crate-private and the caller does without. A test is not a reason to widen
/// the surface. What genuinely exercises the *public* path (driving `nn_l2` /
/// `nn_smooth` through `fit()` the way ferx-r does) stays in
/// `tests/nn_regularization.rs`.
#[cfg(test)]
mod regularizer_fit_tests {
    use approx::assert_relative_eq;

    use crate::parser::model_parser::parse_full_model;
    use crate::read_nonmem_csv;
    use crate::types::{CompiledModel, FitOptions, Population};

    use super::{CovariateMapper, NnRegularizer};

    /// Two-cpt oral DCM with a small MLP mapping (WT, CRCL) → the five PK params.
    /// One file shared with `tests/nn_regularization.rs` and the trust-region
    /// end-to-end check (`include_str!` is visible from both `tests/` and
    /// `#[cfg(test)]` code), so the fixtures cannot drift. The notes in it on why
    /// `center` / `scale` are load-bearing apply to the λ = 0 baseline of
    /// [`l2_shrinks_weights_and_modulator_variation`] in particular.
    const MODEL: &str = include_str!("../../tests/fixtures/two_cpt_dcm_regularized.ferx");

    fn load() -> (CompiledModel, FitOptions, Population) {
        let parsed = parse_full_model(MODEL).expect("model parses with --features nn");
        let population = read_nonmem_csv(
            std::path::Path::new("data/two_cpt_oral_cov.csv"),
            Some(&["WT", "CRCL"]),
            None,
        )
        .expect("dataset loads");
        (parsed.model, parsed.fit_options, population)
    }

    /// Variance of the NN's CL modulator (output 0) across subjects, evaluated at
    /// a given theta. Higher = more covariate-driven spread in the learned curve.
    #[cfg(feature = "slow-tests")]
    fn cl_modulator_variance(model: &CompiledModel, population: &Population, theta: &[f64]) -> f64 {
        let nn = &model.covariate_nns[0];
        let n_w = nn.mapper.n_weights();
        let w = &theta[nn.weights_offset..nn.weights_offset + n_w];
        let cls: Vec<f64> = population
            .subjects
            .iter()
            .map(|s| nn.mapper.forward_raw(w, &s.covariates).expect("forward")[0])
            .collect();
        let n = cls.len() as f64;
        let mean = cls.iter().sum::<f64>() / n;
        cls.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / n
    }

    /// L2 norm of the NN's *weight matrices* (biases excluded) at a given theta —
    /// the quantity `nn_l2` directly shrinks.
    #[cfg(feature = "slow-tests")]
    fn weight_block_sq_norm(model: &CompiledModel, theta: &[f64]) -> f64 {
        let nn = &model.covariate_nns[0];
        let w = &theta[nn.weights_offset..nn.weights_offset + nn.mapper.n_weights()];
        nn.mapper
            .mlp()
            .weight_param_indices()
            .iter()
            .map(|&i| w[i] * w[i])
            .sum()
    }

    /// λ = 0 must leave the regularizer inactive, its penalty exactly 0 and its
    /// gradient contribution exactly 0 — the no-op guarantee that keeps
    /// unregularized fits byte-identical.
    #[test]
    fn regularizer_lambda_zero_is_noop() {
        let (model, mut options, population) = load();
        options.nn_l2_lambda = 0.0;
        options.nn_smooth_lambda = 0.0;
        let reg = NnRegularizer::build(&model, &population, &options);
        assert!(!reg.is_active(), "λ = 0 regularizer must be inactive");

        let theta = model.default_params.theta.clone();
        assert_eq!(reg.penalty_value(&theta), 0.0, "λ = 0 penalty must be 0");

        let mut grad = vec![0.0; theta.len()];
        reg.add_packed_gradient(&theta, &mut grad);
        assert!(
            grad.iter().all(|&g| g == 0.0),
            "λ = 0 gradient contribution must be exactly 0"
        );
    }

    /// The analytic penalty gradient must match central finite differences of
    /// `penalty_value` at λ > 0, and must touch only the NN-weight coordinates.
    #[test]
    fn regularizer_penalty_gradient_matches_fd() {
        let (model, mut options, population) = load();
        options.nn_l2_lambda = 1e-2;
        options.nn_smooth_lambda = 1e-1;
        let reg = NnRegularizer::build(&model, &population, &options);
        assert!(reg.is_active());

        let theta = model.default_params.theta.clone();
        let mut grad = vec![0.0; theta.len()];
        reg.add_packed_gradient(&theta, &mut grad);

        // Only the NN-weight coordinates carry a penalty gradient; everything else
        // (omegas, sigma) must be untouched.
        let nn = &model.covariate_nns[0];
        let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
        for (k, &g) in grad.iter().enumerate() {
            if k < w_lo || k >= w_hi {
                assert_eq!(g, 0.0, "non-NN coordinate {k} must have zero penalty grad");
            }
        }

        // Central FD of penalty_value w.r.t. each NN weight theta.
        let eps = 1e-6;
        let mut p = theta.clone();
        for k in w_lo..w_hi {
            let saved = p[k];
            p[k] = saved + eps;
            let vp = reg.penalty_value(&p);
            p[k] = saved - eps;
            let vm = reg.penalty_value(&p);
            p[k] = saved;
            let fd = (vp - vm) / (2.0 * eps);
            let tol = 1e-6 + 1e-5 * fd.abs();
            assert!(
                (grad[k] - fd).abs() <= tol,
                "penalty grad mismatch at weight {k}: analytic {}, fd {}",
                grad[k],
                fd
            );
        }
    }

    /// The single-pass `penalty_and_gradient` the optimizers call must equal
    /// `penalty_value` + `add_packed_gradient` exactly, and the packed Hessian
    /// adder must touch NN-weight coordinates only (with the exact `2λ` L2
    /// diagonal on weight entries).
    #[test]
    fn single_pass_and_hessian_agree_with_the_separate_calls() {
        let (model, mut options, population) = load();
        options.nn_l2_lambda = 2e-2;
        options.nn_smooth_lambda = 3e-1;
        let reg = NnRegularizer::build(&model, &population, &options);
        let theta = model.default_params.theta.clone();
        let n = theta.len();

        let mut g_sep = vec![0.0; n];
        reg.add_packed_gradient(&theta, &mut g_sep);
        let v_sep = reg.penalty_value(&theta);
        let mut g_one = vec![0.0; n];
        let v_one = reg.penalty_and_gradient(&theta, &mut g_one);
        assert_relative_eq!(v_one, v_sep, max_relative = 1e-12);
        for (a, b) in g_one.iter().zip(&g_sep) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }

        let nn = &model.covariate_nns[0];
        let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
        let mut h = vec![vec![0.0; n]; n];
        reg.add_packed_hessian(&theta, &mut |i, j, v| h[i][j] += v);
        for i in 0..n {
            for j in 0..n {
                let in_block = (w_lo..w_hi).contains(&i) && (w_lo..w_hi).contains(&j);
                if !in_block {
                    assert_eq!(h[i][j], 0.0, "penalty Hessian leaked to ({i}, {j})");
                }
                assert_relative_eq!(h[i][j], h[j][i], epsilon = 1e-12);
            }
        }
        // L2 alone: exact diagonal on weight entries, zero on biases.
        let mut l2 = options.clone();
        l2.nn_smooth_lambda = 0.0;
        let reg_l2 = NnRegularizer::build(&model, &population, &l2);
        let mut h = vec![vec![0.0; n]; n];
        reg_l2.add_packed_hessian(&theta, &mut |i, j, v| h[i][j] += v);
        let weight_idx: std::collections::HashSet<usize> = nn
            .mapper
            .mlp()
            .weight_param_indices()
            .into_iter()
            .map(|i| w_lo + i)
            .collect();
        for i in 0..n {
            for j in 0..n {
                let expect = if i == j && weight_idx.contains(&i) {
                    2.0 * 2e-2
                } else {
                    0.0
                };
                assert_relative_eq!(h[i][j], expect, epsilon = 1e-15);
            }
        }
    }

    /// Growing the L2 strength shrinks the NN weights, flattening the learned
    /// covariate→modulator curve so the across-subject modulator variance
    /// collapses toward 0.
    ///
    /// The robust, deterministic signal is the **fitted weight-block norm**: L2
    /// adds `2λw` to the weight gradient, so a heavier λ pulls the optimum's
    /// weights closer to 0 (monotonically).
    ///
    /// The modulator variance is the *effect* being claimed, and it is only a
    /// meaningful check when the λ = 0 fit actually produces spread to remove. On
    /// this null-covariate dataset the unregularized fit invents a large spurious
    /// CL modulator variance (~480 across subjects) — precisely the overfitting
    /// `nn_l2` exists to suppress — and L2 collapses it to ~0. Asserting both ends
    /// keeps the oracle non-degenerate in the sense CLAUDE.md requires: an
    /// assertion that the regularized modulator is flat is worthless if the
    /// unregularized one was flat too. It was, on raw inputs, because the network
    /// was saturated rather than because it had learned nothing — see the
    /// `center` / `scale` note on `MODEL`.
    /// Gated with `#[cfg]` rather than the usual Tier-3
    /// `#[cfg_attr(not(feature = "slow-tests"), ignore = "…")]`, because this
    /// test lives in `src/` — which codecov measures — instead of `tests/`,
    /// which `codecov.yml` ignores wholesale. An `#[ignore]`d body is compiled
    /// but never run, so on a PR (where slow-tests never run) its ~50 lines
    /// would read as uncovered and drag the patch gate under its 90% floor.
    /// Compiling it out instead makes it a measurement gap rather than a miss,
    /// the same way the rest of the feature-gated surface behaves (#293).
    ///
    /// Bit-rot is caught on every PR by `cargo check --tests --no-default-features
    /// --features ci,nn,slow-tests` (preflight check matrix and the CI `Check`
    /// job), and the body actually *runs* nightly: `slow-tests.yml` passes
    /// `ferx-core/nn` alongside `ferx-core/slow-tests` for exactly this
    /// `nn` + `slow-tests` family (it once passed no `nn`, which left every DCM
    /// Tier-3 fit — this one included — running in no CI job at all).
    #[test]
    #[cfg(feature = "slow-tests")]
    fn l2_shrinks_weights_and_modulator_variation() {
        let (model, options, population) = load();

        let fit_at = |lambda: f64| -> Vec<f64> {
            let mut o = options.clone();
            o.nn_l2_lambda = lambda;
            crate::fit(&model, &population, &model.default_params, &o)
                .unwrap_or_else(|e| panic!("fit at λ={lambda} failed: {e}"))
                .theta
        };

        let t0 = fit_at(0.0);
        let t_mid = fit_at(5.0);
        let t_big = fit_at(100.0);

        let (n0, n_mid, n_big) = (
            weight_block_sq_norm(&model, &t0),
            weight_block_sq_norm(&model, &t_mid),
            weight_block_sq_norm(&model, &t_big),
        );
        let (v0, v_mid, v_big) = (
            cl_modulator_variance(&model, &population, &t0),
            cl_modulator_variance(&model, &population, &t_mid),
            cl_modulator_variance(&model, &population, &t_big),
        );
        eprintln!(
            "weight ‖W‖²: λ=0 {n0:.5}, λ=5 {n_mid:.5}, λ=100 {n_big:.5}\n\
             CL modulator var: λ=0 {v0:.6}, λ=5 {v_mid:.6}, λ=100 {v_big:.6}"
        );

        // Decisive signal: the fitted weight norm shrinks strongly and
        // monotonically with λ (observed here ~2159 → ~0.004 → ~0.003). This is
        // the guaranteed mechanism by which L2 flattens the covariate→modulator
        // map.
        assert!(
            n_mid <= n0 + 1e-9 && n_big <= n_mid + 1e-9,
            "weight norm must be non-increasing in λ (‖W‖²: {n0:.5} → {n_mid:.5} → {n_big:.5})"
        );
        assert!(
            n_big < n0 * 0.5,
            "heavy L2 (λ=100) must more than halve the fitted weight norm \
             ({n_big:.5} vs λ=0 {n0:.5})"
        );

        // The unregularized fit must actually overfit — otherwise the flatness
        // check below passes against a baseline that was already flat and proves
        // nothing.
        assert!(
            v0 > 1.0,
            "the λ=0 fit must invent real spurious CL spread for this test to have \
             a baseline to remove (var {v0:.6}); a near-zero unregularized variance \
             means the fixture is degenerate, not that L2 worked"
        );

        // The effect: L2 collapses that spurious spread toward a constant map.
        // Both regularized fits must be flat; their ordering *relative to each
        // other* is not asserted, because at ~1e-9 the difference between them is
        // float noise rather than an effect of λ.
        assert!(
            v_mid < 1e-3 && v_big < 1e-3,
            "L2 must collapse the spurious CL modulator spread \
             ({v0:.6} → {v_mid:.6} → {v_big:.6})"
        );
    }
    /// Each λ must act alone. `nn_l2` on its own is the likeliest real
    /// configuration, and it must not drag the curvature machinery in with it:
    /// with `nn_smooth = 0` the regularizer skips building a grid entirely, and
    /// the penalty and gradient must equal the pure-L2 ones. Symmetrically for
    /// `nn_smooth` alone.
    ///
    /// Without this, every test drove both λ together or both at zero, so the
    /// single-λ branches in `penalty_value` / `add_packed_gradient` — and the
    /// `CurvatureGrid::default()` arm of `build` — were never taken.
    #[test]
    fn each_lambda_acts_alone() {
        let (model, options, population) = load();
        let theta = model.default_params.theta.clone();
        let nn = &model.covariate_nns[0];
        let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
        let w = &theta[w_lo..w_hi];
        let mlp = nn.mapper.mlp();

        let build = |l2: f64, smooth: f64| {
            let mut o = options.clone();
            o.nn_l2_lambda = l2;
            o.nn_smooth_lambda = smooth;
            NnRegularizer::build(&model, &population, &o)
        };
        let grad_of = |reg: &NnRegularizer| {
            let mut g = vec![0.0; theta.len()];
            reg.add_packed_gradient(&theta, &mut g);
            g
        };

        // --- L2 only: no grid is built, and the penalty is exactly the L2 one.
        let l2_only = build(1e-2, 0.0);
        assert!(l2_only.is_active());
        let expected_l2 = mlp.l2_weight_penalty_value(w, 1e-2);
        assert_relative_eq!(
            l2_only.penalty_value(&theta),
            expected_l2,
            max_relative = 1e-12
        );
        let (_v, expected_l2_g) = mlp.l2_weight_penalty(w, 1e-2);
        let g_l2 = grad_of(&l2_only);
        for (i, e) in expected_l2_g.iter().enumerate() {
            assert_relative_eq!(g_l2[w_lo + i], e, max_relative = 1e-12);
        }

        // --- Smoothness only: the penalty carries no L2 term at all.
        let smooth_only = build(0.0, 1e-1);
        assert!(smooth_only.is_active());
        let p_smooth = smooth_only.penalty_value(&theta);
        assert!(
            p_smooth > 0.0,
            "the fixture's network must have some curvature to penalize"
        );
        let g_smooth = grad_of(&smooth_only);

        // --- Both together must equal the sum of the two acting alone: the
        // terms are additive and independent, which is what lets a caller reason
        // about one λ at a time.
        let both = build(1e-2, 1e-1);
        assert_relative_eq!(
            both.penalty_value(&theta),
            expected_l2 + p_smooth,
            max_relative = 1e-12
        );
        let g_both = grad_of(&both);
        for k in w_lo..w_hi {
            assert_relative_eq!(g_both[k], g_l2[k] + g_smooth[k], max_relative = 1e-9);
        }
    }
}
