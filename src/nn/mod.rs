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

use crate::types::PkParams;

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
    /// [`crate::estimation::fixed_eta_gradient`] to get `∂NLL/∂w` for every NN
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
        self.check_shapes(x, weights)?;
        let (pre, post) = self.forward_cache(x, weights);

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

        Ok(jac)
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
        if center.iter().any(|c| !c.is_finite()) {
            return Err(NnError::InvalidScale {
                input: "center".to_string(),
                value: f64::NAN,
            });
        }
        self.input_center = center;
        self.input_scale = scale;
        Ok(self)
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
            // `exp(y) - 1` would lose precision to cancellation.
            (
                Activation::Softplus,
                &[1e-8, 1e-3, 0.693_147, 1.0, 10.0, 25.0, 400.0],
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
