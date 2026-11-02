//! The element-wise activation functions of a `[covariate_nn]` network.
//!
//! Compiled in **every** build, not only under the `nn` feature: the `.ferx`
//! expression builtins `tanh(x)` and `softplus(x)` evaluate through
//! [`Activation::apply`] / `Activation::apply_g` (one implementation, so a fitted
//! network and its plain-equation export (#1331) cannot drift at softplus's
//! thresholds), and the exporter in [`super::export`] runs in a build without
//! `nn` — the build ferx-r ships.

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

    /// [`Self::apply`] on any number type: the same branches on the value, so the value
    /// is bit-identical and the derivatives are those of the branch taken.
    pub(crate) fn apply_g<T: crate::sens::num::PkNum>(self, x: T) -> T {
        let v = x.val();
        let one = T::from_f64(1.0);
        match self {
            Activation::Identity => x,
            Activation::Relu => {
                if v > 0.0 {
                    x
                } else {
                    T::from_f64(0.0)
                }
            }
            Activation::Softplus => {
                if v > 20.0 {
                    x
                } else if v < -20.0 {
                    x.exp()
                } else {
                    (one + x.exp()).ln()
                }
            }
            Activation::Tanh => {
                let t = v.tanh();
                let d1 = 1.0 - t * t;
                // tanh⁽ⁿ⁾ through n = 4: the covariance step's nested dual reads all four.
                x.lift4(
                    t,
                    d1,
                    -2.0 * t * d1,
                    -2.0 * d1 * (1.0 - 3.0 * t * t),
                    8.0 * t * d1 * (2.0 - 3.0 * t * t),
                )
            }
            Activation::Sigmoid => {
                if v >= 0.0 {
                    one / (one + (-x).exp())
                } else {
                    let ex = x.exp();
                    ex / (one + ex)
                }
            }
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
///
/// The one copy: `categorical`'s `inv_logit` delegates here, now that this module is
/// compiled in every build.
#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}
