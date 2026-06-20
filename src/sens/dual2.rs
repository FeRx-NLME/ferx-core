//! `Dual2<N>` — a small, explicit forward-mode second-order multivariate dual
//! number for analytic PK-parameter sensitivities.
//!
//! This is **not** Enzyme / compile-time AD: it is a plain value-plus-jet struct
//! (value, gradient, Hessian over `N` seeded variables) with hand-written
//! operator rules. Evaluating a closed-form PK solution generic over `Dual2<N>`
//! yields `f`, `∂f/∂pkᵢ`, and `∂²f/∂pkᵢ∂pkⱼ` exactly, in one pass, reusing the
//! solution code instead of hand-deriving each model. It is the
//! [`crate::sens`] counterpart to the hand-written closed-form derivatives, and
//! the natural seed of the future num-dual path.
//!
//! `N` is the number of differentiated inputs (PK parameters), a small const
//! (≤ 8: CL, V/V1, Q2, V2, KA, F, Q3, V3). The Hessian is symmetric; rules
//! maintain that.
// Indexed loops over the `[f64; N]` grad / `[[f64; N]; N]` hess jets read more
// clearly than zipped iterators here, and `Div` is defined as `× recip` (the
// `suspicious_arithmetic_impl` lint is a false positive for that identity).
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual2<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
    pub hess: [[f64; N]; N],
}

impl<const N: usize> Dual2<N> {
    /// A constant (zero gradient and Hessian).
    pub fn constant(value: f64) -> Self {
        Dual2 {
            value,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
        }
    }

    /// Seed input variable `i`: `value` with `∂/∂xᵢ = 1`, all else zero.
    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Dual2 {
            value,
            grad,
            hess: [[0.0; N]; N],
        }
    }

    /// `exp(self)`. With `u = e^x`: `u' = u·x'`, `u'' = u·(x'' + x'⊗x')`.
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = v * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = v * (self.hess[i][j] + self.grad[i] * self.grad[j]);
            }
        }
        Dual2 {
            value: v,
            grad,
            hess,
        }
    }

    /// `sqrt(self)`. With `u = √x`: `u' = x'/(2u)`, `u'' = x''/(2u) − x'⊗x'/(4u³)`.
    pub fn sqrt(self) -> Self {
        // Non-positive argument: the `1/(2u)` and `1/(4u³)` factors are singular.
        // Callers feed `sqrt` non-negative arguments, but a discriminant can round
        // to exactly 0 (or slightly negative) at near-repeated eigenvalues during a
        // line-search step. Return a zero-derivative value so `inf`/`NaN` never
        // enters the chain. (The narrow positive-but-tiny band, where the
        // derivative is large-but-finite, is left to the callers' eigenvalue-gap
        // degeneracy guards.)
        if !(self.value > 0.0) {
            let v = if self.value > 0.0 {
                self.value.sqrt()
            } else {
                0.0
            };
            return Dual2 {
                value: v,
                grad: [0.0; N],
                hess: [[0.0; N]; N],
            };
        }
        let v = self.value.sqrt();
        let inv2u = 0.5 / v;
        let inv4u3 = 0.25 / (v * v * v);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = inv2u * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = inv2u * self.hess[i][j] - inv4u3 * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: v,
            grad,
            hess,
        }
    }

    /// `cos(self)`. With `u = cos(x)`: `u' = −sin(x)·x'`,
    /// `u'' = −cos(x)·x'⊗x' − sin(x)·x''`.
    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -s * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -s * self.hess[i][j] - c * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: c,
            grad,
            hess,
        }
    }

    /// `acos(self)`. With `u = acos(x)`: `u' = g₁·x'`, `u'' = g₁·x'' + g₂·x'⊗x'`,
    /// where `g₁ = −1/√(1−x²)` and `g₂ = −x·(1−x²)^{−3/2}`.
    ///
    /// Self-defending at the `|x|→1` boundary: the value is clamped to `[−1, 1]`
    /// (so `acos` is never NaN on a rounded/saturated arg) and `1−x²` is floored
    /// before the `√`/divide. Without the floor, `g₁ = −1/√0 = −inf` and, for a
    /// clamped-constant argument (`grad = 0`), `grad[i] = −inf·0 = NaN` — which
    /// would silently poison the whole subject's PK sensitivities (PR #381 review
    /// finding #3). The floored derivative stays finite; at a genuine saturation
    /// the second-order term is a bounded approximation rather than a NaN.
    pub fn acos(self) -> Self {
        let v = self.value;
        let v_clamped = if v > 1.0 {
            1.0
        } else if v < -1.0 {
            -1.0
        } else {
            v
        };
        let one_minus = {
            let x = 1.0 - v * v;
            if x > 1e-12 {
                x
            } else {
                1e-12
            }
        };
        let s = one_minus.sqrt();
        let g1 = -1.0 / s;
        let g2 = -v / (one_minus * s);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = g1 * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = g1 * self.hess[i][j] + g2 * self.grad[i] * self.grad[j];
            }
        }
        Dual2 {
            value: v_clamped.acos(),
            grad,
            hess,
        }
    }

    /// `ln(self)`. With `u = ln(x)`: `u' = x'/x`, `u'' = x''/x − x'⊗x'/x²`.
    pub fn ln(self) -> Self {
        let x = self.value;
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] * inv;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] * inv - self.grad[i] * self.grad[j] * inv2;
            }
        }
        Dual2 {
            value: x.ln(),
            grad,
            hess,
        }
    }

    /// `self.powd(e) = self^e`. When `e` is a constant (zero jet) the power rule
    /// `aⁿ` is used directly — exact for any base sign with integer `n`, matching
    /// `f64::powf`. Otherwise the general `exp(e·ln(self))` form is used (requires
    /// `self.value > 0`, as does the underlying `powf`).
    pub fn powd(self, e: Self) -> Self {
        let exp_const =
            e.grad.iter().all(|&g| g == 0.0) && e.hess.iter().flatten().all(|&h| h == 0.0);
        if exp_const {
            let n = e.value;
            let a = self.value;
            let an = a.powf(n);
            // Guard a == 0: derivatives of aⁿ are 0 (n>1) or singular; mirror the
            // value and drop the jet rather than emit inf/NaN.
            if a == 0.0 {
                return Dual2::constant(an);
            }
            let c1 = n * a.powf(n - 1.0);
            let c2 = n * (n - 1.0) * a.powf(n - 2.0);
            let mut grad = [0.0; N];
            let mut hess = [[0.0; N]; N];
            for i in 0..N {
                grad[i] = c1 * self.grad[i];
            }
            for i in 0..N {
                for j in 0..N {
                    hess[i][j] = c1 * self.hess[i][j] + c2 * self.grad[i] * self.grad[j];
                }
            }
            return Dual2 {
                value: an,
                grad,
                hess,
            };
        }
        (e * self.ln()).exp()
    }

    /// `|self|`. Away from the kink `x = 0` this is `±self`; the second derivative
    /// is `sign(x)·x''` (the cusp at 0 is measure-zero and ignored).
    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            -self
        }
    }

    /// `inv_logit(self) = 1/(1+e^{−x})` (logistic sigmoid), via `exp`/`recip`.
    pub fn inv_logit(self) -> Self {
        ((-self).exp() + 1.0).recip()
    }

    /// `logit(self) = ln(x/(1−x)) = ln(x) − ln(1−x)`, via `ln`.
    pub fn logit(self) -> Self {
        self.ln() - ((-self) + 1.0).ln()
    }

    /// `1/self`. With `u = 1/b`: `u' = −b'/b²`, `u'' = −b''/b² + 2·b'⊗b'/b³`.
    pub fn recip(self) -> Self {
        let b = self.value;
        let inv = 1.0 / b;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -self.grad[i] * inv2;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -self.hess[i][j] * inv2 + 2.0 * self.grad[i] * self.grad[j] * inv3;
            }
        }
        Dual2 {
            value: inv,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add for Dual2<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] + rhs.hess[i][j];
            }
        }
        Dual2 {
            value: self.value + rhs.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Sub for Dual2<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const N: usize> Neg for Dual2<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -self.hess[i][j];
            }
        }
        Dual2 {
            value: -self.value,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Mul for Dual2<N> {
    type Output = Self;
    /// Leibniz: `(ab)ᵢ = a·bᵢ + aᵢ·b`, `(ab)ᵢⱼ = a·bᵢⱼ + aᵢ·bⱼ + aⱼ·bᵢ + aᵢⱼ·b`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.value, rhs.value);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = a * rhs.grad[i] + self.grad[i] * b;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = a * rhs.hess[i][j]
                    + self.grad[i] * rhs.grad[j]
                    + self.grad[j] * rhs.grad[i]
                    + self.hess[i][j] * b;
            }
        }
        Dual2 {
            value: a * b,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Div for Dual2<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

// ── Scalar (f64) conveniences ────────────────────────────────────────────────

impl<const N: usize> Mul<f64> for Dual2<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] * s;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] * s;
            }
        }
        Dual2 {
            value: self.value * s,
            grad,
            hess,
        }
    }
}

impl<const N: usize> Add<f64> for Dual2<N> {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Dual2 {
            value: self.value + s,
            ..self
        }
    }
}

/// `s / dual` (scalar numerator).
pub fn scalar_div<const N: usize>(s: f64, d: Dual2<N>) -> Dual2<N> {
    d.recip() * s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sqrt` at a zero (or rounded-negative) argument with a seeded gradient must
    /// return finite, zero derivatives — not `inf`/`NaN` from the singular
    /// `1/(2√x)` / `1/(4x^{3/2})` factors.
    #[test]
    fn dual2_sqrt_zero_argument_is_finite() {
        let x = Dual2::<2>::var(0.0, 0); // value 0, grad[0] = 1
        let r = x.sqrt();
        assert_eq!(r.value, 0.0);
        assert!(r.grad.iter().all(|g| g.is_finite()));
        assert!(r.hess.iter().flatten().all(|h| h.is_finite()));
        assert!(r.grad.iter().all(|&g| g == 0.0));
        // A slightly-negative discriminant (rounding) must also stay finite.
        let neg = Dual2::<2>::var(-1e-18, 0).sqrt();
        assert!(neg.value.is_finite() && neg.grad.iter().all(|g| g.is_finite()));
    }

    /// `f(x,y) = (1/y)·exp(-x/y)` (the 1-cpt IV-bolus shape, amt=1) at a point,
    /// cross-checked against analytic derivatives.
    #[test]
    fn dual2_matches_hand_derivatives_on_iv_shape() {
        let (x, y) = (3.0_f64, 5.0_f64); // x≈CL, y≈V
        let xd = Dual2::<2>::var(x, 0);
        let yd = Dual2::<2>::var(y, 1);
        let f = scalar_div(1.0, yd) * (-(xd / yd)).exp();

        let k = x / y;
        let val = (1.0 / y) * (-k).exp();
        approx::assert_relative_eq!(f.value, val, max_relative = 1e-12);
        // ∂f/∂x = val·(-1/y); ∂f/∂y = val·(k-1)/y  (t=1 here)
        approx::assert_relative_eq!(f.grad[0], val * (-1.0 / y), max_relative = 1e-10);
        approx::assert_relative_eq!(f.grad[1], val * (k - 1.0) / y, max_relative = 1e-10);
        // ∂²f/∂x² = val/y²; symmetric Hessian.
        approx::assert_relative_eq!(f.hess[0][0], val / (y * y), max_relative = 1e-10);
        approx::assert_relative_eq!(f.hess[0][1], f.hess[1][0], max_relative = 1e-12);
    }

    /// Validate a `Dual2<2>` expression's grad/Hessian against central finite
    /// differences of its value, at `(x, y)`.
    fn fd_check<F>(f: F, x: f64, y: f64, gtol: f64, htol: f64)
    where
        F: Fn(Dual2<2>, Dual2<2>) -> Dual2<2>,
    {
        let d = f(Dual2::var(x, 0), Dual2::var(y, 1));
        let v = |a: f64, b: f64| f(Dual2::var(a, 0), Dual2::var(b, 1)).value;
        let h = 1e-5;
        let gx = (v(x + h, y) - v(x - h, y)) / (2.0 * h);
        let gy = (v(x, y + h) - v(x, y - h)) / (2.0 * h);
        approx::assert_relative_eq!(d.grad[0], gx, max_relative = gtol, epsilon = 1e-8);
        approx::assert_relative_eq!(d.grad[1], gy, max_relative = gtol, epsilon = 1e-8);

        let hh = 1e-4;
        let hxx = (v(x + hh, y) - 2.0 * v(x, y) + v(x - hh, y)) / (hh * hh);
        let hyy = (v(x, y + hh) - 2.0 * v(x, y) + v(x, y - hh)) / (hh * hh);
        let hxy = (v(x + hh, y + hh) - v(x + hh, y - hh) - v(x - hh, y + hh) + v(x - hh, y - hh))
            / (4.0 * hh * hh);
        approx::assert_relative_eq!(d.hess[0][0], hxx, max_relative = htol, epsilon = 1e-5);
        approx::assert_relative_eq!(d.hess[1][1], hyy, max_relative = htol, epsilon = 1e-5);
        approx::assert_relative_eq!(d.hess[0][1], hxy, max_relative = htol, epsilon = 1e-5);
        // Hessian symmetry is a structural invariant.
        approx::assert_relative_eq!(d.hess[0][1], d.hess[1][0], max_relative = 1e-12);
    }

    #[test]
    fn ln_matches_fd() {
        fd_check(|x, y| (x * y).ln(), 2.0, 3.0, 1e-6, 1e-3);
        fd_check(|x, y| (x + y * y).ln(), 1.5, 0.7, 1e-6, 1e-3);
    }

    #[test]
    fn pow_constant_exponent_matches_fd() {
        // aⁿ power rule, including a non-integer exponent (positive base).
        fd_check(
            |x, y| x.powd(Dual2::constant(2.5)) * y,
            1.7,
            0.9,
            1e-6,
            1e-3,
        );
        // Integer exponent works for a negative base.
        fd_check(
            |x, y| (x - y).powd(Dual2::constant(3.0)),
            1.0,
            2.5,
            1e-6,
            1e-3,
        );
    }

    #[test]
    fn pow_variable_exponent_matches_fd() {
        // General aᵇ = exp(b·ln a) form (both vary; base > 0).
        fd_check(|x, y| x.powd(y), 1.7, 0.8, 1e-6, 1e-3);
        fd_check(|x, y| (x * y).powd(x), 1.3, 1.1, 1e-6, 2e-3);
    }

    #[test]
    fn abs_matches_fd_away_from_kink() {
        // Positive branch: |x²−y| = x²−y.
        fd_check(|x, y| (x * x - y).abs(), 2.0, 1.0, 1e-6, 1e-3);
        // Negative branch: |x²−y| = −(x²−y).
        fd_check(|x, y| (x * x - y).abs(), 1.0, 3.0, 1e-6, 1e-3);
    }

    #[test]
    fn inv_logit_matches_fd() {
        fd_check(|x, y| (x - y).inv_logit(), 0.6, 0.2, 1e-6, 1e-3);
        fd_check(|x, y| (x * y).inv_logit(), 1.2, 0.5, 1e-6, 1e-3);
    }

    #[test]
    fn logit_matches_fd() {
        // Argument must lie in (0, 1).
        fd_check(|x, y| (x * y).logit(), 0.3, 0.9, 1e-6, 1e-3);
        fd_check(|x, _y| x.logit(), 0.42, 0.0, 1e-6, 1e-3);
    }

    /// `logit` and `inv_logit` are inverses: `inv_logit(logit(p)) = p`, with the
    /// jet propagating consistently.
    #[test]
    fn logit_inv_logit_roundtrip() {
        let p = Dual2::<2>::var(0.37, 0);
        let r = p.logit().inv_logit();
        approx::assert_relative_eq!(r.value, 0.37, max_relative = 1e-12);
        approx::assert_relative_eq!(r.grad[0], 1.0, max_relative = 1e-9);
        approx::assert_relative_eq!(r.hess[0][0], 0.0, epsilon = 1e-7);
    }
}
