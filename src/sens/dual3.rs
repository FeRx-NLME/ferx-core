//! `Dual3<N>` — a small, explicit forward-mode **third-order** multivariate dual
//! number, the third-derivative companion to [`Dual2`](super::dual2::Dual2).
//!
//! It carries `value`, `grad` (`∂/∂xᵢ`), `hess` (`∂²/∂xᵢ∂xⱼ`) **and**
//! `d3` (`∂³/∂xᵢ∂xⱼ∂xₖ`) over `N` seeded variables, with hand-written operator
//! rules. Evaluating a closed-form PK solution generic over `Dual3<N>` yields the
//! exact third-order sensitivities in one pass, reusing the same `*_g<T: PkNum>`
//! source as the value (`f64`) and second-order (`Dual2`) instantiations — no
//! formula is duplicated.
//!
//! Third order is needed **only** by the analytic FOCE/FOCEI covariance Hessian
//! (issue #436): the `log|H̃|` curvature differentiates the inner Hessian through
//! the moving mode a second time (`∂³f/∂η³`, `∂³f/∂η²∂θ`). It is therefore a
//! covariance-step-only type — never instantiated in the optimization hot loop,
//! which stays on `Dual1` (inner) / `Dual2` (outer).
//!
//! Every unary rule is the third-order **Faà di Bruno** expansion of `u = φ(x)`:
//! `uᵢ = φ'·xᵢ`, `uᵢⱼ = φ'·xᵢⱼ + φ''·xᵢxⱼ`, and
//! `uᵢⱼₖ = φ'·xᵢⱼₖ + φ''·(xᵢⱼxₖ + xᵢₖxⱼ + xⱼₖxᵢ) + φ'''·xᵢxⱼxₖ`.
//! The shared [`Dual3::unary`] helper applies it given `(φ, φ', φ'', φ''')`, so a
//! new transcendental only supplies its three derivative scalars.
//!
//! `N` is the number of differentiated inputs (PK parameters), a small const
//! (≤ 8: CL, V/V1, Q2, V2, KA, F, Q3, V3). All derivative tensors are symmetric;
//! the rules maintain that.
// `!(x > 0.0)` in `sqrt` is the deliberate "not a positive number" guard (it also
// catches NaN); `needless_range_loop` / `suspicious_arithmetic_impl` are noise for
// the indexed jet loops and the `Div = × recip` identity.
#![allow(
    clippy::needless_range_loop,
    clippy::suspicious_arithmetic_impl,
    clippy::neg_cmp_op_on_partial_ord
)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual3<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
    pub hess: [[f64; N]; N],
    pub d3: [[[f64; N]; N]; N],
}

impl<const N: usize> Dual3<N> {
    /// A constant (zero gradient, Hessian and third derivative).
    pub fn constant(value: f64) -> Self {
        Dual3 {
            value,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
            d3: [[[0.0; N]; N]; N],
        }
    }

    /// Seed input variable `i`: `value` with `∂/∂xᵢ = 1`, all higher orders zero.
    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Dual3 {
            value,
            grad,
            hess: [[0.0; N]; N],
            d3: [[[0.0; N]; N]; N],
        }
    }

    /// Apply a scalar analytic function `u = φ(x)` to this jet given its value and
    /// first three derivatives at `x` (`v = φ(x)`, `d1 = φ'`, `d2 = φ''`,
    /// `d3v = φ'''`), via the third-order Faà di Bruno chain rule. The single point
    /// of truth for every unary transcendental below.
    #[inline]
    fn unary(self, v: f64, d1: f64, d2: f64, d3v: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut d3 = [[[0.0; N]; N]; N];
        for i in 0..N {
            grad[i] = d1 * self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = d1 * self.hess[i][j] + d2 * self.grad[i] * self.grad[j];
            }
        }
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    d3[i][j][k] = d1 * self.d3[i][j][k]
                        + d2 * (self.hess[i][j] * self.grad[k]
                            + self.hess[i][k] * self.grad[j]
                            + self.hess[j][k] * self.grad[i])
                        + d3v * self.grad[i] * self.grad[j] * self.grad[k];
                }
            }
        }
        Dual3 {
            value: v,
            grad,
            hess,
            d3,
        }
    }

    /// `exp(self)`. `φ = φ' = φ'' = φ''' = eˣ`.
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        self.unary(v, v, v, v)
    }

    /// `sqrt(self)`. `φ' = 1/(2√x)`, `φ'' = −1/(4x^{3/2})`, `φ''' = 3/(8x^{5/2})`.
    ///
    /// Mirrors [`Dual2::sqrt`]: a non-positive argument (a discriminant rounding to
    /// 0 / slightly negative at near-repeated eigenvalues) returns a finite,
    /// zero-derivative value rather than letting the singular factors emit
    /// `inf`/`NaN`.
    pub fn sqrt(self) -> Self {
        if !(self.value > 0.0) {
            return Dual3::constant(if self.value > 0.0 {
                self.value.sqrt()
            } else {
                0.0
            });
        }
        let v = self.value.sqrt();
        let inv = 1.0 / v;
        let d1 = 0.5 * inv;
        let d2 = -0.25 * inv * inv * inv;
        let d3v = 0.375 * inv * inv * inv * inv * inv;
        self.unary(v, d1, d2, d3v)
    }

    /// `cos(self)`. `φ' = −sin`, `φ'' = −cos`, `φ''' = sin`.
    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        self.unary(c, -s, -c, s)
    }

    /// `acos(self)`. With `m = 1−x²`: `φ' = −m^{−1/2}`, `φ'' = −x·m^{−3/2}`,
    /// `φ''' = −(1+2x²)·m^{−5/2}`.
    ///
    /// Mirrors [`Dual2::acos`]'s `|x|→1` self-defence: the value is clamped to
    /// `[−1, 1]` and `m` is floored before the `√`/divide so the derivatives stay
    /// finite at saturation (PR #381 review finding #3).
    pub fn acos(self) -> Self {
        let v = self.value;
        let v_clamped = v.clamp(-1.0, 1.0);
        let one_minus = {
            let x = 1.0 - v * v;
            if x > 1e-12 {
                x
            } else {
                1e-12
            }
        };
        let s = one_minus.sqrt();
        let d1 = -1.0 / s;
        let d2 = -v / (one_minus * s);
        let d3v = -(1.0 + 2.0 * v * v) / (one_minus * one_minus * s);
        self.unary(v_clamped.acos(), d1, d2, d3v)
    }

    /// `ln(self)`. `φ' = 1/x`, `φ'' = −1/x²`, `φ''' = 2/x³`.
    pub fn ln(self) -> Self {
        let x = self.value;
        let inv = 1.0 / x;
        self.unary(x.ln(), inv, -inv * inv, 2.0 * inv * inv * inv)
    }

    /// `self.powd(e) = self^e`. Constant exponent (zero jet) uses the power rule
    /// `aⁿ` directly (exact for any base sign with integer `n`); otherwise the
    /// general `exp(e·ln(self))` form (requires `self.value > 0`).
    pub fn powd(self, e: Self) -> Self {
        let exp_const = e.grad.iter().all(|&g| g == 0.0)
            && e.hess.iter().flatten().all(|&h| h == 0.0)
            && e.d3.iter().flatten().flatten().all(|&t| t == 0.0);
        if exp_const {
            let n = e.value;
            let a = self.value;
            let an = a.powf(n);
            // a == 0: derivatives are 0 (n>2) or singular; mirror value, drop jet.
            if a == 0.0 {
                return Dual3::constant(an);
            }
            let c1 = n * a.powf(n - 1.0);
            let c2 = n * (n - 1.0) * a.powf(n - 2.0);
            let c3 = n * (n - 1.0) * (n - 2.0) * a.powf(n - 3.0);
            return self.unary(an, c1, c2, c3);
        }
        (e * self.ln()).exp()
    }

    /// `|self|`. Away from the kink `x = 0` this is `±self` (the cusp at 0 is
    /// measure-zero and ignored), so every order carries `sign(x)`.
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

    /// `1/self`. `φ' = −1/x²`, `φ'' = 2/x³`, `φ''' = −6/x⁴`.
    pub fn recip(self) -> Self {
        let inv = 1.0 / self.value;
        let inv2 = inv * inv;
        self.unary(inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2)
    }
}

impl<const N: usize> Add for Dual3<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut d3 = [[[0.0; N]; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] + rhs.hess[i][j];
            }
        }
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    d3[i][j][k] = self.d3[i][j][k] + rhs.d3[i][j][k];
                }
            }
        }
        Dual3 {
            value: self.value + rhs.value,
            grad,
            hess,
            d3,
        }
    }
}

impl<const N: usize> Sub for Dual3<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const N: usize> Neg for Dual3<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut d3 = [[[0.0; N]; N]; N];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = -self.hess[i][j];
            }
        }
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    d3[i][j][k] = -self.d3[i][j][k];
                }
            }
        }
        Dual3 {
            value: -self.value,
            grad,
            hess,
            d3,
        }
    }
}

impl<const N: usize> Mul for Dual3<N> {
    type Output = Self;
    /// Leibniz, third order: distribute the three derivative indices between the
    /// two factors (8 terms) —
    /// `(ab)ᵢⱼₖ = a·bᵢⱼₖ + b·aᵢⱼₖ`
    /// `        + aᵢ·bⱼₖ + aⱼ·bᵢₖ + aₖ·bᵢⱼ + bᵢ·aⱼₖ + bⱼ·aᵢₖ + bₖ·aᵢⱼ`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.value, rhs.value);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut d3 = [[[0.0; N]; N]; N];
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
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    d3[i][j][k] = a * rhs.d3[i][j][k]
                        + b * self.d3[i][j][k]
                        + self.grad[i] * rhs.hess[j][k]
                        + self.grad[j] * rhs.hess[i][k]
                        + self.grad[k] * rhs.hess[i][j]
                        + rhs.grad[i] * self.hess[j][k]
                        + rhs.grad[j] * self.hess[i][k]
                        + rhs.grad[k] * self.hess[i][j];
                }
            }
        }
        Dual3 {
            value: a * b,
            grad,
            hess,
            d3,
        }
    }
}

impl<const N: usize> Div for Dual3<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

// ── Scalar (f64) conveniences ────────────────────────────────────────────────

impl<const N: usize> Mul<f64> for Dual3<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; N];
        let mut d3 = [[[0.0; N]; N]; N];
        for i in 0..N {
            grad[i] = self.grad[i] * s;
        }
        for i in 0..N {
            for j in 0..N {
                hess[i][j] = self.hess[i][j] * s;
            }
        }
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    d3[i][j][k] = self.d3[i][j][k] * s;
                }
            }
        }
        Dual3 {
            value: self.value * s,
            grad,
            hess,
            d3,
        }
    }
}

impl<const N: usize> Add<f64> for Dual3<N> {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Dual3 {
            value: self.value + s,
            ..self
        }
    }
}

/// `s / dual` (scalar numerator).
pub fn scalar_div<const N: usize>(s: f64, d: Dual3<N>) -> Dual3<N> {
    d.recip() * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;
    use crate::sens::num::PkNum;

    /// `sqrt` at a zero / rounded-negative argument with a seeded gradient must
    /// return finite, zero derivatives at every order (mirrors the `Dual2` guard).
    #[test]
    fn dual3_sqrt_zero_argument_is_finite() {
        let r = Dual3::<2>::var(0.0, 0).sqrt();
        assert_eq!(r.value, 0.0);
        assert!(r.grad.iter().all(|g| *g == 0.0));
        assert!(r.hess.iter().flatten().all(|h| h.is_finite()));
        assert!(r.d3.iter().flatten().flatten().all(|t| t.is_finite()));
        let neg = Dual3::<2>::var(-1e-18, 0).sqrt();
        assert!(neg.value.is_finite() && neg.d3.iter().flatten().flatten().all(|t| t.is_finite()));
    }

    /// Validate a `Dual3<2>` expression's third derivative `d3[i][j][k]` against
    /// central finite differences of the **`Dual2`** Hessian (`d3 = ∂hess/∂xₖ`).
    /// `f2`/`f3` are the same expression instantiated at the two dual orders (the
    /// generic `*<T: PkNum>` helpers below), so there is no formula duplication.
    fn fd_check3<F2, F3>(f2: F2, f3: F3, x: f64, y: f64, tol: f64)
    where
        F2: Fn(Dual2<2>, Dual2<2>) -> Dual2<2>,
        F3: Fn(Dual3<2>, Dual3<2>) -> Dual3<2>,
    {
        let d = f3(Dual3::var(x, 0), Dual3::var(y, 1));
        // Hessian symmetry across all three axes is a structural invariant.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    approx::assert_relative_eq!(
                        d.d3[i][j][k],
                        d.d3[j][i][k],
                        max_relative = 1e-12,
                        epsilon = 1e-12
                    );
                    approx::assert_relative_eq!(
                        d.d3[i][j][k],
                        d.d3[i][k][j],
                        max_relative = 1e-12,
                        epsilon = 1e-12
                    );
                }
            }
        }
        let hess = |a: f64, b: f64| f2(Dual2::var(a, 0), Dual2::var(b, 1)).hess;
        let h = 1e-4;
        for k in 0..2 {
            let (xp, yp) = if k == 0 { (x + h, y) } else { (x, y + h) };
            let (xm, ym) = if k == 0 { (x - h, y) } else { (x, y - h) };
            let hp = hess(xp, yp);
            let hm = hess(xm, ym);
            for i in 0..2 {
                for j in 0..2 {
                    let fd = (hp[i][j] - hm[i][j]) / (2.0 * h);
                    approx::assert_relative_eq!(
                        d.d3[i][j][k],
                        fd,
                        max_relative = tol,
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    // Generic expressions, written once and instantiated at Dual2 and Dual3.
    fn e_exp<T: PkNum>(x: T, y: T) -> T {
        (x * y).exp()
    }
    fn e_ln<T: PkNum>(x: T, y: T) -> T {
        (x + y * y).ln()
    }
    fn e_sqrt<T: PkNum>(x: T, y: T) -> T {
        (x * x + y).sqrt()
    }
    fn e_pow_const<T: PkNum>(x: T, y: T) -> T {
        x.pow(T::from_f64(2.5)) * y
    }
    fn e_pow_var<T: PkNum>(x: T, y: T) -> T {
        x.pow(y)
    }
    fn e_cos<T: PkNum>(x: T, y: T) -> T {
        (x * y).cos()
    }
    fn e_acos<T: PkNum>(x: T, y: T) -> T {
        (x * y).acos()
    }
    fn e_inv_logit<T: PkNum>(x: T, y: T) -> T {
        (x - y).inv_logit()
    }
    fn e_logit<T: PkNum>(x: T, y: T) -> T {
        (x * y).logit()
    }
    /// 1-cpt IV-bolus shape `(1/y)·exp(−x/y)` (amt=1): exercises Div (`recip`) and
    /// `exp` composed — the dominant pattern in the PK kernels.
    fn e_iv_shape<T: PkNum>(x: T, y: T) -> T {
        (T::from_f64(1.0) / y) * (-(x / y)).exp()
    }

    #[test]
    fn d3_matches_fd_of_dual2_hess() {
        fd_check3(e_exp, e_exp, 1.3, 0.7, 1e-4);
        fd_check3(e_ln, e_ln, 1.5, 0.7, 1e-3);
        fd_check3(e_sqrt, e_sqrt, 1.4, 0.6, 1e-3);
        fd_check3(e_pow_const, e_pow_const, 1.7, 0.9, 1e-3);
        fd_check3(e_pow_var, e_pow_var, 1.7, 0.8, 1e-3);
        fd_check3(e_cos, e_cos, 0.6, 0.9, 1e-3);
        fd_check3(e_acos, e_acos, 0.4, 0.8, 1e-3);
        fd_check3(e_inv_logit, e_inv_logit, 0.6, 0.2, 1e-3);
        fd_check3(e_logit, e_logit, 0.3, 0.9, 1e-3);
        fd_check3(e_iv_shape, e_iv_shape, 3.0, 5.0, 1e-3);
    }

    /// `logit`/`inv_logit` round-trip propagates consistently to third order.
    #[test]
    fn logit_inv_logit_roundtrip_third_order() {
        let p = Dual3::<2>::var(0.37, 0);
        let r = p.logit().inv_logit();
        approx::assert_relative_eq!(r.value, 0.37, max_relative = 1e-12);
        approx::assert_relative_eq!(r.grad[0], 1.0, max_relative = 1e-9);
        approx::assert_relative_eq!(r.hess[0][0], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(r.d3[0][0][0], 0.0, epsilon = 1e-5);
    }
}
