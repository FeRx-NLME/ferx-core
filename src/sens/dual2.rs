//! `Dual2<N>` — the independent forward-mode second-order dual (value, gradient, full
//! Hessian over `N` seeded inputs) the tests check the compiled per-subject plans
//! (`parser::kernel_jit`) against, and the one scalar jet production still needs
//! (`residual_rd2_corr`, the cov-static fold's bit-exact value arithmetic).
//!
//! Every elementary function is `lift2`: `φ(x)` from `φ, φ′, φ″` at the argument, so a
//! unary rule is one line and the Hessian is always `φ′x″ + φ″ x′⊗x′`.
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual2<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
    pub hess: [[f64; N]; N],
}

impl<const N: usize> Dual2<N> {
    pub fn constant(value: f64) -> Self {
        Dual2 {
            value,
            grad: [0.0; N],
            hess: [[0.0; N]; N],
        }
    }

    /// Seed input `i`: `value` with `∂/∂xᵢ = 1`.
    pub fn var(value: f64, i: usize) -> Self {
        let mut d = Self::constant(value);
        d.grad[i] = 1.0;
        d
    }

    /// `φ(self)` from `v = φ(x)`, `d1 = φ′(x)`, `d2 = φ″(x)`:
    /// `u′ = φ′x′`, `u″ = φ′x″ + φ″x′⊗x′`.
    pub fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        let mut o = Self::constant(v);
        for i in 0..N {
            o.grad[i] = d1 * self.grad[i];
            for c in 0..N {
                o.hess[i][c] = d1 * self.hess[i][c] + d2 * self.grad[i] * self.grad[c];
            }
        }
        o
    }

    pub fn exp(self) -> Self {
        let v = self.value.exp();
        self.lift2(v, v, v)
    }

    /// A non-positive argument returns a flat (zero-derivative) value, not `inf`/`NaN`
    /// from the singular `1/(2√x)` factors.
    pub fn sqrt(self) -> Self {
        if !(self.value > 0.0) {
            return Self::constant(0.0);
        }
        let v = self.value.sqrt();
        self.lift2(v, 0.5 / v, -0.25 / (v * v * v))
    }

    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        self.lift2(c, -s, -c)
    }

    /// Self-defending at `|x| → 1` (clamp + floor of `1 − x²`).
    pub fn acos(self) -> Self {
        let x = self.value;
        let om = (1.0 - x * x).max(1e-12);
        let s = om.sqrt();
        self.lift2(x.clamp(-1.0, 1.0).acos(), -1.0 / s, -x / (om * s))
    }

    pub fn ln(self) -> Self {
        let inv = 1.0 / self.value;
        self.lift2(self.value.ln(), inv, -inv * inv)
    }

    pub fn ln_gamma(self) -> Self {
        let x = self.value;
        let (d1, d2) = (
            crate::stats::special::digamma(x),
            crate::stats::special::trigamma(x),
        );
        self.lift2(crate::stats::special::ln_gamma(x), d1, d2)
    }

    /// `self^e`: a constant exponent is the power rule (exact for any base sign with
    /// integer `n`); otherwise `exp(e·ln(self))` (needs `self.value > 0`).
    pub fn powd(self, e: Self) -> Self {
        let const_exp =
            e.grad.iter().all(|&g| g == 0.0) && e.hess.iter().flatten().all(|&h| h == 0.0);
        if !const_exp {
            return (e * self.ln()).exp();
        }
        let (n, a) = (e.value, self.value);
        if a == 0.0 {
            return Self::constant(a.powf(n));
        }
        self.lift2(
            a.powf(n),
            n * a.powf(n - 1.0),
            n * (n - 1.0) * a.powf(n - 2.0),
        )
    }

    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            -self
        }
    }

    pub fn inv_logit(self) -> Self {
        ((-self).exp() + 1.0).recip()
    }

    pub fn logit(self) -> Self {
        self.ln() - ((-self) + 1.0).ln()
    }

    pub fn recip(self) -> Self {
        let inv = 1.0 / self.value;
        self.lift2(inv, -inv * inv, 2.0 * inv * inv * inv)
    }

    /// Bit-for-bit equality of the value and every jet entry.
    pub(crate) fn same_bits(&self, o: &Self) -> bool {
        self.value.to_bits() == o.value.to_bits()
            && self
                .grad
                .iter()
                .zip(&o.grad)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self
                .hess
                .iter()
                .flatten()
                .zip(o.hess.iter().flatten())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Whether every gradient and Hessian entry is finite (the value is checked apart:
    /// an overflow reaches the Hessian first, #1204).
    pub fn jets_finite(self) -> bool {
        self.grad
            .iter()
            .chain(self.hess.iter().flatten())
            .all(|x| x.is_finite())
    }

    /// The largest absolute gradient or Hessian entry; `NaN` if any entry is `NaN`.
    pub fn jet_max_abs(self) -> f64 {
        self.grad
            .iter()
            .chain(self.hess.iter().flatten())
            .fold(0.0_f64, |m, g| {
                if g.is_nan() || m.is_nan() {
                    f64::NAN
                } else {
                    m.max(g.abs())
                }
            })
    }
}

impl<const N: usize> Add for Dual2<N> {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        let mut o = self;
        o.value += r.value;
        for i in 0..N {
            o.grad[i] += r.grad[i];
            for c in 0..N {
                o.hess[i][c] += r.hess[i][c];
            }
        }
        o
    }
}

impl<const N: usize> Neg for Dual2<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut o = self;
        o.value = -o.value;
        for i in 0..N {
            o.grad[i] = -o.grad[i];
            for c in 0..N {
                o.hess[i][c] = -o.hess[i][c];
            }
        }
        o
    }
}

impl<const N: usize> Sub for Dual2<N> {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        self + (-r)
    }
}

impl<const N: usize> Mul for Dual2<N> {
    type Output = Self;
    /// Leibniz: `(ab)ᵢ = a·bᵢ + aᵢ·b`, `(ab)ᵢⱼ = a·bᵢⱼ + aᵢbⱼ + aⱼbᵢ + aᵢⱼ·b`.
    fn mul(self, r: Self) -> Self {
        let (a, b) = (self.value, r.value);
        let mut o = Self::constant(a * b);
        for i in 0..N {
            o.grad[i] = a * r.grad[i] + self.grad[i] * b;
            for c in 0..N {
                o.hess[i][c] = a * r.hess[i][c]
                    + self.grad[i] * r.grad[c]
                    + self.grad[c] * r.grad[i]
                    + self.hess[i][c] * b;
            }
        }
        o
    }
}

impl<const N: usize> Div for Dual2<N> {
    type Output = Self;
    fn div(self, r: Self) -> Self {
        self * r.recip()
    }
}

impl<const N: usize> Mul<f64> for Dual2<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        self * Self::constant(s)
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
