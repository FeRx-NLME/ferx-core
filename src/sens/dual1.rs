//! `Dual1<N>` — a first-order forward-mode multivariate dual number: value plus
//! gradient over `N` seeded variables, with **no Hessian**. It is the light
//! sibling of [`Dual2`](super::dual2::Dual2): evaluating a closed-form PK
//! solution generic over [`PkNum`](super::num::PkNum) in `Dual1<N>` yields `f`
//! and `∂f/∂pkᵢ` only, at roughly half the per-op cost (no `N×N` Hessian to
//! propagate).
//!
//! The inner EBE gradient (Almquist 2015) needs only `f` and `∂f/∂η` — never the
//! second-order `∂²f/∂η²` or the θ-block — so it runs on `Dual1`, while the outer
//! gradient / EBE Hessian keep the full `Dual2` provider. Both share the same
//! generic PK source; only the seeded number type differs (issue #367).
// Indexed loops over the `[f64; N]` grad read more clearly than zipped iterators,
// and `Div` is `× recip` (the `suspicious_arithmetic_impl` lint is a false
// positive for that identity).
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Dual1<const N: usize> {
    pub value: f64,
    pub grad: [f64; N],
}

impl<const N: usize> Dual1<N> {
    /// A constant (zero gradient).
    pub fn constant(value: f64) -> Self {
        Dual1 {
            value,
            grad: [0.0; N],
        }
    }

    /// Seed input variable `i`: `value` with `∂/∂xᵢ = 1`, all else zero.
    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        Dual1 { value, grad }
    }

    /// `exp(self)`: `u = e^x`, `u' = u·x'`.
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = v * self.grad[i];
        }
        Dual1 { value: v, grad }
    }

    /// `ln(self)`: `u' = x'/x`.
    pub fn ln(self) -> Self {
        let inv = 1.0 / self.value;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] * inv;
        }
        Dual1 {
            value: self.value.ln(),
            grad,
        }
    }

    /// `ln Γ(self)`: `u' = ψ(x)·x'`, where ψ is the digamma function. The
    /// first-order rule for the transit absorption `ln Γ(n + 1)` constant on the
    /// analytic ODE sensitivity path (#430).
    pub fn ln_gamma(self) -> Self {
        let d1 = crate::stats::special::digamma(self.value);
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = d1 * self.grad[i];
        }
        Dual1 {
            value: crate::stats::special::ln_gamma(self.value),
            grad,
        }
    }

    /// `sqrt(self)`: `u' = x'/(2√x)`.
    pub fn sqrt(self) -> Self {
        // Non-positive argument: `1/(2√x)` is singular (and `inf·0 = NaN` when a
        // seeded component is 0). Return a zero-derivative value so `inf`/`NaN`
        // never enters the chain (callers guard the argument non-negative; this
        // covers an exact-0 discriminant during a line-search step).
        if !(self.value > 0.0) {
            let v = if self.value > 0.0 {
                self.value.sqrt()
            } else {
                0.0
            };
            return Dual1 {
                value: v,
                grad: [0.0; N],
            };
        }
        let v = self.value.sqrt();
        let inv2u = 0.5 / v;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = inv2u * self.grad[i];
        }
        Dual1 { value: v, grad }
    }

    /// `cos(self)`: `u' = −sin(x)·x'`.
    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = -s * self.grad[i];
        }
        Dual1 { value: c, grad }
    }

    /// `acos(self)`: `u' = g₁·x'`, `g₁ = −1/√(1−x²)`. Same `|x|→1` self-defence as
    /// [`Dual2::acos`](super::dual2::Dual2::acos): clamp the value and floor
    /// `1−x²` so a saturated/constant argument never yields `inf`/`NaN`.
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
        let g1 = -1.0 / one_minus.sqrt();
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = g1 * self.grad[i];
        }
        Dual1 {
            value: v_clamped.acos(),
            grad,
        }
    }

    /// `self^e`. Constant exponent uses the power rule `c₁ = n·aⁿ⁻¹` (exact for any
    /// base sign with integer `n`); otherwise the general `exp(e·ln(self))` form.
    pub fn powd(self, e: Self) -> Self {
        let exp_const = e.grad.iter().all(|&g| g == 0.0);
        if exp_const {
            let n = e.value;
            let a = self.value;
            let an = a.powf(n);
            if a == 0.0 {
                return Dual1::constant(an);
            }
            let c1 = n * a.powf(n - 1.0);
            let mut grad = [0.0; N];
            for i in 0..N {
                grad[i] = c1 * self.grad[i];
            }
            return Dual1 { value: an, grad };
        }
        (e * self.ln()).exp()
    }

    /// `|self|` (the cusp at 0 is measure-zero and ignored).
    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            -self
        }
    }

    /// `inv_logit(self) = 1/(1+e^{−x})`: `u' = u·(1−u)·x'`.
    pub fn inv_logit(self) -> Self {
        let u = 1.0 / (1.0 + (-self.value).exp());
        let d = u * (1.0 - u);
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = d * self.grad[i];
        }
        Dual1 { value: u, grad }
    }

    /// `logit(self) = ln(x/(1−x))`: `u' = x'/(x(1−x))`.
    pub fn logit(self) -> Self {
        let x = self.value;
        let d = 1.0 / (x * (1.0 - x));
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = d * self.grad[i];
        }
        Dual1 {
            value: (x / (1.0 - x)).ln(),
            grad,
        }
    }

    /// `1/self`: `u' = −x'/x²`.
    pub fn recip(self) -> Self {
        let inv = 1.0 / self.value;
        let inv2 = inv * inv;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = -self.grad[i] * inv2;
        }
        Dual1 { value: inv, grad }
    }
}

impl<const N: usize> Add for Dual1<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        Dual1 {
            value: self.value + rhs.value,
            grad,
        }
    }
}

impl<const N: usize> Sub for Dual1<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const N: usize> Neg for Dual1<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        Dual1 {
            value: -self.value,
            grad,
        }
    }
}

impl<const N: usize> Mul for Dual1<N> {
    type Output = Self;
    /// Leibniz: `(ab)ᵢ = a·bᵢ + aᵢ·b`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.value, rhs.value);
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = a * rhs.grad[i] + self.grad[i] * b;
        }
        Dual1 { value: a * b, grad }
    }
}

impl<const N: usize> Div for Dual1<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sqrt` at a zero/rounded-negative argument with a seeded gradient must
    /// return finite, zero derivatives (not `inf`/`NaN` from `1/(2√x)`).
    #[test]
    fn dual1_sqrt_zero_argument_is_finite() {
        let r = Dual1::<2>::var(0.0, 0).sqrt();
        assert_eq!(r.value, 0.0);
        assert!(r.grad.iter().all(|g| g.is_finite() && *g == 0.0));
        let neg = Dual1::<2>::var(-1e-18, 0).sqrt();
        assert!(neg.value.is_finite() && neg.grad.iter().all(|g| g.is_finite()));
    }

    /// Validate a `Dual1<2>` expression's gradient against central finite
    /// differences of its value, at `(x, y)`.
    fn fd_check<F>(f: F, x: f64, y: f64, gtol: f64)
    where
        F: Fn(Dual1<2>, Dual1<2>) -> Dual1<2>,
    {
        let d = f(Dual1::var(x, 0), Dual1::var(y, 1));
        let v = |a: f64, b: f64| f(Dual1::var(a, 0), Dual1::var(b, 1)).value;
        let h = 1e-6;
        let gx = (v(x + h, y) - v(x - h, y)) / (2.0 * h);
        let gy = (v(x, y + h) - v(x, y - h)) / (2.0 * h);
        approx::assert_relative_eq!(d.grad[0], gx, max_relative = gtol, epsilon = 1e-8);
        approx::assert_relative_eq!(d.grad[1], gy, max_relative = gtol, epsilon = 1e-8);
    }

    #[test]
    fn iv_bolus_shape_grad_matches_fd() {
        // (1/y)·exp(−x/y), the 1-cpt IV-bolus shape (amt = 1, t = 1).
        fd_check(|x, y| y.recip() * (-(x / y)).exp(), 3.0, 5.0, 1e-6);
    }

    #[test]
    fn transcendental_grads_match_fd() {
        fd_check(|x, y| (x * y).ln(), 2.0, 3.0, 1e-6);
        fd_check(|x, y| (x + y * y).sqrt(), 1.5, 0.7, 1e-6);
        fd_check(|x, y| x.powd(Dual1::constant(2.5)) * y, 1.7, 0.9, 1e-6);
        fd_check(|x, y| x.powd(y), 1.7, 0.8, 1e-6);
        fd_check(|x, y| (x - y).cos(), 0.6, 0.2, 1e-6);
        fd_check(|x, y| (x * y * Dual1::constant(0.1)).acos(), 1.2, 0.5, 1e-6);
        fd_check(|x, y| (x - y).inv_logit(), 0.6, 0.2, 1e-6);
        fd_check(|x, y| (x * y).logit(), 0.3, 0.9, 1e-6);
    }
}
