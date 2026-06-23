//! `DualMixed<NA, N>` — a *mixed-order* forward-mode dual number: a full
//! first-order gradient over `N` seeded inputs, but a second-order Hessian only
//! over the **first `NA`** of them.
//!
//! It is the partial-Hessian sibling of [`Dual2<N>`](super::dual2::Dual2). The
//! `N×N` Hessian of `Dual2` is replaced by a rectangular `NA×N` block: rows index
//! the `NA` axes that need second order, columns index all `N` axes. The square
//! sub-block among the trailing `N − NA` axes — pairs `(i, j)` with both `i ≥ NA`
//! and `j ≥ NA` — is neither stored nor computed.
//!
//! ## Why
//!
//! The user-ODE outer FOCEI gradient (issue #410, [`super::ode_provider`]) seeds a
//! dual on the `N` individual PK parameters and integrates it through RK45. The
//! η/θ chain that follows reads `∂²f/∂p_i∂p_j` only when **at least one** of
//! `p_i, p_j` carries IIV (an η) — FOCEI never uses `∂²f/∂θ²`, so the
//! second-order block among the IIV-free parameters is dead. Seeding the
//! IIV-bearing parameters as the first `NA` axes lets that block be dropped: the
//! per-step Hessian work falls from `N²` to `NA·N`. See issue #445.
//!
//! Each rule is the exact `Dual2` rule with the Hessian loop restricted to rows
//! `0..NA` (the gradient loop still spans all `N` columns); the parity tests pin
//! the retained entries to `Dual2` bit-for-bit.
// Indexed loops over the `[f64; N]` grad / `[[f64; N]; NA]` hess jets read more
// clearly than zipped iterators here, and `Div` is `× recip` (the
// `suspicious_arithmetic_impl` lint is a false positive for that identity).
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct DualMixed<const NA: usize, const N: usize> {
    pub value: f64,
    /// `∂/∂xᵢ` for all `N` seeded inputs.
    pub grad: [f64; N],
    /// `∂²/∂xᵣ∂x_c` for the first `NA` rows (`r < NA`) against all `N` columns.
    /// The `(N − NA)×(N − NA)` trailing block is intentionally absent.
    pub hess: [[f64; N]; NA],
}

impl<const NA: usize, const N: usize> DualMixed<NA, N> {
    /// A constant (zero gradient and Hessian).
    pub fn constant(value: f64) -> Self {
        DualMixed {
            value,
            grad: [0.0; N],
            hess: [[0.0; N]; NA],
        }
    }

    /// Seed input variable `i`: `value` with `∂/∂xᵢ = 1`, all else zero. `i` may be
    /// any axis in `0..N`; only a seed with `i < NA` will carry second-order rows.
    pub fn var(value: f64, i: usize) -> Self {
        let mut grad = [0.0; N];
        grad[i] = 1.0;
        DualMixed {
            value,
            grad,
            hess: [[0.0; N]; NA],
        }
    }

    /// `exp(self)`. With `u = e^x`: `u' = u·x'`, `u'' = u·(x'' + x'⊗x')`.
    pub fn exp(self) -> Self {
        let v = self.value.exp();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = v * self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = v * (self.hess[r][c] + self.grad[r] * self.grad[c]);
            }
        }
        DualMixed {
            value: v,
            grad,
            hess,
        }
    }

    /// `sqrt(self)`. With `u = √x`: `u' = x'/(2u)`, `u'' = x''/(2u) − x'⊗x'/(4u³)`.
    /// A non-positive argument returns a flat (zero-derivative) value — see
    /// [`Dual2::sqrt`](super::dual2::Dual2::sqrt) for the rationale.
    pub fn sqrt(self) -> Self {
        if !(self.value > 0.0) {
            let v = if self.value > 0.0 {
                self.value.sqrt()
            } else {
                0.0
            };
            return DualMixed {
                value: v,
                grad: [0.0; N],
                hess: [[0.0; N]; NA],
            };
        }
        let v = self.value.sqrt();
        let inv2u = 0.5 / v;
        let inv4u3 = 0.25 / (v * v * v);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = inv2u * self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = inv2u * self.hess[r][c] - inv4u3 * self.grad[r] * self.grad[c];
            }
        }
        DualMixed {
            value: v,
            grad,
            hess,
        }
    }

    /// `cos(self)`. `u' = −sin(x)·x'`, `u'' = −cos(x)·x'⊗x' − sin(x)·x''`.
    pub fn cos(self) -> Self {
        let (s, c) = self.value.sin_cos();
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = -s * self.grad[i];
        }
        for r in 0..NA {
            for col in 0..N {
                hess[r][col] = -s * self.hess[r][col] - c * self.grad[r] * self.grad[col];
            }
        }
        DualMixed {
            value: c,
            grad,
            hess,
        }
    }

    /// `acos(self)`. `u' = g₁·x'`, `u'' = g₁·x'' + g₂·x'⊗x'`, with `g₁ = −1/√(1−x²)`
    /// and `g₂ = −x·(1−x²)^{−3/2}`. Self-defending at `|x|→1` (clamp + floor), as in
    /// [`Dual2::acos`](super::dual2::Dual2::acos).
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
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = g1 * self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = g1 * self.hess[r][c] + g2 * self.grad[r] * self.grad[c];
            }
        }
        DualMixed {
            value: v_clamped.acos(),
            grad,
            hess,
        }
    }

    /// `ln(self)`. `u' = x'/x`, `u'' = x''/x − x'⊗x'/x²`.
    pub fn ln(self) -> Self {
        let x = self.value;
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = self.grad[i] * inv;
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = self.hess[r][c] * inv - self.grad[r] * self.grad[c] * inv2;
            }
        }
        DualMixed {
            value: x.ln(),
            grad,
            hess,
        }
    }

    /// `ln Γ(self)`. With ψ = digamma, ψ′ = trigamma: `u' = ψ(x)·x'`,
    /// `u'' = ψ(x)·x'' + ψ′(x)·x'⊗x'`. The transit absorption `ln Γ(n + 1)`
    /// constant on the analytic ODE sensitivity path (#430); `Dual2 = DualMixed`,
    /// so this also serves the second-order `Dual2` Hessian.
    pub fn ln_gamma(self) -> Self {
        let d1 = crate::stats::special::digamma(self.value);
        let d2 = crate::stats::special::trigamma(self.value);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = d1 * self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = d1 * self.hess[r][c] + d2 * self.grad[r] * self.grad[c];
            }
        }
        DualMixed {
            value: crate::stats::special::ln_gamma(self.value),
            grad,
            hess,
        }
    }

    /// `self^e`. Constant exponent uses the power rule `aⁿ` directly (exact for any
    /// base sign with integer `n`); otherwise the general `exp(e·ln(self))` form
    /// (requires `self.value > 0`). Mirrors [`Dual2::powd`](super::dual2::Dual2::powd).
    pub fn powd(self, e: Self) -> Self {
        let exp_const =
            e.grad.iter().all(|&g| g == 0.0) && e.hess.iter().flatten().all(|&h| h == 0.0);
        if exp_const {
            let n = e.value;
            let a = self.value;
            let an = a.powf(n);
            if a == 0.0 {
                return DualMixed::constant(an);
            }
            let c1 = n * a.powf(n - 1.0);
            let c2 = n * (n - 1.0) * a.powf(n - 2.0);
            let mut grad = [0.0; N];
            let mut hess = [[0.0; N]; NA];
            for i in 0..N {
                grad[i] = c1 * self.grad[i];
            }
            for r in 0..NA {
                for c in 0..N {
                    hess[r][c] = c1 * self.hess[r][c] + c2 * self.grad[r] * self.grad[c];
                }
            }
            return DualMixed {
                value: an,
                grad,
                hess,
            };
        }
        (e * self.ln()).exp()
    }

    /// `|self|`. Away from the kink `x = 0` this is `±self`.
    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            -self
        }
    }

    /// `inv_logit(self) = 1/(1+e^{−x})` (logistic sigmoid).
    pub fn inv_logit(self) -> Self {
        ((-self).exp() + 1.0).recip()
    }

    /// `logit(self) = ln(x) − ln(1−x)`.
    pub fn logit(self) -> Self {
        self.ln() - ((-self) + 1.0).ln()
    }

    /// `1/self`. `u' = −b'/b²`, `u'' = −b''/b² + 2·b'⊗b'/b³`.
    pub fn recip(self) -> Self {
        let b = self.value;
        let inv = 1.0 / b;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = -self.grad[i] * inv2;
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = -self.hess[r][c] * inv2 + 2.0 * self.grad[r] * self.grad[c] * inv3;
            }
        }
        DualMixed {
            value: inv,
            grad,
            hess,
        }
    }
}

impl<const NA: usize, const N: usize> Add for DualMixed<NA, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = self.hess[r][c] + rhs.hess[r][c];
            }
        }
        DualMixed {
            value: self.value + rhs.value,
            grad,
            hess,
        }
    }
}

impl<const NA: usize, const N: usize> Sub for DualMixed<NA, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<const NA: usize, const N: usize> Neg for DualMixed<NA, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = -self.hess[r][c];
            }
        }
        DualMixed {
            value: -self.value,
            grad,
            hess,
        }
    }
}

impl<const NA: usize, const N: usize> Mul for DualMixed<NA, N> {
    type Output = Self;
    /// Leibniz: `(ab)ᵢ = a·bᵢ + aᵢ·b`,
    /// `(ab)ᵣ_c = a·b_rc + aᵣ·b_c + a_c·bᵣ + a_rc·b`.
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.value, rhs.value);
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = a * rhs.grad[i] + self.grad[i] * b;
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = a * rhs.hess[r][c]
                    + self.grad[r] * rhs.grad[c]
                    + self.grad[c] * rhs.grad[r]
                    + self.hess[r][c] * b;
            }
        }
        DualMixed {
            value: a * b,
            grad,
            hess,
        }
    }
}

impl<const NA: usize, const N: usize> Div for DualMixed<NA, N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

// ── Scalar (f64) conveniences ────────────────────────────────────────────────

impl<const NA: usize, const N: usize> Mul<f64> for DualMixed<NA, N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        let mut grad = [0.0; N];
        let mut hess = [[0.0; N]; NA];
        for i in 0..N {
            grad[i] = self.grad[i] * s;
        }
        for r in 0..NA {
            for c in 0..N {
                hess[r][c] = self.hess[r][c] * s;
            }
        }
        DualMixed {
            value: self.value * s,
            grad,
            hess,
        }
    }
}

impl<const NA: usize, const N: usize> Add<f64> for DualMixed<NA, N> {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        DualMixed {
            value: self.value + s,
            ..self
        }
    }
}

/// `s / dual` (scalar numerator).
pub fn scalar_div<const NA: usize, const N: usize>(
    s: f64,
    d: DualMixed<NA, N>,
) -> DualMixed<NA, N> {
    d.recip() * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// Run the same closed-form expression through `Dual2<N>` and `DualMixed<NA,N>`
    /// and assert every *retained* entry matches bit-for-bit: the value, the full
    /// `N`-gradient, and the `NA×N` Hessian rows. (The dropped trailing block has no
    /// counterpart to check.) `f_*` build the same expression generic over the dual
    /// type; we seed identical axes in both.
    fn parity_3<F2, FM>(f2: F2, fm: FM, x: f64, y: f64, z: f64)
    where
        F2: Fn(Dual2<3>, Dual2<3>, Dual2<3>) -> Dual2<3>,
        FM: Fn(DualMixed<2, 3>, DualMixed<2, 3>, DualMixed<2, 3>) -> DualMixed<2, 3>,
    {
        let full = f2(Dual2::var(x, 0), Dual2::var(y, 1), Dual2::var(z, 2));
        let mixed = fm(
            DualMixed::var(x, 0),
            DualMixed::var(y, 1),
            DualMixed::var(z, 2),
        );
        assert_eq!(full.value, mixed.value, "value");
        for i in 0..3 {
            assert_eq!(full.grad[i], mixed.grad[i], "grad[{i}]");
        }
        // Retained rows: r in 0..NA(=2), all 3 columns.
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(full.hess[r][c], mixed.hess[r][c], "hess[{r}][{c}]");
            }
        }
    }

    /// Exercise every rule via the parity harness. `z` (axis 2) is the dropped-row
    /// axis, so each expression must mix it into the retained rows (through products
    /// with axes 0/1) to prove the columns are still right.
    #[test]
    fn mixed_matches_dual2_on_retained_entries() {
        parity_3(
            |a, b, c| (a * b + c).exp(),
            |a, b, c| (a * b + c).exp(),
            1.3,
            0.7,
            0.4,
        );
        parity_3(
            |a, b, c| (a * c) / (b + c),
            |a, b, c| (a * c) / (b + c),
            1.1,
            0.9,
            0.5,
        );
        parity_3(
            |a, b, c| ((a + b * c).ln()) * c,
            |a, b, c| ((a + b * c).ln()) * c,
            1.7,
            0.6,
            0.8,
        );
        parity_3(
            |a, b, c| (a * b * c).sqrt(),
            |a, b, c| (a * b * c).sqrt(),
            1.2,
            0.8,
            0.5,
        );
        parity_3(
            |a, b, c| a.powd(Dual2::constant(2.5)) * c - b,
            |a, b, c| a.powd(DualMixed::constant(2.5)) * c - b,
            1.4,
            0.3,
            0.9,
        );
        parity_3(
            |a, b, c| (a * c).powd(b),
            |a, b, c| (a * c).powd(b),
            1.25,
            0.7,
            1.1,
        );
        parity_3(
            |a, b, c| (a * b - c).abs() + c.recip(),
            |a, b, c| (a * b - c).abs() + c.recip(),
            0.4,
            1.0,
            0.7,
        );
        parity_3(
            |a, b, c| (a * b * c).inv_logit(),
            |a, b, c| (a * b * c).inv_logit(),
            0.6,
            0.5,
            0.7,
        );
        parity_3(
            |a, b, c| (a * b * c).logit(),
            |a, b, c| (a * b * c).logit(),
            0.5,
            0.8,
            0.9,
        );
        parity_3(
            |a, b, c| (a * c).cos() + b,
            |a, b, c| (a * c).cos() + b,
            0.7,
            0.3,
            0.6,
        );
        parity_3(
            |a, b, c| (a * c * 0.5).acos() * b,
            |a, b, c| (a * c * 0.5).acos() * b,
            0.6,
            1.1,
            0.5,
        );
        parity_3(
            |a, b, c| crate::sens::dual2::scalar_div(2.0, a * b + c),
            |a, b, c| scalar_div(2.0, a * b + c),
            1.0,
            0.6,
            0.4,
        );
    }

    /// The dropped trailing block (`hess[i][j]` with both `i ≥ NA` and `j ≥ NA`) is
    /// never materialised — `DualMixed<1,2>` keeps a single Hessian row. Cross-check
    /// the one retained row against `Dual2<2>` for an expression that genuinely
    /// couples axes 0 and 1.
    #[test]
    fn mixed_single_row_matches_dual2() {
        let f = |a: Dual2<2>, b: Dual2<2>| (a * b).exp() + b.ln();
        let g = |a: DualMixed<1, 2>, b: DualMixed<1, 2>| (a * b).exp() + b.ln();
        let full = f(Dual2::var(1.3, 0), Dual2::var(0.7, 1));
        let mixed = g(DualMixed::var(1.3, 0), DualMixed::var(0.7, 1));
        assert_eq!(full.value, mixed.value);
        assert_eq!(full.grad, mixed.grad);
        for c in 0..2 {
            assert_eq!(full.hess[0][c], mixed.hess[0][c], "hess[0][{c}]");
        }
    }

    /// `NA == N` reproduces the full `Dual2` Hessian (the degenerate, no-saving case
    /// the dispatcher avoids, but the type must still be correct there).
    #[test]
    fn mixed_full_width_equals_dual2() {
        let f = |a: Dual2<2>, b: Dual2<2>| (a / b + a).sqrt() * b;
        let g = |a: DualMixed<2, 2>, b: DualMixed<2, 2>| (a / b + a).sqrt() * b;
        let full = f(Dual2::var(2.0, 0), Dual2::var(1.5, 1));
        let mixed = g(DualMixed::var(2.0, 0), DualMixed::var(1.5, 1));
        assert_eq!(full.value, mixed.value);
        assert_eq!(full.grad, mixed.grad);
        assert_eq!(full.hess, mixed.hess);
    }
}
