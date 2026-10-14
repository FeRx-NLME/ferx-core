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
    // In-place arithmetic behind `crate::sens::num::vm::VmArith`, the bytecode VM's
    // operand updates. Each writes over `self` exactly what the by-value operator
    // below computes — same per-entry formula, same association — so a `DualMixed`
    // (several KB at the outer gradient's axis counts) is updated where it sits
    // instead of being moved through temporaries. The `*_scalar_in` forms take a
    // constant operand `c` (right, or left for `rsub`) and evaluate the operator's
    // formula against `DualMixed::constant(c)` term by term, zero products
    // included, so signed zeros and `inf · 0 = NaN` match too.

    /// `self = self + rhs`; same per-entry formula as `Add`.
    #[inline]
    pub(crate) fn add_in(&mut self, rhs: &Self) {
        for i in 0..N {
            self.grad[i] += rhs.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                self.hess[r][c] += rhs.hess[r][c];
            }
        }
        self.value += rhs.value;
    }

    /// `self = self - rhs`. `Sub` is `self + (−rhs)`, and IEEE defines `x − y` as
    /// `x + (−y)`, signed zeros included.
    #[inline]
    pub(crate) fn sub_in(&mut self, rhs: &Self) {
        for i in 0..N {
            self.grad[i] -= rhs.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                self.hess[r][c] -= rhs.hess[r][c];
            }
        }
        self.value -= rhs.value;
    }

    /// `self = rhs - self`.
    #[inline]
    pub(crate) fn rsub_in(&mut self, rhs: &Self) {
        for i in 0..N {
            self.grad[i] = rhs.grad[i] - self.grad[i];
        }
        for r in 0..NA {
            for c in 0..N {
                self.hess[r][c] = rhs.hess[r][c] - self.hess[r][c];
            }
        }
        self.value = rhs.value - self.value;
    }

    /// `self = self * rhs`; same formula and association as `Mul`. The Hessian is
    /// written before the gradient because it reads the old gradient.
    #[inline]
    pub(crate) fn mul_in(&mut self, rhs: &Self) {
        let (a, b) = (self.value, rhs.value);
        for r in 0..NA {
            for c in 0..N {
                self.hess[r][c] = a * rhs.hess[r][c]
                    + self.grad[r] * rhs.grad[c]
                    + self.grad[c] * rhs.grad[r]
                    + self.hess[r][c] * b;
            }
        }
        for i in 0..N {
            self.grad[i] = a * rhs.grad[i] + self.grad[i] * b;
        }
        self.value = a * b;
    }

    /// `self = self / rhs`, i.e. `self · rhs⁻¹` exactly as `Div` forms it.
    #[inline]
    pub(crate) fn div_in(&mut self, rhs: &Self) {
        let r = rhs.recip();
        self.mul_in(&r);
    }

    /// `self = self + constant(c)`: the jet gains `+ 0.0`.
    #[inline]
    pub(crate) fn add_scalar_in(&mut self, c: f64) {
        for i in 0..N {
            self.grad[i] += 0.0;
        }
        for r in 0..NA {
            for col in 0..N {
                self.hess[r][col] += 0.0;
            }
        }
        self.value += c;
    }

    /// `self = self - constant(c)` = `self + (−constant(c))`. The jet gains
    /// `+ (−0.0)`, which is the identity on every `f64` (signed zeros and `NaN`
    /// included), so only the value moves.
    #[inline]
    pub(crate) fn sub_scalar_in(&mut self, c: f64) {
        self.value += -c;
    }

    /// `self = constant(c) - self` = `constant(c) + (−self)`.
    #[inline]
    pub(crate) fn rsub_scalar_in(&mut self, c: f64) {
        for i in 0..N {
            self.grad[i] = 0.0 + (-self.grad[i]);
        }
        for r in 0..NA {
            for col in 0..N {
                self.hess[r][col] = 0.0 + (-self.hess[r][col]);
            }
        }
        self.value = c + (-self.value);
    }

    /// `self = self * constant(c)`: [`Self::mul_in`] with `rhs.grad = rhs.hess = 0`.
    #[inline]
    pub(crate) fn mul_scalar_in(&mut self, c: f64) {
        let a = self.value;
        for r in 0..NA {
            for col in 0..N {
                self.hess[r][col] =
                    a * 0.0 + self.grad[r] * 0.0 + self.grad[col] * 0.0 + self.hess[r][col] * c;
            }
        }
        for i in 0..N {
            self.grad[i] = a * 0.0 + self.grad[i] * c;
        }
        self.value = a * c;
    }

    /// `self = self / constant(c)` = `self · recip(constant(c))`, with the constant's
    /// `recip` jet (`−0·c⁻²`, `−0·c⁻² + 2·0·0·c⁻³`) formed exactly as [`Self::recip`]
    /// forms it.
    #[inline]
    pub(crate) fn div_scalar_in(&mut self, c: f64) {
        let inv = 1.0 / c;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let rg = -0.0 * inv2;
        let rh = -0.0 * inv2 + 2.0 * 0.0 * 0.0 * inv3;
        let a = self.value;
        for r in 0..NA {
            for col in 0..N {
                self.hess[r][col] =
                    a * rh + self.grad[r] * rg + self.grad[col] * rg + self.hess[r][col] * inv;
            }
        }
        for i in 0..N {
            self.grad[i] = a * rg + self.grad[i] * inv;
        }
        self.value = a * inv;
    }
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

    /// Whether every gradient **and** Hessian component is finite. Does **not** look at
    /// `value` — see [`crate::sens::num::PkNum::jets_finite`] for why the two are checked
    /// apart (#1204).
    ///
    /// The Hessian goes first, and by a wide margin: `recip` scales it by `1/x³` against
    /// the gradient's `1/x²`, and a `u' = p·u` integration gives `∂²u/∂p² = t²·u` against
    /// `∂u/∂p = t·u`. On the #1204 repro the value is `7.2e303` and the gradient
    /// `1.4e306` — both finite — while the Hessian is already `NaN`.
    pub fn jets_finite(self) -> bool {
        self.grad.iter().all(|g| g.is_finite()) && self.hess.iter().flatten().all(|h| h.is_finite())
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

    /// #1204's premise, at the arithmetic level: a dual's value and its jets stop being
    /// finite at very different magnitudes, so a value-only check cannot stand in for a
    /// derivative check.
    ///
    /// `recip` is the sharpest case in the library — it scales the gradient by `1/x²` and
    /// the Hessian by `1/x³` against the value's `1/x` — so the three cross `f64::MAX` a
    /// hundred decades apart.
    #[test]
    fn jets_finite_sees_an_overflow_the_value_does_not() {
        // A state that has decayed to 1e-120 with an O(100) seeded gradient: the reciprocal's
        // value and gradient are comfortably finite, its Hessian is not.
        let x = DualMixed::<1, 1> {
            value: 1e-120,
            grad: [100.0],
            hess: [[0.0]],
        };
        let r = x.recip();
        assert!(r.value.is_finite(), "value 1e120 is finite: {}", r.value);
        assert!(
            r.grad[0].is_finite(),
            "gradient 1e242 is finite: {}",
            r.grad[0]
        );
        assert!(
            !r.hess[0][0].is_finite(),
            "the Hessian must have overflowed: {}",
            r.hess[0][0]
        );
        assert!(!r.jets_finite(), "so jets_finite must reject it");
    }

    /// Either order alone is enough to reject. The Hessian overflows first in practice, but
    /// a gradient can go without it (a first-order `Dual1` solve has no Hessian at all), so
    /// the check has to read both rather than pick the one that usually fails.
    #[test]
    fn jets_finite_rejects_a_gradient_only_overflow_too() {
        let grad_only = DualMixed::<1, 1> {
            value: 1.0,
            grad: [f64::INFINITY],
            hess: [[0.0]],
        };
        assert!(grad_only.value.is_finite());
        assert!(!grad_only.jets_finite());

        let hess_only = DualMixed::<1, 1> {
            value: 1.0,
            grad: [1.0],
            hess: [[f64::NAN]],
        };
        assert!(hess_only.value.is_finite() && hess_only.grad[0].is_finite());
        assert!(!hess_only.jets_finite());
    }

    /// And it must not fire on ordinary arithmetic, or the guard would reject every
    /// escalation it saw.
    #[test]
    fn jets_finite_accepts_a_healthy_dual() {
        let x = DualMixed::<2, 2>::var(2.0, 0);
        let y = DualMixed::<2, 2>::var(3.0, 1);
        assert!((x * y).jets_finite());
        assert!((x / y).jets_finite());
        assert!(x.ln().jets_finite());
        assert!(x.sqrt().jets_finite());
    }

    /// Two operands with every jet entry live: non-zero gradients on all three axes
    /// and non-zero Hessian rows, so an in-place formula that drops a term, reads a
    /// gradient after overwriting it, or swaps the operands shows up.
    fn in_place_operands() -> (DualMixed<2, 3>, DualMixed<2, 3>) {
        let x = DualMixed::<2, 3>::var(1.3, 0);
        let y = DualMixed::<2, 3>::var(0.7, 1);
        let z = DualMixed::<2, 3>::var(2.1, 2);
        let a = x * y.exp() + z.ln() * x;
        let b = (y * z).sqrt() - x * x + z;
        (a, b)
    }

    /// Operands where the by-value forms' zero terms are visible: `−0.0` jet
    /// entries (`−0.0 + 0.0 = +0.0`), and an infinite gradient (`∞ · 0 = NaN`).
    fn edge_operands() -> [DualMixed<2, 3>; 2] {
        let signed_zero = DualMixed::<2, 3> {
            value: 2.0,
            grad: [-0.0, 1.5, -0.0],
            hess: [[-0.0, 0.25, -0.0], [0.25, -0.0, 0.0]],
        };
        let infinite = DualMixed::<2, 3> {
            value: 1.0,
            grad: [f64::INFINITY, -2.0, 0.0],
            hess: [[0.0, 1.0, -0.0], [1.0, 0.0, 3.0]],
        };
        [signed_zero, infinite]
    }

    /// `got` equals `want` bit for bit in every entry, any `NaN` matching any `NaN`
    /// (a payload is not part of the arithmetic's contract).
    fn assert_same(got: DualMixed<2, 3>, want: DualMixed<2, 3>, what: &str) {
        let eq = |g: f64, w: f64| g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
        assert!(
            eq(got.value, want.value),
            "{what}: value {} vs {}",
            got.value,
            want.value
        );
        for i in 0..3 {
            assert!(
                eq(got.grad[i], want.grad[i]),
                "{what}: grad[{i}] {:?} vs {:?}",
                got.grad[i],
                want.grad[i]
            );
        }
        for r in 0..2 {
            for c in 0..3 {
                assert!(
                    eq(got.hess[r][c], want.hess[r][c]),
                    "{what}: hess[{r}][{c}] {:?} vs {:?}",
                    got.hess[r][c],
                    want.hess[r][c]
                );
            }
        }
    }

    /// The in-place operand methods the bytecode VM uses are the by-value operators,
    /// bit for bit: same formulas, same association.
    #[test]
    fn in_place_binary_ops_are_bit_identical_to_the_operators() {
        let (a0, b0) = in_place_operands();
        let [e0, e1] = edge_operands();
        for (a, b) in [(a0, b0), (b0, a0), (e0, a0), (a0, e1), (e1, e0)] {
            let run = |f: fn(&mut DualMixed<2, 3>, &DualMixed<2, 3>)| {
                let mut x = a;
                f(&mut x, &b);
                x
            };
            assert_same(run(DualMixed::add_in), a + b, "add_in");
            assert_same(run(DualMixed::sub_in), a - b, "sub_in");
            assert_same(run(DualMixed::rsub_in), b - a, "rsub_in");
            assert_same(run(DualMixed::mul_in), a * b, "mul_in");
            assert_same(run(DualMixed::div_in), a / b, "div_in");
        }
    }

    /// A scalar operand gives exactly the operator against `DualMixed::constant(c)` —
    /// the jet a constant used to be materialised as — including the signed zeros and
    /// `NaN`s its zero jet produces: `+0.0` from `−0.0 + 0.0`, `NaN` from `∞ · 0`.
    #[test]
    fn in_place_scalar_ops_are_bit_identical_against_a_constant() {
        let (a0, _) = in_place_operands();
        let [e0, e1] = edge_operands();
        for a in [a0, e0, e1] {
            for c in [-0.37, 0.0, -0.0, 4.0] {
                let k = DualMixed::<2, 3>::constant(c);
                let run = |f: fn(&mut DualMixed<2, 3>, f64)| {
                    let mut x = a;
                    f(&mut x, c);
                    x
                };
                assert_same(run(DualMixed::add_scalar_in), a + k, "add_scalar_in");
                assert_same(run(DualMixed::sub_scalar_in), a - k, "sub_scalar_in");
                assert_same(run(DualMixed::rsub_scalar_in), k - a, "rsub_scalar_in");
                assert_same(run(DualMixed::mul_scalar_in), a * k, "mul_scalar_in");
                // A literal-left `c * E` is compiled as `E * c` (see `compile_expr_into`);
                // pin that the dual product is exact under that swap.
                assert_same(
                    run(DualMixed::mul_scalar_in),
                    k * a,
                    "mul_scalar_in (k * a)",
                );
                assert_same(run(DualMixed::div_scalar_in), a / k, "div_scalar_in");
            }
        }
        // The cases above must actually exercise the zero terms, or this test cannot
        // tell the exact form from a plain `value ∘ c` shortcut.
        let mut sum = e0;
        DualMixed::add_scalar_in(&mut sum, 1.0);
        assert_eq!(
            sum.grad[0].to_bits(),
            0.0f64.to_bits(),
            "−0.0 + 0.0 must be +0.0"
        );
        let mut prod = e1;
        DualMixed::mul_scalar_in(&mut prod, 0.0);
        assert!(
            prod.hess[0][0].is_nan(),
            "∞ · 0 must reach the Hessian as NaN"
        );
    }
}
