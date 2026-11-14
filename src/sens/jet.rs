//! Forward jets over a base number type: [`DualT`] (first order, `N` directions, nested for
//! second order). Over the recording type they are the likelihood kernel's local derivatives
//! (`∂V/∂f`, the inner Hessian), recorded into its plan.
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use super::num::PkNum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// First-order forward dual in `N` directions over a base `B` (value and tangents are
/// `B`). With `B` the recording type, its tangents are recorded expressions.
#[derive(Clone, Copy, Debug)]
pub struct DualT<const N: usize, B: Copy> {
    pub v: B,
    pub g: [B; N],
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    /// Whether this width's arithmetic runs out of line ([`crate::sens::num::outlined`]).
    const WIDE: bool = crate::sens::num::outlined(N, std::mem::size_of::<Self>());
    #[inline]
    pub fn constant(v: f64) -> Self {
        DualT {
            v: B::from_f64(v),
            g: [B::from_f64(0.0); N],
        }
    }
    #[inline]
    fn chain(self, hv: B, d: B) -> Self {
        if Self::WIDE {
            self.chain_wide(hv, d)
        } else {
            self.chain_body(hv, d)
        }
    }
    /// [`Self::chain`]'s body out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn chain_wide(self, hv: B, d: B) -> Self {
        self.chain_body(hv, d)
    }
    #[inline]
    fn chain_body(self, hv: B, d: B) -> Self {
        let mut g = self.g;
        for k in 0..N {
            g[k] = d * self.g[k];
        }
        DualT { v: hv, g }
    }
}

impl<const N: usize, B: PkNum> Add for DualT<N, B> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        if Self::WIDE {
            self.add_wide(o)
        } else {
            self.add_body(o)
        }
    }
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    #[inline]
    fn add_body(self, o: Self) -> Self {
        let mut g = self.g;
        for k in 0..N {
            g[k] = self.g[k] + o.g[k];
        }
        DualT { v: self.v + o.v, g }
    }
    /// [`Self::add_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn add_wide(self, o: Self) -> Self {
        self.add_body(o)
    }
}
impl<const N: usize, B: PkNum> Sub for DualT<N, B> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        if Self::WIDE {
            self.sub_wide(o)
        } else {
            self.sub_body(o)
        }
    }
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    #[inline]
    fn sub_body(self, o: Self) -> Self {
        let mut g = self.g;
        for k in 0..N {
            g[k] = self.g[k] - o.g[k];
        }
        DualT { v: self.v - o.v, g }
    }
    /// [`Self::sub_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn sub_wide(self, o: Self) -> Self {
        self.sub_body(o)
    }
}
impl<const N: usize, B: PkNum> Mul for DualT<N, B> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        if Self::WIDE {
            self.mul_wide(o)
        } else {
            self.mul_body(o)
        }
    }
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    #[inline]
    fn mul_body(self, o: Self) -> Self {
        let mut g = self.g;
        for k in 0..N {
            g[k] = B::dot2(self.g[k], o.v, self.v, o.g[k]);
        }
        DualT { v: self.v * o.v, g }
    }
    /// [`Self::mul_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn mul_wide(self, o: Self) -> Self {
        self.mul_body(o)
    }
}
impl<const N: usize, B: PkNum> Div for DualT<N, B> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        if Self::WIDE {
            self.div_wide(o)
        } else {
            self.div_body(o)
        }
    }
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    #[inline]
    fn div_body(self, o: Self) -> Self {
        let inv = B::from_f64(1.0) / o.v;
        let q = self.v * inv;
        let mut g = self.g;
        for k in 0..N {
            g[k] = (self.g[k] - q * o.g[k]) * inv;
        }
        DualT { v: q, g }
    }
    /// [`Self::div_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn div_wide(self, o: Self) -> Self {
        self.div_body(o)
    }
}
impl<const N: usize, B: PkNum> Neg for DualT<N, B> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if Self::WIDE {
            self.neg_wide()
        } else {
            self.neg_body()
        }
    }
}

impl<const N: usize, B: PkNum> DualT<N, B> {
    #[inline]
    fn neg_body(self) -> Self {
        let mut g = self.g;
        for k in 0..N {
            g[k] = -self.g[k];
        }
        DualT { v: -self.v, g }
    }
    /// [`Self::neg_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn neg_wide(self) -> Self {
        self.neg_body()
    }
}

impl<const N: usize, B: PkNum> PkNum for DualT<N, B> {
    const SECOND_ORDER: bool = false;
    const JET_ORDER: usize = B::JET_ORDER + 1;
    const WORK: usize = (1 + N) * B::WORK;
    /// `f` on the base with its tangent factor `f′` on the base: exact to the base's order,
    /// with no value read (a recording base records both, #2786).
    #[inline]
    fn jet_unary(self, f: crate::sens::num::JetFn) -> Self {
        self.jet_k(f, 0)
    }
    #[inline]
    fn jet_k(self, f: crate::sens::num::JetFn, k: u8) -> Self {
        self.chain(self.v.jet_k(f, k), self.v.jet_k(f, k + 1))
    }
    // The branch API on the base value (#2786): a recording base records the decision, so a
    // nested dual over it — the kernel's `f`-derivative helper — decides as the base does.
    const RECORDS_BRANCHES: bool = B::RECORDS_BRANCHES;
    #[inline]
    fn br_cmp(self, cc: crate::sens::num::BranchCmp, other: Self) -> bool {
        self.v.br_cmp(cc, other.v)
    }
    #[inline]
    fn br_cmp_c(self, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        self.v.br_cmp_c(cc, c)
    }
    #[inline]
    fn br_shift_cmp(self, base: f64, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        self.v.br_shift_cmp(base, cc, c)
    }
    #[inline]
    fn br_abs_lt(self, tol: f64) -> bool {
        self.v.br_abs_lt(tol)
    }
    #[inline]
    fn br_absdiff_lt(self, other: Self, tol: f64) -> bool {
        self.v.br_absdiff_lt(other.v, tol)
    }
    #[inline]
    fn br_is_nan(self) -> bool {
        self.v.br_is_nan()
    }
    /// Every part selected on the values' condition.
    #[inline]
    fn select_if(cc: crate::sens::num::BranchCmp, x: Self, y: Self, a: Self, b: Self) -> Self {
        DualT {
            v: B::select_if(cc, x.v, y.v, a.v, b.v),
            g: std::array::from_fn(|k| B::select_if(cc, x.v, y.v, a.g[k], b.g[k])),
        }
    }
    #[inline]
    fn value_only(self) -> Self {
        DualT {
            v: self.v.value_only(),
            g: [B::from_f64(0.0); N],
        }
    }
    #[inline]
    fn is_constant(self) -> bool {
        self.v.is_constant() && self.g.iter().all(|d| d.is_constant() && d.val() == 0.0)
    }
    fn from_f64(x: f64) -> Self {
        DualT::constant(x)
    }
    fn var(x: f64, dim: usize) -> Self {
        let mut d = DualT::constant(x);
        if dim < N {
            d.g[dim] = B::from_f64(1.0);
        }
        d
    }
    fn val(self) -> f64 {
        self.v.val()
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        let d = B::from_f64(1.0) / self.v;
        self.chain(self.v.ln(), d)
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        let d = B::from_f64(0.5) / s;
        self.chain(s, d)
    }
    fn pow(self, e: Self) -> Self {
        // A constant exponent (`(AGE − 40)^2`, `C^2` in `[odes]`) is the power rule in `B`,
        // as the forward dual's `pow` has it: `exp(e·ln x)` is NaN at `x ≤ 0` and declined the subject
        // to finite differences (#2093 review). The forward dual's convention at `x = 0` (a zero
        // derivative) is kept, so this differentiates the function the gradient records.
        if e.is_constant() {
            let n = e.v.val();
            if self.v.br_cmp_c(crate::sens::num::BranchCmp::Eq, 0.0) {
                return DualT::constant(0.0f64.powf(n));
            }
            let d1 = B::from_f64(n) * self.v.pow(B::from_f64(n - 1.0));
            return self.chain(self.v.pow(B::from_f64(n)), d1);
        }
        (e * self.ln()).exp()
    }
    fn abs(self) -> Self {
        if self.v.br_cmp_c(crate::sens::num::BranchCmp::Ge, 0.0) {
            self
        } else {
            -self
        }
    }
    fn inv_logit(self) -> Self {
        let u = self.v.inv_logit();
        let d = u * (B::from_f64(1.0) - u);
        self.chain(u, d)
    }
    fn logit(self) -> Self {
        let x = self.v;
        let d = B::from_f64(1.0) / (x * (B::from_f64(1.0) - x));
        self.chain(x.logit(), d)
    }
    fn cos(self) -> Self {
        let s = (B::from_f64(std::f64::consts::FRAC_PI_2) - self.v).cos();
        self.chain(self.v.cos(), -s)
    }
    fn acos(self) -> Self {
        let x = self.v;
        let d = -(B::from_f64(1.0) / (B::from_f64(1.0) - x * x).sqrt());
        self.chain(x.acos(), d)
    }
    fn guard_floor(self, lo: f64) -> Self {
        if !self.v.br_cmp_c(crate::sens::num::BranchCmp::Ge, lo) {
            DualT::constant(lo)
        } else {
            self
        }
    }
    fn ln_gamma(self) -> Self {
        let x = self.v.val();
        // The derivative ψ lifted with its own ψ′ and ψ″ (tetragamma), so a second-order
        // base differentiates it exactly instead of seeing a truncated `0`.
        let d = self.v.lift2(
            crate::stats::special::digamma(x),
            crate::stats::special::trigamma(x),
            crate::stats::special::tetragamma(x),
        );
        self.chain(self.v.ln_gamma(), d)
    }
    fn jets_finite(self) -> bool {
        self.g.iter().all(|x| x.val().is_finite())
    }
    /// The tangents, and the base's own jets of the value and of each tangent.
    fn jet_max_abs(self) -> f64 {
        crate::sens::num::nan_max_abs(
            std::iter::once(self.v.jet_max_abs())
                .chain(self.g.iter().flat_map(|x| [x.val(), x.jet_max_abs()])),
        )
    }
    /// `φ(self)` with `φ = v`, `φ′ = d1`, `φ″ = d2`: the tangent factor `φ′(x)` is lifted
    /// on the base with derivative `φ″`, so a reverse sweep over the tangents is exact. Its own
    /// second derivative `φ‴` is unknown here: exact as `0` for an affine lift (`φ″ = 0`), and
    /// otherwise `NaN`, which a first-order base (`f64`, `R`) never reads and a second-order one
    /// turns into a visible decline rather than a silently truncated jet (#2093 review).
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        let d = self.v.lift2(d1, d2, if d2 == 0.0 { 0.0 } else { f64::NAN });
        self.chain(self.v.lift2(v, d1, d2), d)
    }
    /// The value lifted on the base with the derivatives it tracks, and the tangent factor
    /// `φ′` with one order fewer each — so a base carrying orders of its own is exact.
    fn lift3(self, v: f64, d1: f64, d2: f64, d3: f64) -> Self {
        self.chain(self.v.lift3(v, d1, d2, d3), self.v.lift2(d1, d2, d3))
    }
    fn lift4(self, v: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Self {
        self.chain(
            self.v.lift4(v, d1, d2, d3, d4),
            self.v.lift3(d1, d2, d3, d4),
        )
    }
}

/// Packed lower-triangle index of the symmetric pair `(k, l)`.
#[inline]
pub(crate) const fn tri(k: usize, l: usize) -> usize {
    if l <= k {
        k * (k + 1) / 2 + l
    } else {
        l * (l + 1) / 2 + k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #2093 review (round 2, finding 12): a constant exponent is the power rule, finite at a
    /// negative base where `exp(e·ln x)` is NaN (`(AGE − 40)^2` at AGE = 30). `x = −2 + t₀`,
    /// `x²`: value 4, first derivative `2x = −4`, second `2`. The straddle is a non-constant
    /// exponent at the same base, which keeps the `exp(e·ln x)` route (NaN there).
    #[test]
    fn a_constant_exponent_is_the_power_rule_at_a_negative_base() {
        let x = DualT::<2, f64> {
            v: -2.0,
            g: [1.0, 0.5],
        };
        let y = x.pow(DualT::constant(2.0));
        assert_eq!(y.v, 4.0);
        assert_eq!(y.g, [-4.0, -2.0]);
        let live = DualT::<2, f64> {
            v: 2.0,
            g: [0.0, 1.0],
        };
        assert!(x.pow(live).v.is_nan(), "a live exponent keeps exp(e·ln x)");
    }
}
