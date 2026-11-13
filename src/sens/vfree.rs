//! A forward dual, the tests' independent derivative reference (#2786).
//!
//! [`DualF<N, M>`] carries a value, its `N` tangents and, with `M = N(N+1)/2`, the packed
//! Hessian `h[tri(k, l)] = ∂²/∂t_k∂t_l` (`M = 0`: first order). Every operation is the
//! textbook chain rule on those parts, written here once and shared with nothing the
//! production path runs: the plans' recordings (`parser::kernel_jit`) are checked against
//! it, so a defect in either shows as a disagreement.
#![allow(clippy::needless_range_loop)]

use super::num::PkNum;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Packed lower-triangle index of the symmetric pair `(k, l)`.
#[inline]
pub(crate) const fn tri(k: usize, l: usize) -> usize {
    if l <= k {
        k * (k + 1) / 2 + l
    } else {
        l * (l + 1) / 2 + k
    }
}

/// The packed Hessian entries of row `k` a dual carries: `(k, 0..=k)`, none at first order.
#[inline]
#[allow(clippy::reversed_empty_ranges)]
const fn hrange<const M: usize>(k: usize) -> std::ops::RangeInclusive<usize> {
    if M == 0 {
        1..=0
    } else {
        0..=k
    }
}

/// See the module docs.
///
/// `#[repr(C)]`: the RHS's native code (`parser::rhs_jit`) reads and writes a `DualF<N, M>` as
/// `1 + N + M` contiguous `f64`s, value first.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DualF<const N: usize, const M: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [f64; M],
}

impl<const N: usize, const M: usize> DualF<N, M> {
    /// Whether the native code runs this width out of line ([`crate::sens::num::outlined`]),
    /// where a constant operand takes the generic operation.
    pub(crate) const WIDE: bool = crate::sens::num::outlined(N, std::mem::size_of::<Self>());
    /// An input with value, tangents and Hessian payload.
    #[inline]
    pub fn free(v: f64, g: [f64; N], h: [f64; M]) -> Self {
        DualF { v, g, h }
    }
    #[inline]
    pub fn constant(v: f64) -> Self {
        DualF {
            v,
            g: [0.0; N],
            h: [0.0; M],
        }
    }
    /// Bit-identity of value, tangents and payload (the bytecode VM's no-op check).
    pub fn same_bits(&self, other: &Self) -> bool {
        self.v.to_bits() == other.v.to_bits()
            && self
                .g
                .iter()
                .zip(&other.g)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self
                .h
                .iter()
                .zip(&other.h)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
    /// `φ(self)` with `φ = hv`, `φ′ = h1`, `φ″ = h2`.
    #[inline]
    fn unary(self, hv: f64, h1: f64, h2: f64) -> Self {
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = h1 * self.g[k];
            for l in hrange::<M>(k) {
                let t = tri(k, l);
                h[t] = h1 * self.h[t] + h2 * self.g[k] * self.g[l];
            }
        }
        DualF { v: hv, g, h }
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    pub(crate) fn add_c(self, c: f64) -> Self {
        DualF {
            v: self.v + c,
            ..self
        }
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    pub(crate) fn sub_c(self, c: f64) -> Self {
        DualF {
            v: self.v - c,
            ..self
        }
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    pub(crate) fn rsub_c(self, c: f64) -> Self {
        Self::constant(c) - self
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    pub(crate) fn mul_c(self, c: f64) -> Self {
        if Self::WIDE {
            return self * Self::constant(c);
        }
        let mut r = self;
        r.v *= c;
        r.g.iter_mut().for_each(|x| *x *= c);
        r.h.iter_mut().for_each(|x| *x *= c);
        r
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    pub(crate) fn div_c(self, c: f64) -> Self {
        if Self::WIDE {
            return self / Self::constant(c);
        }
        let ib = 1.0 / c;
        let mut r = self;
        r.v /= c;
        r.g.iter_mut().for_each(|x| *x *= ib);
        r.h.iter_mut().for_each(|x| *x *= ib);
        r
    }
}

impl<const N: usize, const M: usize> Add for DualF<N, M> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut r = self;
        r.v += o.v;
        for k in 0..N {
            r.g[k] += o.g[k];
        }
        for t in 0..M {
            r.h[t] += o.h[t];
        }
        r
    }
}

impl<const N: usize, const M: usize> Sub for DualF<N, M> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut r = self;
        r.v -= o.v;
        for k in 0..N {
            r.g[k] -= o.g[k];
        }
        for t in 0..M {
            r.h[t] -= o.h[t];
        }
        r
    }
}

impl<const N: usize, const M: usize> Mul for DualF<N, M> {
    type Output = Self;
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = self.g[k] * o.v + self.v * o.g[k];
            for l in hrange::<M>(k) {
                let t = tri(k, l);
                h[t] = self.h[t] * o.v + self.v * o.h[t] + self.g[k] * o.g[l] + self.g[l] * o.g[k];
            }
        }
        DualF {
            v: self.v * o.v,
            g,
            h,
        }
    }
}

impl<const N: usize, const M: usize> Div for DualF<N, M> {
    type Output = Self;
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline]
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        let ib = 1.0 / o.v;
        let mut g = [0.0; N];
        for k in 0..N {
            g[k] = (self.g[k] - q * o.g[k]) * ib;
        }
        // c·b = a  ⇒  c_kl = (a_kl − c_k b_l − c_l b_k − c b_kl)/b
        let mut h = [0.0; M];
        for k in 0..N {
            for l in hrange::<M>(k) {
                let t = tri(k, l);
                h[t] = (self.h[t] - g[k] * o.g[l] - g[l] * o.g[k] - q * o.h[t]) * ib;
            }
        }
        DualF { v: q, g, h }
    }
}

impl<const N: usize, const M: usize> Neg for DualF<N, M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut r = self;
        r.v = -r.v;
        r.g.iter_mut().for_each(|x| *x = -*x);
        r.h.iter_mut().for_each(|x| *x = -*x);
        r
    }
}

impl<const N: usize, const M: usize> PkNum for DualF<N, M> {
    const SECOND_ORDER: bool = true;
    const JET_ORDER: usize = if M > 0 { 2 } else { 1 };
    const TAYLOR2_EXACT: bool = true;
    const ETA_HESSIAN: bool = M > 0;
    #[inline]
    fn same_bits_untaped(&self, other: &Self) -> Option<bool> {
        Some(DualF::same_bits(self, other))
    }
    #[inline]
    fn is_constant(self) -> bool {
        self.g.iter().chain(&self.h).all(|&d| d == 0.0)
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        DualF::constant(x)
    }
    #[inline]
    fn var(x: f64, dim: usize) -> Self {
        let mut d = DualF::constant(x);
        if dim < N {
            d.g[dim] = 1.0;
        }
        d
    }
    #[inline]
    fn val(self) -> f64 {
        self.v
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.unary(e, e, e)
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    fn ln(self) -> Self {
        let x = self.v;
        self.unary(x.ln(), 1.0 / x, -1.0 / (x * x))
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.unary(s, 0.5 / s, -0.25 / (s * s * s))
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    fn pow(self, e: Self) -> Self {
        if e.is_constant() {
            let (a, n) = (self.v, e.v);
            if a == 0.0 {
                return DualF::constant(a.powf(n));
            }
            return self.unary(
                a.powf(n),
                n * a.powf(n - 1.0),
                n * (n - 1.0) * a.powf(n - 2.0),
            );
        }
        (e * self.ln()).exp()
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    fn abs(self) -> Self {
        if self.v >= 0.0 {
            self
        } else {
            -self
        }
    }
    fn inv_logit(self) -> Self {
        let u = 1.0 / (1.0 + (-self.v).exp());
        let d1 = u * (1.0 - u);
        self.unary(u, d1, d1 * (1.0 - 2.0 * u))
    }
    fn logit(self) -> Self {
        let x = self.v;
        let d1 = 1.0 / (x * (1.0 - x));
        self.unary((x / (1.0 - x)).ln(), d1, -(1.0 - 2.0 * x) * d1 * d1)
    }
    fn cos(self) -> Self {
        let x = self.v;
        self.unary(x.cos(), -x.sin(), -x.cos())
    }
    fn acos(self) -> Self {
        let x = self.v;
        let s = 1.0 - x * x;
        self.unary(x.acos(), -1.0 / s.sqrt(), -x / (s * s.sqrt()))
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    fn guard_floor(self, lo: f64) -> Self {
        // `!(v >= lo)` floors a NaN too, matching `f64::max` on the value path.
        if !(self.v >= lo) {
            DualF::constant(lo)
        } else {
            self
        }
    }
    fn ln_gamma(self) -> Self {
        let x = self.v;
        self.unary(
            crate::stats::special::ln_gamma(x),
            crate::stats::special::digamma(x),
            crate::stats::special::trigamma(x),
        )
    }
    fn jets_finite(self) -> bool {
        self.g.iter().all(|x| x.is_finite()) && self.h.iter().all(|x| x.is_finite())
    }
    fn jet_max_abs(self) -> f64 {
        crate::sens::num::nan_max_abs(self.g.iter().chain(&self.h).copied())
    }
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        self.unary(v, d1, d2)
    }
    fn lift3(self, v: f64, d1: f64, d2: f64, _d3: f64) -> Self {
        self.unary(v, d1, d2)
    }
    fn lift4(self, v: f64, d1: f64, d2: f64, _d3: f64, _d4: f64) -> Self {
        self.unary(v, d1, d2)
    }
    /// `u + h·Σ cⱼkⱼ`, part for part.
    fn lincomb(u: Self, h: f64, cs: &[f64], ks: &[Self]) -> Self {
        let mut sv = 0.0;
        for j in 0..cs.len() {
            sv += cs[j] * ks[j].v;
        }
        let v = u.v + h * sv;
        let mut g = [0.0; N];
        for k in 0..N {
            let mut sg = 0.0;
            for j in 0..cs.len() {
                sg += cs[j] * ks[j].g[k];
            }
            g[k] = u.g[k] + h * sg;
        }
        let mut hh = [0.0; M];
        for t in 0..M {
            let mut sh = 0.0;
            for j in 0..cs.len() {
                sh += cs[j] * ks[j].h[t];
            }
            hh[t] = u.h[t] + h * sh;
        }
        DualF { v, g, h: hh }
    }
}

impl<const N: usize, const M: usize> crate::sens::num::vm::VmArith for DualF<N, M> {
    const LANES: Option<crate::sens::num::vm::Layout> = Some(crate::sens::num::vm::Layout {
        n: N as u16,
        m: M as u16,
        th: false,
        wide: Self::WIDE,
        tan: 0,
    });
    const CONTIGUOUS: bool = true;
    #[inline]
    fn add_in(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
    #[inline]
    fn sub_in(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
    #[inline]
    fn rsub_in(&mut self, rhs: &Self) {
        *self = *rhs - *self;
    }
    #[inline]
    fn mul_in(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
    #[inline]
    fn div_in(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
    #[inline]
    fn add_scalar_in(&mut self, c: f64) {
        *self = self.add_c(c);
    }
    #[inline]
    fn sub_scalar_in(&mut self, c: f64) {
        *self = self.sub_c(c);
    }
    #[inline]
    fn rsub_scalar_in(&mut self, c: f64) {
        *self = self.rsub_c(c);
    }
    #[inline]
    fn mul_scalar_in(&mut self, c: f64) {
        *self = self.mul_c(c);
    }
    #[inline]
    fn div_scalar_in(&mut self, c: f64) {
        *self = self.div_c(c);
    }
    #[inline]
    fn same_bits(&self, other: &Self) -> bool {
        DualF::same_bits(self, other)
    }
}
