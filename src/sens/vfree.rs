//! A tape-free forward dual for the adjoint's η-only recordings.
//!
//! [`DualF<N, M>`] is a [`DualV<N, M>`] that can never be on the tape: the value, its `N`
//! η-tangents and the `M`-entry η-Hessian payload, without the tape index. Every
//! operation runs `DualV`'s own untaped arithmetic (it builds a free `DualV`, applies the
//! operation, and keeps the parts), so the jets are bit-identical to a tape-free `DualV`
//! recording. What it saves is the index: with no `u32` beside the lanes and no runtime
//! test of it, a value stays in registers. On the inner loop's stiff-ODE recordings a
//! first-order `DualV<3, 0>` ran ~15× slower per operation than a plain dual of the same
//! width (the struct round-tripped through memory, a store-forwarding stall per op).
//!
//! Used by `canonical_jets` (the η-Jacobian, the exact inner Hessian and the quadrature
//! anchors' jets); the outer gradient's recordings, which tape θ, keep `DualV`.
#![allow(clippy::needless_range_loop)]

use super::num::PkNum;
use super::vtape::DualV;
use std::ops::{Add, Div, Mul, Neg, Sub};

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
    /// An input with value, η-tangents and Hessian payload.
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
    #[inline(always)]
    fn up(self) -> DualV<N, M> {
        DualV::free(self.v, self.g, self.h)
    }
    #[inline(always)]
    fn down(d: DualV<N, M>) -> Self {
        DualF {
            v: d.v,
            g: d.g,
            h: d.h,
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
    #[inline(always)]
    pub(crate) fn add_c(self, c: f64) -> Self {
        Self::down(self.up().add_c(c))
    }
    #[inline(always)]
    pub(crate) fn sub_c(self, c: f64) -> Self {
        Self::down(self.up().sub_c(c))
    }
    #[inline(always)]
    pub(crate) fn rsub_c(self, c: f64) -> Self {
        Self::down(self.up().rsub_c(c))
    }
    #[inline(always)]
    pub(crate) fn mul_c(self, c: f64) -> Self {
        Self::down(self.up().mul_c(c))
    }
    #[inline(always)]
    pub(crate) fn div_c(self, c: f64) -> Self {
        Self::down(self.up().div_c(c))
    }
}

macro_rules! binop {
    ($tr:ident, $f:ident) => {
        impl<const N: usize, const M: usize> $tr for DualF<N, M> {
            type Output = Self;
            #[inline(always)]
            fn $f(self, o: Self) -> Self {
                Self::down(self.up().$f(o.up()))
            }
        }
    };
}
binop!(Add, add);
binop!(Sub, sub);
binop!(Mul, mul);
binop!(Div, div);

impl<const N: usize, const M: usize> Neg for DualF<N, M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::down(-self.up())
    }
}

macro_rules! unary {
    ($($f:ident),*) => {
        $(
            #[inline]
            fn $f(self) -> Self {
                Self::down(self.up().$f())
            }
        )*
    };
}

impl<const N: usize, const M: usize> PkNum for DualF<N, M> {
    const SECOND_ORDER: bool = <DualV<N, M> as PkNum>::SECOND_ORDER;
    /// No tape: the jet's own order.
    const JET_ORDER: usize = if M > 0 { 2 } else { 1 };
    const WORK: usize = 1 + N + M;
    const TAYLOR2_EXACT: bool = <DualV<N, M> as PkNum>::TAYLOR2_EXACT;
    const ETA_HESSIAN: bool = <DualV<N, M> as PkNum>::ETA_HESSIAN;
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
        Self::down(<DualV<N, M> as PkNum>::var(x, dim))
    }
    #[inline]
    fn val(self) -> f64 {
        self.v
    }
    unary!(exp, ln, sqrt, abs, inv_logit, logit, cos, acos, ln_gamma);
    #[inline]
    fn pow(self, e: Self) -> Self {
        Self::down(self.up().pow(e.up()))
    }
    #[inline]
    fn guard_floor(self, lo: f64) -> Self {
        Self::down(self.up().guard_floor(lo))
    }
    #[inline]
    fn jets_finite(self) -> bool {
        self.up().jets_finite()
    }
    #[inline]
    fn jet_max_abs(self) -> f64 {
        self.up().jet_max_abs()
    }
    #[inline]
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        Self::down(self.up().lift2(v, d1, d2))
    }
    #[inline]
    fn lift3(self, v: f64, d1: f64, d2: f64, d3: f64) -> Self {
        Self::down(self.up().lift3(v, d1, d2, d3))
    }
    #[inline]
    fn lift4(self, v: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Self {
        Self::down(self.up().lift4(v, d1, d2, d3, d4))
    }
    /// [`DualV::lincomb`]'s untaped arm, term for term (no slice of `DualV`s to build).
    /// Inlined, so the stage's constant coefficients unroll its loops.
    #[inline(always)]
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
    /// `DualV<N, M>`'s untaped arithmetic, which is what every operation here runs.
    const LANES: Option<crate::sens::num::vm::Layout> = Some(crate::sens::num::vm::Layout {
        n: N as u16,
        m: M as u16,
        th: false,
        wide: DualV::<N, M>::WIDE,
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
