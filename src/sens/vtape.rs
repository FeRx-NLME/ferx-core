//! Vector tape for the adjoint FOCEI gradient (`sens::ode_provider::adjoint`): reverse mode over a first-order
//! forward dual in η.
//!
//! [`DualV<N, M>`] is value + `N` η-tangents + a packed symmetric η-Hessian payload
//! (`M = N(N+1)/2`). Value and tangents are recorded on a thread-local tape, one node
//! per operation (op code, parent ids, the node's own value and tangents). The
//! Hessian payload is propagated forward only, never taped. It exists so that the
//! exact inner Hessian `H` and `∇_η F` are known *before* the sweep, which then needs
//! a single adjoint lane. Local partials are recomputed in the sweep from stored
//! forward values, so a dual multiply is one `16 + 8·(N+1)`-byte record.
//!
//! Three hooks on [`PkNum`] keep the integrator's tape small:
//! * `lincomb` turns an RK stage combination into one node;
//! * the LU hooks make a Rosenbrock solve a block node (f64 LU of the value part,
//!   transposed solves in the sweep);
//! * `tape_mark` / `tape_rollback` drop rejected ODE step attempts.
//!
//! Adjoint layout: `adj[(node·(N+1) + comp)·L + lane]`, where comp 0 is the value and
//! `1 + k` is tangent `k`.
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use super::num::PkNum;
use std::cell::UnsafeCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub(crate) const NONE: u32 = u32::MAX;

/// Packed lower-triangle index of the symmetric pair `(k, l)`.
#[inline(always)]
pub(crate) const fn tri(k: usize, l: usize) -> usize {
    if l <= k {
        k * (k + 1) / 2 + l
    } else {
        l * (l + 1) / 2 + k
    }
}

/// The second-order entries a dual carries for row `k`: the packed lower triangle
/// `(k, 0..=k)` of a full η-Hessian, or with `TH` (directional mode, `M = N`) only the
/// pure second derivative `(k, k)` along tangent direction `k` — the diagonal of a
/// product or chain rule never reads an off-diagonal entry, so it is a closed algebra.
/// A first-order dual (`M = 0`) carries none.
#[inline(always)]
#[allow(clippy::reversed_empty_ranges)]
pub(crate) const fn hrange<const TH: bool, const M: usize>(
    k: usize,
) -> std::ops::RangeInclusive<usize> {
    if M == 0 {
        1..=0
    } else if TH {
        k..=k
    } else {
        0..=k
    }
}

/// Storage index of second-order entry `(k, l)` ([`hrange`]).
#[inline(always)]
pub(crate) const fn hidx<const TH: bool>(k: usize, l: usize) -> usize {
    if TH {
        k
    } else {
        tri(k, l)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Op {
    Leaf,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Scale,
    Shift,
    Unary,
    Lin,
    Solve,
}

#[derive(Clone, Copy)]
struct VNode {
    op: Op,
    a: u32,
    b: u32,
    x: u32,
}

/// A factored `W` (value part, f64) plus its first-order values and tape ids, shared by
/// every solve against it.
struct VLu {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
    /// `w[c·n² + idx]`: values of W, component `c` (0 = value, 1.. = tangents).
    w: Vec<f64>,
    widx: Vec<u32>,
    /// `wh[t·n² + idx]`: packed η-Hessian payload of W (forward only).
    wh: Vec<f64>,
}

/// One solve `x = W⁻¹b`. Its outputs are the consecutive nodes `o0 .. o0 + n`.
struct VExt {
    o0: u32,
    lu: u32,
    bidx: Vec<u32>,
    /// `x[c·n + i]`.
    x: Vec<f64>,
}

struct VTape {
    nc: usize,
    nodes: Vec<VNode>,
    vals: Vec<f64>,
    auxf: Vec<f64>,
    auxi: Vec<u32>,
    adj: Vec<f64>,
    lus: Vec<VLu>,
    exts: Vec<VExt>,
    mark: Mark,
    /// Direct-mapped common-subexpression cache: `(key, node)`; `node == NONE` empty.
    cse: Vec<(u64, u64, u32)>,
}

const CSE_BITS: usize = 12;

/// Lengths of every growable buffer at the last [`PkNum::tape_mark`].
#[derive(Clone, Copy, Default)]
struct Mark {
    nodes: usize,
    auxf: usize,
    auxi: usize,
    lus: usize,
    exts: usize,
}

thread_local! {
    static VT: UnsafeCell<VTape> = const { UnsafeCell::new(VTape {
        nc: 0, nodes: Vec::new(), vals: Vec::new(), auxf: Vec::new(), auxi: Vec::new(),
        adj: Vec::new(), lus: Vec::new(), exts: Vec::new(),
        mark: Mark { nodes: 0, auxf: 0, auxi: 0, lus: 0, exts: 0 },
        cse: Vec::new(),
    }) };
}

#[inline(always)]
fn vt() -> &'static mut VTape {
    // SAFETY: the tape is thread-local. Every function in this module takes the
    // reference, uses it, and drops it before calling back into the module, so no two
    // live `&mut` overlap.
    VT.with(|t| unsafe { &mut *t.get() })
}

/// Start a new recording for dual width `N` (`nc = N + 1`). Keeps allocations.
pub(crate) fn vtape_reset(nc: usize) {
    let t = vt();
    t.nc = nc;
    t.nodes.clear();
    t.vals.clear();
    t.auxf.clear();
    t.auxi.clear();
    t.lus.clear();
    t.exts.clear();
    t.mark = Mark::default();
    // The CSE cache is not cleared between recordings: a hit is verified against the
    // *current* tape's node (op, parents, value), so a stale entry can only match a node
    // that is the same pure op on the same parents — exactly what may be reused.
    if t.cse.is_empty() {
        t.cse.resize(1 << CSE_BITS, (0, 0, NONE));
    }
}

#[cfg(test)]
pub(crate) fn vtape_len() -> usize {
    vt().nodes.len()
}

fn vmark() -> usize {
    let t = vt();
    t.mark = Mark {
        nodes: t.nodes.len(),
        auxf: t.auxf.len(),
        auxi: t.auxi.len(),
        lus: t.lus.len(),
        exts: t.exts.len(),
    };
    t.mark.nodes
}

fn vrollback(mark: usize) {
    let t = vt();
    // Only the most recent mark can be restored (one stepper attempt at a time).
    let m = t.mark;
    if m.nodes != mark {
        return;
    }
    t.nodes.truncate(m.nodes);
    t.vals.truncate(m.nodes * t.nc);
    t.auxf.truncate(m.auxf);
    t.auxi.truncate(m.auxi);
    t.lus.truncate(m.lus);
    t.exts.truncate(m.exts);
}

/// Record a pure binary/unary op, reusing an existing node when the same op was already
/// applied to the same parents with the same result (`v`, and `w` = an op-specific
/// discriminator such as φ′). Pure ops on identical inputs give identical values and
/// tangents, so the reuse is exact. A hit is verified against the node itself, so stale
/// entries (rolled-back nodes) can never alias.
#[inline(always)]
fn push_cse<const N: usize>(
    op: Op,
    a: u32,
    b: u32,
    x: u32,
    v: f64,
    w: f64,
    g: &[f64; N],
    hs: &[f64],
) -> u32 {
    let k1 = ((op as u64) << 56) ^ ((a as u64) << 28) ^ (b as u64);
    let k2 = v.to_bits() ^ w.to_bits().rotate_left(17);
    let hk = (k1 ^ k2.rotate_left(29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - CSE_BITS);
    let t = vt();
    let (e1, e2, id) = t.cse[hk as usize];
    if id != NONE && e1 == k1 && e2 == k2 && (id as usize) < t.nodes.len() {
        let nd = t.nodes[id as usize];
        let same_val = t.vals[id as usize * t.nc].to_bits() == v.to_bits();
        let same_aux = match op {
            Op::Unary | Op::Scale => {
                t.auxf.get(nd.x as usize).map(|c| c.to_bits()) == Some(w.to_bits())
            }
            _ => true,
        };
        if nd.op == op && nd.a == a && nd.b == b && same_val && same_aux {
            return id;
        }
    }
    let id = push::<N>(op, a, b, x, v, g, hs);
    vt().cse[hk as usize] = (k1, k2, id);
    id
}

#[inline(always)]
fn push<const N: usize>(op: Op, a: u32, b: u32, x: u32, v: f64, g: &[f64; N], h: &[f64]) -> u32 {
    let t = vt();
    debug_assert!(t.nc == N + 1 || t.nc == N + 1 + h.len());
    t.nodes.push(VNode { op, a, b, x });
    t.vals.push(v);
    t.vals.extend_from_slice(g);
    // A taped-Hessian recording (`vtape_reset(1 + N + M)`) stores `h` with the tangents.
    if t.nc > N + 1 {
        t.vals.extend_from_slice(h);
    }
    (t.nodes.len() - 1) as u32
}

/// Forward dual whose value and tangents live on the vector tape, with second-order
/// entries alongside. With `TH = false` (FOCE/FOCEI) they are the packed symmetric
/// η-Hessian `h[tri(k, l)] = ∂²/∂η_k∂η_l` (`M = N(N+1)/2`), an untaped payload. With
/// `TH = true` (Laplace/AGQ) they are the pure second derivatives along the `N` tangent
/// directions (`M = N`, `h[k] = ∂²/∂t_k²`), and taped, so a reverse sweep reaches
/// quantities built from them ([`vsweep_h`]).
#[derive(Clone, Copy, Debug)]
pub struct DualV<const N: usize, const M: usize, const TH: bool = false> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [f64; M],
    pub i: u32,
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    /// Whether this width's arithmetic runs out of line ([`crate::sens::num::OUTLINE_BYTES`]).
    const WIDE: bool = std::mem::size_of::<Self>() > crate::sens::num::OUTLINE_BYTES;
    #[inline(always)]
    pub fn constant(v: f64) -> Self {
        DualV {
            v,
            g: [0.0; N],
            h: [0.0; M],
            i: NONE,
        }
    }
    /// An input that is not on the tape: a recording made only of these (and constants)
    /// is a plain forward dual — the η-Jacobian (`M = 0`) or the η-jets `(f, G, Q)` —
    /// with no tape traffic at all.
    #[inline(always)]
    pub fn free(v: f64, g: [f64; N], h: [f64; M]) -> Self {
        DualV { v, g, h, i: NONE }
    }
    /// An independent input (tape leaf) with value, tangents and Hessian payload.
    pub fn leaf(v: f64, g: [f64; N], h: [f64; M]) -> Self {
        let i = push::<N>(Op::Leaf, NONE, NONE, NONE, v, &g, &h);
        DualV { v, g, h, i }
    }
    /// Bit-identity of value, tangents, Hessian payload and tape node — the bytecode
    /// VM's no-op check ([`crate::sens::num::vm::VmArith::same_bits`]).
    pub fn same_bits(&self, other: &Self) -> bool {
        self.i == other.i
            && self.v.to_bits() == other.v.to_bits()
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
    pub fn hess(&self, k: usize, l: usize) -> f64 {
        self.h[tri(k, l)]
    }
    #[inline(always)]
    fn zero_tan(&self) -> bool {
        self.g.iter().all(|x| *x == 0.0)
    }
    /// Give a constant that carries non-zero tangents a tape identity (rare).
    #[inline]
    fn mat(self) -> Self {
        if self.i != NONE {
            return self;
        }
        DualV::leaf(self.v, self.g, self.h)
    }
    /// `φ(self)` with `φ = hv`, `φ′ = h1`, `φ″ = h2`; `h3` gives `φ‴`, which only a
    /// taped-Hessian recording stores (its sweep differentiates the taped `h`).
    #[inline(always)]
    fn unary(self, hv: f64, h1: f64, h2: f64, h3: impl FnOnce() -> f64) -> Self {
        if Self::WIDE {
            // `φ‴` only matters to a taped-Hessian recording; evaluated eagerly here so
            // the out-of-line copy is one function, not one per closure type.
            self.unary_wide(hv, h1, h2, if TH { h3() } else { 0.0 })
        } else {
            self.unary_body(hv, h1, h2, h3)
        }
    }
    /// [`Self::unary`]'s body out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn unary_wide(self, hv: f64, h1: f64, h2: f64, h3: f64) -> Self {
        self.unary_body(hv, h1, h2, || h3)
    }
    #[inline(always)]
    fn unary_body(self, hv: f64, h1: f64, h2: f64, h3: impl FnOnce() -> f64) -> Self {
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = h1 * self.g[k];
            for l in hrange::<TH, M>(k) {
                let t = hidx::<TH>(k, l);
                h[t] = h1 * self.h[t] + h2 * self.g[k] * self.g[l];
            }
        }
        if self.i == NONE {
            return DualV {
                v: hv,
                g,
                h,
                i: NONE,
            };
        }
        let x = vt().auxf.len() as u32;
        let i = push_cse::<N>(Op::Unary, self.i, NONE, x, hv, h1, &g, &h);
        if i as usize == vt().nodes.len() - 1 && vt().nodes[i as usize].x == x {
            let t = vt();
            t.auxf.push(h1);
            t.auxf.push(h2);
            if TH {
                t.auxf.push(h3());
            }
        }
        DualV { v: hv, g, h, i }
    }
    #[inline(always)]
    fn scaled(self, c: f64, v: f64, g: [f64; N], h: [f64; M]) -> Self {
        if Self::WIDE {
            self.scaled_wide(c, v, g, h)
        } else {
            self.scaled_body(c, v, g, h)
        }
    }
    /// [`Self::scaled`]'s body out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn scaled_wide(self, c: f64, v: f64, g: [f64; N], h: [f64; M]) -> Self {
        self.scaled_body(c, v, g, h)
    }
    #[inline(always)]
    fn scaled_body(self, c: f64, v: f64, g: [f64; N], h: [f64; M]) -> Self {
        let x = vt().auxf.len() as u32;
        let i = push_cse::<N>(Op::Scale, self.i, NONE, x, v, c, &g, &h);
        if i as usize == vt().nodes.len() - 1 && vt().nodes[i as usize].x == x {
            vt().auxf.push(c);
        }
        DualV { v, g, h, i }
    }
}

impl<const N: usize, const M: usize, const TH: bool> Add for DualV<N, M, TH> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        if Self::WIDE {
            self.add_wide(o)
        } else {
            self.add_body(o)
        }
    }
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    #[inline(always)]
    fn add_body(self, o: Self) -> Self {
        let v = self.v + o.v;
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = self.g[k] + o.g[k];
        }
        for t in 0..M {
            h[t] = self.h[t] + o.h[t];
        }
        let i = match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            (false, true) => push::<N>(Op::Shift, self.i, NONE, NONE, v, &g, &h),
            (true, false) => push::<N>(Op::Shift, o.i, NONE, NONE, v, &g, &h),
            (false, false) => push_cse::<N>(Op::Add, self.i, o.i, NONE, v, 0.0, &g, &h),
        };
        DualV { v, g, h, i }
    }
    /// [`Self::add_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn add_wide(self, o: Self) -> Self {
        self.add_body(o)
    }
}

impl<const N: usize, const M: usize, const TH: bool> Sub for DualV<N, M, TH> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        if Self::WIDE {
            self.sub_wide(o)
        } else {
            self.sub_body(o)
        }
    }
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    #[inline(always)]
    fn sub_body(self, o: Self) -> Self {
        let v = self.v - o.v;
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = self.g[k] - o.g[k];
        }
        for t in 0..M {
            h[t] = self.h[t] - o.h[t];
        }
        let i = match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            (false, true) => push::<N>(Op::Shift, self.i, NONE, NONE, v, &g, &h),
            (true, false) => push_cse::<N>(Op::Neg, o.i, NONE, NONE, v, 0.0, &g, &h),
            (false, false) => push_cse::<N>(Op::Sub, self.i, o.i, NONE, v, 0.0, &g, &h),
        };
        DualV { v, g, h, i }
    }
    /// [`Self::sub_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn sub_wide(self, o: Self) -> Self {
        self.sub_body(o)
    }
}

impl<const N: usize, const M: usize, const TH: bool> Mul for DualV<N, M, TH> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        if Self::WIDE {
            self.mul_wide(o)
        } else {
            self.mul_body(o)
        }
    }
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    #[inline(always)]
    fn mul_body(self, o: Self) -> Self {
        let v = self.v * o.v;
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = self.g[k] * o.v + self.v * o.g[k];
            for l in hrange::<TH, M>(k) {
                let t = hidx::<TH>(k, l);
                h[t] = self.h[t] * o.v + self.v * o.h[t] + self.g[k] * o.g[l] + self.g[l] * o.g[k];
            }
        }
        match (self.i == NONE, o.i == NONE) {
            (true, true) => DualV { v, g, h, i: NONE },
            (true, false) if self.zero_tan() => {
                if self.v == 0.0 {
                    return DualV { v, g, h, i: NONE };
                }
                o.scaled(self.v, v, g, h)
            }
            (false, true) if o.zero_tan() => {
                if o.v == 0.0 {
                    return DualV { v, g, h, i: NONE };
                }
                self.scaled(o.v, v, g, h)
            }
            _ => {
                let (a, b) = (self.mat(), o.mat());
                let i = push_cse::<N>(Op::Mul, a.i, b.i, NONE, v, 0.0, &g, &h);
                DualV { v, g, h, i }
            }
        }
    }
    /// [`Self::mul_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn mul_wide(self, o: Self) -> Self {
        self.mul_body(o)
    }
}

impl<const N: usize, const M: usize, const TH: bool> Div for DualV<N, M, TH> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        if Self::WIDE {
            self.div_wide(o)
        } else {
            self.div_body(o)
        }
    }
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    #[inline(always)]
    fn div_body(self, o: Self) -> Self {
        let q = self.v / o.v;
        let ib = 1.0 / o.v;
        let mut g = [0.0; N];
        for k in 0..N {
            g[k] = (self.g[k] - q * o.g[k]) * ib;
        }
        // c·b = a  ⇒  c_kl = (a_kl − c_k b_l − c_l b_k − c b_kl)/b
        let mut h = [0.0; M];
        for k in 0..N {
            for l in hrange::<TH, M>(k) {
                let t = hidx::<TH>(k, l);
                h[t] = (self.h[t] - g[k] * o.g[l] - g[l] * o.g[k] - q * o.h[t]) * ib;
            }
        }
        match (self.i == NONE, o.i == NONE) {
            (true, true) => DualV {
                v: q,
                g,
                h,
                i: NONE,
            },
            (false, true) if o.zero_tan() => self.scaled(ib, q, g, h),
            _ => {
                let (a, b) = (self.mat(), o.mat());
                let i = push_cse::<N>(Op::Div, a.i, b.i, NONE, q, 0.0, &g, &h);
                DualV { v: q, g, h, i }
            }
        }
    }
    /// [`Self::div_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn div_wide(self, o: Self) -> Self {
        self.div_body(o)
    }
}

impl<const N: usize, const M: usize, const TH: bool> Neg for DualV<N, M, TH> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        if Self::WIDE {
            self.neg_wide()
        } else {
            self.neg_body()
        }
    }
}

impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    #[inline(always)]
    fn neg_body(self) -> Self {
        let mut g = [0.0; N];
        let mut h = [0.0; M];
        for k in 0..N {
            g[k] = -self.g[k];
        }
        for t in 0..M {
            h[t] = -self.h[t];
        }
        let i = if self.i == NONE {
            NONE
        } else {
            push_cse::<N>(Op::Neg, self.i, NONE, NONE, -self.v, 0.0, &g, &h)
        };
        DualV {
            v: -self.v,
            g,
            h,
            i,
        }
    }
    /// [`Self::neg_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn neg_wide(self) -> Self {
        self.neg_body()
    }
}

impl<const N: usize, const M: usize, const TH: bool> PkNum for DualV<N, M, TH> {
    const SECOND_ORDER: bool = true;
    fn from_f64(x: f64) -> Self {
        DualV::constant(x)
    }
    fn var(x: f64, dim: usize) -> Self {
        let mut d = DualV::constant(x);
        if dim < N {
            d.g[dim] = 1.0;
        }
        d
    }
    fn val(self) -> f64 {
        self.v
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.unary(e, e, e, || e)
    }
    fn ln(self) -> Self {
        let x = self.v;
        self.unary(x.ln(), 1.0 / x, -1.0 / (x * x), || 2.0 / (x * x * x))
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.unary(s, 0.5 / s, -0.25 / (s * s * s), || {
            0.375 / (s * s * s * s * s)
        })
    }
    fn pow(self, e: Self) -> Self {
        if e.i == NONE && e.zero_tan() {
            let (a, n) = (self.v, e.v);
            if a == 0.0 {
                return DualV::constant(a.powf(n));
            }
            return self.unary(
                a.powf(n),
                n * a.powf(n - 1.0),
                n * (n - 1.0) * a.powf(n - 2.0),
                || n * (n - 1.0) * (n - 2.0) * a.powf(n - 3.0),
            );
        }
        (e * self.ln()).exp()
    }
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
        self.unary(u, d1, d1 * (1.0 - 2.0 * u), || {
            d1 * (1.0 - 6.0 * u + 6.0 * u * u)
        })
    }
    fn logit(self) -> Self {
        let x = self.v;
        let d1 = 1.0 / (x * (1.0 - x));
        self.unary((x / (1.0 - x)).ln(), d1, -(1.0 - 2.0 * x) * d1 * d1, || {
            2.0 * (1.0 - 3.0 * x + 3.0 * x * x) * d1 * d1 * d1
        })
    }
    fn cos(self) -> Self {
        let x = self.v;
        self.unary(x.cos(), -x.sin(), -x.cos(), || x.sin())
    }
    fn acos(self) -> Self {
        let x = self.v;
        let s = 1.0 - x * x;
        self.unary(x.acos(), -1.0 / s.sqrt(), -x / (s * s.sqrt()), || {
            -(1.0 + 2.0 * x * x) / (s * s * s.sqrt())
        })
    }
    fn guard_floor(self, lo: f64) -> Self {
        // `!(v >= lo)` floors a NaN too, matching `f64::max` on the value path.
        if !(self.v >= lo) {
            DualV::constant(lo)
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
            || crate::stats::special::tetragamma(x),
        )
    }
    fn jets_finite(self) -> bool {
        self.g.iter().all(|x| x.is_finite()) && self.h.iter().all(|x| x.is_finite())
    }
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        // `lift2` carries no `φ‴`: a taped-Hessian sweep through it would be wrong, so it
        // poisons that sweep with NaN (non-finite → the caller's finiteness backstop)
        // rather than silently differentiate a truncated rule. No recording calls it.
        self.unary(v, d1, d2, || f64::NAN)
    }
    fn lincomb(u: Self, h: f64, cs: &[f64], ks: &[Self]) -> Self {
        // Same association as the default (`u + h·Σ cⱼkⱼ`), so values stay bit-identical
        // to the f64 trajectory.
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
        if u.i == NONE && ks.iter().all(|x| x.i == NONE) {
            return DualV {
                v,
                g,
                h: hh,
                i: NONE,
            };
        }
        let t = vt();
        let a = t.auxi.len() as u32;
        let x = t.auxf.len() as u32;
        let mut cnt = 0u32;
        if u.i != NONE {
            t.auxi.push(u.i);
            t.auxf.push(1.0);
            cnt += 1;
        }
        for j in 0..cs.len() {
            if ks[j].i != NONE {
                t.auxi.push(ks[j].i);
                t.auxf.push(h * cs[j]);
                cnt += 1;
            }
        }
        let i = push::<N>(Op::Lin, a, cnt, x, v, &g, &hh);
        DualV { v, g, h: hh, i }
    }
    fn tape_mark() -> usize {
        vmark()
    }
    fn tape_rollback(mark: usize) {
        vrollback(mark)
    }
    fn lu_factor_hook(m: &mut [Self], piv: &mut [usize], n: usize) -> Option<bool> {
        // The block solve node carries first-order rules only; a taped-Hessian recording
        // takes the generic elimination, op by op.
        // A tape-free recording (no operand on the tape) eliminates generically, as any
        // forward dual does.
        if TH || m.iter().all(|x| x.i == NONE) {
            return None;
        }
        let nn = n * n;
        let mut lu: Vec<f64> = m.iter().map(|x| x.v).collect();
        let mut p = vec![0usize; n];
        if !crate::sens::linsolve::lu_factor_in_place_g::<f64>(&mut lu, &mut p, n) {
            return Some(false);
        }
        let mut w = vec![0.0; (N + 1) * nn];
        let mut widx = vec![NONE; nn];
        let mut wh = vec![0.0; M * nn];
        for idx in 0..nn {
            w[idx] = m[idx].v;
            widx[idx] = m[idx].i;
            for k in 0..N {
                w[(k + 1) * nn + idx] = m[idx].g[k];
            }
            for t in 0..M {
                wh[t * nn + idx] = m[idx].h[t];
            }
        }
        let t = vt();
        t.lus.push(VLu {
            n,
            lu,
            piv: p,
            w,
            widx,
            wh,
        });
        // `m` is left unfactored; the solve hook reads the LU through this tagged id (a
        // generic factorization's `piv[0]` is a row index, never tagged).
        piv[0] = LU_TAG | (t.lus.len() - 1);
        Some(true)
    }
    fn lu_solve_hook(_m: &[Self], piv: &[usize], n: usize, b: &mut [Self]) -> bool {
        if piv[0] & LU_TAG == 0 {
            return false;
        }
        let id = piv[0] & !LU_TAG;
        let nn = n * n;
        let mut x = vec![0.0; (N + 1) * n];
        let mut bidx = vec![NONE; n];
        let mut xh = vec![0.0; M * n];
        {
            let t = vt();
            let rec = &t.lus[id];
            for i in 0..n {
                x[i] = b[i].v;
                bidx[i] = b[i].i;
            }
            crate::sens::linsolve::lu_solve_in_place_g::<f64>(&rec.lu, &rec.piv, n, &mut x[0..n]);
            let mut r = vec![0.0; n];
            // x_k = W0⁻¹(b_k − W_k x0)
            for k in 0..N {
                let c = k + 1;
                for i in 0..n {
                    let mut s = b[i].g[k];
                    for j in 0..n {
                        s -= rec.w[c * nn + i * n + j] * x[j];
                    }
                    r[i] = s;
                }
                crate::sens::linsolve::lu_solve_in_place_g::<f64>(&rec.lu, &rec.piv, n, &mut r);
                x[c * n..(c + 1) * n].copy_from_slice(&r);
            }
            // x_kl = W0⁻¹(b_kl − W_k x_l − W_l x_k − W_kl x0)
            for k in 0..N {
                for l in 0..=k {
                    let tt = tri(k, l);
                    for i in 0..n {
                        let mut s = b[i].h[tt];
                        for j in 0..n {
                            s -= rec.w[(k + 1) * nn + i * n + j] * x[(l + 1) * n + j]
                                + rec.w[(l + 1) * nn + i * n + j] * x[(k + 1) * n + j]
                                + rec.wh[tt * nn + i * n + j] * x[j];
                        }
                        r[i] = s;
                    }
                    crate::sens::linsolve::lu_solve_in_place_g::<f64>(&rec.lu, &rec.piv, n, &mut r);
                    xh[tt * n..(tt + 1) * n].copy_from_slice(&r);
                }
            }
        }
        let o0 = vt().nodes.len() as u32;
        for i in 0..n {
            let mut g = [0.0; N];
            let mut h = [0.0; M];
            for k in 0..N {
                g[k] = x[(k + 1) * n + i];
            }
            for t in 0..M {
                h[t] = xh[t * n + i];
            }
            let id_i = push::<N>(Op::Solve, NONE, NONE, NONE, x[i], &g, &h);
            b[i] = DualV {
                v: x[i],
                g,
                h,
                i: id_i,
            };
        }
        vt().exts.push(VExt {
            o0,
            lu: id as u32,
            bidx,
            x,
        });
        true
    }
}

fn lu_tsolve(rec: &VLu, c: &mut [f64]) {
    // Wᵀy = c with W⁻¹ = U⁻¹L⁻¹P  ⇒  y = Pᵀ L⁻ᵀ U⁻ᵀ c.
    let n = rec.n;
    let m = &rec.lu;
    for i in 0..n {
        let mut s = c[i];
        for j in 0..i {
            s -= m[j * n + i] * c[j];
        }
        c[i] = s / m[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = c[i];
        for j in (i + 1)..n {
            s -= m[j * n + i] * c[j];
        }
        c[i] = s;
    }
    for col in (0..n).rev() {
        let p = rec.piv[col];
        if p != col {
            c.swap(col, p);
        }
    }
}

/// Marks a pivot vector whose factorization the block LU hook holds.
const LU_TAG: usize = 1 << 62;

/// The most components a first-order sweep carries: value plus the widest dispatch
/// bucket's tangents (`sens::ode_provider::adjoint`'s 24).
const MAX_NC: usize = 25;

/// Zero the adjoint buffer for `L` lanes and return it for seeding.
pub(crate) fn vadj_init<const L: usize>() -> &'static mut [f64] {
    let t = vt();
    let n = t.nodes.len();
    t.adj.clear();
    t.adj.resize(n * t.nc * L, 0.0);
    &mut t.adj
}

#[inline(always)]
pub(crate) fn vadj_index<const N: usize, const L: usize>(
    node: u32,
    comp: usize,
    lane: usize,
) -> usize {
    (node as usize * (N + 1) + comp) * L + lane
}

/// Backward sweep with `L` lanes over nodes `0 .. top` (seeds placed via
/// [`vadj_init`]; nodes at or above `top` must carry no seed). Returns the adjoint buffer.
pub(crate) fn vsweep<const N: usize, const L: usize>(top: usize) -> &'static [f64] {
    let t = vt();
    let nc = N + 1;
    let stride = nc * L;
    let n = top.min(t.nodes.len());
    let mut next_ext = t.exts.partition_point(|e| (e.o0 as usize) < n);
    let vals = &t.vals;
    let adj = &mut t.adj;
    // One node's adjoints, `nc = N + 1` components (every one written before it is read).
    let mut cb = [[0.0f64; L]; MAX_NC];
    assert!(
        nc <= MAX_NC,
        "dual width {N} above the widest dispatch bucket"
    );
    for i in (0..n).rev() {
        let mut live = false;
        for c in 0..nc {
            for l in 0..L {
                let x = adj[i * stride + c * L + l];
                cb[c][l] = x;
                live |= x != 0.0;
            }
        }
        let nd = t.nodes[i];
        if live {
            match nd.op {
                Op::Leaf | Op::Solve => {}
                Op::Add => {
                    let (a, b) = (nd.a as usize * stride, nd.b as usize * stride);
                    for c in 0..nc {
                        for l in 0..L {
                            adj[a + c * L + l] += cb[c][l];
                            adj[b + c * L + l] += cb[c][l];
                        }
                    }
                }
                Op::Sub => {
                    let (a, b) = (nd.a as usize * stride, nd.b as usize * stride);
                    for c in 0..nc {
                        for l in 0..L {
                            adj[a + c * L + l] += cb[c][l];
                            adj[b + c * L + l] -= cb[c][l];
                        }
                    }
                }
                Op::Shift => {
                    let a = nd.a as usize * stride;
                    for c in 0..nc {
                        for l in 0..L {
                            adj[a + c * L + l] += cb[c][l];
                        }
                    }
                }
                Op::Neg => {
                    let a = nd.a as usize * stride;
                    for c in 0..nc {
                        for l in 0..L {
                            adj[a + c * L + l] -= cb[c][l];
                        }
                    }
                }
                Op::Scale => {
                    let s = t.auxf[nd.x as usize];
                    let a = nd.a as usize * stride;
                    for c in 0..nc {
                        for l in 0..L {
                            adj[a + c * L + l] += s * cb[c][l];
                        }
                    }
                }
                Op::Unary => {
                    let h1 = t.auxf[nd.x as usize];
                    let h2 = t.auxf[nd.x as usize + 1];
                    let av = &vals[nd.a as usize * nc..nd.a as usize * nc + nc];
                    let a = nd.a as usize * stride;
                    for l in 0..L {
                        let mut s = cb[0][l] * h1;
                        for k in 0..N {
                            let ck = cb[1 + k][l];
                            s += ck * h2 * av[1 + k];
                            adj[a + (1 + k) * L + l] += ck * h1;
                        }
                        adj[a + l] += s;
                    }
                }
                Op::Mul => {
                    let av = &vals[nd.a as usize * nc..nd.a as usize * nc + nc];
                    let bv = &vals[nd.b as usize * nc..nd.b as usize * nc + nc];
                    let (a, b) = (nd.a as usize * stride, nd.b as usize * stride);
                    for l in 0..L {
                        let mut sa = cb[0][l] * bv[0];
                        let mut sb = cb[0][l] * av[0];
                        for k in 0..N {
                            let ck = cb[1 + k][l];
                            sa += ck * bv[1 + k];
                            sb += ck * av[1 + k];
                            adj[a + (1 + k) * L + l] += ck * bv[0];
                            adj[b + (1 + k) * L + l] += ck * av[0];
                        }
                        adj[a + l] += sa;
                        adj[b + l] += sb;
                    }
                }
                Op::Div => {
                    let av = &vals[nd.a as usize * nc..nd.a as usize * nc + nc];
                    let bv = &vals[nd.b as usize * nc..nd.b as usize * nc + nc];
                    let q = vals[i * nc];
                    let ib = 1.0 / bv[0];
                    let ib2 = ib * ib;
                    let (a, b) = (nd.a as usize * stride, nd.b as usize * stride);
                    for l in 0..L {
                        let mut sa = cb[0][l] * ib;
                        let mut sb = -cb[0][l] * q * ib;
                        for k in 0..N {
                            let ck = cb[1 + k][l];
                            sa -= ck * bv[1 + k] * ib2;
                            sb += ck * (-av[1 + k] * ib2 + 2.0 * q * bv[1 + k] * ib2);
                            adj[a + (1 + k) * L + l] += ck * ib;
                            adj[b + (1 + k) * L + l] -= ck * q * ib;
                        }
                        adj[a + l] += sa;
                        adj[b + l] += sb;
                    }
                }
                Op::Lin => {
                    let (s0, cnt, x0) = (nd.a as usize, nd.b as usize, nd.x as usize);
                    for j in 0..cnt {
                        let p = t.auxi[s0 + j] as usize * stride;
                        let s = t.auxf[x0 + j];
                        for c in 0..nc {
                            for l in 0..L {
                                adj[p + c * L + l] += s * cb[c][l];
                            }
                        }
                    }
                }
            }
        }
        if next_ext > 0 && t.exts[next_ext - 1].o0 as usize == i {
            next_ext -= 1;
            let ext = &t.exts[next_ext];
            let rec = &t.lus[ext.lu as usize];
            let n_ = rec.n;
            let nn = n_ * n_;
            let o0 = ext.o0 as usize;
            let x0 = &ext.x[0..n_];
            let mut xb0 = vec![0.0; n_];
            let mut y = vec![0.0; n_];
            for l in 0..L {
                let ix = |node: usize, comp: usize| (node * nc + comp) * L + l;
                for r in 0..n_ {
                    xb0[r] = adj[ix(o0 + r, 0)];
                }
                for c in 1..nc {
                    for r in 0..n_ {
                        y[r] = adj[ix(o0 + r, c)];
                    }
                    if y.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    lu_tsolve(rec, &mut y);
                    let xk = &ext.x[c * n_..(c + 1) * n_];
                    for r in 0..n_ {
                        let bi = ext.bidx[r];
                        if bi != NONE {
                            adj[ix(bi as usize, c)] += y[r];
                        }
                        for j in 0..n_ {
                            let wi = rec.widx[r * n_ + j];
                            if wi != NONE {
                                adj[ix(wi as usize, c)] -= y[r] * x0[j];
                                adj[ix(wi as usize, 0)] -= y[r] * xk[j];
                            }
                            xb0[j] -= rec.w[c * nn + r * n_ + j] * y[r];
                        }
                    }
                }
                if xb0.iter().all(|v| *v == 0.0) {
                    continue;
                }
                lu_tsolve(rec, &mut xb0);
                for r in 0..n_ {
                    let bi = ext.bidx[r];
                    if bi != NONE {
                        adj[ix(bi as usize, 0)] += xb0[r];
                    }
                    for j in 0..n_ {
                        let wi = rec.widx[r * n_ + j];
                        if wi != NONE {
                            adj[ix(wi as usize, 0)] -= xb0[r] * x0[j];
                        }
                    }
                }
            }
        }
    }
    &t.adj
}

/// Index of component `comp` (0 = value, `1 + k` = tangent `k`, `1 + N + t` = taped
/// Hessian entry `t`) of `node` in a taped-Hessian adjoint buffer ([`vsweep_h`]).
#[inline(always)]
pub(crate) fn vadj_index_h<const N: usize, const M: usize>(node: u32, comp: usize) -> usize {
    node as usize * (1 + N + M) + comp
}

/// Single-lane backward sweep of a directional second-order recording
/// (`DualV<N, N, true>`, `vtape_reset(1 + 2N)`), over nodes `0 .. top`, with seeds placed
/// on any component via [`vadj_init`] / [`vadj_index_h`]. The second-order rules are the
/// derivatives of the forward ones (`h_k = a_k b + a b_k + 2a_k b_k` of a product,
/// `φ′a_k + φ″a_k²` of a unary, written with `(k, k)` pairs); a quotient is
/// differentiated as `a · (1/b)`, the same function as the stored `a/b`.
pub(crate) fn vsweep_h<const N: usize, const M: usize>(top: usize) -> &'static [f64] {
    let t = vt();
    let nc = 1 + N + M;
    debug_assert_eq!(t.nc, nc, "vsweep_h needs a taped-Hessian recording");
    let n = top.min(t.nodes.len());
    let vals = &t.vals;
    let adj = &mut t.adj;
    debug_assert_eq!(
        M, N,
        "a taped-second-order recording is directional (M = N)"
    );
    // `(k, k)`: entry `k` is the second derivative along tangent direction `k`.
    let pairs: Vec<(usize, usize)> = (0..N).map(|k| (k, k)).collect();
    let mut c = vec![0.0f64; nc];
    let (mut ga, mut gb, mut ub, mut uv) = (
        vec![0.0f64; nc],
        vec![0.0f64; nc],
        vec![0.0f64; nc],
        vec![0.0f64; nc],
    );
    for i in (0..n).rev() {
        c.copy_from_slice(&adj[i * nc..i * nc + nc]);
        if c.iter().all(|x| *x == 0.0) {
            continue;
        }
        let nd = t.nodes[i];
        let lin = |adj: &mut Vec<f64>, p: usize, s: f64, c: &[f64]| {
            for k in 0..nc {
                adj[p * nc + k] += s * c[k];
            }
        };
        match nd.op {
            Op::Leaf => {}
            Op::Solve => unreachable!("the LU block hook is declined on a taped-Hessian recording"),
            Op::Add => {
                lin(adj, nd.a as usize, 1.0, &c);
                lin(adj, nd.b as usize, 1.0, &c);
            }
            Op::Sub => {
                lin(adj, nd.a as usize, 1.0, &c);
                lin(adj, nd.b as usize, -1.0, &c);
            }
            Op::Shift => lin(adj, nd.a as usize, 1.0, &c),
            Op::Neg => lin(adj, nd.a as usize, -1.0, &c),
            Op::Scale => lin(adj, nd.a as usize, t.auxf[nd.x as usize], &c),
            Op::Lin => {
                let (s0, cnt, x0) = (nd.a as usize, nd.b as usize, nd.x as usize);
                for j in 0..cnt {
                    lin(adj, t.auxi[s0 + j] as usize, t.auxf[x0 + j], &c);
                }
            }
            Op::Unary => {
                let x = nd.x as usize;
                let (h1, h2, h3) = (t.auxf[x], t.auxf[x + 1], t.auxf[x + 2]);
                let a = nd.a as usize;
                unary_adjoint::<N>(
                    &pairs,
                    &vals[a * nc..a * nc + nc],
                    &c,
                    h1,
                    h2,
                    h3,
                    &mut adj[a * nc..a * nc + nc],
                );
            }
            Op::Mul => {
                let (a, b) = (nd.a as usize, nd.b as usize);
                ga.iter_mut().for_each(|x| *x = 0.0);
                gb.iter_mut().for_each(|x| *x = 0.0);
                mul_adjoint::<N>(
                    &pairs,
                    &vals[a * nc..a * nc + nc],
                    &vals[b * nc..b * nc + nc],
                    &c,
                    &mut ga,
                    &mut gb,
                );
                // Accumulated separately so `a == b` (a square) adds both halves.
                lin(adj, a, 1.0, &ga);
                lin(adj, b, 1.0, &gb);
            }
            Op::Div => {
                // q = a·u, u = 1/b: u's components from b's, the product rule into (a, u),
                // then u's adjoint into b through the unary rule of 1/x.
                let (a, b) = (nd.a as usize, nd.b as usize);
                let bv = &vals[b * nc..b * nc + nc];
                let (ib, ib2) = (1.0 / bv[0], 1.0 / (bv[0] * bv[0]));
                let (d1, d2, d3) = (-ib2, 2.0 * ib2 * ib, -6.0 * ib2 * ib2);
                uv[0] = ib;
                for k in 0..N {
                    uv[1 + k] = d1 * bv[1 + k];
                }
                for (tt, &(k, l)) in pairs.iter().enumerate() {
                    uv[1 + N + tt] = d1 * bv[1 + N + tt] + d2 * bv[1 + k] * bv[1 + l];
                }
                ga.iter_mut().for_each(|x| *x = 0.0);
                ub.iter_mut().for_each(|x| *x = 0.0);
                mul_adjoint::<N>(
                    &pairs,
                    &vals[a * nc..a * nc + nc],
                    &uv,
                    &c,
                    &mut ga,
                    &mut ub,
                );
                gb.iter_mut().for_each(|x| *x = 0.0);
                unary_adjoint::<N>(&pairs, bv, &ub, d1, d2, d3, &mut gb);
                lin(adj, a, 1.0, &ga);
                lin(adj, b, 1.0, &gb);
            }
        }
    }
    &t.adj
}

/// Adjoint of `c = φ(a)` on `(v, g, h)`: `c_v = φ`, `c_k = φ′a_k`,
/// `c_kl = φ′a_kl + φ″a_k a_l`.
#[inline]
fn unary_adjoint<const N: usize>(
    pairs: &[(usize, usize)],
    a: &[f64],
    c: &[f64],
    h1: f64,
    h2: f64,
    h3: f64,
    out: &mut [f64],
) {
    let mut sv = c[0] * h1;
    for k in 0..N {
        sv += c[1 + k] * h2 * a[1 + k];
        out[1 + k] += c[1 + k] * h1;
    }
    for (tt, &(k, l)) in pairs.iter().enumerate() {
        let ch = c[1 + N + tt];
        if ch == 0.0 {
            continue;
        }
        sv += ch * (h2 * a[1 + N + tt] + h3 * a[1 + k] * a[1 + l]);
        out[1 + N + tt] += ch * h1;
        out[1 + k] += ch * h2 * a[1 + l];
        out[1 + l] += ch * h2 * a[1 + k];
    }
    out[0] += sv;
}

/// Adjoint of `c = a·b` on `(v, g, h)`: `c_k = a_k b + a b_k`,
/// `c_kl = a_kl b + a b_kl + a_k b_l + a_l b_k`, accumulated into `ga`, `gb`.
#[inline]
fn mul_adjoint<const N: usize>(
    pairs: &[(usize, usize)],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    ga: &mut [f64],
    gb: &mut [f64],
) {
    let (mut sa, mut sb) = (c[0] * b[0], c[0] * a[0]);
    for k in 0..N {
        let ck = c[1 + k];
        sa += ck * b[1 + k];
        sb += ck * a[1 + k];
        ga[1 + k] += ck * b[0];
        gb[1 + k] += ck * a[0];
    }
    for (tt, &(k, l)) in pairs.iter().enumerate() {
        let ch = c[1 + N + tt];
        if ch == 0.0 {
            continue;
        }
        let ht = 1 + N + tt;
        sa += ch * b[ht];
        sb += ch * a[ht];
        ga[ht] += ch * b[0];
        gb[ht] += ch * a[0];
        ga[1 + k] += ch * b[1 + l];
        gb[1 + l] += ch * a[1 + k];
        ga[1 + l] += ch * b[1 + k];
        gb[1 + k] += ch * a[1 + l];
    }
    ga[0] += sa;
    gb[0] += sb;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sens::dual2::Dual2;

    /// Exercises every op kind, `lincomb` and a 3×3 LU solve.
    fn prog<T: PkNum>(p0: T, p1: T, e: T) -> T {
        let c = |v: f64| T::from_f64(v);
        let a = (p0 * e).exp() / (p1 + c(1.0));
        let b = T::lincomb(a, 0.3, &[0.5, -0.2], &[p0 * p1, a * e]);
        let mut w = vec![
            c(2.0) + p0 * e,
            p1,
            c(0.3),
            p0 * p1,
            c(3.0) + e,
            c(-0.5),
            c(0.1),
            p1 * e,
            c(4.0) + p0,
        ];
        let mut rhs = vec![b, c(1.0) - p1, p0 * p1 * e];
        let mut piv = [0usize; 3];
        assert!(crate::sens::linsolve::lu_factor_in_place_g(
            &mut w, &mut piv, 3
        ));
        crate::sens::linsolve::lu_solve_in_place_g(&w, &piv, 3, &mut rhs);
        (rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2] + c(1.0))
            .sqrt()
            .ln()
            + b.pow(c(1.7)) / rhs[1]
            - (c(0.2) * e).inv_logit() * c(3.0)
    }

    /// Every op a taped-Hessian recording uses (the LU block declines there, so `prog`'s
    /// solve runs element-wise), plus `lincomb`, `sqrt`, `pow` and a square.
    fn prog_h<T: PkNum>(p0: T, p1: T, e0: T, e1: T) -> T {
        let c = |v: f64| T::from_f64(v);
        let a = (p0 * e0).exp() / (p1 + c(1.0) + e1 * e1);
        let b = T::lincomb(a, 0.3, &[0.5, -0.2], &[p0 * p1, a * e1]);
        let q = (b * b + c(2.0)).sqrt().ln() + b.pow(c(1.7)) / (c(1.5) + p1 * e0);
        q - (c(0.2) * e0 - e1).inv_logit() * c(3.0) + (c(0.3) * p0 * e1).cos()
    }

    /// Directions `u₀ = (1, −0.2)`, `u₁ = (0.3, 1)` in η: the η leaves' tangents are their
    /// components (`η₀.g = [u₀[0], u₁[0]]`, …), as the Laplace recording seeds them.
    const U: [[f64; 2]; 2] = [[1.0, -0.2], [0.3, 1.0]];

    /// `S = w₀·f + Σₘ wₘ D_{uₘ}f + Σₘ w′ₘ D²_{uₘ}f` at `(p, η)`, from the full canonical
    /// Hessian payload (no tape): `D_u f = uᵀ∇f`, `D²_u f = uᵀ∇²f u`.
    fn s_of(p0: f64, p1: f64, e0: f64, e1: f64, w: &[f64; 5]) -> f64 {
        vtape_reset(3);
        let d = |v: f64, k: Option<usize>| {
            let mut g = [0.0; 2];
            if let Some(k) = k {
                g[k] = 1.0;
            }
            DualV::<2, 3>::leaf(v, g, [0.0; 3])
        };
        let f = prog_h(d(p0, None), d(p1, None), d(e0, Some(0)), d(e1, Some(1)));
        let mut s = w[0] * f.v;
        for (m, u) in U.iter().enumerate() {
            let du = u[0] * f.g[0] + u[1] * f.g[1];
            let d2u = u[0] * u[0] * f.hess(0, 0)
                + 2.0 * u[0] * u[1] * f.hess(1, 0)
                + u[1] * u[1] * f.hess(1, 1);
            s += w[1 + m] * du + w[3 + m] * d2u;
        }
        s
    }

    #[test]
    fn directional_second_order_sweep_matches_fd_through_every_component() {
        let (p0, p1, e0, e1) = (0.7, 1.3, -0.4, 0.25);
        let w = [0.6, -1.1, 0.8, 0.5, -0.7];
        vtape_reset(1 + 2 + 2);
        let a = DualV::<2, 2, true>::leaf(p0, [0.0; 2], [0.0; 2]);
        let b = DualV::<2, 2, true>::leaf(p1, [0.0; 2], [0.0; 2]);
        let x = DualV::<2, 2, true>::leaf(e0, [U[0][0], U[1][0]], [0.0; 2]);
        let y = DualV::<2, 2, true>::leaf(e1, [U[0][1], U[1][1]], [0.0; 2]);
        let f = prog_h(a, b, x, y);
        let adj = vadj_init::<1>();
        for (comp, wc) in w.iter().enumerate() {
            adj[vadj_index_h::<2, 2>(f.i, comp)] = *wc;
        }
        let adj = vsweep_h::<2, 2>(f.i as usize + 1).to_vec();
        let got = [a, b, x, y].map(|l| adj[vadj_index_h::<2, 2>(l.i, 0)]);
        let base = [p0, p1, e0, e1];
        // The recording's own components equal the payload's directional derivatives.
        assert!(
            (f.v * w[0] + f.g[0] * w[1] + f.g[1] * w[2] + f.h[0] * w[3] + f.h[1] * w[4]
                - s_of(p0, p1, e0, e1, &w))
            .abs()
                < 1e-12
        );
        for k in 0..4 {
            let fd = |h: f64| {
                let (mut up, mut dn) = (base, base);
                up[k] += h;
                dn[k] -= h;
                (s_of(up[0], up[1], up[2], up[3], &w) - s_of(dn[0], dn[1], dn[2], dn[3], &w))
                    / (2.0 * h)
            };
            let rich = (4.0 * fd(1e-4) - fd(2e-4)) / 3.0;
            // ∂S/∂η reaches third derivatives of `prog_h` (through the taped directional `h`).
            assert!(
                (got[k] - rich).abs() <= 1e-7 * (1.0 + rich.abs()),
                "leaf {k}: sweep {} vs FD {rich}",
                got[k]
            );
        }
    }

    /// Objective `J(p) = out.v + Σ_k wₖ ∂out/∂ηₖ` at η-tangent seeds; returns
    /// `(J, ∂J/∂p0, ∂J/∂p1)` by one sweep.
    fn adjoint(p0v: f64, p1v: f64, wt: [f64; 2]) -> (f64, f64, f64, DualV<2, 3>) {
        vtape_reset(3);
        let pa = DualV::<2, 3>::leaf(p0v, [1.0, 0.0], [0.0; 3]);
        let pb = DualV::<2, 3>::leaf(p1v, [0.0, 0.0], [0.0; 3]);
        let e = DualV::<2, 3> {
            v: 0.4,
            g: [0.0, 1.0],
            h: [0.0; 3],
            i: NONE,
        };
        let o = prog(pa, pb, e);
        let buf = vadj_init::<1>();
        buf[vadj_index::<2, 1>(o.i, 0, 0)] = 1.0;
        buf[vadj_index::<2, 1>(o.i, 1, 0)] = wt[0];
        buf[vadj_index::<2, 1>(o.i, 2, 0)] = wt[1];
        let top = o.i as usize + 1;
        let adj = vsweep::<2, 1>(top);
        let j = o.v + wt[0] * o.g[0] + wt[1] * o.g[1];
        (
            j,
            adj[vadj_index::<2, 1>(pa.i, 0, 0)],
            adj[vadj_index::<2, 1>(pb.i, 0, 0)],
            o,
        )
    }

    #[test]
    fn reverse_over_forward_matches_central_differences() {
        let wt = [0.7, -0.3];
        let (_, ga, gb, _) = adjoint(0.8, 1.3, wt);
        let h = 1e-6;
        let fa = (adjoint(0.8 + h, 1.3, wt).0 - adjoint(0.8 - h, 1.3, wt).0) / (2.0 * h);
        let fb = (adjoint(0.8, 1.3 + h, wt).0 - adjoint(0.8, 1.3 - h, wt).0) / (2.0 * h);
        assert!((ga - fa).abs() < 1e-7 * (1.0 + fa.abs()), "{ga} vs {fa}");
        assert!((gb - fb).abs() < 1e-7 * (1.0 + fb.abs()), "{gb} vs {fb}");
    }

    #[test]
    fn value_tangents_and_hessian_payload_match_dual2() {
        let (_, _, _, o) = adjoint(0.8, 1.3, [0.0, 0.0]);
        let mut q0 = Dual2::<2>::constant(0.8);
        q0.grad[0] = 1.0;
        let q1 = Dual2::<2>::constant(1.3);
        let mut qe = Dual2::<2>::constant(0.4);
        qe.grad[1] = 1.0;
        let r = prog(q0, q1, qe);
        assert!((o.v - r.value).abs() < 1e-13);
        for k in 0..2 {
            assert!((o.g[k] - r.grad[k]).abs() < 1e-12);
            for l in 0..2 {
                assert!(
                    (o.hess(k, l) - r.hess[k][l]).abs() < 1e-10,
                    "h[{k}][{l}] {} vs {}",
                    o.hess(k, l),
                    r.hess[k][l]
                );
            }
        }
    }

    #[test]
    fn rollback_discards_records_after_the_mark() {
        vtape_reset(3);
        let a = DualV::<2, 3>::leaf(1.0, [1.0, 0.0], [0.0; 3]);
        let m = <DualV<2, 3> as PkNum>::tape_mark();
        let _ = a * a + a.exp();
        assert!(vtape_len() > m);
        <DualV<2, 3> as PkNum>::tape_rollback(m);
        assert_eq!(vtape_len(), m);
        // A stale mark is ignored.
        let _ = a * a;
        let n = vtape_len();
        <DualV<2, 3> as PkNum>::tape_rollback(m - 1);
        assert_eq!(vtape_len(), n);
    }

    #[test]
    fn cse_reuses_identical_ops_without_changing_the_gradient() {
        vtape_reset(3);
        let a = DualV::<2, 3>::leaf(1.7, [1.0, 0.0], [0.0; 3]);
        let b = DualV::<2, 3>::leaf(0.6, [0.0, 1.0], [0.0; 3]);
        let q1 = a / b;
        let n1 = vtape_len();
        let q2 = a / b;
        assert_eq!(
            q1.i, q2.i,
            "identical op on identical parents must reuse the node"
        );
        assert_eq!(vtape_len(), n1);
        let e1 = a.exp();
        let l1 = a.ln();
        assert_ne!(
            e1.i, l1.i,
            "different functions of the same input must not alias"
        );
        // Gradient through the reused node equals the analytic one.
        let out = q1 * q2 + e1;
        let buf = vadj_init::<1>();
        buf[vadj_index::<2, 1>(out.i, 0, 0)] = 1.0;
        let adj = vsweep::<2, 1>(out.i as usize + 1);
        let da = adj[vadj_index::<2, 1>(a.i, 0, 0)];
        let want = 2.0 * (1.7 / 0.6) / 0.6 + 1.7f64.exp();
        assert!((da - want).abs() < 1e-12, "{da} vs {want}");
    }

    #[test]
    fn tri_is_symmetric_and_dense() {
        let mut seen = vec![false; 10];
        for k in 0..4 {
            for l in 0..4 {
                assert_eq!(tri(k, l), tri(l, k));
                seen[tri(k, l)] = true;
            }
        }
        assert!(seen.iter().all(|x| *x));
    }
}
