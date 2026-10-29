//! Scalar reverse tape for the adjoint's observation-level likelihood kernel.
//!
//! [`R`] records each operation as one node with up to a few `(parent, partial)`
//! entries on a thread-local tape. [`rsweep`] runs a backward pass with `L` adjoint
//! lanes. [`DualT<N, R>`](DualT) is a first-order forward dual in η over `R`, so a
//! function evaluated on it yields its η-gradient *as tape expressions*, which a reverse
//! sweep can then differentiate (forward-over-reverse).
//!
//! The kernel evaluates the FOCEI subject objective `F(f, G, η)` and the inner
//! gradient `∇_η l(f, G, η)` on this tape, where `f`, `G = ∂f/∂η` and `η` are leaves. One
//! sweep then gives `∂F` and `∂(∇l)_k` with respect to every leaf, whatever likelihood
//! form the kernel implements. This tape is small: one subject's observation rows, never
//! the ODE.
#![allow(clippy::needless_range_loop, clippy::suspicious_arithmetic_impl)]

use super::num::PkNum;
use std::cell::UnsafeCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

const NONE: u32 = u32::MAX;

struct RTape {
    /// Entries of node `i` are `ent[off[i]..off[i + 1]]`.
    off: Vec<u32>,
    ent: Vec<(u32, f64)>,
    adj: Vec<f64>,
    /// Whole RHS calls ([`record_rhs`]), by first output node.
    blocks: Vec<RBlock>,
    /// Every block's input parts' tape ids and values ([`RBlock::at`]).
    block_ids: Vec<u32>,
    block_x: Vec<f64>,
    /// The kernels [`RBlock::kernel`] indexes, held for the recording's life.
    kernels: Vec<std::sync::Arc<dyn crate::sens::num::vm::RhsKernel>>,
}

/// One RHS call over a nested dual recorded whole: its outputs' scalar parts are the
/// entry-less nodes `o0 .. o0 + n_out`, whose adjoints reach the inputs' parts through the
/// kernel's compiled reverse pass.
struct RBlock {
    o0: u32,
    n_out: u32,
    kernel: u32,
    nest: crate::sens::num::vm::Nest,
    /// How many `u` and `params` elements the inputs below hold (then 4 seeds).
    eu: u32,
    ep: u32,
    /// Where this block's input parts start in [`RTape::block_ids`] (tape ids, `NONE`:
    /// untaped) and [`RTape::block_x`] (values): `u`, then `params`, then the seeds.
    at: u32,
}

thread_local! {
    static RT: UnsafeCell<RTape> = const { UnsafeCell::new(RTape {
        off: Vec::new(), ent: Vec::new(), adj: Vec::new(), blocks: Vec::new(),
        block_ids: Vec::new(), block_x: Vec::new(), kernels: Vec::new(),
    }) };
}

#[inline]
fn rt() -> &'static mut RTape {
    // SAFETY: thread-local; every function here drops the reference before any call that
    // re-enters the module.
    RT.with(|t| unsafe { &mut *t.get() })
}

#[inline]
fn push<const K: usize>(parents: [(u32, f64); K]) -> u32 {
    let t = rt();
    let start = t.ent.len();
    for (p, d) in parents {
        if p != NONE && d != 0.0 {
            t.ent.push((p, d));
        }
    }
    if t.ent.len() == start {
        return NONE;
    }
    t.off.push(t.ent.len() as u32);
    (t.off.len() - 2) as u32
}

/// Start a new recording (keeps allocations).
pub(crate) fn rtape_reset() {
    let t = rt();
    t.off.clear();
    t.ent.clear();
    t.blocks.clear();
    t.block_ids.clear();
    t.block_x.clear();
    t.kernels.clear();
    t.off.push(0);
}

/// Nodes on the current recording.
pub(crate) fn rtape_len() -> usize {
    rt().off.len().saturating_sub(1)
}

/// Release the tape's buffers when they hold more than `nodes` nodes' worth: a one-off
/// large recording (the covariance Hessian's) need not pin its peak on every worker.
pub(crate) fn rtape_trim(nodes: usize) {
    let t = rt();
    if t.off.capacity() > nodes || t.ent.capacity() > 2 * nodes || t.adj.capacity() > 8 * nodes {
        *t = RTape {
            off: Vec::new(),
            ent: Vec::new(),
            adj: Vec::new(),
            blocks: Vec::new(),
            block_ids: Vec::new(),
            block_x: Vec::new(),
            kernels: Vec::new(),
        };
        t.off.push(0);
    }
}

/// A new independent input.
pub(crate) fn rinput(v: f64) -> R {
    let t = rt();
    t.off.push(t.ent.len() as u32);
    R {
        v,
        i: (t.off.len() - 2) as u32,
    }
}

/// Backward sweep with `L` lanes; lane `l` is seeded with 1 on `outs[l]`. Returns the
/// node-major adjoint buffer `adj[node·L + lane]`.
pub(crate) fn rsweep<const L: usize>(outs: &[R; L]) -> &'static [f64] {
    let t = rt();
    let n = t.off.len() - 1;
    t.adj.clear();
    t.adj.resize(n * L, 0.0);
    for (l, o) in outs.iter().enumerate() {
        if o.i != NONE {
            t.adj[o.i as usize * L + l] = 1.0;
        }
    }
    let mut next_block = t.blocks.partition_point(|b| (b.o0 as usize) < n);
    let (mut du_bar, mut x_bar) = (Vec::new(), Vec::new());
    let (off, ent) = (&t.off[..], &t.ent[..]);
    let adj = t.adj.as_mut_ptr();
    for i in (0..n).rev() {
        if next_block > 0 && t.blocks[next_block - 1].o0 as usize == i {
            // A block's outputs carry no entries, so it runs when the sweep reaches its
            // first output, when every output's adjoint is complete.
            next_block -= 1;
            let b = &t.blocks[next_block];
            // SAFETY: the adjoint buffer holds `n·L` lanes and nothing else borrows it here.
            let adj_s = unsafe { std::slice::from_raw_parts_mut(adj, n * L) };
            let w = b.nest.size_from(0);
            let ni = (b.eu + b.ep + 4) as usize * w;
            let ids = &t.block_ids[b.at as usize..b.at as usize + ni];
            let x = &t.block_x[b.at as usize..b.at as usize + ni];
            for l in 0..L {
                block_adjoint(
                    &t.kernels,
                    b,
                    ids,
                    x,
                    &mut du_bar,
                    &mut x_bar,
                    adj_s,
                    |node| node * L + l,
                );
            }
        }
        // SAFETY: `i < n` and every entry's parent is an earlier node (`push` only ever
        // records existing ids), so all offsets are inside the `n·L` buffer; the read of
        // node `i`'s lanes completes before any write to a parent's.
        unsafe {
            let a: [f64; L] = std::ptr::read(adj.add(i * L) as *const [f64; L]);
            if a == [0.0; L] {
                continue;
            }
            let (s, e) = (
                *off.get_unchecked(i) as usize,
                *off.get_unchecked(i + 1) as usize,
            );
            for k in s..e {
                let (p, d) = *ent.get_unchecked(k);
                let dst = adj.add(p as usize * L);
                for l in 0..L {
                    *dst.add(l) += a[l] * d;
                }
            }
        }
    }
    &t.adj
}

/// The block adjoint of one [`RBlock`] for one adjoint lane (`idx(node)` its slot).
#[allow(clippy::too_many_arguments)]
fn block_adjoint(
    kernels: &[std::sync::Arc<dyn crate::sens::num::vm::RhsKernel>],
    b: &RBlock,
    ids: &[u32],
    x: &[f64],
    du_bar: &mut Vec<f64>,
    x_bar: &mut Vec<f64>,
    adj: &mut [f64],
    idx: impl Fn(usize) -> usize,
) {
    let n_out = b.n_out as usize;
    du_bar.clear();
    du_bar.resize(n_out, 0.0);
    let mut live = false;
    for (o, d) in du_bar.iter_mut().enumerate() {
        *d = adj[idx(b.o0 as usize + o)];
        live |= *d != 0.0;
    }
    if !live {
        return;
    }
    let w = b.nest.size_from(0);
    let (eu, ep) = (b.eu as usize, b.ep as usize);
    x_bar.clear();
    x_bar.resize(x.len(), 0.0);
    let (xb_u, rest) = x_bar.split_at_mut(eu * w);
    let (xb_p, xb_s) = rest.split_at_mut(ep * w);
    // SAFETY: `x` and `x_bar` hold the inputs' parts at the recorded extent, `du_bar` the
    // outputs'; `record_rhs` checked `has_nest`.
    unsafe {
        kernels[b.kernel as usize].reverse_nest(
            b.nest,
            x.as_ptr(),
            x[eu * w..].as_ptr(),
            x[(eu + ep) * w..].as_ptr(),
            xb_u.as_mut_ptr(),
            xb_p.as_mut_ptr(),
            xb_s.as_mut_ptr(),
            du_bar.as_ptr(),
        )
    };
    for (e, &id) in ids.iter().enumerate() {
        let a = x_bar[e];
        if id != NONE && a != 0.0 {
            adj[idx(id as usize)] += a;
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: tape every RHS operation instead of recording the call whole.
    pub(crate) static RHS_BLOCK_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Record one RHS call over a nested dual on the scalar tape whole (`VmArith::tape_rhs`):
/// `du` from the kernel's native forward over the type's scalar parts, an entry-less node
/// per output part, and an [`RBlock`] whose reverse rule is the kernel's reverse pass.
///
/// The values are the types' own arithmetic, transcribed (`parser::rhs_ir::NestLower`);
/// what changes is the tape — one block, not a node per operation of every part — and so
/// the order a sweep sums the adjoints. `false`, with nothing recorded, when the kernel
/// cannot serve the call.
pub(crate) fn record_rhs<T: crate::sens::num::vm::VmNum>(
    kernel: &std::sync::Arc<dyn crate::sens::num::vm::RhsKernel>,
    u: &[T],
    params: &[T],
    seeds: &[T; 4],
    du: &mut [T],
) -> bool {
    #[cfg(test)]
    if RHS_BLOCK_OFF.with(|c| c.get()) {
        return false;
    }
    let Some(nest) = T::NEST else {
        return false;
    };
    let (eu, ep, edu) = kernel.extent();
    if u.len() < eu || params.len() < ep || du.len() < edu {
        return false;
    }
    if !kernel.has_nest(nest) {
        return false;
    }
    let w = nest.size_from(0);
    let ni = eu + ep + 4;
    let dense = kernel.writes_every_du();
    RECORD_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let (x, ids, out, out_ids) = &mut *sc;
        x.clear();
        x.resize(ni * w, 0.0);
        ids.clear();
        ids.resize(ni * w, NONE);
        out.clear();
        out.resize(edu * w, 0.0);
        out_ids.clear();
        out_ids.resize(edu * w, NONE);
        let inputs = u[..eu].iter().chain(&params[..ep]).chain(seeds.iter());
        for (e, v) in inputs.enumerate() {
            v.put_parts(&mut x[e * w..(e + 1) * w], &mut ids[e * w..(e + 1) * w]);
        }
        // An output the program does not write keeps what `du` held, which the block
        // cannot carry if it was taped; a program that writes every derivative never
        // reads it.
        if !dense {
            for (o, d) in du[..edu].iter().enumerate() {
                d.put_parts(
                    &mut out[o * w..(o + 1) * w],
                    &mut out_ids[o * w..(o + 1) * w],
                );
            }
            if out_ids.iter().any(|&i| i != NONE) {
                return false;
            }
        }
        // SAFETY: `x` holds `u`, `params` and the seeds at the extent, `out` the extent of
        // `du`, all at `nest`'s parts; `has_nest` checked.
        unsafe {
            kernel.forward_nest(
                nest,
                x.as_ptr(),
                x[eu * w..].as_ptr(),
                x[(eu + ep) * w..].as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if ids.iter().any(|&i| i != NONE) {
            let t = rt();
            let k = match t
                .kernels
                .iter()
                .position(|k2| std::sync::Arc::ptr_eq(k2, kernel))
            {
                Some(k) => k,
                None => {
                    t.kernels.push(kernel.clone());
                    t.kernels.len() - 1
                }
            } as u32;
            let o0 = (t.off.len() - 1) as u32;
            for id in out_ids.iter_mut() {
                t.off.push(t.ent.len() as u32);
                *id = (t.off.len() - 2) as u32;
            }
            let at = t.block_ids.len() as u32;
            t.block_ids.extend_from_slice(ids);
            t.block_x.extend_from_slice(x);
            t.blocks.push(RBlock {
                o0,
                n_out: (edu * w) as u32,
                kernel: k,
                nest,
                eu: eu as u32,
                ep: ep as u32,
                at,
            });
        }
        // Nothing taped reaches the call otherwise: plain values.
        for (o, d) in du[..edu].iter_mut().enumerate() {
            *d = T::from_parts(&out[o * w..(o + 1) * w], &out_ids[o * w..(o + 1) * w]);
        }
        true
    })
}

thread_local! {
    /// [`record_rhs`]'s input and output parts (kept apart from the tape it pushes to).
    #[allow(clippy::type_complexity)]
    static RECORD_SCRATCH: std::cell::RefCell<(Vec<f64>, Vec<u32>, Vec<f64>, Vec<u32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

/// Scalar on the reverse tape.
#[derive(Clone, Copy, Debug)]
pub struct R {
    pub v: f64,
    pub i: u32,
}

impl R {
    #[inline]
    pub fn c(v: f64) -> R {
        R { v, i: NONE }
    }
    #[inline]
    fn n1(v: f64, a: R, da: f64) -> R {
        R {
            v,
            i: if a.i == NONE { NONE } else { push([(a.i, da)]) },
        }
    }
    #[inline]
    fn n2(v: f64, a: R, da: f64, b: R, db: f64) -> R {
        R {
            v,
            i: push([(a.i, da), (b.i, db)]),
        }
    }
    /// Tape index of this value (`None` for a constant).
    pub fn idx(self) -> Option<usize> {
        (self.i != NONE).then_some(self.i as usize)
    }
}

impl Add for R {
    type Output = R;
    #[inline]
    fn add(self, b: R) -> R {
        if b.i == NONE {
            return R {
                v: self.v + b.v,
                i: self.i,
            };
        }
        if self.i == NONE {
            return R {
                v: self.v + b.v,
                i: b.i,
            };
        }
        R::n2(self.v + b.v, self, 1.0, b, 1.0)
    }
}
impl Sub for R {
    type Output = R;
    #[inline]
    fn sub(self, b: R) -> R {
        if b.i == NONE {
            return R {
                v: self.v - b.v,
                i: self.i,
            };
        }
        R::n2(self.v - b.v, self, 1.0, b, -1.0)
    }
}
impl Mul for R {
    type Output = R;
    #[inline]
    fn mul(self, b: R) -> R {
        R::n2(self.v * b.v, self, b.v, b, self.v)
    }
}
impl Div for R {
    type Output = R;
    #[inline]
    fn div(self, b: R) -> R {
        let q = self.v / b.v;
        R::n2(q, self, 1.0 / b.v, b, -q / b.v)
    }
}
impl Neg for R {
    type Output = R;
    #[inline]
    fn neg(self) -> R {
        R::n1(-self.v, self, -1.0)
    }
}

impl PkNum for R {
    const SECOND_ORDER: bool = false;
    /// No tape node: nothing downstream differentiates it.
    #[inline]
    fn is_constant(self) -> bool {
        self.i == NONE
    }
    fn from_f64(x: f64) -> Self {
        R::c(x)
    }
    fn var(x: f64, _dim: usize) -> Self {
        rinput(x)
    }
    fn val(self) -> f64 {
        self.v
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        R::n1(e, self, e)
    }
    fn ln(self) -> Self {
        R::n1(self.v.ln(), self, 1.0 / self.v)
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        R::n1(s, self, 0.5 / s)
    }
    fn pow(self, e: Self) -> Self {
        let a = self.v;
        let v = a.powf(e.v);
        let da = if a == 0.0 {
            0.0
        } else {
            e.v * a.powf(e.v - 1.0)
        };
        let de = if a > 0.0 { v * a.ln() } else { 0.0 };
        R::n2(v, self, da, e, de)
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
        R::n1(u, self, u * (1.0 - u))
    }
    fn logit(self) -> Self {
        let x = self.v;
        R::n1((x / (1.0 - x)).ln(), self, 1.0 / (x * (1.0 - x)))
    }
    fn cos(self) -> Self {
        R::n1(self.v.cos(), self, -self.v.sin())
    }
    fn acos(self) -> Self {
        R::n1(self.v.acos(), self, -1.0 / (1.0 - self.v * self.v).sqrt())
    }
    fn guard_floor(self, lo: f64) -> Self {
        if !(self.v >= lo) {
            R::c(lo)
        } else {
            self
        }
    }
    fn ln_gamma(self) -> Self {
        R::n1(
            crate::stats::special::ln_gamma(self.v),
            self,
            crate::stats::special::digamma(self.v),
        )
    }
    fn jets_finite(self) -> bool {
        true
    }
    /// A reverse-only scalar carries no local jet: its derivative is the tape's.
    fn jet_max_abs(self) -> f64 {
        0.0
    }
    fn lift2(self, v: f64, d1: f64, _d2: f64) -> Self {
        R::n1(v, self, d1)
    }
    fn wsum3(base: f64, terms: impl Iterator<Item = (Self, Self, Self)>) -> Self {
        let t = rt();
        let start = t.ent.len();
        let mut v = base;
        for (a, x, y) in terms {
            v += a.v * x.v * y.v;
            for (p, d) in [(a.i, x.v * y.v), (x.i, a.v * y.v), (y.i, a.v * x.v)] {
                if p != NONE && d != 0.0 {
                    t.ent.push((p, d));
                }
            }
        }
        if t.ent.len() == start {
            return R::c(v);
        }
        t.off.push(t.ent.len() as u32);
        R {
            v,
            i: (t.off.len() - 2) as u32,
        }
    }
}

/// First-order forward dual in `N` directions over a base `B` (value and tangents are
/// `B`). With `B = R`, its tangents are tape expressions.
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
            g[k] = self.g[k] * o.v + self.v * o.g[k];
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
        // as `DualV::pow` has it: `exp(e·ln x)` is NaN at `x ≤ 0` and declined the subject
        // to finite differences (#2093 review). `DualV`'s convention at `x = 0` (a zero
        // derivative) is kept, so this differentiates the function the gradient records.
        if e.is_constant() {
            let n = e.v.val();
            if self.v.val() == 0.0 {
                return DualT::constant(0.0f64.powf(n));
            }
            let d1 = B::from_f64(n) * self.v.pow(B::from_f64(n - 1.0));
            return self.chain(self.v.pow(B::from_f64(n)), d1);
        }
        (e * self.ln()).exp()
    }
    fn abs(self) -> Self {
        if self.v.val() >= 0.0 {
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
        if !(self.v.val() >= lo) {
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

/// Second-order forward dual in `N` directions over a base `B`: value, gradient and
/// second-order entries, each a `B`. With `D = false` those are the packed symmetric
/// Hessian (`M = N(N+1)/2`, `h[tri(k, l)]`) — over `f64`, the exact η-Hessian of the
/// adjoint Laplace kernel's `l`. With `D = true` (`M = N`) they are the pure second
/// derivatives along the `N` directions, the algebra of the Laplace recording
/// ([`crate::sens::vtape::DualV`] in its taped mode); over [`R`] they are tape
/// expressions. Derivatives of the elementary functions are written in `B` arithmetic
/// (as in [`DualT`]), so they carry their own derivatives to every order the base tracks.
#[derive(Clone, Copy, Debug)]
pub struct DualH<const N: usize, const M: usize, B: Copy, const D: bool = false> {
    pub v: B,
    pub g: [B; N],
    pub h: [B; M],
}

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    /// Whether this width's arithmetic runs out of line ([`crate::sens::num::outlined`]).
    const WIDE: bool = crate::sens::num::outlined(N, std::mem::size_of::<Self>());
    #[inline]
    pub fn constant(v: f64) -> Self {
        DualH {
            v: B::from_f64(v),
            g: [B::from_f64(0.0); N],
            h: [B::from_f64(0.0); M],
        }
    }
    /// `φ(self)` from `φ(v)`, `φ′(v)`, `φ″(v)` as base values:
    /// `g_k = φ′g_k`, `h_kl = φ′h_kl + φ″g_k g_l`.
    #[inline]
    fn chain2(self, p0: B, p1: B, p2: B) -> Self {
        if Self::WIDE {
            self.chain2_wide(p0, p1, p2)
        } else {
            self.chain2_body(p0, p1, p2)
        }
    }
    /// [`Self::chain2`]'s body out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn chain2_wide(self, p0: B, p1: B, p2: B) -> Self {
        self.chain2_body(p0, p1, p2)
    }
    #[inline]
    fn chain2_body(self, p0: B, p1: B, p2: B) -> Self {
        let g = std::array::from_fn(|k| p1 * self.g[k]);
        let mut h = self.h;
        for k in 0..N {
            for l in crate::sens::vtape::hrange::<D, M>(k) {
                let t = crate::sens::vtape::hidx::<D>(k, l);
                h[t] = p1 * self.h[t] + p2 * self.g[k] * self.g[l];
            }
        }
        DualH { v: p0, g, h }
    }
    /// The packed Hessian entry `(k, l)` (full mode).
    #[inline]
    pub fn hess(&self, k: usize, l: usize) -> B {
        self.h[crate::sens::vtape::tri(k, l)]
    }
}

impl<const N: usize, const M: usize, B: PkNum, const D: bool> Add for DualH<N, M, B, D> {
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

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    #[inline]
    fn add_body(self, o: Self) -> Self {
        DualH {
            v: self.v + o.v,
            g: std::array::from_fn(|k| self.g[k] + o.g[k]),
            h: std::array::from_fn(|t| self.h[t] + o.h[t]),
        }
    }
    /// [`Self::add_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn add_wide(self, o: Self) -> Self {
        self.add_body(o)
    }
}
impl<const N: usize, const M: usize, B: PkNum, const D: bool> Sub for DualH<N, M, B, D> {
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

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    #[inline]
    fn sub_body(self, o: Self) -> Self {
        DualH {
            v: self.v - o.v,
            g: std::array::from_fn(|k| self.g[k] - o.g[k]),
            h: std::array::from_fn(|t| self.h[t] - o.h[t]),
        }
    }
    /// [`Self::sub_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn sub_wide(self, o: Self) -> Self {
        self.sub_body(o)
    }
}
impl<const N: usize, const M: usize, B: PkNum, const D: bool> Mul for DualH<N, M, B, D> {
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

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    #[inline]
    fn mul_body(self, o: Self) -> Self {
        let g = std::array::from_fn(|k| self.g[k] * o.v + self.v * o.g[k]);
        let mut h = self.h;
        for k in 0..N {
            for l in crate::sens::vtape::hrange::<D, M>(k) {
                let t = crate::sens::vtape::hidx::<D>(k, l);
                h[t] = self.h[t] * o.v + self.v * o.h[t] + self.g[k] * o.g[l] + self.g[l] * o.g[k];
            }
        }
        DualH {
            v: self.v * o.v,
            g,
            h,
        }
    }
    /// [`Self::mul_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn mul_wide(self, o: Self) -> Self {
        self.mul_body(o)
    }
}
impl<const N: usize, const M: usize, B: PkNum, const D: bool> Div for DualH<N, M, B, D> {
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

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    #[inline]
    fn div_body(self, o: Self) -> Self {
        // a/b = a · (1/b), with 1/x′ = −1/x², 1/x″ = 2/x³.
        let ib = B::from_f64(1.0) / o.v;
        let d1 = -(ib * ib);
        self * o.chain2(ib, d1, B::from_f64(-2.0) * d1 * ib)
    }
    /// [`Self::div_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn div_wide(self, o: Self) -> Self {
        self.div_body(o)
    }
}
impl<const N: usize, const M: usize, B: PkNum, const D: bool> Neg for DualH<N, M, B, D> {
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

impl<const N: usize, const M: usize, B: PkNum, const D: bool> DualH<N, M, B, D> {
    #[inline]
    fn neg_body(self) -> Self {
        DualH {
            v: -self.v,
            g: std::array::from_fn(|k| -self.g[k]),
            h: std::array::from_fn(|t| -self.h[t]),
        }
    }
    /// [`Self::neg_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn neg_wide(self) -> Self {
        self.neg_body()
    }
}

impl<const N: usize, const M: usize, B: PkNum, const D: bool> PkNum for DualH<N, M, B, D> {
    const SECOND_ORDER: bool = true;
    #[inline]
    fn is_constant(self) -> bool {
        self.v.is_constant()
            && self
                .g
                .iter()
                .chain(&self.h)
                .all(|d| d.is_constant() && d.val() == 0.0)
    }
    fn from_f64(x: f64) -> Self {
        DualH::constant(x)
    }
    fn var(x: f64, dim: usize) -> Self {
        let mut d = DualH::constant(x);
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
        self.chain2(e, e, e)
    }
    fn ln(self) -> Self {
        let d1 = B::from_f64(1.0) / self.v;
        self.chain2(self.v.ln(), d1, -(d1 * d1))
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        let d1 = B::from_f64(0.5) / s;
        self.chain2(s, d1, -(d1 / (B::from_f64(2.0) * self.v)))
    }
    fn pow(self, e: Self) -> Self {
        // The constant-exponent power rule, as [`DualT::pow`] (#2093 review).
        if e.is_constant() {
            let n = e.v.val();
            if self.v.val() == 0.0 {
                return DualH::constant(0.0f64.powf(n));
            }
            let c = B::from_f64;
            let d1 = c(n) * self.v.pow(c(n - 1.0));
            let d2 = c(n * (n - 1.0)) * self.v.pow(c(n - 2.0));
            return self.chain2(self.v.pow(c(n)), d1, d2);
        }
        (e * self.ln()).exp()
    }
    fn abs(self) -> Self {
        if self.v.val() >= 0.0 {
            self
        } else {
            -self
        }
    }
    fn inv_logit(self) -> Self {
        let one = B::from_f64(1.0);
        let u = self.v.inv_logit();
        let d1 = u * (one - u);
        self.chain2(u, d1, d1 * (one - B::from_f64(2.0) * u))
    }
    fn logit(self) -> Self {
        let one = B::from_f64(1.0);
        let x = self.v;
        let d1 = one / (x * (one - x));
        self.chain2(x.logit(), d1, -((one - B::from_f64(2.0) * x) * d1 * d1))
    }
    fn cos(self) -> Self {
        let c = self.v.cos();
        let s = (B::from_f64(std::f64::consts::FRAC_PI_2) - self.v).cos();
        self.chain2(c, -s, -c)
    }
    fn acos(self) -> Self {
        let one = B::from_f64(1.0);
        let x = self.v;
        let w = one - x * x;
        let d1 = -(one / w.sqrt());
        self.chain2(x.acos(), d1, d1 * x / w)
    }
    fn guard_floor(self, lo: f64) -> Self {
        if !(self.v.val() >= lo) {
            DualH::constant(lo)
        } else {
            self
        }
    }
    fn ln_gamma(self) -> Self {
        use crate::stats::special::{digamma, pentagamma, tetragamma, trigamma};
        let x = self.v.val();
        let p1 = self.v.lift2(digamma(x), trigamma(x), tetragamma(x));
        let p2 = self.v.lift2(trigamma(x), tetragamma(x), pentagamma(x));
        self.chain2(self.v.ln_gamma(), p1, p2)
    }
    fn jets_finite(self) -> bool {
        self.g.iter().all(|x| x.val().is_finite()) && self.h.iter().all(|x| x.val().is_finite())
    }
    /// The gradient and Hessian, and the base's own jets of the value and of each entry.
    fn jet_max_abs(self) -> f64 {
        crate::sens::num::nan_max_abs(
            std::iter::once(self.v.jet_max_abs()).chain(
                self.g
                    .iter()
                    .chain(&self.h)
                    .flat_map(|x| [x.val(), x.jet_max_abs()]),
            ),
        )
    }
    /// Without `φ‴` the base cannot differentiate the Hessian's `φ″` factor; exact only
    /// for an affine lift (`φ″ = 0`), else the jet is poisoned with NaN rather than
    /// silently truncated. Callers with a curved `φ` use [`PkNum::lift3`].
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        self.lift3(v, d1, d2, if d2 == 0.0 { 0.0 } else { f64::NAN })
    }
    /// Without `φ⁗` the base cannot differentiate the `φ‴` its `φ″` factor needs: exact
    /// only over a first-order base. Over the covariance step's `DualT` base that is `NaN`,
    /// as in [`Self::lift2`]; callers with a known `φ⁗` use [`PkNum::lift4`].
    fn lift3(self, v: f64, d1: f64, d2: f64, d3: f64) -> Self {
        self.lift4(v, d1, d2, d3, f64::NAN)
    }
    /// The value, `φ′` and `φ″` factors lifted on the base, each with the derivatives
    /// after it.
    fn lift4(self, v: f64, d1: f64, d2: f64, d3: f64, d4: f64) -> Self {
        let p1 = self.v.lift3(d1, d2, d3, d4);
        let p2 = self.v.lift2(d2, d3, d4);
        self.chain2(self.v.lift4(v, d1, d2, d3, d4), p1, p2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rtape_trim` releases a tape grown past its bound, leaves a small one alone, and the
    /// released tape records and sweeps as a fresh one.
    #[test]
    fn rtape_trim_releases_a_large_tape_and_keeps_a_small_one() {
        rtape_reset();
        let mut x = rinput(1.5);
        for _ in 0..1000 {
            x = x * R::c(1.0001);
        }
        assert_eq!(rtape_len(), 1001);
        let _ = rsweep::<1>(&[x]);
        rtape_trim(1 << 20);
        assert!(
            rt().off.capacity() >= 1001,
            "a small tape must keep its buffers"
        );
        rtape_trim(100);
        assert_eq!(rtape_len(), 0);
        assert!(rt().off.capacity() < 1001 && rt().adj.capacity() == 0);
        let y = rinput(2.0);
        let z = y * y;
        assert_eq!(rsweep::<1>(&[z])[y.i as usize], 4.0);
    }

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
        let xh = DualH::<2, 3, f64> {
            v: -2.0,
            g: [1.0, 0.0],
            h: [0.0; 3],
        };
        let yh = xh.pow(DualH::constant(2.0));
        assert_eq!((yh.v, yh.g[0]), (4.0, -4.0));
        assert_eq!(yh.h[crate::sens::vtape::tri(0, 0)], 2.0, "∂²(x²)/∂t₀² = 2");
        let live = DualT::<2, f64> {
            v: 2.0,
            g: [0.0, 1.0],
        };
        assert!(x.pow(live).v.is_nan(), "a live exponent keeps exp(e·ln x)");
    }

    fn prog_h<T: PkNum>(x: T, y: T) -> T {
        let c = |v: f64| T::from_f64(v);
        let a = (x * y).exp() / (c(1.5) + y * y) + (x + c(2.0)).ln() * y.sqrt();
        a.pow(c(1.3)) - (c(0.4) * x - y).inv_logit()
            + (c(0.3) * x).cos()
            + (x + c(3.0)).ln_gamma()
            + y.lift3(
                0.7f64.sin() * 0.0 + y.val().sin(),
                y.val().cos(),
                -y.val().sin(),
                -y.val().cos(),
            )
    }

    #[test]
    fn dual_h_matches_dual2_and_its_reverse_sweep_matches_fd() {
        use crate::sens::dual2::Dual2;
        let (x0, y0) = (0.6, 1.4);
        // (1) over f64: value, gradient, Hessian = Dual2's.
        let hx = DualH::<2, 3, f64> {
            v: x0,
            g: [1.0, 0.0],
            h: [0.0; 3],
        };
        let hy = DualH::<2, 3, f64> {
            v: y0,
            g: [0.0, 1.0],
            h: [0.0; 3],
        };
        let fh = prog_h(hx, hy);
        let fd2 = prog_h(Dual2::<2>::var(x0, 0), Dual2::<2>::var(y0, 1));
        assert!((fh.v - fd2.value).abs() < 1e-13);
        for k in 0..2 {
            assert!((fh.g[k] - fd2.grad[k]).abs() < 1e-12, "grad {k}");
            for l in 0..=k {
                assert!(
                    (fh.hess(k, l) - fd2.hess[k][l]).abs() < 1e-11,
                    "hess {k}{l}"
                );
            }
        }
        // (2) over R: S = det(∇²f) + ∂f/∂x, swept to (x, y) — third derivatives of `prog_h`.
        let s_of = |x: f64, y: f64| {
            let f = prog_h(
                DualH::<2, 3, f64> {
                    v: x,
                    g: [1.0, 0.0],
                    h: [0.0; 3],
                },
                DualH::<2, 3, f64> {
                    v: y,
                    g: [0.0, 1.0],
                    h: [0.0; 3],
                },
            );
            f.hess(0, 0) * f.hess(1, 1) - f.hess(1, 0) * f.hess(1, 0) + f.g[0]
        };
        rtape_reset();
        let (rx, ry) = (rinput(x0), rinput(y0));
        let one = R::c(1.0);
        let zero = R::c(0.0);
        let f = prog_h(
            DualH::<2, 3, R> {
                v: rx,
                g: [one, zero],
                h: [zero; 3],
            },
            DualH::<2, 3, R> {
                v: ry,
                g: [zero, one],
                h: [zero; 3],
            },
        );
        let s = f.hess(0, 0) * f.hess(1, 1) - f.hess(1, 0) * f.hess(1, 0) + f.g[0];
        assert!((s.v - s_of(x0, y0)).abs() < 1e-12);
        let adj = rsweep::<1>(&[s]).to_vec();
        for (k, leaf) in [rx, ry].iter().enumerate() {
            let fd = |h: f64| {
                if k == 0 {
                    (s_of(x0 + h, y0) - s_of(x0 - h, y0)) / (2.0 * h)
                } else {
                    (s_of(x0, y0 + h) - s_of(x0, y0 - h)) / (2.0 * h)
                }
            };
            let rich = (4.0 * fd(1e-4) - fd(2e-4)) / 3.0;
            let got = adj[leaf.i as usize];
            assert!(
                (got - rich).abs() < 1e-7 * (1.0 + rich.abs()),
                "leaf {k}: sweep {got} vs FD {rich}"
            );
        }
    }

    #[test]
    fn wsum3_node_matches_the_binary_expression() {
        // The fused node must give the value and every partial of `b + Σ a·x·y`,
        // including a repeated parent (x = y on a diagonal entry).
        let vals = [(0.3, 1.7, -0.4), (2.1, 0.5, 0.5), (-0.8, 1.1, 0.9)];
        let run = |fused: bool| {
            rtape_reset();
            let ins: Vec<(R, R, R)> = vals
                .iter()
                .map(|&(a, x, y)| (rinput(a), rinput(x), rinput(y)))
                .collect();
            let terms = [ins[0], (ins[1].0, ins[1].1, ins[1].1), ins[2]];
            let out = if fused {
                R::wsum3(0.25, terms.iter().copied())
            } else {
                let mut s = R::c(0.25);
                for &(a, x, y) in &terms {
                    s = s + a * x * y;
                }
                s
            };
            let adj = rsweep::<1>(&[out]).to_vec();
            (
                out.v,
                ins.iter()
                    .flat_map(|&(a, x, y)| {
                        [adj[a.i as usize], adj[x.i as usize], adj[y.i as usize]]
                    })
                    .collect::<Vec<f64>>(),
            )
        };
        let (vf, gf) = run(true);
        let (vb, gb) = run(false);
        assert!((vf - vb).abs() < 1e-15, "value {vf} vs {vb}");
        for (k, (a, b)) in gf.iter().zip(&gb).enumerate() {
            assert!(
                (a - b).abs() < 1e-14,
                "partial {k}: fused {a} vs binary {b}"
            );
        }
    }

    #[test]
    fn reverse_gradient_matches_closed_form() {
        rtape_reset();
        let x = rinput(0.7);
        let y = rinput(1.3);
        let f =
            (x * y).exp() + x.ln() * y - y / x + x.lift2(0.7f64.sin(), 0.7f64.cos(), -0.7f64.sin());
        let adj = rsweep::<1>(&[f]);
        let (xv, yv) = (0.7f64, 1.3f64);
        let dx = yv * (xv * yv).exp() + yv / xv + yv / (xv * xv) + xv.cos();
        let dy = xv * (xv * yv).exp() + xv.ln() - 1.0 / xv;
        assert!((adj[x.i as usize] - dx).abs() < 1e-12);
        assert!((adj[y.i as usize] - dy).abs() < 1e-12);
    }

    /// Forward-over-reverse: `h(p) = Σ_k w_k ∂φ/∂e_k` for `φ = sin(p·e)·p`, swept for
    /// `∂h/∂p` and compared with central differences of `h`.
    #[test]
    fn forward_over_reverse_matches_central_differences() {
        let h_of = |p: f64| -> (f64, f64) {
            rtape_reset();
            let pr = rinput(p);
            let pd = DualT::<2, R> {
                v: pr,
                g: [R::c(0.0); 2],
            };
            let e = DualT::<2, R> {
                v: R::c(0.4),
                g: [R::c(1.0), R::c(0.0)],
            };
            let e2 = DualT::<2, R> {
                v: R::c(0.9),
                g: [R::c(0.0), R::c(1.0)],
            };
            let z = pd * e + e2 * e2;
            let phi = z.lift2(z.val().sin(), z.val().cos(), -z.val().sin()) * pd;
            let h = phi.g[0] * R::c(0.6) + phi.g[1] * R::c(-1.1);
            let adj = rsweep::<1>(&[h]);
            (h.v, adj[pr.i as usize])
        };
        let (_, g) = h_of(1.3);
        let e = 1e-6;
        let fd = (h_of(1.3 + e).0 - h_of(1.3 - e).0) / (2.0 * e);
        assert!((g - fd).abs() < 1e-8, "{g} vs {fd}");
    }
}
