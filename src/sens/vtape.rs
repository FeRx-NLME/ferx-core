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
#[inline]
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
#[inline]
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
#[inline]
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
    /// An output of one whole RHS call ([`VRhs`]); its reverse rule is the block's.
    Rhs,
    /// `acc + Σ aₖ·bₖ` ([`PkNum::dot_fold`]): `a` = start in `auxi` of `[acc, a₀, b₀, …]`,
    /// `b` = the pair count, `x` = start in `auxf` of the untaped factors' components.
    Dot,
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

/// One RHS call recorded whole ([`record_rhs`]): its outputs are the consecutive nodes
/// `o0 .. o0 + n_out`, and their adjoints reach the inputs through the kernel's compiled
/// reverse pass, at the tape's own components ([`VRhs::layout`]).
struct VRhs {
    o0: u32,
    n_out: u32,
    kernel: u32,
    /// The reverse pass's lanes: the tape's `nc` components (value and tangents, plus the
    /// payload when it is taped).
    layout: crate::sens::num::vm::Layout,
    /// How many `u` and `params` elements the inputs hold (then 4 seeds).
    eu: u32,
    ep: u32,
    /// Where the inputs' tape ids (`NONE`: untaped), `u` then `params` then the seeds,
    /// start in [`VTape::rhs_ids`], and their lanes at `layout` in [`VTape::rhs_x`].
    ids_at: u32,
    x_at: u32,
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
    rhss: Vec<VRhs>,
    rhs_ids: Vec<u32>,
    rhs_x: Vec<f64>,
    /// The kernels [`VRhs::kernel`] indexes, each held for the recording's life.
    kernels: Vec<std::sync::Arc<dyn crate::sens::num::vm::RhsKernel>>,
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
    rhss: usize,
    rhs_ids: usize,
    rhs_x: usize,
}

thread_local! {
    static VT: UnsafeCell<VTape> = const { UnsafeCell::new(VTape {
        nc: 0, nodes: Vec::new(), vals: Vec::new(), auxf: Vec::new(), auxi: Vec::new(),
        adj: Vec::new(), lus: Vec::new(), exts: Vec::new(), rhss: Vec::new(),
        rhs_ids: Vec::new(), rhs_x: Vec::new(), kernels: Vec::new(),
        mark: Mark { nodes: 0, auxf: 0, auxi: 0, lus: 0, exts: 0, rhss: 0, rhs_ids: 0, rhs_x: 0 },
        cse: Vec::new(),
    }) };
}

#[inline]
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
    t.rhss.clear();
    t.rhs_ids.clear();
    t.rhs_x.clear();
    t.kernels.clear();
    t.mark = Mark::default();
    // The CSE cache is not cleared between recordings: a hit is verified against the
    // *current* tape's node (op, parents, value), so a stale entry can only match a node
    // that is the same pure op on the same parents — exactly what may be reused.
    if t.cse.is_empty() {
        t.cse.resize(1 << CSE_BITS, (0, 0, NONE));
    }
}

/// Release the tape's buffers when its node store holds more than `nodes` nodes' worth
/// (#2481 §2.7), the vector tape's [`crate::sens::rtape::rtape_trim`]: `vtape_reset` keeps
/// every buffer's capacity so a fit's recordings do not reallocate per subject, which leaves
/// each worker holding its peak recording for the life of the process (an R session). The
/// CSE cache goes with them; the next `vtape_reset` rebuilds it.
pub(crate) fn vtape_trim(nodes: usize) {
    let t = vt();
    if t.nodes.capacity() > nodes {
        *t = VTape {
            nc: 0,
            nodes: Vec::new(),
            vals: Vec::new(),
            auxf: Vec::new(),
            auxi: Vec::new(),
            adj: Vec::new(),
            lus: Vec::new(),
            exts: Vec::new(),
            rhss: Vec::new(),
            rhs_ids: Vec::new(),
            rhs_x: Vec::new(),
            kernels: Vec::new(),
            mark: Mark::default(),
            cse: Vec::new(),
        };
    }
}

/// The node store's capacity, for the trim's test.
#[cfg(test)]
pub(crate) fn vtape_capacity() -> usize {
    vt().nodes.capacity()
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
        rhss: t.rhss.len(),
        rhs_ids: t.rhs_ids.len(),
        rhs_x: t.rhs_x.len(),
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
    t.rhss.truncate(m.rhss);
    t.rhs_ids.truncate(m.rhs_ids);
    t.rhs_x.truncate(m.rhs_x);
}

thread_local! {
    /// Lane buffers for [`record_rhs`] (kept apart from the tape, which it pushes to).
    static RHS_LANES: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Record one RHS call whole (`VmArith::tape_rhs`): `du` from the kernel's native forward
/// over `DualV<N, M, TH>`'s lanes, one [`Op::Rhs`] node per derivative written, and a
/// [`VRhs`] that sends their adjoints to the inputs through the kernel's reverse pass.
///
/// The values are the native forward's, which is the register code's arithmetic bit for
/// bit; what changes is the tape — one block instead of a node per operation — and so the
/// order in which a sweep sums the adjoints. `false`, with nothing recorded, when the
/// kernel cannot serve the call (no native layout, slices shorter than the program reads,
/// or a recording whose components are not this type's).
pub(crate) fn record_rhs<const N: usize, const M: usize, const TH: bool>(
    kernel: &std::sync::Arc<dyn crate::sens::num::vm::RhsKernel>,
    u: &[DualV<N, M, TH>],
    params: &[DualV<N, M, TH>],
    seeds: &[DualV<N, M, TH>; 4],
    du: &mut [DualV<N, M, TH>],
) -> bool {
    use crate::sens::num::vm::{Layout, VmArith};
    #[cfg(test)]
    if RHS_BLOCK_OFF.with(|c| c.get()) {
        return false;
    }
    let Some(full) = <DualV<N, M, TH> as VmArith>::LANES else {
        return false;
    };
    // The tape's components: value and tangents, plus the payload when it is taped.
    let rev = if TH {
        full
    } else {
        Layout {
            n: N as u16,
            m: 0,
            th: false,
            wide: full.wide,
        }
    };
    if vt().nc != rev.lanes() {
        return false;
    }
    let (eu, ep, edu) = kernel.extent();
    if u.len() < eu || params.len() < ep || du.len() < edu {
        return false;
    }
    // An output the program does not write keeps what `du` held, which the block cannot
    // carry if it was taped; a program that writes every derivative never reads it.
    let dense = kernel.writes_every_du();
    if !dense && du[..edu].iter().any(|d| d.i != NONE) {
        return false;
    }
    if !kernel.has_forward(full) || !kernel.has_reverse(rev) {
        return false;
    }
    let (wf, wr) = (full.lanes(), rev.lanes());
    let ni = eu + ep + 4;
    RHS_LANES.with(|cell| {
        let mut b = cell.borrow_mut();
        let need = (ni + edu) * wf;
        if b.len() < need {
            b.resize(need, 0.0);
        }
        let (xin, out) = b.split_at_mut(ni * wf);
        let out = &mut out[..edu * wf];
        let inputs = || u[..eu].iter().chain(&params[..ep]).chain(seeds.iter());
        for (x, l) in inputs().zip(xin.chunks_exact_mut(wf)) {
            x.put_lanes(l);
        }
        for (x, l) in du[..edu].iter().zip(out.chunks_exact_mut(wf)) {
            if dense {
                l.fill(0.0);
            } else {
                x.put_lanes(l);
            }
        }
        // SAFETY: `xin` holds `u`, `params` and the seeds at the extent, `out` the extent
        // of `du`, all at `full`'s lanes; `has_forward` checked.
        unsafe {
            kernel.forward(
                full,
                xin.as_ptr(),
                xin[eu * wf..].as_ptr(),
                xin[(eu + ep) * wf..].as_ptr(),
                out.as_mut_ptr(),
            )
        };
        let k = {
            let t = vt();
            match t
                .kernels
                .iter()
                .position(|k2| std::sync::Arc::ptr_eq(k2, kernel))
            {
                Some(k) => k,
                None => {
                    t.kernels.push(kernel.clone());
                    t.kernels.len() - 1
                }
            }
        } as u32;
        let (ids_at, x_at) = {
            let t = vt();
            let at = (t.rhs_ids.len() as u32, t.rhs_x.len() as u32);
            t.rhs_ids.extend(inputs().map(|x| x.i));
            for l in xin.chunks_exact(wf) {
                t.rhs_x.extend_from_slice(&l[..wr]);
            }
            at
        };
        let o0 = vt().nodes.len() as u32;
        for (d, l) in du[..edu].iter_mut().zip(out.chunks_exact(wf)) {
            let mut g = [0.0; N];
            let mut h = [0.0; M];
            g.copy_from_slice(&l[1..1 + N]);
            h.copy_from_slice(&l[1 + N..1 + N + M]);
            let i = push::<N>(Op::Rhs, NONE, NONE, NONE, l[0], &g, &h);
            *d = DualV { v: l[0], g, h, i };
        }
        vt().rhss.push(VRhs {
            o0,
            n_out: edu as u32,
            kernel: k,
            layout: rev,
            eu: eu as u32,
            ep: ep as u32,
            ids_at,
            x_at,
        });
    });
    true
}

#[cfg(test)]
thread_local! {
    /// Test switch: tape every RHS operation instead of recording the call whole.
    pub(crate) static RHS_BLOCK_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The block adjoint of one [`VRhs`] for one adjoint lane: read the outputs' adjoints
/// from `adj` (component `c` of node `i` at `idx(i, c)`), run the reverse pass, and add the
/// inputs' adjoints into it.
fn rhs_block_adjoint(
    t: &VTape,
    r: &VRhs,
    du_bar: &mut Vec<f64>,
    x_bar: &mut Vec<f64>,
    adj: &mut [f64],
    idx: impl Fn(usize, usize) -> usize,
) {
    let w = r.layout.lanes();
    let (eu, ep) = (r.eu as usize, r.ep as usize);
    let ni = eu + ep + 4;
    let ids = &t.rhs_ids[r.ids_at as usize..r.ids_at as usize + ni];
    let x = &t.rhs_x[r.x_at as usize..r.x_at as usize + ni * w];
    let n_out = r.n_out as usize;
    du_bar.clear();
    du_bar.resize(n_out * w, 0.0);
    let mut live = false;
    for o in 0..n_out {
        for c in 0..w {
            let a = adj[idx(r.o0 as usize + o, c)];
            du_bar[o * w + c] = a;
            live |= a != 0.0;
        }
    }
    if !live {
        return;
    }
    x_bar.clear();
    x_bar.resize(x.len(), 0.0);
    let (xb_u, rest) = x_bar.split_at_mut(eu * w);
    let (xb_p, xb_s) = rest.split_at_mut(ep * w);
    // SAFETY: `x` and `x_bar` hold the inputs at the recorded extent and `layout`'s
    // lanes, `du_bar` the outputs; `record_rhs` checked `has_reverse`.
    unsafe {
        t.kernels[r.kernel as usize].reverse(
            r.layout,
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
        if id != NONE {
            for c in 0..w {
                let a = x_bar[e * w + c];
                if a != 0.0 {
                    adj[idx(id as usize, c)] += a;
                }
            }
        }
    }
}

/// Record a pure binary/unary op, reusing an existing node when the same op was already
/// applied to the same parents with the same record: the value, every stored tangent and
/// payload entry, and (unary/scale) the rule on the aux stream `ws`. Matching the parents
/// and the value alone is not enough, because an untaped operand can carry η-tangents the
/// parents do not show (a free `c` in `c − K` is folded into a `Neg` of `K`). A hit is
/// verified against the node itself, so stale entries (rolled-back nodes) can never alias.
#[inline]
fn push_cse<const N: usize>(
    op: Op,
    a: u32,
    b: u32,
    x: u32,
    v: f64,
    ws: &[f64],
    g: &[f64; N],
    hs: &[f64],
) -> u32 {
    let w = ws.first().copied().unwrap_or(0.0);
    let k1 = ((op as u64) << 56) ^ ((a as u64) << 28) ^ (b as u64);
    let k2 = v.to_bits() ^ w.to_bits().rotate_left(17);
    let hk = (k1 ^ k2.rotate_left(29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - CSE_BITS);
    let t = vt();
    let (e1, e2, id) = t.cse[hk as usize];
    if id != NONE && e1 == k1 && e2 == k2 && (id as usize) < t.nodes.len() {
        let nd = t.nodes[id as usize];
        let same = |x: f64, y: f64| x.to_bits() == y.to_bits();
        // A node is the same record only if every stored component matches, not just the
        // value: two expressions with equal values and parents can still carry different
        // η-tangents (`exp(η₁) − K` and `exp(η₂) − K` at η̂₁ = η̂₂, or `1 − K` beside
        // `exp(η) − K` at η = 0), and the sweep reads the stored ones (#2093 review).
        let row = &t.vals[id as usize * t.nc..(id as usize + 1) * t.nc];
        let stored = std::iter::once(v)
            .chain(g.iter().copied())
            .chain(hs.iter().copied().take(t.nc - 1 - N));
        let same_row = row.iter().zip(stored).all(|(&p, q)| same(p, q));
        // Unary and scale nodes keep their rule on the aux stream: every recorded
        // derivative (`φ′, φ″` and, taped-Hessian, `φ‴`), or the scale factor, must match.
        let same_aux = match op {
            Op::Unary | Op::Scale => t
                .auxf
                .get(nd.x as usize..nd.x as usize + ws.len())
                .is_some_and(|aux| aux.iter().zip(ws).all(|(&p, &q)| same(p, q))),
            _ => true,
        };
        if nd.op == op && nd.a == a && nd.b == b && same_row && same_aux {
            return id;
        }
    }
    let id = push::<N>(op, a, b, x, v, g, hs);
    vt().cse[hk as usize] = (k1, k2, id);
    id
}

#[inline]
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

/// `r` (computed untaped) as one [`Op::Lin`] node over its taped parents with their
/// weights, or `r` itself when no parent is taped. A linear map's partials are its weights,
/// so the node's rule is exact; the values are the caller's.
fn lin_record<const N: usize, const M: usize, const TH: bool>(
    r: DualV<N, M, TH>,
    parents: impl Iterator<Item = (u32, f64)> + Clone,
) -> DualV<N, M, TH> {
    if parents.clone().all(|(i, _)| i == NONE) {
        return r;
    }
    let t = vt();
    let a = t.auxi.len() as u32;
    let x = t.auxf.len() as u32;
    let mut cnt = 0u32;
    for (i, w) in parents {
        if i != NONE {
            t.auxi.push(i);
            t.auxf.push(w);
            cnt += 1;
        }
    }
    let i = push::<N>(Op::Lin, a, cnt, x, r.v, &r.g, &r.h);
    DualV { i, ..r }
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
    /// Whether this width's arithmetic runs out of line ([`crate::sens::num::outlined`]).
    pub(crate) const WIDE: bool = crate::sens::num::outlined(N, std::mem::size_of::<Self>());
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn hess(&self, k: usize, l: usize) -> f64 {
        self.h[tri(k, l)]
    }
    #[inline]
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
        let mut r = DualV {
            v: hv,
            g,
            h,
            i: NONE,
        };
        // The record (if any) only adds the tape index: the value, tangents and payload are
        // the same either way. A cold call that returns just the index keeps `r` in
        // registers on the untaped path; returning a whole `DualV` from it made the join a
        // memory value, and every untaped operation a store-forwarding stall (~5× slower).
        if self.i != NONE {
            r.i = self.unary_taped(h1, h2, if TH { h3() } else { 0.0 }, &r);
        }
        r
    }
    /// The tape record of [`Self::unary`], out of line so the untaped arithmetic above
    /// stays small enough to inline (#2093: forcing the whole body inline was most of
    /// the crate's compile time).
    #[inline(never)]
    #[cold]
    fn unary_taped(&self, h1: f64, h2: f64, h3: f64, r: &Self) -> u32 {
        let x = vt().auxf.len() as u32;
        let rule = [h1, h2, h3];
        let i = push_cse::<N>(
            Op::Unary,
            self.i,
            NONE,
            x,
            r.v,
            &rule[..if TH { 3 } else { 2 }],
            &r.g,
            &r.h,
        );
        if i as usize == vt().nodes.len() - 1 && vt().nodes[i as usize].x == x {
            let t = vt();
            t.auxf.push(h1);
            t.auxf.push(h2);
            if TH {
                t.auxf.push(h3);
            }
        }
        i
    }
    /// A taped value times a constant: always a tape record, so always out of line.
    #[inline(never)]
    fn scaled(&self, c: f64, r: &Self) -> u32 {
        let x = vt().auxf.len() as u32;
        let i = push_cse::<N>(Op::Scale, self.i, NONE, x, r.v, &[c], &r.g, &r.h);
        if i as usize == vt().nodes.len() - 1 && vt().nodes[i as usize].x == x {
            vt().auxf.push(c);
        }
        i
    }
}

/// Arithmetic with a constant operand (the bytecode VM's literals, covariates and `TIME`):
/// the generic operator's values and record, without building and multiplying through a
/// zero dual. The tape tails are the generic ones, handed that constant.
impl<const N: usize, const M: usize, const TH: bool> DualV<N, M, TH> {
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline(always)]
    pub(crate) fn add_c(self, c: f64) -> Self {
        let v = self.v + c;
        let mut r = DualV { v, i: NONE, ..self };
        if self.i != NONE {
            r.i = self.add_taped(&Self::constant(c), &r);
        }
        r
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline(always)]
    pub(crate) fn sub_c(self, c: f64) -> Self {
        let v = self.v - c;
        let mut r = DualV { v, i: NONE, ..self };
        if self.i != NONE {
            r.i = self.sub_taped(&Self::constant(c), &r);
        }
        r
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline(always)]
    pub(crate) fn rsub_c(self, c: f64) -> Self {
        Self::constant(c) - self
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline(always)]
    pub(crate) fn mul_c(self, c: f64) -> Self {
        if Self::WIDE {
            return self * Self::constant(c);
        }
        let v = self.v * c;
        let mut g = self.g;
        let mut h = self.h;
        g.iter_mut().for_each(|x| *x *= c);
        h.iter_mut().for_each(|x| *x *= c);
        let mut r = DualV { v, g, h, i: NONE };
        if self.i != NONE {
            r.i = self.mul_taped(&Self::constant(c), &r);
        }
        r
    }
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
    #[inline(always)]
    pub(crate) fn div_c(self, c: f64) -> Self {
        if Self::WIDE {
            return self / Self::constant(c);
        }
        let q = self.v / c;
        let ib = 1.0 / c;
        let mut g = self.g;
        let mut h = self.h;
        g.iter_mut().for_each(|x| *x *= ib);
        h.iter_mut().for_each(|x| *x *= ib);
        let mut r = DualV {
            v: q,
            g,
            h,
            i: NONE,
        };
        if self.i != NONE {
            r.i = self.div_taped(&Self::constant(c), ib, &r);
        }
        r
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
        let mut r = DualV { v, g, h, i: NONE };
        // The record (if any) only adds the tape index: the value, tangents and payload are
        // the same either way. A cold call that returns just the index keeps `r` in
        // registers on the untaped path; returning a whole `DualV` from it made the join a
        // memory value, and every untaped operation a store-forwarding stall (~5× slower).
        if self.i != NONE || o.i != NONE {
            r.i = self.add_taped(&o, &r);
        }
        r
    }
    /// The tape record of an addition (see [`Self::unary_taped`]).
    #[inline(never)]
    #[cold]
    fn add_taped(&self, o: &Self, r: &Self) -> u32 {
        let (v, g, h) = (r.v, &r.g, &r.h);
        match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            (false, true) => push::<N>(Op::Shift, self.i, NONE, NONE, v, g, h),
            (true, false) => push::<N>(Op::Shift, o.i, NONE, NONE, v, g, h),
            (false, false) => push_cse::<N>(Op::Add, self.i, o.i, NONE, v, &[], g, h),
        }
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
        let mut r = DualV { v, g, h, i: NONE };
        if self.i != NONE || o.i != NONE {
            r.i = self.sub_taped(&o, &r);
        }
        r
    }
    /// The tape record of a subtraction (see [`Self::unary_taped`]).
    #[inline(never)]
    #[cold]
    fn sub_taped(&self, o: &Self, r: &Self) -> u32 {
        let (v, g, h) = (r.v, &r.g, &r.h);
        match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            (false, true) => push::<N>(Op::Shift, self.i, NONE, NONE, v, g, h),
            (true, false) => push_cse::<N>(Op::Neg, o.i, NONE, NONE, v, &[], g, h),
            (false, false) => push_cse::<N>(Op::Sub, self.i, o.i, NONE, v, &[], g, h),
        }
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
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
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
        let mut r = DualV { v, g, h, i: NONE };
        if self.i != NONE || o.i != NONE {
            r.i = self.mul_taped(&o, &r);
        }
        r
    }
    /// The tape record of a product (see [`Self::unary_taped`]).
    #[inline(never)]
    #[cold]
    fn mul_taped(&self, o: &Self, r: &Self) -> u32 {
        match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            (true, false) if self.zero_tan() => {
                if self.v == 0.0 {
                    return NONE;
                }
                o.scaled(self.v, r)
            }
            (false, true) if o.zero_tan() => {
                if o.v == 0.0 {
                    return NONE;
                }
                self.scaled(o.v, r)
            }
            _ => {
                let (a, b) = (self.mat(), o.mat());
                push_cse::<N>(Op::Mul, a.i, b.i, NONE, r.v, &[], &r.g, &r.h)
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
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
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
        let mut r = DualV {
            v: q,
            g,
            h,
            i: NONE,
        };
        if self.i != NONE || o.i != NONE {
            r.i = self.div_taped(&o, ib, &r);
        }
        r
    }
    /// The tape record of a quotient (see [`Self::unary_taped`]).
    #[inline(never)]
    #[cold]
    fn div_taped(&self, o: &Self, ib: f64, r: &Self) -> u32 {
        match (self.i == NONE, o.i == NONE) {
            (true, true) => NONE,
            // A constant exact zero over anything is the constant zero — no derivative, as
            // `mul_taped`'s constant-zero factor. Without it `0 / V` with `V` taped pushes a
            // node, and the zero stops being structural (`linsolve::structural_zero`).
            (true, false) if self.zero_tan() && self.v == 0.0 => NONE,
            (false, true) if o.zero_tan() => self.scaled(ib, r),
            _ => {
                let (a, b) = (self.mat(), o.mat());
                push_cse::<N>(Op::Div, a.i, b.i, NONE, r.v, &[], &r.g, &r.h)
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
        let mut r = DualV {
            v: -self.v,
            g,
            h,
            i: NONE,
        };
        if self.i != NONE {
            r.i = self.neg_taped(&r);
        }
        r
    }
    /// The tape record of a negation (see [`Self::unary_taped`]).
    #[inline(never)]
    #[cold]
    fn neg_taped(&self, r: &Self) -> u32 {
        push_cse::<N>(Op::Neg, self.i, NONE, NONE, r.v, &[], &r.g, &r.h)
    }
    /// [`Self::neg_body`] out of line ([`Self::WIDE`]).
    #[inline(never)]
    fn neg_wide(self) -> Self {
        self.neg_body()
    }
}

impl<const N: usize, const M: usize, const TH: bool> PkNum for DualV<N, M, TH> {
    const SECOND_ORDER: bool = true;
    /// The forward jet's second order, plus the tape's one in θ on the value and tangents;
    /// the directional mode tapes its second-order payload too.
    const JET_ORDER: usize = if TH { 3 } else { 2 };
    /// The value, tangents and payload (`Layout::lanes`), with the value and tangents taped
    /// — and, directionally, the payload too.
    const WORK: usize = {
        let lanes = 1 + N + M;
        let taped = if TH { lanes } else { 1 + N };
        lanes + 2 * taped
    };
    /// The tape differentiates the value and the η-tangents once in θ, and the η-Hessian
    /// payload is untaped: order ≤ 2 throughout. The exact-anchor mode (`TH`) tapes the
    /// payload too, which is order 3.
    const TAYLOR2_EXACT: bool = !TH;
    const ETA_HESSIAN: bool = M > 0;
    const FUSED_FOLDS: bool = true;
    #[inline]
    fn is_taped(self) -> bool {
        self.i != NONE
    }
    #[inline]
    fn is_constant(self) -> bool {
        self.i == NONE && self.g.iter().chain(&self.h).all(|&d| d == 0.0)
    }
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
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
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
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
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
    // Mirrored by hand in `parser/rhs_ir.rs` (the ODE RHS's scalar code): edit both.
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
    /// The η-tangents and their Hessian payload. The θ side lives on the tape and has no
    /// local jet; the linear propagator's self-check sees the η side only.
    fn jet_max_abs(self) -> f64 {
        crate::sens::num::nan_max_abs(self.g.iter().chain(&self.h).copied())
    }
    fn lift2(self, v: f64, d1: f64, d2: f64) -> Self {
        // `lift2` carries no `φ‴`: a taped-Hessian sweep through it would be wrong, so it
        // poisons that sweep with NaN (non-finite → the caller's finiteness backstop)
        // rather than silently differentiate a truncated rule. A curved `φ` with a known
        // `φ‴` goes through `lift3`.
        self.unary(v, d1, d2, || f64::NAN)
    }
    /// Exact on every `DualV` mode: the exact-anchor sweep (`TH`) records `φ‴` for the
    /// Hessian payload's own derivative.
    fn lift3(self, v: f64, d1: f64, d2: f64, d3: f64) -> Self {
        self.unary(v, d1, d2, || d3)
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
    /// The default's values, computed on untaped copies (the same operations, so the same
    /// lanes), with one [`Op::Lin`] node of weights `(1, w₀, w₁, …)`.
    fn axpy_fold(acc: Self, terms: &[(Self, f64)]) -> Self {
        let free = |x: Self| DualV { i: NONE, ..x };
        let mut a = free(acc);
        for &(x, w) in terms {
            a = a + free(x) * Self::from_f64(w);
        }
        lin_record(
            a,
            std::iter::once((acc.i, 1.0)).chain(terms.iter().map(|&(x, w)| (x.i, w))),
        )
    }
    /// The default's values on untaped copies, with one [`Op::Lin`] node of weights
    /// `(s, −s)`.
    fn scaled_diff(a: Self, b: Self, s: f64) -> Self {
        let free = |x: Self| DualV { i: NONE, ..x };
        let r = (free(a) - free(b)) * Self::from_f64(s);
        lin_record(r, [(a.i, s), (b.i, -s)].into_iter())
    }
    fn caxpy_fold(acc: Self, terms: &[(f64, Self)]) -> Self {
        let free = |x: Self| DualV { i: NONE, ..x };
        let mut a = free(acc);
        for &(w, x) in terms {
            a = a + Self::from_f64(w) * free(x);
        }
        lin_record(
            a,
            std::iter::once((acc.i, 1.0)).chain(terms.iter().map(|&(w, x)| (x.i, w))),
        )
    }
    /// The default's values on untaped copies, with one [`Op::Dot`] node.
    fn dot_fold(acc: Self, pairs: &[(Self, Self)]) -> Self {
        let free = |x: Self| DualV { i: NONE, ..x };
        let mut r = free(acc);
        for &(a, b) in pairs {
            r = r + free(a) * free(b);
        }
        if acc.i == NONE && pairs.iter().all(|(a, b)| a.i == NONE && b.i == NONE) {
            return r;
        }
        let t = vt();
        let nc = t.nc;
        let ai = t.auxi.len() as u32;
        let xf = t.auxf.len() as u32;
        t.auxi.push(acc.i);
        let comps = |x: &Self, out: &mut Vec<f64>| {
            out.push(x.v);
            out.extend_from_slice(&x.g);
            if nc > N + 1 {
                out.extend_from_slice(&x.h);
            }
        };
        let mut cnt = 0u32;
        for (a, b) in pairs {
            if a.i == NONE && b.i == NONE {
                continue;
            }
            t.auxi.push(a.i);
            t.auxi.push(b.i);
            if a.i == NONE {
                comps(a, &mut t.auxf);
            }
            if b.i == NONE {
                comps(b, &mut t.auxf);
            }
            cnt += 1;
        }
        let i = push::<N>(Op::Dot, ai, cnt, xf, r.v, &r.g, &r.h);
        DualV { i, ..r }
    }
    #[inline]
    fn detach(self) -> Self {
        DualV { i: NONE, ..self }
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
            // x_kl = W0⁻¹(b_kl − W_k x_l − W_l x_k − W_kl x0), for a dual that carries the
            // packed payload (a first-order `DualV<N, 0>` has none to solve for).
            for k in (0..N).filter(|_| M > 0) {
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
/// bucket's tangents (`sens::ode_provider::adjoint`'s 48).
const MAX_NC: usize = 49;

/// Zero the adjoint buffer for `L` lanes and return it for seeding.
pub(crate) fn vadj_init<const L: usize>() -> &'static mut [f64] {
    let t = vt();
    let n = t.nodes.len();
    t.adj.clear();
    t.adj.resize(n * t.nc * L, 0.0);
    &mut t.adj
}

#[inline]
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
    let mut next_rhs = t.rhss.partition_point(|r| (r.o0 as usize) < n);
    let (mut du_bar, mut x_bar) = (Vec::new(), Vec::new());
    // Out of the tape for the sweep, so a block's reverse pass can read the tape beside it.
    let mut adj_buf = std::mem::take(&mut t.adj);
    let adj = &mut adj_buf;
    let vals = &t.vals;
    // One node's adjoints, `nc = N + 1` components (every one written before it is read).
    let mut cb = [[0.0f64; L]; MAX_NC];
    assert!(
        nc <= MAX_NC,
        "dual width {N} above the widest dispatch bucket"
    );
    for i in (0..n).rev() {
        // The value lane, then the tangents: the same loads as one pass over every
        // component, kept apart so neither loop carries a per-component branch.
        let mut value_live = false;
        for l in 0..L {
            let x = adj[i * stride + l];
            cb[0][l] = x;
            value_live |= x != 0.0;
        }
        let mut tangent_live = false;
        for c in 1..nc {
            for l in 0..L {
                let x = adj[i * stride + c * L + l];
                cb[c][l] = x;
                tangent_live |= x != 0.0;
            }
        }
        let live = value_live || tangent_live;
        let nd = t.nodes[i];
        // Only the value lane is live (#2481 §2.1, as in `vsweep_h`): its adjoint reaches the
        // parents' value lanes alone, by the full rules' own products with every tangent
        // adjoint zero.
        if live && !tangent_live {
            let v = |p: usize| vals[p * nc];
            let at = |p: usize, l: usize| p * stride + l;
            match nd.op {
                Op::Leaf | Op::Solve | Op::Rhs => {}
                Op::Add => {
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] += cb[0][l];
                        adj[at(nd.b as usize, l)] += cb[0][l];
                    }
                }
                Op::Sub => {
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] += cb[0][l];
                        adj[at(nd.b as usize, l)] -= cb[0][l];
                    }
                }
                Op::Shift => {
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] += cb[0][l];
                    }
                }
                Op::Neg => {
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] -= cb[0][l];
                    }
                }
                Op::Scale => {
                    let sc = t.auxf[nd.x as usize];
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] += sc * cb[0][l];
                    }
                }
                Op::Unary => {
                    let h1 = t.auxf[nd.x as usize];
                    for l in 0..L {
                        adj[at(nd.a as usize, l)] += cb[0][l] * h1;
                    }
                }
                Op::Mul => {
                    let (a, b) = (nd.a as usize, nd.b as usize);
                    let (av, bv) = (v(a), v(b));
                    for l in 0..L {
                        adj[at(a, l)] += cb[0][l] * bv;
                        adj[at(b, l)] += cb[0][l] * av;
                    }
                }
                Op::Div => {
                    let (a, b) = (nd.a as usize, nd.b as usize);
                    let q = vals[i * nc];
                    let ib = 1.0 / v(b);
                    for l in 0..L {
                        adj[at(a, l)] += cb[0][l] * ib;
                        adj[at(b, l)] += -cb[0][l] * q * ib;
                    }
                }
                Op::Lin => {
                    let (s0, cnt, x0) = (nd.a as usize, nd.b as usize, nd.x as usize);
                    for j in 0..cnt {
                        let p = t.auxi[s0 + j] as usize;
                        let sc = t.auxf[x0 + j];
                        for l in 0..L {
                            adj[at(p, l)] += sc * cb[0][l];
                        }
                    }
                }
                Op::Dot => {
                    let (s0, cnt) = (nd.a as usize, nd.b as usize);
                    let mut xf = nd.x as usize;
                    let acc = t.auxi[s0];
                    if acc != NONE {
                        for l in 0..L {
                            adj[at(acc as usize, l)] += cb[0][l];
                        }
                    }
                    for k in 0..cnt {
                        let (ia, ib) = (t.auxi[s0 + 1 + 2 * k], t.auxi[s0 + 2 + 2 * k]);
                        let mut part = |id: u32| -> f64 {
                            if id != NONE {
                                v(id as usize)
                            } else {
                                let r = t.auxf[xf];
                                xf += nc;
                                r
                            }
                        };
                        let (av, bv) = (part(ia), part(ib));
                        for l in 0..L {
                            if ia != NONE {
                                adj[at(ia as usize, l)] += cb[0][l] * bv;
                            }
                            if ib != NONE {
                                adj[at(ib as usize, l)] += cb[0][l] * av;
                            }
                        }
                    }
                }
            }
        } else if live {
            match nd.op {
                Op::Leaf | Op::Solve | Op::Rhs => {}
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
                Op::Dot => {
                    let (s0, cnt) = (nd.a as usize, nd.b as usize);
                    let mut xf = nd.x as usize;
                    let acc = t.auxi[s0];
                    if acc != NONE {
                        let p = acc as usize * stride;
                        for c in 0..nc {
                            for l in 0..L {
                                adj[p + c * L + l] += cb[c][l];
                            }
                        }
                    }
                    for k in 0..cnt {
                        let (ia, ib) = (t.auxi[s0 + 1 + 2 * k], t.auxi[s0 + 2 + 2 * k]);
                        // An untaped factor's components were stored beside the node.
                        let mut part = |id: u32| -> &[f64] {
                            if id != NONE {
                                &vals[id as usize * nc..id as usize * nc + nc]
                            } else {
                                let r = &t.auxf[xf..xf + nc];
                                xf += nc;
                                r
                            }
                        };
                        let av = part(ia);
                        let bv = part(ib);
                        // The product rule, as `Op::Mul`'s.
                        for l in 0..L {
                            let mut sa = cb[0][l] * bv[0];
                            let mut sb = cb[0][l] * av[0];
                            for q in 0..N {
                                let cq = cb[1 + q][l];
                                sa += cq * bv[1 + q];
                                sb += cq * av[1 + q];
                                if ia != NONE {
                                    adj[ia as usize * stride + (1 + q) * L + l] += cq * bv[0];
                                }
                                if ib != NONE {
                                    adj[ib as usize * stride + (1 + q) * L + l] += cq * av[0];
                                }
                            }
                            if ia != NONE {
                                adj[ia as usize * stride + l] += sa;
                            }
                            if ib != NONE {
                                adj[ib as usize * stride + l] += sb;
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
        if next_rhs > 0 && t.rhss[next_rhs - 1].o0 as usize == i {
            next_rhs -= 1;
            let r = &t.rhss[next_rhs];
            debug_assert_eq!(r.layout.lanes(), nc);
            for l in 0..L {
                rhs_block_adjoint(t, r, &mut du_bar, &mut x_bar, adj, |node, c| {
                    (node * nc + c) * L + l
                });
            }
        }
    }
    t.adj = adj_buf;
    &t.adj
}

/// Index of component `comp` (0 = value, `1 + k` = tangent `k`, `1 + N + t` = taped
/// Hessian entry `t`) of `node` in a taped-Hessian adjoint buffer ([`vsweep_h`]).
#[inline]
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
    let mut next_rhs = t.rhss.partition_point(|r| (r.o0 as usize) < n);
    let (mut du_bar, mut x_bar) = (Vec::new(), Vec::new());
    // Out of the tape for the sweep, so a block's reverse pass can read the tape beside it.
    let mut adj_buf = std::mem::take(&mut t.adj);
    let adj = &mut adj_buf;
    let vals = &t.vals;
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
        if next_rhs > 0 && t.rhss[next_rhs - 1].o0 as usize == i {
            // A block's outputs carry no rule of their own (`Op::Rhs`), so the block runs
            // as soon as the sweep reaches its first output, when every output's adjoint
            // is complete.
            next_rhs -= 1;
            let r = &t.rhss[next_rhs];
            debug_assert_eq!(r.layout.lanes(), nc);
            rhs_block_adjoint(t, r, &mut du_bar, &mut x_bar, adj, |node, k| node * nc + k);
        }
        c.copy_from_slice(&adj[i * nc..i * nc + nc]);
        let tangent_live = c[1..].iter().any(|x| *x != 0.0);
        if !tangent_live && c[0] == 0.0 {
            continue;
        }
        let nd = t.nodes[i];
        // Only the value lane is live (#2481 §2.1): a value is a function of its parents'
        // values alone, so its adjoint reaches their value lanes and nothing else. The rules
        // below are the full ones with every tangent and payload adjoint zero — the same
        // products, without the `nc`-lane loops. An AGQ node recording is seeded on its value
        // lane only, so its whole subgraph takes this path.
        if !tangent_live {
            let c0 = c[0];
            let v = |p: usize| vals[p * nc];
            match nd.op {
                Op::Leaf | Op::Rhs => {}
                Op::Solve => {
                    unreachable!("the LU block hook is declined on a taped-Hessian recording")
                }
                Op::Add => {
                    adj[nd.a as usize * nc] += c0;
                    adj[nd.b as usize * nc] += c0;
                }
                Op::Sub => {
                    adj[nd.a as usize * nc] += c0;
                    adj[nd.b as usize * nc] += -1.0 * c0;
                }
                Op::Shift => adj[nd.a as usize * nc] += c0,
                Op::Neg => adj[nd.a as usize * nc] += -1.0 * c0,
                Op::Scale => adj[nd.a as usize * nc] += t.auxf[nd.x as usize] * c0,
                Op::Lin => {
                    let (s0, cnt, x0) = (nd.a as usize, nd.b as usize, nd.x as usize);
                    for j in 0..cnt {
                        adj[t.auxi[s0 + j] as usize * nc] += t.auxf[x0 + j] * c0;
                    }
                }
                Op::Dot => {
                    let (s0, cnt) = (nd.a as usize, nd.b as usize);
                    let mut xf = nd.x as usize;
                    if t.auxi[s0] != NONE {
                        adj[t.auxi[s0] as usize * nc] += c0;
                    }
                    for k in 0..cnt {
                        let (ia, ib) = (t.auxi[s0 + 1 + 2 * k], t.auxi[s0 + 2 + 2 * k]);
                        let mut part = |id: u32| -> f64 {
                            if id != NONE {
                                v(id as usize)
                            } else {
                                let r = t.auxf[xf];
                                xf += nc;
                                r
                            }
                        };
                        let (av, bv) = (part(ia), part(ib));
                        if ia != NONE {
                            adj[ia as usize * nc] += c0 * bv;
                        }
                        if ib != NONE {
                            adj[ib as usize * nc] += c0 * av;
                        }
                    }
                }
                Op::Unary => {
                    let h1 = t.auxf[nd.x as usize];
                    adj[nd.a as usize * nc] += c0 * h1;
                }
                Op::Mul => {
                    let (a, b) = (nd.a as usize, nd.b as usize);
                    let (av, bv) = (v(a), v(b));
                    adj[a * nc] += c0 * bv;
                    adj[b * nc] += c0 * av;
                }
                Op::Div => {
                    // As the full rule: the product into (a, 1/b), then 1/b's derivative.
                    let (a, b) = (nd.a as usize, nd.b as usize);
                    let bv = v(b);
                    let (ib, ib2) = (1.0 / bv, 1.0 / (bv * bv));
                    adj[a * nc] += c0 * ib;
                    adj[b * nc] += c0 * v(a) * -ib2;
                }
            }
            continue;
        }
        let lin = |adj: &mut Vec<f64>, p: usize, s: f64, c: &[f64]| {
            for k in 0..nc {
                adj[p * nc + k] += s * c[k];
            }
        };
        match nd.op {
            Op::Leaf | Op::Rhs => {}
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
            Op::Dot => {
                let (s0, cnt) = (nd.a as usize, nd.b as usize);
                let mut xf = nd.x as usize;
                if t.auxi[s0] != NONE {
                    lin(adj, t.auxi[s0] as usize, 1.0, &c);
                }
                for k in 0..cnt {
                    let (ia, ib) = (t.auxi[s0 + 1 + 2 * k], t.auxi[s0 + 2 + 2 * k]);
                    let mut part = |id: u32| -> Vec<f64> {
                        if id != NONE {
                            vals[id as usize * nc..id as usize * nc + nc].to_vec()
                        } else {
                            let r = t.auxf[xf..xf + nc].to_vec();
                            xf += nc;
                            r
                        }
                    };
                    let (av, bv) = (part(ia), part(ib));
                    ga.iter_mut().for_each(|x| *x = 0.0);
                    gb.iter_mut().for_each(|x| *x = 0.0);
                    mul_adjoint::<N>(&pairs, &av, &bv, &c, &mut ga, &mut gb);
                    if ia != NONE {
                        lin(adj, ia as usize, 1.0, &ga);
                    }
                    if ib != NONE {
                        lin(adj, ib as usize, 1.0, &gb);
                    }
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
    t.adj = adj_buf;
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

    /// #2481 §2.7: `vtape_trim` releases a tape grown past its bound, leaves a small one
    /// alone, and the released tape records as a fresh one.
    #[test]
    fn vtape_trim_releases_a_large_tape_and_keeps_a_small_one() {
        use crate::sens::num::PkNum;
        type D = DualV<1, 0>;
        let record = || {
            vtape_reset(2);
            let mut x = D::leaf(1.5, [1.0], []);
            for _ in 0..1000 {
                x = x * x * D::from_f64(0.5);
            }
            x
        };
        let _ = record();
        let grown = vtape_capacity();
        assert!(grown >= 1000, "the recording must be taped: {grown}");
        vtape_trim(1 << 20);
        assert_eq!(
            vtape_capacity(),
            grown,
            "a small tape must keep its buffers"
        );
        vtape_trim(100);
        assert_eq!(vtape_capacity(), 0, "a large tape must be released");
        let _ = record();
        assert!(vtape_len() >= 1000, "a released tape records afresh");
    }

    /// `axpy_fold`, `scaled_diff`, `caxpy_fold` and `dot_fold` on a taped `DualV`: the default per-operation
    /// expression's values bit for bit, and the same adjoints (one linear node against a
    /// node per operation), on a mix of taped and untaped operands.
    #[test]
    fn linear_folds_are_the_per_operation_expression_on_the_tape() {
        use crate::sens::num::PkNum;
        type D = DualV<2, 3>;
        let run = |fused: bool| -> (Vec<u64>, Vec<f64>) {
            vtape_reset(3);
            let a = D::leaf(1.5, [0.25, -1.0], [0.5, 0.0, -0.25]);
            let b = D::leaf(-0.75, [0.0, 2.0], [0.0, 1.0, 0.0]);
            let c = D::free(0.3, [1.0, -0.0], [0.0, 0.0, 2.0]);
            let terms = [(b, 0.125), (c, -3.0), (a, 1e-3)];
            let cterms = [(0.5, a), (-2.0, c), (7.0, b)];
            // `c` is untaped but carries tangents: its components ride beside the node.
            let pairs = [(a, b), (c, a), (b, b), (c, c)];
            let (x, y, z, q) = if fused {
                (
                    D::axpy_fold(a, &terms),
                    D::scaled_diff(b, a, 4.0),
                    D::caxpy_fold(c, &cterms),
                    D::dot_fold(b, &pairs),
                )
            } else {
                let mut acc = a;
                for &(t, w) in &terms {
                    acc = acc + t * D::from_f64(w);
                }
                let mut cacc = c;
                for &(w, t) in &cterms {
                    cacc = cacc + D::from_f64(w) * t;
                }
                let mut dacc = b;
                for &(p, r) in &pairs {
                    dacc = dacc + p * r;
                }
                (acc, (b - a) * D::from_f64(4.0), cacc, dacc)
            };
            let bits: Vec<u64> = [x, y, z, q]
                .iter()
                .flat_map(|d| std::iter::once(d.v).chain(d.g).chain(d.h))
                .map(f64::to_bits)
                .collect();
            let adj = vadj_init::<1>();
            for (d, s) in [(x, 0.7), (y, -1.3), (z, 0.4), (q, -0.9)] {
                adj[vadj_index::<2, 1>(d.i, 0, 0)] += s;
                adj[vadj_index::<2, 1>(d.i, 1, 0)] += 0.5 * s;
                adj[vadj_index::<2, 1>(d.i, 2, 0)] -= 0.25 * s;
            }
            let top = [x.i, y.i, z.i, q.i].into_iter().max().unwrap() as usize + 1;
            let r = vsweep::<2, 1>(top);
            let grads = [a.i, b.i]
                .iter()
                .flat_map(|&i| (0..3).map(move |c| (i, c)))
                .map(|(i, c)| r[vadj_index::<2, 1>(i, c, 0)])
                .collect();
            (bits, grads)
        };
        let (v_op, g_op) = run(false);
        let (v_fused, g_fused) = run(true);
        assert_eq!(v_op, v_fused, "values");
        for (k, (a, b)) in g_op.iter().zip(&g_fused).enumerate() {
            assert!(
                (a - b).abs() <= 1e-15 * (1.0 + a.abs()),
                "adjoint {k}: {a} vs {b}"
            );
        }
        assert!(
            g_op.iter().any(|&g| g != 0.0),
            "the sweep must reach the leaves"
        );
    }
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

    /// A constant exact zero divided by a taped value is the constant zero: no tape node and
    /// zero tangents, as a constant-zero factor in `mul_taped`. `ode::linear` extracts a
    /// generator column as `(f(s·e_j) − f(0))/s`, and `KA·x/V1` at `x = 0` reaches it as
    /// `0/V1`; a node there makes every entry of the row non-structural for the outer gradient.
    /// Mutation: drop the `(true, false)` zero arm in `div_taped` → `z.i` is a node.
    #[test]
    fn a_constant_zero_over_a_taped_value_is_an_untaped_zero() {
        vtape_reset(2);
        let v = DualV::<1, 0>::leaf(2.5, [0.0], []);
        assert_ne!(v.i, NONE, "premise: the divisor is on the tape");
        let z = DualV::<1, 0>::constant(0.0) / v;
        assert_eq!(z.i, NONE);
        assert!(z.v == 0.0 && z.g == [0.0]);
        assert!(z.is_constant());
        // A non-zero constant over a taped value still records.
        assert_ne!((DualV::<1, 0>::constant(1.0) / v).i, NONE);
    }

    /// #2481 §2.1: a sweep seeded on the value lane alone takes the value-lane fast path in
    /// `vsweep` and `vsweep_h`, node by node. Its leaf adjoints are then `∂f/∂leaf` by
    /// reverse mode, and the recording's own forward tangents are the same derivatives by
    /// forward mode — two independent routes that must agree to rounding, through every op
    /// `prog_h` records plus a `dot_fold` (`Op::Dot`), a negation and a shift. Leaves `a`, `b`
    /// carry the unit tangents; `x`, `y` none, so their adjoints are checked against a second
    /// recording that moves the tangents onto them. Mutation: drop the `c0 * bv` term of the
    /// fast path's `Op::Mul` (either sweep) → every leaf's adjoint is wrong.
    #[test]
    fn a_value_lane_sweep_is_the_forward_derivative() {
        fn prog_v<T: PkNum>(p0: T, p1: T, e0: T, e1: T) -> T {
            let f = prog_h(p0, p1, e0, e1);
            let d = T::dot_fold(f, &[(p0, e1), (f, p1)]);
            -(d * e0) + T::from_f64(1.5) + d
        }
        let base = [0.7, 1.3, -0.4, 0.25];
        for seeded in [[0usize, 1], [2, 3]] {
            let tangent = |k: usize| -> [f64; 2] {
                std::array::from_fn(|m| if seeded[m] == k { 1.0 } else { 0.0 })
            };
            // First order (`vsweep`).
            vtape_reset(3);
            let l: [DualV<2, 0>; 4] = std::array::from_fn(|k| DualV::leaf(base[k], tangent(k), []));
            let f = prog_v(l[0], l[1], l[2], l[3]);
            let adj = vadj_init::<1>();
            adj[vadj_index::<2, 1>(f.i, 0, 0)] = 1.0;
            let adj = vsweep::<2, 1>(f.i as usize + 1).to_vec();
            for (m, &k) in seeded.iter().enumerate() {
                let rev = adj[vadj_index::<2, 1>(l[k].i, 0, 0)];
                assert!(
                    (rev - f.g[m]).abs() <= 1e-12 * (1.0 + f.g[m].abs()),
                    "vsweep leaf {k}: reverse {rev} vs forward {}",
                    f.g[m]
                );
            }
            // Directional second order (`vsweep_h`), value lane seeded only.
            vtape_reset(1 + 2 + 2);
            let l: [DualV<2, 2, true>; 4] =
                std::array::from_fn(|k| DualV::leaf(base[k], tangent(k), [0.0; 2]));
            let f = prog_v(l[0], l[1], l[2], l[3]);
            let adj = vadj_init::<1>();
            adj[vadj_index_h::<2, 2>(f.i, 0)] = 1.0;
            let adj = vsweep_h::<2, 2>(f.i as usize + 1).to_vec();
            for (m, &k) in seeded.iter().enumerate() {
                let rev = adj[vadj_index_h::<2, 2>(l[k].i, 0)];
                assert!(
                    (rev - f.g[m]).abs() <= 1e-12 * (1.0 + f.g[m].abs()),
                    "vsweep_h leaf {k}: reverse {rev} vs forward {}",
                    f.g[m]
                );
            }
        }
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

    /// #2093 review: `c − K` with an untaped `c` is recorded as a `Neg` of `K`, so two such
    /// expressions with equal values (`exp(η₁) − K` and `exp(η₂) − K` at η₁ = η₂) have the
    /// same op, parent and value but different η-tangents. Sharing one node made the sweep
    /// read the first one's tangents: `∂/∂K` of `∂(d₂²)/∂η₂` came out 0 instead of −2(…).
    /// The mutation this catches is dropping the stored-row comparison from `push_cse`.
    #[test]
    fn cse_does_not_alias_records_with_different_tangents() {
        vtape_reset(3);
        let k = DualV::<2, 0>::leaf(0.4, [0.0, 0.0], []);
        let c1 = DualV::<2, 0>::free(1.3, [1.0, 0.0], []);
        let c2 = DualV::<2, 0>::free(1.3, [0.0, 1.0], []);
        let d1 = c1 - k;
        let d2 = c2 - k;
        assert_eq!(
            d1.v.to_bits(),
            d2.v.to_bits(),
            "the straddle needs equal values"
        );
        assert_ne!(
            d1.i, d2.i,
            "records with different tangents must not share a node"
        );
        // d₂² has η₂-tangent 2·d₂·1; its K-derivative is −2. Seed that tangent component.
        let out = d2 * d2;
        let buf = vadj_init::<1>();
        buf[vadj_index::<2, 1>(out.i, 2, 0)] = 1.0;
        let adj = vsweep::<2, 1>(out.i as usize + 1);
        let dk = adj[vadj_index::<2, 1>(k.i, 0, 0)];
        assert!(
            (dk + 2.0).abs() < 1e-12,
            "∂/∂K of the η₂-tangent: {dk}, want −2"
        );
        // A unary rule with equal value and φ′ but a different φ″ is a different record.
        vtape_reset(3);
        let x = DualV::<2, 0>::leaf(0.7, [1.0, 0.0], []);
        let u1 = x.lift2(2.0, 3.0, 5.0);
        let u2 = x.lift2(2.0, 3.0, 7.0);
        assert_ne!(u1.i, u2.i, "a different φ″ is a different unary record");
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
