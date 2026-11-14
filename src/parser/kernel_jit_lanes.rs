//! The cold tier of a chunk kernel (#2786): its register program run one dual operation at a
//! time over a [`Layout`]'s lanes, forward and reverse.
//!
//! The lowered scalar code ([`ir::forward`]) is the compiled tiers' input; interpreting it
//! costs a dispatch per lane of every operation. This runs the chunk's [`Flat`] code itself:
//! one dispatch per operation, the lane arithmetic in loops over the layout's tangents and
//! payload pairs ([`ir::pairs`]) — the same formulas, in the same order, as [`ir::Lower`]'s,
//! so the forward lanes are the lowered code's. The reverse applies each operation's adjoint
//! rule, derived from those formulas, in reverse order over the stored forward registers;
//! like [`ir::reverse`], a select passes its adjoint to the operand it chose and a condition,
//! a rounding, a stopped value or a host output carries none.
//!
//! No input lane is assumed: the folding the compiled tiers apply ([`super::fold_chunk`]) is
//! exact, so the lanes agree with theirs wherever no `NaN` or signed zero is involved.
use super::super::{Flat, Unary};
use super::ir::{self, Cc, Layout, Shim1};
use super::DirMask;
use crate::stats::special::gamma_pq_partial;

/// A layout's lane structure: `n` tangents, then the payload pairs `(k, l, lane)`.
pub(super) struct Shape {
    pub(super) lanes: usize,
    n: usize,
    second: bool,
    pairs: Vec<(usize, usize, usize)>,
}

impl Shape {
    pub(super) fn of(layout: Layout) -> Self {
        let n = layout.n as usize;
        let mut pairs = Vec::new();
        for k in 0..n {
            for (l, t) in ir::pairs(&layout, k) {
                pairs.push((k, l, 1 + n + t));
            }
        }
        Shape {
            lanes: layout.lanes(),
            n,
            second: layout.m > 0,
            pairs,
        }
    }
}

/// The value lane alone: what an operation whose result has no live direction computes (its
/// operands have none either, so each formula's other lanes read structural zeros).
static VALUE: Shape = Shape {
    lanes: 1,
    n: 0,
    second: false,
    pairs: Vec::new(),
};

/// A chunk's program at one layout: its lane structure, and which registers can have a
/// non-zero lane past the value — from the inputs' live directions (`spec`, as
/// [`super::fold_chunk`] reads them) carried through the program.
pub(super) struct AtLayout {
    pub(super) sh: Shape,
    /// Per operation: its result has no live direction.
    dead: Vec<bool>,
}

impl AtLayout {
    pub(super) fn of(prog: &LaneProg, layout: Layout, spec: &[(DirMask, u16)]) -> Self {
        let n = layout.n as usize;
        let mut m = vec![DirMask::NONE; prog.n_regs];
        let get = |m: &[DirMask], r: u32| m[r as usize].clone();
        for f in &prog.code {
            let (d, v) = match *f {
                Flat::LoadState { d, i } => (d, spec[i as usize].0.clone()),
                Flat::Add { d, a, b }
                | Flat::Sub { d, a, b }
                | Flat::RSub { d, a, b }
                | Flat::Mul { d, a, b }
                | Flat::Div { d, a, b }
                | Flat::DivRaw { d, a, b } => (d, get(&m, a) | &m[b as usize]),
                Flat::Pow { d, a, e: b } | Flat::GammaPq { d, a, x: b, .. } => {
                    (d, get(&m, a) | &m[b as usize])
                }
                Flat::Select { d, a, b, .. } => (d, get(&m, a) | &m[b as usize]),
                Flat::AddK { d, a, .. }
                | Flat::SubK { d, a, .. }
                | Flat::RSubK { d, a, .. }
                | Flat::MulK { d, a, .. }
                | Flat::DivK { d, a, .. } => (d, get(&m, a)),
                Flat::Unary { d, op, a } => match op {
                    Unary::StopGrad | Unary::Floor | Unary::Ceil | Unary::Round => {
                        (d, DirMask::NONE)
                    }
                    _ => (d, get(&m, a)),
                },
                Flat::LoadSeed { d, .. }
                | Flat::LoadK { d, .. }
                | Flat::Cmp { d, .. }
                | Flat::And { d, .. }
                | Flat::Or { d, .. }
                | Flat::Not { d, .. }
                | Flat::Present { d, .. }
                | Flat::HostOut { d, .. } => (d, DirMask::NONE),
                _ => continue,
            };
            m[d as usize] = v;
        }
        let live: Vec<bool> = m.iter().map(|v| v.any_below(n)).collect();
        AtLayout {
            sh: Shape::of(layout),
            dead: prog
                .code
                .iter()
                .map(|f| def(f).is_some_and(|d| !live[d as usize]))
                .collect(),
        }
    }
}

/// A chunk's program for this tier, or `None` when it has an `if` (the lowered code's arms).
pub(super) struct LaneProg {
    code: Vec<Flat>,
    n_regs: usize,
    n_in: usize,
}

impl LaneProg {
    pub(super) fn of(code: &[Flat], n_regs: usize, n_in: usize) -> Option<Self> {
        if code.iter().any(|f| {
            matches!(
                f,
                Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. }
            )
        }) {
            return None;
        }
        Some(LaneProg {
            code: code.to_vec(),
            n_regs,
            n_in,
        })
    }

    pub(super) fn n_regs(&self) -> usize {
        self.n_regs
    }

    /// The forward over `regs` (`n_regs × lanes`): inputs from `base` through `idx`, literals
    /// from `lits`, outputs stored to `base` through `idx[n_in..]`.
    ///
    /// # Safety
    /// As a chunk kernel's call: every slot in `idx` addresses a lane block of `base`.
    pub(super) unsafe fn forward(
        &self,
        at: &AtLayout,
        wide: bool,
        regs: &mut [f64],
        base: *mut f64,
        idx: *const u32,
        lits: *const f64,
        store: bool,
        scratch: &mut [f64],
    ) {
        let l = at.sh.lanes;
        let slot = |e: usize| base.add(*idx.add(e) as usize * l);
        // A register program is in SSA form: no operation's destination is its operand, so
        // the destination's lanes are written in place while the operands' are read.
        let rp = regs.as_mut_ptr();
        let rd = |r: u32| std::slice::from_raw_parts(rp.add(r as usize * l), l);
        let wd = |r: u32| std::slice::from_raw_parts_mut(rp.add(r as usize * l), l);
        let (t2, t3) = scratch[..2 * l].split_at_mut(l);
        for (f, &dead) in self.code.iter().zip(&at.dead) {
            // A result with no live direction: its value lane alone, its others zero.
            let sh = if dead { &VALUE } else { &at.sh };
            match *f {
                Flat::LoadState { d, i } => {
                    let src = std::slice::from_raw_parts(slot(i as usize), l);
                    wd(d).copy_from_slice(src);
                }
                Flat::LoadSeed { d, k } => konst(wd(d), *lits.add(k as usize)),
                Flat::LoadK { d, c } => konst(wd(d), c),
                Flat::Add { d, a, b } => lanewise(wd(d), rd(a), rd(b), |x, y| x + y),
                Flat::Sub { d, a, b } => lanewise(wd(d), rd(a), rd(b), |x, y| x - y),
                Flat::RSub { d, a, b } => lanewise(wd(d), rd(b), rd(a), |x, y| x - y),
                Flat::Mul { d, a, b } => {
                    mul(sh, rd(a), rd(b), wd(d));
                }
                Flat::DivRaw { d, a, b } => {
                    div(sh, rd(a), rd(b), wd(d));
                }
                Flat::Div { d, a, b } => {
                    if rd(b)[0].abs() < 1e-30 {
                        konst(wd(d), 0.0);
                    } else {
                        div(sh, rd(a), rd(b), wd(d));
                    }
                }
                Flat::AddK { d, a, c } => {
                    let o = wd(d);
                    o.copy_from_slice(rd(a));
                    o[0] += c;
                }
                Flat::SubK { d, a, c } => {
                    let o = wd(d);
                    o.copy_from_slice(rd(a));
                    o[0] -= c;
                }
                Flat::RSubK { d, a, c } => {
                    let (o, x) = (wd(d), rd(a));
                    o[0] = c - x[0];
                    for j in 1..l {
                        o[j] = 0.0 - x[j];
                    }
                }
                Flat::MulK { d, a, c } => {
                    if wide {
                        konst(t2, c);
                        mul(sh, rd(a), t2, wd(d));
                    } else {
                        for (o, &x) in wd(d).iter_mut().zip(rd(a)) {
                            *o = x * c;
                        }
                    }
                }
                Flat::DivK { d, a, c } => {
                    let o = wd(d);
                    if c.abs() < 1e-30 {
                        konst(o, 0.0);
                    } else if wide {
                        konst(t2, c);
                        div(sh, rd(a), t2, o);
                    } else {
                        let x = rd(a);
                        o[0] = x[0] / c;
                        if l > 1 {
                            let ib = 1.0 / c;
                            for j in 1..l {
                                o[j] = x[j] * ib;
                            }
                        }
                    }
                }
                Flat::Pow { d, a, e } => pow_fwd(sh, rd(a), rd(e), wd(d), t2, t3),
                Flat::GammaPq {
                    d,
                    a,
                    x,
                    upper,
                    i,
                    j,
                } => {
                    let p = |di: u8, dj: u8| {
                        gamma_pq_partial(rd(a)[0], rd(x)[0], upper, i + di, j + dj)
                    };
                    let parts = prim_parts(sh, &p);
                    prim(sh, rd(a), rd(x), &parts, wd(d));
                }
                Flat::Unary { d, op, a } => {
                    unary_fwd(sh, op, rd(a), wd(d), t2);
                }
                Flat::Deriv { state, a } => {
                    if store {
                        let dst =
                            std::slice::from_raw_parts_mut(slot(self.n_in + state as usize), l);
                        for (o, &x) in dst.iter_mut().zip(rd(a)) {
                            *o = super::canonical_zero(x);
                        }
                    }
                }
                Flat::Cmp { d, cc, a, b } => {
                    let v = holds(cc, rd(a)[0], rd(b)[0]);
                    konst(wd(d), v as u8 as f64);
                }
                Flat::And { d, a, b } => {
                    let v = rd(a)[0] != 0.0 && rd(b)[0] != 0.0;
                    konst(wd(d), v as u8 as f64);
                }
                Flat::Or { d, a, b } => {
                    let v = rd(a)[0] != 0.0 || rd(b)[0] != 0.0;
                    konst(wd(d), v as u8 as f64);
                }
                Flat::Not { d, a } => {
                    let v = rd(a)[0] == 0.0;
                    konst(wd(d), v as u8 as f64);
                }
                Flat::Present { d, a } => {
                    // Ordered equality is false exactly for a `NaN`.
                    let x = rd(a)[0];
                    konst(wd(d), !x.is_nan() as u8 as f64);
                }
                Flat::HostArg { a } => ir::host::arg(rd(a)[0]),
                Flat::Host { f } => ir::host::call(f),
                Flat::HostOut { d, k } => konst(wd(d), ir::host::out(k)),
                Flat::Select { d, c, a, b } => {
                    let src = if rd(c)[0] != 0.0 { a } else { b };
                    wd(d).copy_from_slice(rd(src));
                }
                Flat::Gate { .. } => unreachable!("resolved before the lane program"),
                Flat::LoadParam { .. }
                | Flat::LoopArg { .. }
                | Flat::Loop { .. }
                | Flat::LoopOut { .. }
                | Flat::Branch { .. }
                | Flat::Jump { .. }
                | Flat::Move { .. } => unreachable!("not in a chunk (`LaneProg::of`)"),
            }
            if dead {
                if let Some(d) = def(f) {
                    wd(d)[1..].fill(0.0);
                }
            }
        }
    }
}

#[inline]
fn konst(out: &mut [f64], v: f64) {
    out.fill(0.0);
    out[0] = v;
}

#[inline]
fn lanewise(out: &mut [f64], a: &[f64], b: &[f64], op: impl Fn(f64, f64) -> f64) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *o = op(x, y);
    }
}

fn holds(cc: Cc, a: f64, b: f64) -> bool {
    cc.holds(a, b)
}

/// `mul_body`: `g_k = a_k·b + a·b_k`, `h_t = ((a_t·b + a·b_t) + a_k·b_l) + a_l·b_k`.
fn mul(sh: &Shape, a: &[f64], b: &[f64], out: &mut [f64]) {
    out[0] = a[0] * b[0];
    for k in 0..sh.n {
        out[1 + k] = a[1 + k] * b[0] + a[0] * b[1 + k];
    }
    for &(k, l, h) in &sh.pairs {
        out[h] = a[h] * b[0] + a[0] * b[h] + a[1 + k] * b[1 + l] + a[1 + l] * b[1 + k];
    }
}

/// `div_body`: `q = a/b`, `ib = 1/b`, `g_k = (a_k − q·b_k)·ib`,
/// `h_t = (((a_t − g_k·b_l) − g_l·b_k) − q·b_t)·ib`.
fn div(sh: &Shape, a: &[f64], b: &[f64], out: &mut [f64]) {
    let q = a[0] / b[0];
    out[0] = q;
    if sh.lanes == 1 {
        return;
    }
    let ib = 1.0 / b[0];
    for k in 0..sh.n {
        out[1 + k] = (a[1 + k] - q * b[1 + k]) * ib;
    }
    for &(k, l, h) in &sh.pairs {
        out[h] = (a[h] - out[1 + k] * b[1 + l] - out[1 + l] * b[1 + k] - q * b[h]) * ib;
    }
}

/// `unary_body` from `φ(v) = h0`, `h1 = φ′`, `h2 = φ″`: `g_k = h1·g_k`,
/// `h_t = h1·h_t + (h2·g_k)·g_l`.
fn unary(sh: &Shape, a: &[f64], h: [f64; 3], out: &mut [f64]) {
    out[0] = h[0];
    for k in 0..sh.n {
        out[1 + k] = h[1] * a[1 + k];
    }
    for &(k, l, t) in &sh.pairs {
        out[t] = h[1] * a[t] + h[2] * a[1 + k] * a[1 + l];
    }
}

/// A unary operation's forward (`t` scratch).
fn unary_fwd(sh: &Shape, op: Unary, a: &[f64], out: &mut [f64], t: &mut [f64]) {
    let x = a[0];
    match op {
        Unary::StopGrad => konst(out, x),
        Unary::Floor => konst(out, x.floor()),
        Unary::Ceil => konst(out, x.ceil()),
        Unary::Round => konst(out, x.round()),
        Unary::Abs => {
            if sh.lanes == 1 {
                out[0] = x.abs();
            } else if x >= 0.0 {
                out.copy_from_slice(a);
            } else {
                for (o, &v) in out.iter_mut().zip(a) {
                    *o = -v;
                }
            }
        }
        Unary::Ln => {
            // `guard_floor(1e-30)`: below it (or `NaN`), `ln` of the constant floor.
            if x >= 1e-30 {
                let h = derivs(Unary::LnRaw, x, sh.second);
                unary(sh, a, [h[0], h[1], h[2]], out);
            } else {
                konst(t, 1e-30);
                let h = derivs(Unary::LnRaw, 1e-30, sh.second);
                unary(sh, t, [h[0], h[1], h[2]], out);
            }
        }
        _ => {
            let h = derivs(op, x, sh.second);
            if sh.lanes == 1 {
                out[0] = h[0];
            } else {
                unary(sh, a, [h[0], h[1], h[2]], out);
            }
        }
    }
}

/// `(φ, φ′, φ″)` as [`ir::Lower`] computes them (`φ″` only for a second-order layout).
fn derivs(op: Unary, x: f64, second: bool) -> [f64; 3] {
    let sh = |s: Shim1| (s.f())(x);
    match op {
        Unary::Exp => {
            let e = x.exp();
            [e, e, e]
        }
        Unary::LnRaw => [x.ln(), 1.0 / x, if second { -1.0 / (x * x) } else { 0.0 }],
        Unary::Sqrt | Unary::SqrtRhs => {
            let x = if op == Unary::SqrtRhs && !(x >= 0.0) {
                0.0
            } else {
                x
            };
            let s = x.sqrt();
            // Flat at a non-positive argument (#2892): no `0.5/0`.
            if !(x > 0.0) {
                [s, 0.0, 0.0]
            } else {
                [s, 0.5 / s, if second { -0.25 / (s * s * s) } else { 0.0 }]
            }
        }
        Unary::InvLogit => {
            let u = 1.0 / (1.0 + (-x).exp());
            let d1 = u * (1.0 - u);
            [u, d1, if second { d1 * (1.0 - 2.0 * u) } else { 0.0 }]
        }
        Unary::Logit => {
            let omx = 1.0 - x;
            let v = (x / omx).ln();
            let d1 = 1.0 / (x * omx);
            [
                v,
                d1,
                if second {
                    -(1.0 - 2.0 * x) * d1 * d1
                } else {
                    0.0
                },
            ]
        }
        Unary::Cos => {
            let c = x.cos();
            [c, -x.sin(), -c]
        }
        Unary::Acos => {
            let s = 1.0 - x * x;
            let rs = s.sqrt();
            [
                x.acos(),
                -1.0 / rs,
                if second { -x / (s * rs) } else { 0.0 },
            ]
        }
        Unary::LnGamma => [
            sh(Shim1::LnGamma),
            sh(Shim1::Digamma),
            if second { sh(Shim1::Trigamma) } else { 0.0 },
        ],
        Unary::Sin => [x.sin(), x.cos(), -x.sin()],
        Unary::Digamma => chain(x, |k| Shim1::polygamma(1 + k), second),
        Unary::Trigamma => chain(x, |k| Shim1::polygamma(2 + k), second),
        Unary::Tetragamma => chain(x, |k| Shim1::polygamma(3 + k), second),
        Unary::ExpRel(k) => chain(x, |j| Shim1::ExpRel(k + j), second),
        Unary::Jet(f, k) => chain(x, |j| Shim1::Jet(f, k + j), second),
        Unary::Ln | Unary::Abs | Unary::StopGrad | Unary::Floor | Unary::Ceil | Unary::Round => {
            unreachable!("handled by `unary_fwd`")
        }
    }
}

fn chain(x: f64, shim: impl Fn(u8) -> Shim1, second: bool) -> [f64; 3] {
    [
        (shim(0).f())(x),
        (shim(1).f())(x),
        if second { (shim(2).f())(x) } else { 0.0 },
    ]
}

/// A binary primitive's partials `[f, f_a, f_x, f_aa, f_ax, f_xx]` (the second order only
/// for a second-order layout).
fn prim_parts(sh: &Shape, p: &dyn Fn(u8, u8) -> f64) -> [f64; 6] {
    let second = if sh.second {
        [p(2, 0), p(1, 1), p(0, 2)]
    } else {
        [0.0; 3]
    };
    let f = p(0, 0);
    if sh.lanes == 1 {
        return [f, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    [f, p(1, 0), p(0, 1), second[0], second[1], second[2]]
}

/// `binary_prim`: `g_k = f_a·a_k + f_x·x_k`, `h_t = f_a·a_t + f_x·x_t + f_aa·a_k·a_l
/// + f_ax·(a_k·x_l + a_l·x_k) + f_xx·x_k·x_l`.
fn prim(sh: &Shape, a: &[f64], x: &[f64], p: &[f64; 6], out: &mut [f64]) {
    let [f, fa, fx, faa, fax, fxx] = *p;
    out[0] = f;
    for k in 0..sh.n {
        out[1 + k] = fa * a[1 + k] + fx * x[1 + k];
    }
    for &(k, l, h) in &sh.pairs {
        out[h] = fa * a[h]
            + fx * x[h]
            + faa * (a[1 + k] * a[1 + l])
            + fax * (a[1 + k] * x[1 + l] + a[1 + l] * x[1 + k])
            + fxx * (x[1 + k] * x[1 + l]);
    }
}

/// the forward dual's `pow` (see `ir::Lower::pow`): a constant exponent lifts `powf` (the constant
/// `powf(a, n)` at `a = 0`); one with a tangent is `exp(e · ln a)`.
fn pow_fwd(sh: &Shape, a: &[f64], e: &[f64], out: &mut [f64], t1: &mut [f64], t2: &mut [f64]) {
    let (av, n) = (a[0], e[0]);
    if sh.lanes == 1 {
        out[0] = av.powf(n);
        return;
    }
    let const_exp = e[1..=sh.n].iter().all(|&x| x == 0.0);
    if const_exp {
        let pv = av.powf(n);
        if av == 0.0 {
            konst(out, pv);
            return;
        }
        let h1 = n * av.powf(n - 1.0);
        let h2 = if sh.second {
            (n * (n - 1.0)) * av.powf(n - 2.0)
        } else {
            0.0
        };
        unary(sh, a, [pv, h1, h2], out);
    } else {
        // `exp(e · ln a)`.
        let h = derivs(Unary::LnRaw, av, sh.second);
        unary(sh, a, h, t1);
        mul(sh, e, t1, t2);
        let x = t2[0].exp();
        unary(sh, t2, [x, x, x], out);
    }
}

#[cfg(test)]
#[path = "kernel_jit_lanes_tests.rs"]
mod tests;

/// The register an operation defines.
fn def(f: &Flat) -> Option<u32> {
    match *f {
        Flat::LoadState { d, .. }
        | Flat::LoadSeed { d, .. }
        | Flat::LoadK { d, .. }
        | Flat::Add { d, .. }
        | Flat::Sub { d, .. }
        | Flat::RSub { d, .. }
        | Flat::Mul { d, .. }
        | Flat::Div { d, .. }
        | Flat::DivRaw { d, .. }
        | Flat::AddK { d, .. }
        | Flat::SubK { d, .. }
        | Flat::RSubK { d, .. }
        | Flat::MulK { d, .. }
        | Flat::DivK { d, .. }
        | Flat::Pow { d, .. }
        | Flat::GammaPq { d, .. }
        | Flat::Unary { d, .. }
        | Flat::Cmp { d, .. }
        | Flat::And { d, .. }
        | Flat::Or { d, .. }
        | Flat::Not { d, .. }
        | Flat::Present { d, .. }
        | Flat::HostOut { d, .. }
        | Flat::Select { d, .. } => Some(d),
        _ => None,
    }
}
