//! The register program over one number type's lanes, as scalar straight-line code.
//!
//! A [`super::RhsRegProgram`] runs over `f64` or a dual whose value and derivative parts
//! are all `f64`s ("lanes"). Every arithmetic operation of such a dual is itself a short
//! sequence of `f64` operations — `DualV`'s `mul_body` is `g = a.g·b.v + a.v·b.g` per
//! tangent, and so on — so the whole program, over a given [`Layout`], is a straight-line
//! program over `f64`s: loads of the inputs' lanes, IEEE arithmetic, selects for the
//! guards, calls for `exp`/`ln`/`powf`/`round`, and stores of the derivatives' lanes.
//!
//! That one form serves two consumers:
//!
//! - **the native code** (`super::jit`) lowers it to Cranelift, one instruction each;
//! - **the reverse pass** ([`reverse`]) is its mechanical adjoint: the forward
//!   instructions, then their local derivatives in reverse order, accumulating into the
//!   inputs' adjoints. Because the form is scalar, one transform differentiates every
//!   layout — `f64`, first-order duals, second-order payloads — with no per-type rules.
//!
//! **The forward instructions are each dual type's own arithmetic, in its own order** (see
//! [`Lanes`]): the operand association, `0 − g` for a reversed subtraction, `ib = 1/b`
//! computed once, the out-of-line duals' full product for a constant operand. Lowered
//! without contraction or folding, the native code is bit for bit the register code.
use super::{Arith, Flat, Unary};

/// An instruction's index in its [`Func`].
pub(super) type V = u32;

/// A call the native code makes into Rust, so the value is the interpreter's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shim1 {
    Exp,
    Ln,
    Round,
}

/// A comparison, producing a condition for [`Inst::Select`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cc {
    Lt,
    Gt,
    Ge,
    Eq,
}

/// One scalar instruction. `f64`-valued unless it is a condition ([`Inst::Cmp`],
/// [`Inst::And`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Inst {
    /// `arg[off]`.
    Load {
        arg: u8,
        off: u32,
    },
    /// A literal, by its bits.
    Const(u64),
    Add(V, V),
    Sub(V, V),
    Mul(V, V),
    Div(V, V),
    Neg(V),
    Abs(V),
    Floor(V),
    Ceil(V),
    Call1(Shim1, V),
    /// `powf(a, e)`.
    Powf(V, V),
    Cmp(Cc, V, V),
    And(V, V),
    /// `cond ? a : b`.
    Select(V, V, V),
    /// `arg[off] = v`.
    Store {
        arg: u8,
        off: u32,
        v: V,
    },
}

/// A straight-line function of `n_args` pointer arguments.
#[derive(Clone, Debug, Default)]
pub(super) struct Func {
    pub(super) insts: Vec<Inst>,
    pub(super) n_args: u8,
}

impl Func {
    pub(super) fn new(n_args: u8) -> Self {
        Func {
            insts: Vec::new(),
            n_args,
        }
    }
    fn push(&mut self, i: Inst) -> V {
        self.insts.push(i);
        (self.insts.len() - 1) as V
    }
    fn k(&mut self, c: f64) -> V {
        self.push(Inst::Const(c.to_bits()))
    }
}

pub(crate) use crate::sens::num::vm::Layout;

/// The payload entries of row `k`: `(l, t)` with `h_t = ∂²/∂η_k∂η_l` (`vtape::hrange`/
/// `hidx`).
fn pairs(layout: &Layout, k: usize) -> Vec<(usize, usize)> {
    if layout.m == 0 {
        Vec::new()
    } else if layout.th {
        vec![(k, k)]
    } else {
        (0..=k).map(|l| (l, k * (k + 1) / 2 + l)).collect()
    }
}

/// The widest dual the native code serves (lanes); a wider one keeps the register code.
pub(super) const MAX_LANES: usize = 256;

/// The direction counts a native function can be built for.
const MAX_N: usize = 48;

/// Every layout a native function can be built for has a fixed slot, so a call finds its
/// function with one index instead of a lookup: `f64`, then the first-order duals, the
/// packed second-order ones and the directional ones, each by direction count and by
/// whether its constant operands run as full products (`wide`).
pub(super) const N_SLOTS: usize = 2 * (1 + 3 * MAX_N);

pub(super) const fn slot(l: Layout) -> Option<usize> {
    let n = l.n as usize;
    let base = if n == 0 {
        if l.m == 0 {
            0
        } else {
            return None;
        }
    } else {
        if n > MAX_N || l.lanes() > MAX_LANES {
            return None;
        }
        let m = l.m as usize;
        if m == 0 && !l.th {
            n
        } else if !l.th && m == n * (n + 1) / 2 {
            MAX_N + n
        } else if l.th && m == n {
            2 * MAX_N + n
        } else {
            return None;
        }
    };
    Some(2 * base + l.wide as usize)
}

/// One value's lanes.
type Lanes = Vec<V>;

/// Emits [`Flat`] instructions over a [`Layout`]'s lanes into a [`Func`].
pub(super) struct Lower<'a> {
    pub(super) f: &'a mut Func,
    pub(super) layout: Layout,
    regs: Vec<Lanes>,
}

impl<'a> Lower<'a> {
    pub(super) fn new(f: &'a mut Func, layout: Layout, n_regs: usize) -> Self {
        Lower {
            f,
            layout,
            regs: vec![Vec::new(); n_regs],
        }
    }
    fn lanes(&self) -> usize {
        self.layout.lanes()
    }
    fn n(&self) -> usize {
        self.layout.n as usize
    }
    fn k(&mut self, c: f64) -> V {
        self.f.k(c)
    }
    fn ins(&mut self, i: Inst) -> V {
        self.f.push(i)
    }
    fn constant(&mut self, v: V) -> Lanes {
        let mut l = vec![v];
        for _ in 1..self.lanes() {
            let z = self.k(0.0);
            l.push(z);
        }
        l
    }
    pub(super) fn load(&mut self, arg: u8, index: usize) -> Lanes {
        let lanes = self.lanes();
        (0..lanes)
            .map(|j| {
                self.ins(Inst::Load {
                    arg,
                    off: (index * lanes + j) as u32,
                })
            })
            .collect()
    }
    pub(super) fn store(&mut self, arg: u8, index: usize, l: &[V]) {
        let lanes = self.lanes();
        for (j, &v) in l.iter().enumerate() {
            self.ins(Inst::Store {
                arg,
                off: (index * lanes + j) as u32,
                v,
            });
        }
    }
    pub(super) fn reg(&self, r: u16) -> Lanes {
        self.regs[r as usize].clone()
    }
    pub(super) fn set(&mut self, d: u16, l: Lanes) {
        self.regs[d as usize] = l;
    }
    fn select(&mut self, c: V, a: &[V], b: &[V]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.ins(Inst::Select(c, x, y)))
            .collect()
    }

    // ── The primitives, as `f64` and `DualV`'s untaped bodies compute them (`+` and `−`
    // are lane by lane, [`Alg`]'s). ──

    /// `mul_body`: `g_k = a.g_k·b.v + a.v·b.g_k`,
    /// `h_t = a.h_t·b.v + a.v·b.h_t + a.g_k·b.g_l + a.g_l·b.g_k`, left to right.
    fn mul(&mut self, a: &[V], b: &[V]) -> Lanes {
        let n = self.n();
        let mut out = vec![0; self.lanes()];
        out[0] = self.ins(Inst::Mul(a[0], b[0]));
        for k in 0..n {
            let x = self.ins(Inst::Mul(a[1 + k], b[0]));
            let y = self.ins(Inst::Mul(a[0], b[1 + k]));
            out[1 + k] = self.ins(Inst::Add(x, y));
            for (l, t) in pairs(&self.layout, k) {
                let h = 1 + n + t;
                let p = self.ins(Inst::Mul(a[h], b[0]));
                let q = self.ins(Inst::Mul(a[0], b[h]));
                let s = self.ins(Inst::Add(p, q));
                let r = self.ins(Inst::Mul(a[1 + k], b[1 + l]));
                let s = self.ins(Inst::Add(s, r));
                let r = self.ins(Inst::Mul(a[1 + l], b[1 + k]));
                out[h] = self.ins(Inst::Add(s, r));
            }
        }
        out
    }
    /// `div_body`: `q = a/b`, `ib = 1/b`, `g_k = (a.g_k − q·b.g_k)·ib`,
    /// `h_t = (a.h_t − g_k·b.g_l − g_l·b.g_k − q·b.h_t)·ib`.
    fn div(&mut self, a: &[V], b: &[V]) -> Lanes {
        let n = self.n();
        let mut out = vec![0; self.lanes()];
        let q = self.ins(Inst::Div(a[0], b[0]));
        out[0] = q;
        if self.lanes() == 1 {
            return out;
        }
        let one = self.k(1.0);
        let ib = self.ins(Inst::Div(one, b[0]));
        for k in 0..n {
            let t = self.ins(Inst::Mul(q, b[1 + k]));
            let d = self.ins(Inst::Sub(a[1 + k], t));
            out[1 + k] = self.ins(Inst::Mul(d, ib));
        }
        for k in 0..n {
            for (l, t) in pairs(&self.layout, k) {
                let h = 1 + n + t;
                let x = self.ins(Inst::Mul(out[1 + k], b[1 + l]));
                let s = self.ins(Inst::Sub(a[h], x));
                let x = self.ins(Inst::Mul(out[1 + l], b[1 + k]));
                let s = self.ins(Inst::Sub(s, x));
                let x = self.ins(Inst::Mul(q, b[h]));
                let s = self.ins(Inst::Sub(s, x));
                out[h] = self.ins(Inst::Mul(s, ib));
            }
        }
        out
    }
    /// `x / y`, or the constant zero when `|y| < 1e-30` (`apply_binary_in`'s guard).
    ///
    /// The quotient the guard discards divides by `1` instead of the tiny `y`: the same
    /// values wherever it is kept, and finite where it is not, so the reverse pass's zero
    /// adjoint through the discarded arm stays zero (`0·∞` would be `NaN`).
    fn guarded_div(&mut self, a: &[V], b: &[V]) -> Lanes {
        let ay = self.ins(Inst::Abs(b[0]));
        let tiny = self.k(1e-30);
        let small = self.ins(Inst::Cmp(Cc::Lt, ay, tiny));
        let one = self.k(1.0);
        let mut bs = b.to_vec();
        bs[0] = self.ins(Inst::Select(small, one, b[0]));
        let q = self.div(a, &bs);
        let z = self.k(0.0);
        let zero = self.constant(z);
        self.select(small, &zero, &q)
    }
    /// A constant right operand: `f64`'s operator, or the dual's `add_c`/`sub_c`/
    /// `rsub_c`/`mul_c`/`div_c`.
    fn scalar(&mut self, op: Arith, a: &[V], c: f64) -> Lanes {
        let cv = self.k(c);
        let mut out = a.to_vec();
        match op {
            Arith::Add => out[0] = self.ins(Inst::Add(a[0], cv)),
            Arith::Sub => out[0] = self.ins(Inst::Sub(a[0], cv)),
            Arith::RSub => {
                // `constant(c) − self`: `0 − g`, which is `+0` where `−g` would be `−0`.
                out[0] = self.ins(Inst::Sub(cv, a[0]));
                for x in out.iter_mut().skip(1) {
                    let z = self.f.k(0.0);
                    *x = self.f.push(Inst::Sub(z, *x));
                }
            }
            Arith::Mul if self.layout.wide => {
                let c = self.constant(cv);
                return self.mul(a, &c);
            }
            Arith::Mul => {
                for x in out.iter_mut() {
                    *x = self.f.push(Inst::Mul(*x, cv));
                }
            }
            Arith::Div if self.layout.wide => {
                let c = self.constant(cv);
                return self.div(a, &c);
            }
            Arith::Div => {
                out[0] = self.ins(Inst::Div(a[0], cv));
                if self.lanes() > 1 {
                    let one = self.k(1.0);
                    let ib = self.ins(Inst::Div(one, cv));
                    for x in out.iter_mut().skip(1) {
                        *x = self.f.push(Inst::Mul(*x, ib));
                    }
                }
            }
        }
        out
    }
    /// `unary_body`: `φ(v) = hv`, `g_k = h1·g_k`, `h_t = h1·h_t + h2·g_k·g_l`.
    fn unary(&mut self, a: &[V], hv: V, h1: V, h2: Option<V>) -> Lanes {
        let n = self.n();
        let mut out = vec![0; self.lanes()];
        out[0] = hv;
        for k in 0..n {
            out[1 + k] = self.ins(Inst::Mul(h1, a[1 + k]));
            for (l, t) in pairs(&self.layout, k) {
                let h = 1 + n + t;
                let h2 = h2.expect("a payload has its second derivative");
                let x = self.ins(Inst::Mul(h1, a[h]));
                let y = self.ins(Inst::Mul(h2, a[1 + k]));
                let y = self.ins(Inst::Mul(y, a[1 + l]));
                out[h] = self.ins(Inst::Add(x, y));
            }
        }
        out
    }
    fn exp(&mut self, a: &[V]) -> Lanes {
        let e = self.ins(Inst::Call1(Shim1::Exp, a[0]));
        self.unary(a, e, e, Some(e))
    }
    /// `DualV::ln`, unguarded: `h1 = 1/x`, `h2 = −1/(x·x)`.
    fn ln(&mut self, a: &[V]) -> Lanes {
        let l = self.ins(Inst::Call1(Shim1::Ln, a[0]));
        if self.lanes() == 1 {
            return vec![l];
        }
        let one = self.k(1.0);
        let h1 = self.ins(Inst::Div(one, a[0]));
        let h2 = (self.layout.m > 0).then(|| {
            let m1 = self.k(-1.0);
            let xx = self.ins(Inst::Mul(a[0], a[0]));
            self.ins(Inst::Div(m1, xx))
        });
        self.unary(a, l, h1, h2)
    }
    fn neg(&mut self, a: &[V]) -> Lanes {
        a.iter().map(|&x| self.ins(Inst::Neg(x))).collect()
    }
    /// `DualV::pow`: a constant exponent (no tangent) is `powf` lifted with
    /// `h1 = n·powf(a, n − 1)` and `h2 = n·(n − 1)·powf(a, n − 2)` (the constant `powf(a,
    /// n)` at `a = 0`); otherwise `exp(e · ln a)`.
    ///
    /// Every arm is computed and the selects keep one, so an arm a select discards runs on
    /// a safe operand (`1`) instead: the kept arm's values are unchanged, and the discarded
    /// ones stay finite, so the reverse pass's zero adjoint through them stays zero.
    fn pow(&mut self, a: &[V], e: &[V]) -> Lanes {
        if self.lanes() == 1 {
            return vec![self.ins(Inst::Powf(a[0], e[0]))];
        }
        let (av, n) = (a[0], e[0]);
        let z = self.k(0.0);
        let one = self.k(1.0);
        let a_zero = self.ins(Inst::Cmp(Cc::Eq, av, z));
        // `zero_tan`: every tangent `== 0.0` (so `−0.0` counts as zero); the payload is not
        // consulted. Without tangents every exponent is constant.
        let mut all_zero: Option<V> = None;
        for k in 0..self.n() {
            let z = self.k(0.0);
            let is = self.ins(Inst::Cmp(Cc::Eq, e[1 + k], z));
            all_zero = Some(match all_zero {
                None => is,
                Some(acc) => self.ins(Inst::And(acc, is)),
            });
        }
        let pv = self.ins(Inst::Powf(av, n));
        // The lifted arm, kept for `a ≠ 0`.
        let al = self.ins(Inst::Select(a_zero, one, av));
        let nm1 = self.ins(Inst::Sub(n, one));
        let pm1 = self.ins(Inst::Powf(al, nm1));
        let h1 = self.ins(Inst::Mul(n, pm1));
        let h2 = (self.layout.m > 0).then(|| {
            let one = self.k(1.0);
            let nm1 = self.ins(Inst::Sub(n, one));
            let nn = self.ins(Inst::Mul(n, nm1));
            let two = self.k(2.0);
            let nm2 = self.ins(Inst::Sub(n, two));
            let pm2 = self.ins(Inst::Powf(al, nm2));
            self.ins(Inst::Mul(nn, pm2))
        });
        let mut a_l = a.to_vec();
        a_l[0] = al;
        let lifted = self.unary(&a_l, pv, h1, h2);
        let flat = self.constant(pv);
        let const_exp = self.select(a_zero, &flat, &lifted);
        let Some(cond) = all_zero else {
            return const_exp;
        };
        // The general arm, kept when the exponent carries a tangent.
        let ones = self.constant(one);
        let a_g = self.select(cond, &ones, a);
        let l = self.ln(&a_g);
        let el = self.mul(e, &l);
        let general = self.exp(&el);
        self.select(cond, &const_exp, &general)
    }

    /// Emit one register instruction; `args` are the argument indices of `u`, `params`,
    /// `seeds` and `du`.
    pub(super) fn flat(&mut self, f: &Flat, args: [u8; 4]) {
        emit(self, f, args)
    }
    fn abs(&mut self, a: &[V]) -> Lanes {
        // `f64::abs` on one lane; `DualV::abs`'s `v ≥ 0` select on a dual.
        if self.lanes() == 1 {
            return vec![self.ins(Inst::Abs(a[0]))];
        }
        let z = self.k(0.0);
        let nonneg = self.ins(Inst::Cmp(Cc::Ge, a[0], z));
        let n = self.neg(a);
        self.select(nonneg, a, &n)
    }
}

/// A number type's arithmetic over its lanes, as scalar instructions: what [`emit`] needs to
/// write any [`Flat`] instruction. An algebra transcribes its type's own operator bodies;
/// the instruction dispatch — the guards, the `ln` floor, the constant operands — is
/// written once, here.
pub(super) trait Alg {
    fn func(&mut self) -> &mut Func;
    /// Lanes per value.
    fn width(&self) -> usize;
    fn regs(&mut self) -> &mut Vec<Lanes>;
    fn regs_ref(&self) -> &[Lanes];
    fn mul(&mut self, a: &[V], b: &[V]) -> Lanes;
    /// `a / b`, or the constant zero when `|b| < 1e-30` (`apply_binary_in`'s guard).
    fn guarded_div(&mut self, a: &[V], b: &[V]) -> Lanes;
    /// A constant right operand, as the type's `*_scalar_in`.
    fn scalar(&mut self, op: Arith, a: &[V], c: f64) -> Lanes;
    fn exp(&mut self, a: &[V]) -> Lanes;
    /// The type's own `ln`, unguarded.
    fn ln(&mut self, a: &[V]) -> Lanes;
    fn abs(&mut self, a: &[V]) -> Lanes;
    fn pow(&mut self, a: &[V], e: &[V]) -> Lanes;

    fn ins(&mut self, i: Inst) -> V {
        self.func().push(i)
    }
    fn k(&mut self, c: f64) -> V {
        self.func().k(c)
    }
    /// The constant whose value is `v` (every other lane zero): `T::from_f64`.
    fn constant(&mut self, v: V) -> Lanes {
        let mut l = vec![v];
        for _ in 1..self.width() {
            let z = self.k(0.0);
            l.push(z);
        }
        l
    }
    fn select(&mut self, c: V, a: &[V], b: &[V]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.ins(Inst::Select(c, x, y)))
            .collect()
    }
    /// Lane by lane — every type's `+`.
    fn add(&mut self, a: &[V], b: &[V]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.ins(Inst::Add(x, y)))
            .collect()
    }
    /// Lane by lane — every type's `−`.
    fn sub(&mut self, a: &[V], b: &[V]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.ins(Inst::Sub(x, y)))
            .collect()
    }
    fn load(&mut self, arg: u8, index: usize) -> Lanes {
        let lanes = self.width();
        (0..lanes)
            .map(|j| {
                self.ins(Inst::Load {
                    arg,
                    off: (index * lanes + j) as u32,
                })
            })
            .collect()
    }
    fn store(&mut self, arg: u8, index: usize, l: &[V]) {
        let lanes = self.width();
        for (j, &v) in l.iter().enumerate() {
            self.ins(Inst::Store {
                arg,
                off: (index * lanes + j) as u32,
                v,
            });
        }
    }
    fn reg(&self, r: u16) -> Lanes {
        self.regs_ref()[r as usize].clone()
    }
    fn set(&mut self, d: u16, l: Lanes) {
        self.regs()[d as usize] = l;
    }
}

impl Alg for Lower<'_> {
    fn func(&mut self) -> &mut Func {
        self.f
    }
    fn width(&self) -> usize {
        self.layout.lanes()
    }
    fn regs(&mut self) -> &mut Vec<Lanes> {
        &mut self.regs
    }
    fn regs_ref(&self) -> &[Lanes] {
        &self.regs
    }
    fn mul(&mut self, a: &[V], b: &[V]) -> Lanes {
        Lower::mul(self, a, b)
    }
    fn guarded_div(&mut self, a: &[V], b: &[V]) -> Lanes {
        Lower::guarded_div(self, a, b)
    }
    fn scalar(&mut self, op: Arith, a: &[V], c: f64) -> Lanes {
        Lower::scalar(self, op, a, c)
    }
    fn exp(&mut self, a: &[V]) -> Lanes {
        Lower::exp(self, a)
    }
    fn ln(&mut self, a: &[V]) -> Lanes {
        Lower::ln(self, a)
    }
    fn abs(&mut self, a: &[V]) -> Lanes {
        Lower::abs(self, a)
    }
    fn pow(&mut self, a: &[V], e: &[V]) -> Lanes {
        Lower::pow(self, a, e)
    }
}

/// Emit one register instruction over any [`Alg`]; `args` are the argument indices of `u`,
/// `params`, `seeds` and `du`.
pub(super) fn emit<A: Alg>(al: &mut A, f: &Flat, args: [u8; 4]) {
    let [u, params, seeds, du] = args;
    match *f {
        Flat::LoadState { d, i } => {
            let l = al.load(u, i as usize);
            al.set(d, l);
        }
        Flat::LoadParam { d, slot } => {
            let l = al.load(params, slot as usize);
            al.set(d, l);
        }
        Flat::LoadSeed { d, k } => {
            let l = al.load(seeds, k as usize);
            al.set(d, l);
        }
        Flat::LoadK { d, c } => {
            let v = al.k(c);
            let l = al.constant(v);
            al.set(d, l);
        }
        Flat::Add { d, a, b } => {
            let l = al.add(&al.reg(a), &al.reg(b));
            al.set(d, l);
        }
        Flat::Sub { d, a, b } => {
            let l = al.sub(&al.reg(a), &al.reg(b));
            al.set(d, l);
        }
        // `rsub_in`: `self = rhs − self`.
        Flat::RSub { d, a, b } => {
            let l = al.sub(&al.reg(b), &al.reg(a));
            al.set(d, l);
        }
        Flat::Mul { d, a, b } => {
            let l = al.mul(&al.reg(a), &al.reg(b));
            al.set(d, l);
        }
        Flat::Div { d, a, b } => {
            let l = al.guarded_div(&al.reg(a), &al.reg(b));
            al.set(d, l);
        }
        Flat::AddK { d, a, c }
        | Flat::SubK { d, a, c }
        | Flat::RSubK { d, a, c }
        | Flat::MulK { d, a, c }
        | Flat::DivK { d, a, c } => {
            let op = match *f {
                Flat::AddK { .. } => Arith::Add,
                Flat::SubK { .. } => Arith::Sub,
                Flat::RSubK { .. } => Arith::RSub,
                Flat::MulK { .. } => Arith::Mul,
                _ => Arith::Div,
            };
            // `apply_scalar_in`'s guard, decided here on the literal: a division by
            // `|c| < 1e-30` is the constant zero.
            let l = if op == Arith::Div && c.abs() < 1e-30 {
                let z = al.k(0.0);
                al.constant(z)
            } else {
                al.scalar(op, &al.reg(a), c)
            };
            al.set(d, l);
        }
        Flat::Pow { d, a, e } => {
            let l = al.pow(&al.reg(a), &al.reg(e));
            al.set(d, l);
        }
        Flat::Unary { d, op, a } => {
            let x = al.reg(a);
            let l = match op {
                Unary::Exp => al.exp(&x),
                Unary::Ln => {
                    // `guard_floor(1e-30)`: `f64::max` (a NaN floors) on one lane,
                    // `DualV`'s `!(v ≥ lo)` → `constant(lo)` on a dual — the same test.
                    let lo = al.k(1e-30);
                    let keep = al.ins(Inst::Cmp(Cc::Ge, x[0], lo));
                    let floor = al.constant(lo);
                    let g = al.select(keep, &x, &floor);
                    al.ln(&g)
                }
                Unary::Abs => al.abs(&x),
                Unary::Floor => {
                    let v = al.ins(Inst::Floor(x[0]));
                    al.constant(v)
                }
                Unary::Ceil => {
                    let v = al.ins(Inst::Ceil(x[0]));
                    al.constant(v)
                }
                Unary::Round => {
                    let v = al.ins(Inst::Call1(Shim1::Round, x[0]));
                    al.constant(v)
                }
            };
            al.set(d, l);
        }
        Flat::Deriv { state, a } => {
            let x = al.reg(a);
            al.store(du, state as usize, &x);
        }
    }
}

/// The whole program over `layout`: `fn(u, params, seeds, du)`.
pub(super) fn forward(code: &[Flat], n_regs: usize, layout: Layout) -> Func {
    let mut f = Func::new(4);
    let mut l = Lower::new(&mut f, layout, n_regs);
    for i in code {
        l.flat(i, [0, 1, 2, 3]);
    }
    f
}

/// The reverse pass of a forward [`Func`] of `n` arguments: a function of `2n` arguments
/// whose argument `n + a` is the adjoint buffer of argument `a`, laid out like it.
///
/// It runs the forward instructions, seeds each stored value's adjoint from its output's
/// adjoint buffer (the last store to a location is the one that counts), applies every
/// instruction's local derivative in reverse order, and adds each loaded location's
/// adjoint into its input's adjoint buffer. Conditions carry no derivative; a select
/// passes the adjoint to the operand it chose.
pub(super) fn reverse(fwd: &Func) -> Func {
    let n = fwd.n_args;
    let mut b = Func::new(2 * n);
    // The forward instructions, unchanged (a store has no value and is dropped).
    let mut map = vec![V::MAX; fwd.insts.len()];
    let mut last_store = std::collections::HashMap::new();
    for (i, inst) in fwd.insts.iter().enumerate() {
        let r = |x: V| map[x as usize];
        let m = match *inst {
            Inst::Store { arg, off, v } => {
                last_store.insert((arg, off), v);
                continue;
            }
            Inst::Load { .. } | Inst::Const(_) => *inst,
            Inst::Add(x, y) => Inst::Add(r(x), r(y)),
            Inst::Sub(x, y) => Inst::Sub(r(x), r(y)),
            Inst::Mul(x, y) => Inst::Mul(r(x), r(y)),
            Inst::Div(x, y) => Inst::Div(r(x), r(y)),
            Inst::Neg(x) => Inst::Neg(r(x)),
            Inst::Abs(x) => Inst::Abs(r(x)),
            Inst::Floor(x) => Inst::Floor(r(x)),
            Inst::Ceil(x) => Inst::Ceil(r(x)),
            Inst::Call1(s, x) => Inst::Call1(s, r(x)),
            Inst::Powf(x, y) => Inst::Powf(r(x), r(y)),
            Inst::Cmp(c, x, y) => Inst::Cmp(c, r(x), r(y)),
            Inst::And(x, y) => Inst::And(r(x), r(y)),
            Inst::Select(c, x, y) => Inst::Select(r(c), r(x), r(y)),
        };
        map[i] = b.push(m);
    }
    // Adjoints, indexed by forward instruction.
    let mut adj: Vec<Option<V>> = vec![None; fwd.insts.len()];
    fn acc(b: &mut Func, adj: &mut [Option<V>], x: V, d: V, negate: bool) {
        let slot = &mut adj[x as usize];
        *slot = Some(match (*slot, negate) {
            (None, false) => d,
            (None, true) => b.push(Inst::Neg(d)),
            (Some(s), false) => b.push(Inst::Add(s, d)),
            (Some(s), true) => b.push(Inst::Sub(s, d)),
        });
    }
    let mut stores: Vec<_> = last_store.into_iter().collect();
    stores.sort_unstable_by_key(|&((a, o), _)| (a, o));
    for ((arg, off), v) in stores {
        let seed = b.push(Inst::Load { arg: n + arg, off });
        acc(&mut b, &mut adj, v, seed, false);
    }
    let mut loads: Vec<((u8, u32), V)> = Vec::new();
    for i in (0..fwd.insts.len()).rev() {
        let Some(g) = adj[i] else { continue };
        let fw = |x: V| map[x as usize];
        match fwd.insts[i] {
            Inst::Load { arg, off } => loads.push(((arg, off), g)),
            Inst::Const(_) | Inst::Store { .. } | Inst::Cmp(..) | Inst::And(..) => {}
            Inst::Floor(_) | Inst::Ceil(_) | Inst::Call1(Shim1::Round, _) => {}
            Inst::Add(x, y) => {
                acc(&mut b, &mut adj, x, g, false);
                acc(&mut b, &mut adj, y, g, false);
            }
            Inst::Sub(x, y) => {
                acc(&mut b, &mut adj, x, g, false);
                acc(&mut b, &mut adj, y, g, true);
            }
            Inst::Mul(x, y) => {
                let dx = b.push(Inst::Mul(g, fw(y)));
                acc(&mut b, &mut adj, x, dx, false);
                let dy = b.push(Inst::Mul(g, fw(x)));
                acc(&mut b, &mut adj, y, dy, false);
            }
            Inst::Div(x, y) => {
                // `q = x/y`: `dx = g/y`, `dy = −(g/y)·q`.
                let t = b.push(Inst::Div(g, fw(y)));
                acc(&mut b, &mut adj, x, t, false);
                let dy = b.push(Inst::Mul(t, map[i]));
                acc(&mut b, &mut adj, y, dy, true);
            }
            Inst::Neg(x) => acc(&mut b, &mut adj, x, g, true),
            Inst::Abs(x) => {
                // `|x|′` as the duals take it: `+1` for `x ≥ 0`.
                let z = b.k(0.0);
                let nonneg = b.push(Inst::Cmp(Cc::Ge, fw(x), z));
                let ng = b.push(Inst::Neg(g));
                let d = b.push(Inst::Select(nonneg, g, ng));
                acc(&mut b, &mut adj, x, d, false);
            }
            Inst::Call1(Shim1::Exp, x) => {
                let d = b.push(Inst::Mul(g, map[i]));
                acc(&mut b, &mut adj, x, d, false);
            }
            Inst::Call1(Shim1::Ln, x) => {
                let d = b.push(Inst::Div(g, fw(x)));
                acc(&mut b, &mut adj, x, d, false);
            }
            Inst::Powf(x, y) => {
                // `r = xʸ`: `dx = g·y·x^(y−1)`; `dy = g·r·ln x` for `x > 0`, and `0` where
                // the logarithm does not exist (`0^y` is flat in `y` for `y > 0`).
                let one = b.k(1.0);
                let ym1 = b.push(Inst::Sub(fw(y), one));
                let p = b.push(Inst::Powf(fw(x), ym1));
                let yp = b.push(Inst::Mul(fw(y), p));
                let dx = b.push(Inst::Mul(g, yp));
                acc(&mut b, &mut adj, x, dx, false);
                let lx = b.push(Inst::Call1(Shim1::Ln, fw(x)));
                let rl = b.push(Inst::Mul(map[i], lx));
                let dy = b.push(Inst::Mul(g, rl));
                let z = b.k(0.0);
                let pos = b.push(Inst::Cmp(Cc::Gt, fw(x), z));
                let dy = b.push(Inst::Select(pos, dy, z));
                acc(&mut b, &mut adj, y, dy, false);
            }
            Inst::Select(c, x, y) => {
                let z = b.k(0.0);
                let dx = b.push(Inst::Select(fw(c), g, z));
                acc(&mut b, &mut adj, x, dx, false);
                let dy = b.push(Inst::Select(fw(c), z, g));
                acc(&mut b, &mut adj, y, dy, false);
            }
        }
    }
    // One read-modify-write per input location, however many loads read it.
    let mut by_loc: std::collections::BTreeMap<(u8, u32), V> = Default::default();
    for (loc, g) in loads {
        let e = by_loc.entry(loc).or_insert(V::MAX);
        *e = if *e == V::MAX {
            g
        } else {
            b.push(Inst::Add(*e, g))
        };
    }
    for ((arg, off), g) in by_loc {
        let old = b.push(Inst::Load { arg: n + arg, off });
        let s = b.push(Inst::Add(old, g));
        b.push(Inst::Store {
            arg: n + arg,
            off,
            v: s,
        });
    }
    b
}

/// Run a [`Func`] in Rust: the reference semantics the native lowering must reproduce, and
/// the tests' oracle for [`reverse`].
#[cfg(test)]
pub(super) fn interpret(f: &Func, args: &mut [&mut [f64]]) {
    assert_eq!(args.len(), f.n_args as usize);
    let mut x = vec![0.0f64; f.insts.len()];
    let mut c = vec![false; f.insts.len()];
    for (i, inst) in f.insts.iter().enumerate() {
        let r = |v: V| x[v as usize];
        x[i] = match *inst {
            Inst::Load { arg, off } => args[arg as usize][off as usize],
            Inst::Const(bits) => f64::from_bits(bits),
            Inst::Add(a, b) => r(a) + r(b),
            Inst::Sub(a, b) => r(a) - r(b),
            Inst::Mul(a, b) => r(a) * r(b),
            Inst::Div(a, b) => r(a) / r(b),
            Inst::Neg(a) => -r(a),
            Inst::Abs(a) => r(a).abs(),
            Inst::Floor(a) => r(a).floor(),
            Inst::Ceil(a) => r(a).ceil(),
            Inst::Call1(Shim1::Exp, a) => r(a).exp(),
            Inst::Call1(Shim1::Ln, a) => r(a).ln(),
            Inst::Call1(Shim1::Round, a) => r(a).round(),
            Inst::Powf(a, b) => r(a).powf(r(b)),
            Inst::Cmp(cc, a, b) => {
                c[i] = match cc {
                    Cc::Lt => r(a) < r(b),
                    Cc::Gt => r(a) > r(b),
                    Cc::Ge => r(a) >= r(b),
                    Cc::Eq => r(a) == r(b),
                };
                0.0
            }
            Inst::And(a, b) => {
                c[i] = c[a as usize] && c[b as usize];
                0.0
            }
            Inst::Select(k, a, b) => {
                if c[k as usize] {
                    r(a)
                } else {
                    r(b)
                }
            }
            Inst::Store { arg, off, v } => {
                args[arg as usize][off as usize] = r(v);
                0.0
            }
        };
    }
}

/// Forward-mode tangents of a [`Func`] in Rust: `dargs[a][off]` is the tangent of input
/// `args[a][off]`, and each store writes its value's tangent into `dargs`. The reverse
/// pass's oracle — an independent derivation of the same scalar code (no finite
/// differences): every instruction's derivative rule applied forwards.
#[cfg(test)]
pub(super) fn interpret_tangent(f: &Func, args: &[&[f64]], dargs: &mut [Vec<f64>]) {
    let mut x = vec![0.0f64; f.insts.len()];
    let mut dx = vec![0.0f64; f.insts.len()];
    let mut c = vec![false; f.insts.len()];
    for (i, inst) in f.insts.iter().enumerate() {
        let r = |v: V| x[v as usize];
        let d = |v: V| dx[v as usize];
        let (val, tan) = match *inst {
            Inst::Load { arg, off } => (
                args[arg as usize][off as usize],
                dargs[arg as usize][off as usize],
            ),
            Inst::Const(bits) => (f64::from_bits(bits), 0.0),
            Inst::Add(a, b) => (r(a) + r(b), d(a) + d(b)),
            Inst::Sub(a, b) => (r(a) - r(b), d(a) - d(b)),
            Inst::Mul(a, b) => (r(a) * r(b), d(a) * r(b) + r(a) * d(b)),
            Inst::Div(a, b) => {
                let q = r(a) / r(b);
                (q, (d(a) - q * d(b)) / r(b))
            }
            Inst::Neg(a) => (-r(a), -d(a)),
            Inst::Abs(a) => (r(a).abs(), if r(a) >= 0.0 { d(a) } else { -d(a) }),
            Inst::Floor(a) => (r(a).floor(), 0.0),
            Inst::Ceil(a) => (r(a).ceil(), 0.0),
            Inst::Call1(Shim1::Exp, a) => {
                let e = r(a).exp();
                (e, e * d(a))
            }
            Inst::Call1(Shim1::Ln, a) => (r(a).ln(), d(a) / r(a)),
            Inst::Call1(Shim1::Round, a) => (r(a).round(), 0.0),
            Inst::Powf(a, b) => {
                let p = r(a).powf(r(b));
                let da = r(b) * r(a).powf(r(b) - 1.0) * d(a);
                let db = if r(a) > 0.0 {
                    p * r(a).ln() * d(b)
                } else {
                    0.0
                };
                (p, da + db)
            }
            Inst::Cmp(cc, a, b) => {
                c[i] = match cc {
                    Cc::Lt => r(a) < r(b),
                    Cc::Gt => r(a) > r(b),
                    Cc::Ge => r(a) >= r(b),
                    Cc::Eq => r(a) == r(b),
                };
                (0.0, 0.0)
            }
            Inst::And(a, b) => {
                c[i] = c[a as usize] && c[b as usize];
                (0.0, 0.0)
            }
            Inst::Select(k, a, b) => {
                if c[k as usize] {
                    (r(a), d(a))
                } else {
                    (r(b), d(b))
                }
            }
            Inst::Store { arg, off, v } => {
                dargs[arg as usize][off as usize] = d(v);
                (0.0, 0.0)
            }
        };
        x[i] = val;
        dx[i] = tan;
    }
}
