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
//!
//! **An inline `if` is computed on both arms and selected.** The register code branches
//! ([`Flat::Branch`]); here both arms are straight-line code and each value the arms write is
//! a [`Inst::Select`] on the condition, so every lane is the taken arm's, bit for bit. The
//! dead arm may compute a `NaN` or an infinity (a `powf` of a negative base, an `exp`
//! overflow) that the select discards — but a reverse pass that merely sent it a zero adjoint
//! would still multiply that zero by its values (`0·∞ = NaN`) into the inputs' adjoints. So
//! every instruction an arm emits carries the condition under which it is live
//! ([`Func::live`]), and the reverse pass selects each of its local derivatives against that
//! condition: a dead arm contributes an exact zero, whatever it computed.
use super::{Arith, Flat, Unary};

/// An instruction's index in its [`Func`].
pub(super) type V = u32;

/// A call the native code makes into Rust, so the value is the interpreter's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shim1 {
    Exp,
    Ln,
    Round,
    /// IEEE square root (exact): lowered to the native instruction, not a call.
    Sqrt,
    Cos,
    Sin,
    Acos,
    /// `stats::special`'s `ln_gamma` and its derivatives.
    LnGamma,
    Digamma,
    Trigamma,
    Tetragamma,
    Pentagamma,
    /// `−sin` (`sin`'s second derivative).
    NegSin,
    /// The `k`-th derivative of `exprel(z) = (e^z − 1)/z`, `NaN` past the fourth.
    ExpRel(u8),
    /// The `k`-th derivative of a function given by its jet, `NaN` past the fourth.
    Jet(crate::sens::num::JetFn, u8),
    /// `NaN`: a derivative order no layout reaches, so a misuse shows as non-finite.
    Nan,
}

impl Shim1 {
    /// The polygamma `ψ⁽ᵏ⁻¹⁾` as the derivative of order `k` of `ln Γ` (`k ≥ 1`); `NaN` past
    /// tetragamma (no layout differentiates further).
    pub(super) fn polygamma(k: u8) -> Shim1 {
        match k {
            0 => Shim1::LnGamma,
            1 => Shim1::Digamma,
            2 => Shim1::Trigamma,
            3 => Shim1::Tetragamma,
            4 => Shim1::Pentagamma,
            _ => Shim1::Nan,
        }
    }
    /// The function a native call (or the interpreter) makes for this shim.
    pub(super) fn f(self) -> NativeUnary {
        extern "C" fn exp(x: f64) -> f64 {
            x.exp()
        }
        extern "C" fn ln(x: f64) -> f64 {
            x.ln()
        }
        extern "C" fn round(x: f64) -> f64 {
            x.round()
        }
        extern "C" fn sqrt(x: f64) -> f64 {
            x.sqrt()
        }
        extern "C" fn cos(x: f64) -> f64 {
            x.cos()
        }
        extern "C" fn sin(x: f64) -> f64 {
            x.sin()
        }
        extern "C" fn acos(x: f64) -> f64 {
            x.acos()
        }
        extern "C" fn ln_gamma(x: f64) -> f64 {
            crate::stats::special::ln_gamma(x)
        }
        extern "C" fn digamma(x: f64) -> f64 {
            crate::stats::special::digamma(x)
        }
        extern "C" fn trigamma(x: f64) -> f64 {
            crate::stats::special::trigamma(x)
        }
        extern "C" fn tetragamma(x: f64) -> f64 {
            crate::stats::special::tetragamma(x)
        }
        extern "C" fn pentagamma(x: f64) -> f64 {
            crate::stats::special::pentagamma(x)
        }
        extern "C" fn exprel<const K: u8>(x: f64) -> f64 {
            if K as usize > 4 {
                f64::NAN
            } else {
                crate::ode::linear::exprel_all(x)[K as usize]
            }
        }
        extern "C" fn neg_sin(x: f64) -> f64 {
            -x.sin()
        }
        extern "C" fn nan(_: f64) -> f64 {
            f64::NAN
        }
        match self {
            Shim1::Exp => exp,
            Shim1::Ln => ln,
            Shim1::Round => round,
            Shim1::Sqrt => sqrt,
            Shim1::Cos => cos,
            Shim1::Sin => sin,
            Shim1::Acos => acos,
            Shim1::LnGamma => ln_gamma,
            Shim1::Digamma => digamma,
            Shim1::Trigamma => trigamma,
            Shim1::Tetragamma => tetragamma,
            Shim1::Pentagamma => pentagamma,
            Shim1::NegSin => neg_sin,
            Shim1::ExpRel(0) => exprel::<0>,
            Shim1::ExpRel(1) => exprel::<1>,
            Shim1::ExpRel(2) => exprel::<2>,
            Shim1::ExpRel(3) => exprel::<3>,
            Shim1::ExpRel(4) => exprel::<4>,
            Shim1::ExpRel(_) | Shim1::Nan => nan,
            Shim1::Jet(f, k) => jet_shim(f, k),
        }
    }
}

/// A native unary function.
pub(super) type NativeUnary = extern "C" fn(f64) -> f64;

/// [`Shim1::Jet`]'s function: entry `k` of `f`'s jet (`NaN` past the fourth).
fn jet_shim(f: crate::sens::num::JetFn, k: u8) -> NativeUnary {
    use crate::sens::num::JetFn;
    extern "C" fn j<const F: u8, const K: u8>(x: f64) -> f64 {
        if K as usize > 4 {
            f64::NAN
        } else {
            JetFn::from_index(F).jet(x)[K as usize]
        }
    }
    macro_rules! entries {
        ($f:expr, $k:expr, $($i:literal),*) => {
            match ($f, $k) {
                $(
                    ($i, 0) => j::<$i, 0> as NativeUnary,
                    ($i, 1) => j::<$i, 1>,
                    ($i, 2) => j::<$i, 2>,
                    ($i, 3) => j::<$i, 3>,
                    ($i, 4) => j::<$i, 4>,
                )*
                _ => j::<0, 5>,
            }
        };
    }
    entries!(f.index(), k, 0, 1, 2, 3, 4, 5, 6, 7)
}

/// The host calls' runtime: the queued arguments and the last call's outputs, per thread. The
/// interpreters and the native code go through the same three functions.
pub(super) mod host {
    use std::cell::RefCell;
    thread_local! {
        static ST: RefCell<(Vec<f64>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    }
    pub(in crate::parser) fn arg(v: f64) {
        ST.with(|s| s.borrow_mut().0.push(v));
    }
    pub(in crate::parser) fn call(f: crate::sens::num::HostFn) {
        ST.with(|s| {
            let (args, out) = &mut *s.borrow_mut();
            f.run(args, out);
            args.clear();
        });
    }
    pub(in crate::parser) fn out(k: u32) -> f64 {
        ST.with(|s| s.borrow().1.get(k as usize).copied().unwrap_or(f64::NAN))
    }
    pub(in crate::parser) extern "C" fn arg_native(v: f64) -> f64 {
        arg(v);
        0.0
    }
    pub(in crate::parser) extern "C" fn call_native(f: f64) -> f64 {
        call(crate::sens::num::HostFn::from_index(f as u16));
        0.0
    }
    pub(in crate::parser) extern "C" fn out_native(k: f64) -> f64 {
        out(k as u32)
    }
}

/// The native function of an [`Inst::Gamma`].
pub(super) fn gamma_shim(upper: bool, i: u8, j: u8) -> extern "C" fn(f64, f64) -> f64 {
    extern "C" fn sh<const U: bool, const I: u8, const J: u8>(a: f64, x: f64) -> f64 {
        crate::stats::special::gamma_pq_partial(a, x, U, I, J)
    }
    macro_rules! table {
        ($($i:literal $j:literal),*) => {
            match (upper, i, j) {
                $((false, $i, $j) => sh::<false, $i, $j>, (true, $i, $j) => sh::<true, $i, $j>,)*
                // Past third order: `NaN` (`gamma_pq_partial`).
                _ => sh::<false, 4, 0>,
            }
        };
    }
    table!(0 0, 1 0, 0 1, 2 0, 1 1, 0 2, 3 0, 2 1, 1 2, 0 3)
}

/// A comparison, producing a condition for [`Inst::Select`]: Rust's (and IEEE's) ordered
/// comparisons, and `!=`, which holds when either side is `NaN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Cc {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Cc {
    /// The same comparison for the branch API ([`PkNum::br_cmp`](crate::sens::num::PkNum)).
    #[inline(always)]
    pub(super) fn branch(self) -> crate::sens::num::BranchCmp {
        use crate::sens::num::BranchCmp;
        match self {
            Cc::Lt => BranchCmp::Lt,
            Cc::Le => BranchCmp::Le,
            Cc::Gt => BranchCmp::Gt,
            Cc::Ge => BranchCmp::Ge,
            Cc::Eq => BranchCmp::Eq,
            Cc::Ne => BranchCmp::Ne,
        }
    }
    /// `a cc b`, as the interpreter's `Cmp*` ops decide it.
    #[inline(always)]
    pub(super) fn holds(self, a: f64, b: f64) -> bool {
        match self {
            Cc::Lt => a < b,
            Cc::Le => a <= b,
            Cc::Gt => a > b,
            Cc::Ge => a >= b,
            Cc::Eq => a == b,
            Cc::Ne => a != b,
        }
    }
}

/// One scalar instruction. `f64`-valued unless it is a condition ([`Inst::Cmp`],
/// [`Inst::And`], [`Inst::Or`]).
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
    /// `∂^(i+j) P(a, x) / ∂aⁱ ∂xʲ` (of `Q` when `upper`): the incomplete gamma primitive and
    /// its local derivatives (`stats::special::gamma_pq_partial`).
    Gamma {
        upper: bool,
        i: u8,
        j: u8,
        a: V,
        x: V,
    },
    Cmp(Cc, V, V),
    And(V, V),
    Or(V, V),
    /// `cond ? a : b`.
    Select(V, V, V),
    /// Its operand's value, with no derivative: the identity forward, no adjoint in
    /// [`reverse`] (`PkNum::value_only`). Not the operand itself: a reverse pass through
    /// `d − value_only(d)` would otherwise cancel `d`'s own derivative.
    StopGrad(V),
    /// A host call ([`host`]): queue an argument, run, read output `k`.
    HostArg(V),
    Host(crate::sens::num::HostFn),
    HostOut(u32),
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
    /// Per instruction: the condition under which its value is used — the arm of an `if` it
    /// was emitted in (the conjunction, for nested ones), or `None` outside every arm. What
    /// [`reverse`] masks a dead arm's derivatives with.
    pub(super) live: Vec<Option<V>>,
    /// The condition the next instruction is emitted under.
    cur: Option<V>,
}

impl Func {
    pub(super) fn new(n_args: u8) -> Self {
        Func {
            insts: Vec::new(),
            n_args,
            live: Vec::new(),
            cur: None,
        }
    }
    fn push(&mut self, i: Inst) -> V {
        self.insts.push(i);
        self.live.push(self.cur);
        (self.insts.len() - 1) as V
    }
    fn k(&mut self, c: f64) -> V {
        self.push(Inst::Const(c.to_bits()))
    }
}

pub(crate) use crate::sens::num::vm::Layout;

/// The payload entries of row `k`: `(l, t)` with `h_t = ∂²/∂η_k∂η_l` (`vtape::hrange`/
/// `hidx`).
pub(super) fn pairs(layout: &Layout, k: usize) -> Vec<(usize, usize)> {
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

/// An `if` [`Lower`] is inside: its condition, and each arm's results so far.
struct Arm {
    /// The condition's value is `0`: the else arm is taken.
    is_else: V,
    /// The conditions the else arm and the code around the `if` are live under.
    else_live: Option<V>,
    outer: Option<V>,
    /// Where the `if` ends, known once the then arm's [`Flat::Jump`] is reached.
    end: Option<usize>,
    then_vals: Vec<(u32, Lanes)>,
    else_vals: Vec<(u32, Lanes)>,
}

/// Emits [`Flat`] instructions over a [`Layout`]'s lanes into a [`Func`].
pub(super) struct Lower<'a> {
    pub(super) f: &'a mut Func,
    pub(super) layout: Layout,
    regs: Vec<Lanes>,
    arms: Vec<Arm>,
}

impl<'a> Lower<'a> {
    pub(super) fn new(f: &'a mut Func, layout: Layout, n_regs: usize) -> Self {
        Lower {
            f,
            layout,
            regs: vec![Vec::new(); n_regs],
            arms: Vec::new(),
        }
    }
    /// Close every `if` that ends at instruction `at`: each value its arms write becomes the
    /// select of the else arm's lanes on `is_else` over the then arm's. A caller that walks
    /// the program calls it at every instruction index, whether or not it emits that
    /// instruction, and once at the end.
    pub(super) fn reach(&mut self, at: usize) {
        while self.arms.last().is_some_and(|a| a.end == Some(at)) {
            let arm = self.arms.pop().expect("checked");
            self.f.cur = arm.outer;
            for (d, t) in &arm.then_vals {
                let e = &arm
                    .else_vals
                    .iter()
                    .find(|(x, _)| x == d)
                    .expect("both arms write an `if`'s value")
                    .1;
                let l = Alg::select(self, arm.is_else, e, t);
                self.set(*d, l);
            }
        }
    }
    /// The conjunction of the current condition with `c`.
    fn within(&mut self, c: V) -> Option<V> {
        Some(match self.f.cur {
            None => c,
            Some(o) => self.f.push(Inst::And(o, c)),
        })
    }
    /// An `if`'s control instruction: open it, switch to its else arm, or record an arm's
    /// result.
    fn control(&mut self, f: &Flat) {
        match *f {
            Flat::Branch { c, .. } => {
                let c0 = self.regs[c as usize][0];
                let z = self.k(0.0);
                let is_else = self.ins(Inst::Cmp(Cc::Eq, c0, z));
                let is_then = self.ins(Inst::Cmp(Cc::Ne, c0, z));
                let outer = self.f.cur;
                let then_live = self.within(is_then);
                let else_live = self.within(is_else);
                self.arms.push(Arm {
                    is_else,
                    else_live,
                    outer,
                    end: None,
                    then_vals: Vec::new(),
                    else_vals: Vec::new(),
                });
                self.f.cur = then_live;
            }
            Flat::Jump { to } => {
                let arm = self.arms.last_mut().expect("a `Jump` closes a then arm");
                arm.end = Some(to as usize);
                self.f.cur = arm.else_live;
            }
            Flat::Move { d, a } => {
                let l = self.regs[a as usize].clone();
                let arm = self.arms.last_mut().expect("a `Move` ends an arm");
                if arm.end.is_none() {
                    arm.then_vals.push((d, l));
                } else {
                    arm.else_vals.push((d, l));
                }
            }
            _ => unreachable!("not a control instruction"),
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
    pub(super) fn reg(&self, r: u32) -> Lanes {
        self.regs[r as usize].clone()
    }
    pub(super) fn set(&mut self, d: u32, l: Lanes) {
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
    /// `DualV::sqrt`, unguarded: `h1 = 0.5/s`, `h2 = −0.25/(s·s·s)`.
    fn sqrt(&mut self, a: &[V]) -> Lanes {
        let s = self.ins(Inst::Call1(Shim1::Sqrt, a[0]));
        if self.lanes() == 1 {
            return vec![s];
        }
        let half = self.k(0.5);
        let h1 = self.ins(Inst::Div(half, s));
        let h2 = (self.layout.m > 0).then(|| {
            let q = self.k(-0.25);
            let ss = self.ins(Inst::Mul(s, s));
            let sss = self.ins(Inst::Mul(ss, s));
            self.ins(Inst::Div(q, sss))
        });
        self.unary(a, s, h1, h2)
    }
    /// `DualV::inv_logit`: `u = 1/(1+e^{−x})`, `h1 = u(1−u)`, `h2 = h1(1−2u)`.
    fn inv_logit(&mut self, a: &[V]) -> Lanes {
        let one = self.k(1.0);
        let nx = self.ins(Inst::Neg(a[0]));
        let e = self.ins(Inst::Call1(Shim1::Exp, nx));
        let den = self.ins(Inst::Add(one, e));
        let u = self.ins(Inst::Div(one, den));
        if self.lanes() == 1 {
            return vec![u];
        }
        let omu = self.ins(Inst::Sub(one, u));
        let d1 = self.ins(Inst::Mul(u, omu));
        let h2 = (self.layout.m > 0).then(|| {
            let two = self.k(2.0);
            let tu = self.ins(Inst::Mul(two, u));
            let t = self.ins(Inst::Sub(one, tu));
            self.ins(Inst::Mul(d1, t))
        });
        self.unary(a, u, d1, h2)
    }
    /// `DualV::logit`: `ln(x/(1−x))`, `h1 = 1/(x(1−x))`, `h2 = −(1−2x)·h1·h1`.
    fn logit(&mut self, a: &[V]) -> Lanes {
        let one = self.k(1.0);
        let x = a[0];
        let omx = self.ins(Inst::Sub(one, x));
        let q = self.ins(Inst::Div(x, omx));
        let v = self.ins(Inst::Call1(Shim1::Ln, q));
        if self.lanes() == 1 {
            return vec![v];
        }
        let xo = self.ins(Inst::Mul(x, omx));
        let d1 = self.ins(Inst::Div(one, xo));
        let h2 = (self.layout.m > 0).then(|| {
            let two = self.k(2.0);
            let tx = self.ins(Inst::Mul(two, x));
            let t = self.ins(Inst::Sub(one, tx));
            let nt = self.ins(Inst::Neg(t));
            let p = self.ins(Inst::Mul(nt, d1));
            self.ins(Inst::Mul(p, d1))
        });
        self.unary(a, v, d1, h2)
    }
    /// `DualV::cos`: `h1 = −sin x`, `h2 = −cos x`.
    fn cos(&mut self, a: &[V]) -> Lanes {
        let c = self.ins(Inst::Call1(Shim1::Cos, a[0]));
        if self.lanes() == 1 {
            return vec![c];
        }
        let s = self.ins(Inst::Call1(Shim1::Sin, a[0]));
        let h1 = self.ins(Inst::Neg(s));
        let h2 = (self.layout.m > 0).then(|| self.ins(Inst::Neg(c)));
        self.unary(a, c, h1, h2)
    }
    /// `DualV::acos`: `s = 1 − x·x`, `h1 = −1/√s`, `h2 = −x/(s·√s)`.
    fn acos(&mut self, a: &[V]) -> Lanes {
        let v = self.ins(Inst::Call1(Shim1::Acos, a[0]));
        if self.lanes() == 1 {
            return vec![v];
        }
        let one = self.k(1.0);
        let xx = self.ins(Inst::Mul(a[0], a[0]));
        let s = self.ins(Inst::Sub(one, xx));
        let rs = self.ins(Inst::Call1(Shim1::Sqrt, s));
        let m1 = self.k(-1.0);
        let h1 = self.ins(Inst::Div(m1, rs));
        let h2 = (self.layout.m > 0).then(|| {
            let nx = self.ins(Inst::Neg(a[0]));
            let srs = self.ins(Inst::Mul(s, rs));
            self.ins(Inst::Div(nx, srs))
        });
        self.unary(a, v, h1, h2)
    }
    /// `DualV::ln_gamma`: `h1 = ψ(x)`, `h2 = ψ'(x)`.
    fn ln_gamma(&mut self, a: &[V]) -> Lanes {
        let v = self.ins(Inst::Call1(Shim1::LnGamma, a[0]));
        if self.lanes() == 1 {
            return vec![v];
        }
        let h1 = self.ins(Inst::Call1(Shim1::Digamma, a[0]));
        let h2 = (self.layout.m > 0).then(|| self.ins(Inst::Call1(Shim1::Trigamma, a[0])));
        self.unary(a, v, h1, h2)
    }
    /// A primitive `f(a, x)` from its partials `(i, j) ↦ ∂^(i+j)f`, second order first (the
    /// shim keeps the last point's partials to the highest order asked):
    /// `g_k = f_a·a_k + f_x·x_k`, `h_t = f_a·a_h + f_x·x_h + f_aa·a_k·a_l
    /// + f_ax·(a_k·x_l + a_l·x_k) + f_xx·x_k·x_l`.
    fn binary_prim(&mut self, a: &[V], x: &[V], part: impl Fn(u8, u8) -> Inst) -> Lanes {
        let n = self.n();
        let mut out = vec![0; self.lanes()];
        let second =
            (self.layout.m > 0).then(|| [part(2, 0), part(1, 1), part(0, 2)].map(|p| self.ins(p)));
        out[0] = self.ins(part(0, 0));
        if self.lanes() == 1 {
            return out;
        }
        let fa = self.ins(part(1, 0));
        let fx = self.ins(part(0, 1));
        for k in 0..n {
            let s = self.ins(Inst::Mul(fa, a[1 + k]));
            let t = self.ins(Inst::Mul(fx, x[1 + k]));
            out[1 + k] = self.ins(Inst::Add(s, t));
            for (l, t) in pairs(&self.layout, k) {
                let h = 1 + n + t;
                let [faa, fax, fxx] = second.expect("a payload has its second derivatives");
                let mut acc = self.ins(Inst::Mul(fa, a[h]));
                let y = self.ins(Inst::Mul(fx, x[h]));
                acc = self.ins(Inst::Add(acc, y));
                let y = self.ins(Inst::Mul(a[1 + k], a[1 + l]));
                let y = self.ins(Inst::Mul(faa, y));
                acc = self.ins(Inst::Add(acc, y));
                let y = self.ins(Inst::Mul(a[1 + k], x[1 + l]));
                let z = self.ins(Inst::Mul(a[1 + l], x[1 + k]));
                let y = self.ins(Inst::Add(y, z));
                let y = self.ins(Inst::Mul(fax, y));
                acc = self.ins(Inst::Add(acc, y));
                let y = self.ins(Inst::Mul(x[1 + k], x[1 + l]));
                let y = self.ins(Inst::Mul(fxx, y));
                out[h] = self.ins(Inst::Add(acc, y));
            }
        }
        out
    }
    fn gamma_pq(&mut self, a: &[V], x: &[V], upper: bool, i: u8, j: u8) -> Lanes {
        let (av, xv) = (a[0], x[0]);
        self.binary_prim(a, x, |di, dj| Inst::Gamma {
            upper,
            i: i + di,
            j: j + dj,
            a: av,
            x: xv,
        })
    }
    /// A unary primitive whose `k`-th derivative is `Call1(shim(k), x)`.
    fn unary_chain(&mut self, a: &[V], shim: impl Fn(u8) -> Shim1) -> Lanes {
        let v = self.ins(Inst::Call1(shim(0), a[0]));
        if self.lanes() == 1 {
            return vec![v];
        }
        let h1 = self.ins(Inst::Call1(shim(1), a[0]));
        let h2 = (self.layout.m > 0).then(|| self.ins(Inst::Call1(shim(2), a[0])));
        self.unary(a, v, h1, h2)
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

    /// Emit register instruction `at` (after [`Self::reach`]ing it); `args` are the argument
    /// indices of `u`, `params`, `seeds` and `du`.
    pub(super) fn flat(&mut self, at: usize, f: &Flat, args: [u8; 4]) {
        self.reach(at);
        match f {
            Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. } => self.control(f),
            _ => emit(self, f, args),
        }
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
    /// `a / b` unguarded (`DualV::div_body`).
    fn raw_div(&mut self, a: &[V], b: &[V]) -> Lanes;
    /// A constant right operand, as the type's `*_scalar_in`.
    fn scalar(&mut self, op: Arith, a: &[V], c: f64) -> Lanes;
    fn exp(&mut self, a: &[V]) -> Lanes;
    /// The type's own `ln`, unguarded.
    fn ln(&mut self, a: &[V]) -> Lanes;
    /// The type's own `sqrt`, unguarded.
    fn sqrt(&mut self, a: &[V]) -> Lanes;
    fn inv_logit(&mut self, a: &[V]) -> Lanes;
    fn logit(&mut self, a: &[V]) -> Lanes;
    fn cos(&mut self, a: &[V]) -> Lanes;
    fn acos(&mut self, a: &[V]) -> Lanes;
    fn ln_gamma(&mut self, a: &[V]) -> Lanes;
    fn abs(&mut self, a: &[V]) -> Lanes;
    fn pow(&mut self, a: &[V], e: &[V]) -> Lanes;
    fn gamma_pq(&mut self, a: &[V], x: &[V], upper: bool, i: u8, j: u8) -> Lanes;
    fn unary_chain(&mut self, a: &[V], shim: &dyn Fn(u8) -> Shim1) -> Lanes;

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
    fn reg(&self, r: u32) -> Lanes {
        self.regs_ref()[r as usize].clone()
    }
    fn set(&mut self, d: u32, l: Lanes) {
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
    fn raw_div(&mut self, a: &[V], b: &[V]) -> Lanes {
        Lower::div(self, a, b)
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
    fn sqrt(&mut self, a: &[V]) -> Lanes {
        Lower::sqrt(self, a)
    }
    fn inv_logit(&mut self, a: &[V]) -> Lanes {
        Lower::inv_logit(self, a)
    }
    fn logit(&mut self, a: &[V]) -> Lanes {
        Lower::logit(self, a)
    }
    fn cos(&mut self, a: &[V]) -> Lanes {
        Lower::cos(self, a)
    }
    fn acos(&mut self, a: &[V]) -> Lanes {
        Lower::acos(self, a)
    }
    fn ln_gamma(&mut self, a: &[V]) -> Lanes {
        Lower::ln_gamma(self, a)
    }
    fn abs(&mut self, a: &[V]) -> Lanes {
        Lower::abs(self, a)
    }
    fn pow(&mut self, a: &[V], e: &[V]) -> Lanes {
        Lower::pow(self, a, e)
    }
    fn gamma_pq(&mut self, a: &[V], x: &[V], upper: bool, i: u8, j: u8) -> Lanes {
        Lower::gamma_pq(self, a, x, upper, i, j)
    }
    fn unary_chain(&mut self, a: &[V], shim: &dyn Fn(u8) -> Shim1) -> Lanes {
        Lower::unary_chain(self, a, shim)
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
        Flat::DivRaw { d, a, b } => {
            let l = al.raw_div(&al.reg(a), &al.reg(b));
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
        Flat::GammaPq {
            d,
            a,
            x,
            upper,
            i,
            j,
        } => {
            let l = al.gamma_pq(&al.reg(a), &al.reg(x), upper, i, j);
            al.set(d, l);
        }
        Flat::Unary { d, op, a } => {
            let x = al.reg(a);
            let l = match op {
                Unary::Exp => al.exp(&x),
                Unary::Sqrt => al.sqrt(&x),
                Unary::InvLogit => al.inv_logit(&x),
                Unary::Logit => al.logit(&x),
                Unary::Cos => al.cos(&x),
                Unary::Acos => al.acos(&x),
                Unary::LnRaw => al.ln(&x),
                Unary::StopGrad => {
                    let v = al.ins(Inst::StopGrad(x[0]));
                    al.constant(v)
                }
                Unary::Sin => al.unary_chain(&x, &|k| {
                    [Shim1::Sin, Shim1::Cos, Shim1::NegSin][k as usize % 3]
                }),
                Unary::Digamma => al.unary_chain(&x, &|k| Shim1::polygamma(1 + k)),
                Unary::Trigamma => al.unary_chain(&x, &|k| Shim1::polygamma(2 + k)),
                Unary::Tetragamma => al.unary_chain(&x, &|k| Shim1::polygamma(3 + k)),
                Unary::ExpRel(k) => al.unary_chain(&x, &|j| Shim1::ExpRel(k + j)),
                Unary::Jet(f, k) => al.unary_chain(&x, &|j| Shim1::Jet(f, k + j)),
                Unary::LnGamma => al.ln_gamma(&x),
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
        // A condition is the dual constant `0` or `1`, as the interpreter pushes it.
        Flat::Cmp { d, cc, a, b } => {
            let (x, y) = (al.reg(a)[0], al.reg(b)[0]);
            let c = al.ins(Inst::Cmp(cc, x, y));
            let l = bool01(al, c);
            al.set(d, l);
        }
        Flat::And { d, a, b } | Flat::Or { d, a, b } => {
            let z = al.k(0.0);
            let (x, y) = (al.reg(a)[0], al.reg(b)[0]);
            let x = al.ins(Inst::Cmp(Cc::Ne, x, z));
            let y = al.ins(Inst::Cmp(Cc::Ne, y, z));
            let c = al.ins(match *f {
                Flat::And { .. } => Inst::And(x, y),
                _ => Inst::Or(x, y),
            });
            let l = bool01(al, c);
            al.set(d, l);
        }
        Flat::Not { d, a } => {
            let z = al.k(0.0);
            let x = al.reg(a)[0];
            let c = al.ins(Inst::Cmp(Cc::Eq, x, z));
            let l = bool01(al, c);
            al.set(d, l);
        }
        // Ordered equality is false exactly for a `NaN`.
        Flat::Present { d, a } => {
            let x = al.reg(a)[0];
            let c = al.ins(Inst::Cmp(Cc::Eq, x, x));
            let l = bool01(al, c);
            al.set(d, l);
        }
        Flat::HostArg { a } => {
            let x = al.reg(a)[0];
            al.ins(Inst::HostArg(x));
        }
        Flat::Host { f } => {
            al.ins(Inst::Host(f));
        }
        Flat::HostOut { d, k } => {
            let v = al.ins(Inst::HostOut(k));
            let l = al.constant(v);
            al.set(d, l);
        }
        Flat::LoopArg { .. } | Flat::Loop { .. } | Flat::LoopOut { .. } => {
            unreachable!("a loop call is the plan's, never a chunk's")
        }
        Flat::Gate { .. } => unreachable!("resolved per variant before lowering"),
        Flat::Select { d, c, a, b } => {
            let z = al.k(0.0);
            let cv = al.reg(c)[0];
            let cond = al.ins(Inst::Cmp(Cc::Ne, cv, z));
            let l = al.select(cond, &al.reg(a), &al.reg(b));
            al.set(d, l);
        }
        Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. } => {
            unreachable!("`Lower::flat` handles an `if`'s control instructions")
        }
    }
}

/// `T::from_f64(if c { 1 } else { 0 })`.
fn bool01<A: Alg>(al: &mut A, c: V) -> Lanes {
    let one = al.k(1.0);
    let z = al.k(0.0);
    let v = al.ins(Inst::Select(c, one, z));
    al.constant(v)
}

/// The whole program over `layout`: `fn(u, params, seeds, du)`.
pub(super) fn forward(code: &[Flat], n_regs: usize, layout: Layout) -> Func {
    let mut f = Func::new(4);
    let mut l = Lower::new(&mut f, layout, n_regs);
    for (at, i) in code.iter().enumerate() {
        l.flat(at, i, [0, 1, 2, 3]);
    }
    l.reach(code.len());
    f
}

/// The reverse pass of a forward [`Func`] of `n` arguments: a function of `2n` arguments
/// whose argument `n + a` is the adjoint buffer of argument `a`, laid out like it.
///
/// It runs the forward instructions, seeds each stored value's adjoint from its output's
/// adjoint buffer (the last store to a location is the one that counts), applies every
/// instruction's local derivative in reverse order, and adds each loaded location's
/// adjoint into its input's adjoint buffer. Conditions carry no derivative; a select
/// passes the adjoint to the operand it chose; an instruction emitted in an `if` arm
/// contributes its local derivatives only where that arm is live ([`Func::live`]).
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
            Inst::Gamma { upper, i, j, a, x } => Inst::Gamma {
                upper,
                i,
                j,
                a: r(a),
                x: r(x),
            },
            Inst::Cmp(c, x, y) => Inst::Cmp(c, r(x), r(y)),
            Inst::And(x, y) => Inst::And(r(x), r(y)),
            Inst::Or(x, y) => Inst::Or(r(x), r(y)),
            Inst::Select(c, x, y) => Inst::Select(r(c), r(x), r(y)),
            Inst::HostArg(x) => Inst::HostArg(r(x)),
            Inst::StopGrad(x) => Inst::StopGrad(r(x)),
            Inst::Host(_) | Inst::HostOut(_) => *inst,
        };
        map[i] = b.push(m);
    }
    // Adjoints, indexed by forward instruction.
    let mut adj: Vec<Option<V>> = vec![None; fwd.insts.len()];
    // The condition the instruction being differentiated is live under, in `b`.
    let mut live: Option<V> = None;
    fn acc(b: &mut Func, adj: &mut [Option<V>], live: Option<V>, x: V, d: V, negate: bool) {
        // A dead arm's derivative is an exact zero, whatever its values were.
        let d = match live {
            Some(c) => {
                let z = b.k(0.0);
                b.push(Inst::Select(c, d, z))
            }
            None => d,
        };
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
        acc(&mut b, &mut adj, live, v, seed, false);
    }
    let mut loads: Vec<((u8, u32), V)> = Vec::new();
    for i in (0..fwd.insts.len()).rev() {
        let Some(g) = adj[i] else { continue };
        let fw = |x: V| map[x as usize];
        live = fwd.live.get(i).copied().flatten().map(fw);
        match fwd.insts[i] {
            // A load is never emitted in an arm (the prologue loads every input).
            Inst::Load { arg, off } => loads.push(((arg, off), g)),
            Inst::Const(_) | Inst::Store { .. } | Inst::Cmp(..) | Inst::And(..) | Inst::Or(..) => {}
            // A host call's outputs are constants of the computation, as is a stopped value.
            Inst::HostArg(_) | Inst::Host(_) | Inst::HostOut(_) | Inst::StopGrad(_) => {}
            Inst::Floor(_) | Inst::Ceil(_) | Inst::Call1(Shim1::Round, _) => {}
            Inst::Add(x, y) => {
                acc(&mut b, &mut adj, live, x, g, false);
                acc(&mut b, &mut adj, live, y, g, false);
            }
            Inst::Sub(x, y) => {
                acc(&mut b, &mut adj, live, x, g, false);
                acc(&mut b, &mut adj, live, y, g, true);
            }
            Inst::Mul(x, y) => {
                let dx = b.push(Inst::Mul(g, fw(y)));
                acc(&mut b, &mut adj, live, x, dx, false);
                let dy = b.push(Inst::Mul(g, fw(x)));
                acc(&mut b, &mut adj, live, y, dy, false);
            }
            Inst::Div(x, y) => {
                // `q = x/y`: `dx = g/y`, `dy = −(g/y)·q`.
                let t = b.push(Inst::Div(g, fw(y)));
                acc(&mut b, &mut adj, live, x, t, false);
                let dy = b.push(Inst::Mul(t, map[i]));
                acc(&mut b, &mut adj, live, y, dy, true);
            }
            Inst::Neg(x) => acc(&mut b, &mut adj, live, x, g, true),
            Inst::Abs(x) => {
                // `|x|′` as the duals take it: `+1` for `x ≥ 0`.
                let z = b.k(0.0);
                let nonneg = b.push(Inst::Cmp(Cc::Ge, fw(x), z));
                let ng = b.push(Inst::Neg(g));
                let d = b.push(Inst::Select(nonneg, g, ng));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Exp, x) => {
                let d = b.push(Inst::Mul(g, map[i]));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Ln, x) => {
                let d = b.push(Inst::Div(g, fw(x)));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Sqrt, x) => {
                let half = b.k(0.5);
                let gh = b.push(Inst::Mul(g, half));
                let d = b.push(Inst::Div(gh, map[i]));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Cos, x) => {
                let s = b.push(Inst::Call1(Shim1::Sin, fw(x)));
                let d = b.push(Inst::Mul(g, s));
                acc(&mut b, &mut adj, live, x, d, true);
            }
            Inst::Call1(Shim1::Sin, x) => {
                let c = b.push(Inst::Call1(Shim1::Cos, fw(x)));
                let d = b.push(Inst::Mul(g, c));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::LnGamma, x) => {
                let p = b.push(Inst::Call1(Shim1::Digamma, fw(x)));
                let d = b.push(Inst::Mul(g, p));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Digamma, x) => {
                let p = b.push(Inst::Call1(Shim1::Trigamma, fw(x)));
                let d = b.push(Inst::Mul(g, p));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(s @ (Shim1::Trigamma | Shim1::Tetragamma), x) => {
                let next = if s == Shim1::Trigamma {
                    Shim1::Tetragamma
                } else {
                    Shim1::Pentagamma
                };
                let p = b.push(Inst::Call1(next, fw(x)));
                let d = b.push(Inst::Mul(g, p));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::NegSin, x) => {
                let c = b.push(Inst::Call1(Shim1::Cos, fw(x)));
                let d = b.push(Inst::Mul(g, c));
                acc(&mut b, &mut adj, live, x, d, true);
            }
            Inst::Call1(Shim1::Jet(f, k), x) => {
                let p = b.push(Inst::Call1(Shim1::Jet(f, k.saturating_add(1)), fw(x)));
                let d = b.push(Inst::Mul(g, p));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::ExpRel(k), x) => {
                let p = b.push(Inst::Call1(Shim1::ExpRel(k.saturating_add(1)), fw(x)));
                let d = b.push(Inst::Mul(g, p));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            // Nothing differentiates pentagamma again; a NaN adjoint makes a misuse loud (a
            // non-finite gradient the checks reject) rather than silently wrong.
            Inst::Call1(Shim1::Pentagamma | Shim1::Nan, x) => {
                let nan = b.k(f64::NAN);
                let d = b.push(Inst::Mul(g, nan));
                acc(&mut b, &mut adj, live, x, d, false);
            }
            Inst::Call1(Shim1::Acos, x) => {
                let one = b.k(1.0);
                let xx = b.push(Inst::Mul(fw(x), fw(x)));
                let s = b.push(Inst::Sub(one, xx));
                let rs = b.push(Inst::Call1(Shim1::Sqrt, s));
                let d = b.push(Inst::Div(g, rs));
                acc(&mut b, &mut adj, live, x, d, true);
            }
            Inst::Powf(x, y) => {
                // `r = xʸ`: `dx = g·y·x^(y−1)`; `dy = g·r·ln x` for `x > 0`, and `0` where
                // the logarithm does not exist (`0^y` is flat in `y` for `y > 0`).
                let one = b.k(1.0);
                let ym1 = b.push(Inst::Sub(fw(y), one));
                let p = b.push(Inst::Powf(fw(x), ym1));
                let yp = b.push(Inst::Mul(fw(y), p));
                let dx = b.push(Inst::Mul(g, yp));
                acc(&mut b, &mut adj, live, x, dx, false);
                let lx = b.push(Inst::Call1(Shim1::Ln, fw(x)));
                let rl = b.push(Inst::Mul(map[i], lx));
                let dy = b.push(Inst::Mul(g, rl));
                let z = b.k(0.0);
                let pos = b.push(Inst::Cmp(Cc::Gt, fw(x), z));
                let dy = b.push(Inst::Select(pos, dy, z));
                acc(&mut b, &mut adj, live, y, dy, false);
            }
            Inst::Gamma {
                upper,
                i,
                j,
                a: x,
                x: y,
            } => {
                let p = |i, j| Inst::Gamma {
                    upper,
                    i,
                    j,
                    a: fw(x),
                    x: fw(y),
                };
                let pa = b.push(p(i + 1, j));
                let dx = b.push(Inst::Mul(g, pa));
                acc(&mut b, &mut adj, live, x, dx, false);
                let px = b.push(p(i, j + 1));
                let dy = b.push(Inst::Mul(g, px));
                acc(&mut b, &mut adj, live, y, dy, false);
            }
            Inst::Select(c, x, y) => {
                let z = b.k(0.0);
                let dx = b.push(Inst::Select(fw(c), g, z));
                acc(&mut b, &mut adj, live, x, dx, false);
                let dy = b.push(Inst::Select(fw(c), z, g));
                acc(&mut b, &mut adj, live, y, dy, false);
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

/// Forward-mode tangents of a [`Func`] of `n` arguments in `k` directions: a function of
/// `2n` arguments whose argument `n + a` holds the tangents of argument `a`, direction `j`
/// of location `off` at `off·k + j`. Every load reads its tangents, every store writes its
/// value's. Over [`reverse`] this is forward-over-reverse: Hessian-vector products of the
/// function the reverse differentiates, one derivative order above the plan's own, with no
/// layout of that order (#2786).
///
/// Tangents are kept sparse: a value no input reaches has none, and emits nothing. A dead
/// arm needs no mask here — its value is only read through the [`Inst::Select`] that chose
/// it, and the select picks the taken arm's tangent.
pub(super) fn tangent(f: &Func, k: usize) -> Func {
    type Tan = Option<Vec<V>>;
    // `Σ ± scale·tangent`, sparse: `None` when every term is.
    fn lin(b: &mut Func, k: usize, terms: &[(&Tan, Option<V>, bool)]) -> Tan {
        let live: Vec<_> = terms.iter().filter(|t| t.0.is_some()).collect();
        if live.is_empty() {
            return None;
        }
        Some(
            (0..k)
                .map(|j| {
                    let mut acc: Option<V> = None;
                    for &&(tv, scale, neg) in &live {
                        let mut d = tv.as_ref().expect("filtered")[j];
                        if let Some(s) = scale {
                            d = b.push(Inst::Mul(d, s));
                        }
                        acc = Some(match (acc, neg) {
                            (None, false) => d,
                            (None, true) => b.push(Inst::Neg(d)),
                            (Some(a), false) => b.push(Inst::Add(a, d)),
                            (Some(a), true) => b.push(Inst::Sub(a, d)),
                        });
                    }
                    acc.expect("non-empty")
                })
                .collect(),
        )
    }
    let n = f.n_args;
    let mut b = Func::new(2 * n);
    let mut map = vec![V::MAX; f.insts.len()];
    let mut tan: Vec<Tan> = vec![None; f.insts.len()];
    for (i, inst) in f.insts.iter().enumerate() {
        b.cur = f.live[i].map(|c| map[c as usize]);
        let fw = |x: V| map[x as usize];
        // The value: the instruction, remapped (a store is emitted after its tangents).
        let m = match *inst {
            Inst::Store { .. } => None,
            Inst::Load { .. } | Inst::Const(_) | Inst::Host(_) | Inst::HostOut(_) => Some(*inst),
            Inst::Add(x, y) => Some(Inst::Add(fw(x), fw(y))),
            Inst::Sub(x, y) => Some(Inst::Sub(fw(x), fw(y))),
            Inst::Mul(x, y) => Some(Inst::Mul(fw(x), fw(y))),
            Inst::Div(x, y) => Some(Inst::Div(fw(x), fw(y))),
            Inst::Neg(x) => Some(Inst::Neg(fw(x))),
            Inst::Abs(x) => Some(Inst::Abs(fw(x))),
            Inst::Floor(x) => Some(Inst::Floor(fw(x))),
            Inst::Ceil(x) => Some(Inst::Ceil(fw(x))),
            Inst::Call1(s, x) => Some(Inst::Call1(s, fw(x))),
            Inst::Powf(x, y) => Some(Inst::Powf(fw(x), fw(y))),
            Inst::Gamma { upper, i, j, a, x } => Some(Inst::Gamma {
                upper,
                i,
                j,
                a: fw(a),
                x: fw(x),
            }),
            Inst::Cmp(c, x, y) => Some(Inst::Cmp(c, fw(x), fw(y))),
            Inst::And(x, y) => Some(Inst::And(fw(x), fw(y))),
            Inst::Or(x, y) => Some(Inst::Or(fw(x), fw(y))),
            Inst::Select(c, x, y) => Some(Inst::Select(fw(c), fw(x), fw(y))),
            Inst::HostArg(x) => Some(Inst::HostArg(fw(x))),
            Inst::StopGrad(x) => Some(Inst::StopGrad(fw(x))),
        };
        if let Some(m) = m {
            map[i] = b.push(m);
        }
        let fw = |x: V| map[x as usize];
        let t = |x: V| &tan[x as usize];
        // `x ↦ s(x)·dx` for a unary `f` with `f′ = ±s`.
        let unary = |b: &mut Func, x: V, s: Shim1, neg: bool| -> Tan {
            t(x).as_ref()?;
            let p = b.push(Inst::Call1(s, fw(x)));
            lin(b, k, &[(t(x), Some(p), neg)])
        };
        let new: Tan = match *inst {
            Inst::Load { arg, off } => Some(
                (0..k)
                    .map(|j| {
                        b.push(Inst::Load {
                            arg: n + arg,
                            off: off * k as u32 + j as u32,
                        })
                    })
                    .collect(),
            ),
            Inst::Store { arg, off, v } => {
                let z = b.k(0.0);
                for j in 0..k {
                    let d = t(v).as_ref().map_or(z, |tv| tv[j]);
                    b.push(Inst::Store {
                        arg: n + arg,
                        off: off * k as u32 + j as u32,
                        v: d,
                    });
                }
                b.push(Inst::Store { arg, off, v: fw(v) });
                None
            }
            Inst::Const(_) | Inst::Cmp(..) | Inst::And(..) | Inst::Or(..) => None,
            Inst::HostArg(_) | Inst::Host(_) | Inst::HostOut(_) | Inst::StopGrad(_) => None,
            Inst::Floor(_) | Inst::Ceil(_) | Inst::Call1(Shim1::Round, _) => None,
            Inst::Add(x, y) => lin(&mut b, k, &[(t(x), None, false), (t(y), None, false)]),
            Inst::Sub(x, y) => lin(&mut b, k, &[(t(x), None, false), (t(y), None, true)]),
            Inst::Neg(x) => lin(&mut b, k, &[(t(x), None, true)]),
            Inst::Mul(x, y) => lin(
                &mut b,
                k,
                &[(t(x), Some(fw(y)), false), (t(y), Some(fw(x)), false)],
            ),
            Inst::Div(x, y) if t(x).is_some() || t(y).is_some() => {
                // `q = x/y`: `dq = (dx − q·dy)/y`.
                let one = b.k(1.0);
                let r = b.push(Inst::Div(one, fw(y)));
                let q = b.push(Inst::Mul(map[i], r));
                lin(&mut b, k, &[(t(x), Some(r), false), (t(y), Some(q), true)])
            }
            Inst::Abs(x) if t(x).is_some() => {
                // `|x|′` as the reverse takes it: `+1` for `x ≥ 0`.
                let z = b.k(0.0);
                let nonneg = b.push(Inst::Cmp(Cc::Ge, fw(x), z));
                let (one, m1) = (b.k(1.0), b.k(-1.0));
                let sg = b.push(Inst::Select(nonneg, one, m1));
                lin(&mut b, k, &[(t(x), Some(sg), false)])
            }
            Inst::Call1(Shim1::Exp, x) => lin(&mut b, k, &[(t(x), Some(map[i]), false)]),
            Inst::Call1(Shim1::Ln, x) if t(x).is_some() => {
                let one = b.k(1.0);
                let r = b.push(Inst::Div(one, fw(x)));
                lin(&mut b, k, &[(t(x), Some(r), false)])
            }
            Inst::Call1(Shim1::Sqrt, x) if t(x).is_some() => {
                let half = b.k(0.5);
                let r = b.push(Inst::Div(half, map[i]));
                lin(&mut b, k, &[(t(x), Some(r), false)])
            }
            Inst::Call1(Shim1::Acos, x) if t(x).is_some() => {
                let one = b.k(1.0);
                let xx = b.push(Inst::Mul(fw(x), fw(x)));
                let s = b.push(Inst::Sub(one, xx));
                let rs = b.push(Inst::Call1(Shim1::Sqrt, s));
                let r = b.push(Inst::Div(one, rs));
                lin(&mut b, k, &[(t(x), Some(r), true)])
            }
            Inst::Div(..) | Inst::Abs(_) => None,
            Inst::Call1(Shim1::Ln | Shim1::Sqrt | Shim1::Acos, _) => None,
            Inst::Call1(Shim1::Cos, x) => unary(&mut b, x, Shim1::Sin, true),
            Inst::Call1(Shim1::Sin, x) => unary(&mut b, x, Shim1::Cos, false),
            Inst::Call1(Shim1::NegSin, x) => unary(&mut b, x, Shim1::Cos, true),
            Inst::Call1(Shim1::LnGamma, x) => unary(&mut b, x, Shim1::Digamma, false),
            Inst::Call1(Shim1::Digamma, x) => unary(&mut b, x, Shim1::Trigamma, false),
            Inst::Call1(Shim1::Trigamma, x) => unary(&mut b, x, Shim1::Tetragamma, false),
            Inst::Call1(Shim1::Tetragamma, x) => unary(&mut b, x, Shim1::Pentagamma, false),
            Inst::Call1(Shim1::Pentagamma | Shim1::Nan, x) => unary(&mut b, x, Shim1::Nan, false),
            Inst::Call1(Shim1::ExpRel(e), x) => {
                unary(&mut b, x, Shim1::ExpRel(e.saturating_add(1)), false)
            }
            Inst::Call1(Shim1::Jet(g, e), x) => {
                unary(&mut b, x, Shim1::Jet(g, e.saturating_add(1)), false)
            }
            Inst::Powf(x, y) if t(x).is_some() || t(y).is_some() => {
                // As the reverse: `y·x^(y−1)` and, for `x > 0`, `r·ln x`.
                let one = b.k(1.0);
                let ym1 = b.push(Inst::Sub(fw(y), one));
                let p = b.push(Inst::Powf(fw(x), ym1));
                let dx = b.push(Inst::Mul(fw(y), p));
                let lx = b.push(Inst::Call1(Shim1::Ln, fw(x)));
                let rl = b.push(Inst::Mul(map[i], lx));
                let z = b.k(0.0);
                let pos = b.push(Inst::Cmp(Cc::Gt, fw(x), z));
                let dy = b.push(Inst::Select(pos, rl, z));
                lin(
                    &mut b,
                    k,
                    &[(t(x), Some(dx), false), (t(y), Some(dy), false)],
                )
            }
            Inst::Gamma {
                upper,
                i: gi,
                j: gj,
                a: x,
                x: y,
            } if t(x).is_some() || t(y).is_some() => {
                let p = |i, j| Inst::Gamma {
                    upper,
                    i,
                    j,
                    a: fw(x),
                    x: fw(y),
                };
                let pa = b.push(p(gi + 1, gj));
                let px = b.push(p(gi, gj + 1));
                lin(
                    &mut b,
                    k,
                    &[(t(x), Some(pa), false), (t(y), Some(px), false)],
                )
            }
            Inst::Powf(..) | Inst::Gamma { .. } => None,
            Inst::Select(c, x, y) if t(x).is_some() || t(y).is_some() => {
                let z = b.k(0.0);
                Some(
                    (0..k)
                        .map(|j| {
                            let a = t(x).as_ref().map_or(z, |v| v[j]);
                            let e = t(y).as_ref().map_or(z, |v| v[j]);
                            b.push(Inst::Select(fw(c), a, e))
                        })
                        .collect(),
                )
            }
            Inst::Select(..) => None,
        };
        tan[i] = new;
    }
    b.cur = None;
    b
}

/// [`tangent`] in `k` directions with the tangents packed into the value arguments: a
/// function of the same arguments over slots of `l·(1 + k)` lanes, location `i·l + lane` of
/// the base function at `i·l(1+k) + lane` and its tangent along `j` at
/// `i·l(1+k) + l + lane·k + j` (`Layout::tan`). Argument 2 (the literals, indexed by slot
/// alone) has no tangents: their loads are zeros.
pub(super) fn tangent_packed(f: &Func, k: usize, l: usize) -> Func {
    let t = tangent(f, k);
    let n = f.n_args;
    let wl = (l * (1 + k)) as u32;
    let (l32, k32) = (l as u32, k as u32);
    let place = |arg: u8, off: u32| -> (u8, u32) {
        if arg < n {
            let (i, lane) = (off / l32, off % l32);
            (arg, i * wl + lane)
        } else {
            let (loc, j) = (off / k32, off % k32);
            let (i, lane) = (loc / l32, loc % l32);
            (arg - n, i * wl + l32 + lane * k32 + j)
        }
    };
    let mut out = Func::new(n);
    out.insts = t
        .insts
        .iter()
        .map(|inst| match *inst {
            Inst::Load { arg, off } if arg == n + 2 => {
                let _ = off;
                Inst::Const(0f64.to_bits())
            }
            Inst::Load { arg, off } => {
                let (arg, off) = place(arg, off);
                Inst::Load { arg, off }
            }
            Inst::Store { arg, off, v } => {
                let (arg, off) = place(arg, off);
                Inst::Store { arg, off, v }
            }
            i => i,
        })
        .collect();
    out.live = t.live.clone();
    out
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
            Inst::Call1(Shim1::Sqrt, a) => r(a).sqrt(),
            Inst::Call1(Shim1::Cos, a) => r(a).cos(),
            Inst::Call1(Shim1::Sin, a) => r(a).sin(),
            Inst::Call1(Shim1::Acos, a) => r(a).acos(),
            Inst::Call1(Shim1::LnGamma, a) => crate::stats::special::ln_gamma(r(a)),
            Inst::Call1(Shim1::Digamma, a) => crate::stats::special::digamma(r(a)),
            Inst::Call1(Shim1::Trigamma, a) => crate::stats::special::trigamma(r(a)),
            Inst::Call1(Shim1::Tetragamma, a) => crate::stats::special::tetragamma(r(a)),
            Inst::Call1(Shim1::Pentagamma, a) => crate::stats::special::pentagamma(r(a)),
            Inst::Call1(
                s @ (Shim1::NegSin | Shim1::ExpRel(_) | Shim1::Jet(..) | Shim1::Nan),
                a,
            ) => (s.f())(r(a)),
            Inst::Powf(a, b) => r(a).powf(r(b)),
            Inst::Gamma { upper, i, j, a, x } => {
                crate::stats::special::gamma_pq_partial(r(a), r(x), upper, i, j)
            }
            Inst::Cmp(cc, a, b) => {
                c[i] = cc.holds(r(a), r(b));
                0.0
            }
            Inst::And(a, b) => {
                c[i] = c[a as usize] && c[b as usize];
                0.0
            }
            Inst::Or(a, b) => {
                c[i] = c[a as usize] || c[b as usize];
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
            Inst::HostArg(a) => {
                host::arg(r(a));
                0.0
            }
            Inst::Host(f) => {
                host::call(f);
                0.0
            }
            Inst::HostOut(k) => host::out(k),
            Inst::StopGrad(a) => r(a),
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
            Inst::Call1(Shim1::Sqrt, a) => {
                let s = r(a).sqrt();
                (s, 0.5 * d(a) / s)
            }
            Inst::Call1(Shim1::Cos, a) => (r(a).cos(), -r(a).sin() * d(a)),
            Inst::Call1(Shim1::Sin, a) => (r(a).sin(), r(a).cos() * d(a)),
            Inst::Call1(Shim1::Acos, a) => (r(a).acos(), -d(a) / (1.0 - r(a) * r(a)).sqrt()),
            Inst::Call1(Shim1::LnGamma, a) => (
                crate::stats::special::ln_gamma(r(a)),
                crate::stats::special::digamma(r(a)) * d(a),
            ),
            Inst::Call1(Shim1::Digamma, a) => (
                crate::stats::special::digamma(r(a)),
                crate::stats::special::trigamma(r(a)) * d(a),
            ),
            Inst::Call1(Shim1::Trigamma, a) => (
                crate::stats::special::trigamma(r(a)),
                crate::stats::special::tetragamma(r(a)) * d(a),
            ),
            Inst::Call1(Shim1::Tetragamma, a) => (
                crate::stats::special::tetragamma(r(a)),
                crate::stats::special::pentagamma(r(a)) * d(a),
            ),
            Inst::Call1(Shim1::Pentagamma, a) => {
                (crate::stats::special::pentagamma(r(a)), f64::NAN)
            }
            Inst::Call1(Shim1::NegSin, a) => (-r(a).sin(), -r(a).cos() * d(a)),
            Inst::Call1(s @ Shim1::ExpRel(k), a) => (
                (s.f())(r(a)),
                (Shim1::ExpRel(k.saturating_add(1)).f())(r(a)) * d(a),
            ),
            Inst::Call1(s @ Shim1::Jet(f, k), a) => (
                (s.f())(r(a)),
                (Shim1::Jet(f, k.saturating_add(1)).f())(r(a)) * d(a),
            ),
            Inst::Call1(Shim1::Nan, _) => (f64::NAN, f64::NAN),
            Inst::Gamma { upper, i, j, a, x } => {
                let f = |i, j| crate::stats::special::gamma_pq_partial(r(a), r(x), upper, i, j);
                (f(i, j), f(i + 1, j) * d(a) + f(i, j + 1) * d(x))
            }
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
                c[i] = cc.holds(r(a), r(b));
                (0.0, 0.0)
            }
            Inst::And(a, b) => {
                c[i] = c[a as usize] && c[b as usize];
                (0.0, 0.0)
            }
            Inst::Or(a, b) => {
                c[i] = c[a as usize] || c[b as usize];
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
            Inst::HostArg(a) => {
                host::arg(r(a));
                (0.0, 0.0)
            }
            Inst::Host(f) => {
                host::call(f);
                (0.0, 0.0)
            }
            Inst::HostOut(k) => (host::out(k), 0.0),
            Inst::StopGrad(a) => (r(a), 0.0),
        };
        x[i] = val;
        dx[i] = tan;
    }
}
