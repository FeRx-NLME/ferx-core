//! PROTOTYPE (#2786): record a subject's whole η-jet recording once, compile it, reuse it.
//!
//! [`Tr`] is a recording number type. Running the inner loop's recorder on it writes the
//! register program ([`Flat`]) of the subject's computation: θ and η are inputs, data are
//! literals, and identical operations on identical operands are one register
//! (hash-consing). The program is lowered by the RHS's own lane code ([`ir::forward`]),
//! so each operation is `DualF`'s arithmetic, lane for lane; then the lanes the inputs fix
//! (η's one-hot seeds, θ's zero tangents) become constants and a folding pass drops the
//! terms they make exactly `0` or `1`; Cranelift compiles the result.
//!
//! **Soundness.** A recording is valid only where every branch the recorder took still
//! goes the same way. A read of an input-dependent value ([`PkNum::val`]) aborts the
//! recording unless it is a guard; a guard ([`Tr::guard_cmp`]) is recompiled as a check,
//! and a failed check returns `None` (the caller takes the generic path). Values the
//! recorder computes in `f64` from θ or η and lifts as literals cannot be seen here, so a
//! compiled recording is validated against the generic path at perturbed inputs before
//! it is used.
//!
//! Every derivative a fit takes goes through the subject's plan; `FERX_KJIT_DEBUG=1` reports
//! aborted recordings.
#[path = "kernel_jit_fixed.rs"]
mod fixed;
#[path = "kernel_jit_lanes.rs"]
mod lanes;
#[path = "kernel_jit_loop.rs"]
mod ode_loop;
#[cfg(test)]
#[path = "kernel_jit_plan_tests.rs"]
mod plan_tests;
pub(crate) use fixed::{record_fixed, record_repeat, Found, SolveFn};

/// What a plan's call (`Flat::Loop`) runs: an ODE segment, or a steady-state fixed point.
#[derive(Clone)]
pub(crate) enum CallKind {
    Ode(Arc<LoopSpec>),
    Fixed(Arc<fixed::FixedSpec>),
    Repeat(Arc<fixed::RepeatSpec>),
}

impl CallKind {
    fn n_in(&self) -> usize {
        match self {
            CallKind::Ode(s) => s.n_in(),
            CallKind::Fixed(f) => f.n_caps,
            CallKind::Repeat(r) => r.n + 1 + r.n_caps,
        }
    }
    fn n_out(&self) -> usize {
        match self {
            CallKind::Ode(s) => s.n_out(),
            CallKind::Fixed(f) => f.n,
            CallKind::Repeat(r) => r.n,
        }
    }
}

/// A call's forward, kept for its reverse.
pub(crate) enum CallLog {
    Ode(Vec<ode_loop::Entry>),
    Fixed(Option<fixed::FixedLog>),
}
use super::ir::{self, Func, Inst, Layout};
use super::{Flat, Unary};
use crate::sens::num::PkNum;
pub(crate) use ode_loop::LoopSpec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Compiled plans serve every derivative (#2786).
pub(crate) fn enabled() -> bool {
    true
}

pub(crate) fn debug() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("FERX_KJIT_DEBUG").is_ok())
}

/// A recorded value: its value (for the recorder's own decisions), its register, and
/// whether it depends on an input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tr {
    v: f64,
    r: u32,
    dep: bool,
}

/// The multiplicative hash rustc uses for its own tables (FxHash): one rotate, xor and
/// multiply per word. For the recorder's maps, whose keys are small integers and are
/// compared in full on a hit — SipHash's flood resistance buys nothing there (#2786).
#[derive(Default, Clone, Copy)]
pub(super) struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl std::hash::Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(w));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// [`Fx`] as a map's hasher.
pub(super) type FxBuild = std::hash::BuildHasherDefault<Fx>;

#[derive(Default)]
struct Rec {
    code: Vec<Flat>,
    n_regs: u32,
    cse: HashMap<(u8, u32, u32, u64), u32, FxBuild>,
    failed: Option<String>,
    /// The literals, in first-use order (argument 2 of the compiled function).
    pool: Vec<f64>,
    /// The conditions already required (`guard_reg`).
    guarded: std::collections::HashSet<(u32, bool), FxBuild>,
    /// The ODE segments recorded as loop calls (`Flat::Loop`'s `id`).
    loops: Vec<CallKind>,
    /// Whether an ODE segment was recorded (as a loop call or exactly, linear): the plan then
    /// holds only under the solver override in force now.
    ode: bool,
    /// Each register's recorded value (`affine_parts` differentiates with them).
    vals: Vec<f64>,
    /// Code positions where a chunk starts (`PkNum::chunk_mark`).
    marks: Vec<usize>,
    /// The η and θ input registers: `(register, index)`.
    eta_regs: Vec<(u32, usize)>,
    theta_regs: Vec<(u32, usize)>,
    guard: Option<u32>,
    /// The exact linear path's segments this recording integrated instead
    /// ([`PkNum::note_linear_decline`]): the plan reports them on every run.
    declined: usize,
    /// How many sub-plan regions ([`fixed::record_sub`]) enclose the code being recorded: a
    /// sub-plan's inputs are dense (any lane may be live), so a loop call recorded inside
    /// one cannot read its inputs' live directions off the open recording.
    sub_depth: u32,
}

thread_local! {
    static REC: RefCell<Option<Rec>> = const { RefCell::new(None) };
    /// The next ODE segment's span (`PkNum::span_hint`): each end's register, value, and
    /// whether it depends on the inputs.
    static SPAN: std::cell::Cell<Option<(u32, f64, bool, u32, f64, bool)>> =
        const { std::cell::Cell::new(None) };
}

fn rec<R>(f: impl FnOnce(&mut Rec) -> R) -> R {
    REC.with(|r| {
        f(r.borrow_mut()
            .as_mut()
            .expect("a kernel_jit recording is open"))
    })
}

fn fail(why: String) {
    rec(|r| {
        if r.failed.is_none() {
            r.failed = Some(why);
        }
    })
}

/// One register: the instruction `mk` builds for it, shared with an identical earlier
/// one when `key` matches.
fn node(key: Option<(u8, u32, u32, u64)>, mk: impl FnOnce(u32) -> Flat) -> u32 {
    rec(|r| {
        if let Some(k) = key {
            if let Some(&x) = r.cse.get(&k) {
                return x;
            }
        }
        let d = r.n_regs;
        if d >= u32::MAX as u32 - 2 {
            if r.failed.is_none() {
                r.failed = Some("register file full".into());
            }
            return 0;
        }
        r.n_regs += 1;
        r.code.push(mk(d as u32));
        if let Some(k) = key {
            r.cse.insert(k, d);
        }
        d
    })
}

thread_local! {
    static GLOG: std::cell::RefCell<Vec<GEntry>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
#[allow(clippy::type_complexity)]
/// TEMP: one recorded guard: site, outcome, constant operand, comparison (255: none), the
/// guarded operand's value.
type GEntry = (&'static std::panic::Location<'static>, bool, u64, u8, f64);
static GLOGS: std::sync::Mutex<Option<HashMap<String, Vec<GEntry>>>> = std::sync::Mutex::new(None);
/// TEMP: the first guard whose outcome differs from the last recording of `key`.
pub(crate) fn guard_log_diff(key: String) {
    if !debug() {
        return;
    }
    let now = GLOG.with(|g| std::mem::take(&mut *g.borrow_mut()));
    let mut m = GLOGS.lock().unwrap();
    let m = m.get_or_insert_with(HashMap::new);
    if let Some(old) = m.get(&key) {
        const CCS: [ir::Cc; 6] = [
            ir::Cc::Lt,
            ir::Cc::Le,
            ir::Cc::Gt,
            ir::Cc::Ge,
            ir::Cc::Eq,
            ir::Cc::Ne,
        ];
        let mut said = false;
        for (i, (o, n)) in old.iter().zip(&now).enumerate() {
            if o.0 != n.0 {
                eprintln!("kjit: guard paths diverge #{i}: {} -> {}", o.0, n.0);
                let lo = i.saturating_sub(4);
                let ctx = |v: &[GEntry]| {
                    v[lo..(i + 3).min(v.len())]
                        .iter()
                        .map(|e| {
                            format!(
                                "{}:{}",
                                e.0.file().rsplit('/').next().unwrap_or(""),
                                e.0.line()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                eprintln!(
                    "kjit:   old [{}] len {}\nkjit:   new [{}] len {}",
                    ctx(old),
                    old.len(),
                    ctx(&now),
                    now.len()
                );
                said = true;
                break;
            }
            let k = f64::from_bits(o.2);
            let fails = match CCS.get(o.3 as usize) {
                Some(cc) if !k.is_nan() => cc.holds(n.4, k) != o.1,
                _ => o.1 != n.1,
            };
            if fails {
                eprintln!(
                    "kjit: guard failed #{i}: {} {:?} k={k:e} now a={:e} (recorded {})",
                    o.0,
                    CCS.get(o.3 as usize),
                    n.4,
                    o.1
                );
                said = true;
                break;
            }
        }
        if !said {
            eprintln!("kjit: guard logs agree ({} vs {})", old.len(), now.len());
        }
    }
    m.insert(key, now);
}

/// Require the condition in register `c` (a dual `0`/`1`) to come out `holds` again — once:
/// a condition already required adds nothing.
#[track_caller]
fn guard_reg(c: u32, holds: bool) {
    guard_reg_k(c, holds, f64::NAN, 255, f64::NAN)
}
#[track_caller]
fn guard_reg_k(c: u32, holds: bool, k: f64, cc: u8, a: f64) {
    if debug() {
        let l = std::panic::Location::caller();
        GLOG.with(|g| g.borrow_mut().push((l, holds, k.to_bits(), cc, a)));
    }
    if !rec(|r| r.guarded.insert((c, holds))) {
        return;
    }
    let c = if holds {
        c
    } else {
        node(None, |d| Flat::Not { d, a: c as u32 })
    };
    let g = rec(|r| r.guard);
    let g = match g {
        None => c,
        Some(p) => node(None, |d| Flat::And {
            d,
            a: p as u32,
            b: c as u32,
        }),
    };
    rec(|r| r.guard = Some(g));
}

fn cc_of(cc: crate::sens::num::BranchCmp) -> ir::Cc {
    use crate::sens::num::BranchCmp as B;
    match cc {
        B::Lt => ir::Cc::Lt,
        B::Le => ir::Cc::Le,
        B::Gt => ir::Cc::Gt,
        B::Ge => ir::Cc::Ge,
        B::Eq => ir::Cc::Eq,
        B::Ne => ir::Cc::Ne,
    }
}

const T_K: u8 = 0;
const T_ADD: u8 = 1;
const T_SUB: u8 = 2;
const T_MUL: u8 = 3;
const T_DIV: u8 = 4;
const T_EXP: u8 = 5;
const T_LN: u8 = 6;
const T_ABS: u8 = 7;
const T_POW: u8 = 8;
const T_ADDK: u8 = 9;
const T_SUBK: u8 = 10;
const T_RSUBK: u8 = 11;
const T_MULK: u8 = 12;
const T_DIVK: u8 = 13;
const T_SQRT: u8 = 14;
const T_INVLOGIT: u8 = 15;
const T_LOGIT: u8 = 16;
const T_COS: u8 = 17;
const T_ACOS: u8 = 18;
const T_LNGAMMA: u8 = 19;
/// `P`; `Q` is the next tag.
const T_GAMMA: u8 = 20;
const T_VMDIV: u8 = 22;
const T_SIN: u8 = 23;
const T_DIGAMMA: u8 = 24;
const T_TRIGAMMA: u8 = 25;
const T_TETRAGAMMA: u8 = 26;
const T_EXPREL: u8 = 27;
const T_STOPGRAD: u8 = 28;
const T_SELECT: u8 = 29;
const T_JET: u8 = 30;
const T_FLOOR: u8 = 31;
const T_GATE: u8 = 32;
/// A comparison: this plus its `ir::Cc`.
const T_CMP: u8 = 40;

impl Tr {
    /// A recorded value; with `FERX_KJIT_DEBUG`, its value is kept for [`replay_check`].
    fn mk(v: f64, r: u32, dep: bool) -> Self {
        REC.with(|x| {
            if let Some(x) = x.borrow_mut().as_mut() {
                let r = r as usize;
                if x.vals.len() <= r {
                    x.vals.resize(r + 1, f64::NAN);
                }
                x.vals[r] = v;
            }
        });
        Tr { v, r, dep }
    }
    /// A literal: an entry of the subject's pool (argument 2), so subjects whose data
    /// differ share code. Its tangents are exact zeros.
    fn k(c: f64) -> Self {
        let key = (T_K, 0, 0, c.to_bits());
        let r = rec(|r| {
            if let Some(&x) = r.cse.get(&key) {
                return x;
            }
            let d = r.n_regs;
            if d >= u32::MAX as u32 - 2 {
                if r.failed.is_none() {
                    r.failed = Some("register file full".into());
                }
                return 0;
            }
            r.n_regs += 1;
            let idx = r.pool.len() as u32;
            r.pool.push(c);
            r.code.push(Flat::LoadSeed {
                d: d as u32,
                k: idx,
            });
            r.cse.insert(key, d);
            d
        });
        Tr::mk(c, r, false)
    }
    /// η direction `i`: a state input (`u[i]`).
    pub(crate) fn eta(i: usize, v: f64) -> Self {
        let r = node(None, |d| Flat::LoadState { d, i: i as u32 });
        rec(|rr| rr.eta_regs.push((r, i)));
        Tr::mk(v, r, true)
    }
    /// θ slot `j`: a parameter input (`params[j]`).
    pub(crate) fn theta(j: usize, v: f64) -> Self {
        let r = node(None, |d| Flat::LoadParam { d, slot: j as u32 });
        rec(|rr| rr.theta_regs.push((r, j)));
        Tr::mk(v, r, true)
    }
    fn bin(self, o: Self, tag: u8, v: f64, mk: fn(u32, u32, u32) -> Flat) -> Self {
        if !self.dep && !o.dep {
            // Data only: a literal (folded here, so it shares and costs nothing).
            return Self::k(v);
        }
        let (a, b) = (self.r as u32, o.r as u32);
        let r = node(Some((tag, self.r, o.r, 0)), |d| mk(d, a, b));
        Tr::mk(v, r, true)
    }
    fn scalar(self, c: f64, tag: u8, v: f64, mk: fn(u32, u32, f64) -> Flat) -> Self {
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r as u32;
        let r = node(Some((tag, self.r, 0, c.to_bits())), |d| mk(d, a, c));
        Tr::mk(v, r, true)
    }
    fn unary(self, tag: u8, op: Unary, v: f64) -> Self {
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r as u32;
        let r = node(Some((tag, self.r, 0, 0)), |d| Flat::Unary { d, op, a });
        Tr::mk(v, r, true)
    }
    /// `∂^(i+j) P(self, x) / ∂selfⁱ ∂xʲ` (of `Q` when `upper`).
    fn gamma_part(self, x: Self, upper: bool, i: u8, j: u8) -> Self {
        let v = crate::stats::special::gamma_pq_partial(self.v, x.v, upper, i, j);
        if !self.dep && !x.dep {
            return Self::k(v);
        }
        let (a, b) = (self.r as u32, x.r as u32);
        let tag = T_GAMMA + upper as u8;
        let r = node(
            Some((tag, self.r, x.r, ((i as u64) << 8) | j as u64)),
            |d| Flat::GammaPq {
                d,
                a,
                x: b,
                upper,
                i,
                j,
            },
        );
        Tr::mk(v, r, true)
    }
    /// A host call on `args`: run now for the values, and recorded to run again on every
    /// evaluation. Its outputs carry no derivative.
    fn host_call(f: crate::sens::num::HostFn, args: &[Self]) -> Vec<Self> {
        let vals: Vec<f64> = args.iter().map(|x| x.v).collect();
        let mut out = Vec::new();
        f.run(&vals, &mut out);
        if args.iter().all(|x| !x.dep) {
            return out.into_iter().map(Self::k).collect();
        }
        for x in args {
            let a = x.r as u32;
            rec(|r| r.code.push(Flat::HostArg { a }));
        }
        rec(|r| r.code.push(Flat::Host { f }));
        out.into_iter()
            .enumerate()
            .map(|(k, v)| {
                let r = node(None, |d| Flat::HostOut { d, k: k as u32 });
                Tr::mk(v, r, true)
            })
            .collect()
    }
    /// `c ≠ 0 ? a : b`, `c` a recorded condition: no guard.
    fn select_reg(c: Self, a: Self, b: Self) -> Self {
        let v = if c.v != 0.0 { a.v } else { b.v };
        if !c.dep {
            return if c.v != 0.0 { a } else { b };
        }
        let (cr, ar, br) = (c.r as u32, a.r as u32, b.r as u32);
        let r = node(Some((T_SELECT, c.r, a.r, b.r as u64)), |d| Flat::Select {
            d,
            c: cr,
            a: ar,
            b: br,
        });
        Tr::mk(v, r, true)
    }
    /// One gate node over both spellings (`Flat::Gate`): its value `take`'s.
    fn gate_node(k: u8, skip: Self, take: Self) -> Self {
        if skip.r == take.r || (!skip.dep && !take.dep) {
            return take;
        }
        let (ar, br) = (skip.r as u32, take.r as u32);
        let r = node(Some((T_GATE, skip.r, take.r, k as u64)), |d| Flat::Gate {
            d,
            a: ar,
            b: br,
            k,
        });
        Tr::mk(take.v, r, true)
    }
    /// `a / b` with the interpreter's guard: `0` where `|b| < 1e-30`.
    fn vm_div(a: Self, b: Self) -> Self {
        let v = if b.v.abs() < 1e-30 { 0.0 } else { a.v / b.v };
        if !a.dep && !b.dep {
            return Self::k(v);
        }
        let (x, y) = (a.r as u32, b.r as u32);
        let r = node(Some((T_VMDIV, a.r, b.r, 0)), |d| Flat::Div {
            d,
            a: x,
            b: y,
        });
        Tr::mk(v, r, true)
    }
    fn sin(self) -> Self {
        self.unary(T_SIN, Unary::Sin, self.v.sin())
    }
    /// The `k`-th derivative of `f` at `self` ([`crate::sens::num::JetFn`]): a different-route
    /// derivative, so the recording is not `exact`.
    fn jet_k(self, f: crate::sens::num::JetFn, k: u8) -> Self {
        let v = if k > 4 {
            f64::NAN
        } else {
            f.jet(self.v)[k as usize]
        };
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r as u32;
        let r = node(Some((T_JET, self.r, f.index() as u32, k as u64)), |d| {
            Flat::Unary {
                d,
                op: Unary::Jet(f, k),
                a,
            }
        });
        Tr::mk(v, r, true)
    }
    /// The `k`-th derivative of `exprel` at `self`: a different-route derivative, so the
    /// recording is not `exact`.
    fn exprel_k(self, k: u8) -> Self {
        let v = crate::ode::linear::exprel_all(self.v)[k as usize];
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r as u32;
        let r = node(Some((T_EXPREL, self.r, 0, k as u64)), |d| Flat::Unary {
            d,
            op: Unary::ExpRel(k),
            a,
        });
        Tr::mk(v, r, true)
    }
    fn digamma(self) -> Self {
        self.unary(
            T_DIGAMMA,
            Unary::Digamma,
            crate::stats::special::digamma(self.v),
        )
    }
    fn trigamma(self) -> Self {
        self.unary(
            T_TRIGAMMA,
            Unary::Trigamma,
            crate::stats::special::trigamma(self.v),
        )
    }
    fn tetragamma(self) -> Self {
        self.unary(
            T_TETRAGAMMA,
            Unary::Tetragamma,
            crate::stats::special::tetragamma(self.v),
        )
    }
    /// A branch on values, recorded as a check: `a cc b` must come out as it did here.
    #[track_caller]
    pub(super) fn guard_cmp(a: Self, cc: ir::Cc, b: Self) -> bool {
        let out = cc.holds(a.v, b.v);
        if a.dep || b.dep {
            let (x, y) = (a.r as u32, b.r as u32);
            let key = (T_CMP + cc as u8, a.r, b.r, 0);
            let c = node(Some(key), |d| Flat::Cmp { d, cc, a: x, b: y });
            // Its value too: a body recorded later may share this condition (hash-consing)
            // and read it as a capture.
            Self::mk(out as u8 as f64, c, true);
            let cci = [
                ir::Cc::Lt,
                ir::Cc::Le,
                ir::Cc::Gt,
                ir::Cc::Ge,
                ir::Cc::Eq,
                ir::Cc::Ne,
            ]
            .iter()
            .position(|&x| x == cc)
            .map_or(255, |i| i as u8);
            guard_reg_k(c, out, if b.dep { f64::NAN } else { b.v }, cci, a.v);
        }
        out
    }
    pub(crate) fn reg(self) -> u32 {
        self.r
    }
    /// TEMP #2786: the recorded value, for diagnosis.
    pub(crate) fn v_dbg(self) -> f64 {
        self.v
    }
}

impl Add for Tr {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        self.bin(o, T_ADD, self.v + o.v, |d, a, b| Flat::Add { d, a, b })
    }
}
impl Sub for Tr {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self.bin(o, T_SUB, self.v - o.v, |d, a, b| Flat::Sub { d, a, b })
    }
}
impl Mul for Tr {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        self.bin(o, T_MUL, self.v * o.v, |d, a, b| Flat::Mul { d, a, b })
    }
}
impl Div for Tr {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        self.bin(o, T_DIV, self.v / o.v, |d, a, b| Flat::DivRaw { d, a, b })
    }
}
impl Neg for Tr {
    type Output = Self;
    /// `x · (−1)` is IEEE negation exactly (`DualV::mul_c`, lane by lane).
    fn neg(self) -> Self {
        self.scalar(-1.0, T_MULK, -self.v, |d, a, c| Flat::MulK { d, a, c })
    }
}

impl PkNum for Tr {
    // A recording is differentiated to any order its layouts ask for, through an order
    // gate where an iteration's accuracy depends on it: it takes the second-order paths,
    // with a first-order jet of its own and no η-Hessian payload to reproduce.
    const SECOND_ORDER: bool = true;
    const TAYLOR2_EXACT: bool = true;
    const JET_ORDER: usize = 1;
    fn is_constant(self) -> bool {
        !self.dep
    }
    /// A chunk boundary: sharing (hash-consing) restarts, so a chunk's text depends only
    /// on its own event, never on which values an earlier event happened to compute.
    fn chunk_mark() {
        REC.with(|r| {
            if let Some(r) = r.borrow_mut().as_mut() {
                r.marks.push(r.code.len());
            }
        });
    }
    #[track_caller]
    fn br_cmp(self, cc: crate::sens::num::BranchCmp, other: Self) -> bool {
        Self::guard_cmp(self, cc_of(cc), other)
    }
    #[track_caller]
    fn br_cmp_c(self, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        Self::guard_cmp(self, cc_of(cc), Self::k(c))
    }
    const RECORDS_BRANCHES: bool = true;
    #[track_caller]
    fn br_shift_cmp(self, base: f64, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        // `base + self`, recorded as the value path computes it, then the comparison guarded.
        let s = Self::k(base) + self;
        Self::guard_cmp(s, cc_of(cc), Self::k(c))
    }
    #[track_caller]
    fn br_abs_lt(self, tol: f64) -> bool {
        let out = self.v.abs() < tol;
        if self.dep {
            let a = self.abs();
            Self::guard_cmp(a, ir::Cc::Lt, Self::k(tol));
        }
        out
    }
    #[track_caller]
    fn br_absdiff_lt(self, other: Self, tol: f64) -> bool {
        let out = (self.v - other.v).abs() < tol;
        if self.dep || other.dep {
            // The difference and its magnitude on the value lane, as `f64` computes them.
            let d = self - other;
            let a = d.abs();
            let t = Self::k(tol);
            let (x, y) = (a.r as u32, t.r as u32);
            let c = node(None, |dd| Flat::Cmp {
                d: dd,
                cc: ir::Cc::Lt,
                a: x,
                b: y,
            });
            guard_reg(c, out);
        }
        out
    }
    #[track_caller]
    fn br_is_nan(self) -> bool {
        let out = self.v.is_nan();
        if self.dep {
            let a = self.r as u32;
            // `Present`: 1 unless NaN.
            let p = node(None, |d| Flat::Present { d, a });
            guard_reg(p, !out);
        }
        out
    }
    fn from_f64(x: f64) -> Self {
        Self::k(x)
    }
    #[track_caller]
    fn var(x: f64, _dim: usize) -> Self {
        fail(format!("var at {}", std::panic::Location::caller()));
        Self::k(x)
    }
    #[track_caller]
    fn val(self) -> f64 {
        if self.dep {
            fail(format!("value read at {}", std::panic::Location::caller()));
        }
        self.v
    }
    fn exp(self) -> Self {
        self.unary(T_EXP, Unary::Exp, self.v.exp())
    }
    fn ln(self) -> Self {
        self.unary(T_LN, Unary::LnRaw, self.v.ln())
    }
    fn abs(self) -> Self {
        self.unary(T_ABS, Unary::Abs, self.v.abs())
    }
    fn pow(self, e: Self) -> Self {
        let v = self.v.powf(e.v);
        if !self.dep && !e.dep {
            return Self::k(v);
        }
        let (a, b) = (self.r as u32, e.r as u32);
        let r = node(Some((T_POW, self.r, e.r, 0)), |d| Flat::Pow { d, a, e: b });
        Tr::mk(v, r, true)
    }
    fn sqrt(self) -> Self {
        self.unary(T_SQRT, Unary::Sqrt, self.v.sqrt())
    }
    /// One primitive, not the iteration: its count of steps depends on the values.
    fn gamma_pq(self, x: Self, upper: bool) -> Self {
        self.gamma_part(x, upper, 0, 0)
    }
    fn host(f: crate::sens::num::HostFn, args: &[Self]) -> Vec<Self> {
        Self::host_call(f, args)
    }
    fn select_if(cc: crate::sens::num::BranchCmp, x: Self, y: Self, a: Self, b: Self) -> Self {
        let holds = cc.holds(x.v, y.v);
        if !x.dep && !y.dep {
            return if holds { a } else { b };
        }
        let cc = cc_of(cc);
        let (xr, yr) = (x.r as u32, y.r as u32);
        let c = node(Some((T_CMP + cc as u8, x.r, y.r, 0)), |d| Flat::Cmp {
            d,
            cc,
            a: xr,
            b: yr,
        });
        Self::select_reg(Tr::mk(holds as u8 as f64, c, true), a, b)
    }
    fn ode_segment(
        rhs: &dyn Fn(&[Self], &[Self], Self, &mut [Self]),
        u0: &[Self],
        t_span: (f64, f64),
        params: &[Self],
        saveat: &[f64],
        interp_at: &[f64],
        opts: &crate::ode::OdeSolverOptions,
        auto: &mut dyn std::any::Any,
    ) -> Option<(
        Vec<crate::ode::SolPointG<Self>>,
        Vec<crate::ode::SolPointG<Self>>,
    )> {
        let auto = auto.downcast_mut::<crate::ode::solver::OdeAutoSwitchState>()?;
        ode_loop::record_segment(rhs, u0, t_span, params, saveat, interp_at, opts, auto)
    }
    fn span_hint(t0: Self, t1: Self) {
        SPAN.with(|s| s.set(Some((t0.r, t0.v, t0.dep, t1.r, t1.v, t1.dep))));
    }
    fn take_span() -> Option<(Self, Self)> {
        SPAN.with(|s| s.take()).map(|(r0, v0, d0, r1, v1, d1)| {
            (
                Tr {
                    v: v0,
                    r: r0,
                    dep: d0,
                },
                Tr {
                    v: v1,
                    r: r1,
                    dep: d1,
                },
            )
        })
    }
    fn span_value(self) -> f64 {
        let named = SPAN.with(|s| {
            s.get().is_some_and(|(r0, v0, _, r1, v1, _)| {
                (r0, v0) == (self.r, self.v) || (r1, v1) == (self.r, self.v)
            })
        });
        if self.dep && !named {
            // Read as any other value: the recording cannot take it.
            return self.val();
        }
        self.v
    }
    /// The value, with a guard that it comes out the same on every evaluation.
    #[track_caller]
    fn br_value(self) -> f64 {
        if self.dep {
            if debug() {
                static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
                let l = std::panic::Location::caller().to_string();
                let mut s = SEEN.lock().unwrap();
                if !s.contains(&l) {
                    eprintln!("kjit: br_value pin at {l}");
                    s.push(l);
                }
            }
            Self::guard_cmp(self, ir::Cc::Eq, Self::k(self.v));
        }
        self.v
    }
    fn seen_value(self) -> f64 {
        self.v
    }
    #[allow(clippy::type_complexity)]
    fn ss_fixed_point(
        n: usize,
        held: &[usize],
        u0: &[f64],
        maps: &[&dyn Fn(&[Self]) -> Option<Vec<Self>>],
        solve: Box<SolveFn>,
    ) -> Option<Option<Vec<Self>>> {
        Some(record_fixed(n, held, u0, maps, solve))
    }
    fn repeat(
        x0: &[Self],
        count: Self,
        body: &dyn Fn(&[Self]) -> Vec<Self>,
    ) -> Option<Option<Vec<Self>>> {
        Some(record_repeat(x0, count, body))
    }
    fn jet_unary(self, f: crate::sens::num::JetFn) -> Self {
        self.jet_k(f, 0)
    }
    fn floor_c(self) -> Self {
        self.unary(T_FLOOR, Unary::Floor, self.v.floor())
    }
    /// The value it saw: a pure scaling, so no guard.
    fn magnitude_hint(self) -> f64 {
        self.v.abs()
    }
    fn value_only(self) -> Self {
        self.unary(T_STOPGRAD, Unary::StopGrad, self.v)
    }
    fn order_gate(k: u8, skip: Self, take: Self) -> Self {
        Tr::gate_node(k, skip, take)
    }
    fn same_node(&self, other: &Self) -> bool {
        self.r == other.r && self.dep == other.dep && self.v.to_bits() == other.v.to_bits()
    }
    fn sin(self) -> Self {
        Tr::sin(self)
    }
    fn note_linear_decline() {
        rec(|r| r.declined += 1);
    }
    /// One primitive with its derivatives (`linear::exprel_derivs`), not the lift.
    fn exprel(self, _orders: usize) -> Self {
        self.exprel_k(0)
    }
    fn affine_parts(
        rhs: &dyn Fn(&[Self], &[Self], Self, &mut [Self]),
        params: &[Self],
        t: f64,
        a: &mut [Self],
        b: &mut [Self],
    ) -> bool {
        affine_parts(rhs, params, t, a, b)
    }
    fn inv_logit(self) -> Self {
        self.unary(T_INVLOGIT, Unary::InvLogit, 1.0 / (1.0 + (-self.v).exp()))
    }
    fn logit(self) -> Self {
        self.unary(T_LOGIT, Unary::Logit, (self.v / (1.0 - self.v)).ln())
    }
    fn cos(self) -> Self {
        self.unary(T_COS, Unary::Cos, self.v.cos())
    }
    fn acos(self) -> Self {
        self.unary(T_ACOS, Unary::Acos, self.v.acos())
    }
    fn ln_gamma(self) -> Self {
        self.unary(
            T_LNGAMMA,
            Unary::LnGamma,
            crate::stats::special::ln_gamma(self.v),
        )
    }
    #[track_caller]
    /// `DualV::guard_floor`: `!(v ≥ lo)` (a `NaN` included) is the constant `lo`. The test
    /// is a value branch, so it is recorded as a guard and the taken arm is kept.
    fn guard_floor(self, lo: f64) -> Self {
        let keep = Self::guard_cmp(self, ir::Cc::Ge, Self::k(lo));
        if keep {
            self
        } else {
            Self::k(lo)
        }
    }
    #[track_caller]
    fn jets_finite(self) -> bool {
        if self.dep {
            fail(format!("jets_finite at {}", std::panic::Location::caller()));
        }
        true
    }
    #[track_caller]
    fn jet_max_abs(self) -> f64 {
        if self.dep {
            fail(format!("jet_max_abs at {}", std::panic::Location::caller()));
        }
        0.0
    }
    #[track_caller]
    fn lift2(self, v: f64, _d1: f64, _d2: f64) -> Self {
        if self.dep {
            fail(format!("lift2 at {}", std::panic::Location::caller()));
        }
        Self::k(v)
    }
    #[track_caller]
    fn lift3(self, v: f64, _d1: f64, _d2: f64, _d3: f64) -> Self {
        if self.dep {
            fail(format!("lift3 at {}", std::panic::Location::caller()));
        }
        Self::k(v)
    }
    #[track_caller]
    fn lift4(self, v: f64, _d1: f64, _d2: f64, _d3: f64, _d4: f64) -> Self {
        if self.dep {
            fail(format!("lift4 at {}", std::panic::Location::caller()));
        }
        Self::k(v)
    }
}

impl crate::sens::num::vm::VmArith for Tr {
    fn add_in(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
    fn sub_in(&mut self, rhs: &Self) {
        *self = *self - *rhs;
    }
    fn rsub_in(&mut self, rhs: &Self) {
        *self = *rhs - *self;
    }
    fn mul_in(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
    fn div_in(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
    fn add_scalar_in(&mut self, c: f64) {
        *self = self.scalar(c, T_ADDK, self.v + c, |d, a, c| Flat::AddK { d, a, c });
    }
    fn sub_scalar_in(&mut self, c: f64) {
        *self = self.scalar(c, T_SUBK, self.v - c, |d, a, c| Flat::SubK { d, a, c });
    }
    fn rsub_scalar_in(&mut self, c: f64) {
        *self = self.scalar(c, T_RSUBK, c - self.v, |d, a, c| Flat::RSubK { d, a, c });
    }
    fn mul_scalar_in(&mut self, c: f64) {
        *self = self.scalar(c, T_MULK, self.v * c, |d, a, c| Flat::MulK { d, a, c });
    }
    fn div_scalar_in(&mut self, c: f64) {
        let v = if c.abs() < 1e-30 { 0.0 } else { self.v / c };
        *self = self.scalar(c, T_DIVK, v, |d, a, c| Flat::DivK { d, a, c });
    }
    fn same_bits(&self, other: &Self) -> bool {
        self.r == other.r && self.dep == other.dep && self.v.to_bits() == other.v.to_bits()
    }
}

/// Whether this thread is recording ([`begin`] … [`end`]).
pub(crate) fn recording_open() -> bool {
    REC.with(|r| r.borrow().is_some())
}

/// Open a recording on this thread.
pub(crate) fn begin() {
    REC.with(|r| *r.borrow_mut() = Some(Rec::default()));
    SPAN.with(|s| s.set(None));
    GLOG.with(|g| g.borrow_mut().clear());
}

/// A finished recording: the program and the guard's register.
pub(crate) struct Recording {
    code: Vec<Flat>,
    n_regs: u32,
    outs: Vec<u32>,
    guard: Option<u32>,
    pool: Vec<f64>,
    vals: Vec<f64>,
    marks: Vec<usize>,
    eta_regs: Vec<(u32, usize)>,
    theta_regs: Vec<(u32, usize)>,
    loops: Vec<CallKind>,
    ode: bool,
    /// The θ slots carry every η lane (a steady-state call's sub-plan, whose inputs are the
    /// outer recording's values): no lane of theirs is known to be zero.
    dense: bool,
    /// The linear segments it integrated instead (`Rec::declined`).
    declined: usize,
}

/// Close the recording with `outs` as the outputs, or the reason it cannot be compiled.
pub(crate) fn end(outs: Option<Vec<u32>>) -> Result<Recording, String> {
    let r = REC
        .with(|r| r.borrow_mut().take())
        .expect("a kernel_jit recording is open");
    if let Some(why) = r.failed {
        return Err(why);
    }
    let outs = outs.ok_or_else(|| "the recorder returned None".to_string())?;
    let (code, marks) = live_code(r.code, r.marks, r.n_regs as usize, &outs, r.guard);
    Ok(Recording {
        code,
        n_regs: r.n_regs,
        outs,
        guard: r.guard,
        pool: r.pool,
        vals: r.vals,
        marks,
        eta_regs: r.eta_regs,
        theta_regs: r.theta_regs,
        loops: r.loops,
        ode: r.ode,
        dense: false,
        declined: r.declined,
    })
}

/// `code` without the instructions nothing reads — not an output, not the guard, nor an
/// instruction that is itself kept — with the chunk marks moved to match. Every instruction is
/// pure, so dropping one changes no result; what it drops is mostly a call `affine_parts`
/// made only to differentiate. The inputs' loads stay: the plan lays slots out from them.
fn live_code(
    code: Vec<Flat>,
    marks: Vec<usize>,
    n_regs: usize,
    outs: &[u32],
    guard: Option<u32>,
) -> (Vec<Flat>, Vec<usize>) {
    let mut live = vec![false; n_regs];
    for &o in outs.iter().chain(guard.iter()) {
        live[o as usize] = true;
    }
    let mut keep = vec![false; code.len()];
    for (i, f) in code.iter().enumerate().rev() {
        let k = match def(f) {
            Some(d) => {
                live[d as usize]
                    || matches!(
                        f,
                        Flat::LoadState { .. } | Flat::LoadParam { .. } | Flat::LoopOut { .. }
                    )
            }
            None => true,
        };
        if k {
            keep[i] = true;
            for u in uses(f).into_iter().flatten() {
                live[u as usize] = true;
            }
        }
    }
    // A mark at position `m` moves to the number of kept instructions before it.
    let mut before = vec![0usize; code.len() + 1];
    for i in 0..code.len() {
        before[i + 1] = before[i] + keep[i] as usize;
    }
    let mut marks: Vec<usize> = marks
        .into_iter()
        .map(|m| before[m.min(code.len())])
        .collect();
    marks.dedup();
    let code = code
        .into_iter()
        .zip(keep)
        .filter_map(|(f, k)| k.then_some(f))
        .collect();
    (code, marks)
}

/// `A = ∂f/∂u` (row-major `n×n`, `n = b.len()`) and `b = f(0)` of a right-hand side affine in
/// the state, at `params` and time `t` (#2786): `b` from a call on literal zeros, `A` by
/// differentiating, one state direction at a time, the recording of a call on fresh variables
/// (which the recording then drops). Exact — no probe, so no cancellation against the forcing
/// — and checked: a column or a guard that still depends on the variables means the RHS is not
/// affine in them, and the recording fails.
pub(crate) fn affine_parts(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]),
    params: &[Tr],
    t: f64,
    a: &mut [Tr],
    b: &mut [Tr],
) -> bool {
    let n = b.len();
    rec(|r| r.ode = true);
    let zeros = vec![Tr::k(0.0); n];
    rhs(&zeros, params, Tr::k(t), b);
    let start = rec(|r| r.code.len());
    let u: Vec<Tr> = (0..n)
        .map(|_| Tr::mk(0.0, node(None, |d| Flat::LoadK { d, c: 0.0 }), true))
        .collect();
    let mut f = vec![Tr::k(0.0); n];
    rhs(&u, params, Tr::k(t), &mut f);
    let (region, vals) = rec(|r| (r.code[start..].to_vec(), r.vals.clone()));
    let h = |r: u32| {
        Tr::mk(
            vals.get(r as usize).copied().unwrap_or(f64::NAN),
            r as u32,
            true,
        )
    };
    for j in 0..n {
        let mut tan: HashMap<u32, Tr, FxBuild> = HashMap::default();
        tan.insert(u[j].r as u32, Tr::k(1.0));
        for g in &region {
            let Some(d) = def(g) else { continue };
            if let Some(t) = tangent(g, &h, &tan) {
                tan.insert(d, t);
            }
        }
        for i in 0..n {
            a[i * n + j] = if f[i].dep {
                tan.get(&(f[i].r as u32))
                    .copied()
                    .unwrap_or_else(|| Tr::k(0.0))
            } else {
                Tr::k(0.0)
            };
        }
    }
    // Affine: no column, and no guard, reads the variables.
    let (code, guard, n_regs) = rec(|r| (r.code.clone(), r.guard, r.n_regs as usize));
    let mut on_u = vec![false; n_regs];
    for x in &u {
        on_u[x.r as usize] = true;
    }
    for g in &code[start..] {
        if let Some(d) = def(g) {
            if uses(g).into_iter().flatten().any(|v| on_u[v as usize]) {
                on_u[d as usize] = true;
            }
        }
    }
    if a.iter().any(|x| x.dep && on_u[x.r as usize]) || guard.is_some_and(|g| on_u[g as usize]) {
        fail("the right-hand side is not affine in the state".into());
        return false;
    }
    true
}

/// The tangent of instruction `g`'s result, given its operands' (`tan`, absent = zero), or
/// `None` when it is zero. `h` rebuilds a register as a recorded value.
fn tangent(g: &Flat, h: &impl Fn(u32) -> Tr, tan: &HashMap<u32, Tr, FxBuild>) -> Option<Tr> {
    let t = |r: u32| tan.get(&r).copied();
    let add = |x: Option<Tr>, y: Option<Tr>| match (x, y) {
        (Some(x), Some(y)) => Some(x + y),
        (x, None) => x,
        (None, y) => y,
    };
    let neg = |x: Option<Tr>| x.map(|x| -x);
    match *g {
        Flat::LoadState { .. }
        | Flat::LoadParam { .. }
        | Flat::LoadSeed { .. }
        | Flat::LoadK { .. } => None,
        Flat::Add { a, b, .. } => add(t(a), t(b)),
        Flat::Sub { a, b, .. } => add(t(a), neg(t(b))),
        Flat::RSub { a, b, .. } => add(t(b), neg(t(a))),
        Flat::Mul { a, b, .. } => add(t(a).map(|x| x * h(b)), t(b).map(|y| h(a) * y)),
        // `q = a/b`: `(ȧ − q·ḃ)/b`, through the same division, so its guard zeroes both.
        Flat::Div { d, a, b } | Flat::DivRaw { d, a, b } => {
            let num = add(t(a), t(b).map(|y| -(h(d) * y)))?;
            Some(if matches!(g, Flat::Div { .. }) {
                Tr::vm_div(num, h(b))
            } else {
                num / h(b)
            })
        }
        Flat::AddK { a, .. } | Flat::SubK { a, .. } => t(a),
        Flat::RSubK { a, .. } => neg(t(a)),
        Flat::MulK { a, c, .. } => t(a).map(|x| x * Tr::k(c)),
        Flat::DivK { a, c, .. } => t(a).map(|x| {
            let mut y = x;
            crate::sens::num::vm::VmArith::div_scalar_in(&mut y, c);
            y
        }),
        Flat::Pow { d, a, e } => {
            let base = t(a).map(|x| h(e) * h(a).pow(h(e) - Tr::k(1.0)) * x);
            let expo = t(e).map(|y| {
                // `aᵉ·ln a`, defined for `a > 0`; `0^e` is flat in `e` there.
                if h(a).br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0) {
                    h(d) * h(a).ln() * y
                } else {
                    Tr::k(0.0)
                }
            });
            add(base, expo)
        }
        Flat::GammaPq {
            a, x, upper, i, j, ..
        } => {
            let p = |di: u8, dj: u8| h(a).gamma_part(h(x), upper, i + di, j + dj);
            add(t(a).map(|y| p(1, 0) * y), t(x).map(|y| p(0, 1) * y))
        }
        Flat::Unary { d, op, a } => {
            let x = t(a)?;
            let (q, v) = (h(d), h(a));
            Some(match op {
                Unary::Exp => q * x,
                Unary::LnRaw => x / v,
                // The interpreter's floor: flat below it.
                Unary::Ln => {
                    if v.br_cmp_c(crate::sens::num::BranchCmp::Ge, 1e-30) {
                        x / v
                    } else {
                        return None;
                    }
                }
                Unary::Sqrt => x / (q * Tr::k(2.0)),
                Unary::InvLogit => q * (Tr::k(1.0) - q) * x,
                Unary::Logit => x / (v * (Tr::k(1.0) - v)),
                Unary::Cos => -(v.sin() * x),
                Unary::Sin => v.cos() * x,
                Unary::Acos => -(x / (Tr::k(1.0) - v * v).sqrt()),
                Unary::LnGamma => v.digamma() * x,
                Unary::Digamma => v.trigamma() * x,
                Unary::Trigamma => v.tetragamma() * x,
                Unary::Abs => {
                    if v.br_cmp_c(crate::sens::num::BranchCmp::Ge, 0.0) {
                        x
                    } else {
                        -x
                    }
                }
                Unary::ExpRel(k) => v.exprel_k(k + 1) * x,
                Unary::Jet(f, k) => v.jet_k(f, k + 1) * x,
                Unary::Tetragamma => {
                    fail(format!("no tangent rule for {op:?}"));
                    return None;
                }
                Unary::Floor | Unary::Ceil | Unary::Round | Unary::StopGrad => return None,
            })
        }
        Flat::Cmp { .. }
        | Flat::And { .. }
        | Flat::Or { .. }
        | Flat::Not { .. }
        | Flat::Present { .. } => None,
        Flat::HostArg { .. } | Flat::Host { .. } | Flat::HostOut { .. } => None,
        Flat::LoopArg { .. } | Flat::Loop { .. } | Flat::LoopOut { .. } => None,
        Flat::Select { c, a, b, .. } => match (t(a), t(b)) {
            (None, None) => None,
            (x, y) => {
                let z = Tr::k(0.0);
                Some(Tr::select_reg(h(c), x.unwrap_or(z), y.unwrap_or(z)))
            }
        },
        Flat::Gate { a, b, k, .. } => match (t(a), t(b)) {
            (None, None) => None,
            (x, y) => {
                let z = Tr::k(0.0);
                Some(Tr::gate_node(k, x.unwrap_or(z), y.unwrap_or(z)))
            }
        },
        Flat::Deriv { .. } | Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. } => None,
    }
}

impl Recording {
    /// `FERX_KJIT_DEBUG`: replay the program on values, with the lowering's semantics (the
    /// guarded division, the `ln` floor, the scalar forms), from the recorded inputs; report
    /// the first register whose replayed value differs from the one the recording computed.
    pub(crate) fn replay_check(&self) {
        let n = self.n_regs as usize;
        if self.vals.len() < n {
            return;
        }
        let mut r = vec![f64::NAN; n];
        for f in &self.code {
            let Some(d) = def(f) else { continue };
            let v = |x: u32| r[x as usize];
            let val = match *f {
                Flat::LoadState { d, .. } | Flat::LoadParam { d, .. } => self.vals[d as usize],
                Flat::LoadSeed { k, .. } => self.pool[k as usize],
                Flat::LoadK { c, .. } => c,
                Flat::Add { a, b, .. } => v(a) + v(b),
                Flat::Sub { a, b, .. } => v(a) - v(b),
                Flat::RSub { a, b, .. } => v(b) - v(a),
                Flat::Mul { a, b, .. } => v(a) * v(b),
                Flat::Div { a, b, .. } => {
                    if v(b).abs() < 1e-30 {
                        0.0
                    } else {
                        v(a) / v(b)
                    }
                }
                Flat::DivRaw { a, b, .. } => v(a) / v(b),
                Flat::AddK { a, c, .. } => v(a) + c,
                Flat::SubK { a, c, .. } => v(a) - c,
                Flat::RSubK { a, c, .. } => c - v(a),
                Flat::MulK { a, c, .. } => v(a) * c,
                Flat::DivK { a, c, .. } => {
                    if c.abs() < 1e-30 {
                        0.0
                    } else {
                        v(a) / c
                    }
                }
                Flat::Pow { a, e, .. } => v(a).powf(v(e)),
                Flat::Select { c, a, b, .. } => {
                    if v(c) != 0.0 {
                        v(a)
                    } else {
                        v(b)
                    }
                }
                Flat::Gate { b, .. } => v(b),
                Flat::GammaPq {
                    a, x, upper, i, j, ..
                } => crate::stats::special::gamma_pq_partial(v(a), v(x), upper, i, j),
                Flat::Unary { a, op, .. } => match op {
                    Unary::Exp => v(a).exp(),
                    Unary::Ln => (if v(a) >= 1e-30 { v(a) } else { 1e-30 }).ln(),
                    Unary::Abs => v(a).abs(),
                    Unary::Sqrt => v(a).sqrt(),
                    Unary::InvLogit => 1.0 / (1.0 + (-v(a)).exp()),
                    Unary::Logit => (v(a) / (1.0 - v(a))).ln(),
                    Unary::Cos => v(a).cos(),
                    Unary::Acos => v(a).acos(),
                    Unary::LnRaw => v(a).ln(),
                    Unary::Sin => v(a).sin(),
                    Unary::Digamma => crate::stats::special::digamma(v(a)),
                    Unary::Trigamma => crate::stats::special::trigamma(v(a)),
                    Unary::Tetragamma => crate::stats::special::tetragamma(v(a)),
                    Unary::ExpRel(k) => crate::ode::linear::exprel_all(v(a))[k as usize],
                    Unary::Jet(f, k) => {
                        if k > 4 {
                            f64::NAN
                        } else {
                            f.jet(v(a))[k as usize]
                        }
                    }
                    Unary::StopGrad => v(a),
                    Unary::LnGamma => crate::stats::special::ln_gamma(v(a)),
                    Unary::Floor => v(a).floor(),
                    Unary::Ceil => v(a).ceil(),
                    Unary::Round => v(a).round(),
                },
                // conditions are checked by the guard, not compared here
                Flat::Cmp { .. }
                | Flat::And { .. }
                | Flat::Or { .. }
                | Flat::Not { .. }
                | Flat::Present { .. }
                | Flat::HostOut { .. }
                | Flat::LoopOut { .. } => {
                    r[d as usize] = self.vals.get(d as usize).copied().unwrap_or(f64::NAN);
                    continue;
                }
                _ => continue,
            };
            r[d as usize] = val;
            let want = self.vals[d as usize];
            if val.to_bits() != want.to_bits() && !(val.is_nan() && want.is_nan()) {
                eprintln!("kjit-replay: register {d} {f:?}: replay {val:e}, recorded {want:e}");
                for g in &self.code {
                    for u in uses(g).into_iter().flatten() {
                        if u == d {
                            continue;
                        }
                    }
                }
                return;
            }
        }
    }
}

/// The folding pass, with `known(arg, off)` saying which loaded lanes are fixed constants.
fn fold_with(src: &Func, known: impl Fn(u8, u32) -> Option<f64>) -> Func {
    let mut out = Func::new(src.n_args);
    let mut map: Vec<ir::V> = Vec::with_capacity(src.insts.len());
    let mut konst: Vec<Option<f64>> = Vec::new();
    let mut zero: Option<ir::V> = None;
    let mut one: Option<ir::V> = None;
    let push = |out: &mut Func, konst: &mut Vec<Option<f64>>, i: Inst| -> ir::V {
        out.insts.push(i);
        out.live.push(None);
        konst.push(match i {
            Inst::Const(b) => Some(f64::from_bits(b)),
            _ => None,
        });
        (out.insts.len() - 1) as ir::V
    };
    let mut kc = |out: &mut Func, konst: &mut Vec<Option<f64>>, c: f64| -> ir::V {
        let slot = if c == 0.0 { &mut zero } else { &mut one };
        if let Some(v) = *slot {
            return v;
        }
        let v = push(out, konst, Inst::Const(c.to_bits()));
        *slot = Some(v);
        v
    };
    for &i in &src.insts {
        let m = |x: ir::V| map[x as usize];
        let is = |konst: &Vec<Option<f64>>, x: ir::V, c: f64| konst[x as usize] == Some(c);
        let v = match i {
            Inst::Load { arg, off } => match known(arg, off) {
                Some(c) => kc(&mut out, &mut konst, c),
                None => push(&mut out, &mut konst, i),
            },
            Inst::Mul(a, b) => {
                let (a, b) = (m(a), m(b));
                if is(&konst, a, 0.0) || is(&konst, b, 0.0) {
                    kc(&mut out, &mut konst, 0.0)
                } else if is(&konst, a, 1.0) {
                    b
                } else if is(&konst, b, 1.0) {
                    a
                } else {
                    push(&mut out, &mut konst, Inst::Mul(a, b))
                }
            }
            Inst::Add(a, b) => {
                let (a, b) = (m(a), m(b));
                if is(&konst, a, 0.0) {
                    b
                } else if is(&konst, b, 0.0) {
                    a
                } else {
                    push(&mut out, &mut konst, Inst::Add(a, b))
                }
            }
            Inst::Sub(a, b) => {
                let (a, b) = (m(a), m(b));
                if is(&konst, b, 0.0) {
                    a
                } else if is(&konst, a, 0.0) {
                    push(&mut out, &mut konst, Inst::Neg(b))
                } else {
                    push(&mut out, &mut konst, Inst::Sub(a, b))
                }
            }
            Inst::Div(a, b) => {
                let (a, b) = (m(a), m(b));
                if is(&konst, a, 0.0) {
                    kc(&mut out, &mut konst, 0.0)
                } else {
                    push(&mut out, &mut konst, Inst::Div(a, b))
                }
            }
            Inst::Neg(a) => {
                let a = m(a);
                if is(&konst, a, 0.0) {
                    a
                } else {
                    push(&mut out, &mut konst, Inst::Neg(a))
                }
            }
            Inst::Select(c, a, b) => {
                let (c, a, b) = (m(c), m(a), m(b));
                if a == b || (konst[a as usize].is_some() && konst[a as usize] == konst[b as usize])
                {
                    a
                } else {
                    push(&mut out, &mut konst, Inst::Select(c, a, b))
                }
            }
            Inst::Abs(a) => push(&mut out, &mut konst, Inst::Abs(m(a))),
            Inst::Floor(a) => push(&mut out, &mut konst, Inst::Floor(m(a))),
            Inst::Ceil(a) => push(&mut out, &mut konst, Inst::Ceil(m(a))),
            Inst::Call1(s, a) => push(&mut out, &mut konst, Inst::Call1(s, m(a))),
            Inst::Powf(a, b) => push(&mut out, &mut konst, Inst::Powf(m(a), m(b))),
            Inst::HostArg(a) => push(&mut out, &mut konst, Inst::HostArg(m(a))),
            Inst::Host(_) | Inst::HostOut(_) => push(&mut out, &mut konst, i),
            Inst::StopGrad(a) => push(&mut out, &mut konst, Inst::StopGrad(m(a))),
            Inst::Gamma { upper, i, j, a, x } => push(
                &mut out,
                &mut konst,
                Inst::Gamma {
                    upper,
                    i,
                    j,
                    a: m(a),
                    x: m(x),
                },
            ),
            Inst::Cmp(cc, a, b) => push(&mut out, &mut konst, Inst::Cmp(cc, m(a), m(b))),
            Inst::And(a, b) => push(&mut out, &mut konst, Inst::And(m(a), m(b))),
            Inst::Or(a, b) => push(&mut out, &mut konst, Inst::Or(m(a), m(b))),
            Inst::Store { arg, off, v } => {
                push(&mut out, &mut konst, Inst::Store { arg, off, v: m(v) })
            }
            Inst::Const(_) => push(&mut out, &mut konst, i),
        };
        map.push(v);
    }
    out
}

// ── The cache: compiled recordings per subject, shared by program text ────────────────

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// A subject of a model at a width. A recording bakes the subject's data in (its times, doses,
/// covariates), so the subject is identified by that data ([`subject_fingerprint`]), not by
/// an address a later subject — a per-occasion or cut copy built and dropped mid-fit — may
/// reuse. The model is identified by its address, which outlives every fit it is used in.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SubjectKey {
    model: usize,
    /// The model's structure ([`model_fingerprint`]): another model built at a dropped one's
    /// address — by a caller that runs no `fit` between them — keys apart.
    model_fp: u64,
    subject: u64,
    /// The mixture class `MIXNUM` resolves to on this thread: a recording takes its branch.
    class: usize,
    /// The stacked `[η, κ]` layout the recording was made under (`0`: none).
    layout: u64,
    n: usize,
    n_theta: usize,
    /// What the plan computes: the predictions (`0`) or a subject term ([`Self::with_kind`]).
    kind: u8,
    /// The computation's own configuration, hashed ([`Self::with_variant`]): a likelihood
    /// kernel's objective, width and parts.
    variant: u64,
}

impl SubjectKey {
    /// The plan recorded under the subject's stacked IOV layout `lay`.
    pub(crate) fn with_layout(
        mut self,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    ) -> Self {
        self.layout = lay.map_or(0, |l| l.fingerprint());
        self
    }
    /// The same subject's plan of another computation (`1`: its non-Gaussian term).
    pub(crate) fn with_kind(mut self, kind: u8) -> Self {
        self.kind = kind;
        self
    }
    /// The plan of one configuration of the computation (`variant` hashes it).
    pub(crate) fn with_variant(mut self, variant: u64) -> Self {
        self.variant = variant;
        self
    }
    pub(crate) fn new(
        model: &crate::types::CompiledModel,
        subject: &crate::types::Subject,
        n: usize,
        n_theta: usize,
    ) -> Self {
        SubjectKey {
            model: model as *const _ as usize,
            model_fp: model_fingerprint(model),
            subject: subject_fingerprint(subject),
            class: crate::parser::model_parser::current_mixture_class(),
            layout: 0,
            n,
            n_theta,
            kind: 0,
            variant: 0,
        }
    }
    pub(crate) fn model(&self) -> usize {
        self.model
    }
}

/// The structure of `m`, hashed: its dimensions, parameter and output names,
/// structural and error models, transforms. Two models a recording would differ on differ here
/// (they declare different parameters, a different error model or transform); one rebuilt from
/// the same text is the same model, whose plans stay valid.
fn model_fingerprint(m: &crate::types::CompiledModel) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = Fx::default();
    (m.n_theta, m.n_eta, m.n_kappa, m.n_epsilon).hash(&mut h);
    m.name.hash(&mut h);
    m.theta_names.hash(&mut h);
    m.eta_names.hash(&mut h);
    m.kappa_names.hash(&mut h);
    m.indiv_param_names.hash(&mut h);
    m.output_columns.hash(&mut h);
    (m.log_transform, m.dv_pre_logged, m.ode_spec.is_some()).hash(&mut h);
    format!("{:?}|{:?}", m.pk_model, m.error_model).hash(&mut h);
    m.pk_indices.hash(&mut h);
    m.eta_map.hash(&mut h);
    m.derived_exprs.len().hash(&mut h);
    #[cfg(feature = "nn")]
    m.covariate_nns.len().hash(&mut h);
    m.mixture.is_some().hash(&mut h);
    m.analytic_readout.is_some().hash(&mut h);
    m.ruv_magnitude.is_some().hash(&mut h);
    m.residual_error_eta.hash(&mut h);
    h.finish()
}

/// Everything of `s` a recording can read, hashed: its records' times, doses, compartments,
/// occasions and covariate values (each map's values in its own iteration order — the same
/// subject hashes the same; an equal one elsewhere may hash apart, which only re-records).
fn subject_fingerprint(s: &crate::types::Subject) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = Fx::default();
    let f = |h: &mut Fx, v: &[f64]| {
        v.len().hash(h);
        for x in v {
            x.to_bits().hash(h);
        }
    };
    let maps = |h: &mut Fx, m: &[HashMap<String, f64>]| {
        m.len().hash(h);
        for c in m {
            c.len().hash(h);
            for v in c.values() {
                v.to_bits().hash(h);
            }
        }
    };
    f(&mut h, &s.obs_times);
    f(&mut h, &s.obs_raw_times);
    f(&mut h, &s.pk_only_times);
    f(&mut h, &s.pk_only_raw_times);
    f(&mut h, &s.reset_times);
    s.obs_cmts.hash(&mut h);
    s.cens.hash(&mut h);
    s.occasions.hash(&mut h);
    s.obs_rgroup.hash(&mut h);
    s.dose_occasions.hash(&mut h);
    s.reset_occasions.hash(&mut h);
    s.pk_only_occasions.hash(&mut h);
    s.fremtype.hash(&mut h);
    for (a, b) in &s.reset_period_times {
        a.to_bits().hash(&mut h);
        b.to_bits().hash(&mut h);
    }
    for t in &s.cmt_toggles {
        (t.pk_only, t.state, t.on).hash(&mut h);
    }
    s.doses.len().hash(&mut h);
    for d in &s.doses {
        for x in [d.time, d.amt, d.rate, d.duration, d.ii] {
            x.to_bits().hash(&mut h);
        }
        (d.ss, d.cmt_idx()).hash(&mut h);
        std::mem::discriminant(&d.rate_mode).hash(&mut h);
        std::mem::discriminant(&d.infusion_def).hash(&mut h);
    }
    maps(&mut h, std::slice::from_ref(&s.covariates));
    maps(&mut h, &s.dose_covariates);
    maps(&mut h, &s.obs_covariates);
    maps(&mut h, &s.pk_only_covariates);
    maps(&mut h, &s.reset_covariates);
    maps(&mut h, &s.endpoint_covariates);
    if !s.obs_records.is_empty() {
        use std::fmt::Write;
        let mut t = String::new();
        let _ = write!(t, "{:?}", s.obs_records);
        t.hash(&mut h);
    }
    s.simulated_mixnum.hash(&mut h);
    h.finish()
}

/// A subject's compiled recording.
pub(crate) type Entry = Arc<Plan>;
type Kernels = Arc<Vec<Entry>>;

fn subjects() -> &'static RwLock<HashMap<SubjectKey, Option<Kernels>>> {
    static S: OnceLock<RwLock<HashMap<SubjectKey, Option<Kernels>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Models a recording failed on for a reason of the model's (an unsupported operation, a
/// value read): every subject would fail the same way, so none tries again.
fn bad_models() -> &'static RwLock<std::collections::HashSet<usize>> {
    static B: OnceLock<RwLock<std::collections::HashSet<usize>>> = OnceLock::new();
    B.get_or_init(Default::default)
}

pub(crate) fn model_bad(model: usize) -> bool {
    bad_models()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&model)
}

pub(crate) fn mark_model_bad(model: usize) {
    bad_models()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(model);
}

/// Forget every subject and model (a new fit: the addresses may be reused).
pub(crate) fn reset() {
    if enabled() {
        GENERATION.fetch_add(1, Relaxed);
        subjects()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        bad_models()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

pub(crate) enum Lookup {
    Missing,
    Bad,
    Found(Kernels),
}

/// Bumped by [`reset`]: a thread's last-subject entry from an earlier fit is stale.
static GENERATION: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The last subject this thread looked up: the inner loop asks for the same subject
    /// many times in a row, so most lookups take no lock and hash nothing.
    static LAST: RefCell<Option<(usize, SubjectKey, Kernels)>> = const { RefCell::new(None) };
}

pub(crate) fn lookup(k: &SubjectKey) -> Lookup {
    let gen = GENERATION.load(Relaxed);
    let hit = LAST.with(|l| match &*l.borrow() {
        Some((g, key, ks)) if *g == gen && key == k => Some(ks.clone()),
        _ => None,
    });
    if let Some(ks) = hit {
        return Lookup::Found(ks);
    }
    let found = match subjects()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(k)
    {
        None => return Lookup::Missing,
        Some(None) => return Lookup::Bad,
        Some(Some(ks)) => ks.clone(),
    };
    LAST.with(|l| *l.borrow_mut() = Some((gen, k.clone(), found.clone())));
    Lookup::Found(found)
}

pub(crate) fn mark_bad(k: &SubjectKey) {
    STATS.bad.fetch_add(1, Relaxed);
    subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(k.clone(), None);
}

/// The variants a subject's cache entry keeps ([`add`]).
const MAX_VARIANTS: usize = 8;

pub(crate) fn add(k: &SubjectKey, kernel: Entry) {
    // A thread's last-subject entry for `k` no longer lists every variant.
    GENERATION.fetch_add(1, Relaxed);
    let mut s = subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut v: Vec<Entry> = match s.get(k) {
        Some(Some(ks)) => ks.as_ref().clone(),
        _ => Vec::new(),
    };
    // At most `MAX_VARIANTS`, the oldest dropped: two variants whose guards both hold at a
    // point recorded the same branches there, so which one serves does not change a bit.
    if v.len() >= MAX_VARIANTS {
        v.remove(0);
    }
    v.push(kernel);
    s.insert(k.clone(), Some(Arc::new(v)));
}

pub(crate) struct Stats {
    pub(crate) compiled: AtomicUsize,
    pub(crate) shared: AtomicUsize,
    pub(crate) bad: AtomicUsize,
    pub(crate) runs: AtomicUsize,
    pub(crate) taped: AtomicUsize,
    pub(crate) cov: AtomicUsize,
    pub(crate) jets2: AtomicUsize,
    pub(crate) spot_fail: AtomicUsize,
    pub(crate) guard_fail: AtomicUsize,
    pub(crate) compile_us: AtomicUsize,
    pub(crate) record_us: AtomicUsize,
}

pub(crate) static STATS: Stats = Stats {
    compiled: AtomicUsize::new(0),
    shared: AtomicUsize::new(0),
    bad: AtomicUsize::new(0),
    runs: AtomicUsize::new(0),
    taped: AtomicUsize::new(0),
    cov: AtomicUsize::new(0),
    jets2: AtomicUsize::new(0),
    spot_fail: AtomicUsize::new(0),
    guard_fail: AtomicUsize::new(0),
    compile_us: AtomicUsize::new(0),
    record_us: AtomicUsize::new(0),
};

/// One line of counters (`FERX_KJIT_DEBUG`), printed at the end of a fit.

pub(crate) fn report() {
    if enabled() && debug() {
        let s = &STATS;
        eprintln!(
            "kjit: compiled {} shared {} bad {} runs {} taped {} cov {} jets2 {} guard-fails {} spot-fails {} record {:.1} ms compile {:.1} ms",
            s.compiled.load(Relaxed),
            s.shared.load(Relaxed),
            s.bad.load(Relaxed),
            s.runs.load(Relaxed),
            s.taped.load(Relaxed),
            s.cov.load(Relaxed),
            s.jets2.load(Relaxed),
            s.guard_fail.load(Relaxed),
            s.spot_fail.load(Relaxed),
            s.record_us.load(Relaxed) as f64 / 1e3,
            s.compile_us.load(Relaxed) as f64 / 1e3,
        );
        let c = chunk_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        eprintln!(
            "kjit: {} distinct chunk kernels",
            c.values().map(Vec::len).sum::<usize>()
        );
    }
}

// ── Chunks: a recording cut at its event marks, compiled once per text, shared ─────────

/// `fn(base, idx, lits)`: inputs and outputs are lane blocks of the plan's buffer at `base`,
/// addressed through the call's slot table `idx` (inputs, then outputs); `lits` are the
/// chunk's own literal values.
pub(crate) type ChunkFn = unsafe extern "C" fn(*mut f64, *const u32, *const f64);
/// A chunk's reverse pass: `fn(base, adj, idx, lits, seeds, stride)` adds `out̄ᵀ·∂out/∂in`
/// into the inputs' adjoint blocks of `adj` (laid out like `base`), reading the outputs'
/// adjoints there — for each of `seeds` adjoint buffers `stride` doubles apart, the forward it
/// recomputes run once for all.
pub(crate) type ChunkRevFn =
    unsafe extern "C" fn(*mut f64, *mut f64, *const u32, *const f64, usize, usize);

pub(crate) struct ChunkKernel {
    /// The entry point: a [`ChunkFn`] or a [`ChunkRevFn`].
    f: *const u8,
    _module: cranelift_jit::JITModule,
}

// SAFETY: immutable once finalized.
unsafe impl Send for ChunkKernel {}
unsafe impl Sync for ChunkKernel {}

/// A chunk's kernel over one (layout, direction): its program run a dual operation at a time
/// ([`lanes`]), then native code compiled from the lowered IR — the quick tier (single-pass
/// register allocation), then the optimizing one. The tiers' lanes agree up to the sign of a
/// zero (the folding's exactness); only the cost moves. A kernel the fit calls a few dozen
/// times is never compiled, one it calls millions of times is optimized.
pub(crate) struct Tiered {
    f: Func,
    layout: Layout,
    n_in: usize,
    rev: bool,
    /// The current native entry point (0: interpret).
    native: std::sync::atomic::AtomicUsize,
    /// Calls so far, counted until the last tier.
    calls: AtomicUsize,
    /// The tier compiled last (0: the interpreter).
    tier: std::sync::atomic::AtomicU8,
    /// A reverse kernel's leading forward instructions ([`ir::reverse`]): the part every
    /// seed of a multi-seed call shares.
    prefix: usize,
    /// The chunk's register program and the layout's lane structure: until it is compiled,
    /// the kernel runs one dual operation at a time over them ([`lanes`]). `None` for a
    /// tangent layout ([`Layout::tan`]), which is compiled on construction.
    lane: Option<(Arc<lanes::LaneProg>, lanes::AtLayout)>,
    /// The modules every native entry point lives in (an older one may still be running).
    modules: Mutex<Vec<ChunkKernel>>,
    /// Itself, for the compile queue ([`compile_queue`]).
    me: std::sync::Weak<Tiered>,
    /// The highest tier the calls so far have earned, and whether a compile is queued.
    want: std::sync::atomic::AtomicU8,
    queued: std::sync::atomic::AtomicBool,
}

/// Queue `k` on the compile threads, started on first use: compiles run off the callers'
/// path, which keep running the current tier (every tier computes the same values, bit for
/// bit) until the native entry point is swapped in. One thread per core: a thread runs only
/// while the queue holds work, so a burst of new kernels (a fit's first iterations) compiles
/// in parallel on whichever cores are free, and an empty queue costs nothing.
fn compile_queue(k: std::sync::Weak<Tiered>) {
    type Rx = Arc<Mutex<std::sync::mpsc::Receiver<std::sync::Weak<Tiered>>>>;
    static Q: OnceLock<Mutex<std::sync::mpsc::Sender<std::sync::Weak<Tiered>>>> = OnceLock::new();
    let tx = Q.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<std::sync::Weak<Tiered>>();
        let rx: Rx = Arc::new(Mutex::new(rx));
        let n = std::thread::available_parallelism().map_or(1, |n| n.get());
        for i in 0..n {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("ferx-kjit-compile-{i}"))
                .spawn(move || loop {
                    let next = rx.lock().unwrap().recv();
                    let Ok(w) = next else { return };
                    if let Some(t) = w.upgrade() {
                        t.compile_wanted();
                    }
                })
                .expect("a compile thread starts");
        }
        Mutex::new(tx)
    });
    let _ = tx.lock().unwrap().send(k);
}

/// The tiers' promotion points, from per-IR-instruction costs measured on this crate's
/// kernels (`two_cpt_oral_ode`'s segments, `wiov`'s events; one core):
///
/// | tier | compile | run |
/// |---|---|---|
/// | dual operations ([`lanes`]) | 0 | ~2× the fixed-width duals |
/// | single-pass regalloc, no optimizer | 1.1 µs | 0.5–0.8 ns (large), 0.15 (small) |
/// | single-pass regalloc, optimizer | 2.2 µs | 0.3–0.4 ns (large), 0.15 (small) |
/// | backtracking regalloc, optimizer | 3 µs at ≤ 2k instructions, 80 µs at 40k | 0.3 (large), 0.09 (small) |
///
/// Each promotion is at the call count where the next tier's compile time is repaid by its
/// faster calls. Both scale with the kernel's size, so the counts hold for any kernel — except
/// the last tier's compile, superlinear in size (register allocation), taken only up to
/// [`TIER3_MAX_INSTS`].
const TIER1_CALLS: usize = 512;
const TIER2_CALLS: usize = 4096;

const TIER3_CALLS: usize = 1 << 16;
/// The largest kernel the backtracking allocator compiles in linear time (measured: 3 µs per
/// instruction up to 2k, 25 µs at 11k, 80 µs at 40k).
const TIER3_MAX_INSTS: usize = 4096;

thread_local! {
    /// The dual-granularity tier's registers and their adjoints.
    static LANE_REGS: RefCell<(Vec<f64>, Vec<f64>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

impl Tiered {
    fn new(
        f: Func,
        layout: Layout,
        n_in: usize,
        rev: bool,
        prefix: usize,
        lane: Option<(Arc<lanes::LaneProg>, lanes::AtLayout)>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Tiered {
            me: me.clone(),
            want: std::sync::atomic::AtomicU8::new(0),
            queued: std::sync::atomic::AtomicBool::new(false),
            prefix,
            lane,
            f,
            layout,
            n_in,
            rev,
            native: std::sync::atomic::AtomicUsize::new(0),
            calls: AtomicUsize::new(0),
            tier: std::sync::atomic::AtomicU8::new(0),
            modules: Mutex::new(Vec::new()),
        })
    }

    /// Run the kernel: `adj` is the adjoint buffer of a reverse kernel (null otherwise).
    ///
    /// # Safety
    /// Every slot in `idx` is below the plan's slot count; `base` (and `adj`) hold that many
    /// lane blocks of this kernel's layout, `lits` the chunk's literals.
    #[inline]
    unsafe fn call(&self, base: *mut f64, adj: *mut f64, idx: *const u32, lits: *const f64) {
        self.count(1);
        let p = self.native.load(std::sync::atomic::Ordering::Acquire);
        if p == 0 {
            self.interpret(base, adj, idx, lits, 1, 0);
        } else if self.rev {
            let f: ChunkRevFn = std::mem::transmute(p);
            static CHECKR: OnceLock<bool> = OnceLock::new();
            if !*CHECKR.get_or_init(|| std::env::var_os("FERX_TMP_TIERCHECK").is_some()) {
                f(base, adj, idx, lits, 1, 0);
                return;
            }
            // TEMP #2786: the interpreter's reverse on the same adjoints, bit for bit.
            let l = self.layout.lanes();
            let take = |adj: *mut f64| -> Vec<f64> {
                (0..self.n_in)
                    .flat_map(|j| {
                        let s = *idx.add(j) as usize;
                        (0..l).map(move |x| *adj.add(s * l + x))
                    })
                    .collect()
            };
            let put = |adj: *mut f64, v: &[f64]| {
                for j in 0..self.n_in {
                    let s = *idx.add(j) as usize;
                    for x in 0..l {
                        *adj.add(s * l + x) = v[j * l + x];
                    }
                }
            };
            let before = take(adj);
            f(base, adj, idx, lits, 1, 0);
            let native = take(adj);
            put(adj, &before);
            self.interpret(base, adj, idx, lits, 1, 0);
            let lane = take(adj);
            put(adj, &native);
            if let Some(k) = (0..native.len()).find(|&k| native[k].to_bits() != lane[k].to_bits()) {
                static SHOWNR: AtomicUsize = AtomicUsize::new(0);
                if SHOWNR.fetch_add(1, Relaxed) < 3 {
                    eprintln!(
                        "TIERCHECK rev in {} lane {}: native {:e} interp {:e} (before {:e}) layout {:?} lane_prog {}",
                        k / l,
                        k % l,
                        native[k],
                        lane[k],
                        before[k],
                        self.layout,
                        self.lane.is_some()
                    );
                    if let Some((prog, _)) = self.lane.as_ref() {
                        for (i, op) in prog.code().iter().enumerate() {
                            eprintln!("TIERCHECK   {i:4} {op:?}");
                        }
                    }
                }
            }
        } else {
            let f: ChunkFn = std::mem::transmute(p);
            f(base, idx, lits);
            // TEMP #2786: the lane tier on the same inputs, bit for bit.
            static CHECK: OnceLock<bool> = OnceLock::new();
            if *CHECK.get_or_init(|| std::env::var_os("FERX_TMP_TIERCHECK").is_some()) {
                if let Some((prog, _)) = self.lane.as_ref() {
                    let l = self.layout.lanes();
                    let n_out = prog.n_out();
                    let read = |base: *mut f64| -> Vec<u64> {
                        (0..n_out)
                            .flat_map(|j| {
                                let s = *idx.add(self.n_in + j) as usize;
                                (0..l).map(move |x| (*base.add(s * l + x)).to_bits())
                            })
                            .collect()
                    };
                    let native = read(base);
                    self.interpret(base, adj, idx, lits, 1, 0);
                    let lane = read(base);
                    if let Some(k) = (0..native.len()).find(|&k| native[k] != lane[k]) {
                        static SHOWN: AtomicUsize = AtomicUsize::new(0);
                        if SHOWN.fetch_add(1, Relaxed) < 3 {
                            eprintln!(
                                "TIERCHECK out {} lane {}: native {:e} lane {:e} layout {:?}",
                                k / l,
                                k % l,
                                f64::from_bits(native[k]),
                                f64::from_bits(lane[k]),
                                self.layout
                            );
                            let ins: Vec<f64> = (0..self.n_in)
                                .map(|j| *base.add(*idx.add(j) as usize * l))
                                .collect();
                            eprintln!("TIERCHECK inputs {ins:?}");
                            for (i, op) in prog.code().iter().enumerate() {
                                eprintln!("TIERCHECK   {i:4} {op:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Count `k` calls, promoting at each tier's point crossed. The count is a plain load
    /// and store, not a locked add — a lost update between threads only delays a promotion
    /// — and each tier is compiled once ([`Self::tier`]).
    #[inline]
    fn count(&self, k: usize) {
        // TEMP #2786: a fixed tier, compiled synchronously (determinism diagnosis).
        static TMP_TIER: OnceLock<Option<u8>> = OnceLock::new();
        if let Some(t) = *TMP_TIER.get_or_init(|| {
            std::env::var("FERX_TMP_TIER")
                .ok()
                .and_then(|v| v.parse().ok())
        }) {
            if t > 0 && self.tier.load(Relaxed) < t {
                self.want.fetch_max(t, Relaxed);
                self.compile_wanted();
            }
            return;
        }
        let n0 = self.calls.load(Relaxed);
        if n0 >= TIER3_CALLS {
            return;
        }
        let n1 = n0 + k;
        self.calls.store(n1, Relaxed);
        if n1 >= TIER1_CALLS {
            self.promote_to(n1);
        }
    }

    /// Compile every tier `n` calls have reached and this kernel has not.
    #[cold]
    fn promote_to(&self, n: usize) {
        let want = if n >= TIER3_CALLS && self.f.insts.len() <= TIER3_MAX_INSTS {
            3
        } else if n >= TIER2_CALLS {
            2
        } else {
            1
        };
        if want <= self.want.fetch_max(want, Relaxed) {
            return;
        }
        if !self.queued.swap(true, Relaxed) {
            compile_queue(self.me.clone());
        }
    }

    /// Compile the tier wanted now, if past the one compiled: the compile thread's step. A
    /// kernel whose calls earned two tiers while it waited is compiled once, at the higher.
    fn compile_wanted(&self) {
        loop {
            let want = self.want.load(Relaxed);
            if want <= self.tier.load(Relaxed) {
                self.queued.store(false, Relaxed);
                // A promotion between the check and the store would find `queued` set and
                // not enqueue: look once more.
                if self.want.load(Relaxed) > self.tier.load(Relaxed)
                    && !self.queued.swap(true, Relaxed)
                {
                    continue;
                }
                return;
            }
            match want {
                1 => self.promote(true, "none"),
                2 => self.promote(true, "speed"),
                _ => self.promote(false, "speed"),
            }
            self.tier.store(want, Relaxed);
        }
    }

    /// A reverse kernel over `seeds` adjoint buffers, `stride` apart: native code runs once
    /// per seed, the interpreter runs the shared forward prefix once.
    ///
    /// # Safety
    /// As [`Self::call`], for each seed's `adj` block.
    unsafe fn call_seeds(
        &self,
        base: *mut f64,
        adj: *mut f64,
        stride: usize,
        seeds: usize,
        idx: *const u32,
        lits: *const f64,
    ) {
        self.count(seeds);
        let p = self.native.load(std::sync::atomic::Ordering::Acquire);
        if p == 0 {
            self.interpret(base, adj, idx, lits, seeds, stride);
        } else {
            let f: ChunkRevFn = std::mem::transmute(p);
            static CHECKS: OnceLock<bool> = OnceLock::new();
            if !*CHECKS.get_or_init(|| std::env::var_os("FERX_TMP_TIERCHECK").is_some()) {
                f(base, adj, idx, lits, seeds, stride);
                return;
            }
            // TEMP #2786: the interpreter's multi-seed reverse on the same adjoints.
            let l = self.layout.lanes();
            let take = |adj: *mut f64| -> Vec<f64> {
                (0..seeds)
                    .flat_map(|sd| {
                        (0..self.n_in).flat_map(move |j| {
                            let s = *idx.add(j) as usize;
                            (0..l).map(move |x| *adj.add(sd * stride + s * l + x))
                        })
                    })
                    .collect()
            };
            let put = |adj: *mut f64, v: &[f64]| {
                let mut k = 0;
                for sd in 0..seeds {
                    for j in 0..self.n_in {
                        let s = *idx.add(j) as usize;
                        for x in 0..l {
                            *adj.add(sd * stride + s * l + x) = v[k];
                            k += 1;
                        }
                    }
                }
            };
            let before = take(adj);
            f(base, adj, idx, lits, seeds, stride);
            let native = take(adj);
            put(adj, &before);
            self.interpret(base, adj, idx, lits, seeds, stride);
            let lane = take(adj);
            put(adj, &before);
            interpret_func(
                &self.f,
                l,
                self.n_in,
                self.rev,
                self.prefix,
                base,
                adj,
                idx,
                lits,
                seeds,
                stride,
            );
            let func = take(adj);
            if let Some(k) = (0..native.len()).find(|&k| native[k].to_bits() != lane[k].to_bits()) {
                eprintln!(
                    "TIERCHECK 3-way at {k}: native {:e} lane {:e} func {:e}",
                    native[k], lane[k], func[k]
                );
            }
            put(adj, &native);
            if let Some(k) = (0..native.len()).find(|&k| native[k].to_bits() != lane[k].to_bits()) {
                static SHOWNS: AtomicUsize = AtomicUsize::new(0);
                if SHOWNS.fetch_add(1, Relaxed) < 3 {
                    eprintln!(
                        "TIERCHECK seeds {seeds} at {k}: native {:e} interp {:e} (before {:e}) layout {:?} lane_prog {}",
                        native[k],
                        lane[k],
                        before[k],
                        self.layout,
                        self.lane.is_some()
                    );
                    if let Some((prog, _)) = self.lane.as_ref() {
                        for (i, op) in prog.code().iter().enumerate() {
                            eprintln!("TIERCHECK   {i:4} {op:?}");
                        }
                    }
                }
            }
        }
    }

    /// Compile the next tier and switch to it; a failed compile keeps the current one.
    fn promote(&self, quick: bool, opt: &str) {
        let t = std::time::Instant::now();
        let Some(k) = lower_chunk_opt(
            &self.f,
            self.layout,
            self.n_in,
            self.rev,
            self.prefix,
            quick,
            opt,
        ) else {
            return;
        };
        let p = k.f as usize;
        self.modules.lock().unwrap().push(k);
        self.native.store(p, std::sync::atomic::Ordering::Release);
        STATS.compiled.fetch_add(1, Relaxed);
        STATS
            .compile_us
            .fetch_add(t.elapsed().as_micros() as usize, Relaxed);
        if debug() {
            eprintln!(
                "kjit-compile ir {} lanes {} m {} rev {} quick {} {opt} {:.1} ms",
                self.f.insts.len(),
                self.layout.lanes(),
                self.layout.m,
                self.rev,
                quick,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// The chunk's program run a dual operation at a time ([`lanes`]): the cold tier. A
    /// reverse's seeds share its forward.
    unsafe fn interpret(
        &self,
        base: *mut f64,
        adj: *mut f64,
        idx: *const u32,
        lits: *const f64,
        seeds: usize,
        stride: usize,
    ) {
        let Some((prog, at)) = self.lane.as_ref() else {
            return interpret_func(
                &self.f,
                self.layout.lanes(),
                self.n_in,
                self.rev,
                self.prefix,
                base,
                adj,
                idx,
                lits,
                seeds,
                stride,
            );
        };
        let l = at.sh.lanes;
        let need = prog.n_regs() * l;
        LANE_REGS.with(|v| {
            let mut v = v.borrow_mut();
            let (regs, ar, scratch) = &mut *v;
            if regs.len() < need {
                regs.resize(need, 0.0);
                ar.resize(need, 0.0);
            }
            if scratch.len() < 10 * l {
                scratch.resize(10 * l, 0.0);
            }
            // Only a forward has a lane program.
            debug_assert!(!self.rev);
            prog.forward(at, self.layout.wide, regs, base, idx, lits, true, scratch);
        })
    }
}

/// A chunk's program, independent of layout: compiled per (layout, direction) on first use.
pub(crate) struct ChunkProgram {
    code: Vec<Flat>,
    n_regs: usize,
    spec: Vec<(u64, u8)>,
    n_in: usize,
    compiled: Mutex<HashMap<(Layout, bool), Option<Arc<Tiered>>>>,
    /// The program for the dual-granularity tier ([`lanes`]), built on first use: per gate
    /// order ([`resolve_gates`]) when the code has a gate, else one at index 0.
    lane: [OnceLock<Option<Arc<lanes::LaneProg>>>; GATE_ORDERS],
}

/// The gate orders a variant can have: a second-order payload, a tangent block and a reverse.
const GATE_ORDERS: usize = 5;

/// `code` as the compiled variant of derivative order `g` runs it (a reverse counting one
/// more): each [`Flat::Gate`] read as the operand that order takes, and every instruction
/// nothing then reads dropped — the refinement steps the variant does not need. `None` when
/// `code` has no gate.
fn resolve_gates(code: &[Flat], n_regs: usize, g: usize) -> Option<Vec<Flat>> {
    if !code.iter().any(|f| matches!(f, Flat::Gate { .. })) {
        return None;
    }
    let mut alias: Vec<u32> = (0..n_regs).map(|r| r as u32).collect();
    let mut out: Vec<Flat> = Vec::with_capacity(code.len());
    for f in code {
        if let Flat::Gate { d, a, b, k } = *f {
            alias[d as usize] = alias[if g >= k as usize { b } else { a } as usize];
            continue;
        }
        // Registers are written once, so renaming the destination leaves it as it was.
        out.push(rename(f, &mut |r| alias[r as usize]));
    }
    // A chunk's effects are its outputs (`Deriv`) and its host calls' queue.
    let mut live = vec![false; n_regs];
    let mut keep = vec![false; out.len()];
    for (i, f) in out.iter().enumerate().rev() {
        let effect = matches!(
            f,
            Flat::Deriv { .. } | Flat::HostArg { .. } | Flat::Host { .. }
        );
        if effect || def(f).is_some_and(|d| live[d as usize]) {
            keep[i] = true;
            for a in uses(f).into_iter().flatten() {
                live[a as usize] = true;
            }
        }
    }
    Some(
        out.into_iter()
            .zip(keep)
            .filter_map(|(f, k)| k.then_some(f))
            .collect(),
    )
}

impl ChunkProgram {
    /// The kernel over `layout`, the reverse pass when `rev` (built on first use).
    fn get(&self, layout: Layout, rev: bool) -> Option<Arc<Tiered>> {
        let mut c = self.compiled.lock().unwrap();
        if let Some(k) = c.get(&(layout, rev)) {
            return k.clone();
        }
        // The steps a gate leaves to a higher order are not this variant's.
        let gate = (layout.order() + rev as usize).min(GATE_ORDERS - 1);
        let gated = resolve_gates(&self.code, self.n_regs, gate);
        let (code, slot) = match &gated {
            Some(c) => (c.as_slice(), gate),
            None => (self.code.as_slice(), 0),
        };
        // A tangent layout's kernel is the base layout's, differentiated forward along its
        // directions (`ir::tangent_packed`).
        let base = layout.base();
        let f = ir::forward(code, self.n_regs, base);
        let f = fold_chunk(&f, base, &self.spec);
        let f = if layout.tan > 0 {
            ir::tangent_packed(&f, layout.tan as usize, base.lanes())
        } else {
            f
        };
        // `ir::reverse` emits each forward instruction but a store first, once.
        let prefix = if rev {
            f.insts
                .iter()
                .filter(|i| !matches!(i, Inst::Store { .. }))
                .count()
        } else {
            0
        };
        let f = if rev { ir::reverse(&f) } else { f };
        // A forward starts on the lane program (a recording has no `if` arms, so every chunk
        // has one); a tangent layout and a reverse pass on the scalar function
        // ([`interpret_func`]), the one definition the compiled tiers lower — so every tier
        // of a kernel gives the same bits.
        let k = if layout.tan > 0 || rev {
            Some(Tiered::new(f, layout, self.n_in, rev, prefix, None))
        } else {
            self.lane[slot]
                .get_or_init(|| lanes::LaneProg::of(code, self.n_regs, self.n_in).map(Arc::new))
                .clone()
                .map(|p| {
                    let at = lanes::AtLayout::of(&p, layout, &self.spec);
                    Tiered::new(f, layout, self.n_in, rev, prefix, Some((p, at)))
                })
        };
        c.insert((layout, rev), k.clone());
        k
    }
}

thread_local! {
    /// [`interpret_func`]'s values, one per instruction.
    static FUNC_VALS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Run a chunk kernel's scalar function in Rust over the plan's buffers, as its native
/// lowering ([`lower_chunk_opt`]) does: the first tier of a kernel with no lane program (a
/// tangent layout's, [`Layout::tan`]). A reverse kernel runs its forward `prefix` once and
/// the rest once per seed, each seed's adjoint block `stride` apart. Conditions are `1.0` /
/// `0.0`.
///
/// # Safety
/// As [`Tiered::call_seeds`].
#[allow(clippy::too_many_arguments)]
unsafe fn interpret_func(
    f: &Func,
    lanes: usize,
    n_in: usize,
    rev: bool,
    prefix: usize,
    base: *mut f64,
    adj: *mut f64,
    idx: *const u32,
    lits: *const f64,
    seeds: usize,
    stride: usize,
) {
    let slot = |e: usize| *idx.add(e) as usize * lanes;
    FUNC_VALS.with(|vals| {
        let mut x = vals.borrow_mut();
        x.clear();
        x.resize(f.insts.len(), 0.0);
        let run = |x: &mut Vec<f64>, from: usize, to: usize, adjp: *mut f64| {
            for at in from..to {
                let r = |v: ir::V| x[v as usize];
                let val = match f.insts[at] {
                    Inst::Load { arg, off } => {
                        let (i, lane) = (off as usize / lanes, off as usize % lanes);
                        match arg {
                            0 => *base.add(slot(i) + lane),
                            2 if lane == 0 => *lits.add(i),
                            2 | 6 => 0.0,
                            4 => *adjp.add(slot(i) + lane),
                            7 => *adjp.add(slot(n_in + i) + lane),
                            _ => f64::NAN,
                        }
                    }
                    Inst::Const(bits) => f64::from_bits(bits),
                    Inst::Add(a, b) => r(a) + r(b),
                    Inst::Sub(a, b) => r(a) - r(b),
                    Inst::Mul(a, b) => r(a) * r(b),
                    Inst::Div(a, b) => r(a) / r(b),
                    Inst::Neg(a) => -r(a),
                    Inst::Abs(a) => r(a).abs(),
                    Inst::Floor(a) => r(a).floor(),
                    Inst::Ceil(a) => r(a).ceil(),
                    Inst::Call1(ir::Shim1::Sqrt, a) => r(a).sqrt(),
                    Inst::Call1(s, a) => (s.f())(r(a)),
                    Inst::Powf(a, b) => r(a).powf(r(b)),
                    Inst::Gamma {
                        upper,
                        i,
                        j,
                        a,
                        x: y,
                    } => (ir::gamma_shim(upper, i, j))(r(a), r(y)),
                    Inst::Cmp(cc, a, b) => f64::from(u8::from(cc.holds(r(a), r(b)))),
                    Inst::And(a, b) => f64::from(u8::from(r(a) != 0.0 && r(b) != 0.0)),
                    Inst::Or(a, b) => f64::from(u8::from(r(a) != 0.0 || r(b) != 0.0)),
                    Inst::Select(c, a, b) => {
                        if r(c) != 0.0 {
                            r(a)
                        } else {
                            r(b)
                        }
                    }
                    Inst::StopGrad(a) => r(a),
                    Inst::HostArg(a) => ir::host::arg_native(r(a)),
                    Inst::Host(h) => ir::host::call_native(h.index() as f64),
                    Inst::HostOut(k) => ir::host::out_native(k as f64),
                    Inst::Store { arg, off, v } => {
                        let (j, lane) = (off as usize / lanes, off as usize % lanes);
                        let y = r(v);
                        match arg {
                            3 if !rev => *base.add(slot(n_in + j) + lane) = y,
                            4 if rev => *adjp.add(slot(j) + lane) = y,
                            _ => {}
                        }
                        y
                    }
                };
                x[at] = val;
            }
        };
        if rev {
            run(&mut x, 0, prefix, adj);
            for sd in 0..seeds {
                run(&mut x, prefix, f.insts.len(), adj.add(sd * stride));
            }
        } else {
            run(&mut x, 0, f.insts.len(), adj);
        }
    });
}

/// A plan step's kernels at one layout.
pub(crate) enum Kern {
    Chunk(Arc<Tiered>),
    Loop(Arc<ode_loop::LoopKernels>),
    Fixed(Arc<fixed::FixedSpec>),
    Repeat(Arc<fixed::RepeatSpec>),
}

/// One step of a plan: a chunk's kernel over its slots, or an ODE segment's loop call.
enum Call {
    Chunk {
        prog: Arc<ChunkProgram>,
        idx: Box<[u32]>,
        lits: Box<[f64]>,
    },
    Loop {
        spec: CallKind,
        /// The call's inputs' slots (`[u0, captures]`) and its outputs'.
        args: Box<[u32]>,
        outs: Box<[u32]>,
    },
}

/// A subject's compiled recording: the chunk calls over one buffer of lane blocks.
/// Slots `0..n_eta` are η (one-hot seeded), then θ, then every chunk output.
pub(crate) struct Plan {
    calls: Vec<Call>,
    n_slots: usize,
    /// The inner loop's layout, served by [`Plan::run`] without a lookup.
    layout: Layout,
    lanes: usize,
    n_eta: usize,
    n_theta: usize,
    outs: Vec<u32>,
    guard: Option<u32>,
    fast: OnceLock<Option<Arc<Vec<Kern>>>>,
    resolved: Mutex<HashMap<(Layout, bool), Option<Arc<Vec<Kern>>>>>,
    /// The solver override an ODE plan was recorded under (`None` for a plan with no ODE
    /// segment): another phase of the fit (tighter tolerances, another method) must not run it.
    ode_override: Option<Option<Option<crate::ode::solver::OdeSolverOverride>>>,
    /// The exact linear path's segments its recording integrated instead: reported to the
    /// solver statistics on every run, as the dual sweeps reported theirs.
    declined: usize,
}

/// Distinct chunk programs, by structure (layout-independent): per hash, the programs with it.
type ChunkCache = HashMap<u64, Vec<Arc<ChunkProgram>>, FxBuild>;

/// Distinct chunk programs, by structure (layout-independent).
fn chunk_cache() -> &'static Mutex<ChunkCache> {
    static C: OnceLock<Mutex<ChunkCache>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// The chunk program of `body` (no input loads): `ins` are its input registers, in order;
/// `outs` the registers it exports, in order; `pool` the recording's literals and `spec` the
/// inputs' structure. Shared by text ([`chunk_cache`]); returns the program and this chunk's
/// literal values.
fn build_chunk(
    body: &[Flat],
    ins: &[u32],
    outs: &[u32],
    pool: &[f64],
    spec: Vec<(u64, u8)>,
    cache: &mut ChunkCache,
) -> (Arc<ChunkProgram>, Vec<f64>) {
    let mut local: HashMap<u32, u32, FxBuild> = HashMap::default();
    for (i, &r) in ins.iter().enumerate() {
        local.insert(r, i as u32);
    }
    let mut next = ins.len() as u32;
    let mut code: Vec<Flat> = (0..ins.len())
        .map(|i| Flat::LoadState {
            d: i as u32,
            i: i as u32,
        })
        .collect();
    let mut lits: Vec<f64> = Vec::new();
    for f in body {
        let d = def(f);
        if let Some(d) = d {
            local.insert(d, next);
            next += 1;
        }
        let g = match *f {
            Flat::LoadSeed { k, .. } => {
                lits.push(pool[k as usize]);
                Flat::LoadSeed {
                    d: local[&d.unwrap()],
                    k: (lits.len() - 1) as u32,
                }
            }
            _ => rename(f, &mut |r| local[&r]),
        };
        code.push(g);
    }
    for (j, &o) in outs.iter().enumerate() {
        code.push(Flat::Deriv {
            state: j as u32,
            a: local[&o],
        });
    }
    // Keyed by the program's structure, a hit confirmed instruction by instruction.
    let key = {
        use std::hash::{Hash, Hasher};
        let mut h = Fx::default();
        spec.hash(&mut h);
        code.hash(&mut h);
        h.finish()
    };
    let hit = cache.get(&key).and_then(|v| {
        v.iter()
            .find(|p| {
                p.spec == spec
                    && p.code.len() == code.len()
                    && p.code.iter().zip(&code).all(|(a, b)| a.same(b))
            })
            .cloned()
    });
    let prog = match hit {
        Some(p) => {
            STATS.shared.fetch_add(1, Relaxed);
            p.clone()
        }
        None => {
            if std::env::var("FERX_KJIT_DUMP").is_ok() {
                eprintln!("kjit-chunk spec={spec:?} lits={lits:?}");
                for (i, f) in code.iter().enumerate() {
                    eprintln!("kjit-chunk   {i:4} {f:?}");
                }
            }
            let p = Arc::new(ChunkProgram {
                code,
                n_regs: next as usize,
                spec,
                n_in: ins.len(),
                compiled: Mutex::new(HashMap::new()),
                lane: Default::default(),
            });
            cache.entry(key).or_default().push(p.clone());
            p
        }
    };
    (prog, lits)
}

/// The registers an instruction reads.
fn uses(f: &Flat) -> [Option<u32>; 3] {
    match *f {
        Flat::Add { a, b, .. }
        | Flat::Sub { a, b, .. }
        | Flat::RSub { a, b, .. }
        | Flat::Mul { a, b, .. }
        | Flat::Div { a, b, .. }
        | Flat::DivRaw { a, b, .. }
        | Flat::Cmp { a, b, .. }
        | Flat::And { a, b, .. }
        | Flat::Or { a, b, .. } => [Some(a), Some(b), None],
        Flat::Pow { a, e, .. } => [Some(a), Some(e), None],
        Flat::GammaPq { a, x, .. } => [Some(a), Some(x), None],
        Flat::Select { c, a, b, .. } => [Some(c), Some(a), Some(b)],
        Flat::Gate { a, b, .. } => [Some(a), Some(b), None],
        Flat::AddK { a, .. }
        | Flat::SubK { a, .. }
        | Flat::RSubK { a, .. }
        | Flat::MulK { a, .. }
        | Flat::DivK { a, .. }
        | Flat::Unary { a, .. }
        | Flat::Not { a, .. }
        | Flat::Present { a, .. }
        | Flat::HostArg { a }
        | Flat::LoopArg { a }
        | Flat::Deriv { a, .. } => [Some(a), None, None],
        Flat::Host { .. } | Flat::HostOut { .. } | Flat::Loop { .. } | Flat::LoopOut { .. } => {
            [None, None, None]
        }
        Flat::LoadState { .. }
        | Flat::LoadParam { .. }
        | Flat::LoadSeed { .. }
        | Flat::LoadK { .. } => [None, None, None],
        Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. } => {
            unreachable!("a kernel recording has no control flow")
        }
    }
}

/// The register an instruction writes.
fn def(f: &Flat) -> Option<u32> {
    match *f {
        Flat::LoadState { d, .. }
        | Flat::LoadParam { d, .. }
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
        | Flat::LoopOut { d, .. }
        | Flat::Select { d, .. }
        | Flat::Gate { d, .. } => Some(d),
        Flat::Deriv { .. }
        | Flat::HostArg { .. }
        | Flat::Host { .. }
        | Flat::LoopArg { .. }
        | Flat::Loop { .. } => None,
        Flat::Branch { .. } | Flat::Jump { .. } | Flat::Move { .. } => None,
    }
}

/// `f` with its registers renamed by `r` (and its destination by `r` too).
fn rename(f: &Flat, r: &mut impl FnMut(u32) -> u32) -> Flat {
    let mut g = *f;
    match &mut g {
        Flat::Add { d, a, b }
        | Flat::Sub { d, a, b }
        | Flat::RSub { d, a, b }
        | Flat::Mul { d, a, b }
        | Flat::Div { d, a, b }
        | Flat::DivRaw { d, a, b }
        | Flat::Cmp { d, a, b, .. }
        | Flat::And { d, a, b }
        | Flat::Or { d, a, b } => {
            *a = r(*a);
            *b = r(*b);
            *d = r(*d);
        }
        Flat::Pow { d, a, e } | Flat::GammaPq { d, a, x: e, .. } => {
            *a = r(*a);
            *e = r(*e);
            *d = r(*d);
        }
        Flat::Select { d, c, a, b } => {
            *c = r(*c);
            *a = r(*a);
            *b = r(*b);
            *d = r(*d);
        }
        Flat::Gate { d, a, b, .. } => {
            *a = r(*a);
            *b = r(*b);
            *d = r(*d);
        }
        Flat::AddK { d, a, .. }
        | Flat::SubK { d, a, .. }
        | Flat::RSubK { d, a, .. }
        | Flat::MulK { d, a, .. }
        | Flat::DivK { d, a, .. }
        | Flat::Unary { d, a, .. }
        | Flat::Not { d, a }
        | Flat::Present { d, a } => {
            *a = r(*a);
            *d = r(*d);
        }
        Flat::LoadState { d, .. }
        | Flat::LoadParam { d, .. }
        | Flat::LoadSeed { d, .. }
        | Flat::LoadK { d, .. }
        | Flat::HostOut { d, .. }
        | Flat::LoopOut { d, .. } => *d = r(*d),
        Flat::Deriv { a, .. } | Flat::HostArg { a } | Flat::LoopArg { a } => *a = r(*a),
        Flat::Branch { .. }
        | Flat::Jump { .. }
        | Flat::Move { .. }
        | Flat::Host { .. }
        | Flat::Loop { .. } => {}
    }
    g
}

/// The bit set on every value that depends on θ or η.
pub(super) const FREE: u64 = 1 << 63;

/// Each register's live directions: the η a value can depend on (bit `k` = η `k`), and
/// [`FREE`] when it depends on any input. A lane outside it is an exact zero, whatever the
/// inputs' values.
fn direction_masks(
    code: &[Flat],
    n_regs: usize,
    eta_regs: &[(u32, usize)],
    dense: bool,
) -> Vec<u64> {
    let mut m = vec![0u64; n_regs];
    for &(r, k) in eta_regs {
        m[r as usize] = 1u64 << k | FREE;
    }
    // A loop call's outputs depend on every input of the call (its `LoopArg`s, which start
    // afresh at each call's first).
    let (mut loop_acc, mut in_args) = (0u64, false);
    for f in code {
        if let Flat::LoopArg { a } = *f {
            if !in_args {
                loop_acc = 0;
                in_args = true;
            }
            loop_acc |= m[a as usize];
            continue;
        }
        in_args = false;
        let Some(d) = def(f) else { continue };
        let v = match *f {
            Flat::LoopOut { .. } => loop_acc,
            Flat::LoadState { .. } => m[d as usize],
            Flat::LoadParam { .. } => {
                if dense {
                    !0
                } else {
                    FREE
                }
            }
            Flat::LoadSeed { .. } | Flat::LoadK { .. } => 0,
            Flat::Cmp { .. }
            | Flat::And { .. }
            | Flat::Or { .. }
            | Flat::Not { .. }
            | Flat::Present { .. }
            | Flat::HostOut { .. } => 0,
            _ => uses(f)
                .iter()
                .flatten()
                .fold(0, |acc, &u| acc | m[u as usize]),
        };
        m[d as usize] = v;
    }
    m
}

impl Recording {
    /// Cut at the marks, compile (or reuse) each chunk, and lay out the subject's plan.
    pub(crate) fn plan(self, layout: Layout, n_eta: usize, n_theta: usize) -> Option<Plan> {
        let lanes = layout.lanes();
        if n_eta > 63 {
            return None;
        }
        let n_regs = self.n_regs as usize;
        let masks = direction_masks(&self.code, n_regs, &self.eta_regs, self.dense);
        let is_input = |f: &Flat| matches!(f, Flat::LoadState { .. } | Flat::LoadParam { .. });
        // Slots: η, θ, then outputs as they appear.
        const NONE: u32 = u32::MAX;
        let mut slot = vec![NONE; n_regs];
        let mut eta_of = vec![u8::MAX; n_regs];
        for &(r, k) in &self.eta_regs {
            slot[r as usize] = k as u32;
            eta_of[r as usize] = k as u8;
        }
        for &(r, j) in &self.theta_regs {
            slot[r as usize] = (n_eta + j) as u32;
        }
        let mut n_slots = n_eta + n_theta;
        // The last instruction reading each register (outputs and the guard: forever).
        let mut last_use = vec![0usize; n_regs];
        for (i, f) in self.code.iter().enumerate() {
            for u in uses(f).into_iter().flatten() {
                last_use[u as usize] = i;
            }
        }
        for &o in self.outs.iter().chain(self.guard.iter()) {
            last_use[o as usize] = usize::MAX;
        }
        // Chunk boundaries: the marks, after dropping the inputs' loads.
        let mut bounds: Vec<usize> = vec![0];
        // Every mark: a chunk is one event, so subjects with the same event kinds share it.
        bounds.extend(
            self.marks
                .iter()
                .copied()
                .filter(|&m| m > 0 && m < self.code.len()),
        );
        // A loop call is a step of its own: `LoopArg`s, `Loop`, `LoopOut`s.
        for (i, f) in self.code.iter().enumerate() {
            let prev_arg = i > 0 && matches!(self.code[i - 1], Flat::LoopArg { .. });
            match f {
                Flat::LoopArg { .. } if !prev_arg => bounds.push(i),
                Flat::Loop { .. } | Flat::LoopOut { .. } => {
                    if !matches!(self.code.get(i + 1), Some(Flat::LoopOut { .. })) {
                        bounds.push(i + 1);
                    }
                }
                _ => {}
            }
        }
        bounds.push(self.code.len());
        bounds.sort_unstable();
        bounds.dedup();
        let mut calls = Vec::new();
        let mut cache = chunk_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for w in bounds.windows(2) {
            let (s, e) = (w[0], w[1]);
            if matches!(self.code[s], Flat::LoopArg { .. }) {
                let mut args = Vec::new();
                let mut outs = Vec::new();
                let mut spec = None;
                for f in &self.code[s..e] {
                    match *f {
                        Flat::LoopArg { a } => args.push(slot[a as usize]),
                        Flat::Loop { id } => spec = self.loops.get(id as usize).cloned(),
                        Flat::LoopOut { d, .. } => {
                            slot[d as usize] = n_slots as u32;
                            outs.push(n_slots as u32);
                            n_slots += 1;
                        }
                        _ => return None,
                    }
                }
                let spec = spec?;
                if args.contains(&NONE) || args.len() != spec.n_in() || outs.len() != spec.n_out() {
                    return None;
                }
                calls.push(Call::Loop {
                    spec,
                    args: args.into_boxed_slice(),
                    outs: outs.into_boxed_slice(),
                });
                continue;
            }
            let body: Vec<&Flat> = self.code[s..e].iter().filter(|f| !is_input(f)).collect();
            if body.is_empty() {
                continue;
            }
            // Inputs by first use; outputs, the registers read past the chunk, by definition.
            let mut ins: Vec<u32> = Vec::new();
            let mut seen: std::collections::HashSet<u32, FxBuild> = Default::default();
            let mut outs: Vec<u32> = Vec::new();
            for f in &body {
                for u in uses(f).into_iter().flatten() {
                    if seen.insert(u) {
                        ins.push(u);
                    }
                }
                if let Some(d) = def(f) {
                    seen.insert(d);
                    if last_use[d as usize] >= e {
                        outs.push(d);
                    }
                }
            }
            // The inputs' structure: live directions, and the exact one of a seed.
            let spec: Vec<(u64, u8)> = ins
                .iter()
                .map(|&r| (masks[r as usize], eta_of[r as usize]))
                .collect();
            let body: Vec<Flat> = body.into_iter().copied().collect();
            let (prog, lits) = build_chunk(&body, &ins, &outs, &self.pool, spec, &mut cache);
            let mut idx: Vec<u32> = ins.iter().map(|&r| slot[r as usize]).collect();
            if idx.contains(&NONE) {
                return None;
            }
            for &o in &outs {
                slot[o as usize] = n_slots as u32;
                idx.push(n_slots as u32);
                n_slots += 1;
            }
            calls.push(Call::Chunk {
                prog,
                idx: idx.into_boxed_slice(),
                lits: lits.into_boxed_slice(),
            });
        }
        let outs: Option<Vec<u32>> = self
            .outs
            .iter()
            .map(|&o| (slot[o as usize] != NONE).then_some(slot[o as usize]))
            .collect();
        let guard = match self.guard {
            Some(g) if slot[g as usize] == NONE => return None,
            Some(g) => Some(slot[g as usize]),
            None => None,
        };
        Some(Plan {
            calls,
            n_slots,
            layout,
            lanes,
            n_eta,
            n_theta,
            outs: outs?,
            guard,
            fast: OnceLock::new(),
            resolved: Mutex::new(HashMap::new()),
            ode_override: self.ode.then(crate::ode::solver::current_ode_override),
            declined: self.declined,
        })
    }
}

thread_local! {
    static BUF: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static ADJ: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// A plan's forward at one layout, kept for its reverse passes ([`Plan::forward_kept`]):
/// every slot's lanes and every loop call's run log. A reverse at this layout, or at one
/// whose lanes are a prefix of it ([`prefix_layout`]), reads it instead of running the
/// forward again.
pub(crate) struct Fwd {
    plan: usize,
    layout: Layout,
    buf: Vec<f64>,
    logs: Vec<CallLog>,
}

impl Fwd {
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }
}

/// Whether `lo`'s lanes are the leading lanes of `hi`'s, each the same quantity: the value
/// alone, or the value and the same tangents without `hi`'s second-order payload.
fn prefix_layout(lo: Layout, hi: Layout) -> bool {
    lo == hi || (lo.m == 0 && !lo.th && lo.wide == hi.wide && (lo.n == 0 || lo.n == hi.n))
}

/// `v`'s blocks of `hi` lanes cut to their leading `lo` lanes.
fn compact(v: &[f64], hi: usize, lo: usize) -> Vec<f64> {
    v.chunks_exact(hi)
        .flat_map(|c| c[..lo].iter().copied())
        .collect()
}

impl Plan {
    pub(crate) fn n_out(&self) -> usize {
        self.outs.len()
    }
    pub(crate) fn n_theta(&self) -> usize {
        self.n_theta
    }
    pub(crate) fn n_eta(&self) -> usize {
        self.n_eta
    }

    /// Every chunk's entry point over `(layout, rev)`, resolved once.
    fn fns(&self, layout: Layout, rev: bool) -> Option<Arc<Vec<Kern>>> {
        let mut r = self.resolved.lock().unwrap();
        if let Some(v) = r.get(&(layout, rev)) {
            return v.clone();
        }
        let v: Option<Vec<Kern>> = self
            .calls
            .iter()
            .map(|c| match c {
                Call::Chunk { prog, .. } => prog.get(layout, rev).map(Kern::Chunk),
                Call::Loop { spec, .. } => match spec {
                    CallKind::Ode(s) => {
                        ode_loop::LoopKernels::resolve(s, layout, rev).map(Kern::Loop)
                    }
                    CallKind::Fixed(f) => Some(Kern::Fixed(f.clone())),
                    CallKind::Repeat(r) => Some(Kern::Repeat(r.clone())),
                },
            })
            .collect();
        let v = v.map(Arc::new);
        r.insert((layout, rev), v.clone());
        v
    }

    /// The forward over `layout` into `buf` (`n_slots × lanes`), η and θ already in place.
    /// The forward over `layout` into `buf`: `false` when a loop call fails.
    fn forward_into(&self, fns: &[Kern], buf: &mut [f64], layout: Layout) -> bool {
        self.forward_logged(fns, buf, layout, None)
    }

    /// [`Self::forward_into`], keeping each loop call's run log when `logs` asks.
    fn forward_logged(
        &self,
        fns: &[Kern],
        buf: &mut [f64],
        layout: Layout,
        mut logs: Option<&mut Vec<CallLog>>,
    ) -> bool {
        if let Some(lg) = logs.as_deref_mut() {
            lg.clear();
        }
        if self.declined > 0 && !recording_open() {
            crate::ode::solver::note_linear_declines(self.declined);
        }
        if let Some(ov) = self.ode_override {
            if crate::ode::solver::current_ode_override() != ov {
                return false;
            }
        }
        let l = layout.lanes();
        let mut auto = crate::ode::solver::OdeAutoSwitchState::default();
        for (c, k) in self.calls.iter().zip(fns) {
            match (c, k) {
                (Call::Chunk { idx, lits, .. }, Kern::Chunk(k)) => {
                    // SAFETY: every slot in `idx` is below `n_slots`; the kernel reads its
                    // inputs' live lanes and its literals, and writes its outputs' lanes, at
                    // the layout it was built for, which is `buf`'s.
                    unsafe {
                        k.call(
                            buf.as_mut_ptr(),
                            std::ptr::null_mut(),
                            idx.as_ptr(),
                            lits.as_ptr(),
                        )
                    };
                }
                (Call::Loop { args, outs, .. }, k @ (Kern::Fixed(_) | Kern::Repeat(_))) => {
                    let inputs: Vec<f64> = args
                        .iter()
                        .flat_map(|&a| buf[a as usize * l..(a as usize + 1) * l].iter().copied())
                        .collect();
                    let mut out = vec![0.0; outs.len() * l];
                    let mut flog = None;
                    let want = logs.is_some();
                    let ok = match k {
                        Kern::Fixed(f) => {
                            f.run(layout, &inputs, &mut out, want.then_some(&mut flog))
                        }
                        Kern::Repeat(r) => {
                            r.run(layout, &inputs, &mut out, want.then_some(&mut flog))
                        }
                        _ => false,
                    };
                    if !ok {
                        return false;
                    }
                    if let Some(lg) = logs.as_deref_mut() {
                        lg.push(CallLog::Fixed(flog));
                    }
                    for (j, &o) in outs.iter().enumerate() {
                        buf[o as usize * l..(o as usize + 1) * l]
                            .copy_from_slice(&out[j * l..(j + 1) * l]);
                    }
                }
                (
                    Call::Loop {
                        spec: CallKind::Ode(spec),
                        args,
                        outs,
                    },
                    Kern::Loop(lk),
                ) => {
                    let inputs: Vec<f64> = args
                        .iter()
                        .flat_map(|&a| buf[a as usize * l..(a as usize + 1) * l].iter().copied())
                        .collect();
                    let mut out = vec![0.0; outs.len() * l];
                    let mut log = Vec::new();
                    let want = logs.is_some();
                    if !ode_loop::run_loop(
                        spec,
                        lk,
                        l,
                        &inputs,
                        &mut out,
                        &mut auto,
                        want.then_some(&mut log),
                    ) {
                        return false;
                    }
                    if let Some(lg) = logs.as_deref_mut() {
                        lg.push(CallLog::Ode(log));
                    }
                    for (j, &o) in outs.iter().enumerate() {
                        buf[o as usize * l..(o as usize + 1) * l]
                            .copy_from_slice(&out[j * l..(j + 1) * l]);
                    }
                }
                _ => return false,
            }
        }
        true
    }

    /// Run over the inner loop's layout at `(theta, eta)` (η seeded one-hot), writing each
    /// output's lanes to `out` (`n_out × lanes`); `false` when a guard fails.
    pub(crate) fn run(&self, theta: &[f64], eta: &[f64], out: &mut [f64]) -> bool {
        let Some(fns) = self.fast.get_or_init(|| self.fns(self.layout, false)) else {
            return false;
        };
        BUF.with(|b| {
            let mut b = b.borrow_mut();
            let need = self.n_slots * self.lanes;
            if b.len() < need {
                b.resize(need, 0.0);
            }
            let buf = &mut b[..need];
            let l = self.lanes;
            for k in 0..self.n_eta {
                let s = &mut buf[k * l..(k + 1) * l];
                s.fill(0.0);
                s[0] = eta[k];
                s[1 + k] = 1.0;
            }
            for j in 0..self.n_theta {
                let s = &mut buf[(self.n_eta + j) * l..(self.n_eta + j + 1) * l];
                s.fill(0.0);
                s[0] = theta[j];
            }
            if !self.forward_into(fns, buf, self.layout) {
                return false;
            }
            if let Some(g) = self.guard {
                if buf[g as usize * l] != 1.0 {
                    return false;
                }
            }
            for (i, &o) in self.outs.iter().enumerate() {
                out[i * l..(i + 1) * l].copy_from_slice(&buf[o as usize * l..(o as usize + 1) * l]);
            }
            true
        })
    }

    /// The forward over any `layout`: η's and θ's lanes from `u` and `params` (laid out at
    /// `layout`'s lanes, η seeded one-hot as every chunk was folded for), the outputs then
    /// the guard (`1` when it holds) into `du`, and the forward kept for reverse passes.
    /// `None` when some chunk cannot be built or a loop call fails.
    pub(crate) fn forward_layout(
        &self,
        layout: Layout,
        u: &[f64],
        params: &[f64],
        du: &mut [f64],
    ) -> Option<Fwd> {
        let f = self.forward_kept(layout, u, params)?;
        let l = layout.lanes();
        for i in 0..self.outs.len() {
            du[i * l..(i + 1) * l].copy_from_slice(self.kept_out(&f, i));
        }
        let g = &mut du[self.outs.len() * l..(self.outs.len() + 1) * l];
        g.fill(0.0);
        g[0] = if self.kept_guard(&f) { 1.0 } else { 0.0 };
        Some(f)
    }

    /// The forward over `layout` from `u` and `params` (laid out at its lanes, η seeded
    /// one-hot), kept for reverse passes: `None` when a loop call fails. The guard is not
    /// judged here ([`Self::kept_guard`]).
    pub(crate) fn forward_kept(&self, layout: Layout, u: &[f64], params: &[f64]) -> Option<Fwd> {
        let fwd = self.fns(layout, false)?;
        let l = layout.lanes();
        let ins = (self.n_eta + self.n_theta) * l;
        let mut buf = vec![0.0; self.n_slots * l];
        buf[..self.n_eta * l].copy_from_slice(&u[..self.n_eta * l]);
        buf[self.n_eta * l..ins].copy_from_slice(&params[..self.n_theta * l]);
        let mut logs = Vec::new();
        self.forward_logged(&fwd, &mut buf, layout, Some(&mut logs))
            .then_some(Fwd {
                plan: self as *const Plan as usize,
                layout,
                buf,
                logs,
            })
    }

    /// Output `i`'s lanes in a kept forward.
    pub(crate) fn kept_out<'a>(&self, f: &'a Fwd, i: usize) -> &'a [f64] {
        let l = f.layout.lanes();
        let o = self.outs[i] as usize;
        &f.buf[o * l..(o + 1) * l]
    }

    /// Whether the recording's guard holds in a kept forward.
    pub(crate) fn kept_guard(&self, f: &Fwd) -> bool {
        self.guard
            .is_none_or(|g| f.buf[g as usize * f.layout.lanes()] == 1.0)
    }

    /// The reverse over `layout` from a kept forward at `layout` or at a layout whose lanes
    /// extend it ([`prefix_layout`]), for `seeds` adjoint seeds at once, each a block of
    /// `du_bar` (`n_out × lanes`) and of `u_bar` and `p_bar`: adds `dūᵀ·∂du/∂(η, θ)`. Each
    /// chunk's (and each loop step's) recomputed forward prefix is shared by every seed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reverse_from(
        &self,
        f: &Fwd,
        layout: Layout,
        seeds: usize,
        u_bar: &mut [f64],
        p_bar: &mut [f64],
        du_bar: &[f64],
    ) -> bool {
        if f.plan != self as *const Plan as usize || !prefix_layout(layout, f.layout) {
            return false;
        }
        let Some(rev) = self.fns(layout, true) else {
            return false;
        };
        let l = layout.lanes();
        let hl = f.layout.lanes();
        let need = self.n_slots * l;
        // The forward at `layout`: the kept one, its trailing lanes dropped when it is wider.
        let mut buf = if hl == l {
            f.buf.clone()
        } else {
            compact(&f.buf, hl, l)
        };
        // An ODE call's log at `layout`: the kept one, its bodies' lanes cut when wider.
        fn cut(
            lg: &[ode_loop::Entry],
            hl: usize,
            l: usize,
        ) -> std::borrow::Cow<'_, [ode_loop::Entry]> {
            if hl == l {
                std::borrow::Cow::Borrowed(lg)
            } else {
                std::borrow::Cow::Owned(
                    lg.iter()
                        .map(|e| match e {
                            ode_loop::Entry::Body { m, kind, vars } => ode_loop::Entry::Body {
                                m: *m,
                                kind: *kind,
                                vars: compact(vars, hl, l),
                            },
                            e => e.clone(),
                        })
                        .collect(),
                )
            }
        }
        // The adjoint buffer, out of its thread-local for the sweep: a call's own sub-plans
        // (`fixed`) run their reverses inside this one.
        let mut a = ADJ.with(|c| std::mem::take(&mut *c.borrow_mut()));
        let ok = (|| {
            a.clear();
            a.resize(seeds * need, 0.0);
            let adj = &mut a[..seeds * need];
            let n_out = self.outs.len() * l;
            for sd in 0..seeds {
                for (i, &o) in self.outs.iter().enumerate() {
                    for c in 0..l {
                        adj[sd * need + o as usize * l + c] += du_bar[sd * n_out + i * l + c];
                    }
                }
            }
            let mut li = f.logs.len();
            for (c, k) in self.calls.iter().zip(rev.iter()).rev() {
                match (c, k) {
                    (Call::Chunk { idx, lits, .. }, Kern::Chunk(k)) => {
                        // SAFETY: as `forward_into`; each seed's `adj` block is laid out like
                        // `buf`.
                        unsafe {
                            k.call_seeds(
                                buf.as_mut_ptr(),
                                adj.as_mut_ptr(),
                                need,
                                seeds,
                                idx.as_ptr(),
                                lits.as_ptr(),
                            )
                        };
                    }
                    (
                        Call::Loop { spec, args, outs },
                        k @ (Kern::Loop(_) | Kern::Fixed(_) | Kern::Repeat(_)),
                    ) if li > 0 => {
                        li -= 1;
                        let inputs: Vec<f64> = args
                            .iter()
                            .flat_map(|&a| {
                                buf[a as usize * l..(a as usize + 1) * l].iter().copied()
                            })
                            .collect();
                        let mut out_adj = Vec::with_capacity(seeds * outs.len() * l);
                        for sd in 0..seeds {
                            let blk = &adj[sd * need..(sd + 1) * need];
                            out_adj.extend(outs.iter().flat_map(|&o| {
                                blk[o as usize * l..(o as usize + 1) * l].iter().copied()
                            }));
                        }
                        let mut in_adj = vec![0.0; seeds * args.len() * l];
                        let ok = match (spec, k, &f.logs[li]) {
                            (CallKind::Ode(spec), Kern::Loop(rk), CallLog::Ode(lg)) => {
                                ode_loop::run_loop_rev(
                                    spec,
                                    rk,
                                    l,
                                    &inputs,
                                    &cut(lg, hl, l),
                                    seeds,
                                    &out_adj,
                                    &mut in_adj,
                                )
                            }
                            (CallKind::Fixed(_), Kern::Fixed(fx), CallLog::Fixed(Some(fl))) => {
                                fx.run_rev(layout, fl, seeds, &out_adj, &mut in_adj)
                            }
                            (CallKind::Repeat(_), Kern::Repeat(rp), CallLog::Fixed(Some(fl))) => {
                                rp.run_rev(layout, fl, seeds, &out_adj, &mut in_adj)
                            }
                            _ => false,
                        };
                        if !ok {
                            return false;
                        }
                        for sd in 0..seeds {
                            for (j, &a) in args.iter().enumerate() {
                                for x in 0..l {
                                    adj[sd * need + a as usize * l + x] +=
                                        in_adj[(sd * args.len() + j) * l + x];
                                }
                            }
                        }
                    }
                    _ => return false,
                }
            }
            let (ne, nt) = (self.n_eta * l, self.n_theta * l);
            for sd in 0..seeds {
                let blk = &adj[sd * need..];
                for (x, y) in u_bar[sd * ne..(sd + 1) * ne].iter_mut().zip(&blk[..ne]) {
                    *x += *y;
                }
                for (x, y) in p_bar[sd * nt..(sd + 1) * nt]
                    .iter_mut()
                    .zip(&blk[ne..ne + nt])
                {
                    *x += *y;
                }
            }
            true
        })();
        ADJ.with(|c| *c.borrow_mut() = a);
        ok
    }
}

/// [`fold`] for a chunk: input `i`'s lanes outside its directions are zero, a seed's own
/// lane is one; literals carry no tangent.
fn fold_chunk(src: &Func, layout: Layout, spec: &[(u64, u8)]) -> Func {
    let lanes = layout.lanes();
    let n = layout.n as usize;
    // Direction `k`'s mask bit: η `k`'s.
    let bit = |k: usize| k as u32;
    let lane_dir = |lane: usize| -> Option<(usize, usize)> {
        // lane 1..=n: direction k; payload lanes: the pair (k, l) of `ir::pairs`' order.
        if lane == 0 {
            None
        } else if lane <= n {
            Some((lane - 1, lane - 1))
        } else {
            let mut t = lane - 1 - n;
            for k in 0..n {
                let row = if layout.th { 1 } else { k + 1 };
                if t < row {
                    let l = if layout.th { k } else { t };
                    return Some((k, l));
                }
                t -= row;
            }
            None
        }
    };
    let known = |arg: u8, off: u32| -> Option<f64> {
        let (idx, lane) = (off as usize / lanes, off as usize % lanes);
        if lane == 0 {
            return None;
        }
        if arg == 2 {
            return Some(0.0);
        }
        if arg != 0 {
            return None;
        }
        let (mask, seed) = spec[idx];
        let (k, l) = lane_dir(lane)?;
        if lane <= n && seed as usize == k {
            return Some(1.0);
        }
        if seed != u8::MAX && lane > n {
            return Some(0.0);
        }
        if mask >> bit(k) & 1 == 0 || mask >> bit(l) & 1 == 0 {
            return Some(0.0);
        }
        None
    };
    fold_with(src, known)
}

/// With `rev`, the first `prefix` instructions are the recomputed forward ([`ir::reverse`]):
/// they run once, the rest in a loop over the seeds.
#[allow(clippy::too_many_arguments)]
fn lower_chunk_opt(
    f: &Func,
    layout: Layout,
    n_in: usize,
    rev: bool,
    prefix: usize,
    quick: bool,
    opt: &str,
) -> Option<ChunkKernel> {
    use super::ir::{Cc, Shim1};
    use cranelift_codegen::ir::condcodes::FloatCC;
    let _ = Shim1::Nan;
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, Signature, Value};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{default_libcall_names, Linkage, Module};
    extern "C" fn sh_powf(x: f64, e: f64) -> f64 {
        x.powf(e)
    }
    let lanes = layout.lanes();
    // The target, per tier: built once (host feature detection, the flags' parse), shared.
    type Isa = Option<cranelift_codegen::isa::OwnedTargetIsa>;
    static ISAS: OnceLock<Mutex<HashMap<(bool, String), Isa>>> = OnceLock::new();
    let isa = ISAS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .entry((quick, opt.to_string()))
        .or_insert_with(|| {
            let mut flags = settings::builder();
            flags.set("opt_level", opt).ok()?;
            // The IR verifier is a debugging check of Cranelift's own invariants.
            flags.set("enable_verifier", "false").ok()?;
            // The quick tier: the single-pass register allocator.
            if quick {
                flags.set("regalloc_algorithm", "single_pass").ok()?;
            }
            flags.set("use_colocated_libcalls", "false").ok()?;
            flags.set("is_pic", "false").ok()?;
            cranelift_native::builder()
                .ok()?
                .finish(settings::Flags::new(flags))
                .ok()
        })
        .clone()?;
    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
    let target = module.target_config();
    let ptr = target.pointer_type();
    let mut sig = module.make_signature();
    for _ in 0..if rev { 6 } else { 3 } {
        sig.params.push(AbiParam::new(ptr));
    }
    let id = module
        .declare_function("chunk", Linkage::Local, &sig)
        .ok()?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let block = b.create_block();
        b.append_block_params_for_function_params(block);
        b.switch_to_block(block);
        let args = b.block_params(block).to_vec();
        let (base, mut adjp, idxp, litp) = if rev {
            (args[0], Some(args[1]), args[2], args[3])
        } else {
            (args[0], None, args[1], args[2])
        };
        // The seed loop: `head(adj, n)` tests `n`, `body` runs the adjoint part on `adj`.
        let seed_loop = rev.then(|| {
            let head = b.create_block();
            b.append_block_param(head, ptr);
            b.append_block_param(head, ptr);
            let body = b.create_block();
            let exit = b.create_block();
            (head, body, exit)
        });
        let mut usig = Signature::new(target.default_call_conv);
        usig.params.push(AbiParam::new(types::F64));
        usig.returns.push(AbiParam::new(types::F64));
        let mut bsig = usig.clone();
        bsig.params.push(AbiParam::new(types::F64));
        let uref = b.import_signature(usig);
        let bref = b.import_signature(bsig);
        let fl = MemFlagsData::trusted();
        // Slot table entry `e`'s byte offset, then its lane block in `base` (or `adj`),
        // each computed once.
        let mut offs: HashMap<usize, Value, FxBuild> = HashMap::default();
        let mut addr: HashMap<(usize, bool), Value, FxBuild> = HashMap::default();
        let slot_addr = |b: &mut FunctionBuilder,
                         offs: &mut HashMap<usize, Value, FxBuild>,
                         addr: &mut HashMap<(usize, bool), Value, FxBuild>,
                         adjp: Option<Value>,
                         e: usize,
                         in_adj: bool|
         -> Value {
            if let Some(&a) = addr.get(&(e, in_adj)) {
                return a;
            }
            let off = *offs.entry(e).or_insert_with(|| {
                let s = b.ins().load(types::I32, fl, idxp, (4 * e) as i32);
                let s = b.ins().uextend(types::I64, s);
                b.ins().imul_imm_u(s, (lanes * 8) as i64)
            });
            let buf = if in_adj {
                adjp.expect("an adjoint buffer")
            } else {
                base
            };
            let a = b.ins().iadd(buf, off);
            addr.insert((e, in_adj), a);
            a
        };
        let mut v: Vec<Value> = Vec::with_capacity(f.insts.len());
        for (at, inst) in f.insts.iter().enumerate() {
            if let (true, Some((head, body, _))) = (at == prefix, seed_loop) {
                // Into the seed loop: the forward's values dominate it.
                b.ins().jump(head, &[args[1].into(), args[4].into()]);
                b.switch_to_block(head);
                let hp = b.block_params(head).to_vec();
                let exit = seed_loop.expect("rev").2;
                b.ins().brif(hp[1], body, &[], exit, &[]);
                b.switch_to_block(body);
                adjp = Some(hp[0]);
                // An adjoint block's address is the seed's: computed afresh in the body.
                addr.retain(|&(_, in_adj), _| !in_adj);
            }
            let r = |x: ir::V| v[x as usize];
            let out = match *inst {
                Inst::Load { arg, off } => {
                    let (i, lane) = (off as usize / lanes, off as usize % lanes);
                    match arg {
                        0 => {
                            let a = slot_addr(&mut b, &mut offs, &mut addr, adjp, i, false);
                            b.ins().load(types::F64, fl, a, (lane * 8) as i32)
                        }
                        // A literal: the pool holds values only; its derivative lanes are zeros
                        // (folded away unless strict, which must still read them as zero).
                        2 if lane == 0 => b.ins().load(types::F64, fl, litp, (i * 8) as i32),
                        2 => b.ins().f64const(0.0),
                        4 if rev => {
                            let a = slot_addr(&mut b, &mut offs, &mut addr, adjp, i, true);
                            b.ins().load(types::F64, fl, a, (lane * 8) as i32)
                        }
                        7 if rev => {
                            let a = slot_addr(&mut b, &mut offs, &mut addr, adjp, n_in + i, true);
                            b.ins().load(types::F64, fl, a, (lane * 8) as i32)
                        }
                        // The literals' adjoints: discarded (read as zero, never stored).
                        6 if rev => b.ins().f64const(0.0),
                        _ => return None,
                    }
                }
                Inst::Const(bits) => b.ins().f64const(f64::from_bits(bits)),
                Inst::Add(x, y) => b.ins().fadd(r(x), r(y)),
                Inst::Sub(x, y) => b.ins().fsub(r(x), r(y)),
                Inst::Mul(x, y) => b.ins().fmul(r(x), r(y)),
                Inst::Div(x, y) => b.ins().fdiv(r(x), r(y)),
                Inst::Neg(x) => b.ins().fneg(r(x)),
                Inst::Abs(x) => b.ins().fabs(r(x)),
                Inst::Floor(x) => b.ins().floor(r(x)),
                Inst::Ceil(x) => b.ins().ceil(r(x)),
                Inst::Call1(Shim1::Sqrt, x) => b.ins().sqrt(r(x)),
                Inst::Call1(s, x) => {
                    let fp = s.f();
                    let callee = b.ins().iconst(ptr, fp as usize as i64);
                    let call = b.ins().call_indirect(uref, callee, &[r(x)]);
                    b.inst_results(call)[0]
                }
                Inst::Powf(x, y) => {
                    let fp: extern "C" fn(f64, f64) -> f64 = sh_powf;
                    let callee = b.ins().iconst(ptr, fp as usize as i64);
                    let call = b.ins().call_indirect(bref, callee, &[r(x), r(y)]);
                    b.inst_results(call)[0]
                }
                Inst::Gamma {
                    upper,
                    i,
                    j,
                    a: x,
                    x: y,
                } => {
                    let fp = ir::gamma_shim(upper, i, j);
                    let callee = b.ins().iconst(ptr, fp as usize as i64);
                    let call = b.ins().call_indirect(bref, callee, &[r(x), r(y)]);
                    b.inst_results(call)[0]
                }
                Inst::Cmp(cc, x, y) => {
                    let cc = match cc {
                        Cc::Lt => FloatCC::LessThan,
                        Cc::Le => FloatCC::LessThanOrEqual,
                        Cc::Gt => FloatCC::GreaterThan,
                        Cc::Ge => FloatCC::GreaterThanOrEqual,
                        Cc::Eq => FloatCC::Equal,
                        Cc::Ne => FloatCC::NotEqual,
                    };
                    b.ins().fcmp(cc, r(x), r(y))
                }
                Inst::And(x, y) => b.ins().band(r(x), r(y)),
                Inst::Or(x, y) => b.ins().bor(r(x), r(y)),
                Inst::Select(c, x, y) => b.ins().select(r(c), r(x), r(y)),
                Inst::StopGrad(x) => r(x),
                Inst::HostArg(_) | Inst::Host(_) | Inst::HostOut(_) => {
                    let (fp, arg): (extern "C" fn(f64) -> f64, Value) = match *inst {
                        Inst::HostArg(x) => (ir::host::arg_native, r(x)),
                        Inst::Host(f) => {
                            (ir::host::call_native, b.ins().f64const(f.index() as f64))
                        }
                        Inst::HostOut(k) => (ir::host::out_native, b.ins().f64const(k as f64)),
                        _ => unreachable!(),
                    };
                    let callee = b.ins().iconst(ptr, fp as usize as i64);
                    let call = b.ins().call_indirect(uref, callee, &[arg]);
                    b.inst_results(call)[0]
                }
                Inst::Store { arg, off, v: x } => {
                    let (j, lane) = (off as usize / lanes, off as usize % lanes);
                    let a = match arg {
                        3 if !rev => slot_addr(&mut b, &mut offs, &mut addr, adjp, n_in + j, false),
                        4 if rev => slot_addr(&mut b, &mut offs, &mut addr, adjp, j, true),
                        6 if rev => {
                            v.push(r(x));
                            continue;
                        }
                        _ => return None,
                    };
                    b.ins().store(fl, r(x), a, (lane * 8) as i32);
                    r(x)
                }
            };
            v.push(out);
        }
        if let Some((head, _, exit)) = seed_loop {
            if f.insts.len() <= prefix {
                // No adjoint part: the loop's blocks still need their terminators.
                b.ins().jump(head, &[args[1].into(), args[4].into()]);
                b.switch_to_block(head);
                let body = seed_loop.expect("rev").1;
                let n = b.block_params(head)[1];
                b.ins().brif(n, body, &[], exit, &[]);
                b.switch_to_block(body);
                b.ins().jump(exit, &[]);
            } else {
                let hp_adj = adjp.expect("in the loop");
                let step = b.ins().imul_imm_s(args[5], 8);
                let next = b.ins().iadd(hp_adj, step);
                // `n − 1` from the head's count.
                let n = b.block_params(head)[1];
                let n1 = b.ins().iadd_imm_s(n, -1);
                b.ins().jump(head, &[next.into(), n1.into()]);
            }
            b.switch_to_block(exit);
        }
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(target);
    }
    module.define_function(id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().ok()?;
    let p = module.get_finalized_function(id);
    // SAFETY: defined with `ChunkFn`'s signature.
    Some(ChunkKernel {
        f: p,
        _module: module,
    })
}
