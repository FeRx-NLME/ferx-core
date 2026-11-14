//! Compiled per-subject plans (#2786): a subject's computation recorded once, compiled, and
//! replayed for every derivative a fit takes of it.
//!
//! [`Tr`] is the recording number type. Running a generic `*_g<T: PkNum>` recorder on it — a
//! subject's predictions, a non-Gaussian likelihood term, an SDE subject's variance rows, or a
//! small θ-only function ([`derivs`]) — writes the register program ([`Flat`]) of that
//! computation: θ and η are inputs, data are literals, and identical operations on identical
//! operands are one register (hash-consing). `Recording::plan` lowers the program by the RHS's
//! own lane code ([`ir::forward`]) at the layout a caller asks for (value, η-gradient, packed
//! η-Hessian), so each operation is the forward-dual arithmetic lane for lane; the lanes the
//! inputs fix (η's one-hot seeds, θ's zero tangents) become constants, a folding pass drops the
//! terms they make exactly `0` or `1`, and Cranelift compiles the result. The plan's reverse
//! (`Plan::reverse_from`) gives the θ-gradients and the contractions of the η-jets the outer
//! gradient, `H` and the covariance R-matrix need — no second copy of any formula.
//!
//! **Soundness.** A plan is valid only where every branch the recorder took still goes the same
//! way. Reading an input-dependent value ([`PkNum::val`]) aborts the recording unless it is a
//! guard; a guard ([`Tr::guard_cmp`]) is compiled as a check, whose verdict is the plan's last
//! output. A plan whose guard fails does not serve: the caller records another variant at that
//! point (a subject keeps up to `MAX_VARIANTS`), so a subject is followed across its branches
//! with exact derivatives on each.
//!
//! **Failure.** A recording that cannot be made ([`end`] says why) or a plan that does not serve
//! at its own recording point marks the subject — or, for a reason of the model's, the model —
//! bad: its derivatives take finite differences, and the fit reports every such subject once
//! (`W_PLAN_FD_FALLBACK`, [`plan_failures`]). It should never happen. `FERX_KJIT_DEBUG=1`
//! prints each failed recording's reason, replays every recording on values to check its
//! lowering, and prints the counters ([`report`]) at the end of a fit.
#[cfg(test)]
#[path = "kernel_jit_fallback_tests.rs"]
mod fallback_tests;
#[path = "kernel_jit_fixed.rs"]
mod fixed;
#[path = "kernel_jit_lanes.rs"]
mod lanes;
#[path = "kernel_jit_loop.rs"]
mod ode_loop;
#[cfg(test)]
#[path = "kernel_jit_plan_tests.rs"]
mod plan_tests;
#[path = "kernel_jit_simd.rs"]
mod simd;
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
    Ode(ode_loop::Log),
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

/// A `Debug` rendering hashed as it is written, with no `String` built (#2786: the
/// fingerprints run on every plan lookup).
impl std::fmt::Write for Fx {
    #[inline]
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        std::hash::Hasher::write(self, s.as_bytes());
        Ok(())
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
    cse: HashMap<Key, u32, FxBuild>,
    /// Why the recording failed, and whether every subject of the model would (a code path
    /// the recorder cannot take) rather than this one at this point.
    failed: Option<(String, bool)>,
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
    /// The plans this recording serves ([`recorded`]): it keeps its hash-consing decisions
    /// (`trace`), from which a template of it can replay another subject's recording.
    family: Option<SubjectKey>,
    trace: Vec<Step>,
    /// Replaying ([`recorded`]): the template whose decisions it takes, the next one, and the
    /// family's other templates.
    replay: Option<Arc<Template>>,
    at: usize,
    alts: Arc<Vec<Arc<Template>>>,
}

/// A hash-consing key: tag, operands, immediate.
type Key = (u8, u32, u32, u64);
/// One hash-consing decision of a recording: the key, its register, and whether it was made
/// there (a miss) rather than shared.
type Step = (Key, u32, bool);

impl Rec {
    /// The register an earlier node with key `k` defined, where hash-consing shares one: the
    /// map's answer, or — replaying a template — the template's, its key checked (a literal's
    /// is the map's either way: its key holds the subject's value). A key the template did not
    /// meet next moves the replay to another of the family's templates that made every
    /// decision so far alike and meets it, or ends it.
    #[inline]
    fn shared(&mut self, k: Key) -> Option<u32> {
        if k.0 != T_K {
            if let Some(t) = &self.replay {
                let at = self.at;
                let meets = |o: &Template| o.trace.get(at).is_some_and(|e| e.0 == k);
                if !meets(t) {
                    let next = self
                        .alts
                        .iter()
                        .find(|o| meets(o) && o.trace.get(..at) == t.trace.get(..at))
                        .cloned();
                    if next.is_none() {
                        self.failed
                            .get_or_insert(("another structure than its template".into(), false));
                    }
                    self.replay = next;
                }
            }
            if let Some(t) = &self.replay {
                let (_, d, made) = t.trace[self.at];
                self.at += 1;
                return (!made).then_some(d);
            }
        }
        let hit = self.cse.get(&k).copied();
        if let Some(d) = hit.filter(|_| k.0 != T_K && self.family.is_some()) {
            self.trace.push((k, d, false));
        }
        hit
    }
    /// `k` made register `d` ([`Self::shared`]'s miss).
    #[inline]
    fn made(&mut self, k: Key, d: u32) {
        if k.0 != T_K {
            if self.replay.is_some() {
                return;
            }
            if self.family.is_some() {
                self.trace.push((k, d, true));
            }
        }
        self.cse.insert(k, d);
    }
    /// Register `r`'s recorded value.
    #[inline]
    fn store(&mut self, r: u32, v: f64) {
        let r = r as usize;
        if self.vals.len() <= r {
            self.vals.resize(r + 1, f64::NAN);
        }
        self.vals[r] = v;
    }
}

thread_local! {
    static REC: RefCell<Option<Rec>> = const { RefCell::new(None) };
    /// The next ODE segment's span (`PkNum::span_hint`): each end's register, value, and
    /// whether it depends on the inputs.
    static SPAN: std::cell::Cell<Option<(u32, f64, bool, u32, f64, bool)>> =
        const { std::cell::Cell::new(None) };
    /// The last recording's emptied hash-consing maps ([`begin`] takes them).
    #[allow(clippy::type_complexity)]
    static SPARE: std::cell::Cell<(
        HashMap<Key, u32, FxBuild>,
        std::collections::HashSet<(u32, bool), FxBuild>,
    )> = std::cell::Cell::new(Default::default());
}

fn rec<R>(f: impl FnOnce(&mut Rec) -> R) -> R {
    REC.with(|r| {
        f(r.borrow_mut()
            .as_mut()
            .expect("a kernel_jit recording is open"))
    })
}

/// Fail the recording on a code path the recorder cannot take: every subject would.
fn fail(why: String) {
    rec(|r| {
        r.failed.get_or_insert((why, true));
    })
}

/// Fail the recording at this subject's point only (a record-time run that failed).
fn fail_here(why: &str) {
    rec(|r| {
        r.failed.get_or_insert((why.into(), false));
    })
}

/// One register: the instruction `mk` builds for it, shared with an identical earlier
/// one when `key` matches.
fn node(key: Option<Key>, mk: impl FnOnce(u32) -> Flat) -> u32 {
    rec(|r| {
        if let Some(k) = key {
            if let Some(x) = r.shared(k) {
                return x;
            }
        }
        let d = r.n_regs;
        if d >= u32::MAX - 2 {
            r.failed.get_or_insert(("register file full".into(), false));
            return 0;
        }
        r.n_regs += 1;
        r.code.push(mk(d));
        if let Some(k) = key {
            r.made(k, d);
        }
        d
    })
}

/// [`node`] and its recorded value ([`Tr::mk`]) in one borrow of the recording: a dependent
/// value.
fn node_v(key: Key, v: f64, mk: impl FnOnce(u32) -> Flat) -> Tr {
    let r = rec(|r| {
        let d = match r.shared(key) {
            Some(x) => x,
            None => {
                let d = r.n_regs;
                if d >= u32::MAX - 2 {
                    r.failed.get_or_insert(("register file full".into(), false));
                    return 0;
                }
                r.n_regs += 1;
                r.code.push(mk(d));
                r.made(key, d);
                d
            }
        };
        r.store(d, v);
        d
    });
    Tr { v, r, dep: true }
}

/// Require the condition in register `c` (a dual `0`/`1`) to come out `holds` again — once:
/// a condition already required adds nothing.
fn guard_reg(c: u32, holds: bool) {
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
                x.store(r, v);
            }
        });
        Tr { v, r, dep }
    }
    /// The recorded value (tests read it without the dependence check of `val`).
    #[cfg(test)]
    pub(crate) fn recorded(self) -> f64 {
        self.v
    }
    /// A literal: an entry of the subject's pool (argument 2), so subjects whose data
    /// differ share code. Its tangents are exact zeros.
    fn k(c: f64) -> Self {
        let key = (T_K, 0, 0, c.to_bits());
        let r = rec(|r| {
            let d = match r.cse.get(&key) {
                Some(&x) => x,
                None => {
                    let d = r.n_regs;
                    if d >= u32::MAX - 2 {
                        r.failed.get_or_insert(("register file full".into(), false));
                        0
                    } else {
                        r.n_regs += 1;
                        let idx = r.pool.len() as u32;
                        r.pool.push(c);
                        r.code.push(Flat::LoadSeed { d, k: idx });
                        r.cse.insert(key, d);
                        d
                    }
                }
            };
            r.store(d, c);
            d
        });
        Tr {
            v: c,
            r,
            dep: false,
        }
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
        let (a, b) = (self.r, o.r);
        node_v((tag, a, b, 0), v, |d| mk(d, a, b))
    }
    fn scalar(self, c: f64, tag: u8, v: f64, mk: fn(u32, u32, f64) -> Flat) -> Self {
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r;
        node_v((tag, a, 0, c.to_bits()), v, |d| mk(d, a, c))
    }
    fn unary(self, tag: u8, op: Unary, v: f64) -> Self {
        if !self.dep {
            return Self::k(v);
        }
        let a = self.r;
        node_v((tag, a, 0, 0), v, |d| Flat::Unary { d, op, a })
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
    fn jet_k_node(self, f: crate::sens::num::JetFn, k: u8) -> Self {
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
        let v = crate::ode::linear::exprel_k(self.v, k as usize);
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
    pub(super) fn guard_cmp(a: Self, cc: ir::Cc, b: Self) -> bool {
        let out = cc.holds(a.v, b.v);
        if a.dep || b.dep {
            let (x, y) = (a.r as u32, b.r as u32);
            let key = (T_CMP + cc as u8, a.r, b.r, 0);
            let c = node(Some(key), |d| Flat::Cmp { d, cc, a: x, b: y });
            // Its value too: a body recorded later may share this condition (hash-consing)
            // and read it as a capture.
            Self::mk(out as u8 as f64, c, true);
            guard_reg(c, out);
        }
        out
    }
    pub(crate) fn reg(self) -> u32 {
        self.r
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
    /// `x · (−1)` is IEEE negation exactly (the forward dual's `mul_c`, lane by lane).
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
    fn br_cmp(self, cc: crate::sens::num::BranchCmp, other: Self) -> bool {
        Self::guard_cmp(self, cc_of(cc), other)
    }
    fn br_cmp_c(self, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        Self::guard_cmp(self, cc_of(cc), Self::k(c))
    }
    const RECORDS_BRANCHES: bool = true;
    fn br_shift_cmp(self, base: f64, cc: crate::sens::num::BranchCmp, c: f64) -> bool {
        // `base + self`, recorded as the value path computes it, then the comparison guarded.
        let s = Self::k(base) + self;
        Self::guard_cmp(s, cc_of(cc), Self::k(c))
    }
    fn br_abs_lt(self, tol: f64) -> bool {
        let out = self.v.abs() < tol;
        if self.dep {
            let a = self.abs();
            Self::guard_cmp(a, ir::Cc::Lt, Self::k(tol));
        }
        out
    }
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
    fn var(x: f64, _dim: usize) -> Self {
        fail("var".into());
        Self::k(x)
    }
    fn val(self) -> f64 {
        if self.dep {
            fail("value read".into());
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
    fn br_value(self) -> f64 {
        if self.dep {
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
        self.jet_k_node(f, 0)
    }
    fn jet_k(self, f: crate::sens::num::JetFn, k: u8) -> Self {
        self.jet_k_node(f, k)
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
    #[allow(clippy::too_many_arguments)]
    fn rhs_jacobian(
        rhs: &dyn Fn(&[Self], &[Self], Self, &mut [Self]),
        u: &[Self],
        params: &[Self],
        t: Self,
        f: &mut [Self],
        jac: &mut [Self],
        ft: &mut [Self],
    ) -> bool {
        jacobian(rhs, u, params, t, f, jac, ft);
        true
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
    /// the forward dual's `guard_floor`: `!(v ≥ lo)` (a `NaN` included) is the constant `lo`. The test
    /// is a value branch, so it is recorded as a guard and the taken arm is kept.
    fn guard_floor(self, lo: f64) -> Self {
        let keep = Self::guard_cmp(self, ir::Cc::Ge, Self::k(lo));
        if keep {
            self
        } else {
            Self::k(lo)
        }
    }
    fn jets_finite(self) -> bool {
        if self.dep {
            fail("jets_finite".into());
        }
        true
    }
    fn jet_max_abs(self) -> f64 {
        if self.dep {
            fail("jet_max_abs".into());
        }
        0.0
    }
    fn lift2(self, v: f64, _d1: f64, _d2: f64) -> Self {
        if self.dep {
            fail("lift2".into());
        }
        Self::k(v)
    }
    fn lift3(self, v: f64, _d1: f64, _d2: f64, _d3: f64) -> Self {
        if self.dep {
            fail("lift3".into());
        }
        Self::k(v)
    }
    fn lift4(self, v: f64, _d1: f64, _d2: f64, _d3: f64, _d4: f64) -> Self {
        if self.dep {
            fail("lift4".into());
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

/// Exact derivatives of a small function of its inputs alone, recorded on the spot (no cache:
/// mixing probabilities, a Jacobian's log-determinant). Per output, its value, gradient and —
/// `second` — row-major Hessian: one forward along every input direction and a reverse seeded
/// on each output's tangent lanes, whose value lanes are the Hessian's rows and tangent lanes
/// the gradient. `None` where `f` declines or the recording fails; its caller then takes finite
/// differences, so the failure is noted against `model`'s fit for every subject
/// ([`note_model_failure`]).
#[allow(clippy::type_complexity)]
pub(crate) fn derivs(
    model: &crate::types::CompiledModel,
    x: &[f64],
    n_out: usize,
    second: bool,
    f: &dyn Fn(&[Tr]) -> Option<Vec<Tr>>,
) -> Option<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    derivs_of(x, n_out, second, f)
        .map_err(|why| note_model_failure(model, &format!("a θ-only recording: {why}")))
        .ok()
}

#[allow(clippy::type_complexity)]
fn derivs_of(
    x: &[f64],
    n_out: usize,
    second: bool,
    f: &dyn Fn(&[Tr]) -> Option<Vec<Tr>>,
) -> Result<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
    let n = x.len();
    begin();
    let y = f(&x
        .iter()
        .enumerate()
        .map(|(j, &v)| Tr::theta(j, v))
        .collect::<Vec<_>>());
    let y = y
        .filter(|y| y.len() == n_out)
        .map(|y| y.iter().map(|v| v.reg()).collect());
    let plan = end(y)
        .map_err(|e| e.0)?
        .plan(Layout::F64, 0, n)
        .ok_or("the recording could not be planned")?;
    let t = if second { n } else { 0 };
    let layout = Layout {
        tan: u8::try_from(t).map_err(|_| "too many inputs for one forward")?,
        ..Layout::F64
    };
    let l = layout.lanes();
    let mut p = vec![0.0; n * l];
    for i in 0..n {
        p[i * l] = x[i];
        if second {
            p[i * l + 1 + i] = 1.0;
        }
    }
    let mut du = vec![0.0; (n_out + 1) * l];
    let fwd = plan
        .forward_layout(layout, &[], &p, &mut du)
        .ok_or("the forward failed")?;
    // Seed `(o, j)`: output `o`'s tangent lane `j` (its value lane without tangents).
    let per = t.max(1);
    let mut du_bar = vec![0.0; n_out * per * n_out * l];
    for o in 0..n_out {
        for j in 0..per {
            du_bar[(o * per + j) * n_out * l + o * l + usize::from(second) * (1 + j)] = 1.0;
        }
    }
    let mut pb = vec![0.0; n_out * per * n * l];
    if !plan.reverse_from(&fwd, layout, n_out * per, &mut [], &mut pb, &du_bar) {
        return Err("the reverse failed".into());
    }
    let row = |o: usize, j: usize, lane: usize| -> Vec<f64> {
        (0..n)
            .map(|i| pb[(o * per + j) * n * l + i * l + lane])
            .collect()
    };
    let vals = (0..n_out).map(|o| du[o * l]).collect();
    let grad = (0..n_out).map(|o| row(o, 0, usize::from(second))).collect();
    let hess = (0..n_out)
        .map(|o| {
            if second {
                (0..n).flat_map(|j| row(o, j, 0)).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let out = (vals, grad, hess);
    let finite = |v: &Vec<Vec<f64>>| v.iter().flatten().all(|x| x.is_finite());
    if finite(&out.1) && finite(&out.2) {
        Ok(out)
    } else {
        Err("a non-finite derivative".into())
    }
}

/// Open a recording on this thread.
pub(crate) fn begin() {
    let (cse, guarded) = SPARE.take();
    REC.with(|r| {
        *r.borrow_mut() = Some(Rec {
            cse,
            guarded,
            ..Rec::default()
        })
    });
    SPAN.with(|s| s.set(None));
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
    tie: Tie,
}

/// A recording's tie to the templates ([`recorded`]): the plans it serves, and either what a
/// template of it replays (its code and marks before [`live_code`], and its hash-consing
/// decisions) or — replayed — the template it reproduced.
#[derive(Default)]
pub(crate) struct Tie {
    family: Option<SubjectKey>,
    raw: Vec<Flat>,
    raw_marks: Vec<usize>,
    trace: Vec<Step>,
    replayed: Option<Arc<Template>>,
}

/// Close the recording with `outs` as the outputs, or the reason it cannot be compiled.
/// The recording, or why it failed and whether that is the model's code path (every
/// subject would fail the same way).
pub(crate) fn end(outs: Option<Vec<u32>>) -> Result<Recording, (String, bool)> {
    STATS.records.fetch_add(1, Relaxed);
    let mut r = REC
        .with(|r| r.borrow_mut().take())
        .expect("a kernel_jit recording is open");
    // The next recording reuses the maps' tables: no growing and rehashing them again.
    r.cse.clear();
    r.guarded.clear();
    SPARE.set((std::mem::take(&mut r.cse), std::mem::take(&mut r.guarded)));
    #[cfg(test)]
    if FORCE_PLAN_FAIL.get() {
        return Err((FORCED_FAILURE.to_string(), false));
    }
    if let Some(f) = r.failed {
        return Err(f);
    }
    let outs = outs.ok_or_else(|| ("the recorder returned None".to_string(), false))?;
    if let Some(t) = r.replay.take() {
        // Every decision the template's, and every instruction, output, guard and mark as its:
        // the recording a fresh run would make, its literals and loop calls aside.
        let p = &t.rec;
        let shape = |c: &CallKind| (c.n_in(), c.n_out());
        let same = r.at == t.trace.len()
            && (r.n_regs, &outs, r.guard, &r.marks) == (p.n_regs, &p.outs, p.guard, &t.raw_marks)
            && (&r.eta_regs, &r.theta_regs) == (&p.eta_regs, &p.theta_regs)
            && (r.ode, r.declined, p.dense) == (p.ode, p.declined, false)
            && r.loops.iter().map(shape).eq(p.loops.iter().map(shape))
            && r.code.len() == t.raw.len()
            && r.code.iter().zip(&t.raw).all(|(a, b)| a.same(b));
        if !same {
            return Err(("another structure than its template".into(), false));
        }
        return Ok(Recording {
            code: Vec::new(),
            n_regs: r.n_regs,
            outs,
            guard: r.guard,
            pool: r.pool,
            vals: Vec::new(),
            marks: Vec::new(),
            eta_regs: r.eta_regs,
            theta_regs: r.theta_regs,
            loops: r.loops,
            ode: r.ode,
            dense: false,
            declined: r.declined,
            tie: Tie {
                family: r.family,
                replayed: Some(t),
                ..Tie::default()
            },
        });
    }
    let (code, marks) = live_code(&r.code, &r.marks, r.n_regs as usize, &outs, r.guard);
    // A template made of it replays from the code as recorded.
    let tie = match r.family {
        Some(f) => Tie {
            family: Some(f),
            raw: r.code,
            raw_marks: r.marks,
            trace: r.trace,
            replayed: None,
        },
        None => Tie::default(),
    };
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
        tie,
    })
}

/// The plans of `family` recorded by `f` (which opens nothing and returns the outputs): by
/// replaying the template the family's last recording was planned from, where one is known —
/// the same hash-consing decisions taken without the table, the code checked against the
/// template's as recorded, and no [`live_code`] or template lookup at the end, since the plan
/// is that template's with this recording's literals and loop calls — else, and wherever the
/// subject turns out to have another structure, by recording afresh.
pub(crate) fn recorded(
    family: &SubjectKey,
    f: impl Fn() -> Option<Vec<u32>>,
) -> Result<Recording, (String, bool)> {
    #[cfg(test)]
    let off = TEMPLATE_PROBE.with(|p| p.get().0);
    #[cfg(not(test))]
    let off = false;
    let alts = (!off)
        .then(|| lock_families().get(family).cloned())
        .flatten();
    if let Some(alts) = alts.filter(|a| !a.is_empty()) {
        begin();
        rec(|r| {
            r.family = Some(family.clone());
            r.replay = Some(alts[0].clone());
            r.alts = alts;
        });
        if let Ok(r) = end(f()) {
            STATS.replayed.fetch_add(1, Relaxed);
            #[cfg(test)]
            REPLAYS.with(|c| c.set(c.get() + 1));
            return Ok(r);
        }
    }
    begin();
    rec(|r| r.family = Some(family.clone()));
    end(f())
}

/// Per family of plans ([`SubjectKey::family`]), the templates its recordings were planned
/// from, the latest first: what [`recorded`] replays.
type Families = HashMap<SubjectKey, Arc<Vec<Arc<Template>>>, FxBuild>;
fn lock_families() -> std::sync::MutexGuard<'static, Families> {
    static F: OnceLock<Mutex<Families>> = OnceLock::new();
    F.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `code` without the instructions nothing reads — not an output, not the guard, nor an
/// instruction that is itself kept — with the chunk marks moved to match. Every instruction is
/// pure, so dropping one changes no result; what it drops is mostly a call `affine_parts`
/// made only to differentiate. The inputs' loads stay: the plan lays slots out from them.
fn live_code(
    code: &[Flat],
    marks: &[usize],
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
    let mut marks: Vec<usize> = marks.iter().map(|m| before[(*m).min(code.len())]).collect();
    marks.dedup();
    let code = code
        .iter()
        .zip(keep)
        .filter_map(|(f, k)| k.then_some(*f))
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
    let sweep = tangents(start);
    for j in 0..n {
        let tan = sweep(u[j].r as u32);
        for i in 0..n {
            a[i * n + j] = tangent_of(&tan, f[i]);
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
        fail_here("the right-hand side is not affine in the state");
        return false;
    }
    true
}

/// The forward tangents of everything the recording has written since instruction `start`:
/// the sweep along one input register (the tangent of every value that depends on it), and
/// the region's values.
fn tangents(start: usize) -> impl Fn(u32) -> HashMap<u32, Tr, FxBuild> {
    let region = rec(|r| r.code[start..].to_vec());
    // The values are read from the open recording when asked: copying them all here made a
    // stiff segment's Jacobians quadratic in the recording. A register the region reads was
    // written before this call, and a register's value is never rewritten.
    let h = move |r: u32| {
        let v = rec(|rc| rc.vals.get(r as usize).copied().unwrap_or(f64::NAN));
        Tr::mk(v, r, true)
    };
    move |seed: u32| {
        let mut tan: HashMap<u32, Tr, FxBuild> = HashMap::default();
        tan.insert(seed, Tr::k(1.0));
        for g in &region {
            let Some(d) = def(g) else { continue };
            if let Some(t) = tangent(g, &h, &tan) {
                tan.insert(d, t);
            }
        }
        tan
    }
}

/// `x`'s tangent in `tan` (an exact zero where it has none).
fn tangent_of(tan: &HashMap<u32, Tr, FxBuild>, x: Tr) -> Tr {
    if x.dep {
        tan.get(&(x.r as u32))
            .copied()
            .unwrap_or_else(|| Tr::k(0.0))
    } else {
        Tr::k(0.0)
    }
}

/// `f(u, t)`, `J = ∂f/∂u` (row-major) and `f_t = ∂f/∂t` of a recorded right-hand side,
/// exactly (#2786): one call on fresh copies of the state and the clock, then its recording
/// differentiated forward along each of them. `J` and `f_t` are recorded expressions, so the
/// reverse pass differentiates them; the copies are linked to `u` and `t`, so everything
/// downstream still depends on them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jacobian(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]),
    u: &[Tr],
    params: &[Tr],
    t: Tr,
    f: &mut [Tr],
    jac: &mut [Tr],
    ft: &mut [Tr],
) {
    let n = u.len();
    let var = |x: Tr| {
        let r = if x.dep {
            let a = x.r as u32;
            node(None, |d| Flat::AddK { d, a, c: 0.0 })
        } else {
            node(None, |d| Flat::LoadK { d, c: x.v })
        };
        Tr::mk(x.v, r, true)
    };
    let uv: Vec<Tr> = u.iter().map(|&x| var(x)).collect();
    let tv = var(t);
    let start = rec(|r| r.code.len());
    rhs(&uv, params, tv, f);
    let sweep = tangents(start);
    for j in 0..n {
        let tan = sweep(uv[j].r as u32);
        for i in 0..n {
            jac[i * n + j] = tangent_of(&tan, f[i]);
        }
    }
    let tan = sweep(tv.r as u32);
    for i in 0..n {
        ft[i] = tangent_of(&tan, f[i]);
    }
}

/// The tangent of instruction `g`'s result, given its operands' (`tan`, absent = zero), or
/// `None` when it is zero. `h` rebuilds a register as a recorded value.
fn tangent(g: &Flat, h: &impl Fn(u32) -> Tr, tan: &HashMap<u32, Tr, FxBuild>) -> Option<Tr> {
    use crate::sens::num::BranchCmp;
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
                // `aᵉ·ln a`, defined for `a > 0`; `0^e` is flat in `e` there. A selection, not
                // a branch (see `Unary::Abs` below); `ln` reads `1` off the domain, so the arm
                // not taken stays finite and its reverse adjoint zero.
                let pos = |x: Tr, y: Tr| Tr::select_if(BranchCmp::Gt, h(a), Tr::k(0.0), x, y);
                pos(h(d) * pos(h(a), Tr::k(1.0)).ln() * y, Tr::k(0.0))
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
                // The interpreter's floor: flat below it, the divisor read as `1` there so the
                // arm not taken stays finite.
                Unary::Ln => {
                    let above = |p: Tr, q: Tr| Tr::select_if(BranchCmp::Ge, v, Tr::k(1e-30), p, q);
                    above(x / above(v, Tr::k(1.0)), Tr::k(0.0))
                }
                // Flat at a non-positive argument (#2892), the divisor read as `1` there so the
                // arm not taken stays finite.
                Unary::Sqrt | Unary::SqrtRhs => {
                    let pos = |p: Tr, q: Tr| Tr::select_if(BranchCmp::Gt, v, Tr::k(0.0), p, q);
                    pos(x / (pos(q, Tr::k(1.0)) * Tr::k(2.0)), Tr::k(0.0))
                }
                Unary::InvLogit => q * (Tr::k(1.0) - q) * x,
                Unary::Logit => x / (v * (Tr::k(1.0) - v)),
                Unary::Cos => -(v.sin() * x),
                Unary::Sin => v.cos() * x,
                Unary::Acos => -(x / (Tr::k(1.0) - v * v).sqrt()),
                Unary::LnGamma => v.digamma() * x,
                Unary::Digamma => v.trigamma() * x,
                Unary::Trigamma => v.tetragamma() * x,
                // `sign(v)·ẋ` as a selection, not a branch: a recorded branch is a guard, and a
                // guard in a tangent fails the whole call when `v` changes sign — inside an ODE
                // loop body (a Rosenbrock step's exact `J`, `jacobian`) that is every
                // integration crossing the kink, which then declines the subject outright.
                Unary::Abs => Tr::select_if(BranchCmp::Ge, v, Tr::k(0.0), x, -x),
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
                    Unary::SqrtRhs => (if v(a) >= 0.0 { v(a) } else { 0.0 }).sqrt(),
                    Unary::InvLogit => 1.0 / (1.0 + (-v(a)).exp()),
                    Unary::Logit => (v(a) / (1.0 - v(a))).ln(),
                    Unary::Cos => v(a).cos(),
                    Unary::Acos => v(a).acos(),
                    Unary::LnRaw => v(a).ln(),
                    Unary::Sin => v(a).sin(),
                    Unary::Digamma => crate::stats::special::digamma(v(a)),
                    Unary::Trigamma => crate::stats::special::trigamma(v(a)),
                    Unary::Tetragamma => crate::stats::special::tetragamma(v(a)),
                    Unary::ExpRel(k) => crate::ode::linear::exprel_k(v(a), k as usize),
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

/// The exact content behind a key's hash (#2786): the bytes a value's `Hash` writes
/// ([`Canon`]). The hash picks the bucket; equality compares the bytes in full, so no collision
/// ever serves one subject (layout, configuration) another's plan.
#[derive(Clone, Debug)]
pub(crate) struct Desc {
    hash: u64,
    bytes: Arc<[u8]>,
}

impl Desc {
    /// The bytes `f` writes.
    fn of(f: impl FnOnce(&mut Canon)) -> Self {
        let mut c = Canon(Vec::new());
        f(&mut c);
        let mut h = Fx::default();
        std::hash::Hasher::write(&mut h, &c.0);
        #[cfg(test)]
        let hash = if COLLIDE.with(std::cell::Cell::get) {
            0
        } else {
            h.0
        };
        #[cfg(not(test))]
        let hash = h.0;
        Desc {
            hash,
            bytes: c.0.into(),
        }
    }
    /// What `v`'s `Hash` writes.
    pub(crate) fn of_value(v: &impl std::hash::Hash) -> Self {
        Self::of(|c| v.hash(c))
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl PartialEq for Desc {
    fn eq(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.bytes, &o.bytes) || (self.hash == o.hash && self.bytes == o.bytes)
    }
}
impl Eq for Desc {}
impl std::hash::Hash for Desc {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.hash.hash(h);
    }
}

/// A `Hasher` that keeps what is written: a value's canonical bytes ([`Desc`]).
pub(crate) struct Canon(Vec<u8>);

impl std::hash::Hasher for Canon {
    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
    fn finish(&self) -> u64 {
        unreachable!("a canonical descriptor is compared, not hashed")
    }
}

impl std::fmt::Write for Canon {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Every [`Desc`] and template shape made on this thread hashes to `0`: a test puts distinct
    /// keys in one bucket and checks that equality alone tells them apart.
    pub(crate) static COLLIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A subject of a model at a width. A recording bakes the subject's data in (its times, doses,
/// covariates), so the subject is identified by that data ([`subject_desc`]), not by an
/// address a later subject — a per-occasion or cut copy built and dropped mid-fit — may reuse.
/// The model is identified by its [`crate::types::PlanId`]. Every field is compared exactly.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct SubjectKey {
    /// The registered fit whose model this is ([`register`]; `0`: none): every fit's plans key
    /// apart and leave with it, so no fit ever clears another's.
    scope: u64,
    /// The model's identity ([`crate::types::PlanId`]): minted once per constructed model, never
    /// reused, so another model built at a dropped one's address keys apart by construction.
    model: u64,
    /// The solver options, the one part of a model a caller may change in place between uses.
    solver: Desc,
    subject: Desc,
    /// A likelihood kernel's observed values ([`Self::with_observed`]).
    observed: Option<Desc>,
    /// The mixture class `MIXNUM` resolves to on this thread: a recording takes its branch.
    class: usize,
    /// The stacked `[η, κ]` layout the recording was made under.
    layout: Option<Desc>,
    n: usize,
    n_theta: usize,
    /// What the plan computes: the predictions (`0`) or a subject term ([`Self::with_kind`]).
    kind: u8,
    /// The computation's own configuration ([`Self::with_variant`]): a likelihood kernel's
    /// objective, width and parts.
    variant: Option<Desc>,
}

impl SubjectKey {
    /// The plan recorded under the subject's stacked IOV layout `lay`.
    pub(crate) fn with_layout(
        mut self,
        lay: Option<&crate::stats::stacked_layout::StackedLayout>,
    ) -> Self {
        self.layout = lay.map(|l| Desc::of(|c| l.canon(c)));
        self
    }
    /// The plan of the subject's observed values too (a likelihood kernel bakes them in).
    pub(crate) fn with_observed(mut self, subject: &crate::types::Subject) -> Self {
        let addr = subject as *const _ as usize;
        self.observed =
            Some(registered(addr).map_or_else(|| observed_desc(subject), |e| e.0[1].clone()));
        self
    }
    /// The plans of every subject of this computation ([`recorded`]): the key without the
    /// subject's data or the fit's scope.
    pub(crate) fn family(&self) -> Self {
        SubjectKey {
            scope: 0,
            subject: Desc::of(|_| {}),
            observed: None,
            ..self.clone()
        }
    }
    /// The same subject's plan of another computation (`1`: its non-Gaussian term).
    pub(crate) fn with_kind(mut self, kind: u8) -> Self {
        self.kind = kind;
        self
    }
    /// The plan of one configuration of the computation.
    pub(crate) fn with_variant(mut self, variant: Desc) -> Self {
        self.variant = Some(variant);
        self
    }
    pub(crate) fn new(
        model: &crate::types::CompiledModel,
        subject: &crate::types::Subject,
        n: usize,
        n_theta: usize,
    ) -> Self {
        let m = registered(model as *const _ as usize);
        SubjectKey {
            scope: m.as_ref().map_or(0, |e| e.1),
            model: model.plan_id.get(),
            solver: m.map_or_else(|| solver_desc(model), |e| e.0[0].clone()),
            subject: registered(subject as *const _ as usize)
                .map_or_else(|| subject_desc(subject), |e| e.0[0].clone()),
            observed: None,
            class: crate::parser::model_parser::current_mixture_class(),
            layout: None,
            n,
            n_theta,
            kind: 0,
            variant: None,
        }
    }
}

/// A fit's registration: its model and subjects, fingerprinted once — [`SubjectKey::new`]
/// reads them by address instead of hashing on every lookup — and a scope its plans are keyed
/// by. The scope borrows what it registered, so nothing it covers can move or change
/// meanwhile; when the last user of a model's scope leaves, that scope's plans are dropped.
/// Anything unregistered (a temporary subject, a test's direct call) is hashed per call and
/// keyed under scope `0`, whose plans leave with the model's id ([`drop_model`]).
pub(crate) struct Prints<'a> {
    addrs: Vec<usize>,
    _borrow: std::marker::PhantomData<&'a ()>,
}

/// Per address: (descriptors — a model's solver options; a subject's design and observed
/// values —, live registrations, scope).
type PrintMap = HashMap<usize, (Arc<[Desc; 2]>, usize, u64), FxBuild>;
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static PRINTS: RwLock<Option<Arc<PrintMap>>> = RwLock::new(None);
static PRINTS_VERSION: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static PRINTS_SEEN: RefCell<(usize, Option<Arc<PrintMap>>)> = const { RefCell::new((0, None)) };
}

/// Register `model` and `subjects` for the scope's lifetime (reference-counted: concurrent
/// fits of one model share its entry and its scope).
pub(crate) fn register<'a>(
    model: &'a crate::types::CompiledModel,
    subjects: &'a [crate::types::Subject],
) -> Prints<'a> {
    let none = Desc::of(|_| {});
    let mut add = vec![(model as *const _ as usize, [solver_desc(model), none])];
    add.extend(
        subjects
            .iter()
            .map(|s| (s as *const _ as usize, [subject_desc(s), observed_desc(s)])),
    );
    edit_prints(|m| {
        let scope = m
            .get(&add[0].0)
            .map_or_else(|| NEXT_SCOPE.fetch_add(1, Relaxed), |e| e.2);
        for (a, d) in &add {
            m.entry(*a).or_insert((Arc::new(d.clone()), 0, scope)).1 += 1;
        }
    });
    Prints {
        addrs: add.into_iter().map(|(a, _)| a).collect(),
        _borrow: std::marker::PhantomData,
    }
}

impl Drop for Prints<'_> {
    fn drop(&mut self) {
        let mut ended = None;
        edit_prints(|m| {
            for (i, a) in self.addrs.iter().enumerate() {
                if let Some(e) = m.get_mut(a) {
                    e.1 -= 1;
                    if e.1 == 0 {
                        // The model's entry is first: its scope ends with its last user.
                        if i == 0 {
                            ended = Some(e.2);
                        }
                        m.remove(a);
                    }
                }
            }
        });
        if let Some(scope) = ended {
            drop_scope(scope);
        }
    }
}

fn edit_prints(f: impl FnOnce(&mut PrintMap)) {
    let mut g = PRINTS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut m: PrintMap = g.as_deref().cloned().unwrap_or_default();
    f(&mut m);
    *g = (!m.is_empty()).then(|| Arc::new(m));
    PRINTS_VERSION.fetch_add(1, Relaxed);
}

/// The registry's entry at `addr`: the thread re-reads the registry only when its version
/// moved, so a lookup takes no lock.
fn registered(addr: usize) -> Option<(Arc<[Desc; 2]>, u64)> {
    let v = PRINTS_VERSION.load(Relaxed);
    PRINTS_SEEN.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != v {
            let g = PRINTS
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *c = (v, g.clone());
        }
        c.1.as_ref()
            .and_then(|m| m.get(&addr))
            .map(|e| (e.0.clone(), e.2))
    })
}

/// The integrator's options of `m`, as canonical bytes: part of the recorded computation (the step bodies,
/// the accepted-step control and the stage count all come from the method and its tolerances), so
/// a plan recorded under RK45 is not the plan of the same model under Rodas5P. Everything else a
/// plan reads of a model is fixed at construction, which [`crate::types::PlanId`] identifies.
fn solver_desc(m: &crate::types::CompiledModel) -> Desc {
    use std::hash::Hash;
    Desc::of(|h| {
        if let Some(o) = m.ode_spec.as_ref().map(|ode| &ode.solver_opts) {
            for x in [o.abstol, o.reltol, o.initial_dt, o.min_dt] {
                x.to_bits().hash(h);
            }
            let method = std::mem::discriminant(&o.method);
            (o.max_steps, method, o.stiff_abort_after, o.auto_switch).hash(h);
        }
    })
}

/// Everything of `s` a recording can read, as its canonical bytes: its records' times, doses,
/// compartments, occasions and covariate values. A covariate map is written as its entries
/// sorted by name, so equal maps are equal whatever their iteration order.
fn subject_desc(s: &crate::types::Subject) -> Desc {
    use std::hash::Hash;
    Desc::of(|h| {
        let f = |h: &mut Canon, v: &[f64]| {
            v.len().hash(h);
            for x in v {
                x.to_bits().hash(h);
            }
        };
        let maps = |h: &mut Canon, m: &[HashMap<String, f64>]| {
            m.len().hash(h);
            for c in m {
                let mut e: Vec<(&String, u64)> = c.iter().map(|(k, v)| (k, v.to_bits())).collect();
                e.sort_unstable();
                e.hash(h);
            }
        };
        f(h, &s.obs_times);
        f(h, &s.obs_raw_times);
        f(h, &s.pk_only_times);
        f(h, &s.pk_only_raw_times);
        f(h, &s.reset_times);
        s.obs_cmts.hash(h);
        s.cens.hash(h);
        s.occasions.hash(h);
        s.obs_rgroup.hash(h);
        s.dose_occasions.hash(h);
        s.reset_occasions.hash(h);
        s.pk_only_occasions.hash(h);
        s.fremtype.hash(h);
        for (a, b) in &s.reset_period_times {
            a.to_bits().hash(h);
            b.to_bits().hash(h);
        }
        for t in &s.cmt_toggles {
            (t.pk_only, t.state, t.on).hash(h);
        }
        s.doses.len().hash(h);
        for d in &s.doses {
            for x in [d.time, d.amt, d.rate, d.duration, d.ii] {
                x.to_bits().hash(h);
            }
            (d.ss, d.cmt_idx()).hash(h);
            std::mem::discriminant(&d.rate_mode).hash(h);
            std::mem::discriminant(&d.infusion_def).hash(h);
        }
        maps(h, std::slice::from_ref(&s.covariates));
        maps(h, &s.dose_covariates);
        maps(h, &s.obs_covariates);
        maps(h, &s.pk_only_covariates);
        maps(h, &s.reset_covariates);
        maps(h, &s.endpoint_covariates);
        // Its records' rendering (`f64` `Debug` round-trips, so it is exact), length-prefixed.
        let mut r = Canon(Vec::new());
        if !s.obs_records.is_empty() {
            use std::fmt::Write;
            let _ = write!(r, "{:?}", s.obs_records);
        }
        r.0.hash(h);
        s.simulated_mixnum.hash(h);
    })
}

/// `s`'s observed values, which a likelihood kernel bakes in beyond the design its predictions
/// read ([`SubjectKey::with_observed`]): subjects of one design share prediction plans and key
/// their kernels apart.
fn observed_desc(s: &crate::types::Subject) -> Desc {
    use std::hash::Hash;
    Desc::of(|h| {
        s.observations.len().hash(h);
        for y in &s.observations {
            y.to_bits().hash(h);
        }
        s.obs_l2.hash(h);
    })
}

/// A subject's compiled recording.
pub(crate) type Entry = Arc<Plan>;
pub(crate) type Kernels = Arc<Variants>;

/// A subject's variants and the one that served last, which callers try first and [`add`] keeps.
#[derive(Default)]
pub(crate) struct Variants {
    plans: Vec<Entry>,
    /// The variant that served last.
    mru: AtomicUsize,
}

impl std::ops::Deref for Variants {
    type Target = [Entry];
    fn deref(&self) -> &[Entry] {
        &self.plans
    }
}

impl Variants {
    /// The variants with their indices, the one that served last first: a subject's points
    /// mostly stay in one regime from call to call.
    pub(crate) fn order(&self) -> impl Iterator<Item = (usize, &Entry)> {
        let m = self.mru.load(Relaxed);
        (m..self.plans.len())
            .chain(0..m)
            .map(|i| (i, &self.plans[i]))
    }

    /// Variant `i` served.
    pub(crate) fn hit(&self, i: usize) {
        self.mru.store(i, Relaxed);
    }

    /// These variants and `plan`, which serves next; past `MAX_VARIANTS` the oldest one that
    /// did not serve last goes. Two variants whose guards both hold at a point recorded the
    /// same branches there, so which one serves does not change a bit.
    fn with(&self, plan: Entry) -> Variants {
        let mut plans = self.plans.clone();
        if plans.len() >= MAX_VARIANTS {
            plans.remove(usize::from(self.mru.load(Relaxed) == 0));
        }
        plans.push(plan);
        Variants {
            mru: AtomicUsize::new(plans.len() - 1),
            plans,
        }
    }
}

fn subjects() -> &'static RwLock<HashMap<SubjectKey, Option<Kernels>>> {
    static S: OnceLock<RwLock<HashMap<SubjectKey, Option<Kernels>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Models a recording failed on for a reason of the model's (an unsupported operation, a
/// value read), with that reason: every subject would fail the same way, so none tries again.
fn bad_models() -> &'static RwLock<HashMap<(u64, u64), String>> {
    static B: OnceLock<RwLock<HashMap<(u64, u64), String>>> = OnceLock::new();
    B.get_or_init(Default::default)
}

/// Why `k`'s model failed to record for a reason of its own code path, if it did.
pub(crate) fn model_bad(k: &SubjectKey) -> Option<String> {
    bad_models()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(k.scope, k.model))
        .cloned()
}

pub(crate) fn mark_model_bad(k: &SubjectKey, why: &str) {
    bad_models()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry((k.scope, k.model))
        .or_insert_with(|| why.to_string());
}

/// Per `(scope, model)`: the subjects (fingerprints) whose plan failed, and the first reason.
/// In theory no plan fails; one that does sends its subject to finite differences for the
/// quantity at hand, and the fit reports it once ([`plan_failures`]).
type Failures = HashMap<(u64, u64), (std::collections::HashSet<Desc>, String)>;
fn failures() -> &'static Mutex<Failures> {
    static F: OnceLock<Mutex<Failures>> = OnceLock::new();
    F.get_or_init(Default::default)
}

/// Note that `k`'s subject fell back to finite differences because its plan failed (`why`).
pub(crate) fn note_failure(k: &SubjectKey, why: &str) {
    failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry((k.scope, k.model))
        .or_insert_with(|| (Default::default(), why.to_string()))
        .0
        .insert(k.subject.clone());
}

/// The failures' entry for `model`'s fit (its [`register`]ed scope).
fn failure_key(model: &crate::types::CompiledModel) -> (u64, u64) {
    let scope = registered(model as *const _ as usize).map_or(0, |e| e.1);
    (scope, model.plan_id.get())
}

/// The failed "subject" a model-level recording notes: one every subject's derivative reads.
fn model_wide() -> Desc {
    Desc::of_value(&"model-wide θ-only recording")
}

/// Note that a derivative of the model's alone (a θ-only recording, [`derivs`]) fell back to
/// finite differences because its recording failed (`why`): it enters every subject's, so
/// the fit counts every subject ([`plan_failures`]).
pub(crate) fn note_model_failure(model: &crate::types::CompiledModel, why: &str) {
    STATS.bad.fetch_add(1, Relaxed);
    failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(failure_key(model))
        .or_insert_with(|| (Default::default(), why.to_string()))
        .0
        .insert(model_wide());
}

/// How many of `subjects` a plan failed for during the fit registered for `model`
/// ([`register`]), and the first failure's reason; `None` when none did. A model-level failure
/// ([`note_model_failure`]) counts them all.
pub(crate) fn plan_failures(
    model: &crate::types::CompiledModel,
    subjects: &[crate::types::Subject],
) -> Option<(usize, String)> {
    let f = failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (failed, why) = f.get(&failure_key(model))?;
    if failed.contains(&model_wide()) {
        return Some((subjects.len(), why.clone()));
    }
    let print = |s: &crate::types::Subject| {
        registered(s as *const _ as usize).map_or_else(|| subject_desc(s), |e| e.0[0].clone())
    };
    let k = subjects
        .iter()
        .filter(|s| failed.contains(&print(s)))
        .count();
    // A per-occasion or cut copy is a subject the population does not hold: count it anyway.
    let k = if k == 0 {
        failed.len().min(subjects.len())
    } else {
        k
    };
    Some((k, why.clone()))
}

/// Forget the plans of every subject and model in scope `scope`: under the subjects' lock, with
/// the generation bumped there, so a thread that read the new generation sees them gone.
fn drop_scope(scope: u64) {
    let mut s = subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    s.retain(|k, _| k.scope != scope);
    GENERATION.fetch_add(1, Relaxed);
    drop(s);
    bad_models()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&(sc, _), _| sc != scope);
    failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&(sc, _), _| sc != scope);
}

/// Forget the plans and bad marks of the model `id`, as its [`crate::types::PlanId`] drops.
pub(crate) fn drop_model(id: u64) {
    clear_kept();
    let mut s = subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    s.retain(|k, _| k.model != id);
    GENERATION.fetch_add(1, Relaxed);
    drop(s);
    lock_families().retain(|k, _| k.model != id);
    bad_models()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&(_, m), _| m != id);
    failures()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&(_, m), _| m != id);
}

/// Forget every unregistered plan (a test's empty cache); a registered fit's are its own.
#[cfg(test)]
pub(crate) fn reset() {
    clear_kept();
    drop_scope(0);
}

pub(crate) enum Lookup {
    Missing,
    Bad,
    Found(Kernels),
}

/// Bumped by [`drop_scope`] and [`drop_model`], under the subjects' lock: a thread's last-subject entry from an
/// earlier fit is stale. [`add`] does not bump it: it refreshes its own thread's entry, and a
/// thread still holding the shorter list only records the variant again.
static GENERATION: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The last subject this thread looked up: the inner loop asks for the same subject
    /// many times in a row, so most lookups take no lock and hash nothing.
    static LAST: RefCell<Option<(usize, SubjectKey, Kernels)>> = const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Test seam: while set, every plan this thread asks for fails ([`end`] returns
    /// [`FORCED_FAILURE`], [`lookup`] finds no cached plan), so each consumer's
    /// finite-difference fallback runs. Carried onto the worker a fit scope runs on
    /// (`FitScope::armed`), so a test sets it inside a one-worker pool.
    pub(crate) static FORCE_PLAN_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The keys marked bad under [`FORCE_PLAN_FAIL`]: kept here, not in the shared cache, so
    /// a forced failure never outlives the switch.
    static FORCED_BAD: RefCell<std::collections::HashSet<SubjectKey>> = RefCell::default();
}

/// Test seam: the subjects (by content) every plan of which fails from now on, process-wide so
/// it reaches every worker — the per-subject fallback fixture of the salvage paths, now that no
/// regular-path shape declines (#2786). Its subjects are ones no other test builds.
#[cfg(test)]
static FAILING_SUBJECTS: std::sync::Mutex<Vec<Desc>> = std::sync::Mutex::new(Vec::new());

/// Make every plan of `subject` fail ([`FAILING_SUBJECTS`]).
#[cfg(test)]
pub(crate) fn fail_plans_of(subject: &crate::types::Subject) {
    let mut f = FAILING_SUBJECTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let d = subject_desc(subject);
    if !f.contains(&d) {
        f.push(d);
    }
}

/// The reason a recording fails under [`FORCE_PLAN_FAIL`].
#[cfg(test)]
pub(crate) const FORCED_FAILURE: &str = "forced plan failure (test seam)";

pub(crate) fn lookup(k: &SubjectKey) -> Lookup {
    #[cfg(test)]
    if FAILING_SUBJECTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&k.subject)
    {
        return Lookup::Bad;
    }
    #[cfg(test)]
    if FORCE_PLAN_FAIL.get() {
        return if FORCED_BAD.with(|b| b.borrow().contains(k)) {
            Lookup::Bad
        } else {
            Lookup::Missing
        };
    }
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

/// Mark `k` as failing to plan (`why`): its callers take the finite-difference path, and the
/// fit reports the subject ([`note_failure`]).
pub(crate) fn mark_bad(k: &SubjectKey, why: &str) {
    STATS.bad.fetch_add(1, Relaxed);
    note_failure(k, why);
    #[cfg(test)]
    if FORCE_PLAN_FAIL.get() {
        FORCED_BAD.with(|b| b.borrow_mut().insert(k.clone()));
        return;
    }
    subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(k.clone(), None);
}

/// The variants a subject's cache entry keeps ([`add`]).
const MAX_VARIANTS: usize = 8;

/// The variants of `k` with `kernel` added, as stored. A caller serves from them and never looks
/// `k` up again: a model dropped on another thread may have emptied the table.
pub(crate) fn add(k: &SubjectKey, kernel: Entry) -> Kernels {
    let mut s = subjects()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let ks = match s.get(k) {
        Some(Some(ks)) => ks.with(kernel),
        _ => Variants::default().with(kernel),
    };
    let ks = Arc::new(ks);
    s.insert(k.clone(), Some(ks.clone()));
    // This thread's last-subject entry lists the new variant too.
    let gen = GENERATION.load(Relaxed);
    LAST.with(|l| *l.borrow_mut() = Some((gen, k.clone(), ks.clone())));
    ks
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
    pub(crate) records: AtomicUsize,
    /// Recordings a template planned ([`Template`]).
    pub(crate) templated: AtomicUsize,
    /// Of the recordings, those replayed from their family's template ([`recorded`]).
    pub(crate) replayed: AtomicUsize,
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
    records: AtomicUsize::new(0),
    templated: AtomicUsize::new(0),
    replayed: AtomicUsize::new(0),
};

/// One line of counters (`FERX_KJIT_DEBUG`), printed at the end of a fit.
pub(crate) fn report() {
    if debug() {
        let s = &STATS;
        eprintln!(
            "kjit: records {} templated {} replayed {} compiled {} shared {} bad {} runs {} taped {} cov {} jets2 {} guard-fails {} spot-fails {} record {:.1} ms compile {:.1} ms",
            s.records.load(Relaxed),
            s.templated.load(Relaxed),
            s.replayed.load(Relaxed),
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
            f(base, adj, idx, lits, 1, 0);
        } else {
            let f: ChunkFn = std::mem::transmute(p);
            f(base, idx, lits);
        }
    }

    /// Count `k` calls, promoting at each tier's point crossed. The count is a plain load
    /// and store, not a locked add — a lost update between threads only delays a promotion
    /// — and each tier is compiled once ([`Self::tier`]).
    #[inline]
    fn count(&self, k: usize) {
        let n0 = self.calls.load(Relaxed);
        if n0 >= TIER3_CALLS {
            return;
        }
        let n1 = n0 + k;
        self.calls.store(n1, Relaxed);
        // Only a call that crosses a tier's point promotes: between them nothing changes, and
        // the shared `want` stays untouched.
        if [TIER1_CALLS, TIER2_CALLS, TIER3_CALLS]
            .iter()
            .any(|&t| n0 < t && n1 >= t)
        {
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
            f(base, adj, idx, lits, seeds, stride);
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
    spec: Vec<(DirMask, u16)>,
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
            ir::tangent_packed(&f, layout.tan as usize, base.lanes(), false)
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
        let (f, prefix) = ir::dedup_calls(&f, prefix);
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
                            3 if !rev => *base.add(slot(n_in + j) + lane) = canonical_zero(y),
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

/// `x` with a `-0` made `+0` (`NaN` and every other value unchanged): what every tier stores as
/// a forward's output. The tiers compute a lane the folding knows to be zero differently: the
/// folded code writes the constant, the dual operations of the interpreter carry the sign
/// IEEE gives `exp(v)*(-1*0)`. A zero's sign carries no meaning, so the output is where
/// they are made to agree bit for bit.
#[inline(always)]
pub(super) fn canonical_zero(x: f64) -> f64 {
    x + 0.0
}

/// Distinct `(layout, direction)` keys a plan resolves lock-free.
const RESOLVED_SLOTS: usize = 16;

/// A plan's kernels over one key, resolved once.
struct Resolved {
    key: (Layout, bool),
    kerns: Option<Vec<Kern>>,
}

/// A plan's resolved kernels: borrowed from its slot, or owned past the slots.
enum Fns<'a> {
    Shared(&'a [Kern]),
    Own(Vec<Kern>),
}

impl std::ops::Deref for Fns<'_> {
    type Target = [Kern];
    fn deref(&self) -> &[Kern] {
        match self {
            Fns::Shared(s) => s,
            Fns::Own(v) => v,
        }
    }
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
        /// The slot of the guard's running value, when this chunk defines it: `0` there and
        /// no later call can make the plan hold, so the forward stops.
        guard: Option<u32>,
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
    /// Every call is a chunk ([`Plan::chunks_only`]).
    chunky: bool,
    /// Each `(layout, direction)`'s resolved kernels, claimed once by the first caller into
    /// a free slot and read without a lock after ([`Plan::fns`]).
    resolved: [OnceLock<Resolved>; RESOLVED_SLOTS],
    /// The solver override an ODE plan was recorded under (`None` for a plan with no ODE
    /// segment): another phase of the fit (tighter tolerances, another method) must not run it.
    ode_override: Option<Option<crate::ode::solver::OdeSolverOverride>>,
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
/// `outs` the registers it exports, in order; `spec` the inputs' structure. Shared by text
/// ([`chunk_cache`]); returns the program and where in the recording's literal pool each of
/// this chunk's literals is.
fn build_chunk(
    body: &[Flat],
    ins: &[u32],
    outs: &[u32],
    spec: Vec<(DirMask, u16)>,
    cache: &mut ChunkCache,
) -> (Arc<ChunkProgram>, Vec<u32>) {
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
    let mut lits: Vec<u32> = Vec::new();
    for f in body {
        let d = def(f);
        if let Some(d) = d {
            local.insert(d, next);
            next += 1;
        }
        let g = match *f {
            Flat::LoadSeed { k, .. } => {
                lits.push(k);
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

/// A set of η directions, and whether a value depends on any input at all ([`DirMask::FREE`]).
/// η `k < 63` is bit `k` of `lo` (bit 63 is the FREE flag); further directions spill into
/// `hi` (empty, so allocation-free, up to 63 η). `all_hi` stands for every direction past `hi`
/// (the dense set); `hi` is kept trimmed and empty under it, so equality is structural.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct DirMask {
    lo: u64,
    hi: Vec<u64>,
    all_hi: bool,
}

impl DirMask {
    const FREE_BIT: u64 = 1 << 63;
    /// No direction, not FREE.
    pub(super) const NONE: DirMask = DirMask {
        lo: 0,
        hi: Vec::new(),
        all_hi: false,
    };
    /// Depends on θ or η, but on no particular direction.
    pub(super) const FREE: DirMask = DirMask {
        lo: Self::FREE_BIT,
        hi: Vec::new(),
        all_hi: false,
    };
    /// Every direction, and FREE.
    pub(super) const ALL: DirMask = DirMask {
        lo: !0,
        hi: Vec::new(),
        all_hi: true,
    };

    /// Direction `k` alone, and FREE.
    fn eta(k: usize) -> DirMask {
        let mut m = DirMask::FREE;
        if k < 63 {
            m.lo |= 1 << k;
        } else {
            let (w, b) = ((k - 63) / 64, (k - 63) % 64);
            m.hi = vec![0; w + 1];
            m.hi[w] = 1 << b;
        }
        m
    }

    /// Whether direction `k` is live.
    pub(super) fn has(&self, k: usize) -> bool {
        if k < 63 {
            self.lo >> k & 1 == 1
        } else {
            self.hi_bit(k - 63)
        }
    }

    fn hi_bit(&self, j: usize) -> bool {
        self.all_hi || self.hi.get(j / 64).is_some_and(|x| x >> (j % 64) & 1 == 1)
    }

    /// Whether any of the first `n` directions is live.
    pub(super) fn any_below(&self, n: usize) -> bool {
        let low = if n >= 63 {
            !Self::FREE_BIT
        } else {
            (1 << n) - 1
        };
        self.lo & low != 0 || (63..n).any(|k| self.hi_bit(k - 63))
    }

    /// Whether no direction is live, whatever FREE says.
    pub(super) fn no_dirs(&self) -> bool {
        self.lo & !Self::FREE_BIT == 0 && !self.all_hi && self.hi.iter().all(|&x| x == 0)
    }

    fn trim(mut self) -> DirMask {
        if self.all_hi {
            self.hi.clear();
        }
        while self.hi.last() == Some(&0) {
            self.hi.pop();
        }
        self
    }
}

impl std::ops::BitOr<&DirMask> for DirMask {
    type Output = DirMask;
    fn bitor(mut self, o: &DirMask) -> DirMask {
        self.lo |= o.lo;
        self.all_hi |= o.all_hi;
        if self.hi.len() < o.hi.len() {
            self.hi.resize(o.hi.len(), 0);
        }
        for (a, b) in self.hi.iter_mut().zip(&o.hi) {
            *a |= b;
        }
        self.trim()
    }
}

impl std::ops::BitOrAssign<&DirMask> for DirMask {
    fn bitor_assign(&mut self, o: &DirMask) {
        *self = std::mem::take(self) | o;
    }
}

impl std::ops::BitAnd<&DirMask> for &DirMask {
    type Output = DirMask;
    fn bitand(self, o: &DirMask) -> DirMask {
        // A word past one side's `hi` is that side's `all_hi` (all ones) or zero.
        let word = |m: &DirMask, i: usize| {
            m.hi.get(i)
                .copied()
                .unwrap_or(if m.all_hi { !0 } else { 0 })
        };
        DirMask {
            lo: self.lo & o.lo,
            hi: (0..self.hi.len().max(o.hi.len()))
                .map(|i| word(self, i) & word(o, i))
                .collect(),
            all_hi: self.all_hi && o.all_hi,
        }
        .trim()
    }
}

/// Each register's live directions: the η a value can depend on (bit `k` = η `k`), and
/// [`DirMask::FREE`] when it depends on any input. A lane outside it is an exact zero, whatever the
/// inputs' values.
fn direction_masks(
    code: &[Flat],
    n_regs: usize,
    eta_regs: &[(u32, usize)],
    dense: bool,
) -> Vec<DirMask> {
    let mut m = vec![DirMask::NONE; n_regs];
    for &(r, k) in eta_regs {
        m[r as usize] = DirMask::eta(k);
    }
    // A loop call's outputs depend on every input of the call (its `LoopArg`s, which start
    // afresh at each call's first).
    let (mut loop_acc, mut in_args) = (DirMask::NONE, false);
    for f in code {
        if let Flat::LoopArg { a } = *f {
            if !in_args {
                loop_acc = DirMask::NONE;
                in_args = true;
            }
            loop_acc |= &m[a as usize];
            continue;
        }
        in_args = false;
        let Some(d) = def(f) else { continue };
        let v = match *f {
            Flat::LoopOut { .. } => loop_acc.clone(),
            Flat::LoadState { .. } => m[d as usize].clone(),
            Flat::LoadParam { .. } => {
                if dense {
                    DirMask::ALL
                } else {
                    DirMask::FREE
                }
            }
            Flat::LoadSeed { .. } | Flat::LoadK { .. } => DirMask::NONE,
            Flat::Cmp { .. }
            | Flat::And { .. }
            | Flat::Or { .. }
            | Flat::Not { .. }
            | Flat::Present { .. }
            | Flat::HostOut { .. } => DirMask::NONE,
            _ => uses(f)
                .iter()
                .flatten()
                .fold(DirMask::NONE, |acc, &u| acc | &m[u as usize]),
        };
        m[d as usize] = v;
    }
    m
}

/// A planned program (#2786): a recording's structure — everything [`Recording::plan`] reads
/// but its literals' values and its loop calls' data — with the plan it laid out and, per call,
/// where that call's literals sit in the pool (a loop call: its id). A subject whose recording
/// has this structure, its data aside (the same events and branches, and the same coincidences
/// among its literals, which share a register by value), takes the plan's calls with its own
/// literals and loop calls instead of planning: what planning would build, bit for bit, since
/// nothing else of it reaches the plan. The structure is hashed in order and a hit is
/// confirmed instruction by instruction ([`Recording::fits`]), so no hash collision serves a
/// wrong plan.
pub(crate) struct Template {
    rec: Recording,
    layout: Layout,
    n_eta: usize,
    n_theta: usize,
    plan: Plan,
    src: Vec<Box<[u32]>>,
    /// The recording as made — its code and marks before [`live_code`], its hash-consing
    /// decisions — which [`recorded`] replays another subject against (empty: a recording no
    /// family keeps, never replayed).
    raw: Vec<Flat>,
    raw_marks: Vec<usize>,
    trace: Vec<Step>,
}

#[cfg(test)]
thread_local! {
    /// This thread's (templates off, recordings planned, of them from a template): a test plans
    /// fresh — neither taking nor leaving a template — and counts what its own calls took.
    pub(crate) static TEMPLATE_PROBE: std::cell::Cell<(bool, usize, usize)> =
        const { std::cell::Cell::new((false, 0, 0)) };
    /// This thread's recordings replayed from their family's template ([`recorded`]).
    pub(crate) static REPLAYS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The planned programs, by [`Recording::shape`].
fn templates() -> &'static Mutex<HashMap<u64, Vec<Arc<Template>>, FxBuild>> {
    static T: OnceLock<Mutex<HashMap<u64, Vec<Arc<Template>>, FxBuild>>> = OnceLock::new();
    T.get_or_init(Default::default)
}

impl Recording {
    /// The structure a template is keyed by, hashed.
    fn shape(&self, layout: Layout, n_eta: usize, n_theta: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = Fx::default();
        (&self.code, self.n_regs, &self.outs, self.guard, &self.marks).hash(&mut h);
        (&self.eta_regs, &self.theta_regs, self.dense).hash(&mut h);
        (layout, n_eta, n_theta).hash(&mut h);
        #[cfg(test)]
        if COLLIDE.with(std::cell::Cell::get) {
            return 0;
        }
        h.finish()
    }

    /// Whether `t` planned this recording's structure.
    fn fits(&self, t: &Template, layout: Layout, n_eta: usize, n_theta: usize) -> bool {
        let r = &t.rec;
        let shape = |c: &CallKind| (c.n_in(), c.n_out());
        (t.layout, t.n_eta, t.n_theta) == (layout, n_eta, n_theta)
            && (r.n_regs, &r.outs, r.guard, &r.marks)
                == (self.n_regs, &self.outs, self.guard, &self.marks)
            && (&r.eta_regs, &r.theta_regs, r.dense)
                == (&self.eta_regs, &self.theta_regs, self.dense)
            && r.code.len() == self.code.len()
            && r.code.iter().zip(&self.code).all(|(a, b)| a.same(b))
            && r.loops.iter().map(shape).eq(self.loops.iter().map(shape))
    }

    /// A template's plan `p` with this recording's literals and loop calls (`src`: per call,
    /// where they sit).
    fn serve(&self, p: &Plan, src: &[Box<[u32]>]) -> Plan {
        let calls: Vec<Call> = p
            .calls
            .iter()
            .zip(src)
            .map(|(c, src)| match c {
                Call::Chunk {
                    prog, idx, guard, ..
                } => Call::Chunk {
                    prog: prog.clone(),
                    idx: idx.clone(),
                    lits: src.iter().map(|&k| self.pool[k as usize]).collect(),
                    guard: *guard,
                },
                Call::Loop { args, outs, .. } => Call::Loop {
                    spec: self.loops[src[0] as usize].clone(),
                    args: args.clone(),
                    outs: outs.clone(),
                },
            })
            .collect();
        Plan {
            chunky: calls.iter().all(|c| matches!(c, Call::Chunk { .. })),
            calls,
            n_slots: p.n_slots,
            layout: p.layout,
            lanes: p.lanes,
            n_eta: p.n_eta,
            n_theta: p.n_theta,
            outs: p.outs.clone(),
            guard: p.guard,
            resolved: Default::default(),
            ode_override: self.ode.then(crate::ode::solver::current_ode_override),
            declined: self.declined,
        }
    }

    /// Cut at the marks, compile (or reuse) each chunk, and lay out the subject's plan — or
    /// take a template's ([`Template`]) when one planned this structure.
    pub(crate) fn plan(mut self, layout: Layout, n_eta: usize, n_theta: usize) -> Option<Plan> {
        if let Some(t) = self.tie.replayed.take() {
            // Its family's template, its structure checked as recorded ([`recorded`]).
            if (t.layout, t.n_eta, t.n_theta) != (layout, n_eta, n_theta) {
                return None;
            }
            #[cfg(test)]
            TEMPLATE_PROBE.with(|p| {
                let (o, n, k) = p.get();
                p.set((o, n + 1, k + 1));
            });
            STATS.templated.fetch_add(1, Relaxed);
            return Some(self.serve(&t.plan, &t.src));
        }
        let family = self.tie.family.take();
        let key = self.shape(layout, n_eta, n_theta);
        let lock = || {
            templates()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        #[cfg(test)]
        let off = TEMPLATE_PROBE.with(|p| p.get().0);
        #[cfg(not(test))]
        let off = false;
        let hit = lock().get(&key).filter(|_| !off).and_then(|v| {
            v.iter()
                .find(|t| self.fits(t, layout, n_eta, n_theta))
                .cloned()
        });
        #[cfg(test)]
        TEMPLATE_PROBE.with(|p| {
            let (o, n, k) = p.get();
            p.set((o, n + 1, k + hit.is_some() as usize));
        });
        // The family replays the template it was planned from — one made by a recording that
        // kept what a replay checks against.
        let adopt = |t: &Arc<Template>| {
            if let Some(f) = family.as_ref().filter(|_| !t.raw.is_empty()) {
                let mut m = lock_families();
                let e = m.entry(f.clone()).or_default();
                if !e.first().is_some_and(|o| Arc::ptr_eq(o, t)) {
                    let mut v: Vec<Arc<Template>> = vec![t.clone()];
                    v.extend(e.iter().filter(|o| !Arc::ptr_eq(o, t)).cloned());
                    *e = Arc::new(v);
                }
            }
        };
        if let Some(t) = hit {
            STATS.templated.fetch_add(1, Relaxed);
            adopt(&t);
            return Some(self.serve(&t.plan, &t.src));
        }
        let (tplan, src) = self.plan_fresh(layout, n_eta, n_theta)?;
        let plan = self.serve(&tplan, &src);
        if off {
            return Some(plan);
        }
        // The template keeps the structure, not the values.
        self.pool = Vec::new();
        self.vals = Vec::new();
        let tie = std::mem::take(&mut self.tie);
        let t = Template {
            rec: self,
            layout,
            n_eta,
            n_theta,
            plan: tplan,
            src,
            raw: tie.raw,
            raw_marks: tie.raw_marks,
            trace: tie.trace,
        };
        let mut m = lock();
        let v = m.entry(key).or_default();
        let t = match v.iter().find(|o| t.rec.fits(o, layout, n_eta, n_theta)) {
            Some(o) => o.clone(),
            None => {
                let t = Arc::new(t);
                v.push(t.clone());
                t
            }
        };
        drop(m);
        adopt(&t);
        Some(plan)
    }

    /// [`Self::plan`] by planning, with per call where its literals sit in the pool (a loop
    /// call: its id).
    fn plan_fresh(
        &self,
        layout: Layout,
        n_eta: usize,
        n_theta: usize,
    ) -> Option<(Plan, Vec<Box<[u32]>>)> {
        let lanes = layout.lanes();
        let n_regs = self.n_regs as usize;
        let masks = direction_masks(&self.code, n_regs, &self.eta_regs, self.dense);
        let is_input = |f: &Flat| matches!(f, Flat::LoadState { .. } | Flat::LoadParam { .. });
        // Slots: η, θ, then outputs as they appear.
        const NONE: u32 = u32::MAX;
        let mut slot = vec![NONE; n_regs];
        let mut eta_of = vec![u16::MAX; n_regs];
        for &(r, k) in &self.eta_regs {
            slot[r as usize] = k as u32;
            eta_of[r as usize] = k as u16;
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
        // The guard's running values: each `And` of `guard_reg` extends the one before it, so
        // a `0` anywhere along the chain is a `0` in the end.
        let mut running = vec![false; n_regs];
        if let Some(mut g) = self.guard {
            running[g as usize] = true;
            for f in self.code.iter().rev() {
                if let Flat::And { d, a, .. } = *f {
                    if d == g {
                        g = a;
                        running[g as usize] = true;
                    }
                }
            }
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
        let (mut calls, mut src) = (Vec::new(), Vec::new());
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
                        Flat::Loop { id } => {
                            spec = self.loops.get(id as usize).cloned();
                            src.push(Box::from([id]));
                        }
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
            let mut running_guard = None;
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
                    if running[d as usize] {
                        running_guard = Some(d);
                    }
                }
            }
            if let Some(g) = running_guard.filter(|g| !outs.contains(g)) {
                outs.push(g);
            }
            // The inputs' structure: live directions, and the exact one of a seed.
            let spec: Vec<(DirMask, u16)> = ins
                .iter()
                .map(|&r| (masks[r as usize].clone(), eta_of[r as usize]))
                .collect();
            let body: Vec<Flat> = body.into_iter().copied().collect();
            let (prog, lits) = build_chunk(&body, &ins, &outs, spec, &mut cache);
            src.push(lits.into_boxed_slice());
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
                lits: Box::default(),
                guard: running_guard.map(|g| slot[g as usize]),
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
        let plan = Plan {
            chunky: calls.iter().all(|c| matches!(c, Call::Chunk { .. })),
            calls,
            n_slots,
            layout,
            lanes,
            n_eta,
            n_theta,
            outs: outs?,
            guard,
            resolved: Default::default(),
            ode_override: None,
            declined: 0,
        };
        Some((plan, src))
    }
}

thread_local! {
    /// Calls a forward has run on this thread (a test counts what the early exit skips).
    #[cfg(test)]
    static CALLS_RUN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BUF: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Kept forwards [`Plan::adopt`] has handed out on this thread (a test counts the reuse).
    #[cfg(test)]
    static ADOPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Value passes [`Plan::run_values`] has served on this thread (a test counts them).
    #[cfg(test)]
    static VALUE_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The plan whose [`Plan::run`] last filled `BUF` on this thread, if that run held.
    static LAST_RUN: std::cell::Cell<Option<LastRun>> = const { std::cell::Cell::new(None) };
    /// [`Plan::run_values`]'s buffer: apart from `BUF`, so a value pass leaves the last
    /// gradient forward ([`Plan::run`]) in `BUF` for [`keep_last_run`].
    static VBUF: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static ADJ: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Buffers of finished passes, reused by the next ones on this thread (`take`/`give`).
    static POOL: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// A zeroed buffer of `n` lanes, from this thread's pool when it has one.
pub(crate) fn take(n: usize) -> Vec<f64> {
    let mut v = POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default();
    v.clear();
    v.resize(n, 0.0);
    v
}

/// Return a buffer to this thread's pool (a few are kept; the rest are freed).
pub(crate) fn give(v: Vec<f64>) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 32 {
            p.push(v);
        }
    });
}

/// A zeroed pooled buffer that returns to the pool on drop ([`take`]/[`give`]).
pub(crate) struct Buf(Vec<f64>);

impl Buf {
    pub(crate) fn new(n: usize) -> Self {
        Buf(take(n))
    }
}

impl std::ops::Deref for Buf {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        &self.0
    }
}

impl std::ops::DerefMut for Buf {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.0
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        give(std::mem::take(&mut self.0));
    }
}

/// What the thread's last [`Plan::run`] left in `BUF`.
#[derive(Clone, Copy)]
struct LastRun {
    plan: usize,
    layout: Layout,
    need: usize,
    n_eta: usize,
    n_theta: usize,
}

/// A subject's forward at the end of its inner solve ([`keep_last_run`]): the slots of a
/// chunk-only plan at its inner layout, which [`Plan::adopt`] turns back into a [`Fwd`] for the
/// outer gradient's reverse, so that does not run the same forward again at `η̂`.
struct Kept {
    plan: usize,
    layout: Layout,
    buf: Vec<f64>,
}

fn kept_table() -> &'static Mutex<HashMap<usize, Kept, FxBuild>> {
    static T: OnceLock<Mutex<HashMap<usize, Kept, FxBuild>>> = OnceLock::new();
    T.get_or_init(Default::default)
}

/// The slot lanes of `buf` hold the forward at `(theta, eta)`: lane 0 of each input slot.
fn inputs_match(buf: &[f64], l: usize, n_eta: usize, theta: &[f64], eta: &[f64]) -> bool {
    eta.len() == n_eta
        && eta
            .iter()
            .enumerate()
            .all(|(k, v)| buf[k * l].to_bits() == v.to_bits())
        && theta
            .iter()
            .enumerate()
            .all(|(j, v)| buf[(n_eta + j) * l].to_bits() == v.to_bits())
}

/// Keep, for `owner` (the subject), the forward this thread's last [`Plan::run`] left in
/// `BUF`, when that was a chunk-only plan's run at exactly `(theta, eta)`. Replaces what the
/// owner had kept.
pub(crate) fn keep_last_run(owner: usize, theta: &[f64], eta: &[f64]) {
    let Some(last) = LAST_RUN.with(|c| c.get()) else {
        return;
    };
    let l = last.layout.lanes();
    let kept = BUF.with(|b| {
        let b = b.borrow();
        let buf = b.get(..last.need)?;
        if last.n_theta != theta.len() || !inputs_match(buf, l, last.n_eta, theta, eta) {
            return None;
        }
        let mut v = take(last.need);
        v.copy_from_slice(buf);
        Some(v)
    });
    if let Some(buf) = kept {
        let old = kept_table().lock().unwrap().insert(
            owner,
            Kept {
                plan: last.plan,
                layout: last.layout,
                buf,
            },
        );
        if let Some(o) = old {
            give(o.buf);
        }
    }
}

/// Value passes served on this thread so far.
#[cfg(test)]
pub(crate) fn value_runs() -> usize {
    VALUE_RUNS.with(|c| c.get())
}

/// Forwards adopted on this thread so far.
#[cfg(test)]
pub(crate) fn adopts() -> usize {
    ADOPTS.with(|c| c.get())
}

/// Drop every kept forward (a new fit or a dropped model: the plans they belong to may go).
pub(crate) fn clear_kept() {
    for (_, k) in kept_table().lock().unwrap().drain() {
        give(k.buf);
    }
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
    #[cfg(feature = "survival")]
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }
}

impl Drop for Fwd {
    fn drop(&mut self) {
        give(std::mem::take(&mut self.buf));
        for lg in self.logs.drain(..) {
            if let CallLog::Ode(lg) = lg {
                lg.give();
            }
        }
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

    /// Every chunk's entry point over `(layout, rev)`, resolved once and then read lock-free.
    /// More distinct keys than slots (none in practice) resolve on every call instead.
    fn fns(&self, layout: Layout, rev: bool) -> Option<Fns<'_>> {
        for slot in &self.resolved {
            let e = slot.get_or_init(|| Resolved {
                key: (layout, rev),
                kerns: self.resolve(layout, rev),
            });
            if e.key == (layout, rev) {
                return e.kerns.as_deref().map(Fns::Shared);
            }
        }
        self.resolve(layout, rev).map(Fns::Own)
    }

    fn resolve(&self, layout: Layout, rev: bool) -> Option<Vec<Kern>> {
        self.calls
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
            .collect()
    }

    /// The forward over `layout` into `buf` (`n_slots × lanes`), η and θ already in place:
    /// see [`Self::forward_logged`].
    fn forward_into(&self, fns: &[Kern], buf: &mut [f64], layout: Layout) -> Option<bool> {
        self.forward_logged(fns, buf, layout, None)
    }

    /// [`Self::forward_into`], keeping each loop call's run log when `logs` asks. `None` when
    /// a loop call fails; `Some(false)` when the guard fails, at the first call that shows it
    /// (`buf` is then only partly written: the guard's own slot still reads `0` if `buf`
    /// started zeroed).
    fn forward_logged(
        &self,
        fns: &[Kern],
        buf: &mut [f64],
        layout: Layout,
        mut logs: Option<&mut Vec<CallLog>>,
    ) -> Option<bool> {
        if let Some(lg) = logs.as_deref_mut() {
            lg.clear();
        }
        if self.declined > 0 && !recording_open() {
            crate::ode::solver::note_linear_declines(self.declined);
        }
        if let Some(ov) = self.ode_override {
            if crate::ode::solver::current_ode_override() != ov {
                return None;
            }
        }
        let l = layout.lanes();
        let mut auto = crate::ode::solver::OdeAutoSwitchState::default();
        for (c, k) in self.calls.iter().zip(fns) {
            match (c, k) {
                (
                    Call::Chunk {
                        idx, lits, guard, ..
                    },
                    Kern::Chunk(k),
                ) => {
                    #[cfg(test)]
                    CALLS_RUN.with(|n| n.set(n.get() + 1));
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
                    if guard.is_some_and(|g| buf[g as usize * l] != 1.0) {
                        return Some(false);
                    }
                }
                (Call::Loop { args, outs, .. }, k @ (Kern::Fixed(_) | Kern::Repeat(_))) => {
                    #[cfg(test)]
                    CALLS_RUN.with(|n| n.set(n.get() + 1));
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
                        return None;
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
                    #[cfg(test)]
                    CALLS_RUN.with(|n| n.set(n.get() + 1));
                    let mut inputs = take(0);
                    inputs.extend(
                        args.iter().flat_map(|&a| {
                            buf[a as usize * l..(a as usize + 1) * l].iter().copied()
                        }),
                    );
                    let mut out = take(outs.len() * l);
                    let want = logs.is_some();
                    let mut log = if want {
                        ode_loop::Log::take()
                    } else {
                        ode_loop::Log::default()
                    };
                    let ok = ode_loop::run_loop(
                        spec,
                        lk,
                        l,
                        &inputs,
                        &mut out,
                        &mut auto,
                        want.then_some(&mut log),
                    );
                    give(inputs);
                    if !ok {
                        give(out);
                        return None;
                    }
                    if let Some(lg) = logs.as_deref_mut() {
                        lg.push(CallLog::Ode(log));
                    }
                    for (j, &o) in outs.iter().enumerate() {
                        buf[o as usize * l..(o as usize + 1) * l]
                            .copy_from_slice(&out[j * l..(j + 1) * l]);
                    }
                    give(out);
                }
                _ => return None,
            }
        }
        Some(true)
    }

    /// Run over the inner loop's layout at `(theta, eta)` (η seeded one-hot), writing each
    /// output's lanes to `out` (`n_out × lanes`); `false` when a guard fails.
    pub(crate) fn run(&self, theta: &[f64], eta: &[f64], out: &mut [f64]) -> bool {
        let Some(fns) = self.fns(self.layout, false) else {
            return false;
        };
        LAST_RUN.with(|c| c.set(None));
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
            if self.forward_into(&fns, buf, self.layout) != Some(true) {
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
            LAST_RUN.with(|c| {
                c.set(self.chunky.then_some(LastRun {
                    plan: self as *const Plan as usize,
                    layout: self.layout,
                    need,
                    n_eta: self.n_eta,
                    n_theta: self.n_theta,
                }))
            });
            true
        })
    }

    /// Whether every call is a chunk: the plan then holds the whole prediction as straight-line
    /// code (an exact linear segment is one), with no loop call a solver controller drives.
    pub(crate) fn chunks_only(&self) -> bool {
        self.chunky
    }

    /// The outputs' values at `(theta, eta)`, no derivative lanes, into `out` (`n_out`): the
    /// forward at [`Layout::F64`] over this thread's buffer, with no kept forward, no run logs
    /// and no allocation. `false` when a guard fails or a call cannot run.
    pub(crate) fn run_values(&self, theta: &[f64], eta: &[f64], out: &mut [f64]) -> bool {
        let Some(fns) = self.fns(Layout::F64, false) else {
            return false;
        };
        VBUF.with(|b| {
            let mut b = b.borrow_mut();
            if b.len() < self.n_slots {
                b.resize(self.n_slots, 0.0);
            }
            let buf = &mut b[..self.n_slots];
            buf[..self.n_eta].copy_from_slice(&eta[..self.n_eta]);
            buf[self.n_eta..self.n_eta + self.n_theta].copy_from_slice(&theta[..self.n_theta]);
            if self.forward_into(&fns, buf, Layout::F64) != Some(true) {
                return false;
            }
            if self.guard.is_some_and(|g| buf[g as usize] != 1.0) {
                return false;
            }
            for (o, &s) in out.iter_mut().zip(&self.outs) {
                *o = buf[s as usize];
            }
            #[cfg(test)]
            VALUE_RUNS.with(|c| c.set(c.get() + 1));
            true
        })
    }

    /// `owner`'s forward kept at the end of its inner solve ([`keep_last_run`]) as a [`Fwd`] at
    /// `layout`, when this plan made it at exactly `(theta, eta)` and `layout` is the inner
    /// one; taken, so one gradient reads it. Its slots are the ones [`Self::forward_kept`] would
    /// compute (the same chunks over the same inputs), a chunk-only plan having no run logs.
    pub(crate) fn adopt(
        &self,
        owner: usize,
        layout: Layout,
        theta: &[f64],
        eta: &[f64],
    ) -> Option<Fwd> {
        if !self.chunky || layout != self.layout {
            return None;
        }
        let me = self as *const Plan as usize;
        let mut t = kept_table().lock().unwrap();
        let k = t.get(&owner)?;
        if k.plan != me || k.layout != layout || k.buf.len() != self.n_slots * self.lanes {
            return None;
        }
        if theta.len() != self.n_theta || !inputs_match(&k.buf, self.lanes, self.n_eta, theta, eta)
        {
            return None;
        }
        let k = t.remove(&owner)?;
        #[cfg(test)]
        ADOPTS.with(|c| c.set(c.get() + 1));
        Some(Fwd {
            plan: me,
            layout,
            buf: k.buf,
            logs: Vec::new(),
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
        let mut buf = take(self.n_slots * l);
        buf[..self.n_eta * l].copy_from_slice(&u[..self.n_eta * l]);
        buf[self.n_eta * l..ins].copy_from_slice(&params[..self.n_theta * l]);
        let mut logs = Vec::new();
        // A failed guard stops the forward early: `buf` started zeroed, so the guard's slot
        // still reads `0` ([`Self::kept_guard`]).
        match self.forward_logged(&fwd, &mut buf, layout, Some(&mut logs)) {
            Some(_) => Some(Fwd {
                plan: self as *const Plan as usize,
                layout,
                buf,
                logs,
            }),
            None => {
                give(buf);
                None
            }
        }
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
            let mut b = take(f.buf.len());
            b.copy_from_slice(&f.buf);
            b
        } else {
            compact(&f.buf, hl, l)
        };
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
                        let mut inputs = take(0);
                        inputs.extend(args.iter().flat_map(|&a| {
                            buf[a as usize * l..(a as usize + 1) * l].iter().copied()
                        }));
                        let mut out_adj = take(0);
                        for sd in 0..seeds {
                            let blk = &adj[sd * need..(sd + 1) * need];
                            out_adj.extend(outs.iter().flat_map(|&o| {
                                blk[o as usize * l..(o as usize + 1) * l].iter().copied()
                            }));
                        }
                        let mut in_adj = take(seeds * args.len() * l);
                        let ok = match (spec, k, &f.logs[li]) {
                            (CallKind::Ode(spec), Kern::Loop(rk), CallLog::Ode(lg)) => {
                                ode_loop::run_loop_rev(
                                    spec,
                                    rk,
                                    l,
                                    &inputs,
                                    lg,
                                    hl,
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
                        give(inputs);
                        give(out_adj);
                        if !ok {
                            give(in_adj);
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
                        give(in_adj);
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
        give(buf);
        ok
    }
}

/// [`fold`] for a chunk: input `i`'s lanes outside its directions are zero, a seed's own
/// lane is one; literals carry no tangent.
fn fold_chunk(src: &Func, layout: Layout, spec: &[(DirMask, u16)]) -> Func {
    let lanes = layout.lanes();
    let n = layout.n as usize;
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
        let (mask, seed) = &spec[idx];
        let seed = *seed;
        let (k, l) = lane_dir(lane)?;
        if lane <= n && seed as usize == k {
            return Some(1.0);
        }
        if seed != u16::MAX && lane > n {
            return Some(0.0);
        }
        if !mask.has(k) || !mask.has(l) {
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
        // The lane-parallel operations as `f64x2` ones ([`simd`]), every other instruction as
        // it is, in the packs' order.
        let packs = simd::packs(f, lanes, rev, prefix);
        let order: Vec<simd::Node> = match &packs {
            Some(p) => p.order.clone(),
            None => (0..f.insts.len() as ir::V)
                .map(simd::Node::Scalar)
                .collect(),
        };
        let pk = packs.as_ref();
        let mut v: Vec<Option<Value>> = vec![None; f.insts.len()];
        let mut pv: Vec<Option<Value>> = vec![None; pk.map_or(0, |p| p.members.len())];
        let mut splats: HashMap<ir::V, Value, FxBuild> = HashMap::default();
        // An instruction's scalar value: a packed one's lane (a store's, its stored value's).
        fn scalar(
            b: &mut FunctionBuilder,
            f: &Func,
            pk: Option<&simd::Packs>,
            v: &mut [Option<Value>],
            pv: &[Option<Value>],
            x: ir::V,
        ) -> Value {
            if let Some(y) = v[x as usize] {
                return y;
            }
            let y = match (f.insts[x as usize], pk.and_then(|p| p.of[x as usize])) {
                (Inst::Store { v: s, .. }, _) => scalar(b, f, pk, v, pv, s),
                (_, Some((q, lane))) => {
                    let vec = pv[q as usize].expect("a pack before its lanes' readers");
                    b.ins().extractlane(vec, lane)
                }
                _ => unreachable!("an operand before its reader"),
            };
            v[x as usize] = Some(y);
            y
        }
        let mut in_body = false;
        for node in order {
            let anchor = match node {
                simd::Node::Scalar(i) => i as usize,
                simd::Node::Pack(p) => {
                    let m = pk.expect("packs").members[p as usize];
                    m[0].max(m[1]) as usize
                }
            };
            if let (false, true, Some((head, body, _))) = (in_body, anchor >= prefix, seed_loop) {
                in_body = true;
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
            let at = match node {
                simd::Node::Scalar(i) => i as usize,
                simd::Node::Pack(p) => {
                    let pks = pk.expect("packs");
                    let [m0, m1] = pks.members[p as usize];
                    // Operand `j` as a vector: its pack's, or one shared value splatted.
                    let vop = |b: &mut FunctionBuilder,
                               v: &mut Vec<Option<Value>>,
                               splats: &mut HashMap<ir::V, Value, FxBuild>,
                               j: usize|
                     -> Value {
                        let (x0, x1) = (
                            simd::operand(&f.insts[m0 as usize], j),
                            simd::operand(&f.insts[m1 as usize], j),
                        );
                        if x0 == x1 {
                            if let Some(&s) = splats.get(&x0) {
                                return s;
                            }
                            let x = scalar(b, f, pk, v, &pv, x0);
                            let s = b.ins().splat(types::F64X2, x);
                            splats.insert(x0, s);
                            s
                        } else {
                            let (q, _) = pks.of[x0 as usize].expect("a paired operand");
                            pv[q as usize].expect("a pack before its readers")
                        }
                    };
                    let out = match f.insts[m0 as usize] {
                        Inst::Load { arg, off } => {
                            let (i, lane) = (off as usize / lanes, off as usize % lanes);
                            let a = match arg {
                                0 => slot_addr(&mut b, &mut offs, &mut addr, adjp, i, false),
                                4 => slot_addr(&mut b, &mut offs, &mut addr, adjp, i, true),
                                7 => slot_addr(&mut b, &mut offs, &mut addr, adjp, n_in + i, true),
                                _ => return None,
                            };
                            Some(b.ins().load(types::F64X2, fl, a, (lane * 8) as i32))
                        }
                        Inst::Store { arg, off, .. } => {
                            let (j, lane) = (off as usize / lanes, off as usize % lanes);
                            let x = vop(&mut b, &mut v, &mut splats, 0);
                            let a = match arg {
                                3 => slot_addr(&mut b, &mut offs, &mut addr, adjp, n_in + j, false),
                                4 => slot_addr(&mut b, &mut offs, &mut addr, adjp, j, true),
                                _ => return None,
                            };
                            // A forward's output is canonical: a zero is `+0`.
                            let stored = if arg == 3 {
                                let z = b.ins().f64const(0.0);
                                let z = b.ins().splat(types::F64X2, z);
                                b.ins().fadd(x, z)
                            } else {
                                x
                            };
                            b.ins().store(fl, stored, a, (lane * 8) as i32);
                            None
                        }
                        Inst::Add(..) => {
                            let (x, y) = (
                                vop(&mut b, &mut v, &mut splats, 0),
                                vop(&mut b, &mut v, &mut splats, 1),
                            );
                            Some(b.ins().fadd(x, y))
                        }
                        Inst::Sub(..) => {
                            let (x, y) = (
                                vop(&mut b, &mut v, &mut splats, 0),
                                vop(&mut b, &mut v, &mut splats, 1),
                            );
                            Some(b.ins().fsub(x, y))
                        }
                        Inst::Mul(..) => {
                            let (x, y) = (
                                vop(&mut b, &mut v, &mut splats, 0),
                                vop(&mut b, &mut v, &mut splats, 1),
                            );
                            Some(b.ins().fmul(x, y))
                        }
                        Inst::Div(..) => {
                            let (x, y) = (
                                vop(&mut b, &mut v, &mut splats, 0),
                                vop(&mut b, &mut v, &mut splats, 1),
                            );
                            Some(b.ins().fdiv(x, y))
                        }
                        Inst::Neg(_) => {
                            let x = vop(&mut b, &mut v, &mut splats, 0);
                            Some(b.ins().fneg(x))
                        }
                        Inst::Abs(_) => {
                            let x = vop(&mut b, &mut v, &mut splats, 0);
                            Some(b.ins().fabs(x))
                        }
                        Inst::Call1(Shim1::Sqrt, _) => {
                            let x = vop(&mut b, &mut v, &mut splats, 0);
                            Some(b.ins().sqrt(x))
                        }
                        _ => return None,
                    };
                    pv[p as usize] = out;
                    continue;
                }
            };
            let inst = &f.insts[at];
            // Every operand's scalar, a packed one's lane read out.
            {
                let (ops, k) = simd::operands(inst);
                for &o in &ops[..k] {
                    scalar(&mut b, f, pk, &mut v, &pv, o);
                }
            }
            let r = |x: ir::V| v[x as usize].expect("an operand before its reader");
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
                            v[at] = Some(r(x));
                            continue;
                        }
                        _ => return None,
                    };
                    // A forward's output is canonical: a zero is `+0` ([`canonical_zero`]).
                    let stored = if arg == 3 && !rev {
                        let z = b.ins().f64const(0.0);
                        b.ins().fadd(r(x), z)
                    } else {
                        r(x)
                    };
                    b.ins().store(fl, stored, a, (lane * 8) as i32);
                    r(x)
                }
            };
            v[at] = Some(out);
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
