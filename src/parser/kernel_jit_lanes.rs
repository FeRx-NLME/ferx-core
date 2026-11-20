//! Prototype (#3080): a chunk's forward lowered with **loops over lanes** instead of unrolled.
//!
//! [`ir::forward`] writes every dual operation out lane by lane, and [`fold_chunk`] then drops
//! the terms the inputs' fixed lanes (η's one-hot seeds, θ's zero tangents, a direction an input
//! does not carry) make exactly `0` or `1`. The kernel's size is therefore the lane count times
//! the program's — at the packed η-Hessian over 11 η, 78 lanes, with each product's payload
//! lane reading four tangent lanes.
//!
//! This lowering keeps the folding and loses the unrolling. Each register's tangent lanes are
//! tracked as **classes** — a known constant (by its bits) or live — exactly the facts
//! `fold_with` knows about the corresponding instructions. An operation's output lanes are
//! split by the classes of the operand lanes each one reads (its *signature*): lanes with one
//! signature fold to one formula, and the classes of the output follow from the signature
//! alone. A group of at least [`Opts::loop_min`] lanes becomes one loop over a table of lane
//! indices (a gather: a payload lane `(k, l)` reads tangent lanes `k` and `l`); a smaller group
//! is written out. The value lane — and every scalar the dual bodies compute once (`1/b`, a
//! unary's `h₁`, `h₂`) — stays straight-line code.
//!
//! **Bits.** Each lane computes the folded program's instructions for that lane: the same IEEE
//! operations on the same operands in the same order, with the same `0`/`1` folds
//! ([`mul`], [`add`], … mirror `fold_with` rule for rule). Only the order *between* lanes
//! changes, and no lane reads another lane of the operation it belongs to except where the
//! unrolled code does too (a quotient's payload reads its own tangents, computed by the earlier
//! loop). So the result is bit-identical to the production kernel, which the tests pin.
//!
//! Test-only: the numbers it measures (`lane_loop_prototype_numbers`) decide whether the
//! reverse, the tangent sweeps and the plan cache follow.
//!
//! **Measured** (macOS arm64, release, the plan's large-kernel tier: single-pass allocator,
//! optimizer on; a 9-state Michaelis–Menten chain, 11 η, packed η-Hessian = 78 lanes; every
//! configuration bit-identical to the production kernel):
//!
//! | body | | Cranelift insts | native | compile | per call |
//! |---|---|---|---|---|---|
//! | Rodas5P step (17,285 flat) | unrolled | 1,672,932 | 17.2 MB | 0.76 s | 5.6 ms |
//! | | looped, `loop_min` 8 | 378,995 | 3.0 MB | 1.36 s | 2.0 ms |
//! | | looped, `loop_min` 16 | 428,199 | 3.9 MB | 0.94 s | 1.9 ms |
//! | RK45 step (773 flat) | unrolled | 111,200 | 681 KiB | 0.04 s | 144 µs |
//! | | looped, `loop_min` 8 | 36,501 | 192 KiB | 0.03 s | 126 µs |
//!
//! So native code shrinks 4–6×, and the call gets *faster* (the unrolled kernel streams its own
//! code through the instruction cache). The production unrolled IR (920k scalar instructions for
//! that step) is never built. Compile time is not yet a win: the looped kernel is one function,
//! where the production one above 24,576 instructions is cut into segments. What a call needs
//! besides the code — the gather tables (8.2 MB here) and one scratch block per register with a
//! live lane (4.7 MB, no reuse) — is the next lever: a payload's `(k, l)` is affine in a nested
//! loop and needs no table, and a register's scratch can be reused once it is dead.
use super::super::Arith;
use super::ir::{Cc, Shim1};
use super::*;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, SigRef, Signature, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

/// A scalar the formulas compute on: an emitted value, or nothing (the class-only pass).
#[derive(Clone, Copy, Debug)]
enum Xv {
    Unit,
    Val(Value),
}

/// One lane's operand, as `fold_with` sees it: a known constant or a computed value.
#[derive(Clone, Copy, Debug)]
enum S {
    K(f64),
    X(Xv),
}

#[derive(Clone, Copy)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
}

/// What the formulas emit into: Cranelift, or nothing (to learn the result's class).
trait Em {
    fn konst(&mut self, c: f64) -> Xv;
    fn bin(&mut self, op: Bin, a: Xv, b: Xv) -> Xv;
    fn neg(&mut self, a: Xv) -> Xv;
    fn sel(&mut self, c: Xv, a: Xv, b: Xv) -> Xv;
}

struct ClassOnly;
impl Em for ClassOnly {
    fn konst(&mut self, _: f64) -> Xv {
        Xv::Unit
    }
    fn bin(&mut self, _: Bin, _: Xv, _: Xv) -> Xv {
        Xv::Unit
    }
    fn neg(&mut self, _: Xv) -> Xv {
        Xv::Unit
    }
    fn sel(&mut self, _: Xv, _: Xv, _: Xv) -> Xv {
        Xv::Unit
    }
}

struct Cl<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
}
fn val(x: Xv) -> Value {
    match x {
        Xv::Val(v) => v,
        Xv::Unit => unreachable!("a class-only value in emitted code"),
    }
}
impl Em for Cl<'_, '_> {
    fn konst(&mut self, c: f64) -> Xv {
        Xv::Val(self.b.ins().f64const(c))
    }
    fn bin(&mut self, op: Bin, a: Xv, b: Xv) -> Xv {
        let (a, b) = (val(a), val(b));
        Xv::Val(match op {
            Bin::Add => self.b.ins().fadd(a, b),
            Bin::Sub => self.b.ins().fsub(a, b),
            Bin::Mul => self.b.ins().fmul(a, b),
            Bin::Div => self.b.ins().fdiv(a, b),
        })
    }
    fn neg(&mut self, a: Xv) -> Xv {
        Xv::Val(self.b.ins().fneg(val(a)))
    }
    fn sel(&mut self, c: Xv, a: Xv, b: Xv) -> Xv {
        Xv::Val(self.b.ins().select(val(c), val(a), val(b)))
    }
}

// ── `fold_with`'s rules, on one lane ─────────────────────────────────────────────────────

fn is(s: S, c: f64) -> bool {
    matches!(s, S::K(k) if k == c)
}
fn mat(e: &mut dyn Em, s: S) -> Xv {
    match s {
        S::K(c) => e.konst(c),
        S::X(x) => x,
    }
}
fn mul(e: &mut dyn Em, a: S, b: S) -> S {
    if is(a, 0.0) || is(b, 0.0) {
        S::K(0.0)
    } else if is(a, 1.0) {
        b
    } else if is(b, 1.0) {
        a
    } else {
        let (x, y) = (mat(e, a), mat(e, b));
        S::X(e.bin(Bin::Mul, x, y))
    }
}
fn add(e: &mut dyn Em, a: S, b: S) -> S {
    if is(a, 0.0) {
        b
    } else if is(b, 0.0) {
        a
    } else {
        let (x, y) = (mat(e, a), mat(e, b));
        S::X(e.bin(Bin::Add, x, y))
    }
}
fn sub(e: &mut dyn Em, a: S, b: S) -> S {
    if is(b, 0.0) {
        a
    } else if is(a, 0.0) {
        let y = mat(e, b);
        S::X(e.neg(y))
    } else {
        let (x, y) = (mat(e, a), mat(e, b));
        S::X(e.bin(Bin::Sub, x, y))
    }
}
fn div(e: &mut dyn Em, a: S, b: S) -> S {
    if is(a, 0.0) {
        S::K(0.0)
    } else {
        let (x, y) = (mat(e, a), mat(e, b));
        S::X(e.bin(Bin::Div, x, y))
    }
}
fn neg(e: &mut dyn Em, a: S) -> S {
    if is(a, 0.0) {
        a
    } else {
        let x = mat(e, a);
        S::X(e.neg(x))
    }
}
fn sel(e: &mut dyn Em, c: S, a: S, b: S) -> S {
    let same = match (a, b) {
        (S::K(p), S::K(q)) => p == q,
        (S::X(Xv::Val(p)), S::X(Xv::Val(q))) => p == q,
        _ => false,
    };
    if same {
        a
    } else {
        let (c, x, y) = (mat(e, c), mat(e, a), mat(e, b));
        S::X(e.sel(c, x, y))
    }
}

// ── The per-lane formulas of the dual bodies (`ir::Lower`'s, operand for operand) ─────────

/// A lane's formula over its operands (tangent lanes and scalars, in the task's order).
type Formula = fn(&mut dyn Em, &[S]) -> S;

fn f_id(_: &mut dyn Em, o: &[S]) -> S {
    o[0]
}
fn f_add(e: &mut dyn Em, o: &[S]) -> S {
    add(e, o[0], o[1])
}
fn f_sub(e: &mut dyn Em, o: &[S]) -> S {
    sub(e, o[0], o[1])
}
fn f_neg(e: &mut dyn Em, o: &[S]) -> S {
    neg(e, o[0])
}
fn f_mul_by(e: &mut dyn Em, o: &[S]) -> S {
    mul(e, o[0], o[1])
}
/// `lane·scalar` (`mul_c`, `div_c`'s `x·ib`), from `lanewise`'s `[scalar, lane]`.
fn f_lane_mul(e: &mut dyn Em, o: &[S]) -> S {
    mul(e, o[1], o[0])
}
/// `0 − g`.
fn f_rsub0(e: &mut dyn Em, o: &[S]) -> S {
    sub(e, S::K(0.0), o[0])
}
fn f_sel(e: &mut dyn Em, o: &[S]) -> S {
    sel(e, o[0], o[1], o[2])
}
/// `mul`'s tangent: `a_k·b_v + a_v·b_k`; operands `[a_k, b_v, a_v, b_k]`.
fn f_mul_g(e: &mut dyn Em, o: &[S]) -> S {
    let x = mul(e, o[0], o[1]);
    let y = mul(e, o[2], o[3]);
    add(e, x, y)
}
/// `mul`'s payload: `((a_h·b_v + a_v·b_h) + a_k·b_l) + a_l·b_k`; operands
/// `[a_h, b_v, a_v, b_h, a_k, b_l, a_l, b_k]`.
fn f_mul_h(e: &mut dyn Em, o: &[S]) -> S {
    let p = mul(e, o[0], o[1]);
    let q = mul(e, o[2], o[3]);
    let s = add(e, p, q);
    let r = mul(e, o[4], o[5]);
    let s = add(e, s, r);
    let r = mul(e, o[6], o[7]);
    add(e, s, r)
}
/// `div`'s tangent: `(a_k − q·b_k)·ib`; operands `[q, b_k, a_k, ib]`.
fn f_div_g(e: &mut dyn Em, o: &[S]) -> S {
    let t = mul(e, o[0], o[1]);
    let d = sub(e, o[2], t);
    mul(e, d, o[3])
}
/// `div`'s payload: `(((a_h − g_k·b_l) − g_l·b_k) − q·b_h)·ib`; operands
/// `[g_k, b_l, a_h, g_l, b_k, q, b_h, ib]`.
fn f_div_h(e: &mut dyn Em, o: &[S]) -> S {
    let x = mul(e, o[0], o[1]);
    let s = sub(e, o[2], x);
    let x = mul(e, o[3], o[4]);
    let s = sub(e, s, x);
    let x = mul(e, o[5], o[6]);
    let s = sub(e, s, x);
    mul(e, s, o[7])
}
/// `unary`'s payload: `h1·a_h + (h2·a_k)·a_l`; operands `[h1, a_h, h2, a_k, a_l]`.
fn f_unary_h(e: &mut dyn Em, o: &[S]) -> S {
    let x = mul(e, o[0], o[1]);
    let y = mul(e, o[2], o[3]);
    let y = mul(e, y, o[4]);
    add(e, x, y)
}

// ── Registers and the lowering ───────────────────────────────────────────────────────────

/// A tangent lane's class: a known constant (by its bits) or live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Cls {
    K(u64),
    L,
}
impl Cls {
    fn of(s: S) -> Cls {
        match s {
            S::K(c) => Cls::K(c.to_bits()),
            S::X(_) => Cls::L,
        }
    }
}

/// A register: its value lane, and its tangent lanes' classes; the live ones at `base` (the
/// address of tangent lane 0, i.e. layout lane 1).
#[derive(Clone)]
struct LReg {
    v: S,
    cls: std::rc::Rc<[Cls]>,
    base: Option<Value>,
}

/// An operand of a lane task.
#[derive(Clone, Copy)]
enum Opd {
    /// Tangent lane `lane` of the operation's register operand `r`.
    Lane(u8, u32),
    Sc(S),
}

/// Where an operation's tangent lanes go.
struct Out {
    cls: Vec<Cls>,
    base: Option<Value>,
    /// An output slot: every lane is stored, a constant one too.
    store_k: bool,
}

/// The prototype's knobs.
#[derive(Clone, Copy, Debug)]
pub(super) struct Opts {
    /// The smallest group of lanes with one signature that becomes a loop.
    pub(super) loop_min: usize,
}

/// What a lowering produced, for the comparison.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Counts {
    /// Cranelift instructions of the function.
    pub(super) clif_insts: usize,
    pub(super) loops: usize,
    /// Lane tasks run in loops, and written out.
    pub(super) looped_lanes: usize,
    pub(super) unrolled_lanes: usize,
    /// `f64` slots of scratch a call needs, and `u32` table entries.
    pub(super) scratch: usize,
    pub(super) table: usize,
    /// Native code bytes.
    pub(super) native_bytes: usize,
}

struct Lw<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    n: usize,
    t: usize,
    /// `(k, l, t)` of every payload lane, in `ir::pairs` order.
    pairs: Vec<(usize, usize, usize)>,
    lanes: usize,
    base_arg: Value,
    idx_arg: Value,
    lits_arg: Value,
    scratch: Value,
    tables_arg: Value,
    tables: Vec<u32>,
    n_slots: usize,
    opts: Opts,
    counts: Counts,
    uref: SigRef,
    ptr: cranelift_codegen::ir::Type,
    fl: MemFlagsData,
}

impl<'b> Lw<'_, 'b> {
    fn em(&mut self) -> Cl<'_, 'b> {
        Cl { b: &mut *self.b }
    }
    fn x(&mut self, s: S) -> Value {
        match s {
            S::K(c) => self.b.ins().f64const(c),
            S::X(x) => val(x),
        }
    }
    fn call(&mut self, s: Shim1, x: S) -> S {
        let x = self.x(x);
        if s == Shim1::Sqrt {
            return S::X(Xv::Val(self.b.ins().sqrt(x)));
        }
        let callee = self.b.ins().iconst(self.ptr, s.f() as usize as i64);
        let c = self.b.ins().call_indirect(self.uref, callee, &[x]);
        S::X(Xv::Val(self.b.inst_results(c)[0]))
    }
    /// A call of the native host hook at `fp` on `x`.
    fn host(&mut self, fp: usize, x: S) -> S {
        let x = self.x(x);
        let callee = self.b.ins().iconst(self.ptr, fp as i64);
        let c = self.b.ins().call_indirect(self.uref, callee, &[x]);
        S::X(Xv::Val(self.b.inst_results(c)[0]))
    }
    fn cmp(&mut self, cc: Cc, a: S, b: S) -> S {
        let (a, b) = (self.x(a), self.x(b));
        let cc = match cc {
            Cc::Lt => FloatCC::LessThan,
            Cc::Le => FloatCC::LessThanOrEqual,
            Cc::Gt => FloatCC::GreaterThan,
            Cc::Ge => FloatCC::GreaterThanOrEqual,
            Cc::Eq => FloatCC::Equal,
            Cc::Ne => FloatCC::NotEqual,
        };
        S::X(Xv::Val(self.b.ins().fcmp(cc, a, b)))
    }
    fn abs_v(&mut self, a: S) -> S {
        let a = self.x(a);
        S::X(Xv::Val(self.b.ins().fabs(a)))
    }
    /// The address of slot table entry `e`'s lane block.
    fn slot(&mut self, e: usize) -> Value {
        let s = self.b.ins().load(types::I32, self.fl, self.idx_arg, (4 * e) as i32);
        let s = self.b.ins().uextend(types::I64, s);
        let off = self.b.ins().imul_imm_u(s, (self.lanes * 8) as i64);
        self.b.ins().iadd(self.base_arg, off)
    }
    fn fresh(&mut self) -> Value {
        let at = self.n_slots * self.t * 8;
        self.n_slots += 1;
        self.b.ins().iadd_imm_s(self.scratch, at as i64)
    }
    fn constant(&self, v: S) -> LReg {
        LReg { v, cls: vec![Cls::K(0.0f64.to_bits()); self.t].into(), base: None }
    }
    fn reg(&self, v: S, out: Out) -> LReg {
        LReg { v, cls: out.cls.into(), base: out.base }
    }
    fn out(&self) -> Out {
        Out { cls: vec![Cls::L; self.t], base: None, store_k: false }
    }

    /// One kind of lane task: output lane `tasks[i].0` is `f` of the operands `tasks[i].1`.
    fn kind(&mut self, f: Formula, rs: &[&LReg], tasks: &[(u32, Vec<Opd>)], out: &mut Out) {
        // Each task's signature, and the class of its result.
        let mut groups: Vec<(Vec<Cls>, Vec<usize>)> = Vec::new();
        let mut at: HashMap<Vec<Cls>, usize> = HashMap::new();
        let mut ins: Vec<S> = Vec::new();
        for (i, (o, ops)) in tasks.iter().enumerate() {
            let sig: Vec<Cls> = ops
                .iter()
                .map(|op| match *op {
                    Opd::Lane(r, l) => rs[r as usize].cls[l as usize],
                    Opd::Sc(s) => Cls::of(s),
                })
                .collect();
            ins.clear();
            ins.extend(sig.iter().map(|c| match *c {
                Cls::K(b) => S::K(f64::from_bits(b)),
                Cls::L => S::X(Xv::Unit),
            }));
            out.cls[*o as usize] = Cls::of(f(&mut ClassOnly, &ins));
            let g = *at.entry(sig.clone()).or_insert_with(|| {
                groups.push((sig, Vec::new()));
                groups.len() - 1
            });
            groups[g].1.push(i);
        }
        let need_base = out.store_k || tasks.iter().any(|(o, _)| out.cls[*o as usize] == Cls::L);
        if need_base && out.base.is_none() {
            out.base = Some(self.fresh());
        }
        for (_, members) in &groups {
            let rep = &tasks[members[0]];
            let res = out.cls[rep.0 as usize];
            if res != Cls::L && !out.store_k {
                continue;
            }
            if members.len() < self.opts.loop_min {
                for &m in members {
                    let (o, ops) = &tasks[m];
                    let ins: Vec<S> = ops
                        .iter()
                        .map(|op| match *op {
                            Opd::Lane(r, l) => {
                                let reg = rs[r as usize];
                                match reg.cls[l as usize] {
                                    Cls::K(b) => S::K(f64::from_bits(b)),
                                    Cls::L => {
                                        let p = reg.base.expect("a live lane has a base");
                                        let fl = self.fl;
                                        let v =
                                            self.b.ins().load(types::F64, fl, p, (8 * l) as i32);
                                        S::X(Xv::Val(v))
                                    }
                                }
                            }
                            Opd::Sc(s) => s,
                        })
                        .collect();
                    let r = f(&mut self.em(), &ins);
                    let v = self.x(r);
                    let p = out.base.expect("allocated");
                    let fl = self.fl;
                    self.b.ins().store(fl, v, p, (8 * *o) as i32);
                    self.counts.unrolled_lanes += 1;
                }
                continue;
            }
            // A loop over a table: per task its output lane, then each live operand lane.
            let live: Vec<usize> = rep
                .1
                .iter()
                .enumerate()
                .filter(|(_, op)| {
                    matches!(op, Opd::Lane(r, l) if rs[*r as usize].cls[*l as usize] == Cls::L)
                })
                .map(|(j, _)| j)
                .collect();
            let stride = 1 + live.len();
            let toff = self.tables.len();
            for &m in members {
                let (o, ops) = &tasks[m];
                self.tables.push(*o);
                for &j in &live {
                    let Opd::Lane(_, l) = ops[j] else { unreachable!("a live lane") };
                    self.tables.push(l);
                }
            }
            let body = self.b.create_block();
            self.b.append_block_param(body, types::I64);
            let exit = self.b.create_block();
            let zero = self.b.ins().iconst(types::I64, 0);
            self.b.ins().jump(body, &[zero.into()]);
            self.b.switch_to_block(body);
            let i = self.b.block_params(body)[0];
            let off = self.b.ins().imul_imm_u(i, (stride * 4) as i64);
            let ep = self.b.ins().iadd(self.tables_arg, off);
            let fl = self.fl;
            let lane_addr = |b: &mut FunctionBuilder, base: Value, slot: usize| -> Value {
                let l = b.ins().load(types::I32, fl, ep, (4 * (toff + slot)) as i32);
                let l = b.ins().uextend(types::I64, l);
                let l = b.ins().ishl_imm_u(l, 3);
                b.ins().iadd(base, l)
            };
            let mut ins: Vec<S> = Vec::with_capacity(rep.1.len());
            let mut li = 0;
            for (j, op) in rep.1.iter().enumerate() {
                ins.push(match *op {
                    Opd::Lane(r, l) => {
                        let reg = rs[r as usize];
                        match reg.cls[l as usize] {
                            Cls::K(b) => S::K(f64::from_bits(b)),
                            Cls::L => {
                                debug_assert_eq!(live[li], j);
                                li += 1;
                                let a = lane_addr(
                                    self.b,
                                    reg.base.expect("a live lane has a base"),
                                    li,
                                );
                                S::X(Xv::Val(self.b.ins().load(types::F64, fl, a, 0)))
                            }
                        }
                    }
                    Opd::Sc(s) => s,
                });
            }
            let r = f(&mut self.em(), &ins);
            let v = self.x(r);
            let a = lane_addr(self.b, out.base.expect("allocated"), 0);
            self.b.ins().store(fl, v, a, 0);
            let i1 = self.b.ins().iadd_imm_s(i, 1);
            let more = self.b.ins().icmp_imm_u(IntCC::UnsignedLessThan, i1, members.len() as i64);
            self.b.ins().brif(more, body, &[i1.into()], exit, &[]);
            self.b.switch_to_block(exit);
            self.counts.loops += 1;
            self.counts.looped_lanes += members.len();
        }
    }

    fn lanewise(&mut self, f: Formula, regs: &[&LReg], scalar: Option<S>) -> Out {
        let mut out = self.out();
        let tasks: Vec<(u32, Vec<Opd>)> = (0..self.t as u32)
            .map(|j| {
                let mut ops: Vec<Opd> = scalar.into_iter().map(Opd::Sc).collect();
                ops.extend((0..regs.len()).map(|r| Opd::Lane(r as u8, j)));
                (j, ops)
            })
            .collect();
        self.kind(f, regs, &tasks, &mut out);
        out
    }

    fn add_r(&mut self, a: &LReg, b: &LReg) -> LReg {
        let v = add(&mut self.em(), a.v, b.v);
        let out = self.lanewise(f_add, &[a, b], None);
        self.reg(v, out)
    }
    fn sub_r(&mut self, a: &LReg, b: &LReg) -> LReg {
        let v = sub(&mut self.em(), a.v, b.v);
        let out = self.lanewise(f_sub, &[a, b], None);
        self.reg(v, out)
    }
    fn select_r(&mut self, c: S, a: &LReg, b: &LReg) -> LReg {
        let v = sel(&mut self.em(), c, a.v, b.v);
        let out = self.lanewise(f_sel, &[a, b], Some(c));
        self.reg(v, out)
    }
    fn neg_r(&mut self, a: &LReg) -> LReg {
        let v = neg(&mut self.em(), a.v);
        let out = self.lanewise(f_neg, &[a], None);
        self.reg(v, out)
    }
    fn mul_r(&mut self, a: &LReg, b: &LReg) -> LReg {
        let v = mul(&mut self.em(), a.v, b.v);
        let mut out = self.out();
        let g: Vec<(u32, Vec<Opd>)> = (0..self.n as u32)
            .map(|k| (k, vec![Opd::Lane(0, k), Opd::Sc(b.v), Opd::Sc(a.v), Opd::Lane(1, k)]))
            .collect();
        self.kind(f_mul_g, &[a, b], &g, &mut out);
        let h: Vec<(u32, Vec<Opd>)> = self
            .pairs
            .iter()
            .map(|&(k, l, t)| {
                let (h, k, l) = ((self.n + t) as u32, k as u32, l as u32);
                let ops = vec![
                    Opd::Lane(0, h),
                    Opd::Sc(b.v),
                    Opd::Sc(a.v),
                    Opd::Lane(1, h),
                    Opd::Lane(0, k),
                    Opd::Lane(1, l),
                    Opd::Lane(0, l),
                    Opd::Lane(1, k),
                ];
                (h, ops)
            })
            .collect();
        self.kind(f_mul_h, &[a, b], &h, &mut out);
        self.reg(v, out)
    }
    fn div_r(&mut self, a: &LReg, b: &LReg) -> LReg {
        let q = div(&mut self.em(), a.v, b.v);
        let ib = div(&mut self.em(), S::K(1.0), b.v);
        let mut out = self.out();
        let g: Vec<(u32, Vec<Opd>)> = (0..self.n as u32)
            .map(|k| (k, vec![Opd::Sc(q), Opd::Lane(1, k), Opd::Lane(0, k), Opd::Sc(ib)]))
            .collect();
        self.kind(f_div_g, &[a, b], &g, &mut out);
        // The payload reads the quotient's own tangents: the output so far, as operand 2.
        let so_far = LReg { v: q, cls: out.cls.clone().into(), base: out.base };
        let h: Vec<(u32, Vec<Opd>)> = self
            .pairs
            .iter()
            .map(|&(k, l, t)| {
                let (h, k, l) = ((self.n + t) as u32, k as u32, l as u32);
                let ops = vec![
                    Opd::Lane(2, k),
                    Opd::Lane(1, l),
                    Opd::Lane(0, h),
                    Opd::Lane(2, l),
                    Opd::Lane(1, k),
                    Opd::Sc(q),
                    Opd::Lane(1, h),
                    Opd::Sc(ib),
                ];
                (h, ops)
            })
            .collect();
        self.kind(f_div_h, &[a, b, &so_far], &h, &mut out);
        self.reg(q, out)
    }
    fn guarded_div_r(&mut self, a: &LReg, b: &LReg) -> LReg {
        let ay = self.abs_v(b.v);
        let small = self.cmp(Cc::Lt, ay, S::K(1e-30));
        let mut bs = b.clone();
        bs.v = sel(&mut self.em(), small, S::K(1.0), b.v);
        let q = self.div_r(a, &bs);
        let zero = self.constant(S::K(0.0));
        self.select_r(small, &zero, &q)
    }
    /// `unary_body` with `φ(v) = hv` and local derivatives `h1`, `h2`.
    fn unary_r(&mut self, a: &LReg, hv: S, h1: S, h2: Option<S>) -> LReg {
        let mut out = self.out();
        let g: Vec<(u32, Vec<Opd>)> =
            (0..self.n as u32).map(|k| (k, vec![Opd::Sc(h1), Opd::Lane(0, k)])).collect();
        self.kind(f_mul_by, &[a], &g, &mut out);
        if !self.pairs.is_empty() {
            let h2 = h2.expect("a payload has its second derivative");
            let h: Vec<(u32, Vec<Opd>)> = self
                .pairs
                .iter()
                .map(|&(k, l, t)| {
                    let h = (self.n + t) as u32;
                    let ops = vec![
                        Opd::Sc(h1),
                        Opd::Lane(0, h),
                        Opd::Sc(h2),
                        Opd::Lane(0, k as u32),
                        Opd::Lane(0, l as u32),
                    ];
                    (h, ops)
                })
                .collect();
            self.kind(f_unary_h, &[a], &h, &mut out);
        }
        self.reg(hv, out)
    }
    fn exp_r(&mut self, a: &LReg) -> LReg {
        let e = self.call(Shim1::Exp, a.v);
        self.unary_r(a, e, e, Some(e))
    }
    fn ln_r(&mut self, a: &LReg) -> LReg {
        let l = self.call(Shim1::Ln, a.v);
        let h1 = div(&mut self.em(), S::K(1.0), a.v);
        let h2 = (!self.pairs.is_empty()).then(|| {
            let xx = mul(&mut self.em(), a.v, a.v);
            div(&mut self.em(), S::K(-1.0), xx)
        });
        self.unary_r(a, l, h1, h2)
    }
    fn sqrt_r(&mut self, a: &LReg) -> LReg {
        let s = self.call(Shim1::Sqrt, a.v);
        let h1 = div(&mut self.em(), S::K(0.5), s);
        let h2 = (!self.pairs.is_empty()).then(|| {
            let ss = mul(&mut self.em(), s, s);
            let sss = mul(&mut self.em(), ss, s);
            div(&mut self.em(), S::K(-0.25), sss)
        });
        self.unary_r(a, s, h1, h2)
    }
    /// `Unary::Sqrt` (`flat_nan`) and `Unary::SqrtRhs` (not): the guarded square roots.
    fn guarded_sqrt_r(&mut self, a: &LReg, flat_nan: bool) -> LReg {
        let pos = self.cmp(Cc::Gt, a.v, S::K(0.0));
        let ones = self.constant(S::K(1.0));
        let g = self.select_r(pos, a, &ones);
        let root = self.sqrt_r(&g);
        let nonneg = self.cmp(Cc::Ge, a.v, S::K(0.0));
        let zero = sel(&mut self.em(), nonneg, a.v, S::K(0.0));
        let v0 = mul(&mut self.em(), zero, S::K(0.0));
        let v = if flat_nan { sel(&mut self.em(), nonneg, v0, S::K(f64::NAN)) } else { v0 };
        let flat = self.constant(v);
        self.select_r(pos, &root, &flat)
    }
    fn scalar_r(&mut self, op: Arith, a: &LReg, c: f64) -> LReg {
        let cv = S::K(c);
        match op {
            Arith::Add => LReg { v: add(&mut self.em(), a.v, cv), ..a.clone() },
            Arith::Sub => LReg { v: sub(&mut self.em(), a.v, cv), ..a.clone() },
            Arith::RSub => {
                let v = sub(&mut self.em(), cv, a.v);
                let out = self.lanewise(f_rsub0, &[a], None);
                self.reg(v, out)
            }
            Arith::Mul => {
                let v = mul(&mut self.em(), a.v, cv);
                let out = self.lanewise(f_lane_mul, &[a], Some(cv));
                self.reg(v, out)
            }
            Arith::Div => {
                let v = div(&mut self.em(), a.v, cv);
                let ib = div(&mut self.em(), S::K(1.0), cv);
                let out = self.lanewise(f_lane_mul, &[a], Some(ib));
                self.reg(v, out)
            }
        }
    }
}

/// Lower `code` at `layout` with lane loops into `module`: `fn(base, idx, lits, scratch,
/// tables)`. `None` where the prototype does not cover an instruction (branches, `pow`, the
/// primitives beyond `exp`/`ln`/`sqrt`, the `wide` and θ-diagonal layouts).
#[allow(clippy::too_many_arguments)]
pub(super) fn lower_looped(
    module: &mut JITModule, name: &str, code: &[Flat], n_regs: usize, n_in: usize, layout: Layout,
    spec: &[(DirMask, u16)], opts: Opts,
) -> Option<(cranelift_module::FuncId, Vec<u32>, Counts)> {
    if layout.wide || layout.th || layout.tan != 0 || layout.n == 0 {
        return None;
    }
    let n = layout.n as usize;
    let lanes = layout.lanes();
    let pairs: Vec<(usize, usize, usize)> = (0..n)
        .flat_map(|k| ir::pairs(&layout, k).into_iter().map(move |(l, t)| (k, l, t)))
        .collect();
    let known = chunk_known(layout, spec);
    let target = module.target_config();
    let ptr = target.pointer_type();
    let mut sig = module.make_signature();
    for _ in 0..5 {
        sig.params.push(AbiParam::new(ptr));
    }
    let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    let (tables, mut counts) = {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let block = b.create_block();
        b.append_block_params_for_function_params(block);
        b.switch_to_block(block);
        let args = b.block_params(block).to_vec();
        let mut usig = Signature::new(target.default_call_conv);
        usig.params.push(AbiParam::new(types::F64));
        usig.returns.push(AbiParam::new(types::F64));
        let uref = b.import_signature(usig);
        let mut lw = Lw {
            b: &mut b,
            n,
            t: lanes - 1,
            pairs,
            lanes,
            base_arg: args[0],
            idx_arg: args[1],
            lits_arg: args[2],
            scratch: args[3],
            tables_arg: args[4],
            tables: Vec::new(),
            n_slots: 0,
            opts,
            counts: Counts::default(),
            uref,
            ptr,
            fl: MemFlagsData::trusted(),
        };
        let mut regs: Vec<Option<LReg>> = vec![None; n_regs];
        for f in code {
            let r = |x: u32| regs[x as usize].clone().expect("a register before its reader");
            let (d, l) = match *f {
                Flat::LoadState { d, i } => {
                    let i = i as usize;
                    let p = lw.slot(i);
                    let fl = lw.fl;
                    let v = lw.b.ins().load(types::F64, fl, p, 0);
                    let cls: Vec<Cls> = (1..lanes)
                        .map(|j| match known(0, (i * lanes + j) as u32) {
                            Some(c) => Cls::K(c.to_bits()),
                            None => Cls::L,
                        })
                        .collect();
                    let base = cls.contains(&Cls::L).then(|| lw.b.ins().iadd_imm_s(p, 8));
                    (d, LReg { v: S::X(Xv::Val(v)), cls: cls.into(), base })
                }
                Flat::LoadSeed { d, k } => {
                    let fl = lw.fl;
                    let v = lw.b.ins().load(types::F64, fl, lw.lits_arg, (8 * k) as i32);
                    (d, lw.constant(S::X(Xv::Val(v))))
                }
                Flat::LoadK { d, c } => (d, lw.constant(S::K(c))),
                Flat::Add { d, a, b } => (d, lw.add_r(&r(a), &r(b))),
                Flat::Sub { d, a, b } => (d, lw.sub_r(&r(a), &r(b))),
                Flat::RSub { d, a, b } => (d, lw.sub_r(&r(b), &r(a))),
                Flat::Mul { d, a, b } => (d, lw.mul_r(&r(a), &r(b))),
                Flat::Div { d, a, b } => (d, lw.guarded_div_r(&r(a), &r(b))),
                Flat::DivRaw { d, a, b } => (d, lw.div_r(&r(a), &r(b))),
                Flat::AddK { d, a, c } => (d, lw.scalar_r(Arith::Add, &r(a), c)),
                Flat::SubK { d, a, c } => (d, lw.scalar_r(Arith::Sub, &r(a), c)),
                Flat::RSubK { d, a, c } => (d, lw.scalar_r(Arith::RSub, &r(a), c)),
                Flat::MulK { d, a, c } => (d, lw.scalar_r(Arith::Mul, &r(a), c)),
                Flat::DivK { d, a: _, c } if c.abs() < 1e-30 => (d, lw.constant(S::K(0.0))),
                Flat::DivK { d, a, c } => (d, lw.scalar_r(Arith::Div, &r(a), c)),
                Flat::Unary { d, op, a } => {
                    let x = r(a);
                    let l = match op {
                        Unary::Exp => lw.exp_r(&x),
                        Unary::LnRaw => lw.ln_r(&x),
                        Unary::Ln => {
                            let keep = lw.cmp(Cc::Ge, x.v, S::K(1e-30));
                            let floor = lw.constant(S::K(1e-30));
                            let g = lw.select_r(keep, &x, &floor);
                            lw.ln_r(&g)
                        }
                        Unary::Sqrt => lw.guarded_sqrt_r(&x, true),
                        Unary::SqrtRhs => lw.guarded_sqrt_r(&x, false),
                        // The value with no derivative; `fold_with` does not see through it, so
                        // even a constant operand gives a computed value.
                        Unary::StopGrad => {
                            let v = lw.x(x.v);
                            lw.constant(S::X(Xv::Val(v)))
                        }
                        Unary::Abs => {
                            let nonneg = lw.cmp(Cc::Ge, x.v, S::K(0.0));
                            let ng = lw.neg_r(&x);
                            lw.select_r(nonneg, &x, &ng)
                        }
                        _ => return None,
                    };
                    (d, l)
                }
                Flat::Cmp { d, cc, a, b } => {
                    let c = lw.cmp(cc, r(a).v, r(b).v);
                    let v = sel(&mut lw.em(), c, S::K(1.0), S::K(0.0));
                    (d, lw.constant(v))
                }
                Flat::Select { d, c, a, b } => {
                    let cond = lw.cmp(Cc::Ne, r(c).v, S::K(0.0));
                    (d, lw.select_r(cond, &r(a), &r(b)))
                }
                // The host calls: scalar, in program order (`ir::host`).
                Flat::HostArg { a } => {
                    let x = r(a).v;
                    lw.host(ir::host::arg_native as HostHook as usize, x);
                    continue;
                }
                Flat::Host { f } => {
                    lw.host(ir::host::call_native as HostHook as usize, S::K(f.index() as f64));
                    continue;
                }
                Flat::HostOut { d, k } => {
                    let v = lw.host(ir::host::out_native as HostHook as usize, S::K(k as f64));
                    (d, lw.constant(v))
                }
                Flat::Deriv { state, a } => {
                    let x = r(a);
                    let p = lw.slot(n_in + state as usize);
                    let v = lw.x(x.v);
                    let fl = lw.fl;
                    lw.b.ins().store(fl, v, p, 0);
                    let base = lw.b.ins().iadd_imm_s(p, 8);
                    let mut out = Out { cls: vec![Cls::L; lw.t], base: Some(base), store_k: true };
                    let tasks: Vec<(u32, Vec<Opd>)> =
                        (0..lw.t as u32).map(|j| (j, vec![Opd::Lane(0, j)])).collect();
                    lw.kind(f_id, &[&x], &tasks, &mut out);
                    continue;
                }
                _ => return None,
            };
            regs[d as usize] = Some(l);
        }
        lw.b.ins().return_(&[]);
        let tables = std::mem::take(&mut lw.tables);
        let mut counts = lw.counts;
        counts.scratch = lw.n_slots * lw.t;
        counts.table = tables.len();
        b.seal_all_blocks();
        b.finalize(target);
        (tables, counts)
    };
    counts.clif_insts = ctx.func.dfg.num_insts();
    module.define_function(id, &mut ctx).ok()?;
    counts.native_bytes = ctx.compiled_code().map_or(0, |c| c.code_info().total_size as usize);
    module.clear_context(&mut ctx);
    Some((id, tables, counts))
}

/// A host hook's native entry point (`ir::host`).
type HostHook = extern "C" fn(f64) -> f64;
/// `fn(base, idx, lits, scratch, tables)`.
type LoopedFn = unsafe extern "C" fn(*mut f64, *const u32, *const f64, *mut f64, *const u32);

/// A lowered looped kernel, its module and its tables.
pub(super) struct Looped {
    module: Option<JITModule>,
    f: LoopedFn,
    tables: Vec<u32>,
    pub(super) counts: Counts,
    pub(super) compile_s: f64,
}

impl Drop for Looped {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: `f` points into this module and dies with it.
            unsafe { m.free_memory() };
        }
    }
}

impl Looped {
    /// Compile `cp` at `layout` (forward, gates resolved at its order) under the plan tier
    /// `(quick, opt)`.
    pub(super) fn new(
        cp: &ChunkProgram, layout: Layout, opts: Opts, quick: bool, opt: &str,
    ) -> Option<Self> {
        let gated = resolve_gates(&cp.code, cp.n_regs, layout.order().min(GATE_ORDERS - 1));
        let code = gated.as_deref().unwrap_or(&cp.code);
        let t0 = std::time::Instant::now();
        let isa = plan_isa(quick, opt)?;
        let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let lowered =
            lower_looped(&mut module, "looped", code, cp.n_regs, cp.n_in, layout, &cp.spec, opts);
        let Some((id, tables, counts)) = lowered else {
            // SAFETY: nothing was handed out.
            unsafe { module.free_memory() };
            return None;
        };
        if module.finalize_definitions().is_err() {
            unsafe { module.free_memory() };
            return None;
        }
        let compile_s = t0.elapsed().as_secs_f64();
        let p = module.get_finalized_function(id);
        // SAFETY: defined with this signature above.
        let f = unsafe { std::mem::transmute::<*const u8, LoopedFn>(p) };
        Some(Looped { module: Some(module), f, tables, counts, compile_s })
    }

    /// Run it on `base` (laid out as the production kernel's: the slots `idx` names).
    ///
    /// # Safety
    /// As [`Tiered::call`].
    pub(super) unsafe fn call(&self, base: *mut f64, idx: *const u32, lits: *const f64) {
        thread_local! {
            static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
        }
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < self.counts.scratch.max(1) {
                s.resize(self.counts.scratch.max(1), 0.0);
            }
            (self.f)(base, idx, lits, s.as_mut_ptr(), self.tables.as_ptr());
        });
    }
}

#[cfg(test)]
#[path = "kernel_jit_lanes_tests.rs"]
mod tests;
