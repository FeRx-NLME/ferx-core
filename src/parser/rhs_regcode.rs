//! Register code for the ODE right-hand side.
//!
//! The RHS runs at every stage of every step of every solve — f64 objective, the adjoint's
//! dual recordings, the covariance step's nested duals — so its cost is most of an ODE fit's
//! time. Measured on `two_cpt_oral_ode` under RK45, the bytecode interpreter plus the
//! per-call `vars` binding were 71% of a FOCEI fit's CPU.
//!
//! [`RhsRegProgram::compile`] translates a straight-line RHS (`AssignBc` / `DiffEqBc`
//! statements, no jumps) once into three-address instructions over one register file. The
//! states, the individual parameters and the time seeds are loaded straight from the RHS's
//! own arguments (`u`, `params`, `seeds`; see [`RhsLayout`]), so a call needs neither the
//! thread-local scratch nor the `vars` binding, and each operation is one dispatch.
//!
//! It is a translation, not an optimisation: every instruction applies the operation the
//! interpreter applies at that point, through the same primitive (the `VmArith` in-place
//! methods with `Op::Div`'s guard, [`super::apply_scalar_in`], the `PkNum` methods), with the
//! interpreter's operand fusion reproduced exactly — a constant or `TIME` right operand is
//! applied as a scalar, a variable right operand in place, everything else as a dual
//! constant. So every value, and every tape node a taped type records, is bit for bit the
//! interpreter's. Nothing is shared between subexpressions (sharing a taped node would change
//! the order a reverse sweep accumulates adjoints in), and the hoist's parameter-only values
//! are recomputed rather than cached (a cache keyed on the parameters measured no faster).
//!
//! A program with any operation whose `f64` meaning the generic primitives do not reproduce
//! exactly (`sqrt`'s guard, `inv_logit`, `logit`, `mod`), that reads anything but its states,
//! parameters, seeds, its own locals, literals and `TIME` (θ, η, covariates, gathers, network
//! outputs, the mixture class), or that reads a local before assigning it, is not compiled and
//! keeps the interpreter.
use super::{apply_scalar_in, Bytecode, Op, Statement};
use crate::sens::num::vm::VmNum;

/// An instruction operand.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Src {
    /// `vars[i]`, read when the instruction runs (a statement's own expression never
    /// sees its assignment, so this is the value the interpreter pushed).
    Var(u32),
    /// A temporary written by an earlier instruction of the same call.
    Tmp(u32),
    /// A literal, as the dual constant the interpreter pushes.
    Konst(f64),
    /// The `TIME` built-in, as the dual constant the interpreter pushes.
    Time,
}

/// A binary arithmetic operation (`Op::is_arith_binary`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arith {
    Add,
    Sub,
    RSub,
    Mul,
    Div,
}

impl Arith {
    fn of(op: Op) -> Option<Self> {
        Some(match op {
            Op::Add => Arith::Add,
            Op::Sub => Arith::Sub,
            Op::RSub => Arith::RSub,
            Op::Mul => Arith::Mul,
            Op::Div => Arith::Div,
            _ => return None,
        })
    }
    fn op(self) -> Op {
        match self {
            Arith::Add => Op::Add,
            Arith::Sub => Op::Sub,
            Arith::RSub => Op::RSub,
            Arith::Mul => Op::Mul,
            Arith::Div => Op::Div,
        }
    }
}

/// A unary operation applied to a materialised value.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Unary {
    Exp,
    Ln,
    Abs,
    Floor,
    Ceil,
    Round,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Ins {
    /// `tmp[dst] = a; apply_binary_in(tmp[dst], op, b)`.
    Bin {
        dst: u32,
        op: Arith,
        a: Src,
        b: Src,
    },
    /// `tmp[dst] = a; apply_scalar_in(tmp[dst], op, c)`: a constant right operand.
    Scalar {
        dst: u32,
        op: Arith,
        a: Src,
        c: f64,
    },
    /// `apply_scalar_in` with `TIME` as the constant.
    ScalarTime {
        dst: u32,
        op: Arith,
        a: Src,
    },
    /// `tmp[dst] = a.pow(e)`.
    Pow {
        dst: u32,
        a: Src,
        e: Src,
    },
    Unary {
        dst: u32,
        op: Unary,
        a: Src,
    },
    /// `vars[slot] = v`.
    Assign {
        slot: u32,
        v: Src,
    },
    /// `du[state] = v`.
    Deriv {
        state: u32,
        v: Src,
    },
}

/// A lowered instruction: every operand is an index into the one register file, so an
/// instruction is a single dispatch. The arithmetic primitive of each arm is the
/// interpreter's own for that operation and operand form (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Flat {
    /// `r[d] = u[i]` (`0` past `u.len()`, as the binding fills a short `u`).
    LoadState {
        d: u16,
        i: u32,
    },
    /// `r[d] = params[slot]` (`0` past `params.len()`, as the binding reads it).
    LoadParam {
        d: u16,
        slot: u32,
    },
    /// `r[d] = seeds[k]`: the `T`, `TAFD`, `TAD` and `MACHEPS` slots.
    LoadSeed {
        d: u16,
        k: u8,
    },
    /// `r[d] = k(c)`: a literal the interpreter pushes as a dual constant.
    LoadK {
        d: u16,
        c: f64,
    },
    /// `r[d] = k(TIME)`.
    LoadTime {
        d: u16,
    },
    Add {
        d: u16,
        a: u16,
        b: u16,
    },
    Sub {
        d: u16,
        a: u16,
        b: u16,
    },
    RSub {
        d: u16,
        a: u16,
        b: u16,
    },
    Mul {
        d: u16,
        a: u16,
        b: u16,
    },
    Div {
        d: u16,
        a: u16,
        b: u16,
    },
    AddK {
        d: u16,
        a: u16,
        c: f64,
    },
    SubK {
        d: u16,
        a: u16,
        c: f64,
    },
    RSubK {
        d: u16,
        a: u16,
        c: f64,
    },
    MulK {
        d: u16,
        a: u16,
        c: f64,
    },
    /// A division by a literal `|c| ≥ 1e-30` (a smaller one is `Zero`).
    DivK {
        d: u16,
        a: u16,
        c: f64,
    },
    /// `r[d] = k(0)`: `Div`'s guard decided at compile time.
    Zero {
        d: u16,
    },
    /// `apply_scalar_in` with `TIME` as the constant.
    ScalarTime {
        d: u16,
        op: Arith,
        a: u16,
    },
    Pow {
        d: u16,
        a: u16,
        e: u16,
    },
    Unary {
        d: u16,
        op: Unary,
        a: u16,
    },
    /// `du[state] = r[a]`.
    Deriv {
        state: u32,
        a: u16,
    },
}

/// Where each of the RHS's `vars` slots comes from, as `OdeRhsProgram`'s binding fills it:
/// the states, the individual parameters (each from its `params` slot), the four seeded
/// time/constant slots. Every other slot is a local the program assigns.
pub(super) struct RhsLayout<'a> {
    pub(super) n_vars: usize,
    pub(super) state_count: usize,
    pub(super) indiv_to_params_slot: &'a [usize],
    /// `[T, TAFD, TAD, MACHEPS]` slots, in the order [`RhsRegProgram::run`]'s `seeds`.
    pub(super) seed_slots: [usize; 4],
}

impl RhsLayout<'_> {
    fn source(&self, slot: usize) -> Option<Flat> {
        let i = u32::try_from(slot).ok()?;
        if slot < self.state_count {
            return Some(Flat::LoadState { d: 0, i });
        }
        if let Some(&p) = self.indiv_to_params_slot.get(slot - self.state_count) {
            return Some(Flat::LoadParam {
                d: 0,
                slot: u32::try_from(p).ok()?,
            });
        }
        let k = self.seed_slots.iter().position(|&s| s == slot)?;
        Some(Flat::LoadSeed { d: 0, k: k as u8 })
    }
}

/// A compiled straight-line RHS; see the module docs.
#[derive(Clone, Debug)]
pub(super) struct RhsRegProgram {
    code: Vec<Flat>,
    n_regs: usize,
    reads_time: bool,
}

impl RhsRegProgram {
    /// [`Self::compile`], except under the test-only `RHS_REGCODE_DISABLED`.
    pub(super) fn compile_unless_disabled(
        stmts: &[Statement],
        layout: &RhsLayout<'_>,
    ) -> Option<Self> {
        #[cfg(test)]
        if super::RHS_REGCODE_DISABLED.with(|c| c.get()) {
            return None;
        }
        Self::compile(stmts, layout)
    }

    /// Compile `stmts` for a `vars` buffer of length `n_vars`, or `None` when the program
    /// is outside the translatable shape (see the module docs) — the caller keeps the
    /// interpreter for it.
    pub(super) fn compile(stmts: &[Statement], layout: &RhsLayout<'_>) -> Option<Self> {
        let n_vars = layout.n_vars;
        let mut b = Builder {
            ins: Vec::new(),
            n_tmp: 0,
            n_vars,
            reads_time: false,
        };
        for s in stmts {
            match s {
                Statement::AssignBc(slot, bc) => {
                    let v = b.expr(bc)?;
                    // The interpreter drops a write past the end of `vars`.
                    if *slot < n_vars {
                        b.ins.push(Ins::Assign {
                            slot: u32::try_from(*slot).ok()?,
                            v,
                        });
                    }
                }
                Statement::DiffEqBc(state, bc) => {
                    let v = b.expr(bc)?;
                    b.ins.push(Ins::Deriv {
                        state: u32::try_from(*state).ok()?,
                        v,
                    });
                }
                _ => return None,
            }
        }
        lower(&b.ins, b.n_tmp, b.reads_time, layout)
    }

    /// Registers the program uses; [`Self::run`]'s `regs` must hold at least this many.
    pub(super) fn n_regs(&self) -> usize {
        self.n_regs
    }

    /// Run the program on the RHS's own inputs: the `du` the interpreter writes after the
    /// binding fills `vars` from `u`, `params` and `seeds`, bit for bit. `regs` is scratch of
    /// at least [`Self::n_regs`] values. `TIME` reads the model-time thread-local, as the
    /// interpreter does.
    #[inline]
    pub(super) fn run<T: VmNum>(
        &self,
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        regs: &mut [T],
    ) {
        let time = if self.reads_time {
            super::current_model_time()
        } else {
            0.0
        };
        self.exec(
            &self.code,
            u,
            params,
            seeds,
            du,
            &mut regs[..self.n_regs],
            time,
        );
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn exec<T: VmNum>(
        &self,
        code: &[Flat],
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        r: &mut [T],
        time: f64,
    ) {
        let k = T::from_f64;
        macro_rules! bin {
            ($d:expr, $a:expr, $f:expr) => {{
                let mut x = r[$a as usize];
                $f(&mut x);
                r[$d as usize] = x;
            }};
        }
        for f in code {
            match *f {
                Flat::LoadState { d, i } => {
                    r[d as usize] = u.get(i as usize).copied().unwrap_or(k(0.0))
                }
                Flat::LoadParam { d, slot } => {
                    r[d as usize] = params.get(slot as usize).copied().unwrap_or(k(0.0))
                }
                Flat::LoadSeed { d, k: i } => r[d as usize] = seeds[i as usize],
                Flat::LoadK { d, c } => r[d as usize] = k(c),
                Flat::LoadTime { d } => r[d as usize] = k(time),
                Flat::Add { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| x.add_in(&y))
                }
                Flat::Sub { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| x.sub_in(&y))
                }
                Flat::RSub { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| x.rsub_in(&y))
                }
                Flat::Mul { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| x.mul_in(&y))
                }
                Flat::Div { d, a, b } => {
                    let y = r[b as usize];
                    if y.val().abs() < 1e-30 {
                        r[d as usize] = k(0.0);
                    } else {
                        bin!(d, a, |x: &mut T| x.div_in(&y))
                    }
                }
                Flat::AddK { d, a, c } => bin!(d, a, |x: &mut T| x.add_scalar_in(c)),
                Flat::SubK { d, a, c } => bin!(d, a, |x: &mut T| x.sub_scalar_in(c)),
                Flat::RSubK { d, a, c } => bin!(d, a, |x: &mut T| x.rsub_scalar_in(c)),
                Flat::MulK { d, a, c } => bin!(d, a, |x: &mut T| x.mul_scalar_in(c)),
                Flat::DivK { d, a, c } => bin!(d, a, |x: &mut T| x.div_scalar_in(c)),
                Flat::Zero { d } => r[d as usize] = k(0.0),
                Flat::ScalarTime { d, op, a } => {
                    bin!(d, a, |x: &mut T| apply_scalar_in(x, op.op(), time))
                }
                Flat::Pow { d, a, e } => r[d as usize] = r[a as usize].pow(r[e as usize]),
                Flat::Unary { d, op, a } => {
                    let x = r[a as usize];
                    r[d as usize] = match op {
                        Unary::Exp => x.exp(),
                        Unary::Ln => x.guard_floor(1e-30).ln(),
                        Unary::Abs => x.abs(),
                        Unary::Floor => k(x.val().floor()),
                        Unary::Ceil => k(x.val().ceil()),
                        Unary::Round => k(x.val().round()),
                    };
                }
                Flat::Deriv { state, a } => {
                    if let Some(x) = du.get_mut(state as usize) {
                        *x = r[a as usize];
                    }
                }
            }
        }
    }
}

/// Lower the builder's operand-form IR to [`Flat`] code over one register file: temps keep
/// their numbers, a slot read before any assignment to it is loaded once in a prologue, a
/// slot read after one reads the assigned value's register (the interpreter's `PushVar`
/// sees the assignment, and so does this), and each unfused literal or `TIME` gets a
/// register of its own.
fn lower(
    ins: &[Ins],
    n_tmp: usize,
    reads_time: bool,
    layout: &RhsLayout<'_>,
) -> Option<RhsRegProgram> {
    let mut next = n_tmp;
    let mut alloc = || -> Option<u16> {
        let r = u16::try_from(next).ok()?;
        next += 1;
        Some(r)
    };
    let mut prologue: Vec<Flat> = Vec::new();
    let mut body: Vec<Flat> = Vec::new();
    // Current register of a slot: its prologue load, or the value last assigned to it.
    let mut slot_reg: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();
    let mut time_reg: Option<u16> = None;
    let mut operand = |s: Src,
                       slot_reg: &mut std::collections::HashMap<u32, u16>,
                       prologue: &mut Vec<Flat>,
                       body: &mut Vec<Flat>|
     -> Option<u16> {
        Some(match s {
            Src::Tmp(t) => u16::try_from(t).ok()?,
            Src::Var(slot) => match slot_reg.get(&slot) {
                Some(&r) => r,
                // A slot read before any assignment: one of the binding's inputs, loaded
                // once. A local read before the program assigns it has no source here
                // (the interpreter would read what the buffer held), so it is declined.
                None => {
                    let d = alloc()?;
                    let load = match layout.source(slot as usize)? {
                        Flat::LoadState { i, .. } => Flat::LoadState { d, i },
                        Flat::LoadParam { slot, .. } => Flat::LoadParam { d, slot },
                        Flat::LoadSeed { k, .. } => Flat::LoadSeed { d, k },
                        _ => return None,
                    };
                    prologue.push(load);
                    slot_reg.insert(slot, d);
                    d
                }
            },
            Src::Konst(c) => {
                let d = alloc()?;
                body.push(Flat::LoadK { d, c });
                d
            }
            Src::Time => match time_reg {
                Some(r) => r,
                None => {
                    let d = alloc()?;
                    prologue.push(Flat::LoadTime { d });
                    time_reg = Some(d);
                    d
                }
            },
        })
    };
    for i in ins {
        let f = match *i {
            Ins::Bin { dst, op, a, b } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body)?;
                let b = operand(b, &mut slot_reg, &mut prologue, &mut body)?;
                let d = u16::try_from(dst).ok()?;
                match op {
                    Arith::Add => Flat::Add { d, a, b },
                    Arith::Sub => Flat::Sub { d, a, b },
                    Arith::RSub => Flat::RSub { d, a, b },
                    Arith::Mul => Flat::Mul { d, a, b },
                    Arith::Div => Flat::Div { d, a, b },
                }
            }
            Ins::Scalar { dst, op, a, c } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body)?;
                let d = u16::try_from(dst).ok()?;
                match op {
                    Arith::Add => Flat::AddK { d, a, c },
                    Arith::Sub => Flat::SubK { d, a, c },
                    Arith::RSub => Flat::RSubK { d, a, c },
                    Arith::Mul => Flat::MulK { d, a, c },
                    Arith::Div if c.abs() < 1e-30 => Flat::Zero { d },
                    Arith::Div => Flat::DivK { d, a, c },
                }
            }
            Ins::ScalarTime { dst, op, a } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body)?;
                Flat::ScalarTime {
                    d: u16::try_from(dst).ok()?,
                    op,
                    a,
                }
            }
            Ins::Pow { dst, a, e } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body)?;
                let e = operand(e, &mut slot_reg, &mut prologue, &mut body)?;
                Flat::Pow {
                    d: u16::try_from(dst).ok()?,
                    a,
                    e,
                }
            }
            Ins::Unary { dst, op, a } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body)?;
                Flat::Unary {
                    d: u16::try_from(dst).ok()?,
                    op,
                    a,
                }
            }
            Ins::Assign { slot, v } => {
                // A local lives in its register: later reads of the slot see this value, as
                // the interpreter's do. Assigning a binding input is a parse error, and a
                // program that did would decline here rather than shadow it.
                if layout.source(slot as usize).is_some() {
                    return None;
                }
                let a = operand(v, &mut slot_reg, &mut prologue, &mut body)?;
                slot_reg.insert(slot, a);
                continue;
            }
            Ins::Deriv { state, v } => {
                let a = operand(v, &mut slot_reg, &mut prologue, &mut body)?;
                Flat::Deriv { state, a }
            }
        };
        body.push(f);
    }
    // The prologue reads every slot at its call-entry value; an assignment renames the slot
    // for the instructions after it, so no prologue load can observe a later write.
    prologue.extend(body);
    Some(RhsRegProgram {
        code: prologue,
        n_regs: next,
        reads_time,
    })
}

struct Builder {
    ins: Vec<Ins>,
    n_tmp: usize,
    n_vars: usize,
    reads_time: bool,
}

impl Builder {
    fn tmp(&mut self) -> Option<u32> {
        let t = u32::try_from(self.n_tmp).ok()?;
        self.n_tmp += 1;
        Some(t)
    }

    /// One expression, simulating `eval_bytecode_g`'s stack symbolically; the operand
    /// left on the stack, or `None` when the bytecode is outside the translatable shape.
    fn expr(&mut self, bc: &Bytecode) -> Option<Src> {
        let ops = bc.ops.as_slice();
        let consts = bc.constants.as_slice();
        let mut stack: Vec<Src> = Vec::with_capacity(bc.max_stack);
        let mut pc = 0;
        while pc < ops.len() {
            let fused = if stack.is_empty() {
                None
            } else {
                ops.get(pc + 1).copied().and_then(Arith::of)
            };
            match (ops[pc], fused) {
                // Operand fusion, exactly as `eval_bytecode_g` applies it: a constant or
                // `TIME` consumed by the next arithmetic op is applied as a scalar, a
                // variable in place. (`PushCov` fuses too there, but the RHS reads none.)
                (Op::PushConst(i), Some(op)) => {
                    let a = stack.pop()?;
                    let c = *consts.get(i as usize)?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::Scalar { dst, op, a, c });
                    stack.push(Src::Tmp(dst));
                    pc += 2;
                    continue;
                }
                (Op::PushTime, Some(op)) => {
                    self.reads_time = true;
                    let a = stack.pop()?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::ScalarTime { dst, op, a });
                    stack.push(Src::Tmp(dst));
                    pc += 2;
                    continue;
                }
                (Op::PushVar(i), Some(op)) if (i as usize) < self.n_vars => {
                    let a = stack.pop()?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::Bin {
                        dst,
                        op,
                        a,
                        b: Src::Var(i),
                    });
                    stack.push(Src::Tmp(dst));
                    pc += 2;
                    continue;
                }
                (Op::PushConst(i), None) => stack.push(Src::Konst(*consts.get(i as usize)?)),
                (Op::PushTime, None) => {
                    self.reads_time = true;
                    stack.push(Src::Time);
                }
                // The interpreter reads an out-of-range slot as 0; not worth a variant.
                (Op::PushVar(i), _) if (i as usize) < self.n_vars => stack.push(Src::Var(i)),
                (op @ (Op::Add | Op::Sub | Op::RSub | Op::Mul | Op::Div), _) => {
                    let b = stack.pop()?;
                    let a = stack.pop()?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::Bin {
                        dst,
                        op: Arith::of(op)?,
                        a,
                        b,
                    });
                    stack.push(Src::Tmp(dst));
                }
                (Op::Pow, _) => {
                    let e = stack.pop()?;
                    let a = stack.pop()?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::Pow { dst, a, e });
                    stack.push(Src::Tmp(dst));
                }
                (op @ (Op::Exp | Op::Ln | Op::Abs | Op::Floor | Op::Ceil | Op::Round), _) => {
                    let a = stack.pop()?;
                    let dst = self.tmp()?;
                    let op = match op {
                        Op::Exp => Unary::Exp,
                        Op::Ln => Unary::Ln,
                        Op::Abs => Unary::Abs,
                        Op::Floor => Unary::Floor,
                        Op::Ceil => Unary::Ceil,
                        _ => Unary::Round,
                    };
                    self.ins.push(Ins::Unary { dst, op, a });
                    stack.push(Src::Tmp(dst));
                }
                // Everything else keeps the interpreter: jumps — and with them the comparisons,
                // connectives and `present()`, which only an inline `if`'s condition emits —
                // the reads the RHS is not
                // handed (θ, η, covariates, gathers, network outputs, the mixture class),
                // an out-of-range variable, and the ops whose `f64` meaning the generic
                // primitives do not reproduce bit for bit (`Sqrt`'s `-0.0` guard,
                // `InvLogit`, `Logit`, `Mod`). Listed one by one: a new `Op` must be
                // decided here, not swallowed by a wildcard.
                (
                    Op::PushVar(_)
                    | Op::PushTheta(_)
                    | Op::PushEta(_)
                    | Op::PushCov(_)
                    | Op::PushMixNum
                    | Op::PushNnOutput(_, _)
                    | Op::PushThetaGather(_)
                    | Op::Sqrt
                    | Op::InvLogit
                    | Op::Logit
                    | Op::Mod
                    | Op::CmpLt
                    | Op::CmpLe
                    | Op::CmpGt
                    | Op::CmpGe
                    | Op::CmpEq
                    | Op::CmpNe
                    | Op::LogicAnd
                    | Op::LogicOr
                    | Op::LogicNot
                    | Op::IsPresent
                    | Op::JumpIfFalse(_)
                    | Op::Jump(_),
                    _,
                ) => return None,
            }
            pc += 1;
        }
        // A balanced program leaves exactly one operand.
        (stack.len() == 1).then(|| stack[0])
    }
}

#[cfg(test)]
#[path = "rhs_regcode_tests.rs"]
mod tests;
