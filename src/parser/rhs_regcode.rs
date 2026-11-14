//! Register code for the ODE right-hand side.
//!
//! The RHS runs at every stage of every step of every solve — f64 objective, the adjoint's
//! dual recordings, the covariance step's nested duals — so its cost is most of an ODE fit's
//! time. Measured on `two_cpt_oral_ode` under RK45, the bytecode interpreter plus the
//! per-call `vars` binding were 71% of a FOCEI fit's CPU.
//!
//! [`RhsRegProgram::compile`] translates an RHS of `AssignBc` / `DiffEqBc` statements once
//! into three-address instructions over one register file. The
//! states, the individual parameters and the time seeds are loaded straight from the RHS's
//! own arguments (`u`, `params`, `seeds`; see [`RhsLayout`]), so a call needs neither the
//! thread-local scratch nor the `vars` binding, and each operation is one dispatch.
//!
//! It is a translation, not an optimisation: every instruction applies the operation the
//! interpreter applies at that point, through the interpreter's own primitive
//! (`apply_binary_in` with `Op::Div`'s guard, `apply_scalar_in`, `apply_unary`, the
//! `VmArith` in-place methods), with the interpreter's operand fusion reproduced exactly — a
//! constant right operand is applied as a scalar, a variable right operand in place, everything else as a dual
//! constant. So every value, and every tape node a taped type records, is bit for bit the
//! interpreter's. Nothing is shared between subexpressions (sharing a taped node would change
//! the order a reverse sweep accumulates adjoints in), and the hoist's parameter-only values
//! are recomputed rather than cached (a cache keyed on the parameters measured no faster).
//!
//! An inline `if` (and the `min`/`max`/`clamp` that desugar to one) is a real branch here:
//! the condition's operands are computed, the comparison writes the dual constant `0`/`1` the
//! interpreter pushes, and only the arm the interpreter takes runs, its result moved into a
//! register both arms write. So a taped type records the live arm's nodes and no others. The
//! native code and its reverse pass compute both arms and select (`rhs_ir`), which is the same
//! values lane for lane; see `rhs_ir`'s docs for why a dead arm cannot leak into a derivative.
//!
//! A program with any operation whose `f64` meaning the generic primitives do not reproduce
//! exactly (`sqrt`'s guard, `inv_logit`, `logit`, `mod`), that reads anything but its states,
//! parameters, seeds, its own locals and literals (θ, η, covariates, gathers, network
//! outputs, the mixture class; `TIME` is the time seed by the time a program is compiled), or
//! that reads a local before assigning it, is not compiled and keeps the interpreter.
use super::{apply_binary_in, apply_scalar_in, apply_unary, Bytecode, Op, Statement};
use crate::sens::num::vm::VmNum;
use ir::Cc;

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
}

/// A unary operation applied to a materialised value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Unary {
    Exp,
    Ln,
    /// `PkNum::ln`, unguarded: kernel recordings only. [`Unary::Ln`] is the interpreter's
    /// `Op::Ln`, floored at `1e-30` (the VM's `guard_floor`).
    LnRaw,
    /// Kernel recordings only: `sin`, the polygammas (the derivatives of `LnGamma`), and the
    /// `k`-th derivative of `exprel(z) = (e^z − 1)/z` (`ode::linear::exprel_derivs`).
    Sin,
    Digamma,
    Trigamma,
    Tetragamma,
    ExpRel(u8),
    /// The `k`-th derivative of a function given by its jet (`JetFn`): kernel recordings only.
    Jet(crate::sens::num::JetFn, u8),
    /// The value as a constant (`PkNum::value_only`): kernel recordings only.
    StopGrad,
    /// `PkNum::sqrt`, unguarded. Only a kernel recording (`kernel_jit`, #2786) emits it; the
    /// RHS compiler keeps `Op::Sqrt`'s `-0.0` guard on the interpreter, so this arm's
    /// `Op::Sqrt` is reached by no register program the interpreter runs.
    Sqrt,
    /// `PkNum::inv_logit`, `logit`, `cos`, `acos` (unguarded): kernel recordings only.
    InvLogit,
    Logit,
    Cos,
    Acos,
    LnGamma,
    Abs,
    Floor,
    Ceil,
    Round,
}

impl Unary {
    fn op(self) -> Op {
        match self {
            Unary::Exp => Op::Exp,
            Unary::Ln => Op::Ln,
            Unary::Sqrt => Op::Sqrt,
            Unary::InvLogit => Op::InvLogit,
            Unary::Logit => Op::Logit,
            Unary::Cos => Op::Cos,
            Unary::Acos
            | Unary::LnGamma
            | Unary::LnRaw
            | Unary::Sin
            | Unary::Digamma
            | Unary::Trigamma
            | Unary::Tetragamma
            | Unary::ExpRel(_)
            | Unary::Jet(..)
            | Unary::StopGrad => {
                unreachable!("only a kernel recording emits these, never interpreted")
            }
            Unary::Abs => Op::Abs,
            Unary::Floor => Op::Floor,
            Unary::Ceil => Op::Ceil,
            Unary::Round => Op::Round,
        }
    }
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
    /// `tmp[dst] = k(1)` when `a cc b` holds on the values, else `k(0)` (`Op::CmpLt`, …).
    Cmp {
        dst: u32,
        cc: Cc,
        a: Src,
        b: Src,
    },
    /// `LogicAnd` / `LogicOr` of two `0`/`1` values.
    Logic {
        dst: u32,
        or: bool,
        a: Src,
        b: Src,
    },
    /// `LogicNot`.
    Not {
        dst: u32,
        a: Src,
    },
    /// `IsPresent`: `k(1)` unless the value is `NaN`.
    Present {
        dst: u32,
        a: Src,
    },
    /// An inline `if`: the instructions up to the matching [`Ins::Else`] run when `c`'s value
    /// is not `0` (`Op::JumpIfFalse`), the ones from there to [`Ins::EndIf`] otherwise.
    Branch {
        c: Src,
    },
    Else,
    EndIf,
    /// `tmp[dst] = v`: an arm's result, in the temporary both arms of an `if` write.
    Move {
        dst: u32,
        v: Src,
    },
}

/// A lowered instruction: every operand is an index into the one register file, so an
/// instruction is a single dispatch. The arithmetic primitive of each arm is the
/// interpreter's own for that operation and operand form (see the module docs).
/// A register of a [`Flat`] program: a recording's (#2786) can number past 65k.
pub(super) type Reg = u32;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Flat {
    /// `r[d] = u[i]` (`0` past `u.len()`, as the binding fills a short `u`).
    LoadState {
        d: Reg,
        i: u32,
    },
    /// `r[d] = params[slot]` (`0` past `params.len()`, as the binding reads it).
    LoadParam {
        d: Reg,
        slot: u32,
    },
    /// `r[d] = seeds[k]`: the `T`, `TAFD`, `TAD` and `MACHEPS` slots; for a kernel
    /// recording (#2786), its literal pool.
    LoadSeed {
        d: Reg,
        k: u32,
    },
    /// `r[d] = k(c)`: a literal the interpreter pushes as a dual constant.
    LoadK {
        d: Reg,
        c: f64,
    },
    Add {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    Sub {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    RSub {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    Mul {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    Div {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    /// `a / b` unguarded — `PkNum`'s division (the forward dual's `div`), which a kernel recording
    /// (`kernel_jit`, #2786) emits; `Div` is the interpreter's guarded one.
    DivRaw {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    /// Kernel recordings only (#2786): a host call. `HostArg`s queue its arguments' values in
    /// order, `Host` runs the analysis on them, and each `HostOut` reads one of its outputs, a
    /// constant of the computation (no derivative).
    HostArg {
        a: Reg,
    },
    Host {
        f: crate::sens::num::HostFn,
    },
    HostOut {
        d: Reg,
        k: u32,
    },
    /// Kernel recordings only (#2786): an ODE segment as a loop call. `LoopArg`s list its
    /// inputs (the initial state, then the outer values its bodies read), `Loop` names the
    /// recording's loop `id`, and each `LoopOut` defines one output (a state component at a
    /// save point). Never inside a chunk: the plan runs the call itself.
    /// Kernel recordings only (#2786): `r[d] = r[c] ≠ 0 ? r[a] : r[b]` (`c` a condition).
    Select {
        d: Reg,
        c: Reg,
        a: Reg,
        b: Reg,
    },
    /// Kernel recordings only (#2786, `PkNum::order_gate`): `r[d] = r[b]` in a compiled
    /// variant whose derivative order reaches `k`, `r[a]` below it. Resolved per variant
    /// before lowering (`kernel_jit::resolve_gates`).
    Gate {
        d: Reg,
        a: Reg,
        b: Reg,
        k: u8,
    },
    LoopArg {
        a: Reg,
    },
    Loop {
        id: u32,
    },
    LoopOut {
        d: Reg,
        k: u32,
    },
    /// `∂^(i+j) P(a, x) / ∂aⁱ ∂xʲ` (of `Q` when `upper`); `i = j = 0` is [`PkNum::gamma_pq`]
    /// as one primitive. Only a kernel recording emits it (#2786).
    GammaPq {
        d: Reg,
        a: Reg,
        x: Reg,
        upper: bool,
        i: u8,
        j: u8,
    },
    AddK {
        d: Reg,
        a: Reg,
        c: f64,
    },
    SubK {
        d: Reg,
        a: Reg,
        c: f64,
    },
    RSubK {
        d: Reg,
        a: Reg,
        c: f64,
    },
    MulK {
        d: Reg,
        a: Reg,
        c: f64,
    },
    /// A division by a literal, through `apply_scalar_in`'s guard.
    DivK {
        d: Reg,
        a: Reg,
        c: f64,
    },
    Pow {
        d: Reg,
        a: Reg,
        e: Reg,
    },
    Unary {
        d: Reg,
        op: Unary,
        a: Reg,
    },
    /// `du[state] = r[a]`.
    Deriv {
        state: u32,
        a: Reg,
    },
    /// `r[d] = k(1)` when `r[a].val() cc r[b].val()`, else `k(0)`: the interpreter's `Cmp*`.
    Cmp {
        d: Reg,
        cc: Cc,
        a: Reg,
        b: Reg,
    },
    /// `r[d] = k(1)` when both values are nonzero (`LogicAnd`).
    And {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    /// `r[d] = k(1)` when either value is nonzero (`LogicOr`).
    Or {
        d: Reg,
        a: Reg,
        b: Reg,
    },
    /// `r[d] = k(1)` when the value is zero (`LogicNot`).
    Not {
        d: Reg,
        a: Reg,
    },
    /// `r[d] = k(1)` unless the value is `NaN` (`IsPresent`).
    Present {
        d: Reg,
        a: Reg,
    },
    /// Continue at `to` (the else arm) when `r[c].val() == 0` (`JumpIfFalse`). The then arm
    /// runs from the next instruction to a [`Flat::Jump`] at `to − 1`, which continues past the
    /// else arm; each arm ends with the [`Flat::Move`]s of its results.
    Branch {
        c: Reg,
        to: u32,
    },
    Jump {
        to: u32,
    },
    /// `r[d] = r[a]`: an arm's result into the register both arms of an `if` write.
    Move {
        d: Reg,
        a: Reg,
    },
}

/// Structural hashing of an instruction, literals by their bits ([`Flat::same`]'s notion of
/// equality): what the compiled path's caches key programs on (#2786), with no text rendering.
impl std::hash::Hash for Flat {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        std::mem::discriminant(self).hash(h);
        match *self {
            Flat::LoadState { d, i } => {
                d.hash(h);
                i.hash(h);
            }
            Flat::LoadParam { d, slot } => {
                d.hash(h);
                slot.hash(h);
            }
            Flat::LoadSeed { d, k } => {
                d.hash(h);
                k.hash(h);
            }
            Flat::LoadK { d, c } => {
                d.hash(h);
                c.to_bits().hash(h);
            }
            Flat::Add { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Sub { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::RSub { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Mul { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Div { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::DivRaw { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::HostArg { a } => {
                a.hash(h);
            }
            Flat::Host { f } => {
                f.hash(h);
            }
            Flat::HostOut { d, k } => {
                d.hash(h);
                k.hash(h);
            }
            Flat::Select { d, c, a, b } => {
                d.hash(h);
                c.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Gate { d, a, b, k } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
                k.hash(h);
            }
            Flat::LoopArg { a } => {
                a.hash(h);
            }
            Flat::Loop { id } => {
                id.hash(h);
            }
            Flat::LoopOut { d, k } => {
                d.hash(h);
                k.hash(h);
            }
            Flat::GammaPq {
                d,
                a,
                x,
                upper,
                i,
                j,
            } => {
                d.hash(h);
                a.hash(h);
                x.hash(h);
                upper.hash(h);
                i.hash(h);
                j.hash(h);
            }
            Flat::AddK { d, a, c } => {
                d.hash(h);
                a.hash(h);
                c.to_bits().hash(h);
            }
            Flat::SubK { d, a, c } => {
                d.hash(h);
                a.hash(h);
                c.to_bits().hash(h);
            }
            Flat::RSubK { d, a, c } => {
                d.hash(h);
                a.hash(h);
                c.to_bits().hash(h);
            }
            Flat::MulK { d, a, c } => {
                d.hash(h);
                a.hash(h);
                c.to_bits().hash(h);
            }
            Flat::DivK { d, a, c } => {
                d.hash(h);
                a.hash(h);
                c.to_bits().hash(h);
            }
            Flat::Pow { d, a, e } => {
                d.hash(h);
                a.hash(h);
                e.hash(h);
            }
            Flat::Unary { d, op, a } => {
                d.hash(h);
                op.hash(h);
                a.hash(h);
            }
            Flat::Deriv { state, a } => {
                state.hash(h);
                a.hash(h);
            }
            Flat::Cmp { d, cc, a, b } => {
                d.hash(h);
                cc.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::And { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Or { d, a, b } => {
                d.hash(h);
                a.hash(h);
                b.hash(h);
            }
            Flat::Not { d, a } => {
                d.hash(h);
                a.hash(h);
            }
            Flat::Present { d, a } => {
                d.hash(h);
                a.hash(h);
            }
            Flat::Branch { c, to } => {
                c.hash(h);
                to.hash(h);
            }
            Flat::Jump { to } => {
                to.hash(h);
            }
            Flat::Move { d, a } => {
                d.hash(h);
                a.hash(h);
            }
        }
    }
}

impl Flat {
    /// Equal instruction for instruction, literals bit for bit (`-0.0` is not `0.0`, a `NaN`
    /// is itself): two programs that are `same` compute the same values.
    pub(super) fn same(&self, other: &Flat) -> bool {
        match (*self, *other) {
            (Flat::LoadState { d: d0, i: i0 }, Flat::LoadState { d: d1, i: i1 }) => {
                d0 == d1 && i0 == i1
            }
            (Flat::LoadParam { d: d0, slot: slot0 }, Flat::LoadParam { d: d1, slot: slot1 }) => {
                d0 == d1 && slot0 == slot1
            }
            (Flat::LoadSeed { d: d0, k: k0 }, Flat::LoadSeed { d: d1, k: k1 }) => {
                d0 == d1 && k0 == k1
            }
            (Flat::LoadK { d: d0, c: c0 }, Flat::LoadK { d: d1, c: c1 }) => {
                d0 == d1 && c0.to_bits() == c1.to_bits()
            }
            (
                Flat::Add {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::Add {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::Sub {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::Sub {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::RSub {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::RSub {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::Mul {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::Mul {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::Div {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::Div {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::DivRaw {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::DivRaw {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (Flat::HostArg { a: a0 }, Flat::HostArg { a: a1 }) => a0 == a1,
            (Flat::Host { f: f0 }, Flat::Host { f: f1 }) => f0 == f1,
            (Flat::HostOut { d: d0, k: k0 }, Flat::HostOut { d: d1, k: k1 }) => {
                d0 == d1 && k0 == k1
            }
            (
                Flat::Select {
                    d: d0,
                    c: c0,
                    a: a0,
                    b: b0,
                },
                Flat::Select {
                    d: d1,
                    c: c1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && c0 == c1 && a0 == a1 && b0 == b1,
            (
                Flat::Gate {
                    d: d0,
                    a: a0,
                    b: b0,
                    k: k0,
                },
                Flat::Gate {
                    d: d1,
                    a: a1,
                    b: b1,
                    k: k1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1 && k0 == k1,
            (Flat::LoopArg { a: a0 }, Flat::LoopArg { a: a1 }) => a0 == a1,
            (Flat::Loop { id: id0 }, Flat::Loop { id: id1 }) => id0 == id1,
            (Flat::LoopOut { d: d0, k: k0 }, Flat::LoopOut { d: d1, k: k1 }) => {
                d0 == d1 && k0 == k1
            }
            (
                Flat::GammaPq {
                    d: d0,
                    a: a0,
                    x: x0,
                    upper: upper0,
                    i: i0,
                    j: j0,
                },
                Flat::GammaPq {
                    d: d1,
                    a: a1,
                    x: x1,
                    upper: upper1,
                    i: i1,
                    j: j1,
                },
            ) => d0 == d1 && a0 == a1 && x0 == x1 && upper0 == upper1 && i0 == i1 && j0 == j1,
            (
                Flat::AddK {
                    d: d0,
                    a: a0,
                    c: c0,
                },
                Flat::AddK {
                    d: d1,
                    a: a1,
                    c: c1,
                },
            ) => d0 == d1 && a0 == a1 && c0.to_bits() == c1.to_bits(),
            (
                Flat::SubK {
                    d: d0,
                    a: a0,
                    c: c0,
                },
                Flat::SubK {
                    d: d1,
                    a: a1,
                    c: c1,
                },
            ) => d0 == d1 && a0 == a1 && c0.to_bits() == c1.to_bits(),
            (
                Flat::RSubK {
                    d: d0,
                    a: a0,
                    c: c0,
                },
                Flat::RSubK {
                    d: d1,
                    a: a1,
                    c: c1,
                },
            ) => d0 == d1 && a0 == a1 && c0.to_bits() == c1.to_bits(),
            (
                Flat::MulK {
                    d: d0,
                    a: a0,
                    c: c0,
                },
                Flat::MulK {
                    d: d1,
                    a: a1,
                    c: c1,
                },
            ) => d0 == d1 && a0 == a1 && c0.to_bits() == c1.to_bits(),
            (
                Flat::DivK {
                    d: d0,
                    a: a0,
                    c: c0,
                },
                Flat::DivK {
                    d: d1,
                    a: a1,
                    c: c1,
                },
            ) => d0 == d1 && a0 == a1 && c0.to_bits() == c1.to_bits(),
            (
                Flat::Pow {
                    d: d0,
                    a: a0,
                    e: e0,
                },
                Flat::Pow {
                    d: d1,
                    a: a1,
                    e: e1,
                },
            ) => d0 == d1 && a0 == a1 && e0 == e1,
            (
                Flat::Unary {
                    d: d0,
                    op: op0,
                    a: a0,
                },
                Flat::Unary {
                    d: d1,
                    op: op1,
                    a: a1,
                },
            ) => d0 == d1 && op0 == op1 && a0 == a1,
            (
                Flat::Deriv {
                    state: state0,
                    a: a0,
                },
                Flat::Deriv {
                    state: state1,
                    a: a1,
                },
            ) => state0 == state1 && a0 == a1,
            (
                Flat::Cmp {
                    d: d0,
                    cc: cc0,
                    a: a0,
                    b: b0,
                },
                Flat::Cmp {
                    d: d1,
                    cc: cc1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && cc0 == cc1 && a0 == a1 && b0 == b1,
            (
                Flat::And {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::And {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (
                Flat::Or {
                    d: d0,
                    a: a0,
                    b: b0,
                },
                Flat::Or {
                    d: d1,
                    a: a1,
                    b: b1,
                },
            ) => d0 == d1 && a0 == a1 && b0 == b1,
            (Flat::Not { d: d0, a: a0 }, Flat::Not { d: d1, a: a1 }) => d0 == d1 && a0 == a1,
            (Flat::Present { d: d0, a: a0 }, Flat::Present { d: d1, a: a1 }) => {
                d0 == d1 && a0 == a1
            }
            (Flat::Branch { c: c0, to: to0 }, Flat::Branch { c: c1, to: to1 }) => {
                c0 == c1 && to0 == to1
            }
            (Flat::Jump { to: to0 }, Flat::Jump { to: to1 }) => to0 == to1,
            (Flat::Move { d: d0, a: a0 }, Flat::Move { d: d1, a: a1 }) => d0 == d1 && a0 == a1,
            _ => false,
        }
    }
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
        Some(Flat::LoadSeed { d: 0, k: k as u32 })
    }
}

/// Why [`RhsRegProgram::compile`] kept the interpreter (#2681): the first construct, in
/// program order, outside the translatable shape, and the statement it is in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RhsRefusal {
    pub(crate) construct: RhsConstruct,
    pub(crate) site: RhsSite,
}

/// The statement an [`RhsRefusal`] was found in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RhsSite {
    /// `d/dt(state)`, by state index.
    Deriv(usize),
    /// An assignment to an `[odes]` intermediate, by `vars` slot.
    Assign(usize),
    /// An `if (…) { … }` statement.
    IfBlock,
}

/// The construct behind an [`RhsRefusal`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RhsConstruct {
    /// An inline `if (…) a else b` whose bytecode is not in the shape `compile_expr_into`
    /// gives one (an `if` the parser emits compiles, #2680): a stray jump, arms that leave
    /// different stacks.
    InlineIf,
    /// An `if (…) { … }` statement.
    IfBlock,
    /// A function whose `f64` meaning the generic primitives do not reproduce, by its
    /// `[odes]` spelling.
    Function(&'static str),
    /// A read of something the RHS is not handed (θ, η, a covariate, …), described.
    Read(&'static str),
    /// A read of an intermediate, by `vars` slot, before the program assigns it.
    ReadBeforeAssign(usize),
    /// Anything else the translation declines: more registers than its 16-bit indices
    /// address, an unbalanced expression, an assignment to a binding input.
    Shape,
}

impl RhsRefusal {
    /// The refusal in the words a user reads, `slot_name` naming a `vars` slot (a state's
    /// slot is its index).
    pub(crate) fn named(&self, slot_name: impl Fn(usize) -> String) -> RhsInterpreted {
        let construct = match self.construct {
            RhsConstruct::InlineIf => "an inline `if`".to_string(),
            RhsConstruct::IfBlock => "an `if (…) { … }` statement".to_string(),
            RhsConstruct::Function(f) => format!("`{f}`"),
            RhsConstruct::Read(what) => format!("a read of {what}"),
            RhsConstruct::ReadBeforeAssign(slot) => {
                format!("a read of `{}` before it is assigned", slot_name(slot))
            }
            RhsConstruct::Shape => "a statement the RHS compiler does not translate".to_string(),
        };
        let site = match self.site {
            RhsSite::Deriv(state) => format!("`d/dt({})`", slot_name(state)),
            RhsSite::Assign(slot) => format!("the assignment to `{}`", slot_name(slot)),
            RhsSite::IfBlock => "the [odes] block".to_string(),
        };
        RhsInterpreted { construct, site }
    }
}

/// An [`RhsRefusal`] with its names resolved, as the RHS-route note reports it (#2681).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RhsInterpreted {
    /// The construct as the note names it: "an inline `if`", "`sqrt`", ….
    pub(crate) construct: String,
    /// The statement as the note names it: "`d/dt(sm)`", "the assignment to `E`", ….
    pub(crate) site: String,
}

/// Which engine runs an `[odes]` RHS ([`super::OdeRhsProgram::rhs_route`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RhsRoute<'a> {
    /// The `f64` native function.
    Native,
    /// The register code, because no `f64` native function could be built.
    RegisterCode(NativeCause),
    /// The bytecode interpreter, because the program did not compile.
    Interpreter(&'a RhsInterpreted),
}

/// Why a compiled RHS has no `f64` native function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NativeCause {
    /// `FERX_NO_RHS_JIT=1`.
    Disabled,
    /// Cranelift has no backend for the host.
    Host,
    /// The host is a target, but this program was not built (past the native compiler's
    /// size limit, or a code-generator failure).
    Declined,
}

impl NativeCause {
    /// The cause, given whether native code is switched on and whether Cranelift targets
    /// the host: the switch first, since with it off nothing asks the host.
    pub(crate) fn of(enabled: bool, host_supported: bool) -> Self {
        if !enabled {
            NativeCause::Disabled
        } else if !host_supported {
            NativeCause::Host
        } else {
            NativeCause::Declined
        }
    }
}

/// The stable codes opening the two RHS-route notes.
pub(crate) const RHS_INTERPRETED_CODE: &str = "W_ODE_RHS_INTERPRETED";
pub(crate) const RHS_NOT_NATIVE_CODE: &str = "W_ODE_RHS_NOT_NATIVE";

/// The note for an RHS that does not run as native code (#2681), or `None` for one that
/// does: `(code, message, details)`. One text for `fit()`'s warnings and `ferx check`.
pub(crate) fn rhs_route_note(
    route: RhsRoute<'_>,
) -> Option<(&'static str, String, serde_json::Value)> {
    match route {
        RhsRoute::Native => None,
        RhsRoute::Interpreter(r) => {
            let msg = format!(
                "{RHS_INTERPRETED_CODE}: the [odes] right-hand side runs on the bytecode \
                 interpreter, not compiled code, because of {} in {}. The \
                 constructs that keep the interpreter are listed in the ODE models page \
                 of the documentation, under \"Native right-hand side\".",
                r.construct, r.site,
            );
            let details = serde_json::json!({
                "route": "interpreter",
                "construct": r.construct,
                "site": r.site,
            });
            Some((RHS_INTERPRETED_CODE, msg, details))
        }
        RhsRoute::RegisterCode(cause) => {
            let (why, token) = match cause {
                NativeCause::Disabled => (
                    "the environment variable `FERX_NO_RHS_JIT=1` switches native code off",
                    "disabled",
                ),
                NativeCause::Host => ("Cranelift cannot generate code for this host", "host"),
                NativeCause::Declined => (
                    "the native compiler did not build this program (it builds programs up \
                     to a size limit)",
                    "declined",
                ),
            };
            let msg = format!(
                "{RHS_NOT_NATIVE_CODE}: the [odes] right-hand side runs as register code \
                 rather than native code, because {why}. The results are identical; only \
                 the speed differs."
            );
            let details = serde_json::json!({ "route": "register_code", "cause": token });
            Some((RHS_NOT_NATIVE_CODE, msg, details))
        }
    }
}

/// Which construct an `Op` outside the translatable shape is, or `None` for an op
/// [`Builder::expr`] translates.
fn refused_construct(op: Op) -> Option<RhsConstruct> {
    Some(match op {
        Op::Sqrt => RhsConstruct::Function("sqrt"),
        Op::InvLogit => RhsConstruct::Function("inv_logit"),
        Op::Logit => RhsConstruct::Function("logit"),
        Op::Tanh => RhsConstruct::Function("tanh"),
        Op::Softplus => RhsConstruct::Function("softplus"),
        Op::Sin => RhsConstruct::Function("sin"),
        Op::Cos => RhsConstruct::Function("cos"),
        Op::Pnorm => RhsConstruct::Function("pnorm"),
        Op::Mod => RhsConstruct::Function("mod"),
        Op::PushTheta(_) => RhsConstruct::Read("a θ"),
        Op::PushThetaGather(_) => RhsConstruct::Read("a θ level"),
        Op::PushEta(_) => RhsConstruct::Read("an η"),
        Op::PushCov(_) => RhsConstruct::Read("a covariate"),
        Op::PushMixNum => RhsConstruct::Read("the mixture class `MIXNUM`"),
        Op::PushNnOutput(_, _) => RhsConstruct::Read("a neural-network output"),
        Op::PushTime => RhsConstruct::Read("the `TIME` built-in"),
        Op::PushVar(_) => RhsConstruct::Read("a variable outside the RHS's slots"),
        // A `Jump` outside an `if`'s shape (`Builder::conditional` consumes the one it emits).
        Op::Jump(_) => RhsConstruct::InlineIf,
        Op::CmpLt
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
        | Op::PushConst(_)
        | Op::Add
        | Op::Sub
        | Op::RSub
        | Op::Mul
        | Op::Div
        | Op::Pow
        | Op::Exp
        | Op::Ln
        | Op::Abs
        | Op::Floor
        | Op::Ceil
        | Op::Round => return None,
    })
}

/// The parameter-only values of one integration ([`NativeF64::run`]), keyed on the exact
/// bits of the parameters (and `MACHEPS` seed) they were computed from.
#[derive(Debug, Default)]
pub(crate) struct HoistCache {
    key: Vec<u64>,
    vals: Vec<f64>,
}

/// A program's `f64` native code as the hoisted pair, resolved once
/// ([`RhsRegProgram::native_f64`]): the parameter-only values are computed when the
/// parameters change and read from a [`HoistCache`] on every other call.
#[derive(Clone, Copy)]
pub(crate) struct NativeF64<'a>(jit::Resolved<'a, (jit::PrepFn, jit::MainFn)>);

impl NativeF64<'_> {
    /// [`RhsRegProgram::try_native`] at `f64`, bit for bit: `true` when it ran (and wrote
    /// `du`), `false` with nothing written when a slice is shorter than the program reads.
    #[inline(always)]
    pub(crate) fn run(
        &self,
        u: &[f64],
        params: &[f64],
        seeds: &[f64; 4],
        du: &mut [f64],
        cache: &mut HoistCache,
    ) -> bool {
        let pair = &self.0;
        if !pair.fits(u.len(), params.len(), du.len()) {
            return false;
        }
        // `fits` covers every slot the program loads.
        let slots = pair.param_slots();
        let fresh = cache.key.len() == slots.len() + 1
            && slots
                .iter()
                .zip(&cache.key)
                .all(|(&s, &k)| params[s as usize].to_bits() == k)
            && cache.key[slots.len()] == seeds[3].to_bits();
        if !fresh {
            cache.key.clear();
            cache
                .key
                .extend(slots.iter().map(|&s| params[s as usize].to_bits()));
            cache.key.push(seeds[3].to_bits());
            cache.vals.clear();
            cache.vals.resize(pair.cache_len(), 0.0);
            // SAFETY: `fits` checked `params`; `cache` holds `cache_len` values.
            unsafe { pair.prep(params.as_ptr(), seeds.as_ptr(), cache.vals.as_mut_ptr()) };
        }
        // SAFETY: `fits` checked the slices, and the key says `cache` was filled from
        // these parameters.
        unsafe {
            pair.main(
                u.as_ptr(),
                params.as_ptr(),
                seeds.as_ptr(),
                du.as_mut_ptr(),
                cache.vals.as_ptr(),
            )
        };
        true
    }
}

/// The `f64` RHS with its forward tangents along `k` directions, one native call
/// ([`RhsRegProgram::native_tangent`]): every input and output element is `1 + k` lanes, the
/// value then its `k` tangents.
#[derive(Clone, Copy)]
pub(crate) struct NativeTan<'a>(jit::Resolved<'a, jit::RhsFn>, usize);

impl NativeTan<'_> {
    /// Lanes of one element.
    pub(crate) fn lanes(&self) -> usize {
        1 + self.1
    }

    /// `du` (value and tangents) from `u`, `params` and the four `seeds`, each element
    /// [`Self::lanes`] long; `false`, nothing written, when a slice is shorter than the
    /// program reads.
    pub(crate) fn run(&self, u: &[f64], params: &[f64], seeds: &[f64], du: &mut [f64]) -> bool {
        let l = self.lanes();
        if seeds.len() < 4 * l || !self.0.fits(u.len() / l, params.len() / l, du.len() / l) {
            return false;
        }
        // SAFETY: `fits` covers every element the program loads or stores, at `l` lanes each,
        // and `seeds` holds four elements; `du` is a separate slice.
        unsafe {
            self.0
                .call(u.as_ptr(), params.as_ptr(), seeds.as_ptr(), du.as_mut_ptr())
        };
        true
    }
}

/// A compiled straight-line RHS; see the module docs.
#[derive(Clone, Debug)]
pub(super) struct RhsRegProgram {
    code: Vec<Flat>,
    n_regs: usize,
    /// The same program as native code, built per lane count on first use (see `rhs_jit`).
    jit: Option<jit::JitCode>,
}

impl RhsRegProgram {
    /// [`Self::compile`], except under the test-only `RHS_REGCODE_DISABLED`, which declines
    /// as [`RhsConstruct::Shape`] at no site in particular.
    pub(super) fn compile_unless_disabled(
        stmts: &[Statement],
        layout: &RhsLayout<'_>,
    ) -> Result<Self, RhsRefusal> {
        #[cfg(test)]
        if super::RHS_REGCODE_DISABLED.with(|c| c.get()) {
            return Err(RhsRefusal {
                construct: RhsConstruct::Shape,
                site: RhsSite::IfBlock,
            });
        }
        Self::compile(stmts, layout)
    }

    /// Compile `stmts` for a `vars` buffer of length `n_vars`, or say why not when the
    /// program is outside the translatable shape (see the module docs) — the caller keeps
    /// the interpreter for it.
    pub(super) fn compile(stmts: &[Statement], layout: &RhsLayout<'_>) -> Result<Self, RhsRefusal> {
        let n_vars = layout.n_vars;
        let mut b = Builder {
            ins: Vec::new(),
            n_tmp: 0,
            n_vars,
        };
        // The statement each instruction came from, for `lower`'s refusals.
        let mut sites: Vec<RhsSite> = Vec::new();
        let shape = |site| RhsRefusal {
            construct: RhsConstruct::Shape,
            site,
        };
        for s in stmts {
            let site = match s {
                Statement::AssignBc(slot, bc) => {
                    let site = RhsSite::Assign(*slot);
                    let v = b
                        .expr(bc)
                        .map_err(|construct| RhsRefusal { construct, site })?;
                    // The interpreter drops a write past the end of `vars`.
                    if *slot < n_vars {
                        b.ins.push(Ins::Assign {
                            slot: u32::try_from(*slot).map_err(|_| shape(site))?,
                            v,
                        });
                    }
                    site
                }
                Statement::DiffEqBc(state, bc) => {
                    let site = RhsSite::Deriv(*state);
                    let v = b
                        .expr(bc)
                        .map_err(|construct| RhsRefusal { construct, site })?;
                    b.ins.push(Ins::Deriv {
                        state: u32::try_from(*state).map_err(|_| shape(site))?,
                        v,
                    });
                    site
                }
                Statement::If { .. } => {
                    return Err(RhsRefusal {
                        construct: RhsConstruct::IfBlock,
                        site: RhsSite::IfBlock,
                    })
                }
                // The unresolved forms never reach a compiled program.
                _ => return Err(shape(RhsSite::IfBlock)),
            };
            sites.resize(b.ins.len(), site);
        }
        lower(&b.ins, &sites, b.n_tmp, layout)
    }

    /// [`RhsRoute::Native`] when the `f64` native function exists (building it on first
    /// use), else the register code and why.
    pub(super) fn route(&self) -> RhsRoute<'static> {
        if self.has_native_f64() {
            RhsRoute::Native
        } else {
            RhsRoute::RegisterCode(jit::native_cause())
        }
    }

    /// Whether a native function exists (or can be built) for `T`'s lanes.
    #[cfg(test)]
    pub(super) fn has_native<T: VmNum>(&self) -> bool {
        T::LANES.is_some_and(|l| self.jit.as_ref().is_some_and(|j| j.has(l)))
    }

    /// Registers the program uses; [`Self::run`]'s `regs` must hold at least this many.
    pub(super) fn n_regs(&self) -> usize {
        self.n_regs
    }

    /// Run the program on the RHS's own inputs: the `du` the interpreter writes after the
    /// binding fills `vars` from `u`, `params` and `seeds`, bit for bit. `regs` is scratch of
    /// at least [`Self::n_regs`] values.
    #[inline]
    pub(super) fn run<T: VmNum>(
        &self,
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        regs: &mut [T],
    ) {
        if self.run_native(u, params, seeds, du) {
            return;
        }
        self.exec(&self.code, u, params, seeds, du, &mut regs[..self.n_regs]);
    }

    /// Whether an `f64` native function exists (built on first use).
    #[inline]
    pub(super) fn has_native_f64(&self) -> bool {
        #[cfg(test)]
        if RHS_JIT_DISABLED.with(|c| c.get()) {
            return false;
        }
        self.jit.as_ref().is_some_and(|j| j.has(ir::Layout::F64))
    }

    /// The `f64` native function, resolved once for a caller that runs it many times
    /// (`DirectRhsF64`). `None` where [`Self::has_native_f64`] is `false`.
    pub(super) fn native_f64(&self) -> Option<NativeF64<'_>> {
        #[cfg(test)]
        if RHS_JIT_DISABLED.with(|c| c.get()) {
            return None;
        }
        self.jit
            .as_ref()?
            .resolve_hoisted(ir::Layout::F64)
            .map(NativeF64)
    }

    /// The `f64` RHS over `k` forward tangent directions as one native function (built on
    /// first use); `None` where there is no native code for it.
    pub(super) fn native_tangent(&self, k: usize) -> Option<NativeTan<'_>> {
        #[cfg(test)]
        if RHS_JIT_DISABLED.with(|c| c.get()) {
            return None;
        }
        let layout = ir::Layout {
            tan: u8::try_from(k).ok()?,
            ..ir::Layout::F64
        };
        self.jit.as_ref()?.resolve(layout).map(|f| NativeTan(f, k))
    }

    /// [`Self::run`] when `T` has a native function, without the register file: `true` when
    /// it ran (and wrote `du`), `false` with nothing written otherwise.
    #[inline]
    pub(super) fn try_native<T: VmNum>(
        &self,
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
    ) -> bool {
        self.run_native(u, params, seeds, du)
    }

    /// [`Self::run`] through the native function when `T` has one (`VmArith::LANES`):
    /// `f64` and the duals with lanes, the contiguous ones in place and the rest through
    /// their lanes when no input is on a tape. `false`, with nothing written, otherwise.
    #[inline(always)]
    fn run_native<T: VmNum>(&self, u: &[T], params: &[T], seeds: &[T; 4], du: &mut [T]) -> bool {
        let Some(layout) = T::LANES else {
            return false;
        };
        let Some(jit) = self.jit.as_ref() else {
            return false;
        };
        #[cfg(test)]
        if RHS_JIT_DISABLED.with(|c| c.get()) {
            return false;
        }
        let Some(f) = jit.resolve(layout) else {
            return false;
        };
        if !f.fits(u.len(), params.len(), du.len()) {
            return false;
        }
        if T::CONTIGUOUS {
            debug_assert_eq!(std::mem::size_of::<T>(), layout.lanes() * 8);
            // SAFETY: a contiguous `T` is `lanes` `f64`s (`#[repr(C)]`, value first); `fits`
            // checked the slices, and `du` is a distinct `&mut`.
            unsafe {
                f.call(
                    u.as_ptr().cast(),
                    params.as_ptr().cast(),
                    seeds.as_ptr().cast(),
                    du.as_mut_ptr().cast(),
                )
            };
            return true;
        }
        let (eu, ep, edu) = f.extent();
        if !(u[..eu].iter().all(T::native_ok)
            && params[..ep].iter().all(T::native_ok)
            && seeds.iter().all(T::native_ok)
            && du[..edu].iter().all(T::native_ok))
        {
            return false;
        }
        let w = layout.lanes();
        LANE_SCRATCH.with(|cell| {
            let mut b = cell.borrow_mut();
            let need = (eu + ep + 4 + edu) * w;
            if b.len() < need {
                b.resize(need, 0.0);
            }
            let (bu, rest) = b.split_at_mut(eu * w);
            let (bp, rest) = rest.split_at_mut(ep * w);
            let (bs, rest) = rest.split_at_mut(4 * w);
            let bd = &mut rest[..edu * w];
            for (x, o) in u[..eu].iter().zip(bu.chunks_exact_mut(w)) {
                x.put_lanes(o);
            }
            for (x, o) in params[..ep].iter().zip(bp.chunks_exact_mut(w)) {
                x.put_lanes(o);
            }
            for (x, o) in seeds.iter().zip(bs.chunks_exact_mut(w)) {
                x.put_lanes(o);
            }
            for (x, o) in du[..edu].iter().zip(bd.chunks_exact_mut(w)) {
                x.put_lanes(o);
            }
            // SAFETY: the buffers hold the extent's elements of `w` lanes each.
            unsafe { f.call(bu.as_ptr(), bp.as_ptr(), bs.as_ptr(), bd.as_mut_ptr()) };
            for (x, l) in du[..edu].iter_mut().zip(bd.chunks_exact(w)) {
                *x = T::from_lanes(l);
            }
        });
        true
    }

    #[inline(always)]
    fn exec<T: VmNum>(
        &self,
        code: &[Flat],
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        r: &mut [T],
    ) {
        self.exec_range(code, 0, code.len(), u, params, seeds, du, r)
    }

    /// [`Self::exec`] over the instructions `start..end`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn exec_range<T: VmNum>(
        &self,
        code: &[Flat],
        start: usize,
        end: usize,
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        r: &mut [T],
    ) {
        let k = T::from_f64;
        macro_rules! bin {
            ($d:expr, $a:expr, $f:expr) => {{
                let mut x = r[$a as usize];
                $f(&mut x);
                r[$d as usize] = x;
            }};
        }
        let b01 = |c: bool| k(if c { 1.0 } else { 0.0 });
        // A recording type (#2786) records a condition as data — the comparisons and their
        // combinations as selections — and both arms of an `if` on it ([`Self::both_arms`]):
        // an ODE step evaluates the right-hand side at states where either arm holds.
        let rec = T::RECORDS_BRANCHES;
        let (zero, one) = (k(0.0), k(1.0));
        use crate::sens::num::BranchCmp::{Eq, Ne};
        let mut pc = start;
        while pc < end {
            let f = &code[pc];
            pc += 1;
            match *f {
                Flat::LoadState { d, i } => {
                    r[d as usize] = u.get(i as usize).copied().unwrap_or_else(|| k(0.0))
                }
                Flat::LoadParam { d, slot } => {
                    r[d as usize] = params.get(slot as usize).copied().unwrap_or_else(|| k(0.0))
                }
                Flat::LoadSeed { d, k: i } => r[d as usize] = seeds[i as usize],
                Flat::LoadK { d, c } => r[d as usize] = k(c),
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
                // The interpreter's own guarded division (`apply_binary_in`).
                Flat::Div { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| apply_binary_in(x, Op::Div, &y))
                }
                Flat::DivRaw { d, a, b } => {
                    let y = r[b as usize];
                    bin!(d, a, |x: &mut T| x.div_in(&y))
                }
                Flat::AddK { d, a, c } => bin!(d, a, |x: &mut T| x.add_scalar_in(c)),
                Flat::SubK { d, a, c } => bin!(d, a, |x: &mut T| x.sub_scalar_in(c)),
                Flat::RSubK { d, a, c } => bin!(d, a, |x: &mut T| x.rsub_scalar_in(c)),
                Flat::MulK { d, a, c } => bin!(d, a, |x: &mut T| x.mul_scalar_in(c)),
                Flat::DivK { d, a, c } => bin!(d, a, |x: &mut T| apply_scalar_in(x, Op::Div, c)),
                Flat::Pow { d, a, e } => r[d as usize] = r[a as usize].pow(r[e as usize]),
                Flat::GammaPq {
                    d,
                    a,
                    x,
                    upper,
                    i,
                    j,
                } => {
                    debug_assert!(i == 0 && j == 0, "a partial is never interpreted");
                    r[d as usize] = r[a as usize].gamma_pq(r[x as usize], upper)
                }
                Flat::Unary { d, op, a } => r[d as usize] = apply_unary(op.op(), r[a as usize]),
                Flat::Deriv { state, a } => {
                    if let Some(x) = du.get_mut(state as usize) {
                        *x = r[a as usize];
                    }
                }
                Flat::Cmp { d, cc, a, b } if rec => {
                    r[d as usize] =
                        T::select_if(cc.branch(), r[a as usize], r[b as usize], one, zero)
                }
                Flat::And { d, a, b } if rec => {
                    let rb = T::select_if(Ne, r[b as usize], zero, one, zero);
                    r[d as usize] = T::select_if(Ne, r[a as usize], zero, rb, zero)
                }
                Flat::Or { d, a, b } if rec => {
                    let rb = T::select_if(Ne, r[b as usize], zero, one, zero);
                    r[d as usize] = T::select_if(Ne, r[a as usize], zero, one, rb)
                }
                Flat::Not { d, a } if rec => {
                    r[d as usize] = T::select_if(Eq, r[a as usize], zero, one, zero)
                }
                // `x == x` is false exactly for a `NaN`.
                Flat::Present { d, a } if rec => {
                    r[d as usize] = T::select_if(Eq, r[a as usize], r[a as usize], one, zero)
                }
                Flat::Cmp { d, cc, a, b } => {
                    r[d as usize] = b01(cc.holds(r[a as usize].val(), r[b as usize].val()))
                }
                Flat::And { d, a, b } => {
                    r[d as usize] = b01(r[a as usize].val() != 0.0 && r[b as usize].val() != 0.0)
                }
                Flat::Or { d, a, b } => {
                    r[d as usize] = b01(r[a as usize].val() != 0.0 || r[b as usize].val() != 0.0)
                }
                Flat::Not { d, a } => r[d as usize] = b01(r[a as usize].val() == 0.0),
                Flat::Present { d, a } => r[d as usize] = b01(!r[a as usize].val().is_nan()),
                Flat::HostArg { .. } | Flat::Host { .. } | Flat::HostOut { .. } => {
                    unreachable!("only a kernel recording makes a host call")
                }
                Flat::LoopArg { .. }
                | Flat::Loop { .. }
                | Flat::LoopOut { .. }
                | Flat::Select { .. }
                | Flat::Gate { .. } => {
                    unreachable!("only a kernel recording makes a loop call, a select or a gate")
                }
                Flat::Branch { c, to } if rec && !r[c as usize].is_constant() => {
                    pc = self.both_arms(code, pc, to as usize, c, u, params, seeds, du, r);
                }
                Flat::Branch { c, to } => {
                    if r[c as usize].val() == 0.0 {
                        pc = to as usize;
                    }
                }
                Flat::Jump { to } => pc = to as usize,
                Flat::Move { d, a } => r[d as usize] = r[a as usize],
            }
        }
    }

    /// Both arms of an `if` on the data condition `r[c]` (a recording type, #2786), the
    /// then-arm at `then_start..else_start` (its closing `Jump` excluded), the else-arm up to
    /// that `Jump`'s target; returns the instruction after the `if`. Every register and
    /// derivative an arm writes becomes the selection of the two on `r[c]`.
    ///
    /// Each arm reads its inputs through `select(arm live, x, value_only(x))`: in the arm not
    /// taken the inputs are constants, so a non-finite local derivative there (a negative
    /// base's power, a negative `sqrt`) meets a zero adjoint inside the arm and never reaches
    /// `x` — the selection hands the arm's adjoints to the constant.
    #[allow(clippy::too_many_arguments)]
    fn both_arms<T: VmNum>(
        &self,
        code: &[Flat],
        then_start: usize,
        else_start: usize,
        c: u32,
        u: &[T],
        params: &[T],
        seeds: &[T; 4],
        du: &mut [T],
        r: &mut [T],
    ) -> usize {
        use crate::sens::num::BranchCmp::{Eq, Ne};
        let zero = T::from_f64(0.0);
        let cond = r[c as usize];
        let (then_end, end) = match code.get(else_start.wrapping_sub(1)) {
            Some(&Flat::Jump { to }) if (to as usize) >= else_start => {
                (else_start - 1, to as usize)
            }
            _ => (else_start, else_start),
        };
        // The registers and derivatives an arm reads and writes.
        let scan = |range: std::ops::Range<usize>| {
            let (mut reads, mut writes, mut derivs) = (Vec::new(), Vec::new(), Vec::new());
            for f in &code[range] {
                let mut g = *f;
                if let Flat::Deriv { state, .. } = g {
                    derivs.push(state as usize);
                }
                let (d, us) = regs_of(&mut g);
                reads.extend(us.into_iter().flatten().map(|x| *x as usize));
                if let Some(d) = d {
                    writes.push(*d as usize);
                }
            }
            (reads, writes, derivs)
        };
        let snap_r = r.to_vec();
        let snap_du = du.to_vec();
        let mut arm = |range: std::ops::Range<usize>, live: crate::sens::num::BranchCmp| {
            r.copy_from_slice(&snap_r);
            du.copy_from_slice(&snap_du);
            let (reads, _, _) = scan(range.clone());
            for i in reads {
                let x = snap_r[i];
                r[i] = T::select_if(live, cond, zero, x, x.value_only());
            }
            self.exec_range(code, range.start, range.end, u, params, seeds, du, r);
            (r.to_vec(), du.to_vec())
        };
        let (r_then, du_then) = arm(then_start..then_end, Ne);
        let (r_else, du_else) = arm(else_start..end, Eq);
        r.copy_from_slice(&snap_r);
        du.copy_from_slice(&snap_du);
        let (_, w_then, d_then) = scan(then_start..then_end);
        let (_, w_else, d_else) = scan(else_start..end);
        for i in w_then.iter().chain(&w_else).copied() {
            let t = if w_then.contains(&i) {
                r_then[i]
            } else {
                snap_r[i]
            };
            let e = if w_else.contains(&i) {
                r_else[i]
            } else {
                snap_r[i]
            };
            r[i] = T::select_if(Ne, cond, zero, t, e);
        }
        for s in d_then.iter().chain(&d_else).copied() {
            if s < du.len() {
                let t = if d_then.contains(&s) {
                    du_then[s]
                } else {
                    snap_du[s]
                };
                let e = if d_else.contains(&s) {
                    du_else[s]
                } else {
                    snap_du[s]
                };
                du[s] = T::select_if(Ne, cond, zero, t, e);
            }
        }
        end
    }
}

/// Lower the builder's operand-form IR to [`Flat`] code over one register file: temps keep
/// their numbers, a slot read before any assignment to it is loaded once in a prologue, a
/// slot read after one reads the assigned value's register (the interpreter's `PushVar`
/// sees the assignment, and so does this), and each unfused literal gets a register of its
/// own; [`reuse_registers`] then packs them by lifetime. An `if`'s markers become a
/// [`Flat::Branch`] and a [`Flat::Jump`] whose targets are patched once the arms are placed.
///
/// `sites[i]` is the statement `ins[i]` came from, named in a refusal.
fn lower(
    ins: &[Ins],
    sites: &[RhsSite],
    n_tmp: usize,
    layout: &RhsLayout<'_>,
) -> Result<RhsRegProgram, RhsRefusal> {
    let mut next = n_tmp;
    let mut alloc = || -> Option<u32> {
        let r = u32::try_from(next).ok()?;
        next += 1;
        Some(r)
    };
    let mut prologue: Vec<Flat> = Vec::new();
    let mut body: Vec<Flat> = Vec::new();
    // Current register of a slot: its prologue load, or the value last assigned to it.
    let mut slot_reg: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    // The refusals below are a construct; the loop adds the statement's site.
    let mut operand = |s: Src,
                       slot_reg: &mut std::collections::HashMap<u32, u32>,
                       prologue: &mut Vec<Flat>,
                       body: &mut Vec<Flat>|
     -> Result<u32, RhsConstruct> {
        Ok(match s {
            Src::Tmp(t) => u32::try_from(t).map_err(|_| RhsConstruct::Shape)?,
            Src::Var(slot) => match slot_reg.get(&slot) {
                Some(&r) => r,
                // A slot read before any assignment: one of the binding's inputs, loaded
                // once. A local read before the program assigns it has no source here
                // (the interpreter would read what the buffer held), so it is declined.
                None => {
                    let d = alloc().ok_or(RhsConstruct::Shape)?;
                    let load = match layout.source(slot as usize) {
                        Some(Flat::LoadState { i, .. }) => Flat::LoadState { d, i },
                        Some(Flat::LoadParam { slot, .. }) => Flat::LoadParam { d, slot },
                        Some(Flat::LoadSeed { k, .. }) => Flat::LoadSeed { d, k },
                        Some(_) => return Err(RhsConstruct::Shape),
                        None => return Err(RhsConstruct::ReadBeforeAssign(slot as usize)),
                    };
                    prologue.push(load);
                    slot_reg.insert(slot, d);
                    d
                }
            },
            Src::Konst(c) => {
                let d = alloc().ok_or(RhsConstruct::Shape)?;
                body.push(Flat::LoadK { d, c });
                d
            }
        })
    };
    // Per open `if`: its `Branch`, then its `Jump` once the else arm starts.
    let mut open: Vec<(usize, Option<usize>)> = Vec::new();
    for (n, i) in ins.iter().enumerate() {
        let site = sites[n];
        let at = |construct| RhsRefusal { construct, site };
        let reg = |r: u32| u32::try_from(r).map_err(|_| at(RhsConstruct::Shape));
        let to = |pos: usize| u32::try_from(pos).map_err(|_| at(RhsConstruct::Shape));
        // An `if`'s markers out of order: the builder emitted something it never does.
        let unbalanced = || at(RhsConstruct::InlineIf);
        let f = match *i {
            Ins::Bin { dst, op, a, b } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let b = operand(b, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let d = reg(dst)?;
                match op {
                    Arith::Add => Flat::Add { d, a, b },
                    Arith::Sub => Flat::Sub { d, a, b },
                    Arith::RSub => Flat::RSub { d, a, b },
                    Arith::Mul => Flat::Mul { d, a, b },
                    Arith::Div => Flat::Div { d, a, b },
                }
            }
            Ins::Scalar { dst, op, a, c } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let d = reg(dst)?;
                match op {
                    Arith::Add => Flat::AddK { d, a, c },
                    Arith::Sub => Flat::SubK { d, a, c },
                    Arith::RSub => Flat::RSubK { d, a, c },
                    Arith::Mul => Flat::MulK { d, a, c },
                    Arith::Div => Flat::DivK { d, a, c },
                }
            }
            Ins::Pow { dst, a, e } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let e = operand(e, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                Flat::Pow { d: reg(dst)?, a, e }
            }
            Ins::Unary { dst, op, a } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                Flat::Unary {
                    d: reg(dst)?,
                    op,
                    a,
                }
            }
            Ins::Assign { slot, v } => {
                // A local lives in its register: later reads of the slot see this value, as
                // the interpreter's do. Assigning a binding input is a parse error, and a
                // program that did would decline here rather than shadow it.
                if layout.source(slot as usize).is_some() {
                    return Err(at(RhsConstruct::Shape));
                }
                let a = operand(v, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                slot_reg.insert(slot, a);
                continue;
            }
            Ins::Deriv { state, v } => {
                let a = operand(v, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                Flat::Deriv { state, a }
            }
            Ins::Cmp { dst, cc, a, b } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let b = operand(b, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                Flat::Cmp {
                    d: reg(dst)?,
                    cc,
                    a,
                    b,
                }
            }
            Ins::Logic { dst, or, a, b } => {
                let a = operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let b = operand(b, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                let d = reg(dst)?;
                if or {
                    Flat::Or { d, a, b }
                } else {
                    Flat::And { d, a, b }
                }
            }
            Ins::Not { dst, a } => Flat::Not {
                d: reg(dst)?,
                a: operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?,
            },
            Ins::Present { dst, a } => Flat::Present {
                d: reg(dst)?,
                a: operand(a, &mut slot_reg, &mut prologue, &mut body).map_err(at)?,
            },
            Ins::Move { dst, v } => Flat::Move {
                d: reg(dst)?,
                a: operand(v, &mut slot_reg, &mut prologue, &mut body).map_err(at)?,
            },
            Ins::Branch { c } => {
                let c = operand(c, &mut slot_reg, &mut prologue, &mut body).map_err(at)?;
                open.push((body.len(), None));
                Flat::Branch { c, to: u32::MAX }
            }
            Ins::Else => {
                let (branch, jump) = open.last_mut().ok_or_else(unbalanced)?;
                *jump = Some(body.len());
                let target = to(body.len() + 1)?;
                if let Flat::Branch { to, .. } = &mut body[*branch] {
                    *to = target;
                }
                Flat::Jump { to: u32::MAX }
            }
            Ins::EndIf => {
                let (_, jump) = open.pop().ok_or_else(unbalanced)?;
                let target = to(body.len())?;
                if let Flat::Jump { to, .. } = &mut body[jump.ok_or_else(unbalanced)?] {
                    *to = target;
                }
                continue;
            }
        };
        body.push(f);
    }
    // Past the loop, a refusal is the last statement's (an `if` it left open).
    let last = |construct| RhsRefusal {
        construct,
        site: sites.last().copied().unwrap_or(RhsSite::IfBlock),
    };
    if !open.is_empty() {
        return Err(last(RhsConstruct::InlineIf));
    }
    // The jump targets were placed in `body`; the prologue goes in front of it.
    let off = u32::try_from(prologue.len()).map_err(|_| last(RhsConstruct::Shape))?;
    for f in &mut body {
        if let Flat::Branch { to, .. } | Flat::Jump { to } = f {
            *to = to
                .checked_add(off)
                .ok_or_else(|| last(RhsConstruct::Shape))?;
        }
    }
    // The prologue reads every slot at its call-entry value; an assignment renames the slot
    // for the instructions after it, so no prologue load can observe a later write.
    prologue.extend(body);
    let (code, n_regs) = reuse_registers(prologue, next);
    Ok(RhsRegProgram {
        jit: Some(jit::JitCode::new(&code, n_regs)),
        code,
        n_regs,
    })
}

/// The register an instruction writes, and the ones it reads.
fn regs_of(f: &mut Flat) -> (Option<&mut u32>, [Option<&mut u32>; 2]) {
    match f {
        Flat::LoadState { d, .. }
        | Flat::LoadParam { d, .. }
        | Flat::LoadSeed { d, .. }
        | Flat::LoadK { d, .. } => (Some(d), [None, None]),
        Flat::Add { d, a, b }
        | Flat::Sub { d, a, b }
        | Flat::RSub { d, a, b }
        | Flat::Mul { d, a, b }
        | Flat::Div { d, a, b }
        | Flat::DivRaw { d, a, b } => (Some(d), [Some(a), Some(b)]),
        Flat::Pow { d, a, e } => (Some(d), [Some(a), Some(e)]),
        Flat::GammaPq { d, a, x, .. } => (Some(d), [Some(a), Some(x)]),
        Flat::AddK { d, a, .. }
        | Flat::SubK { d, a, .. }
        | Flat::RSubK { d, a, .. }
        | Flat::MulK { d, a, .. }
        | Flat::DivK { d, a, .. }
        | Flat::Unary { d, a, .. } => (Some(d), [Some(a), None]),
        Flat::Cmp { d, a, b, .. } | Flat::And { d, a, b } | Flat::Or { d, a, b } => {
            (Some(d), [Some(a), Some(b)])
        }
        Flat::Not { d, a } | Flat::Present { d, a } | Flat::Move { d, a } => {
            (Some(d), [Some(a), None])
        }
        Flat::Deriv { a, .. }
        | Flat::Branch { c: a, .. }
        | Flat::HostArg { a }
        | Flat::LoopArg { a } => (None, [Some(a), None]),
        Flat::Jump { .. } | Flat::Host { .. } | Flat::Loop { .. } => (None, [None, None]),
        Flat::HostOut { d, .. } | Flat::LoopOut { d, .. } => (Some(d), [None, None]),
        Flat::Select { .. } | Flat::Gate { .. } => {
            unreachable!("never register-allocated: kernel recordings only")
        }
    }
}

/// Map the program's virtual registers (one per value: every temporary, load and literal)
/// onto as few physical registers as their lifetimes allow, so `n_regs` follows the number
/// of values live at once rather than the program's length.
///
/// An `if` keeps this linear: a value read in an arm is read at a later position than any
/// instruction of an arm before it, so it is not free while that arm runs, and no value an arm
/// defines is read past the arm except its result, which is written by a [`Flat::Move`] in
/// each arm. That register is taken at the first `Move` and held until its last read (or the
/// second `Move`, when nothing reads it).
///
/// A register is free again after the instruction that reads it last, and that
/// instruction may write its result into it: every arm reads its operands before it writes
/// (`exec` copies them out first, and the native code is SSA), so the reuse changes where
/// values live, never which value an instruction reads. A value written and never read gets
/// a register for its write and gives it back at once.
fn reuse_registers(mut code: Vec<Flat>, n_virtual: usize) -> (Vec<Flat>, usize) {
    const DEAD: usize = usize::MAX;
    let mut last_use = vec![DEAD; n_virtual];
    // The last write of a value both arms of an `if` write (`DEAD` for every other value).
    let mut last_move = vec![DEAD; n_virtual];
    for (i, f) in code.iter_mut().enumerate() {
        if let Flat::Move { d, .. } = f {
            last_move[*d as usize] = i;
        }
        for r in regs_of(f).1.into_iter().flatten() {
            last_use[*r as usize] = i;
        }
    }
    let mut phys: Vec<Option<u32>> = vec![None; n_virtual];
    let mut free: Vec<u32> = Vec::new();
    let mut n_phys: u32 = 0;
    for (i, f) in code.iter_mut().enumerate() {
        let (def, uses) = regs_of(f);
        for r in uses.into_iter().flatten() {
            let v = *r as usize;
            let p = phys[v].expect("a register is written before it is read");
            *r = p;
            if last_use[v] == i {
                // Read twice by this instruction (`x * x`): returned once.
                if !free.contains(&p) {
                    free.push(p);
                }
            }
        }
        if let Some(d) = def {
            let v = *d as usize;
            // The second arm's `Move` writes the register the first one took.
            let p = phys[v].unwrap_or_else(|| {
                free.pop().unwrap_or_else(|| {
                    n_phys += 1;
                    n_phys - 1
                })
            });
            phys[v] = Some(p);
            *d = p;
            let written_for_good = last_move[v] == DEAD || last_move[v] == i;
            if written_for_good && (last_use[v] == DEAD || last_use[v] < i) {
                free.push(p);
            }
        }
    }
    (code, n_phys as usize)
}

/// A program outside the translatable shape for no construct in particular.
const SHAPE: RhsConstruct = RhsConstruct::Shape;

struct Builder {
    ins: Vec<Ins>,
    n_tmp: usize,
    n_vars: usize,
}

impl Builder {
    fn tmp(&mut self) -> Option<u32> {
        let t = u32::try_from(self.n_tmp).ok()?;
        self.n_tmp += 1;
        Some(t)
    }

    /// One expression, simulating `eval_bytecode_g`'s stack symbolically; the operand
    /// left on the stack, or the construct that puts the bytecode outside the
    /// translatable shape.
    fn expr(&mut self, bc: &Bytecode) -> Result<Src, RhsConstruct> {
        let ops = bc.ops.as_slice();
        let consts = bc.constants.as_slice();
        let mut stack: Vec<Src> = Vec::with_capacity(bc.max_stack);
        let end = self.run(ops, consts, 0, ops.len(), &mut stack)?;
        // A balanced program leaves exactly one operand.
        if end == ops.len() && stack.len() == 1 {
            Ok(stack[0])
        } else {
            Err(SHAPE)
        }
    }

    /// Simulate `ops` from `pc` until it reaches `fence`; the `pc` it stops at, which is one
    /// past `fence` when the last operand pushed fuses with the op at `fence`.
    fn run(
        &mut self,
        ops: &[Op],
        consts: &[f64],
        mut pc: usize,
        fence: usize,
        stack: &mut Vec<Src>,
    ) -> Result<usize, RhsConstruct> {
        while pc < fence {
            pc = self.step(ops, consts, pc, stack)?;
        }
        Ok(pc)
    }

    /// One op (two, fused) of `eval_bytecode_g`, at `pc`; the next `pc`.
    fn step(
        &mut self,
        ops: &[Op],
        consts: &[f64],
        pc: usize,
        stack: &mut Vec<Src>,
    ) -> Result<usize, RhsConstruct> {
        let fused = if stack.is_empty() {
            None
        } else {
            ops.get(pc + 1).copied().and_then(Arith::of)
        };
        match (ops[pc], fused) {
            // Operand fusion, exactly as `eval_bytecode_g` applies it: a constant
            // consumed by the next arithmetic op is applied as a scalar, a variable in
            // place. (`PushCov` and `PushTime` fuse too there, but the RHS reads no
            // covariate, and the parser rewrites every `TIME` to the time slot's
            // `PushVar` before compiling — `rewrite_push_time_to_slot`.)
            (Op::PushConst(i), Some(op)) => {
                let a = stack.pop().ok_or(SHAPE)?;
                let c = *consts.get(i as usize).ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Scalar { dst, op, a, c });
                stack.push(Src::Tmp(dst));
                return Ok(pc + 2);
            }
            (Op::PushVar(i), Some(op)) if (i as usize) < self.n_vars => {
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Bin {
                    dst,
                    op,
                    a,
                    b: Src::Var(i),
                });
                stack.push(Src::Tmp(dst));
                return Ok(pc + 2);
            }
            (Op::PushConst(i), None) => {
                stack.push(Src::Konst(*consts.get(i as usize).ok_or(SHAPE)?))
            }
            // The interpreter reads an out-of-range slot as 0; not worth a variant.
            (Op::PushVar(i), _) if (i as usize) < self.n_vars => stack.push(Src::Var(i)),
            (op @ (Op::Add | Op::Sub | Op::RSub | Op::Mul | Op::Div), _) => {
                let b = stack.pop().ok_or(SHAPE)?;
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Bin {
                    dst,
                    op: Arith::of(op).ok_or(SHAPE)?,
                    a,
                    b,
                });
                stack.push(Src::Tmp(dst));
            }
            (Op::Pow, _) => {
                let e = stack.pop().ok_or(SHAPE)?;
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Pow { dst, a, e });
                stack.push(Src::Tmp(dst));
            }
            (op @ (Op::Exp | Op::Ln | Op::Abs | Op::Floor | Op::Ceil | Op::Round), _) => {
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
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
            (op @ (Op::CmpLt | Op::CmpLe | Op::CmpGt | Op::CmpGe | Op::CmpEq | Op::CmpNe), _) => {
                let b = stack.pop().ok_or(SHAPE)?;
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                let cc = match op {
                    Op::CmpLt => Cc::Lt,
                    Op::CmpLe => Cc::Le,
                    Op::CmpGt => Cc::Gt,
                    Op::CmpGe => Cc::Ge,
                    Op::CmpEq => Cc::Eq,
                    _ => Cc::Ne,
                };
                self.ins.push(Ins::Cmp { dst, cc, a, b });
                stack.push(Src::Tmp(dst));
            }
            (op @ (Op::LogicAnd | Op::LogicOr), _) => {
                let b = stack.pop().ok_or(SHAPE)?;
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                let or = matches!(op, Op::LogicOr);
                self.ins.push(Ins::Logic { dst, or, a, b });
                stack.push(Src::Tmp(dst));
            }
            (Op::LogicNot, _) => {
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Not { dst, a });
                stack.push(Src::Tmp(dst));
            }
            (Op::IsPresent, _) => {
                let a = stack.pop().ok_or(SHAPE)?;
                let dst = self.tmp().ok_or(SHAPE)?;
                self.ins.push(Ins::Present { dst, a });
                stack.push(Src::Tmp(dst));
            }
            (Op::JumpIfFalse(at), _) => {
                return self.conditional(ops, consts, pc, at as usize, stack)
            }
            // Everything else keeps the interpreter: a `Jump` outside the shape
            // `compile_expr_into` gives an `if` (`conditional` consumes the one it emits), the
            // reads the RHS is not handed (θ, η, covariates, gathers, network outputs, the
            // mixture class), an out-of-range variable, and the ops whose `f64` meaning the
            // generic primitives do not reproduce bit for bit (`Sqrt`'s `-0.0` guard,
            // `InvLogit`, `Logit`, `Tanh`, `Softplus`, `Sin`, `Cos`, `Pnorm`, `Mod`). Listed one by one: a new `Op`
            // must be decided here, not swallowed by a wildcard.
            (
                Op::PushVar(_)
                | Op::PushTime
                | Op::PushTheta(_)
                | Op::PushEta(_)
                | Op::PushCov(_)
                | Op::PushMixNum
                | Op::PushNnOutput(_, _)
                | Op::PushThetaGather(_)
                | Op::Sqrt
                | Op::InvLogit
                | Op::Logit
                | Op::Tanh
                | Op::Softplus
                | Op::Sin
                | Op::Cos
                | Op::Pnorm
                | Op::Mod
                | Op::Jump(_),
                _,
            ) => return Err(refused_construct(ops[pc]).unwrap_or(SHAPE)),
        }
        Ok(pc + 1)
    }

    /// The inline `if` whose `JumpIfFalse` is at `pc`, in `compile_expr_into`'s shape: the
    /// then arm, a `Jump` past the else arm at `else_at − 1`, the else arm. Each arm is
    /// simulated from the stack the condition leaves, into its own instructions; the entries
    /// the arms leave different become values both arms [`Ins::Move`] into.
    ///
    /// The interpreter's fusion looks across the join: the else arm's last push fuses with an
    /// arithmetic op right after the `if` (`x * (if (…) a else 2)` runs `x *= 2` on that path),
    /// while the then arm's `Jump` lands on the op and runs it unfused. So when the else arm
    /// ends one op later, the then arm runs that op too, as its path does, and both continue
    /// after it.
    ///
    /// A construct an arm cannot translate is reported as itself (a `sqrt` in an arm is
    /// `sqrt`); an `if` whose bytecode is not in this shape is [`RhsConstruct::InlineIf`].
    fn conditional(
        &mut self,
        ops: &[Op],
        consts: &[f64],
        pc: usize,
        else_at: usize,
        stack: &mut Vec<Src>,
    ) -> Result<usize, RhsConstruct> {
        const IF: RhsConstruct = RhsConstruct::InlineIf;
        let c = stack.pop().ok_or(SHAPE)?;
        let jump = else_at.checked_sub(1).ok_or(IF)?;
        let Some(&Op::Jump(end)) = ops.get(jump) else {
            return Err(IF);
        };
        let end = end as usize;
        if !(pc < jump && else_at <= end && end <= ops.len()) {
            return Err(IF);
        }
        let outer = std::mem::take(&mut self.ins);
        // The then arm's last op looks ahead at the `Jump`, so nothing in it fuses past it.
        let mut then_stack = stack.clone();
        if self.run(ops, consts, pc + 1, jump, &mut then_stack)? != jump {
            return Err(IF);
        }
        let mut then_pc = end;
        let mut then_ins = std::mem::take(&mut self.ins);
        let mut else_stack = stack.clone();
        let else_pc = self.run(ops, consts, else_at, end, &mut else_stack)?;
        let mut else_ins = std::mem::take(&mut self.ins);
        if then_pc < else_pc {
            self.ins = then_ins;
            then_pc = self.step(ops, consts, then_pc, &mut then_stack)?;
            then_ins = std::mem::take(&mut self.ins);
        }
        if then_pc != else_pc || then_stack.len() != else_stack.len() {
            return Err(IF);
        }
        let same = |a: &Src, b: &Src| match (a, b) {
            // By bits: `0.0` and `−0.0` are different literals.
            (Src::Konst(x), Src::Konst(y)) => x.to_bits() == y.to_bits(),
            _ => a == b,
        };
        let mut merged = then_stack.clone();
        for (m, (a, b)) in merged.iter_mut().zip(then_stack.iter().zip(&else_stack)) {
            if same(a, b) {
                continue;
            }
            let dst = self.tmp().ok_or(SHAPE)?;
            then_ins.push(Ins::Move { dst, v: *a });
            else_ins.push(Ins::Move { dst, v: *b });
            *m = Src::Tmp(dst);
        }
        self.ins = outer;
        self.ins.push(Ins::Branch { c });
        self.ins.extend(then_ins);
        self.ins.push(Ins::Else);
        self.ins.extend(else_ins);
        self.ins.push(Ins::EndIf);
        *stack = merged;
        Ok(else_pc)
    }
}

#[path = "rhs_ir.rs"]
mod ir;
#[path = "rhs_jit.rs"]
mod jit;
#[path = "kernel_jit.rs"]
pub(crate) mod kernel_jit;

thread_local! {
    /// Lane buffers for a non-contiguous type's native call (`run_native`).
    static LANE_SCRATCH: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
thread_local! {
    /// Test switch: run the register code even where a native function exists, so a test
    /// can compare the two.
    pub(super) static RHS_JIT_DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
#[path = "rhs_regcode_tests.rs"]
mod tests;
