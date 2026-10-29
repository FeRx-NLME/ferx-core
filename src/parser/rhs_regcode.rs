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
//! interpreter applies at that point, through the interpreter's own primitive
//! (`apply_binary_in` with `Op::Div`'s guard, `apply_scalar_in`, `apply_unary`, the
//! `VmArith` in-place methods), with the interpreter's operand fusion reproduced exactly — a
//! constant right operand is applied as a scalar, a variable right operand in place, everything else as a dual
//! constant. So every value, and every tape node a taped type records, is bit for bit the
//! interpreter's. Nothing is shared between subexpressions (sharing a taped node would change
//! the order a reverse sweep accumulates adjoints in), and the hoist's parameter-only values
//! are recomputed rather than cached (a cache keyed on the parameters measured no faster).
//!
//! A program with any operation whose `f64` meaning the generic primitives do not reproduce
//! exactly (`sqrt`'s guard, `inv_logit`, `logit`, `mod`), that reads anything but its states,
//! parameters, seeds, its own locals and literals (θ, η, covariates, gathers, network
//! outputs, the mixture class; `TIME` is the time seed by the time a program is compiled), or
//! that reads a local before assigning it, is not compiled and keeps the interpreter.
use super::{apply_binary_in, apply_scalar_in, apply_unary, Bytecode, Op, Statement};
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
#[derive(Clone, Copy, Debug, PartialEq)]
enum Unary {
    Exp,
    Ln,
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
    /// A division by a literal, through `apply_scalar_in`'s guard.
    DivK {
        d: u16,
        a: u16,
        c: f64,
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

/// A compiled straight-line RHS; see the module docs.
#[derive(Clone, Debug)]
pub(super) struct RhsRegProgram {
    code: Vec<Flat>,
    n_regs: usize,
    /// The same program as native code, built per lane count on first use (see `rhs_jit`).
    jit: Option<jit::JitCode>,
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
        lower(&b.ins, b.n_tmp, layout)
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
    /// `f64` and the forward duals, the contiguous ones in place and `DualV` through its
    /// lanes when no input is on a tape. `false`, with nothing written, otherwise.
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
                Flat::AddK { d, a, c } => bin!(d, a, |x: &mut T| x.add_scalar_in(c)),
                Flat::SubK { d, a, c } => bin!(d, a, |x: &mut T| x.sub_scalar_in(c)),
                Flat::RSubK { d, a, c } => bin!(d, a, |x: &mut T| x.rsub_scalar_in(c)),
                Flat::MulK { d, a, c } => bin!(d, a, |x: &mut T| x.mul_scalar_in(c)),
                Flat::DivK { d, a, c } => bin!(d, a, |x: &mut T| apply_scalar_in(x, Op::Div, c)),
                Flat::Pow { d, a, e } => r[d as usize] = r[a as usize].pow(r[e as usize]),
                Flat::Unary { d, op, a } => r[d as usize] = apply_unary(op.op(), r[a as usize]),
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
/// sees the assignment, and so does this), and each unfused literal gets a register of its
/// own; [`reuse_registers`] then packs them by lifetime.
fn lower(ins: &[Ins], n_tmp: usize, layout: &RhsLayout<'_>) -> Option<RhsRegProgram> {
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
                    Arith::Div => Flat::DivK { d, a, c },
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
    let (code, n_regs) = reuse_registers(prologue, next);
    Some(RhsRegProgram {
        jit: Some(jit::JitCode::new(&code, n_regs)),
        code,
        n_regs,
    })
}

/// The register an instruction writes, and the ones it reads.
fn regs_of(f: &mut Flat) -> (Option<&mut u16>, [Option<&mut u16>; 2]) {
    match f {
        Flat::LoadState { d, .. }
        | Flat::LoadParam { d, .. }
        | Flat::LoadSeed { d, .. }
        | Flat::LoadK { d, .. } => (Some(d), [None, None]),
        Flat::Add { d, a, b }
        | Flat::Sub { d, a, b }
        | Flat::RSub { d, a, b }
        | Flat::Mul { d, a, b }
        | Flat::Div { d, a, b } => (Some(d), [Some(a), Some(b)]),
        Flat::Pow { d, a, e } => (Some(d), [Some(a), Some(e)]),
        Flat::AddK { d, a, .. }
        | Flat::SubK { d, a, .. }
        | Flat::RSubK { d, a, .. }
        | Flat::MulK { d, a, .. }
        | Flat::DivK { d, a, .. }
        | Flat::Unary { d, a, .. } => (Some(d), [Some(a), None]),
        Flat::Deriv { a, .. } => (None, [Some(a), None]),
    }
}

/// Map the program's virtual registers (one per value: every temporary, load and literal)
/// onto as few physical registers as their lifetimes allow, so `n_regs` follows the number
/// of values live at once rather than the program's length.
///
/// A register is free again after the instruction that reads it last, and that
/// instruction may write its result into it: every arm reads its operands before it writes
/// (`exec` copies them out first, and the native code is SSA), so the reuse changes where
/// values live, never which value an instruction reads. A value written and never read gets
/// a register for its write and gives it back at once.
fn reuse_registers(mut code: Vec<Flat>, n_virtual: usize) -> (Vec<Flat>, usize) {
    const DEAD: usize = usize::MAX;
    let mut last_use = vec![DEAD; n_virtual];
    for (i, f) in code.iter_mut().enumerate() {
        for r in regs_of(f).1.into_iter().flatten() {
            last_use[*r as usize] = i;
        }
    }
    let mut phys: Vec<Option<u16>> = vec![None; n_virtual];
    let mut free: Vec<u16> = Vec::new();
    let mut n_phys: u16 = 0;
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
            let p = free.pop().unwrap_or_else(|| {
                n_phys += 1;
                n_phys - 1
            });
            phys[v] = Some(p);
            *d = p;
            if last_use[v] == DEAD || last_use[v] < i {
                free.push(p);
            }
        }
    }
    (code, n_phys as usize)
}

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
                // Operand fusion, exactly as `eval_bytecode_g` applies it: a constant
                // consumed by the next arithmetic op is applied as a scalar, a variable in
                // place. (`PushCov` and `PushTime` fuse too there, but the RHS reads no
                // covariate, and the parser rewrites every `TIME` to the time slot's
                // `PushVar` before compiling — `rewrite_push_time_to_slot`.)
                (Op::PushConst(i), Some(op)) => {
                    let a = stack.pop()?;
                    let c = *consts.get(i as usize)?;
                    let dst = self.tmp()?;
                    self.ins.push(Ins::Scalar { dst, op, a, c });
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

#[path = "rhs_ir.rs"]
mod ir;
#[path = "rhs_jit.rs"]
mod jit;

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
