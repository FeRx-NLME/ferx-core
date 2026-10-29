//! Native code for the register program (Cranelift).
//!
//! [`super::RhsRegProgram`]'s dispatch loop costs ~1.4 ns per instruction; the same
//! operations as machine code cost a fraction of that, and the RHS is most of an ODE fit's
//! time. So a compiled program is compiled once more, per number type's lane layout
//! ([`Layout`]), to native functions over the same inputs: `f64` (every solve of the
//! objective), the first-order duals of the inner gradient, and the second-order duals of
//! the exact inner Hessian — every type whose parts are all `f64`s.
//!
//! **It is the register code's translation, lane for lane.** The program over a layout is
//! first written out as scalar straight-line code (`super::ir`), each dual operation as
//! the IEEE operations of that type's own body in its own order, and that is lowered here
//! one instruction each. Cranelift never contracts a multiply and an add into a fused one
//! and never folds `x·0`, so the arithmetic is IEEE's. `exp`, `ln`, `powf` and `round`
//! call the Rust functions the interpreter calls, through `extern "C"` shims; `floor`,
//! `ceil`, `abs` and `neg` are exact IEEE operations either way. The one thing not pinned is
//! a `NaN`'s payload, which LLVM leaves unspecified for the Rust side too.
//!
//! Three functions can be built per layout, each on first use:
//!
//! - the **whole RHS**, `fn(u, params, seeds, du)`;
//! - the **hoisted pair** ([`HoistPlan`]): `prep` computes the parameter-only values once,
//!   `main` is the RHS reading them, for a caller that keeps them across an integration;
//! - the **reverse pass** (`super::ir::reverse`): `fn(u, params, seeds, du, ū, p̄, s̄, dū)`
//!   adds `dūᵀ·∂du/∂(u, params, seeds)` into the three input adjoints, the RHS's
//!   vector-Jacobian product for a tape that records one node per call.
//!
//! A native function reads its operands without bounds checks, so it runs only when every
//! slot the program reads or writes is inside the slices it is handed ([`Extent`]); the
//! register code covers the rest (it reads a missing state or parameter as `0`, as the
//! binding does).
use super::ir::{self, Cc, Func, Inst, Layout, Shim1};
use super::Flat;
use crate::sens::num::vm::{Nest, RhsKernel};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, Signature, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::sync::{Arc, OnceLock};

/// `fn(u, params, seeds, du)`; every pointer is to elements of one layout's lanes.
type RhsFn = unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64);
/// The hoisted pair ([`HoistPlan`]): `prep(params, seeds, cache)` writes the
/// parameter-only values, `main(u, params, seeds, du, cache)` is the RHS reading them.
pub(super) type PrepFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64);
pub(super) type MainFn =
    unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64, *const f64);
/// The reverse pass: `(u, params, seeds, du, ū, p̄, s̄, dū)`.
type VjpFn = unsafe extern "C" fn(
    *const f64,
    *const f64,
    *const f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
);

extern "C" fn shim_exp(x: f64) -> f64 {
    x.exp()
}
extern "C" fn shim_ln(x: f64) -> f64 {
    x.ln()
}
extern "C" fn shim_powf(x: f64, e: f64) -> f64 {
    x.powf(e)
}
extern "C" fn shim_round(x: f64) -> f64 {
    x.round()
}

/// The highest state, parameter and derivative index the program touches (+1), so a call
/// can check its slices once instead of the native code checking every access.
#[derive(Clone, Copy, Debug, Default)]
struct Extent {
    u: usize,
    params: usize,
    du: usize,
    /// Every derivative below `du` is written, so a call's output never depends on what
    /// `du` held before it.
    du_dense: bool,
}

fn extent(code: &[Flat]) -> Extent {
    let mut e = Extent::default();
    let mut written = Vec::new();
    for f in code {
        match *f {
            Flat::LoadState { i, .. } => e.u = e.u.max(i as usize + 1),
            Flat::LoadParam { slot, .. } => e.params = e.params.max(slot as usize + 1),
            Flat::Deriv { state, .. } => {
                let s = state as usize;
                e.du = e.du.max(s + 1);
                if written.len() <= s {
                    written.resize(s + 1, false);
                }
                written[s] = true;
            }
            _ => {}
        }
    }
    e.du_dense = written.iter().all(|&w| w);
    e
}

/// The instructions whose value depends only on parameters, literals and `MACHEPS` — fixed
/// for a whole integration — and which of them the rest of the program reads.
///
/// A caller that keeps a cache runs `prep` once and `main` per call: `main` loads those
/// values instead of recomputing them. The operations and their inputs are the same, so
/// the values are bit for bit the full function's.
#[derive(Debug, Default)]
struct HoistPlan {
    /// Per instruction: computed by `prep` rather than `main`.
    invariant: Vec<bool>,
    /// Per instruction: its cache slot, when `main` reads its value.
    slot: Vec<Option<u32>>,
    n_slots: usize,
    /// The parameter slots `prep` reads, ascending: the cache's key.
    param_slots: Vec<u32>,
}

fn hoist_plan(code: &[Flat]) -> HoistPlan {
    let n_regs = code
        .iter()
        .flat_map(|f| {
            let mut g = *f;
            let (d, uses) = super::regs_of(&mut g);
            let mut v: Vec<usize> = uses.into_iter().flatten().map(|r| *r as usize).collect();
            v.extend(d.map(|r| *r as usize));
            v
        })
        .max()
        .map_or(0, |m| m + 1);
    let mut producer = vec![usize::MAX; n_regs];
    let mut invariant = vec![false; code.len()];
    let mut read_by_main = vec![false; code.len()];
    let mut param_slots = Vec::new();
    for (i, f) in code.iter().enumerate() {
        let mut g = *f;
        let (d, uses) = super::regs_of(&mut g);
        let from: Vec<usize> = uses
            .into_iter()
            .flatten()
            .map(|r| producer[*r as usize])
            .collect();
        invariant[i] = match *f {
            Flat::LoadParam { slot, .. } => {
                param_slots.push(slot);
                true
            }
            Flat::LoadK { .. } => true,
            // `MACHEPS`; the clock seeds move with `t`.
            Flat::LoadSeed { k, .. } => k == 3,
            Flat::LoadState { .. } | Flat::Deriv { .. } => false,
            _ => from.iter().all(|&p| invariant[p]),
        };
        if !invariant[i] {
            for p in from {
                if invariant[p] {
                    read_by_main[p] = true;
                }
            }
        }
        if let Some(d) = d {
            producer[*d as usize] = i;
        }
    }
    let mut n_slots = 0;
    let slot = read_by_main
        .iter()
        .map(|&r| {
            r.then(|| {
                n_slots += 1;
                (n_slots - 1) as u32
            })
        })
        .collect();
    param_slots.sort_unstable();
    param_slots.dedup();
    HoistPlan {
        invariant,
        slot,
        n_slots,
        param_slots,
    }
}

/// Which function [`emit`] writes from the program.
#[derive(Clone, Copy)]
enum Part<'a> {
    /// The whole RHS (`RhsFn`).
    Full,
    /// The parameter-only instructions, storing what `Main` reads (`PrepFn`).
    Prep(&'a HoistPlan),
    /// The rest, loading the parameter-only values from the cache (`MainFn`).
    Main(&'a HoistPlan),
}

/// The program over `layout` as scalar code, whole or one half of the hoisted pair.
fn emit(code: &[Flat], n_regs: usize, layout: Layout, part: Part<'_>) -> Func {
    match part {
        Part::Full => ir::forward(code, n_regs, layout),
        // `prep(params, seeds, cache)`: no instruction it runs reads `u` or writes `du`.
        Part::Prep(plan) => {
            let mut f = Func::new(3);
            let mut l = ir::Lower::new(&mut f, layout, n_regs);
            for (i, x) in code.iter().enumerate() {
                if !plan.invariant[i] {
                    continue;
                }
                l.flat(x, [0, 0, 1, 2]);
                if let (Some(slot), Some(d)) = (plan.slot[i], def_of(x)) {
                    let lanes = l.reg(d);
                    l.store(2, slot as usize, &lanes);
                }
            }
            f
        }
        Part::Main(plan) => {
            let mut f = Func::new(5);
            let mut l = ir::Lower::new(&mut f, layout, n_regs);
            for (i, x) in code.iter().enumerate() {
                if !plan.invariant[i] {
                    l.flat(x, [0, 1, 2, 3]);
                } else if let (Some(slot), Some(d)) = (plan.slot[i], def_of(x)) {
                    let lanes = l.load(4, slot as usize);
                    l.set(d, lanes);
                }
            }
            f
        }
    }
}

/// The register an instruction writes.
fn def_of(f: &Flat) -> Option<u16> {
    let mut g = *f;
    let (d, _) = super::regs_of(&mut g);
    d.map(|r| *r)
}

/// A JIT module for the host, or `None` when Cranelift cannot target it.
fn new_module() -> Option<JITModule> {
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").ok()?;
    flags.set("use_colocated_libcalls", "false").ok()?;
    flags.set("is_pic", "false").ok()?;
    let isa = cranelift_native::builder()
        .ok()?
        .finish(settings::Flags::new(flags))
        .ok()?;
    Some(JITModule::new(JITBuilder::with_isa(
        isa,
        default_libcall_names(),
    )))
}

/// Native functions and the module that owns their memory.
struct Compiled<F> {
    f: F,
    module: Option<JITModule>,
}

impl<F> Drop for Compiled<F> {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: every function pointer in `f` points into this module's memory and
            // dies with this value; `JitCode` hands out no copy that could outlive the `Arc`.
            unsafe { m.free_memory() };
        }
    }
}

/// Lower `funcs` into one module and finalize it; the addresses in `funcs`' order.
fn compile(funcs: &[(&str, &Func)]) -> Option<(JITModule, Vec<*const u8>)> {
    let mut module = new_module()?;
    let mut ids = Vec::with_capacity(funcs.len());
    for (name, f) in funcs {
        ids.push(lower(&mut module, name, f)?);
    }
    module.finalize_definitions().ok()?;
    let addrs = ids
        .into_iter()
        .map(|id| module.get_finalized_function(id))
        .collect();
    Some((module, addrs))
}

/// Define one [`Func`] in `module`, one Cranelift instruction per scalar instruction.
fn lower(module: &mut JITModule, name: &str, f: &Func) -> Option<cranelift_module::FuncId> {
    let target = module.target_config();
    let ptr = target.pointer_type();
    let call_conv = target.default_call_conv;
    let mut sig = module.make_signature();
    for _ in 0..f.n_args {
        sig.params.push(AbiParam::new(ptr));
    }
    let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let block = b.create_block();
        b.append_block_params_for_function_params(block);
        b.switch_to_block(block);
        let args = b.block_params(block).to_vec();
        let mut unary_sig = Signature::new(call_conv);
        unary_sig.params.push(AbiParam::new(types::F64));
        unary_sig.returns.push(AbiParam::new(types::F64));
        let mut binary_sig = unary_sig.clone();
        binary_sig.params.push(AbiParam::new(types::F64));
        let unary_ref = b.import_signature(unary_sig);
        let binary_ref = b.import_signature(binary_sig);
        let flags = MemFlagsData::trusted();
        let mut v: Vec<Value> = Vec::with_capacity(f.insts.len());
        for inst in &f.insts {
            let r = |x: ir::V| v[x as usize];
            let out = match *inst {
                Inst::Load { arg, off } => {
                    b.ins()
                        .load(types::F64, flags, args[arg as usize], (off * 8) as i32)
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
                Inst::Call1(s, x) => {
                    let f: extern "C" fn(f64) -> f64 = match s {
                        Shim1::Exp => shim_exp,
                        Shim1::Ln => shim_ln,
                        Shim1::Round => shim_round,
                    };
                    let callee = b.ins().iconst(ptr, f as usize as i64);
                    let call = b.ins().call_indirect(unary_ref, callee, &[r(x)]);
                    b.inst_results(call)[0]
                }
                Inst::Powf(x, y) => {
                    let f: extern "C" fn(f64, f64) -> f64 = shim_powf;
                    let callee = b.ins().iconst(ptr, f as usize as i64);
                    let call = b.ins().call_indirect(binary_ref, callee, &[r(x), r(y)]);
                    b.inst_results(call)[0]
                }
                Inst::Cmp(cc, x, y) => {
                    let cc = match cc {
                        Cc::Lt => FloatCC::LessThan,
                        Cc::Gt => FloatCC::GreaterThan,
                        Cc::Ge => FloatCC::GreaterThanOrEqual,
                        Cc::Eq => FloatCC::Equal,
                    };
                    b.ins().fcmp(cc, r(x), r(y))
                }
                Inst::And(x, y) => b.ins().band(r(x), r(y)),
                Inst::Select(c, x, y) => b.ins().select(r(c), r(x), r(y)),
                Inst::Store { arg, off, v: x } => {
                    b.ins()
                        .store(flags, r(x), args[arg as usize], (off * 8) as i32);
                    // A store has no value; nothing reads this slot.
                    r(x)
                }
            };
            v.push(out);
        }
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(target);
    }
    module.define_function(id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    Some(id)
}

/// The native functions of one register program, built on first use per layout and shared
/// by every clone of the program.
#[derive(Clone)]
pub(super) struct JitCode {
    inner: Arc<JitInner>,
    /// The same allocation as a tape's [`RhsKernel`], so recording a call borrows it.
    kernel: Arc<dyn RhsKernel>,
}

struct JitInner {
    code: Vec<Flat>,
    n_regs: usize,
    extent: Extent,
    plan: HoistPlan,
    full: Box<[OnceLock<Option<Compiled<RhsFn>>>]>,
    pairs: Box<[OnceLock<Option<Compiled<(PrepFn, MainFn)>>>]>,
    vjps: Box<[OnceLock<Option<Compiled<VjpFn>>>]>,
    /// Forward and reverse pass per nested dual ([`Nest`]): few per program, so a list.
    nests: std::sync::RwLock<Vec<(Nest, Option<Arc<Compiled<(RhsFn, VjpFn)>>>)>>,
}

// SAFETY: the native code is immutable once finalized and reentrant (it touches only its
// arguments and the stack); the module that owns it is never touched again until it is
// freed, by the last owner.
unsafe impl Send for JitInner {}
unsafe impl Sync for JitInner {}

impl std::fmt::Debug for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCode")
            .field("instructions", &self.inner.code.len())
            .finish()
    }
}

fn slots<F>() -> Box<[OnceLock<Option<Compiled<F>>>]> {
    (0..ir::N_SLOTS).map(|_| OnceLock::new()).collect()
}

impl JitCode {
    pub(super) fn new(code: &[Flat], n_regs: usize) -> Self {
        let inner = Arc::new(JitInner {
            code: code.to_vec(),
            n_regs,
            extent: extent(code),
            plan: hoist_plan(code),
            full: slots(),
            pairs: slots(),
            vjps: slots(),
            nests: Default::default(),
        });
        JitCode {
            kernel: inner.clone(),
            inner,
        }
    }

    /// This program as a tape's [`RhsKernel`].
    pub(super) fn kernel(&self) -> &Arc<dyn RhsKernel> {
        &self.kernel
    }

    /// Whether a native function exists (or can be built) for `layout`.
    pub(super) fn has(&self, layout: Layout) -> bool {
        self.resolve(layout).is_some()
    }

    /// The whole RHS over `layout`, built on first use; `None` for a layout without a slot
    /// or a host Cranelift cannot target.
    pub(super) fn resolve(&self, layout: Layout) -> Option<Resolved<'_, RhsFn>> {
        self.inner.resolve(layout)
    }

    /// The hoisted pair over `layout` ([`HoistPlan`]), built on first use.
    pub(super) fn resolve_hoisted(&self, layout: Layout) -> Option<Resolved<'_, (PrepFn, MainFn)>> {
        self.inner.resolve_hoisted(layout)
    }

    /// The reverse pass over `layout`, built on first use.
    #[cfg(test)]
    pub(super) fn resolve_vjp(&self, layout: Layout) -> Option<Resolved<'_, VjpFn>> {
        self.inner.resolve_vjp(layout)
    }
}

impl JitInner {
    fn resolve(&self, layout: Layout) -> Option<Resolved<'_, RhsFn>> {
        let s = ir::slot(layout)?;
        let inner = self;
        let c = inner.full[s]
            .get_or_init(|| {
                let f = emit(&inner.code, inner.n_regs, layout, Part::Full);
                let (module, a) = compile(&[("rhs", &f)])?;
                // SAFETY: defined with exactly `RhsFn`'s signature.
                let f = unsafe { std::mem::transmute::<*const u8, RhsFn>(a[0]) };
                Some(Compiled {
                    f,
                    module: Some(module),
                })
            })
            .as_ref()?;
        Some(self.resolved(c.f, layout))
    }

    fn resolve_hoisted(&self, layout: Layout) -> Option<Resolved<'_, (PrepFn, MainFn)>> {
        let s = ir::slot(layout)?;
        let inner = self;
        let c = inner.pairs[s]
            .get_or_init(|| {
                let prep = emit(&inner.code, inner.n_regs, layout, Part::Prep(&inner.plan));
                let main = emit(&inner.code, inner.n_regs, layout, Part::Main(&inner.plan));
                let (module, a) = compile(&[("rhs_prep", &prep), ("rhs_main", &main)])?;
                // SAFETY: each defined with exactly its type's signature.
                let f = unsafe {
                    (
                        std::mem::transmute::<*const u8, PrepFn>(a[0]),
                        std::mem::transmute::<*const u8, MainFn>(a[1]),
                    )
                };
                Some(Compiled {
                    f,
                    module: Some(module),
                })
            })
            .as_ref()?;
        Some(self.resolved(c.f, layout))
    }

    fn resolve_vjp(&self, layout: Layout) -> Option<Resolved<'_, VjpFn>> {
        let s = ir::slot(layout)?;
        let inner = self;
        let c = inner.vjps[s]
            .get_or_init(|| {
                let fwd = emit(&inner.code, inner.n_regs, layout, Part::Full);
                let rev = ir::reverse(&fwd);
                let (module, a) = compile(&[("rhs_vjp", &rev)])?;
                // SAFETY: defined with exactly `VjpFn`'s signature (eight pointers).
                let f = unsafe { std::mem::transmute::<*const u8, VjpFn>(a[0]) };
                Some(Compiled {
                    f,
                    module: Some(module),
                })
            })
            .as_ref()?;
        Some(self.resolved(c.f, layout))
    }

    fn resolved<F: Copy>(&self, f: F, layout: Layout) -> Resolved<'_, F> {
        Resolved {
            f,
            extent: self.extent,
            lanes: layout.lanes(),
            plan: &self.plan,
        }
    }
}

impl JitInner {
    /// The forward and reverse pass over `nest`, built on first use.
    fn nest(&self, nest: Nest) -> Option<Arc<Compiled<(RhsFn, VjpFn)>>> {
        if let Some((_, c)) = self.nests.read().ok()?.iter().find(|(n, _)| *n == nest) {
            return c.clone();
        }
        let mut w = self.nests.write().ok()?;
        if let Some((_, c)) = w.iter().find(|(n, _)| *n == nest) {
            return c.clone();
        }
        let built = (|| {
            if nest.size_from(0) > ir::MAX_LANES {
                return None;
            }
            let fwd = ir::forward_nest(&self.code, self.n_regs, nest);
            let rev = ir::reverse(&fwd);
            let (module, a) = compile(&[("rhs_nest", &fwd), ("rhs_nest_vjp", &rev)])?;
            // SAFETY: each defined with exactly its type's signature.
            let f = unsafe {
                (
                    std::mem::transmute::<*const u8, RhsFn>(a[0]),
                    std::mem::transmute::<*const u8, VjpFn>(a[1]),
                )
            };
            Some(Arc::new(Compiled {
                f,
                module: Some(module),
            }))
        })();
        w.push((nest, built.clone()));
        built
    }
}

impl RhsKernel for JitInner {
    fn has_nest(&self, nest: Nest) -> bool {
        self.nest(nest).is_some()
    }
    unsafe fn forward_nest(
        &self,
        nest: Nest,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
    ) {
        if let Some(c) = self.nest(nest) {
            (c.f.0)(u, params, seeds, du)
        }
    }
    unsafe fn reverse_nest(
        &self,
        nest: Nest,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        u_bar: *mut f64,
        p_bar: *mut f64,
        s_bar: *mut f64,
        du_bar: *const f64,
    ) {
        if let Some(c) = self.nest(nest) {
            (c.f.1)(
                u,
                params,
                seeds,
                std::ptr::null_mut(),
                u_bar,
                p_bar,
                s_bar,
                du_bar,
            )
        }
    }
    fn extent(&self) -> (usize, usize, usize) {
        (self.extent.u, self.extent.params, self.extent.du)
    }
    fn writes_every_du(&self) -> bool {
        self.extent.du_dense
    }
    fn has_forward(&self, layout: Layout) -> bool {
        self.resolve(layout).is_some()
    }
    fn has_reverse(&self, layout: Layout) -> bool {
        self.resolve_vjp(layout).is_some()
    }
    unsafe fn forward(
        &self,
        layout: Layout,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
    ) {
        if let Some(f) = self.resolve(layout) {
            f.call(u, params, seeds, du)
        }
    }
    unsafe fn reverse(
        &self,
        layout: Layout,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        u_bar: *mut f64,
        p_bar: *mut f64,
        s_bar: *mut f64,
        du_bar: *const f64,
    ) {
        if let Some(f) = self.resolve_vjp(layout) {
            // The reverse pass never touches `du` (its forward half drops the stores).
            f.vjp(
                u,
                params,
                seeds,
                std::ptr::null_mut(),
                u_bar,
                p_bar,
                s_bar,
                du_bar,
            )
        }
    }
}

/// One native function (or the hoisted pair) of a [`JitCode`], resolved for one layout: a
/// call is the extent check and the call itself. It borrows the code, so it cannot outlive
/// it.
#[derive(Clone, Copy)]
pub(super) struct Resolved<'a, F> {
    f: F,
    extent: Extent,
    lanes: usize,
    plan: &'a HoistPlan,
}

impl<F> Resolved<'_, F> {
    /// Whether `nu`, `np` and `ndu` elements cover what the program reads and writes.
    #[inline(always)]
    pub(super) fn fits(&self, nu: usize, np: usize, ndu: usize) -> bool {
        let e = self.extent;
        nu >= e.u && np >= e.params && ndu >= e.du
    }
    /// How many elements of `u`, `params` and `du` the program touches.
    pub(super) fn extent(&self) -> (usize, usize, usize) {
        (self.extent.u, self.extent.params, self.extent.du)
    }
    pub(super) fn lanes(&self) -> usize {
        self.lanes
    }
}

impl Resolved<'_, RhsFn> {
    /// The RHS.
    ///
    /// # Safety
    /// [`Self::fits`] holds for the slices behind `u`, `params` and `du`; `seeds` holds 4
    /// elements; every element is [`Self::lanes`] contiguous `f64`s, and `du` does not
    /// overlap the inputs.
    #[inline(always)]
    pub(super) unsafe fn call(
        &self,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
    ) {
        (self.f)(u, params, seeds, du)
    }
}

impl Resolved<'_, (PrepFn, MainFn)> {
    /// `f64`s of cache the pair needs.
    pub(super) fn cache_len(&self) -> usize {
        self.plan.n_slots * self.lanes
    }
    /// The parameter slots whose values the cache depends on (with `seeds[3]`).
    pub(super) fn param_slots(&self) -> &[u32] {
        &self.plan.param_slots
    }
    /// Fill `cache` from `params` and `seeds`.
    ///
    /// # Safety
    /// `params` covers the program's extent, `seeds` holds 4 elements and `cache`
    /// [`Self::cache_len`] `f64`s.
    #[inline(always)]
    pub(super) unsafe fn prep(&self, params: *const f64, seeds: *const f64, cache: *mut f64) {
        (self.f.0)(params, seeds, cache)
    }
    /// The RHS from a filled `cache`.
    ///
    /// # Safety
    /// As [`Resolved::call`], and `cache` filled by [`Self::prep`] from the same `params`
    /// and `seeds[3]`.
    #[inline(always)]
    pub(super) unsafe fn main(
        &self,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
        cache: *const f64,
    ) {
        (self.f.1)(u, params, seeds, du, cache)
    }
}

impl Resolved<'_, VjpFn> {
    /// Add `dūᵀ·∂du/∂(u, params, seeds)` into `ū`, `p̄` and `s̄`.
    ///
    /// # Safety
    /// As [`Resolved::call`] for the primal slices; each adjoint buffer is laid out like
    /// its primal (`ū` like `u`, …, `dū` like `du`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) unsafe fn vjp(
        &self,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
        u_bar: *mut f64,
        p_bar: *mut f64,
        s_bar: *mut f64,
        du_bar: *const f64,
    ) {
        (self.f)(u, params, seeds, du, u_bar, p_bar, s_bar, du_bar)
    }
}
