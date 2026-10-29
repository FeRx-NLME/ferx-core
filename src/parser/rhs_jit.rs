//! Native code for the register program (Cranelift).
//!
//! [`super::RhsRegProgram`]'s dispatch loop costs ~1.4 ns per instruction; the same
//! operations as machine code cost a fraction of that, and the RHS is most of an ODE fit's
//! time. So a compiled program is compiled once more, per lane count, to a native function
//! over the same inputs:
//!
//! - **`f64`**, one lane: every solve of the objective;
//! - **`DualF<N, 0>`** for `N < 8`, `N + 1` lanes (value, then the `N` tangents): the inner
//!   gradient's tape-free recordings. A dual from `N = 8` runs its arithmetic out of line
//!   (`crate::sens::num::outlined`), with a different operation order for a constant
//!   operand, so it keeps the register code.
//!
//! **It is the register code's translation, lane for lane.** Each [`Flat`] arm emits the
//! IEEE operations of the primitive that arm calls, in the primitive's own order: the `f64`
//! operators, or `DualV`'s untaped bodies that `DualF` runs (`mul_body`: `g = a.g·b.v +
//! a.v·b.g`; `div_body`: `q = a/b`, `ib = 1/b`, `g = (a.g − q·b.g)·ib`; a constant right
//! operand's `add_c`/`mul_c`/`div_c`; `rsub_c` as `0 − g`, not `−g`). Cranelift never
//! contracts a multiply and an add into a fused one and never folds `x·0`, so the
//! arithmetic is IEEE's. `exp`, `ln`, `powf` and `round` call the Rust functions the
//! interpreter calls, through `extern "C"` shims; `floor`, `ceil`, `abs`, `neg` and `sqrt`
//! are exact IEEE operations either way. The guards are selects on the same predicates
//! (`|b| < 1e-30` for a division, `f64::max`'s NaN rule and `DualV::guard_floor`'s
//! `!(v ≥ lo)` for `ln`, `DualV::abs`'s `v ≥ 0`). The one thing not pinned is a `NaN`'s
//! payload, which LLVM leaves unspecified for the Rust side too.
//!
//! The native function reads its operands without bounds checks, so [`JitCode::call`] runs it
//! only when every slot the program reads or writes is inside the slices it is handed; the
//! register code covers the rest (it reads a missing state or parameter as `0`, as the
//! binding does).
use super::{Arith, Flat, Unary};
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, Signature, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::sync::{Arc, OnceLock};

/// Lane counts a native function can be built for: `f64` (1) and `DualF<N, 0>`, `N ≤ 7`.
pub(super) const MAX_LANES: usize = 8;

/// `fn(u, params, seeds, du)`; every pointer is to `lanes`-wide elements.
type RhsFn = unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64);
/// The hoisted pair ([`HoistPlan`]): `prep(params, seeds, cache)` writes the
/// parameter-only values, `main(u, params, seeds, du, cache)` is the RHS reading them.
type PrepFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64);
type MainFn = unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64, *const f64);

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
}

fn extent(code: &[Flat]) -> Extent {
    let mut e = Extent::default();
    for f in code {
        match *f {
            Flat::LoadState { i, .. } => e.u = e.u.max(i as usize + 1),
            Flat::LoadParam { slot, .. } => e.params = e.params.max(slot as usize + 1),
            Flat::Deriv { state, .. } => e.du = e.du.max(state as usize + 1),
            _ => {}
        }
    }
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

/// Which function [`build`] emits from the program.
#[derive(Clone, Copy)]
enum Part<'a> {
    /// The whole RHS (`RhsFn`).
    Full,
    /// The parameter-only instructions, storing what `Main` reads (`PrepFn`).
    Prep(&'a HoistPlan),
    /// The rest, loading the parameter-only values from the cache (`MainFn`).
    Main(&'a HoistPlan),
}

/// The hoisted pair and the module that owns them.
struct CompiledPair {
    prep: PrepFn,
    main: MainFn,
    module: Option<JITModule>,
}

impl Drop for CompiledPair {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: as `Compiled`'s drop.
            unsafe { m.free_memory() };
        }
    }
}

/// One native function and the module that owns its memory.
struct Compiled {
    f: RhsFn,
    module: Option<JITModule>,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: `f` points into this module's memory and dies with this value;
            // `JitCode` hands out no copy of it that could outlive the `Arc`.
            unsafe { m.free_memory() };
        }
    }
}

/// The native functions of one register program, built on first use per lane count and
/// shared by every clone of the program.
#[derive(Clone)]
pub(super) struct JitCode {
    inner: Arc<JitInner>,
}

struct JitInner {
    code: Vec<Flat>,
    n_regs: usize,
    extent: Extent,
    by_lanes: [OnceLock<Option<Compiled>>; MAX_LANES],
    plan: HoistPlan,
    pairs: [OnceLock<Option<CompiledPair>>; MAX_LANES],
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

impl JitCode {
    pub(super) fn new(code: &[Flat], n_regs: usize) -> Self {
        JitCode {
            inner: Arc::new(JitInner {
                code: code.to_vec(),
                n_regs,
                extent: extent(code),
                by_lanes: Default::default(),
                plan: hoist_plan(code),
                pairs: Default::default(),
            }),
        }
    }

    pub(super) fn has(&self, lanes: usize) -> bool {
        (1..=MAX_LANES).contains(&lanes) && self.function(lanes).is_some()
    }

    fn function(&self, lanes: usize) -> Option<RhsFn> {
        let inner = &*self.inner;
        inner.by_lanes[lanes - 1]
            .get_or_init(|| compile(&inner.code, inner.n_regs, lanes))
            .as_ref()
            .map(|c| c.f)
    }

    /// Run the native function over `lanes`-wide elements, or return `false` (nothing
    /// written) when there is none for `lanes` or a slice is shorter than the program reads.
    ///
    /// # Safety
    /// `u`, `params`, `seeds` and `du` must point to `nu`, `np`, `4` and `ndu` contiguous
    /// elements of `lanes` `f64`s each.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn call(
        &self,
        lanes: usize,
        u: *const f64,
        nu: usize,
        params: *const f64,
        np: usize,
        seeds: *const f64,
        du: *mut f64,
        ndu: usize,
    ) -> bool {
        if !(1..=MAX_LANES).contains(&lanes) {
            return false;
        }
        match self.resolve(lanes) {
            Some(r) => r.call(u, nu, params, np, seeds, du, ndu),
            None => false,
        }
    }

    /// The native function for `lanes`, looked up once for a caller that will run it many
    /// times (an integrator's every stage). It borrows `self`, so it cannot outlive the code.
    pub(super) fn resolve(&self, lanes: usize) -> Option<Resolved<'_>> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return None;
        }
        self.function(lanes).map(|f| Resolved {
            f,
            extent: self.inner.extent,
            _code: std::marker::PhantomData,
        })
    }
}

impl JitCode {
    /// The hoisted pair for `lanes` ([`HoistPlan`]), built on first use.
    pub(super) fn resolve_hoisted(&self, lanes: usize) -> Option<ResolvedPair<'_>> {
        if !(1..=MAX_LANES).contains(&lanes) {
            return None;
        }
        let inner = &*self.inner;
        let pair = inner.pairs[lanes - 1]
            .get_or_init(|| compile_pair(&inner.code, inner.n_regs, lanes, &inner.plan))
            .as_ref()?;
        Some(ResolvedPair {
            prep: pair.prep,
            main: pair.main,
            extent: inner.extent,
            lanes,
            plan: &inner.plan,
        })
    }
}

/// A program's hoisted pair, resolved ([`JitCode::resolve_hoisted`]).
#[derive(Clone, Copy)]
pub(super) struct ResolvedPair<'a> {
    prep: PrepFn,
    main: MainFn,
    extent: Extent,
    lanes: usize,
    plan: &'a HoistPlan,
}

impl ResolvedPair<'_> {
    /// `f64`s of cache the pair needs.
    pub(super) fn cache_len(&self) -> usize {
        self.plan.n_slots * self.lanes
    }

    /// The parameter slots whose values the cache depends on (with `seeds[3]`).
    pub(super) fn param_slots(&self) -> &[u32] {
        &self.plan.param_slots
    }

    /// Whether `nu`, `np` and `ndu` cover what the program reads and writes.
    #[inline(always)]
    pub(super) fn fits(&self, nu: usize, np: usize, ndu: usize) -> bool {
        let e = self.extent;
        nu >= e.u && np >= e.params && ndu >= e.du
    }

    /// Fill `cache` from `params` and `seeds`.
    ///
    /// # Safety
    /// `params` covers the program's extent, `seeds` holds 4 elements and `cache`
    /// [`Self::cache_len`] `f64`s, all of `lanes` `f64`s each.
    #[inline(always)]
    pub(super) unsafe fn prep(&self, params: *const f64, seeds: *const f64, cache: *mut f64) {
        (self.prep)(params, seeds, cache)
    }

    /// The RHS from a filled `cache`.
    ///
    /// # Safety
    /// As [`JitCode::call`] with [`Self::fits`] true, and `cache` filled by [`Self::prep`]
    /// from the same `params` and `seeds[3]`.
    #[inline(always)]
    pub(super) unsafe fn main(
        &self,
        u: *const f64,
        params: *const f64,
        seeds: *const f64,
        du: *mut f64,
        cache: *const f64,
    ) {
        (self.main)(u, params, seeds, du, cache)
    }
}

/// One native function of a [`JitCode`], resolved: a call is the extent check and the
/// call itself.
#[derive(Clone, Copy)]
pub(super) struct Resolved<'a> {
    f: RhsFn,
    extent: Extent,
    _code: std::marker::PhantomData<&'a JitInner>,
}

impl Resolved<'_> {
    /// [`JitCode::call`] without the lookup.
    ///
    /// # Safety
    /// As [`JitCode::call`], for the lane count this was resolved at.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) unsafe fn call(
        &self,
        u: *const f64,
        nu: usize,
        params: *const f64,
        np: usize,
        seeds: *const f64,
        du: *mut f64,
        ndu: usize,
    ) -> bool {
        let e = self.extent;
        if nu < e.u || np < e.params || ndu < e.du {
            return false;
        }
        (self.f)(u, params, seeds, du);
        true
    }
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

/// Build the native function for `lanes`, or `None` when Cranelift cannot target the host.
fn compile(code: &[Flat], n_regs: usize, lanes: usize) -> Option<Compiled> {
    let mut module = new_module()?;
    let id = build(&mut module, "rhs", code, n_regs, lanes, Part::Full)?;
    module.finalize_definitions().ok()?;
    let addr = module.get_finalized_function(id);
    // SAFETY: the function was defined with exactly `RhsFn`'s signature.
    let f = unsafe { std::mem::transmute::<*const u8, RhsFn>(addr) };
    Some(Compiled {
        f,
        module: Some(module),
    })
}

/// Build the hoisted pair for `lanes` ([`HoistPlan`]).
fn compile_pair(
    code: &[Flat],
    n_regs: usize,
    lanes: usize,
    plan: &HoistPlan,
) -> Option<CompiledPair> {
    let mut module = new_module()?;
    let prep = build(
        &mut module,
        "rhs_prep",
        code,
        n_regs,
        lanes,
        Part::Prep(plan),
    )?;
    let main = build(
        &mut module,
        "rhs_main",
        code,
        n_regs,
        lanes,
        Part::Main(plan),
    )?;
    module.finalize_definitions().ok()?;
    let (pa, ma) = (
        module.get_finalized_function(prep),
        module.get_finalized_function(main),
    );
    // SAFETY: each function was defined with exactly its type's signature.
    let (prep, main) = unsafe {
        (
            std::mem::transmute::<*const u8, PrepFn>(pa),
            std::mem::transmute::<*const u8, MainFn>(ma),
        )
    };
    Some(CompiledPair {
        prep,
        main,
        module: Some(module),
    })
}

/// Define one function of the program in `module`: the whole RHS, or one half of the
/// hoisted pair.
fn build(
    module: &mut JITModule,
    name: &str,
    code: &[Flat],
    n_regs: usize,
    lanes: usize,
    part: Part<'_>,
) -> Option<cranelift_module::FuncId> {
    let target = module.target_config();
    let ptr = target.pointer_type();
    let call_conv = target.default_call_conv;
    let n_args = match part {
        Part::Full => 4,
        Part::Prep(_) => 3,
        Part::Main(_) => 5,
    };
    let mut sig = module.make_signature();
    for _ in 0..n_args {
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
        let p = b.block_params(block).to_vec();
        let mut unary_sig = Signature::new(call_conv);
        unary_sig.params.push(AbiParam::new(types::F64));
        unary_sig.returns.push(AbiParam::new(types::F64));
        let mut binary_sig = unary_sig.clone();
        binary_sig.params.push(AbiParam::new(types::F64));
        let unary_ref = b.import_signature(unary_sig);
        let binary_ref = b.import_signature(binary_sig);
        let mut e = Emit {
            b,
            ptr,
            lanes,
            unary_ref,
            binary_ref,
            regs: vec![Vec::new(); n_regs],
        };
        match part {
            Part::Full => {
                for f in code {
                    e.flat(f, p[0], p[1], p[2], p[3]);
                }
            }
            // `prep(params, seeds, cache)`: no instruction it runs reads `u` or writes `du`.
            Part::Prep(plan) => {
                let (params, seeds, cache) = (p[0], p[1], p[2]);
                for (i, f) in code.iter().enumerate() {
                    if !plan.invariant[i] {
                        continue;
                    }
                    e.flat(f, params, params, seeds, cache);
                    if let (Some(slot), Some(d)) = (plan.slot[i], def_of(f)) {
                        let l = e.reg(d);
                        e.store(cache, slot as usize, &l);
                    }
                }
            }
            Part::Main(plan) => {
                let (u, params, seeds, du, cache) = (p[0], p[1], p[2], p[3], p[4]);
                for (i, f) in code.iter().enumerate() {
                    if !plan.invariant[i] {
                        e.flat(f, u, params, seeds, du);
                    } else if let (Some(slot), Some(d)) = (plan.slot[i], def_of(f)) {
                        let l = e.load(cache, slot as usize);
                        e.set(d, l);
                    }
                }
            }
        }
        e.b.ins().return_(&[]);
        e.b.seal_all_blocks();
        e.b.finalize(target);
    }
    module.define_function(id, &mut ctx).ok()?;
    module.clear_context(&mut ctx);
    Some(id)
}

/// The register an instruction writes.
fn def_of(f: &Flat) -> Option<u16> {
    let mut g = *f;
    let (d, _) = super::regs_of(&mut g);
    d.map(|r| *r)
}

/// A register's lanes: `[value, tangent_0, …]`.
type Lanes = Vec<Value>;

struct Emit<'a> {
    b: FunctionBuilder<'a>,
    ptr: types::Type,
    lanes: usize,
    unary_ref: cranelift_codegen::ir::SigRef,
    binary_ref: cranelift_codegen::ir::SigRef,
    regs: Vec<Lanes>,
}

impl Emit<'_> {
    fn k(&mut self, c: f64) -> Value {
        self.b.ins().f64const(c)
    }
    fn constant(&mut self, v: Value) -> Lanes {
        let mut l = vec![v];
        for _ in 1..self.lanes {
            let z = self.k(0.0);
            l.push(z);
        }
        l
    }
    fn load(&mut self, base: Value, index: usize) -> Lanes {
        (0..self.lanes)
            .map(|j| {
                let off = ((index * self.lanes + j) * 8) as i32;
                self.b
                    .ins()
                    .load(types::F64, MemFlagsData::trusted(), base, off)
            })
            .collect()
    }
    fn store(&mut self, base: Value, index: usize, l: &[Value]) {
        for (j, &v) in l.iter().enumerate() {
            let off = ((index * self.lanes + j) * 8) as i32;
            self.b.ins().store(MemFlagsData::trusted(), v, base, off);
        }
    }
    fn call1(&mut self, f: extern "C" fn(f64) -> f64, x: Value) -> Value {
        let callee = self.b.ins().iconst(self.ptr, f as usize as i64);
        let call = self.b.ins().call_indirect(self.unary_ref, callee, &[x]);
        self.b.inst_results(call)[0]
    }
    fn call2(&mut self, f: extern "C" fn(f64, f64) -> f64, x: Value, y: Value) -> Value {
        let callee = self.b.ins().iconst(self.ptr, f as usize as i64);
        let call = self.b.ins().call_indirect(self.binary_ref, callee, &[x, y]);
        self.b.inst_results(call)[0]
    }
    fn reg(&self, r: u16) -> Lanes {
        self.regs[r as usize].clone()
    }
    fn set(&mut self, d: u16, l: Lanes) {
        self.regs[d as usize] = l;
    }
    /// `cond ? a : b`, lane by lane.
    fn select(&mut self, cond: Value, a: &[Value], b: &[Value]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.b.ins().select(cond, x, y))
            .collect()
    }

    // ── The primitives, as `f64` / `DualV`'s untaped bodies compute them. ──
    fn add(&mut self, a: &[Value], b: &[Value]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.b.ins().fadd(x, y))
            .collect()
    }
    fn sub(&mut self, a: &[Value], b: &[Value]) -> Lanes {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| self.b.ins().fsub(x, y))
            .collect()
    }
    fn mul(&mut self, a: &[Value], b: &[Value]) -> Lanes {
        let v = self.b.ins().fmul(a[0], b[0]);
        let mut out = vec![v];
        for k in 1..self.lanes {
            let x = self.b.ins().fmul(a[k], b[0]);
            let y = self.b.ins().fmul(a[0], b[k]);
            out.push(self.b.ins().fadd(x, y));
        }
        out
    }
    fn div(&mut self, a: &[Value], b: &[Value]) -> Lanes {
        let q = self.b.ins().fdiv(a[0], b[0]);
        let mut out = vec![q];
        if self.lanes > 1 {
            let one = self.k(1.0);
            let ib = self.b.ins().fdiv(one, b[0]);
            for k in 1..self.lanes {
                let t = self.b.ins().fmul(q, b[k]);
                let d = self.b.ins().fsub(a[k], t);
                out.push(self.b.ins().fmul(d, ib));
            }
        }
        out
    }
    /// `x / y`, or the dual zero when `|y| < 1e-30` (`Op::Div`'s guard).
    fn guarded_div(&mut self, a: &[Value], b: &[Value]) -> Lanes {
        let q = self.div(a, b);
        let ay = self.b.ins().fabs(b[0]);
        let tiny = self.k(1e-30);
        let small = self.b.ins().fcmp(FloatCC::LessThan, ay, tiny);
        let z = self.k(0.0);
        let zero = self.constant(z);
        self.select(small, &zero, &q)
    }
    fn scalar(&mut self, op: Arith, a: &[Value], c: Value) -> Lanes {
        let mut out = a.to_vec();
        match op {
            Arith::Add => out[0] = self.b.ins().fadd(a[0], c),
            Arith::Sub => out[0] = self.b.ins().fsub(a[0], c),
            Arith::RSub => {
                // `constant(c) − self`: `0 − g`, which is `+0` where `−g` would be `−0`.
                out[0] = self.b.ins().fsub(c, a[0]);
                for x in out.iter_mut().skip(1) {
                    let z = self.b.ins().f64const(0.0);
                    *x = self.b.ins().fsub(z, *x);
                }
            }
            Arith::Mul => {
                for x in out.iter_mut() {
                    *x = self.b.ins().fmul(*x, c);
                }
            }
            Arith::Div => {
                out[0] = self.b.ins().fdiv(a[0], c);
                if self.lanes > 1 {
                    let one = self.b.ins().f64const(1.0);
                    let ib = self.b.ins().fdiv(one, c);
                    for x in out.iter_mut().skip(1) {
                        *x = self.b.ins().fmul(*x, ib);
                    }
                }
            }
        }
        out
    }
    /// `φ(x)` with `φ(v) = hv`, `φ′(v) = h1`: `g = h1·g`.
    fn unary(&mut self, a: &[Value], hv: Value, h1: Value) -> Lanes {
        let mut out = vec![hv];
        for &g in &a[1..] {
            out.push(self.b.ins().fmul(h1, g));
        }
        out
    }
    fn exp(&mut self, a: &[Value]) -> Lanes {
        let e = self.call1(shim_exp, a[0]);
        self.unary(a, e, e)
    }
    /// `DualV::ln`, unguarded: `h1 = 1/x`.
    fn ln(&mut self, a: &[Value]) -> Lanes {
        let l = self.call1(shim_ln, a[0]);
        if self.lanes == 1 {
            return vec![l];
        }
        let one = self.k(1.0);
        let h1 = self.b.ins().fdiv(one, a[0]);
        self.unary(a, l, h1)
    }
    fn neg(&mut self, a: &[Value]) -> Lanes {
        a.iter().map(|&x| self.b.ins().fneg(x)).collect()
    }
    fn pow(&mut self, a: &[Value], e: &[Value]) -> Lanes {
        if self.lanes == 1 {
            return vec![self.call2(shim_powf, a[0], e[0])];
        }
        // `DualV::pow`, constant exponent (`e` has no tangent): `a = 0` gives the constant
        // `powf(a, n)`; otherwise `h1 = n·powf(a, n − 1)`.
        let (av, n) = (a[0], e[0]);
        let pv = self.call2(shim_powf, av, n);
        let one = self.k(1.0);
        let nm1 = self.b.ins().fsub(n, one);
        let pm1 = self.call2(shim_powf, av, nm1);
        let h1 = self.b.ins().fmul(n, pm1);
        let lifted = self.unary(a, pv, h1);
        let z = self.k(0.0);
        let a_zero = self.b.ins().fcmp(FloatCC::Equal, av, z);
        let flat = self.constant(pv);
        let const_exp = self.select(a_zero, &flat, &lifted);
        // Otherwise `(e · a.ln()).exp()`.
        let l = self.ln(a);
        let el = self.mul(e, &l);
        let general = self.exp(&el);
        // `zero_tan`: every tangent `== 0.0` (so `−0.0` counts as zero).
        let mut all_zero: Option<Value> = None;
        for &g in &e[1..] {
            let z = self.k(0.0);
            let is = self.b.ins().fcmp(FloatCC::Equal, g, z);
            all_zero = Some(match all_zero {
                None => is,
                Some(acc) => self.b.ins().band(acc, is),
            });
        }
        let cond = all_zero.expect("lanes > 1");
        self.select(cond, &const_exp, &general)
    }

    fn flat(&mut self, f: &Flat, u: Value, params: Value, seeds: Value, du: Value) {
        match *f {
            Flat::LoadState { d, i } => {
                let l = self.load(u, i as usize);
                self.set(d, l);
            }
            Flat::LoadParam { d, slot } => {
                let l = self.load(params, slot as usize);
                self.set(d, l);
            }
            Flat::LoadSeed { d, k } => {
                let l = self.load(seeds, k as usize);
                self.set(d, l);
            }
            Flat::LoadK { d, c } => {
                let v = self.k(c);
                let l = self.constant(v);
                self.set(d, l);
            }
            Flat::Add { d, a, b } => {
                let l = self.add(&self.reg(a), &self.reg(b));
                self.set(d, l);
            }
            Flat::Sub { d, a, b } => {
                let l = self.sub(&self.reg(a), &self.reg(b));
                self.set(d, l);
            }
            // `rsub_in`: `self = rhs − self`.
            Flat::RSub { d, a, b } => {
                let l = self.sub(&self.reg(b), &self.reg(a));
                self.set(d, l);
            }
            Flat::Mul { d, a, b } => {
                let l = self.mul(&self.reg(a), &self.reg(b));
                self.set(d, l);
            }
            Flat::Div { d, a, b } => {
                let l = self.guarded_div(&self.reg(a), &self.reg(b));
                self.set(d, l);
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
                // `|c| < 1e-30` is the dual zero.
                let l = if op == Arith::Div && c.abs() < 1e-30 {
                    let z = self.k(0.0);
                    self.constant(z)
                } else {
                    let c = self.k(c);
                    self.scalar(op, &self.reg(a), c)
                };
                self.set(d, l);
            }
            Flat::Pow { d, a, e } => {
                let l = self.pow(&self.reg(a), &self.reg(e));
                self.set(d, l);
            }
            Flat::Unary { d, op, a } => {
                let x = self.reg(a);
                let l = match op {
                    Unary::Exp => self.exp(&x),
                    Unary::Ln => {
                        // `guard_floor(1e-30)`: `f64::max` (a NaN floors) on one lane,
                        // `DualV`'s `!(v ≥ lo)` → `constant(lo)` on a dual — the same test.
                        let lo = self.k(1e-30);
                        let keep = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, x[0], lo);
                        let floor = self.constant(lo);
                        let g = self.select(keep, &x, &floor);
                        self.ln(&g)
                    }
                    Unary::Abs if self.lanes == 1 => vec![self.b.ins().fabs(x[0])],
                    Unary::Abs => {
                        let z = self.k(0.0);
                        let nonneg = self.b.ins().fcmp(FloatCC::GreaterThanOrEqual, x[0], z);
                        let n = self.neg(&x);
                        self.select(nonneg, &x, &n)
                    }
                    Unary::Floor => {
                        let v = self.b.ins().floor(x[0]);
                        self.constant(v)
                    }
                    Unary::Ceil => {
                        let v = self.b.ins().ceil(x[0]);
                        self.constant(v)
                    }
                    Unary::Round => {
                        let v = self.call1(shim_round, x[0]);
                        self.constant(v)
                    }
                };
                self.set(d, l);
            }
            Flat::Deriv { state, a } => {
                let x = self.reg(a);
                for (j, &v) in x.iter().enumerate() {
                    let off = ((state as usize * self.lanes + j) * 8) as i32;
                    self.b.ins().store(MemFlagsData::trusted(), v, du, off);
                }
            }
        }
    }
}
