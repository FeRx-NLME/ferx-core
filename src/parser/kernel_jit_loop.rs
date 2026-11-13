//! ODE segments in a compiled recording (#2786).
//!
//! An adaptive integration cannot be recorded as straight-line code: how many steps it takes,
//! and how long each is, depend on the values. So a segment the recording meets becomes a
//! **loop call**. Its step is recorded once as code — each stepper's *start* (RK45's `k1`, a
//! Rosenbrock Jacobian) and *step* (the stages over `dt`), plus the right-hand side alone for
//! the driver's stiffness probes — over fresh variables for the state, the stepper's carried
//! state, `t` and `dt`, reading the outer recording's values as captures. At run time the
//! unchanged adaptive driver (`ode::solver`) runs at `f64` over steppers that run those bodies
//! at the plan's layout: every decision is the driver's, on values; every derivative lane rides
//! in the steppers' value shadows, one plain `f64` stepper per lane, whose interpolation and
//! accept logic is the generic code's own (it is linear in the lanes).
use super::*;
use crate::ode::solver::{self as sol, OdeMethod, OdeSolverOptions, StepBody, StepClock};
use crate::ode::SolPointG;
use std::cell::RefCell;
use std::rc::Rc;

/// A recorded body: a chunk program over `[vars…, captures…]` → `[outputs…, guard]`.
pub(crate) struct LoopBody {
    prog: Arc<ChunkProgram>,
    lits: Box<[f64]>,
    n_vars: usize,
    /// Each of the program's capture inputs, as an index into the call's capture list.
    caps: Vec<usize>,
    n_out: usize,
    /// Whether the body took a branch, so its last output is the guard.
    guarded: bool,
}

/// A method's step as bodies: `start` and `step`, `[u, carried, t, dt] → carried`, and
/// `accept`, `[carried] → [carried, u_new]` (the stepper's own accept: RK45's `k7 → k1`).
pub(crate) struct MethodBodies {
    method: OdeMethod,
    /// The carried state's length.
    c: usize,
    start: LoopBody,
    step: LoopBody,
    accept: LoopBody,
}

/// One ODE segment of a recording.
pub(crate) struct LoopSpec {
    n: usize,
    n_caps: usize,
    /// `[u, t] → f(u, t)`.
    rhs: LoopBody,
    /// Per method its bodies.
    methods: Vec<MethodBodies>,
    t_span: (f64, f64),
    /// The span's ends as captures, when they depend on the inputs (a lagged arrival, a
    /// window's end): the run reads them there instead of `t_span`.
    span_caps: [Option<usize>; 2],
    /// The last `saveat` time is the span's end: it moves with it.
    end_save: bool,
    saveat: Vec<f64>,
    interp_at: Vec<f64>,
    opts: OdeSolverOptions,
    /// The solver override `opts` were resolved under: a phase of the fit that arms another
    /// (a final inner loop or covariance step at tighter tolerances) must not run this call.
    ov: Option<Option<sol::OdeSolverOverride>>,
}

impl LoopSpec {
    /// Outputs: every state component at each `saveat` time, then at each `interp_at` time.
    pub(crate) fn n_out(&self) -> usize {
        self.n * (self.saveat.len() + self.interp_at.len())
    }
    pub(crate) fn n_in(&self) -> usize {
        self.n + self.n_caps
    }
}

/// The methods the driver may ask for under `opts`.
fn methods_for(opts: &OdeSolverOptions) -> Vec<OdeMethod> {
    let mut m = vec![OdeMethod::EXPLICIT_FALLBACK];
    match opts.method {
        OdeMethod::Auto => m.push(crate::ode::stiffness::stiff_method_for(opts)),
        OdeMethod::Linear | OdeMethod::Rk45 => {}
        other => m.push(other),
    }
    m.dedup();
    m
}

/// Record `f` over fresh variables (their record-time values `vals`) as a body, cut out of the
/// open recording. Outer values the body reads become captures, added to `caps`. Returns the
/// body and its outputs' record-time values.
fn record_body(
    vals: &[f64],
    caps: &mut Vec<u32>,
    masks: &[u64],
    dense: u64,
    var_mask: impl Fn(usize) -> u64,
    f: impl FnOnce(&[Tr]) -> Vec<Tr>,
) -> (LoopBody, Vec<f64>) {
    let (start_code, start_reg, outer_guard, outer_guarded) = rec(|r| {
        (
            r.code.len(),
            r.n_regs,
            r.guard.take(),
            std::mem::take(&mut r.guarded),
        )
    });
    let vars: Vec<Tr> = vals
        .iter()
        .map(|&v| Tr::mk(v, node(None, |d| Flat::LoadK { d, c: v }), true))
        .collect();
    let outs = f(&vars);
    let out_vals: Vec<f64> = outs.iter().map(|o| o.v).collect();
    let (region, guard, pool) = rec(|r| {
        let region = r.code.split_off(start_code);
        let g = r.guard.take();
        r.guard = outer_guard;
        r.guarded = outer_guarded;
        r.marks.retain(|&m| m <= start_code);
        // A body's registers exist only inside it: nothing outside may share them.
        r.cse.retain(|_, v| *v < start_reg);
        (region, g, r.pool.clone())
    });
    let var_regs: Vec<u32> = vars.iter().map(|v| v.r as u32).collect();
    let is_var = |r: u32| var_regs.contains(&r);
    let body: Vec<Flat> = region
        .into_iter()
        .filter(|f| !matches!(f, Flat::LoadK { d, .. } if is_var(*d)))
        .collect();
    // Captures: what the body reads that it does not define, past its variables.
    let mut ins: Vec<u32> = var_regs.clone();
    let mut seen: std::collections::HashSet<u32, super::FxBuild> =
        var_regs.iter().copied().collect();
    let mut cap_idx: Vec<usize> = Vec::new();
    for g in &body {
        for u in uses(g).into_iter().flatten() {
            if seen.insert(u) {
                ins.push(u);
            }
        }
        if let Some(d) = def(g) {
            seen.insert(d);
        }
    }
    let mut outs_r: Vec<u32> = outs.iter().map(|o| o.r as u32).collect();
    // An output the body did not compute (a captured or literal register) is an input too —
    // and so is a guard condition shared with one recorded before the body (hash-consing).
    for &o in outs_r.iter().chain(guard.map(|g| g as u32).iter()) {
        if !seen.contains(&o) && seen.insert(o) {
            ins.push(o);
        }
    }
    for &r in &ins[var_regs.len()..] {
        let at = match caps.iter().position(|&c| c == r as u32) {
            Some(i) => i,
            None => {
                caps.push(r as u32);
                caps.len() - 1
            }
        };
        cap_idx.push(at);
    }
    let n_out = outs_r.len();
    if let Some(g) = guard {
        outs_r.push(g as u32);
    }
    let spec: Vec<(u64, u8)> = ins
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            if i < var_regs.len() {
                (var_mask(i) & dense, u8::MAX)
            } else {
                (masks.get(r as usize).copied().unwrap_or(dense), u8::MAX)
            }
        })
        .collect();
    let mut cache = chunk_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (prog, lits) = build_chunk(&body, &ins, &outs_r, &pool, spec, &mut cache);
    (
        LoopBody {
            prog,
            lits: lits.into_boxed_slice(),
            n_vars: var_regs.len(),
            caps: cap_idx,
            n_out,
            guarded: guard.is_some(),
        },
        out_vals,
    )
}

/// Record one ODE segment (see the module docs): the bodies, a value-only run for the
/// record-time outputs, and the call in the outer recording. `None` — the recording fails — if
/// a body cannot be recorded or the run does not complete.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_segment(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]),
    u0: &[Tr],
    t_span: (f64, f64),
    params: &[Tr],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    auto: &mut sol::OdeAutoSwitchState,
) -> Option<(Vec<SolPointG<Tr>>, Vec<SolPointG<Tr>>)> {
    let n = u0.len();
    // The span the walk named, when its ends depend on the inputs.
    let span = <Tr as crate::sens::num::PkNum>::take_span()
        .filter(|(a, b)| (a.v, b.v) == t_span && (a.dep || b.dep));
    // A certified-linear segment the exact path declined is integrated as `auto`.
    let opts = if opts.method == OdeMethod::Linear {
        opts.for_integration()
    } else {
        *opts
    };
    let (t0, tf) = t_span;
    let dt0 = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);
    // Inside a sub-plan region every input lane may be live (its parameters are dense).
    let masks = rec(|r| {
        if r.sub_depth > 0 {
            vec![!0; r.n_regs as usize]
        } else {
            direction_masks(&r.code, r.n_regs as usize, &r.eta_regs, false)
        }
    });
    // Every lane the state can carry: what its inputs do.
    let dense = u0.iter().chain(params).fold(0u64, |m, x| {
        m | masks.get(x.r as usize).copied().unwrap_or(0)
    });
    let dense = if dense & !super::FREE == 0 { !0 } else { dense };
    let mut caps: Vec<u32> = Vec::new();
    let u0v: Vec<f64> = u0.iter().map(|x| x.v).collect();

    // The right-hand side alone: `[u, t] → du`.
    let mut vals = u0v.clone();
    vals.push(t0);
    let (rhs_body, _) = record_body(
        &vals,
        &mut caps,
        &masks,
        dense,
        |i| if i < n { !0 } else { 0 },
        |v| {
            let mut du = vec![Tr::k(0.0); n];
            rhs(&v[..n], params, v[n], &mut du);
            du
        },
    );
    let mut methods = Vec::new();
    for m in methods_for(&opts) {
        let mut sb = sol::make_step_body::<Tr>(n, m);
        sb.set_dense_required(!interp_at.is_empty());
        let mut probe = Vec::new();
        sb.carried_into(&mut probe);
        let c = probe.len();
        // The start, from a zero carried state; the step, from the start's.
        let body = |carried: &[f64], start: bool, caps: &mut Vec<u32>| {
            let mut vals = u0v.clone();
            vals.extend_from_slice(carried);
            vals.push(t0);
            vals.push(dt0);
            record_body(
                &vals,
                caps,
                &masks,
                dense,
                |i| if i < n + c { !0 } else { 0 },
                |v| {
                    let mut sb = sol::make_step_body::<Tr>(n, m);
                    sb.set_dense_required(!interp_at.is_empty());
                    sb.set_carried(&v[n..n + c]);
                    let clock = StepClock {
                        t: t0,
                        dt: dt0,
                        ts: v[n + c],
                        dts: v[n + c + 1],
                    };
                    if start {
                        sb.start(rhs, &v[..n], params, &clock);
                    } else {
                        sb.step(rhs, &v[..n], params, &clock);
                    }
                    let mut out = Vec::new();
                    sb.carried_into(&mut out);
                    out
                },
            )
        };
        let (start, after) = body(&vec![0.0; c], true, &mut caps);
        let (step, stepped) = body(&after, false, &mut caps);
        let (accept, _) = record_body(
            &stepped,
            &mut caps,
            &masks,
            dense,
            |_| !0,
            |v| {
                let mut sb = sol::make_step_body::<Tr>(n, m);
                sb.set_dense_required(!interp_at.is_empty());
                sb.set_carried(v);
                sb.on_accept();
                let mut out = Vec::new();
                sb.carried_into(&mut out);
                out.extend_from_slice(&sb.u_new()[..n]);
                out
            },
        );
        methods.push(MethodBodies {
            method: m,
            c,
            start,
            step,
            accept,
        });
    }
    if rec(|r| r.failed.is_some()) {
        return None;
    }
    // A moving end is one more capture: no body reads it, the driver does.
    let mut span_caps = [None, None];
    if let Some((a, b)) = span {
        for (k, x) in [a, b].into_iter().enumerate() {
            if x.dep {
                let at = caps.iter().position(|&c| c == x.r).unwrap_or_else(|| {
                    caps.push(x.r);
                    caps.len() - 1
                });
                span_caps[k] = Some(at);
            }
        }
    }
    let end_save = span_caps[1].is_some() && saveat.last() == Some(&t_span.1);
    let spec = Arc::new(LoopSpec {
        n,
        n_caps: caps.len(),
        rhs: rhs_body,
        methods,
        t_span,
        span_caps,
        end_save,
        saveat: saveat.to_vec(),
        interp_at: interp_at.to_vec(),
        opts,
        ov: sol::current_ode_override(),
    });
    // The record-time values: a run at the value-only layout.
    let vals = rec(|r| r.vals.clone());
    let mut inputs: Vec<f64> = u0v.clone();
    inputs.extend(
        caps.iter()
            .map(|&c| vals.get(c as usize).copied().unwrap_or(f64::NAN)),
    );
    let mut out = vec![0.0; spec.n_out()];
    let ran = LoopKernels::resolve(&spec, Layout::F64, false)
        .is_some_and(|k| run_loop(&spec, &k, 1, &inputs, &mut out, auto, None));
    if !ran {
        if super::debug() {
            eprintln!(
                "kjit: record-time run failed: caps {caps:?} inputs {:?} span {:?} span_caps {:?}",
                &inputs[n..],
                t_span,
                spec.span_caps
            );
            rec(|r| {
                for &c in &caps {
                    let d = r.code.iter().find(|f| super::def(f) == Some(c as u32));
                    eprintln!(
                        "kjit:   cap {c}: vals {:?} def {d:?}",
                        r.vals.get(c as usize)
                    );
                }
            });
        }
        fail("an ODE segment's record-time run failed".into());
        return None;
    }
    // The call.
    let id = rec(|r| {
        r.ode = true;
        r.loops.push(super::CallKind::Ode(spec.clone()));
        r.loops.len() - 1
    }) as u32;
    for x in u0 {
        let a = x.r as u32;
        rec(|r| r.code.push(Flat::LoopArg { a }));
    }
    for &c in &caps {
        let a = c as u32;
        rec(|r| r.code.push(Flat::LoopArg { a }));
    }
    rec(|r| r.code.push(Flat::Loop { id }));
    let outs: Vec<Tr> = out
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            let r = node(None, |d| Flat::LoopOut { d, k: k as u32 });
            Tr::mk(v, r, true)
        })
        .collect();
    let at = |times: &[f64], base: usize| -> Vec<SolPointG<Tr>> {
        times
            .iter()
            .enumerate()
            .map(|(j, &t)| SolPointG {
                t,
                u: outs[base + j * n..base + (j + 1) * n].to_vec(),
            })
            .collect()
    };
    Some((at(saveat, 0), at(interp_at, saveat.len() * n)))
}

/// A loop call's kernels at one layout: the right-hand side at the value layout (the driver's
/// own probes), and each method's start and step. Resolved once per plan and layout.
pub(crate) struct LoopKernels {
    rhs_f64: Arc<Tiered>,
    /// Per method: start, step, accept.
    methods: Vec<[Arc<Tiered>; 3]>,
    sl: Slots,
    /// Per method, per current carried region: the start's, step's and accept's index tables.
    tables: Vec<[[Vec<u32>; 3]; 2]>,
}

impl LoopKernels {
    /// The kernels at `layout`, the bodies' reverse passes when `rev`.
    pub(crate) fn resolve(spec: &LoopSpec, layout: Layout, rev: bool) -> Option<Arc<LoopKernels>> {
        let get = |b: &LoopBody| b.prog.get(layout, rev);
        let rhs_f64 = spec.rhs.prog.get(Layout::F64, false)?;
        let methods = spec
            .methods
            .iter()
            .map(|m| Some([get(&m.start)?, get(&m.step)?, get(&m.accept)?]))
            .collect::<Option<Vec<_>>>()?;
        let sl = Slots::of(spec);
        let tables = spec.methods.iter().map(|mb| tables(spec, mb, sl)).collect();
        Some(Arc::new(LoopKernels {
            rhs_f64,
            methods,
            sl,
            tables,
        }))
    }
}

/// One event of a loop call's run, in order: what its reverse replays.
#[derive(Clone)]
pub(crate) enum Entry {
    /// A solve (re)starts from the initial state.
    Begin,
    /// A stepper for method `m` starts, its carried state fresh.
    NewStepper,
    /// Method `m`'s body `kind` (0 start, 1 step, 2 accept) ran on these input lanes.
    Body { m: usize, kind: u8, vars: Vec<f64> },
    /// The current state became output point `idx` of the hard (`saveat`) or soft list.
    Point {
        soft: bool,
        idx: usize,
        diverged: bool,
    },
    /// An in-step read-out: no reverse yet.
    Interp,
}

/// Where a loop run's lanes live: one buffer of slots,
/// `[u (n) | carried A (cm) | carried B (cm) | t | dt | captures | out (cm + n) | guard]`, `cm`
/// the largest carried state of the call's methods. Every body reads and writes there in place
/// through an index table; the two carried regions take turns as a body's input and output.
#[derive(Clone, Copy)]
struct Slots {
    u: usize,
    carried: [usize; 2],
    t: usize,
    caps: usize,
    guard: usize,
    total: usize,
}

impl Slots {
    fn of(spec: &LoopSpec) -> Slots {
        let (n, nc) = (spec.n, spec.n_caps);
        let cm = spec.methods.iter().map(|m| m.c).max().unwrap_or(0);
        let t = n + 2 * cm;
        let caps = t + 2;
        let guard = caps + nc;
        Slots {
            u: 0,
            carried: [n, n + cm],
            t,
            caps,
            guard,
            total: guard + 1,
        }
    }
}

/// Body `b`'s index table over `vars` and `outs` slots (and the captures', the guard's).
fn table(b: &LoopBody, sl: Slots, vars: &[usize], outs: &[usize]) -> Vec<u32> {
    let mut t: Vec<u32> = vars.iter().map(|&v| v as u32).collect();
    t.extend(b.caps.iter().map(|&c| (sl.caps + c) as u32));
    t.extend(outs.iter().map(|&o| o as u32));
    if b.guarded {
        t.push(sl.guard as u32);
    }
    t
}

/// A method's index tables, per current carried region: start, step, accept.
fn tables(spec: &LoopSpec, mb: &MethodBodies, sl: Slots) -> [[Vec<u32>; 3]; 2] {
    let (n, c) = (spec.n, mb.c);
    let us: Vec<usize> = (0..n).map(|j| sl.u + j).collect();
    let reg = |r: usize| -> Vec<usize> { (0..c).map(|j| sl.carried[r] + j).collect() };
    [0usize, 1].map(|r| {
        let mut vars = us.clone();
        vars.extend(reg(r));
        vars.push(sl.t);
        vars.push(sl.t + 1);
        let mut acc_outs = reg(1 - r);
        acc_outs.extend(us.iter().copied());
        [
            table(&mb.start, sl, &vars, &reg(1 - r)),
            table(&mb.step, sl, &vars, &reg(1 - r)),
            table(&mb.accept, sl, &reg(r), &acc_outs),
        ]
    })
}

thread_local! {
    /// Value shadows and slot buffers of finished runs, for the next run on this thread.
    static SHADOWS: RefCell<Vec<(OdeMethod, usize, Box<dyn StepBody<f64>>)>> =
        const { RefCell::new(Vec::new()) };
    static BUFS: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// A value shadow for `method` over `n` states: a pooled one reset, or a new one.
fn take_shadow(n: usize, method: OdeMethod, dense: bool) -> Box<dyn StepBody<f64>> {
    let pooled = SHADOWS.with(|p| {
        let mut p = p.borrow_mut();
        p.iter()
            .rposition(|(m, k, _)| *m == method && *k == n)
            .map(|i| p.swap_remove(i).2)
    });
    let mut b = match pooled {
        Some(mut b) => {
            b.reset_start();
            b
        }
        None => sol::make_step_body::<f64>(n, method),
    };
    b.set_dense_required(dense);
    b
}

fn give_shadow(n: usize, method: OdeMethod, b: Box<dyn StepBody<f64>>) {
    SHADOWS.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 16 {
            p.push((method, n, b));
        }
    });
}

/// A loop call running over one layout's kernels.
struct LoopRun {
    spec: Arc<LoopSpec>,
    kern: Arc<LoopKernels>,
    l: usize,
    sl: Slots,
    /// The slots' lanes.
    buf: RefCell<Vec<f64>>,
    /// `[u0 (n), captures]`, `l` lanes each.
    inputs: Vec<f64>,
    st: RefCell<RunState>,
}

#[derive(Default)]
struct RunState {
    hard: Vec<Vec<f64>>,
    soft: Vec<Vec<f64>>,
    pend: Vec<f64>,
    lane: Vec<f64>,
    ok: bool,
    /// The run's events, when its reverse will be asked for.
    log: Option<Vec<Entry>>,
    /// Per-lane shadows of the active stepper, for in-step read-outs (`interp_at`) only.
    active: Option<Rc<RefCell<Vec<Box<dyn StepBody<f64>>>>>>,
}

impl LoopRun {
    fn new(
        spec: &Arc<LoopSpec>,
        kern: &Arc<LoopKernels>,
        l: usize,
        inputs: &[f64],
        log: bool,
    ) -> Self {
        let (n, nc) = (spec.n, spec.n_caps);
        let sl = kern.sl;
        let caps = sl.caps;
        let mut buf = BUFS.with(|b| b.borrow_mut().pop()).unwrap_or_default();
        buf.clear();
        buf.resize(sl.total * l, 0.0);
        buf[caps * l..(caps + nc) * l].copy_from_slice(&inputs[n * l..(n + nc) * l]);
        LoopRun {
            spec: spec.clone(),
            kern: kern.clone(),
            l,
            sl,
            buf: RefCell::new(buf),
            inputs: inputs.to_vec(),
            st: RefCell::new(RunState {
                ok: true,
                log: log.then(Vec::new),
                ..Default::default()
            }),
        }
    }

    /// Run `b`'s kernel `k` in place through `table`; `false` when its guard fails. Logs the
    /// body's inputs when the run keeps a log.
    fn run(&self, b: &LoopBody, k: &Tiered, table: &[u32], m: usize, kind: u8) -> bool {
        let l = self.l;
        let mut buf = self.buf.borrow_mut();
        {
            let mut st = self.st.borrow_mut();
            if let Some(log) = st.log.as_mut() {
                let mut vars = Vec::with_capacity(b.n_vars * l);
                for &s in &table[..b.n_vars] {
                    vars.extend_from_slice(&buf[s as usize * l..(s as usize + 1) * l]);
                }
                log.push(Entry::Body { m, kind, vars });
            }
        }
        // SAFETY: every slot in `table` is below `sl.total`, whose `l` lanes each `buf` holds,
        // laid out as the kernel's layout reads them.
        unsafe {
            k.call(
                buf.as_mut_ptr(),
                std::ptr::null_mut(),
                table.as_ptr(),
                b.lits.as_ptr(),
            )
        };
        let ok = !b.guarded || buf[self.sl.guard * l] == 1.0;
        if !ok && super::debug() {
            let vars: Vec<f64> = table[..b.n_vars]
                .iter()
                .map(|&s| buf[s as usize * l])
                .collect();
            eprintln!("kjit: loop body guard failed: method {m} kind {kind} vars {vars:?}");
        }
        ok
    }

    /// The current state's lanes.
    fn u(&self) -> Vec<f64> {
        let (n, l) = (self.spec.n, self.l);
        self.buf.borrow()[self.sl.u * l..(self.sl.u + n) * l].to_vec()
    }

    fn fail(&self) {
        self.st.borrow_mut().ok = false;
    }
}

impl Drop for LoopRun {
    fn drop(&mut self) {
        let buf = std::mem::take(self.buf.get_mut());
        BUFS.with(|b| {
            let mut b = b.borrow_mut();
            if b.len() < 8 {
                b.push(buf);
            }
        });
    }
}

/// [`sol::LaneHook`]: points take the current lanes.
struct Hook(Rc<LoopRun>);

impl sol::LaneHook for Hook {
    fn stepper(&self, n: usize, method: OdeMethod) -> Box<dyn sol::Stepper<f64>> {
        let run = self.0.clone();
        let m = match run.spec.methods.iter().position(|x| x.method == method) {
            Some(m) => m,
            None => {
                run.fail();
                0
            }
        };
        let mb = &run.spec.methods[m];
        let (meth, c) = (mb.method, mb.c);
        let dense = !run.spec.interp_at.is_empty();
        let mk = || take_shadow(n, meth, dense);
        // Per-lane shadows only to read the continuous extension inside steps.
        let lanes =
            dense.then(|| Rc::new(RefCell::new((0..run.l).map(|_| mk()).collect::<Vec<_>>())));
        {
            let mut st = run.st.borrow_mut();
            st.active = lanes.clone();
            if let Some(log) = st.log.as_mut() {
                log.push(Entry::NewStepper);
            }
        }
        // A fresh stepper's carried state is zero.
        {
            let sl = run.sl;
            let l = run.l;
            let mut buf = run.buf.borrow_mut();
            buf[sl.carried[0] * l..(sl.carried[1] + c) * l].fill(0.0);
        }
        Box::new(Lanes {
            run,
            m,
            method: meth,
            shadow: Some(mk()),
            lanes,
            u_new: vec![0.0; n],
            cur: 0,
            lane: Vec::with_capacity(c),
        })
    }
    fn begin(&self) {
        let run = &self.0;
        let (n, l) = (run.spec.n, run.l);
        run.buf.borrow_mut()[run.sl.u * l..(run.sl.u + n) * l]
            .copy_from_slice(&run.inputs[..n * l]);
        let mut st = run.st.borrow_mut();
        st.hard.clear();
        st.soft.clear();
        if let Some(log) = st.log.as_mut() {
            log.clear();
            log.push(Entry::Begin);
        }
    }
    fn point(&self, soft: bool) {
        let u = self.0.u();
        let mut st = self.0.st.borrow_mut();
        let idx = if soft { st.soft.len() } else { st.hard.len() };
        if soft {
            st.soft.push(u);
        } else {
            st.hard.push(u);
        }
        if let Some(log) = st.log.as_mut() {
            log.push(Entry::Point {
                soft,
                idx,
                diverged: false,
            });
        }
    }
    fn interp(&self, theta: f64, dt: f64, i: usize) {
        let run = &self.0;
        let (n, l) = (run.spec.n, run.l);
        let u = run.u();
        let mut st = run.st.borrow_mut();
        let st = &mut *st;
        if let Some(log) = st.log.as_mut() {
            log.push(Entry::Interp);
        }
        if st.pend.len() != n * l {
            st.pend = vec![0.0; n * l];
        }
        let Some(sh) = st.active.clone() else { return };
        let sh = sh.borrow();
        for lane in 0..l {
            st.lane.clear();
            st.lane.extend((0..n).map(|j| u[j * l + lane]));
            st.pend[i * l + lane] = sh[lane].interpolate_component(theta, &st.lane, dt, i);
        }
    }
    fn interp_done(&self) {
        let mut st = self.0.st.borrow_mut();
        let p = st.pend.clone();
        st.soft.push(p);
    }
    fn tail(&self, soft: bool, diverged: bool) {
        let u = self.0.u();
        let mut st = self.0.st.borrow_mut();
        // `v · NaN` in every lane, as the dual tail is formed.
        let u: Vec<f64> = if diverged {
            u.iter().map(|&v| v * f64::NAN).collect()
        } else {
            u
        };
        let idx = if soft { st.soft.len() } else { st.hard.len() };
        if soft {
            st.soft.push(u);
        } else {
            st.hard.push(u);
        }
        if let Some(log) = st.log.as_mut() {
            log.push(Entry::Point {
                soft,
                idx,
                diverged,
            });
        }
    }
}

/// A stepper over a loop call's lanes: every lane runs through the recorded bodies in the run's
/// buffer; one value shadow holds the value lane for the driver's decisions (the error norm,
/// the stiffness indicator, its flags), as the generic stepper computes them.
struct Lanes {
    run: Rc<LoopRun>,
    m: usize,
    method: OdeMethod,
    /// Always `Some` until drop returns it to the pool.
    shadow: Option<Box<dyn StepBody<f64>>>,
    /// One shadow per lane, for in-step read-outs (`interp_at`) only.
    lanes: Option<Rc<RefCell<Vec<Box<dyn StepBody<f64>>>>>>,
    u_new: Vec<f64>,
    /// Which carried region holds the current state.
    cur: usize,
    lane: Vec<f64>,
}

impl Drop for Lanes {
    fn drop(&mut self) {
        if let Some(b) = self.shadow.take() {
            give_shadow(self.run.spec.n, self.method, b);
        }
    }
}

impl Lanes {
    /// The shadows from the current carried region.
    fn load_shadows(&mut self) {
        let (l, c) = (self.run.l, self.run.spec.methods[self.m].c);
        let base = self.run.sl.carried[self.cur];
        let buf = self.run.buf.borrow();
        self.lane.clear();
        self.lane.extend((0..c).map(|j| buf[(base + j) * l]));
        self.shadow.as_mut().unwrap().set_carried(&self.lane);
        if let Some(sh) = &self.lanes {
            for (lane, s) in sh.borrow_mut().iter_mut().enumerate() {
                self.lane.clear();
                self.lane.extend((0..c).map(|j| buf[(base + j) * l + lane]));
                s.set_carried(&self.lane);
            }
        }
    }
    /// Run body `kind` (0 start, 1 step, 2 accept); the carried regions swap.
    fn body(&mut self, kind: usize) {
        let run = self.run.clone();
        let mb = &run.spec.methods[self.m];
        let b = [&mb.start, &mb.step, &mb.accept][kind];
        let k = &run.kern.methods[self.m][kind];
        if !run.run(
            b,
            k,
            &run.kern.tables[self.m][self.cur][kind],
            self.m,
            kind as u8,
        ) {
            run.fail();
        }
        self.cur = 1 - self.cur;
    }
}

impl sol::Stepper<f64> for Lanes {
    fn attempt(
        &mut self,
        _rhs: &sol::RhsG<'_, f64>,
        u: &[f64],
        _params: &[f64],
        t: f64,
        dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        {
            let (l, ts) = (self.run.l, self.run.sl.t);
            let mut buf = self.run.buf.borrow_mut();
            buf[ts * l] = t;
            buf[(ts + 1) * l] = dt;
        }
        if self.shadow.as_mut().unwrap().needs_start() {
            self.body(0);
        }
        self.body(1);
        self.load_shadows();
        self.u_new
            .copy_from_slice(&self.shadow.as_ref().unwrap().u_new()[..self.run.spec.n]);
        self.shadow.as_mut().unwrap().err(u, dt, opts)
    }
    fn u_new(&self) -> &[f64] {
        &self.u_new
    }
    fn interpolate_component(&self, theta: f64, u_old: &[f64], dt: f64, i: usize) -> f64 {
        self.shadow
            .as_ref()
            .unwrap()
            .interpolate_component(theta, u_old, dt, i)
    }
    fn on_accept(&mut self) {
        self.body(2);
        self.shadow.as_mut().unwrap().on_accept();
        if let Some(sh) = &self.lanes {
            for s in sh.borrow_mut().iter_mut() {
                s.on_accept();
            }
        }
    }
    fn on_reject(&mut self) {}
    fn err_exp(&self) -> f64 {
        self.shadow.as_ref().unwrap().err_exp()
    }
    fn set_dense_required(&mut self, yes: bool) {
        self.shadow.as_mut().unwrap().set_dense_required(yes);
        if let Some(sh) = &self.lanes {
            for s in sh.borrow_mut().iter_mut() {
                s.set_dense_required(yes);
            }
        }
    }
    fn attempt_usable(&self) -> bool {
        self.shadow.as_ref().unwrap().attempt_usable()
    }
    fn stiffness_indicator(&self, dt: f64) -> Option<f64> {
        self.shadow.as_ref().unwrap().stiffness_indicator(dt)
    }
}

/// Run a loop call over `kern` (at `kern`'s layout, `l` lanes): `inputs` its inputs' lanes
/// (`[u0, captures]`), `out` its outputs' (every state component at each `saveat`, then each
/// `interp_at` time). `false` when another solver override is in force than the call was
/// recorded under, or a body's guard failed on some step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop(
    spec: &Arc<LoopSpec>,
    kern: &Arc<LoopKernels>,
    l: usize,
    inputs: &[f64],
    out: &mut [f64],
    auto: &mut sol::OdeAutoSwitchState,
    log: Option<&mut Vec<Entry>>,
) -> bool {
    if sol::current_ode_override() != spec.ov {
        return false;
    }
    let run = Rc::new(LoopRun::new(spec, kern, l, inputs, log.is_some()));
    let n = spec.n;
    // The driver's own right-hand side (its stiffness probes): the body at the value layout,
    // the captures at their value lanes.
    let rhs_run = run.clone();
    let rb = &spec.rhs;
    let n_in = rb.n_vars + rb.caps.len();
    let n_slots = n_in + rb.n_out + rb.guarded as usize;
    let rhs_buf = RefCell::new(vec![0.0; n_slots]);
    let rhs_idx: Vec<u32> = (0..n_slots as u32).collect();
    {
        let mut b = rhs_buf.borrow_mut();
        for (j, &c) in rb.caps.iter().enumerate() {
            b[rb.n_vars + j] = inputs[(n + c) * l];
        }
    }
    let rhs_f = move |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        let r = &rhs_run;
        let b = &r.spec.rhs;
        let mut buf = rhs_buf.borrow_mut();
        buf[..n].copy_from_slice(&u[..n]);
        buf[n] = t;
        // SAFETY: `rhs_idx` names `n_slots` single-lane slots of `buf`.
        unsafe {
            r.kern.rhs_f64.call(
                buf.as_mut_ptr(),
                std::ptr::null_mut(),
                rhs_idx.as_ptr(),
                b.lits.as_ptr(),
            )
        };
        if b.guarded && buf[n_in + b.n_out] != 1.0 {
            if super::debug() {
                eprintln!(
                    "kjit: loop rhs guard failed at t = {t:e}, u = {:?}, buf = {:?}, caps = {:?}",
                    &u[..n],
                    &buf[..],
                    b.caps
                );
            }
            r.fail();
        }
        du[..n].copy_from_slice(&buf[n_in..n_in + n]);
    };
    let u0: Vec<f64> = (0..n).map(|j| inputs[j * l]).collect();
    // The span, from the inputs where it moves with them.
    let end = |k: usize, rec: f64| spec.span_caps[k].map_or(rec, |c| inputs[(n + c) * l]);
    let t_span = (end(0, spec.t_span.0), end(1, spec.t_span.1));
    let moved_saveat;
    let saveat: &[f64] = if spec.end_save {
        let mut v = spec.saveat.clone();
        *v.last_mut().expect("the end") = t_span.1;
        moved_saveat = v;
        &moved_saveat
    } else {
        &spec.saveat
    };
    let hook: Rc<dyn sol::LaneHook> = Rc::new(Hook(run.clone()));
    let (hard, soft) = sol::with_lane_hook(hook, || {
        sol::integrate_nonlinear_f64(
            &rhs_f,
            &u0,
            t_span,
            saveat,
            &spec.interp_at,
            &spec.opts,
            auto,
        )
    });
    let mut st = run.st.borrow_mut();
    if !st.ok || st.hard.len() != hard.len() || st.soft.len() != soft.len() {
        if super::debug() {
            eprintln!(
                "kjit: loop run failed: ok {} hard {}/{} soft {}/{} span {:?} saveat {:?}",
                st.ok,
                st.hard.len(),
                hard.len(),
                st.soft.len(),
                soft.len(),
                t_span,
                saveat
            );
        }
        return false;
    }
    if let (Some(dst), Some(src)) = (log, st.log.take()) {
        *dst = src;
    }
    let mut at = 0;
    for p in st.hard.iter().chain(&st.soft) {
        out[at..at + n * l].copy_from_slice(p);
        at += n * l;
    }
    true
}

/// The reverse of a loop call over `rev`'s reverse kernels at `l` lanes: from the adjoints of its
/// outputs `out_adj`, add those of its inputs into `in_adj` (both laid out as `run_loop`'s,
/// one block per seed of `seeds`). `log` is the forward run's. A discrete adjoint: every lane
/// operation of the run was a recorded body, so the log replayed backward through the bodies'
/// reverse passes is exact; each body's recomputed forward is shared by the seeds. `false`
/// for a run with in-step read-outs (no reverse yet).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop_rev(
    spec: &LoopSpec,
    rev: &LoopKernels,
    l: usize,
    inputs: &[f64],
    log: &[Entry],
    seeds: usize,
    out_adj: &[f64],
    in_adj: &mut [f64],
) -> bool {
    let n = spec.n;
    let n_hard = spec.saveat.len();
    let (out_w, in_w) = (out_adj.len() / seeds, in_adj.len() / seeds);
    let nl = n * l;
    let mut ubar = vec![0.0; seeds * nl];
    let mut cbar: Vec<f64> = Vec::new();
    let mut capbar = vec![0.0; seeds * spec.n_caps * l];
    let cap_w = spec.n_caps * l;
    let mut base: Vec<f64> = Vec::new();
    let mut adj: Vec<f64> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    for e in log.iter().rev() {
        match e {
            Entry::Point {
                soft,
                idx: k,
                diverged,
            } => {
                let at = (if *soft { n_hard + k } else { *k }) * nl;
                for sd in 0..seeds {
                    let src = &out_adj[sd * out_w + at..sd * out_w + at + nl];
                    for (x, &g) in ubar[sd * nl..(sd + 1) * nl].iter_mut().zip(src) {
                        *x += if *diverged { g * f64::NAN } else { g };
                    }
                }
            }
            Entry::Interp => return false,
            Entry::NewStepper => cbar.clear(),
            Entry::Begin => {
                for sd in 0..seeds {
                    for (x, &g) in in_adj[sd * in_w..sd * in_w + nl]
                        .iter_mut()
                        .zip(&ubar[sd * nl..(sd + 1) * nl])
                    {
                        *x += g;
                    }
                }
                break;
            }
            Entry::Body { m, kind, vars } => {
                let mb = &spec.methods[*m];
                let c = mb.c;
                let cl = c * l;
                if cbar.len() != seeds * cl {
                    cbar = vec![0.0; seeds * cl];
                }
                let (b, k) = match kind {
                    0 => (&mb.start, &rev.methods[*m][0]),
                    1 => (&mb.step, &rev.methods[*m][1]),
                    _ => (&mb.accept, &rev.methods[*m][2]),
                };
                let n_in = b.n_vars + b.caps.len();
                let n_slots = n_in + b.n_out + b.guarded as usize;
                let w = n_slots * l;
                base.clear();
                base.resize(w, 0.0);
                base[..b.n_vars * l].copy_from_slice(&vars[..b.n_vars * l]);
                for (j, &cp) in b.caps.iter().enumerate() {
                    base[(b.n_vars + j) * l..(b.n_vars + j + 1) * l]
                        .copy_from_slice(&inputs[(n + cp) * l..(n + cp + 1) * l]);
                }
                adj.clear();
                adj.resize(seeds * w, 0.0);
                // The outputs' adjoints: the carried state's, and an accept's new state's.
                for sd in 0..seeds {
                    let a = &mut adj[sd * w..(sd + 1) * w];
                    a[n_in * l..n_in * l + cl].copy_from_slice(&cbar[sd * cl..(sd + 1) * cl]);
                    if *kind == 2 {
                        a[n_in * l + cl..n_in * l + cl + nl]
                            .copy_from_slice(&ubar[sd * nl..(sd + 1) * nl]);
                    }
                }
                idx.clear();
                idx.extend(0..n_slots as u32);
                // SAFETY: as `LoopRun::call`; each seed's `adj` block is laid out like `base`.
                unsafe {
                    k.call_seeds(
                        base.as_mut_ptr(),
                        adj.as_mut_ptr(),
                        w,
                        seeds,
                        idx.as_ptr(),
                        b.lits.as_ptr(),
                    )
                };
                for sd in 0..seeds {
                    let a = &adj[sd * w..(sd + 1) * w];
                    let cap = &mut capbar[sd * cap_w..(sd + 1) * cap_w];
                    for (j, &cp) in b.caps.iter().enumerate() {
                        for x in 0..l {
                            cap[cp * l + x] += a[(b.n_vars + j) * l + x];
                        }
                    }
                    let (ub, cb) = (
                        &mut ubar[sd * nl..(sd + 1) * nl],
                        &mut cbar[sd * cl..(sd + 1) * cl],
                    );
                    if *kind == 2 {
                        // An accept reads the carried state only; the state it replaced is
                        // read by the bodies before it.
                        cb.copy_from_slice(&a[..cl]);
                        ub.fill(0.0);
                    } else {
                        for (x, &g) in ub.iter_mut().zip(&a[..nl]) {
                            *x += g;
                        }
                        cb.copy_from_slice(&a[nl..nl + cl]);
                    }
                }
            }
        }
    }
    for sd in 0..seeds {
        for (x, &g) in in_adj[sd * in_w + nl..(sd + 1) * in_w]
            .iter_mut()
            .zip(&capbar[sd * cap_w..(sd + 1) * cap_w])
        {
            *x += g;
        }
    }
    true
}
