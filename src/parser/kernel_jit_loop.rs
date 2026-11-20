//! ODE segments in a compiled recording (#2786).
//!
//! An adaptive integration cannot be recorded as straight-line code: how many steps it takes,
//! and how long each is, depend on the values. So a segment the recording meets becomes a
//! **loop call**. Its step is recorded once as code — each stepper's *start* (RK45's `k1`, a
//! Rosenbrock Jacobian) and *step* (the stages over `dt`), plus the right-hand side alone for
//! the driver's stiffness probes — over fresh variables for the state, the stepper's carried
//! state, `t` and `dt`, reading the outer recording's values as captures. At run time the
//! unchanged adaptive driver (`ode::solver`) runs at `f64` over steppers that run those bodies
//! at the plan's layout: every decision is the driver's, on the value lane of the carried state,
//! read in place by the definitions the `f64` steppers' own reads call
//! ([`sol::CarriedLayout`]); every derivative lane rides beside it in the same slots, and a
//! continuous-extension read-out is taken per lane by the same definitions (it is linear in the
//! lanes).
//!
//! A segment of a model with moving switches (#2649) records more bodies: the switch functions
//! alone, `[u, t] → g`, for the driver's event location on values; per method and switch side a
//! step with that switch's arm forced, which the steps locating a crossing take; and per switch
//! and side the **jump** at a crossing, `[u, t] → u⁺`: the state carried through the crossing to
//! the other arm by flows of a value-zero duration whose derivatives are those of the crossing
//! time, so every lane, of every order the layout carries, jumps exactly ([`jump_tr`]). The
//! values do not move.
use super::*;
use crate::ode::solver::{self as sol, OdeMethod, OdeSolverOptions, StepClock};
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
    /// Per variable whether an instruction of the body reads it (#3078). A variable it does
    /// not read cannot move what it computes, so a log need not keep it; one it only passes
    /// through to an output ([`Self::copy_of`]) moves that output alone, its reverse carrying
    /// the output's adjoint back unchanged.
    reads: Box<[bool]>,
    /// Per output the variable it is, when the body passes that variable through unchanged.
    copy_of: Box<[Option<u32>]>,
}

/// A method's step as bodies: `start` and `step`, `[u, carried, t, dt] → carried`, and
/// `accept`, `[carried] → [carried, u_new]` (the stepper's own accept: RK45's `k7 → k1`).
pub(crate) struct MethodBodies {
    method: OdeMethod,
    /// The carried state's length.
    c: usize,
    /// Where the driver's reads sit in the carried state.
    cl: sol::CarriedLayout,
    start: LoopBody,
    step: LoopBody,
    accept: LoopBody,
    /// The step with switch `k`'s arm forced to its side `pos`, at `2k + pos` (#2649): the
    /// steps that locate a crossing integrate the field on the arm before it throughout.
    forced: Vec<LoopBody>,
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
    ov: Option<sol::OdeSolverOverride>,
    /// The moving switches' functions, `[u, t] → g`, run at the value layout (#2649).
    sw_g: Option<LoopBody>,
    /// Per switch `k` and side, at `2k + pos`, the jump `[u, t] → u⁺` ([`jump_tr`]).
    jumps: Vec<LoopBody>,
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

/// Chord steps [`jump_tr`] takes for the crossing time: its error is then `O(|Δp|⁵)`, so its
/// derivatives are exact through fourth order, the highest a recording reads, as the flows'
/// are (`rk4_dual`).
const JUMP_CHORD_STEPS: usize = 4;

/// `rhs`'s field at `(u, t)` with switch `k` of `n_sw` forced to the arm of its side `pos`
/// (#2649), through the switch program (`ode_switch::instrument`); returns the switch's
/// function there.
#[allow(clippy::too_many_arguments)]
fn forced_field(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]), u: &[Tr], params: &[Tr], t: Tr, k: usize,
    n_sw: usize, pos: bool, du: &mut [Tr],
) -> Tr {
    let n = u.len();
    let bias = if pos { sol::SWITCH_FORCE } else { -sol::SWITCH_FORCE };
    let mut ue = u.to_vec();
    ue.extend((0..n_sw).map(|j| Tr::k(if j == k { bias } else { 0.0 })));
    let mut d = vec![Tr::k(0.0); n + n_sw];
    rhs(&ue, params, t, &mut d);
    du[..n].copy_from_slice(&d[..n]);
    d[n + k]
}

/// The state just past moving switch `k` of `n_sw` (#2649), crossed at `(u, t)` from the side
/// where its function `g` is positive when `pos`, over `rhs`'s switch program (`u` longer
/// than the states, `ode_switch::instrument`).
///
/// With `f⁻`/`f⁺` the field on the arm before/after, the crossing time `t + δ(p)` solves
/// `G(δ) = g(φ⁻_δ(u), t + δ) = 0`, `φ⁻` the flow of `f⁻`. Its value is `0` (the driver ended the
/// step there); its derivatives are [`JUMP_CHORD_STEPS`] chord steps from `0` with the exact
/// slope `ġ = ∇g·f⁻ + ∂g/∂t` there, a recorded tangent of the program, each reading `G` at
/// the last and gaining one order. The state past the crossing at the same
/// time `t` is `φ⁺_{−δ}(φ⁻_δ(u))`, each flow one RK4 step over the value-zero `δ`
/// (`rk4_dual`), exact to fourth order. At first order this is the saltation
/// `u⁺ = u + (f⁻ − f⁺)·dt*/dp`, `dt*/dp = −(∂g/∂p + ∇g·∂u/∂p)/ġ`; at the higher orders FOCEI's
/// gradient and the covariance step read (to fourth), it carries the curvature of both flows
/// and of the crossing time, which the first-order formula drops.
#[allow(clippy::too_many_arguments)]
fn jump_tr(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]), u: &[Tr], params: &[Tr], t: Tr, k: usize,
    n_sw: usize, pos: bool,
) -> Vec<Tr> {
    let n = u.len();
    let zero = Tr::k(0.0);
    // The field at `(x, t + s)` on the arm of side `side`, into `du`; returns `g`.
    let field = |x: &[Tr], s: Tr, side: bool, du: &mut [Tr]| -> Tr {
        forced_field(rhs, x, params, t + s, k, n_sw, side, du)
    };
    let (before, after) = (pos, !pos);
    let mut f0 = vec![zero; n];
    let g0 = field(u, zero, before, &mut f0);
    // `ġ = d/ds g(u + s·f⁻, t + s)` at `s = 0`.
    let (mut fs, mut c, mut ft) = ([zero], [zero], [zero]);
    super::jacobian(
        &|s: &[Tr], _: &[Tr], _: Tr, out: &mut [Tr]| {
            let x: Vec<Tr> = u.iter().zip(&f0).map(|(&a, &b)| a + s[0] * b).collect();
            out[0] = field(&x, s[0], before, &mut vec![zero; n]);
        },
        &[zero],
        params,
        zero,
        &mut fs,
        &mut c,
        &mut ft,
    );
    let c = c[0];
    let jet = |x: Tr| x - x.value_only();
    let flow = |x: &[Tr], s0: Tr, h: Tr, side: bool| {
        crate::sens::ode_provider::rk4_dual(x, s0, h, &mut |y: &[Tr], s: Tr, du: &mut [Tr]| {
            field(y, s, side, du);
        })
    };
    // Chord steps from `δ = 0` (where `G` is `g0`): each multiplies the error by
    // `1 − G′(ξ)/ġ = O(δ)`, so after `JUMP_CHORD_STEPS` it is `O(|Δp|^(steps + 1))`.
    let mut d = jet(zero - g0 / c);
    for _ in 1..JUMP_CHORD_STEPS {
        let y = flow(u, zero, d, before);
        let g = field(&y, d, before, &mut vec![zero; n]);
        d = jet(d - g / c);
    }
    let y = flow(u, zero, d, before);
    flow(&y, d, zero - d, after)
}

/// Record `f` over fresh variables (their record-time values `vals`) as a body, cut out of the
/// open recording. Outer values the body reads become captures, added to `caps`. Returns the
/// body and its outputs' record-time values.
fn record_body(
    vals: &[f64], caps: &mut Vec<u32>, masks: &[DirMask], dense: &DirMask,
    var_mask: impl Fn(usize) -> DirMask, f: impl FnOnce(&[Tr]) -> Vec<Tr>,
) -> (LoopBody, Vec<f64>) {
    let (start_code, start_reg, outer_guard, outer_guarded) =
        rec(|r| (r.code.len(), r.n_regs, r.guard.take(), std::mem::take(&mut r.guarded)));
    let vars: Vec<Tr> =
        vals.iter().map(|&v| Tr::mk(v, node(None, |d| Flat::LoadK { d, c: v }), true)).collect();
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
    let body: Vec<Flat> =
        region.into_iter().filter(|f| !matches!(f, Flat::LoadK { d, .. } if is_var(*d))).collect();
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
    // A variable is read when an instruction reads it or it is the guard; an output that is a
    // variable passes it through.
    let reads: Box<[bool]> = var_regs
        .iter()
        .map(|&v| {
            guard.is_some_and(|g| g as u32 == v) || body.iter().any(|g| uses(g).contains(&Some(v)))
        })
        .collect();
    let copy_of: Box<[Option<u32>]> = outs_r[..n_out]
        .iter()
        .map(|o| var_regs.iter().position(|v| v == o).map(|j| j as u32))
        .collect();
    let spec: Vec<(DirMask, u16)> = ins
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            if i < var_regs.len() {
                (&var_mask(i) & dense, u16::MAX)
            } else {
                (masks.get(r as usize).unwrap_or(dense).clone(), u16::MAX)
            }
        })
        .collect();
    let mut cache = chunk_cache().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let (prog, lits) = build_chunk(&body, &ins, &outs_r, spec, &mut cache);
    (
        LoopBody {
            prog,
            lits: lits.iter().map(|&k| pool[k as usize]).collect(),
            n_vars: var_regs.len(),
            caps: cap_idx,
            n_out,
            guarded: guard.is_some(),
            reads,
            copy_of,
        },
        out_vals,
    )
}

/// Record one ODE segment (see the module docs): the bodies, a value-only run for the
/// record-time outputs, and the call in the outer recording. `None` — the recording fails — if
/// a body cannot be recorded or the run does not complete.
#[allow(clippy::too_many_arguments)]
pub(super) fn record_segment(
    rhs: &dyn Fn(&[Tr], &[Tr], Tr, &mut [Tr]), u0: &[Tr], t_span: (f64, f64), params: &[Tr],
    saveat: &[f64], interp_at: &[f64], opts: &OdeSolverOptions, auto: &mut sol::OdeAutoSwitchState,
) -> Option<(Vec<SolPointG<Tr>>, Vec<SolPointG<Tr>>)> {
    let n = u0.len();
    // The span the walk named, when its ends depend on the inputs.
    let span = <Tr as crate::sens::num::RecHooks>::take_span()
        .filter(|(a, b)| (a.v, b.v) == t_span && (a.dep || b.dep));
    // A recording that has already failed is discarded whatever it records next (`end` keeps
    // the first reason): the entry state at every time, rather than bodies nothing will run.
    if rec(|r| r.failed.is_some()) {
        let at = |ts: &[f64]| -> Vec<SolPointG<Tr>> {
            ts.iter().map(|&t| SolPointG { t, u: u0.to_vec() }).collect()
        };
        return Some((at(saveat), at(interp_at)));
    }
    // A certified-linear segment the exact path declined is integrated as `auto`.
    let opts = if opts.method == OdeMethod::Linear { opts.for_integration() } else { *opts };
    let (t0, tf) = t_span;
    let dt0 = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);
    // Inside a sub-plan region every input lane may be live (its parameters are dense).
    let masks = rec(|r| {
        if r.sub_depth > 0 {
            vec![DirMask::ALL; r.n_regs as usize]
        } else {
            direction_masks(&r.code, r.n_regs as usize, &r.eta_regs, false)
        }
    });
    // Every lane the state can carry: what its inputs do.
    let dense = u0
        .iter()
        .chain(params)
        .fold(DirMask::NONE, |m, x| m | masks.get(x.r as usize).unwrap_or(&DirMask::NONE));
    let dense = if dense.no_dirs() { DirMask::ALL } else { dense };
    let mut caps: Vec<u32> = Vec::new();
    let u0v: Vec<f64> = u0.iter().map(|x| x.v).collect();

    // The right-hand side alone: `[u, t] → du`. A model with moving switches says so here.
    let mut vals = u0v.clone();
    vals.push(t0);
    sol::take_switches();
    let (rhs_body, _) = record_body(
        &vals,
        &mut caps,
        &masks,
        &dense,
        |i| if i < n { DirMask::ALL } else { DirMask::NONE },
        |v| {
            let mut du = vec![Tr::k(0.0); n];
            rhs(&v[..n], params, v[n], &mut du);
            du
        },
    );
    let n_sw = sol::take_switches();
    // The switch functions and the jumps (#2649).
    let (sw_g, jumps) = if n_sw == 0 {
        (None, Vec::new())
    } else {
        let state_lanes = |i: usize| if i < n { DirMask::ALL } else { DirMask::NONE };
        let (g, _) = record_body(&vals, &mut caps, &masks, &dense, state_lanes, |v| {
            let mut ue = v[..n].to_vec();
            ue.resize(n + n_sw, Tr::k(0.0));
            let mut du = vec![Tr::k(0.0); n + n_sw];
            rhs(&ue, params, v[n], &mut du);
            du.split_off(n)
        });
        let jumps = (0..2 * n_sw)
            .map(|m| {
                let (k, pos) = (m / 2, m % 2 == 1);
                record_body(&vals, &mut caps, &masks, &dense, state_lanes, |v| {
                    jump_tr(rhs, &v[..n], params, v[n], k, n_sw, pos)
                })
                .0
            })
            .collect();
        (Some(g), jumps)
    };
    let mut methods = Vec::new();
    for m in methods_for(&opts) {
        let mut sb = sol::make_step_body::<Tr>(n, m);
        sb.set_dense_required(!interp_at.is_empty());
        let mut probe = Vec::new();
        sb.carried_into(&mut probe);
        let c = probe.len();
        // The start, from a zero carried state; the step, from the start's.
        let body = |carried: &[f64], start: bool, force: Option<usize>, caps: &mut Vec<u32>| {
            let mut vals = u0v.clone();
            vals.extend_from_slice(carried);
            vals.push(t0);
            vals.push(dt0);
            record_body(
                &vals,
                caps,
                &masks,
                &dense,
                |i| {
                    if i < n + c {
                        DirMask::ALL
                    } else {
                        DirMask::NONE
                    }
                },
                |v| {
                    let mut sb = sol::make_step_body::<Tr>(n, m);
                    sb.set_dense_required(!interp_at.is_empty());
                    sb.set_carried(&v[n..n + c]);
                    let clock = StepClock { t: t0, dt: dt0, ts: v[n + c], dts: v[n + c + 1] };
                    if start {
                        sb.start(rhs, &v[..n], params, &clock);
                    } else if let Some(jm) = force {
                        let forced = |u: &[Tr], p: &[Tr], t: Tr, du: &mut [Tr]| {
                            forced_field(rhs, u, p, t, jm / 2, n_sw, jm % 2 == 1, du);
                        };
                        sb.step(&forced, &v[..n], params, &clock);
                    } else {
                        sb.step(rhs, &v[..n], params, &clock);
                    }
                    let mut out = Vec::new();
                    sb.carried_into(&mut out);
                    out
                },
            )
        };
        let (start, after) = body(&vec![0.0; c], true, None, &mut caps);
        let (step, stepped) = body(&after, false, None, &mut caps);
        let forced = (0..2 * n_sw).map(|jm| body(&after, false, Some(jm), &mut caps).0).collect();
        let (accept, _) = record_body(
            &stepped,
            &mut caps,
            &masks,
            &dense,
            |_| DirMask::ALL,
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
        let cl = sol::CarriedLayout::of(n, m, !interp_at.is_empty());
        debug_assert!(cl.u_new_at() + n <= c);
        methods.push(MethodBodies { method: m, c, cl, start, step, accept, forced });
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
        sw_g,
        jumps,
    });
    // The record-time values: a run at the value-only layout.
    let vals = rec(|r| r.vals.clone());
    let mut inputs: Vec<f64> = u0v.clone();
    inputs.extend(caps.iter().map(|&c| vals.get(c as usize).copied().unwrap_or(f64::NAN)));
    let mut out = vec![0.0; spec.n_out()];
    // No kernel for this segment is structural (the subject's); a run that fails on its values
    // is this point's.
    let Some(kernels) = LoopKernels::resolve(&spec, Layout::F64, Sweep::Forward) else {
        fail_subject("an ODE segment has no kernels");
        return None;
    };
    let ran = run_loop(&spec, &kernels, 1, &inputs, &mut out, auto, None);
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
                    eprintln!("kjit:   cap {c}: vals {:?} def {d:?}", r.vals.get(c as usize));
                }
            });
        }
        fail_here("an ODE segment's record-time run failed");
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
            .map(|(j, &t)| SolPointG { t, u: outs[base + j * n..base + (j + 1) * n].to_vec() })
            .collect()
    };
    Some((at(saveat, 0), at(interp_at, saveat.len() * n)))
}

/// A loop call's kernels at one layout: the right-hand side at the value layout (the driver's
/// own probes), and each method's start and step. Resolved once per plan and layout.
pub(crate) struct LoopKernels {
    rhs_f64: Arc<Tiered>,
    /// The right-hand side's index table: its slots in order.
    rhs_idx: Vec<u32>,
    /// `0, 1, …` over the widest body's slots: a reverse body's index table.
    iota: Vec<u32>,
    /// Per method: start, step, accept.
    methods: Vec<[Arc<Tiered>; 3]>,
    /// Per method, per forced step (`MethodBodies::forced`): its kernel and its index tables
    /// per current carried region.
    forced: Vec<Vec<(Arc<Tiered>, [Vec<u32>; 2])>>,
    /// The switch functions at the value layout, with their index table (#2649).
    sw_f64: Option<(Arc<Tiered>, Vec<u32>)>,
    /// Per jump its kernel and index table.
    jumps: Vec<(Arc<Tiered>, Vec<u32>)>,
    sl: Slots,
    /// Per method, per current carried region: the start's, step's and accept's index tables.
    tables: Vec<[[Vec<u32>; 3]; 2]>,
}

/// A body's output slots in its index table.
fn out_slots<'a>(b: &LoopBody, table: &'a [u32]) -> &'a [u32] {
    let at = b.n_vars + b.caps.len();
    &table[at..at + b.n_out]
}

impl LoopKernels {
    /// Body `kind` of method `m` (0 start, 1 step, 2 accept, `4 + j` forced step `j`; 3 jump
    /// `m`) with carried region `r` current: the body, its kernel, its index table.
    fn body<'a>(
        &'a self, spec: &'a LoopSpec, m: usize, kind: u8, r: usize,
    ) -> (&'a LoopBody, &'a Tiered, &'a [u32]) {
        match kind {
            0..=2 => {
                let (mb, k) = (&spec.methods[m], kind as usize);
                ([&mb.start, &mb.step, &mb.accept][k], &self.methods[m][k], &self.tables[m][r][k])
            }
            3 => (&spec.jumps[m], &self.jumps[m].0, &self.jumps[m].1),
            j => {
                let (k, t) = &self.forced[m][j as usize - 4];
                (&spec.methods[m].forced[j as usize - 4], k, &t[r])
            }
        }
    }

    /// The kernels at `layout` for `sweep` (the bodies' passes; the right-hand side and the
    /// switch functions are always the value forward).
    pub(crate) fn resolve(
        spec: &LoopSpec, layout: Layout, sweep: Sweep,
    ) -> Option<Arc<LoopKernels>> {
        let get = |b: &LoopBody| b.prog.get(layout, sweep);
        let rhs_f64 = spec.rhs.prog.get(Layout::F64, Sweep::Forward)?;
        let methods = spec
            .methods
            .iter()
            .map(|m| Some([get(&m.start)?, get(&m.step)?, get(&m.accept)?]))
            .collect::<Option<Vec<_>>>()?;
        let sl = Slots::of(spec);
        let tables = spec.methods.iter().map(|mb| tables(spec, mb, sl)).collect();
        let slots = |b: &LoopBody| b.n_vars + b.caps.len() + b.n_out + b.guarded as usize;
        let rb = &spec.rhs;
        let widest = spec
            .methods
            .iter()
            .flat_map(|m| [&m.start, &m.step, &m.accept].into_iter().chain(&m.forced))
            .chain(&spec.jumps)
            .map(slots)
            .max()
            .unwrap_or(0);
        let forced = spec
            .methods
            .iter()
            .map(|mb| {
                mb.forced
                    .iter()
                    .map(|b| Some((get(b)?, step_tables(spec, mb, b, sl))))
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let sw_f64 = match &spec.sw_g {
            Some(b) => {
                Some((b.prog.get(Layout::F64, Sweep::Forward)?, (0..slots(b) as u32).collect()))
            }
            None => None,
        };
        // A jump reads the state and the clock, and writes the new state into the first
        // carried region, dead until the fresh stepper that follows it.
        let mut vars: Vec<usize> = (0..spec.n).map(|j| sl.u + j).collect();
        vars.push(sl.t);
        let outs: Vec<usize> = (0..spec.n).map(|j| sl.carried[0] + j).collect();
        let jumps = spec
            .jumps
            .iter()
            .map(|b| Some((get(b)?, table(b, sl, &vars, &outs))))
            .collect::<Option<Vec<_>>>()?;
        Some(Arc::new(LoopKernels {
            rhs_f64,
            rhs_idx: (0..slots(rb) as u32).collect(),
            iota: (0..widest as u32).collect(),
            methods,
            forced,
            sw_f64,
            jumps,
            sl,
            tables,
        }))
    }
}

/// One event of a loop call's run, in order: what its reverse replays.
#[derive(Clone)]
pub(crate) enum Entry {
    /// A solve (re)starts from the initial state. A checkpoint ([`Entry::Ckpt`]) too.
    Begin,
    /// A stepper starts, its carried state fresh: the slots from the first carried region up to
    /// `zero` are zeroed.
    NewStepper { zero: usize },
    /// A checkpoint (#3078): no slot is known here, so a replay of the bodies after it starts
    /// from nothing but the saves they name ([`Log`]).
    Ckpt,
    /// Method `m`'s body `kind` (0 start, 1 step, 2 accept, `4 + j` forced step `j`) ran with
    /// carried region `r` current and the clock at `(t, dt)`; or, `kind` 3, jump `m` (#2649).
    /// Before it, the slots of saves `save..save + n_save` took the lanes logged with them.
    Body { m: usize, kind: u8, r: u8, t: f64, dt: f64, save: u32, n_save: u32 },
    /// The current state became output point `idx` of the hard (`saveat`) or soft list.
    Point { soft: bool, idx: usize, diverged: bool },
    /// An in-step read-out of method `m`'s continuous extension: component `i` of soft sample
    /// `idx`, at abscissa `theta` of a step of size `dt` (both constants of the reverse).
    Interp { m: usize, idx: usize, i: usize, theta: f64, dt: f64 },
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
        // A jump's new state goes into a carried region too.
        let cm = spec.methods.iter().map(|m| m.c).max().unwrap_or(0).max(n);
        let t = n + 2 * cm;
        let caps = t + 2;
        let guard = caps + nc;
        Slots { u: 0, carried: [n, n + cm], t, caps, guard, total: guard + 1 }
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

/// A step-shaped body's index tables (`[u, carried, t, dt] → carried`), per current carried
/// region.
fn step_tables(spec: &LoopSpec, mb: &MethodBodies, b: &LoopBody, sl: Slots) -> [Vec<u32>; 2] {
    let (n, c) = (spec.n, mb.c);
    [0usize, 1].map(|r| {
        let mut vars: Vec<usize> = (0..n).map(|j| sl.u + j).collect();
        vars.extend((0..c).map(|j| sl.carried[r] + j));
        vars.push(sl.t);
        vars.push(sl.t + 1);
        let outs: Vec<usize> = (0..c).map(|j| sl.carried[1 - r] + j).collect();
        table(b, sl, &vars, &outs)
    })
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
    /// Slot buffers of finished runs, for the next run on this thread.
    static BUFS: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// A zeroed-length buffer from this thread's pool.
fn take_buf() -> Vec<f64> {
    let mut b = BUFS.with(|b| b.borrow_mut().pop()).unwrap_or_default();
    b.clear();
    b
}

/// Return a buffer to this thread's pool (a few are kept; the rest are freed).
fn give_buf(b: Vec<f64>) {
    BUFS.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 16 {
            p.push(b);
        }
    });
}

/// Accepted steps between two checkpoints of a logged run (#3078): a reverse replays at most
/// this many steps' bodies from a checkpoint, at once.
const CKPT_EVERY: usize = 1;

/// A loop call's run log (#3078): its events in order, and a **checkpoint** of the lanes its
/// bodies read rather than every body's inputs. A slot's lanes are saved the first time a body
/// reads them since the last [`Entry::Ckpt`] without a body (or a fresh stepper's zeroing)
/// having written them since; the clock rides in the body's entry and the captures once in the
/// log. So the reverse regenerates every body's inputs by running the forward again from the
/// checkpoint — the bodies' own kernels at the forward's layout, on the same inputs, so the
/// same lanes bit for bit ([`Regen`]) — one span between checkpoints at a time.
///
/// A body's input that no instruction of it reads ([`LoopBody::reads`]) is not saved: nothing
/// it computes depends on it, nor its reverse. One it passes through to an output leaves that
/// output unknown when it was unknown itself, so the copy is saved only where a body really
/// reads it — a Rosenbrock start passes the last step's stages through, which the step
/// overwrites unread.
#[derive(Clone, Default)]
pub(crate) struct Log {
    pub(crate) entries: Vec<Entry>,
    /// The saved slots, in order; each one's lanes in `vals`: its values, then per seed of a
    /// rows run its tangents (`l × (1 + seeds)` each).
    saves: Vec<u32>,
    vals: Vec<f64>,
    /// The forward's kernels, its lanes and seeds (0 for a run along one layout), and its
    /// captures' lanes (values, then per seed tangents): what a replay runs.
    kern: Option<Arc<LoopKernels>>,
    l: usize,
    seeds: usize,
    caps: Vec<f64>,
    /// Test builds keep every body's inputs as the log did before #3078, laid out as a replay
    /// regenerates them, and per entry where its block starts: each replay is checked against
    /// it, and [`REPLAY_FROM_FULL`] reverses from it instead.
    #[cfg(test)]
    full: Vec<f64>,
    #[cfg(test)]
    full_at: Vec<usize>,
}

#[cfg(test)]
thread_local! {
    /// Reverse from the full test log instead of a replay: the reference a replay must match.
    pub(crate) static REPLAY_FROM_FULL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Bodies replayed so far on this thread.
    pub(crate) static REPLAYED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// Logs of dropped forwards, for the next forward on this thread.
    static LOGS: RefCell<Vec<Log>> = const { RefCell::new(Vec::new()) };
}

impl Log {
    /// An empty log, its buffers from this thread's pool.
    pub(crate) fn take() -> Log {
        let mut lg = LOGS.with(|p| p.borrow_mut().pop()).unwrap_or_default();
        lg.clear();
        lg
    }
    /// Return a log's buffers to this thread's pool (a few are kept; the rest are freed).
    pub(crate) fn give(mut self) {
        self.kern = None;
        LOGS.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < 16 {
                p.push(self);
            }
        });
    }
    fn clear(&mut self) {
        self.restart();
        self.kern = None;
        self.caps.clear();
        self.l = 0;
        self.seeds = 0;
    }
    /// Drop the events and the checkpoint: a solve starting again.
    fn restart(&mut self) {
        self.entries.clear();
        self.saves.clear();
        self.vals.clear();
        #[cfg(test)]
        {
            self.full.clear();
            self.full_at.clear();
        }
    }
    /// Log an event that is not a body.
    fn push(&mut self, e: Entry) {
        debug_assert!(!matches!(e, Entry::Body { .. }));
        self.entries.push(e);
        #[cfg(test)]
        self.full_at.push(usize::MAX);
    }
    /// The lanes a slot carries in this log: its values and each seed's tangents.
    fn w(&self) -> usize {
        self.l * (1 + self.seeds)
    }
    /// Heap the log holds, in bytes (test builds leave their full copy out).
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<Entry>()
            + self.saves.len() * 4
            + (self.vals.len() + self.caps.len()) * 8
    }
    /// The bytes the log held before #3078: every body's inputs.
    #[cfg(test)]
    pub(crate) fn full_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<Entry>() + self.full.len() * 8
    }
}

/// Free this thread's pooled slot buffers and run logs (`kernel_jit::release_thread_caches`,
/// #3077): each keeps the largest run it ever served. A pool borrowed right now is left.
pub(super) fn release_thread_buffers() {
    let _ = BUFS.try_with(|p| p.try_borrow_mut().map(|mut p| *p = Vec::new()));
    let _ = LOGS.try_with(|p| p.try_borrow_mut().map(|mut p| *p = Vec::new()));
}

/// How many slot buffers and run logs this thread's pools hold (test seam).
#[cfg(test)]
pub(super) fn pooled_buffers() -> usize {
    BUFS.with(|p| p.borrow().len()) + LOGS.with(|p| p.borrow().len())
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
    /// A run along several directions at once (#2980): per seed every slot's tangents, which
    /// follow `buf` through every operation the run applies to it.
    rows: Option<Rows>,
}

/// A rows run's tangents ([`run_loop_rows`]): per seed its slots' tangent lanes (`sl.total ×
/// l` each), its inputs' tangents, and what the run takes per seed — points and the pending
/// continuous extension.
struct Rows {
    seeds: usize,
    tb: RefCell<Vec<f64>>,
    tin: Vec<f64>,
    hard: RefCell<Vec<Vec<f64>>>,
    soft: RefCell<Vec<Vec<f64>>>,
    pend: RefCell<Vec<Vec<f64>>>,
}

#[derive(Default)]
struct RunState {
    /// The points taken so far, `n × l` lanes each: the `saveat` ones and the soft ones.
    hard: Vec<f64>,
    soft: Vec<f64>,
    pend: Vec<f64>,
    ok: bool,
    /// The run's events, when its reverse will be asked for.
    log: Option<Log>,
    /// The active stepper's method, as an index into the call's methods.
    active_m: usize,
    /// The carried region holding the active stepper's last attempt.
    active_cur: usize,
    /// The forced step steps take instead of the step, while a crossing is located (#2649).
    force: Option<usize>,
    /// Per slot, while logging, whether a replay from the last checkpoint reproduces its lanes
    /// here ([`Log`]): a body wrote them, or a save holds them.
    known: Vec<bool>,
    /// `known` at the last [`LoopRun::mark`].
    known_mark: Vec<bool>,
    /// A body's outputs' knownness, scratch.
    known_out: Vec<bool>,
    /// Accepted steps since the last checkpoint.
    since_ckpt: usize,
}

/// How far a run's log reached at one point ([`LoopRun::mark`]): its entries and saves (and
/// the run's `known` slots, kept beside it).
struct LogMark {
    entries: usize,
    saves: usize,
    #[cfg(test)]
    full: usize,
}

impl LoopRun {
    /// Where the log stands now.
    fn mark(&self) -> LogMark {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let Some(log) = st.log.as_ref() else {
            return LogMark {
                entries: 0,
                saves: 0,
                #[cfg(test)]
                full: 0,
            };
        };
        st.known_mark.clone_from(&st.known);
        LogMark {
            entries: log.entries.len(),
            saves: log.saves.len(),
            #[cfg(test)]
            full: log.full.len(),
        }
    }

    /// Drop what the log took since `m`. The slots the dropped bodies wrote hold lanes no
    /// replay makes, so they are unknown again: a later read saves what they hold.
    fn rewind(&self, m: &LogMark) {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let Some(log) = st.log.as_mut() else { return };
        st.known.clone_from(&st.known_mark);
        for e in &log.entries[m.entries..] {
            if let Entry::Body { m: bm, kind, r, .. } = *e {
                let (b, _, table) = self.kern.body(&self.spec, bm, kind, r as usize);
                for &o in out_slots(b, table) {
                    st.known[o as usize] = false;
                }
            }
        }
        log.entries.truncate(m.entries);
        let w = log.w();
        log.saves.truncate(m.saves);
        log.vals.truncate(m.saves * w);
        #[cfg(test)]
        {
            log.full.truncate(m.full);
            log.full_at.truncate(m.entries);
        }
    }

    /// Save slot `s`'s lanes into the log: its values, and each seed's tangents.
    fn save(&self, log: &mut Log, buf: &[f64], s: usize) {
        let l = self.l;
        log.saves.push(s as u32);
        log.vals.extend_from_slice(&buf[s * l..(s + 1) * l]);
        if let Some(r) = self.rows.as_ref() {
            let tb = r.tb.borrow();
            let stride = self.sl.total * l;
            for sd in 0..r.seeds {
                log.vals.extend_from_slice(&tb[sd * stride + s * l..sd * stride + (s + 1) * l]);
            }
        }
    }

    /// An accepted step ended: every [`CKPT_EVERY`] of them, a checkpoint.
    fn accepted(&self) {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let Some(log) = st.log.as_mut() else { return };
        st.since_ckpt += 1;
        if st.since_ckpt >= CKPT_EVERY {
            st.since_ckpt = 0;
            log.push(Entry::Ckpt);
            st.known.fill(false);
        }
    }

    fn new(
        spec: &Arc<LoopSpec>, kern: &Arc<LoopKernels>, l: usize, inputs: &[f64], log: Option<Log>,
        rows: Option<(&[f64], usize)>,
    ) -> Self {
        let (n, nc) = (spec.n, spec.n_caps);
        let sl = kern.sl;
        let caps = sl.caps;
        let mut buf = take_buf();
        buf.resize(sl.total * l, 0.0);
        buf[caps * l..(caps + nc) * l].copy_from_slice(&inputs[n * l..(n + nc) * l]);
        let mut ins = take_buf();
        ins.extend_from_slice(inputs);
        let rows = rows.map(|(tin, seeds)| {
            let (stride, w_in) = (sl.total * l, (n + nc) * l);
            let mut tb = vec![0.0; seeds * stride];
            for sd in 0..seeds {
                tb[sd * stride + caps * l..sd * stride + (caps + nc) * l]
                    .copy_from_slice(&tin[sd * w_in + n * l..(sd + 1) * w_in]);
            }
            let per = || RefCell::new(vec![Vec::new(); seeds]);
            Rows {
                seeds,
                tb: RefCell::new(tb),
                tin: tin.to_vec(),
                hard: per(),
                soft: per(),
                pend: per(),
            }
        });
        let mut log = log;
        let known = if let Some(lg) = log.as_mut() {
            lg.clear();
            lg.kern = Some(kern.clone());
            lg.l = l;
            lg.caps.extend_from_slice(&inputs[n * l..(n + nc) * l]);
            if let Some(r) = rows.as_ref() {
                lg.seeds = r.seeds;
                let w_in = (n + nc) * l;
                for sd in 0..r.seeds {
                    lg.caps.extend_from_slice(&r.tin[sd * w_in + n * l..(sd + 1) * w_in]);
                }
            }
            vec![false; sl.total]
        } else {
            Vec::new()
        };
        LoopRun {
            rows,
            spec: spec.clone(),
            kern: kern.clone(),
            l,
            sl,
            buf: RefCell::new(buf),
            inputs: ins,
            st: RefCell::new(RunState {
                ok: true,
                log,
                hard: take_buf(),
                soft: take_buf(),
                known,
                ..Default::default()
            }),
        }
    }

    /// Run `b`'s kernel `k` in place through `table` (carried region `r` current); `false` when
    /// its guard fails. When the run keeps a log, saves the slots the body reads that a replay
    /// would not reproduce ([`Log`]) and logs the body.
    fn run(&self, b: &LoopBody, k: &Tiered, table: &[u32], m: usize, kind: u8, r: usize) -> bool {
        let l = self.l;
        let mut buf = self.buf.borrow_mut();
        let logging = {
            let mut st = self.st.borrow_mut();
            let st = &mut *st;
            if let Some(log) = st.log.as_mut() {
                let save = log.saves.len();
                let ts = self.sl.t;
                for (&s, &read) in table[..b.n_vars].iter().zip(b.reads.iter()) {
                    let s = s as usize;
                    // The clock rides in the entry.
                    if read && s != ts && s != ts + 1 && !st.known[s] {
                        self.save(log, &buf, s);
                        st.known[s] = true;
                    }
                }
                debug_assert!(
                    buf[ts * l + 1..(ts + 2) * l]
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| { i + 1 == l || x == 0.0 }),
                    "the clock's lanes past its value are zero, as a replay sets them"
                );
                #[cfg(test)]
                {
                    let at = log.full.len();
                    let stride = self.sl.total * l;
                    let tb = self.rows.as_ref().map(|r| r.tb.borrow());
                    for &s in &table[..b.n_vars] {
                        let s = s as usize;
                        log.full.extend_from_slice(&buf[s * l..(s + 1) * l]);
                        for sd in 0..log.seeds {
                            let tb = tb.as_ref().expect("a rows run");
                            log.full.extend_from_slice(
                                &tb[sd * stride + s * l..sd * stride + (s + 1) * l],
                            );
                        }
                    }
                    log.full_at.push(at);
                }
                log.entries.push(Entry::Body {
                    m,
                    kind,
                    r: r as u8,
                    t: buf[ts * l],
                    dt: buf[(ts + 1) * l],
                    save: save as u32,
                    n_save: (log.saves.len() - save) as u32,
                });
                true
            } else {
                false
            }
        };
        debug_assert!(
            buf.len() >= self.sl.total * l && table.iter().all(|&s| (s as usize) < self.sl.total),
            "a body call needs every slot's lanes in `buf`: {} < {} slots × {l} lanes",
            buf.len(),
            self.sl.total
        );
        // SAFETY: every slot in `table` is below `sl.total`, whose `l` lanes each `buf` holds,
        // laid out as the kernel's layout reads them.
        match self.rows.as_ref() {
            // SAFETY: as above, each seed's block of `tb` laid out like `buf`.
            Some(r) => unsafe {
                k.call_seeds(
                    buf.as_mut_ptr(),
                    r.tb.borrow_mut().as_mut_ptr(),
                    self.sl.total * l,
                    r.seeds,
                    table.as_ptr(),
                    b.lits.as_ptr(),
                )
            },
            None => unsafe {
                k.call(buf.as_mut_ptr(), std::ptr::null_mut(), table.as_ptr(), b.lits.as_ptr())
            },
        }
        if logging {
            // An output is known unless it is a copy of a variable a replay does not hold
            // (one passed through, never read): then it holds whatever the replay has there,
            // and a body that reads it later saves it.
            let mut st = self.st.borrow_mut();
            let st = &mut *st;
            let mut known = std::mem::take(&mut st.known_out);
            known.clear();
            known.extend(b.copy_of.iter().map(|c| {
                c.is_none_or(|j| {
                    let s = table[j as usize] as usize;
                    s == self.sl.t || s == self.sl.t + 1 || st.known[s]
                })
            }));
            for (&o, &k) in out_slots(b, table).iter().zip(&known) {
                st.known[o as usize] = k;
            }
            st.known_out = known;
        }
        let ok = !b.guarded || buf[self.sl.guard * l] == 1.0;
        if !ok && super::debug() {
            let vars: Vec<f64> = table[..b.n_vars].iter().map(|&s| buf[s as usize * l]).collect();
            eprintln!("kjit: loop body guard failed: method {m} kind {kind} vars {vars:?}");
        }
        ok
    }

    /// Take the current state's lanes as a point.
    fn take_point(&self, st: &mut RunState, soft: bool, diverged: bool) {
        let (n, l) = (self.spec.n, self.l);
        let buf = self.buf.borrow();
        let u = &buf[self.sl.u * l..(self.sl.u + n) * l];
        let dst = if soft { &mut st.soft } else { &mut st.hard };
        let idx = dst.len() / (n * l);
        if diverged {
            // `v · NaN` in every lane, as the dual tail is formed.
            dst.extend(u.iter().map(|&v| v * f64::NAN));
        } else {
            dst.extend_from_slice(u);
        }
        if let Some(r) = self.rows.as_ref() {
            let tb = r.tb.borrow();
            let stride = self.sl.total * l;
            let mut pts = if soft { r.soft.borrow_mut() } else { r.hard.borrow_mut() };
            for (sd, d) in pts.iter_mut().enumerate() {
                let tu = &tb[sd * stride + self.sl.u * l..sd * stride + (self.sl.u + n) * l];
                if diverged {
                    d.extend(tu.iter().map(|&v| v * f64::NAN));
                } else {
                    d.extend_from_slice(tu);
                }
            }
        }
        if let Some(log) = st.log.as_mut() {
            log.push(Entry::Point { soft, idx, diverged });
        }
    }

    fn fail(&self) {
        self.st.borrow_mut().ok = false;
    }
}

impl Drop for LoopRun {
    fn drop(&mut self) {
        give_buf(std::mem::take(self.buf.get_mut()));
        give_buf(std::mem::take(&mut self.inputs));
        let st = self.st.get_mut();
        give_buf(std::mem::take(&mut st.hard));
        give_buf(std::mem::take(&mut st.soft));
    }
}

/// [`sol::LaneHook`]: points take the current lanes. The second field is the switch
/// functions' scratch.
struct Hook(Rc<LoopRun>, RefCell<Vec<f64>>);

impl sol::LaneHook for Hook {
    fn stepper(&self, _n: usize, method: OdeMethod) -> Box<dyn sol::Stepper<f64>> {
        let run = self.0.clone();
        let m = match run.spec.methods.iter().position(|x| x.method == method) {
            Some(m) => m,
            None => {
                run.fail();
                0
            }
        };
        let mb = &run.spec.methods[m];
        let (cl, c) = (mb.cl, mb.c);
        {
            let mut st = run.st.borrow_mut();
            let st = &mut *st;
            st.active_m = m;
            if let Some(log) = st.log.as_mut() {
                // A replay zeroes the same slots.
                let zero = run.sl.carried[1] + c;
                log.push(Entry::NewStepper { zero });
                st.known[run.sl.carried[0]..zero].fill(true);
            }
        }
        // A fresh stepper's carried state is zero.
        {
            let sl = run.sl;
            let l = run.l;
            let mut buf = run.buf.borrow_mut();
            buf[sl.carried[0] * l..(sl.carried[1] + c) * l].fill(0.0);
            if let Some(r) = run.rows.as_ref() {
                let stride = sl.total * l;
                let mut tb = r.tb.borrow_mut();
                for sd in 0..r.seeds {
                    tb[sd * stride + sl.carried[0] * l..sd * stride + (sl.carried[1] + c) * l]
                        .fill(0.0);
                }
            }
        }
        Box::new(Lanes {
            u_new: vec![0.0; run.spec.n],
            run,
            m,
            cl,
            started: false,
            cur: 0,
            attempt_mark: None,
        })
    }
    fn begin(&self) {
        let run = &self.0;
        let (n, l) = (run.spec.n, run.l);
        run.buf.borrow_mut()[run.sl.u * l..(run.sl.u + n) * l]
            .copy_from_slice(&run.inputs[..n * l]);
        if let Some(r) = run.rows.as_ref() {
            let (stride, w_in) = (run.sl.total * l, (n + run.spec.n_caps) * l);
            let mut tb = r.tb.borrow_mut();
            for sd in 0..r.seeds {
                tb[sd * stride + run.sl.u * l..sd * stride + (run.sl.u + n) * l]
                    .copy_from_slice(&r.tin[sd * w_in..sd * w_in + n * l]);
            }
            for v in r.hard.borrow_mut().iter_mut().chain(r.soft.borrow_mut().iter_mut()) {
                v.clear();
            }
        }
        let mut st = run.st.borrow_mut();
        let st = &mut *st;
        st.hard.clear();
        st.soft.clear();
        if let Some(log) = st.log.as_mut() {
            // A checkpoint: the state's lanes are saved where a body first reads them.
            log.restart();
            log.push(Entry::Begin);
            st.known.fill(false);
            st.since_ckpt = 0;
        }
    }
    fn point(&self, soft: bool) {
        let mut st = self.0.st.borrow_mut();
        self.0.take_point(&mut st, soft, false);
    }
    fn interp(&self, theta: f64, dt: f64, i: usize) {
        let run = &self.0;
        let (n, l, sl) = (run.spec.n, run.l, run.sl);
        let mut st = run.st.borrow_mut();
        let st = &mut *st;
        let (m, idx) = (st.active_m, st.soft.len() / (n * l));
        if let Some(log) = st.log.as_mut() {
            log.push(Entry::Interp { m, idx, i, theta, dt });
        }
        if st.pend.len() != n * l {
            st.pend.clear();
            st.pend.resize(n * l, 0.0);
        }
        // Every lane off the continuous extension, which is linear in the lanes: the step's
        // carried state at that lane's stride, from the state it started at.
        let cl = run.spec.methods[m].cl;
        let buf = run.buf.borrow();
        let base = sl.carried[st.active_cur] * l;
        for lane in 0..l {
            let c = sol::Carried::new(&buf[base + lane..], l);
            let u_old = buf[(sl.u + i) * l + lane];
            st.pend[i * l + lane] = cl.interp(c, theta, u_old, dt, i);
        }
        // The extension is linear in the lanes, so each seed's tangent is the same map of its
        // carried state's tangents.
        if let Some(r) = run.rows.as_ref() {
            let stride = sl.total * l;
            let tb = r.tb.borrow();
            for (sd, p) in r.pend.borrow_mut().iter_mut().enumerate() {
                if p.len() != n * l {
                    p.clear();
                    p.resize(n * l, 0.0);
                }
                let t = &tb[sd * stride..(sd + 1) * stride];
                for lane in 0..l {
                    let c = sol::Carried::new(&t[base + lane..], l);
                    let u_old = t[(sl.u + i) * l + lane];
                    p[i * l + lane] = cl.interp(c, theta, u_old, dt, i);
                }
            }
        }
    }
    fn interp_done(&self) {
        let mut st = self.0.st.borrow_mut();
        let st = &mut *st;
        st.soft.extend_from_slice(&st.pend);
        if let Some(r) = self.0.rows.as_ref() {
            for (s, p) in r.soft.borrow_mut().iter_mut().zip(r.pend.borrow().iter()) {
                s.extend_from_slice(p);
            }
        }
    }
    fn tail(&self, soft: bool, diverged: bool) {
        let mut st = self.0.st.borrow_mut();
        self.0.take_point(&mut st, soft, diverged);
    }
    fn switches(&self) -> usize {
        self.0.spec.jumps.len() / 2
    }
    fn force(&self, arm: Option<(usize, bool)>) {
        self.0.st.borrow_mut().force = arm.map(|(k, pos)| 2 * k + usize::from(pos));
    }
    fn switch_g(&self, u: &[f64], t: f64, out: &mut [f64]) {
        let run = &self.0;
        let (Some(b), Some((k, idx))) = (&run.spec.sw_g, &run.kern.sw_f64) else {
            return;
        };
        let (n, l) = (run.spec.n, run.l);
        let n_in = b.n_vars + b.caps.len();
        let mut buf = self.1.borrow_mut();
        buf.clear();
        buf.resize(idx.len(), 0.0);
        buf[..n].copy_from_slice(&u[..n]);
        buf[n] = t;
        for (j, &c) in b.caps.iter().enumerate() {
            buf[b.n_vars + j] = run.inputs[(n + c) * l];
        }
        // SAFETY: `idx` names every single-lane slot of `buf`, laid out as the body reads them.
        unsafe { k.call(buf.as_mut_ptr(), std::ptr::null_mut(), idx.as_ptr(), b.lits.as_ptr()) };
        if b.guarded && buf[n_in + b.n_out] != 1.0 {
            run.fail();
        }
        out.copy_from_slice(&buf[n_in..n_in + b.n_out]);
    }
    fn jump(&self, k: usize, pos: bool, t: f64) {
        let run = &self.0;
        let m = 2 * k + usize::from(pos);
        let (n, l, sl) = (run.spec.n, run.l, run.sl);
        run.buf.borrow_mut()[sl.t * l] = t;
        let (kern, table) = &run.kern.jumps[m];
        if !run.run(&run.spec.jumps[m], kern, table, m, 3, 0) {
            run.fail();
        }
        let src = sl.carried[0] * l;
        run.buf.borrow_mut().copy_within(src..src + n * l, sl.u * l);
        {
            // A replay copies the same lanes.
            let mut st = run.st.borrow_mut();
            if st.log.is_some() {
                for i in 0..n {
                    st.known[sl.u + i] = st.known[sl.carried[0] + i];
                }
            }
        }
        if let Some(r) = run.rows.as_ref() {
            let stride = sl.total * l;
            let mut tb = r.tb.borrow_mut();
            for sd in 0..r.seeds {
                let o = sd * stride;
                tb.copy_within(o + src..o + src + n * l, o + sl.u * l);
            }
        }
    }
}

/// A stepper over a loop call's lanes: every lane runs through the recorded bodies in the run's
/// buffer, and the driver's decisions (the error norm, the stiffness indicator, usability, the
/// continuous extension) read the value lane of the carried state in place
/// ([`sol::CarriedLayout`]), by the definitions the `f64` stepper's own reads call.
struct Lanes {
    run: Rc<LoopRun>,
    m: usize,
    cl: sol::CarriedLayout,
    /// Whether the carried state holds a step start for the current `(u, t)`.
    started: bool,
    u_new: Vec<f64>,
    /// Which carried region holds the current state.
    cur: usize,
    /// Where the run's logs stood before the last attempt's step body, while the driver has
    /// not yet accepted or rejected it ([`LoopRun::rewind`]).
    attempt_mark: Option<LogMark>,
}

impl Lanes {
    /// Run body `kind` (0 start, 1 step, 2 accept, `4 + j` forced step `j`); the carried
    /// regions swap.
    fn body(&mut self, kind: usize) {
        let run = &self.run;
        let (b, k, table) = run.kern.body(&run.spec, self.m, kind as u8, self.cur);
        if !run.run(b, k, table, self.m, kind as u8, self.cur) {
            run.fail();
        }
        self.cur = 1 - self.cur;
    }
    /// Read `f` off the value lane of the current carried state.
    fn value<R>(&self, f: impl FnOnce(sol::Carried<'_>) -> R) -> R {
        let (l, base) = (self.run.l, self.run.sl.carried[self.cur]);
        let buf = self.run.buf.borrow();
        f(sol::Carried::new(&buf[base * l..], l))
    }
}

impl sol::Stepper<f64> for Lanes {
    fn attempt(
        &mut self, _rhs: &sol::RhsG<'_, f64>, u: &[f64], _params: &[f64], t: f64, dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        {
            let (l, ts) = (self.run.l, self.run.sl.t);
            let mut buf = self.run.buf.borrow_mut();
            buf[ts * l] = t;
            buf[(ts + 1) * l] = dt;
        }
        if !self.started {
            self.body(0);
        }
        let force = self.run.st.borrow().force;
        self.attempt_mark = Some(self.run.mark());
        self.body(force.map_or(1, |j| 4 + j));
        self.started = true;
        self.run.st.borrow_mut().active_cur = self.cur;
        let (l, base) = (self.run.l, self.run.sl.carried[self.cur]);
        let un = base + self.cl.u_new_at();
        let buf = self.run.buf.borrow();
        for (i, x) in self.u_new.iter_mut().enumerate() {
            *x = buf[(un + i) * l];
        }
        self.cl.err(sol::Carried::new(&buf[base * l..], l), u, dt, opts)
    }
    fn u_new(&self) -> &[f64] {
        &self.u_new
    }
    fn interpolate_component(&self, theta: f64, u_old: &[f64], dt: f64, i: usize) -> f64 {
        self.value(|c| self.cl.interp(c, theta, u_old[i], dt, i))
    }
    fn on_accept(&mut self) {
        self.attempt_mark = None;
        self.body(2);
        self.started = self.cl.start_kept_on_accept();
        self.run.accepted();
    }
    /// A rejected attempt leaves no trace (#3065): its step body leaves the logs, and the
    /// carried region it read is current again. The trial may have produced non-finite stages
    /// (a stage state overshooting where the right-hand side has no real value, as `f64::powf`
    /// of a negative base at a non-integer exponent), which the driver rejects on the value
    /// lane exactly as the `f64` stepper does; but kept in the log, the reverse carried the
    /// retry's adjoints back through that body's vector–Jacobian product, `0 · NaN = NaN` in
    /// every term of a stage nothing read, and every derivative of the subject came back `NaN`.
    /// The retry then reads the region the rejected attempt read, not the copies the attempt
    /// passed through, so nothing rests on a step body passing its inputs through bit for bit.
    fn on_reject(&mut self) {
        if let Some(m) = self.attempt_mark.take() {
            self.run.rewind(&m);
            self.cur = 1 - self.cur;
        }
    }
    fn err_exp(&self) -> f64 {
        self.cl.err_exp()
    }
    fn set_dense_required(&mut self, yes: bool) {
        // The bodies were recorded for the call's own read-outs.
        debug_assert_eq!(yes, !self.run.spec.interp_at.is_empty());
    }
    fn attempt_usable(&self) -> bool {
        self.value(|c| self.cl.usable(c))
    }
    fn stiffness_indicator(&self, dt: f64) -> Option<f64> {
        self.value(|c| self.cl.stiffness(c, dt))
    }
}

/// Run a loop call over `kern` (at `kern`'s layout, `l` lanes): `inputs` its inputs' lanes
/// (`[u0, captures]`), `out` its outputs' (every state component at each `saveat`, then each
/// `interp_at` time). `false` when another solver override is in force than the call was
/// recorded under, or a body's guard failed on some step. `log`, when given, receives the
/// run's events for its reverse.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop(
    spec: &Arc<LoopSpec>, kern: &Arc<LoopKernels>, l: usize, inputs: &[f64], out: &mut [f64],
    auto: &mut sol::OdeAutoSwitchState, log: Option<&mut Log>,
) -> bool {
    run_loop_g(spec, kern, l, inputs, out, auto, log, None)
}

/// A loop call along `seeds` directions at once (#2980): `kern` the kernels at the base layout
/// for [`super::Sweep::TangentRows`], `tin` per seed its inputs' tangents (laid out as
/// `inputs`), `tout` per seed its outputs' tangents (as `out`). The step decisions read the
/// values alone, so every seed takes the run's steps; each body runs its primal once and each
/// seed's tangent. `logs`, when given, receive the run's log in the first (its checkpoint
/// carrying every seed's tangents, [`Log`]); the others are left empty. [`run_loop_rev_rows`]
/// reverses every seed from it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop_rows(
    spec: &Arc<LoopSpec>, kern: &Arc<LoopKernels>, l: usize, inputs: &[f64], tin: &[f64],
    seeds: usize, out: &mut [f64], tout: &mut [f64], auto: &mut sol::OdeAutoSwitchState,
    logs: Option<&mut [Log]>,
) -> bool {
    let mut main = logs.as_ref().map(|_| Log::take());
    let rows = RowsIo { tin, seeds, tout };
    let mut io = Some(rows);
    let ok = run_loop_g(spec, kern, l, inputs, out, auto, main.as_mut(), io.as_mut());
    if let (true, Some(logs), Some(mut main)) = (ok, logs, main) {
        for lg in logs.iter_mut() {
            lg.clear();
        }
        if let Some(first) = logs.first_mut() {
            std::mem::swap(first, &mut main);
        }
        main.give();
    }
    ok
}

/// A rows run's inputs and outputs ([`run_loop_rows`]).
struct RowsIo<'a> {
    tin: &'a [f64],
    seeds: usize,
    tout: &'a mut [f64],
}

#[allow(clippy::too_many_arguments)]
fn run_loop_g(
    spec: &Arc<LoopSpec>, kern: &Arc<LoopKernels>, l: usize, inputs: &[f64], out: &mut [f64],
    auto: &mut sol::OdeAutoSwitchState, log: Option<&mut Log>, rows: Option<&mut RowsIo<'_>>,
) -> bool {
    if sol::current_ode_override() != spec.ov {
        return false;
    }
    let mut log = log;
    let run = Rc::new(LoopRun::new(
        spec,
        kern,
        l,
        inputs,
        log.as_deref_mut().map(std::mem::take),
        rows.as_ref().map(|r| (r.tin, r.seeds)),
    ));
    let n = spec.n;
    // The driver's own right-hand side (its stiffness probes): the body at the value layout,
    // the captures at their value lanes.
    let rhs_run = run.clone();
    let rb = &spec.rhs;
    let n_in = rb.n_vars + rb.caps.len();
    let n_slots = n_in + rb.n_out + rb.guarded as usize;
    let mut rb_buf = take_buf();
    rb_buf.resize(n_slots, 0.0);
    for (j, &c) in rb.caps.iter().enumerate() {
        rb_buf[rb.n_vars + j] = inputs[(n + c) * l];
    }
    let rhs_buf = RefCell::new(rb_buf);
    let rhs_f = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
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
                r.kern.rhs_idx.as_ptr(),
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
    let mut u0 = take_buf();
    u0.extend((0..n).map(|j| inputs[j * l]));
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
    let hook: Rc<dyn sol::LaneHook> = Rc::new(Hook(run.clone(), RefCell::new(Vec::new())));
    let (hard, soft) = sol::with_lane_hook(hook, || {
        sol::integrate_nonlinear_f64(&rhs_f, &u0, t_span, saveat, &spec.interp_at, &spec.opts, auto)
    });
    give_buf(u0);
    give_buf(rhs_buf.into_inner());
    let mut st = run.st.borrow_mut();
    let nl = n * l;
    if !st.ok || st.hard.len() != hard.len() * nl || st.soft.len() != soft.len() * nl {
        if super::debug() {
            eprintln!(
                "kjit: loop run failed: ok {} hard {}/{} soft {}/{} span {:?} saveat {:?}",
                st.ok,
                st.hard.len() / nl,
                hard.len(),
                st.soft.len() / nl,
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
    let nh = st.hard.len();
    out[..nh].copy_from_slice(&st.hard);
    out[nh..nh + st.soft.len()].copy_from_slice(&st.soft);
    if let (Some(io), Some(r)) = (rows, run.rows.as_ref()) {
        let w = nh + st.soft.len();
        let (hard, soft) = (r.hard.borrow(), r.soft.borrow());
        for sd in 0..io.seeds {
            let (h, s) = (&hard[sd], &soft[sd]);
            if h.len() != nh || s.len() != st.soft.len() {
                return false;
            }
            io.tout[sd * w..sd * w + nh].copy_from_slice(h);
            io.tout[sd * w + nh..(sd + 1) * w].copy_from_slice(s);
        }
    }
    true
}

/// The reverse of the log entries that are linear in the outputs' adjoints — a saved point
/// and an in-step read-out — over `nb` adjoint blocks of `out_adj` (a seed each in
/// [`run_loop_rev`], a value or tangent block in [`run_loop_rev_rows`]). A read-out is
/// linear in the step's carried state and its start state at its constant `(theta, dt)`;
/// its reverse is the transpose of that map, the coefficients read off the method's
/// [`sol::CarriedLayout`] by unit probes.
#[allow(clippy::too_many_arguments)]
fn rev_linear_entry(
    e: &Entry, spec: &LoopSpec, l: usize, nb: usize, out_adj: &[f64], ubar: &mut [f64],
    cbar: &mut Vec<f64>, probe: &mut Vec<f64>, a_c: &mut Vec<f64>,
) {
    let n_hard = spec.saveat.len();
    let nl = spec.n * l;
    let out_w = out_adj.len() / nb;
    match e {
        Entry::Point { soft, idx: k, diverged } => {
            let at = (if *soft { n_hard + k } else { *k }) * nl;
            for b in 0..nb {
                let src = &out_adj[b * out_w + at..b * out_w + at + nl];
                for (x, &g) in ubar[b * nl..(b + 1) * nl].iter_mut().zip(src) {
                    *x += if *diverged { g * f64::NAN } else { g };
                }
            }
        }
        Entry::Interp { m, idx: k, i, theta, dt } => {
            let mb = &spec.methods[*m];
            let cl = mb.c * l;
            if cbar.len() != nb * cl {
                cbar.clear();
                cbar.resize(nb * cl, 0.0);
            }
            // The sample is `a_u·u_old[i] + Σ_j a_j·carried[j]`: probe each coefficient.
            probe.clear();
            probe.resize(mb.c, 0.0);
            let a_u = mb.cl.interp(sol::Carried::new(probe, 1), *theta, 1.0, *dt, *i);
            a_c.clear();
            for j in 0..mb.c {
                probe[j] = 1.0;
                a_c.push(mb.cl.interp(sol::Carried::new(probe, 1), *theta, 0.0, *dt, *i));
                probe[j] = 0.0;
            }
            let at = (n_hard + k) * nl + i * l;
            for b in 0..nb {
                for x in 0..l {
                    let g = out_adj[b * out_w + at + x];
                    ubar[b * nl + i * l + x] += g * a_u;
                    for (j, &a) in a_c.iter().enumerate() {
                        cbar[b * cl + j * l + x] += g * a;
                    }
                }
            }
        }
        _ => unreachable!("only the linear entries"),
    }
}

/// A reverse's view of its log's body inputs (#3078): the span between two checkpoints that
/// holds the body asked for, run forward again from its checkpoint — the saves restored where
/// the forward took them, the clock and captures set, each body's kernel at the forward's
/// layout through the forward's own index table — its bodies' inputs gathered as they go, laid
/// out as the log kept them before: per body its variables, `w` lanes each (values, then per
/// seed of a rows run its tangents). The forward is deterministic and every tier of a kernel
/// computes the same bits, so every input a body reads is the forward's own; one it does not
/// read, or only passes through, is whatever the replay holds there, which neither its
/// outputs that are read later nor its reverse's adjoints look at.
struct Regen {
    /// The entries regenerated, `lo..=hi`; `hi < lo` before the first.
    lo: usize,
    hi: usize,
    /// Per entry of the span, where its body's inputs start in `vars`.
    offs: Vec<usize>,
    vars: Vec<f64>,
    buf: Vec<f64>,
    tb: Vec<f64>,
}

impl Regen {
    fn new() -> Regen {
        Regen { lo: 1, hi: 0, offs: take_idx(), vars: take_buf(), buf: take_buf(), tb: take_buf() }
    }

    /// Where body entry `i`'s inputs start in `self.vars`.
    fn at(&mut self, spec: &LoopSpec, log: &Log, i: usize) -> usize {
        if !(self.lo..=self.hi).contains(&i) {
            let lo = log.entries[..=i]
                .iter()
                .rposition(|e| matches!(e, Entry::Begin | Entry::Ckpt))
                .unwrap_or(0);
            self.replay(spec, log, lo, i);
        }
        self.offs[i - self.lo]
    }

    /// Run the forward of entries `lo..=hi` again (`lo` a checkpoint), gathering every body's
    /// inputs; the last body's kernel need not run.
    fn replay(&mut self, spec: &LoopSpec, log: &Log, lo: usize, hi: usize) {
        let kern = log.kern.as_ref().expect("a log keeps its kernels");
        let (l, seeds, w) = (log.l, log.seeds, log.w());
        let (sl, n, nc) = (kern.sl, spec.n, spec.n_caps);
        let stride = sl.total * l;
        let (buf, tb) = (&mut self.buf, &mut self.tb);
        buf.clear();
        buf.resize(stride, 0.0);
        tb.clear();
        tb.resize(seeds * stride, 0.0);
        buf[sl.caps * l..(sl.caps + nc) * l].copy_from_slice(&log.caps[..nc * l]);
        for sd in 0..seeds {
            let o = sd * stride + sl.caps * l;
            tb[o..o + nc * l].copy_from_slice(&log.caps[(1 + sd) * nc * l..(2 + sd) * nc * l]);
        }
        self.offs.clear();
        self.vars.clear();
        for i in lo..=hi {
            match log.entries[i] {
                Entry::NewStepper { zero } => {
                    let z = sl.carried[0] * l..zero * l;
                    buf[z.clone()].fill(0.0);
                    for sd in 0..seeds {
                        tb[sd * stride + z.start..sd * stride + z.end].fill(0.0);
                    }
                    self.offs.push(usize::MAX);
                }
                Entry::Body { m, kind, r, t, dt, save, n_save } => {
                    for q in save as usize..(save + n_save) as usize {
                        let s = log.saves[q] as usize;
                        let v = &log.vals[q * w..(q + 1) * w];
                        buf[s * l..(s + 1) * l].copy_from_slice(&v[..l]);
                        for sd in 0..seeds {
                            tb[sd * stride + s * l..sd * stride + (s + 1) * l]
                                .copy_from_slice(&v[(1 + sd) * l..(2 + sd) * l]);
                        }
                    }
                    buf[sl.t * l] = t;
                    buf[(sl.t + 1) * l] = dt;
                    let (b, k, table) = kern.body(spec, m, kind, r as usize);
                    let at = self.vars.len();
                    self.offs.push(at);
                    for &s in &table[..b.n_vars] {
                        let s = s as usize;
                        self.vars.extend_from_slice(&buf[s * l..(s + 1) * l]);
                        for sd in 0..seeds {
                            self.vars.extend_from_slice(
                                &tb[sd * stride + s * l..sd * stride + (s + 1) * l],
                            );
                        }
                    }
                    #[cfg(test)]
                    {
                        REPLAYED.with(|c| c.set(c.get() + 1));
                        let fa = log.full_at[i];
                        let full = &log.full[fa..fa + b.n_vars * w];
                        let got = &mut self.vars[at..];
                        for (j, &read) in b.reads.iter().enumerate() {
                            let (x, y) = (&got[j * w..(j + 1) * w], &full[j * w..(j + 1) * w]);
                            assert!(
                                !read || x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()),
                                "a replayed body input differs from the forward's: entry {i}, \
                                 method {m} kind {kind} variable {j}: {x:?} vs {y:?}"
                            );
                        }
                        if REPLAY_FROM_FULL.with(std::cell::Cell::get) {
                            got[..full.len()].copy_from_slice(full);
                        }
                    }
                    if i == hi {
                        continue;
                    }
                    // SAFETY: as `LoopRun::run`: the forward's kernel through its own table over
                    // a buffer laid out as the forward's (and each seed's block of `tb`).
                    if seeds > 0 {
                        unsafe {
                            k.call_seeds(
                                buf.as_mut_ptr(),
                                tb.as_mut_ptr(),
                                stride,
                                seeds,
                                table.as_ptr(),
                                b.lits.as_ptr(),
                            )
                        }
                    } else {
                        unsafe {
                            k.call(
                                buf.as_mut_ptr(),
                                std::ptr::null_mut(),
                                table.as_ptr(),
                                b.lits.as_ptr(),
                            )
                        }
                    }
                    if kind == 3 {
                        let src = sl.carried[0] * l;
                        buf.copy_within(src..src + n * l, sl.u * l);
                        for sd in 0..seeds {
                            let o = sd * stride;
                            tb.copy_within(o + src..o + src + n * l, o + sl.u * l);
                        }
                    }
                }
                _ => self.offs.push(usize::MAX),
            }
        }
        self.lo = lo;
        self.hi = hi;
    }
}

impl Drop for Regen {
    fn drop(&mut self) {
        give_idx(std::mem::take(&mut self.offs));
        for b in [&mut self.vars, &mut self.buf, &mut self.tb] {
            give_buf(std::mem::take(b));
        }
    }
}

thread_local! {
    /// Offset tables of finished reverses, for the next on this thread.
    static IDX: RefCell<Vec<Vec<usize>>> = const { RefCell::new(Vec::new()) };
}

fn take_idx() -> Vec<usize> {
    let mut v = IDX.with(|b| b.borrow_mut().pop()).unwrap_or_default();
    v.clear();
    v
}

fn give_idx(v: Vec<usize>) {
    IDX.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < 16 {
            p.push(v);
        }
    });
}

/// The reverse of a loop call over `rev`'s reverse kernels at `l` lanes: from the adjoints of its
/// outputs `out_adj`, add those of its inputs into `in_adj` (both laid out as `run_loop`'s,
/// one block per seed of `seeds`). `log` is the forward run's, at `hl ≥ l` lanes: a reverse at
/// a layout whose lanes are a prefix of the forward's reads each body input's leading `l`
/// lanes, regenerated from the log's checkpoints ([`Regen`]). A discrete adjoint: every lane
/// operation of the run was a recorded body, so the log replayed backward through the bodies'
/// reverse passes is exact; each body's recomputed forward is shared by the seeds. An in-step
/// read-out is linear in the step's carried state and its start state, at its constant
/// `(theta, dt)`: its reverse is the transpose of that map, the coefficients read off the
/// method's [`sol::CarriedLayout`] by unit probes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop_rev(
    spec: &LoopSpec, rev: &LoopKernels, l: usize, inputs: &[f64], log: &Log, hl: usize,
    seeds: usize, out_adj: &[f64], in_adj: &mut [f64],
) -> bool {
    let n = spec.n;
    let in_w = in_adj.len() / seeds;
    let nl = n * l;
    let cap_w = spec.n_caps * l;
    let mut ubar = take_buf();
    ubar.resize(seeds * nl, 0.0);
    let mut capbar = take_buf();
    capbar.resize(seeds * cap_w, 0.0);
    if log.l != hl || log.seeds != 0 {
        return false;
    }
    let (mut cbar, mut base, mut adj) = (take_buf(), take_buf(), take_buf());
    let (mut probe, mut a_c) = (take_buf(), take_buf());
    let mut rg = Regen::new();
    for (i, e) in log.entries.iter().enumerate().rev() {
        match e {
            Entry::Point { .. } | Entry::Interp { .. } => rev_linear_entry(
                e, spec, l, seeds, out_adj, &mut ubar, &mut cbar, &mut probe, &mut a_c,
            ),
            Entry::NewStepper { .. } => cbar.clear(),
            Entry::Ckpt => {}
            // A jump (#2649): its new state's adjoint is the state's; its inputs' come back.
            Entry::Body { m, kind: 3, .. } => {
                let b = &spec.jumps[*m];
                let k = &rev.jumps[*m].0;
                let n_in = b.n_vars + b.caps.len();
                let n_slots = n_in + b.n_out + b.guarded as usize;
                let w = n_slots * l;
                base.clear();
                base.resize(w, 0.0);
                let at = rg.at(spec, log, i);
                for j in 0..b.n_vars {
                    base[j * l..(j + 1) * l]
                        .copy_from_slice(&rg.vars[at + j * hl..at + j * hl + l]);
                }
                for (j, &cp) in b.caps.iter().enumerate() {
                    base[(b.n_vars + j) * l..(b.n_vars + j + 1) * l]
                        .copy_from_slice(&inputs[(n + cp) * l..(n + cp + 1) * l]);
                }
                adj.clear();
                adj.resize(seeds * w, 0.0);
                for sd in 0..seeds {
                    adj[sd * w + n_in * l..sd * w + n_in * l + nl]
                        .copy_from_slice(&ubar[sd * nl..(sd + 1) * nl]);
                }
                // SAFETY: as the method bodies' reverse below.
                unsafe {
                    k.call_seeds(
                        base.as_mut_ptr(),
                        adj.as_mut_ptr(),
                        w,
                        seeds,
                        rev.iota.as_ptr(),
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
                    ubar[sd * nl..(sd + 1) * nl].copy_from_slice(&a[..nl]);
                }
            }
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
            Entry::Body { m, kind, .. } => {
                let mb = &spec.methods[*m];
                let c = mb.c;
                let cl = c * l;
                if cbar.len() != seeds * cl {
                    cbar.clear();
                    cbar.resize(seeds * cl, 0.0);
                }
                let (b, k) = match kind {
                    0 => (&mb.start, &rev.methods[*m][0]),
                    1 => (&mb.step, &rev.methods[*m][1]),
                    2 => (&mb.accept, &rev.methods[*m][2]),
                    // A forced step (#2649) is a step.
                    j => (&mb.forced[*j as usize - 4], &rev.forced[*m][*j as usize - 4].0),
                };
                let n_in = b.n_vars + b.caps.len();
                let n_slots = n_in + b.n_out + b.guarded as usize;
                let w = n_slots * l;
                base.clear();
                base.resize(w, 0.0);
                let at = rg.at(spec, log, i);
                for j in 0..b.n_vars {
                    base[j * l..(j + 1) * l]
                        .copy_from_slice(&rg.vars[at + j * hl..at + j * hl + l]);
                }
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
                // SAFETY: as `LoopRun::call`; each seed's `adj` block is laid out like `base`,
                // whose slots in order `rev.iota` names.
                unsafe {
                    k.call_seeds(
                        base.as_mut_ptr(),
                        adj.as_mut_ptr(),
                        w,
                        seeds,
                        rev.iota.as_ptr(),
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
                    let (ub, cb) =
                        (&mut ubar[sd * nl..(sd + 1) * nl], &mut cbar[sd * cl..(sd + 1) * cl]);
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
    for b in [ubar, capbar, cbar, base, adj, probe, a_c] {
        give_buf(b);
    }
    true
}

/// [`run_loop_rev`] for a rows forward (#2980, `super::Sweep::TangentReverseRows` kernels at
/// the base layout, `l` lanes): forward over reverse with the adjoint seed every seed shares.
/// The adjoints are blocks — block 0 their values, block `1 + s` seed `s`'s tangents of them —
/// and every step of the reverse that is linear in them (points, the continuous extension, a
/// fresh stepper, the inputs at the start) applies to each block as `run_loop_rev`'s seeds;
/// a body runs its reverse once and each seed's tangent of it, on the shared `[values |
/// adjoint values]` and each seed's `[tangents | adjoint tangents]` through a doubled slot
/// table. `inputs`/`tin` the values and per-seed tangents of the call's inputs, `logs` the
/// seeds' logs ([`run_loop_rows`]: the first holds the run's, every seed's tangents in its
/// checkpoint), `out_adj`/`in_adj` laid out by block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loop_rev_rows(
    spec: &LoopSpec, rev: &LoopKernels, l: usize, inputs: &[f64], tin: &[f64], logs: &[&Log],
    out_adj: &[f64], in_adj: &mut [f64],
) -> bool {
    let seeds = logs.len();
    let Some(log) = logs.first() else {
        return false;
    };
    if log.seeds != seeds || log.l != l {
        return false;
    }
    let blocks = 1 + seeds;
    let n = spec.n;
    let in_w = in_adj.len() / blocks;
    let nl = n * l;
    let cap_w = spec.n_caps * l;
    let w_in = (n + spec.n_caps) * l;
    let mut ubar = vec![0.0; blocks * nl];
    let mut capbar = vec![0.0; blocks * cap_w];
    let (mut cbar, mut probe, mut a_c) = (Vec::new(), Vec::new(), Vec::new());
    // A body's reverse: the shared buffer and each seed's, through the doubled table — buffers
    // reused across the run's bodies.
    let (mut sh, mut ts, mut tbl) = (Vec::new(), Vec::new(), Vec::new());
    let mut rg = Regen::new();
    for (i, e) in log.entries.iter().enumerate().rev() {
        match e {
            Entry::Point { .. } | Entry::Interp { .. } => rev_linear_entry(
                e, spec, l, blocks, out_adj, &mut ubar, &mut cbar, &mut probe, &mut a_c,
            ),
            Entry::NewStepper { .. } => cbar.clear(),
            Entry::Ckpt => {}
            Entry::Body { m, kind: 3, .. } => {
                let b = &spec.jumps[*m];
                let k = &rev.jumps[*m].0;
                let n_in = b.n_vars + b.caps.len();
                let at = rg.at(spec, log, i);
                let w = rows_body_inputs(
                    b,
                    l,
                    n,
                    w_in,
                    seeds,
                    &rg.vars[at..],
                    inputs,
                    tin,
                    &mut sh,
                    &mut ts,
                );
                for bk in 0..blocks {
                    let a = rows_block(&mut sh, &mut ts, w, bk);
                    a[n_in * l..n_in * l + nl].copy_from_slice(&ubar[bk * nl..(bk + 1) * nl]);
                }
                rows_body_call(k, b, l, seeds, &mut sh, &mut ts, &mut tbl);
                for bk in 0..blocks {
                    let a = rows_block(&mut sh, &mut ts, w, bk);
                    for (j, &cp) in b.caps.iter().enumerate() {
                        for x in 0..l {
                            capbar[bk * cap_w + cp * l + x] += a[(b.n_vars + j) * l + x];
                        }
                    }
                    ubar[bk * nl..(bk + 1) * nl].copy_from_slice(&a[..nl]);
                }
            }
            Entry::Begin => {
                for bk in 0..blocks {
                    for (x, &g) in in_adj[bk * in_w..bk * in_w + nl]
                        .iter_mut()
                        .zip(&ubar[bk * nl..(bk + 1) * nl])
                    {
                        *x += g;
                    }
                }
                break;
            }
            Entry::Body { m, kind, .. } => {
                let mb = &spec.methods[*m];
                let c = mb.c;
                let cl = c * l;
                if cbar.len() != blocks * cl {
                    cbar.clear();
                    cbar.resize(blocks * cl, 0.0);
                }
                let (b, k) = match kind {
                    0 => (&mb.start, &rev.methods[*m][0]),
                    1 => (&mb.step, &rev.methods[*m][1]),
                    2 => (&mb.accept, &rev.methods[*m][2]),
                    j => (&mb.forced[*j as usize - 4], &rev.forced[*m][*j as usize - 4].0),
                };
                let n_in = b.n_vars + b.caps.len();
                let accept = *kind == 2;
                let at = rg.at(spec, log, i);
                let w = rows_body_inputs(
                    b,
                    l,
                    n,
                    w_in,
                    seeds,
                    &rg.vars[at..],
                    inputs,
                    tin,
                    &mut sh,
                    &mut ts,
                );
                for bk in 0..blocks {
                    let a = rows_block(&mut sh, &mut ts, w, bk);
                    a[n_in * l..n_in * l + cl].copy_from_slice(&cbar[bk * cl..(bk + 1) * cl]);
                    if accept {
                        a[n_in * l + cl..n_in * l + cl + nl]
                            .copy_from_slice(&ubar[bk * nl..(bk + 1) * nl]);
                    }
                }
                rows_body_call(k, b, l, seeds, &mut sh, &mut ts, &mut tbl);
                for bk in 0..blocks {
                    let a = rows_block(&mut sh, &mut ts, w, bk);
                    for (j, &cp) in b.caps.iter().enumerate() {
                        for x in 0..l {
                            capbar[bk * cap_w + cp * l + x] += a[(b.n_vars + j) * l + x];
                        }
                    }
                    let (u, c) =
                        (&mut ubar[bk * nl..(bk + 1) * nl], &mut cbar[bk * cl..(bk + 1) * cl]);
                    if accept {
                        c.copy_from_slice(&a[..cl]);
                        u.fill(0.0);
                    } else {
                        for (x, &g) in u.iter_mut().zip(&a[..nl]) {
                            *x += g;
                        }
                        c.copy_from_slice(&a[nl..nl + cl]);
                    }
                }
            }
        }
    }
    for bk in 0..blocks {
        for (x, &g) in in_adj[bk * in_w + nl..(bk + 1) * in_w]
            .iter_mut()
            .zip(&capbar[bk * cap_w..(bk + 1) * cap_w])
        {
            *x += g;
        }
    }
    true
}

/// A rows reverse body's inputs (#2980): the shared `[values | ·]` and each seed's `[tangents |
/// ·]` from `vars` (per variable its values, then each seed's tangents: [`Regen`]), the
/// captures from the call's inputs; the adjoint halves zeroed. Returns the half width `w`
/// (`n_slots × l`).
#[allow(clippy::too_many_arguments)]
fn rows_body_inputs(
    b: &LoopBody, l: usize, n: usize, w_in: usize, seeds: usize, vars: &[f64], inputs: &[f64],
    tin: &[f64], sh: &mut Vec<f64>, ts: &mut Vec<f64>,
) -> usize {
    let hl = (1 + seeds) * l;
    let n_slots = b.n_vars + b.caps.len() + b.n_out + b.guarded as usize;
    let w = n_slots * l;
    sh.clear();
    sh.resize(2 * w, 0.0);
    ts.clear();
    ts.resize(seeds * 2 * w, 0.0);
    for j in 0..b.n_vars {
        sh[j * l..(j + 1) * l].copy_from_slice(&vars[j * hl..j * hl + l]);
        for sd in 0..seeds {
            let o = j * hl + (1 + sd) * l;
            ts[sd * 2 * w + j * l..sd * 2 * w + (j + 1) * l].copy_from_slice(&vars[o..o + l]);
        }
    }
    for (j, &cp) in b.caps.iter().enumerate() {
        let o = (b.n_vars + j) * l;
        sh[o..o + l].copy_from_slice(&inputs[(n + cp) * l..(n + cp + 1) * l]);
        for sd in 0..seeds {
            ts[sd * 2 * w + o..sd * 2 * w + o + l]
                .copy_from_slice(&tin[sd * w_in + (n + cp) * l..sd * w_in + (n + cp + 1) * l]);
        }
    }
    w
}

/// Block `bk`'s adjoint half: the shared buffer's for block 0, seed `bk − 1`'s after.
fn rows_block<'a>(sh: &'a mut [f64], ts: &'a mut [f64], w: usize, bk: usize) -> &'a mut [f64] {
    if bk == 0 {
        &mut sh[w..2 * w]
    } else {
        let o = (bk - 1) * 2 * w;
        &mut ts[o + w..o + 2 * w]
    }
}

/// A rows reverse body's kernel over the shared buffer and every seed's, through its doubled
/// table (built into `tbl`).
fn rows_body_call(
    k: &Tiered, b: &LoopBody, l: usize, seeds: usize, sh: &mut [f64], ts: &mut [f64],
    tbl: &mut Vec<u32>,
) {
    let n_slots = b.n_vars + b.caps.len() + b.n_out + b.guarded as usize;
    let iota: Vec<u32> = (0..n_slots as u32).collect();
    k.doubled(&iota, n_slots as u32, tbl);
    let t2 = &tbl[..];
    // SAFETY: `t2` names slots below `2·n_slots`, whose `l` lanes `sh` and each seed's block of
    // `ts` hold.
    unsafe {
        k.call_seeds(
            sh.as_mut_ptr(),
            ts.as_mut_ptr(),
            2 * n_slots * l,
            seeds,
            t2.as_ptr(),
            b.lits.as_ptr(),
        )
    };
}
