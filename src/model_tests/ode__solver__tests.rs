#[allow(unused_imports)]
use ferx_model::internal::*;
// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::sens::num::NonRecRead;
use ferx_model::sens::num::PkNum;

use approx::assert_relative_eq;
use ferx_model::ode::solver::*;

/// A hook that serves nothing: only its identity matters to the restore tests.
struct NullHook;
impl LaneHook for NullHook {
    fn stepper(&self, _: usize, _: OdeMethod) -> Box<dyn Stepper<f64>> {
        unreachable!()
    }
    fn begin(&self) {}
    fn point(&self, _: bool) {}
    fn interp(&self, _: f64, _: f64, _: usize) {}
    fn interp_done(&self) {}
    fn tail(&self, _: bool, _: bool) {}
    fn switches(&self) -> usize {
        0
    }
    fn force(&self, _: Option<(usize, bool)>) {}
    fn switch_g(&self, _: &[f64], _: f64, _: &mut [f64]) {}
    fn jump(&self, _: usize, _: bool, _: f64) {}
}

/// `with_lane_hook` puts the previous hook back on a panic in `f` too (#2786 review #8): a
/// rayon worker outlives the operation, and a stale hook would serve a dead operation's
/// stepper to every later `f64` solve on that thread. Nested, so both levels are checked:
/// the inner panic leaves the outer hook in force, the outer one's end leaves none. Dies
/// when the restore goes back to running only on a normal return.
#[test]
fn a_lane_hook_is_restored_when_its_closure_panics() {
    use std::rc::Rc;
    assert!(lane_hook::<f64>().is_none(), "a clean thread");
    let outer: Rc<dyn LaneHook> = Rc::new(NullHook);
    let inner: Rc<dyn LaneHook> = Rc::new(NullHook);
    let (o, i) = (outer.clone(), inner.clone());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_lane_hook(o.clone(), || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                with_lane_hook(i.clone(), || panic!("the inner operation fails"))
            }));
            assert!(r.is_err());
            let now = lane_hook::<f64>().expect("the outer hook is back in force");
            assert!(Rc::ptr_eq(&now, &o), "the outer hook, not the dead inner one");
            panic!("the outer operation fails");
        })
    }));
    assert!(r.is_err());
    assert!(lane_hook::<f64>().is_none(), "no hook outlives the panicked operation");
    // Normal return restores too, and the hook is not kept alive by the thread.
    with_lane_hook(outer.clone(), || ());
    assert!(lane_hook::<f64>().is_none());
    assert_eq!(Rc::strong_count(&outer), 2, "only `outer` and its clone `o`");
    assert_eq!(Rc::strong_count(&inner), 2);
}

/// A solve starts with no stale switch count: a right-hand side that said `k` switches in an
/// earlier evaluation on this thread must not make a switch-free solve watch any (#2786
/// review nit). Dies when the solve entry stops clearing it.
#[test]
fn a_solve_starts_with_no_stale_switch_count() {
    note_switches(3);
    let rhs = |u: &[f64], _: &[f64], _: f64, du: &mut [f64]| du[0] = -u[0];
    let out = solve_ode(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &OdeSolverOptions::default());
    assert!((out[0].u[0] - (-1.0f64).exp()).abs() < 1e-4);
    assert_eq!(take_switches(), 0, "the stale count was cleared at entry");
}

// -- Fit-scoped solver-option override (#1212) -------------------------------------
//
// Note for anyone adding to these: an armed override reaches the thread that armed it
// and the workers of the pool keyed to it, and nothing else - so these need no
// serialising against the rest of the binary, and adding a lock would hide the very
// property they exist to check. Two things still follow. Assert from the thread that
// should see the value (arming thread, or inside `pool.install`), since the *absence* of
// an override elsewhere is half of what is being pinned. Idle override pools are reused
// within a bounded worker cache; live leases retain independent fit budgets.

/// The whole merge rule rests on this: `ode_solver_override` treats "equal to the
/// `FitOptions` default" as "the caller expressed no opinion", which is only sound
/// while those defaults are the solver's own. If someone changes one side alone, a
/// caller asking for the new `FitOptions` default would start silently overriding a
/// model file (or stop being able to ask for the solver default at all).
#[test]
fn fit_option_ode_defaults_match_the_solver_defaults() {
    let f = crate::types::FitOptions::default();
    let s = OdeSolverOptions::default();
    assert_eq!(f.ode_reltol, s.reltol);
    assert_eq!(f.ode_abstol, s.abstol);
    assert_eq!(f.ode_max_steps, s.max_steps);
    assert_eq!(f.ode_method, s.method);
    assert_eq!(f.ode_stiff_abort_after, s.stiff_abort_after);
    assert_eq!(f.ode_auto_switch, s.auto_switch);
}

#[test]
fn untouched_fit_options_carry_no_override() {
    assert!(crate::types::FitOptions::default().ode_solver_override().is_empty());
}

/// The no-loosening half of #1212: a caller who never touched `ode_reltol` must not
/// pull a model file that pinned `1e-10` back to the `1e-4` default.
#[test]
fn an_empty_override_keeps_a_pinned_spec_value() {
    let baked = OdeSolverOptions { reltol: 1e-10, abstol: 1e-12, ..Default::default() };
    let merged = OdeSolverOverride::default().apply_to(baked);
    assert_eq!(merged.reltol, 1e-10);
    assert_eq!(merged.abstol, 1e-12);
}

/// ...and the half the issue was filed for: a moved field wins over the baked value,
/// while every field the caller left alone still comes off the spec.
#[test]
fn a_moved_field_wins_and_the_rest_stays_baked() {
    let opts = crate::types::FitOptions {
        ode_reltol: 1e-10,
        ode_method: OdeMethod::Rodas4,
        ode_stiff_abort_after: Some(7),
        ..Default::default()
    };
    let ov = opts.ode_solver_override();
    assert!(!ov.is_empty());
    let baked = OdeSolverOptions {
        reltol: 1e-4,
        abstol: 1e-9,
        max_steps: 55_555,
        method: OdeMethod::Rk45,
        ..Default::default()
    };
    let merged = ov.apply_to(baked);
    assert_eq!(merged.reltol, 1e-10);
    assert_eq!(merged.method, OdeMethod::Rodas4);
    assert_eq!(merged.stiff_abort_after, Some(7));
    // Untouched by the caller - these must not be reset to the *FitOptions* default.
    assert_eq!(merged.abstol, 1e-9);
    assert_eq!(merged.max_steps, 55_555);
    // And the fields no fit option addresses at all are carried through verbatim.
    assert_eq!(merged.initial_dt, baked.initial_dt);
    assert_eq!(merged.min_dt, baked.min_dt);
}

/// `effective_solver_options` is the read path every integration site goes through, and
/// the guard is what keeps a fit's setting from leaking into a later `predict`.
///
/// Nothing here needs serialising against the rest of the binary: the value lives on this
/// thread, so a concurrent test's arming is not visible and this one's is not visible to
/// it. That is the property, not a convenience.
#[test]
fn arming_applies_the_override_and_the_guard_restores_the_previous_one() {
    let baked = OdeSolverOptions::default();
    // Nothing armed: the spec's own value, untouched.
    assert_eq!(effective_solver_options(baked).reltol, baked.reltol);

    let outer =
        arm_ode_solver_override(OdeSolverOverride { reltol: Some(1e-9), ..Default::default() });
    assert_eq!(effective_solver_options(baked).reltol, 1e-9);
    {
        // A nested fit's opinion wins while it runs...
        let _inner = arm_ode_solver_override(OdeSolverOverride {
            reltol: Some(1e-11),
            ..Default::default()
        });
        assert_eq!(effective_solver_options(baked).reltol, 1e-11);
    }
    // ...and hands the setting back rather than disarming the outer fit.
    assert_eq!(effective_solver_options(baked).reltol, 1e-9);
    {
        // An *empty* override still shadows: "this fit expressed no opinion" has to
        // mean the spec's baked value, not the enclosing fit's tolerance.
        let _quiet = arm_ode_solver_override(OdeSolverOverride::default());
        assert_eq!(effective_solver_options(baked).reltol, baked.reltol);
        // ...and a compiled plan sees it as the same as no fit at all (a plan recorded on
        // a shared-pool worker serves the thread that armed an empty override).
        assert_eq!(current_ode_override(), None);
    }
    assert_eq!(effective_solver_options(baked).reltol, 1e-9);
    assert!(current_ode_override().is_some());
    drop(outer);
    assert_eq!(effective_solver_options(baked).reltol, baked.reltol);
}

/// Concurrent fits with different ODE settings must not see each other's — the case a
/// single process-global slot cannot express at all, and the one `ferx-tools` reaches by
/// running `fit()` from a `par_iter` while `pool` lets other callers fit at the same
/// time.
///
/// Two earlier shapes failed exactly here. A global slot restored on drop let one fit's
/// exit disarm another that was still running, and let the last exit leave a value armed
/// with no guard holding it. Publishing the last-armed value to a global mirror replaced
/// that with a quieter version of the same thing: a fit that expressed *no* opinion — the
/// default `FitOptions`, i.e. almost every fit in a process — read another fit's
/// tolerance, and two non-default fits overwrote each other. Neither is bounded in the
/// safe direction: `ode_reltol` takes any positive value, so `1e-2` can land on a fit that
/// asked for `1e-10`, and `ode_method` / `ode_max_steps` have no tighter-or-coarser
/// ordering to fall back on at all.
#[test]
fn concurrent_fits_do_not_see_each_others_overrides() {
    use std::sync::mpsc::channel;
    let baked = OdeSolverOptions::default();

    // Each thread arms, reports, waits until every thread has armed, then checks that it
    // still reads its own value — so the assertions all happen while all three arms and
    // one deliberately-empty arm are live at once.
    let (armed_tx, armed) = channel::<()>();
    let arms = [Some(1e-9), Some(1e-11), Some(1e-13), None];

    std::thread::scope(|s| {
        let mut releases = Vec::new();
        for want in arms {
            let (tx, release) = channel::<()>();
            releases.push(tx);
            let armed_tx = armed_tx.clone();
            s.spawn(move || {
                let _guard = arm_ode_solver_override(OdeSolverOverride {
                    reltol: want,
                    ..Default::default()
                });
                armed_tx.send(()).expect("the test thread is listening");
                release.recv().expect("the test thread releases this check");
                assert_eq!(
                    effective_solver_options(baked).reltol,
                    // An empty override reads as the spec's baked value, and must keep
                    // reading as that however tight a concurrent fit is running.
                    want.unwrap_or(baked.reltol),
                    "a concurrent fit's ODE settings reached this one"
                );
            });
        }
        for _ in arms {
            armed.recv().expect("every fit arms");
        }
        for release in releases {
            release.send(()).expect("a fit is waiting");
        }
    });

    // Nothing armed here at any point, and nothing leaked back.
    assert_eq!(effective_solver_options(baked).reltol, baked.reltol);
}

/// The other half of the same rule, and the one that needs the pool: a worker doing the
/// integrating for a fit is *not* the thread that armed. A worker of the shared pool
/// serves fits that asked for nothing, so it must read the spec's baked options even
/// while another thread holds a tight override; a worker of the pool built for an
/// override must read that override without anyone arming anything on it.
#[test]
fn only_the_pool_built_for_an_override_carries_it() {
    let baked = OdeSolverOptions::default();
    let ov = OdeSolverOverride { reltol: Some(1e-11), ..Default::default() };
    let _tight = arm_ode_solver_override(ov);

    // The shared pool's workers never install an override, so a default-options fit
    // running there is unaffected by the arming on this thread.
    if let Some(pool) = crate::pool::default_fit_pool() {
        pool.install(|| {
            assert_eq!(
                effective_solver_options(baked).reltol,
                baked.reltol,
                "a worker of the shared pool inherited another fit's override"
            );
        });
    }

    // The pool keyed to this override does carry it — on threads that armed nothing.
    let pool = crate::pool::fit_scope_pool(ov.into(), 2).expect("the override pool builds");
    pool.install(|| {
        assert_eq!(effective_solver_options(baked).reltol, 1e-11);
        // ...including inside a nested fan-out, which is where the real integrations
        // happen.
        let seen: Vec<f64> = crate::par::map(8, |_| effective_solver_options(baked).reltol);
        assert!(
            seen.iter().all(|&r| r == 1e-11),
            "a worker of the override pool integrated at the baked tolerance: {seen:?}"
        );
    });
}

/// Every field has to survive the trip onto a pool worker, since the pool is keyed by the
/// override and a key collision would hand a fit someone else's settings. The two cases a
/// "did anything change" assertion cannot catch are here: `stiff_abort_after` of
/// `Some(Some(1))` (abort on the first clamp; `Some(0)` is refused, #1770) must stay
/// distinct from `Some(None)` (no budget), and `auto_switch: Some(false)` from "unset".
#[test]
fn every_overridable_field_reaches_a_pool_worker() {
    let baked = OdeSolverOptions {
        reltol: 1e-4,
        abstol: 1e-6,
        max_steps: 10_000,
        method: OdeMethod::Auto,
        stiff_abort_after: Some(7),
        auto_switch: true,
        ..Default::default()
    };
    for method in [
        OdeMethod::Rk45,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
        OdeMethod::Vern7,
        OdeMethod::Auto,
    ] {
        let ov = OdeSolverOverride {
            reltol: Some(1e-9),
            abstol: Some(1e-12),
            max_steps: Some(123_456),
            method: Some(method),
            stiff_abort_after: Some(Some(1)),
            auto_switch: Some(false),
        };
        let pool = crate::pool::fit_scope_pool(ov.into(), 2).expect("the override pool builds");
        pool.install(|| {
            let eff = effective_solver_options(baked);
            assert_eq!(eff.reltol, 1e-9);
            assert_eq!(eff.abstol, 1e-12);
            assert_eq!(eff.max_steps, 123_456);
            assert_eq!(eff.method, method, "{method:?} did not reach the worker");
            assert_eq!(eff.stiff_abort_after, Some(1));
            assert!(!eff.auto_switch);
        });
    }

    // A budget of `None` is distinct from the field being unset: it has to overwrite the
    // baked `Some(7)` rather than leave it standing.
    let ov = OdeSolverOverride { stiff_abort_after: Some(None), ..Default::default() };
    let pool = crate::pool::fit_scope_pool(ov.into(), 2).expect("the override pool builds");
    pool.install(|| assert_eq!(effective_solver_options(baked).stiff_abort_after, None));
}

/// A call that lands on a thread already carrying its override must still name that
/// override's pool, not fall through to a plain one.
///
/// The trap is specific: skipping the pool "because this thread already has the value"
/// looks free, but the *fallback* for a pinned `threads` is a pool built without the
/// override, so a nested entry point that pinned threads would hand its fan-out to plain
/// workers and integrate at the model file's options — #1212, reached from the inside by
/// the one shape (`PoolPlan` pins `threads`) the tools actually use.
#[test]
fn a_nested_call_that_pins_threads_stays_on_the_override_pool() {
    let baked = OdeSolverOptions::default();
    let opts =
        crate::types::FitOptions { ode_reltol: 1e-11, threads: Some(2), ..Default::default() };
    let ov = opts.ode_solver_override();
    let pool = crate::pool::fit_scope_pool(ov.into(), 2).expect("the override pool builds");
    pool.install(|| {
        // Already a worker carrying `ov` — exactly the state that made the short-circuit
        // look safe.
        assert_eq!(effective_solver_options(baked).reltol, 1e-11);
        crate::pool::install_on_fit_pool(&opts, || {
            assert_eq!(
                effective_solver_options(baked).reltol,
                1e-11,
                "a nested pinned-threads call was handed a pool without the override"
            );
        })
        .expect("the nested call gets its pool");
    });
}

/// The pool table is keyed by the override, so two different settings must not land on one
/// pool — that would hand a fit another's tolerance through the very mechanism meant to
/// keep them apart. Simultaneous leases with the same settings must also remain distinct,
/// or a bootstrap's independently budgeted replicate fits would share one worker budget.
#[test]
fn the_override_pool_table_is_keyed_by_the_override() {
    let a = OdeSolverOverride { reltol: Some(1e-9), ..Default::default() };
    let b = OdeSolverOverride { reltol: Some(1e-10), ..Default::default() };
    let pool_a = crate::pool::fit_scope_pool(a.into(), 2).expect("pool a");
    let pool_b = crate::pool::fit_scope_pool(b.into(), 2).expect("pool b");
    let pool_a_again = crate::pool::fit_scope_pool(a.into(), 2).expect("pool a again");
    assert!(
        pool_a.install(|| std::thread::current().id())
            != pool_a_again.install(|| std::thread::current().id()),
        "concurrent fits with the same override must have independent workers"
    );
    assert!(
        pool_a.install(|| std::thread::current().id())
            != pool_b.install(|| std::thread::current().id()),
        "two different overrides share one pool, so its workers carry the wrong one"
    );
    // Width is part of the key too: a fit that pinned `threads` must not be handed a pool
    // of some other width.
    let narrow = crate::pool::fit_scope_pool(a.into(), 1).expect("pool a, one thread");
    assert!(
        pool_a.install(|| std::thread::current().id())
            != narrow.install(|| std::thread::current().id())
    );
    assert_eq!(narrow.current_num_threads(), 1);
}

#[test]
fn test_exponential_decay() {
    // du/dt = -k*u, u(0) = 1.0, k = 0.1
    // Exact: u(t) = exp(-0.1*t)
    let k = 0.1;
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -k * u[0];
    };
    let saveat = vec![1.0, 5.0, 10.0, 20.0];
    let opts = OdeSolverOptions::default();
    let result = solve_ode(&rhs, &[1.0], (0.0, 20.0), &[], &saveat, &opts);

    assert_eq!(result.len(), saveat.len());
    for (sol, &t) in result.iter().zip(saveat.iter()) {
        let exact = (-k * t).exp();
        assert_relative_eq!(sol.u[0], exact, epsilon = 1e-4);
        assert_relative_eq!(sol.t, t, epsilon = 1e-10);
    }
}

#[test]
fn test_linear_growth() {
    // du/dt = 1.0, u(0) = 0.0
    // Exact: u(t) = t
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let saveat = vec![1.0, 5.0, 10.0];
    let opts = OdeSolverOptions::default();
    let result = solve_ode(&rhs, &[0.0], (0.0, 10.0), &[], &saveat, &opts);

    for (sol, &t) in result.iter().zip(saveat.iter()) {
        assert_relative_eq!(sol.u[0], t, epsilon = 1e-6);
    }
}

#[test]
fn test_two_state_system() {
    // du1/dt = -u1, du2/dt = u1 (transfer from cpt 1 to cpt 2)
    // u1(t) = exp(-t), u2(t) = 1 - exp(-t)
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -u[0];
        du[1] = u[0];
    };
    let saveat = vec![1.0, 5.0, 10.0];
    let opts = OdeSolverOptions::default();
    let result = solve_ode(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &saveat, &opts);

    for (sol, &t) in result.iter().zip(saveat.iter()) {
        assert_relative_eq!(sol.u[0], (-t).exp(), epsilon = 1e-4);
        assert_relative_eq!(sol.u[1], 1.0 - (-t).exp(), epsilon = 1e-4);
    }
}

#[test]
fn test_zero_span_returns_initial() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let saveat = vec![5.0];
    let opts = OdeSolverOptions::default();
    let result = solve_ode(&rhs, &[42.0], (5.0, 5.0), &[], &saveat, &opts);
    assert_eq!(result.len(), 1);
    assert_relative_eq!(result[0].u[0], 42.0, epsilon = 1e-12);
}

// ---- #570: dense soft-sampling (`solve_ode_dense`) ----

/// The decisive #570 guarantee: requesting soft (Hermite) samples must **not**
/// change the hard `saveat` outputs. The soft channel only reads already-
/// accepted step data, so the adaptive step sequence — and thus every hard save
/// — is **bit-identical** to the no-soft-sampling wrapper. Without this, the
/// Gaussian predictions in a shared joint PK-TTE solve would move (the exact
/// regression #570 must avoid). Asserts exact f64 equality, not `approx`.
#[test]
fn solve_ode_dense_soft_samples_do_not_perturb_hard_saves() {
    // Two-state system (decay + a CHZ-like accumulator) so steps actually adapt.
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -0.3 * u[0];
        du[1] = 0.3 * u[0];
    };
    let saveat = vec![1.0, 3.0, 7.0, 12.0];
    let interp_at = vec![0.5, 2.2, 4.9, 8.1, 11.3];
    let opts = OdeSolverOptions::default();

    let reference = solve_ode(&rhs, &[1.0, 0.0], (0.0, 12.0), &[], &saveat, &opts);
    let (hard, soft) =
        solve_ode_dense(&rhs, &[1.0, 0.0], (0.0, 12.0), &[], &saveat, &interp_at, &opts, None);

    assert_eq!(hard.len(), saveat.len());
    assert_eq!(soft.len(), interp_at.len());
    for (a, b) in hard.iter().zip(reference.iter()) {
        assert_eq!(a.t, b.t);
        assert_eq!(a.u, b.u, "soft sampling perturbed a hard save");
    }
}

/// Soft samples reproduce the trajectory: pinned tightly against the analytic
/// solution at small step size (the interpolant itself), and shown to match an
/// independent clamped solve to within solver tolerance at production tolerance
/// (the real operating point of the joint PK-TTE share — CHZ read at event
/// times off the obs-clamped Gaussian solve instead of a dedicated TTE solve).
#[test]
fn solve_ode_dense_soft_samples_are_accurate() {
    // u0 = e^{-0.2 t}; u1 = 1 − e^{-0.2 t} (a cumulative accumulator).
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -0.2 * u[0];
        du[1] = 0.2 * u[0];
    };
    let saveat = vec![2.0, 6.0, 10.0];
    let interp_at = vec![1.0, 6.0, 8.5]; // between, on, and off the saveat grid

    // Tight tolerance ⇒ small steps ⇒ the Hermite read-back is near-exact, so
    // this pins the *interpolant* rather than the solver's own truncation.
    let tight = OdeSolverOptions { abstol: 1e-10, reltol: 1e-9, ..Default::default() };
    let (_h, soft) =
        solve_ode_dense(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &saveat, &interp_at, &tight, None);
    for (i, s) in soft.iter().enumerate() {
        let t = interp_at[i];
        assert_relative_eq!(s.t, t, epsilon = 1e-12);
        assert_relative_eq!(s.u[0], (-0.2 * t).exp(), epsilon = 1e-6);
        assert_relative_eq!(s.u[1], 1.0 - (-0.2 * t).exp(), epsilon = 1e-6);
    }

    // Production tolerance (reltol 1e-4): the read-back matches an independent
    // clamped solve to within solver tolerance even where the obs grid is coarse
    // (the 8.5 point sits inside a large 6→10 step). 1e-3 covers Hermite-over-a-
    // large-step error while still catching a broken interpolant (≫ 1e-3 off).
    let prod = OdeSolverOptions::default();
    let (_h2, soft2) =
        solve_ode_dense(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &saveat, &interp_at, &prod, None);
    let clamped = solve_ode(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &interp_at, &prod);
    for (i, s) in soft2.iter().enumerate() {
        assert_relative_eq!(s.u[0], clamped[i].u[0], epsilon = 1e-3);
        assert_relative_eq!(s.u[1], clamped[i].u[1], epsilon = 1e-3);
    }
}

/// Edge cases: a soft point coinciding with a `saveat`/step boundary lands at
/// `s = 1` (the exact saved state); a point at the span start clamps to `u0`.
#[test]
fn solve_ode_dense_soft_boundary_and_start() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -u[0];
    };
    let saveat = vec![4.0];
    let interp_at = vec![0.0, 4.0];
    let opts = OdeSolverOptions::default();
    let (hard, soft) =
        solve_ode_dense(&rhs, &[1.0], (0.0, 4.0), &[], &saveat, &interp_at, &opts, None);
    assert_eq!(soft.len(), 2);
    assert_relative_eq!(soft[0].u[0], 1.0, epsilon = 1e-12); // t=0 → u0
    assert_eq!(soft[1].u[0], hard[0].u[0]); // t=4 boundary == hard save (exact)
}

/// Zero-width span: the early-return path must populate the soft channel too,
/// keeping `soft.len() == interp_at.len()`.
#[test]
fn solve_ode_dense_zero_span_fills_soft() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = 1.0;
    let opts = OdeSolverOptions::default();
    let (hard, soft) =
        solve_ode_dense(&rhs, &[7.0], (3.0, 3.0), &[], &[3.0], &[3.0, 3.0], &opts, None);
    assert_eq!(hard.len(), 1);
    assert_eq!(soft.len(), 2);
    assert!(soft.iter().all(|p| (p.u[0] - 7.0).abs() < 1e-12));
}

/// Primitive-level degenerate oracle: a constant accumulation rate `λ` is the
/// ODE form of a constant hazard, whose analytic event time for `CHZ = −log u`
/// is `t* = −log(u)/λ`. Here `dCHZ/dt = 2`, `threshold = 10` ⇒ exact crossing at
/// `t* = 5`. The Hermite interpolant is exact on a linear trajectory, so the
/// bisection should hit `5` to round-off.
#[test]
fn until_threshold_constant_rate_is_exact() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 2.0;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [0.0];
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 100.0), &[], &opts, 0, 10.0) {
        ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 5.0, epsilon = 1e-9),
        other => panic!("expected Crossed(5.0), got {other:?}"),
    }
}

/// Drug-driven hazard with a closed form: a decaying drug `dC/dt = −0.5·C`,
/// `C(0)=10`, feeds `dCHZ/dt = 0.1·C` ⇒ `CHZ(t) = 2·(1 − e^{−0.5t})`. Solving
/// `CHZ = 1` gives `t* = 2·ln 2 ≈ 1.386294`. Exercises a non-zero monitor index
/// (`CHZ` is state 1) and a genuinely curved accumulator.
#[test]
fn until_threshold_decaying_drug_hazard_matches_closed_form() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -0.5 * u[0]; // concentration
        du[1] = 0.1 * u[0]; // cumulative hazard
    };
    let opts = OdeSolverOptions { abstol: 1e-10, reltol: 1e-9, ..OdeSolverOptions::default() };
    let mut u = [10.0, 0.0];
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1000.0), &[], &opts, 1, 1.0) {
        ThresholdCrossing::Crossed(t) => {
            assert_relative_eq!(t, 2.0 * std::f64::consts::LN_2, epsilon = 1e-5)
        }
        other => panic!("expected Crossed, got {other:?}"),
    }
}

/// Threshold never reached within the span ⇒ `ReachedEnd`, and `u` is advanced
/// in place to the end-of-span state (so a segmented caller can carry it).
#[test]
fn until_threshold_not_reached_advances_state() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [0.0];
    let outcome = solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 1000.0);
    assert_eq!(outcome, ThresholdCrossing::ReachedEnd);
    assert_relative_eq!(u[0], 10.0, epsilon = 1e-6);
}

/// The event-time driver honours `auto` too (#978), and has to: it is the path a
/// joint PK-TTE fit locates event times on, and a stiff hazard system left on the explicit
/// method exhausts its step budget and comes back `Failed` — which the TTE layer must not
/// launder into a censored subject.
///
/// `d/dt(fast) = -1e4·(fast − 1)` relaxes to 1 on a 1e-4 timescale, so the accumulator
/// `d/dt(chz) = fast` grows at ~1 per unit and crosses 4.0 just after t = 4. RK45 needs
/// `dt < 2e-4` for stability, so it runs out of its default 10 000 steps around t = 2 —
/// before the crossing. (The threshold is 4 and not 2 for exactly that reason: at 2 the
/// explicit method reaches the crossing on its last few steps and the contrast vanishes.)
#[test]
fn until_threshold_escalates_a_stiff_system_and_finds_the_crossing() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -1.0e4 * (u[0] - 1.0);
        du[1] = u[0];
    };

    let mut u = [0.0, 0.0];
    let auto = solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 5.0),
        &[],
        &OdeSolverOptions::default(),
        1,
        4.0,
    );
    match auto {
        ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 4.0001, epsilon = 1e-3),
        other => panic!("expected a crossing under auto, got {other:?}"),
    }

    // Non-vacuity, and the reason the escalation matters here: the same system on the
    // explicit method does not merely run slower, it fails outright.
    let mut u_rk = [0.0, 0.0];
    let explicit = solve_ode_until_threshold(
        &rhs,
        &mut u_rk,
        (0.0, 5.0),
        &[],
        &OdeSolverOptions { method: OdeMethod::Rk45, ..Default::default() },
        1,
        4.0,
    );
    assert!(
        matches!(explicit, ThresholdCrossing::Failed(_)),
        "RK45 was expected to fail on this stiff system, got {explicit:?}"
    );
}

/// The event-time driver's half of the `auto` guard: an escalation that comes back
/// `Failed` is re-run on the explicit method, from the **entry** state (`u` is unspecified
/// after a failure, so the retry cannot reuse it).
///
/// `d/dt(x) = x²` from `x₀ = 1e6` blows up at `t* = 1e-6`; both methods fail, which is the
/// point — what is under test is that the retry *happens*, not that it rescues this
/// system. Counting right-hand-side calls is what makes that observable: a failure the
/// guard did not retry would cost one solve, not two.
#[test]
fn until_threshold_retries_a_failed_escalation_on_the_explicit_method() {
    use std::cell::Cell;
    let calls = Cell::new(0usize);
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        calls.set(calls.get() + 1);
        du[0] = u[0] * u[0];
        du[1] = u[0];
    };
    let entry = [1.0e6, 0.0];

    let mut u = entry;
    let named = solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 1.0),
        &[],
        &OdeSolverOptions { method: OdeMethod::Rodas4, ..Default::default() },
        1,
        f64::INFINITY,
    );
    assert!(matches!(named, ThresholdCrossing::Failed(_)), "{named:?}");
    let named_calls = calls.get();

    calls.set(0);
    let mut u = entry;
    let auto = solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 1.0),
        &[],
        &OdeSolverOptions::default(),
        1,
        f64::INFINITY,
    );
    assert!(matches!(auto, ThresholdCrossing::Failed(_)), "{auto:?}");
    assert!(
        calls.get() > named_calls,
        "auto spent {} right-hand-side calls against rodas4's {named_calls} — the failed \
         escalation was not retried on the explicit method",
        calls.get()
    );
}

/// The quiet half of the event-time guard, and the reason it cannot key on
/// [`ThresholdCrossing::Failed`] alone.
///
/// The monitored accumulator here is `d/dt(chz) = 1` — exactly linear, independent of the
/// stiff state beside it. So when the stiff state cannot be stepped at `min_dt` the driver
/// force-accepts, keeps going, and the monitor sails through: finite, monotone, crossing
/// the threshold at exactly the right time. The outcome is an ordinary `Crossed` built on a
/// trajectory the solver never actually integrated (#959's freeze-pad, on the TTE path),
/// and no inspection of the outcome can tell. Counting right-hand-side calls is what makes
/// the re-solve observable.
#[test]
fn until_threshold_retries_an_escalation_that_stalled_but_still_reported_success() {
    use std::cell::Cell;
    let calls = Cell::new(0usize);
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        calls.set(calls.get() + 1);
        du[0] = -1.0e4 * (u[0] - 1.0);
        du[1] = 1.0;
    };
    // `min_dt = 1.0` is far above what this system's fast mode admits, so every step is a
    // force-accepted clamp.
    let stalling = OdeSolverOptions { initial_dt: 1.0, min_dt: 1.0, ..Default::default() };

    let mut u = [0.0, 0.0];
    let named = solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 5.0),
        &[],
        &OdeSolverOptions { method: OdeMethod::Rodas4, ..stalling },
        1,
        3.0,
    );
    assert!(
        matches!(named, ThresholdCrossing::Crossed(_)),
        "precondition: the stalled stiff solve must still *report success*, got {named:?}"
    );
    let named_calls = calls.get();

    calls.set(0);
    let mut u = [0.0, 0.0];
    let auto = solve_ode_until_threshold(&rhs, &mut u, (0.0, 5.0), &[], &stalling, 1, 3.0);
    assert!(matches!(auto, ThresholdCrossing::Crossed(_)), "{auto:?}");
    assert!(
        calls.get() > named_calls,
        "auto spent {} right-hand-side calls against rodas4's {named_calls} — a stalled \
         escalation that reported success was accepted instead of being re-solved",
        calls.get()
    );
}

/// #1564: a pooled RK45 stepper is a fresh one. A stepper that has stepped (and carried
/// `k7` into `k1`) goes back to the pool when dropped; the next `make_stepper` of that size
/// takes it, reset, and must take exactly the steps a never-used stepper takes. Mutation:
/// drop `have_k1 = false` from `reset` and the reused stepper reuses a stale `k1`.
#[test]
fn a_pooled_rk45_stepper_steps_as_a_fresh_one() {
    let rhs = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0];
        du[1] = p[0] * u[0] - 0.2 * u[1];
    };
    let opts = OdeSolverOptions::default();
    let run = |u0: [f64; 2], p: f64| -> Vec<u64> {
        let mut s = make_stepper::<f64>(2, OdeMethod::Rk45);
        let mut u = u0;
        let mut bits = Vec::new();
        for k in 0..3 {
            let err = s.attempt(&rhs, &u, &[p], 0.1 * k as f64, 0.1, &opts);
            bits.push(err.to_bits());
            u.copy_from_slice(s.u_new());
            bits.extend(u.iter().map(|x| x.to_bits()));
            s.on_accept();
        }
        bits
    };
    RK45_POOL.with(|p| p.borrow_mut().clear());
    let fresh = run([2.0, 0.0], 0.7);
    assert!(RK45_POOL.with(|p| !p.borrow().is_empty()), "a dropped stepper goes back to the pool");
    // A different solve in between leaves stale stages and a carried `k1` behind.
    let _other = run([5.0, 1.0], 1.9);
    assert_eq!(run([2.0, 0.0], 0.7), fresh, "the reused stepper took other steps");
}

/// Review r1 #1: a pool filled with steppers of other state sizes still takes the one
/// just dropped, evicting its oldest entry; refusing the push instead would end the
/// pooling for good once a thread had seen eight other shapes.
#[test]
fn a_full_rk45_pool_evicts_its_oldest_stepper() {
    RK45_POOL.with(|p| p.borrow_mut().clear());
    let held: Vec<_> = (10..10 + RK45_POOL_MAX).map(PooledRk45::<f64>::take).collect();
    drop(held);
    assert_eq!(RK45_POOL.with(|p| p.borrow().len()), RK45_POOL_MAX);
    drop(PooledRk45::<f64>::take(2));
    let sizes: Vec<usize> = RK45_POOL.with(|p| {
        p.borrow()
            .iter()
            .filter_map(|b| b.downcast_ref::<Rk45Stepper<f64>>().map(|s| s.n))
            .collect()
    });
    assert_eq!(sizes.len(), RK45_POOL_MAX);
    assert_eq!(sizes.last(), Some(&2), "the newest stepper is kept: {sizes:?}");
    assert!(!sizes.contains(&10), "the oldest one is evicted: {sizes:?}");
}

/// `Auto` never reaches [`make_stepper`] through a driver — every one resolves it first —
/// but the arm exists so a caller that builds a stepper straight from the options keeps
/// integrating instead of panicking inside a fit's worker thread. Pin that it really is the
/// explicit stepper and not a placeholder.
#[test]
fn make_stepper_falls_back_to_the_explicit_stepper_for_auto() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.3 * u[0];
    let opts = OdeSolverOptions::default();
    let u = [2.0];

    let mut from_auto = make_stepper::<f64>(1, OdeMethod::Auto);
    let mut from_rk45 = make_stepper::<f64>(1, OdeMethod::EXPLICIT_FALLBACK);
    let e_auto = from_auto.attempt(&rhs, &u, &[], 0.0, 0.1, &opts);
    let e_rk45 = from_rk45.attempt(&rhs, &u, &[], 0.0, 0.1, &opts);

    assert_eq!(e_auto, e_rk45);
    assert_eq!(from_auto.u_new(), from_rk45.u_new());
    assert_eq!(from_auto.err_exp(), from_rk45.err_exp());
}

/// A decreasing monitored state (negative rate ⇒ negative hazard) is not a
/// censor — it invalidates the crossing argument and must be a hard failure.
#[test]
fn until_threshold_negative_rate_fails() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -1.0;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [5.0];
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 100.0) {
        ThresholdCrossing::Failed(msg) => assert!(msg.contains("non-monotone"), "msg: {msg}"),
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// Already at/above the threshold at the span start ⇒ `Crossed(t0)` without
/// stepping (the `u ≈ 1 ⇒ threshold ≈ 0` edge, or a non-zero initial state).
#[test]
fn until_threshold_already_crossed_at_start() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [5.0];
    match solve_ode_until_threshold(&rhs, &mut u, (2.0, 10.0), &[], &opts, 0, 3.0) {
        ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 2.0, epsilon = 1e-12),
        other => panic!("expected Crossed(2.0), got {other:?}"),
    }
}

/// A non-finite trajectory cannot yield a meaningful crossing ⇒ hard failure,
/// not a silent censor at the horizon.
#[test]
fn until_threshold_non_finite_fails() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = f64::NAN;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [0.0];
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 10.0) {
        ThresholdCrossing::Failed(_) => {}
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// A zero-width span reaches its end immediately, without stepping.
#[test]
fn until_threshold_zero_span_reaches_end() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let opts = OdeSolverOptions::default();
    let mut u = [0.0];
    assert_eq!(
        solve_ode_until_threshold(&rhs, &mut u, (5.0, 5.0), &[], &opts, 0, 10.0),
        ThresholdCrossing::ReachedEnd
    );
}

/// Exhausting the step budget before reaching `t_end` is a failure, not a
/// censor — a censored draw requires a *clean* integration to the horizon.
#[test]
fn until_threshold_step_budget_exhausted_fails() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 1.0;
    };
    let opts = OdeSolverOptions { max_steps: 2, ..OdeSolverOptions::default() };
    let mut u = [0.0];
    // Threshold far beyond what 2 steps can reach within the (huge) span.
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1e6), &[], &opts, 0, 1e9) {
        ThresholdCrossing::Failed(msg) => assert!(msg.contains("budget"), "msg: {msg}"),
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// `solve_ode_g` over `Dual2` must reproduce the closed-form sensitivities of
/// `du/dt = −k·u`, `u(0)=1` ⇒ `u(t)=e^{−kt}`, `∂u/∂k=−t·e^{−kt}`,
/// `∂²u/∂k²=t²·e^{−kt}`.
#[test]
fn solve_ode_g_sensitivity_matches_closed_form() {
    use crate::sens::dual2::Dual2;
    let rhs = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = -(p[0] * u[0]);
    };
    let opts = OdeSolverOptions { abstol: 1e-10, reltol: 1e-9, ..OdeSolverOptions::default() };
    let k = Dual2::<1>::var(0.1, 0);
    let u0 = [Dual2::<1>::constant(1.0)];
    let res = solve_ode_g(&rhs, &u0, (0.0, 2.0), &[k], &[2.0], &opts);
    let u = &res[0].u[0];
    let e = (-0.2_f64).exp();
    assert_relative_eq!(u.value, e, max_relative = 1e-6);
    assert_relative_eq!(u.grad[0], -2.0 * e, max_relative = 1e-5);
    assert_relative_eq!(u.hess[0][0], 4.0 * e, max_relative = 1e-5);
}

/// The `Dual2` integration's value must track the scalar `solve_ode`.
#[test]
fn solve_ode_g_value_matches_scalar() {
    use crate::sens::dual2::Dual2;
    let rhs_d = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = -(p[0] * u[0]);
    };
    let rhs_f = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0];
    };
    let opts = OdeSolverOptions::default();
    let saveat = [1.0, 5.0, 10.0];
    let rd = solve_ode_g(
        &rhs_d,
        &[Dual2::<1>::constant(1.0)],
        (0.0, 10.0),
        &[Dual2::<1>::var(0.1, 0)],
        &saveat,
        &opts,
    );
    let rf = solve_ode(&rhs_f, &[1.0], (0.0, 10.0), &[0.1], &saveat, &opts);
    for (a, b) in rd.iter().zip(rf.iter()) {
        assert_relative_eq!(a.u[0].value, b.u[0], max_relative = 1e-9, epsilon = 1e-12);
    }
}

#[test]
fn solve_ode_with_stats_counts_attempts() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -10.0 * u[0];
    };
    let opts = OdeSolverOptions {
        initial_dt: 1.0,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(result.len(), 1);
    assert_eq!(stats.attempted_steps, stats.accepted_steps + stats.rejected_steps);
    assert!(stats.accepted_steps > 0, "stats = {stats:?}");
    assert!(stats.rejected_steps > 0, "stats = {stats:?}");
    assert_eq!(stats.min_step_clamped_steps, 0);
}

/// A segment can exhaust `max_steps` without ever touching `min_dt` — the fast-binding
/// failure shape recorded in #1080. The freeze-padded tail must have its own signal rather
/// than presenting the same stats as a completed solve.
#[test]
fn solve_ode_with_stats_counts_an_unfinished_segment_without_clamps() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -u[0];
    let opts = OdeSolverOptions {
        max_steps: 1,
        initial_dt: 0.1,
        method: OdeMethod::Rk45,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(stats.attempted_steps, 1);
    assert_eq!(stats.min_step_clamped_steps, 0);
    assert_eq!(stats.unfinished_segments, 1);
    assert_eq!(stats.discarded_unfinished_segments, 0);
    assert_eq!(result.len(), 1, "the saveat contract remains freeze-padded");
}

#[test]
fn solve_ode_with_stats_counts_min_step_clamped_accepts() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -100.0 * u[0];
    };
    // RK45 named rather than defaulted: `|Re λ|max = 100` here, so under the `auto`
    // default (#978) the probe escalates *and* the `min_dt = 1.0` clamp then trips the
    // guard into a second, explicit solve — which is correct behaviour and would double
    // every counter this test is asserting on. What is under test is the explicit
    // driver's clamp bookkeeping.
    let opts = OdeSolverOptions {
        initial_dt: 1.0,
        min_dt: 1.0,
        abstol: 1e-12,
        reltol: 1e-12,
        method: OdeMethod::Rk45,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let _ = solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(stats.attempted_steps, 1);
    assert_eq!(stats.accepted_steps, 1);
    assert_eq!(stats.rejected_steps, 0);
    assert_eq!(stats.min_step_clamped_steps, 1);
}

#[test]
fn solve_ode_with_stats_counts_nan_blowup_as_min_step_clamped() {
    // A diverging RHS produces a non-finite err_norm. Pinned at min_dt the
    // step is force-accepted to guarantee progress; the clamp counter must
    // still flag it (`!(err_norm <= 1.0)` catches NaN, where `err_norm > 1.0`
    // would not), so a NaN-diverging integration is not mistaken for clean.
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = f64::NAN;
    };
    let opts = OdeSolverOptions {
        initial_dt: 1.0,
        min_dt: 1.0,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let _ = solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(stats.attempted_steps, 1);
    assert_eq!(stats.accepted_steps, 1);
    assert_eq!(stats.rejected_steps, 0);
    assert_eq!(stats.min_step_clamped_steps, 1);
}

#[test]
fn solve_ode_breaks_repeated_min_step_clamps_before_max_steps() {
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = f64::NAN;
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-6,
        min_dt: 1e-6,
        max_steps: 10_000,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(result.len(), 1);
    assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
    assert_eq!(stats.min_step_clamped_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
}

/// Regression for #603 review #4: a *finite-but-stiff* min-step clamp (large finite local
/// error pinned at `min_dt`) must NOT trip the divergence break — only non-finite errors
/// do. Otherwise the remaining save points get frozen-padded with finite-but-wrong values
/// the likelihood would silently accept. High-frequency finite forcing keeps every step
/// clamped with a finite error, so the solve runs its full `max_steps` budget (>> the
/// clamp limit) and returns finite predictions rather than breaking at the limit.
#[test]
fn solve_ode_does_not_break_on_finite_stiff_min_step_clamps() {
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    // Ran the full budget — no early break at MAX_CONSECUTIVE_MIN_STEP_CLAMPS …
    assert_eq!(stats.attempted_steps, opts.max_steps);
    assert!(stats.attempted_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
    // … the segment really was in the stiff min-step-clamp regime …
    assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
    // … and the predictions stayed finite (not NaN-padded).
    assert_eq!(result.len(), 1);
    assert!(result[0].u[0].is_finite());
}

/// #708 / #1080 Part B: with a budget set, the same stability-limited segment stops at the
/// budget instead of grinding out its whole `max_steps` allowance.
///
/// Shares the forcing and options of
/// [`solve_ode_does_not_break_on_finite_stiff_min_step_clamps`] — a segment that clamps on
/// every step and runs the full 200 — so the only difference measured is the budget.
#[test]
fn stiff_abort_after_stops_a_clamping_segment_at_the_budget() {
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let base = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let budget = 5;
    let opts = OdeSolverOptions { stiff_abort_after: Some(budget), ..base };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));

    assert_eq!(stats.min_step_clamped_steps, budget as usize);
    assert_eq!(stats.attempted_steps, budget as usize);
    assert_eq!(stats.stiff_aborted_segments, 1);
    // The tail is freeze-padded, not dropped: the `saveat` contract holds, and the value
    // is finite — the same shape the unrecoverable-clamp break already produced.
    assert_eq!(result.len(), 1);
    assert!(result[0].u[0].is_finite());

    // …and the un-budgeted run is untouched: full `max_steps`, no abort recorded.
    let mut ground_out = OdeSolverStats::default();
    solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &base, Some(&mut ground_out));
    assert_eq!(ground_out.attempted_steps, base.max_steps);
    assert_eq!(ground_out.stiff_aborted_segments, 0);
}

/// A budget wider than the segment's clamp count is a no-op — the abort is a ceiling on
/// cost, not a second convergence criterion.
#[test]
fn stiff_abort_after_above_the_clamp_count_changes_nothing() {
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        stiff_abort_after: Some(10_000),
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    solve_ode_with_stats(&rhs, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut stats));
    assert_eq!(stats.attempted_steps, opts.max_steps);
    assert_eq!(stats.stiff_aborted_segments, 0);
}

/// A healthy segment never clamps, so no budget — however small — may touch it.
#[test]
fn stiff_abort_after_leaves_a_clean_solve_bit_identical() {
    let k = 0.1;
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -k * u[0];
    let saveat = [1.0, 5.0, 10.0];
    let base = OdeSolverOptions::default();
    let budgeted = OdeSolverOptions { stiff_abort_after: Some(1), ..base };
    let a = solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &saveat, &base);
    let b = solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &saveat, &budgeted);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.u[0].to_bits(), y.u[0].to_bits());
    }
}

/// The budget must be read from `.plain()` like every other driver decision, so the `f64`
/// prediction and the `Dual2` sensitivity solve abort on the *same* step — otherwise the
/// gradient would differentiate a trajectory the predictor never produced.
#[test]
fn stiff_abort_after_aborts_identically_on_the_dual_path() {
    use crate::sens::dual2::Dual2;
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        stiff_abort_after: Some(5),
        ..OdeSolverOptions::default()
    };
    let rhs_f64 = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let rhs_dual = |_u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
        du[0] = Dual2::constant(1e6 * (t * 1e7).sin());
    };
    let mut scalar = OdeSolverStats::default();
    solve_ode_with_stats(&rhs_f64, &[1.0], (0.0, 1.0), &[], &[1.0], &opts, Some(&mut scalar));
    let mut dual = OdeSolverStats::default();
    solve_ode_g_with_stats(
        &rhs_dual,
        &[Dual2::<1>::constant(1.0)],
        (0.0, 1.0),
        &[],
        &[1.0],
        &opts,
        Some(&mut dual),
    );
    assert_eq!(scalar, dual);
    assert_eq!(dual.stiff_aborted_segments, 1);
}

/// The threshold driver bails with a named `Failed` rather than freeze-padding: its output
/// *is* the answer (an event time, or a censoring), so a padded crossing would be laundered
/// into the likelihood as though it had been integrated.
#[test]
fn stiff_abort_after_fails_the_threshold_driver_instead_of_padding() {
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        // Monotone (a hazard accumulator) but violently forced, so every step clamps.
        du[0] = 1e6 * (1.0 + (t * 1e7).sin());
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        method: OdeMethod::Rk45,
        stiff_abort_after: Some(3),
        ..OdeSolverOptions::default()
    };
    let mut u = [0.0];
    let outcome = solve_ode_until_threshold(&rhs, &mut u, (0.0, 1.0), &[], &opts, 0, f64::INFINITY);
    match outcome {
        ThresholdCrossing::Failed(msg) => {
            assert!(msg.contains("ode_stiff_abort_after"), "{msg}");
        }
        other => panic!("expected an abort, got {other:?}"),
    }
}

/// The stats scope collects every integration on the thread, nests, and leaves nothing
/// behind — it is what the post-fit pass reads instead of threading a sink through every
/// predictor (#1080 Part B item 2).
#[test]
fn the_stats_scope_collects_and_restores() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.1 * u[0];
    let opts = OdeSolverOptions::default();
    let solve = || {
        solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &[10.0], &opts);
    };

    // No scope: nothing is recorded and nothing is left set.
    solve();
    assert!(!stats_sink_active());

    let outer = SolverStatsScope::enter();
    solve();
    let after_one = outer.collected();
    assert!(after_one.attempted_steps > 0);
    {
        let inner = SolverStatsScope::enter();
        solve();
        // The inner scope starts empty rather than inheriting the outer total …
        assert!(inner.collected().attempted_steps > 0);
        assert!(inner.collected().attempted_steps < after_one.attempted_steps * 2);
    }
    // … and dropping it restores the outer one, whose own total is unchanged by the
    // integrations that ran while it was shadowed.
    assert_eq!(outer.collected(), after_one);
    solve();
    assert_eq!(outer.collected().attempted_steps, after_one.attempted_steps * 2);
    drop(outer);
    assert!(!stats_sink_active());
}

/// A caller that passes its own `&mut OdeSolverStats` keeps getting exactly that, scope or
/// no scope: the sink is a tee, not a redirect.
#[test]
fn the_stats_scope_does_not_steal_a_callers_counters() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.1 * u[0];
    let opts = OdeSolverOptions::default();
    let mut without = OdeSolverStats::default();
    solve_ode_with_stats(&rhs, &[1.0], (0.0, 10.0), &[], &[10.0], &opts, Some(&mut without));

    let scope = SolverStatsScope::enter();
    let mut with = OdeSolverStats::default();
    solve_ode_with_stats(&rhs, &[1.0], (0.0, 10.0), &[], &[10.0], &opts, Some(&mut with));
    assert_eq!(with, without);
    assert_eq!(scope.collected(), without);
}

/// Review follow-up (#1080): the budget must count *abandoned* segments only. A segment
/// whose final step clamps still reaches the end of its span with every `saveat` saved —
/// breaking out there abandons nothing, and reporting it as truncated would make the
/// counter useless for the one question it exists to answer.
#[test]
fn a_clamp_on_the_last_step_is_not_an_abort() {
    // Forcing violent enough to clamp every step, over a span that `min_dt` steps can
    // still cross: 5 steps of 1e-3 reach `tf`, so the fifth clamp lands exactly on the
    // end of the segment.
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        stiff_abort_after: Some(5),
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result =
        solve_ode_with_stats(&rhs, &[1.0], (0.0, 5e-3), &[], &[5e-3], &opts, Some(&mut stats));

    assert_eq!(stats.min_step_clamped_steps, 5, "the fixture must clamp");
    assert_eq!(
        stats.stiff_aborted_segments, 0,
        "the segment reached t_end — nothing was abandoned"
    );
    assert_eq!(result.len(), 1);
    assert!(result[0].u[0].is_finite());
}

/// Review follow-up (#1080): a clamped step can still be the step that brackets the
/// crossing. The budget bounds cost; throwing away an answer the solver has already found
/// is not a cost saving, it is a worse outcome than the one being avoided.
#[test]
fn the_threshold_driver_keeps_a_crossing_found_on_a_clamped_step() {
    // Monotone and violently forced, so the first step clamps — and with a threshold this
    // low, that same first step crosses it.
    let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (1.0 + (t * 1e7).sin());
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        method: OdeMethod::Rk45,
        stiff_abort_after: Some(1),
        ..OdeSolverOptions::default()
    };
    let mut u = [0.0];
    match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1.0), &[], &opts, 0, 1.0) {
        ThresholdCrossing::Crossed(t) => {
            assert!(t > 0.0 && t <= 1e-3, "crossing outside the first step: {t}");
        }
        other => panic!("the crossing must survive the budget, got {other:?}"),
    }
}
/// Dual path mirrors [`solve_ode_does_not_break_on_finite_stiff_min_step_clamps`].
#[test]
fn solve_ode_g_does_not_break_on_finite_stiff_min_step_clamps() {
    use crate::sens::dual2::Dual2;
    let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
        du[0] = Dual2::constant(1e6 * (t * 1e7).sin());
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result = solve_ode_g_with_stats(
        &rhs,
        &[Dual2::<1>::constant(1.0)],
        (0.0, 1.0),
        &[],
        &[1.0],
        &opts,
        Some(&mut stats),
    );

    assert_eq!(stats.attempted_steps, opts.max_steps);
    assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
    assert_eq!(result.len(), 1);
    assert!(result[0].u[0].value.is_finite());
}

#[test]
fn solve_ode_g_breaks_repeated_min_step_clamps_like_scalar() {
    use crate::sens::dual2::Dual2;
    let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = Dual2::constant(f64::NAN);
    };
    let opts = OdeSolverOptions {
        initial_dt: 1e-6,
        min_dt: 1e-6,
        max_steps: 10_000,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let mut stats = OdeSolverStats::default();
    let result = solve_ode_g_with_stats(
        &rhs,
        &[Dual2::<1>::constant(1.0)],
        (0.0, 1.0),
        &[],
        &[1.0],
        &opts,
        Some(&mut stats),
    );

    assert_eq!(result.len(), 1);
    assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
    assert_eq!(stats.min_step_clamped_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
}

#[test]
fn solve_ode_g_with_stats_matches_scalar_step_pattern() {
    use crate::sens::dual2::Dual2;
    let rhs_f = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = -p[0] * u[0];
    };
    let rhs_d = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = -(p[0] * u[0]);
    };
    let opts = OdeSolverOptions {
        initial_dt: 1.0,
        abstol: 1e-12,
        reltol: 1e-12,
        ..OdeSolverOptions::default()
    };
    let saveat = [1.0];
    let mut stats_f = OdeSolverStats::default();
    let mut stats_d = OdeSolverStats::default();

    let _ = solve_ode_with_stats(
        &rhs_f,
        &[1.0],
        (0.0, 1.0),
        &[10.0],
        &saveat,
        &opts,
        Some(&mut stats_f),
    );
    let _ = solve_ode_g_with_stats(
        &rhs_d,
        &[Dual2::<1>::constant(1.0)],
        (0.0, 1.0),
        &[Dual2::<1>::var(10.0, 0)],
        &saveat,
        &opts,
        Some(&mut stats_d),
    );

    assert_eq!(stats_d, stats_f);
}

#[test]
fn test_params_passed_to_rhs() {
    // du/dt = p[0] * u, u(0) = 1
    // with p[0] = -0.5: u(t) = exp(-0.5*t)
    let rhs = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = p[0] * u[0];
    };
    let saveat = vec![2.0];
    let opts = OdeSolverOptions::default();
    let result = solve_ode(&rhs, &[1.0], (0.0, 2.0), &[-0.5], &saveat, &opts);
    assert_relative_eq!(result[0].u[0], (-1.0_f64).exp(), epsilon = 1e-4);
}

/// Regression guard for FSAL (First Same As Last) stage reuse.
///
/// Structural rather than count-based: with FSAL the k1 of step k+1 is
/// reused (swapped in) from the prior step's k7, so the rhs closure is
/// **never** invoked twice in a row at the same `(u, t)`. Without FSAL,
/// k7 of step k and k1 of step k+1 are two separate rhs calls at
/// bit-identical `(u_new, t_new)` — an adjacent duplicate in the call
/// sequence. Recording the `(u, t)` of every rhs call and scanning for
/// any adjacent duplicate detects FSAL removal regardless of iteration
/// count, controller, tolerance, or host platform.
///
/// The earlier modular check `(n - 1) % 6 == 0` was unsharp: FSAL-off
/// produces `n = 7N`, which satisfies the check whenever `N ≡ 1 (mod 6)`
/// — a 1-in-6 silent-pass rate across the population of iteration counts
/// the test might land on.
#[test]
fn test_fsal_reuses_last_stage() {
    use std::cell::RefCell;
    // Record `(u[0], t)` bit patterns at every rhs invocation. Bit
    // equality (rather than `==` on f64) sidesteps any ambiguity about
    // NaN / signed-zero corner cases — though for this smooth ODE there
    // are none.
    let calls: RefCell<Vec<(u64, u64)>> = RefCell::new(Vec::new());
    let rhs = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        calls.borrow_mut().push((u[0].to_bits(), t.to_bits()));
        du[0] = -0.1 * u[0];
    };
    let opts = OdeSolverOptions::default();
    let _ = solve_ode(&rhs, &[1.0], (0.0, 20.0), &[], &[20.0], &opts);
    let calls = calls.into_inner();

    assert!(calls.len() > 7, "solver did not perform multiple steps (calls = {})", calls.len(),);

    let dup_at = calls.windows(2).position(|w| w[0] == w[1]);
    assert!(
        dup_at.is_none(),
        "FSAL appears inactive: rhs called twice consecutively at the \
         same (u, t) at call index {} of {} (k7 of step k and k1 of \
         step k+1 should reuse a single evaluation).",
        dup_at.unwrap(),
        calls.len(),
    );
}

#[test]
fn terminal_stage_verdict_starts_next_event_interval_stiff() {
    let rhs = |u: &[f64], _: &[f64], _: f64, du: &mut [f64]| {
        du[0] = -10.0 * u[0];
    };
    let opts = OdeSolverOptions {
        method: OdeMethod::Auto,
        auto_switch: true,
        min_dt: 0.33,
        abstol: 1e-6,
        ..Default::default()
    };
    let mut state = OdeAutoSwitchState::default();
    let mut stats = OdeSolverStats::default();
    let mut u = vec![1e-12];

    for i in 0..15 {
        let start = i as f64 * 0.33;
        let end = (i + 1) as f64 * 0.33;
        let (sol, _) = solve_ode_dense_with_auto_state(
            &rhs,
            &u,
            (start, end),
            &[],
            &[end],
            &[],
            &opts,
            Some(&mut stats),
            &mut state,
        );
        u.clone_from(&sol.last().unwrap().u);
    }

    assert_eq!(stats.accepted_steps, 15);
    assert_eq!(stats.auto_switched_segments, 0);
    assert_eq!(stats.auto_stiff_segments, 0);
    assert_eq!(stats.min_step_clamped_steps, 0);
    assert_eq!(state.rk_stiff, RK45_STIFFNESS_RATIFY);

    let start = 15.0 * 0.33;
    let end = 16.0 * 0.33;
    let _ = solve_ode_dense_with_auto_state(
        &rhs,
        &u,
        (start, end),
        &[],
        &[end],
        &[],
        &opts,
        Some(&mut stats),
        &mut state,
    );

    assert_eq!(stats.accepted_steps, 16);
    assert_eq!(stats.auto_stiff_segments, 1);
    assert_eq!(state.rk_stiff, 0);
}

fn rk45_indicator(rate: f64, dt: f64) -> f64 {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = rate * u[0];
    };
    let opts = OdeSolverOptions::default();
    let mut stepper = Rk45Stepper::new(1);
    let _ = stepper.attempt(&rhs, &[1.0], &[], 0.0, dt, &opts);
    stepper.stiffness_indicator(dt).unwrap()
}

#[test]
fn rk45_stage_stiffness_indicator_is_dimensionless() {
    // The same decay written in hours and days has the same h*lambda and
    // therefore the same verdict. The old fixed Jacobian-rate threshold
    // changes by 24x under this conversion.
    let hours = rk45_indicator(-10.0, 0.1);
    let days = rk45_indicator(-240.0, 0.1 / 24.0);
    assert_relative_eq!(hours, days, epsilon = 1e-12);
}

#[test]
fn rk45_stiffness_votes_survive_event_intervals() {
    let mut state = OdeAutoSwitchState::default();

    // Think of each group as a short event-to-event solve. No one group
    // reaches Hairer's 15-vote ratification count, but the logical subject
    // solve does.
    for _ in 0..5 {
        assert!(!state.observe_rk45(RK45_STIFFNESS_THRESHOLD * 2.0));
    }
    for _ in 0..5 {
        assert!(!state.observe_rk45(RK45_STIFFNESS_THRESHOLD * 2.0));
    }
    for _ in 0..4 {
        assert!(!state.observe_rk45(RK45_STIFFNESS_THRESHOLD * 2.0));
    }
    assert!(state.observe_rk45(RK45_STIFFNESS_THRESHOLD * 2.0));

    state.reset_rk45_verdict();
    assert!(!state.observe_rk45(0.0));
}

// -- #1539: a diverged state NaN-pads the tail --------------------------------------
//
// Fixture: `central' = −0.1·central` (decoupled) next to `X' = 0.5·X²`, `X(0) = 1`, whose
// closed form `X = 1/(1 − 0.5t)` has a pole at t = 2. `central = 100·e^{−0.1t}` is finite
// and well-defined for every t, so a frozen `central` past the pole is a wrong value served
// as a real one — the #1539 defect — and the `0·X` control integrates it cleanly.

fn diverging_rhs(u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]) {
    du[0] = -0.1 * u[0];
    du[1] = 0.5 * u[1] * u[1];
}

fn control_rhs(u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]) {
    du[0] = -0.1 * u[0];
    du[1] = 0.0 * u[1];
}

/// T1: the saved and interpolated points past the pole are `NaN` in **every** component,
/// the decoupled `central` included; the pre-pole point is an integrated value.
///
/// Engine: `integrate_dense_g` at `T = f64`, through `solve_ode_g_dense` (both the
/// `saveat` and the `interp_at` tail). Mutations that redden it: revert the tail pad to
/// `u.clone()` (both legs); pad only the `saveat` loop (the interp leg).
#[test]
fn a_diverged_state_nan_pads_every_component_of_the_tail() {
    let opts = OdeSolverOptions::default();
    let mut stats = OdeSolverStats::default();
    let (saved, interp) = solve_ode_g_dense::<f64>(
        &diverging_rhs,
        &[100.0, 1.0],
        (0.0, 10.0),
        &[],
        &[1.0, 5.0, 10.0],
        &[0.8, 7.0],
        &opts,
        Some(&mut stats),
    );
    assert_eq!(saved.len(), 3);
    assert_eq!(interp.len(), 2);

    // Before the pole: integrated, against the closed forms.
    for (p, t) in [(&saved[0], 1.0), (&interp[0], 0.8)] {
        assert!(p.u.iter().all(|v| v.is_finite()), "pre-pole t={t}: {:?}", p.u);
        let central = 100.0 * (-0.1 * t).exp();
        let x = 1.0 / (1.0 - 0.5 * t);
        assert!((p.u[0] - central).abs() <= 1e-3 * central, "t={t}: {:?}", p.u);
        assert!((p.u[1] - x).abs() <= 1e-3 * x, "t={t}: {:?}", p.u);
    }
    // Past the pole: nothing was integrated, so nothing is a number — `central` included,
    // which is the component the freeze-pad used to serve as a finite 100·e^{−0.1·2}.
    for p in [&saved[1], &saved[2], &interp[1]] {
        assert!(
            p.u.iter().all(|v| v.is_nan()),
            "post-pole t={}: every component must be NaN, got {:?}",
            p.t,
            p.u
        );
    }
    assert_eq!(stats.diverged_segments, 1);
    assert!(stats.unfinished_segments >= stats.diverged_segments);
}

/// T2: a segment that **enters** already non-finite — the post-dose segment after a
/// divergence — comes back `NaN` in every component and is counted as a divergence, not as
/// a budgeted abort, under the default options and under `ode_stiff_abort_after`. Since
/// #2788 it is not stepped at all (`a_non_finite_entry_state_is_not_integrated`); the two
/// legs pin that the entry guard keeps the counts the stepper's breaks produced.
///
/// Engine: `integrate_resolved_g_inner`'s entry guard at `T = f64`, through
/// `solve_ode_with_stats`. Mutation that reddens it: drop the guard's stats record.
#[test]
fn the_nan_tail_covers_a_segment_that_enters_non_finite() {
    for (label, budget) in [("clamp break", None), ("stiff_abort_after", Some(2))] {
        let opts = OdeSolverOptions { stiff_abort_after: budget, ..OdeSolverOptions::default() };
        let mut stats = OdeSolverStats::default();
        let saved = solve_ode_with_stats(
            &diverging_rhs,
            &[100.0, f64::INFINITY],
            (0.0, 10.0),
            &[],
            &[1.0, 5.0],
            &opts,
            Some(&mut stats),
        );
        assert_eq!(saved.len(), 2, "{label}");
        for p in &saved {
            assert!(
                p.u.iter().all(|v| v.is_nan()),
                "{label}: t={} must be NaN in every component, got {:?}",
                p.t,
                p.u
            );
        }
        assert_eq!(stats.diverged_segments, 1, "{label}: {stats:?}");
        assert_eq!(stats.stiff_aborted_segments, 0, "{label}: {stats:?}");
    }
}

/// T3: the dual leg. A padded point carries `NaN` in the value **and every jet** — the
/// gradient and the Hessian — so the analytic-sensitivity solve cannot hand FOCEI a zero
/// derivative for a prediction that was never integrated. The pre-pole point keeps finite
/// jets, and its `∂central/∂k` matches the closed form `−t·central`.
///
/// Engine: `integrate_dense_g` at `T = Dual2<1>`, through `solve_ode_g_with_stats`.
/// Mutations that redden it: pad with `T::from_f64(NaN)` (a dual constant: `NaN` value,
/// zero jets); revert the pad to `u.clone()`.
#[test]
fn a_dual_nan_tail_is_nan_in_every_jet() {
    use crate::sens::dual2::Dual2;
    let rhs = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = -(p[0] * u[0]);
        du[1] = Dual2::constant(0.5) * u[1] * u[1];
    };
    let k = Dual2::<1>::var(0.1, 0);
    let mut stats = OdeSolverStats::default();
    let saved = solve_ode_g_with_stats(
        &rhs,
        &[Dual2::constant(100.0), Dual2::constant(1.0)],
        (0.0, 10.0),
        &[k],
        &[1.0, 5.0],
        &OdeSolverOptions::default(),
        Some(&mut stats),
    );
    assert_eq!(saved.len(), 2);
    let pre = &saved[0];
    for v in &pre.u {
        assert!(v.value.is_finite() && v.grad[0].is_finite() && v.hess[0][0].is_finite());
    }
    let central = 100.0 * (-0.1f64).exp();
    assert!((pre.u[0].grad[0] + central).abs() <= 1e-3 * central, "{:?}", pre.u[0]);
    for v in &saved[1].u {
        assert!(v.value.is_nan(), "value: {v:?}");
        assert!(v.grad[0].is_nan(), "gradient: {v:?}");
        assert!(v.hess[0][0].is_nan(), "Hessian: {v:?}");
    }
    assert_eq!(stats.diverged_segments, 1);
}

/// T4: the straddle, all three legs in one test so a pad stuck on either branch fails it.
///
/// * **diverge** — `X' = 0.5·X²`: the tail is `NaN`.
/// * **control** — `X' = 0·X`: finishes, never reaches the tail fill, and is bit-identical
///   to the value measured at `75f5aa83`, before #1539.
/// * **finite stop** — the finite-stiff forcing of
///   [`stiff_abort_after_stops_a_clamping_segment_at_the_budget`], stopped by the budget on
///   a finite state: still freeze-padded with the last state, bit-identical to `75f5aa83`
///   (#603 / #959 are not reopened).
///
/// Engine: `integrate_dense_g` at `T = f64`, through `solve_ode_with_stats`. Mutations that
/// redden it: an unconditional `NaN` pad (the finite-stop leg); no pad at all (the diverge
/// leg); a divergence test that reads any component rather than a non-finite one (the
/// control and finite-stop legs).
#[test]
fn the_nan_pad_straddles_diverged_and_finite_stops() {
    let opts = OdeSolverOptions::default();
    let saveat = [1.0, 5.0, 10.0];

    let mut diverged = OdeSolverStats::default();
    let d = solve_ode_with_stats(
        &diverging_rhs,
        &[100.0, 1.0],
        (0.0, 10.0),
        &[],
        &saveat,
        &opts,
        Some(&mut diverged),
    );
    assert!(d[0].u.iter().all(|v| v.is_finite()), "{:?}", d[0].u);
    assert!(d[2].u.iter().all(|v| v.is_nan()), "{:?}", d[2].u);
    assert_eq!(diverged.diverged_segments, 1);

    let mut control = OdeSolverStats::default();
    let c = solve_ode_with_stats(
        &control_rhs,
        &[100.0, 1.0],
        (0.0, 10.0),
        &[],
        &saveat,
        &opts,
        Some(&mut control),
    );
    // `[90.48374180413369, 60.65306897273051, 36.78829764176734]`, against the closed form
    // `100·e^{−0.1t}` = `[90.4837418, 60.6530660, 36.7879441]`.
    let got: Vec<u64> = c.iter().map(|p| p.u[0].to_bits()).collect();
    assert_eq!(
        got,
        [4636067644216122773, 4633732977897763958, 4630374309920469964],
        "control: {:?}",
        c.iter().map(|p| p.u[0]).collect::<Vec<_>>()
    );
    assert_eq!(control.unfinished_segments, 0);
    assert_eq!(control.diverged_segments, 0);
    assert!(c.iter().all(|p| p.u[1] == 1.0));

    let stiff = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        du[0] = 1e6 * (t * 1e7).sin();
    };
    let stiff_opts = OdeSolverOptions {
        initial_dt: 1e-3,
        min_dt: 1e-3,
        max_steps: 200,
        abstol: 1e-12,
        reltol: 1e-12,
        stiff_abort_after: Some(5),
        ..OdeSolverOptions::default()
    };
    let mut finite = OdeSolverStats::default();
    let f = solve_ode_with_stats(
        &stiff,
        &[1.0],
        (0.0, 1.0),
        &[],
        &[0.5, 1.0],
        &stiff_opts,
        Some(&mut finite),
    );
    // Both output times sit past the abort at t = 0.005, so both carry the frozen last
    // state, `453.1784280109106` — finite, and the same bits for each.
    for p in &f {
        assert_eq!(p.u[0].to_bits(), 4646680015408494802, "finite stop: {:?}", p.u);
    }
    assert_eq!(finite.stiff_aborted_segments, 1);
    assert_eq!(finite.diverged_segments, 0);
}

// -- #1578: a diverged unmonitored state is named, not blamed on a budget -----------
//
// Fixture: the #1539 pole `X' = 0.5·X²` next to a monitored accumulator `u0' = 0.01` whose
// threshold (1e9) is never reached. The monitored guard reads only `u[0]`, so the pole in
// `X` is force-accepted at `min_dt` until a budget runs out; that message must lead with
// the state, not with the budget or stiffness.

fn accumulator_next_to_pole(u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]) {
    du[0] = 0.01;
    du[1] = 0.5 * u[1] * u[1];
}

fn failed_text(outcome: ThresholdCrossing) -> String {
    match outcome {
        ThresholdCrossing::Failed(m) => m,
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// T1: the step-budget return names state 1 and keeps the budget clause.
///
/// Engine: `until_threshold_with` at `f64` (rk45). Mutation that reddens it: drop the
/// `with_nonfinite_state_note` call at the budget return.
#[test]
fn threshold_budget_failure_names_the_diverged_state() {
    let opts = OdeSolverOptions { method: OdeMethod::Rk45, ..Default::default() };
    let mut u = [0.0, 1.0];
    let msg = failed_text(solve_ode_until_threshold(
        &accumulator_next_to_pole,
        &mut u,
        (0.0, 10.0),
        &[],
        &opts,
        0,
        1e9,
    ));
    assert!(msg.starts_with("state 1 became non-finite before t=2.0"), "{msg}");
    assert!(msg.contains("; step budget ("), "{msg}");
    assert!(!msg.contains("state 0"), "{msg}");
}

/// T1b: the `ode_stiff_abort_after` return names state 1 and keeps the abort clause.
///
/// Engine: as T1. Mutation that reddens it: drop the helper call at the abort return only.
#[test]
fn threshold_stiff_abort_failure_names_the_diverged_state() {
    let opts = OdeSolverOptions {
        method: OdeMethod::Rk45,
        stiff_abort_after: Some(5),
        ..Default::default()
    };
    let mut u = [0.0, 1.0];
    let msg = failed_text(solve_ode_until_threshold(
        &accumulator_next_to_pole,
        &mut u,
        (0.0, 10.0),
        &[],
        &opts,
        0,
        1e9,
    ));
    assert!(msg.starts_with("state 1 became non-finite before t=2.0"), "{msg}");
    assert!(msg.contains("; aborted at t=2.0"), "{msg}");
    assert!(msg.contains("the segment is stability-limited"), "{msg}");
}

/// T1c twin: a budget exhausted on a **finite** state keeps today's text byte for byte.
/// `X' = 0` with `max_steps = 5` runs out of steps for want of steps, not because anything
/// diverged, independent of any stepper's stall behaviour.
///
/// Engine: as T1. Mutation that reddens it: make the note unconditional.
#[test]
fn threshold_budget_failure_on_a_finite_state_is_unchanged() {
    let flat = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = 0.01;
        du[1] = 0.0;
    };
    let opts = OdeSolverOptions { method: OdeMethod::Rk45, max_steps: 5, ..Default::default() };
    let mut u = [0.0, 1.0];
    let msg = failed_text(solve_ode_until_threshold(&flat, &mut u, (0.0, 1e6), &[], &opts, 0, 1e9));
    assert_eq!(
        msg, "step budget (5) exhausted before reaching t_end=1000000.000000 (reached t=78.100000)",
        "the finite-state budget text must be the pre-#1578 text"
    );
}

/// The helper's own two-state leg (the plural form), which no fixture above reaches.
#[test]
fn nonfinite_state_note_lists_every_non_finite_state() {
    let clause = "step budget (5) exhausted".to_string();
    assert_eq!(
        with_nonfinite_state_note(&[f64::NAN, 1.0, f64::INFINITY], 2.0, None, clause.clone()),
        "states 0, 2 became non-finite before t=2.000000; step budget (5) exhausted"
    );
    assert_eq!(with_nonfinite_state_note(&[0.0, 1.0], 2.0, None, clause.clone()), clause);
}

/// #2087 T5: with labels the note names the states by label, in index order, singular and
/// plural, with no `state(s)` prefix of its own (a label is a full noun phrase); a finite
/// `u` is still the clause alone.
///
/// Mutations that redden it: ignore `labels` (prints `0, 2`); index the labels by the
/// position in the non-finite list instead of the state index; keep the `state(s)` prefix
/// on the labelled form.
#[test]
fn nonfinite_state_note_names_states_by_label() {
    let labels: Vec<String> =
        ["[odes] state `a`", "[odes] state `b`", "the cumulative hazard of CMT=3"]
            .map(String::from)
            .to_vec();
    let clause = "step budget (5) exhausted".to_string();
    assert_eq!(
        with_nonfinite_state_note(
            &[f64::NAN, 1.0, f64::INFINITY],
            2.0,
            Some(&labels),
            clause.clone()
        ),
        "[odes] state `a`, the cumulative hazard of CMT=3 became non-finite before \
         t=2.000000; step budget (5) exhausted"
    );
    assert_eq!(
        with_nonfinite_state_note(&[1.0, f64::NAN, 1.0], 2.0, Some(&labels), clause.clone()),
        "[odes] state `b` became non-finite before t=2.000000; step budget (5) exhausted"
    );
    assert_eq!(
        with_nonfinite_state_note(&[0.0, 1.0, 2.0], 2.0, Some(&labels), clause.clone()),
        clause
    );
}

/// #2087 T5b: the labelled driver names the **monitored** state by its label in both of its
/// own failure sentences (non-finite, decreasing), and the unlabelled driver keeps the
/// `monitored state <k>` text, so the pair straddles the `labels` gate on the same input.
///
/// Engine: the `f64` threshold driver, `rk45`. Mutations that redden it: ignore `labels` in
/// either sentence (the labelled leg prints `monitored state 0`); print the label on the
/// public driver (the `None` leg loses `monitored state 0`).
#[test]
fn threshold_failed_names_the_monitored_state_by_label() {
    let labels: Vec<String> = vec!["the cumulative hazard".into()];
    let opts = OdeSolverOptions { method: OdeMethod::Rk45, ..Default::default() };
    let run = |rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), l: Option<&[String]>| {
        let mut u = [1.0];
        failed_text(solve_ode_until_threshold_labelled(
            rhs,
            &mut u,
            (0.0, 10.0),
            &[],
            &opts,
            0,
            1e300,
            l,
        ))
    };
    let blowup = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    let falling = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -1.0;

    let named = run(&blowup, Some(&labels));
    assert!(named.starts_with("the cumulative hazard became non-finite at t="), "{named}");
    let bare = run(&blowup, None);
    assert!(bare.starts_with("monitored state 0 became non-finite at t="), "{bare}");

    let named = run(&falling, Some(&labels));
    assert!(named.starts_with("the cumulative hazard decreased ("), "{named}");
    let bare = run(&falling, None);
    assert!(bare.starts_with("monitored state 0 decreased ("), "{bare}");
}

/// A segment entered with a non-finite state returns `NaN` at every save and interpolation
/// time without evaluating the RHS once (#2788): no stiffness probe, no stepper, no linear
/// extraction. Before, an all-`NaN` entry paid the probe, `MAX_CONSECUTIVE_MIN_STEP_CLAMPS`
/// forced steps and, on an escalation, the explicit re-solve too — 64% of the linear
/// segments the melphalan FOCEI plateau declined to the adaptive stepper.
///
/// Every method, because the guard sits above the one place they diverge; a partly
/// non-finite state too, which the steppers already turned into an all-`NaN` tail (#1539).
/// The finite control pins that the counter can see a solve, so `0` is not vacuous; the
/// zero-length span pins the gate's other side, where `u0` comes back untouched.
///
/// Mutation (run): delete the guard — every non-finite arm reports RHS calls and attempted
/// steps. The counts it records are pinned by `the_nan_tail_covers_a_segment_that_enters_non_finite`.
#[test]
fn a_non_finite_entry_state_is_not_integrated() {
    use std::cell::Cell;
    let calls = Cell::new(0usize);
    // A stiff two-compartment generator, so `auto` would escalate.
    let rhs = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
        calls.set(calls.get() + 1);
        du[0] = -p[0] * u[0] + p[1] * u[1];
        du[1] = p[0] * u[0] - p[1] * u[1];
    };
    let params = [1e6, 2.0];
    let saveat = [0.5, 1.0];
    let interp = [0.25];
    for method in [OdeMethod::Auto, OdeMethod::Rk45, OdeMethod::Rodas4, OdeMethod::Linear] {
        let opts = OdeSolverOptions { method, ..OdeSolverOptions::default() };
        for u0 in [[f64::NAN, f64::NAN], [f64::NAN, 1.0], [f64::INFINITY, 1.0]] {
            calls.set(0);
            let mut stats = OdeSolverStats::default();
            let mut auto_state = OdeAutoSwitchState::default();
            let (saved, interpolated) = integrate_resolved_g(
                &rhs,
                &u0,
                (0.0, 1.0),
                &params,
                &saveat,
                &interp,
                &opts,
                Some(&mut stats),
                &mut auto_state,
            );
            let label = format!("{method:?} from {u0:?}");
            assert_eq!(calls.get(), 0, "{label}: the RHS was evaluated");
            let diverged = OdeSolverStats {
                unfinished_segments: 1,
                diverged_segments: 1,
                ..OdeSolverStats::default()
            };
            assert_eq!(stats, diverged, "{label}");
            let times: Vec<f64> = saved.iter().map(|p| p.t).collect();
            assert_eq!(times, saveat, "{label}");
            assert_eq!(interpolated.len(), 1, "{label}");
            assert_eq!(interpolated[0].t, 0.25, "{label}");
            for p in saved.iter().chain(&interpolated) {
                assert!(p.u.iter().all(|v| v.is_nan()), "{label}: {:?}", p.u);
            }
        }

        calls.set(0);
        let mut stats = OdeSolverStats::default();
        let finite = integrate_resolved_g(
            &rhs,
            &[1.0, 0.0],
            (0.0, 1.0),
            &params,
            &saveat,
            &[],
            &opts,
            Some(&mut stats),
            &mut OdeAutoSwitchState::default(),
        );
        assert!(calls.get() > 0, "{method:?}: the control never reached the RHS");
        assert!(finite.0.iter().all(|p| p.u.iter().all(|v| v.is_finite())), "{method:?}");

        let zero_length = integrate_resolved_g(
            &rhs,
            &[f64::NAN, 1.0],
            (1.0, 1.0),
            &params,
            &[1.0],
            &[],
            &opts,
            None,
            &mut OdeAutoSwitchState::default(),
        );
        assert!(zero_length.0[0].u[0].is_nan(), "{method:?}");
        assert_eq!(zero_length.0[0].u[1], 1.0, "{method:?}");
    }
}

/// The dual instantiation's `NaN` reaches the jets as well as the value, as the diverged
/// tail's does (#1539): a constant `NaN` would carry a zero gradient into FOCEI.
///
/// Mutation (run): build the result from `T::from_f64(f64::NAN)` — the gradient reads 0.
#[test]
fn a_non_finite_dual_entry_state_carries_nan_jets() {
    use crate::sens::dual2::Dual2;
    let rhs = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
        du[0] = -(p[0] * u[0]);
    };
    let res = solve_ode_g(
        &rhs,
        &[Dual2::<1>::var(f64::NAN, 0)],
        (0.0, 1.0),
        &[Dual2::<1>::var(0.1, 0)],
        &[1.0],
        &OdeSolverOptions::default(),
    );
    assert!(res[0].u[0].value.is_nan());
    assert!(res[0].u[0].grad[0].is_nan());
    assert!(res[0].u[0].hess[0][0].is_nan());
}
