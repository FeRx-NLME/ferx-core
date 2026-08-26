//! Tier-1 tests for the a-priori stiffness probe and `ode_method = auto` (#978).

use super::*;
use crate::ode::solver::{solve_ode_with_stats, OdeSolverStats};
use crate::sens::dual1::Dual1;

/// `du/dt = A u` for a diagonal `A` — eigenvalues are the diagonal, exactly.
fn diagonal_rhs<T: PkNum>(rates: &'static [f64]) -> impl Fn(&[T], &[T], f64, &mut [T]) {
    move |u: &[T], _p: &[T], _t: f64, du: &mut [T]| {
        for (i, r) in rates.iter().enumerate() {
            du[i] = u[i] * T::from_f64(*r);
        }
    }
}

/// A three-state fast-binding system in the shape that motivates the whole feature: the fast
/// eigenvalue is carried by `KON · central`, so it is identically zero before any drug is
/// present and only appears once a dose lands (`[central, target, complex]`).
fn binding_rhs<T: PkNum>(kon: f64, koff: f64) -> impl Fn(&[T], &[T], f64, &mut [T]) {
    move |u: &[T], _p: &[T], _t: f64, du: &mut [T]| {
        let bind = u[0] * u[1] * T::from_f64(kon) - u[2] * T::from_f64(koff);
        du[0] = -u[0] * T::from_f64(0.1) - bind;
        du[1] = T::from_f64(0.5) - u[1] * T::from_f64(0.05) - bind;
        du[2] = bind - u[2] * T::from_f64(0.2);
    }
}

/// The same binding system behind a **first-order absorption depot**
/// (`[depot, central, target, complex]`), with the target *produced* rather than present at
/// dose time. Both factors of the fast term `KON · central · target` are zero at `t = 0`, so
/// the segment starts genuinely non-stiff and only becomes stiff once drug has been absorbed
/// and target has accumulated — the shape a segment-start probe cannot see and an in-place
/// switch exists for (#1080 Part C).
fn latent_binding_rhs<T: PkNum>(kon: f64, koff: f64) -> impl Fn(&[T], &[T], f64, &mut [T]) {
    move |u: &[T], _p: &[T], _t: f64, du: &mut [T]| {
        let absorb = u[0] * T::from_f64(1.0);
        let bind = u[1] * u[2] * T::from_f64(kon) - u[3] * T::from_f64(koff);
        du[0] = -absorb;
        du[1] = absorb - u[1] * T::from_f64(0.1) - bind;
        du[2] = T::from_f64(0.5) - u[2] * T::from_f64(0.05) - bind;
        du[3] = bind - u[3] * T::from_f64(0.2);
    }
}

fn loose() -> OdeSolverOptions {
    OdeSolverOptions {
        method: OdeMethod::Auto,
        ..Default::default()
    }
}

#[test]
fn eigenvalue_probe_matches_the_exact_spectrum_of_a_diagonal_system() {
    let rhs = diagonal_rhs::<f64>(&[-1.0, -1000.0, -0.5]);
    let lambda = max_abs_re_eigenvalue(&rhs, &[1.0, 1.0, 1.0], &[], 0.0).unwrap();
    // FD Jacobian of a linear system is exact up to round-off.
    assert!(
        (lambda - 1000.0).abs() < 1e-3,
        "expected |Re λ|max ≈ 1000, got {lambda}"
    );
}

#[test]
fn eigenvalue_probe_reads_the_fastest_mode_not_the_state_size() {
    // Same spectrum, states four orders of magnitude apart: `λ_max` must not move.
    let rhs = diagonal_rhs::<f64>(&[-2.0, -60.0]);
    let small = max_abs_re_eigenvalue(&rhs, &[1e-4, 1e-4], &[], 0.0).unwrap();
    let large = max_abs_re_eigenvalue(&rhs, &[1e4, 1e4], &[], 0.0).unwrap();
    assert!((small - 60.0).abs() < 1e-3, "{small}");
    assert!((large - 60.0).abs() < 1e-3, "{large}");
}

#[test]
fn eigenvalue_probe_separates_a_known_stiff_from_a_known_non_stiff_system() {
    // A one-compartment PK right-hand side: absorption + elimination, the regime where the
    // explicit default is the right answer.
    let pk = diagonal_rhs::<f64>(&[-1.2, -0.15]);
    let pk_lambda = max_abs_re_eigenvalue(&pk, &[100.0, 0.0], &[], 0.0).unwrap();
    assert!(
        pk_lambda < STIFF_RE_LAMBDA_THRESHOLD,
        "a 1-cpt PK model must not read stiff, got {pk_lambda}"
    );

    let binding = binding_rhs::<f64>(1000.0, 20.0);
    let stiff_lambda = max_abs_re_eigenvalue(&binding, &[5.0, 10.0, 0.0], &[], 0.0).unwrap();
    assert!(
        stiff_lambda >= STIFF_RE_LAMBDA_THRESHOLD,
        "a fast-binding model must read stiff, got {stiff_lambda}"
    );
    // …and with a wide margin either side of the threshold, not a coin flip on it.
    assert!(stiff_lambda / pk_lambda > 100.0);
}

#[test]
fn eigenvalue_probe_is_zero_for_a_constant_right_hand_side() {
    // `J = 0`: a pure zero-order input has no modes at all, so nothing to be stiff about.
    let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = 3.0;
    assert_eq!(max_abs_re_eigenvalue(&rhs, &[0.0], &[], 0.0), Some(0.0));
}

#[test]
fn eigenvalue_probe_declines_to_classify_a_non_finite_system() {
    let nan_rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = f64::NAN;
    assert_eq!(max_abs_re_eigenvalue(&nan_rhs, &[1.0], &[], 0.0), None);

    // Finite at `u`, non-finite at the perturbed state the Jacobian needs.
    let edge_rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
        du[0] = if u[0] > 1.0 { f64::INFINITY } else { -u[0] };
    };
    assert_eq!(max_abs_re_eigenvalue(&edge_rhs, &[1.0], &[], 0.0), None);

    // An empty system is not classifiable either (and must not panic in the eigensolve).
    let empty = |_u: &[f64], _p: &[f64], _t: f64, _du: &mut [f64]| {};
    assert_eq!(max_abs_re_eigenvalue(&empty, &[], &[], 0.0), None);
}

#[test]
fn the_evaluation_point_is_what_makes_binding_stiffness_visible() {
    // The property the per-segment evaluation exists for: a fast-binding model is at its
    // *least* stiff at the declared initial condition, because `KON · central` is zero there.
    // Probing the declared state alone would call this model non-stiff; probing the state each
    // segment actually starts from sees the post-dose Jacobian.
    let rhs = binding_rhs::<f64>(1000.0, 20.0);
    let pre_dose = max_abs_re_eigenvalue(&rhs, &[0.0, 10.0, 0.0], &[], 0.0).unwrap();
    let post_dose = max_abs_re_eigenvalue(&rhs, &[50.0, 10.0, 0.0], &[], 0.0).unwrap();
    assert!(post_dose > 4.0 * pre_dose, "{pre_dose} → {post_dose}");

    let opts = loose();
    assert_eq!(
        resolve_method(&rhs, &[50.0, 10.0, 0.0], &[], 0.0, &opts),
        OdeMethod::Rodas4,
        "the post-dose state must escalate"
    );
}

#[test]
fn a_named_method_is_never_second_guessed_by_the_probe() {
    // Every non-`Auto` method comes back unchanged, on a system the probe would happily call
    // stiff — asking for a solver is asking for that solver.
    let rhs = binding_rhs::<f64>(1000.0, 20.0);
    for named in [
        OdeMethod::Rk45,
        OdeMethod::Vern7,
        OdeMethod::Rosenbrock23,
        OdeMethod::Rodas4,
        OdeMethod::Rodas5P,
    ] {
        let opts = OdeSolverOptions {
            method: named,
            ..Default::default()
        };
        assert_eq!(
            resolve_method(&rhs, &[50.0, 10.0, 0.0], &[], 0.0, &opts),
            named
        );
    }
}

#[test]
fn auto_keeps_the_explicit_default_on_a_non_stiff_or_unclassifiable_system() {
    let pk = diagonal_rhs::<f64>(&[-1.2, -0.15]);
    assert_eq!(
        resolve_method(&pk, &[100.0, 0.0], &[], 0.0, &loose()),
        OdeMethod::Rk45
    );

    // Unclassifiable must fall back to explicit, not guess a stiff method: escalating on a
    // system whose Jacobian could not even be formed is how `auto` would trade a slow answer
    // for a wrong one.
    let nan_rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = f64::NAN;
    assert_eq!(
        resolve_method(&nan_rhs, &[1.0], &[], 0.0, &loose()),
        OdeMethod::Rk45
    );
}

#[test]
fn auto_picks_the_stiff_method_the_tolerance_calls_for() {
    let rhs = binding_rhs::<f64>(1000.0, 20.0);
    let u = [50.0, 10.0, 0.0];

    let at = |reltol: f64| {
        let opts = OdeSolverOptions {
            method: OdeMethod::Auto,
            reltol,
            ..Default::default()
        };
        resolve_method(&rhs, &u, &[], 0.0, &opts)
    };
    // The band boundary is pinned because it is a *consistency* requirement, not a free
    // choice: `OdeMethod`'s docs send a user choosing by hand to `rodas5p` at
    // `ode_reltol <= 1e-9`, and `auto` disagreeing with that would be a gratuitous way for a
    // probed model and a hand-written control stream to land on different steppers.
    assert_eq!(at(1e-4), OdeMethod::Rodas4, "PK-default tolerance");
    assert_eq!(at(1e-8), OdeMethod::Rodas4, "still the workhorse band");
    assert_eq!(at(1e-9), OdeMethod::Rodas5P, "the documented rodas5p cut");
    assert_eq!(at(1e-12), OdeMethod::Rodas5P);
}

#[test]
fn the_probe_decides_identically_at_f64_and_at_a_dual() {
    // The consistency the analytic gradient depends on: the `Dual2`/`Dual1` sensitivity solve
    // must land on the same stepper as the `f64` prediction for the same segment, which holds
    // because the probe reads values only. Seeding a *derivative* on the state is exactly the
    // perturbation that would break it if the probe ever read a jet.
    const N: usize = 2;
    let scalar_rhs = binding_rhs::<f64>(1000.0, 20.0);
    let dual_rhs = binding_rhs::<Dual1<N>>(1000.0, 20.0);

    for u_central in [0.0, 0.05, 5.0, 50.0] {
        let u_f64 = [u_central, 10.0, 0.0];
        let u_dual = [
            Dual1::<N>::var(u_central, 0),
            Dual1::<N>::var(10.0, 1),
            Dual1::<N>::constant(0.0),
        ];
        let l_f64 = max_abs_re_eigenvalue(&scalar_rhs, &u_f64, &[], 0.0).unwrap();
        let l_dual = max_abs_re_eigenvalue(&dual_rhs, &u_dual, &[], 0.0).unwrap();
        assert_eq!(
            l_f64, l_dual,
            "λ must not depend on T (central={u_central})"
        );
        assert_eq!(
            resolve_method(&scalar_rhs, &u_f64, &[], 0.0, &loose()),
            resolve_method(&dual_rhs, &u_dual, &[], 0.0, &loose()),
            "method must not depend on T (central={u_central})"
        );
    }
}

#[test]
fn auto_integrates_a_stiff_system_that_the_explicit_default_cannot_step_through() {
    // `du/dt = -1e4 · u` over 5 time units: an explicit method is stability-limited here,
    // a Rosenbrock method is not. Both must land on the analytical answer; the point of the
    // assertion is that `auto` gets there having *chosen* the stiff method.
    let rhs = diagonal_rhs::<f64>(&[-1.0e4, -0.5]);
    let saveat = [1.0, 5.0];
    let mut stats = OdeSolverStats::default();
    let sol = solve_ode_with_stats(
        &rhs,
        &[1.0, 1.0],
        (0.0, 5.0),
        &[],
        &saveat,
        &loose(),
        Some(&mut stats),
    );
    assert_eq!(
        stats.auto_stiff_segments, 1,
        "the probe must have escalated"
    );
    assert_eq!(
        stats.auto_stiff_rejected, 0,
        "and the result must be usable"
    );
    for (p, &t) in sol.iter().zip(saveat.iter()) {
        assert!(
            (p.u[1] - (-0.5 * t).exp()).abs() < 1e-5,
            "slow mode wrong at t={t}: {}",
            p.u[1]
        );
        assert!(p.u[0].abs() < 1e-6, "fast mode should be dead: {}", p.u[0]);
    }
}

#[test]
fn auto_leaves_the_stats_counters_alone_under_a_named_method() {
    // The counters exist to say "`auto` did something"; a fixed method must never move them,
    // including a fixed *stiff* method on a system the probe would have escalated.
    for named in [OdeMethod::Rk45, OdeMethod::Rodas4] {
        let opts = OdeSolverOptions {
            method: named,
            ..Default::default()
        };
        let mut stats = OdeSolverStats::default();
        let rhs = diagonal_rhs::<f64>(&[-1.0e4, -0.5]);
        solve_ode_with_stats(
            &rhs,
            &[1.0, 1.0],
            (0.0, 5.0),
            &[],
            &[5.0],
            &opts,
            Some(&mut stats),
        );
        assert_eq!(stats.auto_stiff_segments, 0, "{named:?}");
        assert_eq!(stats.auto_stiff_rejected, 0, "{named:?}");
    }
}

#[test]
fn the_guard_re_solves_an_escalation_that_stalled_at_min_dt() {
    // `du/dt = u²` from `u₀ = 1e6` blows up at `t* = 1e-6`, long before the requested horizon.
    // The probe reads `|Re λ|max = 2e6` and escalates; the stiff method then clamps at
    // `min_dt`, and the driver freeze-pads the tail — so the escalated answer comes back
    // *finite* (4.8e282) and wrong. That is the case the guard exists for, and the reason it
    // cannot be a non-finite test alone: the escalated result is discarded and the explicit
    // re-solve is what the caller receives.
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    let mut stats = OdeSolverStats::default();
    let sol = solve_ode_with_stats(
        &rhs,
        &[1.0e6],
        (0.0, 1.0),
        &[],
        &[1.0],
        &loose(),
        Some(&mut stats),
    );
    assert_eq!(stats.auto_stiff_segments, 1);
    assert_eq!(
        stats.auto_stiff_rejected, 1,
        "the stalled escalation must have been rejected"
    );
    assert_eq!(sol.len(), 1);

    // The guard is scoped to escalations: the same system under a named stiff method keeps
    // that method's own (non-finite) answer rather than being silently re-solved.
    let named = OdeSolverOptions {
        method: OdeMethod::Rodas4,
        ..Default::default()
    };
    let mut named_stats = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0e6],
        (0.0, 1.0),
        &[],
        &[1.0],
        &named,
        Some(&mut named_stats),
    );
    assert_eq!(named_stats.auto_stiff_rejected, 0);
}

/// `stiff_abort_after` and the `auto` guard compose the way they must: an escalation that
/// trips the budget is still *discarded* (the guard's trigger is `min_step_clamped_steps > 0`,
/// which an abort has by construction), so the budget makes the failed escalation cheaper
/// without making the caller keep its freeze-padded trajectory.
#[test]
fn a_budgeted_abort_inside_an_escalation_is_still_rejected_by_the_guard() {
    // Same blow-up fixture as `the_guard_re_solves_an_escalation_that_stalled_at_min_dt`.
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    let mut unbudgeted = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0e6],
        (0.0, 1.0),
        &[],
        &[1.0],
        &loose(),
        Some(&mut unbudgeted),
    );

    let budgeted_opts = OdeSolverOptions {
        stiff_abort_after: Some(1),
        ..loose()
    };
    let mut budgeted = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0e6],
        (0.0, 1.0),
        &[],
        &[1.0],
        &budgeted_opts,
        Some(&mut budgeted),
    );

    assert_eq!(budgeted.auto_stiff_segments, 1);
    assert_eq!(
        budgeted.auto_stiff_rejected, 1,
        "an aborted escalation is a stalled escalation"
    );
    // The abort that is *counted* is the explicit re-solve's: this system blows up under any
    // method, so the fallback trips the same budget. The discarded escalation's own abort is
    // deliberately not counted — nobody received that trajectory.
    assert!(budgeted.stiff_aborted_segments >= 1);
    assert!(budgeted.discarded_clamped_steps > 0);
    assert!(
        budgeted.attempted_steps < unbudgeted.attempted_steps,
        "the budget must cost fewer steps: {} vs {}",
        budgeted.attempted_steps,
        unbudgeted.attempted_steps
    );
}

/// The Gershgorin fast path is a *bound*, so it must never sit below the exact `|Re λ|max` —
/// otherwise it would short-circuit a genuinely stiff segment into the explicit method.
#[test]
fn the_gershgorin_bound_never_undercuts_the_exact_spectrum() {
    // Diagonal (bound is exact), strongly coupled, and a rotation-like block whose
    // eigenvalues are complex — the shape where a diagonal-only guess would be badly wrong.
    let cases: Vec<(&str, Box<dyn Fn(&[f64], &[f64], f64, &mut [f64])>, Vec<f64>)> = vec![
        (
            "diagonal",
            Box::new(|u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
                du[0] = -1.0 * u[0];
                du[1] = -1000.0 * u[1];
            }),
            vec![1.0, 1.0],
        ),
        (
            "coupled",
            Box::new(|u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
                du[0] = -2.0 * u[0] + 50.0 * u[1];
                du[1] = 40.0 * u[0] - 3.0 * u[1];
            }),
            vec![1.0, 1.0],
        ),
        (
            "rotation",
            Box::new(|u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
                du[0] = -1.0 * u[0] - 80.0 * u[1];
                du[1] = 80.0 * u[0] - 1.0 * u[1];
            }),
            vec![1.0, 1.0],
        ),
        (
            "binding",
            Box::new(binding_rhs::<f64>(100.0, 1.0)),
            vec![100.0, 10.0, 0.0],
        ),
    ];
    for (name, rhs, u) in cases {
        let exact = max_abs_re_eigenvalue(&rhs, &u, &[], 0.0).unwrap();
        let jac = fd_jacobian(&rhs, &u, &[], 0.0).unwrap();
        let bound = gershgorin_abs_bound(&jac, u.len());
        assert!(
            bound >= exact * (1.0 - 1e-9),
            "{name}: bound {bound} below exact {exact}"
        );
    }
}

/// …and the fast path must not change a single verdict: across five decades of the binding
/// rate — the sweep that motivated the probe — `resolve_method` and the exact eigensolve agree
/// on every segment.
#[test]
fn the_gershgorin_fast_path_agrees_with_the_exact_eigensolve() {
    for kon in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
        let rhs = binding_rhs::<f64>(kon, 1.0);
        for u in [[0.0, 10.0, 0.0], [100.0, 10.0, 0.0], [1.0, 0.1, 5.0]] {
            let opts = loose();
            let exact = max_abs_re_eigenvalue(&rhs, &u, &[], 0.0).unwrap();
            let expected = if exact >= STIFF_RE_LAMBDA_THRESHOLD {
                OdeMethod::Rodas4
            } else {
                OdeMethod::EXPLICIT_FALLBACK
            };
            assert_eq!(
                resolve_method(&rhs, &u, &[], 0.0, &opts),
                expected,
                "kon={kon} u={u:?} lambda={exact}"
            );
        }
    }
}

/// Review follow-up (#1080): a discarded escalation's clamps are attributed to the *discarded*
/// bucket, and a budgeted abort inside one is not reported as a truncated result — the caller
/// received the explicit re-solve, not the attempt that was abandoned.
#[test]
fn a_discarded_escalations_clamps_are_recorded_as_discarded() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    let opts = OdeSolverOptions {
        stiff_abort_after: Some(1),
        ..loose()
    };
    let mut stats = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0e6],
        (0.0, 1.0),
        &[],
        &[1.0],
        &opts,
        Some(&mut stats),
    );

    assert_eq!(stats.auto_stiff_rejected, 1);
    assert!(
        stats.discarded_clamped_steps > 0,
        "the rejected escalation's clamps must be attributed to it: {stats:?}"
    );
    // Whatever clamps survive the subtraction belong to the explicit re-solve, which is the
    // trajectory the caller actually received.
    assert!(
        stats.min_step_clamped_steps >= stats.discarded_clamped_steps,
        "{stats:?}"
    );
}

#[test]
fn auto_round_trips_through_the_fit_option_token() {
    assert_eq!(OdeMethod::parse("auto"), Some(OdeMethod::Auto));
    assert_eq!(OdeMethod::parse("AUTO"), Some(OdeMethod::Auto));
    assert_eq!(OdeMethod::Auto.as_str(), "auto");
    assert_eq!(
        OdeMethod::parse(OdeMethod::Auto.as_str()),
        Some(OdeMethod::Auto)
    );
}

#[test]
fn the_threshold_constants_are_two_views_of_one_number() {
    assert!((STIFF_TAU_FAST * STIFF_RE_LAMBDA_THRESHOLD - 1.0).abs() < 1e-12);
}

#[test]
#[ignore = "measurement harness for #1080 item 3"]
fn measure_probe_cost_against_segment_cost() {
    use crate::ode::solve_ode;
    use std::time::Instant;
    let rhs = binding_rhs::<f64>(100.0, 1.0);
    let u = [100.0, 10.0, 0.0];
    let opts = loose();
    let n = 200_000;
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(resolve_method(&rhs, &u, &[], 0.0, &opts));
    }
    let probe = t0.elapsed().as_secs_f64() / n as f64;

    // A typical inter-dose segment: 12 h with one saveat, from the same state.
    let m = 20_000;
    let t1 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&rhs, &u, (0.0, 12.0), &[], &[12.0], &opts));
    }
    let segment = t1.elapsed().as_secs_f64() / m as f64;

    let explicit = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..Default::default()
    };
    let t2 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&rhs, &u, (0.0, 12.0), &[], &[12.0], &explicit));
    }
    let segment_pinned = t2.elapsed().as_secs_f64() / m as f64;

    println!(
        "STIFF  probe {:.3} us | auto segment {:.3} us | pinned rk45 segment {:.3} us | probe share {:.2}%",
        probe * 1e6,
        segment * 1e6,
        segment_pinned * 1e6,
        100.0 * probe / segment
    );

    // The case per-subject latching would serve: a benign model, where the probe never
    // escalates and is therefore pure overhead on every segment.
    let benign = binding_rhs::<f64>(0.01, 1.0);
    let t3 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(resolve_method(&benign, &u, &[], 0.0, &opts));
    }
    let benign_probe = t3.elapsed().as_secs_f64() / n as f64;
    let t4 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&benign, &u, (0.0, 12.0), &[], &[12.0], &opts));
    }
    let benign_auto = t4.elapsed().as_secs_f64() / m as f64;
    let t5 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&benign, &u, (0.0, 12.0), &[], &[12.0], &explicit));
    }
    let benign_pinned = t5.elapsed().as_secs_f64() / m as f64;
    // …and with a realistic observation grid on the segment, which is what a fit actually
    // integrates (every `saveat` clamps a step, so the segment does more work per probe).
    let saveat = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0];
    let t6 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&benign, &u, (0.0, 24.0), &[], &saveat, &opts));
    }
    let grid_auto = t6.elapsed().as_secs_f64() / m as f64;
    let t7 = Instant::now();
    for _ in 0..m {
        std::hint::black_box(solve_ode(&benign, &u, (0.0, 24.0), &[], &saveat, &explicit));
    }
    let grid_pinned = t7.elapsed().as_secs_f64() / m as f64;
    println!(
        "GRID   auto segment {:.3} us | pinned rk45 segment {:.3} us | auto overhead {:.2}%",
        grid_auto * 1e6,
        grid_pinned * 1e6,
        100.0 * (grid_auto - grid_pinned) / grid_pinned
    );
    println!(
        "BENIGN probe {:.3} us | auto segment {:.3} us | pinned rk45 segment {:.3} us | auto overhead {:.2}%",
        benign_probe * 1e6,
        benign_auto * 1e6,
        benign_pinned * 1e6,
        100.0 * (benign_auto - benign_pinned) / benign_pinned
    );
}

/// #1080 Part C, the measurement the issue asks for **before** any runtime detector is built:
/// does the step-rejection rate see the fast-binding family that `min_step_clamped_steps` is
/// blind to?
///
/// Sweeps `KON` at fixed `KD` (the shape of the #978 review's sweep), integrates each with the
/// pinned explicit default, and scores it against a tight-tolerance reference. Prints, per
/// decade: the accuracy the explicit solve actually delivered, and every runtime counter an
/// in-place switch could be triggered on — rejection rate, accepted steps per output point,
/// min-`dt` clamps — alongside the a-priori probe's verdict at the segment start.
#[test]
#[ignore = "measurement harness for #1080 Part C"]
fn measure_runtime_stiffness_signals_on_the_binding_sweep() {
    use crate::ode::solve_ode;
    use std::time::Instant;

    const KD: f64 = 0.1;
    let saveat: Vec<f64> = (1..=24).map(|h| h as f64).collect();
    let u0 = [100.0, 10.0, 0.0];

    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>9} {:>8} {:>8} {:>8} {:>9} {:>7}",
        "KON",
        "|Re l|max",
        "med relerr",
        "max relerr",
        "wall s",
        "att",
        "rej",
        "clamp",
        "rej rate",
        "acc/pt"
    );
    for kon in [0.1_f64, 1.0, 10.0, 100.0, 1000.0] {
        let rhs = binding_rhs::<f64>(kon, kon * KD);
        let lambda = max_abs_re_eigenvalue(&rhs, &u0, &[], 0.0).unwrap();

        // Reference: a stiff method at tolerances six decades tighter than production, with a
        // step budget it cannot exhaust. Cross-checked against a tight *explicit* solve, so a
        // reference that is itself wrong shows up as disagreement rather than as signal.
        let tight = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-12,
            max_steps: 10_000_000,
            method: OdeMethod::Rodas5P,
            ..Default::default()
        };
        let reference = solve_ode(&rhs, &u0, (0.0, 24.0), &[], &saveat, &tight);
        let tight_explicit = OdeSolverOptions {
            method: OdeMethod::Rk45,
            ..tight
        };
        let cross = solve_ode(&rhs, &u0, (0.0, 24.0), &[], &saveat, &tight_explicit);
        let rel = |a: &crate::ode::SolPoint, b: &crate::ode::SolPoint| {
            (a.u[0] - b.u[0]).abs() / a.u[0].abs().max(1e-30)
        };
        let cross_err = reference
            .iter()
            .zip(&cross)
            .map(|(a, b)| rel(a, b))
            .fold(0.0_f64, f64::max);

        let production = OdeSolverOptions {
            method: OdeMethod::Rk45,
            ..Default::default()
        };
        let mut stats = OdeSolverStats::default();
        let t0 = Instant::now();
        let got = solve_ode_with_stats(
            &rhs,
            &u0,
            (0.0, 24.0),
            &[],
            &saveat,
            &production,
            Some(&mut stats),
        );
        let wall = t0.elapsed().as_secs_f64();

        let mut errs: Vec<f64> = reference.iter().zip(&got).map(|(a, b)| rel(a, b)).collect();
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = errs[errs.len() / 2];
        let max = *errs.last().unwrap();
        let rej_rate = stats.rejected_steps as f64 / stats.attempted_steps.max(1) as f64;
        let acc_per_point = stats.accepted_steps as f64 / saveat.len() as f64;

        println!(
            "{kon:>8} {lambda:>10.1} {median:>12.2e} {max:>12.2e} {wall:>9.4} \
             {:>8} {:>8} {:>8} {rej_rate:>9.3} {acc_per_point:>7.1}   (ref cross-check {cross_err:.1e})",
            stats.attempted_steps, stats.rejected_steps, stats.min_step_clamped_steps
        );
    }
}

/// #1080 Part C: the case an **in-place** switch owns, measured. A depot-absorption binding
/// model starts non-stiff (both factors of `KON · central · target` are zero at dose time) and
/// turns stiff mid-segment, so the segment-start probe reads benign and `auto` keeps the
/// explicit stepper for the whole span.
///
/// Prints, per decade of `KON`: the probe's verdict at `t0` against `|Re λ|max` along the true
/// trajectory, then what each method actually delivered on the segment — the gap between the
/// explicit and the stiff row is what an in-place switch could recover, and the `auto` row is
/// what a user gets today.
#[test]
#[ignore = "measurement harness for #1080 Part C"]
fn measure_the_mid_segment_stiffness_a_start_probe_cannot_see() {
    use crate::ode::solve_ode;
    use std::time::Instant;

    const KD: f64 = 0.1;
    let saveat: Vec<f64> = (1..=24).map(|h| h as f64).collect();
    let u0 = [100.0, 0.0, 0.0, 0.0];

    for kon in [1.0_f64, 10.0, 100.0, 1000.0] {
        let rhs = latent_binding_rhs::<f64>(kon, kon * KD);
        let tight = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-12,
            max_steps: 10_000_000,
            method: OdeMethod::Rodas5P,
            ..Default::default()
        };
        let reference = solve_ode(&rhs, &u0, (0.0, 24.0), &[], &saveat, &tight);

        let lambda0 = max_abs_re_eigenvalue(&rhs, &u0, &[], 0.0).unwrap();
        let (t_cross, lambda_max) = reference.iter().fold((f64::NAN, 0.0_f64), |(tc, lm), p| {
            let l = max_abs_re_eigenvalue(&rhs, &p.u, &[], p.t).unwrap_or(0.0);
            let tc = if tc.is_nan() && l >= STIFF_RE_LAMBDA_THRESHOLD {
                p.t
            } else {
                tc
            };
            (tc, lm.max(l))
        });
        println!(
            "\nKON={kon}: probe at t0 reads {lambda0:.1} ({}), |Re l|max along the trajectory \
             {lambda_max:.1}, first stiff saveat t={t_cross}",
            if lambda0 >= STIFF_RE_LAMBDA_THRESHOLD {
                "stiff"
            } else {
                "NOT stiff"
            }
        );
        println!(
            "{:>12} {:>12} {:>12} {:>9} {:>8} {:>8} {:>8} {:>6}",
            "method", "med relerr", "max relerr", "wall s", "att", "rej", "clamp", "esc"
        );

        for (name, method) in [
            ("rk45", OdeMethod::Rk45),
            ("auto", OdeMethod::Auto),
            ("rodas4", OdeMethod::Rodas4),
        ] {
            let opts = OdeSolverOptions {
                method,
                ..Default::default()
            };
            let mut stats = OdeSolverStats::default();
            let t0 = Instant::now();
            let got = solve_ode_with_stats(
                &rhs,
                &u0,
                (0.0, 24.0),
                &[],
                &saveat,
                &opts,
                Some(&mut stats),
            );
            let wall = t0.elapsed().as_secs_f64();
            let mut errs: Vec<f64> = reference
                .iter()
                .zip(&got)
                .map(|(a, b)| (a.u[1] - b.u[1]).abs() / a.u[1].abs().max(1e-30))
                .collect();
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{name:>12} {:>12.2e} {:>12.2e} {wall:>9.4} {:>8} {:>8} {:>8} {:>6}",
                errs[errs.len() / 2],
                errs.last().unwrap(),
                stats.attempted_steps,
                stats.rejected_steps,
                stats.min_step_clamped_steps,
                stats.auto_stiff_segments,
            );
        }
    }
}

// ── #1080 Part C: mid-segment stepper switching ─────────────────────────────────────────────

/// Hourly saves over a day, from a state whose fast mode does not exist yet.
fn latent_grid() -> (Vec<f64>, [f64; 4]) {
    ((1..=24).map(|h| h as f64).collect(), [100.0, 0.0, 0.0, 0.0])
}

/// The reference the switching tests score against: the stiff workhorse at tolerances six
/// decades tighter than production, with a step budget it cannot exhaust.
fn tight_reference(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    saveat: &[f64],
) -> Vec<crate::ode::SolPoint> {
    let opts = OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-12,
        max_steps: 10_000_000,
        method: OdeMethod::Rodas5P,
        ..Default::default()
    };
    crate::ode::solve_ode(rhs, u0, (0.0, 24.0), &[], saveat, &opts)
}

/// Worst relative error in the central compartment against `reference`.
fn max_rel_err(reference: &[crate::ode::SolPoint], got: &[crate::ode::SolPoint]) -> f64 {
    reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a.u[1] - b.u[1]).abs() / a.u[1].abs().max(1e-30))
        .fold(0.0_f64, f64::max)
}

/// The case the in-place switch exists for, and the premise it rests on: a segment whose
/// Jacobian is benign at the entry state and stiff an hour later.
///
/// The pinned-explicit assertions are not decoration. Both runtime signals #1080 proposed as
/// triggers are blind here — this segment produces **zero** min-`dt` clamps while returning a
/// median 143% relative error — so if the explicit run ever stops being wrong, the trigger this
/// feature is built on needs re-deriving rather than the test re-tuning.
#[test]
fn auto_switches_in_place_when_a_segment_turns_stiff_mid_way() {
    let (saveat, u0) = latent_grid();
    let rhs = latent_binding_rhs::<f64>(100.0, 10.0);
    let reference = tight_reference(&rhs, &u0, &saveat);

    // The probe at the entry state reads benign, so the segment starts explicit — this is what
    // makes the *starting* decision insufficient rather than merely unlucky.
    assert!(
        max_abs_re_eigenvalue(&rhs, &u0, &[], 0.0).unwrap() < STIFF_RE_LAMBDA_THRESHOLD,
        "the entry state must read non-stiff or the test is not testing the switch"
    );

    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..Default::default()
    };
    let mut explicit_stats = OdeSolverStats::default();
    let explicit = solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &pinned,
        Some(&mut explicit_stats),
    );
    assert!(
        max_rel_err(&reference, &explicit) > 1.0,
        "premise: the explicit stepper must get this segment badly wrong"
    );
    assert_eq!(
        explicit_stats.min_step_clamped_steps, 0,
        "premise: and it must do so without clamping — the counter #708 reads is blind here"
    );

    let mut stats = OdeSolverStats::default();
    let switched = solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &loose(),
        Some(&mut stats),
    );
    let err = max_rel_err(&reference, &switched);
    assert!(
        err < 1e-4,
        "the switched solve must track the reference: {err:.3e}"
    );
    assert_eq!(stats.auto_switched_segments, 1, "{stats:?}");
    assert_eq!(
        stats.auto_stiff_segments, 1,
        "a mid-segment escalation is an escalation: {stats:?}"
    );
    assert_eq!(stats.auto_stiff_rejected, 0, "{stats:?}");
    assert!(
        stats.attempted_steps * 10 < explicit_stats.attempted_steps,
        "and it must be cheaper, not only more accurate: {} vs {}",
        stats.attempted_steps,
        explicit_stats.attempted_steps
    );
}

/// `ode_auto_switch = false` puts the segment back on one method chosen at its start — bit for
/// bit, not merely close: the option exists so a user can reproduce a pre-#1080-Part-C fit.
#[test]
fn disabling_the_switch_reproduces_the_segment_start_decision_bit_for_bit() {
    let (saveat, u0) = latent_grid();
    let rhs = latent_binding_rhs::<f64>(100.0, 10.0);

    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..Default::default()
    };
    let explicit = crate::ode::solve_ode(&rhs, &u0, (0.0, 24.0), &[], &saveat, &pinned);

    let no_switch = OdeSolverOptions {
        auto_switch: false,
        ..loose()
    };
    let mut stats = OdeSolverStats::default();
    let got = solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &no_switch,
        Some(&mut stats),
    );
    assert_eq!(stats.auto_switched_segments, 0, "{stats:?}");
    for (a, b) in explicit.iter().zip(&got) {
        assert_eq!(a.u, b.u, "t={}", a.t);
    }
}

/// A benign segment is never switched, and — because the re-probe only reads the state, never
/// anything the stepper owns — its trajectory stays bit-identical to the pinned explicit one.
#[test]
fn a_benign_segment_is_never_switched() {
    let rhs = binding_rhs::<f64>(0.01, 1.0);
    let u0 = [100.0, 10.0, 0.0];
    let saveat: Vec<f64> = (1..=24).map(|h| h as f64).collect();

    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..Default::default()
    };
    let explicit = crate::ode::solve_ode(&rhs, &u0, (0.0, 24.0), &[], &saveat, &pinned);

    let mut stats = OdeSolverStats::default();
    let auto = solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &loose(),
        Some(&mut stats),
    );
    assert_eq!(stats.auto_switched_segments, 0, "{stats:?}");
    assert_eq!(stats.auto_stiff_segments, 0, "{stats:?}");
    for (a, b) in explicit.iter().zip(&auto) {
        assert_eq!(a.u, b.u, "t={}", a.t);
    }
}

/// [`latent_binding_rhs`] with a cumulative-hazard accumulator on top (`dCHZ/dt = 0.05 · Cx`),
/// the shape the event-time driver monitors: monotone non-decreasing, and driven by the state
/// whose formation is what turns the system stiff.
fn latent_binding_with_hazard<T: PkNum>(kon: f64, koff: f64) -> impl Fn(&[T], &[T], f64, &mut [T]) {
    let core = latent_binding_rhs::<T>(kon, koff);
    move |u: &[T], p: &[T], t: f64, du: &mut [T]| {
        core(u, p, t, du);
        du[4] = u[3] * T::from_f64(0.05);
    }
}

/// The event-time driver switches mid-span too (#1080 Part C). It needs it at least as much as
/// the dense driver: its output *is* the likelihood contribution — an event time or a
/// censoring — so a span integrated badly does not merely cost accuracy in a reported
/// prediction, it moves the draw.
#[test]
fn the_event_time_driver_switches_mid_span() {
    use crate::ode::solver::{solve_ode_until_threshold, ThresholdCrossing};

    let rhs = latent_binding_with_hazard::<f64>(100.0, 10.0);
    let u_entry = [100.0, 0.0, 0.0, 0.0, 0.0];
    let threshold = 1.0;
    let crossing = |opts: &OdeSolverOptions| {
        let mut u = u_entry;
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 24.0), &[], opts, 4, threshold) {
            ThresholdCrossing::Crossed(t) => Some(t),
            _ => None,
        }
    };

    let reference = crossing(&OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-12,
        max_steps: 10_000_000,
        method: OdeMethod::Rodas5P,
        ..Default::default()
    })
    .expect("the tight reference must find the crossing");

    let explicit = crossing(&OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..Default::default()
    });
    let switched = crossing(&loose()).expect("the switched solve must find the crossing");

    assert!(
        (switched - reference).abs() < 1e-3,
        "switched crossing {switched} vs reference {reference}"
    );
    // The premise, in this driver's currency: pinned to the explicit stepper the same span
    // either misses the crossing outright or reports it at a materially different time.
    assert!(
        explicit.is_none_or(|t| (t - reference).abs() > 1e-2),
        "premise: the explicit stepper must not already get this right ({explicit:?} vs \
         {reference})"
    );
}

/// The escalation guard covers a segment that reached the stiff stepper **mid-way** exactly as
/// it covers one that started on it: `du/dt = u²` from a state that reads non-stiff, so the
/// segment starts explicit, switches when the blow-up brings `|Re λ|max` past the threshold,
/// and the stiff attempt then fails. The result is discarded and re-solved explicitly, and the
/// clamps that attempt took are attributed to the discarded trajectory rather than to the one
/// the caller received.
#[test]
fn a_stalled_mid_segment_switch_is_discarded_and_re_solved_explicitly() {
    let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = u[0] * u[0];
    let u0 = [5.0];
    assert!(
        max_abs_re_eigenvalue(&rhs, &u0, &[], 0.0).unwrap() < STIFF_RE_LAMBDA_THRESHOLD,
        "the entry state must read non-stiff or this tests the start-escalation guard instead"
    );

    let mut stats = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 0.25),
        &[],
        &[0.25],
        &loose(),
        Some(&mut stats),
    );

    assert_eq!(stats.auto_switched_segments, 1, "{stats:?}");
    assert_eq!(
        stats.auto_stiff_segments, 1,
        "the mid-segment escalation must be counted, or the rejection below has nothing to be \
         a fraction of: {stats:?}"
    );
    assert_eq!(stats.auto_stiff_rejected, 1, "{stats:?}");
    assert!(
        stats.discarded_clamped_steps > 0,
        "the discarded attempt's clamps belong to it, not to the returned trajectory: {stats:?}"
    );
    assert!(
        stats.min_step_clamped_steps >= stats.discarded_clamped_steps,
        "{stats:?}"
    );
}

/// A segment that never leaves the explicit stepper is **not** guarded, even when it clamps:
/// there the fallback is the same solve, so discarding the result would buy a second copy of
/// it. Pinned by counting steps — a re-solve would roughly double them.
///
/// Run with `auto_switch` **on**, and on a system whose verdict the re-probe can never change,
/// so it exercises the guard's `!ran_stiff ⇒ usable` arm rather than the early return that
/// turning switching off would take.
#[test]
fn an_unswitched_explicit_segment_is_not_re_solved() {
    // Clamps at `min_dt` on every step (a rotation the coarse floor cannot resolve) while
    // reading `Re λ = 0` at every state, so no re-probe ever escalates it.
    let rhs = rotating_rhs::<f64>(200.0);
    let opts = coarse_min_dt();
    assert!(opts.auto_switch, "the arm under test needs switching on");
    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..coarse_min_dt()
    };
    let mut auto_stats = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0, 0.0],
        (0.0, 1.0),
        &[],
        &[1.0],
        &opts,
        Some(&mut auto_stats),
    );
    let mut pinned_stats = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &[1.0, 0.0],
        (0.0, 1.0),
        &[],
        &[1.0],
        &pinned,
        Some(&mut pinned_stats),
    );
    assert!(auto_stats.min_step_clamped_steps > 0, "{auto_stats:?}");
    assert_eq!(
        auto_stats.auto_switched_segments, 0,
        "premise: the re-probe must never change its mind here: {auto_stats:?}"
    );
    assert_eq!(auto_stats.auto_stiff_rejected, 0, "{auto_stats:?}");
    assert_eq!(
        auto_stats.attempted_steps, pinned_stats.attempted_steps,
        "an unswitched explicit segment must be solved exactly once"
    );
}

/// The property the analytic gradient depends on, extended to the switch: the `Dual1`/`Dual2`
/// sensitivity solve must change stepper at the *same* accepted step as the `f64` prediction,
/// or the gradient would differentiate a trajectory the predictor never produced. It holds
/// because the re-probe reads `PkNum::val` only — seeding derivatives on the state is exactly
/// the perturbation that would break it if it ever read a jet.
#[test]
fn the_switch_happens_at_the_same_step_at_f64_and_at_a_dual() {
    const N: usize = 2;
    let (saveat, u0) = latent_grid();
    let scalar_rhs = latent_binding_rhs::<f64>(100.0, 10.0);
    let dual_rhs = latent_binding_rhs::<Dual1<N>>(100.0, 10.0);
    let u_dual: Vec<Dual1<N>> = vec![
        Dual1::<N>::var(u0[0], 0),
        Dual1::<N>::var(u0[1], 1),
        Dual1::<N>::constant(u0[2]),
        Dual1::<N>::constant(u0[3]),
    ];

    let mut scalar_stats = OdeSolverStats::default();
    let scalar = solve_ode_with_stats(
        &scalar_rhs,
        &u0,
        (0.0, 24.0),
        &[],
        &saveat,
        &loose(),
        Some(&mut scalar_stats),
    );
    let mut dual_stats = OdeSolverStats::default();
    let dual = crate::ode::solver::solve_ode_g_with_stats(
        &dual_rhs,
        &u_dual,
        (0.0, 24.0),
        &[],
        &saveat,
        &loose(),
        Some(&mut dual_stats),
    );

    assert_eq!(scalar_stats.auto_switched_segments, 1, "{scalar_stats:?}");
    assert_eq!(
        scalar_stats, dual_stats,
        "the step sequence, and the switch inside it, must not depend on T"
    );
    // The counters above are the switch assertion — identical attempts, accepts and switches
    // means both solves changed stepper at the same accepted step, which is the property the
    // analytic gradient needs. The values then agree to round-off rather than bit for bit: the
    // jet's value channel accumulates its own rounding through the stage arithmetic (and
    // through the Rosenbrock linear solve after the switch), which a depot state decaying five
    // orders of magnitude amplifies in *relative* terms while leaving it far inside the
    // solver's own `abstol`.
    for (a, b) in scalar.iter().zip(&dual) {
        for (x, y) in a.u.iter().zip(&b.u) {
            approx::assert_relative_eq!(*x, y.val(), max_relative = 1e-6, epsilon = 1e-9);
        }
    }
}

// ── The guard scores the *stiff* half of a switched segment, not all of it ───────────────────

/// Coarse `min_dt`, so a step that fails its error test is force-accepted rather than shrunk
/// into the 1e-12 floor the production default sets. That is what lets the fixtures below
/// produce min-`dt` clamps on demand, on a chosen stepper, within a handful of steps.
fn coarse_min_dt() -> OdeSolverOptions {
    OdeSolverOptions {
        min_dt: 1e-2,
        initial_dt: 1e-2,
        ..loose()
    }
}

/// A fast **rotation**: `Re λ = 0` at every state, so the probe reads it as non-stiff however
/// often it is asked, while the oscillation itself needs a step far below `coarse_min_dt`'s
/// floor. Every step therefore clamps, on the explicit stepper, with no stiffness anywhere for
/// a re-probe to find.
fn rotating_rhs<T: PkNum>(w: f64) -> impl Fn(&[T], &[T], f64, &mut [T]) {
    move |u: &[T], _p: &[T], _t: f64, du: &mut [T]| {
        du[0] = u[1] * T::from_f64(w);
        du[1] = u[0] * T::from_f64(-w);
    }
}

/// `[x, y, z]`: the rotation above on `(x, y)` for `t < 1`, then a fast pull of `z` towards 1
/// — `|Re λ|max = 500`, comfortably stiff, at `λ·min_dt = -5`, which is outside the explicit
/// stepper's stability region.
///
/// So the segment clamps ~100 times on the explicit stepper *before* any switch, escalates at
/// the first re-probe that lands in the second phase, and is then integrated cleanly. It is the
/// exact shape the escalation guard must not throw away.
///
/// `z` is displaced from its equilibrium by only `1e-6` (accumulated over the first phase),
/// which is what makes the two halves separable: the stiff stepper meets its error test on a
/// perturbation that small at the first attempt and never clamps, while the explicit stepper —
/// whose amplification factor at `hλ = -5` is well above 1 — grows it by orders of magnitude
/// per step until the trajectory is meaningless. Anything larger would clamp on *both*, and a
/// stiff half that clamps is a stall the guard is right to discard.
fn clamps_then_turns_stiff<T: PkNum>() -> impl Fn(&[T], &[T], f64, &mut [T]) {
    move |u: &[T], _p: &[T], t: f64, du: &mut [T]| {
        if t < 1.0 {
            du[0] = u[1] * T::from_f64(200.0);
            du[1] = u[0] * T::from_f64(-200.0);
            du[2] = T::from_f64(1e-6);
        } else {
            du[0] = T::from_f64(0.0);
            du[1] = T::from_f64(0.0);
            du[2] = (T::from_f64(1.0) - u[2]) * T::from_f64(500.0);
        }
    }
}

/// A segment that clamped on the **explicit** stepper and was then rescued by a mid-segment
/// escalation must be kept. Those clamps are the limit of the method the switch replaced, not
/// evidence that the stiff method stalled, and discarding on them hands the caller back the
/// pinned-explicit trajectory the escalation had just avoided.
#[test]
fn explicit_clamps_before_a_switch_do_not_discard_the_stiff_half() {
    let rhs = clamps_then_turns_stiff::<f64>();
    let u0 = [1.0, 0.0, 1.0];

    let mut stats = OdeSolverStats::default();
    let got = solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 2.0),
        &[],
        &[2.0],
        &coarse_min_dt(),
        Some(&mut stats),
    );

    assert_eq!(
        stats.auto_switched_segments, 1,
        "premise: the segment must escalate mid-way: {stats:?}"
    );
    assert!(
        stats.min_step_clamped_steps > 0,
        "premise: the explicit phase must clamp, or the guard has nothing to fire on: {stats:?}"
    );
    assert_eq!(
        stats.stiff_min_step_clamped_steps, 0,
        "the stiff half integrated cleanly, so nothing here may read as a stall: {stats:?}"
    );
    assert_eq!(
        stats.auto_stiff_rejected, 0,
        "a rescued segment must not be discarded: {stats:?}"
    );
    assert_eq!(
        stats.discarded_clamped_steps, 0,
        "nothing was discarded, so no clamp belongs to a discarded trajectory: {stats:?}"
    );

    // What the discard costs, in the currency that matters: pinned to the explicit stepper this
    // system is outside the stability region in its second phase, so the re-solve hands back a
    // diverged trajectory where the switch had produced the right answer.
    assert!(
        (got[0].u[2] - 1.0).abs() < 1e-3,
        "the kept trajectory must be the switched one (z -> 1): {:?}",
        got[0].u
    );
    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..coarse_min_dt()
    };
    let explicit = crate::ode::solve_ode(&rhs, &u0, (0.0, 2.0), &[], &[2.0], &pinned);
    assert!(
        !((explicit[0].u[2] - 1.0).abs() < 1e-3),
        "premise: the explicit fallback must not already be right ({:?})",
        explicit[0].u
    );
}

/// The event-time driver's copy of the same rule. The discard is worse here than in the dense
/// path: the crossing search re-runs from the entry state pinned to the explicit stepper, whose
/// second phase diverges, so a run that *had* found the crossing comes back with a different
/// answer — or none at all, which is a censoring laundered out of a solve that worked.
#[test]
fn explicit_clamps_before_a_switch_do_not_discard_a_crossing() {
    // `[x, y, z, chz]`: `clamps_then_turns_stiff` plus a hazard accumulator driven by `z`, so
    // the crossing time depends on the second phase being integrated rather than amplified.
    let rhs = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
        if t < 1.0 {
            du[0] = 200.0 * u[1];
            du[1] = -200.0 * u[0];
            du[2] = 1e-6;
            du[3] = 0.1;
        } else {
            du[0] = 0.0;
            du[1] = 0.0;
            du[2] = 500.0 * (1.0 - u[2]);
            du[3] = 0.1 * u[2];
        }
    };
    let u_entry = [1.0, 0.0, 1.0, 0.0];
    let threshold = 0.15;
    let tight = OdeSolverOptions {
        abstol: 1e-12,
        reltol: 1e-12,
        max_steps: 10_000_000,
        method: OdeMethod::Rodas5P,
        ..Default::default()
    };

    let mut u = u_entry;
    let reference = match crate::ode::solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 3.0),
        &[],
        &tight,
        3,
        threshold,
    ) {
        crate::ode::solver::ThresholdCrossing::Crossed(t) => t,
        other => panic!("the reference must cross: {other:?}"),
    };
    assert!(
        reference > 1.0,
        "premise: the crossing must land after the switch ({reference})"
    );

    let mut u = u_entry;
    let crossed = match crate::ode::solve_ode_until_threshold(
        &rhs,
        &mut u,
        (0.0, 3.0),
        &[],
        &coarse_min_dt(),
        3,
        threshold,
    ) {
        crate::ode::solver::ThresholdCrossing::Crossed(t) => t,
        other => {
            panic!("the switched run found the crossing; the guard must not undo it: {other:?}")
        }
    };
    assert!(
        (crossed - reference).abs() < 5e-2,
        "switched crossing {crossed} vs reference {reference}"
    );

    // The premise, in this driver's currency: pinned explicit, the same span does not report
    // this crossing correctly — which is what the discarding guard used to hand back.
    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..coarse_min_dt()
    };
    let mut u = u_entry;
    if let crate::ode::solver::ThresholdCrossing::Crossed(t) =
        crate::ode::solve_ode_until_threshold(&rhs, &mut u, (0.0, 3.0), &[], &pinned, 3, threshold)
    {
        assert!(
            (t - reference).abs() > 5e-2,
            "premise: the explicit fallback must not already be right ({t} vs {reference})"
        );
    }
}

/// The abort budget bounds how long *one method* may grind on a segment it cannot step, so a
/// switch starts it again. Without the reset the stiff half inherits the explicit half's spent
/// clamps and is abandoned for a stability limit belonging to the stepper it replaced.
///
/// Pinned at the exact boundary: the budget is set to the number of clamps the explicit phase
/// produces, measured rather than assumed. The abort test runs *after* the accept block that
/// takes the switch, so with the reset the escalation begins on a fresh budget and the segment
/// finishes; without it the budget is already spent at that same step and the stiff half is
/// abandoned before taking one.
#[test]
fn the_abort_budget_starts_again_when_the_stepper_is_swapped() {
    let rhs = clamps_then_turns_stiff::<f64>();
    let u0 = [1.0, 0.0, 1.0];
    let pinned = OdeSolverOptions {
        method: OdeMethod::Rk45,
        ..coarse_min_dt()
    };
    // The explicit phase is exactly `[0, 1)`, and the switched run steps it identically, so its
    // clamp count is the budget the stiff half would inherit.
    let mut first_phase = OdeSolverStats::default();
    solve_ode_with_stats(
        &rhs,
        &u0,
        (0.0, 1.0),
        &[],
        &[1.0],
        &pinned,
        Some(&mut first_phase),
    );
    let budget = u32::try_from(first_phase.min_step_clamped_steps).unwrap();
    assert!(budget > 0, "premise: the explicit phase must clamp");

    let opts = OdeSolverOptions {
        stiff_abort_after: Some(budget),
        ..coarse_min_dt()
    };
    let mut stats = OdeSolverStats::default();
    let got = solve_ode_with_stats(&rhs, &u0, (0.0, 2.0), &[], &[2.0], &opts, Some(&mut stats));
    assert_eq!(
        stats.auto_switched_segments, 1,
        "premise: the segment must reach its mid-segment escalation: {stats:?}"
    );
    assert_eq!(
        stats.stiff_aborted_segments, 0,
        "the stiff half clamped nothing of its own, so nothing may abort it: {stats:?}"
    );
    assert_eq!(
        stats.auto_stiff_rejected, 0,
        "nor may it be discarded: {stats:?}"
    );
    assert!(
        (got[0].u[2] - 1.0).abs() < 1e-3,
        "the segment must still be integrated to the end: {:?}",
        got[0].u
    );
}
