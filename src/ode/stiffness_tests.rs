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
    assert!(budgeted.stiff_aborted_segments >= 1);
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
