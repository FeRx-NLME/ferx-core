use super::*;
use crate::estimation::outer_optimizer::tests::outer_fd_fallback::{
    analytic_and_declining, mk_pop,
};

fn dir(descends: bool, mode_switches: usize) -> DirectionProbe {
    DirectionProbe {
        descends,
        mode_switches,
    }
}

fn probe(n_declined: usize, fit: DirectionProbe, exact: DirectionProbe) -> FailureProbe {
    FailureProbe {
        n_declined,
        gap: if n_declined == 0 { 0.0 } else { 0.42 },
        fit,
        exact,
    }
}

fn open_box(n: usize) -> PackedBounds {
    PackedBounds {
        lower: vec![-10.0; n],
        upper: vec![10.0; n],
    }
}

/// The module table, row by row, and its one precedence: a fit gradient that descends
/// explains nothing at this point whatever the exact one does.
#[test]
fn the_verdict_follows_the_table() {
    use FailureVerdict::*;
    let cases = [
        (dir(false, 0), dir(true, 0), FixedEbeGradient),
        (dir(false, 3), dir(false, 2), ModeJumps(2)),
        (dir(false, 0), dir(false, 0), NotDifferentiable),
        (dir(true, 0), dir(true, 0), NoCauseHere),
        (dir(true, 0), dir(false, 5), NoCauseHere),
    ];
    for (fit, exact, want) in cases {
        assert_eq!(probe(1, fit, exact).verdict(), want, "{fit:?} / {exact:?}");
    }
}

/// Each verdict's sentence, and none of the others': the remedy is the deliverable, so each
/// is asserted with the other verdicts' discriminators absent. Also both sides of the
/// declined-subjects clause.
#[test]
fn the_warning_names_the_verdict_and_only_it() {
    let marks = [
        "omit the EBE-response term",
        "switch mode: the objective jumps",
        "not differentiable at the reported estimates",
        "not where the line search ran out of descent",
    ];
    let cases = [
        (probe(7, dir(false, 0), dir(true, 0)), 0),
        (probe(7, dir(false, 0), dir(false, 4)), 1),
        (probe(0, dir(false, 0), dir(false, 0)), 2),
        (probe(0, dir(true, 0), dir(true, 0)), 3),
    ];
    for (p, want) in cases {
        let w = outer_failure_warning(&p, 20);
        assert!(w.starts_with("The outer optimizer stopped without converging."));
        for (k, mark) in marks.iter().enumerate() {
            assert_eq!(w.contains(mark), k == want, "{mark:?} in: {w}");
        }
        assert_eq!(
            w.contains("`reconverge_gradient_interval = 1`"),
            want == 0,
            "the remedy goes with its verdict: {w}"
        );
        if p.n_declined == 0 {
            assert!(w.contains("every subject has the exact analytic outer gradient"));
            assert!(!w.contains("carries a fallback"), "got: {w}");
        } else {
            assert!(
                w.contains("carries a fallback for 7 of 20 subjects there and is 42% off"),
                "got: {w}"
            );
            assert!(!w.contains("every subject has the exact"), "got: {w}");
        }
    }
    let w = outer_failure_warning(&probe(7, dir(false, 0), dir(false, 4)), 20);
    assert!(w.contains("the EBEs of 4 subjects switch"), "got: {w}");
}

/// A smooth objective descends along its gradient at the first step.
#[test]
fn a_smooth_objective_descends() {
    let x = [1.0, -0.5];
    let g = [2.0, -1.0];
    let f = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
    let eta0 = vec![DVector::from_vec(vec![0.1])];
    let calls = std::cell::Cell::new(0);
    let p = probe_direction(&g, &x, &open_box(2), f(&x), &eta0, &[1.0], |v| {
        calls.set(calls.get() + 1);
        (f(v), eta0.clone())
    })
    .expect("probed");
    assert_eq!(p, dir(true, 0));
    assert_eq!(calls.get(), 1, "it stops at the first step that descends");
}

/// A jump: every step off `x` costs 5 OFV, and one of two subjects' EBEs moves by 2 SDs
/// (the other by a hundredth). No step descends, and the switch is counted. The kink twin
/// — `10·|x|` differentiated from the right, so `-g` crosses it — moves no EBE.
#[test]
fn a_jump_counts_its_mode_switches_and_a_kink_counts_none() {
    let x = [0.0];
    let eta0 = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![0.0])];
    let sd = [0.5];
    let jumped = vec![DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.005])];
    let p = probe_direction(&[3.0], &x, &open_box(1), 0.0, &eta0, &sd, |v| {
        (5.0 + v[0] * v[0], jumped.clone())
    })
    .expect("probed");
    assert_eq!(p, dir(false, 1));

    let p = probe_direction(&[10.0], &x, &open_box(1), 0.0, &eta0, &sd, |v| {
        (10.0 * v[0].abs(), eta0.clone())
    })
    .expect("probed");
    assert_eq!(p, dir(false, 0));

    // A NaN EBE left its mode too; `f64::max`-style folding would have dropped it.
    let lost = vec![
        DVector::from_vec(vec![f64::NAN]),
        DVector::from_vec(vec![0.0]),
    ];
    let p = probe_direction(&[3.0], &x, &open_box(1), 0.0, &eta0, &sd, |_| {
        (1.0, lost.clone())
    })
    .expect("probed");
    assert_eq!(p, dir(false, 1));
}

/// A step that only pushes into an active bound cannot be formed; the free coordinate of a
/// partly bound gradient still is.
#[test]
fn a_gradient_into_its_bound_is_not_probed() {
    let bounds = PackedBounds {
        lower: vec![0.0, -10.0],
        upper: vec![10.0, 10.0],
    };
    let eta0 = vec![DVector::from_vec(vec![0.0])];
    let f = |v: &[f64]| v[0] + v[1] * v[1];
    let none = probe_direction(&[1.0, 0.0], &[0.0, 0.0], &bounds, 0.0, &eta0, &[1.0], |v| {
        (f(v), eta0.clone())
    });
    assert_eq!(none, None);
    let some = probe_direction(
        &[1.0, 2.0],
        &[0.0, 1.0],
        &bounds,
        f(&[0.0, 1.0]),
        &eta0,
        &[1.0],
        |v| {
            assert_eq!(v[0], 0.0, "the bound coordinate stays put");
            (f(v), eta0.clone())
        },
    );
    assert_eq!(some, Some(dir(true, 0)));
}

/// Only a gradient-driven optimizer's own unconverged stop is probed.
#[test]
fn only_an_unconverged_gradient_stop_is_probed() {
    let opts = FitOptions {
        outer_maxiter: 100,
        ..FitOptions::default()
    };
    assert!(wants_failure_probe(Optimizer::NloptLbfgs, false, 40, &opts));
    assert!(!wants_failure_probe(Optimizer::NloptLbfgs, true, 40, &opts));
    assert!(
        !wants_failure_probe(Optimizer::NloptLbfgs, false, 100, &opts),
        "budget spent"
    );
    assert!(!wants_failure_probe(Optimizer::Bobyqa, false, 40, &opts));
    assert!(!wants_failure_probe(
        Optimizer::TrustRegion,
        false,
        40,
        &opts
    ));
    let flag = crate::cancel::CancelFlag::new();
    flag.cancel();
    let cancelled = FitOptions {
        cancel: Some(flag),
        ..opts.clone()
    };
    assert!(!wants_failure_probe(
        Optimizer::NloptLbfgs,
        false,
        40,
        &cancelled
    ));
}

/// The probe on a real population, near the initial estimates (not an optimum, so the
/// exact gradient descends), around one served subject and its twin:
/// - the twin's gradient forced to a numerical decline (the log's test switch): the fit's
///   gradient carries its held-EBE salvage and differs from the exact one;
/// - the SS-infusion twin the closed form declines for its shape: that salvage is already
///   reconverged, so the two gradients agree;
/// - the served subject alone: one gradient, probed once.
#[test]
fn the_probe_measures_the_fallback_on_a_real_population() {
    let (model, analytic, declining) = analytic_and_declining();
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..FitOptions::default()
    };
    // Off the start's lag of 0.5, which lands on the 0.5 h observation: a zero prediction
    // on the proportional floor puts the OFV at 5.8e13, too steep for any step to resolve.
    let mut moved = model.default_params.clone();
    let lag = moved
        .theta_names
        .iter()
        .position(|n| n == "TVLAG")
        .expect("TVLAG");
    moved.theta[lag] = 0.2;
    let template = &model.default_params;
    let params = &moved;
    let run = |pop: &Population, force: Option<usize>| {
        let mu_k = compute_mu_k(&model, &params.theta, opts.mu_referencing);
        let (ehs, hms, _, _) = run_inner_loop_warm_seeded(
            &model,
            pop,
            params,
            opts.inner_maxiter,
            opts.inner_tol,
            None,
            Some(&mu_k),
            opts.min_obs_for_convergence_check as usize,
            opts.inner_restarts,
            InnerHessianSeed::for_options(&opts),
            &opts,
        );
        let n = pop.subjects.len();
        probe_with_logs(
            &model,
            pop,
            params,
            params,
            &ehs,
            &hms,
            &opts,
            |reconverge| {
                let log = if reconverge {
                    OuterFdDeclineLog::reconverging(n)
                } else {
                    OuterFdDeclineLog::new(n)
                };
                if let Some(i) = force {
                    log.force_decline(i);
                }
                log
            },
        )
        .expect("probed")
    };
    let pair = mk_pop(vec![analytic.clone(), analytic.clone()]);
    let held = run(&pair, Some(1));
    assert_eq!(held.n_declined, 1);
    assert!(
        held.gap.is_finite() && held.gap > 1e-3,
        "the held-EBE salvage is not the reconverged gradient: {held:?}"
    );
    assert!(held.exact.descends, "the start is not an optimum: {held:?}");
    // The straddle: the same pair unforced is one exact gradient.
    let served = run(&pair, None);
    assert_eq!(served.n_declined, 0);
    assert_eq!(served.gap, 0.0);
    assert_eq!(served.fit, served.exact);

    let structural = run(&mk_pop(vec![analytic.clone(), declining]), None);
    assert_eq!(structural.n_declined, 1);
    assert!(
        structural.gap.is_finite() && structural.gap < 1e-12,
        "a structural decline is reconverged in the fit's gradient too: {structural:?}"
    );
    // Production's entry point builds the same logs.
    let pop = mk_pop(vec![analytic]);
    let mu_k = compute_mu_k(&model, &params.theta, opts.mu_referencing);
    let (ehs, hms, _, _) = run_inner_loop_warm_seeded(
        &model,
        &pop,
        params,
        opts.inner_maxiter,
        opts.inner_tol,
        None,
        Some(&mu_k),
        opts.min_obs_for_convergence_check as usize,
        opts.inner_restarts,
        InnerHessianSeed::for_options(&opts),
        &opts,
    );
    let exact =
        diagnose_outer_failure(&model, &pop, template, params, &ehs, &hms, &opts).expect("probed");
    assert_eq!(exact.n_declined, 0);
    assert!(exact.fit.descends, "{exact:?}");
}

/// A gradient ten times steeper than the objective (a biased one, as a held-EBE fallback
/// can be): the objective does fall along it, but by 1% of each step's prediction, which
/// is not descent a line search can accept. The same objective along its true slope is.
#[test]
fn a_gradient_that_overstates_the_slope_does_not_descend() {
    let x = [0.0];
    let eta0 = vec![DVector::from_vec(vec![0.0])];
    // f falls with slope 1 along +x.
    let f = |v: &[f64]| -v[0];
    let biased = probe_direction(&[-100.0], &x, &open_box(1), 0.0, &eta0, &[1.0], |v| {
        (f(v), eta0.clone())
    })
    .expect("probed");
    assert_eq!(biased, dir(false, 0));
    let true_slope = probe_direction(&[-1.0], &x, &open_box(1), 0.0, &eta0, &[1.0], |v| {
        (f(v), eta0.clone())
    })
    .expect("probed");
    assert_eq!(true_slope, dir(true, 0));
}

/// Steps are sized on the free part of the gradient: a coordinate pinned against its bound
/// does not shrink the step along the others. With the pinned component 100× the free one
/// (and the free one large enough that `MAX_STEP` does not bind), sizing on the whole
/// gradient would predict a drop of `1e-4` at the first step, not 1.
/// Both bounds, one coordinate each.
#[test]
fn a_bound_coordinate_does_not_size_the_step() {
    let bounds = PackedBounds {
        lower: vec![0.0, -10.0],
        upper: vec![10.0, 10.0],
    };
    let eta0 = vec![DVector::from_vec(vec![0.0])];
    for (g, x) in [([1000.0, 10.0], [0.0, 0.0]), ([-1000.0, 10.0], [10.0, 0.0])] {
        let first = std::cell::Cell::new(None);
        probe_direction(&g, &x, &bounds, 0.0, &eta0, &[1.0], |v| {
            if first.get().is_none() {
                first.set(Some(g[0] * (x[0] - v[0]) + g[1] * (x[1] - v[1])));
            }
            (1.0, eta0.clone())
        })
        .expect("probed");
        let predicted = first.get().expect("evaluated");
        assert!(
            (predicted - 1.0).abs() < 1e-12,
            "g = {g:?}: first predicted drop {predicted}"
        );
    }
}

/// The hook: an unconverged gradient stop gains the probe's warning, a converged one (the
/// straddle) does not. The result is an evaluation at the moved start, marked as a stop.
#[test]
fn an_unconverged_stop_carries_the_probe_warning() {
    let (model, analytic, _) = analytic_and_declining();
    let pop = mk_pop(vec![analytic]);
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 100,
        ..FitOptions::default()
    };
    let mut moved = model.default_params.clone();
    let lag = moved
        .theta_names
        .iter()
        .position(|n| n == "TVLAG")
        .expect("TVLAG");
    moved.theta[lag] = 0.2;
    for converged in [false, true] {
        let mut result = evaluate_at_initial_params(&model, &pop, &moved, &opts);
        result.converged = converged;
        result.n_iterations = 10;
        let before = result.warnings.len();
        attach_failure_probe(
            Optimizer::NloptLbfgs,
            &model,
            &pop,
            &model.default_params,
            &opts,
            &mut result,
        );
        let added: Vec<&String> = result.warnings[before..].iter().collect();
        if converged {
            assert!(added.is_empty(), "got: {added:?}");
        } else {
            assert_eq!(added.len(), 1, "got: {added:?}");
            assert!(
                added[0].starts_with("The outer optimizer stopped without converging.")
                    && added[0].contains("every subject has the exact analytic outer gradient")
                    && added[0].contains("not where the line search ran out of descent"),
                "got: {}",
                added[0]
            );
        }
    }
}
