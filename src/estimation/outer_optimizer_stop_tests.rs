use super::*;
use crate::estimation::outer_optimizer::tests::outer_fd_fallback::{
    analytic_and_declining, mk_pop,
};

fn focei() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..FitOptions::default()
    }
}

/// The gate, one condition at a time from a case it accepts: each flip alone must turn
/// it off. Mutation that reddens it: dropping any one condition.
#[test]
fn the_report_gate_turns_off_on_each_condition() {
    let (model, _, _) = analytic_and_declining();
    let template = &model.default_params;
    let opts = focei();
    let gate = |resolved, converged, m: &CompiledModel, t: &ModelParameters, o: &FitOptions| {
        wants_stop_report(resolved, converged, m, t, o)
    };
    for optimizer in [
        Optimizer::NloptLbfgs,
        Optimizer::Slsqp,
        Optimizer::Mma,
        Optimizer::Lbfgs,
        Optimizer::Bfgs,
    ] {
        assert!(
            gate(optimizer, false, &model, template, &opts),
            "{optimizer:?}"
        );
    }
    assert!(!gate(Optimizer::Bobyqa, false, &model, template, &opts));
    assert!(!gate(Optimizer::NloptLbfgs, true, &model, template, &opts));

    let iov = crate::parser::model_parser::parse_model_string(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#,
    )
    .expect("IOV model parses");
    assert!(!gate(
        Optimizer::NloptLbfgs,
        false,
        &iov,
        &iov.default_params,
        &opts
    ));

    let reconverging = FitOptions {
        reconverge_gradient_interval: 1,
        ..focei()
    };
    assert!(!gate(
        Optimizer::NloptLbfgs,
        false,
        &model,
        template,
        &reconverging
    ));
    let every_other = FitOptions {
        reconverge_gradient_interval: 2,
        ..focei()
    };
    assert!(gate(
        Optimizer::NloptLbfgs,
        false,
        &model,
        template,
        &every_other
    ));

    let (mut fd, _, _) = analytic_and_declining();
    fd.gradient_method = GradientMethod::Fd;
    assert!(!gate(Optimizer::NloptLbfgs, false, &fd, template, &opts));

    let agq = FitOptions {
        method: EstimationMethod::Laplace,
        ..focei()
    };
    assert!(!gate(Optimizer::NloptLbfgs, false, &model, template, &agq));

    let mixture = crate::parser::model_parser::parse_model_string(
        r#"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
"#,
    )
    .expect("mixture model parses");
    assert!(
        mixture.default_params.mixture.is_some(),
        "fixture is a mixture"
    );
    assert!(!gate(
        Optimizer::NloptLbfgs,
        false,
        &mixture,
        &mixture.default_params,
        &opts
    ));

    let cancel = crate::cancel::CancelFlag::new();
    cancel.cancel();
    let cancelled = FitOptions {
        cancel: Some(cancel),
        ..focei()
    };
    assert!(!gate(
        Optimizer::NloptLbfgs,
        false,
        &model,
        template,
        &cancelled
    ));
}

/// Each sentence of both arms, and neither arm's text in the other. Deleting any sentence
/// reddens this test.
#[test]
fn the_warning_names_its_case_and_only_it() {
    let exact = stop_report_warning(
        StopGradientGap {
            n_held: 0,
            gap: 0.0,
        },
        5,
    );
    let held = stop_report_warning(
        StopGradientGap {
            n_held: 2,
            gap: 0.123456,
        },
        5,
    );
    for w in [&exact, &held] {
        assert!(
            w.starts_with("The outer optimizer stopped without converging (#2713)."),
            "{w}"
        );
        assert!(
            w.contains(
                "EBEs that change mode between nearby estimates, or a kink in the model \
                 such as `if`, `abs` or `max`."
            ),
            "{w}"
        );
    }
    const EXACT: [&str; 3] = [
        "At the reported estimates every subject's outer gradient is exact (analytic, or \
         reconverged), so the fixed-EBE fallback does not bias the final gradient",
        "though it can still have biased the gradient at a line-search trial point.",
        "Unless the stop was the evaluation budget, look first at the objective:",
    ];
    const HELD: [&str; 4] = [
        "At the reported estimates 2 of 5 subjects carry fixed-EBE outer gradients, and the \
         gradient the fit used there is 12.346% off the exact one.",
        "Unless the stop was the evaluation budget, test whether that caused it by rerunning \
         with `reconverge_gradient_interval = 1`:",
        "if the stop goes away, the fallback was the cause;",
        "if it stays, look at the objective:",
    ];
    for s in EXACT {
        assert!(
            exact.contains(s) && !held.contains(s),
            "{s:?}\n{exact}\n{held}"
        );
    }
    for s in HELD {
        assert!(
            held.contains(s) && !exact.contains(s),
            "{s:?}\n{exact}\n{held}"
        );
    }
}

/// A coordinate at a bound the gradient pushes against is left out of the norm; the same
/// coordinate pushed away from the bound, or off the bound, is kept. Mutations that redden
/// it: dropping either side's clause, or flipping a sign test.
#[test]
fn a_coordinate_pinned_at_its_bound_is_not_measured() {
    let bounds = PackedBounds {
        lower: vec![0.0, 0.0, 0.0],
        upper: vec![1.0, 1.0, 1.0],
    };
    let v = [3.0, 4.0, 12.0];
    let norm = |x: [f64; 3], g: [f64; 3]| free_norm(&x, &bounds, &g, &v);
    assert_eq!(
        norm([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        13.0,
        "nothing at a bound"
    );
    assert_eq!(
        norm([0.5, 0.5, 0.0], [1.0, 1.0, 1.0]),
        5.0,
        "pinned at the lower"
    );
    assert_eq!(
        norm([0.5, 0.5, 0.0], [1.0, 1.0, -1.0]),
        13.0,
        "leaving the lower"
    );
    assert_eq!(
        norm([0.5, 0.5, 1.0], [1.0, 1.0, -1.0]),
        5.0,
        "pinned at the upper"
    );
    assert_eq!(
        norm([0.5, 0.5, 1.0], [1.0, 1.0, 1.0]),
        13.0,
        "leaving the upper"
    );
}

/// The held subjects are the declines the assembly's salvage left on the held-EBE
/// gradient, under the same predicate the assembly uses: a numerical decline is held, a
/// structural one is not, nor any under a reconverging log, nor a non-Gaussian subject
/// with records (#1778). Mutations that redden it: filtering on structural declines alone
/// (the TTE case), or dropping the filter (every case but the first).
#[cfg(feature = "survival")]
#[test]
fn only_the_held_ebe_salvage_counts_as_held() {
    let (model, analytic, _) = analytic_and_declining();
    let pop = mk_pop(vec![analytic]);
    let numerical = OuterFdDeclineLog::new(1);
    numerical.record(0);
    assert_eq!(
        held_subjects(&model, &pop, &numerical),
        vec![0],
        "numerical"
    );
    let structural = OuterFdDeclineLog::new(1);
    structural.record(0);
    structural.record_structural(0);
    assert!(
        held_subjects(&model, &pop, &structural).is_empty(),
        "structural"
    );
    let reconverging = OuterFdDeclineLog::reconverging(1);
    reconverging.record(0);
    assert!(
        held_subjects(&model, &pop, &reconverging).is_empty(),
        "reconverging"
    );

    let tte = crate::parser::model_parser::parse_model_string(
        &std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file"),
    )
    .expect("parse");
    let (tte_pop, _) = crate::api::read_population_for(
        &tte,
        &None,
        "data/tte_exponential.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("data");
    assert!(
        !tte_pop.subjects[0].obs_records.is_empty(),
        "fixture: the TTE subject has records"
    );
    let log = OuterFdDeclineLog::new(tte_pop.subjects.len());
    log.record(0);
    assert!(
        held_subjects(&tte, &tte_pop, &log).is_empty(),
        "a non-Gaussian decline is reconverged, not held"
    );
}

/// The fixture subject at an `η` where its exact `H` is indefinite: the analytic gradient
/// declines it at the mode system, so its gradient is the held-EBE fallback.
fn off_mode_eta(model: &CompiledModel, subject: &Subject) -> Vec<f64> {
    let params = &model.default_params;
    let grid = [-2.5, -1.5, -0.75, 0.75, 1.5, 2.5];
    let mut points = Vec::new();
    for &a in &grid {
        for &b in &grid {
            for &c in &grid {
                points.push(vec![a, b, c]);
            }
        }
    }
    points
        .into_iter()
        .find(|e| {
            crate::sens::ode_provider::adjoint::exact_inner_hessian(model, subject, params, e)
                .is_some_and(|h| h.cholesky().is_none())
        })
        .expect("a point where the exact H is indefinite")
}

/// The gap on a real population against the brute-force one: the fit's gradient and the
/// exact one each assembled over the whole population, with the held-EBE and reconverging
/// logs. The production function reconverges only the held subjects; the two must agree.
/// The straddle: with the analytic subject at a served `η` nothing is held (the
/// out-of-scope twin is structural, reconverged in both) and the gap is exactly zero.
/// Mutations that redden it: reconverging the held subjects with a held-EBE log (gap 0),
/// or counting the structural decline as held.
#[test]
fn the_gap_is_the_reconverged_difference() {
    let (model, analytic, mut declining) = analytic_and_declining();
    // The twin's SS infusion every 72 h rather than 24 h: at 24 h its reconverged salvage
    // caps an SS equilibration, whose warning lands in the process-wide sink and fails a
    // concurrently running `fit()` test that accounts for every warning (#1289). Measured:
    // the sink stays empty at 72 h.
    declining.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, true, 72.0)];
    // The plans' failure is keyed by the subject's content, so re-key it after the edit.
    crate::parser::model_parser::rhs_regcode::kernel_jit::fail_plans_of(&declining);
    let pop = mk_pop(vec![analytic, declining]);
    let params = &model.default_params;
    let opts = focei();
    let hms: Vec<DMatrix<f64>> = pop
        .subjects
        .iter()
        .map(|s| DMatrix::zeros(s.observations.len(), model.n_eta))
        .collect();
    let off = off_mode_eta(&model, &pop.subjects[0]);
    let PackedStart {
        packed: x, bounds, ..
    } = pack_with_bounds(params);

    let at = |eta0: &[f64]| {
        let ehs = vec![
            DVector::from_column_slice(eta0),
            DVector::zeros(model.n_eta),
        ];
        let gap =
            stop_gradient_gap(&model, &pop, params, params, &ehs, &hms, &opts).expect("measured");
        let full = |log: &OuterFdDeclineLog| {
            population_gradient_sens_mixed_dx(
                &x,
                params,
                &model,
                &pop,
                &ehs,
                &hms,
                &bounds,
                &opts,
                OuterTrial::unknown(),
                log,
                false,
            )
            .0
        };
        let fit_log = OuterFdDeclineLog::new(2);
        let g_fit = full(&fit_log);
        assert!(
            fit_log.declined_indices().contains(&1) && fit_log.is_structural(1),
            "fixture: the twin is a structural decline"
        );
        let g_exact = full(&OuterFdDeclineLog::reconverging(2));
        assert!(
            g_fit.iter().chain(&g_exact).all(|v| v.is_finite()),
            "finite gradients"
        );
        let diff: Vec<f64> = g_fit.iter().zip(&g_exact).map(|(a, b)| a - b).collect();
        let brute =
            free_norm(&x, &bounds, &g_exact, &diff) / free_norm(&x, &bounds, &g_exact, &g_exact);
        (gap, brute)
    };

    let (gap, brute) = at(&off);
    assert_eq!(
        gap.n_held, 1,
        "the off-mode subject is held, the structural one not"
    );
    // Measured 0.99934 (the held-EBE gradient at a far off-mode `η` is nearly orthogonal
    // to the reconverged one), and the subset equals the brute force to every digit.
    assert!(gap.gap > 0.5, "the fallback must show: {gap:?}");
    assert!(
        (gap.gap - brute).abs() <= 1e-9 * brute,
        "subset {} vs brute force {brute}",
        gap.gap
    );

    let (gap, brute) = at(&[0.0; 3]);
    assert_eq!(gap.n_held, 0, "at a served eta nothing is held");
    assert_eq!(
        (gap.gap, brute),
        (0.0, 0.0),
        "and the gradients agree exactly"
    );
}

/// The hook in [`optimize_population`]: an L-BFGS fit cut short by its budget gets the
/// report, and the same fit under BOBYQA does not. Mutation that reddens it: dropping the
/// `attach_stop_report` call.
#[test]
fn an_unconverged_gradient_fit_carries_the_report() {
    let (model, analytic, _) = analytic_and_declining();
    let pop = mk_pop(vec![analytic]);
    const LEAD: &str = "The outer optimizer stopped without converging (#2713).";
    for (optimizer, wants) in [(Optimizer::NloptLbfgs, true), (Optimizer::Bobyqa, false)] {
        let opts = FitOptions {
            optimizer,
            outer_maxiter: 1,
            ..focei()
        };
        let result = optimize_population(&model, &pop, &model.default_params, &opts);
        assert!(!result.converged, "{optimizer:?}: fixture must stop short");
        let report: Vec<&String> = result
            .warnings
            .iter()
            .filter(|w| w.starts_with(LEAD))
            .collect();
        assert_eq!(
            report.len(),
            usize::from(wants),
            "{optimizer:?}: {:?}",
            result.warnings
        );
        if wants {
            assert!(
                report[0].contains("every subject's outer gradient is exact"),
                "{}",
                report[0]
            );
        }
    }
}
