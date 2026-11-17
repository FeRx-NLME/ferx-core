use super::*;
use crate::types::{WarningCode, WarningSeverity};

fn shift(fit_ofv: f64, tight_ofv: f64) -> OdeToleranceShift {
    OdeToleranceShift {
        fit_reltol: 1e-4,
        fit_abstol: 1e-6,
        tight_reltol: 1e-5,
        tight_abstol: 1e-7,
        fit_ofv,
        tight_ofv,
        inner_tol: 1e-5,
    }
}

/// The gate straddles [`OFV_SHIFT_WARN`] on both signs, and a non-finite side warns: a
/// comparison that measured nothing is not evidence of a settled objective.
#[test]
fn the_gate_warns_past_the_threshold_on_either_side_and_on_a_non_finite_side() {
    assert!(!shift(-100.0, -100.0).not_settled());
    assert!(!shift(-100.0, -100.0 - 0.9 * OFV_SHIFT_WARN).not_settled());
    assert!(!shift(-100.0, -100.0 + 0.9 * OFV_SHIFT_WARN).not_settled());
    assert!(shift(-100.0, -100.0 - 1.1 * OFV_SHIFT_WARN).not_settled());
    assert!(shift(-100.0, -100.0 + 1.1 * OFV_SHIFT_WARN).not_settled());
    assert!(shift(-100.0, f64::NAN).not_settled());
    assert!(shift(f64::INFINITY, -100.0).not_settled());
    assert!(not_settled_warning(&shift(-100.0, -100.05)).is_none());
}

/// Every sentence of the finite-shift message, and the absence of the non-finite one.
#[test]
fn the_finite_shift_message_states_the_tolerances_the_shift_and_the_remedy() {
    let msg = not_settled_warning(&shift(-5165.3135, -5171.4432)).expect("a 6-unit shift warns");
    for clause in [
        "W_ODE_OFV_NOT_SETTLED: ",
        "re-solving the EBEs and the objective at the final estimates",
        "with ode_reltol = 1e-5 / ode_abstol = 1e-7 instead of the fit's 1e-4 / 1e-6",
        "moves the OFV by -6.1297 (-5165.3135 -> -5171.4432), more than 0.1 units",
        "the objective is not settled at the fit's ODE tolerance",
        "OFV differences of that size between models may be integration error",
        "Refit with ode_reltol / ode_abstol at most 1e-5 / 1e-7",
        "tightening until the OFV stops moving (#2994)",
    ] {
        assert!(msg.contains(clause), "missing {clause:?} in {msg}");
    }
    assert!(!msg.contains("non-finite"), "{msg}");
    let classified = crate::types::classify_warning(&msg);
    assert_eq!(classified.category, WarningCode::OdeSolver);
    assert_eq!(classified.severity, WarningSeverity::Warning);
}

/// The non-finite branch says it could not measure, not that the OFV moved.
#[test]
fn the_non_finite_message_says_the_check_could_not_measure() {
    let msg = not_settled_warning(&shift(-100.0, f64::NAN)).expect("a NaN side warns");
    assert!(
        msg.contains("gives a non-finite objective (-100.0000 -> NaN)"),
        "{msg}"
    );
    assert!(msg.contains("could not be measured"), "{msg}");
    assert!(!msg.contains("moves the OFV"), "{msg}");
    assert!(
        msg.contains("Refit with ode_reltol / ode_abstol at most"),
        "{msg}"
    );
    let classified = crate::types::classify_warning(&msg);
    assert_eq!(classified.category, WarningCode::OdeSolver);
}

#[test]
fn only_the_objectives_pop_nll_opts_scores_are_in_scope() {
    use EstimationMethod::*;
    for m in [Foce, FoceI, FoceGn, FoceGnHybrid, Laplace] {
        assert!(method_in_scope(m), "{m:?}");
    }
    for m in [Saem, Imp, Impmap, Bayes, Vi] {
        assert!(!method_in_scope(m), "{m:?}");
    }
}

fn mm_iv() -> crate::api::PreparedRun {
    crate::api::prepare_run("examples/mm_iv.ferx", Some("data/mm_iv.csv")).expect("prepare mm_iv")
}

/// One decade on each tolerance, ten times the step budget, the caller's method kept; nothing
/// for a closed form or a fit already at the floor.
#[test]
fn the_override_tightens_one_decade_and_keeps_the_callers_method() {
    let prep = mm_iv();
    let model = &prep.parsed.model;
    let base = model.integrator_spec().unwrap().effective_solver_opts();
    let opts = FitOptions {
        ode_method: crate::ode::OdeMethod::Rodas5P,
        ..FitOptions::default()
    };
    let o = tightened_override(model, &opts).expect("an ODE model at the default is checked");
    assert_eq!(o.reltol, Some(base.reltol * 0.1));
    assert_eq!(o.abstol, Some(base.abstol * 0.1));
    assert_eq!(o.max_steps, Some(base.max_steps * 10));
    assert_eq!(o.method, Some(crate::ode::OdeMethod::Rodas5P));

    // #2994 review r2: a `reltol` at its floor still has its `abstol` tightened...
    let mut floored = mm_iv().parsed.model;
    floored.ode_spec.as_mut().unwrap().solver_opts.reltol = RELTOL_FLOOR;
    let o = tightened_override(&floored, &opts).expect("abstol 1e-6 can still be tightened");
    assert_eq!(o.reltol, Some(RELTOL_FLOOR), "reltol stays at its floor");
    assert_eq!(o.abstol, Some(base.abstol * 0.1));
    // ...and the mirror image...
    let mut abs_floored = mm_iv().parsed.model;
    abs_floored.ode_spec.as_mut().unwrap().solver_opts.abstol = ABSTOL_FLOOR;
    let o = tightened_override(&abs_floored, &opts).expect("reltol can still be tightened");
    assert_eq!(o.reltol, Some(base.reltol * 0.1));
    assert_eq!(o.abstol, Some(ABSTOL_FLOOR));
    // ...and only both at their floors skips the check.
    let mut both = mm_iv().parsed.model;
    let so = &mut both.ode_spec.as_mut().unwrap().solver_opts;
    so.reltol = RELTOL_FLOOR;
    so.abstol = ABSTOL_FLOOR;
    assert!(tightened_override(&both, &opts).is_none());
    let mut above = mm_iv().parsed.model;
    above.ode_spec.as_mut().unwrap().solver_opts.reltol = 1e-11;
    assert_eq!(
        tightened_override(&above, &opts).unwrap().reltol,
        Some(RELTOL_FLOOR),
        "the re-solve never goes below the floor"
    );

    let warfarin = crate::api::prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .expect("prepare warfarin");
    assert!(tightened_override(&warfarin.parsed.model, &opts).is_none());
}

fn evaluate(prep: &crate::api::PreparedRun, tol: f64) -> crate::types::FitResult {
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        ode_reltol: tol,
        ode_abstol: tol,
        ode_method: crate::ode::OdeMethod::Rk45,
        ..prep.parsed.fit_options.clone()
    };
    crate::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("the evaluation runs")
}

fn not_settled(r: &crate::types::FitResult) -> Option<&String> {
    r.warnings
        .iter()
        .find(|w| w.starts_with("W_ODE_OFV_NOT_SETTLED:"))
}

/// End to end, a differential pair straddling the gate: the same evaluation at a tolerance
/// loose enough to leave the objective unsettled warns, typed, with the measured shift in the
/// details; at a tight one it does not. Either half alone is satisfied by a check that always
/// or never fires.
#[test]
fn fit_warns_when_the_ofv_moves_at_a_tighter_tolerance_and_not_when_settled() {
    let prep = mm_iv();

    // Measured on this fixture (macOS arm64): 1e-3 → 1e-4 moves the OFV by +1.414, 1e-10 →
    // 1e-11 by -3.2e-8. So the pair sits 14× above and 3e6× below the threshold.
    let loose = evaluate(&prep, 1e-3);
    let msg = not_settled(&loose)
        .unwrap_or_else(|| panic!("no W_ODE_OFV_NOT_SETTLED in {:?}", loose.warnings));
    assert!(
        msg.contains("ode_reltol = 1e-4 / ode_abstol = 1e-4 instead of the fit's 1e-3 / 1e-3"),
        "{msg}"
    );
    assert!(msg.contains("moves the OFV by"), "{msg}");
    let entry = loose
        .warnings_structured
        .iter()
        .find(|e| e.message.starts_with("W_ODE_OFV_NOT_SETTLED:"))
        .expect("typed at source");
    assert_eq!(entry.category, WarningCode::OdeSolver);
    assert_eq!(entry.severity, WarningSeverity::Warning);
    let details = entry.details.as_ref().expect("details");
    let d = details["shift"].as_f64().expect("a finite shift");
    assert!(d.abs() > OFV_SHIFT_WARN, "shift {d}");
    let fit_ofv = details["fit_ofv"].as_f64().unwrap();
    assert!(
        (fit_ofv - loose.ofv).abs() < 1e-6 * loose.ofv.abs().max(1.0),
        "the fit-tolerance side re-scores the reported OFV: {fit_ofv} vs {}",
        loose.ofv
    );

    let tight = evaluate(&prep, 1e-10);
    assert!(
        not_settled(&tight).is_none(),
        "a settled objective warned: {:?}",
        tight.warnings
    );
}

fn details_of(r: &crate::types::FitResult) -> Option<&serde_json::Value> {
    r.warnings_structured
        .iter()
        .find(|e| e.message.starts_with("W_ODE_OFV_NOT_SETTLED:"))
        .and_then(|e| e.details.as_ref())
}

fn chain(
    prep: &crate::api::PreparedRun,
    methods: Vec<EstimationMethod>,
    tol: f64,
) -> crate::types::FitResult {
    let opts = FitOptions {
        verbose: false,
        methods: methods.clone(),
        agq_eval_only: methods.contains(&EstimationMethod::Laplace),
        outer_maxiter: 0,
        vi_iters: 1,
        run_covariance_step: false,
        ode_reltol: tol,
        ode_abstol: tol,
        ode_method: crate::ode::OdeMethod::Rk45,
        ..prep.parsed.fit_options.clone()
    };
    crate::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("the chain runs")
}

/// #2994 review r1: a trailing `agq_eval_only` readout publishes its own quadrature marginal
/// while the stored modes stay the estimator's, so the check must re-score the readout's
/// objective. The fit-tolerance side of the check is the published OFV, and that OFV is a
/// different number from the FOCEI objective the check used to score — the straddle is
/// asserted, so the comparison cannot pass on the estimator's objective.
#[test]
fn an_agq_readout_is_checked_on_the_objective_it_publishes() {
    let prep = mm_iv();
    let readout = chain(
        &prep,
        vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        1e-3,
    );
    let focei = chain(&prep, vec![EstimationMethod::FoceI], 1e-3);
    let gap = (readout.ofv - focei.ofv).abs();
    assert!(
        gap > 1e-3,
        "fixture: the readout must publish a different objective ({} vs {})",
        readout.ofv,
        focei.ofv
    );
    let d = details_of(&readout)
        .unwrap_or_else(|| panic!("no W_ODE_OFV_NOT_SETTLED in {:?}", readout.warnings));
    let fit_ofv = d["fit_ofv"].as_f64().expect("finite");
    assert!(
        (fit_ofv - readout.ofv).abs() < 1e-3 * gap,
        "the check scored {fit_ofv}, the readout published {} (FOCEI: {})",
        readout.ofv,
        focei.ofv
    );
}

/// #2994 review r1: `[vi, laplace]` + `agq_eval_only` — the recommended way to finish a VI
/// fit — publishes the readout's objective, so it is checked even though the last estimating
/// stage (VI) is out of scope. It used to be skipped.
#[test]
fn a_vi_fit_finished_by_an_agq_readout_is_checked() {
    let prep = mm_iv();
    let r = chain(
        &prep,
        vec![EstimationMethod::Vi, EstimationMethod::Laplace],
        1e-3,
    );
    let d =
        details_of(&r).unwrap_or_else(|| panic!("no W_ODE_OFV_NOT_SETTLED in {:?}", r.warnings));
    let fit_ofv = d["fit_ofv"].as_f64().expect("finite");
    assert!(
        (fit_ofv - r.ofv).abs() < 1e-6 * r.ofv.abs().max(1.0),
        "the check scored {fit_ofv}, the readout published {}",
        r.ofv
    );
}

/// A check that applied but could not run says so, typed, and is not the not-settled finding.
#[test]
fn a_check_that_could_not_run_is_reported_not_silent() {
    let msg = not_checked_warning("pool build failed");
    for clause in [
        "W_ODE_OFV_SETTLEMENT_UNCHECKED: ",
        "whether the OFV is settled at the fit's ODE tolerance could not be checked",
        "could not be completed, because pool build failed.",
        "The fit and its OFV are unaffected",
        "refit once with tighter ode_reltol / ode_abstol and compare the OFV",
    ] {
        assert!(msg.contains(clause), "missing {clause:?} in {msg}");
    }
    assert!(!msg.contains("W_ODE_OFV_NOT_SETTLED"), "{msg}");
    let (_, entry) = crate::api::postfit::ode_ofv_not_checked_warning("pool build failed");
    assert_eq!(entry.category, WarningCode::OdeSolver);
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert_eq!(
        entry.details.as_ref().unwrap()["reason"],
        "pool build failed"
    );
    let classified = crate::types::classify_warning(&msg);
    assert_eq!(classified.category, WarningCode::OdeSolver);
    assert_eq!(classified.severity, WarningSeverity::Warning);
}

/// Each counter that marks a padded or abandoned trajectory makes the tight pass unclean, by
/// name; a clamp or stop in an `auto` attempt the guard discarded does not.
#[test]
fn an_unclean_tight_pass_names_the_counters_that_were_not_clean() {
    use crate::ode::solver::OdeSolverStats;
    assert_eq!(unclean_integration(&OdeSolverStats::default()), None);
    let discarded = OdeSolverStats {
        min_step_clamped_steps: 4,
        discarded_clamped_steps: 4,
        unfinished_segments: 2,
        discarded_unfinished_segments: 2,
        ..Default::default()
    };
    assert_eq!(
        unclean_integration(&discarded),
        None,
        "discarded attempts were re-solved"
    );
    for (st, want) in [
        (
            OdeSolverStats {
                min_step_clamped_steps: 3,
                discarded_clamped_steps: 1,
                ..Default::default()
            },
            "2 steps clamped at the minimum step size",
        ),
        (
            OdeSolverStats {
                unfinished_segments: 1,
                ..Default::default()
            },
            "1 unfinished segments",
        ),
        (
            OdeSolverStats {
                stiff_aborted_segments: 5,
                ..Default::default()
            },
            "5 stiff-aborted segments",
        ),
        (
            OdeSolverStats {
                diverged_segments: 1,
                ..Default::default()
            },
            "1 diverged segments",
        ),
        (
            OdeSolverStats {
                abandoned_non_finite_timeline: 2,
                ..Default::default()
            },
            "2 abandoned non-finite timelines",
        ),
    ] {
        assert_eq!(unclean_integration(&st).as_deref(), Some(want));
    }
}

/// #2994 review r2, end to end on real solves: a tighter pass that clamps at `min_dt` returns
/// a finite objective that never met its tolerance, so it is reported as unchecked rather than
/// compared. `min_dt = 0.5` is where the fit's own tolerance (1e-8) integrates cleanly — the
/// post-fit solver pass carries no `W_ODE_SOLVER_DIAGNOSTICS` — while 1e-9 force-accepts steps
/// there (measured: macOS arm64). The straddle is asserted, so the fixture cannot quietly turn
/// into one where the fit itself is unclean.
#[test]
fn a_tight_pass_that_clamped_is_reported_unchecked_not_compared() {
    let mut prep = mm_iv();
    prep.parsed
        .model
        .ode_spec
        .as_mut()
        .unwrap()
        .solver_opts
        .min_dt = 0.5;
    let r = evaluate(&prep, 1e-8);
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.starts_with("W_ODE_SOLVER_DIAGNOSTICS")),
        "fixture: the fit's own tolerance must integrate cleanly: {:?}",
        r.warnings
    );
    assert!(not_settled(&r).is_none(), "{:?}", r.warnings);
    let msg = r
        .warnings
        .iter()
        .find(|w| w.starts_with("W_ODE_OFV_SETTLEMENT_UNCHECKED:"))
        .unwrap_or_else(|| panic!("no unchecked warning in {:?}", r.warnings));
    assert!(
        msg.contains("because the tighter solve did not integrate cleanly at the re-found EBEs:"),
        "{msg}"
    );
    assert!(
        msg.contains("steps clamped at the minimum step size"),
        "{msg}"
    );
    let entry = r
        .warnings_structured
        .iter()
        .find(|e| e.message.starts_with("W_ODE_OFV_SETTLEMENT_UNCHECKED:"))
        .expect("typed at source");
    assert_eq!(entry.category, WarningCode::OdeSolver);
}

/// #2994 review r2: an estimating Laplace stage is checked under its own stage options, whose
/// `inner_tol` the stage tightens to at most 1e-8, not under the top-level ones (1e-5 here).
#[test]
fn an_estimating_laplace_stage_is_checked_under_its_own_stage_options() {
    let prep = mm_iv();
    let opts = FitOptions {
        verbose: false,
        method: EstimationMethod::Laplace,
        methods: Vec::new(),
        outer_maxiter: 0,
        inner_tol: 1e-5,
        run_covariance_step: false,
        ode_reltol: 1e-3,
        ode_abstol: 1e-3,
        ode_method: crate::ode::OdeMethod::Rk45,
        ..prep.parsed.fit_options.clone()
    };
    let r = crate::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("the evaluation runs");
    let d =
        details_of(&r).unwrap_or_else(|| panic!("no W_ODE_OFV_NOT_SETTLED in {:?}", r.warnings));
    let tol = d["inner_tol"].as_f64().expect("inner_tol recorded");
    assert!(
        tol <= 1e-8,
        "checked at inner_tol = {tol}, not the Laplace stage's"
    );
}
