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

    let mut floored = mm_iv().parsed.model;
    floored.ode_spec.as_mut().unwrap().solver_opts.reltol = RELTOL_FLOOR;
    assert!(tightened_override(&floored, &opts).is_none());
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
