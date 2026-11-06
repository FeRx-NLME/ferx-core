//! Tier-1 tests for the `stalled_at_init` fit warning (#997 §2).
//!
//! The *predicate* (`crate::stalled_at_init`) already existed and is tested in
//! `model_selection_tests.rs`. What was missing — and what these tests pin — is
//! that a fit which never left its initial estimates now says so on the result,
//! next to `boundary_estimate`, instead of being reachable only by a caller who
//! thought to ask. A stalled fit usually reports `converged: true`, so nothing
//! else in the output distinguishes it from a good one.
use super::*;
use crate::types::test_helpers::minimal_fit_result;
use crate::types::{Population, WarningCode, WarningSeverity};

/// A fit sitting exactly on its initial estimates, with no optimizer verdict
/// recorded (so the natural-scale reading is what fires).
fn stalled_result() -> crate::types::FitResult {
    let mut r = minimal_fit_result();
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    r.left_init = None;
    r.converged = true;
    r
}

/// The warning as a default, estimating fit sees it.
fn estimating_fit_warning(r: &crate::types::FitResult) -> Option<(String, WarningEntry)> {
    stalled_at_init_warning(r, &FitOptions::default())
}

#[test]
fn a_fit_that_moved_produces_no_warning() {
    let mut r = stalled_result();
    // One free theta 2% away from its start is enough to have left init.
    r.theta[0] = r.theta_init[0] * 1.02;
    assert!(
        estimating_fit_warning(&r).is_none(),
        "a fit that moved must not be flagged"
    );
}

#[test]
fn a_fit_that_never_moved_is_flagged_with_its_own_category() {
    let r = stalled_result();
    let (msg, entry) = estimating_fit_warning(&r).expect("nothing moved — this must warn");

    assert_eq!(entry.category, WarningCode::StalledAtInit);
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(
        msg.contains("W_STALLED_AT_INIT"),
        "the message must carry its token so re-classification recovers the code: {msg}"
    );
    // The message has to name what did not move, or the user has nothing to act
    // on — a bare "the fit stalled" is the `converged: true` problem restated.
    assert!(
        msg.contains("CL"),
        "the unmoved free THETA must be named: {msg}"
    );
}

/// `rebuild_warnings_structured` re-classifies any message it has no native
/// entry for, so the round trip has to land back on `StalledAtInit` — and
/// specifically **not** on `Convergence`. That distinction is the whole point:
/// a stalled fit typically reports `converged: true` (a fit that never moved has
/// a perfectly flat objective trace to plateau on), so a consumer branching on
/// `convergence` would read the opposite of what happened.
#[test]
fn the_message_reclassifies_to_stalled_at_init_and_not_to_convergence() {
    let (msg, _) = estimating_fit_warning(&stalled_result()).expect("must warn");
    let reclassified = crate::types::classify_warning(&msg);
    assert_eq!(
        reclassified.category,
        WarningCode::StalledAtInit,
        "re-classifying the plain message text must recover the code, got {:?}",
        reclassified.category
    );
    assert_ne!(reclassified.category, WarningCode::Convergence);
}

/// The two readings behind the predicate report differently, because they are
/// measured in different spaces with different tolerances (the optimizer's is a
/// 1% move in *scaled packed* space, the fallback's a 1% move on the natural
/// scale). A consumer comparing one fit's verdict against another's needs to
/// know which fired.
#[test]
fn the_details_name_which_reading_produced_the_verdict() {
    let mut r = stalled_result();

    let (_, entry) = estimating_fit_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    assert_eq!(d["verdict_source"], "natural_scale");

    // With the optimizer's own escape test recorded, that is what is read —
    // even though the natural-scale comparison would fire here too.
    r.left_init = Some(false);
    let (_, entry) = estimating_fit_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    assert_eq!(d["verdict_source"], "optimizer_escape_test");

    // And the optimizer's verdict wins in the other direction as well: it says
    // the fit escaped, so there is no warning, despite θ/Ω/σ all reading unmoved.
    r.left_init = Some(true);
    assert!(
        estimating_fit_warning(&r).is_none(),
        "the optimizer's own verdict is preferred when the result carries one"
    );
}

/// The `theta` payload is what an agent reads to see how far the fit is from
/// having said anything. FIXed parameters are not part of it — they were never
/// going to move, so listing them as "unmoved" is noise that makes a two-free-θ
/// stall look like a twelve-parameter one.
#[test]
fn the_details_list_free_thetas_with_their_initial_values_and_skip_fixed_ones() {
    let mut r = stalled_result();
    r.theta_fixed[1] = true; // V

    let (_, entry) = estimating_fit_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    let thetas = d["theta"].as_array().expect("theta array");

    let names: Vec<&str> = thetas
        .iter()
        .map(|t| t["parameter"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["CL", "KA"],
        "the FIXed theta must not be listed as unmoved"
    );
    assert_eq!(thetas[0]["estimate"], r.theta[0]);
    assert_eq!(thetas[0]["init"], r.theta_init[0]);
}

/// An evaluation-only run (`outer_maxiter = 0`, NONMEM `MAXEVAL=0`) reports the
/// objective at the initial estimates **because that is what was asked for**.
/// The predicate is correct there and the warning is pure noise, so it is
/// suppressed — the one case where "the estimates are the initial values" is the
/// answer rather than a symptom.
#[test]
fn an_evaluation_only_run_is_not_reported_as_a_stall() {
    let mut r = stalled_result();
    r.outer_maxiter = 0;
    assert!(estimating_fit_warning(&r).is_none());

    // …and the suppression is scoped to that request alone: one iteration is a
    // real (if tiny) fit, and a stall there is worth saying.
    r.outer_maxiter = 1;
    assert!(estimating_fit_warning(&r).is_some());
}

/// #2716: a chain with no estimating stage reports the initial estimates by request, as
/// `outer_maxiter = 0` does, while leaving `outer_maxiter` at its default. The result is
/// held fixed across every row, so the options alone decide: an estimating fit that did
/// not move is still a stall, including one followed by a readout, and including one
/// whose eval-only flag names a method it does not run.
#[test]
fn only_a_chain_with_no_estimating_stage_is_exempt() {
    use EstimationMethod::{FoceI, Imp, Impmap, Laplace};
    let mut r = stalled_result();
    r.outer_maxiter = 500;
    let single = |method| FitOptions {
        method,
        ..Default::default()
    };
    let chain = |methods: &[EstimationMethod]| FitOptions {
        methods: methods.to_vec(),
        ..Default::default()
    };
    let imp_eval = |o: FitOptions| FitOptions {
        imp_eval_only: true,
        ..o
    };
    let agq_eval = |o: FitOptions| FitOptions {
        agq_eval_only: true,
        ..o
    };

    for (name, opts) in [
        ("standalone imp_eval_only", imp_eval(single(Imp))),
        ("standalone imp_eval_only, chain", imp_eval(chain(&[Imp]))),
        ("standalone agq_eval_only", agq_eval(single(Laplace))),
        (
            "standalone agq_eval_only, chain",
            agq_eval(chain(&[Laplace])),
        ),
    ] {
        assert!(
            stalled_at_init_warning(&r, &opts).is_none(),
            "{name}: evaluating at the initial estimates was the request"
        );
    }

    for (name, opts) in [
        ("estimating imp", single(Imp)),
        ("estimating laplace", single(Laplace)),
        ("focei then imp readout", imp_eval(chain(&[FoceI, Imp]))),
        ("focei then agq readout", agq_eval(chain(&[FoceI, Laplace]))),
        ("inert imp_eval_only on focei", imp_eval(single(FoceI))),
        ("inert imp_eval_only on laplace", imp_eval(single(Laplace))),
        (
            "imp_eval_only leaves impmap estimating",
            imp_eval(single(Impmap)),
        ),
    ] {
        assert!(
            stalled_at_init_warning(&r, &opts).is_some(),
            "{name}: an estimator ran and did not move"
        );
    }
}

/// The options-level rows above cannot see `fit()` handing the wrong options to the
/// warning, so run both standalone evaluators for real. The `stalled_at_init` premise is
/// what makes the absence mean something: the predicate still fires on these results, and
/// `outer_maxiter` is not the gate that silenced it.
#[test]
fn standalone_evaluators_report_no_stall_through_fit() {
    use crate::api::nonfinite_objective_tests::subject_with_dose_time;
    let model = crate::parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse");
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("1", 0.0, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
            subject_with_dose_time("3", 0.0, 0.9),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let base = FitOptions {
        run_covariance_step: false,
        threads: Some(1),
        ..Default::default()
    };
    for (name, opts) in [
        (
            "imp_eval_only",
            FitOptions {
                method: EstimationMethod::Imp,
                imp_eval_only: true,
                imp_samples: 20,
                imp_seed: Some(1),
                ..base.clone()
            },
        ),
        (
            "agq_eval_only",
            FitOptions {
                method: EstimationMethod::Laplace,
                agq_eval_only: true,
                ..base.clone()
            },
        ),
    ] {
        let r = fit(&model, &pop, &model.default_params, &opts).expect("fit");
        assert_ne!(r.outer_maxiter, 0, "premise ({name})");
        assert_eq!(
            crate::stalled_at_init(&r),
            Some(true),
            "premise ({name}): the estimates are the initial values"
        );
        assert!(
            !r.warnings.iter().any(|w| w.contains("W_STALLED_AT_INIT")),
            "{name}: {:?}",
            r.warnings
        );
        assert!(
            !r.warnings_structured
                .iter()
                .any(|e| e.category == WarningCode::StalledAtInit),
            "{name}: {:?}",
            r.warnings_structured
        );
    }
}

/// A fit with nothing free cannot stall, and must not be reported as if it had.
/// `n_parameters == 0` is the `FIX`-everything case — every "estimate" equals
/// its initial value by construction.
#[test]
fn a_fully_fixed_fit_is_not_a_stall() {
    let mut r = stalled_result();
    r.n_parameters = 0;
    assert!(estimating_fit_warning(&r).is_none());
}

/// #2506: a start the optimizer verified to be an optimum is a warm start and is not
/// flagged at all; any other stall may still have started at one (the verification
/// can be inconclusive, out of budget, or not run by the estimator), so the message
/// hedges rather than asserting the objective is meaningless.
#[test]
fn the_message_hedges_on_a_start_that_was_not_verified() {
    let mut r = stalled_result();
    r.left_init = Some(false);

    r.stationary_start = true;
    assert!(
        estimating_fit_warning(&r).is_none(),
        "a start the optimizer verified is not a stall"
    );

    for (left, label) in [(Some(false), "unverified"), (None, "natural scale")] {
        r.left_init = left;
        r.stationary_start = false;
        let (msg, _) = estimating_fit_warning(&r).expect("an unverified stall must warn");
        assert!(
            msg.contains("Unless the initial values were already an optimum")
                && msg.contains("which this fit could not verify"),
            "{label}: {msg}"
        );
    }
}
