//! Tier-1 tests for the post-fit ODE-solver diagnostics warning (#1080 Part B item 2).
//!
//! Before this, no production path reported solver statistics at all:
//! `ode_predictions_with_solver_stats` had no caller outside the test suite, so a model whose
//! segments `auto` escalated — or, worse, escalated and then had the escalation *rejected* —
//! finished with a clean-looking fit and no way for the user to know. These tests pin both the
//! message construction and the end-to-end wiring, because the wiring is the part that was
//! missing.
use super::*;
use crate::ode::OdeSolverStats;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// A rapid-equilibrium two-state model: the exchange rate `KFAST` sets `|Re λ|max`, so the
/// same source is a stiff or a benign fixture depending on one theta.
fn two_state_model(kfast: f64) -> CompiledModel {
    let src = format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  theta KFAST({kfast}, 1e-6, 1e6)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KF = KFAST
[structural_model]
  ode(obs_cmt=central, states=[central, periph])
[odes]
  d/dt(central) = -(CL / V) * central - KF * central + KF * periph
  d/dt(periph)  = KF * central - KF * periph
[error_model]
  DV ~ proportional(PROP)
"#
    );
    parse_model_string(&src).expect("parse")
}

fn subject(id: &str, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| scale * 50.0 / (1.0 + t)).collect();
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn population() -> Population {
    Population {
        subjects: vec![subject("1", 1.0), subject("2", 1.2)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// One outer iteration is enough — the warning is built from the post-fit prediction pass,
/// not from the optimizer's history.
fn one_iteration_opts() -> FitOptions {
    FitOptions {
        outer_maxiter: 1,
        run_covariance_step: false,
        ..Default::default()
    }
}

fn ode_solver_entry(result: &FitResult) -> Option<&WarningEntry> {
    result
        .warnings_structured
        .iter()
        .find(|e| e.category == WarningCode::OdeSolver)
}

// ── message construction ─────────────────────────────────────────────────────

#[test]
fn a_clean_solve_produces_no_warning() {
    let stats = OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 380,
        rejected_steps: 20,
        ..Default::default()
    };
    assert!(ode_solver_diagnostics_warning(&stats, &FitOptions::default()).is_none());
}

#[test]
fn an_escalation_that_worked_is_reported_as_info() {
    let stats = OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 400,
        auto_stiff_segments: 240,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a note");
    assert!(msg.contains("240"), "{msg}");
    assert_eq!(entry.severity, WarningSeverity::Info);
    assert_eq!(entry.category, WarningCode::OdeSolver);
    assert_eq!(
        entry.details.as_ref().unwrap()["auto_stiff_segments"],
        serde_json::json!(240)
    );
}

/// A mid-segment switch (#1080 Part C) is reported as a clause on the escalation note, not as
/// a problem of its own: `auto` deciding late is what the feature is for on a model whose
/// stiffness only appears after a dose. What the note has to say is that the decision was taken
/// part-way through, because it changes how the step counters above it read.
#[test]
fn a_mid_segment_switch_is_reported_on_the_escalation_note() {
    let stats = OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 400,
        auto_stiff_segments: 12,
        auto_switched_segments: 9,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a note");
    assert_eq!(entry.severity, WarningSeverity::Info);
    // Spelled out in full, not by fragment: a line-continuation left unescaped inside the
    // clause's literal collapses into a run of indentation that only the whole sentence sees.
    assert!(
        msg.contains(
            "9 of them changed stepper part-way through the segment, because a mid-segment \
             re-probe disagreed with the verdict the segment started on (ode_auto_switch)."
        ),
        "{msg}"
    );
    assert!(
        !msg.contains("  "),
        "no run of blank space may reach a user: {msg}"
    );
    assert_eq!(
        entry.details.as_ref().unwrap()["auto_switched_segments"],
        serde_json::json!(9)
    );
}

/// …and a fit that switched *and* did not integrate cleanly says both: the switch clause rides
/// the warning too, so the counters it explains are never presented without it.
///
/// It gets its own wording there. The warning's body is a list of clamped steps and abandoned
/// segments, so the info note's "N of them" would attach to whichever clause happened to come
/// last and read as "9 of the clamped steps".
#[test]
fn a_switch_is_reported_alongside_an_unclean_solve() {
    let stats = OdeSolverStats {
        min_step_clamped_steps: 12,
        auto_stiff_segments: 12,
        auto_switched_segments: 9,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(
        msg.contains(
            "9 segment(s) changed stepper part-way through, because a mid-segment re-probe \
             disagreed with the verdict the segment started on (ode_auto_switch)."
        ),
        "{msg}"
    );
    assert!(
        !msg.contains("of them"),
        "the warning body gives \"of them\" no antecedent: {msg}"
    );
    assert!(
        !msg.contains("  "),
        "no run of blank space may reach a user: {msg}"
    );
    assert_eq!(classify_warning(&msg).category, WarningCode::OdeSolver);
}

/// A rejected escalation is the actionable one: the probe was right about the stiffness and
/// wrong about the method, and only the user can pick another.
#[test]
fn a_rejected_escalation_is_a_warning_that_names_the_next_thing_to_try() {
    let stats = OdeSolverStats {
        min_step_clamped_steps: 12,
        auto_stiff_segments: 240,
        auto_stiff_rejected: 3,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert_eq!(entry.category, WarningCode::OdeSolver);
    assert!(msg.contains("3 of 240"), "{msg}");
    assert!(msg.contains("rodas5p"), "{msg}");
    assert!(msg.contains("clamped at the minimum step size"), "{msg}");
    // The free-text classifier must reach the same code, so a caller that only sees
    // `FitResult.warnings` and re-classifies still gets `ode_solver`.
    assert_eq!(classify_warning(&msg).category, WarningCode::OdeSolver);
}

/// The guard cannot repair a segment when its explicit re-solve fails too. This is more severe
/// than an ordinary rejected escalation: the caller received the failed fallback because there
/// is no third solve, and the warning must say that plainly (#1080 review follow-up).
#[test]
fn a_failed_explicit_fallback_reports_that_both_attempts_failed() {
    let stats = OdeSolverStats {
        auto_stiff_segments: 3,
        auto_stiff_rejected: 1,
        auto_fallback_failed: 1,
        unfinished_segments: 2,
        discarded_unfinished_segments: 1,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(msg.contains("1 segment(s) had both attempts fail"), "{msg}");
    // Self-contained: the clause must not lean on the `rejected` clause happening to precede it.
    assert!(!msg.contains("of those explicit re-solves"), "{msg}");
    assert!(msg.contains("1 returned segment(s) stopped"), "{msg}");
    let details = entry.details.as_ref().unwrap();
    assert_eq!(details["auto_fallback_failed"], serde_json::json!(1));
    assert_eq!(details["kept_unfinished_segments"], serde_json::json!(1));
}

#[test]
fn an_unfinished_named_solve_is_warned_without_an_auto_rejection() {
    let stats = OdeSolverStats {
        attempted_steps: 10,
        accepted_steps: 10,
        unfinished_segments: 1,
        ..Default::default()
    };
    let opts = FitOptions {
        ode_method: crate::ode::OdeMethod::Rk45,
        ..Default::default()
    };
    let (msg, entry) = ode_solver_diagnostics_warning(&stats, &opts).expect("a warning");
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(msg.contains("1 returned segment(s) stopped"), "{msg}");
    assert!(!msg.contains("stiff escalation"), "{msg}");
}

/// `stiff_aborted_segments` is a subset of `unfinished_segments` by construction — the budgeted
/// abort only fires while `t < tf`. The warning's clauses must partition the damaged segments,
/// so a reader who adds them up gets the number of segments rather than twice it (#1080 review).
#[test]
fn an_abandoned_segment_is_not_also_reported_as_an_unfinished_one() {
    let stats = OdeSolverStats {
        unfinished_segments: 3,
        stiff_aborted_segments: 3,
        ..Default::default()
    };
    let opts = FitOptions {
        ode_stiff_abort_after: Some(5),
        ..Default::default()
    };
    let (msg, entry) = ode_solver_diagnostics_warning(&stats, &opts).expect("a warning");
    assert!(msg.contains("3 segment(s) were abandoned early"), "{msg}");
    assert!(
        !msg.contains("returned segment(s) stopped"),
        "the abort clause already reports all three: {msg}"
    );
    // The raw roll-up stays in the payload; the docs say it is not disjoint from the abort count.
    let details = entry.details.as_ref().unwrap();
    assert_eq!(details["kept_unfinished_segments"], serde_json::json!(3));
    assert_eq!(details["stiff_aborted_segments"], serde_json::json!(3));
}

/// One unfinished segment beyond the abandoned ones still earns its own clause, and names the
/// causes an abort clause would not explain.
#[test]
fn an_unfinished_segment_beyond_the_abandoned_ones_is_still_reported() {
    let stats = OdeSolverStats {
        unfinished_segments: 3,
        stiff_aborted_segments: 2,
        ..Default::default()
    };
    let (msg, _) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert!(msg.contains("1 returned segment(s) stopped"), "{msg}");
    assert!(msg.contains("ode_max_steps"), "{msg}");
}

#[test]
fn an_abort_names_the_budget_that_caused_it() {
    let stats = OdeSolverStats {
        min_step_clamped_steps: 8,
        stiff_aborted_segments: 4,
        ..Default::default()
    };
    let opts = FitOptions {
        ode_method: crate::ode::OdeMethod::Rodas4,
        ode_stiff_abort_after: Some(2),
        ..Default::default()
    };
    let (msg, entry) = ode_solver_diagnostics_warning(&stats, &opts).expect("a warning");
    assert!(msg.contains("ode_stiff_abort_after = 2"), "{msg}");
    assert!(msg.contains("4 segment(s)"), "{msg}");
    // The method the user actually asked for is named, so the advice is not about `auto`.
    assert!(msg.contains("ode_method = rodas4"), "{msg}");
    assert_eq!(
        entry.details.as_ref().unwrap()["stiff_aborted_segments"],
        serde_json::json!(4)
    );
}

/// Review follow-up (#1080): a rejected escalation must not be reported as freeze-padding. The
/// guard re-solved the segment explicitly, so the trajectory the caller received was not the
/// clamped one — and since a stall *is* the rejection trigger, every rejection would otherwise
/// carry that false claim.
#[test]
fn a_rejected_escalations_clamps_do_not_claim_freeze_padding() {
    let stats = OdeSolverStats {
        min_step_clamped_steps: 12,
        discarded_clamped_steps: 12,
        auto_stiff_segments: 240,
        auto_stiff_rejected: 3,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert!(
        !msg.contains("clamped at the minimum step size"),
        "no clamp survived into a returned trajectory: {msg}"
    );
    assert!(msg.contains("3 of 240"), "{msg}");
    assert_eq!(
        entry.details.as_ref().unwrap()["kept_clamped_steps"],
        serde_json::json!(0)
    );

    // …and a clamp in the *kept* solve still is reported, alongside the rejection.
    let with_kept = OdeSolverStats {
        min_step_clamped_steps: 14,
        discarded_clamped_steps: 12,
        ..stats
    };
    let (msg, _) =
        ode_solver_diagnostics_warning(&with_kept, &FitOptions::default()).expect("a warning");
    assert!(msg.contains("2 step(s) clamped"), "{msg}");
}

/// The informational note must re-classify as `Info`, not be promoted to a `Warning`, when a
/// consumer only has the plain message text (reading back a `{model}-fit.yaml`).
#[test]
fn the_escalation_note_keeps_its_severity_through_classification() {
    let stats = OdeSolverStats {
        auto_stiff_segments: 7,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a note");
    assert_eq!(entry.severity, WarningSeverity::Info);
    let reclassified = classify_warning(&msg);
    assert_eq!(reclassified.severity, WarningSeverity::Info);
    assert_eq!(reclassified.category, WarningCode::OdeSolver);
}

/// The stats scope has to cover the closed-form absorption models too: they carry no
/// `ode_spec`, but their TV-covariate / `TIME` / IOV / SS subjects integrate the ODE twin,
/// which `sync_ode_solver_opts` configures with the same solver options.
#[test]
fn the_solver_scope_gate_covers_the_absorption_ode_twin() {
    let ode = two_state_model(1.0);
    assert!(integrates_odes(&ode));

    let transit = parse_model_string(
        r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  theta TVMTT(1.0, 0.01, 24.0)
  theta TVN(3.0, 0.1, 50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  NN  = TVN
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NN, mtt=MTT)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse");
    assert!(
        transit.ode_spec.is_none(),
        "the fixture must be the closed form, not an ODE model"
    );
    assert!(
        transit.absorption_ode_equivalent.is_some(),
        "the fixture must carry a twin"
    );
    assert!(
        integrates_odes(&transit),
        "a model that integrates only through its twin still has solver statistics"
    );
}

// ── end-to-end wiring ────────────────────────────────────────────────────────

/// The wiring test: a stiff ODE model fitted through `fit()` must come back *saying* that
/// `auto` escalated. This is the half that did not exist before #1080 — the counters were
/// already correct, and entirely invisible.
#[test]
fn a_stiff_ode_fit_reports_the_solver_decisions() {
    let model = two_state_model(1000.0);
    let pop = population();
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    let entry = ode_solver_entry(&result).expect("an ode_solver warning");
    let escalated = entry.details.as_ref().unwrap()["auto_stiff_segments"]
        .as_u64()
        .unwrap();
    assert!(escalated > 0, "the probe should have escalated: {entry:?}");
    assert!(
        result.warnings.iter().any(|w| w.contains("W_ODE_SOLVER_")),
        "the plain-text warnings must carry it too: {:?}",
        result.warnings
    );
}

/// …and a model the probe reads as non-stiff stays silent, so the diagnostic does not become
/// noise every ODE fit prints.
#[test]
fn a_benign_ode_fit_reports_nothing() {
    let model = two_state_model(0.1);
    let pop = population();
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    assert!(
        ode_solver_entry(&result).is_none(),
        "unexpected solver warning: {:?}",
        result.warnings
    );
}

/// A jet-only rejection gets its own clause, because its remedy is not the one an ordinary
/// rejection carries (#1204). "Name a different stiff method" is useless here — the stiff
/// method integrated fine; what failed is that the trajectory reached a magnitude the
/// sensitivities cannot represent.
#[test]
fn a_jet_rejection_says_the_derivatives_overflowed_not_that_the_method_failed() {
    let stats = OdeSolverStats {
        auto_stiff_segments: 12,
        auto_stiff_rejected: 2,
        auto_stiff_rejected_jets: 2,
        auto_fallback_failed: 2,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(msg.contains("2 of those 2 rejection(s)"), "{msg}");
    assert!(msg.contains("edge of double precision"), "{msg}");
    assert!(msg.contains("units and scaling"), "{msg}");
    let details = entry.details.as_ref().unwrap();
    assert_eq!(details["auto_stiff_rejected_jets"], serde_json::json!(2));
    assert_eq!(classify_warning(&msg).category, WarningCode::OdeSolver);
}

/// The clause stays out of the way of an ordinary rejection, which is the common case: a
/// stiff solve the guard threw out on a `min_dt` stall has nothing to do with jets, and
/// telling that user to check their units would be a wrong lead.
#[test]
fn an_ordinary_rejection_carries_no_jet_clause() {
    let stats = OdeSolverStats {
        auto_stiff_segments: 12,
        auto_stiff_rejected: 2,
        min_step_clamped_steps: 5,
        discarded_clamped_steps: 5,
        ..Default::default()
    };
    let (msg, entry) =
        ode_solver_diagnostics_warning(&stats, &FitOptions::default()).expect("a warning");
    assert!(!msg.contains("edge of double precision"), "{msg}");
    assert!(msg.contains("rodas5p"), "{msg}");
    let details = entry.details.as_ref().unwrap();
    assert_eq!(details["auto_stiff_rejected_jets"], serde_json::json!(0));
}

/// The second half of the post-fit sweep (#1204): the diagnostic scope has to observe a
/// **dual** solve, not only the `f64` prediction pass.
///
/// Without this the jet-rejection clause could never fire in production — `f64` carries no
/// jets, so the only path that can take that decision is the sensitivity solve — and the
/// counter would be exactly the dead diagnostic #1080 item 3 existed to remove. Asserting the
/// sweep deposits counters of its own is what pins that it runs at all.
#[test]
fn the_post_fit_sweep_observes_the_sensitivity_solve_too() {
    let model = two_state_model(1000.0);
    let pop = population();
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    let mut params = model.default_params.clone();
    params.theta = result.theta.clone();
    let etas: Vec<DVector<f64>> = result.subjects.iter().map(|s| s.eta.clone()).collect();
    let scope = crate::ode::solver::SolverStatsScope::enter();
    sweep_sensitivity_solver_stats(&model, &pop, &params, &etas, None);
    let stats = scope.collected();
    assert!(
        stats.accepted_steps > 0,
        "the sensitivity sweep integrated nothing: {stats:?}"
    );
    assert!(
        stats.auto_stiff_segments > 0,
        "and it should see the same escalations the prediction pass does: {stats:?}"
    );
}

/// …and it costs nothing on a model that has no ODE sensitivity path to sweep. The helper is
/// called on every ODE fit, so a model on FD (or on a closed-form solution) must exit before
/// it integrates anything.
#[test]
fn the_sensitivity_sweep_is_a_no_op_off_the_analytic_ode_path() {
    let model = closed_form_model();
    let pop = population();
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    let mut params = model.default_params.clone();
    params.theta = result.theta.clone();
    let etas: Vec<DVector<f64>> = result.subjects.iter().map(|s| s.eta.clone()).collect();
    let scope = crate::ode::solver::SolverStatsScope::enter();
    sweep_sensitivity_solver_stats(&model, &pop, &params, &etas, None);
    let stats = scope.collected();
    assert_eq!(stats.accepted_steps, 0, "{stats:?}");
    assert_eq!(stats.attempted_steps, 0, "{stats:?}");
}

/// A one-compartment closed-form twin of [`two_state_model`]: no `[odes]`, so
/// `ode_inner_grad_supported_model` is false and the sweep has nothing to do.
fn closed_form_model() -> CompiledModel {
    parse_model_string(
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
    .expect("parse")
}
