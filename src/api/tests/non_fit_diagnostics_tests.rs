//! Tier-1 tests for the model/data and ODE-solver diagnostics on the **non-`fit()`** entry
//! points (#1280 / #1304).
//!
//! # What was measured on `main` before this
//!
//! One fixture, two defects, both reproduced at `233246d1` (2026-09-17) rather than read off
//! the issues. The fixture is a 1-cpt `[odes]` model whose right-hand side reads `TAFD` under
//! an `SS=1, II=12` dose — the #1139 case, which `fit()` has named since #1278.
//!
//! * **#1280.** `fit()` returned 8 warnings, among them
//!   `W_STEADY_STATE_ABSOLUTE_TIME` and `W_ODE_SOLVER_DIAGNOSTICS`. `predict()` on the same
//!   model returned `[NaN, NaN]` with nothing attached (it has no channel at all), and
//!   `simulate_with_options_diag()` returned its rows with `warnings == []`.
//! * **#1304 / the residual of #959.** On a stiff two-state model at `ode_max_steps = 500`,
//!   `predict()` returned `49.98522138377198` at **t = 0.5, 2, 8 and 24** — four identical
//!   numbers to 17 figures on a decaying curve, which is the #959 freeze-pad signature — and
//!   `simulate_with_options_diag()` again returned `warnings == []`. Nothing on either path
//!   distinguished those rows from a healthy solve.
//!
//! After this change the same two calls carry
//! `W_STEADY_STATE_ABSOLUTE_TIME` + `W_ODE_SOLVER_DIAGNOSTICS`, and the stiff one carries
//! `"2 returned segment(s) stopped before their requested end time and freeze-padded the
//! remaining output times with the last state"`.
//!
//! # Tiering
//!
//! Tier 1. Nothing here calls `fit()`. The model/data half needs no integration at all (an
//! analytic model); the solver half starves `ode_max_steps` so the misbehaving segments give
//! up after a handful of steps rather than grinding to a budget.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::types::{DoseEvent, Population, Subject};

// ── fixtures ─────────────────────────────────────────────────────────────────

/// Analytic 1-cpt IV. No `[odes]`, so no scope is ever opened and the solver half of the
/// bundle is structurally absent — which is what makes it the control for the model/data half.
fn analytic_model() -> CompiledModel {
    parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parse")
}

/// A two-state rapid-equilibrium `[odes]` model. `kfast` sets the stiffness and `max_steps`
/// the budget, so the same source is a clean or a budget-starved fixture depending on two
/// numbers — the straddle the solver-half tests need.
fn ode_model(kfast: f64, max_steps: usize) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  theta KFAST({kfast}, 1e-6, 1e8)
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
[fit_options]
  ode_method = rk45
  ode_max_steps = {max_steps}
"#
    ))
    .expect("parse")
}

/// `n` subjects with one bolus at `t = 0` and four observations on a decaying curve.
fn pop(n: usize) -> Population {
    let subjects = (0..n)
        .map(|i| Subject {
            id: format!("{}", i + 1),
            obs_times: vec![0.5, 2.0, 8.0, 24.0],
            observations: vec![50.0, 20.0, 5.0, 1.0],
            obs_cmts: vec![1; 4],
            cens: vec![0; 4],
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// The same population with the dose flagged `SS=1, II=0` — the `W_STEADY_STATE_II` finding,
/// which is data-only (no integration, no parameter values) and so is the cheapest member of
/// the model/data bundle to trip.
fn pop_ss_bad_ii(n: usize) -> Population {
    let mut p = pop(n);
    for s in &mut p.subjects {
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 0.0)];
    }
    p
}

fn has(warnings: &[String], token: &str) -> bool {
    warnings.iter().any(|w| w.contains(token))
}

const SS_II: &str = "SS=1 doses with missing or non-positive II";
const SOLVER: &str = "W_ODE_SOLVER_DIAGNOSTICS";
const FREEZE_PAD: &str = "stopped before their requested end time and freeze-padded";

// ── #1280: the model/data bundle reaches the non-fit entry points ────────────

/// The #1280 defect itself, on the entry point that had no channel at all.
///
/// Regression this catches: `predict_diag` not running `check_model_data_warnings`. Mutation
/// — delete the `check_model_data_warnings` call from `postfit::non_fit_diagnostics` and this
/// test fails, as do
/// `simulate_carries_the_model_data_bundle_that_simulate_used_to_drop` and
/// `every_diagnostic_carrying_entry_point_reports_the_same_bundle`.
///
/// The analytic model is deliberate: it integrates nothing, so the finding cannot arrive via
/// the solver half by accident.
#[test]
fn predict_carries_the_model_data_bundle_that_predict_used_to_drop() {
    let m = analytic_model();
    let p = pop_ss_bad_ii(1);
    let out = predict_diag(&m, &p, &m.default_params).unwrap();
    assert!(
        has(&out.warnings, SS_II),
        "predict_diag must report the same model/data findings fit() does: {:?}",
        out.warnings
    );
    // The straddle, under the *old* behaviour as well as the new: the identical call on a
    // population whose dose is not SS reports nothing. Without this the assertion above passes
    // on an implementation that pushes a fixed string into every prediction.
    let clean = predict_diag(&m, &pop(1), &m.default_params).unwrap();
    assert!(
        clean.warnings.is_empty(),
        "a well-formed model/data pair must come back silent, or the warning above proves \
         nothing: {:?}",
        clean.warnings
    );
}

/// The same for `simulate_with_options_diag`, which *had* a channel and put nothing in it.
///
/// Mutation: as above. Also dies if the `non_fit_diagnostics` call is dropped from only the
/// non-propensity branch of `simulate_with_options_diag` (this fixture takes that branch).
#[test]
fn simulate_carries_the_model_data_bundle_that_simulate_used_to_drop() {
    let m = analytic_model();
    let out = simulate_with_options_diag(
        &m,
        &pop_ss_bad_ii(1),
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .expect("simulate");
    assert!(
        has(&out.warnings, SS_II),
        "simulate_with_options_diag must carry the bundle: {:?}",
        out.warnings
    );
    let clean = simulate_with_options_diag(
        &m,
        &pop(1),
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .expect("simulate");
    assert!(
        clean.warnings.is_empty(),
        "and stay silent on a clean one: {:?}",
        clean.warnings
    );
}

/// The propensity-matching branch of `simulate_with_options_diag` returns from its own
/// `Ok(...)`, not through the branch above, so it needs its own assertion.
///
/// Mutation: delete the `non_fit_diagnostics` call from the propensity branch only — every
/// other test here stays green (none of them ask for matching) and this one fails. That is the
/// #1229 "two redundant gates cover for each other" shape avoided by construction: the two
/// returns are separate code, so they need separate tests.
#[test]
fn the_propensity_branch_of_simulate_carries_the_bundle_too() {
    let m = analytic_model();
    let out = simulate_with_options_diag(
        &m,
        &pop_ss_bad_ii(3),
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(7),
            match_method: Some(crate::propensity_match::MatchMethod::Nearest),
            ..Default::default()
        },
    )
    .expect("simulate with matching");
    assert!(
        has(&out.warnings, SS_II),
        "the matched branch returns from its own Ok(..) and must carry the bundle: {:?}",
        out.warnings
    );
}

/// `predict()` keeps its signature and its silence — the wrapper must not start allocating or
/// printing, and above all its **rows must be identical** to the `_diag` form's.
///
/// Regression this catches: the `par_iter` restructure that threads per-subject
/// `OdeSolverStats` through `assemble` reassociating or reordering a prediction. Bit-equality,
/// not a tolerance: nothing in this change may move a number.
#[test]
fn predict_returns_exactly_the_diag_forms_rows() {
    let m = ode_model(1.0, 10_000);
    let p = pop(3);
    let rows = predict(&m, &p, &m.default_params).unwrap();
    let out = predict_diag(&m, &p, &m.default_params).unwrap();
    assert_eq!(rows.len(), out.results.len());
    assert!(!rows.is_empty(), "the fixture must produce rows to compare");
    for (a, b) in rows.iter().zip(out.results.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.time.to_bits(), b.time.to_bits());
        assert_eq!(
            a.pred.to_bits(),
            b.pred.to_bits(),
            "predict() and predict_diag().unwrap() must return bit-identical predictions \
             (subject {}, t = {})",
            a.id,
            a.time
        );
        assert!(
            a.pred.is_finite(),
            "the clean fixture must integrate — a NaN here would make the comparison above \
             pass on two broken paths (subject {}, t = {})",
            a.id,
            a.time
        );
    }
}

// ── #1304: the ODE solver diagnostics reach the non-fit entry points ─────────

/// The #959 residual, on `predict()`: a budget-starved segment freeze-pads its tail and the
/// call now says so.
///
/// Regression this catches: no `SolverStatsScope` open around the prediction pass, which is
/// the whole of #1304 item 1. Mutation — make `postfit::solver_stats_scope` return `None`
/// unconditionally, and this test and its two siblings below fail while every model/data test
/// above stays green.
///
/// The assertion is on the **freeze-pad clause**, not merely on the token: a message that
/// fired for some other reason (a clamped step, an escalation note) would satisfy a
/// token-only check while saying nothing about the padded rows that are the defect.
#[test]
fn predict_reports_a_freeze_padded_segment_instead_of_serving_it_silently() {
    // 20 steps is nowhere near enough for this exchange rate, so the segment gives up almost
    // immediately — the diagnostic without the grind.
    let m = ode_model(1e5, 20);
    let out = predict_diag(&m, &pop(2), &m.default_params).unwrap();
    assert!(
        has(&out.warnings, SOLVER) && has(&out.warnings, FREEZE_PAD),
        "a segment that stopped early and padded its tail must be named: {:?}",
        out.warnings
    );
    // The straddle. Same model source, same population, same code path — only the budget and
    // the stiffness differ, and a solve that finishes must say nothing. Under the old behaviour
    // *both* sides were silent, so this pair is what makes the assertion above a test of the
    // scope rather than of a constant.
    //
    // Each arm runs at **its own** `default_params`. Handing the clean model the starved one's
    // θ (`KFAST = 1e5`) put the stiffness back and made this arm report two padded segments of
    // its own — a straddle that does not straddle, caught here rather than shipped.
    let clean_m = ode_model(1.0, 10_000);
    let clean = predict_diag(&clean_m, &pop(2), &clean_m.default_params).unwrap();
    assert!(
        !has(&clean.warnings, SOLVER),
        "a clean integration must not carry a solver diagnostic: {:?}",
        clean.warnings
    );
}

/// The same on `simulate()`.
///
/// Separate from the `predict()` test rather than folded into it because the two open their
/// scopes by different mechanisms — `predict` one per subject task inside a `par_iter`,
/// `simulate` one around its serial loop — so a mutation of either must redden its own side.
/// Deleting `with_solver_stats` from `simulate_with_options_diag` leaves the `predict` test
/// green and fails this one.
#[test]
fn simulate_reports_a_freeze_padded_segment_instead_of_serving_it_silently() {
    let opts = SimulateOptions {
        seed: Some(3),
        ..Default::default()
    };
    let m = ode_model(1e5, 20);
    let out = simulate_with_options_diag(&m, &pop(2), &m.default_params, 1, &opts).expect("sim");
    assert!(
        has(&out.warnings, SOLVER) && has(&out.warnings, FREEZE_PAD),
        "simulate must name a freeze-padded segment: {:?}",
        out.warnings
    );
    let clean_m = ode_model(1.0, 10_000);
    let clean = simulate_with_options_diag(&clean_m, &pop(2), &clean_m.default_params, 1, &opts)
        .expect("sim");
    assert!(
        !has(&clean.warnings, SOLVER),
        "…and stay silent on a clean one: {:?}",
        clean.warnings
    );
}

/// `predict()`'s scope must be **per subject task**, not one on the calling thread.
///
/// Regression this catches: hoisting the scope out of the `par_iter` closure to wrap the whole
/// pass — which compiles, reads cleaner, and silently reports only the subjects rayon happened
/// to run on the calling thread. The population is large enough that a multi-worker pool
/// distributes it, and every subject is equally bad, so a thread-local-on-one-thread scope
/// under-counts.
///
/// Asserted as a **ratio against the single-subject count**, because the absolute step counts
/// are stepper-dependent. One subject produces `n` padded segments; sixteen identical subjects
/// must produce `16n`, and a hoisted scope would produce far fewer.
#[test]
fn predicts_solver_counters_cover_every_subject_not_just_the_calling_threads() {
    let m = ode_model(1e5, 20);
    let one = predict_diag(&m, &pop(1), &m.default_params).unwrap();
    let many = predict_diag(&m, &pop(16), &m.default_params).unwrap();
    let n_one = segment_count(&one.warnings).expect("one subject must report padded segments");
    let n_many = segment_count(&many.warnings).expect("sixteen subjects must report them too");
    assert_eq!(
        n_many,
        16 * n_one,
        "every subject's integrations must be counted, not only the calling thread's \
         (1 subject: {n_one}, 16 subjects: {n_many})"
    );
}

/// The `N` from `"N returned segment(s) stopped before their requested end time"`.
fn segment_count(warnings: &[String]) -> Option<usize> {
    let w = warnings.iter().find(|w| w.contains(FREEZE_PAD))?;
    let head = &w[..w.find(FREEZE_PAD)?];
    head.rsplit(|c: char| !c.is_ascii_digit())
        .find(|s| !s.is_empty())?
        .parse()
        .ok()
}

// ── the shared bundle ────────────────────────────────────────────────────────

/// One implementation, and **every** entry point proves it: on a fixture that trips both
/// halves, `predict_diag`, `simulate_with_options_diag` (in *both* of its branches),
/// `simulate_with_uncertainty_diag` (#1645) and `simulate_adaptive` report the same findings.
///
/// Regression this catches: a per-entry-point filter — the fix #1280 explicitly rules out —
/// creeping back in. Mutation: drop any one code from any caller's list and the set equality
/// for that caller fails, naming it.
///
/// **All four arms are here because two of them were missing and the gap was demonstrable.**
/// With only `predict` + ordinary `simulate` compared, two named escape edits survived the
/// whole suite: returning solver diagnostics *without* the model/data findings from
/// `simulate_adaptive`, and discarding the collected solver stats in `simulate`'s propensity
/// branch alone. Both are green on the two-arm version and red here, verified by running them.
///
/// Compared on the `W_`/`E_` code tokens rather than on whole messages: `simulate()` also
/// carries per-subject simulation diagnostics that `predict()` has no analogue for, and the
/// solver clause's step counts legitimately differ (simulate integrates `n_sim` replicates of
/// each subject, and the adaptive driver runs its own engine). What must not differ is *which
/// findings* are reported.
#[test]
fn every_diagnostic_carrying_entry_point_reports_the_same_bundle() {
    // Both halves live: an `SS=1, II=0` dose (model/data) on a budget-starved stiff ODE
    // (solver). Asserted below rather than assumed — a fixture that trips only one half would
    // make this a test of the other one alone.
    let m = ode_model(1e5, 20);
    let p = pop_ss_bad_ii(2);
    let opts = |mm| SimulateOptions {
        seed: Some(11),
        match_method: mm,
        ..Default::default()
    };

    let from_predict = codes(&predict_diag(&m, &p, &m.default_params).unwrap().warnings);
    assert!(
        from_predict.contains(&SOLVER.to_string()),
        "the fixture must trip the solver half: {from_predict:?}"
    );
    assert!(
        from_predict.len() >= 2,
        "…and the model/data half too, or this compares one finding with itself: \
         {from_predict:?}"
    );
    let uncertainty = simulate_with_uncertainty_diag(
        &m,
        &p,
        &super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params),
        &uncertainty_opts(),
    )
    .expect("sim");

    for (label, got) in [
        (
            "simulate",
            codes(
                &simulate_with_options_diag(&m, &p, &m.default_params, 1, &opts(None))
                    .expect("sim")
                    .warnings,
            ),
        ),
        (
            "simulate (propensity)",
            codes(
                &simulate_with_options_diag(
                    &m,
                    &p,
                    &m.default_params,
                    1,
                    &opts(Some(crate::propensity_match::MatchMethod::Nearest)),
                )
                .expect("sim")
                .warnings,
            ),
        ),
        (
            "simulate_with_uncertainty_diag",
            codes(&uncertainty.warnings),
        ),
    ] {
        assert_eq!(
            from_predict, got,
            "{label} must report the same findings predict does; a per-entry-point filter is \
             exactly what #1280 rules out"
        );
    }
    // The uncertainty run's solver counters are its own, over every draw (#1645).
    let solver = uncertainty
        .warnings
        .iter()
        .find(|w| w.contains(SOLVER))
        .expect("solver half");
    assert!(
        solver.contains("from this simulate_with_uncertainty() run over all parameter draws")
            && solver.contains("at the drawn parameter sets"),
        "{solver}"
    );
}

/// Two asymptotic draws, two replicates each, seeded.
fn uncertainty_opts() -> SimulateUncertaintyOptions {
    SimulateUncertaintyOptions {
        n_uncertainty_draws: 2,
        n_sim_per_draw: 2,
        method: crate::estimation::uncertainty_samples::UncertaintyMethod::Asymptotic,
        seed: Some(17),
    }
}

/// `simulate_adaptive` is the third caller and gets its own arm, because it cannot share the
/// fixture above: it runs its own reactive engine on its own population shape (decision times,
/// a controller, no pre-scheduled dose grid).
///
/// Regression this catches: the named escape edit "return solver diagnostics without model/data
/// findings from adaptive simulation", which was green against the whole suite before this.
/// Mutation, run: replace `non_fit_diagnostics` in `simulate_adaptive` with a bare
/// `ode_solver_diagnostics_warning` and only this test dies.
#[test]
fn simulate_adaptive_reports_both_halves_of_the_bundle() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction};

    // Same starved-budget ODE the adaptive suite uses, plus an `SS=1, II=0` base dose so the
    // model/data half is live alongside the solver half.
    let m = parse_model_string(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_method = rk45
  ode_max_steps = 2
"#,
    )
    .expect("parse");

    let obs_times = vec![6.0, 30.0, 54.0];
    let subject = Subject {
        id: "1".into(),
        // SS with II = 0 — the `W_STEADY_STATE_II` finding, model/data half.
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 0.0)],
        obs_times: obs_times.clone(),
        observations: vec![0.0; obs_times.len()],
        obs_cmts: vec![1; obs_times.len()],
        cens: vec![0; obs_times.len()],
        occasions: vec![1u32; obs_times.len()],
        dose_occasions: vec![1u32],
        ..Default::default()
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        // The frozen-replay verifier compares two freeze-padded trajectories here; it is not
        // what is under test and its outcome is not the assertion.
        verify: false,
        ..Default::default()
    };
    let controller = || |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let res = simulate_adaptive(&m, &population, &m.default_params, 1, controller, &opts)
        .expect("adaptive sim runs");

    assert!(
        has(&res.warnings, SS_II),
        "the model/data half must reach simulate_adaptive: {:?}",
        res.warnings
    );
    assert!(
        has(&res.warnings, SOLVER),
        "…alongside the solver half, or the fixture only tests one of them: {:?}",
        res.warnings
    );
}

/// T12 (#1645): on a dose-free design the adaptive driver withholds `W_NO_DOSES` — its
/// controller supplies the regimen — while `simulate_with_options_diag` on the same input still
/// reports it. Measured before the arm existed: both returned it, from a `read_nonmem_csv` of an
/// `ID,TIME,DV,MDV` grid (the ferx-r glue dropped it by hand on the adaptive path).
///
/// A straddle on the phase, plus a control: another reader finding still reaches the adaptive
/// warnings, so the arm cannot pass by dropping reader warnings wholesale.
#[test]
fn simulate_adaptive_withholds_no_doses_and_only_there() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction};

    let m = parse_model_string(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .expect("parse");
    assert!(
        !m.is_algebraic(),
        "the filter's own predicate must not be what hides it"
    );
    let obs_times = vec![6.0, 30.0, 54.0];
    let subject = Subject {
        id: "1".into(),
        doses: Vec::new(),
        obs_times: obs_times.clone(),
        observations: vec![0.0; obs_times.len()],
        obs_cmts: vec![1; obs_times.len()],
        cens: vec![0; obs_times.len()],
        ..Default::default()
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: vec![
            "W_NO_DOSES: parsed zero dose events across all 1 subject(s)".into(),
            "W_ADDL_MISSING_II: subject '1' has ADDL with no II".into(),
        ],
    };
    let opts = AdaptiveSimulateOptions {
        seed: Some(1),
        decision_times: vec![0.0, 24.0],
        verify: false,
        ..Default::default()
    };
    let controller = || |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
    let adaptive = simulate_adaptive(&m, &population, &m.default_params, 1, controller, &opts)
        .expect("adaptive sim runs")
        .warnings;
    assert!(!has(&adaptive, "W_NO_DOSES"), "{adaptive:?}");
    assert!(
        has(&adaptive, "W_ADDL_MISSING_II"),
        "…while other reader findings still reach it: {adaptive:?}"
    );

    let static_sim = simulate_with_options_diag(
        &m,
        &population,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(1),
            ..Default::default()
        },
    )
    .expect("sim")
    .warnings;
    assert!(
        has(&static_sim, "W_NO_DOSES"),
        "a static simulation's regimen comes only from the data: {static_sim:?}"
    );
}

/// The `W_`/`E_` tokens a warnings list carries, sorted and deduplicated. A message with no
/// token contributes its first six words, so an untokenised finding still participates rather
/// than vanishing from the comparison.
fn codes(warnings: &[String]) -> Vec<String> {
    let mut out: Vec<String> = warnings
        .iter()
        .map(|w| {
            w.split_whitespace()
                .find(|t| t.starts_with("W_") || t.starts_with("E_"))
                .map(|t| t.trim_end_matches(':').to_string())
                .unwrap_or_else(|| w.split_whitespace().take(6).collect::<Vec<_>>().join(" "))
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The row-only entry points stay silent **by contract**, and this is the list.
///
/// Regression this catches: a future entry point quietly joining the silent set, or one of
/// these three starting to print to stderr instead of returning. Each has a `_diag` twin one
/// call away — `simulate_with_uncertainty`'s since #1645.
#[test]
fn the_row_only_entry_points_stay_silent_by_contract() {
    let m = analytic_model();
    let p = pop_ss_bad_ii(1);
    // `predict` → rows only, and they are the `_diag` form's rows.
    assert_eq!(
        predict(&m, &p, &m.default_params).unwrap().len(),
        predict_diag(&m, &p, &m.default_params)
            .unwrap()
            .results
            .len()
    );
    // `simulate_with_options` → rows only, from an output that *did* carry warnings.
    let opts = SimulateOptions {
        seed: Some(5),
        ..Default::default()
    };
    let diag = simulate_with_options_diag(&m, &p, &m.default_params, 1, &opts).expect("sim");
    let rows = simulate_with_options(&m, &p, &m.default_params, 1, &opts).expect("sim");
    assert!(
        !diag.warnings.is_empty(),
        "the fixture must produce warnings, or 'the wrapper drops them' is vacuous"
    );
    assert_eq!(rows.len(), diag.results.len());
}

/// T11 (#1645): `simulate_with_uncertainty` is `simulate_with_uncertainty_diag(..).results`,
/// bit for bit, on the same seed — from an output that did carry warnings.
///
/// Regression this catches: the wrapper and the `_diag` form drifting into two
/// implementations — one of them drawing an extra variate, reordering draws, or skipping a
/// draw the other keeps.
#[test]
fn simulate_with_uncertainty_is_its_diag_forms_rows_bit_for_bit() {
    let m = analytic_model();
    let p = pop_ss_bad_ii(2);
    let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
    let diag = simulate_with_uncertainty_diag(&m, &p, &fit, &uncertainty_opts()).expect("sim");
    let rows = simulate_with_uncertainty(&m, &p, &fit, &uncertainty_opts()).expect("sim");
    assert!(
        has(&diag.warnings, SS_II),
        "the fixture must produce warnings, or 'the wrapper drops them' is vacuous: {:?}",
        diag.warnings
    );
    let key = |r: &SimulationResult| {
        let value = match r.outcome {
            SimOutcome::Continuous { value } => value.to_bits(),
            #[allow(unreachable_patterns)]
            _ => panic!("a Gaussian fixture simulates continuous rows: {r:?}"),
        };
        (
            r.draw,
            r.sim,
            r.id.clone(),
            r.time.to_bits(),
            r.cmt,
            r.ipred.to_bits(),
            value,
        )
    };
    assert_eq!(rows.len(), 2 * 2 * 2 * p.subjects[0].obs_times.len());
    assert!(rows.iter().all(|r| r.ipred.is_finite()), "{rows:?}");
    assert_eq!(
        rows.iter().map(key).collect::<Vec<_>>(),
        diag.results.iter().map(key).collect::<Vec<_>>()
    );
}

/// T10 (#1645): the uncertainty `_diag` relays data-reader findings through the shared filter,
/// not from `Population::warnings` raw — the relay the ferx-r glue needs on that path.
///
/// A straddle on the filter's predicate, as in
/// `reader_warnings_go_through_the_same_suppression_filter_fit_uses`: `W_NO_DOSES` is withheld
/// on a compartment-free model and reported on a compartment one.
#[test]
fn simulate_with_uncertainty_diag_filters_reader_warnings_like_fit() {
    const NO_DOSES: &str = "W_NO_DOSES: no dose records found";
    let algebraic = parse_model_string(
        "[parameters]\n  theta TVE0(8.0, 0.1, 100.0)\n  theta TVEMAX(4.0, 0.1, 100.0)\n  \
         theta TVET50(1.0, 0.01, 100.0)\n  omega ETA_E0 ~ 0.04\n  sigma ADD ~ 1.0 \
         (variance)\n[individual_parameters]\n  E0   = TVE0 * exp(ETA_E0)\n  EMAX = TVEMAX\n  \
         ET50 = TVET50\n[structural_model]\n  EFF = EMAX * TIME / (ET50 + TIME)\n  y   = E0 - \
         EFF\n[error_model]\n  DV ~ additive(ADD)\n",
    )
    .expect("parse");
    assert!(algebraic.is_algebraic());
    let mut p = pop(1);
    p.subjects[0].doses.clear();
    p.warnings.push(NO_DOSES.into());

    let run = |m: &CompiledModel| {
        let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
        simulate_with_uncertainty_diag(m, &p, &fit, &uncertainty_opts())
            .expect("sim")
            .warnings
    };
    let suppressed = run(&algebraic);
    assert!(
        !has(&suppressed, "W_NO_DOSES"),
        "a compartment-free model has nothing to dose: {suppressed:?}"
    );
    let compartmental = analytic_model();
    assert!(!compartmental.is_algebraic());
    let reported = run(&compartmental);
    assert!(
        has(&reported, "W_NO_DOSES"),
        "…while on a compartment model it is a real finding: {reported:?}"
    );
}

// ── phase wording (#1304) ────────────────────────────────────────────────────

/// A `predict()` diagnostic must not claim to describe a fit.
///
/// Regression this catches: reusing `ode_solver_diagnostics_warning`'s post-fit wording
/// verbatim from the new call sites, which would tell a `predict()` caller that their *final
/// estimates* misbehaved — at parameters no optimizer ever visited, on a model that may never
/// have been fitted.
///
/// A differential pair on the **same counters**, so the only difference is the phase. Under a
/// mutation that hard-codes either wording, one of the two arms fails.
#[test]
fn the_message_names_the_pass_that_produced_the_counters() {
    let stats = crate::ode::OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 380,
        min_step_clamped_steps: 7,
        ..Default::default()
    };
    let o = FitOptions::default();
    let (postfit, _) =
        ode_solver_diagnostics_warning(&stats, &o, SolverStatsPhase::PostfitPredictions)
            .expect("a warning");
    let (from_predict, entry) =
        ode_solver_diagnostics_warning(&stats, &o, SolverStatsPhase::Predict).expect("a warning");

    assert!(
        postfit.contains("at the final estimates")
            && postfit.contains("from the post-fit prediction pass"),
        "the post-fit phase keeps its wording unchanged: {postfit}"
    );
    assert!(
        !from_predict.contains("final estimates"),
        "a predict() pass runs at the caller's parameters, not at any fit's estimates: \
         {from_predict}"
    );
    assert!(
        from_predict.contains("at the supplied parameters")
            && from_predict.contains("from this predict() pass"),
        "…and must say which pass it did describe: {from_predict}"
    );
    assert_eq!(
        entry.details.as_ref().unwrap()["phase"],
        serde_json::json!("predict"),
        "the structured payload must carry the phase too — a consumer reading back a \
         {{model}}-fit.yaml keys off it"
    );
    assert!(
        !from_predict.contains("  "),
        "no run of blank space may reach a user: {from_predict}"
    );
    // `simulate` says "and replicates": it integrates `n_sim` copies of each subject, so its
    // counters are not comparable with a one-pass-per-subject sweep's.
    let (from_simulate, _) =
        ode_solver_diagnostics_warning(&stats, &o, SolverStatsPhase::Simulate).expect("a warning");
    assert!(
        from_simulate.contains("from this simulate() pass over all subjects and replicates"),
        "{from_simulate}"
    );
    // The uncertainty run (#1645) integrates every draw at its own parameter set, so it names
    // neither "the supplied parameters" nor a single pass, and keys its payload apart.
    let (from_uncertainty, entry) =
        ode_solver_diagnostics_warning(&stats, &o, SolverStatsPhase::SimulateUncertainty)
            .expect("a warning");
    assert!(
        from_uncertainty.contains("at the drawn parameter sets")
            && !from_uncertainty.contains("supplied parameters")
            && !from_uncertainty.contains("final estimates"),
        "{from_uncertainty}"
    );
    assert_eq!(
        entry.details.as_ref().unwrap()["phase"],
        serde_json::json!("simulate_with_uncertainty")
    );
}

/// The abandoned-walk clause and the escalation note carry the phase too — they are the two
/// other places the old wording said "at the final estimates".
///
/// Regression this catches: parameterising only the lead-in. Both clauses are reached by their
/// own `if`, so a fix applied to one and not the other passes every test above.
#[test]
fn the_abandoned_clause_and_the_escalation_note_carry_the_phase() {
    let o = FitOptions::default();
    let (abandoned, _) = ode_solver_diagnostics_warning(
        &crate::ode::OdeSolverStats {
            abandoned_non_finite_timeline: 2,
            ..Default::default()
        },
        &o,
        SolverStatsPhase::Simulate,
    )
    .expect("a warning");
    assert!(
        !abandoned.contains("final estimates") && abandoned.contains("at the supplied parameters"),
        "{abandoned}"
    );
    let (note, _) = ode_solver_diagnostics_warning(
        &crate::ode::OdeSolverStats {
            attempted_steps: 400,
            accepted_steps: 400,
            auto_stiff_segments: 12,
            ..Default::default()
        },
        &o,
        SolverStatsPhase::Predict,
    )
    .expect("an info note");
    assert!(
        !note.contains("final estimates") && note.contains("at the supplied parameters"),
        "{note}"
    );
}

// ── the synthesized FitOptions view ──────────────────────────────────────────

/// A non-fit entry point must name the `ode_method` the model is *actually* integrated at.
///
/// Regression this catches: passing `FitOptions::default()` at the new call sites, which would
/// tell a user running `ode_method = rodas5p` from their model file that `ode_method = auto`
/// had misbehaved, and send them to knobs they are not using. Mutation — return
/// `FitOptions::default()` from `solver_reporting_options` and the second assert fails.
///
/// Measured end-to-end as well as on the helper: the `ode_model` fixture pins `rk45`, and the
/// message it produces says `rk45`.
#[test]
fn the_reported_ode_method_is_the_models_own_not_the_fitoptions_default() {
    let m = ode_model(1e5, 20);
    assert_ne!(
        crate::ode::OdeMethod::Rk45,
        FitOptions::default().ode_method,
        "this test is vacuous unless the fixture's method differs from the default"
    );
    assert_eq!(
        solver_reporting_options(&m).ode_method,
        crate::ode::OdeMethod::Rk45
    );
    let out = predict_diag(&m, &pop(1), &m.default_params).unwrap();
    let w = out
        .warnings
        .iter()
        .find(|w| w.contains(SOLVER))
        .expect("a solver warning");
    assert!(
        w.contains("ode_method = rk45"),
        "the message must name the method the model runs at: {w}"
    );
}

/// `solver_reporting_options` carries exactly the `FitOptions` fields
/// `ode_solver_diagnostics_warning` reads — asserted against the source, not assumed.
///
/// Regression this catches: a third `options.<field>` read added to the warning builder. That
/// field would then be served from `FitOptions::default()` on every non-fit entry point —
/// silently, since the message would still be well-formed. This fails instead, naming the
/// field.
#[test]
fn the_solver_warning_reads_only_the_two_options_this_view_carries() {
    let src = include_str!("../postfit.rs");
    let start = src
        .find("pub(crate) fn ode_solver_diagnostics_warning(")
        .expect("the function must exist");
    // Its body ends at the next item at column 0 after it.
    let end = start
        + src[start..]
            .find("\n/// Extract standard errors")
            .expect("the following item must exist");
    let body = &src[start..end];
    // `options` and its field can be split across a line break by rustfmt (`budget = options`
    // / `.ode_stiff_abort_after`), so skip whitespace after the receiver rather than matching
    // the literal `options.` — that literal silently found only one of the two reads.
    let mut read: Vec<&str> = body
        .match_indices("options")
        .filter_map(|(i, _)| {
            let tail = body[i + "options".len()..].trim_start();
            let tail = tail.strip_prefix('.')?;
            let n = tail
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(tail.len());
            Some(&tail[..n])
        })
        .collect();
    read.sort();
    read.dedup();
    assert_eq!(
        read,
        vec!["ode_method", "ode_stiff_abort_after"],
        "`solver_reporting_options` synthesizes a FitOptions carrying exactly these fields; \
         a field read here and not carried there is served from FitOptions::default() on \
         every non-fit entry point"
    );
}

// ── the scope gate ───────────────────────────────────────────────────────────

/// A closed-form model opens no scope: entering one *activates* per-segment recording, so the
/// path that has nothing to record must not pay for it.
///
/// Regression this catches: dropping the `integrates_odes` gate. Asserted as a straddle — an
/// `[odes]` model must open one, or "returns None" is satisfied by a function that always
/// does, which would put #1304 straight back.
#[test]
fn the_scope_is_opened_only_for_a_model_that_integrates_something() {
    assert!(
        solver_stats_scope(&analytic_model()).is_none(),
        "a closed-form model has no segments to record"
    );
    let scope = solver_stats_scope(&ode_model(1.0, 10_000));
    assert!(scope.is_some(), "an [odes] model must open one");
    drop(scope);
}

// ── #1280 note 1: what fires on a simulation template ────────────────────────

/// The whole bundle goes into `simulate()`'s warnings — including for a caller simulating from
/// a design template with `DV = .`. #1280 asks what fires on those; this is the answer,
/// measured.
///
/// Only one member of the bundle reads `DV` at all: `W_ADDITIVE_INIT_SCALE`, which compares a
/// `combined` error model's additive SD start against the median `|DV|`. On a template that
/// median is over an empty set (non-finite values are filtered) or zero, and
/// `additive_init_scale_check` returns `None` for `data_scale <= 0.0` — so the finding is
/// **inert on a template**, and the concern does not need a filter to answer it.
///
/// Regression this catches: a future bundle member that reads `DV` without guarding the
/// template case, which would fire on every design-template simulation. The paired
/// real-data arm is what makes the silence meaningful: the same model on a population with
/// observations *does* warn.
#[test]
fn a_dv_free_simulation_template_does_not_trip_the_dv_reading_member_of_the_bundle() {
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 (sd)\n  sigma ADD ~ 1e-6 \
         (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = \
         TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ \
         combined(PROP, ADD)\n",
    )
    .expect("parse");

    // Matched on the message, not on the `W_ADDITIVE_INIT_SCALE` token: `non_fit_diagnostics`
    // pushes `Diagnostic::message`, and this check's code lives only in the `Diagnostic`'s
    // `code` field, never in its text. `fit()` does the same (`classify_warning` recovers the
    // category from the wording), so the two channels agree — but a token match here would
    // have passed vacuously on an empty list.
    const ADDITIVE_INIT: &str = "additive SD initial estimate";

    // The control: with real observations the finding fires, so the fixture is capable of it.
    let with_dv = predict_diag(&m, &pop(1), &m.default_params).unwrap();
    assert!(
        has(&with_dv.warnings, ADDITIVE_INIT),
        "the fixture must be able to trip the DV-reading member, or its silence below \
         proves nothing: {:?}",
        with_dv.warnings
    );

    // The template: a `DV = .` design population, read exactly as
    // `read_population_for_simulation` leaves it — NaN observations.
    let mut template = pop(1);
    for s in &mut template.subjects {
        s.observations = vec![f64::NAN; s.obs_times.len()];
    }
    let out = simulate_with_options_diag(
        &m,
        &template,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(2),
            ..Default::default()
        },
    )
    .expect("sim");
    assert!(
        !has(&out.warnings, ADDITIVE_INIT),
        "the one DV-reading member of the bundle must stay inert on a design template: {:?}",
        out.warnings
    );
}

/// The production `SolverStatsScope::enter` sites are exactly the four this enum has phases
/// for — pinned at **exact counts per file**, against the source.
///
/// Regression this catches: a fifth entry point opening a scope and reporting it under some
/// other phase's wording, or — the #1304 defect in reverse — a scope opened and never routed
/// into a warning at all. Either way the count moves and this fails, naming the file.
///
/// The four, and the phase each answers to:
///
/// | site | phase |
/// |---|---|
/// | `postfit::compute_subject_results` (per subject task) | `PostfitPredictions` |
/// | `postfit::sweep_sensitivity_solver_stats` (per subject task) | `PostfitPredictions` |
/// | `postfit::solver_stats_scope` (the shared gate) | `Simulate` / `SimulateAdaptive` |
/// | `predict::predict_diag` (per subject task) | `Predict` |
///
/// `simulate.rs` and `adaptive.rs` must read **zero**: both go through `solver_stats_scope`, so
/// a direct `enter()` there would bypass the `integrates_odes` gate and activate per-segment
/// recording on a closed-form model. None of these four files carries a `#[cfg(test)]` module
/// (asserted below), so the whole file is production source and no filtering is needed.
#[test]
fn the_production_scope_sites_are_the_phases_this_enum_lists() {
    const SITES: [(&str, &str, usize); 4] = [
        ("postfit.rs", include_str!("../postfit.rs"), 3),
        ("predict.rs", include_str!("../predict.rs"), 1),
        ("simulate.rs", include_str!("../simulate.rs"), 0),
        ("adaptive.rs", include_str!("../adaptive.rs"), 0),
    ];
    for (name, src, want) in SITES {
        assert_eq!(
            src.matches("#[cfg(test)]").count(),
            0,
            "{name} gained a #[cfg(test)] module — this scan counts the whole file, so it \
             would start counting test scopes as production ones"
        );
        assert_eq!(
            src.matches("SolverStatsScope::enter").count(),
            want,
            "{name}: the production SolverStatsScope sites are pinned; a new one needs a \
             SolverStatsPhase and a warnings channel to report into, or it is the #1304 \
             defect again"
        );
    }
}

// ── what the bundle carries, and what it does not (review round 1) ───────────

/// Every model-and-data source `fit()` seeds its warnings from reaches the non-`fit()` entry
/// points too — asserted end to end, one arm per source.
///
/// Regression this catches: the shipped first version carried **only**
/// `check_model_data_warnings`, while the rustdoc, `docs/warnings.qmd` and the changelog all
/// said "the same bundle `fit()` reports". A reader-produced `W_ADDL_MISSING_II` was absent
/// from `predict_diag` despite that contract. Mutation: delete any one `out.extend(...)` from
/// `postfit::non_fit_diagnostics` and the matching arm below fails, naming the source.
///
/// Each arm carries a **control**: the same call on a model or population without that defect
/// must not report it, so no arm can pass on an implementation that pushes a constant.
#[test]
fn the_bundle_carries_every_model_and_data_source_fit_carries() {
    // (2) `Population::warnings` — the data reader's own findings.
    let m = analytic_model();
    let mut p = pop_ss_bad_ii(1);
    p.warnings
        .push("W_ADDL_MISSING_II: subject '1' has ADDL with no II".into());
    let out = predict_diag(&m, &p, &m.default_params).unwrap();
    assert!(
        has(&out.warnings, "W_ADDL_MISSING_II"),
        "a reader warning must reach predict_diag — it is the example the review named: {:?}",
        out.warnings
    );
    assert!(
        !has(
            &predict_diag(&m, &pop_ss_bad_ii(1), &m.default_params)
                .unwrap()
                .warnings,
            "W_ADDL_MISSING_II"
        ),
        "…and must not appear when the population carries none"
    );

    // (3) `check_model_data_warnings` — already covered above, re-asserted here so this test
    // fails as a whole if the ordering of the four sources drops one.
    assert!(has(&out.warnings, SS_II), "{:?}", out.warnings);

    // (1) `CompiledModel::parse_warnings`. Asserted by injection: the parser produces these
    // only for specific malformed inputs, and the property under test is that the *source* is
    // read, not which strings that source can hold.
    let mut with_parse_warning = ode_model(1.0, 10_000);
    with_parse_warning
        .parse_warnings
        .push("W_TEST_PARSE: injected parse warning".into());
    let with_parse = predict_diag(
        &with_parse_warning,
        &pop(1),
        &with_parse_warning.default_params,
    )
    .unwrap();
    assert!(
        has(&with_parse.warnings, "W_TEST_PARSE"),
        "a parse warning must reach predict_diag — `W_ABSORPTION_TWIN_DECLINED` changes what \
         a prediction does: {:?}",
        with_parse.warnings
    );
    let clean = ode_model(1.0, 10_000);
    assert!(
        !has(
            &predict_diag(&clean, &pop(1), &clean.default_params)
                .unwrap()
                .warnings,
            "W_TEST_PARSE"
        ),
        "…and must not appear on a model that carries none"
    );

    // (4) the experimental-feature notice, through the same call.
    let (exp_model, exp_pop) = experimental_feature_fixture();
    let exp = predict_diag(&exp_model, &exp_pop, &exp_model.default_params).unwrap();
    assert!(
        has(&exp.warnings, "EXPERIMENTAL feature"),
        "an experimental-feature notice must reach predict_diag — a feature is experimental \
         whichever door it is used through: {:?}",
        exp.warnings
    );
    assert!(
        !has(
            &predict_diag(&clean, &pop(1), &clean.default_params)
                .unwrap()
                .warnings,
            "EXPERIMENTAL feature"
        ),
        "…and a non-experimental model must not carry one"
    );

    // And all four reach `simulate` and not only `predict`, so the shared helper is what is
    // being tested rather than one call site.
    let sim = simulate_with_options_diag(
        &m,
        &p,
        &m.default_params,
        1,
        &SimulateOptions {
            seed: Some(4),
            ..Default::default()
        },
    )
    .expect("sim");
    assert!(
        has(&sim.warnings, "W_ADDL_MISSING_II"),
        "{:?}",
        sim.warnings
    );
    assert!(has(&sim.warnings, SS_II), "{:?}", sim.warnings);
}

/// A model that trips [`crate::api::check_experimental_features`], and a population for it.
fn experimental_feature_fixture() -> (CompiledModel, Population) {
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.1 \
         (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = \
         TVV\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  \
         d/dt(central) = -(CL/V) * central\n[diffusion]\n  central ~ 0.01\n[error_model]\n  \
         DV ~ proportional(PROP)\n",
    )
    .expect("parse");
    assert!(
        !crate::api::check_experimental_features(&m).is_empty(),
        "the fixture must actually trip an experimental-feature notice"
    );
    (m, pop(1))
}

/// The findings that are about a **fit** stay out of the bundle, and the reason is structural.
///
/// Regression this catches: "aggregate everything `fit()` reports" taken literally, which would
/// put the estimator/optimizer option warnings on `predict()` — findings about an estimator
/// that is not running, served from a `FitOptions::default()` the caller never chose, which is
/// the trap `solver_reporting_options` exists to avoid.
///
/// Asserted as a **straddle**: the same model raises the option warning through
/// `check_model_options` and not through `predict_diag`. Without the first arm this would pass
/// on a model that simply never raises it.
#[test]
fn the_bundle_excludes_the_findings_that_are_about_a_fit() {
    let m = parse_model_string(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(20.0, 0.001, 500.0)\n  \
         omega ETA_CL ~ 0.0 FIX\n  sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  \
         CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
         v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
    )
    .expect("parse");
    // `vi_kl = mc` + `vi_omega_update = adam` → `W_VI_OMEGA_UNANCHORED`. Chosen because it is
    // model-independent: the point is that an option *combination* warning has no business on
    // an entry point that reads no options, and this one cannot be confused with a model defect.
    let opts = FitOptions {
        method: crate::types::EstimationMethod::Vi,
        vi_kl: crate::types::ViKl::Mc,
        vi_omega_update: crate::types::ViOmegaUpdate::Adam,
        ..Default::default()
    };
    let option_diags = crate::api::check_model_options(&m, &opts);
    let fit_only: Vec<String> = option_diags
        .iter()
        .filter(|d| !d.is_error())
        .map(|d| d.message.clone())
        .collect();
    assert!(
        !fit_only.is_empty(),
        "the fixture must raise at least one option warning, or the exclusion below is \
         vacuous: {option_diags:?}"
    );
    let out = predict_diag(&m, &pop(1), &m.default_params).unwrap();
    for w in &fit_only {
        assert!(
            !out.warnings.contains(w),
            "an estimator-configuration warning must not reach predict(), which runs no \
             estimator: {w}"
        );
    }
}

/// The **informational** escalation note names its pass too, and a discarded escalation does
/// not tell a non-`fit()` caller that "the fit" paid for it.
///
/// Regression this catches: parameterising only the clauses the first round of tests happened
/// to exercise. Measured before the fix, on `SolverStatsPhase::Predict`: the note read
/// `"… to a stiff stepper at the supplied parameters; every other segment used rk45, no
/// escalation was rejected …"` with **no** provenance sentence, and a `Simulate` rejection read
/// `"… and the fit paid for both solves …"`. Both passed the round-1 tests, which checked
/// provenance only on the unclean branch and `at the supplied parameters` only on the
/// informational one — each defect hid in the clause the other test did not look at.
///
/// Every non-fit phase is asserted, not just one: three share `at_label` (the uncertainty run
/// has its own) but all have distinct `provenance` and `payer` strings, so a phase that lost its
/// own wording would otherwise be covered by its neighbours.
#[test]
fn the_informational_note_and_the_rejection_clause_name_their_pass_too() {
    let o = FitOptions::default();
    let escalation_only = crate::ode::OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 400,
        auto_stiff_segments: 12,
        ..Default::default()
    };
    let rejected = crate::ode::OdeSolverStats {
        attempted_steps: 400,
        accepted_steps: 380,
        auto_stiff_segments: 12,
        auto_stiff_rejected: 5,
        ..Default::default()
    };

    for (phase, pass, payer) in [
        (
            SolverStatsPhase::Predict,
            "from this predict() pass",
            "this predict() pass paid for both solves",
        ),
        (
            SolverStatsPhase::Simulate,
            "from this simulate() pass",
            "this simulate() pass paid for both solves",
        ),
        (
            SolverStatsPhase::SimulateAdaptive,
            "from this simulate_adaptive() pass",
            "this simulate_adaptive() run paid for both solves",
        ),
        (
            SolverStatsPhase::SimulateUncertainty,
            "from this simulate_with_uncertainty() run over all parameter draws, subjects and \
             replicates",
            "this simulate_with_uncertainty() run paid for both solves",
        ),
    ] {
        let (note, entry) = ode_solver_diagnostics_warning(&escalation_only, &o, phase)
            .expect("an escalation is a note");
        assert_eq!(entry.severity, WarningSeverity::Info, "{note}");
        assert!(
            note.contains(pass),
            "{phase:?}: the informational note must name the pass its counters came from: \
             {note}"
        );
        assert!(
            !note.contains("post-fit prediction pass") && !note.contains("final estimates"),
            "{phase:?}: …and must not claim to describe a fit: {note}"
        );
        assert!(
            !note.contains("  "),
            "no run of blank space may reach a user: {note}"
        );

        let (rej, _) =
            ode_solver_diagnostics_warning(&rejected, &o, phase).expect("a rejection is a warning");
        assert!(
            rej.contains(payer),
            "{phase:?}: a discarded escalation must name what actually paid for both solves: \
             {rej}"
        );
        assert!(
            !rej.contains("the fit paid"),
            "{phase:?}: …and a caller who has not run a fit must not be told one paid: {rej}"
        );
    }

    // The straddle: the post-fit phase keeps every word it had.
    let (note, _) =
        ode_solver_diagnostics_warning(&escalation_only, &o, SolverStatsPhase::PostfitPredictions)
            .expect("a note");
    assert!(
        note.contains("at the final estimates")
            && note.contains("Counters are from the post-fit prediction pass over all subjects."),
        "{note}"
    );
    let (rej, _) =
        ode_solver_diagnostics_warning(&rejected, &o, SolverStatsPhase::PostfitPredictions)
            .expect("a warning");
    assert!(rej.contains("the fit paid for both solves"), "{rej}");
}

/// Reader warnings pass through the **same suppression filter** `fit()` and `ferx check` use,
/// so all three suppress exactly the same ones.
///
/// Regression this catches: extending `Population::warnings` unfiltered. That reads as
/// harmless — it reports *more* — but the one warning the filter removes is
/// `W_NO_DOSES` on a compartment-free (`is_algebraic`) model, where having no doses is not a
/// defect but the whole point: the equation is the model. Reporting it would tell every
/// `$PRED`-style regression that its data are malformed, on the entry point such a model is
/// most likely to be used from. Measured: removing the filter left all 20 tests green before
/// this one existed.
///
/// A **straddle on the filter's own predicate**: the identical warning on a compartment model
/// must still be reported, so this cannot pass on an implementation that drops reader warnings
/// altogether (which `the_bundle_carries_every_model_and_data_source_fit_carries` also guards
/// from the other side).
#[test]
fn reader_warnings_go_through_the_same_suppression_filter_fit_uses() {
    const NO_DOSES: &str = "W_NO_DOSES: no dose records found";

    // Compartment-free: `is_algebraic()` is true, so `W_NO_DOSES` is suppressed.
    let algebraic = parse_model_string(
        "[parameters]\n  theta TVE0(8.0, 0.1, 100.0)\n  theta TVEMAX(4.0, 0.1, 100.0)\n  \
         theta TVET50(1.0, 0.01, 100.0)\n  omega ETA_E0 ~ 0.04\n  sigma ADD ~ 1.0 \
         (variance)\n[individual_parameters]\n  E0   = TVE0 * exp(ETA_E0)\n  EMAX = TVEMAX\n  \
         ET50 = TVET50\n[structural_model]\n  EFF = EMAX * TIME / (ET50 + TIME)\n  y   = E0 - \
         EFF\n[error_model]\n  DV ~ additive(ADD)\n",
    )
    .expect("parse");
    assert!(
        algebraic.is_algebraic(),
        "the fixture must be the model class the filter is about, or this tests nothing"
    );

    let mut p = pop(1);
    p.subjects[0].doses.clear();
    p.warnings.push(NO_DOSES.into());

    let suppressed = predict_diag(&algebraic, &p, &algebraic.default_params).unwrap();
    assert!(
        !has(&suppressed.warnings, "W_NO_DOSES"),
        "a compartment-free model has nothing to dose; telling it its data are malformed is \
         exactly what the shared filter exists to prevent: {:?}",
        suppressed.warnings
    );

    // The straddle: the same warning on a compartment model is a real finding and is reported.
    let compartmental = analytic_model();
    assert!(!compartmental.is_algebraic());
    let reported = predict_diag(&compartmental, &p, &compartmental.default_params).unwrap();
    assert!(
        has(&reported.warnings, "W_NO_DOSES"),
        "…while on a model that does have compartments it is a real finding: {:?}",
        reported.warnings
    );
}

// ── #1539: a diverged state is NaN, and the warning says so ──────────────────

/// The #1539 model: `central` observed and decaying with `k = CL/V = 0.1`; `X` decoupled,
/// unobserved, and running to its pole at t = 2 when `x_rate = 0.5` (`X = 1/(1 − 0.5t)`).
/// `x_rate = 0.0` is the control. `ode_method` is left at the default (`auto`) because that is
/// what the issue measured, and it is what makes the stiff-escalation clauses live.
fn divergent_x_model(x_rate: &str) -> CompiledModel {
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central, X])
[odes]
  init(X) = 1.0
  d/dt(central) = -(CL / V) * central
  d/dt(X) = {x_rate} * X * X
[error_model]
  DV ~ proportional(PROP)
"#
    ))
    .expect("parse")
}

/// One bolus of 100 into `central` at t = 12; observations at 1 / 20 / 40.
fn divergent_x_pop() -> Population {
    let mut p = pop(1);
    let s = &mut p.subjects[0];
    s.obs_times = vec![1.0, 20.0, 40.0];
    s.observations = vec![0.0, 45.0, 6.0];
    s.obs_cmts = vec![1; 3];
    s.cens = vec![0; 3];
    s.doses = vec![DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)];
    p
}

/// T5 (#1539): the issue repro through `predict_diag`.
///
/// Measured at `75f5aa83`: `central` = `[0.0, 99.99999999935972, 99.99999999935972]` — frozen
/// at the post-dose value, 16× the closed form at t = 40 — with a `W_ODE_SOLVER_DIAGNOSTICS`
/// that blamed stiffness and advised `rodas5p`. Now the rows past the pole are `NaN`, and the
/// message says why and advises nothing a solver setting could do.
///
/// The control straddles it on the same code path: `0·X` integrates, matches
/// `100·e^{−0.1(t−12)}` at the typical η = 0, and carries no solver warning at all.
///
/// Engine: `predict_diag` → `ode_predictions` → `integrate_dense_g`, `T = f64`. Mutations that
/// redden it: revert the tail pad (the `is_nan` legs); drop the diverged clause, or any one of
/// its sentences (the clause legs); keep the knob / `rodas5p` / freeze-pad wording when every
/// damaged segment diverged (the negative legs).
#[test]
fn predict_serves_nan_past_a_diverged_state_and_says_so() {
    let p = divergent_x_pop();
    let control_m = divergent_x_model("0.0");
    let control = predict_diag(&control_m, &p, &control_m.default_params).unwrap();
    let m = divergent_x_model("0.5");
    let out = predict_diag(&m, &p, &m.default_params).unwrap();
    let c: Vec<f64> = control.results.iter().map(|r| r.pred).collect();
    let d: Vec<f64> = out.results.iter().map(|r| r.pred).collect();
    assert_eq!(c.len(), 3);
    assert_eq!(d.len(), 3);

    assert_eq!(
        d[0].to_bits(),
        c[0].to_bits(),
        "pre-dose row: {d:?} vs {c:?}"
    );
    assert!(d[0].is_finite());
    for (i, t) in [(1, 20.0), (2, 40.0)] {
        let want = 100.0 * (-0.1f64 * (t - 12.0)).exp();
        assert!(
            c[i].is_finite() && (c[i] - want).abs() <= 1e-3 * want,
            "control {c:?}"
        );
        assert!(
            d[i].is_nan(),
            "t={t} must be NaN, not a frozen value: {d:?}"
        );
    }

    assert!(!has(&control.warnings, SOLVER), "{:?}", control.warnings);
    let msg = out
        .warnings
        .iter()
        .find(|w| w.contains(SOLVER))
        .unwrap_or_else(|| panic!("a solver warning: {:?}", out.warnings));
    for want in DIVERGED_CLAUSE_SENTENCES {
        assert!(msg.contains(want), "missing {want:?}: {msg}");
    }
    assert!(msg.contains("this predict() pass"), "{msg}");
    for unwanted in ADVICE_A_DIVERGENCE_MUST_NOT_GET {
        assert!(!msg.contains(unwanted), "must not say {unwanted:?}: {msg}");
    }
}

/// One fragment per sentence of the #1539 clause and its advice, so deleting any one of them
/// reddens every test that loops over this list.
pub(super) const DIVERGED_CLAUSE_SENTENCES: [&str; 8] = [
    "returned segment(s) had a state become non-finite",
    "the [odes] right-hand side diverged",
    "so every output time after that point is NaN rather than an integrated value",
    "The NaN covers every state of the segment, not only the one that diverged",
    "the states are integrated together, so none of them was integrated past that point",
    "The diverged segment(s) are not an ode_method or tolerance problem",
    "no stepper integrates past a state that has become non-finite",
    "check the [odes] right-hand side and the parameter values that drive it",
];

/// The solver advice that is false for a segment whose state went non-finite.
pub(super) const ADVICE_A_DIVERGENCE_MUST_NOT_GET: [&str; 5] = [
    "consider a different ode_method",
    "rodas5p",
    "the stiffness probe was right",
    "freeze-padded with the last state",
    "stability-limited",
];
