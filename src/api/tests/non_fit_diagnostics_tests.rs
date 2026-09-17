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
    let out = predict_diag(&m, &p, &m.default_params);
    assert!(
        has(&out.warnings, SS_II),
        "predict_diag must report the same model/data findings fit() does: {:?}",
        out.warnings
    );
    // The straddle, under the *old* behaviour as well as the new: the identical call on a
    // population whose dose is not SS reports nothing. Without this the assertion above passes
    // on an implementation that pushes a fixed string into every prediction.
    let clean = predict_diag(&m, &pop(1), &m.default_params);
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
    let rows = predict(&m, &p, &m.default_params);
    let out = predict_diag(&m, &p, &m.default_params);
    assert_eq!(rows.len(), out.results.len());
    assert!(!rows.is_empty(), "the fixture must produce rows to compare");
    for (a, b) in rows.iter().zip(out.results.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.time.to_bits(), b.time.to_bits());
        assert_eq!(
            a.pred.to_bits(),
            b.pred.to_bits(),
            "predict() and predict_diag() must return bit-identical predictions \
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
    let out = predict_diag(&m, &pop(2), &m.default_params);
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
    let clean = predict_diag(&clean_m, &pop(2), &clean_m.default_params);
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
    let one = predict_diag(&m, &pop(1), &m.default_params);
    let many = predict_diag(&m, &pop(16), &m.default_params);
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

/// One implementation, and the entry points prove it: on a fixture that trips **both** halves,
/// `predict_diag` and `simulate_with_options_diag` report the same findings.
///
/// Regression this catches: a per-entry-point filter — the fix #1280 explicitly rules out —
/// creeping back in. Mutation: drop any one code from either caller's list and the set
/// equality fails.
///
/// Compared on the `W_`/`E_` code tokens rather than on whole messages: `simulate()` also
/// carries per-subject simulation diagnostics that `predict()` has no analogue for, and the
/// solver clause's step counts legitimately differ (simulate integrates `n_sim` replicates of
/// each subject). What must not differ is *which findings* are reported.
#[test]
fn every_diagnostic_carrying_entry_point_reports_the_same_bundle() {
    let mut m = ode_model(1e5, 20);
    // Both halves live: an `SS=1, II=0` dose (model/data) on a budget-starved stiff ODE
    // (solver). Checked below rather than assumed — a fixture that trips only one half would
    // make this a test of the other one alone.
    m.default_params = m.default_params.clone();
    let p = pop_ss_bad_ii(2);
    let from_predict = codes(&predict_diag(&m, &p, &m.default_params).warnings);
    let from_simulate = codes(
        &simulate_with_options_diag(
            &m,
            &p,
            &m.default_params,
            1,
            &SimulateOptions {
                seed: Some(11),
                ..Default::default()
            },
        )
        .expect("sim")
        .warnings,
    );
    assert!(
        from_predict.contains(&SOLVER.to_string()),
        "the fixture must trip the solver half on both sides: {from_predict:?}"
    );
    assert!(
        from_predict.len() >= 2,
        "…and the model/data half too, or this compares one finding with itself: \
         {from_predict:?}"
    );
    assert_eq!(
        from_predict, from_simulate,
        "predict and simulate must report the same findings; a per-entry-point filter is \
         exactly what #1280 rules out"
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
/// these three starting to print to stderr instead of returning. `simulate_with_uncertainty`
/// is the one that has findings and no channel to put them in; `predict` and
/// `simulate_with_options` have a `_diag` twin one call away.
#[test]
fn the_row_only_entry_points_stay_silent_by_contract() {
    let m = analytic_model();
    let p = pop_ss_bad_ii(1);
    // `predict` → rows only, and they are the `_diag` form's rows.
    assert_eq!(
        predict(&m, &p, &m.default_params).len(),
        predict_diag(&m, &p, &m.default_params).results.len()
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
    let out = predict_diag(&m, &pop(1), &m.default_params);
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
    let with_dv = predict_diag(&m, &pop(1), &m.default_params);
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
