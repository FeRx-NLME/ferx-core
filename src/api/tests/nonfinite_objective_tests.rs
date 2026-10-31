//! Tier-1 tests for #1303: `converged` may not be `true` at an objective that is
//! not a reportable population OFV.
//!
//! **The regression.** Measured at `fcfe8b10`, on the two-subject FOCEI fit one of
//! whose subjects carries a `NaN` dose time (the #1296 fixture, reused here):
//!
//! ```text
//! ofv       = NaN
//! converged = true
//! ```
//!
//! There was no production gate anywhere in `src/api` or `src/estimation` tying the
//! boolean to a finite objective, so every consumer that keys on it — the R
//! wrapper, `ferx-tools`' model-space search via `Strictness::require_converged`,
//! an agent reading the fit YAML — was told the run succeeded. #1296 had already
//! added a warning naming the abandoned walks; a warning is easier to ignore than
//! a flag is to misread.
//!
//! **Why `is_finite()` is not the gate.** The inner objective clamps a blown-up
//! value to a finite `~1e20` sentinel and the outer one doubles it, so a *repelled*
//! fit comes back finite and an `is_finite()` test waves it through. The shared
//! predicate is `ofv_is_valid` (`is_finite() && < DIVERGENCE_OFV`), and G2 below
//! pins that the sentinel half is load-bearing.
use super::*;
use crate::estimation::outer_optimizer::{
    gate_converged_on_objective, nonfinite_objective_reason, publishes_no_objective,
    DIVERGENCE_OFV, NONFINITE_OBJECTIVE_TOKEN,
};
use crate::parser::model_parser::parse_model_string;
use crate::types::{BloqMethod, ViFinalOfv, WarningCode, WarningSeverity};

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
//  G1 — the reported regression, end to end
// ─────────────────────────────────────────────────────────────────────────────

/// A one-compartment closed form — no ODE solve, so a chain test that has to run
/// two real estimators stays inside the Tier-1 time budget.
fn one_cpt_model() -> CompiledModel {
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

/// The #1296 model: a rapid-equilibrium two-state ODE, benign at `KFAST = 1`.
///
/// `pub(super)` so #1617's sibling test module can reuse it rather than copy it: the same
/// population is the only end-to-end input that makes *every* objective non-finite, which
/// is the entry condition for `no_feasible_eval_warning`'s non-finite arm. One fixture, two
/// properties — see `api::no_feasible_eval_tests::an_all_non_finite_run_reports_the_non_finite_arm_end_to_end`.
pub(super) fn two_state_model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  theta KFAST(1.0, 1e-6, 1e6)
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
"#,
    )
    .expect("parse")
}

/// `pub(super)` for the same reason as [`two_state_model`].
pub(super) fn subject_with_dose_time(id: &str, dose_time: f64, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| scale * 50.0 / (1.0 + t)).collect();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(dose_time, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

fn one_iteration_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 1,
        run_covariance_step: false,
        threads: Some(1),
        ..Default::default()
    }
}

/// **G1 — the regression itself.** `fit()` over a population one of whose subjects
/// has an unorderable timeline must not report `converged: true`.
///
/// The fixture is #1296's, unchanged, and the premise it asserts is the reason this
/// is reachable at all: nothing upstream rejects the population. `check_model_data`
/// checks dose *attributes* (`ALAG`/`F`/`D`/`R`, #1286), not record *times*, so the
/// subject reaches the engine, the `timeline_has_non_finite` guard `NaN`-fills it,
/// and the objective is `NaN` for the whole population — including the healthy
/// subject, which is #1235's "one bad subject in ten destroys the objective for
/// everyone".
///
/// Three assertions, and each is separately load-bearing:
///
/// 1. **`ofv` is still `NaN`.** Non-degeneracy: if a later change made this
///    population produce a finite objective, the test would pass for a reason that
///    has nothing to do with the gate. The fixture has to keep delivering the
///    defect's input.
/// 2. **`converged` is `false`.** The regression.
/// 3. **A warning says why**, carries the `W_NONFINITE_OBJECTIVE` token, and
///    classifies `Critical`. The issue asks for the flag *and* the reason.
///
/// Mutation (run): delete the `gate_converged_on_objective` call in
/// `outer_optimizer`'s NLopt exit **and** the `nonfinite_objective_warning` call in
/// `fit_inner` → `converged` is `true` again and assertions 2 and 3 fire. Deleting
/// only the `fit_inner` call leaves this green (the estimator gate already demotes
/// on this path) — which is why G5 exists: it is the case only the `fit_inner` call
/// can catch, and it is a different number, not a second copy of this one.
#[test]
fn a_fit_whose_objective_is_nan_is_not_reported_converged() {
    let model = two_state_model();
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("bad", f64::NAN, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assert!(
        crate::api::validation::check_model_data(&model, &pop).is_empty(),
        "the fixture must reach the engines — if a data check now rejects it, this test is \
         pinning the wrong path and the NaN never happens"
    );

    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    // (1) The fixture still produces the defect's input.
    assert!(
        result.ofv.is_nan(),
        "this fixture exists to produce a NaN objective; it produced {}. The gate below is \
         then being tested against the wrong input.",
        result.ofv
    );
    // (2) The regression.
    assert!(
        !result.converged,
        "#1303: fit() reported converged = true at ofv = {}. Every quantity derived from the \
         objective is meaningless here, and `converged` is the field a consumer keys on.",
        result.ofv
    );
    // (3) ...and it says why.
    let entry = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains(NONFINITE_OBJECTIVE_TOKEN))
        .unwrap_or_else(|| {
            panic!(
                "the demotion must carry a typed warning naming the objective; got {:?}",
                result
                    .warnings_structured
                    .iter()
                    .map(|w| &w.message)
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        entry.severity,
        WarningSeverity::Critical,
        "a meaningless objective invalidates the OFV, AIC, BIC and every SE — not an \
         advisory: {entry:?}"
    );
    assert_eq!(entry.category, WarningCode::Convergence, "{entry:?}");
    assert!(
        entry.message.contains("NaN"),
        "the message must name *which* way the objective failed, not just that it did: {}",
        entry.message
    );
    // Exactly one, not "at least one". The estimator's gate and fit()'s gate both
    // fire on this run — deliberately, since neither may stay silent because the
    // other spoke (G3b) — so `fit_inner` drops the estimator's plain-string copy
    // and keeps the typed one. Without that explicit de-duplication a user reads
    // the same paragraph twice.
    assert_eq!(
        result
            .warnings
            .iter()
            .filter(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN))
            .count(),
        1,
        "the plain-text warnings must carry it exactly once — that is what the CLI and the \
         fit YAML print: {:?}",
        result.warnings
    );
    assert_eq!(
        result
            .warnings_structured
            .iter()
            .filter(|w| w.message.contains(NONFINITE_OBJECTIVE_TOKEN))
            .count(),
        1,
        "...and so must the structured ones: {:?}",
        result.warnings_structured
    );
}

/// **G10 — the documented `details` payload is present on the *common* path.**
/// (Review of #1303, finding 2.)
///
/// `docs/warnings.qmd` documents `ofv` / `reason` / `divergence_cutoff` /
/// `method` for this code, and a programmatic consumer is told to read the
/// numbers there rather than parse prose. Before the review that payload existed
/// only for the rare case first discovered at the fit-level total (G5's prior
/// overflow): on every ordinary run the estimator had already demoted the
/// boolean, the fit-level helper returned early, no native `WarningEntry` was
/// built, and `rebuild_warnings_structured` fell back to string classification —
/// which has no `Convergence` enrichment arm in `diagnostic_details`, so
/// `details` came back `None`. The docs and the code disagreed exactly on the
/// path a user actually hits.
///
/// Same fixture as G1, asserting the half G1 does not.
///
/// Mutation (run): restore the `!*converged` early return in
/// `gate_converged_on_objective` → `details` is `None` and this fires. Drop the
/// `warnings.retain(...)` de-duplication in `fit_inner` → G1's exactly-one
/// assertions fire instead.
#[test]
fn the_demotion_carries_its_documented_details_payload() {
    let model = two_state_model();
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("bad", f64::NAN, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");
    assert!(result.ofv.is_nan(), "fixture premise: {}", result.ofv);

    let entry = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains(NONFINITE_OBJECTIVE_TOKEN))
        .expect("the demotion must be a typed entry");
    let details = entry.details.as_ref().unwrap_or_else(|| {
        panic!(
            "docs/warnings.qmd documents a details payload for this code; the entry carries \
             none, so the structured warning was string-classified rather than emitted typed \
             at source: {entry:?}"
        )
    });
    // JSON has no non-finite numbers, so the value is carried as a string here.
    assert_eq!(details["ofv"], "NaN", "{details}");
    assert_eq!(details["reason"], "NaN", "{details}");
    assert_eq!(
        details["divergence_cutoff"].as_f64(),
        Some(DIVERGENCE_OFV),
        "the consumer needs the cutoff to tell a sentinel from a real objective: {details}"
    );
    assert_eq!(
        details["method"],
        EstimationMethod::FoceI.label(),
        "the payload names the stage that produced the objective: {details}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  G2/G3 — the gate itself
// ─────────────────────────────────────────────────────────────────────────────

/// **G2 — the sentinel trap.** `is_finite()` alone is not the gate, and this is the
/// case that proves it: the inner objective clamps a blown-up individual
/// contribution to `~1e20` and the outer objective doubles it, so a repelled fit
/// reports a perfectly finite `2e20`. #1295 measured exactly that — the same
/// population, one line changed, `ofv` moving from `NaN` to `2e20`, which is why
/// #1303 says the `NaN` half alone would have been an accident of which sentinel
/// the objective happened to use.
///
/// The reason distinguishes the sentinel from an objective that is merely past the
/// cutoff (#2038). A proportional error on a prediction of 0 at the `1e-12` variance
/// floor scores `DV² × 1e12` per observation, finite: the `4.6e14` here was measured
/// on N6's fixture before #2039. That is not the clamp, and calling it "at the
/// divergence sentinel" sent the reader looking for a clamped subject that did not
/// exist. Both sides of `SUBJECT_SENTINEL_NLL` are listed, so moving the boundary
/// either way fires.
///
/// Mutation (run): weaken `ofv_is_valid` to `ofv.is_finite()` → the `2e20`,
/// `DIVERGENCE_OFV` and `4.6e14` cases fire. Compare against `DIVERGENCE_OFV`
/// instead of `SUBJECT_SENTINEL_NLL` in `nonfinite_objective_reason` → the two "too
/// large" cases fire; drop the arm → they fire too.
#[test]
fn the_gate_rejects_the_finite_sentinel_not_only_nan() {
    // The trap, stated: the sentinel is finite.
    assert!(
        2e20_f64.is_finite(),
        "if this ever stops holding the rest of this test is measuring nothing"
    );

    for (ofv, what) in [
        (f64::NAN, "NaN"),
        (f64::INFINITY, "infinite"),
        (f64::NEG_INFINITY, "infinite"),
        (2e20, "at the divergence sentinel"),
        (1e20, "at the divergence sentinel"),
        (4.619_219_710_896_776e14, "too large to be a fit"),
        (DIVERGENCE_OFV, "too large to be a fit"),
    ] {
        let mut converged = true;
        let msg = gate_converged_on_objective(&mut converged, ofv)
            .unwrap_or_else(|| panic!("ofv = {ofv:?} must demote the verdict"));
        assert!(!converged, "ofv = {ofv:?} must demote the verdict");
        assert!(
            msg.contains(NONFINITE_OBJECTIVE_TOKEN),
            "every demotion carries the token the classifier keys on: {msg}"
        );
        assert!(
            msg.contains(what),
            "the message must name which way it failed; ofv = {ofv:?} is `{what}`: {msg}"
        );
        for other in [
            "NaN",
            "infinite",
            "at the divergence sentinel",
            "too large to be a fit",
        ] {
            assert!(
                other == what || !msg.contains(other),
                "ofv = {ofv:?} is `{what}`, and the message also says `{other}`: {msg}"
            );
        }
        assert_eq!(nonfinite_objective_reason(ofv), what, "ofv = {ofv:?}");
    }
}

/// **G3 — the gate must not fire on a real objective.** A gate that demoted
/// everything would pass G1 and G2 and be worse than the defect.
///
/// Both signs, and the extremes of the valid range. The negative extreme is
/// deliberate: the cutoff is **one-sided**, because a large negative −2 log L is
/// legitimate (log-transformed DV on a large cohort), and
/// `ofv_is_valid_rejects_the_clamped_sentinel_not_just_non_finite` pins that
/// independently.
#[test]
fn the_gate_leaves_a_real_objective_alone() {
    for ofv in [-1e15, -286.0, 0.0, 12_345.678, DIVERGENCE_OFV - 1.0] {
        let mut converged = true;
        assert!(
            gate_converged_on_objective(&mut converged, ofv).is_none(),
            "ofv = {ofv:?} is a legitimate population objective and must not be demoted"
        );
        assert!(converged, "ofv = {ofv:?}");
        // ...and it does not *promote* either: a run that already failed stays failed.
        let mut already_failed = false;
        assert!(gate_converged_on_objective(&mut already_failed, ofv).is_none());
        assert!(!already_failed, "ofv = {ofv:?}");
    }
}

/// **G3b — a run that already failed for another reason still learns its
/// objective is unusable.** (Review of #1303, finding 2.)
///
/// The gate is a predicate on the **objective**, not on the verdict handed to it.
/// The first version returned early on `!*converged`, which looked like sensible
/// de-duplication and was not: "the optimizer hit its evaluation budget" and "the
/// objective is not a number" are different facts with different consequences —
/// provisional estimates in the first case, nothing usable at all in the second —
/// and whichever fired first silenced the other.
///
/// It also had a second, quieter effect: because every estimator demotes the
/// boolean before `fit()`'s assembly sees it, the fit-level helper's early return
/// meant the typed `details` payload was **never built on the common path**, so
/// the payload `docs/warnings.qmd` documents existed only for the rare
/// prior-overflow case. G10 is that half.
///
/// Mutation (run): restore `!*converged ||` to either gate's early return → this
/// fires on the arm whose gate was changed, and G10 fires too.
#[test]
fn a_run_that_already_failed_still_reports_an_unusable_objective() {
    let mut converged = false;
    let msg = gate_converged_on_objective(&mut converged, f64::NAN).expect(
        "an optimizer that stopped for another reason must still be told its objective is NaN",
    );
    assert!(!converged);
    assert!(msg.contains(NONFINITE_OBJECTIVE_TOKEN), "{msg}");

    // The MCEM composite has the same shape, on its own rule (#528's cap).
    let mut converged = false;
    let msg = crate::estimation::impmap::gate_converged_on_mcem_objective(&mut converged, 1e35)
        .expect("a #528 runaway must be reported whatever else already failed");
    assert!(!converged);
    assert!(msg.contains(NONFINITE_OBJECTIVE_TOKEN), "{msg}");
}

// ─────────────────────────────────────────────────────────────────────────────
//  G4 — the one exemption
// ─────────────────────────────────────────────────────────────────────────────

/// **G4 — VI's deliberate `NaN`, both directions.**
///
/// `method = vi` with the default `vi_final_ofv = none` publishes `ofv = NaN` *on
/// purpose*: the ELBO is a lower bound on the log likelihood, not a −2 log L, and
/// `ViFinalOfv::None`'s argument is that no number is safer than a number that
/// looks like an OFV and is not one. A blanket "demote whenever the objective is
/// non-finite" would therefore report `converged: false` for **every default VI
/// fit** — a false alarm on the common path, and one `ferx-tools`'
/// `require_converged` would turn into a rejected search candidate.
///
/// Both directions are asserted, because either alone is satisfied by a wrong
/// implementation: an exemption keyed on the *method* would also exempt
/// `vi_final_ofv = laplace`, which publishes a real `2·pop_nll` and must be gated
/// like everything else. The exemption is a *setting*, not an estimator.
///
/// Mutation (run): drop the `vi_final_ofv` half of `publishes_no_objective` → the
/// `laplace` case stops demoting and the second half fires. Drop the whole
/// exemption → the `none` case demotes and the first half fires.
#[test]
fn vi_publishing_no_objective_is_exempt_but_vi_publishing_one_is_not() {
    let default_vi = FitOptions {
        method: EstimationMethod::Vi,
        vi_final_ofv: ViFinalOfv::None,
        ..Default::default()
    };
    let laplace_vi = FitOptions {
        vi_final_ofv: ViFinalOfv::Laplace,
        ..default_vi.clone()
    };

    assert!(publishes_no_objective(EstimationMethod::Vi, &default_vi));
    assert!(!publishes_no_objective(EstimationMethod::Vi, &laplace_vi));
    // Not a property of the method on its own, and not one any other method has.
    assert!(!publishes_no_objective(
        EstimationMethod::FoceI,
        &default_vi
    ));

    let mut converged = true;
    assert!(
        crate::api::nonfinite_objective_warning(
            &mut converged,
            f64::NAN,
            EstimationMethod::Vi,
            &default_vi
        )
        .is_none(),
        "the default VI NaN is a declaration that no objective was published, not a failed one"
    );
    assert!(
        converged,
        "every default VI fit would otherwise report converged: false"
    );

    let mut converged = true;
    let (msg, entry) = crate::api::nonfinite_objective_warning(
        &mut converged,
        f64::NAN,
        EstimationMethod::Vi,
        &laplace_vi,
    )
    .expect("`vi_final_ofv = laplace` publishes a real 2*pop_nll and is gated like any other");
    assert!(!converged);
    assert!(msg.contains(NONFINITE_OBJECTIVE_TOKEN));
    assert_eq!(entry.severity, WarningSeverity::Critical);
}

/// **G11 — the exemption survives a trailing evaluation-only IMP stage.**
/// (Review of #1303, finding 1 — a `converged: true` fit demoted to `false`.)
///
/// `method = [vi, imp]` with `imp_eval_only = true` is the **documented** way to
/// finish a VI fit: `ViFinalOfv`'s own doc points at it, because it consumes VI's
/// parameters, EBEs and Hessians and reports a genuine importance-sampled
/// −2 log L alongside the IS diagnostics.
///
/// The trap is where that number lands. An `imp_eval_only` stage deliberately
/// **does not touch `result`** — it writes `FitResult::importance_sampling` and
/// `continue`s, so the preceding estimator's objective stays in `FitResult::ofv`.
/// For this chain that objective is VI's deliberate `NaN`. Keying the exemption
/// on `chain.last()` therefore read it as *IMP's* failed objective and demoted a
/// converged VI fit — the exact configuration the exemption exists to protect.
///
/// So the gate is keyed on the tracked **owner** of `result.ofv`, and the premise
/// this test asserts first is the fact that makes the two different: the IS
/// likelihood is finite while `ofv` is still `NaN`. Without that premise the test
/// would pass on a chain where IMP never ran.
///
/// The `agq_eval_only` twin is the mirror image and is why the owner is tracked
/// rather than switched back to `final_method`: an AGQ readout *does* replace
/// `prev.ofv`, so there the eval-only stage is the owner. G12 pins that direction.
///
/// Mutation (run): `objective_stage.unwrap_or(final_method)` →
/// `*chain.last().unwrap()` in `fit_inner` → this fires; every other test in this
/// file stays green, which is why the reviewer found it and the suite did not.
#[test]
fn a_trailing_eval_only_imp_stage_does_not_claim_vis_deliberate_nan() {
    let model = one_cpt_model();
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
    let opts = FitOptions {
        methods: vec![EstimationMethod::Vi, EstimationMethod::Imp],
        imp_eval_only: true,
        vi_final_ofv: ViFinalOfv::None,
        vi_iters: 40,
        imp_samples: 20,
        run_covariance_step: false,
        threads: Some(1),
        ..Default::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");

    // Premise 1: the published objective really is VI's deliberate NaN — the
    // eval-only IMP stage did not replace it.
    assert!(
        result.ofv.is_nan(),
        "premise: under `vi_final_ofv = none` the published ofv is VI's deliberate NaN; got {}",
        result.ofv
    );
    // Premise 2: IMP really ran and produced a real objective, which went
    // somewhere else. Without this the chain under test never happened.
    let is = result
        .importance_sampling
        .as_ref()
        .expect("premise: the eval-only IMP stage must have produced an IS result");
    assert!(
        is.minus2_log_likelihood.is_finite(),
        "premise: IMP's own objective is finite and is reported here, not in `ofv`: {}",
        is.minus2_log_likelihood
    );

    // The regression: VI's declaration that it published no objective must not be
    // read as IMP's failure to produce one.
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
        "the deliberate VI NaN was reported as a non-finite-objective failure: {:?}",
        result.warnings
    );
    assert_eq!(
        result.converged,
        fit(
            &model,
            &pop,
            &model.default_params,
            &FitOptions {
                methods: vec![EstimationMethod::Vi],
                imp_eval_only: false,
                ..opts.clone()
            }
        )
        .expect("solo VI fit")
        .converged,
        "appending a successful IMP readout must not change VI's convergence verdict — that \
         is the whole point of an evaluation-only stage"
    );
}

/// **G12 — the mirror image: an AGQ readout *is* the owner.**
///
/// `agq_eval_only` is evaluation-only like `imp_eval_only`, and the two differ in
/// exactly the thing the gate needs: an AGQ readout **replaces `prev.ofv`** with
/// its quadrature marginal, so the eval-only stage owns the published objective
/// there, while an IMP readout leaves the estimator's in place.
///
/// That asymmetry is why the owner is *tracked* rather than derived from the
/// chain — `final_method` (which fixes G11) is wrong here, and `chain.last()`
/// (which is right here) is what broke G11. This test is the half that stops the
/// G11 fix from being "switch to `final_method`".
///
/// Mutation (run): `objective_stage.unwrap_or(final_method)` → `final_method` in
/// `fit_inner`, i.e. ignore the AGQ arm's ownership → the objective is attributed
/// to VI, `publishes_no_objective` exempts it, and this fires.
#[test]
fn an_agq_readout_owns_the_objective_it_writes() {
    let model = two_state_model();
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("bad", f64::NAN, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let opts = FitOptions {
        methods: vec![EstimationMethod::Vi, EstimationMethod::Laplace],
        agq_eval_only: true,
        n_agq: 3,
        vi_final_ofv: ViFinalOfv::None,
        vi_iters: 40,
        run_covariance_step: false,
        threads: Some(1),
        ..Default::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");

    // The AGQ readout replaced `ofv` with its own quadrature marginal, which the
    // NaN-timeline subject poisons. That is a real failed objective, not VI's
    // deliberate absence of one, so the VI exemption must not reach it.
    //
    // Measured: the quadrature comes back at the **sentinel** (`2e20`), not `NaN`
    // — `agq_population_nll` sums per-subject contributions that the inner
    // objective has already clamped. So this arm also exercises the half of the
    // gate `is_finite()` would miss, end to end, which G2 only pins as a unit.
    assert!(
        !result.ofv.is_nan() && result.ofv >= DIVERGENCE_OFV,
        "fixture premise: the AGQ readout must publish a clamped (finite, sentinel-sized) \
         objective here; got {}. If this became a real number the fixture stopped producing \
         the defect's input.",
        result.ofv
    );
    assert!(
        !result.converged,
        "the AGQ readout owns the published objective, so its NaN must demote: ofv = {}",
        result.ofv
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
        "{:?}",
        result.warnings
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  G5 — the case only the fit()-level gate can see
// ─────────────────────────────────────────────────────────────────────────────

/// **G5 — a finite likelihood plus a non-finite prior penalty.**
///
/// This is what stops the `fit_inner` call from being a redundant copy of the
/// estimators' gate, and the difference is a *number*, not a code path: every
/// estimator gates `OuterResult::ofv`, which is the **clean** −2 log L (the prior
/// penalty is excluded from it by construction, so AIC/BIC stay information
/// criteria). `FitResult::ofv` is `ofv_data + ofv_prior`, and `ofv_prior` is
/// evaluated in `fit_inner` *after* the last optimizer has returned. So the
/// published objective is a quantity no estimator ever saw, and only the
/// `fit_inner` gate can check it.
///
/// The fixture makes the penalty overflow rather than merely be large. The prior
/// sits on `VSHIFT`, whose **negative lower bound** makes it pack on the identity
/// scale (`theta_packs_log(-1.0) == false`), and that is what the fixture needs:
/// on a log-packed coordinate `sd` is first converted to a relative spread and
/// `sqrt(ln(1 + rse²))` underflows to `0` — which `PriorSet::build` rejects
/// outright — so no log-scale prior can produce an infinite penalty. On the
/// identity scale the declared `sd` *is* the packed SD. With the prior centred at
/// `1e6` and `VSHIFT` boxed into `[-1, 1]`, `z = (x − 1e6)/1e-200 ≈ −1e206` and
/// `z² = +inf`, while the likelihood at those (perfectly ordinary) estimates stays
/// finite — `VSHIFT` only shifts `V` by at most 1 out of 10.
///
/// Both halves are asserted, and the first is what makes the second mean anything:
/// if `ofv_data` were also non-finite the estimators' gate would already have
/// demoted and this would be G1 again.
///
/// Mutation (run): delete the `nonfinite_objective_warning` call in `fit_inner` →
/// `converged` is `true` at `ofv = inf` and this fires, with G1 and G7 staying green.
#[test]
fn a_non_finite_prior_penalty_demotes_a_fit_whose_likelihood_is_finite() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  theta VSHIFT(0.0, -1.0, 1.0) prior(1e6, sd = 1e-200)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV + VSHIFT
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#;
    let model = parse_model_string(src).expect("parse");
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("1", 0.0, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let result = fit(&model, &pop, &model.default_params, &one_iteration_opts()).expect("fit");

    // The premise: the likelihood half is fine. Without this the test is G1.
    assert!(
        result.ofv_data.is_finite(),
        "this fixture must isolate the *penalty* as the non-finite half; ofv_data = {}",
        result.ofv_data
    );
    assert!(
        !result.ofv_prior.is_finite(),
        "the prior penalty must overflow for this test to reach the fit()-level gate; \
         ofv_prior = {}",
        result.ofv_prior
    );
    assert!(
        !result.ofv.is_finite(),
        "the published objective is ofv_data + ofv_prior: {}",
        result.ofv
    );
    assert!(
        !result.converged,
        "#1303: the objective fit() publishes is {}, so the run did not converge on a \
         solution of the problem posed — whatever the optimizer reported about the \
         unpenalized likelihood it was handed",
        result.ofv
    );
    // #2267 review r1: every subject's contribution is usable here, so the warning
    // must say the prior is what failed rather than end without naming anything.
    let warning = result
        .warnings
        .iter()
        .find(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN))
        .unwrap_or_else(|| panic!("{:?}", result.warnings));
    assert!(
        warning.ends_with(
            " No subject can be named: every subject's post-hoc FOCEI contribution at \
             these estimates is usable, and the parameter-prior penalty, a population-level \
             term no subject carries, is not (inf)."
        ),
        "{warning}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  G6 — the warning classifies deterministically
// ─────────────────────────────────────────────────────────────────────────────

/// **G6 — the token, not the prose, decides the category.**
///
/// A warning re-classified from flat text (a multi-start splice, a checkpoint
/// restore, `rebuild_warnings_structured`) has to land back on the same
/// `(Critical, Convergence)` verdict the emitter used. The message quotes the
/// offending value and lists candidate causes — "infusion duration", "residual
/// variance", "covariate model" — any of which a later edit could grow into a
/// phrase one of the prose arms below it claims, so the arm matches the `W_` token
/// and sits ahead of every prose arm.
///
/// Mutation (run): delete the `w_nonfinite_objective` arm → the message still
/// happens to contain "did not converge" today and stays green, which is the whole
/// argument for the token arm; change the message's wording as well and it fires.
/// Asserted here on a *reworded* copy carrying only the token, so the test does not
/// depend on that coincidence.
#[test]
fn the_demotion_warning_classifies_as_a_critical_convergence_failure() {
    let real = {
        let mut converged = true;
        gate_converged_on_objective(&mut converged, f64::NAN).expect("a demotion")
    };
    let entry = crate::types::classify_warning(&real);
    assert_eq!(entry.severity, WarningSeverity::Critical, "{entry:?}");
    assert_eq!(entry.category, WarningCode::Convergence, "{entry:?}");

    // The token alone must be enough — no prose from the current wording.
    let token_only = format!("{NONFINITE_OBJECTIVE_TOKEN}: something happened.");
    let entry = crate::types::classify_warning(&token_only);
    assert_eq!(entry.severity, WarningSeverity::Critical, "{entry:?}");
    assert_eq!(entry.category, WarningCode::Convergence, "{entry:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
//  G7 — the estimator's own published verdict
// ─────────────────────────────────────────────────────────────────────────────

/// **G7 — `OuterResult.converged`, not just `FitResult.converged`.**
///
/// `optimize_population` is **public API** and `ferx-tools` calls it (and
/// `run_foce_gn`, `run_imp`, `run_impmap`, `run_bayes`) directly, reading
/// `.converged` off the returned `OuterResult` without going anywhere near
/// `fit_inner`. So the `fit()`-level gate is not enough on its own: a caller on
/// that path would still be told a `NaN`-objective run succeeded.
///
/// This is what makes the estimator-side gate separately load-bearing
/// *behaviourally* rather than only through G9's source scan.
///
/// Mutation (run): delete the `gate_converged_on_objective` call at
/// `outer_optimizer`'s NLopt exit → this fires, while G1 stays green because
/// `fit_inner`'s own gate still covers the `FitResult`. That asymmetry is the
/// point: they are two surfaces, not one gate written twice.
#[test]
fn the_optimizer_publishes_its_own_gated_verdict_not_only_fit() {
    let model = two_state_model();
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("bad", f64::NAN, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let out = crate::estimation::outer_optimizer::optimize_population(
        &model,
        &pop,
        &model.default_params,
        &one_iteration_opts(),
    );
    assert!(
        !out.ofv.is_finite(),
        "the fixture must still hand the optimizer a poisoned objective; got {}",
        out.ofv
    );
    assert!(
        !out.converged,
        "#1303: a public `OuterResult` reported converged = true at ofv = {} — this is the \
         value `ferx-tools` reads when it drives an optimizer directly",
        out.ofv
    );
    assert!(
        out.warnings
            .iter()
            .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
        "the reason travels on the OuterResult too, not only on the FitResult: {:?}",
        out.warnings
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  G9 — the pairing is structural
// ─────────────────────────────────────────────────────────────────────────────

/// **G9 — every place that publishes a `(converged, ofv)` pair gates it.**
///
/// Nothing in the type system stops a thirteenth `OuterResult` literal from being
/// written with an ungated `converged`, and per-site tests cannot close that: they
/// can only cover sites that already exist. The issue asks for the rule to be
/// enforced *wherever `converged` is set*, not patched at one call site, and this
/// is what makes that claim checkable.
///
/// The reason it has to be a scan rather than a constructor is that `OuterResult`
/// is **public API with public fields** (`ferx-tools` calls `run_foce_gn`,
/// `run_imp`, `run_impmap`, `run_bayes` and `optimize_population` directly and
/// reads `.converged` without going near `fit_inner`), so privatising the field to
/// force a constructor is a breaking change belonging to its own PR.
///
/// Counted per file, exact both ways — a removed gate fails as loudly as an
/// ungated new site:
///
/// * a **publishing site** is an `OuterResult { … }` or `FitResult { … }` literal
///   whose `converged` field is not the literal `false` (which is always safe);
/// * a **gate call** is `gate_converged_on_objective`, `gate_converged_on_mcem_objective`
///   or `nonfinite_objective_warning`, excluding each one's own definition.
///
/// Test modules are excluded: `_tests.rs` siblings and inline `mod …tests {` blocks
/// build `FitResult`s as scaffolding, and scaffolding has no objective to gate.
///
/// Mutation (run): delete any one gate call → that file's pair disagrees and this
/// fires naming it. Add an `OuterResult` literal with `converged: some_bool` in a
/// new file → the file appears with `gates: 0` and this fires naming it.
#[test]
fn every_published_convergence_verdict_is_gated_on_its_objective() {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("readable directory") {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources(&root.join("src"), &mut files);
    files.sort();
    assert!(
        files.len() > 100,
        "the scan found only {} files under src/ — it is measuring the walk, not the code",
        files.len()
    );

    let gate_names = [
        "gate_converged_on_objective",
        "gate_converged_on_mcem_objective",
        "nonfinite_objective_warning",
    ];
    // file -> (publishing sites, gate calls)
    let mut found: BTreeMap<String, (usize, usize)> = BTreeMap::new();

    for file in &files {
        let rel = file
            .strip_prefix(&root)
            .expect("under the manifest dir")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.ends_with("_tests.rs") || rel.contains("/tests/") || rel.contains("test_helpers") {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("readable source");
        // Drop the inline test module, which builds `FitResult`s as scaffolding.
        let lines: Vec<&str> = text.lines().collect();
        let end = lines
            .iter()
            .position(|l| l.starts_with("mod ") && l.contains("test") && l.ends_with('{'))
            .unwrap_or(lines.len());
        let lines = &lines[..end];

        let mut publishing = 0usize;
        let mut gates = 0usize;
        for (i, line) in lines.iter().enumerate() {
            // Comments are where these names legitimately appear everywhere.
            let code = line.split("//").next().unwrap_or("");
            for name in gate_names {
                if code.contains(&format!("{name}(")) && !code.contains(&format!("fn {name}(")) {
                    gates += 1;
                }
            }
            if !(code.contains("OuterResult {") || code.contains("FitResult {")) {
                continue;
            }
            // The struct-definition lines (`pub struct OuterResult {`) are not literals.
            if code.contains("struct ") {
                continue;
            }
            // The `converged` field of this literal: the first one at or after the
            // opening brace. A literal that has none is a partial / functional-update
            // form and inherits whatever it is based on.
            let field = lines[i..]
                .iter()
                .take(80)
                .find_map(|l| l.trim().strip_prefix("converged"))
                .map(|rest| rest.trim_start_matches([':', ' ']).trim_end_matches(','));
            match field {
                // `converged: false` can never over-claim.
                Some("false") => {}
                Some(_) => publishing += 1,
                None => {}
            }
        }
        if publishing > 0 || gates > 0 {
            found.insert(rel, (publishing, gates));
        }
    }

    let expected: BTreeMap<String, (usize, usize)> = [
        // Each estimator gates the `OuterResult` it returns.
        ("src/estimation/bayes.rs", (1, 1)),
        ("src/estimation/gauss_newton.rs", (2, 2)),
        // Two MCEM exits; `gate_converged_on_mcem_objective` is defined here and
        // calls the shared gate once inside its own body, hence three calls.
        ("src/estimation/impmap.rs", (2, 3)),
        ("src/estimation/outer_optimizer.rs", (2, 2)),
        // The profiled outer loop (#1065); its never-solved exit publishes
        // `converged: false`.
        ("src/estimation/profile.rs", (1, 1)),
        ("src/estimation/saem.rs", (1, 1)),
        ("src/estimation/trust_region.rs", (1, 1)),
        ("src/estimation/vi/run.rs", (1, 1)),
        // Two synthetic evaluation-only `OuterResult`s (the AGQ readout and
        // standalone `imp_eval_only`) plus the one `FitResult` fit() publishes.
        ("src/api/fit.rs", (3, 3)),
        // `nonfinite_objective_warning` is defined here and is the fit()-level gate.
        ("src/api/postfit.rs", (0, 1)),
        // Checkpoint restore: `converged: w.converged` copies a verdict a previous
        // fit already gated on its own objective, together with the `ofv` that
        // verdict was about. Re-gating here would be a second rule applied to a
        // number this code did not compute, and would silently rewrite a bundle's
        // recorded history; `fitrx` round-trip tests pin that a restored result
        // equals what was written.
        ("src/io/fitrx.rs", (1, 0)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "a `(converged, ofv)` pair is published somewhere that does not gate it, or a gate \
         was removed. #1303: `converged` is the field a consumer keys on, and at a NaN / \
         infinite / sentinel objective it must be false — enforced at every publishing site, \
         not at one call site, because `OuterResult` is public API that `ferx-tools` reads \
         without going through fit(). If a new site genuinely publishes no objective (VI \
         under `vi_final_ofv = none` is the only one today, and `publishes_no_objective` \
         owns that), route it through the same gate and add it here with the reason."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  N — the warning names the subjects behind it (#2038)
// ─────────────────────────────────────────────────────────────────────────────

/// The `W_NONFINITE_OBJECTIVE` pair `fit()` builds for a sentinel objective, so the
/// helper is handed the message and entry it is handed in production.
fn sentinel_warning() -> (String, crate::types::WarningEntry) {
    let mut converged = true;
    crate::api::nonfinite_objective_warning(
        &mut converged,
        2e20,
        EstimationMethod::FoceI,
        &FitOptions::default(),
    )
    .expect("a sentinel objective is demoted")
}

/// A subject whose engine clock runs 50 ahead of its written `TIME`, as after a reset,
/// so a `TIME` read off the engine clock instead of the written one is visible.
fn shifted_subject(id: &str) -> Subject {
    let mut s = subject_with_dose_time(id, 0.0, 1.0);
    s.obs_raw_times = s.obs_times.clone();
    s.obs_times = s.obs_times.iter().map(|t| t + 50.0).collect();
    s
}

/// The contribution source of a FOCEI fit: the contributions are its own terms.
fn foce_source() -> ContributionSource {
    ContributionSource::of(EstimationMethod::FoceI, &FitOptions::default(), false, 0.0)
}

/// A term scorer for a model with no separately scored term.
fn no_terms(
    _: usize,
    _: &Subject,
    _: &SubjectResult,
) -> Vec<crate::api::postfit::ContributionTerm> {
    Vec::new()
}

fn population_of(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// **N1 — every cell of the sentence, in one message.**
///
/// The input space is the subject's contribution (valid, at the sentinel, `NaN`, or
/// finite but past the cutoff) × its records' own likelihood terms (all usable, or
/// not on some records) × its predictions (finite, or not on some records). On
/// [`one_cpt_model`]'s proportional error an `IPRED` of 0 is scored at the variance
/// floor, `DV² × 1e12`, so `shifted_subject`'s `DV`s (33.3, 16.7, 5.6, 2 at `TIME`
/// 0.5, 2, 8, 24) straddle the `1e14` cutoff between `TIME` 8 and 24. Four subjects
/// cover the cells that change the text:
///
/// | subject | contribution | `IPRED` | `PRED` | named as |
/// |---|---|---|---|---|
/// | `A` | `2e20` | 0 at `TIME` 2 and 24, `NaN` at 8 | `NaN` at 24 | terms 2 of 4 from `TIME` 2; predictions 2 of 4 from `TIME` 8 |
/// | `B` | `12.34` | `NaN` | `NaN` | not named |
/// | `C` | `4.6e14` | 0 at `TIME` 0.5 | finite | terms 1 of 4 at `TIME` 0.5; "every IPRED and PRED finite" |
/// | `D` | `NaN` | finite | finite | "every … term usable"; "every IPRED and PRED finite" |
///
/// `B` is the straddle: its predictions are worse than `A`'s, and it is the subject
/// with no Gaussian observation whose `NaN` predictions are the "no prediction
/// available" convention. Its contribution is valid, so it is not named. `A`'s zeros
/// straddle the per-record cutoff (`16.7² × 1e12 ≥ 1e14 > 2² × 1e12`), and its first
/// unusable term comes before its first non-finite prediction, which is #2266's
/// shape. The whole appended sentence is compared exactly, so deleting any clause of
/// it fires.
///
/// Mutation (run):
/// * select on non-finite predictions instead of the contribution → `B` is named and
///   `C` and `D` are not;
/// * select on `ofv_contribution >= DIVERGENCE_OFV` → `D`'s `NaN` drops out;
/// * count only `IPRED` → `A` reads "1 of 4" in the prediction clause;
/// * take the `TIME` from `obs_times` → `A` reads `TIME 58`;
/// * flag a term only when it is not finite → `A`'s starts at `TIME 8` and `C`
///   reads "every … term usable";
/// * drop the `NaN`-prediction check ahead of the floor → `A` reads "1 of 4";
/// * drop the "{n} of {N} subjects" lead, the contribution, either records clause,
///   or the `TIME`, `DV` or `IPRED` of the term clause → the sentence comparison
///   fires.
#[test]
fn the_warning_names_each_unusable_subject_with_its_first_bad_record() {
    let pop = population_of(vec![
        shifted_subject("A"),
        shifted_subject("B"),
        shifted_subject("C"),
        shifted_subject("D"),
    ]);
    let mut a = crate::types::test_helpers::dummy_subject("A", 1, 4);
    a.ofv_contribution = 2e20;
    a.ipred[1] = 0.0;
    a.ipred[2] = f64::NAN;
    a.ipred[3] = 0.0;
    a.pred[3] = f64::NAN;
    let mut b = crate::types::test_helpers::dummy_subject("B", 1, 4);
    b.ipred = vec![f64::NAN; 4];
    b.pred = vec![f64::NAN; 4];
    let mut c = crate::types::test_helpers::dummy_subject("C", 1, 4);
    c.ofv_contribution = 4.6e14;
    c.ipred[0] = 0.0;
    let mut d = crate::types::test_helpers::dummy_subject("D", 1, 4);
    d.ofv_contribution = f64::NAN;

    let m = one_cpt_model();
    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &[a, b, c, d],
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );

    let added = full
        .strip_prefix(&msg)
        .unwrap_or_else(|| panic!("the gate's message is kept whole and extended: {full}"));
    assert_eq!(
        added,
        " 3 of 4 subjects have an objective contribution that is itself unusable: \
         A (contribution 2e20, an unusable likelihood term on 2 of 4 observations, the \
         first at TIME 2 (DV 16.666666666666668, IPRED 0); a non-finite IPRED or PRED on 2 \
         of 4 observations, the first at TIME 8); C (contribution 460000000000000.0, an \
         unusable likelihood term on 1 of 4 observations, the first at TIME 0.5 \
         (DV 33.333333333333336, IPRED 0); every IPRED and PRED finite); D (contribution \
         NaN, every observation's likelihood term usable on its own; every IPRED and PRED \
         finite)."
    );
    assert_eq!(
        entry.message, full,
        "the structured message and the string one say the same thing"
    );

    let details = entry.details.as_ref().expect("the gate's payload");
    assert_eq!(
        details["reason"], "at the divergence sentinel",
        "the gate's own keys are kept: {details}"
    );
    let subjects = details["unusable_subjects"]
        .as_array()
        .expect("an unusable_subjects array");
    assert_eq!(
        subjects,
        &vec![
            serde_json::json!({"id": "A", "ofv_contribution": 2e20,
                "nonfinite_predictions": 2, "n_obs": 4, "first_time": 8.0,
                "unusable_terms": 2, "first_unusable_term_time": 2.0, "failed_terms": []}),
            serde_json::json!({"id": "C", "ofv_contribution": 4.6e14,
                "nonfinite_predictions": 0, "n_obs": 4, "first_time": null,
                "unusable_terms": 1, "first_unusable_term_time": 0.5, "failed_terms": []}),
            serde_json::json!({"id": "D", "ofv_contribution": "NaN",
                "nonfinite_predictions": 0, "n_obs": 4, "first_time": null,
                "unusable_terms": 0, "first_unusable_term_time": null, "failed_terms": []}),
        ],
        "JSON has no NaN, so a non-finite contribution is carried as a string"
    );
}

/// **N2 — nothing to name, and the warning says why** (#2267; before it, the message
/// was left as the gate wrote it). A sentinel FOCEI objective whose every subject has
/// a valid post-fit contribution, with no prior, has two explanations, told apart by
/// the sum of the contributions: a usable sum means the objective was evaluated at
/// another point than the subjects are scored at; a sum past the cutoff means it
/// failed only as a sum. Never "0 of 2 subjects".
///
/// Mutation (run): swap the `ofv_is_valid(total)` arms → both sentences fire; leave
/// the prior out of `total` → no change here (prior 0), which T5 covers.
#[test]
fn a_warning_with_no_unusable_subject_says_why_it_names_nobody() {
    let m = one_cpt_model();
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let subjects = [
        crate::types::test_helpers::dummy_subject("A", 1, 4),
        crate::types::test_helpers::dummy_subject("B", 1, 4),
    ];
    let (msg, mut entry) = sentinel_warning();
    let out = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &subjects,
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert_eq!(
        out.strip_prefix(&msg).expect("kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, and so is their sum (24.68), so the objective was evaluated \
         at another point than the estimates and EBEs the subjects are scored at."
    );
    assert_eq!(entry.message, out);
    let details = entry.details.as_ref().expect("payload");
    assert_eq!(
        details["unusable_subjects"],
        serde_json::json!([]),
        "{details}"
    );
    assert_eq!(details["ofv_prior"], 0.0, "{details}");

    let large: Vec<SubjectResult> = ["A", "B"]
        .iter()
        .map(|id| {
            let mut s = crate::types::test_helpers::dummy_subject(id, 1, 4);
            s.ofv_contribution = 6e13;
            s
        })
        .collect();
    let (msg, mut entry) = sentinel_warning();
    let out = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &large,
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert_eq!(
        out.strip_prefix(&msg).expect("kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, but their sum (120000000000000.0) is past the cutoff. The \
         objective fails only as a sum; largest first, they are A (60000000000000.0), \
         B (60000000000000.0)."
    );
    let details = entry.details.as_ref().expect("payload");
    assert_eq!(
        details["largest_subjects"],
        serde_json::json!([{"id": "A", "ofv_contribution": 6e13},
            {"id": "B", "ofv_contribution": 6e13}]),
        "{details}"
    );
    assert_eq!(
        (&details["reason"], &details["reason_read_from"]),
        (
            &serde_json::json!("at the divergence sentinel"),
            &serde_json::json!("total")
        ),
        "a 2e20 objective over subjects summing to 1.2e14 is not their sum: {details}"
    );
}

/// **N3 — a result that does not line up with the population gets no `TIME`.**
/// `subjects` is in population order, and the `TIME` is read from the population's
/// subject at the same position only when its `ID` matches. Here the results are in
/// the reverse order, so `B`'s position holds `A`'s data: printing a `TIME` would
/// print `A`'s.
///
/// Mutation (run): drop the `d.id == s.id` filter → the message names a `TIME` for
/// `B` and `first_time` is `8.0`, and this fires.
#[test]
fn a_subject_not_matched_to_its_data_is_named_without_a_time() {
    let m = one_cpt_model();
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let mut b = crate::types::test_helpers::dummy_subject("B", 1, 4);
    b.ofv_contribution = 2e20;
    b.ipred[2] = f64::NAN;
    let a = crate::types::test_helpers::dummy_subject("A", 1, 4);
    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &[b, a],
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert!(
        full.ends_with(
            "B (contribution 2e20, likelihood terms not checked; a non-finite IPRED or PRED \
             on 1 of 4 observations)."
        ),
        "{full}"
    );
    let subjects = entry.details.as_ref().expect("payload")["unusable_subjects"].clone();
    assert_eq!(
        subjects[0]["first_time"],
        serde_json::Value::Null,
        "{subjects}"
    );
    assert_eq!(subjects[0]["nonfinite_predictions"], 1, "{subjects}");
    assert_eq!(
        subjects[0]["first_unusable_term_time"],
        serde_json::Value::Null,
        "{subjects}"
    );
    assert_eq!(
        subjects[0]["unusable_terms"],
        serde_json::Value::Null,
        "unchecked, not zero: {subjects}"
    );
}

/// **N3b — the third way `first_time` is `null`** (#2148 review): the subject is
/// matched and has a non-finite prediction, but the written `TIME` of that record is
/// itself not finite. JSON has no `NaN`, so serde writes `null`, while the message
/// still prints the value. Pins the docs' list of the three `null` cases: with a
/// finite `TIME` at that record the field would be a number.
#[test]
fn a_non_finite_written_time_is_carried_as_null() {
    let m = one_cpt_model();
    let mut data = shifted_subject("A");
    data.obs_raw_times[2] = f64::NAN;
    let pop = population_of(vec![data]);
    let mut a = crate::types::test_helpers::dummy_subject("A", 1, 4);
    a.ofv_contribution = 2e20;
    a.ipred[2] = f64::NAN;
    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg,
        &mut entry,
        &[a],
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert!(full.ends_with("the first at TIME NaN)."), "{full}");
    assert!(
        full.contains("the first at TIME NaN (DV 5.555555555555555, IPRED NaN);"),
        "{full}"
    );
    let subjects = entry.details.as_ref().expect("payload")["unusable_subjects"].clone();
    assert_eq!(
        subjects[0]["first_time"],
        serde_json::Value::Null,
        "{subjects}"
    );
    assert_eq!(subjects[0]["nonfinite_predictions"], 1, "{subjects}");
}

/// **N4 — the message names ten; the details list them all.** A population where
/// every subject is unusable (one bad covariate model) would otherwise put a
/// paragraph per subject into one warning.
///
/// Mutation (run): drop the `take` → the message names `s11` and `s12`; drop the
/// "and N more" clause → its assertion fires; cap the payload too → the length
/// assertion fires.
#[test]
fn the_message_caps_the_named_subjects_and_the_details_do_not() {
    let m = one_cpt_model();
    let ids: Vec<String> = (1..=12).map(|k| format!("s{k:02}")).collect();
    let pop = population_of(ids.iter().map(|id| shifted_subject(id)).collect());
    let subjects: Vec<SubjectResult> = ids
        .iter()
        .map(|id| {
            let mut s = crate::types::test_helpers::dummy_subject(id, 1, 4);
            s.ofv_contribution = 2e20;
            s
        })
        .collect();
    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg,
        &mut entry,
        &subjects,
        &pop,
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );

    assert!(full.contains(" 12 of 12 subjects "), "{full}");
    assert!(full.contains("s10 (contribution 2e20"), "{full}");
    assert!(
        !full.contains("s11 (") && !full.contains("s12 ("),
        "only {} are named in the message: {full}",
        crate::api::postfit::NONFINITE_SUBJECTS_NAMED
    );
    assert!(
        full.ends_with(
            "; and 2 more, listed as `unusable_subjects` in the structured warning's details \
             (Rust API, JSON and YAML output, .fitrx; not yet in the R package)."
        ),
        "{full}"
    );
    let listed = entry.details.as_ref().expect("payload")["unusable_subjects"]
        .as_array()
        .expect("subjects")
        .len();
    assert_eq!(listed, 12, "the details list every subject");
}

/// The #2038 models: warfarin with `CL ∝ (WT / 70)^0.75`, which is `NaN` at a negative
/// `WT`, on the closed form and on the equivalent ODE.
fn wt_power_model(ode: bool) -> CompiledModel {
    let structural = if ode {
        "[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  \
         d/dt(depot)   = -KA * depot\n  d/dt(central) = KA * depot / V - (CL / V) * central\n"
    } else {
        "[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n"
    };
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT / 70)^0.75
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
{structural}[error_model]
  DV ~ proportional(PROP_ERR)
"#
    ))
    .expect("parse")
}

/// Which records of warfarin subject 3 get a negative `WT` (every other record has 70).
#[derive(Clone, Copy)]
enum NegativeWt {
    Nowhere,
    EveryRecord,
    FourthObservation,
}

/// `data/warfarin.csv` with a `WT` column.
fn warfarin_with_wt(neg: NegativeWt) -> Population {
    use std::io::Write;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/warfarin.csv");
    let text = std::fs::read_to_string(path).expect("data/warfarin.csv");
    let mut lines = text.lines();
    let mut csv = format!("{},WT\n", lines.next().expect("header"));
    let mut s3_obs = 0;
    for line in lines {
        let cols: Vec<&str> = line.split(',').collect();
        let s3 = cols[0] == "3";
        let obs = cols[3] == "0";
        if s3 && obs {
            s3_obs += 1;
        }
        let negative = s3
            && match neg {
                NegativeWt::Nowhere => false,
                NegativeWt::EveryRecord => true,
                NegativeWt::FourthObservation => obs && s3_obs == 4,
            };
        csv.push_str(&format!("{line},{}\n", if negative { "-5" } else { "70" }));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), Some(&["WT"]), None).expect("read the #2038 fixture")
}

fn evaluate(model: &CompiledModel, pop: &Population) -> FitResult {
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        threads: Some(1),
        ..Default::default()
    };
    fit(model, pop, &model.default_params, &opts).expect("fit")
}

/// The one `W_NONFINITE_OBJECTIVE` entry and its `unusable_subjects` payload.
fn named_subjects(result: &FitResult) -> (String, Vec<serde_json::Value>) {
    let entry = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains(NONFINITE_OBJECTIVE_TOKEN))
        .unwrap_or_else(|| panic!("no {NONFINITE_OBJECTIVE_TOKEN}: {:?}", result.warnings));
    let subjects = entry.details.as_ref().expect("payload")["unusable_subjects"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    (entry.message.clone(), subjects)
}

/// **N5 — end to end, the issue's reproduction.**
///
/// One subject in ten with a negative `WT` on every record makes `CL` `NaN` there
/// and repels the fit. Before #2038 the warning listed every possible cause and named
/// no subject; the sdtab did not say which either, since its `NaN` rows look like the
/// "no prediction available" convention. The control (`WT = 70` everywhere) is part
/// of the test: it pins that the naming is caused by the negative weight and not by
/// the fixture, and that a healthy fit stays silent.
///
/// Mutation (run): delete the `name_nonfinite_subjects` call in `fit_inner` → the
/// message names nobody and the `unusable_subjects` payload is absent. Moving it before the
/// `warnings.retain(...)` de-duplication is caught by G1 and G10 instead, not here:
/// the estimator's copy no longer ends with the extended message, so both copies
/// survive, but at `outer_maxiter = 0` this fit has no estimator copy to keep.
#[test]
fn a_negative_covariate_on_every_record_names_its_subject() {
    let model = wt_power_model(false);

    let control = evaluate(&model, &warfarin_with_wt(NegativeWt::Nowhere));
    assert!(
        crate::estimation::outer_optimizer::ofv_is_valid(control.ofv),
        "premise: WT = 70 everywhere is a healthy fit; ofv = {}",
        control.ofv
    );
    assert!(
        !control
            .warnings
            .iter()
            .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
        "{:?}",
        control.warnings
    );

    let result = evaluate(&model, &warfarin_with_wt(NegativeWt::EveryRecord));
    assert_eq!(result.ofv, 2e20, "premise: one clamped subject");
    let (message, subjects) = named_subjects(&result);
    assert!(
        message.ends_with(
            " 1 of 10 subjects have an objective contribution that is itself unusable: \
             3 (contribution 2e20, an unusable likelihood term on 11 of 11 observations, the \
             first at TIME 0.5 (DV 4.3905, IPRED NaN); a non-finite IPRED or PRED on 11 of \
             11 observations, the first at TIME 0.5, failed terms: Gaussian residual term = \
             NaN, curvature (H̃ not positive definite) = NaN)."
        ),
        "{message}"
    );
    assert_eq!(subjects.len(), 1, "only subject 3 is named: {subjects:?}");
    assert_eq!(
        result
            .warnings
            .iter()
            .filter(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN))
            .collect::<Vec<_>>(),
        vec![&message],
        "the plain-text warning is the named one, once"
    );
}

/// **N6 — end to end, one bad record, on both engines.** A negative `WT` on subject
/// 3's fourth observation only: each engine carries the `NaN` into that record and
/// every later one, and the sentence names the first of them.
///
/// Before #2039 the closed form served a finite prediction of 0 at that record, which
/// a proportional error scored at about `4.6e14`, so it named the subject "every IPRED
/// and PRED finite". Since #2039 it serves the `NaN`, and both engines must give the
/// same sentence. Each failure names its engine.
#[test]
fn a_negative_covariate_on_one_record_names_the_record() {
    let pop = warfarin_with_wt(NegativeWt::FourthObservation);

    for (engine, ode) in [("ODE", true), ("closed form", false)] {
        let result = evaluate(&wt_power_model(ode), &pop);
        let (message, subjects) = named_subjects(&result);
        assert!(
            message.ends_with(
                " 1 of 10 subjects have an objective contribution that is itself unusable: \
                 3 (contribution 2e20, an unusable likelihood term on 8 of 11 observations, \
                 the first at TIME 4 (DV 11.0602, IPRED NaN); a non-finite IPRED or PRED on 8 \
                 of 11 observations, the first at TIME 4, failed terms: Gaussian residual \
                 term = NaN, curvature (H̃ not positive definite) = NaN)."
            ),
            "{engine}: {message}"
        );
        assert_eq!(
            subjects.len(),
            1,
            "{engine}: only subject 3 is named: {subjects:?}"
        );
    }
}

/// **N1b — a runaway residual-error η is named although `IWRES` reads 0** (#2266
/// review). Under `iiv_on_ruv` the record's variance is scaled by `exp(2η)`, which
/// overflows to `inf` at `η = 400`: the likelihood scores that record at `inf`, while
/// `IWRES`, divided by `√V`, reads `0`. A selector on `IWRES` reported "every …
/// term usable" here. The control is the same subject at `η = 0`, so the naming is
/// caused by the η, and the subject's `IWRES` is set to `0` in both so it cannot be
/// what decides.
///
/// Mutation (run): select on `IWRES²` → the `η = 400` half fires; drop the
/// `residual_var_scale` factor from the term → the same.
#[test]
fn a_runaway_residual_error_eta_names_every_record() {
    let m = parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  omega ETA_RUV ~ 0.05
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
  iiv_on_ruv = ETA_RUV
"#,
    )
    .expect("parse");
    let pop = population_of(vec![shifted_subject("A")]);
    for (eta_ruv, want) in [
        (
            400.0,
            "an unusable likelihood term on 4 of 4 observations, the first at TIME 0.5 \
             (DV 33.333333333333336, IPRED 1);",
        ),
        (
            0.0,
            "every observation's likelihood term usable on its own;",
        ),
    ] {
        let mut a = crate::types::test_helpers::dummy_subject("A", 2, 4);
        a.ofv_contribution = 2e20;
        a.eta[1] = eta_ruv;
        a.iwres = vec![0.0; 4];
        let (msg, mut entry) = sentinel_warning();
        let full = name_nonfinite_subjects(
            msg,
            &mut entry,
            &[a],
            &pop,
            &m,
            &m.default_params,
            None,
            &foce_source(),
            &no_terms,
        );
        assert!(full.contains(want), "η_RUV = {eta_ruv}: {full}");
    }
}

/// **N1c — a censored record is scored by its own M3 term, not as a Gaussian one.**
/// Under `bloq_method = m3` a `CENS = 1` row contributes `−log Φ((LLOQ − f)/√V)`, which
/// is small when the prediction is far below the limit; the Gaussian term of the
/// same `DV` and an `IPRED` of 0 is `DV² × 1e12`. One record, both methods: the term
/// has to come from the likelihood's own censoring branch, not a Gaussian formula.
#[test]
fn a_censored_record_is_scored_by_its_m3_term() {
    let pop = population_of(vec![{
        let mut d = shifted_subject("A");
        d.cens[0] = 1;
        d
    }]);
    let mut a = crate::types::test_helpers::dummy_subject("A", 1, 4);
    a.ofv_contribution = 2e20;
    a.ipred[0] = 0.0;
    for (bloq, want) in [
        (
            BloqMethod::M3,
            "every observation's likelihood term usable on its own;",
        ),
        (
            BloqMethod::Drop,
            "an unusable likelihood term on 1 of 4 observations, the first at TIME 0.5 \
             (DV 33.333333333333336, IPRED 0);",
        ),
    ] {
        let mut m = one_cpt_model();
        m.bloq_method = bloq;
        let (msg, mut entry) = sentinel_warning();
        let full = name_nonfinite_subjects(
            msg,
            &mut entry,
            &[a.clone()],
            &pop,
            &m,
            &m.default_params,
            None,
            &foce_source(),
            &no_terms,
        );
        assert!(full.contains(want), "{}: {full}", bloq.label());
    }
}

/// `data/warfarin.csv` with a `WT` column, where subject 3 gains an observation at
/// `TIME 0` (with the dose) and every one of its observations reads `dv`.
fn warfarin_with_constant_dv(dv: f64) -> Population {
    use std::io::Write;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/warfarin.csv");
    let text = std::fs::read_to_string(path).expect("data/warfarin.csv");
    let mut lines = text.lines();
    let mut csv = format!("{},WT\n", lines.next().expect("header"));
    for line in lines {
        let cols: Vec<&str> = line.split(',').collect();
        match (cols[0], cols[3]) {
            ("3", "1") => {
                csv.push_str(&format!("{line},70\n3,0,{dv},0,.,1,0,0,70\n"));
            }
            ("3", _) => csv.push_str(&format!("3,{},{dv},0,.,1,0,0,70\n", cols[1])),
            _ => csv.push_str(&format!("{line},70\n")),
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), Some(&["WT"]), None).expect("read the #2266 fixture")
}

/// **N7 — the issue's reproduction (#2266): both engines name the `TIME 0` record.**
///
/// Subject 3 of warfarin gets an observation at `TIME 0`, with the oral dose, so both
/// engines predict a finite `0` there whatever η̂ is, and a proportional error scores
/// it at the guarded variance `1e-12`, `DV² × 1e12`. That record alone makes the contribution
/// unusable. Two `DV` levels, measured:
///
/// | `DV` | engine | η̂ | predictions | prediction clause |
/// |---|---|---|---|---|
/// | 200 | both | ≈ (−5.7, −3.0, 1.3) | all finite | "every IPRED and PRED finite" |
/// | 1e6 | closed form | runs away | `IPRED` `NaN` at `TIME` 0, 0 after | first at `TIME 0` |
/// | 1e6 | ODE | runs away | `IPRED` 0 at `TIME` 0, `NaN` after | first at `TIME 0.5` |
///
/// At 1e6 the EBE runs away and the engines disagree on the predictions, which is
/// what #2266 reported: the prediction clause names `TIME 0.5` on the ODE and the
/// closed form, at 200, names no record at all. (The fixture used `DV` 20000 until
/// #2441: with the absolute `1e-12` variance floor gone, every later row of subject 3
/// is scored at its own variance, and at 20000 the closed form no longer runs away —
/// all its predictions stay finite — so the two engines stopped straddling.) The likelihood-term clause must name
/// `TIME 0` in all four cells. The prediction clauses are asserted too, so the cells
/// keep straddling: if both engines stopped disagreeing, the fixture would no longer
/// test the property the issue is about.
///
/// Mutation (run): name the first non-finite prediction in the term clause →
/// the ODE at 1e6 names `TIME 0.5` and both engines at 200 name nothing; flag a
/// term only when it is not finite → the ODE at 1e6 and both at 200 fire.
#[test]
fn a_runaway_ebe_names_the_record_with_the_unusable_term_on_both_engines() {
    let expected_predictions = |dv: f64, ode: bool| match (dv == 200.0, ode) {
        (true, _) => "every IPRED and PRED finite",
        (false, false) => "the first at TIME 0, failed term",
        (false, true) => {
            "a non-finite IPRED or PRED on 11 of 12 observations, the first at TIME 0.5, failed \
             term"
        }
    };
    for dv in [200.0, 1e6] {
        let pop = warfarin_with_constant_dv(dv);
        for (engine, ode) in [("ODE", true), ("closed form", false)] {
            let result = evaluate(&wt_power_model(ode), &pop);
            let (message, subjects) = named_subjects(&result);
            assert_eq!(
                subjects.len(),
                1,
                "{engine} at DV {dv}: only subject 3 is named: {subjects:?}"
            );
            assert_eq!(subjects[0]["id"], "3", "{engine} at DV {dv}");
            assert_eq!(
                subjects[0]["first_unusable_term_time"], 0.0,
                "{engine} at DV {dv}: {subjects:?}"
            );
            assert!(
                message.contains(&format!(
                    "observations, the first at TIME 0 (DV {dv}, IPRED "
                )),
                "{engine} at DV {dv}: the term clause names TIME 0: {message}"
            );
            assert!(
                message.contains(expected_predictions(dv, ode)),
                "{engine} at DV {dv}: premise, the prediction clause: {message}"
            );
        }
    }
}

/// One subject `A` (`shifted_subject`) whose first record reads `dv`, named by a
/// sentinel warning with `ipred0` as that record's `IPRED`; returns the sentence.
fn name_first_record(
    model: &CompiledModel,
    params: &crate::types::ModelParameters,
    mixest: Option<&[usize]>,
    dv: f64,
    ipred0: f64,
) -> String {
    let mut data = shifted_subject("A");
    data.observations[0] = dv;
    let pop = population_of(vec![data]);
    let mut a = crate::types::test_helpers::dummy_subject("A", model.n_eta, 4);
    a.eta = nalgebra::DVector::zeros(model.n_eta);
    a.ofv_contribution = 2e20;
    a.ipred = vec![ipred0, 2.5, 3.0, 2.0];
    let (msg, mut entry) = sentinel_warning();
    name_nonfinite_subjects(
        msg,
        &mut entry,
        &[a],
        &pop,
        model,
        params,
        mixest,
        &foce_source(),
        &no_terms,
    )
}

const NAMES_FIRST: &str =
    "an unusable likelihood term on 1 of 4 observations, the first at TIME 0.5";
const NAMES_NONE: &str = "every observation's likelihood term usable on its own;";

/// **N1d — the term is the scorer's, not a floored copy** (#2266 review 2). FOCEI
/// scores a record at its raw `IPRED`: a proportional error (σ = 0.2) on `DV = 20`
/// at `IPRED = −1` has `V = 0.04` and a term of `21²/0.04 + ln 0.04 ≈ 1.1e4`. Flooring
/// the prediction to `1e-12` first, as the SAEM/IS objective did until #2440, scores it at
/// `20² × 1e12 = 4e14` and names it. The straddle is `IPRED = 0`, where `V` itself is
/// at its floor and the record is unusable either way.
///
/// Mutation (run): floor the prediction before the variance → the `−1` half fires.
#[test]
fn a_negative_prediction_is_scored_as_the_scorer_scores_it() {
    let m = parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.2 (sd)
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
    let p = &m.default_params;
    let got = name_first_record(&m, p, None, 20.0, -1.0);
    assert!(got.contains(NAMES_NONE), "IPRED −1: {got}");
    let got = name_first_record(&m, p, None, 20.0, 0.0);
    assert!(
        got.contains(&format!("{NAMES_FIRST} (DV 20, IPRED 0);")),
        "IPRED 0: {got}"
    );
}

/// Two classes of one proportional error: class 1's σ is `1e-7` (sd), class 2's `0.5`.
fn mixture_model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.04
  sigma EPS ~ 1e-7 (sd)
[mixture]
  nsub = 2
  logit(1) = MIXL
  sigma(2) EPS ~ 0.5 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(EPS)
"#,
    )
    .expect("parse")
}

/// **N1e — a mixture subject is scored under its fitted class's Σ** (#2266 review 2).
/// Class 1's proportional σ is `1e-7` (sd), class 2's is `0.5`. At `DV = 33`,
/// `IPRED = 1` the term is `32²/1e-14 ≈ 1e17` under class 1 and `≈ 4e3` under class 2:
/// a subject assigned to class 2 must not be named, and the same subject in class 1
/// must be. The base Σ is class 1's, so reading `params.sigma` instead of the class's
/// names the class-2 subject.
///
/// Mutation (run): pass the base `params` instead of `class_params` → the class-2
/// half fires.
#[test]
fn a_mixture_subject_is_scored_under_its_fitted_class() {
    let m = mixture_model();
    let p = &m.default_params;
    let got = name_first_record(&m, p, Some(&[1]), 33.0, 1.0);
    assert!(got.contains(NAMES_NONE), "class 2: {got}");
    // Class 1 names the other records too (their residuals are as large).
    let got = name_first_record(&m, p, Some(&[0]), 33.0, 1.0);
    assert!(
        got.contains("observations, the first at TIME 0.5 (DV 33, IPRED 1);"),
        "class 1: {got}"
    );
}

/// **N1e′ — the selector and the clauses agree on a mixture subject's class**
/// (#2266 review 3), end to end through `compute_subject_results`. Every `DV` is
/// `1000`, far above `IPRED` (≤ 10 from a 100 dose into `V = 10`): under class 1's
/// σ = 1e-7 each record scores `≥ (990/1e-6)² ≈ 1e18`, past the `1e14` cutoff;
/// under class 2's σ = 0.5 the subject's contribution is `≈ 1e7`. A class-2 subject
/// must therefore get a usable contribution and not be named at all, while the same
/// subject in class 1 is named. Scoring the contribution with the base (class-1)
/// Σ — what the guard alone does, since it moves only `MIXNUM` — names the class-2
/// subject, with its records then described as usable under class 2.
///
/// Mutation (run): drop the `class_params` swap in `compute_subject_results` → the
/// class-2 half fires.
#[test]
fn a_mixture_subject_is_selected_under_its_fitted_class() {
    let m = mixture_model();
    let mut data = shifted_subject("A");
    data.observations = vec![1000.0; 4];
    let pop = population_of(vec![data]);
    let eta = nalgebra::DVector::zeros(m.n_eta);
    let h = nalgebra::DMatrix::zeros(4, m.n_eta);
    let run = |class: usize| {
        let (subjects, _) = compute_subject_results(
            &m,
            &pop,
            &m.default_params,
            std::slice::from_ref(&eta),
            std::slice::from_ref(&h),
            &[Vec::new()],
            true,
            Some(&[class]),
            false,
        );
        let c = subjects[0].ofv_contribution;
        let (msg, mut entry) = sentinel_warning();
        let got = name_nonfinite_subjects(
            msg.clone(),
            &mut entry,
            &subjects,
            &pop,
            &m,
            &m.default_params,
            Some(&[class]),
            &foce_source(),
            &no_terms,
        );
        let unnamed = !got.contains(" subjects have an objective contribution that is itself");
        (c, unnamed, got)
    };
    let (c2, unnamed2, got2) = run(1);
    assert!(
        c2.is_finite() && c2 < 1e14,
        "class 2: contribution {c2} is usable"
    );
    assert!(unnamed2, "class 2 is not named: {got2}");
    let (c1, unnamed1, got1) = run(0);
    assert!(
        !(c1 < 1e14),
        "premise, class 1: contribution {c1} past the cutoff"
    );
    assert!(
        !unnamed1
            && got1.contains(
                "an unusable likelihood term on 4 of 4 observations, the first at TIME 0.5"
            ),
        "class 1 is named, by all four records: {got1}"
    );
}

/// **N1f — the fitted residual correlation, not the declared one** (#2266 review 2).
/// A combined error `V = (fσ_p)² + σ_a² + 2ρ·fσ_p·σ_a` with `fσ_p = σ_a = 1` (at
/// `IPRED = 10`) is `2` at the declared `ρ = 0` and `0` at a fitted `ρ = −1`, floored
/// to `1e-12`, so `DV = 30` scores `20² × 1e12 = 4e14` there and `≈ 200` at `ρ = 0`. Setting only the parameters' `ρ` must name the record; the
/// declared model alone must not.
///
/// Mutation (run): build the covariance from the model's declaration
/// (`residual_cov_theta`) → the fitted half fires.
#[test]
fn the_fitted_residual_correlation_is_the_one_scored() {
    let m = parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP, ADD) = [0.01, 0.0, 1.0]
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(PROP, ADD)
"#,
    )
    .expect("parse");
    let declared = &m.default_params;
    assert_eq!(
        declared.residual_correlations.len(),
        1,
        "premise: one declared correlation, at ρ = 0: {:?}",
        declared.residual_correlations
    );
    let got = name_first_record(&m, declared, None, 30.0, 10.0);
    assert!(got.contains(NAMES_NONE), "declared ρ = 0: {got}");
    let mut fitted = declared.clone();
    fitted.residual_correlations[0].rho = -1.0;
    let got = name_first_record(&m, &fitted, None, 30.0, 10.0);
    assert!(
        got.contains(&format!("{NAMES_FIRST} (DV 30, IPRED 10);")),
        "fitted ρ = −1: {got}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  T — which term failed, and which objective the contributions belong to (#2267)
// ─────────────────────────────────────────────────────────────────────────────

fn term(name: &str, nll: f64) -> crate::api::postfit::ContributionTerm {
    crate::api::postfit::ContributionTerm {
        name: name.into(),
        nll,
    }
}

/// **T1 — the sentence names the term that failed.**
///
/// The input space is the subject's contribution (valid or not) × its separately
/// scored terms (none, all usable, one failed, two failed) × whether it has a Gaussian
/// observation. The cells that change the text:
///
/// | subject | contribution | Gaussian obs | terms | named as |
/// |---|---|---|---|---|
/// | `T` | `2e20` | none | TTE `1e20` | "no Gaussian observation, failed term: …" |
/// | `U` | `NaN` | 4, finite | TTE `3.25` | "every IPRED and PRED finite, every separately scored term usable" |
/// | `V` | `2e18` | 4, finite | the IOV + FREM refusal | "failed term: IOV + FREM refusal = 1e18" |
/// | `W` | `2e20` | 4, finite | TTE `NaN`, categorical `1e20` | "failed terms: …, …" |
/// | `X` | `12.34` | 4, finite | TTE `1e20` | not named |
///
/// `X` is the straddle: a failing term with a valid contribution is not named, so the
/// selector is still the contribution and the terms only describe it. `T` is the
/// issue's TTE-only subject, which used to read "every IPRED and PRED finite" because
/// it has no prediction at all.
///
/// Mutation (run): drop the `n_obs == 0` arm → `T` reads "every IPRED and PRED
/// finite"; filter the failed terms on `is_finite()` instead of `ofv_is_valid` → `T`
/// and `V` lose their term and read "every separately scored term usable"; drop the "every
/// separately scored term usable" arm, the singular/plural split, or the value → the
/// sentence comparison fires; select on the terms → `X` is named.
#[test]
fn the_sentence_names_the_term_that_failed() {
    let ids = ["T", "U", "V", "W", "X"];
    let pop = population_of(ids.iter().map(|id| shifted_subject(id)).collect());
    let mut t = crate::types::test_helpers::dummy_subject("T", 1, 0);
    t.ofv_contribution = 2e20;
    let mut u = crate::types::test_helpers::dummy_subject("U", 1, 4);
    u.ofv_contribution = f64::NAN;
    let mut v = crate::types::test_helpers::dummy_subject("V", 1, 4);
    v.ofv_contribution = 2e18;
    let mut w = crate::types::test_helpers::dummy_subject("W", 1, 4);
    w.ofv_contribution = 2e20;
    let x = crate::types::test_helpers::dummy_subject("X", 1, 4);
    let scorer = |i: usize, data: &Subject, s: &SubjectResult| {
        assert_eq!((data.id.as_str(), s.id.as_str()), (ids[i], ids[i]));
        match ids[i] {
            "T" | "X" => vec![term("time-to-event on CMT 2", 1e20)],
            "U" => vec![term("time-to-event on CMT 2", 3.25)],
            "V" => vec![term("IOV + FREM refusal", 1e18)],
            "W" => vec![
                term("time-to-event on CMT 2", f64::NAN),
                term("categorical", 1e20),
            ],
            other => panic!("{other}"),
        }
    };

    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &[t, u, v, w, x],
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &foce_source(),
        &scorer,
    );
    assert_eq!(
        full.strip_prefix(&msg).expect("the gate's message is kept"),
        " 4 of 5 subjects have an objective contribution that is itself unusable: \
         T (contribution 2e20, no Gaussian observation, failed term: time-to-event on CMT 2 = \
         1e20); U (contribution NaN, every observation's likelihood term usable on its own; \
         every IPRED and PRED finite, every separately scored term usable); V (contribution \
         2e18, every observation's likelihood term usable on its own; every IPRED and PRED \
         finite, failed term: IOV + FREM refusal = 1e18); W (contribution 2e20, every \
         observation's likelihood term usable on its own; every IPRED and PRED finite, failed \
         terms: \
         time-to-event on CMT 2 = NaN, categorical = 1e20)."
    );
    let details = entry.details.as_ref().expect("payload");
    assert_eq!(details["contribution_score"], "FOCEI", "{details}");
    assert_eq!(
        details["contributions_are_objective_terms"], true,
        "{details}"
    );
    let failed: Vec<&serde_json::Value> = details["unusable_subjects"]
        .as_array()
        .expect("subjects")
        .iter()
        .map(|s| &s["failed_terms"])
        .collect();
    assert_eq!(
        failed,
        vec![
            &serde_json::json!([{"term": "time-to-event on CMT 2", "nll": 1e20}]),
            &serde_json::json!([]),
            &serde_json::json!([{"term": "IOV + FREM refusal", "nll": 1e18}]),
            &serde_json::json!([
                {"term": "time-to-event on CMT 2", "nll": "NaN"},
                {"term": "categorical", "nll": 1e20}
            ]),
        ]
    );
}

/// **T2 — which objectives the post-hoc score is a term of.** Classified by the
/// objective published, not the estimator: every `2·pop_nll` objective (FOCE family,
/// SAEM, IMP, IMPMAP, VI's Laplace final OFV) sums the contributions
/// `compute_subject_results` scores; quadrature, BAYES's posterior-mean NLL, a mixture
/// marginal and an interaction mismatch are named, with node count and mixture.
///
/// Mutation (run): drop any clause of `same` → its row fires; drop the `n_agq`, BAYES
/// or mixture suffix → its row fires; go back to a FOCE-family method list → the
/// SAEM/IMP/IMPMAP/VI rows fire.
#[test]
fn the_contribution_source_names_every_objective_it_is_not_a_term_of() {
    use EstimationMethod::*;
    let opts = |interaction: bool, n_agq: usize| FitOptions {
        interaction,
        n_agq,
        ..FitOptions::default()
    };
    let cases: [(EstimationMethod, bool, usize, bool, &str, Option<&str>); 15] = [
        (Foce, false, 1, false, "FOCE", None),
        (FoceI, true, 1, false, "FOCEI", None),
        (FoceGn, true, 1, false, "FOCEI", None),
        (FoceGnHybrid, false, 1, false, "FOCE", None),
        (FoceI, true, 3, false, "FOCEI", Some("FOCEI (n_agq = 3)")),
        (Laplace, true, 1, false, "FOCEI", Some("LAPLACE")),
        (
            Laplace,
            true,
            5,
            false,
            "FOCEI",
            Some("LAPLACE (n_agq = 5)"),
        ),
        // SAEM, IMP, IMPMAP and VI's `vi_final_ofv = laplace` publish `2·pop_nll`,
        // the sum of these scores (review r1).
        (Saem, true, 1, false, "FOCEI", None),
        (Imp, false, 1, false, "FOCE", None),
        (Impmap, true, 1, false, "FOCEI", None),
        (Vi, true, 1, false, "FOCEI", None),
        (Saem, true, 1, true, "FOCEI", Some("SAEM mixture")),
        (Bayes, true, 1, false, "FOCEI", Some("BAYES posterior-mean")),
        (FoceI, true, 1, true, "FOCEI", Some("FOCEI mixture")),
        // A FOCE objective whose contributions were scored under interaction is not
        // their sum either.
        (Foce, true, 1, false, "FOCEI", Some("FOCE")),
    ];
    for (method, interaction, n_agq, mixture, score, objective) in cases {
        let got = ContributionSource::of(method, &opts(interaction, n_agq), mixture, 0.0);
        assert_eq!(
            (got.score, got.objective.as_deref()),
            (score, objective),
            "{method:?}, interaction {interaction}, n_agq {n_agq}, mixture {mixture}"
        );
    }
}

/// **T3 — an objective the contributions are not terms of says so, and says why it
/// names nobody.** The same subjects under the FOCEI source and a Laplace one: with no
/// unusable contribution the FOCEI warning gives N2's sum sentence, and the
/// Laplace one the sentence saying its objective is not split; with one, the
/// Laplace list is followed by the caveat.
///
/// Without unusable contributions the Laplace sentence says the scores cannot locate the
/// failure and reports their sum on both sides of the cutoff (review r6: it used to
/// attribute the failure to a part of Laplace the score does not share, which does not
/// follow when the scores themselves sum past the cutoff).
///
/// Mutation (run): restore the unconditional `named.is_empty()` return → the "No
/// subject can be named" assertion fires; drop the caveat → the second assertion
/// fires; append either under the FOCEI source → the two controls fire; swap the sum's
/// two arms → both Laplace sentences fire.
#[test]
fn a_non_foce_objective_says_whose_contributions_it_lists_or_why_it_names_nobody() {
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let laplace = ContributionSource::of(
        EstimationMethod::Laplace,
        &FitOptions::default(),
        false,
        0.0,
    );
    let healthy = [
        crate::types::test_helpers::dummy_subject("A", 1, 4),
        crate::types::test_helpers::dummy_subject("B", 1, 4),
    ];
    let mut b = crate::types::test_helpers::dummy_subject("B", 1, 4);
    b.ofv_contribution = 2e20;
    let one_bad = [crate::types::test_helpers::dummy_subject("A", 1, 4), b];

    let (msg, mut entry) = sentinel_warning();
    let control = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &healthy,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert!(
        !control.contains("LAPLACE") && !control.contains("not split by subject"),
        "a FOCEI objective's contributions are its terms: {control}"
    );

    let (msg, mut entry) = sentinel_warning();
    let nobody = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &healthy,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &laplace,
        &no_terms,
    );
    assert_eq!(
        nobody.strip_prefix(&msg).expect("kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, but the LAPLACE objective is not their sum and ferx does not \
         split it by subject, so these scores cannot locate its failure; their sum (24.68) \
         is usable."
    );
    // The other side of the sum (review r6): scores each usable whose sum is past the
    // cutoff are reported as such, not as evidence the failure is Laplace's own.
    let large: Vec<SubjectResult> = ["A", "B"]
        .iter()
        .map(|id| {
            let mut s = crate::types::test_helpers::dummy_subject(id, 1, 4);
            s.ofv_contribution = 6e13;
            s
        })
        .collect();
    let (msg_large, mut entry_large) = sentinel_warning();
    let past = name_nonfinite_subjects(
        msg_large.clone(),
        &mut entry_large,
        &large,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &laplace,
        &no_terms,
    );
    assert!(
        past.ends_with(
            "so these scores cannot locate its failure; their sum (120000000000000.0) is past \
             the cutoff as well."
        ),
        "{past}"
    );
    let details = entry.details.as_ref().expect("payload");
    assert_eq!(
        details["unusable_subjects"],
        serde_json::json!([]),
        "{details}"
    );
    assert_eq!(
        details["contributions_are_objective_terms"], false,
        "{details}"
    );
    assert_eq!(entry.message, nobody);

    let (msg, mut entry) = sentinel_warning();
    let control = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &one_bad,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert!(
        control.ends_with("every IPRED and PRED finite)."),
        "{control}"
    );

    let (msg, mut entry) = sentinel_warning();
    let listed = name_nonfinite_subjects(
        msg,
        &mut entry,
        &one_bad,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &laplace,
        &no_terms,
    );
    assert!(
        listed.ends_with(
            "B (contribution 2e20, every observation's likelihood term usable on its own; \
             every IPRED and PRED finite). These are post-hoc FOCEI \
             contributions at the EBEs, not terms of the LAPLACE objective, which ferx does \
             not split by subject."
        ),
        "{listed}"
    );
}

/// An exponential TTE-only model whose rate carries `exp(Z)`, on the 30-subject
/// `data/tte_exponential.csv` with a `Z` column: `0` everywhere, or `1000` on subject
/// 3, where the rate overflows to `inf` and `tte_nll_from_curves` maps the subject to
/// its `1e20` sentinel.
#[cfg(feature = "survival")]
fn overflowing_tte(overflow: bool) -> (CompiledModel, Population) {
    use std::io::Write;
    let model = parse_model_string(
        r#"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA + Z)
"#,
    )
    .expect("parse");
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/tte_exponential.csv");
    let text = std::fs::read_to_string(path).expect("data/tte_exponential.csv");
    let mut lines = text.lines();
    let mut csv = format!("{},Z\n", lines.next().expect("header"));
    for line in lines {
        let z = if overflow && line.starts_with("3,") {
            "1000"
        } else {
            "0"
        };
        csv.push_str(&format!("{line},{z}\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let (pop, _) = crate::api::read_population_for(
        &model,
        &None,
        f.path().to_str().expect("utf-8 path"),
        None,
        None,
        None,
        &[],
    )
    .expect("data");
    (model, pop)
}

/// **T4 — end to end, the issue's TTE reproduction, on a FOCE-family objective and a
/// Laplace one.** Subject 3 of a TTE-only population has an overflowing hazard. Before
/// #2267 it was named "every IPRED and PRED finite", which it meets trivially: it has
/// no Gaussian prediction. Now the sentence names the TTE term at its sentinel, and
/// under Laplace says the contributions are FOCEI ones. The control (`Z = 0`) pins that
/// the naming is caused by the overflow and that a healthy fit stays silent.
///
/// Mutation (run): pass a scorer returning no term in `fit_inner` → the "failed
/// term" clause is missing; pass `ContributionSource::of(EstimationMethod::FoceI, …)`
/// there → the Laplace caveat is missing.
#[cfg(feature = "survival")]
#[test]
fn an_overflowing_hazard_names_its_time_to_event_term() {
    let run = |method: EstimationMethod, overflow: bool| {
        let (model, pop) = overflowing_tte(overflow);
        let opts = FitOptions {
            method,
            outer_maxiter: 0,
            run_covariance_step: false,
            threads: Some(1),
            ..Default::default()
        };
        fit(&model, &pop, &model.default_params, &opts).expect("fit")
    };
    let subject_3 = "3 (contribution 2e20, no Gaussian observation, failed term: \
                     time-to-event on CMT 2 = 1e20).";
    for (method, tail) in [
        (EstimationMethod::FoceI, String::new()),
        (
            EstimationMethod::Laplace,
            " These are post-hoc FOCEI contributions at the EBEs, not terms of the LAPLACE \
             objective, which ferx does not split by subject."
                .to_string(),
        ),
    ] {
        let control = run(method, false);
        assert!(
            crate::estimation::outer_optimizer::ofv_is_valid(control.ofv),
            "{method:?}: premise: Z = 0 is a healthy fit; ofv = {}",
            control.ofv
        );
        assert!(
            !control
                .warnings
                .iter()
                .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
            "{method:?}: {:?}",
            control.warnings
        );

        let result = run(method, true);
        let (message, subjects) = named_subjects(&result);
        let want = format!(
            " 1 of 30 subjects have an objective contribution that is itself unusable: \
             {subject_3}{tail}"
        );
        assert!(message.ends_with(&want), "{method:?}: {message}");
        assert_eq!(
            subjects[0]["failed_terms"],
            serde_json::json!([{"term": "time-to-event on CMT 2", "nll": 1e20}]),
            "{method:?}"
        );
    }
}

/// **T5 — the prior penalty is named, before and after a list.** An unusable prior
/// alongside a named subject is appended after the list; alongside none it is the
/// reason given, ahead of the Laplace and sum explanations; and a usable prior enters
/// the sum, so `6e13 + 6e13 + 5` reads as the sum past the cutoff with the prior
/// quoted. The end-to-end form is G5's last assertion.
///
/// Mutation (run): drop the `prior_failed` arm → the second assertion reads the
/// Laplace sentence; drop the post-list clause → the first fires; leave the prior out
/// of `total` or of the clause → the third fires.
#[test]
fn an_unusable_prior_penalty_is_named_as_a_population_level_term() {
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let source =
        |method, prior| ContributionSource::of(method, &FitOptions::default(), false, prior);
    let mut b = crate::types::test_helpers::dummy_subject("B", 1, 4);
    b.ofv_contribution = 2e20;
    let one_bad = [crate::types::test_helpers::dummy_subject("A", 1, 4), b];
    let healthy = [
        crate::types::test_helpers::dummy_subject("A", 1, 4),
        crate::types::test_helpers::dummy_subject("B", 1, 4),
    ];

    let (msg, mut entry) = sentinel_warning();
    let listed = name_nonfinite_subjects(
        msg,
        &mut entry,
        &one_bad,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &source(EstimationMethod::FoceI, f64::NAN),
        &no_terms,
    );
    assert!(
        listed.ends_with(
            "B (contribution 2e20, every observation's likelihood term usable on its own; \
             every IPRED and PRED finite). The parameter-prior \
             penalty, a population-level term no subject carries, is unusable as well (NaN)."
        ),
        "{listed}"
    );
    assert_eq!(entry.details.as_ref().expect("payload")["ofv_prior"], "NaN");

    let (msg, mut entry) = sentinel_warning();
    let nobody = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &healthy,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &source(EstimationMethod::Laplace, f64::INFINITY),
        &no_terms,
    );
    assert_eq!(
        nobody.strip_prefix(&msg).expect("kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, and the parameter-prior penalty, a population-level term no \
         subject carries, is not (inf)."
    );

    let large: Vec<SubjectResult> = ["A", "B"]
        .iter()
        .map(|id| {
            let mut s = crate::types::test_helpers::dummy_subject(id, 1, 4);
            s.ofv_contribution = 6e13;
            s
        })
        .collect();
    let (msg, mut entry) = sentinel_warning();
    let sum = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &large,
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &source(EstimationMethod::FoceI, 5.0),
        &no_terms,
    );
    assert_eq!(
        sum.strip_prefix(&msg).expect("kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, and so is the parameter-prior penalty (5.0), but their sum \
         (120000000000005.0) is past the cutoff. The objective fails only as a sum; largest \
         first, they are A (60000000000000.0), B (60000000000000.0)."
    );
}

/// **T6 — a term is judged on the contribution's OFV scale.** Its `nll` is raw and the
/// contribution carries `2·nll`, so a raw `6e13` term is a `1.2e14` contribution and is
/// named, while a raw `4e13` one (`8e13`) is not; both are shown raw.
///
/// Mutation (run): compare the raw `nll` with the cutoff → the `6e13` term drops out
/// of the sentence and of `failed_terms`.
#[test]
fn a_term_is_judged_on_the_ofv_scale_and_shown_raw() {
    let pop = population_of(vec![shifted_subject("A")]);
    let mut a = crate::types::test_helpers::dummy_subject("A", 1, 0);
    a.ofv_contribution = 2e14;
    let scorer = |_: usize, _: &Subject, _: &SubjectResult| {
        vec![
            term("time-to-event on CMT 2", 6e13),
            term("categorical", 4e13),
        ]
    };
    let (msg, mut entry) = sentinel_warning();
    let full = name_nonfinite_subjects(
        msg,
        &mut entry,
        &[a],
        &pop,
        &one_cpt_model(),
        &one_cpt_model().default_params,
        None,
        &foce_source(),
        &scorer,
    );
    assert!(
        full.ends_with(
            "A (contribution 200000000000000.0, no Gaussian observation, failed term: \
             time-to-event on CMT 2 = 60000000000000.0)."
        ),
        "{full}"
    );
    assert_eq!(
        entry.details.as_ref().expect("payload")["unusable_subjects"][0]["failed_terms"],
        serde_json::json!([{"term": "time-to-event on CMT 2", "nll": 6e13}])
    );
}

/// A one-compartment IV bolus with proportional error, one η on `CL`, optionally under
/// `bloq_method = m3`.
fn proportional_iv(m3: bool) -> CompiledModel {
    let fit_options = if m3 {
        "[fit_options]\n  bloq_method = m3\n"
    } else {
        ""
    };
    parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.1 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
{fit_options}"#
    ))
    .expect("parse")
}

/// **T7 — the reviewer's case, end to end: a Gaussian residual term at the variance
/// floor** (#2267 review r3). Subject 3's dose is at `TIME 30`, after all four of its
/// observations, so every prediction is a finite `0`. A proportional error then scores
/// each observation at the variance floor, about `DV² × 1e12`, and the contribution is
/// past the cutoff with nothing non-finite anywhere. Before the Gaussian split the
/// sentence ended at "every IPRED and PRED finite" and named no term. Now it names the
/// residual term, and only it: the prior and curvature are usable, which is what makes
/// the attribution mean something. FOCE names its own Sheiner–Beal part, the
/// `linearized residual term` (#2267 review r4), its residual at `PRED`, also `0`. The
/// control (dose at 0 for everyone) stays silent.
///
/// Mutation (run): drop the Gaussian parts from `contribution_terms` → `failed_terms` is
/// empty and the sentence ends at "every IPRED and PRED finite", the review's report.
#[test]
fn a_finite_zero_prediction_names_the_gaussian_residual_term() {
    let model = proportional_iv(false);
    let pop = |late: bool| {
        population_of(vec![
            subject_with_dose_time("1", 0.0, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
            subject_with_dose_time("3", if late { 30.0 } else { 0.0 }, 0.9),
        ])
    };
    for (method, residual) in [
        (EstimationMethod::FoceI, "Gaussian residual term"),
        (EstimationMethod::Foce, "linearized residual term"),
    ] {
        let opts = FitOptions {
            method,
            outer_maxiter: 0,
            run_covariance_step: false,
            threads: Some(1),
            ..Default::default()
        };
        let control = fit(&model, &pop(false), &model.default_params, &opts).expect("fit");
        assert!(
            !control
                .warnings
                .iter()
                .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
            "{method:?}: {:?}",
            control.warnings
        );

        let result = fit(&model, &pop(true), &model.default_params, &opts).expect("fit");
        let (message, subjects) = named_subjects(&result);
        assert_eq!(subjects.len(), 1, "{method:?}: {subjects:?}");
        assert_eq!(subjects[0]["id"], "3");
        let failed = subjects[0]["failed_terms"]
            .as_array()
            .expect("failed_terms");
        assert_eq!(
            failed
                .iter()
                .map(|t| t["term"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec![residual],
            "{method:?}: only the residual term fails: {message}"
        );
        let nll = failed[0]["nll"].as_f64().expect("a finite residual term");
        assert!(
            nll >= 0.5 * DIVERGENCE_OFV,
            "{method:?}: past the cutoff on the 2·nll scale: {nll}"
        );
        assert!(
            message.contains(&format!(
                "3 (contribution {:?}, an unusable likelihood term on 2 of 4 observations, the \
                 first at TIME 0.5 (DV 30, IPRED 0); every IPRED and PRED finite, failed term: \
                 {residual} = {nll:?})",
                result.subjects[2].ofv_contribution
            )),
            "{method:?}: {message}"
        );
    }
}

/// **T8 — each branch records the parts of its own contribution** (#2267 review r3,
/// r4). The parts come from the marginal `compute_subject_results` scores, recorded on
/// the way, so on a healthy subject `2·Σ parts` reproduces `foce_subject_nll_cov`'s
/// contribution on every branch it takes:
///
/// * FOCEI, row-wise, with an M3-censored row so the censored term is live;
/// * FOCE, the Sheiner–Beal marginal: linearized residual and covariance parts;
/// * FOCEI with a compound-symmetry serial correlation, the dense-`R` marginal.
///
/// Each branch is straddled against the form a single splitter would have used: the
/// FOCEI row-wise parts miss the FOCE contribution by `0.90`, and the independent-row
/// FOCEI contribution misses the dense one by `0.030` — both far past the bound — so a
/// split of the wrong branch fails the sum. Then each failure a branch can name is fed
/// in one at a time: a runaway `η̂` fails the prior, a non-finite `∂f/∂η` fails FOCEI's
/// curvature and FOCE's `R̃`, and a correlation of `1.5` fails the dense `R` alone.
///
/// Measured: realised gaps `6.3e-16` (row-wise), `1.1e-16` (FOCE), `1.8e-15` (dense),
/// against a bound of `1e-12·max(|c|, 1)`, at least 500× headroom. The censored term is `0.0017`, so dropping it moves
/// the sum by `3.5e-3`.
///
/// Mutation (run): drop the `0.5·` on the dense residual part → the dense sum is off;
/// drop FOCE's censored part → its name list fires; name the dense `R` failure as the
/// curvature → the last assertion fires. The r3 splitter (row-wise FOCEI parts for
/// every branch) is what the two straddles measure.
#[test]
fn each_branch_records_the_parts_of_its_own_contribution() {
    use crate::stats::residual_error::{ResidualCov, SerialCov};
    use crate::types::ResidualSerialKind;
    let model = proportional_iv(true);
    // Data the model fits (`C = 10·exp(−0.1·t)` at the initial estimates: 9.5, 8.2, 4.5,
    // 0.9), the last row censored at an LLOQ of 1.0, close enough above its prediction that
    // its `−log Φ` is not rounded to 0 (at 1.5 it was, and the sum could not see the
    // censored term). A fixture the model cannot fit makes `H̃` itself non-PD through the
    // censored row, and the contribution is then the sentinel this test needs to be absent.
    let mut censored = subject_with_dose_time("1", 0.0, 1.0);
    censored.observations = vec![9.0, 8.5, 4.2, 1.0];
    censored.cens = vec![0, 0, 0, 1];
    // The dense branch takes uncensored subjects only (a censored row is scored row-wise),
    // in one residual block.
    let mut quantified = censored.clone();
    quantified.observations[3] = 0.95;
    quantified.cens = vec![0; 4];
    quantified.obs_rgroup = vec![1; 4];
    let params = model.default_params.clone();
    let independent = model.residual_cov_at(&params);
    let serial = |rho: f64| ResidualCov {
        pairs: &[],
        serial: Some(SerialCov {
            kind: ResidualSerialKind::Cs,
            rho,
        }),
    };
    let ebe = |subject: &Subject| {
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            ..Default::default()
        };
        crate::estimation::inner_optimizer::find_ebe_for_stage(
            &model, subject, &params, 200, 1e-8, None, None, 0, &opts,
        )
    };
    let parts = |subject: &Subject,
                 eta: &DVector<f64>,
                 h: &DMatrix<f64>,
                 cov: ResidualCov<'_>,
                 interaction: bool| {
        crate::stats::likelihood::foce_subject_contribution_parts(
            &model,
            subject,
            &params.theta,
            eta,
            h,
            &params.omega,
            &params.sigma.values,
            cov,
            interaction,
        )
    };
    let contribution = |subject: &Subject,
                        eta: &DVector<f64>,
                        h: &DMatrix<f64>,
                        cov: ResidualCov<'_>,
                        interaction: bool| {
        2.0 * crate::stats::likelihood::foce_subject_nll_cov(
            &model,
            subject,
            &params.theta,
            eta,
            h,
            &params.omega,
            &params.sigma.values,
            cov,
            interaction,
        )
    };
    let twice_sum = |list: &[(String, f64)]| list.iter().map(|(_, v)| 2.0 * v).sum::<f64>();
    let failing = |list: &[(String, f64)]| -> Vec<String> {
        list.iter()
            .filter(|(_, v)| !crate::estimation::outer_optimizer::ofv_is_valid(2.0 * v))
            .map(|(n, _)| n.clone())
            .collect()
    };

    let c_ebe = ebe(&censored);
    let q_ebe = ebe(&quantified);
    let branches = [
        (
            "FOCEI row-wise",
            &censored,
            &c_ebe,
            independent,
            true,
            vec![
                "Gaussian residual term",
                "censored (M3) term",
                "random-effect prior",
                "curvature log-determinant",
            ],
        ),
        (
            "FOCE",
            &censored,
            &c_ebe,
            independent,
            false,
            vec![
                "linearized residual term",
                "censored (M3) term",
                "linearized covariance log-determinant",
            ],
        ),
        (
            "FOCEI dense",
            &quantified,
            &q_ebe,
            serial(0.5),
            true,
            vec![
                "Gaussian residual term",
                "random-effect prior",
                "curvature log-determinant",
            ],
        ),
    ];
    for (label, subject, e, cov, interaction, want) in branches {
        let list = parts(subject, &e.eta, &e.h_matrix, cov, interaction);
        let names: Vec<&str> = list.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, want, "{label}");
        let c = contribution(subject, &e.eta, &e.h_matrix, cov, interaction);
        assert!(
            list.iter().all(|(_, v)| v.is_finite()) && c.is_finite() && failing(&list).is_empty(),
            "{label}: {list:?} vs {c}"
        );
        let gap = (twice_sum(&list) - c).abs();
        eprintln!("{label}: contribution {c}, gap {gap:e}");
        assert!(
            gap <= 1e-12 * c.abs().max(1.0),
            "{label}: 2·Σ parts = {}, contribution = {c}: {list:?}",
            twice_sum(&list)
        );
    }

    // The straddles: the r3 splitter's row-wise FOCEI parts are not the FOCE contribution,
    // and the independent-row FOCEI marginal is not the dense one.
    let foce = contribution(&censored, &c_ebe.eta, &c_ebe.h_matrix, independent, false);
    let rowwise = twice_sum(&parts(
        &censored,
        &c_ebe.eta,
        &c_ebe.h_matrix,
        independent,
        true,
    ));
    let dense = contribution(&quantified, &q_ebe.eta, &q_ebe.h_matrix, serial(0.5), true);
    let undense = contribution(&quantified, &q_ebe.eta, &q_ebe.h_matrix, independent, true);
    eprintln!(
        "FOCE {foce} vs row-wise FOCEI parts {rowwise}; dense {dense} vs independent {undense}"
    );
    assert!((foce - rowwise).abs() > 1e-3, "{foce} vs {rowwise}");
    assert!((dense - undense).abs() > 1e-3, "{dense} vs {undense}");

    // η̂ = 1e8 against ω² = 0.04: ½η²/ω² ≈ 1.25e17. The prediction is `0` there (CL is
    // `inf`), so the residual fails too. The prior is asserted to be named, and its value.
    let runaway = DVector::from_vec(vec![1e8]);
    let prior = parts(&censored, &runaway, &c_ebe.h_matrix, independent, true);
    let (_, prior_nll) = prior
        .iter()
        .find(|(n, _)| n == "random-effect prior")
        .expect("prior part");
    assert!(
        *prior_nll > 1e17 && failing(&prior).contains(&"random-effect prior".to_string()),
        "{prior:?}"
    );

    let mut bad_h = c_ebe.h_matrix.clone();
    bad_h[(0, 0)] = f64::NAN;
    assert_eq!(
        failing(&parts(&censored, &c_ebe.eta, &bad_h, independent, true)),
        vec!["curvature (H̃ not positive definite)".to_string()],
        "a non-finite ∂f/∂η fails FOCEI's curvature and nothing else"
    );
    assert_eq!(
        failing(&parts(&censored, &c_ebe.eta, &bad_h, independent, false)),
        vec!["linearized covariance R̃ (not positive definite)".to_string()],
        "and FOCE's R̃ and nothing else"
    );
    // Compound symmetry at ρ = 1.5 over four rows has eigenvalue 1 − ρ < 0: the dense `R`
    // does not factor, while every row's own variance is usable.
    assert_eq!(
        failing(&parts(
            &quantified,
            &q_ebe.eta,
            &q_ebe.h_matrix,
            serial(1.5),
            true
        )),
        vec!["Gaussian residual covariance R (not positive definite)".to_string()],
    );
}

/// A two-occasion subject (observations at 1–3 and 4–6, one dose per occasion) with a
/// once-a-day kappa on `CL`, proportional error.
fn proportional_iov() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(2.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#,
    )
    .expect("parse")
}

/// Subject `id` of [`proportional_iov`], dosed at `dose_times` (one per occasion).
fn iov_subject(id: &str, dose_times: [f64; 2], scale: f64) -> Subject {
    let mut s = subject_with_dose_time(id, dose_times[0], scale);
    s.doses = dose_times
        .iter()
        .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    s.obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    s.observations = [36.0, 28.0, 21.0, 34.0, 26.0, 19.0]
        .iter()
        .map(|y| scale * y)
        .collect();
    s.obs_cmts = vec![1; 6];
    s.cens = vec![0; 6];
    s.occasions = vec![1, 1, 1, 2, 2, 2];
    s.dose_occasions = vec![1, 2];
    s
}

/// **T9 — an IOV subject names the part of its stacked marginal that failed** (#2267
/// review r6). Before, an IOV subject got no split at all: the reviewer's
/// finite-zero-prediction case on a subject scored with κ read "every IPRED and PRED
/// finite" and named no term. Subject 3's doses are after all its observations, so every
/// prediction is a finite `0` and its proportional residual is scored at the variance
/// floor; the term named is each method's own residual part, and only it. The control
/// (doses at 0 and 3.5 for everyone) stays silent.
///
/// Mutation (run): send an IOV subject to `non_gaussian_terms` as before → `failed_terms`
/// is empty and the sentence ends at "every IPRED and PRED finite".
#[test]
fn an_iov_subject_names_the_part_of_its_stacked_marginal_that_failed() {
    let model = proportional_iov();
    let pop = |late: bool| {
        population_of(vec![
            iov_subject("1", [0.0, 3.5], 1.0),
            iov_subject("2", [0.0, 3.5], 1.1),
            iov_subject("3", if late { [30.0, 31.0] } else { [0.0, 3.5] }, 0.9),
        ])
    };
    for (method, residual) in [
        (EstimationMethod::FoceI, "Gaussian residual term"),
        (EstimationMethod::Foce, "linearized residual term"),
    ] {
        let opts = FitOptions {
            method,
            outer_maxiter: 0,
            run_covariance_step: false,
            threads: Some(1),
            ..Default::default()
        };
        let control = fit(&model, &pop(false), &model.default_params, &opts).expect("fit");
        assert!(
            !control
                .warnings
                .iter()
                .any(|w| w.contains(NONFINITE_OBJECTIVE_TOKEN)),
            "{method:?}: {:?}",
            control.warnings
        );

        let result = fit(&model, &pop(true), &model.default_params, &opts).expect("fit");
        // The κ path is what the mutation below exercises: an IOV-routed subject.
        assert_eq!(model.n_kappa, 1, "premise: an IOV model");
        let (message, subjects) = named_subjects(&result);
        assert_eq!(subjects.len(), 1, "{method:?}: {subjects:?}");
        assert_eq!(subjects[0]["id"], "3");
        let failed: Vec<&str> = subjects[0]["failed_terms"]
            .as_array()
            .expect("failed_terms")
            .iter()
            .map(|t| t["term"].as_str().unwrap())
            .collect();
        assert_eq!(failed, vec![residual], "{method:?}: {message}");
        assert!(
            message.contains(&format!(
                "every IPRED and PRED finite, failed term: {residual} = "
            )),
            "{method:?}: {message}"
        );
    }
}

/// **T10 — an IOV subject's parts are its stacked marginal's own.** On a healthy
/// two-occasion subject, `2·Σ parts` reproduces `foce_subject_nll_iov_linearized` under
/// FOCEI and FOCE, and a non-finite κ column of `H` fails FOCEI's curvature and FOCE's
/// `R̃` alone. The straddle: the same subject split without κ (the η-only marginal)
/// sums to a different number, so these parts are the stacked system's.
///
/// Measured: gaps `0` under both flags, against a bound of `1e-12·max(|c|, 1)`; the
/// η-only sums differ from the stacked contribution by `210` (FOCEI) and `361` (FOCE),
/// against the straddle's `0.1`. (The point is not an EBE and the data not this model's
/// fit, so the contributions are large; the identity holds at any point.)
///
/// Mutation (run): pass `None` for the sink in the IOV FOCE branch → its name list is
/// empty and fires (T9 fails too).
#[test]
fn an_iov_subjects_parts_sum_to_its_stacked_contribution() {
    let model = proportional_iov();
    let subject = iov_subject("1", [0.0, 3.5], 1.0);
    let p = &model.default_params;
    let omega_iov = p.omega_iov.as_ref().expect("Ω_IOV");
    let eta = DVector::from_vec(vec![0.05]);
    let kappas = vec![DVector::from_vec(vec![0.1]), DVector::from_vec(vec![-0.1])];
    // `∂f/∂η_CL` at this point, from the closed form `f = (100/V)·exp(−CL/V·t)`'s
    // shape; the identity holds for any `H`, so it only has to be finite and non-zero.
    let h = DMatrix::from_fn(6, 1, |j, _| -0.04 * subject.obs_times[j]);
    let lin = crate::stats::likelihood::iov_linearization(
        &model, &subject, &p.theta, &eta, &h, &p.omega, &kappas, omega_iov,
    )
    .expect("one κ per occasion");
    let failing = |list: &[(String, f64)]| -> Vec<String> {
        list.iter()
            .filter(|(_, v)| !crate::estimation::outer_optimizer::ofv_is_valid(2.0 * v))
            .map(|(n, _)| n.clone())
            .collect()
    };
    for (interaction, names, broken) in [
        (
            true,
            vec![
                "Gaussian residual term",
                "random-effect prior",
                "curvature log-determinant",
            ],
            "curvature (H̃ not positive definite)",
        ),
        (
            false,
            vec![
                "linearized residual term",
                "linearized covariance log-determinant",
            ],
            "linearized covariance R̃ (not positive definite)",
        ),
    ] {
        let parts = crate::stats::likelihood::foce_subject_iov_contribution_parts(
            &model,
            &subject,
            &p.theta,
            &p.sigma.values,
            model.residual_cov_at(p),
            interaction,
            &lin,
        );
        assert_eq!(
            parts.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            names,
            "interaction {interaction}"
        );
        let c = 2.0
            * crate::stats::likelihood::foce_subject_nll_iov_linearized(
                &model,
                &subject,
                &p.theta,
                &p.sigma.values,
                model.residual_cov_at(p),
                interaction,
                &lin,
            );
        let summed: f64 = parts.iter().map(|(_, v)| 2.0 * v).sum();
        assert!(
            c.is_finite() && failing(&parts).is_empty(),
            "interaction {interaction}: {parts:?} vs {c}"
        );
        eprintln!(
            "interaction {interaction}: contribution {c}, gap {:e}",
            (summed - c).abs()
        );
        assert!(
            (summed - c).abs() <= 1e-12 * c.abs().max(1.0),
            "interaction {interaction}: 2·Σ parts = {summed}, contribution = {c}"
        );

        let eta_only: f64 = crate::stats::likelihood::foce_subject_contribution_parts(
            &model,
            &subject,
            &p.theta,
            &eta,
            &h,
            &p.omega,
            &p.sigma.values,
            model.residual_cov_at(p),
            interaction,
        )
        .iter()
        .map(|(_, v)| 2.0 * v)
        .sum();
        eprintln!("interaction {interaction}: η-only sum {eta_only}");
        assert!(
            (eta_only - c).abs() > 0.1,
            "interaction {interaction}: the η-only split {eta_only} vs stacked {c}"
        );

        let mut bad = crate::stats::likelihood::iov_linearization(
            &model, &subject, &p.theta, &eta, &h, &p.omega, &kappas, omega_iov,
        )
        .expect("one κ per occasion");
        bad.h_full[(0, 1)] = f64::NAN;
        let parts = crate::stats::likelihood::foce_subject_iov_contribution_parts(
            &model,
            &subject,
            &p.theta,
            &p.sigma.values,
            model.residual_cov_at(p),
            interaction,
            &bad,
        );
        assert_eq!(
            failing(&parts),
            vec![broken.to_string()],
            "interaction {interaction}: a non-finite κ column fails only {broken}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  L — the label is read from the subjects, and a sum names its largest (#2265)
// ─────────────────────────────────────────────────────────────────────────────

/// The `W_NONFINITE_OBJECTIVE` pair `fit()` builds for `ofv` under `method`.
fn warning_at(ofv: f64, method: EstimationMethod) -> (String, crate::types::WarningEntry) {
    let mut converged = true;
    crate::api::nonfinite_objective_warning(&mut converged, ofv, method, &FitOptions::default())
        .expect("an unusable objective is demoted")
}

/// **L1 — the label comes from the subjects when they are the objective (#2265).**
/// The gate draws "at the divergence sentinel" against "too large to be a fit" on the
/// total at `1e20`, which misreads an unclamped subject past `1e20` and the IOV + FREM
/// refusal at `1e18`. `fit()` has the subjects, so it rereads the label from them.
///
/// The input space is the subjects' evidence (a contribution at a rail, a term at one,
/// a non-finite contribution, or none of these) × whether the reread applies (the
/// objective is finite, the contributions are its terms, their sum is past the cutoff
/// and adds up to it). One row per cell that changes the label or `reason_read_from`:
///
/// | row | total reads | subjects read | applies | label |
/// |---|---|---|---|---|
/// | unclamped `1.44e20` | sentinel | too large | yes | too large |
/// | refusal `2e18` | too large | sentinel | yes | sentinel |
/// | hazard term at `1e20` in `2.5e20` | sentinel | sentinel | yes | sentinel |
/// | `NaN` contribution, objective `2e20` | sentinel | sentinel | yes | sentinel |
/// | `1.44e20` under a `2e20` objective | sentinel | — | no: another point | sentinel |
/// | `1.44e20` under `1.44e20·(1 + 1e-10)` | sentinel | too large | yes: rounding | too large |
/// | `1.99e20` under a `2e20` objective | sentinel | — | no: 0.5% out | sentinel |
/// | `NaN` objective | `NaN` | — | no: not finite | `NaN` |
/// | `1.44e20` under Laplace | sentinel | — | no: not its terms | sentinel |
/// | `9.99e13` under `1e14` | too large | — | no: sum usable | too large |
///
/// The first and the third straddle the term check: the same kind of finite subject
/// past `1e20`, told apart only by the hazard term at the rail. The first and the fifth
/// differ only in the gated objective. The sixth and the seventh bracket
/// [`crate::api::postfit::REREAD_AGREEMENT`] (`1e-9`) a decade or more on each side: a
/// gap at the summation-rounding scale rereads, and one at 0.5% does not.
///
/// Mutation (run):
/// * drop the rail check on the contribution → the refusal row reads "too large";
/// * drop the term check → the hazard row reads "too large";
/// * drop the non-finite clause → the `NaN` contribution row reads "too large";
/// * drop the agreement check → the "another point" and "0.5% out" rows read "too large";
/// * `REREAD_AGREEMENT = 1e-2` (the first version) → the "0.5% out" row reads "too large";
///   `1e-11` → the "rounding" row reads `total`;
/// * drop `source.objective.is_none()` → the Laplace row reads "too large";
/// * drop `!ofv_is_valid(total)` → the last row reads `subjects`;
/// * rebuild the message without the reread → every relabelled row's prefix fires;
/// * keep `details["reason"]` from the total → the first two rows' `reason` fires.
#[test]
fn the_label_is_read_from_the_subjects_when_they_are_the_objective() {
    let m = one_cpt_model();
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let hazard = |_: usize, s: &Subject, _: &SubjectResult| {
        if s.id == "A" {
            vec![
                term("time-to-event on CMT 2", 1e20),
                term("Gaussian residual", 2.5e19),
            ]
        } else {
            Vec::new()
        }
    };
    let sentinel = "at the divergence sentinel";
    let large = "too large to be a fit";
    let foce = EstimationMethod::FoceI;
    type Row<'a> = (
        &'a str,
        f64,
        f64,
        EstimationMethod,
        &'a crate::api::postfit::TermScorer<'a>,
        &'a str,
        &'a str,
    );
    let rows: [Row<'_>; 10] = [
        (
            "unclamped",
            1.44e20,
            1.44e20 + 12.34,
            foce,
            &no_terms,
            large,
            "subjects",
        ),
        (
            "refusal",
            2e18,
            2e18 + 12.34,
            foce,
            &no_terms,
            sentinel,
            "subjects",
        ),
        (
            "hazard term",
            2.5e20,
            2.5e20 + 12.34,
            foce,
            &hazard,
            sentinel,
            "subjects",
        ),
        (
            "NaN contribution",
            f64::NAN,
            2e20 + 12.34,
            foce,
            &no_terms,
            sentinel,
            "subjects",
        ),
        (
            "another point",
            1.44e20,
            2e20,
            foce,
            &no_terms,
            sentinel,
            "total",
        ),
        (
            "rounding",
            1.44e20,
            1.44e20 * (1.0 + 1e-10),
            foce,
            &no_terms,
            large,
            "subjects",
        ),
        (
            "0.5% out", 1.99e20, 2e20, foce, &no_terms, sentinel, "total",
        ),
        (
            "NaN objective",
            f64::NAN,
            f64::NAN,
            foce,
            &no_terms,
            "NaN",
            "total",
        ),
        (
            "not its terms",
            1.44e20,
            1.44e20 + 12.34,
            EstimationMethod::Laplace,
            &no_terms,
            sentinel,
            "total",
        ),
        (
            "sum usable",
            4.995e13,
            1e14,
            foce,
            &no_terms,
            large,
            "total",
        ),
    ];
    for (row, a_contribution, ofv, method, scorer, reason, read_from) in rows {
        let mut a = crate::types::test_helpers::dummy_subject("A", 1, 4);
        a.ofv_contribution = a_contribution;
        let mut b = crate::types::test_helpers::dummy_subject("B", 1, 4);
        if row == "sum usable" {
            b.ofv_contribution = 4.995e13;
        }
        let (msg, mut entry) = warning_at(ofv, method);
        let source = ContributionSource::of(method, &FitOptions::default(), false, 0.0);
        let out = name_nonfinite_subjects(
            msg,
            &mut entry,
            &[a, b],
            &pop,
            &m,
            &m.default_params,
            None,
            &source,
            scorer,
        );
        let want = crate::estimation::outer_optimizer::nonfinite_objective_message(ofv, reason);
        assert!(
            out.starts_with(&want),
            "{row}: the gate's sentence under the label {reason:?}: {out}"
        );
        assert_eq!(entry.message, out, "{row}");
        let details = entry.details.as_ref().expect("payload");
        assert_eq!(details["reason"], reason, "{row}: {details}");
        assert_eq!(details["reason_read_from"], read_from, "{row}: {details}");
    }
}

/// **L1b — the reread end to end, on #2265's case.** L1 hands `name_nonfinite_subjects`
/// a built `WarningEntry`; this runs `fit()`, so the gated `details["ofv"]` round trip,
/// `ContributionSource::of` and the prior fold are the production ones. Subject A's
/// first observation is at `TIME 0.5`, before its dose at `1.0`, so the closed form
/// predicts exactly `0` there whatever the parameters; the proportional variance is
/// `0`, guarded to `1e-12`, and its `DV` of 12000 scores `12000² / 1e-12 = 1.44e20`.
/// That is past `1e20` and finite, so the estimator's total reads "at the divergence
/// sentinel" and only the reread can say "too large to be a fit". Both halves are
/// asserted, so the fixture keeps straddling the `1e20` line it exists to test.
///
/// It also measures the gap [`crate::api::postfit::REREAD_AGREEMENT`] bounds,
/// `|clamped_total − ofv| / ofv` at the reread, printed under `--nocapture`. Four
/// cells: FOCEI and FOCE, each at #2265's `1.44e20` and, under a prior on `TVCL`, at
/// `DV = 12.25`, `1.5e14`, just past the cutoff. There one ulp is `0.03`, so the prior
/// (`72.4` / `2.83`) shows in the sum, and the test asserts that it does. Measured on
/// macOS arm64, the gap is exactly `0` in all four. The 12.25 cells read "too large" on
/// the total too, so they pin that the reread applies, not that it relabels.
///
/// Mutation (run): make the reread filter reject everything (`.filter(|_| false)`) →
/// `reason` reads the total's "at the divergence sentinel" in the `1.44e20` cells, and
/// `reason_read_from` reads `total` in all four. Dropping `+ source.prior` from the
/// production `clamped_total` is not caught, and cannot be at this bound: a prior of
/// ordinary size is below `1e-9` of any objective past the cutoff (here `4.8e-13`), so
/// the fold moves the reread only for a prior past `1e5`. What this test pins about the
/// prior is the measurement: the gap is `0` with the prior in the sum.
#[test]
fn a_fit_rereads_an_unclamped_subject_past_1e20_as_too_large() {
    use crate::estimation::outer_optimizer::{
        nonfinite_objective_message, REASON_AT_SENTINEL, REASON_TOO_LARGE,
    };
    use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
    let priored = || {
        parse_model_string(
            r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)  prior(1.5, rse = 25%)
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
    };
    let plain: &dyn Fn() -> CompiledModel = &one_cpt_model;
    let priored: &dyn Fn() -> CompiledModel = &priored;
    for (case, model, method, dv, total_reads) in [
        (
            "#2265, FOCEI",
            plain,
            EstimationMethod::FoceI,
            12000.0,
            REASON_AT_SENTINEL,
        ),
        (
            "#2265, FOCE",
            plain,
            EstimationMethod::Foce,
            12000.0,
            REASON_AT_SENTINEL,
        ),
        (
            "past the cutoff, priored, FOCEI",
            priored,
            EstimationMethod::FoceI,
            12.25,
            REASON_TOO_LARGE,
        ),
        (
            "past the cutoff, priored, FOCE",
            priored,
            EstimationMethod::Foce,
            12.25,
            REASON_TOO_LARGE,
        ),
    ] {
        let model = model();
        let mut a = subject_with_dose_time("A", 1.0, 1.0);
        a.observations[0] = dv;
        let pop = population_of(vec![a, subject_with_dose_time("B", 0.0, 1.0)]);
        let opts = FitOptions {
            method,
            outer_maxiter: 3,
            ..one_iteration_opts()
        };
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");

        let a = &result.subjects[0];
        assert_eq!(
            a.ipred[0], 0.0,
            "{case}: the pre-dose prediction is exactly 0"
        );
        assert!(
            a.ofv_contribution.is_finite() && a.ofv_contribution != 2.0 * SUBJECT_SENTINEL_NLL,
            "{case}: A is large on its own terms, not clamped: {}",
            a.ofv_contribution
        );
        assert_eq!(
            nonfinite_objective_reason(result.ofv),
            total_reads,
            "{case}: {}",
            result.ofv
        );

        let entry = result
            .warnings_structured
            .iter()
            // `fit()`'s copy, the one with the payload: under a prior the estimator's
            // sentence quotes `ofv_data`, so the de-duplication keeps it too.
            .find(|w| w.message.contains(NONFINITE_OBJECTIVE_TOKEN) && w.details.is_some())
            .unwrap_or_else(|| {
                panic!(
                    "{case}: no {NONFINITE_OBJECTIVE_TOKEN}: {:?}",
                    result.warnings
                )
            });
        let details = entry.details.as_ref().expect("payload");
        let ofv = details["ofv"]
            .as_f64()
            .expect("a finite objective is a number");
        let prior = details["ofv_prior"].as_f64().expect("the prior");
        let clamped_total = result
            .subjects
            .iter()
            .map(|s| s.ofv_contribution)
            .sum::<f64>()
            + prior;
        assert!(clamped_total.is_finite(), "{case}: {clamped_total}");
        let gap = (clamped_total - ofv).abs() / ofv.abs();
        let ulp = f64::from_bits(ofv.to_bits() + 1) - ofv;
        eprintln!(
            "L1b {case}: ofv = {ofv:e}, subjects + prior = {clamped_total:e}, \
             prior = {prior:e}, ulp = {ulp:e}, |clamped_total − ofv| / ofv = {gap:e}"
        );
        // The priored cells are where the fold is visible: a prior below one ulp of
        // the objective would agree with or without it.
        assert_eq!(
            prior.abs() > ulp,
            case.contains("priored"),
            "{case}: prior {prior:e}, ulp {ulp:e}"
        );
        assert!(
            gap <= crate::api::postfit::REREAD_AGREEMENT,
            "{case}: the subjects and the prior are the objective, to rounding: {gap:e}"
        );

        assert_eq!(details["reason"], REASON_TOO_LARGE, "{case}: {details}");
        assert_eq!(details["reason_read_from"], "subjects", "{case}: {details}");
        assert!(
            entry
                .message
                .starts_with(&nonfinite_objective_message(ofv, REASON_TOO_LARGE)),
            "{case}: the gate's sentence carries the reread label: {}",
            entry.message
        );
    }
}

/// **L2 — a sum past the cutoff names its largest contributions (#2265).** Ten
/// subjects of `1.6e13` each sum past `1e14` with none unusable on its own, and before
/// this the warning named nobody. Twelve subjects here, in ascending order of
/// contribution so the sort is visible, two more than the message names, so the
/// "of N" count and the share are live.
///
/// Mutation (run): drop the sort → `S01` leads; drop the truncation → twelve listed;
/// drop the share or the count, or the "fails only as a sum" clause → the sentence
/// comparison fires; leave `largest_subjects` out of the details → its assertion fires.
#[test]
fn a_sum_past_the_cutoff_names_its_largest_contributions() {
    let m = one_cpt_model();
    let subjects: Vec<SubjectResult> = (1..=12)
        .map(|i| {
            let mut s = crate::types::test_helpers::dummy_subject(&format!("S{i:02}"), 1, 4);
            s.ofv_contribution = i as f64 * 2e12;
            s
        })
        .collect();
    // 2e12 · (1 + … + 12) = 1.56e14; the ten largest are 1.5e14 of it, 96%.
    let (msg, mut entry) = warning_at(1.56e14, EstimationMethod::FoceI);
    let out = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &subjects,
        &population_of(Vec::new()),
        &m,
        &m.default_params,
        None,
        &foce_source(),
        &no_terms,
    );
    assert_eq!(
        out.strip_prefix(&msg)
            .expect("the label agrees, so the sentence is kept"),
        " No subject can be named: every subject's post-hoc FOCEI contribution at these \
         estimates is usable, but their sum (156000000000000.0) is past the cutoff. The \
         objective fails only as a sum; the 10 largest of 12, together 96% of the sum of \
         the contributions, are S12 (24000000000000.0), S11 (22000000000000.0), \
         S10 (20000000000000.0), S09 (18000000000000.0), S08 (16000000000000.0), \
         S07 (14000000000000.0), S06 (12000000000000.0), S05 (10000000000000.0), \
         S04 (8000000000000.0), S03 (6000000000000.0)."
    );
    let details = entry.details.as_ref().expect("payload");
    let largest = details["largest_subjects"].as_array().expect("an array");
    assert_eq!(
        largest.len(),
        crate::api::postfit::NONFINITE_SUBJECTS_NAMED,
        "{details}"
    );
    assert_eq!(
        largest[0],
        serde_json::json!({"id": "S12", "ofv_contribution": 2.4e13})
    );
    assert_eq!(details["reason_read_from"], "subjects", "{details}");
}
