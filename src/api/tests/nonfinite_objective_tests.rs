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
/// `methods = vi, imp` with `imp_eval_only = true` is the **documented** way to
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
                "unusable_terms": 2, "first_unusable_term_time": 2.0}),
            serde_json::json!({"id": "C", "ofv_contribution": 4.6e14,
                "nonfinite_predictions": 0, "n_obs": 4, "first_time": null,
                "unusable_terms": 1, "first_unusable_term_time": 0.5}),
            serde_json::json!({"id": "D", "ofv_contribution": "NaN",
                "nonfinite_predictions": 0, "n_obs": 4, "first_time": null,
                "unusable_terms": 0, "first_unusable_term_time": null}),
        ],
        "JSON has no NaN, so a non-finite contribution is carried as a string"
    );
}

/// **N2 — nothing to name, nothing added.** A sentinel objective whose every subject
/// has a valid post-fit contribution (the objective was clamped at a trial point,
/// not at the estimates the subjects are scored at) keeps the gate's message and
/// payload exactly, rather than appending "0 of 2 subjects".
///
/// Mutation (run): drop the `named.is_empty()` return → the message grows a
/// "0 of 2 subjects" sentence and `details` an `unusable_subjects` key, and this fires.
#[test]
fn a_warning_with_no_unusable_subject_is_left_as_the_gate_wrote_it() {
    let m = one_cpt_model();
    let pop = population_of(vec![shifted_subject("A"), shifted_subject("B")]);
    let subjects = [
        crate::types::test_helpers::dummy_subject("A", 1, 4),
        crate::types::test_helpers::dummy_subject("B", 1, 4),
    ];
    let (msg, mut entry) = sentinel_warning();
    let before = entry.clone();
    let out = name_nonfinite_subjects(
        msg.clone(),
        &mut entry,
        &subjects,
        &pop,
        &m,
        &m.default_params,
        None,
    );
    assert_eq!(out, msg);
    assert_eq!(entry.message, before.message);
    assert_eq!(entry.details, before.details);
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
    let full = name_nonfinite_subjects(msg, &mut entry, &[a], &pop, &m, &m.default_params, None);
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
             11 observations, the first at TIME 0.5)."
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
                 of 11 observations, the first at TIME 4)."
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
        let full =
            name_nonfinite_subjects(msg, &mut entry, &[a], &pop, &m, &m.default_params, None);
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
/// it at the variance floor, `DV² × 1e12`. That record alone makes the contribution
/// unusable. Two `DV` levels, measured:
///
/// | `DV` | engine | η̂ | predictions | prediction clause |
/// |---|---|---|---|---|
/// | 200 | both | ≈ (−5.7, −3.0, 1.3) | all finite | "every IPRED and PRED finite" |
/// | 20000 | closed form | ≈ (−6.6, −726, −8.6) | `IPRED` `NaN` at `TIME` 0, 0 after | first at `TIME 0` |
/// | 20000 | ODE | ≈ (−3.8e10, −3.9e10, 8.0e10) | `IPRED` 0 at `TIME` 0, `NaN` after | first at `TIME 0.5` |
///
/// At 20000 the EBE runs away and the engines disagree on the predictions, which is
/// what #2266 reported: the prediction clause names `TIME 0.5` on the ODE and the
/// closed form, at 200, names no record at all. The likelihood-term clause must name
/// `TIME 0` in all four cells. The prediction clauses are asserted too, so the cells
/// keep straddling: if both engines stopped disagreeing, the fixture would no longer
/// test the property the issue is about.
///
/// Mutation (run): name the first non-finite prediction in the term clause →
/// the ODE at 20000 names `TIME 0.5` and both engines at 200 name nothing; flag a
/// term only when it is not finite → the ODE at 20000 and both at 200 fire.
#[test]
fn a_runaway_ebe_names_the_record_with_the_unusable_term_on_both_engines() {
    let expected_predictions = |dv: f64, ode: bool| match (dv == 200.0, ode) {
        (true, _) => "every IPRED and PRED finite",
        (false, false) => "the first at TIME 0)",
        (false, true) => {
            "a non-finite IPRED or PRED on 11 of 12 observations, the first at TIME 0.5)"
        }
    };
    for dv in [200.0, 20000.0] {
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
    name_nonfinite_subjects(msg, &mut entry, &[a], &pop, model, params, mixest)
}

const NAMES_FIRST: &str =
    "an unusable likelihood term on 1 of 4 observations, the first at TIME 0.5";
const NAMES_NONE: &str = "every observation's likelihood term usable on its own;";

/// **N1d — the term is the scorer's, not a floored copy** (#2266 review 2). FOCEI
/// scores a record at its raw `IPRED`: a proportional error (σ = 0.2) on `DV = 20`
/// at `IPRED = −1` has `V = 0.04` and a term of `21²/0.04 + ln 0.04 ≈ 1.1e4`. Flooring
/// the prediction to `1e-12` first, as the SAEM/IS objective does, scores it at
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
        );
        (c, got == msg, got)
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
