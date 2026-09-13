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
use crate::types::{ViFinalOfv, WarningCode, WarningSeverity};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
//  G1 — the reported regression, end to end
// ─────────────────────────────────────────────────────────────────────────────

/// The #1296 model: a rapid-equilibrium two-state ODE, benign at `KFAST = 1`.
fn two_state_model() -> CompiledModel {
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

fn subject_with_dose_time(id: &str, dose_time: f64, scale: f64) -> Subject {
    let obs_times = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times.iter().map(|t| scale * 50.0 / (1.0 + t)).collect();
    Subject {
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
    // Exactly one, not "at least one": the estimator's gate and fit()'s gate both
    // apply, and the shared gate's `!*converged` early return is what stops the
    // second from re-explaining a verdict the first already demoted. Without it a
    // user would read the same paragraph twice.
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
/// Mutation (run): weaken `ofv_is_valid` to `ofv.is_finite()` → the `2e20` and
/// `DIVERGENCE_OFV` cases fire.
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
        (DIVERGENCE_OFV, "at the divergence sentinel"),
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
        assert_eq!(nonfinite_objective_reason(ofv), what, "ofv = {ofv:?}");
    }
}

/// **G3 — the two ways this gate could fire when it must not.** A gate that
/// demoted everything would pass G1 and G2 and be worse than the defect.
///
/// * A real population objective — both signs, and the extremes of the valid range.
///   The negative extreme is deliberate: the cutoff is **one-sided**, because a
///   large negative −2 log L is legitimate (log-transformed DV on a large cohort),
///   and `ofv_is_valid_rejects_the_clamped_sentinel_not_just_non_finite` pins that
///   independently.
/// * An *already* failed verdict must come back with no message. Otherwise a fit
///   that failed for some other reason would grow a second, wrong explanation —
///   and, worse, the `!*converged` early return is what stops the MCEM gate from
///   reporting the shared reason for a #528 runaway.
#[test]
fn the_gate_leaves_a_real_objective_and_an_already_failed_verdict_alone() {
    for ofv in [-1e15, -286.0, 0.0, 12_345.678, DIVERGENCE_OFV - 1.0] {
        let mut converged = true;
        assert!(
            gate_converged_on_objective(&mut converged, ofv).is_none(),
            "ofv = {ofv:?} is a legitimate population objective and must not be demoted"
        );
        assert!(converged, "ofv = {ofv:?}");
    }
    // An optimizer that already said "no" is not re-explained.
    let mut converged = false;
    assert!(gate_converged_on_objective(&mut converged, f64::NAN).is_none());
    assert!(!converged);
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
