//! Tier-2 end-to-end checks for #1382: a fit reports which estimator produced its
//! standard errors, eigenvalues and condition number.
//!
//! **The gap.** `FitResult` carried `cov_condition_number` and `cov_eigenvalues`
//! and printed standard errors, but recorded nothing about which of `R⁻¹`, `S⁻¹`
//! or the `R⁻¹SR⁻¹` sandwich they came out of; `covariance_method` lived on
//! `FitOptions`, which a fit object, a `{model}-fit.yaml` or a `.fitrx` bundle
//! does not carry. The reporter measured the same fit at condition number
//! **1.42e8** under the sandwich and **3.68e5** under `covariance_method = s` —
//! both correct for their estimator, and indistinguishable in the output. That
//! matters the moment the figure is compared against NONMEM, whose `$COVARIANCE`
//! default is `RSR` and not `R`.
//!
//! **Tier 2, not Tier 1** (#1382 review): these call `fit()` / `run_covariance`
//! across the public API and return after a single outer iteration — CLAUDE.md's
//! Tier-2 shape. The rule itself
//! (`estimation::covariance::published_covariance_method`), the serde token, the
//! warning payload and the per-site structural scan stay at Tier 1 in
//! `src/api/tests/covariance_method_label_tests.rs`, which calls no `fit()`.
//! Both tiers run on every PR: CI's `Tests + coverage (core)` job is
//! `cargo llvm-cov --workspace --tests`, which builds and runs `tests/*.rs`.
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    fit, run_covariance, CompiledModel, CovarianceMethod, CovarianceStatus, DoseEvent,
    EstimationMethod, FitOptions, Population, Subject, WarningCode,
};
use std::collections::HashMap;

/// One-compartment IV closed form — no ODE solve, so a covariance step that has
/// to build a real FD Hessian stays fast.
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

fn subject(id: &str, scale: f64) -> Subject {
    let obs_times: Vec<f64> = vec![0.5, 2.0, 8.0, 24.0];
    let observations = obs_times
        .iter()
        .map(|t| scale * 10.0 * (-0.1 * t).exp())
        .collect();
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

/// Twelve subjects, not two: the cross-product `S = Σᵢ gᵢgᵢᵀ` has rank at most
/// the subject count, so a fixture with fewer subjects than free parameters makes
/// `covariance_method = s` fail as *rank-deficient* and publish no matrix at all.
/// The tests below would then be asserting `None == None` for a reason that has
/// nothing to do with the label.
fn population() -> Population {
    Population {
        subjects: (0..12)
            .map(|i| subject(&format!("{i}"), 1.0 + 0.05 * i as f64))
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn opts(method: CovarianceMethod, explicit: bool) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 1,
        run_covariance_step: true,
        covariance_method: method,
        covariance_method_set: explicit,
        threads: Some(1),
        ..Default::default()
    }
}

/// **G1 — the reported gap, end to end: the fit object names its estimator.**
///
/// Three arms, and the set is the test: a single arm is satisfied by a field
/// hard-coded to `Hessian`. `s` and `rsr` are the arms that cannot be faked, and
/// `s` is the one the reporter used to get 3.68e5 where the sandwich gave 1.42e8.
///
/// The premise — that the covariance step actually produced a matrix — is
/// asserted first, because `covariance_method` is `None` whenever it did not, and
/// a fixture that silently stopped producing one would make the real assertion
/// vacuous.
///
/// Mutation (run): hard-code `published_covariance_method` to `Some(Hessian)` →
/// the `s` and `rsr` arms fire. Drop the `fit_inner` assignment → all three fire.
#[test]
fn a_fit_reports_the_estimator_its_standard_errors_came_from() {
    let model = one_cpt_model();
    let pop = population();

    for requested in [
        CovarianceMethod::Hessian,
        CovarianceMethod::CrossProduct,
        CovarianceMethod::Sandwich,
    ] {
        let fit = fit(
            &model,
            &pop,
            &model.default_params,
            &opts(requested, /* explicit */ true),
        )
        .expect("fit");

        assert!(
            fit.covariance_matrix.is_some(),
            "premise: the covariance step must produce a matrix under \
             covariance_method = {}, else the label below is `None` for a reason \
             unrelated to #1382. warnings: {:?}",
            requested.label(),
            fit.warnings
        );
        assert_eq!(
            fit.covariance_method,
            Some(requested),
            "#1382: the fit reports no estimator, or the wrong one, for the SEs / \
             eigenvalues / condition number it published under covariance_method = {}",
            requested.label()
        );
    }
}

/// **G2 — the pairing, in both directions.** A matrix without a label is the
/// reported bug; a label without a matrix is the mirror image, and it is worse —
/// it invites a reader to compare a condition number that was never computed.
///
/// `run_covariance_step = false` is the cheap half. The `Failed` /
/// `SirFallback` halves are covered by `published_covariance_method`'s `None`
/// arm at Tier 1 rather than by standing up a singular Hessian here.
#[test]
fn a_fit_with_no_covariance_matrix_names_no_estimator() {
    let model = one_cpt_model();
    let pop = population();
    let mut o = opts(CovarianceMethod::Sandwich, true);
    o.run_covariance_step = false;

    let fit = fit(&model, &pop, &model.default_params, &o).expect("fit");

    assert!(fit.covariance_matrix.is_none(), "premise");
    assert_eq!(
        fit.covariance_status,
        CovarianceStatus::NotRequested,
        "premise"
    );
    assert_eq!(
        fit.covariance_method, None,
        "an estimator name must never outlive the matrix it describes — here there \
         are no SEs, no eigenvalues and no condition number for `rsr` to label"
    );
    assert!(
        fit.cov_condition_number.is_none() && fit.cov_eigenvalues.is_none(),
        "premise: the quantities the label exists for are absent too"
    );
}

/// **G3 — `run_covariance` relabels, and does not inherit — including in the
/// machine-readable payload.**
///
/// The standalone entry point is a second publishing site: it clones the incoming
/// fit and overwrites the covariance block, and `ferx-tools` / the R wrapper reach
/// it without going through `fit()`. Three failures it pins, none visible from G1:
///
/// 1. **A stale label on the typed field**, if the clone is not overwritten.
/// 2. **A stale label in `warnings_structured`** (#1382 review). Measured before
///    the fix, re-running this `r` fit under `s` returned
///    `covariance_method = Some(CrossProduct)` beside a retained entry reading
///    `{"covariance_method": "r", "condition_number": 4091320907.9}` — a payload
///    contradicting the field next to it, and carrying the *old* condition number
///    while `cov_condition_number` had been recomputed.
/// 3. **New covariance warnings with no structured entry at all**, since nothing
///    rebuilt the structured list after extending the flat one.
///
/// The incoming fit is required to carry a covariance-step warning, because
/// without one there is no stale payload to contradict and the test would pass on
/// an empty list — the fixture is ill-conditioned enough to trip the eigenvalue
/// floor under `r`, which is what supplies it.
///
/// Mutation (run): delete the `out.covariance_method = …` assignment → (1) fires.
/// Delete the `retain` → (2) fires. Delete the `rebuild_warnings_structured` call
/// → (3) fires.
#[test]
fn rerunning_the_covariance_step_relabels_the_result_and_its_payloads() {
    let model = one_cpt_model();
    let pop = population();

    let fitted = fit(
        &model,
        &pop,
        &model.default_params,
        &opts(CovarianceMethod::Hessian, true),
    )
    .expect("fit");
    assert_eq!(
        fitted.covariance_method,
        Some(CovarianceMethod::Hessian),
        "premise: the incoming fit must already carry a *different* label, or the \
         overwrite below is indistinguishable from doing nothing"
    );
    let stale_payloads: Vec<_> = fitted
        .warnings_structured
        .iter()
        .filter(|w| is_covariance_code(&w.category))
        .collect();
    assert!(
        !stale_payloads.is_empty(),
        "premise: the incoming fit must carry a covariance-step warning whose \
         payload names `r`, or there is no stale provenance for the re-run to \
         correct and this test cannot fail. structured: {:?}",
        fitted
            .warnings_structured
            .iter()
            .map(|w| &w.category)
            .collect::<Vec<_>>()
    );
    assert!(
        stale_payloads.iter().any(|w| {
            w.details
                .as_ref()
                .and_then(|d| d.get("covariance_method"))
                .map(|m| m == "r")
                .unwrap_or(false)
        }),
        "premise: that warning's payload must name `r`: {stale_payloads:?}"
    );

    let rerun = run_covariance(
        &fitted,
        Some(&model),
        Some(&pop),
        &opts(CovarianceMethod::CrossProduct, true),
    )
    .expect("run_covariance");

    assert!(
        rerun.covariance_matrix.is_some(),
        "premise: {:?}",
        rerun.warnings
    );
    // (1) the typed field
    assert_eq!(
        rerun.covariance_method,
        Some(CovarianceMethod::CrossProduct),
        "run_covariance published an S⁻¹ matrix under the incoming fit's R⁻¹ label"
    );
    // (2) no payload still claims the superseded estimator
    for entry in &rerun.warnings_structured {
        let named = entry
            .details
            .as_ref()
            .and_then(|d| d.get("covariance_method"))
            .and_then(|m| m.as_str())
            .map(|s| s.to_string());
        if let Some(named) = named {
            assert_eq!(
                named, "s",
                "a warning payload still names the superseded estimator after a \
                 re-run under `s` — the provenance error this issue exists to fix, \
                 one level down: {entry:?}"
            );
        }
    }
    // ...and the superseded covariance-step warnings are gone from the flat list
    // too, since the step they describe has been replaced.
    assert!(
        !rerun
            .warnings
            .iter()
            .any(|w| w.contains("eigenvalue floor applied to FD Hessian")),
        "the `r` step's eigenvalue-floor warning describes a step that no longer \
         exists — it cannot even arise under the cross-product: {:?}",
        rerun.warnings
    );
    // (3) every warning has a structured entry
    assert_eq!(
        rerun.warnings.len(),
        rerun.warnings_structured.len(),
        "flat and structured warning lists disagree after a re-run:\nflat: {:?}\nstructured: {:?}",
        rerun.warnings,
        rerun
            .warnings_structured
            .iter()
            .map(|w| &w.message)
            .collect::<Vec<_>>()
    );
}

fn is_covariance_code(c: &WarningCode) -> bool {
    matches!(
        c,
        WarningCode::CovarianceStep
            | WarningCode::CovarianceFailed
            | WarningCode::CovarianceRegularized
            | WarningCode::ConditionNumber
    )
}
