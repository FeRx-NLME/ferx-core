//! Integration tests for `[derived]` and `[output]` blocks (Steps 5–10).
//!
//! Tier 2: parse-time validation (fast, no `slow-tests` gate). Tier 3 tests
//! that require a full fit to convergence are gated with `#[cfg_attr(..., ignore)]`.

use ferx_core::api::{check_model_data, tafd_tad_for_subject};
use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::types::{DoseEvent, Population, Subject};
use ferx_core::{fit, FitOptions};
use std::collections::HashMap;

// ── Minimal model template ───────────────────────────────────────────────────

const BASE_MODEL: &str = "
[parameters]
  theta CL(1.0, 0, 100)
  theta V(10.0, 0, 1000)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = exp(log(CL) + ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

fn base_with_extra(extra: &str) -> String {
    format!("{BASE_MODEL}\n{extra}")
}

fn one_dose_population() -> Population {
    let obs_times = vec![1.0, 4.0, 12.0, 24.0];
    let n_obs = obs_times.len();
    let mut cov = HashMap::new();
    cov.insert("WT".to_string(), 70.0);
    Population {
        covariate_names: vec!["WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![5.0, 3.0, 1.5, 0.7],
            obs_cmts: vec![1; n_obs],
            covariates: cov,
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; n_obs],
            occasions: vec![],
            dose_occasions: vec![],
        }],
    }
}

// ── Parser validation tests ───────────────────────────────────────────────────

#[test]
fn derived_name_conflict_returns_err() {
    let src = base_with_extra("[derived]\n  IPRED = CL / V");
    let result = parse_full_model(&src);
    assert!(result.is_err(), "IPRED is a built-in column — must error");
    let msg = match result {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(
        msg.contains("E_DERIVED_NAME_CONFLICT"),
        "expected E_DERIVED_NAME_CONFLICT in: {msg}"
    );
}

#[test]
fn derived_eta_name_conflict_returns_err() {
    let src = base_with_extra("[derived]\n  ETA_CL = CL / V");
    let result = parse_full_model(&src);
    assert!(result.is_err(), "ETA_CL is an eta name — must error");
    let msg = match result {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(
        msg.contains("E_DERIVED_NAME_CONFLICT"),
        "expected E_DERIVED_NAME_CONFLICT in: {msg}"
    );
}

#[test]
fn derived_theta_name_conflict_returns_err() {
    let src = base_with_extra("[derived]\n  CL = V / V");
    let result = parse_full_model(&src);
    assert!(result.is_err(), "CL is an indiv param name — must error");
    let msg = match result {
        Err(e) => e,
        Ok(_) => panic!("expected Err"),
    };
    assert!(msg.contains("E_DERIVED_NAME_CONFLICT"));
}

#[test]
fn output_unknown_column_emits_error() {
    let src = base_with_extra("[output]\n  UNKNOWN_COLUMN");
    let model = parse_full_model(&src).expect("parse ok").model;
    let pop = one_dose_population();
    let diags = check_model_data(&model, &pop);
    let codes: Vec<&str> = diags.iter().map(|d| d.code.as_str()).collect();
    assert!(
        codes.contains(&"E_OUTPUT_UNKNOWN_COLUMN"),
        "expected E_OUTPUT_UNKNOWN_COLUMN in: {codes:?}"
    );
}

#[test]
fn output_duplicate_tafd_emits_warning() {
    let src = base_with_extra("[output]\n  TAFD");
    let model = parse_full_model(&src).expect("parse ok").model;
    let pop = one_dose_population();
    let diags = check_model_data(&model, &pop);
    let codes: Vec<&str> = diags.iter().map(|d| d.code.as_str()).collect();
    assert!(
        codes.contains(&"W_OUTPUT_DUPLICATE"),
        "expected W_OUTPUT_DUPLICATE in: {codes:?}"
    );
}

#[test]
fn output_valid_covariate_no_error() {
    let src = base_with_extra("[output]\n  WT");
    let model = parse_full_model(&src).expect("parse ok").model;
    let pop = one_dose_population();
    let diags = check_model_data(&model, &pop);
    let errors: Vec<&str> = diags
        .iter()
        .filter(|d| d.severity == ferx_core::diagnostics::Severity::Error)
        .map(|d| d.code.as_str())
        .collect();
    assert!(
        errors.is_empty(),
        "WT is a covariate — no errors expected; got: {errors:?}"
    );
}

#[test]
fn derived_integral_step_ignored_warning_for_dv() {
    let src = base_with_extra("[derived]\n  AUC = integral(DV, from=0, to=24, step=0.1)");
    let result = parse_full_model(&src).expect("parse ok");
    let has_step_warn = result
        .model
        .parse_warnings
        .iter()
        .any(|w| w.contains("W_DERIVED_STEP_IGNORED"));
    assert!(
        has_step_warn,
        "expected W_DERIVED_STEP_IGNORED when step= given for DV integral; got: {:?}",
        result.model.parse_warnings
    );
}

// ── TAFD/TAD helper unit tests ────────────────────────────────────────────────

fn make_subject_with_doses(obs_times: Vec<f64>, doses: Vec<DoseEvent>) -> Subject {
    let n = obs_times.len();
    Subject {
        id: "1".into(),
        doses,
        obs_times,
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: vec![],
        obs_covariates: vec![],
        pk_only_times: vec![],
        pk_only_covariates: vec![],
        reset_times: vec![],
        cens: vec![0; n],
        occasions: vec![],
        dose_occasions: vec![],
    }
}

#[test]
fn tafd_correct_single_dose() {
    // Dose at t=5, obs at t=10 → TAFD = 5
    let subj = make_subject_with_doses(
        vec![10.0],
        vec![DoseEvent::new(5.0, 100.0, 1, 0.0, false, 0.0)],
    );
    let (tafd, _) = tafd_tad_for_subject(&subj, 0, 0.0);
    assert!((tafd - 5.0).abs() < 1e-10, "TAFD should be 5, got {tafd}");
}

#[test]
fn tafd_nan_when_no_dose() {
    // No doses → TAFD is NaN
    let subj = make_subject_with_doses(vec![10.0], vec![]);
    let (tafd, _) = tafd_tad_for_subject(&subj, 0, 0.0);
    assert!(
        tafd.is_nan(),
        "TAFD should be NaN with no doses, got {tafd}"
    );
}

#[test]
fn tad_nan_when_dose_after_obs() {
    // Dose at t=20, obs at t=1 → no prior dose → TAD is NaN
    let subj = make_subject_with_doses(
        vec![1.0],
        vec![DoseEvent::new(20.0, 100.0, 1, 0.0, false, 0.0)],
    );
    let (_, tad) = tafd_tad_for_subject(&subj, 0, 0.0);
    assert!(
        tad.is_nan(),
        "TAD should be NaN when dose is after obs, got {tad}"
    );
}

#[test]
fn tad_ss_modular() {
    // SS dose at t=0, II=12, obs at t=50 → TAD = 50 mod 12 = 2
    let mut dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 12.0);
    dose.ss = true;
    let subj = make_subject_with_doses(vec![50.0], vec![dose]);
    let (_, tad) = tafd_tad_for_subject(&subj, 0, 0.0);
    assert!((tad - 2.0).abs() < 1e-10, "TAD(SS) should be 2, got {tad}");
}

#[test]
fn tad_after_addl_expanded_doses() {
    // Dose at t=0, plus explicit doses at t=24,48 (simulating ADDL=2, II=24).
    // Obs at t=50 → last effective dose at t=48 → TAD=2
    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(48.0, 100.0, 1, 0.0, false, 0.0),
    ];
    let subj = make_subject_with_doses(vec![50.0], doses);
    let (_, tad) = tafd_tad_for_subject(&subj, 0, 0.0);
    assert!((tad - 2.0).abs() < 1e-10, "TAD should be 2, got {tad}");
}

// ── End-to-end: fit() produces finite extra_columns ──────────────────────────

/// End-to-end coverage: after a short FOCE-I fit, extra_columns from [derived]
/// and [output] must be populated with finite, non-NaN values for every subject.
/// Exercises the full post-fit derived pipeline including PerRow, Aggregate, and
/// the output-column echo.
#[test]
fn fit_produces_finite_derived_and_output_columns() {
    const MODEL: &str = "
[parameters]
  theta CL(1.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[derived]
  Cmax = max(IPRED)
  AUC  = integral(IPRED, from=0, to=24)

[output]
  CL

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = one_dose_population();

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short fit must not error");

    // Every subject result must have extra_columns for Cmax, AUC, and CL.
    for sr in &result.subjects {
        for name in &["Cmax", "AUC", "CL"] {
            let col = sr
                .extra_columns
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("extra column '{name}' missing for subject {}", sr.id));
            assert!(
                !col.1.is_empty(),
                "column '{name}' is empty for subject {}",
                sr.id
            );
            for &v in &col.1 {
                assert!(
                    v.is_finite(),
                    "column '{name}' has non-finite value {v} for subject {}",
                    sr.id
                );
            }
        }
    }
}

/// Regression: an [output] covariate referenced with a different case than the
/// dataset header must echo the covariate value, not NaN. `validate_output_columns`
/// accepts the name case-insensitively (`wt` matches header `WT`), so the post-fit
/// echo in `compute_extra_output_columns` must resolve it case-insensitively too.
#[test]
fn output_covariate_case_insensitive_echo() {
    const MODEL: &str = "
[parameters]
  theta CL(1.0, 0.01, 50.0)
  theta V(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V  = V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[output]
  wt

[fit_options]
  method   = focei
  maxiter  = 2
  gradient = fd
";
    let model = parse_model_string(MODEL).expect("model must parse");
    // Population covariate header is uppercase `WT` = 70.0; the [output] entry
    // is lowercase `wt`.
    let pop = one_dose_population();

    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short fit must not error");

    for sr in &result.subjects {
        let col = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case("wt"))
            .unwrap_or_else(|| panic!("output column 'wt' missing for subject {}", sr.id));
        assert!(
            !col.1.is_empty(),
            "column 'wt' is empty for subject {}",
            sr.id
        );
        for &v in &col.1 {
            assert_eq!(
                v, 70.0,
                "output 'wt' must echo covariate header WT=70 for subject {}, got {v}",
                sr.id
            );
        }
    }
}

/// Parser must reject step=0 and step negative in [derived] integral().
#[test]
fn integral_step_zero_is_rejected_at_parse() {
    let src = base_with_extra("[derived]\n  AUC = integral(IPRED, from=0, to=24, step=0)");
    let err = parse_full_model(&src)
        .err()
        .expect("step=0 must be rejected at parse time");
    assert!(
        err.contains("step") && err.contains("positive"),
        "error should cite step= and positive, got: {err}"
    );
}

/// Parser must reject window=0 in [derived] integral().
#[test]
fn integral_window_zero_is_rejected_at_parse() {
    let src = base_with_extra("[derived]\n  AUC = integral(IPRED, window=0)");
    let err = parse_full_model(&src)
        .err()
        .expect("window=0 must be rejected at parse time");
    assert!(
        err.contains("window") && err.contains("positive"),
        "error should cite window= and positive, got: {err}"
    )
}
