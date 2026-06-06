//! Integration tests for `validate_model_file` (the engine behind `ferx check`).
//!
//! These exercise the public validation API end-to-end: parse a real example,
//! catch a missing block, catch a data/model covariate mismatch, and prove the
//! refactor kept `fit()`'s error string byte-identical to the diagnostic.
//! All return immediately (no fit convergence), so they belong in Tier 2.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv, validate_model_file, FitOptions};
use std::path::Path;

/// Write `content` to a uniquely-named temp `.ferx` file and return its path.
fn temp_model(tag: &str, content: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ferx_check_{}_{}_{}.ferx",
        tag,
        std::process::id(),
        n
    ));
    std::fs::write(&path, content).expect("write temp model");
    path
}

const COV_MODEL: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WGT / 70.0)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
";

#[test]
fn valid_example_passes_with_no_diagnostics() {
    let report = validate_model_file("examples/warfarin_bobyqa.ferx", None);
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    assert_eq!(report.error_count(), 0);
    assert_eq!(report.model, "warfarin_bobyqa");
}

#[test]
fn missing_block_is_reported_as_e_missing_block() {
    // No [error_model] block.
    let model = temp_model(
        "missing_block",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)
",
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid);
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_MISSING_BLOCK");
    assert_eq!(d.block.as_deref(), Some("error_model"));
    let _ = std::fs::remove_file(&model);
}

#[test]
fn missing_covariate_is_reported_with_data() {
    // bioavailability.csv carries no covariate columns, but the model references WGT.
    let model = temp_model("missing_cov", COV_MODEL);
    let report = validate_model_file(model.to_str().unwrap(), Some("data/bioavailability.csv"));
    assert!(!report.valid);
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_MISSING_COVARIATE")
        .expect("expected E_MISSING_COVARIATE");
    assert!(d.message.contains("WGT"));
    assert!(d.suggestion.is_some());
    let _ = std::fs::remove_file(&model);
}

#[test]
fn no_data_means_no_covariate_check() {
    // Same model, but without --data the covariate check does not run, so the
    // model is structurally valid.
    let model = temp_model("no_data", COV_MODEL);
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    let _ = std::fs::remove_file(&model);
}

/// Regression guard: the message `fit()` produces for a missing covariate must
/// stay byte-identical to the diagnostic `validate_model_file` reports — both
/// now flow through the shared `check_covariates`.
#[test]
fn fit_error_matches_check_diagnostic_for_missing_covariate() {
    let model_path = temp_model("fit_regression", COV_MODEL);
    let report = validate_model_file(
        model_path.to_str().unwrap(),
        Some("data/bioavailability.csv"),
    );
    let diag_msg = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_MISSING_COVARIATE")
        .expect("diagnostic present")
        .message
        .clone();

    let model = parse_full_model_file(&model_path).unwrap().model;
    let pop = read_nonmem_csv(Path::new("data/bioavailability.csv"), None, None).unwrap();
    let fit_err = fit(&model, &pop, &model.default_params, &FitOptions::default())
        .expect_err("fit must reject the missing covariate before fitting");

    assert_eq!(diag_msg, fit_err);
    let _ = std::fs::remove_file(&model_path);
}

/// Block-level line numbers are recorded on the parsed model.
#[test]
fn parser_records_block_header_lines() {
    let model_path = temp_model("block_lines", COV_MODEL);
    let parsed = parse_full_model_file(&model_path).unwrap();
    // `[parameters]` is line 1; `[individual_parameters]` line 7; `[error_model]` line 14.
    assert_eq!(parsed.block_lines.get("parameters"), Some(&1));
    assert_eq!(parsed.block_lines.get("individual_parameters"), Some(&7));
    assert_eq!(parsed.block_lines.get("error_model"), Some(&14));
    let _ = std::fs::remove_file(&model_path);
}

// ── [covariates] block (issue #182) ─────────────────────────────────────────

const COV_DECL_MODEL: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  WT   continuous
  CRCL continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT / 70.0)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The bundled covariate example validates cleanly against its dataset.
#[test]
fn covariates_example_validates() {
    let report = validate_model_file(
        "examples/two_cpt_oral_cov.ferx",
        Some("data/two_cpt_oral_cov.csv"),
    );
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
}

/// `read_nonmem_csv_with_covariates` produces a table echoing the declared
/// columns, one row per input record (incl. dose rows).
#[test]
fn covariate_table_built_from_declarations() {
    use ferx_core::{read_nonmem_csv_with_covariates, CovariateDecl, CovariateKind};
    let decls = vec![
        CovariateDecl {
            name: "WT".into(),
            kind: CovariateKind::Continuous,
        },
        CovariateDecl {
            name: "CRCL".into(),
            kind: CovariateKind::Continuous,
        },
    ];
    let (pop, table) =
        read_nonmem_csv_with_covariates(Path::new("data/two_cpt_oral_cov.csv"), &decls, &[], None)
            .unwrap();
    assert_eq!(table.names, vec!["WT", "CRCL"]);
    // One row per input record — strictly more than the observation count
    // (dose rows are included), and at least as many as the obs total.
    assert!(table.rows.len() >= pop.n_obs());
    assert!(table.rows.iter().all(|r| r.values.len() == 2));
}

/// A `[covariates]` block declaring a column absent from the data is rejected
/// by `ferx check` with `E_MISSING_COVARIATE`.
#[test]
fn declared_covariate_absent_from_data_is_reported() {
    // bioavailability.csv has no WT/CRCL columns.
    let model = temp_model("cov_decl_missing", COV_DECL_MODEL);
    let report = validate_model_file(model.to_str().unwrap(), Some("data/bioavailability.csv"));
    assert!(!report.valid);
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_MISSING_COVARIATE")
        .expect("expected E_MISSING_COVARIATE");
    assert!(d.message.contains("WT") || d.message.contains("CRCL"));
    let _ = std::fs::remove_file(&model);
}

/// A covariate used in the model but not declared in `[covariates]` is allowed
/// (still usable) — the parser warns rather than erroring.
#[test]
fn undeclared_referenced_covariate_warns_not_errors() {
    let model = temp_model(
        "cov_undeclared",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  CRCL continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT / 70.0)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    );
    // The model parses successfully (no parse error)...
    let parsed = parse_full_model_file(&model).expect("model should parse");
    assert!(
        parsed
            .model
            .parse_warnings
            .iter()
            .any(|w| w.contains("WT") && w.contains("not declared")),
        "expected an undeclared-covariate warning, got: {:?}",
        parsed.model.parse_warnings
    );
    // ...and `ferx check` (no data) reports no errors.
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    let _ = std::fs::remove_file(&model);
}

/// Write `content` to a uniquely-named temp `.csv` file and return its path.
fn temp_data(tag: &str, content: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ferx_check_{}_{}_{}.csv",
        tag,
        std::process::id(),
        n
    ));
    std::fs::write(&path, content).expect("write temp data");
    path
}

/// `ferx check` now reads through the same strict reader the fit uses, so a
/// non-numeric value in a declared covariate is caught at check time (parity
/// with fit) instead of passing check and then failing the fit.
#[test]
fn check_reports_non_numeric_declared_covariate() {
    let model = temp_model(
        "cov_nonnumeric",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  WT  continuous
  SEX categorical

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT / 70.0)
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    );
    // SEX is declared categorical but coded as strings — must be rejected.
    let data = temp_data(
        "cov_nonnumeric",
        "ID,TIME,DV,EVID,AMT,WT,SEX\n1,0,.,1,100,70,M\n1,1,5.0,0,.,70,M\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some(data.to_str().unwrap()));
    assert!(!report.valid);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "E_COVARIATE_NOT_NUMERIC" && d.message.contains("SEX")),
        "expected E_COVARIATE_NOT_NUMERIC for SEX, got: {:?}",
        report.diagnostics
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&data);
}

/// Regression: a covariate referenced by the model but absent from the data is
/// still caught (E_MISSING_COVARIATE) even when a [covariates] block is present
/// — the block must not mask the missing-covariate guard.
#[test]
fn referenced_absent_covariate_errors_even_with_block() {
    let model = temp_model(
        "cov_masking",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVAGE(0.1, 0.001, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  WT continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT / 70.0) * (AGE / 40.0)^TVAGE
  V  = TVV

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    );
    // WT present (declared, numeric) but AGE (referenced, undeclared) is absent.
    let data = temp_data(
        "cov_masking",
        "ID,TIME,DV,EVID,AMT,WT\n1,0,.,1,100,70\n1,1,5.0,0,.,70\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some(data.to_str().unwrap()));
    assert!(!report.valid);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "E_MISSING_COVARIATE" && d.message.contains("AGE")),
        "expected E_MISSING_COVARIATE for AGE, got: {:?}",
        report.diagnostics
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&data);
}

/// The report serializes to JSON with the documented shape.
#[test]
fn report_serializes_to_json() {
    let report = validate_model_file("examples/warfarin_bobyqa.ferx", None);
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains("\"valid\":true"));
    assert!(json.contains("\"model\":\"warfarin_bobyqa\""));
    assert!(json.contains("\"diagnostics\":[]"));
}
