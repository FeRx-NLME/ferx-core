//! Tier-2 integration tests for the optional `[data]` model-file block (#690).
//!
//! Tests exercise the full parse → resolve → fit boundary via the public API.
//! They return quickly (a small `fit()` with a handful of outer iterations)
//! and do not need the `slow-tests` gate.

use ferx_core::{run_model_with_data_inits, validate_model_file};
use std::path::Path;

/// Minimal 1-cpt IV model — 2 outer iterations so the tests are fast. `{data}`
/// is substituted with a `[data]` block (or left empty) per test.
const MODEL_TEMPLATE: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method  = focei
  maxiter = 2
  gradient = fd

{data}
";

/// Writes a model file (with an optional `[data] path = data.csv`) plus a copy
/// of `data/one_cpt_iv.csv` into a fresh temp dir, so a relative `[data] path`
/// resolves correctly regardless of the test runner's cwd.
fn write_model_and_data(dir: &Path, with_data_block: bool) -> std::path::PathBuf {
    let csv_src = Path::new("data/one_cpt_iv.csv");
    std::fs::copy(csv_src, dir.join("data.csv")).expect("copy one_cpt_iv.csv");

    let data_block = if with_data_block {
        "[data]\n  path = data.csv\n"
    } else {
        ""
    };
    let model_src = MODEL_TEMPLATE.replace("{data}", data_block);
    let model_path = dir.join("m.ferx");
    std::fs::write(&model_path, model_src).expect("write model file");
    model_path
}

#[test]
fn fit_falls_back_to_model_declared_data_path_when_none_given() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_and_data(dir.path(), true);

    let (result, _population) = run_model_with_data_inits(model_path.to_str().unwrap(), None, None)
        .expect("fit should succeed using [data] block");
    assert!(
        result.warnings.iter().all(|w| !w.contains("overridden")),
        "no override happened, so no override warning expected: {:?}",
        result.warnings
    );
    assert_eq!(
        result.data_path.as_deref(),
        Some(dir.path().join("data.csv").to_str().unwrap())
    );
}

#[test]
fn fit_without_data_anywhere_errors_clearly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_and_data(dir.path(), false);

    let err = run_model_with_data_inits(model_path.to_str().unwrap(), None, None)
        .expect_err("no [data] block and no external path must error");
    assert!(
        err.contains("no dataset specified"),
        "unexpected error: {err}"
    );
}

#[test]
fn fit_explicit_data_path_overrides_model_declared_path_with_warning() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_and_data(dir.path(), true);

    // A second copy of the CSV under a different name stands in for a
    // deliberately different dataset passed on the CLI.
    let override_path = dir.path().join("override.csv");
    std::fs::copy("data/one_cpt_iv.csv", &override_path).expect("copy override csv");

    let (result, _population) = run_model_with_data_inits(
        model_path.to_str().unwrap(),
        Some(override_path.to_str().unwrap()),
        None,
    )
    .expect("fit should succeed using the overriding --data path");

    assert_eq!(
        result.data_path.as_deref(),
        Some(override_path.to_str().unwrap())
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("overridden") && w.contains("override.csv")),
        "expected an override warning, got: {:?}",
        result.warnings
    );
}

#[test]
fn check_uses_model_declared_data_path_when_no_data_flag_given() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_and_data(dir.path(), true);

    let report = validate_model_file(model_path.to_str().unwrap(), None);
    assert_eq!(
        report.data.as_deref(),
        Some(dir.path().join("data.csv").to_str().unwrap())
    );
    assert!(
        report.diagnostics.iter().all(|d| d.code != "E_PARSE"),
        "model + declared data must parse and read cleanly: {:?}",
        report.diagnostics
    );
}

#[test]
fn check_explicit_data_flag_overrides_model_declared_path_with_warning() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = write_model_and_data(dir.path(), true);
    let override_path = dir.path().join("override.csv");
    std::fs::copy("data/one_cpt_iv.csv", &override_path).expect("copy override csv");

    let report = validate_model_file(
        model_path.to_str().unwrap(),
        Some(override_path.to_str().unwrap()),
    );
    assert_eq!(
        report.data.as_deref(),
        Some(override_path.to_str().unwrap())
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "W_DATA_PATH_OVERRIDE"),
        "expected a W_DATA_PATH_OVERRIDE diagnostic: {:?}",
        report.diagnostics
    );
}
