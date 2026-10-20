//! `validate_model_file_with` checks a model file against the options a fit with
//! call-time overrides runs with (#2165), where `validate_model_file` can only see
//! the file's own `[fit_options]` / `[data_selection]`.
//!
//! Every override test is a **differential pair**: the same model and data, once
//! with no overrides and once with them, asserting the finding changes. A test
//! asserting only the overridden side would pass against an implementation that
//! ignores `overrides` whenever the file alone happens to give that answer.
//! Each pair covers one of the readers the issue names: the data read's
//! `iov_column` and the model–data rule's `iov_occasion`, the selection filter,
//! `check_model_options` and `check_parameter_priors`.
//!
//! No `fit()` runs here — a check reads the data and stops — so this is Tier 2
//! and fast.

use ferx_core::{validate_model_file, validate_model_file_with, CheckReport};
use std::io::Write;
use tempfile::NamedTempFile;

fn pairs(kv: &[(&str, &str)]) -> Vec<(String, String)> {
    kv.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn error_codes(report: &CheckReport) -> Vec<&str> {
    report
        .diagnostics
        .iter()
        .filter(|d| d.severity == ferx_core::Severity::Error)
        .map(|d| d.code.as_str())
        .collect()
}

fn write_tmp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    f
}

const IOV_MODEL: &str = "examples/warfarin_iov.ferx";
/// No `OCC` column.
const NO_OCC_DATA: &str = "data/warfarin.csv";

/// The measurement from the issue: warfarin_iov with no `iov_column` and no OCC
/// column validates with exactly one error, `E_IOV_MISSING_OCC`, yet fits with
/// `iov_occasion = dose`. The override on `iov_column` reaches the data read (the
/// file's `iov_column = OCC` is cleared), and the one on `iov_occasion` reaches the
/// model–data rule (occasions derived from doses need no column).
#[test]
fn iov_occasion_override_clears_the_missing_occ_error() {
    let no_column = validate_model_file_with(
        IOV_MODEL,
        Some(NO_OCC_DATA),
        &pairs(&[("iov_column", "none")]),
    );
    assert_eq!(
        error_codes(&no_column),
        vec!["E_IOV_MISSING_OCC"],
        "{:#?}",
        no_column.diagnostics
    );

    let derived = validate_model_file_with(
        IOV_MODEL,
        Some(NO_OCC_DATA),
        &pairs(&[("iov_column", "none"), ("iov_occasion", "dose")]),
    );
    assert!(
        error_codes(&derived).is_empty(),
        "iov_occasion = dose derives occasions from the doses: {:#?}",
        derived.diagnostics
    );
}

/// The file's `iov_column = OCC` against data with no OCC column fails the read
/// itself; overriding `iov_column` away changes the finding, so the override is
/// what the data read was handed.
#[test]
fn iov_column_override_reaches_the_data_read() {
    let file_only = validate_model_file(IOV_MODEL, Some(NO_OCC_DATA));
    let overridden = validate_model_file_with(
        IOV_MODEL,
        Some(NO_OCC_DATA),
        &pairs(&[("iov_column", "none")]),
    );
    assert!(
        !error_codes(&file_only).is_empty(),
        "{:#?}",
        file_only.diagnostics
    );
    assert_ne!(
        error_codes(&file_only),
        error_codes(&overridden),
        "file-only: {:#?}\noverridden: {:#?}",
        file_only.diagnostics,
        overridden.diagnostics
    );
}

/// A one-state model whose `[error_model]` covers `CMT=1` only, against data
/// observed on `{1, 2}` (fixture A of `check_applies_data_selection`): rejected
/// unfiltered, accepted once a call-time `ignore` drops the CMT-2 rows.
const CMT_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -CL/V * central

[error_model]
  CMT=1: DV ~ proportional(PROP)

[fit_options]
  method     = focei
  covariance = false
";

const CMT_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                       1,0,.,1,100,1,1\n\
                       1,1,9.0,0,.,1,0\n\
                       1,2,8.2,0,.,1,0\n\
                       2,0,.,1,100,1,1\n\
                       2,1,9.4,0,.,1,0\n\
                       2,1,1.2,0,.,2,0\n\
                       2,2,1.0,0,.,2,0\n";

#[test]
fn selection_override_reaches_the_data_read() {
    let model = write_tmp(CMT_MODEL, ".ferx");
    let data = write_tmp(CMT_CSV, ".csv");
    let (m, d) = (
        model.path().to_str().unwrap(),
        data.path().to_str().unwrap(),
    );

    let file_only = validate_model_file_with(m, Some(d), &[]);
    assert!(
        error_codes(&file_only).contains(&"E_PER_CMT_ERROR_MODEL"),
        "{:#?}",
        file_only.diagnostics
    );
    for clause in [("ignore", "CMT == 2"), ("accept", "CMT == 1")] {
        let filtered = validate_model_file_with(m, Some(d), &pairs(&[clause]));
        assert!(
            error_codes(&filtered).is_empty(),
            "{clause:?}: {:#?}",
            filtered.diagnostics
        );
    }
}

/// `check_model_options` reads the merged options: trust-region cannot carry an
/// IOV model's kappas.
#[test]
fn optimizer_override_reaches_check_model_options() {
    let file_only = validate_model_file_with(IOV_MODEL, None, &[]);
    assert!(
        !error_codes(&file_only).contains(&"E_OPTIMIZER_IOV"),
        "{:#?}",
        file_only.diagnostics
    );
    let overridden =
        validate_model_file_with(IOV_MODEL, None, &pairs(&[("optimizer", "trust_region")]));
    assert!(
        error_codes(&overridden).contains(&"E_OPTIMIZER_IOV"),
        "{:#?}",
        overridden.diagnostics
    );
}

/// `check_parameter_priors` reads the merged options: SAEM does not apply priors.
/// The reverse direction too — a later pair wins over an earlier one.
#[test]
fn method_override_reaches_check_parameter_priors() {
    let prior = "examples/warfarin_prior.ferx";
    let file_only = validate_model_file_with(prior, None, &[]);
    assert!(
        error_codes(&file_only).is_empty(),
        "{:#?}",
        file_only.diagnostics
    );
    let saem = validate_model_file_with(prior, None, &pairs(&[("method", "saem")]));
    assert!(
        error_codes(&saem).contains(&"E_PRIOR_METHOD_UNSUPPORTED"),
        "{:#?}",
        saem.diagnostics
    );
    let back = validate_model_file_with(
        prior,
        None,
        &pairs(&[("method", "saem"), ("method", "focei")]),
    );
    assert!(
        error_codes(&back).is_empty(),
        "the later `method` wins: {:#?}",
        back.diagnostics
    );
}

/// A pair that cannot be applied is `E_FIT_OPTION`, one per pair, and nothing
/// else: the remaining checks would describe options no fit runs with. Both arms
/// of `apply_fit_option` — an unknown key (`Ok(false)`) and a bad value (`Err`).
#[test]
fn an_override_that_does_not_apply_is_e_fit_option_and_terminal() {
    let report = validate_model_file_with(
        IOV_MODEL,
        Some(NO_OCC_DATA),
        &pairs(&[
            ("no_such_option", "1"),
            ("maxiter", "many"),
            ("iov_occasion", "dose"),
        ]),
    );
    assert!(!report.valid);
    assert_eq!(
        error_codes(&report),
        vec!["E_FIT_OPTION", "E_FIT_OPTION"],
        "{:#?}",
        report.diagnostics
    );
    assert_eq!(report.diagnostics.len(), 2, "{:#?}", report.diagnostics);
    assert!(
        report.diagnostics[0].message.contains("`no_such_option`")
            && report.diagnostics[0]
                .message
                .contains("not a known fit option"),
        "{}",
        report.diagnostics[0].message
    );
    assert!(
        report.diagnostics[0]
            .suggestion
            .as_deref()
            .is_some_and(|s| s.contains("`[data_selection]`")),
        "{:?}",
        report.diagnostics[0].suggestion
    );
    assert!(
        report.diagnostics[1].message.contains("`maxiter = many`"),
        "{}",
        report.diagnostics[1].message
    );
    assert_eq!(report.model, "warfarin_iov");
}

/// No overrides is `validate_model_file`, diagnostic for diagnostic.
#[test]
fn empty_overrides_is_validate_model_file() {
    let a = validate_model_file(IOV_MODEL, Some(NO_OCC_DATA));
    let b = validate_model_file_with(IOV_MODEL, Some(NO_OCC_DATA), &[]);
    assert!(!a.diagnostics.is_empty());
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
}

/// `bloq_method` reaches the checks too: M3 does not combine with a `block_sigma`.
#[test]
fn bloq_override_reaches_the_checks() {
    let model = "examples/correlated_residual_combined.ferx";
    const CODE: &str = "E_BLOCK_SIGMA_M3_UNSUPPORTED";
    let file_only = validate_model_file_with(model, None, &[]);
    assert!(
        !error_codes(&file_only).contains(&CODE),
        "{:#?}",
        file_only.diagnostics
    );
    let m3 = validate_model_file_with(model, None, &pairs(&[("bloq_method", "m3")]));
    assert!(error_codes(&m3).contains(&CODE), "{:#?}", m3.diagnostics);
}

/// A `method` value the parser refuses is `E_FIT_OPTION` as well — `method` takes
/// the `[fit_options]` route, not `apply_fit_option`'s.
#[test]
fn a_bad_method_override_is_e_fit_option() {
    let report = validate_model_file_with(IOV_MODEL, None, &pairs(&[("method", "nope")]));
    assert_eq!(
        error_codes(&report),
        vec!["E_FIT_OPTION"],
        "{:#?}",
        report.diagnostics
    );
    assert!(
        report.diagnostics[0].message.contains("`method = nope`"),
        "{}",
        report.diagnostics[0].message
    );
}
