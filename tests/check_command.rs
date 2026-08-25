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

/// #1040: a misspelled optional block used to leave `ferx check` reporting
/// `valid: true` while the fit silently ran with the default method and no
/// covariance step. It is now an error, located at the offending header.
#[test]
fn unknown_block_is_reported_as_e_unknown_block() {
    let model = temp_model(
        "unknown_block",
        &format!("{COV_MODEL}\n[fit_option]\n  method = focei\n  covariance = true\n"),
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid, "unknown block must invalidate the report");
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("fit_option"));
    assert_eq!(d.line, Some(17));
    assert_eq!(
        d.suggestion.as_deref(),
        Some("did you mean `[fit_options]`?")
    );
    assert!(
        d.message.contains("Valid blocks: "),
        "message must enumerate the valid set: {}",
        d.message
    );
    let _ = std::fs::remove_file(&model);
}

/// `[initial_values]` was ferx's own spelling for initial estimates before they
/// moved inline into `[parameters]`. The parser stopped reading it and — because
/// unknown names were dropped in silence — never said so. It gets its own code
/// and a remediation rather than a bare "unknown block", since no valid name is
/// close enough for a did-you-mean.
#[test]
fn deprecated_block_is_reported_as_e_deprecated_block() {
    let model = temp_model(
        "deprecated_block",
        &format!("{COV_MODEL}\n[initial_values]\n  theta = [0.2, 10.0]\n"),
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(
        !report.valid,
        "a deprecated block must invalidate the report"
    );
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_DEPRECATED_BLOCK");
    assert_eq!(d.block.as_deref(), Some("initial_values"));
    assert_eq!(d.line, Some(17));
    assert!(
        d.message.contains("`[parameters]`") && d.message.contains("delete it"),
        "message must name the replacement: {}",
        d.message
    );
    assert!(d.suggestion.is_none());
    let _ = std::fs::remove_file(&model);
}

/// The same rejection reaches `fit()` through `first_error`, byte-identical —
/// there is no separate path for the batch and the fail-fast callers.
#[test]
fn unknown_block_also_fails_the_parser_directly() {
    let model = temp_model(
        "unknown_block_parse",
        &format!("{COV_MODEL}\n[scalings]\n  V = 1.0\n"),
    );
    let err = match parse_full_model_file(Path::new(model.to_str().unwrap())) {
        Err(e) => e,
        Ok(_) => panic!("unknown block must fail the parse"),
    };
    assert!(err.starts_with("Unknown block `[scalings]`"), "{err}");
    assert!(err.contains("did you mean `[scaling]`"), "{err}");
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
fn dose_attr_double_use_is_reported_as_its_own_code() {
    // #993. The whole point of minting a code instead of leaving this as `E_PARSE`
    // is that a consumer (`ferxtranslate`, ferx-r) can act on it without matching
    // prose — but the mapping itself keys off a prose substring, so it needs a test
    // or a reworded message silently downgrades every parse-time case to `E_PARSE`.
    let model = temp_model(
        "dose_attr_double_use",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central * F

[error_model]
  DV ~ proportional(PROP_ERR)
",
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid);
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_DOSE_ATTR_DOUBLE_USE");
    // The message already opens with the block, so attaching one too would print it
    // twice — the renderer prefixes whatever `block` it is given.
    assert_eq!(d.block, None);
    assert!(
        d.message.starts_with("[odes]:") && d.message.contains("bioavailability"),
        "{}",
        d.message
    );
    let _ = std::fs::remove_file(&model);
}

#[test]
fn analytical_dose_attr_double_use_is_reported_as_its_own_code() {
    // #1004. The analytical message ends in a *different* remediation clause than
    // the ODE one (drop the mapping, not rename), so it needs its own sentinel in
    // `parse_error_to_diagnostic` — and therefore its own test, or the analytical
    // half silently reports `E_PARSE` while the ODE half reports the real code.
    let model = temp_model(
        "analytical_dose_attr_double_use",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  F  = TVF

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[scaling]
  obs_scale = 1000.0 / F

[error_model]
  DV ~ proportional(PROP_ERR)
",
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid);
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_DOSE_ATTR_DOUBLE_USE");
    assert_eq!(d.block, None);
    assert!(
        d.message.starts_with("[scaling]:")
            && d.message.contains("bioavailability")
            && d.message.contains("remove the `f=F` mapping"),
        "{}",
        d.message
    );
    let _ = std::fs::remove_file(&model);
}

#[test]
fn single_endpoint_sigma_order_mismatch_is_reported_as_its_own_code() {
    // #1001. `proportional(S_SMALL)` with `S_BIG` declared first used to validate
    // clean and then fit against `S_BIG`. Like #993 the code exists so a consumer
    // (`ferxtranslate`, ferx-r) can offer the mechanical fix — reorder one of the
    // two lists — without matching prose, and like #993 the mapping itself keys off
    // a prose substring, so it needs a test or a reworded message silently
    // downgrades every case to `E_PARSE`.
    let model = temp_model(
        "sigma_order_mismatch",
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma S_BIG ~ 2.0 (sd)
  sigma S_SMALL ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(S_SMALL)
",
    );
    let report = validate_model_file(model.to_str().unwrap(), None);
    assert!(!report.valid);
    // Assert the count before indexing, so a regression that reports nothing fails
    // with this line rather than an opaque index-out-of-bounds panic.
    assert_eq!(
        report.diagnostics.len(),
        1,
        "expected exactly one diagnostic, got: {:?}",
        report.diagnostics
    );
    let d = &report.diagnostics[0];
    assert_eq!(d.code, "E_SIGMA_ORDER_MISMATCH");
    // The message already opens with the block, so attaching one too would print it
    // twice — the renderer prefixes whatever `block` it is given.
    assert_eq!(d.block, None);
    assert!(
        d.message.starts_with("[error_model]")
            && d.message.contains("S_SMALL")
            && d.message.contains("S_BIG"),
        "{}",
        d.message
    );
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

/// A twin-less flip-flop transit model (an inert `[scaling] obs_scale = 1` block declines the
/// ODE-twin desugar; ke = CL/V = 0.5 ≥ KTR = (3+1)/20 = 0.2) is rejected by `ferx check` with a
/// hard `E_TRANSIT_FLIP_FLOP` error, mirroring `fit()` (#776). (A `lagtime=`/`f=` form now gets a
/// twin and auto-routes instead — see #735.)
#[test]
fn twin_less_flip_flop_transit_is_rejected_by_check() {
    let model = temp_model(
        "flipflop_notwin",
        "\
[parameters]
  theta TVCL(2.0, 0.001, 50.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[scaling]
  obs_scale = 1

[error_model]
  DV ~ proportional(PROP)
",
    );
    let data = temp_data(
        "flipflop_notwin",
        "ID,TIME,AMT,EVID,DV,MDV\n1,0,100,1,.,1\n1,2,0,1.0,0\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some(data.to_str().unwrap()));
    assert!(!report.valid, "twin-less flip-flop must fail ferx check");
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "E_TRANSIT_FLIP_FLOP")
        .expect("expected E_TRANSIT_FLIP_FLOP diagnostic");
    assert!(
        d.message.contains("flip-flop") && d.message.contains("no ODE twin"),
        "unexpected message: {}",
        d.message
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&data);
}

/// `ferx check` surfaces the structural transit rejects (here a non-depot `CMT≠1` dose) as
/// `E_TRANSIT_UNSUPPORTED`, so a clean check and a fit agree on transit models — previously
/// only `fit()` reported these (#776 review). (Steady-state dosing and infusion are no longer
/// among these rejects — they reroute to the ODE twin, #719 — so this test exercises the
/// still-unsupported non-depot dose, a permanent structural restriction of the closed form.)
#[test]
fn unsupported_transit_feature_is_rejected_by_check() {
    // In-domain (ke = 0.05 < KTR = 0.2), so this is NOT a flip-flop reject; the non-depot dose is.
    let model = temp_model(
        "transit_ss",
        "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP)
",
    );
    let data = temp_data(
        "transit_nondepot",
        "ID,TIME,AMT,EVID,DV,MDV,CMT\n1,0,100,1,.,1,2\n1,2,0,0,1.0,0,1\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some(data.to_str().unwrap()));
    assert!(
        !report.valid,
        "a non-depot (CMT≠1) dose on a transit model must fail ferx check"
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "E_TRANSIT_UNSUPPORTED"),
        "expected E_TRANSIT_UNSUPPORTED, got: {:?}",
        report
            .diagnostics
            .iter()
            .map(|d| &d.code)
            .collect::<Vec<_>>()
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&data);
}

/// `ferx check` rejects an SS dose combined with an absorption lagtime on the closed-form
/// transit model (`check_absorption_closed_form_support`'s lag branch) — the SS+lagtime
/// pre-arrival seed is still bolus-only, so this stays out of scope even though plain SS now
/// reroutes to the ODE twin (#719). (A `lagtime=` mapping also drops the twin itself, so this
/// exercises the message text rather than isolating the lag condition from the no-twin one —
/// `one_cpt_transit` has a single fixed dose route, so there is no second compartment to
/// construct a lag-elsewhere case against, unlike the ODE-path scoping covered in
/// `tests/ss_absorption_lag_gate.rs`.) Previously untested (review follow-up on PR #834): the
/// sibling `E_ABSORPTION_SS_LAG` ODE-path branch had a test but this closed-form counterpart
/// did not.
#[test]
fn ss_plus_lag_transit_is_rejected_by_check() {
    let model = temp_model(
        "transit_ss_lag",
        "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  theta TVLAG(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT
  ALAG = TVLAG

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, lagtime=ALAG)

[error_model]
  DV ~ proportional(PROP)
",
    );
    let data = temp_data(
        "transit_ss_lag",
        "ID,TIME,AMT,EVID,DV,MDV,SS,II\n1,0,100,1,.,1,1,24\n1,2,0,1.0,0,0,0\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some(data.to_str().unwrap()));
    assert!(
        !report.valid,
        "SS + lagtime transit dose must fail ferx check"
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "E_TRANSIT_UNSUPPORTED" && d.message.contains("lagtime")),
        "expected E_TRANSIT_UNSUPPORTED mentioning lagtime, got: {:?}",
        report
            .diagnostics
            .iter()
            .map(|d| (&d.code, &d.message))
            .collect::<Vec<_>>()
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&data);
}

/// `ferx check` and `fit()` must agree on which data-reader warnings a user sees.
///
/// The reader is model-blind, so a dose-free dataset raises `W_NO_DOSES` ("check
/// that the dataset has an AMT column"). For a compartment-free model (#811) that
/// is its normal shape — every MBMA dataset is dose-free — so `fit()` suppresses
/// it. `ferx check` re-emits `population.warnings` wholesale, so without the
/// shared filter it warned on every such model while `fit()` on the same pair was
/// clean.
#[test]
fn compartment_free_model_does_not_warn_about_missing_doses() {
    let report = validate_model_file(
        "examples/emax_timecourse.ferx",
        Some("data/emax_timecourse.csv"),
    );
    assert!(
        report.valid,
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.message.contains("W_NO_DOSES")),
        "a compartment-free model has nothing to dose: {:?}",
        report
            .diagnostics
            .iter()
            .map(|d| (&d.code, &d.message))
            .collect::<Vec<_>>()
    );

    // And the same pair through `fit()` — the agreement this test is about.
    let model = parse_full_model_file(Path::new("examples/emax_timecourse.ferx"))
        .expect("example parses")
        .model;
    let pop = read_nonmem_csv(Path::new("data/emax_timecourse.csv"), None, None)
        .expect("example data loads");
    let opts = FitOptions {
        outer_maxiter: 1,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
    assert!(
        !result.warnings.iter().any(|w| w.contains("W_NO_DOSES")),
        "fit() must suppress it too: {:?}",
        result.warnings
    );
}

/// A *compartment* model on a dose-free dataset must still be warned about — the
/// suppression is scoped to models that cannot be dosed, not to dose-free data.
#[test]
fn compartment_model_still_warns_about_missing_doses() {
    let model = temp_model(
        "no_doses_pk",
        "[parameters]\n  theta TVE0(10.0, 0.1, 100.0)\n  sigma PROP ~ 0.02 (sd)\n\n\
         [individual_parameters]\n  CL = TVE0\n  V = TVE0\n\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n",
    );
    let report = validate_model_file(model.to_str().unwrap(), Some("data/emax_timecourse.csv"));
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.message.contains("W_NO_DOSES")),
        "a PK model on dose-free data is still worth flagging: {:?}",
        report
            .diagnostics
            .iter()
            .map(|d| (&d.code, &d.message))
            .collect::<Vec<_>>()
    );
    let _ = std::fs::remove_file(&model);
}
