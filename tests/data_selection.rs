//! Tier-2 integration tests for `[data_selection]` — IGNORE/ACCEPT filtering.
//!
//! Tests exercise the full parse → filter → fit boundary via the public API.
//! They return quickly (a small `fit()` with a handful of outer iterations) and
//! do not need the `slow-tests` gate.

use ferx_core::io::datareader::{read_nonmem_csv_filtered, SelectionFilter};
use ferx_core::io::filter_expr::RowContext;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::collections::HashMap;
use std::path::Path;

// ─── Test data ────────────────────────────────────────────────────────────────

const ONE_CPT_IV_DATA: &str = "data/one_cpt_iv.csv";

/// Minimal 1-cpt IV model — 3 outer iterations so the tests are fast.
const MODEL_SRC: &str = r"
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
  maxiter = 3
  gradient = fd
";

// ─── Parser tests ─────────────────────────────────────────────────────────────

#[test]
fn parser_populates_ignore_exprs() {
    let src =
        format!("{}\n[data_selection]\n  ignore = DV < 1.0\n  ignore = EVID != 0\n", MODEL_SRC);
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.fit_options.ignore_exprs, vec!["DV < 1.0", "EVID != 0"]);
    assert!(parsed.fit_options.accept_exprs.is_empty());
}

#[test]
fn parser_accepts_c_eq_c_and_bare_shorthand() {
    // NONMEM `IGNORE(C.EQ.C)` spellings must parse at the block level.
    let src = format!("{}\n[data_selection]\n  ignore = C == C\n  ignore = C\n", MODEL_SRC);
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.fit_options.ignore_exprs, vec!["C == C", "C"]);
}

#[test]
fn parser_populates_accept_exprs() {
    let src = format!("{}\n[data_selection]\n  accept = DV >= 1.0\n", MODEL_SRC);
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.fit_options.accept_exprs, vec!["DV >= 1.0"]);
    assert!(parsed.fit_options.ignore_exprs.is_empty());
}

#[test]
fn parser_populates_ignore_subjects() {
    let src = format!("{}\n[data_selection]\n  ignore_subjects = [1, 3]\n", MODEL_SRC);
    let parsed = parse_full_model(&src).expect("parse ok");
    assert_eq!(parsed.fit_options.ignore_subjects, vec!["1", "3"]);
}

#[test]
fn parser_deduplicates_identical_exprs() {
    let src =
        format!("{}\n[data_selection]\n  ignore = DV < 1.0\n  ignore = DV < 1.0\n", MODEL_SRC);
    let parsed = parse_full_model(&src).expect("parse ok");
    // Second identical line must be silently deduplicated.
    assert_eq!(parsed.fit_options.ignore_exprs, vec!["DV < 1.0"]);
}

#[test]
fn parser_rejects_pipe_or() {
    let src = format!("{}\n[data_selection]\n  ignore = DV < 1.0 || EVID == 0\n", MODEL_SRC);
    assert!(parse_full_model(&src).is_err(), "expected parse error for ||");
}

#[test]
fn parser_rejects_unknown_key() {
    let src = format!("{}\n[data_selection]\n  bad_key = something\n", MODEL_SRC);
    assert!(parse_full_model(&src).is_err(), "expected parse error for unknown key");
}

// ─── SelectionFilter unit tests ──────────────────────────────────────────────

fn empty_cov() -> HashMap<String, f64> {
    HashMap::new()
}

fn empty_str_cov() -> &'static HashMap<String, String> {
    static M: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();
    M.get_or_init(HashMap::new)
}

fn obs_ctx<'a>(id: &'a str, dv: f64, cov: &'a HashMap<String, f64>) -> RowContext<'a> {
    RowContext {
        id,
        time: 1.0,
        dv,
        evid: 0,
        amt: 0.0,
        cmt: 1,
        rate: 0.0,
        mdv: 0,
        cens: 0,
        ii: 0.0,
        ss: false,
        covariates: cov,
        str_covariates: empty_str_cov(),
    }
}

#[test]
fn filter_ignore_excludes_matching_row() {
    let f = SelectionFilter::from_opts(&["DV < 1.0".to_string()], &[], &[]).expect("ok");
    let cov = empty_cov();
    let (excluded, which) = f.should_exclude(&obs_ctx("1", 0.5, &cov));
    assert!(excluded);
    assert!(which.unwrap().contains("DV < 1.0"));
}

#[test]
fn filter_ignore_passes_non_matching_row() {
    let f = SelectionFilter::from_opts(&["DV < 1.0".to_string()], &[], &[]).expect("ok");
    let cov = empty_cov();
    let (excluded, _) = f.should_exclude(&obs_ctx("1", 2.0, &cov));
    assert!(!excluded);
}

#[test]
fn filter_accept_excludes_row_that_fails_condition() {
    let f = SelectionFilter::from_opts(&[], &["DV >= 1.0".to_string()], &[]).expect("ok");
    let cov = empty_cov();
    let (excluded, which) = f.should_exclude(&obs_ctx("1", 0.5, &cov));
    assert!(excluded);
    assert!(which.unwrap().contains("DV >= 1.0"));
}

#[test]
fn filter_accept_passes_row_that_meets_condition() {
    let f = SelectionFilter::from_opts(&[], &["DV >= 1.0".to_string()], &[]).expect("ok");
    let cov = empty_cov();
    let (excluded, _) = f.should_exclude(&obs_ctx("1", 2.0, &cov));
    assert!(!excluded);
}

#[test]
fn filter_ignore_subjects_excludes_whole_subject() {
    let f = SelectionFilter::from_opts(&[], &[], &["5".to_string()]).expect("ok");
    let cov = empty_cov();
    let (excluded, _) = f.should_exclude(&obs_ctx("5", 1.0, &cov));
    assert!(excluded);
}

#[test]
fn filter_ignore_subjects_passes_other_subject() {
    let f = SelectionFilter::from_opts(&[], &[], &["5".to_string()]).expect("ok");
    let cov = empty_cov();
    let (excluded, _) = f.should_exclude(&obs_ctx("1", 1.0, &cov));
    assert!(!excluded);
}

// ─── End-to-end: read with filter → fit ─────────────────────────────────────

#[test]
fn read_filtered_population_excludes_low_dv_obs() {
    let path = Path::new(ONE_CPT_IV_DATA);
    if !path.exists() {
        eprintln!("Skipping: {} not found", ONE_CPT_IV_DATA);
        return;
    }

    // Without filter: 90 obs rows across 10 subjects.
    let unfiltered = read_nonmem_csv(path, None, None).expect("read ok");
    let n_obs_all = unfiltered.n_obs();

    // With ignore = DV < 1.0: should exclude several obs rows.
    let filter =
        SelectionFilter::from_opts(&["DV < 1.0".to_string()], &[], &[]).expect("filter ok");
    let filtered = read_nonmem_csv_filtered(path, None, None, &filter).expect("filtered read ok");

    let excl = filtered.exclusions.as_ref().expect("exclusions present");
    // one_cpt_iv.csv has 100 non-header rows.
    assert_eq!(excl.n_records_total, 100);
    assert!(excl.n_obs_excluded > 0, "at least one obs row with DV < 1.0 must be excluded");
    // Dose rows carry DV='.' (missing → NaN), so `DV < 1.0` must NOT catch them.
    assert_eq!(
        excl.n_dose_excluded, 0,
        "dose rows have missing DV and must not match a DV comparison"
    );
    assert!(filtered.n_obs() < n_obs_all, "filtered population has fewer observations");
    assert!(
        excl.fired_ignore.iter().any(|s| s.contains("DV < 1.0")),
        "fired_ignore must record the matching clause"
    );
}

#[test]
fn ignore_c_eq_c_drops_comment_rows() {
    // Mirror NONMEM's `$DATA data.csv IGNORE(C.EQ.C)`: a `C` label column whose
    // value is the literal "C" marks a comment row that must be dropped, while
    // numeric "0" rows are kept. Two equivalent spellings are exercised:
    // `ignore = C == C` and the bare shorthand `ignore = C`.
    let mut path = std::env::temp_dir();
    path.push("ferx_ignore_c_eq_c.csv");
    let csv = "C,ID,TIME,DV,EVID,AMT\n\
               0,1,0,.,1,100\n\
               0,1,1,5.0,0,.\n\
               C,1,2,999,0,.\n\
               0,1,3,2.0,0,.\n";
    std::fs::write(&path, csv).expect("write temp csv");

    // Baseline: no filter → 3 observation rows (incl. the comment row).
    let all = read_nonmem_csv(&path, None, None).expect("read ok");
    assert_eq!(all.n_obs(), 3);

    for clause in ["C == C", "C"] {
        let filter =
            SelectionFilter::from_opts(&[clause.to_string()], &[], &[]).expect("filter ok");
        let filtered = read_nonmem_csv_filtered(&path, None, None, &filter).expect("read ok");
        assert_eq!(
            filtered.n_obs(),
            2,
            "clause `{clause}` must drop exactly the C-labelled comment row"
        );
        let excl = filtered.exclusions.as_ref().expect("exclusions present");
        assert_eq!(excl.n_obs_excluded, 1, "clause `{clause}`: one obs excluded");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ignore_subject_removes_subject_entirely() {
    let path = Path::new(ONE_CPT_IV_DATA);
    if !path.exists() {
        eprintln!("Skipping: {} not found", ONE_CPT_IV_DATA);
        return;
    }

    let unfiltered = read_nonmem_csv(path, None, None).expect("read ok");
    let n_subjects_all = unfiltered.subjects.len();

    let filter = SelectionFilter::from_opts(&[], &[], &["3".to_string()]).expect("ok");
    let filtered = read_nonmem_csv_filtered(path, None, None, &filter).expect("read ok");

    assert_eq!(
        filtered.subjects.len(),
        n_subjects_all - 1,
        "one fewer subject after ignore_subjects = [3]"
    );
    let excl = filtered.exclusions.as_ref().expect("exclusions present");
    assert!(
        excl.excluded_subject_ids.contains(&"3".to_string()),
        "excluded_subject_ids must contain '3'"
    );
}

#[test]
fn fit_result_carries_exclusions() {
    let path = Path::new(ONE_CPT_IV_DATA);
    if !path.exists() {
        eprintln!("Skipping: {} not found", ONE_CPT_IV_DATA);
        return;
    }

    let model = ferx_core::parse_model_string(MODEL_SRC).expect("parse ok");
    let filter =
        SelectionFilter::from_opts(&["DV < 1.0".to_string()], &[], &[]).expect("filter ok");
    let pop = read_nonmem_csv_filtered(path, None, None, &filter).expect("read ok");

    let mut opts = FitOptions::default();
    opts.outer_maxiter = 2;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit ok");
    let excl = result.exclusions.as_ref().expect("exclusions on FitResult");
    assert!(excl.n_obs_excluded > 0, "fit result carries positive obs exclusion count");
}
