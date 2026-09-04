//! `candidates.csv` (#1178).

use ferx_core::StrictnessVerdict;

use super::*;
use crate::search::candidate::{CandidateError, CandidateResult, FeatureVector};
use crate::search::test_support::converged_fit;

fn result(id: &str) -> CandidateResult {
    CandidateResult {
        id: id.to_string(),
        hash: "ab".repeat(32),
        parent: Some("base".into()),
        features: FeatureVector::new().with("CL-WT", "pow"),
        fit: Some(converged_fit(12.0)),
        ofv: Some(12.0),
        converged: Some(true),
        verdict: StrictnessVerdict {
            passed: true,
            failures: vec![],
            skipped: vec![],
        },
        criterion: 17.0,
        seconds: 2.5,
        error: None,
        duplicate_of: None,
        reused: false,
    }
}

fn rows(dir: &std::path::Path) -> Vec<Vec<String>> {
    let mut reader = csv::Reader::from_path(table_path(dir)).expect("candidates.csv");
    let header: Vec<String> = reader
        .headers()
        .expect("header")
        .iter()
        .map(str::to_string)
        .collect();
    let mut out = vec![header];
    for record in reader.records() {
        out.push(record.expect("row").iter().map(str::to_string).collect());
    }
    out
}

#[test]
fn a_passing_candidate_fills_every_column() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_table(dir.path(), &[result("cand")]).expect("write");

    let rows = rows(dir.path());
    assert_eq!(rows[0], COLUMNS.to_vec());
    let row = &rows[1];
    let cell = |name: &str| row[COLUMNS.iter().position(|c| *c == name).unwrap()].as_str();
    assert_eq!(cell("id"), "cand");
    assert_eq!(cell("parent"), "base");
    assert_eq!(cell("hash").len(), 64);
    assert_eq!(cell("features"), "CL-WT=pow");
    assert_eq!(cell("criterion"), "17.000000");
    assert_eq!(cell("ofv"), "12.000000");
    assert_eq!(cell("converged"), "true");
    assert_eq!(cell("passed"), "true");
    assert_eq!(cell("failures"), "");
    assert_eq!(cell("seconds"), "2.500000");
    assert_eq!(cell("error"), "");
    assert_eq!(cell("duplicate_of"), "");
    assert_eq!(cell("reused"), "false");
}

#[test]
fn a_failing_candidate_carries_its_reasons_in_one_cell() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut failed = result("bad");
    failed.verdict = StrictnessVerdict {
        passed: false,
        failures: vec![
            "did not converge (`converged = false`)".into(),
            "condition number 1.2e4 exceeds 1.0e3".into(),
        ],
        skipped: vec!["correlation: no covariance matrix".into()],
    };
    write_table(dir.path(), &[failed]).expect("write");

    let rows = rows(dir.path());
    let cell = |name: &str| rows[1][COLUMNS.iter().position(|c| *c == name).unwrap()].clone();
    // The CSV writer quotes the cell, so a reason containing a comma survives
    // reading back as one field rather than splitting the row.
    assert_eq!(
        cell("failures"),
        "did not converge (`converged = false`); condition number 1.2e4 exceeds 1.0e3"
    );
    assert_eq!(cell("skipped"), "correlation: no covariance matrix");
    assert_eq!(cell("passed"), "false");
}

#[test]
fn a_candidate_without_a_fit_leaves_its_numeric_cells_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut failed = result("broken");
    failed.fit = None;
    failed.ofv = None;
    failed.converged = None;
    failed.criterion = f64::NAN;
    failed.error = Some(CandidateError::model("does not compile"));
    write_table(dir.path(), &[failed]).expect("write");

    let rows = rows(dir.path());
    let cell = |name: &str| rows[1][COLUMNS.iter().position(|c| *c == name).unwrap()].clone();
    // Not "NaN": an empty cell is what "there is no criterion" means, and it is
    // what a spreadsheet and a `read.csv` both read as missing.
    assert_eq!(cell("criterion"), "");
    assert_eq!(cell("ofv"), "");
    assert_eq!(cell("converged"), "");
    assert_eq!(cell("error"), "does not compile");
}

#[test]
fn a_reused_row_reports_its_ofv_after_the_cached_fit_is_gone() {
    // The degraded resume the journal is built for: `fits/<hash>.json` was
    // deleted or truncated, so `fit` is `None` — but the journal row it came
    // from still holds the OFV and the convergence flag, and reading those two
    // columns off `fit` alone would blank them for a candidate that fitted
    // perfectly well. This is the whole reason `ofv`/`converged` sit beside
    // `fit` rather than inside it.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut lost = result("reused");
    lost.fit = None;
    lost.reused = true;
    write_table(dir.path(), &[lost]).expect("write");

    let rows = rows(dir.path());
    let cell = |name: &str| rows[1][COLUMNS.iter().position(|c| *c == name).unwrap()].clone();
    assert_eq!(cell("ofv"), "12.000000");
    assert_eq!(cell("converged"), "true");
    assert_eq!(cell("criterion"), "17.000000");
}

#[test]
fn a_cancelled_runs_partial_table_does_not_replace_a_complete_one() {
    // `csv::Writer::from_path` truncates, so a run cancelled two candidates
    // into a resume would otherwise overwrite the 500-row table of the run it
    // resumed with a two-row one.
    let dir = tempfile::tempdir().expect("tempdir");
    let complete: Vec<CandidateResult> = ["a", "b", "c"].iter().map(|id| result(id)).collect();
    write_table(dir.path(), &complete).expect("write");
    write_partial_table(dir.path(), &[result("a")]).expect("write partial");

    assert_eq!(
        rows(dir.path()).len(),
        4,
        "the complete table was rewritten"
    );
    let mut reader =
        csv::Reader::from_path(partial_table_path(dir.path())).expect("candidates.partial.csv");
    assert_eq!(reader.records().count(), 1);
}

#[test]
fn a_completed_run_clears_the_partial_table_beside_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_partial_table(dir.path(), &[result("a")]).expect("write partial");
    write_table(dir.path(), &[result("a"), result("b")]).expect("write");
    assert!(
        !partial_table_path(dir.path()).exists(),
        "a stale partial outlived the complete table"
    );
}

#[test]
fn a_failed_write_leaves_the_previous_table_intact() {
    // The `.part` + rename is what makes this true: without it the truncate
    // happens first and a failure mid-write destroys the old table. A
    // *directory* at the temp path is a write that cannot succeed.
    let dir = tempfile::tempdir().expect("tempdir");
    write_table(dir.path(), &[result("a"), result("b")]).expect("write");
    let before = std::fs::read_to_string(table_path(dir.path())).expect("read");

    std::fs::create_dir(dir.path().join("candidates.csv.part")).expect("blocker");
    assert!(write_table(dir.path(), &[result("c")]).is_err());

    assert_eq!(
        std::fs::read_to_string(table_path(dir.path())).expect("read"),
        before,
        "a failed write truncated the previous table"
    );
}

#[test]
fn rows_keep_the_order_they_were_given_in() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results: Vec<CandidateResult> = ["c", "a", "b"].iter().map(|id| result(id)).collect();
    write_table(dir.path(), &results).expect("write");

    let rows = rows(dir.path());
    let ids: Vec<&str> = rows[1..].iter().map(|r| r[0].as_str()).collect();
    assert_eq!(ids, vec!["c", "a", "b"], "the table sorted its rows");
}

#[test]
fn writing_the_table_creates_the_directory() {
    let dir = tempfile::tempdir().expect("tempdir");
    let nested = dir.path().join("does/not/exist");
    write_table(&nested, &[result("cand")]).expect("write");
    assert!(table_path(&nested).exists());
}
