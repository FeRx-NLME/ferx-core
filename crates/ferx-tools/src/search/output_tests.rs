//! `candidates.csv` (#1178).

use ferx_core::StrictnessVerdict;

use super::*;
use crate::search::candidate::{CandidateResult, FeatureVector};
use crate::search::test_support::converged_fit;

fn result(id: &str) -> CandidateResult {
    CandidateResult {
        id: id.to_string(),
        hash: "ab".repeat(32),
        parent: Some("base".into()),
        features: FeatureVector::new().with("CL-WT", "pow"),
        fit: Some(converged_fit(12.0)),
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
    failed.criterion = f64::NAN;
    failed.error = Some("does not compile".into());
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
