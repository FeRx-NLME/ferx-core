//! The candidate journal, the fit cache and the resume manifest (#1178).

use super::*;
use crate::search::candidate::{Criterion, FeatureVector};
use crate::search::test_support::{converged_fit, population};

fn record(id: &str, hash: &str) -> CandidateRecord {
    CandidateRecord {
        id: id.to_string(),
        hash: hash.to_string(),
        parent: None,
        features: FeatureVector::new().with("CL-WT", "pow"),
        criterion: Some(12.5),
        ofv: Some(10.0),
        converged: true,
        passed: true,
        failures: vec![],
        skipped: vec![],
        seconds: 1.5,
        error: None,
        has_fit: false,
    }
}

fn options() -> RunOptions {
    RunOptions {
        criterion: Criterion::Ofv,
        n_starts: 2,
        ..RunOptions::default()
    }
}

// ── append and read ──────────────────────────────────────────────────────────

#[test]
fn rows_round_trip_in_completion_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = Journal::create(dir.path(), &[]).expect("create");
    journal.append(&record("b", "bb"), None);
    journal.append(&record("a", "aa"), None);
    journal.into_result().expect("no write failure");

    let back = read_records(&journal_path(dir.path()));
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].id, "b", "the log was reordered");
    assert_eq!(back[1].id, "a");
    assert_eq!(back[0], record("b", "bb"));
}

#[test]
fn a_truncated_or_garbled_line_is_dropped_and_the_rest_survives() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = journal_path(dir.path());
    let good = serde_json::to_string(&record("a", "aa")).expect("serialize");
    let cut = &good[..good.len() / 2];
    std::fs::write(&path, format!("{good}\nnot json at all\n{good}\n{cut}"))
        .expect("write journal");

    let back = read_records(&path);
    assert_eq!(back.len(), 2, "a malformed line took a good one with it");
    assert!(back.iter().all(|r| r.id == "a"));
}

#[test]
fn a_missing_journal_is_an_empty_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(read_records(&journal_path(dir.path())).is_empty());
}

#[test]
fn creating_the_journal_rewrites_it_from_the_kept_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = journal_path(dir.path());
    std::fs::write(&path, "garbage that must not survive\n").expect("seed");

    let kept = vec![record("a", "aa")];
    let journal = Journal::create(dir.path(), &kept).expect("create");
    journal.append(&record("b", "bb"), None);
    journal.into_result().expect("no write failure");

    let back = read_records(&path);
    assert_eq!(back.len(), 2);
    assert_eq!(back[0].id, "a");
    assert_eq!(back[1].id, "b");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(!text.contains("garbage"));
    // The `.part` sibling the rewrite went through is gone.
    assert!(!dir.path().join("search_journal.jsonl.part").exists());
}

// ── the fit cache ────────────────────────────────────────────────────────────

#[test]
fn a_cached_fit_round_trips_and_a_missing_one_is_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal = Journal::create(dir.path(), &[]).expect("create");
    let mut row = record("a", "aa");
    row.has_fit = true;
    let fit = converged_fit(42.0);
    journal.append(&row, Some(&fit));
    journal.into_result().expect("no write failure");

    let loaded = load_fit(dir.path(), "aa").expect("the cached fit");
    assert_eq!(loaded.ofv, 42.0);
    assert_eq!(loaded.theta, fit.theta);
    assert!(loaded.converged);

    assert!(load_fit(dir.path(), "never-written").is_none());
}

#[test]
fn an_unreadable_cached_fit_is_none_rather_than_a_panic() {
    // The cache is a cache: a corrupt file costs the reused candidate its
    // `FitResult`, never the run.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(fits_dir(dir.path())).expect("fits dir");
    std::fs::write(fit_path(dir.path(), "aa"), "{ not a fit }").expect("write");
    assert!(load_fit(dir.path(), "aa").is_none());
}

// ── the manifest ─────────────────────────────────────────────────────────────

#[test]
fn the_manifest_round_trips_through_disk() {
    let dir = tempfile::tempdir().expect("tempdir");
    let manifest = SearchManifest::new(&options(), &population(&["1", "2"]));
    let path = manifest_path(dir.path());
    manifest.write(&path).expect("write");
    assert_eq!(SearchManifest::read(&path).expect("read"), manifest);
    assert_eq!(manifest.criterion, "ofv");
    assert_eq!(manifest.n_starts, 2);
    assert_eq!(manifest.n_subjects, 2);
}

#[test]
fn an_identical_manifest_is_compatible() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = population(&["1", "2"]);
    let manifest = SearchManifest::new(&options(), &data);
    let disk = SearchManifest::new(&options(), &data);
    assert!(manifest.check_compatible(&disk, dir.path()).is_ok());
}

#[test]
fn each_difference_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = population(&["1", "2"]);
    let manifest = SearchManifest::new(&options(), &data);

    let cases: [(SearchManifest, &str); 4] = [
        (
            SearchManifest::new(
                &RunOptions {
                    criterion: Criterion::Aic,
                    ..options()
                },
                &data,
            ),
            "ranking criterion",
        ),
        (
            SearchManifest::new(
                &RunOptions {
                    n_starts: 5,
                    ..options()
                },
                &data,
            ),
            "starts",
        ),
        (
            SearchManifest::new(
                &RunOptions {
                    strictness: ferx_core::Strictness::none(),
                    ..options()
                },
                &data,
            ),
            "strictness",
        ),
        (
            SearchManifest::new(&options(), &population(&["1", "3"])),
            "dataset",
        ),
    ];

    for (disk, expected) in cases {
        let err = manifest
            .check_compatible(&disk, dir.path())
            .expect_err("an incompatible resume must be refused");
        assert!(
            err.contains(expected),
            "expected the message to name `{expected}`, got: {err}"
        );
    }
}

#[test]
fn the_data_fingerprint_moves_with_the_data_and_not_with_anything_else() {
    let a = SearchManifest::new(&options(), &population(&["1", "2"]));
    let same = SearchManifest::new(&options(), &population(&["1", "2"]));
    let renamed = SearchManifest::new(&options(), &population(&["1", "9"]));
    assert_eq!(a.data_fingerprint, same.data_fingerprint);
    assert_ne!(
        a.data_fingerprint, renamed.data_fingerprint,
        "a different subject set hashed the same"
    );
    assert_eq!(a.data_fingerprint.len(), 64);
}
