//! The candidate journal, the fit cache and the resume manifest (#1178).

use std::collections::HashMap;

use ferx_core::{
    BloqMethod, CancelFlag, DoseEvent, EstimationMethod, FitOptions, GradientMethod, Population,
    Subject,
};

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

    let cases: [(SearchManifest, &str); 5] = [
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
        (
            SearchManifest::new(
                &RunOptions {
                    fit_options: Some(ferx_core::FitOptions::default()),
                    ..options()
                },
                &data,
            ),
            "fit settings",
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

/// One subject with every kind of content the fingerprint has to see: doses,
/// observations, compartments, censoring, occasions, and covariates.
fn loaded_subject(id: &str) -> Subject {
    let mut covariates = HashMap::new();
    covariates.insert("WT".to_string(), 70.0);
    covariates.insert("AGE".to_string(), 40.0);
    Subject {
        id: id.to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0],
        observations: vec![5.5, 2.5],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        occasions: vec![1, 1],
        covariates,
        ..Default::default()
    }
}

fn loaded_population() -> Population {
    Population {
        subjects: vec![loaded_subject("1"), loaded_subject("2")],
        covariate_names: vec!["WT".into(), "AGE".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

#[test]
fn the_fingerprint_sees_the_contents_and_not_only_the_shape() {
    // The regression this exists to catch: a fingerprint over ids and counts
    // alone leaves a resume blind to a *changed dataset of the same shape*. It
    // then reuses every candidate and returns criteria computed from the old
    // data — a wrong ranking, with nothing downstream able to detect it. Every
    // mutation below preserves the subject ids and the observation counts.
    let base = SearchManifest::new(&options(), &loaded_population());

    let mutate = |f: &dyn Fn(&mut Population)| {
        let mut data = loaded_population();
        f(&mut data);
        SearchManifest::new(&options(), &data)
    };

    let cases: [(&str, &dyn Fn(&mut Population)); 7] = [
        ("a DV value", &|d: &mut Population| {
            d.subjects[0].observations[1] = 2.6;
        }),
        ("an observation time", &|d: &mut Population| {
            d.subjects[1].obs_times[0] = 1.5;
        }),
        ("a dose amount", &|d: &mut Population| {
            d.subjects[0].doses[0].amt = 200.0;
        }),
        ("a dose compartment", &|d: &mut Population| {
            // `cmt` is private, so the whole event is rebuilt into another one.
            d.subjects[0].doses[0] = DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0);
        }),
        ("an observation compartment", &|d: &mut Population| {
            d.subjects[0].obs_cmts[1] = 2;
        }),
        ("a censoring flag", &|d: &mut Population| {
            d.subjects[0].cens[0] = 1;
        }),
        ("a covariate value", &|d: &mut Population| {
            d.subjects[1].covariates.insert("WT".into(), 90.0);
        }),
    ];

    for (what, mutation) in cases {
        let changed = mutate(mutation);
        assert_eq!(
            (changed.n_subjects, changed.n_observations),
            (base.n_subjects, base.n_observations),
            "the {what} case changed the shape, so it does not test the contents"
        );
        assert_ne!(
            changed.data_fingerprint, base.data_fingerprint,
            "changing {what} left the fingerprint unmoved"
        );
        assert!(
            base.check_compatible(&changed, std::path::Path::new("."))
                .is_err(),
            "a resume against changed {what} was accepted"
        );
    }
}

#[test]
fn the_fingerprint_is_stable_across_repeated_construction() {
    // Covariates live in `HashMap`s, whose iteration order varies per process.
    // Hashing them unsorted would produce a fresh digest on every call and
    // refuse every resume — the failure mode opposite to the one above.
    let first = SearchManifest::new(&options(), &loaded_population());
    for _ in 0..8 {
        let again = SearchManifest::new(&options(), &loaded_population());
        assert_eq!(again.data_fingerprint, first.data_fingerprint);
    }
    assert!(first
        .check_compatible(
            &SearchManifest::new(&options(), &loaded_population()),
            std::path::Path::new(".")
        )
        .is_ok());
}

// ── the fit-settings fingerprint ─────────────────────────────────────────────

fn with_fit_options(fit_options: Option<FitOptions>) -> SearchManifest {
    SearchManifest::new(
        &RunOptions {
            fit_options,
            ..options()
        },
        &loaded_population(),
    )
}

#[test]
fn the_fit_settings_fingerprint_moves_with_every_setting_that_changes_a_fit() {
    // A candidate's hash covers its own `[fit_options]` block, but not an
    // override handed to the runner — that lives outside `ModelText`. Without
    // this fingerprint a run could be resumed under another method, iteration
    // cap or censoring convention and reuse scores the override never produced.
    let base = FitOptions::default();
    let reference = with_fit_options(Some(base.clone()));

    let mutate = |f: &dyn Fn(&mut FitOptions)| {
        let mut options = base.clone();
        f(&mut options);
        with_fit_options(Some(options))
    };

    let cases: [(&str, &dyn Fn(&mut FitOptions)); 6] = [
        ("the estimation method", &|o: &mut FitOptions| {
            o.method = EstimationMethod::Saem;
        }),
        ("the outer iteration cap", &|o: &mut FitOptions| {
            o.outer_maxiter = 7;
        }),
        ("the covariance step", &|o: &mut FitOptions| {
            o.run_covariance_step = !o.run_covariance_step;
        }),
        ("the BLOQ convention", &|o: &mut FitOptions| {
            o.bloq_method = BloqMethod::M3;
        }),
        ("the gradient method", &|o: &mut FitOptions| {
            o.gradient_method = GradientMethod::Fd;
        }),
        ("the multi-start seed", &|o: &mut FitOptions| {
            o.multi_start_seed = Some(1234);
        }),
    ];

    for (what, mutation) in cases {
        let changed = mutate(mutation);
        assert_ne!(
            changed.fit_options_fingerprint, reference.fit_options_fingerprint,
            "changing {what} left the fit-settings fingerprint unmoved"
        );
        assert!(
            reference
                .check_compatible(&changed, std::path::Path::new("."))
                .is_err(),
            "a resume against changed {what} was accepted"
        );
    }
}

#[test]
fn the_fit_settings_fingerprint_ignores_this_runs_scheduling() {
    // The runner overrides these four per candidate, so they say nothing about
    // the numbers a journalled row holds. Including them would refuse resumes
    // that are in fact identical.
    let base = FitOptions::default();
    let reference = with_fit_options(Some(base.clone()));

    let mut rescheduled = base.clone();
    rescheduled.threads = Some(7);
    rescheduled.cancel = Some(CancelFlag::new());
    rescheduled.n_starts = 9;
    rescheduled.verbose = true;
    rescheduled.user_set_keys = vec!["method".to_string()];
    let same = with_fit_options(Some(rescheduled));

    assert_eq!(
        same.fit_options_fingerprint, reference.fit_options_fingerprint,
        "a run-scheduling knob leaked into the fit-settings fingerprint"
    );
    assert!(reference
        .check_compatible(&same, std::path::Path::new("."))
        .is_ok());
}

#[test]
fn no_override_and_an_override_are_not_interchangeable() {
    // `None` means "each candidate's own `[fit_options]`" — a different thing
    // from any particular override, even the default one.
    let none = with_fit_options(None);
    let default_override = with_fit_options(Some(FitOptions::default()));
    assert!(none.fit_options_fingerprint.is_none());
    assert!(default_override.fit_options_fingerprint.is_some());
    assert!(none
        .check_compatible(&default_override, std::path::Path::new("."))
        .is_err());
    assert!(default_override
        .check_compatible(&none, std::path::Path::new("."))
        .is_err());
}
