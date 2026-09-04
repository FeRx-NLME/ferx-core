//! Orchestration tests for [`Runner`] (#1178).
//!
//! Every test here drives [`Runner::run_with_fitter`] rather than `run`, so the
//! thing under test is the orchestration — dedup, the thread budget, the
//! journal, resume, cancellation — and not the engine. That is not a
//! convenience: "two fits are in flight at once" is not observable from outside
//! `fit()`, so the concurrency bound could not be *asserted* against real fits,
//! only assumed. The real compile-and-fit path is covered end-to-end in
//! `tests/search_runner_end_to_end.rs`.
//!
//! The fixtures — including the one genuine evaluation every `FitResult` here
//! is cloned from — live in [`crate::search::test_support`].

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ferx_core::{BicType, EstimationMethod, FitOptions, Strictness};

use super::*;
use crate::search::candidate::{Candidate, Criterion, FeatureVector, RunOptions};
use crate::search::test_support::{candidate, converged_fit, population, same_model_two_ways};

/// `RunOptions` with every gate off — the tests that are not about strictness
/// do not want an evaluation's init stall failing them.
fn lenient() -> RunOptions {
    RunOptions {
        criterion: Criterion::Ofv,
        strictness: Strictness::none(),
        n_starts: 1,
        resume: false,
        fit_options: None,
    }
}

/// A fitter that records which candidates it was asked for.
struct Recorder {
    calls: Mutex<Vec<String>>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    fn record(&self, id: &str) {
        self.calls.lock().unwrap().push(id.to_string());
    }

    fn ids(&self) -> Vec<String> {
        let mut ids = self.calls.lock().unwrap().clone();
        ids.sort();
        ids
    }
}

/// `expect_err` on a `Result<RunReport, _>` dumps every `FitResult` the report
/// holds into the panic message — megabytes of noise around a one-line failure.
fn expect_refusal(outcome: Result<RunReport, String>, what: &str) -> String {
    match outcome {
        Err(message) => message,
        Ok(report) => panic!(
            "{what}: the run was accepted ({} fitted, {} reused)",
            report.fitted, report.reused
        ),
    }
}

// ── identity ─────────────────────────────────────────────────────────────────

#[test]
fn two_candidates_with_the_same_canonical_text_are_fitted_once() {
    let (a, b) = same_model_two_ways();
    // The premise, asserted rather than assumed: the two are *not* the same
    // bytes, so a dedup that keyed on the rendered text would fit both and this
    // test would still be testing something.
    assert_ne!(a.render(), b.render(), "the fixtures are byte-identical");
    assert_eq!(a.canonical_hash(), b.canonical_hash());

    let candidates = vec![
        Candidate::new("first", a),
        Candidate::new("second", b).parent("first"),
    ];
    let recorder = Recorder::new();
    let report = Runner::new()
        .threads(1)
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |c, _| {
            recorder.record(&c.id);
            Ok(converged_fit(100.0))
        })
        .expect("run");

    assert_eq!(recorder.ids(), vec!["first"], "the duplicate was fitted");
    assert_eq!(report.fitted, 1);
    assert_eq!(report.deduped, 1);

    // Both candidates still come back — a search that generated a duplicate
    // needs to know its score, not to find its id missing from the report.
    assert_eq!(report.results.len(), 2);
    assert_eq!(report.results[1].duplicate_of.as_deref(), Some("first"));
    assert_eq!(report.results[1].criterion, report.results[0].criterion);
    assert!(report.results[1].fit.is_none());
    assert!(report.results[0].fit.is_some());
    // Provenance is the duplicate's own, not the representative's.
    assert_eq!(report.results[1].parent.as_deref(), Some("first"));
}

#[test]
fn candidates_with_different_text_are_each_fitted() {
    let candidates = vec![
        candidate("a", "[parameters]\ntheta CL = 1\n"),
        candidate("b", "[parameters]\ntheta CL = 2\n"),
    ];
    let recorder = Recorder::new();
    let report = Runner::new()
        .threads(1)
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |c, _| {
            recorder.record(&c.id);
            Ok(converged_fit(100.0))
        })
        .expect("run");

    assert_eq!(recorder.ids(), vec!["a", "b"]);
    assert_eq!(report.fitted, 2);
    assert_eq!(report.deduped, 0);
    assert!(report.results.iter().all(|r| r.duplicate_of.is_none()));
}

#[test]
fn duplicate_ids_are_rejected() {
    let candidates = vec![
        candidate("same", "[parameters]\ntheta CL = 1\n"),
        candidate("same", "[parameters]\ntheta CL = 2\n"),
    ];
    let err = expect_refusal(
        Runner::new().run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Ok(converged_fit(1.0))
        }),
        "duplicate ids must be refused",
    );
    assert!(err.contains("`same`"), "unhelpful message: {err}");
}

// ── the thread budget ────────────────────────────────────────────────────────

/// Counts concurrent fitter calls, and records whether two were ever in flight
/// at the same moment.
struct Gauge {
    live: AtomicUsize,
    max: AtomicUsize,
    paired: AtomicBool,
}

impl Gauge {
    fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            paired: AtomicBool::new(false),
        }
    }

    /// Enter, dwell for the **whole** of `dwell`, and leave.
    ///
    /// Two properties the assertions need, and one trap that cost a mutation
    /// check. Dwelling makes overlap observable at all, and recording
    /// [`paired`](Self::paired) proves the run really was concurrent, so
    /// `max <= 2` bounds something that happened rather than a run that
    /// serialised for unrelated reasons.
    ///
    /// The trap: an earlier version *left as soon as* a second caller appeared.
    /// Every worker then spent microseconds inside, so even an unbounded pool
    /// rarely had three in flight at once and `max <= 2` passed under a
    /// deliberately broken thread budget. Holding the full dwell is what makes
    /// a wider pool actually reach a wider `max`.
    fn enter(&self, dwell: Duration) {
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(live, Ordering::SeqCst);
        let deadline = Instant::now() + dwell;
        while Instant::now() < deadline {
            if self.live.load(Ordering::SeqCst) >= 2 {
                self.paired.store(true, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn the_thread_budget_bounds_concurrent_fits() {
    let candidates: Vec<Candidate> = (0..6)
        .map(|i| candidate(&format!("c{i}"), &format!("[parameters]\ntheta CL = {i}\n")))
        .collect();
    let gauge = Gauge::new();
    let report = Runner::new()
        .threads(2)
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            gauge.enter(Duration::from_millis(60));
            Ok(converged_fit(10.0))
        })
        .expect("run");

    assert_eq!(report.fitted, 6);
    assert!(
        gauge.paired.load(Ordering::SeqCst),
        "no two fits were ever in flight together: the bound below is vacuous"
    );
    assert!(
        gauge.max.load(Ordering::SeqCst) <= 2,
        "a `threads = 2` run reached {} concurrent fits",
        gauge.max.load(Ordering::SeqCst)
    );
}

#[test]
fn a_single_thread_budget_serialises_the_fits() {
    let candidates: Vec<Candidate> = (0..3)
        .map(|i| candidate(&format!("c{i}"), &format!("[parameters]\ntheta CL = {i}\n")))
        .collect();
    let gauge = Gauge::new();
    Runner::new()
        .threads(1)
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            gauge.enter(Duration::from_millis(20));
            Ok(converged_fit(10.0))
        })
        .expect("run");

    assert_eq!(
        gauge.max.load(Ordering::SeqCst),
        1,
        "a `threads = 1` run overlapped its fits"
    );
    assert!(!gauge.paired.load(Ordering::SeqCst));
}

#[test]
fn the_inner_fit_is_pinned_to_the_plans_share_of_the_budget() {
    // The other half of #1115: the outer width is not the whole story, the
    // inner `fit()` has to be told what is left for it or the two levels
    // oversubscribe. Six candidates over two threads is `2 × 1`.
    let candidates: Vec<Candidate> = (0..6)
        .map(|i| candidate(&format!("c{i}"), &format!("[parameters]\ntheta CL = {i}\n")))
        .collect();
    let seen = Mutex::new(Vec::new());
    Runner::new()
        .threads(2)
        .run_with_fitter(
            &candidates,
            &population(&["1"]),
            &lenient(),
            |_, threads| {
                seen.lock().unwrap().push(threads);
                Ok(converged_fit(10.0))
            },
        )
        .expect("run");
    assert_eq!(seen.lock().unwrap().as_slice(), &[1, 1, 1, 1, 1, 1]);

    // And with more budget than candidates, the surplus goes to the inner fit
    // rather than to idle outer workers.
    let two: Vec<Candidate> = candidates.iter().take(2).cloned().collect();
    let seen = Mutex::new(Vec::new());
    Runner::new()
        .threads(8)
        .run_with_fitter(&two, &population(&["1"]), &lenient(), |_, threads| {
            seen.lock().unwrap().push(threads);
            Ok(converged_fit(10.0))
        })
        .expect("run");
    assert_eq!(seen.lock().unwrap().as_slice(), &[4, 4]);
}

// ── strictness and failures ──────────────────────────────────────────────────

#[test]
fn a_candidate_failing_strictness_keeps_its_reasons() {
    let candidates = vec![
        candidate("good", "[parameters]\ntheta CL = 1\n"),
        candidate("bad", "[parameters]\ntheta CL = 2\n"),
    ];
    let options = RunOptions {
        criterion: Criterion::Ofv,
        strictness: Strictness {
            require_converged: true,
            ..Strictness::none()
        },
        ..lenient()
    };
    let report = Runner::new()
        .threads(1)
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            let mut result = converged_fit(10.0);
            if c.id == "bad" {
                result.converged = false;
            }
            Ok(result)
        })
        .expect("run");

    assert_eq!(report.results.len(), 2, "a failing candidate was dropped");
    let bad = &report.results[1];
    assert!(!bad.verdict.passed);
    assert_eq!(bad.verdict.failures.len(), 1);
    assert!(bad.verdict.failures[0].contains("converge"));
    assert!(!bad.eligible());
    // Its criterion is still recorded — the gate says "do not rank this", not
    // "there is nothing to see".
    assert_eq!(bad.criterion, 10.0);
    assert!(report.results[0].eligible());
    assert_eq!(report.best().map(|r| r.id.as_str()), Some("good"));
}

#[test]
fn a_failed_fit_is_reported_with_its_error() {
    let candidates = vec![candidate("broken", "[parameters]\ntheta CL = 1\n")];
    let report = Runner::new()
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Err("does not compile: unknown block `[wat]`".to_string())
        })
        .expect("run");

    let result = &report.results[0];
    assert!(result.fit.is_none());
    assert!(result.criterion.is_nan());
    assert!(!result.verdict.passed);
    assert!(result.verdict.failures[0].contains("unknown block"));
    assert_eq!(
        result.error.as_deref(),
        Some("does not compile: unknown block `[wat]`")
    );
    assert!(!result.eligible());
    assert!(report.best().is_none());
}

#[test]
fn best_is_the_lowest_criterion_among_eligible_candidates() {
    let candidates: Vec<Candidate> = (0..3)
        .map(|i| candidate(&format!("c{i}"), &format!("[parameters]\ntheta CL = {i}\n")))
        .collect();
    let options = RunOptions {
        criterion: Criterion::Bic(BicType::Fixed),
        strictness: Strictness {
            require_converged: true,
            ..Strictness::none()
        },
        ..lenient()
    };
    let report = Runner::new()
        .threads(1)
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            // c1 is the lowest OFV but did not converge, so c2 must win.
            let ofv = match c.id.as_str() {
                "c0" => 100.0,
                "c1" => 1.0,
                _ => 50.0,
            };
            let mut result = converged_fit(ofv);
            result.converged = c.id != "c1";
            Ok(result)
        })
        .expect("run");

    assert_eq!(report.best().map(|r| r.id.as_str()), Some("c2"));
}

// ── cancellation ─────────────────────────────────────────────────────────────

#[test]
fn cancellation_stops_between_candidates_and_returns_partial_results() {
    let candidates: Vec<Candidate> = (0..5)
        .map(|i| candidate(&format!("c{i}"), &format!("[parameters]\ntheta CL = {i}\n")))
        .collect();
    let flag = CancelFlag::new();
    let recorder = Recorder::new();
    let report = Runner::new()
        .threads(1)
        .cancel(flag.clone())
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |c, _| {
            recorder.record(&c.id);
            flag.cancel();
            Ok(converged_fit(1.0))
        })
        .expect("a cancelled run still reports");

    assert!(report.cancelled);
    assert_eq!(recorder.ids().len(), 1, "the run kept fitting after cancel");
    assert_eq!(report.results.len(), 1, "the finished candidate was lost");
    assert_eq!(report.results[0].id, "c0");
}

#[test]
fn a_run_cancelled_before_it_starts_fits_nothing() {
    let candidates = vec![candidate("c0", "[parameters]\ntheta CL = 1\n")];
    let flag = CancelFlag::new();
    flag.cancel();
    let report = Runner::new()
        .cancel(flag)
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            panic!("nothing may be fitted after a pre-run cancel");
        })
        .expect("run");
    assert!(report.cancelled);
    assert!(report.results.is_empty());
}

#[test]
fn a_pre_run_cancel_does_not_truncate_an_existing_journal() {
    // The reason the flag is checked *before* `Journal::create`: opening the
    // journal rewrites it, and a cancelled run must not be what destroys the
    // previous run's recovery data.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("c0", "[parameters]\ntheta CL = 1\n")];
    let runner = Runner::new().threads(1).cache_dir(dir.path());
    let mut options = lenient();
    runner
        .run_with_fitter(&candidates, &population(&["1"]), &options, |_, _| {
            Ok(converged_fit(7.0))
        })
        .expect("first run");
    let before = std::fs::read_to_string(journal::journal_path(dir.path())).expect("journal");
    assert!(!before.trim().is_empty());

    options.resume = true;
    let flag = CancelFlag::new();
    flag.cancel();
    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .cancel(flag)
        .run_with_fitter(&candidates, &population(&["1"]), &options, |_, _| {
            panic!("nothing may be fitted after a pre-run cancel")
        })
        .expect("cancelled run");

    let after = std::fs::read_to_string(journal::journal_path(dir.path())).expect("journal");
    assert_eq!(before, after, "the cancelled run rewrote the journal");
}

// ── journal and resume ───────────────────────────────────────────────────────

/// Run `ids` through a fresh runner over `dir`, returning what it fitted.
fn run_over(
    dir: &std::path::Path,
    candidates: &[Candidate],
    options: &RunOptions,
) -> (RunReport, Vec<String>) {
    let recorder = Recorder::new();
    let report = Runner::new()
        .threads(1)
        .cache_dir(dir)
        .run_with_fitter(candidates, &population(&["1"]), options, |c, _| {
            recorder.record(&c.id);
            let ofv = 100.0 + c.id.len() as f64;
            Ok(converged_fit(ofv))
        })
        .expect("run");
    (report, recorder.ids())
}

#[test]
fn a_resumed_run_refits_nothing_already_journalled() {
    let dir = tempfile::tempdir().expect("tempdir");
    let first = vec![
        candidate("a", "[parameters]\ntheta CL = 1\n"),
        candidate("bb", "[parameters]\ntheta CL = 2\n"),
    ];
    let (initial, fitted) = run_over(dir.path(), &first, &lenient());
    assert_eq!(fitted, vec!["a", "bb"]);
    assert_eq!(initial.fitted, 2);

    let mut second = first.clone();
    second.push(candidate("ccc", "[parameters]\ntheta CL = 3\n"));
    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let (resumed, fitted) = run_over(dir.path(), &second, &options);

    assert_eq!(fitted, vec!["ccc"], "a journalled candidate was refitted");
    assert_eq!((resumed.fitted, resumed.reused), (1, 2));
    assert_eq!(resumed.results.len(), 3);
    assert!(resumed.results[0].reused && resumed.results[1].reused);
    assert!(!resumed.results[2].reused);
    // The reused rows carry the criterion the interrupted run computed…
    assert_eq!(resumed.results[0].criterion, initial.results[0].criterion);
    assert_eq!(resumed.results[1].criterion, initial.results[1].criterion);
    // …and their cached `FitResult`, so a resumed run is not a degraded one.
    assert_eq!(
        resumed.results[0].fit.as_ref().map(|f| f.ofv),
        initial.results[0].fit.as_ref().map(|f| f.ofv)
    );
}

#[test]
fn a_resumed_run_matches_on_the_canonical_hash_not_the_id() {
    // The same model under a new id must still be reused: a search that renames
    // its candidates between steps (or reaches one by another route) would
    // otherwise refit everything it had already paid for.
    let dir = tempfile::tempdir().expect("tempdir");
    let (a, b) = same_model_two_ways();
    let (_, fitted) = run_over(dir.path(), &[Candidate::new("step1", a)], &lenient());
    assert_eq!(fitted, vec!["step1"]);

    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let (resumed, fitted) = run_over(dir.path(), &[Candidate::new("step2", b)], &options);
    assert!(fitted.is_empty(), "the same model was fitted twice");
    assert_eq!(resumed.reused, 1);
    assert_eq!(resumed.results[0].id, "step2");
}

#[test]
fn a_truncated_final_journal_line_costs_one_refit_and_no_more() {
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![
        candidate("a", "[parameters]\ntheta CL = 1\n"),
        candidate("bb", "[parameters]\ntheta CL = 2\n"),
    ];
    run_over(dir.path(), &candidates, &lenient());

    // Simulate a kill mid-write: chop the last line in half.
    let path = journal::journal_path(dir.path());
    let text = std::fs::read_to_string(&path).expect("journal");
    let mut lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2);
    let cut = &lines[1][..lines[1].len() / 2];
    lines[1] = cut;
    std::fs::write(&path, lines.join("\n")).expect("truncate");

    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let (resumed, fitted) = run_over(dir.path(), &candidates, &options);
    assert_eq!(fitted, vec!["bb"], "the intact row was not reused");
    assert_eq!((resumed.fitted, resumed.reused), (1, 1));

    // And the rewritten journal is well-formed again — the next resume reads
    // two rows, not one row and a fragment.
    let records = journal::read_records(&journal::journal_path(dir.path()));
    assert_eq!(records.len(), 2);
}

#[test]
fn a_resume_against_a_different_criterion_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    run_over(dir.path(), &candidates, &lenient());

    let options = RunOptions {
        resume: true,
        criterion: Criterion::Bic(BicType::Mixed),
        ..lenient()
    };
    let err = expect_refusal(
        Runner::new().cache_dir(dir.path()).run_with_fitter(
            &candidates,
            &population(&["1"]),
            &options,
            |_, _| Ok(converged_fit(1.0)),
        ),
        "resuming under another criterion must be refused",
    );
    assert!(
        err.contains("ranking criterion"),
        "unhelpful message: {err}"
    );
}

#[test]
fn a_resume_against_a_different_dataset_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    run_over(dir.path(), &candidates, &lenient());

    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let err = expect_refusal(
        Runner::new().cache_dir(dir.path()).run_with_fitter(
            &candidates,
            &population(&["1", "2"]),
            &options,
            |_, _| Ok(converged_fit(1.0)),
        ),
        "resuming against other data must be refused",
    );
    assert!(err.contains("dataset"), "unhelpful message: {err}");
}

#[test]
fn a_resume_against_the_same_shaped_but_edited_dataset_is_refused() {
    // The dangerous case, because nothing about it *looks* different: same
    // subject ids, same observation counts, one changed measurement. A
    // fingerprint over the shape alone would reuse every candidate and return
    // scores computed from the previous dataset.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];

    let mut data = population(&["1", "2"]);
    for subject in &mut data.subjects {
        subject.obs_times = vec![1.0, 4.0];
        subject.observations = vec![5.5, 2.5];
        subject.obs_cmts = vec![1, 1];
    }
    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &data, &lenient(), |_, _| {
            Ok(converged_fit(1.0))
        })
        .expect("first run");

    let mut edited = data.clone();
    edited.subjects[1].observations[0] = 5.6;
    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let err = expect_refusal(
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(&candidates, &edited, &options, |_, _| {
                Ok(converged_fit(1.0))
            }),
        "resuming against edited data must be refused",
    );
    assert!(err.contains("dataset"), "unhelpful message: {err}");

    // The premise, so the test cannot pass for the wrong reason: the edit did
    // not change the shape the old fingerprint saw.
    assert_eq!(edited.subjects.len(), data.subjects.len());
    assert_eq!(
        edited.subjects[1].observations.len(),
        data.subjects[1].observations.len()
    );
}

#[test]
fn a_resume_under_different_fit_settings_is_refused() {
    // A candidate's hash covers the `[fit_options]` in its own model text, but
    // an override handed to the runner sits outside `ModelText` — so without
    // the manifest's fit-settings fingerprint this resume would reuse scores
    // produced under another method or iteration cap.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    let with = |method: EstimationMethod, resume: bool| {
        let mut fit_options = FitOptions::default();
        fit_options.method = method;
        RunOptions {
            fit_options: Some(fit_options),
            resume,
            ..lenient()
        }
    };

    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(
            &candidates,
            &population(&["1"]),
            &with(EstimationMethod::FoceI, false),
            |_, _| Ok(converged_fit(1.0)),
        )
        .expect("first run");

    let err = expect_refusal(
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(
                &candidates,
                &population(&["1"]),
                &with(EstimationMethod::Saem, true),
                |_, _| Ok(converged_fit(1.0)),
            ),
        "resuming under other fit settings must be refused",
    );
    assert!(err.contains("fit settings"), "unhelpful message: {err}");

    // The same settings still resume, so the gate is not simply refusing
    // everything with an override.
    let (report, _) = (
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(
                &candidates,
                &population(&["1"]),
                &with(EstimationMethod::FoceI, true),
                |_, _| panic!("a journalled candidate was refitted"),
            )
            .expect("resume under the original settings"),
        (),
    );
    assert_eq!((report.fitted, report.reused), (0, 1));
}

#[test]
fn without_resume_the_journal_is_started_over() {
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    run_over(dir.path(), &candidates, &lenient());
    let (report, fitted) = run_over(dir.path(), &candidates, &lenient());

    assert_eq!(
        fitted,
        vec!["a"],
        "an unrequested resume reused a candidate"
    );
    assert_eq!(report.reused, 0);
    assert_eq!(
        journal::read_records(&journal::journal_path(dir.path())).len(),
        1,
        "the journal accumulated a second copy of the same candidate"
    );
}

// ── the table ────────────────────────────────────────────────────────────────

#[test]
fn every_candidate_appears_in_the_table_in_submission_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (a, b) = same_model_two_ways();
    let candidates = vec![
        Candidate::new("good", a).features(FeatureVector::new().with("CL-WT", "pow")),
        Candidate::new("dup", b),
        candidate("bad", "[parameters]\ntheta CL = 9\n"),
        candidate("broken", "[parameters]\ntheta CL = 8\n"),
    ];
    let options = RunOptions {
        strictness: Strictness {
            require_converged: true,
            ..Strictness::none()
        },
        ..lenient()
    };
    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            match c.id.as_str() {
                "broken" => Err("the model does not compile".to_string()),
                "bad" => {
                    let mut result = converged_fit(3.0);
                    result.converged = false;
                    Ok(result)
                }
                _ => Ok(converged_fit(1.0)),
            }
        })
        .expect("run");

    let text = std::fs::read_to_string(output::table_path(dir.path())).expect("candidates.csv");
    let rows: Vec<&str> = text.lines().collect();
    assert!(rows[0].starts_with("id,parent,hash,features,criterion"));
    assert_eq!(rows.len(), 5, "not every candidate reached the table");
    assert!(rows[1].starts_with("good,"));
    assert!(rows[1].contains("CL-WT=pow"));
    assert!(rows[2].starts_with("dup,"));
    assert!(rows[2].contains("good"), "the duplicate lost its source");
    // The two failures are present *with their reasons* — a search report that
    // silently omitted them could not be told apart from one whose candidate
    // generator never produced them.
    assert!(rows[3].starts_with("bad,"));
    assert!(rows[3].contains("converge"));
    assert!(rows[4].starts_with("broken,"));
    assert!(rows[4].contains("does not compile"));
}

#[test]
fn a_run_without_a_cache_directory_writes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let before: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert!(before.is_empty());

    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    let report = Runner::new()
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Ok(converged_fit(1.0))
        })
        .expect("run");
    assert_eq!(report.results.len(), 1);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

// ── options plumbing ─────────────────────────────────────────────────────────

#[test]
fn the_default_options_retry_and_gate() {
    // The two non-negotiables of the epic, pinned so a "simplification" cannot
    // quietly turn a search into single-start, ungated ranking.
    let options = RunOptions::default();
    assert_eq!(options.n_starts, 3);
    assert!(options.strictness.require_converged);
    assert_eq!(options.criterion, Criterion::Bic(BicType::Mixed));
    assert!(!options.resume);
    assert!(options.fit_options.is_none());
}
