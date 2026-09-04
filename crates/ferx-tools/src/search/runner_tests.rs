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
            Err(CandidateError::model(
                "does not compile: unknown block `[wat]`",
            ))
        })
        .expect("run");

    let result = &report.results[0];
    assert!(result.fit.is_none());
    assert!(result.criterion.is_nan());
    assert!(!result.verdict.passed);
    assert!(result.verdict.failures[0].contains("unknown block"));
    assert_eq!(
        result.error.as_ref().map(|e| e.message.as_str()),
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
                "broken" => Err(CandidateError::model("the model does not compile")),
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

// ── the results outrank the files ────────────────────────────────────────────

#[test]
fn a_journal_that_cannot_be_written_warns_and_keeps_every_fit() {
    // An overnight run that hits ENOSPC on candidate 3 of 500 used to return a
    // `String` and nothing else: the fits already finished, and every one still
    // in flight, went with it. The journal is a *recovery log* derived from the
    // results — losing it costs the resume, not the run.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![
        candidate("a", "[parameters]\ntheta CL = 1\n"),
        candidate("b", "[parameters]\ntheta CL = 2\n"),
    ];
    // A plain file where the fit cache wants its directory: every `store_fit`
    // then fails from inside the parallel loop, exactly as a full disk would.
    std::fs::write(journal::fits_dir(dir.path()), b"not a directory").expect("blocker");

    let report = Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Ok(converged_fit(3.0))
        })
        .expect("a journal failure is not a run failure");

    assert_eq!(report.results.len(), 2, "the finished fits were discarded");
    assert_eq!(report.fitted, 2);
    assert!(report.results.iter().all(|r| r.fit.is_some()));
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    assert!(
        report.warnings[0].contains("not resumable"),
        "unhelpful warning: {}",
        report.warnings[0]
    );
    // …and the report the run *can* still write is written.
    assert!(output::table_path(dir.path()).exists());
}

#[test]
fn a_cancelled_run_does_not_overwrite_the_table_it_resumed() {
    // `csv::Writer::from_path` truncates, so writing a cancelled run's partial
    // rows to `candidates.csv` destroys the complete table of the run being
    // resumed — the one human-readable artefact the module promises.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![
        candidate("a", "[parameters]\ntheta CL = 1\n"),
        candidate("b", "[parameters]\ntheta CL = 2\n"),
        candidate("c", "[parameters]\ntheta CL = 3\n"),
    ];
    let (complete, _) = run_over(dir.path(), &candidates, &lenient());
    assert_eq!(complete.results.len(), 3);
    let table_before = std::fs::read_to_string(output::table_path(dir.path())).expect("table");

    let flag = CancelFlag::new();
    let report = Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .cancel(flag.clone())
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            flag.cancel();
            Ok(converged_fit(1.0))
        })
        .expect("a cancelled run still reports");
    assert!(report.cancelled);
    assert!(
        report.results.len() < 3,
        "the run was not actually cut short"
    );

    assert_eq!(
        std::fs::read_to_string(output::table_path(dir.path())).expect("table"),
        table_before,
        "the cancelled run overwrote the complete table"
    );
    // The partial rows are not thrown away either — they go beside it.
    let partial = std::fs::read_to_string(output::partial_table_path(dir.path())).expect("partial");
    assert_eq!(partial.lines().count(), report.results.len() + 1);
}

#[test]
fn a_completed_run_clears_the_partial_table_of_the_run_it_resumed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    output::write_partial_table(dir.path(), &[]).expect("stale partial");
    run_over(dir.path(), &candidates, &lenient());
    assert!(!output::partial_table_path(dir.path()).exists());
}

// ── failures are not permanent ───────────────────────────────────────────────

/// The three candidates a resume has to tell apart: one that fitted, one whose
/// failure describes the machine, and one whose failure describes the model.
fn ok_flaky_broken() -> Vec<Candidate> {
    vec![
        candidate("ok", "[parameters]\ntheta CL = 1\n"),
        candidate("flaky", "[parameters]\ntheta CL = 2\n"),
        candidate("broken", "[parameters]\ntheta CL = 3\n"),
    ]
}

fn first_run_over(dir: &std::path::Path, candidates: &[Candidate]) -> RunReport {
    Runner::new()
        .threads(1)
        .cache_dir(dir)
        .run_with_fitter(
            candidates,
            &population(&["1"]),
            &lenient(),
            |c, _| match c.id.as_str() {
                "flaky" => Err(CandidateError::environment(
                    "cannot build the fit pool: Resource temporarily unavailable",
                )),
                "broken" => Err(CandidateError::model(
                    "candidate `broken` does not compile: unknown block `[wat]`",
                )),
                _ => Ok(converged_fit(1.0)),
            },
        )
        .expect("first run")
}

#[test]
fn a_resume_refits_an_environment_failure_and_trusts_a_model_failure() {
    // The two `Err`s are one string from inside the runner, which is why the
    // classification is made where the failure is raised. Journalling a pool
    // that could not be built as final would have one bad minute mark a
    // fittable model dead for the rest of the search's life; re-parsing a model
    // that does not compile, on every resume, for ever, is the opposite waste.
    //
    // Both candidates are here so the pair straddles the predicate: a filter
    // that dropped *every* error row, or none, agrees with the fix on one of
    // them, and a test carrying only one would pass either way.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = ok_flaky_broken();
    let first = first_run_over(dir.path(), &candidates);
    assert_eq!(first.results.len(), 3);
    assert!(first.results[1].error.as_ref().is_some_and(|e| e.retryable));
    assert!(first.results[2]
        .error
        .as_ref()
        .is_some_and(|e| !e.retryable));

    let recorder = Recorder::new();
    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    let second = Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            recorder.record(&c.id);
            Ok(converged_fit(2.0))
        })
        .expect("resumed run");

    assert_eq!(
        recorder.ids(),
        vec!["flaky"],
        "the resume refit the wrong set"
    );
    assert_eq!((second.fitted, second.reused), (1, 2));
    assert!(second.results[1].error.is_none());
    assert_eq!(second.results[1].ofv, Some(2.0));
    // The model failure comes back with its reason rather than as a hole in the
    // report — it is reused, not dropped.
    let reused_failure = second.results[2].error.as_ref().expect("reason");
    assert!(reused_failure.message.contains("unknown block"));
    assert!(!reused_failure.retryable);
    assert!(second.results[2].reused);

    // A third resume reuses the refit too, so the successful retry landed in
    // the journal rather than the candidate being refitted for ever.
    let third = Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            panic!("`{}` was refitted after a successful run", c.id)
        })
        .expect("second resume");
    assert_eq!((third.fitted, third.reused), (0, 3));
}

#[test]
fn a_journal_row_written_before_the_flag_existed_is_refitted() {
    // `CandidateRecord::retryable` defaults to `true` on a row that predates
    // the field, so an old journal refits its failures instead of believing
    // them. The safe direction: a needless refit costs time, a wrongly trusted
    // failure costs the candidate.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = ok_flaky_broken();
    first_run_over(dir.path(), &candidates);

    // Strip the field from every row, as a journal written by the previous
    // version would have.
    let path = journal::journal_path(dir.path());
    let aged: String = std::fs::read_to_string(&path)
        .expect("journal")
        .lines()
        .map(|line| {
            let mut row: serde_json::Value = serde_json::from_str(line).expect("row");
            row.as_object_mut().expect("object").remove("retryable");
            format!("{row}\n")
        })
        .collect();
    assert!(!aged.contains("retryable"), "the field survived the strip");
    std::fs::write(&path, aged).expect("write");

    let recorder = Recorder::new();
    let options = RunOptions {
        resume: true,
        ..lenient()
    };
    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &options, |c, _| {
            recorder.record(&c.id);
            Ok(converged_fit(2.0))
        })
        .expect("resumed run");
    assert_eq!(recorder.ids(), vec!["broken", "flaky"]);
}

// ── the shared population ────────────────────────────────────────────────────

#[test]
fn a_model_with_no_level_block_leaves_the_population_untouched() {
    // `compile_and_fit` skips its per-candidate `Population` clone on exactly
    // the emptiness test `bind_theta_levels` early-returns on, and the two
    // cannot be allowed to drift: if the binder ever mutated a population for a
    // model with no level block, the runner would hand `fit()` an unbound one.
    // Pinned on the binder rather than on the clone because the absence of a
    // copy is not observable from outside.
    let text = std::fs::read_to_string(crate::search::test_support::MODEL).expect("model source");
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(&text).expect("parse");
    assert!(
        parsed.model.theta_blocks().level_blocks().is_empty(),
        "the fixture declares a level block, so it cannot test the skip"
    );

    let prepared = ferx_core::prepare_run(
        crate::search::test_support::MODEL,
        Some(crate::search::test_support::DATA),
    )
    .expect("warfarin model + data load");
    let before = SearchManifest::new(&lenient(), &prepared.population).data_fingerprint;
    let mut population = prepared.population.clone();
    ferx_core::bind_theta_levels(&mut parsed, &text, &mut population).expect("bind");

    assert_eq!(
        SearchManifest::new(&lenient(), &population).data_fingerprint,
        before,
        "the binder mutated a population the runner now shares between candidates"
    );
}

// ── one run per directory ────────────────────────────────────────────────────

#[test]
fn a_second_run_over_a_live_directory_is_refused_by_name() {
    // Not a nicety: `Journal::create` renames a fresh file over
    // `search_journal.jsonl`, so the run already appending keeps writing into
    // an unlinked inode and every row it produces afterwards is gone, with
    // nothing on disk to say so.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    let held = journal::DirLock::acquire(dir.path()).expect("lock");

    let err = expect_refusal(
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
                panic!("nothing may be fitted while the directory is held")
            }),
        "a second run over a live directory must be refused",
    );
    assert!(err.contains("already in use"), "unhelpful message: {err}");
    assert!(
        err.contains("delete"),
        "the message does not say how to clear a stale lock: {err}"
    );

    // Releasing it lets the next run through, so the lock is not a one-way door.
    drop(held);
    Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Ok(converged_fit(1.0))
        })
        .expect("run after the lock was released");
    assert!(
        !journal::lock_path(dir.path()).exists(),
        "the run kept the lock"
    );
}

#[test]
#[cfg(unix)]
fn a_lock_left_behind_by_a_dead_process_is_taken_over() {
    // Resuming after a hard kill is the case the journal exists for, and a hard
    // kill releases no lock file — so a lock that had to be deleted by hand
    // would break exactly the workflow the directory is built around.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];

    // A pid that is not running. Spawning and reaping a real process is what
    // makes this a *dead* pid rather than one that was never alive, which the
    // takeover has to treat the same way but which a reader could dismiss.
    let mut corpse = std::process::Command::new("true")
        .spawn()
        .expect("spawn a process to outlive");
    let dead = corpse.id();
    corpse.wait().expect("reap");
    std::fs::create_dir_all(dir.path()).expect("dir");
    std::fs::write(journal::lock_path(dir.path()), format!("pid {dead}\n")).expect("stale lock");

    let report = Runner::new()
        .threads(1)
        .cache_dir(dir.path())
        .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
            Ok(converged_fit(1.0))
        })
        .expect("a stale lock must not block a resume");
    assert_eq!(report.fitted, 1);
    assert!(!journal::lock_path(dir.path()).exists());

    // The other side of the predicate: *this* process is running, so its lock
    // is not stale and the same code path refuses instead of stealing.
    std::fs::write(
        journal::lock_path(dir.path()),
        format!("pid {}\n", std::process::id()),
    )
    .expect("live lock");
    let err = expect_refusal(
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(&candidates, &population(&["1"]), &lenient(), |_, _| {
                Ok(converged_fit(1.0))
            }),
        "a live owner must not be stolen from",
    );
    assert!(err.contains("already in use"), "unhelpful message: {err}");
    std::fs::remove_file(journal::lock_path(dir.path())).expect("cleanup");
}

#[test]
fn a_lock_whose_owner_cannot_be_read_is_never_stolen() {
    // Takeover is the only path that deletes another run's lock, so anything it
    // cannot positively establish has to fall on the refusing side — an empty
    // file (the write after `create_new` did not land) or a garbled one.
    for contents in ["", "pid \n", "held by someone", "pid not-a-number"] {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path()).expect("dir");
        std::fs::write(journal::lock_path(dir.path()), contents).expect("lock");
        let err = journal::DirLock::acquire(dir.path())
            .err()
            .unwrap_or_else(|| panic!("`{contents:?}` was treated as stale"));
        assert!(err.contains("already in use"), "unhelpful message: {err}");
        assert!(journal::lock_path(dir.path()).exists());
    }
}

#[test]
fn a_run_refused_after_it_took_the_lock_still_releases_it() {
    // The refusal has to happen *past* the acquisition or this passes for the
    // wrong reason, so it is an incompatible resume — which is checked with the
    // directory already claimed.
    let dir = tempfile::tempdir().expect("tempdir");
    let candidates = vec![candidate("a", "[parameters]\ntheta CL = 1\n")];
    run_over(dir.path(), &candidates, &lenient());

    let options = RunOptions {
        criterion: Criterion::Aic,
        resume: true,
        ..lenient()
    };
    let err = expect_refusal(
        Runner::new()
            .threads(1)
            .cache_dir(dir.path())
            .run_with_fitter(&candidates, &population(&["1"]), &options, |_, _| {
                Ok(converged_fit(1.0))
            }),
        "an incompatible resume must be refused",
    );
    assert!(
        err.contains("ranking criterion"),
        "unhelpful message: {err}"
    );
    assert!(
        !journal::lock_path(dir.path()).exists(),
        "a refused run left the directory locked"
    );
}
