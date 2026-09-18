//! Tests for the bounded-admission lane runner (#1329).
//!
//! The load-bearing one is [`the_shape_this_replaces_admits_more_jobs_than_its_width`]:
//! it drives the **old** spelling — `width` Rayon workers running a `par_iter`
//! over the jobs — through the same probe and shows it admits more than `width`
//! live jobs. Without it, [`at_most_width_jobs_are_live_at_once`] is green for a
//! reason nobody has checked, since "the peak never exceeded the bound" is also
//! what a run with no concurrency at all reports.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use ferx_core::PoolPlan;
use rayon::prelude::*;

use super::run_in_lanes;

/// Long enough that a lane which *could* be joined by a sibling has every
/// chance to be, short enough that the one failing path — no concurrency where
/// the test demands it — does not dominate a `--lib` run. Only reached when the
/// assertion is about to fail: as soon as `target` jobs are live, every waiter
/// is woken at once.
const DEADLINE: Duration = Duration::from_secs(1);

/// A job that records how many of its peers are running alongside it.
///
/// Each call marks itself live, enters a nested pool **the way
/// [`ferx_core::fit`] does** — that nesting is the whole mechanism under test,
/// since it is the blocked outer worker that steals — waits there until `target`
/// jobs are concurrently live, and then leaves.
struct LiveProbe {
    /// `(live, peak)` under one lock so the peak cannot miss an overlap that
    /// two separate counters would let slip between them.
    state: Mutex<(usize, usize)>,
    woken: Condvar,
    calls: Mutex<Vec<usize>>,
}

impl LiveProbe {
    fn new() -> Self {
        Self {
            state: Mutex::new((0, 0)),
            woken: Condvar::new(),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Returns whether `target` jobs were ever live while this one was.
    fn run(&self, i: usize, target: usize) -> bool {
        self.calls.lock().unwrap().push(i);
        {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            state.1 = state.1.max(state.0);
            self.woken.notify_all();
        }

        let reached = PoolPlan::new(1, 1)
            .install(|| {
                let started = Instant::now();
                let mut state = self.state.lock().unwrap();
                while state.0 < target {
                    let Some(left) = DEADLINE.checked_sub(started.elapsed()) else {
                        break;
                    };
                    // Releases the lock while waiting, so the peers this job is
                    // waiting *for* can take it and increment.
                    let (next, timeout) = self.woken.wait_timeout(state, left).unwrap();
                    state = next;
                    if timeout.timed_out() {
                        break;
                    }
                }
                state.0 >= target
            })
            .expect("nested pool");

        let mut state = self.state.lock().unwrap();
        state.0 -= 1;
        self.woken.notify_all();
        reached
    }

    fn peak(&self) -> usize {
        self.state.lock().unwrap().1
    }

    fn called(&self) -> Vec<usize> {
        let mut calls = self.calls.lock().unwrap().clone();
        calls.sort_unstable();
        calls
    }
}

/// The bound, at three widths. `peak == width` rather than `peak <= width`:
/// `<=` is also satisfied by a runner that never overlaps anything, so it would
/// pass on an implementation that had quietly become serial.
#[test]
fn at_most_width_jobs_are_live_at_once() {
    for width in [1usize, 2, 4] {
        let probe = LiveProbe::new();
        let n_jobs = 4 * width;
        let _ = run_in_lanes(width, n_jobs, |i| probe.run(i, width).then_some(i)).expect("lanes");

        assert_eq!(
            probe.peak(),
            width,
            "width {width}: {} jobs were live at once, not {width}",
            probe.peak()
        );
        // Not read off the returned results: a job that timed out waiting for
        // the last round's missing peers returns `None`, which is a scheduling
        // artefact of the probe rather than a lane that lost a job.
        assert_eq!(
            probe.called(),
            (0..n_jobs).collect::<Vec<_>>(),
            "width {width}: every job index must be claimed exactly once"
        );
    }
}

/// The defect this module exists to remove, driven through the same probe: one
/// Rayon worker, a `par_iter` over the jobs, and each job nesting a pool the way
/// `fit()` does. The worker blocked on a nested `install` steals the next job, so
/// two are live on one worker.
///
/// If this ever goes green the premise of the lane runner is wrong and
/// [`at_most_width_jobs_are_live_at_once`] is pinning nothing, which is exactly
/// when a red test is wanted.
#[test]
fn the_shape_this_replaces_admits_more_jobs_than_its_width() {
    let probe = LiveProbe::new();
    let jobs: Vec<usize> = (0..8).collect();
    PoolPlan::new(1, 1)
        .install(|| {
            jobs.par_iter()
                .filter_map(|&i| probe.run(i, 2).then_some(i))
                .collect::<Vec<_>>()
        })
        .expect("outer pool");

    assert!(
        probe.peak() >= 2,
        "a one-worker pool running a `par_iter` of pool-nesting jobs peaked at {} \
         live jobs; the admission defect (#1329) is that a worker blocked on the \
         nested `install` keeps stealing, so it should exceed its width of 1",
        probe.peak()
    );
}

/// Every index exactly once, and the results in index order — the two
/// properties every caller's journalling, seed mapping and final assembly
/// already relied on `par_iter().filter_map().collect()` for.
#[test]
fn every_job_runs_once_and_results_return_in_index_order() {
    let seen = Mutex::new(Vec::new());
    let out = run_in_lanes(4, 50, |i| {
        seen.lock().unwrap().push(i);
        Some(i * 10)
    })
    .expect("lanes");

    assert_eq!(out, (0..50).map(|i| i * 10).collect::<Vec<_>>());
    let mut seen = seen.into_inner().unwrap();
    seen.sort_unstable();
    assert_eq!(seen, (0..50).collect::<Vec<_>>());
}

/// `None` is dropped and does not shift the surviving results, so a cancelled or
/// flag-unwound job still leaves the rest in order.
#[test]
fn none_results_are_dropped_without_disturbing_the_order() {
    let out = run_in_lanes(3, 10, |i| (i % 3 == 0).then_some(i)).expect("lanes");
    assert_eq!(out, vec![0, 3, 6, 9]);
}

/// A wider budget than there is work spawns one lane per job, not per budget
/// unit, and an empty queue runs no lanes at all.
#[test]
fn width_is_clamped_to_the_work_available() {
    let calls = AtomicUsize::new(0);
    let out = run_in_lanes(64, 2, |i| {
        calls.fetch_add(1, Ordering::Relaxed);
        Some(i)
    })
    .expect("lanes");
    assert_eq!(out, vec![0, 1]);
    assert_eq!(calls.load(Ordering::Relaxed), 2);

    let none: Vec<usize> = run_in_lanes(8, 0, |_| unreachable!("no jobs to run")).expect("lanes");
    assert!(none.is_empty());
}

/// Width `0` would mean "let the runtime decide", the ambiguity this function
/// removes; it is one lane, not zero (which would drop every job silently).
#[test]
fn zero_width_is_one_lane_rather_than_no_lanes() {
    let out = run_in_lanes(0, 5, Some).expect("lanes");
    assert_eq!(out, vec![0, 1, 2, 3, 4]);
}

/// A panicking job took the whole run with it under `par_iter`; it still does.
/// Silently dropping it would turn a bug in a fit into a bootstrap that reports
/// 199 of 200 replicates and no reason for the missing one.
#[test]
fn a_job_panic_reaches_the_caller() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_in_lanes(2, 6, |i| {
            assert_ne!(i, 4, "job 4 is the scripted panic");
            Some(i)
        })
    }));
    let payload = result.expect_err("the panic should have crossed the lane boundary");
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .expect("the panic payload should survive the re-raise");
    assert!(
        message.contains("job 4 is the scripted panic"),
        "the original payload should be re-raised, not a lane-join error: {message}"
    );
}
