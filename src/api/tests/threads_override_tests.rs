//! `--threads` beats `[fit_options] threads` (#1416).
//!
//! The defect these pin: a model file carrying `threads = 8` silently won over an
//! explicit `ferx model.ferx --threads 1`, so a benchmark pinned to one core in
//! fact ran on eight and every timing it produced was wrong by ~5×. Nothing in
//! the console output contradicted the flag.
//!
//! The merge rule under test keys on **whether the caller named the key**, not on
//! whether its value differs from `FitOptions::default()` — which is why
//! `Some(0)` ("use the engine default, and I mean it") has to override a pinned
//! `8` just as `Some(1)` does.

use super::run::apply_threads_override;
use crate::types::{classify_warning, FitOptions, WarningCode, WarningSeverity};

fn with_file_threads(threads: Option<usize>) -> FitOptions {
    FitOptions {
        threads,
        ..Default::default()
    }
}

#[test]
fn an_explicit_thread_count_beats_the_model_file_and_says_so() {
    // The reported bug, directly: the file says 8, the caller says 1.
    let mut options = with_file_threads(Some(8));
    let warning = apply_threads_override(&mut options, Some(1));

    assert_eq!(options.threads, Some(1), "the caller's count must win");
    let warning = warning.expect("a silent override is what #1416 was about");
    assert!(
        warning.contains("thread count overridden")
            && warning.contains(" 1 ")
            && warning.contains("threads = 8"),
        "the warning must name both counts, got: {warning}"
    );
}

#[test]
fn an_unmentioned_flag_leaves_the_model_file_in_charge() {
    // Mutation guard for an unconditional override: `None` is the flag being
    // absent from argv, and a library caller with no front end passes it. If this
    // reddens, `[fit_options] threads` has stopped working at all.
    let mut options = with_file_threads(Some(8));
    let warning = apply_threads_override(&mut options, None);

    assert_eq!(options.threads, Some(8));
    assert!(warning.is_none());
}

#[test]
fn an_explicitly_named_default_still_overrides_a_pinned_count() {
    // `--threads 0` / `--threads auto` is a caller who asked for the engine's own
    // worker count. Under a "does the value differ from the default" rule this
    // would be indistinguishable from an absent flag and the file's 8 would stand
    // — which is exactly the rule the fix must NOT use.
    let mut options = with_file_threads(Some(8));
    let warning = apply_threads_override(&mut options, Some(0));

    assert_eq!(options.threads, Some(0));
    let warning = warning.expect("0 vs 8 is still a disagreement");
    assert!(
        warning.contains("the default worker count") && warning.contains("threads = 8"),
        "0 must be reported by name, not as the number 0: {warning}"
    );
}

#[test]
fn agreement_is_not_a_conflict() {
    let mut options = with_file_threads(Some(4));
    assert!(apply_threads_override(&mut options, Some(4)).is_none());
    assert_eq!(options.threads, Some(4));
}

#[test]
fn an_unpinned_model_file_is_not_a_conflict() {
    // `None` and `Some(0)` both mean the file pinned nothing, so there is no
    // second opinion to report — but the override still lands.
    for from_file in [None, Some(0)] {
        let mut options = with_file_threads(from_file);
        let warning = apply_threads_override(&mut options, Some(4));
        assert_eq!(options.threads, Some(4), "from_file = {from_file:?}");
        assert!(
            warning.is_none(),
            "an unpinned file is not a disagreement (from_file = {from_file:?}), got: {warning:?}"
        );
    }
}

#[test]
fn the_override_warning_reaches_the_structured_surface_as_a_threads_warning() {
    // The message is routed by substring in `classify_warning`, so an earlier arm
    // claiming it (or a reworded message) would file it under `general` and drop
    // it out of the typed/JSON `threads` bucket.
    let mut options = with_file_threads(Some(8));
    let warning = apply_threads_override(&mut options, Some(1)).expect("warning");

    let entry = classify_warning(&warning);
    assert_eq!(entry.category, WarningCode::Threads);
    assert_eq!(entry.severity, WarningSeverity::Warning);
}
