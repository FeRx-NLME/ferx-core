//! `RAYON_NUM_THREADS` as a worker-count pin when nothing else pins one (#2800), and the
//! oversubscription warning for any explicit count (#1811).
//!
//! The defect: every ferx pool is built with an explicit `.num_threads(…)`, which overrides
//! Rayon's own lookup of the variable, so a container started with `--cpus 8
//! RAYON_NUM_THREADS=8` ran every fit on 7 workers and nothing said why.
//!
//! The environment is **injected** into the pure resolver — never `std::env::set_var`,
//! which races every other test in this binary. `available = 8` throughout, so the
//! automatic count is 7: an env pin of exactly 8 then tells "used as given" apart from
//! "minus one", and 16 tells it apart from the cap of 8.

use super::{resolve_fit_threads, thread_warnings, ThreadSource};
use crate::types::{classify_warning, WarningCode, WarningSeverity};

const AVAILABLE: usize = 8;
const AUTO: usize = 7; // cap_default_threads(8)

fn resolve(pinned: Option<usize>, declared: usize, env: Option<&str>) -> (usize, ThreadSource) {
    let r = resolve_fit_threads(pinned, declared, env, AVAILABLE);
    (r.n, r.source)
}

// ── the precedence matrix ────────────────────────────────────────────────────

#[test]
fn a_pinned_threads_beats_the_environment() {
    // R's `threads = 3`, `[fit_options] threads = 3`, or a CLI `--threads 3` that the
    // #1416 merge already wrote into `FitOptions::threads`. Swapping the precedence so the
    // env is consulted first reddens this.
    assert_eq!(resolve(Some(3), 0, Some("16")), (3, ThreadSource::Pinned));
    // And beats a declared process-wide width too.
    assert_eq!(resolve(Some(3), 5, Some("16")), (3, ThreadSource::Pinned));
}

#[test]
fn a_pinned_fit_never_reports_an_unusable_environment() {
    // The variable is not consulted at all for a pinned fit, so telling the user it "was
    // ignored" would blame a setting that had no say either way.
    let r = resolve_fit_threads(Some(3), 0, Some("x"), AVAILABLE);
    assert_eq!(
        (r.n, r.source, r.env_ignored),
        (3, ThreadSource::Pinned, None)
    );
}

#[test]
fn a_declared_process_wide_width_beats_the_environment() {
    assert_eq!(resolve(None, 5, Some("16")), (5, ThreadSource::Declared));
}

#[test]
fn a_valid_environment_pin_is_used_as_given_without_the_minus_one() {
    // The reported container: 8 CPUs, RAYON_NUM_THREADS=8, and the user expects 8, not 7.
    // Restoring the auto path's "minus one" on the env branch reddens this.
    assert_eq!(resolve(None, 0, Some("8")), (8, ThreadSource::Env));
}

#[test]
fn a_valid_environment_pin_is_not_capped_at_eight() {
    // Also above `available`: honoured, and warned about separately (below).
    assert_eq!(resolve(None, 0, Some("16")), (16, ThreadSource::Env));
}

#[test]
fn the_engine_default_spelling_lets_the_environment_apply() {
    // `pinned = Some(0)` is the CLI's `--threads auto` and R's `threads = 0`: "the engine
    // default", which now includes the environment. Only a positive count is a pin.
    assert_eq!(resolve(Some(0), 0, Some("6")), (6, ThreadSource::Env));
}

#[test]
fn no_environment_falls_back_to_the_capped_default() {
    let r = resolve_fit_threads(None, 0, None, AVAILABLE);
    assert_eq!(
        (r.n, r.source, r.env_ignored),
        (AUTO, ThreadSource::Auto, None)
    );
}

#[test]
fn zero_and_empty_mean_the_default_and_are_not_reported() {
    // `0` is Rayon's own documented spelling of "the default", as `threads = 0` is ferx's;
    // an empty value is the shell's `VAR= cmd` idiom for "unset". Neither is a mistake.
    for env in ["0", ""] {
        let r = resolve_fit_threads(None, 0, Some(env), AVAILABLE);
        assert_eq!(
            (r.n, r.source, r.env_ignored),
            (AUTO, ThreadSource::Auto, None),
            "env = {env:?}"
        );
    }
}

#[test]
fn an_unparseable_environment_is_ignored_and_remembered_for_the_warning() {
    // Parsed exactly as Rayon parses it (`usize::from_str`, no trimming), so a value Rayon
    // would ignore for a bootstrap's lanes is ignored here too — " 8" included.
    for env in ["x", "-2", " 8", "8 ", "2.5"] {
        let r = resolve_fit_threads(None, 0, Some(env), AVAILABLE);
        assert_eq!((r.n, r.source), (AUTO, ThreadSource::Auto), "env = {env:?}");
        assert_eq!(r.env_ignored.as_deref(), Some(env), "env = {env:?}");
    }
}

// ── the warnings: each sentence names the test that dies without it ──────────

#[test]
fn an_ignored_environment_is_reported_by_name_value_and_outcome() {
    let w = thread_warnings(AUTO, ThreadSource::Auto, Some("x"), AVAILABLE, 100, false);
    assert_eq!(w.len(), 1, "{w:?}");
    let w = &w[0];
    // What was wrong (the variable and its value), what happened to it, and what the fit
    // used instead — deleting any clause reddens one of these.
    assert!(w.contains("RAYON_NUM_THREADS = \"x\""), "{w}");
    assert!(w.contains("not a positive integer"), "{w}");
    assert!(w.contains("was ignored"), "{w}");
    assert!(w.contains("engine default of 7 thread(s)"), "{w}");
    let e = classify_warning(w);
    assert_eq!(
        (e.category, e.severity),
        (WarningCode::Threads, WarningSeverity::Warning)
    );
}

#[test]
fn no_ignored_environment_no_env_warning() {
    let w = thread_warnings(AUTO, ThreadSource::Auto, None, AVAILABLE, 100, false);
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn an_environment_pin_above_the_cpus_warns_and_says_to_change_the_variable() {
    let w = thread_warnings(16, ThreadSource::Env, None, AVAILABLE, 100, false);
    assert_eq!(w.len(), 1, "{w:?}");
    let w = &w[0];
    assert!(w.contains("16 threads requested"), "{w}");
    assert!(w.contains("(from `RAYON_NUM_THREADS`)"), "{w}");
    assert!(w.contains("exceed the 8 logical CPU(s) available"), "{w}");
    assert!(w.contains("time-share the cores"), "{w}");
    // The advice must be one the user can act on: `threads = 0` cannot lower an env pin,
    // because the env pin *is* what the engine default resolves to.
    assert!(
        w.contains("unset RAYON_NUM_THREADS or lower it to at most 8"),
        "{w}"
    );
    assert!(!w.contains("threads = 0"), "{w}");
    let e = classify_warning(w);
    assert_eq!(
        (e.category, e.severity),
        (WarningCode::Threads, WarningSeverity::Warning)
    );
}

#[test]
fn a_pinned_threads_above_the_cpus_warns_and_says_to_change_threads() {
    // #1811: the same check for a pin from `threads`, with the advice that fits it.
    let w = thread_warnings(16, ThreadSource::Pinned, None, AVAILABLE, 100, false);
    assert_eq!(w.len(), 1, "{w:?}");
    let w = &w[0];
    assert!(w.contains("(pinned by `threads`)"), "{w}");
    assert!(
        w.contains("consider threads = 0 (engine default) or at most 8"),
        "{w}"
    );
    assert!(!w.contains("unset RAYON_NUM_THREADS"), "{w}");
}

#[test]
fn a_declared_width_above_the_cpus_names_the_flag() {
    let w = thread_warnings(16, ThreadSource::Declared, None, AVAILABLE, 100, false);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(
        w[0].contains("(pinned process-wide by `--threads`)"),
        "{}",
        w[0]
    );
}

#[test]
fn a_count_at_the_cpus_does_not_warn() {
    // The boundary: `n_used == available` is a full machine, not an oversubscribed one.
    for source in [
        ThreadSource::Pinned,
        ThreadSource::Declared,
        ThreadSource::Env,
    ] {
        let w = thread_warnings(AVAILABLE, source, None, AVAILABLE, 100, false);
        assert!(w.is_empty(), "{source:?}: {w:?}");
    }
}

#[test]
fn the_automatic_count_never_warns_about_oversubscription() {
    // Only an explicit request can oversubscribe. Auto reaching `n_used > available` is the
    // ambient-pool fallback after a failed build, which the user did not ask for.
    let w = thread_warnings(16, ThreadSource::Auto, None, AVAILABLE, 100, false);
    assert!(w.is_empty(), "{w:?}");
}

#[test]
fn the_more_threads_than_subjects_note_names_where_the_count_came_from() {
    let w = thread_warnings(8, ThreadSource::Env, None, AVAILABLE, 3, false);
    assert_eq!(w.len(), 1, "{w:?}");
    let w = &w[0];
    assert!(
        w.contains("8 threads configured (from `RAYON_NUM_THREADS`) but only 3 subject(s)"),
        "{w}"
    );
    assert!(w.contains("consider threads = 3"), "{w}");
    let e = classify_warning(w);
    assert_eq!(
        (e.category, e.severity),
        (WarningCode::Threads, WarningSeverity::Info)
    );

    let auto = thread_warnings(7, ThreadSource::Auto, None, AVAILABLE, 3, false);
    assert!(auto[0].contains("(engine default)"), "{}", auto[0]);
}

#[test]
fn the_saem_note_is_unchanged() {
    let w = thread_warnings(4, ThreadSource::Auto, None, AVAILABLE, 6, true);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("Consider threads = 3 for SAEM"), "{}", w[0]);
    let none = thread_warnings(3, ThreadSource::Auto, None, AVAILABLE, 6, true);
    assert!(none.is_empty(), "{none:?}");
}
