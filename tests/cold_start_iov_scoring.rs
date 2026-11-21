//! #2993: a FOCEI model that fits must also be scorable with `maxiter = 0`.
//!
//! The E7820 transit-absorption + urine model (IOV `kappa` on MTT, SS and N over six
//! occasions) used to return the failure sentinel on every no-optimizer evaluation —
//! `6e20` at its initial estimates, `2e20` at the estimates of its own fit — while the
//! second, warm-started evaluation of a fit scored normally and NONMEM scored both
//! points. One subject (ID 4) carried it: its *cold* inner solve landed where `H̃` is not
//! positive definite, the subject's objective clamped to `SUBJECT_SENTINEL_NLL`, and the
//! population OFV became `2·1e20`. `W_NONFINITE_OBJECTIVE` already named the subject and
//! the failed term (`curvature (H̃ not positive definite) = NaN`); what was missing was a
//! score.
//!
//! The fixture is that one subject (90 records, de-identified: DV simulated from the
//! model on a cloned design) at the fit's estimates, so this is a single cold evaluation —
//! Tier 2, no convergence loop.
//!
//! Bisected (first-parent, `c08b47da8..4f3ce9b`): the population evaluation first scores
//! at #2736 (`4d164d9cc`), which made `inner_restarts` reach subjects with occasions; the
//! default `inner_restarts = 1` scan moves subject 4 off the bad cold point. The cold
//! solve without the scan (`inner_restarts = 0`) scores on today's `main` as well, so both
//! routes are pinned: either one regressing returns the sentinel again.
//!
//! Mutation (run): at `9f033a523` (the parent of #2736) both tests fail with
//! `ofv = 2e20`.

use ferx_core::{fit, prepare_run, FitOptions, FitResult};

const MODEL: &str = "tests/fixtures/cold_start_iov_2993/ltbs.ferx";
const DATA: &str = "tests/fixtures/cold_start_iov_2993/data.csv";

fn evaluate(inner_restarts: Option<usize>) -> FitResult {
    let prep = prepare_run(MODEL, Some(DATA)).expect("prepare the #2993 fixture");
    assert_eq!(prep.population.subjects.len(), 1, "premise: one subject");
    assert_eq!(
        prep.parsed.fit_options.outer_maxiter, 0,
        "premise: the fixture is an evaluation, not a fit"
    );
    let base = prep.parsed.fit_options.clone();
    let opts = FitOptions {
        verbose: false,
        threads: Some(1),
        inner_restarts: inner_restarts.unwrap_or(base.inner_restarts),
        ..base
    };
    fit(&prep.parsed.model, &prep.population, &prep.init_params, &opts).expect("evaluation")
}

/// The sentinel is finite (`2e20`), so `is_finite()` cannot see it: bound the magnitude.
/// Measured on macOS arm64 at `4af057421`: `-46.637765` with the default restart scan,
/// `-37.609989` with `inner_restarts = 0` (the two land in different modes of this
/// subject's conditional density). The bound `< 0` sits ≥ 37 OFV units above either and twenty orders of
/// magnitude below the sentinel, so it tests "scored", not "scored to a given digit" — a
/// platform's libm moving the mode by a few units must not redden it.
fn assert_scored(r: &FitResult, label: &str) {
    assert!(
        r.ofv.is_finite() && r.ofv < 0.0,
        "{label}: a cold maxiter = 0 evaluation must score this subject, got ofv = {} \
         (the failure sentinel is 2e20); warnings: {:?}",
        r.ofv,
        r.warnings
    );
    assert!(
        !r.warnings.iter().any(|w| w.contains("W_NONFINITE_OBJECTIVE")),
        "{label}: no non-finite-objective demotion expected: {:?}",
        r.warnings
    );
}

#[test]
fn cold_evaluation_scores_the_iov_subject_with_the_default_restart_scan() {
    let r = evaluate(None);
    assert!(r.evaluation_only, "premise: maxiter = 0 is an evaluation-only run");
    assert_scored(&r, "default inner_restarts");
}

#[test]
fn cold_evaluation_scores_the_iov_subject_without_restarts() {
    let r = evaluate(Some(0));
    assert_scored(&r, "inner_restarts = 0");
}
