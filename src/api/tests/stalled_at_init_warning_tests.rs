//! Tier-1 tests for the `stalled_at_init` fit warning (#997 §2).
//!
//! The *predicate* (`crate::stalled_at_init`) already existed and is tested in
//! `model_selection_tests.rs`. What was missing — and what these tests pin — is
//! that a fit which never left its initial estimates now says so on the result,
//! next to `boundary_estimate`, instead of being reachable only by a caller who
//! thought to ask. A stalled fit usually reports `converged: true`, so nothing
//! else in the output distinguishes it from a good one.
use super::*;
use crate::types::test_helpers::minimal_fit_result;
use crate::types::{WarningCode, WarningSeverity};

/// A fit sitting exactly on its initial estimates, with no optimizer verdict
/// recorded (so the natural-scale reading is what fires).
fn stalled_result() -> crate::types::FitResult {
    let mut r = minimal_fit_result();
    r.theta_init = r.theta.clone();
    r.omega_init = r.omega.clone();
    r.sigma_init = r.sigma.clone();
    r.left_init = None;
    r.converged = true;
    r
}

#[test]
fn a_fit_that_moved_produces_no_warning() {
    let mut r = stalled_result();
    // One free theta 2% away from its start is enough to have left init.
    r.theta[0] = r.theta_init[0] * 1.02;
    assert!(
        stalled_at_init_warning(&r).is_none(),
        "a fit that moved must not be flagged"
    );
}

#[test]
fn a_fit_that_never_moved_is_flagged_with_its_own_category() {
    let r = stalled_result();
    let (msg, entry) = stalled_at_init_warning(&r).expect("nothing moved — this must warn");

    assert_eq!(entry.category, WarningCode::StalledAtInit);
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert!(
        msg.contains("W_STALLED_AT_INIT"),
        "the message must carry its token so re-classification recovers the code: {msg}"
    );
    // The message has to name what did not move, or the user has nothing to act
    // on — a bare "the fit stalled" is the `converged: true` problem restated.
    assert!(
        msg.contains("CL"),
        "the unmoved free THETA must be named: {msg}"
    );
}

/// `rebuild_warnings_structured` re-classifies any message it has no native
/// entry for, so the round trip has to land back on `StalledAtInit` — and
/// specifically **not** on `Convergence`. That distinction is the whole point:
/// a stalled fit typically reports `converged: true` (a fit that never moved has
/// a perfectly flat objective trace to plateau on), so a consumer branching on
/// `convergence` would read the opposite of what happened.
#[test]
fn the_message_reclassifies_to_stalled_at_init_and_not_to_convergence() {
    let (msg, _) = stalled_at_init_warning(&stalled_result()).expect("must warn");
    let reclassified = crate::types::classify_warning(&msg);
    assert_eq!(
        reclassified.category,
        WarningCode::StalledAtInit,
        "re-classifying the plain message text must recover the code, got {:?}",
        reclassified.category
    );
    assert_ne!(reclassified.category, WarningCode::Convergence);
}

/// The two readings behind the predicate report differently, because they are
/// measured in different spaces with different tolerances (the optimizer's is a
/// 1% move in *scaled packed* space, the fallback's a 1% move on the natural
/// scale). A consumer comparing one fit's verdict against another's needs to
/// know which fired.
#[test]
fn the_details_name_which_reading_produced_the_verdict() {
    let mut r = stalled_result();

    let (_, entry) = stalled_at_init_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    assert_eq!(d["verdict_source"], "natural_scale");

    // With the optimizer's own escape test recorded, that is what is read —
    // even though the natural-scale comparison would fire here too.
    r.left_init = Some(false);
    let (_, entry) = stalled_at_init_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    assert_eq!(d["verdict_source"], "optimizer_escape_test");

    // And the optimizer's verdict wins in the other direction as well: it says
    // the fit escaped, so there is no warning, despite θ/Ω/σ all reading unmoved.
    r.left_init = Some(true);
    assert!(
        stalled_at_init_warning(&r).is_none(),
        "the optimizer's own verdict is preferred when the result carries one"
    );
}

/// The `theta` payload is what an agent reads to see how far the fit is from
/// having said anything. FIXed parameters are not part of it — they were never
/// going to move, so listing them as "unmoved" is noise that makes a two-free-θ
/// stall look like a twelve-parameter one.
#[test]
fn the_details_list_free_thetas_with_their_initial_values_and_skip_fixed_ones() {
    let mut r = stalled_result();
    r.theta_fixed[1] = true; // V

    let (_, entry) = stalled_at_init_warning(&r).expect("must warn");
    let d = entry.details.as_ref().expect("details payload");
    let thetas = d["theta"].as_array().expect("theta array");

    let names: Vec<&str> = thetas
        .iter()
        .map(|t| t["parameter"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["CL", "KA"],
        "the FIXed theta must not be listed as unmoved"
    );
    assert_eq!(thetas[0]["estimate"], r.theta[0]);
    assert_eq!(thetas[0]["init"], r.theta_init[0]);
}

/// An evaluation-only run (`outer_maxiter = 0`, NONMEM `MAXEVAL=0`) reports the
/// objective at the initial estimates **because that is what was asked for**.
/// The predicate is correct there and the warning is pure noise, so it is
/// suppressed — the one case where "the estimates are the initial values" is the
/// answer rather than a symptom.
#[test]
fn an_evaluation_only_run_is_not_reported_as_a_stall() {
    let mut r = stalled_result();
    r.outer_maxiter = 0;
    assert!(stalled_at_init_warning(&r).is_none());

    // …and the suppression is scoped to that request alone: one iteration is a
    // real (if tiny) fit, and a stall there is worth saying.
    r.outer_maxiter = 1;
    assert!(stalled_at_init_warning(&r).is_some());
}

/// A fit with nothing free cannot stall, and must not be reported as if it had.
/// `n_parameters == 0` is the `FIX`-everything case — every "estimate" equals
/// its initial value by construction.
#[test]
fn a_fully_fixed_fit_is_not_a_stall() {
    let mut r = stalled_result();
    r.n_parameters = 0;
    assert!(stalled_at_init_warning(&r).is_none());
}
