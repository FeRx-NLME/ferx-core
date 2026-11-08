//! Tier-1 cells for [`FitResult::status`] (#2799), on hand-built results. Row and control
//! names follow the #2799 plan (rows 1–16, C1–C10) and its audit (E1, E2); each failure
//! message names its cell, so a mutation names the cells it kills.

use super::*;
use crate::types::test_helpers::empty_fit_result;
use crate::types::{classify_warning, HeldAtBound, ParameterKind};

const COLLAPSED: &str = "ETA_STUDY";
const COLLAPSED_ROW: usize = 3;
const N_PACKED: usize = 5;
const CRITICAL_MSG: &str = "Correlation matrix condition number 3.2e9 exceeds 1e6";
const COV_FAILED_MSG: &str = "Covariance step failed: Hessian is not positive definite";
const COLLAPSE_WARNING: &str = "Internal optimizer parameter guard reached by estimate(s): \
                                ETA_STUDY (lower guard, collapse)";

/// The guard axis after #2797 audit 1.
#[derive(Clone, Copy, Debug)]
enum Guard {
    /// No collapse.
    N,
    /// Collapse held by a clean step: `cov_fixed` marks it.
    H,
    /// Collapse, step ran (`Computed`) but did not hold it: floored `R` or `S` fallback.
    R,
    /// Collapse, step failed: no matrix, `cov_fixed` empty.
    F,
}

fn collapse(name: &str, packed_index: usize) -> HeldAtBound {
    HeldAtBound {
        name: name.to_string(),
        kind: ParameterKind::Omega,
        index: 0,
        packed_index,
        reason: HeldReason::CollapsedAtGuard,
    }
}

fn mask(held: &[usize]) -> Vec<bool> {
    (0..N_PACKED).map(|i| held.contains(&i)).collect()
}

fn with_warnings(mut r: FitResult, msgs: &[&str]) -> FitResult {
    r.warnings = msgs.iter().map(|m| m.to_string()).collect();
    r.warnings_structured = msgs.iter().map(|m| classify_warning(m)).collect();
    r
}

fn cell(critical: bool, guard: Guard, budget: bool) -> FitResult {
    let mut r = FitResult {
        covariance_status: CovarianceStatus::Computed,
        cov_fixed: mask(&[]),
        eval_budget_spent: budget,
        ..empty_fit_result()
    };
    let mut msgs: Vec<&str> = Vec::new();
    if critical {
        msgs.push(CRITICAL_MSG);
    }
    match guard {
        Guard::N => {}
        Guard::H => {
            r.held_at_bound = vec![collapse(COLLAPSED, COLLAPSED_ROW)];
            r.cov_fixed = mask(&[COLLAPSED_ROW]);
            msgs.push(COLLAPSE_WARNING);
        }
        Guard::R => {
            r.held_at_bound = vec![collapse(COLLAPSED, COLLAPSED_ROW)];
            msgs.push(COLLAPSE_WARNING);
        }
        Guard::F => {
            r.held_at_bound = vec![collapse(COLLAPSED, COLLAPSED_ROW)];
            r.covariance_status = CovarianceStatus::Failed;
            r.cov_fixed = Vec::new();
            // The step's own Critical warning rides along, as `fit()` emits it: the
            // classifier must read the failure from `covariance_status` once, not twice.
            msgs.push(COV_FAILED_MSG);
            msgs.push(COLLAPSE_WARNING);
        }
    }
    with_warnings(r, &msgs)
}

fn budget() -> StatusReason {
    StatusReason::BudgetSpent
}
fn crit() -> StatusReason {
    StatusReason::CriticalWarning(WarningCode::ConditionNumber)
}
fn cov() -> StatusReason {
    StatusReason::CovarianceFailed
}
fn guard(names: &[&str]) -> StatusReason {
    StatusReason::CollapsedAtGuard {
        parameters: names.iter().map(|s| s.to_string()).collect(),
    }
}

/// The exact reason list, and both verdicts: `converged = true` gives `Converged` exactly
/// when the list is empty, `converged = false` always `NotConverged` with the same list.
/// Returns the mismatch instead of panicking, so a table can report every cell a mutation
/// kills rather than only the first.
fn check_cell(name: &str, base: FitResult, want: Vec<StatusReason>) -> Result<(), String> {
    let mut yes = base.clone();
    yes.converged = true;
    let mut no = base;
    no.converged = false;
    let want_yes = if want.is_empty() {
        FitStatus::Converged
    } else {
        FitStatus::ConvergedWithIssues {
            reasons: want.clone(),
        }
    };
    let want_no = FitStatus::NotConverged { reasons: want };
    let (got_yes, got_no) = (yes.status(), no.status());
    if got_yes != want_yes {
        return Err(format!(
            "{name}, converged = true: got {got_yes:?}, want {want_yes:?}"
        ));
    }
    if got_no != want_no {
        return Err(format!(
            "{name}, converged = false: got {got_no:?}, want {want_no:?}"
        ));
    }
    Ok(())
}

fn assert_cell(name: &str, base: FitResult, want: Vec<StatusReason>) {
    if let Err(e) = check_cell(name, base, want) {
        panic!("{e}");
    }
}

/// Plan rows 1–16: critical × guard × budget, each with both verdicts.
#[test]
fn status_cross_product_rows_1_to_16() {
    use Guard::*;
    let s = || guard(&[COLLAPSED]);
    #[rustfmt::skip]
    let rows: Vec<(&str, bool, Guard, bool, Vec<StatusReason>)> = vec![
        ("row 1",  false, N, false, vec![]),
        ("row 2",  false, N, true,  vec![budget()]),
        ("row 3",  false, H, false, vec![]),
        ("row 4",  false, H, true,  vec![budget()]),
        ("row 5",  false, R, false, vec![s()]),
        ("row 6",  false, R, true,  vec![budget(), s()]),
        ("row 7",  false, F, false, vec![cov(), s()]),
        ("row 8",  false, F, true,  vec![budget(), cov(), s()]),
        ("row 9",  true,  N, false, vec![crit()]),
        ("row 10", true,  N, true,  vec![budget(), crit()]),
        ("row 11", true,  H, false, vec![crit()]),
        ("row 12", true,  H, true,  vec![budget(), crit()]),
        ("row 13", true,  R, false, vec![crit(), s()]),
        ("row 14", true,  R, true,  vec![budget(), crit(), s()]),
        ("row 15", true,  F, false, vec![crit(), cov(), s()]),
        ("row 16", true,  F, true,  vec![budget(), crit(), cov(), s()]),
    ];
    let failed: Vec<String> = rows
        .into_iter()
        .filter_map(|(name, critical, g, b, want)| {
            check_cell(name, cell(critical, g, b), want).err()
        })
        .collect();
    assert!(
        failed.is_empty(),
        "{} cell(s) failed:
{}",
        failed.len(),
        failed.join(
            "
"
        )
    );
}

/// C1: a failed step with no collapse — `covariance_failed` and **not** `collapsed_at_guard`
/// (the shared-arm negative).
#[test]
fn status_c1_step_failed_without_collapse() {
    let mut r = cell(false, Guard::N, false);
    r.covariance_status = CovarianceStatus::Failed;
    r.cov_fixed = Vec::new();
    let r = with_warnings(r, &[COV_FAILED_MSG]);
    assert_cell("C1", r, vec![cov()]);
}

/// C2: a `ProfiledBound` entry beside a failed step is not a collapse (#2112's informed bound).
#[test]
fn status_c2_profiled_bound_is_not_an_issue() {
    let mut r = cell(false, Guard::N, false);
    r.covariance_status = CovarianceStatus::Failed;
    r.cov_fixed = Vec::new();
    r.held_at_bound = vec![HeldAtBound {
        reason: HeldReason::ProfiledBound,
        ..collapse("PLACEBO[STUDY=7]", 1)
    }];
    let r = with_warnings(r, &[COV_FAILED_MSG]);
    assert_cell("C2", r, vec![cov()]);
}

/// C3: a collapse with no covariance step has no `R` to judge it: not an issue.
#[test]
fn status_c3_collapse_without_a_step_is_not_an_issue() {
    let mut r = cell(false, Guard::R, false);
    r.covariance_status = CovarianceStatus::NotRequested;
    r.cov_fixed = Vec::new();
    assert_cell("C3", r, vec![]);
}

/// C4: a collapse beside an SIR fallback, unmarked, is listed; the fallback is not a failure.
#[test]
fn status_c4_sir_fallback_unmarked_collapse() {
    let mut r = cell(false, Guard::R, false);
    r.covariance_status = CovarianceStatus::SirFallback;
    assert_cell("C4", r, vec![guard(&[COLLAPSED])]);
}

/// C5: a stale chain-prefixed Critical `covariance_failed` warning beside a final `Computed`
/// step raises nothing — neither `CovarianceFailed` nor `CriticalWarning(CovarianceFailed)`.
#[test]
fn status_c5_stale_covariance_failed_warning_is_not_counted() {
    let r = with_warnings(
        cell(false, Guard::N, false),
        &["[FOCE] Covariance step failed: Hessian is not positive definite"],
    );
    assert_eq!(
        (
            r.warnings_structured[0].severity.clone(),
            r.warnings_structured[0].category
        ),
        (WarningSeverity::Critical, WarningCode::CovarianceFailed),
        "C5 fixture: the stale entry must be a Critical covariance_failed, or the cell is vacuous"
    );
    assert_cell("C5", r, vec![]);
}

/// C6: a collapse beside a `Computed` step whose `cov_fixed` came back empty reads "not
/// held" — the conservative side.
#[test]
fn status_c6_empty_cov_fixed_reads_not_held() {
    let mut r = cell(false, Guard::R, false);
    r.cov_fixed = Vec::new();
    assert_cell("C6", r, vec![guard(&[COLLAPSED])]);
}

/// C7: two Critical entries of one code give that code once; a Warning-severity collapse
/// warning is not a reason.
#[test]
fn status_c7_duplicate_codes_once_and_warning_severity_ignored() {
    let r = with_warnings(
        cell(false, Guard::N, false),
        &[
            CRITICAL_MSG,
            COLLAPSE_WARNING,
            "[FOCE] Hessian is ill-conditioned",
        ],
    );
    assert_eq!(
        r.warnings_structured[1].severity,
        WarningSeverity::Warning,
        "C7 fixture: the collapse warning must be Warning-severity"
    );
    assert_eq!(
        r.warnings_structured[2].category,
        WarningCode::ConditionNumber,
        "C7 fixture: the second entry must share the first one's code"
    );
    assert_cell("C7", r, vec![crit()]);
}

/// C8: two collapses, one held and one not — only the unheld name is listed.
#[test]
fn status_c8_only_the_unheld_collapse_is_listed() {
    let mut r = cell(false, Guard::N, false);
    r.held_at_bound = vec![collapse("ETA_KA", 2), collapse(COLLAPSED, COLLAPSED_ROW)];
    r.cov_fixed = mask(&[2]);
    assert_cell("C8", r, vec![guard(&[COLLAPSED])]);
}

/// C9: a converged run carrying the multi-start failure note (whose sibling error text holds
/// Critical phrases) has no reason: the note classifies `Info`.
#[test]
fn status_c9_multi_start_note_is_not_a_reason() {
    let r = with_warnings(
        cell(false, Guard::N, false),
        &[
            "Multi-start: 1 of 4 starts failed: start 3: Outer optimizer did not converge; \
           Covariance step failed: Hessian is not positive definite",
        ],
    );
    assert_cell("C9", r, vec![]);
}

/// C10: an earlier chain stage's Critical convergence warning counts, next to a final
/// converged stage (parity with today's ferx-r `STATUS:` tail).
#[test]
fn status_c10_earlier_stage_critical_warning_counts() {
    let r = with_warnings(
        cell(false, Guard::N, false),
        &["[FOCE] Outer optimization did not converge"],
    );
    assert_eq!(
        r.warnings_structured[0].source_method.as_deref(),
        Some("FOCE")
    );
    assert_cell(
        "C10",
        r,
        vec![StatusReason::CriticalWarning(WarningCode::Convergence)],
    );
}

/// E1: an evaluation-only run with nothing else → `[evaluation_only]`.
#[test]
fn status_e1_evaluation_only() {
    let mut r = cell(false, Guard::N, false);
    r.evaluation_only = true;
    assert_cell("E1", r, vec![StatusReason::EvaluationOnly]);
}

/// E2: evaluation-only plus an unheld collapse (probe G's shape), and every reason at once
/// so the full fixed order is pinned.
#[test]
fn status_e2_evaluation_only_with_collapse_and_full_order() {
    let mut r = cell(false, Guard::R, false);
    r.evaluation_only = true;
    assert_cell(
        "E2",
        r,
        vec![StatusReason::EvaluationOnly, guard(&[COLLAPSED])],
    );
    let mut all = cell(true, Guard::F, true);
    all.evaluation_only = true;
    assert_cell(
        "E2 full order",
        all,
        vec![
            StatusReason::EvaluationOnly,
            budget(),
            crit(),
            cov(),
            s_guard(),
        ],
    );
}

fn s_guard() -> StatusReason {
    guard(&[COLLAPSED])
}

/// Audit item 9: with no budget, evaluation-only or unheld collapse, the reason labels equal
/// today's critical categories (what ferx-r's `STATUS:` tail prints) as a **set**.
#[test]
fn status_reasons_match_todays_critical_categories_as_a_set() {
    use std::collections::BTreeSet;
    let mut c1 = cell(false, Guard::N, false);
    c1.covariance_status = CovarianceStatus::Failed;
    let c1 = with_warnings(c1, &[COV_FAILED_MSG, CRITICAL_MSG]);
    let cases = [
        ("row 1", cell(false, Guard::N, false)),
        ("row 3", cell(false, Guard::H, false)),
        ("row 9", cell(true, Guard::N, false)),
        ("row 11", cell(true, Guard::H, false)),
        ("C1 + critical, failure first", c1),
    ];
    for (name, r) in cases {
        let today: BTreeSet<String> = r
            .warnings_structured
            .iter()
            .filter(|w| w.severity == WarningSeverity::Critical)
            .map(|w| w.category.as_str().to_string())
            .collect();
        let now: BTreeSet<String> = r.status().reasons().iter().map(|s| s.label()).collect();
        assert_eq!(now, today, "{name}");
    }
}

/// Each status label exact, and the three distinct (the with-issues label is not the
/// converged one with a suffix trimmed off).
#[test]
fn status_labels_are_pinned() {
    let with = FitStatus::ConvergedWithIssues {
        reasons: vec![cov()],
    };
    let not = FitStatus::NotConverged { reasons: vec![] };
    assert_eq!(FitStatus::Converged.label(), "CONVERGED");
    assert_eq!(with.label(), "CONVERGED WITH ISSUES");
    assert_eq!(not.label(), "NOT CONVERGED");
    assert_eq!(FitStatus::Converged.token(), "converged");
    assert_eq!(with.token(), "converged_with_issues");
    assert_eq!(not.token(), "not_converged");
    assert!(FitStatus::Converged.reasons().is_empty());
    assert_eq!(with.reasons(), &[cov()]);
}

/// Each reason label exact, with its negative: the collapse label names its parameters (the
/// unheld two, comma-separated) and says nothing about covariance.
#[test]
fn status_reason_labels_are_pinned() {
    assert_eq!(StatusReason::EvaluationOnly.label(), "evaluation_only");
    assert_eq!(budget().label(), "budget_spent");
    assert_eq!(crit().label(), "condition_number");
    assert_eq!(cov().label(), "covariance_failed");
    let two = guard(&["ETA_STUDY", "ETA_KA"]).label();
    assert_eq!(two, "collapsed_at_guard: ETA_STUDY, ETA_KA");
    assert!(!two.contains("covariance"), "{two}");
    // C8's form through the classifier: the held name is absent.
    let mut r = cell(false, Guard::N, false);
    r.held_at_bound = vec![collapse("ETA_KA", 2), collapse(COLLAPSED, COLLAPSED_ROW)];
    r.cov_fixed = mask(&[2]);
    let labels: Vec<String> = r.status().reasons().iter().map(|s| s.label()).collect();
    assert_eq!(labels, ["collapsed_at_guard: ETA_STUDY"]);
    assert!(!labels[0].contains("ETA_KA"), "{labels:?}");
}
