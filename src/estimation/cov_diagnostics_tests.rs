//! Tier-1 tests for the covariance regularization diagnostic (#520, reworked for #1508's
//! review).
//!
//! The message is code, so its **input space** is enumerated rather than sampled:
//!
//! | axis | values |
//! |---|---|
//! | route | analytic R-matrix, FD stencil, **hybrid** (#1514's per-subject salvage) |
//! | severity | minor, moderate, severe |
//! | which magnitude carried the grade | inflation only, indefiniteness only, both, neither |
//! | declined clauses | none, one with a remedy, one without, **several** (mixed) |
//! | ODE tolerance | never integrates, `[odes]`, closed form **reaching its ODE twin**, at/under the plateau |
//!
//! and every conditional sentence is asserted on **both sides of its own gate inside one
//! test**, so a gate stuck on one branch reddens rather than passing half. Each test names the
//! sentence it exists to kill; deleting that sentence from
//! [`super::format_regularized_warning`] must redden the named test and no other.
//!
//! None of this runs a fit: the facts are a plain struct and the formatter is pure, which is
//! the whole reason the split exists.

use super::*;
use crate::types::{classify_warning, DoseEvent, Subject, WarningCode, WarningSeverity};

/// A regularization that is genuinely benign: the spectrum is non-negative, the smallest
/// eigenvalue sits just under the floor, and the floor barely moves the reported variance.
fn benign_facts() -> CovRegularizationFacts<'static> {
    CovRegularizationFacts {
        source: CovHessianSource::FdStencil,
        n_clipped: 1,
        n_free: 13,
        min_eigenvalue: 1.0e-8,
        max_eigenvalue: 1.0e4,
        floor: 1.0e-6,
        variance_inflation: 1.001,
        declines: &[],
        ode: None,
    }
}

/// The 2026-09-15 measurement from #520: 1 of 13 eigenvalues clipped — 7%, which the old
/// count-fraction grading called "minor … Standard errors are likely reliable" — next to an SE
/// inflated 4400× (so a variance inflated ~1.9e7×) and a `%RSE` of 24982.
fn measured_severe_facts() -> CovRegularizationFacts<'static> {
    CovRegularizationFacts {
        source: CovHessianSource::FdStencil,
        n_clipped: 1,
        n_free: 13,
        min_eigenvalue: -3.197e1,
        max_eigenvalue: 8.416e3,
        floor: 8.416e-7,
        variance_inflation: 1.9e7,
        declines: &[],
        ode: None,
    }
}

// ── C1: severity is a magnitude, not a count ────────────────────────────────────────────

#[test]
fn severity_is_graded_on_magnitude_not_on_the_clipped_count() {
    // The C1 regression. Both cells clip exactly 1 of 13 eigenvalues — identical count
    // fraction, 7%, which is the *only* input the pre-#520 grading had. Under that grading
    // both printed "minor. Standard errors are likely reliable."; they must now disagree,
    // which is the property a count-based implementation cannot have.
    let benign = benign_facts();
    let measured = measured_severe_facts();
    assert_eq!(benign.n_clipped, measured.n_clipped);
    assert_eq!(benign.n_free, measured.n_free);

    assert_eq!(benign.grade().severity, CovSeverity::Minor);
    assert_eq!(
        measured.grade().severity,
        CovSeverity::Severe,
        "|min eig|/max eig = {:.2e}, variance inflation = {:.2e} must not grade as minor",
        measured.neg_ratio(),
        measured.variance_inflation,
    );

    // And the sentence the user reads, not only the enum: the exact string #520 reported.
    let msg = format_regularized_warning(&measured);
    assert!(
        !msg.contains("Standard errors are likely reliable"),
        "{msg}"
    );
    assert!(msg.contains("severity: severe"), "{msg}");
    assert!(format_regularized_warning(&benign).contains("severity: minor"));
}

#[test]
fn a_negative_eigenvalue_alone_grades_severe() {
    // Kills the `neg_ratio` leg of `grade`. Variance inflation is pinned at 1.0, so
    // deleting the `neg_ratio` comparison leaves this at Minor. Paired with the test below,
    // this is what stops the two legs covering for each other (AGENTS.md's redundant-gate
    // hole): each leg is asserted with the other one inert.
    assert_eq!(grade(1e-3, 1.0).severity, CovSeverity::Severe);
    assert_eq!(grade(1e-7, 1.0).severity, CovSeverity::Moderate);
    assert_eq!(grade(1e-10, 1.0).severity, CovSeverity::Minor);
    // And the cause is attributed to the leg that fired, which is what selects the
    // interpretation (#1508 review §6).
    assert_eq!(grade(1e-3, 1.0).cause, CovSeverityCause::Indefiniteness);
    assert_eq!(grade(1e-7, 1.0).cause, CovSeverityCause::Indefiniteness);
    assert_eq!(grade(1e-10, 1.0).cause, CovSeverityCause::Neither);
}

#[test]
fn variance_inflation_alone_grades_severe() {
    // Kills the `variance_inflation` leg. `neg_ratio = 0` is the reachable near-singular case:
    // the whole spectrum is positive, nothing is indefinite, and the floor still manufactured
    // the reported variance.
    assert_eq!(grade(0.0, 1.0e6).severity, CovSeverity::Severe);
    assert_eq!(grade(0.0, 2.0).severity, CovSeverity::Moderate);
    assert_eq!(grade(0.0, 1.01).severity, CovSeverity::Minor);
    assert_eq!(grade(0.0, 1.0e6).cause, CovSeverityCause::Inflation);
    assert_eq!(grade(0.0, 2.0).cause, CovSeverityCause::Inflation);
    // Unbounded inflation — a coordinate whose entire variance came from floored directions.
    assert_eq!(grade(0.0, f64::INFINITY).severity, CovSeverity::Severe);
}

#[test]
fn both_magnitudes_crossing_is_its_own_cause() {
    // The fourth cell of the cause enum, and the one that keeps the `(true, true)` arm from
    // being unreachable: collapsing it into either single-leg arm reddens here.
    assert_eq!(grade(1e-3, 1.0e6).cause, CovSeverityCause::Both);
    assert_eq!(grade(1e-7, 2.0).cause, CovSeverityCause::Both);
}

#[test]
fn severity_boundaries_are_straddled_on_both_sides() {
    // Each threshold asserted just below and just above, so a `>` silently becoming `>=` (or
    // a constant drifting) reddens rather than shifting one cell quietly.
    assert_eq!(grade(1e-6, 1.0).severity, CovSeverity::Moderate);
    assert_eq!(grade(1.01e-6, 1.0).severity, CovSeverity::Severe);
    assert_eq!(grade(1e-9, 1.0).severity, CovSeverity::Minor);
    assert_eq!(grade(1.01e-9, 1.0).severity, CovSeverity::Moderate);
    assert_eq!(grade(0.0, 4.0).severity, CovSeverity::Moderate);
    assert_eq!(grade(0.0, 4.01).severity, CovSeverity::Severe);
    assert_eq!(grade(0.0, 1.02).severity, CovSeverity::Minor);
    assert_eq!(grade(0.0, 1.03).severity, CovSeverity::Moderate);
}

#[test]
fn severity_tiers_match_the_calibration_table() {
    // #1508 review §5: the thresholds are measured, not picked, and this replays the measured
    // table from the module's threshold comment through `grade`. A constant that drifts away
    // from the measurement reddens here, naming the row it stopped matching.
    //
    // The rows are the 2026-09-21 sweep on the stiff fixture's FD-route free-block Hessian
    // (its smallest eigenvalue varied through the floor, everything else as measured) plus the
    // two endpoint fits. `realised` is the worst relative error of a reported standard error
    // against the exact inverse of the same Hessian — an independent truth, not the inflation's
    // own reference — and the sweep confirmed `realised = √inflation − 1` to a worst relative
    // residual of 4.2e-3 over four orders of magnitude.
    struct Row {
        label: &'static str,
        neg_ratio: f64,
        inflation: f64,
        /// Worst realised relative error of a reported SE, as a fraction (1.0 = 100 %).
        realised: f64,
        want: CovSeverity,
    }
    let rows = [
        Row {
            label: "lam/floor 1e4",
            neg_ratio: 1.878e-3,
            inflation: 1.5775e5,
            realised: 397.86,
            want: CovSeverity::Severe,
        },
        Row {
            label: "lam/floor 1e3",
            neg_ratio: 1.878e-3,
            inflation: 1.5777e4,
            realised: 124.66,
            want: CovSeverity::Severe,
        },
        Row {
            label: "lam/floor 1e2",
            neg_ratio: 1.878e-3,
            inflation: 1.5786e3,
            realised: 38.733,
            want: CovSeverity::Severe,
        },
        Row {
            label: "lam/floor 10",
            neg_ratio: 1.878e-3,
            inflation: 1.5876e2,
            realised: 11.600,
            want: CovSeverity::Severe,
        },
        Row {
            label: "lam/floor 3",
            neg_ratio: 1.878e-3,
            inflation: 4.8327e1,
            realised: 5.9517,
            want: CovSeverity::Severe,
        },
        Row {
            label: "lam/floor 1",
            neg_ratio: 1.878e-3,
            inflation: 1.6776e1,
            realised: 3.0958,
            want: CovSeverity::Severe,
        },
        Row {
            label: "stiff/analytic",
            neg_ratio: 5.65e-7,
            inflation: 8.816e4,
            realised: 295.9,
            want: CovSeverity::Severe,
        },
        Row {
            label: "stiff/fd",
            neg_ratio: 4.65e-3,
            inflation: 1.366e10,
            realised: 1.169e5,
            want: CovSeverity::Severe,
        },
        // The third endpoint fit: the 3-cpt IV ODE example on the FD covariance route at the
        // default tolerance, 180 subjects, `analytic_cov_hessian = false` (so the declined
        // clause it names is `Disabled`, the one whose remedy is the sole blocker).
        Row {
            label: "3cpt/fd",
            neg_ratio: 3.70e-3,
            inflation: 3.920e8,
            realised: 1.9799e4,
            want: CovSeverity::Severe,
        },
    ];
    for row in &rows {
        assert!(
            row.realised.is_finite() && row.inflation.is_finite(),
            "{}: a recorded measurement must be a number, not a sentinel",
            row.label
        );
        // The law the cuts are drawn on, re-derived from the recorded pair rather than
        // restated: if it ever stops holding, the thresholds stop meaning a stated SE error.
        let predicted = row.inflation.sqrt() - 1.0;
        let residual = (row.realised - predicted).abs() / row.realised;
        assert!(
            residual < 5e-3,
            "{}: realised {:.4e} vs sqrt(inflation) - 1 = {predicted:.4e} (residual {residual:.1e})",
            row.label,
            row.realised,
        );
        assert_eq!(
            grade(row.neg_ratio, row.inflation).severity,
            row.want,
            "{}: neg_ratio {:.2e}, inflation {:.3e}, realised worst SE error {:.1}% must grade \
             {:?}",
            row.label,
            row.neg_ratio,
            row.inflation,
            row.realised * 100.0,
            row.want,
        );
    }

    // The two cuts, stated as the SE errors they were drawn at, so a constant edited without
    // the comment reddens. `MODERATE_INFLATION` is the 1 % line and `SEVERE_INFLATION` the
    // 100 % line, both through `realised = sqrt(inflation) - 1`.
    let at = |realised: f64| (1.0 + realised).powi(2);
    assert_eq!(grade(0.0, at(0.01) * 1.001).severity, CovSeverity::Moderate);
    assert_eq!(grade(0.0, at(0.01) * 0.999).severity, CovSeverity::Minor);
    assert_eq!(grade(0.0, at(1.00) * 1.001).severity, CovSeverity::Severe);
    assert_eq!(grade(0.0, at(1.00) * 0.999).severity, CovSeverity::Moderate);
    // And the headroom the cut was given: the ambient reproducibility of a reported SE on a
    // clean fit, measured at 1.9e-4 relative on the transit tolerance sweep, must stay minor.
    assert_eq!(grade(0.0, at(1.9e-4)).severity, CovSeverity::Minor);
}

#[test]
fn severe_inflation_threshold_supports_the_word_mostly() {
    // The severe/inflation interpretation says the affected SEs "come mostly from the floor".
    // That is a claim about a number: the inflation is `var_returned / var_unfloored`, so
    // "mostly" (more than half of the variance being floor-derived) needs the threshold above
    // 2.0. Lowering `SEVERE_INFLATION` below that makes the sentence false before any test of
    // the wording itself would notice, which is why the constant is asserted rather than the
    // prose. Measured cut is 4.0 — a reported variance 4× too big, i.e. an SE 2× too big.
    let just_over = grade(0.0, 4.01);
    assert_eq!(just_over.severity, CovSeverity::Severe);
    assert_eq!(just_over.cause, CovSeverityCause::Inflation);
    assert!(
        just_over
            .interpretation()
            .contains("come mostly from the floor"),
        "{}",
        just_over.interpretation()
    );
    // 4.01 means at least 75% of the returned variance is floor-derived — comfortably
    // "mostly". A threshold at or below 2.0 would break that, so pin the frontier:
    assert_eq!(
        grade(0.0, 2.0).severity,
        CovSeverity::Moderate,
        "a variance only doubled by the floor is not 'mostly from the floor'"
    );
}

#[test]
fn neg_ratio_is_zero_on_a_non_negative_spectrum() {
    // `|min λ| / λ_max` is only meaningful when the Hessian is actually indefinite. A tiny
    // positive minimum is near-singular, not indefinite, and must not be reported as a
    // negative ratio (nor, via `abs()`, as a large one).
    let mut f = benign_facts();
    assert_eq!(f.neg_ratio(), 0.0);
    f.min_eigenvalue = -1.0;
    f.max_eigenvalue = 100.0;
    assert!((f.neg_ratio() - 0.01).abs() < 1e-15);
}

// ── The label fix: the route the message names ──────────────────────────────────────────

#[test]
fn the_message_names_the_route_that_was_actually_floored() {
    // The label bug from #520: "eigenvalue floor applied to FD Hessian" was printed on the
    // analytic R-matrix route too, where no objective stencil and no `fd_hessian_step` exist.
    // Both sides of the gate in one test — the facts differ only in `source`.
    let mut analytic = measured_severe_facts();
    analytic.source = CovHessianSource::AnalyticRMatrix;
    let fd = measured_severe_facts();

    let m_analytic = format_regularized_warning(&analytic);
    let m_fd = format_regularized_warning(&fd);

    assert!(m_analytic.contains("the analytic R-matrix"), "{m_analytic}");
    assert!(!m_analytic.contains("FD Hessian"), "{m_analytic}");
    assert!(m_fd.contains("the FD Hessian"), "{m_fd}");
    assert!(!m_fd.contains("analytic R-matrix ("), "{m_fd}");
}

// ── C2: the declined gate clauses ───────────────────────────────────────────────────────

#[test]
fn the_declined_clause_sentence_is_fd_route_only() {
    // Both sides of the route gate on one decline value. On the analytic route there is no
    // clause to name — the analytic route *is* what a clause would have declined — so the
    // sentence must be absent, not merely differently worded.
    let mut fd = measured_severe_facts();
    fd.declines = &[CovScopeDecline::ExpressionScale];
    let mut analytic = fd;
    analytic.source = CovHessianSource::AnalyticRMatrix;

    assert!(format_regularized_warning(&fd).contains("declined because"));
    assert!(!format_regularized_warning(&analytic).contains("declined because"));
}

#[test]
fn a_clause_with_a_remedy_prints_it_and_one_without_prints_none() {
    // Both sides of the `remedy()` gate in one test. `obs_scale` is the clause the #520
    // review singled out: a one-line rewrite moves the fit to the analytic route, and a user
    // who is not told that has no way to find it.
    let mut with = measured_severe_facts();
    with.declines = &[CovScopeDecline::ExpressionScale];
    let m = format_regularized_warning(&with);
    assert!(m.contains("[scaling] obs_scale = ... is in use"), "{m}");
    assert!(m.contains("[scaling] y = central / V"), "{m}");
    assert!(m.contains("moves the fit onto the analytic route"), "{m}");

    let mut without = measured_severe_facts();
    without.declines = &[CovScopeDecline::LogTransform];
    let m = format_regularized_warning(&without);
    assert!(m.contains("the model is log-transform-both-sides."), "{m}");
    // No invented advice: the FD stencil is the correct route for an LTBS model.
    assert!(!m.contains("moves the fit onto the analytic route"), "{m}");
    assert!(!m.contains("stays on the finite-difference route"), "{m}");
}

#[test]
fn a_remedy_only_promises_a_route_change_when_it_clears_every_clause() {
    // #1508 review §3, and both sides of the gate in one test. The old wording was
    // unconditional, so a model that declined for *two* reasons was told that clearing one of
    // them "moves the fit onto the analytic route" — which it does not.
    //
    // Sole blocker, with a remedy: the promise is true and must be made.
    let mut sole = measured_severe_facts();
    sole.declines = &[CovScopeDecline::GradientFd];
    let m = format_regularized_warning(&sole);
    assert!(
        m.contains("Dropping gradient = fd moves the fit onto the analytic route."),
        "{m}"
    );

    // Same remedy, but a second clause that no flag clears (a non-Gaussian endpoint is a
    // property of the model the user asked for). The promise must NOT be made, and the
    // remaining blocker must be named as such.
    let mut multi = measured_severe_facts();
    multi.declines = &[
        CovScopeDecline::GradientFd,
        CovScopeDecline::NonGaussianEndpoint,
    ];
    let m = format_regularized_warning(&multi);
    assert!(
        m.contains("gradient = fd and the model has a non-Gaussian"),
        "{m}"
    );
    assert!(
        !m.contains("moves the fit onto the analytic route"),
        "the second clause is still blocking: {m}"
    );
    assert!(
        m.contains("stays on the finite-difference route until all of them are cleared"),
        "{m}"
    );
    assert!(
        m.contains("the remaining clause has no one-line remedy"),
        "{m}"
    );
}

#[test]
fn two_remediable_clauses_promise_the_route_only_together() {
    // The third cell: every clause is remediable, but none of them alone is enough. The verb
    // has to agree with that — "together move", not "moves" — and dropping the plural arm
    // leaves a sentence that promises each action does it on its own.
    let mut facts = measured_severe_facts();
    facts.declines = &[CovScopeDecline::Disabled, CovScopeDecline::GradientFd];
    let m = format_regularized_warning(&facts);
    assert!(
        m.contains(
            "Setting analytic_cov_hessian = true and dropping gradient = fd together move the \
             fit onto the analytic route."
        ),
        "{m}"
    );
    // Both clauses are named, joined, and the sentence is one sentence.
    assert!(
        m.contains("declined because analytic_cov_hessian = false and gradient = fd."),
        "{m}"
    );
}

#[test]
fn several_unremediable_clauses_invent_no_advice() {
    // The fourth cell: more than one clause, none with a remedy. The list is still printed —
    // a user who knows *why* can decide whether to change the model — but nothing is promised.
    let mut facts = measured_severe_facts();
    facts.declines = &[
        CovScopeDecline::LogTransform,
        CovScopeDecline::Frem,
        CovScopeDecline::Mixture,
    ];
    let m = format_regularized_warning(&facts);
    assert!(
        m.contains(
            "the model is log-transform-both-sides, the model is a FREM model and the model is \
             a mixture model."
        ),
        "{m}"
    );
    assert!(!m.contains("moves the fit onto the analytic route"), "{m}");
    assert!(!m.contains("one-line remedy"), "{m}");
}

#[test]
fn two_remediable_clauses_next_to_an_unremediable_one_name_the_plural_forms() {
    // Plural on both halves at once — two actions, one residual blocker. Dropping either
    // plural arm leaves an agreement error in a user-visible sentence, which no test on the
    // singular cells can see.
    let mut facts = measured_severe_facts();
    facts.declines = &[
        CovScopeDecline::Disabled,
        CovScopeDecline::GradientFd,
        CovScopeDecline::LogTransform,
    ];
    let m = format_regularized_warning(&facts);
    assert!(
        m.contains(
            "Setting analytic_cov_hessian = true and dropping gradient = fd clear those clauses"
        ),
        "{m}"
    );
    assert!(
        m.contains("the remaining clause has no one-line remedy"),
        "{m}"
    );

    // And two residual blockers: "clauses have", not "clause has".
    let mut two_left = measured_severe_facts();
    two_left.declines = &[
        CovScopeDecline::GradientFd,
        CovScopeDecline::LogTransform,
        CovScopeDecline::Frem,
    ];
    let m = format_regularized_warning(&two_left);
    assert!(
        m.contains("Dropping gradient = fd clears that clause"),
        "{m}"
    );
    assert!(
        m.contains("the remaining clauses have no one-line remedy"),
        "{m}"
    );
}

#[test]
fn every_decline_clause_has_non_empty_prose() {
    // A variant added to the gate without message text would otherwise reach a user as an
    // empty "declined because ." — the failure mode a `#[non_exhaustive]`-style match cannot
    // catch, since every arm compiles.
    for decline in CovScopeDecline::ALL {
        assert!(!decline.clause().is_empty(), "{decline:?}");
        let mut facts = measured_severe_facts();
        let one = [decline];
        facts.declines = &one;
        let msg = format_regularized_warning(&facts);
        assert!(msg.contains(decline.clause()), "{decline:?}: {msg}");
        assert!(!msg.contains("because ."), "{decline:?}: {msg}");
        // Every message, whatever the clause, still classifies as the same warning code.
        let entry = classify_warning(&msg);
        assert_eq!(entry.severity, WarningSeverity::Warning, "{decline:?}");
        assert_eq!(
            entry.category,
            WarningCode::CovarianceRegularized,
            "{decline:?}"
        );
    }
}

#[test]
fn a_remedy_is_an_action_not_a_promise() {
    // The split #1508 review §3 forced: `remedy()` returns the action alone, and the promise
    // lives in the formatter where the number of clauses is known. A remedy string that
    // smuggles the promise back in would make the multi-clause sentence self-contradictory.
    for decline in CovScopeDecline::ALL {
        if let Some(action) = decline.remedy() {
            assert!(
                !action.contains("moves the fit"),
                "{decline:?}: the promise belongs to the formatter, not to the action: {action}"
            );
        }
    }
}

// ── C2: the ODE tolerance sentence ──────────────────────────────────────────────────────

fn ode_facts(reltol: f64, abstol: f64) -> OdeToleranceFacts {
    OdeToleranceFacts {
        reltol,
        abstol,
        via_twin: false,
    }
}

#[test]
fn the_ode_tolerance_sentence_straddles_the_plateau() {
    // Both sides of `looser_than_cov_plateau` in one test. At the default the sentence is
    // true and must appear; at the plateau the advice "tighten to 1e-6 / 1e-8" would be
    // advice to change nothing, so the sentence must be absent.
    let mut loose = measured_severe_facts();
    loose.ode = Some(ode_facts(1e-4, 1e-6));
    let m = format_regularized_warning(&loose);
    assert!(m.contains("ode_reltol = 1e-4 / ode_abstol = 1e-6"), "{m}");
    assert!(m.contains("amplifies integration noise by 1/h"), "{m}");

    let mut tight = measured_severe_facts();
    tight.ode = Some(ode_facts(1e-6, 1e-8));
    assert!(!format_regularized_warning(&tight).contains("amplifies integration noise"));

    // A half-tightened fit is still looser than the plateau on one of the two knobs, and is
    // told so — the gate is `||`, not `&&`.
    let mut half = measured_severe_facts();
    half.ode = Some(ode_facts(1e-6, 1e-6));
    assert!(format_regularized_warning(&half).contains("amplifies integration noise"));
}

#[test]
fn the_ode_tolerance_sentence_is_fd_route_only() {
    // The analytic R-matrix never second-differences the *objective*, so the `1/h²`
    // amplification argument does not apply to it — the #520 measurement is flat across eight
    // orders of magnitude there. Same ODE facts, both routes, one test.
    let mut fd = measured_severe_facts();
    fd.ode = Some(ode_facts(1e-4, 1e-6));
    let mut analytic = fd;
    analytic.source = CovHessianSource::AnalyticRMatrix;

    assert!(format_regularized_warning(&fd).contains("amplifies integration noise"));
    assert!(!format_regularized_warning(&analytic).contains("amplifies integration noise"));
}

#[test]
fn a_model_that_never_integrates_is_told_nothing_about_ode_tolerances() {
    // `ode: None` is the never-integrates cell. No sentence, on either route.
    let facts = measured_severe_facts();
    assert!(facts.ode.is_none());
    let m = format_regularized_warning(&facts);
    assert!(!m.contains("ode_reltol"), "{m}");
}

#[test]
fn the_tolerance_sentence_names_the_twin_when_the_twin_is_what_integrates() {
    // #1508 review §4's user-visible half. A `one_cpt_transit` model has no `[odes]` block, so
    // "this model integrates ODEs" would be wrong even though the tolerance genuinely applies
    // — `sync_ode_solver_opts` stamps it onto the absorption twin. Both sides of `via_twin` in
    // one test, so collapsing the two labels reddens.
    let mut own = measured_severe_facts();
    own.ode = Some(OdeToleranceFacts {
        reltol: 1e-4,
        abstol: 1e-6,
        via_twin: false,
    });
    let mut twin = measured_severe_facts();
    twin.ode = Some(OdeToleranceFacts {
        reltol: 1e-4,
        abstol: 1e-6,
        via_twin: true,
    });
    let m_own = format_regularized_warning(&own);
    let m_twin = format_regularized_warning(&twin);
    assert!(m_own.contains("this model integrates ODEs at"), "{m_own}");
    assert!(!m_own.contains("absorption ODE twin"), "{m_own}");
    assert!(
        m_twin.contains("this model's closed-form absorption ODE twin integrates at"),
        "{m_twin}"
    );
    // Both still carry the advice, since the mechanism is identical.
    assert!(m_own.contains("amplifies integration noise"), "{m_own}");
    assert!(m_twin.contains("amplifies integration noise"), "{m_twin}");
}

// ── The head sentence, and the token the warning taxonomy keys on ───────────────────────

#[test]
fn the_head_sentence_carries_the_classification_token_and_the_numbers() {
    // `classify_warning` routes `WarningCode::CovarianceRegularized` on the substring
    // "covariance step regularized"; rewording the head without this test would silently
    // reclassify the warning (the next arm in the chain, "ill-conditioned", would not catch
    // it either, so it would fall through to the generic tail).
    let msg = format_regularized_warning(&measured_severe_facts());
    assert!(msg.starts_with("Covariance step regularized:"), "{msg}");
    let entry = classify_warning(&msg);
    assert_eq!(entry.severity, WarningSeverity::Warning);
    assert_eq!(entry.category, WarningCode::CovarianceRegularized);
    // The magnitudes the grade was made from are in the message, so a reader can check it.
    assert!(
        msg.contains("1 of 13 free-block eigenvalues clipped"),
        "{msg}"
    );
    assert!(msg.contains("min eig = -3.197e1"), "{msg}");
    assert!(msg.contains("max eig = 8.416e3"), "{msg}");
    assert!(msg.contains("|min eig|/max eig ="), "{msg}");
    assert!(
        msg.contains("worst inflation of a reported variance ="),
        "{msg}"
    );
}

#[test]
fn unbounded_inflation_is_worded_not_printed_as_inf() {
    // `f64::INFINITY` is reachable (a reported parameter loading entirely on floored
    // directions) and `{:.3e}` would render it as `inf`, which reads as a bug rather than as
    // a finding.
    let mut facts = measured_severe_facts();
    facts.variance_inflation = f64::INFINITY;
    let msg = format_regularized_warning(&facts);
    assert!(msg.contains("= unbounded;"), "{msg}");
    // "inf" as a standalone rendering, not the substring inside "inflation"/"confidence".
    assert!(!msg.contains("= inf;"), "{msg}");
    assert!(!msg.contains("floor = inf"), "{msg}");
}

#[test]
fn each_severity_tier_carries_its_own_interpretation() {
    // Three distinct sentences, and the severe one must not be reachable from the minor text.
    // Deleting `CovGrade::interpretation`'s severe arm (or collapsing two arms) reddens here.
    let mut minor = benign_facts();
    minor.variance_inflation = 1.0;
    let mut moderate = benign_facts();
    moderate.variance_inflation = 2.0;
    let severe = measured_severe_facts();

    let texts: Vec<String> = [minor, moderate, severe]
        .iter()
        .map(format_regularized_warning)
        .collect();
    assert!(texts[0].ends_with("Standard errors are likely reliable."));
    assert!(texts[1].contains("interpret them with caution"));
    assert!(texts[2].contains("come mostly from the floor rather than from the data"));
    assert_ne!(texts[0], texts[1]);
    assert_ne!(texts[1], texts[2]);
}

#[test]
fn the_interpretation_follows_the_magnitude_that_triggered_the_tier() {
    // #1508 review §6. Severity is an `OR`, and the severe sentence used to claim the SEs
    // "come mostly from the floor" in **both** legs — false in the leg where the supplied
    // inflation says the floor contributed nothing to the reported variances. Both legs, both
    // tiers, in one test, so a formatter that ignores the cause reddens.
    let indefinite_only = CovRegularizationFacts {
        // `a_negative_eigenvalue_alone_grades_severe`'s exact cell: neg_ratio 1e-3,
        // inflation 1.0.
        min_eigenvalue: -1.0,
        max_eigenvalue: 1.0e3,
        variance_inflation: 1.0,
        ..measured_severe_facts()
    };
    assert!((indefinite_only.neg_ratio() - 1e-3).abs() < 1e-15);
    let g = indefinite_only.grade();
    assert_eq!(g.severity, CovSeverity::Severe);
    assert_eq!(g.cause, CovSeverityCause::Indefiniteness);
    let m = format_regularized_warning(&indefinite_only);
    assert!(
        !m.contains("come mostly from the floor"),
        "the inflation says the floor contributed nothing to the reported variances: {m}"
    );
    assert!(
        m.contains(
            "The Hessian was materially altered by the floor and these standard errors \
                    are not reliable"
        ),
        "{m}"
    );

    // The inflation leg of the same tier keeps the sentence that is true there.
    let mut inflation_only = measured_severe_facts();
    inflation_only.min_eigenvalue = 1.0e-8;
    inflation_only.max_eigenvalue = 1.0e4;
    assert_eq!(inflation_only.neg_ratio(), 0.0);
    assert_eq!(inflation_only.grade().cause, CovSeverityCause::Inflation);
    assert!(
        format_regularized_warning(&inflation_only).contains("come mostly from the floor"),
        "the inflation leg's sentence is true and must stay"
    );

    // And the same split one tier down.
    let moderate_indefinite = CovRegularizationFacts {
        min_eigenvalue: -1.0,
        max_eigenvalue: 1.0e8,
        variance_inflation: 1.0,
        ..measured_severe_facts()
    };
    let g = moderate_indefinite.grade();
    assert_eq!(g.severity, CovSeverity::Moderate);
    assert_eq!(g.cause, CovSeverityCause::Indefiniteness);
    let m = format_regularized_warning(&moderate_indefinite);
    assert!(
        m.contains("the reported standard errors do not load on the altered directions"),
        "{m}"
    );
    assert!(
        !m.contains("comes from the floor rather than from the data"),
        "{m}"
    );

    let mut moderate_inflation = benign_facts();
    moderate_inflation.variance_inflation = 2.0;
    assert_eq!(
        moderate_inflation.grade().cause,
        CovSeverityCause::Inflation
    );
    assert!(
        format_regularized_warning(&moderate_inflation)
            .contains("comes from the floor rather than from the data"),
        "the moderate inflation leg keeps its own sentence"
    );
}

// ── C2, the inner-gradient half, and the route the tolerance facts are read from ────────

const CLOSED_FORM_MODEL: &str = "
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

/// A closed-form transit model — no `[odes]` block, but an absorption ODE twin built at parse
/// time (#814). Which of its subjects integrate depends on the subject, which is the whole
/// point of #1508 review §4.
const TRANSIT_MODEL: &str = "
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  MTT = TVMTT
  NTR = TVN

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP_ERR)
";

fn ode_model_src(tail: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V)*central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)
{tail}"
    )
}

/// One subject, dosed the way `bolus` says: an instantaneous bolus keeps a closed-form transit
/// model on its closed form, an infusion reroutes it to the absorption ODE twin
/// (`CompiledModel::effective_for`).
fn subject_with_dose(bolus: bool) -> Subject {
    let rate = if bolus { 0.0 } else { 10.0 };
    Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, rate, false, 0.0)],
        ..Default::default()
    }
}

fn population_of(subjects: Vec<Subject>) -> crate::types::Population {
    crate::types::Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

#[test]
fn the_tolerance_facts_are_read_from_the_route_the_subjects_take() {
    // #1508 review §4, at the gate itself. `OdeToleranceFacts::from_model` keyed on
    // `model.ode_spec`, so a closed-form transit model whose subjects reroute to the
    // absorption ODE twin — with IOV, every one of them — read as "no integration" and the
    // whole C2 guidance was suppressed on a fit whose FD stencil was reading integrator noise.
    //
    // All three cells in one test, on the SAME model, so the difference is the route and
    // nothing else: mutating the twin walk away leaves the middle assertion red.
    let transit = crate::parser::model_parser::parse_model_string(TRANSIT_MODEL).unwrap();
    assert!(
        transit.ode_spec.is_none(),
        "a one_cpt_transit model is closed form"
    );
    assert!(
        transit.absorption_ode_equivalent.is_some(),
        "and carries an absorption ODE twin"
    );

    // Bolus-only population: nothing reroutes, nothing integrates, no facts.
    let bolus_pop = population_of(vec![subject_with_dose(true)]);
    assert!(
        OdeToleranceFacts::from_route(&transit, &bolus_pop).is_none(),
        "a transit model whose subjects all keep the closed form never integrates"
    );

    // One infusion subject is enough: that subject integrates on the twin.
    let infusion_pop = population_of(vec![subject_with_dose(true), subject_with_dose(false)]);
    let facts = OdeToleranceFacts::from_route(&transit, &infusion_pop)
        .expect("an infusion subject routes to the ODE twin");
    assert!(facts.via_twin, "{facts:?}");
    assert_eq!(facts.reltol, 1e-4);
    assert_eq!(facts.abstol, 1e-6);
    assert!(facts.looser_than_cov_plateau());

    // And a model with its own `[odes]` block is never reported as a twin.
    let ode = crate::parser::model_parser::parse_model_string(&ode_model_src("")).unwrap();
    let facts = OdeToleranceFacts::from_route(&ode, &bolus_pop).expect("an [odes] model");
    assert!(!facts.via_twin, "{facts:?}");

    // A model that cannot integrate at all, on a population that would reroute one if it
    // could: still `None`, so the predicate is about the twin and not about the dose shape.
    let closed = crate::parser::model_parser::parse_model_string(CLOSED_FORM_MODEL).unwrap();
    assert!(closed.absorption_ode_equivalent.is_none());
    assert!(OdeToleranceFacts::from_route(&closed, &infusion_pop).is_none());
}

#[test]
fn the_fd_inner_gradient_note_straddles_the_plateau_and_the_model_class() {
    // Three cells of one gate in one test: a model that never integrates (no note at all),
    // one integrating at the default (tighten + escalate + scope), one already at the plateau
    // (scope only — telling a fit at 1e-9 to tighten to 1e-6 would be false).
    let pop = population_of(vec![subject_with_dose(true)]);
    let closed = crate::parser::model_parser::parse_model_string(CLOSED_FORM_MODEL).unwrap();
    assert!(
        fd_inner_gradient_tolerance_note(&closed, &pop).is_none(),
        "a model that never integrates has no integration noise for a tolerance to remove"
    );

    let default_tol = crate::parser::model_parser::parse_model_string(&ode_model_src("")).unwrap();
    let note = fd_inner_gradient_tolerance_note(&default_tol, &pop).expect("ODE model gets a note");
    assert!(
        note.contains("ode_reltol = 1e-4 / ode_abstol = 1e-6"),
        "{note}"
    );
    assert!(note.contains("next step 1e-9"), "{note}");
    assert!(note.contains("analytic sensitivity scope"), "{note}");

    let tight = crate::parser::model_parser::parse_model_string(&ode_model_src(
        "\n[fit_options]\n  ode_reltol = 1e-9\n  ode_abstol = 1e-11\n",
    ))
    .unwrap();
    let note = fd_inner_gradient_tolerance_note(&tight, &pop).expect("ODE model gets a note");
    assert!(!note.contains("next step 1e-9"), "{note}");
    assert!(!note.contains("ode_reltol = 1e-9"), "{note}");
    assert!(note.contains("analytic sensitivity scope"), "{note}");
}

#[test]
fn the_fd_inner_gradient_note_reaches_a_rerouted_closed_form_model() {
    // The cell #1508 review §4 found empty: the note is about a finite-difference inner
    // gradient reading integration noise, and a transit subject on the ODE twin has exactly
    // that. Both sides of the reroute in one test, on one model.
    let transit = crate::parser::model_parser::parse_model_string(TRANSIT_MODEL).unwrap();
    let bolus = population_of(vec![subject_with_dose(true)]);
    let infusion = population_of(vec![subject_with_dose(false)]);

    assert!(
        fd_inner_gradient_tolerance_note(&transit, &bolus).is_none(),
        "a transit population that never leaves the closed form is told nothing"
    );
    let note = fd_inner_gradient_tolerance_note(&transit, &infusion)
        .expect("a rerouted transit subject integrates");
    assert!(
        note.contains("this model's closed-form absorption ODE twin integrates at"),
        "{note}"
    );
    assert!(
        note.contains("ode_reltol = 1e-4 / ode_abstol = 1e-6"),
        "{note}"
    );
}

// ── The gate that names the clauses is the gate that fires ──────────────────────────────

#[test]
fn the_scope_gate_names_the_clause_it_declines_on() {
    // `covariance_scope_decline` IS the gate `covariance_sensitivities` runs, so these are
    // assertions about the real routing decision, not about a parallel description of it.
    // Two clauses that a user can act on, plus the in-scope control that must decline nothing.
    use crate::sens::provider::covariance_scope_decline;

    let subject = Subject {
        id: "1".to_string(),
        ..Default::default()
    };

    let mut in_scope = crate::parser::model_parser::parse_model_string(CLOSED_FORM_MODEL).unwrap();
    assert_eq!(covariance_scope_decline(&in_scope, &subject, false), None);

    // `[scaling] obs_scale = V` — the idiom the #520 review found on `warfarin_ode_lagtime`.
    let scaled = crate::parser::model_parser::parse_model_string(&CLOSED_FORM_MODEL.replace(
        "[error_model]",
        "[scaling]\n  obs_scale = V\n\n[error_model]",
    ))
    .unwrap();
    assert_eq!(
        covariance_scope_decline(&scaled, &subject, false),
        Some(CovScopeDecline::ExpressionScale),
    );

    // `gradient = fd` reaches the model through `GradientMethod::effective` in `fit()`, not
    // through the parser, so the field is what the gate reads and what the test sets.
    in_scope.gradient_method = crate::types::GradientMethod::Fd;
    assert_eq!(
        covariance_scope_decline(&in_scope, &subject, false),
        Some(CovScopeDecline::GradientFd),
    );
}

#[test]
fn the_exhaustive_walk_reports_every_clause_and_the_routing_walk_still_short_circuits() {
    // The two halves of one walk (#1508 review §3). `covariance_scope_declines` must find
    // *both* clauses on a model that carries both, and `covariance_scope_decline` must still
    // return the first one — the routing decision is unchanged, which is the property that
    // makes the refactor safe.
    use crate::sens::provider::{covariance_scope_decline, covariance_scope_declines};

    let subject = Subject {
        id: "1".to_string(),
        ..Default::default()
    };
    let mut both = crate::parser::model_parser::parse_model_string(&CLOSED_FORM_MODEL.replace(
        "[error_model]",
        "[scaling]\n  obs_scale = V\n\n[error_model]",
    ))
    .unwrap();
    both.gradient_method = crate::types::GradientMethod::Fd;

    let all = covariance_scope_declines(&both, &subject, false);
    assert!(
        all.contains(&CovScopeDecline::GradientFd)
            && all.contains(&CovScopeDecline::ExpressionScale),
        "{all:?}"
    );
    // Gate order, not set order: `gradient = fd` is evaluated before the scaling clause.
    assert_eq!(all[0], CovScopeDecline::GradientFd);
    assert_eq!(
        covariance_scope_decline(&both, &subject, false),
        Some(CovScopeDecline::GradientFd),
        "the routing walk keeps its short-circuit",
    );

    // An in-scope model reports nothing on either walk — the `is_empty()` / `is_none()` pair
    // that the routing decision actually tests.
    let in_scope = crate::parser::model_parser::parse_model_string(CLOSED_FORM_MODEL).unwrap();
    assert!(covariance_scope_declines(&in_scope, &subject, false).is_empty());
    assert_eq!(covariance_scope_decline(&in_scope, &subject, false), None);
}

// ── #1514: the hybrid route and the salvage note ────────────────────────────────────────

#[test]
fn the_message_names_the_hybrid_route_and_keeps_its_stencil_guidance() {
    // Three routes, one test, because the two gates the route drives disagree about where the
    // hybrid sits: the *label* must be its own (neither "the analytic R-matrix" — part of it
    // was second-differenced — nor "the FD Hessian" — most of it was not), while the
    // stencil-only tail must be **present**, because a stencil genuinely ran, over exactly the
    // subjects the declined clause names. Collapsing the hybrid onto either neighbour breaks
    // one of the two halves asserted here.
    let mut analytic = measured_severe_facts();
    analytic.declines = &[CovScopeDecline::ExpressionScale];
    let mut fd = analytic;
    fd.source = CovHessianSource::FdStencil;
    let mut hybrid = analytic;
    hybrid.source = CovHessianSource::HybridRMatrix;
    analytic.source = CovHessianSource::AnalyticRMatrix;

    let (m_analytic, m_fd, m_hybrid) = (
        format_regularized_warning(&analytic),
        format_regularized_warning(&fd),
        format_regularized_warning(&hybrid),
    );

    assert!(
        m_hybrid.contains("the hybrid analytic/FD R-matrix"),
        "{m_hybrid}"
    );
    assert!(!m_analytic.contains("hybrid"), "{m_analytic}");
    assert!(!m_fd.contains("hybrid"), "{m_fd}");

    // The tail: present on FD and hybrid, absent on the pure analytic route. The two routes
    // word it differently (`the_decline_sentence_names_the_subjects_that_declined_not_the_
    // whole_fit` owns that distinction), so the token asserted here is the clause itself —
    // the part that is common to both and is the reason the tail exists.
    let clause = "[scaling] obs_scale = ... is in use";
    assert!(m_fd.contains("declined because"), "{m_fd}");
    assert!(m_fd.contains(clause), "{m_fd}");
    assert!(m_hybrid.contains(clause), "{m_hybrid}");
    assert!(!m_analytic.contains(clause), "{m_analytic}");
    assert!(!m_analytic.contains("declined"), "{m_analytic}");
}

#[test]
fn the_decline_sentence_names_the_subjects_that_declined_not_the_whole_fit() {
    // #1516 review §2. The tail is *present* on both routes (asserted above), but its subject
    // is not the same: on the population stencil the fit declined, on the hybrid route only the
    // salvaged minority did. A 100-subject fit with three event-walk subjects told "the exact
    // analytic covariance R-matrix was declined" reads it as a statement about all 100.
    //
    // Both routes in one test, and each asserts the *other* route's wording is absent, so a
    // gate stuck on either branch reddens here rather than passing half.
    let mut fd = measured_severe_facts();
    fd.declines = &[CovScopeDecline::EventWalkSubject];
    fd.source = CovHessianSource::FdStencil;
    let mut hybrid = fd;
    hybrid.source = CovHessianSource::HybridRMatrix;

    let (m_fd, m_hybrid) = (
        format_regularized_warning(&fd),
        format_regularized_warning(&hybrid),
    );
    assert!(
        m_fd.contains(
            "The exact analytic covariance R-matrix was declined because at least one subject \
             routes to the event-driven walk."
        ),
        "{m_fd}"
    );
    assert!(
        m_hybrid.contains(
            "The finite-differenced subjects declined the exact analytic covariance R-matrix \
             because at least one subject routes to the event-driven walk."
        ),
        "{m_hybrid}"
    );
    assert!(
        !m_hybrid.contains("R-matrix was declined"),
        "the whole-fit wording must not survive on the hybrid route: {m_hybrid}"
    );
    assert!(
        !m_fd.contains("The finite-differenced subjects declined"),
        "the per-subject wording must not leak onto the population stencil: {m_fd}"
    );
}

#[test]
fn the_remedy_sentences_move_the_subjects_not_the_fit_on_the_hybrid_route() {
    // The other two sentences of the same tail, both cells, both routes. The promise ("moves
    // X onto the analytic route") and the residual blocker ("so X stays on the finite-
    // difference route") each name who X is, and on the hybrid route X is the salvaged
    // subjects — the rest of the population is already analytic.
    let mut hybrid = measured_severe_facts();
    hybrid.declines = &[CovScopeDecline::GradientFd];
    hybrid.source = CovHessianSource::HybridRMatrix;
    let mut fd = hybrid;
    fd.source = CovHessianSource::FdStencil;

    let m_hybrid = format_regularized_warning(&hybrid);
    let m_fd = format_regularized_warning(&fd);
    assert!(
        m_hybrid.contains("Dropping gradient = fd moves those subjects onto the analytic route."),
        "{m_hybrid}"
    );
    assert!(
        m_fd.contains("Dropping gradient = fd moves the fit onto the analytic route."),
        "{m_fd}"
    );

    // The unremedied cell: one action, one blocker with none.
    let mut hybrid_mixed = hybrid;
    hybrid_mixed.declines = &[
        CovScopeDecline::GradientFd,
        CovScopeDecline::EventWalkSubject,
    ];
    let mut fd_mixed = hybrid_mixed;
    fd_mixed.source = CovHessianSource::FdStencil;

    let m_hybrid_mixed = format_regularized_warning(&hybrid_mixed);
    let m_fd_mixed = format_regularized_warning(&fd_mixed);
    assert!(
        m_hybrid_mixed.contains(
            "so those subjects stay on the finite-difference route until all of them are cleared"
        ),
        "{m_hybrid_mixed}"
    );
    assert!(
        m_fd_mixed.contains(
            "so the fit stays on the finite-difference route until all of them are \
                      cleared"
        ),
        "{m_fd_mixed}"
    );
}

#[test]
fn the_ode_tolerance_sentence_reaches_the_hybrid_route_too() {
    // The `1/h²` amplification argument is about a mechanism, not a route name: on the hybrid
    // route the salvaged subjects' terms *are* second differences of the objective, so a
    // tolerance that is too loose for the stencil is still too loose for them. Both sides of
    // the gate, same ODE facts.
    let mut hybrid = measured_severe_facts();
    hybrid.ode = Some(ode_facts(1e-4, 1e-6));
    hybrid.source = CovHessianSource::HybridRMatrix;
    let mut analytic = hybrid;
    analytic.source = CovHessianSource::AnalyticRMatrix;

    assert!(format_regularized_warning(&hybrid).contains("amplifies integration noise"));
    assert!(!format_regularized_warning(&analytic).contains("amplifies integration noise"));
}

#[test]
fn the_offdiag_nan_warning_says_what_each_route_actually_lost() {
    // #1514 review §1. `hess += stencil.hess` means a non-finite cross-partial leaves behind
    // whatever was already in the entry, and that differs by route: the zero initialisation on
    // the population stencil (correlation wholly absent), the in-scope subjects' analytic term
    // on the hybrid (only the declined subjects' share missing). Both sides of the gate in one
    // test, with each asserted *not* to carry the other's claim — split across two tests, a
    // gate stuck on one branch would still pass half.
    let names = "TVCL, TVV";
    let fd = format_offdiag_nan_warning(names, CovHessianSource::FdStencil);
    let hybrid = format_offdiag_nan_warning(names, CovHessianSource::HybridRMatrix);

    assert!(fd.contains("Cross-partial correlation set to 0"), "{fd}");
    assert!(!fd.contains("analytically assembled"), "{fd}");

    assert!(
        hybrid.contains("keep only the analytically assembled subjects' contribution"),
        "{hybrid}"
    );
    assert!(
        hybrid.contains("the finite-differenced subjects' share of them is missing"),
        "{hybrid}"
    );
    // The false claim this fix exists to remove. On the hybrid route the entry is not zero.
    assert!(!hybrid.contains("set to 0"), "{hybrid}");

    // What both routes must keep, because it is true on both and it is the actionable half:
    // the named parameters, the over-optimism, and the knob. Deleting any of the three from
    // either arm reddens here.
    for (label, msg) in [("fd", &fd), ("hybrid", &hybrid)] {
        assert!(msg.contains(names), "{label}: {msg}");
        assert!(msg.contains("may be over-optimistic"), "{label}: {msg}");
        assert!(msg.contains("Try tuning fd_hessian_step"), "{label}: {msg}");
        // The token `classify_warning` keys on — a reworded message that dropped it would
        // silently demote the warning out of `covariance_regularized`.
        let entry = classify_warning(msg);
        assert_eq!(
            entry.category,
            WarningCode::CovarianceRegularized,
            "{label}: {msg}"
        );
    }
}

#[test]
fn the_salvage_note_is_absent_when_nothing_was_salvaged() {
    // The note's *presence* is the whole statement that the hybrid route ran, so the empty
    // cell has to be silent rather than say "0 of 10". Both degenerate inputs, because either
    // one alone would be satisfied by a formatter that keyed off the other.
    assert!(format_salvage_note(&[], 10).is_none());
    assert!(format_salvage_note(&["a"], 0).is_none());
}

#[test]
fn the_salvage_note_agrees_with_its_own_counts() {
    // Singular and plural in one test: four nouns and a verb agree with `ids.len()`, and the
    // "remaining" clause agrees with `n_total - ids.len()`, which is a *different* count.
    // Keying both on one count is the agreement bug this shape invites.
    let one = format_salvage_note(&["42"], 55).expect("one salvaged subject");
    assert!(one.contains("W_COV_ANALYTIC_SALVAGE"), "{one}");
    assert!(one.contains("1 of 55 subjects (ID 42) is outside"), "{one}");
    assert!(one.contains("its information term was"), "{one}");
    assert!(one.contains("its own marginal"), "{one}");
    assert!(
        one.contains("the remaining 54 subjects were assembled analytically"),
        "{one}"
    );

    // The note's second sentence, which no count reaches: it is the answer to the question
    // the first sentence provokes ("so are my standard errors a blend of two things?"), and
    // without an assertion here deleting it would kill no test at all.
    assert!(
        one.contains("only the named subjects' terms use a different estimator"),
        "{one}"
    );

    let many = format_salvage_note(&["7", "13", "42"], 10).expect("three salvaged subjects");
    assert!(
        many.contains("3 of 10 subjects (IDs 7, 13 and 42) are outside"),
        "{many}"
    );
    assert!(many.contains("their information terms were"), "{many}");
    assert!(many.contains("their own marginals"), "{many}");
    assert!(
        many.contains("the remaining 7 subjects were assembled analytically"),
        "{many}"
    );

    // The one cell where "remaining" is singular while the salvaged count is plural — the pair
    // that a single shared count gets wrong in one direction or the other.
    let almost_all = format_salvage_note(&["a", "b"], 3).expect("two of three");
    assert!(
        almost_all.contains("the remaining subject was assembled analytically"),
        "{almost_all}"
    );
}

#[test]
fn the_salvage_note_dedupes_ids_and_caps_the_list() {
    // Two independent reductions of the id list, asserted separately because each can be
    // deleted without the other reddening.
    //
    // Dedupe is a *printing* reduction only (#1516 review §3). The caller passes one id per
    // population index, and each index carries its own information term, so three salvaged
    // subjects two of which share an id string are still three salvaged terms and seventeen
    // analytic ones. Deriving the count from the deduped list instead printed "2 of 20 … the
    // remaining 18", under-reporting the salvage and over-reporting the analytic remainder by
    // the same one subject — so both counts are asserted next to the list they disagree with.
    let dup = format_salvage_note(&["4", "4", "9"], 20).expect("three salvaged subjects");
    assert!(dup.contains("3 of 20 subjects (IDs 4 and 9)"), "{dup}");
    assert!(
        dup.contains("the remaining 17 subjects were assembled analytically"),
        "{dup}"
    );

    // Cap: twelve ids print ten and summarise the rest, while the count stays the true one.
    let ids: Vec<String> = (0..12).map(|i| i.to_string()).collect();
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let capped = format_salvage_note(&refs, 40).expect("twelve salvaged subjects");
    assert!(capped.contains("12 of 40 subjects"), "{capped}");
    assert!(
        capped.contains("0, 1, 2, 3, 4, 5, 6, 7, 8, 9 and 2 more"),
        "{capped}"
    );
    assert!(
        !capped.contains(", 10,"),
        "the 11th id must not be printed: {capped}"
    );

    // Exactly at the cap nothing is summarised — the boundary the `>` above sits on.
    let ten: Vec<&str> = refs[..10].to_vec();
    let at_cap = format_salvage_note(&ten, 40).expect("ten salvaged subjects");
    assert!(!at_cap.contains("more"), "{at_cap}");
}

#[test]
fn the_salvage_note_classifies_as_an_informational_covariance_note() {
    // The note is a route report, not a degradation: the information matrix is complete and
    // the estimates and OFV are untouched. (The covariance itself does move a little — the
    // salvaged subjects' terms come off a different estimator — which is why the note exists
    // at all.) `CovarianceRegularized` asserts the eigenvalue floor fired, which is a
    // different and more serious claim, so the classification is pinned here, next to the text
    // it is keyed on.
    let note = format_salvage_note(&["3"], 10).expect("one salvaged subject");
    let entry = classify_warning(&note);
    assert_eq!(entry.category, WarningCode::CovarianceStep);
    assert_eq!(entry.severity, WarningSeverity::Info);
}
