//! Tier-1 tests for the covariance regularization diagnostic (#520).
//!
//! The message is code, so its **input space** is enumerated rather than sampled:
//!
//! | axis | values |
//! |---|---|
//! | route | analytic R-matrix, FD stencil |
//! | severity | minor, moderate, severe |
//! | declined clause | `None`, one with a remedy, one without |
//! | ODE tolerance | closed form, looser than the plateau, at/under the plateau |
//!
//! and every conditional sentence is asserted on **both sides of its own gate inside one
//! test**, so a gate stuck on one branch reddens rather than passing half. Each test names the
//! sentence it exists to kill; deleting that sentence from
//! [`super::format_regularized_warning`] must redden the named test and no other.
//!
//! None of this runs a fit: the facts are a plain struct and the formatter is pure, which is
//! the whole reason the split exists.

use super::*;
use crate::types::{classify_warning, WarningCode, WarningSeverity};

/// A regularization that is genuinely benign: the spectrum is non-negative, the smallest
/// eigenvalue sits just under the floor, and the floor barely moves the variance.
fn benign_facts() -> CovRegularizationFacts {
    CovRegularizationFacts {
        source: CovHessianSource::FdStencil,
        n_clipped: 1,
        n_free: 13,
        min_eigenvalue: 1.0e-8,
        max_eigenvalue: 1.0e4,
        floor: 1.0e-6,
        variance_inflation: 1.05,
        decline: None,
        ode: None,
    }
}

/// The 2026-09-15 measurement from #520: 1 of 13 eigenvalues clipped — 7%, which the old
/// count-fraction grading called "minor … Standard errors are likely reliable" — next to an SE
/// inflated 4400× (so a variance inflated ~1.9e7×) and a `%RSE` of 24982.
fn measured_severe_facts() -> CovRegularizationFacts {
    CovRegularizationFacts {
        source: CovHessianSource::FdStencil,
        n_clipped: 1,
        n_free: 13,
        min_eigenvalue: -3.197e1,
        max_eigenvalue: 8.416e3,
        floor: 8.416e-7,
        variance_inflation: 1.9e7,
        decline: None,
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

    assert_eq!(benign.severity(), CovSeverity::Minor);
    assert_eq!(
        measured.severity(),
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
    // Kills the `neg_ratio` leg of `grade_severity`. Variance inflation is pinned at 1.0, so
    // deleting the `neg_ratio` comparison leaves this at Minor. Paired with the test below,
    // this is what stops the two legs covering for each other (CLAUDE.md's redundant-gate
    // hole): each leg is asserted with the other one inert.
    assert_eq!(grade_severity(1e-3, 1.0), CovSeverity::Severe);
    assert_eq!(grade_severity(1e-7, 1.0), CovSeverity::Moderate);
    assert_eq!(grade_severity(1e-10, 1.0), CovSeverity::Minor);
}

#[test]
fn variance_inflation_alone_grades_severe() {
    // Kills the `variance_inflation` leg. `neg_ratio = 0` is the reachable near-singular case:
    // the whole spectrum is positive, nothing is indefinite, and the floor still manufactured
    // the reported variance.
    assert_eq!(grade_severity(0.0, 1.0e6), CovSeverity::Severe);
    assert_eq!(grade_severity(0.0, 10.0), CovSeverity::Moderate);
    assert_eq!(grade_severity(0.0, 1.5), CovSeverity::Minor);
    // Unbounded inflation — a coordinate whose entire variance came from floored directions.
    assert_eq!(grade_severity(0.0, f64::INFINITY), CovSeverity::Severe);
}

#[test]
fn severity_boundaries_are_straddled_on_both_sides() {
    // Each threshold asserted just below and just above, so a `>` silently becoming `>=` (or
    // a constant drifting) reddens rather than shifting one cell quietly.
    assert_eq!(grade_severity(1e-6, 1.0), CovSeverity::Moderate);
    assert_eq!(grade_severity(1.01e-6, 1.0), CovSeverity::Severe);
    assert_eq!(grade_severity(1e-9, 1.0), CovSeverity::Minor);
    assert_eq!(grade_severity(1.01e-9, 1.0), CovSeverity::Moderate);
    assert_eq!(grade_severity(0.0, 100.0), CovSeverity::Moderate);
    assert_eq!(grade_severity(0.0, 100.01), CovSeverity::Severe);
    assert_eq!(grade_severity(0.0, 4.0), CovSeverity::Minor);
    assert_eq!(grade_severity(0.0, 4.01), CovSeverity::Moderate);
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
    // analytic R-matrix route too, where no stencil and no step size exist. Both sides of the
    // gate in one test — the facts differ only in `source`.
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

// ── C2: the declined gate clause ────────────────────────────────────────────────────────

#[test]
fn the_declined_clause_sentence_is_fd_route_only() {
    // Both sides of the route gate on one decline value. On the analytic route there is no
    // clause to name — the analytic route *is* what a clause would have declined — so the
    // sentence must be absent, not merely differently worded.
    let mut fd = measured_severe_facts();
    fd.decline = Some(CovScopeDecline::ExpressionScale);
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
    with.decline = Some(CovScopeDecline::ExpressionScale);
    let m = format_regularized_warning(&with);
    assert!(m.contains("[scaling] obs_scale = ... is in use"), "{m}");
    assert!(m.contains("[scaling] y = central / V"), "{m}");

    let mut without = measured_severe_facts();
    without.decline = Some(CovScopeDecline::LogTransform);
    let m = format_regularized_warning(&without);
    assert!(m.contains("the model is log-transform-both-sides."), "{m}");
    // No invented advice: the FD stencil is the correct route for an LTBS model.
    assert!(!m.contains("moves the fit onto the analytic route"), "{m}");
}

#[test]
fn every_decline_clause_has_non_empty_prose() {
    // A variant added to the gate without message text would otherwise reach a user as an
    // empty "declined because ." — the failure mode a `#[non_exhaustive]`-style match cannot
    // catch, since every arm compiles.
    for decline in [
        CovScopeDecline::Disabled,
        CovScopeDecline::Mixture,
        CovScopeDecline::ExactHessianAnchor,
        CovScopeDecline::ModelOutOfScope,
        CovScopeDecline::GradientFd,
        CovScopeDecline::NonGaussianEndpoint,
        CovScopeDecline::Frem,
        CovScopeDecline::SelectedErrorSpec,
        CovScopeDecline::LogTransform,
        CovScopeDecline::IovShape,
        CovScopeDecline::ExpressionScale,
        CovScopeDecline::AnalyticReadout,
        CovScopeDecline::AnalyticalInit,
        CovScopeDecline::ResidualErrorEta,
        CovScopeDecline::ResidualCorrelations,
        CovScopeDecline::CustomRuvMagnitude,
        CovScopeDecline::EventWalkSubject,
        CovScopeDecline::IndivParamProgram,
        CovScopeDecline::PerSubjectBail,
    ] {
        assert!(!decline.clause().is_empty(), "{decline:?}");
        let mut facts = measured_severe_facts();
        facts.decline = Some(decline);
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

// ── C2: the ODE tolerance sentence ──────────────────────────────────────────────────────

#[test]
fn the_ode_tolerance_sentence_straddles_the_plateau() {
    // Both sides of `looser_than_cov_plateau` in one test. At the default the sentence is
    // true and must appear; at the plateau the advice "tighten to 1e-6 / 1e-8" would be
    // advice to change nothing, so the sentence must be absent.
    let mut loose = measured_severe_facts();
    loose.ode = Some(OdeToleranceFacts {
        reltol: 1e-4,
        abstol: 1e-6,
    });
    let m = format_regularized_warning(&loose);
    assert!(m.contains("ode_reltol = 1e-4 / ode_abstol = 1e-6"), "{m}");
    assert!(m.contains("amplifies integration noise by 1/h"), "{m}");

    let mut tight = measured_severe_facts();
    tight.ode = Some(OdeToleranceFacts {
        reltol: 1e-6,
        abstol: 1e-8,
    });
    assert!(!format_regularized_warning(&tight).contains("amplifies integration noise"));

    // A half-tightened fit is still looser than the plateau on one of the two knobs, and is
    // told so — the gate is `||`, not `&&`.
    let mut half = measured_severe_facts();
    half.ode = Some(OdeToleranceFacts {
        reltol: 1e-6,
        abstol: 1e-6,
    });
    assert!(format_regularized_warning(&half).contains("amplifies integration noise"));
}

#[test]
fn the_ode_tolerance_sentence_is_fd_route_only() {
    // The analytic R-matrix never second-differences the objective, so no tolerance argument
    // applies to it — the #520 measurement is flat across eight orders of magnitude there.
    // Same ODE facts, both routes, one test.
    let ode = Some(OdeToleranceFacts {
        reltol: 1e-4,
        abstol: 1e-6,
    });
    let mut fd = measured_severe_facts();
    fd.ode = ode;
    let mut analytic = fd;
    analytic.source = CovHessianSource::AnalyticRMatrix;

    assert!(format_regularized_warning(&fd).contains("amplifies integration noise"));
    assert!(!format_regularized_warning(&analytic).contains("amplifies integration noise"));
}

#[test]
fn a_closed_form_model_is_told_nothing_about_ode_tolerances() {
    // `ode: None` is the closed-form cell. No sentence, on either route.
    let facts = measured_severe_facts();
    assert!(facts.ode.is_none());
    let m = format_regularized_warning(&facts);
    assert!(!m.contains("ode_reltol"), "{m}");
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
        msg.contains("worst variance inflation from the floor ="),
        "{msg}"
    );
}

#[test]
fn unbounded_inflation_is_worded_not_printed_as_inf() {
    // `f64::INFINITY` is reachable (a coordinate loading entirely on floored directions) and
    // `{:.3e}` would render it as `inf`, which reads as a bug rather than as a finding.
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
    // Deleting `CovSeverity::interpretation`'s severe arm (or collapsing two arms) reddens here.
    let mut minor = benign_facts();
    minor.variance_inflation = 1.0;
    let mut moderate = benign_facts();
    moderate.variance_inflation = 10.0;
    let severe = measured_severe_facts();

    let texts: Vec<String> = [minor, moderate, severe]
        .iter()
        .map(format_regularized_warning)
        .collect();
    assert!(texts[0].ends_with("Standard errors are likely reliable."));
    assert!(texts[1].contains("interpreted with caution"));
    assert!(texts[2].contains("come mostly from the floor rather than from the data"));
    assert_ne!(texts[0], texts[1]);
    assert_ne!(texts[1], texts[2]);
}

// ── C2, the inner-gradient half ─────────────────────────────────────────────────────────

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

#[test]
fn the_fd_inner_gradient_note_straddles_the_plateau_and_the_model_class() {
    // Three cells of one gate in one test: closed form (no note at all), ODE at the default
    // (tighten + escalate + scope), ODE already at the plateau (scope only — telling a fit at
    // 1e-9 to tighten to 1e-6 would be false).
    let closed = crate::parser::model_parser::parse_model_string(CLOSED_FORM_MODEL).unwrap();
    assert!(
        fd_inner_gradient_tolerance_note(&closed).is_none(),
        "a closed-form model has no integration noise for a tolerance to remove"
    );

    let default_tol = crate::parser::model_parser::parse_model_string(&ode_model_src("")).unwrap();
    let note = fd_inner_gradient_tolerance_note(&default_tol).expect("ODE model gets a note");
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
    let note = fd_inner_gradient_tolerance_note(&tight).expect("ODE model gets a note");
    assert!(!note.contains("next step 1e-9"), "{note}");
    assert!(!note.contains("ode_reltol = 1e-9"), "{note}");
    assert!(note.contains("analytic sensitivity scope"), "{note}");
}

// ── The gate that names the clause is the gate that fires ───────────────────────────────

#[test]
fn the_scope_gate_names_the_clause_it_declines_on() {
    // `covariance_scope_decline` IS the gate `covariance_sensitivities` runs, so these are
    // assertions about the real routing decision, not about a parallel description of it.
    // Two clauses that a user can act on, plus the in-scope control that must decline nothing.
    use crate::sens::provider::covariance_scope_decline;

    let subject = crate::types::Subject {
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
