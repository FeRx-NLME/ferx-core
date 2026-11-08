//! #2797: the coordinates the covariance step holds at the internal guard, the
//! `held_at_bound` list built from them, the conditioning helper, and the guard warning's
//! sentence about the standard errors — Tier 1, no fit.

use super::*;
use crate::estimation::covariance::{condition_on_held, held_free_positions};
use crate::estimation::parameterization::pack_params;
use crate::types::{
    classify_warning, HeldReason, OmegaMatrix, ParameterKind, SigmaVector, WarningCode,
    WarningSeverity,
};
use nalgebra::DMatrix;

/// The packed lower rail of a log-packed Ω / κ diagonal (`L_ii = e^-6`, ω = 6.144e-6).
const OMEGA_RAIL_L: f64 = -6.0;
/// The packed lower rail of a log-packed σ.
const SIGMA_RAIL: f64 = -8.0;

fn diag_chol(l: &[f64], names: &[&str]) -> OmegaMatrix {
    let n = l.len();
    let mut chol = DMatrix::zeros(n, n);
    for (i, &v) in l.iter().enumerate() {
        chol[(i, i)] = v;
    }
    OmegaMatrix::from_chol_factor(
        chol,
        names.iter().map(|s| s.to_string()).collect(),
        true,
        DMatrix::from_element(n, n, true),
    )
}

/// θ (TVCL on the internal floor, TVV on its declared lower bound), Ω diagonal
/// (ETA_A collapsed, ETA_B ran to the upper rail, ETA_C FIX on the lower rail), σ at its
/// lower rail, κ diagonal at its lower rail. Packed: `[TVCL, TVV, ETA_A, ETA_B, ETA_C,
/// PROP_ERR, K_A]`.
fn params_with_every_kind_of_hit() -> ModelParameters {
    ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![1e-10, 5.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.0, 5.0],
        theta_upper: vec![f64::INFINITY, 500.0],
        theta_fixed: vec![false; 2],
        omega: diag_chol(
            &[OMEGA_RAIL_L.exp(), 6.0_f64.exp(), OMEGA_RAIL_L.exp()],
            &["ETA_A", "ETA_B", "ETA_C"],
        ),
        omega_fixed: vec![false, false, true],
        sigma: SigmaVector {
            values: vec![SIGMA_RAIL.exp()],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: Some(diag_chol(&[OMEGA_RAIL_L.exp()], &["K_A"])),
        kappa_fixed: vec![false],
        mixture: None,
    }
}

/// Test 1 (plan §Tests): exactly the collapses are held — θ on the internal floor, the Ω
/// diagonal, σ, κ — and, in the same test so a predicate stuck on either branch reddens
/// it, an upper-rail Ω, a FIX Ω on the rail and a θ on a *declared* bound are not.
#[test]
fn the_step_holds_collapses_and_nothing_else() {
    let p = params_with_every_kind_of_hit();
    // Every coordinate is on a rail: the walk sees all of them (a FIX one is skipped
    // before the side is read, the declared θ bound is not internal).
    assert_eq!(super::held_guard_coordinates(&p), [0, 2, 5, 6]);

    let list = super::held_at_bound_list(&p, &[]);
    let got: Vec<(&str, ParameterKind, usize, usize, HeldReason)> = list
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index, h.reason))
        .collect();
    use HeldReason::CollapsedAtGuard as C;
    assert_eq!(
        got,
        [
            ("TVCL", ParameterKind::Theta, 0, 0, C),
            ("ETA_A", ParameterKind::Omega, 0, 2, C),
            ("PROP_ERR", ParameterKind::Sigma, 0, 5, C),
            ("K_A", ParameterKind::Kappa, 0, 6, C),
        ]
    );

    // The negative half on its own: off the rails, nothing is held.
    let mut interior = p.clone();
    interior.theta = vec![0.5, 6.0];
    interior.omega = diag_chol(&[0.3, 0.3, 0.3], &["ETA_A", "ETA_B", "ETA_C"]);
    interior.sigma.values = vec![0.1];
    interior.omega_iov = Some(diag_chol(&[0.3], &["K_A"]));
    assert!(super::held_guard_coordinates(&interior).is_empty());
    assert!(super::held_at_bound_list(&interior, &[]).is_empty());
}

/// D4 and D1 on a block Ω: a block's `L_ii` on the rail is held and listed as `Omega(i)`
/// (the second diagonal sits at packed offset 2 in the column-major lower triangle, so a
/// naive `p - omega_start` index would say 2); an off-diagonal at its −10 rail is a
/// runaway and is not held.
#[test]
fn a_block_diagonal_on_the_rail_is_held_and_its_off_diagonal_runaway_is_not() {
    let mut p = params_with_every_kind_of_hit();
    p.theta = vec![0.5, 6.0];
    p.sigma.values = vec![0.1];
    p.omega_iov = None;
    p.kappa_fixed = Vec::new();
    let mut chol = DMatrix::zeros(2, 2);
    chol[(0, 0)] = 1.0;
    chol[(1, 0)] = -10.0;
    chol[(1, 1)] = OMEGA_RAIL_L.exp();
    p.omega = OmegaMatrix::from_chol_factor(
        chol,
        vec!["ETA_CL".into(), "ETA_V".into()],
        false,
        DMatrix::from_element(2, 2, true),
    );
    p.omega_fixed = vec![false; 2];
    // Packed: [TVCL, TVV, L00, L10, L11, PROP_ERR].
    assert_eq!(super::held_guard_coordinates(&p), [4]);
    let list = super::held_at_bound_list(&p, &[]);
    assert_eq!(list.len(), 1);
    assert_eq!(
        (
            list[0].name.as_str(),
            list[0].kind,
            list[0].index,
            list[0].packed_index
        ),
        ("ETA_V", ParameterKind::Omega, 1, 4)
    );
}

/// D8's source half: profiled levels come first as `ProfiledBound` in packed order, and a
/// θ the profile pinned is not listed twice.
#[test]
fn profiled_levels_join_the_list_in_packed_order() {
    let p = params_with_every_kind_of_hit();
    let list = super::held_at_bound_list(&p, &[1]);
    let got: Vec<(usize, HeldReason)> = list.iter().map(|h| (h.packed_index, h.reason)).collect();
    assert_eq!(
        got,
        [
            (0, HeldReason::CollapsedAtGuard),
            (1, HeldReason::ProfiledBound),
            (2, HeldReason::CollapsedAtGuard),
            (5, HeldReason::CollapsedAtGuard),
            (6, HeldReason::CollapsedAtGuard),
        ]
    );
    assert_eq!(list[1].name, "TVV");
    // A level both profiled-at-bound and on the internal floor is listed once, as the
    // profile's.
    let both = super::held_at_bound_list(&p, &[0]);
    assert_eq!(both.iter().filter(|h| h.packed_index == 0).count(), 1);
    assert_eq!(both[0].reason, HeldReason::ProfiledBound);
}

/// T-1b (audit): the step reads the guard set off `x_hat`, never off `template`. The
/// estimators hand the **start** as the template, so a template off the rail with an
/// estimate on it must still hold; and the converse must not.
#[test]
fn the_step_reads_the_guard_set_off_the_estimate_not_the_template() {
    let mut start = params_with_every_kind_of_hit();
    start.theta = vec![0.5, 6.0];
    start.omega = diag_chol(&[0.30_f64.sqrt(); 3], &["ETA_A", "ETA_B", "ETA_C"]);
    start.sigma.values = vec![0.1];
    start.omega_iov = None;
    start.kappa_fixed = Vec::new();
    let mut estimate = start.clone();
    estimate.omega = diag_chol(
        &[OMEGA_RAIL_L.exp(), 0.30_f64.sqrt(), 0.30_f64.sqrt()],
        &["ETA_A", "ETA_B", "ETA_C"],
    );
    let free: Vec<usize> = (0..6).filter(|&i| i != 4).collect(); // ETA_C is FIX
    assert_eq!(
        held_free_positions(&pack_params(&estimate), &start, &free),
        [2]
    );
    assert!(held_free_positions(&pack_params(&start), &estimate, &free).is_empty());
}

/// `condition_on_held` is the inverse of the sub-block over the kept coordinates, embedded
/// with zero held rows and columns — the matrix a FIX hold gives — and declines (the caller
/// keeps its own matrix) when nothing or everything is held, or the inversion declines.
#[test]
fn condition_on_held_is_the_sub_block_inverse_embedded_with_zero_rows() {
    let m = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]);
    let inv = |s: &DMatrix<f64>, _: &[usize]| s.clone().try_inverse();
    let got = condition_on_held(&m, &[1], inv).expect("held");
    let sub = DMatrix::from_row_slice(2, 2, &[4.0, 0.5, 0.5, 2.0])
        .try_inverse()
        .unwrap();
    for (a, &i) in [0usize, 2].iter().enumerate() {
        for (b, &j) in [0usize, 2].iter().enumerate() {
            assert_eq!(got[(i, j)], sub[(a, b)], "({i},{j})");
        }
    }
    for k in 0..3 {
        assert_eq!(got[(1, k)], 0.0);
        assert_eq!(got[(k, 1)], 0.0);
    }
    // Not the unconditional inverse: the held coordinate's correlation moved the others.
    let full = m.clone().try_inverse().unwrap();
    assert_ne!(got[(0, 0)], full[(0, 0)]);

    assert!(condition_on_held(&m, &[], inv).is_none());
    assert!(condition_on_held(&m, &[0, 1, 2], inv).is_none());
    assert!(condition_on_held(&m, &[1], |_, _| None).is_none());
    // `keep` is the kept positions, in order.
    condition_on_held(&m, &[1], |s, keep| {
        assert_eq!(keep, [0, 2]);
        s.clone().try_inverse()
    });
}

// ── The guard warning's SE sentence: the audit's 7 cells ─────────────────────────────────

const HELD_SENTENCE: &str = "the other standard errors are computed with them held at the floor.";

/// σ collapsed (packed 3 in `[TVCL, TVV, ETA_CL, PROP_ERR]`), optionally with ETA_CL run
/// to its upper rail.
fn guard_params(collapse: bool, runaway: bool) -> ModelParameters {
    let mut p = params_with_every_kind_of_hit();
    p.theta = vec![0.5, 6.0];
    p.omega = diag_chol(&[if runaway { 6.0_f64.exp() } else { 0.3 }], &["ETA_CL"]);
    p.omega_fixed = vec![false];
    p.sigma.values = vec![if collapse { SIGMA_RAIL.exp() } else { 0.1 }];
    p.omega_iov = None;
    p.kappa_fixed = Vec::new();
    p
}

/// The step's matrix for each cell: held (σ's row zero), not held (a floored `R` or an `S`
/// fallback: a full positive diagonal), or none.
fn step_matrix(held: Option<bool>) -> Option<DMatrix<f64>> {
    held.map(|held| {
        let mut m = DMatrix::from_diagonal_element(4, 4, 0.01);
        if held {
            m[(3, 3)] = 0.0;
        }
        m
    })
}

/// One cell: the message, its structured entry and the same text through
/// `classify_warning` (the multi-start splice path), which must recover the category and
/// the severity — the new sentence avoids every substring an earlier branch keys on.
fn cell(collapse: bool, runaway: bool, held: Option<bool>) -> (String, serde_json::Value) {
    let p = guard_params(collapse, runaway);
    let cov = step_matrix(held);
    let mut converged = true;
    let (msg, entry) =
        super::runaway_guard_warning_for(&mut converged, &p, cov.as_ref()).expect("a guard hit");
    let severity = if runaway {
        WarningSeverity::Critical
    } else {
        WarningSeverity::Warning
    };
    assert_eq!(entry.severity, severity, "{msg}");
    let flat = classify_warning(&msg);
    assert_eq!(flat.category, WarningCode::ParameterAtRunawayGuard, "{msg}");
    assert_eq!(flat.severity, severity, "{msg}");
    (msg, entry.details.unwrap())
}

#[test]
fn collapse_only_clean_r_says_the_other_ses_are_conditional() {
    let (msg, details) = cell(true, false, Some(true));
    assert!(
        msg.contains("Their standard errors are not reported; "),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert_eq!(details["parameters"][0]["held"], true);
}

#[test]
fn collapse_only_floored_r_reports_no_se_and_claims_nothing_about_the_others() {
    let (msg, details) = cell(true, false, Some(false));
    assert!(
        msg.ends_with(" Their standard errors are not reported."),
        "{msg}"
    );
    assert!(!msg.contains(HELD_SENTENCE), "{msg}");
    assert_eq!(details["parameters"][0]["held"], false);
}

#[test]
fn collapse_only_without_a_matrix_has_no_se_sentence() {
    let (msg, details) = cell(true, false, None);
    assert!(!msg.contains("standard errors"), "{msg}");
    assert_eq!(details["parameters"][0]["held"], false);
}

#[test]
fn collapse_and_runaway_clean_r_names_the_collapse_subset_only() {
    let (msg, details) = cell(true, true, Some(true));
    assert!(
        msg.contains(
            " The standard errors of the collapsed parameter(s) PROP_ERR are not reported; "
        ),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert!(!msg.contains("parameter(s) ETA_CL"), "{msg}");
    assert!(!msg.contains("Their standard errors"), "{msg}");
    let hits = details["parameters"].as_array().unwrap();
    let held = |name: &str| hits.iter().find(|h| h["parameter"] == name).unwrap()["held"].clone();
    assert_eq!(held("PROP_ERR"), true);
    assert_eq!(held("ETA_CL"), false, "a runaway is never held");
}

#[test]
fn collapse_and_runaway_floored_r_names_the_subset_without_the_conditional_claim() {
    let (msg, _) = cell(true, true, Some(false));
    assert!(
        msg.ends_with(
            " The standard errors of the collapsed parameter(s) PROP_ERR are not reported."
        ),
        "{msg}"
    );
    assert!(!msg.contains(HELD_SENTENCE), "{msg}");
}

#[test]
fn collapse_and_runaway_without_a_matrix_has_no_se_sentence() {
    let (msg, _) = cell(true, true, None);
    assert!(!msg.contains("standard errors"), "{msg}");
}

#[test]
fn runaway_only_never_gets_an_se_sentence() {
    for held in [Some(true), Some(false), None] {
        let (msg, details) = cell(false, true, held);
        assert!(!msg.contains("standard errors"), "{held:?}: {msg}");
        assert_eq!(details["parameters"][0]["held"], false, "{held:?}");
    }
}
