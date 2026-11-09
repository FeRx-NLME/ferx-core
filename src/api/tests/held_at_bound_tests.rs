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

// ── `[mixture]` per-class overrides (#2809) ──────────────────────────────────────────────

/// Packed: `[TVCL, TVV, MIXL2, MIXL3, ETA_CL, ETA_V, PROP_ERR, ADD_ERR, ETA_CL_MIX2,
/// ETA_V_MIX2, ADD_ERR_MIX3]` — three classes; `omega(2) ETA_CL` free, `omega(2) ETA_V`
/// FIX and always on the rail, `sigma(3) ADD_ERR` free. The Σ override names σ index 1
/// and class 3, the Ω one η index 0 and class 2, so a swap of the two address tables, or
/// a dropped `+ 1`, changes the listed `(kind, index, class)`.
///
/// `ov_omega` / `ov_sigma`: the free overrides on their lower rail; `base_v`: ETA_V
/// collapsed (`Some(false)`) or run to the upper rail (`Some(true)`); `None` interior.
fn mixture_params(ov_omega: bool, ov_sigma: bool, base_v: Option<bool>) -> ModelParameters {
    let mut p = params_with_every_kind_of_hit();
    p.theta = vec![0.5, 6.0, 0.0, 0.0];
    p.theta_names = vec!["TVCL".into(), "TVV".into(), "MIXL2".into(), "MIXL3".into()];
    p.theta_lower = vec![0.0, 0.0, -10.0, -10.0];
    p.theta_upper = vec![f64::INFINITY, 500.0, 10.0, 10.0];
    p.theta_fixed = vec![false; 4];
    let l_v = match base_v {
        None => 0.3,
        Some(false) => OMEGA_RAIL_L.exp(),
        Some(true) => 6.0_f64.exp(),
    };
    let names = ["ETA_CL", "ETA_V"];
    p.omega = diag_chol(&[0.25, l_v], &names);
    p.omega_fixed = vec![false; 2];
    p.sigma = SigmaVector {
        values: vec![0.1, 0.2],
        names: vec!["PROP_ERR".into(), "ADD_ERR".into()],
    };
    p.sigma_fixed = vec![false; 2];
    p.omega_iov = None;
    p.kappa_fixed = Vec::new();
    let ov_l = if ov_omega { OMEGA_RAIL_L.exp() } else { 0.2 };
    let ov_s = if ov_sigma { SIGMA_RAIL.exp() } else { 0.15 };
    p.mixture = Some(crate::types::MixtureParams {
        omega: vec![
            p.omega.clone(),
            diag_chol(&[ov_l, OMEGA_RAIL_L.exp()], &names),
            p.omega.clone(),
        ],
        sigma: vec![
            p.sigma.clone(),
            p.sigma.clone(),
            SigmaVector {
                values: vec![0.1, ov_s],
                names: p.sigma.names.clone(),
            },
        ],
        omega_override_addr: vec![(1, 0), (1, 1)],
        omega_override_fixed: vec![false, true],
        sigma_override_addr: vec![(2, 1)],
        sigma_override_fixed: vec![false],
    });
    p
}

/// T1 (#2809): a collapsed per-class override is held like a base coordinate; a FIX
/// override on the rail is not; nor is one off the rail. Restoring the D3 range filter
/// drops 8 and 10.
#[test]
fn a_collapsed_mixture_override_is_held_and_a_fix_one_is_not() {
    let p = mixture_params(true, true, None);
    assert_eq!(
        crate::estimation::parameterization::coordinate_names(&p)[8..],
        ["ETA_CL_MIX2", "ETA_V_MIX2", "ADD_ERR_MIX3"],
        "packed order the fixture relies on"
    );
    assert_eq!(super::held_guard_coordinates(&p), [8, 10]);
    // Beside a base collapse, both are held.
    assert_eq!(
        super::held_guard_coordinates(&mixture_params(true, true, Some(false))),
        [5, 8, 10]
    );
    // Off the rail (the FIX override still on it): nothing.
    assert!(super::held_guard_coordinates(&mixture_params(false, false, None)).is_empty());
}

/// T2 (#2809): the list names each override by its base `(kind, index)` and its 1-based
/// class, and lists no base entry for it. Kills: `packed_diagonal_parameter` returning
/// `None` past the base segments (the list is empty), the Σ range read through
/// `omega_override_addr` (`(Sigma, 0, Some(2))`), and a dropped `+ 1` (classes 1 and 2).
#[test]
fn a_held_override_is_listed_with_its_class_and_never_as_the_base() {
    let list = super::held_at_bound_list(&mixture_params(true, true, None), &[]);
    let got: Vec<_> = list
        .iter()
        .map(|h| {
            (
                h.name.as_str(),
                h.kind,
                h.index,
                h.packed_index,
                h.reason,
                h.class,
            )
        })
        .collect();
    use HeldReason::CollapsedAtGuard as C;
    assert_eq!(
        got,
        [
            ("ETA_CL_MIX2", ParameterKind::Omega, 0, 8, C, Some(2)),
            ("ADD_ERR_MIX3", ParameterKind::Sigma, 1, 10, C, Some(3)),
        ]
    );
    assert!(list.iter().all(|h| h.class.is_some()), "{list:?}");

    // A base collapse beside them keeps `class: None`.
    let list = super::held_at_bound_list(&mixture_params(true, true, Some(false)), &[]);
    let base: Vec<_> = list.iter().filter(|h| h.class.is_none()).collect();
    assert_eq!(base.len(), 1, "{list:?}");
    assert_eq!(
        (base[0].name.as_str(), base[0].kind, base[0].index),
        ("ETA_V", ParameterKind::Omega, 1)
    );
}

/// The covariance a clean `R` gives `mixture_params`: every held coordinate's row zero,
/// the FIX override's row zero as for any FIX coordinate.
fn mixture_step_matrix(zero_rows: &[usize]) -> DMatrix<f64> {
    let mut m = DMatrix::from_diagonal_element(11, 11, 0.01);
    for &p in zero_rows.iter().chain(&[9]) {
        m[(p, p)] = 0.0;
    }
    m
}

/// T3 (#2809): `apply_held_at_bound` writes no `0.0` for a held override — it has no
/// `se_*` slot, and its `(kind, index)` names the base, whose SE would be zeroed — while a
/// base collapse in the same call still gets its `0.0` (both sides of the gate).
#[test]
fn applying_a_held_override_leaves_the_base_se_untouched() {
    let p = mixture_params(true, true, Some(false));
    let mut r = crate::types::test_helpers::minimal_fit_result();
    r.omega = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.0625, 3.8e-11]));
    r.covariance_matrix = Some(mixture_step_matrix(&[5, 8, 10]));
    let se_eta_cl = 0.019_272_6_f64;
    let se_add = 0.002_f64;
    r.se_omega = Some(vec![se_eta_cl, 0.03]);
    r.se_sigma = Some(vec![0.001, se_add]);
    super::apply_held_at_bound(&p, &mut r);
    assert_eq!(r.held_at_bound.len(), 3, "{:?}", r.held_at_bound);
    let se_omega = r.se_omega.unwrap();
    let se_sigma = r.se_sigma.unwrap();
    assert_eq!(se_omega[0].to_bits(), se_eta_cl.to_bits(), "base ETA_CL SE");
    assert_eq!(se_sigma[1].to_bits(), se_add.to_bits(), "base ADD_ERR SE");
    assert_eq!(se_omega[1], 0.0, "the base collapse ETA_V is still zeroed");
    assert_eq!(se_sigma[0].to_bits(), 0.001_f64.to_bits());
}

/// One guard-warning cell on `mixture_params` (T6, #2809).
fn mixture_cell(
    p: &ModelParameters,
    cov: Option<DMatrix<f64>>,
) -> Option<(String, serde_json::Value)> {
    let mut converged = true;
    super::runaway_guard_warning_for(&mut converged, p, cov.as_ref())
        .map(|(msg, entry)| (msg, entry.details.unwrap()))
}

fn held_of(details: &serde_json::Value, name: &str) -> serde_json::Value {
    details["parameters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["parameter"] == name)
        .unwrap_or_else(|| panic!("{name} not in {details}"))["held"]
        .clone()
}

/// `msg` names `full` and never the bare base name `base` (the base parameter did not
/// collapse; its override did).
fn names_only_the_override(msg: &str, full: &str, base: &str) {
    assert!(msg.contains(full), "{msg}");
    assert!(!msg.replace(full, "").contains(base), "bare {base}: {msg}");
}

#[test]
fn override_collapse_clean_r_is_held_and_named_with_its_class() {
    let p = mixture_params(true, false, None);
    let (msg, details) = mixture_cell(&p, Some(mixture_step_matrix(&[8]))).unwrap();
    names_only_the_override(&msg, "ETA_CL_MIX2", "ETA_CL");
    assert!(
        msg.contains("Their standard errors are not reported; "),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert_eq!(held_of(&details, "ETA_CL_MIX2"), true);
}

#[test]
fn override_collapse_floored_r_reports_no_se_and_claims_no_hold() {
    let p = mixture_params(true, false, None);
    let (msg, details) = mixture_cell(&p, Some(mixture_step_matrix(&[]))).unwrap();
    names_only_the_override(&msg, "ETA_CL_MIX2", "ETA_CL");
    assert!(
        msg.ends_with(" Their standard errors are not reported."),
        "{msg}"
    );
    assert!(!msg.contains(HELD_SENTENCE), "{msg}");
    assert_eq!(held_of(&details, "ETA_CL_MIX2"), false);
}

#[test]
fn override_collapse_without_a_matrix_has_no_se_sentence() {
    let p = mixture_params(true, false, None);
    let (msg, details) = mixture_cell(&p, None).unwrap();
    names_only_the_override(&msg, "ETA_CL_MIX2", "ETA_CL");
    assert!(!msg.contains("not reported"), "{msg}");
    assert_eq!(held_of(&details, "ETA_CL_MIX2"), false);
}

#[test]
fn override_collapse_and_runaway_names_the_override_in_the_collapse_subset() {
    let p = mixture_params(true, false, Some(true));
    let (msg, details) = mixture_cell(&p, Some(mixture_step_matrix(&[8]))).unwrap();
    assert!(
        msg.contains(
            " The standard errors of the collapsed parameter(s) ETA_CL_MIX2 are not reported; "
        ),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert!(!msg.contains("parameter(s) ETA_V"), "{msg}");
    assert_eq!(held_of(&details, "ETA_CL_MIX2"), true);
    assert_eq!(held_of(&details, "ETA_V"), false, "a runaway is never held");
}

#[test]
fn base_and_override_collapse_clean_r_holds_both() {
    let p = mixture_params(true, false, Some(false));
    let (msg, details) = mixture_cell(&p, Some(mixture_step_matrix(&[5, 8]))).unwrap();
    assert!(msg.contains("ETA_V"), "{msg}");
    assert!(msg.contains("ETA_CL_MIX2"), "{msg}");
    assert!(
        msg.contains("Their standard errors are not reported; "),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert_eq!(held_of(&details, "ETA_V"), true);
    assert_eq!(held_of(&details, "ETA_CL_MIX2"), true);
}

#[test]
fn a_fix_override_on_the_rail_raises_no_guard_warning() {
    let p = mixture_params(false, false, None);
    assert!(
        mixture_cell(&p, Some(mixture_step_matrix(&[]))).is_none(),
        "ETA_V_MIX2 is FIX on the rail"
    );
}

#[test]
fn sigma_override_collapse_clean_r_is_held_and_named_with_its_class() {
    let p = mixture_params(false, true, None);
    let (msg, details) = mixture_cell(&p, Some(mixture_step_matrix(&[10]))).unwrap();
    names_only_the_override(&msg, "ADD_ERR_MIX3", "ADD_ERR");
    // The warning's "held" and the list must agree (#2831 review r1 row 4): a Σ branch in
    // `packed_diagonal_parameter` returning `None` keeps the sentence but drops the entry —
    // the warning would claim a hold nothing records.
    let list = super::held_at_bound_list(&p, &[]);
    assert_eq!(
        list.iter()
            .map(|h| (h.name.as_str(), h.kind, h.index, h.class))
            .collect::<Vec<_>>(),
        [("ADD_ERR_MIX3", ParameterKind::Sigma, 1, Some(3))]
    );
    assert!(
        msg.contains("Their standard errors are not reported; "),
        "{msg}"
    );
    assert!(msg.ends_with(HELD_SENTENCE), "{msg}");
    assert_eq!(held_of(&details, "ADD_ERR_MIX3"), true);
}
