use super::*;
use nalgebra::DMatrix;

#[test]
fn test_cov_diagnostics_none_input() {
    let (ev, cn) = cov_diagnostics(None);
    assert!(ev.is_none());
    assert!(cn.is_none());
}

#[test]
fn test_cov_diagnostics_fewer_than_two_free_params() {
    // 2×2 matrix where only one param is free (second has zero diagonal)
    let mut m = DMatrix::<f64>::zeros(2, 2);
    m[(0, 0)] = 4.0;
    let (ev, cn) = cov_diagnostics(Some(&m));
    assert!(ev.is_none());
    assert!(cn.is_none());
}

#[test]
fn test_cov_diagnostics_excludes_fixed_params_zero_diagonal() {
    // 3×3 covariance; middle param is fixed (zero row/col).
    // Free subblock [[4, 0.5], [0.5, 2]] is non-singular, so condition
    // number must be finite and eigenvalues length must be 2.
    let mut m = DMatrix::<f64>::zeros(3, 3);
    m[(0, 0)] = 4.0;
    m[(0, 2)] = 0.5;
    m[(2, 0)] = 0.5;
    m[(2, 2)] = 2.0;
    let (ev, cn) = cov_diagnostics(Some(&m));
    let ev = ev.expect("eigenvalues must be Some");
    let cn = cn.expect("condition_number must be Some");
    assert_eq!(ev.len(), 2, "must have 2 eigenvalues (one per free param)");
    assert!(
        cn.is_finite(),
        "condition_number must be finite for non-singular subblock"
    );
    assert!(cn > 0.0);
    // Eigenvalues must be sorted descending
    assert!(ev[0] >= ev[1]);
}

#[test]
fn test_cov_diagnostics_inf_condition_number_for_non_positive_eigenvalue() {
    // Construct a 2×2 covariance matrix whose free-param correlation matrix
    // is [[1, r], [r, 1]] with |r| > 1 — not PSD, so min eigenvalue < 0.
    // (r = 1.5 → eigenvalues 2.5 and -0.5)
    let mut m = DMatrix::<f64>::zeros(2, 2);
    m[(0, 0)] = 1.0;
    m[(0, 1)] = 1.5; // cor = 1.5/sqrt(1*1) = 1.5 > 1 → non-PSD
    m[(1, 0)] = 1.5;
    m[(1, 1)] = 1.0;
    let (ev, cn) = cov_diagnostics(Some(&m));
    let cn = cn.expect("condition_number must be Some");
    assert!(
        cn.is_infinite(),
        "condition_number must be Inf when min eigenvalue ≤ 0, got {cn}"
    );
    let ev = ev.expect("eigenvalues must be Some");
    assert!(
        ev.last().copied().unwrap_or(1.0) <= 0.0,
        "min eigenvalue must be ≤ 0"
    );
}

#[test]
fn test_cov_diagnostics_inf_condition_number_for_near_zero_eigenvalue() {
    // Simulate a floating-point near-zero negative eigenvalue (e.g. -1e-15)
    // that a well-conditioned matrix could produce due to numerical noise.
    // The tolerance guard (> 1e-10) must treat this as singular → INFINITY.
    let mut m = DMatrix::<f64>::zeros(2, 2);
    m[(0, 0)] = 1.0;
    m[(0, 1)] = 1.0 - 1e-15; // cor ≈ 1 → min eigenvalue ≈ 0 (or tiny negative)
    m[(1, 0)] = 1.0 - 1e-15;
    m[(1, 1)] = 1.0;
    let (_, cn) = cov_diagnostics(Some(&m));
    let cn = cn.expect("condition_number must be Some");
    assert!(
        cn.is_infinite(),
        "condition_number must be Inf for near-singular matrix (min_ev ≤ 1e-10), got {cn}"
    );
}

#[test]
fn test_cov_diagnostics_identity_covariance() {
    // Diagonal covariance → correlation matrix is identity → all eigenvalues 1.
    let m = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0]));
    let (ev, cn) = cov_diagnostics(Some(&m));
    let ev = ev.expect("eigenvalues must be Some");
    let cn = cn.expect("condition_number must be Some");
    for &e in &ev {
        assert!((e - 1.0).abs() < 1e-12, "eigenvalue must be 1.0, got {e}");
    }
    assert!(
        (cn - 1.0).abs() < 1e-12,
        "condition_number must be 1.0, got {cn}"
    );
}

// ── resolve_covariance_status ────────────────────────────────────────────

#[test]
fn cov_status_not_requested_when_step_off() {
    // When the covariance step is off, neither a (stale) covariance matrix
    // nor a fallback result can change the reported status.
    assert_eq!(
        resolve_covariance_status(false, true, true),
        CovarianceStatus::NotRequested
    );
    assert_eq!(
        resolve_covariance_status(false, false, false),
        CovarianceStatus::NotRequested
    );
}

#[test]
fn cov_status_computed_takes_precedence_over_fallback() {
    // A real covariance matrix always wins, even if a fallback also ran.
    assert_eq!(
        resolve_covariance_status(true, true, false),
        CovarianceStatus::Computed
    );
    assert_eq!(
        resolve_covariance_status(true, true, true),
        CovarianceStatus::Computed
    );
}

#[test]
fn cov_status_sir_fallback_when_no_matrix_but_fallback_ran() {
    // The branch the SIR-fallback wiring depends on: no H⁻¹ covariance, but
    // the |eigenvalue|-rectified SIR fallback produced a result.
    assert_eq!(
        resolve_covariance_status(true, false, true),
        CovarianceStatus::SirFallback
    );
}

#[test]
fn cov_status_failed_when_requested_but_nothing_produced() {
    assert_eq!(
        resolve_covariance_status(true, false, false),
        CovarianceStatus::Failed
    );
}

// ── is_last_estimating_stage: covariance runs once, at chain end (#615) ──
use EstimationMethod::*;

#[test]
fn cov_stage_single_method_is_last() {
    assert!(is_last_estimating_stage(&[Foce], 0, false));
}

#[test]
fn cov_stage_only_final_estimator_in_plain_chain() {
    // [foce, saem]: covariance only on saem (the last stage), never on foce.
    let chain = [Foce, Saem];
    assert!(!is_last_estimating_stage(&chain, 0, false));
    assert!(is_last_estimating_stage(&chain, 1, false));
}

#[test]
fn cov_stage_estimating_imp_owns_step_not_predecessor() {
    // [saem, imp] with estimating IMP (imp_eval_only = false): the trailing
    // IMP is a real estimator and owns the covariance step, so saem must NOT
    // also run it — otherwise covariance is computed twice (#615).
    let chain = [Saem, Imp];
    assert!(!is_last_estimating_stage(&chain, 0, false));
    assert!(is_last_estimating_stage(&chain, 1, false));
}

#[test]
fn cov_stage_eval_only_imp_cedes_step_to_predecessor() {
    // [saem, imp] with imp_eval_only = true: trailing IMP is a likelihood
    // evaluation, so saem is the last estimating stage and owns covariance.
    let chain = [Saem, Imp];
    assert!(is_last_estimating_stage(&chain, 0, true));
    // The eval-only IMP stage itself never runs the covariance step (handled
    // by the eval-only branch in fit), but as the last stage it still reports
    // as last-estimating; gating there is a no-op since it skips covariance.
    assert!(is_last_estimating_stage(&chain, 1, true));
}

#[test]
fn cov_stage_three_method_chain_estimating_imp() {
    // [foce, saem, imp] estimating: only the final imp owns covariance.
    let chain = [Foce, Saem, Imp];
    assert!(!is_last_estimating_stage(&chain, 0, false));
    assert!(!is_last_estimating_stage(&chain, 1, false));
    assert!(is_last_estimating_stage(&chain, 2, false));
}
