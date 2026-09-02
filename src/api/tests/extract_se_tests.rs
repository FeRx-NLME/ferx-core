use super::extract_standard_errors;
use crate::types::*;
use nalgebra::DMatrix;

/// Helper for tests where the BSV omega is a single diagonal eta and only
/// the IOV block varies. BSV-omega tests below build their own template
/// inline because they need a 3-eta block omega.
fn make_template(omega_iov: Option<OmegaMatrix>, kappa_fixed: Vec<bool>) -> ModelParameters {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0],
        theta_names: vec!["TVCL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov,
        kappa_fixed,
        mixture: None,
    }
}

// ── BSV omega ────────────────────────────────────────────────────────────

/// Block omega with n_eta = 3 (numerically diagonal).  se_omega is now the
/// full lower triangle (length 6), column-major.  The diagonal elements at
/// LT positions 0, 3, 5 should match the old formula; off-diagonals should
/// also be finite (non-NaN).
#[test]
fn test_se_omega_block_n3_full_lower_triangle() {
    let mut mat = DMatrix::<f64>::zeros(3, 3);
    mat[(0, 0)] = 0.04;
    mat[(1, 1)] = 0.09;
    mat[(2, 2)] = 0.16;
    let omega = OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into(), "E3".into()], false);
    let template = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0],
        theta_names: vec!["TVCL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega,
        omega_fixed: vec![false; 3],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    // Packed layout: theta(1) + omega_block(6) + sigma(1) = 8.
    // Within the omega block (start = 1): L[0,0]=1, L[1,0]=2, L[2,0]=3,
    // L[1,1]=4, L[2,1]=5, L[2,2]=6.
    let n = 8;
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        cov[(i, i)] = ((i + 1) as f64).powi(2);
    }
    let (_, se_omega, _, _) = extract_standard_errors(&Some(cov), &template);
    let se = se_omega.unwrap();
    // Full LT: [omega(0,0), omega(1,0), omega(2,0), omega(1,1), omega(2,1), omega(2,2)]
    assert_eq!(se.len(), 6);
    // Diagonal SEs: same as before (omega is numerically diagonal → off-diag L=0).
    // omega(0,0) at LT[0]: 2 * 0.04 * 2.0 = 0.16
    assert!((se[0] - 0.16).abs() < 1e-12, "se(0,0) = {}", se[0]);
    // omega(1,1) at LT[3]: 2 * 0.09 * 5.0 = 0.90
    assert!((se[3] - 0.90).abs() < 1e-12, "se(1,1) = {}", se[3]);
    // omega(2,2) at LT[5]: 2 * 0.16 * 7.0 = 2.24
    assert!((se[5] - 2.24).abs() < 1e-12, "se(2,2) = {}", se[5]);
    // Off-diagonals should be finite
    for (idx, &v) in se.iter().enumerate() {
        assert!(v.is_finite(), "se[{}] not finite", idx);
    }

    // Verify the omega_se_at helper
    use crate::types::omega_se_at;
    let se_opt = Some(se);
    assert!((omega_se_at(&se_opt, 3, 0, 0).unwrap() - 0.16).abs() < 1e-12);
    assert!((omega_se_at(&se_opt, 3, 1, 1).unwrap() - 0.90).abs() < 1e-12);
    assert!((omega_se_at(&se_opt, 3, 2, 2).unwrap() - 2.24).abs() < 1e-12);
    // Symmetric: omega_se_at(1,0) == omega_se_at(0,1)
    assert_eq!(omega_se_at(&se_opt, 3, 1, 0), omega_se_at(&se_opt, 3, 0, 1));
}

/// Block omega with non-zero off-diagonals: verify off-diagonal SEs are
/// positive and that they differ from the (incorrect) zero that would
/// result from a diagonal-only implementation.
#[test]
fn test_se_omega_block_offdiag_positive() {
    // Ω = [[0.09, 0.02], [0.02, 0.04]]  (corr ≈ 0.33)
    let mut mat = DMatrix::<f64>::zeros(2, 2);
    mat[(0, 0)] = 0.09;
    mat[(1, 1)] = 0.04;
    mat[(0, 1)] = 0.02;
    mat[(1, 0)] = 0.02;
    let omega = OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into()], false);
    let template = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0],
        theta_names: vec!["TVCL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega,
        omega_fixed: vec![false; 2],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    // Packed: theta(1) + omega_block(3) + sigma(1) = 5.  Identity cov.
    let cov = Some(DMatrix::<f64>::identity(5, 5));
    let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
    let se = se_omega.unwrap();
    // Full LT: [omega(0,0), omega(1,0), omega(1,1)]
    assert_eq!(se.len(), 3);
    assert!(se[0] > 0.0, "diagonal SE(0,0) should be positive");
    assert!(se[1] > 0.0, "off-diagonal SE(1,0) should be positive");
    assert!(se[2] > 0.0, "diagonal SE(1,1) should be positive");
    // omega_se_at helper
    use crate::types::omega_se_at;
    let se_opt = Some(se);
    assert!(omega_se_at(&se_opt, 2, 1, 0).unwrap() > 0.0);
    // diagonal-only format returns None for off-diag
    let diag_only = Some(vec![0.1, 0.2]);
    assert!(omega_se_at(&diag_only, 2, 1, 0).is_none());
}

/// Diagonal omega path is unaffected by the fix; this guards the simple case.
#[test]
fn test_se_omega_diagonal_unchanged() {
    let omega = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["E1".into(), "E2".into()]);
    let template = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0],
        theta_names: vec!["TVCL".into()],
        theta_lower: vec![0.1],
        theta_upper: vec![50.0],
        theta_fixed: vec![false],
        omega,
        omega_fixed: vec![false; 2],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    // Packed layout: theta(1) + omega_diag(2) + sigma(1) = 4. Identity cov.
    let cov = Some(DMatrix::<f64>::identity(4, 4));
    let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
    let se = se_omega.unwrap();
    assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
    assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
}

/// Theta SE back-transform must respect the packing scale. A log-packed theta
/// (lower bound >= 0) is optimized as log(theta), so SE(theta) = theta * SE(x);
/// an identity-packed theta (negative lower bound, e.g. an exposure–hazard slope
/// or covariate exponent) is optimized on the natural scale, so SE(theta) = SE(x)
/// with no multiply. Regression for the bug where every theta was multiplied by
/// its estimate, mis-scaling (and sign-flipping for negative estimates) the SE
/// of identity-packed thetas.
#[test]
fn test_se_theta_respects_packing_scale() {
    // theta[0] = TVCL, lower 0.1 >= 0  → log-packed
    // theta[1] = BETA, lower -10 < 0   → identity-packed (can be negative)
    let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["E1".into()]);
    let template = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![2.0, 0.25],
        theta_names: vec!["TVCL".into(), "BETA".into()],
        theta_lower: vec![0.1, -10.0],
        theta_upper: vec![50.0, 10.0],
        theta_fixed: vec![false, false],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: vec![],
        mixture: None,
    };
    // Packed layout: theta(2) + omega(1) + sigma(1) = 4.
    // Set diagonal so packed SEs are theta:0.1, theta:0.3 (the rest unused here).
    let mut cov = DMatrix::<f64>::zeros(4, 4);
    cov[(0, 0)] = 0.1_f64.powi(2); // se_packed[0] = 0.1
    cov[(1, 1)] = 0.3_f64.powi(2); // se_packed[1] = 0.3
    cov[(2, 2)] = 1.0;
    cov[(3, 3)] = 1.0;
    let (se_theta, _, _, _) = extract_standard_errors(&Some(cov), &template);
    let se = se_theta.unwrap();
    // log-packed: SE = theta * SE(x) = 2.0 * 0.1 = 0.2
    assert!(
        (se[0] - 0.2).abs() < 1e-12,
        "log-packed theta SE = {}",
        se[0]
    );
    // identity-packed: SE = SE(x) = 0.3 (NOT 0.25 * 0.3 = 0.075)
    assert!(
        (se[1] - 0.3).abs() < 1e-12,
        "identity-packed theta SE = {} (must be the packed SE, not estimate-scaled)",
        se[1]
    );
}

// ── IOV (kappa) ──────────────────────────────────────────────────────────

/// Identity covariance matrix gives unit SEs on the packed scale.  After
/// the delta-method transform `SE(var_i) = 2 * var_i * SE(log L_ii)`, the
/// returned se_kappa[i] equals 2 * variance_i for diagonal IOV.
#[test]
fn test_se_kappa_diagonal_uses_correct_index() {
    let iov = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["KAPPA_CL".into(), "KAPPA_V".into()]);
    let template = make_template(Some(iov), vec![false, false]);
    // Packed layout: [theta, omega(1), sigma, kappa_1, kappa_2] → 5 entries.
    // Identity cov means se_packed = [1, 1, 1, 1, 1].
    let cov = Some(DMatrix::<f64>::identity(5, 5));
    let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
    let se = se_kappa.expect("se_kappa should be Some when omega_iov is set");
    assert_eq!(se.len(), 2);
    assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
    assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
}

/// Block kappa with n=3: column-major Cholesky packing places L[1,1] at
/// offset 3 and L[2,2] at offset 5 within the IOV block.  Mirrors the BSV
/// regression test above.
#[test]
fn test_se_kappa_block_n3_column_major_indexing() {
    let mut mat = DMatrix::<f64>::zeros(3, 3);
    mat[(0, 0)] = 0.04;
    mat[(1, 1)] = 0.09;
    mat[(2, 2)] = 0.16;
    let _chol = mat.clone().cholesky().unwrap().l();
    let iov = OmegaMatrix::from_matrix(mat, vec!["K1".into(), "K2".into(), "K3".into()], false);
    let template = make_template(Some(iov), vec![false; 3]);
    // Packed layout: theta(1) + omega_diag(1) + sigma(1) + kappa_block(6) = 9
    // Within the kappa block (start = 3): L11=0, L21=1, L31=2, L22=3, L32=4, L33=5
    // → diagonals at packed indices 3, 6, 8.
    // Build a cov matrix with distinct diagonal entries so we can verify
    // which index each SE pulls from.
    let n = 9;
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
    }
    let (_, _, _, se_kappa) = extract_standard_errors(&Some(cov), &template);
    let se = se_kappa.unwrap();
    // L[0,0] at idx 3 → se_packed = 4 → se_kappa[0] = 2 * 0.04 * 4 = 0.32
    assert!((se[0] - 2.0 * 0.04 * 4.0).abs() < 1e-12, "got {}", se[0]);
    // L[1,1] at idx 6 → se_packed = 7 → se_kappa[1] = 2 * 0.09 * 7 = 1.26
    assert!((se[1] - 2.0 * 0.09 * 7.0).abs() < 1e-12, "got {}", se[1]);
    // L[2,2] at idx 8 → se_packed = 9 → se_kappa[2] = 2 * 0.16 * 9 = 2.88
    assert!((se[2] - 2.0 * 0.16 * 9.0).abs() < 1e-12, "got {}", se[2]);
}

#[test]
fn test_se_kappa_none_when_no_iov() {
    let template = make_template(None, vec![]);
    let cov = Some(DMatrix::<f64>::identity(3, 3));
    let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
    assert!(se_kappa.is_none());
}

#[test]
fn test_se_kappa_none_when_no_cov() {
    let iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
    let template = make_template(Some(iov), vec![false]);
    let (_, _, _, se_kappa) = extract_standard_errors(&None, &template);
    assert!(se_kappa.is_none());
}
