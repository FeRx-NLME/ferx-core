use super::compute_param_corr;
use nalgebra::DMatrix;

fn names(ns: &[&str]) -> Vec<String> {
    ns.iter().map(|s| s.to_string()).collect()
}

/// Lognormal pair: uses the bivariate lognormal formula.
#[test]
fn lognormal_pair() {
    // ω = [[0.09, 0.045], [0.045, 0.09]]
    let w11 = 0.09_f64;
    let w12 = 0.045_f64;
    let mut omega = DMatrix::zeros(2, 2);
    omega[(0, 0)] = w11;
    omega[(1, 1)] = w11;
    omega[(0, 1)] = w12;
    omega[(1, 0)] = w12;

    let mut warnings = Vec::new();
    let corr = compute_param_corr(
        &omega,
        &[true, true],
        &names(&["ETA_CL", "ETA_V"]),
        "test",
        &mut warnings,
    )
    .expect("should return Some for block omega");

    assert!(warnings.is_empty());
    // diagonal must be 1
    assert!((corr[(0, 0)] - 1.0).abs() < 1e-12);
    assert!((corr[(1, 1)] - 1.0).abs() < 1e-12);
    // lognormal formula: (exp(w12) - 1) / sqrt((exp(w11)-1)*(exp(w11)-1))
    let expected = (w12.exp() - 1.0) / (w11.exp() - 1.0);
    assert!(
        (corr[(0, 1)] - expected).abs() < 1e-10,
        "lognormal corr {:.6} != expected {:.6}",
        corr[(0, 1)],
        expected
    );
}

/// Additive pair: falls back to eta-level formula (cov/sqrt(var_i*var_j)).
#[test]
fn additive_pair() {
    let w11 = 4.0_f64;
    let w12 = 1.0_f64;
    let mut omega = DMatrix::zeros(2, 2);
    omega[(0, 0)] = w11;
    omega[(1, 1)] = w11;
    omega[(0, 1)] = w12;
    omega[(1, 0)] = w12;

    let mut warnings = Vec::new();
    let corr = compute_param_corr(
        &omega,
        &[false, false],
        &names(&["ETA_CL", "ETA_V"]),
        "test",
        &mut warnings,
    )
    .expect("should return Some");

    assert!(warnings.is_empty());
    let expected = w12 / w11;
    assert!((corr[(0, 1)] - expected).abs() < 1e-12);
}

/// Mixed pair (one lognormal, one additive) falls back to eta-level and emits a warning.
#[test]
fn mixed_pair_warns_and_falls_back() {
    let w11 = 0.09_f64;
    let w12 = 0.03_f64;
    let mut omega = DMatrix::zeros(2, 2);
    omega[(0, 0)] = w11;
    omega[(1, 1)] = w11;
    omega[(0, 1)] = w12;
    omega[(1, 0)] = w12;

    let mut warnings = Vec::new();
    let corr = compute_param_corr(
        &omega,
        &[true, false],
        &names(&["ETA_CL", "ETA_V"]),
        "test",
        &mut warnings,
    )
    .expect("should return Some");

    assert_eq!(warnings.len(), 1, "expected one warning");
    assert!(warnings[0].contains("mixed"));
    // eta-level fallback
    let expected = w12 / w11;
    assert!((corr[(0, 1)] - expected).abs() < 1e-12);
}

/// Diagonal omega returns None (no off-diagonals to report).
#[test]
fn diagonal_returns_none() {
    let mut omega = DMatrix::zeros(2, 2);
    omega[(0, 0)] = 0.09;
    omega[(1, 1)] = 0.04;
    let mut warnings = Vec::new();
    let result = compute_param_corr(
        &omega,
        &[true, true],
        &names(&["A", "B"]),
        "test",
        &mut warnings,
    );
    assert!(result.is_none());
    assert!(warnings.is_empty());
}
