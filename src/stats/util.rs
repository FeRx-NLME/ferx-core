//! Shared f64-only aggregate numerics (log-sum-exp family, effective sample size).
//!
//! These are the *sample-combinator* helpers used by the marginal-likelihood
//! estimators (AGQ, importance sampling). They are deliberately separate from
//! [`crate::stats::special`], which is scoped to *differentiable* special
//! functions on the Dual2 gradient path; everything here is plain `f64` and off
//! any sensitivity path.
//!
//! There are intentionally TWO logsumexp functions with DIFFERENT edge-case
//! contracts — do not collapse them into one:
//!   * [`log_sum_exp`]         — simple: folds ALL elements into max/sum; NaN
//!                               propagates; returns `max` when `max` is not finite.
//!   * [`log_sum_exp_normalised`] — FILTERS non-finite before max/sum (NaN ignored),
//!                               has a dedicated +inf-splitting branch, and returns
//!                               the normalised weights alongside the lse.

/// Numerically stable `log Σ exp(xᵢ)` (simple contract, moved verbatim from
/// `agq::logsumexp`). All elements enter the fold, so a `NaN` propagates and a
/// `-inf` is ignored via `f64::max`; returns `max` when `max` is not finite.
pub(crate) fn log_sum_exp(xs: &[f64]) -> f64 {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// Stable `log(eᵃ + eᵇ)` for a two-component mixture denominator (moved verbatim
/// from `importance::logsumexp2`). `NEG_INFINITY`-safe.
pub(crate) fn log_sum_exp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Numerically stable `log Σ exp(xᵢ)` plus the normalised weights `wᵢ` (moved
/// verbatim from `importance::logsumexp_with_normalised`). Filters non-finite
/// entries out of the max/sum (so a `NaN` log-weight is ignored, not propagated),
/// splits mass equally across any `+inf` entries, and returns `(NEG_INFINITY, [])`
/// on empty input.
pub(crate) fn log_sum_exp_normalised(xs: &[f64]) -> (f64, Vec<f64>) {
    if xs.is_empty() {
        return (f64::NEG_INFINITY, Vec::new());
    }

    let n_pos_inf = xs
        .iter()
        .filter(|x| x.is_infinite() && x.is_sign_positive())
        .count();
    if n_pos_inf > 0 {
        let w = 1.0 / n_pos_inf as f64;
        let weights = xs
            .iter()
            .map(|x| {
                if x.is_infinite() && x.is_sign_positive() {
                    w
                } else {
                    0.0
                }
            })
            .collect();
        return (f64::INFINITY, weights);
    }

    let m = xs
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return (f64::NEG_INFINITY, vec![0.0; xs.len()]);
    }
    let mut sum = 0.0;
    let mut shifted: Vec<f64> = Vec::with_capacity(xs.len());
    for &x in xs {
        let s = if x.is_finite() { (x - m).exp() } else { 0.0 };
        shifted.push(s);
        sum += s;
    }
    let lse = m + sum.ln();
    let weights: Vec<f64> = if sum > 0.0 {
        shifted.iter().map(|&s| s / sum).collect()
    } else {
        vec![0.0; xs.len()]
    };
    (lse, weights)
}

/// Kish effective sample size `1 / Σ w̃ₖ²` from normalised weights (moved verbatim
/// from `importance::ess_from_weights`). Returns `0.0` when weights are empty or
/// all-zero. NOTE: this is the IS-weight ESS — NOT the Geyer autocorrelation
/// chain-ESS in `crate::stats::convergence::effective_sample_size`; the two are
/// different statistics and must never be conflated.
#[inline]
pub(crate) fn ess_from_weights(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if sum_sq > 0.0 {
        1.0 / sum_sq
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_is_stable_and_handles_sentinels() {
        let want = (1.0f64.exp() + 2.0f64.exp()).ln();
        assert!((log_sum_exp(&[1.0, 2.0]) - want).abs() < 1e-12);
        assert!((log_sum_exp(&[1e5, 1e5]) - (1e5 + 2.0f64.ln())).abs() < 1e-9);
        assert!((log_sum_exp(&[-1e20, 3.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn log_sum_exp2_matches_direct_and_handles_neginf() {
        let a = -1.3_f64;
        let b = 2.7_f64;
        let want = (a.exp() + b.exp()).ln();
        assert!((log_sum_exp2(a, b) - want).abs() < 1e-12);
        assert!((log_sum_exp2(f64::NEG_INFINITY, 5.0) - 5.0).abs() < 1e-12);
        assert!((log_sum_exp2(5.0, f64::NEG_INFINITY) - 5.0).abs() < 1e-12);
        assert_eq!(
            log_sum_exp2(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert!(
            (log_sum_exp2(1000.0, 1001.0) - (1001.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-9
        );
    }

    #[test]
    fn log_sum_exp_normalised_extreme_spread_and_shift_invariance() {
        let (lse, w) = log_sum_exp_normalised(&[1000.0, 1001.0]);
        assert!((lse - (1001.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-10);
        assert!(((w.iter().sum::<f64>()) - 1.0).abs() < 1e-12);
        assert!((w[1] / w[0] - 1.0_f64.exp()).abs() < 1e-10);
        let xs = vec![0.1, -0.5, 2.3, -1.2, 0.8];
        let (_, w1) = log_sum_exp_normalised(&xs);
        let (_, w2) = log_sum_exp_normalised(&xs.iter().map(|x| x + 17.4).collect::<Vec<_>>());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn log_sum_exp_normalised_edge_cases() {
        let (lse, w) = log_sum_exp_normalised(&[]);
        assert_eq!(lse, f64::NEG_INFINITY);
        assert!(w.is_empty());
        let (lse, w) = log_sum_exp_normalised(&[f64::NEG_INFINITY; 3]);
        assert_eq!(lse, f64::NEG_INFINITY);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
        let (lse, w) = log_sum_exp_normalised(&[f64::NAN, 2.0, 3.0, f64::NEG_INFINITY]);
        assert!((lse - (3.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-10);
        assert_eq!(w[0], 0.0);
        assert!(w[1] > 0.0);
        assert!(w[2] > w[1]);
        assert_eq!(w[3], 0.0);
        let (lse, w) = log_sum_exp_normalised(&[1.0, f64::INFINITY, f64::NAN, f64::INFINITY]);
        assert_eq!(lse, f64::INFINITY);
        assert_eq!(w, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn ess_from_weights_kish_and_zero_guard() {
        let k = 100usize;
        let w = vec![1.0 / k as f64; k];
        assert!((ess_from_weights(&w) - k as f64).abs() < 1e-10);
        assert_eq!(ess_from_weights(&[0.0, 0.0, 0.0]), 0.0);
        assert_eq!(ess_from_weights(&[]), 0.0);
    }
}
