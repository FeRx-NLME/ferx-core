use crate::types::{ErrorModel, ErrorSpec, SubjectResult};

const MIN_VARIANCE: f64 = 1e-12;

/// Compute residual variance for a single observation
/// sigma_values: [sigma1] for additive/proportional, [sigma1, sigma2] for combined
pub fn residual_variance(error_model: ErrorModel, f_pred: f64, sigma_values: &[f64]) -> f64 {
    let v = match error_model {
        ErrorModel::Additive => {
            // V = sigma1^2
            sigma_values[0] * sigma_values[0]
        }
        ErrorModel::Proportional => {
            // V = (f * sigma1)^2
            let fs = f_pred * sigma_values[0];
            fs * fs
        }
        ErrorModel::Combined => {
            // V = (f * sigma1)^2 + sigma2^2
            let prop = f_pred * sigma_values[0];
            prop * prop + sigma_values[1] * sigma_values[1]
        }
    };
    v.max(MIN_VARIANCE)
}

/// Compute the R diagonal (vector of residual variances for all observations),
/// dispatching the error model per observation by compartment. `obs_cmts` is
/// parallel to `ipreds` (`subject.obs_cmts`); for single-endpoint models the
/// CMT is ignored.
pub fn compute_r_diag(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
) -> Vec<f64> {
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| error_spec.variance_at(cmt, f, sigma_values))
        .collect()
}

/// Individual weighted residual: IWRES_j = (y_j - f_j) / sqrt(V_j)
pub fn iwres(obs: f64, ipred: f64, error_model: ErrorModel, sigma_values: &[f64]) -> f64 {
    let v = residual_variance(error_model, ipred, sigma_values);
    (obs - ipred) / v.sqrt()
}

/// Compute IWRES for all observations, dispatching the error model per
/// observation by compartment (`obs_cmts` parallel to `observations`/`ipreds`).
pub fn compute_iwres(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
) -> Vec<f64> {
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at(cmt, f, sigma_values);
            (y - f) / v.sqrt()
        })
        .collect()
}

/// Compute pooled lag-1 autocorrelation diagnostics on IWRES across subjects.
///
/// Subjects with fewer than 2 finite IWRES values are skipped.
/// Returns `(lag1_r, durbin_watson)` where DW = 2.0 indicates no autocorrelation.
/// Returns `(f64::NAN, f64::NAN)` when no subject has enough observations.
pub fn iwres_autocorrelation(subjects: &[SubjectResult]) -> (f64, f64) {
    // Accumulators for Durbin-Watson: Σ(eᵢ - eᵢ₋₁)², Σeᵢ²
    let mut dw_num = 0.0_f64;
    let mut dw_den = 0.0_f64;

    // Accumulators for pooled lag-1 Pearson r
    let mut sum_xy = 0.0_f64; // Σ e[t] * e[t+1]
    let mut sum_x = 0.0_f64; // Σ e[t]
    let mut sum_y = 0.0_f64; // Σ e[t+1]
    let mut sum_x2 = 0.0_f64; // Σ e[t]²
    let mut sum_y2 = 0.0_f64; // Σ e[t+1]²
    let mut n_pairs: usize = 0;

    for subj in subjects {
        let valid: Vec<f64> = subj
            .iwres
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if valid.len() < 2 {
            continue;
        }
        // DW accumulation
        dw_den += valid.iter().map(|e| e * e).sum::<f64>();
        for w in valid.windows(2) {
            let diff = w[1] - w[0];
            dw_num += diff * diff;
        }
        // Lag-1 Pearson accumulation
        for w in valid.windows(2) {
            let x = w[0];
            let y = w[1];
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xy += x * y;
            n_pairs += 1;
        }
    }

    if n_pairs == 0 {
        return (f64::NAN, f64::NAN);
    }

    let n = n_pairs as f64;
    let lag1_r = {
        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    };

    let dw = if dw_den == 0.0 {
        f64::NAN
    } else {
        dw_num / dw_den
    };

    (lag1_r, dw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GradientMethod;
    use approx::assert_relative_eq;

    #[test]
    fn test_additive_variance() {
        let v = residual_variance(ErrorModel::Additive, 10.0, &[0.5]);
        assert_relative_eq!(v, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_additive_variance_independent_of_prediction() {
        let v1 = residual_variance(ErrorModel::Additive, 1.0, &[0.5]);
        let v2 = residual_variance(ErrorModel::Additive, 100.0, &[0.5]);
        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance() {
        // V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0
        let v = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        assert_relative_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance_scales_with_prediction() {
        let v1 = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        let v2 = residual_variance(ErrorModel::Proportional, 20.0, &[0.1]);
        assert_relative_eq!(v2 / v1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance() {
        // V = (f * sigma1)^2 + sigma2^2 = (10 * 0.1)^2 + 0.5^2 = 1.0 + 0.25 = 1.25
        let v = residual_variance(ErrorModel::Combined, 10.0, &[0.1, 0.5]);
        assert_relative_eq!(v, 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_min_variance_floor() {
        // Proportional with f=0 gives V=0, should be floored to MIN_VARIANCE
        let v = residual_variance(ErrorModel::Proportional, 0.0, &[0.1]);
        assert_relative_eq!(v, MIN_VARIANCE, epsilon = 1e-20);
    }

    #[test]
    fn test_iwres_perfect_prediction() {
        let r = iwres(10.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iwres_known_value() {
        // IWRES = (y - f) / sqrt(V) = (12 - 10) / sqrt(1) = 2.0
        let r = iwres(12.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_length() {
        // Single-endpoint model (Additive): CMT is ignored.
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let ipreds = vec![1.0, 2.0, 3.0];
        let obs_cmts = vec![1, 1, 1];
        let r = compute_r_diag(&model.error_spec, &ipreds, &obs_cmts, &[0.5]);
        assert_eq!(r.len(), 3);
        // Additive variance is sigma^2 regardless of prediction/CMT.
        for v in &r {
            assert_relative_eq!(*v, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_compute_iwres_vectorized() {
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let obs = vec![12.0, 22.0];
        let ipreds = vec![10.0, 20.0];
        let obs_cmts = vec![1, 1];
        let result = compute_iwres(&obs, &ipreds, &obs_cmts, &model.error_spec, &[1.0]);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
    }

    fn make_subject(iwres: Vec<f64>) -> SubjectResult {
        use nalgebra::DVector;
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::zeros(0),
            ipred: vec![0.0; iwres.len()],
            pred: vec![0.0; iwres.len()],
            iwres,
            cwres: vec![],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![],
            n_obs: 0,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
        }
    }

    #[test]
    fn test_dw_monotone_positive_autocorrelation() {
        // Monotonically increasing → strong positive autocorrelation → DW near 0
        let subj = make_subject(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw < 1.5,
            "expected DW < 1.5 for monotone sequence, got {dw}"
        );
        assert!(r > 0.5, "expected positive lag-1 r, got {r}");
    }

    #[test]
    fn test_dw_alternating_negative_autocorrelation() {
        // Alternating signs → strong negative autocorrelation → DW near 4
        let subj = make_subject(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 2.5,
            "expected DW > 2.5 for alternating sequence, got {dw}"
        );
    }

    #[test]
    fn test_dw_uncorrelated_near_two() {
        // White-noise-like residuals → DW near 2 (no autocorrelation)
        let subj = make_subject(vec![1.0, -0.5, 0.2, 0.8, -0.3, -0.7, 0.4, 0.1]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 1.5 && dw < 2.5,
            "expected DW near 2 for white-noise sequence, got {dw}"
        );
    }

    #[test]
    fn test_nan_iwres_skipped() {
        let subj = make_subject(vec![f64::NAN, 1.0, 2.0, f64::NAN, 3.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        // Should not panic and should produce a finite result based on valid values
        assert!(
            dw.is_finite(),
            "DW should be finite after skipping NaN entries"
        );
    }

    #[test]
    fn test_single_observation_subject_skipped() {
        let single = make_subject(vec![1.0]);
        let multi = make_subject(vec![1.0, 2.0, 3.0]);
        let (r, dw) = iwres_autocorrelation(&[single, multi]);
        assert!(dw.is_finite());
        assert!(r.is_finite());
    }

    #[test]
    fn test_no_valid_subjects_returns_nan() {
        let subj = make_subject(vec![1.0]); // < 2 valid
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(r.is_nan());
        assert!(dw.is_nan());
    }
}
