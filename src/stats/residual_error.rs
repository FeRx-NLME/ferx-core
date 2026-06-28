use crate::types::{ErrorModel, ErrorSpec, ResidualCorrelation, SubjectResult};
use nalgebra::DMatrix;

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

/// Compute residual variances including fixed residual correlations from
/// `block_sigma`.
pub fn compute_r_diag_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> Vec<f64> {
    if correlations.is_empty() {
        return compute_r_diag(error_spec, ipreds, obs_cmts, sigma_values);
    }
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| {
            error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations)
        })
        .collect()
}

fn observation_time_key(obs_times: &[f64], j: usize) -> u64 {
    obs_times.get(j).copied().unwrap_or(0.0).to_bits()
}

fn observation_occasion_key(occasions: &[u32], j: usize) -> u32 {
    occasions.get(j).copied().unwrap_or(0)
}

fn same_residual_block(
    obs_times: &[f64],
    _obs_raw_times: &[f64],
    occasions: &[u32],
    j: usize,
    k: usize,
) -> bool {
    observation_time_key(obs_times, j) == observation_time_key(obs_times, k)
        && observation_occasion_key(occasions, j) == observation_occasion_key(occasions, k)
}

fn cross_observation_covariance(
    load_j: &[(usize, f64)],
    load_k: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let mut cov = 0.0;
    for corr in correlations {
        let j_has_i = load_j.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let j_has_j = load_j.iter().any(|(idx, _)| *idx == corr.sigma_j);
        let k_has_i = load_k.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let k_has_j = load_k.iter().any(|(idx, _)| *idx == corr.sigma_j);
        if (j_has_i && j_has_j) || (k_has_i && k_has_j) {
            continue;
        }
        let Some(&si) = sigma_values.get(corr.sigma_i) else {
            return f64::NAN;
        };
        let Some(&sj) = sigma_values.get(corr.sigma_j) else {
            return f64::NAN;
        };
        let cov_ij = corr.rho * si * sj;
        let ci_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        let cj_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        if let (Some(ci), Some(cj)) = (ci_j, cj_k) {
            cov += ci * cj * cov_ij;
        }

        let cj_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        let ci_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        if let (Some(cj), Some(ci)) = (cj_j, ci_k) {
            cov += cj * ci * cov_ij;
        }
    }
    cov
}

/// Build the subject-level residual covariance matrix `R`.
///
/// The diagonal is the existing per-observation residual variance, including
/// within-observation `block_sigma` terms for `combined(...)` endpoints. When a
/// `block_sigma` off-diagonal connects sigmas used by different endpoint rows,
/// rows at the same subject time and occasion receive the corresponding
/// cross-observation covariance. This mirrors NONMEM-style paired endpoint
/// records such as total/unbound assays written as separate rows.
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> DMatrix<f64> {
    // The bare-sigma `R` is exactly the custom-magnitude `R` with an all-ones
    // multiplier: an empty `mult` makes every observation's loading multiplier
    // default to 1.0. Delegate to the scaled builder so the diagonal-variance
    // and cross-covariance formulas live in one place (#484 review #9) — a fix
    // to the correlation math can no longer diverge the magnitude path from the
    // bare path.
    compute_r_matrix_with_correlations_scaled(
        error_spec,
        ipreds,
        obs_cmts,
        obs_times,
        obs_raw_times,
        occasions,
        sigma_values,
        correlations,
        &[],
    )
}

/// Build the subject residual covariance matrix `R` with a per-observation
/// custom magnitude (#484). `mult` is the `[obs][sigma-slot]` multiplier matrix
/// from [`crate::types::RuvMagnitude::eval_obs`]; each observation's sigma
/// loadings are scaled by its row before forming the diagonal variance and any
/// `block_sigma` cross-covariance. A `mult` whose rows are all ones reproduces
/// [`compute_r_matrix_with_correlations`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations_scaled(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: &[Vec<f64>],
) -> DMatrix<f64> {
    let n = ipreds.len();
    let mut r = DMatrix::<f64>::zeros(n, n);
    let ones: Vec<f64> = Vec::new();
    let row = |j: usize| -> &[f64] { mult.get(j).map(|v| v.as_slice()).unwrap_or(&ones) };
    for j in 0..n {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        r[(j, j)] = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, row(j));
    }
    if correlations.is_empty() {
        return r;
    }
    // Scale each observation's loadings by its multiplier row before computing
    // the cross-observation covariance.
    let scale_loadings = |j: usize| -> Vec<(usize, f64)> {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        let m = row(j);
        error_spec
            .sigma_loadings(cmt, f, sigma_values.len())
            .into_iter()
            .map(|(idx, coeff)| (idx, coeff * m.get(idx).copied().unwrap_or(1.0)))
            .collect()
    };
    let loadings: Vec<Vec<(usize, f64)>> = (0..n).map(scale_loadings).collect();
    for j in 0..n {
        if loadings[j].is_empty() {
            continue;
        }
        for k in (j + 1)..n {
            if loadings[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, j, k)
            {
                continue;
            }
            let cov = cross_observation_covariance(
                &loadings[j],
                &loadings[k],
                sigma_values,
                correlations,
            );
            if cov != 0.0 {
                r[(j, k)] = cov;
                r[(k, j)] = cov;
            }
        }
    }
    r
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

/// Compute IWRES using residual variances that include fixed `block_sigma`
/// correlations. With no correlations and no custom magnitude this is exactly
/// [`compute_iwres`].
///
/// `ruv_mult` is the per-observation custom-magnitude multiplier matrix (#484)
/// from [`crate::types::CompiledModel::ruv_obs_mult`]; `None` reproduces the
/// legacy unscaled IWRES. When present, each observation's residual variance is
/// scaled by its multiplier row so the sdtab IWRES matches the magnitude-aware
/// OFV variance (otherwise late/covariate-varying rows are systematically
/// mis-scaled).
pub fn compute_iwres_with_correlations(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    if let Some(mult) = ruv_mult {
        // variance_at_scaled handles empty `correlations` (no cross terms) and a
        // short/empty multiplier row (slots default to 1.0), so this one path
        // covers correlated and uncorrelated custom-magnitude models alike.
        return observations
            .iter()
            .zip(ipreds.iter())
            .zip(obs_cmts.iter())
            .enumerate()
            .map(|(j, ((&y, &f), &cmt))| {
                let m = mult.get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                let v = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, m);
                (y - f) / v.sqrt()
            })
            .collect();
    }
    if correlations.is_empty() {
        return compute_iwres(observations, ipreds, obs_cmts, error_spec, sigma_values);
    }
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations);
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
    use crate::types::{EndpointError, GradientMethod};
    use approx::assert_relative_eq;
    use std::collections::HashMap;

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
    fn test_combined_variance_with_residual_correlation() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_empty_matches_diagonal() {
        // With no correlations the helper must be identical to compute_r_diag.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0, 20.0];
        let cmts = [0usize, 0];
        let sigma = [0.2, 1.0];
        let plain = compute_r_diag(&spec, &ipreds, &cmts, &sigma);
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[]);
        assert_eq!(plain, with);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_applies_cross_term() {
        // Each observation's diagonal variance gains the 2·f·ρ·σ₁·σ₂ cross term.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0];
        let cmts = [0usize];
        let sigma = [0.2, 1.0];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[corr]);
        assert_relative_eq!(with[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_links_paired_endpoints() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0];
        let cmts = [1usize, 2, 2];
        let times = [1.0, 1.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
        );
        assert_relative_eq!(r[(0, 0)], (50.0_f64 * 0.3).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(1, 1)], (5.0_f64 * 0.2).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(1, 0)], r[(0, 1)], epsilon = 1e-12);
        assert_eq!(r[(0, 2)], 0.0);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_uses_shifted_time_after_reset() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let shifted_times = [1.0, 1.0, 101.0, 101.0];
        let raw_times = [1.0, 1.0, 1.0, 1.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };

        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &shifted_times,
            &raw_times,
            &[],
            &sigma,
            &[corr],
        );

        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(2, 3)], 40.0 * 4.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(1, 2)], 0.0);
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

    #[test]
    fn test_compute_iwres_with_correlations_applies_cross_term() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let result = compute_iwres_with_correlations(
            &[12.0],
            &[10.0],
            &[1],
            &spec,
            &[0.2, 1.0],
            &[corr],
            None,
        );
        assert_relative_eq!(result[0], 2.0 / 7.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_compute_iwres_with_correlations_empty_matches_diagonal() {
        let spec = ErrorSpec::Single(ErrorModel::Additive);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let plain = compute_iwres(&obs, &ipreds, &obs_cmts, &spec, &[1.0]);
        let with =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &[1.0], &[], None);
        assert_eq!(plain, with);
    }

    #[test]
    fn test_compute_iwres_with_correlations_applies_custom_magnitude() {
        // #484 review #4: the sdtab IWRES must use the per-observation magnitude
        // multiplier, so a row whose multiplier ≠ 1 is scaled by it. Proportional
        // error: V = (f·m·σ)², so IWRES = (y−f)/(f·m·σ).
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let sigma = [0.2];
        // Row 0 bare (mult 1), row 1 inflated by 2.
        let mult = vec![vec![1.0], vec![2.0]];
        let scaled = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&mult),
        );
        assert_relative_eq!(scaled[0], 2.0 / (10.0 * 1.0 * 0.2), epsilon = 1e-12);
        assert_relative_eq!(scaled[1], 2.0 / (20.0 * 2.0 * 0.2), epsilon = 1e-12);

        // An all-ones multiplier reproduces the unscaled IWRES exactly.
        let ones = vec![vec![1.0], vec![1.0]];
        let unit = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&ones),
        );
        let bare =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &sigma, &[], None);
        assert_relative_eq!(unit[0], bare[0], epsilon = 1e-12);
        assert_relative_eq!(unit[1], bare[1], epsilon = 1e-12);
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
