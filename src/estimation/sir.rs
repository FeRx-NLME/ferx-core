//! Sampling Importance Resampling (SIR) for parameter uncertainty estimation.
//!
//! Implements the SIR procedure described in Dosne et al. (2017):
//! "Improving the estimation of parameter uncertainty distributions in
//! nonlinear mixed effects models using sampling importance resampling"
//!
//! SIR provides a non-parametric estimate of parameter uncertainty that is
//! more robust than the asymptotic covariance matrix.

use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::pop_nll;
use crate::estimation::parameterization::{
    compute_bounds, compute_mu_k, pack_params, packed_fixed_mask, unpack_params,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{weighted::WeightedIndex, ChiSquared, Distribution, StandardNormal};
use rayon::prelude::*;

/// Results from the SIR procedure.
#[derive(Debug, Clone)]
pub struct SirResult {
    /// 95% CI (2.5th, 97.5th percentile) for each theta on original scale
    pub ci_theta: Vec<(f64, f64)>,
    /// 95% CI for each omega diagonal element
    pub ci_omega: Vec<(f64, f64)>,
    /// 95% CI for each sigma
    pub ci_sigma: Vec<(f64, f64)>,
    /// Effective sample size (ESS = 1 / sum(w_k^2))
    pub effective_sample_size: f64,
    /// Resampled packed parameter vectors, retained when
    /// `FitOptions.sir_keep_samples = true`. `None` otherwise.
    /// Length equals `FitOptions.sir_resamples` when populated.
    pub resamples_packed: Option<Vec<Vec<f64>>>,
}

/// Math kernel for the SIR procedure. Operates on pre-built parameter and
/// EBE arrays; users should typically call the higher-level
/// [`run_sir`](crate::run_sir) wrapper in `estimation::run_sir`, which takes
/// a `FitResult` and handles `ModelParameters` reconstruction, EBE
/// extraction, and source-file integrity checks.
///
/// # Arguments
/// * `model` - The compiled model
/// * `population` - The dataset
/// * `params` - ML parameter estimates
/// * `eta_hats` - ML EBE estimates (for warm-starting inner loop)
/// * `proposal_cov` - Covariance matrix in packed (log-transformed) parameter space
/// * `ofv_hat` - OFV at ML estimates
/// * `options` - Fit options containing SIR settings
pub fn run_sir_core(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    proposal_cov: &DMatrix<f64>,
    ofv_hat: f64,
    options: &FitOptions,
) -> Result<SirResult, String> {
    let n_samples = options.sir_samples;
    let n_resamples = options.sir_resamples;

    if n_resamples > n_samples {
        return Err("sir_resamples must be <= sir_samples".to_string());
    }

    // Pack ML estimates as the proposal center
    let x_hat = pack_params(params);
    let n_packed = x_hat.len();

    if proposal_cov.nrows() != n_packed || proposal_cov.ncols() != n_packed {
        return Err(format!(
            "Covariance matrix dimensions ({},{}) don't match packed parameters ({})",
            proposal_cov.nrows(),
            proposal_cov.ncols(),
            n_packed,
        ));
    }

    // Restrict the proposal to the free subspace. `compute_covariance` zeroes
    // the rows/cols of FIX-ed parameters, and `compute_bounds` pins their
    // bounds to `lower == upper == x_hat[i]`. Sampling on the full space would
    // (after regularising the singular covariance) perturb fixed indices by
    // ~sqrt(reg) ≈ 1e-4, which then fails the strict bounds check on every
    // sample — yielding "All SIR samples had invalid weights" for any model
    // with at least one FIX-ed parameter. Sampling on the free block instead
    // keeps fixed indices exactly at `x_hat`, and uses `d = n_free` as the
    // Student-t dimensionality so the importance weights are consistent.
    let fixed_mask = packed_fixed_mask(params);
    let free_idx: Vec<usize> = (0..n_packed).filter(|&i| !fixed_mask[i]).collect();
    let n_free = free_idx.len();
    if n_free == 0 {
        return Err("run_sir_core: every packed parameter is FIX — nothing to sample.".to_string());
    }

    // Symmetrize first, then extract the free block (rows/cols of non-FIX
    // indices) before Cholesky.
    let sym_cov_full = (proposal_cov + proposal_cov.transpose()) * 0.5;
    let mut sub_cov = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            sub_cov[(a, b)] = sym_cov_full[(i, j)];
        }
    }

    // Cholesky-decompose the free block; regularise with an eigenvalue floor
    // if the free block is not strictly positive definite.
    let proposal_chol = match sub_cov.clone().cholesky() {
        Some(c) => c.l(),
        None => {
            let eig = sub_cov.clone().symmetric_eigen();
            let min_eig = eig.eigenvalues.min();
            let reg = if min_eig < 1e-8 {
                -min_eig + 1e-8
            } else {
                1e-8
            };
            let reg_cov = &sub_cov + DMatrix::identity(n_free, n_free) * reg;
            if options.verbose {
                eprintln!(
                    "  SIR: free-block proposal covariance not PD (min eigenvalue = {:.2e}), regularizing",
                    min_eig
                );
            }
            reg_cov
                .cholesky()
                .ok_or("Proposal covariance could not be made positive definite")?
                .l()
        }
    };

    // Log-determinant of the free-block proposal covariance (for density
    // computation). Uses n_free, matching the dimensionality of the Student-t.
    let log_det_proposal = 2.0 * (0..n_free).map(|i| proposal_chol[(i, i)].ln()).sum::<f64>();

    let mut rng = match options.sir_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(12345),
    };

    if options.verbose {
        eprintln!(
            "  SIR: drawing {} samples, resampling {}...",
            n_samples, n_resamples
        );
    }

    // Step 1: Pre-generate all samples (RNG is sequential).
    // Use a multivariate Student-t proposal with nu degrees of freedom.
    // Sampling: draw z ~ N(0,I), chi2 ~ chi2(nu), then scale z by sqrt(nu/chi2).
    // Heavier tails than MVN improve ESS for parameters near boundaries (e.g. omega variances).
    let nu = options.sir_df;
    let chi2_dist = ChiSquared::new(nu).map_err(|e| format!("sir_df invalid: {e}"))?;

    let d = n_free as f64;
    // Cache lgamma terms that are constant across all samples.
    let log_norm =
        lgamma((nu + d) / 2.0) - lgamma(nu / 2.0) - (d / 2.0) * (nu * std::f64::consts::PI).ln();
    // At the centre the quadratic form is 0, so log_q_hat = log_norm - 0.5*log_det.
    let log_q_hat = log_norm - 0.5 * log_det_proposal;

    let mut z_vectors: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let z_free: Vec<f64> = (0..n_free).map(|_| rng.sample(StandardNormal)).collect();
        let chi2: f64 = chi2_dist.sample(&mut rng);
        let scale = (nu / chi2).sqrt();
        let z_vec_free = DVector::from_column_slice(&z_free);
        let delta_free = &proposal_chol * &z_vec_free * scale;
        // Build the full packed sample: free indices get x_hat + delta_free,
        // fixed indices stay pinned at x_hat (so the strict bounds check
        // `lower == upper == x_hat[i]` passes).
        let mut x_k = x_hat.clone();
        for (a, &i) in free_idx.iter().enumerate() {
            x_k[i] += delta_free[a];
        }
        samples.push(x_k);
        // store L_free⁻¹(delta_free) = z_free * scale for the quadratic form
        // in log_q_k. Length = n_free.
        z_vectors.push(z_free.into_iter().map(|zi| zi * scale).collect());
    }

    // Step 2: Evaluate importance weights in parallel (warm-started inner loop)
    let inner_maxiter = options.inner_maxiter;
    let inner_tol = options.inner_tol;
    let interaction = options.interaction;
    let bounds = compute_bounds(params);

    let log_weights: Vec<f64> = samples
        .par_iter()
        .zip(z_vectors.par_iter())
        .map(|(x_k, z)| {
            if crate::cancel::is_cancelled(&options.cancel) {
                return f64::NEG_INFINITY;
            }
            // Reject samples outside parameter bounds (avoids wasting inner-loop work)
            let out_of_bounds = x_k
                .iter()
                .zip(bounds.lower.iter().zip(bounds.upper.iter()))
                .any(|(&x, (&lo, &hi))| x < lo || x > hi);
            if out_of_bounds {
                return f64::NEG_INFINITY;
            }

            let params_k = unpack_params(x_k, params);

            // Check for invalid parameters: theta, sigma, and omega
            let theta_invalid = params_k.theta.iter().any(|&t| !t.is_finite() || t <= 0.0);
            let sigma_invalid = params_k
                .sigma
                .values
                .iter()
                .any(|&s| !s.is_finite() || s <= 0.0);
            let n_eta = params_k.omega.dim();
            let omega_invalid = (0..n_eta).any(|i| {
                let var = params_k.omega.matrix[(i, i)];
                let lii = params_k.omega.chol[(i, i)];
                !var.is_finite() || var <= 0.0 || !lii.is_finite() || lii <= 0.0
            });
            if theta_invalid || sigma_invalid || omega_invalid {
                return f64::NEG_INFINITY;
            }

            // Run inner loop warm-started from ML EBEs
            let sir_mu_k = compute_mu_k(model, &params_k.theta, options.mu_referencing);
            let (ehs, hms, _, _kappas) = run_inner_loop_warm(
                model,
                population,
                &params_k,
                inner_maxiter,
                inner_tol,
                Some(eta_hats),
                Some(&sir_mu_k),
                0, // SIR: no EBE convergence tracking
            );

            // Compute OFV
            let nll_k = pop_nll(
                model,
                population,
                &params_k,
                &ehs,
                &hms,
                &_kappas,
                interaction,
            );
            let ofv_k = 2.0 * nll_k;
            if !ofv_k.is_finite() {
                return f64::NEG_INFINITY;
            }

            let dofv = ofv_k - ofv_hat;

            // Log Student-t proposal density at x_k.
            // z holds the scaled standardised residual L^{-1}(x_k - x_hat), so
            // the quadratic form is z^T z (already in the scaled space).
            let quad_form: f64 = z.iter().map(|zi| zi * zi).sum();
            let log_q_k =
                log_norm - 0.5 * log_det_proposal - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();

            // Importance weight: log w_k = -0.5 * dOFV_k - log_q_k + log_q_hat
            -0.5 * dofv - log_q_k + log_q_hat
        })
        .collect();

    // Step 2: Normalize weights using log-sum-exp trick
    let max_log_w = log_weights
        .iter()
        .cloned()
        .filter(|w| w.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    if max_log_w == f64::NEG_INFINITY {
        return Err("All SIR samples had invalid weights".to_string());
    }

    let weights: Vec<f64> = log_weights
        .iter()
        .map(|lw| (lw - max_log_w).exp())
        .collect();
    let sum_w: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

    // Effective sample size
    let sum_w2: f64 = normalized_weights.iter().map(|w| w * w).sum();
    let ess = if sum_w2 > 0.0 { 1.0 / sum_w2 } else { 0.0 };

    if options.verbose {
        eprintln!("  SIR: effective sample size = {:.1}", ess);
    }

    // Step 3: Resample with replacement proportional to weights
    let weighted_dist = WeightedIndex::new(&weights)
        .map_err(|e| format!("Failed to build weighted sampler: {}", e))?;
    let resampled_indices: Vec<usize> = (0..n_resamples)
        .map(|_| weighted_dist.sample(&mut rng))
        .collect();

    // Step 4: Unpack resampled parameter vectors and compute CIs
    let n_theta = params.theta.len();
    let n_eta = params.omega.dim();
    let n_sigma = params.sigma.values.len();

    let mut theta_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_theta];
    let mut omega_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_eta];
    let mut sigma_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_sigma];

    for &idx in &resampled_indices {
        let p = unpack_params(&samples[idx], params);
        for (j, &th) in p.theta.iter().enumerate() {
            theta_samples[j].push(th);
        }
        for j in 0..n_eta {
            omega_samples[j].push(p.omega.matrix[(j, j)]);
        }
        for (j, &s) in p.sigma.values.iter().enumerate() {
            sigma_samples[j].push(s);
        }
    }

    let ci_theta: Vec<(f64, f64)> = theta_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_omega: Vec<(f64, f64)> = omega_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_sigma: Vec<(f64, f64)> = sigma_samples.iter().map(|s| percentile_ci(s)).collect();

    let resamples_packed = if options.sir_keep_samples {
        Some(
            resampled_indices
                .iter()
                .map(|&idx| samples[idx].clone())
                .collect(),
        )
    } else {
        None
    };

    Ok(SirResult {
        ci_theta,
        ci_omega,
        ci_sigma,
        effective_sample_size: ess,
        resamples_packed,
    })
}

/// Log-gamma function via the Lanczos approximation (g=7, n=9 coefficients).
/// Accurate to ~15 significant figures for x > 0.5.
fn lgamma(x: f64) -> f64 {
    // Lanczos coefficients (g=7)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let xm1 = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (xm1 + (i + 1) as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Compute 2.5th and 97.5th percentiles from a sample.
fn percentile_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f64) * 0.025).floor() as usize;
    let hi_idx = ((n as f64) * 0.975).ceil() as usize;
    let lo = sorted[lo_idx.min(n - 1)];
    let hi = sorted[hi_idx.min(n - 1)];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_ci_sorted() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let (lo, hi) = percentile_ci(&values);
        assert!(lo >= 0.02 && lo <= 0.03, "lo={}", lo);
        assert!(hi >= 0.97 && hi <= 0.98, "hi={}", hi);
    }

    #[test]
    fn test_percentile_ci_single() {
        let (lo, hi) = percentile_ci(&[5.0]);
        assert_eq!(lo, 5.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn test_percentile_ci_empty() {
        let (lo, hi) = percentile_ci(&[]);
        assert!(lo.is_nan());
        assert!(hi.is_nan());
    }

    #[test]
    fn test_lgamma_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(0.5) = ln(sqrt(pi))
        assert!((lgamma(1.0)).abs() < 1e-12);
        assert!((lgamma(2.0)).abs() < 1e-12);
        let expected_half = (std::f64::consts::PI.sqrt()).ln();
        assert!(
            (lgamma(0.5) - expected_half).abs() < 1e-10,
            "lgamma(0.5)={}",
            lgamma(0.5)
        );
        // lgamma(5) = ln(4!) = ln(24)
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
    }

    /// Student-t log-density at the centre must equal log_q_hat (quadratic form = 0).
    #[test]
    fn test_student_t_density_at_centre() {
        let nu = 5.0_f64;
        let d = 3.0_f64;
        let log_det = 0.0_f64; // identity covariance
        let log_q_hat = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det;
        // At centre, quad_form = 0, so log_q_k should equal log_q_hat
        let quad_form = 0.0_f64;
        let log_q_k = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det
            - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();
        assert!((log_q_k - log_q_hat).abs() < 1e-12);
    }

    /// Large nu should recover near-normal proposal (lgamma ratio converges).
    #[test]
    fn test_large_nu_approaches_normal() {
        // For nu=1000, d=2, the Student-t log-density should be very close
        // to the MVN log-density at the same quadratic form.
        let nu = 1000.0_f64;
        let d = 2.0_f64;
        let log_det = 0.5_f64;
        let quad_form = 1.5_f64;

        let log_t = lgamma((nu + d) / 2.0)
            - lgamma(nu / 2.0)
            - (d / 2.0) * (nu * std::f64::consts::PI).ln()
            - 0.5 * log_det
            - ((nu + d) / 2.0) * (1.0 + quad_form / nu).ln();

        let log_mvn = -0.5 * (d * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form);

        assert!(
            (log_t - log_mvn).abs() < 0.01,
            "Student-t (nu=1000) vs MVN: diff = {:.4e}",
            (log_t - log_mvn).abs()
        );
    }
}
