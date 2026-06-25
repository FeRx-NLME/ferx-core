//! Parameter-uncertainty sampling for `simulate_with_uncertainty()`.
//!
//! Provides `draw_parameter_samples()`, which produces `Vec<ModelParameters>`
//! draws from the population parameter uncertainty distribution. Two sources
//! are supported:
//!
//! * `UncertaintyMethod::Asymptotic` — multivariate normal in the packed
//!   (log-theta, Cholesky-omega, log-sigma) parameter space, using
//!   `FitResult.covariance_matrix` as the proposal covariance.
//! * `UncertaintyMethod::Sir` — sample with replacement from
//!   `FitResult.sir_resamples_packed` (requires `sir = true` and
//!   `sir_keep_samples = true` at fit time).
//!
//! Each draw is unpacked via [`unpack_params`] so theta, Omega, and Sigma are
//! perturbed coherently (they share one packed vector).

use crate::estimation::parameterization::{
    compute_bounds, packed_fixed_mask, unpack_params, PackedBounds,
};
use crate::types::{FitResult, ModelParameters, OmegaMatrix, SigmaVector};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

/// How parameter-uncertainty draws are produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyMethod {
    /// MVN in packed log-space using `FitResult.covariance_matrix`. Fast and
    /// parametric; requires a successful covariance step.
    Asymptotic,
    /// Reuse the resampled parameter vectors retained from the SIR step.
    /// Requires `FitOptions.sir = true` AND `sir_keep_samples = true`.
    Sir,
    // Bootstrap — reserved for a future implementation (subject resampling +
    // refit). Adding it is a separate, larger feature.
}

/// Reconstruct the fitted `ModelParameters` from a `FitResult` using the
/// `CompiledModel`'s `default_params` for structure (bounds, FIX flags,
/// diagonal flag, IOV structure).
///
/// `FitResult` stores the fitted theta/Omega/Sigma as plain fields but not as
/// a `ModelParameters` value; callers of `simulate_with_uncertainty()` and
/// `draw_parameter_samples()` need a full template, so this helper builds one.
pub fn fitted_params_from_result(
    fit_result: &FitResult,
    model: &crate::types::CompiledModel,
) -> ModelParameters {
    let template = &model.default_params;
    let omega_diagonal = template.omega.diagonal;
    let omega = OmegaMatrix::from_matrix_with_mask(
        fit_result.omega.clone(),
        fit_result.eta_names.clone(),
        omega_diagonal,
        template.omega.free_mask.clone(),
    );
    let omega_iov = template.omega_iov.as_ref().map(|iov_template| {
        let m = fit_result
            .omega_iov
            .clone()
            .unwrap_or_else(|| iov_template.matrix.clone());
        OmegaMatrix::from_matrix_with_mask(
            m,
            iov_template.eta_names.clone(),
            iov_template.diagonal,
            iov_template.free_mask.clone(),
        )
    });
    ModelParameters {
        theta: fit_result.theta.clone(),
        theta_names: fit_result.theta_names.clone(),
        theta_lower: template.theta_lower.clone(),
        theta_upper: template.theta_upper.clone(),
        theta_fixed: fit_result.theta_fixed.clone(),
        omega,
        omega_fixed: fit_result.omega_fixed.clone(),
        sigma: SigmaVector {
            values: fit_result.sigma.clone(),
            names: fit_result.sigma_names.clone(),
        },
        sigma_fixed: fit_result.sigma_fixed.clone(),
        omega_iov,
        kappa_fixed: fit_result.kappa_fixed.clone(),
    }
}

/// Symmetrise + Cholesky-decompose a covariance matrix, regularising the
/// eigenvalue floor when the matrix is not strictly positive definite.
/// Returns the lower-triangular Cholesky factor `L` such that `L * L^T ≈ cov`.
pub(crate) fn regularised_cholesky(cov: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = cov.nrows();
    if cov.ncols() != n {
        return Err(format!(
            "Covariance matrix must be square, got ({}, {})",
            cov.nrows(),
            cov.ncols()
        ));
    }
    let sym = (cov + cov.transpose()) * 0.5;
    if let Some(c) = sym.clone().cholesky() {
        return Ok(c.l());
    }
    let eig = sym.clone().symmetric_eigen();
    let min_eig = eig.eigenvalues.min();
    let reg = if min_eig < 1e-8 {
        -min_eig + 1e-8
    } else {
        1e-8
    };
    let reg_cov = &sym + DMatrix::identity(n, n) * reg;
    reg_cov
        .cholesky()
        .map(|c| c.l())
        .ok_or_else(|| "Covariance could not be made positive definite".to_string())
}

/// Clamp packed indices flagged as FIX to their pinned packed value (taken
/// from `x_hat`). `compute_bounds()` already pins fixed indices via
/// `lower == upper`, so a continuous MVN draw would otherwise fail the bounds
/// check for any model with a FIX'd theta / omega / sigma / kappa. This is
/// equivalent to sampling only the free subspace: fixed parameters carry no
/// uncertainty and must equal their pinned value.
fn clamp_fixed_indices(x: &mut [f64], fixed_mask: &[bool], x_hat: &[f64]) {
    for (i, &is_fixed) in fixed_mask.iter().enumerate() {
        if is_fixed {
            x[i] = x_hat[i];
        }
    }
}

/// Check that a candidate packed parameter vector unpacks to a parameter set
/// that is in-bounds and has positive theta / sigma / omega-diagonal values.
/// `bounds` is taken by reference to avoid recomputing it for each candidate
/// (it doesn't change across draws within a single sampler call).
fn candidate_is_valid(x: &[f64], template: &ModelParameters, bounds: &PackedBounds) -> bool {
    for (i, &xi) in x.iter().enumerate() {
        if xi < bounds.lower[i] || xi > bounds.upper[i] {
            return false;
        }
    }
    let p = unpack_params(x, template);
    if p.theta.iter().any(|&t| !t.is_finite() || t <= 0.0) {
        return false;
    }
    if p.sigma.values.iter().any(|&s| !s.is_finite() || s <= 0.0) {
        return false;
    }
    let n_eta = p.omega.dim();
    for i in 0..n_eta {
        let var = p.omega.matrix[(i, i)];
        let lii = p.omega.chol[(i, i)];
        if !var.is_finite() || var <= 0.0 || !lii.is_finite() || lii <= 0.0 {
            return false;
        }
    }
    true
}

/// Draw `n_draws` parameter samples from the uncertainty distribution.
///
/// Each draw is a fully unpacked `ModelParameters` (theta, Omega, Sigma —
/// and IOV omega when present) suitable for handing to `simulate_inner()`.
///
/// Out-of-bounds or invalid draws are rejected and resampled, up to
/// `10 * n_draws` total attempts.
pub fn draw_parameter_samples(
    fit_result: &FitResult,
    template: &ModelParameters,
    n_draws: usize,
    method: UncertaintyMethod,
    rng: &mut impl Rng,
) -> Result<Vec<ModelParameters>, String> {
    if n_draws == 0 {
        return Ok(Vec::new());
    }
    match method {
        UncertaintyMethod::Asymptotic => draw_asymptotic(fit_result, template, n_draws, rng),
        UncertaintyMethod::Sir => draw_sir(fit_result, template, n_draws, rng),
    }
}

fn draw_asymptotic(
    fit_result: &FitResult,
    template: &ModelParameters,
    n_draws: usize,
    rng: &mut impl Rng,
) -> Result<Vec<ModelParameters>, String> {
    let cov = fit_result.covariance_matrix.as_ref().ok_or_else(|| {
        "Asymptotic uncertainty requires FitResult.covariance_matrix; run the \
         fit with `covariance = true` and ensure the covariance step succeeds."
            .to_string()
    })?;
    let x_hat = crate::estimation::parameterization::pack_params(template);
    let n_packed = x_hat.len();
    if cov.nrows() != n_packed {
        return Err(format!(
            "Covariance matrix ({}x{}) doesn't match packed parameters ({})",
            cov.nrows(),
            cov.ncols(),
            n_packed
        ));
    }
    let chol = regularised_cholesky(cov)?;
    let bounds = compute_bounds(template);
    let fixed_mask = packed_fixed_mask(template);

    let max_tries = 10 * n_draws;
    let mut draws = Vec::with_capacity(n_draws);
    let mut tries = 0usize;
    while draws.len() < n_draws {
        if tries >= max_tries {
            return Err(format!(
                "Asymptotic sampler: only {}/{} valid draws after {} attempts \
                 (covariance may be ill-conditioned or near a bound)",
                draws.len(),
                n_draws,
                tries
            ));
        }
        tries += 1;
        let z: Vec<f64> = (0..n_packed).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let delta = &chol * z_vec;
        let mut x_k: Vec<f64> = x_hat.iter().zip(delta.iter()).map(|(a, b)| a + b).collect();
        // Fixed parameters carry no uncertainty — pin them to x_hat before
        // bounds-checking. Without this, `compute_bounds` (which sets
        // lower == upper for fixed indices) would reject every draw for any
        // model with a FIX'd theta / omega / sigma / kappa.
        clamp_fixed_indices(&mut x_k, &fixed_mask, &x_hat);
        if !candidate_is_valid(&x_k, template, &bounds) {
            continue;
        }
        draws.push(unpack_params(&x_k, template));
    }
    Ok(draws)
}

fn draw_sir(
    fit_result: &FitResult,
    template: &ModelParameters,
    n_draws: usize,
    rng: &mut impl Rng,
) -> Result<Vec<ModelParameters>, String> {
    let pool = fit_result.sir_resamples_packed.as_ref().ok_or_else(|| {
        "SIR uncertainty requires FitResult.sir_resamples_packed; run the fit \
         with `sir = true` and `sir_keep_samples = true`."
            .to_string()
    })?;
    if pool.is_empty() {
        return Err("SIR resample pool is empty".to_string());
    }
    let expected_len = crate::estimation::parameterization::packed_len(template);
    if pool[0].len() != expected_len {
        return Err(format!(
            "SIR resample length ({}) doesn't match packed parameters ({}) — \
             the FitResult and template may come from different models",
            pool[0].len(),
            expected_len
        ));
    }

    // Bounds-rejection sampling. SIR already filtered for finite weights, but
    // we still validate to be defensive against extreme proposal samples that
    // slipped through. Precompute bounds + fixed mask once per call.
    let bounds = compute_bounds(template);
    let fixed_mask = packed_fixed_mask(template);
    let x_hat = crate::estimation::parameterization::pack_params(template);
    let max_tries = 10 * n_draws;
    let mut draws = Vec::with_capacity(n_draws);
    let mut tries = 0usize;
    while draws.len() < n_draws {
        if tries >= max_tries {
            return Err(format!(
                "SIR sampler: only {}/{} valid draws after {} attempts",
                draws.len(),
                n_draws,
                tries
            ));
        }
        tries += 1;
        let idx = rng.random_range(0..pool.len());
        let mut x_k = pool[idx].clone();
        // Re-pin fixed indices defensively: SIR samples should already
        // respect the pin (SIR's own bounds use `compute_bounds`), but
        // clamping is cheap insurance against drift / off-by-epsilon issues.
        clamp_fixed_indices(&mut x_k, &fixed_mask, &x_hat);
        if !candidate_is_valid(&x_k, template, &bounds) {
            continue;
        }
        draws.push(unpack_params(&x_k, template));
    }
    Ok(draws)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ErrorModel, OmegaMatrix, SigmaVector};
    use nalgebra::DMatrix;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Build a minimal `ModelParameters` template for unit testing the
    /// sampler. Two thetas, one diagonal Omega (1 eta), one sigma. No IOV.
    fn tiny_template() -> ModelParameters {
        let omega_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![0.04]));
        let omega = OmegaMatrix::from_matrix(omega_matrix, vec!["eta_CL".to_string()], true);
        ModelParameters {
            theta: vec![1.0, 5.0],
            theta_names: vec!["CL".to_string(), "V".to_string()],
            theta_lower: vec![1e-3, 1e-3],
            theta_upper: vec![1e6, 1e6],
            theta_fixed: vec![false, false],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["prop_err".to_string()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        }
    }

    /// Build a minimal `FitResult` carrying just the fields the asymptotic
    /// sampler reads (theta/omega/sigma + covariance_matrix). Other fields
    /// are filled with sensible defaults.
    fn fit_with_cov(template: &ModelParameters, cov: DMatrix<f64>) -> FitResult {
        FitResult {
            method: crate::types::EstimationMethod::FoceI,
            method_chain: vec![],
            converged: true,
            ofv: 0.0,
            aic: 0.0,
            bic: 0.0,
            theta: template.theta.clone(),
            theta_names: template.theta_names.clone(),
            eta_names: template.omega.eta_names.clone(),
            omega: template.omega.matrix.clone(),
            sigma: template.sigma.values.clone(),
            sigma_names: template.sigma.names.clone(),
            error_model: ErrorModel::Proportional,
            covariance_matrix: Some(cov),
            se_theta: None,
            se_omega: None,
            se_sigma: None,
            theta_fixed: template.theta_fixed.clone(),
            omega_fixed: template.omega_fixed.clone(),
            sigma_fixed: template.sigma_fixed.clone(),
            omega_init_as_sd: vec![false; template.omega.matrix.nrows()],
            sigma_init_as_sd: vec![false; template.sigma.values.len()],
            subjects: vec![],
            n_obs: 0,
            n_subjects: 0,
            n_parameters: 0,
            n_iterations: 0,
            interaction: true,
            warnings: vec![],
            warnings_structured: vec![],
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            impmap_trace: None,
            bayes: None,
            omega_iov: None,
            kappa_names: vec![],
            kappa_fixed: vec![],
            kappa_init_as_sd: vec![],
            se_kappa: None,
            shrinkage_kappa: vec![],
            shrinkage_kappa_by_occ: vec![],
            ebe_kappas: vec![],
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: String::new(),
            gradient_method_outer: String::new(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 1,
            nlopt_missing_algorithms: vec![],
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: crate::types::CovarianceStatus::Computed,
            shrinkage_eta: vec![],
            cond_dist: None,
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: String::new(),
            ferx_version: String::new(),
            eta_param_info: vec![],
            theta_transform: vec![],
            sigma_types: vec![],
            cov_eigenvalues: None,
            cov_condition_number: None,
            eta_log_transformed: vec![],
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            model_text: None,
            theta_init: template.theta.clone(),
            omega_init: template.omega.matrix.clone(),
            sigma_init: template.sigma.values.clone(),
            obs_time_range: None,
            final_gradient: None,
            optimizer: "bobyqa".to_string(),
            n_starts: 1,
            multi_start_seed: None,
            saem_seed: None,
            sir_seed: None,
            imp_seed: None,
            npde_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 0,
            outer_gtol: 0.0,
            inits_from_nca: None,
            covariate_names: Vec::new(),
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
            exclusions: None,
        }
    }

    #[test]
    fn asymptotic_mean_recovers_xhat() {
        let template = tiny_template();
        // Tiny diagonal covariance in packed space (4 packed params:
        // log(theta1), log(theta2), log(L_omega), log(sigma)).
        let n_packed = crate::estimation::parameterization::packed_len(&template);
        let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
        let fit = fit_with_cov(&template, cov);

        let mut rng = StdRng::seed_from_u64(42);
        let draws = draw_parameter_samples(
            &fit,
            &template,
            2000,
            UncertaintyMethod::Asymptotic,
            &mut rng,
        )
        .unwrap();
        assert_eq!(draws.len(), 2000);

        // Empirical theta means should be close to template theta.
        let mean_th1: f64 = draws.iter().map(|p| p.theta[0]).sum::<f64>() / draws.len() as f64;
        let mean_th2: f64 = draws.iter().map(|p| p.theta[1]).sum::<f64>() / draws.len() as f64;
        assert!((mean_th1 - 1.0).abs() < 0.05, "mean_th1 = {}", mean_th1);
        assert!((mean_th2 - 5.0).abs() < 0.25, "mean_th2 = {}", mean_th2);
    }

    #[test]
    fn asymptotic_errors_without_covariance() {
        let template = tiny_template();
        let mut fit = fit_with_cov(
            &template,
            DMatrix::identity(
                crate::estimation::parameterization::packed_len(&template),
                crate::estimation::parameterization::packed_len(&template),
            ) * 0.01,
        );
        fit.covariance_matrix = None;
        let mut rng = StdRng::seed_from_u64(0);
        let err =
            draw_parameter_samples(&fit, &template, 10, UncertaintyMethod::Asymptotic, &mut rng)
                .unwrap_err();
        assert!(err.contains("covariance"));
    }

    #[test]
    fn sir_errors_without_resamples() {
        let template = tiny_template();
        let fit = fit_with_cov(
            &template,
            DMatrix::identity(
                crate::estimation::parameterization::packed_len(&template),
                crate::estimation::parameterization::packed_len(&template),
            ) * 0.01,
        );
        let mut rng = StdRng::seed_from_u64(0);
        let err = draw_parameter_samples(&fit, &template, 10, UncertaintyMethod::Sir, &mut rng)
            .unwrap_err();
        assert!(err.contains("sir_keep_samples"));
    }

    #[test]
    fn sir_draws_from_pool() {
        let template = tiny_template();
        let n_packed = crate::estimation::parameterization::packed_len(&template);
        let mut fit = fit_with_cov(&template, DMatrix::identity(n_packed, n_packed) * 0.01);
        // Build a pool of 5 deterministic resamples around the ML estimate.
        let x_hat = crate::estimation::parameterization::pack_params(&template);
        let pool: Vec<Vec<f64>> = (0..5)
            .map(|k| {
                let mut xk = x_hat.clone();
                xk[0] += 0.01 * k as f64;
                xk
            })
            .collect();
        fit.sir_resamples_packed = Some(pool);

        let mut rng = StdRng::seed_from_u64(123);
        let draws =
            draw_parameter_samples(&fit, &template, 50, UncertaintyMethod::Sir, &mut rng).unwrap();
        assert_eq!(draws.len(), 50);
        // All thetas must come from the small perturbed pool.
        for d in &draws {
            assert!(d.theta[0] >= 1.0 && d.theta[0] <= 1.0 * (0.04_f64).exp());
        }
    }

    #[test]
    fn bounds_rejection_respects_upper() {
        let mut template = tiny_template();
        // Pin theta_1's upper bound just above its current value so almost
        // any positive perturbation is rejected.
        template.theta_upper[0] = 1.05;
        let n_packed = crate::estimation::parameterization::packed_len(&template);
        // Diagonal covariance with small variance keeps draws near x_hat.
        let cov = DMatrix::identity(n_packed, n_packed) * 0.001;
        let fit = fit_with_cov(&template, cov);
        let mut rng = StdRng::seed_from_u64(99);
        let draws =
            draw_parameter_samples(&fit, &template, 50, UncertaintyMethod::Asymptotic, &mut rng)
                .unwrap();
        for d in &draws {
            assert!(
                d.theta[0] <= 1.05 + 1e-12,
                "theta_1 = {} exceeded upper",
                d.theta[0]
            );
        }
    }

    #[test]
    fn regularised_cholesky_handles_non_pd() {
        // Build a symmetric but indefinite matrix.
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let l = regularised_cholesky(&m).unwrap();
        // L * L^T should approximately match the regularised version.
        let prod = &l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(prod[(i, j)].is_finite());
            }
        }
    }

    /// Regression test for the Copilot review on PR #7: when a parameter is
    /// FIX'd, `compute_bounds` sets `lower == upper` for its packed index, so
    /// a continuous MVN draw used to be rejected almost surely. We now
    /// re-pin fixed indices to `x_hat` before bounds-checking, so the
    /// sampler should succeed and the returned theta/sigma/omega should
    /// equal the template's value for any FIX'd parameter.
    #[test]
    fn asymptotic_fixed_parameters_pinned_to_template() {
        let mut template = tiny_template();
        template.theta_fixed = vec![true, false]; // Pin TVCL
        template.sigma_fixed = vec![true]; // Pin sigma too
        let n_packed = crate::estimation::parameterization::packed_len(&template);
        // Use a covariance with non-zero entries at the fixed indices to
        // prove the clamp does the work (real fits set those rows/cols to
        // zero, but the sampler shouldn't depend on that).
        let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
        let fit = fit_with_cov(&template, cov);
        let mut rng = StdRng::seed_from_u64(7);
        let draws =
            draw_parameter_samples(&fit, &template, 50, UncertaintyMethod::Asymptotic, &mut rng)
                .expect("FIX'd parameters should not exhaust max_tries");
        assert_eq!(draws.len(), 50);
        // Fixed indices are pinned on the *packed* scale (log-space) so the
        // natural-scale round-trip through `unpack_params` is ULP-accurate
        // rather than bit-exact.
        for d in &draws {
            assert!(
                (d.theta[0] - 1.0).abs() < 1e-12,
                "fixed TVCL drifted: {}",
                d.theta[0]
            );
            assert!(
                (d.sigma.values[0] - 0.1).abs() < 1e-12,
                "fixed sigma drifted: {}",
                d.sigma.values[0]
            );
        }
        let any_v_varies = draws.iter().any(|d| (d.theta[1] - 5.0).abs() > 1e-6);
        assert!(
            any_v_varies,
            "free parameter V was inadvertently pinned by the clamp"
        );
    }
}
