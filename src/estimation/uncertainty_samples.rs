//! Parameter-uncertainty sampling for `simulate_with_uncertainty()`.
//!
//! Provides `draw_parameter_samples()`, which produces `Vec<ModelParameters>`
//! draws from the population parameter uncertainty distribution. Two sources
//! are supported:
//!
//! * `UncertaintyMethod::Asymptotic` — multivariate normal in the packed
//!   (log-theta, Cholesky-omega, log-sigma) parameter space, using
//!   `FitResult.covariance_matrix` as the proposal covariance. A
//!   `LogitProbability` θ is drawn on the logit scale instead (#1548, see
//!   `LogitThetaCoord`).
//! * `UncertaintyMethod::Sir` — sample with replacement from
//!   `FitResult.sir_resamples_packed` (requires `sir = true` and
//!   `sir_keep_samples = true` at fit time).
//!
//! Each draw is unpacked via [`unpack_params`] so theta, Omega, and Sigma are
//! perturbed coherently (they share one packed vector).

use crate::estimation::parameterization::{
    pack_with_bounds, theta_packs_log, unpack_params, PackedBounds, PackedStart,
};
use crate::types::{FitResult, ModelParameters, OmegaMatrix, SigmaVector, ThetaTransform};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, RngExt};
use rand_distr::StandardNormal;

/// How parameter-uncertainty draws are produced.
///
/// `Default` is [`UncertaintyMethod::Asymptotic`] — the standard MVN method,
/// which needs only a successful covariance step. `Sir` additionally requires
/// the SIR resample pool to have been retained at fit time, so it is not a
/// safe default. Deriving `Default` here is what lets
/// [`crate::SimulateUncertaintyOptions`] derive it too, so out-of-crate
/// callers (the R wrapper) can construct it with `..Default::default()` and
/// stay source-compatible when a field is added (#529).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UncertaintyMethod {
    /// MVN in packed log-space using `FitResult.covariance_matrix`. Fast and
    /// parametric; requires a successful covariance step.
    #[default]
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
        // Prefer the fit's own `block_sigma` correlations (#847) — they are the
        // estimated values — and fall back to the model declaration for a
        // FitResult written before the field existed.
        residual_correlations: if fit_result.residual_correlations.is_empty() {
            template.residual_correlations.clone()
        } else {
            fit_result.residual_correlations.clone()
        },
        residual_correlation_fixed: if fit_result.residual_correlation_fixed.is_empty() {
            template.residual_correlation_fixed.clone()
        } else {
            fit_result.residual_correlation_fixed.clone()
        },
        omega_iov,
        kappa_fixed: fit_result.kappa_fixed.clone(),
        mixture: None,
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

/// Largest `|logit(θ)|` a logit-scale draw coordinate may take: `logit(1 −
/// 1e-15)`, the clamp `Op::Logit` applies to its argument. Converting a
/// declared bound of `θ ≥ 1` (a `LogitProbability` θ declared with an upper
/// bound above 1, or with none — the parser defaults it to `1e9`) to the logit
/// scale would otherwise give `+inf`.
const LOGIT_DRAW_LIMIT: f64 = 34.538776394910684;

/// A θ coordinate the uncertainty samplers draw on the **logit** scale (#1548).
///
/// A `LogitProbability` θ enters the model as `inv_logit(logit(θ) + η)`, so it
/// lives on (0, 1). Its packed coordinate is `ln θ` (or `θ` itself when the
/// declared lower bound is negative, see `theta_packs_log`), and a normal
/// draw on that scale has no ceiling at 1: a draw above 1 is either rejected
/// by the bounds check (truncating the distribution) or, when the declared
/// upper bound is above 1, accepted and silently clamped to `F = 1` by
/// `Op::Logit`. Drawing `y = logit(θ)` instead keeps every draw inside (0, 1).
///
/// The proposal covariance is carried onto the logit scale by the delta
/// method, `Cov_y = D Cov_x D` with `D_ii = dy/dx` at the estimate — the same
/// construction ferx-r uses for the reported CI of such a θ, so the draws and
/// the interval agree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LogitThetaCoord {
    /// Packed index (θ occupies the first `n_theta` packed slots).
    pub(crate) index: usize,
    /// Whether the packed coordinate is `ln θ` (true) or `θ` (false).
    pub(crate) log_packed: bool,
}

impl LogitThetaCoord {
    fn theta_of(self, x: f64) -> f64 {
        if self.log_packed {
            x.exp()
        } else {
            x
        }
    }

    /// Packed coordinate → logit scale.
    pub(crate) fn x_to_y(self, x: f64) -> f64 {
        let t = self.theta_of(x);
        (t / (1.0 - t)).ln()
    }

    /// Logit scale → packed coordinate. `ln(inv_logit(y))` is written as
    /// `−ln(1 + e^{−y})` so it stays accurate for a large negative `y`.
    ///
    /// Above [`LOGIT_DRAW_LIMIT`] this returns `+inf`, which every packed
    /// upper bound rejects: from `y ≈ 36.7` on, `inv_logit(y)` (or
    /// `exp(ln inv_logit(y))` after unpacking) rounds to exactly `1.0`, so
    /// accepting the draw would bring back the `F = 1` point mass this type
    /// exists to remove.
    pub(crate) fn y_to_x(self, y: f64) -> f64 {
        if y > LOGIT_DRAW_LIMIT {
            f64::INFINITY
        } else if self.log_packed {
            -(-y).exp().ln_1p()
        } else {
            1.0 / (1.0 + (-y).exp())
        }
    }

    /// `dx/dy` at packed coordinate `x`: `1 − θ` for `x = ln θ`, `θ(1 − θ)`
    /// for `x = θ`.
    pub(crate) fn dx_dy(self, x: f64) -> f64 {
        let t = self.theta_of(x);
        if self.log_packed {
            1.0 - t
        } else {
            t * (1.0 - t)
        }
    }
}

/// The θ coordinates to draw on the logit scale: every free `LogitProbability`
/// θ. A FIX'd one carries no uncertainty and stays pinned. An empty
/// `theta_transform` — a `FitResult` written before the field existed —
/// selects nothing, which is the pre-#1548 behaviour.
///
/// A free `LogitProbability` θ whose estimate is not strictly inside (0, 1) is
/// an `Err`: it has no logit, so there is no logit-scale draw to make, and
/// falling back to the packed-scale draw would hand back the draws above 1
/// that `Op::Logit` clamps to a point mass. It is reachable only with a
/// declared upper bound above 1, where the clamped likelihood is flat past 1.
pub(crate) fn logit_theta_coords(
    template: &ModelParameters,
    theta_transform: &[ThetaTransform],
    fixed_mask: &[bool],
) -> Result<Vec<LogitThetaCoord>, String> {
    let mut coords = Vec::new();
    for i in 0..template.theta.len() {
        if theta_transform.get(i) != Some(&ThetaTransform::LogitProbability) || fixed_mask[i] {
            continue;
        }
        let t = template.theta[i];
        if !(t > 0.0 && t < 1.0) {
            let name = template
                .theta_names
                .get(i)
                .map(String::as_str)
                .unwrap_or("?");
            return Err(format!(
                "Cannot draw parameter uncertainty for {name}: it is used as \
                 inv_logit(logit({name}) + ETA), so it must lie strictly inside \
                 (0, 1), but its estimate is {t}. Declare its upper bound below 1 \
                 (e.g. 0.999) and refit."
            ));
        }
        coords.push(LogitThetaCoord {
            index: i,
            log_packed: theta_packs_log(template.theta_lower[i]),
        });
    }
    Ok(coords)
}

/// Move a packed centre and its covariance onto the draw scale: each logit
/// coordinate becomes `logit(θ̂)` and its row and column of `cov` are scaled by
/// `dy/dx` at the estimate (the delta method). Other coordinates are untouched.
pub(crate) fn to_draw_scale(
    x_hat: &[f64],
    cov: &DMatrix<f64>,
    coords: &[LogitThetaCoord],
) -> (Vec<f64>, DMatrix<f64>) {
    let mut y_hat = x_hat.to_vec();
    let mut cov_y = cov.clone();
    for c in coords {
        let i = c.index;
        y_hat[i] = c.x_to_y(x_hat[i]);
        let dy_dx = 1.0 / c.dx_dy(x_hat[i]);
        cov_y.row_mut(i).scale_mut(dy_dx);
        cov_y.column_mut(i).scale_mut(dy_dx);
    }
    (y_hat, cov_y)
}

/// Move packed bounds onto the draw scale. A θ bound at or beyond 0 / 1 maps
/// to `∓LOGIT_DRAW_LIMIT` instead of `∓inf`.
pub(crate) fn bounds_to_draw_scale(
    bounds: &PackedBounds,
    coords: &[LogitThetaCoord],
) -> PackedBounds {
    let mut lower = bounds.lower.clone();
    let mut upper = bounds.upper.clone();
    for c in coords {
        let i = c.index;
        // A bound outside (0, 1) has no logit (`x_to_y` is NaN there, or ±inf
        // exactly at 0 / 1), so it becomes the limit on its own side.
        let lo = c.x_to_y(lower[i]);
        lower[i] = if lo.is_nan() {
            -LOGIT_DRAW_LIMIT
        } else {
            lo.max(-LOGIT_DRAW_LIMIT)
        };
        let hi = c.x_to_y(upper[i]);
        upper[i] = if hi.is_nan() {
            LOGIT_DRAW_LIMIT
        } else {
            hi.min(LOGIT_DRAW_LIMIT)
        };
    }
    PackedBounds { lower, upper }
}

/// Map a draw-scale vector back to the packed scale, in place.
pub(crate) fn from_draw_scale(y: &mut [f64], coords: &[LogitThetaCoord]) {
    for c in coords {
        y[c.index] = c.y_to_x(y[c.index]);
    }
}

/// `ln |dx/dy|` of the draw-scale change of variables at packed vector `x` —
/// the Jacobian SIR adds to its importance weights so that drawing on the
/// logit scale changes the proposal but not the target (#1548).
pub(crate) fn log_abs_jacobian(x: &[f64], coords: &[LogitThetaCoord]) -> f64 {
    coords.iter().map(|c| c.dx_dy(x[c.index]).abs().ln()).sum()
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
    let PackedStart {
        packed: x_hat,
        bounds,
        fixed: fixed_mask,
        // #1307's pack-move list is not this caller's object.
        moves: _,
    } = pack_with_bounds(template);
    let n_packed = x_hat.len();
    if cov.nrows() != n_packed || cov.ncols() != n_packed {
        return Err(format!(
            "Covariance matrix ({}x{}) doesn't match packed parameters ({})",
            cov.nrows(),
            cov.ncols(),
            n_packed
        ));
    }
    // Draw a `LogitProbability` θ on the logit scale (#1548); every other
    // coordinate is drawn on its packed scale as before.
    let logit_coords = logit_theta_coords(template, &fit_result.theta_transform, &fixed_mask)?;
    let (y_hat, cov_y) = to_draw_scale(&x_hat, cov, &logit_coords);
    let chol = regularised_cholesky(&cov_y)?;

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
        let mut x_k: Vec<f64> = y_hat.iter().zip(delta.iter()).map(|(a, b)| a + b).collect();
        from_draw_scale(&mut x_k, &logit_coords);
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
    // slipped through. Precompute the packed start, bounds and fixed mask in
    // one walk, once per call.
    let PackedStart {
        packed: x_hat,
        bounds,
        fixed: fixed_mask,
        // #1307's pack-move list is not this caller's object.
        moves: _,
    } = pack_with_bounds(template);
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

    /// #529: `UncertaintyMethod` derives `Default` so
    /// `SimulateUncertaintyOptions` can derive it too. Pin the variant — a
    /// silent move of `#[default]` to `Sir` would make a spread-constructed
    /// options value demand a SIR pool that most fits never retained.
    #[test]
    fn uncertainty_method_defaults_to_asymptotic() {
        assert_eq!(UncertaintyMethod::default(), UncertaintyMethod::Asymptotic);
    }
    use crate::types::{ErrorModel, OmegaMatrix, SigmaVector};
    use nalgebra::DMatrix;
    // 4000 draws: SE(mean) = 1/sqrt(4000) = 0.016, SE(sd) = 0.011. Measured
    // worst |error| over the two fixtures: 0.013 (mean), 0.016 (sd).
    const MEAN_TOL: f64 = 0.06;
    const SD_TOL: f64 = 0.05;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Build a minimal `ModelParameters` template for unit testing the
    /// sampler. Two thetas, one diagonal Omega (1 eta), one sigma. No IOV.
    fn tiny_template() -> ModelParameters {
        let omega_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![0.04]));
        let omega = OmegaMatrix::from_matrix(omega_matrix, vec!["eta_CL".to_string()], true);
        ModelParameters {
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
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
            mixture: None,
        }
    }

    /// Build a minimal `FitResult` carrying just the fields the asymptotic
    /// sampler reads (theta/omega/sigma + covariance_matrix). Other fields
    /// are filled with sensible defaults.
    fn fit_with_cov(template: &ModelParameters, cov: DMatrix<f64>) -> FitResult {
        FitResult {
            ofv_data: 0.0,
            ofv_prior: 0.0,
            prior_summary: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            se_residual_correlations: None,
            covariate_relations: Vec::new(),
            restored_from_checkpoint: false,
            method: crate::types::EstimationMethod::FoceI,
            method_chain: vec![],
            method_wall_times_secs: vec![],
            covariance_wall_time_secs: 0.0,
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
            residual_correlations: Vec::new(),
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
            vi: None,
            omega_iov: None,
            kappa_names: vec![],
            kappa_fixed: vec![],
            kappa_init_as_sd: vec![],
            kappa_weights: Vec::new(),
            kappa_weight_typical: Vec::new(),
            se_kappa: None,
            shrinkage_kappa: vec![],
            shrinkage_kappa_by_occ: vec![],
            ebe_kappas: vec![],
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            saem_mh_accept_tail: None,
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
            covariance_method: Some(crate::types::CovarianceMethod::Hessian),
            shrinkage_eta: vec![],
            cond_dist: None,
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: String::new(),
            ferx_version: String::new(),
            environment: crate::environment::EnvironmentInfo::default(),
            eta_param_info: vec![],
            theta_transform: vec![],
            sigma_types: vec![],
            cov_eigenvalues: None,
            cov_condition_number: None,
            bic_inputs: Default::default(),
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
            final_gradient_source: None,
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
            packed_estimate: None,
            left_init: None,
            omega_is_diagonal: None,
            kappa_is_diagonal: None,
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

    /// `tiny_template()` plus a third θ, `F`, declared on (`lower`, `upper`),
    /// and a `FitResult` whose covariance gives every coordinate variance
    /// `1e-4` except `F`'s packed coordinate (index 2), which gets `var_f`.
    fn logit_fixture(
        f_hat: f64,
        lower: f64,
        upper: f64,
        var_f: f64,
        transform: ThetaTransform,
    ) -> (ModelParameters, FitResult) {
        let mut template = tiny_template();
        template.theta.push(f_hat);
        template.theta_names.push("F".to_string());
        template.theta_lower.push(lower);
        template.theta_upper.push(upper);
        template.theta_fixed.push(false);
        let n_packed = crate::estimation::parameterization::packed_len(&template);
        let mut cov = DMatrix::identity(n_packed, n_packed) * 1e-4;
        cov[(2, 2)] = var_f;
        let mut fit = fit_with_cov(&template, cov);
        fit.theta_transform = vec![ThetaTransform::Log, ThetaTransform::Log, transform];
        (template, fit)
    }

    fn logit(p: f64) -> f64 {
        (p / (1.0 - p)).ln()
    }

    fn mean_sd(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        (m, var.sqrt())
    }

    fn draw_f(template: &ModelParameters, fit: &FitResult, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        draw_parameter_samples(fit, template, n, UncertaintyMethod::Asymptotic, &mut rng)
            .expect("draws")
            .iter()
            .map(|p| p.theta[2])
            .collect()
    }

    /// #1548: a `LogitProbability` θ packed as `ln θ` is drawn logit-normally,
    /// with the delta-method SD `sd(ln θ) / (1 − θ̂)`. Fixture = the bundled
    /// `bioavailability` declaration `THETA_F(0.70, 0.001, 0.999)`, with
    /// `sd(ln θ) = 0.3` so the logit SD is exactly 1.
    #[test]
    fn asymptotic_logit_probability_draws_are_logit_normal() {
        let (template, fit) =
            logit_fixture(0.7, 0.001, 0.999, 0.09, ThetaTransform::LogitProbability);
        let f = draw_f(&template, &fit, 4000, 11);
        assert!(
            f.iter().all(|&t| t > 0.0 && t < 0.999),
            "draw left (0, 0.999)"
        );
        // The log-normal draw this replaces puts 12% of its mass above 0.999
        // (all rejected) and piles the survivors against the bound: ~0.6% of
        // draws land in (0.99, 0.999), vs `P(Z > (logit 0.99 − logit 0.7)) =
        // 1e-4` for the logit-normal one — about 24 draws against 0.4.
        let near_ceiling = f.iter().filter(|&&t| t > 0.99).count();
        assert!(near_ceiling <= 4, "{near_ceiling}/4000 draws above 0.99");
        let y: Vec<f64> = f.iter().map(|&t| logit(t)).collect();
        let (m, sd) = mean_sd(&y);
        assert!(
            (m - logit(0.7)).abs() < MEAN_TOL,
            "logit mean {m} vs {}",
            logit(0.7)
        );
        assert!(
            (sd - 1.0).abs() < SD_TOL,
            "logit sd {sd} vs delta-method 1.0"
        );
    }

    /// #1548: with the declared upper bound above 1 (here 5), a log-normal draw
    /// above 1 was accepted and clamped to `F = 1` by `Op::Logit`. Drawn on the
    /// logit scale none reaches 1. The same fixture with the θ marked `Log` is
    /// the straddle: it keeps the log-normal draw and must reach past 1, or
    /// this fixture could not see the defect at all.
    #[test]
    fn asymptotic_logit_probability_never_reaches_one_with_a_wide_upper_bound() {
        let (template, fit) =
            logit_fixture(0.7, 0.001, 5.0, 0.09, ThetaTransform::LogitProbability);
        let f = draw_f(&template, &fit, 2000, 5);
        let at_or_above_one = f.iter().filter(|&&t| t >= 1.0).count();
        assert_eq!(at_or_above_one, 0, "logit draws reached F >= 1");

        let (template_log, fit_log) = logit_fixture(0.7, 0.001, 5.0, 0.09, ThetaTransform::Log);
        let f_log = draw_f(&template_log, &fit_log, 2000, 5);
        let log_above_one = f_log.iter().filter(|&&t| t >= 1.0).count();
        assert!(
            log_above_one > 100,
            "straddle: a log-normal draw must pass 1 on this fixture, got {log_above_one}/2000"
        );
    }

    /// #1548: an identity-packed `LogitProbability` θ (declared lower bound
    /// < 0) uses `dy/dx = 1 / (θ̂ (1 − θ̂))`. `sd(θ) = 0.21` at `θ̂ = 0.7` gives
    /// a logit SD of exactly 1.
    #[test]
    fn asymptotic_identity_packed_logit_probability_uses_its_own_jacobian() {
        let (template, fit) = logit_fixture(
            0.7,
            -1.0,
            5.0,
            0.21 * 0.21,
            ThetaTransform::LogitProbability,
        );
        let f = draw_f(&template, &fit, 4000, 17);
        assert!(f.iter().all(|&t| t > 0.0 && t < 1.0), "draw left (0, 1)");
        let y: Vec<f64> = f.iter().map(|&t| logit(t)).collect();
        let (m, sd) = mean_sd(&y);
        assert!((m - logit(0.7)).abs() < MEAN_TOL, "logit mean {m}");
        assert!(
            (sd - 1.0).abs() < SD_TOL,
            "logit sd {sd} vs delta-method 1.0"
        );
    }

    /// `x_to_y` / `y_to_x` invert each other on both packings, `dx_dy` is the
    /// derivative of `y_to_x`, and `log_abs_jacobian` sums its log.
    #[test]
    fn logit_theta_coord_round_trips_and_differentiates() {
        for log_packed in [true, false] {
            let c = LogitThetaCoord {
                index: 0,
                log_packed,
            };
            for t in [1e-6, 0.05, 0.5, 0.7, 0.999] {
                let x = if log_packed { f64::ln(t) } else { t };
                let y = c.x_to_y(x);
                assert!((y - logit(t)).abs() < 1e-9, "x_to_y({x}) = {y}");
                assert!((c.y_to_x(y) - x).abs() < 1e-12 * x.abs().max(1.0));
                let h = 1e-6;
                let fd = (c.y_to_x(y + h) - c.y_to_x(y - h)) / (2.0 * h);
                assert!(
                    (fd - c.dx_dy(x)).abs() < 1e-7,
                    "log_packed={log_packed} t={t}: fd {fd} vs {}",
                    c.dx_dy(x)
                );
                assert_eq!(log_abs_jacobian(&[x], &[c]), c.dx_dy(x).ln());
            }
        }
        assert_eq!(log_abs_jacobian(&[0.3], &[]), 0.0);
    }

    /// A declared bound at or beyond 0 / 1 has no logit and maps to the limit
    /// on its own side; one inside (0, 1) maps to its logit.
    #[test]
    fn bounds_to_draw_scale_maps_declared_bounds() {
        let b = PackedBounds {
            lower: vec![f64::ln(0.001), -1.0, 0.0],
            upper: vec![f64::ln(0.999), 5.0, 1.0],
        };
        let coords = [
            LogitThetaCoord {
                index: 0,
                log_packed: true,
            },
            LogitThetaCoord {
                index: 1,
                log_packed: false,
            },
            LogitThetaCoord {
                index: 2,
                log_packed: false,
            },
        ];
        let y = bounds_to_draw_scale(&b, &coords);
        assert!((y.lower[0] - logit(0.001)).abs() < 1e-9);
        assert!((y.upper[0] - logit(0.999)).abs() < 1e-9);
        assert_eq!(
            (y.lower[1], y.upper[1]),
            (-LOGIT_DRAW_LIMIT, LOGIT_DRAW_LIMIT)
        );
        assert_eq!(
            (y.lower[2], y.upper[2]),
            (-LOGIT_DRAW_LIMIT, LOGIT_DRAW_LIMIT)
        );
        assert!((LOGIT_DRAW_LIMIT - logit(1.0 - 1e-15)).abs() < 1e-3);
    }

    /// `to_draw_scale` scales row and column `i` by `dy/dx` at the estimate —
    /// so the diagonal by its square and an off-diagonal once — and moves only
    /// the logit coordinate's centre.
    #[test]
    fn to_draw_scale_applies_the_delta_method() {
        let x_hat = [f64::ln(0.7), 2.0];
        let cov = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.5]);
        let c = LogitThetaCoord {
            index: 0,
            log_packed: true,
        };
        let (y_hat, cov_y) = to_draw_scale(&x_hat, &cov, &[c]);
        let d = 1.0 / 0.3;
        assert!((y_hat[0] - logit(0.7)).abs() < 1e-12);
        assert_eq!(y_hat[1], 2.0);
        assert!((cov_y[(0, 0)] - 0.09 * d * d).abs() < 1e-12);
        assert!((cov_y[(0, 1)] - 0.02 * d).abs() < 1e-12);
        assert!((cov_y[(1, 0)] - 0.02 * d).abs() < 1e-12);
        assert_eq!(cov_y[(1, 1)], 0.5);
    }

    /// Every free `LogitProbability` θ is selected; a FIX'd one and an empty
    /// transform list (an old `FitResult`) select nothing; the packing follows
    /// the declared lower bound.
    #[test]
    fn logit_theta_coords_selects_free_logit_probability_thetas() {
        let (mut template, _) =
            logit_fixture(0.7, 0.001, 0.999, 0.09, ThetaTransform::LogitProbability);
        let tt = [
            ThetaTransform::Log,
            ThetaTransform::Logit,
            ThetaTransform::LogitProbability,
        ];
        let free = [false; 5];
        assert_eq!(
            logit_theta_coords(&template, &tt, &free).unwrap(),
            vec![LogitThetaCoord {
                index: 2,
                log_packed: true
            }]
        );
        assert!(logit_theta_coords(&template, &[], &free)
            .unwrap()
            .is_empty());
        let mut fixed = free;
        fixed[2] = true;
        assert!(logit_theta_coords(&template, &tt, &fixed)
            .unwrap()
            .is_empty());
        template.theta_lower[2] = -1.0;
        assert_eq!(
            logit_theta_coords(&template, &tt, &free).unwrap(),
            vec![LogitThetaCoord {
                index: 2,
                log_packed: false
            }]
        );
    }

    /// A free `LogitProbability` estimate at or outside (0, 1) — reachable with
    /// an upper bound above 1 — is an `Err` naming the θ, not a silent fall
    /// back to the packed-scale draw that reaches past 1. FIX'd, the same
    /// estimate is fine: it is pinned, never drawn.
    #[test]
    fn logit_theta_coords_rejects_a_free_estimate_outside_the_unit_interval() {
        let tt = [
            ThetaTransform::Log,
            ThetaTransform::Log,
            ThetaTransform::LogitProbability,
        ];
        for (f_hat, lower) in [(1.0, 0.001), (1.2, 0.001), (0.0, -1.0), (-0.3, -1.0)] {
            let (template, _) =
                logit_fixture(f_hat, lower, 5.0, 0.09, ThetaTransform::LogitProbability);
            let err = logit_theta_coords(&template, &tt, &[false; 5]).unwrap_err();
            assert!(
                err.contains("uncertainty for F") && err.contains(&format!("is {f_hat}")),
                "{err}"
            );
            let mut fixed = [false; 5];
            fixed[2] = true;
            assert!(logit_theta_coords(&template, &tt, &fixed)
                .unwrap()
                .is_empty());
        }
    }

    /// The `Err` reaches `draw_parameter_samples`'s caller.
    #[test]
    fn asymptotic_errors_on_a_logit_probability_estimate_above_one() {
        let (template, fit) =
            logit_fixture(1.2, 0.001, 5.0, 0.09, ThetaTransform::LogitProbability);
        let mut rng = StdRng::seed_from_u64(0);
        let err =
            draw_parameter_samples(&fit, &template, 10, UncertaintyMethod::Asymptotic, &mut rng)
                .unwrap_err();
        assert!(err.contains("uncertainty for F"), "{err}");
    }

    /// From `y ≈ 36.7` on, `inv_logit(y)` rounds to exactly 1.0 — through
    /// `unpack_params`'s `exp` for a log-packed θ too. `y_to_x` returns `+inf`
    /// above `LOGIT_DRAW_LIMIT` so the bounds check rejects the draw. The
    /// fixture draws with `sd(logit θ) = 30`, so ~13% of proposals land above
    /// the limit and ~12% above 36.7: without the cut-off those return F = 1.
    #[test]
    fn asymptotic_logit_draws_never_saturate_to_one() {
        let c = LogitThetaCoord {
            index: 0,
            log_packed: false,
        };
        assert_eq!(c.y_to_x(40.0), f64::INFINITY);
        assert!(c.y_to_x(LOGIT_DRAW_LIMIT) < 1.0);
        let c_log = LogitThetaCoord {
            index: 0,
            log_packed: true,
        };
        assert_eq!(c_log.y_to_x(40.0), f64::INFINITY);
        assert!(c_log.y_to_x(LOGIT_DRAW_LIMIT).exp() < 1.0);

        // sd(x) = 30 · dx/dy at 0.7: θ(1 − θ) = 0.21 identity-packed, 1 − θ = 0.3 log-packed.
        for (lower, sd_x) in [(-1.0, 30.0 * 0.21), (0.001, 30.0 * 0.3)] {
            let (template, fit) = logit_fixture(
                0.7,
                lower,
                5.0,
                sd_x * sd_x,
                ThetaTransform::LogitProbability,
            );
            let f = draw_f(&template, &fit, 2000, 3);
            let at_one = f.iter().filter(|&&t| t >= 1.0).count();
            assert_eq!(at_one, 0, "lower {lower}: {at_one}/2000 draws at F = 1");
        }
    }

    /// A covariance with the right row count but the wrong column count is an
    /// `Err`, not an out-of-range panic in the draw-scale column scaling.
    #[test]
    fn asymptotic_rejects_a_non_square_covariance() {
        let (template, mut fit) =
            logit_fixture(0.7, 0.001, 0.999, 0.09, ThetaTransform::LogitProbability);
        let n = crate::estimation::parameterization::packed_len(&template);
        fit.covariance_matrix = Some(DMatrix::identity(n, 2) * 0.01);
        let mut rng = StdRng::seed_from_u64(0);
        let err =
            draw_parameter_samples(&fit, &template, 10, UncertaintyMethod::Asymptotic, &mut rng)
                .unwrap_err();
        assert!(err.contains("doesn't match packed parameters"), "{err}");
    }
}
