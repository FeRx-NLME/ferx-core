/// Distribution over per-subject random effects (η) for SAEM.
///
/// The two implementations are:
/// - [`GaussianOmega`]: multivariate normal — wraps the existing [`OmegaMatrix`]
///   path and is used for parity testing of the copula arm's plumbing.
/// - `VineCopulaOmega` (future phase): flexible marginals + pair-wise tail
///   dependence via a vine copula.
///
/// Every method operates on η values in their natural (untransformed) scale,
/// the same scale passed throughout `run_saem`.
use crate::types::{ModelParameters, OmegaMatrix};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;

/// The minimum value applied to free diagonal entries of Ω during the M-step.
///
/// Mirrors `SAEM_OMEGA_DIAG_FLOOR` in `estimation/saem.rs`; must stay in sync.
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Distribution over per-subject random effects η.
///
/// Provides the two load-bearing pieces of the SAEM copula arm:
///
/// - **E-step**: `log_prior(η)` replaces the Gaussian quadratic form in the
///   MH acceptance ratio (`obs_nll_subject_into + dist.log_prior(η)`).
/// - **M-step**: `mstep_update(etas, gamma)` replaces the `(1/N) Σ ηηᵀ`
///   sample-covariance update and subsequent SA averaging.
pub trait RandomEffectDistribution: Send + Sync {
    /// Negative log-prior density: `−log p(η)`.
    ///
    /// For the Gaussian path this equals `0.5 (η′ Ω⁻¹ η + log|Ω|)`,
    /// which is exactly the term that `individual_nll_into` adds on top of
    /// `obs_nll_subject_into`.
    fn log_prior(&self, eta: &[f64]) -> f64;

    /// Update the distribution from the pooled E-step η samples (M-step).
    ///
    /// `sampled_etas[i]` is the accepted η for subject i from this iteration.
    /// `gamma` is the SAEM step-size (1 during exploration, 1/(k−K1) during
    /// convergence) applied to the SA sufficient-statistic weighting.
    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64);

    /// Draw `n` joint η samples — used for simulation and reporting.
    ///
    /// `where Self: Sized` excludes this method from dyn dispatch, keeping
    /// the rest of the trait dyn-compatible while preserving the generic `Rng`
    /// argument for concrete callers.
    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>>
    where
        Self: Sized;

    /// Working-proposal Cholesky used to scale MH perturbations.
    ///
    /// For `GaussianOmega` this is `omega.chol` (the Cholesky of the current
    /// Ω); for the vine path it is updated from the sample variance of the
    /// pooled etas each iteration.
    fn proposal_chol(&self) -> &DMatrix<f64>;

    /// Gaussian-equivalent `OmegaMatrix` derived from the current distribution
    /// state.
    ///
    /// For `GaussianOmega` this is just the wrapped `OmegaMatrix`. For the vine
    /// path it is the sample-variance covariance matrix (the "Gaussian-equivalent
    /// OMEGA" kept for reporting and output).
    fn to_omega_matrix(&self) -> &OmegaMatrix;
}

/// Multivariate-normal random-effect distribution wrapping the existing
/// [`OmegaMatrix`] path.
///
/// Used in the copula arm to prove that the plumbing is correct: when
/// `GaussianOmega` drives the copula arm, the results must be numerically
/// identical to those from the frozen Gaussian arm on all existing SAEM cases.
/// It is **not** used in the production Gaussian path of `run_saem`.
pub struct GaussianOmega {
    /// Current Ω estimate (updated each M-step after the burn-in).
    pub omega: OmegaMatrix,
    /// SA sufficient statistic s₂ = (1−γ) s₂_prev + γ (1/N) Σ ηᵢηᵢᵀ.
    pub s2: DMatrix<f64>,
    /// Structural free-parameter mask from the initial model declaration.
    /// Off-diagonal entries that are structurally absent (different `block_omega`
    /// groups) remain `false`; they are zeroed before each M-step update to
    /// prevent sampling correlations from bleeding across blocks.
    free_mask: DMatrix<bool>,
    /// Per-eta fixed flags (`true` = FIX). Fixed entries are restored from
    /// `initial_matrix` after each SA update so they never move.
    omega_fixed: Vec<bool>,
    /// Ω from the initial model parameters, used to restore fixed entries.
    initial_matrix: DMatrix<f64>,
}

impl GaussianOmega {
    /// Construct from the initial model parameters.
    ///
    /// `s2` is seeded to `init_params.omega.matrix` so the first M-step
    /// after the burn-in starts from the same sufficient statistic as the
    /// Gaussian arm.
    pub fn from_init_params(init_params: &ModelParameters) -> Self {
        let omega = init_params.omega.clone();
        let s2 = omega.matrix.clone();
        let free_mask = omega.free_mask.clone();
        let omega_fixed = init_params.omega_fixed.clone();
        let initial_matrix = omega.matrix.clone();
        Self {
            omega,
            s2,
            free_mask,
            omega_fixed,
            initial_matrix,
        }
    }
}

impl RandomEffectDistribution for GaussianOmega {
    fn log_prior(&self, eta: &[f64]) -> f64 {
        if !self.omega.log_det.is_finite() {
            return 1e20;
        }
        let eta_vec = DVector::from_column_slice(eta);
        let eta_prior = eta_vec.dot(&(&self.omega.inv * &eta_vec));
        0.5 * (eta_prior + self.omega.log_det)
    }

    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() {
            return;
        }
        let dim = self.omega.dim();
        let n = sampled_etas.len();

        // Compute (1/N) Σ ηᵢηᵢᵀ — same as the Gaussian arm.
        let mut eta_outer = DMatrix::zeros(dim, dim);
        for eta in sampled_etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n as f64;

        // SA sufficient-statistic update.
        self.s2 = (1.0 - gamma) * &self.s2 + gamma * &eta_outer;

        // Build new Ω from s2, applying the same structural constraints as
        // the Gaussian M-step in run_saem.
        let mut new_omega = self.s2.clone();

        // Zero structurally-absent off-diagonals (entries not in free_mask).
        for i in 0..dim {
            for j in 0..dim {
                if !self.free_mask[(i, j)] {
                    new_omega[(i, j)] = 0.0;
                }
            }
        }
        // Restore fixed entries from the initial matrix.
        for i in 0..dim {
            for j in 0..dim {
                let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = self.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    new_omega[(i, j)] = self.initial_matrix[(i, j)];
                }
            }
        }
        // Floor free diagonal entries to keep Ω positive-definite.
        for i in 0..dim {
            let fixed = self.omega_fixed.get(i).copied().unwrap_or(false);
            if !fixed && new_omega[(i, i)] < OMEGA_DIAG_FLOOR {
                new_omega[(i, i)] = OMEGA_DIAG_FLOOR;
            }
        }

        self.omega =
            OmegaMatrix::from_matrix(new_omega, self.omega.eta_names.clone(), self.omega.diagonal);
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let dim = self.omega.dim();
        let l = &self.omega.chol;
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
                let z_vec = DVector::from_column_slice(&z);
                (l * z_vec).iter().copied().collect()
            })
            .collect()
    }

    fn proposal_chol(&self) -> &DMatrix<f64> {
        &self.omega.chol
    }

    fn to_omega_matrix(&self) -> &OmegaMatrix {
        &self.omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pk::EventPkParams;
    use crate::stats::likelihood::{individual_nll_into, obs_nll_subject_into};
    use crate::types::test_helpers::analytical_model;
    use crate::types::{
        DoseEvent, FitOptions, GradientMethod, ModelParameters, Population, Subject,
    };
    use std::collections::HashMap;

    /// Build the minimal (model, population, init_params) needed for parity tests.
    fn make_parity_fixtures() -> (crate::types::CompiledModel, Subject, ModelParameters) {
        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 8.0],
            observations: vec![2.5, 1.8, 0.9],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![],
            dose_occasions: vec![],
        };
        let init_params = model.default_params.clone();
        (model, subj, init_params)
    }

    /// Rung 1 parity: GaussianOmega.log_prior(η) must equal the prior term
    /// that individual_nll_into adds on top of obs_nll_subject_into.
    ///
    /// individual_nll_into = obs_nll + 0.5(η′Ω⁻¹η + log|Ω|)
    ///                     = obs_nll + GaussianOmega::log_prior(η)
    ///
    /// So: log_prior = individual_nll_into − obs_nll_subject_into.
    /// This test proves the two code paths are mathematically identical.
    #[test]
    fn gaussian_omega_log_prior_matches_individual_nll_prior_term() {
        let (model, subj, init_params) = make_parity_fixtures();
        let gom = GaussianOmega::from_init_params(&init_params);
        let theta = &init_params.theta;
        let sigma = &init_params.sigma.values;
        let omega = &init_params.omega;

        // Test at several η values including the zero vector and a non-zero one.
        let test_etas: &[Vec<f64>] = &[vec![0.0], vec![0.5], vec![-0.3], vec![1.2]];

        for eta in test_etas {
            let mut scratch = EventPkParams::default();
            let full_nll =
                individual_nll_into(&model, &subj, theta, eta, omega, sigma, &mut scratch);
            let mut scratch2 = EventPkParams::default();
            let obs_nll = obs_nll_subject_into(&model, &subj, theta, sigma, eta, &mut scratch2);
            let prior_from_trait = gom.log_prior(eta);
            let prior_from_nll = full_nll - obs_nll;

            assert!(
                (prior_from_trait - prior_from_nll).abs() < 1e-9,
                "η={eta:?}: GaussianOmega.log_prior={prior_from_trait:.12e} \
                 ≠ individual_nll − obs_nll={prior_from_nll:.12e}"
            );
        }
    }

    /// mstep_update with gamma=1 and a single sample must set Ω to ηηᵀ
    /// (clamped at the floor for diagonal entries), matching the closed-form
    /// M-step of the first SAEM exploration iteration.
    #[test]
    fn gaussian_omega_mstep_gamma1_produces_sample_covariance() {
        let (_, _, init_params) = make_parity_fixtures();
        let mut gom = GaussianOmega::from_init_params(&init_params);

        let eta = vec![0.4_f64];
        gom.mstep_update(&[eta.clone()], 1.0);

        // With one sample and gamma=1: Ω should be ηηᵀ = 0.16.
        let expected = eta[0] * eta[0];
        let actual = gom.omega.matrix[(0, 0)];
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected Ω[0,0]={expected}, got {actual}"
        );
    }

    /// The diagonal floor is applied when the sample covariance would go below
    /// OMEGA_DIAG_FLOOR, preventing the chain from collapsing.
    #[test]
    fn gaussian_omega_mstep_applies_diagonal_floor() {
        let (_, _, init_params) = make_parity_fixtures();
        let mut gom = GaussianOmega::from_init_params(&init_params);

        // eta = 0 → ηηᵀ = 0 → would undercut floor.
        gom.mstep_update(&[vec![0.0]], 1.0);
        assert!(
            gom.omega.matrix[(0, 0)] >= OMEGA_DIAG_FLOOR,
            "Ω diagonal must be ≥ floor after mstep_update"
        );
    }

    /// sample() draws correctly-shaped MVN samples from the current omega.
    #[test]
    fn gaussian_omega_sample_shape_and_finite() {
        let (_, _, init_params) = make_parity_fixtures();
        let gom = GaussianOmega::from_init_params(&init_params);
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = gom.sample(100, &mut rng);
        assert_eq!(samples.len(), 100);
        for s in &samples {
            assert_eq!(s.len(), gom.omega.dim());
            assert!(s.iter().all(|x| x.is_finite()));
        }
    }
}
