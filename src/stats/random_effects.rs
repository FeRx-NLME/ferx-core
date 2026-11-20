//! Random-effect (`eta`) prior distribution abstraction for SAEM.
//!
//! The default Gaussian SAEM path in `estimation/saem.rs` uses `OmegaMatrix`
//! directly and stays untouched. [`RandomEffectDistribution`] provides the
//! interface consumed by the opt-in `omega_dist = vine` SAEM estimator (and
//! [`GaussianOmega`] provides an exact parity reference for Rung-1 tests).

use crate::types::{ModelParameters, OmegaMatrix};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Prior distribution over a per-subject random-effect vector `eta in R^d`.
pub trait RandomEffectDistribution: Send + Sync {
    /// Dimension `d = n_eta`.
    fn dim(&self) -> usize;

    /// Prior negative log-likelihood in FOCE/NONMEM convention (omitting `(d/2) ln(2*pi)`):
    /// `prior_nll(eta) = -ln p(eta) - (d/2) ln(2*pi)`.
    ///
    /// Adding this to `obs_nll_subject_into` matches `individual_nll` bit-for-bit
    /// when `p(eta) = N(0, Omega)`.
    fn prior_nll(&self, eta: &[f64]) -> f64;

    /// Exact normalized joint log-density `ln p(eta)`.
    #[inline]
    fn log_density(&self, eta: &[f64]) -> f64 {
        let half_d_ln_2pi = 0.5 * (self.dim() as f64) * std::f64::consts::TAU.ln();
        -self.prior_nll(eta) - half_d_ln_2pi
    }

    /// Gradient `d/d(eta) [prior_nll(eta)] = -d/d(eta) [ln p(eta)]`.
    fn prior_nll_grad(&self, eta: &[f64]) -> Vec<f64>;

    /// Robbins-Monro stochastic-approximation M-step update at step size `gamma`.
    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64);

    /// Draw `n` independent joint `eta` vectors from the fitted distribution.
    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>>
    where
        Self: Sized;

    /// Lower-triangular Cholesky factor scaling the symmetric block MH proposal.
    fn proposal_chol(&self) -> &DMatrix<f64>;

    /// Per-coordinate marginal standard deviations (used by the componentwise MH sweep).
    fn marginal_sds(&self) -> Vec<f64>;

    /// Secondary Gaussian-equivalent `OmegaMatrix` summary.
    fn to_omega_matrix(&self) -> &OmegaMatrix;
}

/// Multivariate-normal `N(0, Omega)` implementation of [`RandomEffectDistribution`]
/// used for Rung-1 parity verification against the production Gaussian SAEM path.
#[derive(Debug, Clone)]
pub struct GaussianOmega {
    omega: OmegaMatrix,
    s2: DMatrix<f64>,
    initial_omega: OmegaMatrix,
    omega_fixed: Vec<bool>,
}

impl GaussianOmega {
    /// Construct from initial model parameters.
    pub fn from_init_params(init_params: &ModelParameters) -> Self {
        let omega = init_params.omega.clone();
        let s2 = omega.matrix.clone();
        let initial_omega = omega.clone();
        Self { omega, s2, initial_omega, omega_fixed: init_params.omega_fixed.clone() }
    }
}

impl RandomEffectDistribution for GaussianOmega {
    fn dim(&self) -> usize {
        self.omega.dim()
    }

    fn prior_nll(&self, eta: &[f64]) -> f64 {
        if !self.omega.log_det.is_finite() {
            return 1e20;
        }
        let ev = DVector::from_column_slice(eta);
        let quad = ev.dot(&(&self.omega.inv * &ev));
        0.5 * (quad + self.omega.log_det)
    }

    fn prior_nll_grad(&self, eta: &[f64]) -> Vec<f64> {
        let ev = DVector::from_column_slice(eta);
        let g = &self.omega.inv * ev;
        g.iter().copied().collect()
    }

    fn mstep_update(&mut self, sampled_etas: &[Vec<f64>], gamma: f64) {
        if sampled_etas.is_empty() || gamma <= 0.0 {
            return;
        }
        let d = self.omega.dim();
        let mut eta_outer = DMatrix::zeros(d, d);
        for eta in sampled_etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= sampled_etas.len() as f64;
        self.s2 = (1.0 - gamma) * &self.s2 + gamma * &eta_outer;

        let mut mat = self.s2.clone();
        for i in 0..d {
            for j in 0..d {
                if !self.initial_omega.free_mask[(i, j)] {
                    mat[(i, j)] = 0.0;
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = self.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    mat[(i, j)] = self.initial_omega.matrix[(i, j)];
                }
            }
        }
        for i in 0..d {
            let fi = self.omega_fixed.get(i).copied().unwrap_or(false);
            if !fi && mat[(i, i)] < 1e-6 {
                mat[(i, i)] = 1e-6;
            }
        }
        self.omega = OmegaMatrix::from_matrix_with_mask(
            mat,
            self.initial_omega.eta_names.clone(),
            self.initial_omega.diagonal,
            self.initial_omega.free_mask.clone(),
        );
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let d = self.omega.dim();
        let l = self.omega.sim_draw_factor(Some(&self.initial_omega), &self.omega_fixed);
        (0..n)
            .map(|_| {
                let z: Vec<f64> = (0..d).map(|_| StandardNormal.sample(rng)).collect();
                (&*l * DVector::from_column_slice(&z)).iter().copied().collect()
            })
            .collect()
    }

    fn proposal_chol(&self) -> &DMatrix<f64> {
        &self.omega.chol
    }

    fn marginal_sds(&self) -> Vec<f64> {
        (0..self.omega.dim()).map(|i| self.omega.matrix[(i, i)].max(1e-6).sqrt()).collect()
    }

    fn to_omega_matrix(&self) -> &OmegaMatrix {
        &self.omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::likelihood::{individual_nll, obs_nll_subject_into};
    use crate::types::test_helpers::analytical_model;
    use crate::types::{DoseEvent, GradientMethod, Subject};

    #[test]
    fn gaussian_omega_prior_nll_matches_individual_nll_bit_for_bit() {
        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 12.0],
            observations: vec![3.2, 2.1, 0.9],
            obs_cmts: vec![1, 1, 1],
            cens: vec![0, 0, 0],
            ..Default::default()
        };
        let params = &model.default_params;
        let gauss = GaussianOmega::from_init_params(params);
        let eta = vec![0.18];
        let mut scratch = crate::pk::EventPkParams::default();
        let obs_only = obs_nll_subject_into(
            &model,
            &subj,
            &params.theta,
            &params.sigma.values,
            crate::stats::residual_error::ResidualCov::block_sigma(&[]),
            &eta,
            &mut scratch,
        );
        let via_trait = obs_only + gauss.prior_nll(&eta);
        let via_prod =
            individual_nll(&model, &subj, &params.theta, &eta, &params.omega, &params.sigma.values);
        assert!(
            (via_trait - via_prod).abs() < 1e-12,
            "trait {via_trait} vs individual_nll {via_prod}"
        );
    }
}
