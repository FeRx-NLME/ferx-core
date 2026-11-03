//! The independence kernels of the SAEM E-step: NONMEM's modes 1 and 1B (*NONMEM Users
//! Guide, Introduction to NONMEM 7.5*, §I.38).
//!
//! For each sample NONMEM runs, in order, `ISAMPLE_M1` mode-1 proposals from the
//! population distribution `N(0, Ω)`, `ISAMPLE_M1B` mode-1B proposals from "the
//! individual conditional mean and individual conditional variance collected from
//! previous iterations", then the random-walk modes 2 (block) and 3 (componentwise).
//! ferx had only the last two. Both independence kernels propose without reference to
//! the chain's state, which is what lets a chain cross its posterior in one move where a
//! random walk takes a fraction of an SD per accepted step, and mode 1 alone is what gets
//! a chain out of a region the others cannot leave.
//!
//! Each is a Metropolis–Hastings kernel for an independence proposal `q`: a proposal
//! `η*` is accepted with probability `π(η*) q(η) / (π(η) q(η*))`, where `π` is the same
//! individual objective the random-walk kernels score. A poor `q` costs acceptance, never
//! validity. Under mode 1, `q` is the prior factor of `π` itself, so the ratio is the
//! likelihood ratio alone.
//!
//! Mode 1B's moments are each subject's running mean and covariance of its draws, over
//! every iteration so far ([`ConditionalMoments`]). An adaptive proposal of this kind keeps
//! the chain's target because the adaptation diminishes: the `k`-th draw moves the
//! moments by `O(1/k)` (Roberts & Rosenthal 2007, as for the Haario sampler). The kernel
//! waits for more than `2·n_η` draws, so the covariance is not singular by construction.

use super::MhScratch;
use crate::stats::likelihood::individual_nll_prepared;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// A Gaussian independence proposal `N(mean, L·Lᵀ)`.
#[derive(Debug, Clone)]
pub(crate) struct GaussianProposal {
    mean: Vec<f64>,
    /// Lower Cholesky factor of the proposal covariance.
    l: DMatrix<f64>,
}

impl GaussianProposal {
    /// The proposal with covariance `cov`, or `None` where `cov` is not positive definite
    /// (or not finite, or the wrong size).
    pub(crate) fn from_covariance(mean: Vec<f64>, cov: DMatrix<f64>) -> Option<Self> {
        if cov.nrows() != mean.len()
            || cov.ncols() != mean.len()
            || !cov.iter().chain(&mean).all(|v| v.is_finite())
        {
            return None;
        }
        let l = cov.cholesky()?.l();
        Some(Self { mean, l })
    }

    /// The population proposal of mode 1, `N(0, Ω)`, from `Ω`'s lower Cholesky factor.
    pub(crate) fn prior(chol: &DMatrix<f64>) -> Self {
        Self {
            mean: vec![0.0; chol.nrows()],
            l: chol.clone(),
        }
    }

    /// `ln q(x)` up to the normalising constant, which cancels in an independence ratio:
    /// `−½ (x − m)ᵀ (L Lᵀ)⁻¹ (x − m) = −½ ‖L⁻¹(x − m)‖²`.
    pub(crate) fn quad(&self, x: &[f64]) -> f64 {
        let n = self.mean.len();
        let mut w = vec![0.0; n];
        let mut s = 0.0;
        for i in 0..n {
            let mut v = x[i] - self.mean[i];
            for j in 0..i {
                v -= self.l[(i, j)] * w[j];
            }
            w[i] = v / self.l[(i, i)];
            s += w[i] * w[i];
        }
        -0.5 * s
    }

    /// A draw `m + L z` for a standard normal `z`, into `out`.
    pub(crate) fn draw_into(&self, z: &[f64], out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate().take(self.mean.len()) {
            *o = self.mean[i] + (0..=i).map(|j| self.l[(i, j)] * z[j]).sum::<f64>();
        }
    }
}

/// One subject's running mean and covariance of its η draws over the iterations so far
/// (Welford), the moments of mode 1B's proposal.
#[derive(Debug, Clone)]
pub(crate) struct ConditionalMoments {
    n: usize,
    mean: DVector<f64>,
    /// Sum of outer products of deviations, `Σ (x − x̄)(x − x̄)ᵀ`.
    m2: DMatrix<f64>,
}

impl ConditionalMoments {
    pub(crate) fn new(n_eta: usize) -> Self {
        Self {
            n: 0,
            mean: DVector::zeros(n_eta),
            m2: DMatrix::zeros(n_eta, n_eta),
        }
    }

    /// Add one iteration's draw.
    pub(crate) fn push(&mut self, x: &[f64]) {
        let x = DVector::from_column_slice(x);
        self.n += 1;
        let d = &x - &self.mean;
        self.mean += &d / self.n as f64;
        let d2 = &x - &self.mean;
        self.m2 += &d * d2.transpose();
    }

    /// Mode 1B's proposal from these moments: `None` until more than `2·n_η` draws, or
    /// where the sample covariance is not positive definite.
    pub(crate) fn proposal(&self) -> Option<GaussianProposal> {
        let d = self.mean.len();
        if self.n <= 2 * d {
            return None;
        }
        let cov = &self.m2 / (self.n - 1) as f64;
        GaussianProposal::from_covariance(self.mean.iter().copied().collect(), cov)
    }
}

/// `n_steps` independence Metropolis–Hastings moves for one subject from `prop`, in place
/// on `eta`. Returns `(n_accepted, nll)`. The target and its scoring are the block random
/// walk's ([`super::mh_steps`]); only the proposal and the `q` ratio differ.
#[allow(clippy::too_many_arguments)]
pub(crate) fn independence_steps(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    prop: &GaussianProposal,
    rng: &mut impl Rng,
    n_steps: usize,
    scratch: &mut MhScratch,
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> (usize, f64) {
    let n_eta = eta.len();
    let mut nll = nll_current;
    let mut log_q_cur = prop.quad(eta);
    let mut n_accepted = 0;
    let MhScratch {
        nll: nll_scratch,
        prep,
        z,
        eta_prop,
        ..
    } = scratch;
    for _ in 0..n_steps {
        for slot in z.iter_mut() {
            *slot = rng.sample(StandardNormal);
        }
        prop.draw_into(z.as_slice(), &mut eta_prop[..n_eta]);
        let nll_prop = individual_nll_prepared(
            model,
            subject,
            theta,
            eta_prop,
            omega,
            sigma_values,
            prep,
            schedule,
            nll_scratch,
        );
        let log_q_prop = prop.quad(eta_prop);
        // π ∝ exp(−nll): log α = (nll − nll*) + (ln q(η) − ln q(η*)). A non-finite
        // proposal objective compares false and is rejected.
        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < (nll - nll_prop) + (log_q_cur - log_q_prop) {
            eta.copy_from_slice(eta_prop);
            nll = nll_prop;
            log_q_cur = log_q_prop;
            n_accepted += 1;
        }
    }
    (n_accepted, nll)
}

#[cfg(test)]
#[path = "saem_independence_tests.rs"]
mod tests;
