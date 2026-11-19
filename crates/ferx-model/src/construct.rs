//! Constructors ferx-core needs for `#[non_exhaustive]` types of this crate. They exist for
//! ferx-core's own users; across the crate boundary they would also stop ferx-core from
//! building the values it produces. Not re-exported by ferx-core, so not public API.
use crate::types::ViResult;

/// Every field of [`ViResult`], as a plain struct.
pub struct ViResultParts {
    pub neg_two_elbo: f64,
    pub data_term: f64,
    pub kl_term: f64,
    pub n_iterations: usize,
    pub converged: bool,
    pub family: String,
    pub n_mc_samples: usize,
    pub kl: String,
    pub n_kl_fallback_subjects: usize,
    pub elbo_trace: Vec<f64>,
    pub objective_trace: Vec<f64>,
    pub eta_means: Vec<Vec<f64>>,
    pub kappa_means: Vec<Vec<Vec<f64>>>,
    pub eta_covs: Vec<Vec<Vec<f64>>>,
    pub n_fd_subjects: usize,
    pub elbo_tightness_ratio: f64,
    pub superseded_by: Option<String>,
}

/// A [`ViResult`] from its fields.
pub fn vi_result(p: ViResultParts) -> ViResult {
    ViResult {
        neg_two_elbo: p.neg_two_elbo,
        data_term: p.data_term,
        kl_term: p.kl_term,
        n_iterations: p.n_iterations,
        converged: p.converged,
        family: p.family,
        n_mc_samples: p.n_mc_samples,
        kl: p.kl,
        n_kl_fallback_subjects: p.n_kl_fallback_subjects,
        elbo_trace: p.elbo_trace,
        objective_trace: p.objective_trace,
        eta_means: p.eta_means,
        kappa_means: p.kappa_means,
        eta_covs: p.eta_covs,
        n_fd_subjects: p.n_fd_subjects,
        elbo_tightness_ratio: p.elbo_tightness_ratio,
        superseded_by: p.superseded_by,
    }
}
