//! Gauss-Hermite rules for adaptive quadrature and the AGQ analytic-score scope. Below the
//! estimators so the parser and the adjoint can name them (#3041).
use crate::cold_sort::ColdSort;
use crate::types::CompiledModel;
use nalgebra::DMatrix;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Upper bound on `n_agq`. Beyond ~20 nodes the Golub–Welsch eigenproblem starts to lose
/// the extreme nodes to round-off, and the marginal accuracy gain is nil for any realistic
/// NLME integrand.
pub const MAX_AGQ_NODES: usize = 21;

pub(crate) static GAUSS_HERMITE_RULES: [OnceLock<(Vec<f64>, Vec<f64>, Vec<f64>)>; MAX_AGQ_NODES] =
    [const { OnceLock::new() }; MAX_AGQ_NODES];

pub(crate) fn cached_gauss_hermite(n: usize) -> (&'static [f64], &'static [f64], &'static [f64]) {
    assert!((1..=MAX_AGQ_NODES).contains(&n));
    let (nodes, weights, log_weights) = GAUSS_HERMITE_RULES[n - 1].get_or_init(|| {
        let (nodes, weights) = gauss_hermite(n);
        let log_weights = weights.iter().map(|w| w.ln()).collect();
        (nodes, weights, log_weights)
    });
    (nodes, weights, log_weights)
}

/// Gauss–Hermite nodes and weights for the *physicists'* weight `e^{−x²}` (so the weights
/// sum to `√π`), via Golub–Welsch: the nodes are the eigenvalues of the symmetric
/// tridiagonal Jacobi matrix with zero diagonal and off-diagonal `β_k = √(k/2)`, and the
/// weights are `√π · v_{0,k}²` from the first component of each unit eigenvector.
///
/// Computing them beats a hard-coded table: it is exact at every `n` (no transcription
/// risk), and `n = 1` falls out as the 1×1 zero matrix → node `0`, weight `√π`, which is
/// precisely what makes AGQ collapse to Laplace.
pub(crate) fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1, "gauss_hermite needs at least one node");
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let beta = (k as f64 / 2.0).sqrt();
        j[(k - 1, k)] = beta;
        j[(k, k - 1)] = beta;
    }
    let eig = j.symmetric_eigen();
    let sqrt_pi = PI.sqrt();
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let v0 = eig.eigenvectors[(0, k)];
            (eig.eigenvalues[k], sqrt_pi * v0 * v0)
        })
        .collect();
    // Ascending node order: the grid is then deterministic and reproducible run to run,
    // independent of whatever order the eigensolver happened to converge in.
    pairs.cold_sort_by(&|a, b| a.0.partial_cmp(&b.0).expect("GH nodes are finite"));
    pairs.into_iter().unzip()
}

/// Alias kept for `build_info::gradient_method_outer`, which reports scope by model.
/// Identical to [`analytic_score_supported`] — the predicate is purely model-level.
pub fn analytic_score_supported_model(model: &CompiledModel) -> bool {
    analytic_score_supported(model)
}

/// Whether the AGQ/Laplace analytic score applies to `model`: **the same predicate
/// FOCE/FOCEI gate on** ([`analytic_outer_gradient_available`](crate::sens::provider::analytic_outer_gradient_available)),
/// with no separate copy, so widening the adjoint's scope extends FOCE, FOCEI, AGQ and Laplace
/// together (#251). Model-level only: a subject the adjoint declines at run time takes the
/// per-subject finite-difference score.
pub fn analytic_score_supported(model: &CompiledModel) -> bool {
    crate::sens::provider::analytic_outer_gradient_available(model)
}
