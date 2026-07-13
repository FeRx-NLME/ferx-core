//! Adaptive Gauss–Hermite quadrature (AGQ) — the marginal-likelihood objective (#251).
//!
//! # What it computes
//!
//! FOCE/FOCEI approximate each subject's marginal likelihood
//! `L_i = ∫ p(y_i|η) φ(η; 0, Ω) dη` with a *single* Gaussian centred at the empirical-Bayes
//! mode. AGQ keeps that centring but evaluates the integrand on a Gauss–Hermite grid laid
//! around the mode, so the approximation improves with the node count instead of being
//! fixed. Writing `l_i(η) = log p(y_i|η) + log φ(η; 0, Ω)` for the exact conditional
//! log-likelihood, `η̂` for the mode and `H = −∇²l_i(η̂)`:
//!
//! ```text
//!   L_i ≈ 2^(d/2) · |Σ^{1/2}| · Σ_j [ (Π_k w_{j,k}) · exp(‖z_j‖²) · exp(l_i(η̂ + √2·Σ^{1/2}·z_j)) ]
//! ```
//!
//! with `Σ = H⁻¹`, `z_j` the tensor-product Gauss–Hermite nodes and `w` the GH weights. The
//! `exp(‖z_j‖²)` factor undoes the `e^{−z²}` that the Hermite weights carry, so what is
//! actually integrated is the *full* integrand rather than a Gaussian-shaped surrogate.
//!
//! **Two properties follow, and they are the whole point of the method:**
//!
//! 1. **`n_agq = 1` is exactly Laplace.** The one-point rule is `z = 0`, `w = √π`, so the
//!    sum collapses to `(2π)^(d/2) · |H|^(−1/2) · exp(l_i(η̂))` — the Laplace approximation,
//!    term for term. This is not an approximation of an approximation; it is an identity,
//!    and [`tests::one_node_agq_equals_laplace`] pins it.
//! 2. **No Gaussian-residual assumption.** `l_i` is evaluated through
//!    [`individual_nll_into_with_schedule`], the model's *actual* likelihood — so
//!    time-to-event and categorical endpoints are integrated as faithfully as Gaussian
//!    ones. That is what FOCE/FOCEI structurally cannot do, and why AGQ exists here.
//!
//! # Constant convention
//!
//! `individual_nll_*` omits the `(d + n_obs)/2 · log(2π)` constants (NONMEM's "objective
//! function without constant"). Writing `G_i = ∫ exp(−nll(η)) dη`, the OFV in that same
//! convention is `−2·Σ_i log G_i + Σ_i d·log(2π)`, so this module's per-subject
//! contribution — the quantity the outer loop doubles into an OFV — is
//!
//! ```text
//!   agq_nll_i = (d/2)·log(π) + ½·log|H| − logΣexp_j[ Σ_k log w_{j,k} + ‖z_j‖² − nll(η_j) ]
//! ```
//!
//! which at one node reduces to `nll(η̂) + ½·log|H|`: the FOCEI Laplace per-subject NLL,
//! in the same units as [`crate::stats::likelihood::foce_subject_nll`]. AGQ OFVs are
//! therefore directly comparable to FOCE/FOCEI OFVs from this engine.
//!
//! # Why the Hessian is finite-differenced
//!
//! `H` is the Hessian of the *true* integrand, obtained by central differences of
//! `individual_nll`. It is deliberately **not**
//! [`crate::estimation::importance_sampling::compute_posterior_hessian`], which builds the
//! Gauss-Newton form `Ω⁻¹ + JᵀR⁻¹J`: that carries no curvature at all from TTE or
//! categorical endpoints, i.e. it is blind on exactly the models AGQ is here to serve, and
//! would scale their grids by `Ω` alone. Note the grid scaling only affects *accuracy at
//! finite n* — AGQ is consistent under any invertible scaling — but at `n_agq = 1` it enters
//! the objective directly through `½·log|H|`, so a blind `H` would silently not be Laplace.
//!
//! # Grid pruning: why there isn't any
//!
//! Dropping low-weight nodes looks free and is not. The *corrected* weight
//! `w_k · exp(z_k²)` is `Θ(1)` across the grid — the Hermite weight's `e^{−z²}` decay is
//! exactly what the correction cancels — so a threshold on `w_k` prunes nodes whose actual
//! contribution is not small. The contribution is only known after `nll(η_j)` is evaluated,
//! which is the entire cost. Dimensionality is therefore bounded by [`MAX_AGQ_GRID`]
//! instead, and a sparse (Smolyak) rule is the real answer for `d = 5..8` (issue #251).

use nalgebra::DMatrix;
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::estimation::importance_sampling::build_proposal;
use crate::estimation::inner_optimizer::cacheable_schedule;
use crate::pk;
use crate::stats::likelihood::individual_nll_into_with_schedule;
use crate::types::{CompiledModel, ModelParameters, Population, Subject};

/// Upper bound on `n_agq`. Beyond ~20 nodes the Golub–Welsch eigenproblem starts to lose
/// the extreme nodes to round-off, and the marginal accuracy gain is nil for any realistic
/// NLME integrand.
pub const MAX_AGQ_NODES: usize = 21;

/// Hard cap on the tensor-grid size `n_agq^n_eta`, enforced at model-check time by
/// [`crate::api::check_model_options`]. The tensor rule costs one full likelihood
/// evaluation per node per subject per outer iteration, so the grid — not the node count —
/// is the quantity a user needs protecting from. 100k nodes on an ODE model is already
/// hours; past that a fit is not slow, it is wrong to have started.
pub const MAX_AGQ_GRID: usize = 100_000;

/// The sentinel a non-finite likelihood collapses to, matching `individual_nll`'s own
/// convention so a diverged node sorts as "impossible" rather than poisoning the sum.
const NLL_SENTINEL: f64 = 1e20;

/// Number of nodes in the tensor grid, saturating at [`usize::MAX`] rather than
/// overflowing — callers compare against [`MAX_AGQ_GRID`] and reject long before that.
pub fn grid_size(n_nodes: usize, n_eta: usize) -> usize {
    n_nodes.saturating_pow(n_eta as u32)
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
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("GH nodes are finite"));
    pairs.into_iter().unzip()
}

/// Numerically stable `log Σ exp(xᵢ)`.
fn logsumexp(xs: &[f64]) -> f64 {
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

/// Central-difference Hessian of the exact conditional NLL w.r.t. η at `eta_hat`.
///
/// Step `hᵢ = max(ε^¼ · √Ωᵢᵢ, 1e-4)`: `ε^¼` is the standard second-difference step (it
/// balances truncation against the `ε/h²` round-off blow-up), scaled by the prior SD
/// because that is η's natural unit. The `1e-4` floor keeps a near-zero (or FIXed-small)
/// Ωᵢᵢ from driving `h` so small that `ε/h²` swamps the curvature.
fn fd_posterior_hessian(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> DMatrix<f64> {
    let d = eta_hat.len();
    let mut nll_at = |eta: &[f64]| -> f64 {
        individual_nll_into_with_schedule(
            model,
            subject,
            &params.theta,
            eta,
            &params.omega,
            &params.sigma.values,
            scratch,
            schedule,
        )
    };

    let steps: Vec<f64> = (0..d)
        .map(|i| {
            let sd = params.omega.matrix[(i, i)].max(0.0).sqrt();
            (f64::EPSILON.powf(0.25) * sd).max(1e-4)
        })
        .collect();

    let f0 = nll_at(eta_hat);
    let mut h = DMatrix::<f64>::zeros(d, d);
    let mut eta = eta_hat.to_vec();

    for i in 0..d {
        let hi = steps[i];
        eta[i] = eta_hat[i] + hi;
        let f_plus = nll_at(&eta);
        eta[i] = eta_hat[i] - hi;
        let f_minus = nll_at(&eta);
        eta[i] = eta_hat[i];
        h[(i, i)] = (f_plus - 2.0 * f0 + f_minus) / (hi * hi);
    }

    for i in 0..d {
        for j in (i + 1)..d {
            let (hi, hj) = (steps[i], steps[j]);
            let mut at = |si: f64, sj: f64| -> f64 {
                eta[i] = eta_hat[i] + si * hi;
                eta[j] = eta_hat[j] + sj * hj;
                let v = nll_at(&eta);
                eta[i] = eta_hat[i];
                eta[j] = eta_hat[j];
                v
            };
            let mixed =
                (at(1.0, 1.0) - at(1.0, -1.0) - at(-1.0, 1.0) + at(-1.0, -1.0)) / (4.0 * hi * hj);
            h[(i, j)] = mixed;
            h[(j, i)] = mixed;
        }
    }
    h
}

/// AGQ marginal NLL for one subject, in the same "without constant" units as
/// [`crate::stats::likelihood::foce_subject_nll`] — see the module docs for the derivation.
///
/// `eta_hat` is the converged EBE mode from the shared inner loop; AGQ does **not**
/// re-optimise it, it only lays the grid around it. `nodes`/`log_weights` are the 1-D
/// Gauss–Hermite rule, hoisted by the caller so the eigenproblem is solved once per
/// population rather than once per subject.
pub(crate) fn agq_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_hat: &[f64],
    nodes: &[f64],
    log_weights: &[f64],
) -> f64 {
    let d = eta_hat.len();
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    let schedule = cacheable_schedule(model, subject);

    // No random effects: the "integral" is a point mass and AGQ degenerates to the
    // conditional likelihood itself. (The formula below would agree — an empty tensor
    // product is the single empty node — but there is no Σ to factor, so short-circuit.)
    if d == 0 {
        return individual_nll_into_with_schedule(
            model,
            subject,
            &params.theta,
            eta_hat,
            &params.omega,
            &params.sigma.values,
            &mut scratch,
            schedule.as_ref(),
        );
    }

    let h = fd_posterior_hessian(
        model,
        subject,
        params,
        eta_hat,
        &mut scratch,
        schedule.as_ref(),
    );
    // `build_proposal` applies the relative jitter and — if the FD Hessian came back
    // indefinite (a loosely-converged mode, a flat direction) — falls back to the Ω-scale
    // factor. That fallback keeps AGQ *consistent* (any invertible scale integrates to the
    // same limit); it only costs nodes-worth of efficiency, which is the right failure mode.
    let Some(proposal) = build_proposal(&h, &params.omega.inv, d) else {
        return NLL_SENTINEL;
    };

    let n = nodes.len();
    let mut terms = Vec::with_capacity(grid_size(n, d));
    let mut idx = vec![0usize; d];
    let mut z = vec![0.0f64; d];
    let mut step = vec![0.0f64; d];
    let mut eta = vec![0.0f64; d];

    loop {
        let mut log_w = 0.0;
        let mut z_sq = 0.0;
        for k in 0..d {
            let zk = nodes[idx[k]];
            z[k] = zk;
            z_sq += zk * zk;
            log_w += log_weights[idx[k]];
        }
        // η_j = η̂ + √2 · Σ^{1/2} · z_j — the adaptive transform.
        proposal.apply_l_sigma(&z, &mut step, std::f64::consts::SQRT_2);
        for k in 0..d {
            eta[k] = eta_hat[k] + step[k];
        }
        let nll = individual_nll_into_with_schedule(
            model,
            subject,
            &params.theta,
            &eta,
            &params.omega,
            &params.sigma.values,
            &mut scratch,
            schedule.as_ref(),
        );
        // A diverged node returns `individual_nll`'s 1e20 sentinel, which lands here as a
        // ~−1e20 log-term — negligible under logsumexp unless *every* node diverged, in
        // which case the subject correctly reports the sentinel back.
        terms.push(log_w + z_sq - nll);

        // Mixed-radix increment over the d-dimensional tensor grid.
        let mut k = 0;
        while k < d {
            idx[k] += 1;
            if idx[k] < n {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
        if k == d {
            break;
        }
    }

    let nll = 0.5 * d as f64 * PI.ln() + 0.5 * proposal.log_det_inv_scale - logsumexp(&terms);
    if nll.is_finite() {
        nll
    } else {
        NLL_SENTINEL
    }
}

/// Population AGQ objective: `Σ_i agq_subject_nll_i`. The outer loop doubles this into an
/// OFV, exactly as it does [`crate::estimation::outer_optimizer::pop_nll`].
///
/// Parallel over subjects (the grid sweep stays serial *within* a subject, matching
/// importance sampling), then reduced **serially in subject order** — a rayon `.sum()`
/// folds along thread-count-dependent split boundaries and f64 addition is not associative,
/// which would make the OFV depend on the thread count (#703).
pub(crate) fn agq_population_nll(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[nalgebra::DVector<f64>],
    n_nodes: usize,
) -> f64 {
    let (nodes, weights) = gauss_hermite(n_nodes);
    let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();

    let per_subject: Vec<f64> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            agq_subject_nll(
                model,
                subject,
                params,
                eta_hats[i].as_slice(),
                &nodes,
                &log_weights,
            )
        })
        .collect();
    per_subject.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golub–Welsch must reproduce the textbook physicists' Hermite rule.
    #[test]
    fn gauss_hermite_matches_known_rules() {
        // n = 1: the rule that makes AGQ collapse to Laplace.
        let (x, w) = gauss_hermite(1);
        assert!((x[0] - 0.0).abs() < 1e-14);
        assert!((w[0] - PI.sqrt()).abs() < 1e-14);

        // n = 3: nodes ±√(3/2), 0; weights √π/6, 2√π/3, √π/6.
        let (x, w) = gauss_hermite(3);
        let r = (1.5f64).sqrt();
        for (got, want) in x.iter().zip([-r, 0.0, r]) {
            assert!((got - want).abs() < 1e-12, "node {got} != {want}");
        }
        let sp = PI.sqrt();
        for (got, want) in w.iter().zip([sp / 6.0, 2.0 * sp / 3.0, sp / 6.0]) {
            assert!((got - want).abs() < 1e-12, "weight {got} != {want}");
        }
    }

    /// Weights sum to `√π = ∫ e^{−x²} dx`, and the rule is exact for polynomials up to
    /// degree `2n − 1` — the defining property of an n-point Gaussian rule.
    #[test]
    fn gauss_hermite_integrates_polynomials_exactly() {
        for n in 1..=MAX_AGQ_NODES {
            let (x, w) = gauss_hermite(n);
            let total: f64 = w.iter().sum();
            assert!(
                (total - PI.sqrt()).abs() < 1e-10,
                "n={n}: weights sum to {total}, want √π"
            );
            // ∫ x² e^{−x²} dx = √π/2 — needs degree 2, exact for every n ≥ 2.
            if n >= 2 {
                let m2: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| wi * xi * xi).sum();
                assert!(
                    (m2 - PI.sqrt() / 2.0).abs() < 1e-10,
                    "n={n}: ∫x²e^{{−x²}} = {m2}, want √π/2"
                );
            }
        }
    }

    /// The identity the whole method rests on: with one node the AGQ formula *is* the
    /// Laplace approximation. Checked here on the raw arithmetic — a 1-D integrand whose
    /// exact Laplace value is known in closed form — so a regression in the constants
    /// (the `(d/2)·log π`, the `√2` scaling, the `exp(‖z‖²)` correction) is caught by a
    /// unit test rather than by a converging fit.
    #[test]
    fn one_node_agq_equals_laplace() {
        // Integrand exp(−nll) with nll(η) = ½·a·(η − m)² + c, i.e. an exact Gaussian.
        // Laplace is exact here: −log ∫ = c + ½·log(a) − ½·log(2π), and in the
        // "without constant" convention the per-subject NLL is nll(m) + ½·log|H| = c + ½·log a.
        let (a, m, c) = (2.5f64, 0.3f64, 1.7f64);
        let nll = |eta: f64| 0.5 * a * (eta - m).powi(2) + c;

        let (nodes, weights) = gauss_hermite(1);
        let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();

        // H = a (exact second derivative); Σ^{1/2} = a^{−½}.
        let h = a;
        let terms: Vec<f64> = (0..1)
            .map(|j| {
                let z = nodes[j];
                let eta = m + std::f64::consts::SQRT_2 * h.powf(-0.5) * z;
                log_w[j] + z * z - nll(eta)
            })
            .collect();
        let agq = 0.5 * PI.ln() + 0.5 * h.ln() - logsumexp(&terms);
        let laplace = nll(m) + 0.5 * h.ln();
        assert!(
            (agq - laplace).abs() < 1e-12,
            "1-node AGQ {agq} != Laplace {laplace}"
        );
    }

    /// With enough nodes AGQ must recover the *exact* marginal on an integrand where the
    /// truth is known — here a Gaussian, where Laplace is already exact, plus a genuinely
    /// non-Gaussian (skewed) integrand where it is not. The second case is the one that
    /// proves the extra nodes are doing real work.
    #[test]
    fn many_nodes_recover_exact_marginal_on_skewed_integrand() {
        // nll(η) = ½η² + η⁴/12  ⇒  ∫exp(−nll) has no closed form; get truth by fine
        // trapezoid on a wide grid (the integrand decays like e^{−η⁴/12}).
        let nll = |e: f64| 0.5 * e * e + e.powi(4) / 12.0;
        let (lo, hi, steps) = (-12.0f64, 12.0f64, 2_000_000);
        let dx = (hi - lo) / steps as f64;
        let truth: f64 = (0..=steps)
            .map(|i| {
                let e = lo + i as f64 * dx;
                let f = (-nll(e)).exp();
                if i == 0 || i == steps {
                    0.5 * f
                } else {
                    f
                }
            })
            .sum::<f64>()
            * dx;
        // Reference in the module's convention: (d/2)·log(2π) − log G, with d = 1.
        let want = 0.5 * (2.0 * PI).ln() - truth.ln();

        // Mode is η = 0; H = nll''(0) = 1.
        let (m, h) = (0.0f64, 1.0f64);
        let agq_with = |n: usize| -> f64 {
            let (nodes, weights) = gauss_hermite(n);
            let terms: Vec<f64> = (0..n)
                .map(|j| {
                    let z = nodes[j];
                    let eta = m + std::f64::consts::SQRT_2 * h.powf(-0.5) * z;
                    weights[j].ln() + z * z - nll(eta)
                })
                .collect();
            0.5 * PI.ln() + 0.5 * h.ln() - logsumexp(&terms)
        };

        // Laplace is badly biased on this integrand — it is *exactly* 0 here (the mode
        // term and the ½log|H| term cancel), against a true marginal NLL of ~0.1368.
        let laplace = agq_with(1);
        assert!(
            (laplace - want).abs() > 1e-3,
            "Laplace {laplace} unexpectedly matched truth {want}; the test integrand is not \
             non-Gaussian enough to prove the node sweep does anything"
        );

        // Adding nodes closes that gap monotonically. The 21-node residual (~3e-6) is
        // genuine Gauss–Hermite truncation, not a defect: the transformed integrand here is
        // exp(−z⁴/3), which no finite polynomial rule integrates exactly. A regression in
        // the transform or the constants would miss by O(0.1), not O(1e-6).
        let errs: Vec<f64> = [1usize, 3, 7, 21]
            .iter()
            .map(|&n| (agq_with(n) - want).abs())
            .collect();
        for w in errs.windows(2) {
            assert!(
                w[1] < w[0],
                "AGQ error must shrink with node count, got {errs:?}"
            );
        }
        assert!(
            *errs.last().unwrap() < 1e-5,
            "21-node AGQ error {} too large (truth {want})",
            errs.last().unwrap()
        );
    }

    #[test]
    fn grid_size_saturates_instead_of_overflowing() {
        assert_eq!(grid_size(3, 4), 81);
        assert_eq!(grid_size(1, 50), 1);
        // 21^50 overflows u64/usize many times over; must saturate, not wrap to something
        // small that would sneak past the MAX_AGQ_GRID check.
        assert_eq!(grid_size(21, 50), usize::MAX);
    }

    #[test]
    fn logsumexp_is_stable_and_handles_sentinels() {
        let want = (1.0f64.exp() + 2.0f64.exp()).ln();
        assert!((logsumexp(&[1.0, 2.0]) - want).abs() < 1e-12);
        // Huge magnitudes must not overflow.
        assert!((logsumexp(&[1e5, 1e5]) - (1e5 + 2.0f64.ln())).abs() < 1e-9);
        // A diverged node (the −1e20 term) is simply ignored next to a live one.
        assert!((logsumexp(&[-1e20, 3.0]) - 3.0).abs() < 1e-12);
    }
}
