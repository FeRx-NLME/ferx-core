//! Simulation-based goodness-of-fit diagnostics: NPDE and NPD.
//!
//! Unlike CWRES — which linearises the model around the conditional mode and so
//! inherits the bias of a first-order approximation — NPDE and NPD are built
//! entirely from Monte-Carlo simulation under the fitted population model. They
//! are therefore robust to model nonlinearity and to non-Gaussian random
//! effects (Brendel et al. 2006; Comets et al. 2008, the `npde` R package).
//!
//! For each observation `y_ij` we simulate `K` replicates under the fitted
//! `θ/Ω/Σ` (sampling `η ~ N(0, Ω)` and residual `ε`), evaluated at the subject's
//! own observation design:
//!
//! - **NPD** (no decorrelation): `pd_ij = F̂_sim(y_ij)` is the empirical CDF of
//!   the simulated values at observation `ij`; `npd_ij = Φ⁻¹(pd_ij)`.
//! - **NPDE** (decorrelated): within each subject, the observed and simulated
//!   vectors are decorrelated with the empirical mean and Cholesky factor of the
//!   simulated covariance (the Brendel/Comets procedure) *before* the empirical
//!   CDF and inverse-normal transform.
//!
//! Empirical-CDF probabilities that land at 0 or 1 are clamped to
//! `[1/(2K), 1 − 1/(2K)]` before `Φ⁻¹`, per the `npde`-package convention, so the
//! transformed scores stay finite.
//!
//! ## Censored / degenerate observations
//!
//! Censored (`CENS != 0`) observations are emitted as `NaN`, mirroring the
//! IWRES/CWRES convention (see `compute_subject_results`): their `DV` carries the
//! LLOQ under M3, not a real measurement, so an empirical-CDF score would be
//! meaningless. NPD is masked per row; for NPDE the entire subject is `NaN` when
//! it has any censored row, because the within-subject decorrelation would
//! otherwise mix the LLOQ value into the *un*censored rows' scores. M3/BLQ needs
//! the predictive-CDF variant and is out of scope here (issue #260).
//!
//! NPDE also requires `K > n_obs` replicates per subject for a full-rank
//! simulated covariance; with too few replicates the covariance is singular and
//! NPDE is `NaN` (NPD is still computed). IOV (`kappa`) models reuse the
//! simulation path's convention of holding kappas at zero, so NPDE/NPD on IOV
//! models omit the inter-occasion component of the predictive variance.

use crate::api::model_preds;
use crate::stats::special::normal_inv_cdf;
use crate::types::{CompiledModel, ModelParameters, Population};
use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Seed used when the caller leaves `npde_seed` unset, so the diagnostic is
/// reproducible across invocations.
const DEFAULT_NPDE_SEED: u64 = 42;

/// The seed actually used given the optional `[fit_options] npde_seed` override:
/// the explicit value when set, otherwise the built-in default. Recording this
/// resolved value (rather than the `Option`) is what lets a run be reproduced
/// from the fit output alone.
pub fn effective_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or(DEFAULT_NPDE_SEED)
}

/// Per-subject NPDE/NPD vectors, each parallel to the subject's observation list.
#[derive(Debug, Clone)]
pub struct SubjectNpde {
    /// Normalized prediction discrepancies (no decorrelation). `NaN` on censored
    /// observations.
    pub npd: Vec<f64>,
    /// Normalized prediction distribution errors (decorrelated within subject).
    /// `NaN` for the whole subject when the simulated covariance is rank-deficient
    /// (`K <= n_obs`) or when the subject has any censored observation; `npd` is
    /// still finite on the uncensored rows there.
    pub npde: Vec<f64>,
}

/// Compute NPDE and NPD for every subject by Monte-Carlo simulation under the
/// fitted parameters `params`. `nsim` is the number of replicates per subject
/// (`K`); `seed` makes the draw reproducible.
///
/// Subjects are simulated in parallel, each with its own RNG seeded from
/// `seed + subject_index`, so the result is independent of the rayon schedule
/// and reproducible for a fixed `seed`.
pub fn compute_npde_npd(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    nsim: usize,
    seed: Option<u64>,
) -> Vec<SubjectNpde> {
    let base_seed = effective_seed(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            let n_obs = subject.observations.len();

            // sims[(j, k)] — observation j, replicate k. Column-per-replicate
            // layout so the covariance is one D·Dᵀ gemm and decorrelation is one
            // batched triangular solve.
            let mut sims = DMatrix::<f64>::zeros(n_obs, nsim);
            for k in 0..nsim {
                // η ~ N(0, Ω) via the Cholesky factor; pad zero kappas for IOV.
                let z: Vec<f64> = (0..n_eta).map(|_| normal.sample(&mut rng)).collect();
                let eta = &params.omega.chol * DVector::from_column_slice(&z);
                let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
                eta_slice.resize(n_eta + model.n_kappa, 0.0);

                let pk = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates);
                let ipreds = model_preds(model, subject, &pk, &params.theta, &eta_slice);

                for (j, &ip) in ipreds.iter().enumerate() {
                    let var =
                        model.residual_variance_at(subject.obs_cmts[j], ip, &params.sigma.values);
                    let eps: f64 = normal.sample(&mut rng);
                    sims[(j, k)] = ip + var.sqrt() * eps;
                }
            }

            let npd = npd_scores(&subject.observations, &subject.cens, &sims);
            let npde = npde_scores(&subject.observations, &subject.cens, &sims);
            SubjectNpde { npd, npde }
        })
        .collect()
}

/// Per-observation empirical-CDF normal scores, without decorrelation. Censored
/// rows and rows with a non-finite observed value yield `NaN`.
fn npd_scores(observed: &[f64], cens: &[u8], sims: &DMatrix<f64>) -> Vec<f64> {
    (0..observed.len())
        .map(|j| {
            if cens.get(j).copied().unwrap_or(0) != 0 {
                return f64::NAN;
            }
            empirical_score(observed[j], sims.row(j).iter().copied())
        })
        .collect()
}

/// Per-observation empirical-CDF normal scores after decorrelating the observed
/// and simulated vectors with the empirical mean and Cholesky factor of the
/// simulated covariance. Returns an all-`NaN` vector when the covariance is
/// rank-deficient (`K <= n_obs`), when it stays non-PD after jitter, or when the
/// subject has any censored observation (decorrelation would mix the LLOQ into
/// the uncensored rows).
fn npde_scores(observed: &[f64], cens: &[u8], sims: &DMatrix<f64>) -> Vec<f64> {
    let n = observed.len();
    let k = sims.ncols();
    if n == 0 {
        return Vec::new();
    }
    // Need K > n_obs for a full-rank empirical covariance; censoring invalidates
    // the within-subject decorrelation entirely.
    if k <= n || cens.iter().any(|&c| c != 0) {
        return vec![f64::NAN; n];
    }

    // Empirical mean (per observation = per row).
    let mean: DVector<f64> = sims.column_sum() / k as f64;

    // Centered replicates and the empirical covariance via a single gemm:
    // cov = C·Cᵀ / (K-1), where C is the column-centered n×K matrix. The K vs K-1
    // divisor only scales the decorrelation matrix uniformly, which leaves the
    // within-dimension ranking — and hence the NPDE — unchanged.
    let mut centered = sims.clone();
    for mut col in centered.column_iter_mut() {
        col -= &mean;
    }
    let mut cov = &centered * centered.transpose() / (k - 1) as f64;

    // Cholesky L of the covariance; retry once with a small diagonal jitter if it
    // is only numerically (not structurally — K > n is guaranteed above) non-PD.
    let chol = nalgebra::Cholesky::new(cov.clone()).or_else(|| {
        let mean_diag = (0..n).map(|j| cov[(j, j)]).sum::<f64>() / n as f64;
        let jitter = if mean_diag > 0.0 {
            mean_diag * 1e-6
        } else {
            1e-12
        };
        for j in 0..n {
            cov[(j, j)] += jitter;
        }
        nalgebra::Cholesky::new(cov)
    });
    let l = match chol {
        Some(c) => c.l(),
        None => return vec![f64::NAN; n],
    };

    // Decorrelate via forward substitution: w = L⁻¹ (x − mean). One batched solve
    // for all replicates, one for the observed vector. The Cholesky factor `l` is
    // non-singular (positive diagonal) by construction, so the triangular solve
    // never returns `None`.
    let solve = |b: &DMatrix<f64>| {
        l.solve_lower_triangular(b)
            .expect("Cholesky factor is non-singular, so the triangular solve cannot fail")
    };
    let sims_d = solve(&centered);
    let obs_centered =
        DMatrix::from_iterator(n, 1, observed.iter().zip(mean.iter()).map(|(y, m)| y - m));
    let obs_d = solve(&obs_centered);

    (0..n)
        .map(|j| empirical_score(obs_d[j], sims_d.row(j).iter().copied()))
        .collect()
}

/// Empirical-CDF normal score of `y` against the simulated values `sims`:
/// `Φ⁻¹` of the clamped proportion of (finite) simulated values below `y`.
/// Returns `NaN` when `y` or all simulated values are non-finite.
fn empirical_score(y: f64, sims: impl Iterator<Item = f64>) -> f64 {
    if !y.is_finite() {
        return f64::NAN;
    }
    let mut n_less = 0usize;
    let mut n_equal = 0usize;
    let mut n_finite = 0usize;
    for v in sims {
        if !v.is_finite() {
            continue;
        }
        n_finite += 1;
        if v < y {
            n_less += 1;
        } else if v == y {
            n_equal += 1;
        }
    }
    if n_finite == 0 {
        return f64::NAN;
    }
    // Mid-rank for exact ties (rare with continuous simulations).
    let pd = (n_less as f64 + 0.5 * n_equal as f64) / n_finite as f64;
    normal_inv_cdf(clamp_prob(pd, n_finite))
}

/// Clamp an empirical-CDF probability away from 0 and 1 to `[1/(2K), 1 − 1/(2K)]`
/// so the inverse-normal transform stays finite (npde-package convention).
fn clamp_prob(p: f64, k: usize) -> f64 {
    let lo = 1.0 / (2.0 * k as f64);
    let hi = 1.0 - lo;
    p.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Build an n_obs×k simulation matrix from a `[replicate][obs]` slice.
    fn sims_matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
        let k = rows.len();
        let n = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut m = DMatrix::zeros(n, k);
        for (c, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                m[(j, c)] = v;
            }
        }
        m
    }

    #[test]
    fn clamp_prob_keeps_interior_and_clamps_edges() {
        assert_eq!(clamp_prob(0.0, 100), 1.0 / 200.0);
        assert_eq!(clamp_prob(1.0, 100), 1.0 - 1.0 / 200.0);
        assert_eq!(clamp_prob(0.5, 100), 0.5);
    }

    #[test]
    fn npd_scores_median_is_zero() {
        // Observed equals the simulated median → pd = 0.5 → Φ⁻¹(0.5) = 0.
        let sims = sims_matrix(&(0..=100).map(|v| vec![v as f64]).collect::<Vec<_>>());
        let scores = npd_scores(&[50.0], &[0], &sims);
        assert_relative_eq!(scores[0], 0.0, epsilon = 0.02);
    }

    #[test]
    fn npd_scores_clamps_below_all_sims() {
        // Observed below every simulated value → pd = 0 → clamped, finite, negative.
        let sims = sims_matrix(&(1..=100).map(|v| vec![v as f64]).collect::<Vec<_>>());
        let scores = npd_scores(&[-10.0], &[0], &sims);
        assert!(scores[0].is_finite());
        assert!(scores[0] < 0.0);
        // pd clamped to 1/200 → Φ⁻¹(0.005) ≈ -2.576.
        assert_relative_eq!(scores[0], normal_inv_cdf(0.005), epsilon = 1e-9);
    }

    #[test]
    fn npd_scores_nan_on_censored_row() {
        // Two observations, second censored → its NPD is NaN, the first is finite.
        let sims = sims_matrix(
            &(0..=100)
                .map(|v| vec![v as f64, v as f64])
                .collect::<Vec<_>>(),
        );
        let scores = npd_scores(&[50.0, 50.0], &[0, 1], &sims);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_nan());
    }

    #[test]
    fn effective_seed_resolves_default_and_override() {
        // Unset falls back to the built-in default; an explicit value passes through.
        assert_eq!(effective_seed(None), DEFAULT_NPDE_SEED);
        assert_eq!(effective_seed(Some(20240601)), 20240601);
    }

    #[test]
    fn empirical_score_nan_on_nonfinite_observed() {
        assert!(empirical_score(f64::NAN, [1.0, 2.0, 3.0].into_iter()).is_nan());
    }

    #[test]
    fn empirical_score_skips_nonfinite_sims() {
        // One NaN replicate is ignored; the score reflects only the finite ones.
        let s = empirical_score(2.5, [1.0, 2.0, f64::NAN, 3.0, 4.0].into_iter());
        assert!(s.is_finite());
        // 2 of 4 finite sims below 2.5 → pd = 0.5 → 0.
        assert_relative_eq!(s, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn empirical_score_nan_when_all_sims_nonfinite() {
        assert!(empirical_score(1.0, [f64::NAN, f64::INFINITY].into_iter()).is_nan());
    }

    #[test]
    fn npde_scores_identity_when_independent_unit_variance() {
        // K replicates of a 2-vector with independent ~N(0,1) columns and near-zero
        // mean: decorrelation is ≈ identity, so decorrelated and raw scores agree.
        let k = 400;
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let a = normal_inv_cdf((i as f64 + 0.5) / k as f64);
                let b = normal_inv_cdf(((i * 7 % k) as f64 + 0.5) / k as f64);
                vec![a, b]
            })
            .collect();
        let sims = sims_matrix(&rows);
        let observed = [0.3, -0.4];
        let raw = npd_scores(&observed, &[0, 0], &sims);
        let dec = npde_scores(&observed, &[0, 0], &sims);
        assert!(dec.iter().all(|v| v.is_finite()));
        assert_relative_eq!(dec[0], raw[0], epsilon = 0.15);
        assert_relative_eq!(dec[1], raw[1], epsilon = 0.15);
    }

    #[test]
    fn npde_scores_nan_when_rank_deficient() {
        // K = 2 replicates but n_obs = 2 → K <= n_obs → singular covariance → NaN.
        let sims = sims_matrix(&[vec![1.0, 2.0], vec![1.5, 2.5]]);
        let out = npde_scores(&[1.0, 2.0], &[0, 0], &sims);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn npde_scores_empty_for_zero_observations() {
        let sims = DMatrix::<f64>::zeros(0, 10);
        assert!(npde_scores(&[], &[], &sims).is_empty());
    }

    #[test]
    fn npde_scores_jitter_rescues_zero_variance_row() {
        // K > n_obs so the rank guard passes, but observation 0 is constant across
        // replicates → cov[0,0] = 0 → covariance is non-PD → the jitter retry must
        // rescue the Cholesky and still yield finite scores.
        let k = 20;
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|i| vec![5.0, i as f64]) // row 0 constant, row 1 varies
            .collect();
        let sims = sims_matrix(&rows);
        let out = npde_scores(&[5.0, 10.0], &[0, 0], &sims);
        assert_eq!(out.len(), 2);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "jitter path must yield finite NPDE, got {out:?}"
        );
    }

    #[test]
    fn npde_scores_nan_when_covariance_is_nan() {
        // A non-finite simulated value makes the covariance non-finite; Cholesky
        // fails even after jitter, so the whole subject's NPDE is NaN.
        let k = 10;
        let mut rows: Vec<Vec<f64>> = (0..k).map(|i| vec![i as f64, (k - i) as f64]).collect();
        rows[3][0] = f64::NAN;
        let sims = sims_matrix(&rows);
        let out = npde_scores(&[1.0, 2.0], &[0, 0], &sims);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn npde_scores_nan_for_subject_with_any_censoring() {
        let k = 50;
        let rows: Vec<Vec<f64>> = (0..k).map(|i| vec![i as f64, (k - i) as f64]).collect();
        let sims = sims_matrix(&rows);
        // Second row censored → whole subject's NPDE is NaN (decorrelation invalid).
        let out = npde_scores(&[10.0, 20.0], &[0, 1], &sims);
        assert!(out.iter().all(|v| v.is_nan()));
    }
}
