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
//! NPDE is `NaN` (NPD is still computed).
//!
//! ## Inter-occasion variability (IOV)
//!
//! For an IOV (`kappa`) model the reference distribution draws one independent
//! `κ ~ N(0, Ω_IOV)` **per occasion** and routes the replicate through
//! [`crate::pk::predict_iov`], mirroring what `simulate()` does (#723 / #734).
//! Holding every κ at zero — the pre-#734 behaviour — left the reference
//! distribution without its inter-occasion component, so the scores came out
//! over-dispersed (the diagnostic understated its own spread) for exactly the
//! models IOV was declared on. Non-IOV models keep the unchanged fast path and
//! draw no extra randoms, so their scores are byte-identical.

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
///
/// For an IOV model the reference distribution draws one independent occasion
/// `κ ~ N(0, Ω_IOV)` per occasion group (#734). That needs `params.omega_iov`,
/// which every fitted or parsed `ModelParameters` of a `kappa` model carries; a
/// caller that *rebuilds* the parameters and drops it (the #1019 failure mode on
/// the R bridge) gets the κ = 0 reference instead of a panic — a post-fit
/// diagnostic is the wrong place to abort a completed fit. Use
/// `fitted_params_from_result` to keep the IOV block.
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
            // Custom residual magnitude (#484): η-independent (θ/covariate/TIME
            // only), so build the per-observation multiplier matrix once and
            // reuse it across all replicates.
            let ruv_mult = model.ruv_obs_mult(subject, &params.theta);
            // IOV (#734): the per-occasion κ draw needs the subject's occasion
            // groups in the exact order `predict_iov` indexes its `kappas`
            // argument by. A function of the subject alone, so build it once and
            // reuse across replicates. Empty when the subject carries no
            // occasion labels, in which case `predict_iov` falls back to κ = 0
            // (matching the fit-time no-occasion diagnostic).
            let iov: Option<(&crate::types::OmegaMatrix, Vec<(u32, Vec<usize>)>)> = params
                .omega_iov
                .as_ref()
                .filter(|_| model.n_kappa > 0)
                .map(|om| (om, crate::stats::likelihood::iov_occasion_groups(subject)));
            for k in 0..nsim {
                // η ~ N(0, Ω) via the Cholesky factor; pad zero kappas for IOV.
                let z: Vec<f64> = (0..n_eta).map(|_| normal.sample(&mut rng)).collect();
                let eta = &params.omega.chol * DVector::from_column_slice(&z);
                let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
                eta_slice.resize(n_eta + model.n_kappa, 0.0);

                // IOV models (#734): one independent κ ~ N(0, Ω_IOV) per
                // occasion through the occasion-aware `predict_iov`, mirroring
                // `simulate()`'s `emit_subject_rows` (#723). Holding κ at zero
                // would leave the reference distribution without its
                // inter-occasion component and over-disperse every score.
                // Non-IOV models take the TV-covariate-aware dispatcher
                // unchanged — matching simulate()/predict() (#506): a per-event
                // covariate snapshot must drive NPDE IPREDs, not the
                // baseline-only `pk_param_fn(subject.covariates)` — and draw no
                // extra randoms, so their sims are byte-identical.
                let ipreds = match &iov {
                    Some((omega_iov, occ_groups)) => {
                        let kappas: Vec<Vec<f64>> = (0..occ_groups.len())
                            .map(|_| {
                                let z: Vec<f64> = (0..model.n_kappa)
                                    .map(|_| normal.sample(&mut rng))
                                    .collect();
                                (&omega_iov.chol * DVector::from_column_slice(&z))
                                    .iter()
                                    .copied()
                                    .collect()
                            })
                            .collect();
                        crate::pk::predict_iov(
                            model,
                            subject,
                            &params.theta,
                            &eta_slice[..n_eta],
                            &kappas,
                        )
                    }
                    None => crate::pk::compute_predictions_with_tv(
                        model,
                        subject,
                        &params.theta,
                        &eta_slice,
                    ),
                };

                // IIV on residual error (#409): the simulated eta draw includes
                // the residual-error eta, so scale the residual variance by
                // exp(2·η_ruv) — i.e. simulate `Y = IPRED + EPS·EXP(η_ruv)`.
                let ruv_scale = model.residual_var_scale(&eta_slice);
                for (j, &ip) in ipreds.iter().enumerate() {
                    let var = model.sim_residual_variance(
                        subject,
                        j,
                        ip,
                        &params.sigma.values,
                        ruv_scale,
                        ruv_mult.as_ref().map(|m| m[j].as_slice()),
                    );
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
fn npd_scores(observed: &[f64], cens: &[i8], sims: &DMatrix<f64>) -> Vec<f64> {
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
fn npde_scores(observed: &[f64], cens: &[i8], sims: &DMatrix<f64>) -> Vec<f64> {
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

    /// Regression for #506: NPDE/NPD must simulate against the time-varying
    /// covariate snapshots, like `simulate()`/`predict()`, not the baseline-only
    /// `pk_param_fn(subject.covariates)`. Each observation is placed exactly on
    /// the TV-aware IPRED, so its replicate sims center on it and NPD ≈ 0 on
    /// every row. If NPDE regressed to the baseline-only predictor, the WT
    /// 140/210 rows would sit several residual-SDs from the WT=70 sims and NPD
    /// would blow up.
    #[test]
    fn compute_npde_honours_tv_covariates() {
        let (model, mut subject) = crate::types::test_helpers::tv_cov_iv_model_and_subject();
        let params = model.default_params.clone();
        let tv_ipred = crate::pk::compute_predictions_with_tv(&model, &subject, &params.theta, &[]);
        subject.observations = tv_ipred.clone();
        let population = Population {
            subjects: vec![subject],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let out = compute_npde_npd(&model, &population, &params, 1000, Some(506));
        assert_eq!(out.len(), 1);
        for (j, &npd) in out[0].npd.iter().enumerate() {
            assert!(
                npd.abs() < 0.25,
                "row {j}: NPD={npd} not ≈0 — TV covariate snapshot ignored?"
            );
        }
    }

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

    /// Log-scale SD of the one-row reference distribution in
    /// [`iov_npde_model_and_population`] when the occasion κ is sampled:
    /// `sqrt(ω²_V + ω²_κ) = sqrt(0.01 + 0.04)`. Holding κ at zero (the pre-#734
    /// behaviour) leaves `sqrt(ω²_V) = 0.1`.
    const IOV_REF_SD: f64 = 0.223_606_797_749_979;
    /// The κ-less spread the same fixture collapses to.
    const BSV_ONLY_SD: f64 = 0.1;

    /// One-subject IOV fixture for the κ-draw tests below.
    ///
    /// A bolus at `t = 0` with a single observation at the same time, so the
    /// prediction is exactly `AMT / V` (ferx applies a dose before an
    /// observation at equal TIME) and
    /// `log IPRED = log(AMT/TVV) − η_V − κ_V`. The reference distribution of
    /// that row is therefore log-normal about `log(10)` with SD
    /// [`IOV_REF_SD`] — closed form, not a second engine — plus a deliberately
    /// negligible residual (`σ = 1e-4` proportional). The observation is placed
    /// exactly `IOV_REF_SD` above the median on the log scale, so a correct
    /// reference gives `NPD ≈ Φ⁻¹(Φ(1)) = 1` and the *implied* spread
    /// `IOV_REF_SD / NPD` is directly comparable to the truth.
    fn iov_npde_model_and_population() -> (CompiledModel, Population) {
        use crate::types::{DoseEvent, Subject};
        let model = crate::parser::model_parser::parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_V ~ 0.01
  kappa KAPPA_V ~ 0.04
  sigma PROP ~ 0.0001 (sd)

[individual_parameters]
  CL = TVCL
  V  = TVV * exp(ETA_V + KAPPA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
",
        )
        .expect("IOV npde fixture parses");
        assert_eq!(model.n_eta, 1, "one BSV eta");
        assert_eq!(model.n_kappa, 1, "one occasion kappa");

        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.0],
            observations: vec![10.0 * IOV_REF_SD.exp()],
            obs_cmts: vec![1],
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            ..Default::default()
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        (model, population)
    }

    /// The log-scale spread the NPD score implies for the fixture's single row:
    /// the observation sits `IOV_REF_SD` above the reference median, so
    /// `NPD ≈ IOV_REF_SD / sd` and `sd ≈ IOV_REF_SD / NPD`.
    fn implied_reference_sd(out: &[SubjectNpde]) -> f64 {
        assert_eq!(out.len(), 1);
        let npd = out[0].npd[0];
        assert!(npd.is_finite(), "NPD must be finite, got {npd}");
        IOV_REF_SD / npd
    }

    /// **Regression for #734: the NPDE/NPD reference distribution must sample the
    /// occasion κ.**
    ///
    /// `compute_npde_npd` built its reference with every κ held at zero, so for an
    /// IOV model the reference carried no inter-occasion component and the scores
    /// came out over-dispersed — the diagnostic understating its own spread, the
    /// same bug class #723 fixed one path over in `simulate()`.
    ///
    /// The pair straddles the fix. Both arms run the *same* RNG stream (the κ
    /// draws happen either way; the control merely scales them by a zero
    /// `Ω_IOV`), so the only difference is the κ magnitude. Under the old code
    /// both arms return the κ-less spread and the first assertion fails; the
    /// control arm is asserted too, so the test cannot silently become a
    /// tautology if the reference stops depending on `Ω_IOV`.
    ///
    /// Tolerances are measured, not argued. At `K = 20_000`, swept over six
    /// seeds, the realised implied SDs span 0.22149–0.22644 (truth 0.223607;
    /// worst deviation 1.3%) and 0.09857–0.10084 (truth 0.1; worst 1.4%) — the
    /// Monte-Carlo SE of the score is ≈1% of the implied SD, and this fixture's
    /// seed (734) lands at 0.22259 / 0.09871. The 3% bands below are ≈2× the
    /// worst realised deviation, tight enough that a dropped κ (a 2.2× error on
    /// the first arm) or a wrong-scale draw cannot pass either way.
    #[test]
    fn npde_reference_spread_includes_occasion_kappa() {
        let (model, population) = iov_npde_model_and_population();
        let params = model.default_params.clone();
        assert!(
            params.omega_iov.is_some(),
            "fixture must carry a fitted Ω_IOV"
        );
        let nsim = 20_000;
        let seed = Some(734);

        let with_iov = compute_npde_npd(&model, &population, &params, nsim, seed);
        let sd_with = implied_reference_sd(&with_iov);

        // Control: Ω_IOV forced to zero. The per-occasion κ draws still happen
        // (RNG stays aligned), but scale to zero — the pre-#734 reference.
        let mut zero_iov = params.clone();
        {
            let om = zero_iov.omega_iov.as_mut().expect("Ω_IOV present");
            om.chol.fill(0.0);
            om.matrix.fill(0.0);
        }
        let without_iov = compute_npde_npd(&model, &population, &zero_iov, nsim, seed);
        let sd_without = implied_reference_sd(&without_iov);

        eprintln!(
            "#734 implied reference SD: with Ω_IOV = {sd_with:.5} (target {IOV_REF_SD:.5}), \
             zero Ω_IOV = {sd_without:.5} (target {BSV_ONLY_SD:.5})"
        );
        assert!(
            (sd_with / IOV_REF_SD - 1.0).abs() < 0.03,
            "NPDE reference spread {sd_with:.5} does not recover sqrt(ω²_V + ω²_κ) = \
             {IOV_REF_SD:.5} — occasion κ dropped from the reference distribution?"
        );
        // The straddle: the same fixture, with Ω_IOV = 0, must land on the
        // κ-less spread. If this drifts up to IOV_REF_SD the arms no longer
        // straddle the fix and the assertion above proves nothing.
        assert!(
            (sd_without / BSV_ONLY_SD - 1.0).abs() < 0.03,
            "zero-Ω_IOV control spread {sd_without:.5} is not the κ-less \
             {BSV_ONLY_SD:.5} — the two arms no longer straddle the #734 fix"
        );
    }

    /// A caller-rebuilt `ModelParameters` that drops the IOV block (the #1019
    /// failure mode on the R bridge) must not panic the diagnostic: the
    /// reference falls back to κ = 0 — documented on `compute_npde_npd`, and the
    /// same spread the zero-`Ω_IOV` control above produces, reached by the other
    /// branch (no κ draws at all, so the RNG stream differs).
    #[test]
    fn npde_without_omega_iov_falls_back_to_zero_kappa() {
        let (model, population) = iov_npde_model_and_population();
        let mut params = model.default_params.clone();
        params.omega_iov = None;
        let out = compute_npde_npd(&model, &population, &params, 20_000, Some(734));
        let sd = implied_reference_sd(&out);
        assert!(
            (sd / BSV_ONLY_SD - 1.0).abs() < 0.03,
            "missing Ω_IOV must fall back to κ = 0 (spread {BSV_ONLY_SD:.5}), got {sd:.5}"
        );
    }

    /// An IOV model whose data carries **no occasion labels** routes through
    /// `predict_iov` with an empty `kappas` (κ = 0 everywhere) instead of the
    /// non-IOV dispatcher it used before #734. That swap must not change the
    /// prediction: `predict_iov` applied the divisive `[scaling]` block inside
    /// its per-occasion loop and so returned *unscaled* predictions on
    /// occasion-less data until #723's review caught it. The reference here is
    /// the κ-less spread, and it is reached over an unlabelled subject — the
    /// same path that bug lived on.
    #[test]
    fn npde_iov_without_occasion_labels_matches_the_kappa_less_reference() {
        let (model, mut population) = iov_npde_model_and_population();
        population.subjects[0].occasions.clear();
        population.subjects[0].dose_occasions.clear();
        let out = compute_npde_npd(
            &model,
            &population,
            &model.default_params,
            20_000,
            Some(734),
        );
        let sd = implied_reference_sd(&out);
        assert!(
            (sd / BSV_ONLY_SD - 1.0).abs() < 0.03,
            "an occasion-less IOV subject must score against the κ-less reference \
             (spread {BSV_ONLY_SD:.5}), got {sd:.5} — a prediction-scale regression on \
             `predict_iov`'s empty-occasion path would land here"
        );
    }
}
