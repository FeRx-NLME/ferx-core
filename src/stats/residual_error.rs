use crate::types::{ErrorModel, ErrorSpec, ResidualCorrelation, SubjectResult};
use nalgebra::DMatrix;

const MIN_VARIANCE: f64 = 1e-12;

/// Compute residual variance for a single observation
/// sigma_values: [sigma1] for additive/proportional, [sigma1, sigma2] for combined
pub fn residual_variance(error_model: ErrorModel, f_pred: f64, sigma_values: &[f64]) -> f64 {
    let v = match error_model {
        ErrorModel::Additive => {
            // V = sigma1^2
            sigma_values[0] * sigma_values[0]
        }
        ErrorModel::Proportional => {
            // V = (f * sigma1)^2
            let fs = f_pred * sigma_values[0];
            fs * fs
        }
        ErrorModel::Combined => {
            // V = (f * sigma1)^2 + sigma2^2
            let prop = f_pred * sigma_values[0];
            prop * prop + sigma_values[1] * sigma_values[1]
        }
    };
    v.max(MIN_VARIANCE)
}

/// Compute the R diagonal (vector of residual variances for all observations),
/// dispatching the error model per observation by compartment. `obs_cmts` is
/// parallel to `ipreds` (`subject.obs_cmts`); for single-endpoint models the
/// CMT is ignored.
pub fn compute_r_diag(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
) -> Vec<f64> {
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| error_spec.variance_at(cmt, f, sigma_values))
        .collect()
}

/// Compute residual variances including fixed residual correlations from
/// `block_sigma`.
pub fn compute_r_diag_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> Vec<f64> {
    if correlations.is_empty() {
        return compute_r_diag(error_spec, ipreds, obs_cmts, sigma_values);
    }
    ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| {
            error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations)
        })
        .collect()
}

fn observation_time_key(obs_times: &[f64], j: usize) -> u64 {
    obs_times.get(j).copied().unwrap_or(0.0).to_bits()
}

fn observation_occasion_key(occasions: &[u32], j: usize) -> u32 {
    occasions.get(j).copied().unwrap_or(0)
}

fn same_residual_block(
    obs_times: &[f64],
    _obs_raw_times: &[f64],
    occasions: &[u32],
    j: usize,
    k: usize,
) -> bool {
    observation_time_key(obs_times, j) == observation_time_key(obs_times, k)
        && observation_occasion_key(occasions, j) == observation_occasion_key(occasions, k)
}

fn cross_observation_covariance(
    load_j: &[(usize, f64)],
    load_k: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let mut cov = 0.0;
    for corr in correlations {
        let j_has_i = load_j.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let j_has_j = load_j.iter().any(|(idx, _)| *idx == corr.sigma_j);
        let k_has_i = load_k.iter().any(|(idx, _)| *idx == corr.sigma_i);
        let k_has_j = load_k.iter().any(|(idx, _)| *idx == corr.sigma_j);
        if (j_has_i && j_has_j) || (k_has_i && k_has_j) {
            continue;
        }
        let Some(&si) = sigma_values.get(corr.sigma_i) else {
            return f64::NAN;
        };
        let Some(&sj) = sigma_values.get(corr.sigma_j) else {
            return f64::NAN;
        };
        let cov_ij = corr.rho * si * sj;
        let ci_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        let cj_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        if let (Some(ci), Some(cj)) = (ci_j, cj_k) {
            cov += ci * cj * cov_ij;
        }

        let cj_j = load_j
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_j)
            .map(|(_, coeff)| *coeff);
        let ci_k = load_k
            .iter()
            .find(|(idx, _)| *idx == corr.sigma_i)
            .map(|(_, coeff)| *coeff);
        if let (Some(cj), Some(ci)) = (cj_j, ci_k) {
            cov += cj * ci * cov_ij;
        }
    }
    cov
}

/// Build the subject-level residual covariance matrix `R`.
///
/// The diagonal is the existing per-observation residual variance, including
/// within-observation `block_sigma` terms for `combined(...)` endpoints. When a
/// `block_sigma` off-diagonal connects sigmas used by different endpoint rows,
/// rows at the same subject time and occasion receive the corresponding
/// cross-observation covariance. This mirrors NONMEM-style paired endpoint
/// records such as total/unbound assays written as separate rows.
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> DMatrix<f64> {
    // NOTE: deliberately NOT delegated to `_scaled` with an empty multiplier.
    // The diagonal here goes through `compute_r_diag` → `residual_variance`,
    // which forms the proportional variance as `(f·σ)·(f·σ)`; `variance_at_scaled`
    // (the `_scaled` diagonal) forms it as `((f·f)·σ)·σ`. The two are equal in
    // exact arithmetic but differ by ~1 ULP under IEEE-754 reassociation on the
    // f-dependent term (~55% of proportional/combined rows), so delegating would
    // silently shift the bare-sigma R — and every OFV/CWRES built on it — off
    // its current bit-for-bit value. Keep the legacy diagonal form here. (The
    // magnitude path already uses the `_scaled` association by construction.)
    let n = ipreds.len();
    let mut r = DMatrix::<f64>::zeros(n, n);
    // Write the diagonal variance directly into `r` (mirrors
    // `compute_r_diag_with_correlations`) rather than building a throwaway Vec
    // and copying it in — this runs once per subject per FOCE inner/outer
    // iteration. The empty-correlation case delegates to `variance_at`, the
    // legacy `(f·σ)·(f·σ)` association the comment above relies on.
    if correlations.is_empty() {
        for (j, (&f, &cmt)) in ipreds.iter().zip(obs_cmts.iter()).enumerate() {
            r[(j, j)] = error_spec.variance_at(cmt, f, sigma_values);
        }
        return r;
    }
    for (j, (&f, &cmt)) in ipreds.iter().zip(obs_cmts.iter()).enumerate() {
        r[(j, j)] = error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations);
    }

    let loadings: Vec<Vec<(usize, f64)>> = ipreds
        .iter()
        .zip(obs_cmts.iter())
        .map(|(&f, &cmt)| error_spec.sigma_loadings(cmt, f, sigma_values.len()))
        .collect();
    for j in 0..n {
        if loadings[j].is_empty() {
            continue;
        }
        for k in (j + 1)..n {
            if loadings[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, j, k)
            {
                continue;
            }
            let cov = cross_observation_covariance(
                &loadings[j],
                &loadings[k],
                sigma_values,
                correlations,
            );
            if cov != 0.0 {
                r[(j, k)] = cov;
                r[(k, j)] = cov;
            }
        }
    }
    r
}

/// Build the subject residual covariance matrix `R` with a per-observation
/// custom magnitude (#484). `mult` is the `[obs][sigma-slot]` multiplier matrix
/// from [`crate::types::RuvMagnitude::eval_obs`]; each observation's sigma
/// loadings are scaled by its row before forming the diagonal variance and any
/// `block_sigma` cross-covariance. NOTE: a `mult` whose rows are all ones does
/// **not** reproduce [`compute_r_matrix_with_correlations`] bit-for-bit. The
/// bare path forms the diagonal as `(f·σ)·(f·σ)` whereas the scaled path uses
/// `variance_at_scaled`'s reassociated `((f·f)·σ)·σ` form, which differs by
/// ~1 ULP on ~55% of proportional/combined rows. The two diagonal builders are
/// kept separate deliberately — do NOT re-collapse the bare path into
/// `_scaled(..., &[])` (that is the exact regression this revert prevents).
#[allow(clippy::too_many_arguments)]
pub fn compute_r_matrix_with_correlations_scaled(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: &[Vec<f64>],
) -> DMatrix<f64> {
    let n = ipreds.len();
    let mut r = DMatrix::<f64>::zeros(n, n);
    let ones: Vec<f64> = Vec::new();
    let row = |j: usize| -> &[f64] { mult.get(j).map(|v| v.as_slice()).unwrap_or(&ones) };
    for j in 0..n {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        r[(j, j)] = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, row(j));
    }
    if correlations.is_empty() {
        return r;
    }
    // Scale each observation's loadings by its multiplier row before computing
    // the cross-observation covariance.
    let scale_loadings = |j: usize| -> Vec<(usize, f64)> {
        let f = ipreds[j];
        let cmt = obs_cmts.get(j).copied().unwrap_or(0);
        let m = row(j);
        error_spec
            .sigma_loadings(cmt, f, sigma_values.len())
            .into_iter()
            .map(|(idx, coeff)| (idx, coeff * m.get(idx).copied().unwrap_or(1.0)))
            .collect()
    };
    let loadings: Vec<Vec<(usize, f64)>> = (0..n).map(scale_loadings).collect();
    for j in 0..n {
        if loadings[j].is_empty() {
            continue;
        }
        for k in (j + 1)..n {
            if loadings[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, j, k)
            {
                continue;
            }
            let cov = cross_observation_covariance(
                &loadings[j],
                &loadings[k],
                sigma_values,
                correlations,
            );
            if cov != 0.0 {
                r[(j, k)] = cov;
                r[(k, j)] = cov;
            }
        }
    }
    r
}

/// `∂R/∂f_m` matrices of the dense residual covariance built by
/// [`compute_r_matrix_with_correlations`] (or its `_scaled` variant) — one
/// symmetric `n×n` matrix per observation `m`.
///
/// `dr[m]` is nonzero only in row/column `m`, because every entry of `R`
/// depends on the prediction vector only through the two observations it
/// couples, and each sigma loading coefficient is affine in `f` (proportional
/// slot loads `f`, additive slot a constant). So:
///
/// * diagonal: `∂R_mm/∂f_m` — the `f`-derivative of `variance_at_scaled`,
///   including any within-observation `block_sigma` cross term;
/// * off-diagonal `(m,k)` in the same residual block: `∂R_mk/∂f_m`, which —
///   because [`cross_observation_covariance`] is bilinear in the two
///   observations' loadings — is exactly that cross-covariance evaluated with
///   observation `m`'s *slope* loadings ([`ErrorSpec::sigma_loading_slopes`])
///   in place of its value loadings.
///
/// Feeds the FOCEI interaction Hessian term
/// `B_kl = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l)` with `∂R/∂η_k = Σ_m H[m,k]·dr[m]`.
/// `mult` is the #484 per-observation magnitude matrix (`None` ⇒ all ones),
/// applied identically to the value and slope loadings so the derivative tracks
/// the magnitude-scaled `R`.
#[allow(clippy::too_many_arguments)]
pub fn compute_dr_df_matrices(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: Option<&[Vec<f64>]>,
) -> Vec<DMatrix<f64>> {
    let n = ipreds.len();
    let empty: Vec<f64> = Vec::new();
    let mrow = |j: usize| -> &[f64] {
        mult.and_then(|m| m.get(j))
            .map(|v| v.as_slice())
            .unwrap_or(&empty)
    };
    let m_at = |j: usize, idx: usize| -> f64 { mrow(j).get(idx).copied().unwrap_or(1.0) };
    let cmt_at = |j: usize| -> usize { obs_cmts.get(j).copied().unwrap_or(0) };

    // Per-observation value loadings (coeff·mult) and slope loadings
    // (∂coeff/∂f·mult). The slope loadings carry the SAME slot presence as the
    // value loadings (additive slots appear with slope 0), so the bilinear
    // cross-covariance and its within-observation skip logic behave identically.
    let vload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loadings(cmt_at(j), ipreds[j], sigma_values.len())
                .into_iter()
                .map(|(idx, c)| (idx, c * m_at(j, idx)))
                .collect()
        })
        .collect();
    let sload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loading_slopes(cmt_at(j), sigma_values.len())
                .into_iter()
                .map(|(idx, s)| (idx, s * m_at(j, idx)))
                .collect()
        })
        .collect();

    let mut out = vec![DMatrix::<f64>::zeros(n, n); n];
    for m in 0..n {
        if vload[m].is_empty() {
            continue;
        }
        out[m][(m, m)] = diag_self_deriv(&vload[m], &sload[m], sigma_values, correlations);
        for k in 0..n {
            if k == m
                || vload[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, m, k)
            {
                continue;
            }
            let d = cross_observation_covariance(&sload[m], &vload[k], sigma_values, correlations);
            out[m][(m, k)] = d;
            out[m][(k, m)] = d;
        }
    }
    out
}

/// `∂R_mm/∂f_m`: the `f`-derivative of the diagonal residual variance
/// [`ErrorSpec::variance_at_scaled`], built from observation `m`'s value
/// loadings and their `f`-slopes (both already magnitude-scaled).
///
/// `V_mm = Σ_s (c_s σ_s)² + Σ_corr 2 c_i c_j ρ σ_i σ_j` (the within-observation
/// `block_sigma` cross term), so with `c_s' = slope_s`:
/// `∂V_mm/∂f = Σ_s 2 c_s c_s' σ_s² + Σ_corr 2 ρ σ_i σ_j (c_i' c_j + c_i c_j')`.
/// The `.max(1e-12)` variance floor is treated as inactive here, matching the
/// diagonal interaction path's [`ErrorSpec::dvar_df`].
fn diag_self_deriv(
    vload: &[(usize, f64)],
    sload: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let coeff = |loads: &[(usize, f64)], slot: usize| -> Option<f64> {
        loads.iter().find(|(i, _)| *i == slot).map(|(_, c)| *c)
    };
    let sig = |idx: usize| -> f64 { sigma_values.get(idx).copied().unwrap_or(0.0) };
    let mut d = 0.0;
    for &(idx, c) in vload {
        let s = coeff(sload, idx).unwrap_or(0.0);
        let sg = sig(idx);
        d += 2.0 * c * s * sg * sg;
    }
    for corr in correlations {
        let (Some(ci), Some(cj)) = (coeff(vload, corr.sigma_i), coeff(vload, corr.sigma_j)) else {
            continue;
        };
        let si_s = coeff(sload, corr.sigma_i).unwrap_or(0.0);
        let sj_s = coeff(sload, corr.sigma_j).unwrap_or(0.0);
        d += 2.0 * corr.rho * sig(corr.sigma_i) * sig(corr.sigma_j) * (si_s * cj + ci * sj_s);
    }
    d
}

/// Second-order `∂²R/∂f_a∂f_b` tensor of the dense residual covariance — the
/// curvature companion to [`compute_dr_df_matrices`], returned as `d2r[a][b]`,
/// an `n×n` symmetric matrix per ordered prediction pair `(a, b)`
/// (`d2r[a][b] == d2r[b][a]` by equality of mixed partials).
///
/// Every sigma loading coefficient is *affine* in `f` (proportional slot loads
/// `f`, additive slot is constant), so the second `f`-derivative of any loading
/// is zero. Combined with the fact that each `R` entry couples at most two
/// observations, only two families of second derivatives survive:
///
/// * `d2r[m][m]` has a single nonzero entry at `(m, m)`:
///   `∂²R_mm/∂f_m²` — the second `f`-derivative of `variance_at_scaled`. With
///   `c_s` the value loading, `c_s' = slope_s`, `c_s'' = 0`, the within-obs
///   `block_sigma` cross term `2 ρ σ_i σ_j c_i c_j` differentiates twice to
///   `4 ρ σ_i σ_j c_i' c_j'` (see [`diag_self_second_deriv`]). The off-diagonal
///   entries of `R` have `∂²R_mk/∂f_m² = cross(c_m'', c_k) = 0`, so they do not
///   appear here.
/// * `d2r[m][k]` for `m ≠ k` in the same residual block has nonzero entries at
///   `(m, k)` and `(k, m)`: `∂²R_mk/∂f_m∂f_k`. Because
///   [`cross_observation_covariance`] is bilinear in the two observations'
///   loadings, this mixed partial is exactly that cross-covariance evaluated
///   with *both* observations' slope loadings — `cross(slope_m, slope_k)`.
///
/// Feeds the dense FOCEI outer-gradient curvature coefficients (the `β`/`α'`
/// reservoir in `sens_outer_gradient`) and the inner Hessian response
/// correction, via `∂²R/∂η_k∂η_l = Σ_{m,m'} H[m,k] H[m',l] · d2r[m][m']`
/// (plus the `Σ_m (∂²f_m/∂η_k∂η_l) · ∂R/∂f_m` term carried by the first-order
/// [`compute_dr_df_matrices`]). `mult` is the #484 per-observation magnitude
/// matrix, applied to the slope loadings identically to the first-order path.
#[allow(clippy::too_many_arguments)]
pub fn compute_d2r_df2_matrices(
    error_spec: &ErrorSpec,
    ipreds: &[f64],
    obs_cmts: &[usize],
    obs_times: &[f64],
    obs_raw_times: &[f64],
    occasions: &[u32],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    mult: Option<&[Vec<f64>]>,
) -> Vec<Vec<DMatrix<f64>>> {
    let n = ipreds.len();
    let empty: Vec<f64> = Vec::new();
    let mrow = |j: usize| -> &[f64] {
        mult.and_then(|m| m.get(j))
            .map(|v| v.as_slice())
            .unwrap_or(&empty)
    };
    let m_at = |j: usize, idx: usize| -> f64 { mrow(j).get(idx).copied().unwrap_or(1.0) };
    let cmt_at = |j: usize| -> usize { obs_cmts.get(j).copied().unwrap_or(0) };

    // Value loadings gate the emptiness skip exactly as `compute_dr_df_matrices`
    // (an observation with no sigma loadings contributes nothing); slope
    // loadings carry the math (the only nonzero second derivatives).
    let vload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loadings(cmt_at(j), ipreds[j], sigma_values.len())
                .into_iter()
                .map(|(idx, c)| (idx, c * m_at(j, idx)))
                .collect()
        })
        .collect();
    let sload: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|j| {
            error_spec
                .sigma_loading_slopes(cmt_at(j), sigma_values.len())
                .into_iter()
                .map(|(idx, s)| (idx, s * m_at(j, idx)))
                .collect()
        })
        .collect();

    let mut out = vec![vec![DMatrix::<f64>::zeros(n, n); n]; n];
    for m in 0..n {
        if vload[m].is_empty() {
            continue;
        }
        // Diagonal curvature ∂²R_mm/∂f_m².
        out[m][m][(m, m)] = diag_self_second_deriv(&sload[m], sigma_values, correlations);
        // Mixed partial ∂²R_mk/∂f_m∂f_k for the off-diagonal cross-covariance.
        for k in (m + 1)..n {
            if vload[k].is_empty()
                || !same_residual_block(obs_times, obs_raw_times, occasions, m, k)
            {
                continue;
            }
            let d = cross_observation_covariance(&sload[m], &sload[k], sigma_values, correlations);
            if d != 0.0 {
                out[m][k][(m, k)] = d;
                out[m][k][(k, m)] = d;
                out[k][m][(m, k)] = d;
                out[k][m][(k, m)] = d;
            }
        }
    }
    out
}

/// `∂²R_mm/∂f_m²`: the second `f`-derivative of the diagonal residual variance.
/// With value loadings affine in `f` (`c_s'' = 0`),
/// `∂²V_mm/∂f² = Σ_s 2 (c_s')² σ_s² + Σ_corr 4 ρ σ_i σ_j c_i' c_j'`,
/// i.e. the slope-only companion of [`diag_self_deriv`]. The slope loadings
/// carry the same slot presence as the value loadings (additive slots appear
/// with slope 0), so iterating them is sufficient.
fn diag_self_second_deriv(
    sload: &[(usize, f64)],
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
) -> f64 {
    let slope = |slot: usize| -> f64 {
        sload
            .iter()
            .find(|(i, _)| *i == slot)
            .map(|(_, s)| *s)
            .unwrap_or(0.0)
    };
    let sig = |idx: usize| -> f64 { sigma_values.get(idx).copied().unwrap_or(0.0) };
    let mut d = 0.0;
    for &(idx, s) in sload {
        let sg = sig(idx);
        d += 2.0 * s * s * sg * sg;
    }
    for corr in correlations {
        let si_s = slope(corr.sigma_i);
        let sj_s = slope(corr.sigma_j);
        d += 4.0 * corr.rho * sig(corr.sigma_i) * sig(corr.sigma_j) * si_s * sj_s;
    }
    d
}

/// Individual weighted residual: IWRES_j = (y_j - f_j) / sqrt(V_j)
pub fn iwres(obs: f64, ipred: f64, error_model: ErrorModel, sigma_values: &[f64]) -> f64 {
    let v = residual_variance(error_model, ipred, sigma_values);
    (obs - ipred) / v.sqrt()
}

/// Compute IWRES for all observations, dispatching the error model per
/// observation by compartment (`obs_cmts` parallel to `observations`/`ipreds`).
pub fn compute_iwres(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
) -> Vec<f64> {
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at(cmt, f, sigma_values);
            (y - f) / v.sqrt()
        })
        .collect()
}

/// Compute IWRES using residual variances that include fixed `block_sigma`
/// correlations. With no correlations and no custom magnitude this is exactly
/// [`compute_iwres`].
///
/// `ruv_mult` is the per-observation custom-magnitude multiplier matrix (#484)
/// from [`crate::types::CompiledModel::ruv_obs_mult`]; `None` reproduces the
/// legacy unscaled IWRES. When present, each observation's residual variance is
/// scaled by its multiplier row so the sdtab IWRES matches the magnitude-aware
/// OFV variance (otherwise late/covariate-varying rows are systematically
/// mis-scaled).
pub fn compute_iwres_with_correlations(
    observations: &[f64],
    ipreds: &[f64],
    obs_cmts: &[usize],
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
    correlations: &[ResidualCorrelation],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    if let Some(mult) = ruv_mult {
        // variance_at_scaled handles empty `correlations` (no cross terms) and a
        // short/empty multiplier row (slots default to 1.0), so this one path
        // covers correlated and uncorrelated custom-magnitude models alike.
        return observations
            .iter()
            .zip(ipreds.iter())
            .zip(obs_cmts.iter())
            .enumerate()
            .map(|(j, ((&y, &f), &cmt))| {
                let m = mult.get(j).map(|v| v.as_slice()).unwrap_or(&[]);
                let v = error_spec.variance_at_scaled(cmt, f, sigma_values, correlations, m);
                (y - f) / v.sqrt()
            })
            .collect();
    }
    if correlations.is_empty() {
        return compute_iwres(observations, ipreds, obs_cmts, error_spec, sigma_values);
    }
    observations
        .iter()
        .zip(ipreds.iter())
        .zip(obs_cmts.iter())
        .map(|((&y, &f), &cmt)| {
            let v = error_spec.variance_at_with_correlations(cmt, f, sigma_values, correlations);
            (y - f) / v.sqrt()
        })
        .collect()
}

/// Compute pooled lag-1 autocorrelation diagnostics on IWRES across subjects.
///
/// Subjects with fewer than 2 finite IWRES values are skipped.
/// Returns `(lag1_r, durbin_watson)` where DW = 2.0 indicates no autocorrelation.
/// Returns `(f64::NAN, f64::NAN)` when no subject has enough observations.
pub fn iwres_autocorrelation(subjects: &[SubjectResult]) -> (f64, f64) {
    // Accumulators for Durbin-Watson: Σ(eᵢ - eᵢ₋₁)², Σeᵢ²
    let mut dw_num = 0.0_f64;
    let mut dw_den = 0.0_f64;

    // Accumulators for pooled lag-1 Pearson r
    let mut sum_xy = 0.0_f64; // Σ e[t] * e[t+1]
    let mut sum_x = 0.0_f64; // Σ e[t]
    let mut sum_y = 0.0_f64; // Σ e[t+1]
    let mut sum_x2 = 0.0_f64; // Σ e[t]²
    let mut sum_y2 = 0.0_f64; // Σ e[t+1]²
    let mut n_pairs: usize = 0;

    for subj in subjects {
        let valid: Vec<f64> = subj
            .iwres
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if valid.len() < 2 {
            continue;
        }
        // DW accumulation
        dw_den += valid.iter().map(|e| e * e).sum::<f64>();
        for w in valid.windows(2) {
            let diff = w[1] - w[0];
            dw_num += diff * diff;
        }
        // Lag-1 Pearson accumulation
        for w in valid.windows(2) {
            let x = w[0];
            let y = w[1];
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xy += x * y;
            n_pairs += 1;
        }
    }

    if n_pairs == 0 {
        return (f64::NAN, f64::NAN);
    }

    let n = n_pairs as f64;
    let lag1_r = {
        let num = n * sum_xy - sum_x * sum_y;
        let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    };

    let dw = if dw_den == 0.0 {
        f64::NAN
    } else {
        dw_num / dw_den
    };

    (lag1_r, dw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EndpointError, GradientMethod};
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    #[test]
    fn test_additive_variance() {
        let v = residual_variance(ErrorModel::Additive, 10.0, &[0.5]);
        assert_relative_eq!(v, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_additive_variance_independent_of_prediction() {
        let v1 = residual_variance(ErrorModel::Additive, 1.0, &[0.5]);
        let v2 = residual_variance(ErrorModel::Additive, 100.0, &[0.5]);
        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance() {
        // V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0
        let v = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        assert_relative_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance_scales_with_prediction() {
        let v1 = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        let v2 = residual_variance(ErrorModel::Proportional, 20.0, &[0.1]);
        assert_relative_eq!(v2 / v1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance() {
        // V = (f * sigma1)^2 + sigma2^2 = (10 * 0.1)^2 + 0.5^2 = 1.0 + 0.25 = 1.25
        let v = residual_variance(ErrorModel::Combined, 10.0, &[0.1, 0.5]);
        assert_relative_eq!(v, 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance_with_residual_correlation() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let v = spec.variance_at_with_correlations(1, 10.0, &[0.2, 1.0], &[corr]);
        assert_relative_eq!(v, 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_empty_matches_diagonal() {
        // With no correlations the helper must be identical to compute_r_diag.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0, 20.0];
        let cmts = [0usize, 0];
        let sigma = [0.2, 1.0];
        let plain = compute_r_diag(&spec, &ipreds, &cmts, &sigma);
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[]);
        assert_eq!(plain, with);
    }

    #[test]
    fn test_compute_r_diag_with_correlations_applies_cross_term() {
        // Each observation's diagonal variance gains the 2·f·ρ·σ₁·σ₂ cross term.
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let ipreds = [10.0];
        let cmts = [0usize];
        let sigma = [0.2, 1.0];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let with = compute_r_diag_with_correlations(&spec, &ipreds, &cmts, &sigma, &[corr]);
        assert_relative_eq!(with[0], 7.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_links_paired_endpoints() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0];
        let cmts = [1usize, 2, 2];
        let times = [1.0, 1.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
        );
        assert_relative_eq!(r[(0, 0)], (50.0_f64 * 0.3).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(1, 1)], (5.0_f64 * 0.2).powi(2), epsilon = 1e-12);
        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(1, 0)], r[(0, 1)], epsilon = 1e-12);
        assert_eq!(r[(0, 2)], 0.0);
    }

    // Central-difference check: ∂R/∂f_m from `compute_dr_df_matrices` must match
    // a finite-difference perturbation of `compute_r_matrix_with_correlations` for
    // every observation, on a paired-endpoint cross-correlated model.
    #[test]
    fn test_compute_dr_df_matrices_matches_finite_difference_paired() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Combined,
                    sigma_idx: vec![0, 2],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3, 1.5];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations(&spec, f, &cmts, &times, &[], &[], &sigma, &[corr])
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // With a #484 per-observation magnitude matrix, ∂R/∂f must track the
    // *scaled* covariance `compute_r_matrix_with_correlations_scaled`.
    #[test]
    fn test_compute_dr_df_matrices_matches_finite_difference_scaled() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        // Non-trivial per-observation, per-slot multipliers.
        let mult = vec![
            vec![1.0, 1.2],
            vec![0.8, 1.0],
            vec![1.1, 0.9],
            vec![1.3, 1.0],
        ];
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
            Some(&mult),
        );
        let n = ipreds.len();
        let r_at = |f: &[f64]| {
            compute_r_matrix_with_correlations_scaled(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &sigma,
                &[corr],
                &mult,
            )
        };
        let h = 1e-4;
        for m in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[m] += h;
            fm[m] -= h;
            let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
            for p in 0..n {
                for q in 0..n {
                    assert_relative_eq!(
                        dr[m][(p, q)],
                        fd[(p, q)],
                        epsilon = 1e-5,
                        max_relative = 1e-4
                    );
                }
            }
        }
    }

    // ∂²R/∂f_a∂f_b must match a central difference of the first-order
    // ∂R/∂f machinery: d2r[a][b] ≈ (dr(f+h·e_b)[a] − dr(f−h·e_b)[a]) / 2h.
    // Mixed CMTs (proportional + combined) and a cross-endpoint block_sigma
    // correlation exercise both the diagonal-curvature and the bilinear
    // off-diagonal mixed-partial branches.
    #[test]
    fn test_compute_d2r_df2_matrices_matches_finite_difference_paired() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Combined,
                    sigma_idx: vec![0, 2],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3, 1.5];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let d2r = compute_d2r_df2_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        let n = ipreds.len();
        let dr_at = |f: &[f64]| {
            compute_dr_df_matrices(&spec, f, &cmts, &times, &[], &[], &sigma, &[corr], None)
        };
        let h = 1e-4;
        for b in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[b] += h;
            fm[b] -= h;
            let drp = dr_at(&fp);
            let drm = dr_at(&fm);
            for a in 0..n {
                let fd = (&drp[a] - &drm[a]) / (2.0 * h);
                for p in 0..n {
                    for q in 0..n {
                        assert_relative_eq!(
                            d2r[a][b][(p, q)],
                            fd[(p, q)],
                            epsilon = 1e-5,
                            max_relative = 1e-4
                        );
                    }
                }
            }
        }
    }

    // Same FD check with a #484 per-observation magnitude matrix: ∂²R/∂f²
    // must track the *scaled* covariance through `compute_dr_df_matrices`.
    #[test]
    fn test_compute_d2r_df2_matrices_matches_finite_difference_scaled() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let times = [1.0, 1.0, 2.0, 2.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.4,
        };
        let mult = vec![
            vec![1.0, 1.2],
            vec![0.8, 1.0],
            vec![1.1, 0.9],
            vec![1.3, 1.0],
        ];
        let d2r = compute_d2r_df2_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
            Some(&mult),
        );
        let n = ipreds.len();
        let dr_at = |f: &[f64]| {
            compute_dr_df_matrices(
                &spec,
                f,
                &cmts,
                &times,
                &[],
                &[],
                &sigma,
                &[corr],
                Some(&mult),
            )
        };
        let h = 1e-4;
        for b in 0..n {
            let mut fp = ipreds.to_vec();
            let mut fm = ipreds.to_vec();
            fp[b] += h;
            fm[b] -= h;
            let drp = dr_at(&fp);
            let drm = dr_at(&fm);
            for a in 0..n {
                let fd = (&drp[a] - &drm[a]) / (2.0 * h);
                for p in 0..n {
                    for q in 0..n {
                        assert_relative_eq!(
                            d2r[a][b][(p, q)],
                            fd[(p, q)],
                            epsilon = 1e-5,
                            max_relative = 1e-4
                        );
                    }
                }
            }
        }
    }

    // The diagonal self-derivative must include a *within-observation*
    // `block_sigma` cross term (combined error with σ_prop ↔ σ_add correlated),
    // which `dvar_df` alone omits.
    #[test]
    fn test_compute_dr_df_matrices_within_obs_cross_term() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let sigma = [0.3, 1.2];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        let ipreds = [40.0];
        let cmts = [1usize];
        let times = [1.0];
        let dr = compute_dr_df_matrices(
            &spec,
            &ipreds,
            &cmts,
            &times,
            &[],
            &[],
            &sigma,
            &[corr],
            None,
        );
        // V = (f·σ0)² + σ1² + 2·f·1·ρ·σ0·σ1; ∂V/∂f = 2·f·σ0² + 2·ρ·σ0·σ1.
        let expected = 2.0 * 40.0 * 0.3 * 0.3 + 2.0 * 0.5 * 0.3 * 1.2;
        assert_relative_eq!(dr[0][(0, 0)], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_r_matrix_with_correlations_uses_shifted_time_after_reset() {
        let spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        let ipreds = [50.0, 5.0, 40.0, 4.0];
        let cmts = [1usize, 2, 1, 2];
        let shifted_times = [1.0, 1.0, 101.0, 101.0];
        let raw_times = [1.0, 1.0, 1.0, 1.0];
        let sigma = [0.2, 0.3];
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };

        let r = compute_r_matrix_with_correlations(
            &spec,
            &ipreds,
            &cmts,
            &shifted_times,
            &raw_times,
            &[],
            &sigma,
            &[corr],
        );

        assert_relative_eq!(r[(0, 1)], 50.0 * 5.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_relative_eq!(r[(2, 3)], 40.0 * 4.0 * 0.5 * 0.3 * 0.2, epsilon = 1e-12);
        assert_eq!(r[(0, 3)], 0.0);
        assert_eq!(r[(1, 2)], 0.0);
    }

    #[test]
    fn test_min_variance_floor() {
        // Proportional with f=0 gives V=0, should be floored to MIN_VARIANCE
        let v = residual_variance(ErrorModel::Proportional, 0.0, &[0.1]);
        assert_relative_eq!(v, MIN_VARIANCE, epsilon = 1e-20);
    }

    #[test]
    fn test_iwres_perfect_prediction() {
        let r = iwres(10.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iwres_known_value() {
        // IWRES = (y - f) / sqrt(V) = (12 - 10) / sqrt(1) = 2.0
        let r = iwres(12.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_length() {
        // Single-endpoint model (Additive): CMT is ignored.
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let ipreds = vec![1.0, 2.0, 3.0];
        let obs_cmts = vec![1, 1, 1];
        let r = compute_r_diag(&model.error_spec, &ipreds, &obs_cmts, &[0.5]);
        assert_eq!(r.len(), 3);
        // Additive variance is sigma^2 regardless of prediction/CMT.
        for v in &r {
            assert_relative_eq!(*v, 0.25, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_compute_iwres_vectorized() {
        let model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
        let obs = vec![12.0, 22.0];
        let ipreds = vec![10.0, 20.0];
        let obs_cmts = vec![1, 1];
        let result = compute_iwres(&obs, &ipreds, &obs_cmts, &model.error_spec, &[1.0]);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_iwres_with_correlations_applies_cross_term() {
        let spec = ErrorSpec::Single(ErrorModel::Combined);
        let corr = crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        };
        // V = (10 * 0.2)^2 + 1^2 + 2 * 10 * 0.5 * 0.2 * 1 = 7.
        let result = compute_iwres_with_correlations(
            &[12.0],
            &[10.0],
            &[1],
            &spec,
            &[0.2, 1.0],
            &[corr],
            None,
        );
        assert_relative_eq!(result[0], 2.0 / 7.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_compute_iwres_with_correlations_empty_matches_diagonal() {
        let spec = ErrorSpec::Single(ErrorModel::Additive);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let plain = compute_iwres(&obs, &ipreds, &obs_cmts, &spec, &[1.0]);
        let with =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &[1.0], &[], None);
        assert_eq!(plain, with);
    }

    #[test]
    fn compute_r_matrix_diagonal_keeps_legacy_association() {
        // Bit-reproducibility guard. The bare-sigma R diagonal must keep
        // `residual_variance`'s `(f·σ)·(f·σ)` association, NOT the
        // `((f·f)·σ)·σ` form `variance_at_scaled` uses. The two are equal in
        // exact arithmetic but differ by ~1 ULP under IEEE-754 on ~55% of
        // proportional/combined rows — so delegating `compute_r_matrix_with_correlations`
        // to `_scaled` with an empty multiplier would silently shift every
        // proportional/combined FOCE OFV and CWRES off its bit-for-bit value.
        // `f = 32.451, σ = 0.159` is one such divergent pair.
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let f = 32.451_f64;
        let s = 0.159_f64;
        let legacy = (f * s) * (f * s);
        let reassociated = ((f * f) * s) * s;
        assert_ne!(
            legacy.to_bits(),
            reassociated.to_bits(),
            "fixture must be a pair where the two associations differ"
        );
        let r = compute_r_matrix_with_correlations(&spec, &[f], &[1], &[0.0], &[], &[], &[s], &[]);
        assert_eq!(
            r[(0, 0)].to_bits(),
            legacy.to_bits(),
            "R diagonal must use the legacy (f·σ)·(f·σ) association"
        );
    }

    #[test]
    fn test_compute_iwres_with_correlations_applies_custom_magnitude() {
        // #484 review #4: the sdtab IWRES must use the per-observation magnitude
        // multiplier, so a row whose multiplier ≠ 1 is scaled by it. Proportional
        // error: V = (f·m·σ)², so IWRES = (y−f)/(f·m·σ).
        let spec = ErrorSpec::Single(ErrorModel::Proportional);
        let obs = [12.0, 22.0];
        let ipreds = [10.0, 20.0];
        let obs_cmts = [1, 1];
        let sigma = [0.2];
        // Row 0 bare (mult 1), row 1 inflated by 2.
        let mult = vec![vec![1.0], vec![2.0]];
        let scaled = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&mult),
        );
        assert_relative_eq!(scaled[0], 2.0 / (10.0 * 1.0 * 0.2), epsilon = 1e-12);
        assert_relative_eq!(scaled[1], 2.0 / (20.0 * 2.0 * 0.2), epsilon = 1e-12);

        // An all-ones multiplier reproduces the unscaled IWRES exactly.
        let ones = vec![vec![1.0], vec![1.0]];
        let unit = compute_iwres_with_correlations(
            &obs,
            &ipreds,
            &obs_cmts,
            &spec,
            &sigma,
            &[],
            Some(&ones),
        );
        let bare =
            compute_iwres_with_correlations(&obs, &ipreds, &obs_cmts, &spec, &sigma, &[], None);
        assert_relative_eq!(unit[0], bare[0], epsilon = 1e-12);
        assert_relative_eq!(unit[1], bare[1], epsilon = 1e-12);
    }

    fn make_subject(iwres: Vec<f64>) -> SubjectResult {
        use nalgebra::DVector;
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::zeros(0),
            ipred: vec![0.0; iwres.len()],
            pred: vec![0.0; iwres.len()],
            iwres,
            cwres: vec![],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![],
            n_obs: 0,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
        }
    }

    #[test]
    fn test_dw_monotone_positive_autocorrelation() {
        // Monotonically increasing → strong positive autocorrelation → DW near 0
        let subj = make_subject(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw < 1.5,
            "expected DW < 1.5 for monotone sequence, got {dw}"
        );
        assert!(r > 0.5, "expected positive lag-1 r, got {r}");
    }

    #[test]
    fn test_dw_alternating_negative_autocorrelation() {
        // Alternating signs → strong negative autocorrelation → DW near 4
        let subj = make_subject(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 2.5,
            "expected DW > 2.5 for alternating sequence, got {dw}"
        );
    }

    #[test]
    fn test_dw_uncorrelated_near_two() {
        // White-noise-like residuals → DW near 2 (no autocorrelation)
        let subj = make_subject(vec![1.0, -0.5, 0.2, 0.8, -0.3, -0.7, 0.4, 0.1]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        assert!(
            dw > 1.5 && dw < 2.5,
            "expected DW near 2 for white-noise sequence, got {dw}"
        );
    }

    #[test]
    fn test_nan_iwres_skipped() {
        let subj = make_subject(vec![f64::NAN, 1.0, 2.0, f64::NAN, 3.0]);
        let (_r, dw) = iwres_autocorrelation(&[subj]);
        // Should not panic and should produce a finite result based on valid values
        assert!(
            dw.is_finite(),
            "DW should be finite after skipping NaN entries"
        );
    }

    #[test]
    fn test_single_observation_subject_skipped() {
        let single = make_subject(vec![1.0]);
        let multi = make_subject(vec![1.0, 2.0, 3.0]);
        let (r, dw) = iwres_autocorrelation(&[single, multi]);
        assert!(dw.is_finite());
        assert!(r.is_finite());
    }

    #[test]
    fn test_no_valid_subjects_returns_nan() {
        let subj = make_subject(vec![1.0]); // < 2 valid
        let (r, dw) = iwres_autocorrelation(&[subj]);
        assert!(r.is_nan());
        assert!(dw.is_nan());
    }
}
