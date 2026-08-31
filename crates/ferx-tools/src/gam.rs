//! GAM-based covariate pre-screening (#1114).
//!
//! For each ETA × covariate pair, fits `η_i ~ f(cov_i)` independently and
//! ranks covariates by AIC improvement over the null model `η_i ~ 1`. High-
//! ΔAIC covariates are then prioritised in an SCM (via `[covariate_model]`
//! declarations or a search harness).
//!
//! This is the Rust equivalent of Xpose4's `xpose.gam()` (Jonsson & Karlsson,
//! *Pharm Res* 1999). Like Xpose, it uses independent regressions (not
//! stepwise backfitting), which is appropriate for the pre-screening role
//! where speed and interpretability matter more than joint optimisation.
//!
//! # Relationship to `[covariate_model]` (#1111)
//!
//! GAM screening identifies *candidate* covariate–parameter pairs and suggests
//! whether a linear or flexible (spline) functional form is more supported by
//! the data. The resulting ranking feeds into a modeller's decision about
//! which relations to declare in `[covariate_model]`, but the two features are
//! deliberately decoupled: GAM screening operates on post-hoc EBEs from any
//! fit result; `[covariate_model]` operates on the model file itself.
//!
//! # Shrinkage caveat
//!
//! EBE-based covariate screening is only informative when ETA shrinkage is
//! low (< 30%). At high shrinkage the EBEs regress toward zero and the
//! relationship between `η̂_i` and covariates is attenuated. [`gam_screen`]
//! emits a warning for each ETA whose shrinkage exceeds
//! [`GamOptions::shrinkage_warn_threshold`].

use ferx_core::{CovariateKind, CovariateTable, FitResult, Population};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

// ── Public types ─────────────────────────────────────────────────────────────

/// Options for [`gam_screen`].
#[derive(Debug, Clone)]
pub struct GamOptions {
    /// ETAs to screen. `None` = all ETAs in the fit result.
    pub etas: Option<Vec<String>>,
    /// Covariates to screen. `None` = all covariates in the population.
    pub covariates: Option<Vec<String>>,
    /// Natural-spline degrees of freedom to try for continuous covariates.
    /// Each value in this list is tried in addition to the linear form
    /// (when `include_linear` is true). Default: `[2, 3]`.
    pub spline_df: Vec<usize>,
    /// Include the linear form (`η ~ 1 + x`) as a candidate. Default: true.
    pub include_linear: bool,
    /// Warn when ETA shrinkage exceeds this fraction. Default: 0.30 (30%).
    pub shrinkage_warn_threshold: f64,
}

impl Default for GamOptions {
    fn default() -> Self {
        Self {
            etas: None,
            covariates: None,
            spline_df: vec![2, 3],
            include_linear: true,
            shrinkage_warn_threshold: 0.30,
        }
    }
}

/// Winning functional form for a single covariate in one ETA's GAM screening.
#[derive(Debug, Clone, PartialEq)]
pub enum CovariateForm {
    /// Linear form: `η ~ 1 + x`.
    Linear,
    /// Natural cubic spline with `df` degrees of freedom.
    Spline { df: usize },
    /// One-hot-encoded categorical (reference = lowest observed level).
    Categorical,
}

/// GAM result for one covariate in one ETA's screening.
#[derive(Debug, Clone)]
pub struct CovariateScore {
    pub covariate: String,
    /// `AIC_null − AIC_best`. Positive = covariate improves the null model.
    pub delta_aic: f64,
    /// The winning form (lowest AIC among the candidates tried).
    pub best_form: CovariateForm,
    /// AIC of the best model.
    pub aic: f64,
    /// R² of the best model (0 for Categorical when design is trivial).
    pub r_squared: f64,
}

/// GAM screening results for one ETA, ranked by [`CovariateScore::delta_aic`].
#[derive(Debug, Clone)]
pub struct EtaGamResult {
    pub eta_name: String,
    /// ETA shrinkage (`1 − SD(η̂) / √ω`) from the fit result.
    pub shrinkage: f64,
    /// Null-model AIC on the full set of subjects with non-NaN ETA values.
    pub aic_null: f64,
    /// Covariate scores, ranked by `delta_aic` descending (best first).
    pub covariate_scores: Vec<CovariateScore>,
}

/// Full GAM screening result returned by [`gam_screen`].
#[derive(Debug, Clone)]
pub struct GamResult {
    pub eta_results: Vec<EtaGamResult>,
    pub warnings: Vec<String>,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Screen covariates for each ETA using independent GAM regressions.
///
/// `fit` and `pop` must correspond to the same model run — subjects are
/// aligned by index (same order as produced by [`ferx_core::fit`]).
///
/// Covariates are read from `pop.subjects[i].covariates` (the subject-
/// representative, time-constant value). The covariate kind (continuous vs.
/// categorical) is taken from `fit.covariate_table` when available; when not
/// declared in a `[covariates]` block, the kind falls back to a heuristic:
/// categorical if all values are within 1 × 10⁻⁶ of an integer and there are
/// ≤ 10 unique values.
pub fn gam_screen(fit: &FitResult, pop: &Population, opts: &GamOptions) -> GamResult {
    let mut warnings = Vec::new();

    // Determine which ETA names and covariate names to screen.
    let eta_names: Vec<&str> = match &opts.etas {
        Some(names) => names
            .iter()
            .map(|s| s.as_str())
            .filter(|n| fit.eta_names.iter().any(|en| en.as_str() == *n))
            .collect(),
        None => fit.eta_names.iter().map(|s| s.as_str()).collect(),
    };
    let cov_names: Vec<&str> = match &opts.covariates {
        Some(names) => names.iter().map(|s| s.as_str()).collect(),
        None => pop.covariate_names.iter().map(|s| s.as_str()).collect(),
    };

    if cov_names.is_empty() {
        warnings.push(
            "No covariates to screen; Population.covariate_names is empty. \
             Declare covariates with a [covariates] block or pass opts.covariates explicitly."
                .into(),
        );
        return GamResult {
            eta_results: vec![],
            warnings,
        };
    }

    if eta_names.is_empty() {
        warnings.push(
            "No ETAs to screen; FitResult.eta_names is empty or no requested ETA was found.".into(),
        );
        return GamResult {
            eta_results: vec![],
            warnings,
        };
    }

    // Build per-covariate (name, values-per-subject, kind) once.
    let cov_data: Vec<(String, Vec<f64>, CovariateKind)> = cov_names
        .iter()
        .map(|&name| {
            let kind = determine_cov_kind(name, &fit.covariate_table, &pop.subjects);
            let values: Vec<f64> = pop
                .subjects
                .iter()
                .map(|s| s.covariates.get(name).copied().unwrap_or(f64::NAN))
                .collect();
            (name.to_string(), values, kind)
        })
        .collect();

    // Screen each ETA in parallel (independent fits).
    let eta_results_raw: Vec<(EtaGamResult, Vec<String>)> = eta_names
        .par_iter()
        .map(|&eta_name| {
            let eta_idx = fit
                .eta_names
                .iter()
                .position(|n| n == eta_name)
                .unwrap_or(usize::MAX);

            let shrinkage = if eta_idx < fit.shrinkage_eta.len() {
                fit.shrinkage_eta[eta_idx]
            } else {
                f64::NAN
            };

            let mut eta_warnings = Vec::new();
            if let Some(w) = shrinkage_warning(eta_name, shrinkage, opts.shrinkage_warn_threshold) {
                eta_warnings.push(w);
            }

            let eta_values: Vec<f64> = fit
                .subjects
                .iter()
                .map(|s| {
                    if eta_idx < s.eta.len() {
                        s.eta[eta_idx]
                    } else {
                        f64::NAN
                    }
                })
                .collect();

            let cov_refs: Vec<(&str, &[f64], CovariateKind)> = cov_data
                .iter()
                .map(|(name, vals, kind)| (name.as_str(), vals.as_slice(), *kind))
                .collect();

            let (aic_null, scores) = screen_eta_raw(&eta_values, &cov_refs, opts);

            let result = EtaGamResult {
                eta_name: eta_name.to_string(),
                shrinkage,
                aic_null,
                covariate_scores: scores,
            };

            (result, eta_warnings)
        })
        .collect();

    let mut eta_results = Vec::with_capacity(eta_results_raw.len());
    for (result, eta_warns) in eta_results_raw {
        warnings.extend(eta_warns);
        eta_results.push(result);
    }

    GamResult {
        eta_results,
        warnings,
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Determine the kind of a covariate: prefer the `[covariates]`-block
/// declaration (via `covariate_table`); fall back to a heuristic.
fn determine_cov_kind(
    name: &str,
    covariate_table: &Option<CovariateTable>,
    subjects: &[ferx_core::Subject],
) -> CovariateKind {
    if let Some(table) = covariate_table {
        if let Some(pos) = table.names.iter().position(|n| n == name) {
            return table.kinds[pos];
        }
    }
    // Heuristic: categorical if all values are near-integer and ≤ 10 unique.
    let values: Vec<f64> = subjects
        .iter()
        .filter_map(|s| s.covariates.get(name).copied())
        .filter(|v| !v.is_nan())
        .collect();
    if values.is_empty() {
        return CovariateKind::Continuous;
    }
    let near_int = values.iter().all(|&v| (v - v.round()).abs() < 1e-6);
    if near_int {
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        if sorted.len() <= 10 {
            return CovariateKind::Categorical;
        }
    }
    CovariateKind::Continuous
}

/// Emit a shrinkage warning string if the threshold is exceeded.
pub(crate) fn shrinkage_warning(eta_name: &str, shrinkage: f64, threshold: f64) -> Option<String> {
    if !shrinkage.is_nan() && shrinkage > threshold {
        Some(format!(
            "{}: shrinkage {:.1}% exceeds the {:.0}% threshold; \
             EBE-based covariate screening may be unreliable.",
            eta_name,
            shrinkage * 100.0,
            threshold * 100.0,
        ))
    } else {
        None
    }
}

/// Screen all covariates against one set of ETA values.
///
/// Returns the global null AIC (computed on all subjects with non-NaN ETA) and
/// the per-covariate scores ranked by `delta_aic` descending.
///
/// `pub(crate)` for unit tests.
pub(crate) fn screen_eta_raw(
    eta_values: &[f64],
    covariates: &[(&str, &[f64], CovariateKind)],
    opts: &GamOptions,
) -> (f64, Vec<CovariateScore>) {
    // Filter to subjects with a valid ETA.
    let valid_eta: DVector<f64> = DVector::from_iterator(
        eta_values.iter().filter(|&&v| !v.is_nan()).count(),
        eta_values.iter().filter(|&&v| !v.is_nan()).copied(),
    );

    if valid_eta.len() < 3 {
        return (f64::NAN, vec![]);
    }

    // Global null model (intercept only) AIC.
    let n_all = valid_eta.len();
    let x_null_all = DMatrix::from_element(n_all, 1, 1.0);
    let (aic_null, _) = match ols_aic(&x_null_all, &valid_eta) {
        Some(v) => v,
        None => return (f64::NAN, vec![]),
    };

    let mut scores = Vec::with_capacity(covariates.len());

    for &(cov_name, cov_values, kind) in covariates {
        // Collect subjects with valid ETA and valid covariate.
        let pairs: Vec<(f64, f64)> = eta_values
            .iter()
            .zip(cov_values.iter())
            .filter(|(&e, &c)| !e.is_nan() && !c.is_nan())
            .map(|(&e, &c)| (e, c))
            .collect();

        let n = pairs.len();
        if n < 3 {
            continue;
        }

        let y = DVector::from_iterator(n, pairs.iter().map(|&(e, _)| e));
        let x_vals: Vec<f64> = pairs.iter().map(|&(_, c)| c).collect();

        // Per-covariate null AIC (only over subjects with valid covariate).
        let x_null = DMatrix::from_element(n, 1, 1.0);
        let (aic_null_local, _) = match ols_aic(&x_null, &y) {
            Some(v) => v,
            None => continue,
        };

        // Fit candidate forms and pick the one with the lowest AIC.
        let mut best_aic = f64::INFINITY;
        let mut best_r2 = 0.0;
        let mut best_form = CovariateForm::Linear; // overwritten below

        match kind {
            CovariateKind::Categorical => {
                let dummies = categorical_design(&x_vals);
                let n_dummies = dummies.ncols();
                // Design: [intercept | dummies]
                let mut x_cat = DMatrix::zeros(n, n_dummies + 1);
                for row in 0..n {
                    x_cat[(row, 0)] = 1.0;
                    for col in 0..n_dummies {
                        x_cat[(row, col + 1)] = dummies[(row, col)];
                    }
                }
                if let Some((aic, r2)) = ols_aic(&x_cat, &y) {
                    if aic < best_aic {
                        best_aic = aic;
                        best_r2 = r2;
                        best_form = CovariateForm::Categorical;
                    }
                }
            }
            CovariateKind::Continuous => {
                // Linear form.
                if opts.include_linear {
                    let mut x_lin = DMatrix::zeros(n, 2);
                    for (row, &v) in x_vals.iter().enumerate() {
                        x_lin[(row, 0)] = 1.0;
                        x_lin[(row, 1)] = v;
                    }
                    if let Some((aic, r2)) = ols_aic(&x_lin, &y) {
                        if aic < best_aic {
                            best_aic = aic;
                            best_r2 = r2;
                            best_form = CovariateForm::Linear;
                        }
                    }
                }

                // Spline forms.
                for &df in &opts.spline_df {
                    // Need at least df+2 subjects to fit df+1 params + intercept.
                    if df < 1 || n <= df + 1 {
                        continue;
                    }
                    let basis = ns_basis(&x_vals, df);
                    // Design: [intercept | basis columns]
                    let mut x_spl = DMatrix::zeros(n, df + 1);
                    for row in 0..n {
                        x_spl[(row, 0)] = 1.0;
                        for col in 0..df {
                            x_spl[(row, col + 1)] = basis[(row, col)];
                        }
                    }
                    if let Some((aic, r2)) = ols_aic(&x_spl, &y) {
                        if aic < best_aic {
                            best_aic = aic;
                            best_r2 = r2;
                            best_form = CovariateForm::Spline { df };
                        }
                    }
                }
            }
        }

        if best_aic.is_infinite() {
            continue; // no valid form fit
        }

        scores.push(CovariateScore {
            covariate: cov_name.to_string(),
            delta_aic: aic_null_local - best_aic,
            best_form,
            aic: best_aic,
            r_squared: best_r2,
        });
    }

    // Rank by delta_aic descending (most important covariate first).
    scores.sort_by(|a, b| {
        b.delta_aic
            .partial_cmp(&a.delta_aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    (aic_null, scores)
}

/// Natural cubic spline basis matrix of shape `n × df` (ESL §5.2.1).
///
/// With K = df + 1 knots (boundary = min/max; interior at quantiles `i/df`
/// for `i = 1..df−1`), the columns are:
///
/// - col 0: `x` (N₂ in ESL notation)
/// - col k (k = 1..df−1): `d_{k}(x) − d_{K−1}(x)`
///
/// where `d_j(x) = [(x − ξ_j)³₊ − (x − ξ_K)³₊] / (ξ_K − ξ_j)`.
///
/// No intercept column is included; callers prepend one.
///
/// `pub(crate)` for unit tests.
pub(crate) fn ns_basis(x: &[f64], df: usize) -> DMatrix<f64> {
    let n = x.len();
    if df == 0 || n == 0 {
        return DMatrix::zeros(n, 0);
    }

    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let xi_min = sorted[0];
    let xi_max = *sorted.last().unwrap();

    // Build K = df+1 knots: boundary + (df-1) interior at evenly-spaced quantiles.
    let k_total = df + 1;
    let mut knots = Vec::with_capacity(k_total);
    knots.push(xi_min);
    for i in 1..df {
        knots.push(quantile_sorted(&sorted, i as f64 / df as f64));
    }
    knots.push(xi_max);

    // d_{K-1}(x) is subtracted from every higher-order column.
    // K-1 (1-indexed) = knots[df-1] (0-indexed).
    let xi_km1 = knots[df - 1]; // ξ_{K-1}

    let mut mat = DMatrix::zeros(n, df);
    for (row, &xi) in x.iter().enumerate() {
        // Column 0: N₂(x) = x.
        mat[(row, 0)] = xi;

        // Columns 1..df-1: N_{k+2}(x) = d_k(x) − d_{K-1}(x) for k = 1..K-2.
        // k (1-indexed) maps to knots[k-1] (0-indexed).
        // k runs from 1 to K-2 = df-1, giving columns 1..df-1.
        let d_km1 = d_func(xi, xi_km1, xi_max);
        for col in 1..df {
            let xi_k = knots[col - 1]; // k-th knot (k = col, 1-indexed → knots[col-1])
            let d_k = d_func(xi, xi_k, xi_max);
            mat[(row, col)] = d_k - d_km1;
        }
    }

    mat
}

/// Helper: `d_j(x) = [(x − ξ_j)³₊ − (x − ξ_K)³₊] / (ξ_K − ξ_j)`.
///
/// `xi_j` is the j-th knot, `xi_k` is the maximum knot ξ_K.
#[inline]
pub(crate) fn d_func(x: f64, xi_j: f64, xi_k: f64) -> f64 {
    let denom = xi_k - xi_j;
    if denom.abs() < 1e-15 {
        return 0.0;
    }
    let tp_j = (x - xi_j).max(0.0).powi(3);
    let tp_k = (x - xi_k).max(0.0).powi(3);
    (tp_j - tp_k) / denom
}

/// Linear interpolation quantile on a sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// OLS fit: returns `(AIC, R²)` or `None` when X'X is singular.
///
/// AIC = n·ln(RSS/n) + 2·p (Gaussian / least-squares AIC).
///
/// `pub(crate)` for unit tests.
pub(crate) fn ols_aic(x: &DMatrix<f64>, y: &DVector<f64>) -> Option<(f64, f64)> {
    let xtx = x.transpose() * x;
    let xty = x.transpose() * y;
    let chol = xtx.cholesky()?;
    let beta = chol.solve(&xty);
    let residuals = y - x * beta;
    let rss = residuals.norm_squared();
    let n = y.len() as f64;
    let p = x.ncols() as f64;
    let aic = n * (rss / n).ln() + 2.0 * p;
    let mean_y = y.mean();
    let sst = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>();
    let r2 = if sst < 1e-20 { 0.0 } else { 1.0 - rss / sst };
    Some((aic, r2))
}

/// One-hot encoding of a categorical covariate, shape `n × (levels − 1)`.
///
/// The lowest observed level is the reference (dropped). Returns an empty
/// matrix when there is only one unique level.
///
/// `pub(crate)` for unit tests.
pub(crate) fn categorical_design(x: &[f64]) -> DMatrix<f64> {
    let mut levels: Vec<f64> = x.to_vec();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    let n_levels = levels.len();
    if n_levels <= 1 {
        return DMatrix::zeros(x.len(), 0);
    }

    let n_dummies = n_levels - 1;
    let mut mat = DMatrix::zeros(x.len(), n_dummies);
    for (row, &val) in x.iter().enumerate() {
        for (col, &level) in levels[1..].iter().enumerate() {
            if (val - level).abs() < 1e-10 {
                mat[(row, col)] = 1.0;
            }
        }
    }
    mat
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ns_basis ─────────────────────────────────────────────────────────────

    #[test]
    fn ns_basis_df1_returns_x() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let basis = ns_basis(&x, 1);
        assert_eq!(basis.nrows(), 5);
        assert_eq!(basis.ncols(), 1);
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                (basis[(i, 0)] - xi).abs() < 1e-12,
                "col 0 should be x at row {i}"
            );
        }
    }

    #[test]
    fn ns_basis_df2_three_knots() {
        // x = 0..10 → knots = [0, 5, 10]
        let x: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let basis = ns_basis(&x, 2);
        assert_eq!(basis.nrows(), 11);
        assert_eq!(basis.ncols(), 2);

        // Column 0 must equal x.
        for (i, &xi) in x.iter().enumerate() {
            assert!(
                (basis[(i, 0)] - xi).abs() < 1e-12,
                "col 0 should be x at row {i}"
            );
        }

        // Column 1 = d_1(x) − d_2(x) with knots [0.0, 5.0, 10.0].
        // d_1 uses ξ_1=0.0, ξ_K=10.0; d_2 uses ξ_2=5.0, ξ_K=10.0.
        for (i, &xi) in x.iter().enumerate() {
            let d1 = d_func(xi, 0.0, 10.0);
            let d2 = d_func(xi, 5.0, 10.0);
            let expected = d1 - d2;
            assert!(
                (basis[(i, 1)] - expected).abs() < 1e-12,
                "col 1 mismatch at row {i}: got {}, expected {expected}",
                basis[(i, 1)]
            );
        }
    }

    // ── ols_aic ───────────────────────────────────────────────────────────────

    #[test]
    fn ols_null_aic_formula() {
        // y = [1,2,3,4,5], null model (intercept only, mean = 3).
        // RSS = 4+1+0+1+4 = 10
        // AIC = 5·ln(10/5) + 2·1 = 5·ln(2) + 2 ≈ 5.466
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x = DMatrix::from_element(5, 1, 1.0);
        let (aic, r2) = ols_aic(&x, &y).expect("null model OLS should succeed");
        let expected_aic = 5.0 * (10.0_f64 / 5.0).ln() + 2.0;
        assert!(
            (aic - expected_aic).abs() < 1e-10,
            "got {aic}, expected {expected_aic}"
        );
        // Null model R² = 0.
        assert!(r2.abs() < 1e-10, "null model R² should be 0, got {r2}");
    }

    #[test]
    fn ols_singular_returns_none() {
        // Constant column makes X'X singular.
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = DMatrix::from_vec(3, 2, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(ols_aic(&x, &y).is_none());
    }

    // ── shrinkage_warning ─────────────────────────────────────────────────────

    #[test]
    fn high_shrinkage_emits_warning() {
        let w = shrinkage_warning("ETA_CL", 0.50, 0.30);
        assert!(w.is_some(), "expected warning for 50% shrinkage");
        let text = w.unwrap();
        assert!(text.contains("ETA_CL"), "warning should name the ETA");
        assert!(
            text.contains("50.0%") || text.contains("50%"),
            "warning should include shrinkage value: {text}"
        );
    }

    #[test]
    fn low_shrinkage_no_warning() {
        assert!(shrinkage_warning("ETA_CL", 0.20, 0.30).is_none());
        assert!(shrinkage_warning("ETA_CL", f64::NAN, 0.30).is_none());
    }

    // ── screen_eta_raw ────────────────────────────────────────────────────────

    #[test]
    fn uncorrelated_cov_near_zero_delta_aic() {
        // Deterministic pseudo-random but unrelated eta and covariate.
        let eta: Vec<f64> = (0..30).map(|i| (i as f64 * 0.17 + 0.31).sin()).collect();
        let cov: Vec<f64> = (0..30).map(|i| (i as f64 * 0.13 + 0.71).cos()).collect();
        let opts = GamOptions::default();
        let cov_refs = [("COV", cov.as_slice(), CovariateKind::Continuous)];
        let (_aic_null, scores) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(!scores.is_empty());
        assert!(
            scores[0].delta_aic.abs() < 8.0,
            "unrelated data should have small |Δ AIC|, got {}",
            scores[0].delta_aic
        );
    }

    #[test]
    fn strong_linear_signal_gives_large_delta_aic() {
        // y = 2x with small deterministic noise. The covariate is strongly
        // informative regardless of whether linear or a low-df spline wins the
        // form comparison.
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let eta: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * xi + (i as f64 * 0.31).sin() * 0.5)
            .collect();
        let opts = GamOptions::default();
        let cov_refs = [("WT", x.as_slice(), CovariateKind::Continuous)];
        let (_aic_null, scores) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(!scores.is_empty());
        assert!(
            scores[0].delta_aic > 5.0,
            "strong linear signal: Δ AIC should be > 5, got {}",
            scores[0].delta_aic
        );
        // Either linear or a low-df spline is acceptable; both capture the trend.
        assert!(
            matches!(
                scores[0].best_form,
                CovariateForm::Linear | CovariateForm::Spline { .. }
            ),
            "unexpected form for linear data: {:?}",
            scores[0].best_form
        );
    }

    #[test]
    fn quadratic_signal_spline_beats_linear() {
        // y = (x − 14.5)² — a symmetric parabola.
        // The OLS linear fit through a symmetric parabola has slope ≈ 0, so
        // linear gives virtually no Δ AIC over the null. A spline with df ≥ 2
        // captures the curvature and should win clearly.
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let eta: Vec<f64> = x.iter().map(|&xi| (xi - 14.5_f64).powi(2)).collect();
        let opts = GamOptions {
            spline_df: vec![2, 3],
            include_linear: true,
            ..Default::default()
        };
        let cov_refs = [("X", x.as_slice(), CovariateKind::Continuous)];
        let (_aic_null, scores) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(!scores.is_empty());
        // Spline form must win over linear.
        assert!(
            matches!(scores[0].best_form, CovariateForm::Spline { .. }),
            "spline should beat linear for quadratic data, got {:?}",
            scores[0].best_form
        );
        // And delta_aic should be substantial.
        assert!(
            scores[0].delta_aic > 10.0,
            "quadratic signal: Δ AIC should be > 10, got {}",
            scores[0].delta_aic
        );
    }

    #[test]
    fn categorical_covariate_detects_group_effect() {
        // Binary covariate: eta is clearly higher in group 1.
        let x: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let eta: Vec<f64> = x.iter().map(|&xi| xi * 3.0 + 0.1).collect();
        let opts = GamOptions::default();
        let cov_refs = [("SEX", x.as_slice(), CovariateKind::Categorical)];
        let (_aic_null, scores) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(!scores.is_empty());
        assert_eq!(
            scores[0].best_form,
            CovariateForm::Categorical,
            "categorical form should be selected for a binary covariate"
        );
        assert!(
            scores[0].delta_aic > 5.0,
            "clear group effect should give Δ AIC > 5, got {}",
            scores[0].delta_aic
        );
    }

    #[test]
    fn too_few_subjects_returns_empty() {
        let eta = vec![1.0, 2.0]; // n < 3
        let cov = vec![0.0, 1.0];
        let opts = GamOptions::default();
        let cov_refs = [("X", cov.as_slice(), CovariateKind::Continuous)];
        let (aic_null, scores) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(aic_null.is_nan());
        assert!(scores.is_empty());
    }

    // ── Xpose4 anchor ─────────────────────────────────────────────────────────
    //
    // Gold-standard reference values computed in R with gam::gam() + ns(),
    // which is the statistical engine xpose4::xpose.gam() uses internally
    // (xpose4 defaults: smoother3=ns,arg3=df=2; smoother4=ns,arg4=df=3;
    // steppit=FALSE for independent screening).
    //
    // Because gam::gam(y ~ ns(x, df=k)) with Gaussian family and our
    // ols_aic formula produce identical ΔAIC (the constant n(1+ln2π) cancels),
    // the reference values below are bit-for-bit reproducible from R.
    //
    // Anchor data (n=60, seed=20240101):
    //   ETA_CL ~ 0.40*(WT−70)/70 + N(0, 0.15²)  → WT ranks #1
    //   ETA_V  ~ 0.35*(SEX−0.5) + N(0, 0.20²)   → SEX ranks #1
    //   CRCL   ~ noise covariate (no true effect)
    //
    // R script: docs/gam_anchor_reference/gen_anchor.R
    #[test]
    fn xpose4_anchor_delta_aic_matches_reference() {
        // Parse the anchor EBE CSV (generated with R set.seed(20240101)).
        const CSV: &str = include_str!("../tests/data/gam_anchor_ebes.csv");

        let mut lines = CSV.lines();
        // R's write.csv quotes all field names; strip the surrounding quotes.
        let header: Vec<String> = lines
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.trim_matches('"').to_string())
            .collect();
        let col = |name: &str| header.iter().position(|h| h == name).unwrap();

        let (ic, iv, iwt, icrcl, isex) = (
            col("ETA_CL"),
            col("ETA_V"),
            col("WT"),
            col("CRCL"),
            col("SEX"),
        );

        let (mut eta_cl, mut eta_v, mut wt, mut crcl, mut sex) =
            (vec![], vec![], vec![], vec![], vec![]);
        for line in lines {
            let c: Vec<&str> = line.split(',').collect();
            let p = |i: usize| c[i].parse::<f64>().unwrap();
            eta_cl.push(p(ic));
            eta_v.push(p(iv));
            wt.push(p(iwt));
            crcl.push(p(icrcl));
            sex.push(p(isex));
        }

        let opts = GamOptions::default();

        // Reference Δ AIC from R gam::gam (= xpose4 defaults, steppit=FALSE).
        // Tolerance 1e-4 (R parity check shows max |diff| = 2.7e-5 on real data).
        let tol = 1e-4_f64;

        // ── ETA_CL ────────────────────────────────────────────────────────────
        let covs_cl: &[(&str, &[f64], CovariateKind)] = &[
            ("WT", wt.as_slice(), CovariateKind::Continuous),
            ("CRCL", crcl.as_slice(), CovariateKind::Continuous),
            ("SEX", sex.as_slice(), CovariateKind::Categorical),
        ];
        let (_, scores_cl) = screen_eta_raw(&eta_cl, covs_cl, &opts);

        let find = |scores: &Vec<CovariateScore>, name: &str| {
            scores
                .iter()
                .find(|s| s.covariate == name)
                .unwrap()
                .delta_aic
        };

        assert!(
            (find(&scores_cl, "WT") - 1.882_144).abs() < tol,
            "ETA_CL × WT: got {:.6}",
            find(&scores_cl, "WT")
        );
        assert!(
            (find(&scores_cl, "CRCL") - -0.525_482).abs() < tol,
            "ETA_CL × CRCL: got {:.6}",
            find(&scores_cl, "CRCL")
        );
        assert!(
            (find(&scores_cl, "SEX") - -1.383_530).abs() < tol,
            "ETA_CL × SEX: got {:.6}",
            find(&scores_cl, "SEX")
        );
        assert_eq!(scores_cl[0].covariate, "WT", "WT must rank #1 for ETA_CL");

        // ── ETA_V ─────────────────────────────────────────────────────────────
        let covs_v: &[(&str, &[f64], CovariateKind)] = &[
            ("WT", wt.as_slice(), CovariateKind::Continuous),
            ("CRCL", crcl.as_slice(), CovariateKind::Continuous),
            ("SEX", sex.as_slice(), CovariateKind::Categorical),
        ];
        let (_, scores_v) = screen_eta_raw(&eta_v, covs_v, &opts);

        assert!(
            (find(&scores_v, "WT") - -1.655_242).abs() < tol,
            "ETA_V × WT: got {:.6}",
            find(&scores_v, "WT")
        );
        assert!(
            (find(&scores_v, "CRCL") - -1.491_354).abs() < tol,
            "ETA_V × CRCL: got {:.6}",
            find(&scores_v, "CRCL")
        );
        assert!(
            (find(&scores_v, "SEX") - 22.295_612).abs() < tol,
            "ETA_V × SEX: got {:.6}",
            find(&scores_v, "SEX")
        );
        assert_eq!(scores_v[0].covariate, "SEX", "SEX must rank #1 for ETA_V");
    }

    // ── Speed benchmark ───────────────────────────────────────────────────────
    //
    // Ignored by default; run with:
    //   cargo test -p ferx-tools --lib --release -- gam_speed --ignored --nocapture
    //
    // Exercises a large synthetic dataset: 2 000 subjects, 8 ETAs, 15 covariates
    // (120 covariates × 3 forms = 360 OLS fits per run).  Runs sequential and
    // parallel (rayon over ETAs) so you can see the parallelism gain.
    #[test]
    #[ignore = "speed benchmark — run with: cargo test -p ferx-tools --lib --release -- gam_speed --ignored --nocapture"]
    fn gam_speed_large_dataset() {
        use std::time::Instant;

        const N: usize = 2_000;
        const N_ETAS: usize = 8;
        const N_COVS: usize = 15;

        // Deterministic synthetic ETAs (no RNG dependency).
        // Pattern: sine waves with incommensurable frequencies per ETA.
        let eta_data: Vec<Vec<f64>> = (0..N_ETAS)
            .map(|e| {
                (0..N)
                    .map(|i| ((i as f64 * (e as f64 * 0.7 + 1.1)) * 0.003_141).sin() * 0.35)
                    .collect()
            })
            .collect();

        // Covariates: 3 categorical (c % 5 == 0) + 12 continuous.
        let cov_data: Vec<(String, Vec<f64>, CovariateKind)> = (0..N_COVS)
            .map(|c| {
                if c % 5 == 0 {
                    let vals: Vec<f64> = (0..N)
                        .map(|i| if (i + c * 3) % 2 == 0 { 0.0 } else { 1.0 })
                        .collect();
                    (format!("COV_{c:02}"), vals, CovariateKind::Categorical)
                } else {
                    // Continuous: mix of range, centre and scale for realism.
                    let centre = 30.0 + c as f64 * 5.0;
                    let vals: Vec<f64> = (0..N)
                        .map(|i| {
                            centre + ((i as f64 * (c as f64 * 0.4 + 0.9)) * 0.002_718).cos() * 20.0
                        })
                        .collect();
                    (format!("COV_{c:02}"), vals, CovariateKind::Continuous)
                }
            })
            .collect();

        let cov_refs: Vec<(&str, &[f64], CovariateKind)> = cov_data
            .iter()
            .map(|(n, v, k)| (n.as_str(), v.as_slice(), *k))
            .collect();

        let opts = GamOptions::default();
        let total_pairs = N_ETAS * N_COVS;

        // ── Sequential (one ETA after another, single thread) ─────────────────
        // Warm-up.
        for e in 0..N_ETAS {
            let _ = screen_eta_raw(&eta_data[e], &cov_refs, &opts);
        }
        let t0 = Instant::now();
        let n_seq_runs = 5;
        for _ in 0..n_seq_runs {
            for e in 0..N_ETAS {
                let _ = screen_eta_raw(&eta_data[e], &cov_refs, &opts);
            }
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_seq_runs as f64;

        // ── Parallel (rayon par_iter over ETAs, mirrors gam_screen internals) ──
        // Warm-up.
        (0..N_ETAS).into_par_iter().for_each(|e| {
            let _ = screen_eta_raw(&eta_data[e], &cov_refs, &opts);
        });
        let t1 = Instant::now();
        let n_par_runs = 20;
        for _ in 0..n_par_runs {
            (0..N_ETAS).into_par_iter().for_each(|e| {
                let _ = screen_eta_raw(&eta_data[e], &cov_refs, &opts);
            });
        }
        let par_ms = t1.elapsed().as_secs_f64() * 1000.0 / n_par_runs as f64;

        println!(
            "\n╔══ GAM speed benchmark ══════════════════════════════════╗\n\
             ║  dataset  : {N} subjects, {N_ETAS} ETAs, {N_COVS} covariates\n\
             ║  OLS fits : {total_pairs} pairs × 3 forms = {} per run\n\
             ╠══ sequential ═══════════════════════════════════════════╣\n\
             ║  {seq_ms:.2} ms/run   ({:.3} ms/pair)\n\
             ╠══ parallel (rayon over ETAs) ═══════════════════════════╣\n\
             ║  {par_ms:.2} ms/run   ({:.3} ms/pair)   speedup {:.1}×\n\
             ╚═════════════════════════════════════════════════════════╝",
            total_pairs * 3,
            seq_ms / total_pairs as f64,
            par_ms / total_pairs as f64,
            seq_ms / par_ms,
        );
    }
}
