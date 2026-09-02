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
    let eta_rows: Vec<&[f64]> = fit.subjects.iter().map(|s| s.eta.as_slice()).collect();
    screen_from_parts(
        &fit.eta_names,
        &fit.shrinkage_eta,
        &eta_rows,
        &fit.covariate_table,
        pop,
        opts,
    )
}

/// The body of [`gam_screen`], taking only the pieces of the [`FitResult`] it
/// actually reads.
///
/// Split out so the name-selection and column-collection logic is reachable
/// from a unit test: `FitResult` has ~70 fields and no `Default`, so a test
/// cannot construct one, while `Population` has six public fields and
/// `Subject` derives `Default`. Every caller-visible behaviour of
/// [`gam_screen`] lives here.
///
/// `eta_rows[i]` is subject `i`'s EBE vector, indexed by position in
/// `all_eta_names`.
pub(crate) fn screen_from_parts(
    all_eta_names: &[String],
    shrinkage_eta: &[f64],
    eta_rows: &[&[f64]],
    covariate_table: &Option<CovariateTable>,
    pop: &Population,
    opts: &GamOptions,
) -> GamResult {
    let mut warnings = Vec::new();

    // Determine which ETA names to screen. A requested name the fit does not
    // carry is reported rather than dropped in silence.
    let eta_names: Vec<&str> = match &opts.etas {
        Some(names) => {
            let mut kept = Vec::new();
            for n in names {
                if all_eta_names.iter().any(|en| en == n) {
                    kept.push(n.as_str());
                } else {
                    warnings.push(format!(
                        "Requested ETA '{n}' is not in the fit result; skipped."
                    ));
                }
            }
            kept
        }
        None => all_eta_names.iter().map(|s| s.as_str()).collect(),
    };

    // Same for covariates. Membership is tested against the subject records
    // rather than `pop.covariate_names`, because an in-memory `Population`
    // built by a caller may carry covariates without populating that list.
    let cov_names: Vec<&str> = match &opts.covariates {
        Some(names) => {
            let mut kept = Vec::new();
            for n in names {
                let present = pop
                    .subjects
                    .iter()
                    .any(|s| s.covariates.contains_key(n.as_str()));
                if present {
                    kept.push(n.as_str());
                } else {
                    warnings.push(format!(
                        "Requested covariate '{n}' is carried by no subject in the dataset; skipped."
                    ));
                }
            }
            kept
        }
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
            let kind = determine_cov_kind(name, covariate_table, &pop.subjects);
            let values: Vec<f64> = pop
                .subjects
                .iter()
                .map(|s| s.covariates.get(name).copied().unwrap_or(f64::NAN))
                .collect();
            (name.to_string(), values, kind)
        })
        .collect();

    // Transpose the per-subject EBE vectors into one column per screened ETA.
    let eta_idx: Vec<Option<usize>> = eta_names
        .iter()
        .map(|&name| all_eta_names.iter().position(|n| n == name))
        .collect();
    let eta_cols_owned: Vec<Vec<f64>> = eta_idx
        .iter()
        .map(|&idx| {
            eta_rows
                .iter()
                .map(|row| match idx {
                    Some(i) if i < row.len() => row[i],
                    _ => f64::NAN,
                })
                .collect()
        })
        .collect();
    let shrink: Vec<f64> = eta_idx
        .iter()
        .map(|&idx| match idx {
            Some(i) if i < shrinkage_eta.len() => shrinkage_eta[i],
            _ => f64::NAN,
        })
        .collect();

    let cov_names_ref: Vec<&str> = cov_data.iter().map(|(n, _, _)| n.as_str()).collect();
    let cov_cols: Vec<&[f64]> = cov_data.iter().map(|(_, v, _)| v.as_slice()).collect();
    let cov_kinds: Vec<CovariateKind> = cov_data.iter().map(|(_, _, k)| *k).collect();
    let eta_cols: Vec<&[f64]> = eta_cols_owned.iter().map(|v| v.as_slice()).collect();

    let mut result = gam_screen_raw(
        &eta_names,
        &eta_cols,
        &shrink,
        &cov_names_ref,
        &cov_cols,
        &cov_kinds,
        opts,
    );
    warnings.append(&mut result.warnings);
    result.warnings = warnings;
    result
}

/// Low-level GAM screening that accepts pre-aggregated, aligned per-subject
/// data as plain slices.
///
/// Use this when constructing a [`FitResult`] / [`Population`] pair is
/// inconvenient — for example from R or Python bindings where the data has
/// already been collated on the host-language side.
///
/// ## Alignment contract
///
/// All per-subject slices (`eta_cols[i]`, `cov_cols[j]`) must have the same
/// length (one entry per subject, same subject order). `f64::NAN` marks a
/// missing value; subjects with a NaN ETA are excluded from that ETA's
/// regressions; subjects with a NaN covariate are excluded from that
/// covariate's regression.
///
/// - `eta_names`, `eta_cols`, and `shrinkage` must all have length `n_eta`.
/// - `cov_names`, `cov_cols`, and `cov_kinds` must all have length `n_cov`.
///
/// # Panics
///
/// Panics when any of those length invariants is violated. A mismatched column
/// would otherwise be truncated by `zip` to the shortest input and produce a
/// silently wrong ranking — the worst outcome for a tool whose entire output
/// is an ordering.
#[allow(clippy::too_many_arguments)]
pub fn gam_screen_raw(
    eta_names: &[&str],
    eta_cols: &[&[f64]],
    shrinkage: &[f64],
    cov_names: &[&str],
    cov_cols: &[&[f64]],
    cov_kinds: &[CovariateKind],
    opts: &GamOptions,
) -> GamResult {
    assert_eq!(
        eta_names.len(),
        eta_cols.len(),
        "gam_screen_raw: eta_names and eta_cols must have the same length"
    );
    assert_eq!(
        eta_names.len(),
        shrinkage.len(),
        "gam_screen_raw: eta_names and shrinkage must have the same length"
    );
    assert_eq!(
        cov_names.len(),
        cov_cols.len(),
        "gam_screen_raw: cov_names and cov_cols must have the same length"
    );
    assert_eq!(
        cov_names.len(),
        cov_kinds.len(),
        "gam_screen_raw: cov_names and cov_kinds must have the same length"
    );
    let n_subjects = eta_cols.first().map(|c| c.len()).unwrap_or(0);
    for (i, col) in eta_cols.iter().enumerate() {
        assert_eq!(
            col.len(),
            n_subjects,
            "gam_screen_raw: eta_cols[{i}] has {} entries, expected {n_subjects}",
            col.len()
        );
    }
    for (j, col) in cov_cols.iter().enumerate() {
        assert_eq!(
            col.len(),
            n_subjects,
            "gam_screen_raw: cov_cols[{j}] has {} entries, expected {n_subjects}",
            col.len()
        );
    }

    let cov_refs: Vec<(&str, &[f64], CovariateKind)> = cov_names
        .iter()
        .zip(cov_cols.iter())
        .zip(cov_kinds.iter())
        .map(|((&name, &vals), &kind)| (name, vals, kind))
        .collect();

    let results_and_warnings: Vec<(EtaGamResult, Vec<String>)> = eta_names
        .par_iter()
        .zip(eta_cols.par_iter())
        .zip(shrinkage.par_iter())
        .map(|((&eta_name, &eta_vals), &shrink)| {
            let mut eta_warnings = Vec::new();
            if let Some(w) = shrinkage_warning(eta_name, shrink, opts.shrinkage_warn_threshold) {
                eta_warnings.push(w);
            }
            let (aic_null, covariate_scores, screen_warnings) =
                screen_eta_raw(eta_vals, &cov_refs, opts);
            eta_warnings.extend(
                screen_warnings
                    .into_iter()
                    .map(|w| format!("{eta_name}: {w}")),
            );
            let result = EtaGamResult {
                eta_name: eta_name.to_string(),
                shrinkage: shrink,
                aic_null,
                covariate_scores,
            };
            (result, eta_warnings)
        })
        .collect();

    let mut eta_results = Vec::with_capacity(results_and_warnings.len());
    let mut warnings = Vec::new();
    for (result, w) in results_and_warnings {
        warnings.extend(w);
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

/// Emit a shrinkage warning string when the threshold is exceeded, or when
/// shrinkage could not be determined at all.
///
/// A `NaN` shrinkage means the fit reported none for this ETA (a restored fit,
/// or a method that does not populate `shrinkage_eta`). Staying silent there
/// would skip the module's central caveat in exactly the case where the
/// precondition is *unknown*, so it gets its own warning.
pub(crate) fn shrinkage_warning(eta_name: &str, shrinkage: f64, threshold: f64) -> Option<String> {
    if shrinkage.is_nan() {
        return Some(format!(
            "{eta_name}: ETA shrinkage is unavailable for this fit, so the \
             low-shrinkage precondition for EBE-based covariate screening \
             could not be checked."
        ));
    }
    if shrinkage > threshold {
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
/// Returns the global null AIC (computed on all subjects with non-NaN ETA),
/// the per-covariate scores ranked by `delta_aic` descending, and one warning
/// per covariate that had to be skipped.
///
/// Every `continue` in this function drops a covariate from the ranking, where
/// "dropped" and "screened, found unimportant" look identical to a reader of
/// the result — so each one emits a warning saying which happened and why.
///
/// `pub(crate)` for unit tests.
pub(crate) fn screen_eta_raw(
    eta_values: &[f64],
    covariates: &[(&str, &[f64], CovariateKind)],
    opts: &GamOptions,
) -> (f64, Vec<CovariateScore>, Vec<String>) {
    let mut warnings = Vec::new();

    // Filter to subjects with a valid ETA.
    let valid_eta: DVector<f64> = DVector::from_iterator(
        eta_values.iter().filter(|&&v| !v.is_nan()).count(),
        eta_values.iter().filter(|&&v| !v.is_nan()).copied(),
    );

    if valid_eta.len() < 3 {
        warnings.push(format!(
            "only {} subject(s) have a usable EBE; at least 3 are needed to screen.",
            valid_eta.len()
        ));
        return (f64::NAN, vec![], warnings);
    }

    // A constant EBE column — a fixed random effect, or one the data never
    // identified — makes RSS zero for *every* model including the null, so both
    // AICs are −∞ and every ΔAIC is `−∞ − (−∞)` = NaN. A NaN compares equal to
    // nothing, so `partial_cmp` falls back to `Equal` and the score lands in the
    // ranking looking like a number. Refuse the ETA outright instead.
    let mean_eta = valid_eta.mean();
    let sst_eta = valid_eta
        .iter()
        .map(|&v| (v - mean_eta).powi(2))
        .sum::<f64>();
    // Two different failures, reported as two different things. Only NaN is filtered out of
    // `valid_eta`, so an EBE of ±∞ reaches here and makes the sum non-finite — telling that
    // user the column is "constant" would hide a fit blow-up behind a benign message.
    if !sst_eta.is_finite() {
        warnings.push(
            "the EBEs are not all finite, so the screening regressions cannot be fitted; \
             nothing screened."
                .into(),
        );
        return (f64::NAN, vec![], warnings);
    }
    if sst_eta <= 0.0 {
        warnings.push(
            "the EBEs are constant across subjects, so no covariate can explain them; \
             nothing screened."
                .into(),
        );
        return (f64::NAN, vec![], warnings);
    }

    // Global null model (intercept only) AIC.
    let n_all = valid_eta.len();
    let x_null_all = DMatrix::from_element(n_all, 1, 1.0);
    let (aic_null, _) = match ols_aic(&x_null_all, &valid_eta) {
        Some(v) => v,
        None => {
            warnings.push("the null model could not be fitted; nothing screened.".into());
            return (f64::NAN, vec![], warnings);
        }
    };

    let mut scores = Vec::with_capacity(covariates.len());

    for &(cov_name, cov_values, kind) in covariates {
        if cov_values.len() != eta_values.len() {
            warnings.push(format!(
                "{cov_name}: covariate column has {} entries but there are {} subjects; skipped.",
                cov_values.len(),
                eta_values.len()
            ));
            continue;
        }

        // Collect subjects with valid ETA and valid covariate.
        let pairs: Vec<(f64, f64)> = eta_values
            .iter()
            .zip(cov_values.iter())
            .filter(|(&e, &c)| !e.is_nan() && !c.is_nan())
            .map(|(&e, &c)| (e, c))
            .collect();

        let n = pairs.len();
        if n < 3 {
            warnings.push(format!(
                "{cov_name}: only {n} subject(s) have both an EBE and a value; skipped."
            ));
            continue;
        }

        let y = DVector::from_iterator(n, pairs.iter().map(|&(e, _)| e));
        let x_vals: Vec<f64> = pairs.iter().map(|&(_, c)| c).collect();

        // Per-covariate null AIC (only over subjects with valid covariate).
        let x_null = DMatrix::from_element(n, 1, 1.0);
        let (aic_null_local, _) = match ols_aic(&x_null, &y) {
            Some(v) => v,
            None => {
                warnings.push(format!(
                    "{cov_name}: the null model could not be fitted on its {n} subjects; skipped."
                ));
                continue;
            }
        };
        // The whole-column check above does not cover this: the EBEs can vary
        // across the population and still be constant over the subset that has
        // a value for *this* covariate, which puts the same NaN into ΔAIC.
        if !aic_null_local.is_finite() {
            // `−∞` is the constant-EBE case (RSS = 0); anything else non-finite is a blown-up
            // residual, which is a different problem and gets a different sentence.
            let cause = if aic_null_local == f64::NEG_INFINITY {
                "the EBEs are constant over its"
            } else {
                "the null model's residuals are not finite over its"
            };
            warnings.push(format!(
                "{cov_name}: {cause} {n} subjects, so ΔAIC is undefined; skipped."
            ));
            continue;
        }

        // Fit candidate forms and pick the one with the lowest AIC.
        let mut best_aic = f64::INFINITY;
        let mut best_r2 = 0.0;
        let mut best_form = CovariateForm::Linear; // overwritten below

        match kind {
            CovariateKind::Categorical => {
                // Count the levels BEFORE building anything: a label with one
                // level per subject would otherwise allocate an n × (n−1)
                // design — O(n²) — only to be rejected on the next line.
                let levels = categorical_levels(&x_vals);
                let n_dummies = levels.len().saturating_sub(1);
                let n_params = n_dummies + 1; // intercept + dummies

                // A single-level categorical carries no information, and a
                // high-cardinality one carries too much. For a label with `k`
                // levels over `n` subjects and no real signal,
                // `E[RSS/RSS₀] ≈ (n−k)/(n−1)`, so
                //
                //     ΔAIC ≈ n·ln((n−1)/(n−k)) − 2(k−1)
                //
                // which is negative for small `k` but crosses zero near
                // `k ≈ 0.8n` and reaches +474 at `k = n−1, n = 100`: the fit's
                // freedom to interpolate outruns the `2p` penalty, and a SITE
                // or STUDY identifier ranks above every real covariate. Refusing
                // only `k = n` (residual df ≥ 1) leaves that whole band open.
                //
                // So the design must leave at least as many residual degrees of
                // freedom as it spends parameters — `n ≥ 2p`, i.e. `k ≲ n/2`,
                // where the expression above is still comfortably negative
                // (≈ −30 at `k = n/2, n = 100`). Nothing of value is lost: a
                // label in the discarded band scores negative when AIC works at
                // all, so it would rank last either way.
                //
                // The threshold is a guard, not a scoring change. Switching to
                // AICc would handle this more smoothly but would move every
                // ΔAIC by ~0.1, breaking the R / xpose4 parity this module is
                // validated against.
                if n_dummies == 0 {
                    warnings.push(format!(
                        "{cov_name}: only one distinct level over the {n} screened subjects; \
                         skipped."
                    ));
                    continue;
                }
                if n < 2 * n_params {
                    warnings.push(format!(
                        "{cov_name}: {} levels over {n} subjects spends {n_params} parameters and \
                         leaves only {} residual degrees of freedom, too few for AIC to \
                         discriminate signal from interpolation; skipped.",
                        levels.len(),
                        n.saturating_sub(n_params)
                    ));
                    continue;
                }

                // Design: [intercept | dummies]
                let dummies = categorical_design(&x_vals, &levels);
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
                // Centre and scale before building any design matrix. AIC and
                // R² are invariant under an affine reparameterisation of the
                // design's column space (RSS and p are both unchanged), so the
                // ranking is untouched — but the truncated-power spline basis
                // cubes its input, so for a covariate of order 10²–10³ the raw
                // design spans ~10⁹ and the least-squares solve loses most of
                // its digits. R's `ns()` sidesteps this by returning a
                // QR-orthonormalised basis.
                let mean = x_vals.iter().sum::<f64>() / n as f64;
                let sd = (x_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
                if !sd.is_finite() || sd <= 0.0 {
                    warnings.push(format!(
                        "{cov_name}: constant over the {n} screened subjects; skipped."
                    ));
                    continue;
                }
                let z: Vec<f64> = x_vals.iter().map(|v| (v - mean) / sd).collect();

                // Linear form.
                if opts.include_linear {
                    let mut x_lin = DMatrix::zeros(n, 2);
                    for (row, &v) in z.iter().enumerate() {
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
                    // The same residual-degrees-of-freedom rule the categorical branch
                    // applies, for the same reason: `n > p` alone admits a fit with one
                    // residual df, where `n·ln(RSS/n)` dives and `2p` does not keep up.
                    // `spline_df` is a public `GamOptions` field, and even the default
                    // `[2, 3]` reaches this on a small population — `n = 5`, `df = 3` spends
                    // 4 parameters and leaves 1 — so a continuous covariate could reproduce
                    // the same ranking pathology a high-cardinality label does.
                    if df < 1 || n < 2 * (df + 1) {
                        continue;
                    }
                    let basis = ns_basis(&z, df);
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

        // `== INFINITY`, not `is_infinite()`: the sentinel is +∞ ("no form was
        // fitted"), but a perfect fit gives `n·ln(0) = −∞`, which is the
        // strongest signal the screen can produce. `is_infinite()` matched both
        // and dropped it.
        if best_aic == f64::INFINITY {
            warnings.push(format!(
                "{cov_name}: no candidate form could be fitted (singular design); skipped."
            ));
            continue;
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

    (aic_null, scores, warnings)
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

/// OLS fit: returns `(AIC, R²)`, or `None` when the design is rank-deficient
/// or leaves no residual degrees of freedom.
///
/// AIC = n·ln(RSS/n) + 2·p (Gaussian / least-squares AIC).
///
/// Solved by SVD of `X` rather than a Cholesky of `XᵀX`: forming the normal
/// equations squares the condition number, and the caller's spline design is a
/// truncated-power basis whose columns are cubes, so the normal equations lose
/// roughly twice the digits the problem needs. The SVD also gives an honest
/// rank test, which is what lets a singular design be reported as skipped
/// rather than silently scored.
///
/// `pub(crate)` for unit tests.
pub(crate) fn ols_aic(x: &DMatrix<f64>, y: &DVector<f64>) -> Option<(f64, f64)> {
    let n = y.len();
    let p = x.ncols();

    // Residual df ≥ 1. At n == p the fit is saturated, RSS is 0 and
    // `n·ln(RSS/n)` is −∞, so a saturated design would beat every honest one.
    if p == 0 || n <= p {
        return None;
    }

    let svd = x.clone().svd(true, true);
    let smax = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    if smax <= 0.0 || !smax.is_finite() {
        return None;
    }
    let tol = smax * (n.max(p) as f64) * f64::EPSILON;
    if svd.singular_values.iter().any(|&s| s <= tol) {
        return None; // rank-deficient design
    }

    let beta = svd.solve(y, tol).ok()?;
    let residuals = y - x * beta;
    let rss = residuals.norm_squared();
    let n_f = n as f64;
    let p_f = p as f64;
    let aic = n_f * (rss / n_f).ln() + 2.0 * p_f;
    let mean_y = y.mean();
    let sst = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>();
    let r2 = if sst < 1e-20 { 0.0 } else { 1.0 - rss / sst };
    Some((aic, r2))
}

/// The distinct levels of a categorical covariate, ascending.
///
/// Separate from [`categorical_design`] so a caller can judge the cardinality
/// before paying for an `n × (levels − 1)` matrix it may be about to reject.
///
/// `pub(crate)` for unit tests.
pub(crate) fn categorical_levels(x: &[f64]) -> Vec<f64> {
    let mut levels: Vec<f64> = x.to_vec();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    levels
}

/// One-hot encoding of a categorical covariate, shape `n × (levels − 1)`.
///
/// The lowest observed level is the reference (dropped). Returns an empty
/// matrix when there is only one unique level.
///
/// Takes the levels from [`categorical_levels`] rather than recomputing them,
/// so a caller can reject a design on its cardinality before paying to build
/// it — for a label with one level per subject that matrix is `n × (n − 1)`.
///
/// `pub(crate)` for unit tests.
pub(crate) fn categorical_design(x: &[f64], levels: &[f64]) -> DMatrix<f64> {
    if levels.len() <= 1 {
        return DMatrix::zeros(x.len(), 0);
    }

    let n_dummies = levels.len() - 1;
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
    }

    #[test]
    fn unavailable_shrinkage_warns_rather_than_staying_silent() {
        // A fit that reports no shrinkage for this ETA leaves the module's
        // central precondition unchecked, which is worth saying out loud —
        // silence there reads identically to "shrinkage is fine".
        let w = shrinkage_warning("ETA_CL", f64::NAN, 0.30)
            .expect("NaN shrinkage must produce its own warning");
        assert!(w.contains("ETA_CL"), "warning should name the ETA: {w}");
        assert!(
            w.contains("unavailable"),
            "warning should say shrinkage is unavailable: {w}"
        );
    }

    // ── screen_eta_raw ────────────────────────────────────────────────────────

    #[test]
    fn uncorrelated_cov_near_zero_delta_aic() {
        // Deterministic pseudo-random but unrelated eta and covariate.
        let eta: Vec<f64> = (0..30).map(|i| (i as f64 * 0.17 + 0.31).sin()).collect();
        let cov: Vec<f64> = (0..30).map(|i| (i as f64 * 0.13 + 0.71).cos()).collect();
        let opts = GamOptions::default();
        let cov_refs = [("COV", cov.as_slice(), CovariateKind::Continuous)];
        let (_aic_null, scores, _warns) = screen_eta_raw(&eta, &cov_refs, &opts);
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
        let (_aic_null, scores, _warns) = screen_eta_raw(&eta, &cov_refs, &opts);
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
        let (_aic_null, scores, _warns) = screen_eta_raw(&eta, &cov_refs, &opts);
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
        let (_aic_null, scores, _warns) = screen_eta_raw(&eta, &cov_refs, &opts);
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
        let (aic_null, scores, _warns) = screen_eta_raw(&eta, &cov_refs, &opts);
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
        let (_, scores_cl, _) = screen_eta_raw(&eta_cl, covs_cl, &opts);

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
        let (_, scores_v, _) = screen_eta_raw(&eta_v, covs_v, &opts);

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

    // ── ols_aic guards ────────────────────────────────────────────────────────

    #[test]
    fn ols_saturated_design_returns_none() {
        // n == p: the fit is saturated, RSS is 0 and `n·ln(RSS/n)` is −∞, so a
        // saturated design would beat every honest one on arithmetic alone.
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert!(ols_aic(&x, &y).is_none(), "n == p must be refused");
    }

    #[test]
    fn ols_empty_design_returns_none() {
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = DMatrix::zeros(3, 0);
        assert!(ols_aic(&x, &y).is_none());
    }

    #[test]
    fn ols_survives_an_ill_conditioned_design() {
        // Column of order 10³ alongside its cube: the normal equations span
        // ~10¹⁸ and a Cholesky of XᵀX loses the fit entirely. The SVD path
        // still recovers the exact least-squares residual.
        let n = 40;
        let xs: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64).collect();
        let y = DVector::from_iterator(n, xs.iter().map(|&v| 3.0 * v));
        let mut x = DMatrix::zeros(n, 3);
        for (row, &v) in xs.iter().enumerate() {
            x[(row, 0)] = 1.0;
            x[(row, 1)] = v;
            x[(row, 2)] = v.powi(3);
        }
        let (aic, r2) = ols_aic(&x, &y).expect("ill-conditioned but full-rank design must fit");
        assert!(aic.is_finite() || aic == f64::NEG_INFINITY, "got {aic}");
        assert!(
            r2 > 0.999,
            "an exact linear relation should give R² ≈ 1, got {r2}"
        );
    }

    // ── screen_eta_raw degenerate inputs ──────────────────────────────────────

    #[test]
    fn constant_covariate_is_skipped_with_a_warning() {
        let eta: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let cov = vec![70.0_f64; 20];
        let opts = GamOptions::default();
        let cov_refs = [("WT", cov.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(scores.is_empty(), "a constant covariate carries no signal");
        assert!(
            warns
                .iter()
                .any(|w| w.contains("WT") && w.contains("constant")),
            "the skip must be reported, not silent: {warns:?}"
        );
    }

    #[test]
    fn single_level_categorical_is_skipped_like_a_constant_continuous() {
        // The same degenerate case as above, on the other branch: it used to be
        // scored with delta_aic = 0.0 while the continuous twin was dropped.
        let eta: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let cov = vec![1.0_f64; 20];
        let opts = GamOptions::default();
        let cov_refs = [("SEX", cov.as_slice(), CovariateKind::Categorical)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            scores.is_empty(),
            "a one-level categorical carries no signal"
        );
        assert!(
            warns
                .iter()
                .any(|w| w.contains("SEX") && w.contains("one distinct level")),
            "the skip must be reported: {warns:?}"
        );
    }

    #[test]
    fn saturated_categorical_is_skipped_not_ranked_first() {
        // A label with one level per subject (a STUDY or SITE id) saturates the
        // design: RSS → 0, `n·ln(RSS/n)` dives, and the 2p penalty does not
        // keep up, so it would win the ranking outright against a covariate
        // that carries real signal.
        let n = 12;
        let wt: Vec<f64> = (0..n).map(|i| 60.0 + 2.0 * i as f64).collect();
        let eta: Vec<f64> = wt
            .iter()
            .enumerate()
            .map(|(i, &w)| 0.02 * (w - 70.0) + (i as f64 * 0.7).sin() * 0.05)
            .collect();
        let site: Vec<f64> = (0..n).map(|i| i as f64).collect(); // one level per subject
        let opts = GamOptions::default();
        let cov_refs = [
            ("WT", wt.as_slice(), CovariateKind::Continuous),
            ("SITE", site.as_slice(), CovariateKind::Categorical),
        ];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            !scores.iter().any(|s| s.covariate == "SITE"),
            "a saturated categorical must not be scored: {:?}",
            scores
                .iter()
                .map(|s| (&s.covariate, s.delta_aic))
                .collect::<Vec<_>>()
        );
        assert_eq!(scores.first().map(|s| s.covariate.as_str()), Some("WT"));
        assert!(
            warns
                .iter()
                .any(|w| w.contains("SITE") && w.contains("residual degrees of freedom")),
            "the skip must name the reason: {warns:?}"
        );
    }

    #[test]
    fn near_saturated_categorical_is_skipped_not_ranked_first() {
        // The `n > p` guard alone was not enough. A label with n−1 levels over
        // 100 unrelated ETA values has one residual degree of freedom, passes
        // that guard, and scores ΔAIC ≈ +474 with R² ≈ 0.999 — ranking a SITE
        // identifier far above a covariate carrying real signal.
        let n = 100;
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37 + 0.11).sin()).collect();
        let site: Vec<f64> = (0..n)
            .map(|i| if i == 0 { 1.0 } else { i as f64 })
            .collect();
        assert_eq!(
            categorical_levels(&site).len(),
            n - 1,
            "fixture must be near-saturated, not saturated"
        );
        let wt: Vec<f64> = (0..n).map(|i| 60.0 + 0.5 * i as f64).collect();

        let opts = GamOptions::default();
        let cov_refs = [
            ("SITE", site.as_slice(), CovariateKind::Categorical),
            ("WT", wt.as_slice(), CovariateKind::Continuous),
        ];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            !scores.iter().any(|s| s.covariate == "SITE"),
            "a near-saturated label must not be scored: {:?}",
            scores
                .iter()
                .map(|s| (&s.covariate, s.delta_aic))
                .collect::<Vec<_>>()
        );
        assert!(
            warns
                .iter()
                .any(|w| w.contains("SITE") && w.contains("residual degrees of freedom")),
            "the skip must name the reason: {warns:?}"
        );
    }

    #[test]
    fn a_categorical_with_room_to_spare_is_still_scored() {
        // The cardinality guard must not swallow ordinary factors: 3 levels
        // over 30 subjects spends 3 parameters and leaves 27.
        let n = 30;
        let grp: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let eta: Vec<f64> = grp
            .iter()
            .enumerate()
            .map(|(i, &g)| g * 0.4 + (i as f64 * 0.7).sin() * 0.02)
            .collect();
        let opts = GamOptions::default();
        let cov_refs = [("GRP", grp.as_slice(), CovariateKind::Categorical)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert_eq!(scores.len(), 1, "warnings: {warns:?}");
        assert_eq!(scores[0].best_form, CovariateForm::Categorical);
        assert!(scores[0].delta_aic > 5.0, "got {}", scores[0].delta_aic);
    }

    #[test]
    fn the_cardinality_guard_sits_exactly_at_two_parameters_per_subject() {
        // n = 20: 10 levels spends 10 parameters and is admitted; 11 levels
        // spends 11 and is not. Pins the boundary so a later "tidy-up" of the
        // inequality is a red test rather than a silent widening.
        let n = 20;
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31 + 0.2).sin()).collect();
        let ten: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let eleven: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();
        let opts = GamOptions::default();

        let (_, scores, _) = screen_eta_raw(
            &eta,
            &[("TEN", ten.as_slice(), CovariateKind::Categorical)],
            &opts,
        );
        assert_eq!(
            scores.len(),
            1,
            "10 levels over 20 subjects must be admitted"
        );

        let (_, scores, _) = screen_eta_raw(
            &eta,
            &[("ELEVEN", eleven.as_slice(), CovariateKind::Categorical)],
            &opts,
        );
        assert!(
            scores.is_empty(),
            "11 levels over 20 subjects must be refused"
        );
    }

    #[test]
    fn a_spline_is_refused_the_residual_df_a_categorical_is_refused() {
        // The guard has to be two-sided. `spline_df` is a public option and even the
        // default `[2, 3]` reaches this on a small population: 5 subjects with df = 3
        // spends 4 parameters and leaves 1 residual df, which is the same arithmetic that
        // let a 99-level label score +474.
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7 + 0.4).sin()).collect();
        let opts = GamOptions {
            spline_df: vec![3],
            include_linear: false,
            ..Default::default()
        };
        let cov_refs = [("X", x.as_slice(), CovariateKind::Continuous)];
        let (_, scores, _) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            scores.is_empty(),
            "df = 3 over 5 subjects leaves 1 residual df and must be refused, got {:?}",
            scores.iter().map(|s| s.delta_aic).collect::<Vec<_>>()
        );

        // Eight subjects spend the same 4 parameters and leave 4 — admitted.
        let n = 8;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 1.7 + 0.4).sin()).collect();
        let cov_refs = [("X", x.as_slice(), CovariateKind::Continuous)];
        let (_, scores, _) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert_eq!(scores.len(), 1, "df = 3 over 8 subjects must be admitted");
    }

    #[test]
    fn a_caller_supplied_spline_df_near_n_cannot_win_on_arithmetic() {
        // `GamOptions::spline_df` is public, so the pathological df is reachable without a
        // small dataset: 20 subjects and df = 15 leaves 4 residual df.
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 0.9 + 0.2).sin()).collect();
        let opts = GamOptions {
            spline_df: vec![15],
            include_linear: false,
            ..Default::default()
        };
        let cov_refs = [("X", x.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            scores.is_empty(),
            "a df spending three quarters of the subjects must be refused; warnings: {warns:?}"
        );
    }

    #[test]
    fn a_non_finite_ebe_is_not_reported_as_constant() {
        // Only NaN is filtered out of the EBE column, so ±∞ reaches the variance check and
        // makes it non-finite. Saying "constant" there hides a fit blow-up behind a
        // benign-sounding message.
        let mut eta: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        eta[7] = f64::INFINITY;
        let cov: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let opts = GamOptions::default();
        let cov_refs = [("X", cov.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(scores.is_empty());
        assert!(
            warns.iter().any(|w| w.contains("not all finite")),
            "a non-finite EBE must be named as such: {warns:?}"
        );
        assert!(
            !warns.iter().any(|w| w.contains("constant")),
            "a varying column must not be called constant: {warns:?}"
        );
    }

    #[test]
    fn constant_etas_produce_no_scores_and_a_warning() {
        // Every model including the null fits perfectly, so both AICs are −∞
        // and ΔAIC is `−∞ − (−∞)` = NaN — which sorted as "equal" and sat in
        // the ranking looking like a score.
        let eta = vec![0.0_f64; 20];
        let cov: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let opts = GamOptions::default();
        let cov_refs = [("X", cov.as_slice(), CovariateKind::Continuous)];
        let (aic_null, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(aic_null.is_nan());
        assert!(scores.is_empty(), "no covariate can explain a constant EBE");
        assert!(
            warns.iter().any(|w| w.contains("constant")),
            "the refusal must be reported: {warns:?}"
        );
    }

    #[test]
    fn constant_etas_over_one_covariates_subset_skip_only_that_covariate() {
        // The EBEs vary across the population but are constant over the
        // subjects that have a value for this covariate — same NaN, and not
        // caught by the whole-column check.
        let n = 24;
        let eta: Vec<f64> = (0..n)
            .map(|i| if i < 12 { 0.0 } else { i as f64 * 0.1 })
            .collect();
        let patchy: Vec<f64> = (0..n)
            .map(|i| if i < 12 { i as f64 } else { f64::NAN })
            .collect();
        let full: Vec<f64> = (0..n).map(|i| 70.0 + i as f64).collect();
        let opts = GamOptions::default();
        let cov_refs = [
            ("PATCHY", patchy.as_slice(), CovariateKind::Continuous),
            ("FULL", full.as_slice(), CovariateKind::Continuous),
        ];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            !scores.iter().any(|s| s.covariate == "PATCHY"),
            "a covariate whose subset has constant EBEs must be skipped"
        );
        assert!(
            scores.iter().any(|s| s.covariate == "FULL"),
            "the other covariate must still be screened"
        );
        assert!(
            warns
                .iter()
                .any(|w| w.contains("PATCHY") && w.contains("undefined")),
            "{warns:?}"
        );
    }

    #[test]
    fn no_score_is_ever_nan() {
        // The contract the ranking depends on: `sort_by` falls back to `Equal`
        // for a NaN, so a NaN ΔAIC is not merely wrong, it is unordered.
        let n = 40;
        let eta: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let flat = vec![1.0_f64; n];
        let ramp: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let label: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let opts = GamOptions::default();
        let cov_refs = [
            ("FLAT", flat.as_slice(), CovariateKind::Continuous),
            ("RAMP", ramp.as_slice(), CovariateKind::Continuous),
            ("LABEL", label.as_slice(), CovariateKind::Categorical),
        ];
        let (_, scores, _) = screen_eta_raw(&eta, &cov_refs, &opts);
        for s in &scores {
            assert!(!s.delta_aic.is_nan(), "{} produced a NaN ΔAIC", s.covariate);
        }
    }

    #[test]
    fn perfect_fit_is_reported_rather_than_dropped() {
        // RSS = 0 gives AIC = n·ln(0) = −∞. The "no form fitted" sentinel is
        // +∞, and testing it with `is_infinite()` matched both — so the
        // strongest signal the screen can produce disappeared from the report.
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let eta: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let opts = GamOptions {
            spline_df: vec![],
            ..Default::default()
        };
        let cov_refs = [("X", x.as_slice(), CovariateKind::Continuous)];
        let (_, scores, _) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert_eq!(scores.len(), 1, "an exact relation must still be reported");
        assert!(
            scores[0].delta_aic > 0.0,
            "an exact relation must rank as a strong improvement, got {}",
            scores[0].delta_aic
        );
    }

    #[test]
    fn ill_conditioned_covariate_still_ranks() {
        // CRCL-scale values (order 10²) fed to a truncated-power basis whose
        // columns are cubes: the uncentred design ran to ~10⁹ and the fit came
        // back singular, dropping a covariate that carries obvious signal.
        let n = 40;
        let crcl: Vec<f64> = (0..n).map(|i| 400.0 + 3.0 * i as f64).collect();
        let eta: Vec<f64> = crcl
            .iter()
            .enumerate()
            .map(|(i, &c)| 0.004 * (c - 460.0) + (i as f64 * 0.9).sin() * 0.02)
            .collect();
        let opts = GamOptions::default();
        let cov_refs = [("CRCL", crcl.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert_eq!(
            scores.len(),
            1,
            "covariate was dropped; warnings: {warns:?}"
        );
        assert!(
            scores[0].delta_aic > 10.0,
            "a strong signal at CRCL scale should survive, got {}",
            scores[0].delta_aic
        );
    }

    #[test]
    fn mismatched_covariate_column_is_skipped_with_a_warning() {
        // `zip` would silently truncate to the shorter of the two and score a
        // different subject set than the caller asked for.
        let eta: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let cov: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let opts = GamOptions::default();
        let cov_refs = [("SHORT", cov.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(scores.is_empty());
        assert!(
            warns
                .iter()
                .any(|w| w.contains("SHORT") && w.contains("15")),
            "the length mismatch must be reported: {warns:?}"
        );
    }

    #[test]
    fn all_missing_covariate_is_skipped_with_a_warning() {
        let eta: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let cov = vec![f64::NAN; 20];
        let opts = GamOptions::default();
        let cov_refs = [("GONE", cov.as_slice(), CovariateKind::Continuous)];
        let (_, scores, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(scores.is_empty());
        assert!(
            warns.iter().any(|w| w.contains("GONE")),
            "an all-missing covariate must be reported: {warns:?}"
        );
    }

    #[test]
    fn too_few_subjects_reports_why() {
        let eta = vec![1.0, 2.0];
        let cov = vec![0.0, 1.0];
        let opts = GamOptions::default();
        let cov_refs = [("X", cov.as_slice(), CovariateKind::Continuous)];
        let (_, _, warns) = screen_eta_raw(&eta, &cov_refs, &opts);
        assert!(
            warns.iter().any(|w| w.contains("at least 3")),
            "an empty result must say why: {warns:?}"
        );
    }

    // ── gam_screen_raw ────────────────────────────────────────────────────────

    fn two_eta_screen_inputs() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 40;
        let wt: Vec<f64> = (0..n).map(|i| 60.0 + i as f64).collect();
        let noise: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 1.7 + 0.4).cos() * 10.0)
            .collect();
        // ETA_CL tracks WT; ETA_V tracks nothing.
        let eta_cl: Vec<f64> = wt
            .iter()
            .enumerate()
            .map(|(i, &w)| 0.03 * (w - 80.0) + (i as f64 * 0.9).sin() * 0.05)
            .collect();
        let eta_v: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.41 + 0.2).sin() * 0.2)
            .collect();
        (wt, noise, eta_cl, eta_v)
    }

    #[test]
    fn gam_screen_raw_ranks_each_eta_independently() {
        let (wt, noise, eta_cl, eta_v) = two_eta_screen_inputs();
        let opts = GamOptions::default();
        let result = gam_screen_raw(
            &["ETA_CL", "ETA_V"],
            &[eta_cl.as_slice(), eta_v.as_slice()],
            &[0.05, 0.10],
            &["WT", "NOISE"],
            &[wt.as_slice(), noise.as_slice()],
            &[CovariateKind::Continuous, CovariateKind::Continuous],
            &opts,
        );

        assert_eq!(result.eta_results.len(), 2);
        let cl = &result.eta_results[0];
        assert_eq!(cl.eta_name, "ETA_CL");
        assert_eq!(cl.shrinkage, 0.05);
        assert!(cl.aic_null.is_finite());
        assert_eq!(
            cl.covariate_scores[0].covariate, "WT",
            "WT must rank first for the ETA it drives"
        );
        assert!(cl.covariate_scores[0].delta_aic > cl.covariate_scores[1].delta_aic);

        let v = &result.eta_results[1];
        assert_eq!(v.eta_name, "ETA_V");
        assert!(
            v.covariate_scores.iter().all(|s| s.delta_aic < 5.0),
            "no covariate drives ETA_V: {:?}",
            v.covariate_scores
                .iter()
                .map(|s| (&s.covariate, s.delta_aic))
                .collect::<Vec<_>>()
        );
        // Low shrinkage on both ETAs, so nothing to warn about.
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn gam_screen_raw_propagates_shrinkage_and_skip_warnings() {
        let (wt, _noise, eta_cl, _eta_v) = two_eta_screen_inputs();
        let flat = vec![70.0_f64; wt.len()];
        let opts = GamOptions::default();
        let result = gam_screen_raw(
            &["ETA_CL"],
            &[eta_cl.as_slice()],
            &[0.62], // above the 30% default threshold
            &["WT", "FLAT"],
            &[wt.as_slice(), flat.as_slice()],
            &[CovariateKind::Continuous, CovariateKind::Continuous],
            &opts,
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("shrinkage 62.0%")),
            "{:?}",
            result.warnings
        );
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.starts_with("ETA_CL:") && w.contains("FLAT")),
            "a per-covariate skip must be attributed to its ETA: {:?}",
            result.warnings
        );
    }

    #[test]
    #[should_panic(expected = "eta_names and shrinkage must have the same length")]
    fn gam_screen_raw_rejects_a_short_shrinkage_vector() {
        let (wt, _, eta_cl, eta_v) = two_eta_screen_inputs();
        let opts = GamOptions::default();
        let _ = gam_screen_raw(
            &["ETA_CL", "ETA_V"],
            &[eta_cl.as_slice(), eta_v.as_slice()],
            &[0.05], // one short
            &["WT"],
            &[wt.as_slice()],
            &[CovariateKind::Continuous],
            &opts,
        );
    }

    #[test]
    #[should_panic(expected = "cov_cols[0] has")]
    fn gam_screen_raw_rejects_a_short_covariate_column() {
        let (wt, _, eta_cl, _) = two_eta_screen_inputs();
        let opts = GamOptions::default();
        let _ = gam_screen_raw(
            &["ETA_CL"],
            &[eta_cl.as_slice()],
            &[0.05],
            &["WT"],
            &[&wt[..wt.len() - 1]], // one subject short
            &[CovariateKind::Continuous],
            &opts,
        );
    }

    // ── determine_cov_kind ────────────────────────────────────────────────────

    fn subject_with(id: &str, covs: &[(&str, f64)]) -> ferx_core::Subject {
        let mut s = ferx_core::Subject {
            id: id.to_string(),
            ..Default::default()
        };
        for &(name, value) in covs {
            s.covariates.insert(name.to_string(), value);
        }
        s
    }

    #[test]
    fn declared_kind_beats_the_heuristic() {
        // WT is near-integer with few levels here, so the heuristic would call
        // it categorical — the `[covariates]` declaration must win.
        let subjects = vec![
            subject_with("1", &[("WT", 70.0)]),
            subject_with("2", &[("WT", 80.0)]),
            subject_with("3", &[("WT", 70.0)]),
        ];
        let table = Some(CovariateTable {
            names: vec!["WT".into()],
            kinds: vec![CovariateKind::Continuous],
            rows: vec![],
        });
        assert_eq!(
            determine_cov_kind("WT", &table, &subjects),
            CovariateKind::Continuous
        );
        // A name the table does not declare still falls through to the heuristic.
        assert_eq!(
            determine_cov_kind("MISSING", &table, &subjects),
            CovariateKind::Continuous
        );
    }

    #[test]
    fn heuristic_calls_few_valued_integers_categorical() {
        let subjects: Vec<_> = (0..12)
            .map(|i| subject_with(&i.to_string(), &[("SEX", (i % 2) as f64)]))
            .collect();
        assert_eq!(
            determine_cov_kind("SEX", &None, &subjects),
            CovariateKind::Categorical
        );
    }

    #[test]
    fn heuristic_calls_many_valued_integers_continuous() {
        // > 10 distinct near-integer levels: an age column, not a label.
        let subjects: Vec<_> = (0..20)
            .map(|i| subject_with(&i.to_string(), &[("AGE", 20.0 + i as f64)]))
            .collect();
        assert_eq!(
            determine_cov_kind("AGE", &None, &subjects),
            CovariateKind::Continuous
        );
    }

    #[test]
    fn heuristic_calls_non_integers_continuous_and_survives_no_values() {
        let subjects = vec![
            subject_with("1", &[("CRCL", 88.4)]),
            subject_with("2", &[("CRCL", 91.2)]),
            subject_with("3", &[("CRCL", f64::NAN)]),
        ];
        assert_eq!(
            determine_cov_kind("CRCL", &None, &subjects),
            CovariateKind::Continuous
        );
        // A name no subject carries has no values to judge.
        assert_eq!(
            determine_cov_kind("ABSENT", &None, &subjects),
            CovariateKind::Continuous
        );
    }

    // ── screen_from_parts (the body of `gam_screen`) ──────────────────────────

    fn population_from(subjects: Vec<ferx_core::Subject>, cov_names: &[&str]) -> Population {
        Population {
            subjects,
            covariate_names: cov_names.iter().map(|s| s.to_string()).collect(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// A 40-subject population whose ETA_CL tracks WT and whose ETA_V tracks
    /// nothing, laid out the way `gam_screen` receives it from a `FitResult`.
    fn screen_fixture() -> (Vec<String>, Vec<f64>, Vec<Vec<f64>>, Population) {
        let (wt, noise, eta_cl, eta_v) = two_eta_screen_inputs();
        let subjects: Vec<_> = (0..wt.len())
            .map(|i| subject_with(&i.to_string(), &[("WT", wt[i]), ("NOISE", noise[i])]))
            .collect();
        let eta_rows: Vec<Vec<f64>> = (0..wt.len()).map(|i| vec![eta_cl[i], eta_v[i]]).collect();
        (
            vec!["ETA_CL".into(), "ETA_V".into()],
            vec![0.05, 0.10],
            eta_rows,
            population_from(subjects, &["WT", "NOISE"]),
        )
    }

    #[test]
    fn screen_from_parts_ranks_the_driving_covariate_first() {
        let (names, shrink, eta_rows, pop) = screen_fixture();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let result = screen_from_parts(&names, &shrink, &rows, &None, &pop, &GamOptions::default());

        assert_eq!(result.eta_results.len(), 2);
        assert_eq!(result.eta_results[0].eta_name, "ETA_CL");
        assert_eq!(result.eta_results[0].covariate_scores[0].covariate, "WT");
        assert_eq!(result.eta_results[0].shrinkage, 0.05);
        assert_eq!(result.eta_results[1].eta_name, "ETA_V");
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn screen_from_parts_honours_the_eta_and_covariate_filters() {
        let (names, shrink, eta_rows, pop) = screen_fixture();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let opts = GamOptions {
            etas: Some(vec!["ETA_V".into()]),
            covariates: Some(vec!["NOISE".into()]),
            ..Default::default()
        };
        let result = screen_from_parts(&names, &shrink, &rows, &None, &pop, &opts);

        assert_eq!(result.eta_results.len(), 1);
        assert_eq!(result.eta_results[0].eta_name, "ETA_V");
        // Shrinkage is taken by ETA *name*, not by filtered position.
        assert_eq!(result.eta_results[0].shrinkage, 0.10);
        assert_eq!(result.eta_results[0].covariate_scores.len(), 1);
        assert_eq!(result.eta_results[0].covariate_scores[0].covariate, "NOISE");
    }

    #[test]
    fn screen_from_parts_reports_names_it_could_not_find() {
        let (names, shrink, eta_rows, pop) = screen_fixture();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let opts = GamOptions {
            etas: Some(vec!["ETA_CL".into(), "ETA_KA".into()]),
            covariates: Some(vec!["WT".into(), "CRCL".into()]),
            ..Default::default()
        };
        let result = screen_from_parts(&names, &shrink, &rows, &None, &pop, &opts);

        assert_eq!(result.eta_results.len(), 1, "ETA_KA is not in the fit");
        assert!(
            result.warnings.iter().any(|w| w.contains("ETA_KA")),
            "{:?}",
            result.warnings
        );
        assert!(
            result.warnings.iter().any(|w| w.contains("CRCL")),
            "a covariate no subject carries must be reported, not screened as \
             all-missing: {:?}",
            result.warnings
        );
        assert_eq!(result.eta_results[0].covariate_scores.len(), 1);
    }

    #[test]
    fn screen_from_parts_uses_the_declared_covariate_kinds() {
        let n = 30;
        let grp: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let eta: Vec<f64> = grp
            .iter()
            .enumerate()
            .map(|(i, &g)| g * 0.5 + (i as f64 * 0.7).sin() * 0.02)
            .collect();
        let subjects: Vec<_> = (0..n)
            .map(|i| subject_with(&i.to_string(), &[("GRP", grp[i])]))
            .collect();
        let pop = population_from(subjects, &["GRP"]);
        let eta_rows: Vec<Vec<f64>> = eta.iter().map(|&e| vec![e]).collect();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let table = Some(CovariateTable {
            names: vec!["GRP".into()],
            kinds: vec![CovariateKind::Categorical],
            rows: vec![],
        });

        let result = screen_from_parts(
            &["ETA_CL".to_string()],
            &[0.05],
            &rows,
            &table,
            &pop,
            &GamOptions::default(),
        );
        assert_eq!(
            result.eta_results[0].covariate_scores[0].best_form,
            CovariateForm::Categorical
        );
    }

    #[test]
    fn screen_from_parts_reports_an_empty_covariate_set() {
        let (names, shrink, eta_rows, _) = screen_fixture();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let empty = population_from(vec![], &[]);
        let result = screen_from_parts(
            &names,
            &shrink,
            &rows,
            &None,
            &empty,
            &GamOptions::default(),
        );
        assert!(result.eta_results.is_empty());
        assert!(
            result.warnings.iter().any(|w| w.contains("No covariates")),
            "{:?}",
            result.warnings
        );
    }

    #[test]
    fn screen_from_parts_reports_an_empty_eta_set() {
        let (_, _, eta_rows, pop) = screen_fixture();
        let rows: Vec<&[f64]> = eta_rows.iter().map(|r| r.as_slice()).collect();
        let result = screen_from_parts(&[], &[], &rows, &None, &pop, &GamOptions::default());
        assert!(result.eta_results.is_empty());
        assert!(
            result.warnings.iter().any(|w| w.contains("No ETAs")),
            "{:?}",
            result.warnings
        );
    }

    #[test]
    fn screen_from_parts_marks_a_missing_ebe_column_as_unavailable() {
        // A fit whose per-subject EBE vector is shorter than `eta_names` (or
        // whose shrinkage vector is) must produce NaNs and say so, not index
        // out of bounds.
        let (names, _, eta_rows, pop) = screen_fixture();
        let short: Vec<Vec<f64>> = eta_rows.iter().map(|r| vec![r[0]]).collect();
        let rows: Vec<&[f64]> = short.iter().map(|r| r.as_slice()).collect();
        let result = screen_from_parts(&names, &[], &rows, &None, &pop, &GamOptions::default());

        assert_eq!(result.eta_results.len(), 2);
        assert!(result.eta_results[1].aic_null.is_nan());
        assert!(result.eta_results[1].covariate_scores.is_empty());
        assert!(
            result.warnings.iter().any(|w| w.contains("unavailable")),
            "missing shrinkage must be reported: {:?}",
            result.warnings
        );
    }
}
