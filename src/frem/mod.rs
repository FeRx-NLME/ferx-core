//! FREM (Full Random Effects Model) data and model transformation.
//!
//! Transforms a base PK model + dataset into a FREM model that treats
//! covariates as additional dependent variables. The covariance structure
//! of an extended omega matrix captures covariate-parameter relationships
//! implicitly.
//!
//! # Workflow
//!
//! 1. Parse the base model + read the base dataset
//! 2. [`transform_dataset_for_frem`] — augment dataset with covariate pseudo-observations
//! 3. [`generate_frem_model`] — write a new `.ferx` model file with extended parameters
//! 4. Fit the resulting FREM model normally

use crate::types::{CompiledModel, Population};
use nalgebra::DMatrix;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// Prior-fit parameter values used to seed a generated FREM model's initial
/// values, instead of the base model's declared inits (issue #239). Typically
/// built from a completed [`crate::types::FitResult`] of the base model: its
/// `theta`/`theta_names` and `omega`/`eta_names` carry exactly this shape.
///
/// Names are matched case-insensitively against the base model's declared
/// theta/eta names; a name with no match is left at its declared init value,
/// so a fit from a slightly different model (extra or missing parameters)
/// degrades gracefully rather than erroring.
#[derive(Debug, Clone)]
pub struct FremFitInit {
    /// Theta name -> fitted value.
    pub theta: Vec<(String, f64)>,
    /// Eta names, parallel to the rows/columns of `omega`.
    pub eta_names: Vec<String>,
    /// Fitted omega (BSV) covariance matrix, indexed by `eta_names`.
    pub omega: DMatrix<f64>,
}

/// Statistics and metadata from the FREM data transformation.
#[derive(Debug, Clone)]
pub struct FremDataInfo {
    /// Final covariate names (including binarized categoricals).
    pub covariate_names: Vec<String>,
    /// Population mean of each covariate.
    pub covariate_means: Vec<f64>,
    /// Population variance of each covariate.
    pub covariate_variances: Vec<f64>,
    /// FREMTYPE value for each covariate (100, 200, 300, ...).
    pub fremtype_map: Vec<(String, u16)>,
    /// Number of PK etas in the base model.
    pub n_base_etas: usize,
}

/// Result of [`prepare_frem`].
#[derive(Debug, Clone)]
pub struct FremPrepareResult {
    pub model_path: std::path::PathBuf,
    pub data_path: std::path::PathBuf,
    pub covariate_means: Vec<(String, f64)>,
    pub covariate_variances: Vec<(String, f64)>,
    pub fremtype_map: Vec<(String, u16)>,
    pub n_total_etas: usize,
    /// Advisory messages surfaced at conversion time (e.g. estimated parameters
    /// without a random effect, which IMP/IMPMAP estimate poorly — see #406).
    pub warnings: Vec<String>,
}

/// Information about how a categorical covariate is expanded into indicators.
#[derive(Debug, Clone)]
struct CategoricalExpansion {
    /// Original covariate name (e.g., "RACE").
    original_name: String,
    /// Reference level (most frequent; omitted from indicators).
    _reference_level: f64,
    /// Non-reference levels, sorted ascending. Each becomes an indicator column.
    indicator_levels: Vec<f64>,
    /// Indicator column names (e.g., ["RACE_2", "RACE_3"]).
    indicator_names: Vec<String>,
}

/// Transform a dataset for FREM by adding covariate pseudo-observation rows.
///
/// For each subject, inserts one row per covariate with:
/// - TIME = time of first observation
/// - DV = covariate value
/// - EVID = 0, MDV = 0, AMT = 0
/// - FREMTYPE = covariate index * 100 (100, 200, 300, ...)
///
/// Categorical covariates with K > 2 levels are automatically binarized into
/// K-1 indicator variables (matching PsN/NONMEM convention). The most frequent
/// level is used as the reference. Binary categoricals (K = 2) are kept as-is.
///
/// Missing covariate values (default: -99) are handled by:
/// - Excluding them from population mean/variance computation
/// - Omitting the pseudo-observation row for that subject/covariate
/// This matches PsN/NONMEM FREM behaviour where the omega correlation structure
/// effectively imputes missing covariates.
///
/// Returns the augmented CSV content (as a string) and metadata.
pub fn transform_dataset_for_frem(
    population: &Population,
    base_model: &CompiledModel,
    covariates: &[String],
    categorical_covariates: &[String],
    missing_value: Option<f64>,
) -> Result<(String, FremDataInfo), String> {
    let missing_val = missing_value.unwrap_or(-99.0);
    // Validate covariates exist in the dataset.
    for cov in covariates {
        let found = population
            .covariate_names
            .iter()
            .any(|n| n.eq_ignore_ascii_case(cov));
        if !found {
            return Err(format!(
                "FREM covariate '{}' not found in dataset (available: {:?})",
                cov, population.covariate_names
            ));
        }
    }

    // ── Categorical expansion ──────────────────────────────────────
    // For each categorical covariate with K > 2 levels, expand into K-1
    // indicator variables. Binary (K=2) categoricals are kept as-is.
    let mut expansions: HashMap<String, CategoricalExpansion> = HashMap::new();
    for cat_name in categorical_covariates {
        // Collect unique levels across all subjects, excluding missing values.
        let mut level_counts: HashMap<i64, usize> = HashMap::new();
        for subj in &population.subjects {
            if let Some(&val) = subj.covariates.get(cat_name) {
                if val.is_finite() && (val - missing_val).abs() > 0.5 {
                    *level_counts.entry(val as i64).or_insert(0) += 1;
                }
            }
        }
        let n_levels = level_counts.len();
        if n_levels <= 2 {
            continue; // Binary or constant — keep as-is, no expansion needed.
        }

        // Reference level = most frequent.
        let reference = *level_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .unwrap()
            .0;

        // Non-reference levels, sorted ascending.
        let mut other_levels: Vec<i64> = level_counts
            .keys()
            .copied()
            .filter(|&l| l != reference)
            .collect();
        other_levels.sort();

        let indicator_names: Vec<String> = other_levels
            .iter()
            .map(|l| format!("{}_{}", cat_name, l))
            .collect();
        let indicator_levels: Vec<f64> = other_levels.iter().map(|&l| l as f64).collect();

        expansions.insert(
            cat_name.clone(),
            CategoricalExpansion {
                original_name: cat_name.clone(),
                _reference_level: reference as f64,
                indicator_levels,
                indicator_names,
            },
        );
    }

    // Build the final expanded covariate list: replace expanded categoricals
    // with their indicators, keep everything else as-is.
    let mut expanded_covariates: Vec<String> = Vec::new();
    for cov in covariates {
        if let Some(exp) = expansions.get(cov) {
            expanded_covariates.extend(exp.indicator_names.iter().cloned());
        } else {
            expanded_covariates.push(cov.clone());
        }
    }

    // Helper: look up the value of an expanded covariate for a subject.
    // Returns None if the covariate is missing (equals missing_val).
    let get_expanded_value = |subj: &crate::types::Subject, exp_name: &str| -> Option<f64> {
        // Check if this is an indicator from an expansion.
        for exp in expansions.values() {
            if let Some(pos) = exp.indicator_names.iter().position(|n| n == exp_name) {
                let original_val = subj.covariates.get(&exp.original_name)?;
                // If the original categorical is missing, all indicators are missing.
                if (original_val - missing_val).abs() < 0.5 {
                    return None;
                }
                let level = exp.indicator_levels[pos];
                return Some(if (*original_val - level).abs() < 0.5 {
                    1.0
                } else {
                    0.0
                });
            }
        }
        // Not an indicator — look up the original covariate.
        let val = subj.covariates.get(exp_name).copied()?;
        if (val - missing_val).abs() < 0.5 {
            None // Continuous covariate is missing
        } else {
            Some(val)
        }
    };

    // Collect expanded covariate values across all subjects for statistics.
    let n_exp = expanded_covariates.len();
    let mut cov_values: Vec<Vec<f64>> = vec![Vec::new(); n_exp];
    for subject in &population.subjects {
        for (k, exp_name) in expanded_covariates.iter().enumerate() {
            if let Some(val) = get_expanded_value(subject, exp_name) {
                if val.is_finite() {
                    cov_values[k].push(val);
                }
            }
        }
    }

    // Compute means and variances.
    let mut covariate_means = Vec::with_capacity(n_exp);
    let mut covariate_variances = Vec::with_capacity(n_exp);
    for (k, vals) in cov_values.iter().enumerate() {
        if vals.is_empty() {
            return Err(format!(
                "FREM covariate '{}' has no valid (finite) values across subjects",
                expanded_covariates[k]
            ));
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var = if vals.len() > 1 {
            vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
        } else {
            1.0 // fallback for single-subject
        };
        covariate_means.push(mean);
        covariate_variances.push(if var > 1e-10 { var } else { 1e-10 }); // guard against zero variance
    }

    // Build FREMTYPE map for expanded covariates.
    let fremtype_map: Vec<(String, u16)> = expanded_covariates
        .iter()
        .enumerate()
        .map(|(k, name)| (name.clone(), (k as u16 + 1) * 100))
        .collect();

    // Collect indicator column names that need to be added to the CSV.
    let mut indicator_col_names: Vec<String> = Vec::new();
    for exp in expansions.values() {
        for name in &exp.indicator_names {
            indicator_col_names.push(name.clone());
        }
    }
    indicator_col_names.sort(); // deterministic column order

    // Build augmented CSV.
    // Header: ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,II,SS,CENS,FREMTYPE,<original_covariates>,<indicator_cols>
    let mut csv = String::new();
    let mut header_parts = vec![
        "ID".to_string(),
        "TIME".to_string(),
        "DV".to_string(),
        "EVID".to_string(),
        "AMT".to_string(),
        "CMT".to_string(),
        "RATE".to_string(),
        "MDV".to_string(),
        "II".to_string(),
        "SS".to_string(),
        "CENS".to_string(),
        "FREMTYPE".to_string(),
    ];
    for cov_name in &population.covariate_names {
        header_parts.push(cov_name.clone());
    }
    for ind_name in &indicator_col_names {
        header_parts.push(ind_name.clone());
    }
    csv.push_str(&header_parts.join(","));
    csv.push('\n');

    // Helper to append original + indicator covariate columns for a subject row.
    let append_cov_columns = |row: &mut Vec<String>, subj: &crate::types::Subject| {
        for cov_name in &population.covariate_names {
            row.push(
                subj.covariates
                    .get(cov_name)
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|| ".".to_string()),
            );
        }
        for ind_name in &indicator_col_names {
            row.push(
                get_expanded_value(subj, ind_name)
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|| ".".to_string()),
            );
        }
    };

    for subject in &population.subjects {
        let first_obs_time = subject.obs_times.first().copied().unwrap_or(0.0);

        // Collect all records for this subject, then sort by time.
        // Each record: (time, evid_priority, row_string)
        // evid_priority: 0=dose (EVID=1), 1=cov pseudo-obs, 2=PK obs
        // Within the same time, doses come first, then pseudo-obs, then PK obs.
        let mut records: Vec<(f64, u8, String)> = Vec::new();

        // Dose rows.
        for dose in &subject.doses {
            let mut row = vec![
                subject.id.clone(),
                format!("{}", dose.time),
                ".".to_string(),
                "1".to_string(), // EVID
                format!("{}", dose.amt),
                format!("{}", dose.cmt),
                format!("{}", dose.rate),
                "1".to_string(), // MDV
                if dose.ii > 0.0 {
                    format!("{}", dose.ii)
                } else {
                    "0".to_string()
                }, // II
                if dose.ss {
                    "1".to_string()
                } else {
                    "0".to_string()
                }, // SS
                "0".to_string(), // CENS
                "0".to_string(), // FREMTYPE
            ];
            append_cov_columns(&mut row, subject);
            records.push((dose.time, 0, row.join(",")));
        }

        // Covariate pseudo-observation rows.
        // Skip pseudo-obs for covariates where the subject has a missing value.
        for (k, exp_name) in expanded_covariates.iter().enumerate() {
            let cov_val = match get_expanded_value(subject, exp_name) {
                Some(v) => v,
                None => continue, // Missing covariate — omit pseudo-obs row
            };
            let ft = fremtype_map[k].1;
            let mut row = vec![
                subject.id.clone(),
                format!("{}", first_obs_time),
                format!("{}", cov_val),
                "0".to_string(), // EVID
                "0".to_string(), // AMT
                "1".to_string(), // CMT (will be overridden by FREMTYPE dispatch)
                "0".to_string(), // RATE
                "0".to_string(), // MDV
                "0".to_string(), // II
                "0".to_string(), // SS
                "0".to_string(), // CENS
                format!("{}", ft),
            ];
            append_cov_columns(&mut row, subject);
            records.push((first_obs_time, 1, row.join(",")));
        }

        // Original PK observation rows.
        for (j, (&time, &dv)) in subject
            .obs_times
            .iter()
            .zip(subject.observations.iter())
            .enumerate()
        {
            let cmt = subject.obs_cmts.get(j).copied().unwrap_or(1);
            let cens_flag = subject.cens.get(j).copied().unwrap_or(0);
            let mut row = vec![
                subject.id.clone(),
                format!("{}", time),
                format!("{}", dv),
                "0".to_string(), // EVID
                "0".to_string(), // AMT
                format!("{}", cmt),
                "0".to_string(),          // RATE
                "0".to_string(),          // MDV
                "0".to_string(),          // II
                "0".to_string(),          // SS
                format!("{}", cens_flag), // CENS
                "0".to_string(),          // FREMTYPE (PK observation)
            ];
            append_cov_columns(&mut row, subject);
            records.push((time, 2, row.join(",")));
        }

        // Sort by (time, evid_priority) to ensure chronological order.
        // Within the same time: doses first, then pseudo-obs, then PK obs.
        records.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });

        for (_, _, row_str) in &records {
            csv.push_str(row_str);
            csv.push('\n');
        }
    }

    let info = FremDataInfo {
        covariate_names: expanded_covariates,
        covariate_means,
        covariate_variances,
        fremtype_map,
        n_base_etas: base_model.n_eta,
    };

    Ok((csv, info))
}

/// Generate a FREM `.ferx` model file from a base model and FREM metadata.
///
/// The generated model extends the base with:
/// - Fixed theta for each covariate's typical value (TV_COV)
/// - Extended block omega with PK + covariate etas
/// - Fixed near-zero sigma for covariate observations (EPSCOV)
/// - Individual parameters for covariates: `COV_X = TV_X + ETA_X_FREM`
/// - FREM fit options mapping FREMTYPE → (theta, eta) predictions
pub fn generate_frem_model(
    base_model_text: &str,
    base_model: &CompiledModel,
    frem_info: &FremDataInfo,
    output_data_path: &Path,
    fit_init: Option<&FremFitInit>,
) -> Result<String, String> {
    // Parse the base model text to extract blocks.
    let blocks = parse_blocks(base_model_text)?;

    let mut model = String::new();

    // ── [parameters] block ──
    model.push_str("# FREM model (auto-generated)\n\n");
    model.push_str("[parameters]\n");

    // Copy base thetas, substituting the init value with the prior fit's
    // estimate when one is supplied (issue #239) so a subsequent fit of the
    // FREM model warm-starts from converged PK parameters instead of the
    // original declared inits.
    let theta_init_re = Regex::new(r"(?i)^\s*theta\s+(\w+)\s*\(\s*([0-9eE.+-]+)").unwrap();
    if let Some(params_lines) = blocks.get("parameters") {
        for line in params_lines {
            let trimmed = line.trim();
            if trimmed.starts_with("theta ") {
                let out = match fit_init {
                    Some(fi) => override_theta_init(trimmed, fi, &theta_init_re),
                    None => trimmed.to_string(),
                };
                model.push_str(&format!("  {}\n", out));
            }
        }
    }

    // Add fixed covariate thetas.
    for (k, cov_name) in frem_info.covariate_names.iter().enumerate() {
        let mean = frem_info.covariate_means[k];
        model.push_str(&format!(
            "  theta TV_{}({}, FIX)\n",
            cov_name.to_uppercase(),
            mean
        ));
    }
    model.push('\n');

    // Build extended block_omega: PK etas + covariate etas.
    let n_pk = base_model.n_eta;
    let n_cov = frem_info.covariate_names.len();
    let n_total = n_pk + n_cov;

    let mut all_eta_names: Vec<String> = base_model.eta_names.clone();
    for cov_name in &frem_info.covariate_names {
        all_eta_names.push(format!("ETA_{}_FREM", cov_name.to_uppercase()));
    }

    // Build the omega matrix: PK diagonal from base, COV diagonal from variance,
    // off-diagonals = small scaled values.
    let mut omega_matrix = vec![vec![0.0f64; n_total]; n_total];
    // PK-PK block from base model, overridden entry-by-entry with the prior
    // fit's omega (issue #239) when a matching eta pair is present.
    for i in 0..n_pk {
        for j in 0..n_pk {
            let declared = base_model.default_params.omega.matrix[(i, j)];
            omega_matrix[i][j] = fit_init
                .and_then(|fi| {
                    fit_omega_value(fi, &base_model.eta_names[i], &base_model.eta_names[j])
                })
                .unwrap_or(declared);
        }
    }
    // COV-COV diagonal from sample variances.
    for k in 0..n_cov {
        omega_matrix[n_pk + k][n_pk + k] = frem_info.covariate_variances[k];
    }
    // PK-COV cross-terms: small initial values for gradient signal.
    for i in 0..n_pk {
        for k in 0..n_cov {
            let cross = 0.01 * (omega_matrix[i][i] * omega_matrix[n_pk + k][n_pk + k]).sqrt();
            omega_matrix[i][n_pk + k] = cross;
            omega_matrix[n_pk + k][i] = cross;
        }
    }

    // Write block_omega as lower triangle.
    let eta_names_str = all_eta_names.join(", ");
    model.push_str(&format!("  block_omega ({}) = [\n", eta_names_str));
    for i in 0..n_total {
        model.push_str("    ");
        for j in 0..=i {
            if j > 0 {
                model.push_str(", ");
            }
            model.push_str(&format!("{:.6e}", omega_matrix[i][j]));
        }
        if i < n_total - 1 {
            model.push(',');
        }
        model.push('\n');
    }
    model.push_str("  ]\n\n");

    // Copy base sigmas.
    if let Some(params_lines) = blocks.get("parameters") {
        for line in params_lines {
            let trimmed = line.trim();
            if trimmed.starts_with("sigma ") {
                model.push_str(&format!("  {}\n", trimmed));
            }
        }
    }
    // Add fixed covariate sigma.
    model.push_str("  sigma EPSCOV ~ 1e-6 FIX\n\n");

    // ── [individual_parameters] block ──
    model.push_str("[individual_parameters]\n");
    if let Some(indiv_lines) = blocks.get("individual_parameters") {
        for line in indiv_lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                model.push_str(&format!("  {}\n", trimmed));
            }
        }
    }
    // Add covariate individual parameters.
    for cov_name in &frem_info.covariate_names {
        let upper = cov_name.to_uppercase();
        model.push_str(&format!(
            "  COV_{} = TV_{} + ETA_{}_FREM\n",
            upper, upper, upper
        ));
    }
    model.push('\n');

    // ── [structural_model] block ──
    model.push_str("[structural_model]\n");
    if let Some(struct_lines) = blocks.get("structural_model") {
        for line in struct_lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                model.push_str(&format!("  {}\n", trimmed));
            }
        }
    }
    model.push('\n');

    // ── Structural-companion blocks carried over verbatim ──
    // [scaling] and [odes] describe the structural prediction and are NOT
    // reconstructed elsewhere; dropping them silently changes the model the
    // FREM run fits. Notably, omitting `[scaling] obs_scale` (e.g. NONMEM's
    // `CP = A*1000/V`) rescales every prediction, which the estimator then
    // compensates by collapsing a PK typical value (TVCL drove to ~1e-2 instead
    // of ~7 on the workshop FREM model). Copy each block as-is when present.
    for block_name in ["scaling", "odes"] {
        if let Some(block_lines) = blocks.get(block_name) {
            if block_lines
                .iter()
                .any(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
            {
                model.push_str(&format!("[{}]\n", block_name));
                for line in block_lines {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        model.push_str(&format!("  {}\n", trimmed));
                    }
                }
                model.push('\n');
            }
        }
    }

    // ── [error_model] block ──
    model.push_str("[error_model]\n");
    if let Some(err_lines) = blocks.get("error_model") {
        for line in err_lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                model.push_str(&format!("  {}\n", trimmed));
            }
        }
    }
    model.push('\n');

    // ── [fit_options] block ──
    model.push_str("[fit_options]\n");
    // Copy base fit options; preserve the user's method if present, default to focei.
    let mut has_method = false;
    if let Some(fit_lines) = blocks.get("fit_options") {
        for line in fit_lines {
            let trimmed = line.trim();
            if trimmed.starts_with("method") {
                has_method = true;
                model.push_str(&format!("  {}\n", trimmed));
            } else if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("frem_")
            {
                model.push_str(&format!("  {}\n", trimmed));
            }
        }
    }
    if !has_method {
        model.push_str("  method     = focei\n");
    }

    // Add FREM-specific fit options.
    // FREMTYPE column is auto-detected by datareader; no fit_option needed.
    let mut frem_preds: Vec<String> = Vec::new();
    for cov_name in &frem_info.covariate_names {
        let upper = cov_name.to_uppercase();
        let ft = frem_info
            .fremtype_map
            .iter()
            .find(|(n, _)| n == cov_name)
            .map(|(_, v)| *v)
            .unwrap();
        frem_preds.push(format!("TV_{}/ETA_{}_FREM:{}", upper, upper, ft));
    }
    model.push_str(&format!("  frem_predictions = {}\n", frem_preds.join(", ")));
    model.push_str("  frem_sigma = EPSCOV\n");

    // Data path reference (as a comment for documentation).
    model.push_str(&format!("\n# Data: {}\n", output_data_path.display()));

    Ok(model)
}

/// Look up the fitted omega value for an eta pair by name (case-insensitive).
/// Returns `None` when either name is absent from `fit_init`, so the caller
/// falls back to the base model's declared omega. Also returns `None` (rather
/// than panicking) when a matched name index falls outside `omega` — i.e. a
/// malformed `FremFitInit` whose `eta_names` is longer than the matrix is
/// dimensioned — so an inconsistent public-API input degrades gracefully.
fn fit_omega_value(fit_init: &FremFitInit, name_i: &str, name_j: &str) -> Option<f64> {
    let idx_i = fit_init
        .eta_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name_i))?;
    let idx_j = fit_init
        .eta_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name_j))?;
    if idx_i >= fit_init.omega.nrows() || idx_j >= fit_init.omega.ncols() {
        return None;
    }
    Some(fit_init.omega[(idx_i, idx_j)])
}

/// Advisory warnings for a `fit_init` that shares no theta / eta names with the
/// base model — the likely sign a fit of a *different* model was passed in, so
/// the generated FREM model silently fell back to the declared inits (issue
/// #239). Returns one message per axis (theta, eta) that has candidate names on
/// both sides yet matches none; an empty vec means the fit lines up (or has
/// nothing to compare). Pure over its inputs so it is unit-testable without the
/// file IO of [`prepare_frem`].
fn fit_init_name_warnings(base_model: &CompiledModel, fi: &FremFitInit) -> Vec<String> {
    let mut warnings = Vec::new();
    let theta_matched = base_model
        .theta_names
        .iter()
        .any(|n| fi.theta.iter().any(|(fn_, _)| fn_.eq_ignore_ascii_case(n)));
    if !base_model.theta_names.is_empty() && !fi.theta.is_empty() && !theta_matched {
        warnings.push(
            "FREM conversion: the supplied `fit` has no theta names matching the base \
             model, so the model file's declared theta init values were used instead."
                .to_string(),
        );
    }
    let eta_matched = base_model
        .eta_names
        .iter()
        .any(|n| fi.eta_names.iter().any(|fn_| fn_.eq_ignore_ascii_case(n)));
    if !base_model.eta_names.is_empty() && !fi.eta_names.is_empty() && !eta_matched {
        warnings.push(
            "FREM conversion: the supplied `fit`'s omega has no eta names matching the \
             base model, so the model file's declared omega init values were used instead."
                .to_string(),
        );
    }
    warnings
}

/// Rewrite a `theta NAME(init, ...)` declaration line, substituting `init`
/// with the value from a prior fit (issue #239) when the fit has an entry
/// for `NAME`. Bounds, `FIX` flags, and trailing comments are left byte-for-byte
/// untouched — only the numeric init token matched by `re` is replaced.
fn override_theta_init(line: &str, fit_init: &FremFitInit, re: &Regex) -> String {
    let Some(caps) = re.captures(line) else {
        return line.to_string();
    };
    let name = &caps[1];
    let value = match fit_init
        .theta
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
    {
        Some((_, v)) => *v,
        None => return line.to_string(),
    };
    let init_span = caps.get(2).unwrap();
    format!(
        "{}{}{}",
        &line[..init_span.start()],
        value,
        &line[init_span.end()..]
    )
}

/// Resolve which covariates (and their categorical/continuous split) go into the
/// FREM model. The model's `[covariates]` block is the source of truth: it
/// declares the covariates and tags each continuous or categorical.
///
/// - The `[covariates]` block is **required**; without it there is nothing to
///   FREM, and an error is returned.
/// - `covariate_filter` is an optional **subset filter** over the declared
///   covariates. Empty → use every declared covariate. Non-empty → use only the
///   named ones, in the order given; each name must be declared in the block
///   (an undeclared name is an error).
/// - The categorical/continuous split always comes from each selected
///   declaration's `kind`, unless `categorical_override` is supplied non-empty
///   (an escape hatch for callers that don't go through a `[covariates]` block's
///   kinds).
fn resolve_frem_covariates(
    covariate_filter: &[String],
    categorical_override: Option<&[String]>,
    covariate_decls: Option<&[crate::types::CovariateDecl]>,
) -> Result<(Vec<String>, Vec<String>), String> {
    use crate::types::CovariateKind;

    // The [covariates] block is the source of truth and is required. Distinguish
    // an absent block from a present-but-empty one so the diagnostic is accurate.
    let decls = match covariate_decls {
        Some(d) if !d.is_empty() => d,
        Some(_) => {
            return Err(
                "The model's [covariates] block is empty. Declare at least one \
                        covariate (each tagged continuous or categorical) for FREM to fold in."
                    .to_string(),
            );
        }
        None => {
            return Err(
                "FREM needs covariates declared in the model's [covariates] block (each \
                        tagged continuous or categorical). The model has no [covariates] block — \
                        add one listing the covariates to fold into the FREM model."
                    .to_string(),
            );
        }
    };

    // Select declared covariates: all of them, or the named subset.
    let selected: Vec<&crate::types::CovariateDecl> = if covariate_filter.is_empty() {
        decls.iter().collect()
    } else {
        let mut sel: Vec<&crate::types::CovariateDecl> = Vec::with_capacity(covariate_filter.len());
        for name in covariate_filter {
            // Match the declaration case-insensitively, consistent with how
            // covariates are matched against the dataset elsewhere; the declared
            // (canonical) name is what gets used downstream.
            let decl = decls
                .iter()
                .find(|d| d.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| {
                    format!(
                        "FREM covariate '{}' is not declared in the model's [covariates] block. \
                         The `covariates` argument selects a subset of the declared covariates \
                         (declared: {}).",
                        name,
                        decls
                            .iter()
                            .map(|d| d.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            // Reject duplicates so we never emit a covariate (and its eta /
            // FREMTYPE rows) twice, which would produce a degenerate model.
            if sel.iter().any(|d| d.name == decl.name) {
                return Err(format!(
                    "FREM covariate '{}' is listed more than once in the `covariates` filter.",
                    decl.name
                ));
            }
            sel.push(decl);
        }
        sel
    };

    let names: Vec<String> = selected.iter().map(|d| d.name.clone()).collect();
    let cats: Vec<String> = match categorical_override {
        Some(c) if !c.is_empty() => {
            // The override must reference covariates actually in the FREM set,
            // else it silently points at a covariate that won't be modelled.
            for cov in c {
                if !names.iter().any(|n| n.eq_ignore_ascii_case(cov)) {
                    return Err(format!(
                        "FREM categorical override '{}' is not among the selected covariates ({}).",
                        cov,
                        names.join(", ")
                    ));
                }
            }
            c.to_vec()
        }
        _ => selected
            .iter()
            .filter(|d| d.kind == CovariateKind::Categorical)
            .map(|d| d.name.clone())
            .collect(),
    };
    Ok((names, cats))
}

/// Orchestrate FREM preparation: parse model, read data, transform, generate, write.
///
/// The model's `[covariates]` block defines the covariates to fold into the FREM
/// omega block and tags each continuous or categorical — it is the source of
/// truth and is **required**.
///
/// `covariates` is an optional **subset filter** over the declared covariates:
/// empty means "use every declared covariate"; a non-empty list selects only
/// those (each must be declared in the block, else an error). It does not
/// introduce covariates the model hasn't declared.
///
/// `categorical_covariates` is an optional override for the categorical split;
/// when `None`/empty the split is taken from each selected declaration's `kind`.
///
/// `fit_init` optionally seeds the generated FREM model's PK theta and omega
/// init values from a completed fit of the base model (issue #239), so a
/// subsequent fit of the FREM model warm-starts from converged parameters
/// instead of the base model file's declared inits. `None` preserves the
/// prior behaviour of copying the declared inits verbatim.
#[allow(clippy::too_many_arguments)]
pub fn prepare_frem(
    model_path: &Path,
    data_path: &Path,
    covariates: &[String],
    categorical_covariates: Option<&[String]>,
    output_model_path: Option<&Path>,
    output_data_path: Option<&Path>,
    missing_value: Option<f64>,
    fit_init: Option<&FremFitInit>,
) -> Result<FremPrepareResult, String> {
    use crate::io::datareader::read_nonmem_csv_mapped;
    use crate::parser::model_parser::parse_full_model_file;

    // Full parse so the optional `[covariates]` block is available for fallback.
    let parsed = parse_full_model_file(model_path)?;
    let base_model = &parsed.model;
    // Honour the model's `[data]` column mapping (#730) so FREM prep reads the
    // same columns as `fit()`/`check` do — otherwise a mapped TIME/DV header is
    // missed here even though it works everywhere else.
    let population = read_nonmem_csv_mapped(data_path, None, None, &parsed.column_map)?;

    // Resolve the covariate list (explicit args, else the model's [covariates]
    // block) — see `resolve_frem_covariates`.
    let (resolved_covariates, resolved_categorical) = resolve_frem_covariates(
        covariates,
        categorical_covariates,
        parsed.covariate_decls.as_deref(),
    )?;

    // Transform dataset.
    let (csv_content, frem_info) = transform_dataset_for_frem(
        &population,
        base_model,
        &resolved_covariates,
        &resolved_categorical,
        missing_value,
    )?;

    // Determine output paths.
    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));

    let default_model_path = model_dir.join(format!("{}_frem.ferx", model_stem));
    let default_data_path = model_dir.join(format!("{}_frem_data.csv", model_stem));

    let out_model = output_model_path.unwrap_or(&default_model_path);
    let out_data = output_data_path.unwrap_or(&default_data_path);

    // Read base model text for model generation.
    let base_text = std::fs::read_to_string(model_path)
        .map_err(|e| format!("Failed to read model file: {}", e))?;

    // Generate FREM model.
    let model_text = generate_frem_model(&base_text, base_model, &frem_info, out_data, fit_init)?;

    // Write outputs.
    std::fs::write(out_data, &csv_content)
        .map_err(|e| format!("Failed to write FREM data CSV: {}", e))?;
    std::fs::write(out_model, &model_text)
        .map_err(|e| format!("Failed to write FREM model file: {}", e))?;

    let n_total = base_model.n_eta + frem_info.covariate_names.len();

    // Conversion-time advisory: estimated parameters with no random effect are
    // estimated poorly by IMP/IMPMAP (the importance-weighted M-step is biased for
    // weakly-identified fixed effects — see #406). Flag them now so the user can
    // add an ETA before fitting; ferx mu-references automatically.
    let mut warnings: Vec<String> = Vec::new();
    let no_eta = crate::estimation::impmap::non_fixed_thetas_without_eta(
        base_model,
        &base_model.default_params.theta_fixed,
    );
    if !no_eta.is_empty() {
        warnings.push(format!(
            "FREM conversion: estimated parameter(s) [{}] have no associated ETA. When fitting \
             this model with IMP/IMPMAP, a fixed-effect-only parameter is estimated solely through \
             the importance-weighted M-step, which is biased for weakly-identified parameters and \
             may converge to the wrong value. Add an ETA to each (e.g. `P = TVP * exp(ETA_P)` with \
             a small, optionally FIX, omega — ferx mu-references automatically), hold it FIX, or \
             fit with FOCEI.",
            no_eta.join(", ")
        ));
    }

    // Advise when `fit_init` was supplied but shares no names with the base
    // model — most likely a fit of a different model was passed in, so the
    // generated FREM model silently fell back to the declared inits.
    if let Some(fi) = fit_init {
        warnings.extend(fit_init_name_warnings(base_model, fi));
    }

    Ok(FremPrepareResult {
        model_path: out_model.to_path_buf(),
        data_path: out_data.to_path_buf(),
        covariate_means: frem_info
            .covariate_names
            .iter()
            .zip(frem_info.covariate_means.iter())
            .map(|(n, &m)| (n.clone(), m))
            .collect(),
        covariate_variances: frem_info
            .covariate_names
            .iter()
            .zip(frem_info.covariate_variances.iter())
            .map(|(n, &v)| (n.clone(), v))
            .collect(),
        fremtype_map: frem_info.fremtype_map.clone(),
        n_total_etas: n_total,
        warnings,
    })
}

/// Simple block parser for .ferx model files.
/// Returns a map of block name → lines within that block.
fn parse_blocks(content: &str) -> Result<HashMap<String, Vec<String>>, String> {
    let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_block: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let name = trimmed[1..trimmed.len() - 1].to_string();
            current_block = Some(name.clone());
            blocks.entry(name).or_default();
        } else if let Some(ref block) = current_block {
            blocks.get_mut(block).unwrap().push(line.to_string());
        }
    }

    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    #[allow(unused_imports)]
    use std::collections::HashMap;

    fn make_test_population() -> Population {
        let subjects = vec![
            Subject {
                id: "1".to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 2.0, 4.0],
                obs_raw_times: vec![1.0, 2.0, 4.0],
                observations: vec![5.0, 8.0, 6.0],
                obs_cmts: vec![1, 1, 1],
                covariates: {
                    let mut m = HashMap::new();
                    m.insert("WT".to_string(), 70.0);
                    m.insert("AGE".to_string(), 30.0);
                    m
                },
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: Vec::new(),
            },
            Subject {
                id: "2".to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 2.0],
                obs_raw_times: vec![1.0, 2.0],
                observations: vec![4.0, 7.0],
                obs_cmts: vec![1, 1],
                covariates: {
                    let mut m = HashMap::new();
                    m.insert("WT".to_string(), 80.0);
                    m.insert("AGE".to_string(), 40.0);
                    m
                },
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: Vec::new(),
            },
        ];
        Population {
            subjects,
            covariate_names: vec!["WT".to_string(), "AGE".to_string()],
            dv_column: "dv".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    }

    fn make_test_model() -> CompiledModel {
        CompiledModel {
            has_conditional_eta_params: false,
            name: "test".into(),
            pk_model: PkModel::OneCptOral,
            error_model: ErrorModel::Proportional,
            error_spec: ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|_, _, _, _t: f64| PkParams::default()),
            n_theta: 3,
            n_eta: 3,
            n_epsilon: 1,
            n_kappa: 0,
            theta_names: vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            kappa_names: Vec::new(),
            indiv_param_names: vec!["CL".into(), "V".into(), "KA".into()],
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: vec![0.2, 10.0, 1.5],
                theta_names: vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
                theta_lower: vec![0.001, 0.1, 0.01],
                theta_upper: vec![10.0, 500.0, 50.0],
                theta_fixed: vec![false, false, false],
                omega: OmegaMatrix::from_diagonal(
                    &[0.09, 0.04, 0.30],
                    vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
                ),
                omega_fixed: vec![false, false, false],
                sigma: SigmaVector {
                    values: vec![0.02],
                    names: vec!["PROP_ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: vec![false, false, false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: Some(Box::new(|_t, _c| vec![0.2, 10.0, 1.5])),
            pk_indices: vec![0, 1, 4],
            eta_map: vec![0, 1, 2],
            pk_idx_f64: vec![0.0, 1.0, 4.0],
            sel_flat: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            dose_attr_map: Default::default(),
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec![],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            transit_ode_equivalent: None,
        }
    }

    #[test]
    fn test_transform_dataset_row_count() {
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string(), "AGE".to_string()];
        let (csv, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        // Subject 1: 1 dose + 2 cov obs + 3 PK obs = 6 rows
        // Subject 2: 1 dose + 2 cov obs + 2 PK obs = 5 rows
        // Total: 11 data rows + 1 header = 12 lines
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 12);
        assert_eq!(info.covariate_names.len(), 2);
        assert_eq!(info.n_base_etas, 3);
    }

    #[test]
    fn test_transform_dataset_fremtype_values() {
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string(), "AGE".to_string()];
        let (csv, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        assert_eq!(info.fremtype_map[0], ("WT".to_string(), 100));
        assert_eq!(info.fremtype_map[1], ("AGE".to_string(), 200));

        // Check that FREMTYPE column values are correct.
        let lines: Vec<&str> = csv.lines().collect();
        let header = lines[0];
        let ft_col = header.split(',').position(|h| h == "FREMTYPE").unwrap();

        // Line 2 = dose row for subject 1 → FREMTYPE=0
        assert_eq!(lines[1].split(',').nth(ft_col).unwrap(), "0");
        // Lines 3-4 = covariate obs for subject 1 → FREMTYPE=100, 200
        assert_eq!(lines[2].split(',').nth(ft_col).unwrap(), "100");
        assert_eq!(lines[3].split(',').nth(ft_col).unwrap(), "200");
        // Lines 5-7 = PK obs for subject 1 → FREMTYPE=0
        assert_eq!(lines[4].split(',').nth(ft_col).unwrap(), "0");
    }

    #[test]
    fn test_transform_dataset_covariate_means() {
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string(), "AGE".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        // WT: (70 + 80) / 2 = 75
        assert!((info.covariate_means[0] - 75.0).abs() < 1e-10);
        // AGE: (30 + 40) / 2 = 35
        assert!((info.covariate_means[1] - 35.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_dataset_covariate_variances() {
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        // WT: var = ((70-75)^2 + (80-75)^2) / (2-1) = 50
        assert!((info.covariate_variances[0] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_dataset_missing_covariate_errors() {
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["NONEXISTENT".to_string()];
        let result = transform_dataset_for_frem(&pop, &model, &covs, &[], None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NONEXISTENT"));
    }

    #[test]
    fn test_generate_frem_model_valid_ferx() {
        let base_text = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  maxiter = 300
";

        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string(), "AGE".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        let result = generate_frem_model(
            base_text,
            &model,
            &info,
            Path::new("test_frem_data.csv"),
            None,
        );
        assert!(result.is_ok());

        let model_text = result.unwrap();

        // Should contain fixed covariate thetas.
        assert!(model_text.contains("theta TV_WT("));
        assert!(model_text.contains("FIX"));
        assert!(model_text.contains("theta TV_AGE("));

        // Should contain block_omega with all etas.
        assert!(model_text.contains("block_omega"));
        assert!(model_text.contains("ETA_CL"));
        assert!(model_text.contains("ETA_WT_FREM"));
        assert!(model_text.contains("ETA_AGE_FREM"));

        // Should contain covariate individual parameters.
        assert!(model_text.contains("COV_WT = TV_WT + ETA_WT_FREM"));
        assert!(model_text.contains("COV_AGE = TV_AGE + ETA_AGE_FREM"));

        // Should contain EPSCOV.
        assert!(model_text.contains("EPSCOV"));

        // Should contain FREM fit options.
        assert!(model_text.contains("frem_predictions"));
        assert!(model_text.contains("frem_sigma = EPSCOV"));
    }

    #[test]
    fn test_generate_frem_model_omega_dimensions() {
        let base_text = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        let model_text =
            generate_frem_model(base_text, &model, &info, Path::new("test.csv"), None).unwrap();

        // 3 PK etas + 1 cov eta = 4 total, lower triangle has 4*(4+1)/2 = 10 values
        let block_start = model_text.find("block_omega").unwrap();
        let bracket_start = model_text[block_start..].find('[').unwrap() + block_start;
        let bracket_end = model_text[bracket_start..].find(']').unwrap() + bracket_start;
        let values_str = &model_text[bracket_start + 1..bracket_end];
        let n_values: usize = values_str
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .count();
        assert_eq!(n_values, 10); // 4*(4+1)/2
    }

    #[test]
    fn test_generate_frem_model_uses_fit_init_theta_and_omega() {
        // Issue #239: when a prior fit is supplied, its theta/omega estimates
        // seed the generated FREM model's PK init values instead of the base
        // model file's declared inits.
        let base_text = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        let fit_init = FremFitInit {
            theta: vec![
                ("TVCL".to_string(), 0.321),
                ("TVV".to_string(), 12.7),
                ("TVKA".to_string(), 1.42),
            ],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.11, 0.05, 0.28])),
        };

        let model_text = generate_frem_model(
            base_text,
            &model,
            &info,
            Path::new("test.csv"),
            Some(&fit_init),
        )
        .unwrap();

        // Fitted theta values replace the declared inits, bounds untouched.
        assert!(
            model_text.contains("theta TVCL(0.321, 0.001, 10.0)"),
            "expected fitted TVCL init, got:\n{model_text}"
        );
        assert!(
            model_text.contains("theta TVV(12.7, 0.1, 500.0)"),
            "expected fitted TVV init, got:\n{model_text}"
        );
        assert!(
            model_text.contains("theta TVKA(1.42, 0.01, 50.0)"),
            "expected fitted TVKA init, got:\n{model_text}"
        );

        // Fitted omega diagonal (0.11, 0.05, 0.28) replaces the declared
        // (0.09, 0.04, 0.30) on the PK-PK block of the block_omega lower
        // triangle: rows 0, 2, 5 (1-indexed diagonal positions 1, 3, 6).
        let block_start = model_text.find("block_omega").unwrap();
        let bracket_start = model_text[block_start..].find('[').unwrap() + block_start;
        let bracket_end = model_text[bracket_start..].find(']').unwrap() + bracket_start;
        let rows: Vec<Vec<f64>> = model_text[bracket_start + 1..bracket_end]
            .lines()
            .map(|l| l.trim().trim_end_matches(','))
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.split(',')
                    .map(|v| v.trim().parse::<f64>().unwrap())
                    .collect()
            })
            .collect();
        assert!(
            (rows[0][0] - 0.11).abs() < 1e-9,
            "ETA_CL diag: {:?}",
            rows[0]
        );
        assert!(
            (rows[1][1] - 0.05).abs() < 1e-9,
            "ETA_V diag: {:?}",
            rows[1]
        );
        assert!(
            (rows[2][2] - 0.28).abs() < 1e-9,
            "ETA_KA diag: {:?}",
            rows[2]
        );
    }

    #[test]
    fn test_generate_frem_model_tolerates_fit_init_omega_dim_mismatch() {
        // Regression: a malformed `FremFitInit` whose `eta_names` is longer than
        // its `omega` is dimensioned must degrade gracefully (fall back to the
        // declared omega) instead of panicking on an out-of-bounds index inside
        // `fit_omega_value`. The base model has 3 PK etas; here `omega` is only
        // 2x2, so the ETA_KA lookup (index 2) is out of range.
        let base_text = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        // eta_names names all 3 PK etas, but omega is only 2x2 (ETA_KA -> idx 2
        // is out of bounds).
        let fit_init = FremFitInit {
            theta: vec![("TVCL".to_string(), 0.321)],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.11, 0.05])),
        };

        // Must not panic; the in-range ETA_CL/ETA_V entries are seeded from the
        // fit, ETA_KA falls back to the declared 0.30.
        let model_text = generate_frem_model(
            base_text,
            &model,
            &info,
            Path::new("test.csv"),
            Some(&fit_init),
        )
        .unwrap();

        let block_start = model_text.find("block_omega").unwrap();
        let bracket_start = model_text[block_start..].find('[').unwrap() + block_start;
        let bracket_end = model_text[bracket_start..].find(']').unwrap() + bracket_start;
        let rows: Vec<Vec<f64>> = model_text[bracket_start + 1..bracket_end]
            .lines()
            .map(|l| l.trim().trim_end_matches(','))
            .filter(|l| !l.is_empty())
            .map(|l| {
                l.split(',')
                    .map(|v| v.trim().parse::<f64>().unwrap())
                    .collect()
            })
            .collect();
        assert!(
            (rows[0][0] - 0.11).abs() < 1e-9,
            "ETA_CL diag: {:?}",
            rows[0]
        );
        assert!(
            (rows[1][1] - 0.05).abs() < 1e-9,
            "ETA_V diag: {:?}",
            rows[1]
        );
        // ETA_KA out of range in fit omega -> declared 0.30 retained.
        assert!(
            (rows[2][2] - 0.30).abs() < 1e-9,
            "ETA_KA diag: {:?}",
            rows[2]
        );
    }

    #[test]
    fn test_fit_init_name_warnings() {
        let model = make_test_model(); // thetas TVCL/TVV/TVKA, etas ETA_CL/V/KA

        // Names line up (case-insensitively) -> no advisory.
        let ok = FremFitInit {
            theta: vec![("tvcl".into(), 0.3)],
            eta_names: vec!["eta_cl".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.1])),
        };
        assert!(fit_init_name_warnings(&model, &ok).is_empty());

        // A different model's fit: no theta and no eta names match -> two
        // advisories, one per axis.
        let wrong = FremFitInit {
            theta: vec![("FOO".into(), 1.0)],
            eta_names: vec!["ETA_BAR".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.1])),
        };
        let w = fit_init_name_warnings(&model, &wrong);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("theta names"));
        assert!(w[1].contains("eta names"));

        // Theta matches but eta does not -> only the eta advisory.
        let theta_only = FremFitInit {
            theta: vec![("TVCL".into(), 0.3)],
            eta_names: vec!["ETA_BAR".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.1])),
        };
        let w = fit_init_name_warnings(&model, &theta_only);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("eta names"));

        // Nothing to compare (empty fit) -> no advisory.
        let empty = FremFitInit {
            theta: vec![],
            eta_names: vec![],
            omega: DMatrix::zeros(0, 0),
        };
        assert!(fit_init_name_warnings(&model, &empty).is_empty());
    }

    #[test]
    fn test_override_theta_init_fallbacks() {
        let re = Regex::new(r"(?i)^\s*theta\s+(\w+)\s*\(\s*([0-9eE.+-]+)").unwrap();
        let fit = FremFitInit {
            theta: vec![("TVCL".into(), 0.321)],
            eta_names: vec![],
            omega: DMatrix::zeros(0, 0),
        };

        // Matched name -> init token replaced, bounds untouched.
        assert_eq!(
            override_theta_init("theta TVCL(0.2, 0.001, 10.0)", &fit, &re),
            "theta TVCL(0.321, 0.001, 10.0)"
        );
        // Name not present in the fit -> line returned unchanged.
        assert_eq!(
            override_theta_init("theta TVV(10.0, 0.1, 500.0)", &fit, &re),
            "theta TVV(10.0, 0.1, 500.0)"
        );
        // Line the regex cannot parse -> returned unchanged.
        assert_eq!(
            override_theta_init("theta MALFORMED", &fit, &re),
            "theta MALFORMED"
        );
    }

    #[test]
    fn test_fit_omega_value_lookup_and_bounds() {
        let fit = FremFitInit {
            theta: vec![],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into()],
            omega: DMatrix::from_row_slice(2, 2, &[0.11, 0.02, 0.02, 0.05]),
        };
        // In-range, case-insensitive, symmetric.
        assert_eq!(fit_omega_value(&fit, "eta_cl", "eta_v"), Some(0.02));
        assert_eq!(fit_omega_value(&fit, "ETA_V", "ETA_V"), Some(0.05));
        // Name absent -> None.
        assert_eq!(fit_omega_value(&fit, "ETA_CL", "ETA_KA"), None);

        // Name present but index outside the (smaller) omega -> None, no panic.
        let mismatched = FremFitInit {
            theta: vec![],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            omega: DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.11, 0.05])),
        };
        assert_eq!(fit_omega_value(&mismatched, "ETA_KA", "ETA_KA"), None);
        assert_eq!(fit_omega_value(&mismatched, "ETA_CL", "ETA_KA"), None);
    }

    #[test]
    fn test_generate_frem_model_preserves_scaling_block() {
        // Regression (#406): the base model's `[scaling] obs_scale` block must be
        // carried into the generated FREM model. Dropping it rescales every
        // prediction (here NONMEM's CP = A*1000/V), which the estimator then
        // compensates by collapsing a PK typical value (TVCL → ~1e-2).
        let base_text = r"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[scaling]
  obs_scale = 0.001

[error_model]
  DV ~ proportional(PROP_ERR)
";
        let pop = make_test_population();
        let model = make_test_model();
        let covs = vec!["WT".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        let model_text =
            generate_frem_model(base_text, &model, &info, Path::new("test.csv"), None).unwrap();

        assert!(
            model_text.contains("[scaling]"),
            "generated FREM model dropped the [scaling] block:\n{model_text}"
        );
        assert!(
            model_text.contains("obs_scale = 0.001"),
            "generated FREM model dropped obs_scale:\n{model_text}"
        );
        // The generated model must still parse (block placement is valid).
        crate::parser::model_parser::parse_model_string(&model_text)
            .expect("generated FREM model parses");
    }

    #[test]
    fn frem_partition_samples_missing_covariate_etas() {
        // Regression (#406): a subject missing a covariate pseudo-obs row (the
        // FREM data omits rows for missing covariate values) must NOT bail to the
        // unstable full-dimensional IS. subject_frem_partition puts the missing
        // covariate eta into the *sampled* set (with the PK etas) and pins only
        // the observed covariate eta at its data deviation.
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize)); // FREMTYPE 100 -> (theta 0, eta 1) OBSERVED
        map.insert(300u16, (0usize, 3usize)); // FREMTYPE 300 -> (theta 0, eta 3) MISSING
        let fc = FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        };
        let mut subj = make_test_population().subjects.remove(0);
        subj.obs_times = vec![1.0, 1.0];
        subj.obs_raw_times = vec![1.0, 1.0];
        subj.observations = vec![5.0, 7.0]; // row1 = PK obs, row2 = FREMTYPE 100 cov obs
        subj.obs_cmts = vec![1, 1];
        subj.cens = vec![0, 0];
        subj.fremtype = vec![0, 100]; // FREMTYPE 300 (eta 3) has NO row -> missing

        let theta = vec![0.2, 10.0, 1.5];
        let (sampled, observed, d) =
            crate::estimation::importance_sampling::subject_frem_partition(
                &subj,
                &theta,
                &fc,
                &[0, 2], // pk etas
                &[1, 3], // covariate etas
            )
            .expect("Some when at least one covariate observed");

        assert_eq!(
            sampled,
            vec![0, 2, 3],
            "missing cov eta 3 joins the sampled PK set"
        );
        assert_eq!(observed, vec![1], "only eta 1 (FREMTYPE 100) is observed");
        assert_eq!(d.len(), 1);
        assert!((d[0] - (7.0 - 0.2)).abs() < 1e-12, "d = cov_obs - TV");
    }

    #[test]
    fn obs_nll_does_not_clamp_negative_covariate_pseudo_obs() {
        // Regression (#406): a FREM covariate pseudo-observation predicts a
        // covariate *value* (TV+eta), which can be ≤ 0 for centered/standardized/
        // log-scale covariates. obs_nll_subject_into must NOT clamp that prediction
        // to 1e-12 (which would fabricate a huge residual and corrupt the
        // Rao-Blackwellised IS marginal/weights). Here the covariate obs is -5.0
        // and eta is chosen so the prediction is exactly -5.0 (residual 0), so the
        // row must contribute only 0.5·ln(R). Without the fix the clamped
        // prediction (1e-12) gives residual ≈ -5 and obs_nll ≈ 0.5·(25/R) ≈ 3.1e4.
        let mut model = make_test_model();
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize)); // FREMTYPE 100 -> (theta TVCL idx 0, eta idx 1)
        model.frem_config = Some(FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0, // sigma[0] = 0.02 -> R = 4e-4
        });

        let mut subj = make_test_population().subjects.remove(0);
        subj.obs_times = vec![1.0];
        subj.obs_raw_times = vec![1.0];
        subj.observations = vec![-5.0]; // negative covariate pseudo-obs
        subj.obs_cmts = vec![1];
        subj.cens = vec![0];
        subj.fremtype = vec![100];

        let theta = model.default_params.theta.clone(); // [0.2, 10.0, 1.5]
                                                        // pred(cov row) = theta[0] + eta[1] = 0.2 + eta[1]; want -5.0 -> eta[1] = -5.2
        let eta = vec![0.0, -5.2, 0.0];
        let sigma = model.default_params.sigma.values.clone();
        let mut scratch = crate::pk::EventPkParams::with_capacity_for(&subj);

        let nll = crate::stats::likelihood::obs_nll_subject_into(
            &model,
            &subj,
            &theta,
            &sigma,
            &eta,
            &mut scratch,
        );
        assert!(nll.is_finite(), "obs_nll should be finite, got {nll}");
        // R = 0.02² = 4e-4; with residual 0 the row contributes 0.5·ln(4e-4) ≈ -3.9.
        assert!(
            nll < 1.0,
            "negative covariate pseudo-obs with residual 0 should give a small \
             obs_nll (~-3.9), not the clamped ~3.1e4; got {nll}"
        );
    }

    #[test]
    fn test_parse_blocks() {
        let content = "[parameters]\n  theta TVCL(0.2)\n\n[error_model]\n  DV ~ additive(ERR)\n";
        let blocks = parse_blocks(content).unwrap();
        assert!(blocks.contains_key("parameters"));
        assert!(blocks.contains_key("error_model"));
        assert_eq!(blocks["parameters"].len(), 2); // theta line + empty line
    }

    fn decl(name: &str, kind: CovariateKind) -> CovariateDecl {
        CovariateDecl {
            name: name.to_string(),
            kind,
        }
    }

    #[test]
    fn resolve_covariates_uses_all_declared_when_no_filter() {
        // Empty filter → every declared covariate; categorical split from `kind`.
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("CRCL", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let (covs, cats) = resolve_frem_covariates(&[], None, Some(&decls)).unwrap();
        assert_eq!(covs, vec!["WT", "CRCL", "SEX"]);
        assert_eq!(cats, vec!["SEX"]);
    }

    #[test]
    fn apply_frem_prediction_override_replaces_only_covariate_rows() {
        // FREMTYPE > 0 rows are replaced with theta[k] + eta[m]; PK rows (FREMTYPE 0)
        // are left untouched.
        let mut model = make_test_model();
        // Map FREMTYPE 100 -> (theta TVCL idx 0, eta ETA_V idx 1).
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize));
        model.frem_config = Some(FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        });

        let mut subj = make_test_population().subjects.remove(0); // 3 obs
        subj.fremtype = vec![0, 100, 0];

        let theta = model.default_params.theta.clone(); // [0.2, 10.0, 1.5]
        let eta = vec![0.1, 0.2, 0.3];
        let mut preds = vec![5.0, 8.0, 6.0];
        crate::pk::apply_frem_prediction_override(&model, &subj, &theta, &eta, &mut preds);

        assert_eq!(preds[0], 5.0); // PK row untouched
        assert_eq!(preds[2], 6.0); // PK row untouched
        assert_eq!(preds[1], theta[0] + eta[1]); // covariate row = TVCL + ETA_V = 0.4
    }

    #[test]
    fn resolve_covariates_filter_selects_subset_in_filter_order() {
        // A non-empty filter selects a subset; order follows the filter, kinds
        // come from the block.
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("CRCL", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let (covs, cats) =
            resolve_frem_covariates(&["SEX".to_string(), "WT".to_string()], None, Some(&decls))
                .unwrap();
        assert_eq!(covs, vec!["SEX", "WT"]);
        assert_eq!(cats, vec!["SEX"]);
    }

    #[test]
    fn resolve_covariates_filter_with_undeclared_name_errors() {
        // Filtering to a covariate not in the block is an error.
        let decls = vec![decl("WT", CovariateKind::Continuous)];
        let err = resolve_frem_covariates(&["AGE".to_string()], None, Some(&decls)).unwrap_err();
        assert!(
            err.contains("AGE") && err.contains("[covariates]"),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_covariates_categorical_override() {
        // The override forces the categorical split regardless of block kinds.
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        let (covs, cats) =
            resolve_frem_covariates(&[], Some(&["WT".to_string()]), Some(&decls)).unwrap();
        assert_eq!(covs, vec!["WT", "SEX"]);
        assert_eq!(cats, vec!["WT"]);
    }

    #[test]
    fn resolve_covariates_filter_is_case_insensitive_and_canonicalizes() {
        // Filter matches the declaration case-insensitively; the declared
        // (canonical) name is returned.
        let decls = vec![decl("WT", CovariateKind::Continuous)];
        let (covs, _) = resolve_frem_covariates(&["wt".to_string()], None, Some(&decls)).unwrap();
        assert_eq!(covs, vec!["WT"]);
    }

    #[test]
    fn resolve_covariates_filter_rejects_duplicates() {
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("AGE", CovariateKind::Continuous),
        ];
        // Exact and case-variant duplicates both error.
        assert!(
            resolve_frem_covariates(&["WT".to_string(), "WT".to_string()], None, Some(&decls))
                .is_err()
        );
        let err =
            resolve_frem_covariates(&["WT".to_string(), "wt".to_string()], None, Some(&decls))
                .unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn resolve_covariates_categorical_override_must_be_selected() {
        // An override naming a covariate that isn't in the FREM set is an error.
        let decls = vec![
            decl("WT", CovariateKind::Continuous),
            decl("SEX", CovariateKind::Categorical),
        ];
        // SEX is excluded by the filter, so overriding it categorical is invalid.
        let err = resolve_frem_covariates(
            &["WT".to_string()],
            Some(&["SEX".to_string()]),
            Some(&decls),
        )
        .unwrap_err();
        assert!(err.contains("not among the selected"), "got: {err}");
    }

    fn make_test_population_with_race() -> Population {
        let mut subjects = Vec::new();
        // 3 subjects with RACE: 1 (most common), 2, 3
        for (id, race) in &[("1", 1.0), ("2", 1.0), ("3", 2.0), ("4", 3.0)] {
            subjects.push(Subject {
                id: id.to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0],
                obs_raw_times: vec![1.0],
                observations: vec![5.0],
                obs_cmts: vec![1],
                covariates: {
                    let mut m = HashMap::new();
                    m.insert("WT".to_string(), 70.0);
                    m.insert("RACE".to_string(), *race);
                    m
                },
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: Vec::new(),
            });
        }
        Population {
            subjects,
            covariate_names: vec!["WT".to_string(), "RACE".to_string()],
            dv_column: "dv".to_string(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn test_categorical_binarization_expands_race() {
        let pop = make_test_population_with_race();
        let model = make_test_model();
        let covs = vec!["WT".to_string(), "RACE".to_string()];
        let cats = vec!["RACE".to_string()];
        let (csv, info) = transform_dataset_for_frem(&pop, &model, &covs, &cats, None).unwrap();

        // RACE has 3 levels (1,2,3) → expanded to RACE_2, RACE_3.
        // Final covariates: WT, RACE_2, RACE_3 (3 total, not 2).
        assert_eq!(info.covariate_names, vec!["WT", "RACE_2", "RACE_3"]);
        assert_eq!(info.fremtype_map.len(), 3);
        assert_eq!(info.fremtype_map[0], ("WT".to_string(), 100));
        assert_eq!(info.fremtype_map[1], ("RACE_2".to_string(), 200));
        assert_eq!(info.fremtype_map[2], ("RACE_3".to_string(), 300));

        // Check RACE_2 mean: 1 out of 4 subjects has RACE=2 → mean=0.25
        assert!((info.covariate_means[1] - 0.25).abs() < 1e-10);
        // Check RACE_3 mean: 1 out of 4 has RACE=3 → mean=0.25
        assert!((info.covariate_means[2] - 0.25).abs() < 1e-10);

        // CSV should have RACE_2 and RACE_3 indicator columns.
        let header = csv.lines().next().unwrap();
        assert!(header.contains("RACE_2"));
        assert!(header.contains("RACE_3"));

        // Each subject should have 3 pseudo-obs (WT, RACE_2, RACE_3).
        let ft_col = header.split(',').position(|h| h == "FREMTYPE").unwrap();
        let dv_col = header.split(',').position(|h| h == "DV").unwrap();
        let lines: Vec<&str> = csv.lines().collect();

        // Subject 1 (RACE=1): RACE_2 pseudo-obs DV=0, RACE_3 pseudo-obs DV=0
        // Find subject 1's FREMTYPE=200 row (RACE_2)
        let subj1_race2 = lines.iter().find(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            parts[0] == "1" && parts[ft_col] == "200"
        });
        assert!(subj1_race2.is_some());
        let parts: Vec<&str> = subj1_race2.unwrap().split(',').collect();
        assert_eq!(parts[dv_col], "0"); // RACE=1 → RACE_2 indicator = 0

        // Subject 3 (RACE=2): RACE_2 pseudo-obs DV=1
        let subj3_race2 = lines.iter().find(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            parts[0] == "3" && parts[ft_col] == "200"
        });
        assert!(subj3_race2.is_some());
        let parts: Vec<&str> = subj3_race2.unwrap().split(',').collect();
        assert_eq!(parts[dv_col], "1"); // RACE=2 → RACE_2 indicator = 1
    }

    #[test]
    fn test_binary_categorical_not_expanded() {
        // NCI-like binary covariate (levels 0, 1) should NOT be expanded.
        let mut pop = make_test_population();
        for (i, subj) in pop.subjects.iter_mut().enumerate() {
            subj.covariates
                .insert("NCI".to_string(), if i == 0 { 0.0 } else { 1.0 });
        }
        pop.covariate_names.push("NCI".to_string());

        let model = make_test_model();
        let covs = vec!["WT".to_string(), "NCI".to_string()];
        let cats = vec!["NCI".to_string()];
        let (_, info) = transform_dataset_for_frem(&pop, &model, &covs, &cats, None).unwrap();

        // NCI has only 2 levels → kept as-is, not expanded.
        assert_eq!(info.covariate_names, vec!["WT", "NCI"]);
        assert_eq!(info.fremtype_map.len(), 2);
    }

    #[test]
    fn test_missing_covariate_values_excluded() {
        // Subject 1: WT=70, AGE=30, HT=170
        // Subject 2: WT=80, AGE=40, HT=-99 (missing)
        let mut pop = make_test_population();
        for subj in pop.subjects.iter_mut() {
            if subj.id == "1" {
                subj.covariates.insert("HT".to_string(), 170.0);
            } else {
                subj.covariates.insert("HT".to_string(), -99.0);
            }
        }
        pop.covariate_names.push("HT".to_string());

        let model = make_test_model();
        let covs = vec!["WT".to_string(), "HT".to_string(), "AGE".to_string()];
        let (csv, info) = transform_dataset_for_frem(&pop, &model, &covs, &[], None).unwrap();

        // HT mean should be computed from subject 1 only (170.0), not (170 + (-99)) / 2.
        let ht_idx = info.covariate_names.iter().position(|n| n == "HT").unwrap();
        assert!(
            (info.covariate_means[ht_idx] - 170.0).abs() < 1e-10,
            "HT mean = {} (expected 170.0)",
            info.covariate_means[ht_idx]
        );

        // Subject 2 should NOT have a FREMTYPE=200 (HT) pseudo-obs row.
        let lines: Vec<&str> = csv.lines().collect();
        let header = lines[0];
        let ft_col = header.split(',').position(|h| h == "FREMTYPE").unwrap();
        let id_col = header.split(',').position(|h| h == "ID").unwrap();

        let subj2_ht = lines.iter().find(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            parts[id_col] == "2" && parts[ft_col] == "200"
        });
        assert!(
            subj2_ht.is_none(),
            "Subject 2 should not have HT pseudo-obs (HT is missing)"
        );

        // Subject 1 should have the HT pseudo-obs row.
        let subj1_ht = lines.iter().find(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            parts[id_col] == "1" && parts[ft_col] == "200"
        });
        assert!(subj1_ht.is_some(), "Subject 1 should have HT pseudo-obs");
    }

    #[test]
    fn test_missing_categorical_excluded() {
        // RACE: subject 1=1, subject 2=-99 (missing), subject 3=2
        // (Need 3 subjects for a polychotomous test.)
        let mut pop = make_test_population();
        // Add a third subject.
        pop.subjects.push(Subject {
            id: "3".to_string(),
            observations: vec![3.0],
            obs_times: vec![1.0],
            obs_raw_times: vec![1.0],
            obs_cmts: vec![1],
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            covariates: {
                let mut m = HashMap::new();
                m.insert("WT".to_string(), 90.0);
                m.insert("AGE".to_string(), 50.0);
                m.insert("RACE".to_string(), 2.0);
                m
            },
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: Vec::new(),
        });
        // Subject 1: RACE=1, Subject 2: RACE=-99
        pop.subjects[0].covariates.insert("RACE".to_string(), 1.0);
        pop.subjects[1].covariates.insert("RACE".to_string(), -99.0);
        pop.covariate_names.push("RACE".to_string());

        let model = make_test_model();
        let covs = vec!["WT".to_string(), "RACE".to_string()];
        let cats = vec!["RACE".to_string()];
        let (csv, info) = transform_dataset_for_frem(&pop, &model, &covs, &cats, None).unwrap();

        // RACE has 3 subjects but only 2 valid (1 and 2) → K=2 levels → binary, not expanded.
        // So "RACE" stays as-is in covariate_names.
        assert!(
            info.covariate_names.contains(&"RACE".to_string()),
            "Binary RACE should not be expanded: {:?}",
            info.covariate_names
        );

        // Subject 2 (RACE=-99) should not have a RACE pseudo-obs row.
        let lines: Vec<&str> = csv.lines().collect();
        let header = lines[0];
        let ft_col = header.split(',').position(|h| h == "FREMTYPE").unwrap();
        let id_col = header.split(',').position(|h| h == "ID").unwrap();

        // RACE is the 2nd covariate → FREMTYPE=200
        let subj2_race = lines.iter().find(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            parts[id_col] == "2" && parts[ft_col] == "200"
        });
        assert!(
            subj2_race.is_none(),
            "Subject 2 should not have RACE pseudo-obs (missing)"
        );
    }

    #[test]
    fn resolve_covariates_requires_block() {
        // Missing vs empty [covariates] block → distinct, accurate messages.
        let missing = resolve_frem_covariates(&[], None, None).unwrap_err();
        assert!(missing.contains("no [covariates] block"), "got: {missing}");
        let empty = resolve_frem_covariates(&[], None, Some(&[])).unwrap_err();
        assert!(empty.contains("is empty"), "got: {empty}");
        // A filter doesn't change the requirement.
        assert!(resolve_frem_covariates(&["WT".to_string()], None, None).is_err());
    }
}
