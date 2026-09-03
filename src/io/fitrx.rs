//! `.fitrx` save/load — zip-of-JSON-plus-CSV bundle for fit objects.
//!
//! Layout (all entries are deflate-compressed inside a zip archive):
//!
//! - `manifest.json`     — format version, ferx version, timestamp, entry index
//! - `fit.json`          — scalars / vectors / matrices on `FitResult`
//! - `ebes.csv`          — per-subject EBEs (`ID, eta_1..eta_n, ofv_contribution, n_obs`;
//!                          a mixture fit appends `MIXEST, PMIX_1..K`)
//! - `ebes_kappa.csv`    — per-(subject, occasion) kappa EBEs (only when `n_kappa > 0`)
//! - `conddist.csv`      — per-subject conditional η mean/SD/mode (only when `conddist = true`, SAEM-only)
//! - `predictions.csv`   — per-observation predictions joined with TIME/DV
//! - `model.ferx`        — verbatim model source
//! - `warnings.txt`      — one warning per line (mirrors `fit.json` for grep)
//! - `data.csv`          — copy of the input NONMEM CSV (only when caller opts in)
//!
//! See `docs/file-formats/fitrx.qmd` for the field-by-field schema.

use crate::types::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

pub const FORMAT_VERSION: &str = "1";

// Serde helpers for non-finite floats (non-finite -> JSON `null`, `null` ->
// `NaN`) live in the shared `crate::io::serde_nan` module; fields reference them
// via `#[serde(with = "crate::io::serde_nan::{scalar,vec,vec_vec,opt}")]`.

/// Errors from `.fitrx` save/load.
#[derive(Debug, thiserror::Error)]
pub enum FitrxError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported .fitrx format_version {0:?}; expected {expected:?}", expected = FORMAT_VERSION)]
    UnsupportedVersion(String),
    #[error("corrupt or missing entry: {0}")]
    Corrupt(String),
}

/// Options for [`save_fit`].
#[derive(Debug, Clone, Default)]
pub struct SaveFitOptions {
    /// When `Some(path)`, the file at `path` is embedded verbatim as `data.csv`
    /// inside the archive. When `None`, no data is bundled.
    pub include_data: Option<PathBuf>,
}

/// Result of [`load_fit`].
#[derive(Debug, Clone)]
pub struct LoadedFit {
    pub fit: FitResult,
    pub model_source: String,
    /// `Some` only when `data.csv` was bundled. Re-parsed via the standard
    /// NONMEM CSV reader; covariate auto-detection uses the same defaults.
    pub population: Option<Population>,
    pub manifest: Manifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: String,
    pub ferx_version: String,
    pub model_name: String,
    pub created_at: String,
    pub entries: Vec<String>,
}

// ---------------------------------------------------------------------------
// Wire structs (decoupled from `FitResult`)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct FitWire {
    method: String,
    method_chain: Vec<String>,
    converged: bool,
    ofv: f64,
    aic: f64,
    bic: f64,
    n_obs: usize,
    n_subjects: usize,
    n_parameters: usize,
    n_iterations: usize,
    interaction: bool,
    wall_time_secs: f64,
    // Absent on bundles saved before #713 split out per-method / covariance
    // timing; loaders default to an empty Vec / 0.0.
    #[serde(default)]
    method_wall_times_secs: Vec<f64>,
    #[serde(default)]
    covariance_wall_time_secs: f64,
    n_threads_used: usize,
    uses_ode_solver: bool,
    uses_sde: bool,
    gradient_method_inner: String,
    gradient_method_outer: String,
    nlopt_missing_algorithms: Vec<String>,
    covariance_status: String,
    covariance_n_evals_estimated: Option<usize>,
    trace_path: Option<String>,
    ebe_convergence_warnings: u32,
    max_unconverged_subjects: u32,
    total_ebe_fallbacks: u32,
    warnings: Vec<String>,
    saem_mu_ref_m_step_evals_saved: Option<u64>,
    #[serde(default)]
    saem_n_subjects_hmc: Option<usize>,

    theta: ThetaWire,
    omega: OmegaWire,
    sigma: SigmaWire,
    error_model: String,
    #[serde(with = "crate::io::serde_nan::scalar")]
    shrinkage_eps: f64,
    #[serde(default = "default_nan", with = "crate::io::serde_nan::scalar")]
    iwres_lag1_r: f64,
    #[serde(default = "default_nan", with = "crate::io::serde_nan::scalar")]
    dw_statistic: f64,
    covariance_matrix: Option<MatrixWire>,
    cov_eigenvalues: Option<Vec<f64>>,
    #[serde(with = "crate::io::serde_nan::opt")]
    cov_condition_number: Option<f64>,

    sir: Option<SirWire>,
    iov: Option<IovWire>,

    eta_param_info: Vec<EtaParamInfoWire>,
    model_name: String,
    ferx_version: String,
    // Absent on bundles saved before #704 added this field; loaders default
    // to `EnvironmentInfo::default()`'s "unknown" placeholder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    environment: Option<crate::environment::EnvironmentInfo>,
    // Source-file provenance. All four are optional and default to absent on
    // older bundles (produced before these fields existed) so they keep loading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_text: Option<String>,
    #[serde(default)]
    theta_init: Vec<f64>,
    #[serde(default)]
    omega_init: Option<MatrixWire>,
    #[serde(default)]
    sigma_init: Vec<f64>,
    #[serde(default)]
    obs_time_range: Option<(f64, f64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_gradient: Option<Vec<f64>>,
    // ── Run settings ─────────────────────────────────────────────────────────
    #[serde(default)]
    optimizer: String,
    #[serde(default = "default_one")]
    n_starts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    multi_start_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    saem_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sir_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imp_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    npde_seed: Option<u64>,
    #[serde(default)]
    bloq_method: String,
    #[serde(default)]
    outer_maxiter: usize,
    #[serde(default)]
    outer_gtol: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inits_from_nca: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    covariate_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    input_columns: Vec<String>,
    /// `[covariate_nn]` block metadata. Absent (None) on bundles produced
    /// before this field existed or when ferx-core was built without
    /// `--features nn`. Loaders gracefully default to an empty Vec.
    #[cfg(feature = "nn")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    neural_networks: Option<Vec<crate::types::NeuralNetworkInfo>>,
}

#[derive(Serialize, Deserialize)]
struct ThetaWire {
    names: Vec<String>,
    estimates: Vec<f64>,
    se: Option<Vec<f64>>,
    fixed: Vec<bool>,
    transform: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct OmegaWire {
    names: Vec<String>,
    matrix: MatrixWire,
    se: Option<Vec<f64>>,
    fixed: Vec<bool>,
    log_transformed: Vec<bool>,
    param_corr: Option<MatrixWire>,
    #[serde(with = "crate::io::serde_nan::vec")]
    shrinkage: Vec<f64>,
    /// Per-eta SD-init flag (see `FitResult.omega_init_as_sd`). Optional /
    /// defaulted so .fitrx files written before issue #5 still load — older
    /// bundles deserialize with an all-`false` vector.
    #[serde(default)]
    init_as_sd: Vec<bool>,
}

#[derive(Serialize, Deserialize)]
struct SigmaWire {
    names: Vec<String>,
    estimates: Vec<f64>,
    se: Option<Vec<f64>>,
    fixed: Vec<bool>,
    types: Vec<String>,
    /// Per-sigma SD-init flag (see `FitResult.sigma_init_as_sd`). Defaulted
    /// for backward compatibility with .fitrx files from before issue #5.
    #[serde(default)]
    init_as_sd: Vec<bool>,
    /// `block_sigma` correlations. Absent in bundles written before issue #1100
    /// and omitted for the common diagonal-sigma case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    residual_correlations: Vec<crate::types::ResidualCorrelation>,
    /// FIX flags parallel to `residual_correlations` (#847). Absent in bundles
    /// written before the off-diagonal became estimable, where every correlation
    /// was fixed by construction — hence the `true` fill on load.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    residual_correlation_fixed: Vec<bool>,
    /// Standard errors for the estimated correlations, natural ρ scale (#847).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    se_residual_correlations: Option<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct SirWire {
    ci_theta: Option<Vec<(f64, f64)>>,
    ci_omega: Option<Vec<(f64, f64)>>,
    ci_sigma: Option<Vec<(f64, f64)>>,
    ess: Option<f64>,
    /// Retained packed-parameter draws when `sir_keep_samples = true` was set.
    /// `None` otherwise; consumed by `simulate_with_uncertainty()`.
    resamples_packed: Option<Vec<Vec<f64>>>,
}

#[derive(Serialize, Deserialize)]
struct IovWire {
    kappa_names: Vec<String>,
    kappa_fixed: Vec<bool>,
    se_kappa: Option<Vec<f64>>,
    #[serde(with = "crate::io::serde_nan::vec")]
    shrinkage_kappa: Vec<f64>,
    /// Per-occasion kappa shrinkage: `[occ_idx][kappa_idx]`.
    /// Defaulted for backward compatibility with older .fitrx files.
    #[serde(default, with = "crate::io::serde_nan::vec_vec")]
    shrinkage_kappa_by_occ: Vec<Vec<f64>>,
    omega_iov: MatrixWire,
    omega_iov_param_corr: Option<MatrixWire>,
    /// Per-kappa SD-init flag. Defaulted for backward compatibility with
    /// .fitrx files from before issue #5.
    #[serde(default)]
    kappa_init_as_sd: Vec<bool>,
    /// Per-kappa `weight = <expr>` source text and the median arm weight it
    /// evaluated to (#1031). Both empty for a model with no weighted kappa, and
    /// defaulted so a bundle written before #1031 still reads.
    #[serde(default)]
    kappa_weights: Vec<Option<String>>,
    #[serde(default)]
    kappa_weight_typical: Vec<Option<f64>>,
}

#[derive(Serialize, Deserialize)]
struct EtaParamInfoWire {
    eta_name: String,
    param_type: String,
    linked_theta: Option<String>,
    individual_param_name: String,
}

/// Row-major dense matrix serialization. `data.len() == rows * cols`.
/// Non-finite entries (NaN, ±Inf) are written as JSON `null` and read back
/// as `NaN`, so matrices from failed fits survive a round-trip without
/// hitting serde_json's "non-finite f64 has no JSON representation" wall.
#[derive(Serialize, Deserialize)]
struct MatrixWire {
    rows: usize,
    cols: usize,
    #[serde(with = "crate::io::serde_nan::vec")]
    data: Vec<f64>,
}

impl MatrixWire {
    fn from(m: &DMatrix<f64>) -> Self {
        let rows = m.nrows();
        let cols = m.ncols();
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(m[(i, j)]);
            }
        }
        Self { rows, cols, data }
    }

    fn into_dmatrix(self) -> Result<DMatrix<f64>, FitrxError> {
        if self.data.len() != self.rows * self.cols {
            return Err(FitrxError::Corrupt(format!(
                "matrix data length {} does not match {}×{}",
                self.data.len(),
                self.rows,
                self.cols
            )));
        }
        // DMatrix::from_iterator fills column-major; build row-major manually.
        let mut m = DMatrix::<f64>::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m[(i, j)] = self.data[i * self.cols + j];
            }
        }
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// Enum <-> string mappings (kept local so types.rs stays unchanged)
// ---------------------------------------------------------------------------

fn default_nan() -> f64 {
    f64::NAN
}

fn default_one() -> usize {
    1
}

/// Generate the `<enum> <-> &str` wire-format mapping pair for one enum.
///
/// Emits `$to_fn(v) -> &'static str` and `$from_fn(&str) -> Result<$enum, _>`
/// from a single variant/literal table, so the two can never drift apart. The
/// string literals are the on-disk `.fitrx` wire format, so they must stay
/// byte-identical; the `$to_arg` type lets a mapping take its enum by reference
/// (e.g. `&CovarianceStatus`) while `$from_fn` always returns it by value.
macro_rules! str_enum_map {
    (
        $enum:ident, $label:literal,
        $to_fn:ident($to_arg:ty), $from_fn:ident,
        { $($variant:ident => $lit:literal),+ $(,)? }
    ) => {
        fn $to_fn(v: $to_arg) -> &'static str {
            match v {
                $($enum::$variant => $lit,)+
            }
        }

        fn $from_fn(s: &str) -> Result<$enum, FitrxError> {
            Ok(match s {
                $($lit => $enum::$variant,)+
                _ => {
                    return Err(FitrxError::Corrupt(format!(
                        concat!("unknown ", $label, " {:?}"),
                        s
                    )))
                }
            })
        }
    };
}

str_enum_map!(EstimationMethod, "method", method_to_str(EstimationMethod), method_from_str, {
    Foce => "foce",
    FoceI => "focei",
    FoceGn => "foce_gn",
    FoceGnHybrid => "foce_gn_hybrid",
    Saem => "saem",
    Imp => "imp",
    Impmap => "impmap",
    Bayes => "bayes",
    // `agq` was removed (#251); it was unreleased, so no persisted bundle carries it.
    Laplace => "laplace",
    Vi => "vi",
});

str_enum_map!(ErrorModel, "error_model", error_model_to_str(ErrorModel), error_model_from_str, {
    Additive => "additive",
    Proportional => "proportional",
    Combined => "combined",
});

str_enum_map!(
    CovarianceStatus,
    "covariance_status",
    covariance_status_to_str(&CovarianceStatus),
    covariance_status_from_str,
    {
        NotRequested => "not_requested",
        Computed => "computed",
        Failed => "failed",
        SirFallback => "sir_fallback",
    }
);

str_enum_map!(
    ThetaTransform,
    "theta_transform",
    theta_transform_to_str(ThetaTransform),
    theta_transform_from_str,
    {
        Identity => "identity",
        Log => "log",
        Logit => "logit",
        LogitProbability => "logit_probability",
    }
);

str_enum_map!(SigmaType, "sigma_type", sigma_type_to_str(SigmaType), sigma_type_from_str, {
    Proportional => "proportional",
    Additive => "additive",
});

str_enum_map!(
    EtaParamType,
    "eta_param_type",
    eta_param_type_to_str(EtaParamType),
    eta_param_type_from_str,
    {
        LogNormal => "log_normal",
        Additive => "additive",
        Logit => "logit",
        LogitProbability => "logit_probability",
        Custom => "custom",
    }
);

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

/// Write a fit bundle to `path`.
///
/// `model_source` is the verbatim text of the `.ferx` model; it is embedded so
/// a future [`load_fit`] can recompile the model and run `predict()` against
/// the loaded fit.
pub fn save_fit(
    result: &FitResult,
    population: &Population,
    model_source: &str,
    path: &Path,
    opts: SaveFitOptions,
) -> Result<(), FitrxError> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    let zopts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    let mut entries: Vec<String> = Vec::new();
    entries.push("manifest.json".into());

    // --- fit.json ----------------------------------------------------------
    let wire = build_fit_wire(result);
    zip.start_file("fit.json", zopts)?;
    zip.write_all(&serde_json::to_vec_pretty(&wire)?)?;
    zip.write_all(b"\n")?;
    entries.push("fit.json".into());

    // --- ebes.csv ----------------------------------------------------------
    zip.start_file("ebes.csv", zopts)?;
    write_ebes_csv(&mut zip, result)?;
    entries.push("ebes.csv".into());

    // --- ebes_kappa.csv (only when IOV present) ----------------------------
    if !result.kappa_names.is_empty() && !result.ebe_kappas.is_empty() {
        zip.start_file("ebes_kappa.csv", zopts)?;
        write_ebes_kappa_csv(&mut zip, result)?;
        entries.push("ebes_kappa.csv".into());
    }

    // --- conddist.csv (only when the SAEM conditional-distribution pass ran) ---
    if let Some(cd) = &result.cond_dist {
        zip.start_file("conddist.csv", zopts)?;
        write_conddist_csv(&mut zip, result, cd)?;
        entries.push("conddist.csv".into());
    }

    // --- predictions.csv ---------------------------------------------------
    zip.start_file("predictions.csv", zopts)?;
    write_predictions_csv(&mut zip, result, population)?;
    entries.push("predictions.csv".into());

    // --- model.ferx --------------------------------------------------------
    let effective_source = result.model_text.as_deref().unwrap_or(model_source);
    zip.start_file("model.ferx", zopts)?;
    zip.write_all(effective_source.as_bytes())?;
    entries.push("model.ferx".into());

    // --- warnings.txt ------------------------------------------------------
    zip.start_file("warnings.txt", zopts)?;
    for w in &result.warnings {
        writeln!(zip, "{}", w)?;
    }
    entries.push("warnings.txt".into());

    // --- data.csv (optional) ----------------------------------------------
    if let Some(data_path) = &opts.include_data {
        let mut src = File::open(data_path).map_err(|e| {
            FitrxError::Io(std::io::Error::new(
                e.kind(),
                format!("opening {} for bundling: {}", data_path.display(), e),
            ))
        })?;
        let mut buf = Vec::new();
        src.read_to_end(&mut buf)?;
        zip.start_file("data.csv", zopts)?;
        zip.write_all(&buf)?;
        entries.push("data.csv".into());
    }

    // --- manifest.json (written last so it can list every entry) ----------
    let manifest = Manifest {
        format_version: FORMAT_VERSION.into(),
        ferx_version: result.ferx_version.clone(),
        model_name: result.model_name.clone(),
        created_at: iso8601_now(),
        entries,
    };
    zip.start_file("manifest.json", zopts)?;
    zip.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
    zip.write_all(b"\n")?;

    zip.finish()?;
    Ok(())
}

fn build_fit_wire(r: &FitResult) -> FitWire {
    FitWire {
        method: method_to_str(r.method).into(),
        method_chain: r
            .method_chain
            .iter()
            .map(|m| method_to_str(*m).into())
            .collect(),
        converged: r.converged,
        ofv: r.ofv,
        aic: r.aic,
        bic: r.bic,
        n_obs: r.n_obs,
        n_subjects: r.n_subjects,
        n_parameters: r.n_parameters,
        n_iterations: r.n_iterations,
        interaction: r.interaction,
        wall_time_secs: r.wall_time_secs,
        method_wall_times_secs: r.method_wall_times_secs.clone(),
        covariance_wall_time_secs: r.covariance_wall_time_secs,
        n_threads_used: r.n_threads_used,
        uses_ode_solver: r.uses_ode_solver,
        uses_sde: r.uses_sde,
        gradient_method_inner: r.gradient_method_inner.clone(),
        gradient_method_outer: r.gradient_method_outer.clone(),
        nlopt_missing_algorithms: r.nlopt_missing_algorithms.clone(),
        covariance_status: covariance_status_to_str(&r.covariance_status).into(),
        covariance_n_evals_estimated: r.covariance_n_evals_estimated,
        trace_path: r.trace_path.clone(),
        ebe_convergence_warnings: r.ebe_convergence_warnings,
        max_unconverged_subjects: r.max_unconverged_subjects,
        total_ebe_fallbacks: r.total_ebe_fallbacks,
        warnings: r.warnings.clone(),
        saem_mu_ref_m_step_evals_saved: r.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: r.saem_n_subjects_hmc,
        theta: ThetaWire {
            names: r.theta_names.clone(),
            estimates: r.theta.clone(),
            se: r.se_theta.clone(),
            fixed: r.theta_fixed.clone(),
            transform: r
                .theta_transform
                .iter()
                .map(|t| theta_transform_to_str(*t).into())
                .collect(),
        },
        omega: OmegaWire {
            names: r.eta_names.clone(),
            matrix: MatrixWire::from(&r.omega),
            se: r.se_omega.clone(),
            fixed: r.omega_fixed.clone(),
            log_transformed: r.eta_log_transformed.clone(),
            param_corr: r.omega_param_corr.as_ref().map(MatrixWire::from),
            shrinkage: r.shrinkage_eta.clone(),
            init_as_sd: r.omega_init_as_sd.clone(),
        },
        sigma: SigmaWire {
            names: r.sigma_names.clone(),
            estimates: r.sigma.clone(),
            se: r.se_sigma.clone(),
            fixed: r.sigma_fixed.clone(),
            types: r
                .sigma_types
                .iter()
                .map(|t| sigma_type_to_str(*t).into())
                .collect(),
            init_as_sd: r.sigma_init_as_sd.clone(),
            residual_correlations: r.residual_correlations.clone(),
            residual_correlation_fixed: r.residual_correlation_fixed.clone(),
            se_residual_correlations: r.se_residual_correlations.clone(),
        },
        error_model: error_model_to_str(r.error_model).into(),
        shrinkage_eps: r.shrinkage_eps,
        iwres_lag1_r: r.iwres_lag1_r,
        dw_statistic: r.dw_statistic,
        covariance_matrix: r.covariance_matrix.as_ref().map(MatrixWire::from),
        cov_eigenvalues: r.cov_eigenvalues.clone(),
        cov_condition_number: r.cov_condition_number,
        sir: if r.sir_ci_theta.is_some() || r.sir_ess.is_some() || r.sir_resamples_packed.is_some()
        {
            Some(SirWire {
                ci_theta: r.sir_ci_theta.clone(),
                ci_omega: r.sir_ci_omega.clone(),
                ci_sigma: r.sir_ci_sigma.clone(),
                ess: r.sir_ess,
                resamples_packed: r.sir_resamples_packed.clone(),
            })
        } else {
            None
        },
        iov: r.omega_iov.as_ref().map(|m| IovWire {
            kappa_names: r.kappa_names.clone(),
            kappa_fixed: r.kappa_fixed.clone(),
            se_kappa: r.se_kappa.clone(),
            shrinkage_kappa: r.shrinkage_kappa.clone(),
            shrinkage_kappa_by_occ: r.shrinkage_kappa_by_occ.clone(),
            omega_iov: MatrixWire::from(m),
            omega_iov_param_corr: r.omega_iov_param_corr.as_ref().map(MatrixWire::from),
            kappa_init_as_sd: r.kappa_init_as_sd.clone(),
            kappa_weights: r.kappa_weights.clone(),
            kappa_weight_typical: r.kappa_weight_typical.clone(),
        }),
        eta_param_info: r
            .eta_param_info
            .iter()
            .map(|i| EtaParamInfoWire {
                eta_name: i.eta_name.clone(),
                param_type: eta_param_type_to_str(i.param_type).into(),
                linked_theta: i.linked_theta.clone(),
                individual_param_name: i.individual_param_name.clone(),
            })
            .collect(),
        model_name: r.model_name.clone(),
        ferx_version: r.ferx_version.clone(),
        environment: Some(r.environment.clone()),
        model_path: r.model_path.clone(),
        data_path: r.data_path.clone(),
        model_hash: r.model_hash.clone(),
        data_hash: r.data_hash.clone(),
        model_text: r.model_text.clone(),
        theta_init: r.theta_init.clone(),
        omega_init: Some(MatrixWire::from(&r.omega_init)),
        sigma_init: r.sigma_init.clone(),
        obs_time_range: r.obs_time_range,
        final_gradient: r.final_gradient.clone(),
        optimizer: r.optimizer.clone(),
        n_starts: r.n_starts,
        multi_start_seed: r.multi_start_seed,
        saem_seed: r.saem_seed,
        sir_seed: r.sir_seed,
        imp_seed: r.imp_seed,
        npde_seed: r.npde_seed,
        bloq_method: r.bloq_method.clone(),
        outer_maxiter: r.outer_maxiter,
        outer_gtol: r.outer_gtol,
        inits_from_nca: r.inits_from_nca.clone(),
        covariate_names: r.covariate_names.clone(),
        input_columns: r.input_columns.clone(),
        #[cfg(feature = "nn")]
        neural_networks: if r.neural_networks.is_empty() {
            None
        } else {
            Some(r.neural_networks.clone())
        },
    }
}

fn write_ebes_csv<W: Write>(w: &mut W, r: &FitResult) -> Result<(), FitrxError> {
    let n_eta = r.omega.nrows();
    let mut header = String::from("ID");
    for k in 0..n_eta {
        let name = r.eta_names.get(k).map(|s| s.as_str()).unwrap_or("eta");
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",ofv_contribution,n_obs");
    // Mixture (#983): a mixture fit carries per-subject `MIXEST` (most-probable
    // class) and `PMIX_1..K` (posterior class weights). Append them after the
    // fixed base columns so a non-mixture bundle is byte-identical to before and
    // the reader can pick them up by header name. `K` comes from the first
    // subject that has posteriors (all subjects of a mixture fit have them).
    let mix_k = r
        .subjects
        .iter()
        .find_map(|s| s.pmix.as_ref().map(|p| p.len()));
    if let Some(k) = mix_k {
        header.push_str(",MIXEST");
        for j in 1..=k {
            header.push_str(&format!(",PMIX_{j}"));
        }
    }
    writeln!(w, "{}", header)?;
    for s in &r.subjects {
        let mut row = csv_escape(&s.id);
        for k in 0..n_eta {
            row.push(',');
            row.push_str(&fmt_f64(s.eta[k]));
        }
        row.push(',');
        row.push_str(&fmt_f64(s.ofv_contribution));
        row.push(',');
        row.push_str(&s.n_obs.to_string());
        if let Some(k) = mix_k {
            // MIXEST is 1-based; write 0 for a subject that somehow lacks it so
            // the column count stays rectangular (should not happen — posteriors
            // are set together for every subject of a mixture fit).
            row.push(',');
            row.push_str(&s.mixest.unwrap_or(0).to_string());
            for j in 0..k {
                row.push(',');
                let p = s
                    .pmix
                    .as_ref()
                    .and_then(|v| v.get(j))
                    .copied()
                    .unwrap_or(0.0);
                row.push_str(&fmt_f64(p));
            }
        }
        writeln!(w, "{}", row)?;
    }
    Ok(())
}

fn write_ebes_kappa_csv<W: Write>(w: &mut W, r: &FitResult) -> Result<(), FitrxError> {
    let n_kappa = r.kappa_names.len();
    let mut header = String::from("ID,OCC");
    for k in 0..n_kappa {
        header.push(',');
        header.push_str(&r.kappa_names[k]);
    }
    writeln!(w, "{}", header)?;
    for (si, s) in r.subjects.iter().enumerate() {
        if si >= r.ebe_kappas.len() {
            continue;
        }
        for (occ_idx, kappa) in r.ebe_kappas[si].iter().enumerate() {
            let mut row = csv_escape(&s.id);
            row.push(',');
            row.push_str(&(occ_idx + 1).to_string());
            for k in 0..n_kappa {
                row.push(',');
                row.push_str(&fmt_f64(kappa.get(k).copied().unwrap_or(f64::NAN)));
            }
            writeln!(w, "{}", row)?;
        }
    }
    Ok(())
}

fn write_conddist_csv<W: Write>(w: &mut W, r: &FitResult, cd: &CondDist) -> Result<(), FitrxError> {
    writeln!(w, "ID,ETA,COND_MEAN,COND_SD,COND_MODE")?;
    for (i, s) in r.subjects.iter().enumerate() {
        let mean_i = cd.cond_mean.get(i);
        let sd_i = cd.cond_sd.get(i);
        for (j, eta_name) in r.eta_names.iter().enumerate() {
            let mean = mean_i.and_then(|m| m.get(j)).copied().unwrap_or(f64::NAN);
            let sd = sd_i.and_then(|v| v.get(j)).copied().unwrap_or(f64::NAN);
            // The conditional mode is the EBE already on the subject result.
            let mode = s.eta.get(j).copied().unwrap_or(f64::NAN);
            let mut row = csv_escape(&s.id);
            row.push(',');
            row.push_str(&csv_escape(eta_name));
            row.push(',');
            row.push_str(&fmt_f64(mean));
            row.push(',');
            row.push_str(&fmt_f64(sd));
            row.push(',');
            row.push_str(&fmt_f64(mode));
            writeln!(w, "{}", row)?;
        }
    }
    Ok(())
}

fn write_predictions_csv<W: Write>(
    w: &mut W,
    r: &FitResult,
    p: &Population,
) -> Result<(), FitrxError> {
    let any_cens = r.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = p.subjects.iter().any(|s| !s.occasions.is_empty());
    // NPDE/NPD are only present when the fit ran with npde_nsim > 0.
    let any_npde = r
        .subjects
        .iter()
        .any(|s| !s.npde.is_empty() || !s.npd.is_empty());

    let mut header = String::from("ID,TIME,DV,PRED,IPRED,CWRES,IWRES");
    if any_npde {
        header.push_str(",NPDE,NPD");
    }
    header.push_str(",EBE_OFV,N_OBS");
    if any_cens {
        header.push_str(",CENS");
    }
    if any_occ {
        header.push_str(",OCC");
    }
    writeln!(w, "{}", header)?;

    for (si, sr) in r.subjects.iter().enumerate() {
        let subj = &p.subjects[si];
        for j in 0..sr.ipred.len() {
            let mut row = csv_escape(&sr.id);
            row.push(',');
            // Report the raw data TIME (so the bundle's predictions.csv joins back
            // to the input CSV and to the sdtab, which is also raw); `obs_times`
            // may be the internal shifted timeline for stacked reset occasions.
            // Falls back to `obs_times` when no raw vector was recorded.
            row.push_str(&fmt_f64(
                subj.obs_raw_times
                    .get(j)
                    .copied()
                    .unwrap_or(subj.obs_times[j]),
            ));
            row.push(',');
            row.push_str(&fmt_f64(subj.observations[j]));
            row.push(',');
            row.push_str(&fmt_f64(sr.pred[j]));
            row.push(',');
            row.push_str(&fmt_f64(sr.ipred[j]));
            row.push(',');
            row.push_str(&fmt_f64(sr.cwres[j]));
            row.push(',');
            row.push_str(&fmt_f64(sr.iwres[j]));
            if any_npde {
                row.push(',');
                row.push_str(&fmt_f64(sr.npde.get(j).copied().unwrap_or(f64::NAN)));
                row.push(',');
                row.push_str(&fmt_f64(sr.npd.get(j).copied().unwrap_or(f64::NAN)));
            }
            row.push(',');
            row.push_str(&fmt_f64(sr.ofv_contribution));
            row.push(',');
            row.push_str(&sr.n_obs.to_string());
            if any_cens {
                row.push(',');
                row.push_str(&(sr.cens.get(j).copied().unwrap_or(0)).to_string());
            }
            if any_occ {
                row.push(',');
                row.push_str(&(subj.occasions.get(j).copied().unwrap_or(0)).to_string());
            }
            writeln!(w, "{}", row)?;
        }
    }
    Ok(())
}

fn fmt_f64(v: f64) -> String {
    // Shared CSV float formatter (NaN → empty, else 6 dp). See
    // `io::output::fmt_num` for the single source of truth.
    crate::io::output::fmt_num(v)
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        s.to_string()
    }
}

fn iso8601_now() -> String {
    // Hand-formatted UTC timestamp from SystemTime to avoid a `time` dep.
    // Accurate enough for "when was this fit saved"; we don't need leap-second
    // precision and we explicitly ignore subsecond fields.
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let (y, mo, d, h, mi, s) = epoch_to_utc(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

/// Convert seconds-since-epoch to (year, month, day, hour, minute, second) UTC.
/// Civil-from-days algorithm by Howard Hinnant — handles dates through year 9999.
fn epoch_to_utc(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let time_of_day = secs.rem_euclid(86_400) as u32;
    let h = time_of_day / 3600;
    let mi = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    // Days since 1970-01-01 → civil date (Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d, h, mi, s)
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Read a fit bundle from `path`.
pub fn load_fit(path: &Path) -> Result<LoadedFit, FitrxError> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let manifest: Manifest = read_json(&mut archive, "manifest.json")?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(FitrxError::UnsupportedVersion(manifest.format_version));
    }

    let wire: FitWire = read_json(&mut archive, "fit.json")?;
    let ebes_csv = read_text(&mut archive, "ebes.csv")?;
    let ebes_kappa_csv = if archive.file_names().any(|n| n == "ebes_kappa.csv") {
        Some(read_text(&mut archive, "ebes_kappa.csv")?)
    } else {
        None
    };
    let conddist_csv = if archive.file_names().any(|n| n == "conddist.csv") {
        Some(read_text(&mut archive, "conddist.csv")?)
    } else {
        None
    };
    let preds_csv = read_text(&mut archive, "predictions.csv")?;
    let model_source = read_text(&mut archive, "model.ferx")?;

    // data.csv is optional — re-parse only when present.
    let population = if archive.file_names().any(|n| n == "data.csv") {
        let data_csv_bytes = read_bytes(&mut archive, "data.csv")?;
        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), &data_csv_bytes)?;
        // data.csv is bundled with its original headers, so honour the model's
        // `[data]` column mapping (#730) when re-reading — otherwise a fit that
        // mapped e.g. `DV = CONC` cannot round-trip through load_fit. Parse the
        // bundled model source for the map; fall back to no mapping if it can't
        // be parsed (leave the read to fail with its own diagnostic).
        let column_map = crate::parser::model_parser::parse_full_model(&model_source)
            .map(|m| m.column_map)
            .unwrap_or_default();
        Some(
            crate::io::datareader::read_nonmem_csv_mapped(tmp.path(), None, None, &column_map)
                .map_err(FitrxError::Corrupt)?,
        )
    } else {
        None
    };

    let n_eta = wire.omega.matrix.rows;
    let subjects = parse_subjects(&ebes_csv, &preds_csv, ebes_kappa_csv.as_deref(), n_eta)?;
    let ebe_kappas = if let Some(csv) = ebes_kappa_csv.as_deref() {
        parse_ebe_kappas(
            csv,
            &subjects.iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
        )?
    } else {
        Vec::new()
    };

    let mut fit = wire_to_fit_result(wire, subjects, ebe_kappas)?;
    fit.model_text = Some(model_source.clone());
    if let Some(csv) = conddist_csv.as_deref() {
        fit.cond_dist = Some(parse_conddist_csv(csv, &fit)?);
    }

    Ok(LoadedFit {
        fit,
        model_source,
        population,
        manifest,
    })
}

fn read_bytes<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>, FitrxError> {
    let mut file = archive
        .by_name(name)
        .map_err(|_| FitrxError::Corrupt(format!("missing entry {}", name)))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

fn read_text<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<String, FitrxError> {
    let buf = read_bytes(archive, name)?;
    String::from_utf8(buf)
        .map_err(|e| FitrxError::Corrupt(format!("invalid UTF-8 in entry {}: {}", name, e)))
}

fn read_json<T: serde::de::DeserializeOwned, R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<T, FitrxError> {
    let buf = read_bytes(archive, name)?;
    Ok(serde_json::from_slice(&buf)?)
}

/// Column indices of the `PMIX_{k}` headers, ordered by class number `k` rather
/// than raw header position (#984 review). ferx writes `PMIX_1..K` in order, but a
/// CSV tool that reordered columns (e.g. alphabetically — `PMIX_10` before
/// `PMIX_2`) would otherwise silently swap class probabilities on restore, since
/// the restored `pmix` vector is indexed by class. A non-numeric suffix sorts last
/// deterministically.
fn pmix_column_order(header_names: &[&str]) -> Vec<usize> {
    let mut named: Vec<(usize, usize)> = header_names
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            c.strip_prefix("PMIX_")
                .map(|suf| (suf.parse::<usize>().unwrap_or(usize::MAX), i))
        })
        .collect();
    named.sort_by_key(|&(cls, _)| cls);
    named.into_iter().map(|(_, i)| i).collect()
}

fn parse_subjects(
    ebes_csv: &str,
    preds_csv: &str,
    _ebes_kappa_csv: Option<&str>,
    n_eta: usize,
) -> Result<Vec<SubjectResult>, FitrxError> {
    // ebes.csv → ordered list of (id, eta, ofv_contribution, n_obs).
    let mut lines = ebes_csv.lines();
    let header = lines
        .next()
        .ok_or_else(|| FitrxError::Corrupt("ebes.csv: empty".into()))?;
    // The base columns (ID, eta_1..n, ofv_contribution, n_obs) are fixed at the
    // front; a mixture bundle appends MIXEST + PMIX_1..K after them (#983). So the
    // base count is a *minimum*, and the optional mixture columns are located by
    // header name rather than position.
    let base_cols = 1 + n_eta + 2;
    let header_names: Vec<&str> = header.split(',').collect();
    let header_cols = header_names.len();
    if header_cols < base_cols {
        return Err(FitrxError::Corrupt(format!(
            "ebes.csv header has {} columns, expected at least {}",
            header_cols, base_cols
        )));
    }
    // Optional mixture columns (#983): MIXEST + PMIX_1..K, by header name.
    let mixest_idx = header_names.iter().position(|&c| c == "MIXEST");
    // Restore PMIX columns in class-number order, not raw header order (#984
    // review): the restored `pmix` vector is indexed by class, so a CSV tool that
    // reordered columns would otherwise silently swap class probabilities.
    let pmix_idxs = pmix_column_order(&header_names);
    let mut subjects: Vec<SubjectResult> = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() != header_cols {
            return Err(FitrxError::Corrupt(format!(
                "ebes.csv row {} has {} fields, expected {}",
                i + 1,
                fields.len(),
                header_cols
            )));
        }
        let id = fields[0].clone();
        let mut eta = DVector::<f64>::zeros(n_eta);
        for k in 0..n_eta {
            eta[k] = fields[1 + k]
                .parse::<f64>()
                .map_err(|_| FitrxError::Corrupt(format!("ebes.csv: bad eta in row {}", i + 1)))?;
        }
        let ofv = fields[1 + n_eta]
            .parse::<f64>()
            .map_err(|_| FitrxError::Corrupt(format!("ebes.csv: bad ofv in row {}", i + 1)))?;
        let n_obs = fields[2 + n_eta]
            .parse::<usize>()
            .map_err(|_| FitrxError::Corrupt(format!("ebes.csv: bad n_obs in row {}", i + 1)))?;
        // Mixture posteriors (#983): restore MIXEST + PMIX_1..K when the bundle
        // carries them. A `MIXEST` of 0 is the sentinel for "no class" → None.
        let mixest = mixest_idx.and_then(|mi| fields[mi].parse::<usize>().ok().filter(|&m| m != 0));
        let pmix = if pmix_idxs.is_empty() {
            None
        } else {
            let mut v = Vec::with_capacity(pmix_idxs.len());
            for &pi in &pmix_idxs {
                v.push(fields[pi].parse::<f64>().map_err(|_| {
                    FitrxError::Corrupt(format!("ebes.csv: bad PMIX in row {}", i + 1))
                })?);
            }
            Some(v)
        };
        subjects.push(SubjectResult {
            id,
            eta,
            ipred: Vec::new(),
            pred: Vec::new(),
            iwres: Vec::new(),
            cwres: Vec::new(),
            npde: Vec::new(),
            npd: Vec::new(),
            ofv_contribution: ofv,
            cens: Vec::new(),
            n_obs,
            // Mixture posteriors round-trip through ebes.csv's optional
            // MIXEST/PMIX_* columns (#983); None for a non-mixture bundle.
            pmix,
            mixest,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
            // The .fitrx checkpoint format carries only the Gaussian per-observation
            // vectors (`predictions.csv`), so a restored result has **no** discrete
            // rows and nothing recomputes them — `sdtab()` on a restored binary fit
            // emits no discrete rows, unlike the live run's own sdtab. Round-tripping
            // them belongs with the checkpoint format, not here (#897 follow-up).
            #[cfg(feature = "survival")]
            discrete_rows: Vec::new(),
        });
    }

    // predictions.csv → fill ipred/pred/iwres/cwres/cens positionally.
    let mut plines = preds_csv.lines();
    let pheader = plines
        .next()
        .ok_or_else(|| FitrxError::Corrupt("predictions.csv: empty".into()))?;
    let col: HashMap<&str, usize> = pheader
        .split(',')
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect();
    let require = |c: &str| -> Result<usize, FitrxError> {
        col.get(c)
            .copied()
            .ok_or_else(|| FitrxError::Corrupt(format!("predictions.csv missing column {}", c)))
    };
    let id_i = require("ID")?;
    let pred_i = require("PRED")?;
    let ipred_i = require("IPRED")?;
    let cwres_i = require("CWRES")?;
    let iwres_i = require("IWRES")?;
    // NPDE/NPD are optional — present only when the fit ran with npde_nsim > 0.
    let npde_i = col.get("NPDE").copied();
    let npd_i = col.get("NPD").copied();
    let cens_i = col.get("CENS").copied();
    // N_OBS is always written by `write_predictions_csv`, but treat it as
    // optional so a hand-built bundle without it still loads. When present, it
    // is validated per row against the subject we're filling — the ID order
    // check alone can't catch two same-ID blocks written in the wrong order.
    let n_obs_i = col.get("N_OBS").copied();

    // Rows are written one contiguous block per subject, in the same order as
    // ebes.csv (see `write_predictions_csv`), with exactly `n_obs` rows each.
    // Consume `subjects[i].n_obs` rows for subject i before advancing, using
    // the row ID only as a consistency check. Keying on ID instead (as this
    // once did) collapses distinct subjects that share a textual ID — e.g. an
    // ID reused across studies or a reset-split subject — routing every
    // duplicate's rows to one subject and leaving the other with zero, which
    // then trips the n_obs cross-validation below.
    let parse_opt = |s: &str| -> f64 {
        if s.is_empty() {
            f64::NAN
        } else {
            s.parse().unwrap_or(f64::NAN)
        }
    };

    let mut cur = 0usize;
    let mut filled = 0usize;
    for (i, line) in plines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        let id = fields
            .get(id_i)
            .ok_or_else(|| FitrxError::Corrupt(format!("predictions.csv row {}: short", i + 1)))?;
        // Advance past subjects whose observation block is complete (including
        // zero-observation subjects, which consume no prediction rows).
        while cur < subjects.len() && filled == subjects[cur].n_obs {
            cur += 1;
            filled = 0;
        }
        if cur >= subjects.len() {
            return Err(FitrxError::Corrupt(format!(
                "predictions.csv row {}: more rows than ebes.csv accounts for (id {:?})",
                i + 1,
                id
            )));
        }
        let idx = cur;
        if id != &subjects[idx].id {
            return Err(FitrxError::Corrupt(format!(
                "predictions.csv row {}: ID {:?} out of order; expected subject {:?} \
                 (predictions must follow ebes.csv subject order)",
                i + 1,
                id,
                subjects[idx].id
            )));
        }
        // Row-level N_OBS guard: catches two same-ID blocks written out of
        // order, where the ID check above passes but the block sizes differ.
        if let Some(j) = n_obs_i {
            if let Some(raw) = fields.get(j) {
                if let Ok(row_n_obs) = raw.parse::<usize>() {
                    if row_n_obs != subjects[idx].n_obs {
                        return Err(FitrxError::Corrupt(format!(
                            "predictions.csv row {}: N_OBS {} disagrees with ebes.csv \
                             n_obs {} for subject {:?}",
                            i + 1,
                            row_n_obs,
                            subjects[idx].n_obs,
                            subjects[idx].id
                        )));
                    }
                }
            }
        }
        subjects[idx].pred.push(parse_opt(&fields[pred_i]));
        subjects[idx].ipred.push(parse_opt(&fields[ipred_i]));
        subjects[idx].cwres.push(parse_opt(&fields[cwres_i]));
        subjects[idx].iwres.push(parse_opt(&fields[iwres_i]));
        if let Some(j) = npde_i {
            subjects[idx]
                .npde
                .push(fields.get(j).map(|s| parse_opt(s)).unwrap_or(f64::NAN));
        }
        if let Some(j) = npd_i {
            subjects[idx]
                .npd
                .push(fields.get(j).map(|s| parse_opt(s)).unwrap_or(f64::NAN));
        }
        let c = match cens_i {
            Some(j) => fields
                .get(j)
                .and_then(|v| v.parse::<i8>().ok())
                .unwrap_or(0),
            None => 0,
        };
        subjects[idx].cens.push(c);
        filled += 1;
    }

    // Cross-validate every subject's per-observation vector length against
    // the `n_obs` we read from ebes.csv. A mismatch means the bundle was
    // hand-edited or produced by a buggy writer; better to fail clearly
    // here than to let downstream code panic on a length mismatch.
    for s in &subjects {
        if s.ipred.len() != s.n_obs {
            return Err(FitrxError::Corrupt(format!(
                "subject {:?}: predictions.csv has {} rows but ebes.csv reports n_obs = {}",
                s.id,
                s.ipred.len(),
                s.n_obs
            )));
        }
    }

    Ok(subjects)
}

fn parse_ebe_kappas(
    ebes_kappa_csv: &str,
    subject_ids: &[String],
) -> Result<Vec<Vec<DVector<f64>>>, FitrxError> {
    let mut lines = ebes_kappa_csv.lines();
    let header = lines
        .next()
        .ok_or_else(|| FitrxError::Corrupt("ebes_kappa.csv: empty".into()))?;
    let cols: Vec<&str> = header.split(',').collect();
    if cols.len() < 3 || cols[0] != "ID" || cols[1] != "OCC" {
        return Err(FitrxError::Corrupt(
            "ebes_kappa.csv header must start with ID,OCC,...".into(),
        ));
    }
    let n_kappa = cols.len() - 2;

    let mut by_id: HashMap<String, usize> = HashMap::new();
    for (idx, id) in subject_ids.iter().enumerate() {
        by_id.insert(id.clone(), idx);
    }
    // We collect (occ, kappa) pairs per subject and place them by OCC at the
    // end, so file order doesn't determine occasion-index. The writer always
    // writes in order, but a hand-edited or third-party-written bundle could
    // shuffle rows and would otherwise silently associate kappas with the
    // wrong occasion.
    let mut staged: Vec<Vec<(u32, DVector<f64>)>> = vec![Vec::new(); subject_ids.len()];

    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() != cols.len() {
            return Err(FitrxError::Corrupt(format!(
                "ebes_kappa.csv row {} has {} fields, expected {}",
                i + 1,
                fields.len(),
                cols.len()
            )));
        }
        let idx = *by_id.get(&fields[0]).ok_or_else(|| {
            FitrxError::Corrupt(format!("ebes_kappa.csv: unknown ID {:?}", fields[0]))
        })?;
        let occ = fields[1].parse::<u32>().map_err(|_| {
            FitrxError::Corrupt(format!(
                "ebes_kappa.csv row {}: bad OCC value {:?} (expected positive integer)",
                i + 1,
                fields[1]
            ))
        })?;
        if occ == 0 {
            return Err(FitrxError::Corrupt(format!(
                "ebes_kappa.csv row {}: OCC must be 1-based, got 0",
                i + 1
            )));
        }
        let mut v = DVector::<f64>::zeros(n_kappa);
        for k in 0..n_kappa {
            v[k] = fields[2 + k].parse::<f64>().map_err(|_| {
                FitrxError::Corrupt(format!("ebes_kappa.csv: bad kappa in row {}", i + 1))
            })?;
        }
        staged[idx].push((occ, v));
    }

    // Resolve per-subject occasion slots. Each subject's `staged` vector is
    // sorted by OCC; we validate that OCCs are contiguous 1..n and that no
    // index is duplicated.
    let mut out: Vec<Vec<DVector<f64>>> = Vec::with_capacity(subject_ids.len());
    for (idx, mut rows) in staged.into_iter().enumerate() {
        if rows.is_empty() {
            out.push(Vec::new());
            continue;
        }
        rows.sort_by_key(|(occ, _)| *occ);
        let n = rows.len();
        for (i, (occ, _)) in rows.iter().enumerate() {
            let expected = (i + 1) as u32;
            if *occ != expected {
                return Err(FitrxError::Corrupt(format!(
                    "ebes_kappa.csv: subject {:?} has OCC {} but expected {} (rows must form a 1..{} sequence)",
                    subject_ids[idx], occ, expected, n
                )));
            }
        }
        out.push(rows.into_iter().map(|(_, v)| v).collect());
    }
    Ok(out)
}

// Parses conddist.csv (ID,ETA,COND_MEAN,COND_SD,COND_MODE) back into a
// `CondDist`. `samples`/`nsamp`/`burnin` aren't carried by this CSV — the
// bundle only round-trips the mean/SD/mode summary, not retained MCMC draws —
// so they come back empty/zero. `shrinkage` isn't stored either; it's a pure
// function of `cond_mean` and Ω, so it's recomputed here with the same
// formula as `saem_conddist::run_conditional_distribution`, giving a value
// that matches the original bit-for-bit rather than a redundant serialized
// copy.
//
// The CSV must carry exactly one row per (subject, eta): duplicate or missing
// rows are rejected rather than silently loaded as NaN, and each COND_MODE is
// validated against the EBE already loaded from `ebes.csv`, so a truncated or
// hand-edited bundle fails fast.
fn parse_conddist_csv(csv: &str, fit: &FitResult) -> Result<CondDist, FitrxError> {
    let n_subjects = fit.subjects.len();
    let n_eta = fit.eta_names.len();

    let mut by_id: HashMap<&str, usize> = HashMap::new();
    for (idx, s) in fit.subjects.iter().enumerate() {
        by_id.insert(s.id.as_str(), idx);
    }
    let mut by_eta: HashMap<&str, usize> = HashMap::new();
    for (idx, name) in fit.eta_names.iter().enumerate() {
        by_eta.insert(name.as_str(), idx);
    }

    let mut lines = csv.lines();
    let header = lines
        .next()
        .ok_or_else(|| FitrxError::Corrupt("conddist.csv: empty".into()))?;
    if header != "ID,ETA,COND_MEAN,COND_SD,COND_MODE" {
        return Err(FitrxError::Corrupt(format!(
            "conddist.csv: unexpected header {:?}",
            header
        )));
    }

    let mut cond_mean = vec![vec![f64::NAN; n_eta]; n_subjects];
    let mut cond_sd = vec![vec![f64::NAN; n_eta]; n_subjects];
    // Track which (subject, eta) cells the CSV actually filled, so a truncated
    // or duplicate-filled bundle is rejected up front rather than silently
    // loading NaNs (and NaN shrinkage) later.
    let mut seen = vec![vec![false; n_eta]; n_subjects];

    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() != 5 {
            return Err(FitrxError::Corrupt(format!(
                "conddist.csv row {} has {} fields, expected 5",
                i + 1,
                fields.len()
            )));
        }
        let si = *by_id.get(fields[0].as_str()).ok_or_else(|| {
            FitrxError::Corrupt(format!("conddist.csv: unknown ID {:?}", fields[0]))
        })?;
        let ei = *by_eta.get(fields[1].as_str()).ok_or_else(|| {
            FitrxError::Corrupt(format!("conddist.csv: unknown ETA {:?}", fields[1]))
        })?;
        if seen[si][ei] {
            return Err(FitrxError::Corrupt(format!(
                "conddist.csv: duplicate row for ID {:?} ETA {:?}",
                fields[0], fields[1]
            )));
        }
        seen[si][ei] = true;
        let parse_or_nan = |field: &str| -> Result<f64, FitrxError> {
            if field.is_empty() {
                Ok(f64::NAN)
            } else {
                field.parse::<f64>().map_err(|_| {
                    FitrxError::Corrupt(format!(
                        "conddist.csv row {}: bad float {:?}",
                        i + 1,
                        field
                    ))
                })
            }
        };
        cond_mean[si][ei] = parse_or_nan(&fields[2])?;
        cond_sd[si][ei] = parse_or_nan(&fields[3])?;
        // COND_MODE is the conditional mode = the EBE already carried on
        // `fit.subjects[si].eta` (loaded from ebes.csv). It isn't re-stored, but
        // we validate it agrees with that EBE so a mismatched or hand-edited
        // bundle fails fast rather than passing off inconsistent data. Both are
        // produced by the same `fmt_f64`, so a consistent bundle matches exactly.
        let expected_mode = fmt_f64(fit.subjects[si].eta.get(ei).copied().unwrap_or(f64::NAN));
        if fields[4] != expected_mode {
            return Err(FitrxError::Corrupt(format!(
                "conddist.csv: COND_MODE {:?} for ID {:?} ETA {:?} disagrees with EBE {:?}",
                fields[4], fields[0], fields[1], expected_mode
            )));
        }
    }

    // Every (subject, eta) pair must be present exactly once. The duplicate
    // check above covers "more than once"; this covers a truncated bundle.
    for (si, s) in fit.subjects.iter().enumerate() {
        for (ei, eta_name) in fit.eta_names.iter().enumerate() {
            if !seen[si][ei] {
                return Err(FitrxError::Corrupt(format!(
                    "conddist.csv: missing row for ID {:?} ETA {:?}",
                    s.id, eta_name
                )));
            }
        }
    }

    let shrinkage: Vec<f64> = (0..n_eta)
        .map(|j| {
            if n_subjects < 2 {
                return f64::NAN;
            }
            let omega_jj = fit.omega[(j, j)];
            if omega_jj < crate::estimation::saem::SAEM_OMEGA_DIAG_FLOOR {
                return f64::NAN;
            }
            let m: f64 = cond_mean.iter().map(|cm| cm[j]).sum::<f64>() / n_subjects as f64;
            let var: f64 = cond_mean.iter().map(|cm| (cm[j] - m).powi(2)).sum::<f64>()
                / (n_subjects - 1) as f64;
            1.0 - var.sqrt() / omega_jj.sqrt()
        })
        .collect();

    Ok(CondDist {
        cond_mean,
        cond_sd,
        samples: vec![Vec::new(); n_subjects],
        shrinkage,
        nsamp: 0,
        burnin: 0,
    })
}

// Validate every "parallel array" invariant in the wire layout so that
// downstream consumers that index by position (theta_names[i] matched with
// theta_estimates[i], etc.) can't panic on a malformed file. The Rust
// writer always produces consistent lengths; this check exists to keep
// loaders robust against hand-edited or third-party-written bundles.
fn validate_parallel_lengths(w: &FitWire) -> Result<(), FitrxError> {
    let bail = |msg: String| -> Result<(), FitrxError> { Err(FitrxError::Corrupt(msg)) };

    let n_theta = w.theta.estimates.len();
    if w.theta.names.len() != n_theta {
        return bail(format!(
            "theta.names ({}) and theta.estimates ({}) length mismatch",
            w.theta.names.len(),
            n_theta
        ));
    }
    if w.theta.fixed.len() != n_theta {
        return bail(format!(
            "theta.fixed ({}) does not match theta.estimates ({})",
            w.theta.fixed.len(),
            n_theta
        ));
    }
    if w.theta.transform.len() != n_theta {
        return bail(format!(
            "theta.transform ({}) does not match theta.estimates ({})",
            w.theta.transform.len(),
            n_theta
        ));
    }
    if let Some(se) = &w.theta.se {
        if se.len() != n_theta {
            return bail(format!(
                "theta.se ({}) does not match theta.estimates ({})",
                se.len(),
                n_theta
            ));
        }
    }

    let n_eta = w.omega.matrix.rows;
    if w.omega.matrix.cols != n_eta {
        return bail(format!(
            "omega.matrix is {}×{}; expected square",
            n_eta, w.omega.matrix.cols
        ));
    }
    if w.omega.names.len() != n_eta {
        return bail(format!(
            "omega.names ({}) does not match omega.matrix dim ({})",
            w.omega.names.len(),
            n_eta
        ));
    }
    if w.omega.fixed.len() != n_eta {
        return bail(format!(
            "omega.fixed ({}) does not match omega.matrix dim ({})",
            w.omega.fixed.len(),
            n_eta
        ));
    }
    if w.omega.log_transformed.len() != n_eta {
        return bail(format!(
            "omega.log_transformed ({}) does not match omega.matrix dim ({})",
            w.omega.log_transformed.len(),
            n_eta
        ));
    }
    if !w.omega.shrinkage.is_empty() && w.omega.shrinkage.len() != n_eta {
        return bail(format!(
            "omega.shrinkage ({}) does not match omega.matrix dim ({})",
            w.omega.shrinkage.len(),
            n_eta
        ));
    }
    // `init_as_sd` (issue #5): backward-compat empty vec is fine, but any
    // other non-matching length is a corrupt bundle — don't silently pad it.
    if !w.omega.init_as_sd.is_empty() && w.omega.init_as_sd.len() != n_eta {
        return bail(format!(
            "omega.init_as_sd ({}) does not match omega.matrix dim ({})",
            w.omega.init_as_sd.len(),
            n_eta
        ));
    }

    let n_sigma = w.sigma.estimates.len();
    if w.sigma.names.len() != n_sigma {
        return bail(format!(
            "sigma.names ({}) and sigma.estimates ({}) length mismatch",
            w.sigma.names.len(),
            n_sigma
        ));
    }
    if w.sigma.fixed.len() != n_sigma {
        return bail(format!(
            "sigma.fixed ({}) does not match sigma.estimates ({})",
            w.sigma.fixed.len(),
            n_sigma
        ));
    }
    if w.sigma.types.len() != n_sigma {
        return bail(format!(
            "sigma.types ({}) does not match sigma.estimates ({})",
            w.sigma.types.len(),
            n_sigma
        ));
    }
    if !w.sigma.init_as_sd.is_empty() && w.sigma.init_as_sd.len() != n_sigma {
        return bail(format!(
            "sigma.init_as_sd ({}) does not match sigma.estimates ({})",
            w.sigma.init_as_sd.len(),
            n_sigma
        ));
    }
    let mut seen_corr_pairs = std::collections::HashSet::new();
    for corr in &w.sigma.residual_correlations {
        if corr.sigma_i >= n_sigma || corr.sigma_j >= n_sigma {
            return bail(format!(
                "sigma residual correlation index ({}, {}) is out of bounds for {} sigmas",
                corr.sigma_i, corr.sigma_j, n_sigma
            ));
        }
        // `|rho| == 1` is rejected alongside `> 1`: a perfectly correlated pair
        // makes the subject-level `R` exactly singular, which would otherwise
        // surface as a NaN/Inf OFV on the next evaluation rather than as a
        // load-time error.
        if corr.sigma_i == corr.sigma_j || !corr.rho.is_finite() || corr.rho.abs() >= 1.0 {
            return bail(format!(
                "invalid sigma residual correlation ({}, {}, rho={})",
                corr.sigma_i, corr.sigma_j, corr.rho
            ));
        }
        // A repeated (i, j) pair would double-count the cross term in
        // `cross_observation_covariance` and emit duplicate YAML keys.
        let pair = (
            corr.sigma_i.min(corr.sigma_j),
            corr.sigma_i.max(corr.sigma_j),
        );
        if !seen_corr_pairs.insert(pair) {
            return bail(format!(
                "duplicate sigma residual correlation for pair ({}, {})",
                pair.0, pair.1
            ));
        }
    }
    // #847 companions to `residual_correlations`: both are optional (a pre-#847
    // bundle carries neither), but a present one must be parallel — a truncated
    // FIX vector would silently report an estimated correlation as free, and a
    // truncated SE vector would mis-align the SE with the pair it belongs to.
    let n_corr = w.sigma.residual_correlations.len();
    if !w.sigma.residual_correlation_fixed.is_empty()
        && w.sigma.residual_correlation_fixed.len() != n_corr
    {
        return bail(format!(
            "sigma.residual_correlation_fixed ({}) does not match sigma.residual_correlations ({})",
            w.sigma.residual_correlation_fixed.len(),
            n_corr
        ));
    }
    if let Some(se) = &w.sigma.se_residual_correlations {
        if se.len() != n_corr {
            return bail(format!(
                "sigma.se_residual_correlations ({}) does not match sigma.residual_correlations ({})",
                se.len(),
                n_corr
            ));
        }
    }
    // IOV init_as_sd: same backward-compat rule as omega/sigma. Only validate
    // when an `iov` section is present (otherwise there's no n_kappa to match
    // against).
    if let Some(iov) = &w.iov {
        let n_kappa = iov.kappa_names.len();
        if !iov.kappa_init_as_sd.is_empty() && iov.kappa_init_as_sd.len() != n_kappa {
            return bail(format!(
                "iov.kappa_init_as_sd ({}) does not match iov.kappa_names ({})",
                iov.kappa_init_as_sd.len(),
                n_kappa
            ));
        }
    }
    Ok(())
}

fn wire_to_fit_result(
    mut w: FitWire,
    subjects: Vec<SubjectResult>,
    ebe_kappas: Vec<Vec<DVector<f64>>>,
) -> Result<FitResult, FitrxError> {
    validate_parallel_lengths(&w)?;
    let method = method_from_str(&w.method)?;
    let method_chain: Vec<EstimationMethod> = w
        .method_chain
        .iter()
        .map(|s| method_from_str(s))
        .collect::<Result<_, _>>()?;

    let theta_transform: Vec<ThetaTransform> = w
        .theta
        .transform
        .iter()
        .map(|s| theta_transform_from_str(s))
        .collect::<Result<_, _>>()?;
    let sigma_types: Vec<SigmaType> = w
        .sigma
        .types
        .iter()
        .map(|s| sigma_type_from_str(s))
        .collect::<Result<_, _>>()?;

    let eta_param_info: Vec<EtaParamInfo> = w
        .eta_param_info
        .into_iter()
        .map(|i| {
            Ok::<EtaParamInfo, FitrxError>(EtaParamInfo {
                eta_name: i.eta_name,
                param_type: eta_param_type_from_str(&i.param_type)?,
                linked_theta: i.linked_theta,
                individual_param_name: i.individual_param_name,
            })
        })
        .collect::<Result<_, _>>()?;

    let omega = w.omega.matrix.into_dmatrix()?;
    let omega_param_corr = w.omega.param_corr.map(|m| m.into_dmatrix()).transpose()?;
    let omega_init = {
        match w.omega_init {
            Some(m) => m.into_dmatrix()?,
            // Pre-PR bundles lack omega_init.  Fall back to the converged
            // omega rather than a zero matrix — zeros are not positive-definite
            // and would break any Cholesky-based consumer.
            None => omega.clone(),
        }
    };
    let covariance_matrix = w.covariance_matrix.map(|m| m.into_dmatrix()).transpose()?;

    let (
        omega_iov,
        kappa_names,
        kappa_fixed,
        kappa_init_as_sd,
        kappa_weights,
        kappa_weight_typical,
        se_kappa,
        shrinkage_kappa,
        shrinkage_kappa_by_occ,
        omega_iov_param_corr,
    ) = match w.iov {
        Some(iov) => {
            // `validate_parallel_lengths` has already ensured that
            // `kappa_init_as_sd` is either empty (pre-issue-#5 bundle) or
            // exactly `n_kappa` long. Promote the empty case to all-false so
            // downstream code can index it uniformly with `kappa_names`.
            let n_k = iov.kappa_names.len();
            let init_as_sd = if iov.kappa_init_as_sd.is_empty() {
                vec![false; n_k]
            } else {
                iov.kappa_init_as_sd
            };
            (
                Some(iov.omega_iov.into_dmatrix()?),
                iov.kappa_names,
                iov.kappa_fixed,
                init_as_sd,
                iov.kappa_weights,
                iov.kappa_weight_typical,
                iov.se_kappa,
                iov.shrinkage_kappa,
                iov.shrinkage_kappa_by_occ,
                iov.omega_iov_param_corr
                    .map(|m| m.into_dmatrix())
                    .transpose()?,
            )
        }
        None => (
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            Vec::new(),
            None,
        ),
    };

    let (sir_ci_theta, sir_ci_omega, sir_ci_sigma, sir_ess, sir_resamples_packed) = match w.sir {
        Some(s) => (
            s.ci_theta,
            s.ci_omega,
            s.ci_sigma,
            s.ess,
            s.resamples_packed,
        ),
        None => (None, None, None, None, None),
    };

    // `validate_parallel_lengths` has already ensured that omega/sigma
    // `init_as_sd` are either empty (pre-issue-#5 bundle) or exactly the
    // expected length. Promote the empty case to all-false here. Computed
    // up-front because the FitResult literal below moves `w.omega.names`
    // and `w.sigma.names` into other fields.
    let n_eta_w = w.omega.names.len();
    let n_sigma_w = w.sigma.names.len();
    let omega_init_as_sd_resolved = if w.omega.init_as_sd.is_empty() {
        vec![false; n_eta_w]
    } else {
        std::mem::take(&mut w.omega.init_as_sd)
    };
    let sigma_init_as_sd_resolved = if w.sigma.init_as_sd.is_empty() {
        vec![false; n_sigma_w]
    } else {
        std::mem::take(&mut w.sigma.init_as_sd)
    };

    // A bundle written before #847 has correlations but no FIX vector: back then
    // every `block_sigma` off-diagonal was held fixed, so that is what it meant.
    let residual_correlation_fixed_resolved = if w.sigma.residual_correlations.is_empty() {
        Vec::new()
    } else if w.sigma.residual_correlation_fixed.is_empty() {
        vec![true; w.sigma.residual_correlations.len()]
    } else {
        std::mem::take(&mut w.sigma.residual_correlation_fixed)
    };

    Ok(FitResult {
        restored_from_checkpoint: true,
        covariate_relations: Vec::new(),
        method,
        method_chain,
        method_wall_times_secs: w.method_wall_times_secs,
        covariance_wall_time_secs: w.covariance_wall_time_secs,
        converged: w.converged,
        ofv: w.ofv,
        aic: w.aic,
        bic: w.bic,
        theta: w.theta.estimates,
        theta_names: w.theta.names,
        eta_names: w.omega.names,
        omega,
        sigma: w.sigma.estimates,
        sigma_names: w.sigma.names,
        residual_correlations: w.sigma.residual_correlations,
        residual_correlation_fixed: residual_correlation_fixed_resolved,
        error_model: error_model_from_str(&w.error_model)?,
        covariance_matrix,
        se_theta: w.theta.se,
        se_omega: w.omega.se,
        se_sigma: w.sigma.se,
        se_residual_correlations: w.sigma.se_residual_correlations,
        theta_fixed: w.theta.fixed,
        omega_fixed: w.omega.fixed,
        sigma_fixed: w.sigma.fixed,
        omega_init_as_sd: omega_init_as_sd_resolved,
        sigma_init_as_sd: sigma_init_as_sd_resolved,
        subjects,
        n_obs: w.n_obs,
        n_subjects: w.n_subjects,
        n_parameters: w.n_parameters,
        n_iterations: w.n_iterations,
        interaction: w.interaction,
        warnings_structured: w
            .warnings
            .iter()
            .map(|s| crate::types::classify_warning(s))
            .collect(),
        warnings: w.warnings,
        sir_ci_theta,
        sir_ci_omega,
        sir_ci_sigma,
        sir_ess,
        sir_resamples_packed,
        // .fitrx v1: importance_sampling is not serialised — re-run via
        // `methods = [..., imp]` if the consumer needs the IS LL.
        importance_sampling: None,
        impmap_trace: None,
        bayes: None,
        // .fitrx v1 does not serialise the variational posteriors; re-run
        // `method = vi` if the consumer needs them.
        vi: None,
        omega_iov,
        kappa_names,
        kappa_fixed,
        kappa_init_as_sd,
        kappa_weights,
        kappa_weight_typical,
        se_kappa,
        shrinkage_kappa,
        shrinkage_kappa_by_occ,
        ebe_kappas,
        saem_mu_ref_m_step_evals_saved: w.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: w.saem_n_subjects_hmc,
        gradient_method_inner: w.gradient_method_inner,
        gradient_method_outer: w.gradient_method_outer,
        uses_ode_solver: w.uses_ode_solver,
        uses_sde: w.uses_sde,
        n_threads_used: w.n_threads_used,
        nlopt_missing_algorithms: w.nlopt_missing_algorithms,
        covariance_n_evals_estimated: w.covariance_n_evals_estimated,
        trace_path: w.trace_path,
        ebe_convergence_warnings: w.ebe_convergence_warnings,
        max_unconverged_subjects: w.max_unconverged_subjects,
        total_ebe_fallbacks: w.total_ebe_fallbacks,
        covariance_status: covariance_status_from_str(&w.covariance_status)?,
        shrinkage_eta: w.omega.shrinkage,
        // Populated by `load_fit` from conddist.csv when present (#675);
        // fit.json/FitWire carries no cond_dist field of its own.
        cond_dist: None,
        shrinkage_eps: w.shrinkage_eps,
        iwres_lag1_r: w.iwres_lag1_r,
        dw_statistic: w.dw_statistic,
        wall_time_secs: w.wall_time_secs,
        model_name: w.model_name,
        ferx_version: w.ferx_version,
        environment: w.environment.unwrap_or_default(),
        eta_param_info,
        theta_transform,
        sigma_types,
        cov_eigenvalues: w.cov_eigenvalues,
        cov_condition_number: w.cov_condition_number,
        eta_log_transformed: w.omega.log_transformed,
        omega_param_corr,
        omega_iov_param_corr,
        model_path: w.model_path,
        data_path: w.data_path,
        model_hash: w.model_hash,
        data_hash: w.data_hash,
        model_text: w.model_text,
        theta_init: w.theta_init,
        omega_init,
        sigma_init: w.sigma_init,
        obs_time_range: w.obs_time_range,
        final_gradient: w.final_gradient,
        optimizer: w.optimizer,
        n_starts: w.n_starts,
        multi_start_seed: w.multi_start_seed,
        saem_seed: w.saem_seed,
        sir_seed: w.sir_seed,
        imp_seed: w.imp_seed,
        npde_seed: w.npde_seed,
        bloq_method: w.bloq_method,
        outer_maxiter: w.outer_maxiter,
        outer_gtol: w.outer_gtol,
        inits_from_nca: w.inits_from_nca,
        covariate_names: w.covariate_names,
        input_columns: w.input_columns,
        #[cfg(feature = "nn")]
        neural_networks: w.neural_networks.unwrap_or_default(),
        // The covariate table is not persisted in the .fitrx bundle (yet); a
        // round-tripped result therefore has no covariate table.
        covariate_table: None,
        exclusions: None,
        // Transient (`#[serde(skip)]`) and not persisted: a reloaded fit has no
        // optimizer packed vector, so `run_covariance` re-packs from omega (#816).
        packed_estimate: None,
    })
}

/// Minimal CSV row parser handling quoted fields and doubled-quote escapes.
/// Sufficient for the columns we write (no embedded newlines).
fn parse_csv_row(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_quotes) {
            ('"', true) => {
                if matches!(chars.peek(), Some('"')) {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ('"', false) => in_quotes = true,
            (',', false) => {
                out.push(std::mem::take(&mut cur));
            }
            (ch, _) => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn pmix_column_order_sorts_by_class_number() {
        // #984 review: restore PMIX by class number, not header position — so an
        // alphabetically-reordered header (PMIX_10 before PMIX_2, K ≥ 10) maps each
        // column to the right class instead of silently swapping probabilities.
        // In-order header → identity mapping onto the PMIX columns.
        let h = [
            "ID", "eta_1", "ofv", "n_obs", "MIXEST", "PMIX_1", "PMIX_2", "PMIX_3",
        ];
        assert_eq!(pmix_column_order(&h), vec![5, 6, 7]);

        // Alphabetically reordered (PMIX_1, PMIX_10, PMIX_11, PMIX_2, ...) must be
        // re-sorted to class order 1,2,...,10,11 → the column *positions* follow.
        let h2 = [
            "ID", "PMIX_1", "PMIX_10", "PMIX_11", "PMIX_2", "PMIX_3", "PMIX_4", "PMIX_5", "PMIX_6",
            "PMIX_7", "PMIX_8", "PMIX_9",
        ];
        // class 1..9 sit at positions 1,4,5,6,7,8,9,10,11; class 10,11 at 2,3.
        assert_eq!(
            pmix_column_order(&h2),
            vec![1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3]
        );

        // No PMIX columns → empty.
        assert!(pmix_column_order(&["ID", "eta_1", "ofv", "n_obs"]).is_empty());
    }

    fn dummy_subject(id: &str, n_eta: usize, n_obs: usize) -> SubjectResult {
        SubjectResult {
            id: id.into(),
            eta: DVector::from_vec((0..n_eta).map(|k| 0.1 * (k as f64 + 1.0)).collect()),
            ipred: (0..n_obs).map(|j| 1.0 + j as f64).collect(),
            pred: (0..n_obs).map(|j| 1.5 + j as f64).collect(),
            iwres: (0..n_obs).map(|j| 0.01 * j as f64).collect(),
            cwres: (0..n_obs).map(|j| -0.02 * j as f64).collect(),
            npde: vec![],
            npd: vec![],
            ofv_contribution: 12.34,
            cens: vec![0; n_obs],
            n_obs,
            pmix: None,
            mixest: None,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
            #[cfg(feature = "survival")]
            discrete_rows: Vec::new(),
        }
    }

    fn dummy_population(ids: &[&str], n_obs_each: usize) -> Population {
        let mut subjects = Vec::new();
        for id in ids {
            subjects.push(Subject {
                id: (*id).to_string(),
                doses: vec![],
                obs_times: (0..n_obs_each).map(|j| j as f64).collect(),
                obs_raw_times: Vec::new(),
                observations: (0..n_obs_each).map(|j| 5.0 + j as f64).collect(),
                obs_cmts: vec![1; n_obs_each],
                covariates: HashMap::new(),
                dose_covariates: vec![],
                obs_covariates: vec![],
                pk_only_times: vec![],
                pk_only_covariates: vec![],
                reset_times: vec![],
                reset_covariates: Vec::new(),
                cens: vec![0; n_obs_each],
                occasions: vec![],
                obs_l2: Vec::new(),
                dose_occasions: vec![],
                reset_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: vec![],
            });
        }
        Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn minimal_fit_result() -> FitResult {
        let n_eta = 2;
        FitResult {
            residual_correlation_fixed: Vec::new(),
            se_residual_correlations: None,
            covariate_relations: Vec::new(),
            restored_from_checkpoint: false,
            method: EstimationMethod::FoceI,
            method_chain: vec![EstimationMethod::FoceI],
            method_wall_times_secs: vec![1.234],
            covariance_wall_time_secs: 0.0,
            converged: true,
            ofv: 100.0,
            aic: 110.0,
            bic: 115.0,
            theta: vec![1.0, 2.0, 0.5],
            theta_names: vec!["CL".into(), "V".into(), "KA".into()],
            eta_names: vec!["eta_CL".into(), "eta_V".into()],
            omega: DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]),
            sigma: vec![0.05],
            sigma_names: vec!["prop".into()],
            residual_correlations: Vec::new(),
            error_model: ErrorModel::Proportional,
            covariance_matrix: Some(DMatrix::<f64>::identity(3, 3)),
            se_theta: Some(vec![0.01, 0.02, 0.005]),
            se_omega: Some(vec![0.01, 0.02]),
            se_sigma: Some(vec![0.001]),
            theta_fixed: vec![false, false, false],
            omega_fixed: vec![false, false],
            sigma_fixed: vec![false],
            omega_init_as_sd: vec![false, false],
            sigma_init_as_sd: vec![false],
            subjects: vec![dummy_subject("S1", n_eta, 3), dummy_subject("S2", n_eta, 2)],
            n_obs: 5,
            n_subjects: 2,
            n_parameters: 6,
            n_iterations: 10,
            interaction: true,
            warnings: vec!["watch out".into()],
            warnings_structured: vec![crate::types::classify_warning("watch out")],
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            impmap_trace: None,
            bayes: None,
            vi: None,
            omega_iov: None,
            kappa_names: vec![],
            kappa_fixed: vec![],
            kappa_init_as_sd: vec![],
            kappa_weights: Vec::new(),
            kappa_weight_typical: Vec::new(),
            se_kappa: None,
            shrinkage_kappa: vec![],
            shrinkage_kappa_by_occ: vec![],
            ebe_kappas: vec![],
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: "analytic (Dual2)".into(),
            gradient_method_outer: "finite differences".into(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 4,
            nlopt_missing_algorithms: vec![],
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: CovarianceStatus::Computed,
            shrinkage_eta: vec![0.1, 0.15],
            cond_dist: None,
            shrinkage_eps: 0.05,
            iwres_lag1_r: 0.12,
            dw_statistic: 1.75,
            wall_time_secs: 1.234,
            model_name: "test_model".into(),
            ferx_version: "0.1.0".into(),
            environment: crate::environment::detect(),
            eta_param_info: vec![
                EtaParamInfo {
                    eta_name: "eta_CL".into(),
                    param_type: EtaParamType::LogNormal,
                    linked_theta: Some("CL".into()),
                    individual_param_name: "CL".into(),
                },
                EtaParamInfo {
                    eta_name: "eta_V".into(),
                    param_type: EtaParamType::LogNormal,
                    linked_theta: Some("V".into()),
                    individual_param_name: "V".into(),
                },
            ],
            theta_transform: vec![
                ThetaTransform::Log,
                ThetaTransform::Log,
                ThetaTransform::Log,
            ],
            sigma_types: vec![SigmaType::Proportional],
            cov_eigenvalues: Some(vec![1.0, 0.5, 0.2]),
            cov_condition_number: Some(5.0),
            eta_log_transformed: vec![true, true],
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            model_text: None,
            theta_init: vec![1.0, 2.0, 0.5],
            omega_init: DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]),
            sigma_init: vec![0.05],
            obs_time_range: Some((0.25, 24.0)),
            final_gradient: None,
            optimizer: "slsqp".to_string(),
            n_starts: 1,
            multi_start_seed: None,
            saem_seed: None,
            sir_seed: None,
            imp_seed: None,
            npde_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 300,
            outer_gtol: 1e-4,
            inits_from_nca: None,
            covariate_names: vec!["WT".into(), "AGE".into()],
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
            exclusions: None,
            packed_estimate: None,
        }
    }

    #[test]
    fn json_result_has_versioned_schema_and_round_trips() {
        // #777: the JSON output must be the *complete* FitResult under a
        // versioned schema. Idempotent round-trip (serialize → deserialize →
        // re-serialize equals the first serialization) proves every serialized
        // field also deserializes back, i.e. no field is write-only / lossy.
        let fit = minimal_fit_result();
        let v1 = fit.to_json_value();

        // Top-level schema_version is present and pinned.
        assert_eq!(v1["schema_version"], FitResult::JSON_SCHEMA_VERSION);

        // A few representative fields survive with the agent-friendly shapes.
        assert_eq!(v1["ofv"], 100.0);
        assert_eq!(v1["omega"]["rows"], 2);
        assert_eq!(v1["omega"]["cols"], 2);
        assert_eq!(v1["omega"]["data"], serde_json::json!([0.1, 0.0, 0.0, 0.2]));
        assert_eq!(v1["subjects"][0]["eta"].as_array().unwrap().len(), 2);
        assert!(
            v1.get("residual_correlations").is_none(),
            "diagonal-sigma JSON must remain byte-compatible"
        );

        // Round-trip: unknown `schema_version` is ignored on the way back in.
        let back: FitResult = serde_json::from_value(v1.clone()).unwrap();
        let v2 = back.to_json_value();
        assert_eq!(v1, v2, "JSON round-trip is not idempotent (lossy field?)");
    }

    #[test]
    fn json_result_includes_residual_correlations() {
        let mut fit = minimal_fit_result();
        fit.sigma.push(1.0);
        fit.sigma_names.push("add".into());
        fit.sigma_fixed.push(false);
        fit.sigma_types.push(SigmaType::Additive);
        fit.sigma_init_as_sd.push(false);
        fit.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 1,
            sigma_j: 0,
            rho: 0.5,
        }];

        let value = fit.to_json_value();
        assert_eq!(
            value["residual_correlations"],
            serde_json::json!([{"sigma_i": 1, "sigma_j": 0, "rho": 0.5}])
        );
        let restored: FitResult = serde_json::from_value(value).unwrap();
        assert_eq!(restored.residual_correlations, fit.residual_correlations);
    }

    #[test]
    fn fitrx_rejects_invalid_residual_correlations() {
        let mut fit = minimal_fit_result();
        fit.sigma.push(1.0);
        fit.sigma_names.push("add".into());
        fit.sigma_fixed.push(false);
        fit.sigma_types.push(SigmaType::Additive);
        fit.sigma_init_as_sd.push(false);
        let mut wire = build_fit_wire(&fit);

        for invalid in [
            crate::types::ResidualCorrelation {
                sigma_i: 0,
                sigma_j: 2,
                rho: 0.5,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 0,
                sigma_j: 0,
                rho: 0.5,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 0,
                rho: 1.5,
            },
            // |rho| == 1 is singular, not merely extreme: reject it too.
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 0,
                rho: 1.0,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 0,
                rho: -1.0,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 0,
                rho: f64::NAN,
            },
        ] {
            wire.sigma.residual_correlations = vec![invalid];
            assert!(
                validate_parallel_lengths(&wire).is_err(),
                "accepted invalid correlation: {invalid:?}"
            );
        }

        // The same (i, j) pair twice would double-count the cross term.
        wire.sigma.residual_correlations = vec![
            crate::types::ResidualCorrelation {
                sigma_i: 1,
                sigma_j: 0,
                rho: 0.5,
            },
            crate::types::ResidualCorrelation {
                sigma_i: 0,
                sigma_j: 1,
                rho: 0.25,
            },
        ];
        let err = validate_parallel_lengths(&wire).unwrap_err();
        assert!(
            format!("{err}").contains("duplicate sigma residual correlation"),
            "got: {err}"
        );

        // A well-formed single pair still loads.
        wire.sigma.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 1,
            sigma_j: 0,
            rho: 0.5,
        }];
        assert!(validate_parallel_lengths(&wire).is_ok());
    }

    #[test]
    fn json_result_maps_non_finite_floats_to_null() {
        // #777 review: guarantee that non-finite floats serialize to JSON
        // `null` and never panic `to_json_value()` / `write_result_json()`.
        // serde_json's `Value`/string serializers map non-finite f64 to null
        // (no `arbitrary_precision` feature here), so this holds without a
        // sanitizing pass — this test locks that contract in.
        let mut fit = minimal_fit_result();
        fit.ofv = f64::INFINITY;
        fit.shrinkage_eps = f64::NAN;
        fit.iwres_lag1_r = f64::NEG_INFINITY;
        // A missing covariate value is stored as NaN in the covariate table.
        fit.covariate_table = Some(crate::types::CovariateTable {
            names: vec!["WT".into()],
            kinds: vec![crate::types::CovariateKind::Continuous],
            rows: vec![crate::types::CovariateRow {
                id: "S1".into(),
                time: 0.0,
                evid: 0,
                values: vec![f64::NAN],
            }],
        });

        let v = fit.to_json_value(); // must not panic
        assert!(v["ofv"].is_null());
        assert!(v["shrinkage_eps"].is_null());
        assert!(v["iwres_lag1_r"].is_null());
        assert!(v["covariate_table"]["rows"][0]["values"][0].is_null());

        // Full write path also succeeds (pretty-printer never panics on null).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan-fit.json");
        crate::io::output::write_result_json(&fit, path.to_str().unwrap()).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(parsed["ofv"].is_null());
    }

    #[test]
    fn write_result_json_writes_parseable_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run-fit.json");
        let fit = minimal_fit_result();
        crate::io::output::write_result_json(&fit, path.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["schema_version"], FitResult::JSON_SCHEMA_VERSION);
        assert_eq!(parsed["converged"], true);
    }

    #[test]
    fn roundtrip_minimal_fit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run1.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        let l = &loaded.fit;
        assert_eq!(l.method, r.method);
        assert_eq!(l.method_chain, r.method_chain);
        assert_eq!(l.converged, r.converged);
        assert!((l.ofv - r.ofv).abs() < 1e-9);
        assert_eq!(l.theta, r.theta);
        assert_eq!(l.theta_names, r.theta_names);
        assert_eq!(l.eta_names, r.eta_names);
        assert_eq!(l.omega, r.omega);
        assert_eq!(l.sigma, r.sigma);
        assert_eq!(l.error_model, r.error_model);
        assert_eq!(l.covariance_matrix, r.covariance_matrix);
        assert_eq!(l.se_theta, r.se_theta);
        assert_eq!(l.theta_fixed, r.theta_fixed);
        assert_eq!(l.warnings, r.warnings);
        assert_eq!(l.covariance_status, r.covariance_status);
        assert_eq!(l.environment, r.environment);
        assert_eq!(l.subjects.len(), r.subjects.len());
        for (a, b) in l.subjects.iter().zip(r.subjects.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.eta, b.eta);
            assert_eq!(a.n_obs, b.n_obs);
            assert!((a.ofv_contribution - b.ofv_contribution).abs() < 1e-9);
            assert_eq!(a.ipred.len(), b.ipred.len());
            for (x, y) in a.ipred.iter().zip(b.ipred.iter()) {
                assert!((x - y).abs() < 1e-6);
            }
        }
        assert_eq!(loaded.model_source, "model source\n");
        assert!(loaded.population.is_none());
        assert_eq!(loaded.manifest.format_version, FORMAT_VERSION);
    }

    #[test]
    fn roundtrip_preserves_residual_correlations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("block-sigma.fitrx");
        let mut r = minimal_fit_result();
        r.sigma = vec![0.2, 1.0];
        r.sigma_names = vec!["PROP_ERR".into(), "ADD_ERR".into()];
        r.sigma_fixed = vec![true, true];
        r.sigma_init_as_sd = vec![false, false];
        r.sigma_types = vec![SigmaType::Proportional, SigmaType::Additive];
        r.sigma_init = r.sigma.clone();
        r.se_sigma = None;
        r.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];

        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.residual_correlations, r.residual_correlations);
    }

    #[test]
    fn roundtrip_mixture_pmix_mixest() {
        // #983: a mixture fit's per-subject MIXEST + PMIX_1..K survive the .fitrx
        // checkpoint via ebes.csv's optional trailing columns. Before, load_fit
        // hard-coded them to None, so a restored mixture fit emitted no
        // MIXEST/PMIX_* sdtab columns.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mix.fitrx");
        let mut r = minimal_fit_result();
        for (i, s) in r.subjects.iter_mut().enumerate() {
            let p0 = 0.2 + 0.3 * i as f64; // 0.2, 0.5 → distinct per subject
            s.pmix = Some(vec![p0, 1.0 - p0]);
            s.mixest = Some(if p0 >= 0.5 { 1 } else { 2 });
        }
        let expected: Vec<(Option<usize>, Vec<f64>)> = r
            .subjects
            .iter()
            .map(|s| (s.mixest, s.pmix.clone().unwrap()))
            .collect();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "m\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.subjects.len(), expected.len());
        for (s, (mixest, pmix)) in loaded.fit.subjects.iter().zip(&expected) {
            assert_eq!(s.mixest, *mixest, "MIXEST for {}", s.id);
            let got = s.pmix.as_ref().expect("PMIX restored");
            assert_eq!(got.len(), pmix.len());
            for (a, b) in got.iter().zip(pmix) {
                assert!((a - b).abs() < 1e-9, "PMIX {a} vs {b}");
            }
        }
    }

    #[test]
    fn roundtrip_non_mixture_has_no_pmix() {
        // A non-mixture bundle carries no MIXEST/PMIX columns and restores to
        // None — the appended columns are strictly opt-in (#983).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "m\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        for s in &loaded.fit.subjects {
            assert!(s.pmix.is_none(), "no PMIX for a non-mixture fit");
            assert!(s.mixest.is_none(), "no MIXEST for a non-mixture fit");
        }
    }

    /// The `restored_from_checkpoint` flag is set by `load_fit` and only by `load_fit`.
    /// This is the *wiring* test: a live fit carries `false`, a saved-then-loaded fit
    /// carries `true`. Without it, the flag could silently fail to be set (wrong literal,
    /// bypassed path) and `write_sdtab_csv`'s restore guard would never fire on a real
    /// restored fit — the guard's own test sets the flag by hand and cannot catch that.
    #[test]
    fn restored_flag_is_set_only_by_load_fit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.fitrx");
        let r = minimal_fit_result();
        assert!(
            !r.restored_from_checkpoint,
            "a freshly built fit must not be marked restored"
        );
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        assert!(
            loaded.fit.restored_from_checkpoint,
            "load_fit must mark the reconstructed result as restored"
        );
    }

    #[test]
    fn roundtrip_bundled_data_honours_column_mapping() {
        // #730 regression: data.csv is bundled with its original headers, so
        // load_fit must read it through the model's `[data]` column mapping.
        // Before the fix the loader used the unmapped reader and failed with
        // `Missing TIME column` on a TAFD/CONC dataset.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mapped.fitrx");

        // A dataset whose time/response columns are named TAFD/CONC.
        let data_csv = dir.path().join("input.csv");
        std::fs::write(
            &data_csv,
            "ID,TAFD,CONC,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n",
        )
        .unwrap();

        // Valid model that maps TAFD→TIME and CONC→DV in its `[data]` block.
        let model_source = "\
[parameters]
  theta TVCL(1.0)
  theta TVV(1.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[data]
  path = input.csv
  TIME = TAFD
  DV   = CONC
";

        // `p` must line up with `minimal_fit_result`'s subjects for the bundle's
        // predictions.csv join; the mapped read under test is of the separate
        // `include_data` file, not `p`.
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(
            &r,
            &p,
            model_source,
            &path,
            SaveFitOptions {
                include_data: Some(data_csv.clone()),
            },
        )
        .unwrap();

        let loaded = load_fit(&path).expect("load_fit reads the mapped TAFD/CONC data.csv");
        let pop = loaded.population.expect("data.csv was bundled");
        // The mapping fed CONC into DV: real observations were parsed.
        let n_obs: usize = pop.subjects.iter().map(|s| s.observations.len()).sum();
        assert_eq!(n_obs, 2, "mapped observations should be read");
        assert!(!pop.covariate_names.contains(&"TAFD".to_string()));
        assert!(!pop.covariate_names.contains(&"CONC".to_string()));
    }

    #[test]
    fn roundtrip_preserves_npde_npd() {
        // When the fit ran with npde_nsim > 0, the predictions.csv carries NPDE/NPD
        // and the loader must restore them (regression for the silent-drop bug).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("npde.fitrx");
        let mut r = minimal_fit_result();
        r.subjects[0].npde = vec![0.1, -0.2, 0.3];
        r.subjects[0].npd = vec![0.15, -0.25, 0.35];
        r.subjects[1].npde = vec![-1.0, 1.0];
        r.subjects[1].npd = vec![-0.9, 0.9];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        for (a, b) in loaded.fit.subjects.iter().zip(r.subjects.iter()) {
            assert_eq!(a.npde.len(), b.npde.len(), "subject {} npde len", a.id);
            for (x, y) in a.npde.iter().zip(b.npde.iter()) {
                assert!((x - y).abs() < 1e-6, "npde {x} vs {y}");
            }
            for (x, y) in a.npd.iter().zip(b.npd.iter()) {
                assert!((x - y).abs() < 1e-6, "npd {x} vs {y}");
            }
        }
    }

    #[test]
    fn roundtrip_omits_npde_when_absent() {
        // No NPDE/NPD computed → the columns are absent and load leaves them empty.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonpde.fitrx");
        let r = minimal_fit_result(); // dummy subjects have empty npde/npd
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        for s in &loaded.fit.subjects {
            assert!(s.npde.is_empty(), "npde must stay empty when not written");
            assert!(s.npd.is_empty(), "npd must stay empty when not written");
        }
    }

    #[test]
    fn roundtrip_preserves_npde_seed() {
        // The effective NPDE seed survives the save/load round-trip; `None`
        // (NPDE did not run) round-trips as `None`.
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);

        let mut r = minimal_fit_result();
        r.npde_seed = Some(20240601);
        let with = dir.path().join("seed.fitrx");
        save_fit(&r, &p, "src\n", &with, SaveFitOptions::default()).unwrap();
        assert_eq!(load_fit(&with).unwrap().fit.npde_seed, Some(20240601));

        let none = dir.path().join("noseed.fitrx");
        let r0 = minimal_fit_result(); // npde_seed defaults to None
        save_fit(&r0, &p, "src\n", &none, SaveFitOptions::default()).unwrap();
        assert_eq!(load_fit(&none).unwrap().fit.npde_seed, None);
    }

    #[test]
    fn fit_wire_missing_environment_defaults_to_none() {
        // Simulates a `.fitrx` bundle saved before #704 added `environment` to
        // `FitWire`: the key is simply absent from `fit.json`.
        let r = minimal_fit_result();
        let wire = build_fit_wire(&r);
        let mut value = serde_json::to_value(&wire).unwrap();
        value.as_object_mut().unwrap().remove("environment");
        let reloaded: FitWire = serde_json::from_value(value).unwrap();
        assert!(reloaded.environment.is_none());
    }

    #[test]
    fn method_str_roundtrips_every_variant() {
        // Guards against a new EstimationMethod variant being added to
        // method_to_str (write) but not method_from_str (read), which silently
        // breaks .fitrx load for fits produced with that method (e.g. #380's
        // `bayes`). Exhaustive over the enum so the compiler-style coverage is
        // enforced at test time.
        use crate::types::EstimationMethod::*;
        for m in [Foce, FoceI, FoceGn, FoceGnHybrid, Saem, Imp, Impmap, Bayes] {
            let s = method_to_str(m);
            assert_eq!(
                method_from_str(s).expect("method_from_str must accept method_to_str output"),
                m,
                "method round-trip failed for {s:?}"
            );
        }
    }

    #[test]
    fn roundtrip_with_kappa() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run2.fitrx");
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.05]));
        r.kappa_names = vec!["kappa_CL".into()];
        r.kappa_fixed = vec![false];
        r.shrinkage_kappa = vec![0.1];
        r.ebe_kappas = vec![
            vec![DVector::from_vec(vec![0.01]), DVector::from_vec(vec![0.02])],
            vec![DVector::from_vec(vec![-0.01])],
        ];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert!(loaded.fit.omega_iov.is_some());
        assert_eq!(loaded.fit.kappa_names, r.kappa_names);
        assert_eq!(loaded.fit.ebe_kappas.len(), 2);
        assert_eq!(loaded.fit.ebe_kappas[0].len(), 2);
        assert!((loaded.fit.ebe_kappas[0][0][0] - 0.01).abs() < 1e-9);
        assert_eq!(loaded.fit.ebe_kappas[1].len(), 1);
    }

    /// A weighted kappa (#1031) carries its declaration and the arm size it was
    /// read against into the bundle, so a reloaded fit still reports the
    /// effective SD rather than a bare gamma^2.
    #[test]
    fn roundtrip_with_a_weighted_kappa() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weighted_kappa.fitrx");
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[4.0]));
        r.kappa_names = vec!["KAPPA_EMAX".into()];
        r.kappa_fixed = vec![false];
        r.shrinkage_kappa = vec![0.1];
        r.kappa_weights = vec![Some("NARM".into())];
        r.kappa_weight_typical = vec![Some(200.0)];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.kappa_weights, vec![Some("NARM".to_string())]);
        assert_eq!(loaded.fit.kappa_weight_typical, vec![Some(200.0)]);
    }

    #[test]
    fn roundtrip_with_conddist() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conddist.fitrx");
        let mut r = minimal_fit_result();
        r.cond_dist = Some(CondDist {
            cond_mean: vec![vec![0.11, -0.19], vec![0.29, 0.41]],
            cond_sd: vec![vec![0.05, 0.06], vec![0.07, 0.08]],
            samples: vec![Vec::new(), Vec::new()],
            shrinkage: vec![0.2, 0.3],
            nsamp: 500,
            burnin: 100,
        });
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        let cd = loaded.fit.cond_dist.expect("cond_dist should round-trip");
        assert_eq!(cd.cond_mean, r.cond_dist.as_ref().unwrap().cond_mean);
        assert_eq!(cd.cond_sd, r.cond_dist.as_ref().unwrap().cond_sd);
        // samples/nsamp/burnin aren't carried by conddist.csv — they come back
        // empty/zero rather than the original values.
        assert!(cd.samples.iter().all(|s| s.is_empty()));
        assert_eq!(cd.nsamp, 0);
        assert_eq!(cd.burnin, 0);
        // shrinkage is recomputed from cond_mean + omega, not stored verbatim,
        // but for this fixture (2 subjects, well-conditioned omega) it should
        // reproduce the same values the original formula would give.
        for j in 0..2 {
            let m: f64 = (cd.cond_mean[0][j] + cd.cond_mean[1][j]) / 2.0;
            let var = ((cd.cond_mean[0][j] - m).powi(2) + (cd.cond_mean[1][j] - m).powi(2)) / 1.0;
            let expected = 1.0 - var.sqrt() / r.omega[(j, j)].sqrt();
            assert!((cd.shrinkage[j] - expected).abs() < 1e-9);
        }

        assert!(loaded.manifest.entries.iter().any(|e| e == "conddist.csv"));
    }

    #[test]
    fn roundtrip_omits_conddist_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_conddist.fitrx");
        let r = minimal_fit_result(); // cond_dist: None by default
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        assert!(loaded.fit.cond_dist.is_none());
        assert!(!loaded.manifest.entries.iter().any(|e| e == "conddist.csv"));
    }

    // Full valid conddist.csv for a `minimal_fit_result()` fit: S1/S2 with
    // eta_names [eta_CL, eta_V] and EBEs [0.1, 0.2] (from `dummy_subject`).
    const VALID_CONDDIST_CSV: &str = "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
        S1,eta_CL,0.11,0.05,0.100000\n\
        S1,eta_V,-0.19,0.06,0.200000\n\
        S2,eta_CL,0.29,0.07,0.100000\n\
        S2,eta_V,0.41,0.08,0.200000\n";

    #[test]
    fn parse_conddist_csv_accepts_complete_bundle() {
        let fit = minimal_fit_result();
        let cd = parse_conddist_csv(VALID_CONDDIST_CSV, &fit).unwrap();
        assert_eq!(cd.cond_mean, vec![vec![0.11, -0.19], vec![0.29, 0.41]]);
        assert_eq!(cd.cond_sd, vec![vec![0.05, 0.06], vec![0.07, 0.08]]);
    }

    #[test]
    fn parse_conddist_csv_rejects_duplicate_row() {
        let fit = minimal_fit_result();
        let csv = format!("{}S1,eta_CL,0.11,0.05,0.100000\n", VALID_CONDDIST_CSV);
        let err = parse_conddist_csv(&csv, &fit).unwrap_err();
        assert!(
            matches!(&err, FitrxError::Corrupt(m) if m.contains("duplicate")),
            "expected duplicate error, got {err:?}"
        );
    }

    #[test]
    fn parse_conddist_csv_rejects_missing_row() {
        let fit = minimal_fit_result();
        // Drop the last (S2, eta_V) row.
        let csv = "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
            S1,eta_CL,0.11,0.05,0.100000\n\
            S1,eta_V,-0.19,0.06,0.200000\n\
            S2,eta_CL,0.29,0.07,0.100000\n";
        let err = parse_conddist_csv(csv, &fit).unwrap_err();
        assert!(
            matches!(&err, FitrxError::Corrupt(m) if m.contains("missing")),
            "expected missing-row error, got {err:?}"
        );
    }

    #[test]
    fn parse_conddist_csv_rejects_mode_ebe_mismatch() {
        let fit = minimal_fit_result();
        // Corrupt one COND_MODE so it no longer matches the loaded EBE (0.1).
        let csv = "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
            S1,eta_CL,0.11,0.05,0.999999\n\
            S1,eta_V,-0.19,0.06,0.200000\n\
            S2,eta_CL,0.29,0.07,0.100000\n\
            S2,eta_V,0.41,0.08,0.200000\n";
        let err = parse_conddist_csv(csv, &fit).unwrap_err();
        assert!(
            matches!(&err, FitrxError::Corrupt(m) if m.contains("COND_MODE")),
            "expected COND_MODE mismatch error, got {err:?}"
        );
    }

    #[test]
    fn roundtrip_with_covariance_failed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run3.fitrx");
        let mut r = minimal_fit_result();
        r.covariance_status = CovarianceStatus::Failed;
        r.covariance_matrix = None;
        r.se_theta = None;
        r.se_omega = None;
        r.se_sigma = None;
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.covariance_status, CovarianceStatus::Failed);
        assert!(loaded.fit.covariance_matrix.is_none());
        assert!(loaded.fit.se_theta.is_none());
    }

    #[test]
    fn include_data_bundles_csv() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path().join("data.csv");
        std::fs::write(
            &data_path,
            "ID,TIME,DV,EVID,AMT,CMT\n1,0,0,1,100,1\n1,1,5,0,0,1\n",
        )
        .unwrap();
        let path = dir.path().join("run4.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(
            &r,
            &p,
            "src\n",
            &path,
            SaveFitOptions {
                include_data: Some(data_path),
            },
        )
        .unwrap();
        let loaded = load_fit(&path).unwrap();
        assert!(loaded.population.is_some());
    }

    #[test]
    fn bad_zip_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.fitrx");
        std::fs::write(&path, b"not a zip file").unwrap();
        let err = load_fit(&path).unwrap_err();
        match err {
            FitrxError::Zip(_) | FitrxError::Io(_) => {}
            other => panic!("expected zip/io error, got {:?}", other),
        }
    }

    #[test]
    fn manifest_records_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.manifest.format_version, "1");
        assert_eq!(loaded.manifest.ferx_version, "0.1.0");
        assert!(loaded.manifest.entries.contains(&"fit.json".to_string()));
    }

    #[test]
    fn epoch_to_utc_known_dates() {
        // 1970-01-01T00:00:00Z
        assert_eq!(epoch_to_utc(0), (1970, 1, 1, 0, 0, 0));
        // 2000-01-01T00:00:00Z = 946_684_800
        assert_eq!(epoch_to_utc(946_684_800), (2000, 1, 1, 0, 0, 0));
        // 2024-01-01T00:00:00Z = 1_704_067_200
        assert_eq!(epoch_to_utc(1_704_067_200), (2024, 1, 1, 0, 0, 0));
        // 2026-05-15T00:00:00Z = 1_778_803_200
        assert_eq!(epoch_to_utc(1_778_803_200), (2026, 5, 15, 0, 0, 0));
    }

    #[test]
    fn roundtrip_preserves_nan_and_inf() {
        // Fits with sparse data produce NaN shrinkage_eps; singular Hessians
        // produce f64::INFINITY cov_condition_number. Both must survive the
        // JSON round-trip (encoded as `null` on disk, NaN on load).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan.fitrx");
        let mut r = minimal_fit_result();
        r.shrinkage_eps = f64::NAN;
        r.shrinkage_eta = vec![f64::NAN, 0.15];
        r.cov_condition_number = Some(f64::INFINITY);
        // Plant a NaN inside the covariance matrix data too.
        if let Some(cov) = r.covariance_matrix.as_mut() {
            cov[(0, 0)] = f64::NAN;
        }
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert!(loaded.fit.shrinkage_eps.is_nan());
        assert!(loaded.fit.shrinkage_eta[0].is_nan());
        assert_eq!(loaded.fit.shrinkage_eta[1], 0.15);
        // cov_condition_number was +Inf; round-trips as None (null) under the
        // current serde_nan::opt adapter. Either NaN or None is fine —
        // both mean "could not be computed reliably" downstream.
        assert!(loaded
            .fit
            .cov_condition_number
            .map(|v| !v.is_finite())
            .unwrap_or(true));
        let cov = loaded.fit.covariance_matrix.as_ref().expect("cov matrix");
        assert!(cov[(0, 0)].is_nan());
    }

    #[test]
    fn load_rejects_predictions_row_count_mismatch() {
        // Hand-build a .fitrx with a one-row-short predictions.csv vs the
        // n_obs reported in ebes.csv. parse_subjects should detect it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad-rows.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        // Rewrite predictions.csv with one row removed for S1.
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if name == "predictions.csv" {
                let csv = String::from_utf8(buf).unwrap();
                let mut lines: Vec<&str> = csv.lines().collect();
                // Drop the last data row (last S2 observation) so the final
                // subject comes up one short. Positional parsing fills S2 with
                // the rows it has, then the n_obs cross-validation catches the
                // shortfall at the tail. (Dropping a middle row instead would
                // trip the earlier ID-order check, a different code path.)
                lines.pop();
                buf = lines.join("\n").into_bytes();
                buf.push(b'\n');
            }
            entries.push((name, buf));
        }
        let bad = dir.path().join("bad-rows-rewritten.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&bad).unwrap());
        for (name, body) in entries {
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&body).unwrap();
        }
        zw.finish().unwrap();

        let err = load_fit(&bad).unwrap_err();
        match err {
            FitrxError::Corrupt(msg) => assert!(msg.contains("n_obs"), "msg = {}", msg),
            other => panic!("expected Corrupt, got {:?}", other),
        }
    }

    #[test]
    fn load_handles_duplicate_subject_ids() {
        // Two distinct subjects can share a textual ID (an ID reused across
        // studies, or a reset-split subject). ebes.csv / predictions.csv are
        // written one contiguous block per subject in subject order, so the
        // loader must assign prediction rows positionally. Keying on ID (as it
        // once did) collapses the two "12" subjects — routing every row to one
        // and leaving the other with zero, which then trips the n_obs check.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup-ids.fitrx");
        let mut r = minimal_fit_result();
        let n_eta = 2;
        let a = dummy_subject("12", n_eta, 3);
        let mut b = dummy_subject("12", n_eta, 5);
        // Offset the second block so we can prove the two never cross-mix.
        b.ipred = b.ipred.iter().map(|v| v + 100.0).collect();
        r.subjects = vec![a, b];
        r.n_obs = 8;
        r.n_subjects = 2;
        let p = dummy_population(&["12", "12"], 5);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        let s = &loaded.fit.subjects;
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].id, "12");
        assert_eq!(s[1].id, "12");
        assert_eq!(s[0].ipred.len(), 3, "first block keeps its own n_obs");
        assert_eq!(s[1].ipred.len(), 5, "second block keeps its own n_obs");
        assert!(
            s[0].ipred.iter().all(|&v| v < 100.0),
            "no bleed into block A"
        );
        assert!(s[1].ipred.iter().all(|&v| v > 100.0), "block B rows intact");
    }

    #[test]
    fn load_rejects_misordered_duplicate_id_blocks() {
        // Two same-ID blocks written in the wrong order: the ID order check
        // passes (both are "12"), so the per-row N_OBS guard is what catches it.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dup-swapped.fitrx");
        let mut r = minimal_fit_result();
        let n_eta = 2;
        r.subjects = vec![dummy_subject("12", n_eta, 3), dummy_subject("12", n_eta, 5)];
        r.n_obs = 8;
        r.n_subjects = 2;
        let p = dummy_population(&["12", "12"], 5);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        // Reorder predictions.csv so the 5-row block precedes the 3-row block.
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if name == "predictions.csv" {
                let csv = String::from_utf8(buf).unwrap();
                let lines: Vec<&str> = csv.lines().collect();
                // header, then 3 rows (block A), then 5 rows (block B).
                let header = lines[0];
                let block_a = &lines[1..4];
                let block_b = &lines[4..9];
                let mut reordered = vec![header];
                reordered.extend_from_slice(block_b);
                reordered.extend_from_slice(block_a);
                buf = reordered.join("\n").into_bytes();
                buf.push(b'\n');
            }
            entries.push((name, buf));
        }
        let bad = dir.path().join("dup-swapped-rewritten.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&bad).unwrap());
        for (name, body) in entries {
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&body).unwrap();
        }
        zw.finish().unwrap();

        let err = load_fit(&bad).unwrap_err();
        match err {
            FitrxError::Corrupt(msg) => assert!(msg.contains("N_OBS"), "msg = {}", msg),
            other => panic!("expected Corrupt, got {:?}", other),
        }
    }

    #[test]
    fn load_rejects_parallel_array_mismatch() {
        // Hand-edit fit.json to drop a theta_fixed entry; the loader should
        // reject the bundle before downstream code can panic on an index.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad-parallel.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if name == "fit.json" {
                let mut v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
                let fixed = v["theta"]["fixed"].as_array_mut().unwrap();
                fixed.pop(); // induce length mismatch
                buf = serde_json::to_vec_pretty(&v).unwrap();
            }
            entries.push((name, buf));
        }
        let bad = dir.path().join("bad-parallel-rewritten.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&bad).unwrap());
        for (name, body) in entries {
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&body).unwrap();
        }
        zw.finish().unwrap();

        let err = load_fit(&bad).unwrap_err();
        match err {
            FitrxError::Corrupt(msg) => {
                assert!(msg.contains("theta.fixed"), "msg = {}", msg)
            }
            other => panic!("expected Corrupt, got {:?}", other),
        }
    }

    #[test]
    fn parse_ebe_kappas_uses_occ_column() {
        // OCC=2 row appears first; the loader must still slot kappas at
        // [occ-1], not at file-order positions.
        let csv = "ID,OCC,kappa_CL\nS1,2,0.020000\nS1,1,0.010000\n";
        let result = parse_ebe_kappas(csv, &["S1".into()]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        assert!((result[0][0][0] - 0.01).abs() < 1e-9);
        assert!((result[0][1][0] - 0.02).abs() < 1e-9);
    }

    #[test]
    fn parse_ebe_kappas_rejects_duplicate_occ() {
        let csv = "ID,OCC,kappa_CL\nS1,1,0.010000\nS1,1,0.020000\n";
        let err = parse_ebe_kappas(csv, &["S1".into()]).unwrap_err();
        match err {
            FitrxError::Corrupt(msg) => assert!(msg.contains("OCC"), "msg = {}", msg),
            other => panic!("expected Corrupt, got {:?}", other),
        }
    }

    #[test]
    fn parse_ebe_kappas_rejects_zero_occ() {
        let csv = "ID,OCC,kappa_CL\nS1,0,0.010000\n";
        let err = parse_ebe_kappas(csv, &["S1".into()]).unwrap_err();
        match err {
            FitrxError::Corrupt(msg) => assert!(msg.contains("1-based"), "msg = {}", msg),
            other => panic!("expected Corrupt, got {:?}", other),
        }
    }

    // ── Fix 3: omega_init backward-compat fallback ───────────────────────────

    /// Loading a .fitrx bundle that pre-dates the omega_init field should fall
    /// back to the converged omega, not to a zero matrix.  A zero matrix is not
    /// positive-definite and would break any Cholesky-based consumer.
    #[test]
    fn omega_init_fallback_uses_omega_not_zeros() {
        use std::io::{Read as _, Write as _};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no-omega-init.fitrx");
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        // Strip omega_init from fit.json to simulate a pre-PR bundle.
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().to_string();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if name == "fit.json" {
                let mut v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
                v.as_object_mut().unwrap().remove("omega_init");
                buf = serde_json::to_vec_pretty(&v).unwrap();
            }
            entries.push((name, buf));
        }
        let patched = dir.path().join("no-omega-init-patched.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&patched).unwrap());
        for (name, body) in entries {
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&body).unwrap();
        }
        zw.finish().unwrap();

        let loaded = load_fit(&patched).unwrap();
        let omega_init = &loaded.fit.omega_init;
        let omega = &loaded.fit.omega;

        // Should match the converged omega, not be an all-zero matrix.
        assert_eq!(
            omega_init, omega,
            "omega_init should fall back to converged omega, got {:?}",
            omega_init
        );
        assert!(
            omega_init.iter().any(|&v| v != 0.0),
            "omega_init must not be all zeros"
        );
    }

    #[test]
    fn covariate_names_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cov-roundtrip.fitrx");
        let r = minimal_fit_result(); // has covariate_names: ["WT", "AGE"]
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(
            loaded.fit.covariate_names,
            vec!["WT".to_string(), "AGE".to_string()]
        );
    }

    #[test]
    fn input_columns_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ic-roundtrip.fitrx");
        let mut r = minimal_fit_result();
        r.input_columns = vec![
            "ID".into(),
            "TIME".into(),
            "DV".into(),
            "AMT".into(),
            "WT".into(),
        ];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.input_columns, r.input_columns);
    }
}
