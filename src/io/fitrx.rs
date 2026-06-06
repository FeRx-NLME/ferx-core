//! `.fitrx` save/load — zip-of-JSON-plus-CSV bundle for fit objects.
//!
//! Layout (all entries are deflate-compressed inside a zip archive):
//!
//! - `manifest.json`     — format version, ferx version, timestamp, entry index
//! - `fit.json`          — scalars / vectors / matrices on `FitResult`
//! - `ebes.csv`          — per-subject EBEs (`ID, eta_1..eta_n, ofv_contribution, n_obs`)
//! - `ebes_kappa.csv`    — per-(subject, occasion) kappa EBEs (only when `n_kappa > 0`)
//! - `predictions.csv`   — per-observation predictions joined with TIME/DV
//! - `model.ferx`        — verbatim model source
//! - `warnings.txt`      — one warning per line (mirrors `fit.json` for grep)
//! - `data.csv`          — copy of the input NONMEM CSV (only when caller opts in)
//!
//! See `docs/src/file-formats/fitrx.md` for the field-by-field schema.

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

// ---------------------------------------------------------------------------
// Serde helpers for non-finite floats
// ---------------------------------------------------------------------------
//
// JSON has no representation for NaN/±Inf. `serde_json` writes them as
// `null` on serialize, which is correct per the spec but means the default
// `Deserialize` for `f64` then fails when reading the value back. These
// helpers make the round-trip lossless: non-finite serializes to `null`,
// and `null` deserializes back to `NaN`. Use via `#[serde(with = "...")]`
// on fields that may legitimately carry NaN (shrinkage, cov_condition_number
// when the Hessian is singular, etc.).

mod f64_nan_as_null {
    use serde::{de::Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f64, ser: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() {
            ser.serialize_f64(*value)
        } else {
            ser.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<f64, D::Error> {
        let opt: Option<f64> = Option::deserialize(de)?;
        Ok(opt.unwrap_or(f64::NAN))
    }
}

mod vec_f64_nan_as_null {
    use serde::{de::Deserialize, ser::SerializeSeq, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Vec<f64>, ser: S) -> Result<S::Ok, S::Error> {
        let mut seq = ser.serialize_seq(Some(value.len()))?;
        for v in value {
            if v.is_finite() {
                seq.serialize_element(v)?;
            } else {
                seq.serialize_element(&Option::<f64>::None)?;
            }
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<f64>, D::Error> {
        let opts: Vec<Option<f64>> = Vec::deserialize(de)?;
        Ok(opts.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
    }
}

mod vec_vec_f64_nan_as_null {
    use serde::{de::Deserialize, ser::SerializeSeq, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Vec<Vec<f64>>, ser: S) -> Result<S::Ok, S::Error> {
        let mut outer = ser.serialize_seq(Some(value.len()))?;
        for inner in value {
            // Delegate each inner Vec<f64> to the same NaN-as-null logic.
            struct NanSafeSlice<'a>(&'a [f64]);
            impl serde::Serialize for NanSafeSlice<'_> {
                fn serialize<S2: Serializer>(&self, s: S2) -> Result<S2::Ok, S2::Error> {
                    use serde::ser::SerializeSeq as _;
                    let mut seq = s.serialize_seq(Some(self.0.len()))?;
                    for v in self.0 {
                        if v.is_finite() {
                            seq.serialize_element(v)?;
                        } else {
                            seq.serialize_element(&Option::<f64>::None)?;
                        }
                    }
                    seq.end()
                }
            }
            outer.serialize_element(&NanSafeSlice(inner))?;
        }
        outer.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Vec<Vec<f64>>, D::Error> {
        let outer: Vec<Vec<Option<f64>>> = Vec::deserialize(de)?;
        Ok(outer
            .into_iter()
            .map(|inner| inner.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
            .collect())
    }
}

mod opt_f64_nan_as_null {
    use serde::{de::Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Option<f64>, ser: S) -> Result<S::Ok, S::Error> {
        match value {
            Some(v) if v.is_finite() => ser.serialize_some(v),
            _ => ser.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<f64>, D::Error> {
        Option::<f64>::deserialize(de)
    }
}

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
    #[serde(with = "f64_nan_as_null")]
    shrinkage_eps: f64,
    #[serde(default = "default_nan", with = "f64_nan_as_null")]
    iwres_lag1_r: f64,
    #[serde(default = "default_nan", with = "f64_nan_as_null")]
    dw_statistic: f64,
    covariance_matrix: Option<MatrixWire>,
    cov_eigenvalues: Option<Vec<f64>>,
    #[serde(with = "opt_f64_nan_as_null")]
    cov_condition_number: Option<f64>,

    sir: Option<SirWire>,
    iov: Option<IovWire>,

    eta_param_info: Vec<EtaParamInfoWire>,
    model_name: String,
    ferx_version: String,
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
    is_seed: Option<u64>,
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
    #[serde(with = "vec_f64_nan_as_null")]
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
    #[serde(with = "vec_f64_nan_as_null")]
    shrinkage_kappa: Vec<f64>,
    /// Per-occasion kappa shrinkage: `[occ_idx][kappa_idx]`.
    /// Defaulted for backward compatibility with older .fitrx files.
    #[serde(default, with = "vec_vec_f64_nan_as_null")]
    shrinkage_kappa_by_occ: Vec<Vec<f64>>,
    omega_iov: MatrixWire,
    omega_iov_param_corr: Option<MatrixWire>,
    /// Per-kappa SD-init flag. Defaulted for backward compatibility with
    /// .fitrx files from before issue #5.
    #[serde(default)]
    kappa_init_as_sd: Vec<bool>,
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
    #[serde(with = "vec_f64_nan_as_null")]
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

fn method_to_str(m: EstimationMethod) -> &'static str {
    match m {
        EstimationMethod::Foce => "foce",
        EstimationMethod::FoceI => "focei",
        EstimationMethod::FoceGn => "foce_gn",
        EstimationMethod::FoceGnHybrid => "foce_gn_hybrid",
        EstimationMethod::Saem => "saem",
        EstimationMethod::Imp => "imp",
    }
}

fn method_from_str(s: &str) -> Result<EstimationMethod, FitrxError> {
    Ok(match s {
        "foce" => EstimationMethod::Foce,
        "focei" => EstimationMethod::FoceI,
        "foce_gn" => EstimationMethod::FoceGn,
        "foce_gn_hybrid" => EstimationMethod::FoceGnHybrid,
        "saem" => EstimationMethod::Saem,
        "imp" => EstimationMethod::Imp,
        _ => return Err(FitrxError::Corrupt(format!("unknown method {:?}", s))),
    })
}

fn error_model_to_str(m: ErrorModel) -> &'static str {
    match m {
        ErrorModel::Additive => "additive",
        ErrorModel::Proportional => "proportional",
        ErrorModel::Combined => "combined",
    }
}

fn error_model_from_str(s: &str) -> Result<ErrorModel, FitrxError> {
    Ok(match s {
        "additive" => ErrorModel::Additive,
        "proportional" => ErrorModel::Proportional,
        "combined" => ErrorModel::Combined,
        _ => return Err(FitrxError::Corrupt(format!("unknown error_model {:?}", s))),
    })
}

fn covariance_status_to_str(s: &CovarianceStatus) -> &'static str {
    match s {
        CovarianceStatus::NotRequested => "not_requested",
        CovarianceStatus::Computed => "computed",
        CovarianceStatus::Failed => "failed",
    }
}

fn covariance_status_from_str(s: &str) -> Result<CovarianceStatus, FitrxError> {
    Ok(match s {
        "not_requested" => CovarianceStatus::NotRequested,
        "computed" => CovarianceStatus::Computed,
        "failed" => CovarianceStatus::Failed,
        _ => {
            return Err(FitrxError::Corrupt(format!(
                "unknown covariance_status {:?}",
                s
            )))
        }
    })
}

fn theta_transform_to_str(t: ThetaTransform) -> &'static str {
    match t {
        ThetaTransform::Identity => "identity",
        ThetaTransform::Log => "log",
        ThetaTransform::Logit => "logit",
        ThetaTransform::LogitProbability => "logit_probability",
    }
}

fn theta_transform_from_str(s: &str) -> Result<ThetaTransform, FitrxError> {
    Ok(match s {
        "identity" => ThetaTransform::Identity,
        "log" => ThetaTransform::Log,
        "logit" => ThetaTransform::Logit,
        "logit_probability" => ThetaTransform::LogitProbability,
        _ => {
            return Err(FitrxError::Corrupt(format!(
                "unknown theta_transform {:?}",
                s
            )))
        }
    })
}

fn sigma_type_to_str(t: SigmaType) -> &'static str {
    match t {
        SigmaType::Proportional => "proportional",
        SigmaType::Additive => "additive",
    }
}

fn sigma_type_from_str(s: &str) -> Result<SigmaType, FitrxError> {
    Ok(match s {
        "proportional" => SigmaType::Proportional,
        "additive" => SigmaType::Additive,
        _ => return Err(FitrxError::Corrupt(format!("unknown sigma_type {:?}", s))),
    })
}

fn eta_param_type_to_str(t: EtaParamType) -> &'static str {
    match t {
        EtaParamType::LogNormal => "log_normal",
        EtaParamType::Additive => "additive",
        EtaParamType::Logit => "logit",
        EtaParamType::LogitProbability => "logit_probability",
        EtaParamType::Custom => "custom",
    }
}

fn eta_param_type_from_str(s: &str) -> Result<EtaParamType, FitrxError> {
    Ok(match s {
        "log_normal" => EtaParamType::LogNormal,
        "additive" => EtaParamType::Additive,
        "logit" => EtaParamType::Logit,
        "logit_probability" => EtaParamType::LogitProbability,
        "custom" => EtaParamType::Custom,
        _ => {
            return Err(FitrxError::Corrupt(format!(
                "unknown eta_param_type {:?}",
                s
            )))
        }
    })
}

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
        is_seed: r.is_seed,
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

fn write_predictions_csv<W: Write>(
    w: &mut W,
    r: &FitResult,
    p: &Population,
) -> Result<(), FitrxError> {
    let any_cens = r.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = p.subjects.iter().any(|s| !s.occasions.is_empty());

    let mut header = String::from("ID,TIME,DV,PRED,IPRED,CWRES,IWRES,EBE_OFV,N_OBS");
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
            row.push_str(&fmt_f64(subj.obs_times[j]));
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
    if v.is_nan() {
        String::new()
    } else {
        format!("{:.6}", v)
    }
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
    let preds_csv = read_text(&mut archive, "predictions.csv")?;
    let model_source = read_text(&mut archive, "model.ferx")?;

    // data.csv is optional — re-parse only when present.
    let population = if archive.file_names().any(|n| n == "data.csv") {
        let data_csv_bytes = read_bytes(&mut archive, "data.csv")?;
        let tmp = tempfile::NamedTempFile::new()?;
        std::fs::write(tmp.path(), &data_csv_bytes)?;
        Some(
            crate::io::datareader::read_nonmem_csv(tmp.path(), None, None)
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

    Ok(LoadedFit {
        fit,
        model_source,
        population,
        manifest,
    })
}

fn read_json<T: serde::de::DeserializeOwned, R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<T, FitrxError> {
    let mut file = archive
        .by_name(name)
        .map_err(|_| FitrxError::Corrupt(format!("missing entry {}", name)))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(serde_json::from_str(&buf)?)
}

fn read_text<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<String, FitrxError> {
    let mut file = archive
        .by_name(name)
        .map_err(|_| FitrxError::Corrupt(format!("missing entry {}", name)))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(buf)
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
    let expected_cols = 1 + n_eta + 2;
    let header_cols = header.split(',').count();
    if header_cols != expected_cols {
        return Err(FitrxError::Corrupt(format!(
            "ebes.csv header has {} columns, expected {}",
            header_cols, expected_cols
        )));
    }
    let mut subjects: Vec<SubjectResult> = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() != expected_cols {
            return Err(FitrxError::Corrupt(format!(
                "ebes.csv row {} has {} fields, expected {}",
                i + 1,
                fields.len(),
                expected_cols
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
        subjects.push(SubjectResult {
            id,
            eta,
            ipred: Vec::new(),
            pred: Vec::new(),
            iwres: Vec::new(),
            cwres: Vec::new(),
            ofv_contribution: ofv,
            cens: Vec::new(),
            n_obs,
        });
    }

    // predictions.csv → fill ipred/pred/iwres/cwres/cens grouped by ID,
    // assuming subject rows are contiguous (which save_fit guarantees).
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
    let cens_i = col.get("CENS").copied();

    let mut by_id: HashMap<String, usize> = HashMap::new();
    for (idx, s) in subjects.iter().enumerate() {
        by_id.insert(s.id.clone(), idx);
    }

    for (i, line) in plines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        let id = fields
            .get(id_i)
            .ok_or_else(|| FitrxError::Corrupt(format!("predictions.csv row {}: short", i + 1)))?
            .clone();
        let idx = *by_id
            .get(&id)
            .ok_or_else(|| FitrxError::Corrupt(format!("predictions.csv: unknown ID {:?}", id)))?;
        let parse_opt = |s: &str| -> f64 {
            if s.is_empty() {
                f64::NAN
            } else {
                s.parse().unwrap_or(f64::NAN)
            }
        };
        subjects[idx].pred.push(parse_opt(&fields[pred_i]));
        subjects[idx].ipred.push(parse_opt(&fields[ipred_i]));
        subjects[idx].cwres.push(parse_opt(&fields[cwres_i]));
        subjects[idx].iwres.push(parse_opt(&fields[iwres_i]));
        let c = match cens_i {
            Some(j) => fields
                .get(j)
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0),
            None => 0,
        };
        subjects[idx].cens.push(c);
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

    Ok(FitResult {
        method,
        method_chain,
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
        error_model: error_model_from_str(&w.error_model)?,
        covariance_matrix,
        se_theta: w.theta.se,
        se_omega: w.omega.se,
        se_sigma: w.sigma.se,
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
        omega_iov,
        kappa_names,
        kappa_fixed,
        kappa_init_as_sd,
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
        shrinkage_eps: w.shrinkage_eps,
        iwres_lag1_r: w.iwres_lag1_r,
        dw_statistic: w.dw_statistic,
        wall_time_secs: w.wall_time_secs,
        model_name: w.model_name,
        ferx_version: w.ferx_version,
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
        is_seed: w.is_seed,
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

    fn dummy_subject(id: &str, n_eta: usize, n_obs: usize) -> SubjectResult {
        SubjectResult {
            id: id.into(),
            eta: DVector::from_vec((0..n_eta).map(|k| 0.1 * (k as f64 + 1.0)).collect()),
            ipred: (0..n_obs).map(|j| 1.0 + j as f64).collect(),
            pred: (0..n_obs).map(|j| 1.5 + j as f64).collect(),
            iwres: (0..n_obs).map(|j| 0.01 * j as f64).collect(),
            cwres: (0..n_obs).map(|j| -0.02 * j as f64).collect(),
            ofv_contribution: 12.34,
            cens: vec![0; n_obs],
            n_obs,
        }
    }

    fn dummy_population(ids: &[&str], n_obs_each: usize) -> Population {
        let mut subjects = Vec::new();
        for id in ids {
            subjects.push(Subject {
                id: (*id).to_string(),
                doses: vec![],
                obs_times: (0..n_obs_each).map(|j| j as f64).collect(),
                observations: (0..n_obs_each).map(|j| 5.0 + j as f64).collect(),
                obs_cmts: vec![1; n_obs_each],
                covariates: HashMap::new(),
                dose_covariates: vec![],
                obs_covariates: vec![],
                pk_only_times: vec![],
                pk_only_covariates: vec![],
                reset_times: vec![],
                cens: vec![0; n_obs_each],
                occasions: vec![],
                dose_occasions: vec![],
            });
        }
        Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
        }
    }

    fn minimal_fit_result() -> FitResult {
        let n_eta = 2;
        FitResult {
            method: EstimationMethod::FoceI,
            method_chain: vec![EstimationMethod::FoceI],
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
            omega_iov: None,
            kappa_names: vec![],
            kappa_fixed: vec![],
            kappa_init_as_sd: vec![],
            se_kappa: None,
            shrinkage_kappa: vec![],
            shrinkage_kappa_by_occ: vec![],
            ebe_kappas: vec![],
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: "autodiff".into(),
            gradient_method_outer: "autodiff".into(),
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
            shrinkage_eps: 0.05,
            iwres_lag1_r: 0.12,
            dw_statistic: 1.75,
            wall_time_secs: 1.234,
            model_name: "test_model".into(),
            ferx_version: "0.1.0".into(),
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
            is_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 300,
            outer_gtol: 1e-4,
            inits_from_nca: None,
            covariate_names: vec!["WT".into(), "AGE".into()],
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
        }
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
        // current opt_f64_nan_as_null adapter. Either NaN or None is fine —
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
                // Drop the second data row (first S1 observation) to simulate
                // a writer that miscounted.
                lines.remove(2);
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
