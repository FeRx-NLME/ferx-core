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
//! - `predictions_discrete.csv` — per-record binary-endpoint diagnostics (only when the
//!                          fit has any; `ebes.csv` then carries `n_obs_discrete`)
//! - `covtab.csv`        — the covariate table (only with a `[covariates]` block)
//! - `trace.csv`         — the optimizer trace (only with `optimizer_trace = true`)
//! - `model.ferx`        — verbatim model source
//! - `warnings.txt`      — one warning per line (mirrors `fit.json` for grep)
//! - `data.csv`          — copy of the input NONMEM CSV (only when caller opts in)
//! - `interface_extras.json` — an interface's own data, opaque to core (only when set)
//!
//! See `docs/file-formats/fitrx.qmd` for the field-by-field schema.

use crate::parser::model_parser::{DataBindings, LevelBinding};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

pub const FORMAT_VERSION: &str = "2";

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
    /// [`save_fit`] refused a fit whose recorded settings [`load_fit`] would reject as
    /// `Corrupt` (#1770): a `scoring_settings` or `sir_settings` value outside the
    /// `[fit_options]` domain, which only a `FitResult` edited after the fit can hold.
    /// Also a data file or model text whose SHA-256 is not the one the fit recorded
    /// (#2696): that is not what was fitted. Nothing is written.
    #[error("refusing to write .fitrx: {0}")]
    Unwritable(String),
}

/// Options for [`save_fit`].
#[derive(Debug, Clone, Default)]
pub struct SaveFitOptions {
    /// When `Some(path)`, the file at `path` is embedded verbatim as `data.csv`
    /// inside the archive, provided its bytes match `FitResult::data_hash` (a fit
    /// with no recorded hash is embedded unchecked). When `None`, no data is bundled.
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
    /// The model-selection verdicts as the saving ferx reached them.
    pub verdicts: FitVerdicts,
}

/// One subject's data columns as `predictions.csv` holds them.
#[derive(Debug, Clone, Default)]
struct RecordedRows {
    time: Vec<f64>,
    dv: Vec<f64>,
    occ: Vec<u32>,
}

/// Where `predictions.csv` gets the columns [`FitResult`] does not hold.
enum ObservedRows<'a> {
    Population(&'a Population),
    Recorded(&'a [RecordedRows]),
}

impl ObservedRows<'_> {
    fn any_occ(&self) -> bool {
        match self {
            Self::Population(p) => p.subjects.iter().any(|s| !s.occasions.is_empty()),
            Self::Recorded(rows) => rows.iter().any(|r| !r.occ.is_empty()),
        }
    }

    /// The raw data TIME, so the bundle's predictions.csv joins back to the input CSV
    /// and to the sdtab, which is also raw; `obs_times` may be the internal shifted
    /// timeline for stacked reset occasions. Falls back to `obs_times` when no raw
    /// vector was recorded.
    fn time(&self, subject: usize, j: usize) -> f64 {
        match self {
            Self::Population(p) => p.subjects[subject].readout_time(j),
            Self::Recorded(rows) => rows[subject].time[j],
        }
    }

    fn dv(&self, subject: usize, j: usize) -> f64 {
        match self {
            Self::Population(p) => p.subjects[subject].observations[j],
            Self::Recorded(rows) => rows[subject].dv[j],
        }
    }

    fn occ(&self, subject: usize, j: usize) -> u32 {
        let occasions = match self {
            Self::Population(p) => &p.subjects[subject].occasions,
            Self::Recorded(rows) => &rows[subject].occ,
        };
        occasions.get(j).copied().unwrap_or(0)
    }
}

/// The [`crate::model_selection`] verdicts recorded in `fit.json`, derived from the fit
/// when it was saved so a reader with no engine has them. A loaded [`FitResult`] gives the
/// same answers through those functions; this is what the file says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitVerdicts {
    /// [`stalled_at_init`](crate::model_selection::stalled_at_init).
    pub stalled_at_init: Option<bool>,
    /// [`estimate_near_boundary`](crate::model_selection::estimate_near_boundary).
    pub estimate_near_boundary: bool,
    /// [`max_abs_correlation`](crate::model_selection::max_abs_correlation).
    pub max_abs_correlation: Option<f64>,
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
    /// `NaN` for a VI fit under the default `vi_final_ofv = none` (#2058), and so
    /// `aic` / `bic` with it; written as `null`, read back as `NaN`.
    #[serde(with = "crate::io::serde_nan::scalar")]
    ofv: f64,
    /// Parameter-prior split (#254): the data likelihood and the prior's share of `ofv`.
    /// `ofv` alone is the penalized total, so storing only it would relabel that as the
    /// data likelihood and break the `aic == ofv_data + 2k` invariant `aic` was computed
    /// under. `NaN` for a VI fit, like `ofv` (#2058).
    #[serde(with = "crate::io::serde_nan::scalar")]
    ofv_data: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    ofv_prior: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    prior_summary: Vec<crate::types::PriorSummary>,
    #[serde(with = "crate::io::serde_nan::scalar")]
    aic: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    bic: f64,
    n_obs: usize,
    n_subjects: usize,
    n_parameters: usize,
    n_iterations: usize,
    interaction: bool,
    wall_time_secs: f64,
    method_wall_times_secs: Vec<f64>,
    covariance_wall_time_secs: f64,
    n_threads_used: usize,
    uses_ode_solver: bool,
    ode_linear_exact: bool,
    uses_sde: bool,
    gradient_method_inner: String,
    gradient_method_outer: String,
    nlopt_missing_algorithms: Vec<String>,
    covariance_status: String,
    /// `r` / `s` / `rsr` — which estimator produced `covariance_matrix` (#1382).
    /// Absent when the fit ran no covariance step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    covariance_method: Option<String>,
    covariance_n_evals_estimated: Option<usize>,
    trace_path: Option<String>,
    ebe_convergence_warnings: u32,
    max_unconverged_subjects: u32,
    total_ebe_fallbacks: u32,
    /// `FitResult::total_ebe_nm_polished` (#2571).
    total_ebe_nm_polished: u32,
    warnings: Vec<String>,
    /// The structured warnings that carry a `details` payload (#2268), as
    /// `WarningEntry` JSON. Only those: every other entry is rebuilt from
    /// `warnings` on load, exactly as a fresh fit builds it. Kept as raw values
    /// so a bundle naming a `WarningCode` this version does not know still loads
    /// (that entry is re-classified from its message instead).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    warning_details: Vec<serde_json::Value>,
    saem_mu_ref_m_step_evals_saved: Option<u64>,
    saem_n_subjects_hmc: Option<usize>,
    /// Post-burn-in tail MH acceptance (#1444).
    saem_mh_accept_tail: Option<f64>,

    theta: ThetaWire,
    omega: OmegaWire,
    sigma: SigmaWire,
    error_model: String,
    #[serde(with = "crate::io::serde_nan::scalar")]
    shrinkage_eps: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    iwres_lag1_r: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    dw_statistic: f64,
    covariance_matrix: Option<MatrixWire>,
    cov_eigenvalues: Option<Vec<f64>>,
    #[serde(with = "crate::io::serde_nan::opt")]
    cov_condition_number: Option<f64>,
    /// Which packed coordinates of `covariance_matrix` the step held (#424), parallel to
    /// its rows. Absent when there is no matrix or no mask for it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cov_fixed: Vec<bool>,
    bic_inputs: BicInputs,
    // #1065. Absent on every fit without a profiled block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    profiled_blocks: Vec<String>,
    // #2112. Absent on every fit with no profiled level on its bound.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    profiled_levels_at_bound: Vec<usize>,
    // #2797. Absent on every fit with no held coordinate, and on every bundle written
    // before the field existed; such a bundle's `profiled_levels_at_bound` is synthesised
    // into `ProfiledBound` entries on load (`held_at_bound_on_load`). An entry of a
    // parameter family this build does not know is dropped on read (#2817).
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "crate::types::held_at_bound_lenient"
    )]
    held_at_bound: Vec<crate::types::HeldAtBound>,

    sir: Option<SirWire>,
    iov: Option<IovWire>,
    /// The engine results that serialize whole. Their floats can be non-finite, which
    /// `serde_json` writes as `null`, so they are read through [`nan_tolerant`].
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "nan_tolerant"
    )]
    bayes: Option<BayesResult>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "nan_tolerant"
    )]
    importance_sampling: Option<ImportanceSamplingResult>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "nan_tolerant"
    )]
    impmap_trace: Option<ImpmapTrace>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exclusions: Option<ExclusionSummary>,

    /// The [`FitVerdicts`], derived from the fit at write time.
    stalled_at_init: Option<bool>,
    estimate_near_boundary: bool,
    #[serde(with = "crate::io::serde_nan::opt")]
    max_abs_correlation: Option<f64>,

    eta_param_info: Vec<EtaParamInfoWire>,
    model_name: String,
    ferx_version: String,
    environment: crate::environment::EnvironmentInfo,
    // Source-file provenance: absent for a fit not read from files.
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
    theta_init: Vec<f64>,
    omega_init: MatrixWire,
    // The optimizer's own init-escape verdict (#1177). Absent for an engine that gives
    // none; `stalled_at_init` then compares the estimates against `theta_init` & co.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    left_init: Option<bool>,
    // Whether a fit that did not leave its start was verified to have started at an
    // optimum (#2506). Absent when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    stationary_start: bool,
    // The packed Ω / κ layout (#1177).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    omega_is_diagonal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kappa_is_diagonal: Option<bool>,
    sigma_init: Vec<f64>,
    obs_time_range: Option<(f64, f64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_gradient: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_gradient_source: Option<String>,
    // The covariance step's Newton decrement and the optimizer's stop state (#2785).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    newton_decrement: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optimizer_stop: Option<String>,
    // #2799: the facts `FitResult::status` reads beside `converged`. A bundle from before
    // them reads `false`, which is what such a fit reported (no typed budget fact, and an
    // evaluation-only run was told apart only by `outer_maxiter = 0`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    eval_budget_spent: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    evaluation_only: bool,
    // ── Run settings ─────────────────────────────────────────────────────────
    optimizer: String,
    n_starts: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    multi_start_seed: Option<u64>,
    /// The per-start records of a multi-start fit (#2867). Absent for a single-start fit,
    /// and in a bundle written before the field, which then loads with no rows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    starts: Vec<crate::types::StartRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    saem_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sir_seed: Option<u64>,
    /// The settings of the stage that produced the estimates (#426).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scoring_settings: Option<ScoringSettingsWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imp_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    npde_seed: Option<u64>,
    bloq_method: String,
    outer_maxiter: usize,
    outer_gtol: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inits_from_nca: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    covariate_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    input_columns: Vec<String>,
    /// Covariate name → `continuous` / `categorical`, for the `covtab.csv` columns.
    /// Absent exactly when that entry is.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    covariate_types: BTreeMap<String, String>,
    /// `[covariate_nn]` block metadata. Absent for a model with none, and not read by a
    /// build without `--features nn`.
    #[cfg(feature = "nn")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    neural_networks: Vec<crate::types::NeuralNetworkInfo>,
    /// The fitted model's data-derived bindings (#1621). Absent for a model with none.
    #[serde(default, skip_serializing_if = "DataBindingsWire::is_empty")]
    data_bindings: DataBindingsWire,
    /// The settings the fit read its data with, and a fingerprint of the
    /// population it was given (#1685). Absent for a fit that has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reader_settings: Option<crate::api::ReaderSettings>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    population_fingerprint: Option<crate::types::PopulationFingerprint>,
    /// The IOV occasion rule the fit ran under (#1783).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    iov_occasion: Option<crate::types::IovOccasionRule>,
    /// The outer optimizer's packed parameter vector at the estimate (#1815), so a
    /// post-hoc `run_covariance` / `run_sir` on the reloaded fit centres on the
    /// same point the inline step did. Absent for engines with no packed vector
    /// (SAEM / IMP / Bayes). VI writes it too, but its vector never matches its stored Ω
    /// (#1847), so a reloaded VI fit takes the fallback. Consumers use it only when
    /// it unpacks bit-for-bit to the stored estimates (`fit_packed_estimate`), so an
    /// edited fit cannot be evaluated at a stale centre.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::io::serde_nan::opt_vec"
    )]
    packed_estimate: Option<Vec<f64>>,
    /// The fitted `[mixture]` per-class Ω/Σ override values (#1765), natural
    /// scale, in `omega_override_addr` then `sigma_override_addr` order. Unlike
    /// `packed_estimate` it is written by every estimator that fits an override,
    /// SAEM included, so a reloaded mixture fit rebuilds its class matrices
    /// without the packed vector. Absent for a fit with no override.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::io::serde_nan::opt_vec"
    )]
    mixture_overrides: Option<Vec<f64>>,
    /// One descriptor per `mixture_overrides` entry (#2833): name, family, base index,
    /// class, packed row, FIX state. Absent for a fit with no override and in a bundle
    /// written before the field, which then loads empty (rows go unlabelled, nothing
    /// else reads it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    mixture_override_params: Vec<crate::types::MixtureOverrideParam>,
    /// Standard errors parallel to `mixture_overrides` (#2833), on its scales.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::io::serde_nan::opt_vec"
    )]
    se_mixture_overrides: Option<Vec<f64>>,
    /// The variational-inference result (#2058), in the layout of the YAML
    /// writer's `vi:` section. Absent for every other method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    vi: Option<ViWire>,
}

/// Wire form of [`ViResult`]: its fields in the order and under the names the
/// YAML writer's `vi:` section uses, with the per-subject posteriors gathered into
/// `eta_posterior` entries keyed by subject ID. Every float goes through
/// `serde_nan`, since the ELBO of a diverged run is `NaN`.
#[derive(Serialize, Deserialize)]
struct ViWire {
    #[serde(with = "crate::io::serde_nan::scalar")]
    neg_two_elbo: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    data_term: f64,
    #[serde(with = "crate::io::serde_nan::scalar")]
    kl_term: f64,
    family: String,
    n_iterations: usize,
    n_mc_samples: usize,
    kl: String,
    n_kl_fallback_subjects: usize,
    converged: bool,
    n_fd_subjects: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    superseded_by: Option<String>,
    #[serde(with = "crate::io::serde_nan::scalar")]
    elbo_tightness_ratio: f64,
    #[serde(with = "crate::io::serde_nan::vec")]
    elbo_trace: Vec<f64>,
    /// Empty is `ViResult::objective_trace`'s "identical to `elbo_trace`"
    /// sentinel, so the key is omitted then, as in the YAML.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        with = "crate::io::serde_nan::vec"
    )]
    objective_trace: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    eta_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    eta_posterior: Vec<ViPosteriorWire>,
}

/// One subject's variational posterior. `cov` and `kappa_means` are `None` when
/// `ViResult` has no entry for the subject (rather than an empty one), so an
/// absent key and an empty list each round-trip as themselves.
#[derive(Serialize, Deserialize)]
struct ViPosteriorWire {
    id: String,
    #[serde(with = "crate::io::serde_nan::vec")]
    mean: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov: Option<NanRows>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kappa_means: Option<NanRows>,
}

/// A row-major `Vec<Vec<f64>>` with non-finite cells as `null`, usable inside an
/// `Option` (which a `#[serde(with)]` adapter cannot reach).
#[derive(Serialize, Deserialize)]
struct NanRows(#[serde(with = "crate::io::serde_nan::vec_vec")] Vec<Vec<f64>>);

impl ViWire {
    fn from_result(v: &ViResult, r: &FitResult) -> Self {
        let eta_posterior = v
            .eta_means
            .iter()
            .enumerate()
            .map(|(i, mean)| ViPosteriorWire {
                // The YAML writer's fallback for a mean with no subject row.
                id: r
                    .subjects
                    .get(i)
                    .map_or_else(|| "unknown".to_string(), |s| s.id.clone()),
                mean: mean.clone(),
                cov: v.eta_covs.get(i).cloned().map(NanRows),
                kappa_means: v.kappa_means.get(i).cloned().map(NanRows),
            })
            .collect();
        ViWire {
            neg_two_elbo: v.neg_two_elbo,
            data_term: v.data_term,
            kl_term: v.kl_term,
            family: v.family.clone(),
            n_iterations: v.n_iterations,
            n_mc_samples: v.n_mc_samples,
            kl: v.kl.clone(),
            n_kl_fallback_subjects: v.n_kl_fallback_subjects,
            converged: v.converged,
            n_fd_subjects: v.n_fd_subjects,
            superseded_by: v.superseded_by.clone(),
            elbo_tightness_ratio: v.elbo_tightness_ratio,
            elbo_trace: v.elbo_trace.clone(),
            objective_trace: v.objective_trace.clone(),
            eta_names: if v.eta_means.is_empty() {
                Vec::new()
            } else {
                r.eta_names.clone()
            },
            eta_posterior,
        }
    }

    /// Back to a [`ViResult`]. An entry's `id` must name the subject at its
    /// position in `ebes.csv`: the posteriors are stored by position, so a bundle
    /// whose subjects were reordered or edited would otherwise attach one
    /// subject's posterior to another.
    fn into_result(self, subjects: &[SubjectResult]) -> Result<ViResult, FitrxError> {
        for (i, p) in self.eta_posterior.iter().enumerate() {
            if let Some(s) = subjects.get(i) {
                if p.id != s.id {
                    return Err(FitrxError::Corrupt(format!(
                        "vi.eta_posterior[{i}] is subject {:?}, but ebes.csv row {} is {:?}",
                        p.id,
                        i + 1,
                        s.id
                    )));
                }
            }
        }
        // `None` everywhere means `ViResult` carried no list at all; otherwise a
        // missing entry is a subject the list stopped short of. Positions are kept
        // either way, so only a list that ends early can be written back shorter.
        let rows = |pick: fn(&ViPosteriorWire) -> Option<&NanRows>| -> Vec<Vec<Vec<f64>>> {
            let n = self
                .eta_posterior
                .iter()
                .rposition(|p| pick(p).is_some())
                .map_or(0, |last| last + 1);
            self.eta_posterior[..n]
                .iter()
                .map(|p| pick(p).map(|r| r.0.clone()).unwrap_or_default())
                .collect()
        };
        let eta_covs = rows(|p| p.cov.as_ref());
        let kappa_means = rows(|p| p.kappa_means.as_ref());
        Ok(ViResult {
            neg_two_elbo: self.neg_two_elbo,
            data_term: self.data_term,
            kl_term: self.kl_term,
            n_iterations: self.n_iterations,
            converged: self.converged,
            family: self.family,
            n_mc_samples: self.n_mc_samples,
            kl: self.kl,
            n_kl_fallback_subjects: self.n_kl_fallback_subjects,
            elbo_trace: self.elbo_trace,
            objective_trace: self.objective_trace,
            eta_means: self.eta_posterior.into_iter().map(|p| p.mean).collect(),
            kappa_means,
            eta_covs,
            n_fd_subjects: self.n_fd_subjects,
            elbo_tightness_ratio: self.elbo_tightness_ratio,
            superseded_by: self.superseded_by,
        })
    }
}

/// Wire form of [`DataBindings`]: the same fields, keyed by `BTreeMap` so
/// `fit.json` is byte-deterministic (the in-memory maps are `HashMap`s).
#[derive(Serialize, Deserialize, Default)]
struct DataBindingsWire {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    levels: BTreeMap<String, LevelBinding>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    covariate_stats: BTreeMap<String, CovariateSummary>,
}

impl DataBindingsWire {
    fn is_empty(&self) -> bool {
        self.levels.is_empty() && self.covariate_stats.is_empty()
    }
}

impl From<&DataBindings> for DataBindingsWire {
    fn from(b: &DataBindings) -> Self {
        Self {
            levels: b
                .levels
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            covariate_stats: b
                .covariate_stats
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

impl From<DataBindingsWire> for DataBindings {
    fn from(w: DataBindingsWire) -> Self {
        let mut b = DataBindings::default();
        b.levels = w.levels.into_iter().collect();
        b.covariate_stats = w.covariate_stats.into_iter().collect();
        b
    }
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
    /// Per-eta SD-init flag (see `FitResult.omega_init_as_sd`).
    init_as_sd: Vec<bool>,
}

#[derive(Serialize, Deserialize)]
struct SigmaWire {
    names: Vec<String>,
    estimates: Vec<f64>,
    se: Option<Vec<f64>>,
    fixed: Vec<bool>,
    types: Vec<String>,
    /// Per-sigma SD-init flag (see `FitResult.sigma_init_as_sd`).
    init_as_sd: Vec<bool>,
    /// `block_sigma` correlations (#1100). Omitted for the common diagonal-sigma case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    residual_correlations: Vec<crate::types::ResidualCorrelation>,
    /// FIX flags parallel to `residual_correlations` (#847).
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
    /// IOV kappa variance CIs (#1705). Omitted for a fit without IOV.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ci_kappa: Option<Vec<(f64, f64)>>,
    ess: Option<f64>,
    /// Retained packed-parameter draws when `sir_keep_samples = true` was set.
    /// `None` otherwise; consumed by `simulate_with_uncertainty()`.
    resamples_packed: Option<Vec<Vec<f64>>>,
    /// The settings the draws were scored under (#1758).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    settings: Option<SirSettingsWire>,
}

/// [`SirSettings`](crate::estimation::sir::SirSettings) on the wire. The enums
/// travel as their `[fit_options]` tokens and are decoded here, so an unknown
/// token is `Corrupt` like every other enum on this wire (#1382) rather than a
/// bare JSON error.
#[derive(Serialize, Deserialize)]
struct SirSettingsWire {
    samples: usize,
    resamples: usize,
    seed: u64,
    /// A finite `df` is a JSON number; the multivariate-normal proposal (`df = ∞`),
    /// which JSON has no number for, is the `[fit_options]` token `"normal"` (#1819).
    /// Held as a raw value and decoded in `into_settings`, so `null` or an unknown
    /// token is `Corrupt` naming the field rather than a bare JSON error.
    df: serde_json::Value,
    scale: String,
    keep_samples: bool,
    /// Flattened (#426): the record is nested in memory (`SirSettings::scoring`) but
    /// flat on the wire.
    #[serde(flatten)]
    scoring: ScoringSettingsWire,
}

/// [`ScoringSettings`](crate::ScoringSettings) on the wire (#426): the top-level
/// `scoring_settings` block, and the flattened scoring half of `sir.settings`. Enums
/// travel as their `[fit_options]` tokens, as in [`SirSettingsWire`].
#[derive(Serialize, Deserialize)]
struct ScoringSettingsWire {
    inner_maxiter: usize,
    inner_tol: f64,
    inner_restarts: usize,
    inner_screen: usize,
    inner_twin_seed: bool,
    mu_referencing: bool,
    n_agq: usize,
    inner_optimizer: String,
    ebe_warm_start: bool,
    ode_reltol: f64,
    ode_abstol: f64,
    ode_max_steps: usize,
    ode_method: String,
    ode_stiff_abort_after: Option<u32>,
    ode_auto_switch: bool,
    gradient_method: String,
    /// The covariate-NN strengths (#1352). Absent for a fit with none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    nn_l2_lambda: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    nn_smooth_lambda: Option<f64>,
}

/// An enum's serde token: its `[fit_options]` spelling.
fn enum_token<T: Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        other => unreachable!("a unit enum serialises to a string token, got {other:?}"),
    }
}

/// The enum a token names; an unknown token is `Corrupt` (#1382).
fn enum_from_token<T: serde::de::DeserializeOwned>(
    block: &str,
    field: &str,
    token: String,
) -> Result<T, FitrxError> {
    serde_json::from_value(serde_json::Value::String(token.clone()))
        .map_err(|_| FitrxError::Corrupt(format!("unknown {block}.{field} {token:?}")))
}

/// `sir.settings.df` for the multivariate-normal proposal: the `[fit_options] sir_df`
/// token for `df = ∞` (#1819). `"mvn"` is also read, as the parser reads it.
const SIR_DF_NORMAL: &str = "normal";

impl From<&crate::estimation::sir::SirSettings> for SirSettingsWire {
    fn from(s: &crate::estimation::sir::SirSettings) -> Self {
        Self {
            samples: s.samples,
            resamples: s.resamples,
            seed: s.seed,
            df: if s.df == f64::INFINITY {
                serde_json::Value::String(SIR_DF_NORMAL.to_string())
            } else {
                // `From<f64>` writes a non-finite value as `null`, which the reader refuses.
                serde_json::Value::from(s.df)
            },
            scale: enum_token(&s.scale),
            keep_samples: s.keep_samples,
            scoring: ScoringSettingsWire::from(&s.scoring),
        }
    }
}

impl From<&crate::ScoringSettings> for ScoringSettingsWire {
    fn from(s: &crate::ScoringSettings) -> Self {
        Self {
            inner_maxiter: s.inner_maxiter,
            inner_tol: s.inner_tol,
            inner_restarts: s.inner_restarts,
            inner_screen: s.inner_screen,
            inner_twin_seed: s.inner_twin_seed,
            mu_referencing: s.mu_referencing,
            n_agq: s.n_agq,
            inner_optimizer: enum_token(&s.inner_optimizer),
            ebe_warm_start: s.ebe_warm_start,
            ode_reltol: s.ode_reltol,
            ode_abstol: s.ode_abstol,
            ode_max_steps: s.ode_max_steps,
            ode_method: enum_token(&s.ode_method),
            ode_stiff_abort_after: s.ode_stiff_abort_after,
            ode_auto_switch: s.ode_auto_switch,
            gradient_method: enum_token(&s.gradient_method),
            nn_l2_lambda: s.nn_l2_lambda,
            nn_smooth_lambda: s.nn_smooth_lambda,
        }
    }
}

impl SirSettingsWire {
    /// Decoded and held to the domain `[fit_options]` enforces for the same keys
    /// (#1767): `run_sir` adopts a recorded value wherever its caller left the
    /// default, so a value the parser would reject must not enter through the bundle.
    /// The domain is `SirSettings::check_domain`, the one `save_fit` writes under (#1770).
    fn into_settings(self) -> Result<crate::estimation::sir::SirSettings, FitrxError> {
        let not_a_df = |got: &dyn std::fmt::Display| {
            FitrxError::Corrupt(format!(
                "sir.settings.df must be >= 1 or \"normal\", got {got}"
            ))
        };
        let df = match &self.df {
            serde_json::Value::String(tok)
                if tok.eq_ignore_ascii_case(SIR_DF_NORMAL) || tok.eq_ignore_ascii_case("mvn") =>
            {
                f64::INFINITY
            }
            serde_json::Value::Number(n) => n.as_f64().ok_or_else(|| not_a_df(n))?,
            other => return Err(not_a_df(other)),
        };
        let settings = crate::estimation::sir::SirSettings {
            samples: self.samples,
            resamples: self.resamples,
            seed: self.seed,
            df,
            scale: enum_from_token("sir.settings", "scale", self.scale)?,
            keep_samples: self.keep_samples,
            scoring: self.scoring.into_settings("sir.settings")?,
        };
        settings
            .check_domain()
            .map_err(|e| FitrxError::Corrupt(e.in_block("sir.settings")))?;
        Ok(settings)
    }
}

impl ScoringSettingsWire {
    /// Decoded and held to the `[fit_options]` domain, as [`SirSettingsWire`] is:
    /// `run_covariance` and `run_sir` adopt a recorded value wherever the caller left the
    /// default. `block` names the key the record was read from in a `Corrupt`.
    ///
    /// The domain is `ScoringSettings::check_domain`, shared with `save_fit` and `fit()`
    /// (#1770). It refuses `gradient_method = ad` too: `ad` parses as a token but no run
    /// records it — `fit()` refuses it (`E_AD_RETIRED`) and `ScoringSettings::of_run`
    /// records a post-hoc step's retired `ad` as the `auto` it ran — and a recorded value
    /// is adopted wherever the caller left `auto`, so a hand-edited `ad` is refused here.
    fn into_settings(self, block: &str) -> Result<crate::ScoringSettings, FitrxError> {
        let settings = crate::ScoringSettings {
            inner_maxiter: self.inner_maxiter,
            inner_tol: self.inner_tol,
            inner_restarts: self.inner_restarts,
            inner_screen: self.inner_screen,
            inner_twin_seed: self.inner_twin_seed,
            mu_referencing: self.mu_referencing,
            n_agq: self.n_agq,
            inner_optimizer: enum_from_token(block, "inner_optimizer", self.inner_optimizer)?,
            ebe_warm_start: self.ebe_warm_start,
            ode_reltol: self.ode_reltol,
            ode_abstol: self.ode_abstol,
            ode_max_steps: self.ode_max_steps,
            ode_method: enum_from_token(block, "ode_method", self.ode_method)?,
            ode_stiff_abort_after: self.ode_stiff_abort_after,
            ode_auto_switch: self.ode_auto_switch,
            gradient_method: enum_from_token(block, "gradient_method", self.gradient_method)?,
            nn_l2_lambda: self.nn_l2_lambda,
            nn_smooth_lambda: self.nn_smooth_lambda,
        };
        settings
            .check_domain()
            .map_err(|e| FitrxError::Corrupt(e.in_block(block)))?;
        Ok(settings)
    }
}

#[derive(Serialize, Deserialize)]
struct IovWire {
    kappa_names: Vec<String>,
    kappa_fixed: Vec<bool>,
    se_kappa: Option<Vec<f64>>,
    #[serde(with = "crate::io::serde_nan::vec")]
    shrinkage_kappa: Vec<f64>,
    /// Per-occasion kappa shrinkage: `[occ_idx][kappa_idx]`.
    #[serde(with = "crate::io::serde_nan::vec_vec")]
    shrinkage_kappa_by_occ: Vec<Vec<f64>>,
    omega_iov: MatrixWire,
    omega_iov_param_corr: Option<MatrixWire>,
    /// Per-kappa SD-init flag.
    kappa_init_as_sd: Vec<bool>,
    /// Per-kappa `weight = <expr>` source text and the median arm weight it
    /// evaluated to (#1031). Both empty for a model with no weighted kappa.
    kappa_weights: Vec<Option<String>>,
    kappa_weight_typical: Vec<Option<f64>>,
    /// Scale of each kappa, parallel to `kappa_names`, spelled as
    /// `eta_param_info`'s `param_type` (#1643).
    kappa_param_types: Vec<String>,
    /// Each kappa's level, parallel to `kappa_names`, for a model with nested levels
    /// (#1082 PR4). Empty otherwise, and then not written.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    kappa_levels: Vec<String>,
    /// The levels whose κ̂ are bundled, one `ebes_kappa_<LEVEL>.csv` each (#1082 PR4),
    /// outermost first. Empty, and absent from the JSON, for every other model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    levels: Vec<LevelWire>,
}

/// One level's entry in [`IovWire::levels`]: the shape of its `ebes_kappa_<LEVEL>.csv`.
#[derive(Serialize, Deserialize)]
struct LevelWire {
    name: String,
    columns: Vec<String>,
    kappa_names: Vec<String>,
}

/// The discrete-rows entry and the `ebes.csv` column that sizes each subject's block.
const DISCRETE_ENTRY: &str = "predictions_discrete.csv";
const DISCRETE_HEADER: [&str; 7] = ["ID", "CMT", "TIME", "DV", "PRED", "IPRED", "IWRES"];
const N_OBS_DISCRETE: &str = "n_obs_discrete";

/// The time-to-event-rows entry and the `ebes.csv` column that sizes each subject's block
/// (#911). The header is `ID, CMT, TIME, DV` followed by the sdtab's own residual column
/// list, so the two cannot drift.
#[cfg(feature = "survival")]
const TTE_ENTRY: &str = "predictions_tte.csv";
#[cfg(feature = "survival")]
const N_OBS_TTE: &str = "n_obs_tte";
#[cfg(feature = "survival")]
const TTE_KEY_COLUMNS: [&str; 4] = ["ID", "CMT", "TIME", "DV"];
/// The optional ninth column of `predictions_tte.csv` (#1889): the record's npde.
#[cfg(feature = "survival")]
const TTE_NPDE_COLUMN: &str = "NPDE";
#[cfg(feature = "survival")]
const TTE_HEADER_LEN: usize = TTE_KEY_COLUMNS.len() + crate::io::output::TTE_RESIDUAL_COLUMNS.len();
#[cfg(feature = "survival")]
const TTE_HEADER: [&str; TTE_HEADER_LEN] = {
    let residuals = crate::io::output::TTE_RESIDUAL_COLUMNS;
    let mut h = [""; TTE_HEADER_LEN];
    let mut i = 0;
    while i < TTE_KEY_COLUMNS.len() {
        h[i] = TTE_KEY_COLUMNS[i];
        i += 1;
    }
    let mut j = 0;
    while j < residuals.len() {
        h[TTE_KEY_COLUMNS.len() + j] = residuals[j];
        j += 1;
    }
    h
};

const COVTAB_ENTRY: &str = "covtab.csv";
const COVTAB_BASE_COLUMNS: [&str; 3] = ["ID", "TIME", "EVID"];
const TRACE_ENTRY: &str = "trace.csv";
const EXTRAS_ENTRY: &str = "interface_extras.json";
const DATA_ENTRY: &str = "data.csv";
/// The entries an existing bundle's bytes can be changed in, in [`save_fit`]'s order.
const REWRITABLE_ENTRIES: [&str; 2] = [DATA_ENTRY, EXTRAS_ENTRY];

/// The bundle entry holding level `name`'s κ̂ (#1082 PR4).
fn level_kappa_entry(name: &str) -> String {
    format!("ebes_kappa_{name}.csv")
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
    LogCombined => "log_combined",
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

/// Refuse a fit whose recorded settings the reader would reject (#1770), with the reader's
/// own domain check and field naming, so a bundle that cannot load is never written: the
/// failure is reported where the bad value is, not at a later `load_fit`.
///
/// `fit()` holds every value it records to that domain, so only a `FitResult` edited after
/// the fit reaches this. The SIR seed is not checked: the writer derives `sir_seed` from
/// `sir_settings.seed` (see `build_fit_wire`).
///
/// The fields a caller can set by hand get the same treatment: the covariate table's
/// shape, each subject's `n_obs` against the prediction rows it heads, and
/// `interface_extras`, which is opaque to core but has to be the JSON its entry's name
/// promises to every other reader of the bundle. The wire's own invariants
/// are checked by `save_fit` on the wire it builds.
fn check_writable(r: &FitResult) -> Result<(), FitrxError> {
    let refuse = |block: &str, e: crate::estimation::fit_inputs::OutOfDomain| {
        FitrxError::Unwritable(e.in_block(block))
    };
    if let Some(s) = &r.scoring_settings {
        s.check_domain()
            .map_err(|e| refuse("scoring_settings", e))?;
    }
    if let Some(s) = &r.sir_settings {
        s.check_domain().map_err(|e| refuse("sir.settings", e))?;
    }
    if let Some(table) = &r.covariate_table {
        if table.kinds.len() != table.names.len() {
            return Err(FitrxError::Unwritable(format!(
                "covariate_table has {} kinds for {} names",
                table.kinds.len(),
                table.names.len()
            )));
        }
        if let Some(row) = table
            .rows
            .iter()
            .find(|w| w.values.len() != table.names.len())
        {
            return Err(FitrxError::Unwritable(format!(
                "covariate_table: a row of subject {:?} has {} values for {} names",
                row.id,
                row.values.len(),
                table.names.len()
            )));
        }
    }
    // `ebes.csv` states `n_obs` and `predictions.csv` writes `ipred.len()` rows, and the
    // reader requires the two to agree.
    if let Some(s) = r.subjects.iter().find(|s| s.n_obs != s.ipred.len()) {
        return Err(FitrxError::Unwritable(format!(
            "subject {:?} has n_obs = {} and {} predictions",
            s.id,
            s.n_obs,
            s.ipred.len()
        )));
    }
    if let Some(extras) = &r.interface_extras {
        check_interface_extras(extras)?;
    }
    Ok(())
}

fn check_interface_extras(extras: &str) -> Result<(), FitrxError> {
    serde_json::from_str::<serde::de::IgnoredAny>(extras)
        .map_err(|e| FitrxError::Unwritable(format!("interface_extras is not valid JSON: {e}")))?;
    Ok(())
}

/// Refuse data whose bytes are not the ones the fit hashed: a file edited since the fit
/// is not the data that was fitted (#2696). `name` is what the message calls the data.
/// With no recorded hash there is nothing to compare.
fn check_data_hash(data: &[u8], expected: Option<&str>, name: &str) -> Result<(), FitrxError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = crate::io::hash::sha256_bytes(data);
    if actual != expected {
        return Err(FitrxError::Unwritable(format!(
            "{name} is not the data that was fitted: its SHA-256 is {actual}, the fit \
             recorded {expected}. It has changed since the fit was produced; bundle the \
             original file, or refit."
        )));
    }
    Ok(())
}

/// [`check_data_hash`] for the model text a bundle would hold.
fn check_model_hash(r: &FitResult, text: &str) -> Result<(), FitrxError> {
    let Some(expected) = &r.model_hash else {
        return Ok(());
    };
    let actual = crate::io::hash::sha256_bytes(text.as_bytes());
    if &actual != expected {
        let name = r.model_path.as_deref().unwrap_or("the model source");
        return Err(FitrxError::Unwritable(format!(
            "the model text for {name} is not the model that was fitted: its SHA-256 is \
             {actual}, the fit recorded {expected}."
        )));
    }
    Ok(())
}

/// Read the file [`SaveFitOptions::include_data`] names, with the name its refusal uses.
fn read_included_data(opts: &SaveFitOptions) -> Result<Option<(Vec<u8>, String)>, FitrxError> {
    let Some(data_path) = &opts.include_data else {
        return Ok(None);
    };
    let bytes = std::fs::read(data_path).map_err(|e| {
        FitrxError::Io(std::io::Error::new(
            e.kind(),
            format!("opening {} for bundling: {}", data_path.display(), e),
        ))
    })?;
    Ok(Some((bytes, format!("data file {}", data_path.display()))))
}

/// The directory a temporary sibling of `target` goes in. A bare file name has an empty
/// parent, which is the working directory.
fn parent_dir(target: &Path) -> &Path {
    match target.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

/// The file a write to `path` lands in: `path` with any symlink at it followed, including
/// one whose referent does not exist yet, which `canonicalize` cannot resolve. Only the
/// last component needs following, as the OS resolves the directories on the way to it.
fn follow_symlinks(path: &Path) -> std::io::Result<PathBuf> {
    // Linux's own limit on links in one path resolution.
    const MAX_LINKS: usize = 40;
    let mut target = path.to_path_buf();
    for _ in 0..MAX_LINKS {
        match std::fs::read_link(&target) {
            // `join` keeps an absolute link as it is, and puts a relative one where the
            // OS reads it: against the directory the link is in.
            Ok(link) => target = parent_dir(&target).join(link),
            Err(_) => return Ok(target),
        }
    }
    Err(std::io::Error::other(format!(
        "{}: too many levels of symbolic links",
        path.display()
    )))
}

/// Replace the file at `path` with what `write` produces, or leave it as it was.
///
/// `write` fills a temporary file in the target's directory, which is renamed over the
/// target only once it is complete and synced, so a failure at any point leaves an
/// existing bundle intact and no partial file behind. The outcome otherwise matches
/// `File::create`: a symlink is followed, an existing file keeps its permissions, a
/// read-only one is refused, and a new one gets the umask's.
fn replace_file(
    path: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> Result<(), FitrxError> {
    let target = follow_symlinks(path)?;
    let existing = std::fs::metadata(&target).ok().map(|m| m.permissions());
    if existing.as_ref().is_some_and(|p| p.readonly()) {
        return Err(FitrxError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} is read-only", target.display()),
        )));
    }
    let mut builder = tempfile::Builder::new();
    builder.prefix(".fitrx-").suffix(".tmp");
    // A temporary file is private by default; a bundle is not.
    #[cfg(unix)]
    builder.permissions(std::os::unix::fs::PermissionsExt::from_mode(0o666));
    let mut tmp = builder.tempfile_in(parent_dir(&target))?;
    write(tmp.as_file_mut())?;
    if let Some(permissions) = existing {
        tmp.as_file().set_permissions(permissions)?;
    }
    tmp.as_file().sync_all()?;
    tmp.persist(&target).map_err(|e| FitrxError::Io(e.error))?;
    Ok(())
}

/// Write a fit bundle to `path`.
///
/// `model_source` is the verbatim text of the `.ferx` model; it is embedded so
/// a future [`load_fit`] can recompile the model and run `predict()` against
/// the loaded fit.
///
/// The bundle is built in memory and moved into place by rename, so a failed save
/// leaves whatever was at `path` untouched. Data or model text that does not match
/// the hash the fit recorded is refused with [`FitrxError::Unwritable`].
pub fn save_fit(
    result: &FitResult,
    population: &Population,
    model_source: &str,
    path: &Path,
    opts: SaveFitOptions,
) -> Result<(), FitrxError> {
    let bytes = save_fit_to_bytes(result, population, model_source, opts)?;
    replace_file(path, |file| file.write_all(&bytes))
}

/// [`save_fit`] to memory: the bytes of the bundle it would write.
pub fn save_fit_to_bytes(
    result: &FitResult,
    population: &Population,
    model_source: &str,
    opts: SaveFitOptions,
) -> Result<Vec<u8>, FitrxError> {
    let data = read_included_data(&opts)?;
    let data = data.as_ref().map(|(b, name)| (b.as_slice(), name.as_str()));
    encode_bundle(
        result,
        ObservedRows::Population(population),
        model_source,
        data,
    )
}

/// Write `result` to `path` with no [`Population`], taking what it lacks from `bundle`,
/// the bytes of the bundle it was loaded from.
///
/// `predictions.csv`'s `TIME`, `DV` and `OCC` columns have no field on [`FitResult`], so
/// they come from `bundle`, as does its `data.csv` unless
/// [`SaveFitOptions::include_data`] names another file, and the model text when `result`
/// has none. Otherwise this is [`save_fit`], with its atomic write and hash checks.
///
/// Those columns only belong to `result` if its subjects are the bundle's, so anything
/// else is refused with [`FitrxError::Unwritable`]: the subjects must have the same IDs in
/// the same order and the same `PRED`, `IPRED`, `CWRES` and `IWRES`. The values are
/// compared because IDs can repeat, and the residuals because they depend on `DV`. The
/// save is also refused when `bundle` has no `TIME` or `DV` column.
pub fn save_fit_from_bundle(
    result: &FitResult,
    bundle: &[u8],
    path: &Path,
    opts: SaveFitOptions,
) -> Result<(), FitrxError> {
    let mut archive = ZipArchive::new(Cursor::new(bundle))?;
    read_manifest(&mut archive)?;
    let wire: FitWire = read_json(&mut archive, "fit.json")?;
    let ebes_csv = read_text(&mut archive, "ebes.csv")?;
    let preds_csv = read_text(&mut archive, "predictions.csv")?;
    let subjects = parse_subjects(&ebes_csv, &preds_csv, None, wire.omega.matrix.rows)?;
    let rows = parse_recorded_rows(&preds_csv, &subjects);
    check_bundle_rows(result, &subjects, &rows)?;

    let model_source = read_text(&mut archive, "model.ferx")?;
    let included = read_included_data(&opts)?;
    let bundled = if included.is_none() && archive.index_for_name(DATA_ENTRY).is_some() {
        Some(read_bytes(&mut archive, DATA_ENTRY)?)
    } else {
        None
    };
    let data = match (&included, &bundled) {
        (Some((bytes, name)), _) => Some((bytes.as_slice(), name.as_str())),
        (None, Some(bytes)) => Some((bytes.as_slice(), "the bundled data.csv")),
        (None, None) => None,
    };
    let bytes = encode_bundle(result, ObservedRows::Recorded(&rows), &model_source, data)?;
    replace_file(path, |file| file.write_all(&bytes))
}

/// Equal value for value, `NaN` matching `NaN`: a missing cell is `NaN` on both sides.
fn same_values(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

/// Refuse `fit` unless `rows`, read from a bundle alongside `subjects`, are its own rows.
/// See [`save_fit_from_bundle`].
fn check_bundle_rows(
    fit: &FitResult,
    subjects: &[SubjectResult],
    rows: &[RecordedRows],
) -> Result<(), FitrxError> {
    const ADVICE: &str = "pass the population to `save_fit`";
    if subjects.len() != fit.subjects.len() {
        return Err(FitrxError::Unwritable(format!(
            "the bundle has {} subjects, the fit has {}; {ADVICE}",
            subjects.len(),
            fit.subjects.len()
        )));
    }
    for (i, ((subject, bundled), row)) in fit.subjects.iter().zip(subjects).zip(rows).enumerate() {
        if subject.id != bundled.id {
            return Err(FitrxError::Unwritable(format!(
                "subject {} is {:?} in the fit and {:?} in the bundle; {ADVICE}",
                i + 1,
                subject.id,
                bundled.id
            )));
        }
        for (column, ours, theirs) in [
            ("PRED", &subject.pred, &bundled.pred),
            ("IPRED", &subject.ipred, &bundled.ipred),
            ("CWRES", &subject.cwres, &bundled.cwres),
            ("IWRES", &subject.iwres, &bundled.iwres),
        ] {
            if !same_values(ours, theirs) {
                return Err(FitrxError::Unwritable(format!(
                    "subject {} ({:?}) has a different {column} in the fit and the bundle; \
                     {ADVICE}",
                    i + 1,
                    subject.id
                )));
            }
        }
        let n = bundled.ipred.len();
        if row.time.len() != n || row.dv.len() != n {
            return Err(FitrxError::Unwritable(format!(
                "the bundle's predictions.csv has {} TIME and {} DV values for subject {:?}, \
                 which has {n} predictions; {ADVICE}",
                row.time.len(),
                row.dv.len(),
                subject.id
            )));
        }
    }
    Ok(())
}

/// Serialize a fit. `data` is the bytes for `data.csv` with the name a refusal uses.
fn encode_bundle(
    result: &FitResult,
    rows: ObservedRows<'_>,
    model_source: &str,
    data: Option<(&[u8], &str)>,
) -> Result<Vec<u8>, FitrxError> {
    check_writable(result)?;
    let effective_source = result.model_text.as_deref().unwrap_or(model_source);
    check_model_hash(result, effective_source)?;
    if let Some((bytes, name)) = data {
        check_data_hash(bytes, result.data_hash.as_deref(), name)?;
    }
    let wire = build_fit_wire(result);
    // The reader's own shape checks, so a hand-built result with a short parallel vector
    // is refused here rather than saved as a bundle `load_fit` calls corrupt.
    validate_parallel_lengths(&wire).map_err(|e| match e {
        FitrxError::Corrupt(msg) => FitrxError::Unwritable(msg),
        other => other,
    })?;
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let zopts = bundle_entry_options();

    let mut entries: Vec<String> = Vec::new();
    entries.push("manifest.json".into());

    // --- fit.json ----------------------------------------------------------
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

    // --- ebes_kappa_<LEVEL>.csv (nested levels, #1082 PR4) -----------------
    if result.omega_iov.is_some() {
        for level in &result.ebe_kappas_by_level {
            let entry = level_kappa_entry(&level.level);
            zip.start_file(entry.as_str(), zopts)?;
            write_level_kappa_csv(&mut zip, result, level)?;
            entries.push(entry);
        }
    }

    // --- conddist.csv (only when the SAEM conditional-distribution pass ran) ---
    if let Some(cd) = &result.cond_dist {
        zip.start_file("conddist.csv", zopts)?;
        write_conddist_csv(&mut zip, result, cd)?;
        entries.push("conddist.csv".into());
    }

    // --- predictions.csv ---------------------------------------------------
    zip.start_file("predictions.csv", zopts)?;
    write_predictions_csv(&mut zip, result, &rows)?;
    entries.push("predictions.csv".into());

    // --- predictions_discrete.csv (only when the fit has discrete rows) ----
    if has_discrete_rows(result) {
        zip.start_file(DISCRETE_ENTRY, zopts)?;
        write_predictions_discrete_csv(&mut zip, result)?;
        entries.push(DISCRETE_ENTRY.into());
    }

    // --- predictions_tte.csv (only when the fit has time-to-event rows, #911) ---
    #[cfg(feature = "survival")]
    if has_tte_rows(result) {
        zip.start_file(TTE_ENTRY, zopts)?;
        write_predictions_tte_csv(&mut zip, result)?;
        entries.push(TTE_ENTRY.into());
    }

    // --- covtab.csv (only with a `[covariates]` block) ---------------------
    if let Some(table) = &result.covariate_table {
        zip.start_file(COVTAB_ENTRY, zopts)?;
        write_covtab_csv(&mut zip, table)?;
        entries.push(COVTAB_ENTRY.into());
    }

    // --- trace.csv (only with `optimizer_trace = true`) --------------------
    if let Some(trace) = &result.trace_csv {
        zip.start_file(TRACE_ENTRY, zopts)?;
        zip.write_all(trace.as_bytes())?;
        entries.push(TRACE_ENTRY.into());
    }

    // --- model.ferx --------------------------------------------------------
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
    if let Some((bytes, _)) = data {
        zip.start_file(DATA_ENTRY, zopts)?;
        zip.write_all(bytes)?;
        entries.push(DATA_ENTRY.into());
    }

    // --- interface_extras.json (only when an interface set it) -------------
    if let Some(extras) = &result.interface_extras {
        zip.start_file(EXTRAS_ENTRY, zopts)?;
        zip.write_all(extras.as_bytes())?;
        entries.push(EXTRAS_ENTRY.into());
    }

    // --- manifest.json (written last so it can list every entry) ----------
    let manifest = Manifest {
        format_version: FORMAT_VERSION.into(),
        ferx_version: result.ferx_version.clone(),
        model_name: result.model_name.clone(),
        created_at: iso8601_now(),
        entries,
    };
    write_manifest(&mut zip, &manifest)?;
    Ok(zip.finish()?.into_inner())
}

fn bundle_entry_options() -> SimpleFileOptions {
    SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated)
}

fn write_manifest<W: Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    manifest: &Manifest,
) -> Result<(), FitrxError> {
    zip.start_file("manifest.json", bundle_entry_options())?;
    zip.write_all(&serde_json::to_vec_pretty(manifest)?)?;
    zip.write_all(b"\n")?;
    Ok(())
}

/// Add `data` to the bundle `bundle` as `data.csv`, replacing any it has, and return the
/// new bundle's bytes.
///
/// The bytes are checked against the `data_hash` in the bundle's `fit.json` first, and a
/// mismatch is refused with [`FitrxError::Unwritable`], calling them `name` (the file they
/// came from, say). Every other entry is carried over unchanged.
pub fn bundle_with_data(bundle: &[u8], data: &[u8], name: &str) -> Result<Vec<u8>, FitrxError> {
    let mut archive = ZipArchive::new(Cursor::new(bundle))?;
    read_manifest(&mut archive)?;
    let wire: FitWire = read_json(&mut archive, "fit.json")?;
    check_data_hash(data, wire.data_hash.as_deref(), name)?;
    rewrite_bundle(bundle, DATA_ENTRY, Some(data))
}

/// Set `interface_extras.json` on the bundle `bundle`, or remove it with `None`, and
/// return the new bundle's bytes. Text that is not well-formed JSON is refused, as
/// [`save_fit`] refuses it. Every other entry is carried over unchanged.
pub fn bundle_with_interface_extras(
    bundle: &[u8],
    extras: Option<&str>,
) -> Result<Vec<u8>, FitrxError> {
    if let Some(extras) = extras {
        check_interface_extras(extras)?;
    }
    rewrite_bundle(bundle, EXTRAS_ENTRY, extras.map(str::as_bytes))
}

/// Copy `bundle` with the entry `name` (one of [`REWRITABLE_ENTRIES`]) set to `content`,
/// or gone when `None`. Those entries and the manifest are written last, in [`save_fit`]'s
/// order, so an edited bundle has the layout of one saved that way; the rest are copied
/// still compressed. The manifest keeps its `created_at` and lists the new entries.
fn rewrite_bundle(
    bundle: &[u8],
    name: &str,
    content: Option<&[u8]>,
) -> Result<Vec<u8>, FitrxError> {
    let mut archive = ZipArchive::new(Cursor::new(bundle))?;
    let mut manifest = read_manifest(&mut archive)?;
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let mut entries = vec!["manifest.json".to_string()];
    for i in 0..archive.len() {
        let file = archive.by_index_raw(i)?;
        let entry = file.name()?.to_string();
        if entry != "manifest.json" && !REWRITABLE_ENTRIES.contains(&entry.as_str()) {
            zip.raw_copy_file(file)?;
            entries.push(entry);
        }
    }
    for entry in REWRITABLE_ENTRIES {
        if entry == name {
            if let Some(content) = content {
                zip.start_file(entry, bundle_entry_options())?;
                zip.write_all(content)?;
                entries.push(entry.to_string());
            }
        } else if let Some(i) = archive.index_for_name(entry) {
            zip.raw_copy_file(archive.by_index_raw(i)?)?;
            entries.push(entry.to_string());
        }
    }
    manifest.entries = entries;
    write_manifest(&mut zip, &manifest)?;
    Ok(zip.finish()?.into_inner())
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
        ofv_data: r.ofv_data,
        ofv_prior: r.ofv_prior,
        prior_summary: r.prior_summary.clone(),
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
        ode_linear_exact: r.ode_linear_exact,
        uses_sde: r.uses_sde,
        gradient_method_inner: r.gradient_method_inner.clone(),
        gradient_method_outer: r.gradient_method_outer.clone(),
        nlopt_missing_algorithms: r.nlopt_missing_algorithms.clone(),
        covariance_status: covariance_status_to_str(&r.covariance_status).into(),
        // `CovarianceMethod::label` rather than a second `str_enum_map!` table:
        // the bundle stores the same `r`/`s`/`rsr` token the model file writes and
        // the YAML reports, so there is one spelling to keep in step (#1382).
        covariance_method: r.covariance_method.map(|m| m.label().to_string()),
        covariance_n_evals_estimated: r.covariance_n_evals_estimated,
        trace_path: r.trace_path.clone(),
        ebe_convergence_warnings: r.ebe_convergence_warnings,
        max_unconverged_subjects: r.max_unconverged_subjects,
        total_ebe_fallbacks: r.total_ebe_fallbacks,
        total_ebe_nm_polished: r.total_ebe_nm_polished,
        warnings: r.warnings.clone(),
        warning_details: r
            .warnings_structured
            .iter()
            .filter(|e| e.details.is_some())
            .filter_map(|e| serde_json::to_value(e).ok())
            .collect(),
        saem_mu_ref_m_step_evals_saved: r.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: r.saem_n_subjects_hmc,
        saem_mh_accept_tail: r.saem_mh_accept_tail,
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
        cov_fixed: r.cov_fixed.clone(),
        bic_inputs: r.bic_inputs,
        profiled_blocks: r.profiled_blocks.clone(),
        profiled_levels_at_bound: r.profiled_levels_at_bound.clone(),
        held_at_bound: r.held_at_bound.clone(),
        sir: if r.sir_ci_theta.is_some() || r.sir_ess.is_some() || r.sir_resamples_packed.is_some()
        {
            Some(SirWire {
                ci_theta: r.sir_ci_theta.clone(),
                ci_omega: r.sir_ci_omega.clone(),
                ci_sigma: r.sir_ci_sigma.clone(),
                ci_kappa: r.sir_ci_kappa.clone(),
                ess: r.sir_ess,
                resamples_packed: r.sir_resamples_packed.clone(),
                settings: r.sir_settings.as_ref().map(SirSettingsWire::from),
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
            kappa_param_types: r
                .kappa_param_types
                .iter()
                .map(|&t| eta_param_type_to_str(t).into())
                .collect(),
            kappa_levels: r.kappa_levels.clone(),
            levels: r
                .ebe_kappas_by_level
                .iter()
                .map(|l| LevelWire {
                    name: l.level.clone(),
                    columns: l.columns.clone(),
                    kappa_names: l.kappa_names.clone(),
                })
                .collect(),
        }),
        bayes: r.bayes.clone(),
        importance_sampling: r.importance_sampling.clone(),
        impmap_trace: r.impmap_trace.clone(),
        exclusions: r.exclusions.clone(),
        stalled_at_init: crate::model_selection::stalled_at_init(r),
        estimate_near_boundary: crate::model_selection::estimate_near_boundary(r),
        max_abs_correlation: crate::model_selection::max_abs_correlation(r),
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
        environment: r.environment.clone(),
        model_path: r.model_path.clone(),
        data_path: r.data_path.clone(),
        model_hash: r.model_hash.clone(),
        data_hash: r.data_hash.clone(),
        model_text: r.model_text.clone(),
        theta_init: r.theta_init.clone(),
        omega_init: MatrixWire::from(&r.omega_init),
        left_init: r.left_init,
        stationary_start: r.stationary_start,
        omega_is_diagonal: r.omega_is_diagonal,
        kappa_is_diagonal: r.kappa_is_diagonal,
        sigma_init: r.sigma_init.clone(),
        obs_time_range: r.obs_time_range,
        final_gradient: r.final_gradient.clone(),
        final_gradient_source: r.final_gradient_source.clone(),
        newton_decrement: r.newton_decrement,
        optimizer_stop: r.optimizer_stop.clone(),
        eval_budget_spent: r.eval_budget_spent,
        evaluation_only: r.evaluation_only,
        optimizer: r.optimizer.clone(),
        n_starts: r.n_starts,
        multi_start_seed: r.multi_start_seed,
        starts: r.starts.clone(),
        saem_seed: r.saem_seed,
        // `sir_settings.seed` is canonical when SIR ran (#1770): it is the seed the draws
        // used and the one `run_sir` repeats, and the field a caller edits to change a
        // re-run. `sir_seed` is derived from it, so the pair the reader checks agrees.
        sir_seed: r.sir_settings.as_ref().map_or(r.sir_seed, |s| Some(s.seed)),
        scoring_settings: r.scoring_settings.as_ref().map(ScoringSettingsWire::from),
        imp_seed: r.imp_seed,
        npde_seed: r.npde_seed,
        bloq_method: r.bloq_method.clone(),
        outer_maxiter: r.outer_maxiter,
        outer_gtol: r.outer_gtol,
        inits_from_nca: r.inits_from_nca.clone(),
        covariate_names: r.covariate_names.clone(),
        input_columns: r.input_columns.clone(),
        covariate_types: r
            .covariate_table
            .iter()
            .flat_map(|t| t.names.iter().zip(&t.kinds))
            .map(|(name, kind)| (name.clone(), kind.label().to_string()))
            .collect(),
        #[cfg(feature = "nn")]
        neural_networks: r.neural_networks.clone(),
        data_bindings: DataBindingsWire::from(&r.data_bindings),
        reader_settings: r.reader_settings.clone(),
        population_fingerprint: r.population_fingerprint.clone(),
        iov_occasion: r.iov_occasion.clone(),
        packed_estimate: r.packed_estimate.clone(),
        mixture_overrides: r.mixture_overrides.clone(),
        mixture_override_params: r.mixture_override_params.clone(),
        se_mixture_overrides: r.se_mixture_overrides.clone(),
        vi: r.vi.as_ref().map(|v| ViWire::from_result(v, r)),
    }
}

fn has_discrete_rows(r: &FitResult) -> bool {
    r.subjects.iter().any(|s| !s.discrete_rows.is_empty())
}

#[cfg(feature = "survival")]
fn has_tte_rows(r: &FitResult) -> bool {
    r.subjects.iter().any(|s| !s.tte_rows.is_empty())
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
    // A bundle with discrete rows says how many are each subject's, as `n_obs` does for
    // `predictions.csv`: an ID alone cannot place a row when two subjects share it.
    let any_discrete = has_discrete_rows(r);
    if any_discrete {
        header.push(',');
        header.push_str(N_OBS_DISCRETE);
    }
    // The same for time-to-event rows (#911), so a bundle without them keeps its old bytes.
    #[cfg(feature = "survival")]
    let any_tte = has_tte_rows(r);
    #[cfg(feature = "survival")]
    if any_tte {
        header.push(',');
        header.push_str(N_OBS_TTE);
    }
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
        if any_discrete {
            row.push(',');
            row.push_str(&s.discrete_rows.len().to_string());
        }
        #[cfg(feature = "survival")]
        if any_tte {
            row.push(',');
            row.push_str(&s.tte_rows.len().to_string());
        }
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

/// One level's κ̂ (#1082 PR4): `ID,GROUP,<label columns…>,<kappas…>`, one row per group of
/// the level per subject. `GROUP` is the group's 1-based position in its subject's list
/// (first-record order); the label columns name it, outermost level first. A label is
/// written in full (shortest round-trip form), since it identifies the group and must
/// reload to the same bits; a κ̂ exactly too, as every bundle double is (`fmt_f64`, #1847),
/// NaN as empty.
fn write_level_kappa_csv<W: Write>(
    w: &mut W,
    r: &FitResult,
    level: &crate::types::LevelKappaEbes,
) -> Result<(), FitrxError> {
    let mut header = String::from("ID,GROUP");
    for c in level.columns.iter().chain(&level.kappa_names) {
        header.push(',');
        header.push_str(c);
    }
    writeln!(w, "{}", header)?;
    for (s, groups) in r.subjects.iter().zip(&level.groups) {
        for (gi, g) in groups.iter().enumerate() {
            let mut row = csv_escape(&s.id);
            row.push(',');
            row.push_str(&(gi + 1).to_string());
            for &v in &g.labels {
                row.push(',');
                row.push_str(&v.to_string());
            }
            for &v in &g.kappa {
                row.push(',');
                row.push_str(&fmt_f64(v));
            }
            writeln!(w, "{}", row)?;
        }
    }
    Ok(())
}

/// Parse one `ebes_kappa_<LEVEL>.csv` back into its groups, per subject in
/// `subject_ids` order. Rows are placed by `GROUP`, which must run 1..n per subject.
fn parse_level_kappa_csv(
    csv: &str,
    level: LevelWire,
    subject_ids: &[String],
) -> Result<crate::types::LevelKappaEbes, FitrxError> {
    let entry = level_kappa_entry(&level.name);
    let corrupt = |msg: String| FitrxError::Corrupt(format!("{entry}: {msg}"));
    let mut lines = csv.lines();
    let header = lines.next().ok_or_else(|| corrupt("empty".into()))?;
    let cols: Vec<&str> = header.split(',').collect();
    let n_labels = level.columns.len();
    let n_kappa = level.kappa_names.len();
    let expected: Vec<&str> = ["ID", "GROUP"]
        .into_iter()
        .chain(level.columns.iter().map(String::as_str))
        .chain(level.kappa_names.iter().map(String::as_str))
        .collect();
    if cols != expected {
        return Err(corrupt(format!(
            "header {header:?} does not match fit.json's level (expected {:?})",
            expected.join(",")
        )));
    }
    let by_id: HashMap<&str, usize> = subject_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut staged: Vec<Vec<(usize, crate::types::LevelGroupEbe)>> =
        vec![Vec::new(); subject_ids.len()];
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_row(line);
        if fields.len() != cols.len() {
            return Err(corrupt(format!(
                "row {} has {} fields, expected {}",
                i + 1,
                fields.len(),
                cols.len()
            )));
        }
        let idx = *by_id
            .get(fields[0].as_str())
            .ok_or_else(|| corrupt(format!("unknown ID {:?}", fields[0])))?;
        let group = fields[1]
            .parse::<usize>()
            .ok()
            .filter(|&g| g > 0)
            .ok_or_else(|| corrupt(format!("row {}: bad GROUP {:?}", i + 1, fields[1])))?;
        let num = |f: &String| {
            f.parse::<f64>()
                .map_err(|_| corrupt(format!("row {}: bad number {f:?}", i + 1)))
        };
        let labels = fields[2..2 + n_labels]
            .iter()
            .map(num)
            .collect::<Result<_, _>>()?;
        // An empty κ̂ cell is a NaN estimate, as the writer prints it (review r1 #3).
        let kappa = fields[2 + n_labels..2 + n_labels + n_kappa]
            .iter()
            .map(|f| if f.is_empty() { Ok(f64::NAN) } else { num(f) })
            .collect::<Result<_, _>>()?;
        staged[idx].push((group, crate::types::LevelGroupEbe { labels, kappa }));
    }
    let mut groups = Vec::with_capacity(staged.len());
    for (idx, mut rows) in staged.into_iter().enumerate() {
        rows.sort_by_key(|(g, _)| *g);
        if rows.iter().enumerate().any(|(n, (g, _))| *g != n + 1) {
            return Err(corrupt(format!(
                "subject {:?}: GROUP must run 1..n without gaps or repeats",
                subject_ids[idx]
            )));
        }
        groups.push(rows.into_iter().map(|(_, g)| g).collect());
    }
    Ok(crate::types::LevelKappaEbes {
        level: level.name,
        columns: level.columns,
        kappa_names: level.kappa_names,
        groups,
    })
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
    rows: &ObservedRows<'_>,
) -> Result<(), FitrxError> {
    let any_cens = r.subjects.iter().any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = rows.any_occ();
    // NPDE/NPD are only present when the fit ran with npde_nsim > 0.
    let any_npde = r
        .subjects
        .iter()
        .any(|s| !s.npde.is_empty() || !s.npd.is_empty());

    let mut header = String::from("ID,TIME,DV,PRED,IPRED,CWRES,IWRES");
    if any_npde {
        header.push_str(",NPDE,NPD,EPRED,PWRES");
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
        for j in 0..sr.ipred.len() {
            let mut row = csv_escape(&sr.id);
            row.push(',');
            row.push_str(&fmt_f64(rows.time(si, j)));
            row.push(',');
            row.push_str(&fmt_f64(rows.dv(si, j)));
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
                row.push(',');
                row.push_str(&fmt_f64(sr.epred.get(j).copied().unwrap_or(f64::NAN)));
                row.push(',');
                row.push_str(&fmt_f64(sr.pwres.get(j).copied().unwrap_or(f64::NAN)));
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
                row.push_str(&rows.occ(si, j).to_string());
            }
            writeln!(w, "{}", row)?;
        }
    }
    Ok(())
}

/// `predictions_discrete.csv`: one block of rows per subject with discrete records, in
/// `ebes.csv` order, sized by that subject's `n_obs_discrete`.
fn write_predictions_discrete_csv<W: Write>(w: &mut W, r: &FitResult) -> Result<(), FitrxError> {
    writeln!(w, "{}", DISCRETE_HEADER.join(","))?;
    for s in &r.subjects {
        for d in &s.discrete_rows {
            writeln!(
                w,
                "{},{},{},{},{},{},{}",
                csv_escape(&s.id),
                d.cmt,
                fmt_f64(d.time),
                fmt_f64(d.dv),
                fmt_f64(d.pred),
                fmt_f64(d.ipred),
                fmt_f64(d.iwres)
            )?;
        }
    }
    Ok(())
}

/// `predictions_tte.csv` (#911): one block of rows per subject with time-to-event records,
/// in `ebes.csv` order, sized by that subject's `n_obs_tte`. The residual cells are written
/// in `TTE_HEADER`'s order, looked up by name, so a reordered column list cannot shift a
/// value under the wrong header.
#[cfg(feature = "survival")]
fn write_predictions_tte_csv<W: Write>(w: &mut W, r: &FitResult) -> Result<(), FitrxError> {
    // The npde column is written only when some record has a score (#1889), so a fit
    // without one writes the entry it always did, byte for byte.
    let with_npde = crate::types::any_tte_npde(&r.subjects);
    let mut header = TTE_HEADER.join(",");
    if with_npde {
        header.push(',');
        header.push_str(TTE_NPDE_COLUMN);
    }
    writeln!(w, "{header}")?;
    for s in &r.subjects {
        for t in &s.tte_rows {
            let mut line = format!(
                "{},{},{},{}",
                csv_escape(&s.id),
                t.cmt,
                fmt_f64(t.time),
                fmt_f64(t.dv)
            );
            for name in crate::io::output::TTE_RESIDUAL_COLUMNS {
                line.push(',');
                line.push_str(&fmt_f64(tte_residual(t, name)));
            }
            if with_npde {
                line.push(',');
                line.push_str(&fmt_f64(t.npde));
            }
            writeln!(w, "{line}")?;
        }
    }
    Ok(())
}

/// The residual of `t` under sdtab column `name`, one of `TTE_RESIDUAL_COLUMNS`, for the
/// writer. The bundle shares the sdtab's column names, not its field mapping:
/// `append_tte_rows` has its own, and T1 pins this one against literal entry text, since a
/// reader built on the same mistake would round-trip it exactly.
#[cfg(feature = "survival")]
fn tte_residual(t: &crate::types::TteObsDiagnostic, name: &str) -> f64 {
    match name {
        "COXSNELL" => t.coxsnell,
        "MARTINGALE" => t.martingale,
        "DEVIANCE" => t.deviance,
        "PIT" => t.pit,
        other => unreachable!("no TteObsDiagnostic field for sdtab column {other:?}"),
    }
}

/// [`tte_residual`]'s field by `&mut`, for the reader.
#[cfg(feature = "survival")]
fn tte_residual_mut<'a>(t: &'a mut crate::types::TteObsDiagnostic, name: &str) -> &'a mut f64 {
    match name {
        "COXSNELL" => &mut t.coxsnell,
        "MARTINGALE" => &mut t.martingale,
        "DEVIANCE" => &mut t.deviance,
        "PIT" => &mut t.pit,
        other => unreachable!("no TteObsDiagnostic field for sdtab column {other:?}"),
    }
}

/// `covtab.csv`: `ID, TIME, EVID, <covariates...>`, one row per input record. Exact doubles
/// like every bundle CSV, where the `-covtab.csv` output beside a run rounds for reading.
fn write_covtab_csv<W: Write>(w: &mut W, t: &CovariateTable) -> Result<(), FitrxError> {
    let mut header = COVTAB_BASE_COLUMNS.join(",");
    for name in &t.names {
        header.push(',');
        header.push_str(&csv_escape(name));
    }
    writeln!(w, "{header}")?;
    for row in &t.rows {
        let mut line = csv_escape(&row.id);
        line.push(',');
        line.push_str(&fmt_f64(row.time));
        line.push(',');
        line.push_str(&row.evid.to_string());
        for &v in &row.values {
            line.push(',');
            line.push_str(&fmt_f64(v));
        }
        writeln!(w, "{line}")?;
    }
    Ok(())
}

/// The bundle's CSV float: the shortest text that reads back as the same `f64`, so a
/// reloaded fit holds the bits the saved one held (#1631, #1847). `NaN` is an empty cell
/// and `±inf` is `inf` / `-inf`, as before. `{:?}` rather than `{}` so a value far from
/// one takes an exponent (`1e-7`, `1e20`) instead of a run of zeros; both round-trip.
fn fmt_f64(v: f64) -> String {
    if v.is_nan() {
        String::new()
    } else {
        format!("{v:?}")
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
    load_bundle(File::open(path)?, true)
}

/// [`load_fit`] from a bundle's bytes.
pub fn load_fit_from_bytes(bundle: &[u8]) -> Result<LoadedFit, FitrxError> {
    load_bundle(Cursor::new(bundle), true)
}

/// [`load_fit`] without re-reading a bundled `data.csv`: the population is
/// always `None`.
///
/// For a caller that needs only the fit and the model source, such as the DCM
/// exporter (`ferx export-dcm`). Re-reading the data parses the bundled model to
/// learn its column mapping; a `[covariate_nn]` model does not parse in a build
/// without `nn`, and the fallback read without the mapping can then fail on a
/// renamed column — refusing an export that never needed the data.
pub fn load_fit_without_population(path: &Path) -> Result<LoadedFit, FitrxError> {
    load_bundle(File::open(path)?, false)
}

/// [`load_fit_without_population`] from a bundle's bytes.
pub fn load_fit_from_bytes_without_population(bundle: &[u8]) -> Result<LoadedFit, FitrxError> {
    load_bundle(Cursor::new(bundle), false)
}

/// The bundle's manifest, once its `format_version` is known to be this one.
fn read_manifest<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
) -> Result<Manifest, FitrxError> {
    let manifest: Manifest = read_json(archive, "manifest.json")?;
    if manifest.format_version != FORMAT_VERSION {
        return Err(FitrxError::UnsupportedVersion(manifest.format_version));
    }
    Ok(manifest)
}

fn load_bundle<R: Read + std::io::Seek>(
    reader: R,
    read_population: bool,
) -> Result<LoadedFit, FitrxError> {
    let mut archive = ZipArchive::new(reader)?;
    let manifest = read_manifest(&mut archive)?;

    let wire: FitWire = read_json(&mut archive, "fit.json")?;
    let ebes_csv = read_text(&mut archive, "ebes.csv")?;
    let ebes_kappa_csv = read_optional_text(&mut archive, "ebes_kappa.csv")?;
    let conddist_csv = read_optional_text(&mut archive, "conddist.csv")?;
    let preds_csv = read_text(&mut archive, "predictions.csv")?;
    let model_source = read_text(&mut archive, "model.ferx")?;
    let data_csv = if read_population && archive.index_for_name(DATA_ENTRY).is_some() {
        Some(read_bytes(&mut archive, DATA_ENTRY)?)
    } else {
        None
    };

    let n_eta = wire.omega.matrix.rows;
    let mut subjects = parse_subjects(&ebes_csv, &preds_csv, ebes_kappa_csv.as_deref(), n_eta)?;
    let discrete_csv = read_optional_text(&mut archive, DISCRETE_ENTRY)?;
    attach_discrete_rows(&mut subjects, &ebes_csv, discrete_csv.as_deref())?;
    // Without `survival` the rows have no type to land in: the entry and the `n_obs_tte`
    // column are ignored, as a reader older than #911 ignores them.
    #[cfg(feature = "survival")]
    {
        let tte_csv = read_optional_text(&mut archive, TTE_ENTRY)?;
        attach_tte_rows(&mut subjects, &ebes_csv, tte_csv.as_deref())?;
    }
    let ebe_kappas = if let Some(csv) = ebes_kappa_csv.as_deref() {
        parse_ebe_kappas(
            csv,
            &subjects.iter().map(|s| s.id.clone()).collect::<Vec<_>>(),
        )?
    } else {
        Vec::new()
    };

    // Nested levels (#1082 PR4): one `ebes_kappa_<LEVEL>.csv` per level fit.json names.
    let mut wire = wire;
    let level_wires = wire
        .iov
        .as_mut()
        .map(|iov| std::mem::take(&mut iov.levels))
        .unwrap_or_default();
    let subject_ids: Vec<String> = subjects.iter().map(|s| s.id.clone()).collect();
    let mut by_level = Vec::with_capacity(level_wires.len());
    for level in level_wires {
        let entry = level_kappa_entry(&level.name);
        if archive.index_for_name(&entry).is_none() {
            return Err(FitrxError::Corrupt(format!(
                "fit.json names level `{}`, but the bundle has no {entry}",
                level.name
            )));
        }
        let csv = read_text(&mut archive, &entry)?;
        by_level.push(parse_level_kappa_csv(&csv, level, &subject_ids)?);
    }

    let verdicts = FitVerdicts {
        stalled_at_init: wire.stalled_at_init,
        estimate_near_boundary: wire.estimate_near_boundary,
        max_abs_correlation: wire.max_abs_correlation,
    };
    let covariate_types = std::mem::take(&mut wire.covariate_types);
    let mut fit = wire_to_fit_result(wire, subjects, ebe_kappas)?;
    fit.ebe_kappas_by_level = by_level;
    fit.model_text = Some(model_source.clone());
    fit.covariate_table = match read_optional_text(&mut archive, COVTAB_ENTRY)? {
        Some(csv) => Some(parse_covtab_csv(&csv, &covariate_types)?),
        None if covariate_types.is_empty() => None,
        None => {
            return Err(FitrxError::Corrupt(format!(
                "fit.json has covariate_types, but the bundle has no {COVTAB_ENTRY}"
            )))
        }
    };
    fit.trace_csv = read_optional_text(&mut archive, TRACE_ENTRY)?;
    fit.interface_extras = read_optional_text(&mut archive, EXTRAS_ENTRY)?;
    if let Some(csv) = conddist_csv.as_deref() {
        fit.cond_dist = Some(parse_conddist_csv(csv, &fit)?);
    }

    // data.csv is optional — re-parse only when present.
    let population = match data_csv {
        Some(bytes) => {
            let (population, note) = read_bundled_population(&bytes, &model_source, &fit)?;
            if let Some(note) = note {
                fit.warnings.push(note);
                crate::api::rebuild_warnings_structured(&mut fit);
            }
            population
        }
        None => None,
    };

    Ok(LoadedFit {
        fit,
        model_source,
        population,
        manifest,
        verdicts,
    })
}

/// `predictions.csv`'s data columns, per subject. `parse_subjects` has checked the rows
/// against `subjects`. A column the file lacks leaves its vectors empty, which only
/// [`save_fit_from_bundle`] minds: a bundle that loads without it still loads.
fn parse_recorded_rows(preds_csv: &str, subjects: &[SubjectResult]) -> Vec<RecordedRows> {
    let mut lines = preds_csv.lines();
    let header = parse_csv_row(lines.next().unwrap_or_default());
    let column = |name: &str| header.iter().position(|c| c == name);
    let (time_i, dv_i, occ_i) = (column("TIME"), column("DV"), column("OCC"));
    let number = |fields: &[String], i: usize| {
        let cell = fields.get(i).map_or("", String::as_str);
        parse_cell(cell).unwrap_or(f64::NAN)
    };
    let mut rows = lines.filter(|l| !l.trim().is_empty()).map(parse_csv_row);
    subjects
        .iter()
        .map(|s| {
            let mut recorded = RecordedRows::default();
            for fields in rows.by_ref().take(s.n_obs) {
                if let Some(i) = time_i {
                    recorded.time.push(number(&fields, i));
                }
                if let Some(i) = dv_i {
                    recorded.dv.push(number(&fields, i));
                }
                if let Some(i) = occ_i {
                    recorded
                        .occ
                        .push(fields.get(i).and_then(|c| c.parse().ok()).unwrap_or(0));
                }
            }
            recorded
        })
        .collect()
}

/// Re-read a bundle's `data.csv` as the population the fit was given.
///
/// A fit that records its reader settings and a population fingerprint (#1685) is
/// read with those settings — its `[data_selection]` and any row filter its caller
/// added included — given its level index columns from the fit's `data_bindings`,
/// and checked against the fingerprint. When that read cannot reproduce the
/// fingerprinted population, the bundle still loads: the population is `None` and
/// the returned note says why (#1685 review r1 #2). The θ, Ω and σ do not depend on
/// how this version reads a CSV, and `run_sir` / `run_covariance` refuse a wrong
/// population where it is used.
///
/// A fit made in memory (`prepare_run` + [`fit`](crate::fit)) records a fingerprint
/// but no reader settings (#1941). Its `data.csv` is read with the settings the
/// model file states ([`ReaderSettings::from_parsed`](crate::ReaderSettings::from_parsed):
/// `[data]` renames, `[covariates]`, `iov_column`, `[data_selection]`) and checked
/// against the fingerprint the same way, so an IOV fit reloads with its occasions —
/// read routing-only, it came back with none and its occasion column as a
/// covariate. A caller that built the population some other way gets `None` and the
/// note, not a population known to be wrong.
///
/// A fit with no current fingerprint is read with the model's `[data]` renames and
/// endpoint routing only, unverified, since that read does not claim to reproduce the
/// fit's.
fn read_bundled_population(
    bytes: &[u8],
    model_source: &str,
    fit: &FitResult,
) -> Result<(Option<Population>, Option<String>), FitrxError> {
    let tmp = tempfile::NamedTempFile::new()?;
    std::fs::write(tmp.path(), bytes)?;
    let data_path = tmp.path().to_str().ok_or_else(|| {
        FitrxError::Corrupt("data.csv: the temporary path is not UTF-8".to_string())
    })?;
    let parsed = crate::parser::model_parser::parse_full_model(model_source);
    let fingerprint = fit
        .population_fingerprint
        .as_ref()
        .filter(|f| f.is_current());
    match (parsed, &fit.reader_settings, fingerprint) {
        (Ok(mut parsed), recorded, Some(fingerprint)) => {
            // A fit made in memory (`prepare_run` + `fit`) records no settings but
            // does record a fingerprint (#1941): read it the way the model file reads
            // its data — `[covariates]`, `iov_column`, `[data_selection]` included —
            // since the routing-only read gave an IOV fit no occasions.
            let settings = match recorded {
                Some(settings) => settings.clone(),
                None => crate::api::ReaderSettings::from_parsed(&parsed),
            };
            let replayed = (|| -> Result<Population, String> {
                // Both codes below are dropped on purpose: the text only feeds the
                // replay warning. Both are listed among the uncoded sites (#1772).
                let (mut population, _) =
                    crate::api::read_population_with_diag(&parsed.model, &settings, data_path)
                        .map_err(|d| d.message)?;
                // The fit's population carries the level index columns its binding
                // wrote; write them the same way, from the fit's own bindings.
                if !parsed.model.theta_blocks().level_blocks().is_empty() {
                    crate::api::bind_from_fit_on(
                        &mut parsed,
                        model_source,
                        Some(&mut population),
                        &fit.data_bindings,
                    )
                    // The code is dropped on purpose: this text only feeds the
                    // replay warning below. #1775 lists it among the uncoded sites.
                    .map_err(|d| d.message)?;
                }
                match fingerprint.first_difference(&population) {
                    None => Ok(population),
                    Some(d) => Err(format!("it is not the population the fit was given: {d}")),
                }
            })();
            let read_with = if recorded.is_some() {
                "the fit's recorded reader settings"
            } else {
                "the model file's reader settings (the fit recorded none)"
            };
            Ok(match replayed {
                Ok(population) => (Some(population), None),
                Err(why) => (
                    None,
                    Some(format!(
                        "load_fit: the bundled data.csv was not loaded. Read with \
                         {read_with}, {why}. The bundled data, or how \
                         this version of ferx reads it, has changed since the fit was \
                         saved. The estimates are loaded; to run `run_sir` or \
                         `run_covariance`, pass the fit's population or refit."
                    )),
                ),
            })
        }
        // data.csv is bundled with its original headers, so honour the model's
        // `[data]` column mapping (#730) when re-reading — otherwise a fit that
        // mapped e.g. `DV = CONC` cannot round-trip through load_fit — and route
        // the model's non-Gaussian endpoints (#1199): a joint PK-TTE bundle read
        // model-blind came back with *no* event records, and `fit()` now rejects
        // such a population outright (`E_ENDPOINT_UNROUTED`). Fall back to the
        // plain mapped read only if the model can't be parsed (leave the read to
        // fail with its own diagnostic).
        (Ok(m), _, _) => {
            crate::api::read_population_routed_by(&m.model, tmp.path(), None, &m.column_map)
                .map(|p| (Some(p), None))
                .map_err(FitrxError::Corrupt)
        }
        (Err(_), _, _) => {
            crate::io::datareader::read_nonmem_csv_mapped(tmp.path(), None, None, &[])
                .map(|p| (Some(p), None))
                .map_err(FitrxError::Corrupt)
        }
    }
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

/// [`read_text`] for an entry the bundle may not have.
fn read_optional_text<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<Option<String>, FitrxError> {
    if archive.index_for_name(name).is_none() {
        return Ok(None);
    }
    read_text(archive, name).map(Some)
}

/// A `fit.json` value read with each `null` on an `f64` as `NaN`, for a type that
/// serializes whole and so has no per-field adapter (`serde_nan::NullAsNan`).
fn nan_tolerant<'de, D, T>(de: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    Option::<serde_json::Value>::deserialize(de)?
        .map(|v| T::deserialize(crate::io::serde_nan::NullAsNan(v)))
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn read_json<T: serde::de::DeserializeOwned, R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<T, FitrxError> {
    // Parsed to a `Value` and read through `NullAsNan`, the reader `FitResult::from_json_str`
    // uses: the types `fit.json` shares with `FitResult` are then deserialized by one
    // `Deserializer` instead of two, and a non-finite float written as `null` reads back
    // as `NaN` wherever it lands.
    let buf = read_bytes(archive, name)?;
    let value: serde_json::Value = serde_json::from_slice(&buf)?;
    Ok(T::deserialize(crate::io::serde_nan::NullAsNan(value))?)
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
            epred: Vec::new(),
            pwres: Vec::new(),
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
            // Filled by `attach_discrete_rows` from `predictions_discrete.csv`.
            discrete_rows: Vec::new(),
            // Filled by `attach_tte_rows` from `predictions_tte.csv` (#911); empty for a
            // bundle written before that entry existed.
            #[cfg(feature = "survival")]
            tte_rows: Vec::new(),
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
    // EPRED/PWRES (#2871) ride with NPDE; a file written before them carries none.
    let epred_i = col.get("EPRED").copied();
    let pwres_i = col.get("PWRES").copied();
    let cens_i = col.get("CENS").copied();
    // Validated per row against the subject we're filling: the ID order check alone
    // can't catch two same-ID blocks written in the wrong order.
    let n_obs_i = require("N_OBS")?;
    let n_cols = col.len();

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
        if fields.len() != n_cols {
            return Err(FitrxError::Corrupt(format!(
                "predictions.csv row {} has {} fields, expected {n_cols}",
                i + 1,
                fields.len()
            )));
        }
        let id = &fields[id_i];
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
        let raw = &fields[n_obs_i];
        let row_n_obs = raw.parse::<usize>().map_err(|_| {
            FitrxError::Corrupt(format!("predictions.csv row {}: bad N_OBS {raw:?}", i + 1))
        })?;
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
        if let Some(j) = epred_i {
            subjects[idx]
                .epred
                .push(fields.get(j).map(|s| parse_opt(s)).unwrap_or(f64::NAN));
        }
        if let Some(j) = pwres_i {
            subjects[idx]
                .pwres
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

/// Fill `discrete_rows` from `predictions_discrete.csv`: each subject's block is the next
/// `n_obs_discrete` rows, with the row ID as a check, as `predictions.csv` is read. The
/// column and the entry come together, so either one alone is `Corrupt`.
fn attach_discrete_rows(
    subjects: &mut [SubjectResult],
    ebes_csv: &str,
    discrete_csv: Option<&str>,
) -> Result<(), FitrxError> {
    let corrupt = |msg: String| FitrxError::Corrupt(format!("{DISCRETE_ENTRY}: {msg}"));
    let mut ebes = ebes_csv.lines();
    let count_col = ebes
        .next()
        .and_then(|h| h.split(',').position(|c| c == N_OBS_DISCRETE));
    let (count_col, csv) = match (count_col, discrete_csv) {
        (None, None) => return Ok(()),
        (Some(col), Some(csv)) => (col, csv),
        (Some(_), None) => {
            return Err(FitrxError::Corrupt(format!(
                "ebes.csv has {N_OBS_DISCRETE}, but the bundle has no {DISCRETE_ENTRY}"
            )))
        }
        (None, Some(_)) => return Err(corrupt(format!("ebes.csv has no {N_OBS_DISCRETE}"))),
    };
    let counts = ebes
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = parse_csv_row(line);
            let raw = fields.get(count_col).map_or("", String::as_str);
            raw.parse::<usize>()
                .map_err(|_| FitrxError::Corrupt(format!("ebes.csv: bad {N_OBS_DISCRETE} {raw:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut lines = csv.lines();
    let header = lines.next().ok_or_else(|| corrupt("empty".into()))?;
    if header.split(',').ne(DISCRETE_HEADER) {
        return Err(corrupt(format!(
            "header {header:?}, expected {:?}",
            DISCRETE_HEADER.join(",")
        )));
    }
    let mut rows = lines.filter(|line| !line.trim().is_empty()).enumerate();
    for (s, n) in subjects.iter_mut().zip(counts) {
        for _ in 0..n {
            let (i, line) = rows.next().ok_or_else(|| {
                corrupt(format!(
                    "ran out of rows at subject {:?}, which {N_OBS_DISCRETE} gives {n}",
                    s.id
                ))
            })?;
            let fields = parse_csv_row(line);
            if fields.len() != DISCRETE_HEADER.len() {
                return Err(corrupt(format!(
                    "row {} has {} fields, expected {}",
                    i + 1,
                    fields.len(),
                    DISCRETE_HEADER.len()
                )));
            }
            if fields[0] != s.id {
                return Err(corrupt(format!(
                    "row {}: ID {:?} out of order; expected subject {:?}",
                    i + 1,
                    fields[0],
                    s.id
                )));
            }
            let cmt = fields[1]
                .parse::<usize>()
                .map_err(|_| corrupt(format!("row {}: bad CMT {:?}", i + 1, fields[1])))?;
            let num = |f: &str| {
                parse_cell(f).map_err(|_| corrupt(format!("row {}: bad number {f:?}", i + 1)))
            };
            s.discrete_rows.push(crate::types::DiscreteObsDiagnostic {
                cmt,
                time: num(&fields[2])?,
                dv: num(&fields[3])?,
                pred: num(&fields[4])?,
                ipred: num(&fields[5])?,
                iwres: num(&fields[6])?,
            });
        }
    }
    match rows.next() {
        Some((i, _)) => Err(corrupt(format!(
            "row {} is past the rows {N_OBS_DISCRETE} accounts for",
            i + 1
        ))),
        None => Ok(()),
    }
}

/// Fill `tte_rows` from `predictions_tte.csv` (#911): each subject's block is the next
/// `n_obs_tte` rows, with the row ID as a check. The column and the entry come together,
/// so either one alone is `Corrupt`; neither is a bundle written before #911, and leaves
/// every `tte_rows` empty. A copy of [`attach_discrete_rows`] rather than a shared generic:
/// these rows are `survival`-gated and those are not.
#[cfg(feature = "survival")]
fn attach_tte_rows(
    subjects: &mut [SubjectResult],
    ebes_csv: &str,
    tte_csv: Option<&str>,
) -> Result<(), FitrxError> {
    let corrupt = |msg: String| FitrxError::Corrupt(format!("{TTE_ENTRY}: {msg}"));
    let mut ebes = ebes_csv.lines();
    let count_col = ebes
        .next()
        .and_then(|h| h.split(',').position(|c| c == N_OBS_TTE));
    let (count_col, csv) = match (count_col, tte_csv) {
        (None, None) => return Ok(()),
        (Some(col), Some(csv)) => (col, csv),
        (Some(_), None) => {
            return Err(FitrxError::Corrupt(format!(
                "ebes.csv has {N_OBS_TTE}, but the bundle has no {TTE_ENTRY}"
            )))
        }
        (None, Some(_)) => return Err(corrupt(format!("ebes.csv has no {N_OBS_TTE}"))),
    };
    let counts = ebes
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = parse_csv_row(line);
            let raw = fields.get(count_col).map_or("", String::as_str);
            raw.parse::<usize>()
                .map_err(|_| FitrxError::Corrupt(format!("ebes.csv: bad {N_OBS_TTE} {raw:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut lines = csv.lines();
    let header = lines.next().ok_or_else(|| corrupt("empty".into()))?;
    // Two headers are accepted (#1889): the eight columns every bundle since #911
    // carries, and those plus a trailing `NPDE`, written only when a record has a score.
    let base = TTE_HEADER.join(",");
    let has_npde = match header.strip_prefix(base.as_str()) {
        Some("") => false,
        Some(rest) if rest.strip_prefix(',') == Some(TTE_NPDE_COLUMN) => true,
        _ => {
            return Err(corrupt(format!(
                "header {header:?}, expected {base:?} or \"{base},{TTE_NPDE_COLUMN}\""
            )))
        }
    };
    let n_fields = TTE_HEADER.len() + usize::from(has_npde);
    let mut rows = lines.filter(|line| !line.trim().is_empty()).enumerate();
    for (s, n) in subjects.iter_mut().zip(counts) {
        for _ in 0..n {
            let (i, line) = rows.next().ok_or_else(|| {
                corrupt(format!(
                    "ran out of rows at subject {:?}, which {N_OBS_TTE} gives {n}",
                    s.id
                ))
            })?;
            let fields = parse_csv_row(line);
            if fields.len() != n_fields {
                return Err(corrupt(format!(
                    "row {} has {} fields, expected {}",
                    i + 1,
                    fields.len(),
                    n_fields
                )));
            }
            if fields[0] != s.id {
                return Err(corrupt(format!(
                    "row {}: ID {:?} out of order; expected subject {:?}",
                    i + 1,
                    fields[0],
                    s.id
                )));
            }
            let cmt = fields[1]
                .parse::<usize>()
                .map_err(|_| corrupt(format!("row {}: bad CMT {:?}", i + 1, fields[1])))?;
            let num = |f: &str| {
                parse_cell(f).map_err(|_| corrupt(format!("row {}: bad number {f:?}", i + 1)))
            };
            let mut row = crate::types::TteObsDiagnostic {
                cmt,
                time: num(&fields[2])?,
                dv: num(&fields[3])?,
                coxsnell: f64::NAN,
                martingale: f64::NAN,
                deviance: f64::NAN,
                pit: f64::NAN,
                npde: f64::NAN,
            };
            for (name, cell) in TTE_HEADER.iter().zip(&fields).skip(TTE_KEY_COLUMNS.len()) {
                *tte_residual_mut(&mut row, name) = num(cell)?;
            }
            if has_npde {
                row.npde = num(&fields[TTE_HEADER.len()])?;
            }
            s.tte_rows.push(row);
        }
    }
    match rows.next() {
        Some((i, _)) => Err(corrupt(format!(
            "row {} is past the rows {N_OBS_TTE} accounts for",
            i + 1
        ))),
        None => Ok(()),
    }
}

/// A bundle CSV float cell: the inverse of [`fmt_f64`], so empty is `NaN`.
fn parse_cell(cell: &str) -> Result<f64, std::num::ParseFloatError> {
    if cell.is_empty() {
        Ok(f64::NAN)
    } else {
        cell.parse()
    }
}

/// `covtab.csv` back into its table. The covariates and their order come from the header,
/// and each must have its kind in `covariate_types` and the reverse.
fn parse_covtab_csv(
    csv: &str,
    types: &BTreeMap<String, String>,
) -> Result<CovariateTable, FitrxError> {
    let corrupt = |msg: String| FitrxError::Corrupt(format!("{COVTAB_ENTRY}: {msg}"));
    let mut lines = csv.lines();
    let header = parse_csv_row(lines.next().ok_or_else(|| corrupt("empty".into()))?);
    let n_base = COVTAB_BASE_COLUMNS.len();
    if header.len() < n_base || header[..n_base] != COVTAB_BASE_COLUMNS {
        return Err(corrupt(format!(
            "header {header:?} does not start {:?}",
            COVTAB_BASE_COLUMNS.join(",")
        )));
    }
    let names = header[n_base..].to_vec();
    let kinds = names
        .iter()
        .map(|name| {
            let label = types
                .get(name)
                .ok_or_else(|| corrupt(format!("no covariate_types entry for {name:?}")))?;
            [CovariateKind::Continuous, CovariateKind::Categorical]
                .into_iter()
                .find(|k| k.label() == label)
                .ok_or_else(|| corrupt(format!("unknown covariate type {label:?} for {name:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(extra) = types.keys().find(|k| !names.contains(k)) {
        return Err(corrupt(format!(
            "no column for covariate_types entry {extra:?}"
        )));
    }
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = parse_csv_row(line);
        if fields.len() != header.len() {
            return Err(corrupt(format!(
                "row {} has {} fields, expected {}",
                i + 1,
                fields.len(),
                header.len()
            )));
        }
        let num = |f: &String| {
            parse_cell(f).map_err(|_| corrupt(format!("row {}: bad number {f:?}", i + 1)))
        };
        let time = num(&fields[1])?;
        let evid = fields[2]
            .parse::<u32>()
            .map_err(|_| corrupt(format!("row {}: bad EVID {:?}", i + 1, fields[2])))?;
        let values = fields[n_base..].iter().map(num).collect::<Result<_, _>>()?;
        rows.push(CovariateRow {
            id: std::mem::take(&mut fields[0]),
            time,
            evid,
            values,
        });
    }
    Ok(CovariateTable { names, kinds, rows })
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
        // bundle fails fast rather than passing off inconsistent data. Compared as
        // the value both cells read back as, not as text: a consistent bundle
        // writes both with one formatter, but which one depends on its writer
        // (six decimals before #1847, the shortest round-trip text after, R's
        // own shortest spelling in ferx-r). `+0` and `-0` agree, as in `run_covariance`'s
        // EBE comparison, and two `NaN`s do.
        let ebe = fit.subjects[si].eta.get(ei).copied().unwrap_or(f64::NAN);
        let mode = parse_or_nan(&fields[4])?;
        if mode != ebe && !(mode.is_nan() && ebe.is_nan()) {
            let expected_mode = fmt_f64(ebe);
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
    if w.omega.init_as_sd.len() != n_eta {
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
    if w.sigma.init_as_sd.len() != n_sigma {
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
    // #847 companions to `residual_correlations`, which must be parallel: a truncated
    // FIX vector would silently report an estimated correlation as free, and a truncated
    // SE vector would mis-align the SE with the pair it belongs to.
    let n_corr = w.sigma.residual_correlations.len();
    if w.sigma.residual_correlation_fixed.len() != n_corr {
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
    // #2833: the override descriptors and SEs are each absent or parallel to the values.
    let n_overrides = w.mixture_overrides.as_ref().map_or(0, Vec::len);
    if !w.mixture_override_params.is_empty() && w.mixture_override_params.len() != n_overrides {
        return bail(format!(
            "mixture_override_params ({}) does not match mixture_overrides ({})",
            w.mixture_override_params.len(),
            n_overrides
        ));
    }
    if let Some(se) = &w.se_mixture_overrides {
        if se.len() != n_overrides {
            return bail(format!(
                "se_mixture_overrides ({}) does not match mixture_overrides ({})",
                se.len(),
                n_overrides
            ));
        }
    }
    // `cov_fixed` is empty or parallel to the covariance matrix.
    let cov_dim = w.covariance_matrix.as_ref().map(|m| m.rows);
    if !w.cov_fixed.is_empty() && cov_dim != Some(w.cov_fixed.len()) {
        return bail(format!(
            "cov_fixed ({}) does not match the covariance matrix ({})",
            w.cov_fixed.len(),
            cov_dim.map_or("none".to_string(), |d| format!("dim {d}"))
        ));
    }
    if let Some(iov) = &w.iov {
        let n_kappa = iov.kappa_names.len();
        // Ω_IOV's shape, as Ω's above (#1833): a 1×2 loaded `Ok` and panicked in the
        // first post-hoc Cholesky.
        let (rows, cols) = (iov.omega_iov.rows, iov.omega_iov.cols);
        if rows != cols {
            return bail(format!("iov.omega_iov is {rows}×{cols}; expected square"));
        }
        if rows != n_kappa {
            return bail(format!(
                "iov.omega_iov dim ({rows}) does not match iov.kappa_names ({n_kappa})"
            ));
        }
        if iov.kappa_init_as_sd.len() != n_kappa {
            return bail(format!(
                "iov.kappa_init_as_sd ({}) does not match iov.kappa_names ({})",
                iov.kappa_init_as_sd.len(),
                n_kappa
            ));
        }
        if !iov.kappa_levels.is_empty() && iov.kappa_levels.len() != n_kappa {
            return bail(format!(
                "iov.kappa_levels ({}) does not match iov.kappa_names ({})",
                iov.kappa_levels.len(),
                n_kappa
            ));
        }
        // Each bundled level holds kappas `kappa_levels` puts at it (review r1 #11).
        for level in &iov.levels {
            for name in &level.kappa_names {
                let at = iov
                    .kappa_names
                    .iter()
                    .position(|k| k == name)
                    .and_then(|k| iov.kappa_levels.get(k));
                if at != Some(&level.name) {
                    return bail(format!(
                        "iov.levels names kappa `{name}` at level `{}`, but iov.kappa_levels \
                         puts it at {}",
                        level.name,
                        at.map_or("no level".to_string(), |l| format!("`{l}`"))
                    ));
                }
            }
        }
        if iov.kappa_param_types.len() != n_kappa {
            return bail(format!(
                "iov.kappa_param_types ({}) does not match iov.kappa_names ({})",
                iov.kappa_param_types.len(),
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
    let vi = w.vi.take().map(|v| v.into_result(&subjects)).transpose()?;
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
    let omega_init = w.omega_init.into_dmatrix()?;
    let covariance_matrix = w.covariance_matrix.map(|m| m.into_dmatrix()).transpose()?;

    let (
        omega_iov,
        kappa_names,
        kappa_fixed,
        kappa_init_as_sd,
        kappa_weights,
        kappa_weight_typical,
        kappa_param_types,
        se_kappa,
        shrinkage_kappa,
        shrinkage_kappa_by_occ,
        omega_iov_param_corr,
        kappa_levels,
    ) = match w.iov {
        Some(iov) => (
            Some(iov.omega_iov.into_dmatrix()?),
            iov.kappa_names,
            iov.kappa_fixed,
            iov.kappa_init_as_sd,
            iov.kappa_weights,
            iov.kappa_weight_typical,
            iov.kappa_param_types
                .iter()
                .map(|t| eta_param_type_from_str(t))
                .collect::<Result<Vec<_>, _>>()?,
            iov.se_kappa,
            iov.shrinkage_kappa,
            iov.shrinkage_kappa_by_occ,
            iov.omega_iov_param_corr
                .map(|m| m.into_dmatrix())
                .transpose()?,
            iov.kappa_levels,
        ),
        None => (
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
        ),
    };

    let (
        sir_ci_theta,
        sir_ci_omega,
        sir_ci_sigma,
        sir_ci_kappa,
        sir_ess,
        sir_resamples_packed,
        sir_settings,
    ) = match w.sir {
        Some(s) => (
            s.ci_theta,
            s.ci_omega,
            s.ci_sigma,
            s.ci_kappa,
            s.ess,
            s.resamples_packed,
            s.settings.map(SirSettingsWire::into_settings).transpose()?,
        ),
        None => (None, None, None, None, None, None, None),
    };
    let scoring_settings = w
        .scoring_settings
        .map(|s| s.into_settings("scoring_settings"))
        .transpose()?;
    // Two copies of one seed (#1767): a record whose seed differs from `sir_seed`
    // would make `fit.sir_seed` report one run and `run_sir` reproduce another.
    if let Some(st) = sir_settings.as_ref() {
        if w.sir_seed != Some(st.seed) {
            return Err(FitrxError::Corrupt(format!(
                "sir_seed {:?} disagrees with sir.settings.seed {}",
                w.sir_seed, st.seed
            )));
        }
    }

    // `proposal_df` is `∞` for a multivariate-normal proposal, which JSON writes as `null`
    // and `nan_tolerant` reads as `NaN`. No proposal has a `NaN` df, so this is exact.
    let importance_sampling = w.importance_sampling.map(|mut imp| {
        if imp.proposal_df.is_nan() {
            imp.proposal_df = f64::INFINITY;
        }
        imp
    });

    let held_at_bound = held_at_bound_on_load(
        std::mem::take(&mut w.held_at_bound),
        &w.profiled_levels_at_bound,
        &w.theta.names,
    );
    let mut fit = FitResult {
        restored_from_checkpoint: true,
        covariate_relations: Vec::new(),
        method,
        method_chain,
        method_wall_times_secs: w.method_wall_times_secs,
        covariance_wall_time_secs: w.covariance_wall_time_secs,
        converged: w.converged,
        ofv: w.ofv,
        ofv_data: w.ofv_data,
        ofv_prior: w.ofv_prior,
        prior_summary: std::mem::take(&mut w.prior_summary),
        aic: w.aic,
        bic: w.bic,
        theta: w.theta.estimates,
        theta_names: w.theta.names,
        eta_names: w.omega.names,
        omega,
        sigma: w.sigma.estimates,
        sigma_names: w.sigma.names,
        residual_correlations: w.sigma.residual_correlations,
        residual_correlation_fixed: w.sigma.residual_correlation_fixed,
        error_model: error_model_from_str(&w.error_model)?,
        covariance_matrix,
        se_theta: w.theta.se,
        se_omega: w.omega.se,
        se_sigma: w.sigma.se,
        se_residual_correlations: w.sigma.se_residual_correlations,
        theta_fixed: w.theta.fixed,
        omega_fixed: w.omega.fixed,
        sigma_fixed: w.sigma.fixed,
        omega_init_as_sd: w.omega.init_as_sd,
        sigma_init_as_sd: w.sigma.init_as_sd,
        subjects,
        n_obs: w.n_obs,
        n_subjects: w.n_subjects,
        n_parameters: w.n_parameters,
        n_iterations: w.n_iterations,
        interaction: w.interaction,
        // The entries with a `details` payload come off the wire; the rebuild
        // below keys them to their `warnings` message and classifies the rest.
        warnings_structured: w
            .warning_details
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect(),
        warnings: w.warnings,
        sir_ci_theta,
        sir_ci_omega,
        sir_ci_sigma,
        sir_ci_kappa,
        sir_ess,
        sir_resamples_packed,
        importance_sampling,
        impmap_trace: w.impmap_trace,
        bayes: w.bayes,
        vi,
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
        kappa_levels,
        // Filled by `load_fit` from the `ebes_kappa_<LEVEL>.csv` entries.
        ebe_kappas_by_level: Vec::new(),
        saem_mu_ref_m_step_evals_saved: w.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: w.saem_n_subjects_hmc,
        saem_mh_accept_tail: w.saem_mh_accept_tail,
        gradient_method_inner: w.gradient_method_inner,
        gradient_method_outer: w.gradient_method_outer,
        uses_ode_solver: w.uses_ode_solver,
        ode_linear_exact: w.ode_linear_exact,
        uses_sde: w.uses_sde,
        n_threads_used: w.n_threads_used,
        nlopt_missing_algorithms: w.nlopt_missing_algorithms,
        covariance_n_evals_estimated: w.covariance_n_evals_estimated,
        trace_path: w.trace_path,
        // Filled by `load_fit` from `trace.csv`.
        trace_csv: None,
        ebe_convergence_warnings: w.ebe_convergence_warnings,
        max_unconverged_subjects: w.max_unconverged_subjects,
        total_ebe_fallbacks: w.total_ebe_fallbacks,
        total_ebe_nm_polished: w.total_ebe_nm_polished,
        covariance_status: covariance_status_from_str(&w.covariance_status)?,
        // Absent key → `None`; present-but-unrecognised → `Corrupt`, the same verdict
        // every other enum on this wire gets, rather than silently degrading to `None`.
        covariance_method: w
            .covariance_method
            .as_deref()
            .map(|s| {
                crate::types::CovarianceMethod::from_label(s)
                    .ok_or_else(|| FitrxError::Corrupt(format!("unknown covariance_method {s:?}")))
            })
            .transpose()?,
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
        environment: w.environment,
        eta_param_info,
        kappa_param_types,
        theta_transform,
        sigma_types,
        cov_eigenvalues: w.cov_eigenvalues,
        cov_condition_number: w.cov_condition_number,
        cov_fixed: w.cov_fixed,
        bic_inputs: w.bic_inputs,
        profiled_blocks: w.profiled_blocks,
        profiled_levels_at_bound: w.profiled_levels_at_bound,
        held_at_bound,
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
        final_gradient_source: w.final_gradient_source,
        newton_decrement: w.newton_decrement,
        optimizer_stop: w.optimizer_stop,
        eval_budget_spent: w.eval_budget_spent,
        evaluation_only: w.evaluation_only,
        optimizer: w.optimizer,
        n_starts: w.n_starts,
        multi_start_seed: w.multi_start_seed,
        starts: w.starts,
        saem_seed: w.saem_seed,
        sir_seed: w.sir_seed,
        sir_settings,
        scoring_settings,
        imp_seed: w.imp_seed,
        npde_seed: w.npde_seed,
        bloq_method: w.bloq_method,
        outer_maxiter: w.outer_maxiter,
        outer_gtol: w.outer_gtol,
        inits_from_nca: w.inits_from_nca,
        covariate_names: w.covariate_names,
        input_columns: w.input_columns,
        #[cfg(feature = "nn")]
        neural_networks: w.neural_networks,
        data_bindings: w.data_bindings.into(),
        reader_settings: w.reader_settings,
        population_fingerprint: w.population_fingerprint,
        iov_occasion: w.iov_occasion,
        // Filled by `load_fit` from `covtab.csv` and `interface_extras.json`.
        covariate_table: None,
        interface_extras: None,
        exclusions: w.exclusions,
        // `FitResult`'s own serde skips it (#1815). `None` for SAEM / IMP / Bayes;
        // consumers validate it against the estimates above before reusing it.
        packed_estimate: w.packed_estimate,
        mixture_overrides: w.mixture_overrides,
        mixture_override_params: w.mixture_override_params,
        se_mixture_overrides: w.se_mixture_overrides,
        left_init: w.left_init,
        stationary_start: w.stationary_start,
        omega_is_diagonal: w.omega_is_diagonal,
        kappa_is_diagonal: w.kappa_is_diagonal,
    };
    crate::api::rebuild_warnings_structured(&mut fit);
    Ok(fit)
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

/// A bundle's `held_at_bound` (#2797): its own, or — for a bundle written before the field
/// existed, which carries only `profiled_levels_at_bound` — those levels as
/// [`crate::types::HeldReason::ProfiledBound`] entries, so a reader has one rule for old
/// and new bundles. A θ's packed index is its θ index.
fn held_at_bound_on_load(
    held: Vec<crate::types::HeldAtBound>,
    profiled_levels_at_bound: &[usize],
    theta_names: &[String],
) -> Vec<crate::types::HeldAtBound> {
    if !held.is_empty() {
        return held;
    }
    profiled_levels_at_bound
        .iter()
        .map(|&k| crate::types::HeldAtBound {
            name: theta_names.get(k).cloned().unwrap_or_default(),
            kind: crate::types::ParameterKind::Theta,
            index: k,
            packed_index: k,
            reason: crate::types::HeldReason::ProfiledBound,
            class: None,
        })
        .collect()
}

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

    use crate::types::test_helpers::dummy_subject;

    fn dummy_population(ids: &[&str], n_obs_each: usize) -> Population {
        let mut subjects = Vec::new();
        for id in ids {
            subjects.push(Subject {
                obs_rgroup: Vec::new(),
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
                pk_only_raw_times: Vec::new(),
                reset_period_times: Vec::new(),
                pk_only_covariates: vec![],
                reset_times: vec![],
                reset_covariates: Vec::new(),
                cmt_toggles: Vec::new(),
                cens: vec![0; n_obs_each],
                occasions: vec![],
                obs_l2: Vec::new(),
                dose_occasions: vec![],
                reset_occasions: Vec::new(),
                pk_only_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: vec![],
                endpoint_covariates: Vec::new(),
                filled_cells: Vec::new(),
                simulated_mixnum: None,
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

    use crate::types::test_helpers::minimal_fit_result;

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
        wire.sigma.residual_correlation_fixed = vec![false];
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

    /// `FitResult::from_json_str` reads back the `null`s `write_result_json`
    /// writes for non-finite floats: a plain `f64` (top level, and nested in a
    /// `Vec` inside an `Option` struct) as `NaN`, an `Option<f64>` as `None`,
    /// finite values unchanged. The plain serde read of the same file fails, so
    /// the straddle is pinned too.
    #[test]
    fn from_json_str_reads_non_finite_nulls_back_as_nan() {
        let mut fit = minimal_fit_result();
        fit.ofv = 123.5;
        fit.iwres_lag1_r = f64::NAN;
        fit.dw_statistic = f64::INFINITY;
        fit.cov_condition_number = None;
        fit.covariate_table = Some(crate::types::CovariateTable {
            names: vec!["WT".into(), "AGE".into()],
            kinds: vec![crate::types::CovariateKind::Continuous; 2],
            rows: vec![crate::types::CovariateRow {
                id: "S1".into(),
                time: 0.0,
                evid: 0,
                values: vec![f64::NAN, 41.0],
            }],
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan-fit.json");
        crate::io::output::write_result_json(&fit, path.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        let err = serde_json::from_str::<FitResult>(&text).unwrap_err();
        assert!(err.to_string().contains("null"), "{err}");

        let back = FitResult::from_json_str(&text).unwrap();
        assert_eq!(back.ofv, 123.5);
        assert!(back.iwres_lag1_r.is_nan(), "{}", back.iwres_lag1_r);
        assert!(back.dw_statistic.is_nan(), "{}", back.dw_statistic);
        assert_eq!(back.cov_condition_number, None);
        let row = &back.covariate_table.unwrap().rows[0];
        assert!(row.values[0].is_nan(), "{:?}", row.values);
        assert_eq!(row.values[1], 41.0);
        assert_eq!(back.theta, fit.theta);
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

    /// #2268: a warning's `details` payload survives the YAML write and a `.fitrx`
    /// save and reload unchanged. The fit carries one at-source entry with a payload
    /// under a chain prefix (so the reload has to rebuild the `[FOCEI] ` key to match
    /// it) and one plain message, which reloads classified and without a payload.
    #[test]
    fn warning_details_survive_yaml_and_fitrx_round_trip() {
        use crate::types::{WarningCode, WarningEntry, WarningSeverity};
        let details = serde_json::json!({
            "ofv": "NaN",
            "unusable_subjects": [{"id": "S1", "n_obs": 3}, {"id": "S2", "n_obs": 3}],
        });
        let native = WarningEntry {
            severity: WarningSeverity::Critical,
            category: WarningCode::Convergence,
            message: "W_NONFINITE_OBJECTIVE: the objective is \"NaN\"; subject `a\\b`\nand 2 more"
                .into(),
            source_method: Some("FOCEI".into()),
            details: Some(details.clone()),
        };
        const PLAIN: &str = "a plain note: \"quoted\"";
        let mut r = minimal_fit_result();
        r.warnings = vec![format!("[FOCEI] {}", native.message), PLAIN.into()];
        r.warnings_structured = vec![native.clone()];
        crate::api::rebuild_warnings_structured(&mut r);
        assert_eq!(r.warnings_structured.len(), 2);
        let json = |e: &WarningEntry| serde_json::to_value(e).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run1.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        let l = load_fit(&path).unwrap().fit;
        assert_eq!(l.warnings, r.warnings);
        assert_eq!(l.warnings_structured.len(), 2);
        assert_eq!(
            json(&l.warnings_structured[0]),
            json(&native),
            "the payload reloads"
        );
        let plain = &l.warnings_structured[1];
        assert_eq!((plain.message.as_str(), &plain.details), (PLAIN, &None));

        // The YAML of the reloaded fit is one valid document — the messages hold
        // `"`, `\` and a newline — and lists the payload, and only that warning.
        let yaml_path = dir.path().join("run1-fit.yaml");
        crate::io::output::write_estimates_yaml(&l, yaml_path.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&yaml_path).unwrap();
        let yaml: serde_json::Value = serde_saphyr::from_str(&text)
            .unwrap_or_else(|e| panic!("the fit YAML does not parse: {e}\n{text}"));
        assert_eq!(yaml["warnings"], serde_json::json!(r.warnings), "{text}");
        let block = yaml["warning_details"].as_array().expect("block written");
        assert_eq!(block.len(), 1, "{text}");
        assert_eq!(
            block[0],
            serde_json::json!({
                "code": "convergence",
                "severity": "Critical",
                "source_method": "FOCEI",
                "message": native.message,
                "details": details,
            })
        );

        // A bundle saved before #2268 has no `warning_details`, and one naming a
        // code this version does not know still loads: both re-classify the message.
        let mut wire = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(wire["warning_details"].as_array().map(Vec::len), Some(1));
        wire["warning_details"][0]["category"] = "a_future_code".into();
        let unknown: FitWire = serde_json::from_value(wire.clone()).unwrap();
        wire.as_object_mut().unwrap().remove("warning_details");
        let old: FitWire = serde_json::from_value(wire).unwrap();
        for (label, w) in [("unknown code", unknown), ("pre-#2268", old)] {
            let l = wire_to_fit_result(w, r.subjects.clone(), Vec::new()).unwrap();
            let e = &l.warnings_structured[0];
            assert_eq!(e.message, native.message, "{label}");
            assert_eq!(e.source_method.as_deref(), Some("FOCEI"), "{label}");
            assert!(e.details.is_none(), "{label}");
        }

        // A fit with no payload writes neither the wire key nor the YAML block.
        let bare = minimal_fit_result();
        let wire = serde_json::to_value(build_fit_wire(&bare)).unwrap();
        assert!(wire.get("warning_details").is_none());
        crate::io::output::write_estimates_yaml(&bare, yaml_path.to_str().unwrap()).unwrap();
        assert!(!std::fs::read_to_string(&yaml_path)
            .unwrap()
            .contains("warning_details"));
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
        r.residual_correlation_fixed = vec![true];

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

    /// `load_fit_without_population` reads the fit and model source and leaves the
    /// bundled data alone. The case it exists for: a bundle whose model this build
    /// cannot parse (here a misspelled block stands in for a `[covariate_nn]` model
    /// in a build without `nn`), with renamed columns. `load_fit` then falls back
    /// to the unmapped read and fails on the missing `TIME` column — refusing a
    /// caller that never needed the data, such as `ferx export-dcm`.
    #[test]
    fn load_fit_without_population_skips_the_bundled_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unparsable.fitrx");
        let data_csv = dir.path().join("input.csv");
        std::fs::write(
            &data_csv,
            "ID,TAFD,CONC,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n",
        )
        .unwrap();
        let model_source = "[not_a_block]\n[data]\n  TIME = TAFD\n  DV = CONC\n";
        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(
            &r,
            &p,
            model_source,
            &path,
            SaveFitOptions {
                include_data: Some(data_csv),
            },
        )
        .unwrap();

        let err = load_fit(&path).expect_err("the control: the full load replays the data");
        assert!(err.to_string().contains("TIME"), "{err}");
        let loaded = load_fit_without_population(&path).unwrap();
        assert!(loaded.population.is_none());
        assert_eq!(loaded.model_source, model_source);
        assert_eq!(loaded.fit.theta, r.theta);
        assert_eq!(loaded.fit.model_text.as_deref(), Some(model_source));
    }

    #[cfg(feature = "survival")]
    #[test]
    fn roundtrip_bundled_data_routes_the_model_endpoints() {
        // #1199: a joint PK-TTE bundle re-read through the model-blind reader came
        // back with *no* event records (and its event rows as Gaussian CMT-3
        // observations), so anything fed the reloaded population re-fit or
        // predicted the Gaussian half only. `load_fit` now routes by the bundled
        // model. The reference is the routed reader on the same file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("joint.fitrx");
        let model_source = std::fs::read_to_string("examples/pktte_joint.ferx").unwrap();
        let data_csv = PathBuf::from("data/pktte_joint.csv");

        let r = minimal_fit_result();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(
            &r,
            &p,
            &model_source,
            &path,
            SaveFitOptions {
                include_data: Some(data_csv),
            },
        )
        .unwrap();

        let loaded = load_fit(&path).expect("load_fit reads the bundled joint data.csv");
        let pop = loaded.population.expect("data.csv was bundled");
        let events: usize = pop.subjects.iter().map(|s| s.obs_records.len()).sum();
        let gaussian_on_3: usize = pop
            .subjects
            .iter()
            .map(|s| s.obs_cmts.iter().filter(|&&c| c == 3).count())
            .sum();

        let m = crate::parser::model_parser::parse_full_model(&model_source)
            .unwrap()
            .model;
        let (want, _) = crate::api::read_population_for(
            &m,
            &None,
            "data/pktte_joint.csv",
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        let want_events: usize = want.subjects.iter().map(|s| s.obs_records.len()).sum();
        assert!(want_events > 0, "the fixture carries event rows");
        assert_eq!(
            events, want_events,
            "reloaded population must carry every event record"
        );
        assert_eq!(
            gaussian_on_3, 0,
            "…and none of them as a Gaussian CMT-3 row"
        );
    }

    /// The discrete arm of the same reload: a `[binary_model]` bundle comes back
    /// with its rows as `DiscreteState` records, none in the Gaussian grid.
    #[cfg(feature = "survival")]
    #[test]
    fn roundtrip_bundled_data_routes_a_binary_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("binary.fitrx");
        let model_source = std::fs::read_to_string("examples/binary_logistic.ferx").unwrap();
        let data = "data/binary_logistic.csv";

        save_fit(
            &minimal_fit_result(),
            &dummy_population(&["S1", "S2"], 3),
            &model_source,
            &path,
            SaveFitOptions {
                include_data: Some(PathBuf::from(data)),
            },
        )
        .unwrap();

        let pop = load_fit(&path)
            .expect("load_fit reads the bundled binary data.csv")
            .population
            .expect("data.csv was bundled");
        let records: usize = pop.subjects.iter().map(|s| s.obs_records.len()).sum();
        let gaussian_on_3: usize = pop
            .subjects
            .iter()
            .map(|s| s.obs_cmts.iter().filter(|&&c| c == 3).count())
            .sum();

        let m = crate::parser::model_parser::parse_full_model(&model_source)
            .unwrap()
            .model;
        let (want, _) =
            crate::api::read_population_for(&m, &None, data, None, None, None, &[]).unwrap();
        let want_records: usize = want.subjects.iter().map(|s| s.obs_records.len()).sum();
        assert!(want_records > 0, "the fixture carries binary rows");
        assert_eq!(
            records, want_records,
            "reloaded population must carry every record"
        );
        assert_eq!(
            gaussian_on_3, 0,
            "…and none of them as a Gaussian CMT-3 row"
        );
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
        r.subjects[0].epred = vec![4.0, 3.0, 2.0];
        r.subjects[0].pwres = vec![0.5, -0.6, 0.7];
        r.subjects[1].epred = vec![1.5, 1.25];
        r.subjects[1].pwres = vec![f64::NAN, f64::NAN];
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
            // #2871: EPRED / PWRES ride in the same file; a NaN PWRES (a masked
            // subject) loads back as NaN.
            assert_eq!(a.epred.len(), b.epred.len(), "subject {} epred len", a.id);
            assert_eq!(a.pwres.len(), b.pwres.len(), "subject {} pwres len", a.id);
            for (x, y) in a.epred.iter().zip(&b.epred) {
                assert!((x - y).abs() < 1e-6, "epred {x} vs {y}");
            }
            for (x, y) in a.pwres.iter().zip(&b.pwres) {
                assert!(
                    (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-6,
                    "pwres {x} vs {y}"
                );
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
            assert!(s.epred.is_empty() && s.pwres.is_empty());
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

    /// The tally behind the BIC variants and the optimizer's init-escape
    /// verdict (#1177) round-trip. `minimal_fit_result` carries the *default*
    /// tally, which is indistinguishable from a dropped field, so this uses a
    /// non-default one — the mutation `bic_inputs: Default::default()` in
    /// `build_fit_wire` is unobservable otherwise.
    #[test]
    fn bic_inputs_and_left_init_round_trip_through_fitrx() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run1.fitrx");
        let mut r = minimal_fit_result();
        r.bic_inputs = crate::types::BicInputs {
            n_obs: 155,
            theta_random: 2,
            theta_fixed: 1,
            omega: 2,
            kappa: 0,
            sigma: 1,
            sigma_random: false,
        };
        r.left_init = Some(false);
        r.omega_is_diagonal = Some(false);
        r.kappa_is_diagonal = Some(true);
        assert_ne!(r.bic_inputs, crate::types::BicInputs::default());
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let l = load_fit(&path).unwrap().fit;
        assert_eq!(l.bic_inputs, r.bic_inputs);
        assert_eq!(l.left_init, Some(false));
        assert_eq!(l.omega_is_diagonal, Some(false));
        assert_eq!(l.kappa_is_diagonal, Some(true));
        // And the loaded bundle ranks: the tally partitions `n_parameters`.
        assert!(
            crate::model_selection::bic(&l, crate::model_selection::BicType::Mixed).is_finite()
        );
    }

    /// #1065: `profiled_blocks` survives a bundle, and a fit without one writes no key
    /// (so a bundle saved before the field existed reads back empty).
    #[test]
    fn profiled_blocks_round_trip_through_fitrx_and_are_omitted_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);

        let path = dir.path().join("profiled.fitrx");
        let mut r = minimal_fit_result();
        r.profiled_blocks = vec!["PLACEBO".into(), "ARMFX".into()];
        // #2112: the levels the covariance step held on their bound, read by the SE
        // printers of a reloaded fit.
        r.profiled_levels_at_bound = vec![1, 4];
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap().fit;
        assert_eq!(loaded.profiled_blocks, ["PLACEBO", "ARMFX"]);
        assert_eq!(loaded.profiled_levels_at_bound, [1, 4]);

        let r = minimal_fit_result();
        let wire = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert!(wire.get("profiled_blocks").is_none(), "{wire}");
        assert!(wire.get("profiled_levels_at_bound").is_none(), "{wire}");
        let path = dir.path().join("plain.fitrx");
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap().fit;
        assert!(loaded.profiled_blocks.is_empty());
        assert!(loaded.profiled_levels_at_bound.is_empty());
    }

    /// #2797 (D8): `held_at_bound` round-trips, as `snake_case` reasons, is omitted when
    /// empty, and a bundle written before the field existed — `profiled_levels_at_bound`
    /// only — loads with those levels as `ProfiledBound` entries, so a reader has one rule
    /// for old and new bundles.
    #[test]
    fn held_at_bound_round_trips_and_an_old_bundle_synthesises_its_profiled_levels() {
        use crate::types::{HeldAtBound, HeldReason, ParameterKind};
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);

        let mut r = minimal_fit_result();
        r.held_at_bound = vec![HeldAtBound {
            name: "ETA_KA".into(),
            kind: ParameterKind::Omega,
            index: 0,
            packed_index: r.theta.len(),
            reason: HeldReason::CollapsedAtGuard,
            class: None,
        }];
        let path = dir.path().join("held.fitrx");
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        assert_eq!(
            fit_json(&path)["held_at_bound"][0]["reason"],
            "collapsed_at_guard"
        );
        assert_eq!(load_fit(&path).unwrap().fit.held_at_bound, r.held_at_bound);

        // Omitted when empty.
        let wire = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        assert!(wire.get("held_at_bound").is_none(), "{wire}");

        // An old bundle: only `profiled_levels_at_bound = [0, 2]`.
        let mut old = minimal_fit_result();
        old.theta_names = (0..old.theta.len()).map(|i| format!("L{i}")).collect();
        old.profiled_levels_at_bound = vec![0, 2];
        let path = dir.path().join("old.fitrx");
        save_fit(&old, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        assert!(fit_json(&path).get("held_at_bound").is_none());
        let loaded = load_fit(&path).unwrap().fit;
        let got: Vec<(String, ParameterKind, usize, usize, HeldReason)> = loaded
            .held_at_bound
            .iter()
            .map(|h| (h.name.clone(), h.kind, h.index, h.packed_index, h.reason))
            .collect();
        assert_eq!(
            got,
            [
                (
                    "L0".to_string(),
                    ParameterKind::Theta,
                    0,
                    0,
                    HeldReason::ProfiledBound
                ),
                (
                    "L2".to_string(),
                    ParameterKind::Theta,
                    2,
                    2,
                    HeldReason::ProfiledBound
                ),
            ]
        );
        assert_eq!(loaded.profiled_levels_at_bound, [0, 2]);
    }

    /// T5 (#2809): a `[mixture]` override entry's `class` round-trips as a 1-based number;
    /// a base entry writes no `class` key at all (so a bundle without an override entry is
    /// byte-unchanged); and an entry written before the field existed loads as `None`.
    #[test]
    fn held_at_bound_class_round_trips_and_is_absent_for_a_base_entry() {
        use crate::types::{HeldAtBound, HeldReason, ParameterKind};
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        let entry = |name: &str, packed_index, class| HeldAtBound {
            name: name.into(),
            kind: ParameterKind::Omega,
            index: 0,
            packed_index,
            reason: HeldReason::CollapsedAtGuard,
            class,
        };
        let mut r = minimal_fit_result();
        let n = r.theta.len();
        r.held_at_bound = vec![
            entry("eta_CL", n, None),
            entry("eta_CL_MIX2", n + 3, Some(2)),
        ];
        let path = dir.path().join("class.fitrx");
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();
        let json = fit_json(&path);
        let held = &json["held_at_bound"];
        assert!(held[0].get("class").is_none(), "{held}");
        assert_eq!(held[1]["class"], 2, "{held}");
        assert_eq!(load_fit(&path).unwrap().fit.held_at_bound, r.held_at_bound);

        // An entry from a bundle written before #2809.
        let old: HeldAtBound = serde_json::from_value(serde_json::json!({
            "name": "eta_CL", "kind": "Omega", "index": 0, "packed_index": n,
            "reason": "collapsed_at_guard",
        }))
        .unwrap();
        assert_eq!(old, entry("eta_CL", n, None));
    }

    /// The held-at-bound fixture for #2817: θ `V` (index 1) collapsed on the guard and θ
    /// `KA` (index 2) a profiled level, so a reader that mishandles one entry cannot pass
    /// on the strength of the other.
    fn two_held_thetas() -> FitResult {
        use crate::types::{HeldAtBound, HeldReason, ParameterKind};
        let mut r = minimal_fit_result();
        r.held_at_bound = vec![
            HeldAtBound {
                name: "V".into(),
                kind: ParameterKind::Theta,
                index: 1,
                packed_index: 1,
                reason: HeldReason::CollapsedAtGuard,
                class: None,
            },
            HeldAtBound {
                name: "KA".into(),
                kind: ParameterKind::Theta,
                index: 2,
                packed_index: 2,
                reason: HeldReason::ProfiledBound,
                class: None,
            },
        ];
        r
    }

    /// Rename the reason of `held_at_bound[0]` in a fit JSON value to one no build knows.
    fn inject_future_reason(v: &mut serde_json::Value) {
        assert_eq!(v["held_at_bound"][0]["reason"], "collapsed_at_guard", "{v}");
        v["held_at_bound"][0]["reason"] = "some_future_reason".into();
    }

    /// #2817: a bundle whose `held_at_bound` names a reason this build does not know — as a
    /// later ferx that added one would write — loads, rather than failing the whole
    /// `load_fit`. The unknown entry reads as `HeldReason::Unknown` and still counts as
    /// held (SE `N/A`, YAML `se: ~`); the neighbouring known entry and the not-held θ are
    /// untouched. Under the strict derive (no `#[serde(other)]`) `load_fit` returns `Err`
    /// and the first `unwrap` fails.
    #[test]
    fn a_bundle_with_an_unknown_held_reason_loads_and_the_entry_stays_held() {
        use crate::io::output::{se_held, write_estimates_yaml};
        use crate::types::{HeldReason, ParameterKind};
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        let r = two_held_thetas();
        let src = dir.path().join("known.fitrx");
        save_fit(&r, &p, "model source\n", &src, SaveFitOptions::default()).unwrap();
        let dst = dir.path().join("future.fitrx");
        rewrite_bundle(&src, &dst, |name, body| {
            Some(if name == "fit.json" {
                edit_json(body, inject_future_reason)
            } else {
                body
            })
        });
        assert_eq!(
            fit_json(&dst)["held_at_bound"][0]["reason"],
            "some_future_reason"
        );

        let loaded = load_fit(&dst)
            .expect("an unknown held reason must not fail the bundle")
            .fit;
        let reasons: Vec<HeldReason> = loaded.held_at_bound.iter().map(|h| h.reason).collect();
        assert_eq!(reasons, [HeldReason::Unknown, HeldReason::ProfiledBound]);
        assert_eq!(loaded.held_at_bound[0].name, "V");
        assert_eq!(loaded.held_at_bound[0].packed_index, 1);

        // Displays: held, so no SE — and the not-held θ keeps its number.
        assert_eq!(
            se_held(&loaded, ParameterKind::Theta, 1),
            Some(HeldReason::Unknown)
        );
        assert_eq!(se_held(&loaded, ParameterKind::Theta, 0), None);
        let yaml_path = dir.path().join("est.yaml");
        write_estimates_yaml(&loaded, yaml_path.to_str().unwrap()).unwrap();
        let yaml = std::fs::read_to_string(&yaml_path).unwrap();
        assert!(
            yaml.contains(
                "  V:\n    estimate: 2.000000\n    se: ~\n    rse_pct: ~\n    held: unknown\n"
            ),
            "{yaml}"
        );
        assert!(yaml.contains("    se: 0.010000\n"), "{yaml}");
    }

    /// #2817: the same unknown reason through `FitResult::from_json_str` (the
    /// `--output-format json` reader), and the plain serde read of that file — which the
    /// strict derive would refuse.
    #[test]
    fn fit_result_json_with_an_unknown_held_reason_reads_back() {
        use crate::types::HeldReason;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fit.json");
        crate::io::output::write_result_json(&two_held_thetas(), path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        inject_future_reason(&mut v);
        let text = serde_json::to_string(&v).unwrap();

        let back = FitResult::from_json_str(&text)
            .expect("an unknown held reason must not fail the FitResult JSON");
        let reasons: Vec<HeldReason> = back.held_at_bound.iter().map(|h| h.reason).collect();
        assert_eq!(reasons, [HeldReason::Unknown, HeldReason::ProfiledBound]);

        let reason: HeldReason = serde_json::from_str("\"some_future_reason\"").unwrap();
        assert_eq!(reason, HeldReason::Unknown);
        // Known tokens are unaffected, and `Unknown` writes the token it reads back from.
        for r in [
            HeldReason::ProfiledBound,
            HeldReason::CollapsedAtGuard,
            HeldReason::Unknown,
        ] {
            let tok = serde_json::to_string(&r).unwrap();
            assert_eq!(tok, format!("\"{}\"", r.as_str()));
            assert_eq!(serde_json::from_str::<HeldReason>(&tok).unwrap(), r);
        }
    }

    /// Rename the `kind` of `held_at_bound[0]` in a fit JSON value to a family no build
    /// knows.
    fn inject_future_kind(v: &mut serde_json::Value) {
        assert_eq!(v["held_at_bound"][0]["kind"], "Theta", "{v}");
        v["held_at_bound"][0]["kind"] = "SomeFutureKind".into();
    }

    /// #2817 (review r1 row 2): a `held_at_bound` entry of a parameter family this build
    /// does not know is dropped on load, and the rest of the bundle and the list load, both
    /// through `load_fit` and through `FitResult::from_json_str`. Without the lenient read,
    /// the strict `ParameterKind` derive fails both `expect`s.
    #[test]
    fn an_unknown_held_kind_is_dropped_and_the_rest_loads() {
        use crate::types::{HeldReason, ParameterKind};
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        let r = two_held_thetas();
        let src = dir.path().join("known.fitrx");
        save_fit(&r, &p, "model source\n", &src, SaveFitOptions::default()).unwrap();
        let dst = dir.path().join("future-kind.fitrx");
        rewrite_bundle(&src, &dst, |name, body| {
            Some(if name == "fit.json" {
                edit_json(body, inject_future_kind)
            } else {
                body
            })
        });
        let survivors = |list: &[crate::types::HeldAtBound]| {
            list.iter()
                .map(|h| (h.name.clone(), h.kind, h.index, h.reason))
                .collect::<Vec<_>>()
        };
        let want = [(
            "KA".to_string(),
            ParameterKind::Theta,
            2,
            HeldReason::ProfiledBound,
        )];
        let loaded = load_fit(&dst)
            .expect("an unknown held kind must not fail the bundle")
            .fit;
        assert_eq!(survivors(&loaded.held_at_bound), want);

        let json = dir.path().join("fit.json");
        crate::io::output::write_result_json(&r, json.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        inject_future_kind(&mut v);
        let back = FitResult::from_json_str(&serde_json::to_string(&v).unwrap())
            .expect("an unknown held kind must not fail the FitResult JSON");
        assert_eq!(survivors(&back.held_at_bound), want);

        // The negative: every known kind survives the lenient read.
        let mut all = two_held_thetas();
        for (k, kind) in [
            ParameterKind::Omega,
            ParameterKind::Sigma,
            ParameterKind::Kappa,
        ]
        .into_iter()
        .enumerate()
        {
            let mut h = all.held_at_bound[0].clone();
            h.kind = kind;
            h.index = k;
            all.held_at_bound.push(h);
        }
        let text = serde_json::to_string(&all).unwrap();
        assert_eq!(
            FitResult::from_json_str(&text).unwrap().held_at_bound,
            all.held_at_bound
        );
    }

    /// `final_gradient_source` has to survive the checkpoint alongside the vector
    /// it labels (#997 §1). A bundle that kept the gradient and dropped the label
    /// is worse than one that kept neither: the reader then has a plausible vector
    /// and no way to tell an optimizer's own stationarity claim from an
    /// after-the-fact finite-difference check on a run that had none.
    ///
    /// Pinned with `"finite_difference"` rather than `"optimizer"` — the label a
    /// `Default`-ing or hard-coded mutation would not produce by accident.
    #[test]
    fn final_gradient_and_its_source_round_trip_through_fitrx() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grad.fitrx");
        let mut r = minimal_fit_result();
        r.final_gradient = Some(vec![0.25, -1.5, 3.0]);
        r.final_gradient_source = Some("finite_difference".to_string());
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let l = load_fit(&path).unwrap().fit;
        assert_eq!(l.final_gradient, Some(vec![0.25, -1.5, 3.0]));
        assert_eq!(
            l.final_gradient_source.as_deref(),
            Some("finite_difference")
        );
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
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
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

    /// Doubles no six-decimal text can hold: a sum with no short decimal, a value smaller
    /// than the last place, one with eleven significant digits, π, and one far above one.
    const AWKWARD: [f64; 5] = [
        0.1 + 0.2,
        3e-7,
        -1.234_567_891_23e-5,
        std::f64::consts::PI,
        1e20,
    ];

    /// #1847 / #1631: every CSV double in a bundle reads back as the bits it was written
    /// from — the EBEs, κ̂, the per-subject objective, the predictions and the conditional
    /// distribution — so a reloaded fit holds the in-memory EBEs `run_covariance` compares
    /// against the cold solve. Mutation: `fmt_f64` back to six decimals → dies on the first
    /// EBE; `{}` instead of `{:?}` → still green (both round-trip), which is why the
    /// spelling is pinned separately in `fmt_f64_is_shortest_round_trip_text`.
    #[test]
    fn roundtrip_csv_doubles_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exact.fitrx");
        let mut r = minimal_fit_result();
        for (i, s) in r.subjects.iter_mut().enumerate() {
            s.eta = DVector::from_vec(vec![AWKWARD[i], AWKWARD[i + 2]]);
            s.ofv_contribution = AWKWARD[i + 3];
            for (j, y) in s.ipred.iter_mut().enumerate() {
                *y = AWKWARD[(i + j) % AWKWARD.len()];
            }
        }
        r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.05]));
        r.kappa_names = vec!["kappa_CL".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        r.shrinkage_kappa = vec![0.1];
        r.ebe_kappas = vec![
            vec![
                DVector::from_vec(vec![AWKWARD[0]]),
                DVector::from_vec(vec![AWKWARD[1]]),
            ],
            vec![DVector::from_vec(vec![AWKWARD[2]])],
        ];
        r.cond_dist = Some(CondDist {
            cond_mean: vec![vec![AWKWARD[3], AWKWARD[1]], vec![AWKWARD[2], AWKWARD[0]]],
            cond_sd: vec![vec![AWKWARD[1], AWKWARD[3]], vec![AWKWARD[4], AWKWARD[2]]],
            samples: vec![Vec::new(), Vec::new()],
            shrinkage: vec![0.2, 0.3],
            nsamp: 500,
            burnin: 100,
        });
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let l = load_fit(&path).unwrap().fit;

        let bits = |xs: &mut dyn Iterator<Item = f64>| xs.map(f64::to_bits).collect::<Vec<_>>();
        for (a, b) in l.subjects.iter().zip(&r.subjects) {
            assert_eq!(
                bits(&mut a.eta.iter().copied()),
                bits(&mut b.eta.iter().copied()),
                "eta"
            );
            assert_eq!(
                a.ofv_contribution.to_bits(),
                b.ofv_contribution.to_bits(),
                "ofv"
            );
            assert!(!a.ipred.is_empty(), "premise: predictions are read back");
            assert_eq!(
                bits(&mut a.ipred.iter().copied()),
                bits(&mut b.ipred.iter().copied()),
                "ipred"
            );
        }
        let kappas = |f: &FitResult| {
            bits(
                &mut f
                    .ebe_kappas
                    .iter()
                    .flatten()
                    .flat_map(|k| k.iter().copied()),
            )
        };
        assert_eq!(kappas(&l), kappas(&r), "kappa");
        let (lc, rc) = (l.cond_dist.unwrap(), r.cond_dist.unwrap());
        assert_eq!(
            bits(&mut lc.cond_mean.iter().flatten().copied()),
            bits(&mut rc.cond_mean.iter().flatten().copied()),
            "cond_mean"
        );
        assert_eq!(
            bits(&mut lc.cond_sd.iter().flatten().copied()),
            bits(&mut rc.cond_sd.iter().flatten().copied()),
            "cond_sd"
        );
    }

    /// The spelling: shortest round-trip text, an exponent far from one, the empty cell for
    /// `NaN` and the `inf` tokens kept. Mutation: `{}` → the `1e-7` / `1e20` rows die;
    /// six decimals → every finite row dies; drop the `NaN` arm → the empty-cell row dies.
    #[test]
    fn fmt_f64_is_shortest_round_trip_text() {
        assert_eq!(fmt_f64(0.1), "0.1");
        assert_eq!(fmt_f64(1.0), "1.0");
        assert_eq!(fmt_f64(1e-7), "1e-7");
        assert_eq!(fmt_f64(1e20), "1e20");
        assert_eq!(fmt_f64(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(fmt_f64(f64::NAN), "");
        assert_eq!(fmt_f64(f64::INFINITY), "inf");
        assert_eq!(fmt_f64(f64::NEG_INFINITY), "-inf");
        for x in AWKWARD {
            assert_eq!(fmt_f64(x).parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
    }

    /// COND_MODE is checked against the EBE by value: a bundle written before #1847 (six
    /// decimals), one written now (shortest text) and one in another writer's spelling all
    /// load, while a mode one ULP off the EBE is still refused. Mutation: compare the text
    /// with `fmt_f64` (the pre-#1847 check) → the six-decimal and `1e-1` rows die.
    #[test]
    fn parse_conddist_csv_compares_cond_mode_by_value() {
        let fit = minimal_fit_result();
        for (mode_cl, mode_v) in [("0.100000", "0.200000"), ("0.1", "0.2"), ("1e-1", "2.0e-1")] {
            let csv = format!(
                "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
                 S1,eta_CL,0.11,0.05,{mode_cl}\nS1,eta_V,-0.19,0.06,{mode_v}\n\
                 S2,eta_CL,0.29,0.07,{mode_cl}\nS2,eta_V,0.41,0.08,{mode_v}\n"
            );
            parse_conddist_csv(&csv, &fit)
                .unwrap_or_else(|e| panic!("COND_MODE spelled {mode_cl:?}: {e:?}"));
        }
        let off = format!("{:?}", 0.1_f64.next_up());
        let csv = format!(
            "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
             S1,eta_CL,0.11,0.05,{off}\nS1,eta_V,-0.19,0.06,0.2\n\
             S2,eta_CL,0.29,0.07,0.1\nS2,eta_V,0.41,0.08,0.2\n"
        );
        let err = parse_conddist_csv(&csv, &fit).unwrap_err();
        assert!(
            matches!(&err, FitrxError::Corrupt(m) if m.contains("COND_MODE")),
            "one ULP off: expected COND_MODE mismatch, got {err:?}"
        );
    }

    /// #2092 review Q1: a `conddist.csv` that ferx-r's `ferx_save_fit` writes loads. R
    /// spells each double as the shortest of `%.15g` / `%.16g` / `%.17g` that reads back,
    /// so a value is full precision and an exponent has two digits; the cells below are R's
    /// own output for these doubles (`Rscript`, `.fitrx_double_text`'s rule). Before #1847
    /// the check compared that text with the EBE at six decimals, so every R bundle whose
    /// EBEs were not six-decimal numbers was refused as corrupt. The second CSV is R's `-0`
    /// against an EBE of `+0`, which agree (review nit). Mutations: compare the text with
    /// `fmt_f64` → the R-spelling half dies (`e-07` vs `e-7`); compare bits → the `-0` half
    /// dies; the pre-#1847 six-decimal text compare → the R-spelling half dies.
    #[test]
    fn parse_conddist_csv_accepts_r_written_spelling() {
        let mut fit = minimal_fit_result();
        fit.subjects[0].eta = DVector::from_vec(vec![0.027974420148319645, 1.2345678901234568e-7]);
        fit.subjects[1].eta = DVector::from_vec(vec![-0.06290220762098092, 0.0]);
        let csv = |s2_v: &str| {
            format!(
                "ID,ETA,COND_MEAN,COND_SD,COND_MODE\n\
                 S1,eta_CL,0.11,0.05,0.027974420148319645\n\
                 S1,eta_V,-0.19,0.06,1.2345678901234568e-07\n\
                 S2,eta_CL,0.29,0.07,-0.06290220762098092\n\
                 S2,eta_V,0.41,0.08,{s2_v}\n"
            )
        };
        parse_conddist_csv(&csv("0"), &fit)
            .unwrap_or_else(|e| panic!("R's full-precision spelling: {e:?}"));
        parse_conddist_csv(&csv("-0"), &fit)
            .unwrap_or_else(|e| panic!("R's -0 against an EBE of +0: {e:?}"));
    }

    /// #1705: a SIR fit's kappa intervals survive the bundle, per kappa and in
    /// order. Two kappas with distinct intervals, so a wire that dropped the
    /// field on either the save or the load side, or reordered it, fails here.
    #[test]
    fn roundtrip_keeps_sir_ci_kappa() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sir_kappa.fitrx");
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_row_slice(2, 2, &[156.0, 0.0, 0.0, 0.05]));
        r.kappa_names = vec!["KAPPA_ARM".into(), "KAPPA_CL".into()];
        r.kappa_fixed = vec![false, false];
        r.kappa_init_as_sd = vec![false; 2];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 2];
        r.shrinkage_kappa = vec![0.1, 0.1];
        r.sir_ess = Some(143.2);
        r.sir_ci_theta = Some(vec![(1.0, 2.0)]);
        r.sir_ci_kappa = Some(vec![(83.2, 367.4), (0.02, 0.12)]);
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.sir_ci_kappa, r.sir_ci_kappa);
    }

    /// A SIR record with every field, and every enum, off its default — so a
    /// wire that dropped a field, or decoded a token to the default, differs.
    fn off_default_sir_settings() -> crate::estimation::sir::SirSettings {
        crate::estimation::sir::SirSettings {
            samples: 321,
            resamples: 123,
            seed: 4242,
            df: 3.0,
            scale: crate::types::SirScale::Natural,
            keep_samples: true,
            scoring: off_default_scoring_settings(),
        }
    }

    /// A scoring record with every field, and every enum, off its default (#426).
    fn off_default_scoring_settings() -> crate::ScoringSettings {
        crate::ScoringSettings {
            inner_maxiter: 17,
            inner_tol: 3e-4,
            inner_restarts: 4,
            inner_screen: 7,
            inner_twin_seed: false,
            mu_referencing: false,
            n_agq: 5,
            inner_optimizer: crate::types::InnerOptimizer::NelderMead,
            ebe_warm_start: true,
            ode_reltol: 1e-7,
            ode_abstol: 1e-9,
            ode_max_steps: 777,
            ode_method: crate::ode::OdeMethod::Rodas5P,
            ode_stiff_abort_after: Some(9),
            ode_auto_switch: false,
            gradient_method: crate::types::GradientMethod::Fd,
            nn_l2_lambda: Some(0.25),
            nn_smooth_lambda: Some(0.5),
        }
    }

    fn sir_fit_with_settings() -> FitResult {
        let mut r = minimal_fit_result();
        r.sir_ess = Some(50.0);
        r.sir_ci_theta = Some(vec![(1.0, 2.0)]);
        r.sir_seed = Some(4242);
        r.sir_settings = Some(off_default_sir_settings());
        r
    }

    /// T9 (#1758): the SIR record survives the bundle, field for field.
    /// Mutations: drop any wire field (it fails to compile — the wire is
    /// exhaustive in both directions), or map a token to the default enum (the
    /// equality dies, since every enum here is off-default).
    #[test]
    fn roundtrip_keeps_sir_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sir_settings.fitrx");
        let r = sir_fit_with_settings();
        assert_ne!(
            r.sir_settings,
            Some(crate::estimation::sir::SirSettings::default())
        );
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.sir_settings, r.sir_settings);
        assert_eq!(loaded.fit.sir_seed, Some(4242));
    }

    /// T9 (#1758): the enums travel as their `[fit_options]` tokens, the
    /// spelling a user reads in the model file. Mutation: drop a `rename_all` or
    /// the `rodas5p` rename (that variant's token changes).
    #[test]
    fn sir_settings_enums_travel_as_their_fit_options_tokens() {
        use crate::ode::OdeMethod;
        use crate::types::{InnerOptimizer, SirScale};
        for s in [SirScale::Packed, SirScale::Natural] {
            assert_eq!(enum_token(&s), s.label());
        }
        for m in [
            OdeMethod::Rk45,
            OdeMethod::Rosenbrock23,
            OdeMethod::Rodas4,
            OdeMethod::Rodas5P,
            OdeMethod::Vern7,
            OdeMethod::Auto,
        ] {
            assert_eq!(enum_token(&m), m.as_str());
            assert_eq!(OdeMethod::parse(&enum_token(&m)), Some(m));
        }
        // The `[fit_options] inner_optimizer` tokens (`parse_fit_option`).
        for (o, tok) in [
            (InnerOptimizer::Auto, "auto"),
            (InnerOptimizer::Bfgs, "bfgs"),
            (InnerOptimizer::Lbfgs, "lbfgs"),
            (InnerOptimizer::NelderMead, "nelder_mead"),
        ] {
            assert_eq!(enum_token(&o), tok);
        }
    }

    /// T9 (#1758, #1382): an unrecognised token is `Corrupt`, naming the
    /// field, not a silent degrade to the default scale. The Rust variant name
    /// (`Natural`) is the likeliest wrong spelling. Mutation: decode an unknown
    /// token to `SirScale::default()` — the load succeeds.
    #[test]
    fn an_unknown_sir_settings_token_is_corrupt() {
        let r = sir_fit_with_settings();
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(value["sir"]["settings"]["scale"], "natural");
        value["sir"]["settings"]["scale"] = serde_json::json!("Natural");
        let wire: FitWire = serde_json::from_value(value).unwrap();
        match wire_to_fit_result(wire, r.subjects.clone(), Vec::new()) {
            Err(FitrxError::Corrupt(msg)) => {
                assert!(msg.contains("sir.settings.scale"), "{msg}");
                assert!(msg.contains("\"Natural\""), "{msg}");
            }
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }

    /// Load `sir_fit_with_settings()` with `edit` applied to its wire JSON.
    fn load_edited_sir_wire(
        edit: impl FnOnce(&mut serde_json::Value),
    ) -> Result<FitResult, FitrxError> {
        let r = sir_fit_with_settings();
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        edit(&mut value);
        let wire: FitWire = serde_json::from_value(value).unwrap();
        wire_to_fit_result(wire, r.subjects.clone(), Vec::new())
    }

    /// #1767 finding 6: a recorded value outside the domain `[fit_options]` enforces
    /// for the same key is `Corrupt`, naming the field and the value — `run_sir`
    /// would otherwise adopt it past the parser's gate. Each row also loads the
    /// boundary value, so the gate is straddled: a check that rejected everything
    /// (or an off-by-one bound) fails the accept half, a deleted check the reject
    /// half. Mutations: delete any one check; `> 1.0` for `df`; `1..MAX` for `n_agq`.
    #[test]
    fn sir_settings_out_of_domain_values_are_corrupt() {
        let max_agq = crate::estimation::agq::MAX_AGQ_NODES;
        let agq_shows = format!("got {}", max_agq + 1);
        let rows: [(&str, serde_json::Value, serde_json::Value, &str); 6] = [
            ("df", serde_json::json!(0.5), serde_json::json!(1.0), "0.5"),
            (
                "ode_reltol",
                serde_json::json!(0.0),
                serde_json::json!(1e-12),
                "got 0",
            ),
            (
                "ode_abstol",
                serde_json::json!(-1e-6),
                serde_json::json!(1e-12),
                "-0.000001",
            ),
            (
                "ode_max_steps",
                serde_json::json!(0),
                serde_json::json!(1),
                "got 0",
            ),
            (
                "n_agq",
                serde_json::json!(max_agq + 1),
                serde_json::json!(max_agq),
                &agq_shows,
            ),
            (
                "ode_stiff_abort_after",
                serde_json::json!(0),
                serde_json::json!(1),
                "got 0",
            ),
        ];
        for (field, reject, accept, shows) in rows {
            let rejected = load_edited_sir_wire(|v| v["sir"]["settings"][field] = reject.clone());
            match rejected {
                Err(FitrxError::Corrupt(msg)) => {
                    assert!(
                        msg.contains(&format!("sir.settings.{field} ")),
                        "{field}: {msg}"
                    );
                    assert!(msg.contains(shows), "{field}: {msg}");
                }
                other => panic!("{field}: expected Corrupt, got {:?}", other.map(|_| ())),
            }
            let loaded = load_edited_sir_wire(|v| v["sir"]["settings"][field] = accept.clone())
                .unwrap_or_else(|e| panic!("{field} = {accept} must load: {e:?}"));
            assert!(loaded.sir_settings.is_some(), "{field}");
        }
    }

    /// #1770: `save_fit` refuses a fit whose recorded settings `load_fit` would refuse,
    /// before creating the file, and a fit at each boundary value writes and loads. So a
    /// bundle is never "written, then load fails". Each row runs on both records: `sir.settings`
    /// and the top-level `scoring_settings` every fit carries. Mutations: drop
    /// `check_writable` (the reject half writes, then the reload is `Corrupt`); drop either
    /// record's check (that record's rows write); move the check after `File::create` (the
    /// file exists).
    #[test]
    fn save_fit_refuses_settings_the_reader_would_reject() {
        type Edit = fn(&mut crate::estimation::sir::SirSettings, bool);
        let rows: [(&str, Edit, &str); 7] = [
            ("df", |s, bad| s.df = if bad { 0.5 } else { 1.0 }, "got 0.5"),
            (
                "n_agq",
                |s, bad| s.scoring.n_agq = if bad { 0 } else { 1 },
                "got 0",
            ),
            (
                "ode_max_steps",
                |s, bad| s.scoring.ode_max_steps = if bad { 0 } else { 1 },
                "got 0",
            ),
            (
                "ode_stiff_abort_after",
                |s, bad| s.scoring.ode_stiff_abort_after = Some(if bad { 0 } else { 1 }),
                "got 0",
            ),
            (
                "ode_reltol",
                |s, bad| s.scoring.ode_reltol = if bad { f64::NAN } else { 1e-300 },
                "got NaN",
            ),
            (
                "ode_abstol",
                |s, bad| s.scoring.ode_abstol = if bad { 0.0 } else { f64::MIN_POSITIVE },
                "got 0",
            ),
            (
                "gradient_method",
                |s, bad| {
                    s.scoring.gradient_method = if bad {
                        crate::types::GradientMethod::Ad
                    } else {
                        crate::types::GradientMethod::Auto
                    }
                },
                "got \"ad\"",
            ),
        ];
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        for (field, edit, shows) in rows {
            for block in ["sir.settings", "scoring_settings"] {
                // `df` is SIR's alone.
                if field == "df" && block == "scoring_settings" {
                    continue;
                }
                let with = |bad: bool| {
                    let mut r = sir_fit_with_settings();
                    let mut st = off_default_sir_settings();
                    edit(&mut st, bad);
                    if block == "sir.settings" {
                        r.sir_settings = Some(st);
                    } else {
                        r.scoring_settings = Some(st.scoring);
                    }
                    r
                };
                let path = dir.path().join(format!("{block}-{field}.fitrx"));
                match save_fit(&with(true), &p, "src\n", &path, SaveFitOptions::default()) {
                    Err(FitrxError::Unwritable(msg)) => {
                        assert!(msg.contains(&format!("{block}.{field} ")), "{msg}");
                        assert!(msg.contains(shows), "{block}.{field}: {msg}");
                    }
                    other => panic!("{block}.{field}: expected Unwritable, got {other:?}"),
                }
                assert!(
                    !path.exists(),
                    "{block}.{field}: a refused fit wrote a file"
                );

                let good = with(false);
                save_fit(&good, &p, "src\n", &path, SaveFitOptions::default())
                    .unwrap_or_else(|e| panic!("{block}.{field}: boundary must write: {e}"));
                let loaded = load_fit(&path)
                    .unwrap_or_else(|e| panic!("{block}.{field}: boundary must load: {e}"))
                    .fit;
                assert_eq!(loaded.sir_settings, good.sir_settings, "{block}.{field}");
                assert_eq!(
                    loaded.scoring_settings, good.scoring_settings,
                    "{block}.{field}"
                );
            }
        }
    }

    /// #1770: `sir_settings.seed` is canonical. A caller who edits it after the fit (as
    /// the SIR docs suggest, to change a re-run) and saves gets a bundle that loads, with
    /// `sir_seed` following the record — the reader refuses the pair when they disagree.
    /// The control: with no record, `sir_seed` is written as held (a pre-#1758 fit).
    /// Mutation: write `sir_seed: r.sir_seed` again (the load is `Corrupt`).
    #[test]
    fn an_edited_sir_settings_seed_writes_a_bundle_that_loads() {
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        let path = dir.path().join("seed.fitrx");
        let mut r = sir_fit_with_settings();
        r.sir_settings.as_mut().unwrap().seed = 99;
        assert_eq!(
            r.sir_seed,
            Some(4242),
            "premise: the pair disagrees in memory"
        );
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap().fit;
        assert_eq!(loaded.sir_seed, Some(99));
        assert_eq!(loaded.sir_settings.unwrap().seed, 99);

        let mut legacy = sir_fit_with_settings();
        legacy.sir_settings = None;
        legacy.sir_seed = Some(7);
        save_fit(&legacy, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        assert_eq!(load_fit(&path).unwrap().fit.sir_seed, Some(7));
    }

    /// #1819: the multivariate-normal proposal (`df = ∞`) travels as the `[fit_options]`
    /// token `"normal"` — JSON has no number for it, and serde_json would write `null`,
    /// which no reader accepts. Mutation: write `df` as a plain number again (the token
    /// assertion dies, and the reload is `Corrupt` on the `null`).
    #[test]
    fn a_normal_sir_proposal_round_trips_as_its_fit_options_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sir_normal.fitrx");
        let mut r = sir_fit_with_settings();
        r.sir_settings.as_mut().unwrap().df = f64::INFINITY;
        let value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(value["sir"]["settings"]["df"], "normal");
        // The control: a finite df stays a JSON number, so the change is additive.
        let finite = serde_json::to_value(build_fit_wire(&sir_fit_with_settings())).unwrap();
        assert_eq!(finite["sir"]["settings"]["df"], serde_json::json!(3.0));

        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.sir_settings, r.sir_settings);
        assert_eq!(loaded.fit.sir_settings.unwrap().df, f64::INFINITY);
    }

    /// #1819: on read, `"normal"` and `"mvn"` (any case, as `[fit_options]` reads them)
    /// are `df = ∞`; `null`, an unknown token and a df below 1 are `Corrupt` naming
    /// `sir.settings.df`. Mutations: drop the `mvn` arm (its row fails to load); accept
    /// any string as ∞ (`"gaussian"` loads); map `null` to ∞ (it loads).
    #[test]
    fn sir_settings_df_tokens_and_their_refusals() {
        for tok in ["normal", "mvn", "NORMAL"] {
            let loaded = load_edited_sir_wire(|v| v["sir"]["settings"]["df"] = tok.into())
                .unwrap_or_else(|e| panic!("{tok} must load: {e:?}"));
            assert_eq!(loaded.sir_settings.unwrap().df, f64::INFINITY, "{tok}");
        }
        for (bad, shows) in [
            (serde_json::Value::Null, "got null"),
            (serde_json::json!("gaussian"), "got \"gaussian\""),
            (serde_json::json!(0.5), "got 0.5"),
        ] {
            match load_edited_sir_wire(|v| v["sir"]["settings"]["df"] = bad.clone()) {
                Err(FitrxError::Corrupt(msg)) => {
                    assert!(msg.contains("sir.settings.df "), "{bad}: {msg}");
                    assert!(msg.contains(shows), "{bad}: {msg}");
                }
                other => panic!("{bad}: expected Corrupt, got {:?}", other.map(|_| ())),
            }
        }
    }

    /// #1767 finding 7: `sir_seed` and `sir.settings.seed` are two copies of one seed;
    /// a bundle where they disagree, or where the record has no `sir_seed` beside it,
    /// is `Corrupt`. The control — the two equal — loads (that is
    /// `roundtrip_keeps_sir_settings`). Mutation: delete the check (both load).
    #[test]
    fn a_sir_seed_disagreeing_with_the_record_is_corrupt() {
        for (seed, shows) in [
            (serde_json::json!(4243), "Some(4243)"),
            (serde_json::Value::Null, "None"),
        ] {
            match load_edited_sir_wire(|v| v["sir_seed"] = seed.clone()) {
                Err(FitrxError::Corrupt(msg)) => {
                    assert!(msg.contains("sir.settings.seed 4242"), "{msg}");
                    assert!(msg.contains(shows), "{msg}");
                }
                other => panic!(
                    "sir_seed {seed}: expected Corrupt, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
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
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        r.shrinkage_kappa = vec![0.1];
        r.kappa_weights = vec![Some("NARM".into())];
        r.kappa_weight_typical = vec![Some(200.0)];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.kappa_weights, vec![Some("NARM".to_string())]);
        assert_eq!(loaded.fit.kappa_weight_typical, vec![Some(200.0)]);
    }

    /// #1643: each kappa's scale rides the bundle, by kappa index, so a reloaded
    /// fit still prints an SD for an additive kappa. Two kappas of different
    /// types, so a wire that dropped or reordered the field fails here.
    #[test]
    fn roundtrip_kappa_param_types() {
        use crate::types::EtaParamType::{Additive, LogNormal};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kappa_types.fitrx");
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_row_slice(2, 2, &[156.0, 0.0, 0.0, 0.05]));
        r.kappa_names = vec!["KAPPA_ARM".into(), "KAPPA_CL".into()];
        r.kappa_fixed = vec![false, false];
        r.kappa_init_as_sd = vec![false; 2];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 2];
        r.shrinkage_kappa = vec![0.1, 0.1];
        r.kappa_param_types = vec![Additive, LogNormal];
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.kappa_param_types, vec![Additive, LogNormal]);
    }

    /// A bundle whose `iov.omega_iov` is not square, or whose dimension is not
    /// `iov.kappa_names`' length, is `Corrupt` (#1833), as `omega.matrix` already is.
    ///
    /// Measured before on `160cc9a3`: a saved IOV fit with `iov.omega_iov` edited to 1×2
    /// loaded `Ok`, and the first post-hoc call panicked in its Cholesky. The control arm
    /// decodes the unedited wire, so the two edited arms differ from it by the shape alone.
    /// Mutation — delete the new check in `validate_parallel_lengths` and both edited arms
    /// decode `Ok`.
    #[test]
    fn fit_wire_rejects_a_non_square_or_mis_sized_omega_iov() {
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.05]));
        r.kappa_names = vec!["KAPPA_CL".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        r.shrinkage_kappa = vec![0.1];
        let value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        let decode = |v: serde_json::Value| {
            let w: FitWire = serde_json::from_value(v).unwrap();
            wire_to_fit_result(w, Vec::new(), Vec::new())
        };
        let ok = decode(value.clone()).expect("the unedited wire decodes");
        assert_eq!(ok.omega_iov.as_ref().map(|m| m.shape()), Some((1, 1)));

        for (rows, cols, data, want) in [
            (
                1,
                2,
                vec![0.05, 0.0],
                "iov.omega_iov is 1×2; expected square",
            ),
            (
                2,
                2,
                vec![0.05, 0.0, 0.0, 0.05],
                "iov.omega_iov dim (2) does not match iov.kappa_names (1)",
            ),
        ] {
            let mut v = value.clone();
            v["iov"]["omega_iov"] = serde_json::json!({"rows": rows, "cols": cols, "data": data});
            match decode(v) {
                Err(FitrxError::Corrupt(msg)) => assert!(msg.contains(want), "{msg}"),
                other => panic!("{rows}×{cols}: expected Corrupt({want:?}), got {other:?}"),
            }
        }
    }

    /// Bindings that exercise every shape the wire carries: a one-level block
    /// under `none`, a `sum_to_zero_within` block with two groups, a `ref` block,
    /// and two covariates whose statistics are not exactly representable in
    /// decimal (so a lossy float path shows up as a bit difference).
    fn sample_data_bindings() -> crate::parser::model_parser::DataBindings {
        use crate::parser::model_parser::{DataBindings, LevelBinding, LevelContrast};
        let level = |labels: &[&str], groups: &[usize], contrast| LevelBinding {
            labels: labels.iter().map(|s| s.to_string()).collect(),
            groups: groups.to_vec(),
            contrast,
        };
        let mut b = DataBindings::default();
        b.levels.insert(
            "A_ONE".into(),
            level(&["STUDY=1"], &[0], LevelContrast::Unconstrained),
        );
        b.levels.insert(
            "B_WITHIN".into(),
            level(
                &[
                    "STUDY=1,TIME=1",
                    "STUDY=1,TIME=2",
                    "STUDY=2,TIME=1",
                    "STUDY=2,TIME=2",
                ],
                &[0, 0, 1, 1],
                LevelContrast::SumToZeroWithin,
            ),
        );
        b.levels.insert(
            "C_REF".into(),
            level(&["ARM=0", "ARM=1"], &[0, 0], LevelContrast::Ref),
        );
        b.covariate_stats.insert(
            "WT".into(),
            CovariateSummary {
                median: 71.475,
                mean: 215.2 / 3.0,
                min: 0.1 + 0.2,
                max: 90.2,
                mode: 61.3,
                levels: vec![0.1 + 0.2, 61.3, 90.2],
            },
        );
        b.covariate_stats.insert(
            "AGE".into(),
            CovariateSummary {
                median: 1.0 / 3.0,
                mean: std::f64::consts::PI,
                min: 1e-300,
                max: 1e300,
                mode: 2.0 / 3.0,
                levels: vec![1e-300, 1.0 / 3.0, 1e300],
            },
        );
        b
    }

    /// #1621 T3. The data-derived bindings round-trip through a `.fitrx`
    /// bundle exactly (`f64` fields compared by `==` through `PartialEq`, so a
    /// lossy float path fails), and `fit.json` carries them with their blocks
    /// in sorted order and the contrast as its DSL token.
    ///
    /// Mutations — omit the field from `build_fit_wire` (write
    /// `DataBindingsWire::default()`) or map it to `Default` on load: the
    /// loaded bindings are empty and the `assert_eq!` dies.
    #[test]
    fn roundtrip_data_bindings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bindings.fitrx");
        let mut r = minimal_fit_result();
        r.data_bindings = sample_data_bindings();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let text = {
            use std::io::Read as _;
            let mut f = archive.by_name("fit.json").unwrap();
            let mut s = String::new();
            f.read_to_string(&mut s).unwrap();
            s
        };
        let wire: serde_json::Value = serde_json::from_str(&text).unwrap();
        let lv = &wire["data_bindings"]["levels"];
        assert_eq!(lv["A_ONE"]["contrast"], "none");
        assert_eq!(lv["B_WITHIN"]["contrast"], "sum_to_zero_within");
        assert_eq!(lv["B_WITHIN"]["groups"], serde_json::json!([0, 0, 1, 1]));
        assert_eq!(lv["C_REF"]["contrast"], "ref");
        // `BTreeMap` on the wire: blocks in sorted order, whatever the
        // in-memory `HashMap` iterates in.
        let (a, b, c) = (
            text.find("\"A_ONE\"").unwrap(),
            text.find("\"B_WITHIN\"").unwrap(),
            text.find("\"C_REF\"").unwrap(),
        );
        assert!(a < b && b < c, "{text}");
        let stats = &text[text.find("\"covariate_stats\"").unwrap()..];
        assert!(stats.find("\"AGE\"").unwrap() < stats.find("\"WT\"").unwrap());

        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.data_bindings, r.data_bindings);
        assert_eq!(
            loaded.fit.data_bindings.covariate_stats["AGE"]
                .mean
                .to_bits(),
            std::f64::consts::PI.to_bits()
        );
    }

    /// #1685 T13, #1783 T6. The reader settings, the population fingerprint and the
    /// occasion rule survive `save_fit` → `load_fit` unchanged; a fit without them
    /// writes none of the keys, and a `fit.json` from before #1685 / #1783 (the keys
    /// deleted) loads them as `None`.
    ///
    /// Mutations — omit either field from `build_fit_wire`, or drop it on load: the
    /// round-trip equality dies; drop `#[serde(default)]` from either wire field:
    /// the old-bundle `from_value` fails.
    #[test]
    fn roundtrip_reader_settings_and_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("provenance.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        let mut r = minimal_fit_result();
        let mut settings = crate::api::ReaderSettings::default();
        settings.column_map = vec![("DV".into(), "CONC".into())];
        settings.iov_column = Some("OCC".into());
        settings.ignore_exprs = vec!["TIME > 24".into()];
        settings.ignore_subjects = vec!["3".into()];
        settings.fallback_columns = Some(vec!["WT".into()]);
        r.reader_settings = Some(settings);
        r.population_fingerprint = Some(crate::types::PopulationFingerprint::of(&p));
        // #1783: the occasion rule, in its one variant that carries data.
        r.iov_occasion = Some(crate::types::IovOccasionRule::TimeWindows(vec![24.0, 48.5]));
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.reader_settings, r.reader_settings);
        assert_eq!(loaded.fit.population_fingerprint, r.population_fingerprint);
        assert_eq!(loaded.fit.iov_occasion, r.iov_occasion);

        const KEYS: [&str; 3] = ["reader_settings", "population_fingerprint", "iov_occasion"];
        let plain = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        for key in KEYS {
            assert!(plain.get(key).is_none(), "{key}: written when absent");
        }
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(
            value["iov_occasion"],
            serde_json::json!({ "time_windows": [24.0, 48.5] }),
            "the documented spelling"
        );
        for key in KEYS {
            assert!(
                value.as_object_mut().unwrap().remove(key).is_some(),
                "{key}: must be written when set, or the removal tests nothing"
            );
        }
        let old: FitWire = serde_json::from_value(value).unwrap();
        assert!(old.reader_settings.is_none() && old.population_fingerprint.is_none());
        assert!(old.iov_occasion.is_none());
    }

    /// #1815 T1: the optimizer's packed vector round-trips through the bundle bit
    /// for bit — including a non-finite element, which goes out as `null` and comes
    /// back as `NaN` like every other float vector in `fit.json`. A fit without one
    /// writes no key, and a bundle from before #1815 (key deleted) loads `None`.
    ///
    /// Mutations — read `packed_estimate: None` on load: the bit comparison dies;
    /// drop `skip_serializing_if`: the absent-key assert dies (written as `null`);
    /// drop the `opt_vec` adapter: the NaN element cannot be read back and
    /// `load_fit` fails.
    #[test]
    fn roundtrip_packed_estimate_bits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("packed.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        let packed = vec![
            0.1 + 0.2,
            -0.0,
            5e-324,
            -1.797_693_134_862_315_7e308,
            f64::NAN,
        ];
        let mut r = minimal_fit_result();
        r.packed_estimate = Some(packed.clone());
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let back = load_fit(&path)
            .unwrap()
            .fit
            .packed_estimate
            .expect("a bundle saved with a packed vector loads one");
        assert_eq!(back.len(), packed.len());
        for (i, (got, want)) in back.iter().zip(&packed).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "[{i}]: NaN must come back as NaN, got {got}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "[{i}]: {got} vs {want}");
            }
        }

        let plain = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        assert!(
            plain.get("packed_estimate").is_none(),
            "packed_estimate: written when absent"
        );
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(value["packed_estimate"][4], serde_json::Value::Null);
        assert!(
            value
                .as_object_mut()
                .unwrap()
                .remove("packed_estimate")
                .is_some(),
            "must be written when set, or the removal tests nothing"
        );
        let old: FitWire = serde_json::from_value(value).unwrap();
        assert!(old.packed_estimate.is_none());
    }

    /// #1765: a `[mixture]` fit's recorded override values survive `save_fit` →
    /// `load_fit` bit for bit; a fit without them writes no key; and a bundle from
    /// before #1765 (key deleted) loads `None`, with nothing else disturbed.
    ///
    /// Mutations — read `mixture_overrides: None` on load: the bit comparison dies;
    /// write `None`: the same; drop `skip_serializing_if`: the absent-key assert dies.
    #[test]
    fn roundtrip_mixture_overrides_bits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixture.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        let values = vec![0.097_248_853_500_388_42, 0.1 + 0.2, 5e-324];
        let mut r = minimal_fit_result();
        r.mixture_overrides = Some(values.clone());
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let back = load_fit(&path)
            .unwrap()
            .fit
            .mixture_overrides
            .expect("a bundle saved with override values loads them");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back), bits(&values));

        let plain = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        assert!(
            plain.get("mixture_overrides").is_none(),
            "mixture_overrides: written when absent"
        );
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert!(
            value
                .as_object_mut()
                .unwrap()
                .remove("mixture_overrides")
                .is_some(),
            "must be written when set, or the removal tests nothing"
        );
        let old: FitWire = serde_json::from_value(value).unwrap();
        assert!(old.mixture_overrides.is_none());
    }

    /// #2833 T5: the override descriptors and SEs round-trip; a bundle without them (what
    /// one written before #2833 is) loads empty / `None`; a descriptor list or SE vector
    /// that is not parallel to `mixture_overrides` is refused.
    ///
    /// Mutations — drop either wire field (write or read): the round-trip `assert_eq!`
    /// dies; drop `serde(default)`: the old bundle fails to load; drop either length check:
    /// the mismatched bundle loads.
    #[test]
    fn roundtrip_mixture_override_params_and_se() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.mixture_overrides = Some(vec![0.3, 0.7]);
        r.mixture_override_params = vec![
            crate::types::MixtureOverrideParam {
                name: "eta_CL_MIX2".into(),
                kind: crate::types::ParameterKind::Omega,
                index: 0,
                class: 2,
                packed_index: 6,
                fixed: false,
            },
            crate::types::MixtureOverrideParam {
                name: "prop_MIX3".into(),
                kind: crate::types::ParameterKind::Sigma,
                index: 0,
                class: 3,
                packed_index: 7,
                fixed: true,
            },
        ];
        r.se_mixture_overrides = Some(vec![0.064_975_8, 0.0]);
        let path = saved(&dir, &r);
        let back = load_fit(&path).unwrap().fit;
        assert_eq!(back.mixture_override_params, r.mixture_override_params);
        assert_eq!(back.se_mixture_overrides, r.se_mixture_overrides);

        // Written only when set.
        let plain = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        for key in ["mixture_override_params", "se_mixture_overrides"] {
            assert!(plain.get(key).is_none(), "{key}: written when absent");
        }

        // A bundle from before #2833: both keys gone, loads with the values alone.
        let old = dir.path().join("old.fitrx");
        rewrite_bundle(&path, &old, |name, body| {
            Some(if name == "fit.json" {
                edit_json(body, |f| {
                    let o = f.as_object_mut().unwrap();
                    assert!(o.remove("mixture_override_params").is_some());
                    assert!(o.remove("se_mixture_overrides").is_some());
                })
            } else {
                body
            })
        });
        let loaded = load_fit(&old).unwrap().fit;
        assert!(loaded.mixture_override_params.is_empty());
        assert!(loaded.se_mixture_overrides.is_none());
        assert_eq!(loaded.mixture_overrides, r.mixture_overrides);

        // Not parallel to the values: refused, each with its own message.
        for (key, short, msg) in [
            (
                "mixture_override_params",
                serde_json::json!([serde_json::to_value(&r.mixture_override_params[0]).unwrap()]),
                "mixture_override_params (1) does not match mixture_overrides (2)",
            ),
            (
                "se_mixture_overrides",
                serde_json::json!([0.1, 0.2, 0.3]),
                "se_mixture_overrides (3) does not match mixture_overrides (2)",
            ),
        ] {
            let edited = dir.path().join(format!("{key}.fitrx"));
            rewrite_bundle(&path, &edited, |name, body| {
                Some(if name == "fit.json" {
                    edit_json(body, |f| f[key] = short.clone())
                } else {
                    body
                })
            });
            assert_eq!(load_corrupt(load_fit(&edited)), msg);
        }
    }

    /// #1685 T4, probe D (re-measured at `045ca1b2`: `load_fit` returned 30 subjects
    /// for the fit's 29, and 60 doses for its 30). A bundle that records its reader
    /// settings re-reads `data.csv` with them — `ignore_subjects`, a dose-row filter,
    /// the level index columns from the fit's bindings — and the loaded population
    /// is the fingerprinted one. Control: the same fit saved without the two fields
    /// (what a bundle from before #1685 is), or with a fingerprint of another scheme,
    /// loads as today. A bundle whose recorded settings no longer reproduce the
    /// population loads its estimates with no population and a note naming the
    /// difference (review r1 #2; it was `Corrupt`).
    ///
    /// Mutations — ignore the recorded settings in `load_fit`: `Select` / `DoseFilter`
    /// load 30 subjects / 60 doses; skip the level bind: `Level` loads no population;
    /// skip the verification: the tampered bundle loads its population; drop the
    /// note: its equality dies.
    #[test]
    fn load_fit_reads_the_bundled_data_as_the_fit_did() {
        use crate::estimation::fit_inputs::test_fixtures::{case, Kind};
        let doses = |p: &Population| p.subjects.iter().map(|s| s.doses.len()).sum::<usize>();
        let dir = tempfile::tempdir().unwrap();
        let save_load =
            |fit: &FitResult, c: &crate::estimation::fit_inputs::test_fixtures::Case| {
                let path = dir.path().join("t4.fitrx");
                save_fit(
                    fit,
                    &c.prep.population,
                    fit.model_text.as_deref().unwrap(),
                    &path,
                    SaveFitOptions {
                        include_data: fit.data_path.as_ref().map(PathBuf::from),
                    },
                )
                .unwrap();
                load_fit(&path)
            };
        for (kind, fitted, raw) in [
            (Kind::Select, (29, 29), (30, 30)),
            (Kind::DoseFilter, (30, 30), (30, 60)),
            (Kind::Level, (30, 30), (30, 30)),
        ] {
            let c = case(kind);
            let loaded = save_load(&c.fit, &c).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            let p = loaded.population.expect("bundled");
            assert_eq!((p.subjects.len(), doses(&p)), fitted, "{kind:?}");
            let fp = c.fit.population_fingerprint.as_ref().unwrap();
            assert_eq!(fp.first_difference(&p), None, "{kind:?}");

            let mut legacy = c.fit.clone();
            legacy.reader_settings = None;
            legacy.population_fingerprint = None;
            let p = save_load(&legacy, &c).unwrap().population.unwrap();
            assert_eq!((p.subjects.len(), doses(&p)), raw, "{kind:?} legacy");

            let mut stale = c.fit.clone();
            stale.population_fingerprint = stale
                .population_fingerprint
                .map(|f| f.with_scheme(crate::types::POPULATION_FINGERPRINT_SCHEME + 1));
            let p = save_load(&stale, &c).unwrap().population.unwrap();
            assert_eq!((p.subjects.len(), doses(&p)), raw, "{kind:?} stale scheme");
        }

        let c = case(Kind::DoseFilter);
        let mut tampered = c.fit.clone();
        tampered
            .reader_settings
            .as_mut()
            .unwrap()
            .ignore_exprs
            .clear();
        // Review r1 #2: the bundle still loads — estimates intact, no population, and
        // a note naming the difference — rather than failing as `Corrupt`.
        let loaded = save_load(&tampered, &c).expect("the estimates still load");
        assert!(loaded.population.is_none());
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&loaded.fit.theta), bits(&c.fit.theta));
        const NOTE: &str = "load_fit: the bundled data.csv was not loaded. Read with the fit's \
             recorded reader settings, it is not the population the fit was given: the doses \
             of subject `1` differ: 2 in this population, 1 in the fit's, with the same \
             observation records. The bundled data, or how this version of ferx reads it, has \
             changed since the fit was saved. The estimates are loaded; to run `run_sir` or \
             `run_covariance`, pass the fit's population or refit.";
        assert_eq!(loaded.fit.warnings.last().map(String::as_str), Some(NOTE));
        assert!(
            loaded
                .fit
                .warnings_structured
                .iter()
                .any(|w| w.message == NOTE),
            "the structured warnings are rebuilt"
        );
        // A matching bundle carries no such note.
        let clean = save_load(&c.fit, &c).unwrap();
        assert!(!clean
            .fit
            .warnings
            .iter()
            .any(|w| w.starts_with("load_fit:")));
    }

    /// #1941: a fit made in memory (`prepare_run` + `fit`) records a fingerprint but
    /// no reader settings. Its bundled `data.csv` is read with the model file's
    /// settings, so the IOV model's `iov_column` gives the reloaded population its
    /// occasions and `OCC` is not read as a covariate; the read is checked against
    /// the fingerprint. Measured on the old routing-only read: 0 occasions for the
    /// fit's 220 and `covariate_names == ["OCC"]`, which `run_covariance` refused as
    /// "extra `OCC`". Controls: with no fingerprint (a pre-#1685 bundle) the legacy
    /// routing-only read is kept — and it is the occasion-less population, so the
    /// fixture straddles the fix; with a fingerprint of another population the
    /// bundle loads no population and a note naming the model file's settings.
    ///
    /// Mutations — route the settings-less arm back to `read_population_routed_by`:
    /// the occasion count and the covariate names die; skip the verification: the
    /// tampered bundle loads its population; drop the `read_with` branch: the note
    /// equality dies.
    #[test]
    fn load_fit_reads_an_in_memory_iov_fit_with_the_model_file_settings() {
        let prep =
            crate::api::prepare_run("examples/warfarin_iov.ferx", Some("data/warfarin_iov.csv"))
                .unwrap();
        let occasions =
            |p: &Population| p.subjects.iter().map(|s| s.occasions.len()).sum::<usize>();
        let want = occasions(&prep.population);
        assert!(want > 0, "the fixture must carry occasions");
        assert!(!prep.population.covariate_names.iter().any(|c| c == "OCC"));

        let dir = tempfile::tempdir().unwrap();
        let save_load = |fit: &FitResult| {
            let path = dir.path().join("iov.fitrx");
            save_fit(
                fit,
                &prep.population,
                &prep.model_text,
                &path,
                SaveFitOptions {
                    include_data: Some(PathBuf::from("data/warfarin_iov.csv")),
                },
            )
            .unwrap();
            load_fit(&path).unwrap()
        };
        let mut fit = minimal_fit_result();
        fit.model_text = Some(prep.model_text.clone());
        fit.reader_settings = None;
        fit.population_fingerprint =
            Some(crate::types::PopulationFingerprint::of(&prep.population));

        let loaded = save_load(&fit);
        let p = loaded.population.expect("the in-memory fit's data reloads");
        assert_eq!(occasions(&p), want);
        assert_eq!(p.covariate_names, prep.population.covariate_names);
        let fp = fit.population_fingerprint.as_ref().unwrap();
        assert_eq!(fp.first_difference(&p), None);
        assert!(!loaded
            .fit
            .warnings
            .iter()
            .any(|w| w.starts_with("load_fit:")));

        // No fingerprint: the legacy routing-only read, unchanged and occasion-less.
        let mut legacy = fit.clone();
        legacy.population_fingerprint = None;
        let p = save_load(&legacy).population.unwrap();
        assert_eq!(occasions(&p), 0);
        assert_eq!(p.covariate_names, vec!["OCC".to_string()]);

        // A fingerprint the model file's settings do not reproduce: no population.
        let mut other = prep.population.clone();
        other.subjects.pop();
        let mut tampered = fit.clone();
        tampered.population_fingerprint = Some(crate::types::PopulationFingerprint::of(&other));
        let loaded = save_load(&tampered);
        assert!(loaded.population.is_none());
        let note = loaded
            .fit
            .warnings
            .last()
            .expect("a note says why the data was not loaded");
        assert!(
            note.starts_with(
                "load_fit: the bundled data.csv was not loaded. Read with the model \
                 file's reader settings (the fit recorded none), it is not the population \
                 the fit was given: "
            ),
            "{note}"
        );
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
        assert_eq!(loaded.manifest.format_version, "2");
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
            let name = f.name().unwrap().to_string();
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
            let name = f.name().unwrap().to_string();
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
            let name = f.name().unwrap().to_string();
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

    // ── #1082 PR4: ebes_kappa_<LEVEL>.csv ────────────────────────────────────

    fn arm_level() -> LevelWire {
        LevelWire {
            name: "ARM".into(),
            columns: vec!["STUDY".into(), "ARM".into()],
            kappa_names: vec!["K_ARM".into()],
        }
    }

    fn corrupt_message(r: Result<crate::types::LevelKappaEbes, FitrxError>) -> String {
        match r {
            Err(FitrxError::Corrupt(msg)) => msg,
            other => panic!("expected Corrupt, got {other:?}"),
        }
    }

    /// Rows are placed by `GROUP`, not by file order.
    #[test]
    fn a_level_kappa_csv_places_its_rows_by_group() {
        let csv = "ID,GROUP,STUDY,ARM,K_ARM\nS1,2,1,2,0.200000\nS1,1,1,1,0.100000\n\
                   S2,1,2,1,-0.300000\n";
        let l = parse_level_kappa_csv(csv, arm_level(), &["S1".into(), "S2".into()]).unwrap();
        assert_eq!(l.level, "ARM");
        assert_eq!(l.groups[0].len(), 2);
        assert_eq!(l.groups[0][0].labels, [1.0, 1.0]);
        assert_eq!(l.groups[0][1].kappa, [0.2]);
        assert_eq!(l.groups[1][0].kappa, [-0.3]);
    }

    /// A NaN κ̂ is written as an empty cell and reads back as NaN; a label keeps every
    /// digit (it names the group, which the sdtab matches by bits).
    #[test]
    fn a_level_kappa_csv_round_trips_a_nan_estimate_and_an_exact_label() {
        let level = crate::types::LevelKappaEbes {
            level: "ARM".into(),
            columns: vec!["STUDY".into(), "ARM".into()],
            kappa_names: vec!["K_ARM".into()],
            groups: vec![vec![
                crate::types::LevelGroupEbe {
                    labels: vec![1.0, 0.1234567],
                    kappa: vec![f64::NAN],
                },
                crate::types::LevelGroupEbe {
                    labels: vec![1.0, 1e-7],
                    kappa: vec![0.25],
                },
            ]],
        };
        let mut r = minimal_fit_result();
        r.subjects.truncate(1);
        let mut buf = Vec::new();
        write_level_kappa_csv(&mut buf, &r, &level).unwrap();
        let csv = String::from_utf8(buf).unwrap();
        let back = parse_level_kappa_csv(&csv, arm_level(), &[r.subjects[0].id.clone()]).unwrap();
        let g = &back.groups[0];
        assert!(g[0].kappa[0].is_nan(), "{csv}");
        assert_eq!(g[1].kappa, [0.25]);
        assert_eq!(g[0].labels[1].to_bits(), 0.1234567f64.to_bits(), "{csv}");
        assert_eq!(g[1].labels[1].to_bits(), 1e-7f64.to_bits(), "{csv}");
    }

    /// A one-level fit's `fit.json` carries neither new key, so its bundle is the bytes it
    /// was before #1082 PR4 (review r1 #12).
    #[test]
    fn a_one_level_fit_json_has_no_level_keys() {
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_element(1, 1, 0.5));
        r.kappa_names = vec!["K".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        let json = serde_json::to_string(&build_fit_wire(&r)).unwrap();
        assert!(json.contains("\"kappa_names\""), "{json}");
        assert!(!json.contains("kappa_levels"), "{json}");
        assert!(!json.contains("\"levels\""), "{json}");
    }

    #[test]
    fn a_level_list_must_agree_with_kappa_levels() {
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_element(1, 1, 0.5));
        r.kappa_names = vec!["K_ARM".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        r.kappa_levels = vec!["STUDY".into()];
        r.ebe_kappas_by_level = vec![crate::types::LevelKappaEbes {
            level: "ARM".into(),
            columns: vec!["STUDY".into(), "ARM".into()],
            kappa_names: vec!["K_ARM".into()],
            groups: Vec::new(),
        }];
        let msg = validate_parallel_lengths(&build_fit_wire(&r))
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("iov.levels names kappa `K_ARM` at level `ARM`, but iov.kappa_levels puts it at `STUDY`"),
            "{msg}"
        );
    }

    #[test]
    fn a_level_kappa_csv_must_match_its_level_and_number_its_groups_from_one() {
        let ids = ["S1".to_string()];
        let msg = corrupt_message(parse_level_kappa_csv(
            "ID,GROUP,ARM,K_ARM\nS1,1,1,0.1\n",
            arm_level(),
            &ids,
        ));
        assert!(msg.contains("ebes_kappa_ARM.csv: header"), "{msg}");
        assert!(msg.contains("ID,GROUP,STUDY,ARM,K_ARM"), "{msg}");
        let msg = corrupt_message(parse_level_kappa_csv(
            "ID,GROUP,STUDY,ARM,K_ARM\nS1,2,1,1,0.1\n",
            arm_level(),
            &ids,
        ));
        assert!(msg.contains("GROUP must run 1..n"), "{msg}");
        let msg = corrupt_message(parse_level_kappa_csv(
            "ID,GROUP,STUDY,ARM,K_ARM\nS1,0,1,1,0.1\n",
            arm_level(),
            &ids,
        ));
        assert!(msg.contains("bad GROUP"), "{msg}");
        let msg = corrupt_message(parse_level_kappa_csv(
            "ID,GROUP,STUDY,ARM,K_ARM\nS9,1,1,1,0.1\n",
            arm_level(),
            &ids,
        ));
        assert!(msg.contains("unknown ID"), "{msg}");
    }

    /// fit.json names each bundled level; a bundle that names one and lacks its CSV is
    /// corrupt rather than loading with that level silently empty.
    #[test]
    fn a_bundle_naming_a_level_without_its_csv_is_corrupt() {
        use std::io::{Read as _, Write as _};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested.fitrx");
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_element(1, 1, 0.5));
        r.kappa_names = vec!["K_ARM".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false; 1];
        r.kappa_param_types = vec![EtaParamType::LogNormal; 1];
        r.kappa_levels = vec!["ARM".into()];
        let group = |v: f64| crate::types::LevelGroupEbe {
            labels: vec![1.0, 1.0],
            kappa: vec![v],
        };
        r.ebe_kappas_by_level = vec![crate::types::LevelKappaEbes {
            level: "ARM".into(),
            columns: vec!["STUDY".into(), "ARM".into()],
            kappa_names: vec!["K_ARM".into()],
            groups: r.subjects.iter().map(|_| vec![group(0.25)]).collect(),
        }];
        let ids: Vec<&str> = r.subjects.iter().map(|s| s.id.as_str()).collect();
        let p = dummy_population(&ids, 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let whole = load_fit(&path).unwrap();
        assert_eq!(whole.fit.kappa_levels, ["ARM"]);
        assert_eq!(whole.fit.ebe_kappas_by_level, r.ebe_kappas_by_level);

        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let patched = dir.path().join("nested-no-level-csv.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&patched).unwrap());
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().unwrap().to_string();
            if name == "ebes_kappa_ARM.csv" {
                continue;
            }
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&buf).unwrap();
        }
        zw.finish().unwrap();
        match load_fit(&patched) {
            Err(FitrxError::Corrupt(msg)) => assert!(
                msg.contains(
                    "fit.json names level `ARM`, but the bundle has no ebes_kappa_ARM.csv"
                ),
                "{msg}"
            ),
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }

    // ── Fix 3: omega_init backward-compat fallback ───────────────────────────

    /// #1382: a bundle that stores a covariance matrix must also store which
    /// estimator built it, as the same `r` / `s` / `rsr` token `[fit_options]`
    /// accepts — a runlog reader can then paste the value straight back.
    ///
    /// `rsr`, not the default: a round trip through `Hessian` is also a round
    /// trip through `CovarianceMethod::default()`, so a loader that dropped the
    /// field and fell back to the default would pass.
    ///
    /// Mutation: drop `covariance_method` from `FitWire`, or write
    /// `format!("{m:?}")` instead of `m.label()` → this fires (as `None` and as a
    /// `Corrupt` load error respectively).
    #[test]
    fn covariance_method_roundtrips_as_the_fit_option_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cov-method.fitrx");
        let mut r = minimal_fit_result();
        r.covariance_method = Some(CovarianceMethod::Sandwich);
        assert_ne!(
            Some(CovarianceMethod::default()),
            r.covariance_method,
            "premise: this must not be the default, or a dropped field round-trips \
             by accident"
        );
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        // The stored token is the model-file spelling, not a Rust variant name.
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let wire: serde_json::Value = {
            use std::io::Read as _;
            let mut f = archive.by_name("fit.json").unwrap();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            serde_json::from_slice(&buf).unwrap()
        };
        assert_eq!(wire["covariance_method"], serde_json::json!("rsr"));

        let loaded = load_fit(&path).unwrap();
        assert_eq!(
            loaded.fit.covariance_method,
            Some(CovarianceMethod::Sandwich)
        );
    }

    /// #2571: `total_ebe_nm_polished` survives a save/load, and a bundle written before the
    /// field existed loads it as 0, the count every fit reported until then.
    #[test]
    fn total_ebe_nm_polished_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nm-polished.fitrx");
        let mut r = minimal_fit_result();
        r.total_ebe_nm_polished = 7;
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        assert_eq!(load_fit(&path).unwrap().fit.total_ebe_nm_polished, 7);
    }

    /// #1382: an unrecognised token is `Corrupt`, the same verdict every other
    /// enum on this wire gets — not a silent degrade to `None`, which would turn
    /// a damaged bundle into one that merely looks unlabelled.
    #[test]
    fn an_unknown_covariance_method_token_is_corrupt() {
        use std::io::{Read as _, Write as _};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad-cov-method.fitrx");
        let mut r = minimal_fit_result();
        r.covariance_method = Some(CovarianceMethod::Hessian);
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let mut archive = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().unwrap().to_string();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if name == "fit.json" {
                let mut v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
                v.as_object_mut().unwrap().insert(
                    "covariance_method".into(),
                    serde_json::json!("Hessian"), // the Rust variant name, not the token
                );
                buf = serde_json::to_vec_pretty(&v).unwrap();
            }
            entries.push((name, buf));
        }
        let patched = dir.path().join("bad-cov-method-patched.fitrx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&patched).unwrap());
        for (name, body) in entries {
            zw.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(&body).unwrap();
        }
        zw.finish().unwrap();

        let err = load_fit(&patched).expect_err("an unknown token must not load");
        assert!(
            format!("{err:?}").contains("covariance_method"),
            "the error must name the offending field: {err:?}"
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

    // ── #426: the scoring record ───────────────────────────────────────────

    /// T11 (#426): `scoring_settings` survives the bundle, field for field, beside a SIR
    /// record whose scoring half differs from it. Mutations: drop the field from the writer
    /// or the reader (the first equality dies); map a token to the default enum (every enum
    /// here is off-default).
    #[test]
    fn roundtrip_keeps_scoring_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scoring_settings.fitrx");
        let mut r = sir_fit_with_settings();
        r.scoring_settings = Some(crate::ScoringSettings {
            inner_tol: 1e-8,
            ..off_default_scoring_settings()
        });
        assert_ne!(r.scoring_settings, Some(crate::ScoringSettings::default()));
        assert_ne!(
            r.scoring_settings.as_ref(),
            r.sir_settings.as_ref().map(|s| &s.scoring)
        );
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.scoring_settings, r.scoring_settings);
        assert_eq!(loaded.fit.sir_settings, r.sir_settings);
    }

    /// T11 (#426): the top-level record is held to the `[fit_options]` domain like the SIR
    /// record, and a refusal names the block it read. One row per check, the reject and the
    /// boundary accept, as in `sir_settings_out_of_domain_values_are_corrupt`. Mutation:
    /// decode the block with the `sir.settings` prefix (every message names the wrong block).
    #[test]
    fn scoring_settings_out_of_domain_values_are_corrupt() {
        let mut r = minimal_fit_result();
        r.scoring_settings = Some(off_default_scoring_settings());
        let load = |field: &str, v: serde_json::Value| {
            let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
            value["scoring_settings"][field] = v;
            let wire: FitWire = serde_json::from_value(value).unwrap();
            wire_to_fit_result(wire, r.subjects.clone(), Vec::new())
        };
        for (field, reject, accept) in [
            (
                "ode_reltol",
                serde_json::json!(0.0),
                serde_json::json!(1e-12),
            ),
            (
                "ode_abstol",
                serde_json::json!(-1.0),
                serde_json::json!(1e-12),
            ),
            ("ode_max_steps", serde_json::json!(0), serde_json::json!(1)),
            ("n_agq", serde_json::json!(0), serde_json::json!(1)),
            (
                "ode_stiff_abort_after",
                serde_json::json!(0),
                serde_json::json!(1),
            ),
            (
                "inner_optimizer",
                serde_json::json!("Lbfgs"),
                serde_json::json!("lbfgs"),
            ),
            (
                "gradient_method",
                serde_json::json!("bogus"),
                serde_json::json!("fd"),
            ),
            (
                "gradient_method",
                serde_json::json!("ad"),
                serde_json::json!("auto"),
            ),
            (
                "nn_l2_lambda",
                serde_json::json!(-1.0),
                serde_json::json!(0.0),
            ),
            (
                "nn_smooth_lambda",
                serde_json::json!(-0.5),
                serde_json::json!(0.0),
            ),
        ] {
            match load(field, reject) {
                Err(FitrxError::Corrupt(msg)) => assert!(
                    msg.contains(&format!("scoring_settings.{field} ")),
                    "{field}: {msg}"
                ),
                other => panic!("{field}: expected Corrupt, got {:?}", other.map(|_| ())),
            }
            let loaded = load(field, accept.clone())
                .unwrap_or_else(|e| panic!("{field} = {accept} must load: {e:?}"));
            assert!(loaded.scoring_settings.is_some(), "{field}");
        }
    }

    /// T8 (#1835): `gradient_method` on the wire, in both records. A bundle written before the
    /// field existed has no key in `scoring_settings` nor in the flattened `sir.settings`, and
    /// each reads `auto` — the default, so a post-hoc step takes the caller's route as before;
    /// a recorded `ad`, which no fit can produce (`E_AD_RETIRED`), is `Corrupt` in each, with
    /// the domain named. The fixture records `fd` in both, so the round trip
    /// (`roundtrip_keeps_scoring_settings`) and the removal below each see a non-default value.
    /// Mutations: drop the serde default (the old-bundle arms fail to deserialize); accept
    /// `ad` (its arms load `Ok`); decode every token as `auto` (the round trip dies).
    #[test]
    fn a_recorded_ad_gradient_method_is_corrupt() {
        let r = {
            let mut r = sir_fit_with_settings();
            r.scoring_settings = Some(off_default_scoring_settings());
            r
        };
        let load = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
            edit(&mut value);
            let wire: FitWire = serde_json::from_value(value).expect("the wire deserializes");
            wire_to_fit_result(wire, r.subjects.clone(), Vec::new())
        };
        let written = serde_json::to_value(build_fit_wire(&r)).unwrap();
        assert_eq!(written["scoring_settings"]["gradient_method"], "fd");
        assert_eq!(written["sir"]["settings"]["gradient_method"], "fd");

        for block in ["scoring_settings", "sir.settings"] {
            let got = load(&|v: &mut serde_json::Value| match block {
                "scoring_settings" => v["scoring_settings"]["gradient_method"] = "ad".into(),
                _ => v["sir"]["settings"]["gradient_method"] = "ad".into(),
            });
            match got {
                Err(FitrxError::Corrupt(msg)) => assert_eq!(
                    msg,
                    format!("{block}.gradient_method must be auto or fd, got \"ad\"")
                ),
                other => panic!("{block}: expected Corrupt, got {:?}", other.map(|_| ())),
            }
        }
    }

    /// A `method = vi` fit as `vi_final_ofv = none` leaves it (#2058): `ofv`, `aic`,
    /// `bic` and a priored fit's `ofv_data` are `NaN`, and `vi` carries every field —
    /// a `NaN` tightness ratio and trace point, a penalized `objective_trace`, a
    /// `superseded_by` label and per-occasion κ means — so a dropped key shows.
    fn vi_fit_with_nan_ofv() -> FitResult {
        let mut r = minimal_fit_result();
        r.method = EstimationMethod::Vi;
        r.method_chain = vec![EstimationMethod::Vi];
        r.ofv = f64::NAN;
        r.ofv_data = f64::NAN;
        r.ofv_prior = 1.5;
        r.aic = f64::NAN;
        r.bic = f64::NAN;
        r.prior_summary = vec![crate::types::PriorSummary {
            name: "CL".into(),
            prior_value: 1.0,
            estimate: 1.0,
            shift_in_prior_sds: 0.0,
            penalty: 1.5,
            family: "lognormal".into(),
            prior_lower_95: 0.5,
            prior_upper_95: 2.0,
        }];
        r.vi = Some(ViResult {
            neg_two_elbo: -12.5,
            data_term: 3.25,
            kl_term: 0.125,
            n_iterations: 3,
            converged: false,
            family: "full_rank".into(),
            n_mc_samples: 4,
            kl: "analytic".into(),
            n_kl_fallback_subjects: 1,
            elbo_trace: vec![1.0, f64::NAN, -12.5],
            objective_trace: vec![2.0, 1.5, -12.0],
            eta_means: vec![vec![0.1, -0.2], vec![0.3, 0.4]],
            kappa_means: vec![vec![vec![0.01], vec![-0.02]], vec![]],
            eta_covs: vec![
                vec![vec![0.5, 0.1], vec![0.1, 0.25]],
                vec![vec![0.4, 0.0], vec![0.0, 0.2]],
            ],
            n_fd_subjects: 2,
            elbo_tightness_ratio: f64::NAN,
            superseded_by: Some("focei".into()),
        });
        r
    }

    /// #2058: a VI fit's `NaN` OFV / AIC / BIC used to make `fit.json` unloadable
    /// (`invalid type: null, expected f64`), and its `vi` result was dropped on every
    /// round trip. Both survive now: the four objectives come back `NaN` and `vi` comes
    /// back identical. `Debug` is the comparison because `ViResult` has no
    /// `PartialEq` and prints `NaN` as itself, where `==` would fail on it.
    /// Mutations: drop `scalar` from any of `ofv` / `aic` / `bic` (load fails);
    /// `vi: None` on either side; drop any one `ViWire` field.
    #[test]
    fn vi_fit_with_nan_ofv_round_trips_through_fitrx() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vi.fitrx");
        let r = vi_fit_with_nan_ofv();
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&r, &p, "model source\n", &path, SaveFitOptions::default()).unwrap();

        let l = load_fit(&path).expect("a NaN-OFV VI bundle must load").fit;
        assert_eq!(l.method, EstimationMethod::Vi);
        for (name, v) in [
            ("ofv", l.ofv),
            ("ofv_data", l.ofv_data),
            ("aic", l.aic),
            ("bic", l.bic),
        ] {
            assert!(v.is_nan(), "{name} must come back NaN, got {v}");
        }
        assert_eq!(l.ofv_prior, 1.5);
        assert_eq!(
            format!("{:?}", l.vi),
            format!("{:?}", r.vi),
            "the VI result must survive the round trip"
        );

        // The wire encoding the loaders on both sides agree on: `null`, not a
        // missing key and not a string.
        let wire = serde_json::to_value(build_fit_wire(&r)).unwrap();
        for key in ["ofv", "ofv_data", "aic", "bic"] {
            assert!(wire[key].is_null(), "{key} must be written as null");
        }
        assert_eq!(wire["vi"]["eta_posterior"][1]["id"], "S2");
        assert_eq!(
            wire["vi"]["eta_names"],
            serde_json::json!(["eta_CL", "eta_V"])
        );
    }

    /// A VI result with no per-subject covariances and no κ (a model without IOV)
    /// keeps both lists empty rather than gaining one empty entry per subject — the
    /// per-entry `None` is what tells "no list" from "an empty list for this subject".
    /// And a fit with no `vi` writes no key, so every non-VI bundle is unchanged.
    #[test]
    fn vi_without_covs_or_kappas_round_trips_and_non_vi_writes_no_key() {
        let mut r = vi_fit_with_nan_ofv();
        let vi = r.vi.as_mut().unwrap();
        vi.eta_covs.clear();
        vi.kappa_means.clear();
        vi.objective_trace.clear();
        vi.superseded_by = None;
        let wire = serde_json::to_value(build_fit_wire(&r)).unwrap();
        for key in ["cov", "kappa_means"] {
            assert!(wire["vi"]["eta_posterior"][0].get(key).is_none(), "{key}");
        }
        assert!(wire["vi"].get("objective_trace").is_none());
        let back: FitWire = serde_json::from_value(wire).unwrap();
        let l = wire_to_fit_result(back, r.subjects.clone(), Vec::new()).unwrap();
        assert_eq!(format!("{:?}", l.vi), format!("{:?}", r.vi));

        let plain = serde_json::to_value(build_fit_wire(&minimal_fit_result())).unwrap();
        assert!(
            plain.get("vi").is_none(),
            "a non-VI fit must not write `vi`"
        );
        let back: FitWire = serde_json::from_value(plain).unwrap();
        let l = wire_to_fit_result(back, minimal_fit_result().subjects, Vec::new()).unwrap();
        assert!(l.vi.is_none());
    }

    /// The posteriors are stored by position, so an entry whose `id` names another
    /// subject than its `ebes.csv` row would attach one subject's posterior to
    /// another: `Corrupt`, naming both. The unedited bundle loading in the test
    /// above is the accept side of the same gate.
    #[test]
    fn vi_posterior_id_disagreeing_with_ebes_is_corrupt() {
        let r = vi_fit_with_nan_ofv();
        let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
        value["vi"]["eta_posterior"][1]["id"] = "S9".into();
        let wire: FitWire = serde_json::from_value(value).unwrap();
        match wire_to_fit_result(wire, r.subjects.clone(), Vec::new()) {
            Err(FitrxError::Corrupt(msg)) => assert_eq!(
                msg,
                "vi.eta_posterior[1] is subject \"S9\", but ebes.csv row 2 is \"S2\""
            ),
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }

    /// Copy the bundle at `src` to `dst`, passing each entry through `edit`, which returns
    /// the entry's new body or `None` to drop it.
    fn rewrite_bundle(src: &Path, dst: &Path, edit: impl Fn(&str, Vec<u8>) -> Option<Vec<u8>>) {
        let mut archive = ZipArchive::new(File::open(src).unwrap()).unwrap();
        let mut out = ZipWriter::new(File::create(dst).unwrap());
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().unwrap().to_string();
            let mut body = Vec::new();
            f.read_to_end(&mut body).unwrap();
            drop(f);
            if let Some(body) = edit(&name, body) {
                out.start_file(name, SimpleFileOptions::default()).unwrap();
                out.write_all(&body).unwrap();
            }
        }
        out.finish().unwrap();
    }

    /// One JSON entry of a bundle, edited.
    fn edit_json(body: Vec<u8>, edit: impl Fn(&mut serde_json::Value)) -> Vec<u8> {
        let mut v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        edit(&mut v);
        serde_json::to_vec_pretty(&v).unwrap()
    }

    fn entry_text(path: &Path, name: &str) -> Option<String> {
        let mut archive = ZipArchive::new(File::open(path).unwrap()).unwrap();
        read_optional_text(&mut archive, name).unwrap()
    }

    fn fit_json(path: &Path) -> serde_json::Value {
        serde_json::from_str(&entry_text(path, "fit.json").unwrap()).unwrap()
    }

    /// Save `r` beside `dir`, returning the path; the fixture's two subjects, three
    /// observations each.
    fn saved(dir: &tempfile::TempDir, r: &FitResult) -> PathBuf {
        let path = dir.path().join("v2.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        path
    }

    fn load_corrupt(r: Result<LoadedFit, FitrxError>) -> String {
        match r {
            Err(FitrxError::Corrupt(msg)) => msg,
            other => panic!("expected Corrupt, got {:?}", other.map(|_| ())),
        }
    }

    /// #2695: a v1 bundle is refused with the version error. Its `fit.json` also lacks
    /// keys v2 requires, so a version check placed after the parse would answer with a
    /// JSON error instead, and that is the mutation this catches.
    #[test]
    fn a_v1_bundle_is_refused_with_the_version_error() {
        let dir = tempfile::tempdir().unwrap();
        let v2 = saved(&dir, &minimal_fit_result());
        let v1 = dir.path().join("v1.fitrx");
        rewrite_bundle(&v2, &v1, |name, body| {
            Some(match name {
                "manifest.json" => edit_json(body, |m| m["format_version"] = "1".into()),
                "fit.json" => edit_json(body, |f| {
                    let f = f.as_object_mut().unwrap();
                    assert!(f.remove("ofv_data").is_some(), "premise: v2 writes the key");
                    f.remove("environment");
                }),
                _ => body,
            })
        });
        let err = load_fit(&v1).map(|_| ()).unwrap_err();
        assert!(
            matches!(&err, FitrxError::UnsupportedVersion(v) if v == "1"),
            "got {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "unsupported .fitrx format_version \"1\"; expected \"2\""
        );
    }

    /// #2867: the per-start rows of a multi-start fit survive a save and load unchanged —
    /// a row that ran, one with no OFV, one under a penalty and a failed one with its code —
    /// and a single-start fit writes no `starts` key, so its `fit.json` is what it was. A
    /// bundle without the key loads with no rows.
    #[test]
    fn the_per_start_rows_round_trip() {
        use crate::types::StartRecord;
        let dir = tempfile::tempdir().unwrap();
        let single = saved(&dir, &minimal_fit_result());
        assert!(fit_json(&single).get("starts").is_none());
        assert!(load_fit(&single).unwrap().fit.starts.is_empty());

        // The status is a token and labels, not a typed enum: a reason this build has never
        // heard of (written by a later one) is read back as it was written.
        let ran = StartRecord {
            start: 0,
            ofv: Some(-221.877048),
            penalized: false,
            objective: None,
            converged: true,
            status: Some("converged_with_issues".into()),
            status_reasons: vec!["budget_spent".into(), "a_future_reason".into()],
            n_iterations: 84,
            wall_time_secs: 1.25,
            theta: vec![7.4577, 22.1467],
            selected: true,
            error: None,
            error_code: None,
        };
        let mut r = minimal_fit_result();
        r.n_starts = 4;
        r.starts = vec![
            ran.clone(),
            StartRecord {
                start: 1,
                ofv: None,
                status: Some("converged".into()),
                status_reasons: Vec::new(),
                selected: false,
                ..ran.clone()
            },
            StartRecord {
                start: 2,
                penalized: true,
                objective: Some(-200.5),
                converged: false,
                status: Some("not_converged".into()),
                status_reasons: Vec::new(),
                selected: false,
                ..ran.clone()
            },
            StartRecord {
                start: 3,
                ofv: None,
                penalized: false,
                objective: None,
                converged: false,
                status: None,
                status_reasons: Vec::new(),
                n_iterations: 0,
                wall_time_secs: 0.0,
                theta: Vec::new(),
                selected: false,
                error: Some("n_agq must be at least 1".into()),
                error_code: Some("E_AGQ_NODES_ZERO".into()),
            },
        ];
        let path = saved(&dir, &r);
        let json = fit_json(&path);
        assert_eq!(json["starts"].as_array().map(Vec::len), Some(4));
        assert_eq!(load_fit(&path).unwrap().fit.starts, r.starts);

        // A start that ended on a non-finite θ must not make the bundle unloadable: the value
        // is written as `null` and read back as `NaN`.
        r.starts[2].theta = vec![f64::NAN, 22.1467];
        let path = saved(&dir, &r);
        assert!(fit_json(&path)["starts"][2]["theta"][0].is_null());
        let loaded = load_fit(&path).unwrap().fit.starts;
        assert!(loaded[2].theta[0].is_nan(), "{:?}", loaded[2].theta);
        assert_eq!(loaded[2].theta[1], 22.1467);
        assert_eq!(loaded[3], r.starts[3]);
    }

    /// #2695: no key has a default for a file that predates it. A v2 `fit.json` missing
    /// one of these is damaged, and refusing it is what removing the defaults bought.
    /// Each mutation restoring a `#[serde(default)]` kills its row.
    #[test]
    fn a_v2_fit_json_missing_a_required_key_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let v2 = saved(&dir, &minimal_fit_result());
        for key in [
            "ofv_data",
            "ofv_prior",
            "environment",
            "omega_init",
            "theta_init",
            "sigma_init",
            "bic_inputs",
            "iwres_lag1_r",
            "dw_statistic",
            "n_starts",
            "optimizer",
            "ode_linear_exact",
            "total_ebe_nm_polished",
            "method_wall_times_secs",
            "estimate_near_boundary",
            "max_abs_correlation",
        ] {
            let edited = dir.path().join("edited.fitrx");
            rewrite_bundle(&v2, &edited, |name, body| {
                Some(if name == "fit.json" {
                    edit_json(body, |f| {
                        assert!(
                            f.as_object_mut().unwrap().remove(key).is_some(),
                            "premise: {key} is written"
                        );
                    })
                } else {
                    body
                })
            });
            match load_fit(&edited) {
                Err(FitrxError::Json(e)) => {
                    assert!(e.to_string().contains(key), "{key}: {e}");
                }
                other => panic!("{key}: expected a JSON error, got {:?}", other.map(|_| ())),
            }
        }
    }

    /// #2695: the parallel vectors have no "older bundle" empty form any more. An empty one
    /// is a length mismatch like any other.
    #[test]
    fn an_empty_parallel_vector_is_corrupt() {
        let mut r = minimal_fit_result();
        r.omega_iov = Some(DMatrix::from_element(1, 1, 0.04));
        r.kappa_names = vec!["K".into()];
        r.kappa_fixed = vec![false];
        r.kappa_init_as_sd = vec![false];
        r.kappa_param_types = vec![EtaParamType::LogNormal];
        let load = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut value = serde_json::to_value(build_fit_wire(&r)).unwrap();
            edit(&mut value);
            let wire: FitWire = serde_json::from_value(value).unwrap();
            wire_to_fit_result(wire, r.subjects.clone(), Vec::new()).map(|_| ())
        };
        load(&|_| ()).expect("premise: the unedited wire decodes");
        for (key, edit) in [
            (
                "omega.init_as_sd",
                (|v| v["omega"]["init_as_sd"] = serde_json::json!([]))
                    as fn(&mut serde_json::Value),
            ),
            ("sigma.init_as_sd", |v| {
                v["sigma"]["init_as_sd"] = serde_json::json!([])
            }),
            ("iov.kappa_init_as_sd", |v| {
                v["iov"]["kappa_init_as_sd"] = serde_json::json!([])
            }),
            ("iov.kappa_param_types", |v| {
                v["iov"]["kappa_param_types"] = serde_json::json!([])
            }),
        ] {
            match load(&edit) {
                Err(FitrxError::Corrupt(msg)) => assert!(msg.starts_with(key), "{key}: {msg}"),
                other => panic!("{key}: expected Corrupt, got {other:?}"),
            }
        }
    }

    /// #2695: what the reader calls corrupt is not written. A hand-built fit with a short
    /// parallel vector is refused with the reader's own words, and leaves no file.
    #[test]
    fn save_fit_refuses_a_fit_the_reader_would_call_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refused.fitrx");
        let mut r = minimal_fit_result();
        r.omega_init_as_sd.clear();
        let p = dummy_population(&["S1", "S2"], 3);
        match save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()) {
            Err(FitrxError::Unwritable(msg)) => {
                assert!(msg.starts_with("omega.init_as_sd (0)"), "{msg}")
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
        assert!(!path.exists());
    }

    /// #2695 review: `N_OBS` is required and every value is read. Dropping the column or
    /// blanking one cell would otherwise turn off the guard that catches two same-ID blocks
    /// in the wrong order, and a short row would index past its fields.
    #[test]
    fn predictions_csv_needs_a_readable_n_obs_on_every_row() {
        let dir = tempfile::tempdir().unwrap();
        let good = saved(&dir, &minimal_fit_result());
        load_fit(&good).expect("premise: the unedited bundle loads");
        let edited = dir.path().join("edited.fitrx");
        let load_with = |edit: &dyn Fn(&str) -> String| {
            rewrite_bundle(&good, &edited, |name, body| {
                Some(if name == "predictions.csv" {
                    edit(std::str::from_utf8(&body).unwrap()).into_bytes()
                } else {
                    body
                })
            });
            load_corrupt(load_fit(&edited))
        };
        // Drop the column at `at` from every line.
        let without = |csv: &str, at: usize| -> String {
            csv.lines()
                .map(|l| {
                    let mut f: Vec<&str> = l.split(',').collect();
                    f.remove(at);
                    f.join(",") + "\n"
                })
                .collect()
        };
        let n_obs_col = |csv: &str| {
            let header = csv.lines().next().unwrap();
            header.split(',').position(|c| c == "N_OBS").unwrap()
        };
        let no_column = load_with(&|csv| without(csv, n_obs_col(csv)));
        assert_eq!(no_column, "predictions.csv missing column N_OBS");
        let blank = load_with(&|csv| {
            let at = n_obs_col(csv);
            let mut lines: Vec<String> = csv.lines().map(String::from).collect();
            let mut f: Vec<&str> = lines[2].split(',').collect();
            f[at] = "";
            lines[2] = f.join(",");
            lines.join("\n") + "\n"
        });
        assert_eq!(blank, "predictions.csv row 2: bad N_OBS \"\"");
        let short = load_with(&|csv| {
            let mut lines: Vec<String> = csv.lines().map(String::from).collect();
            lines[1] = lines[1].rsplit_once(',').unwrap().0.to_string();
            lines.join("\n") + "\n"
        });
        assert!(short.starts_with("predictions.csv row 1 has "), "{short}");
    }

    /// #2695: `ofv_data` and `ofv_prior` are written for every fit, so an unpriored fit's
    /// reload restores what it held rather than reconstructing it from `ofv`. The fixture
    /// holds a value `ofv` could not have supplied.
    #[test]
    fn ofv_split_is_written_for_an_unpriored_fit() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        assert!(r.prior_summary.is_empty(), "premise: no prior");
        r.ofv_data = r.ofv + 3.5;
        r.ofv_prior = -3.5;
        let loaded = load_fit(&saved(&dir, &r)).unwrap().fit;
        assert_eq!(loaded.ofv_data, r.ofv + 3.5);
        assert_eq!(loaded.ofv_prior, -3.5);
    }

    fn discrete_row(cmt: usize, time: f64, iwres: f64) -> crate::types::DiscreteObsDiagnostic {
        crate::types::DiscreteObsDiagnostic {
            cmt,
            time,
            dv: 1.0,
            pred: 0.1 + 0.2,
            ipred: 1e-7,
            iwres,
        }
    }

    /// #2695: discrete rows round-trip, each to its own subject. The two subjects share an
    /// ID, so only `n_obs_discrete` can place the rows, and the first has fewer rows than
    /// the second. The doubles need all their digits and one cell is `NaN`.
    #[test]
    fn roundtrip_discrete_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("discrete.fitrx");
        let mut r = minimal_fit_result();
        r.subjects[1].id = r.subjects[0].id.clone();
        let id = r.subjects[0].id.clone();
        r.subjects[0].discrete_rows = vec![discrete_row(3, 1.5, f64::NAN)];
        r.subjects[1].discrete_rows =
            vec![discrete_row(3, 2.5, 0.25), discrete_row(4, 3.5, -1.0 / 3.0)];
        let p = dummy_population(&[id.as_str(), id.as_str()], 3);
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();

        let loaded = load_fit(&path).unwrap();
        assert!(loaded
            .manifest
            .entries
            .contains(&"predictions_discrete.csv".to_string()));
        let got = &loaded.fit.subjects;
        assert_eq!(got[0].discrete_rows.len(), 1);
        assert_eq!(got[1].discrete_rows.len(), 2);
        assert_eq!(got[0].discrete_rows[0].time, 1.5);
        assert!(got[0].discrete_rows[0].iwres.is_nan());
        assert_eq!(got[1].discrete_rows[1], r.subjects[1].discrete_rows[1]);
        assert_eq!(got[1].discrete_rows[0], r.subjects[1].discrete_rows[0]);
        assert_eq!(
            entry_text(&path, "ebes.csv")
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .split(',')
                .filter(|c| *c == "n_obs_discrete")
                .count(),
            1
        );
    }

    /// #2695 review: the rows outlive a second save, in every build: storing them does not
    /// need `survival`, which only computes them. Run under both feature sets (the default
    /// build is the one that used to drop them). Mutation: gate the entry on `survival`
    /// again → dies in the default build.
    #[test]
    fn discrete_rows_survive_a_load_and_re_save() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.subjects[1].discrete_rows = vec![discrete_row(3, 2.5, 0.25)];
        let loaded = load_fit(&saved(&dir, &r)).unwrap();
        let again = dir.path().join("again.fitrx");
        let p = dummy_population(&["S1", "S2"], 3);
        save_fit(&loaded.fit, &p, "src\n", &again, SaveFitOptions::default()).unwrap();
        let back = load_fit(&again).unwrap().fit;
        assert!(back.subjects[0].discrete_rows.is_empty());
        assert_eq!(back.subjects[1].discrete_rows, r.subjects[1].discrete_rows);
    }

    /// The other side of the gate: a fit with no discrete rows has neither the entry nor
    /// the column, so its `ebes.csv` is the plain one.
    #[test]
    fn a_fit_without_discrete_rows_writes_neither_entry_nor_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = saved(&dir, &minimal_fit_result());
        assert_eq!(entry_text(&path, "predictions_discrete.csv"), None);
        assert!(!entry_text(&path, "ebes.csv")
            .unwrap()
            .contains("n_obs_discrete"));
    }

    /// #2695: the count column and the entry vouch for each other, and a row that does not
    /// fit its count is refused rather than given to a neighbour.
    #[test]
    fn damaged_discrete_rows_are_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.subjects[0].discrete_rows = vec![discrete_row(3, 1.5, 0.5)];
        r.subjects[1].discrete_rows = vec![discrete_row(3, 2.5, 0.5)];
        let good = saved(&dir, &r);
        let (s0, s1) = (r.subjects[0].id.clone(), r.subjects[1].id.clone());
        load_fit(&good).expect("premise: the unedited bundle loads");
        let edited = dir.path().join("edited.fitrx");
        let load_with = |edit: &dyn Fn(String) -> Option<String>| {
            rewrite_bundle(&good, &edited, |name, body| {
                if name == "predictions_discrete.csv" {
                    edit(String::from_utf8(body).unwrap()).map(String::into_bytes)
                } else {
                    Some(body)
                }
            });
            load_corrupt(load_fit(&edited))
        };

        let no_entry = load_with(&|_| None);
        assert!(
            no_entry.contains("no predictions_discrete.csv"),
            "{no_entry}"
        );
        let one_row = load_with(&|csv| Some(csv.lines().take(2).collect::<Vec<_>>().join("\n")));
        assert!(one_row.contains("ran out of rows"), "{one_row}");
        let extra = load_with(&|csv| Some(format!("{csv}{s1},3,9.0,1.0,0.5,0.5,0.5\n")));
        assert!(extra.contains("past the rows"), "{extra}");
        let swapped = load_with(&|csv| Some(csv.replacen(&s0, "OTHER", 1)));
        assert!(swapped.contains("out of order"), "{swapped}");
        let bad_num = load_with(&|csv| Some(csv.replace("1.5", "abc")));
        assert!(bad_num.contains("bad number \"abc\""), "{bad_num}");
        let bad_header = load_with(&|csv| Some(csv.replacen("CMT", "CPT", 1)));
        assert!(bad_header.contains("header"), "{bad_header}");

        rewrite_bundle(&good, &edited, |name, body| {
            Some(if name == "ebes.csv" {
                String::from_utf8(body)
                    .unwrap()
                    .replace(",n_obs_discrete", ",other")
                    .into_bytes()
            } else {
                body
            })
        });
        let no_column = load_corrupt(load_fit(&edited));
        assert!(
            no_column.contains("ebes.csv has no n_obs_discrete"),
            "{no_column}"
        );
    }

    /// A binary + TTE model (#911): what `write_sdtab_csv` parses to decide the refusal.
    #[cfg(feature = "survival")]
    const TTE_MODEL_911: &str = "
[parameters]
  theta TH0(0.1, 0.001, 10.0)
  omega ETA ~ 0.1
[binary_model]
  cmt   = 3
  logit = TH0 + ETA
[event_model]
  cmt    = 2
  family = exponential
  scale  = TH0 * exp(ETA)
";

    #[cfg(feature = "survival")]
    fn tte_row_911(
        time: f64,
        dv: f64,
        coxsnell: f64,
        deviance: f64,
    ) -> crate::types::TteObsDiagnostic {
        crate::types::TteObsDiagnostic {
            cmt: 2,
            time,
            dv,
            coxsnell,
            martingale: 0.1 + 0.2,
            deviance,
            pit: 1e-300,
            npde: f64::NAN,
        }
    }

    /// Every field of a TTE row as bits, so `NaN` compares equal to itself and `-0.0`
    /// differs from `0.0`.
    #[cfg(feature = "survival")]
    fn tte_bits(rows: &[crate::types::TteObsDiagnostic]) -> Vec<(usize, [u64; 6])> {
        rows.iter()
            .map(|t| {
                (
                    t.cmt,
                    [t.time, t.dv, t.coxsnell, t.martingale, t.deviance, t.pit].map(f64::to_bits),
                )
            })
            .collect()
    }

    /// T1's fit (#911): three subjects, the first and third sharing an ID so only
    /// `n_obs_tte` can place their rows. S0 has one binary row and two TTE rows, S1 none,
    /// S2 three. The cells need every digit (`0.1 + 0.2`, `1e-300`), and hold an interval
    /// censor's `NaN`, a deviance of `-0.0` and a round-off-negative Cox–Snell residual.
    #[cfg(feature = "survival")]
    fn tte_fit_911() -> (FitResult, Population) {
        let mut r = minimal_fit_result();
        let s2 = r.subjects[0].clone();
        r.subjects.push(s2);
        r.n_subjects = 3;
        r.subjects[0].discrete_rows = vec![discrete_row(3, 1.5, 0.25)];
        r.subjects[0].tte_rows = vec![
            tte_row_911(5.0, 1.0, -1e-17, -0.0),
            tte_row_911(7.5, f64::NAN, 0.75, -0.5),
        ];
        r.subjects[2].tte_rows = vec![
            tte_row_911(2.0, 0.0, 0.1 + 0.2, -0.75),
            tte_row_911(4.0, 1.0, 1e-300, 0.5),
            tte_row_911(6.0, 1.0, 0.0, f64::INFINITY),
        ];
        r.model_text = Some(TTE_MODEL_911.to_string());
        let id0 = r.subjects[0].id.clone();
        let id1 = r.subjects[1].id.clone();
        let p = dummy_population(&[id0.as_str(), id1.as_str(), id0.as_str()], 3);
        (r, p)
    }

    #[cfg(feature = "survival")]
    fn save_911(dir: &tempfile::TempDir, name: &str, r: &FitResult, p: &Population) -> PathBuf {
        let path = dir.path().join(name);
        save_fit(r, p, TTE_MODEL_911, &path, SaveFitOptions::default()).unwrap();
        path
    }

    /// T1 (#911): TTE rows round-trip bit for bit, each to its own subject, through
    /// `predictions_tte.csv` and the `n_obs_tte` column, which appears exactly once.
    ///
    /// Dies on: the entry not written (no rows back), `attach_tte_rows` skipped, a field
    /// written through `{:.6}` (`0.1 + 0.2`, `1e-300`), two subjects' blocks swapped (S0
    /// and S2 share an ID, so only the count places them).
    #[cfg(feature = "survival")]
    #[test]
    fn roundtrip_tte_rows() {
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_911();
        let path = save_911(&dir, "tte.fitrx", &r, &p);
        let loaded = load_fit(&path).unwrap();
        assert!(loaded
            .manifest
            .entries
            .contains(&"predictions_tte.csv".to_string()));
        for (si, (got, want)) in loaded.fit.subjects.iter().zip(&r.subjects).enumerate() {
            assert_eq!(
                tte_bits(&got.tte_rows),
                tte_bits(&want.tte_rows),
                "subject {si}"
            );
            assert_eq!(got.discrete_rows, want.discrete_rows, "subject {si}");
        }
        let header = entry_text(&path, "ebes.csv").unwrap();
        let header = header.lines().next().unwrap();
        assert_eq!(header.split(',').filter(|c| *c == "n_obs_tte").count(), 1);
        assert!(
            header.ends_with(",n_obs,n_obs_discrete,n_obs_tte"),
            "{header}"
        );
        // The entry's own text, not only its round trip: the writer and the reader share
        // a name→field map, so a swapped pair (DEVIANCE ↔ PIT) would round-trip exactly
        // (review r1 #1). Row 2 holds the interval censor's empty DV; row 5 an event with
        // ΔH = 0, whose +∞ deviance the bundle writes `inf` where the sdtab leaves it empty.
        let entry = entry_text(&path, "predictions_tte.csv").unwrap();
        let lines: Vec<&str> = entry.lines().collect();
        assert_eq!(lines[0], "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE,PIT");
        assert_eq!(
            lines[1],
            "S1,2,5.0,1.0,-1e-17,0.30000000000000004,-0.0,1e-300"
        );
        assert_eq!(lines[2], "S1,2,7.5,,0.75,0.30000000000000004,-0.5,1e-300");
        assert_eq!(lines[5], "S1,2,6.0,1.0,0.0,0.30000000000000004,inf,1e-300");
        assert_eq!(lines.len(), 6);
    }

    /// T2 (#911), the exit condition: `sdtab()` and `write_sdtab_csv` of the restored
    /// binary + TTE fit equal the live ones, bit for bit and byte for byte.
    ///
    /// Dies on: `attach_tte_rows` skipped (the restored fit is refused), the blocks
    /// swapped (S0's and S2's rows change places under one ID), the refusal keyed on the
    /// TTE endpoint alone.
    #[cfg(feature = "survival")]
    #[test]
    fn restored_sdtab_equals_live() {
        use crate::io::output::{sdtab, write_sdtab_csv};
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_911();
        let loaded = load_fit(&save_911(&dir, "tte.fitrx", &r, &p)).unwrap().fit;
        assert!(loaded.restored_from_checkpoint, "premise: restored");

        let live = sdtab(&r, &p);
        let back = sdtab(&loaded, &p);
        let names = |c: &[(String, Vec<f64>)]| c.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(&back), names(&live));
        assert!(
            names(&live).iter().any(|n| n == "COXSNELL"),
            "premise: TTE columns"
        );
        for ((name, want), (_, got)) in live.iter().zip(&back) {
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(got), bits(want), "column {name}");
        }

        let live_csv = dir.path().join("live.csv");
        let back_csv = dir.path().join("back.csv");
        write_sdtab_csv(&r, &p, live_csv.to_str().unwrap()).unwrap();
        write_sdtab_csv(&loaded, &p, back_csv.to_str().unwrap())
            .expect("a restored fit with its TTE rows writes sdtab");
        assert_eq!(
            std::fs::read(&back_csv).unwrap(),
            std::fs::read(&live_csv).unwrap()
        );
    }

    /// `tte_fit_911` with npde scores on its records (#1889): full-digit values, one
    /// record left unscored (NaN), and a negative zero.
    #[cfg(feature = "survival")]
    fn tte_fit_1889() -> (FitResult, Population) {
        let (mut r, p) = tte_fit_911();
        r.subjects[0].tte_rows[0].npde = -1.25;
        r.subjects[2].tte_rows[0].npde = 0.1 + 0.2;
        r.subjects[2].tte_rows[1].npde = -0.0;
        r.subjects[2].tte_rows[2].npde = 2.807033768343811;
        (r, p)
    }

    #[cfg(feature = "survival")]
    fn tte_npde_bits(r: &FitResult) -> Vec<Vec<u64>> {
        r.subjects
            .iter()
            .map(|s| s.tte_rows.iter().map(|t| t.npde.to_bits()).collect())
            .collect()
    }

    /// T11 (#1889): a bundle whose TTE records carry an npde writes a ninth `NPDE`
    /// column, and the restored rows and the restored sdtab equal the live ones bit for
    /// bit, the score included.
    ///
    /// Dies on: the column not written (restored scores are NaN); the ninth cell read
    /// into `pit` (the entry text and the `pit` bits); a score written through a
    /// rounding format (`0.1 + 0.2`).
    #[cfg(feature = "survival")]
    #[test]
    fn roundtrip_tte_npde() {
        use crate::io::output::sdtab;
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_1889();
        let path = save_911(&dir, "npde.fitrx", &r, &p);
        let entry = entry_text(&path, "predictions_tte.csv").unwrap();
        let lines: Vec<&str> = entry.lines().collect();
        assert_eq!(
            lines[0],
            "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE,PIT,NPDE"
        );
        assert_eq!(
            lines[1],
            "S1,2,5.0,1.0,-1e-17,0.30000000000000004,-0.0,1e-300,-1.25"
        );
        // The unscored record: an empty ninth cell.
        assert_eq!(lines[2], "S1,2,7.5,,0.75,0.30000000000000004,-0.5,1e-300,");
        assert_eq!(
            lines[3],
            "S1,2,2.0,0.0,0.30000000000000004,0.30000000000000004,-0.75,1e-300,0.30000000000000004"
        );
        assert_eq!(lines.len(), 6);

        let loaded = load_fit(&path).unwrap().fit;
        assert_eq!(tte_npde_bits(&loaded), tte_npde_bits(&r));
        for (si, (got, want)) in loaded.subjects.iter().zip(&r.subjects).enumerate() {
            assert_eq!(
                tte_bits(&got.tte_rows),
                tte_bits(&want.tte_rows),
                "subject {si}"
            );
        }
        let (live, back) = (sdtab(&r, &p), sdtab(&loaded, &p));
        assert!(
            live.iter().any(|(n, _)| n == "NPDE"),
            "premise: NPDE column"
        );
        assert_eq!(live.len(), back.len());
        for ((name, want), (back_name, got)) in live.iter().zip(&back) {
            assert_eq!(name, back_name);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(got), bits(want), "column {name}");
        }
    }

    /// T12 (#1889): a fit with no TTE npde writes the entry it wrote before the score
    /// existed, byte for byte, and an eight-column entry loads with every score NaN,
    /// both the one this build writes and one cut down from a scored bundle.
    ///
    /// Dies on: the ninth column always written; only the nine-column header accepted
    /// (the bundle is `Corrupt`); a missing score read as 0.
    #[cfg(feature = "survival")]
    #[test]
    fn a_bundle_without_tte_npde_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (plain, p) = tte_fit_911();
        let path = save_911(&dir, "plain.fitrx", &plain, &p);
        assert_eq!(
            entry_text(&path, "predictions_tte.csv").unwrap(),
            "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE,PIT\n\
             S1,2,5.0,1.0,-1e-17,0.30000000000000004,-0.0,1e-300\n\
             S1,2,7.5,,0.75,0.30000000000000004,-0.5,1e-300\n\
             S1,2,2.0,0.0,0.30000000000000004,0.30000000000000004,-0.75,1e-300\n\
             S1,2,4.0,1.0,1e-300,0.30000000000000004,0.5,1e-300\n\
             S1,2,6.0,1.0,0.0,0.30000000000000004,inf,1e-300\n"
        );
        let loaded = load_fit(&path).unwrap().fit;
        assert!(loaded
            .subjects
            .iter()
            .all(|s| s.tte_rows.iter().all(|t| t.npde.is_nan())));
        assert_eq!(
            loaded
                .subjects
                .iter()
                .map(|s| s.tte_rows.len())
                .sum::<usize>(),
            5
        );

        // A scored bundle with its ninth column cut off by hand: the pre-#1889 shape.
        let (scored, p) = tte_fit_1889();
        let nine = save_911(&dir, "scored.fitrx", &scored, &p);
        let eight = dir.path().join("eight.fitrx");
        rewrite_bundle(&nine, &eight, |name, body| {
            if name != "predictions_tte.csv" {
                return Some(body);
            }
            let text = String::from_utf8(body).unwrap();
            let mut out = String::new();
            for line in text.lines() {
                out.push_str(&line[..line.rfind(',').unwrap()]);
                out.push('\n');
            }
            Some(out.into_bytes())
        });
        let loaded = load_fit(&eight).unwrap().fit;
        for (si, (got, want)) in loaded.subjects.iter().zip(&scored.subjects).enumerate() {
            assert_eq!(
                tte_bits(&got.tte_rows),
                tte_bits(&want.tte_rows),
                "subject {si}"
            );
            assert!(got.tte_rows.iter().all(|t| t.npde.is_nan()), "subject {si}");
        }
    }

    /// T13 (#1889): any header but the two accepted ones is `Corrupt`, and the message
    /// names both.
    ///
    /// Dies on: any header starting with the eight columns accepted (the ten-column
    /// and the misnamed ninth); the message naming one header only.
    #[cfg(feature = "survival")]
    #[test]
    fn other_tte_headers_are_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_1889();
        let good = save_911(&dir, "good.fitrx", &r, &p);
        let edited = dir.path().join("edited.fitrx");
        let eight = "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE,PIT";
        for header in [
            "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE".to_string(),
            format!("{eight},NPDE,NPD"),
            format!("{eight},NPD"),
            format!("{eight}X"),
            format!("{eight},"),
        ] {
            rewrite_bundle(&good, &edited, |name, body| {
                if name != "predictions_tte.csv" {
                    return Some(body);
                }
                let text = String::from_utf8(body).unwrap();
                let rest = &text[text.find('\n').unwrap()..];
                Some(format!("{header}{rest}").into_bytes())
            });
            let msg = load_corrupt(load_fit(&edited));
            assert_eq!(
                msg,
                format!(
                    "predictions_tte.csv: header {header:?}, expected {eight:?} or \
                     \"{eight},NPDE\""
                ),
                "header {header:?}"
            );
        }
    }

    /// The bundle at `src` as a writer before #911 left it: no `predictions_tte.csv`, and
    /// no `n_obs_tte` column in `ebes.csv` (the last column of a non-mixture fit).
    #[cfg(feature = "survival")]
    fn strip_tte_911(src: &Path, dst: &Path) {
        rewrite_bundle(src, dst, |name, body| match name {
            "predictions_tte.csv" => None,
            "ebes.csv" => {
                let text = String::from_utf8(body).unwrap();
                assert!(
                    text.lines().next().unwrap().ends_with(",n_obs_tte"),
                    "premise: n_obs_tte is the last column"
                );
                let mut out = String::new();
                for line in text.lines() {
                    out.push_str(&line[..line.rfind(',').unwrap()]);
                    out.push('\n');
                }
                Some(out.into_bytes())
            }
            _ => Some(body),
        });
    }

    /// T3 (#911): a bundle written before #911 still loads, with every `tte_rows` empty,
    /// and its sdtab is still refused, with the message saying the bundle predates the
    /// TTE rows. A load + re-save cannot recover them (cell A5): there are none to write.
    ///
    /// Dies on: the strip rejected as `Corrupt` (the entry made mandatory), the refusal
    /// keyed on the rows alone without the TTE endpoint (T6's binary cell), on a TTE
    /// endpoint with rows (T2).
    #[cfg(feature = "survival")]
    #[test]
    fn pre_911_bundle_keeps_the_refusal() {
        use crate::io::output::write_sdtab_csv;
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_911();
        let new = save_911(&dir, "new.fitrx", &r, &p);
        let old = dir.path().join("old.fitrx");
        strip_tte_911(&new, &old);
        assert_eq!(entry_text(&old, "predictions_tte.csv"), None);

        let loaded = load_fit(&old).expect("a pre-#911 bundle loads").fit;
        assert!(loaded.subjects.iter().all(|s| s.tte_rows.is_empty()));
        assert_eq!(
            loaded.subjects[0].discrete_rows,
            r.subjects[0].discrete_rows
        );
        let out = dir.path().join("sdtab.csv");
        let err = write_sdtab_csv(&loaded, &p, out.to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("written before time-to-event rows were bundled"),
            "{err}"
        );
        assert!(err.contains("#911"), "{err}");
        assert!(!out.exists());

        // A5: loading and saving again writes no entry, so the refusal stands.
        let again = save_911(&dir, "again.fitrx", &loaded, &p);
        assert_eq!(entry_text(&again, "predictions_tte.csv"), None);
        let reloaded = load_fit(&again).unwrap().fit;
        assert!(write_sdtab_csv(&reloaded, &p, out.to_str().unwrap()).is_err());
    }

    /// T4 (#911): the count column and the entry vouch for each other, and a row that does
    /// not fit its count is refused rather than given to a neighbour.
    ///
    /// Dies on: each guard of `attach_tte_rows` deleted in turn (its own assert).
    #[cfg(feature = "survival")]
    #[test]
    fn damaged_tte_rows_are_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.subjects[0].tte_rows = vec![tte_row_911(1.5, 1.0, 0.5, -0.5)];
        r.subjects[1].tte_rows = vec![tte_row_911(2.5, 0.0, 0.5, -0.5)];
        let good = saved(&dir, &r);
        let (s0, s1) = (r.subjects[0].id.clone(), r.subjects[1].id.clone());
        load_fit(&good).expect("premise: the unedited bundle loads");
        let edited = dir.path().join("edited.fitrx");
        let load_with = |edit: &dyn Fn(String) -> Option<String>| {
            rewrite_bundle(&good, &edited, |name, body| {
                if name == "predictions_tte.csv" {
                    edit(String::from_utf8(body).unwrap()).map(String::into_bytes)
                } else {
                    Some(body)
                }
            });
            load_corrupt(load_fit(&edited))
        };

        let no_entry = load_with(&|_| None);
        assert!(no_entry.contains("no predictions_tte.csv"), "{no_entry}");
        let one_row = load_with(&|csv| Some(csv.lines().take(2).collect::<Vec<_>>().join("\n")));
        assert!(one_row.contains("ran out of rows"), "{one_row}");
        let extra = load_with(&|csv| Some(format!("{csv}{s1},2,9.0,1.0,0.5,0.5,0.5,0.5\n")));
        assert!(extra.contains("past the rows"), "{extra}");
        let swapped = load_with(&|csv| Some(csv.replacen(&s0, "OTHER", 1)));
        assert!(swapped.contains("out of order"), "{swapped}");
        let bad_num = load_with(&|csv| Some(csv.replace("1.5", "abc")));
        assert!(bad_num.contains("bad number \"abc\""), "{bad_num}");
        let bad_cmt =
            load_with(&|csv| Some(csv.replacen(&format!("{s0},2,"), &format!("{s0},x,"), 1)));
        assert!(bad_cmt.contains("bad CMT \"x\""), "{bad_cmt}");
        let short = load_with(&|csv| {
            let mut lines: Vec<String> = csv.lines().map(String::from).collect();
            let cut = lines[1].rfind(',').unwrap();
            lines[1].truncate(cut);
            Some(lines.join("\n") + "\n")
        });
        assert!(short.contains("fields, expected 8"), "{short}");
        let bad_header = load_with(&|csv| Some(csv.replacen("PIT", "PITT", 1)));
        assert!(bad_header.contains("header"), "{bad_header}");

        rewrite_bundle(&good, &edited, |name, body| {
            Some(if name == "ebes.csv" {
                String::from_utf8(body)
                    .unwrap()
                    .replace(",n_obs_tte", ",other")
                    .into_bytes()
            } else {
                body
            })
        });
        let no_column = load_corrupt(load_fit(&edited));
        assert!(
            no_column.contains("ebes.csv has no n_obs_tte"),
            "{no_column}"
        );

        rewrite_bundle(&good, &edited, |name, body| {
            Some(if name == "ebes.csv" {
                let text = String::from_utf8(body).unwrap();
                let mut lines = text.lines();
                let mut out = format!("{}\n", lines.next().unwrap());
                for line in lines {
                    out.push_str(&line[..line.rfind(',').unwrap()]);
                    out.push_str(",-1\n");
                }
                out.into_bytes()
            } else {
                body
            })
        });
        let bad_count = load_corrupt(load_fit(&edited));
        assert!(bad_count.contains("bad n_obs_tte \"-1\""), "{bad_count}");
    }

    /// T5 (#911): a fit without TTE rows writes neither the entry nor the column, so its
    /// `ebes.csv` is byte for byte the layout a writer before #911 produced, plain and with
    /// discrete rows.
    ///
    /// Dies on: the entry or the column written unconditionally.
    #[test]
    fn a_fit_without_tte_rows_writes_neither_entry_nor_column() {
        let dir = tempfile::tempdir().unwrap();
        let plain = saved(&dir, &minimal_fit_result());
        assert_eq!(entry_text(&plain, "predictions_tte.csv"), None);
        let ebes = entry_text(&plain, "ebes.csv").unwrap();
        assert!(
            ebes.lines()
                .next()
                .unwrap()
                .ends_with(",ofv_contribution,n_obs"),
            "{ebes}"
        );
        assert!(ebes.lines().nth(1).unwrap().ends_with(",3"), "{ebes}");

        let mut r = minimal_fit_result();
        r.subjects[1].discrete_rows = vec![discrete_row(3, 2.5, 0.25)];
        let dir2 = tempfile::tempdir().unwrap();
        let discrete = saved(&dir2, &r);
        assert_eq!(entry_text(&discrete, "predictions_tte.csv"), None);
        let ebes = entry_text(&discrete, "ebes.csv").unwrap();
        let mut lines = ebes.lines();
        assert!(
            lines.next().unwrap().ends_with(",n_obs,n_obs_discrete"),
            "{ebes}"
        );
        assert!(lines.next().unwrap().ends_with(",3,0"), "{ebes}");
        assert!(lines.next().unwrap().ends_with(",2,1"), "{ebes}");
    }

    /// T7 (#911): the rows outlive a load and a second save, written from the restored
    /// rows (cell A6).
    ///
    /// Dies on: the second save dropping the entry, the reader not filling `tte_rows`
    /// (the writer then has nothing to write).
    #[cfg(feature = "survival")]
    #[test]
    fn tte_rows_survive_a_load_and_re_save() {
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = tte_fit_911();
        let loaded = load_fit(&save_911(&dir, "first.fitrx", &r, &p))
            .unwrap()
            .fit;
        let back = load_fit(&save_911(&dir, "again.fitrx", &loaded, &p))
            .unwrap()
            .fit;
        for (si, (got, want)) in back.subjects.iter().zip(&r.subjects).enumerate() {
            assert_eq!(
                tte_bits(&got.tte_rows),
                tte_bits(&want.tte_rows),
                "subject {si}"
            );
        }
    }

    /// T8 (#911): a build without `survival` loads a bundle that carries TTE rows (what a
    /// survival build writes, built here from text), ignoring the entry and the column,
    /// and keeps its discrete rows. Also the forward-compat proxy for a reader older than
    /// #911 (cell F1), which knows neither name.
    ///
    /// Dies on: an unconditional `attach_tte_rows`-style read of the entry or the column
    /// that refuses what it does not know.
    #[cfg(not(feature = "survival"))]
    #[test]
    fn tte_bundle_loads_without_survival() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.subjects[1].discrete_rows = vec![discrete_row(3, 2.5, 0.25)];
        let base = saved(&dir, &r);
        let (s0, s1) = (r.subjects[0].id.clone(), r.subjects[1].id.clone());
        let with_tte = dir.path().join("with_tte.fitrx");
        let tte_csv = format!(
            "ID,CMT,TIME,DV,COXSNELL,MARTINGALE,DEVIANCE,PIT\n\
             {s0},2,5.0,1.0,0.5,0.5,-0.5,0.6\n\
             {s1},2,6.0,0.0,0.25,-0.25,-0.7,0.7\n\
             {s1},2,7.0,,0.3,,,0.8\n"
        );
        let mut archive = ZipArchive::new(File::open(&base).unwrap()).unwrap();
        let mut out = ZipWriter::new(File::create(&with_tte).unwrap());
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).unwrap();
            let name = f.name().unwrap().to_string();
            let mut body = Vec::new();
            f.read_to_end(&mut body).unwrap();
            drop(f);
            if name == "ebes.csv" {
                let text = String::from_utf8(body).unwrap();
                let mut lines = text.lines();
                let mut edited = format!("{},n_obs_tte\n", lines.next().unwrap());
                for (line, n) in lines.zip([1, 2]) {
                    edited.push_str(&format!("{line},{n}\n"));
                }
                body = edited.into_bytes();
            }
            // Index the new entry, as a survival build's writer does (review r1 #5).
            if name == "manifest.json" {
                body = edit_json(body, |v| {
                    v["entries"]
                        .as_array_mut()
                        .unwrap()
                        .push("predictions_tte.csv".into())
                });
            }
            out.start_file(name, SimpleFileOptions::default()).unwrap();
            out.write_all(&body).unwrap();
        }
        out.start_file("predictions_tte.csv", SimpleFileOptions::default())
            .unwrap();
        out.write_all(tte_csv.as_bytes()).unwrap();
        out.finish().unwrap();

        let loaded = load_fit(&with_tte).expect("a TTE bundle loads without survival");
        assert!(
            loaded
                .manifest
                .entries
                .contains(&"predictions_tte.csv".to_string()),
            "premise: the fixture indexes the entry"
        );
        assert!(loaded.fit.subjects[0].discrete_rows.is_empty());
        assert_eq!(
            loaded.fit.subjects[1].discrete_rows,
            r.subjects[1].discrete_rows
        );
    }

    /// A covariate table whose declaration order is not alphabetical, so a reader taking
    /// the order from the `covariate_types` object would swap the columns.
    fn covariate_fixture() -> CovariateTable {
        CovariateTable {
            names: vec!["WT".into(), "AGE".into(), "SEX".into()],
            kinds: vec![
                CovariateKind::Continuous,
                CovariateKind::Continuous,
                CovariateKind::Categorical,
            ],
            rows: vec![
                CovariateRow {
                    id: "Smith, 2019".into(),
                    time: 0.0,
                    evid: 1,
                    values: vec![0.1 + 0.2, f64::NAN, 1.0],
                },
                CovariateRow {
                    id: "S2".into(),
                    time: f64::NAN,
                    evid: 0,
                    values: vec![1e-7, 40.0, 0.0],
                },
            ],
        }
    }

    /// #2695: `covtab.csv` and `covariate_types` round-trip the table: order, kinds, a
    /// quoted ID, a missing value, a missing TIME and exact doubles. `PartialEq` cannot
    /// see `NaN`, so those cells are asserted on their own.
    #[test]
    fn roundtrip_covariate_table() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        let table = covariate_fixture();
        r.covariate_table = Some(table.clone());
        let path = saved(&dir, &r);

        assert_eq!(
            fit_json(&path)["covariate_types"],
            serde_json::json!({"WT": "continuous", "AGE": "continuous", "SEX": "categorical"})
        );
        let got = load_fit(&path).unwrap().fit.covariate_table.unwrap();
        assert_eq!(got.names, table.names);
        assert_eq!(got.kinds, table.kinds);
        assert_eq!(got.rows.len(), 2);
        assert_eq!(got.rows[0].id, "Smith, 2019");
        assert_eq!(got.rows[0].time, 0.0);
        assert_eq!(got.rows[0].evid, 1);
        assert_eq!(got.rows[0].values[0], 0.1 + 0.2);
        assert!(got.rows[0].values[1].is_nan());
        assert_eq!(got.rows[0].values[2], 1.0);
        assert!(got.rows[1].time.is_nan());
        assert_eq!(got.rows[1].evid, 0);
        assert_eq!(got.rows[1].values, vec![1e-7, 40.0, 0.0]);
    }

    /// A table with no rows still comes back, and a fit with no table writes nothing.
    #[test]
    fn an_empty_covariate_table_round_trips_and_none_writes_no_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        let path = saved(&dir, &r);
        assert_eq!(entry_text(&path, "covtab.csv"), None);
        assert!(fit_json(&path).get("covariate_types").is_none());
        assert!(load_fit(&path).unwrap().fit.covariate_table.is_none());

        let mut table = covariate_fixture();
        table.rows.clear();
        r.covariate_table = Some(table.clone());
        let got = load_fit(&saved(&dir, &r)).unwrap().fit.covariate_table;
        assert_eq!(got, Some(table));
    }

    /// #2695: `covtab.csv` and `covariate_types` must describe the same covariates.
    #[test]
    fn a_covtab_disagreeing_with_its_types_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.covariate_table = Some(covariate_fixture());
        let good = saved(&dir, &r);
        let edited = dir.path().join("edited.fitrx");
        let load_with = |edit: &dyn Fn(&str, Vec<u8>) -> Option<Vec<u8>>| {
            rewrite_bundle(&good, &edited, edit);
            load_corrupt(load_fit(&edited))
        };
        let types = |edit: fn(&mut serde_json::Value)| {
            move |name: &str, body: Vec<u8>| {
                Some(if name == "fit.json" {
                    edit_json(body, |f| edit(&mut f["covariate_types"]))
                } else {
                    body
                })
            }
        };

        let no_entry = load_with(&|name, body| (name != "covtab.csv").then_some(body));
        assert!(no_entry.contains("no covtab.csv"), "{no_entry}");
        let no_type = load_with(&types(|t| {
            t.as_object_mut().unwrap().remove("AGE");
        }));
        assert!(
            no_type.contains("no covariate_types entry for \"AGE\""),
            "{no_type}"
        );
        let extra = load_with(&types(|t| t["CRCL"] = "continuous".into()));
        assert!(
            extra.contains("no column for covariate_types entry \"CRCL\""),
            "{extra}"
        );
        let unknown = load_with(&types(|t| t["SEX"] = "ordinal".into()));
        assert!(
            unknown.contains("unknown covariate type \"ordinal\""),
            "{unknown}"
        );
    }

    /// #2695: a table a reader would refuse is not written.
    #[test]
    fn save_fit_refuses_a_misshapen_covariate_table() {
        let dir = tempfile::tempdir().unwrap();
        let p = dummy_population(&["S1", "S2"], 3);
        let path = dir.path().join("refused.fitrx");
        for (table, want) in [
            (
                CovariateTable {
                    kinds: vec![CovariateKind::Continuous],
                    ..covariate_fixture()
                },
                "1 kinds for 3 names",
            ),
            (
                {
                    let mut t = covariate_fixture();
                    t.rows[1].values.pop();
                    t
                },
                "has 2 values for 3 names",
            ),
        ] {
            let mut r = minimal_fit_result();
            r.covariate_table = Some(table);
            match save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()) {
                Err(FitrxError::Unwritable(msg)) => assert!(msg.contains(want), "{msg}"),
                other => panic!("expected Unwritable, got {other:?}"),
            }
            assert!(!path.exists(), "a refused fit leaves nothing behind");
        }
    }

    /// #2695: the trace text comes back byte for byte, a header-only one included, and
    /// `trace_path` stays what it was: the temporary file need not exist any more.
    #[test]
    fn roundtrip_trace_csv() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        assert_eq!(entry_text(&saved(&dir, &r), "trace.csv"), None);

        r.trace_path = Some("/gone/trace.csv".into());
        for text in [
            "iter,method,ofv\n0,focei,12.500000\n1,focei,NA\n",
            "iter,method,ofv\n",
        ] {
            r.trace_csv = Some(text.into());
            let loaded = load_fit(&saved(&dir, &r)).unwrap();
            assert_eq!(loaded.fit.trace_csv.as_deref(), Some(text));
            assert_eq!(loaded.fit.trace_path.as_deref(), Some("/gone/trace.csv"));
            assert!(loaded.manifest.entries.contains(&"trace.csv".to_string()));
        }
    }

    /// #2695: core returns `interface_extras` as it was given: the key order and number
    /// spelling here are the ones a parse and re-serialize would change. It survives a
    /// second save, and is a separate entry so `fit.json` stays the engine's own.
    #[test]
    fn interface_extras_are_returned_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let text = "{ \"z_call\": {\"maxiter\": 1.50}, \"a_note\": \"µg\", \"big\": 1e400 }\n";
        let mut r = minimal_fit_result();
        r.interface_extras = Some(text.into());
        let path = saved(&dir, &r);
        assert_eq!(
            entry_text(&path, "interface_extras.json").as_deref(),
            Some(text)
        );
        assert!(fit_json(&path).get("interface_extras").is_none());

        let loaded = load_fit(&path).unwrap();
        assert_eq!(loaded.fit.interface_extras.as_deref(), Some(text));
        assert!(loaded
            .manifest
            .entries
            .contains(&"interface_extras.json".to_string()));
        let again = load_fit(&saved(&dir, &loaded.fit)).unwrap().fit;
        assert_eq!(again.interface_extras.as_deref(), Some(text));
    }

    #[test]
    fn a_fit_without_interface_extras_writes_no_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = saved(&dir, &minimal_fit_result());
        assert_eq!(entry_text(&path, "interface_extras.json"), None);
        assert_eq!(load_fit(&path).unwrap().fit.interface_extras, None);
    }

    /// The entry is named `.json`, and every other reader of the bundle takes it at its
    /// word, so text that is not JSON is refused at the save. On load core does not look.
    #[test]
    fn save_fit_refuses_interface_extras_that_are_not_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("refused.fitrx");
        let mut r = minimal_fit_result();
        r.interface_extras = Some("{not json".into());
        let p = dummy_population(&["S1", "S2"], 3);
        match save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()) {
            Err(FitrxError::Unwritable(msg)) => {
                assert!(
                    msg.starts_with("interface_extras is not valid JSON"),
                    "{msg}"
                )
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
        assert!(!path.exists());

        r.interface_extras = Some("{}".into());
        let good = saved(&dir, &r);
        let edited = dir.path().join("edited.fitrx");
        rewrite_bundle(&good, &edited, |name, body| {
            Some(if name == "interface_extras.json" {
                b"{not json".to_vec()
            } else {
                body
            })
        });
        let loaded = load_fit(&edited).expect("core does not interpret the entry");
        assert_eq!(loaded.fit.interface_extras.as_deref(), Some("{not json"));
    }

    /// #2799: `eval_budget_spent` and `evaluation_only` round-trip, and a bundle written
    /// before them (neither key present) loads both as `false`.
    #[test]
    fn roundtrip_status_facts() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.eval_budget_spent = true;
        r.evaluation_only = true;
        let path = saved(&dir, &r);
        let back = load_fit(&path).unwrap().fit;
        assert!(back.eval_budget_spent && back.evaluation_only);

        let edited = dir.path().join("old.fitrx");
        rewrite_bundle(&path, &edited, |name, body| {
            Some(if name == "fit.json" {
                edit_json(body, |f| {
                    let m = f.as_object_mut().unwrap();
                    assert!(m.remove("eval_budget_spent").is_some());
                    assert!(m.remove("evaluation_only").is_some());
                })
            } else {
                body
            })
        });
        let old = load_fit(&edited).unwrap().fit;
        assert!(!old.eval_budget_spent && !old.evaluation_only);
    }

    /// #2695: `cov_fixed` round-trips, its pattern not all one value, and holds to the
    /// matrix on both sides of the file.
    #[test]
    fn roundtrip_cov_fixed() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.covariance_matrix = Some(DMatrix::identity(3, 3));
        r.cov_fixed = vec![false, true, false];
        let path = saved(&dir, &r);
        assert_eq!(
            fit_json(&path)["cov_fixed"],
            serde_json::json!([false, true, false])
        );
        assert_eq!(load_fit(&path).unwrap().fit.cov_fixed, r.cov_fixed);

        let edited = dir.path().join("edited.fitrx");
        rewrite_bundle(&path, &edited, |name, body| {
            Some(if name == "fit.json" {
                edit_json(body, |f| f["cov_fixed"] = serde_json::json!([true, false]))
            } else {
                body
            })
        });
        let msg = load_corrupt(load_fit(&edited));
        assert_eq!(
            msg,
            "cov_fixed (2) does not match the covariance matrix (dim 3)"
        );

        r.covariance_matrix = None;
        let p = dummy_population(&["S1", "S2"], 3);
        let refused = dir.path().join("refused.fitrx");
        match save_fit(&r, &p, "src\n", &refused, SaveFitOptions::default()) {
            Err(FitrxError::Unwritable(msg)) => {
                assert_eq!(
                    msg,
                    "cov_fixed (3) does not match the covariance matrix (none)"
                )
            }
            other => panic!("expected Unwritable, got {other:?}"),
        }
    }

    /// #2695: the verdicts are in `fit.json` under their own names, come back on
    /// `LoadedFit`, and agree with the engine's functions on both the fit and its reload,
    /// which is the claim that lets a reader use either. Both values of each verdict.
    #[test]
    fn roundtrip_verdicts() {
        let dir = tempfile::tempdir().unwrap();
        let quiet = minimal_fit_result();
        let mut flagged = minimal_fit_result();
        flagged.left_init = Some(false);
        flagged.n_parameters = 3;
        flagged.theta_init = flagged.theta.clone();
        flagged.sigma_init = flagged.sigma.clone();
        flagged.omega_init = flagged.omega.clone();
        let mut cov = DMatrix::identity(3, 3);
        cov[(0, 1)] = 0.75;
        cov[(1, 0)] = 0.75;
        flagged.covariance_matrix = Some(cov);
        flagged
            .warnings
            .push("profiled level(s) on a declared bound: TVCL".into());
        crate::api::rebuild_warnings_structured(&mut flagged);
        assert_eq!(
            crate::model_selection::stalled_at_init(&flagged),
            Some(true)
        );
        assert!(crate::model_selection::estimate_near_boundary(&flagged));
        assert_eq!(
            crate::model_selection::max_abs_correlation(&flagged),
            Some(0.75)
        );

        let path = saved(&dir, &flagged);
        let wire = fit_json(&path);
        assert_eq!(wire["stalled_at_init"], serde_json::json!(true));
        assert_eq!(wire["estimate_near_boundary"], serde_json::json!(true));
        assert_eq!(wire["max_abs_correlation"], serde_json::json!(0.75));
        let loaded = load_fit(&path).unwrap();
        assert_eq!(
            loaded.verdicts,
            FitVerdicts {
                stalled_at_init: Some(true),
                estimate_near_boundary: true,
                max_abs_correlation: Some(0.75),
            }
        );
        assert_eq!(
            crate::model_selection::stalled_at_init(&loaded.fit),
            Some(true)
        );
        assert!(crate::model_selection::estimate_near_boundary(&loaded.fit));
        assert_eq!(
            crate::model_selection::max_abs_correlation(&loaded.fit),
            Some(0.75)
        );

        let loaded = load_fit(&saved(&dir, &quiet)).unwrap();
        assert_eq!(
            loaded.verdicts,
            FitVerdicts {
                stalled_at_init: crate::model_selection::stalled_at_init(&quiet),
                estimate_near_boundary: false,
                max_abs_correlation: Some(0.0),
            }
        );

        let mut no_matrix = minimal_fit_result();
        no_matrix.covariance_matrix = None;
        let path = saved(&dir, &no_matrix);
        assert!(fit_json(&path)["max_abs_correlation"].is_null());
        assert_eq!(load_fit(&path).unwrap().verdicts.max_abs_correlation, None);
    }

    /// #2695: `bayes` round-trips, with the `NaN` a one-chain R̂ gives in a summary, in
    /// `max_rhat` and in a retained draw. Without `nan_tolerant` the `null` fails the load.
    #[test]
    fn roundtrip_bayes() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.bayes = Some(BayesResult {
            summaries: vec![PosteriorSummary {
                name: "TVCL".into(),
                mean: 0.1 + 0.2,
                sd: 0.5,
                q025: 0.1,
                median: 0.3,
                q975: 0.7,
                rhat: f64::NAN,
                ess_bulk: 100.5,
                ess_tail: 90.25,
                mcse: 0.01,
            }],
            n_chains: 1,
            n_warmup: 200,
            n_draws_per_chain: 300,
            n_divergent: 4,
            max_rhat: f64::NAN,
            draws: Some(vec![1.5, f64::NAN]),
        });
        let got = load_fit(&saved(&dir, &r)).unwrap().fit.bayes.unwrap();
        assert_eq!(got.summaries[0].name, "TVCL");
        assert_eq!(got.summaries[0].mean, 0.1 + 0.2);
        assert_eq!(got.summaries[0].ess_tail, 90.25);
        assert!(got.summaries[0].rhat.is_nan());
        assert!(got.max_rhat.is_nan());
        assert_eq!(
            (
                got.n_chains,
                got.n_warmup,
                got.n_draws_per_chain,
                got.n_divergent
            ),
            (1, 200, 300, 4)
        );
        let draws = got.draws.unwrap();
        assert_eq!(draws[0], 1.5);
        assert!(draws[1].is_nan());
    }

    /// #2695: `importance_sampling` round-trips. `proposal_df` straddles the one reader
    /// rule: `∞` (written `null`) comes back `∞`, a finite df as itself, while a `NaN`
    /// elsewhere stays `NaN`.
    #[test]
    fn roundtrip_importance_sampling() {
        let dir = tempfile::tempdir().unwrap();
        for df in [f64::INFINITY, 4.0] {
            let mut r = minimal_fit_result();
            r.importance_sampling = Some(ImportanceSamplingResult {
                minus2_log_likelihood: 123.25,
                mc_standard_error: f64::NAN,
                low_ess_subjects: vec![("S3".into(), 0.05)],
                n_samples: 500,
                proposal_df: df,
                ess_min: 0.05,
                ess_median: 0.5,
                kappa_treatment: KappaTreatment::Marginalized,
                adapt_passes: 2,
            });
            let got = load_fit(&saved(&dir, &r))
                .unwrap()
                .fit
                .importance_sampling
                .unwrap();
            assert_eq!(got.proposal_df, df);
            assert_eq!(got.minus2_log_likelihood, 123.25);
            assert!(got.mc_standard_error.is_nan());
            assert_eq!(got.low_ess_subjects, vec![("S3".to_string(), 0.05)]);
            assert_eq!((got.n_samples, got.adapt_passes), (500, 2));
            assert_eq!(got.kappa_treatment, KappaTreatment::Marginalized);
        }
    }

    /// #2695: `impmap_trace` round-trips, its sentinel iteration and a `NaN` OFV included.
    #[test]
    fn roundtrip_impmap_trace() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.impmap_trace = Some(ImpmapTrace {
            rows: vec![ImpmapTraceRow {
                iteration: -1_000_000_000,
                theta: vec![0.1 + 0.2],
                omega_lower_tri: vec![0.04, f64::NAN],
                sigma: vec![0.5],
                ofv: f64::NAN,
            }],
            theta_names: vec!["THETA1".into()],
            omega_names: vec!["OMEGA(1,1)".into(), "OMEGA(2,1)".into()],
            sigma_names: vec!["SIGMA(1)".into()],
        });
        let got = load_fit(&saved(&dir, &r))
            .unwrap()
            .fit
            .impmap_trace
            .unwrap();
        assert_eq!(got.theta_names, vec!["THETA1"]);
        assert_eq!(got.omega_names, vec!["OMEGA(1,1)", "OMEGA(2,1)"]);
        assert_eq!(got.sigma_names, vec!["SIGMA(1)"]);
        let row = &got.rows[0];
        assert_eq!(row.iteration, -1_000_000_000);
        assert_eq!(row.theta, vec![0.1 + 0.2]);
        assert_eq!(row.omega_lower_tri[0], 0.04);
        assert!(row.omega_lower_tri[1].is_nan());
        assert_eq!(row.sigma, vec![0.5]);
        assert!(row.ofv.is_nan());
    }

    /// #2695: `exclusions` round-trips, every count distinct so a swap cannot pass.
    #[test]
    fn roundtrip_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = minimal_fit_result();
        r.exclusions = Some(ExclusionSummary {
            excluded_subject_ids: vec!["S9".into()],
            n_obs_excluded: 1,
            obs_cmts_excluded: vec![2, 3],
            n_dose_excluded: 4,
            n_other_excluded: 5,
            n_records_total: 60,
            fired_ignore: vec!["AGE > 90".into()],
            fired_accept: vec!["SEX == 1".into()],
        });
        let got = load_fit(&saved(&dir, &r)).unwrap().fit.exclusions.unwrap();
        assert_eq!(got.excluded_subject_ids, vec!["S9"]);
        assert_eq!(got.n_obs_excluded, 1);
        assert_eq!(got.obs_cmts_excluded, vec![2, 3]);
        assert_eq!(got.n_dose_excluded, 4);
        assert_eq!(got.n_other_excluded, 5);
        assert_eq!(got.n_records_total, 60);
        assert_eq!(got.fired_ignore, vec!["AGE > 90"]);
        assert_eq!(got.fired_accept, vec!["SEX == 1"]);
    }

    /// The optional engine results are absent from `fit.json` for a fit without them, and
    /// load as `None`.
    #[test]
    fn a_fit_without_engine_results_writes_no_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = saved(&dir, &minimal_fit_result());
        let wire = fit_json(&path);
        for key in [
            "bayes",
            "importance_sampling",
            "impmap_trace",
            "exclusions",
            "cov_fixed",
        ] {
            assert!(wire.get(key).is_none(), "{key}");
        }
        let fit = load_fit(&path).unwrap().fit;
        assert!(fit.bayes.is_none() && fit.importance_sampling.is_none());
        assert!(fit.impmap_trace.is_none() && fit.exclusions.is_none());
    }

    // --- Atomic write, hash checks, in-memory bundles (#2696) -----------------

    const DATA_2696: &str = "ID,TIME,DV,EVID,AMT,CMT\n1,0,0,1,100,1\n1,1,5,0,0,1\n";

    fn fixture_2696() -> (FitResult, Population) {
        (minimal_fit_result(), dummy_population(&["S1", "S2"], 3))
    }

    /// A bundle's entries by name, with the manifest's timestamp blanked: it is the only
    /// thing two saves of one fit may differ in.
    fn entries_2696(bundle: &[u8]) -> BTreeMap<String, Vec<u8>> {
        let mut archive = ZipArchive::new(Cursor::new(bundle)).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.name_for_index(i).unwrap().unwrap().into_owned())
            .collect();
        let mut out: BTreeMap<String, Vec<u8>> = names
            .iter()
            .map(|n| (n.clone(), read_bytes(&mut archive, n).unwrap()))
            .collect();
        let mut manifest: Manifest = serde_json::from_slice(&out["manifest.json"]).unwrap();
        manifest.created_at.clear();
        out.insert(
            "manifest.json".into(),
            serde_json::to_vec(&manifest).unwrap(),
        );
        out
    }

    fn manifest_2696(bundle: &[u8]) -> Manifest {
        let mut archive = ZipArchive::new(Cursor::new(bundle)).unwrap();
        read_json(&mut archive, "manifest.json").unwrap()
    }

    fn dir_names_2696(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn assert_refused_2696(err: FitrxError, wants: &[&str]) {
        let FitrxError::Unwritable(msg) = err else {
            panic!("expected Unwritable, got {err:?}");
        };
        for want in wants {
            assert!(msg.contains(want), "{want:?} not in: {msg}");
        }
    }

    /// The write itself failing after partial output: the old file survives and the
    /// partial one does not.
    ///
    /// Mutation: writing to `path` directly reddens the content assertion, and leaking
    /// the temporary file the directory listing.
    #[test]
    fn a_write_failing_midway_leaves_the_existing_file_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.fitrx");
        std::fs::write(&path, b"the old bundle").unwrap();
        let err = replace_file(&path, |file| {
            file.write_all(b"partial")?;
            Err(std::io::Error::other("disk full"))
        })
        .unwrap_err();
        assert!(err.to_string().contains("disk full"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), b"the old bundle");
        assert_eq!(dir_names_2696(dir.path()), ["run.fitrx"]);
    }

    /// `save_fit` failing after it has started on the bundle: `include_data` is the last
    /// thing it reads, and it used to fail with the target already truncated.
    #[test]
    fn a_failed_save_leaves_the_existing_bundle_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.fitrx");
        let (r, p) = fixture_2696();
        save_fit(&r, &p, "src\n", &path, SaveFitOptions::default()).unwrap();
        let before = std::fs::read(&path).unwrap();
        let missing = dir.path().join("gone.csv");
        let err = save_fit(
            &r,
            &p,
            "src\n",
            &path,
            SaveFitOptions {
                include_data: Some(missing.clone()),
            },
        )
        .unwrap_err();
        assert!(matches!(err, FitrxError::Io(_)), "{err:?}");
        assert!(err.to_string().contains("gone.csv"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(dir_names_2696(dir.path()), ["run.fitrx"]);
    }

    /// The rename is the step that can still fail once the bytes are written.
    #[test]
    fn a_failed_rename_leaves_no_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("taken");
        std::fs::create_dir(&path).unwrap();
        let err = replace_file(&path, |file| file.write_all(b"x")).unwrap_err();
        assert!(matches!(err, FitrxError::Io(_)), "{err:?}");
        assert_eq!(dir_names_2696(dir.path()), ["taken"]);
        assert!(dir_names_2696(&path).is_empty());
    }

    #[test]
    fn a_bare_file_name_is_written_in_the_working_directory() {
        assert_eq!(parent_dir(Path::new("run.fitrx")), Path::new("."));
        assert_eq!(parent_dir(Path::new("a/run.fitrx")), Path::new("a"));
    }

    /// A temporary file is 0600; a bundle has to come out as `File::create` left it, or
    /// a run directory shared with a group stops being readable by it.
    #[cfg(unix)]
    #[test]
    fn the_rename_keeps_what_file_create_would_have_left() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let dir = tempfile::tempdir().unwrap();

        let reference = dir.path().join("reference");
        File::create(&reference).unwrap();
        let new = dir.path().join("new.fitrx");
        replace_file(&new, |file| file.write_all(b"x")).unwrap();
        assert_eq!(mode(&new), mode(&reference));

        let shared = dir.path().join("shared.fitrx");
        std::fs::write(&shared, b"old").unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o640)).unwrap();
        replace_file(&shared, |file| file.write_all(b"new")).unwrap();
        assert_eq!(mode(&shared), 0o640);
        assert_eq!(std::fs::read(&shared).unwrap(), b"new");

        let link = dir.path().join("latest.fitrx");
        std::os::unix::fs::symlink(&shared, &link).unwrap();
        replace_file(&link, |file| file.write_all(b"via link")).unwrap();
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&shared).unwrap(), b"via link");

        // `File::create` fails on a read-only file; a rename alone would not.
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o440)).unwrap();
        let err = replace_file(&shared, |file| file.write_all(b"no")).unwrap_err();
        assert!(err.to_string().contains("read-only"), "{err}");
        assert_eq!(std::fs::read(&shared).unwrap(), b"via link");
    }

    /// The link stays a link and its referent appears, as with `File::create`, through a
    /// chain and a relative link. A link into a missing directory and a cycle both fail
    /// with the link untouched.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_creates_its_referent() {
        use std::os::unix::fs::symlink;
        let is_link = |p: &Path| {
            std::fs::symlink_metadata(p)
                .unwrap()
                .file_type()
                .is_symlink()
        };
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();

        let referent = sub.join("real.fitrx");
        let inner = sub.join("inner.fitrx");
        let outer = dir.path().join("outer.fitrx");
        // Relative to `sub`, where the link is, not to the working directory.
        symlink("real.fitrx", &inner).unwrap();
        symlink(&inner, &outer).unwrap();
        replace_file(&outer, |file| file.write_all(b"new")).unwrap();
        assert_eq!(std::fs::read(&referent).unwrap(), b"new");
        assert!(is_link(&inner) && is_link(&outer));
        assert_eq!(
            dir_names_2696(&sub),
            ["inner.fitrx", "real.fitrx"],
            "no temporary file left, and nothing written beside the outer link"
        );

        let nowhere = dir.path().join("nowhere.fitrx");
        symlink(dir.path().join("missing/real.fitrx"), &nowhere).unwrap();
        let err = replace_file(&nowhere, |file| file.write_all(b"x")).unwrap_err();
        assert!(matches!(err, FitrxError::Io(_)), "{err:?}");
        assert!(is_link(&nowhere));

        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        symlink(&b, &a).unwrap();
        symlink(&a, &b).unwrap();
        let err = replace_file(&a, |file| file.write_all(b"x")).unwrap_err();
        assert!(err.to_string().contains("too many levels"), "{err}");
        assert!(is_link(&a) && is_link(&b));
    }

    /// An edited file is refused by name with the bundle untouched, the file it hashes
    /// to is accepted, and a fit with no hash has nothing to check against.
    #[test]
    fn data_edited_since_the_fit_is_refused_and_named() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("pk_study.csv");
        std::fs::write(&data, DATA_2696).unwrap();
        let path = dir.path().join("run.fitrx");
        let (mut r, p) = fixture_2696();
        r.data_hash = Some(crate::io::hash::sha256_bytes(DATA_2696.as_bytes()));
        let opts = || SaveFitOptions {
            include_data: Some(data.clone()),
        };
        save_fit(&r, &p, "src\n", &path, opts()).expect("the fitted bytes are bundled");
        let before = std::fs::read(&path).unwrap();

        let edited = DATA_2696.replace("1,1,5", "1,1,6");
        std::fs::write(&data, &edited).unwrap();
        let err = save_fit(&r, &p, "src\n", &path, opts()).unwrap_err();
        let actual = crate::io::hash::sha256_bytes(edited.as_bytes());
        assert_refused_2696(
            err,
            &[
                data.to_str().unwrap(),
                "not the data that was fitted",
                &actual,
                r.data_hash.as_deref().unwrap(),
            ],
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);

        r.data_hash = None;
        let bytes = save_fit_to_bytes(&r, &p, "src\n", opts()).unwrap();
        assert_eq!(entries_2696(&bytes)["data.csv"], edited.as_bytes());
    }

    /// Both places the text can come from, `model_text` winning as it does for the
    /// entry, so a stale argument beside a good `model_text` is not refused.
    #[test]
    fn model_text_that_is_not_the_hashed_model_is_refused() {
        let (mut r, p) = fixture_2696();
        let save =
            |r: &FitResult, src: &str| save_fit_to_bytes(r, &p, src, SaveFitOptions::default());
        r.model_text = None;
        r.model_hash = Some(crate::io::hash::sha256_bytes(b"fitted\n"));
        r.model_path = Some("models/run1.ferx".into());
        save(&r, "fitted\n").expect("the fitted text");
        assert_refused_2696(
            save(&r, "edited\n").unwrap_err(),
            &["models/run1.ferx", "not the model that was fitted"],
        );

        r.model_text = Some("fitted\n".into());
        let bytes = save(&r, "edited\n").expect("model_text is what is written");
        assert_eq!(entries_2696(&bytes)["model.ferx"], b"fitted\n");
        r.model_text = Some("edited\n".into());
        assert_refused_2696(
            save(&r, "fitted\n").unwrap_err(),
            &["not the model that was fitted"],
        );

        r.model_path = None;
        assert_refused_2696(save(&r, "x").unwrap_err(), &["the model source"]);
        r.model_hash = None;
        save(&r, "x").expect("no recorded hash, nothing to compare");
    }

    #[test]
    fn bytes_and_a_file_hold_the_same_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data.csv");
        std::fs::write(&data, DATA_2696).unwrap();
        let path = dir.path().join("run.fitrx");
        let (mut r, p) = fixture_2696();
        r.interface_extras = Some(r#"{"k": 1}"#.into());
        let opts = || SaveFitOptions {
            include_data: Some(data.clone()),
        };
        save_fit(&r, &p, "src\n", &path, opts()).unwrap();
        let bytes = save_fit_to_bytes(&r, &p, "src\n", opts()).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(entries_2696(&bytes), entries_2696(&on_disk));
        assert_eq!(
            manifest_2696(&bytes).entries,
            manifest_2696(&on_disk).entries
        );

        let from_file = load_fit(&path).unwrap();
        let from_bytes = load_fit_from_bytes(&bytes).unwrap();
        assert_eq!(from_bytes.model_source, from_file.model_source);
        assert_eq!(from_bytes.fit.theta, from_file.fit.theta);
        assert_eq!(from_bytes.fit.interface_extras, r.interface_extras);
        assert!(from_bytes.population.is_some() && from_file.population.is_some());

        let bare = load_fit_from_bytes_without_population(&bytes).unwrap();
        assert!(bare.population.is_none());
        assert_eq!(bare.fit.theta, from_file.fit.theta);
    }

    #[test]
    fn bytes_that_are_not_a_current_bundle_do_not_load() {
        assert!(matches!(
            load_fit_from_bytes(b"not a zip file"),
            Err(FitrxError::Zip(_) | FitrxError::Io(_))
        ));
        let (r, p) = fixture_2696();
        let good = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        let mut manifest = manifest_2696(&good);
        manifest.format_version = "1".into();
        write_manifest(&mut zip, &manifest).unwrap();
        let old = zip.finish().unwrap().into_inner();
        for result in [
            load_fit_from_bytes(&old).map(|_| ()),
            bundle_with_data(&old, b"x", "d.csv").map(|_| ()),
            bundle_with_interface_extras(&old, None).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(FitrxError::UnsupportedVersion(ref v)) if v == "1"),
                "{result:?}"
            );
        }
    }

    /// Adding data and extras to bytes gives the bundle a save with them gives, down to
    /// the manifest's entry order.
    #[test]
    fn edited_bytes_match_a_bundle_saved_with_the_same_content() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data.csv");
        std::fs::write(&data, DATA_2696).unwrap();
        let (mut r, p) = fixture_2696();
        r.data_hash = Some(crate::io::hash::sha256_bytes(DATA_2696.as_bytes()));
        let plain = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        assert!(!entries_2696(&plain).contains_key("data.csv"));

        // Extras first: `data.csv` still has to land ahead of them.
        let extras = r#"{ "a" : 1.50 }"#;
        let edited = bundle_with_interface_extras(&plain, Some(extras)).unwrap();
        let edited = bundle_with_data(&edited, DATA_2696.as_bytes(), "data.csv").unwrap();

        r.interface_extras = Some(extras.into());
        let saved = save_fit_to_bytes(
            &r,
            &p,
            "src\n",
            SaveFitOptions {
                include_data: Some(data),
            },
        )
        .unwrap();
        assert_eq!(entries_2696(&edited), entries_2696(&saved));
        assert_eq!(
            manifest_2696(&edited).entries,
            manifest_2696(&saved).entries
        );
        assert_eq!(
            manifest_2696(&edited).created_at,
            manifest_2696(&plain).created_at
        );

        let loaded = load_fit_from_bytes(&edited).unwrap();
        assert_eq!(loaded.fit.interface_extras.as_deref(), Some(extras));
        assert!(loaded.population.is_some());
    }

    #[test]
    fn data_added_to_bytes_is_checked_against_the_bundles_hash() {
        let (mut r, p) = fixture_2696();
        r.data_hash = Some(crate::io::hash::sha256_bytes(DATA_2696.as_bytes()));
        let bundle = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let err = bundle_with_data(&bundle, b"ID,TIME\n", "pk_study.csv").unwrap_err();
        assert_refused_2696(err, &["pk_study.csv", "not the data that was fitted"]);

        // With no hash a second call replaces the first, leaving one entry.
        r.data_hash = None;
        let bundle = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let once = bundle_with_data(&bundle, b"first\n", "a.csv").unwrap();
        let twice = bundle_with_data(&once, b"second\n", "b.csv").unwrap();
        assert_eq!(entries_2696(&twice)["data.csv"], b"second\n");
        let listed = manifest_2696(&twice).entries;
        assert_eq!(listed.iter().filter(|e| *e == "data.csv").count(), 1);
    }

    /// The set, replace and remove arms, each leaving `data.csv` and the rest alone.
    #[test]
    fn interface_extras_can_be_replaced_and_removed_on_bytes() {
        let (r, p) = fixture_2696();
        let plain = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let with_data = bundle_with_data(&plain, DATA_2696.as_bytes(), "d.csv").unwrap();
        let one = bundle_with_interface_extras(&with_data, Some("[1]")).unwrap();
        let two = bundle_with_interface_extras(&one, Some("[2]")).unwrap();
        assert_eq!(entries_2696(&two)[EXTRAS_ENTRY], b"[2]");
        assert_eq!(entries_2696(&two)["data.csv"], DATA_2696.as_bytes());

        let removed = bundle_with_interface_extras(&two, None).unwrap();
        assert_eq!(entries_2696(&removed), entries_2696(&with_data));
        assert!(!manifest_2696(&removed)
            .entries
            .iter()
            .any(|e| e == EXTRAS_ENTRY));

        let err = bundle_with_interface_extras(&plain, Some("{not json")).unwrap_err();
        assert_refused_2696(err, &["interface_extras is not valid JSON"]);
    }

    /// Nothing in the loaded fit holds TIME, DV or OCC, so this fails on any re-save
    /// that does not carry them. The values are off the dummy population's defaults and
    /// the raw times differ from `obs_times`, so a fallback cannot produce them.
    #[test]
    fn a_fit_loaded_without_data_can_be_saved_again() {
        let dir = tempfile::tempdir().unwrap();
        let (mut r, mut p) = fixture_2696();
        // `load_fit` always fills this from `model.ferx`, so a fit without it would
        // differ in `fit.json` for a reason that has nothing to do with the re-save.
        r.model_text = Some("src\n".into());
        for s in &mut p.subjects {
            s.obs_raw_times = vec![0.25, 1.5, 24.125];
            s.observations = vec![1.1, f64::NAN, 3.3];
            s.occasions = vec![1, 1, 2];
        }
        let first = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();

        let loaded = load_fit_from_bytes_without_population(&first).unwrap();
        assert!(loaded.population.is_none());
        let path = dir.path().join("second.fitrx");
        save_fit_from_bundle(&loaded.fit, &first, &path, SaveFitOptions::default()).unwrap();

        let (a, b) = (
            entries_2696(&first),
            entries_2696(&std::fs::read(&path).unwrap()),
        );
        let preds = std::str::from_utf8(&b["predictions.csv"]).unwrap();
        assert!(
            preds.contains(",OCC") && preds.contains("24.125"),
            "{preds}"
        );
        assert_eq!(a, b);
    }

    /// With no `model_text` the bundle's `model.ferx` is the text written.
    #[test]
    fn a_fit_without_model_text_takes_the_bundles() {
        let dir = tempfile::tempdir().unwrap();
        let (r, p) = fixture_2696();
        let bundle = save_fit_to_bytes(&r, &p, "bundled\n", SaveFitOptions::default()).unwrap();
        let mut fit = load_fit_from_bytes(&bundle).unwrap().fit;
        fit.model_text = None;
        let path = dir.path().join("out.fitrx");
        save_fit_from_bundle(&fit, &bundle, &path, SaveFitOptions::default()).unwrap();
        let written = entries_2696(&std::fs::read(&path).unwrap());
        assert_eq!(written["model.ferx"], b"bundled\n");
    }

    /// A bundled `data.csv` still answers to the fit's hash, and `include_data` replaces it.
    #[test]
    fn a_resave_keeps_the_bundled_data() {
        let dir = tempfile::tempdir().unwrap();
        let (mut r, p) = fixture_2696();
        r.data_hash = Some(crate::io::hash::sha256_bytes(DATA_2696.as_bytes()));
        let data = dir.path().join("data.csv");
        std::fs::write(&data, DATA_2696).unwrap();
        let bytes = save_fit_to_bytes(
            &r,
            &p,
            "src\n",
            SaveFitOptions {
                include_data: Some(data.clone()),
            },
        )
        .unwrap();
        let mut fit = load_fit_from_bytes_without_population(&bytes).unwrap().fit;
        let out = dir.path().join("out.fitrx");
        save_fit_from_bundle(&fit, &bytes, &out, SaveFitOptions::default()).unwrap();
        let kept = entries_2696(&std::fs::read(&out).unwrap());
        assert_eq!(kept["data.csv"], DATA_2696.as_bytes());

        std::fs::write(&data, "ID,TIME\n").unwrap();
        let opts = SaveFitOptions {
            include_data: Some(data.clone()),
        };
        let err = save_fit_from_bundle(&fit, &bytes, &out, opts.clone()).unwrap_err();
        assert_refused_2696(err, &[data.to_str().unwrap()]);

        fit.data_hash = None;
        save_fit_from_bundle(&fit, &bytes, &out, opts).unwrap();
        let replaced = entries_2696(&std::fs::read(&out).unwrap());
        assert_eq!(replaced["data.csv"], b"ID,TIME\n");

        fit.data_hash = Some("0".repeat(64));
        let err = save_fit_from_bundle(&fit, &bytes, &out, SaveFitOptions::default()).unwrap_err();
        assert_refused_2696(err, &["the bundled data.csv"]);
    }

    /// Two subjects with one ID and equal-length results, as a reused ID gives: the ID
    /// check cannot tell them apart, so only the values can.
    fn twins_2696() -> (FitResult, Vec<u8>) {
        let (mut r, p) = fixture_2696();
        r.subjects[1].id = r.subjects[0].id.clone();
        for (k, s) in r.subjects.iter_mut().enumerate() {
            let shift = k as f64 + 1.0;
            s.n_obs = 3;
            for v in [&mut s.pred, &mut s.ipred, &mut s.cwres, &mut s.iwres] {
                *v = vec![shift, shift * 2.0, f64::NAN];
            }
        }
        let bundle = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let fit = load_fit_from_bytes(&bundle).unwrap().fit;
        (fit, bundle)
    }

    /// TIME and DV would otherwise land on the wrong subject with no error. Each edit
    /// leaves every length alone, so only the identity check can see it, and nothing is
    /// written. The untouched fit passes, which also shows `NaN` matching `NaN`.
    #[test]
    fn subjects_that_are_not_the_bundles_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.fitrx");
        let save = |fit: &FitResult, bundle: &[u8]| {
            save_fit_from_bundle(fit, bundle, &out, SaveFitOptions::default())
        };

        let (r, p) = fixture_2696();
        let bundle = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let mut swapped = load_fit_from_bytes(&bundle).unwrap().fit;
        swapped.subjects.swap(0, 1);
        assert_refused_2696(
            save(&swapped, &bundle).unwrap_err(),
            &["subject 1 is \"S2\" in the fit and \"S1\" in the bundle"],
        );
        let mut renamed = load_fit_from_bytes(&bundle).unwrap().fit;
        renamed.subjects[1].id = "S9".into();
        assert_refused_2696(
            save(&renamed, &bundle).unwrap_err(),
            &["subject 2 is \"S9\""],
        );

        let (fit, bundle) = twins_2696();
        let mut swapped = fit.clone();
        swapped.subjects.swap(0, 1);
        assert_refused_2696(
            save(&swapped, &bundle).unwrap_err(),
            &["subject 1 (\"S1\") has a different PRED"],
        );
        assert!(!out.exists());
        save(&fit, &bundle).expect("the bundle's own subjects");
    }

    /// One column at a time, so a column dropped from the comparison reddens its own row.
    #[test]
    fn each_compared_column_can_refuse_on_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.fitrx");
        let (fit, bundle) = twins_2696();
        let columns: [(&str, fn(&mut SubjectResult) -> &mut Vec<f64>); 4] = [
            ("PRED", |s| &mut s.pred),
            ("IPRED", |s| &mut s.ipred),
            ("CWRES", |s| &mut s.cwres),
            ("IWRES", |s| &mut s.iwres),
        ];
        for (column, field) in columns {
            let mut edited = fit.clone();
            field(&mut edited.subjects[1])[0] += 1.0;
            let err = save_fit_from_bundle(&edited, &bundle, &out, SaveFitOptions::default())
                .unwrap_err();
            let want = format!("subject 2 (\"S1\") has a different {column} ");
            assert_refused_2696(err, &[&want]);
        }
    }

    /// `n_obs` alone edited passes every per-observation comparison, and the writer would
    /// state it in `ebes.csv` above a different number of `predictions.csv` rows. Both
    /// save paths, since the bundle `load_fit` calls corrupt was reachable through each.
    #[test]
    fn an_n_obs_that_is_not_the_prediction_count_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.fitrx");
        let (r, p) = fixture_2696();
        let bundle = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let mut fit = load_fit_from_bytes(&bundle).unwrap().fit;
        fit.subjects[1].n_obs += 1;
        let want = "subject \"S2\" has n_obs = 3 and 2 predictions";

        let err = save_fit_from_bundle(&fit, &bundle, &out, SaveFitOptions::default());
        assert_refused_2696(err.unwrap_err(), &[want]);
        let err = save_fit(&fit, &p, "src\n", &out, SaveFitOptions::default());
        assert_refused_2696(err.unwrap_err(), &[want]);
        assert!(!out.exists());

        fit.subjects[1].n_obs -= 1;
        save_fit_from_bundle(&fit, &bundle, &out, SaveFitOptions::default()).unwrap();
        load_fit(&out).expect("the unedited count loads");
    }

    #[test]
    fn same_values_needs_equal_length_and_treats_nan_as_equal() {
        assert!(same_values(&[1.0, f64::NAN], &[1.0, f64::NAN]));
        assert!(!same_values(&[1.0, f64::NAN], &[1.0, 2.0]));
        assert!(!same_values(&[1.0], &[1.0, 2.0]));
    }

    /// Rows that do not cover the fit are refused rather than written short or panicked
    /// on: fewer subjects than the bundle, and a `predictions.csv` with no TIME.
    #[test]
    fn a_bundle_whose_rows_do_not_cover_the_fit_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.fitrx");
        let (r, p) = fixture_2696();
        let bytes = save_fit_to_bytes(&r, &p, "src\n", SaveFitOptions::default()).unwrap();
        let save = |fit: &FitResult, bundle: &[u8]| {
            save_fit_from_bundle(fit, bundle, &out, SaveFitOptions::default())
        };

        let mut fewer = load_fit_from_bytes(&bytes).unwrap().fit;
        fewer.subjects.pop();
        assert_refused_2696(
            save(&fewer, &bytes).unwrap_err(),
            &["the bundle has 2 subjects, the fit has 1"],
        );

        let preds = String::from_utf8(entries_2696(&bytes)["predictions.csv"].clone()).unwrap();
        let no_time = preds.replacen("ID,TIME,", "ID,XTIME,", 1);
        let mut archive = ZipArchive::new(Cursor::new(&bytes[..])).unwrap();
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        for i in 0..archive.len() {
            let file = archive.by_index_raw(i).unwrap();
            if file.name().unwrap() != "predictions.csv" {
                zip.raw_copy_file(file).unwrap();
            }
        }
        zip.start_file("predictions.csv", bundle_entry_options())
            .unwrap();
        zip.write_all(no_time.as_bytes()).unwrap();
        let rewritten = zip.finish().unwrap().into_inner();

        let fit = load_fit_from_bytes(&rewritten)
            .expect("TIME is not needed to load")
            .fit;
        assert_refused_2696(save(&fit, &rewritten).unwrap_err(), &["0 TIME", "\"S1\""]);
        assert!(!out.exists());
    }

    #[test]
    fn bytes_that_are_not_a_bundle_are_refused_by_a_resave() {
        let dir = tempfile::tempdir().unwrap();
        let (r, _) = fixture_2696();
        let out = dir.path().join("out.fitrx");
        let err = save_fit_from_bundle(&r, b"not a zip", &out, SaveFitOptions::default());
        assert!(matches!(err, Err(FitrxError::Zip(_) | FitrxError::Io(_))));
        assert!(!out.exists());
    }
}
