//! Non-parametric case bootstrap (#1140).
//!
//! Resample whole subjects with replacement, refit the same model to each
//! replicate, and summarise the spread of the estimates into bias, standard
//! errors and confidence intervals. Feature target: `PsN::bootstrap`.
//!
//! # Why a tool and not a `ferx-core` feature
//!
//! This is 200+ calls to [`ferx_core::fit`] over resampled data and nothing
//! else — no new numerics. That is exactly the `ferx-tools` side of the #1114
//! boundary rule: *if it calls `fit()` more than once, it is a tool.*
//!
//! # What the bootstrap buys over `$COVARIANCE`
//!
//! ferx's standard errors are the pure `R⁻¹` matrix (NONMEM `$COVARIANCE
//! MATRIX=R`), which is asymptotic and symmetric by construction. The bootstrap
//! distribution assumes neither: it survives a failed or non-PD covariance step,
//! and it shows asymmetry in poorly identified parameters instead of averaging
//! it away. [`ParameterSummary`] reports both intervals side by side so the
//! disagreement between them is visible.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use ferx_core::{
    fit, CompiledModel, FitOptions, FitResult, ModelParameters, OmegaMatrix, Population,
    PreparedRun, SigmaVector, WarningCode,
};
use rayon::prelude::*;

pub mod journal;
pub mod manifest;
pub mod output;
pub mod resample;
pub mod summary;

pub use manifest::RunManifest;
pub use resample::{Replicate, SampleSize, Strata};
pub use summary::{BootstrapSummary, ParameterSummary};

/// Everything the bootstrap needs beyond the model and data themselves.
///
/// Defaults mirror PsN's, including the two skip filters that are on by default
/// there — a replicate whose minimization terminated, or whose estimate sits on
/// a boundary, is not a draw from the sampling distribution of a converged fit,
/// so including it biases the interval.
#[derive(Debug, Clone)]
pub struct BootstrapOptions {
    /// Number of bootstrap datasets. PsN's default, and its stated rule of
    /// thumb for standard errors, is 200.
    pub samples: usize,
    /// Master seed. Every replicate's draw is derived from this and its own
    /// index — see [`resample::replicate_seed`].
    pub seed: u64,
    pub sample_size: SampleSize,
    /// Column defining the resampling strata; `None` for an unstratified run.
    pub stratify_on: Option<String>,
    /// Start each replicate from the base fit's final estimates (PsN default).
    pub update_inits: bool,
    /// Fit the original dataset first. Required for `update_inits`, for bias,
    /// and for the standard-error intervals.
    pub run_base_model: bool,
    /// Run the covariance step for each replicate. Off by default: it is most
    /// of the cost, and the bootstrap standard error comes from the spread of
    /// the estimates, not from any per-replicate `R⁻¹`.
    pub keep_covariance: bool,
    /// Replicates to fit concurrently. `None` uses the Rayon default.
    pub threads: Option<usize>,
    pub skip_minimization_terminated: bool,
    pub skip_estimate_near_boundary: bool,
    pub skip_covariance_step_terminated: bool,
    pub skip_with_covstep_warnings: bool,
    /// Compute Δofv: evaluate each replicate's parameter vector on the
    /// *original* dataset with no estimation (`outer_maxiter = 0`, NONMEM
    /// `MAXEVAL=0`) and subtract the original fit's OFV.
    pub dofv: bool,
    /// Where the CSV artefacts are written. `None` writes nothing, and also
    /// disables the incremental journal that `resume` reads.
    pub directory: Option<PathBuf>,
    /// Two-sided confidence level, in percent.
    pub confidence_level: f64,
    /// Continue an interrupted run in `directory`, refitting only the sample
    /// indices its `raw_results.csv` does not already carry (#1143).
    ///
    /// Sound because a replicate's draw is a pure function of `(seed, index)`,
    /// so a reused replicate is bit-for-bit the one a fresh run would produce.
    pub resume: bool,
    /// With `resume`, refit the replicates whose recorded fit *errored* instead
    /// of carrying the failure forward.
    ///
    /// Off by default, matching PsN: a fit that failed usually fails again, and
    /// paying for it a second time to reach the same place is worse than the
    /// alternative. Turn it on when the failure was a transient resource
    /// problem — an out-of-memory kill, a full disk — rather than the model.
    pub retry_failed: bool,
}

impl BootstrapOptions {
    /// Checks that hold wherever these options are used — including
    /// [`resummarize`], which reaches the same statistics without going through
    /// [`run_bootstrap`].
    ///
    /// Splitting validation out is not symmetry for its own sake: before it
    /// existed, `--summarize --ci 100` walked past every check and hit the
    /// `normal_quantile` assertion, exiting 101 instead of reporting a bad
    /// argument.
    pub fn validate(&self) -> Result<(), String> {
        if self.samples == 0 {
            return Err("--samples must be at least 1".to_string());
        }
        if !(self.confidence_level > 0.0 && self.confidence_level < 100.0) {
            return Err(format!(
                "--ci must be a confidence level in (0, 100), got {}",
                self.confidence_level
            ));
        }
        Ok(())
    }

    /// Checks that only apply to a fresh run.
    ///
    /// The covariance filters read a diagnostic that only exists when the step
    /// actually ran, so asking for one without `--keep-covariance` would drop a
    /// filter the user requested without saying so. Deliberately *not* part of
    /// [`Self::validate`]: under `--summarize` the diagnostics are already in
    /// `raw_results.csv`, so filtering on them is exactly the point and needs no
    /// covariance step now.
    fn validate_for_run(&self) -> Result<(), String> {
        if !self.keep_covariance
            && (self.skip_covariance_step_terminated || self.skip_with_covstep_warnings)
        {
            return Err(
                "--skip-covariance-step-terminated / --skip-with-covstep-warnings filter on the \
                 covariance step, which is off for replicate fits by default. Add \
                 --keep-covariance (slower) or drop the filter."
                    .to_string(),
            );
        }
        if self.resume && self.directory.is_none() {
            return Err(
                "--resume continues the run in a directory, so it needs --directory naming one \
                 that an earlier run wrote."
                    .to_string(),
            );
        }
        if self.retry_failed && !self.resume {
            return Err(
                "--retry-failed re-fits replicates that a previous run recorded as failed, so it \
                 only means anything together with --resume."
                    .to_string(),
            );
        }
        Ok(())
    }
}

impl Default for BootstrapOptions {
    fn default() -> Self {
        BootstrapOptions {
            samples: 200,
            seed: 1,
            sample_size: SampleSize::Original,
            stratify_on: None,
            update_inits: true,
            run_base_model: true,
            keep_covariance: false,
            threads: None,
            skip_minimization_terminated: true,
            skip_estimate_near_boundary: true,
            skip_covariance_step_terminated: false,
            skip_with_covstep_warnings: false,
            dofv: false,
            directory: None,
            confidence_level: 95.0,
            resume: false,
            retry_failed: false,
        }
    }
}

/// One fitted replicate, reduced to what the summary and the raw-results file
/// need.
///
/// Deliberately not a [`FitResult`]: that carries per-subject diagnostics for
/// every individual, and holding 200 of them costs more memory than the whole
/// rest of the run.
#[derive(Debug, Clone)]
pub struct ReplicateResult {
    /// 1-based. Index 0 is the original dataset.
    pub index: usize,
    /// The flat parameter vector — see [`flatten_estimates`].
    pub estimates: Vec<f64>,
    /// Per-replicate standard errors, aligned with [`Self::estimates`] and
    /// present only when the covariance step ran. An individual entry is `None`
    /// where core reports no SE for that coordinate — an IOV *covariance*, for
    /// instance, since `se_kappa` carries only the diagonal variances.
    pub standard_errors: Option<Vec<Option<f64>>>,
    pub ofv: f64,
    pub converged: bool,
    pub estimate_near_boundary: bool,
    pub covariance_step_successful: bool,
    pub covariance_step_warnings: bool,
    /// Wall time for this fit, seconds.
    pub seconds: f64,
    /// `Some` when [`fit`] returned an error; the replicate is then excluded.
    pub error: Option<String>,
    /// Δofv against the original dataset, when `--dofv` was requested.
    pub delta_ofv: Option<f64>,
}

/// The full result of a bootstrap run.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Flat parameter names, matching [`ReplicateResult::estimates`].
    pub parameter_names: Vec<String>,
    /// The base-model fit on the original dataset, when it was run.
    pub original: Option<ReplicateResult>,
    /// One entry per requested sample, in index order.
    pub replicates: Vec<ReplicateResult>,
    /// Which subjects each replicate drew, index-aligned with `replicates`.
    pub draws: Vec<Replicate>,
    /// The original dataset's subject IDs, in dataset order.
    pub subject_ids: Vec<String>,
    pub summary: BootstrapSummary,
    /// Chi-square reference degrees of freedom for the Δofv plot: the number of
    /// estimated (non-fixed) parameters.
    pub n_estimated_parameters: usize,
    /// How many replicates `--resume` took from the run directory instead of
    /// refitting. Zero for a fresh run.
    pub n_reused: usize,
}

// ── flat parameter vector ───────────────────────────────────────────────────

/// One entry of the flat parameter vector.
///
/// Every consumer of the flat vector — names, estimates, standard errors, the
/// rebuild into [`ModelParameters`], the estimated-parameter count — is derived
/// from [`coordinates`] rather than re-deriving its own traversal. That is not
/// tidiness: five hand-written loops over "theta, then the free lower triangle,
/// then sigma" is five chances to drift, and a drift here is silent. Two of
/// them had already drifted before this enum existed — the parameter count
/// missed a block Omega's off-diagonals, and the standard errors were read in a
/// different order from the names they were written under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coord {
    Theta(usize),
    /// Between-subject Omega, lower triangle, `i >= j`.
    Omega(usize, usize),
    Sigma(usize),
    /// Inter-occasion Omega (kappa), lower triangle, `i >= j`.
    OmegaIov(usize, usize),
}

/// The flat parameter vector's coordinates, in order.
///
/// Theta, then the free lower triangle of the between-subject Omega, then
/// sigma, then the free lower triangle of the IOV Omega. The IOV block goes
/// last so that a model without IOV produces exactly the columns it did before
/// IOV was carried at all.
fn coordinates(template: &ModelParameters) -> Vec<Coord> {
    let mut out: Vec<Coord> = (0..template.theta.len()).map(Coord::Theta).collect();
    for i in 0..template.omega.dim() {
        for j in 0..=i {
            if template.omega.free_mask[(i, j)] {
                out.push(Coord::Omega(i, j));
            }
        }
    }
    out.extend((0..template.sigma.values.len()).map(Coord::Sigma));
    if let Some(iov) = &template.omega_iov {
        for i in 0..iov.dim() {
            for j in 0..=i {
                if iov.free_mask[(i, j)] {
                    out.push(Coord::OmegaIov(i, j));
                }
            }
        }
    }
    out
}

/// Whether this coordinate is held fixed, and so is not an estimated parameter.
///
/// The `*_fixed` flags are per-eta / per-kappa, not per-covariance-element, so a
/// block off-diagonal counts as fixed exactly when either of its two etas is.
fn is_fixed(template: &ModelParameters, coord: Coord) -> bool {
    let flag = |flags: &[bool], i: usize| flags.get(i).copied().unwrap_or(false);
    match coord {
        Coord::Theta(i) => flag(&template.theta_fixed, i),
        Coord::Omega(i, j) => flag(&template.omega_fixed, i) || flag(&template.omega_fixed, j),
        Coord::Sigma(i) => flag(&template.sigma_fixed, i),
        Coord::OmegaIov(i, j) => flag(&template.kappa_fixed, i) || flag(&template.kappa_fixed, j),
    }
}

/// Names for the flat parameter vector.
///
/// Omega entries are named by their etas — `OMEGA(CL,CL)` — rather than by
/// PsN's positional `OMEGA(1,1)`. The file is ferx's own, and a positional index
/// silently means something different the moment an eta is added.
pub fn parameter_names(template: &ModelParameters) -> Vec<String> {
    let eta = &template.omega.eta_names;
    let kappa = template
        .omega_iov
        .as_ref()
        .map(|iov| iov.eta_names.clone())
        .unwrap_or_default();
    coordinates(template)
        .into_iter()
        .map(|c| match c {
            Coord::Theta(i) => template.theta_names[i].clone(),
            Coord::Omega(i, j) => format!("OMEGA({},{})", eta[i], eta[j]),
            Coord::Sigma(i) => template.sigma.names[i].clone(),
            Coord::OmegaIov(i, j) => format!("OMEGA_IOV({},{})", kappa[i], kappa[j]),
        })
        .collect()
}

/// Flatten a fitted model into the vector [`parameter_names`] labels.
///
/// `template` supplies the structure — which Omega entries are free parameters
/// rather than structural zeros — because a [`FitResult`] carries the matrices
/// but not their masks.
pub fn flatten_estimates(template: &ModelParameters, result: &FitResult) -> Vec<f64> {
    coordinates(template)
        .into_iter()
        .map(|c| match c {
            Coord::Theta(i) => result.theta[i],
            Coord::Omega(i, j) => result.omega[(i, j)],
            Coord::Sigma(i) => result.sigma[i],
            // A fitted IOV Omega is always present when the model declares one;
            // fall back to the template so a caller cannot get a wrong number
            // in the unreachable case.
            Coord::OmegaIov(i, j) => result
                .omega_iov
                .as_ref()
                .map(|m| m[(i, j)])
                .or_else(|| template.omega_iov.as_ref().map(|m| m.matrix[(i, j)]))
                .unwrap_or(f64::NAN),
        })
        .collect()
}

/// The per-replicate standard errors, in the *same order as the names*.
///
/// Each one is looked up by its `(i, j)` coordinate rather than read off a
/// parallel vector, because the packed layouts differ: `FitResult::se_omega` is
/// the **column-major** lower triangle for a block Omega and carries structural
/// zeros, whereas the flat vector here is row-major and omits them. Zipping the
/// two would mislabel every block-Omega SE from the third element on, and shift
/// the sigma SEs as well. [`ferx_core::omega_se_at`] owns that indexing.
///
/// `None` for the whole vector when the covariance step did not run — which is
/// the default, and why the bootstrap standard error is the spread of the
/// estimates rather than an average of these. `None` for an individual entry
/// where core does not report an SE: `se_kappa` carries only the IOV diagonal
/// variances, so an IOV covariance has no reported SE at all.
fn flat_standard_errors(
    template: &ModelParameters,
    result: &FitResult,
) -> Option<Vec<Option<f64>>> {
    result.se_theta.as_ref()?;
    let n_eta = template.omega.dim();
    Some(
        coordinates(template)
            .into_iter()
            .map(|c| match c {
                Coord::Theta(i) => result.se_theta.as_ref().and_then(|v| v.get(i).copied()),
                Coord::Omega(i, j) => ferx_core::omega_se_at(&result.se_omega, n_eta, i, j),
                Coord::Sigma(i) => result.se_sigma.as_ref().and_then(|v| v.get(i).copied()),
                // Core reports one SE per IOV *variance*; off-diagonal IOV
                // covariances have none.
                Coord::OmegaIov(i, j) if i == j => {
                    result.se_kappa.as_ref().and_then(|v| v.get(i).copied())
                }
                Coord::OmegaIov(_, _) => None,
            })
            .collect(),
    )
}

/// Inverse of [`flatten_estimates`]: rebuild [`ModelParameters`] from a flat
/// vector, keeping every structural property of the template (bounds, fixed
/// flags, block structure, IOV mask).
///
/// This is what `--update-inits` and `--dofv` both need — and going through the
/// flat vector rather than the `FitResult` means both work equally well from a
/// stored `raw_results.csv`.
pub fn params_from_estimates(template: &ModelParameters, flat: &[f64]) -> ModelParameters {
    let mut params = template.clone();
    let mut omega = template.omega.matrix.clone();
    let mut omega_iov = template.omega_iov.as_ref().map(|m| m.matrix.clone());
    let mut sigma = template.sigma.values.clone();

    for (k, coord) in coordinates(template).into_iter().enumerate() {
        let Some(&value) = flat.get(k) else { break };
        match coord {
            Coord::Theta(i) => params.theta[i] = value,
            Coord::Omega(i, j) => {
                omega[(i, j)] = value;
                omega[(j, i)] = value;
            }
            Coord::Sigma(i) => sigma[i] = value,
            Coord::OmegaIov(i, j) => {
                if let Some(m) = omega_iov.as_mut() {
                    m[(i, j)] = value;
                    m[(j, i)] = value;
                }
            }
        }
    }

    params.omega = OmegaMatrix::from_matrix_with_mask(
        omega,
        template.omega.eta_names.clone(),
        template.omega.diagonal,
        template.omega.free_mask.clone(),
    );
    params.sigma = SigmaVector {
        values: sigma,
        names: template.sigma.names.clone(),
    };
    params.omega_iov = match (omega_iov, &template.omega_iov) {
        (Some(m), Some(t)) => Some(OmegaMatrix::from_matrix_with_mask(
            m,
            t.eta_names.clone(),
            t.diagonal,
            t.free_mask.clone(),
        )),
        _ => None,
    };
    params
}

// ── strata ──────────────────────────────────────────────────────────────────

/// Read the stratification column straight from the dataset, one label per
/// subject in `population` order.
///
/// Read from the CSV rather than from `Subject::covariates` for two reasons: the
/// stratification column is usually a study or group identifier that the model
/// never declares as a covariate (so it is not on the subject at all), and the
/// values are compared as *strings*, so a non-numeric group label works.
///
/// Enforces PsN's rule that the column take exactly one value per individual —
/// "the algorithm requires that an individual can unambiguously be categorized
/// according to the stratification variable" — naming the offending subject.
pub fn strata_from_csv(
    data_path: &str,
    column_map: &[(String, String)],
    population: &Population,
    stratify_on: &str,
) -> Result<Strata, String> {
    let id_header = column_map
        .iter()
        .find(|(role, _)| role.eq_ignore_ascii_case("id"))
        .map(|(_, header)| header.clone());

    let mut reader = csv::Reader::from_path(data_path)
        .map_err(|e| format!("--stratify-on: cannot read `{data_path}`: {e}"))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("--stratify-on: cannot read the header row of `{data_path}`: {e}"))?
        .clone();

    let find = |name: &str| {
        headers
            .iter()
            .position(|h| h.trim().eq_ignore_ascii_case(name))
    };
    let id_col = id_header
        .as_deref()
        .and_then(find)
        .or_else(|| find("ID"))
        .ok_or_else(|| format!("--stratify-on: no ID column found in `{data_path}`"))?;
    let strat_col = find(stratify_on).ok_or_else(|| {
        format!(
            "--stratify-on: column `{stratify_on}` is not in `{data_path}`. Columns present: {:?}",
            headers.iter().collect::<Vec<_>>()
        )
    })?;

    let mut by_id: HashMap<String, String> = HashMap::new();
    for record in reader.records() {
        let record =
            record.map_err(|e| format!("--stratify-on: malformed row in `{data_path}`: {e}"))?;
        let (Some(id), Some(value)) = (record.get(id_col), record.get(strat_col)) else {
            continue;
        };
        let (id, value) = (id.trim().to_string(), value.trim().to_string());
        match by_id.get(&id) {
            Some(seen) if seen != &value => {
                return Err(format!(
                    "--stratify-on: subject `{id}` has more than one value in column \
                     `{stratify_on}` (`{seen}` and `{value}`), so it cannot be unambiguously \
                     assigned to a stratum. Bootstrapping resamples whole individuals, so the \
                     stratification column must be constant within each ID."
                ));
            }
            Some(_) => {}
            None => {
                by_id.insert(id, value);
            }
        }
    }

    let mut labels = Vec::with_capacity(population.subjects.len());
    for subject in &population.subjects {
        let label = by_id.get(&subject.id).ok_or_else(|| {
            format!(
                "--stratify-on: subject `{}` has no row in `{data_path}` carrying column \
                 `{stratify_on}`",
                subject.id
            )
        })?;
        labels.push(label.clone());
    }
    Ok(Strata::from_labels(&labels, stratify_on))
}

// ── the run ─────────────────────────────────────────────────────────────────

/// Fit options for a replicate: quiet, single-threaded, no covariance step
/// unless asked, and no checkpoint file.
///
/// Each of those is load-bearing:
///
/// * **quiet** — 200 fits of per-iteration output is unreadable, and the
///   warnings are still captured on each `FitResult`;
/// * **`threads = 1`** — the bootstrap already parallelises over replicates, and
///   nesting Rayon pools oversubscribes the machine (the N×N problem this repo
///   hit in CI);
/// * **no checkpoint** — every replicate would otherwise write and resume the
///   *same* `{model}.tmp`, so they would restore each other's state;
/// * **no SIR** — a per-replicate uncertainty step inside an uncertainty
///   analysis is pure cost.
fn replicate_options(base: &FitOptions, keep_covariance: bool) -> FitOptions {
    let mut o = base.clone();
    o.verbose = false;
    o.threads = Some(1);
    o.run_covariance_step = keep_covariance;
    o.checkpoint = false;
    o.checkpoint_path = None;
    o.sir = false;
    o
}

fn diagnostics_from(result: &FitResult) -> (bool, bool, bool) {
    let near_boundary = result
        .warnings_structured
        .iter()
        .any(|w| w.category == WarningCode::BoundaryEstimate);
    let cov_failed = result
        .warnings_structured
        .iter()
        .any(|w| w.category == WarningCode::CovarianceFailed);
    let cov_warnings = result.warnings_structured.iter().any(|w| {
        matches!(
            w.category,
            WarningCode::CovarianceRegularized | WarningCode::CovarianceStep
        )
    });
    let cov_ok = result.covariance_matrix.is_some() && !cov_failed;
    (near_boundary, cov_ok, cov_warnings)
}

/// Refuse a mixture model. **Settled, not pending** — see #1145.
///
/// A mixture's classes are identified only up to relabelling, so replicate *k*'s
/// "class 1" need not be the original fit's class 1. Averaging estimates across
/// replicates without resolving that label switching inflates the standard error
/// toward the *between-class separation*, which is a property of the model and
/// not a sampling uncertainty — and it does so while every replicate converges
/// and the table looks entirely reasonable. The error grows with how well
/// separated the classes are, i.e. it is worst exactly when the mixture is best
/// identified and therefore most likely to be believed.
///
/// Relabelling rules exist (ordering on a canonical statistic; matching each
/// replicate's per-subject `mixest` assignments against the base fit's) and would
/// work most of the time. #1145 closed on the judgement that "most of the time"
/// is the wrong standard for the artefact a reader trusts *instead of* redoing
/// the analysis, and that each rule adds a knob whose setting silently changes
/// the answer.
///
/// The alternative offered in the message is [SIR], which never re-optimises —
/// it samples around the ML estimates and reweights by the likelihood — so there
/// is no second basin to fall into and no labelling to resolve. The covariance
/// step is likewise well defined at a single fit's fixed labelling.
///
/// Consequence for this module: `MixtureParams`' per-class Omega/Sigma overrides
/// are deliberately *not* represented in [`coordinates`]. There is no longer a
/// reason to add them.
///
/// [SIR]: https://ferx-nlme.github.io/ferx-core/estimation/sir.html
fn reject_mixture_model(params: &ModelParameters) -> Result<(), String> {
    if params.mixture.is_some() {
        return Err(
            "the bootstrap does not support mixture models ([mixture] block), and will not: \
             a mixture's classes are identified only up to relabelling, so a replicate's \
             class 1 need not be the original fit's class 1, and averaging estimates across \
             replicates mixes them — inflating the standard error toward the between-class \
             separation while every replicate converges and the table looks reasonable. \
             For parameter uncertainty on a mixture model use SIR (`sir = true`), which \
             reweights samples around the estimates instead of re-fitting and so cannot \
             switch labels."
                .to_string(),
        );
    }
    Ok(())
}

/// Refuse a model whose predictions depend on the subject identifier.
///
/// Resampling with replacement necessarily renames the duplicate copies of a
/// subject (see [`resample::build_population`]), so a model that reads `ID` as a
/// covariate would silently see different values than it did in the base fit.
/// PsN documents the same hazard as a known bug — "model code that relies on ID
/// numbers will lead to errors" — and leaves it to the user. Fail instead.
fn reject_id_dependent_model(model: &CompiledModel) -> Result<(), String> {
    reject_id_dependent(&model.referenced_covariates)
}

/// The predicate behind [`reject_id_dependent_model`], split out so it can be
/// unit-tested without constructing a whole [`CompiledModel`].
fn reject_id_dependent(referenced: &[String]) -> Result<(), String> {
    if referenced.iter().any(|c| c.eq_ignore_ascii_case("id")) {
        return Err(
            "this model uses `ID` as a covariate, which the bootstrap cannot support: \
             resampling with replacement draws the same subject more than once, and the \
             copies must enter the fit as independent individuals, so their IDs are \
             necessarily renamed. Re-express the covariate on a column that describes the \
             subject (study, arm, group) rather than identifies it."
                .to_string(),
        );
    }
    Ok(())
}

/// Run the bootstrap.
pub fn run_bootstrap(
    prepared: &PreparedRun,
    options: &BootstrapOptions,
) -> Result<BootstrapResult, String> {
    reject_id_dependent_model(&prepared.parsed.model)?;
    reject_mixture_model(&prepared.init_params)?;
    options.validate()?;
    options.validate_for_run()?;

    let template = &prepared.init_params;
    let names = parameter_names(template);
    let subject_ids: Vec<String> = prepared
        .population
        .subjects
        .iter()
        .map(|s| s.id.clone())
        .collect();

    let strata = match &options.stratify_on {
        None => Strata::unstratified(subject_ids.len()),
        Some(column) => strata_from_csv(
            &prepared.data_path,
            &prepared.parsed.column_map,
            &prepared.population,
            column,
        )?,
    };
    let allocation = strata.allocation(&options.sample_size)?;
    if allocation.iter().all(|(_, n)| *n == 0) {
        return Err("the resampling allocation draws 0 subjects".to_string());
    }

    // ── the draws ───────────────────────────────────────────────────────────
    // Computed before anything is fitted, because `--resume` needs the draw of
    // every replicate it *reuses* in order to rewrite the three draw files. It
    // can have them without a single fit having happened: a draw is a pure
    // function of `(seed, index)`.
    let draws: Vec<Replicate> = (1..=options.samples)
        .map(|i| resample::draw(&strata, &allocation, options.seed, i))
        .collect();

    // ── resume ──────────────────────────────────────────────────────────────
    let manifest = RunManifest::new(
        options,
        prepared.model_hash.clone(),
        prepared.data_hash.clone(),
        &names,
    );
    let (reused_original, reused) = if options.resume {
        let dir = options
            .directory
            .as_ref()
            .expect("validate_for_run rejects --resume without --directory");
        load_resumable(dir, &manifest, options)?
    } else {
        (None, Vec::new())
    };
    let n_reused = reused.len();
    let already_done: std::collections::HashSet<usize> = reused.iter().map(|r| r.index).collect();

    // ── the incremental journal ─────────────────────────────────────────────
    // Opened before the first fit so that a kill at any point from here on
    // leaves a directory the next `--resume` can read.
    let journal = match &options.directory {
        Some(dir) => {
            let j = journal::Journal::create(
                dir,
                &names,
                &subject_ids,
                reused_original.as_ref(),
                &reused,
                &draws,
                options,
            )?;
            manifest.write(&journal::manifest_path(dir))?;
            Some(j)
        }
        None => None,
    };

    // ── the base fit ────────────────────────────────────────────────────────
    let base_options = replicate_options(&prepared.parsed.fit_options, options.keep_covariance);
    let original = match reused_original {
        // A resumed run must **not** refit the base model. `--update-inits`
        // starts every replicate from its estimates, and a second fit can land
        // on a slightly different optimum — so refitting it would start the new
        // replicates from a different point than the ones already on disk, and
        // the resumed run would no longer equal the uninterrupted one.
        Some(base) => Some(base),
        None if options.run_base_model => {
            let started = Instant::now();
            let result = fit(
                &prepared.parsed.model,
                &prepared.population,
                template,
                // The base fit keeps the model file's own covariance setting: it
                // is the run whose `R⁻¹` standard errors the bootstrap is being
                // compared against.
                &{
                    let mut o = base_options.clone();
                    o.run_covariance_step = prepared.parsed.fit_options.run_covariance_step;
                    o
                },
            )?;
            let (near_boundary, cov_ok, cov_warn) = diagnostics_from(&result);
            let base = ReplicateResult {
                index: 0,
                estimates: flatten_estimates(template, &result),
                standard_errors: flat_standard_errors(template, &result),
                ofv: result.ofv,
                converged: result.converged,
                estimate_near_boundary: near_boundary,
                covariance_step_successful: cov_ok,
                covariance_step_warnings: cov_warn,
                seconds: started.elapsed().as_secs_f64(),
                error: None,
                delta_ofv: None,
            };
            if let Some(j) = &journal {
                j.append(&base, None);
            }
            Some(base)
        }
        None => None,
    };

    // `--update-inits` starts every replicate from the base fit's estimates —
    // PsN's default, and worth a lot here: the replicates differ from the
    // original only by resampling, so its optimum is the best available start.
    let replicate_init = match (&original, options.update_inits) {
        (Some(base), true) => params_from_estimates(template, &base.estimates),
        _ => template.clone(),
    };

    // ── the replicates ──────────────────────────────────────────────────────
    let run_one = |replicate: &Replicate| -> ReplicateResult {
        let population = resample::build_population(&prepared.population, replicate);
        let started = Instant::now();
        let outcome = fit(
            &prepared.parsed.model,
            &population,
            &replicate_init,
            &base_options,
        );
        let seconds = started.elapsed().as_secs_f64();
        let result = match outcome {
            Ok(result) => {
                let (near_boundary, cov_ok, cov_warn) = diagnostics_from(&result);
                ReplicateResult {
                    index: replicate.index,
                    estimates: flatten_estimates(template, &result),
                    standard_errors: flat_standard_errors(template, &result),
                    ofv: result.ofv,
                    converged: result.converged,
                    estimate_near_boundary: near_boundary,
                    covariance_step_successful: cov_ok,
                    covariance_step_warnings: cov_warn,
                    seconds,
                    error: None,
                    delta_ofv: None,
                }
            }
            Err(e) => ReplicateResult {
                index: replicate.index,
                estimates: Vec::new(),
                standard_errors: None,
                ofv: f64::NAN,
                converged: false,
                estimate_near_boundary: false,
                covariance_step_successful: false,
                covariance_step_warnings: false,
                seconds,
                error: Some(e),
                delta_ofv: None,
            },
        };
        if let Some(j) = &journal {
            j.append(&result, Some(replicate));
        }
        result
    };

    let todo: Vec<Replicate> = draws
        .iter()
        .filter(|d| !already_done.contains(&d.index))
        .cloned()
        .collect();

    let mut replicates = match options.threads {
        Some(n) if n > 1 => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| format!("failed to build the bootstrap thread pool: {e}"))?
            .install(|| todo.par_iter().map(run_one).collect::<Vec<_>>()),
        Some(_) => todo.iter().map(run_one).collect(),
        None => todo.par_iter().map(run_one).collect(),
    };

    // The journal's handles must be closed before `write_all` truncates the
    // same paths below, and this is where a write failure inside the parallel
    // loop surfaces.
    if let Some(j) = journal {
        j.into_result()?;
    }

    replicates.extend(reused);
    replicates.sort_by_key(|r| r.index);

    // ── Δofv ────────────────────────────────────────────────────────────────
    if options.dofv {
        let base = original.as_ref().ok_or(
            "--dofv needs the original fit's OFV as its reference; it cannot be combined with \
             --no-run-base-model",
        )?;
        compute_delta_ofv(prepared, template, base.ofv, &base_options, &mut replicates)?;
    }

    let n_estimated_parameters = count_estimated(template);
    let summary = summary::summarize(&names, original.as_ref(), &replicates, options);

    let result = BootstrapResult {
        parameter_names: names,
        original,
        replicates,
        draws,
        subject_ids,
        summary,
        n_estimated_parameters,
        n_reused,
    };

    // Rewrites what the journal appended, this time in index order. A completed
    // run's artefacts are therefore byte-identical however its replicates were
    // scheduled — and identical to an uninterrupted run's, which is what makes
    // `--resume` assertable rather than merely plausible.
    if let Some(dir) = &options.directory {
        output::write_all(dir, &result, options)?;
    }
    Ok(result)
}

/// Read an interrupted run's completed replicates back out of its directory.
///
/// Returns `(original, replicates)` — what may be reused rather than refitted.
/// Anything dropped here is simply refitted, so the read errs towards dropping:
/// the only unsafe outcome is *keeping* a row that does not belong to this run,
/// which is what [`RunManifest::check_compatible`] rules out.
fn load_resumable(
    dir: &std::path::Path,
    manifest: &RunManifest,
    options: &BootstrapOptions,
) -> Result<(Option<ReplicateResult>, Vec<ReplicateResult>), String> {
    let raw = journal::raw_results_path(dir);
    if !raw.exists() {
        return Err(format!(
            "--resume continues an interrupted run, but `{}` has no raw_results.csv. Drop \
             --resume to start the run from sample 1.",
            raw.display()
        ));
    }
    let on_disk = RunManifest::read(&journal::manifest_path(dir))?;
    manifest.check_compatible(&on_disk, dir)?;

    let (file_names, original, replicates) = output::read_raw_results(&raw)?;
    if file_names != manifest.parameter_names {
        return Err(format!(
            "--resume: the columns of `{}` are {:?}, but this model's parameters are {:?}. The \
             directory belongs to a different run.",
            raw.display(),
            file_names,
            manifest.parameter_names
        ));
    }

    // A recorded failure is kept by default — PsN's answer, and the right one
    // when the failure is deterministic, which it usually is. `--retry-failed`
    // is for the case where it was not: an out-of-memory kill, a full disk.
    let keep = |r: &ReplicateResult| !(options.retry_failed && r.error.is_some());

    let mut seen = std::collections::HashSet::new();
    let mut kept = Vec::new();
    for r in replicates {
        // Out-of-range indices cannot happen while `--samples` is pinned by the
        // manifest, but a hand-edited file must not inject a replicate that no
        // draw corresponds to.
        if r.index == 0 || r.index > options.samples || !keep(&r) {
            continue;
        }
        if seen.insert(r.index) {
            kept.push(r);
        }
    }
    kept.sort_by_key(|r| r.index);

    // With `--no-run-base-model` the stored base fit is not part of this run,
    // so it is not carried forward — and must not be, or it would reappear as a
    // `sample = 0` row the user asked not to have.
    let original = original
        .filter(|_| options.run_base_model)
        .filter(|r| keep(r));
    Ok((original, kept))
}

/// Recompute the summary of a finished run from its `raw_results.csv`, under a
/// different set of exclusion criteria — PsN's `-summarize`.
///
/// Refits nothing. This is what the "too many samples were filtered out" case in
/// the PsN guide's recovery section asks for: the estimates are already on disk,
/// and so are the diagnostics the filters read, so changing
/// `--no-skip-minimization-terminated` (say) is a re-read, not a re-run.
///
/// Rewrites `bootstrap_results.csv` and `bootstrap_diagnostics.csv` in place and
/// returns the new summary.
pub fn resummarize(
    directory: &std::path::Path,
    options: &BootstrapOptions,
) -> Result<BootstrapSummary, String> {
    options.validate()?;
    let raw = directory.join("raw_results.csv");
    if !raw.exists() {
        return Err(format!(
            "--summarize needs an existing bootstrap run directory containing \
             `raw_results.csv`; `{}` has none",
            directory.display()
        ));
    }
    let (names, original, replicates) = output::read_raw_results(&raw)?;
    let replicate_count = replicates.len();
    let summary = summary::summarize(&names, original.as_ref(), &replicates, options);
    let result = BootstrapResult {
        parameter_names: names,
        original,
        replicates,
        // The draws are not recoverable from `raw_results.csv`, and are not
        // needed: `--summarize` rewrites only the two statistics files, and the
        // per-replicate `included_*` / `sample_keys` files the original run
        // wrote are still valid — the draws did not change, only which of them
        // count.
        draws: Vec::new(),
        subject_ids: Vec::new(),
        summary: summary.clone(),
        // Carried forward from the run that wrote the directory: it is a
        // property of the model, not of the exclusion criteria being changed.
        n_estimated_parameters: output::read_diagnostic(
            &directory.join("bootstrap_diagnostics.csv"),
            "chi_square_df",
        )
        .unwrap_or(0.0) as usize,
        // `--summarize` refits nothing, so every replicate it reports is one an
        // earlier run produced.
        n_reused: replicate_count,
    };
    output::write_bootstrap_results(&directory.join("bootstrap_results.csv"), &result)?;
    output::write_diagnostics(&directory.join("bootstrap_diagnostics.csv"), &result)?;
    Ok(summary)
}

/// Evaluate every replicate's parameter vector on the **original** dataset with
/// no estimation, and record `ofv − ofv_original`.
///
/// `outer_maxiter = 0` is already exactly NONMEM's `MAXEVAL=0` in ferx's outer
/// optimizer, so this is an evaluation, not a one-step fit. The resulting
/// distribution should sit at or below a chi-square with
/// [`BootstrapResult::n_estimated_parameters`] degrees of freedom; if it does
/// not, the PsN guide's advice applies — prefer another uncertainty method such
/// as SIR.
fn compute_delta_ofv(
    prepared: &PreparedRun,
    template: &ModelParameters,
    ofv_original: f64,
    base_options: &FitOptions,
    replicates: &mut [ReplicateResult],
) -> Result<(), String> {
    let mut eval = base_options.clone();
    eval.outer_maxiter = 0;
    eval.run_covariance_step = false;

    let deltas: Vec<(usize, Option<f64>)> = replicates
        .par_iter()
        .map(|r| {
            if r.error.is_some() || r.estimates.is_empty() {
                return (r.index, None);
            }
            let params = params_from_estimates(template, &r.estimates);
            let ofv = fit(&prepared.parsed.model, &prepared.population, &params, &eval)
                .map(|f| f.ofv)
                .ok();
            (r.index, ofv.map(|o| o - ofv_original))
        })
        .collect();

    let by_index: HashMap<usize, Option<f64>> = deltas.into_iter().collect();
    for r in replicates.iter_mut() {
        r.delta_ofv = by_index.get(&r.index).copied().flatten();
    }
    Ok(())
}

/// Non-fixed parameters — the chi-square degrees of freedom for the Δofv
/// reference distribution.
fn count_estimated(template: &ModelParameters) -> usize {
    // Counted over the flat vector's own coordinates, not over the `*_fixed`
    // flag vectors. Those are per-eta, so counting them misses a block Omega's
    // off-diagonals entirely: a free 2x2 block is three estimated parameters and
    // two flags. Getting this wrong writes a chi-square reference with too few
    // degrees of freedom, which makes the Δofv distribution look better than it
    // is — the one direction a diagnostic must never fail in.
    coordinates(template)
        .into_iter()
        .filter(|c| !is_fixed(template, *c))
        .count()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
