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

pub mod output;
pub mod resample;
pub mod summary;

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
    /// Where the CSV artefacts are written. `None` writes nothing.
    pub directory: Option<PathBuf>,
    /// Two-sided confidence level, in percent.
    pub confidence_level: f64,
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
    /// Per-replicate standard errors, present only with `keep_covariance`.
    pub standard_errors: Option<Vec<f64>>,
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
}

// ── flat parameter vector ───────────────────────────────────────────────────

/// Names for the flat parameter vector: every theta, then the free lower
/// triangle of Omega, then every sigma.
///
/// Omega entries are named by their etas — `OMEGA(CL,CL)` — rather than by
/// PsN's positional `OMEGA(1,1)`. The file is ferx's own, and a positional index
/// silently means something different the moment an eta is added.
pub fn parameter_names(template: &ModelParameters) -> Vec<String> {
    let mut names = template.theta_names.clone();
    let eta = &template.omega.eta_names;
    for i in 0..template.omega.dim() {
        for j in 0..=i {
            if template.omega.free_mask[(i, j)] {
                names.push(format!("OMEGA({},{})", eta[i], eta[j]));
            }
        }
    }
    names.extend(template.sigma.names.iter().cloned());
    names
}

/// Flatten a fitted model into the vector [`parameter_names`] labels.
///
/// `template` supplies the structure — which Omega entries are free parameters
/// rather than structural zeros — because a [`FitResult`] carries the matrix but
/// not the mask.
pub fn flatten_estimates(template: &ModelParameters, result: &FitResult) -> Vec<f64> {
    let mut v = result.theta.clone();
    for i in 0..template.omega.dim() {
        for j in 0..=i {
            if template.omega.free_mask[(i, j)] {
                v.push(result.omega[(i, j)]);
            }
        }
    }
    v.extend(result.sigma.iter().copied());
    v
}

/// Inverse of [`flatten_estimates`]: rebuild [`ModelParameters`] from a flat
/// vector, keeping every structural property of the template (bounds, fixed
/// flags, block structure, IOV Omega, mixture).
///
/// This is what `--update-inits` and `--dofv` both need — and going through the
/// flat vector rather than the `FitResult` means both work equally well from a
/// stored `raw_results.csv`.
pub fn params_from_estimates(template: &ModelParameters, flat: &[f64]) -> ModelParameters {
    let n_theta = template.theta.len();
    let mut params = template.clone();
    params.theta = flat[..n_theta].to_vec();

    let dim = template.omega.dim();
    let mut m = template.omega.matrix.clone();
    let mut k = n_theta;
    for i in 0..dim {
        for j in 0..=i {
            if template.omega.free_mask[(i, j)] {
                m[(i, j)] = flat[k];
                m[(j, i)] = flat[k];
                k += 1;
            }
        }
    }
    params.omega = OmegaMatrix::from_matrix_with_mask(
        m,
        template.omega.eta_names.clone(),
        template.omega.diagonal,
        template.omega.free_mask.clone(),
    );
    params.sigma = SigmaVector {
        values: flat[k..].to_vec(),
        names: template.sigma.names.clone(),
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

/// The per-replicate standard errors, flattened in [`parameter_names`] order.
///
/// `None` when the covariance step did not run — which is the default, and why
/// the bootstrap standard error is the spread of the estimates rather than an
/// average of these.
fn flat_standard_errors(result: &FitResult) -> Option<Vec<f64>> {
    let theta = result.se_theta.as_ref()?;
    let mut se = theta.clone();
    se.extend(result.se_omega.clone().unwrap_or_default());
    se.extend(result.se_sigma.clone().unwrap_or_default());
    Some(se)
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
    if options.samples == 0 {
        return Err("--samples must be at least 1".to_string());
    }
    if !(0.0..100.0).contains(&options.confidence_level) || options.confidence_level <= 0.0 {
        return Err("--ci must be a confidence level in (0, 100)".to_string());
    }
    // Both covariance-step filters read a diagnostic that only exists when the
    // step actually ran. Silently ignoring them would drop a filter the user
    // asked for without saying so.
    if !options.keep_covariance
        && (options.skip_covariance_step_terminated || options.skip_with_covstep_warnings)
    {
        return Err(
            "--skip-covariance-step-terminated / --skip-with-covstep-warnings filter on the \
             covariance step, which is off for replicate fits by default. Add \
             --keep-covariance (slower) or drop the filter."
                .to_string(),
        );
    }

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

    // ── the base fit ────────────────────────────────────────────────────────
    let base_options = replicate_options(&prepared.parsed.fit_options, options.keep_covariance);
    let original = if options.run_base_model {
        let started = Instant::now();
        let result = fit(
            &prepared.parsed.model,
            &prepared.population,
            template,
            // The base fit keeps the model file's own covariance setting: it is
            // the run whose `R⁻¹` standard errors the bootstrap is being
            // compared against.
            &{
                let mut o = base_options.clone();
                o.run_covariance_step = prepared.parsed.fit_options.run_covariance_step;
                o
            },
        )?;
        let (near_boundary, cov_ok, cov_warn) = diagnostics_from(&result);
        Some(ReplicateResult {
            index: 0,
            estimates: flatten_estimates(template, &result),
            standard_errors: flat_standard_errors(&result),
            ofv: result.ofv,
            converged: result.converged,
            estimate_near_boundary: near_boundary,
            covariance_step_successful: cov_ok,
            covariance_step_warnings: cov_warn,
            seconds: started.elapsed().as_secs_f64(),
            error: None,
            delta_ofv: None,
        })
    } else {
        None
    };

    // `--update-inits` starts every replicate from the base fit's estimates —
    // PsN's default, and worth a lot here: the replicates differ from the
    // original only by resampling, so its optimum is the best available start.
    let replicate_init = match (&original, options.update_inits) {
        (Some(base), true) => params_from_estimates(template, &base.estimates),
        _ => template.clone(),
    };

    // ── the replicates ──────────────────────────────────────────────────────
    let draws: Vec<Replicate> = (1..=options.samples)
        .map(|i| resample::draw(&strata, &allocation, options.seed, i))
        .collect();

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
        match outcome {
            Ok(result) => {
                let (near_boundary, cov_ok, cov_warn) = diagnostics_from(&result);
                ReplicateResult {
                    index: replicate.index,
                    estimates: flatten_estimates(template, &result),
                    standard_errors: flat_standard_errors(&result),
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
        }
    };

    let mut replicates = match options.threads {
        Some(n) if n > 1 => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| format!("failed to build the bootstrap thread pool: {e}"))?
            .install(|| draws.par_iter().map(run_one).collect::<Vec<_>>()),
        Some(_) => draws.iter().map(run_one).collect(),
        None => draws.par_iter().map(run_one).collect(),
    };
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
    };

    if let Some(dir) = &options.directory {
        output::write_all(dir, &result, options)?;
    }
    Ok(result)
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
    let raw = directory.join("raw_results.csv");
    if !raw.exists() {
        return Err(format!(
            "--summarize needs an existing bootstrap run directory containing \
             `raw_results.csv`; `{}` has none",
            directory.display()
        ));
    }
    let (names, original, replicates) = output::read_raw_results(&raw)?;
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
    let theta = template.theta_fixed.iter().filter(|f| !**f).count();
    let omega = template.omega_fixed.iter().filter(|f| !**f).count();
    let sigma = template.sigma_fixed.iter().filter(|f| !**f).count();
    theta + omega + sigma
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
