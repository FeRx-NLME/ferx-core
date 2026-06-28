use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::theta_packs_log;
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv, read_nonmem_csv_filtered, read_nonmem_csv_filtered_tte,
    read_nonmem_csv_with_covariates, read_nonmem_csv_with_covariates_filtered,
    read_nonmem_csv_with_covariates_tte, SelectionFilter, ERR_COV_MISSING_COLUMNS,
    ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{compute_iwres_with_correlations, iwres_autocorrelation};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Build the `FitResult.neural_networks` summary from the compiled model's
/// `[covariate_nn]` blocks. Empty when no NN blocks are present, so output
/// writers can always iterate `result.neural_networks` without branching.
#[cfg(feature = "nn")]
fn build_neural_network_infos(model: &CompiledModel) -> Vec<NeuralNetworkInfo> {
    use crate::nn::CovariateMapper;
    model
        .covariate_nns
        .iter()
        .map(|nn| NeuralNetworkInfo {
            name: nn.name.clone(),
            shape: nn.mapper.mlp().layer_sizes().to_vec(),
            hidden_activation: nn.mapper.mlp().hidden_activation().as_str().to_string(),
            output_activation: nn.mapper.mlp().output_activation().as_str().to_string(),
            n_weights: nn.mapper.n_weights(),
            weights_offset: nn.weights_offset,
            input_names: nn.mapper.input_names().to_vec(),
            output_names: nn.mapper.output_names().to_vec(),
        })
        .collect()
}
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Route predictions through analytical PK or ODE solver, then apply
/// `model.scaling` so simulate / predict / post-fit IPRED see the same
/// scaled output as the estimation dispatcher in `pk::compute_predictions_with_tv_into_with_schedule`.
///
/// `theta` and `eta` are required so that `ScalingSpec::ExpressionScale`
/// can evaluate its `scale_fn(theta, eta, covariates)`. Callers that don't
/// have a separate eta vector (population predictions) pass an all-zero eta.
///
/// Production code routes through [`pk::compute_predictions_with_tv`] (the
/// TV-covariate-aware dispatcher) instead; this baseline-only helper now only
/// backs the TV-vs-no-TV gap assertions in the regression tests.
#[cfg(test)]
pub(crate) fn model_preds(
    model: &CompiledModel,
    subject: &Subject,
    pk_params: &PkParams,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    let mut preds = if let Some(ref ode_spec) = model.ode_spec {
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values, theta, eta)
    } else {
        // Resolve any modeled-`RATE` doses (#324/#394, e.g. `RATE=-2` → `D{cmt}`)
        // to a concrete duration/rate before the analytical closed form — mirrors
        // the ODE `resolve_subject_doses` step inside `compute_predictions_ode`.
        // Borrowed (no allocation) for the all-`Fixed` common case.
        let resolved = crate::ode::resolve_subject_doses(
            subject,
            model.active_dose_attr_map(),
            &pk_params.values,
        );
        pk::compute_predictions(model.pk_model, &resolved, pk_params)
    };
    pk::apply_scaling(model, subject, theta, eta, &mut preds);
    pk::apply_log_transform(model, &mut preds);
    preds
}

/// Log-transform every observation (including M3 LLOQ values carried on CENS
/// rows — they live in the same `observations` vector) in place, for LTBS case 2
/// (`log(DV) ~ additive`, natural-scale data). Returns the count of non-positive
/// DV values, which are floored to [`crate::pk::LTBS_FLOOR`] before the log so the
/// result stays finite. Case 1 (`DV ~ log_additive`, `dv_pre_logged`) must NOT
/// call this — the DV is already on the log scale.
fn log_transform_observations(pop: &mut Population) -> usize {
    let mut n_nonpos = 0usize;
    for subject in &mut pop.subjects {
        for v in &mut subject.observations {
            if *v <= 0.0 {
                n_nonpos += 1;
            }
            *v = v.max(crate::pk::LTBS_FLOOR).ln();
        }
    }
    n_nonpos
}

/// Run a model file with a NONMEM-format CSV dataset.
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_with_data(
    model_path: &str,
    data_path: &str,
) -> Result<(FitResult, Population), String> {
    run_model_with_data_inits(model_path, data_path, None)
}

/// Like [`run_model_with_data`], but lets the caller (e.g. the CLI's
/// `--inits-from-nca` flag) override the model file's `inits_from_nca` fit
/// option. When `inits_override` is `None` the model-file value is used as-is;
/// when `Some(method)` it forces that NCA strategy regardless of the file.
pub fn run_model_with_data_inits(
    model_path: &str,
    data_path: &str,
    inits_override: Option<crate::suggest_start::NcaInit>,
) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    set_model_name(&mut parsed.model, model_path);
    if let Some(method) = inits_override {
        parsed.fit_options.inits_from_nca = Some(method);
    }

    eprintln!("Model: {}", parsed.model.name);

    let iov_col = parsed.fit_options.iov_column.as_deref();
    let sel_filter = build_selection_filter(&parsed.fit_options)?;
    let (population, covariate_table) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data_path,
        None,
        iov_col,
        sel_filter.as_ref(),
    )?;
    eprintln!(
        "Data:  {} subjects, {} observations from {}",
        population.subjects.len(),
        population.n_obs(),
        data_path
    );

    let init_params = build_init_params(&parsed);
    // Sync the resolved gradient method from fit_options onto the model so
    // `resolve_gradient_method` (which reads `model.gradient_method`) honours
    // the file's `gradient = ...` key. Mirrors `fit_from_files` (SDE forces FD).
    parsed.model.gradient_method = if parsed.model.is_sde()
        && parsed.fit_options.gradient_method != crate::types::GradientMethod::Fd
    {
        crate::types::GradientMethod::Fd
    } else {
        parsed.fit_options.gradient_method
    };
    let mut result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    result.covariate_table = covariate_table;
    // Hash both inputs *after* the fit so we don't double up disk reads
    // (the model and CSV are already in the page cache from parse + read
    // upstream). Errors here are non-fatal: the fit already succeeded, and
    // a missing hash just disables the integrity check in run_sir.
    result.model_path = Some(model_path.to_string());
    result.data_path = Some(data_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    result.data_hash = crate::io::hash::sha256_file(Path::new(data_path)).ok();
    result.model_text = std::fs::read_to_string(model_path).ok();
    Ok((result, population))
}

/// Run a model file with simulated data (from [simulation] block).
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_simulate(model_path: &str) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;
    use std::collections::HashMap;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    let sim_spec = parsed
        .simulation
        .clone()
        .ok_or("Model file has no [simulation] block — use --data instead")?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    // TTE endpoints (survival only): a synthetic subject must carry one
    // right-censored template row per cause CMT — otherwise `--simulate` emits
    // zero TTE rows (the synthetic `obs_records` are empty). The administrative
    // `[simulation] horizon` is the censoring time for those rows; the per-subject
    // draw then overwrites each template with its realised outcome (#522). Compute
    // the cause-CMT list once, outside the per-subject loop.
    #[cfg(feature = "survival")]
    let tte_cmts: Vec<usize> = parsed.model.tte_cmts();
    #[cfg(feature = "survival")]
    if !tte_cmts.is_empty() && sim_spec.horizon.is_none() {
        return Err("[simulation] with a TTE endpoint requires `horizon = <t>` \
             (the administrative censoring time at which event-free subjects are \
             right-censored)"
            .to_string());
    }

    // A synthetic design must have something to observe. The parser's
    // `times`-or-`horizon` rule is deliberately permissive (a pure-TTE model has
    // no continuous `times`), so enforce the model-specific requirement here, once
    // the endpoints are known (#522 review):
    //   - a model with a residual-error (continuous) endpoint needs `times` — this
    //     keeps the pre-#522 contract and closes the gap where a Gaussian, or a
    //     joint PK+TTE, model given only `horizon` would silently build
    //     zero-`observation` subjects and fit on empty continuous data;
    //   - a pure-TTE model (no error model, hence no sigma) instead needs a
    //     `horizon`, already enforced above.
    // A declared `[error_model]` is the signal for "produces continuous
    // observations": it is the only thing that allocates sigma, and every model
    // otherwise carries a default `pk_model`, so the structural model alone can't
    // distinguish intent.
    let model_has_continuous = !parsed.model.default_params.sigma.values.is_empty();
    let model_has_tte = parsed.model.has_tte();
    if sim_spec.obs_times.is_empty() {
        if model_has_continuous {
            return Err(
                "[simulation] has no `times`, but the model has a continuous \
                 (residual-error) endpoint that needs observation times — add \
                 `times = [...]` (a joint PK + TTE design needs both `times` and a \
                 `horizon`)"
                    .to_string(),
            );
        }
        if !model_has_tte {
            return Err(
                "[simulation] has no `times` and the model has no TTE endpoint \
                 to observe at a `horizon` — nothing to simulate"
                    .to_string(),
            );
        }
    }

    // Build template population
    let subjects: Vec<Subject> = (1..=sim_spec.n_subjects)
        .map(|i| Subject {
            id: format!("{}", i),
            doses: vec![DoseEvent::new(
                0.0,
                sim_spec.dose_amt,
                sim_spec.dose_cmt,
                0.0,
                false,
                0.0,
            )],
            obs_times: sim_spec.obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; sim_spec.obs_times.len()],
            obs_cmts: vec![1; sim_spec.obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; sim_spec.obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            // One right-censored template row per cause CMT, at the administrative
            // horizon (overwritten by the draw). Empty when the model has no TTE
            // endpoint, reproducing the previous all-Gaussian behaviour. The
            // `expect` cannot fire: a non-empty `tte_cmts` guarantees `horizon` is
            // `Some` (the TTE-requires-horizon check above returns early). (#522)
            #[cfg(feature = "survival")]
            obs_records: tte_cmts
                .iter()
                .map(|&cmt| crate::types::ObsRecord::Event {
                    time: sim_spec
                        .horizon
                        .expect("TTE endpoints require a horizon (checked above)"),
                    event_type: crate::types::EventType::RightCensored,
                    entry_time: 0.0,
                    cmt,
                })
                .collect(),
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // Simulate
    eprintln!(
        "Simulating {} subjects (seed={})...",
        sim_spec.n_subjects, sim_spec.seed
    );
    // Pass the administrative `horizon` through so TTE causes censor at the
    // planned study end (#522). `match_method = None` makes this identical to
    // `simulate_with_seed` for the Gaussian path (same seeded RNG, no matching),
    // and the `None`-matching branch cannot error.
    let sim_results = simulate_with_options(
        &parsed.model,
        &template,
        &parsed.model.default_params,
        1,
        &SimulateOptions {
            seed: Some(sim_spec.seed),
            match_method: None,
            horizon: sim_spec.horizon,
        },
    )
    .map_err(|e| format!("simulation failed: {e}"))?;

    // Group the simulation results by subject id once (they are emitted in
    // subject order, but a map keeps the write-back independent of that), then
    // write each subject's outcomes back in a single pass instead of re-scanning
    // all of `sim_results` per subject twice (Gaussian, then TTE).
    let mut sims_by_id: HashMap<&str, Vec<&SimulationResult>> = HashMap::new();
    for s in &sim_results {
        sims_by_id.entry(s.id.as_str()).or_default().push(s);
    }

    let mut population = template;
    for subject in &mut population.subjects {
        let Some(sims) = sims_by_id.get(subject.id.as_str()) else {
            continue;
        };

        // Gaussian write-back: only continuous outcomes map onto `observations`.
        // A TTE `Event` row would trip `continuous_value()`'s debug-assert, so
        // skip it here (the TTE outcomes go into `obs_records` below). Enumerating
        // *after* the filter makes `j` index the continuous subsequence, matching
        // the per-subject observation grid.
        for (j, s) in sims
            .iter()
            .filter(|s| matches!(s.outcome, crate::types::SimOutcome::Continuous { .. }))
            .enumerate()
        {
            if j < subject.observations.len() {
                // Under LTBS the simulated DV is on the log scale and may be
                // negative, so the positivity floor only applies to natural-scale
                // simulation.
                let v = s.outcome.continuous_value();
                subject.observations[j] = if parsed.model.log_transform {
                    v
                } else {
                    v.max(0.001)
                };
            }
        }

        // TTE write-back (#522): replace the subject's template rows with the
        // realised simulated outcomes — `Exact` at the drawn event time, or
        // `RightCensored` at the horizon when no cause fired — so the fitted
        // dataset (and any output) carries the simulated events rather than the
        // placeholders.
        #[cfg(feature = "survival")]
        {
            let events: Vec<crate::types::ObsRecord> = sims
                .iter()
                .filter_map(|s| match s.outcome {
                    crate::types::SimOutcome::Event { time, observed } => {
                        Some(crate::types::ObsRecord::Event {
                            time,
                            event_type: if observed {
                                crate::types::EventType::Exact
                            } else {
                                crate::types::EventType::RightCensored
                            },
                            // Synthetic `[simulation]` subjects always enter at 0
                            // (no left truncation), matching the template row
                            // above; keep the two in sync if this path ever gains
                            // delayed entry.
                            entry_time: 0.0,
                            cmt: s.cmt,
                        })
                    }
                    _ => None,
                })
                .collect();
            if !events.is_empty() {
                subject.obs_records = events;
            }
        }
    }

    // `n_obs()` counts only Gaussian observations; add the simulated TTE event
    // rows so a TTE-only `--simulate` run doesn't misleadingly report "0
    // observations" (#522 review).
    #[cfg(feature = "survival")]
    let n_records = population.n_obs()
        + population
            .subjects
            .iter()
            .map(|s| s.obs_records.len())
            .sum::<usize>();
    #[cfg(not(feature = "survival"))]
    let n_records = population.n_obs();
    eprintln!(
        "Loaded {} subjects, {} observations",
        population.subjects.len(),
        n_records
    );

    let init_params = build_init_params(&parsed);
    let mut result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    // No data file to hash — data is simulated in-process. Hash the model
    // post-fit (same pattern as `run_model_with_data`); failures are
    // non-fatal and just disable the integrity check in `run_sir`.
    result.model_path = Some(model_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    result.model_text = std::fs::read_to_string(model_path).ok();
    Ok((result, population))
}

/// Legacy alias
pub fn run_from_file(path: &str) -> Result<FitResult, String> {
    run_model_simulate(path).map(|(r, _)| r)
}

fn set_model_name(model: &mut CompiledModel, path: &str) {
    if model.name == "Unnamed" {
        if let Some(stem) = Path::new(path).file_stem().and_then(|s| s.to_str()) {
            model.name = stem.to_string();
        }
    }
}

fn build_init_params(parsed: &ParsedModel) -> ModelParameters {
    parsed.model.default_params.clone()
}

/// Fail early if the model references covariates that the data doesn't carry.
/// Case-sensitive: `CRCL` and `crcl` are distinct names. Historically a missing
/// covariate silently evaluated to zero, which left fits stuck at the initial
/// estimates with no visible diagnostic (see commit introducing this check).
///
/// Returns a diagnostic per problem (here, at most one). The message text is
/// kept byte-for-byte identical to the historical `Err(String)` so `fit()`'s
/// error — produced via [`first_error`] — is unchanged.
fn check_covariates(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();

    if missing.is_empty() {
        return Vec::new();
    }

    let available = if population.covariate_names.is_empty() {
        "(none)".to_string()
    } else {
        population.covariate_names.join(", ")
    };
    vec![Diagnostic::error(
        "E_MISSING_COVARIATE",
        format!(
            "Model references covariate(s) not found in data (case-sensitive): {}. \
             Available covariate columns: {}.",
            missing.join(", "),
            available
        ),
    )
    .with_suggestion(format!("available covariate columns: {}", available))]
}

/// Covariates referenced by the model but missing from the `[covariates]`
/// declaration. These are still read (leniently) so the model works; the parser
/// has already warned that they ought to be declared.
fn undeclared_referenced(model: &CompiledModel, decls: &[CovariateDecl]) -> Vec<String> {
    model
        .referenced_covariates
        .iter()
        .filter(|c| !decls.iter().any(|d| &d.name == *c))
        .cloned()
        .collect()
}

/// Single covariate-aware reader used by every file-based entry point (`fit`
/// wrappers and `ferx check`), so they all apply identical covariate validation.
/// Build a `SelectionFilter` from a model file's `FitOptions` alone.
/// Returns `None` when no selection rules are set.
fn build_selection_filter(opts: &FitOptions) -> Result<Option<SelectionFilter>, String> {
    if opts.ignore_exprs.is_empty()
        && opts.accept_exprs.is_empty()
        && opts.ignore_subjects.is_empty()
    {
        return Ok(None);
    }
    SelectionFilter::from_opts(
        &opts.ignore_exprs,
        &opts.accept_exprs,
        &opts.ignore_subjects,
    )
    .map(Some)
}

/// Build a `SelectionFilter` merging the model file's rules with a caller-supplied
/// `FitOptions` (e.g. from the R wrapper). Conditions from both sources are
/// deduplicated and OR'd (ignore) / AND'd (accept) together.
fn build_selection_filter_merged(
    model_opts: &FitOptions,
    call_opts: &FitOptions,
) -> Result<Option<SelectionFilter>, String> {
    // Merge by accumulating unique strings from both sources.
    let mut ignore = model_opts.ignore_exprs.clone();
    let mut accept = model_opts.accept_exprs.clone();
    let mut subjects = model_opts.ignore_subjects.clone();
    for s in &call_opts.ignore_exprs {
        let t = s.trim().to_string();
        if !ignore.iter().any(|e| e == &t) {
            ignore.push(t);
        }
    }
    for s in &call_opts.accept_exprs {
        let t = s.trim().to_string();
        if !accept.iter().any(|e| e == &t) {
            accept.push(t);
        }
    }
    for s in &call_opts.ignore_subjects {
        // Strip surrounding quotes so a caller-supplied `"3"` matches the same
        // subject as a `.ferx` `ignore_subjects = 3` (the model-file parser
        // already quote-strips). Without this the two sources disagree and a
        // duplicate across them fails to dedup.
        let t = s
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .trim()
            .to_string();
        if !t.is_empty() && !subjects.iter().any(|e| e == &t) {
            subjects.push(t);
        }
    }
    if ignore.is_empty() && accept.is_empty() && subjects.is_empty() {
        return Ok(None);
    }
    SelectionFilter::from_opts(&ignore, &accept, &subjects).map(Some)
}

/// Read a [`Population`] from `data_path` using the correct reader for `model`.
///
/// When the model declares a `[covariates]` block this routes through the strict
/// reader (validates declared columns exist + are numeric, builds the table, and
/// reads referenced-but-undeclared covariates leniently as `extra`). Otherwise
/// it falls back to the lenient reader with `fallback_columns` (the legacy
/// `covariate_columns` argument, or `None` for auto-detect).
///
/// When the model contains `[event_model]` blocks (TTE endpoints), TTE rows are
/// automatically routed to `subject.obs_records` instead of the Gaussian parallel
/// vectors. Library consumers (e.g. the R glue) should call this instead of the
/// individual `read_nonmem_csv*` functions so that TTE routing is applied.
pub fn read_population_for(
    model: &CompiledModel,
    covariate_decls: &Option<Vec<CovariateDecl>>,
    data_path: &str,
    fallback_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
) -> Result<(Population, Option<CovariateTable>), String> {
    // Extract TTE CMTs from model endpoints so the reader can route TTE rows
    // to obs_records instead of the Gaussian parallel Vecs.
    #[cfg(feature = "survival")]
    let tte_cmts: std::collections::HashSet<usize> = model
        .endpoints
        .iter()
        .filter_map(|(&cmt, ep)| {
            if matches!(ep, EndpointLikelihood::Tte { .. }) {
                Some(cmt)
            } else {
                None
            }
        })
        .collect();
    #[cfg(not(feature = "survival"))]
    let tte_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    if tte_cmts.is_empty() {
        // Gaussian-only model: use the existing (faster) path without TTE overhead.
        match (covariate_decls, filter) {
            (Some(decls), Some(sel)) => {
                let extra = undeclared_referenced(model, decls);
                let (pop, table) = read_nonmem_csv_with_covariates_filtered(
                    Path::new(data_path),
                    decls,
                    &extra,
                    iov_column,
                    sel,
                )?;
                Ok((pop, Some(table)))
            }
            (Some(decls), None) => {
                let extra = undeclared_referenced(model, decls);
                let (pop, table) = read_nonmem_csv_with_covariates(
                    Path::new(data_path),
                    decls,
                    &extra,
                    iov_column,
                )?;
                Ok((pop, Some(table)))
            }
            (None, Some(sel)) => Ok((
                read_nonmem_csv_filtered(Path::new(data_path), fallback_columns, iov_column, sel)?,
                None,
            )),
            (None, None) => Ok((
                read_nonmem_csv(Path::new(data_path), fallback_columns, iov_column)?,
                None,
            )),
        }
    } else {
        // Model has TTE endpoints: use TTE-aware reader so obs_records are populated.
        match covariate_decls {
            Some(decls) => {
                let extra = undeclared_referenced(model, decls);
                let (pop, table) = read_nonmem_csv_with_covariates_tte(
                    Path::new(data_path),
                    decls,
                    &extra,
                    iov_column,
                    filter,
                    &tte_cmts,
                )?;
                Ok((pop, Some(table)))
            }
            None => {
                let pop = read_nonmem_csv_filtered_tte(
                    Path::new(data_path),
                    fallback_columns,
                    iov_column,
                    filter,
                    &tte_cmts,
                )?;
                Ok((pop, None))
            }
        }
    }
}

/// Map an error string from [`read_population_for`] onto a `ferx check`
/// diagnostic, so the covariate-validation failures the strict reader raises at
/// fit time surface with the same code/block in `ferx check` (rather than as a
/// generic `E_DATA`). Classification keys off the reader's stable message
/// prefixes ([`ERR_COV_MISSING_COLUMNS`] / [`ERR_COV_NON_NUMERIC`]).
fn covariate_read_diagnostic(err: &str, path: &str) -> Diagnostic {
    if err.starts_with(ERR_COV_MISSING_COLUMNS) {
        Diagnostic::error("E_MISSING_COVARIATE", err.to_string()).with_block("covariates")
    } else if err.starts_with(ERR_COV_NON_NUMERIC) {
        Diagnostic::error("E_COVARIATE_NOT_NUMERIC", err.to_string()).with_block("covariates")
    } else {
        Diagnostic::error(
            "E_DATA",
            format!("Failed to read data file '{}': {}", path, err),
        )
    }
}

/// Per-CMT scaling needs every observed CMT to have an entry in the
/// `ScalingSpec::PerCmt` / `OdeReadout::PerCmt` map. Wraps the existing
/// `pk::validate_per_cmt_scaling` (which the parser can't run — it doesn't see
/// the data), preserving its message verbatim.
fn check_per_cmt_scaling(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    match pk::validate_per_cmt_scaling(model, &population.subjects) {
        Ok(()) => Vec::new(),
        Err(msg) => vec![Diagnostic::error("E_PER_CMT_SCALING", msg).with_block("scaling")],
    }
}

/// Per-CMT (multi-endpoint) error models: every observed CMT must have a
/// matching `CMT=N:` entry in `[error_model]`.
fn check_per_cmt_error_model(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let crate::types::ErrorSpec::PerCmt(map) = &model.error_spec else {
        return Vec::new();
    };
    use std::collections::BTreeSet;
    let mut missing = BTreeSet::new();
    for subj in &population.subjects {
        for &cmt in &subj.obs_cmts {
            if !map.contains_key(&cmt) {
                missing.insert(cmt);
            }
        }
    }
    if missing.is_empty() {
        return Vec::new();
    }
    let list = missing
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    vec![Diagnostic::error(
        "E_PER_CMT_ERROR_MODEL",
        format!(
            "[error_model] has no entry for observed compartment(s) {}; \
             add a `CMT=N: DV ~ ...` line for each observed CMT.",
            list
        ),
    )
    .with_block("error_model")]
}

/// All data-dependent *fatal* compatibility checks between a compiled model and
/// a dataset, collected into one diagnostic list. Shared by `fit()` (which
/// stops at the first error via [`first_error`]) and `ferx check` (which
/// reports every finding). Check order matches the historical inline order in
/// `fit()` so the first error is unchanged: covariates, scaling, error model,
/// iov occasions.
pub fn check_model_data(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let mut diags = check_covariates(model, population);
    diags.extend(check_per_cmt_scaling(model, population));
    diags.extend(check_per_cmt_error_model(model, population));
    diags.extend(check_iov_occasions(model, population));
    diags.extend(check_absorption_dosing(model, population));
    diags.extend(check_modeled_dose_rates(model, population));
    diags.extend(validate_output_columns(model, population));
    diags
}

/// IOV models require occasion labels in the dataset. When `n_kappa > 0` but
/// every subject has an empty `occasions` vector the kappa random effects are
/// silently ignored — catch this early instead.
fn check_iov_occasions(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    if model.n_kappa == 0 {
        return Vec::new();
    }
    // `all()` on an empty iterator is vacuously true; an empty population is not
    // a missing-OCC problem so skip the check when there are no subjects.
    let all_empty = !population.subjects.is_empty()
        && population.subjects.iter().all(|s| s.occasions.is_empty());
    if !all_empty {
        return Vec::new();
    }
    vec![Diagnostic::error(
        "E_IOV_MISSING_OCC",
        "Model declares kappa (IOV) parameters but no occasion labels were found in the \
         dataset. Set `iov_column = \"OCC\"` (or the relevant column name) in \
         [fit_options] so that per-occasion kappas can be estimated.",
    )
    .with_block("fit_options")]
}

/// Built-in absorption input-rate models (e.g. `transit()`) are integrated
/// through the standard ODE prediction paths. Two combinations are not yet
/// supported and are rejected here — loudly — rather than silently mis-modeled:
///   - a steady-state dose (`SS=1`) into an input-rate compartment: periodic
///     steady state with an unfinished `R_in` tail needs dedicated treatment
///     (a later phase of `plans/absorption-models.md`);
///   - an input-rate model combined with a `[diffusion]` block (the SDE/EKF
///     path), whose Kalman propagation does not carry the `R_in` forcing.
fn check_absorption_dosing(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let Some(ode) = &model.ode_spec else {
        return Vec::new();
    };
    if ode.input_rate.is_empty() {
        return Vec::new();
    }
    let mut diags = Vec::new();

    // input-rate + [diffusion]/EKF (model-level): the EKF propagator does not
    // apply the R_in forcing, so the absorption term would be silently dropped.
    if !ode.diffusion_var.is_empty() {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_DIFFUSION",
                "A built-in absorption input-rate model (e.g. transit() or igd()) cannot yet \
                 be combined with a [diffusion] block (the SDE/EKF path): the EKF \
                 propagation does not carry the input-rate forcing. Remove the \
                 [diffusion] block or the absorption term.",
            )
            .with_block("odes"),
        );
    }

    // SS=1 dose into an input-rate compartment (data-level): the steady-state
    // equilibration applies the dose as a bolus pulse, not as R_in over the cycle.
    use std::collections::BTreeSet;
    let cmts: BTreeSet<usize> = ode.input_rate.iter().map(|f| f.cmt + 1).collect();
    let has_ss = population.subjects.iter().any(|s| {
        s.doses
            .iter()
            .any(|d| d.ss && d.ii > 0.0 && cmts.contains(&d.cmt))
    });
    if has_ss {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_SS",
                "Steady-state dosing (SS=1) into a built-in absorption input-rate \
                 compartment (e.g. transit() or igd()) is not yet supported. Expand the \
                 run-in with explicit dosing records, or remove the absorption term.",
            )
            .with_block("odes"),
        );
    }

    // Infusion (RATE>0) into an input-rate compartment (data-level): the dose
    // would be delivered twice — once as the `+rate` infusion injection in the
    // ODE RHS wrapper, and again as `R_in(tad)` superposed by the input-rate
    // forcing — silently ~doubling exposure. A transit dose carries its mass
    // through `R_in` from the bolus amount; an infusion rate on that record is
    // undefined, so reject it loudly. Both modeled coded-RATE forms (RATE=-2
    // duration, RATE=-1 rate) are also infusions (`is_infusion()` is true for
    // them), so they are caught here too.
    let has_infusion = population.subjects.iter().any(|s| {
        s.doses
            .iter()
            .any(|d| d.is_infusion() && cmts.contains(&d.cmt))
    });
    if has_infusion {
        diags.push(
            Diagnostic::error(
                "E_ABSORPTION_RATE",
                "An infusion (RATE>0) into a built-in absorption input-rate \
                 compartment (e.g. transit() or igd()) is not supported: the dose mass is \
                 delivered through the input-rate function R_in computed from the dose \
                 amount, so an infusion rate would double-count it. Use a plain bolus \
                 dose record (RATE=0) into the absorption compartment.",
            )
            .with_block("odes"),
        );
    }

    // Parameter-domain validation (data-level): an out-of-domain or non-finite
    // input-rate parameter (e.g. transit `mtt ≤ 0` / `n < 0`, or igd `mat`/`cv2
    // ≤ 0`) would otherwise propagate as a NaN through the ODE RHS and surface
    // only as an opaque fit failure. Evaluated on typical values (η = 0) per
    // subject, so a covariate relationship that pushes a subject's typical
    // parameter out of range is caught too. Reported once — a single fatal
    // error already halts the fit.
    // Pathway-fraction structural check (#388, model-level): fractions must
    // *partition* the dose on a compartment. ≥2 input-rate terms on one compartment
    // therefore must **each** carry a fraction (`FR*fn(...)`); a bare term alongside
    // a fractioned one (or two bare terms) would deliver the full `F·Dose` more than
    // once. A fraction is only meaningful across ≥2 partitioning terms, so a *lone*
    // fractioned term is rejected too (a single pathway is written bare).
    use std::collections::BTreeMap;
    let mut frac_count: BTreeMap<usize, (usize, usize)> = BTreeMap::new(); // cmt -> (total, fractioned)
    for f in &ode.input_rate {
        let e = frac_count.entry(f.cmt).or_insert((0, 0));
        e.0 += 1;
        if f.frac_slot.is_some() {
            e.1 += 1;
        }
    }
    for (&cmt, &(total, fractioned)) in &frac_count {
        if total >= 2 && fractioned != total {
            diags.push(
                Diagnostic::error(
                    "E_ABSORPTION_FRACTION",
                    format!(
                        "Compartment {} has {total} built-in absorption input-rate terms but \
                         only {fractioned} carry a pathway fraction. When more than one term \
                         feeds a compartment, each must be written `FR*fn(...)` with the \
                         fractions summing to 1 — otherwise the dose mass is counted more than \
                         once.",
                        cmt + 1
                    ),
                )
                .with_block("odes"),
            );
        } else if total == 1 && fractioned == 1 {
            // A *lone* fractioned term can't partition anything: `FR*fn(...)` as the
            // only input on a compartment is only ever valid at `FR = 1` (≡ a bare
            // term), so reject it here with a precise message instead of letting the
            // per-subject sum-check below report the confusing "fractions sum to 0.6,
            // not 1" (review #1). A single pathway is written bare; `F` handles
            // partial absorption.
            diags.push(
                Diagnostic::error(
                    "E_ABSORPTION_FRACTION",
                    format!(
                        "Compartment {} has a single input-rate term carrying a pathway fraction \
                         (`FR*fn(...)`). A pathway fraction only applies when ≥2 terms split the \
                         dose across a compartment — write a single pathway as a bare \
                         `+ fn(...)`, and use bioavailability `F` for partial absorption.",
                        cmt + 1
                    ),
                )
                .with_block("odes"),
            );
        }
    }

    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    'subjects: for subject in &population.subjects {
        let pk = (model.pk_param_fn)(&model.default_params.theta, &zero_eta, &subject.covariates);
        for forcing in &ode.input_rate {
            if let Err(msg) = forcing.validate(&pk.values) {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_DOMAIN",
                        format!(
                            "Built-in absorption input-rate parameter out of domain at typical \
                             values (subject {}): {msg}. Constrain the parameter so it stays \
                             positive (e.g. a log-normal `P = TVP * exp(ETA_P)` \
                             parameterisation).",
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
        }

        // Pathway-fraction value checks (#388, η = 0, per subject so a covariate on a
        // fraction is caught too): each fraction in (0, 1], and the fractions on a
        // compartment sum to ≈ 1. Reported once — a single fatal error halts the fit.
        let mut frac_sum: BTreeMap<usize, f64> = BTreeMap::new();
        for forcing in &ode.input_rate {
            let Some(slot) = forcing.frac_slot else {
                continue;
            };
            let fr = pk.values.get(slot).copied().unwrap_or(1.0);
            if !(fr > 0.0 && fr <= 1.0) {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_FRACTION",
                        format!(
                            "Pathway fraction out of range at typical values (subject {}): \
                             {fr} is not in (0, 1]. A `FR*fn(...)` multiplier is a \
                             dose-splitting fraction — constrain it to (0, 1] (e.g. a logit \
                             `FR = 1/(1+exp(-X))` parameterisation).",
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
            *frac_sum.entry(forcing.cmt).or_insert(0.0) += fr;
        }
        for (&cmt, &sum) in &frac_sum {
            // Only a genuine ≥2-pathway split is checked for Σ ≈ 1; a lone fractioned
            // term is already rejected structurally above, so don't double-report it
            // with the misleading "sum to <FR>, not 1" here (review #1).
            let multi = frac_count.get(&cmt).is_some_and(|&(total, _)| total >= 2);
            if multi && (sum - 1.0).abs() > 1e-4 {
                diags.push(
                    Diagnostic::error(
                        "E_ABSORPTION_FRACTION",
                        format!(
                            "Pathway fractions on compartment {} sum to {sum} at typical values \
                             (subject {}), not 1. Declare fractions that partition the dose — \
                             e.g. a parameter `FR` and a complementary `FR2 = 1 - FR`.",
                            cmt + 1,
                            subject.id
                        ),
                    )
                    .with_block("odes"),
                );
                break 'subjects;
            }
        }
    }

    diags
}

/// NONMEM coded `RATE=-2` (modeled infusion duration → `D{cmt}`) needs a
/// model-aware check the datareader cannot make (it has no model). It is fatal
/// — never a silent fall-through to a bolus (the original #324 bug):
///   - **Matching `D{cmt}` parameter.** A `RATE=-2` dose into compartment `n`
///     requires a `D{n}` parameter (so `resolve_rate` has a slot to read);
///     otherwise it is rejected. Supported on both engines: ODE models record
///     the slot on `ode_spec.dose_attr_map`, analytical models (#394) on
///     `model.dose_attr_map`.
///
/// Reported once per offending compartment (naming the first dose that hits it),
/// so a dataset with many `RATE=-2` rows yields one actionable error per cause.
/// (The `D{n}`-is-also-an-RHS-rate-constant name collision is handled the same
/// way `F{n}` is — a documented reserved-name note in `docs/`, not a runtime
/// check — see ode-models.md.)
fn check_modeled_dose_rates(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    use crate::types::{DoseAttr, RateMode};
    let mut diags = Vec::new();
    // De-dup by (attribute, compartment) so N identical coded-RATE rows give one
    // error, not N — and a `D{cmt}` and an `R{cmt}` dose into the same compartment
    // are reported independently rather than masking each other.
    let mut reported: std::collections::HashSet<(DoseAttr, usize)> =
        std::collections::HashSet::new();
    for subject in &population.subjects {
        for dose in &subject.doses {
            // Map the modeled mode to its dose attribute + NONMEM-faithful naming
            // (the `RATE` code, the noun, the `$PK` parameter letter, and the
            // engine-agnostic error code). `Fixed` doses need no slot.
            let (attr, code, kind, err_code, param) = match dose.rate_mode {
                RateMode::Fixed => continue,
                RateMode::ModeledDuration => (
                    DoseAttr::Duration,
                    "-2",
                    "duration",
                    "E_MODELED_DURATION_NO_PARAM",
                    "D",
                ),
                RateMode::ModeledRate => {
                    (DoseAttr::Rate, "-1", "rate", "E_MODELED_RATE_NO_PARAM", "R")
                }
            };
            if !reported.insert((attr, dose.cmt)) {
                continue;
            }
            let cmt = dose.cmt;
            // A coded-`RATE` dose into compartment `cmt` requires a matching
            // `D{cmt}`/`R{cmt}` parameter so `resolve_rate` has a slot to read — for
            // BOTH engines. `active_dose_attr_map()` returns the engine-correct map
            // (the `OdeSpec`'s for ODE models, the analytical field otherwise,
            // #324), so an absent slot is the same actionable error on either engine.
            let has_slot = model
                .active_dose_attr_map()
                .indexed_slot(attr, cmt)
                .is_some();
            if !has_slot {
                diags.push(
                    Diagnostic::error(
                        err_code,
                        format!(
                            "subject {}, time {}: RATE={code} (modeled infusion {kind}) into \
                             compartment {cmt} requires a `{param}{cmt}` parameter in \
                             [individual_parameters], but none is declared. Add \
                             `{param}{cmt} = ...` (the modeled {kind}), or supply an explicit \
                             positive RATE.",
                            subject.id, dose.time
                        ),
                    )
                    .with_block("individual_parameters"),
                );
            }
        }
    }
    diags
}

/// Precondition shared by [`predict`] and the `simulate*` family: every
/// modeled-`RATE` dose (#324: `RATE=-2` → `D{cmt}` duration, `RATE=-1` → `R{cmt}`
/// rate) must be supported by the model — the matching `D{cmt}`/`R{cmt}`
/// parameter exists and infuses an infusable compartment, on either engine.
///
/// `fit()` enforces this via [`first_error`] over the full [`check_model_data`],
/// but `predict()` / `simulate()` deliberately skip that data-check (they assume
/// a model the caller already validated, and run no other data validation). A
/// modeled dose slipping through would otherwise hit one of two failure modes
/// downstream that the per-path `debug_assert!` tripwires only catch in
/// debug/test builds — silently in release: a 0-rate "infusion" on the
/// analytical path, or [`DoseEvent::resolve_rate`]'s slot `.expect`. This gate
/// turns both into a loud, actionable panic carrying the same diagnostic message
/// `check_model_data` would have produced, reusing the single-source-of-truth
/// [`check_modeled_dose_rates`]. It is O(doses) and runs once per public call
/// (not in the inner loop), and is a no-op for the common all-`Fixed` dataset.
pub(crate) fn assert_modeled_doses_supported(model: &CompiledModel, population: &Population) {
    if let Err(msg) = first_error(&check_modeled_dose_rates(model, population)) {
        panic!(
            "predict()/simulate() received a dose the model cannot honour: {msg}\n\
             (fit() reports this as an error rather than panicking; validate with \
             `check_model_data` before predicting on untrusted input.)"
        );
    }
}

/// Model + estimation-option *compatibility* checks that don't depend on data:
/// estimation method vs an SDE (`[diffusion]`) model, IMP chain placement, and
/// optimizer vs IOV. These mirror the guards at the top of `fit_inner`, so a
/// clean `ferx check` and a `fit()` agree on which method/model combinations are
/// rejected (rather than reporting `valid: true` and then failing at fit time).
/// `fit_inner` consumes these via [`first_error`]; message text is identical to
/// the historical inline guards. Check order matches `fit_inner` so the first
/// error is unchanged.
pub fn check_model_options(model: &CompiledModel, options: &FitOptions) -> Vec<Diagnostic> {
    let chain = options.method_chain();
    let mut diags = Vec::new();

    // SDE ([diffusion]) is incompatible with SAEM, with the Gauss-Newton
    // methods, and with the analytic-sensitivity gradient path (EKF estimation
    // requires FD-FOCE/FOCEI).
    if model.is_sde() {
        if chain.iter().any(|&m| m == EstimationMethod::Saem) {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "method = saem is not compatible with a [diffusion] block. \
                     SDE / EKF estimation requires FOCE or FOCEI. Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if chain
            .iter()
            .any(|&m| matches!(m, EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid))
        {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "SDE ([diffusion]) is not supported with method = gn or gn_hybrid. \
                     Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if chain.iter().any(|&m| m == EstimationMethod::Impmap) {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "method = impmap is not compatible with a [diffusion] block. \
                     The EKF process-noise variance is not threaded through the IMPMAP \
                     importance-sampling likelihood. Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        // `gradient_method = ad` is rejected unconditionally below (E_AD_RETIRED),
        // so it needs no SDE-specific case here.
    }

    // IMPMAP does not yet support inter-occasion variability (κ / [iov]); the κ
    // sufficient statistics and Ω_iov M-step are a planned follow-up. Surface it
    // at check time so `ferx check` rejects it rather than the fit failing at
    // runtime (possibly after a chained warm-up stage has already run).
    if model.n_kappa > 0 && chain.iter().any(|&m| m == EstimationMethod::Impmap) {
        diags.push(
            Diagnostic::error(
                "E_IMPMAP_IOV_UNSUPPORTED",
                "method = impmap does not yet support inter-occasion variability \
                 (κ / [iov]). Use method = saem or method = focei for IOV models.",
            )
            .with_block("fit_options"),
        );
    }

    // Explicit `gradient_method = ad`: the Enzyme autodiff path was retired in
    // favour of the hand-rolled `Dual2` analytic sensitivities. Reject it rather
    // than silently running a different method. `auto` (analytic where in scope,
    // else FD) and `fd` are unaffected.
    if options.gradient_method == crate::types::GradientMethod::Ad {
        diags.push(
            Diagnostic::error(
                "E_AD_RETIRED",
                "gradient_method = ad is no longer supported: the Enzyme automatic-differentiation \
                 path was retired in favour of the analytic `Dual2` sensitivities. Set \
                 gradient_method = auto (uses the exact analytic gradient where it is in scope and \
                 falls back to finite differences otherwise) or fd.",
            )
            .with_block("fit_options"),
        );
    }

    // Custom residual-error magnitude (#484) is wired through the FOCE/FOCEI
    // objective (data term + Laplace curvature) only. SAEM's σ/θ M-step, the
    // Gauss-Newton BHHH gradient, and the importance-sampling likelihood read
    // the residual variance through paths that do not yet apply the
    // per-observation magnitude, so reject them up front rather than silently
    // fitting a mis-specified error model.
    if model.has_custom_ruv_magnitude() {
        for &m in &chain {
            if !matches!(m, EstimationMethod::Foce | EstimationMethod::FoceI) {
                diags.push(
                    Diagnostic::error(
                        "E_RUV_MAGNITUDE_METHOD_UNSUPPORTED",
                        "a custom residual-error magnitude (an [error_model] sigma written as an \
                         expression of TIME / covariates / thetas) is currently supported for \
                         method = foce and method = focei only. SAEM, GN, GN-hybrid, and \
                         importance-sampling paths do not yet apply the per-observation magnitude.",
                    )
                    .with_block("error_model"),
                );
            }
        }
    }

    if !model.residual_correlations.is_empty() {
        for &m in &chain {
            if !matches!(m, EstimationMethod::Foce | EstimationMethod::Saem) {
                diags.push(
                    Diagnostic::error(
                        "E_BLOCK_SIGMA_METHOD_UNSUPPORTED",
                        "block_sigma correlated residual errors are currently supported for \
                         method = foce and method = saem only. FOCEI, GN, and \
                         importance-sampling paths still use diagonal residual-error derivatives.",
                    )
                    .with_block("fit_options"),
                );
                break;
            }
        }
        if matches!(model.bloq_method, BloqMethod::M3) {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_M3_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     M3 censored-observation likelihoods.",
                )
                .with_block("fit_options"),
            );
        }
        if model.frem_config.is_some() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_FREM_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     FREM covariate pseudo-observations.",
                )
                .with_block("fit_options"),
            );
        }
        if model.residual_error_eta.is_some() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_IIV_ON_RUV_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     iiv_on_ruv residual-error scaling.",
                )
                .with_block("fit_options"),
            );
        }
        if model.n_kappa > 0 {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_IOV_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     inter-occasion variability (κ / [iov]) because the IOV inner \
                     objective does not yet use the full residual covariance matrix.",
                )
                .with_block("fit_options"),
            );
        }
        // SDE is unsupported for SAEM only: the FOCE non-interaction path adds the
        // EKF process-noise `p_obs` into the dense R (foce_subject_nll_standard),
        // so FOCE + block_sigma + SDE works, but SAEM's M-step data term
        // (`obs_nll_subject_into`) carries no `p_obs`, so its E-step sampler and
        // M-step objective would optimize different likelihoods. FOCE keeps the
        // combination it supported before this PR added SAEM support.
        let chain_has_saem = chain.iter().any(|&m| m == EstimationMethod::Saem);
        if chain_has_saem && model.is_sde() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_SDE_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     method = saem and SDE / EKF process-noise likelihoods.",
                )
                .with_block("fit_options"),
            );
        }
        // TTE is unsupported for every method: the TTE Laplace path
        // (foce_subject_nll_interaction_with_tte) accumulates R diagonally and
        // cannot represent the cross-endpoint covariance, so it would silently
        // drop the correlation for FOCE as well as SAEM.
        if model.has_tte() {
            diags.push(
                Diagnostic::error(
                    "E_BLOCK_SIGMA_TTE_UNSUPPORTED",
                    "block_sigma correlated residual errors are not yet supported with \
                     time-to-event endpoints.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // `imp` may appear at most once in a chain. By default it is an MCEM
    // estimator (NONMEM `METHOD=IMP`) and may sit anywhere in the chain. With
    // `imp_eval_only = true` (NONMEM `IMP EONLY=1`) it instead evaluates the
    // marginal −2 log L at fixed parameters and must be the terminal stage —
    // placing an evaluator mid-chain would leave `FitResult.importance_sampling`
    // computed at parameters the following stage then overwrites.
    if chain.iter().any(|&m| m == EstimationMethod::Imp) {
        let n_imp = chain
            .iter()
            .filter(|&&m| m == EstimationMethod::Imp)
            .count();
        if n_imp > 1 {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` may appear at most once in a chain.",
                )
                .with_block("fit_options"),
            );
        }
        if options.imp_eval_only && chain.last().copied() != Some(EstimationMethod::Imp) {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` with `imp_eval_only = true` must be the final stage of the chain \
                     — placing the evaluator mid-chain would leave `FitResult.importance_sampling` \
                     populated with a log-likelihood computed at parameters that the following \
                     stage then overwrites. Move `imp` to the end, or drop `imp_eval_only` to run \
                     it as an estimator.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // The trust-region outer optimizer does not thread kappas through its OFV.
    if model.n_kappa > 0 && options.optimizer == Optimizer::TrustRegion {
        diags.push(
            Diagnostic::error(
                "E_OPTIMIZER_IOV",
                "optimizer = trust_region does not support IOV (n_kappa > 0). \
                 Use optimizer = bobyqa, slsqp, lbfgs, nlopt_lbfgs, mma, or bfgs \
                 for models with kappa declarations.",
            )
            .with_block("fit_options"),
        );
    }

    diags
}

/// Data-dependent *warning*-level checks: malformed steady-state rows, EVID=3/4
/// resets under an SDE model, and a negative typical-value lag time. These are
/// non-fatal — `fit()` pushes their messages into `FitResult.warnings` and
/// proceeds; `ferx check` reports them as `Warning` diagnostics. Message text
/// is identical to the historical inline strings.
///
/// Feature-presence (data-independent) notices such as the experimental-feature
/// warnings live in [`check_experimental_features`] instead, so `ferx check`
/// surfaces them even without a `--data` file.
pub fn check_model_data_warnings(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // SS=1 with II ≤ 0 — the SS branch is gated on `dose.ii > 0`, so the dose
    // is silently treated as a single (non-SS) dose.
    let n_ss_bad_ii = population
        .subjects
        .iter()
        .filter(|s| s.doses.iter().any(|d| d.ss && d.ii <= 0.0))
        .count();
    if n_ss_bad_ii > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_II",
            format!(
                "{} subject(s) have SS=1 doses with missing or non-positive II. \
                 SS predictions require II > 0 — these doses are treated as \
                 non-SS (no steady-state pre-equilibration). Set II in the \
                 dataset or remove the SS flag.",
                n_ss_bad_ii
            ),
        ));
    }

    // SS=1 infusion with T_inf > II (overlapping pulses). The analytical 1-/2-/3-
    // cpt closed forms now superpose the overlapping past pulse train (#379), so
    // these are handled exactly. Only ODE models — and subjects that route to the
    // event-driven walker (EVID 3/4 resets) — still skip SS pre-equilibration.
    let analytic_handles_overlap = model.ode_spec.is_none();
    // The effective infusion length is
    // `d.duration` for an ordinary infusion, but for a modeled dose (RATE=-2 →
    // `D{cmt}` duration, or RATE=-1 → `R{cmt}` rate; #324) it is unresolved here
    // (`rate`/`duration` are 0 until `resolve_rate`), so resolve it at the
    // typical-value point through the engine-correct `active_dose_attr_map()`.
    //
    // Resolution goes through the single-source-of-truth `DoseEvent::resolve_rate`
    // (the same rule + floor clamps the integrator applies at runtime), not a
    // hand-rolled slot read, so the warning's notion of "duration" can't drift
    // from the integrator's. It is a *typical-value* heuristic: it uses init
    // theta, eta = 0, and this subject's covariates, whereas runtime uses
    // per-occasion eta/IOV — a modeled SS infusion whose duration crosses `II`
    // only on some occasions may not be flagged here (and conversely a typical
    // overlap may not occur on every occasion). The runtime SS-skip is the
    // backstop; this catches the common typical-value / covariate-driven overlap.
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let effective_duration = |s: &Subject, d: &DoseEvent| -> f64 {
        use crate::types::{DoseAttr, RateMode};
        // The dose attribute whose slot holds this modeled dose's value.
        let attr = match d.rate_mode {
            RateMode::Fixed => return d.duration,
            RateMode::ModeledDuration => DoseAttr::Duration,
            RateMode::ModeledRate => DoseAttr::Rate,
        };
        // Guard the slot's existence: a modeled dose with no matching parameter is
        // an upstream *error* (`E_MODELED_DURATION_NO_PARAM` / `E_MODELED_RATE_NO_PARAM`),
        // but this warnings pass must stay panic-free if run on such a model rather
        // than hit `resolve_rate`'s slot `.expect`.
        let attr_map = model.active_dose_attr_map();
        if attr_map.indexed_slot(attr, d.cmt).is_some() {
            let pk = (model.pk_param_fn)(&init_params.theta, &zero_eta, &s.covariates);
            d.resolve_rate(attr_map, &pk.values).duration
        } else {
            0.0
        }
    };
    let n_ss_overlapping_inf = population
        .subjects
        .iter()
        .filter(|s| {
            let overlapping = s
                .doses
                .iter()
                .any(|d| d.ss && d.ii > 0.0 && d.is_infusion() && effective_duration(s, d) > d.ii);
            overlapping && (!analytic_handles_overlap || s.has_resets())
        })
        .count();
    if n_ss_overlapping_inf > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_INFUSION",
            format!(
                "{} subject(s) have SS=1 infusions with T_inf > II (overlapping \
                 pulses) on a model path that does not pre-equilibrate steady \
                 state (ODE model, or EVID=3/4 resets routing to the event-driven \
                 walker) — the dose is applied as a single (non-SS) infusion, so \
                 the system is not at steady state at the dose time. The analytical \
                 1-/2-/3-cpt closed forms do handle this case; use a shorter \
                 infusion (T_inf ≤ II) or remove the SS flag otherwise.",
                n_ss_overlapping_inf
            ),
        ));
    }

    // SS=1 dose combined with an [initial_conditions] baseline (#521). The SS
    // closed form already assumes an infinite periodic dose history (the system
    // is pre-equilibrated at the dose time), while the analytical init lays down a
    // separate transient baseline that decays from t=0. Superposing the two is a
    // valid linear combination but rarely what a user intends — an SS dose and an
    // explicit starting amount usually express the same idea twice, double-counting
    // the baseline. Surface it rather than silently combine them.
    if !model.analytical_init.is_empty() {
        let n_ss_init = population
            .subjects
            .iter()
            .filter(|s| s.has_ss_doses())
            .count();
        if n_ss_init > 0 {
            diags.push(Diagnostic::warning(
                "W_STEADY_STATE_INIT",
                format!(
                    "{} subject(s) have SS=1 doses while the model declares an \
                     [initial_conditions] baseline. The steady-state closed form \
                     already pre-equilibrates an infinite dose history; the init \
                     baseline is superposed on top as a separate decaying transient, \
                     which double-counts the starting amount. Remove the SS flag or \
                     the [initial_conditions] block unless this superposition is \
                     intended.",
                    n_ss_init
                ),
            ));
        }
    }

    // Modeled coded-RATE parameters (`D{cmt}` for RATE=-2 duration, `R{cmt}` for
    // RATE=-1 rate; #324) that are non-positive at the initial typical-value point
    // (eta = 0). `resolve_rate` clamps a transient `D ≤ 0` / `R ≤ 0` to its floor
    // so `AMT/D` / `AMT/R` stays finite mid-search, but a non-positive value *at
    // the initial estimate* signals a misspecified parameterisation — a `D ≤ 0`
    // delivers the dose as a bolus-like spike, an `R ≤ 0` as a near-zero trickle
    // over a near-infinite duration — and the fit can converge wrong with no other
    // diagnostic. Flag it (analogous to W_NEGATIVE_LAGTIME) and point at a
    // positive-link parameterisation. Engine-agnostic via `active_dose_attr_map()`
    // (covers analytical models too, #394/#395). De-duped per compartment per kind.
    {
        use crate::types::{DoseAttr, DoseEvent, RateMode};
        let attr_map = model.active_dose_attr_map();
        let mut nonpos_dur: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut nonpos_rate: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for s in &population.subjects {
            let mut pk_at_init: Option<crate::types::PkParams> = None;
            for d in &s.doses {
                let (attr, floor) = match d.rate_mode {
                    RateMode::Fixed => continue,
                    RateMode::ModeledDuration => (DoseAttr::Duration, DoseEvent::DURATION_FLOOR),
                    RateMode::ModeledRate => (DoseAttr::Rate, DoseEvent::RATE_FLOOR),
                };
                if let Some(slot) = attr_map.indexed_slot(attr, d.cmt) {
                    let pk = pk_at_init.get_or_insert_with(|| {
                        (model.pk_param_fn)(&init_params.theta, &zero_eta, &s.covariates)
                    });
                    if pk.values[slot] <= floor {
                        match attr {
                            DoseAttr::Duration => nonpos_dur.insert(d.cmt),
                            DoseAttr::Rate => nonpos_rate.insert(d.cmt),
                            DoseAttr::F | DoseAttr::Lag => unreachable!("only D/R reach here"),
                        };
                    }
                }
            }
        }
        for cmt in nonpos_dur {
            diags.push(Diagnostic::warning(
                "W_MODELED_DURATION_NONPOSITIVE",
                format!(
                    "Modeled infusion duration D{cmt} (RATE=-2 into compartment \
                     {cmt}) evaluates to ≤ 0 at the initial typical-value point \
                     (eta = 0). A non-positive duration is clamped to {floor:e} to \
                     keep AMT/D finite, which delivers the dose as a bolus-like \
                     spike rather than an infusion — the fit may converge to a \
                     wrong optimum. Use a positive-link parameterisation \
                     (e.g. D{cmt} = exp(...)).",
                    cmt = cmt,
                    floor = DoseEvent::DURATION_FLOOR,
                ),
            ));
        }
        for cmt in nonpos_rate {
            diags.push(Diagnostic::warning(
                "W_MODELED_RATE_NONPOSITIVE",
                format!(
                    "Modeled infusion rate R{cmt} (RATE=-1 into compartment \
                     {cmt}) evaluates to ≤ 0 at the initial typical-value point \
                     (eta = 0). A non-positive rate is clamped to {floor:e} to keep \
                     the implied duration AMT/R finite, which delivers the dose as a \
                     near-zero trickle over a near-infinite duration — the fit may \
                     converge to a wrong optimum. Use a positive-link \
                     parameterisation (e.g. R{cmt} = exp(...)).",
                    cmt = cmt,
                    floor = DoseEvent::RATE_FLOOR,
                ),
            ));
        }
    }

    // EVID=3/4 resets are not honoured on the EKF/SDE path.
    if model.is_sde() {
        let n_reset_sde = population
            .subjects
            .iter()
            .filter(|s| s.has_resets())
            .count();
        if n_reset_sde > 0 {
            diags.push(Diagnostic::warning(
                "W_SDE_RESET",
                format!(
                    "{} subject(s) have EVID=3/4 reset rows with a [diffusion] (SDE) \
                     model. System resets are not yet honoured on the EKF/SDE path — \
                     the resets are ignored and compartment amounts carry through. \
                     Use an ODE or analytical model if resets are required.",
                    n_reset_sde
                ),
            ));
        }
    }

    // Negative typical-value lag time at the initial point (eta = 0).
    if model.has_lagtime() {
        if let Some(first_subj) = population.subjects.first() {
            let zero_eta = vec![0.0_f64; model.n_eta];
            let pk = (model.pk_param_fn)(&init_params.theta, &zero_eta, &first_subj.covariates);
            if pk.lagtime() < 0.0 {
                diags.push(Diagnostic::warning(
                    "W_NEGATIVE_LAGTIME",
                    format!(
                        "Lagtime evaluates to {:.4} (< 0) at the initial typical-value \
                         point (eta = 0). Negative lagtimes are physically nonsensical \
                         and are not clamped — consider an exp() or other positive-link \
                         parameterisation.",
                        pk.lagtime()
                    ),
                ));
            }
            // Compartment-indexed `ALAGn` lags (issue #369) live in their own
            // spare slots, so the bare check above never sees them — flag each one
            // that is negative at the typical-value point. Ordered by compartment
            // for deterministic diagnostics.
            if let Some(ode) = &model.ode_spec {
                for cmt in 1..=ode.n_states {
                    let Some(slot) = ode
                        .dose_attr_map
                        .indexed_slot(crate::types::DoseAttr::Lag, cmt)
                    else {
                        continue;
                    };
                    let lag = pk.values.get(slot).copied().unwrap_or(0.0);
                    if lag < 0.0 {
                        diags.push(Diagnostic::warning(
                            "W_NEGATIVE_LAGTIME",
                            format!(
                                "ALAG{cmt} (compartment-{cmt} lag) evaluates to {lag:.4} (< 0) \
                                 at the initial typical-value point (eta = 0). Negative lagtimes \
                                 are physically nonsensical and are not clamped — consider an \
                                 exp() or other positive-link parameterisation."
                            ),
                        ));
                    }
                }
            }
        }
    }

    diags
}

/// Feature-presence (data-independent) *warning*-level checks for experimental
/// features (issue #175). Stochastic differential equations and neural-network
/// components are classified `experimental` in the Feature Maturity docs: tested
/// only on a handful of toy examples. We emit a warning whenever they are used
/// so results are applied with appropriate caution.
///
/// Kept separate from [`check_model_data_warnings`] because these depend only on
/// the compiled `model`, not on the dataset — so `ferx check model.ferx` (no
/// `--data`) and `fit()` both surface them. Non-fatal: `fit()` pushes the
/// messages into `FitResult.warnings`; `ferx check` reports them as warnings.
pub fn check_experimental_features(model: &CompiledModel) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    if model.is_sde() {
        diags.push(Diagnostic::warning(
            "W_EXPERIMENTAL_SDE",
            "Stochastic differential equations ([diffusion] / Extended Kalman \
             Filter) are an EXPERIMENTAL feature: validated only on a small set \
             of toy examples, with estimator support limited to FOCE/FOCEI. \
             Standard errors and convergence behaviour are not yet proven across \
             diverse datasets — validate results carefully before relying on \
             them. See the Feature Maturity page in the documentation.",
        ));
    }
    #[cfg(feature = "nn")]
    if !model.covariate_nns.is_empty() {
        diags.push(Diagnostic::warning(
            "W_EXPERIMENTAL_NN",
            "Neural-network model components ([covariate_nn] / deep compartment \
             models) are an EXPERIMENTAL feature: validated only on a small set \
             of toy examples. Standard errors for network weights are not \
             reliable and the syntax may still change — validate results \
             carefully before relying on them. See the Feature Maturity page in \
             the documentation.",
        ));
    }

    diags
}

/// Map a free-text parser error string to a single structured [`Diagnostic`].
/// Recognises the `"Missing [X] block"` shape (→ `E_MISSING_BLOCK`, with the
/// block name attached) and the `--features nn` gate (→ `E_NN_FEATURE_DISABLED`);
/// everything else is a generic `E_PARSE`.
fn parse_error_to_diagnostic(err: &str) -> Diagnostic {
    if let Some(rest) = err.strip_prefix("Missing [") {
        if let Some(end) = rest.find(']') {
            let block = &rest[..end];
            return Diagnostic::error("E_MISSING_BLOCK", err.to_string()).with_block(block);
        }
    }
    if err.contains("[covariate_nn]") && err.contains("--features nn") {
        return Diagnostic::error("E_NN_FEATURE_DISABLED", err.to_string())
            .with_block("covariate_nn");
    }
    Diagnostic::error("E_PARSE", err.to_string())
}

/// Validate a model file (and optionally a dataset) **without fitting**.
///
/// Runs the parser plus every data-independent and data-dependent check,
/// collecting *all* findings into a [`CheckReport`] (rather than stopping at the
/// first, as `fit()` does). This is the engine behind the `ferx check` CLI
/// command and is the fast `author → diagnose → fix` loop for tools and agents.
///
/// When `data_path` is `None`, only parse/structural and model/option
/// compatibility validation runs (no data is read). When it is `Some`, the CSV
/// is read and the covariate / per-CMT / steady-state / lag-time checks run as
/// well.
pub fn validate_model_file(model_path: &str, data_path: Option<&str>) -> CheckReport {
    use crate::parser::model_parser::parse_full_model_file;

    let model_name = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let data = data_path.map(|s| s.to_string());

    // 1. Parse. A parse failure is terminal — without an AST there is nothing
    //    further to validate, so return a report carrying just that diagnostic.
    let parsed = match parse_full_model_file(Path::new(model_path)) {
        Ok(p) => p,
        Err(e) => {
            return CheckReport::new(model_name, data, vec![parse_error_to_diagnostic(&e)]);
        }
    };

    let mut diags: Vec<Diagnostic> = Vec::new();

    // 2a. Parse-time warnings collected during parsing (unused parameters,
    //     mu-referencing diagnostics, etc.). Each warning embeds its own block
    //     context in the message text; we use W_PARSE as the generic code here
    //     rather than a narrower code that would mislabel unrelated warnings.
    for w in &parsed.model.parse_warnings {
        let code = if w.contains("declared in [parameters] but not referenced") {
            "W_UNUSED_PARAM"
        } else if w.contains("W_DERIVED_COVARIATE_SHADOW") {
            "W_DERIVED_COVARIATE_SHADOW"
        } else if w.contains("W_DERIVED_STEP_IGNORED") {
            "W_DERIVED_STEP_IGNORED"
        } else {
            "W_PARSE"
        };
        diags.push(Diagnostic::warning(code, w.clone()));
    }

    // 2b. Model / estimation-option compatibility (data-independent): catches
    //    method/model combinations that `fit()` rejects before fitting, so a
    //    clean check and a fit agree. Uses the parsed `[fit_options]`, mirroring
    //    what the CLI fit path (`run_model_with_data`) passes to `fit()`.
    diags.extend(check_model_options(&parsed.model, &parsed.fit_options));

    // 2c. Experimental-feature notices (data-independent): these depend only on
    //    the model, so they surface from `ferx check model.ferx` even without a
    //    `--data` file.
    diags.extend(check_experimental_features(&parsed.model));

    // 3. Data-dependent checks (only when a dataset is supplied). Read through
    //    the same covariate-aware chokepoint the fit uses, so `ferx check` and
    //    `fit()` apply identical covariate validation (declared columns present
    //    + numeric). A covariate-validation failure surfaces as the matching
    //    diagnostic rather than a generic read error.
    if let Some(path) = data_path {
        let iov_col = parsed.fit_options.iov_column.as_deref();
        match read_population_for(
            &parsed.model,
            &parsed.covariate_decls,
            path,
            None,
            iov_col,
            None,
        ) {
            Ok((population, _table)) => {
                // Surface datareader warnings (ADDL missing II, IOV OCC missing)
                // into the check report so `ferx check` sees the same findings as `fit()`.
                for w in &population.warnings {
                    let code = if w.starts_with("W_ADDL_MISSING_II") {
                        "W_ADDL_MISSING_II"
                    } else if w.starts_with("W_IOV_OCC_MISSING") {
                        "W_IOV_OCC_MISSING"
                    } else {
                        "W_DATA"
                    };
                    diags.push(Diagnostic::warning(code, w.clone()));
                }
                diags.extend(check_model_data(&parsed.model, &population));
                let init_params = parsed.model.default_params.clone();
                diags.extend(check_model_data_warnings(
                    &parsed.model,
                    &population,
                    &init_params,
                ));
            }
            Err(e) => {
                diags.push(covariate_read_diagnostic(&e, path));
            }
        }
    }

    // 4. Attach block-level line numbers to any diagnostic that named a block.
    for d in &mut diags {
        if d.line.is_none() {
            if let Some(block) = &d.block {
                if let Some(&ln) = parsed.block_lines.get(block) {
                    d.line = Some(ln);
                }
            }
        }
    }

    CheckReport::new(model_name, data, diags)
}

/// High-level fit: model file path + data file path → FitResult
pub fn fit_from_files(
    model_path: &str,
    data_path: &str,
    covariate_columns: Option<&[&str]>,
    options: Option<FitOptions>,
) -> Result<FitResult, String> {
    // Parse the full model so an authoritative `[covariates]` block is visible
    // here (the file's `[fit_options]` are still ignored — the caller's
    // `options` win, preserving historical behaviour).
    let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(model_path))?;
    let mut model = parsed.model;
    // A `[covariates]` declaration takes precedence over the explicit
    // `covariate_columns` argument; otherwise fall back to the argument (or
    // legacy auto-detect when both are absent).
    let opts = options.unwrap_or_default();
    let sel_filter_fit = build_selection_filter_merged(&parsed.fit_options, &opts)?;
    let (population, covariate_table) = read_population_for(
        &model,
        &parsed.covariate_decls,
        data_path,
        covariate_columns,
        None,
        sel_filter_fit.as_ref(),
    )?;
    model.bloq_method = opts.bloq_method;
    // SDE models have no analytic-sensitivity path — force FD.
    model.gradient_method =
        if model.is_sde() && opts.gradient_method != crate::types::GradientMethod::Fd {
            crate::types::GradientMethod::Fd
        } else {
            opts.gradient_method
        };
    let mut result = fit(&model, &population, &model.default_params, &opts)?;
    result.covariate_table = covariate_table;
    // Hash inputs post-fit (same pattern as `run_model_with_data`). The
    // model and CSV were already read by `parse_model_file` and
    // `read_nonmem_csv` upstream, so the OS page cache typically serves
    // these reads; failures are non-fatal and just disable the integrity
    // check in `run_sir`.
    result.model_path = Some(model_path.to_string());
    result.data_path = Some(data_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    result.data_hash = crate::io::hash::sha256_file(Path::new(data_path)).ok();
    result.model_text = std::fs::read_to_string(model_path).ok();
    Ok(result)
}

/// Perturb initial parameters for multi-start optimisation.
///
/// Start 0 always returns the unmodified params. Starts 1..n multiply each
/// log-packed theta by `exp(N(0, sigma))` and shift identity-packed thetas
/// (negative lower bound) by `sigma * N(0,1)`. Omega and sigma are left
/// unchanged — their starting values are typically less important than theta.
fn perturb_init(
    params: &ModelParameters,
    start_idx: usize,
    sigma: f64,
    base_seed: u64,
) -> ModelParameters {
    if start_idx == 0 {
        return params.clone();
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(base_seed.wrapping_add(start_idx as u64));
    let normal = Normal::new(0.0_f64, 1.0_f64).expect("normal dist");
    let mut p = params.clone();
    for (i, t) in p.theta.iter_mut().enumerate() {
        let lower = p.theta_lower.get(i).copied().unwrap_or(0.0);
        if theta_packs_log(lower) {
            *t *= (sigma * normal.sample(&mut rng)).exp();
        } else {
            *t += sigma * normal.sample(&mut rng);
        }
        // Clamp to bounds to avoid starting outside the feasible region
        let lo = p.theta_lower.get(i).copied().unwrap_or(f64::NEG_INFINITY);
        let hi = p.theta_upper.get(i).copied().unwrap_or(f64::INFINITY);
        *t = t.clamp(lo, hi);
    }
    p
}

/// Main fit entry point: CompiledModel + Population → FitResult.
///
/// When `options.threads` is `Some(n)`, the fit runs inside a scoped rayon
/// pool of `n` workers, so this setting is per-call (different fits in the
/// same process can use different thread counts). When `None`, rayon's
/// global pool is used (one worker per logical CPU).
///
/// `[data_selection]` filtering (`options.ignore_exprs` / `accept_exprs` /
/// `ignore_subjects`) is **not** applied here: it happens at CSV read time in
/// the file-based entry points (`run_model_with_data`, `fit_from_files`). This
/// function expects an already-filtered `Population` and simply echoes its
/// `exclusions` summary onto the result. Callers building a `Population` in
/// memory should filter their records beforehand.
pub fn fit(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // Apply the fit-scoped inner-loop optimizer choice before any EBE solve runs.
    // `Auto` (the default) reproduces the historical size-based dispatch, so this
    // is a no-op unless the user pinned `inner_optimizer`.
    crate::estimation::inner_optimizer::set_inner_optimizer(options.inner_optimizer);
    crate::estimation::inner_optimizer::set_ebe_warm_start(options.ebe_warm_start);
    // LTBS sanity checks for hand-built `CompiledModel`s. The parser already
    // enforces these for `.ferx` models, but a Rust caller could otherwise set
    // `log_transform = true` together with a proportional/combined error or a
    // per-CMT spec, which would make the likelihood inconsistent (predictions
    // log-wrapped while variance still expects natural-scale `f`). Fail fast.
    if model.log_transform {
        if !matches!(model.error_model, ErrorModel::Additive) {
            return Err(
                "LTBS (`log_transform = true`) requires `error_model = Additive`; \
                 proportional/combined error on the log scale is not supported"
                    .to_string(),
            );
        }
        if !matches!(model.error_spec, ErrorSpec::Single(_)) {
            return Err(
                "LTBS (`log_transform = true`) is not supported with per-CMT \
                 (`ErrorSpec::PerCmt`) error models"
                    .to_string(),
            );
        }
        if model.diffusion_theta_start.is_some() {
            return Err(
                "LTBS (`log_transform = true`) is not supported with an SDE \
                 model (`diffusion_theta_start = Some(_)`)"
                    .to_string(),
            );
        }
    }
    // IIV on residual error (`iiv_on_ruv`, #409) validation.
    if let Some(k) = model.residual_error_eta {
        // The residual-error eta must be a dedicated random effect: the FOCEI
        // `c̃` column assumes its prediction-Jacobian column is zero (it is not a
        // structural/individual-parameter eta). Reject a dual-use eta.
        if let Some(name) = model.eta_names.get(k) {
            if model.eta_param_info.iter().any(|e| &e.eta_name == name) {
                return Err(format!(
                    "[error_model] iiv_on_ruv = {name}: this eta is also used in \
                     [individual_parameters]; the residual-error random effect must be a \
                     dedicated omega not shared with a structural parameter"
                ));
            }
        }
        // IIV-on-RUV is inherently an interaction model (`Y = IPRED + EPS·EXP(ETA)`
        // makes the residual variance η-dependent). Non-interaction FOCE/GN cannot
        // represent it — its marginal integrates the residual eta out through a
        // sensitivity column that is identically zero. Require FOCEI or a
        // Monte-Carlo estimator (IMP/IMPMAP/SAEM).
        let methods: Vec<EstimationMethod> = if options.methods.is_empty() {
            vec![options.method]
        } else {
            options.methods.clone()
        };
        for m in &methods {
            let non_interaction = match m {
                EstimationMethod::Foce => true,
                EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => !options.interaction,
                _ => false,
            };
            if non_interaction {
                return Err(format!(
                    "IIV on residual error (iiv_on_ruv) requires an interaction or \
                     Monte-Carlo method: use method = focei, imp, impmap, or saem (got {m:?} \
                     with interaction = false). NONMEM `Y = IPRED + EPS*EXP(ETA)` is an \
                     INTERACTION model."
                ));
            }
        }
    }
    // Data-dependent fatal checks (covariates present, per-CMT scaling and
    // per-CMT error-model coverage). These can't run in the parser — it doesn't
    // see the data. `ferx check` runs the same `check_model_data` to report
    // every finding; here we stop at the first error to preserve fit()'s
    // historical fail-fast behavior and exact error strings.
    first_error(&check_model_data(model, population))?;
    // If any subject has per-event covariate snapshots that don't carry
    // a variation in covariates the model actually references (e.g.
    // DAY / STIME columns in NONMEM-format datasets), clear those
    // snapshots so the downstream prediction path routes through the
    // cheap analytical/no-TV fast path instead of the event-driven
    // path. Bigger wins on SAD-style datasets where every subject has
    // a varying DAY column but no model expression touches DAY.
    // Log-transform-both-sides (LTBS) case 2 (`log(DV) ~ additive`): the data's
    // DV is on the natural scale, so log-transform every observation once here,
    // before any prediction is scored against it. Case 1 (`DV ~ log_additive`,
    // `dv_pre_logged`) leaves the already-log DV untouched. Logging into the
    // owned clone leaves the caller's `Population` (and any `simulate` reuse of
    // it) unmodified, and avoids double-logging on repeated `fit()` calls.
    let needs_dv_log = model.log_transform && !model.dv_pre_logged;
    let mut ltbs_warnings: Vec<String> = Vec::new();
    let pop_pruned: std::borrow::Cow<Population> = {
        let needs_prune = population.subjects.iter().any(|s| {
            !s.dose_covariates.is_empty()
                || !s.obs_covariates.is_empty()
                || !s.pk_only_covariates.is_empty()
        });
        if needs_prune || needs_dv_log {
            let mut p = population.clone();
            if needs_prune {
                p.prune_irrelevant_tv_covariates(&model.referenced_covariates);
            }
            if needs_dv_log {
                let n_nonpos = log_transform_observations(&mut p);
                if n_nonpos > 0 {
                    ltbs_warnings.push(format!(
                        "LTBS (log(DV) ~ ...): {n_nonpos} observation(s) had DV ≤ 0, which \
                         cannot be log-transformed; they were floored to log({LTBS_FLOOR:e}). \
                         Check the data scale, or use `DV ~ log_additive(...)` if DV is \
                         already log-transformed.",
                        LTBS_FLOOR = crate::pk::LTBS_FLOOR,
                    ));
                }
            }
            std::borrow::Cow::Owned(p)
        } else {
            std::borrow::Cow::Borrowed(population)
        }
    };
    let pop_ref: &Population = &*pop_pruned;

    // Single-start fast path (default)
    if options.n_starts <= 1 {
        let res = match options.threads {
            Some(n) if n > 0 => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| format!("failed to build rayon pool with {} threads: {}", n, e))?;
                pool.install(|| fit_inner(model, pop_ref, init_params, options))
            }
            _ => fit_inner(model, pop_ref, init_params, options),
        };
        return res.map(|mut result| {
            result.warnings.splice(0..0, ltbs_warnings);
            rebuild_warnings_structured(&mut result);
            result
        });
    }

    // Multi-start: run n_starts fits in parallel, return the lowest-OFV converged result.
    // `threads` controls per-subject parallelism inside each start; in multi-start mode
    // we let the global rayon pool handle both levels (outer start × inner per-subject).
    // Creating a new ThreadPool per start inside an outer into_par_iter() spawns n_starts
    // independent pools that all compete on the same CPUs, causing oversubscription —
    // so we only honour `threads` when the global pool hasn't been entered yet (single-start
    // path above). Here we always use the global pool for the outer par_iter.
    let base_seed: u64 = options.multi_start_seed.unwrap_or(42);
    let base_saem_seed: u64 = options.saem_seed.unwrap_or(12345);
    let base_bayes_seed: u64 = options.bayes_seed.unwrap_or(12345);
    let n = options.n_starts;
    let sigma = options.start_sigma;

    // Warn once (before the parallel section) that global_search only runs on start 0.
    let mut pre_warnings: Vec<String> = ltbs_warnings;
    if options.global_search && n > 1 {
        pre_warnings.push(format!(
            "global_search = true with n_starts = {n}: CRS2-LM only runs on start 0 \
             (it ignores the starting point and would override the theta perturbation \
             on starts 1..{n})"
        ));
    }

    let results: Vec<(usize, Result<FitResult, String>)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let init_k = perturb_init(init_params, k, sigma, base_seed);
            // Per-start option overrides for k > 0:
            // - saem_seed / bayes_seed: derive from base so each start gets a different
            //   MH/MCMC trajectory. The Bayes sampler keys off bayes_seed, so without
            //   perturbing it every start runs an identical RNG trajectory (differing
            //   only by the perturbed init) — wasted compute and false multi-start
            //   robustness. Start 0 keeps the user's seeds for reproducibility.
            // - global_search: CRS2-LM ignores the starting point and samples freely in
            //   [lower, upper], so running it on starts 1..n overrides the perturbation
            //   and makes multi-start a no-op for those starts. Only run it on start 0.
            let opts_k_storage;
            let opts_ref: &FitOptions = if k == 0 {
                options
            } else {
                opts_k_storage = FitOptions {
                    saem_seed: Some(base_saem_seed.wrapping_add(k as u64)),
                    bayes_seed: Some(base_bayes_seed.wrapping_add(k as u64)),
                    global_search: false,
                    ..options.clone()
                };
                &opts_k_storage
            };
            (k, fit_inner(model, pop_ref, &init_k, opts_ref))
        })
        .collect();

    // Pick best converged result; fall back to best unconverged if none converged.
    let mut best: Option<(usize, FitResult)> = None;
    let mut failed_starts: Vec<String> = Vec::new();
    for (k, res) in results {
        match res {
            Ok(r) => {
                let better = match &best {
                    None => true,
                    Some((_, b)) => {
                        // Prefer converged over unconverged; then lower OFV
                        (!b.converged && r.converged)
                            || (b.converged == r.converged && r.ofv < b.ofv)
                    }
                };
                if better {
                    best = Some((k, r));
                }
            }
            Err(e) => failed_starts.push(format!("start {k}: {e}")),
        }
    }

    match best {
        None => Err("All multi-start fits failed".to_string()),
        Some((k, mut result)) => {
            result.warnings.splice(0..0, pre_warnings);
            if !failed_starts.is_empty() {
                result.warnings.push(format!(
                    "Multi-start: {} of {n} starts failed: {}",
                    failed_starts.len(),
                    failed_starts.join("; ")
                ));
            }
            if !result.converged {
                result.warnings.push(format!(
                    "No multi-start run converged ({n} starts); returning best OFV from start {k}"
                ));
            } else if k > 0 {
                result.warnings.push(format!(
                    "Multi-start: best result from start {k}/{n} (OFV = {:.4})",
                    result.ofv
                ));
            }
            rebuild_warnings_structured(&mut result);
            Ok(result)
        }
    }
}

/// Rebuild `warnings_structured` from the current `warnings` vec.
///
/// Called after all late-injected warnings (LTBS splice, multi-start metadata)
/// have been appended so the structured field is always in sync with the flat list.
fn rebuild_warnings_structured(result: &mut FitResult) {
    result.warnings_structured = result
        .warnings
        .iter()
        .map(|w| crate::types::classify_warning(w))
        .collect();
}

/// Probe whether NLopt CRS2-LM (used for global_search) is available.
fn probe_nlopt_algorithms() -> Vec<String> {
    fn dummy_obj(_x: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()) -> f64 {
        0.0
    }
    let available = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            1,
            dummy_obj,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if available.is_err() {
        vec![
            "NLopt CRS2-LM not available in this build — global_search = true will fail. \
             Install a full NLopt build: brew install nlopt / apt install libnlopt-dev"
                .to_string(),
        ]
    } else {
        vec![]
    }
}

// ── Step 7: [output] validation and TAFD/TAD helpers ────────────────────────

/// Mandatory sdtab column names that are always written — declaring them in
/// [output] is allowed but produces a W_OUTPUT_DUPLICATE warning.
const OUTPUT_MANDATORY: &[&str] = &[
    "ID", "TIME", "DV", "CENS", "OCC", "CMT", "PRED", "IPRED", "CWRES", "IWRES", "NPDE", "NPD",
    "EBE_OFV", "N_OBS", "TAFD", "TAD",
];

/// Validate `model.output_columns` against known quantities, emitting
/// `W_OUTPUT_DUPLICATE` and `E_OUTPUT_UNKNOWN_COLUMN` diagnostics.
pub fn validate_output_columns(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let derived_names: Vec<&str> = model
        .derived_exprs
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    let cov_names = &population.covariate_names;

    for col in &model.output_columns {
        // Already in mandatory minimum, or an ETA (reported via ebe_etas, not sdtab)?
        let is_eta = model.eta_names.iter().any(|e| e.eq_ignore_ascii_case(col));
        if OUTPUT_MANDATORY.iter().any(|m| m.eq_ignore_ascii_case(col)) || is_eta {
            let msg = if is_eta {
                // sdtab is per-observation only; per-subject EBEs live in
                // `ebe_etas` on the R side, so an ETA can't be an sdtab column.
                format!(
                    "[output] column `{col}` is an ETA estimate, reported via `ebe_etas` \
                     rather than as an sdtab column; the declaration is ignored"
                )
            } else {
                format!(
                    "[output] column `{col}` is already written to sdtab automatically; \
                     the declaration is ignored"
                )
            };
            diags.push(Diagnostic::warning("W_OUTPUT_DUPLICATE", msg));
            continue;
        }
        // Valid if it's a covariate, indiv param, or derived name
        let known = cov_names.iter().any(|c| c.eq_ignore_ascii_case(col))
            || model
                .indiv_param_names
                .iter()
                .any(|p| p.eq_ignore_ascii_case(col))
            || derived_names.iter().any(|d| d.eq_ignore_ascii_case(col));
        if !known {
            let mut candidates: Vec<&str> = cov_names.iter().map(|s| s.as_str()).collect();
            candidates.extend(model.indiv_param_names.iter().map(|s| s.as_str()));
            candidates.extend(derived_names.iter().copied());
            candidates.extend(OUTPUT_MANDATORY.iter().copied());
            diags.push(Diagnostic::error(
                "E_OUTPUT_UNKNOWN_COLUMN",
                format!(
                    "[output] column `{col}` is not recognised as a covariate, individual \
                     parameter, or derived expression. Known: {}",
                    candidates.join(", ")
                ),
            ));
        }
    }
    diags
}

/// Time after the most recent **absorbed** dose at time `t` (SS-aware), shifting
/// each dose by its own lag from `dose_lagtimes`. Missing entries — a slice
/// shorter than `subject.doses`, or `&[]` — default to zero lag. Returns NaN when
/// no dose has been absorbed by `t`. Shared by the per-observation TAD column and
/// the model-based integral grid so both apply identical per-dose-lag logic.
fn tad_at_time(subject: &Subject, t: f64, dose_lagtimes: &[f64]) -> f64 {
    let last_dose_eff = subject
        .doses
        .iter()
        .enumerate()
        .filter_map(|(d, dose)| {
            let lag = dose_lagtimes.get(d).copied().unwrap_or(0.0);
            if dose.time + lag > t + 1e-12 {
                return None;
            }
            let eff = if dose.ss && dose.ii > 0.0 {
                let elapsed = t - (dose.time + lag);
                t - elapsed.rem_euclid(dose.ii)
            } else {
                dose.time + lag
            };
            Some(eff)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if last_dose_eff.is_finite() {
        t - last_dose_eff
    } else {
        f64::NAN
    }
}

/// Compute TAFD (time after first dose) and TAD (time after last dose, SS-aware)
/// for observation index `obs_idx` of `subject`.
///
/// `dose_lagtimes[d]` is the absorption lag for dose `d`, evaluated with that
/// dose's occasion kappa and covariate snapshot (see [`crate::pk::predict_iov`]).
/// Each dose's effective arrival is `dose.time + dose_lagtimes[d]`, so under a lag
/// that varies across doses — IOV on the lag, or a time-varying covariate — a dose
/// given in one occasion is shifted by its *own* lag rather than the observation's,
/// which matters for the most-recent-dose pick (e.g. BID dosing spanning two
/// occasions). Missing entries default to zero lag, so callers with no lag can
/// pass `&[]`. TAFD is unaffected — measured from the raw first-dose time, not the
/// lagged arrival.
pub fn tafd_tad_for_subject(
    subject: &Subject,
    obs_idx: usize,
    dose_lagtimes: &[f64],
) -> (f64, f64) {
    let obs_time = subject.obs_times[obs_idx];
    let first_dose_time = subject.occasion_first_dose_time(obs_time);
    let tafd = if first_dose_time.is_finite() {
        obs_time - first_dose_time
    } else {
        f64::NAN
    };
    let tad = tad_at_time(subject, obs_time, dose_lagtimes);
    (tafd, tad)
}

// ── Step 8: post-fit extra column computation ────────────────────────────────

/// Build a per-observation HashMap mapping `model.indiv_param_names` to their
/// values from `pk`.
fn build_indiv_map(pk: &PkParams, names: &[String], pk_indices: &[usize]) -> HashMap<String, f64> {
    names
        .iter()
        .zip(pk_indices.iter())
        .map(|(name, &idx)| (name.clone(), pk.values[idx]))
        .collect()
}

/// Trapezoid integration over (time, value) pairs.
/// Observation times are not guaranteed to be sorted (preserved in input row
/// order), so sort by time before integrating to prevent negative dt windows.
fn trapezoid(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return f64::NAN;
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut auc = 0.0;
    for w in sorted.windows(2) {
        let dt = w[1].0 - w[0].0;
        auc += dt * (w[0].1 + w[1].1) * 0.5;
    }
    auc
}

/// Compute all [derived] and [output] columns post-fit, storing results in
/// each SubjectResult's `extra_columns` field.
pub(crate) fn compute_extra_output_columns(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    kappas_per_subject: &[Vec<DVector<f64>>],
    subjects: &mut [SubjectResult],
) {
    use crate::types::{AggFunction, DerivedContext, DerivedKind, IntegralStep, IntegralWindow};

    let derived_names: Vec<&str> = model
        .derived_exprs
        .iter()
        .map(|s| s.name.as_str())
        .collect();

    for (si, sr) in subjects.iter_mut().enumerate() {
        let subject = &population.subjects[si];
        let eta_hat = sr.eta.as_slice();
        let n_obs = sr.ipred.len();

        // Per-observation full eta vector [BSV η … | occasion κ …].
        //
        // `eta_hat` (= `sr.eta`) is BSV-only (length `n_eta`); for IOV models
        // (`n_kappa > 0`) `pk_param_fn` and `[derived]` expressions expect the
        // full `n_eta + n_kappa` vector, with the kappas belonging to *this
        // observation's occasion*. Mirror `pk::predict_iov`'s occasion→kappa
        // selection exactly so the post-fit derived/diagnostic columns use the
        // same per-occasion kappa as the predictions that drove the fit. Without
        // this the kappa slots silently read 0 for every observation (issue #238).
        let subj_kappas: &[DVector<f64>] = kappas_per_subject
            .get(si)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let occ_groups = crate::stats::likelihood::split_obs_by_occasion(subject);
        let mut occ_to_k: HashMap<u32, usize> = HashMap::with_capacity(occ_groups.len());
        for (k, (occ_id, _)) in occ_groups.iter().enumerate() {
            occ_to_k.insert(*occ_id, k);
        }
        let combined_for = |occ_id: u32| -> Vec<f64> {
            let mut c = Vec::with_capacity(eta_hat.len() + model.n_kappa);
            c.extend_from_slice(eta_hat);
            if model.n_kappa > 0 {
                match occ_to_k.get(&occ_id) {
                    Some(&k) if k < subj_kappas.len() => {
                        c.extend_from_slice(subj_kappas[k].as_slice())
                    }
                    _ => c.extend(std::iter::repeat_n(0.0, model.n_kappa)),
                }
            }
            c
        };
        let per_obs_eta_full: Vec<Vec<f64>> = (0..n_obs)
            .map(|j| combined_for(subject.occasions.get(j).copied().unwrap_or(0)))
            .collect();

        // Per-dose absorption lag, each evaluated with that dose's occasion kappa
        // and covariate snapshot (mirrors predict_iov's per-dose PK params). TAD
        // shifts every dose by its own lag, so a dose given in one occasion is not
        // mis-shifted by the observation's lag — matters when the lag varies across
        // doses (IOV on the lag, or a time-varying covariate) and dosing spans the
        // differing values (e.g. BID across two occasions). Computed once per
        // subject (dose-indexed). Skipped entirely when the model declares no lag:
        // `dose_lagtimes` stays empty and `tad_at_time` falls back to zero lag,
        // so the common no-lag case pays nothing for this per-dose pass.
        let dose_lagtimes: Vec<f64> = if model.has_lagtime() {
            (0..subject.doses.len())
                .map(|d| {
                    let occ = subject.dose_occasions.get(d).copied().unwrap_or(0);
                    let eta_d = combined_for(occ);
                    let pk_d = (model.pk_param_fn)(theta, &eta_d, subject.dose_cov(d));
                    // On ODE models the lag is keyed by dose compartment (`ALAGn`;
                    // issue #369), so resolve through `dose_attr_map` — the same
                    // single source of truth the prediction paths use — rather than
                    // the bare `PK_IDX_LAGTIME` slot, which a model declaring only
                    // `ALAG2` leaves at 0 (TAD would then ignore that route's lag).
                    // The analytical engine has one fixed route → the bare lag.
                    match &model.ode_spec {
                        Some(ode) => ode
                            .dose_attr_map
                            .lagtime(subject.doses[d].cmt, &pk_d.values),
                        None => pk_d.lagtime(),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        // Per-observation PK params, indiv maps, TAFD, TAD
        let mut per_obs_cov: Vec<&HashMap<String, f64>> = Vec::with_capacity(n_obs);
        let mut per_obs_indiv: Vec<HashMap<String, f64>> = Vec::with_capacity(n_obs);
        let mut per_obs_tafd: Vec<f64> = Vec::with_capacity(n_obs);
        let mut per_obs_tad: Vec<f64> = Vec::with_capacity(n_obs);

        for (j, eta_full) in per_obs_eta_full.iter().enumerate() {
            let cov_j = subject.obs_cov(j);
            let pk_j = (model.pk_param_fn)(theta, eta_full, cov_j);
            let indiv_j = build_indiv_map(&pk_j, &model.indiv_param_names, &model.pk_indices);
            let (tafd_j, tad_j) = tafd_tad_for_subject(subject, j, &dose_lagtimes);
            per_obs_cov.push(cov_j);
            per_obs_indiv.push(indiv_j);
            per_obs_tafd.push(tafd_j);
            per_obs_tad.push(tad_j);
        }

        // Store per-obs TAD (with individual lagtime) so output.rs can use it
        // for the mandatory TAD column without re-evaluating PK parameters.
        sr.per_obs_tad = per_obs_tad.clone();

        // Compartment states and names for [derived] expressions.
        // Empty slices are used for observations where states are not available
        // (IOV subjects, analytical TV-covariate subjects — see W_DERIVED_CMT_* warnings).
        let model_cmt_names: &[String] = model
            .ode_spec
            .as_ref()
            .map(|s| s.state_names.as_slice())
            .unwrap_or_else(|| model.analytical_compartment_names());
        let per_obs_cmts: Vec<&[f64]> = (0..n_obs)
            .map(|j| {
                sr.compartment_states
                    .get(j)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            })
            .collect();

        // Session infrastructure for EVID=3/4 stacked subjects.
        // For subjects with no resets (the common case) n_sessions=1, session_obs[0]
        // holds all observation indices, session_shift[0]=0, and obs_session[j]=0
        // for every j — zero overhead, identical downstream behaviour.
        let raw_time_of = |j: usize| -> f64 {
            subject
                .obs_raw_times
                .get(j)
                .copied()
                .unwrap_or(subject.obs_times[j])
        };
        let n_sessions = subject.reset_times.len() + 1;
        let (session_obs, session_shift): (Vec<Vec<usize>>, Vec<f64>) = {
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_sessions];
            for j in 0..n_obs {
                // 1e-9: datareader inserts RESET_SEGMENT_GAP = 1.0 h between
                // sessions, so no real observation lands within 1e-9 h of a
                // reset boundary.  Larger than the ±1e-12 used for integral
                // window filters, which must match exact user-supplied endpoints.
                let s = subject
                    .reset_times
                    .iter()
                    .filter(|&&r| r <= subject.obs_times[j] + 1e-9)
                    .count();
                groups[s].push(j);
            }
            let shifts: Vec<f64> = groups
                .iter()
                .map(|g| {
                    g.first()
                        .map(|&j| subject.obs_times[j] - raw_time_of(j))
                        .unwrap_or(0.0)
                })
                .collect();
            (groups, shifts)
        };
        // Invert session_obs: obs_session[j] = session index for observation j.
        // Derived by inversion in O(n_obs) rather than re-scanning reset_times.
        let mut obs_session = vec![0usize; n_obs];
        for (s, indices) in session_obs.iter().enumerate() {
            for &j in indices {
                obs_session[j] = s;
            }
        }

        // [output] columns: covariates + indiv params not already in derived
        for col_name in &model.output_columns {
            if derived_names
                .iter()
                .any(|d| d.eq_ignore_ascii_case(col_name))
            {
                continue; // will be filled by derived pass below
            }
            // Skip mandatory/duplicate columns
            if OUTPUT_MANDATORY
                .iter()
                .any(|m| m.eq_ignore_ascii_case(col_name))
                || model
                    .eta_names
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(col_name))
            {
                continue;
            }
            let mut col_vals = Vec::with_capacity(n_obs);
            for j in 0..n_obs {
                // Resolve covariates and individual parameters case-insensitively:
                // validate_output_columns accepts the [output] name regardless of
                // case, so the echo must match a header like `WT` against a
                // declared `wt` rather than silently producing NaN.
                let v = per_obs_cov[j]
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(col_name))
                    .map(|(_, v)| v)
                    .or_else(|| {
                        per_obs_indiv[j]
                            .iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case(col_name))
                            .map(|(_, v)| v)
                    })
                    .copied()
                    .unwrap_or(f64::NAN);
                col_vals.push(v);
            }
            sr.extra_columns.push((col_name.clone(), col_vals));
        }

        // [derived] columns, evaluated in declaration order.
        // prev_derived_vecs stores the full per-row vector for each column evaluated
        // so far. For Aggregate/Integral (same scalar every row), all elements are
        // identical. This allows sequential references (`B = f(A)`) to see the
        // correct per-row value at index j, not just the last row's value.
        let mut prev_derived_vecs: HashMap<String, Vec<f64>> = HashMap::new();

        for spec in &model.derived_exprs {
            let col_vals: Vec<f64> = match &spec.kind {
                DerivedKind::PerRow { eval } => (0..n_obs)
                    .map(|j| {
                        let row_prev: HashMap<String, f64> = prev_derived_vecs
                            .iter()
                            .map(|(k, v)| (k.clone(), v[j]))
                            .collect();
                        let ctx = DerivedContext {
                            theta,
                            eta: &per_obs_eta_full[j],
                            indiv_params: &per_obs_indiv[j],
                            covariates: per_obs_cov[j],
                            ipred: sr.ipred[j],
                            pred: sr.pred[j],
                            dv: subject.observations[j],
                            time: raw_time_of(j),
                            tafd: per_obs_tafd[j],
                            tad: per_obs_tad[j],
                            prev_derived: &row_prev,
                            compartments: per_obs_cmts[j],
                            compartment_names: model_cmt_names,
                        };
                        eval(&ctx)
                    })
                    .collect(),

                DerivedKind::Aggregate {
                    func,
                    value,
                    filter,
                } => {
                    let mut qualifying: Vec<(usize, f64)> = Vec::new();
                    for j in 0..n_obs {
                        let row_prev: HashMap<String, f64> = prev_derived_vecs
                            .iter()
                            .map(|(k, v)| (k.clone(), v[j]))
                            .collect();
                        let ctx = DerivedContext {
                            theta,
                            eta: &per_obs_eta_full[j],
                            indiv_params: &per_obs_indiv[j],
                            covariates: per_obs_cov[j],
                            ipred: sr.ipred[j],
                            pred: sr.pred[j],
                            dv: subject.observations[j],
                            time: raw_time_of(j),
                            tafd: per_obs_tafd[j],
                            tad: per_obs_tad[j],
                            prev_derived: &row_prev,
                            compartments: per_obs_cmts[j],
                            compartment_names: model_cmt_names,
                        };
                        let include = filter.as_ref().map_or(true, |f| f(&ctx));
                        if include {
                            qualifying.push((j, value(&ctx)));
                        }
                    }
                    let scalar = if qualifying.is_empty() {
                        f64::NAN
                    } else {
                        match func {
                            AggFunction::Max => qualifying
                                .iter()
                                .map(|(_, v)| *v)
                                .fold(f64::NEG_INFINITY, f64::max),
                            AggFunction::Min => qualifying
                                .iter()
                                .map(|(_, v)| *v)
                                .fold(f64::INFINITY, f64::min),
                            AggFunction::Tmax => {
                                // Time of maximum value; raw_time_of returns dataset
                                // TIME so the sdtab column reflects the user's clock.
                                qualifying
                                    .iter()
                                    .max_by(|(_, a), (_, b)| {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(j, _)| raw_time_of(*j))
                                    .unwrap_or(f64::NAN)
                            }
                        }
                    };
                    vec![scalar; n_obs]
                }

                DerivedKind::Integral {
                    integrand,
                    condition,
                    data_based,
                    uses_compartments,
                    window,
                    step,
                } => {
                    // Trapezoidal integral over [from, to] in raw-clock coordinates,
                    // restricted to the observation indices in `j_indices`.
                    //
                    // Raw time is used for the window filter, the trapezoid x-axis, and
                    // ctx.time so user expressions see the dataset TIME column value.
                    // TAFD and TAD come from per_obs_tafd/tad (shifted clock; the shift
                    // cancels because doses are on the same shifted timeline).
                    //
                    // Returns NaN when fewer than two points fall in [from, to] —
                    // correct for sparse or empty sessions; never silently inherited.
                    let eval_integral_obs_for = |j_indices: &[usize], from: f64, to: f64| -> f64 {
                        let pts: Vec<(f64, f64)> = j_indices
                            .iter()
                            .filter_map(|&j| {
                                let t_raw = raw_time_of(j);
                                if t_raw < from - 1e-12 || t_raw > to + 1e-12 {
                                    return None;
                                }
                                let row_prev: HashMap<String, f64> = prev_derived_vecs
                                    .iter()
                                    .map(|(k, v)| (k.clone(), v[j]))
                                    .collect();
                                let ctx = DerivedContext {
                                    theta,
                                    eta: &per_obs_eta_full[j],
                                    indiv_params: &per_obs_indiv[j],
                                    covariates: per_obs_cov[j],
                                    ipred: sr.ipred[j],
                                    pred: sr.pred[j],
                                    dv: subject.observations[j],
                                    time: t_raw,
                                    tafd: per_obs_tafd[j],
                                    tad: per_obs_tad[j],
                                    prev_derived: &row_prev,
                                    compartments: per_obs_cmts[j],
                                    compartment_names: model_cmt_names,
                                };
                                if condition.as_ref().map_or(false, |f| !f(&ctx)) {
                                    return None;
                                }
                                Some((t_raw, integrand(&ctx)))
                            })
                            .collect();
                        trapezoid(&pts)
                    };

                    let use_obs = *data_based || matches!(step, IntegralStep::ObsTimes);

                    // Per-session grid snapshots: covariate, lagtime, and indiv params
                    // from each session's first observation.  Only allocated for
                    // model-based integrals (`!use_obs`); stays empty — and is never
                    // indexed — when `use_obs = true`.
                    //
                    // This is the same "representative first-obs" approximation the old
                    // single-session grid used; it extends correctly per-session here.
                    let session_grid_cov: Vec<&HashMap<String, f64>> = if use_obs {
                        vec![]
                    } else {
                        session_obs
                            .iter()
                            .map(|g| {
                                g.first()
                                    .map(|&j| per_obs_cov[j])
                                    .unwrap_or(&subject.covariates)
                            })
                            .collect()
                    };
                    // Per-session representative full eta vector (BSV η + the κ of
                    // the session's first observation's occasion). Mirrors the
                    // first-obs approximation used for session_grid_cov/indiv, so a
                    // model-based integral over an IOV session uses that occasion's
                    // κ rather than κ=0 (issue #238).
                    let session_grid_eta_full: Vec<&[f64]> = if use_obs {
                        vec![]
                    } else {
                        session_obs
                            .iter()
                            .map(|g| {
                                g.first()
                                    .map(|&j| per_obs_eta_full[j].as_slice())
                                    .unwrap_or(eta_hat)
                            })
                            .collect()
                    };
                    let session_grid_indiv: Vec<HashMap<String, f64>> = if use_obs {
                        vec![]
                    } else {
                        session_obs
                            .iter()
                            .map(|g| {
                                g.first()
                                    .map(|&j| per_obs_indiv[j].clone())
                                    .unwrap_or_default()
                            })
                            .collect()
                    };

                    // Fine-grid trapezoidal integral for session `session_idx`.
                    // `from` / `to` must be in the shifted internal clock (raw + shift,
                    // clamped to session boundaries by `session_grid_window`).
                    // Nearest-IPRED and LOCF are restricted to the session's own obs
                    // so cross-session contamination can't occur.
                    // ctx.time is the shifted grid point — a known limitation: grid
                    // expressions referencing TIME see the internal clock, not raw TIME.
                    let eval_integral_grid = |from: f64, to: f64, session_idx: usize| -> f64 {
                        let grid_cov = session_grid_cov[session_idx];
                        let indiv_s = &session_grid_indiv[session_idx];
                        let grid_eta_full = session_grid_eta_full[session_idx];
                        let n_steps = match step {
                            IntegralStep::Fixed(s) => {
                                let n = ((to - from) / s).ceil() as usize + 1;
                                n.max(2)
                            }
                            _ => 501,
                        };
                        let dt = (to - from) / (n_steps - 1) as f64;
                        let grid_times: Vec<f64> =
                            (0..n_steps).map(|k| from + k as f64 * dt).collect();

                        // Pre-compute per-grid-point compartment states when the integrand
                        // references compartments[i] or named state variables. For ODE models
                        // we re-run the solver at grid points (exact); for analytical models
                        // we evaluate the superposition formula at each grid point.
                        let grid_cmt_states: Vec<Vec<f64>> = if *uses_compartments {
                            if model.n_kappa > 0 {
                                // IOV subjects: a single fixed PK snapshot (one occasion's
                                // kappa) cannot represent a dose history spanning multiple
                                // occasions — the analytical superposition / single-pass
                                // solve here would mix occasions and be silently wrong
                                // (the same reason predict_iov uses the event-driven path).
                                // Return empty so every grid point evaluates to NaN,
                                // consistent with per-obs compartment_states being empty
                                // for IOV subjects. W_DERIVED_CMT_IOV_UNSUPPORTED explains why.
                                vec![]
                            } else if let Some(ref ode) = model.ode_spec {
                                let pk_j = (model.pk_param_fn)(theta, grid_eta_full, grid_cov);
                                crate::ode::ode_dense_solve_states(
                                    ode,
                                    &pk_j.values,
                                    theta,
                                    grid_eta_full,
                                    subject,
                                    &grid_times,
                                )
                            } else if !model.analytical_init.is_empty() {
                                // Analytical model + [initial_conditions] baseline (#521):
                                // the superposition state reconstruction does not seed the
                                // baseline amount, so states would disagree with the
                                // init-aware ipred. Return empty so every grid point
                                // evaluates to NaN, consistent with per-obs compartment_states
                                // being empty for baseline models. W_DERIVED_INIT_ANALYTICAL
                                // in fit_inner tells the user why.
                                vec![]
                            } else if subject.has_resets() {
                                // Analytical model + EVID=3/4 reset: superposition is invalid
                                // across reset boundaries. Return empty so every grid point
                                // evaluates to NaN, consistent with per-obs compartment_states
                                // being empty for such subjects. W_DERIVED_CMT_RESET_ANALYTICAL
                                // in fit_inner tells the user why.
                                vec![]
                            } else if subject.has_tv_covariates() {
                                // Analytical model + TV covariates: superposition would use
                                // a single fixed PK snapshot (grid_cov) while ipred honours
                                // per-observation TV parameters — the states would be
                                // silently wrong and finite rather than NaN.  Return empty
                                // (same as the per-obs path in compute_predictions_with_states)
                                // so every grid point evaluates to NaN, consistent with
                                // W_DERIVED_CMT_TV_ANALYTICAL warning.
                                vec![]
                            } else if crate::pk::has_oral_depot_infusion(model.pk_model, subject) {
                                // Analytical oral model + zero-order input into the depot
                                // (#400): the superposition state helper models an oral
                                // infusion as a depot bypass and cannot express a depot
                                // zero-order input, so it would return silently-wrong finite
                                // amounts. Return empty so every grid point evaluates to NaN,
                                // matching the per-obs path in compute_predictions_with_states
                                // and the W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL warning.
                                vec![]
                            } else {
                                let pk_j = (model.pk_param_fn)(theta, grid_eta_full, grid_cov);
                                crate::pk::analytical_state_at_times(
                                    model.pk_model,
                                    subject,
                                    &pk_j,
                                    &grid_times,
                                )
                            }
                        } else {
                            vec![]
                        };

                        let pts: Vec<(f64, f64)> = grid_times
                            .iter()
                            .enumerate()
                            .filter_map(|(k, &t)| {
                                let tafd_k = {
                                    let fd = subject.occasion_first_dose_time(t);
                                    if fd.is_finite() {
                                        t - fd
                                    } else {
                                        f64::NAN
                                    }
                                };
                                // Same per-dose-lag TAD as the per-observation column
                                // (shared `tad_at_time`), so a `[derived]` integral over
                                // TAD agrees with the `sdtab` TAD column under IOV/TV-cov
                                // lag — not the old session-representative scalar lag.
                                let tad_k = tad_at_time(subject, t, &dose_lagtimes);
                                // Nearest IPRED from this session's observations only.
                                let nearest_ipred = session_obs[session_idx]
                                    .iter()
                                    .map(|&j| (subject.obs_times[j], sr.ipred[j]))
                                    .min_by(|&(ta, _), &(tb, _)| {
                                        (ta - t)
                                            .abs()
                                            .partial_cmp(&(tb - t).abs())
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(_, ip)| ip)
                                    .unwrap_or(f64::NAN);
                                // Session-restricted LOCF for prev_derived.
                                let grid_prev_t: HashMap<String, f64> = prev_derived_vecs
                                    .iter()
                                    .map(|(name, vals)| {
                                        let val = session_obs[session_idx]
                                            .iter()
                                            .map(|&j| (subject.obs_times[j], vals[j]))
                                            .filter(|&(obs_t, _)| obs_t <= t + 1e-12)
                                            .last()
                                            .map(|(_, v)| v)
                                            .or_else(|| {
                                                session_obs[session_idx].first().map(|&j| vals[j])
                                            })
                                            .unwrap_or(f64::NAN);
                                        (name.clone(), val)
                                    })
                                    .collect();
                                let grid_cmts: &[f64] = if *uses_compartments {
                                    grid_cmt_states.get(k).map(|v| v.as_slice()).unwrap_or(&[])
                                } else {
                                    &[]
                                };
                                let ctx = DerivedContext {
                                    theta,
                                    eta: grid_eta_full,
                                    indiv_params: indiv_s,
                                    covariates: grid_cov,
                                    ipred: nearest_ipred,
                                    pred: nearest_ipred,
                                    dv: f64::NAN,
                                    time: t,
                                    tafd: tafd_k,
                                    tad: tad_k,
                                    prev_derived: &grid_prev_t,
                                    compartments: grid_cmts,
                                    compartment_names: model_cmt_names,
                                };
                                if condition.as_ref().map_or(false, |f| !f(&ctx)) {
                                    return None;
                                }
                                Some((t, integrand(&ctx)))
                            })
                            .collect();
                        trapezoid(&pts)
                    };

                    // Translate a raw-clock [from_raw, to_raw] window into the shifted
                    // internal clock for session `s`, clamped so the grid never escapes
                    // the session's boundaries.  Returns None when the window lies
                    // entirely outside the session (grid should yield NaN).
                    //
                    // Clamping is only a no-op for the common crossover case where the
                    // EVID=4 reset occurs at raw TIME=0 (so from_raw+shift == reset).
                    // For resets at raw TIME>0 the lower clamp prevents the grid from
                    // starting before the session, and the upper clamp prevents it from
                    // crossing into the next session.
                    let session_grid_window =
                        |s: usize, from_raw: f64, to_raw: f64| -> Option<(f64, f64)> {
                            let reset_start = if s == 0 {
                                f64::NEG_INFINITY
                            } else {
                                subject.reset_times[s - 1]
                            };
                            let reset_end =
                                subject.reset_times.get(s).copied().unwrap_or(f64::INFINITY);
                            let from_sh = (from_raw + session_shift[s]).max(reset_start);
                            let to_sh = (to_raw + session_shift[s]).min(reset_end);
                            if from_sh < to_sh {
                                Some((from_sh, to_sh))
                            } else {
                                None
                            }
                        };

                    match window {
                        IntegralWindow::Explicit { from, to } => {
                            // Unified loop: single-session subjects (n_sessions=1)
                            // produce one iteration covering all obs — identical result
                            // to the old `vec![val; n_obs]` scalar path.  Multi-session
                            // subjects integrate each session independently; sessions
                            // with no obs in the window return NaN (never inherited).
                            let mut result = vec![f64::NAN; n_obs];
                            for (s, j_indices) in session_obs.iter().enumerate() {
                                if j_indices.is_empty() {
                                    continue;
                                }
                                let val = if use_obs {
                                    eval_integral_obs_for(j_indices, *from, *to)
                                } else {
                                    match session_grid_window(s, *from, *to) {
                                        Some((fs, ts)) => eval_integral_grid(fs, ts, s),
                                        None => f64::NAN,
                                    }
                                };
                                for &j in j_indices {
                                    result[j] = val;
                                }
                            }
                            result
                        }
                        IntegralWindow::Periodic { period, anchor } => {
                            // Per-observation integral whose window is aligned to the
                            // raw-clock period containing obs j.  Session restriction
                            // prevents Session 1 and Session 2 observations at the same
                            // raw TIME from contaminating each other's AUC.
                            (0..n_obs)
                                .map(|j| {
                                    let t_raw = raw_time_of(j);
                                    let n_periods = ((t_raw - anchor) / period).floor();
                                    let from_raw = anchor + n_periods * period;
                                    let to_raw = from_raw + period;
                                    let s = obs_session[j];
                                    if use_obs {
                                        eval_integral_obs_for(&session_obs[s], from_raw, to_raw)
                                    } else {
                                        match session_grid_window(s, from_raw, to_raw) {
                                            Some((fs, ts)) => eval_integral_grid(fs, ts, s),
                                            None => f64::NAN,
                                        }
                                    }
                                })
                                .collect()
                        }
                    }
                }
            };

            // Store full per-row vector so subsequent derived columns can
            // look up the correct value at each observation row index j.
            prev_derived_vecs.insert(spec.name.clone(), col_vals.clone());
            sr.extra_columns.push((spec.name.clone(), col_vals));
        }
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    let fit_start = Instant::now();
    let chain = options.method_chain();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let mut unsupported_warnings = options.unsupported_keys_warnings();
    // Surface a "no method specified, defaulting to FOCEI" notice through the
    // same channel so it reaches both stderr and FitResult.warnings.
    if let Some(w) = options.method_default_warning() {
        unsupported_warnings.push(w);
    }

    // Capture thread count before chain runs (current_num_threads() reports
    // whichever Rayon pool is active — scoped pool when threads=Some, else global).
    let n_threads_used = rayon::current_num_threads();

    // Initialise the per-iteration optimizer trace if requested.
    if options.optimizer_trace {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("/tmp/ferx_trace_{}_{}.csv", pid, ts);
        if let Err(e) = crate::estimation::trace::init(path.clone()) {
            eprintln!("[ferx] warning: could not open trace file {}: {}", path, e);
        } else {
            eprintln!("[ferx] optimizer trace → {}", path);
        }
    }

    // Reset gradient timing counters for this fit so FERX_TIME_GRADIENTS
    // readouts are per-call rather than cumulative across a long R session.
    let time_gradients = std::env::var("FERX_TIME_GRADIENTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if time_gradients {
        crate::estimation::inner_optimizer::GRADIENT_TIMINGS.reset();
    }
    if options.verbose {
        let chain_str: Vec<&str> = chain.iter().map(|m| m.label()).collect();
        // rayon::current_num_threads() reports whichever pool par_iter would use
        // from the current call — the scoped pool when options.threads is Some,
        // otherwise the global pool. So this stays accurate in both paths.
        let n_threads = rayon::current_num_threads();
        let thread_word = if n_threads == 1 { "thread" } else { "threads" };
        if !unsupported_warnings.is_empty() {
            eprintln!("--- Warnings ---");
            for w in &unsupported_warnings {
                eprintln!("  * {}", w);
            }
            eprintln!();
        }
        eprintln!(
            "Starting estimation (chain: {}) on {} {}...",
            chain_str.join(" → "),
            n_threads,
            thread_word
        );
        eprintln!(
            "  {} subjects, {} observations",
            population.subjects.len(),
            population.n_obs()
        );
        eprintln!(
            "  {} thetas, {} etas, {} sigmas",
            model.n_theta, model.n_eta, model.n_epsilon
        );
        // Report the route each method actually uses. Gradient-based estimators
        // (FOCE/FOCEI/GN) are driven by the inner-loop gradient; IMP consumes
        // the EBE Hessian built via that same route. SAEM is sampling-based, so
        // it reports its E-step kernel (MH/HMC) instead of a gradient route.
        let uses_gradient_route = chain.iter().any(|m| {
            matches!(
                m,
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::FoceGn
                    | EstimationMethod::FoceGnHybrid
                    | EstimationMethod::Imp
                    | EstimationMethod::Impmap
            )
        });
        if uses_gradient_route {
            eprintln!(
                "  gradient: {}",
                crate::estimation::inner_optimizer::gradient_route_summary(
                    model,
                    population,
                    options.gradient_method,
                )
            );
        }
        if chain.iter().any(|m| *m == EstimationMethod::Saem) {
            eprintln!(
                "  sampler:  {}",
                crate::estimation::saem::saem_sampler_summary(model, options)
            );
        }
    }

    // Model / estimation-option compatibility guards: SDE vs SAEM / GN / AD,
    // IMP chain placement, and trust-region vs IOV. Extracted into
    // `check_model_options` so `ferx check` reports the same incompatibilities;
    // here we stop at the first error to preserve fail-fast behavior and exact
    // error strings. (Per-CMT error models cannot reach the EKF path — the
    // parser rejects Form C `y[CMT=N]` readouts on SDE models — so an SDE model
    // is always single-endpoint here, which the EKF residual-variance
    // assumption in stats/likelihood.rs relies on.)
    first_error(&check_model_options(model, options))?;

    // Pre-compute n_params (uses init_params, available before chain runs).
    let fixed_mask = crate::estimation::parameterization::packed_fixed_mask(init_params);
    let n_params_pre = fixed_mask.iter().filter(|&&b| !b).count();

    // Probe NLopt algorithm availability only when global_search will actually
    // run — otherwise the CRS2-LM warning is misleading for users who never
    // requested it.
    let nlopt_missing = if options.global_search {
        probe_nlopt_algorithms()
    } else {
        Vec::new()
    };

    // Covariance step cost warning: fire before chain so user sees it
    // immediately. Use checked_mul so an absurd parameter count cannot wrap
    // and produce a bogus estimate; on overflow we still warn but suppress
    // the numeric estimate.
    let covariance_n_evals_estimated = if options.run_covariance_step && n_params_pre > 30 {
        n_params_pre.checked_mul(n_params_pre)
    } else {
        None
    };

    // Compute observation time range from the population.
    let obs_time_range: Option<(f64, f64)> = {
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for s in &population.subjects {
            for &t in &s.obs_times {
                if t < mn {
                    mn = t;
                }
                if t > mx {
                    mx = t;
                }
            }
        }
        if mn.is_finite() {
            Some((mn, mx))
        } else {
            None
        }
    };

    // Run each stage in sequence, feeding params forward.
    let n_stages = chain.len();
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = model.parse_warnings.clone();
    accumulated_warnings.extend(unsupported_warnings);
    // Data-reader warnings (W_ADDL_MISSING_II, W_IOV_OCC_MISSING) accumulated
    // by read_nonmem_csv into population.warnings.
    accumulated_warnings.extend(population.warnings.iter().cloned());

    // Inner-gradient FD-fallback notice for the gradient-driven methods: if some
    // (but not all) subjects fall outside the analytic provider's scope, surface
    // it through warnings (not just the startup banner) per the CLAUDE.md rule.
    if chain.iter().any(|m| {
        matches!(
            m,
            EstimationMethod::Foce
                | EstimationMethod::FoceI
                | EstimationMethod::FoceGn
                | EstimationMethod::FoceGnHybrid
                | EstimationMethod::Imp
                | EstimationMethod::Impmap
        )
    }) {
        if let Some(w) = crate::estimation::inner_optimizer::fd_fallback_warning(
            model,
            population,
            &init_params.theta,
        ) {
            accumulated_warnings.push(w);
        }
    }

    // Emit NLopt / covariance warnings before any work starts.
    accumulated_warnings.extend(nlopt_missing.iter().cloned());

    // Data-dependent warnings: malformed steady-state rows, EVID=3/4 resets
    // under an SDE model, and a negative typical-value lag time. Extracted into
    // `check_model_data_warnings` so `ferx check` reports the same findings;
    // message text is unchanged. Probed against `population` (not the pruned
    // copy) and `init_params`, matching the historical inline checks.
    for d in check_model_data_warnings(model, population, init_params) {
        accumulated_warnings.push(d.message);
    }
    // Experimental-feature notices (data-independent; see check_experimental_features).
    for d in check_experimental_features(model) {
        accumulated_warnings.push(d.message);
    }
    if options.run_covariance_step && n_params_pre > 30 {
        if let Some(n_evals) = covariance_n_evals_estimated {
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → {} OFV evaluations \
                 (finite-difference Hessian). This may take several minutes \
                 on complex models.",
                n_params_pre, n_evals
            ));
        } else {
            // n_params² overflowed usize — warn without the (wrapped) number.
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → n² OFV evaluations \
                 (finite-difference Hessian). Estimate exceeds usize range; \
                 expect this to be very slow.",
                n_params_pre
            ));
        }
    }

    // inits_from_nca: derive NCA-based starting values before the optimizer
    // loop, using the strategy the user selected (nca / nca_sweep / nca_ebe).
    if let Some(method) = options.inits_from_nca {
        let suggested = crate::suggest_start::inits_from_nca(model, population, method);
        stage_params = suggested.params;
        accumulated_warnings.extend(suggested.warnings);
    }

    // Warn if any subject has a non-numeric ID.  sdtab() parses subject IDs
    // as f64 and falls back to a 1-based loop index when parsing fails; the
    // fallback produces a misleading ID column that breaks downstream joins.
    // NONMEM data always uses numeric IDs, so this fires only for malformed
    // input.
    let non_numeric_ids: Vec<&str> = population
        .subjects
        .iter()
        .filter(|s| s.id.parse::<f64>().is_err())
        .map(|s| s.id.as_str())
        .collect();
    if !non_numeric_ids.is_empty() {
        accumulated_warnings.push(format!(
            "Non-numeric subject IDs detected ({} subject(s), e.g. {:?}). \
             The sdtab ID column will fall back to a 1-based loop index for \
             these subjects, which will break any downstream join by ID.",
            non_numeric_ids.len(),
            non_numeric_ids.first().unwrap_or(&""),
        ));
    }

    // Capture initial parameter values after NCA override so the stored
    // values reflect what the optimizer actually started from.  Placed here
    // rather than at the top of the function so that inits_from_nca-derived
    // values are captured correctly (init_params is never mutated; only
    // stage_params is updated by the NCA block above).
    let theta_init = stage_params.theta.clone();
    let omega_init = stage_params.omega.matrix.clone();
    let sigma_init = stage_params.sigma.values.clone();

    let mut total_iterations: usize = 0;
    let mut is_result: Option<ImportanceSamplingResult> = None;

    for (stage_idx, &method) in chain.iter().enumerate() {
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }
        let is_last = stage_idx + 1 == n_stages;
        let mut stage_opts = options.clone();
        stage_opts.method = method;
        stage_opts.methods = Vec::new();
        // Per-stage interaction flag: FOCEI=on, FOCE=off, others inherit from user options.
        match method {
            EstimationMethod::FoceI => stage_opts.interaction = true,
            EstimationMethod::Foce => stage_opts.interaction = false,
            _ => {}
        }
        // Run the covariance step on the last *estimating* stage. IMP is a
        // likelihood evaluation (NONMEM EONLY=1 equivalent), not an estimator,
        // so when IMP follows an estimator the preceding stage is effectively
        // the final estimating stage and should compute covariance / SIR.
        let is_last_estimating = is_last
            || chain[stage_idx + 1..]
                .iter()
                .all(|&m| m == EstimationMethod::Imp);
        if !is_last_estimating {
            stage_opts.run_covariance_step = false;
            stage_opts.sir = false;
        }
        // Bayesian estimation reports posterior credible intervals, not a
        // Hessian-based covariance matrix; the FD covariance / SIR steps are
        // meaningless (and wasteful) for it.
        if matches!(method, EstimationMethod::Bayes) {
            stage_opts.run_covariance_step = false;
            stage_opts.sir = false;
        }

        if options.verbose && n_stages > 1 {
            eprintln!(
                "\n── Stage {}/{}: {} ──",
                stage_idx + 1,
                n_stages,
                method.label()
            );
        }

        // IMP evaluation-only stage (`imp_eval_only`, NONMEM `IMP EONLY=1`): not an
        // estimator. Consumes the previous stage's params / EBEs / Hessians,
        // writes its result to `is_result`, and skips the params/result update at
        // the bottom of the loop so the preceding stage's `OuterResult` continues
        // to be the canonical one. The default estimating IMP path is handled by
        // the `EstimationMethod::Imp` arm of the `match method` below.
        if method == EstimationMethod::Imp && stage_opts.imp_eval_only {
            // Standalone IMP (no preceding estimator): evaluate the EBEs/Hessians
            // at the initial parameters so IMP can report the −2 log L there.
            // This synthetic stage also becomes the canonical `OuterResult` so
            // the rest of the fit (sdtab, FitResult) sees the (unchanged) params.
            if result.is_none() {
                let mu_k = crate::estimation::parameterization::compute_mu_k(
                    model,
                    &stage_params.theta,
                    stage_opts.mu_referencing,
                );
                let (eta_hats, h_matrices, _stats, kappas) =
                    crate::estimation::inner_optimizer::run_inner_loop_warm(
                        model,
                        population,
                        &stage_params,
                        stage_opts.inner_maxiter,
                        stage_opts.inner_tol,
                        None,
                        Some(&mu_k),
                        stage_opts.min_obs_for_convergence_check as usize,
                    );
                let nll = crate::estimation::outer_optimizer::pop_nll(
                    model,
                    population,
                    &stage_params,
                    &eta_hats,
                    &h_matrices,
                    &kappas,
                    stage_opts.interaction,
                );
                result = Some(crate::estimation::outer_optimizer::OuterResult {
                    params: stage_params.clone(),
                    ofv: 2.0 * nll,
                    converged: true,
                    n_iterations: 0,
                    eta_hats,
                    h_matrices,
                    kappas,
                    covariance_matrix: None,
                    warnings: Vec::new(),
                    saem_mu_ref_m_step_evals_saved: None,
                    saem_n_subjects_hmc: None,
                    ebe_convergence_warnings: 0,
                    max_unconverged_subjects: 0,
                    total_ebe_fallbacks: 0,
                    final_gradient: None,
                    sir_fallback_proposal: None,
                    impmap_trace: None,
                    bayes: None,
                    cond_dist: None,
                });
            }
            let prev = result.as_ref().expect(
                "IMP stage: prior OuterResult must exist (synthesised above when standalone)",
            );
            match crate::estimation::importance_sampling::run_importance_sampling(
                model,
                population,
                &prev.params,
                &prev.eta_hats,
                &prev.h_matrices,
                &prev.kappas,
                &stage_opts,
            ) {
                Ok(r) => {
                    // Surface a *separate* warning for any subject whose
                    // ESS-fraction collapsed to zero. These are already in
                    // `low_ess_subjects` (assuming threshold > 0), but
                    // complete proposal collapse is qualitatively distinct
                    // from merely-low ESS — each collapsed subject inflates
                    // the reported MC SE by ~1 unit (see Geweke variance
                    // fallback in `importance_sampling.rs`).
                    let collapsed: Vec<&str> = r
                        .low_ess_subjects
                        .iter()
                        .filter(|(_, f)| *f <= 0.0)
                        .map(|(id, _)| id.as_str())
                        .collect();
                    if !collapsed.is_empty() {
                        let preview = if collapsed.len() <= 5 {
                            collapsed.join(", ")
                        } else {
                            let head = collapsed[..5].join(", ");
                            format!("{} (+{} more)", head, collapsed.len() - 5)
                        };
                        let msg = format!(
                            "IMP: {} subject(s) had ESS = 0 (proposal collapse): {}. \
                             The reported MC SE is inflated by ~1 per collapsed subject; \
                             consider raising `imp_samples` or `imp_proposal_df`, \
                             or check the EBE/Hessian quality of these subjects.",
                            collapsed.len(),
                            preview
                        );
                        accumulated_warnings.push(if n_stages > 1 {
                            format!("[IMP] {}", msg)
                        } else {
                            msg
                        });
                    }
                    is_result = Some(r);
                }
                Err(e) => {
                    accumulated_warnings.push(if n_stages > 1 {
                        format!("[IMP] {}", e)
                    } else {
                        format!("IMP: {}", e)
                    });
                }
            }
            continue;
        }

        let stage_result = match method {
            EstimationMethod::Saem => {
                saem::run_saem(model, population, &stage_params, &stage_opts)?
            }
            EstimationMethod::Impmap => {
                // Warm-start the first MAP inner loop from the preceding stage's
                // EBEs when chained (e.g. [focei, impmap] / [saem, impmap]).
                let warm = result.as_ref().map(|r| r.eta_hats.as_slice());
                crate::estimation::impmap::run_impmap(
                    model,
                    population,
                    &stage_params,
                    warm,
                    &stage_opts,
                )?
            }
            EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
                crate::estimation::gauss_newton::run_foce_gn(
                    model,
                    population,
                    &stage_params,
                    &stage_opts,
                )
            }
            EstimationMethod::Imp => {
                // Estimating IMP (NONMEM `METHOD=IMP`). The evaluation-only path
                // (`imp_eval_only`) is handled by the IMP branch above and never
                // reaches here. Warm-start from the preceding stage's EBEs when
                // chained (e.g. [saem, imp]).
                let warm = result.as_ref().map(|r| r.eta_hats.as_slice());
                crate::estimation::impmap::run_imp(
                    model,
                    population,
                    &stage_params,
                    warm,
                    &stage_opts,
                )?
            }
            EstimationMethod::Bayes => {
                crate::estimation::bayes::run_bayes(model, population, &stage_params, &stage_opts)?
            }
            _ => optimize_population(model, population, &stage_params, &stage_opts),
        };

        stage_params = stage_result.params.clone();
        total_iterations += stage_result.n_iterations;
        for w in &stage_result.warnings {
            accumulated_warnings.push(if n_stages > 1 {
                format!("[{}] {}", method.label(), w)
            } else {
                w.clone()
            });
        }
        result = Some(stage_result);

        // NONMEM-comparable IMP / IMPMAP objective. The reported `OuterResult.ofv`
        // is a final FOCE *Laplace* pass (kept for cross-method AIC/BIC
        // comparability, like SAEM). NONMEM `METHOD=IMP` instead reports the
        // importance-sampling Monte-Carlo *marginal* −2 log L (the `.ext` #OBJV).
        // Evaluate that marginal at the final estimates and surface it alongside
        // on `FitResult.importance_sampling`, so callers comparing to NONMEM read
        // the matching number. Best-effort: a failure (e.g. SDE, IOV without
        // Ω_iov, n_eta = 0) leaves the field unset with a warning, never aborts.
        if is_last && matches!(method, EstimationMethod::Imp | EstimationMethod::Impmap) {
            let r = result.as_ref().expect("stage result was just set");
            let mut marg_opts = stage_opts.clone();
            // `run_importance_sampling` reads the `imp_*` knobs; for IMPMAP map the
            // `impmap_*` knobs onto them so the final eval mirrors the method's
            // own sample count / proposal df / seed.
            if method == EstimationMethod::Impmap {
                marg_opts.imp_samples = stage_opts.impmap_samples;
                marg_opts.imp_seed = stage_opts.impmap_seed;
                marg_opts.imp_low_ess_threshold = stage_opts.impmap_low_ess_threshold;
                // A Gaussian IMPMAP proposal (`impmap_proposal_df = ∞`, opt-in)
                // cannot be sampled by the finite-t IS evaluator. The marginal is
                // proposal-independent in expectation, so fall back to a finite-t
                // eval proposal (heavier tails ⇒ bounded weights). The default
                // `impmap_proposal_df = 4` passes through unchanged.
                let df = stage_opts.impmap_proposal_df;
                marg_opts.imp_proposal_df = if df.is_finite() && df >= 1.0 { df } else { 5.0 };
            }
            match crate::estimation::importance_sampling::run_importance_sampling(
                model,
                population,
                &r.params,
                &r.eta_hats,
                &r.h_matrices,
                &r.kappas,
                &marg_opts,
            ) {
                Ok(is) => is_result = Some(is),
                Err(e) => accumulated_warnings.push(if n_stages > 1 {
                    format!("[{}] marginal −2 log L eval skipped: {}", method.label(), e)
                } else {
                    format!("marginal −2 log L eval skipped: {}", e)
                }),
            }
        }
    }

    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    let mut result = result.expect("method chain must have at least one stage");
    // Overwrite with chain-aware totals
    result.n_iterations = total_iterations;
    result.warnings = accumulated_warnings;

    // Thread efficiency warnings (post-chain, uses n_threads_used captured above).
    let n_subjects = population.subjects.len();
    if n_subjects > 0 && n_threads_used > n_subjects {
        // `threads = 0` is not a valid Rayon pool size, so for n_subjects = 1
        // we still suggest a 1-thread pool.
        let suggested = n_subjects.max(1);
        result.warnings.push(format!(
            "{} threads configured but only {} subject(s) — consider threads = {} to reduce \
             scheduling overhead (no speed benefit beyond n_subjects)",
            n_threads_used, n_subjects, suggested
        ));
    }
    // SAEM-specific: MH scheduling has higher per-subject overhead than FOCE.
    // Skip when n_subjects < 2 (n_subjects/2 = 0 is meaningless and the prior
    // warning already covers the n_threads > n_subjects case).
    if chain.iter().any(|&m| m == EstimationMethod::Saem) && n_subjects >= 2 {
        let suggested = (n_subjects / 2).max(1);
        if n_threads_used > suggested {
            result.warnings.push(format!(
                "SAEM with more threads than subjects/2 may be slower due to MH scheduling \
                 overhead. Consider threads = {} for SAEM.",
                suggested
            ));
        }
    }

    // Compute per-subject diagnostics
    let mut subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        &result.kappas,
        options.interaction,
    );

    // Post-fit: compute [derived] and [output] columns, and populate per_obs_tad
    // (with individual lagtime) for the mandatory TAD column in output.rs.
    if !model.derived_exprs.is_empty() || !model.output_columns.is_empty() || model.has_lagtime() {
        compute_extra_output_columns(
            model,
            population,
            &result.params.theta,
            &result.kappas,
            &mut subjects,
        );
    }

    // Post-fit: simulation-based NPDE / NPD diagnostics (issue #260). Opt-in via
    // `[fit_options] npde_nsim`; skipped entirely when 0 so the common path pays
    // nothing. Subjects are built in population order, so the zip aligns.
    if options.npde_nsim > 0 {
        let per_subj = crate::stats::npde::compute_npde_npd(
            model,
            population,
            &result.params,
            options.npde_nsim,
            options.npde_seed,
        );
        for (sr, sn) in subjects.iter_mut().zip(per_subj) {
            sr.npde = sn.npde;
            sr.npd = sn.npd;
        }
    }

    let n_obs = population.n_obs();
    let n_params = n_params_pre;

    let ofv = result.ofv;
    let aic = ofv + 2.0 * n_params as f64;
    // BIC = OFV + k·ln(n). For TTE-only models n_obs == 0 (no Gaussian records),
    // giving ln(0) = -inf. Use total record count (Gaussian + TTE) so BIC is finite.
    #[cfg(feature = "survival")]
    let n_for_bic: usize = n_obs
        + population
            .subjects
            .iter()
            .map(|s| s.obs_records.len())
            .sum::<usize>();
    #[cfg(not(feature = "survival"))]
    let n_for_bic: usize = n_obs;
    let bic = if n_for_bic > 0 {
        ofv + n_params as f64 * (n_for_bic as f64).ln()
    } else {
        f64::NAN
    };

    // Extract SEs from covariance matrix using converged parameter values
    let (se_theta, se_omega, se_sigma, se_kappa) =
        extract_standard_errors(&result.covariance_matrix, &result.params);

    // Optional SIR step
    let mut warnings = result.warnings;

    // Warn when [derived] expressions that reference compartments[i] will
    // silently evaluate to NaN due to unsupported model/subject configurations.
    // Gate on `uses_compartments` so that a `[derived]` block with only IPRED/DV
    // integrals (no compartment references) does not emit spurious CMT warnings.
    if model.derived_exprs.iter().any(|s| s.uses_compartments) {
        // IOV (kappa) subjects: the predict_iov path does not compute compartment
        // states — they stay as vec![] so compartments[i] yields NaN.
        if result.kappas.iter().any(|ks| !ks.is_empty()) {
            warnings.push(
                "W_DERIVED_CMT_IOV_UNSUPPORTED: subjects with IOV (kappa) parameters \
                 do not have compartment states available; [derived] expressions that \
                 reference compartments[i] evaluate to NaN for those subjects."
                    .to_string(),
            );
        }
        // Analytical TV-covariate subjects: states would be computed with baseline
        // PK params while ipred uses time-varying params — inconsistency is worse
        // than NaN, so the states path returns empty for such subjects.
        if model.ode_spec.is_none() && population.subjects.iter().any(|s| s.has_tv_covariates()) {
            warnings.push(
                "W_DERIVED_CMT_TV_ANALYTICAL: analytical model with time-varying \
                 covariates — compartment states are not available for subjects \
                 with TV covariates; [derived] expressions that reference \
                 compartments[i] evaluate to NaN for those subjects."
                    .to_string(),
            );
        }
        // ODE TV-covariate subjects: states are computed via a deterministic pass
        // using first-obs PK params — approximate when CL/V/etc. vary over time.
        // ipred (from the event-driven path) is exact; only states are approximate.
        if model.ode_spec.is_some() && population.subjects.iter().any(|s| s.has_tv_covariates()) {
            warnings.push(
                "W_DERIVED_CMT_TV_ODE: ODE model with time-varying covariates — \
                 compartment states for TV-covariate subjects are approximate \
                 (first-observation PK parameters used for the deterministic state \
                 pass; ipred is exact). Use compartments[i] results with care for \
                 those subjects."
                    .to_string(),
            );
        }
        // Analytical model with EVID=3/4 resets: superposition is invalid across
        // reset boundaries. Per-obs compartment states are empty (→ NaN) and the
        // grid-integral path also returns NaN for affected sessions.
        // ODE models with resets are handled correctly (ode_dense_solve_states applies
        // the reset as a break-point); this warning is analytical-only.
        if model.ode_spec.is_none() && population.subjects.iter().any(|s| s.has_resets()) {
            warnings.push(
                "W_DERIVED_CMT_RESET_ANALYTICAL: analytical model with EVID=3/4 \
                 reset events — compartment states and compartment-based integrals \
                 are not available for subjects with resets; [derived] expressions \
                 that reference compartments[i] evaluate to NaN for those subjects. \
                 Use an ODE model if compartment states across resets are required."
                    .to_string(),
            );
        }
        // Analytical model with an [initial_conditions] baseline (#521): ipred is
        // init-aware, but the superposition state reconstruction does not seed the
        // baseline amount into the compartment vectors, so per-obs states (and the
        // grid-integral path) return empty (→ NaN) rather than report amounts that
        // disagree with ipred.
        if model.ode_spec.is_none() && !model.analytical_init.is_empty() {
            warnings.push(
                "W_DERIVED_INIT_ANALYTICAL: analytical model with an \
                 [initial_conditions] baseline — compartment states are not \
                 available for baseline models (predictions are exact); [derived] \
                 expressions that reference compartments[i] evaluate to NaN. Use an \
                 ODE model with init(...) in [odes] if compartment amounts are required."
                    .to_string(),
            );
        }
        // Analytical oral model with a zero-order input into the depot (#400):
        // the superposition state helper models an oral infusion as a depot
        // bypass, so it cannot express a depot zero-order input. ipred is exact
        // (event-driven path), but per-obs compartment states return empty (→ NaN)
        // rather than report silently-wrong amounts.
        if model.ode_spec.is_none()
            && population
                .subjects
                .iter()
                .any(|s| crate::pk::has_oral_depot_infusion(model.pk_model, s))
        {
            warnings.push(
                "W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL: analytical oral model \
                 with a zero-order input into the depot (RATE=-2 D1 / infusion into \
                 compartment 1) — compartment states are not available for those \
                 subjects (predictions are exact); [derived] expressions that \
                 reference compartments[i] evaluate to NaN for them. Use an ODE model \
                 if depot/central compartment amounts are required."
                    .to_string(),
            );
        }
    }

    // Report detected mu-referencing relationships (only when feature is enabled)
    if options.mu_referencing && !model.mu_refs.is_empty() {
        let mut names: Vec<&String> = model.mu_refs.keys().collect();
        names.sort();
        warnings.push(format!(
            "mu-ref: {}",
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // When M3 censoring is combined with non-interaction FOCE, mixing linearized
    // Gaussian residuals with non-linearized log Φ terms gives inconsistent
    // OFVs near the LOQ boundary. The FOCE dispatcher routes affected
    // subjects through FOCEI internally — surface the promotion to the user.
    if matches!(model.bloq_method, BloqMethod::M3)
        && matches!(
            options.method,
            EstimationMethod::Foce | EstimationMethod::FoceGn
        )
        && !options.interaction
        && population
            .subjects
            .iter()
            .any(|s| s.has_censored_observation())
    {
        warnings.push(
            "M3 censoring handling requires FOCEI semantics; subjects with CENS!=0 \
             rows were evaluated with η-interaction. Set method=focei explicitly \
             to silence this notice."
                .to_string(),
        );
    }
    let sir_result = if options.sir && !crate::cancel::is_cancelled(&options.cancel) {
        if let Some(ref cov) = result.covariance_matrix {
            if options.verbose {
                eprintln!("\nRunning SIR...");
            }
            match crate::estimation::sir::run_sir_core(
                model,
                population,
                &result.params,
                &result.eta_hats,
                cov,
                result.ofv,
                options,
            ) {
                Ok(sir) => Some(sir),
                Err(e) => {
                    warnings.push(format!("SIR failed: {}", e));
                    None
                }
            }
        } else {
            warnings.push(
                "SIR requested but covariance matrix is not available. \
                 Enable covariance = true in [fit_options]."
                    .to_string(),
            );
            None
        }
    } else {
        None
    };

    // SIR fallback: when the FD Hessian is non-PD and covariance_fallback = sir,
    // run SIR with the rectified |eigenvalue| proposal built inside compute_covariance.
    let sir_fallback_result = resolve_sir_fallback(
        options,
        result.covariance_matrix.is_some(),
        sir_result.is_some(),
        result.sir_fallback_proposal.as_ref(),
        model,
        population,
        &result.params,
        &result.eta_hats,
        result.ofv,
        &mut warnings,
    );

    // `final_method` reports the last *estimating* stage. An evaluation-only IMP
    // (`imp_eval_only`) doesn't produce parameters, so a chain like `[saem, imp]`
    // surfaces as `method = SAEM`. Estimating IMP (the default) does produce
    // parameters and is reported like any other estimator. The full chain is
    // preserved in `method_chain`.
    let final_method = chain
        .iter()
        .rev()
        .copied()
        .find(|&m| !(m == EstimationMethod::Imp && options.imp_eval_only))
        .unwrap_or(*chain.last().expect("chain non-empty"));
    let grad_inner =
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, model);
    let grad_outer = crate::build_info::gradient_method_outer(
        &crate::build_info::BUILD_INFO,
        final_method,
        options.optimizer,
        model,
    );

    // Flush and close the trace file; capture path for FitResult.
    let trace_path = crate::estimation::trace::finish();

    // Shrinkage
    let shrinkage_eta = compute_eta_shrinkage(&subjects, &result.params.omega.matrix);
    let shrinkage_eps = compute_eps_shrinkage(&subjects);
    let (shrinkage_kappa, shrinkage_kappa_by_occ) =
        if let Some(ref omega_iov) = result.params.omega_iov {
            (
                compute_kappa_shrinkage(&result.kappas, &omega_iov.matrix),
                compute_kappa_shrinkage_by_occ(&result.kappas, &omega_iov.matrix),
            )
        } else {
            (Vec::new(), Vec::new())
        };

    if let Some(w) = eps_shrinkage_warning(shrinkage_eps) {
        warnings.push(w);
    }

    let (iwres_lag1_r, dw_statistic) = iwres_autocorrelation(&subjects);

    // Covariance status. Bayesian fits report posterior credible intervals
    // instead of a Hessian covariance, so the covariance step is never
    // "requested" for them (reporting it as FAILED would be misleading).
    let covariance_status = resolve_covariance_status(
        options.run_covariance_step && result.bayes.is_none(),
        result.covariance_matrix.is_some(),
        sir_fallback_result.is_some(),
    );

    let wall_time_secs = fit_start.elapsed().as_secs_f64();

    let (cov_eigenvalues, cov_condition_number) =
        cov_diagnostics(result.covariance_matrix.as_ref());

    // Derive per-eta lognormal flags from mu_refs, keyed by eta name.
    // Etas absent from mu_refs (conditional / complex / logit) are treated as
    // additive (false) and a warning is added when they participate in a block.
    let eta_log_transformed: Vec<bool> = result
        .params
        .omega
        .eta_names
        .iter()
        .map(|name| {
            model
                .mu_refs
                .get(name)
                .map(|r| r.log_transformed)
                .unwrap_or(false)
        })
        .collect();

    let omega_param_corr = compute_param_corr(
        &result.params.omega.matrix,
        &eta_log_transformed,
        &result.params.omega.eta_names,
        "omega_param_corr",
        &mut warnings,
    );

    let omega_iov_param_corr = result.params.omega_iov.as_ref().and_then(|iov| {
        let kappa_log: Vec<bool> = model
            .kappa_names
            .iter()
            .map(|name| {
                model
                    .kappa_mu_refs
                    .get(name)
                    .map(|r| r.log_transformed)
                    .unwrap_or(false)
            })
            .collect();
        compute_param_corr(
            &iov.matrix,
            &kappa_log,
            &model.kappa_names,
            "omega_iov_param_corr",
            &mut warnings,
        )
    });

    // DW autocorrelation warnings
    if dw_statistic.is_finite() {
        if dw_statistic < 1.5 {
            let mut msg = format!(
                "Positive IWRES autocorrelation detected (Durbin-Watson = {:.2}). \
                Structural model may be missing dynamics. Consider a transit \
                absorption model, additional compartment, or IOV on ka/F.",
                dw_statistic
            );
            if model.ode_spec.is_some() {
                msg.push_str(" For ODE models, SDE process noise may also help.");
            }
            warnings.push(msg);
        } else if dw_statistic > 2.5 {
            warnings.push(format!(
                "Negative IWRES autocorrelation detected (Durbin-Watson = {:.2}). \
                Possible over-parameterization or misspecified error model.",
                dw_statistic
            ));
        }
    }

    // Reported outer optimizer. For the FOCE/FOCEI path with the default `auto`,
    // surface the concrete optimizer `auto` resolved to (e.g. `auto (nlopt_lbfgs)`)
    // so the output records what actually ran (#490).
    let optimizer_label: String = match final_method {
        EstimationMethod::Saem => "saem".to_string(),
        EstimationMethod::FoceGn => "gn".to_string(),
        EstimationMethod::FoceGnHybrid => "gn".to_string(),
        // IMP/IMPMAP never run the outer optimizer — their M-step uses an
        // internal BOBYQA regardless of `options.optimizer`, so report that
        // rather than a setting that had no effect.
        EstimationMethod::Impmap => "impmap-bobyqa".to_string(),
        EstimationMethod::Imp => "imp-bobyqa".to_string(),
        _ => {
            if options.optimizer == Optimizer::Auto {
                format!("auto ({})", options.optimizer.resolve_auto(model).label())
            } else {
                options.optimizer.label().to_string()
            }
        }
    };

    let fit_result = FitResult {
        method: final_method,
        method_chain: chain.clone(),
        converged: result.converged,
        ofv,
        aic,
        bic,
        theta: result.params.theta.clone(),
        theta_names: result.params.theta_names.clone(),
        eta_names: result.params.omega.eta_names.clone(),
        omega: result.params.omega.matrix.clone(),
        sigma: result.params.sigma.values.clone(),
        sigma_names: result.params.sigma.names.clone(),
        error_model: model.error_model,
        covariance_matrix: result.covariance_matrix,
        se_theta,
        se_omega,
        se_sigma,
        theta_fixed: result.params.theta_fixed.clone(),
        omega_fixed: result.params.omega_fixed.clone(),
        sigma_fixed: result.params.sigma_fixed.clone(),
        omega_init_as_sd: model.omega_init_as_sd.clone(),
        sigma_init_as_sd: model.sigma_init_as_sd.clone(),
        subjects,
        n_obs,
        n_subjects: population.subjects.len(),
        n_parameters: n_params,
        n_iterations: result.n_iterations,
        interaction: options.interaction,
        warnings,
        warnings_structured: vec![],
        // If the normal SIR ran, use that; otherwise use the fallback result.
        sir_ci_theta: sir_result
            .as_ref()
            .or(sir_fallback_result.as_ref())
            .map(|s| s.ci_theta.clone()),
        sir_ci_omega: sir_result
            .as_ref()
            .or(sir_fallback_result.as_ref())
            .map(|s| s.ci_omega.clone()),
        sir_ci_sigma: sir_result
            .as_ref()
            .or(sir_fallback_result.as_ref())
            .map(|s| s.ci_sigma.clone()),
        sir_ess: sir_result
            .as_ref()
            .or(sir_fallback_result.as_ref())
            .map(|s| s.effective_sample_size),
        sir_resamples_packed: sir_result
            .as_ref()
            .or(sir_fallback_result.as_ref())
            .and_then(|s| s.resamples_packed.clone()),
        importance_sampling: is_result,
        impmap_trace: result.impmap_trace.clone(),
        bayes: result.bayes.clone(),
        omega_iov: result.params.omega_iov.as_ref().map(|m| m.matrix.clone()),
        kappa_names: model.kappa_names.clone(),
        kappa_fixed: result.params.kappa_fixed.clone(),
        kappa_init_as_sd: model.kappa_init_as_sd.clone(),
        se_kappa,
        shrinkage_kappa,
        shrinkage_kappa_by_occ,
        ebe_kappas: result.kappas.clone(),
        saem_mu_ref_m_step_evals_saved: result.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: result.saem_n_subjects_hmc,
        gradient_method_inner: grad_inner.as_str().to_string(),
        gradient_method_outer: grad_outer.as_str().to_string(),
        uses_ode_solver: model.is_ode_based(),
        uses_sde: model.is_sde(),
        n_threads_used,
        nlopt_missing_algorithms: nlopt_missing,
        covariance_n_evals_estimated,
        trace_path,
        ebe_convergence_warnings: result.ebe_convergence_warnings,
        max_unconverged_subjects: result.max_unconverged_subjects,
        total_ebe_fallbacks: result.total_ebe_fallbacks,
        covariance_status,
        shrinkage_eta,
        cond_dist: result.cond_dist.clone(),
        shrinkage_eps,
        iwres_lag1_r,
        dw_statistic,
        wall_time_secs,
        model_name: model.name.clone(),
        ferx_version: env!("CARGO_PKG_VERSION").to_string(),
        eta_param_info: model.eta_param_info.clone(),
        theta_transform: model.theta_transform.clone(),
        sigma_types: model
            .error_spec
            .sigma_types(result.params.sigma.values.len()),
        cov_eigenvalues,
        cov_condition_number,
        eta_log_transformed,
        omega_param_corr,
        omega_iov_param_corr,
        // Path/hash fields stay None at this layer; `fit_from_files` and the
        // CLI populate them after a successful fit. In-memory `fit()` callers
        // don't have meaningful paths.
        model_path: None,
        data_path: None,
        model_hash: None,
        data_hash: None,
        model_text: None,
        theta_init,
        omega_init,
        sigma_init,
        obs_time_range,
        final_gradient: result.final_gradient.clone(),
        optimizer: optimizer_label,
        n_starts: options.n_starts,
        multi_start_seed: options.multi_start_seed,
        saem_seed: options.saem_seed,
        sir_seed: options.sir_seed,
        imp_seed: options.imp_seed,
        // Record the *resolved* NPDE seed (default included) so the diagnostic
        // is reproducible from the output; `None` when NPDE did not run.
        npde_seed: if options.npde_nsim > 0 {
            Some(crate::stats::npde::effective_seed(options.npde_seed))
        } else {
            None
        },
        bloq_method: model.bloq_method.label().to_string(),
        outer_maxiter: options.outer_maxiter,
        outer_gtol: options.outer_gtol,
        inits_from_nca: options.inits_from_nca.map(|m| {
            use crate::suggest_start::NcaInit;
            match m {
                NcaInit::Nca => "nca",
                NcaInit::Sweep => "nca_sweep",
                NcaInit::Ebe => "nca_ebe",
            }
            .to_string()
        }),
        covariate_names: population.covariate_names.clone(),
        input_columns: population.input_columns.clone(),
        #[cfg(feature = "nn")]
        neural_networks: build_neural_network_infos(model),
        // Populated by the file-based entry points (`fit_from_files`,
        // `run_model_with_data`) when the model declares a `[covariates]`
        // block; the in-memory `fit()` path has no raw rows to echo.
        covariate_table: None,
        exclusions: population.exclusions.clone(),
    };

    if time_gradients {
        let (an_c, an_n, fd_c, fd_n, jac_an_c, jac_an_n, jac_fd_c, jac_fd_n) =
            crate::estimation::inner_optimizer::GRADIENT_TIMINGS.snapshot();
        let ms = |n: u64| (n as f64) / 1_000_000.0;
        let avg_us = |n: u64, c: u64| {
            if c == 0 {
                0.0
            } else {
                (n as f64) / (c as f64) / 1_000.0
            }
        };
        eprintln!("--- Gradient timings (FERX_TIME_GRADIENTS=1) ---");
        eprintln!(
            "  BFGS (analytic): {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            an_c,
            ms(an_n),
            avg_us(an_n, an_c)
        );
        eprintln!(
            "  BFGS (FD):       {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            fd_c,
            ms(fd_n),
            avg_us(fd_n, fd_c)
        );
        eprintln!(
            "  Jac  (analytic): {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_an_c,
            ms(jac_an_n),
            avg_us(jac_an_n, jac_an_c)
        );
        eprintln!(
            "  Jac  (FD):       {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_fd_c,
            ms(jac_fd_n),
            avg_us(jac_fd_n, jac_fd_c)
        );
    }

    Ok(fit_result)
}

/// Eigenvalues and condition number of the correlation matrix of free
/// (non-fixed) parameters.  Fixed parameters have zero diagonal in the
/// covariance matrix and are excluded so that the correlation scaling does not
/// divide by zero and the condition number reflects only the identifiable
/// parameter space.
///
/// Returns `(None, None)` when `cov` is `None` or fewer than two free
/// parameters exist (after excluding parameters whose diagonal entry is
/// `<= 0`).  Parameters with non-positive diagonals are treated as fixed and
/// silently excluded; the remaining free subblock is used for the computation.
/// Threshold below which an off-diagonal omega/kappa entry is treated as
/// structurally zero for correlation reporting.  Matches the threshold used
/// in `io/output.rs` when emitting the `correlation:` field.
const OFFDIAG_EPS: f64 = 1e-15;

/// Compute a parameter-level correlation matrix from an omega/kappa matrix.
///
/// For lognormal pairs uses `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`.
/// For additive pairs uses `ω_ij/√(ω_ii·ω_jj)` (eta-level).
/// Mixed pairs fall back to eta-level and append a warning.
/// Returns `None` when the matrix is diagonal (no off-diagonals above
/// `OFFDIAG_EPS`).
fn compute_param_corr(
    omega: &DMatrix<f64>,
    log_transformed: &[bool],
    names: &[String],
    warn_prefix: &str,
    warnings: &mut Vec<String>,
) -> Option<DMatrix<f64>> {
    let n = omega.nrows();
    debug_assert_eq!(
        log_transformed.len(),
        n,
        "log_transformed must be parallel to omega diagonal (got {} for n={})",
        log_transformed.len(),
        n,
    );
    debug_assert_eq!(
        names.len(),
        n,
        "names must be parallel to omega diagonal (got {} for n={})",
        names.len(),
        n,
    );
    let has_offdiag = (0..n).any(|i| (0..i).any(|j| omega[(i, j)].abs() > OFFDIAG_EPS));
    if !has_offdiag {
        return None;
    }
    let mut corr = DMatrix::identity(n, n);
    for i in 0..n {
        for j in 0..i {
            let cov = omega[(i, j)];
            if cov.abs() <= OFFDIAG_EPS {
                continue;
            }
            let w_ii = omega[(i, i)];
            let w_jj = omega[(j, j)];
            let lt_i = *log_transformed.get(i).unwrap_or(&false);
            let lt_j = *log_transformed.get(j).unwrap_or(&false);
            let c = if lt_i && lt_j {
                let num = cov.exp() - 1.0;
                let den = ((w_ii.exp() - 1.0) * (w_jj.exp() - 1.0)).sqrt();
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            } else if !lt_i && !lt_j {
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            } else {
                let name_i = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let name_j = names.get(j).map(|s| s.as_str()).unwrap_or("?");
                warnings.push(format!(
                    "{}: {} × {} have mixed lognormal/additive parameterizations; \
                     falling back to eta-level correlation",
                    warn_prefix, name_i, name_j
                ));
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            };
            corr[(i, j)] = c;
            corr[(j, i)] = c;
        }
    }
    Some(corr)
}

/// Resolve the reported [`CovarianceStatus`] from the three signals that
/// determine it: whether the covariance step was requested, whether it produced
/// a covariance matrix, and whether the SIR fallback (`covariance_fallback =
/// sir`) produced a result. Pulled out of `fit()` so the precedence — a real
/// covariance always wins over a fallback, which wins over a plain failure — is
/// unit-testable without driving a full fit to a non-PD Hessian.
fn resolve_covariance_status(
    run_covariance_step: bool,
    has_covariance_matrix: bool,
    has_sir_fallback: bool,
) -> CovarianceStatus {
    if !run_covariance_step {
        CovarianceStatus::NotRequested
    } else if has_covariance_matrix {
        CovarianceStatus::Computed
    } else if has_sir_fallback {
        CovarianceStatus::SirFallback
    } else {
        CovarianceStatus::Failed
    }
}

/// Pure gate for the non-PD-Hessian SIR fallback: should it run? It fires only
/// when the user opted in (`covariance_fallback = sir`), the FD-Hessian
/// covariance did **not** succeed (`!has_covariance_matrix`), a normal
/// `sir = true` run did **not** already produce intervals (`!normal_sir_ran`),
/// and `compute_covariance` actually handed back a fallback proposal
/// (`has_fallback_proposal`). Split out of [`resolve_sir_fallback`] so the
/// decision is unit-testable without driving a fit to a non-PD Hessian (#264).
fn should_run_sir_fallback(
    fallback_is_sir: bool,
    has_covariance_matrix: bool,
    normal_sir_ran: bool,
    has_fallback_proposal: bool,
) -> bool {
    fallback_is_sir && !has_covariance_matrix && !normal_sir_ran && has_fallback_proposal
}

/// Run the non-PD-Hessian SIR fallback when [`should_run_sir_fallback`] permits.
///
/// Returns `Some(SirResult)` when the fallback fired and SIR succeeded; `None`
/// when the gate declined, the run was cancelled, or SIR itself failed (the
/// failure case pushes a `"SIR fallback failed: …"` warning). Extracted from
/// `fit_inner` so the gate → `run_sir_core` → warning wiring is exercised by a
/// unit test with a controlled (tame) proposal, rather than relying on a real
/// non-PD fit — which the optimizer's fixed warmup budget cannot reach and a
/// degenerate fixture cannot reliably survive in SIR (#264).
#[allow(clippy::too_many_arguments)]
fn resolve_sir_fallback(
    options: &FitOptions,
    has_covariance_matrix: bool,
    normal_sir_ran: bool,
    fallback_proposal: Option<&DMatrix<f64>>,
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    ofv: f64,
    warnings: &mut Vec<String>,
) -> Option<crate::estimation::sir::SirResult> {
    if crate::cancel::is_cancelled(&options.cancel) {
        return None;
    }
    if !should_run_sir_fallback(
        options.covariance_fallback == CovarianceFallback::Sir,
        has_covariance_matrix,
        normal_sir_ran,
        fallback_proposal.is_some(),
    ) {
        return None;
    }
    let proposal =
        fallback_proposal.expect("should_run_sir_fallback guarantees a proposal is present");
    if options.verbose {
        eprintln!("\nRunning SIR fallback (non-PD Hessian)...");
    }
    match crate::estimation::sir::run_sir_core(
        model, population, params, eta_hats, proposal, ofv, options,
    ) {
        Ok(sir) => Some(sir),
        Err(e) => {
            warnings.push(format!("SIR fallback failed: {}", e));
            None
        }
    }
}

fn cov_diagnostics(cov: Option<&DMatrix<f64>>) -> (Option<Vec<f64>>, Option<f64>) {
    let cov = match cov {
        Some(m) => m,
        None => return (None, None),
    };
    let n = cov.nrows();
    let free: Vec<usize> = (0..n).filter(|&i| cov[(i, i)] > 0.0).collect();
    if free.len() < 2 {
        return (None, None);
    }
    let sub = DMatrix::from_fn(free.len(), free.len(), |a, b| cov[(free[a], free[b])]);
    let std_devs: Vec<f64> = (0..free.len()).map(|a| sub[(a, a)].sqrt()).collect();
    let cor = DMatrix::from_fn(free.len(), free.len(), |a, b| {
        sub[(a, b)] / (std_devs[a] * std_devs[b])
    });
    let eig = cor.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let min_ev = eigenvalues.last().copied().unwrap_or(0.0);
    let max_ev = eigenvalues.first().copied().unwrap_or(0.0);
    let condition_number = if min_ev > 1e-10 {
        max_ev / min_ev
    } else {
        f64::INFINITY
    };
    (Some(eigenvalues), Some(condition_number))
}

/// Compute per-subject diagnostics (IPRED, PRED, IWRES, CWRES)
fn compute_subject_results(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];
            let kappas: &[DVector<f64>] = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };

            // Individual predictions: f(eta_hat), with occasion-specific kappas for IOV.
            // Uses the continuous per-occasion-aware prediction (issue #104) for IOV
            // and the TV-aware dispatcher for everyone else — so the sdtab IPRED/IWRES
            // match the IPRED that drove the FOCEI marginal at fit time.
            //
            // Previously this branch called `model_preds` with a single per-subject
            // `pk_params_ind` from `subject.covariates`, which on TV-covariate data
            // silently took the **non-TV** dose-superposition path while the OFV
            // was being computed on the event-driven path that honours per-event
            // covariate snapshots. Result: sdtab IPRED collapsed to ~0 in the
            // terminal phase for subjects with even mild TV covariates, IWRES
            // exploded, and the EPS-shrinkage warning fired even when the actual
            // fit (and the inner-loop EBE) were fine. Caught on the jasmine peds
            // vancomycin testdata — see `[[focei-laplace-not-sheiner-beal]]`.
            // For IOV subjects: ipred via predict_iov; compartment states are not
            // yet supported on the IOV path (tracked as follow-up), so they stay empty.
            // For all other subjects: compute_predictions_with_states returns both ipred
            // and the per-obs compartment state vector in one pass.
            let (ipred, compartment_states) = if !kappas.is_empty() {
                let kappa_slices: Vec<Vec<f64>> =
                    kappas.iter().map(|k| k.as_slice().to_vec()).collect();
                let iov_ipred = crate::pk::predict_iov(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                    &kappa_slices,
                );
                (iov_ipred, vec![])
            } else {
                crate::pk::compute_predictions_with_states(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                )
            };

            // Population predictions: f(eta = 0, kappa = 0). Routes through the
            // TV-cov-aware predictor (the same path `predict()` uses), so the sdtab
            // PRED column honours time-varying covariates, EVID=3/4 resets, and the
            // FREM prediction override — and stays consistent with the public
            // `predict()` output (#456).
            let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
            let pred =
                crate::pk::compute_predictions_with_tv(model, subject, &params.theta, &zero_eta);

            // Per-observation custom residual magnitude (#484): η-independent
            // (θ/covariate/TIME only), so build it once and feed both the IWRES
            // and CWRES diagnostics so they match the magnitude-aware OFV.
            let ruv_mult = model.ruv_obs_mult(subject, &params.theta);

            // IWRES (NaN on censored rows — see compute_cwres for CWRES handling).
            let mut iwres = compute_iwres_with_correlations(
                &subject.observations,
                &ipred,
                &subject.obs_cmts,
                &model.error_spec,
                &params.sigma.values,
                &model.residual_correlations,
                ruv_mult.as_deref(),
            );
            // IIV on residual error (#409): the individual residual SD is scaled
            // by exp(η̂_ruv), so IWRES = (y−f)/(SD·exp(η̂_ruv)) = base / exp(η̂_ruv).
            // FREM covariate rows have no PK residual; their IWRES is left as-is.
            let ruv_sd = model.residual_var_scale(eta.as_slice()).sqrt();
            if ruv_sd != 1.0 {
                for (j, w) in iwres.iter_mut().enumerate() {
                    if subject.fremtype.get(j).copied().unwrap_or(0) == 0 {
                        *w /= ruv_sd;
                    }
                }
            }
            for (j, c) in subject.cens.iter().enumerate() {
                if *c != 0 {
                    iwres[j] = f64::NAN;
                }
            }

            // CWRES
            let frem_r_override = build_frem_r_override(
                model.frem_config.as_ref(),
                &subject.fremtype,
                &params.sigma.values,
            );
            let cwres = compute_cwres(
                subject,
                &ipred,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                &model.error_spec,
                &model.residual_correlations,
                frem_r_override.as_deref(),
                model.residual_error_eta,
                ruv_mult.as_deref(),
            );

            // OFV contribution
            let ofv_i = if !kappas.is_empty() {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when kappas non-empty");
                foce_subject_nll_iov(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                    kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                )
            };

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                // Filled post-fit (only when npde_nsim > 0); see compute_npde_npd.
                npde: vec![],
                npd: vec![],
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
                n_obs: subject.observations.len(),
                extra_columns: vec![],
                per_obs_tad: vec![],
                compartment_states,
            }
        })
        .collect()
}

/// Kappa shrinkage pooled across all subject-occasion pairs.
///
/// `1 - sqrt(mean(κ̂²)) / sqrt(omega_iov_kk)` for each kappa k, where the mean
/// runs over every (subject, occasion) pair.  Returns NaN for a given kappa when
/// the corresponding diagonal of `omega_iov` is non-positive or when fewer than
/// two (subject, occasion) observations are available.
pub(crate) fn compute_kappa_shrinkage(
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_iov: &DMatrix<f64>,
) -> Vec<f64> {
    let n_kappa = omega_iov.nrows();
    if n_kappa == 0 {
        return vec![];
    }
    // Flatten all per-subject per-occasion kappa vectors into one iterator.
    let all_kappas: Vec<&DVector<f64>> = kappas_per_subject
        .iter()
        .flat_map(|occ_kappas| occ_kappas.iter())
        .collect();
    let n = all_kappas.len();
    if n < 2 {
        return vec![f64::NAN; n_kappa];
    }
    (0..n_kappa)
        .map(|k| {
            let var = omega_iov[(k, k)];
            if var <= 0.0 {
                return f64::NAN;
            }
            let ms = all_kappas.iter().map(|kv| kv[k].powi(2)).sum::<f64>() / n as f64;
            1.0 - ms.sqrt() / var.sqrt()
        })
        .collect()
}

/// Kappa shrinkage broken out by occasion index.
///
/// Returns `shrinkage_by_occ[occ_idx][kappa_idx]` where `occ_idx` is the
/// **0-based position within each subject's own occasion list** — i.e. the
/// order in which distinct OCC values were first encountered in that subject's
/// rows (matching `split_obs_by_occasion`).
///
/// **Important limitation for unbalanced designs:** `occ_idx` is a position
/// index, *not* the raw OCC column value.  When subjects have different OCC
/// sequences (e.g., a late-entry subject whose data begins at OCC 2), their
/// position 0 maps to OCC 2 while other subjects' position 0 maps to OCC 1.
/// Pooling across position 0 then mixes kappas from different occasions.
/// For unbalanced designs use the pooled `shrinkage_kappa` instead, and
/// interpret per-occasion values only when the OCC column is aligned across
/// all subjects.
///
/// Returns an empty outer vec when fewer than two distinct occasions are present
/// or no kappa parameters exist.
pub(crate) fn compute_kappa_shrinkage_by_occ(
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_iov: &DMatrix<f64>,
) -> Vec<Vec<f64>> {
    let n_kappa = omega_iov.nrows();
    if n_kappa == 0 {
        return vec![];
    }
    // Determine max number of occasions across subjects.
    let n_occ = kappas_per_subject
        .iter()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);
    if n_occ < 2 {
        return vec![];
    }
    (0..n_occ)
        .map(|occ_idx| {
            let occ_kappas: Vec<&DVector<f64>> = kappas_per_subject
                .iter()
                .filter_map(|occ_vecs| occ_vecs.get(occ_idx))
                .collect();
            let n = occ_kappas.len();
            (0..n_kappa)
                .map(|k| {
                    let var = omega_iov[(k, k)];
                    if var <= 0.0 || n < 2 {
                        return f64::NAN;
                    }
                    let ms = occ_kappas.iter().map(|kv| kv[k].powi(2)).sum::<f64>() / n as f64;
                    1.0 - ms.sqrt() / var.sqrt()
                })
                .collect()
        })
        .collect()
}

/// ETA shrinkage: `1 - sqrt(mean(eta_hat_k^2)) / sqrt(omega_kk)` for each random effect k.
///
/// Uses the uncentered second moment with `n` divisor (NONMEM / PsN / Monolix
/// convention), reflecting the population assumption that `E[eta_k] = 0`. This
/// differs from the centered, unbiased sample variance (n-1 divisor) — for small
/// `n` the unbiased form inflates SD by sqrt(n/(n-1)) and routinely produces
/// spurious negative shrinkage even on well-fit models.
pub(crate) fn compute_eta_shrinkage(subjects: &[SubjectResult], omega: &DMatrix<f64>) -> Vec<f64> {
    let n_eta = omega.nrows();
    let n_subj = subjects.len();
    if n_subj < 2 || n_eta == 0 {
        return vec![f64::NAN; n_eta];
    }
    (0..n_eta)
        .map(|k| {
            let omega_var = omega[(k, k)];
            if omega_var <= 0.0 {
                return f64::NAN;
            }
            let omega_sd = omega_var.sqrt();
            let ms = subjects.iter().map(|s| s.eta[k].powi(2)).sum::<f64>() / n_subj as f64;
            1.0 - ms.sqrt() / omega_sd
        })
        .collect()
}

/// EPS shrinkage: `1 - sqrt(mean(IWRES^2))` across all valid (non-NaN) residuals.
///
/// IWRES has model-imposed mean 0 and variance 1, so the uncentered second
/// moment with `n` divisor is the natural estimator (matches NONMEM).
pub(crate) fn compute_eps_shrinkage(subjects: &[SubjectResult]) -> f64 {
    let vals: Vec<f64> = subjects
        .iter()
        .flat_map(|s| s.iwres.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let ms = vals.iter().map(|v| v.powi(2)).sum::<f64>() / n as f64;
    1.0 - ms.sqrt()
}

/// Threshold below which negative `shrinkage_eps` triggers a warning.
///
/// Small negative values are normal sampling noise around 0 on well-fit models
/// (the NONMEM uncentered estimator has a small downward bias when the sample
/// mean of IWRES is non-zero). Past this threshold the residual error model
/// genuinely fails to absorb the residuals at the EBE etas and the user should
/// see it.
const EPS_SHRINKAGE_WARN_THRESHOLD: f64 = -0.05;

/// Build the user-facing warning for notably-negative EPS shrinkage, or
/// `None` if the value is finite and above the threshold (or NaN).
pub(crate) fn eps_shrinkage_warning(shrinkage_eps: f64) -> Option<String> {
    if !shrinkage_eps.is_finite() || shrinkage_eps >= EPS_SHRINKAGE_WARN_THRESHOLD {
        return None;
    }
    Some(format!(
        "EPS shrinkage is notably negative ({:.1}%): mean(IWRES^2) > 1, \
         which means the residual error model does not absorb the residuals \
         at the final EBE etas. Common causes: SAEM converged to a local \
         optimum with under-fit sigma (try `method = [saem, focei]` to polish \
         with FOCEI, or different starts); model misspecification on a subset \
         of subjects; sigma at a bound. Inspect the IWRES distribution in the \
         sdtab.",
        100.0 * shrinkage_eps
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_subject(eta: Vec<f64>, iwres: Vec<f64>) -> SubjectResult {
        let n = iwres.len();
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::from_vec(eta),
            ipred: vec![0.0; n],
            pred: vec![0.0; n],
            iwres,
            cwres: vec![0.0; n],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![0; n],
            n_obs: n,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
        }
    }

    #[test]
    fn test_eta_shrinkage_zero_when_eta_rms_matches_omega_sd() {
        // NONMEM convention: shrinkage = 1 - sqrt(mean(eta^2)) / sqrt(omega).
        // omega = 1.0, eta = [+1, -1] => mean(eta^2) = 1 => shrinkage = 0.
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert_eq!(sh.len(), 1);
        assert!(
            (sh[0]).abs() < 1e-10,
            "expected ~0 shrinkage, got {}",
            sh[0]
        );
    }

    #[test]
    fn test_eta_shrinkage_positive_when_etas_shrunk() {
        // Etas close to zero → shrinkage > 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects: Vec<SubjectResult> = (0..10)
            .map(|_| make_subject(vec![0.01], vec![0.0]))
            .collect();
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0] > 0.5, "expected high shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_uses_n_not_n_minus_1_divisor() {
        // Regression: with the old (centered, n-1) estimator, eta = [+a, -a] gave
        // SD = a*sqrt(2), so omega = 2.0 was needed to land at shrinkage 0.
        // With the NONMEM (uncentered, n) estimator, mean(eta^2) = a^2, so the
        // same eta values + omega = 2.0 must now give a clearly positive value:
        // shrinkage = 1 - 1/sqrt(2) ≈ 0.293.
        let omega = DMatrix::from_diagonal_element(1, 1, 2.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        let expected = 1.0 - 1.0 / 2.0_f64.sqrt();
        assert!(
            (sh[0] - expected).abs() < 1e-10,
            "expected {}, got {}",
            expected,
            sh[0]
        );
    }

    #[test]
    fn test_eta_shrinkage_nan_when_omega_zero() {
        let omega = DMatrix::zeros(1, 1);
        let subjects = vec![
            make_subject(vec![0.1], vec![0.0]),
            make_subject(vec![-0.1], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN when omega=0");
    }

    #[test]
    fn test_eta_shrinkage_nan_when_fewer_than_2_subjects() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![make_subject(vec![0.5], vec![0.0])];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN with only 1 subject");
    }

    #[test]
    fn test_eps_shrinkage_near_zero_for_unit_normal_iwres() {
        // NONMEM convention: shrinkage = 1 - sqrt(mean(IWRES^2)).
        // IWRES = [+1, -1] => mean(IWRES^2) = 1 => shrinkage = 0.
        let subjects = vec![
            make_subject(vec![0.0], vec![1.0]),
            make_subject(vec![0.0], vec![-1.0]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "expected ~0 eps shrinkage, got {}", sh);
    }

    #[test]
    fn test_eps_shrinkage_nan_for_fewer_than_2_residuals() {
        let subjects = vec![make_subject(vec![0.0], vec![0.5])];
        assert!(compute_eps_shrinkage(&subjects).is_nan());
    }

    #[test]
    fn test_eps_shrinkage_negative_when_iwres_inflated() {
        // IWRES with mean(IWRES^2) > 1 must produce a negative value, not be
        // clamped. Matches the SAEM repro on `nmdata_20230216_1.csv` where
        // mean(IWRES^2) ~ 2.45 -> shrinkage ~ -0.566.
        let subjects = vec![
            make_subject(vec![0.0], vec![2.0]),
            make_subject(vec![0.0], vec![-2.0]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!(sh < 0.0, "expected negative shrinkage, got {}", sh);
        assert!((sh - (1.0 - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eps_shrinkage_warning_emits_below_threshold() {
        let w = eps_shrinkage_warning(-0.10).expect("expected warning");
        assert!(w.contains("mean(IWRES^2) > 1"));
        assert!(w.contains("-10.0%"));
    }

    #[test]
    fn test_eps_shrinkage_warning_silent_above_threshold() {
        // Tiny negatives are noise — no warning.
        assert!(eps_shrinkage_warning(-0.01).is_none());
        // Positive shrinkage — no warning.
        assert!(eps_shrinkage_warning(0.20).is_none());
        // Right at the boundary — no warning (uses `<`).
        assert!(eps_shrinkage_warning(-0.05).is_none());
        // NaN — no warning.
        assert!(eps_shrinkage_warning(f64::NAN).is_none());
    }

    // ── kappa shrinkage ──────────────────────────────────────────────────────

    fn make_kappas(vals: Vec<Vec<f64>>) -> Vec<Vec<DVector<f64>>> {
        // vals[subj_idx][occ_idx] = single-kappa value
        vals.into_iter()
            .map(|occ_vals| {
                occ_vals
                    .into_iter()
                    .map(|v| DVector::from_vec(vec![v]))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_kappa_shrinkage_pooled_zero_when_rms_matches_omega_sd() {
        // omega_iov = 1.0; kappas = [+1, -1] across 2 subjects × 1 occasion
        // mean(κ²) = 1 → shrinkage = 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let kappas = make_kappas(vec![vec![1.0], vec![-1.0]]);
        let sh = compute_kappa_shrinkage(&kappas, &omega);
        assert_eq!(sh.len(), 1);
        assert!((sh[0]).abs() < 1e-10, "expected ~0, got {}", sh[0]);
    }

    #[test]
    fn test_kappa_shrinkage_pooled_positive_when_shrunk() {
        // kappas near zero → shrinkage > 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let kappas = make_kappas(vec![
            vec![0.01, 0.02],
            vec![-0.01, -0.02],
            vec![0.01, 0.02],
            vec![-0.01, -0.02],
        ]);
        let sh = compute_kappa_shrinkage(&kappas, &omega);
        assert!(sh[0] > 0.9, "expected high shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_kappa_shrinkage_pooled_nan_when_omega_zero() {
        let omega = DMatrix::zeros(1, 1);
        let kappas = make_kappas(vec![vec![0.1], vec![-0.1]]);
        let sh = compute_kappa_shrinkage(&kappas, &omega);
        assert!(sh[0].is_nan());
    }

    #[test]
    fn test_kappa_shrinkage_pooled_nan_when_fewer_than_2_obs() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let kappas = make_kappas(vec![vec![0.5]]);
        let sh = compute_kappa_shrinkage(&kappas, &omega);
        assert!(sh[0].is_nan());
    }

    #[test]
    fn test_kappa_shrinkage_by_occ_returns_empty_for_single_occasion() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let kappas = make_kappas(vec![vec![0.5], vec![-0.5]]);
        let sh = compute_kappa_shrinkage_by_occ(&kappas, &omega);
        assert!(sh.is_empty(), "expected empty for 1 occasion, got {:?}", sh);
    }

    #[test]
    fn test_kappa_shrinkage_by_occ_values() {
        // 4 subjects, 2 occasions.
        // OCC 1: kappas = [+1, -1, +1, -1] → mean(κ²) = 1 → shrinkage = 0 with omega=1
        // OCC 2: kappas = [0.1, -0.1, 0.1, -0.1] → mean(κ²) = 0.01 → high shrinkage
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let kappas = make_kappas(vec![
            vec![1.0, 0.1],
            vec![-1.0, -0.1],
            vec![1.0, 0.1],
            vec![-1.0, -0.1],
        ]);
        let sh = compute_kappa_shrinkage_by_occ(&kappas, &omega);
        assert_eq!(sh.len(), 2, "expected 2 occasions");
        assert!(
            (sh[0][0]).abs() < 1e-10,
            "occ 1 shrinkage ~0, got {}",
            sh[0][0]
        );
        assert!(sh[1][0] > 0.8, "occ 2 shrinkage high, got {}", sh[1][0]);
    }

    #[test]
    fn test_kappa_shrinkage_two_kappas_independent() {
        // n_kappa = 2: each kappa parameter should be computed independently.
        // kappa 0: RMS = 1.0 → shrinkage = 0 with omega_00 = 1.0
        // kappa 1: RMS = 0.1 → shrinkage = 1 - 0.1/1.0 = 0.9 with omega_11 = 1.0
        let omega = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        // Each subject has 1 occasion; kappa vector is [k0_val, k1_val].
        let kappas: Vec<Vec<DVector<f64>>> = vec![
            vec![DVector::from_vec(vec![1.0, 0.1])],
            vec![DVector::from_vec(vec![-1.0, -0.1])],
        ];
        let sh = compute_kappa_shrinkage(&kappas, &omega);
        assert_eq!(sh.len(), 2);
        assert!((sh[0]).abs() < 1e-10, "kappa 0 shrinkage ~0, got {}", sh[0]);
        assert!(
            (sh[1] - 0.9).abs() < 1e-10,
            "kappa 1 shrinkage ~0.9, got {}",
            sh[1]
        );
    }

    #[test]
    fn test_eps_shrinkage_ignores_nan_iwres() {
        // Censored rows have NaN IWRES — they must be filtered out.
        // After filtering, two values with mean(IWRES^2)=1 remain => shrinkage = 0.
        let subjects = vec![
            make_subject(vec![0.0], vec![1.0, f64::NAN]),
            make_subject(vec![0.0], vec![-1.0, f64::NAN]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "NaN IWRES not filtered, got {}", sh);
    }
}

/// Index of L[i,j] (i ≥ j) in column-major lower-triangle packing.
///
/// Layout: for j in 0..n { for i in j..n { ... } }, so column j starts at
/// offset Σ_{k<j}(n−k) = j·n − j·(j−1)/2.
#[inline]
fn chol_lt_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i >= j && i < n);
    // Column j starts at offset j*n - j*(j-1)/2.
    // For j==0: offset = 0. For j==1: offset = n. For j==2: offset = 2n-1.
    let col_offset = if j == 0 { 0 } else { j * n - j * (j - 1) / 2 };
    col_offset + (i - j)
}

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
fn extract_standard_errors(
    cov: &Option<DMatrix<f64>>,
    template: &ModelParameters,
) -> (
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
) {
    let cov = match cov {
        Some(c) => c,
        None => return (None, None, None, None),
    };

    let n = cov.nrows();
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    // SE on packed scale
    let se_packed: Vec<f64> = (0..n)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Theta: SE on original scale. Log-packed thetas (lower bound >= 0) are
    // optimized as x = log(theta), so SE(theta) = theta * SE(x) (delta method).
    // Identity-packed thetas (negative lower bound — e.g. covariate exponents,
    // exposure–hazard slopes) are optimized on the natural scale already, so
    // SE(theta) = SE(x): multiplying by the estimate would mis-scale it (and
    // flip the sign for a negative estimate). See `theta_packs_log`.
    let se_theta: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log(template.theta_lower[i]) {
                template.theta[i] * se_packed[i]
            } else {
                se_packed[i]
            }
        })
        .collect();

    // Omega: SE via multivariate delta method on Cholesky parameterization.
    //
    // Ω = L L^T, so omega_ij = Σ_{k≤min(i,j)} L_ik * L_jk.
    // Packed params: x = log(L_ii) for diagonals, x = L_ij for off-diags.
    // SE²(omega_ij) = g^T * C_omega * g, where g = ∂omega_ij/∂x.
    //
    // For diagonal omega the off-diagonal L elements are zero, so the formula
    // simplifies to the original: SE(omega_ii) = 2 * omega_ii * SE(log L_ii).
    // For block omega we compute the full lower triangle.
    let omega_start = n_theta;
    let se_omega: Vec<f64> = if template.omega.diagonal {
        (0..n_eta)
            .map(|i| {
                let idx = omega_start + i;
                if idx < n {
                    2.0 * template.omega.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let n_lt = n_eta * (n_eta + 1) / 2;
        let l = &template.omega.chol;

        // Extract omega sub-block of the full covariance matrix.
        let cov_omega = cov.view((omega_start, omega_start), (n_lt, n_lt));

        let mut se_vec = Vec::with_capacity(n_lt);
        // Column-major lower-triangle: for j in 0..n, for i in j..n
        for j in 0..n_eta {
            for i in j..n_eta {
                // Build gradient of omega_{ij} w.r.t. packed omega params.
                // omega_{ij} = Σ_{k=0}^{j} L_{ik} * L_{jk}
                let mut grad = vec![0.0f64; n_lt];
                for k in 0..=j {
                    let idx_ik = chol_lt_idx(i, k, n_eta);
                    let idx_jk = chol_lt_idx(j, k, n_eta);
                    // Chain rule: ∂L_{ab}/∂x_{ab} = L_{ab} if a==b (log), else 1.
                    let chain_ik = if i == k { l[(i, k)] } else { 1.0 };
                    let chain_jk = if j == k { l[(j, k)] } else { 1.0 };
                    grad[idx_ik] += l[(j, k)] * chain_ik;
                    if i != j {
                        grad[idx_jk] += l[(i, k)] * chain_jk;
                    } else {
                        // i == j: both terms contribute to the same index
                        grad[idx_ik] += l[(i, k)] * chain_ik;
                    }
                }
                // SE²(omega_{ij}) = g^T * C_omega * g
                let mut var = 0.0;
                for a in 0..n_lt {
                    if grad[a] == 0.0 {
                        continue;
                    }
                    for b in 0..n_lt {
                        if grad[b] == 0.0 {
                            continue;
                        }
                        var += grad[a] * cov_omega[(a, b)] * grad[b];
                    }
                }
                se_vec.push(if var > 0.0 { var.sqrt() } else { 0.0 });
            }
        }
        se_vec
    };

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start
        + if template.omega.diagonal {
            n_eta
        } else {
            n_eta * (n_eta + 1) / 2
        };
    let se_sigma: Vec<f64> = (0..n_sigma)
        .map(|i| {
            let idx = sigma_start + i;
            if idx < n {
                template.sigma.values[i] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // IOV (kappa): SE for diagonal variances of omega_iov.
    //
    // The packed Cholesky layout is column-major (see `pack_params`):
    // L[i,i] sits at offset `i*n - i*(i-1)/2` within the IOV block.
    // Same delta-method approximation as `se_omega`: SE(var_i) ≈ 2 * var_i * SE(log L_ii),
    // which is exact for diagonal IOV and a first-order approximation for block_kappa.
    // Off-diagonal covariance SEs are not currently reported (matches BSV omega).
    let kappa_start = sigma_start + n_sigma;
    let se_kappa: Option<Vec<f64>> = template.omega_iov.as_ref().map(|iov| {
        let n_kappa = iov.dim();
        (0..n_kappa)
            .map(|i| {
                let idx = if iov.diagonal {
                    kappa_start + i
                } else {
                    kappa_start + i * n_kappa - i * (i.saturating_sub(1)) / 2
                };
                if idx < n {
                    2.0 * iov.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    });

    (Some(se_theta), Some(se_omega), Some(se_sigma), se_kappa)
}

/// Simulate observations from a model with given parameters (random seed).
///
/// Data-reader warnings (e.g. missing II for ADDL doses) are not echoed here;
/// callers that obtained `population` via [`read_nonmem_csv`] should inspect
/// `population.warnings` before calling this function.
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult> {
    let mut rng = rand::rng();
    simulate_inner(model, population, params, n_sim, &mut rng)
}

/// Simulate with a fixed seed for reproducibility.
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(model, population, params, n_sim, &mut rng)
}

/// Options controlling [`simulate_with_options`].
#[derive(Debug, Clone, Default)]
pub struct SimulateOptions {
    /// Seed for reproducibility. `None` draws from entropy.
    pub seed: Option<u64>,
    /// When `Some(method)`, reassign each replicate's drawn etas to subjects by
    /// **propensity-score matching** against the subjects' fitted (posthoc)
    /// etas — Mahalanobis matching under the model `Ω` via the chosen
    /// [`MatchMethod`]. This restores the design↔eta association present in
    /// adaptively-dosed real-world data and corrects the resulting VPC bias
    /// (see [`crate::propensity_match`]). `None` disables matching.
    ///
    /// Requires `population` to be observed data: every subject must carry
    /// observations so its posthoc eta can be computed. Has no effect for the
    /// synthetic `[simulation]` block (no observed designs to match against).
    pub match_method: Option<MatchMethod>,
    /// Administrative censoring horizon for TTE endpoints (#522). When `Some(t)`,
    /// `t` overrides every TTE record's per-record `observation_window` so a
    /// re-simulated event-bearing subject censors at the planned study end `t`
    /// rather than drawing unbounded — the decoupled horizon a competing-risks VPC
    /// needs. `None` keeps the per-record window. No effect on Gaussian endpoints.
    pub horizon: Option<f64>,
}

/// Simulate observations, optionally with propensity-score matching.
///
/// With `opts.match_method == None` this is identical to
/// [`simulate_with_seed`] (or [`simulate`] when `opts.seed` is `None`). With a
/// `Some(method)`, the freshly drawn etas of each replicate are reassigned to
/// subjects so each subject's observed design is paired with a drawn eta close
/// (under the model `Ω` Mahalanobis metric) to that subject's fitted eta. The
/// fitted (posthoc) etas are computed once from `params` + the observed
/// `population`.
///
/// Returns `Err` if matching is requested but the population is empty or any
/// subject has no observations.
pub fn simulate_with_options(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<Vec<SimulationResult>, String> {
    use rand::SeedableRng;

    // ODE-accumulated (joint PK-TTE) hazards can't be event-time-sampled yet: there is
    // no closed-form inverse-CDF, and the drug-driven ODE event-location root-finder is
    // Slice 2.2 (#564). Reject up front so a joint PK-TTE model gets a clear error here
    // instead of silently producing no TTE rows (`tte_cause_params` returns None for it).
    #[cfg(feature = "survival")]
    if model.endpoints.values().any(|e| {
        matches!(
            e,
            crate::types::EndpointLikelihood::Tte {
                hazard: crate::types::HazardSpec::OdeAccumulated { .. }
            }
        )
    }) {
        return Err(
            "simulate does not yet support ODE-accumulated TTE hazards (joint PK-TTE); \
             drug-driven event-time sampling lands in Slice 2.2 (#564). Use an analytic \
             [event_model] family to simulate, or simulate the PK separately."
                .to_string(),
        );
    }

    // Validate the TTE horizon on the library path too — the `.ferx` parser
    // already rejects a non-finite / non-positive horizon, but a direct caller of
    // this API must get the same guard: a NaN window makes every `t_event < window`
    // test false (silent NaN event times), and a `<= 0` horizon censors every
    // subject at or before entry (#522 review).
    if let Some(h) = opts.horizon {
        if !h.is_finite() || h <= 0.0 {
            return Err(format!(
                "SimulateOptions.horizon must be finite and > 0 (got {h})"
            ));
        }
        // A horizon below a subject's TTE entry_time would censor it before it
        // entered observation (a row with time = h < entry_time). The
        // `[simulation]`-block path always enters at 0, but a left-truncated
        // population passed to this API must be rejected (#522 review).
        #[cfg(feature = "survival")]
        for subject in &population.subjects {
            for record in &subject.obs_records {
                let crate::types::ObsRecord::Event { entry_time, .. } = record;
                if *entry_time > h {
                    return Err(format!(
                        "SimulateOptions.horizon ({h}) is below subject '{}' entry_time \
                         ({entry_time}); the administrative horizon must be ≥ every \
                         subject's entry time",
                        subject.id
                    ));
                }
            }
        }
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::make_rng(),
    };

    // Guard the modeled-`RATE` dose precondition up front (#324). The
    // non-propensity branch reaches it via the `simulate_inner_with_draw`
    // chokepoint, but the propensity branch first runs a full inner EBE pass
    // (`run_inner_loop_warm` below) that integrates every subject — on an
    // unsupported config that would hit the per-path tripwire (silently in
    // release) or `resolve_rate`'s opaque `.expect` *before* the chokepoint
    // guard. Asserting here makes both branches fail with the same actionable
    // diagnostic; it is a no-op O(doses) scan on the common all-`Fixed` dataset.
    assert_modeled_doses_supported(model, population);

    let method = match opts.match_method {
        Some(m) => m,
        None => {
            return Ok(simulate_inner_with_draw(
                model,
                population,
                params,
                n_sim,
                1,
                None,
                opts.horizon,
                &mut rng,
            ));
        }
    };

    if population.subjects.is_empty() {
        return Err(
            "propensity-score matching requires a non-empty observed population".to_string(),
        );
    }
    if let Some(s) = population
        .subjects
        .iter()
        .find(|s| s.observations.is_empty())
    {
        return Err(format!(
            "propensity-score matching requires observations for every subject \
             (to compute posthoc etas); subject '{}' has none",
            s.id
        ));
    }

    // Fitted (posthoc) BSV etas depend only on the observed data + params, so
    // compute them once and reuse across replicates. The inner-loop budget here
    // is a self-contained MAP pass (this entry point takes no FitOptions); the
    // tolerances only need to localize each EBE well enough to match on, not to
    // reproduce a specific fit's inner settings.
    let (eta_hats, _h, _stats, _kappas) = crate::estimation::inner_optimizer::run_inner_loop_warm(
        model, population, params, 100, 1e-6, None, None, 1,
    );

    // A divergent EBE can come back non-finite (`find_ebe` only gates its
    // `converged` flag on a finite nll, not the returned eta). A NaN/Inf eta
    // would poison the Mahalanobis cost matrix and make the optimal-assignment
    // solver spin forever (NaN compares false against every candidate), so fail
    // loudly here instead.
    if let Some((i, _)) = eta_hats
        .iter()
        .enumerate()
        .find(|(_, e)| e.iter().any(|x| !x.is_finite()))
    {
        return Err(format!(
            "propensity-score matching: the posthoc eta for subject '{}' is \
             non-finite (its EBE did not converge); cannot match",
            population.subjects[i].id
        ));
    }

    let omega_inv = &params.omega.inv;
    Ok(simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        Some((&eta_hats, omega_inv, method)),
        opts.horizon,
        &mut rng,
    ))
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    // `simulate` / `simulate_with_seed` carry no horizon; the per-record window
    // applies. An explicit `[simulation] horizon` enters via `simulate_with_options`.
    simulate_inner_with_draw(model, population, params, n_sim, 1, None, None, rng)
}

/// Emit all observation rows for one subject given a fully-formed `eta_slice`
/// (length `n_eta + n_kappa`). Draws only residual epsilons from `rng`; the eta
/// is supplied by the caller (freshly sampled, or propensity-matched).
#[allow(clippy::too_many_arguments)]
/// Display time for observation row `j`: the raw data TIME when available
/// (matches sdtab / input), falling back to the internal `obs_times` clock,
/// which may be the shifted clock for stacked reset occasions. Shared by every
/// simulation row emitter so the static and reactive paths cannot drift.
fn obs_row_time(subject: &Subject, j: usize) -> f64 {
    subject
        .obs_raw_times
        .get(j)
        .copied()
        .unwrap_or(subject.obs_times[j])
}

fn emit_subject_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    eta_slice: &[f64],
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    horizon: Option<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
) {
    // `horizon` is consumed only by the TTE path below; without the `survival`
    // feature there are no TTE endpoints, so discard it to avoid an unused-arg warn.
    #[cfg(not(feature = "survival"))]
    let _ = horizon;

    // Use the same TV-covariate-aware dispatcher as estimation and post-fit
    // diagnostics. For no-TV subjects this stays on the one-pk-param fast path.
    let ipreds = pk::compute_predictions_with_tv(model, subject, &params.theta, eta_slice);

    // Add residual error (Gaussian path). IIV on residual error (#409): the
    // drawn `eta_slice` includes η_ruv, so scale the residual variance by
    // exp(2·η_ruv) — i.e. simulate `Y = IPRED + EPS·EXP(η_ruv)`.
    let ruv_scale = model.residual_var_scale(eta_slice);
    // Per-observation custom residual magnitude (#484): η-independent, so build
    // the [obs][sigma-slot] matrix once per subject and index it per row.
    let ruv_mult = model.ruv_obs_mult(subject, &params.theta);
    for (j, &ipred) in ipreds.iter().enumerate() {
        // FREM covariate pseudo-observations (FREMTYPE>0) use the additive
        // covariate sigma, not the PK error model applied to the θ+η override
        // that `compute_predictions_with_tv` now writes into FREM rows.
        let var = model.sim_residual_variance(
            subject,
            j,
            ipred,
            &params.sigma.values,
            ruv_scale,
            ruv_mult.as_ref().map(|m| m[j].as_slice()),
        );
        let eps: f64 = rng.sample(normal);
        let value = ipred + var.sqrt() * eps;

        results.push(SimulationResult {
            draw,
            sim,
            id: subject.id.clone(),
            // Raw data TIME (matches sdtab / input); `obs_times` may be
            // the internal shifted clock for stacked reset occasions.
            time: obs_row_time(subject, j),
            cmt: subject.obs_cmts[j],
            ipred,
            outcome: SimOutcome::Continuous { value },
        });
    }

    // TTE simulation path (requires survival feature)
    #[cfg(feature = "survival")]
    crate::survival::simulate_tte(
        model,
        subject,
        &params.theta,
        eta_slice,
        draw,
        sim,
        horizon,
        rng,
        results,
    );
}

/// `matched`, when `Some((fitted_etas, omega_inv, method))`, reassigns each
/// replicate's drawn etas to subjects by propensity-score matching against
/// `fitted_etas` (Mahalanobis matching under `omega_inv` via `method`; see
/// `crate::propensity_match`). `None` is the standard per-subject independent
/// draw and reproduces the previous behaviour byte-for-byte (same RNG draw
/// order).
#[allow(clippy::too_many_arguments)]
fn simulate_inner_with_draw<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    draw: usize,
    matched: Option<(&[DVector<f64>], &nalgebra::DMatrix<f64>, MatchMethod)>,
    horizon: Option<f64>,
    rng: &mut R,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    // Single chokepoint for every `simulate*` variant (both `simulate_inner` and
    // the propensity path funnel through here). Guard the modeled-`RATE` dose
    // precondition once per call, as `predict()` does — `simulate()` runs no
    // data-check otherwise. #324.
    assert_modeled_doses_supported(model, population);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        match matched {
            Some((fitted, omega_inv, method)) => {
                // Draw a pool of one eta per subject for this replicate, then
                // reassign the draws to subjects by matching them to the fitted
                // (posthoc) etas. Each subject keeps its own observed design.
                let n = population.subjects.len();
                let pool: Vec<DVector<f64>> = (0..n)
                    .map(|_| {
                        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                        &params.omega.chol * DVector::from_column_slice(&z)
                    })
                    .collect();
                let assign = crate::propensity_match::match_draws_to_fitted(
                    &pool, fitted, omega_inv, method,
                );
                for (i, subject) in population.subjects.iter().enumerate() {
                    let mut eta_slice: Vec<f64> = pool[assign[i]].iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    emit_subject_rows(
                        model,
                        subject,
                        params,
                        &eta_slice,
                        draw,
                        sim,
                        normal,
                        horizon,
                        rng,
                        &mut results,
                    );
                }
            }
            None => {
                for subject in &population.subjects {
                    // Sample eta from N(0, Omega); append zero kappas for IOV models.
                    let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
                    let z_vec = DVector::from_column_slice(&z);
                    let eta = &params.omega.chol * z_vec;
                    let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
                    eta_slice.resize(n_eta + model.n_kappa, 0.0);
                    emit_subject_rows(
                        model,
                        subject,
                        params,
                        &eta_slice,
                        draw,
                        sim,
                        normal,
                        horizon,
                        rng,
                        &mut results,
                    );
                }
            }
        }
    }

    results
}

// ======================================================================
// Adaptive (state-reactive / feedback) dosing — epic #391, experimental.
//
// Public entry point that wraps the `pub(crate)` reactive driver
// (`ode_predictions_adaptive`) with per-(subject, replicate) orchestration, the
// tagged-row output schema (Part D), and the frozen-schedule replay verifier
// (Part E backbone, default-on). Ipred monitors only; the `Dv` (assay-noised)
// path and per-subject RNG substreams land in S1.5.
// ======================================================================

use crate::sim::adaptive::{
    AdaptiveRun, ControllerCtx, DecisionLogEntry, DoseAction, DoseLedgerEntry, MonitorSpec,
};

/// Options for [`simulate_adaptive`].
///
/// `#[non_exhaustive]`: later knobs (per-subject schedules, the `Dv` path) land
/// as added fields; construct via [`AdaptiveSimulateOptions::default`] and assign
/// the fields you need.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveSimulateOptions {
    /// Seed for reproducibility. `None` draws η from entropy.
    pub seed: Option<u64>,
    /// Decision schedule on the subject clock — the times the controller is
    /// consulted. The same schedule is used for every subject in this slice.
    /// Must be non-empty: an empty schedule never consults the controller and is
    /// rejected (it would otherwise be a silent dose-free run).
    pub decision_times: Vec<f64>,
    /// Signals the controller may read at each decision. **Ipred only** here — a
    /// [`crate::sim::adaptive::ObserveMode::Dv`] monitor is rejected (it needs
    /// the per-subject RNG substreams of S1.5).
    pub monitors: Vec<MonitorSpec>,
    /// Run the frozen-schedule replay verifier after each (subject, replicate)
    /// (default `true`). A divergence is a typed `Err` that taints the whole
    /// result — never a buried warning (#391 Part E).
    pub verify: bool,
    /// Per-run decision cap — the runaway / closed-loop guard. The driver errors
    /// if a schedule exceeds it.
    pub max_decisions: usize,
}

impl Default for AdaptiveSimulateOptions {
    fn default() -> Self {
        Self {
            seed: None,
            decision_times: Vec::new(),
            monitors: Vec::new(),
            verify: true,
            max_decisions: 10_000,
        }
    }
}

/// Result of [`simulate_adaptive`]: the three Part-D artifacts that must agree,
/// returned as one verified unit so a caller can never pair a trajectory with
/// the wrong ledger. All three are long-form rows tagged by `(draw, sim, id)`.
///
/// `#[non_exhaustive]`: the remaining Part-D artifacts (per-subject metrics,
/// population summary, run manifest) land as added fields without breaking
/// callers who receive this struct.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveSimulationResult {
    /// Per-observation predictions: one row per (replicate, subject, obs time).
    /// **Ipred only** in this slice — `ipred` and the `Continuous` value are both
    /// the individual prediction; residual / assay error (the realized DV) and
    /// its per-subject RNG substream arrive in S1.5.
    pub trajectories: Vec<SimulationResult>,
    /// Every dose the controllers actually issued, tagged by `(draw, sim)`.
    pub ledger: Vec<DoseLedgerEntry>,
    /// One row per decision (including holds), in schedule order, up to and
    /// including any `Stop`.
    pub decisions: Vec<DecisionLogEntry>,
}

/// Simulate state-reactive ("adaptive" / feedback) dosing over a population
/// (epic #391, **experimental**).
///
/// For each subject and each of `n_sim` replicates: draw η ~ N(0, Ω), then run
/// one integration driven by a **fresh** controller minted from
/// `make_controller`. The controller is consulted at each `opts.decision_times`
/// point — it reads the declared `opts.monitors` (and the live state /
/// covariates / dose history via [`ControllerCtx`]) and returns the
/// [`DoseAction`]s to apply. The engine owns the timeline, applies
/// bioavailability / lag downstream exactly as for a static dose, and records
/// every realized dose ([`AdaptiveSimulationResult::ledger`]) and every decision
/// ([`AdaptiveSimulationResult::decisions`], including holds).
///
/// ## Controller factory — one per (subject, replicate)
///
/// `make_controller` is a **factory**, not a single shared closure: a fresh
/// controller is built for each run. Real controllers carry per-subject state
/// (debounce / `confirm` counters, windowed AUC, the current titration rung); a
/// single shared `FnMut` would leak that state across subjects and replicates —
/// a silent wrong answer that stateless test controllers never expose. A fresh
/// controller per run makes the isolation structural. A stateless rule is just a
/// factory whose closure ignores its environment, e.g.
/// `|| |ctx: &ControllerCtx| { … }`.
///
/// ## Requirements — typed errors, never a silent wrong answer
///
/// - **Non-empty decision schedule.** An empty `opts.decision_times` never
///   consults the controller (a silent dose-free run) and is rejected.
/// - **ODE model.** The reactive driver runs on the ODE engine; a model with no
///   `[odes]` block is rejected.
/// - **Dose-free subjects.** The regimen is controller-driven; a subject that
///   already carries `doses` is rejected (augmenting a pre-scheduled regimen is a
///   later step).
/// - **Ipred monitors only.** A `Dv` monitor is rejected (needs S1.5).
/// - **Verification (default on).** Each run's realized ledger is replayed
///   through the static engine and checked against the reactive trajectory; a
///   divergence beyond solver tolerance is an `Err`.
///
/// The `draw` tag on every row is `1`: this slice carries no parameter
/// uncertainty (that is Part C / a later slice), only between-subject η.
pub fn simulate_adaptive<F, C>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    make_controller: F,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String>
where
    F: Fn() -> C,
    C: FnMut(&ControllerCtx) -> Vec<DoseAction>,
{
    use rand::SeedableRng;

    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "simulate_adaptive requires an ODE model (the reactive driver runs on the ODE \
         engine); this model has no [odes] block"
            .to_string()
    })?;

    // An empty schedule means the controller is never consulted: the result is a
    // dose-free simulation that the verifier (replaying an empty ledger) passes
    // trivially. That is almost always a forgotten `decision_times` (the field
    // defaults to empty), so reject it rather than return a silent no-op — the
    // same "never a silent wrong answer" contract as the other preconditions.
    if opts.decision_times.is_empty() {
        return Err("simulate_adaptive requires a non-empty decision schedule \
             (`AdaptiveSimulateOptions::decision_times`); with no decision times the controller \
             is never consulted and no dose is ever issued"
            .to_string());
    }

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::make_rng(),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    // Root seed for the controller-assay substreams (DV-mode monitors, S1.5).
    // Resolved independently of the η `rng` above so that enabling a `Dv` monitor
    // never shifts the η draws (the all-`Ipred` path stays byte-identical). With no
    // run seed it is drawn from a fresh entropy source, matching the η stream's own
    // nondeterminism.
    let assay_root: u64 = match opts.seed {
        Some(s) => s,
        None => {
            let mut entropy: rand::rngs::StdRng = rand::make_rng();
            entropy.random::<u64>()
        }
    };

    let mut trajectories: Vec<SimulationResult> = Vec::new();
    let mut ledger: Vec<DoseLedgerEntry> = Vec::new();
    let mut decisions: Vec<DecisionLogEntry> = Vec::new();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        for subject in &population.subjects {
            // Draw η ~ N(0, Ω); append zero kappas for IOV models (the reactive
            // driver is BSV-only in this slice).
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let eta = &params.omega.chol * DVector::from_column_slice(&z);
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            let pk = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates);

            // Controller-assay capability for any `Dv` monitor: resolve the
            // endpoint's residual variance by CMT (scaled by this subject's
            // `ruv_scale` for IIV-on-RUV), or `None` when no [error_model] covers
            // that compartment (S1.5 edge a). The base seed keys this
            // (subject, replicate)'s controller-assay substream.
            let ruv_scale = model.residual_var_scale(&eta_slice);
            let sigma = &params.sigma.values;
            let resid_var = |cmt: usize, ipred: f64| -> Option<f64> {
                if !model.has_residual_error_for_cmt(cmt, sigma) {
                    return None;
                }
                Some(model.residual_variance_at(cmt, ipred, sigma) * ruv_scale)
            };
            let assay = crate::sim::adaptive::AssayNoise {
                resid_var: &resid_var,
                base_seed: crate::sim::adaptive::subject_assay_base_seed(
                    assay_root,
                    &subject.id,
                    sim,
                ),
            };

            // A fresh controller per (subject, replicate) — see the factory note.
            let mut controller = make_controller();
            let run: AdaptiveRun = crate::ode::ode_predictions_adaptive(
                ode,
                &pk.values,
                &params.theta,
                &eta_slice,
                subject,
                &opts.decision_times,
                &opts.monitors,
                &mut controller,
                opts.max_decisions,
                Some(&assay),
            )
            .map_err(|e| format!("subject '{}' (sim {sim}): {e}", subject.id))?;

            if opts.verify {
                crate::ode::verify_adaptive_frozen_replay(
                    ode,
                    &pk.values,
                    &params.theta,
                    &eta_slice,
                    subject,
                    &opts.decision_times,
                    &run,
                )
                .map_err(|e| {
                    format!(
                        "frozen-schedule replay verification failed for subject '{}' (sim {sim}): {e}",
                        subject.id
                    )
                })?;
            }

            // Tagged trajectory rows (Ipred only). `run.predictions` is indexed
            // by the subject's observation grid, exactly like `emit_subject_rows`.
            for (j, &pred) in run.predictions.iter().enumerate() {
                trajectories.push(SimulationResult {
                    draw: 1,
                    sim,
                    id: subject.id.clone(),
                    time: obs_row_time(subject, j),
                    cmt: subject.obs_cmts[j],
                    ipred: pred,
                    outcome: SimOutcome::Continuous { value: pred },
                });
            }

            // Stamp the replicate tags onto the ledger + decision rows — the
            // single-subject driver emits draw/sim = 0.
            for mut e in run.ledger {
                e.draw = 1;
                e.sim = sim;
                ledger.push(e);
            }
            for mut d in run.decisions {
                d.draw = 1;
                d.sim = sim;
                decisions.push(d);
            }
        }
    }

    Ok(AdaptiveSimulationResult {
        trajectories,
        ledger,
        decisions,
    })
}

/// Options controlling `simulate_with_uncertainty()`.
#[derive(Debug, Clone)]
pub struct SimulateUncertaintyOptions {
    /// Number of parameter sets to draw from the uncertainty distribution.
    pub n_uncertainty_draws: usize,
    /// Number of eta/eps replicates simulated *per* parameter draw.
    pub n_sim_per_draw: usize,
    /// How to draw the parameter sets — asymptotic MVN or SIR resamples.
    pub method: crate::estimation::uncertainty_samples::UncertaintyMethod,
    /// Optional seed for reproducibility. `None` draws from entropy.
    pub seed: Option<u64>,
}

/// Simulate observations while propagating parameter uncertainty.
///
/// For each of `opts.n_uncertainty_draws` parameter sets drawn from the
/// uncertainty distribution (asymptotic MVN around the ML estimate or stored
/// SIR resamples), simulate `opts.n_sim_per_draw` replicates of every subject
/// — sampling etas from the drawn Omega and epsilons from the drawn Sigma.
///
/// Total rows returned: `n_uncertainty_draws * n_sim_per_draw * n_subjects *
/// n_obs`. Each `SimulationResult` carries the originating `draw` and `sim`
/// indices so downstream code can compute per-time uncertainty bands.
pub fn simulate_with_uncertainty(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<Vec<SimulationResult>, String> {
    use rand::SeedableRng;

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        // Re-seed StdRng from entropy so simulate-without-seed is still
        // independent across calls but uses a uniform RNG type internally.
        None => rand::make_rng(),
    };

    let template =
        crate::estimation::uncertainty_samples::fitted_params_from_result(fit_result, model);
    let draws = crate::estimation::uncertainty_samples::draw_parameter_samples(
        fit_result,
        &template,
        opts.n_uncertainty_draws,
        opts.method,
        &mut rng,
    )?;

    // Final size is deterministic, so we can size the buffer once and avoid
    // repeated reallocations for large simulations.
    let total_obs: usize = population.subjects.iter().map(|s| s.obs_times.len()).sum();
    let mut results =
        Vec::with_capacity(opts.n_uncertainty_draws * opts.n_sim_per_draw * total_obs);
    for (k, params) in draws.iter().enumerate() {
        let mut rows = simulate_inner_with_draw(
            model,
            population,
            params,
            opts.n_sim_per_draw,
            k + 1,
            None,
            None,
            &mut rng,
        );
        results.append(&mut rows);
    }
    Ok(results)
}

/// A single simulated observation.
///
/// `draw` is the uncertainty draw index (1-based). For `simulate()` /
/// `simulate_with_seed()`, which use point-estimate parameters, `draw` is
/// always `1`. For `simulate_with_uncertainty()` it spans
/// `1..=n_uncertainty_draws`. `sim` is the replicate index *within* a draw.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub draw: usize,
    pub sim: usize,
    pub id: String,
    /// For Gaussian rows: the scheduled observation time from the subject's grid.
    /// For TTE rows: the sampled event time (equals `SimOutcome::Event::time`; the
    /// outer field exists for uniform iteration without matching on `outcome`).
    pub time: f64,
    /// CMT column value for this observation row. For Gaussian subjects this mirrors the data
    /// file's CMT (e.g. 1 for a central-compartment PK endpoint — not necessarily 0). For TTE
    /// rows (requires `survival` feature) it matches the `[event_model] cmt` declaration.
    pub cmt: usize,
    /// Individual prediction at η (Gaussian path only; NAN for non-Gaussian).
    pub ipred: f64,
    /// Simulated observation outcome.  For Gaussian: `SimOutcome::Continuous { value }`.
    /// For TTE (requires `survival` feature): `SimOutcome::Event { time, observed }`.
    pub outcome: SimOutcome,
}

/// Predict concentrations for a population using given parameters (no random effects).
///
/// Data-reader warnings (e.g. missing II for ADDL doses) are not echoed here;
/// callers that obtained `population` via [`read_nonmem_csv`] should inspect
/// `population.warnings` before calling this function.
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult> {
    // `predict()` runs no data-check (unlike `fit()`); guard the one
    // model-aware dose precondition so a modeled-`RATE` dose can't reach the
    // predictor unresolved (silent-wrong analytical / `.expect` panic). #324.
    assert_modeled_doses_supported(model, population);

    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();

    for subject in &population.subjects {
        let preds = pk::compute_predictions_with_tv(model, subject, &params.theta, &zero_eta);

        for (j, &pred) in preds.iter().enumerate() {
            results.push(PredictionResult {
                id: subject.id.clone(),
                // Raw data TIME (matches sdtab / input); `obs_times` may be the
                // internal shifted clock for stacked reset occasions.
                time: subject
                    .obs_raw_times
                    .get(j)
                    .copied()
                    .unwrap_or(subject.obs_times[j]),
                pred,
            });
        }
    }

    results
}

/// A single prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,
}

// ── TTE / survival prediction ─────────────────────────────────────────────────

/// Survival function prediction for one (subject, time) grid point.
#[cfg(feature = "survival")]
#[derive(Debug, Clone)]
pub struct SurvivalPredictionResult {
    /// Subject ID.
    pub id: String,
    /// CMT of the TTE endpoint.
    pub cmt: usize,
    /// Time at which S(t), H(t), h(t) are evaluated.
    pub time: f64,
    /// Cause-specific survival probability S(t) = exp(−H(t)) (this CMT alone).
    pub survival: f64,
    /// Cumulative hazard H(t) for this CMT.
    pub cum_hazard: f64,
    /// Instantaneous hazard h(t) for this CMT.
    pub hazard: f64,
    /// Cause-specific cumulative incidence F(t) = ∫₀ᵗ h(u)·S_all(u) du — the
    /// probability of having had *this* event type by t in the presence of the
    /// other (competing) causes. Equals 1 − survival when there is a single
    /// endpoint. Across all TTE CMTs, Σ cif + survival_all = 1.
    pub cif: f64,
    /// All-cause survival S_all(t) = exp(−Σ_j H_j(t)) over every TTE CMT — the
    /// probability of no event of any type by t. Equals `survival` when there is
    /// a single endpoint.
    pub survival_all: f64,
    /// Median survival time T₅₀ (where S(T₅₀) = 0.5); analytic closed form.
    pub median_survival: f64,
    /// Mean survival time E[T] = ∫₀^∞ S(t) dt; analytic for Exponential,
    /// numerical midpoint rule (2 000 steps) for Weibull and Gompertz.
    pub mean_survival: f64,
}

/// Linear-interpolated median survival time from a cumulative-hazard grid: the time
/// where `H(t) = ln 2` (i.e. `S(t) = 0.5`). Used for ODE-accumulated hazards, whose
/// median has no closed form. Returns NaN if the grid never reaches `ln 2`.
#[cfg(feature = "survival")]
fn grid_median_from_cumhaz(time_grid: &[f64], cum_haz: &[f64]) -> f64 {
    let ln2 = std::f64::consts::LN_2;
    // H(0) = 0 for a cumulative hazard, so if it has already reached ln2 by the first
    // grid point the median lies in (0, grid[0]] — interpolate from the origin.
    if let (Some(&t0), Some(&h0)) = (time_grid.first(), cum_haz.first()) {
        if h0.is_finite() && h0 >= ln2 && t0 > 0.0 {
            return t0 * ln2 / h0;
        }
    }
    for i in 1..time_grid.len() {
        let (h0, h1) = (cum_haz[i - 1], cum_haz[i]);
        if h0.is_finite() && h1.is_finite() && h0 < ln2 && h1 >= ln2 && h1 > h0 {
            let frac = (ln2 - h0) / (h1 - h0);
            return time_grid[i - 1] + frac * (time_grid[i] - time_grid[i - 1]);
        }
    }
    f64::NAN
}

/// Compute survival function predictions for TTE endpoints.
///
/// For each subject and each TTE CMT in `model.endpoints`, evaluates the
/// cause-specific `S(t) = exp(−H(t))`, `H(t)`, and `h(t)` at every point in
/// `time_grid` using population typical values (η = 0). When the model has
/// multiple TTE CMTs (competing risks) it also reports, per CMT, the
/// cause-specific cumulative incidence `F(t)` and the all-cause survival
/// `S_all(t) = exp(−Σ_j H_j(t))`, computed together so that
/// `Σ_k F_k(t) + S_all(t) = 1` holds at every grid point (see [`cif_curves`]).
///
/// Returns an empty Vec when the model has no TTE endpoints.
#[cfg(feature = "survival")]
pub fn predict_survival(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    time_grid: &[f64],
) -> Vec<SurvivalPredictionResult> {
    use crate::survival::{
        cif_curves, hazard_and_cum_hazard, mean_survival, median_survival, tte_cause_params,
    };

    // The competing-risks CIF telescopes the all-cause survival drop, which
    // requires the grid in ascending time order; sort a local copy so the
    // per-cause `cif` and the `Σ_k F_k + S_all = 1` invariant are correct for any
    // caller-supplied grid. A no-op for the already-sorted common case.
    let mut sorted_grid: Vec<f64> = time_grid.to_vec();
    sorted_grid.sort_by(f64::total_cmp);
    let time_grid: &[f64] = &sorted_grid;

    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();

    for subject in &population.subjects {
        // Per-cause hazard h(t) and cumulative hazard H(t) over the grid, plus the
        // distributional summaries, at the typical values (η = 0). Analytic families
        // use the closed forms; an ODE-accumulated (joint PK-TTE) hazard reads H(t)
        // from the integrated CHZ state and h(t) from its derivative. The all-cause
        // survival and CIF need every cause's H(t), so collect all causes up front.
        #[allow(clippy::type_complexity)]
        let mut rows: Vec<(usize, Vec<f64>, Vec<f64>, f64, f64)> = Vec::new();
        for (&cmt, endpoint) in &model.endpoints {
            let crate::types::EndpointLikelihood::Tte { hazard } = endpoint else {
                continue;
            };
            match hazard {
                crate::types::HazardSpec::Analytic { .. } => {
                    let Some((family, params_vec)) =
                        tte_cause_params(endpoint, &params.theta, &zero_eta, &subject.covariates)
                    else {
                        continue;
                    };
                    let mut h_row = Vec::with_capacity(time_grid.len());
                    let mut cum_row = Vec::with_capacity(time_grid.len());
                    for &t in time_grid {
                        let (h_val, cum_h) = hazard_and_cum_hazard(family, t, &params_vec);
                        h_row.push(h_val);
                        cum_row.push(cum_h);
                    }
                    let t_median = median_survival(family, &params_vec);
                    let t_mean = mean_survival(family, &params_vec);
                    rows.push((cmt, h_row, cum_row, t_median, t_mean));
                }
                crate::types::HazardSpec::OdeAccumulated { chz_state } => {
                    if model.ode_spec.is_none() {
                        continue;
                    }
                    // Read H(t)/h(t) from the augmented ODE solve — shared with the TTE
                    // likelihood via `crate::survival::ode_cumhaz_hazard`.
                    let (cum_row, h_row) = crate::survival::ode_cumhaz_hazard(
                        model,
                        subject,
                        *chz_state,
                        &params.theta,
                        &zero_eta,
                        time_grid,
                    );
                    // Median where S(t) = 0.5 ⇔ H(t) = ln2, linearly interpolated on the
                    // grid (NaN if the grid never reaches it). Mean needs ∫₀^∞ S and is
                    // left NaN for ODE hazards (a numerical-to-∞ summary is a follow-up).
                    let t_median = grid_median_from_cumhaz(time_grid, &cum_row);
                    rows.push((cmt, h_row, cum_row, t_median, f64::NAN));
                }
            }
        }
        if rows.is_empty() {
            continue;
        }

        let chz: Vec<Vec<f64>> = rows.iter().map(|r| r.2.clone()).collect();
        let (cif, s_all) = cif_curves(&chz);

        for (k, (cmt, h_row, cum_row, t_median, t_mean)) in rows.iter().enumerate() {
            for (i, &t) in time_grid.iter().enumerate() {
                results.push(SurvivalPredictionResult {
                    id: subject.id.clone(),
                    cmt: *cmt,
                    time: t,
                    survival: (-cum_row[i]).exp(),
                    cum_hazard: cum_row[i],
                    hazard: h_row[i],
                    cif: cif[k][i],
                    survival_all: s_all[i],
                    median_survival: *t_median,
                    mean_survival: *t_mean,
                });
            }
        }
    }

    results
}

#[cfg(all(test, feature = "survival"))]
mod survival_predict_tests {
    use super::grid_median_from_cumhaz;

    #[test]
    fn grid_median_interpolates_and_handles_no_crossing() {
        let ln2 = std::f64::consts::LN_2;
        let grid = [0.0, 1.0, 2.0, 3.0];
        // H crosses ln2 between t=1 and t=2; linear interp puts the median at t=1.5
        // (frac = (ln2 − 0.5·ln2)/(1.5·ln2 − 0.5·ln2) = 0.5).
        let cum = [0.0, ln2 * 0.5, ln2 * 1.5, ln2 * 3.0];
        let m = grid_median_from_cumhaz(&grid, &cum);
        assert!(
            (m - 1.5).abs() < 1e-9,
            "median should interpolate to 1.5; got {m}"
        );
        // H never reaches ln2 on the grid → NaN.
        let never = grid_median_from_cumhaz(&grid, &[0.0, 0.1, 0.2, 0.3]);
        assert!(never.is_nan(), "no crossing → NaN; got {never}");
        // Median at/before the first grid point (H(grid[0]) ≥ ln2): interpolated from the
        // origin (0,0). grid[0]=2, H=2·ln2 ⇒ median = 2·ln2 / (2·ln2) = 1.0.
        let early = grid_median_from_cumhaz(&[2.0, 4.0], &[ln2 * 2.0, ln2 * 4.0]);
        assert!(
            (early - 1.0).abs() < 1e-9,
            "median before the first grid point interpolates from the origin; got {early}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci`.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod iov_integration {
    use super::fit;
    use crate::types::*;

    use std::collections::HashMap;

    // ── Model ────────────────────────────────────────────────────────────────
    fn make_iov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
        };
        CompiledModel {
            name: "iov_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            // pk_param_fn: eta[0]=BSV for CL, eta[1]=KAPPA_CL (appended by IOV path)
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL = TVCL * exp(ETA_CL + KAPPA_CL)
                p.values[1] = theta[1]; // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
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
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
        }
    }

    // ── Population ───────────────────────────────────────────────────────────
    // 4 subjects, each with 2 occasions.  Times 1–3 = occasion 1, times 4–6 = occ 2.
    // Observations are plausible IV-bolus concentrations (dose=100).
    fn make_iov_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let dose_occ = vec![1u32, 2]; // two dose rows: one per occasion
        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0]),
            ("S2", vec![40.0, 32.0, 24.0, 38.0, 29.0, 22.0]),
            ("S3", vec![33.0, 25.0, 19.0, 31.0, 24.0, 18.0]),
            ("S4", vec![42.0, 33.0, 25.0, 39.0, 30.0, 23.0]),
        ];

        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: obs_times.clone(),
                obs_raw_times: Vec::new(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 6],
                occasions: occasions.clone(),
                dose_occasions: dose_occ.clone(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn fast_opts(
        method: EstimationMethod,
        optimizer: Optimizer,
        mu_referencing: bool,
    ) -> FitOptions {
        FitOptions {
            method,
            methods: Vec::new(),
            outer_maxiter: 60,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: method == EstimationMethod::FoceI,
            mu_referencing,
            optimizer,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────
    fn assert_iov_fit_ok(result: &FitResult) {
        assert!(result.ofv.is_finite(), "OFV must be finite");
        assert!(result.omega_iov.is_some(), "omega_iov must be populated");
        let iov_diag = result.omega_iov.as_ref().unwrap()[(0, 0)];
        assert!(
            iov_diag > 0.0,
            "omega_iov diagonal must be positive, got {iov_diag}"
        );
        assert_eq!(result.kappa_names, vec!["KAPPA_CL"], "kappa name mismatch");
        assert_eq!(result.ebe_kappas.len(), 4, "expected kappas for 4 subjects");
        for (i, subj_kappas) in result.ebe_kappas.iter().enumerate() {
            assert_eq!(subj_kappas.len(), 2, "subject {i} should have 2 occasions");
        }
    }

    // ── Tests: FOCE + all outer optimizers ───────────────────────────────────

    #[test]
    fn test_iov_foce_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_auto_reports_resolved_optimizer() {
        // `optimizer = auto` (the default) must resolve per-model and the
        // FitResult must report the concrete pick so the output records what
        // actually ran (#490). This IOV model is outside the analytic-gradient
        // scope, so `auto` lands on bobyqa.
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Auto, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
        let resolved = Optimizer::Auto.resolve_auto(&model);
        assert_eq!(result.optimizer, format!("auto ({})", resolved.label()));
        assert_eq!(resolved, Optimizer::Bobyqa);
    }

    #[test]
    fn test_iov_foce_slsqp() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Slsqp, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Lbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_nlopt_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::NloptLbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_mma() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Mma, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_bfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    /// End-to-end: the **same IOV model written two ways** — closed-form `one_cpt_iv`
    /// vs a user `[odes]` twin (`d/dt central = -(CL/V)·central`, `y = central/V`) —
    /// must converge to the **same optimum** (OFV + θ) on the same data. This exercises
    /// the full ODE IOV pipeline (analytic inner stacked-η gradient + analytic outer
    /// block-Ω gradient) through a real fit, not just per-evaluation gradient parity
    /// (#439 ODE IOV).
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~1 min: two full IOV fits, one over RK45); opt in with --features slow-tests"
    )]
    fn test_iov_ode_fit_matches_analytical_twin() {
        use crate::parser::model_parser::parse_model_string;
        const ANALYTICAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
        const ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
        let ana = parse_model_string(ANALYTICAL).expect("parse analytical IOV");
        let ode = parse_model_string(ODE).expect("parse ODE IOV");
        assert!(
            crate::sens::ode_provider::ode_iov_supported(&ode),
            "ODE twin must be ODE-IOV-provider supported (analytic inner+outer)"
        );
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);

        let ra = fit(&ana, &pop, &ana.default_params, &opts).expect("analytical fit");
        let ro = fit(&ode, &pop, &ode.default_params, &opts).expect("ODE fit");
        assert_iov_fit_ok(&ra);
        assert_iov_fit_ok(&ro);

        // Same structural model → same minimum (small slack for ODE integration vs
        // closed form, and the differing inner gradient routes).
        approx::assert_relative_eq!(ra.ofv, ro.ofv, max_relative = 1e-3, epsilon = 0.5);
        for k in 0..2 {
            approx::assert_relative_eq!(
                ra.theta[k],
                ro.theta[k],
                max_relative = 0.02,
                epsilon = 1e-2
            );
        }
        let iov_a = ra.omega_iov.as_ref().unwrap()[(0, 0)];
        let iov_o = ro.omega_iov.as_ref().unwrap()[(0, 0)];
        approx::assert_relative_eq!(iov_a, iov_o, max_relative = 0.10, epsilon = 1e-3);
    }

    // ── Tests: FOCEI ─────────────────────────────────────────────────────────

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_focei_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: mu-referencing ─────────────────────────────────────────────────

    #[test]
    fn test_iov_foce_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts)
            .expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_focei_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts)
            .expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: GN and GN_Hybrid ───────────────────────────────────────────────

    #[test]
    fn test_iov_gn() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("GN fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_gn_hybrid() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("GN hybrid fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Test: SAEM + IOV now supported (Step 11) ─────────────────────────────

    #[test]
    fn test_iov_saem_succeeds() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_ok(),
            "SAEM with IOV must succeed, got: {:?}",
            result.err()
        );
        let fr = result.unwrap();
        assert!(
            fr.ofv.is_finite(),
            "SAEM IOV OFV must be finite, got {}",
            fr.ofv
        );
        assert!(
            fr.omega_iov.is_some(),
            "omega_iov must be present in result"
        );
    }

    // ── Test: SAEM in a chained methods sequence + IOV succeeds ──────────────
    #[test]
    fn test_iov_saem_in_methods_chain_succeeds() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Foce];
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_ok(),
            "SAEM in methods chain with IOV must succeed, got: {:?}",
            result.err()
        );
        let fr = result.unwrap();
        assert!(
            fr.ofv.is_finite(),
            "chained SAEM+FOCE IOV OFV must be finite"
        );
        assert!(
            fr.omega_iov.is_some(),
            "omega_iov must survive the FOCE polishing step in a chained run"
        );
    }

    // ── Test: IMP in a chained methods sequence + IOV exercises the IS IOV path ─
    // `methods = [foce, imp]` on a kappa-bearing model drives the importance-
    // sampling marginal-likelihood step through its IOV branch
    // (`obs_nll_iov_fixed_kappa`, `compute_posterior_hessian`,
    // `subject_is_estimate`, `build_proposals`). κ is held at its EBE, so the
    // reported −2LL is a partial marginal (see `KappaTreatment::FixedAtMode`).
    #[test]
    fn test_iov_imp_chain_runs_importance_sampling() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        opts.methods = vec![EstimationMethod::Foce, EstimationMethod::Imp];
        opts.imp_samples = 200; // keep the per-subject sampling cheap
        opts.imp_seed = Some(42); // deterministic proposal draws
                                  // The IS IOV branch is the evaluation-only path; the estimating IMP
                                  // M-step does not yet support IOV (refused up front).
        opts.imp_eval_only = true;
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_ok(),
            "FOCE→IMP chain with IOV must succeed, got: {:?}",
            result.err()
        );
        let fr = result.unwrap();
        let is = fr
            .importance_sampling
            .as_ref()
            .expect("importance_sampling result must be populated by the IMP stage");
        assert!(
            is.minus2_log_likelihood.is_finite(),
            "IS marginal −2LL must be finite, got {}",
            is.minus2_log_likelihood
        );
        assert!(
            is.mc_standard_error.is_finite() && is.mc_standard_error >= 0.0,
            "IS Monte-Carlo SE must be finite and non-negative, got {}",
            is.mc_standard_error
        );
        assert_eq!(is.n_samples, 200, "n_samples should echo the IS budget");
        assert!(
            fr.omega_iov.is_some(),
            "omega_iov must survive into the IS-augmented result"
        );
    }

    // ── Test: trust-region optimizer + IOV must return Err ────────────────────
    // trust_region.rs currently passes `&[]` for kappas to pop_nll, which would
    // silently route the OFV through the non-IOV path. Guard at api.rs blocks
    // that before any wrong number escapes.
    #[test]
    fn test_iov_trust_region_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_err(),
            "trust_region with IOV must return an error"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("trust_region") && msg.contains("IOV"),
            "error message should mention trust_region and IOV, got: {msg}"
        );
    }

    // When a model has kappa declarations but the dataset carries no occasion
    // labels, `fit()` must return an error and `check_model_data` must surface
    // E_IOV_MISSING_OCC — rather than silently ignoring the kappas.
    #[test]
    fn test_iov_missing_occ_returns_err() {
        let model = make_iov_model();
        // Population built without occasion labels (empty `occasions` vectors).
        let mut pop = make_iov_population();
        for subj in &mut pop.subjects {
            subj.occasions.clear();
        }
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_err(),
            "IOV model without occasion labels must error"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("iov_column") || msg.contains("OCC") || msg.contains("occasion"),
            "error message should mention the missing occasion column, got: {msg}"
        );
    }

    #[test]
    fn test_check_model_data_flags_missing_occ() {
        use crate::diagnostics::Severity;
        let model = make_iov_model();
        let mut pop = make_iov_population();
        for subj in &mut pop.subjects {
            subj.occasions.clear();
        }
        let diags = super::check_model_data(&model, &pop);
        let d = diags
            .iter()
            .find(|d| d.code == "E_IOV_MISSING_OCC")
            .expect("expected E_IOV_MISSING_OCC diagnostic");
        assert_eq!(d.severity, Severity::Error);
        assert!(d.message.contains("iov_column") || d.message.contains("kappa"));
        assert_eq!(d.block.as_deref(), Some("fit_options"));
    }

    // `ferx check` must surface the same trust_region+IOV incompatibility that
    // `fit()` rejects — without it, a model could report `valid: true` and then
    // fail at fit time. `check_model_options` is the shared source of truth.
    #[test]
    fn test_check_model_options_flags_trust_region_iov() {
        let model = make_iov_model();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
        let diags = super::check_model_options(&model, &opts);
        let d = diags
            .iter()
            .find(|d| d.code == "E_OPTIMIZER_IOV")
            .expect("expected E_OPTIMIZER_IOV diagnostic");
        // Same wording fit() produces (regression against the extracted guard).
        assert!(d.message.contains("trust_region") && d.message.contains("IOV"));

        // A compatible optimizer produces no compatibility diagnostics.
        let ok_opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        assert!(super::check_model_options(&model, &ok_opts).is_empty());
    }

    // block_sigma correlated residual errors are wired into ordinary Gaussian
    // FOCE and SAEM; every other method in the chain must be rejected up front.
    #[test]
    fn test_check_model_options_block_sigma_rejects_unsupported_methods() {
        let mut model =
            crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
        model.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];

        // FOCEI is rejected.
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
        let diags = super::check_model_options(&model, &opts);
        let d = diags
            .iter()
            .find(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED")
            .expect("expected E_BLOCK_SIGMA_METHOD_UNSUPPORTED for FOCEI");
        assert!(d.is_error() && d.message.contains("method = foce") && d.message.contains("saem"));

        // SAEM is accepted for the ordinary Gaussian subset.
        let saem = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        assert!(!super::check_model_options(&model, &saem)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"));

        // Plain FOCE is accepted (no block_sigma diagnostic).
        let foce = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        assert!(!super::check_model_options(&model, &foce)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"));

        model.n_kappa = 1;
        assert!(super::check_model_options(&model, &saem)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_IOV_UNSUPPORTED"));
    }

    // The block_sigma + SDE rejection is scoped to SAEM: the FOCE dense path adds
    // the EKF process-noise `p_obs` into R and worked before SAEM support was
    // added, so a FOCE-only chain must stay accepted. Regression guard against
    // the over-broad method-agnostic rejection (PR #557 review).
    #[test]
    fn test_check_model_options_block_sigma_sde_rejected_for_saem_not_foce() {
        let mut model =
            crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
        model.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];
        // is_sde() reads diffusion_theta_start; flag the model as an SDE/EKF model.
        model.diffusion_theta_start = Some(0);

        let saem = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        assert!(super::check_model_options(&model, &saem)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_SDE_UNSUPPORTED"));

        let foce = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        assert!(!super::check_model_options(&model, &foce)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_SDE_UNSUPPORTED"));
    }

    // The block_sigma + TTE rejection is blanket (every method): the TTE Laplace
    // path accumulates R diagonally and cannot carry the cross-endpoint
    // covariance, so it must be rejected for FOCE as well as SAEM (PR #557).
    #[cfg(feature = "survival")]
    #[test]
    fn test_check_model_options_block_sigma_tte_rejected_all_methods() {
        let mut model =
            crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
        model.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];
        // has_tte() scans endpoints for a Tte variant; attach a trivial one.
        model.endpoints.insert(
            2,
            crate::types::EndpointLikelihood::Tte {
                hazard: crate::types::HazardSpec::Analytic {
                    family: crate::types::HazardFamily::Exponential,
                    param_fn: Box::new(|_theta, _eta, _cov| vec![0.1]),
                },
            },
        );

        for method in [EstimationMethod::Saem, EstimationMethod::Foce] {
            let opts = fast_opts(method, Optimizer::Bobyqa, false);
            assert!(
                super::check_model_options(&model, &opts)
                    .iter()
                    .any(|d| d.code == "E_BLOCK_SIGMA_TTE_UNSUPPORTED"),
                "TTE + block_sigma must be rejected for {method:?}"
            );
        }
    }

    #[test]
    fn test_saem_accepts_block_sigma_cross_endpoint_fit() {
        use crate::parser::model_parser::parse_model_string;
        use std::collections::HashMap;

        let model = parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
        )
        .expect("cross-endpoint block_sigma ODE model parses");

        let mut subjects = Vec::new();
        for (id, dose_amt, obs) in [
            ("1", 100.0, vec![17.0, 8.0, 15.0, 7.0]),
            ("2", 80.0, vec![14.0, 6.8, 12.0, 6.0]),
        ] {
            subjects.push(Subject {
                id: id.into(),
                doses: vec![DoseEvent::new(0.0, dose_amt, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 1.0, 2.0, 2.0],
                obs_raw_times: Vec::new(),
                observations: obs,
                obs_cmts: vec![1, 2, 1, 2],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 4],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            });
        }
        let population = Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mut opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        opts.saem_n_exploration = 1;
        opts.saem_n_convergence = 0;
        opts.saem_n_mh_steps = 1;
        opts.saem_adapt_interval = 10;
        opts.saem_omega_burnin = 0;
        opts.saem_seed = Some(548);

        let result = fit(&model, &population, &model.default_params, &opts)
            .expect("SAEM fit with cross-endpoint block_sigma should succeed");
        assert!(result.ofv.is_finite(), "SAEM OFV must be finite");
    }

    // Regression: the SAEM final OFV (computed via `2·pop_nll` with
    // `interaction = true`) must include the block_sigma cross-endpoint
    // covariance. The interaction Laplace accumulator builds R diagonally, so
    // before #557 SAEM silently dropped the correlation and reported the
    // diagonal OFV (~4 units low here) instead of the FOCE/NONMEM value. At
    // fully-fixed parameters both estimators evaluate the same conditional
    // likelihood, so their OFVs must agree.
    #[test]
    fn test_saem_block_sigma_ofv_matches_foce() {
        use crate::parser::model_parser::parse_model_string;
        use std::collections::HashMap;

        // Single-endpoint combined error with a fixed correlated $SIGMA block —
        // the docs validation model (NONMEM FOCE OFV 18.7158, ferx FOCE 18.7274).
        let model = parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0) FIX
  theta TVV(10.0, 0.1, 100.0) FIX
  omega ETA_CL ~ 0.04 FIX
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
",
        )
        .expect("fixed-param block_sigma combined model parses");

        let mut subjects = Vec::new();
        for (id, obs) in [
            ("1", vec![9.5, 8.0, 6.2]),
            ("2", vec![10.5, 7.4, 5.6]),
            ("3", vec![8.8, 7.9, 6.7]),
        ] {
            subjects.push(Subject {
                id: id.into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 2.0, 4.0],
                obs_raw_times: Vec::new(),
                observations: obs,
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 3],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            });
        }
        let population = Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        // FOCE evaluation (all params fixed → maxiter 0 just evaluates the OFV).
        let mut foce_opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        foce_opts.outer_maxiter = 0;
        let foce_ofv = fit(&model, &population, &model.default_params, &foce_opts)
            .expect("FOCE block_sigma evaluation succeeds")
            .ofv;

        // SAEM evaluation at the same fixed params.
        let mut saem_opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        saem_opts.saem_n_exploration = 2;
        saem_opts.saem_n_convergence = 0;
        saem_opts.saem_n_mh_steps = 1;
        saem_opts.saem_adapt_interval = 10;
        saem_opts.saem_omega_burnin = 0;
        saem_opts.saem_seed = Some(12345);
        let saem_ofv = fit(&model, &population, &model.default_params, &saem_opts)
            .expect("SAEM block_sigma evaluation succeeds")
            .ofv;

        // Params are fixed, so the EBEs (hence the conditional OFV) are identical
        // regardless of estimator; the dense covariance must be carried by both.
        assert!(
            (saem_ofv - foce_ofv).abs() < 1e-3,
            "SAEM OFV {saem_ofv} must match FOCE OFV {foce_ofv} (block_sigma correlation dropped?)"
        );
    }

    #[test]
    fn test_check_model_options_block_sigma_rejects_iov() {
        let mut model = make_iov_model();
        model.residual_correlations = vec![crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 0,
            rho: 0.5,
        }];

        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let diags = super::check_model_options(&model, &opts);
        let d = diags
            .iter()
            .find(|d| d.code == "E_BLOCK_SIGMA_IOV_UNSUPPORTED")
            .expect("expected E_BLOCK_SIGMA_IOV_UNSUPPORTED for block_sigma + IOV");

        assert!(d.is_error());
        assert!(d.message.contains("inter-occasion"));
        assert_eq!(d.block.as_deref(), Some("fit_options"));
    }

    // IMPMAP does not yet support IOV; `ferx check` must flag it up front rather
    // than letting the fit fail at runtime (review finding #3).
    #[test]
    fn test_check_model_options_flags_impmap_iov() {
        let model = make_iov_model();
        let opts = fast_opts(EstimationMethod::Impmap, Optimizer::Bobyqa, false);
        let diags = super::check_model_options(&model, &opts);
        let d = diags
            .iter()
            .find(|d| d.code == "E_IMPMAP_IOV_UNSUPPORTED")
            .expect("expected E_IMPMAP_IOV_UNSUPPORTED diagnostic");
        assert!(d.is_error() && d.message.contains("inter-occasion"));
    }

    // The Enzyme AD path was retired, so explicitly requesting AD must error
    // rather than silently running a different method. `auto`/`fd` must still pass.
    #[test]
    fn ad_requested_errors_now_that_ad_is_retired() {
        let model = make_iov_model();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);

        opts.gradient_method = crate::types::GradientMethod::Ad;
        let diags = super::check_model_options(&model, &opts);
        assert!(
            diags
                .iter()
                .any(|d| d.code == "E_AD_RETIRED" && d.is_error()),
            "explicit gradient_method=ad must error now that AD is retired, got: {diags:?}"
        );

        for gm in [
            crate::types::GradientMethod::Auto,
            crate::types::GradientMethod::Fd,
        ] {
            opts.gradient_method = gm;
            assert!(
                !super::check_model_options(&model, &opts)
                    .iter()
                    .any(|d| d.code == "E_AD_RETIRED"),
                "gradient_method={gm:?} must not trigger E_AD_RETIRED"
            );
        }
    }

    /// Regression for review finding #5 (IOV + compartments[i]).
    ///
    /// When the model has a [derived] expression that references `compartments[i]`
    /// and the fit has IOV subjects, `W_DERIVED_CMT_IOV_UNSUPPORTED` must be
    /// emitted. The `predict_iov` path does not compute compartment states; the
    /// per-subject `compartment_states` vec stays empty (`vec![]`), so any
    /// `compartments[i]` reference evaluates to NaN. The warning makes this
    /// explicit rather than silent.
    #[test]
    fn iov_with_compartments_derived_emits_unsupported_warning() {
        let mut model = make_iov_model();
        // Inject a derived expression that sets uses_compartments = true,
        // just like a parsed `[derived] cmt0 = compartments[0]` would.
        model.derived_exprs.push(DerivedExprSpec {
            name: "cmt0".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx| ctx.compartments.first().copied().unwrap_or(f64::NAN)),
            },
            uses_compartments: true,
        });
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params.clone(), &opts).expect("fit must succeed");

        // Warning must be present.
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("W_DERIVED_CMT_IOV_UNSUPPORTED")),
            "expected W_DERIVED_CMT_IOV_UNSUPPORTED warning; got: {:?}",
            result.warnings
        );
        // Compartment states for IOV subjects must be entirely empty (outer vec
        // is vec![], not vec![vec![]; n_obs]) — the predict_iov path never
        // populates them.
        for sr in &result.subjects {
            assert!(
                sr.compartment_states.is_empty(),
                "IOV subject {} must have empty compartment_states (len={}), \
                 got {}",
                sr.id,
                sr.ipred.len(),
                sr.compartment_states.len()
            );
        }
    }

    /// #400: an analytical oral model with a zero-order input into the depot
    /// (an infusion into cmt 1) cannot have its compartment states expressed by
    /// the superposition state helper (which models an oral infusion as a depot
    /// bypass). So `compartment_states` is left empty (→ NaN compartments) and
    /// `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL` makes that explicit, just
    /// like the reset/IOV/TV cases. Predictions themselves stay exact.
    #[test]
    fn analytical_oral_depot_infusion_with_compartments_derived_emits_warning() {
        use crate::parser::model_parser::parse_full_model;
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
        let mut model = parse_full_model(src).expect("model parses").model;
        assert!(model.ode_spec.is_none(), "model must be analytical");
        // Inject a derived expression that references compartments[0] so the
        // warning is gated on (mirrors a parsed `[derived] cmt0 = compartments[0]`).
        model.derived_exprs.push(DerivedExprSpec {
            name: "cmt0".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx| ctx.compartments.first().copied().unwrap_or(f64::NAN)),
            },
            uses_compartments: true,
        });

        // One subject with an explicit zero-order infusion into the depot (cmt 1):
        // rate 25 over AMT/rate = 4 h, then first-order KA absorption.
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0, 12.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.8, 1.4, 1.6, 0.9, 0.4],
            obs_cmts: vec![2; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let pop = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params.clone(), &opts).expect("fit must succeed");

        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL")),
            "expected W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL warning; got: {:?}",
            result.warnings
        );
        // Predictions must still be finite (the event-driven path computed them);
        // only the compartment states degrade.
        for sr in &result.subjects {
            assert!(
                sr.compartment_states.is_empty(),
                "depot-infusion subject must have empty compartment_states (got {})",
                sr.compartment_states.len()
            );
            assert!(
                sr.ipred.iter().all(|p| p.is_finite()),
                "predictions must be finite"
            );
        }
    }
}

#[cfg(test)]
mod extract_se_tests {
    use super::extract_standard_errors;
    use crate::types::*;
    use nalgebra::DMatrix;

    /// Helper for tests where the BSV omega is a single diagonal eta and only
    /// the IOV block varies. BSV-omega tests below build their own template
    /// inline because they need a 3-eta block omega.
    fn make_template(omega_iov: Option<OmegaMatrix>, kappa_fixed: Vec<bool>) -> ModelParameters {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov,
            kappa_fixed,
        }
    }

    // ── BSV omega ────────────────────────────────────────────────────────────

    /// Block omega with n_eta = 3 (numerically diagonal).  se_omega is now the
    /// full lower triangle (length 6), column-major.  The diagonal elements at
    /// LT positions 0, 3, 5 should match the old formula; off-diagonals should
    /// also be finite (non-NaN).
    #[test]
    fn test_se_omega_block_n3_full_lower_triangle() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let omega =
            OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into(), "E3".into()], false);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 3],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_block(6) + sigma(1) = 8.
        // Within the omega block (start = 1): L[0,0]=1, L[1,0]=2, L[2,0]=3,
        // L[1,1]=4, L[2,1]=5, L[2,2]=6.
        let n = 8;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2);
        }
        let (_, se_omega, _, _) = extract_standard_errors(&Some(cov), &template);
        let se = se_omega.unwrap();
        // Full LT: [omega(0,0), omega(1,0), omega(2,0), omega(1,1), omega(2,1), omega(2,2)]
        assert_eq!(se.len(), 6);
        // Diagonal SEs: same as before (omega is numerically diagonal → off-diag L=0).
        // omega(0,0) at LT[0]: 2 * 0.04 * 2.0 = 0.16
        assert!((se[0] - 0.16).abs() < 1e-12, "se(0,0) = {}", se[0]);
        // omega(1,1) at LT[3]: 2 * 0.09 * 5.0 = 0.90
        assert!((se[3] - 0.90).abs() < 1e-12, "se(1,1) = {}", se[3]);
        // omega(2,2) at LT[5]: 2 * 0.16 * 7.0 = 2.24
        assert!((se[5] - 2.24).abs() < 1e-12, "se(2,2) = {}", se[5]);
        // Off-diagonals should be finite
        for (idx, &v) in se.iter().enumerate() {
            assert!(v.is_finite(), "se[{}] not finite", idx);
        }

        // Verify the omega_se_at helper
        use crate::types::omega_se_at;
        let se_opt = Some(se);
        assert!((omega_se_at(&se_opt, 3, 0, 0).unwrap() - 0.16).abs() < 1e-12);
        assert!((omega_se_at(&se_opt, 3, 1, 1).unwrap() - 0.90).abs() < 1e-12);
        assert!((omega_se_at(&se_opt, 3, 2, 2).unwrap() - 2.24).abs() < 1e-12);
        // Symmetric: omega_se_at(1,0) == omega_se_at(0,1)
        assert_eq!(omega_se_at(&se_opt, 3, 1, 0), omega_se_at(&se_opt, 3, 0, 1));
    }

    /// Block omega with non-zero off-diagonals: verify off-diagonal SEs are
    /// positive and that they differ from the (incorrect) zero that would
    /// result from a diagonal-only implementation.
    #[test]
    fn test_se_omega_block_offdiag_positive() {
        // Ω = [[0.09, 0.02], [0.02, 0.04]]  (corr ≈ 0.33)
        let mut mat = DMatrix::<f64>::zeros(2, 2);
        mat[(0, 0)] = 0.09;
        mat[(1, 1)] = 0.04;
        mat[(0, 1)] = 0.02;
        mat[(1, 0)] = 0.02;
        let omega = OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into()], false);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 2],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed: theta(1) + omega_block(3) + sigma(1) = 5.  Identity cov.
        let cov = Some(DMatrix::<f64>::identity(5, 5));
        let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
        let se = se_omega.unwrap();
        // Full LT: [omega(0,0), omega(1,0), omega(1,1)]
        assert_eq!(se.len(), 3);
        assert!(se[0] > 0.0, "diagonal SE(0,0) should be positive");
        assert!(se[1] > 0.0, "off-diagonal SE(1,0) should be positive");
        assert!(se[2] > 0.0, "diagonal SE(1,1) should be positive");
        // omega_se_at helper
        use crate::types::omega_se_at;
        let se_opt = Some(se);
        assert!(omega_se_at(&se_opt, 2, 1, 0).unwrap() > 0.0);
        // diagonal-only format returns None for off-diag
        let diag_only = Some(vec![0.1, 0.2]);
        assert!(omega_se_at(&diag_only, 2, 1, 0).is_none());
    }

    /// Diagonal omega path is unaffected by the fix; this guards the simple case.
    #[test]
    fn test_se_omega_diagonal_unchanged() {
        let omega = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["E1".into(), "E2".into()]);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 2],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_diag(2) + sigma(1) = 4. Identity cov.
        let cov = Some(DMatrix::<f64>::identity(4, 4));
        let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
        let se = se_omega.unwrap();
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    /// Theta SE back-transform must respect the packing scale. A log-packed theta
    /// (lower bound >= 0) is optimized as log(theta), so SE(theta) = theta * SE(x);
    /// an identity-packed theta (negative lower bound, e.g. an exposure–hazard slope
    /// or covariate exponent) is optimized on the natural scale, so SE(theta) = SE(x)
    /// with no multiply. Regression for the bug where every theta was multiplied by
    /// its estimate, mis-scaling (and sign-flipping for negative estimates) the SE
    /// of identity-packed thetas.
    #[test]
    fn test_se_theta_respects_packing_scale() {
        // theta[0] = TVCL, lower 0.1 >= 0  → log-packed
        // theta[1] = BETA, lower -10 < 0   → identity-packed (can be negative)
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["E1".into()]);
        let template = ModelParameters {
            theta: vec![2.0, 0.25],
            theta_names: vec!["TVCL".into(), "BETA".into()],
            theta_lower: vec![0.1, -10.0],
            theta_upper: vec![50.0, 10.0],
            theta_fixed: vec![false, false],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(2) + omega(1) + sigma(1) = 4.
        // Set diagonal so packed SEs are theta:0.1, theta:0.3 (the rest unused here).
        let mut cov = DMatrix::<f64>::zeros(4, 4);
        cov[(0, 0)] = 0.1_f64.powi(2); // se_packed[0] = 0.1
        cov[(1, 1)] = 0.3_f64.powi(2); // se_packed[1] = 0.3
        cov[(2, 2)] = 1.0;
        cov[(3, 3)] = 1.0;
        let (se_theta, _, _, _) = extract_standard_errors(&Some(cov), &template);
        let se = se_theta.unwrap();
        // log-packed: SE = theta * SE(x) = 2.0 * 0.1 = 0.2
        assert!(
            (se[0] - 0.2).abs() < 1e-12,
            "log-packed theta SE = {}",
            se[0]
        );
        // identity-packed: SE = SE(x) = 0.3 (NOT 0.25 * 0.3 = 0.075)
        assert!(
            (se[1] - 0.3).abs() < 1e-12,
            "identity-packed theta SE = {} (must be the packed SE, not estimate-scaled)",
            se[1]
        );
    }

    // ── IOV (kappa) ──────────────────────────────────────────────────────────

    /// Identity covariance matrix gives unit SEs on the packed scale.  After
    /// the delta-method transform `SE(var_i) = 2 * var_i * SE(log L_ii)`, the
    /// returned se_kappa[i] equals 2 * variance_i for diagonal IOV.
    #[test]
    fn test_se_kappa_diagonal_uses_correct_index() {
        let iov =
            OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["KAPPA_CL".into(), "KAPPA_V".into()]);
        let template = make_template(Some(iov), vec![false, false]);
        // Packed layout: [theta, omega(1), sigma, kappa_1, kappa_2] → 5 entries.
        // Identity cov means se_packed = [1, 1, 1, 1, 1].
        let cov = Some(DMatrix::<f64>::identity(5, 5));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        let se = se_kappa.expect("se_kappa should be Some when omega_iov is set");
        assert_eq!(se.len(), 2);
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    /// Block kappa with n=3: column-major Cholesky packing places L[1,1] at
    /// offset 3 and L[2,2] at offset 5 within the IOV block.  Mirrors the BSV
    /// regression test above.
    #[test]
    fn test_se_kappa_block_n3_column_major_indexing() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let _chol = mat.clone().cholesky().unwrap().l();
        let iov = OmegaMatrix::from_matrix(mat, vec!["K1".into(), "K2".into(), "K3".into()], false);
        let template = make_template(Some(iov), vec![false; 3]);
        // Packed layout: theta(1) + omega_diag(1) + sigma(1) + kappa_block(6) = 9
        // Within the kappa block (start = 3): L11=0, L21=1, L31=2, L22=3, L32=4, L33=5
        // → diagonals at packed indices 3, 6, 8.
        // Build a cov matrix with distinct diagonal entries so we can verify
        // which index each SE pulls from.
        let n = 9;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
        }
        let (_, _, _, se_kappa) = extract_standard_errors(&Some(cov), &template);
        let se = se_kappa.unwrap();
        // L[0,0] at idx 3 → se_packed = 4 → se_kappa[0] = 2 * 0.04 * 4 = 0.32
        assert!((se[0] - 2.0 * 0.04 * 4.0).abs() < 1e-12, "got {}", se[0]);
        // L[1,1] at idx 6 → se_packed = 7 → se_kappa[1] = 2 * 0.09 * 7 = 1.26
        assert!((se[1] - 2.0 * 0.09 * 7.0).abs() < 1e-12, "got {}", se[1]);
        // L[2,2] at idx 8 → se_packed = 9 → se_kappa[2] = 2 * 0.16 * 9 = 2.88
        assert!((se[2] - 2.0 * 0.16 * 9.0).abs() < 1e-12, "got {}", se[2]);
    }

    #[test]
    fn test_se_kappa_none_when_no_iov() {
        let template = make_template(None, vec![]);
        let cov = Some(DMatrix::<f64>::identity(3, 3));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        assert!(se_kappa.is_none());
    }

    #[test]
    fn test_se_kappa_none_when_no_cov() {
        let iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let template = make_template(Some(iov), vec![false]);
        let (_, _, _, se_kappa) = extract_standard_errors(&None, &template);
        assert!(se_kappa.is_none());
    }
}

#[cfg(test)]
mod tests_cov_diagnostics {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_cov_diagnostics_none_input() {
        let (ev, cn) = cov_diagnostics(None);
        assert!(ev.is_none());
        assert!(cn.is_none());
    }

    #[test]
    fn test_cov_diagnostics_fewer_than_two_free_params() {
        // 2×2 matrix where only one param is free (second has zero diagonal)
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 4.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        assert!(ev.is_none());
        assert!(cn.is_none());
    }

    #[test]
    fn test_cov_diagnostics_excludes_fixed_params_zero_diagonal() {
        // 3×3 covariance; middle param is fixed (zero row/col).
        // Free subblock [[4, 0.5], [0.5, 2]] is non-singular, so condition
        // number must be finite and eigenvalues length must be 2.
        let mut m = DMatrix::<f64>::zeros(3, 3);
        m[(0, 0)] = 4.0;
        m[(0, 2)] = 0.5;
        m[(2, 0)] = 0.5;
        m[(2, 2)] = 2.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        let ev = ev.expect("eigenvalues must be Some");
        let cn = cn.expect("condition_number must be Some");
        assert_eq!(ev.len(), 2, "must have 2 eigenvalues (one per free param)");
        assert!(
            cn.is_finite(),
            "condition_number must be finite for non-singular subblock"
        );
        assert!(cn > 0.0);
        // Eigenvalues must be sorted descending
        assert!(ev[0] >= ev[1]);
    }

    #[test]
    fn test_cov_diagnostics_inf_condition_number_for_non_positive_eigenvalue() {
        // Construct a 2×2 covariance matrix whose free-param correlation matrix
        // is [[1, r], [r, 1]] with |r| > 1 — not PSD, so min eigenvalue < 0.
        // (r = 1.5 → eigenvalues 2.5 and -0.5)
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.5; // cor = 1.5/sqrt(1*1) = 1.5 > 1 → non-PSD
        m[(1, 0)] = 1.5;
        m[(1, 1)] = 1.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        let cn = cn.expect("condition_number must be Some");
        assert!(
            cn.is_infinite(),
            "condition_number must be Inf when min eigenvalue ≤ 0, got {cn}"
        );
        let ev = ev.expect("eigenvalues must be Some");
        assert!(
            ev.last().copied().unwrap_or(1.0) <= 0.0,
            "min eigenvalue must be ≤ 0"
        );
    }

    #[test]
    fn test_cov_diagnostics_inf_condition_number_for_near_zero_eigenvalue() {
        // Simulate a floating-point near-zero negative eigenvalue (e.g. -1e-15)
        // that a well-conditioned matrix could produce due to numerical noise.
        // The tolerance guard (> 1e-10) must treat this as singular → INFINITY.
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.0 - 1e-15; // cor ≈ 1 → min eigenvalue ≈ 0 (or tiny negative)
        m[(1, 0)] = 1.0 - 1e-15;
        m[(1, 1)] = 1.0;
        let (_, cn) = cov_diagnostics(Some(&m));
        let cn = cn.expect("condition_number must be Some");
        assert!(
            cn.is_infinite(),
            "condition_number must be Inf for near-singular matrix (min_ev ≤ 1e-10), got {cn}"
        );
    }

    #[test]
    fn test_cov_diagnostics_identity_covariance() {
        // Diagonal covariance → correlation matrix is identity → all eigenvalues 1.
        let m = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0]));
        let (ev, cn) = cov_diagnostics(Some(&m));
        let ev = ev.expect("eigenvalues must be Some");
        let cn = cn.expect("condition_number must be Some");
        for &e in &ev {
            assert!((e - 1.0).abs() < 1e-12, "eigenvalue must be 1.0, got {e}");
        }
        assert!(
            (cn - 1.0).abs() < 1e-12,
            "condition_number must be 1.0, got {cn}"
        );
    }

    // ── resolve_covariance_status ────────────────────────────────────────────

    #[test]
    fn cov_status_not_requested_when_step_off() {
        // When the covariance step is off, neither a (stale) covariance matrix
        // nor a fallback result can change the reported status.
        assert_eq!(
            resolve_covariance_status(false, true, true),
            CovarianceStatus::NotRequested
        );
        assert_eq!(
            resolve_covariance_status(false, false, false),
            CovarianceStatus::NotRequested
        );
    }

    #[test]
    fn cov_status_computed_takes_precedence_over_fallback() {
        // A real covariance matrix always wins, even if a fallback also ran.
        assert_eq!(
            resolve_covariance_status(true, true, false),
            CovarianceStatus::Computed
        );
        assert_eq!(
            resolve_covariance_status(true, true, true),
            CovarianceStatus::Computed
        );
    }

    #[test]
    fn cov_status_sir_fallback_when_no_matrix_but_fallback_ran() {
        // The branch the SIR-fallback wiring depends on: no H⁻¹ covariance, but
        // the |eigenvalue|-rectified SIR fallback produced a result.
        assert_eq!(
            resolve_covariance_status(true, false, true),
            CovarianceStatus::SirFallback
        );
    }

    #[test]
    fn cov_status_failed_when_requested_but_nothing_produced() {
        assert_eq!(
            resolve_covariance_status(true, false, false),
            CovarianceStatus::Failed
        );
    }
}

#[cfg(test)]
mod tests_sir_fallback {
    use super::*;
    use std::path::Path;

    // ── should_run_sir_fallback (pure gate, #264) ────────────────────────────

    #[test]
    fn sir_fallback_gate_fires_only_when_all_conditions_hold() {
        // Opted in, no real covariance, no normal SIR, proposal present.
        assert!(should_run_sir_fallback(true, false, false, true));
    }

    #[test]
    fn sir_fallback_gate_blocked_by_each_condition() {
        // Each single deviation from the firing case blocks the fallback.
        assert!(!should_run_sir_fallback(false, false, false, true)); // covariance_fallback != sir
        assert!(!should_run_sir_fallback(true, true, false, true)); // a real H⁻¹ covariance exists
        assert!(!should_run_sir_fallback(true, false, true, true)); // a normal sir=true run already produced CIs
        assert!(!should_run_sir_fallback(true, false, false, false)); // compute_covariance produced no proposal
    }

    // ── resolve_sir_fallback (gate + run_sir_core + status, #264) ─────────────

    fn warfarin_fixture() -> (
        CompiledModel,
        Population,
        ModelParameters,
        Vec<DVector<f64>>,
        DMatrix<f64>,
    ) {
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let params = model.default_params.clone();
        let eta_hats: Vec<DVector<f64>> = (0..pop.subjects.len())
            .map(|_| DVector::zeros(params.omega.dim()))
            .collect();
        // Tame fallback-style proposal: small PD diagonal in packed space, so
        // draws stay near valid parameters (positive θ/σ, PD Ω) and SIR yields
        // finite weights. A real non-PD fixture risks a wide proposal whose draws
        // overflow `exp(...)` → "all invalid weights" → status `Failed`.
        let n_packed = crate::estimation::parameterization::pack_params(&params).len();
        let proposal = DMatrix::from_diagonal(&DVector::from_element(n_packed, 0.01));
        (model, pop, params, eta_hats, proposal)
    }

    /// `resolve_sir_fallback` short-circuits to `None` (without touching the SIR
    /// machinery) when the gate declines — here because `covariance_fallback`
    /// defaults to `none`. No warning is emitted for a simple decline.
    #[test]
    fn resolve_sir_fallback_is_none_when_option_off() {
        let (model, pop, params, eta_hats, proposal) = warfarin_fixture();
        let opts = FitOptions::default(); // covariance_fallback = None
        let mut warnings = Vec::new();
        let result = resolve_sir_fallback(
            &opts,
            false,
            false,
            Some(&proposal),
            &model,
            &pop,
            &params,
            &eta_hats,
            0.0,
            &mut warnings,
        );
        assert!(
            result.is_none(),
            "fallback must not fire when covariance_fallback = none"
        );
        assert!(
            warnings.is_empty(),
            "no warning when the gate simply declines: {warnings:?}"
        );
    }

    /// End-to-end fallback wiring (#264): with `covariance_fallback = sir`, no
    /// real covariance, and a tame PD proposal (the part a real non-PD fit can't
    /// reliably deliver), `resolve_sir_fallback` runs SIR and returns a result
    /// whose θ/Ω/σ credible intervals are populated and finite — and the status
    /// the caller derives from it is `SirFallback`. Slow: a full SIR pass
    /// (sampling + per-draw population likelihood).
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: full SIR pass; opt in with --features slow-tests"
    )]
    fn resolve_sir_fallback_fires_and_yields_finite_cis() {
        let (model, pop, params, eta_hats, proposal) = warfarin_fixture();
        let mut opts = FitOptions::default();
        opts.covariance_fallback = CovarianceFallback::Sir;
        opts.verbose = false;
        opts.sir_samples = 400;
        opts.sir_resamples = 200;
        // Own the determinism explicitly rather than leaning on run_sir_core's
        // `None => fixed seed` fallback, so a future change to that fallback can't
        // silently make this sampling test flaky.
        opts.sir_seed = Some(20240612);

        let mut warnings = Vec::new();
        let result = resolve_sir_fallback(
            &opts,
            false,
            false,
            Some(&proposal),
            &model,
            &pop,
            &params,
            &eta_hats,
            // ofv_hat cancels in the SIR log-sum-exp weight normalisation, so any
            // finite value yields identical CIs — 0.0 keeps the fixture simple.
            0.0,
            &mut warnings,
        );

        // Derive the reported status from the actual outcome, *before* unwrapping,
        // so this checks the real fire→status mapping rather than a constant.
        assert_eq!(
            resolve_covariance_status(true, false, result.is_some()),
            CovarianceStatus::SirFallback
        );
        let sir = result.expect("fallback should fire and SIR should succeed with a tame proposal");

        assert!(!sir.ci_theta.is_empty(), "theta CIs must be populated");
        for (lo, hi) in sir
            .ci_theta
            .iter()
            .chain(&sir.ci_omega)
            .chain(&sir.ci_sigma)
        {
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "SIR-fallback CI must be finite and ordered, got ({lo}, {hi})"
            );
        }
        assert!(
            sir.effective_sample_size.is_finite() && sir.effective_sample_size > 0.0,
            "ESS must be finite and positive, got {}",
            sir.effective_sample_size
        );
        assert!(
            !warnings.iter().any(|w| w.contains("SIR fallback failed")),
            "no failure warning expected on the success path: {warnings:?}"
        );
    }
}

#[cfg(test)]
mod tests_param_corr {
    use super::compute_param_corr;
    use nalgebra::DMatrix;

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    /// Lognormal pair: uses the bivariate lognormal formula.
    #[test]
    fn lognormal_pair() {
        // ω = [[0.09, 0.045], [0.045, 0.09]]
        let w11 = 0.09_f64;
        let w12 = 0.045_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[true, true],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some for block omega");

        assert!(warnings.is_empty());
        // diagonal must be 1
        assert!((corr[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-12);
        // lognormal formula: (exp(w12) - 1) / sqrt((exp(w11)-1)*(exp(w11)-1))
        let expected = (w12.exp() - 1.0) / (w11.exp() - 1.0);
        assert!(
            (corr[(0, 1)] - expected).abs() < 1e-10,
            "lognormal corr {:.6} != expected {:.6}",
            corr[(0, 1)],
            expected
        );
    }

    /// Additive pair: falls back to eta-level formula (cov/sqrt(var_i*var_j)).
    #[test]
    fn additive_pair() {
        let w11 = 4.0_f64;
        let w12 = 1.0_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[false, false],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some");

        assert!(warnings.is_empty());
        let expected = w12 / w11;
        assert!((corr[(0, 1)] - expected).abs() < 1e-12);
    }

    /// Mixed pair (one lognormal, one additive) falls back to eta-level and emits a warning.
    #[test]
    fn mixed_pair_warns_and_falls_back() {
        let w11 = 0.09_f64;
        let w12 = 0.03_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[true, false],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some");

        assert_eq!(warnings.len(), 1, "expected one warning");
        assert!(warnings[0].contains("mixed"));
        // eta-level fallback
        let expected = w12 / w11;
        assert!((corr[(0, 1)] - expected).abs() < 1e-12);
    }

    /// Diagonal omega returns None (no off-diagonals to report).
    #[test]
    fn diagonal_returns_none() {
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = 0.09;
        omega[(1, 1)] = 0.04;
        let mut warnings = Vec::new();
        let result = compute_param_corr(
            &omega,
            &[true, true],
            &names(&["A", "B"]),
            "test",
            &mut warnings,
        );
        assert!(result.is_none());
        assert!(warnings.is_empty());
    }
}

#[cfg(test)]
mod simulate_with_uncertainty_tests {
    //! End-to-end smoke tests for `simulate_with_uncertainty`. The parameter
    //! sampler itself is exercised in `estimation::uncertainty_samples::tests`;
    //! these tests verify the wiring: row count, draw index range, and SIR
    //! pool reuse.

    use super::*;
    use crate::estimation::uncertainty_samples::UncertaintyMethod;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    fn tiny_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        CompiledModel {
            name: "uncertainty_smoke".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
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
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
        }
    }

    fn tiny_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0];
        let subjects: Vec<Subject> = (0..2)
            .map(|i| Subject {
                id: format!("S{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                obs_raw_times: Vec::new(),
                observations: vec![30.0, 22.0, 16.0],
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: vec![1, 1, 1],
                dose_occasions: vec![1],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// Build a synthetic `FitResult` carrying the fitted theta/Omega/Sigma
    /// from `template` plus a small identity covariance in packed log-space.
    /// Avoids invoking `fit()` (slow) while still exercising the full
    /// `simulate_with_uncertainty` wiring.
    fn synthetic_fit(template: &ModelParameters) -> FitResult {
        let n_packed = crate::estimation::parameterization::packed_len(template);
        let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
        FitResult {
            method: EstimationMethod::FoceI,
            method_chain: vec![EstimationMethod::FoceI],
            converged: true,
            ofv: 0.0,
            aic: 0.0,
            bic: 0.0,
            theta: template.theta.clone(),
            theta_names: template.theta_names.clone(),
            eta_names: template.omega.eta_names.clone(),
            omega: template.omega.matrix.clone(),
            sigma: template.sigma.values.clone(),
            sigma_names: template.sigma.names.clone(),
            error_model: ErrorModel::Proportional,
            covariance_matrix: Some(cov),
            se_theta: None,
            se_omega: None,
            se_sigma: None,
            theta_fixed: template.theta_fixed.clone(),
            omega_fixed: template.omega_fixed.clone(),
            sigma_fixed: template.sigma_fixed.clone(),
            omega_init_as_sd: vec![false; template.omega.matrix.nrows()],
            sigma_init_as_sd: vec![false; template.sigma.values.len()],
            subjects: vec![],
            n_obs: 6,
            n_subjects: 2,
            n_parameters: n_packed,
            n_iterations: 0,
            interaction: true,
            warnings: vec![],
            warnings_structured: vec![],
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            impmap_trace: None,
            bayes: None,
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
            gradient_method_inner: String::new(),
            gradient_method_outer: String::new(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 1,
            nlopt_missing_algorithms: vec![],
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: CovarianceStatus::Computed,
            shrinkage_eta: vec![],
            cond_dist: None,
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: String::new(),
            ferx_version: String::new(),
            eta_param_info: vec![],
            theta_transform: vec![],
            sigma_types: vec![],
            cov_eigenvalues: None,
            cov_condition_number: None,
            eta_log_transformed: vec![],
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            model_text: None,
            theta_init: template.theta.clone(),
            omega_init: template.omega.matrix.clone(),
            sigma_init: template.sigma.values.clone(),
            obs_time_range: None,
            final_gradient: None,
            optimizer: "bobyqa".to_string(),
            n_starts: 1,
            multi_start_seed: None,
            saem_seed: None,
            sir_seed: None,
            imp_seed: None,
            npde_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 0,
            outer_gtol: 0.0,
            inits_from_nca: None,
            covariate_names: Vec::new(),
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
            exclusions: None,
        }
    }

    #[test]
    fn asymptotic_row_count_and_draw_range() {
        let model = tiny_model();
        let pop = tiny_population();
        let fit = synthetic_fit(&model.default_params);

        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 3,
            n_sim_per_draw: 2,
            method: UncertaintyMethod::Asymptotic,
            seed: Some(7),
        };
        let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
        // 3 draws * 2 sims * 2 subjects * 3 obs = 36 rows
        assert_eq!(rows.len(), 36);

        let mut draws: Vec<usize> = rows.iter().map(|r| r.draw).collect();
        draws.sort();
        draws.dedup();
        assert_eq!(draws, vec![1, 2, 3]);

        let mut sims: Vec<usize> = rows.iter().map(|r| r.sim).collect();
        sims.sort();
        sims.dedup();
        assert_eq!(sims, vec![1, 2]);
    }

    #[test]
    fn legacy_simulate_emits_draw_one() {
        // The original simulate() path should tag every row with draw = 1,
        // preserving a sensible default for callers that don't propagate
        // parameter uncertainty.
        let model = tiny_model();
        let pop = tiny_population();
        let rows = simulate_with_seed(&model, &pop, &model.default_params, 2, 42);
        assert!(rows.iter().all(|r| r.draw == 1));
    }

    #[test]
    fn compute_npde_npd_shapes_finite_and_reproducible() {
        // Drives the full post-fit NPDE/NPD simulation path on a real (model,
        // population, params). nsim > n_obs (3) so the decorrelated NPDE is
        // non-NaN, and a fixed seed must reproduce bit-for-bit.
        let model = tiny_model();
        let pop = tiny_population();
        let nsim = 200;

        let a = crate::stats::npde::compute_npde_npd(
            &model,
            &pop,
            &model.default_params,
            nsim,
            Some(7),
        );
        let b = crate::stats::npde::compute_npde_npd(
            &model,
            &pop,
            &model.default_params,
            nsim,
            Some(7),
        );

        assert_eq!(a.len(), pop.subjects.len());
        for (sn, subj) in a.iter().zip(pop.subjects.iter()) {
            assert_eq!(sn.npd.len(), subj.observations.len());
            assert_eq!(sn.npde.len(), subj.observations.len());
            assert!(sn.npd.iter().all(|v| v.is_finite()), "NPD finite");
            assert!(sn.npde.iter().all(|v| v.is_finite()), "NPDE finite");
        }
        // Reproducible across calls with the same seed.
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.npd, sb.npd);
            assert_eq!(sa.npde, sb.npde);
        }
    }

    #[test]
    fn fit_rejects_ltbs_with_proportional_error() {
        // Defensive guard against hand-built `CompiledModel`s: the parser already
        // rejects this combination, but a Rust caller flipping `log_transform = true`
        // on a model with proportional/combined error would silently mis-fit (the
        // prediction is log-wrapped but the variance still expects natural-scale f).
        let mut model = tiny_model(); // tiny_model uses Proportional error
        model.log_transform = true;
        let pop = tiny_population();
        let opts = FitOptions::default();
        let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
        assert!(
            err.contains("LTBS") && err.contains("Additive"),
            "expected LTBS+proportional rejection, got: {err}"
        );
    }

    #[test]
    fn sir_path_reuses_pool() {
        let model = tiny_model();
        let pop = tiny_population();
        let mut fit = synthetic_fit(&model.default_params);
        // Build a 4-element resample pool: small perturbations of x_hat in
        // packed log-space. Tests will sample with replacement from this pool.
        let x_hat = crate::estimation::parameterization::pack_params(&model.default_params);
        let pool: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                let mut xk = x_hat.clone();
                xk[0] += 0.005 * (k as f64);
                xk
            })
            .collect();
        fit.sir_resamples_packed = Some(pool);

        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 5,
            n_sim_per_draw: 1,
            method: UncertaintyMethod::Sir,
            seed: Some(11),
        };
        let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
        // 5 draws * 1 sim * 2 subjects * 3 obs = 30 rows
        assert_eq!(rows.len(), 30);
    }

    #[test]
    fn asymptotic_errors_without_covariance_step() {
        let model = tiny_model();
        let pop = tiny_population();
        let mut fit = synthetic_fit(&model.default_params);
        fit.covariance_matrix = None;
        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 2,
            n_sim_per_draw: 1,
            method: UncertaintyMethod::Asymptotic,
            seed: Some(0),
        };
        let err = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap_err();
        assert!(err.contains("covariance"));
    }
}

// ── SDE end-to-end integration ───────────────────────────────────────────────

#[cfg(test)]
mod sde_integration {
    use super::fit;
    use crate::parser::model_parser::parse_full_model;
    use crate::types::*;
    use std::collections::HashMap;

    /// 1-cpt IV ODE model with a [diffusion] block on the central compartment.
    /// Sigma (ADD) is fixed so that the diffusion parameter must absorb residual
    /// variance. We verify:
    ///   (a) uses_sde = true
    ///   (b) DIFF_CENTRAL is estimated positive
    ///   (c) OFV is finite and the fit converges
    ///   (d) OFV with diffusion <= OFV without diffusion (diffusion can only help)
    const SDE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
  central ~ 0.5

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    /// Same model without the [diffusion] block (for OFV comparison).
    const BASE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    fn make_sde_population() -> Population {
        // 4 subjects, single IV bolus dose=100 at t=0, observations at 3 times.
        // The ODE `d/dt(central) = -(CL/V) * central` describes the amount in
        // the central compartment (mg) — ferx adds `dose.amt` directly to the
        // state, so for an IV bolus the state IS the dose in amount units.
        // Observations must therefore also be in amount (mg), not concentration.
        // True amounts from a 1-cpt model with CL=5, V=50 (k = 0.1/h):
        //   t=1: A(t) = 100·exp(-0.1) = 90.48
        //   t=4: A(t) = 100·exp(-0.4) = 67.03
        //   t=8: A(t) = 100·exp(-0.8) = 44.93
        // Values below are symmetric ±5% perturbations of the true amounts
        // (two subjects below, two above) so the population sample remains
        // centered on the analytical trajectory.
        let obs_times = vec![1.0, 4.0, 8.0];
        let dvs: &[(&str, Vec<f64>)] = &[
            // -5% across all times
            ("S1", vec![85.96, 63.68, 42.68]),
            // +5% across all times
            ("S2", vec![95.00, 70.38, 47.18]),
            // -3% across all times
            ("S3", vec![87.77, 65.02, 43.58]),
            // +3% across all times
            ("S4", vec![93.19, 69.04, 46.28]),
        ];
        let subjects = dvs
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                obs_raw_times: Vec::new(),
                observations: obs.clone(),
                obs_cmts: vec![1; 3],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 3],
                occasions: vec![1u32; 3],
                dose_occasions: vec![1u32],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn fast_foce_opts() -> FitOptions {
        FitOptions {
            method: EstimationMethod::Foce,
            methods: Vec::new(),
            outer_maxiter: 80,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: false,
            mu_referencing: false,
            optimizer: Optimizer::Slsqp,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    #[test]
    fn test_sde_fit_smoke() {
        // Combined smoke test: one SDE fit, three assertions. Each EKF FOCE
        // fit takes ~30–50 min on the 2-core CI runner, so the previous
        // 3-tests-1-assertion split tripled CI wall for no extra coverage.
        let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let pop = make_sde_population();
        let opts = fast_foce_opts();
        let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
            .expect("SDE fit should succeed");
        assert!(result.uses_sde, "uses_sde must be true");
        assert!(
            result.ofv.is_finite(),
            "OFV must be finite, got {}",
            result.ofv
        );
        let diff_idx = result
            .theta_names
            .iter()
            .position(|n| n == "DIFF_CENTRAL")
            .expect("DIFF_CENTRAL must be in theta_names");
        let diff_val = result.theta[diff_idx];
        assert!(
            diff_val > 0.0,
            "DIFF_CENTRAL must be positive, got {diff_val}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_sde_ofv_le_base_ofv() {
        // Reference: the OFV from the identical model fit without the
        // [diffusion] block (BASE_MODEL_SRC). Since [diffusion] adds an extra
        // free parameter (DIFF_CENTRAL ≥ 0) and the EKF observation variance
        // collapses to the residual-only variance when DIFF_CENTRAL → 0, the
        // SDE OFV must be ≤ the base OFV at the optimum.
        // The +1 unit of slack absorbs numerical noise from finite-difference
        // gradients, NLopt's stopping tolerance (`outer_gtol = 1e-3`), and the
        // truncated `outer_maxiter = 80` cap in `fast_foce_opts`; without the
        // slack we'd flake on iterations where the SDE fit stopped a hair
        // short of the base fit's OFV.
        let pop = make_sde_population();
        let opts = fast_foce_opts();

        let parsed_base = parse_full_model(BASE_MODEL_SRC).expect("base model should parse");
        let base_result = fit(
            &parsed_base.model,
            &pop,
            &parsed_base.model.default_params,
            &opts,
        )
        .expect("base fit should succeed");

        let parsed_sde = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let sde_result = fit(
            &parsed_sde.model,
            &pop,
            &parsed_sde.model.default_params,
            &opts,
        )
        .expect("SDE fit should succeed");

        assert!(
            sde_result.ofv <= base_result.ofv + 1.0,
            "SDE OFV ({}) should not be worse than base OFV ({}) by more than 1 unit",
            sde_result.ofv,
            base_result.ofv,
        );
    }

    /// SDE + gn / gn_hybrid must fail with a clear error message.
    #[test]
    fn sde_gn_returns_error() {
        use crate::types::EstimationMethod;

        let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let pop = {
            // Minimal single-subject population (no data needed — error fires before fitting).
            use crate::types::{DoseEvent, Population, Subject};
            let subj = Subject {
                id: "1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0],
                obs_raw_times: Vec::new(),
                observations: vec![1.0],
                obs_cmts: vec![1],
                covariates: HashMap::new(),
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
                obs_records: vec![],
            };
            Population {
                subjects: vec![subj],
                covariate_names: Vec::new(),
                dv_column: "DV".into(),
                input_columns: vec![],
                exclusions: None,
                warnings: vec![],
            }
        };

        for method in [EstimationMethod::FoceGn, EstimationMethod::FoceGnHybrid] {
            let opts = FitOptions {
                method,
                ..FitOptions::default()
            };
            let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts);
            assert!(result.is_err(), "expected error for {:?} + SDE", method);
            let msg = result.unwrap_err();
            assert!(
                msg.contains("gn") || msg.contains("gn_hybrid"),
                "error message should mention gn: {msg}"
            );
        }
    }

    /// Issue #175: an SDE ([diffusion]) model must surface the experimental
    /// feature warning, classified into the `experimental` category. The check
    /// is data-independent (`check_experimental_features` takes only the model),
    /// so `ferx check` reports it even without a `--data` file. Fast — no fit.
    #[test]
    fn sde_emits_experimental_warning() {
        let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let diags = super::check_experimental_features(&parsed.model);
        let exp = diags
            .iter()
            .find(|d| d.code == "W_EXPERIMENTAL_SDE")
            .expect("SDE model should emit W_EXPERIMENTAL_SDE");
        assert_eq!(exp.severity, crate::diagnostics::Severity::Warning);
        assert_eq!(
            crate::types::classify_warning(&exp.message).category,
            "experimental"
        );

        // Sanity: a non-SDE model must NOT emit the experimental warning.
        let base = parse_full_model(BASE_MODEL_SRC).expect("base model should parse");
        assert!(
            super::check_experimental_features(&base.model)
                .iter()
                .all(|d| d.code != "W_EXPERIMENTAL_SDE"),
            "non-SDE model should not emit W_EXPERIMENTAL_SDE"
        );
    }
}

#[cfg(test)]
mod multi_start_tests {
    use super::perturb_init;
    use crate::estimation::parameterization::theta_packs_log;
    use crate::types::{FitOptions, ModelParameters, OmegaMatrix, SigmaVector};

    fn make_params(
        theta: Vec<f64>,
        theta_lower: Vec<f64>,
        theta_upper: Vec<f64>,
    ) -> ModelParameters {
        let n = theta.len();
        ModelParameters {
            theta,
            theta_names: (0..n).map(|i| format!("T{i}")).collect(),
            theta_lower,
            theta_upper,
            theta_fixed: vec![false; n],
            omega: OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        }
    }

    #[test]
    fn test_perturb_start0_is_identity() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        let perturbed = perturb_init(&p, 0, 0.5, 42);
        assert_eq!(perturbed.theta, p.theta);
    }

    #[test]
    fn test_perturb_changes_theta() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        let perturbed = perturb_init(&p, 1, 0.3, 42);
        // With sigma=0.3 and seed=43 (42+1), at least one theta should differ
        let changed = perturbed
            .theta
            .iter()
            .zip(p.theta.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "start 1 should perturb theta");
    }

    #[test]
    fn test_perturb_stays_in_bounds() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        for k in 1..=10 {
            let perturbed = perturb_init(&p, k, 2.0, 42); // large sigma to stress-test bounds
            for (i, &t) in perturbed.theta.iter().enumerate() {
                assert!(
                    t >= p.theta_lower[i],
                    "start {k}: theta[{i}]={t} < lower={}",
                    p.theta_lower[i]
                );
                assert!(
                    t <= p.theta_upper[i],
                    "start {k}: theta[{i}]={t} > upper={}",
                    p.theta_upper[i]
                );
            }
        }
    }

    #[test]
    fn test_perturb_identity_packed_theta() {
        // theta_lower < 0 → identity packing → additive perturbation
        let p = make_params(vec![0.5], vec![-5.0], vec![5.0]);
        assert!(!theta_packs_log(p.theta_lower[0]));
        let perturbed = perturb_init(&p, 1, 0.3, 99);
        assert!(perturbed.theta[0] >= -5.0 && perturbed.theta[0] <= 5.0);
    }

    #[test]
    fn test_n_starts_option_parsed() {
        let mut opts = FitOptions::default();
        assert_eq!(opts.n_starts, 1);
        opts.n_starts = 4;
        assert_eq!(opts.n_starts, 4);
    }

    #[test]
    fn test_n_starts_and_seed_via_parser() {
        use crate::parser::model_parser::apply_fit_option;
        let mut opts = FitOptions::default();
        apply_fit_option(&mut opts, "n_starts", "4").expect("n_starts parses");
        assert_eq!(opts.n_starts, 4);
        apply_fit_option(&mut opts, "multi_start_seed", "123").expect("multi_start_seed parses");
        assert_eq!(opts.multi_start_seed, Some(123));
        apply_fit_option(&mut opts, "start_sigma", "0.5").expect("start_sigma parses");
        assert!((opts.start_sigma - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_per_start_saem_seed_derivation() {
        let base: u64 = 12345;
        // Each start k > 0 gets base + k; start 0 keeps the base unchanged.
        assert_eq!(base.wrapping_add(0), 12345);
        assert_eq!(base.wrapping_add(1), 12346);
        assert_eq!(base.wrapping_add(7), 12352);
        // All derived seeds are distinct.
        let seeds: Vec<u64> = (0..8).map(|k| base.wrapping_add(k)).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(unique.len(), 8);
        // wrapping_add is defined at u64::MAX.
        assert_eq!(u64::MAX.wrapping_add(1), 0);
    }
}

#[cfg(test)]
mod tests_sdtab_tv_cov {
    use super::*;
    use crate::types::{
        BloqMethod, CompiledModel, DoseEvent, ErrorModel, ErrorSpec, GradientMethod,
        ModelParameters, OmegaMatrix, PkModel, PkParams, Population, ScalingSpec, SigmaVector,
        Subject,
    };
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    /// Regression: on a subject with time-varying covariates the sdtab IPRED
    /// (`SubjectResult.ipred`) must come from the **TV-aware** prediction path
    /// — `compute_predictions_with_tv` — so it agrees with the IPRED used by
    /// the FOCEI marginal during the fit.
    ///
    /// Before this fix, `compute_subject_results` called `model_preds` with a
    /// single per-subject `pk_params` derived from `subject.covariates`. For
    /// TV-covariate subjects that silently used the wrong covariate snapshot
    /// for every observation, producing sdtab IPREDs that drifted to ~0 after
    /// the first dose and inflated IWRES into 30+. The OFV was fine because
    /// the FOCEI marginal already routed through `compute_predictions_with_tv`
    /// — only the post-fit diagnostic path was broken.
    ///
    /// This test constructs a minimal 1-cpt IV bolus model where `pk_param_fn`
    /// reads `WT` to scale CL, and a subject whose `obs_covariates` carry
    /// `WT = [70, 140, 210]` (vs `subject.covariates["WT"] = 70`). The TV
    /// path gives a strictly different concentration profile from the no-TV
    /// path, so the assertion `sdtab IPRED == compute_predictions_with_tv`
    /// fails *loudly* if the dispatch ever regresses to `model_preds`.
    #[test]
    fn test_sdtab_ipred_honours_tv_covariates() {
        // ── Minimal CompiledModel: 1-cpt IV bolus, CL scaled by per-event WT ──
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0], // TVCL = 5, TVV = 50
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "tv_cov_sdtab_regression".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            // CL = TVCL · exp(η_CL) · (WT/70) — reads WT from the covariate map
            // that `compute_predictions_with_tv` substitutes per-event from
            // `obs_covariates` / `dose_covariates`. With WT changing per obs
            // the TV path produces a profile that the (broken) no-TV path
            // can't match.
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * eta[0].exp() * (wt / 70.0);
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params: default_params.clone(),
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["WT".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
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
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
        };

        // Subject with TV WT: subject.covariates["WT"] = 70 (the no-TV snapshot)
        // but obs_covariates have WT = [70, 140, 210] at the three observation
        // times. dose_covariates set to the same WT=70 the dose-time snapshot
        // would carry.
        let mut subj_cov = HashMap::new();
        subj_cov.insert("WT".to_string(), 70.0);
        let mut obs_covs: Vec<HashMap<String, f64>> = Vec::new();
        for wt in [70.0_f64, 140.0, 210.0] {
            let mut m = HashMap::new();
            m.insert("WT".to_string(), wt);
            obs_covs.push(m);
        }
        let mut dose_covs: Vec<HashMap<String, f64>> = Vec::new();
        let mut m = HashMap::new();
        m.insert("WT".to_string(), 70.0);
        dose_covs.push(m);
        let subject = Subject {
            id: "TVS".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0],
            obs_raw_times: Vec::new(),
            observations: vec![10.0, 5.0, 2.5], // placeholders; values don't matter for the IPRED check
            obs_cmts: vec![1, 1, 1],
            covariates: subj_cov,
            dose_covariates: dose_covs,
            obs_covariates: obs_covs,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // Sanity: this subject must be classified TV — that's the regime the
        // bug lived in.
        assert!(
            subject.has_tv_covariates(),
            "test setup wrong: subject must have TV covariates"
        );

        let population = Population {
            subjects: vec![subject.clone()],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        // Fixed EBE at η = 0; H matrix is irrelevant for the IPRED check but
        // must be the right shape for CWRES not to panic.
        let eta_hats = vec![DVector::from_vec(vec![0.0])];
        let h_matrices = vec![DMatrix::from_element(3, 1, 0.5)];
        let kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new()];

        // Reference IPRED: the TV-aware dispatcher the FOCEI marginal uses.
        let ipred_reference = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &default_params.theta,
            eta_hats[0].as_slice(),
        );
        // Sanity: the TV path must NOT collapse to the no-TV path here. If
        // both paths produced the same IPRED, this regression test would
        // trivially pass even if the dispatch in `compute_subject_results`
        // regressed to `model_preds` — so we verify the TV vs no-TV gap is
        // visible before relying on the equality assertion below.
        let pk_no_tv = (model.pk_param_fn)(
            &default_params.theta,
            eta_hats[0].as_slice(),
            &subject.covariates,
        );
        let ipred_no_tv = model_preds(
            &model,
            &subject,
            &pk_no_tv,
            &default_params.theta,
            eta_hats[0].as_slice(),
        );
        let gap: f64 = ipred_reference
            .iter()
            .zip(ipred_no_tv.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            gap > 1e-3,
            "test setup wrong: TV and no-TV IPRED paths must differ noticeably \
             for this regression test to mean anything; got gap = {gap}, \
             ipred_tv = {ipred_reference:?}, ipred_no_tv = {ipred_no_tv:?}"
        );

        // The actual regression assertion: `compute_subject_results` IPRED
        // must equal the TV-aware reference. If the dispatch ever falls back
        // to `model_preds` it will be `ipred_no_tv` instead — failure here.
        let results = compute_subject_results(
            &model,
            &population,
            &default_params,
            &eta_hats,
            &h_matrices,
            &kappas,
            true,
        );
        assert_eq!(results.len(), 1);
        let sdtab_ipred = &results[0].ipred;
        assert_eq!(sdtab_ipred.len(), 3);
        for (j, (&got, &expected)) in sdtab_ipred.iter().zip(ipred_reference.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "sdtab IPRED at obs {j} = {got}, expected (TV-aware) {expected} \
                 — `compute_subject_results` must route IPRED through \
                 `compute_predictions_with_tv` for TV-covariate subjects"
            );
        }

        // Regression for #456: the sdtab population PRED column (f at η = 0) must
        // ALSO route through the TV-aware predictor, so it honours the per-event
        // covariate snapshot and agrees with the public `predict()` output. Before
        // the fix it used `model_preds`, which reads the single `subject.covariates`
        // snapshot for every observation and so silently ignored the TV breakpoints.
        let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
        let pred_reference = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &default_params.theta,
            &zero_eta,
        );
        let pk_pred_no_tv =
            (model.pk_param_fn)(&default_params.theta, &zero_eta, &subject.covariates);
        let pred_no_tv = model_preds(
            &model,
            &subject,
            &pk_pred_no_tv,
            &default_params.theta,
            &zero_eta,
        );
        let pred_gap: f64 = pred_reference
            .iter()
            .zip(pred_no_tv.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            pred_gap > 1e-3,
            "test setup wrong: TV and no-TV PRED paths must differ noticeably; \
             got gap = {pred_gap}"
        );
        let sdtab_pred = &results[0].pred;
        assert_eq!(sdtab_pred.len(), 3);
        for (j, (&got, &expected)) in sdtab_pred.iter().zip(pred_reference.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "sdtab PRED at obs {j} = {got}, expected (TV-aware) {expected} \
                 — `compute_subject_results` must route PRED through \
                 `compute_predictions_with_tv` for TV-covariate subjects"
            );
        }
    }

    #[test]
    fn test_sdtab_iwres_uses_block_sigma_correlation() {
        use crate::parser::model_parser::parse_full_model;

        let src = r#"
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.0] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;
        let model = parse_full_model(src).expect("model parses").model;
        assert_eq!(model.residual_correlations.len(), 1);

        let mut subject = Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let eta = DVector::from_vec(vec![0.0]);
        let ipred = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            eta.as_slice(),
        );
        subject.observations[0] = ipred[0] + 2.0;

        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let eta_hats = vec![eta];
        let h_matrices = vec![DMatrix::zeros(1, 1)];
        let kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new()];

        let results = compute_subject_results(
            &model,
            &population,
            &model.default_params,
            &eta_hats,
            &h_matrices,
            &kappas,
            false,
        );

        let f = ipred[0];
        let expected_v = (f * 0.2).powi(2) + 1.0 + 2.0 * f * 0.5 * 0.2;
        let expected_iwres = 2.0 / expected_v.sqrt();
        assert!(
            (results[0].iwres[0] - expected_iwres).abs() < 1e-12,
            "sdtab IWRES must include block_sigma cross term; got {}, expected {}",
            results[0].iwres[0],
            expected_iwres
        );
    }

    /// Regression for #506: simulation must honour time-varying covariate
    /// snapshots on observation rows. The pre-fix simulation path evaluated
    /// `pk_param_fn` once with `subject.covariates`, so changing `WT` on obs rows
    /// had no effect on emitted IPRED/DV unless the baseline dose row changed.
    #[test]
    fn test_simulate_honours_tv_covariates() {
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega: OmegaMatrix::from_diagonal(&[], vec![]),
            omega_fixed: Vec::new(),
            sigma: SigmaVector {
                values: vec![0.0],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "tv_cov_sim_regression".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * (wt / 70.0);
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 0,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: Vec::new(),
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params: default_params.clone(),
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: Vec::new(),
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: Vec::new(),
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["WT".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
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
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
        };

        let mut baseline_cov = HashMap::new();
        baseline_cov.insert("WT".to_string(), 70.0);

        let dose_covariates = vec![HashMap::from([("WT".to_string(), 70.0)])];
        let obs_covariates = vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 140.0)]),
            HashMap::from([("WT".to_string(), 210.0)]),
        ];
        let subject = Subject {
            id: "SIMTV".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0, 0.0, 0.0],
            obs_cmts: vec![1, 1, 1],
            covariates: baseline_cov,
            dose_covariates,
            obs_covariates,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(subject.has_tv_covariates());

        let reference =
            crate::pk::compute_predictions_with_tv(&model, &subject, &default_params.theta, &[]);
        let pk_no_tv = (model.pk_param_fn)(&default_params.theta, &[], &subject.covariates);
        let no_tv = model_preds(&model, &subject, &pk_no_tv, &default_params.theta, &[]);
        let gap: f64 = reference
            .iter()
            .zip(no_tv.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            gap > 1e-3,
            "test setup wrong: TV and baseline-only simulation paths must differ; \
             got gap={gap}, tv={reference:?}, no_tv={no_tv:?}"
        );

        let population = Population {
            subjects: vec![subject],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let rows = simulate_with_seed(&model, &population, &default_params, 1, 506);
        assert_eq!(rows.len(), reference.len());
        for (j, (row, &expected)) in rows.iter().zip(reference.iter()).enumerate() {
            assert!(
                (row.ipred - expected).abs() < 1e-12,
                "simulation IPRED at obs {j} = {}, expected TV-aware {expected}",
                row.ipred
            );
        }
    }
}

#[cfg(test)]
mod tests_derived_session_clock {
    //! Tests for the EVID=3/4 session-clock fixes in `compute_extra_output_columns`.
    //!
    //! All tests build a two-session subject whose second occasion has raw TIME
    //! restarting from 0 (identical to the first session) but whose internal
    //! `obs_times` are shifted so that ferx-core's monotonic timeline is
    //! maintained.  The fixes ensure that `[derived]` columns see raw TIME,
    //! not the shifted internal clock.

    use super::*;
    use crate::types::{
        AggFunction, BloqMethod, CompiledModel, DerivedContext, DerivedExprSpec, DerivedKind,
        ErrorModel, ErrorSpec, GradientMethod, IndivParamPartials, IntegralStep, IntegralWindow,
        ModelParameters, OmegaMatrix, PkModel, PkParams, Population, ScalingSpec, SigmaVector,
        Subject,
    };
    use nalgebra::DVector;
    use std::collections::HashMap;

    // ── shared helpers ────────────────────────────────────────────────────────

    /// Minimal CompiledModel — 1-cpt IV, returns constant PK params, no LTO.
    /// Caller supplies `derived_exprs`.
    fn minimal_model(derived_exprs: Vec<DerivedExprSpec>) -> CompiledModel {
        CompiledModel {
            name: "test_session".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Additive,
            error_spec: ErrorSpec::Single(ErrorModel::Additive),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|_, _, _| PkParams::default()),
            n_theta: 0,
            n_eta: 0,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: Vec::new(),
            eta_names: Vec::new(),
            indiv_param_names: Vec::new(),
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: Vec::new(),
                theta_names: Vec::new(),
                theta_lower: Vec::new(),
                theta_upper: Vec::new(),
                theta_fixed: Vec::new(),
                omega: OmegaMatrix::from_diagonal(&[], vec![]),
                omega_fixed: Vec::new(),
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: Some(Box::new(|_t, _c| vec![])),
            pk_indices: Vec::new(),
            eta_map: Vec::new(),
            pk_idx_f64: Vec::new(),
            sel_flat: Vec::new(),
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs,
            output_columns: Vec::new(),
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
        }
    }

    /// Build a two-session subject whose second occasion has raw TIME restarting
    /// from 0.
    ///
    /// Session 0: raw [0, 1, 4]  →  internal [0, 1, 4]
    /// Session 1: raw [0, 1, 4]  →  internal [5, 6, 9]  (shift = 5)
    ///
    /// `reset_times[0] = 5.0` marks the boundary.
    fn two_session_subject() -> Subject {
        Subject {
            id: "S1".into(),
            doses: Vec::new(),
            // Session 0 at 0,1,4 — Session 1 shifted by 5 to 5,6,9
            obs_times: vec![0.0, 1.0, 4.0, 5.0, 6.0, 9.0],
            obs_raw_times: vec![0.0, 1.0, 4.0, 0.0, 1.0, 4.0],
            observations: vec![1.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: vec![5.0], // boundary at shifted t=5
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Minimal SubjectResult for a subject with `n_obs` observations and η=[] .
    fn sr_for(n_obs: usize) -> SubjectResult {
        SubjectResult {
            id: "S1".into(),
            eta: DVector::from_vec(vec![]),
            ipred: vec![1.0; n_obs],
            pred: vec![1.0; n_obs],
            iwres: vec![0.0; n_obs],
            cwres: vec![0.0; n_obs],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![0; n_obs],
            n_obs,
            extra_columns: Vec::new(),
            per_obs_tad: Vec::new(),
            compartment_states: Vec::new(),
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// PerRow `[derived]` column must expose raw dataset TIME, not the internal
    /// shifted clock.  For a two-session subject the second session's raw times
    /// [0, 1, 4] are identical to the first session's; the shifted times are
    /// [5, 6, 9].  If the fix is correct the column values are [0,1,4,0,1,4].
    #[test]
    fn derived_per_row_time_is_raw_clock() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "T".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx: &DerivedContext| ctx.time),
            },
            uses_compartments: false,
        }];
        let model = minimal_model(derived_exprs);
        let subject = two_session_subject();
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let mut subjects_results = vec![sr_for(6)];
        compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results);
        let col = &subjects_results[0].extra_columns[0].1;
        let expected = vec![0.0, 1.0, 4.0, 0.0, 1.0, 4.0];
        for (j, (&got, &exp)) in col.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "PerRow TIME at obs {j}: got {got}, expected {exp} (raw clock)"
            );
        }
    }

    /// Aggregate Tmax must return raw TIME.  With the two-session subject and
    /// ipred = [1,2,3,3,2,1] (session 0 peaks at raw t=4, session 1 peaks at
    /// raw t=0 which is the third session-1 obs) the AggFunction::Tmax over all
    /// rows should return the raw time of the global IPRED maximum.
    ///
    /// The global max IPRED value is 3 at index j=2 (raw t=4, shifted t=4) and
    /// also at j=3 (raw t=0, shifted t=5).  The first maximum encountered is
    /// j=2 with raw t=4, not shifted t=4 or t=5 — both agree here so this test
    /// verifies the raw path doesn't regress.  A harder variant follows.
    #[test]
    fn derived_aggregate_tmax_returns_raw_time() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "TMAX".into(),
            kind: DerivedKind::Aggregate {
                func: AggFunction::Tmax,
                value: Box::new(|ctx: &DerivedContext| ctx.ipred),
                filter: None,
            },
            uses_compartments: false,
        }];
        let model = minimal_model(derived_exprs);
        let subject = two_session_subject();
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let mut sr = sr_for(6);
        // ipred peak is at j=4 (shifted t=6, raw t=1) which should give tmax=1.
        sr.ipred = vec![1.0, 2.0, 1.5, 0.5, 3.0, 1.0];
        let mut subjects_results = vec![sr];
        compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results);
        let col = &subjects_results[0].extra_columns[0].1;
        // All entries should be 1.0 (raw time of peak at j=4).
        for &v in col {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "Tmax should be raw time 1.0, got {v}"
            );
        }
    }

    /// Obs-based integral over explicit window [0, 4] must produce the correct
    /// per-session AUC for a two-session (EVID=4-like) subject.
    ///
    /// Both sessions have raw times [0, 1, 4].  Integrand = ctx.time.
    /// AUC = trapezoid([(0,0),(1,1),(4,4)]) = 0·Δt₁ + (0+1)/2·1 + (1+4)/2·3 = 0.5 + 7.5 = 8.0
    ///
    /// With the old (broken) code, session 1's shifted times [5,6,9] would all
    /// fail the window filter [0,4] → NaN for every session-1 row.
    #[test]
    fn derived_integral_obs_per_session_explicit_window() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "AUC".into(),
            kind: DerivedKind::Integral {
                integrand: Box::new(|ctx: &DerivedContext| ctx.time),
                condition: None,
                data_based: true,
                uses_compartments: false,
                window: IntegralWindow::Explicit { from: 0.0, to: 4.0 },
                step: IntegralStep::ObsTimes,
            },
            uses_compartments: false,
        }];
        let model = minimal_model(derived_exprs);
        let subject = two_session_subject();
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let mut subjects_results = vec![sr_for(6)];
        compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results);
        let col = &subjects_results[0].extra_columns[0].1;
        // Expected AUC = 8.0 for every row in each session.
        for (j, &v) in col.iter().enumerate() {
            assert!(
                (v - 8.0).abs() < 1e-12,
                "Integral obs j={j}: got {v}, expected 8.0 (per-session raw-clock AUC)"
            );
        }
    }

    /// Periodic integral aligns windows to raw TIME, not shifted time.
    ///
    /// Period=5, anchor=0.  All raw obs at [0, 1, 4] satisfy floor(t/5)=0, so
    /// every obs lands in the first period window [0, 5).  All three per-session
    /// points contribute → AUC = trapezoid([(0,0),(1,1),(4,4)]) = 8.0.
    ///
    /// With the old (broken) code, session-1 obs at shifted times [5, 6, 9] give
    /// floor(t/5) = 1 → window [5, 10).  Integrating (5,5),(6,6),(9,9) yields 28.0,
    /// not 8.0 — a clear mismatch caught by the `v == 8.0` assertion.
    ///
    /// After the fix, session-1 obs use raw t ∈ {0, 1, 4} → floor(t/5) = 0 →
    /// window [0, 5) → correct AUC = 8.0.
    #[test]
    fn derived_integral_periodic_uses_raw_clock() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "AUC_TAU".into(),
            kind: DerivedKind::Integral {
                integrand: Box::new(|ctx: &DerivedContext| ctx.time),
                condition: None,
                data_based: true,
                uses_compartments: false,
                window: IntegralWindow::Periodic {
                    period: 5.0,
                    anchor: 0.0,
                },
                step: IntegralStep::ObsTimes,
            },
            uses_compartments: false,
        }];
        let model = minimal_model(derived_exprs);
        let subject = two_session_subject();
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let mut subjects_results = vec![sr_for(6)];
        compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results);
        let col = &subjects_results[0].extra_columns[0].1;
        // All obs land in the raw-clock window [0,5); all three per-session
        // points contribute → AUC=8.0 for every row.
        for (j, &v) in col.iter().enumerate() {
            assert!(
                (v - 8.0).abs() < 1e-12,
                "Periodic integral at obs {j}: got {v}, expected 8.0"
            );
        }
    }

    /// Single-session subjects are unaffected by the multi-session path.
    ///
    /// A plain subject with no resets should produce the same AUC as before
    /// (regression guard).
    #[test]
    fn derived_integral_single_session_unchanged() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "AUC".into(),
            kind: DerivedKind::Integral {
                integrand: Box::new(|ctx: &DerivedContext| ctx.time),
                condition: None,
                data_based: true,
                uses_compartments: false,
                window: IntegralWindow::Explicit { from: 0.0, to: 4.0 },
                step: IntegralStep::ObsTimes,
            },
            uses_compartments: false,
        }];
        let model = minimal_model(derived_exprs);
        let subject = Subject {
            id: "SINGLE".into(),
            doses: Vec::new(),
            obs_times: vec![0.0, 1.0, 4.0],
            obs_raw_times: vec![0.0, 1.0, 4.0],
            observations: vec![1.0; 3],
            obs_cmts: vec![1; 3],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 3],
            occasions: vec![1, 1, 1],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let mut subjects_results = vec![sr_for(3)];
        compute_extra_output_columns(&model, &population, &[], &[], &mut subjects_results);
        let col = &subjects_results[0].extra_columns[0].1;
        // AUC = trapezoid([(0,0),(1,1),(4,4)]) = 8.0
        for &v in col {
            assert!(
                (v - 8.0).abs() < 1e-12,
                "Single-session AUC should be 8.0, got {v}"
            );
        }
    }
}

#[cfg(test)]
mod tests_derived_iov_kappa {
    //! Regression tests for issue #238: `compute_extra_output_columns` must
    //! thread each observation's **occasion** kappa into `pk_param_fn` and into
    //! `DerivedContext.eta`, instead of silently using a BSV-only eta vector
    //! (kappas → 0). Both the individual-parameter map (driving `[derived]`
    //! expressions and `[output]` columns) and `ctx.eta` are checked.

    use super::*;
    use crate::types::{
        BloqMethod, CompiledModel, DerivedContext, DerivedExprSpec, DerivedKind, DoseEvent,
        ErrorModel, ErrorSpec, GradientMethod, IndivParamPartials, ModelParameters, OmegaMatrix,
        PkModel, PkParams, Population, ScalingSpec, SigmaVector, Subject, PK_IDX_LAGTIME,
    };
    use nalgebra::DVector;
    use std::collections::HashMap;

    /// 1-cpt IV model with one BSV eta (`ETA_CL`) and one IOV kappa
    /// (`KAPPA_CL`). CL = 10 · exp(κ); the kappa is read from `eta[1]` with a
    /// `.get(1)` guard, so the *broken* (BSV-only) path would read κ=0 → CL=10
    /// for every observation, while the fix yields the per-occasion CL.
    fn minimal_iov_model(derived_exprs: Vec<DerivedExprSpec>) -> CompiledModel {
        CompiledModel {
            frem_config: None,
            residual_error_eta: None,
            analytical_init: Vec::new(),
            ruv_magnitude: None,
            name: "test_iov_kappa".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Additive,
            error_spec: ErrorSpec::Single(ErrorModel::Additive),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|_theta: &[f64], eta: &[f64], _cov: &HashMap<String, f64>| {
                let kappa = eta.get(1).copied().unwrap_or(0.0);
                let mut p = PkParams::default();
                p.values[0] = 10.0 * kappa.exp(); // CL slot
                p
            }),
            n_theta: 0,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: Vec::new(),
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into()],
            indiv_param_partials: IndivParamPartials::empty(),
            default_params: ModelParameters {
                theta: Vec::new(),
                theta_names: Vec::new(),
                theta_lower: Vec::new(),
                theta_upper: Vec::new(),
                theta_fixed: Vec::new(),
                omega: OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]),
                omega_fixed: vec![false],
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: Some(OmegaMatrix::from_diagonal(&[1.0], vec!["KAPPA_CL".into()])),
                kappa_fixed: vec![false],
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: Some(Box::new(|_t, _c| vec![])),
            pk_indices: vec![0],
            eta_map: Vec::new(),
            pk_idx_f64: Vec::new(),
            sel_flat: Vec::new(),
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs,
            output_columns: Vec::new(),
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        }
    }

    /// Subject with two occasions: obs 0,1 on occasion 1; obs 2,3 on occasion 2.
    fn two_occasion_subject() -> Subject {
        Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: Vec::new(),
            obs_times: vec![0.0, 1.0, 2.0, 3.0],
            obs_raw_times: vec![0.0, 1.0, 2.0, 3.0],
            observations: vec![1.0; 4],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn sr_iov(n_obs: usize) -> SubjectResult {
        SubjectResult {
            id: "S1".into(),
            eta: DVector::from_vec(vec![0.0]), // BSV η = 0 so CL is driven purely by κ
            ipred: vec![1.0; n_obs],
            pred: vec![1.0; n_obs],
            iwres: vec![0.0; n_obs],
            cwres: vec![0.0; n_obs],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![0; n_obs],
            n_obs,
            extra_columns: Vec::new(),
            per_obs_tad: Vec::new(),
            compartment_states: Vec::new(),
        }
    }

    /// Both the indiv-param map (CL, via `pk_param_fn`) and `ctx.eta` must carry
    /// the per-observation occasion kappa. With κ₁ = ln 2 (occasion 1) and
    /// κ₂ = ln 3 (occasion 2), CL = 10·exp(κ) = [20, 20, 30, 30] and the kappa
    /// exposed through `ctx.eta[1]` = [ln2, ln2, ln3, ln3]. The pre-fix code
    /// produced CL = 10 for every row (κ silently 0).
    #[test]
    fn derived_and_indiv_use_per_occasion_kappa() {
        let ln2 = 2.0_f64.ln();
        let ln3 = 3.0_f64.ln();

        let derived_exprs = vec![
            // CL_OUT exercises the pk_param_fn call that builds per_obs_indiv.
            DerivedExprSpec {
                name: "CL_OUT".into(),
                kind: DerivedKind::PerRow {
                    eval: Box::new(|ctx: &DerivedContext| {
                        ctx.indiv_params.get("CL").copied().unwrap_or(f64::NAN)
                    }),
                },
                uses_compartments: false,
            },
            // K_OUT exercises DerivedContext.eta threading (eta[1] = occasion κ).
            DerivedExprSpec {
                name: "K_OUT".into(),
                kind: DerivedKind::PerRow {
                    eval: Box::new(|ctx: &DerivedContext| ctx.eta.get(1).copied().unwrap_or(-1.0)),
                },
                uses_compartments: false,
            },
        ];

        let model = minimal_iov_model(derived_exprs);
        let population = Population {
            subjects: vec![two_occasion_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        // ebe_kappas[subject][occasion]; occasion order matches split_obs_by_occasion
        // (first-seen): occasion 1 → index 0 (κ=ln2), occasion 2 → index 1 (κ=ln3).
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![
            DVector::from_vec(vec![ln2]),
            DVector::from_vec(vec![ln3]),
        ]];
        let mut subjects_results = vec![sr_iov(4)];

        compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

        let cols = &subjects_results[0].extra_columns;
        let cl = &cols.iter().find(|(n, _)| n == "CL_OUT").unwrap().1;
        let kout = &cols.iter().find(|(n, _)| n == "K_OUT").unwrap().1;

        let expected_cl = [20.0, 20.0, 30.0, 30.0];
        let expected_k = [ln2, ln2, ln3, ln3];
        for (j, (&got, &exp)) in cl.iter().zip(expected_cl.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-9,
                "CL_OUT[{j}] (occasion {}): got {got}, expected {exp}",
                if j < 2 { 1 } else { 2 }
            );
        }
        for (j, (&got, &exp)) in kout.iter().zip(expected_k.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "ctx.eta[1] at obs {j}: got {got}, expected occasion κ {exp}"
            );
        }
    }

    /// Defensive path: when a subject's kappa vector is missing (e.g. fewer
    /// occasion entries than occasions seen), the kappa slots fall back to 0
    /// rather than panicking — CL collapses to the κ=0 value (10).
    #[test]
    fn missing_kappa_falls_back_to_zero() {
        let derived_exprs = vec![DerivedExprSpec {
            name: "CL_OUT".into(),
            kind: DerivedKind::PerRow {
                eval: Box::new(|ctx: &DerivedContext| {
                    ctx.indiv_params.get("CL").copied().unwrap_or(f64::NAN)
                }),
            },
            uses_compartments: false,
        }];
        let model = minimal_iov_model(derived_exprs);
        let population = Population {
            subjects: vec![two_occasion_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        // Empty kappas for the subject → every occasion lookup misses → κ=0.
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]];
        let mut subjects_results = vec![sr_iov(4)];

        compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

        let cl = &subjects_results[0].extra_columns[0].1;
        for (j, &v) in cl.iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-9,
                "CL_OUT[{j}] with missing kappa should fall back to κ=0 → 10, got {v}"
            );
        }
    }

    /// Caller-level regression for the per-dose-occasion absorption lag
    /// (follow-up to #238). `compute_extra_output_columns` must build TAD using
    /// each *dose's* occasion lag, not the observation's. The model puts IOV on
    /// the lag (`lag = 1.0 + κ`); a subject is dosed BID across two occasions:
    ///   morning dose @0  (occasion 1, κ=0.0 → lag 1.0)
    ///   evening dose @12 (occasion 2, κ=0.5 → lag 1.5)
    /// with observations @2 (occ 1) and @13 (occ 2). At obs @13 the evening dose
    /// arrives at 13.5 — not yet absorbed — so TAD counts from the morning dose's
    /// arrival at 1.0 → 12.0. Applying the obs-occasion lag (1.5) to every dose,
    /// as before this follow-up, would give 11.5.
    #[test]
    fn tad_uses_per_dose_occasion_lag() {
        let mut model = minimal_iov_model(vec![]);
        // Declare an absorption lag (ALAG → PK_IDX_LAGTIME) so `model.has_lagtime()`
        // holds; compute_extra_output_columns only builds per-dose lags for models
        // that declare a lag.
        model.indiv_param_names = vec!["CL".into(), "ALAG".into()];
        model.pk_indices = vec![0, PK_IDX_LAGTIME];
        // Drive the absorption lag (slot PK_IDX_LAGTIME) from the occasion kappa.
        model.pk_param_fn = Box::new(|_theta: &[f64], eta: &[f64], _cov: &HashMap<String, f64>| {
            let kappa = eta.get(1).copied().unwrap_or(0.0);
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = 1.0 + kappa;
            p
        });

        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![2.0, 13.0],
            obs_raw_times: vec![2.0, 13.0],
            observations: vec![1.0, 1.0],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0],
            occasions: vec![1, 2],
            dose_occasions: vec![1, 2],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        // ebe_kappas in split_obs_by_occasion order: occ 1 → idx 0 (κ=0.0),
        // occ 2 → idx 1 (κ=0.5).
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.5]),
        ]];
        let mut subjects_results = vec![sr_iov(2)];

        compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

        let tad = &subjects_results[0].per_obs_tad;
        assert!(
            (tad[0] - 1.0).abs() < 1e-9,
            "obs@2 TAD: morning dose arrives @1.0 → 1.0, got {}",
            tad[0]
        );
        assert!(
            (tad[1] - 12.0).abs() < 1e-9,
            "obs@13 TAD must be 12.0 (evening dose uses its own occ-2 lag 1.5 → arrives \
             @13.5, excluded; counts from morning @1.0). The pre-follow-up obs-occasion \
             lag would give 11.5. Got {}",
            tad[1]
        );
    }

    /// Regression for the per-compartment TAD lag (issue #369). On an ODE model
    /// declaring `ALAGn`, the TAD column must anchor each dose on *its own*
    /// compartment's lag — resolved through `dose_attr_map`, the same single
    /// source of truth the prediction paths use — not the bare `PK_IDX_LAGTIME`
    /// slot, which a model declaring only `ALAG2` leaves at 0. A dose into
    /// compartment 2 at t=0 with `ALAG2 = 2` effectively arrives at t=2, so an
    /// observation at t=3 has TAD = 1.0. The pre-fix code read the bare lag (0)
    /// and reported TAD = 3.0.
    #[test]
    fn tad_uses_per_compartment_alag_on_ode_models() {
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG2(2.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
        let model = crate::parser::model_parser::parse_full_model(src)
            .expect("parse ok")
            .model;
        assert!(model.has_lagtime(), "ALAG2 must enable has_lagtime()");

        // Dose into compartment 2 (central) at t=0; observe cmt 2 at t=3.
        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)],
            obs_times: vec![3.0],
            obs_raw_times: vec![3.0],
            observations: vec![1.0],
            obs_cmts: vec![2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let theta = model.default_params.theta.clone();
        let kappas: Vec<Vec<DVector<f64>>> = Vec::new(); // no IOV
        let mut subjects_results = vec![sr_iov(1)];

        compute_extra_output_columns(&model, &population, &theta, &kappas, &mut subjects_results);

        let tad = &subjects_results[0].per_obs_tad;
        // 3 − (dose@0 + ALAG2=2) = 1.0; the pre-fix bare-lag path gives 3.0.
        assert!((tad[0] - 1.0).abs() < 1e-9, "tad: {}", tad[0]);
    }

    /// Regression for issue #369: a negative compartment-indexed `ALAGn` must
    /// raise `W_NEGATIVE_LAGTIME`. The bare-slot check alone never sees the
    /// `ALAG2` spare slot, so a bad per-route lag would otherwise slip through.
    #[test]
    fn negative_alag_emits_negative_lagtime_warning() {
        let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAG2(-1.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  ALAG2 = TVLAG2

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -CL/V * depot
  d/dt(central) =  CL/V * depot - CL/V * central

[error_model]
  DV ~ proportional(PROP)
"#;
        let model = crate::parser::model_parser::parse_full_model(src)
            .expect("parse ok")
            .model;

        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0)],
            obs_times: vec![3.0],
            obs_raw_times: vec![3.0],
            observations: vec![1.0],
            obs_cmts: vec![2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };

        let diags = check_model_data_warnings(&model, &population, &model.default_params);
        let neg = diags
            .iter()
            .find(|d| d.code == "W_NEGATIVE_LAGTIME")
            .expect("a negative ALAG2 must raise W_NEGATIVE_LAGTIME");
        assert!(
            neg.message.contains("ALAG2") && neg.message.contains("compartment-2"),
            "warning must name the offending compartment-indexed lag, got: {}",
            neg.message
        );
    }

    /// The per-compartment negative-lag scan is ODE-only. An analytical model can
    /// still report `has_lagtime()` (lag bound via `pk_indices`), so
    /// `check_model_data_warnings` must take the `ode_spec == None` path: the bare
    /// negative lag still warns, but no compartment-indexed `ALAGn` is emitted.
    #[test]
    fn negative_lag_scan_skips_analytical_models() {
        let mut model = minimal_iov_model(vec![]);
        // Analytical (ode_spec stays None); has_lagtime() via pk_indices carrying
        // PK_IDX_LAGTIME; bare lag evaluates negative.
        model.indiv_param_names = vec!["CL".into(), "LAGTIME".into()];
        model.pk_indices = vec![0, PK_IDX_LAGTIME];
        model.pk_param_fn = Box::new(|_t: &[f64], _e: &[f64], _c: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = -1.0;
            p
        });
        assert!(model.has_lagtime() && model.ode_spec.is_none());

        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: vec![1.0],
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };

        let diags = check_model_data_warnings(&model, &population, &model.default_params);
        assert!(
            diags.iter().any(|d| d.code == "W_NEGATIVE_LAGTIME"),
            "bare negative lag must still warn on an analytical model"
        );
        assert!(
            !diags.iter().any(|d| d.message.contains("ALAG")),
            "no compartment-indexed ALAGn warning for an analytical (non-ODE) model"
        );
    }

    /// An SS=1 dose combined with an [initial_conditions] baseline double-counts
    /// the starting amount (the SS closed form already pre-equilibrates an
    /// infinite dose history). `check_model_data_warnings` must surface it
    /// (W_STEADY_STATE_INIT) rather than silently superpose the two (#521 review).
    #[test]
    fn ss_dose_with_analytical_init_warns() {
        let src = r#"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = crate::parser::model_parser::parse_full_model(src)
            .expect("parse ok")
            .model;
        assert_eq!(model.analytical_init.len(), 1);

        // Subject with a steady-state dose (ss=true, ii=24).
        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
            obs_times: vec![1.0, 6.0],
            obs_raw_times: vec![1.0, 6.0],
            observations: vec![1.0, 1.0],
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };

        let diags = check_model_data_warnings(&model, &population, &model.default_params);
        assert!(
            diags.iter().any(|d| d.code == "W_STEADY_STATE_INIT"),
            "SS dose + [initial_conditions] must raise W_STEADY_STATE_INIT; got: {:?}",
            diags.iter().map(|d| &d.code).collect::<Vec<_>>()
        );
    }

    /// No SS dose → no W_STEADY_STATE_INIT, even with an init baseline present.
    #[test]
    fn non_ss_dose_with_analytical_init_does_not_warn() {
        let src = r#"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[initial_conditions]
  init(central) = 30 * V

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let model = crate::parser::model_parser::parse_full_model(src)
            .expect("parse ok")
            .model;
        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            obs_raw_times: vec![1.0],
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let diags = check_model_data_warnings(&model, &population, &model.default_params);
        assert!(
            !diags.iter().any(|d| d.code == "W_STEADY_STATE_INIT"),
            "no SS dose → no W_STEADY_STATE_INIT"
        );
    }

    /// Caller-level regression: the per-dose lag uses each dose's *covariate*
    /// snapshot (`dose_cov`), not the observation's — so a lag depending on a
    /// time-varying covariate is shifted by conditions at dosing, matching
    /// `predict_iov`. Lag = `LAGCOV`: the dose sees `LAGCOV=1` (lag 1.0), the
    /// observation sees `LAGCOV=5` (lag 5.0). With a dose @0 and obs @3, the dose
    /// covariate gives arrival @1.0 → TAD 2.0; the obs covariate would push arrival
    /// to @5.0 (after the obs) → TAD NaN. Asserting TAD = 2.0 proves `dose_cov` is
    /// used. (No IOV here — this isolates the covariate basis, not kappa.)
    #[test]
    fn tad_lag_uses_dose_covariate_not_obs() {
        let mut model = minimal_iov_model(vec![]);
        model.indiv_param_names = vec!["CL".into(), "ALAG".into()];
        model.pk_indices = vec![0, PK_IDX_LAGTIME];
        // Lag = LAGCOV covariate (time-varying); independent of eta/kappa.
        model.pk_param_fn = Box::new(|_theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            p.values[PK_IDX_LAGTIME] = cov.get("LAGCOV").copied().unwrap_or(0.0);
            p
        });

        let cov_dose = HashMap::from([("LAGCOV".to_string(), 1.0)]);
        let cov_obs = HashMap::from([("LAGCOV".to_string(), 5.0)]);
        let subject = Subject {
            fremtype: Vec::new(),
            id: "S1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![3.0],
            obs_raw_times: vec![3.0],
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            dose_covariates: vec![cov_dose],
            obs_covariates: vec![cov_obs],
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0],
            occasions: vec![1],
            dose_occasions: vec![1],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![subject],
            covariate_names: vec!["LAGCOV".into()],
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![DVector::from_vec(vec![0.0])]];
        let mut subjects_results = vec![sr_iov(1)];

        compute_extra_output_columns(&model, &population, &[], &kappas, &mut subjects_results);

        let tad = &subjects_results[0].per_obs_tad;
        assert!(
            (tad[0] - 2.0).abs() < 1e-9,
            "TAD must use the dose covariate (LAGCOV=1 → lag 1.0 → arrival @1.0 → TAD 2.0); \
             the obs covariate (LAGCOV=5) would push arrival to @5.0 (after the obs) → NaN. \
             Got {}",
            tad[0]
        );
    }
}

// ── Tests: adaptive (state-reactive) dosing — simulate_adaptive (#391 S1.4b) ──
#[cfg(test)]
mod adaptive_sim_tests {
    use super::*;
    use crate::parser::model_parser::parse_model_string;
    use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
    use std::collections::HashMap;

    // 1-cpt IV ODE, state = central amount, readout y = central (amount units).
    // CL=5, V=50 → k = 0.1/h. ETA_CL is declared (the parser requires an omega)
    // but **not referenced** by CL, so predictions are η-invariant and the whole
    // simulation is effectively deterministic — the bit-exact oracle below relies
    // on this (a drawn-but-unused η leaves the trajectory at the η=0 value).
    const ODE_NO_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

    // Same structural model, with between-subject variability on CL.
    const ODE_IIV: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
"#;

    // Analytical (non-ODE) twin — used to assert simulate_adaptive rejects it.
    const ANALYTICAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#;

    fn subj(id: &str, obs_times: Vec<f64>, doses: Vec<DoseEvent>) -> Subject {
        let n = obs_times.len();
        let n_dose = doses.len();
        Subject {
            id: id.to_string(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n],
            obs_cmts: vec![1; n],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: vec![1u32; n],
            dose_occasions: vec![1u32; n_dose],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn population(subjects: Vec<Subject>) -> Population {
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// A fixed 100 mg bolus into cmt 1 at every decision — the degenerate
    /// (state-independent) controller.
    fn fixed_bolus() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
        |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
    }

    #[test]
    fn degenerate_oracle_matches_static_predict_bit_for_bit() {
        // A controller that doses at every decision must reproduce the static
        // engine on the same realized schedule. Last obs (54) is the global max
        // and a dose lands at every decision, so the segment structures align and
        // agreement is exact (the verifier, on by default, also passes).
        let model = parse_model_string(ODE_NO_IIV).expect("parse no-IIV ODE model");
        let decisions = vec![0.0, 24.0, 48.0];
        let obs = vec![6.0, 30.0, 54.0];

        let pop = population(vec![subj("1", obs.clone(), vec![])]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: decisions.clone(),
            ..Default::default()
        };
        let res = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
            .expect("adaptive sim runs");

        // Static reference: same doses pre-scheduled, run through predict() (η=0,
        // IPRED) — exactly what the realized ledger encodes.
        let static_doses: Vec<DoseEvent> = decisions
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let static_pop = population(vec![subj("1", obs.clone(), static_doses)]);
        let preds = predict(&model, &static_pop, &model.default_params);

        assert_eq!(res.trajectories.len(), obs.len());
        assert_eq!(res.ledger.len(), 3, "a dose at every decision");
        for (traj, pred) in res.trajectories.iter().zip(preds.iter()) {
            assert!(
                (traj.ipred - pred.pred).abs() <= 1e-9 + 1e-9 * pred.pred.abs(),
                "adaptive IPRED {} != static predict {} at t={}",
                traj.ipred,
                pred.pred,
                traj.time
            );
        }
    }

    #[test]
    fn reactive_holds_run_and_pass_the_default_verifier() {
        // Genuinely reactive: dose 100 only when the monitored central amount is
        // below 50. k = 0.1/h, so after the t=0 dose the amount is 100·e^{-0.2} ≈
        // 81.9 at t=2 and ≈67.0 at t=4 — both > 50 → hold. Exactly one realized
        // dose, and the t=2 / t=4 holds make the driver break where the static
        // replay of the ledger (dose @0 only) does not — so this run passes
        // through the verifier's *tolerance* branch, not the bit-exact path.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![1.0, 3.0, 5.0], vec![])]);
        let monitors = vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: vec![0.0, 2.0, 4.0],
            monitors,
            ..Default::default() // verify = true
        };
        let make = || {
            |ctx: &ControllerCtx| {
                if ctx.signal("A").expect("monitor A declared") < 50.0 {
                    vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
                } else {
                    vec![DoseAction::Hold]
                }
            }
        };
        let res = simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts)
            .expect("verifier passes on the reactive run");
        // One decision per schedule point is logged, including the two holds.
        assert_eq!(res.decisions.len(), 3);
        // Only the t=0 decision (amount 0 < 50) dosed.
        assert_eq!(res.ledger.len(), 1);
        assert_eq!(res.ledger[0].time, 0.0);
    }

    #[test]
    fn fresh_controller_per_subject_prevents_state_leak() {
        // The "no happy paths" guarantee of the factory signature: a *stateful*
        // controller that doses only on its first call must dose every subject —
        // proving each gets a fresh controller. A single shared FnMut would dose
        // only the first subject (its counter would already be spent), so this
        // asserts the leak is structurally impossible.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![
            subj("A", vec![1.0], vec![]),
            subj("B", vec![1.0], vec![]),
            subj("C", vec![1.0], vec![]),
        ]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            decision_times: vec![0.0],
            ..Default::default()
        };
        let make = || {
            let mut calls = 0usize;
            move |_ctx: &ControllerCtx| {
                let actions = if calls == 0 {
                    vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
                } else {
                    vec![DoseAction::Hold]
                };
                calls += 1;
                actions
            }
        };
        let res =
            simulate_adaptive(&model, &pop, &model.default_params, 1, make, &opts).expect("runs");
        assert_eq!(
            res.ledger.len(),
            3,
            "every subject's first decision should dose (fresh controller each)"
        );
        let mut ids: Vec<&str> = res.ledger.iter().map(|e| e.subject.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, ["A", "B", "C"]);
    }

    #[test]
    fn replicate_tags_are_stamped_on_every_row() {
        // The single-subject driver emits draw/sim = 0; the orchestrator must
        // stamp the real replicate index onto trajectory, ledger, and decision
        // rows so the artifacts join.
        let model = parse_model_string(ODE_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![6.0, 30.0], vec![])]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(42),
            decision_times: vec![0.0, 24.0],
            ..Default::default()
        };
        let res = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
            .expect("runs");

        let sims: std::collections::BTreeSet<usize> = res.ledger.iter().map(|e| e.sim).collect();
        assert_eq!(sims, [1, 2].into_iter().collect());
        assert!(res.ledger.iter().all(|e| e.draw == 1));
        assert!(res
            .trajectories
            .iter()
            .all(|t| t.draw == 1 && (t.sim == 1 || t.sim == 2)));
        assert!(res
            .decisions
            .iter()
            .all(|d| d.draw == 1 && (d.sim == 1 || d.sim == 2)));
    }

    #[test]
    fn same_seed_is_deterministic() {
        let model = parse_model_string(ODE_IIV).expect("parse");
        let pop = population(vec![
            subj("1", vec![6.0, 30.0], vec![]),
            subj("2", vec![6.0, 30.0], vec![]),
        ]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(7),
            decision_times: vec![0.0, 24.0],
            ..Default::default()
        };
        let a = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
            .expect("run a");
        let b = simulate_adaptive(&model, &pop, &model.default_params, 2, fixed_bolus, &opts)
            .expect("run b");

        let ip = |r: &AdaptiveSimulationResult| {
            r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>()
        };
        assert_eq!(
            ip(&a),
            ip(&b),
            "IPRED trajectories must be seed-reproducible"
        );
        assert_eq!(a.ledger, b.ledger);
        assert_eq!(a.decisions, b.decisions);
    }

    #[test]
    fn rejects_analytical_model() {
        let model = parse_model_string(ANALYTICAL).expect("parse analytical");
        let pop = population(vec![subj("1", vec![6.0], vec![])]);
        let opts = AdaptiveSimulateOptions {
            decision_times: vec![0.0],
            ..Default::default()
        };
        let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
            .expect_err("analytical model must be rejected");
        assert!(err.contains("ODE"), "got: {err}");
    }

    #[test]
    fn rejects_empty_decision_schedule() {
        // The default `decision_times` is empty; a caller who forgets to set it
        // would otherwise get a silent dose-free run that passes the verifier
        // trivially. That must be a typed error, not a happy-path no-op.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![6.0], vec![])]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(1),
            ..Default::default() // decision_times left empty
        };
        let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
            .expect_err("an empty decision schedule must be rejected");
        assert!(err.contains("decision schedule"), "got: {err}");
    }

    #[test]
    fn rejects_subject_with_prescheduled_doses() {
        // The regimen is controller-driven; a subject that already carries doses
        // is a typed error (forwarded from the driver), never a silent merge.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let dosed = subj(
            "1",
            vec![6.0],
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        );
        let pop = population(vec![dosed]);
        let opts = AdaptiveSimulateOptions {
            decision_times: vec![0.0],
            ..Default::default()
        };
        let err = simulate_adaptive(&model, &pop, &model.default_params, 1, fixed_bolus, &opts)
            .expect_err("pre-dosed subject must be rejected");
        assert!(err.contains("dose-free"), "got: {err}");
    }

    #[test]
    fn dv_monitor_without_error_model_is_rejected() {
        // Edge (a): a DV monitor on a model with no residual error (here sigma is
        // stripped) is a typed error, never a fabricated σ.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let mut params = model.default_params.clone();
        params.sigma.values = vec![]; // no [error_model] coverage for the monitor
        let pop = population(vec![subj("1", vec![6.0], vec![])]);
        let opts = AdaptiveSimulateOptions {
            decision_times: vec![0.0],
            monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
            ..Default::default()
        };
        let err = simulate_adaptive(&model, &pop, &params, 1, fixed_bolus, &opts)
            .expect_err("DV monitor with no error model must be rejected");
        assert!(err.contains("error_model"), "got: {err}");
    }

    #[test]
    fn verify_can_be_disabled() {
        // The verifier is the default-on safety net; it can be turned off for a
        // large run once the controller is trusted. Disabling it still produces
        // the same trajectories/ledger (verification observes, it does not alter).
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![6.0, 30.0], vec![])]);
        let mk = |verify: bool| AdaptiveSimulateOptions {
            seed: Some(3),
            decision_times: vec![0.0, 24.0],
            verify,
            ..Default::default()
        };
        let checked = simulate_adaptive(
            &model,
            &pop,
            &model.default_params,
            1,
            fixed_bolus,
            &mk(true),
        )
        .expect("verified run");
        let unchecked = simulate_adaptive(
            &model,
            &pop,
            &model.default_params,
            1,
            fixed_bolus,
            &mk(false),
        )
        .expect("unverified run");
        assert_eq!(checked.ledger, unchecked.ledger);
        let ip = |r: &AdaptiveSimulationResult| {
            r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>()
        };
        assert_eq!(ip(&checked), ip(&unchecked));
    }

    // ----- S1.5: DV-mode (assay-noised) monitors --------------------------

    /// Factory (mirrors `fixed_bolus`): titrate on the *measured* (DV) central
    /// amount "A" — dose 100 mg when it falls below 50, else hold.
    fn dv_threshold() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
        |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        }
    }

    #[test]
    fn dv_monitor_run_passes_default_verifier() {
        // End-to-end: a controller titrating on the DV signal, verify = true. Covers
        // the assay-noise wiring (resid_var closure + per-subject base seed) and the
        // driver's DV branch, and confirms the frozen-replay verifier stays green on
        // a DV run (it replays realized doses, not decisions).
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![2.0, 6.0, 10.0], vec![])]);
        let opts = AdaptiveSimulateOptions {
            seed: Some(7),
            decision_times: vec![0.0, 4.0, 8.0],
            monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
            ..Default::default()
        };
        let res = simulate_adaptive(&model, &pop, &model.default_params, 1, dv_threshold, &opts)
            .expect("DV run passes the verifier");
        assert_eq!(res.decisions.len(), 3);
        // The DV value the controller saw is recorded, tagged as the Dv mode.
        assert!(res
            .decisions
            .iter()
            .all(|d| d.observed_signals[0].mode == ObserveMode::Dv));
    }

    #[test]
    fn dv_monitor_collapses_to_ipred_as_sigma_to_zero() {
        // sigma -> 0: titrating on the DV reproduces titrating on the latent IPRED,
        // realized-ledger for realized-ledger.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let pop = population(vec![subj("1", vec![2.0, 6.0, 10.0], vec![])]);
        let mut params = model.default_params.clone();
        params.sigma.values = vec![1e-12];
        let decision_times = vec![0.0, 4.0, 8.0];

        let dv_opts = AdaptiveSimulateOptions {
            seed: Some(7),
            decision_times: decision_times.clone(),
            monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
            ..Default::default()
        };
        let ipred_opts = AdaptiveSimulateOptions {
            seed: Some(7),
            decision_times,
            monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Ipred)],
            ..Default::default()
        };
        let dv = simulate_adaptive(&model, &pop, &params, 1, dv_threshold, &dv_opts).expect("dv");
        let ip =
            simulate_adaptive(&model, &pop, &params, 1, dv_threshold, &ipred_opts).expect("ipred");
        // Compare the *dosing decisions* (time/amt/cmt/rate), not the recorded
        // observed signals: the residual variance floors at MIN_VARIANCE, so the DV
        // readout never collapses to IPRED bit-for-bit — but the decisions it drives
        // do. (The exact bit-for-bit collapse is pinned at the driver level with a
        // literal zero-variance resolver.)
        let schedule = |r: &AdaptiveSimulationResult| {
            r.ledger
                .iter()
                .map(|e| (e.time, e.amt, e.cmt, e.rate))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            schedule(&dv),
            schedule(&ip),
            "DV with sigma→0 must make the same dosing decisions as IPRED"
        );
    }

    #[test]
    fn adding_dv_monitor_does_not_perturb_eta_trajectory() {
        // Isolation: the controller-assay substream is disjoint from the η stream,
        // so enabling a DV monitor leaves the η draws (hence the IPRED trajectory)
        // bit-identical. Uses ODE_IIV (η on CL) and a signal-independent controller,
        // so any trajectory difference could only come from a shifted η draw.
        let model = parse_model_string(ODE_IIV).expect("parse");
        let pop = population(vec![
            subj("1", vec![6.0, 30.0], vec![]),
            subj("2", vec![6.0, 30.0], vec![]),
        ]);
        let opts = |monitors: Vec<MonitorSpec>| AdaptiveSimulateOptions {
            seed: Some(3),
            decision_times: vec![0.0, 24.0],
            monitors,
            ..Default::default()
        };
        let no_mon = simulate_adaptive(
            &model,
            &pop,
            &model.default_params,
            3,
            fixed_bolus,
            &opts(vec![]),
        )
        .expect("no monitor");
        let with_dv = simulate_adaptive(
            &model,
            &pop,
            &model.default_params,
            3,
            fixed_bolus,
            &opts(vec![MonitorSpec::new("A", 1, ObserveMode::Dv)]),
        )
        .expect("dv monitor");
        let ip = |r: &AdaptiveSimulationResult| {
            r.trajectories.iter().map(|t| t.ipred).collect::<Vec<_>>()
        };
        assert_eq!(
            ip(&no_mon),
            ip(&with_dv),
            "a DV monitor must not shift the η draws / IPRED trajectory"
        );
    }

    #[test]
    fn dv_assay_draws_are_permutation_invariant() {
        // The controller-assay substream is keyed by subject id, so a subject's DV
        // draws — and thus its decisions — do not depend on iteration order. A large
        // residual CV makes the noise flip decisions (so a position-keyed stream
        // *would* change the ledger), and ODE_NO_IIV keeps η out of the picture.
        let model = parse_model_string(ODE_NO_IIV).expect("parse");
        let mut params = model.default_params.clone();
        params.sigma.values = vec![0.3]; // 30% proportional CV: noise is consequential
        let opts = AdaptiveSimulateOptions {
            seed: Some(11),
            decision_times: vec![0.0, 4.0, 8.0],
            monitors: vec![MonitorSpec::new("A", 1, ObserveMode::Dv)],
            ..Default::default()
        };
        let a = subj("A", vec![2.0, 6.0, 10.0], vec![]);
        let b = subj("B", vec![2.0, 6.0, 10.0], vec![]);
        let pop_ab = population(vec![a.clone(), b.clone()]);
        let pop_ba = population(vec![b, a]);
        let r_ab = simulate_adaptive(&model, &pop_ab, &params, 1, dv_threshold, &opts).expect("ab");
        let r_ba = simulate_adaptive(&model, &pop_ba, &params, 1, dv_threshold, &opts).expect("ba");

        let ledger_for = |r: &AdaptiveSimulationResult, id: &str| {
            r.ledger
                .iter()
                .filter(|e| e.subject == id)
                .map(|e| (e.time, e.amt))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ledger_for(&r_ab, "A"),
            ledger_for(&r_ba, "A"),
            "A's DV-driven doses must be order-independent"
        );
        assert_eq!(
            ledger_for(&r_ab, "B"),
            ledger_for(&r_ba, "B"),
            "B's DV-driven doses must be order-independent"
        );

        // Non-vacuity: the assay noise must actually be id-keyed (not a no-op), or
        // the invariance above could hold even under a position-keyed stream. At
        // t=4 the trough is ~67 (non-zero, so proportional noise is live), and A and
        // B observe *different* measured values — confirming the noise is consequential.
        let dv_at = |r: &AdaptiveSimulationResult, id: &str, didx: usize| {
            r.decisions
                .iter()
                .find(|d| d.subject == id && d.decision_idx == didx)
                .map(|d| d.observed_signals[0].value)
                .expect("decision logged")
        };
        assert_ne!(
            dv_at(&r_ab, "A", 1),
            dv_at(&r_ab, "B", 1),
            "id-keyed assay noise must differ between subjects (test would be vacuous otherwise)"
        );
    }
}
