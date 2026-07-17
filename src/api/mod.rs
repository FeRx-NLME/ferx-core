use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::{
    chol_lt_idx, lower_tri_iter, omega_packed_len, theta_packs_log,
};
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_filtered_tte, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    read_nonmem_csv_with_covariates_tte, SelectionFilter, ERR_COV_MISSING_COLUMNS,
    ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations,
    compute_r_matrix_with_correlations_scaled, iwres_autocorrelation,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// ── validation subsystem (extracted verbatim; see src/api/validation.rs) ──
mod validation;
pub(crate) use validation::{
    apply_iov_occasion_rule, assert_absorption_closed_form_support,
    assert_absorption_dosing_supported, assert_absorption_flip_flop_no_twin,
    assert_analytic_readout_support, assert_modeled_doses_supported,
    check_absorption_closed_form_support, check_absorption_dosing,
    check_absorption_flip_flop_no_twin, check_analytic_readout_support, check_covariates,
    check_modeled_dose_rates,
};
#[cfg(feature = "survival")]
pub(crate) use validation::{
    assert_survival_tv_covariates, check_rtte_records, check_survival_tv_covariates,
};
pub use validation::{
    check_experimental_features, check_model_data, check_model_data_rule,
    check_model_data_warnings, check_model_options, validate_model_file, validate_output_columns,
};

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

/// Worker-stack size for ferx-managed Rayon pools.
///
/// Wide ODE+IOV analytic sensitivities instantiate `Dual2<M>` with `M` close to 100.
/// Each dual carries an `M x M` Hessian, and the ODE/event-walk frames hold several
/// of them at once. Rayon/default pthread stacks can be as small as 2 MiB on macOS,
/// which overflows before Rust can unwind. Use a larger fit-scoped stack for Rayon
/// workers that may evaluate these gradients.
pub(crate) const FIT_RAYON_STACK_SIZE: usize = 32 * 1024 * 1024;

pub(crate) fn fit_thread_pool_builder() -> rayon::ThreadPoolBuilder {
    rayon::ThreadPoolBuilder::new().stack_size(FIT_RAYON_STACK_SIZE)
}

/// Ceiling applied to the unpinned default thread count (#707).
const DEFAULT_THREADS_CAP: usize = 8;

/// Core cap arithmetic for the unpinned default thread count, split out from
/// [`default_thread_count`] so it is testable without depending on the host's actual
/// core count: leave one core free for the OS/other work, and don't scale past
/// [`DEFAULT_THREADS_CAP`] even on much larger machines, since most fits see no benefit
/// from spreading across every core and (notably on Apple Silicon) not all cores are equal.
fn cap_default_threads(available: usize) -> usize {
    available.saturating_sub(1).clamp(1, DEFAULT_THREADS_CAP)
}

/// Default worker-thread count used when nothing pins an explicit count (`threads` unset
/// or `auto`/`0`, and no explicit `--threads`/[`configure_global_thread_pool`] call). See
/// [`cap_default_threads`] for the cap logic (#707).
pub(crate) fn default_thread_count() -> usize {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cap_default_threads(available)
}

/// Set when a caller has explicitly sized the process-wide Rayon pool (currently: the CLI's
/// `--threads N` via [`configure_global_thread_pool`]), so [`default_fit_pool`] knows to
/// honor that explicit choice rather than applying the [`default_thread_count`] cap (#707).
static GLOBAL_THREADS_EXPLICIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Explicitly size the process-wide Rayon pool and mark it as user-chosen. Intended for a
/// CLI binary sizing its one process-wide pool from `--threads N` before the first fit;
/// library callers that want a pinned thread count for a single fit should use
/// `FitOptions::threads` instead, which scopes a fit-local pool via [`build_fit_pool`].
///
/// `n_threads` must be positive — a caller wanting the engine's own default should simply
/// not call this at all, rather than pass `0` (which Rayon would otherwise silently treat
/// as "pick automatically", masking the caller's intent). The explicit-override flag is
/// only set once `build_global` actually succeeds, so a failed call (e.g. the global pool
/// was already initialized elsewhere) leaves [`default_fit_pool`] applying the #707 cap
/// rather than incorrectly deferring to whatever the ambient pool happens to be.
pub fn configure_global_thread_pool(n_threads: usize) -> Result<(), String> {
    if n_threads == 0 {
        return Err("thread count must be positive".to_string());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .map_err(|e| format!("failed to configure thread pool with {n_threads} threads: {e}"))?;
    GLOBAL_THREADS_EXPLICIT.store(true, std::sync::atomic::Ordering::Release);
    Ok(())
}

/// Build a fit-scoped Rayon pool with the ferx worker stack and an explicit thread
/// count. Used only when the caller pins `options.threads` to a positive value; the
/// common (unpinned) path reuses the shared [`default_fit_pool`] instead.
pub(crate) fn build_fit_pool(n_threads: usize) -> Result<rayon::ThreadPool, String> {
    fit_thread_pool_builder()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("failed to build rayon pool with {n_threads} threads: {e}"))
}

/// The process-wide fit pool, built once with the ferx worker stack (32 MiB) so wide
/// ODE+IOV analytic gradients do not overflow the platform-default worker stack. Shared
/// across every default-threads `fit()` call so batch / concurrent callers do not each
/// spawn-and-tear-down a fresh `N × 32 MiB` pool (which oversubscribes CPUs and can
/// exhaust address space).
///
/// Sized by [`default_thread_count`] (available cores minus one, capped at 8 — #707): most
/// fits gain little from spreading across every core, and not all cores are equal on
/// asymmetric platforms (e.g. Apple Silicon E-cores). A caller that explicitly sized the
/// global pool via [`configure_global_thread_pool`] (the CLI's `--threads N`) is honored
/// instead — that call marks [`GLOBAL_THREADS_EXPLICIT`] before this pool is built.
///
/// Returns `None` only if the one-time build fails (e.g. resource limits); callers then
/// run on the ambient pool rather than aborting the fit.
pub(crate) fn default_fit_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n_threads = if GLOBAL_THREADS_EXPLICIT.load(std::sync::atomic::Ordering::Acquire) {
            rayon::current_num_threads()
        } else {
            default_thread_count()
        };
        fit_thread_pool_builder()
            .num_threads(n_threads)
            .build()
            .ok()
    })
    .as_ref()
}

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
        let resolved = crate::dosing::resolve_subject_doses(
            subject,
            model.active_dose_attr_map(),
            &pk_params.values,
        );
        pk::compute_predictions(model.pk_model, &resolved, pk_params)
    };
    // Analytic Form C readout (#650): replaces the built-in concentration. No-op
    // for ODE models (handled inside `compute_predictions_ode`) and when unset.
    pk::apply_analytic_readout(model, subject, theta, eta, &mut preds);
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

/// True if two paths point at the same file. The model's `[data] path` is
/// dir-joined to the model file's directory by `parse_full_model_file`, while
/// an externally supplied path (CLI `--data`, R) is passed through raw — so
/// the same file can differ textually (`./warfarin.csv` vs `warfarin.csv`).
/// Falls back to plain string equality when either path doesn't resolve on
/// disk (e.g. fixture names in unit tests).
fn paths_equivalent(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (Path::new(a).canonicalize(), Path::new(b).canonicalize()) {
        (Ok(pa), Ok(pb)) => pa == pb,
        _ => false,
    }
}

/// Resolves the dataset path a fit should use, applying override-with-warning
/// semantics between a model file's optional `[data] path = ...` (#690) and an
/// externally supplied path (CLI `--data`, R).
///
/// - Neither given → `Err` (nothing to fit against).
/// - Only one given → that one, no warning.
/// - Both given and equal (see [`paths_equivalent`]) → the shared path, no
///   warning.
/// - Both given and different → `external_path` wins; a warning is returned
///   (not printed — see "Warning and Error Conventions" in CLAUDE.md) for the
///   caller to attach to `FitResult.warnings` or a `ferx check` diagnostic.
pub fn resolve_data_path(
    model_data_path: Option<&str>,
    external_data_path: Option<&str>,
) -> Result<(String, Option<String>), String> {
    match (external_data_path, model_data_path) {
        (Some(ext), Some(model_p)) if !paths_equivalent(ext, model_p) => Ok((
            ext.to_string(),
            Some(format!(
                "dataset path overridden: using `{ext}` instead of the model's \
                 `[data] path = {model_p}`"
            )),
        )),
        (Some(ext), _) => Ok((ext.to_string(), None)),
        (None, Some(model_p)) => Ok((model_p.to_string(), None)),
        (None, None) => Err(
            "no dataset specified — pass a data path, or add a `[data]` block \
             (`path = ...`) to the model file"
                .to_string(),
        ),
    }
}

/// Run a model file with a NONMEM-format CSV dataset. `data_path` is `None` to
/// rely solely on the model's own `[data]` block (#690); when both are given,
/// `data_path` overrides the model's, with a warning recorded on the result.
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_with_data(
    model_path: &str,
    data_path: Option<&str>,
) -> Result<(FitResult, Population), String> {
    run_model_with_data_inits(model_path, data_path, None)
}

/// Like [`run_model_with_data`], but lets the caller (e.g. the CLI's
/// `--inits-from-nca` flag) override the model file's `inits_from_nca` fit
/// option. When `inits_override` is `None` the model-file value is used as-is;
/// when `Some(method)` it forces that NCA strategy regardless of the file.
pub fn run_model_with_data_inits(
    model_path: &str,
    data_path: Option<&str>,
    inits_override: Option<crate::suggest_start::NcaInit>,
) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    set_model_name(&mut parsed.model, model_path);
    if let Some(method) = inits_override {
        parsed.fit_options.inits_from_nca = Some(method);
    }

    eprintln!("Model: {}", parsed.model.name);

    let (data_path, data_path_warning) = resolve_data_path(parsed.data_path.as_deref(), data_path)?;
    let data_path = data_path.as_str();

    let iov_col = parsed.fit_options.iov_column.as_deref();
    let sel_filter = build_selection_filter(&parsed.fit_options)?;
    let (mut population, covariate_table) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data_path,
        None,
        iov_col,
        sel_filter.as_ref(),
        &parsed.column_map,
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

    // Hash both inputs up front (needed before the fit for the checkpoint
    // integrity check, #755) and reuse the digests for the post-fit result
    // stamping below — so we still hash each file only once. Errors are
    // non-fatal: a missing hash just disables the resume/integrity checks.
    let model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    let data_hash = crate::io::hash::sha256_file(Path::new(data_path)).ok();

    // Checkpoint / restart (#755): write `{model_stem}.tmp` next to the CLI
    // outputs and resume from it on a re-run of the same model + data. Disabled
    // by `[fit_options] checkpoint = false`. The CLI's `--clean` flag removes an
    // existing checkpoint before this runs, forcing a fresh start.
    if parsed.fit_options.checkpoint {
        let stem = Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        parsed.fit_options.checkpoint_path = Some(format!("{stem}.tmp"));
        parsed.fit_options.checkpoint_model_hash = model_hash.clone();
        parsed.fit_options.checkpoint_data_hash = data_hash.clone();
    }

    let mut result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    result.covariate_table = covariate_table;
    if let Some(w) = data_path_warning {
        result.warnings.push(w);
        rebuild_warnings_structured(&mut result);
    }
    result.model_path = Some(model_path.to_string());
    result.data_path = Some(data_path.to_string());
    result.model_hash = model_hash;
    result.data_hash = data_hash;
    result.model_text = std::fs::read_to_string(model_path).ok();
    derive_output_occasions(&parsed.model, &parsed.fit_options, &mut population);
    Ok((result, population))
}

/// Mirror [`fit`]'s model-side IOV occasion derivation onto a caller-owned
/// population so downstream output — sdtab's `OCC` column and any per-occasion
/// diagnostic keyed on [`Subject::occasions`] — reflects the derived labels.
/// `fit()` derives occasions only on its internal population clone, so without
/// this the returned population still has empty `occasions` under a
/// `dose`/`time(...)` rule and the OCC column silently vanishes versus an
/// equivalent `iov_column` run (#757). No-op unless the model declares kappa and
/// a derived rule is active; `had_column = false` + a throwaway sink because the
/// override warning already fired inside `fit()`.
fn derive_output_occasions(
    model: &CompiledModel,
    options: &FitOptions,
    population: &mut Population,
) {
    if model.n_kappa == 0 || options.iov_occasion == IovOccasionRule::Column {
        return;
    }
    let mut sink = Vec::new();
    apply_iov_occasion_rule(population, &options.iov_occasion, false, &mut sink);
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
    // CTMM (#759) cannot be simulated yet — sampling a discrete-state trajectory from
    // the generator is a later slice. Return a clear Err here rather than fall through
    // to the "nothing to simulate" message (CTMM-only) or the all-zero Gaussian emitter
    // (with `times`); the library `simulate()` chokepoint panics on the same case.
    #[cfg(feature = "markov")]
    if parsed.model.has_ctmm() {
        return Err(
            "[simulation]: a [markov_model] (CTMM) endpoint cannot be simulated yet — \
             sampling a discrete-state trajectory from the generator is a later slice. \
             fit() supports CTMM; --simulate does not."
                .to_string(),
        );
    }
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
            obs_l2: Vec::new(),
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
            // `obs_records` is unconditional since Phase 4.0, but the only records
            // this synthetic TTE subject carries are `Event`s (survival-gated); with
            // the feature off there is no TTE endpoint, so the vec is empty.
            #[cfg(not(feature = "survival"))]
            obs_records: Vec::new(),
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
    let SimulationOutput {
        results: sim_results,
        warnings: sim_warnings,
    } = simulate_with_options_diag(
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
    // Surface any per-subject simulation diagnostics (#762/#763) alongside the fit
    // warnings so the CLI reports a degenerate simulated subject rather than dropping it.
    // `fit()` already built `warnings_structured` from the fit warnings; rebuild it
    // unconditionally so the appended sim warnings also reach the typed / JSON surface
    // (#778/#779), where `classify_warning` tags them `WarningCode::Simulation`. The
    // rebuild is idempotent — a pure function of `result.warnings` — so an empty
    // `sim_warnings` (the common case, and every non-survival build) just reproduces the
    // fit's structured list.
    result.warnings.extend(sim_warnings);
    rebuild_warnings_structured(&mut result);
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

/// Reject an adaptive-dosing simulation on a covariate-selected error model (#658).
///
/// The adaptive assay resolves residual variance by the monitored **compartment**
/// number (`residual_variance_at(cmt, …)`), but `ErrorSpec::Selected`'s `endpoints`
/// map is keyed by the selector's **0-based branch index**, not CMT. A CMT-keyed
/// lookup misses and `variance_at` returns `NaN`, silently corrupting the assay
/// draw. The combination has no coherent meaning today, so reject it loudly rather
/// than emit NaN observations.
fn reject_selected_error_for_adaptive(model: &CompiledModel) -> Result<(), String> {
    if matches!(model.error_spec, ErrorSpec::Selected { .. }) {
        return Err(
            "adaptive-dosing simulation does not support a covariate-selected `[error_model]` \
             (`if (COV …) { … } else { … }`): the assay keys residual error by the monitored \
             compartment, but a selected error model keys endpoints by covariate branch. Use a \
             single-endpoint error model for the monitored signal."
                .to_string(),
        );
    }
    Ok(())
}

/// Reject the model / data combinations the reactive driver cannot yet simulate
/// *faithfully*. The adaptive path never applies a reset or carries process noise,
/// so a system reset (EVID=3/4) or an SDE `[diffusion]` model would each be
/// **silently** wrong — a violation of the "never a silent wrong answer" contract
/// this module promises. Until each is properly supported (#391 follow-ups), reject
/// it with a typed error. Both public entry points funnel through
/// `run_adaptive_population`, so guarding there covers `simulate_adaptive` and
/// `simulate_adaptive_from_spec`.
///
/// Time-varying covariates (and a `TIME`-in-PK model) are **no longer** rejected:
/// the driver recomputes PK per event/segment from the covariate active in that
/// segment (#700). Inter-occasion variability (IOV / `kappa`) is **no longer**
/// rejected either: a fresh κ is drawn per decision window and threaded through the
/// per-segment eta (#701), with occasion = decision index.
fn reject_unsupported_adaptive(
    model: &CompiledModel,
    population: &Population,
) -> Result<(), String> {
    if model.is_sde() {
        return Err(
            "adaptive-dosing simulation does not support stochastic (`[diffusion]` / SDE) \
             models: the reactive integrator is deterministic and would silently drop the \
             process-noise term. Use the deterministic ODE model for adaptive dosing."
                .to_string(),
        );
    }
    // Time-varying covariates (and `TIME`-in-PK) are now supported via per-event PK
    // recomputation in the reactive driver (#700); they are no longer rejected here.
    for subject in &population.subjects {
        if subject.has_resets() {
            return Err(format!(
                "adaptive-dosing simulation does not support system-reset events (EVID=3/4) \
                 (subject '{}'): the reactive driver never applies the reset, so the compartment \
                 state would silently fail to zero. Remove reset rows for adaptive runs.",
                subject.id
            ));
        }
    }
    Ok(())
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
    column_map: &[(String, String)],
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

    // Discrete-state CMTs route to `ObsRecord::DiscreteState` (integer DV), like TTE
    // rows route to `ObsRecord::Event` — both need the non-Gaussian reader path. This
    // set covers binary/categorical (#760) and CTMM (#759) endpoints, which share the
    // discrete-state plumbing.
    #[cfg(feature = "survival")]
    let binary_cmts: std::collections::HashSet<usize> = {
        let mut discrete: std::collections::HashSet<usize> =
            model.binary_cmts().into_iter().collect();
        #[cfg(feature = "markov")]
        discrete.extend(model.ctmm_cmts());
        discrete
    };
    #[cfg(not(feature = "survival"))]
    let binary_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    if tte_cmts.is_empty() && binary_cmts.is_empty() {
        // Gaussian-only model: use the existing (faster) path without TTE overhead.
        match (covariate_decls, filter) {
            (Some(decls), Some(sel)) => {
                let extra = undeclared_referenced(model, decls);
                let (pop, table) = read_nonmem_csv_with_covariates_filtered_mapped(
                    Path::new(data_path),
                    decls,
                    &extra,
                    iov_column,
                    sel,
                    column_map,
                )?;
                Ok((pop, Some(table)))
            }
            (Some(decls), None) => {
                let extra = undeclared_referenced(model, decls);
                let (pop, table) = read_nonmem_csv_with_covariates_mapped(
                    Path::new(data_path),
                    decls,
                    &extra,
                    iov_column,
                    column_map,
                )?;
                Ok((pop, Some(table)))
            }
            (None, Some(sel)) => Ok((
                read_nonmem_csv_filtered_mapped(
                    Path::new(data_path),
                    fallback_columns,
                    iov_column,
                    sel,
                    column_map,
                )?,
                None,
            )),
            (None, None) => Ok((
                read_nonmem_csv_mapped(
                    Path::new(data_path),
                    fallback_columns,
                    iov_column,
                    column_map,
                )?,
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
                    &binary_cmts,
                    column_map,
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
                    &binary_cmts,
                    column_map,
                )?;
                Ok((pop, None))
            }
        }
    }
}

/// High-level fit: model file path + data file path → FitResult
///
/// `data_path` is `None` to rely solely on the model's own `[data]` block
/// (#690); when both are given, `data_path` overrides the model's, with a
/// warning recorded on the result (see [`resolve_data_path`]).
pub fn fit_from_files(
    model_path: &str,
    data_path: Option<&str>,
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
    let (data_path, data_path_warning) = resolve_data_path(parsed.data_path.as_deref(), data_path)?;
    let data_path = data_path.as_str();
    let (population, covariate_table) = read_population_for(
        &model,
        &parsed.covariate_decls,
        data_path,
        covariate_columns,
        None,
        sel_filter_fit.as_ref(),
        &parsed.column_map,
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
    if let Some(w) = data_path_warning {
        result.warnings.push(w);
        rebuild_warnings_structured(&mut result);
    }
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

/// Multi-start ranking: should the candidate `(c_ofv, c_conv)` replace the
/// current best `(b_ofv, b_conv)`?
///
/// **Validity is the primary key.** A run whose OFV is non-finite or
/// sentinel-large (block_sigma `R` gone indefinite, a divergent peripheral
/// compartment, …) is a failed fit even if it reports `converged = true` — the
/// inner objective returns the ~1e20 sentinel and the outer optimizer can
/// "converge" on it. Such a run must never beat a valid but unconverged start
/// (e.g. the exact-inits start 0), which the old "converged-first" rule allowed,
/// so multi-start could return a divergence with a huge OFV. Within the same
/// validity class: prefer converged, then lower OFV.
///
/// The validity cutoff sits well below the ~1e20 inner-objective sentinel but
/// far above any legitimate population OFV, so a real fit never trips it; a NaN
/// OFV is non-finite and therefore also invalid, so it can never block a finite
/// valid candidate.
fn multistart_prefers(b_ofv: f64, b_conv: bool, c_ofv: f64, c_conv: bool) -> bool {
    /// OFVs at or above this are treated as diverged/invalid (see above).
    const DIVERGENCE_OFV: f64 = 1e14;
    let valid = |o: f64| o.is_finite() && o < DIVERGENCE_OFV;
    match (valid(b_ofv), valid(c_ofv)) {
        (false, true) => true,
        (true, false) => false,
        _ => (!b_conv && c_conv) || (b_conv == c_conv && c_ofv < b_ofv),
    }
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
    // Start the SS-equilibration non-convergence sink clean so a prior in-process call's residue
    // can't leak into this fit's warnings; drained back out just before `Ok(result)` (#867).
    crate::dosing::clear_ss_nonconvergence_warnings();
    // Reject one_cpt_transit + unsupported feature (SS/IOV/TV-cov/infusion, #386)
    // before any prediction reaches the superposition dispatch's `unreachable!` arms.
    if let Some(e) = check_absorption_closed_form_support(model, population) {
        return Err(e);
    }
    // Reject a twin-less transit closed form that starts in the flip-flop regime (#776):
    // it returns an identically-zero profile that silently degenerates the objective and,
    // unlike a twin-carrying model, has no ODE twin to reroute to. Parameter-dependent, so
    // evaluated at η = 0 typical values here rather than in `check_absorption_closed_form_support`.
    if let Some(e) = check_absorption_flip_flop_no_twin(model, population, &init_params.theta) {
        return Err(e);
    }
    // Reject a depot-referencing analytic Form C readout on reset subjects (#650):
    // the depot amount can't be superposed across an EVID=3/4 reset, so predicting
    // would silently read a zero depot. Fail loudly instead.
    if let Some(e) = check_analytic_readout_support(model, population) {
        return Err(e);
    }
    // RTTE (repeated-event) likelihoods — clock-forward telescoping and clock-reset
    // gap-time — require time-sorted records and do not support interval-censored /
    // non-finite times, all of which would otherwise fold into a silent 1e20 sentinel.
    // Reject a hand-built model/dataset up front.
    #[cfg(feature = "survival")]
    if let Some(e) = check_rtte_records(model, population) {
        return Err(e);
    }
    // A survival hazard that references a time-varying covariate would be silently
    // evaluated at the baseline value (analytic hazard) or with PK params frozen at t=0
    // (ODE-accumulated hazard) — reject rather than mis-fit (#741).
    #[cfg(feature = "survival")]
    if let Some(e) = check_survival_tv_covariates(model, population) {
        return Err(e);
    }
    // Binary/categorical (#760): reject an observed state outside {0,1} up front. The
    // datareader accepts any non-negative integer DV on a discrete CMT (it can't tell
    // binary from ordinal), so the Bernoulli endpoint must reject `state ≥ 2` itself —
    // fail-loud rather than silently score a non-binary code.
    #[cfg(feature = "survival")]
    for cmt in model.binary_cmts() {
        for subject in &population.subjects {
            crate::categorical::validate_binary_states(cmt, &subject.obs_records)?;
        }
    }
    // CTMM (#759): reject an observed DV code that is not one of the endpoint's declared
    // `states` up front (the datareader accepts any non-negative integer on a discrete
    // CMT). Otherwise the code→index map would miss and fold into a silent sentinel.
    #[cfg(feature = "markov")]
    for (cmt, endpoint) in &model.endpoints {
        if let EndpointLikelihood::Ctmm {
            state_codes,
            generator_states,
            ..
        } = endpoint
        {
            // Time-inhomogeneous (drug/PD-driven Q(t), #817): an intensity references a
            // model ODE state, scored by the per-gap occupancy integration in
            // `ctmm_endpoint_nll_inhomogeneous`. That requires an ODE model — guaranteed at
            // parse (`generator_states` indices point into `ode_spec`), asserted here for a
            // hand-built `CompiledModel`.
            if !generator_states.is_empty() && model.ode_spec.is_none() {
                return Err(format!(
                    "[markov_model] cmt = {cmt}: a transition intensity references a model state \
                     (time-inhomogeneous Q(t)), but the model has no ODE system to supply it."
                ));
            }
            for subject in &population.subjects {
                crate::markov::endpoint::validate_ctmm_states(
                    *cmt,
                    state_codes,
                    &subject.obs_records,
                )?;
                // Out-of-order observation times would otherwise silently collapse the
                // subject to the 1e20 sentinel (datareader sorts doses, not obs).
                crate::markov::endpoint::validate_ctmm_times(*cmt, &subject.obs_records)?;
            }
        }
    }
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
    first_error(&check_model_data_rule(
        model,
        population,
        &options.iov_occasion,
    ))?;
    // AGQ + IOV: the tensor grid is `n_agq^d` with `d = n_eta + K·n_kappa`, and `K` (the
    // occasion count) is a property of the *data*, not the model — so this cap cannot be
    // checked in `check_model_options`, which never sees the population. Check it here, once
    // the occasions are known, against the worst subject.
    //
    // `method = laplace` / `focei` at `n_agq = 1` is a single node regardless of `d`, so it
    // always passes; only `n_agq > 1` (adaptive quadrature) can trip this. `agq_nodes()`
    // fires for both quadrature methods, so name whichever the user actually wrote (#251
    // review #4) rather than hardcoding "laplace".
    if let Some(n_nodes) = options.agq_nodes() {
        if model.n_kappa > 0 {
            let max_occ = population
                .subjects
                .iter()
                .map(|s| crate::stats::likelihood::iov_occasion_groups(s).len())
                .max()
                .unwrap_or(0);
            let d = model.n_eta + max_occ * model.n_kappa;
            let grid = crate::estimation::agq::grid_size(n_nodes, d);
            if grid > crate::estimation::agq::MAX_AGQ_GRID {
                let label = if matches!(options.method, EstimationMethod::Laplace) {
                    "laplace"
                } else {
                    "focei"
                };
                return Err(format!(
                    "method = {label} with n_agq = {n_nodes} needs a {n_nodes}^{d} = {grid}-node \
                     tensor grid per subject per iteration — over the {} limit. Under IOV the \
                     integral is over the stacked (η, κ₁..κ_{max_occ}), so the dimension is \
                     n_eta ({}) + occasions ({max_occ}) × n_kappa ({}) = {d}. Lower n_agq \
                     (n_agq = 1 is the single-node method — always tractable), or use \
                     saem / imp, whose cost does not grow with the random-effect dimension.",
                    crate::estimation::agq::MAX_AGQ_GRID,
                    model.n_eta,
                    model.n_kappa,
                ));
            }
        }
    }
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
    // A model-side IOV occasion rule (`iov_occasion = dose | time(...)`) derives
    // the occasion labels from each subject's timeline instead of a data column.
    // Only meaningful when the model actually declares kappa: without it the
    // derived labels feed nothing, so skip the clone + derivation entirely and
    // tell the user the option is a no-op rather than silently paying for it.
    let iov_rule_set = options.iov_occasion != IovOccasionRule::Column;
    if iov_rule_set && model.n_kappa == 0 {
        ltbs_warnings.push(
            "iov_occasion is set in [fit_options] but the model declares no kappa (IOV) \
             random effects, so it has no effect and no occasions were derived."
                .to_string(),
        );
    }
    let derive_occ = iov_rule_set && model.n_kappa > 0;
    let pop_pruned: std::borrow::Cow<Population> = {
        let needs_prune = population.subjects.iter().any(|s| {
            !s.dose_covariates.is_empty()
                || !s.obs_covariates.is_empty()
                || !s.pk_only_covariates.is_empty()
        });
        if needs_prune || needs_dv_log || derive_occ {
            let mut p = population.clone();
            if needs_prune {
                p.prune_irrelevant_tv_covariates(&model.referenced_covariates);
            }
            if derive_occ {
                apply_iov_occasion_rule(
                    &mut p,
                    &options.iov_occasion,
                    options.iov_column.is_some(),
                    &mut ltbs_warnings,
                );
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

    // A model-side occasion rule must actually partition the timeline. The
    // dataset-column path is guarded by `E_IOV_MISSING_OCC`; the derived path
    // suppresses that check (labels arrive at fit time), so re-establish the same
    // invariant on the *derived* labels here (#757):
    //   - if every subject collapses to a single occasion, the per-occasion kappa
    //     is added once per subject exactly like an eta — confounded with
    //     between-subject variability and unidentifiable — so error, as the
    //     column path would have;
    //   - if the count explodes (e.g. `iov_occasion = dose` on an ADDL train,
    //     where each expanded dose opens an occasion), the inner per-subject
    //     problem balloons; warn so the multiplication isn't silent.
    if derive_occ {
        // Distinct occasion labels seen across observations and doses, per subject.
        let distinct_occ = |s: &Subject| -> usize {
            let mut occs: Vec<u32> = s.occasions.clone();
            occs.extend_from_slice(&s.dose_occasions);
            occs.sort_unstable();
            occs.dedup();
            occs.len()
        };
        let max_occ = pop_ref.subjects.iter().map(distinct_occ).max().unwrap_or(0);
        if max_occ <= 1 {
            return Err(
                "The `iov_occasion` rule produced only a single occasion for every subject, \
                 so the per-occasion kappa (IOV) cannot be separated from between-subject \
                 variability (it is unidentifiable). Use `iov_occasion = time(...)` with \
                 breakpoints inside the sampling window, `iov_occasion = dose` on a \
                 multiple-dose design, or supply an `iov_column`."
                    .to_string(),
            );
        }
        // Above this many occasions per subject the derivation has almost
        // certainly over-partitioned (e.g. a long ADDL dose train); the fit still
        // runs but the inner kappa dimensionality is n_iov * max_occ per subject.
        const IOV_OCCASION_WARN_LIMIT: usize = 20;
        if max_occ > IOV_OCCASION_WARN_LIMIT {
            ltbs_warnings.push(format!(
                "iov_occasion derived {max_occ} occasions for at least one subject — the \
                 inner per-subject problem grows with the occasion count. If this came from \
                 an ADDL/steady-state dose train, prefer `iov_occasion = time(...)` windows \
                 or an `iov_column` that groups administrations into fewer occasions."
            ));
        }
    }

    // Single-start fast path (default)
    if options.n_starts <= 1 {
        let run = || fit_inner(model, pop_ref, init_params, options);
        // A pinned positive `threads` builds a fit-scoped pool sized to it (and surfaces a
        // build failure as `Err`, as before). The unpinned default reuses the shared
        // big-stack pool; if that one-time build ever failed we run on the ambient pool
        // rather than turning a previously-successful default fit into an `Err`.
        let res = match options.threads.filter(|&n| n > 0) {
            Some(n) => build_fit_pool(n)?.install(run),
            None => match default_fit_pool() {
                Some(pool) => pool.install(run),
                None => run(),
            },
        };
        return res.map(|mut result| {
            result.warnings.splice(0..0, ltbs_warnings);
            // Surface any SS-equilibration non-convergence seen during this (default, single-start)
            // fit's prediction passes (#867). This is the common path — `n_starts` defaults to 1 —
            // so it must drain the sink too, not just the multi-start arm below.
            for w in crate::dosing::take_ss_nonconvergence_warnings() {
                result.warnings.push(w);
            }
            rebuild_warnings_structured(&mut result);
            result
        });
    }

    // Multi-start: run n_starts fits in parallel, return the lowest-OFV converged result.
    // `threads` controls per-subject parallelism inside a single-start fit; in multi-start
    // mode the shared fit pool handles both levels (outer start × inner per-subject), so we
    // do not narrow it to `threads` here — that would cap the combined fan-out below the
    // available cores. Running one shared pool (rather than a fresh ThreadPool per start
    // inside the outer into_par_iter()) also avoids spawning n_starts independent pools that
    // all compete on the same CPUs, causing oversubscription.
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

    let par_starts = || -> Vec<(usize, Result<FitResult, String>)> {
        (0..n)
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
            .collect()
    };

    // Run the start fan-out on the shared big-stack pool; fall back to the ambient pool
    // only if its one-time build failed.
    let results: Vec<(usize, Result<FitResult, String>)> = match default_fit_pool() {
        Some(pool) => pool.install(par_starts),
        None => par_starts(),
    };

    // Pick the best start (see `multistart_prefers` for the ranking).
    let mut best: Option<(usize, FitResult)> = None;
    let mut failed_starts: Vec<String> = Vec::new();
    for (k, res) in results {
        match res {
            Ok(r) => {
                let better = match &best {
                    None => true,
                    Some((_, b)) => multistart_prefers(b.ofv, b.converged, r.ofv, r.converged),
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
            // Surface any SS-equilibration non-convergence seen during this fit's prediction passes
            // (#867). The capped nonlinear pulse-train can silently under-report the SS trough and
            // bias estimates low; the sink deduplicated it across every objective evaluation and
            // multi-start replicate to a single message.
            for w in crate::dosing::take_ss_nonconvergence_warnings() {
                result.warnings.push(w);
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
///
/// Each message is classified into a typed [`crate::types::WarningCode`] +
/// severity and — for the numeric diagnostics whose values are already stored
/// as typed fields on the `FitResult` — enriched with a machine-readable
/// `details` payload sourced from those fields (issue #781, increment 1). The
/// numbers thus come from typed state, not from re-parsing the message prose;
/// converting the remaining ~80 string push-sites to full at-source emission is
/// a later increment.
fn rebuild_warnings_structured(result: &mut FitResult) {
    // Entries emitted typed at source (`native_warnings`, already carrying a
    // `details` payload) are keyed by message and take precedence; every other
    // message is string-classified and then field-enriched. Keying by message
    // keeps this idempotent — a second rebuild re-captures the native entries.
    // Key by the *original* warning string: `classify_warning` strips a
    // `[METHOD]` chain prefix into `source_method`, so a native entry produced
    // in a multi-stage chain must reconstruct that prefix to match the raw
    // `warnings` entry (an unprefixed entry keys by its message unchanged).
    let native: std::collections::HashMap<String, crate::types::WarningEntry> =
        std::mem::take(&mut result.warnings_structured)
            .into_iter()
            .map(|e| {
                let key = match &e.source_method {
                    Some(m) => format!("[{m}] {}", e.message),
                    None => e.message.clone(),
                };
                (key, e)
            })
            .collect();
    let stats = DiagStats {
        dw_statistic: result.dw_statistic,
        iwres_lag1_r: result.iwres_lag1_r,
        shrinkage_eps: result.shrinkage_eps,
        cov_condition_number: result.cov_condition_number,
        cov_eigenvalues: result.cov_eigenvalues.as_deref(),
        shrinkage_eta: &result.shrinkage_eta,
        eta_names: &result.eta_names,
    };
    let entries: Vec<crate::types::WarningEntry> = result
        .warnings
        .iter()
        .map(|w| {
            if let Some(entry) = native.get(w) {
                entry.clone()
            } else {
                let mut e = crate::types::classify_warning(w);
                e.details = diagnostic_details(&e.category, &stats);
                e
            }
        })
        .collect();
    result.warnings_structured = entries;
}

/// Typed inputs for [`diagnostic_details`], sourced from `FitResult`'s fields.
/// `Default` gives finite-zero scalars, `None` options, and empty slices, so a
/// test can set only the fields a case needs via struct-update syntax.
#[derive(Default)]
struct DiagStats<'a> {
    dw_statistic: f64,
    iwres_lag1_r: f64,
    shrinkage_eps: f64,
    cov_condition_number: Option<f64>,
    cov_eigenvalues: Option<&'a [f64]>,
    shrinkage_eta: &'a [f64],
    eta_names: &'a [String],
}

/// Machine-readable `details` payload for the numeric diagnostic warning codes,
/// sourced from the typed `FitResult` fields rather than parsed from the message
/// text. A non-finite (`NaN`/`±Inf`) statistic is never emitted as a JSON `null`
/// (`serde_json` maps non-finite floats to `null`) — instead:
///
/// - the single/paired-stat codes (`DwAutocorrelation`, `EpsShrinkage`,
///   `ConditionNumber`) return `None` (whole `details` key omitted) when a
///   required statistic is missing or non-finite;
/// - the covariance codes (`CovarianceFailed`, `CovarianceRegularized`) build a
///   partial object, **skipping** individual non-finite/absent fields and
///   emitting whatever is available; they return `None` only when nothing is.
///
/// `cov_condition_number` in particular is documented as `+Inf` for a
/// near-singular parameter space, so it is dropped from the payload there.
fn diagnostic_details(
    code: &crate::types::WarningCode,
    s: &DiagStats,
) -> Option<serde_json::Value> {
    use crate::types::WarningCode;
    match code {
        WarningCode::DwAutocorrelation
            if s.dw_statistic.is_finite() && s.iwres_lag1_r.is_finite() =>
        {
            Some(serde_json::json!({
                "durbin_watson": s.dw_statistic,
                "iwres_lag1_autocorr": s.iwres_lag1_r,
            }))
        }
        WarningCode::EpsShrinkage if s.shrinkage_eps.is_finite() => Some(serde_json::json!({
            "eps_shrinkage": s.shrinkage_eps,
            "eps_shrinkage_pct": 100.0 * s.shrinkage_eps,
        })),
        WarningCode::EtaShrinkage => {
            // The high-shrinkage ETAs behind the message, with each shrinkage
            // as a percent — sourced from the typed `shrinkage_eta` field.
            let high: Vec<serde_json::Value> = high_shrinkage_eta_indices(s.shrinkage_eta)
                .into_iter()
                .map(|i| {
                    serde_json::json!({
                        "eta": eta_label(s.eta_names, i),
                        "shrinkage_pct": 100.0 * s.shrinkage_eta[i],
                    })
                })
                .collect();
            if high.is_empty() {
                None
            } else {
                Some(serde_json::json!({
                    "threshold_pct": 100.0 * ETA_SHRINKAGE_WARN_THRESHOLD,
                    "high_shrinkage_etas": high,
                }))
            }
        }
        WarningCode::ConditionNumber => match s.cov_condition_number {
            Some(c) if c.is_finite() => Some(serde_json::json!({ "condition_number": c })),
            _ => None,
        },
        WarningCode::CovarianceFailed | WarningCode::CovarianceRegularized => {
            covariance_details(s.cov_condition_number, s.cov_eigenvalues)
        }
        _ => None,
    }
}

/// `details` for the covariance-step warning codes: the condition number and,
/// when the covariance matrix was produced (so eigenvalues exist — typically
/// the regularized case, not a hard failure), the smallest eigenvalue and the
/// count of negative ones (which diagnose a non-PD Hessian). Non-finite values
/// are skipped; returns `None` if nothing usable is available.
fn covariance_details(
    cov_condition_number: Option<f64>,
    cov_eigenvalues: Option<&[f64]>,
) -> Option<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    if let Some(c) = cov_condition_number {
        if c.is_finite() {
            obj.insert("condition_number".to_string(), serde_json::json!(c));
        }
    }
    if let Some(eigs) = cov_eigenvalues {
        let finite: Vec<f64> = eigs.iter().copied().filter(|v| v.is_finite()).collect();
        if !finite.is_empty() {
            let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let n_neg = finite.iter().filter(|&&v| v < 0.0).count();
            obj.insert("min_eigenvalue".to_string(), serde_json::json!(min));
            obj.insert(
                "n_negative_eigenvalues".to_string(),
                serde_json::json!(n_neg),
            );
        }
    }
    if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj))
    }
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
/// values from `pk`. Individual parameters the parser synthesized for a direct-θ/η
/// Form-C readout (`__ferx_ro_*`, #486) are internal — they are skipped so they never
/// surface as a user-facing EBE / sdtab column.
fn build_indiv_map(pk: &PkParams, names: &[String], pk_indices: &[usize]) -> HashMap<String, f64> {
    names
        .iter()
        .zip(pk_indices.iter())
        .filter(|(name, _)| !crate::parser::model_parser::is_synthetic_readout_param(name))
        .map(|(name, &idx)| (name.clone(), pk.values[idx]))
        .collect()
}

#[cfg(test)]
#[path = "tests/multistart_prefers_tests.rs"]
mod multistart_prefers_tests;

#[cfg(test)]
#[path = "tests/build_indiv_map_tests.rs"]
mod build_indiv_map_tests;

/// Trapezoid integration over (time, value) pairs.
/// Observation times are not guaranteed to be sorted (preserved in input row
/// order), so sort by time before integrating to prevent negative dt windows.
///
/// `pub(crate)` so the reactive-dosing signal-AUC pass
/// ([`crate::ode::adaptive_window_signal_aucs`], #391 S2.5b) shares this one
/// implementation rather than carrying a second copy of the rule.
pub(crate) fn trapezoid(points: &[(f64, f64)]) -> f64 {
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
        let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
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
                    // Evaluate at this dose's time so a `TIME`-dependent lag (or
                    // any TIME-built-in parameter) is honoured per dose (#610).
                    let pk_d = (model.pk_param_fn)(
                        theta,
                        &eta_d,
                        subject.dose_cov(d),
                        subject.doses[d].time,
                    );
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
            // Evaluate at this observation's time so the sdtab individual-parameter
            // columns honour the `TIME` built-in per row, matching IPRED (#610).
            let pk_j = (model.pk_param_fn)(theta, eta_full, cov_j, subject.obs_times[j]);
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
                            } else if crate::parser::model_parser::compiled_model_uses_time_builtin(
                                model,
                            ) {
                                // TIME built-in in the individual parameters: a single
                                // fixed PK snapshot (grid_cov at one time) cannot
                                // represent parameters that vary along the grid, while
                                // ipred honours each event's time via the event-driven
                                // path. Return empty so every grid point evaluates to
                                // NaN — the same convention as IOV/TV/reset above, and
                                // consistent with compute_predictions_with_states (#610).
                                vec![]
                            } else if let Some(ref ode) = model.ode_spec {
                                // Time-independent params: one snapshot (t=0) is exact
                                // across the grid; the ODE solver supplies its own clock.
                                let pk_j = (model.pk_param_fn)(theta, grid_eta_full, grid_cov, 0.0);
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
                                // Time-independent params: one snapshot (t=0) is exact
                                // across the grid (uses_time handled above).
                                let pk_j = (model.pk_param_fn)(theta, grid_eta_full, grid_cov, 0.0);
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

fn saem_non_mu_referenced_individual_params_warning(model: &CompiledModel) -> Option<String> {
    let mut names = Vec::new();
    for (param_name, &eta_idx) in model.indiv_param_names.iter().zip(model.eta_map.iter()) {
        if eta_idx < 0 {
            continue;
        }
        let Some(eta_name) = model.eta_names.get(eta_idx as usize) else {
            continue;
        };
        if !model.mu_refs.contains_key(eta_name) {
            names.push(param_name.as_str());
        }
    }

    if names.is_empty() {
        None
    } else {
        Some(format!(
            "individual parameter(s) not mu-referenced: {}. This can strongly \
             affect convergence; prefer forms such as `CL = TVCL * exp(ETA_CL)` when possible.",
            names.join(", ")
        ))
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // LTBS needs the inner EBE loop converged tighter than the default for
    // reproducible standard errors (see `FitOptions::effective_inner_tol` /
    // `LTBS_FIT_INNER_TOL`). Resolve it once here so both the outer optimisation and
    // the covariance step (which reconverges tighter still via `effective_cov_inner_tol`)
    // work from the same tightened tolerance. `min` never loosens an explicit
    // user setting; non-LTBS models are untouched (same `options` reference).
    let ltbs_opts;
    let options = {
        let eff = options.effective_inner_tol(model.uses_closed_form_ltbs_inner());
        if eff < options.inner_tol {
            ltbs_opts = FitOptions {
                inner_tol: eff,
                ..options.clone()
            };
            &ltbs_opts
        } else {
            options
        }
    };
    let fit_start = Instant::now();
    let chain = options.method_chain();
    let n_stages = chain.len();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let mut pre_run_warnings = options.unsupported_keys_warnings();
    // Surface a "no method specified, defaulting to FOCEI" notice through the
    // same channel so it reaches both stderr and FitResult.warnings.
    if let Some(w) = options.method_default_warning() {
        pre_run_warnings.push(w);
    }
    // Emitted whenever the chain runs SAEM, independent of `options.mu_referencing`:
    // the non-mu-referenced case is *most* at risk under SAEM when mu-centering is
    // off, so gating on the opt-in flag would silence the warning exactly when it
    // matters most (#621). This is assembled before the startup banner so verbose
    // runs show it before SAEM begins.
    if chain.iter().any(|&m| m == EstimationMethod::Saem) {
        if let Some(w) = saem_non_mu_referenced_individual_params_warning(model) {
            pre_run_warnings.push(if n_stages > 1 {
                format!("[SAEM] {w}")
            } else {
                format!("SAEM: {w}")
            });
        }
    }

    // RTTE (repeated time-to-event) fit under a Laplace-based method (FOCE/FOCEI/GN)
    // severely underestimates the frailty variance ω² at low event rates (Karlsson et
    // al. 2009: −91% to −96% bias below a 43% event rate). SAEM and IMP are unbiased.
    // Warn but keep Laplace available — it is fast and fine for fixed-effects or
    // high-event-rate RTTE. §3.3 of `plans/tte-survival-markov.md`. Two gates: (1) the
    // model carries a frailty (`n_eta > 0`); a fixed-effects RTTE fit has no ω² to
    // underestimate, so the warning would be spurious. (2) The **final/estimating** stage
    // must be Laplace-based — a warm-start chain like `[focei, imp]` or `[saem, focei]`
    // whose terminal estimator is SAEM/IMP is already unbiased and must not warn.
    // (`n_eta > 0` can over-fire on a joint model whose PK — not hazard — carries the
    // etas; that spurious *advisory* is preferable to probing the hazard, which silently
    // misses a covariate-gated frailty like `exp(ETA·WT)` and drops the warning entirely.)
    #[cfg(feature = "survival")]
    if model.has_rtte()
        && model.n_eta > 0
        && matches!(
            chain.last(),
            Some(
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::FoceGn
                    | EstimationMethod::FoceGnHybrid
            )
        )
    {
        pre_run_warnings.push(
            "RTTE fit under a Laplace-based method (FOCE/FOCEI/GN) can severely \
             underestimate the frailty variance ω² at low event rates (Karlsson et al. \
             2009). Prefer `method = saem` or `method = imp`; Laplace remains available \
             for fixed-effects or high-event-rate RTTE."
                .to_string(),
        );
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
        // Header carries one `val:<name>`/`grad:<name>` column per optimized
        // coordinate. The coordinate structure (and hence the names) is fixed
        // across a method chain, so the template's names serve every stage.
        let coord_names = crate::estimation::parameterization::coordinate_names(init_params);
        if let Err(e) = crate::estimation::trace::init(path.clone(), &coord_names) {
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
        if !pre_run_warnings.is_empty() {
            eprintln!("--- Warnings ---");
            for w in &pre_run_warnings {
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
        // AGQ included: it runs the same inner EBE solve as FOCE/FOCEI (it only replaces
        // the *population* objective), so the inner analytic-vs-FD η-gradient route this
        // reports is exactly as relevant to it.
        let uses_gradient_route = chain.iter().any(|m| {
            matches!(
                m,
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::FoceGn
                    | EstimationMethod::FoceGnHybrid
                    | EstimationMethod::Imp
                    | EstimationMethod::Impmap
                    | EstimationMethod::Laplace
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
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = model.parse_warnings.clone();
    accumulated_warnings.extend(pre_run_warnings);
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
                | EstimationMethod::Laplace
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
    // Per-stage convergence wall time, parallel to `chain`/`method_chain`
    // (#713). Excludes the covariance step, which is timed separately below
    // and only ever runs on the last estimating stage.
    let mut method_wall_times_secs: Vec<f64> = Vec::with_capacity(n_stages);
    let mut covariance_wall_time_secs: f64 = 0.0;

    // ── Checkpoint / restart (#755) ──────────────────────────────────────────
    // With a checkpoint path configured, resume from a compatible saved
    // checkpoint (unless it fails the model/data integrity or layout check) and
    // arm the periodic writer. `start_stage` skips chain stages that were
    // already completed before the interruption.
    //
    // Clear any sink left active by a *previous* fit on this (pooled) worker
    // thread that returned early with an error and so never hit
    // `finish_success`. Without this, a subsequent checkpoint-disabled fit —
    // which skips `init` below — would inherit the stale sink and leak writes /
    // removals to the old path.
    crate::io::checkpoint::abandon();
    let mut start_stage = 0usize;
    if options.checkpoint {
        if let Some(ckpt_path) = options.checkpoint_path.as_deref() {
            let coord_names = crate::estimation::parameterization::coordinate_names(init_params);
            if let Some(cp) = crate::io::checkpoint::load(ckpt_path) {
                // Resume only when both integrity hashes are present *and* match.
                // Treating absent hashes (None == None) as a match would let a
                // run resume with no integrity guarantee at all — exactly the
                // stale-state resume this check exists to prevent. The file
                // runner always supplies both hashes; a direct `fit()` caller
                // that omits them simply starts fresh.
                let hashes_present = cp.model_hash.is_some()
                    && cp.data_hash.is_some()
                    && options.checkpoint_model_hash.is_some()
                    && options.checkpoint_data_hash.is_some();
                let hashes_match = cp.model_hash == options.checkpoint_model_hash
                    && cp.data_hash == options.checkpoint_data_hash;
                let hashes_ok = hashes_present && hashes_match;
                let layout_ok = cp.coord_names == coord_names
                    && cp.stage_idx < n_stages
                    && cp.packed.len()
                        == crate::estimation::parameterization::packed_len(init_params);
                if hashes_ok && layout_ok {
                    stage_params =
                        crate::estimation::parameterization::unpack_params(&cp.packed, init_params);
                    start_stage = cp.stage_idx;
                    let banner = format!(
                        "Resuming from checkpoint {}: stage {}/{} ({}), iteration {}, OFV {:.4}. \
                         Pass --clean to start fresh.",
                        ckpt_path,
                        cp.stage_idx + 1,
                        n_stages,
                        chain.get(cp.stage_idx).map(|m| m.label()).unwrap_or("?"),
                        cp.iter,
                        cp.ofv,
                    );
                    if options.verbose {
                        eprintln!("{banner}");
                    }
                    accumulated_warnings.push(banner);
                } else {
                    // Provably stale/incompatible = hashes present but different,
                    // or a layout mismatch. Only then delete the file. When hashes
                    // are simply unavailable we can't prove staleness, so we start
                    // fresh but keep the checkpoint (a later run that does supply
                    // hashes may still use it; a successful fit removes it anyway).
                    let stale = (hashes_present && !hashes_match) || !layout_ok;
                    let reason = if hashes_present && !hashes_match {
                        "model or data changed since it was written"
                    } else if !layout_ok {
                        "its parameter layout does not match this model"
                    } else {
                        "its model/data integrity hashes are unavailable"
                    };
                    let msg = format!(
                        "Ignoring checkpoint {} ({}); starting fresh.",
                        ckpt_path, reason
                    );
                    if options.verbose {
                        eprintln!("{msg}");
                    }
                    accumulated_warnings.push(msg);
                    if stale {
                        crate::io::checkpoint::remove(ckpt_path);
                    }
                }
            }
            crate::io::checkpoint::init(
                ckpt_path.to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
                options.checkpoint_model_hash.clone(),
                options.checkpoint_data_hash.clone(),
                chain.iter().map(|m| m.label().to_string()).collect(),
                coord_names,
                options.checkpoint_interval_secs,
            );
        }
    }

    for (stage_idx, &method) in chain.iter().enumerate() {
        // Resume: stages before the checkpoint's stage are already done. Record
        // zero wall time for them so `method_wall_times_secs` stays parallel to
        // `chain` (downstream reporting zips the two).
        if stage_idx < start_stage {
            method_wall_times_secs.push(0.0);
            continue;
        }
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }
        crate::io::checkpoint::set_stage(stage_idx);
        let stage_start = std::time::Instant::now();
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
        // AGQ resolves `auto` to L-BFGS, because it *has* a gradient: the analytic
        // posterior-weighted score over the quadrature nodes (`estimation::agq`), with the
        // reconverged-FD gradient as the always-correct fallback.
        //
        // BOBYQA — what `resolve_auto` hands every other FD-only fit — **false-converges**
        // on this objective (the #317 failure mode): on warfarin it stops 0.015 OFV units
        // above the optimum at the default `inner_tol`, and *worsens* to 0.18 as `inner_tol`
        // is tightened, because a smoother objective merely lets its trust-region stopping
        // rule trip sooner (63 -> 34 evaluations, still reporting "converged"). Its ω
        // estimates land ~3% off NONMEM LAPLACIAN. L-BFGS on the analytic gradient matches
        // NONMEM to 4-5 significant figures on every parameter, in 0.39 s against FOCEI's
        // 0.29 s. See docs/estimation/agq.qmd. An explicit `optimizer = ...` is honoured.
        if matches!(method, EstimationMethod::Laplace) && stage_opts.optimizer == Optimizer::Auto {
            stage_opts.optimizer = Optimizer::NloptLbfgs;
        }
        // The **quadrature** methods need a tighter EBE than plain FOCE/FOCEI: their analytic
        // grid gradient assumes b̂ is exactly the mode (Bartlett identity), so a loose b̂ from
        // a poor start yields a non-descent direction that stalls the outer optimizer.
        // Measured on warfarin (#251): the shared default 1e-5 fails to converge from a 1.3×
        // start, while 1e-8 is robust across realistic starts at negligible cost near the
        // optimum (the OFV is insensitive to inner_tol there). This is anchor-independent —
        // `focei` with `n_agq > 1` reuses the same mode-dependent gradient (#251 review #3),
        // so it needs the tighter EBE too; only plain FOCE/FOCEI (`agq_nodes() == None`, whose
        // Gauss-Newton `log|H̃|` is forgiving of a loose b̂) keeps the looser default. `.min`
        // so an explicit tighter `inner_tol` is preserved.
        if stage_opts.agq_nodes().is_some() {
            stage_opts.inner_tol = stage_opts.inner_tol.min(1e-8);
        }
        // Run the covariance step (and SIR) only on the last *estimating* stage,
        // so a chain doesn't recompute the expensive FD covariance after every
        // method (#615). See `is_last_estimating_stage` for the eval-only-IMP rule.
        let is_last_estimating = is_last_estimating_stage(&chain, stage_idx, options.imp_eval_only);
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
                        stage_opts.inner_restarts,
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
                    covariance_wall_time_secs: 0.0,
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
                    packed_estimate: None,
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
            method_wall_times_secs.push(stage_start.elapsed().as_secs_f64());
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

        let stage_cov_secs = stage_result.covariance_wall_time_secs;
        method_wall_times_secs
            .push((stage_start.elapsed().as_secs_f64() - stage_cov_secs).max(0.0));
        covariance_wall_time_secs += stage_cov_secs;

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
    // Warnings emitted *typed at source* (with a `details` payload) rather than
    // string-classified. Seeded into `FitResult.warnings_structured`;
    // `rebuild_warnings_structured` prefers these over re-classifying the string.
    let mut native_warnings: Vec<WarningEntry> = Vec::new();

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

    // M3 censoring under non-interaction FOCE is a consistent Sheiner–Beal fit (the
    // censored rows leave the linearized marginal and re-enter as `−logΦ` at the
    // population variance; plain FOCE no longer promotes censored subjects to interaction
    // — non-IOV since #367, IOV since #591). It is a *different optimum* from FOCEI-M3,
    // mirroring NONMEM `METHOD=1 LAPLACE` with vs without `INTER`. Since M3 is run with
    // interaction in most practice, surface the non-interaction choice so a user does not
    // report FOCE-M3 estimates while expecting the FOCEI-M3 ones (#599).
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
            "M3 censoring under FOCE uses non-interaction (Sheiner–Beal) semantics; \
             FOCE-M3 and FOCEI-M3 are different optima (as in NONMEM METHOD=1 LAPLACE \
             with vs without INTER). Set method=focei for interaction semantics."
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

    // Estimation completed: delete the resume checkpoint (nothing left to
    // resume). Any post-estimation error below still returns Err, but a fresh
    // fit — not a resume — is the right recovery for those. No-op when
    // checkpointing was not active.
    crate::io::checkpoint::finish_success();

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
    if let Some(w) = eta_shrinkage_warning(&shrinkage_eta, &result.params.omega.eta_names) {
        warnings.push(w);
    }
    // Theta estimates pinned to an optimizer bound — emitted typed at source
    // with a `details` payload (#781).
    if let Some((msg, entry)) = boundary_estimate_warning(&result.params) {
        warnings.push(msg);
        native_warnings.push(entry);
    }
    // Imprecisely estimated thetas (high relative standard error) — likewise
    // emitted typed at source with `details` (#781).
    if let Some((msg, entry)) = inflated_rse_warning(&se_theta, &result.params) {
        warnings.push(msg);
        native_warnings.push(entry);
    }
    // Twin-less transit/IG absorption closed form whose fitted per-subject EBE crosses
    // into the flip-flop regime — a silently degenerate subject the η = 0 fit-start
    // reject could not catch (#785). Emitted typed at source.
    if let Some((msg, entry)) =
        absorption_flip_flop_ebe_warning(model, population, &result.params.theta, &result.eta_hats)
    {
        warnings.push(msg);
        native_warnings.push(entry);
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
                format!(
                    "auto ({})",
                    options
                        .optimizer
                        .resolve_auto(model, options.interaction)
                        .label()
                )
            } else {
                options.optimizer.label().to_string()
            }
        }
    };

    let mut fit_result = FitResult {
        method: final_method,
        method_chain: chain.clone(),
        method_wall_times_secs,
        covariance_wall_time_secs,
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
        // The optimizer's exact packed vector (FOCE/FOCEI paths), so a later
        // `run_covariance` reproduces this fit's covariance step bit-for-bit (#816
        // follow-up). `None` for estimators that don't pack in Cholesky space.
        packed_estimate: result.packed_estimate,
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
        warnings_structured: native_warnings,
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
        environment: crate::environment::detect(),
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

    // Highly correlated parameter pairs — emitted typed at source with a
    // `details` payload (#781). Appended post-construction because it needs the
    // packed parameter names, which are derived from the assembled `FitResult`;
    // `rebuild_warnings_structured` preserves this native entry by message.
    if let Some((msg, entry)) = high_correlation_warning(&fit_result) {
        fit_result.warnings.push(msg);
        fit_result.warnings_structured.push(entry);
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

/// Whether `stage_idx` is the last *estimating* stage of `chain` — the single
/// stage that owns the covariance / SIR step. Running it on every stage is
/// wasteful and not expected (#615); only the final estimate needs SEs.
///
/// An evaluation-only IMP (`imp_eval_only`, NONMEM `IMP EONLY=1`) is a likelihood
/// evaluation, not an estimator, so trailing eval-only IMP stages cede the
/// covariance step to the preceding estimator. A default (estimating) IMP is a
/// real estimator that owns the step itself, so it must not be skipped. Split out
/// of `fit()` so the per-stage gating is unit-testable.
fn is_last_estimating_stage(
    chain: &[EstimationMethod],
    stage_idx: usize,
    imp_eval_only: bool,
) -> bool {
    let is_last = stage_idx + 1 == chain.len();
    let trailing_eval_only_imp = imp_eval_only
        && chain[stage_idx + 1..]
            .iter()
            .all(|&m| m == EstimationMethod::Imp);
    is_last || trailing_eval_only_imp
}

/// Resolve the reported [`CovarianceStatus`] from the three signals that
/// determine it: whether the covariance step was requested, whether it produced
/// a covariance matrix, and whether the SIR fallback (`covariance_fallback =
/// sir`) produced a result. Pulled out of `fit()` so the precedence — a real
/// covariance always wins over a fallback, which wins over a plain failure — is
/// unit-testable without driving a full fit to a non-PD Hessian.
pub(crate) fn resolve_covariance_status(
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

pub(crate) fn cov_diagnostics(cov: Option<&DMatrix<f64>>) -> (Option<Vec<f64>>, Option<f64>) {
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
                model.error_spec.obs_keys(subject).as_ref(),
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

/// ETA-shrinkage warning threshold (fraction). 0.30 is the classic
/// Savic & Karlsson (2009) 30% rule of thumb: above it, an EBE-based
/// diagnostic — and the individual estimates of that random effect — are
/// poorly informed by the data (η shrinks toward 0).
const ETA_SHRINKAGE_WARN_THRESHOLD: f64 = 0.30;

/// Indices of the ETAs whose shrinkage meets/exceeds the warning threshold.
/// `shrinkage` is per-ETA shrinkage as a fraction (0..1); non-finite entries
/// are ignored.
pub(crate) fn high_shrinkage_eta_indices(shrinkage: &[f64]) -> Vec<usize> {
    shrinkage
        .iter()
        .enumerate()
        .filter(|(_, &s)| s.is_finite() && s >= ETA_SHRINKAGE_WARN_THRESHOLD)
        .map(|(i, _)| i)
        .collect()
}

/// Label for the `i`-th ETA, falling back to a non-empty `eta_{i}` placeholder
/// when the name is missing — so both the message and the `details` payload
/// stay unambiguous.
fn eta_label(eta_names: &[String], i: usize) -> String {
    eta_names
        .get(i)
        .cloned()
        .unwrap_or_else(|| format!("eta_{i}"))
}

/// Build the user-facing warning for high ETA shrinkage, or `None` when no ETA
/// exceeds the threshold. `shrinkage` is per-ETA shrinkage as a fraction;
/// `eta_names` labels them in the same order.
pub(crate) fn eta_shrinkage_warning(shrinkage: &[f64], eta_names: &[String]) -> Option<String> {
    let high = high_shrinkage_eta_indices(shrinkage);
    if high.is_empty() {
        return None;
    }
    let list = high
        .iter()
        .map(|&i| format!("{} ({:.0}%)", eta_label(eta_names, i), 100.0 * shrinkage[i]))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "High ETA shrinkage (\u{2265} {:.0}%): {}. EBE-based diagnostics for these \
         random effects are unreliable and the data poorly inform their individual \
         estimates; consider removing the IIV on the affected parameter(s) or \
         collecting more informative data.",
        100.0 * ETA_SHRINKAGE_WARN_THRESHOLD,
        list
    ))
}

/// Relative tolerance (a fraction of the packed bound range) within which a
/// theta estimate counts as sitting on its lower/upper optimizer bound.
const BOUNDARY_REL_TOL: f64 = 1e-3;

/// Which bound a theta estimate is pinned to and the **effective** natural-space
/// bound value the optimizer actually constrains to, or `None` when the estimate
/// is interior. Evaluated in the optimizer's *packed* space (log when the lower
/// bound is `>= 0`, else identity) so the proximity test is scale-appropriate —
/// a log-scaled parameter is "at" its bound within a constant factor, not a
/// constant absolute gap.
///
/// The floors/caps mirror `compute_bounds` / `pack_params`: a log-packed theta
/// floors its estimate and lower bound at `1e-10` and caps the upper at `1e9`,
/// so the reported bound matches the constraint that was actually active (and
/// agrees with the estimate). Degenerate/non-finite bounds yield `None`.
fn theta_boundary_side(est: f64, lower: f64, upper: f64) -> Option<(&'static str, f64)> {
    use crate::estimation::parameterization::theta_packs_log;
    let (lo_eff, hi_eff, pe, pl, pu) = if theta_packs_log(lower) {
        let lo = lower.max(1e-10);
        let hi = upper.min(1e9);
        (lo, hi, est.max(1e-10).ln(), lo.ln(), hi.ln())
    } else {
        (lower, upper, est, lower, upper)
    };
    let range = pu - pl;
    if !range.is_finite() || range <= 0.0 || !pe.is_finite() {
        return None;
    }
    let frac = (pe - pl) / range;
    if frac <= BOUNDARY_REL_TOL {
        Some(("lower", lo_eff))
    } else if frac >= 1.0 - BOUNDARY_REL_TOL {
        Some(("upper", hi_eff))
    } else {
        None
    }
}

/// Free (non-fixed) theta estimates pinned to an optimizer bound, as
/// `(name, estimate, effective_bound, side)` per hit.
fn boundary_estimates(params: &ModelParameters) -> Vec<(String, f64, f64, &'static str)> {
    let mut hits = Vec::new();
    for i in 0..params.theta.len() {
        if params.theta_fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        let est = params.theta[i];
        if let Some((side, bound)) =
            theta_boundary_side(est, params.theta_lower[i], params.theta_upper[i])
        {
            let name = params
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("THETA{}", i + 1));
            hits.push((name, est, bound, side));
        }
    }
    hits
}

/// Construct a fit-end [`WarningEntry`] with the invariant fields every native
/// emitter shares (`severity: Warning`, `source_method: None`), varying only
/// `category`, `message`, and `details`.
fn warning_entry(
    category: WarningCode,
    message: String,
    details: Option<serde_json::Value>,
) -> WarningEntry {
    WarningEntry {
        severity: WarningSeverity::Warning,
        category,
        message,
        source_method: None,
        details,
    }
}

/// Shared scaffold for the list-style fit-end warnings: given a set of `hits`,
/// return `None` when empty, otherwise join a human `list` string (`line` per
/// hit), format the message (`msg` over the joined list), build the structured
/// `details`, and wrap it in a [`WarningEntry`] as `Some((message, entry))`.
fn list_warning<H>(
    hits: Vec<H>,
    category: WarningCode,
    line: impl Fn(&H) -> String,
    msg: impl Fn(&str) -> String,
    details: impl Fn(&[H]) -> serde_json::Value,
) -> Option<(String, WarningEntry)> {
    if hits.is_empty() {
        return None;
    }
    let list = hits.iter().map(|h| line(h)).collect::<Vec<_>>().join(", ");
    let msg = msg(&list);
    let entry = warning_entry(category, msg.clone(), Some(details(&hits)));
    Some((msg, entry))
}

/// Build the human message + native structured entry (with `details`) for
/// theta estimates pinned to an optimizer bound, or `None` when none are.
fn boundary_estimate_warning(params: &ModelParameters) -> Option<(String, WarningEntry)> {
    list_warning(
        boundary_estimates(params),
        WarningCode::BoundaryEstimate,
        |(name, est, _bound, side)| format!("{name} ({est:.4} at {side} bound)"),
        |list| {
            format!(
                "Parameter estimate(s) pinned to an optimizer bound: {list}. This often \
                 indicates non-identifiability or a too-tight bound; inspect the affected \
                 parameter(s) and consider relaxing the bound or simplifying the model."
            )
        },
        |hits| {
            let params_json: Vec<serde_json::Value> = hits
                .iter()
                .map(|(name, est, bound, side)| {
                    serde_json::json!({
                        "parameter": name,
                        "estimate": est,
                        "bound": bound,
                        "side": side,
                    })
                })
                .collect();
            serde_json::json!({ "parameters": params_json })
        },
    )
}

/// Relative-standard-error threshold (percent) above which a free THETA is
/// flagged as imprecisely estimated. 50% is a common rule of thumb for fixed
/// effects (well-estimated parameters typically sit well below it).
const RSE_WARN_THRESHOLD_PCT: f64 = 50.0;

/// Free (non-fixed) THETA estimates with a relative standard error above the
/// threshold, as `(name, estimate, se, rse_pct)`. Empty when the covariance
/// step produced no SEs, or none are imprecise. `RSE = 100 * se / |estimate|`.
fn inflated_rse_thetas(
    se_theta: &Option<Vec<f64>>,
    params: &ModelParameters,
) -> Vec<(String, f64, f64, f64)> {
    let Some(se) = se_theta else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for i in 0..params.theta.len() {
        if params.theta_fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        let (est, se_i) = (params.theta[i], se.get(i).copied().unwrap_or(f64::NAN));
        if est.abs() <= 1e-12 || !se_i.is_finite() {
            continue;
        }
        let rse = 100.0 * se_i / est.abs();
        if rse.is_finite() && rse > RSE_WARN_THRESHOLD_PCT {
            let name = params
                .theta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("THETA{}", i + 1));
            hits.push((name, est, se_i, rse));
        }
    }
    hits
}

/// Build the human message + native structured entry (with `details`) for
/// free THETAs whose relative standard error exceeds the threshold, or `None`.
fn inflated_rse_warning(
    se_theta: &Option<Vec<f64>>,
    params: &ModelParameters,
) -> Option<(String, WarningEntry)> {
    list_warning(
        inflated_rse_thetas(se_theta, params),
        WarningCode::InflatedRse,
        // One decimal so a borderline value (e.g. 50.4%) is not displayed as
        // "50%" — which would read as below the threshold it just tripped.
        |(name, _est, _se, rse)| format!("{name} ({rse:.1}%)"),
        |list| {
            format!(
                "High relative standard error (RSE > {:.0}%): {list}. These parameter(s) \
                 are imprecisely estimated — often a sign of over-parameterization or \
                 data that do not inform them; consider simplifying the model.",
                RSE_WARN_THRESHOLD_PCT
            )
        },
        |hits| {
            let params_json: Vec<serde_json::Value> = hits
                .iter()
                .map(|(name, est, se, rse)| {
                    serde_json::json!({
                        "parameter": name,
                        "estimate": est,
                        "se": se,
                        "rse_pct": rse,
                    })
                })
                .collect();
            serde_json::json!({
                "threshold_pct": RSE_WARN_THRESHOLD_PCT,
                "parameters": params_json,
            })
        },
    )
}

/// Absolute correlation above which a parameter pair is flagged as
/// collinear/non-identified. `0.95` is a common rule of thumb.
const CORRELATION_WARN_THRESHOLD: f64 = 0.95;

/// THETA (fixed-effect) pairs whose estimate correlation exceeds the threshold,
/// as `(name_a, name_b, correlation)`. Restricted to the THETA block — it
/// occupies the first `theta_names.len()` packed entries with unambiguous names,
/// and fixed-effect collinearity is the most actionable/interpretable case
/// (omega/sigma live in Cholesky space, where both the correlation and the
/// packed labels are murkier). Fixed / zero-variance thetas (diagonal `<= 0`)
/// are skipped. Empty when there is no covariance matrix.
fn high_correlation_pairs(
    cov: Option<&DMatrix<f64>>,
    theta_names: &[String],
) -> Vec<(String, String, f64)> {
    let Some(cov) = cov else {
        return Vec::new();
    };
    let n_theta = theta_names.len().min(cov.nrows());
    let mut pairs = Vec::new();
    for a in 0..n_theta {
        if cov[(a, a)] <= 0.0 {
            continue;
        }
        for b in (a + 1)..n_theta {
            if cov[(b, b)] <= 0.0 {
                continue;
            }
            // sqrt(var_a) * sqrt(var_b) avoids the overflow/underflow that
            // `(var_a * var_b).sqrt()` can hit for extreme variances.
            let denom = cov[(a, a)].sqrt() * cov[(b, b)].sqrt();
            if denom <= 0.0 || !denom.is_finite() {
                continue;
            }
            let r = cov[(a, b)] / denom;
            if r.is_finite() && r.abs() >= CORRELATION_WARN_THRESHOLD {
                pairs.push((theta_names[a].clone(), theta_names[b].clone(), r));
            }
        }
    }
    pairs
}

/// Build the human message + native structured entry (with `details`) for
/// highly correlated THETA pairs in the fit's covariance matrix, or `None`.
fn high_correlation_warning(result: &FitResult) -> Option<(String, WarningEntry)> {
    list_warning(
        high_correlation_pairs(result.covariance_matrix.as_ref(), &result.theta_names),
        WarningCode::HighCorrelation,
        |(a, b, r)| format!("{a} ~ {b} ({r:.2})"),
        |list| {
            format!(
                "Highly correlated parameter pair(s) (|r| >= {CORRELATION_WARN_THRESHOLD:.2}): \
                 {list}. Highly correlated estimates indicate over-parameterization or \
                 non-identifiability; consider fixing or removing one of each pair."
            )
        },
        |pairs| {
            let pairs_json: Vec<serde_json::Value> = pairs
                .iter()
                .map(|(a, b, r)| {
                    serde_json::json!({ "parameter_a": a, "parameter_b": b, "correlation": r })
                })
                .collect();
            serde_json::json!({
                "threshold": CORRELATION_WARN_THRESHOLD,
                "pairs": pairs_json,
            })
        },
    )
}

/// Build the human message + native structured entry for a **twin-less** transit /
/// IG absorption closed form whose *fitted* per-subject EBE crosses into the
/// flip-flop regime, or `None`.
///
/// The up-front [`check_absorption_flip_flop_no_twin`] reject only samples η = 0
/// typical values. The flip-flop boundary is parameter-dependent, so a subject
/// whose empirical-Bayes CL/V (or MTT/N, MAT/CV²) drives `ke` past the tilting
/// abscissa passes that check but still hits the closed form's clamp — an
/// identically-zero profile that silently degenerates *that subject's* likelihood
/// contribution (#785). A twin-*carrying* model reroutes such subjects to the ODE
/// twin per-eval (correct), so this fires only when there is no twin (a `lagtime` /
/// `f` / user-`[odes]` form declined the desugar).
///
/// Runs at fit end, once the EBEs are known, and *warns* rather than erroring — the
/// fit already completed — pointing at the ODE `transit()`/`igd()` forcing form,
/// which reroutes per-eval at the actual η.
fn absorption_flip_flop_ebe_warning(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
) -> Option<(String, WarningEntry)> {
    // Only a twin-less transit/IG closed form can degenerate this way: a twin-carrying
    // model reroutes per-eval, a non-absorption model has no flip-flop regime.
    if !matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ) || model.absorption_ode_equivalent.is_some()
    {
        return None;
    }
    // Twin-less transit/IG closed forms reject IOV up front, so `eta_hats` carry exactly
    // the `n_eta` BSV etas (`n_kappa == 0`). Guard the length so an unexpected shape
    // skips the subject rather than panicking inside `pk_params_at_time`.
    let expected = model.n_eta + model.n_kappa;
    let crossers: Vec<String> = population
        .subjects
        .iter()
        .zip(eta_hats)
        .filter(|(subject, eta)| {
            eta.len() == expected
                && crate::pk::absorption_flip_flop_at(model, subject, theta, eta.as_slice())
        })
        .map(|(subject, _)| subject.id.clone())
        .collect();
    if crossers.is_empty() {
        return None;
    }
    let is_transit = matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit
    );
    let (abscissa_label, ode_fn, params) = if is_transit {
        ("transit rate KTR = (n+1)/mtt", "transit()", "MTT / CL")
    } else {
        ("IG abscissa 1/(2·MAT·CV²)", "igd()", "MAT / CV² / CL")
    };
    let id_list = crossers.join(", ");
    let msg = format!(
        "{} subject(s) [{}] enter the analytic absorption closed form's clamp region — the \
         flip-flop regime (disposition rate ≥ {}), or (for a 2-cpt model) coincident disposition \
         eigenvalues — at their fitted empirical-Bayes estimates, where the closed form returns \
         an identically-zero concentration profile, silently degenerating those subjects' \
         likelihood contributions. The typical-value (η = 0) parameters are in-domain, so this \
         is not caught at fit start, and this twin-less model (a user `[odes]` / `[scaling]` / \
         `[initial_conditions]` form, or a parameter whose name collides with a reserved \
         `f`/`lagtime` slot) has no ODE twin to reroute to. \
         Rewrite it as an explicit ODE `{}` model (which reroutes per-eval at the actual η), or \
         check the {} starting estimates.",
        model.pk_model.canonical_name(),
        id_list,
        abscissa_label,
        ode_fn,
        params
    );
    let entry = warning_entry(
        WarningCode::FlipFlop,
        msg.clone(),
        Some(serde_json::json!({
            "phase": "ebe",
            "model": model.pk_model.canonical_name(),
            "subjects": crossers,
        })),
    );
    Some((msg, entry))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
pub(crate) fn extract_standard_errors(
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
            // Guard a truncated `cov` (fewer rows than `template.theta`) the same
            // way the omega/sigma/kappa branches below do — report 0.0 rather than
            // panicking away an otherwise-converged fit.
            if i >= n {
                0.0
            } else if theta_packs_log(template.theta_lower[i]) {
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
        let n_lt = omega_packed_len(n_eta, false);
        let l = &template.omega.chol;

        // Extract omega sub-block of the full covariance matrix.
        let cov_omega = cov.view((omega_start, omega_start), (n_lt, n_lt));

        let mut se_vec = Vec::with_capacity(n_lt);
        // Column-major lower-triangle order — single source: `lower_tri_iter`.
        for (i, j) in lower_tri_iter(n_eta, false) {
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
        se_vec
    };

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start + omega_packed_len(n_eta, template.omega.diagonal);
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
    // The packed Cholesky layout is column-major (see `pack_params`); the flat
    // index of `L[i,i]` within the IOV block is `chol_lt_idx(i, i, n_kappa)` (the
    // single source of that offset — do not re-spell the formula here).
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
                    kappa_start + chol_lt_idx(i, i, n_kappa)
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

/// Validate the preventable preconditions of TTE event-time simulation, returning a
/// clear `Err` for a caller to act on.
///
/// **Repeated events (RTTE, `type = rtte`, Slice 3.3)** are simulated as a recurrent
/// stream by `simulate_rtte_stream` (analytic hazards only). Several preconditions are
/// *user-fixable* and would otherwise corrupt the stream silently, so they are rejected
/// here (a clean `Err` from `simulate_with_options`; the `Vec`-returning entry points
/// re-raise the same message as a panic):
/// - a **finite, positive horizon** is required — the recurrent stream is generated up
///   to that administrative window; there is no implicit one;
/// - **left truncation** (`entry_time > 0`) *is* supported (#740): the stream is drawn on
///   the time origin conditioned on survival to entry, the simulate dual of the fit-side
///   conditioning (clock-forward seeds its clock at entry; clock-reset conditions its first
///   sojourn, convention B). `entry_time == 0` is byte-identical to the non-truncated draw;
/// - exactly **one recurrent stream per subject** — multiple RTTE CMTs, or an RTTE
///   cause mixed with a competing single-event TTE cause, are a later slice (they need
///   a shared-horizon multi-stream draw);
/// - **EVID-3/4 resets** are rejected — a reset would disturb the hazard clock
///   mid-stream (selective per-state reset is a later slice).
///
/// For **ODE-accumulated (joint PK-TTE)** endpoints, a model with no such endpoint is
/// otherwise unaffected (returns `Ok`).
///
/// Drug-driven event times are sampled by integrating the augmented ODE until the
/// cumulative hazard reaches `−log u` (Slice 2.2). Two conditions cannot be sampled
/// and are *user-fixable*, so they are reported here rather than panicking deep in
/// sampling:
/// - a **finite, positive horizon** is required — a drug-driven hazard can vanish
///   and never fire, so there is no implicit observation window to censor at;
/// - **EVID-3/4 resets** and **left truncation** (`entry_time > 0`) on an ODE-TTE
///   subject are unsupported (a full reset would zero the hazard accumulator
///   mid-flight; conditional sampling past entry is a deferred follow-up).
///
/// A genuinely divergent hazard (negative / non-finite) is *not* preventable here —
/// it surfaces loudly during integration (never as a silent censor).
#[cfg(feature = "survival")]
fn validate_tte_simulatable(
    model: &CompiledModel,
    population: &Population,
    horizon: Option<f64>,
) -> Result<(), String> {
    use crate::types::{EndpointLikelihood, HazardSpec, ObsRecord, TteRecurrence};

    // RTTE (repeated-event, Slice 3.3) preconditions. The stream generator
    // (`survival::simulate_rtte_stream`) handles a single analytic recurrent cause
    // per subject; the cases below are user-fixable and would otherwise corrupt the
    // draw silently, so reject them at every simulate entry point (`simulate*`,
    // uncertainty).
    let is_rtte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Repeated { .. },
                ..
            })
        )
    };
    let is_single_tte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Single,
                ..
            })
        )
    };
    let has_rtte = model.endpoints.values().any(|e| {
        matches!(
            e,
            EndpointLikelihood::Tte {
                recurrence: TteRecurrence::Repeated { .. },
                ..
            }
        )
    });
    if has_rtte {
        // A finite, positive administrative horizon bounds the recurrent stream.
        match horizon {
            Some(h) if h.is_finite() && h > 0.0 => {}
            _ => {
                return Err(
                    "Repeated time-to-event (RTTE, `type = rtte`) simulation requires a finite, \
                     positive administrative horizon: set `[simulation] horizon` (or \
                     `SimulateOptions.horizon`). The recurrent event stream is generated up to \
                     that horizon."
                        .to_string(),
                );
            }
        }
        for subject in &population.subjects {
            // Distinct RTTE CMTs on this subject, plus whether it also carries a
            // competing single-event TTE cause (a non-recurrent `Tte` sibling).
            let mut rtte_cmts = std::collections::BTreeSet::new();
            let mut has_single_tte_sibling = false;
            for r in &subject.obs_records {
                let ObsRecord::Event { cmt, .. } = r else {
                    continue;
                };
                // Left truncation (delayed entry, entry_time > 0) IS supported for RTTE
                // simulation (#740): the stream is drawn on the time origin conditioned on
                // survival to entry — clock-forward seeds its conditioning clock at entry,
                // clock-reset conditions its first sojourn (convention B). This is the
                // simulate dual of the fit-side conditioning; see
                // [`crate::survival::simulate_rtte_stream`].
                if is_rtte(cmt) {
                    rtte_cmts.insert(*cmt);
                } else if is_single_tte(cmt) {
                    has_single_tte_sibling = true;
                }
            }
            if rtte_cmts.is_empty() {
                continue;
            }
            if rtte_cmts.len() > 1 {
                return Err(format!(
                    "RTTE simulation supports one recurrent stream per subject, but subject '{}' \
                     has RTTE endpoints on multiple CMTs ({:?}); multi-cause RTTE simulation is a \
                     later slice",
                    subject.id, rtte_cmts
                ));
            }
            if has_single_tte_sibling {
                return Err(format!(
                    "RTTE simulation does not support an RTTE cause combined with a competing \
                     single-event TTE cause on subject '{}'; simulate them separately",
                    subject.id
                ));
            }
            if !subject.reset_times.is_empty() {
                return Err(format!(
                    "RTTE simulation does not support EVID=3/4 resets (a reset would disturb the \
                     hazard clock mid-stream; selective per-state reset is a later slice); subject \
                     '{}' has resets",
                    subject.id
                ));
            }
        }
    }
    let is_ode_tte = |cmt: &usize| {
        matches!(
            model.endpoints.get(cmt),
            Some(EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            })
        )
    };
    if !model.endpoints.values().any(|e| {
        matches!(
            e,
            EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            }
        )
    }) {
        return Ok(());
    }
    match horizon {
        Some(h) if h.is_finite() && h > 0.0 => {}
        _ => {
            return Err(
                "ODE-accumulated TTE (joint PK-TTE) simulation requires a finite, positive \
                 administrative horizon: set `[simulation] horizon` (or `SimulateOptions.horizon`). \
                 A drug-driven hazard can vanish, so there is no implicit observation window."
                    .to_string(),
            );
        }
    }
    for subject in &population.subjects {
        if !subject
            .obs_records
            .iter()
            .any(|r| matches!(r, ObsRecord::Event { cmt, .. } if is_ode_tte(cmt)))
        {
            continue;
        }
        if !subject.reset_times.is_empty() {
            return Err(format!(
                "ODE-accumulated TTE simulation does not support EVID=3/4 resets (a full reset \
                 zeros the cumulative hazard mid-flight; selective per-state reset is a later \
                 slice); subject '{}' has resets",
                subject.id
            ));
        }
        if let Some(ObsRecord::Event { entry_time, .. }) = subject
            .obs_records
            .iter()
            .find(|r| matches!(r, ObsRecord::Event { cmt, entry_time, .. } if is_ode_tte(cmt) && *entry_time > 0.0))
        {
            return Err(format!(
                "ODE-accumulated TTE simulation does not support left truncation \
                 (entry_time={entry_time} for subject '{}'); conditional sampling past entry is \
                 a deferred follow-up",
                subject.id
            ));
        }
    }
    Ok(())
}

/// Simulate observations under `opts`, returning only the observation rows.
///
/// Thin wrapper over [`simulate_with_options_diag`] that discards the per-subject
/// simulation diagnostics ([`SimulationOutput::warnings`]). Use the `_diag` form when a
/// degenerate-subject warning (#762 / #763) must be surfaced (e.g. a population VPC).
pub fn simulate_with_options(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<Vec<SimulationResult>, String> {
    simulate_with_options_diag(model, population, params, n_sim, opts).map(|o| o.results)
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
///
/// This is the diagnostics-returning form: the [`SimulationOutput`] carries both the
/// rows and any non-fatal per-subject warnings (a degenerate hazard draw, #763; a
/// degenerate recurrent stream skipped, #762). [`simulate_with_options`] is the thin
/// wrapper that returns only the rows.
pub fn simulate_with_options_diag(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    opts: &SimulateOptions,
) -> Result<SimulationOutput, String> {
    use rand::SeedableRng;

    // Start the SS-equilibration non-convergence sink clean, then drain it into this run's
    // `warnings` at each return (#867): a capped nonlinear pulse-train can silently under-report
    // the SS trough, biasing simulated concentrations low.
    crate::dosing::clear_ss_nonconvergence_warnings();

    // ODE-accumulated (joint PK-TTE) TTE simulation samples drug-driven event times
    // via the augmented-ODE root-finder (Slice 2.2). Validate its preventable
    // preconditions up front so a caller gets a clean Err here rather than a panic
    // deep in sampling (a finite horizon is required; resets and left truncation are
    // not yet supported for an ODE hazard).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, opts.horizon)?;

    // Parity with `fit()`: a referenced covariate absent from the data would
    // silently read 0.0 (e.g. a `Selected` error model's `if (FREE==0)` selector
    // would route every row to branch 0, applying the wrong residual variance
    // with no diagnostic). Reject it here the same way `fit()` does (#658).
    first_error(&check_covariates(model, population))?;

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
                let crate::types::ObsRecord::Event { entry_time, .. } = record else {
                    continue;
                };
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
    // The built-in absorption input-rate guard (#588) is hoisted for the same
    // reason: otherwise a malformed multi-pathway / SS / infusion absorption model
    // integrates the whole warm-EBE pass first and fails only at the chokepoint,
    // with a confusable "EBE did not converge" instead of the real cause.
    assert_modeled_doses_supported(model, population);
    assert_absorption_closed_form_support(model, population);
    assert_absorption_flip_flop_no_twin(model, population, &params.theta);
    // A time-varying covariate on a survival hazard would be silently frozen — panic
    // rather than return a subtly wrong prediction / simulation (#741; fit() Err's).
    #[cfg(feature = "survival")]
    assert_survival_tv_covariates(model, population);
    assert_absorption_dosing_supported(model, population);

    let method = match opts.match_method {
        Some(m) => m,
        None => {
            let mut warnings = Vec::new();
            let results = simulate_inner_with_draw(
                model,
                population,
                params,
                n_sim,
                1,
                None,
                opts.horizon,
                &mut rng,
                &mut warnings,
            );
            warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
            return Ok(SimulationOutput { results, warnings });
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
        model, population, params, 100, 1e-6, None, None, 1, 0,
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
    let mut warnings = Vec::new();
    let results = simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        Some((&eta_hats, omega_inv, method)),
        opts.horizon,
        &mut rng,
        &mut warnings,
    );
    warnings.extend(crate::dosing::take_ss_nonconvergence_warnings());
    Ok(SimulationOutput { results, warnings })
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
    // These entry points return only the rows; per-subject simulation diagnostics
    // (#762 degenerate RTTE, #763 degenerate hazard draw) are collected but surfaced
    // only on the `simulate_with_options` path (`SimulationOutput.warnings`).
    let mut warnings = Vec::new();
    simulate_inner_with_draw(
        model,
        population,
        params,
        n_sim,
        1,
        None,
        None,
        rng,
        &mut warnings,
    )
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
    warnings: &mut Vec<String>,
) {
    // `horizon` and `warnings` are consumed only by the TTE path below; without the
    // `survival` feature there are no TTE endpoints, so discard them to avoid unused-arg warns.
    #[cfg(not(feature = "survival"))]
    let _ = (horizon, &mut *warnings);

    // Predict IPRED. For IOV models (`n_kappa > 0`) route through the
    // occasion-aware `predict_iov`, drawing one independent κ ~ N(0, Ω_IOV) per
    // occasion — otherwise every occasion would share the same (κ = 0) parameters
    // and the simulated data would carry NO inter-occasion variability, silently
    // under-dispersing a VPC relative to the fitted model / NONMEM `$SIM`. The
    // caller zeroes the κ tail of `eta_slice`; only the BSV head (`..n_eta`) is
    // used here, and the κ draws happen on this occasion-aware branch instead.
    // Non-IOV models keep the TV-covariate-aware fast-path dispatcher unchanged
    // and draw no extra randoms, so their output is byte-identical.
    let ipreds = if model.n_kappa > 0 {
        let omega_iov = params
            .omega_iov
            .as_ref()
            .expect("omega_iov is present whenever the model declares kappa (n_kappa > 0)");
        // One κ vector per occasion group, in `iov_occasion_groups` order — the
        // exact order `predict_iov` indexes its `kappas` argument by. Empty when
        // the subject carries no occasion labels, in which case `predict_iov`
        // falls back to κ = 0 (matching the fit-time no-occasion diagnostic).
        let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
        let kappas: Vec<Vec<f64>> = (0..occ_groups.len())
            .map(|_| {
                let z: Vec<f64> = (0..model.n_kappa).map(|_| rng.sample(normal)).collect();
                (&omega_iov.chol * DVector::from_column_slice(&z))
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        pk::predict_iov(
            model,
            subject,
            &params.theta,
            &eta_slice[..model.n_eta],
            &kappas,
        )
    } else {
        pk::compute_predictions_with_tv(model, subject, &params.theta, eta_slice)
    };

    // Add residual error (Gaussian path). IIV on residual error (#409): the
    // drawn `eta_slice` includes η_ruv, so scale the residual variance by
    // exp(2·η_ruv) — i.e. simulate `Y = IPRED + EPS·EXP(η_ruv)`.
    let ruv_scale = model.residual_var_scale(eta_slice);
    // Per-observation custom residual magnitude (#484): η-independent, so build
    // the [obs][sigma-slot] matrix once per subject and index it per row.
    let ruv_mult = model.ruv_obs_mult(subject, &params.theta);
    // `block_sigma` cross-endpoint correlations (#672): draw the full
    // multivariate residual vector from the dense `R` that estimation already
    // builds (`compute_r_matrix_with_correlations`), so the simulated data
    // reproduces the fitted covariance instead of independent per-row draws.
    // FREM covariate pseudo-observations don't participate in the correlation
    // (mirrors the `has_frem_rows` gate in
    // `stats/likelihood.rs::individual_nll_into_with_schedule`). That gate also
    // carries a `!has_censored_m3` term we deliberately drop here: it exists so
    // the likelihood's M3 BLOQ integral falls back to the scalar path, whereas
    // simulate() draws the residual first and applies censoring afterwards, so a
    // to-be-censored row should still be drawn from the correlated R. (A
    // `block_sigma` + M3 model is rejected at fit by `check_model_options`
    // regardless, so the two paths can only differ on an unfitted fixed model.)
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if !model.residual_correlations.is_empty() && !has_frem_rows && !ipreds.is_empty() {
        emit_correlated_residual_rows(
            model,
            subject,
            params,
            &ipreds,
            ruv_scale,
            ruv_mult.as_deref(),
            draw,
            sim,
            normal,
            rng,
            results,
        );
    } else {
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
        warnings,
    );
}

/// Draw the correlated residual vector for one subject's Gaussian observation
/// rows from the dense `R` built by [`compute_r_matrix_with_correlations`] (the
/// same matrix FOCE/FOCEI/SAEM/`imp` evaluate the likelihood against), instead
/// of the per-row independent draw `emit_subject_rows` otherwise uses. Callers
/// must already have excluded FREM rows and the empty-correlation case.
///
/// R is factored with a PSD-safe symmetric-eigen square root, so a singular
/// (e.g. `rho = ±1`) or mildly indefinite fixed `block_sigma` yields a valid
/// draw instead of a Cholesky panic. Subjects whose R is diagonal (no paired
/// rows) take a cheap per-row draw and skip the factorization entirely.
#[allow(clippy::too_many_arguments)]
fn emit_correlated_residual_rows<R: rand::Rng>(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    ipreds: &[f64],
    ruv_scale: f64,
    ruv_mult: Option<&[Vec<f64>]>,
    draw: usize,
    sim: usize,
    normal: rand_distr::Normal<f64>,
    rng: &mut R,
    results: &mut Vec<SimulationResult>,
) {
    let err_keys = model.error_spec.obs_keys(subject);
    let mut r = match ruv_mult {
        Some(mult) => compute_r_matrix_with_correlations_scaled(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &params.sigma.values,
            &model.residual_correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            &model.error_spec,
            ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            &subject.obs_l2,
            &params.sigma.values,
            &model.residual_correlations,
        ),
    };
    if ruv_scale != 1.0 {
        r *= ruv_scale;
    }
    let n = ipreds.len();
    // z ~ N(0, Iₙ); the correlated residual is a matrix square-root factor of R
    // times z (`Cov(F·z) = F·Fᵀ = R`).
    let z = DVector::from_iterator(n, (0..n).map(|_| rng.sample(normal)));
    // Fast path: if R has no nonzero off-diagonal the subject has no actually
    // paired rows (every observation sits in its own residual block), so R is
    // diagonal and the draw is the same independent per-row draw the scalar
    // path uses. Skip the O(n³) factorization — this keeps a densely-sampled
    // endpoint of a `block_sigma` model cheap, and reproduces the scalar path's
    // RNG-for-RNG output for such subjects.
    let has_offdiag = (0..n).any(|j| ((j + 1)..n).any(|k| r[(j, k)] != 0.0));
    let eps = if !has_offdiag {
        DVector::from_iterator(n, (0..n).map(|j| r[(j, j)].max(0.0).sqrt() * z[j]))
    } else {
        // A fitted or fixed `block_sigma` can be positive-SEMIdefinite rather
        // than strictly positive-definite — a perfect cross-endpoint
        // correlation (`rho = ±1`, which the parser accepts on the inclusive
        // [-1, 1] range) makes R singular, so a Cholesky factor doesn't exist
        // and would panic the whole simulation. Use the symmetric-eigen square
        // root `V·diag(√max(λ,0))`, which is well defined for any PSD R and
        // clamps tiny negative eigenvalues (round-off, or a mildly indefinite
        // fixed R) to zero instead of aborting.
        let eig = r.symmetric_eigen();
        let mut factor = eig.eigenvectors;
        for (k, &lambda) in eig.eigenvalues.iter().enumerate() {
            let s = lambda.max(0.0).sqrt();
            factor.column_mut(k).scale_mut(s);
        }
        factor * z
    };
    for (j, &ipred) in ipreds.iter().enumerate() {
        let value = ipred + eps[j];
        results.push(SimulationResult {
            draw,
            sim,
            id: subject.id.clone(),
            time: obs_row_time(subject, j),
            cmt: subject.obs_cmts[j],
            ipred,
            outcome: SimOutcome::Continuous { value },
        });
    }
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
    warnings: &mut Vec<String>,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    // Single chokepoint for every `simulate*` variant (both `simulate_inner` and
    // the propensity path funnel through here). Guard the modeled-`RATE` dose
    // precondition once per call, as `predict()` does — `simulate()` runs no
    // data-check otherwise. #324.
    assert_modeled_doses_supported(model, population);
    assert_absorption_closed_form_support(model, population);
    assert_absorption_flip_flop_no_twin(model, population, &params.theta);
    // A time-varying covariate on a survival hazard would be silently frozen — panic
    // rather than return a subtly wrong prediction / simulation (#741; fit() Err's).
    #[cfg(feature = "survival")]
    assert_survival_tv_covariates(model, population);
    assert_absorption_dosing_supported(model, population);
    // CTMM (#759) has no simulation path yet: the Gaussian/PK emitter below would write
    // meaningless all-zero DV rows for a discrete-state endpoint (the generator is never
    // sampled). Fail loud rather than return garbage — fit() supports CTMM, simulate()
    // does not. Mirrors the survival guards above; the CLI `--simulate` path returns this
    // as a clean Err in `run_model_simulate`. This is the single simulate chokepoint.
    #[cfg(feature = "markov")]
    assert!(
        !model.has_ctmm(),
        "simulate()/predict() does not support a [markov_model] (CTMM) endpoint yet — its \
         discrete-state trajectory has no simulation path, so the Gaussian emitter would \
         produce meaningless all-zero observations. CTMM simulation is a later slice."
    );

    // ODE-accumulated TTE simulation has preventable preconditions (finite horizon,
    // no resets / left truncation). `simulate_with_options` checks them first and
    // returns a clean Err; the Vec-returning `simulate` / `simulate_with_seed` (and
    // the uncertainty path) funnel through here and cannot signal an error, so
    // enforce the identical contract as a panic rather than emitting wrong rows.
    #[cfg(feature = "survival")]
    if let Err(e) = validate_tte_simulatable(model, population, horizon) {
        panic!("{e}");
    }

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
                        &mut *warnings,
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
                        &mut *warnings,
                    );
                }
            }
        }
    }

    results
}

// ======================================================================
// Adaptive (state-reactive / feedback) dosing — epic #391, beta.
//
// Two public entry points wrap the `pub(crate)` reactive driver
// (`ode_predictions_adaptive_impl`) with shared per-(subject, replicate)
// orchestration (`run_adaptive_population`), the tagged-row output schema
// (Part D), and the frozen-schedule replay verifier (Part E backbone,
// default-on): `simulate_adaptive` takes a hand-written controller closure, and
// `simulate_adaptive_from_spec` compiles a declarative `[adaptive_dosing]` block
// into the same engine. Both support Ipred and `Dv` (assay-noised) monitors on
// the S1.5 controller-assay substreams.
// ======================================================================

use crate::sim::adaptive::{
    AdaptiveRun, AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction,
    DoseLedgerEntry, MonitorSpec,
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
    /// Signals the controller may read at each decision. Each monitor resolves on
    /// its own [`crate::sim::adaptive::ObserveMode`]: `Ipred` reads the latent
    /// prediction; `Dv` adds the endpoint's assay-noise draw (`IPRED + ε·√σ²`) on
    /// the monitor's per-subject RNG substream (#391 S1.5).
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
/// `#[non_exhaustive]`: the remaining Part-D artifacts (population summary, run
/// manifest) land as added fields without breaking callers who receive this struct.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdaptiveSimulationResult {
    /// Per-observation predictions: one row per (replicate, subject, obs time).
    /// `ipred` and the `Continuous` value are the individual prediction. Assay-
    /// noised DV monitoring (used for the controller's decisions) is applied inside
    /// the run via each `Dv` monitor's RNG substream, not emitted as a separate row.
    pub trajectories: Vec<SimulationResult>,
    /// Every dose the controllers actually issued, tagged by `(draw, sim)`.
    pub ledger: Vec<DoseLedgerEntry>,
    /// One row per decision (including holds), in schedule order, up to and
    /// including any `Stop`.
    pub decisions: Vec<DecisionLogEntry>,
    /// Per-subject outcome metrics (#391 S2.4): one row per realized `(subject,
    /// draw, sim)` run — cumulative dose, dose-change / hold / discontinuation
    /// counts, the observed-signal summary, and `pct_time_in_window` when the block
    /// declares a `target_window`. Derived from `ledger` + `decisions` alone (no
    /// re-integration). The population summary *with bands* rides with the
    /// uncertainty slice (S5), where bands carry meaning.
    pub metrics: Vec<AdaptiveSubjectMetrics>,
}

/// Simulate state-reactive ("adaptive" / feedback) dosing over a population
/// (epic #391, **beta**).
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
    let ode = model.ode_spec.as_ref().ok_or_else(|| {
        "simulate_adaptive requires an ODE model (the reactive driver runs on the ODE \
         engine); this model has no [odes] block"
            .to_string()
    })?;

    // The adaptive assay keys residual variance by the monitored compartment
    // number (`residual_variance_at(cmt, …)`), but a `Selected` error model's
    // endpoints are keyed by the covariate selector's 0-based branch index, not
    // CMT — `map.get(&cmt)` would miss and `variance_at` returns NaN, corrupting
    // the assay draw. Reject the combination rather than emit NaN observations (#658).
    reject_selected_error_for_adaptive(model)?;

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

    // Programmatic path: cmt-based monitors (no compiled `observe` expression) and
    // a `Vec<DoseAction>` controller with no rule provenance. Adapt both into the
    // engine's paired-monitor / `ControllerDecision` contract; the all-`None`
    // `observe`/`rule` keeps the reactive driver byte-for-byte identical to the
    // pre-S2 behaviour.
    let monitors: Vec<crate::sim::adaptive::AdaptiveMonitor> = opts
        .monitors
        .iter()
        .map(|spec| crate::sim::adaptive::AdaptiveMonitor {
            spec,
            observe: None,
        })
        .collect();
    let make = move || {
        let mut c = make_controller();
        move |ctx: &ControllerCtx| crate::sim::adaptive::ControllerDecision {
            actions: c(ctx),
            rule: None,
        }
    };
    run_adaptive_population(
        model,
        ode,
        population,
        params,
        n_sim,
        &opts.decision_times,
        &monitors,
        make,
        // Programmatic path: no declarative block, so no target band — the
        // window-dependent metric (`pct_time_in_window`) is left unreported.
        None,
        // ...and no `auc_target`, so the signal-AUC pass is skipped and
        // `auc_target_attainment` is left unreported.
        None,
        opts,
    )
}

/// Shared per-(subject, replicate) orchestration behind [`simulate_adaptive`] and
/// [`simulate_adaptive_from_spec`]: draw η ~ N(0, Ω), mint a **fresh** controller,
/// run the reactive ODE driver, (optionally) verify the frozen-schedule replay,
/// and stamp the replicate tags onto the trajectory / ledger / decision rows.
///
/// The decision schedule and monitors are passed explicitly: the programmatic
/// entry takes them from `opts`, the declarative entry derives them from the
/// `[adaptive_dosing]` spec. Each [`AdaptiveMonitor`](crate::sim::adaptive) carries
/// its own optional compiled `observe` expression (the declarative signal) or
/// `None` (the programmatic cmt readout, byte-for-byte unchanged), and the
/// controller returns a `ControllerDecision` so the ledger can record which rule
/// fired.
///
/// Both entries resolve `ode` and validate the schedule before calling, so this
/// helper assumes them well-formed and focuses on the run loop.
#[allow(clippy::too_many_arguments)]
fn run_adaptive_population<F, C>(
    model: &CompiledModel,
    ode: &crate::ode::OdeSpec,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    decision_times: &[f64],
    monitors: &[crate::sim::adaptive::AdaptiveMonitor],
    make_controller: F,
    target_window: Option<(f64, f64)>,
    auc_target: Option<(f64, f64)>,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String>
where
    F: Fn() -> C,
    C: FnMut(&ControllerCtx) -> crate::sim::adaptive::ControllerDecision,
{
    // Reject model/data the reactive driver cannot faithfully simulate (IOV,
    // time-varying covariates, resets, SDE) with a typed error — never a silent
    // wrong answer (#391). Both public entry points funnel through here.
    reject_unsupported_adaptive(model, population)?;

    // #721: the reactive path skipped the shared dose-precondition guards that
    // `simulate()` / `predict()` / `fit()` all run before integrating — modeled `RATE`
    // (#324), analytic-absorption closed-form model/data compatibility, and built-in
    // absorption input-rate pathway fractions / SS / infusion / domain (#588). Without
    // them a feedback-dosed model with, e.g., malformed absorption fractions, an
    // out-of-domain absorption parameter, or (once base regimens are supported, #702) a
    // modeled `RATE` would silently mis-deliver the dose — the exact class #588 closed
    // for the static paths. Surface them as a typed error (the `check_*` form `fit()`
    // uses, and the form `reject_unsupported_adaptive` just above already uses) rather
    // than the panic `simulate()` / `predict()` raise via `assert_*`: this
    // `Result`-returning chokepoint — and its ferx-r FFI surface — then fails with one
    // uniform, recoverable error contract instead of aborting the process.
    // (`check_absorption_closed_form_support` is a no-op here — it returns `None` unless
    // the model is an analytic absorption closed form, which this ODE-only path rejects
    // up front — but is wired for parity and #702 base-regimen support.)
    first_error(&check_modeled_dose_rates(model, population))?;
    if let Some(msg) = check_absorption_closed_form_support(model, population) {
        return Err(msg);
    }
    first_error(&check_absorption_dosing(model, population))?;

    // The occasion bookkeeping assumes a strictly-increasing decision schedule:
    // `occasion_of` (per-decision-window κ selection) scans ascending and stops at the
    // first later time, and `decision_index_of` keys windows by exact time bits. An
    // unsorted schedule would mis-map a record to the wrong occasion's κ, and a
    // duplicated time would split the window-open index from the materialiser's
    // occasion — both silently, since the frozen-replay verifier reuses the same
    // occasion arrays and cannot catch a shared-wrong-input (#701 review). The
    // declarative `[adaptive_dosing]` path already enforces this on its `at` list via
    // `validate_increasing_finite`; guard the programmatic `simulate_adaptive` path
    // here too, at the shared funnel, with the *same* validator (finite + strictly
    // ascending ⇒ no duplicates) so the two paths cannot drift.
    crate::sim::adaptive::validate_increasing_finite(decision_times, "`decision_times`")?;

    use rand::SeedableRng;

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
    let mut metrics: Vec<AdaptiveSubjectMetrics> = Vec::new();

    // `auc_target_attainment` (#391 S2.5b) integrates a dense grid with a single PK
    // snapshot — exact only for constant-covariate subjects. A time-varying (or
    // TIME-in-PK) subject would silently get a frozen-PK AUC (there is no exact
    // per-event dense-grid solver — even the fit path approximates TV states with a
    // warning, and the adaptive result has no warnings channel). So reject the
    // combination loudly (#700) rather than report a wrong metric. Every other
    // adaptive output — predictions, decisions, the dose ledger, `target_window` — is
    // fully per-event covariate-aware; only this one exposure metric is deferred.
    if auc_target.is_some()
        && (model.n_kappa > 0
            || population
                .subjects
                .iter()
                .any(|s| crate::pk::subject_needs_per_event_pk(model, s)))
    {
        return Err(
            "adaptive-dosing `auc_target` is not yet supported for time-varying-covariate, \
             TIME-in-PK, or IOV (`kappa`) subjects: its exposure metric integrates a dense grid \
             from a single frozen PK snapshot, which would be silently wrong when the PK changes \
             across the horizon (a drifting covariate or a per-occasion κ). Drop `auc_target` (all \
             other outputs remain per-event / per-occasion aware), or track #700/#701 for a \
             per-event AUC."
                .to_string(),
        );
    }

    // Reused per-event PK buffer (#700): filled in place per (sim, subject) on the
    // time-varying path via `compute_event_pk_params_into`, so the hot loop keeps
    // its backing `Vec`s instead of allocating three fresh ones per replicate. The
    // values are recomputed every iteration (η is redrawn); only the allocation is
    // reused. Unused on the constant path (`event_pk` stays `None`).
    let mut event_pk_buf = crate::pk::EventPkParams::default();

    for sim_idx in 0..n_sim {
        let sim = sim_idx + 1;
        for subject in &population.subjects {
            // Draw η ~ N(0, Ω). `eta_slice` (η with κ appended as zeros) is the
            // **baseline** window — the parameters before the first decision, and
            // the reactive driver's fixed eta on a non-IOV run.
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let eta = &params.omega.chol * DVector::from_column_slice(&z);
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            // Inter-occasion variability (#701): draw a fresh κ per decision window
            // and build the per-window eta `[η_bsv | κ_g]` the driver threads through
            // each segment, plus `decision_pk[g]` = the PK at decision g under
            // occasion g's κ. Occasion = decision index (per-decision-window). κ is
            // drawn on a dedicated substream keyed by (subject, replicate), disjoint
            // from the η and assay streams, so a non-IOV model (`n_kappa == 0`) draws
            // nothing and this whole block is skipped — the path stays byte-identical.
            let (eta_occ, decision_pk): (
                Option<Vec<Vec<f64>>>,
                Option<Vec<crate::types::PkParams>>,
            ) = if model.n_kappa > 0 {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when n_kappa > 0");
                let kappa_base =
                    crate::sim::adaptive::subject_kappa_base_seed(assay_root, &subject.id, sim);
                let n_occ = decision_times.len();
                let mut eta_occ = Vec::with_capacity(n_occ);
                let mut decision_pk = Vec::with_capacity(n_occ);
                for g in 0..n_occ {
                    let zk: Vec<f64> = (0..model.n_kappa)
                        .map(|k| crate::sim::adaptive::kappa_standard_normal(kappa_base, g, k))
                        .collect();
                    let kappa_g = &omega_iov.chol * DVector::from_column_slice(&zk);
                    let mut e: Vec<f64> = eta_slice[..n_eta].to_vec();
                    e.extend(kappa_g.iter().copied());
                    // PK at decision g under occasion g's κ, at the covariate the
                    // driver actually sees at that decision. This MUST match the
                    // driver's live `decision_cov` (predictions.rs) exactly: the
                    // obs-coincident snapshot on a record, else the LOCF covariate of
                    // the most-recent record carried forward — NOT the frozen t=0
                    // baseline. On a time-varying-covariate + IOV model whose decision
                    // lands between records, a baseline fallback here would freeze the
                    // decision's covariate at t=0 (the exact #700 defect) while the
                    // driver's readout uses LOCF — and the frozen-replay verifier,
                    // which reuses this same `decision_pk`, could not catch it. Sharing
                    // the one `locf_decision_cov` helper the driver uses makes the two
                    // covariate resolutions a single source of truth (#701 review).
                    let dcov = crate::ode::predictions::locf_decision_cov(
                        decision_times[g],
                        subject,
                        &subject.covariates,
                    );
                    decision_pk.push((model.pk_param_fn)(
                        &params.theta,
                        &e,
                        dcov,
                        decision_times[g],
                    ));
                    eta_occ.push(e);
                }
                (Some(eta_occ), Some(decision_pk))
            } else {
                (None, None)
            };

            // Per-event PK when PK varies across the horizon: an IOV κ that switches
            // per occasion (#701), or a time-varying covariate / pk-only row / `TIME`
            // built-in (#700). The reactive driver then resolves PK per segment from
            // these snapshots. `None` ⇒ constant PK, driven from the frozen `pk`
            // snapshot below. The IOV snapshot is owned (recomputed per replicate as κ
            // is redrawn); the #700-only path reuses `event_pk_buf`. One shared
            // predicate (`subject_needs_per_event_pk`) gates the #700 branch, the
            // materialiser, and the `auc_target` guard so they cannot drift apart.
            let event_pk_iov: Option<crate::pk::EventPkParams> = eta_occ.as_ref().map(|eo| {
                // IOV: each obs / pk-only record carries its occasion's κ (and any
                // #700 covariate); records before the first decision use the baseline.
                crate::pk::compute_event_pk_params_iov(
                    model,
                    subject,
                    &params.theta,
                    &eta_slice,
                    eo,
                    decision_times,
                )
            });
            let event_pk: Option<&crate::pk::EventPkParams> =
                if let Some(ev) = event_pk_iov.as_ref() {
                    Some(ev)
                } else if crate::pk::subject_needs_per_event_pk(model, subject) {
                    crate::pk::compute_event_pk_params_into(
                        model,
                        subject,
                        &params.theta,
                        &eta_slice,
                        &mut event_pk_buf,
                    );
                    Some(&event_pk_buf)
                } else {
                    None
                };

            // Baseline PK snapshot at the start of the simulation horizon (t=0).
            let pk = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates, 0.0);

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
            let run: AdaptiveRun = crate::ode::ode_predictions_adaptive_impl(
                ode,
                &pk.values,
                event_pk,
                decision_pk.as_deref(),
                eta_occ.as_deref(),
                &params.theta,
                &eta_slice,
                subject,
                decision_times,
                monitors,
                &mut controller,
                opts.max_decisions,
                Some(&assay),
            )
            .map_err(|e| format!("subject '{}' (sim {sim}): {e}", subject.id))?;

            if opts.verify {
                // #748: the frozen-replay verifier below reuses the *same* precomputed
                // snapshots (`event_pk`, `decision_pk`, `eta_occ`) the driver did, so it
                // validates that the two *consume* them identically — never that they are
                // *correct*. A build-loop that fed a leaf the wrong covariate / occasion /
                // κ (the #732 & #739 "decision covariate frozen at t=0" class) is applied
                // by both sides and passes the replay bit-exact. Close that blind spot by
                // independently re-deriving each snapshot from primitives and bit-asserting
                // it against what the driver was handed — default-on, before the replay.
                verify_adaptive_snapshots(
                    model,
                    params,
                    subject,
                    &eta_slice,
                    decision_times,
                    assay_root,
                    sim,
                    eta_occ.as_deref(),
                    decision_pk.as_deref(),
                    event_pk,
                )
                .map_err(|e| {
                    format!(
                        "adaptive snapshot verification failed for subject '{}' (sim {sim}): {e}",
                        subject.id
                    )
                })?;

                crate::ode::verify_adaptive_frozen_replay(
                    ode,
                    &pk.values,
                    event_pk,
                    decision_pk.as_deref(),
                    eta_occ.as_deref(),
                    &params.theta,
                    &eta_slice,
                    subject,
                    decision_times,
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

            // Signal-AUC pass for the exposure metric (#391 S2.5b): run only when an
            // `auc_target` is declared (its sole consumer) and a monitor exists. It
            // re-integrates the realized ledger on its own dense grid — separate from,
            // and never perturbing, the reactive run + verifier above.
            //
            // Windowed over the **realized** decision times (`run.decisions`), NOT the
            // full scheduled `decision_times`: after a `Stop` the controller
            // discontinues and the later scheduled decisions never happen, so scoring
            // their dose-free, washed-out windows would fold discontinuation — already
            // a first-class outcome (`discontinued` / `time_to_discontinuation`) — into
            // the exposure metric as silent misses (double-counting one event into two
            // metrics). Confining to realized decisions keeps `auc_target_attainment`
            // a clean "of the windows we dosed, how many hit target", on the same
            // realized basis as `pct_time_in_window`. For a run that never
            // discontinues the two decision lists are identical.
            //
            // Dropping the post-`Stop` windows loses no *dosed* window: a declarative
            // `Stop` is dose-free (`adaptive_control::Controller::apply` maps it to
            // `[DoseAction::Stop]`, never `[dose, Stop]`), so the last realized window
            // already covers the last dose. The one controller that *can* dose-then-
            // stop is the programmatic `Vec<DoseAction>` API, and that path runs with
            // `auc_target = None` (the AUC pass is skipped) — so a dose issued *at* a
            // stop never coincides with this metric. If that ever changes (a
            // dose-on-stop reaching the AUC pass), the final dose's window would need
            // explicit handling — the `debug_assert!` in the match arm below is the
            // tripwire for exactly that, and `sim::adaptive::run_has_dose_on_stop` (with
            // its unit test plus the `..._after_discontinuation` test) pins the
            // dose-free-`Stop` invariant this relies on.
            let window_aucs: Vec<f64> = match (auc_target, monitors.first()) {
                (Some(_), Some(mon)) => {
                    // Tripwire (see the note above): realized-window scoring is exact only
                    // while no `Stop` carries a dose. Unreachable today; this fires in
                    // debug/test builds if a future change ever routes a dose-on-stop here,
                    // rather than silently under-reporting that final dose's exposure.
                    debug_assert!(
                        !crate::sim::adaptive::run_has_dose_on_stop(&run.decisions),
                        "auc_target_attainment: a Stop carried a final dose (`[dose, Stop]`); \
                         its post-stop exposure window is unscored under realized-window \
                         scoring — the dose-free-Stop invariant no longer holds, so that \
                         window now needs explicit handling"
                    );
                    let realized_decision_times: Vec<f64> =
                        run.decisions.iter().map(|d| d.time).collect();
                    crate::ode::adaptive_window_signal_aucs(
                        ode,
                        &pk.values,
                        &params.theta,
                        &eta_slice,
                        subject,
                        &realized_decision_times,
                        &run.ledger,
                        mon.observe,
                        mon.spec.cmt,
                    )
                }
                _ => Vec::new(),
            };

            // Per-subject outcome metrics for this run, computed from its realized
            // ledger + decision log (S2.4) and the window AUC series (S2.5b). Taken
            // before the rows are moved into the population vectors below; the
            // `(subject, draw, sim)` key is stamped here rather than read from the
            // rows (whose tags the single-subject driver still leaves at 0).
            metrics.push(crate::sim::adaptive::compute_subject_metrics(
                &subject.id,
                1,
                sim,
                &run.ledger,
                &run.decisions,
                target_window,
                auc_target,
                &window_aucs,
            ));

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
        metrics,
    })
}

/// Validate that the per-(subject, replicate) PK snapshots the reactive driver was
/// handed — and which the frozen-schedule replay verifier reuses **verbatim** — are
/// *correct*, not merely *consumed identically* by both engines (#748).
///
/// `run_adaptive_population` precomputes, once, the PK snapshots it feeds to both the
/// driver and `verify_adaptive_frozen_replay`: the baseline `pk` (t=0), `eta_occ[g]`
/// (the per-decision-window `[η_bsv | κ_g]`), `decision_pk[g]` (the PK at decision
/// `g`), and `event_pk` (the per-record obs / pk-only PK). Because the replay consumes
/// the identical arrays, a **build-loop** error — a wrong covariate, occasion, or κ
/// fed to a leaf — is applied by both sides and passes the replay bit-exact. That is
/// exactly how the "decision covariate frozen at t=0" defect reached production twice
/// (#732, #739): each time only a hand-written regression test using an off-record
/// `f_applied` readout could catch it, because the default-on verifier could not.
///
/// This check closes that blind spot for the three per-occasion / per-event snapshots
/// #748 names. It **independently re-derives** `eta_occ`, `decision_pk`, and `event_pk`
/// from the run's *primitive* inputs — θ, the drawn BSV η (`eta_slice`), the subject's
/// covariates, the decision schedule, and (re-drawn here) the per-occasion κ substream
/// — by calling the **leaf** rules directly (`pk_param_fn`, `locf_decision_cov`,
/// `occasion_of`, `subject_kappa_base_seed` / `kappa_standard_normal`), NOT by
/// re-invoking the build loop's composite assembler. In particular `event_pk` is
/// rebuilt record-by-record inline (occasion via `occasion_of`, PK via `pk_param_fn`
/// at the record's own covariate) rather than through `compute_event_pk_params_iov`,
/// so a wrong occasion or covariate the *composite* would introduce is caught too —
/// not only a wrong array argument. It then bit-asserts (`to_bits`) each re-derivation
/// against what the driver received; `pk_param_fn` and the seeded κ substream are pure,
/// so a correct build agrees to the bit and a divergent one fails loudly, default-on,
/// for **every** model.
///
/// The teeth come from this being a **second orchestration over the same leaves**: it
/// **must not** be refactored to call the build loop's composite assembler, or a defect
/// edited into that assembler would be mirrored on both sides and slip through again.
/// The leaf rules themselves (the covariate LOCF rule, the occasion map, the κ draw)
/// are the single source of truth and are unit-tested in isolation; θ and `eta_slice`
/// are trusted primitive draws (not "built" snapshots) so — like the seed and Ω — they
/// are consumed, not re-derived.
///
/// Scope: the constant-path baseline `pk` (a single t=0 evaluation at the subject-static
/// covariate) is a shared snapshot too, but a trivial one — it is left to the degenerate
/// oracle rather than re-derived per run here; extending the check to it is a possible
/// follow-up. On the constant-covariate / non-IOV path (no `eta_occ`, no `event_pk`)
/// this function is therefore a no-op.
#[allow(clippy::too_many_arguments)]
fn verify_adaptive_snapshots(
    model: &CompiledModel,
    params: &ModelParameters,
    subject: &Subject,
    eta_slice: &[f64],
    decision_times: &[f64],
    assay_root: u64,
    sim: usize,
    eta_occ: Option<&[Vec<f64>]>,
    decision_pk: Option<&[PkParams]>,
    event_pk: Option<&crate::pk::EventPkParams>,
) -> Result<(), String> {
    let n_eta = model.n_eta;

    // ── IOV path: re-derive eta_occ + decision_pk, then event_pk inline ─────────
    if let (Some(eta_occ), Some(decision_pk)) = (eta_occ, decision_pk) {
        let omega_iov = params.omega_iov.as_ref().ok_or_else(|| {
            "an IOV run (eta_occ / decision_pk present) carries no omega_iov — a build-loop \
             invariant violation (#748)"
                .to_string()
        })?;
        let kappa_base =
            crate::sim::adaptive::subject_kappa_base_seed(assay_root, &subject.id, sim);
        let n_occ = decision_times.len();

        if eta_occ.len() != n_occ || decision_pk.len() != n_occ {
            return Err(format!(
                "occasion snapshot arrays are mis-sized: expected {n_occ} (one per decision), \
                 got eta_occ.len()={}, decision_pk.len()={} (#748)",
                eta_occ.len(),
                decision_pk.len()
            ));
        }

        let mut eta_occ_check: Vec<Vec<f64>> = Vec::with_capacity(n_occ);
        for g in 0..n_occ {
            // Re-draw occasion g's κ on the dedicated substream and assemble the
            // per-window eta [η_bsv | κ_g] — an independent copy of the build loop.
            let zk: Vec<f64> = (0..model.n_kappa)
                .map(|k| crate::sim::adaptive::kappa_standard_normal(kappa_base, g, k))
                .collect();
            let kappa_g = &omega_iov.chol * DVector::from_column_slice(&zk);
            let mut e: Vec<f64> = eta_slice[..n_eta].to_vec();
            e.extend(kappa_g.iter().copied());

            if !slice_bits_eq(&e, &eta_occ[g]) {
                return Err(format!(
                    "eta_occ[{g}] (occasion κ for the decision at t={}) diverges from an \
                     independent re-draw — a wrong occasion→κ keying or [η | κ] assembly in the \
                     build loop. The frozen-replay verifier reuses this array and cannot catch \
                     it (#748).",
                    decision_times[g]
                ));
            }

            // decision_pk[g] must equal pk_param_fn at the covariate the driver's live
            // `decision_cov` uses — LOCF of the most-recent record, NOT the frozen t=0
            // baseline (the twice-fixed #732 / #739 defect).
            let dcov = crate::ode::predictions::locf_decision_cov(
                decision_times[g],
                subject,
                &subject.covariates,
            );
            let dpk = (model.pk_param_fn)(&params.theta, &e, dcov, decision_times[g]);
            if !pk_bits_eq(&dpk, &decision_pk[g]) {
                return Err(format!(
                    "decision_pk[{g}] (decision at t={}) diverges from an independent \
                     re-derivation at the LOCF decision covariate — the 'decision covariate \
                     frozen at t=0' class fixed in #732 / #739. Both the driver and the \
                     frozen-replay verifier consume this snapshot, so the replay cannot catch \
                     it (#748).",
                    decision_times[g]
                ));
            }
            eta_occ_check.push(e);
        }

        // event_pk (per-record obs / pk-only PK) re-derived from the *check's* occasion
        // eta, so a corrupt build eta_occ cannot launder itself into event_pk.
        let ev = event_pk.ok_or_else(|| {
            "an IOV run carries no event_pk — κ makes PK per-occasion, so obs / pk-only records \
             must each carry a per-event snapshot (#748)"
                .to_string()
        })?;
        return check_event_pk_records(
            model,
            &params.theta,
            eta_slice,
            subject,
            decision_times,
            Some(&eta_occ_check),
            ev,
        );
    }

    // ── TV-covariate / TIME-in-PK path without IOV ──────────────────────────────
    // Every record uses the baseline η; re-derive per record and compare.
    if let Some(ev) = event_pk {
        return check_event_pk_records(
            model,
            &params.theta,
            eta_slice,
            subject,
            decision_times,
            None,
            ev,
        );
    }

    // ── Constant-covariate path: only the baseline `pk` (checked above) applies. ─
    Ok(())
}

/// Two `f64`s are the same to the bit (so two runs of the *same* deterministic
/// computation agree, and a NaN equals itself). Used by the #748 snapshot check,
/// where any difference is a real build divergence, not solver slack.
#[inline]
fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Bit-equality of two PK parameter snapshots (#748).
fn pk_bits_eq(a: &PkParams, b: &PkParams) -> bool {
    a.values
        .iter()
        .zip(b.values.iter())
        .all(|(x, y)| f64_bits_eq(*x, *y))
}

/// Bit-equality of two eta slices (#748).
fn slice_bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f64_bits_eq(*x, *y))
}

/// Re-derive each per-record (obs / EVID=2 pk-only) PK snapshot **inline** — occasion
/// via [`occasion_of`](crate::pk::occasion_of), PK via `pk_param_fn` at the record's
/// own covariate — and bit-check it against the `event_pk` the driver used (#748).
///
/// This is an *independent second derivation*, NOT a re-call of
/// `compute_event_pk_params_iov`: a wrong occasion grouping or per-record covariate the
/// composite builder would introduce is caught here, not only a wrong array argument.
/// `eta_occ_check` is the independently re-derived per-window eta (IOV); `None` ⇒ the
/// non-IOV path, where every record uses the baseline `eta_slice`. The dose-free
/// adaptive base subject carries no dose events, so `event_pk.dose` must stay empty.
fn check_event_pk_records(
    model: &CompiledModel,
    theta: &[f64],
    eta_slice: &[f64],
    subject: &Subject,
    decision_times: &[f64],
    eta_occ_check: Option<&[Vec<f64>]>,
    got: &crate::pk::EventPkParams,
) -> Result<(), String> {
    // Eta in effect at a record time `t`: occasion `g`'s per-window eta when IOV is
    // active and a window is open, else the baseline (matches the driver's `eta_for`
    // over the same `occasion_of`).
    let eta_at = |t: f64| -> &[f64] {
        match eta_occ_check {
            Some(eo) => match crate::pk::occasion_of(decision_times, t) {
                Some(g) => eo[g].as_slice(),
                None => eta_slice,
            },
            None => eta_slice,
        }
    };

    if got.dose.len() != subject.doses.len() {
        return Err(format!(
            "event_pk.dose has {} entr(ies) but the subject has {} dose record(s) (#748)",
            got.dose.len(),
            subject.doses.len()
        ));
    }
    if got.obs.len() != subject.obs_times.len() {
        return Err(format!(
            "event_pk.obs has {} entr(ies) but the subject has {} observation record(s) (#748)",
            got.obs.len(),
            subject.obs_times.len()
        ));
    }
    for j in 0..subject.obs_times.len() {
        let t = subject.obs_times[j];
        let want = (model.pk_param_fn)(theta, eta_at(t), subject.obs_cov(j), t);
        if !pk_bits_eq(&want, &got.obs[j]) {
            return Err(format!(
                "event_pk.obs[{j}] (record at t={t}) diverges from an independent re-derivation \
                 — a wrong per-event covariate or occasion κ in the per-event PK builder, reused \
                 verbatim by the frozen-replay verifier, which cannot catch it (#748)"
            ));
        }
    }
    if got.pk_only.len() != subject.pk_only_times.len() {
        return Err(format!(
            "event_pk.pk_only has {} entr(ies) but the subject has {} EVID=2 record(s) (#748)",
            got.pk_only.len(),
            subject.pk_only_times.len()
        ));
    }
    for m in 0..subject.pk_only_times.len() {
        let t = subject.pk_only_times[m];
        let want = (model.pk_param_fn)(theta, eta_at(t), subject.pk_only_cov(m), t);
        if !pk_bits_eq(&want, &got.pk_only[m]) {
            return Err(format!(
                "event_pk.pk_only[{m}] (EVID=2 record at t={t}) diverges from an independent \
                 re-derivation — a wrong per-event covariate or occasion κ (#748)"
            ));
        }
    }
    Ok(())
}

/// Simulate a declarative `[adaptive_dosing]` block over a population — the
/// file-driven counterpart to [`simulate_adaptive`] (epic #391, **beta**).
///
/// Where [`simulate_adaptive`] takes a hand-written controller closure, this entry
/// point takes the parsed `[adaptive_dosing]` `spec` and compiles it into the
/// *same* reactive engine: the `observe` expression becomes the monitored signal,
/// the `when … : …` ladder becomes the controller, and the block's `at` becomes
/// the decision schedule. Everything downstream — the dose ledger, the decision
/// log (including holds), and the frozen-schedule replay verifier — is identical
/// to the programmatic path, so the declarative block inherits every S1 guarantee
/// (a re-emitted fixed regimen reproduces [`simulate`] bit-for-bit via the
/// verifier; a genuinely reactive schedule is replay-checked each run).
///
/// Obtain `spec` from [`crate::parse_full_model_file`]
/// (`ParsedModel::adaptive_dosing`, `None` when the model declares no block).
///
/// ## The spec owns the schedule and the monitor
///
/// The decision schedule (`spec.at`) and the monitored signal (`spec.observe`)
/// come from the block, so `opts.decision_times` and `opts.monitors` **must be
/// left empty**: setting either is a typed error rather than a silently-ignored
/// field (the spec would otherwise have two sources of truth). `opts.seed`,
/// `opts.verify`, and `opts.max_decisions` apply exactly as for
/// [`simulate_adaptive`].
///
/// The `observe` expression is compiled against the model, so it titrates on the
/// derived signal it names — `observe = central / V` drives on the concentration,
/// not the raw compartment amount. With `with_assay_error`, the designated
/// endpoint's residual error noises the reading on the S1.5 controller-assay
/// substream; an `observe` whose endpoint is ambiguous is rejected at parse time.
pub fn simulate_adaptive_from_spec(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    spec: &crate::sim::adaptive::AdaptiveDosingSpec,
    opts: &AdaptiveSimulateOptions,
) -> Result<AdaptiveSimulationResult, String> {
    // The spec is the single source of truth for the decision schedule and the
    // monitored signal; an `opts` that *also* sets them is ambiguous (two
    // schedules, two monitors) — reject it rather than silently pick one.
    if !opts.decision_times.is_empty() {
        return Err(
            "simulate_adaptive_from_spec takes its decision schedule from the \
             [adaptive_dosing] block's `at`; leave `opts.decision_times` empty"
                .to_string(),
        );
    }
    if !opts.monitors.is_empty() {
        return Err(
            "simulate_adaptive_from_spec takes its monitor from the [adaptive_dosing] \
             block's `observe`; leave `opts.monitors` empty"
                .to_string(),
        );
    }

    // Compile the block against the model: the `observe` expression, the single
    // `signal` monitor (carrying the assay endpoint under `with_assay_error`), and
    // the per-(subject, replicate) controller factory. This is also the ODE gate —
    // the reactive driver runs on the ODE engine, so an analytical model is rejected
    // here rather than allowed to silently produce nothing.
    let compiled = crate::sim::adaptive_control::compile_adaptive(model, spec)
        .map_err(|e| format!("simulate_adaptive_from_spec: {e}"))?;
    // Parity with `fit()`: model-referenced covariates (e.g. a `Selected` error
    // model's selector) must be present too, not just the `observe` signal (#658).
    first_error(&check_covariates(model, population))?;
    // A `Selected` error model keys endpoints by selector branch, not CMT, so the
    // compartment-keyed assay would draw NaN — reject it (see the helper's note, #658).
    reject_selected_error_for_adaptive(model)?;
    // An `observe` covariate absent from the data would silently read 0.0 and
    // drive the controller off a wrong signal (`central / WT` → central / 0 = inf).
    // Apply the same loud check fits use for model covariates (`check_covariates`).
    let missing: Vec<&str> = compiled
        .observe_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();
    if !missing.is_empty() {
        let available = if population.covariate_names.is_empty() {
            "(none)".to_string()
        } else {
            population.covariate_names.join(", ")
        };
        return Err(format!(
            "simulate_adaptive_from_spec: [adaptive_dosing] `observe` references covariate(s) \
             not found in data (case-sensitive): {}. Available covariate columns: {}.",
            missing.join(", "),
            available
        ));
    }
    let ode = model
        .ode_spec
        .as_ref()
        .expect("compile_adaptive accepted the model, so it carries an ODE spec");
    // Pair the single `signal` monitor with its compiled `observe` expression (the
    // latent/`Ipred` case); under `Dv` `compiled.observe` is `None`, so the driver
    // reads the model's own output for that cmt — value and σ from one source.
    let monitors = vec![crate::sim::adaptive::AdaptiveMonitor {
        spec: &compiled.monitors[0],
        observe: compiled.observe.as_ref(),
    }];

    run_adaptive_population(
        model,
        ode,
        population,
        params,
        n_sim,
        &spec.at,
        &monitors,
        compiled.make_controller.as_ref(),
        // The declarative block's optional therapeutic band feeds the
        // `pct_time_in_window` metric (it never influences dosing).
        spec.target_window,
        // ...and the optional exposure band feeds `auc_target_attainment` (likewise
        // metrics-only); `Some` here is what turns on the signal-AUC pass.
        spec.auc_target,
        opts,
    )
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

    // ODE-accumulated TTE event-time simulation needs a finite horizon, which this
    // uncertainty path does not yet expose — validate for a clean Err here (the
    // inner chokepoint would otherwise enforce the same contract as a panic).
    #[cfg(feature = "survival")]
    validate_tte_simulatable(model, population, None)?;

    // Parity with `fit()`: reject a referenced covariate absent from the data
    // rather than silently reading it as 0.0 (a `Selected` error-model selector
    // would otherwise route every row to branch 0). See #658.
    first_error(&check_covariates(model, population))?;

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
    // Per-subject simulation diagnostics (#762/#763) are collected but not surfaced on
    // this uncertainty-aggregation path (its return is the flat row vec); the underlying
    // per-subject handling — no whole-run panic, degenerate subjects censored — still
    // applies. Use `simulate_with_options` when the warnings matter.
    let mut sim_warnings: Vec<String> = Vec::new();
    for (k, params) in draws.iter().enumerate() {
        // A parameter draw can land in the flip-flop regime even when the point
        // estimate is in-domain. For a twin-less transit/IG closed form,
        // `simulate_inner_with_draw`'s `assert_absorption_flip_flop_no_twin` would then
        // panic — aborting the *entire* uncertainty run. Skip such a draw with a
        // recorded warning instead, so the remaining draws still yield results (#786).
        // The single-shot `predict()`/`simulate()` paths keep the panic (their
        // Vec-returning contract). Twin-carrying models return `None` here and proceed
        // (they reroute per-eval), so this only skips genuinely un-simulatable draws.
        if let Some(msg) = check_absorption_flip_flop_no_twin(model, population, &params.theta) {
            sim_warnings.push(format!("uncertainty draw {} skipped — {}", k + 1, msg));
            continue;
        }
        let mut rows = simulate_inner_with_draw(
            model,
            population,
            params,
            opts.n_sim_per_draw,
            k + 1,
            None,
            None,
            &mut rng,
            &mut sim_warnings,
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

/// The result of [`simulate_with_options`]: the simulated rows plus any non-fatal
/// per-subject diagnostics collected during the run.
///
/// `warnings` is the simulation analogue of [`FitResult::warnings`]: a subject whose
/// draw degenerates — a non-positive / non-finite analytic hazard rate that produces
/// no event (#763), or a hazard so extreme its recurrent stream is skipped rather than
/// materialised (#762) — is handled *per subject* (censored / skipped, the run
/// continues) and named here, instead of silently vanishing into the censored rows or
/// aborting the whole run. Empty for a clean simulation. The simpler `simulate()` /
/// `simulate_with_seed()` entry points apply the same per-subject handling but return
/// only the rows (no diagnostics channel) — use `simulate_with_options` when the
/// warnings matter (e.g. a population VPC).
#[derive(Debug, Clone, Default)]
pub struct SimulationOutput {
    pub results: Vec<SimulationResult>,
    pub warnings: Vec<String>,
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
    assert_absorption_closed_form_support(model, population);
    assert_absorption_flip_flop_no_twin(model, population, &params.theta);
    // A time-varying covariate on a survival hazard would be silently frozen — panic
    // rather than return a subtly wrong prediction / simulation (#741; fit() Err's).
    #[cfg(feature = "survival")]
    assert_survival_tv_covariates(model, population);
    assert_analytic_readout_support(model, population);
    assert_absorption_dosing_supported(model, population);

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
/// **RTTE (`type = rtte`) semantics.** This computes single-event quantities from the
/// hazard curve, so for a repeated-event endpoint `survival`, `median_survival`,
/// `mean_survival` and `cif` describe **time to the *first* event**, not the recurrent
/// process. For `clock = forward` (Andersen–Gill), the recurrent quantity — the expected
/// event count `E[N(t)] = H(t)` — is the `cum_hazard` field (with `hazard` its rate
/// `h(t)`). For `clock = reset` (gap-time / renewal), `cum_hazard` is the cumulative
/// hazard of a single gap evaluated at *absolute* time and is **not** the renewal mean
/// `E[N(t)]`, so it is not a meaningful recurrent quantity here. A recurrence-aware
/// predictor is a later slice (3.3); until then read `cum_hazard`/`hazard` only for
/// clock-forward RTTE, not the survival summaries and not for clock-reset.
///
/// Returns an empty Vec when the model has no TTE endpoints.
#[cfg(feature = "survival")]
pub fn predict_survival(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    time_grid: &[f64],
) -> Vec<SurvivalPredictionResult> {
    // Deliberately no `assert_absorption_flip_flop_no_twin` guard here (unlike
    // `predict`/`simulate`): a survival prediction cannot be corrupted by a degenerate
    // twin-less-flip-flop transit PK. A hazard that reads the PK is ODE-accumulated (the
    // model then carries `ode_spec` and never takes the closed-form transit path), and a
    // closed-form-family hazard does not read the PK at all — so the closed form's
    // clamped `0` can never reach `S(t)`. See #776.
    use crate::survival::{
        cif_curves, hazard_and_cum_hazard, mean_survival, median_survival, tte_cause_params,
    };

    // Like predict()/simulate(), the survival curves read the hazard at a frozen
    // baseline covariate snapshot — a time-varying covariate on the hazard would be
    // silently applied at its baseline, so fail loudly instead (#741).
    assert_survival_tv_covariates(model, population);

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
            let crate::types::EndpointLikelihood::Tte { hazard, .. } = endpoint else {
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
#[path = "tests/survival_predict_tests.rs"]
mod survival_predict_tests;

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci`.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests/iov_integration.rs"]
mod iov_integration;

#[cfg(test)]
#[path = "tests/extract_se_tests.rs"]
mod extract_se_tests;

#[cfg(test)]
#[path = "tests/tests_cov_diagnostics.rs"]
mod tests_cov_diagnostics;

#[cfg(test)]
#[path = "tests/tests_sir_fallback.rs"]
mod tests_sir_fallback;

#[cfg(test)]
#[path = "tests/tests_param_corr.rs"]
mod tests_param_corr;

#[cfg(test)]
#[path = "tests/simulate_with_uncertainty_tests.rs"]
mod simulate_with_uncertainty_tests;

// ── SDE end-to-end integration ───────────────────────────────────────────────

#[cfg(test)]
#[path = "tests/sde_integration.rs"]
mod sde_integration;

#[cfg(test)]
#[path = "tests/multi_start_tests.rs"]
mod multi_start_tests;

#[cfg(test)]
#[path = "tests/tests_sdtab_tv_cov.rs"]
mod tests_sdtab_tv_cov;

#[cfg(test)]
#[path = "tests/tests_derived_session_clock.rs"]
mod tests_derived_session_clock;

#[cfg(test)]
#[path = "tests/tests_derived_iov_kappa.rs"]
mod tests_derived_iov_kappa;

// ── Tests: adaptive (state-reactive) dosing — simulate_adaptive (#391 S1.4b) ──
#[cfg(test)]
#[path = "tests/adaptive_sim_tests.rs"]
mod adaptive_sim_tests;

// ── Tests: #748 independent adaptive-snapshot correctness check ──────────────
//
// The frozen-replay verifier reuses the precomputed snapshots (`eta_occ`,
// `decision_pk`, `event_pk`) the driver was handed, so it validates their
// *consumption*, not their *correctness* — a build-loop that feeds a leaf the
// wrong covariate / occasion / κ passes it bit-exact. These tests pin the teeth of
// `verify_adaptive_snapshots`, the independent re-derivation that closes that gap:
// a canonical build is accepted, and each deliberately-corrupted snapshot class
// (the twice-fixed #732 / #739 "decision covariate frozen at t=0" defect included)
// is rejected. Every corruption is constructed to differ from the canonical build,
// so the tests fail if the check ever regresses to a no-op.
#[cfg(test)]
#[path = "tests/adaptive_snapshot_verify_tests.rs"]
mod adaptive_snapshot_verify_tests;
