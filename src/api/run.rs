#![allow(unused_imports)]
//! Extracted verbatim from `api/mod.rs` (production peel). See the module-
//! doc / Key Modules table for the split rationale.
use super::*;
use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::{
    chol_lt_idx, lower_tri_iter, omega_packed_len, theta_packs_log,
};
use crate::estimation::saem;
use crate::io::datareader::{
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_mapped, read_nonmem_csv_routed,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    MissingDvPolicy, ObsRouting, SelectionFilter, ERR_COV_MISSING_COLUMNS, ERR_COV_NON_NUMERIC,
};
use crate::pk;
use crate::propensity_match::MatchMethod;
use crate::sim::adaptive::{
    AdaptiveRun, AdaptiveSubjectMetrics, ControllerCtx, DecisionLogEntry, DoseAction,
    DoseLedgerEntry, MonitorSpec,
};
use crate::stats::likelihood::{
    build_frem_r_override, compute_cwres, foce_subject_nll, foce_subject_nll_iov,
};
use crate::stats::residual_error::{
    compute_iwres_with_correlations, compute_r_matrix_with_correlations,
    compute_r_matrix_with_correlations_scaled, iwres_autocorrelation,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Log-transform every observation (including M3 LLOQ values carried on CENS
/// rows — they live in the same `observations` vector) in place, for LTBS case 2
/// (`log(DV) ~ additive`, natural-scale data). Returns the count of non-positive
/// DV values, which are floored to [`crate::pk::LTBS_FLOOR`] before the log so the
/// result stays finite. Case 1 (`DV ~ log_additive`, `dv_pre_logged`) must NOT
/// call this — the DV is already on the log scale.
pub(crate) fn log_transform_observations(pop: &mut Population) -> usize {
    let mut n_nonpos = 0usize;
    for subject in &mut pop.subjects {
        for v in &mut subject.observations {
            // A non-finite DV is left alone. `f64::max` returns the *non-NaN*
            // operand, so `NaN.max(LTBS_FLOOR).ln()` would quietly turn a NaN
            // design placeholder (#957) into a finite, extreme "observation" and
            // let a mis-routed simulation population fit to completion. Leave it
            // non-finite so `E_NONFINITE_DV` (and any downstream `is_finite`
            // guard) still sees it.
            if !v.is_finite() {
                continue;
            }
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

    // #1064: a `theta NAME[COL, ...]` block declares one θ per observed
    // combination, so its level count is a property of the data. Bind it now —
    // this synthesizes the per-record index column on every subject and
    // re-parses the model with the real θ count — before anything reads
    // `parsed.model`'s parameter vector.
    {
        let model_text = std::fs::read_to_string(model_path)
            .map_err(|e| format!("Failed to re-read model file for level binding: {e}"))?;
        crate::api::bind_theta_levels(&mut parsed, &model_text, &mut population)?;
    }

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
pub(crate) fn derive_output_occasions(
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
/// What [`simulation_design_covariates`] returns: the population's covariate
/// names, in declaration order, plus one covariate map per synthetic subject.
pub(crate) type SimulationDesignCovariates = (Vec<String>, Vec<HashMap<String, f64>>);

/// Resolve `[simulation] covariate NAME = ...` into the population's covariate
/// names plus one covariate map per synthetic subject (#1083).
///
/// A simulated trial exists to invent arms that are in no dataset — *"what would
/// a 300-subject arm of this design look like?"* — so there is no row to read
/// `NARM` or `WPSE` from, and the covariates the design turns on have to be
/// stated as part of the design. Requiring them is the convention: an unresolved
/// weight has no defensible default (a silent `1` gives a simulated arm the
/// variability of a single-subject arm, a silent `0` gives it none, and
/// `BinOp::Div` underflows to `0.0` rather than `inf`, so neither shows up as a
/// `NaN`).
///
/// The missing-value report is deliberately *not* the shared `check_covariates`
/// message: on this path there is no data file, so "not found in data … Available
/// covariate columns: (none)" reads as a typo report on a column that was never
/// going to exist, and it names no fix.
///
/// Returned in declaration order. The per-subject vector has one entry per
/// subject; the parser has already broadcast a scalar and checked every list
/// against `n_subjects`, so a short list cannot reach here.
///
/// Subject-level only: `obs_cov` / `dose_cov` fall back to `Subject::covariates`
/// when the per-record vectors are empty, and a synthetic arm's design covariates
/// — its size, its reported standard error — are constant over the arm by
/// construction.
pub(crate) fn simulation_design_covariates(
    sim_spec: &SimulationSpec,
    model: &CompiledModel,
) -> Result<SimulationDesignCovariates, String> {
    let names: Vec<String> = sim_spec.covariates.iter().map(|(n, _)| n.clone()).collect();

    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        // A a level block index column (#1064) is synthesized per record by
        // `bind_theta_levels` from the design's own covariates and observation
        // grid — it is not something the user can state, so demanding it here
        // would make every level-block model unsimulatable, naming a column with a
        // reserved `__level_` prefix as the fix.
        .filter(|name| !crate::api::levels::is_level_index_column(name))
        .filter(|name| !names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "[simulation]: the model references covariate(s) {} that the simulation design \
             does not supply. A simulated trial invents arms that are in no dataset, so their \
             covariates — an arm size behind `weight = N`, a reported standard error behind \
             `weight = SE`, a body weight — are part of the design and must be stated: add \
             `covariate {} = <value>` (or `= [v1, ...]`, one per subject) to [simulation].",
            missing
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", "),
            missing[0],
        ));
    }

    let per_subject: Vec<HashMap<String, f64>> = (0..sim_spec.n_subjects)
        .map(|i| {
            sim_spec
                .covariates
                .iter()
                .map(|(n, vals)| (n.clone(), vals[i]))
                .collect()
        })
        .collect();
    Ok((names, per_subject))
}

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
    // Binary endpoints observe on the **fixed grid** (§8.8.2), so — like a continuous
    // endpoint and unlike TTE — they need `times`, not a `horizon` (#760 Slice 1b).
    #[cfg(feature = "survival")]
    let binary_cmts: Vec<usize> = parsed.model.binary_cmts();
    #[cfg(not(feature = "survival"))]
    let binary_cmts: Vec<usize> = Vec::new();
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
        if !binary_cmts.is_empty() {
            return Err(
                "[simulation] has no `times`, but the model has a [binary_model] endpoint. \
                 A binary outcome is observed on a fixed schedule, not located in time like \
                 a TTE event, so it needs `times = [...]` (a `horizon` alone gives it nothing \
                 to observe)"
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

    // The Gaussian observation grid exists only for a model that actually has a
    // continuous endpoint. A binary-only (or TTE + binary) design also needs `times` —
    // discrete outcomes are observed on the fixed grid — but materialising the grid as
    // `observations`/`obs_cmts` would fabricate |times| phantom Gaussian rows on CMT 1,
    // which `check_per_cmt_error_model` then rejects ("[error_model] has no entry for
    // observed compartment(s) 1"). The binary rows live in `obs_records` instead, so a
    // model with no residual error contributes an empty Gaussian grid.
    let n_gauss = if model_has_continuous {
        sim_spec.obs_times.len()
    } else {
        0
    };
    // Per-subject covariate values from `[simulation] covariate NAME = ...` (#1083).
    let (sim_covariate_names, sim_subject_covariates) =
        simulation_design_covariates(&sim_spec, &parsed.model)?;
    let subject_covariates = |i: usize| -> HashMap<String, f64> {
        sim_subject_covariates
            .get(i - 1)
            .cloned()
            .unwrap_or_default()
    };

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
            obs_times: if model_has_continuous {
                sim_spec.obs_times.clone()
            } else {
                Vec::new()
            },
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_gauss],
            obs_cmts: vec![1; n_gauss],
            covariates: subject_covariates(i),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_gauss],
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
                // Binary template rows: one placeholder per (binary CMT × observation
                // time), since a binary endpoint observes on the fixed grid. `state = 0`
                // is a placeholder overwritten by the draw in the write-back below —
                // the same template-then-realise shape the TTE rows above use (#522).
                // Without these the sampler has no records to walk and `--simulate`
                // emits zero binary rows (#760 Slice 1b).
                .chain(binary_cmts.iter().flat_map(|&cmt| {
                    sim_spec.obs_times.iter().map(move |&time| {
                        crate::types::ObsRecord::DiscreteState {
                            time,
                            // Synthetic `[simulation]` subjects carry no resets, so the
                            // internal and user clocks coincide.
                            raw_time: time,
                            state: 0,
                            cmt,
                        }
                    })
                }))
                .collect(),
            // `obs_records` is unconditional since Phase 4.0, but the only records
            // this synthetic TTE subject carries are `Event`s (survival-gated); with
            // the feature off there is no TTE endpoint, so the vec is empty.
            #[cfg(not(feature = "survival"))]
            obs_records: Vec::new(),
        })
        .collect();
    let mut template = Population {
        subjects,
        covariate_names: sim_covariate_names,
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // #1064: bind level blocks against the synthetic design, exactly as
    // the data path binds them against a dataset — the levels come from the
    // `[simulation]` covariates and observation grid. Every level gets the
    // declaration's broadcast init, since the DSL has no way to state per-level
    // simulation values; a design that needs distinct ones should use the
    // explicit `theta NAME[N]` form and its own index column.
    {
        let model_text = std::fs::read_to_string(model_path)
            .map_err(|e| format!("Failed to re-read model file for level binding: {e}"))?;
        crate::api::bind_theta_levels(&mut parsed, &model_text, &mut template)?;
    }
    let template = template;

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
    // A subject that produced *zero* simulated rows (a binary-only design whose
    // linear predictor is NaN for every record — see the binary write-back below)
    // still owns template `DiscreteState` placeholders that must be cleaned up, so
    // the per-subject body has to run even with no draws. Binding to an empty slice
    // (rather than `continue`-ing) lets the placeholder-dropping pass reach it; the
    // Gaussian and TTE write-backs are self-guarding no-ops on an empty `sims`.
    let no_sims: Vec<&SimulationResult> = Vec::new();
    for subject in &mut population.subjects {
        let sims = sims_by_id.get(subject.id.as_str()).unwrap_or(&no_sims);

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
                // Replace only the TTE rows. An earlier version assigned `events`
                // wholesale, which silently destroyed the binary `DiscreteState`
                // templates created above — so a joint TTE + binary `--simulate`
                // emitted zero binary rows, reintroducing the very bug the binary
                // producer exists to fix.
                subject
                    .obs_records
                    .retain(|r| !matches!(r, crate::types::ObsRecord::Event { .. }));
                subject.obs_records.splice(0..0, events);
            }
        }

        // Binary write-back (#760 Slice 1b): stamp each simulated 0/1 outcome onto its
        // template `DiscreteState` row, so the fitted dataset carries the drawn states
        // rather than the all-zero placeholders. Unlike the TTE arm above this is an
        // **in-place merge**, not a wholesale replace, so it composes with whatever
        // records already exist instead of dropping them.
        //
        // Rows are paired per CMT in emission order: `simulate_binary` walks
        // `obs_records` in order and pushes one row per record, and `sims` preserves
        // that order, so the k-th `Category` row on a CMT is the k-th `DiscreteState`
        // row on that CMT. A short `sims` list (records skipped for a NaN predictor —
        // warned about at draw time) leaves the remaining templates untouched rather
        // than shifting every later row onto the wrong record.
        #[cfg(feature = "survival")]
        {
            // Pair each simulated draw with its template by identity `(cmt, raw_time)`,
            // holding a queue per key so that duplicated `[simulation] times` (the parser
            // permits `times = [0.5, 0.5, 2]`) each receive their own draw instead of
            // collapsing onto one. A per-CMT queue alone would misalign after a skipped
            // record; a bare map would drop a duplicate's draw. This does both.
            //
            // A template that receives **no** draw is removed, not left at its
            // placeholder. `simulate_binary` skips a record whose predictor is NaN
            // (warning at draw time), and a placeholder `state = 0` left behind would
            // pass `validate_binary_states` and then be scored by the Bernoulli
            // likelihood as a genuine observed failure — fabricated data, which is worse
            // than the misalignment this replaced.
            let mut drawn: HashMap<(usize, u64), std::collections::VecDeque<usize>> =
                HashMap::new();
            for s in sims {
                if let crate::types::SimOutcome::Category { state } = s.outcome {
                    drawn
                        .entry((s.cmt, s.time.to_bits()))
                        .or_default()
                        .push_back(state);
                }
            }
            // Runs unconditionally — and now genuinely so. Two gates once suppressed it:
            // an inner `!drawn.is_empty()` (a partially-NaN subject kept its placeholders)
            // and the outer per-subject `continue` on a `sims_by_id` miss (a *fully*
            // NaN-skipped binary-only subject produces zero rows, so `sims_by_id.get`
            // returned `None` and every placeholder reached `fit()` as a fabricated
            // observed 0). The loop now binds an empty `sims` instead of `continue`-ing,
            // so this drop reaches the all-NaN subject too.
            {
                subject.obs_records.retain_mut(|rec| {
                    let crate::types::ObsRecord::DiscreteState {
                        state,
                        cmt,
                        raw_time,
                        ..
                    } = rec
                    else {
                        return true; // not a discrete row — untouched
                    };
                    // Only this endpoint family's rows are ours to stamp or drop. A CTMM
                    // `DiscreteState` row is also a discrete row, and matching it here
                    // would silently delete it once CTMM simulation lands (#820) —
                    // unreachable today only because CTMM `--simulate` is rejected
                    // upstream, which is exactly the kind of latent data loss this PR
                    // exists to remove.
                    if !binary_cmts.contains(cmt) {
                        return true;
                    }
                    match drawn
                        .get_mut(&(*cmt, raw_time.to_bits()))
                        .and_then(|q| q.pop_front())
                    {
                        Some(next) => {
                            *state = next;
                            true
                        }
                        // No draw for this record (NaN predictor): drop it rather than
                        // fit its placeholder as an observed 0.
                        None => false,
                    }
                });
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

/// Covariates referenced by the model but missing from the `[covariates]`
/// declaration. These are still read (leniently) so the model works; the parser
/// has already warned that they ought to be declared.
fn undeclared_referenced(model: &CompiledModel, decls: &[CovariateDecl]) -> Vec<String> {
    model
        .referenced_covariates
        .iter()
        // A a level block index column (#1064) is synthesized by
        // `bind_theta_levels` after the read, never present in the CSV — asking
        // the reader for it would fail on a column the user never wrote.
        .filter(|c| !crate::api::levels::is_level_index_column(c))
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
pub(crate) fn build_selection_filter_merged(
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
    read_population_for_policy(
        model,
        covariate_decls,
        data_path,
        fallback_columns,
        iov_column,
        filter,
        column_map,
        MissingDvPolicy::Skip,
    )
}

/// [`read_population_for`] for a dataset that will be **simulated from** rather
/// than fitted (#957).
///
/// Identical in every respect but one: an `EVID=0`, `MDV=0` record whose `DV`
/// cell is missing (`.` / `NA` / blank) is kept as a **design point** — a
/// sampling time whose observation has not been generated yet — instead of being
/// skipped as a forgotten `MDV=1` (#258). Writing `DV = .` at every sampling time
/// is the natural way to express a design (and is what NONMEM's `$SIMULATION`
/// accepts), so the fitting reading would otherwise return zero simulated rows
/// for the most idiomatic template there is.
///
/// The kept row carries a placeholder DV (`NaN` for a Gaussian endpoint, the
/// endpoint's first declared state code for an integer-coded one) which the
/// simulated value replaces; do **not** pass the returned population to [`fit`].
/// That contract is enforced, not merely documented: a non-finite observation is
/// rejected by `E_NONFINITE_DV` ([`check_model_data`]) at `fit()` entry.
/// `MDV=1` still excludes the record — that is the user explicitly saying it is
/// not an observation.
pub fn read_population_for_simulation(
    model: &CompiledModel,
    covariate_decls: &Option<Vec<CovariateDecl>>,
    data_path: &str,
    fallback_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
    column_map: &[(String, String)],
) -> Result<(Population, Option<CovariateTable>), String> {
    read_population_for_policy(
        model,
        covariate_decls,
        data_path,
        fallback_columns,
        iov_column,
        filter,
        column_map,
        MissingDvPolicy::KeepAsDesign,
    )
}

/// Shared body of [`read_population_for`] (fit: `MissingDvPolicy::Skip`) and
/// [`read_population_for_simulation`] (`KeepAsDesign`). The policy is the only
/// difference between them.
#[allow(clippy::too_many_arguments)]
fn read_population_for_policy(
    model: &CompiledModel,
    covariate_decls: &Option<Vec<CovariateDecl>>,
    data_path: &str,
    fallback_columns: Option<&[&str]>,
    iov_column: Option<&str>,
    filter: Option<&SelectionFilter>,
    column_map: &[(String, String)],
    missing_dv: MissingDvPolicy,
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
    // Binary **∪ CTMM** — every CMT whose integer DV routes to `ObsRecord::DiscreteState`.
    // Deliberately not named `binary_cmts`: this function used to share that name with the
    // binary-only `Vec` above, and the two are equal today only because CTMM `--simulate`
    // is rejected upstream. When CTMM simulation lands (#820) they diverge.
    let discrete_cmts: std::collections::HashSet<usize> = {
        let mut discrete: std::collections::HashSet<usize> =
            model.binary_cmts().into_iter().collect();
        #[cfg(feature = "markov")]
        discrete.extend(model.ctmm_cmts());
        discrete
    };
    #[cfg(not(feature = "survival"))]
    let discrete_cmts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // One reader call covers every combination: the routing sets are empty for a
    // Gaussian-only model (so no row is routed to `obs_records`), and the reader
    // builds the covariate read set from `decls` when the model declares
    // `[covariates]`, from `fallback_columns` otherwise.
    // Placeholder DV code for an integer-coded design row: the endpoint's first
    // declared state code, so a `state_codes` table that is not 0-based (e.g.
    // `1`/`2`) never gets an undecodable `0` placeholder. Only CTMM declares
    // codes; Binary is `{0,1}` and keeps the `0` default.
    #[cfg(feature = "markov")]
    let design_states: HashMap<usize, usize> = model
        .endpoints
        .iter()
        .filter_map(|(&cmt, ep)| match ep {
            EndpointLikelihood::Ctmm { state_codes, .. } => {
                state_codes.first().map(|&code| (cmt, code))
            }
            _ => None,
        })
        .collect();
    #[cfg(not(feature = "markov"))]
    let design_states: HashMap<usize, usize> = HashMap::new();

    let routing = ObsRouting::tte_and_discrete(&tte_cmts, &discrete_cmts)
        .with_missing_dv(missing_dv)
        .with_design_states(design_states);
    let (decls, extra) = match covariate_decls {
        Some(d) => (Some(d.as_slice()), undeclared_referenced(model, d)),
        None => (None, Vec::new()),
    };
    read_nonmem_csv_routed(
        Path::new(data_path),
        fallback_columns,
        decls,
        &extra,
        iov_column,
        filter,
        &routing,
        column_map,
    )
}
