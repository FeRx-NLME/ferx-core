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
    read_nonmem_csv_filtered_mapped, read_nonmem_csv_mapped,
    read_nonmem_csv_with_covariates_filtered_mapped, read_nonmem_csv_with_covariates_mapped,
    SelectionFilter, ERR_COV_MISSING_COLUMNS, ERR_COV_NON_NUMERIC,
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
    let mut parsed = crate::parser::model_parser::parse_full_model_file(Path::new(model_path))?;
    // A `[covariates]` declaration takes precedence over the explicit
    // `covariate_columns` argument; otherwise fall back to the argument (or
    // legacy auto-detect when both are absent).
    let opts = options.unwrap_or_default();
    let sel_filter_fit = build_selection_filter_merged(&parsed.fit_options, &opts)?;
    let (data_path, data_path_warning) = resolve_data_path(parsed.data_path.as_deref(), data_path)?;
    let data_path = data_path.as_str();
    let (mut population, covariate_table) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data_path,
        covariate_columns,
        None,
        sel_filter_fit.as_ref(),
        &parsed.column_map,
    )?;
    // #1064: bind level blocks against the data before anything reads
    // the parameter vector — the level count, and therefore `n_theta`, is a
    // property of the dataset. Mirrors `run_model_with_data_inits`.
    {
        let model_text = std::fs::read_to_string(model_path)
            .map_err(|e| format!("Failed to re-read model file for level binding: {e}"))?;
        crate::api::bind_theta_levels(&mut parsed, &model_text, &mut population)?;
    }
    let mut model = parsed.model;
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
pub(crate) fn perturb_init(
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
/// valid candidate. See `estimation::outer_optimizer::ofv_is_valid`.
pub(crate) fn multistart_prefers(b_ofv: f64, b_conv: bool, c_ofv: f64, c_conv: bool) -> bool {
    let valid = crate::estimation::outer_optimizer::ofv_is_valid;
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
    // #1064: a `theta NAME[...]` block has no levels until it is bound
    // to data. Fitting one unbound would gather out of an empty level table and
    // predict NaN everywhere; refuse, and name the two ways out.
    if !model.theta_blocks.unbound_level_blocks().is_empty() {
        return Err(format!(
            "`theta {}[...]` was never bound to data, so it has no levels. Fit \
             through a file entry point (`fit_from_files`, `run_model_with_data`, or the \
             CLI), which binds level blocks against the dataset — or declare the block \
             explicitly as `theta {}[N](...)` and index it with your own column.",
            model.theta_blocks.unbound_level_blocks().join("`, `theta "),
            model.theta_blocks.unbound_level_blocks()[0],
        ));
    }
    // Mixture models (#977). Phase 3 wires the K-fold log-sum-exp FOCE/FOCEI
    // objective via the derivative-free (BOBYQA) outer optimizer. Other
    // estimators, inter-occasion variability, and adaptive-Gauss-Hermite
    // marginalisation are not yet supported — reject them clearly rather than
    // silently ignoring the mixture structure.
    if model.mixture.is_some() {
        // The objective keys the mixture path off `params.mixture` (carried from
        // `init_params` through `unpack_params`). If a caller passes custom
        // `init_params` with `mixture: None` — e.g. params rebuilt from a prior
        // `FitResult` — the fit would silently run the single-population objective
        // with `MIXNUM` pinned to the class-1 default. Fail loudly instead: the
        // per-class Omega/Sigma must be present.
        if init_params.mixture.is_none() {
            return Err(
                "mixture model ([mixture] block, #977) requires per-class Omega/Sigma in \
                 `init_params.mixture`; pass the parsed model's `default_params` (or rebuild \
                 them from the model file) rather than params with `mixture: None`"
                    .to_string(),
            );
        }
        for m in options.method_chain() {
            if !matches!(
                m,
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::Saem
                    | EstimationMethod::Bayes
                    // IMP / IMPMAP run a class-partitioned MCEM (per-class IS
                    // E-step + responsibility-weighted M-steps) when estimating,
                    // and form the class-marginal `−2 Σ log Σ_k p_ik L_ik` on the
                    // objective-evaluation (EONLY) path (#985). IOV, FREM, SDE
                    // and per-class Ω/Σ overrides are refused inside
                    // `run_mcem_mixture`.
                    | EstimationMethod::Imp
                    | EstimationMethod::Impmap
            ) {
                return Err(format!(
                    "mixture models (#977) currently support FOCE / FOCEI / SAEM / Bayes / IMP / \
                     IMPMAP; the {} method is not yet wired for mixtures (#985)",
                    m.label()
                ));
            }
        }
        // Per-class Omega/Sigma overrides under Bayes (#985): the Bayes Omega block
        // is a single conjugate draw shared across classes, so `omega(k)`/`sigma(k)`
        // cannot be honoured. `bayes.rs` re-checks this defensively, but reject it
        // here — before any stage runs — so `method = [focei, bayes]` does not burn a
        // full FOCEI fit only to fail at the hand-off.
        if options.method_chain().contains(&EstimationMethod::Bayes) {
            if let Some(mp) = init_params.mixture.as_ref() {
                if !mp.omega_override_addr.is_empty() || !mp.sigma_override_addr.is_empty() {
                    return Err(
                        "Bayesian estimation (method = bayes) does not yet support per-class \
                         Omega/Sigma overrides (omega(k)/sigma(k)) in a mixture; Omega/Sigma are \
                         shared across classes. Fit with FOCE/FOCEI, or drop the per-class \
                         overrides (#985)."
                            .to_string(),
                    );
                }
            }
        }
        // Inter-occasion variability under a mixture (#985): the per-class inner
        // solve carries per-occasion κ for the shared base `omega_iov` (each class's
        // `class_params` keeps it), and the packed layout already interleaves the κ
        // segment ahead of the per-class Ω/Σ override tail. The FOCE marginal uses
        // `foce_subject_nll_iov` per class and the outer gradient routes to FD, so
        // IOV + mixture now fits (previously rejected here).
    }
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
    // #1064: `auto` switches away from BOBYQA above `BOBYQA_MAX_DIM` on its own
    // (see `Optimizer::resolve_auto`); an explicit `optimizer = bobyqa` is the
    // user's call, but it must not be a silent hour. BOBYQA interpolates a
    // quadratic over the whole space, so the model it has to build grows
    // quadratically in the parameter count.
    {
        let dim = model.free_packed_dim();
        if options.optimizer == Optimizer::Bobyqa && dim > crate::types::BOBYQA_MAX_DIM {
            pre_warnings.push(format!(
                "optimizer = bobyqa with {dim} free parameters: BOBYQA builds a quadratic \
                 interpolation model over the whole parameter space, which grows \
                 quadratically in the parameter count and is unlikely to make progress at \
                 this size. `optimizer = nlopt_lbfgs` (or the default `auto`, which switches \
                 above {}) costs O(n) finite-difference passes instead.",
                crate::types::BOBYQA_MAX_DIM
            ));
        }
    }
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

pub(crate) fn saem_non_mu_referenced_individual_params_warning(
    model: &CompiledModel,
) -> Option<String> {
    let mut names = Vec::new();
    for (param_name, &eta_idx) in model.indiv_param_names.iter().zip(model.eta_map.iter()) {
        if eta_idx < 0 {
            continue;
        }
        // Skip parser-internal readout parameters (#486). A `[scaling] y = ... + ETA(k)`
        // readout is desugared into `__ferx_ro_eta{k} = ETA(k)`, which `detect_mu_refs` has
        // no pattern for — so it lands here and the user is told to mu-reference a parameter
        // that does not appear in their model file and that they cannot act on. Matches the
        // four sibling consumers of this list (`output_columns.rs`, `validation.rs` ×2,
        // `model_parser.rs`), all of which already filter (PR #950 review #3).
        if crate::parser::model_parser::is_synthetic_readout_param(param_name) {
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
    let option_diags = check_model_options(model, options);
    first_error(&option_diags)?;

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
    //
    // Through the shared filter, so `ferx check` suppresses exactly what `fit()`
    // does — see `reader_warning_suppressed`.
    accumulated_warnings.extend(
        population
            .warnings
            .iter()
            .filter(|w| !crate::api::validation::reader_warning_suppressed(model, w))
            .cloned(),
    );

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
    // Warning-severity *option* diagnostics (e.g. W_GN_NO_RANDOM_EFFECTS, #1006).
    // `first_error` above consumes only errors, so without this a warning added to
    // `check_model_options` is reported by `ferx check` and silently dropped by
    // `fit()` — the same treatment `check_model_data_warnings` gets, one line up.
    for d in option_diags.iter().filter(|d| !d.is_error()) {
        accumulated_warnings.push(d.message.clone());
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
                    mixture_posteriors: None,
                });
            }
            let prev = result.as_ref().expect(
                "IMP stage: prior OuterResult must exist (synthesised above when standalone)",
            );
            // Mixture (#985): evaluate the class-marginal IS objective
            // −2 Σ log Σ_k p_ik L_ik (per-class MAP + IS, combined), rather than
            // the single-population marginal.
            let is_call = if model.mixture.is_some() {
                crate::estimation::importance_sampling::run_importance_sampling_mixture(
                    model,
                    population,
                    &prev.params,
                    &stage_opts,
                )
            } else {
                crate::estimation::importance_sampling::run_importance_sampling(
                    model,
                    population,
                    &prev.params,
                    &prev.eta_hats,
                    &prev.h_matrices,
                    &prev.kappas,
                    &stage_opts,
                )
            };
            match is_call {
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
        for w in stage_result
            .warnings
            .iter()
            .filter(|w| keep_gn_zero_eta_warning(w, is_last_estimating))
        {
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
            let marg_call = if model.mixture.is_some() {
                crate::estimation::importance_sampling::run_importance_sampling_mixture(
                    model, population, &r.params, &marg_opts,
                )
            } else {
                crate::estimation::importance_sampling::run_importance_sampling(
                    model,
                    population,
                    &r.params,
                    &r.eta_hats,
                    &r.h_matrices,
                    &r.kappas,
                    &marg_opts,
                )
            };
            match marg_call {
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

    // Compute per-subject diagnostics. For a mixture (#985) each subject's
    // diagnostics are evaluated under its own MIXEST class: `result.eta_hats` are
    // the winning class's EBEs, so predictions built with `MIXNUM` at the class-1
    // default would mix a class-2 η̂ with class-1 typical values.
    let mixest_classes: Option<Vec<usize>> = result
        .mixture_posteriors
        .as_ref()
        .map(|mp| mp.mixest.clone());
    //
    // ODE models run this pass inside a solver-statistics scope (#1080 Part B): it is the one
    // production sweep that integrates every subject at the final estimates through the
    // ordinary dispatch, so it is where `min_dt` clamps and `auto`'s escalation/rejection
    // decisions can be observed without threading a stats sink through every predictor. Costs
    // one thread-local read per segment on the ODE path and nothing at all elsewhere.
    let solver_stats_scope =
        integrates_odes(model).then(crate::ode::solver::SolverStatsScope::enter);
    let mut subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        &result.kappas,
        options.interaction,
        mixest_classes.as_deref(),
    );
    let ode_solver_stats = solver_stats_scope
        .map(|scope| scope.collected())
        .unwrap_or_default();

    // Mixture (#977 Phase 5): thread the converged per-subject posteriors onto
    // each SubjectResult so output.rs can emit the PMIX_1..PMIX_K and MIXEST
    // columns. MIXEST is stored 1-based to match the NONMEM `MIXEST` convention.
    if let Some(mp) = &result.mixture_posteriors {
        for (sr, (pmix, mixest)) in subjects
            .iter_mut()
            .zip(mp.pmix.iter().zip(mp.mixest.iter()))
        {
            sr.pmix = Some(pmix.clone());
            sr.mixest = Some(mixest + 1);
        }
    }

    // Post-fit: compute [derived] and [output] columns, and populate per_obs_tad
    // (with individual lagtime) for the mandatory TAD column in output.rs.
    if !model.derived_exprs.is_empty() || !model.output_columns.is_empty() || model.has_lagtime() {
        compute_extra_output_columns(
            model,
            population,
            &result.params.theta,
            &result.kappas,
            &mut subjects,
            mixest_classes.as_deref(),
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
        // Analytical model with a dose into a compartment the superposition state
        // helper cannot express — a zero-order input into the oral depot (#400),
        // or any bolus outside compartment 1 / infusion outside central (#375).
        // ipred is exact (those subjects reroute to the event-driven walk), but
        // per-obs compartment states return empty (→ NaN) rather than report
        // silently-wrong amounts.
        if model.ode_spec.is_none()
            && population
                .subjects
                .iter()
                .any(|s| crate::pk::dose_needs_event_walk(model.pk_model, s))
        {
            warnings.push(
                "W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL: analytical model with a \
                 dose the closed-form superposition cannot place — a zero-order input \
                 into the oral depot (RATE=-2 D1 / infusion into compartment 1), or a \
                 bolus into a compartment other than 1, or an infusion into a \
                 non-central compartment — compartment states are not available for \
                 those subjects (predictions are exact); [derived] expressions that \
                 reference compartments[i] evaluate to NaN for them. Use an ODE model \
                 if per-compartment amounts are required."
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
                Ok(sir) => {
                    for w in &sir.warnings {
                        warnings.push(format!("SIR: {}", w));
                    }
                    Some(sir)
                }
                Err(e) => {
                    warnings.push(format!("SIR failed: {}", e));
                    None
                }
            }
        } else {
            // No covariance matrix — but this is not the end of the road for a
            // `sir = true` request: the non-PD fallback below is armed by
            // `sir = true` as well as by `covariance_fallback = sir` (#972).
            // The warning, if SIR really cannot run, is emitted after the
            // fallback has had its chance.
            None
        }
    } else {
        None
    };

    // SIR fallback: when the FD Hessian is non-PD and the user asked for SIR
    // (`covariance_fallback = sir` or `sir = true`), run SIR with the rectified
    // |eigenvalue| proposal built inside compute_covariance.
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

    // Only now, with both SIR paths resolved, can we tell a user who asked for
    // SIR *why* they got none — and point them at the right knob (#972).
    if !crate::cancel::is_cancelled(&options.cancel) {
        if let Some(msg) = sir_unavailable_warning(
            options.sir,
            options.run_covariance_step,
            result.bayes.is_some(),
            result.covariance_matrix.is_some(),
            result.sir_fallback_proposal.is_some(),
            sir_result.is_some() || sir_fallback_result.is_some(),
        ) {
            warnings.push(msg);
        }
    }

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
    // ODE-solver health at the final estimates (#1080 Part B): min-`dt` clamps, `auto`
    // escalations, and escalations the guard discarded — none of which any production path
    // reported before. Emitted typed at source, at `Info` severity when the only thing to
    // report is that `auto` escalated and it worked.
    if let Some((msg, entry)) = ode_solver_diagnostics_warning(&ode_solver_stats, options) {
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
        restored_from_checkpoint: false,
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
        // Left empty unless a kappa actually carries a weight, so an ordinary
        // IOV fit's YAML/fitrx output is byte-identical to before #1031.
        kappa_weights: if model.has_weighted_kappa() {
            model
                .kappa_weights
                .iter()
                .map(|w| w.as_ref().map(|w| w.expr.clone()))
                .collect()
        } else {
            Vec::new()
        },
        kappa_weight_typical: kappa_weight_typicals(model, population, &result.params.theta),
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
