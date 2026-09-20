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

/// Predict concentrations for a population using given parameters (no random effects).
///
/// Thin wrapper over [`predict_diag`] that discards the diagnostics
/// ([`PredictionOutput::warnings`]). Use the `_diag` form when a model/data finding or an ODE
/// solver diagnostic must be surfaced — this form returns the same rows with nothing attached,
/// which is what #1280 / #1304 were filed about. It is kept because the R wrapper and every
/// existing caller bind this signature.
///
/// Data-reader warnings (e.g. missing II for ADDL doses) are not echoed here **or** by
/// [`predict_diag`]; callers that obtained `population` via [`crate::read_nonmem_csv`] should
/// inspect `population.warnings` before calling either.
///
/// Subjects are evaluated in parallel when more than one worker is available.
/// Calls made inside a [`PoolPlan`] reuse that enclosing pool; standalone calls
/// use ferx's persistent default pool. Results are still returned in population
/// and observation order.
///
/// **Gaussian rows only.** Non-Gaussian endpoints keep their own entry points, because
/// their prediction is not a scalar concentration: TTE → [`predict_survival`], binary →
/// [`predict_categorical`]. A model whose only endpoint is non-Gaussian therefore gets an
/// empty vec here — call the matching predictor instead. (CTMM has no predictor at all
/// yet, so a CTMM model with *no* continuous endpoint is rejected fail-loud below rather
/// than returning empty; a mixed continuous + CTMM model still gets its Gaussian rows.)
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult> {
    predict_diag(model, population, params)
        .unwrap_or_else(|e| panic!("{e}"))
        .results
}

/// The rows [`predict`] returns, plus the diagnostics it discards.
///
/// `warnings` carries the non-fit diagnostics bundle: parse warnings, data-reader warnings,
/// the model/data checks (`W_STEADY_STATE_*`, `W_SDE_*`, `W_NEGATIVE_LAGTIME`, …),
/// experimental-feature notices, and — for an `[odes]` model — the ODE-solver diagnostics of
/// this prediction pass (`W_ODE_SOLVER_DIAGNOSTICS`). Empty for a clean prediction on a
/// well-formed model.
///
/// It is **not** everything `fit()` reports, and the difference is principled rather than a
/// subset: findings whose subject is the *fit* — the estimator/optimizer option warnings, the
/// packed-start rails, the covariance and shrinkage notes — stay out, because no optimizer is
/// running here and this function takes no `FitOptions` to report them against. The exact list
/// and the reasoning live on `api::postfit::non_fit_diagnostics` (internal; an intra-doc link
/// would not resolve from public docs), and `docs/warnings.qmd#non-fit-bundle` is the
/// user-facing version.
///
/// `#[non_exhaustive]` from the start (contrast `OdeSolverStats`, #1302): a structured-entry
/// field alongside `warnings` is then an additive change rather than a breaking one.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PredictionOutput {
    /// One row per (subject, Gaussian observation), in population and observation order.
    pub results: Vec<PredictionResult>,
    /// Non-fatal model/data and ODE-solver findings for this pass. See the struct docs for
    /// what the bundle contains.
    pub warnings: Vec<String>,
}

/// [`predict`] with the diagnostics attached (#1280 / #1304) — the predict-side twin of
/// [`crate::simulate_with_options_diag`].
///
/// Before this, every model/data warning ferx computes reached exactly two entry points
/// (`fit()` and `ferx check`), and every ODE solver diagnostic reached one (`fit()`). A model
/// `fit()` refuses to stay quiet about — an `SS=1` dose on an `[odes]` right-hand side reading
/// `TAFD`, say — came back through `predict()` as a column of `NaN` with nothing said, and a
/// segment that exhausted `ode_max_steps` came back as a column of *identical* finite numbers
/// (#959), which is worse: it is plottable.
///
/// Returns exactly what [`predict`] returns in `results`; the rows are unchanged and no
/// prediction moves by a ULP. The scope that collects the solver counters is opened per subject
/// task and only on a model that integrates something, so a closed-form model takes the
/// identical path it did before.
///
/// # Errors
///
/// A model/data precondition failure is an `Err` carrying the bare check message —
/// byte-identical to what `fit()` returns for the same input (#898): a dose the model cannot
/// route or honour, a covariate the data does not carry, an unrouted non-Gaussian endpoint, an
/// unbound `[covariate_model]`, an unsupported absorption / readout / survival combination.
/// [`predict`] re-raises that same text as a panic, having no channel to return it on. Adding
/// an eleventh check is explicitly *not* how a warning-severity finding reaches `predict()`;
/// that is what `warnings` is for.
pub fn predict_diag(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Result<PredictionOutput, String> {
    // `predict()` runs no data-check (unlike `fit()`); guard the one
    // model-aware dose precondition so a modeled-`RATE` dose can't reach the
    // predictor unresolved (silent-wrong analytical / `.expect` panic). #324.
    first_error(&check_modeled_dose_rates(model, population))?;
    // Every identifier the parser could not bind resolves as a covariate, and a
    // covariate absent from the data reads as 0.0 — so an undefined name anywhere in
    // the model (notably `[scaling]`, #1028) silently collapsed the prediction. `fit()`
    // and `simulate()` already refuse this; match them here.
    first_error(&check_covariates(model, population))?;
    // A population read by the model-blind `read_nonmem_csv` carries a declared
    // endpoint's rows as Gaussian observations, and this function would return a
    // *concentration* for the event row (#1199). `fit()` Err's on the same signature.
    // (`false`: the no-records half of the check is fit-only.)
    first_error(&check_endpoint_routing(model, population, false))?;
    // …and that no `[covariate_model]` relation is still waiting on the
    // data-derived statistics that build it (#1111): an unresolved relation
    // simply is not in the compiled expression, and a dropped covariate effect
    // is invisible in a prediction.
    crate::api::assert_covariate_model_bound(model)?;
    // …and that every dose names a compartment the analytical engine can route
    // it into, so an unroutable infusion errors here with subject/time context
    // instead of panicking deep inside the event-driven walk (#375).
    first_error(&check_dose_compartments(model, population))?;
    check_absorption_closed_form_support(model, population).map_or(Ok(()), Err)?;
    check_absorption_flip_flop_no_twin(model, population, &params.theta).map_or(Ok(()), Err)?;
    // A time-varying covariate on a survival hazard would be silently frozen — refuse
    // rather than return a subtly wrong prediction / simulation (#741; as fit() does).
    #[cfg(feature = "survival")]
    check_survival_tv_covariates(model, population).map_or(Ok(()), Err)?;
    check_analytic_readout_support(model, population).map_or(Ok(()), Err)?;
    first_error(&check_absorption_dosing(model, population))?;
    // CTMM (#759) has no prediction path: its records live in `obs_records`, which the
    // Gaussian loop below never visits, so a CTMM-only model would silently return an
    // empty vec rather than an occupancy π(t). `simulate()`'s twin assert already
    // *claims* to cover predict() — it does not, because it sits in the simulate
    // chokepoint — so state the contract here too. Occupancy prediction is #820.
    // Fail loud only when the call would otherwise return an *empty* vec because CTMM is
    // the only thing to predict. The precise test is "are there continuous observations to
    // predict at all" — i.e. does any subject have a non-empty `obs_times` grid — not
    // "is sigma non-empty" (a CTMM-only model may still declare an `[error_model]`, so a
    // non-empty sigma does not imply continuous rows) and not "is there a Gaussian
    // endpoint" (`EndpointLikelihood::Gaussian` is never inserted into `endpoints` for a
    // plain PK model, so that check would reject every healthy mixed PK + CTMM model). A
    // mixed model with continuous data passes and returns its Gaussian rows; the CTMM rows
    // are simply absent, exactly as a binary endpoint's are (occupancy prediction is #820).
    #[cfg(feature = "markov")]
    if model.has_ctmm() && population.subjects.iter().all(|s| s.obs_times.is_empty()) {
        return Err(
            "predict() does not support a [markov_model] (CTMM) endpoint yet, and this population \
             has no continuous observations either — so the call would return an empty vec rather \
             than an occupancy π(t). State-occupancy prediction is a later slice (#820)."
                .to_string(),
        );
    }

    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    // Whether there is anything for a `SolverStatsScope` to record. Entering a scope
    // *activates* per-segment recording, so a closed-form model must not open one — it has no
    // segments, and the tee would be pure overhead on the path that needs it least.
    let collect_solver_stats = super::integrates_odes(model);
    // One subject's predictions plus whatever its integrations deposited. The scope is
    // **per subject task**, not one around the whole pass: it is thread-local, so a single
    // scope opened on the calling thread would see only the subjects rayon happened to run
    // there and report clean counters for the rest — the same trap `compute_subject_results`
    // documents. Declared before the work so it is still open while that work runs, and read
    // before it drops.
    let predict_one = |subject: &Subject| -> (Vec<f64>, crate::ode::OdeSolverStats) {
        let scope = collect_solver_stats.then(crate::ode::solver::SolverStatsScope::enter);
        let preds = pk::compute_predictions_with_tv(model, subject, &params.theta, &zero_eta);
        let stats = scope.as_ref().map(|s| s.collected()).unwrap_or_default();
        (preds, stats)
    };
    // Merged **in subject order**. The merge is over `usize` counters so the order does not
    // change the sums; it is fixed anyway, so a future non-additive field cannot become
    // worker-count dependent without this comment being wrong.
    let assemble = |per_subject: Vec<(Vec<f64>, crate::ode::OdeSolverStats)>| {
        let mut results = Vec::with_capacity(population.n_obs());
        let mut stats = crate::ode::OdeSolverStats::default();
        for (subject, (preds, subject_stats)) in population.subjects.iter().zip(per_subject) {
            stats.merge(&subject_stats);
            results.extend(preds.into_iter().enumerate().map(|(j, pred)| {
                PredictionResult {
                    id: subject.id.clone(),
                    // Raw data TIME (matches sdtab / input); `obs_times` may be the
                    // internal shifted clock for stacked reset occasions.
                    time: subject
                        .obs_raw_times
                        .get(j)
                        .copied()
                        .unwrap_or(subject.obs_times[j]),
                    pred,
                }
            }));
        }
        (results, stats)
    };
    let predict_subjects = || assemble(population.subjects.par_iter().map(predict_one).collect());
    let predict_subjects_serial =
        || assemble(population.subjects.iter().map(predict_one).collect());

    // A caller already running on a Rayon worker owns the thread budget (for
    // example `PoolPlan` around many predictions), so use that enclosing pool.
    // Standalone calls use ferx's persistent, capped, large-stack pool rather
    // than Rayon's process-global pool or a fresh pool per call.
    let (results, stats) = if rayon::current_thread_index().is_some() {
        if rayon::current_num_threads() == 1 {
            predict_subjects_serial()
        } else {
            predict_subjects()
        }
    } else {
        match super::pool::default_fit_pool() {
            Some(pool) if pool.current_num_threads() > 1 => pool.install(predict_subjects),
            Some(pool) => pool.install(predict_subjects_serial),
            None if rayon::current_num_threads() > 1 => predict_subjects(),
            None => predict_subjects_serial(),
        }
    };
    Ok(PredictionOutput {
        results,
        warnings: super::non_fit_diagnostics(
            model,
            population,
            params,
            &stats,
            super::SolverStatsPhase::Predict,
        ),
    })
}

/// A single prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,
}

/// Category probabilities for every binary-endpoint record in the population
/// (#760 Slice 1b) — the categorical analogue of [`predict_survival`].
///
/// [`predict`] cannot serve this: its [`PredictionResult`] carries a single `f64`,
/// and a categorical prediction is a probability vector (§8.8.1). It is a separate
/// entry point rather than a change to `predict`'s return type so the existing
/// Gaussian signature — which the R wrapper binds — stays untouched.
///
/// Predictions are at `η = 0` (the population-typical subject), matching [`predict`]'s
/// own convention; the EBE-conditioned per-subject values are an sdtab concern.
/// Returns an empty vec for a model with no `Binary` endpoint.
///
/// # Errors
///
/// A time-varying covariate on the linear predictor, or a population loaded without endpoint
/// routing, is an `Err` carrying the text `fit()` returns for the same input (#898).
#[cfg(feature = "survival")]
pub fn predict_categorical(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Result<Vec<EndpointPredictionResult>, String> {
    // Same guard `predict()` and `fit()` apply: a time-varying covariate on the linear
    // predictor would be silently frozen at its baseline value, since `LinearPredictorFn`
    // takes no time argument (#741). Without this, `predict_categorical` was the one
    // public entry point that returned quietly-wrong probabilities.
    check_survival_tv_covariates(model, population).map_or(Ok(()), Err)?;
    // And the routing precondition `predict()` applies (#1199): `predict_binary` walks
    // `obs_records`, so a population read model-blind — its binary rows in the
    // Gaussian grid — would come back empty, indistinguishable from a model with no
    // binary endpoint.
    first_error(&check_endpoint_routing(model, population, false))?;
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();
    for subject in &population.subjects {
        crate::categorical::predict_binary(model, subject, &params.theta, &zero_eta, &mut results);
    }
    Ok(results)
}

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
    /// Mean survival time `E[T]` = ∫₀^∞ S(t) dt; analytic for Exponential,
    /// numerical midpoint rule (2 000 steps) for Weibull and Gompertz.
    pub mean_survival: f64,
}

/// Linear-interpolated median survival time from a cumulative-hazard grid: the time
/// where `H(t) = ln 2` (i.e. `S(t) = 0.5`). Used for ODE-accumulated hazards, whose
/// median has no closed form. Returns NaN if the grid never reaches `ln 2`.
#[cfg(feature = "survival")]
pub(crate) fn grid_median_from_cumhaz(time_grid: &[f64], cum_haz: &[f64]) -> f64 {
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
/// `Σ_k F_k(t) + S_all(t) = 1` holds at every grid point (see `cif_curves`).
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
///
/// # Errors
///
/// A time-varying covariate on a hazard, or a dose into a compartment the model cannot
/// deliver into, is an `Err` carrying the text `fit()` returns for the same input (#898).
#[cfg(feature = "survival")]
pub fn predict_survival(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    time_grid: &[f64],
) -> Result<Vec<SurvivalPredictionResult>, String> {
    // Deliberately no `check_absorption_flip_flop_no_twin` guard here (unlike
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
    // silently applied at its baseline, so refuse instead (#741).
    check_survival_tv_covariates(model, population).map_or(Ok(()), Err)?;

    // A joint PK-TTE hazard reads a PK prediction, so an unroutable dose silently
    // changes the exposure the hazard sees. This entry point was the one member of
    // the `predict`/`simulate` family missing the guard (#899); it is a no-op for a
    // pure-TTE model, where nothing asks the PK predictor for a value.
    first_error(&check_dose_compartments(model, population))?;

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

    Ok(results)
}
