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

/// Build a per-observation HashMap mapping `model.indiv_param_names` to their
/// values from `pk`. Individual parameters the parser synthesized for a direct-θ/η
/// Form-C readout (`__ferx_ro_*`, #486) are internal — they are skipped so they never
/// surface as a user-facing EBE / sdtab column.
pub(crate) fn build_indiv_map(
    pk: &PkParams,
    names: &[String],
    pk_indices: &[usize],
) -> HashMap<String, f64> {
    names
        .iter()
        .zip(pk_indices.iter())
        .filter(|(name, _)| !crate::parser::model_parser::is_synthetic_readout_param(name))
        .map(|(name, &idx)| (name.clone(), pk.values[idx]))
        .collect()
}

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
/// `mixest` (#985) is the fitted per-subject mixture class (0-based); `None` for
/// non-mixture fits. `[derived]` / `[output]` expressions and `pk_param_fn` can
/// branch on `MIXNUM`, so each subject's columns are evaluated under its own class
/// guard rather than the class-1 default.
pub(crate) fn compute_extra_output_columns(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    kappas_per_subject: &[Vec<DVector<f64>>],
    subjects: &mut [SubjectResult],
    mixest: Option<&[usize]>,
) {
    use crate::types::{AggFunction, DerivedContext, DerivedKind, IntegralStep, IntegralWindow};

    let derived_names: Vec<&str> = model
        .derived_exprs
        .iter()
        .map(|s| s.name.as_str())
        .collect();

    for (si, sr) in subjects.iter_mut().enumerate() {
        let _mix_guard = mixest
            .and_then(|m| m.get(si))
            .map(|&c| crate::parser::model_parser::MixtureClassGuard::enter(c + 1));
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
                            .lagtime(subject.doses[d].cmt_raw(), &pk_d.values),
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
                                // NaN — the same convention as IOV/TV/reset above.
                                //
                                // **Deliberately the narrow predicate.** The condition
                                // this arm needs is "the PK *snapshot* is not valid
                                // across the grid", which is a property of
                                // `pk_param_fn` — exactly what the narrow predicate
                                // asks. An `[odes]` RHS that reads `TAD`/`TAFD`/`T`/
                                // `TIME` does *not* make `pk_param_fn` time-dependent:
                                // the clock it reads is the solver's, and the dense
                                // driver sets its own per-segment `TAD` anchor
                                // (`integrate_segment`, with #1073's finite
                                // first-arrival fallback). For such a model the `t=0`
                                // snapshot is exact and the dense arm below returns a
                                // correct number, so widening to
                                // `pk::model_uses_time_anywhere` here would replace it
                                // with an all-NaN column.
                                //
                                // Nor is `compute_predictions_with_states` a precedent
                                // for widening: its ODE branch has two arms and *both*
                                // return populated states — `uses_time` routes to
                                // `ode_predictions_event_driven_with_states`, which
                                // still computes states via `ode_dense_solve_states`.
                                // The `vec![]` convention lives only in its analytical
                                // branch. Widening here would make a `[derived]`
                                // integral NaN while the adjacent per-obs
                                // `compartments[i]` column stayed finite.
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
                            } else if model.is_algebraic() {
                                // A compartment-free model (#811) has no compartments to
                                // report on the grid, and its `pk_model` is a placeholder the
                                // superposition below would read as a real one — returning
                                // all-zero one-compartment amounts as if they were this
                                // model's state. Empty → NaN, the convention every other
                                // out-of-scope case here uses. Mirrors the per-obs branch in
                                // `compute_predictions_with_states`.
                                vec![]
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
                            } else if crate::pk::dose_needs_event_walk(model.pk_model, subject) {
                                // Analytical model + a dose the superposition state helper
                                // cannot place: a zero-order input into the oral depot
                                // (#400), any bolus outside compartment 1, or any infusion
                                // outside central (#375). `single_dose_states` never reads
                                // `dose.cmt` — it picks the closed form from the *model* —
                                // so it would return silently-wrong finite amounts (measured
                                // 38% low on a `two_cpt_iv` peripheral-bolus AUC against a
                                // 1e-12 ODE twin). Return empty so every grid point evaluates
                                // to NaN, matching the per-obs path in
                                // compute_predictions_with_states and the
                                // W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL warning, which
                                // promises exactly that. Must stay the *same* predicate as
                                // the per-obs path: when this branch used the narrower
                                // `has_oral_depot_infusion` the two disagreed, and a
                                // `[derived]` grid integral returned a confident wrong number
                                // while the adjacent per-obs column was NaN.
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
