use crate::pk;
use crate::stats::residual_error::compute_r_matrix_with_correlations;
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Route predictions through analytical PK or ODE solver depending on model,
/// honouring per-event PK parameters when the subject has time-varying
/// covariates. The TV-aware dispatcher in `pk::compute_predictions_with_tv`
/// handles the analytical / ODE / event-driven branching.
///
/// This is the canonical predictions entry point for FOCE/FOCEI inner-loop
/// objectives. Callers must pass the same `(theta, eta)` they use elsewhere
/// in the NLL — `pk_param_fn` is invoked internally once per event (TV path)
/// or once per subject (no-TV path).
///
/// Allocates a fresh `EventPkParams` scratch on each call. Hot loops should
/// use [`model_predictions_into`] with a reused scratch buffer instead.
#[inline]
fn model_predictions(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    pk::compute_predictions_with_tv(model, subject, theta, eta)
}

/// Caller-owned-scratch variant of [`model_predictions`] that also
/// accepts an optional pre-built
/// [`pk::event_driven::EventSchedule`]. Used by FOCE inner-loop callers
/// (BFGS line search, post-convergence eval) that build the schedule
/// once per `find_ebe` call and reuse it across many `(theta, eta)`
/// evaluations of the same subject. SAEM and other callers pass `None`
/// — the no-TV fast path doesn't consume the schedule, and the
/// dispatcher falls back to building one on demand on the TV path.
#[inline]
fn model_predictions_into_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> Vec<f64> {
    pk::compute_predictions_with_tv_into_with_schedule(
        model, subject, theta, eta, scratch, schedule,
    )
}

#[inline]
pub(crate) fn m3_logcdf(limit: f64, f: f64, sd: f64, cens: i8) -> f64 {
    let z = if cens < 0 {
        (f - limit) / sd
    } else {
        (limit - f) / sd
    };
    log_normal_cdf(z)
}

/// Compute individual negative log-likelihood for EBE estimation (inner loop objective).
///
/// NLL(eta | subject) = 0.5 * [eta'*Omega_inv*eta + log|Omega|
///                             + sum_j( term_j )]
/// where term_j is:
///   - `(y_j - f_j)² / V_j + log(V_j)` for quantified observations, or
///   - `-2·log Φ((LLOQ_j - f_j)/√V_j)` for M3 left-censored rows (CENS=1), or
///   - `-2·log Φ((f_j - ULOQ_j)/√V_j)` for M3 right-censored rows (CENS=-1).
pub fn individual_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
) -> f64 {
    // Allocate-on-each-call wrapper — see `individual_nll_into` for
    // the scratch-aware version used by SAEM's MH loop.
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    individual_nll_into(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        &mut scratch,
    )
}

/// Same as [`individual_nll`] but uses a caller-owned scratch buffer.
/// The hot-path entry point for SAEM's MH proposals: a single buffer
/// allocated outside the per-subject MH loop is reused across all
/// proposed `eta`s, eliminating the per-call `Vec<PkParams>` churn
/// that previously dominated SAEM allocator pressure on TV-cov data.
pub fn individual_nll_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    scratch: &mut pk::EventPkParams,
) -> f64 {
    individual_nll_into_with_schedule(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        scratch,
        None,
    )
}

/// TTE negative log-likelihood for an ODE-accumulated hazard (joint PK-TTE, #564).
///
/// Solves the subject's augmented ODE at `(theta, eta)` and reads the cumulative
/// hazard `H(t) = u[chz_state]` (and, for exact events, the instantaneous hazard
/// `h(t) = u̇[chz_state]`) at the event / censor / entry / interval times, then feeds
/// them to the shared [`crate::survival::tte_nll_from_curves`] per-record likelihood.
/// Re-solves the ODE on every call, so the inner EBE search and the FOCEI FD-Hessian
/// (which perturbs η) each get a consistent, freshly-integrated hazard.
#[cfg(feature = "survival")]
fn tte_ode_nll(
    model: &CompiledModel,
    subject: &Subject,
    chz_state: usize,
    records: &[ObsRecord],
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    // Distinct times where H(t) — and h(t) for exact events — is needed.
    let mut times: Vec<f64> = Vec::new();
    for r in records {
        let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            ..
        } = r;
        if *entry_time > 0.0 {
            times.push(*entry_time);
        }
        match event_type {
            EventType::IntervalCensored { left, right } => {
                times.push(*left);
                times.push(*right);
            }
            _ => times.push(*time),
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("TTE event times are finite"));
    times.dedup();

    // Solve the augmented ODE once and read H/h at those times — shared with
    // predict_survival via `ode_cumhaz_hazard` (a missing ODE or a short solve yields
    // NaN curves, which `tte_nll_from_curves` maps to the 1e20 sentinel). The closures
    // look up by the same f64 values that populated `times`, so the search is exact.
    let (cum, haz) =
        crate::survival::ode_cumhaz_hazard(model, subject, chz_state, theta, eta, &times);
    let cumhaz_at = |t: f64| -> f64 {
        times
            .binary_search_by(|x| x.partial_cmp(&t).unwrap())
            .map_or(f64::NAN, |i| cum[i])
    };
    let hazard_at = |t: f64| -> f64 {
        times
            .binary_search_by(|x| x.partial_cmp(&t).unwrap())
            .map_or(f64::NAN, |i| haz[i])
    };

    crate::survival::tte_nll_from_curves(records, cumhaz_at, hazard_at)
}

/// Dispatch a TTE endpoint's per-subject NLL on its hazard representation: the
/// closed-form [`crate::survival::tte_data_term`] for `Analytic`, or the
/// ODE-accumulated [`tte_ode_nll`] for joint PK-TTE.
#[cfg(feature = "survival")]
fn tte_endpoint_nll(
    model: &CompiledModel,
    subject: &Subject,
    hazard: &HazardSpec,
    records: &[ObsRecord],
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    match hazard {
        HazardSpec::Analytic { .. } => {
            crate::survival::tte_data_term(records, hazard, theta, eta, &subject.covariates)
        }
        HazardSpec::OdeAccumulated { chz_state } => {
            tte_ode_nll(model, subject, *chz_state, records, theta, eta)
        }
    }
}

/// Hot-path variant that additionally threads through a pre-built
/// [`pk::event_driven::EventSchedule`]. The FOCE inner-loop obj closure
/// and Jacobian build the schedule once per `find_ebe` call and reuse
/// it across all BFGS iterations.
pub fn individual_nll_into_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    // Ω⁻¹ and log|Ω| are pre-computed in `OmegaMatrix::from_matrix_*`.
    // Hot-path users (FOCE inner BFGS, SAEM MH) call this 100s–1000s of
    // times per subject per outer iter — recomputing Cholesky+inverse
    // here used to dominate small-omega problems.
    if !omega.log_det.is_finite() {
        return 1e20;
    }
    let omega_inv = &omega.inv;
    let log_det_omega = omega.log_det;

    // Eta prior: eta' * Omega_inv * eta
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(omega_inv * &eta_vec));

    // Compute individual predictions using the caller's scratch buffer
    // for per-event PK params (only consumed on the TV-cov path; ignored
    // on the no-TV fast path).
    let preds = model_predictions_into_with_schedule(model, subject, theta, eta, scratch, schedule);
    // For SDE models, compute per-observation EKF process-noise variance and
    // add it to the residual variance to form V_total.
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta, sigma_values)
    } else {
        Vec::new()
    };
    // IIV on residual error: per-subject scale on the residual *variance*
    // (`EPS·EXP(ETA)` → V·exp(2·η_ruv)). 1.0 when no `iiv_on_ruv` is set.
    // Does not touch FREM covariate pseudo-observations (handled below before
    // this factor is applied) or the EKF process noise `p_obs`.
    let ruv_scale = model.residual_var_scale(eta);
    // Per-observation custom residual magnitude (#484), evaluated once here (it
    // is η-independent) and shared with the diagonal and dense paths so the EBE
    // objective matches the outer OFV's variance.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let mut data_ll = 0.0;
    let has_censored_m3 =
        matches!(model.bloq_method, BloqMethod::M3) && subject.has_censored_observation();
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if !model.residual_correlations.is_empty() && !has_censored_m3 && !has_frem_rows {
        match dense_residual_data_term(
            model,
            subject,
            &preds,
            sigma_values,
            ruv_scale,
            &p_obs,
            ruv_mult.as_deref(),
        ) {
            Some(term) => data_ll += term,
            None => return 1e20,
        }
    } else {
        for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            // FREM dispatch: covariate pseudo-observations use theta+eta as
            // prediction and a near-zero additive sigma.
            let fremtype_val = subject.fremtype.get(j).copied().unwrap_or(0);
            if fremtype_val > 0 {
                if let Some(ref fc) = model.frem_config {
                    if let Some(&(theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&fremtype_val) {
                        let frem_pred = theta[theta_idx] + eta[eta_idx];
                        let frem_sigma = sigma_values[fc.covariate_sigma_index];
                        let frem_v = (frem_sigma * frem_sigma).max(1e-12);
                        let resid = y - frem_pred;
                        data_ll += resid * resid / frem_v + frem_v.ln();
                        continue;
                    }
                }
            }
            let v_resid = model.residual_variance_at_scaled(
                subject.obs_cmts[j],
                f_pred,
                sigma_values,
                ruv_mult.as_ref().map(|m| m[j].as_slice()),
            ) * ruv_scale;
            let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if matches!(model.bloq_method, BloqMethod::M3) && cens != 0 {
                data_ll += -2.0 * m3_logcdf(y, f_pred, v.sqrt(), cens);
            } else {
                let resid = y - f_pred;
                data_ll += resid * resid / v + v.ln();
            }
        }
    }

    // TTE data term: add −log p(events | η, θ) for each TTE endpoint.
    // Only compiled and executed when the `survival` feature is enabled and
    // the subject has non-Gaussian obs_records.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        use crate::types::EndpointLikelihood;
        // Iterate model.endpoints (typically 1–3 entries) rather than scanning
        // obs_records for unique CMTs — avoids the HashSet and one pass over records.
        for (cmt, endpoint) in &model.endpoints {
            if let EndpointLikelihood::Tte { hazard } = endpoint {
                let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned()
                    .collect();
                if records_for_cmt.is_empty() {
                    continue; // subject has no records for this TTE CMT
                }
                // tte_endpoint_nll returns a raw NLL; multiply by 2 to match the
                // Gaussian data_ll convention (everything is halved at the end).
                data_ll +=
                    2.0 * tte_endpoint_nll(model, subject, hazard, &records_for_cmt, theta, eta);
            }
        }
    }

    let nll = 0.5 * (eta_prior + log_det_omega + data_ll);
    // Guard a non-finite prediction the same way we guard a non-finite Ω above:
    // an ODE integration can blow up to NaN/inf when the EBE search pushes eta
    // into an extreme region, which would otherwise poison the inner optimizer
    // (e.g. the Nelder-Mead simplex sort). Return the large finite sentinel so
    // the bad point sorts as worst and gets reflected away. See issue #97.
    if nll.is_finite() {
        nll
    } else {
        1e20
    }
}

/// Dense correlated-residual (block_sigma) Gaussian data term for one subject.
///
/// Builds the residual covariance `R` from [`compute_r_matrix_with_correlations`],
/// scales it by `ruv_scale`, adds the per-observation EKF process-noise `p_obs`
/// to the diagonal (`&[]` on the SAEM / non-SDE paths), then returns the
/// un-halved quadratic form plus log-determinant `rᵀ R⁻¹ r + log|R|`. Returns
/// `None` when `R` is not positive-definite so the caller can emit its `1e20`
/// sentinel.
///
/// Shared by the FOCE/E-step ([`individual_nll_into_with_schedule`]) and the
/// SAEM M-step ([`obs_nll_subject_into`]) so both evaluate an identical
/// conditional likelihood — the two previously carried divergent copies, one of
/// which floored predictions while the other did not, biasing SAEM σ/θ when a
/// prediction reached ≤ 0 (#557).
fn dense_residual_data_term(
    model: &CompiledModel,
    subject: &Subject,
    preds: &[f64],
    sigma_values: &[f64],
    ruv_scale: f64,
    p_obs: &[f64],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Option<f64> {
    let mut r = match ruv_mult {
        Some(mult) => crate::stats::residual_error::compute_r_matrix_with_correlations_scaled(
            &model.error_spec,
            preds,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            &model.residual_correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            &model.error_spec,
            preds,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            &model.residual_correlations,
        ),
    };
    if ruv_scale != 1.0 {
        r *= ruv_scale;
    }
    for (j, v) in p_obs.iter().enumerate() {
        if j < r.nrows() {
            r[(j, j)] += *v;
        }
    }
    let chol = r.cholesky()?;
    let residuals = DVector::from_iterator(
        subject.observations.len(),
        subject
            .observations
            .iter()
            .zip(preds.iter())
            .map(|(&y, &f)| y - f),
    );
    let solved = chol.solve(&residuals);
    Some(residuals.dot(&solved) + chol_log_det(&chol.l()))
}

/// Observation-only NLL for a single subject with ETAs held fixed.
///
/// Returns the data term `−log p(y_i | η, θ, σ)` (no prior, no |Ω| term) — the
/// piece that participates in the SAEM M-step gradient and the IS-LL numerator.
///
/// Under M3, censored rows contribute the matching normal-tail likelihood
/// instead of the Gaussian residual term.
pub(crate) fn obs_nll_subject_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    pk_scratch: &mut pk::EventPkParams,
) -> f64 {
    let preds = pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
    obs_nll_subject_from_preds(model, subject, &preds, theta, sigma_values, eta)
}

/// Observation-only NLL from *precomputed* predictions.
///
/// Identical to [`obs_nll_subject_into`] except the caller supplies `preds`
/// (predictions are independent of `σ`), letting the SAEM M-step σ-gradient FD
/// loop reuse one ODE solve across all σ perturbations instead of re-solving per
/// element (#557). `theta` is only consumed by the TTE hazard term.
#[cfg_attr(not(feature = "survival"), allow(unused_variables))]
pub(crate) fn obs_nll_subject_from_preds(
    model: &CompiledModel,
    subject: &Subject,
    preds: &[f64],
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
) -> f64 {
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    // FREM covariate rows use EPSCOV, not the PK residual error (see
    // build_frem_r_override); FREM covariate rows are never BLOQ.
    let frem_ov =
        build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma_values);
    // IIV on residual error: scale the PK residual variance by exp(2·η_ruv).
    // FREM covariate rows keep their own (unscaled) EPSCOV variance.
    let ruv_scale = model.residual_var_scale(eta);
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let mut nll = 0.0;
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if !model.residual_correlations.is_empty() && !m3 && !has_frem_rows {
        // block_sigma + SDE is rejected up front (E_BLOCK_SIGMA_SDE_UNSUPPORTED)
        // for the SAEM M-step, so no EKF process noise enters here.
        match dense_residual_data_term(
            model,
            subject,
            preds,
            sigma_values,
            ruv_scale,
            &[],
            ruv_mult.as_deref(),
        ) {
            Some(term) => nll += 0.5 * term,
            None => return 1e20,
        }
    } else {
        for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            let frem_var = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
            // FREM covariate pseudo-observations predict a covariate *value* (any
            // real: centered/standardized/log-scale covariates are routinely ≤ 0),
            // not a concentration — do NOT clamp them to 1e-12. Clamping a negative
            // covariate prediction up to 1e-12 fabricates a huge residual and, on the
            // Rao-Blackwellised path, breaks the `obs_nll(η_c=d) ≈ const` assumption
            // (#406). Ordinary PK rows keep the positivity clamp.
            let f = if frem_var.is_some() { f } else { f.max(1e-12) };
            let v = match frem_var {
                Some(vv) => vv.max(1e-12),
                None => (model.residual_variance_at_scaled(
                    subject.obs_cmts[j],
                    f,
                    sigma_values,
                    ruv_mult.as_ref().map(|m| m[j].as_slice()),
                ) * ruv_scale)
                    .max(1e-12),
            };
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if m3 && cens != 0 {
                nll += -m3_logcdf(y, f, v.sqrt(), cens);
            } else {
                nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
            }
        }
    }

    // TTE data term: add −log p(events | η, θ) so the SAEM theta M-step
    // gradient receives TTE hazard contributions, not just Gaussian residuals.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        use crate::types::EndpointLikelihood;
        for (cmt, endpoint) in &model.endpoints {
            if let EndpointLikelihood::Tte { hazard } = endpoint {
                // ObsRecord::Event is the only variant (DiscreteState/Count deferred);
                // the `..` pattern captures all EventType variants (Exact, RightCensored,
                // IntervalCensored), so this filter correctly passes every TTE record type.
                let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned()
                    .collect();
                if records_for_cmt.is_empty() {
                    continue;
                }
                nll += tte_endpoint_nll(model, subject, hazard, &records_for_cmt, theta, eta);
            }
        }
    }

    nll
}

/// Compute per-observation EKF process-noise variance (p_obs) for an SDE model.
///
/// Returns an empty vec when `model.is_sde()` is false — callers should check
/// `model.is_sde()` before calling this to avoid an unnecessary ODE pass.
fn ekf_p_obs(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    sigma_values: &[f64],
) -> Vec<f64> {
    let (start, state_indices) = match model.diffusion_theta_start {
        Some(s) => (s, &model.diffusion_state_indices),
        None => return Vec::new(),
    };
    let ode = match model.ode_spec.as_ref() {
        Some(o) => o,
        None => return Vec::new(),
    };

    // Build current diffusion_var from the live theta slice — this is what
    // changes each outer iteration as the optimizer updates diffusion thetas.
    let mut diffusion_var = vec![0.0f64; ode.n_states];
    for (k, &state_idx) in state_indices.iter().enumerate() {
        let theta_idx = start + k;
        if theta_idx < theta.len() && state_idx < ode.n_states {
            diffusion_var[state_idx] = theta[theta_idx].max(0.0);
        }
    }

    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
    // EKF process-noise variance uses a single error model. This is sound: a
    // per-CMT (multi-endpoint) error model needs a Form C `y[CMT=N]` readout to
    // observe multiple compartments, and the parser rejects Form C on SDE
    // models — so an SDE model is always `ErrorSpec::Single` and the
    // representative `model.error_model` is exact here.
    debug_assert!(
        matches!(model.error_spec, ErrorSpec::Single(_)),
        "EKF path reached with a non-Single error spec (per-CMT + SDE should be unreachable)"
    );
    let error_model = model.error_model;

    // Temporarily shadow ode_spec with updated diffusion_var for this call.
    // We cannot mutate model.ode_spec, so we pass diffusion_var separately
    // via a local OdeSpec-like struct. Since solve_ekf takes rhs + n_states
    // + obs_cmt_idx + diffusion_var as separate args, we call it directly.
    // TODO: unify EKF ipred with likelihood ipred to avoid double ODE evaluation
    let (_, p_obs) = crate::ode::ode_predictions_ekf_with_diffusion(
        ode,
        &pk.values,
        subject,
        &diffusion_var,
        |f_pred| crate::stats::residual_error::residual_variance(error_model, f_pred, sigma_values),
    );
    p_obs
}

/// Log-determinant of Omega via Cholesky: log|Omega| = 2 * sum(log(L_ii))
fn omega_log_det(omega: &OmegaMatrix) -> f64 {
    let n = omega.chol.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = omega.chol[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// FOCE per-subject negative log-likelihood.
///
/// Non-interaction (standard FOCE):
///   NLL_i = 0.5 * [(y - f0)' * R_tilde_inv * (y - f0) + log|R_tilde|]
///   where f0 = f(eta_hat) - H * eta_hat  (linearized population prediction)
///         R_tilde = H * Omega * H' + R(f0)
///
/// When M3 censoring is active and the subject has any CENS!=0 row, we route through
/// the interaction path: mixing a linearized Gaussian term with a non-linearized
/// `log Φ(·)` censored term produces inconsistent OFVs near the LOQ boundary, so we
/// promote the whole subject to FOCEI — which is what NONMEM LAPLACE+M3 does in
/// practice.
///
/// Multiplicative factor on the residual variance from an IIV-on-RUV eta
/// (`Y = IPRED + EPS·EXP(ETA)`): `exp(2·eta[k])` for `Some(k)` in range, else
/// `1.0`. Mirrors [`CompiledModel::residual_var_scale`] for call sites that
/// hold the eta slice and the index but not the `&CompiledModel`.
#[inline]
pub(crate) fn ruv_scale_from(eta: &[f64], residual_error_eta: Option<usize>) -> f64 {
    match residual_error_eta {
        Some(k) => eta.get(k).map(|&e| (2.0 * e).exp()).unwrap_or(1.0),
        None => 1.0,
    }
}

pub fn foce_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    // Individual predictions at eta_hat (per-event PK when subject has TV covariates).
    let ipreds = model_predictions(model, subject, theta, eta_hat.as_slice());

    // Per-observation custom residual magnitude (#484). η-independent, so one
    // evaluation here feeds both the data term and the Laplace curvature term.
    let ruv_mult = model.ruv_obs_mult(subject, theta);

    // For SDE models, inflate R with the EKF process-noise variance.
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat.as_slice(), sigma_values)
    } else {
        Vec::new()
    };

    // FREM R-diagonal override: for FREMTYPE > 0, use covariate sigma^2 instead
    // of the PK error model variance. Built once and passed through; None when
    // no FREM config is active.
    let frem_r_override =
        build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma_values);

    let m3_active =
        matches!(model.bloq_method, BloqMethod::M3) && subject.has_censored_observation();

    // TTE Laplace correction: when the subject has TTE obs_records, we compute
    // the FD Hessian of the TTE data term w.r.t. η and add it to hrh inside
    // the interaction path so log|H̃| includes both Gaussian and TTE curvature.
    // For pure-TTE subjects (no Gaussian obs), the interaction path still runs
    // but h_matrix is empty and hrh comes entirely from the TTE Hessian.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        use crate::survival::{data_term_hessian_fd, shi_step_sizes};
        use crate::types::EndpointLikelihood;

        // Compute TTE data NLL and FD Hessian, summed over all TTE CMTs.
        // Iterate model.endpoints (typically 1–3 entries) rather than scanning
        // obs_records for unique CMTs — avoids the HashSet and one pass over records.
        let n_eta = eta_hat.len();
        let mut tte_nll_at_mode = 0.0_f64;
        let mut tte_h = DMatrix::<f64>::zeros(n_eta, n_eta);

        for (cmt, endpoint) in &model.endpoints {
            if let EndpointLikelihood::Tte { hazard } = endpoint {
                let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned()
                    .collect();
                if records_for_cmt.is_empty() {
                    continue; // subject has no records for this TTE CMT
                }
                // Closure that evaluates the TTE NLL at a given eta vector — re-solving
                // the ODE per perturbed η for the ODE-accumulated (joint PK-TTE) case.
                let tte_fn = |eta_eval: &[f64]| -> f64 {
                    tte_endpoint_nll(model, subject, hazard, &records_for_cmt, theta, eta_eval)
                };
                tte_nll_at_mode += tte_fn(eta_hat.as_slice());
                if n_eta > 0 {
                    let steps = shi_step_sizes(&tte_fn, eta_hat.as_slice());
                    tte_h += data_term_hessian_fd(&tte_fn, eta_hat.as_slice(), &steps);
                }
            }
        }

        return foce_subject_nll_interaction_with_tte(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            model.bloq_method,
            &p_obs,
            tte_nll_at_mode,
            tte_h,
            // Promote to interaction when M3 censoring is active, matching the
            // non-TTE branch below so M3 subjects always get the CᵀC correction.
            interaction || m3_active,
            model.residual_error_eta,
            ruv_mult.as_deref(),
        );
    }

    // M3 no longer force-promotes to interaction: plain FOCE keeps a consistent
    // FOCE (Sheiner–Beal) objective for the whole subject, with the censored rows
    // entering the standard path as `−logΦ` terms (excluded from R̃). FOCEI still
    // takes the interaction path. This matches NONMEM `METHOD=1 LAPLACE` with vs
    // without INTER (FOCE-M3 and FOCEI-M3 are genuinely different optima).
    let _ = m3_active;
    // block_sigma cross-endpoint residual covariance is only carried by the
    // non-interaction (Sheiner–Beal) path via `compute_r_matrix_with_correlations`;
    // the interaction accumulator builds R diagonally and would silently drop the
    // off-diagonal covariance. FOCEI + block_sigma is rejected up front
    // (E_BLOCK_SIGMA_METHOD_UNSUPPORTED), and SAEM — which evaluates this OFV with
    // `interaction = true` for AIC/BIC comparability — has no interaction-aware
    // correlated objective to match, so route any correlated model through the
    // FOCE objective instead of returning a correlation-free OFV (#557).
    if interaction && model.residual_correlations.is_empty() {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            model.bloq_method,
            &p_obs,
            frem_r_override.as_deref(),
            model.residual_error_eta,
            ruv_mult.as_deref(),
        )
    } else {
        // FOCE (no interaction): evaluate the residual variance R at the
        // population prediction f(η=0) — NONMEM's no-interaction semantics —
        // not the SB-linearized f0. f0 = f(η̂) − H·η̂ can extrapolate to
        // near-zero/negative concentrations on a nonlinear (e.g. oral) model,
        // collapsing R(f0)=(f0·σ)² to the floor and making R̃ ill-conditioned;
        // f(η=0) is the physically sensible typical-individual prediction
        // (always ≥0). Skipped for additive error (variance is f-independent,
        // so f0 and f(η=0) give the same R) to keep that path bit-identical.
        let pop_preds: Option<Vec<f64>> = if model.error_spec.has_f_dependent_variance() {
            let zeros = vec![0.0_f64; eta_hat.len()];
            Some(model_predictions(model, subject, theta, &zeros))
        } else {
            None
        };
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            &model.residual_correlations,
            model.bloq_method,
            &p_obs,
            frem_r_override.as_deref(),
            pop_preds.as_deref(),
            ruv_mult.as_deref(),
        )
    }
}

/// Standard FOCE (no interaction). When any CENS rows are present AND
/// `bloq_method == M3`, the dispatcher has already routed to the interaction
/// Build per-observation R-diagonal overrides for FREM covariate pseudo-observations.
/// Returns `None` when FREM is inactive (no config or empty fremtype).
/// Overwrite the residual-variance diagonal at FREM covariate pseudo-observation
/// rows with the per-row overrides built by [`build_frem_r_override`]. `None`
/// entries (ordinary PK observations) are left untouched. Indices past the end
/// of `r_diag` are skipped defensively.
pub fn apply_frem_r_overrides(r_diag: &mut [f64], overrides: &[Option<f64>]) {
    for (j, ov) in overrides.iter().enumerate() {
        if let (Some(v), true) = (ov, j < r_diag.len()) {
            r_diag[j] = *v;
        }
    }
}

pub fn build_frem_r_override(
    frem_config: Option<&FremConfig>,
    fremtype: &[u16],
    sigma_values: &[f64],
) -> Option<Vec<Option<f64>>> {
    let fc = frem_config?;
    if fremtype.is_empty() {
        return None;
    }
    Some(
        fremtype
            .iter()
            .map(|&ft| {
                if ft > 0 && fc.fremtype_to_indices.contains_key(&ft) {
                    let s = sigma_values[fc.covariate_sigma_index];
                    Some(if s * s > 1e-12 { s * s } else { 1e-12 })
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn apply_frem_r_matrix_overrides(r_matrix: &mut DMatrix<f64>, overrides: &[Option<f64>]) {
    for (j, ov) in overrides.iter().enumerate() {
        if let (Some(v), true) = (ov, j < r_matrix.nrows()) {
            for k in 0..r_matrix.ncols() {
                r_matrix[(j, k)] = 0.0;
                r_matrix[(k, j)] = 0.0;
            }
            r_matrix[(j, j)] = *v;
        }
    }
}

/// path — so inside this function the only case we need to handle is
/// `bloq_method == Drop` (treat CENS rows as ordinary obs) or no CENS at all.
pub fn foce_subject_nll_standard(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_correlations: &[ResidualCorrelation],
    bloq_method: BloqMethod,
    p_obs: &[f64],
    frem_r_override: Option<&[Option<f64>]>,
    // When `Some`, evaluate the residual variance R at these predictions
    // instead of the SB-linearized f0. Used to evaluate R at the population
    // prediction f(η=0) — NONMEM's no-interaction semantics — which is always
    // ≥0, avoiding the f0 zero-crossing pathology on oral proportional models.
    r_pred_override: Option<&[f64]>,
    // Per-observation custom residual magnitude (#484); `None` on the legacy
    // path. Scales each observation's sigma loadings in R.
    ruv_mult: Option<&[Vec<f64>]>,
) -> f64 {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat (linearized population prediction)
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R; inflate the diagonal with EKF process-noise variance for SDE models,
    // then overwrite FREM covariate rows with their EPSCOV² overrides. The
    // override must come last so it survives the r_pred_override re-evaluation
    // of R.
    let r_eval: &[f64] = r_pred_override.unwrap_or(&f0);
    let mut r_matrix = match ruv_mult {
        Some(mult) => crate::stats::residual_error::compute_r_matrix_with_correlations_scaled(
            error_spec,
            r_eval,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            residual_correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            error_spec,
            r_eval,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            residual_correlations,
        ),
    };
    for (j, v) in p_obs.iter().enumerate() {
        if j < r_matrix.nrows() {
            r_matrix[(j, j)] += *v;
        }
    }
    if let Some(overrides) = frem_r_override {
        apply_frem_r_matrix_overrides(&mut r_matrix, overrides);
    }
    let r_diag: Vec<f64> = (0..r_matrix.nrows()).map(|j| r_matrix[(j, j)]).collect();

    // M3 BLOQ under FOCE: the censored rows leave the Sheiner–Beal marginal (R̃ and
    // the quadratic form are built over the quantified rows only) and re-enter as
    // `−logΦ((LLOQ − f(η̂))/√R⁰)` — the univariate marginal CDF, with R at the
    // population η (no interaction). Mirrors the FOCEI handling, where censored
    // rows are excluded from H̃ and added to the data term.
    let m3 = matches!(bloq_method, BloqMethod::M3) && subject.has_censored_observation();
    let quant: Vec<usize> = if m3 {
        (0..n_obs)
            .filter(|&j| subject.cens.get(j).copied().unwrap_or(0) == 0)
            .collect()
    } else {
        (0..n_obs).collect()
    };

    let mut bloq_term = 0.0;
    if m3 {
        for j in 0..n_obs {
            if subject.cens.get(j).copied().unwrap_or(0) != 0 {
                let z = (subject.observations[j] - ipreds[j]) / r_diag[j].sqrt();
                bloq_term += -2.0 * log_normal_cdf(z);
            }
        }
    }

    // R_tilde = H_q * Omega * H_q' + R_q over the quantified rows; quad form
    // and log|R̃| over the same submatrix. (Non-M3: quant = all rows, unchanged.)
    let nq = quant.len();
    let h_q = if m3 {
        DMatrix::from_fn(nq, model_n_eta(h_matrix), |i, k| h_matrix[(quant[i], k)])
    } else {
        h_matrix.clone()
    };
    let r_q = if m3 {
        select_symmetric_submatrix(&r_matrix, &quant)
    } else {
        r_matrix
    };
    let r_tilde = compute_r_tilde_with_r(&h_q, &omega.matrix, &r_q);

    let chol = match r_tilde.clone().cholesky() {
        Some(c) => c,
        None => return 1e20,
    };

    // Residuals: y - f0 over the quantified rows.
    let residuals: DVector<f64> =
        DVector::from_iterator(nq, quant.iter().map(|&j| subject.observations[j] - f0[j]));

    let solved = chol.solve(&residuals);
    let quad_form = residuals.dot(&solved);
    let log_det_r = chol_log_det(&chol.l());

    0.5 * (quad_form + log_det_r + bloq_term)
}

/// Number of η columns in the inner-Jacobian `h_matrix`.
#[inline]
fn model_n_eta(h_matrix: &DMatrix<f64>) -> usize {
    h_matrix.ncols()
}

/// FOCEI INTER per-subject −2·log marginal — Almquist 2015 Laplace form.
///
/// Per-subject objective (without the `N·log(2π)` data-side constant that
/// NONMEM and nlmixr2 also drop):
///
/// ```text
///   data_ll(η̂) + η̂'·Ω⁻¹·η̂ + log|Ω| + log|H̃|
/// ```
///
/// where
///   `data_ll(η̂) = Σⱼ [(yⱼ − fⱼ)² / Rⱼ + log Rⱼ]`     (R evaluated at η̂)
///   `H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹`            (Almquist 2015 eq. 15)
///   `a_{j,k} = ∂fⱼ/∂η_k`                              (rows of `h_matrix`)
///   `c̃_{j,k} = (∂Rⱼ/∂η_k) / Rⱼ = (∂Rⱼ/∂fⱼ)·a_{j,k} / Rⱼ`
///                                                     (chain rule;
///                                                      `∂R/∂f` from
///                                                      `ErrorSpec::dvar_df`)
///
/// The `½·c̃'·c̃` piece is the **INTER correction**: it captures the
/// η-dependence of the residual variance in the conditional Hessian. It
/// vanishes for additive (η-independent R) error, in which case `H̃`
/// reduces to the FOCE-non-interaction `a'·diag(1/R)·a + Ω⁻¹`.
///
/// This matches NONMEM's `METHOD=1 INTER` and nlmixr2's `est="focei"` —
/// independently verified on the jasmine peds vanco dataset: at NONMEM's
/// converged params, NM reports OFV 66 539, nlmixr2 66 727, and ferx's
/// Almquist Laplace agrees to within FD-vs-analytical-sensitivity noise.
/// The Python reconstruction of NM's per-subject OBJ from its own (η̂, ETC,
/// IPRED) using this exact formula reproduces NM's reported OFV to within
/// 0.013 out of 66 539 — confirming Almquist 2015 first-order is what
/// NONMEM computes.
///
/// The previous implementation used the Sheiner–Beal linearised marginal
/// `(y − f₀)' R̃⁻¹ (y − f₀) + log|R̃|` with `R̃ = HΩH' + R(η̂)`. For
/// nonlinear PK with INTER, that form diverges from the Laplace value at
/// large |η|, and the outer optimiser can exploit the gap to drive `σ_add`
/// small (the negative-EPS-shrinkage symptom on jasmine peds vanco). See
/// `[[focei-laplace-not-sheiner-beal]]` memory.
///
/// With `bloq_method == M3`, censored observations are dropped from the
/// Gaussian residual sum and the H̃ accumulation, and instead contribute
/// the matching normal-tail likelihood evaluated at η̂.
pub fn foce_subject_nll_interaction(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    bloq_method: BloqMethod,
    p_obs: &[f64],
    frem_r_override: Option<&[Option<f64>]>,
    // IIV-on-RUV eta index, or `None`. The per-subject residual-variance scale
    // `exp(2·η̂_ruv)` and the extra `c̃` column are derived from `eta_hat`.
    residual_error_eta: Option<usize>,
    // Per-observation custom residual magnitude (#484); `None` on legacy path.
    ruv_mult: Option<&[Vec<f64>]>,
) -> f64 {
    let n_eta = eta_hat.len();
    let ruv_scale = ruv_scale_from(eta_hat.as_slice(), residual_error_eta);
    let Some(g) = gaussian_foce_accum(
        subject,
        ipreds,
        h_matrix,
        error_spec,
        sigma_values,
        bloq_method,
        p_obs,
        n_eta,
        frem_r_override,
        residual_error_eta,
        ruv_scale,
        ruv_mult,
    ) else {
        return 1e20;
    };

    // η̂'Ω⁻¹η̂  +  log|Ω|  (both cached on OmegaMatrix).
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat));
    // H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹.  log|H̃| via Cholesky.
    let htilde = g.hrh + 0.5 * g.ctc + &omega.inv;
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => return 1e20,
    };

    0.5 * (g.data_ll + eta_prior + omega.log_det + log_det_htilde + g.bloq_term)
}

/// FOCEI NLL with both Gaussian interaction terms and a TTE Laplace correction.
///
/// Adds the TTE data-term at η̂ to `data_ll` and the FD Hessian of the TTE
/// data term to `hrh` before computing log|H̃|.  The Gaussian path is unchanged.
///
/// `tte_data_nll`  — the pre-computed TTE NLL at η̂ (sum over all TTE endpoints).
///   Scaled by 2 to match the convention that data_ll is halved at the end.
/// `tte_hessian`  — FD Hessian of the *raw* TTE NLL w.r.t. η (un-halved).
///   Added to `hrh` before the `log|H̃|` computation.
/// `interaction`  — when `false` (plain FOCE) the η-dependence of the residual
///   variance is ignored: the `½·CᵀC` interaction term is dropped from `H̃`.
///   For pure-TTE subjects CᵀC is all-zero, so this only matters for mixed
///   PK+TTE models run under FOCE.
#[cfg(feature = "survival")]
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_interaction_with_tte(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    bloq_method: BloqMethod,
    p_obs: &[f64],
    tte_data_nll: f64,                 // sum of raw TTE NLLs at η̂ (one per TTE CMT)
    tte_hessian: DMatrix<f64>,         // FD Hessian of the raw TTE NLL w.r.t. η
    interaction: bool,                 // include the ½·CᵀC interaction term (FOCEI) or not (FOCE)
    residual_error_eta: Option<usize>, // IIV-on-RUV eta index (or None)
    // Per-observation custom residual magnitude (#484); `None` on the legacy
    // path. Scales the Gaussian (PK) rows' variance and f-derivative; the TTE
    // rows carry no residual variance so the multiplier never touches them.
    ruv_mult: Option<&[Vec<f64>]>,
) -> f64 {
    let n_eta = eta_hat.len();
    let ruv_scale = ruv_scale_from(eta_hat.as_slice(), residual_error_eta);
    let Some(g) = gaussian_foce_accum(
        subject,
        ipreds,
        h_matrix,
        error_spec,
        sigma_values,
        bloq_method,
        p_obs,
        n_eta,
        None, // TTE path does not support FREM R-override
        residual_error_eta,
        ruv_scale,
        ruv_mult, // custom residual magnitude (#484) — scales the Gaussian PK rows
    ) else {
        return 1e20;
    };

    // Combine Gaussian and TTE data terms.
    // TTE NLL is scaled by 2 here to match the Gaussian data_ll convention
    // (both are halved at the end via the 0.5 factor).
    let data_ll = g.data_ll + 2.0 * tte_data_nll;
    // Accumulate TTE Hessian into the Gaussian Jacobian outer-product matrix.
    let hrh = g.hrh + tte_hessian;

    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat));
    // FOCEI adds the ½·CᵀC interaction curvature; plain FOCE omits it.
    let htilde = if interaction {
        hrh + 0.5 * g.ctc + &omega.inv
    } else {
        hrh + &omega.inv
    };
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => return 1e20,
    };

    0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde + g.bloq_term)
}

/// Output of [`gaussian_foce_accum`].
struct GaussianFoceTerms {
    /// Σⱼ [rⱼ²/Vⱼ + ln Vⱼ] over quantified observations.
    data_ll: f64,
    /// Σⱼ aⱼ'aⱼ/Vⱼ — Jacobian outer-product / variance (H̃ numerator).
    hrh: DMatrix<f64>,
    /// Σⱼ c̃ⱼ'c̃ⱼ — INTER curvature; multiplied by ½ and added for FOCEI.
    ctc: DMatrix<f64>,
    /// Σⱼ censored normal-tail terms (M3 method).
    bloq_term: f64,
}

/// Shared Gaussian accumulation loop for the FOCE/FOCEI interaction path.
///
/// Computes the per-observation Hessian terms from the Gaussian residuals and
/// their variance derivatives. Returns `None` if any observation variance is
/// non-finite or non-positive (callers should return the 1e20 sentinel).
///
/// Both [`foce_subject_nll_interaction`] and the TTE variant call this helper
/// to eliminate the identical inner loop that previously existed in both.
fn gaussian_foce_accum(
    subject: &Subject,
    ipreds: &[f64],
    h_matrix: &DMatrix<f64>,
    error_spec: &ErrorSpec,
    sigma_values: &[f64],
    bloq_method: BloqMethod,
    p_obs: &[f64],
    n_eta: usize,
    frem_r_override: Option<&[Option<f64>]>,
    // IIV on residual error (`Y = IPRED + EPS·EXP(ETA)`). `residual_error_eta`
    // is the eta index that scales the residual SD; `ruv_scale = exp(2·η̂_ruv)`
    // multiplies R. `(None, 1.0)` reproduces the no-IIV-on-RUV behaviour.
    residual_error_eta: Option<usize>,
    ruv_scale: f64,
    // Per-observation custom residual magnitude (#484); `None` on the legacy
    // path. Scales the variance and its f-derivative per observation.
    ruv_mult: Option<&[Vec<f64>]>,
) -> Option<GaussianFoceTerms> {
    let n_obs = subject.observations.len();
    let mult_row = |j: usize| -> Option<&[f64]> { ruv_mult.map(|m| m[j].as_slice()) };

    // Partition observation indices into quantified vs censored (M3 only).
    let (quant_idx, bloq_idx): (Vec<usize>, Vec<usize>) = (0..n_obs).partition(|&j| {
        !(matches!(bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0)
    });

    // Accumulate data_ll at η̂ and the conditional Hessian pieces over the
    // quantified rows.  For SDE the EKF process-noise variance `p_obs` inflates
    // R additively, treated as η-independent here (EKF-vs-FOCEI cross terms are
    // dropped under Almquist's first-order convention).
    let mut data_ll = 0.0_f64;
    let mut hrh = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut ctc = DMatrix::<f64>::zeros(n_eta, n_eta);
    for &j in &quant_idx {
        let f = ipreds[j];
        // FREM override: use covariate sigma^2 for FREMTYPE > 0 observations.
        // FREM covariate pseudo-observations are NOT scaled by the residual-error
        // eta (it acts on the PK residual only), so apply `ruv_scale` and the
        // residual-eta c̃ column only on ordinary PK rows.
        let frem_ov = frem_r_override.and_then(|o| o.get(j)).and_then(|v| *v);
        let is_pk_row = frem_ov.is_none();
        let v_resid = if let Some(v) = frem_ov {
            v
        } else {
            match mult_row(j) {
                Some(m) => {
                    error_spec.variance_at_scaled(subject.obs_cmts[j], f, sigma_values, &[], m)
                        * ruv_scale
                }
                None => error_spec.variance_at(subject.obs_cmts[j], f, sigma_values) * ruv_scale,
            }
        };
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            return None;
        }
        let r = subject.observations[j] - f;
        data_ll += r * r / v + v.ln();

        // a_j = row j of H (∂f_j/∂η); c̃_{j,k} = (∂R_j/∂η_k)/R_j.
        // For a PK eta: (∂R/∂f)·a / R; scaling R by exp(2η_ruv) multiplies both
        // ∂R/∂f and R, so the factor cancels — hence scale dvar_df too.
        // For the residual eta: ∂R/∂η_ruv = 2·R, so the column is the constant 2.
        let aj = h_matrix.row(j);
        let dvar_df = if is_pk_row {
            match mult_row(j) {
                Some(m) => {
                    error_spec.dvar_df_scaled(subject.obs_cmts[j], f, sigma_values, m) * ruv_scale
                }
                None => error_spec.dvar_df(subject.obs_cmts[j], f, sigma_values) * ruv_scale,
            }
        } else {
            0.0 // FREM rows: additive near-zero sigma, ∂R/∂f = 0
        };
        let c_scale = dvar_df / v;
        let inv_v = 1.0 / v;
        let c_ruv = |k: usize| -> f64 {
            if is_pk_row && Some(k) == residual_error_eta {
                2.0
            } else {
                c_scale * aj[k]
            }
        };
        for a in 0..n_eta {
            let aa = aj[a];
            let ca = c_ruv(a);
            for b in 0..n_eta {
                hrh[(a, b)] += aa * aj[b] * inv_v;
                ctc[(a, b)] += ca * c_ruv(b);
            }
        }
    }

    // Censored contributions at η̂ (ipred-based variance).
    let mut bloq_term = 0.0;
    for &j in &bloq_idx {
        let limit = subject.observations[j];
        let f = ipreds[j];
        let v = match mult_row(j) {
            Some(m) => {
                error_spec.variance_at_scaled(subject.obs_cmts[j], f, sigma_values, &[], m)
                    * ruv_scale
            }
            None => error_spec.variance_at(subject.obs_cmts[j], f, sigma_values) * ruv_scale,
        };
        if !(v.is_finite() && v > 0.0) {
            return None;
        }
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        bloq_term += -2.0 * m3_logcdf(limit, f, v.sqrt(), cens);
    }

    Some(GaussianFoceTerms {
        data_ll,
        hrh,
        ctc,
        bloq_term,
    })
}

/// R_tilde = H * Omega * H' + diag(r_diag)
pub(crate) fn compute_r_tilde(
    h: &DMatrix<f64>,
    omega: &DMatrix<f64>,
    r_diag: &[f64],
) -> DMatrix<f64> {
    let n_obs = h.nrows();
    let h_omega = h * omega;
    let mut r_tilde = &h_omega * h.transpose();
    for j in 0..n_obs {
        r_tilde[(j, j)] += r_diag[j];
    }
    r_tilde
}

/// R_tilde = H * Omega * H' + R for models with a non-diagonal residual
/// covariance matrix.
pub(crate) fn compute_r_tilde_with_r(
    h: &DMatrix<f64>,
    omega: &DMatrix<f64>,
    r: &DMatrix<f64>,
) -> DMatrix<f64> {
    let h_omega = h * omega;
    &h_omega * h.transpose() + r
}

fn select_symmetric_submatrix(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), indices.len(), |row, col| {
        matrix[(indices[row], indices[col])]
    })
}

/// log-determinant from Cholesky factor L: 2 * sum(log(L_ii))
pub(crate) fn chol_log_det(l: &DMatrix<f64>) -> f64 {
    let n = l.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = l[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// IOV-aware FOCE per-subject NLL — a *proper* linearised marginal over the
/// full random-effect vector `b = [η, κ₁, …, κ_K]`.
///
/// The per-occasion κ draws are integrated out by the same Sheiner–Beal
/// marginal that handles the BSV η: we assemble the augmented sensitivity
/// matrix `H_full = [∂f/∂η │ ∂f/∂κ₁ │ … │ ∂f/∂κ_K]` and the block-diagonal
/// prior covariance `Σ_b = blkdiag(Ω_bsv, Ω_iov, …, Ω_iov)` (K copies), then
/// evaluate the ordinary FOCE/FOCEI form `0.5·[(y−f₀)ᵀ R̃⁻¹ (y−f₀) + log|R̃|]`
/// with `R̃ = H_full Σ_b H_fullᵀ + R`.
///
/// Because `∂f/∂κ_k` is non-zero only on occasion-k's observation rows (κ_k
/// enters only that occasion's predictions, under the cross-occasion
/// dose-carryover convention of `individual_nll_iov`), the κ columns are
/// block-structured and the κ blocks of `Σ_b` couple only same-occasion rows
/// — independent occasions stay independent in `R̃`.
///
/// This replaces the earlier shortcut (BSV-only linearisation plus an explicit
/// `0.5·Σ_k[κᵀΩ_iov⁻¹κ + log|Ω_iov|]` MAP penalty). That penalty omitted the
/// κ-block Laplace determinant `log|H_κᵀR⁻¹H_κ + Ω_iov⁻¹|`; in a correct
/// marginal `log|Ω| + log|J|` combine into the bounded `log|R̃/R|`, so dropping
/// `log|J|` left a bare `+0.5·K·log|Ω_iov|` that → −∞ as Ω_iov → 0, leaving
/// `omega_iov` unidentified and the FOCE OFV not comparable to NONMEM / SAEM.
/// See issue #101. With the augmented form, no separate κ prior is added (it
/// is already folded into `R̃`), and the K=0 case reduces exactly to
/// [`foce_subject_nll`].
///
/// `kappas[k]` is the EBE kappa vector for occasion k (same order as
/// `split_obs_by_occasion`).  When `kappas` is empty, falls through to the
/// non-IOV path (no overhead for non-IOV subjects or models).
pub fn foce_subject_nll_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega_bsv: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
    kappas: &[DVector<f64>],
    omega_iov: &OmegaMatrix,
) -> f64 {
    if kappas.is_empty() {
        return foce_subject_nll(
            model,
            subject,
            theta,
            eta_hat,
            h_matrix,
            omega_bsv,
            sigma_values,
            interaction,
        );
    }

    let occ_groups = split_obs_by_occasion(subject);
    let n_obs = subject.obs_times.len();
    let n_eta = eta_hat.len();
    let n_iov = omega_iov.matrix.nrows();
    // Defensive: the EBE pipeline always yields exactly one κ vector per
    // occasion group, each of width n_iov. A mismatch would silently leave the
    // unmatched occasions' ipreds (and H columns) at 0.0 and score the
    // augmented marginal against wrong predictions, so bail with the large
    // finite sentinel — mirroring the guards in `individual_nll_iov`.
    if kappas.len() != occ_groups.len() || kappas.iter().any(|k| k.len() != n_iov) {
        return 1e20;
    }
    let k_occ = occ_groups.len();
    let n_b = n_eta + k_occ * n_iov;

    // ipreds at the joint EBE via the continuous, per-occasion-aware prediction
    // (proper cross-occasion carryover; issue #104), plus the augmented
    // H-matrix. BSV columns come from the passed-in H (FD of the same prediction
    // w.r.t. η, in `compute_jacobian_fd_iov`); the κ columns are FD here.
    // Because κ_k changes occasion-k's clearance, it affects occasion-k's
    // observations AND the carryover into later occasions — so a κ column is
    // dense across rows, exactly what FD of the continuous prediction captures
    // (the old Option-A version wrote κ_k only on occasion-k's rows).
    let kappa_slices: Vec<Vec<f64>> = kappas.iter().map(|k| k.as_slice().to_vec()).collect();
    let ipreds = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kappa_slices);

    let mut h_full = DMatrix::zeros(n_obs, n_b);
    for j in 0..n_obs {
        for c in 0..n_eta {
            h_full[(j, c)] = h_matrix[(j, c)];
        }
    }
    // Reused κ buffer: perturb one element in place and restore it, rather than
    // cloning all occasions' κ twice per FD step.
    let mut kpert = kappa_slices.clone();
    const EPS: f64 = 1e-6;
    for k in 0..k_occ {
        let col_base = n_eta + k * n_iov;
        for ki in 0..n_iov {
            let orig = kpert[k][ki];
            let step = EPS * (1.0 + orig.abs());
            kpert[k][ki] = orig + step;
            let preds_plus = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kpert);
            kpert[k][ki] = orig - step;
            let preds_minus = pk::predict_iov(model, subject, theta, eta_hat.as_slice(), &kpert);
            kpert[k][ki] = orig;
            let inv_2step = 1.0 / (2.0 * step);
            for j in 0..n_obs {
                h_full[(j, col_base + ki)] = (preds_plus[j] - preds_minus[j]) * inv_2step;
            }
        }
    }

    // Joint EBE vector b̂ = [η̂, κ̂₁, …, κ̂_K].
    let mut b_hat = DVector::zeros(n_b);
    for i in 0..n_eta {
        b_hat[i] = eta_hat[i];
    }
    for (k, kap) in kappas.iter().enumerate() {
        for ki in 0..n_iov {
            b_hat[n_eta + k * n_iov + ki] = kap[ki];
        }
    }

    // Block-diagonal prior covariance Σ_b = blkdiag(Ω_bsv, Ω_iov × K).
    // `from_matrix` regularises if a sub-block is not PD, matching the
    // robustness of the non-IOV OmegaMatrix path; the standard/interaction
    // FOCE routines below read only `Σ_b.matrix`.
    let sigma_b_mat = build_block_diag_omega(&omega_bsv.matrix, &omega_iov.matrix, k_occ);
    let sigma_b = OmegaMatrix::from_matrix(sigma_b_mat, Vec::new(), false);

    // The augmented system is now an ordinary FOCE/FOCEI marginal: κ is
    // integrated out through R̃ exactly like η, so no separate κ prior is
    // added (doing so would double-count the random-effect penalty).
    let m3_active =
        matches!(model.bloq_method, BloqMethod::M3) && subject.has_censored_observation();
    let p_obs_iov = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat.as_slice(), sigma_values)
    } else {
        Vec::new()
    };
    // IOV + FREM is unsupported: the augmented b̂ vector and block-diagonal
    // Σ_b are not set up for FREM R-overrides.  Return a sentinel NLL so the
    // optimizer steers away from this region rather than silently ignoring FREM.
    if model.frem_config.is_some() && subject.fremtype.iter().any(|&ft| ft > 0) {
        return 1e18;
    }
    // Per-observation custom residual magnitude (#484); η/κ-independent.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    if interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            model.bloq_method,
            &p_obs_iov,
            None, // IOV + FREM unsupported (guarded above)
            model.residual_error_eta,
            ruv_mult.as_deref(),
        )
    } else {
        // FOCE (no interaction): evaluate R at the population prediction with
        // all random effects zero (η=0, κ=0), matching the non-IOV marginal so
        // the zero-κ / Ω_iov→0 reduction collapses exactly to the BSV marginal.
        // Additive error keeps f0 (bit-identical).
        let pop_preds: Option<Vec<f64>> = if model.error_spec.has_f_dependent_variance() {
            let zeros_eta = vec![0.0_f64; n_eta];
            let zero_kappas: Vec<Vec<f64>> = kappa_slices
                .iter()
                .map(|k| vec![0.0_f64; k.len()])
                .collect();
            Some(pk::predict_iov(
                model,
                subject,
                theta,
                &zeros_eta,
                &zero_kappas,
            ))
        } else {
            None
        };
        foce_subject_nll_standard(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            &model.residual_correlations,
            model.bloq_method,
            &p_obs_iov,
            None,
            pop_preds.as_deref(),
            ruv_mult.as_deref(),
        )
    }
}

/// Population FOCE objective with IOV: sum over all subjects using
/// `foce_subject_nll_iov`.  `kappas_per_subject[i]` holds the per-occasion
/// kappa EBEs for subject i (empty slice = no IOV for that subject).
pub fn foce_population_nll_iov(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    omega_bsv: &OmegaMatrix,
    omega_iov: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let kappas = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };
            foce_subject_nll_iov(
                model,
                subject,
                theta,
                &eta_hats[i],
                &h_matrices[i],
                omega_bsv,
                sigma_values,
                interaction,
                kappas,
                omega_iov,
            )
        })
        .sum::<f64>()
}

/// Population FOCE objective: sum over all subjects
pub fn foce_population_nll(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            foce_subject_nll(
                model,
                subject,
                theta,
                &eta_hats[i],
                &h_matrices[i],
                omega,
                sigma_values,
                interaction,
            )
        })
        .sum::<f64>()
}

/// Compute CWRES (Conditional Weighted Residuals) for a subject.
/// Censored observations get `NaN` since a weighted Gaussian residual is undefined
/// when the observed value is censored.
#[allow(clippy::too_many_arguments)]
pub fn compute_cwres(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_correlations: &[ResidualCorrelation],
    frem_r_override: Option<&[Option<f64>]>,
    // IIV-on-RUV eta index (or None). Scales the residual diagonal `R` by
    // exp(2·η̂_ruv) so CWRES uses the subject's actual residual SD (#409).
    residual_error_eta: Option<usize>,
    // Per-observation custom residual magnitude (#484) matrix, or `None` on the
    // legacy path. Scales each row's residual variance so sdtab CWRES matches
    // the magnitude-aware OFV (the multiplier is f-independent, so the matrix
    // built at `ipred` applies unchanged at the SB-linearized `f0`).
    ruv_mult: Option<&[Vec<f64>]>,
) -> Vec<f64> {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R_tilde
    let mut r_matrix = match ruv_mult {
        Some(mult) => crate::stats::residual_error::compute_r_matrix_with_correlations_scaled(
            error_spec,
            &f0,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            residual_correlations,
            mult,
        ),
        None => compute_r_matrix_with_correlations(
            error_spec,
            &f0,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma_values,
            residual_correlations,
        ),
    };
    if let Some(overrides) = frem_r_override {
        apply_frem_r_matrix_overrides(&mut r_matrix, overrides);
    }
    let ruv_scale = ruv_scale_from(eta_hat.as_slice(), residual_error_eta);
    if ruv_scale != 1.0 {
        if let Some(overrides) = frem_r_override {
            for j in 0..r_matrix.nrows() {
                let j_is_frem = overrides.get(j).and_then(|x| *x).is_some();
                for k in 0..r_matrix.ncols() {
                    let k_is_frem = overrides.get(k).and_then(|x| *x).is_some();
                    if !j_is_frem && !k_is_frem {
                        r_matrix[(j, k)] *= ruv_scale;
                    }
                }
            }
        } else {
            r_matrix *= ruv_scale;
        }
    }
    let r_tilde = compute_r_tilde_with_r(h_matrix, &omega.matrix, &r_matrix);

    // CWRES_j = (y_j - f0_j) / sqrt(R_tilde_jj), or NaN if censored.
    (0..n_obs)
        .map(|j| {
            if subject.cens.get(j).copied().unwrap_or(0) != 0 {
                f64::NAN
            } else {
                let resid = subject.observations[j] - f0[j];
                let var = r_tilde[(j, j)].max(1e-12);
                resid / var.sqrt()
            }
        })
        .collect()
}

/// Group observation indices by occasion (preserving first-seen order of occasions).
/// Returns `Vec<(occ_id, Vec<obs_index>)>` sorted by first appearance of the occasion.
pub fn split_obs_by_occasion(subject: &Subject) -> Vec<(u32, Vec<usize>)> {
    let mut occ_order: Vec<u32> = Vec::new();
    let mut occ_map: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (j, &occ) in subject.occasions.iter().enumerate() {
        if !occ_map.contains_key(&occ) {
            occ_order.push(occ);
            occ_map.insert(occ, Vec::new());
        }
        occ_map.get_mut(&occ).unwrap().push(j);
    }
    occ_order
        .into_iter()
        .map(|occ| (occ, occ_map.remove(&occ).unwrap()))
        .collect()
}

/// Map each occasion id to its group index `k` (the order [`split_obs_by_occasion`]
/// returns). Shared by the closed-form (`build_iov_sources`) and ODE (`run_subject_iov` /
/// `run_subject_iov_eta`) IOV providers so the occasion→group mapping is defined once
/// (#466 review round 2).
pub fn iov_occ_to_k(occ_groups: &[(u32, Vec<usize>)]) -> std::collections::HashMap<u32, usize> {
    occ_groups
        .iter()
        .enumerate()
        .map(|(k, (occ, _))| (*occ, k))
        .collect()
}

/// The combined effect vector `[η_bsv, κ_g]` for occasion group `g`, sliced from a
/// subject's stacked `[η_bsv, κ₁..κ_K]` vector. The **single source** of the κ-axis offset
/// (`base = n_eta + g·n_kappa`) so the closed-form and ODE IOV providers cannot seed κ onto
/// different axes — a change to the stacked layout now touches one place (#466 review
/// round 2).
pub fn iov_combined_effect(
    stacked_eta: &[f64],
    n_eta: usize,
    n_kappa: usize,
    g: usize,
) -> Vec<f64> {
    let mut c = Vec::with_capacity(n_eta + n_kappa);
    c.extend_from_slice(&stacked_eta[..n_eta]);
    let base = n_eta + g * n_kappa;
    c.extend_from_slice(&stacked_eta[base..base + n_kappa]);
    c
}

/// Build a block-diagonal omega from BSV omega and K copies of IOV omega.
/// Used for the extended H-matrix in the FOCE outer loop with IOV.
pub fn build_block_diag_omega(
    omega_bsv: &DMatrix<f64>,
    omega_iov: &DMatrix<f64>,
    n_occasions: usize,
) -> DMatrix<f64> {
    let n_bsv = omega_bsv.nrows();
    let n_iov = omega_iov.nrows();
    let n_total = n_bsv + n_occasions * n_iov;
    let mut m = DMatrix::zeros(n_total, n_total);
    // BSV block
    for i in 0..n_bsv {
        for j in 0..n_bsv {
            m[(i, j)] = omega_bsv[(i, j)];
        }
    }
    // K copies of IOV block
    for k in 0..n_occasions {
        let offset = n_bsv + k * n_iov;
        for i in 0..n_iov {
            for j in 0..n_iov {
                m[(offset + i, offset + j)] = omega_iov[(i, j)];
            }
        }
    }
    m
}

/// IOV-aware individual NLL: uses per-occasion kappas.
///
/// `kappas[k]` is the kappa vector for the k-th unique occasion (in the order
/// returned by `split_obs_by_occasion`).  When `kappas` is empty, falls back
/// to the standard (no-IOV) `individual_nll` path.
///
/// **Cross-occasion dose carryover (issue #104).** Predictions are computed by
/// [`pk::predict_iov`], which builds per-event PK parameters carrying each
/// event's occasion kappa and propagates the compartment amounts continuously
/// across occasion boundaries (via the event-driven solver). A dose given in an
/// earlier occasion therefore decays through a later occasion with the *later*
/// occasion's clearance — matching NONMEM's integration model. This replaced
/// the earlier "Option A" superposition, which scored each occasion against the
/// whole dose history with a single clearance and biased the likelihood on
/// no-washout designs. The FD Jacobian (`compute_jacobian_fd_iov`) and the
/// augmented marginal (`foce_subject_nll_iov`) use the same prediction, so NLL
/// and gradients stay consistent.
pub fn individual_nll_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    omega: &OmegaMatrix,
    omega_iov: Option<&OmegaMatrix>,
    sigma_values: &[f64],
) -> f64 {
    if kappas.is_empty() {
        return individual_nll(model, subject, theta, eta, omega, sigma_values);
    }

    // BSV eta prior
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => return 1e20,
    };
    let log_det_omega = omega_log_det(omega);
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(&omega_inv * &eta_vec));

    // Kappa priors and IOV log-det
    let (iov_inv, log_det_iov) = if let Some(iov) = omega_iov {
        let inv = match iov.matrix.clone().cholesky() {
            Some(chol) => chol.inverse(),
            None => return 1e20,
        };
        (inv, omega_log_det(iov))
    } else {
        (DMatrix::identity(1, 1), 0.0) // unreachable when kappas non-empty
    };

    let mut kappa_prior = 0.0;
    for kap in kappas {
        let kap_vec = DVector::from_column_slice(kap);
        kappa_prior += kap_vec.dot(&(&iov_inv * &kap_vec));
    }
    let k_occasions = kappas.len();

    // Data NLL — single continuous prediction with per-event occasion kappa
    // (proper cross-occasion carryover; issue #104).
    let preds = pk::predict_iov(model, subject, theta, eta, kappas);
    // FREM covariate pseudo-observations use the covariate sigma (EPSCOV), not
    // the PK residual error, so the FREM etas are sampled against the right
    // variance (mirrors the FOCE paths and the non-IOV individual_nll).
    let frem_ov =
        build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma_values);
    // IIV on residual error (#409): η_ruv is a BSV eta, indexed into `eta`.
    let ruv_scale = model.residual_var_scale(eta);
    let mut data_ll = 0.0;
    for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let v = match frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
            Some(vv) => vv,
            None => {
                model.residual_variance_at(subject.obs_cmts[j], f_pred, sigma_values) * ruv_scale
            }
        };
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        if matches!(model.bloq_method, BloqMethod::M3) && cens != 0 {
            data_ll += -2.0 * m3_logcdf(y, f_pred, v.sqrt(), cens);
        } else {
            let resid = y - f_pred;
            data_ll += resid * resid / v + v.ln();
        }
    }

    // TTE data term: same convention as individual_nll_into_with_schedule —
    // multiply by 2.0 so the final 0.5 factor gives a net weight of 1.0×.
    // Kappas are PK-only; the hazard param_fn uses BSV eta, not kappas.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        use crate::survival::tte_data_term;
        use crate::types::EndpointLikelihood;
        for (cmt, endpoint) in &model.endpoints {
            if let EndpointLikelihood::Tte { hazard } = endpoint {
                let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned()
                    .collect();
                if records_for_cmt.is_empty() {
                    continue;
                }
                data_ll +=
                    2.0 * tte_data_term(&records_for_cmt, hazard, theta, eta, &subject.covariates);
            }
        }
    }

    0.5 * (eta_prior + log_det_omega + kappa_prior + (k_occasions as f64) * log_det_iov + data_ll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BloqMethod, DoseEvent, EndpointError, ErrorModel, ErrorSpec, GradientMethod, PkModel,
        PkParams, ResidualCorrelation,
    };
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn make_simple_subject() -> Subject {
        Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            obs_raw_times: Vec::new(),
            observations: vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn make_omega(var: f64) -> OmegaMatrix {
        OmegaMatrix::from_diagonal(&[var], vec!["ETA_CL".into()])
    }

    fn make_model() -> CompiledModel {
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL uses combined eta[0]
                p.values[1] = theta[1]; // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params: crate::types::ModelParameters {
                theta: vec![5.0, 50.0],
                theta_names: vec!["TVCL".into(), "TVV".into()],
                theta_lower: vec![0.01, 1.0],
                theta_upper: vec![100.0, 500.0],
                theta_fixed: vec![false; 2],
                omega: make_omega(0.09),
                omega_fixed: vec![false],
                sigma: crate::types::SigmaVector {
                    values: vec![0.05],
                    names: vec!["PROP_ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
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
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            n_kappa: 0,
            kappa_names: Vec::new(),
            kappa_mu_refs: HashMap::new(),
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
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

    #[test]
    fn obs_nll_subject_into_uses_cross_endpoint_residual_covariance() {
        let mut model = make_model();
        model.error_spec = ErrorSpec::PerCmt(HashMap::from([
            (
                1,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![1],
                },
            ),
            (
                2,
                EndpointError {
                    error_model: ErrorModel::Proportional,
                    sigma_idx: vec![0],
                },
            ),
        ]));
        model.residual_correlations = vec![ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 1,
            rho: 0.5,
        }];

        let mut subj = make_simple_subject();
        subj.obs_times = vec![1.0, 1.0];
        subj.obs_raw_times = Vec::new();
        subj.observations = vec![1.8, 2.4];
        subj.obs_cmts = vec![1, 2];
        subj.cens = vec![0, 0];
        subj.occasions = vec![1, 1];
        subj.fremtype = Vec::new();
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let sigma = vec![0.2, 0.3];
        let mut scratch = pk::EventPkParams::default();

        let dense = obs_nll_subject_into(&model, &subj, &theta, &sigma, &eta, &mut scratch);
        model.residual_correlations.clear();
        let diagonal = obs_nll_subject_into(&model, &subj, &theta, &sigma, &eta, &mut scratch);

        assert!(dense.is_finite());
        assert!(diagonal.is_finite());
        assert!(
            (dense - diagonal).abs() > 1e-6,
            "cross-endpoint residual covariance must contribute to SAEM obs NLL"
        );
    }

    #[test]
    fn m3_logcdf_uses_upper_tail_for_negative_cens() {
        let sd = 2.0;
        let f = 12.0;
        let uloq = 10.0;
        let lloq = 14.0;

        let upper = m3_logcdf(uloq, f, sd, -1);
        let lower = m3_logcdf(lloq, f, sd, 1);

        assert!((upper - log_normal_cdf(1.0)).abs() < 1e-12);
        assert!((lower - log_normal_cdf(1.0)).abs() < 1e-12);

        // NONMEM 7.5.1 anchor in tests/nonmem/right_censored_m3.{ctl,csv,lst}:
        // two identical CENS=-1 rows with z=(12-10)/2=1 give OFV
        // 0.69101514210943182. ferx uses the A&S CDF approximation, so compare
        // within a numerical tolerance rather than bit-for-bit.
        let nonmem_ofv = 0.691_015_142_109_431_8;
        let ferx_ofv = -4.0 * upper;
        assert!((ferx_ofv - nonmem_ofv).abs() < 1e-6);
    }

    #[test]
    fn test_ltbs_individual_nll_matches_additive_on_log_scale() {
        // Under LTBS the inner-loop NLL must score the (already-log-scale)
        // observations against log(prediction) with additive variance σ². This
        // checks the prediction sink's log-wrap flows through `individual_nll`.
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
        model.log_transform = true;

        let theta = vec![5.0, 50.0];
        let eta = vec![0.0]; // eta_prior = 0
        let omega = make_omega(0.09);
        let sigma = vec![0.3]; // additive SD on the log scale

        // Observations on the log scale (what `fit()` produces for case 2).
        let mut subj = make_simple_subject();
        for v in &mut subj.observations {
            *v = v.ln();
        }

        // Manual reference: log(natural prediction), additive variance σ².
        let mut natural_model = make_model();
        natural_model.log_transform = false;
        let natural = pk::compute_predictions_with_tv(&natural_model, &subj, &theta, &eta);
        let var = sigma[0] * sigma[0];
        let mut data_ll = 0.0;
        for (j, &f_nat) in natural.iter().enumerate() {
            let log_f = f_nat.max(pk::LTBS_FLOOR).ln();
            let resid = subj.observations[j] - log_f;
            data_ll += resid * resid / var + var.ln();
        }
        let expected = 0.5 * (omega.log_det + data_ll);

        let got = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        approx::assert_relative_eq!(got, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_split_obs_by_occasion_two_occ() {
        let subj = make_simple_subject();
        let groups = split_obs_by_occasion(&subj);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, 1);
        assert_eq!(groups[0].1, vec![0, 1, 2]);
        assert_eq!(groups[1].0, 2);
        assert_eq!(groups[1].1, vec![3, 4, 5]);
    }

    #[test]
    fn test_split_obs_by_occasion_empty() {
        let mut subj = make_simple_subject();
        subj.occasions = Vec::new();
        subj.obs_times = Vec::new();
        subj.observations = Vec::new();
        subj.obs_cmts = Vec::new();
        subj.cens = Vec::new();
        let groups = split_obs_by_occasion(&subj);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_individual_nll_iov_no_kappa_same_as_base() {
        let model = make_model();
        let subj = make_simple_subject();
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let omega = make_omega(0.09);
        let sigma = vec![0.05];

        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        let iov = individual_nll_iov(&model, &subj, &theta, &eta, &[], &omega, None, &sigma);
        approx::assert_relative_eq!(base, iov, epsilon = 1e-10);
    }

    #[test]
    fn test_individual_nll_finite_sentinel_on_nonfinite_eta() {
        // Regression for issue #97: when the EBE search wanders into an extreme
        // region (here a non-finite eta, standing in for an ODE blow-up), the
        // NLL must return the large finite sentinel, never a non-finite value.
        // A NaN/inf leaking out poisons the inner Nelder-Mead simplex sort and
        // aborts the fit; this guard mirrors the existing non-finite Ω guard.
        //
        // Note the analytical PK path scrubs NaN via `.max()`/`.min()`
        // (`NaN.max(1e-30) == 1e-30`), so the non-finiteness here enters through
        // the eta-prior term `η'Ω⁻¹η`, which is exactly the quantity the inner
        // optimizer drives.
        let model = make_model();
        let subj = make_simple_subject();
        let omega = make_omega(0.09);
        let nll = individual_nll(
            &model,
            &subj,
            &[5.0, 50.0],
            &[f64::INFINITY],
            &omega,
            &[0.05],
        );
        assert!(nll.is_finite(), "NLL must stay finite, got {nll}");
        assert_eq!(
            nll, 1e20,
            "a non-finite NLL should map to the 1e20 sentinel"
        );
    }

    #[test]
    fn test_individual_nll_iov_with_kappa_adds_prior() {
        let model = make_model();
        let subj = make_simple_subject();
        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let omega = make_omega(0.09);
        let omega_iov = make_omega(0.01);
        let sigma = vec![0.05];

        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        // Non-zero kappas add a kappa prior ≥ 0, so IOV NLL ≥ base NLL.
        let kappas = vec![vec![0.1], vec![-0.1]];
        let iov = individual_nll_iov(
            &model,
            &subj,
            &theta,
            &eta,
            &kappas,
            &omega,
            Some(&omega_iov),
            &sigma,
        );
        // Kappa prior is positive → IOV NLL should differ from base
        assert!(
            (iov - base).abs() > 1e-6,
            "IOV NLL={}, base NLL={}",
            iov,
            base
        );
    }

    /// A model whose CL depends on both the BSV eta and the per-occasion
    /// kappa (`combined[1]`), so the kappa block genuinely enters the
    /// augmented R̃. The kappa read is defensive so the BSV-only
    /// `foce_subject_nll` path (which passes a length-1 eta) doesn't panic.
    fn make_iov_kappa_model() -> CompiledModel {
        let mut model = make_model();
        model.pk_param_fn = Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
            let mut p = PkParams::default();
            let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
            p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL
            p.values[1] = theta[1]; // V
            p
        });
        model
    }

    /// Issue #101: `foce_subject_nll_iov` must be a proper augmented marginal,
    /// not a BSV FOCE term plus an additive kappa MAP penalty.
    #[test]
    fn test_foce_subject_nll_iov_is_proper_marginal() {
        let model = make_iov_kappa_model();
        let subj = make_simple_subject(); // occasions [1,1,1,2,2,2]
        let theta = vec![5.0, 50.0];
        let eta_hat = DVector::from_vec(vec![0.1]);
        let omega_bsv = make_omega(0.09);
        let sigma = vec![0.05];

        // BSV-only H via central FD of predictions w.r.t. eta[0] at kappa = 0.
        let n_obs = subj.observations.len();
        let mut h_bsv = DMatrix::zeros(n_obs, 1);
        let eps = 1e-6;
        let pp = model_predictions(&model, &subj, &theta, &[0.1 + eps]);
        let pm = model_predictions(&model, &subj, &theta, &[0.1 - eps]);
        for j in 0..n_obs {
            h_bsv[(j, 0)] = (pp[j] - pm[j]) / (2.0 * eps);
        }

        // (1) Reduction: zero kappas + Ω_iov → 0 collapses to the BSV-only
        //     marginal. The OLD code added 0.5·K·log|Ω_iov| = log(1e-12) ≈ -27.6,
        //     so this assertion fails without the proper-marginal fix.
        let base = foce_subject_nll(
            &model, &subj, &theta, &eta_hat, &h_bsv, &omega_bsv, &sigma, false,
        );
        let zero_kappas = vec![DVector::zeros(1), DVector::zeros(1)];
        let reduced = foce_subject_nll_iov(
            &model,
            &subj,
            &theta,
            &eta_hat,
            &h_bsv,
            &omega_bsv,
            &sigma,
            false,
            &zero_kappas,
            &make_omega(1e-12),
        );
        // max_relative (not epsilon): these are O(1e5), and the residual κ-block
        // contribution at Ω_iov = 1e-12 is ~1e-11 relative. The old additive
        // penalty would shift by ~27.6 absolute (≈1.5e-4 relative) and fail.
        approx::assert_relative_eq!(reduced, base, max_relative = 1e-9);

        // (2) The marginal responds to Ω_iov through R̃ (the determinant term
        //     the old penalty was missing): with non-zero kappas, two different
        //     Ω_iov give materially different, finite OFVs.
        let kappas = vec![
            DVector::from_vec(vec![0.08]),
            DVector::from_vec(vec![-0.05]),
        ];
        let nll = |iov_var: f64| {
            foce_subject_nll_iov(
                &model,
                &subj,
                &theta,
                &eta_hat,
                &h_bsv,
                &omega_bsv,
                &sigma,
                false,
                &kappas,
                &make_omega(iov_var),
            )
        };
        let small = nll(0.005);
        let large = nll(0.5);
        assert!(small.is_finite() && large.is_finite());
        assert!(
            (small - large).abs() > 1e-6,
            "Ω_iov must change the marginal OFV (small={small}, large={large})"
        );
    }

    /// Regression for the FOCE+proportional fix: the residual variance must be
    /// evaluated at a supplied population prediction `f(η=0)`, not the
    /// SB-linearized `f0 = ipred − H·η̂`. When `f0` crosses zero (a nonlinear
    /// model's linearization undershooting), `R(f0) = (f0·σ)²` collapses to the
    /// floor and that observation's huge weight blows up the marginal — the
    /// pathology that made FOCE+proportional multimodal with an indefinite
    /// covariance. Passing `r_pred_override = Some(positive preds)` must avoid it.
    #[test]
    fn foce_standard_variance_uses_override_not_zero_crossing_f0() {
        let subject = make_simple_subject(); // 6 obs
        let omega = make_omega(0.09);
        let sigma = vec![0.2]; // proportional SD
        let error_spec = ErrorSpec::Single(ErrorModel::Proportional);

        // ipreds all positive; H·η̂ drives the first f0 component to exactly 0.
        let ipreds = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let eta_hat = DVector::from_vec(vec![1.0]);
        let h_matrix = DMatrix::from_column_slice(6, 1, &[10.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        // f0 = ipred − H·η̂ = [0, 15, 25, 35, 45, 55] → first R(f0) hits the floor.

        let nll_f0 = foce_subject_nll_standard(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            &[],
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );
        let nll_override = foce_subject_nll_standard(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            &[],
            BloqMethod::Drop,
            &[],
            None,
            Some(&ipreds),
            None,
        );

        assert!(nll_override.is_finite() && nll_f0.is_finite());
        // The f0 path's near-floored first-observation variance inflates the
        // marginal above the override path (which weights by the true ~positive
        // prediction). The two differ by a clear, deterministic margin (~56 on
        // this construction) — confirming R is evaluated at the override, not f0.
        // (The HΩH' term in R̃ cushions the floored R(f0), so the gap is moderate
        // rather than catastrophic, but it is well above FP noise.)
        assert!(
            nll_f0 - nll_override > 20.0,
            "override must change the SB marginal (R evaluated at f(η=0), not f0): \
             nll_f0={nll_f0}, nll_override={nll_override}"
        );
    }

    /// Regression for the FREM r_diag merge collision: `frem_r_override` must
    /// reach the residual-variance diagonal that feeds R̃, even though
    /// `r_pred_override` re-evaluates R afterward. Before the fix the override
    /// loop ran on a `r_diag` that was immediately shadowed and discarded, so
    /// FREM covariate rows silently used the PK error variance and the marginal
    /// was identical with or without the override.
    #[test]
    fn foce_standard_applies_frem_r_override() {
        let subject = make_simple_subject(); // 6 obs
        let omega = make_omega(0.09);
        let sigma = vec![0.2]; // proportional SD
        let error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        let ipreds = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let eta_hat = DVector::from_vec(vec![1.0]);
        let h_matrix = DMatrix::from_column_slice(6, 1, &[10.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

        let call = |frem: Option<&[Option<f64>]>| {
            foce_subject_nll_standard(
                &subject,
                &ipreds,
                &eta_hat,
                &h_matrix,
                &omega,
                &sigma,
                &error_spec,
                &[],
                BloqMethod::Drop,
                &[],
                frem,
                None,
                None,
            )
        };

        // Override the first row's residual variance with a value far from the
        // PK error model's R(f0)[0] = (10·0.2)² = 4.
        let overrides = [Some(250.0), None, None, None, None, None];
        let nll_plain = call(None);
        let nll_frem = call(Some(&overrides));

        assert!(nll_plain.is_finite() && nll_frem.is_finite());
        assert!(
            (nll_plain - nll_frem).abs() > 1e-6,
            "frem_r_override must change the marginal OFV (plain={nll_plain}, frem={nll_frem})"
        );
    }

    #[test]
    fn test_build_block_diag_omega_structure() {
        let bsv = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.09, 0.04]));
        let iov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.01]));
        let combined = build_block_diag_omega(&bsv, &iov, 2);
        // 2 BSV + 2*1 IOV = 4x4
        assert_eq!(combined.nrows(), 4);
        assert_eq!(combined.ncols(), 4);
        assert_eq!(combined[(0, 0)], 0.09);
        assert_eq!(combined[(1, 1)], 0.04);
        assert_eq!(combined[(2, 2)], 0.01); // occ 1 kappa
        assert_eq!(combined[(3, 3)], 0.01); // occ 2 kappa
        assert_eq!(combined[(0, 2)], 0.0); // off-block must be zero
    }

    /// Algebraic identity check: under additive (η-independent R) error, the
    /// Almquist `½·c̃'·c̃` INTER correction is identically zero, so
    /// `H̃ = a'·diag(1/R)·a + Ω⁻¹`. We hand-compute the closed-form Laplace
    /// value from the same H, R, η̂ and assert bit-for-bit agreement with
    /// `foce_subject_nll_interaction`.
    ///
    /// (Replaces the previous `test_focei_matches_foce_when_r_is_eta_independent`,
    /// which asserted FOCEI INTER == FOCE non-INTER exactly under additive
    /// error. That identity only holds for the Sheiner–Beal form; the
    /// Almquist Laplace form NONMEM/nlmixr2 use does *not* satisfy it because
    /// the two forms approximate the same true marginal differently for
    /// nonlinear models. See `[[focei-laplace-not-sheiner-beal]]`.)
    #[test]
    fn test_focei_laplace_additive_matches_handcomputed_hessian() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);

        let theta = vec![5.0, 50.0];
        let eta_hat = nalgebra::DVector::from_vec(vec![0.05]);
        let omega = make_omega(0.09);
        let sigma = vec![1.0];

        let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
        let n_obs = subj.obs_times.len();
        let eps = 1e-6;
        let mut h = DMatrix::zeros(n_obs, 1);
        let h_step = eps * (1.0 + eta_hat[0].abs());
        let preds_p =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
        let preds_m =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
        for i in 0..n_obs {
            h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
        }

        let espec = ErrorSpec::Single(ErrorModel::Additive);
        let focei = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &espec,
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );

        // Hand-compute the Laplace value with c̃ ≡ 0 (additive R).
        let r = sigma[0] * sigma[0]; // ErrorModel::Additive: R = σ²
        let mut data_ll = 0.0;
        for j in 0..n_obs {
            let res = subj.observations[j] - ipreds[j];
            data_ll += res * res / r + r.ln();
        }
        let eta_prior = eta_hat.dot(&(&omega.inv * &eta_hat));
        let mut htilde_scalar = omega.inv[(0, 0)];
        for j in 0..n_obs {
            htilde_scalar += h[(j, 0)] * h[(j, 0)] / r;
        }
        let log_det_htilde = htilde_scalar.ln(); // 1×1 case
        let expected = 0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde);

        assert!(
            (focei - expected).abs() < 1e-9,
            "FOCEI Laplace ({}) must equal hand-computed value ({}) under \
             additive error; diff = {}",
            focei,
            expected,
            focei - expected,
        );
    }

    /// IIV on residual error (#409): with a *dedicated* residual-error eta and
    /// additive error, predictions do not depend on that eta (so its `H`/`a`
    /// column is zero), but the FOCEI marginal must (a) scale R by exp(2·η̂_ruv)
    /// in the data term and (b) add the constant `c̃_{j,ruv}=2` column to the
    /// `½·c̃'·c̃` curvature, giving `H̃ = 0.5·(4·n_obs) + Ω⁻¹ = 2·n_obs + 1/ω`.
    /// We hand-compute the whole marginal and assert bit-for-bit agreement.
    #[test]
    fn test_focei_iiv_on_ruv_matches_handcomputed() {
        let subj = make_simple_subject();
        let n_obs = subj.observations.len();
        // Residual-error eta only: its prediction-Jacobian column is zero.
        let eta = 0.2_f64;
        let eta_hat = nalgebra::DVector::from_vec(vec![eta]);
        let omega_var = 0.05_f64;
        let omega = OmegaMatrix::from_diagonal(&[omega_var], vec!["ETA_RUV".into()]);
        let sigma = vec![2.0_f64];
        let espec = ErrorSpec::Single(ErrorModel::Additive);
        // Arbitrary predictions; the residual eta does not enter them.
        let ipreds = vec![48.0, 38.0, 32.0, 44.0, 36.0, 26.0];
        let h = DMatrix::<f64>::zeros(n_obs, 1); // ∂f/∂η_ruv ≡ 0

        let focei = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &espec,
            BloqMethod::Drop,
            &[],
            None,    // no FREM override
            Some(0), // ETA_RUV is eta index 0
            None,    // no custom magnitude
        );

        // Hand-computed marginal.
        let s = (2.0 * eta).exp();
        let r = sigma[0] * sigma[0] * s; // additive R, scaled
        let mut data_ll = 0.0;
        for j in 0..n_obs {
            let res = subj.observations[j] - ipreds[j];
            data_ll += res * res / r + r.ln();
        }
        let eta_prior = eta * eta / omega_var;
        // H̃ = hrh(0) + 0.5·ctc + Ω⁻¹; ctc(ruv,ruv) = Σ_j 2² = 4·n_obs.
        let htilde = 0.5 * (4.0 * n_obs as f64) + 1.0 / omega_var;
        let expected = 0.5 * (data_ll + eta_prior + omega.log_det + htilde.ln());

        assert!(
            (focei - expected).abs() < 1e-9,
            "FOCEI IIV-on-RUV marginal ({focei}) must equal hand-computed ({expected}); \
             diff = {}",
            focei - expected
        );

        // Sanity: passing `None` (no residual eta) drops both the scaling and the
        // c̃ column, so the marginal must differ.
        let focei_none = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &espec,
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );
        assert!(
            (focei - focei_none).abs() > 1e-6,
            "residual-eta marginal must differ from the no-RUV-eta marginal"
        );
    }

    #[test]
    fn compute_cwres_does_not_scale_frem_rows_by_iiv_on_ruv() {
        let subject = Subject {
            id: "1".to_string(),
            doses: Vec::new(),
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![11.0, 21.0],
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
            fremtype: vec![0, 100],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let ipreds = vec![10.0, 20.0];
        let eta_hat = DVector::from_vec(vec![0.5]);
        let h = DMatrix::<f64>::zeros(2, 1);
        let omega = OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_RUV".into()]);
        let sigma = vec![2.0, 0.5];
        let overrides = vec![None, Some(0.25)];

        let cwres = compute_cwres(
            &subject,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &ErrorSpec::Single(ErrorModel::Additive),
            &[],
            Some(&overrides),
            Some(0),
            None,
        );

        let pk_expected = 1.0 / (sigma[0] * eta_hat[0].exp());
        let frem_expected = 1.0 / sigma[1];
        assert_relative_eq!(cwres[0], pk_expected, epsilon = 1e-12);
        assert_relative_eq!(cwres[1], frem_expected, epsilon = 1e-12);
    }

    #[test]
    fn compute_cwres_applies_custom_magnitude() {
        // #484 review #4: sdtab CWRES must use the per-observation magnitude.
        // With H = 0 the SB prediction f0 = ipred and R_tilde = R (diagonal), so
        // for additive error CWRES_j = (y_j − f_j) / (m_j·σ).
        let subject = Subject {
            id: "1".to_string(),
            doses: Vec::new(),
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: vec![12.0, 22.0],
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
            fremtype: vec![0, 0],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let ipreds = vec![10.0, 20.0];
        let eta_hat = DVector::from_vec(vec![0.0]);
        let h = DMatrix::<f64>::zeros(2, 1);
        let omega = OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]);
        let sigma = vec![2.0];
        // Row 0 bare (m = 1), row 1 inflated (m = 3).
        let mult = vec![vec![1.0], vec![3.0]];

        let cwres = compute_cwres(
            &subject,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &ErrorSpec::Single(ErrorModel::Additive),
            &[],
            None,
            None,
            Some(&mult),
        );
        assert_relative_eq!(cwres[0], 2.0 / (1.0 * 2.0), epsilon = 1e-12);
        assert_relative_eq!(cwres[1], 2.0 / (3.0 * 2.0), epsilon = 1e-12);
    }

    /// IIV on residual error (#409): `individual_nll` must multiply the residual
    /// variance by exp(2·η_ruv). At η_ruv = 0 the scale is 1 (bit-identical to no
    /// IIV-on-RUV); at η_ruv ≠ 0 the value must change, and by exactly the
    /// closed-form amount for additive error.
    #[test]
    fn test_individual_nll_scales_residual_variance() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
        let theta = vec![5.0, 50.0];
        let omega = make_omega(0.09);
        let sigma = vec![2.0];

        // η_ruv = 0 → scale 1 → identical to the no-RUV-eta model.
        let eta0 = vec![0.0];
        let base0 = individual_nll(&model, &subj, &theta, &eta0, &omega, &sigma);
        model.residual_error_eta = Some(0);
        let ruv0 = individual_nll(&model, &subj, &theta, &eta0, &omega, &sigma);
        assert!(
            (base0 - ruv0).abs() < 1e-12,
            "η_ruv=0 must give scale 1 (base {base0}, ruv {ruv0})"
        );

        // η_ruv = 0.3 → variance ×exp(0.6). Difference vs the unscaled model is
        // 0.5·Σ_j[(1/s − 1)·res²/σ² + ln s] (prior/|Ω| terms cancel).
        let eta = vec![0.3];
        model.residual_error_eta = None;
        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        model.residual_error_eta = Some(0);
        let scaled = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        let s = (2.0_f64 * 0.3).exp();
        let preds = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta);
        let sig2 = sigma[0] * sigma[0];
        let mut delta = 0.0;
        for (j, &f) in preds.iter().enumerate() {
            let res = subj.observations[j] - f;
            delta += (1.0 / s - 1.0) * (res * res / sig2) + s.ln();
        }
        let expected = base + 0.5 * delta;
        assert!(
            (scaled - expected).abs() < 1e-9,
            "individual_nll IIV-on-RUV scaling mismatch: got {scaled}, expected {expected}"
        );
    }

    /// `obs_nll_subject_into` (the IS/IMPMAP/SAEM data term) must apply the same
    /// exp(2·η_ruv) variance scaling.
    #[test]
    fn test_obs_nll_subject_into_scales_residual_variance() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
        let theta = vec![5.0, 50.0];
        let sigma = vec![2.0];
        let eta = vec![0.4];
        let mut scratch = pk::EventPkParams::with_capacity_for(&subj);

        let base = obs_nll_subject_into(&model, &subj, &theta, &sigma, &eta, &mut scratch);
        model.residual_error_eta = Some(0);
        let scaled = obs_nll_subject_into(&model, &subj, &theta, &sigma, &eta, &mut scratch);

        let s = (2.0_f64 * 0.4).exp();
        let preds = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta);
        let sig2 = sigma[0] * sigma[0];
        let mut expected = 0.0;
        for (j, &f) in preds.iter().enumerate() {
            let f = f.max(1e-12);
            let v = (sig2 * s).max(1e-12);
            expected += 0.5 * (v.ln() + (subj.observations[j] - f).powi(2) / v);
        }
        assert!(
            (scaled - expected).abs() < 1e-9,
            "obs_nll IIV-on-RUV scaling mismatch: got {scaled}, expected {expected}, base {base}"
        );
    }

    /// Confirms the Almquist `½·c̃'·c̃` INTER correction is actually wired:
    /// switching from additive (c̃ ≡ 0) to combined error (`dvar_df ≠ 0`,
    /// hence c̃ ≠ 0) must change the Laplace H̃ and therefore the per-subject
    /// OFV by a non-trivial amount, even when the proportional-component
    /// magnitude is small enough that the *data* term barely shifts.
    ///
    /// Catches a regression where the c̃'c̃ accumulator is dropped or zeroed —
    /// in that case the additive and combined Laplace values would coincide
    /// up to the tiny `(prop·f)² → R` difference in the data quadratic.
    #[test]
    fn test_focei_laplace_combined_uses_inter_correction() {
        let subj = make_simple_subject();
        let mut model = make_model();
        model.error_model = ErrorModel::Combined;
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);

        let theta = vec![5.0, 50.0];
        let eta_hat = nalgebra::DVector::from_vec(vec![0.1]);
        let omega = make_omega(0.09);
        // Two sigmas for Combined: (prop, add). The proportional component is
        // intentionally small relative to additive so the data-side R hardly
        // changes — any noticeable OFV shift is then due to the c̃'c̃ piece.
        let sigma_combined = vec![0.02, 1.0];
        let sigma_additive_only = vec![1.0]; // matches Combined R when prop ≈ 0

        let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
        let n_obs = subj.obs_times.len();
        let eps = 1e-6;
        let mut h = DMatrix::zeros(n_obs, 1);
        let h_step = eps * (1.0 + eta_hat[0].abs());
        let preds_p =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
        let preds_m =
            pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
        for i in 0..n_obs {
            h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
        }

        let espec_combined = ErrorSpec::Single(ErrorModel::Combined);
        let espec_additive = ErrorSpec::Single(ErrorModel::Additive);
        let focei_combined = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma_combined,
            &espec_combined,
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );
        let focei_additive = foce_subject_nll_interaction(
            &subj,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma_additive_only,
            &espec_additive,
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );
        let gap = focei_combined - focei_additive;
        assert!(
            gap.abs() > 1e-3,
            "FOCEI Laplace must respond to the Almquist `½·c̃'·c̃` INTER \
             correction; combined ({}) and additive ({}) gave gap = {} — \
             too small, c̃'c̃ likely not being accumulated.",
            focei_combined,
            focei_additive,
            gap,
        );
    }

    /// Joint PK-TTE (ODE-accumulated hazard) must give a finite per-subject NLL on every
    /// estimation entry point: the inner EBE NLL, the SAEM M-step, and the FOCEI Laplace
    /// objective (which adds the TTE FD-Hessian). Fast — a handful of ODE solves, no outer
    /// optimisation (the full fit is the Tier-3 test in `tte_convergence.rs`). This is the
    /// per-PR coverage for the ODE-hazard likelihood path (#564).
    #[cfg(feature = "survival")]
    #[test]
    fn joint_pktte_ode_hazard_nll_paths_finite() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
        let model = parse_model_string(src).expect("joint PK-TTE model must parse");
        let mut subject = make_simple_subject();
        // make_simple_subject doses CMT 1 (depot) with 6 PK obs; add a TTE event on CMT 2.
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];

        let p = &model.default_params;
        let eta = [0.0_f64];
        let n_obs = subject.observations.len();

        // (1) Inner EBE NLL: individual_nll → tte_endpoint_nll → tte_ode_nll.
        let inner = individual_nll(&model, &subject, &p.theta, &eta, &p.omega, &p.sigma.values);
        assert!(
            inner.is_finite(),
            "inner joint NLL must be finite; got {inner}"
        );

        // (2) SAEM M-step: obs_nll_subject_from_preds (Gaussian preds supplied).
        let preds = vec![30.0_f64; n_obs];
        let saem =
            obs_nll_subject_from_preds(&model, &subject, &preds, &p.theta, &p.sigma.values, &eta);
        assert!(
            saem.is_finite(),
            "SAEM M-step joint NLL must be finite; got {saem}"
        );

        // (3) FOCEI Laplace objective, incl. the ODE-TTE FD-Hessian term. A zero Gaussian
        // sensitivity matrix is fine here — the prior + TTE Hessian keep H̃ positive-definite.
        let eta_hat = DVector::from_vec(vec![0.0]);
        let h_matrix = DMatrix::<f64>::zeros(n_obs, 1);
        let foce = foce_subject_nll(
            &model,
            &subject,
            &p.theta,
            &eta_hat,
            &h_matrix,
            &p.omega,
            &p.sigma.values,
            true,
        );
        assert!(
            foce.is_finite(),
            "FOCEI joint NLL must be finite; got {foce}"
        );

        // The ODE hazard must actually contribute: moving η changes the inner NLL.
        let moved = individual_nll(
            &model,
            &subject,
            &p.theta,
            &[0.4],
            &p.omega,
            &p.sigma.values,
        );
        assert!(
            moved.is_finite() && (moved - inner).abs() > 1e-9,
            "η must move the joint NLL (inner={inner}, moved={moved})"
        );
    }
}
