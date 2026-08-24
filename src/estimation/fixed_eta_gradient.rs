//! Per-subject gradients of the **observation-only** NLL with random effects held
//! fixed — `∂/∂(θ, σ)` of `−log p(yᵢ | ηᵢ, θ, σ)` in the packed
//! `[log_theta | log_sigma]` space.
//!
//! This is the "fixed-η partial", **not** a marginal-likelihood gradient: it
//! deliberately omits the `∂η̂/∂θ` implicit-function terms and the `½log|H̃|`
//! curvature terms that FOCE/FOCEI carry (see
//! [`crate::estimation::sens_outer_gradient::subject_packed_gradient`] for those).
//! That makes it the right primitive for any estimator whose η is *given* rather
//! than *profiled*:
//!
//! - SAEM's M-step, where η is a draw from the E-step ([`crate::estimation::saem`]);
//! - variational inference, where η is a reparameterized draw from `q_φ`
//!   ([`crate::estimation::vi`]).
//!
//! Extracted verbatim from `saem.rs` so both callers share one copy; SAEM's
//! behaviour is unchanged.
//!
//! Both entry points return `(nll, grad)` so the caller gets the objective for
//! free alongside its gradient. `lower[i] == upper[i]` marks a pinned coordinate:
//! it contributes 0 and skips its finite-difference evaluation.

use crate::pk::EventPkParams;
use crate::stats::likelihood::{obs_nll_subject_from_preds, obs_nll_subject_into};
use crate::types::*;

#[cfg(test)]
#[path = "fixed_eta_gradient_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// IOV-aware observation NLL for M-step (no priors, per-occasion predictions)
// ---------------------------------------------------------------------------

/// Compute the observation-only NLL for an IOV subject in the SAEM M-step.
///
/// ETAs and kappas are held fixed (sampled values from the E-step).  For each
/// occasion k the combined `[eta, kappa_k]` vector is used to compute predictions;
/// only the observations belonging to that occasion are scored.  No eta or kappa
/// prior terms are included — those are handled by the SA sufficient-statistic
/// update for Ω_bsv and Ω_iov separately.
pub(crate) fn obs_nll_subject_into_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    _pk_scratch: &mut crate::pk::EventPkParams,
) -> f64 {
    use crate::stats::likelihood::m3_logcdf;
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    // Continuous per-occasion-aware prediction (issue #104) — same model the
    // E-step (`individual_nll_iov`) and FOCEI use, so E and M steps stay
    // consistent. `_pk_scratch` is retained for signature stability but unused
    // (predict_iov manages its own per-event params).
    let preds = crate::pk::predict_iov(model, subject, theta, eta, kappas);
    // FREM covariate pseudo-observations (FREMTYPE > 0) use the covariate sigma
    // (EPSCOV), not the PK residual error — otherwise their near-zero residuals
    // drag PROP/ADD toward zero. See build_frem_r_override.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );
    // IIV on residual error (#409): scale the PK residual variance by
    // exp(2·η_ruv); FREM rows keep their own variance.
    let ruv_scale = model.residual_var_scale(eta);
    // #484/#1029: per-observation residual-magnitude multiplier (θ/covariate/TIME
    // only, never η), so the M-step scores the same variance the E-step and every
    // other estimator does.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut total_nll = 0.0_f64;
    for j in 0..subject.observations.len() {
        // Floors protect log(0) in the M-step objective. individual_nll_iov
        // (the E-step evaluator) does not floor — see obs_nll_subject_grad_iov
        // for why the asymmetry is intentional.
        let f = preds[j].max(1e-12);
        let v = match frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
            Some(vv) => vv.max(1e-12),
            None => (model.residual_variance_at_scaled(
                err_keys[j],
                f,
                sigma_values,
                ruv_mult.as_ref().map(|m| m[j].as_slice()),
            ) * ruv_scale)
                .max(1e-12),
        };
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        if m3 && cens != 0 {
            total_nll += -m3_logcdf(subject.observations[j], f, v.sqrt(), cens);
        } else {
            total_nll += 0.5 * (v.ln() + (subject.observations[j] - f).powi(2) / v);
        }
    }

    // Non-Gaussian data term at raw-NLL weight (1×) to match the true NLL (Gaussian obs
    // already contribute at 0.5·(log v + r²/v)): TTE endpoints plus the discrete
    // (binary/categorical) term. No joint-share on the IOV M-step path, and its analytic TTE
    // term matches the `tte_data_term` this site inlined.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        crate::stats::likelihood::accumulate_non_gaussian_nll(
            model,
            subject,
            theta,
            eta,
            None,
            1.0,
            &mut total_nll,
        );
    }

    total_nll
}

/// Gradient of the IOV observation NLL w.r.t. the SAEM packed vector
/// `[log_theta | log_sigma]` for one subject with ETAs and kappas fixed.
///
/// Sigma gradient is analytical (same formula as the non-IOV path but summed
/// across all occasions' observations).  Theta gradient uses forward-FD of
/// per-occasion predictions, chain-rule'd through the per-observation obs_nll.
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_grad_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    theta_packs_log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    pk_scratch: &mut crate::pk::EventPkParams,
) -> (f64, Vec<f64>) {
    let n = n_theta + n_sigma;
    // IOV + block_sigma is rejected up front (E_BLOCK_SIGMA_IOV_UNSUPPORTED), so
    // `residual_correlations` is never set on this IOV path — only M3 (and TTE,
    // under the `survival` feature) need the full-FD fallback here.
    let fd_all = matches!(model.bloq_method, BloqMethod::M3);
    // Fall back to full FD when TTE endpoints are present: the analytic non-M3
    // path is Gaussian-only and would silently zero hazard-parameter gradients.
    #[cfg(feature = "survival")]
    let fd_all = fd_all || !model.endpoints.is_empty();

    if fd_all {
        // M3 / TTE path: forward-FD of obs_nll_subject_into_iov.
        let nll_base =
            obs_nll_subject_into_iov(model, subject, theta, sigma_values, eta, kappas, pk_scratch);
        let mut grad = vec![0.0f64; n];
        let h = 1e-5;
        for i in 0..n {
            if lower[i] == upper[i] {
                continue;
            }
            if i < n_theta {
                let mut theta_p = theta.to_vec();
                let delta = h * (1.0 + theta[i].abs());
                theta_p[i] += delta;
                let nll_p = obs_nll_subject_into_iov(
                    model,
                    subject,
                    &theta_p,
                    sigma_values,
                    eta,
                    kappas,
                    pk_scratch,
                );
                let raw = (nll_p - nll_base) / delta;
                grad[i] = if theta_packs_log_mask[i] {
                    theta[i] * raw
                } else {
                    raw
                };
            } else {
                let k = i - n_theta;
                let mut sigma_p = sigma_values.to_vec();
                let delta = h * (1.0 + sigma_values[k].abs());
                sigma_p[k] += delta;
                let nll_p = obs_nll_subject_into_iov(
                    model, subject, theta, &sigma_p, eta, kappas, pk_scratch,
                );
                grad[i] = sigma_values[k] * (nll_p - nll_base) / delta;
            }
        }
        return (nll_base, grad);
    }

    // Non-M3 path: continuous per-occasion-aware base predictions (issue #104).
    let n_obs = subject.observations.len();
    let preds = crate::pk::predict_iov(model, subject, theta, eta, kappas);
    // FREM covariate rows use EPSCOV, not the PK residual error (see
    // build_frem_r_override); their variance is η-independent so dvar_df = 0.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );
    // IIV on residual error (#409): per-subject `exp(2·η_ruv)` scale on the PK
    // residual variance (FREM rows excluded). η_ruv is a BSV eta, indexed into
    // `eta`.  See the non-IOV `obs_nll_subject_grad` for the score-consistency
    // argument behind scaling V, dV/df, and dV/dlogσ together.
    let ruv_scale = model.residual_var_scale(eta);
    // #484/#1029: per-observation residual-magnitude multiplier. It rides the
    // sigma loadings, so V, ∂V/∂f, and ∂V/∂log σ all take their `_scaled` forms
    // together — the same score-consistency argument as `ruv_scale` below.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    let mut nll_base = 0.0_f64;
    let mut all_preds_base = vec![0.0f64; n_obs];
    let mut residuals = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut d_nll_d_f = vec![0.0f64; n_obs];
    let mut obs_var_scale = vec![1.0f64; n_obs];

    for j in 0..n_obs {
        let cmt = err_keys[j];
        let f = preds[j].max(1e-12);
        let mult_j = ruv_mult.as_ref().map(|m| m[j].as_slice());
        let frem_vj = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
        let s = if frem_vj.is_some() { 1.0 } else { ruv_scale };
        obs_var_scale[j] = s;
        let v = match frem_vj {
            Some(vv) => vv.max(1e-12),
            None => {
                (model.residual_variance_at_scaled(cmt, f, sigma_values, mult_j) * s).max(1e-12)
            }
        };
        let resid = subject.observations[j] - f;
        nll_base += 0.5 * (v.ln() + resid * resid / v);
        all_preds_base[j] = f;
        residuals[j] = resid;
        variances[j] = v;
        let dv_df = match (frem_vj, mult_j) {
            (Some(_), _) => 0.0,
            (None, Some(m)) => model.error_spec.dvar_df_scaled(cmt, f, sigma_values, m) * s,
            (None, None) => model.error_spec.dvar_df(cmt, f, sigma_values) * s,
        };
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    let mut grad = vec![0.0f64; n];

    // Theta gradient: forward-FD of the continuous prediction (one perturbed
    // prediction per theta; κ affects later occasions via carryover so the
    // sensitivity is captured across all rows).
    let h_fd = 1e-5;
    for i in 0..n_theta {
        if lower[i] == upper[i] {
            continue;
        }
        let delta = h_fd * (1.0 + theta[i].abs());
        let mut theta_p = theta.to_vec();
        theta_p[i] += delta;
        let preds_p = crate::pk::predict_iov(model, subject, &theta_p, eta, kappas);
        // With a custom magnitude active, θ moves the residual variance through
        // *two* channels — the prediction (`∂nll/∂f · ∂f/∂θ`) and, when the
        // magnitude expression names a θ, the variance directly. The analytic
        // chain below carries only the first, so difference the whole Gaussian
        // NLL instead: the prediction solve this loop already pays for is the
        // dominant cost, and re-evaluating the magnitude bytecode is negligible
        // beside it. A magnitude-free model keeps the exact legacy chain, so no
        // existing fit changes numerically (#484/#1029).
        let mut d_obs_nll = 0.0_f64;
        if ruv_mult.is_some() {
            let mult_p = model.ruv_obs_mult(subject, &theta_p);
            let nll_p = crate::stats::residual_error::gaussian_obs_nll_scaled(
                &model.error_spec,
                &err_keys,
                &subject.observations,
                &preds_p,
                sigma_values,
                &model.residual_correlations,
                ruv_scale,
                frem_ov.as_deref(),
                mult_p.as_deref(),
            );
            d_obs_nll = (nll_p - nll_base) / delta;
        } else {
            for j in 0..n_obs {
                d_obs_nll += d_nll_d_f[j] * (preds_p[j] - all_preds_base[j]) / delta;
            }
        }
        grad[i] = if theta_packs_log_mask[i] {
            theta[i] * d_obs_nll
        } else {
            d_obs_nll
        };
    }

    // Sigma gradient: analytical — same formula as non-IOV, summed over all obs.
    for k in 0..n_sigma {
        let i = n_theta + k;
        if lower[i] == upper[i] {
            continue;
        }
        let g: f64 = (0..n_obs)
            .map(|j| {
                let f = all_preds_base[j];
                let v = variances[j];
                let resid = residuals[j];
                // d(v_j)/d(log sigma_k); zero unless sigma_k enters obs j's
                // endpoint, so per-CMT each sigma picks up only its own
                // endpoint's observations. The #484/#1029 magnitude rides slot
                // k's loading, so it scales this derivative by m_k² — the same
                // `_scaled` pairing V/∂V∂f already take above.
                let ratio = match ruv_mult.as_ref().map(|m| m[j].as_slice()) {
                    Some(m) => {
                        model
                            .error_spec
                            .dvar_dlogsigma_scaled(err_keys[j], k, f, sigma_values, m)
                    }
                    None => model
                        .error_spec
                        .dvar_dlogsigma(err_keys[j], k, f, sigma_values),
                } * obs_var_scale[j];
                0.5 * ratio * (1.0 / v - resid * resid / (v * v))
            })
            .sum();
        grad[i] = g;
    }

    (nll_base, grad)
}

/// Gradient of `obs_nll` w.r.t. the SAEM packed parameter vector
/// `[log_theta_0 … log_theta_{P-1} | log_sigma_0 … log_sigma_{Q-1}]`
/// for a single subject with ETAs held fixed.
///
/// For non-M3 models:
/// - Sigma: analytical from the residual-variance formula (no extra predict call).
/// - Theta: forward-FD of `compute_predictions_with_tv_into` + chain rule through
///   obs_nll (one extra predict call per non-pinned theta, not one full-subject
///   NLL call).
///
/// For M3 models (complex Mills-ratio sigma gradient): forward-FD of
/// `obs_nll_subject_into` for all parameters.
///
/// `lower`/`upper` are the packed-space bounds used to detect pinned dimensions
/// (`lower[i] == upper[i]`); pinned dimensions contribute 0 to the gradient and
/// skip their FD call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_grad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
    theta_packs_log_mask: &[bool],
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    pk_scratch: &mut EventPkParams,
) -> (f64, Vec<f64>) {
    let n = n_theta + n_sigma;
    let fd_all =
        matches!(model.bloq_method, BloqMethod::M3) || !model.residual_correlations.is_empty();
    // Fall back to the full-FD path when TTE endpoints are present: the analytic
    // non-M3 path is Gaussian-only and would silently zero hazard-parameter gradients.
    #[cfg(feature = "survival")]
    let fd_all = fd_all || !model.endpoints.is_empty();

    if fd_all {
        // M3 / TTE / dense residual-covariance path: forward-FD of
        // obs_nll_subject_into for all parameters. Predictions are σ-independent,
        // so solve the model once and reuse the base predictions across every σ
        // perturbation — only θ perturbations need a fresh solve (#557).
        let preds_base =
            crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
        let nll_base =
            obs_nll_subject_from_preds(model, subject, &preds_base, theta, sigma_values, eta);
        let mut grad = vec![0.0f64; n];
        let h = 1e-5;
        for i in 0..n {
            if lower[i] == upper[i] {
                continue;
            }
            if i < n_theta {
                let mut theta_p = theta.to_vec();
                let delta = h * (1.0 + theta[i].abs());
                theta_p[i] += delta;
                let nll_p =
                    obs_nll_subject_into(model, subject, &theta_p, sigma_values, eta, pk_scratch);
                let raw = (nll_p - nll_base) / delta;
                grad[i] = if theta_packs_log_mask[i] {
                    theta[i] * raw
                } else {
                    raw
                };
            } else {
                let k = i - n_theta;
                let mut sigma_p = sigma_values.to_vec();
                let delta = h * (1.0 + sigma_values[k].abs());
                sigma_p[k] += delta;
                let nll_p =
                    obs_nll_subject_from_preds(model, subject, &preds_base, theta, &sigma_p, eta);
                // log-packing for sigma: d/d(log_sigma_k) = sigma_k * d/d(sigma_k)
                grad[i] = sigma_values[k] * (nll_p - nll_base) / delta;
            }
        }
        return (nll_base, grad);
    }

    // Non-M3 path.
    let preds_base =
        crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);

    let mut nll_base = 0.0f64;
    let n_obs = subject.observations.len();

    // FREM covariate rows use EPSCOV, not the PK residual error (see
    // build_frem_r_override); their variance is η-independent so dvar_df = 0.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma_values,
    );

    // IIV on residual error (#409): per-subject scale on the PK residual
    // variance (`exp(2·η_ruv)`). FREM covariate rows are not scaled, so we hold
    // a per-obs scale and apply it consistently to V, dV/df, and dV/dlogσ so the
    // analytical score stays exact.
    let ruv_scale = model.residual_var_scale(eta);
    // #484/#1029: per-observation residual-magnitude multiplier, applied to V,
    // dV/df, and dV/dlogσ together for the same score-consistency reason as
    // `ruv_scale`.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    // per-obs residual, variance, d(obs_nll)/d(f_j), and the variance scale used.
    let mut residuals = vec![0.0f64; n_obs];
    let mut variances = vec![0.0f64; n_obs];
    let mut d_nll_d_f = vec![0.0f64; n_obs];
    let mut obs_var_scale = vec![1.0f64; n_obs];

    for j in 0..n_obs {
        let cmt = err_keys[j];
        let f = preds_base[j].max(1e-12);
        let mult_j = ruv_mult.as_ref().map(|m| m[j].as_slice());
        let frem_vj = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
        let s = if frem_vj.is_some() { 1.0 } else { ruv_scale };
        obs_var_scale[j] = s;
        let v = match frem_vj {
            Some(vv) => vv.max(1e-12),
            None => {
                (model.residual_variance_at_scaled(cmt, f, sigma_values, mult_j) * s).max(1e-12)
            }
        };
        let resid = subject.observations[j] - f;
        nll_base += 0.5 * (v.ln() + resid * resid / v);
        residuals[j] = resid;
        variances[j] = v;
        // d(obs_nll_j)/d(f_j) = -resid/V + 0.5 * (dV/df) * (1/V - resid²/V²)
        let dv_df = match (frem_vj, mult_j) {
            (Some(_), _) => 0.0,
            (None, Some(m)) => model.error_spec.dvar_df_scaled(cmt, f, sigma_values, m) * s,
            (None, None) => model.error_spec.dvar_df(cmt, f, sigma_values) * s,
        };
        d_nll_d_f[j] = -resid / v + 0.5 * dv_df * (1.0 / v - resid * resid / (v * v));
    }

    let mut grad = vec![0.0f64; n];

    // Theta gradient: forward-FD of predictions, chain rule through obs_nll.
    let h_fd = 1e-5;
    for i in 0..n_theta {
        if lower[i] == upper[i] {
            continue;
        }
        let delta = h_fd * (1.0 + theta[i].abs());
        let mut theta_p = theta.to_vec();
        theta_p[i] += delta;
        let preds_p =
            crate::pk::compute_predictions_with_tv_into(model, subject, &theta_p, eta, pk_scratch);
        // Difference on raw predictions — do NOT clip before differencing.
        // Clipping both pp and pb at 1e-12 before subtracting would produce a
        // zero difference whenever pb < 1e-12, silently zeroing the gradient.
        //
        // With a magnitude active, θ reaches the variance directly as well as
        // through the prediction, and this chain carries only the second
        // channel — so difference the whole Gaussian NLL instead (the IOV twin
        // of this fork carries the same reasoning). Magnitude-free models keep
        // the exact legacy chain and are numerically unchanged.
        let d_obs_nll: f64 = if ruv_mult.is_some() {
            let mult_p = model.ruv_obs_mult(subject, &theta_p);
            let nll_p = crate::stats::residual_error::gaussian_obs_nll_scaled(
                &model.error_spec,
                &err_keys,
                &subject.observations,
                &preds_p,
                sigma_values,
                &model.residual_correlations,
                ruv_scale,
                frem_ov.as_deref(),
                mult_p.as_deref(),
            );
            (nll_p - nll_base) / delta
        } else {
            d_nll_d_f
                .iter()
                .zip(preds_p.iter().zip(preds_base.iter()))
                .map(|(&dl, (&pp, &pb))| dl * (pp - pb) / delta)
                .sum()
        };
        grad[i] = if theta_packs_log_mask[i] {
            theta[i] * d_obs_nll
        } else {
            d_obs_nll
        };
    }

    // Sigma gradient: analytical.
    // d(obs_nll)/d(log_sigma_k) = Σ_j 0.5 * ratio_jk * (1/V_j - resid_j²/V_j²)
    // where ratio_jk = sigma_k * dV_j/d_sigma_k.
    for k in 0..n_sigma {
        let i = n_theta + k;
        if lower[i] == upper[i] {
            continue;
        }
        let g: f64 = (0..n_obs)
            .map(|j| {
                let f = preds_base[j].max(1e-12);
                let v = variances[j];
                let resid = residuals[j];
                // ratio = d(V_j)/d(log sigma_k); zero unless sigma_k enters
                // obs j's endpoint (so per-CMT each sigma sums only over its
                // own endpoint's observations). The #484/#1029 magnitude rides
                // slot k's loading, scaling this derivative by m_k².
                let ratio = match ruv_mult.as_ref().map(|m| m[j].as_slice()) {
                    Some(m) => {
                        model
                            .error_spec
                            .dvar_dlogsigma_scaled(err_keys[j], k, f, sigma_values, m)
                    }
                    None => model
                        .error_spec
                        .dvar_dlogsigma(err_keys[j], k, f, sigma_values),
                } * obs_var_scale[j];
                0.5 * ratio * (1.0 / v - resid * resid / (v * v))
            })
            .sum();
        grad[i] = g;
    }

    (nll_base, grad)
}
