use crate::pk;
use crate::stats::likelihood::{
    individual_nll_into_with_schedule, individual_nll_iov, split_obs_by_occasion,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "autodiff")]
use crate::ad::ad_gradients::{self, FlatDoseData};

/// Materialise the per-observation `obs_scale` array for one subject,
/// suitable for passing to the analytical AD entry points
/// (`individual_nll_ad`, `predict_all_ad`) as a `Const` slice.
///
/// Computes pk once per call only when `model.scaling.needs_pk_eval()`
/// — i.e. there's at least one `ExpressionScale` closure (top level or
/// nested in `PerCmt`) that consults pk. Scalar-only scaling skips the
/// pk_param_fn call (which can be expensive on models with parsed
/// expressions or NN forward passes). (Caught by Copilot review on PR
/// #85.)
#[cfg(feature = "autodiff")]
pub(crate) fn build_scale_array_for_ad(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    let pk_owned;
    let pk_ref: &PkParams = if model.scaling.needs_pk_eval() {
        pk_owned = (model.pk_param_fn)(theta, eta, &subject.covariates);
        &pk_owned
    } else {
        // Safe placeholder: no ExpressionScale closure will fire when
        // needs_pk_eval() is false, so pk values are never read.
        static DEFAULT_PK: PkParams = PkParams {
            values: [0.0; crate::types::MAX_PK_PARAMS],
        };
        &DEFAULT_PK
    };
    model
        .scaling
        .build_obs_scale_array(theta, eta, &subject.covariates, pk_ref, &subject.obs_cmts)
}

/// Build a per-event scale array for the event-driven AD entry points
/// (`individual_nll_event_driven_ad`, `predict_all_event_driven_ad`).
///
/// Length = `event_data.event_times.len()`. Obs events (`event_kinds[i]`
/// in (0.5, 1.5)) get the corresponding per-observation scale; non-obs
/// events (dose, pk-only) get `1.0`. Padding non-obs entries to `1.0` is
/// essential — the AD body divides `conc / obs_scale[ev_idx]` for every
/// event before the `is_obs` mask drops the non-obs contributions, and
/// NaN/0 in a non-obs slot would propagate through the masked add as NaN
/// (per IEEE 754 `0 * NaN = NaN`).
#[cfg(feature = "autodiff")]
pub(crate) fn build_event_scale_array_for_ad(
    model: &CompiledModel,
    subject: &Subject,
    event_data: &crate::ad::event_driven_ad::FlatEventData,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    let obs_scales = build_scale_array_for_ad(model, subject, theta, eta);
    let n_events = event_data.event_times.len();
    let mut event_scales = vec![1.0; n_events];
    for ev_idx in 0..n_events {
        // event_kinds: 0 = dose, 1 = obs, 2 = pk-only. Only obs entries
        // route to the per-observation scale; everything else stays 1.0.
        let kind = event_data.event_kinds[ev_idx];
        if kind > 0.5 && kind < 1.5 {
            let obs_idx = event_data.event_orig_idx_f64[ev_idx] as usize;
            if let Some(&s) = obs_scales.get(obs_idx) {
                event_scales[ev_idx] = s;
            }
        }
    }
    event_scales
}

/// Resolve [`GradientMethod::Auto`] to a concrete AD/FD choice for this model.
/// Returns `true` for AD, `false` for FD.
///
/// Policy (`Auto` case): prefer AD whenever it is available. Empirically
/// (`FERX_TIME_GRADIENTS=1` on 1-cpt oral, 2-cpt infusion, 3-cpt infusion)
/// reverse-mode AD is 1.5-5x faster per BFGS gradient call than central FD
/// across the tested range of models — the tape/backward overhead is
/// dominated by the savings from one gradient call vs `2·n_eta` forward
/// perturbations, even at small `n_eta`.
///
/// AD requires (a) the crate compiled with `feature = "autodiff"` and
/// (b) the model to have `tv_fn` populated (analytical PK path only).
/// ODE models have no AD path, so `Auto` resolves to FD there.
///
/// For subjects with time-varying covariates, the *event-driven* AD path
/// is used when the structural model is in
/// [`crate::ad::event_driven_ad::supports_event_driven_ad`] (currently
/// 1- and 2-cpt IV bolus + infusion). Other models with TV covariates
/// fall back to FD — the single-snapshot AD path can't honour per-event
/// covariate values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InnerGradientMethod {
    /// Finite differences. Used when AD is unavailable/disabled, when
    /// the model has no `tv_fn`, or when the subject has TV covariates
    /// on a model the event-driven AD path doesn't support yet.
    Fd,
    /// Reverse-mode AD with a single per-subject `tv_adjusted` vector
    /// — the legacy fast path. Correct only when the subject has no
    /// time-varying covariates.
    AdSingleSnapshot,
    /// Reverse-mode AD with per-event `tv` arrays — required for
    /// time-varying covariates to be reflected in gradients.
    AdEventDriven,
}

fn resolve_gradient_method(model: &CompiledModel, subject: &Subject) -> InnerGradientMethod {
    #[cfg(not(feature = "autodiff"))]
    {
        let _ = model;
        let _ = subject;
        return InnerGradientMethod::Fd;
    }
    #[cfg(feature = "autodiff")]
    {
        if model.tv_fn.is_none() {
            return InnerGradientMethod::Fd;
        }
        let want_ad = match model.gradient_method {
            GradientMethod::Ad => true,
            GradientMethod::Fd => false,
            GradientMethod::Auto => true,
        };
        if !want_ad {
            return InnerGradientMethod::Fd;
        }
        // SS=1 doses in the AD paths would require threading `dose.ss` and
        // `dose.ii` through the AD-instrumented propagators and adding
        // closed-form SS branches inside them. Until that lands, fall back
        // to FD whenever the subject has any SS dose — otherwise AD
        // gradients (computed against the single-dose response) would not
        // match the SS-aware predictions from `predict_concentration`.
        if subject.has_ss_doses() {
            return InnerGradientMethod::Fd;
        }
        // System resets (EVID=3/4) zero the compartment state mid-record.
        // The AD-instrumented propagators have no reset event, so their
        // gradients wouldn't match the reset-aware predictions from the
        // event-driven path. Fall back to FD until the AD path learns
        // resets — the FD path routes through the reset-aware analytical
        // propagator (see `pk::compute_predictions`).
        if subject.has_resets() {
            return InnerGradientMethod::Fd;
        }
        // Lagtime in the event-driven AD path would require threading
        // per-dose `lagtime` through the AD-instrumented propagators and
        // their infusion-window checks. The single-snapshot AD path
        // already handles lagtime (see `ad_gradients.rs::predict_all_ad`).
        // Until the event-driven AD path is updated, fall back to FD when
        // a TV-cov subject is paired with a lagtime-bearing model.
        if subject.has_tv_covariates() {
            if crate::ad::event_driven_ad::supports_event_driven_ad(model.pk_model)
                && !model.has_lagtime()
            {
                InnerGradientMethod::AdEventDriven
            } else {
                InnerGradientMethod::Fd
            }
        } else {
            InnerGradientMethod::AdSingleSnapshot
        }
    }
}

/// One-line summary of the inner-loop gradient route **actually resolved**
/// across the population, for the startup banner. Reflects the per-subject
/// resolution in [`resolve_gradient_method`] — including AD→FD fallbacks for
/// SS doses, system resets, TV-covariate models the event-driven AD path
/// doesn't support, ODE/`tv_fn`-less models, or a build without the
/// `autodiff` feature.
///
/// `requested` is the user's [`FitOptions::gradient_method`], appended in
/// brackets so a fallback is visible. It is taken as a parameter rather than
/// read from `model.gradient_method` because the latter is mutated by
/// compatibility rules (e.g. an SDE model is forced to `Fd` regardless of the
/// request) — the banner should report what the user asked for, not the
/// post-compatibility value.
pub(crate) fn gradient_route_summary(
    model: &CompiledModel,
    population: &Population,
    requested: GradientMethod,
) -> String {
    let (mut fd, mut ss, mut ed) = (0usize, 0usize, 0usize);
    for subject in &population.subjects {
        match resolve_gradient_method(model, subject) {
            InnerGradientMethod::Fd => fd += 1,
            InnerGradientMethod::AdSingleSnapshot => ss += 1,
            InnerGradientMethod::AdEventDriven => ed += 1,
        }
    }
    // Show per-route counts only when the population splits across routes;
    // a single uniform route reads cleanly as just its label.
    let mixed = [fd, ss, ed].iter().filter(|&&c| c > 0).count() > 1;
    let mut parts: Vec<String> = Vec::new();
    for (count, label) in [
        (ed, "AD (event-driven)"),
        (ss, "AD (single-snapshot)"),
        (fd, "FD"),
    ] {
        if count > 0 {
            parts.push(if mixed {
                format!("{label} ×{count}")
            } else {
                label.to_string()
            });
        }
    }
    let resolved = if parts.is_empty() {
        "n/a".to_string()
    } else {
        parts.join(", ")
    };

    let requested_label = match requested {
        GradientMethod::Auto => "auto",
        GradientMethod::Ad => "AD",
        GradientMethod::Fd => "FD",
    };
    #[cfg(not(feature = "autodiff"))]
    let note = "; autodiff not compiled in";
    #[cfg(feature = "autodiff")]
    let note = "";

    format!("{resolved}  [requested: {requested_label}{note}]")
}

/// Global per-fit timing counters for gradient/Jacobian calls. Printed by
/// [`fit_inner`] when `FERX_TIME_GRADIENTS=1` in the environment. Atomics so
/// multiple rayon workers can update concurrently without locking.
pub(crate) struct GradientTimings {
    pub ad_calls: AtomicU64,
    pub ad_nanos: AtomicU64,
    pub fd_calls: AtomicU64,
    pub fd_nanos: AtomicU64,
    pub jac_ad_calls: AtomicU64,
    pub jac_ad_nanos: AtomicU64,
    pub jac_fd_calls: AtomicU64,
    pub jac_fd_nanos: AtomicU64,
}

impl GradientTimings {
    const fn new() -> Self {
        Self {
            ad_calls: AtomicU64::new(0),
            ad_nanos: AtomicU64::new(0),
            fd_calls: AtomicU64::new(0),
            fd_nanos: AtomicU64::new(0),
            jac_ad_calls: AtomicU64::new(0),
            jac_ad_nanos: AtomicU64::new(0),
            jac_fd_calls: AtomicU64::new(0),
            jac_fd_nanos: AtomicU64::new(0),
        }
    }
    #[inline]
    fn record_ad(&self, ns: u64) {
        self.ad_calls.fetch_add(1, Ordering::Relaxed);
        self.ad_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_fd(&self, ns: u64) {
        self.fd_calls.fetch_add(1, Ordering::Relaxed);
        self.fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_ad(&self, ns: u64) {
        self.jac_ad_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_ad_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_fd(&self, ns: u64) {
        self.jac_fd_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    pub(crate) fn reset(&self) {
        self.ad_calls.store(0, Ordering::Relaxed);
        self.ad_nanos.store(0, Ordering::Relaxed);
        self.fd_calls.store(0, Ordering::Relaxed);
        self.fd_nanos.store(0, Ordering::Relaxed);
        self.jac_ad_calls.store(0, Ordering::Relaxed);
        self.jac_ad_nanos.store(0, Ordering::Relaxed);
        self.jac_fd_calls.store(0, Ordering::Relaxed);
        self.jac_fd_nanos.store(0, Ordering::Relaxed);
    }
    pub(crate) fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            self.ad_calls.load(Ordering::Relaxed),
            self.ad_nanos.load(Ordering::Relaxed),
            self.fd_calls.load(Ordering::Relaxed),
            self.fd_nanos.load(Ordering::Relaxed),
            self.jac_ad_calls.load(Ordering::Relaxed),
            self.jac_ad_nanos.load(Ordering::Relaxed),
            self.jac_fd_calls.load(Ordering::Relaxed),
            self.jac_fd_nanos.load(Ordering::Relaxed),
        )
    }
}

pub(crate) static GRADIENT_TIMINGS: GradientTimings = GradientTimings::new();

/// Result of inner optimization for a single subject
pub struct EbeResult {
    pub eta: DVector<f64>,
    pub h_matrix: DMatrix<f64>,
    /// True when the optimizer (BFGS or Nelder-Mead) met its tolerance criterion.
    /// False on iteration-limit exit regardless of which optimizer was used.
    pub converged: bool,
    /// True when the BFGS optimizer failed and Nelder-Mead was invoked as fallback.
    pub used_fallback: bool,
    /// L2 gradient norm at the solution; 0.0 when Nelder-Mead was used.
    pub grad_norm: f64,
    pub nll: f64,
    /// Per-occasion kappas (empty when n_kappa == 0).
    /// `kappas[k]` corresponds to the k-th unique occasion (same order as
    /// `split_obs_by_occasion`).
    pub kappas: Vec<DVector<f64>>,
}

/// Aggregate statistics from running the inner loop over all subjects.
#[derive(Debug, Default, Clone)]
pub struct InnerLoopStats {
    /// Subjects whose optimizer did not meet the convergence tolerance.
    pub n_unconverged: usize,
    /// Subjects for which the BFGS→Nelder-Mead fallback was triggered.
    pub n_fallback: usize,
}

/// Find Empirical Bayes Estimates (EBEs) for a single subject via BFGS.
///
/// When `mu_k` is provided (mu-referencing active), the inner optimizer works
/// in psi-space where `psi = eta_true + mu_k`.  The objective is evaluated as
/// `individual_nll(psi - mu_k)`, so the model always receives `eta_true`.
/// Warm starts (in `eta_true` space) are converted to psi-space on entry;
/// the returned EbeResult always holds `eta_true = psi - mu_k`.
///
/// When `mu_k` is None every shift is zero and the behaviour is identical to
/// the original (eta-space) implementation.
pub fn find_ebe(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    eta_init: Option<&[f64]>,
    mu_k: Option<&[f64]>,
) -> EbeResult {
    let n_eta = model.n_eta;

    // ── IOV branch ─────────────────────────────────────────────────────────
    // When the model has kappa declarations AND this subject has occasion labels,
    // optimize over the flat vector [bsv_eta (n_eta), kappa_1 (n_kappa), ..., kappa_K (n_kappa)].
    if model.n_kappa > 0 && !subject.occasions.is_empty() {
        return find_ebe_iov(model, subject, params, max_iter, tol, eta_init, mu_k);
    }

    // mu: shift vector (zeros when no mu-referencing)
    let mu: Vec<f64> = mu_k.map(|m| m.to_vec()).unwrap_or_else(|| vec![0.0; n_eta]);

    // psi_init: warm start converted to psi-space, or prior mode (psi = mu, eta_true = 0)
    let mut psi: Vec<f64> = match eta_init {
        Some(warm) => warm.iter().zip(mu.iter()).map(|(e, m)| e + m).collect(),
        None => mu.clone(),
    };

    // Per-subject scratch buffers, built once and reused across every
    // BFGS line-search obj call and every Jacobian perturbation. The
    // EventSchedule pre-computes the merged event timeline + per-interval
    // infusion-bound construction (subject-static, doesn't depend on
    // theta/eta) so the event_driven_predictions hot path doesn't have
    // to re-sort + re-allocate on every call. The EventPkParams scratch
    // recycles the per-event Vec<PkParams> backing storage.
    //
    // Both are built only when this subject takes the TV-cov event-driven
    // analytical path — for the no-TV fast path the schedule is None and
    // event_driven_predictions is never called.
    let pk_scratch_cell = RefCell::new(pk::EventPkParams::with_capacity_for(subject));
    // Skip the schedule cache when the model declares lagtime: lagtime can
    // be eta-dependent and the schedule bakes per-dose times in, so a
    // cached schedule would go stale as the inner BFGS varies eta. The
    // non-cached path (`event_driven_predictions`) rebuilds the schedule
    // per call using the current per-dose PkParams (which carry lagtime).
    // Reset-bearing subjects (EVID=3/4) also take the event-driven analytical
    // path, so they benefit from a cached schedule too — the schedule now
    // includes reset events.
    let schedule = if (subject.has_tv_covariates() || subject.has_resets())
        && model.ode_spec.is_none()
        && pk::event_driven::supports_event_driven(model.pk_model)
        && !model.has_lagtime()
    {
        Some(pk::event_driven::EventSchedule::for_subject(
            subject,
            model.pk_model,
            &[],
        ))
    } else {
        None
    };

    // Objective in psi-space: model always receives eta_true = psi - mu.
    let obj = |p: &[f64]| -> f64 {
        let mut scratch = pk_scratch_cell.borrow_mut();
        let eta_t: Vec<f64> = p.iter().zip(mu.iter()).map(|(pi, mi)| pi - mi).collect();
        individual_nll_into_with_schedule(
            model,
            subject,
            &params.theta,
            &eta_t,
            &params.omega,
            &params.sigma.values,
            &mut scratch,
            schedule.as_ref(),
        )
    };

    // Resolve Auto → concrete method based on model/eta characteristics.
    // Autodiff is only available when the crate was compiled with the feature
    // and the model provides tv_fn (the parser attaches it for analytical PK).
    let grad_method = resolve_gradient_method(model, subject);

    // ── Per-subject AD helpers, built ONCE per find_ebe call ──
    //
    // theta is fixed for the whole inner-loop (BFGS gradient calls) AND
    // for the post-convergence Jacobian, so all the per-event tv arrays
    // and dose-flat arrays are stable across both. Hoist them out of
    // the inner closures and out of the Jacobian site so each helper is
    // only constructed once per outer iteration per subject — was twice
    // before this refactor, with `pk_param_fn` re-evaluated per event.
    #[cfg(feature = "autodiff")]
    let (
        ad_dose_data,
        ad_omega_inv_flat,
        ad_log_det_omega,
        ad_cens_f64,
        ad_tv_adjusted,
        ad_event_data,
        ad_tv_per_event,
    ) = if grad_method != InnerGradientMethod::Fd {
        let dose_data = FlatDoseData::from_subject(subject);
        let omega_inv = params
            .omega
            .matrix
            .clone()
            .cholesky()
            .map(|c| c.inverse())
            .unwrap_or_else(|| nalgebra::DMatrix::identity(n_eta, n_eta));
        let mut omega_inv_flat = Vec::with_capacity(n_eta * n_eta);
        for i in 0..n_eta {
            for j in 0..n_eta {
                omega_inv_flat.push(omega_inv[(i, j)]);
            }
        }
        // Same early-return on degenerate omega as before — must run
        // before any heavy helper allocation.
        let log_det_omega = {
            let mut ld = 0.0;
            for i in 0..n_eta {
                let lii = params.omega.chol[(i, i)];
                ld += if lii > 0.0 {
                    lii.ln()
                } else {
                    return EbeResult {
                        eta: DVector::zeros(n_eta),
                        h_matrix: DMatrix::zeros(0, 0),
                        converged: false,
                        used_fallback: false,
                        grad_norm: 0.0,
                        nll: 1e20,
                        kappas: Vec::new(),
                    };
                };
            }
            2.0 * ld
        };
        // Under M3, feed actual CENS flags so the AD path applies
        // -log Φ to censored rows. Otherwise pass zeros — Enzyme
        // traces the Gaussian branch for every observation.
        let cens_f64: Vec<f64> = if matches!(model.bloq_method, BloqMethod::M3) {
            subject.cens.iter().map(|&c| c as f64).collect()
        } else {
            vec![0.0; subject.observations.len()]
        };
        // Per-method helpers — only one of the two is built.
        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("resolve_gradient_method guarantees tv_fn for AD branches");
        let (tv_adjusted, event_data, tv_per_event) = match grad_method {
            InnerGradientMethod::AdSingleSnapshot => {
                (Some(tv_fn(&params.theta, &subject.covariates)), None, None)
            }
            InnerGradientMethod::AdEventDriven => (
                None,
                Some(crate::ad::event_driven_ad::FlatEventData::from_subject(
                    subject,
                )),
                Some(crate::ad::event_driven_ad::FlatEventTv::from_subject(
                    model,
                    subject,
                    &params.theta,
                )),
            ),
            InnerGradientMethod::Fd => (None, None, None),
        };
        (
            Some(dose_data),
            Some(omega_inv_flat),
            Some(log_det_omega),
            Some(cens_f64),
            tv_adjusted,
            event_data,
            tv_per_event,
        )
    } else {
        (None, None, None, None, None, None, None)
    };

    // Try BFGS — AD gradient when `grad_method` is one of the AD variants,
    // FD otherwise. The AD gradient of individual_nll w.r.t. psi equals the
    // gradient w.r.t. eta_true (chain rule: d/dpsi = d/d(eta_true), since
    // psi = eta_true + mu).
    #[cfg(feature = "autodiff")]
    let result = if grad_method != InnerGradientMethod::Fd {
        let dose_data = ad_dose_data.as_ref().unwrap();
        let omega_inv_flat = ad_omega_inv_flat.as_ref().unwrap();
        let log_det_omega = ad_log_det_omega.unwrap();
        let cens_f64 = ad_cens_f64.as_ref().unwrap();
        let mu_ad = mu.clone();

        match grad_method {
            InnerGradientMethod::AdSingleSnapshot => {
                let tv_adjusted = ad_tv_adjusted.as_ref().unwrap();
                let grad_fn = |p: &[f64]| -> Vec<f64> {
                    let eta_t: Vec<f64> =
                        p.iter().zip(mu_ad.iter()).map(|(pi, mi)| pi - mi).collect();
                    let t0 = std::time::Instant::now();
                    let obs_scale = build_scale_array_for_ad(model, subject, &params.theta, &eta_t);
                    let (_, g) = ad_gradients::compute_nll_gradient_ad(
                        &eta_t,
                        tv_adjusted,
                        omega_inv_flat,
                        log_det_omega,
                        &params.sigma.values,
                        dose_data,
                        &subject.obs_times,
                        &subject.observations,
                        cens_f64,
                        model.pk_model,
                        model.error_model,
                        &model.pk_idx_f64,
                        &model.sel_flat,
                        &obs_scale,
                        model.log_transform,
                    );
                    GRADIENT_TIMINGS.record_ad(t0.elapsed().as_nanos() as u64);
                    g
                };
                bfgs_minimize_with_grad(&obj, &grad_fn, &mut psi, n_eta, max_iter, tol)
            }
            InnerGradientMethod::AdEventDriven => {
                let event_data = ad_event_data.as_ref().unwrap();
                let tv_per_event = ad_tv_per_event.as_ref().unwrap();
                let grad_fn = |p: &[f64]| -> Vec<f64> {
                    let eta_t: Vec<f64> =
                        p.iter().zip(mu_ad.iter()).map(|(pi, mi)| pi - mi).collect();
                    let t0 = std::time::Instant::now();
                    let event_scale = build_event_scale_array_for_ad(
                        model,
                        subject,
                        event_data,
                        &params.theta,
                        &eta_t,
                    );
                    let (_, g) = crate::ad::event_driven_ad::compute_nll_gradient_event_driven_ad(
                        &eta_t,
                        tv_per_event,
                        omega_inv_flat,
                        log_det_omega,
                        &params.sigma.values,
                        event_data,
                        &subject.observations,
                        cens_f64,
                        model.pk_model,
                        model.error_model,
                        &model.pk_idx_f64,
                        &model.sel_flat,
                        &event_scale,
                        model.log_transform,
                    );
                    GRADIENT_TIMINGS.record_ad(t0.elapsed().as_nanos() as u64);
                    g
                };
                bfgs_minimize_with_grad(&obj, &grad_fn, &mut psi, n_eta, max_iter, tol)
            }
            InnerGradientMethod::Fd => unreachable!("guarded above"),
        }
    } else {
        bfgs_minimize(&obj, &mut psi, n_eta, max_iter, tol)
    };

    #[cfg(not(feature = "autodiff"))]
    let result = {
        let _ = grad_method; // silence unused warning on stable builds
        bfgs_minimize(&obj, &mut psi, n_eta, max_iter, tol)
    };

    // If BFGS failed, try Nelder-Mead from the prior mode (psi = mu, eta_true = 0)
    let bfgs_converged = result;
    let (nm_converged, used_fallback) = if !bfgs_converged {
        psi = mu.clone();
        let nm_ok = nelder_mead_minimize(&obj, &mut psi, n_eta, max_iter * 5, tol);
        (nm_ok, true)
    } else {
        (false, false)
    };

    let ebe_converged = bfgs_converged || nm_converged;
    let nll = obj(&psi);

    // Recover eta_true = psi - mu (mean-zero, NONMEM-compatible output)
    let eta_true: Vec<f64> = psi.iter().zip(mu.iter()).map(|(p, m)| p - m).collect();

    // Compute Jacobian at eta_true — match the gradient path so the H matrix
    // is consistent with the gradient that drove convergence. Reuses the
    // same per-subject helpers built once at the top of find_ebe; previously
    // these were rebuilt here, doubling the per-subject helper cost.
    #[cfg(feature = "autodiff")]
    let h_matrix = match grad_method {
        InnerGradientMethod::AdSingleSnapshot => {
            let tv_adjusted = ad_tv_adjusted.as_ref().unwrap();
            let dose_data = ad_dose_data.as_ref().unwrap();
            let t0 = std::time::Instant::now();
            let obs_scale = build_scale_array_for_ad(model, subject, &params.theta, &eta_true);
            let j = ad_gradients::compute_jacobian_ad(
                &eta_true,
                tv_adjusted,
                dose_data,
                &subject.obs_times,
                subject.obs_times.len(),
                model.pk_model,
                &model.pk_idx_f64,
                &model.sel_flat,
                &obs_scale,
                model.log_transform,
            );
            GRADIENT_TIMINGS.record_jac_ad(t0.elapsed().as_nanos() as u64);
            j
        }
        InnerGradientMethod::AdEventDriven => {
            // Forward-mode AD Jacobian — kernel lives in
            // `ad::event_driven_ad_jac` (sibling module so the AD pass
            // stays isolated from the reverse-mode NLL pass; sharing
            // helpers tripped Enzyme's reverse-mode type deduction).
            let event_data = ad_event_data.as_ref().unwrap();
            let tv_per_event = ad_tv_per_event.as_ref().unwrap();
            let t0 = std::time::Instant::now();
            let event_scale = build_event_scale_array_for_ad(
                model,
                subject,
                event_data,
                &params.theta,
                &eta_true,
            );
            let j = crate::ad::event_driven_ad_jac::compute_jacobian_event_driven_ad(
                &eta_true,
                tv_per_event,
                event_data,
                subject.obs_times.len(),
                model.pk_model,
                &model.pk_idx_f64,
                &model.sel_flat,
                &event_scale,
                model.log_transform,
            );
            GRADIENT_TIMINGS.record_jac_ad(t0.elapsed().as_nanos() as u64);
            j
        }
        InnerGradientMethod::Fd => {
            let mut scratch = pk_scratch_cell.borrow_mut();
            let t0 = std::time::Instant::now();
            let j = compute_jacobian_fd(
                model,
                subject,
                &params.theta,
                &eta_true,
                &mut scratch,
                schedule.as_ref(),
            );
            GRADIENT_TIMINGS.record_jac_fd(t0.elapsed().as_nanos() as u64);
            j
        }
    };

    #[cfg(not(feature = "autodiff"))]
    let h_matrix = {
        let mut scratch = pk_scratch_cell.borrow_mut();
        let t0 = std::time::Instant::now();
        let j = compute_jacobian_fd(
            model,
            subject,
            &params.theta,
            &eta_true,
            &mut scratch,
            schedule.as_ref(),
        );
        GRADIENT_TIMINGS.record_jac_fd(t0.elapsed().as_nanos() as u64);
        j
    };

    EbeResult {
        eta: DVector::from_column_slice(&eta_true),
        h_matrix,
        converged: ebe_converged,
        used_fallback,
        grad_norm: 0.0, // not computed to avoid extra FD calls; available via nll.is_finite()
        nll,
        kappas: Vec::new(),
    }
}

/// IOV inner optimizer: optimizes [bsv_psi, kappa_1, ..., kappa_K] jointly,
/// where bsv_psi = bsv_eta + mu (matches the non-IOV path's mu-referencing
/// shift). Kappas are zero-centered IOV draws and are not mu-shifted.
/// Forces FD gradient (no AD path for IOV in Option A).
///
/// When `mu_k` is provided the BSV block is optimised in psi-space
/// (`psi = eta_true + mu_k`) so mu-referencing benefits also apply to the BSV
/// etas when IOV is active.  The returned `EbeResult.eta` is always `eta_true`.
fn find_ebe_iov(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    eta_init: Option<&[f64]>,
    mu_k: Option<&[f64]>,
) -> EbeResult {
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;

    let occ_groups = split_obs_by_occasion(subject);
    let k_occasions = occ_groups.len();

    let n_flat = n_eta + k_occasions * n_kappa;

    // BSV mu shift (zeros when no mu-referencing). Kappas are not shifted.
    let mu: Vec<f64> = mu_k.map(|m| m.to_vec()).unwrap_or_else(|| vec![0.0; n_eta]);

    // Initial flat vector: BSV portion is psi-space (warm + mu, defaulting
    // to mu = prior mode); kappa portion starts at zero (prior mode for IOV).
    let mut x = vec![0.0; n_flat];
    x[..n_eta].copy_from_slice(&mu);
    if let Some(warm) = eta_init {
        for i in 0..n_eta.min(warm.len()) {
            x[i] = warm[i] + mu[i];
        }
    }

    let omega_iov_ref = params.omega_iov.as_ref();

    let obj = |p: &[f64]| -> f64 {
        // Recover bsv_eta = psi - mu; kappas pass through unchanged.
        let eta_t: Vec<f64> = p[..n_eta]
            .iter()
            .zip(mu.iter())
            .map(|(pi, mi)| pi - mi)
            .collect();
        let kappas: Vec<Vec<f64>> = (0..k_occasions)
            .map(|k| p[n_eta + k * n_kappa..n_eta + (k + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            model,
            subject,
            &params.theta,
            &eta_t,
            &kappas,
            &params.omega,
            omega_iov_ref,
            &params.sigma.values,
        )
    };

    let bfgs_converged = bfgs_minimize(&obj, &mut x, n_flat, max_iter, tol);
    let (nm_converged, used_fallback) = if !bfgs_converged {
        // Reset to prior mode: bsv_psi = mu (eta_true = 0), kappas = 0.
        x = vec![0.0; n_flat];
        x[..n_eta].copy_from_slice(&mu);
        let nm_ok = nelder_mead_minimize(&obj, &mut x, n_flat, max_iter * 5, tol);
        (nm_ok, true)
    } else {
        (false, false)
    };

    let nll = obj(&x);
    // Recover bsv_eta = psi - mu (mean-zero, NONMEM-compatible output).
    let bsv_eta: Vec<f64> = x[..n_eta]
        .iter()
        .zip(mu.iter())
        .map(|(p, m)| p - m)
        .collect();
    let kappas_vec: Vec<DVector<f64>> = (0..k_occasions)
        .map(|k| DVector::from_column_slice(&x[n_eta + k * n_kappa..n_eta + (k + 1) * n_kappa]))
        .collect();

    // H-matrix: BSV columns only, perturbing eta with kappas fixed at EBE values
    let kappas_slices: Vec<Vec<f64>> = kappas_vec.iter().map(|k| k.as_slice().to_vec()).collect();
    let h_matrix = compute_jacobian_fd_iov(model, subject, &params.theta, &bsv_eta, &kappas_slices);

    EbeResult {
        eta: DVector::from_column_slice(&bsv_eta),
        h_matrix,
        converged: (bfgs_converged || nm_converged) && nll.is_finite(),
        used_fallback,
        grad_norm: 0.0,
        nll,
        kappas: kappas_vec,
    }
}

/// Jacobian d(pred)/d(bsv_eta) with kappas fixed. Returns an n_obs × n_eta
/// matrix.
///
/// Uses the continuous, per-occasion-aware prediction (`pk::predict_iov`), so a
/// BSV-eta perturbation flows through the whole timeline (it shifts every
/// occasion's clearance) and the column is dense across rows — consistent with
/// the NLL value in `individual_nll_iov`, which uses the same prediction. The
/// occasion list is recovered inside `predict_iov`, so `occ_groups` is no longer
/// needed here. See issue #104.
fn compute_jacobian_fd_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
) -> DMatrix<f64> {
    let n_obs = subject.obs_times.len();
    let n_eta = eta.len();
    let eps = 1e-6;
    let mut h = DMatrix::zeros(n_obs, n_eta);
    let mut eta_pert = eta.to_vec();

    for col in 0..n_eta {
        let h_step = eps * (1.0 + eta[col].abs());
        eta_pert[col] = eta[col] + h_step;
        let preds_plus = pk::predict_iov(model, subject, theta, &eta_pert, kappas);
        eta_pert[col] = eta[col] - h_step;
        let preds_minus = pk::predict_iov(model, subject, theta, &eta_pert, kappas);
        eta_pert[col] = eta[col];

        let inv = 1.0 / (2.0 * h_step);
        for j in 0..n_obs {
            h[(j, col)] = (preds_plus[j] - preds_minus[j]) * inv;
        }
    }

    h
}

/// BFGS minimization with backtracking line search.
/// Uses analytical-style gradient via forward FD with small step.
fn bfgs_minimize(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = gradient_fd(obj, x, n);
    let mut first_step = true;

    for _iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt();

        // Scale initial Hessian so first step is O(1) not O(gnorm)
        if first_step && gnorm > 1.0 {
            let scale = 1.0 / gnorm;
            h_inv *= scale;
            first_step = false;
        }
        if gnorm < tol {
            return true;
        }

        // Search direction
        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 {
            // Reset to steepest descent
            h_inv = DMatrix::identity(n, n);
            let d: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let alpha = backtracking_line_search(obj, x, &d, &g, n);
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            g = gradient_fd(obj, x, n);
            continue;
        }

        let alpha = backtracking_line_search(obj, x, &d, &g, n);
        if alpha < 1e-16 {
            return false;
        }

        // s = alpha * d
        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }

        let g_new = gradient_fd(obj, x, n);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        // BFGS update
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let sy = s_vec.dot(&y_vec);
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            let eye = DMatrix::identity(n, n);
            let s_yt = rho * &s_vec * y_vec.transpose();
            let y_st = rho * &y_vec * s_vec.transpose();
            let s_st = rho * &s_vec * s_vec.transpose();
            h_inv = (&eye - &s_yt) * &h_inv * (&eye - &y_st) + s_st;
        }

        g = g_new;
    }

    false
}

/// BFGS minimization with an externally-provided gradient function (for AD).
#[cfg(feature = "autodiff")]
fn bfgs_minimize_with_grad(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = grad(x);
    let mut first_step = true;

    for _iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt();

        if first_step && gnorm > 1.0 {
            let scale = 1.0 / gnorm;
            h_inv *= scale;
            first_step = false;
        }

        if gnorm < tol {
            return true;
        }

        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 {
            h_inv = DMatrix::identity(n, n);
            let d: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let alpha = backtracking_line_search(obj, x, &d, &g, n);
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            g = grad(x);
            continue;
        }

        let alpha = backtracking_line_search(obj, x, &d, &g, n);
        if alpha < 1e-16 {
            return false;
        }

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }

        let g_new = grad(x);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let sy = s_vec.dot(&y_vec);
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            let eye = DMatrix::identity(n, n);
            let s_yt = rho * &s_vec * y_vec.transpose();
            let y_st = rho * &y_vec * s_vec.transpose();
            let s_st = rho * &s_vec * s_vec.transpose();
            h_inv = (&eye - &s_yt) * &h_inv * (&eye - &y_st) + s_st;
        }

        g = g_new;
    }

    false
}

/// Nelder-Mead simplex minimization (fallback)
fn nelder_mead_minimize(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x.to_vec());
    for i in 0..n {
        let mut point = x.to_vec();
        let delta = if point[i].abs() > 1e-8 {
            0.05 * point[i].abs()
        } else {
            0.00025
        };
        point[i] += delta;
        simplex.push(point);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|p| obj(p)).collect();

    for _iter in 0..max_iter {
        let mut indices: Vec<usize> = (0..=n).collect();
        // NaN-safe: a non-finite objective (e.g. an ODE prediction that blew
        // up at a simplex vertex) sorts as worst rather than panicking on the
        // `None` that `partial_cmp` returns for NaN. See issue #97.
        indices.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Greater)
        });

        let best = indices[0];
        let worst = indices[n];
        let second_worst = indices[n - 1];

        let frange = fvals[worst] - fvals[best];
        if frange < tol {
            x.copy_from_slice(&simplex[best]);
            return true;
        }

        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst][j]))
            .collect();
        let fr = obj(&reflected);

        if fr < fvals[second_worst] && fr >= fvals[best] {
            simplex[worst] = reflected;
            fvals[worst] = fr;
            continue;
        }

        if fr < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let fe = obj(&expanded);
            if fe < fr {
                simplex[worst] = expanded;
                fvals[worst] = fe;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = fr;
            }
            continue;
        }

        let contracted: Vec<f64> = (0..n)
            .map(|j| centroid[j] + rho * (simplex[worst][j] - centroid[j]))
            .collect();
        let fc = obj(&contracted);
        if fc < fvals[worst] {
            simplex[worst] = contracted;
            fvals[worst] = fc;
            continue;
        }

        let best_point = simplex[best].clone();
        for i in 0..=n {
            if i != best {
                for j in 0..n {
                    simplex[i][j] = best_point[j] + sigma * (simplex[i][j] - best_point[j]);
                }
                fvals[i] = obj(&simplex[i]);
            }
        }
    }

    // NaN-safe min: a non-finite vertex objective must not panic here either.
    let best = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
        .map(|(i, _)| i)
        .unwrap();
    x.copy_from_slice(&simplex[best]);
    false
}

/// Backtracking line search with Armijo condition
fn backtracking_line_search(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    n: usize,
) -> f64 {
    let c1 = 1e-4;
    let shrink = 0.5;
    let mut alpha = 1.0;
    let f0 = obj(x);
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();

    let mut x_new = vec![0.0; n];
    for _ in 0..40 {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = obj(&x_new);
        if f_new <= f0 + c1 * alpha * dg {
            return alpha;
        }
        alpha *= shrink;
    }
    alpha
}

/// Central finite difference gradient (optimized step size)
fn gradient_fd(obj: &dyn Fn(&[f64]) -> f64, x: &[f64], n: usize) -> Vec<f64> {
    let t0 = std::time::Instant::now();
    let mut g = vec![0.0; n];
    let mut x_work = x.to_vec();
    for i in 0..n {
        let h = 1e-7 * (1.0 + x[i].abs());
        x_work[i] = x[i] + h;
        let fp = obj(&x_work);
        x_work[i] = x[i] - h;
        let fm = obj(&x_work);
        g[i] = (fp - fm) / (2.0 * h);
        x_work[i] = x[i];
    }
    GRADIENT_TIMINGS.record_fd(t0.elapsed().as_nanos() as u64);
    g
}

/// Compute Jacobian H = d(predictions)/d(eta) via finite differences.
/// H is n_obs x n_eta.
///
/// Reuses a caller-owned `EventPkParams` scratch and an optional
/// pre-built `EventSchedule` so each of the `2 * n_eta` perturbed
/// prediction calls avoids the per-event-param Vec allocation and
/// the per-call event-merge sort.
fn compute_jacobian_fd(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> DMatrix<f64> {
    let n_obs = subject.obs_times.len();
    let n_eta = eta.len();
    let eps = 1e-6;

    let mut h = DMatrix::zeros(n_obs, n_eta);
    let mut eta_pert = eta.to_vec();

    for j in 0..n_eta {
        let h_step = eps * (1.0 + eta[j].abs());

        eta_pert[j] = eta[j] + h_step;
        let preds_plus = pk::compute_predictions_with_tv_into_with_schedule(
            model, subject, theta, &eta_pert, scratch, schedule,
        );

        eta_pert[j] = eta[j] - h_step;
        let preds_minus = pk::compute_predictions_with_tv_into_with_schedule(
            model, subject, theta, &eta_pert, scratch, schedule,
        );

        for i in 0..n_obs {
            h[(i, j)] = (preds_plus[i] - preds_minus[i]) / (2.0 * h_step);
        }

        eta_pert[j] = eta[j];
    }

    h
}

/// Run inner loop for all subjects (parallel via rayon).
/// Warm-starts from previous EBEs when available.
pub fn run_inner_loop(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
) -> (
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    InnerLoopStats,
    Vec<Vec<DVector<f64>>>,
) {
    run_inner_loop_warm(model, population, params, max_iter, tol, None, None, 0)
}

/// Run inner loop with optional warm-start EBEs and optional mu-referencing shift.
///
/// `prev_etas` — previous-iteration EBEs in eta_true space (used as warm starts).
/// `mu_k`      — mu shift vector from `compute_mu_k`; `None` means no mu-referencing.
/// `min_obs`   — subjects with fewer observations than this are excluded from the
///               `n_unconverged` count in `InnerLoopStats` (but still run normally).
///               Pass `0` to count all subjects regardless of observation count.
///
/// Returns `(eta_hats, h_matrices, stats, kappas_per_subject)`.
/// `kappas_per_subject[i]` contains per-occasion kappa EBEs for subject i; it is
/// empty for non-IOV subjects or when `model.n_kappa == 0`.
pub fn run_inner_loop_warm(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    prev_etas: Option<&[DVector<f64>]>,
    mu_k: Option<&[f64]>,
    min_obs: usize,
) -> (
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    InnerLoopStats,
    Vec<Vec<DVector<f64>>>,
) {
    use rayon::prelude::*;

    let results: Vec<EbeResult> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let init = prev_etas.map(|pe| pe[i].as_slice());
            find_ebe(model, subject, params, max_iter, tol, init, mu_k)
        })
        .collect();

    let stats = InnerLoopStats {
        n_unconverged: results
            .iter()
            .zip(population.subjects.iter())
            .filter(|(r, s)| !r.converged && s.observations.len() >= min_obs.max(1))
            .count(),
        n_fallback: results.iter().filter(|r| r.used_fallback).count(),
    };
    let eta_hats: Vec<DVector<f64>> = results.iter().map(|r| r.eta.clone()).collect();
    let h_matrices: Vec<DMatrix<f64>> = results.iter().map(|r| r.h_matrix.clone()).collect();
    let kappas: Vec<Vec<DVector<f64>>> = results.into_iter().map(|r| r.kappas).collect();

    (eta_hats, h_matrices, stats, kappas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_loop_stats_default() {
        let s = InnerLoopStats::default();
        assert_eq!(s.n_unconverged, 0);
        assert_eq!(s.n_fallback, 0);
    }

    #[test]
    fn test_ebe_result_converged_flag() {
        // Verify EbeResult struct has the expected fields.
        let r = EbeResult {
            eta: nalgebra::DVector::zeros(2),
            h_matrix: nalgebra::DMatrix::identity(2, 2),
            converged: true,
            used_fallback: false,
            grad_norm: 0.0,
            nll: 1.5,
            kappas: Vec::new(),
        };
        assert!(r.converged);
        assert!(!r.used_fallback);
        assert_eq!(r.grad_norm, 0.0);
    }

    #[test]
    fn test_inner_loop_stats_min_obs_filter() {
        // min_obs filter: subjects with fewer obs than min_obs are excluded
        // from n_unconverged count. We exercise this logic by constructing
        // InnerLoopStats manually (simulating what run_inner_loop_warm does).
        let results = vec![
            EbeResult {
                eta: nalgebra::DVector::zeros(1),
                h_matrix: nalgebra::DMatrix::identity(1, 1),
                converged: false, // unconverged
                used_fallback: false,
                grad_norm: 0.0,
                nll: 1.0,
                kappas: Vec::new(),
            },
            EbeResult {
                eta: nalgebra::DVector::zeros(1),
                h_matrix: nalgebra::DMatrix::identity(1, 1),
                converged: false, // also unconverged
                used_fallback: true,
                grad_norm: 0.0,
                nll: 2.0,
                kappas: Vec::new(),
            },
        ];
        // Simulate filter: first subject has 1 obs (below min_obs=2), second has 3 obs.
        let obs_counts = [1_usize, 3_usize];
        let min_obs = 2_usize;
        let n_unconverged = results
            .iter()
            .zip(obs_counts.iter())
            .filter(|(r, &n_obs)| !r.converged && n_obs >= min_obs.max(1))
            .count();
        let n_fallback = results.iter().filter(|r| r.used_fallback).count();
        // Only second subject counts (3 obs >= 2); first is filtered out.
        assert_eq!(n_unconverged, 1);
        // Both fallback counts regardless of min_obs.
        assert_eq!(n_fallback, 1);
    }

    #[test]
    fn test_nelder_mead_nan_objective_does_not_panic() {
        // Regression for issue #97: when a simplex vertex evaluates to a NaN
        // objective (e.g. an ODE prediction blowing up during the EBE search),
        // the `partial_cmp().unwrap()` sort used to panic — and, unwinding
        // through the non-unwinding optimizer callback, abort the whole fit.
        // NaN must now sort as worst and get reflected away instead.
        let obj = |x: &[f64]| -> f64 {
            if x[0] < 0.0 {
                // The "blow-up" region: objective is non-finite here.
                f64::NAN
            } else {
                (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)
            }
        };
        // Seed the simplex entirely inside the NaN region so the very first
        // sort encounters only NaN vertices.
        let mut x = vec![-1.0, -1.0];
        // The contract under test is "does not panic"; the return flag and
        // final point are secondary. Coordinates must stay finite.
        let _converged = nelder_mead_minimize(&obj, &mut x, 2, 200, 1e-8);
        assert!(
            x.iter().all(|v| v.is_finite()),
            "Nelder-Mead must leave the point finite, got {x:?}"
        );
    }
}

#[cfg(test)]
mod iov_tests {
    use super::*;
    use crate::types::{
        BloqMethod, DoseEvent, ErrorModel, GradientMethod, OmegaMatrix, PkModel, PkParams,
        SigmaVector,
    };
    use std::collections::HashMap;

    #[test]
    fn gradient_route_summary_reports_route_taken_not_requested() {
        // make_iov_model has `tv_fn: None` and the default `gradient_method:
        // Auto`. With no `tv_fn`, AD is unavailable, so the route resolves to
        // FD in every build — even though the *requested* method is `auto`.
        // The banner must report the route actually taken (FD) and surface the
        // request, so a silent AD→FD fallback is visible.
        let model = make_iov_model();
        let population = Population {
            subjects: vec![make_iov_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
        };
        // `requested` is the user's FitOptions value, passed independently of
        // model.gradient_method (which compatibility rules may have mutated).
        let summary = gradient_route_summary(&model, &population, GradientMethod::Auto);
        assert!(
            summary.starts_with("FD"),
            "tv_fn=None must resolve to FD, got: {summary}"
        );
        // Matches both "[requested: auto]" (autodiff build) and
        // "[requested: auto; autodiff not compiled in]" (ci build).
        assert!(
            summary.contains("[requested: auto"),
            "summary must surface the requested method, got: {summary}"
        );
        // The bracket reflects the passed `requested`, not model.gradient_method
        // — guards against regressing to the SDE-mislabel Copilot flagged on #117.
        let fd_summary = gradient_route_summary(&model, &population, GradientMethod::Fd);
        assert!(
            fd_summary.contains("[requested: FD"),
            "bracket must echo the requested arg, got: {fd_summary}"
        );
    }

    fn make_iov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = crate::types::ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.01, 1.0],
            theta_upper: vec![100.0, 500.0],
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
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                // eta[0] = bsv, eta[1] = kappa (combined)
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
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
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        }
    }

    fn make_iov_subject() -> Subject {
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            observations: vec![40.0, 32.0, 25.0, 38.0, 30.0, 22.0],
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
        }
    }

    #[test]
    fn test_find_ebe_iov_two_occasions_returns_two_kappas() {
        let model = make_iov_model();
        let subject = make_iov_subject();
        let params = model.default_params.clone();
        let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None);
        assert_eq!(result.kappas.len(), 2, "Expected 2 kappas for 2 occasions");
        assert_eq!(result.kappas[0].len(), 1);
        assert_eq!(result.kappas[1].len(), 1);
        assert!(result.converged || result.nll.is_finite());
    }

    #[test]
    fn test_find_ebe_iov_h_matrix_dimensions() {
        let model = make_iov_model();
        let subject = make_iov_subject();
        let params = model.default_params.clone();
        let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None);
        // H-matrix: n_obs × n_eta (BSV only, kappas fixed)
        assert_eq!(result.h_matrix.nrows(), subject.obs_times.len());
        assert_eq!(result.h_matrix.ncols(), model.n_eta);
    }

    #[test]
    fn test_find_ebe_no_iov_kappas_empty() {
        // A model without IOV should return empty kappas
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let default_params = crate::types::ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.01, 1.0],
            theta_upper: vec![100.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "no_iov".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
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
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        };
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0],
            observations: vec![40.0, 32.0, 20.0],
            obs_cmts: vec![1; 3],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 3],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        };
        let params = model.default_params.clone();
        let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None);
        assert!(result.kappas.is_empty());
    }

    #[test]
    fn test_find_ebe_iov_honors_mu_shift() {
        // With mu-referencing, the IOV inner loop must shift its BSV optimization
        // variable by mu so the returned EBE is mean-zero (psi - mu), matching
        // the non-IOV path's NONMEM-compatible convention. Two equivalent fits
        // — same data, same params, but expressed with vs. without a mu shift —
        // should yield essentially the same returned BSV eta.
        let model = make_iov_model();
        let subject = make_iov_subject();
        let params = model.default_params.clone();

        // Fit without mu_k.
        let r1 = find_ebe(&model, &subject, &params, 200, 1e-5, None, None);

        // Fit with a non-zero mu_k. If mu were dropped, BSV eta would shift by
        // -mu; with the fix, BSV eta is recovered as psi - mu and matches r1.
        let mu = vec![0.1];
        let r2 = find_ebe(&model, &subject, &params, 200, 1e-5, None, Some(&mu));

        assert!(r1.converged && r2.converged);
        assert!(
            (r1.eta[0] - r2.eta[0]).abs() < 1e-4,
            "mu shift not applied: r1.eta={}, r2.eta={}",
            r1.eta[0],
            r2.eta[0],
        );
    }
}
