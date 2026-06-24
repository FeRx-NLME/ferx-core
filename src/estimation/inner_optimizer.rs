use crate::pk;
use crate::stats::likelihood::{
    individual_nll_into_with_schedule, individual_nll_iov, split_obs_by_occasion,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The inner-loop η-gradient route resolved for a subject. Reported in the
/// startup banner ([`gradient_route_summary`]) and used by [`find_ebe`].
///
/// The Enzyme AD path was retired; the two live routes are the exact analytic
/// `Dual2` η-gradient ([`analytic_eta_nll_gradient`]) and central finite
/// differences. The choice is [`analytic_inner_grad_supported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InnerGradientMethod {
    /// Exact analytic η-gradient from the `Dual2` sensitivity provider — one
    /// provider evaluation per inner step (vs FD's `~2·n_eta+1` predictions).
    Analytic,
    /// Central finite differences. Used when the provider can't serve the model
    /// (ODE, LTBS, expression scaling, time-varying covariates, SDE) or the
    /// `FERX_NO_ANALYTIC_INNER` escape hatch is set.
    Fd,
}

/// Model-level features that classify a model as outside the closed-form
/// inner-gradient scope, returning `Some(reason)` (else `None`). Historically
/// gated the retired Enzyme AD inner gradient; retained as a named-reason
/// classifier (the live scope check is [`analytic_inner_grad_supported`]).
#[allow(dead_code)]
pub(crate) fn analytical_ad_unsupported(model: &CompiledModel) -> Option<&'static str> {
    // Non-log-normal ETA: additive (`tv + eta`), logit (`inv_logit(... + eta)`),
    // logit-probability, or custom/unrecognised. The kernels apply `exp(eta)`
    // unconditionally and ignore `EtaParamType`.
    if model
        .eta_param_info
        .iter()
        .any(|e| e.param_type != EtaParamType::LogNormal)
    {
        return Some("non-log-normal ETA parameterisation");
    }
    // Log-transform-both-sides (`log_additive`, `log(DV) ~ ...`). The `+100`
    // log-wrap Jacobian diverges from the FD reference: small on well-conditioned
    // data, but on ill-conditioned FOCEI-INTER fits it drives a spurious
    // variance-collapsed optimum (the symptom that surfaced this, ferx-r#154).
    if model.log_transform {
        return Some("log-transform-both-sides (LTBS) error model");
    }
    // Conditional individual-parameter expressions, e.g.
    // `if (WT > 70) { CL = TVCL * (WT/70)^0.75 * exp(ETA_CL) } else { ... }`.
    // The ETA stays log-normal so `eta_param_info` looks ordinary, but the
    // parameter is assigned inside an `if`-branch the analytical kernels can't
    // represent. The parser sets this flag (and also disables mu-referencing)
    // when an if-branch assigns an eta-bearing parameter.
    if model.has_conditional_eta_params {
        return Some("conditional (if-branch) individual-parameter expression");
    }
    // Eta-dependent `[scaling] obs_scale` expression. `build_obs_scale_array`
    // freezes the scale subject-static, so the AD Jacobian drops
    // `d obs_scale / d eta` (see `ScalingSpec::breaks_ad_inner_gradient`).
    if model.scaling.breaks_ad_inner_gradient() {
        return Some("eta-dependent obs_scale (ExpressionScale)");
    }
    // Time-to-event (`[event_model]`) endpoint. The analytical single-snapshot
    // AD kernel computes the PK-observation NLL, not the hazard/survival
    // likelihood, so the eta-gradient through the hazard (especially shape
    // params) is wrong - `tte_weibull` / `tte_gompertz` diverged ~2-5 OFV from
    // FD under AD. Route TTE models to FD. (Always false without `survival`.)
    if model.has_tte() {
        return Some("time-to-event ([event_model]) hazard likelihood");
    }
    // IIV on residual error (`Y = IPRED + EPS*EXP(ETA)`). The analytical kernels
    // build the residual variance from σ alone and have no rule for the
    // per-subject `exp(2·η_ruv)` scaling; the EBE search must see the scaled
    // variance, so route these models to FD (which differences the real
    // `individual_nll`, where the scale is applied). See #409.
    if model.residual_error_eta.is_some() {
        return Some("IIV on residual error (iiv_on_ruv)");
    }
    None
}

pub(crate) fn resolve_gradient_method(
    model: &CompiledModel,
    subject: &Subject,
) -> InnerGradientMethod {
    if analytic_inner_grad_supported(model, subject) {
        InnerGradientMethod::Analytic
    } else {
        InnerGradientMethod::Fd
    }
}

/// One-line summary of the inner-loop gradient route **actually resolved**
/// across the population, for the startup banner. Reflects the per-subject
/// resolution in [`resolve_gradient_method`] — the analytic `Dual2` η-gradient
/// where it is in scope, central FD elsewhere (ODE / LTBS / expression scaling /
/// TV-covariate / SDE models, or `gradient = fd`).
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
    let (mut analytic, mut fd) = (0usize, 0usize);
    for subject in &population.subjects {
        match resolve_gradient_method(model, subject) {
            InnerGradientMethod::Analytic => analytic += 1,
            InnerGradientMethod::Fd => fd += 1,
        }
    }
    // Show per-route counts only when the population splits across routes;
    // a single uniform route reads cleanly as just its label.
    let mixed = [analytic, fd].iter().filter(|&&c| c > 0).count() > 1;
    let mut parts: Vec<String> = Vec::new();
    for (count, label) in [(analytic, "analytic (Dual2)"), (fd, "FD")] {
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
        GradientMethod::Ad => "AD (retired → analytic)",
        GradientMethod::Fd => "FD",
    };

    format!("{resolved}  [requested: {requested_label}]")
}

/// Warning when *some but not all* subjects fall back to the FD inner gradient
/// (outside the analytic provider's scope — SS+reset, time-varying covariates,
/// oral infusion, modeled-duration doses, …). Returns `None` for a uniform
/// population: all-analytic needs no warning, and all-FD is a model-level property
/// already obvious from the banner and the model itself. Surfaced into
/// `FitResult.warnings` per the CLAUDE.md convention that non-fatal issues go
/// through `warnings`, not the startup banner alone.
///
/// Uses the actual light provider at the prior mode (`η = 0`) so it catches the
/// *per-point* fallbacks (modeled-duration, SS+reset, oral infusion) that the
/// coarse model-level [`resolve_gradient_method`] does not.
pub(crate) fn fd_fallback_warning(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
) -> Option<String> {
    let zeros = vec![0.0; model.n_eta];
    let n_total = population.subjects.len();
    let n_fd = population
        .subjects
        .iter()
        .filter(|s| crate::sens::provider::subject_eta_grad(model, s, theta, &zeros).is_none())
        .count();
    if n_fd > 0 && n_fd < n_total {
        Some(format!(
            "{n_fd} of {n_total} subjects use finite-difference inner gradients \
             (outside the analytic provider's scope, e.g. steady-state + reset, \
             time-varying covariates, or modeled-duration doses); their results \
             are correct but slower."
        ))
    } else {
        None
    }
}

/// Global per-fit timing counters for gradient/Jacobian calls. Printed by
/// [`fit_inner`] when `FERX_TIME_GRADIENTS=1` in the environment. Atomics so
/// multiple rayon workers can update concurrently without locking.
pub(crate) struct GradientTimings {
    pub analytic_calls: AtomicU64,
    pub analytic_nanos: AtomicU64,
    pub fd_calls: AtomicU64,
    pub fd_nanos: AtomicU64,
    pub jac_analytic_calls: AtomicU64,
    pub jac_analytic_nanos: AtomicU64,
    pub jac_fd_calls: AtomicU64,
    pub jac_fd_nanos: AtomicU64,
}

impl GradientTimings {
    const fn new() -> Self {
        Self {
            analytic_calls: AtomicU64::new(0),
            analytic_nanos: AtomicU64::new(0),
            fd_calls: AtomicU64::new(0),
            fd_nanos: AtomicU64::new(0),
            jac_analytic_calls: AtomicU64::new(0),
            jac_analytic_nanos: AtomicU64::new(0),
            jac_fd_calls: AtomicU64::new(0),
            jac_fd_nanos: AtomicU64::new(0),
        }
    }
    #[inline]
    fn record_analytic(&self, ns: u64) {
        self.analytic_calls.fetch_add(1, Ordering::Relaxed);
        self.analytic_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_fd(&self, ns: u64) {
        self.fd_calls.fetch_add(1, Ordering::Relaxed);
        self.fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_analytic(&self, ns: u64) {
        self.jac_analytic_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_analytic_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_fd(&self, ns: u64) {
        self.jac_fd_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    pub(crate) fn reset(&self) {
        self.analytic_calls.store(0, Ordering::Relaxed);
        self.analytic_nanos.store(0, Ordering::Relaxed);
        self.fd_calls.store(0, Ordering::Relaxed);
        self.fd_nanos.store(0, Ordering::Relaxed);
        self.jac_analytic_calls.store(0, Ordering::Relaxed);
        self.jac_analytic_nanos.store(0, Ordering::Relaxed);
        self.jac_fd_calls.store(0, Ordering::Relaxed);
        self.jac_fd_nanos.store(0, Ordering::Relaxed);
    }
    pub(crate) fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            self.analytic_calls.load(Ordering::Relaxed),
            self.analytic_nanos.load(Ordering::Relaxed),
            self.fd_calls.load(Ordering::Relaxed),
            self.fd_nanos.load(Ordering::Relaxed),
            self.jac_analytic_calls.load(Ordering::Relaxed),
            self.jac_analytic_nanos.load(Ordering::Relaxed),
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

    if inner_profile_enabled() {
        PROFILE_INNER_SOLVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // ── IOV branch ─────────────────────────────────────────────────────────
    // When the model has kappa declarations AND this subject has occasion labels,
    // optimize over the flat vector [bsv_eta (n_eta), kappa_1 (n_kappa), ..., kappa_K (n_kappa)].
    if model.n_kappa > 0 && !subject.occasions.is_empty() {
        return find_ebe_iov(model, subject, params, max_iter, tol, eta_init, mu_k);
    }

    // mu: shift vector (zeros when no mu-referencing)
    // The inner EBE is optimised directly in eta_true space. Mu-referencing is a
    // pure reparametrisation of the search frame (`psi = eta_true + mu`) — it does
    // NOT change the EBE, since the minimum of `individual_nll(eta) + eta'Ω⁻¹eta`
    // is invariant to the constant shift. Searching the offset psi-space mis-scaled
    // the FD gradient step (`~|psi|`) for **additive** mu-refs, where `mu = TVx` is
    // large (e.g. 8) while the curvature lives at `eta ~ O(1)`; the biased gradient
    // drove the inner loop to a wrong eta and a degenerate marginal (issue #302).
    // mu-referencing's real benefit (the H-column gradient reuse) lives in the
    // OUTER loop, so dropping the shift here is correct and leaves the AD inner
    // path bit-identical (an exact gradient is shift-invariant).
    let _ = mu_k;
    let mut eta: Vec<f64> = match eta_init {
        Some(warm) => warm.to_vec(),
        None => vec![0.0; n_eta],
    };

    // FREM-aware cold-start: initialise each covariate pseudo-obs eta at its
    // data-implied mode `cov_obs − TV`. These etas are pinned by their
    // pseudo-observations (precision ≫ prior), so this is essentially their
    // exact posterior mode. Starting them at 0 instead leaves them ~±40 off,
    // and the block-Ω⁻¹ PK↔covariate coupling turns that error into a large
    // spurious force on the PK etas — which is what sent a handful of subjects'
    // PK etas running away (V≈e⁻⁹, MAT≈e¹¹) and produced modes with obs-NLL
    // ~1e7–1e8 that wrecked the IMP proposal (issue #406). Only on a cold start;
    // a warm start already carries good covariate etas.
    if eta_init.is_none() {
        if let Some(fc) = model.frem_config.as_ref() {
            for (j, &ft) in subject.fremtype.iter().enumerate() {
                if ft == 0 {
                    continue;
                }
                if let Some(&(theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&ft) {
                    if eta_idx < n_eta
                        && theta_idx < params.theta.len()
                        && j < subject.observations.len()
                    {
                        eta[eta_idx] = subject.observations[j] - params.theta[theta_idx];
                    }
                }
            }
        }
    }

    // Diagonal preconditioner for the inner BFGS. FREM posteriors are extremely
    // multi-scale: PK etas have curvature ~1e2 and scale ~0.1, covariate
    // pseudo-obs etas have curvature ~1e6 (EPSCOV variance) and scale ~±40, and
    // near-fixed etas reach ~1e10. With the default H0 = I the search direction
    // is mis-scaled by up to ~1e8 per dimension and BFGS never reaches the true
    // joint mode — the returned η̂ then has an absurd obs-NLL, which makes the
    // IMP/IMPMAP importance proposal (centred on that mode) collapse to ~0 ESS
    // and diverge (issue #406). The preconditioner sets H0 = diag(precondᵢ) with
    // precondᵢ ≈ posterior variance of etaᵢ = 1/(Ω⁻¹ᵢᵢ + dataᵢ), where dataᵢ is
    // the analytic FREM pseudo-obs precision (J=1, R=EPSCOV²); covariate dims
    // get a near-Newton step in one iteration, PK dims fall back to the prior
    // conditional scale. `None` for non-FREM models → identity H0 (unchanged).
    let precond: Option<Vec<f64>> = build_inner_preconditioner(model, subject, params, n_eta);
    // The preconditioner accelerates the inner search (it is the BFGS H0), but it
    // drives the convergence *test* only for FREM, where the raw L2 gradient norm
    // is dominated by the sharp covariate pseudo-obs dims and never reaches `tol`
    // (issue #406). For general FOCE/FOCEI fits the stop test stays raw L2, so the
    // converged EBE — and the estimates — are independent of H0: preconditioning
    // changes only the path to the mode, not the mode itself.
    let stop_precond: Option<&[f64]> = if model.frem_config.is_some() {
        precond.as_deref()
    } else {
        None
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
    // Also skip the cache when bioavailability `F` could reshape a rate-defined
    // infusion window: `F` scales such an infusion's *duration* (#419), which
    // moves the baked-in break times as the inner BFGS varies eta (the same
    // staleness reason as `has_lagtime`). The non-cached path rebuilds per call
    // with the current `F`. Duration-defined infusions keep the cache (`F` scales
    // their rate, not the window).
    let schedule = if (subject.has_tv_covariates() || subject.has_resets())
        && model.ode_spec.is_none()
        && pk::event_driven::supports_event_driven(model.pk_model)
        && !model.has_lagtime()
        && !(model.has_bioavailability() && subject.has_rate_defined_infusion())
    {
        Some(pk::event_driven::EventSchedule::for_subject(
            subject,
            model.pk_model,
            &subject.doses,
            &[],
        ))
    } else {
        None
    };

    // Objective evaluated directly at eta_true (the optimiser variable).
    let obj = |e: &[f64]| -> f64 {
        let mut scratch = pk_scratch_cell.borrow_mut();
        individual_nll_into_with_schedule(
            model,
            subject,
            &params.theta,
            e,
            &params.omega,
            &params.sigma.values,
            &mut scratch,
            schedule.as_ref(),
        )
    };

    // BFGS with the exact analytic η-gradient from the sensitivity provider when
    // in scope (Almquist et al. 2015): one provider evaluation per inner step
    // instead of the FD gradient's ~2·n_eta+1 predictions, and exact → fewer
    // steps. Per-point FD fallback if the provider can't serve a given (θ, η).
    let result = {
        if analytic_inner_grad_supported(model, subject) {
            let profile = inner_profile_enabled();
            let agrad = |e: &[f64]| -> Vec<f64> {
                let t0 = std::time::Instant::now();
                match analytic_eta_nll_gradient_with_schedule(
                    model,
                    subject,
                    &params.theta,
                    e,
                    &params.omega,
                    &params.sigma.values,
                    schedule.as_ref(),
                ) {
                    Some(g) => {
                        GRADIENT_TIMINGS.record_analytic(t0.elapsed().as_nanos() as u64);
                        if profile {
                            PROFILE_INNER_ANALYTIC_GRAD
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        g
                    }
                    None => {
                        let g = gradient_fd(&obj, e, n_eta);
                        GRADIENT_TIMINGS.record_fd(t0.elapsed().as_nanos() as u64);
                        if profile {
                            PROFILE_INNER_FD_FALLBACK
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        g
                    }
                }
            };
            inner_minimize_with_grad(
                &obj,
                &agrad,
                &mut eta,
                n_eta,
                max_iter,
                tol,
                precond.as_deref(),
                stop_precond,
            )
        } else {
            inner_minimize(
                &obj,
                &mut eta,
                n_eta,
                max_iter,
                tol,
                precond.as_deref(),
                stop_precond,
            )
        }
    };

    // If BFGS failed, fall back to Nelder-Mead. With `ebe_warm_start` (opt-in, off
    // by default) seed the simplex from the BFGS partial η̂ — a weakly-identified η that BFGS
    // ran far out sits on the steep prior slope, so NM slides to the mode in far
    // fewer iterations than refining from η=0 in the flat basin. A non-finite
    // partial is unusable, so cold-start from η=0 there. With the flag off, always
    // cold-start from η=0 (bit-identical to the historical behaviour).
    let bfgs_converged = result;
    let (nm_converged, used_fallback) = if !bfgs_converged {
        let warm = ebe_warm_start_enabled() && eta.iter().all(|v| v.is_finite());
        if !warm {
            eta = vec![0.0; n_eta];
        }
        let nm_ok = nelder_mead_minimize(&obj, &mut eta, n_eta, max_iter * 5, tol);
        (nm_ok, true)
    } else {
        (false, false)
    };

    let ebe_converged = bfgs_converged || nm_converged;
    let nll = obj(&eta);

    // The optimiser variable already is eta_true (mean-zero, NONMEM-compatible).
    let eta_true: Vec<f64> = eta;

    // Compute Jacobian at eta_true — match the gradient path so the H matrix
    // is consistent with the gradient that drove convergence. Reuses the
    // same per-subject helpers built once at the top of find_ebe; previously
    // these were rebuilt here, doubling the per-subject helper cost.
    // Inner half of the gradient-path policy ("gradient-based optimizers use
    // sensitivities, FD fallback"): an exact analytic ∂f/∂η Jacobian when the
    // model is in the supported analytical PK scope (1-/2-/3-cpt), else `None`
    // and we keep the AD/FD Jacobian below. Perf follow-up: skip building the FD Jacobian
    // when the analytic one is available — for this first landing it is computed
    // and then overridden, which keeps the diff minimal and trivially
    // revertible while the values come from the exact sensitivities.
    let t_jac = std::time::Instant::now();
    let analytic_jac: Option<DMatrix<f64>> =
        crate::sens::provider::subject_eta_jacobian(model, subject, &params.theta, &eta_true)
            .map(|j| DMatrix::from_row_slice(subject.obs_times.len(), n_eta, &j));
    if analytic_jac.is_some() {
        GRADIENT_TIMINGS.record_jac_analytic(t_jac.elapsed().as_nanos() as u64);
    }

    // When the exact analytic Jacobian is available, skip the FD fallback
    // entirely — previously it was always computed and then discarded by an
    // `unwrap_or`, a full O(n_eta) sweep per subject per outer iteration that
    // directly undercut the speed premise (PR #381 review finding #10).
    let h_matrix = match analytic_jac {
        Some(j) => j,
        None => {
            // FD Jacobian fallback for models the analytic provider doesn't cover.
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

    // IOV models are not FREM, so no inner preconditioner (identity H0).
    let bfgs_converged = inner_minimize(&obj, &mut x, n_flat, max_iter, tol, None, None);
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

    // Overwrite FREM pseudo-observation rows with exact analytical Jacobian.
    if let Some(ref fc) = model.frem_config {
        if !subject.fremtype.is_empty() {
            for (i, &ft) in subject.fremtype.iter().enumerate() {
                if ft > 0 {
                    if let Some(&(_theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&ft) {
                        for j in 0..n_eta {
                            h[(i, j)] = if j == eta_idx { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
        }
    }

    h
}

/// BFGS minimization with backtracking line search.
/// Uses analytical-style gradient via forward FD with small step.
/// L-BFGS two-loop recursion: the search direction `d = −H·g` from the bounded
/// `(s, y, ρ)` history, with implicit initial Hessian `H₀ = γI`,
/// `γ = sᵀy / yᵀy` of the most recent pair (Nocedal & Wright, Alg. 7.4). With an
/// empty history this returns `−g` (steepest descent), so the first step matches
/// the old dense-BFGS start. A diagonal `precond` (FREM, issue #406) replaces the
/// scalar `γ` initial Hessian with `H₀ = diag(precond)`, so the central scaling
/// step is per-dimension instead of a single ill-scaled `γ`.
fn lbfgs_direction(
    g: &[f64],
    s_hist: &[Vec<f64>],
    y_hist: &[Vec<f64>],
    rho_hist: &[f64],
    n: usize,
    precond: Option<&[f64]>,
) -> Vec<f64> {
    let dotp = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let m = s_hist.len();
    let mut q = g.to_vec();
    let mut alpha = vec![0.0; m];
    for i in (0..m).rev() {
        let a = rho_hist[i] * dotp(&s_hist[i], &q);
        alpha[i] = a;
        for j in 0..n {
            q[j] -= a * y_hist[i][j];
        }
    }
    let gamma = if m > 0 {
        let sy = dotp(&s_hist[m - 1], &y_hist[m - 1]);
        let yy = dotp(&y_hist[m - 1], &y_hist[m - 1]);
        if yy > 1e-12 {
            sy / yy
        } else {
            1.0
        }
    } else {
        1.0
    };
    // Central H₀·q: a diagonal preconditioner (`H₀ = diag(precond)`) when supplied,
    // else the scalar `γI` of standard L-BFGS.
    let mut z: Vec<f64> = match precond {
        Some(p) => q.iter().zip(p).map(|(qi, pi)| pi * qi).collect(),
        None => q.iter().map(|qi| gamma * qi).collect(),
    };
    for i in 0..m {
        let b = rho_hist[i] * dotp(&y_hist[i], &z);
        for j in 0..n {
            z[j] += (alpha[i] - b) * s_hist[i][j];
        }
    }
    z.iter().map(|zi| -zi).collect()
}

/// Number of curvature pairs retained by the L-BFGS history.
const LBFGS_MEMORY: usize = 8;

/// Inner-problem dimension at/above which L-BFGS replaces dense BFGS. Below it,
/// the dense `n×n` inverse-Hessian Newton-converges in a few steps and is faster
/// (benchmarked: dense wins for `n ≲ 8`, L-BFGS wins 2× at n=64, 17× at n=256 —
/// see `inner_solver_scaling_bench`). The threshold sits well above the typical
/// PK `n_eta` (≤ ~8) and modest IOV, so only genuinely high-dimensional inner
/// problems (large IOV: `n_eta + K·n_kappa`) take the L-BFGS path. Only consulted
/// in [`InnerOptimizer::Auto`]; an explicit `inner_optimizer` pins the solver.
pub const INNER_LBFGS_MIN_DIM: usize = 32;

/// Fit-scoped inner-loop optimizer mode, set once per fit from
/// `FitOptions::inner_optimizer` via [`set_inner_optimizer`] and read by the inner
/// dispatch. Stored as the [`InnerOptimizer`] discriminant (`0 = Auto`, the
/// default), so a fit that never sets it behaves exactly as before. A plain
/// process-global (not threaded through every `find_ebe` caller) because the
/// inner loop fans out over subjects via rayon and they all read one fit setting.
static INNER_OPT_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Set the inner-loop optimizer for the current fit. Call once at fit start.
pub fn set_inner_optimizer(mode: crate::types::InnerOptimizer) {
    use crate::types::InnerOptimizer::*;
    let code = match mode {
        Auto => 0,
        Bfgs => 1,
        Lbfgs => 2,
        NelderMead => 3,
    };
    INNER_OPT_MODE.store(code, std::sync::atomic::Ordering::Relaxed);
}

fn inner_optimizer_mode() -> crate::types::InnerOptimizer {
    use crate::types::InnerOptimizer::*;
    match INNER_OPT_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Bfgs,
        2 => Lbfgs,
        3 => NelderMead,
        _ => Auto,
    }
}

/// Fit-scoped flag for [`FitOptions::ebe_warm_start`](crate::types::FitOptions),
/// set via [`set_ebe_warm_start`] and read in the EBE Nelder–Mead fallback. Defaults
/// to `false` to match `FitOptions::default()` (the historical cold-restart
/// behaviour); a plain process-global for the same reason as [`INNER_OPT_MODE`]
/// (the inner loop fans out over subjects via rayon).
static EBE_WARM_START: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set whether the inner NM fallback warm-starts from the BFGS partial. Call once
/// at fit start.
pub fn set_ebe_warm_start(on: bool) {
    EBE_WARM_START.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn ebe_warm_start_enabled() -> bool {
    EBE_WARM_START.load(std::sync::atomic::Ordering::Relaxed)
}

/// `FERX_PROFILE=1` attribution counters for the inner loop: how many EBE solves
/// run, and per inner gradient step whether the exact analytic gradient served it
/// or it fell back to the `~2·n_eta+1`-prediction FD gradient. A high fallback
/// rate is the prime suspect when inner value-eval (prediction) counts balloon.
pub static PROFILE_INNER_SOLVES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROFILE_INNER_ANALYTIC_GRAD: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static PROFILE_INNER_FD_FALLBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn inner_profile_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("FERX_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// `FERX_NO_ANALYTIC_INNER=1` forces the FD inner gradient everywhere (A-B toggle).
/// Cached in a `OnceLock`: the value cannot change mid-run, and this is queried per
/// subject on every inner-loop entry (issue #438 review).
fn no_analytic_inner_forced() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("FERX_NO_ANALYTIC_INNER")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Print the accumulated inner-loop attribution profile (no-op unless `FERX_PROFILE=1`).
pub fn profile_report() {
    if !inner_profile_enabled() {
        return;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let solves = PROFILE_INNER_SOLVES.load(Relaxed);
    let ana = PROFILE_INNER_ANALYTIC_GRAD.load(Relaxed);
    let fd = PROFILE_INNER_FD_FALLBACK.load(Relaxed);
    if solves > 0 {
        let tot = (ana + fd).max(1);
        eprintln!(
            "[profile] inner: {} EBE solves; {} analytic-grad steps, {} FD-fallback steps ({:.2}% fallback)",
            solves,
            ana,
            fd,
            100.0 * fd as f64 / tot as f64
        );
    }
}

/// Model-level half of [`analytic_inner_grad_supported`]: every gate that does
/// not depend on the subject. `build_info::gradient_method_inner` reports the
/// inner route off **this same** predicate, so the reported `gradient_method_inner`
/// cannot drift from what `find_ebe` actually runs (PR #381 review #9).
pub(crate) fn analytic_inner_grad_supported_model(model: &CompiledModel) -> bool {
    // Escape hatch / A-B toggle: force the FD inner gradient everywhere.
    if no_analytic_inner_forced() {
        return false;
    }
    // The user explicitly requested finite differences.
    if matches!(model.gradient_method, GradientMethod::Fd) {
        return false;
    }
    if model.is_sde() {
        return false;
    }
    // LTBS keeps the FD inner gradient. The provider's generic closed forms and
    // the objective's `compute_predictions` agree only to ~1e-9, and the LTBS
    // `g = ln(f)` wrap with a small additive-on-log σ amplifies that mismatch in
    // the covariance OFV second-difference Hessian (the analytic-EBE minimum sits
    // ~1e-9 off the objective's, enough to corrupt the curvature and inflate the
    // SEs ~5×). FD reconverges the *objective's* own EBE, so the Hessian stays
    // clean. The analytic *outer* gradient still serves LTBS (the fit matches
    // NONMEM); only the inner EBE finder reverts here.
    if model.log_transform {
        return false;
    }
    // `ExpressionScale` keeps the FD inner gradient (the light provider doesn't
    // carry the scale quotient-rule); the analytic *outer* gradient still serves
    // it. Mirrors the LTBS choice above.
    if matches!(
        model.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ) {
        return false;
    }
    // IIV on residual error (#409): the Dual2 kernels build the residual variance
    // from σ alone and carry no `exp(2·η_ruv)` scaling rule, so the EBE search
    // must reconverge against the scaled `individual_nll` under FD (same reason as
    // `analytical_ad_unsupported`).
    if model.residual_error_eta.is_some() {
        return false;
    }
    crate::sens::provider::analytical_supported(model)
}

/// Whether the exact analytic η-gradient of the individual NLL
/// ([`analytic_eta_nll_gradient`]) applies to this model/subject: the model must
/// be in scope ([`analytic_inner_grad_supported_model`]) and the *subject* must
/// not carry features the light inner provider can't serve. Survival obs records
/// decline; time-varying covariates / oral infusion are served by the event-driven
/// inner walk (`subject_eta_grad_tvcov`, #447) when `tvcov_analytical_supported`.
fn analytic_inner_grad_supported(model: &CompiledModel, subject: &Subject) -> bool {
    // Survival/TTE observation records carry a likelihood term that neither inner
    // provider models — the analytical path declines below, and the light ODE walk
    // (`run_subject_eta`) iterates only `subject.obs_times`, so it would silently
    // omit the survival term. Guard both routes up front.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        return false;
    }
    // ODE models use the light `Dual1` inner provider (#410) with their own
    // per-subject scope ([`ode_inner_grad_supported`]). The global escape hatches
    // plus the model-level exclusions the analytical path applies in
    // `analytic_inner_grad_supported_model` still hold here:
    //   - IIV on residual error (#409): the dual kernels build the residual
    //     variance from σ alone, with no `exp(2·η_ruv)` scaling, so the EBE must
    //     reconverge against the scaled `individual_nll` under FD.
    //   - LTBS: the analytic-EBE minimum can sit off the objective's own EBE and
    //     corrupt the covariance Hessian / SEs (~5× on the analytical path). The
    //     ODE `Dual1` walk shares `solve_ode_g` with the objective so the mismatch
    //     is likely benign, but that is unmeasured — decline until a cov-SE test on
    //     an LTBS ODE model validates it (#438 review).
    if model.ode_spec.is_some() {
        if no_analytic_inner_forced()
            || matches!(model.gradient_method, GradientMethod::Fd)
            || model.is_sde()
            || model.residual_error_eta.is_some()
            || model.log_transform
        {
            return false;
        }
        return crate::sens::provider::ode_inner_grad_supported(model, subject);
    }
    if !analytic_inner_grad_supported_model(model) {
        return false;
    }
    // TV-cov / oral-infusion subjects now get the light event-driven inner gradient
    // (`subject_eta_grad_tvcov`, #447); trust the provider's `None` for the residual
    // out-of-scope cases (it matches the outer TV-cov scope). Other subjects keep the
    // static superposition inner. (The survival guard is hoisted to the top of this
    // function, so it covers this path too.)
    if subject.has_tv_covariates()
        || crate::sens::provider::subject_has_oral_infusion(model, subject)
    {
        return crate::sens::provider::tvcov_analytical_supported(model);
    }
    true
}

/// Exact η-gradient of the individual NLL `½(η'Ω⁻¹η + ln|Ω| + Σ_j[ε_j²/v_j + ln v_j])`
/// from the analytic sensitivity provider — the closed-form analog of the
/// sensitivity-equation gradient (Almquist, Leander & Jirstrand 2015). Replaces
/// the FD gradient's `~2·n_eta+1` predictions per inner step with one provider
/// evaluation. `None` when the provider can't serve this `(θ, η)` (degenerate
/// params / out of scope), so the caller falls back to FD for that point.
///
/// Per observation `j`, with `f = f_j(η)`, `ε = y_j − f`, `v = R(f)` the residual
/// variance and `R'(f)` its `f`-derivative:
/// ```text
///   ∂nll/∂η_k = Σ_j ∂f_j/∂η_k · ( −ε/v + ½·R'(f)·(1/v − ε²/v²) ) + (Ω⁻¹η)_k
/// ```
/// On an M3-censored row (`CENS=1`, with `y` carrying the LLOQ) the data term is
/// `−logΦ(z)`, `z = (y−f)/√v`, so its per-row coefficient becomes
/// `h·( 1/√v + (y−f)·R'(f)/(2·v^{3/2}) )` with `h = φ(z)/Φ(z)` the inverse Mills
/// ratio — matching the censored branch of [`individual_nll`].
/// `∂/∂f` of the M3 censored per-observation data term `−logΦ(z)`,
/// `z = (y−f)/√v`, where `y` carries the LLOQ, `v = R(f)` is the residual
/// variance and `dv_df = R'(f)`. Multiplying by `∂f/∂η_k` yields the censored
/// row's contribution to `∂nll/∂η_k`. `h = φ(z)/Φ(z)` is the inverse Mills ratio,
/// evaluated through logs so it stays finite in the far tail (`Φ(z)→0` when the
/// prediction sits well above the LLOQ).
#[inline]
fn m3_censored_dterm_df(y: f64, f: f64, v: f64, dv_df: f64) -> f64 {
    let sqrt_v = v.sqrt();
    let z = (y - f) / sqrt_v;
    let ln_phi = -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln();
    let h = (ln_phi - crate::stats::special::log_normal_cdf(z)).exp();
    h * (1.0 / sqrt_v + (y - f) * dv_df / (2.0 * v * sqrt_v))
}

/// Exact analytic `∂NLL_i/∂η` from the light first-order sensitivity provider:
/// `Σ_j (∂nll/∂f_j)·(∂f_j/∂η) + Ω⁻¹η`. `Some` only when the model is in the
/// provider's scope (returns `None` for ODE / TV-cov / oral-infusion / SS+reset /
/// expression-scale subjects). Shared by the inner EBE loop and the HMC sampler so
/// both estimators use the same Dual2 gradient (replacing the retired Enzyme path).
pub(crate) fn analytic_eta_nll_gradient(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma: &[f64],
) -> Option<Vec<f64>> {
    analytic_eta_nll_gradient_with_schedule(model, subject, theta, eta, omega, sigma, None)
}

/// As [`analytic_eta_nll_gradient`], but reusing the per-subject `EventSchedule` the
/// inner optimizer cached once, so the TV-cov provider doesn't rebuild it every inner
/// BFGS step (#449 re-review #6). `None` rebuilds locally.
#[allow(clippy::too_many_arguments)]
pub(crate) fn analytic_eta_nll_gradient_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> Option<Vec<f64>> {
    // Light first-order provider (value + ∂f/∂η only); the inner gradient never
    // needs the second-order / θ blocks the full `subject_sensitivities` carries.
    let sens = crate::sens::provider::subject_eta_grad_with_schedule(
        model,
        subject,
        theta,
        eta,
        cached_schedule,
    )?;
    let n_eta = model.n_eta;
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    let mut grad = vec![0.0_f64; n_eta];
    for (j, obs) in sens.iter().enumerate() {
        let y = subject.observations[j];
        let cmt = subject.obs_cmts[j];
        let f = obs.f;
        let v = model.residual_variance_at(cmt, f, sigma);
        if !(v > 0.0) {
            return None;
        }
        let dv_df = model.error_spec.dvar_df(cmt, f, sigma);
        let coef = if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
            m3_censored_dterm_df(y, f, v, dv_df)
        } else {
            let eps = y - f;
            -eps / v + 0.5 * dv_df * (1.0 / v - eps * eps / (v * v))
        };
        for k in 0..n_eta {
            grad[k] += coef * obs.df_deta[k];
        }
    }
    // Prior: ∂/∂η ½ η'Ω⁻¹η = Ω⁻¹η.
    let eta_v = nalgebra::DVector::from_column_slice(eta);
    let prior = &omega.inv * &eta_v;
    for (k, g) in grad.iter_mut().enumerate() {
        *g += prior[k];
    }
    Some(grad)
}

/// Build the diagonal inner-BFGS preconditioner (the search `H0`) for a subject.
///
/// FREM models (issue #406): `Some(diag)` with `diag[i]` ≈ the posterior variance
/// of `etaᵢ`, `1 / (Ω⁻¹ᵢᵢ + dataᵢ)`. `dataᵢ` accumulates the analytic precision of
/// each FREM covariate pseudo-observation that maps to `etaᵢ` (prediction = TV+eta,
/// so the Jacobian is 1 and the row contributes `1/R` with `R = EPSCOV²`); PK /
/// non-covariate dims have `dataᵢ = 0` and fall back to `1/Ω⁻¹ᵢᵢ`.
///
/// General FOCE/FOCEI models: `Some(1/Ω⁻¹ᵢᵢ)` — the prior conditional scale per η,
/// so a correlated or multi-scale Ω does not mis-scale the search. `None` only when
/// Ω⁻¹ has no usable diagonal (→ identity `H0`).
///
/// This preconditioner is the BFGS `H0` only. Whether it also drives the
/// convergence *test* is decided by the caller (`find_ebe`): FREM uses it for both
/// (raw L2 never reaches `tol` there); general fits stop on raw L2, so `H0` changes
/// only the path to the mode, not the converged EBE.
fn build_inner_preconditioner(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    n_eta: usize,
) -> Option<Vec<f64>> {
    if let Some(fc) = model.frem_config.as_ref() {
        return preconditioner_from_parts(
            fc,
            &subject.fremtype,
            &params.omega.inv,
            &params.sigma.values,
            n_eta,
        );
    }
    // General FOCE/FOCEI: scale each inner BFGS dimension by its prior conditional
    // variance `1/Ω⁻¹ᵢᵢ`, so a correlated or multi-scale Ω does not mis-scale the
    // identity-H0 search. UVM's block Ω, for example, gives η_V2 ≈ 8× the scale of
    // η_CL; with H0 = I that direction is mis-stepped and BFGS spends extra
    // iterations learning the curvature. Same diagonal mechanism the FREM path
    // uses, minus the covariate pseudo-obs precision (not cheaply available per-η
    // here). `find_ebe` keeps the raw-L2 stop test for this path, so the H0 only
    // changes the path to the mode — the converged EBE is unchanged.
    inner_preconditioner_from_omega(&params.omega.inv, n_eta)
}

/// Diagonal inner-BFGS preconditioner `precondᵢ = 1/Ω⁻¹ᵢᵢ` for general
/// (non-FREM) FOCE/FOCEI fits. Split out for unit testing.
fn inner_preconditioner_from_omega(omega_inv: &DMatrix<f64>, n_eta: usize) -> Option<Vec<f64>> {
    if n_eta == 0 {
        return None;
    }
    // Ω⁻¹ is the n_eta×n_eta BSV inverse; the loop indexes its diagonal to n_eta.
    debug_assert!(
        omega_inv.nrows() >= n_eta,
        "Ω⁻¹ ({}×{}) smaller than n_eta ({n_eta})",
        omega_inv.nrows(),
        omega_inv.ncols()
    );
    let mut precond = vec![1.0_f64; n_eta];
    let mut usable = false;
    for (i, p) in precond.iter_mut().enumerate() {
        let d = omega_inv[(i, i)];
        if d.is_finite() && d > 0.0 {
            *p = 1.0 / d;
            usable = true;
        }
    }
    usable.then_some(precond)
}

/// Pure core of [`build_inner_preconditioner`] (no `CompiledModel`/`Subject`
/// dependency, so it is unit-testable in isolation). See that function for the
/// rationale; `omega_inv` is Ω⁻¹ and `sigma` the σ values.
fn preconditioner_from_parts(
    fc: &FremConfig,
    fremtype: &[u16],
    omega_inv: &DMatrix<f64>,
    sigma: &[f64],
    n_eta: usize,
) -> Option<Vec<f64>> {
    if n_eta == 0 {
        return None;
    }
    let r_cov = {
        let s = sigma[fc.covariate_sigma_index];
        let v = s * s;
        if v > 1e-12 {
            v
        } else {
            1e-12
        }
    };
    let inv_r = 1.0 / r_cov;
    let mut data_prec = vec![0.0_f64; n_eta];
    for &ft in fremtype.iter() {
        if ft > 0 {
            if let Some(&(_theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&ft) {
                if eta_idx < n_eta {
                    data_prec[eta_idx] += inv_r;
                }
            }
        }
    }
    let mut precond = vec![1.0_f64; n_eta];
    for (i, p) in precond.iter_mut().enumerate() {
        let prec = omega_inv[(i, i)].max(0.0) + data_prec[i];
        if prec > 0.0 {
            *p = 1.0 / prec;
        }
    }
    Some(precond)
}

/// Initial inverse-Hessian for the inner BFGS: `diag(precond)` when a
/// preconditioner is supplied, else identity.
fn init_h_inv(n: usize, precond: Option<&[f64]>) -> DMatrix<f64> {
    match precond {
        Some(p) => DMatrix::from_diagonal(&DVector::from_column_slice(p)),
        None => DMatrix::identity(n, n),
    }
}

/// Convergence metric. With a preconditioner the natural stopping test is the
/// preconditioned (≈ Newton-decrement) norm `√(Σ gᵢ²·precondᵢ)`, which is
/// commensurate across the multi-scale dimensions; the raw L2 norm would be
/// dominated by the sharp covariate dims and never fall below `tol`.
fn grad_norm_metric(g: &[f64], precond: Option<&[f64]>) -> f64 {
    match precond {
        Some(p) => g
            .iter()
            .zip(p.iter())
            .map(|(&gi, &pi)| gi * gi * pi)
            .sum::<f64>()
            .sqrt(),
        None => g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt(),
    }
}

/// Whether to take the L-BFGS path for inner dimension `n` under the current
/// [`inner_optimizer_mode`]. `Auto` consults the [`INNER_LBFGS_MIN_DIM`] threshold;
/// an explicit `Bfgs`/`Lbfgs` pins it; `NelderMead` is handled by the callers
/// before this is reached (it ignores the gradient).
fn inner_use_lbfgs(n: usize) -> bool {
    use crate::types::InnerOptimizer::*;
    match inner_optimizer_mode() {
        Auto => n >= INNER_LBFGS_MIN_DIM,
        Lbfgs => true,
        // Bfgs and NelderMead never take the L-BFGS branch (NelderMead is dispatched
        // earlier); Bfgs forces dense.
        _ => false,
    }
}

/// Inner EBE minimization with a finite-difference gradient. Dispatches per the
/// fit-scoped [`inner_optimizer_mode`] (dense BFGS / L-BFGS / Nelder–Mead); in
/// `Auto` it falls back to the [`INNER_LBFGS_MIN_DIM`] size threshold. All
/// gradient-based variants converge to the same EBE (stationary point of
/// `individual_nll + ½η'Ω⁻¹η`).
fn inner_minimize(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
    precond: Option<&[f64]>,
    stop_precond: Option<&[f64]>,
) -> bool {
    if matches!(
        inner_optimizer_mode(),
        crate::types::InnerOptimizer::NelderMead
    ) {
        return nelder_mead_minimize(obj, x, n, max_iter, tol);
    }
    let grad = |p: &[f64]| gradient_fd(obj, p, n);
    if inner_use_lbfgs(n) {
        lbfgs_core(obj, &grad, x, n, max_iter, tol, precond, stop_precond)
    } else {
        dense_bfgs_core(obj, &grad, x, n, max_iter, tol, precond, stop_precond)
    }
}

/// Inner EBE minimization with an externally-provided gradient (analytic
/// sensitivities or AD). Same fit-scoped dispatch as [`inner_minimize`]; the
/// `NelderMead` mode ignores the supplied gradient.
#[allow(clippy::too_many_arguments)]
fn inner_minimize_with_grad(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
    precond: Option<&[f64]>,
    stop_precond: Option<&[f64]>,
) -> bool {
    if matches!(
        inner_optimizer_mode(),
        crate::types::InnerOptimizer::NelderMead
    ) {
        return nelder_mead_minimize(obj, x, n, max_iter, tol);
    }
    if inner_use_lbfgs(n) {
        lbfgs_core(obj, grad, x, n, max_iter, tol, precond, stop_precond)
    } else {
        dense_bfgs_core(obj, grad, x, n, max_iter, tol, precond, stop_precond)
    }
}

/// Shared L-BFGS driver: two-loop direction + backtracking line search, bounded
/// `(s, y, ρ)` history. `grad` supplies the gradient (FD or AD).
#[allow(clippy::too_many_arguments)]
fn lbfgs_core(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
    precond: Option<&[f64]>,
    stop_precond: Option<&[f64]>,
) -> bool {
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    let mut g = grad(x);
    let mut f_cur = obj(x);

    for _iter in 0..max_iter {
        // Stopping metric. `stop_precond` is `Some` only for FREM, where the raw
        // L2 norm would be dominated by the sharp covariate pseudo-obs dims and
        // never fall below `tol` (issue #406), so the preconditioned (≈ Newton-
        // decrement) norm is required. For general fits `stop_precond` is `None`
        // → raw L2, so the converged EBE is independent of the `precond` H0 used
        // to accelerate the search above.
        let gnorm = grad_norm_metric(&g, stop_precond);
        if gnorm < tol {
            return true;
        }

        let mut d = lbfgs_direction(&g, &s_hist, &y_hist, &rho_hist, n, precond);
        // Guard against a non-descent direction (e.g. after a bad curvature
        // pair) by falling back to (preconditioned) steepest descent.
        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 {
            d = match precond {
                Some(p) => g.iter().zip(p).map(|(gi, pi)| -gi * pi).collect(),
                None => g.iter().map(|gi| -gi).collect(),
            };
        }

        let (alpha, f_new) = backtracking_line_search(obj, x, &d, &g, n, f_cur);
        // No sufficient-decrease step found: report non-convergence so the caller
        // takes the fallback path rather than accepting a non-stationary η̂.
        if alpha == 0.0 {
            return false;
        }
        f_cur = f_new;

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }

        let g_new = grad(x);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
        if sy > 1e-12 {
            if s_hist.len() == LBFGS_MEMORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        g = g_new;
    }

    false
}

/// Dense (`n×n` inverse-Hessian) BFGS driver, retained for low-dimensional inner
/// problems where it beats L-BFGS (no two-loop bookkeeping) and for the
/// solver-scaling benchmark. `grad` supplies the gradient (FD or analytic).
#[allow(clippy::too_many_arguments)]
fn dense_bfgs_core(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
    precond: Option<&[f64]>,
    stop_precond: Option<&[f64]>,
) -> bool {
    let mut h_inv = init_h_inv(n, precond);
    let mut g = grad(x);
    // Track the objective at the current iterate so the line search never has to
    // recompute `obj(x)` (one prediction walk per inner step on the hot path).
    let mut f_cur = obj(x);
    let mut first_step = true;

    for _iter in 0..max_iter {
        // `stop_precond` is `Some` only for FREM (issue #406); general fits stop
        // on the raw L2 norm so the converged EBE is independent of the `precond`
        // H0 that accelerates the search.
        let gnorm = grad_norm_metric(&g, stop_precond);

        // Scale initial Hessian so first step is O(1) not O(gnorm). Only for the
        // identity-H0 path (`precond.is_none()`), where `stop_precond` is also
        // `None`, so `gnorm` here is the raw L2 norm; a diagonal preconditioner
        // already sets the per-dim scale.
        if precond.is_none() && first_step && gnorm > 1.0 {
            h_inv *= 1.0 / gnorm;
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
            // Reset to the (preconditioned) steepest-descent metric, not raw
            // identity — for FREM the preconditioner is what keeps the descent
            // direction commensurate across the multi-scale dimensions.
            h_inv = init_h_inv(n, precond);
            let d: Vec<f64> = (-&h_inv * &g_vec).iter().copied().collect();
            let (alpha, f_new) = backtracking_line_search(obj, x, &d, &g, n, f_cur);
            // Even steepest descent found no sufficient-decrease step: report
            // non-convergence so the caller takes the Nelder–Mead fallback,
            // which re-solves from η=0 and reaches the true mode. Claiming
            // convergence here would hand the outer FOCE gradient a
            // non-stationary η̂ and stall the population optimiser.
            if alpha == 0.0 {
                return false;
            }
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            f_cur = f_new;
            g = grad(x);
            continue;
        }

        let (alpha, f_new) = backtracking_line_search(obj, x, &d, &g, n, f_cur);
        // No sufficient-decrease step found: not yet at the mode (the descent
        // direction exists but the line search exhausted its trials). Report
        // failure so Nelder–Mead re-solves and the outer gradient sees a true
        // stationary η̂.
        if alpha == 0.0 {
            return false;
        }

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        f_cur = f_new;

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

/// Maximum trial steps in the backtracking line search before it gives up.
/// With quadratic interpolation a sufficient-decrease step is normally found in
/// 1–3 trials; the cap only bites on directions with no representable decrease
/// (i.e. the iterate is already at the posterior mode to machine precision).
const MAX_LINE_SEARCH_TRIALS: usize = 30;

/// Backtracking line search with an Armijo sufficient-decrease test, choosing
/// each successive trial step by **safeguarded quadratic interpolation** rather
/// than fixed halving. Fitting a quadratic through the known `f0`, the slope
/// `dg = ∇f·d`, and the latest trial value lands on (or near) the Armijo step in
/// far fewer evaluations than repeated `α ← α/2`, which on this inner objective
/// routinely needed ~20 backtracks and frequently exhausted the cap.
///
/// `f0` is the objective at `x`, supplied by the caller (the inner BFGS already
/// tracks it), so the line search no longer recomputes `obj(x)` on every call.
///
/// Returns `(alpha, f_at_x_plus_alpha_d)`. `alpha == 0.0` signals that no
/// sufficient-decrease step exists along `d` (non-descent direction, or the
/// directional decrease is below numerical resolution); the caller treats that
/// as a stationary point.
fn backtracking_line_search(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    n: usize,
    f0: f64,
) -> (f64, f64) {
    let c1 = 1e-4;
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
    // Not a descent direction: nothing to do (caller falls back / stops).
    if !(dg < 0.0) {
        return (0.0, f0);
    }

    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = obj(&x_new);
        if f_new <= f0 + c1 * alpha * dg {
            return (alpha, f_new);
        }
        // Minimiser of the quadratic matching f0, dg (slope at 0) and f_new at
        // the current alpha. Safeguard into [0.1·α, 0.5·α] so a flat/non-convex
        // sample still makes definite progress (never larger than plain halving,
        // never a near-zero collapse).
        let denom = 2.0 * (f_new - f0 - dg * alpha);
        let alpha_quad = if denom > 0.0 {
            -dg * alpha * alpha / denom
        } else {
            0.5 * alpha
        };
        alpha = alpha_quad.clamp(0.1 * alpha, 0.5 * alpha);
        if alpha < 1e-16 {
            break;
        }
    }
    (0.0, f0)
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

    // Overwrite FREM pseudo-observation rows with exact analytical Jacobian.
    // For FREMTYPE > 0 observations, prediction = theta[k] + eta[m], so
    // ∂Y/∂η_j = 1 if j == m, 0 otherwise. The FD values for these rows
    // are noisy (esp. cross-terms that should be exactly 0) and corrupt
    // the posterior Hessian used by the IS proposal.
    if let Some(ref fc) = model.frem_config {
        if !subject.fremtype.is_empty() {
            for (i, &ft) in subject.fremtype.iter().enumerate() {
                if ft > 0 {
                    if let Some(&(_theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&ft) {
                        for j in 0..n_eta {
                            h[(i, j)] = if j == eta_idx { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
        }
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

    /// The M3 censored coefficient `∂/∂f[−logΦ((y−f)/√v)]` must equal a central
    /// finite difference of that data term — across additive (`dv_df = 0`) and
    /// f-dependent (`dv_df ≠ 0`, e.g. proportional/combined) variance, and across
    /// the regimes `f < LLOQ`, `f ≈ LLOQ`, and `f ≫ LLOQ` (deep tail, where the
    /// inverse Mills ratio's log-domain evaluation matters).
    #[test]
    fn m3_censored_dterm_df_matches_fd() {
        // Per-row censored data term −logΦ(z), z = (y−f)/√v(f), with v(f) a
        // generic affine-in-f² residual variance: v = sig_add² + (sig_prop·f)².
        let term = |y: f64, f: f64, sig_add: f64, sig_prop: f64| -> f64 {
            let v = sig_add * sig_add + sig_prop * sig_prop * f * f;
            let z = (y - f) / v.sqrt();
            -crate::stats::special::log_normal_cdf(z)
        };
        let lloq = 1.0_f64;
        let cases = [
            // (f, sig_add, sig_prop)
            (0.6, 0.2, 0.0),  // additive, f below LLOQ
            (1.0, 0.2, 0.0),  // additive, f at LLOQ
            (0.8, 0.0, 0.25), // proportional, dv_df ≠ 0
            (0.7, 0.15, 0.2), // combined, dv_df ≠ 0
            (3.0, 0.2, 0.0),  // f ≫ LLOQ: deep tail (Φ(z)→0)
        ];
        for (f, sig_add, sig_prop) in cases {
            let v = sig_add * sig_add + sig_prop * sig_prop * f * f;
            let dv_df = 2.0 * sig_prop * sig_prop * f; // ∂v/∂f
            let analytic = m3_censored_dterm_df(lloq, f, v, dv_df);
            // `normal_cdf` is a rational approximation (~1.5e-7 abs error); a tiny
            // FD step amplifies that noise (noise/h), so use a moderate step where
            // truncation and approximation error both sit well under the band.
            let h = 1e-3;
            let fd = (term(lloq, f + h, sig_add, sig_prop) - term(lloq, f - h, sig_add, sig_prop))
                / (2.0 * h);
            assert!(
                (analytic - fd).abs() < 1e-3 * (1.0 + fd.abs()),
                "f={f}, sig_add={sig_add}, sig_prop={sig_prop}: analytic {analytic} vs FD {fd}"
            );
        }
    }

    /// End-to-end: the analytic M3 inner η-gradient must match a central finite
    /// difference of the inner objective (`individual_nll_into_with_schedule`,
    /// which carries the `−2·logΦ(z)` censored term) on the real warfarin BLOQ
    /// model + data — exercising the full wiring (provider, cens lookup, coef
    /// dispatch), not just the isolated coefficient.
    #[test]
    fn analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq() {
        use std::cell::RefCell;
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin_bloq.ferx"))
                .expect("warfarin BLOQ model parses");
        assert!(
            matches!(model.bloq_method, crate::types::BloqMethod::M3),
            "model must be M3"
        );
        let pop =
            crate::io::datareader::read_nonmem_csv(Path::new("data/warfarin_bloq.csv"), None, None)
                .expect("warfarin BLOQ data loads");
        let subject = pop
            .subjects
            .iter()
            .find(|s| s.cens.iter().any(|&c| c != 0))
            .expect("at least one subject with a censored row");

        let theta = &model.default_params.theta;
        let omega = &model.default_params.omega;
        let sigma = &model.default_params.sigma.values;
        let eta = vec![0.12, -0.05, 0.2];

        let analytic = analytic_eta_nll_gradient(&model, subject, theta, &eta, omega, sigma)
            .expect("analytic M3 inner gradient must be supported");

        let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(subject));
        let obj = |e: &[f64]| -> f64 {
            let mut s = scratch.borrow_mut();
            individual_nll_into_with_schedule(&model, subject, theta, e, omega, sigma, &mut s, None)
        };
        let fd = gradient_fd(&obj, &eta, model.n_eta);

        for k in 0..model.n_eta {
            assert!(
                (analytic[k] - fd[k]).abs() < 1e-4 * (1.0 + fd[k].abs()),
                "η[{k}]: analytic {} vs FD {}",
                analytic[k],
                fd[k]
            );
        }
    }

    /// Dense BFGS vs L-BFGS scaling with inner dimension `n`, on an
    /// ill-conditioned 1-D-Laplacian quadratic `½xᵀLx − 1ᵀx` (cond ≈ (n/π)², so
    /// the solve needs ~O(n) curvature updates — representative of a curved inner
    /// NLL). Both use the **analytic** gradient `Lx − 1` so the per-iteration cost
    /// is dominated by the solver's linear algebra, not the gradient: dense is
    /// `O(n²)`/step (matvec + rank-2 update), L-BFGS `O(m·n)`/step. Isolates the
    /// solver, unlike a real fit where the prediction/FD cost dominates.
    #[test]
    #[ignore = "bench: cargo test --release ... -- --ignored --nocapture inner_solver_scaling_bench"]
    fn inner_solver_scaling_bench() {
        use std::time::Instant;
        eprintln!("inner-solver scaling (analytic-gradient Laplacian quadratic):");
        for &n in &[4usize, 8, 16, 32, 64, 128, 256] {
            // f(x) = ½ Σ_i (x_i − x_{i-1})² + ½ x_0²  −  Σ_i x_i   (x_{-1}=0).
            let obj = move |x: &[f64]| -> f64 {
                let mut f = 0.5 * x[0] * x[0];
                for i in 1..n {
                    let d = x[i] - x[i - 1];
                    f += 0.5 * d * d;
                }
                f - x.iter().sum::<f64>()
            };
            // grad = L x − 1, L the Dirichlet 1-D Laplacian (tridiag 2,−1).
            let grad = move |x: &[f64]| -> Vec<f64> {
                let mut g = vec![0.0; n];
                for i in 0..n {
                    let mut v = 2.0 * x[i];
                    if i > 0 {
                        v -= x[i - 1];
                    }
                    if i + 1 < n {
                        v -= x[i + 1];
                    }
                    g[i] = v - 1.0;
                }
                g
            };
            let runs = 50;
            let time_it = |solver: &dyn Fn(&mut [f64]) -> bool| -> f64 {
                let t0 = Instant::now();
                for _ in 0..runs {
                    let mut x = vec![0.0; n];
                    std::hint::black_box(solver(&mut x));
                }
                t0.elapsed().as_secs_f64() * 1e3 / runs as f64
            };
            let t_dense = time_it(&|x| dense_bfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None));
            let t_lbfgs = time_it(&|x| lbfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None));
            eprintln!(
                "  n={n:4}  dense={t_dense:8.3} ms  lbfgs={t_lbfgs:8.3} ms  dense/lbfgs={:.2}x",
                t_dense / t_lbfgs
            );
        }
    }

    /// The interpolating backtracking line search returns a step that satisfies
    /// the Armijo sufficient-decrease test and strictly lowers the objective,
    /// using only a handful of trial evaluations (the property the FOCEI inner
    /// loop relies on — fixed halving used ~20 here and frequently hit the cap).
    #[test]
    fn line_search_finds_armijo_step_quickly() {
        // f(x) = (x − 3)²; at x = 0 the unit Newton-less step −g overshoots the
        // minimiser, so a fixed-halving search would backtrack repeatedly.
        let obj = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };
        let x = [0.0];
        let g = [2.0 * (x[0] - 3.0)]; // = −6
        let d = [-g[0]]; // steepest descent, dg = −36 < 0
        let f0 = obj(&x);
        let evals = std::cell::Cell::new(0usize);
        let counting = |xx: &[f64]| {
            evals.set(evals.get() + 1);
            obj(xx)
        };
        let (alpha, f_new) = backtracking_line_search(&counting, &x, &d, &g, 1, f0);
        let evals = evals.get();
        assert!(alpha > 0.0, "a descent step must be found");
        let c1 = 1e-4;
        let dg: f64 = d.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
        assert!(
            f_new <= f0 + c1 * alpha * dg,
            "returned step must satisfy Armijo"
        );
        assert!(f_new < f0, "objective must strictly decrease");
        assert!(
            evals <= 5,
            "interpolation should converge in a few evals, got {evals}"
        );
    }

    /// A non-descent direction (dg ≥ 0) yields `alpha == 0` and leaves the
    /// objective baseline untouched — the signal the inner BFGS uses to stop /
    /// fall back rather than step uphill.
    #[test]
    fn line_search_rejects_non_descent_direction() {
        let obj = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };
        let x = [0.0];
        let g = [2.0 * (x[0] - 3.0)]; // = −6
        let d = [g[0]]; // SAME sign as g → dg = +36 ≥ 0 (ascent)
        let f0 = obj(&x);
        let (alpha, f_new) = backtracking_line_search(&obj, &x, &d, &g, 1, f0);
        assert_eq!(alpha, 0.0);
        assert_eq!(f_new, f0);
    }

    /// The refactored dense BFGS (objective-tracked line search) still drives a
    /// well-conditioned quadratic to its analytic minimiser.
    #[test]
    fn dense_bfgs_converges_on_quadratic() {
        // f(x) = (x0−1)² + 4(x1+2)², minimiser (1, −2).
        let obj =
            |x: &[f64]| -> f64 { (x[0] - 1.0) * (x[0] - 1.0) + 4.0 * (x[1] + 2.0) * (x[1] + 2.0) };
        let grad = |x: &[f64]| -> Vec<f64> { vec![2.0 * (x[0] - 1.0), 8.0 * (x[1] + 2.0)] };
        let mut x = vec![0.0, 0.0];
        let ok = dense_bfgs_core(&obj, &grad, &mut x, 2, 200, 1e-10, None, None);
        assert!(ok, "BFGS should report convergence");
        assert!((x[0] - 1.0).abs() < 1e-6, "x0 = {}", x[0]);
        assert!((x[1] + 2.0).abs() < 1e-6, "x1 = {}", x[1]);
    }

    #[test]
    fn test_inner_loop_stats_default() {
        let s = InnerLoopStats::default();
        assert_eq!(s.n_unconverged, 0);
        assert_eq!(s.n_fallback, 0);
    }

    // ── FREM inner-loop preconditioner (issue #406) ──────────────────────────

    #[test]
    fn preconditioner_scales_each_dim_by_its_own_curvature() {
        // 4 etas: 2 PK (dims 0,1; no FREM pseudo-obs) and 2 covariate (dims 2,3;
        // FREMTYPE 100→eta2, 200→eta3). The covariate pseudo-obs precision is
        // 1/R = 1/(EPSCOV²) = 1e6; PK dims have no data term and fall back to the
        // prior conditional scale 1/Ω⁻¹ᵢᵢ.
        let mut fremtype_to_indices = std::collections::HashMap::new();
        fremtype_to_indices.insert(100u16, (5usize, 2usize));
        fremtype_to_indices.insert(200u16, (6usize, 3usize));
        let fc = FremConfig {
            fremtype_to_indices,
            covariate_sigma_index: 1,
        };
        // Ω⁻¹: PK precisions 10 and 4; covariate prior precisions tiny (0.01).
        let omega_inv =
            DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 4.0, 0.01, 0.01]));
        // sigma[1] = EPSCOV = 1e-3 (SD) → R = 1e-6 → data precision 1e6.
        let sigma = [0.3, 1e-3];
        // One PK obs row (ft=0) plus one pseudo-obs row per covariate.
        let fremtype = [0u16, 100, 200];

        let p = preconditioner_from_parts(&fc, &fremtype, &omega_inv, &sigma, 4)
            .expect("Some for n_eta > 0");

        // PK dims: 1/Ω⁻¹ᵢᵢ.
        assert!((p[0] - 0.1).abs() < 1e-9, "p0 = {}", p[0]);
        assert!((p[1] - 0.25).abs() < 1e-9, "p1 = {}", p[1]);
        // Covariate dims: 1/(0.01 + 1e6) ≈ 1e-6 — sharply smaller than PK.
        assert!(p[2] < 1.1e-6 && p[2] > 0.9e-6, "p2 = {}", p[2]);
        assert!(p[3] < 1.1e-6 && p[3] > 0.9e-6, "p3 = {}", p[3]);
        // The whole point: covariate dims get a step scale ~1e5× tighter than PK,
        // so a single preconditioned BFGS step is near-Newton for them.
        assert!(p[0] / p[2] > 1e4);
    }

    #[test]
    fn preconditioner_is_none_for_zero_eta() {
        let fc = FremConfig {
            fremtype_to_indices: std::collections::HashMap::new(),
            covariate_sigma_index: 0,
        };
        let omega_inv = DMatrix::<f64>::zeros(0, 0);
        assert!(preconditioner_from_parts(&fc, &[], &omega_inv, &[1e-3], 0).is_none());
    }

    /// The general (non-FREM) inner preconditioner inverts the Ω⁻¹ diagonal so
    /// each BFGS dimension is scaled by its prior conditional variance, giving a
    /// well-scaled H0 for multi-scale / correlated Ω.
    #[test]
    fn inner_precond_from_omega_inverts_diagonal() {
        // Diagonal Ω⁻¹ = diag(10, 2, 0.5) → precond = diag(0.1, 0.5, 2.0).
        let omega_inv = DMatrix::from_diagonal(&DVector::from_column_slice(&[10.0, 2.0, 0.5]));
        let p = inner_preconditioner_from_omega(&omega_inv, 3).expect("usable diagonal");
        assert!((p[0] - 0.1).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
        assert!((p[2] - 2.0).abs() < 1e-12);
        // n_eta == 0 → None (identity H0).
        assert!(inner_preconditioner_from_omega(&DMatrix::<f64>::zeros(0, 0), 0).is_none());
        // A non-positive diagonal entry is skipped but a usable one still yields Some.
        let mixed = DMatrix::from_diagonal(&DVector::from_column_slice(&[0.0, 4.0]));
        let pm = inner_preconditioner_from_omega(&mixed, 2).expect("one usable entry");
        assert_eq!(pm[0], 1.0); // untouched default for the zero diagonal
        assert!((pm[1] - 0.25).abs() < 1e-12);
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
    fn test_frem_jacobian_overrides_fd_with_exact_values() {
        use crate::types::{
            DoseEvent, ErrorModel, GradientMethod, OmegaMatrix, PkModel, PkParams, SigmaVector,
        };
        use std::collections::HashMap;

        // Build a minimal model with 3 etas: CL, V, COV_WT(FREM)
        let omega = OmegaMatrix::from_diagonal(
            &[0.09, 0.09, 100.0],
            vec!["ETA_CL".into(), "ETA_V".into(), "ETA_WT_FREM".into()],
        );
        let default_params = crate::types::ModelParameters {
            theta: vec![10.0, 100.0, 90.0],
            theta_names: vec!["TVCL".into(), "TVV".into(), "TV_WT".into()],
            theta_lower: vec![0.01, 1.0, 0.0],
            theta_upper: vec![100.0, 500.0, 200.0],
            theta_fixed: vec![false, false, true],
            omega,
            omega_fixed: vec![false, false, false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["RUV".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        let model = CompiledModel {
            has_conditional_eta_params: false,
            name: "frem_jac_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Additive,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Additive),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL
                p.values[1] = theta[1] * eta[1].exp(); // V
                p
            }),
            n_theta: 3,
            n_eta: 3,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: vec![],
            theta_names: vec!["TVCL".into(), "TVV".into(), "TV_WT".into()],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_WT_FREM".into()],
            indiv_param_names: vec!["CL".into(), "V".into(), "COV_WT".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false; 3],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0, 1, 2],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: crate::types::BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::default(),
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: crate::types::ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            dose_attr_map: Default::default(),
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
            frem_config: Some(crate::types::FremConfig {
                fremtype_to_indices: {
                    let mut m = std::collections::HashMap::new();
                    m.insert(100u16, (2usize, 2usize)); // TV_WT / ETA_WT_FREM
                    m
                },
                covariate_sigma_index: 0,
            }),
            residual_error_eta: None,
        };

        // Subject: 2 PK obs + 1 FREM obs
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 0.0],
            obs_raw_times: Vec::new(),
            observations: vec![5.0, 3.0, 90.0],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: vec![0, 0, 100], // last obs is FREM
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };

        let theta = [10.0, 100.0, 90.0];
        let eta = [0.1, -0.05, 2.5];

        let mut scratch = pk::EventPkParams::default();
        let jac = compute_jacobian_fd(&model, &subject, &theta, &eta, &mut scratch, None);

        // Row 2 (FREM obs) must be exactly [0, 0, 1]
        assert_eq!(jac[(2, 0)], 0.0, "FREM row: ∂Y/∂η_CL must be exactly 0");
        assert_eq!(jac[(2, 1)], 0.0, "FREM row: ∂Y/∂η_V must be exactly 0");
        assert_eq!(jac[(2, 2)], 1.0, "FREM row: ∂Y/∂η_COV must be exactly 1");

        // PK rows should be non-zero for at least CL (row 0, col 0)
        assert!(
            jac[(0, 0)].abs() > 1e-10,
            "PK row: ∂Y/∂η_CL should be nonzero"
        );
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
            exclusions: None,
            warnings: vec![],
        };
        // `requested` is the user's FitOptions value, passed independently of
        // model.gradient_method (which compatibility rules may have mutated).
        let summary = gradient_route_summary(&model, &population, GradientMethod::Auto);
        assert!(
            summary.starts_with("FD"),
            "tv_fn=None must resolve to FD, got: {summary}"
        );
        // The bracket echoes the requested method, e.g. "[requested: auto]".
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

    /// Regression: `gradient = fd` must force the FD inner route on an
    /// analytic-supported model (previously the executor ignored
    /// `model.gradient_method`, so the option silently ran the Dual2 path while
    /// `build_info` reported FD). Uses the bundled warfarin model, which is in the
    /// analytic provider's scope (1-cpt oral, no LTBS / TV-cov / SDE).
    #[test]
    fn gradient_fd_forces_fd_inner_route() {
        use std::path::Path;
        let mut model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let subj = &pop.subjects[0];

        model.gradient_method = GradientMethod::Auto;
        assert_eq!(
            resolve_gradient_method(&model, subj),
            InnerGradientMethod::Analytic,
            "auto must resolve to the analytic route for the warfarin model"
        );
        model.gradient_method = GradientMethod::Fd;
        assert_eq!(
            resolve_gradient_method(&model, subj),
            InnerGradientMethod::Fd,
            "gradient = fd must force the FD inner route"
        );
    }

    /// `fd_fallback_warning` fires only for a *mixed* population — some subjects
    /// analytic, some on FD (here a modeled-duration `RATE=-2` subject, which the
    /// provider declines per-point). Uniform populations return `None`.
    #[test]
    fn fd_fallback_warning_fires_only_for_mixed_population() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let theta = &model.default_params.theta;
        let analytic = pop.subjects[0].clone();
        let mut fd_subj = pop.subjects[0].clone();
        let mut d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        d.rate_mode = crate::types::RateMode::ModeledDuration;
        fd_subj.doses.push(d);
        let mk_pop = |subjects| Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let mixed = mk_pop(vec![analytic.clone(), fd_subj]);
        let w = fd_fallback_warning(&model, &mixed, theta).expect("mixed population warns");
        assert!(w.contains("1 of 2"), "got: {w}");

        // Uniform analytic → no warning.
        assert!(fd_fallback_warning(&model, &mk_pop(vec![analytic]), theta).is_none());
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
        }
    }

    fn make_iov_subject() -> Subject {
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            obs_raw_times: Vec::new(),
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
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
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

    /// Pinning `inner_optimizer` to dense BFGS vs L-BFGS must reach the *same* EBE
    /// — both are gradient-based solvers of the same convex inner objective, so the
    /// explicit choice only changes the path, not the stationary point. Guards the
    /// `inner_optimizer` dispatch (and that pinning bypasses the size threshold).
    #[test]
    fn inner_optimizer_pin_reaches_same_ebe() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::InnerOptimizer;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
            obs_raw_times: Vec::new(),
            observations: vec![18.0, 16.0, 13.0, 9.0, 4.5, 2.2],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1; 6],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();

        set_inner_optimizer(InnerOptimizer::Bfgs);
        let bfgs = find_ebe(&model, &subject, &params, 200, 1e-8, None, None);
        set_inner_optimizer(InnerOptimizer::Lbfgs);
        let lbfgs = find_ebe(&model, &subject, &params, 200, 1e-8, None, None);
        set_inner_optimizer(InnerOptimizer::Auto);

        assert!(bfgs.converged && lbfgs.converged, "both must converge");
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                bfgs.eta[k],
                lbfgs.eta[k],
                max_relative = 1e-5,
                epsilon = 1e-7
            );
        }
    }

    /// `set_ebe_warm_start` round-trips through the fit-scoped global the EBE
    /// fallback reads, and defaults to `false` (matching `FitOptions::default`).
    #[test]
    fn ebe_warm_start_flag_round_trips() {
        assert!(!ebe_warm_start_enabled(), "default must be off");
        set_ebe_warm_start(true);
        assert!(ebe_warm_start_enabled());
        set_ebe_warm_start(false);
        assert!(!ebe_warm_start_enabled());
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
        };
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0],
            obs_raw_times: Vec::new(),
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
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None);
        assert!(result.kappas.is_empty());
    }

    /// Regression guard for #302: the non-IOV inner EBE must be invariant to the
    /// mu-reference shift — it is a pure reparametrization of the search frame.
    /// The bug was searching the offset psi-space (`psi = eta + mu`), which
    /// mis-scaled the FD gradient step (`~|psi|`) for a LARGE mu (additive
    /// mu-refs, `mu = TVx`), driving the EBE to a wrong point. A large `mu_k`
    /// must yield the same `eta_true` as `mu_k = None`.
    #[test]
    fn find_ebe_noniov_invariant_to_large_mu_shift() {
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
            frem_config: None,
            residual_error_eta: None,
            name: "noniov_mu".into(),
            has_conditional_eta_params: false,
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
            dose_attr_map: Default::default(),
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
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        };
        let subject = Subject {
            fremtype: Vec::new(),
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0],
            obs_raw_times: Vec::new(),
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
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let r_none = find_ebe(&model, &subject, &params, 200, 1e-6, None, None);
        // A large mu (e.g. an additive mu-ref's typical value) is the case that
        // mis-converged in psi-space; the EBE must be unchanged.
        let r_mu = find_ebe(&model, &subject, &params, 200, 1e-6, None, Some(&[8.0]));
        assert!(
            (r_none.eta[0] - r_mu.eta[0]).abs() < 1e-9,
            "non-IOV EBE must be mu-shift invariant: none={}, mu=8 -> {}",
            r_none.eta[0],
            r_mu.eta[0]
        );
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
        // This fixture's BSV mode sits far out (η̂ ≈ −7.6) where the FD-gradient
        // inner objective is very flat: the gnorm < 1e-5 stop is satisfied across
        // an η basin wider than 1e-4, so the exact landing point is line-search
        // path dependent. The invariant under test is that mu-referencing is
        // *honored* — if it were dropped the two runs would differ by ~mu (0.1).
        // The realised gap (~2e-4) is two orders of magnitude smaller, so a 1e-3
        // bound robustly distinguishes "applied" from "dropped".
        assert!(
            (r1.eta[0] - r2.eta[0]).abs() < 1e-3,
            "mu shift not applied: r1.eta={}, r2.eta={}",
            r1.eta[0],
            r2.eta[0],
        );
    }

    // The model classes below (non-log-normal ETA, LTBS, conditional params,
    // eta-dependent obs_scale, TTE) sit outside the closed-form inner-gradient
    // scope. This tests the named-reason predicate directly; the live routing
    // gate is `analytic_inner_grad_supported`. It is build-independent and runs
    // in the FD-only `ci` CI build, guarding the classification logic (#278).
    #[test]
    fn analytical_ad_unsupported_flags_each_class() {
        let mut model = make_iov_model();
        // Plain log-normal fixture -> supported.
        assert!(analytical_ad_unsupported(&model).is_none());

        // Non-log-normal ETA.
        model.eta_param_info = vec![crate::types::EtaParamInfo {
            eta_name: "ETA_CL".into(),
            param_type: crate::types::EtaParamType::Additive,
            linked_theta: None,
            individual_param_name: "CL".into(),
        }];
        assert!(analytical_ad_unsupported(&model).is_some());
        model.eta_param_info.clear();
        assert!(analytical_ad_unsupported(&model).is_none());

        // LTBS.
        model.log_transform = true;
        assert!(analytical_ad_unsupported(&model).is_some());
        model.log_transform = false;
        assert!(analytical_ad_unsupported(&model).is_none());

        // Conditional parameter (structured flag).
        model.has_conditional_eta_params = true;
        assert!(analytical_ad_unsupported(&model).is_some());
        model.has_conditional_eta_params = false;
        assert!(analytical_ad_unsupported(&model).is_none());

        // Expression-scale obs_scale (conservatively AD-unsafe; could read eta).
        model.scaling = crate::types::ScalingSpec::ExpressionScale {
            scale_fn: Box::new(|_, _, _, _| 1.0),
            deriv: None,
        };
        assert!(analytical_ad_unsupported(&model).is_some());
        model.scaling = crate::types::ScalingSpec::ScalarScale(1000.0);
        assert!(analytical_ad_unsupported(&model).is_none());
    }
}
