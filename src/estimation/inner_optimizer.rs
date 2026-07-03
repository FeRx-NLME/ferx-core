use crate::pk;
use crate::stats::likelihood::{
    individual_nll_into_with_schedule, individual_nll_iov, iov_occasion_groups,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::collections::BTreeMap;
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
    /// (ODE, LTBS, time-varying covariates, SDE) or the `FERX_NO_ANALYTIC_INNER`
    /// escape hatch is set. (η-dependent `ExpressionScale` is now analytic, #486.)
    Fd,
}

/// Model-level features that classify a model as outside the closed-form
/// inner-gradient scope, returning `Some(reason)` (else `None`). Historically
/// gated the retired Enzyme AD inner gradient; retained as a named-reason
/// classifier (the live scope check is [`analytic_inner_grad_supported`]).
#[allow(dead_code)]
pub(crate) fn analytical_ad_unsupported(model: &CompiledModel) -> Option<&'static str> {
    if !model.residual_correlations.is_empty() {
        return Some("correlated residual error");
    }
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
    // Eta-dependent `[scaling] obs_scale`. VESTIGIAL: the retired Enzyme-AD path froze the
    // scale subject-static and dropped `d obs_scale / d eta`, so it routed here to FD. The
    // LIVE inner path now serves a differentiable `ExpressionScale` analytically via the
    // η-quotient rule (#486, `provider::apply_expression_scale_inner`), and
    // `analytic_inner_grad_supported_model` does NOT bail on it. This branch is retained
    // only as the historical named reason (this whole fn is dead — see the header); it no
    // longer reflects routing. The divergence is pinned by `analytical_ad_unsupported_flags_each_class`.
    if model.scaling.breaks_ad_inner_gradient() {
        return Some(
            "eta-dependent obs_scale (ExpressionScale) [vestigial: served analytically since #486]",
        );
    }
    // Time-to-event (`[event_model]`) endpoint. The analytical single-snapshot
    // AD kernel computes the PK-observation NLL, not the hazard/survival
    // likelihood, so the eta-gradient through the hazard (especially shape
    // params) is wrong - `tte_weibull` / `tte_gompertz` diverged ~2-5 OFV from
    // FD under AD. Route TTE models to FD. (Always false without `survival`.)
    if model.has_tte() {
        return Some("time-to-event ([event_model]) hazard likelihood");
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
/// where it is in scope, central FD elsewhere (ODE / LTBS / TV-covariate / SDE
/// models, or `gradient = fd`; η-dependent `ExpressionScale` is analytic, #486).
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
        match resolve_gradient_method_for_reporting(model, subject, &model.default_params.theta) {
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

fn resolve_gradient_method_for_reporting(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
) -> InnerGradientMethod {
    if model.n_kappa == 0 {
        return resolve_gradient_method(model, subject);
    }
    if iov_inner_subject_route(model, subject, theta).is_some() {
        InnerGradientMethod::Analytic
    } else {
        InnerGradientMethod::Fd
    }
}

fn iov_inner_subject_route(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
) -> Option<Vec<crate::sens::provider::ObsGrad>> {
    if !crate::sens::provider::iov_sens_supported(model)
        || model.default_params.omega_iov.is_none()
        || analytic_inner_common_bail(model)
        || subject_has_survival_records(subject)
    {
        return None;
    }
    let k_occasions = iov_occasion_groups(subject).len();
    let n_flat = model.n_eta + k_occasions * model.n_kappa;
    let stacked = vec![0.0; n_flat];
    crate::sens::provider::subject_eta_grad_iov(model, subject, theta, &stacked)
}

fn iov_fd_reason(model: &CompiledModel, subject: &Subject) -> &'static str {
    if matches!(model.gradient_method, GradientMethod::Fd) {
        return "gradient = fd";
    }
    if analytic_inner_common_bail(model) {
        return "model-level analytic inner fallback";
    }
    if model.default_params.omega_iov.is_none() {
        return "missing omega_iov";
    }
    if subject_has_survival_records(subject) {
        return "survival/TTE observations";
    }
    if !crate::sens::provider::iov_sens_supported(model) {
        return "model outside IOV analytic scope";
    }
    if model.ode_spec.is_some() {
        // Single scan for the periodic steady-state predicate, mirroring
        // `ode_iov_subject_supported`'s hoisted `has_ss` so the attribution order and
        // the gate can't drift.
        let has_ss = subject.has_periodic_ss_dose();
        // Modeled-`RATE`/duration doses are analytic under IOV since #486 (the per-occasion
        // modeled-window jet rides the rate-off saltation) — including combined with
        // steady-state (#486: `equilibrate_ss_state_g` now threads the same jet into its
        // per-cycle active/quiet split), EXCEPT when a `D{cmt}`/`R{cmt}` slot is absent —
        // mirror `ode_iov_subject_supported`'s screen so this attribution can't drift.
        if !subject.all_doses_fixed() {
            let attr_map = model.active_dose_attr_map();
            let all_slots_present = subject.doses.iter().all(|d| {
                matches!(d.rate_mode, crate::types::RateMode::Fixed)
                    || crate::sens::ode_provider::modeled_slot_for(attr_map, d).is_some()
            });
            if !all_slots_present {
                return "modeled RATE/DURATION dose with missing D/R slot";
            }
        }
        // Mirror the SS gates of `ode_iov_subject_supported`, in the same order
        // (they are checked *before* the occasion/axis gates below, so omitting
        // them would misattribute an SS bail to a later reason). #590 review. A
        // steady-state rate-defined infusion under `F ≠ 1`, and steady-state combined
        // with an estimated lagtime, are both analytic now (#486).
        if has_ss
            && model
                .ode_spec
                .as_ref()
                .and_then(|o| o.rhs_program.as_ref())
                .is_some_and(|p| p.uses_time_vars())
        {
            return "steady-state dose + time-dependent ODE RHS";
        }
        // #486: a steady-state dose combined with a built-in absorption input-rate forcing
        // (`zero_order`/`first_order`) declines to FD — the dual SS equilibration does not
        // spread a periodic zero-order window / first-order tail over the cycle. Mirror
        // `ode_iov_subject_supported`'s bail, in the same order, so this attribution can't
        // drift.
        if has_ss
            && model
                .ode_spec
                .as_ref()
                .is_some_and(|o| !o.input_rate.is_empty())
        {
            return "steady-state dose + built-in absorption forcing";
        }
        let occ_groups = iov_occasion_groups(subject);
        if occ_groups.is_empty() {
            return "no observation occasions";
        }
        let n_stacked = model.n_eta + occ_groups.len() * model.n_kappa;
        let m_dim = model.n_theta + n_stacked;
        if m_dim > crate::sens::ode_provider::MAX_ODE_IOV_AXES {
            return "ODE IOV stacked axis cap";
        }
    }
    // Reached only for an FD subject (the caller invokes this after
    // `iov_inner_subject_route(..).is_none()`), so this is the catch-all for any
    // provider bail not enumerated above — never the analytic case.
    "subject outside IOV analytic scope"
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
    if model.n_kappa != 0 {
        return iov_fd_fallback_warning(model, population, theta);
    }
    let zeros = vec![0.0; model.n_eta];
    let n_total = population.subjects.len();
    let n_fd = population
        .subjects
        .iter()
        .filter(|s| crate::sens::provider::subject_eta_grad(model, s, theta, &zeros).is_none())
        .count();
    // Warn on a mixed population (some subjects fall back per-point), and also when
    // the *whole* population falls back while the model-level report claims analytic
    // — e.g. TV-cov + LTBS, where `analytic_inner_grad_supported_model` (and hence
    // `build_info::gradient_method_inner`) reports analytic but every subject's inner
    // EBE gradient actually runs FD. Without this, that mislabel would go uncorrected
    // (#381 review #9 / #665 review). A model that already reports FD and runs FD
    // everywhere needs no warning.
    let model_reports_analytic = analytic_inner_grad_supported_model(model);
    if n_fd > 0 && (n_fd < n_total || model_reports_analytic) {
        Some(format!(
            "{n_fd} of {n_total} subjects use finite-difference inner gradients \
             (outside the analytic provider's scope, e.g. steady-state + reset, \
             time-varying covariates + LTBS, or modeled-duration doses); their \
             results are correct but slower."
        ))
    } else {
        None
    }
}

fn iov_fd_fallback_warning(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
) -> Option<String> {
    let n_total = population.subjects.len();
    let mut n_fd = 0usize;
    let mut reasons: BTreeMap<&'static str, usize> = BTreeMap::new();
    for subject in &population.subjects {
        if iov_inner_subject_route(model, subject, theta).is_none() {
            n_fd += 1;
            *reasons.entry(iov_fd_reason(model, subject)).or_insert(0) += 1;
        }
    }
    // Mirror the non-IOV contract: only warn for a *mixed* population. An
    // all-analytic (`n_fd == 0`) population needs no warning, and an all-FD
    // (`n_fd == n_total`) one is already obvious from the `finite-difference`
    // banner and the model itself (e.g. `gradient = fd`, LTBS). #590 review.
    if n_fd == 0 || n_fd == n_total {
        return None;
    }
    let reason_text = reasons
        .into_iter()
        .map(|(reason, count)| format!("{reason}: {count}"))
        .collect::<Vec<_>>()
        .join("; ");
    Some(format!(
        "{n_fd} of {n_total} subjects use finite-difference inner gradients \
         in the IOV loop ({reason_text}); their results are correct but slower."
    ))
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
    /// `iov_occasion_groups`).
    pub kappas: Vec<DVector<f64>>,
    /// True when the subject was hard-rejected at its inner start (a pathological
    /// ODE+IOV warm-start NLL — see [`reject_ode_iov_inner_start`]). The returned
    /// `eta`/`h_matrix` are then a degenerate placeholder (off-mode η, zero H), so the
    /// outer loop must reject the whole trial rather than fold them into an accepted
    /// OFV. Unlike plain non-convergence this forces rejection regardless of
    /// `max_unconverged_frac` or the `min_obs` filter (#603 review #1/#2).
    pub hard_reject: bool,
}

/// Aggregate statistics from running the inner loop over all subjects.
#[derive(Debug, Default, Clone)]
pub struct InnerLoopStats {
    /// Subjects whose optimizer did not meet the convergence tolerance.
    pub n_unconverged: usize,
    /// Subjects for which the BFGS→Nelder-Mead fallback was triggered.
    pub n_fallback: usize,
    /// Subjects hard-rejected at their inner start (pathological ODE+IOV warm-start
    /// NLL). Any non-zero count forces the outer EBE guard to reject the trial,
    /// regardless of `max_unconverged_frac` or the `min_obs` filter (#603 review #1/#2).
    pub n_start_rejected: usize,
}

/// Inner-EBE fallback shared by [`find_ebe`] and [`find_ebe_iov`], invoked when the inner
/// BFGS reports non-convergence. Keeps the lower-objective of the BFGS partial and a single
/// Nelder–Mead restart, so a `false`-on-a-converged search (gradient-noise floor / line-
/// search exhaustion at the mode, #555) cannot have its correct η̂ discarded by an NM
/// restart that wanders into a worse basin. The same policy serves IOV and non-IOV so the
/// two paths cannot drift into contradictory convergence behaviour.
///
/// The restart seeds from the BFGS `partial` when `ebe_warm_start` is set (it sits on the
/// steep prior slope, so NM reaches the mode in fewer steps), else from `cold_seed` (η=0
/// for non-IOV, `[μ, 0…]` for IOV — the historical reset point). Exactly **one** NM solve
/// runs, so enabling `ebe_warm_start` is never slower than leaving it off.
///
/// Returns `(eta, nm_converged)`. The **value** is the lower-objective of the BFGS partial
/// and the NM restart — the substantive #555 fix: the previous code overwrote `eta` with the
/// NM restart unconditionally, discarding a correct partial that BFGS had reached but not
/// gnorm-verified. `nm_converged` is the Nelder–Mead convergence flag (as the pre-#555 code
/// reported), so the per-subject convergence/diagnostic semantics are unchanged; the η̂
/// *value* fed to the FOCEI gradient is what improves. A non-finite `obj(partial)` (NaN/∞)
/// makes the partial unusable so the NM result is taken.
fn argmin_inner_fallback(
    obj: &dyn Fn(&[f64]) -> f64,
    partial: &[f64],
    cold_seed: &[f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, bool) {
    let partial_f = obj(partial);
    let partial_usable = partial_f.is_finite();
    let warm = ebe_warm_start_enabled() && partial_usable;
    let mut eta_nm = if warm {
        partial.to_vec()
    } else {
        cold_seed.to_vec()
    };
    let nm_ok = nelder_mead_minimize(obj, &mut eta_nm, n, max_iter * 5, tol);
    let f_nm = obj(&eta_nm);
    // Keep the partial unless NM is *strictly* better. Written as a positive comparison so a
    // non-finite `f_nm` (NM diverged) leaves `nm_strictly_better = false` and the finite
    // partial is kept.
    let nm_strictly_better = f_nm < partial_f;
    let best = if partial_usable && !nm_strictly_better {
        partial.to_vec()
    } else {
        eta_nm
    };
    (best, nm_ok)
}

/// An ODE-based model that also carries IOV (`κ`) random effects. The inner EBE path for
/// these models is the expensive one this module special-cases (per-vertex ODE +
/// steady-state work), so both the Nelder-Mead skip and the start-rejection gate key off
/// this single classifier (#603 review #8).
fn is_ode_iov(model: &CompiledModel) -> bool {
    model.ode_spec.is_some() && model.n_kappa > 0
}

/// Nelder-Mead is a useful last-resort recovery for low-dimensional closed-form EBEs, but
/// it is not a practical recovery strategy for ODE+IOV. A single bad outer line-search
/// point can otherwise launch simplex searches where each vertex is a full ODE and
/// steady-state prediction. Keep the BFGS partial and report the subject unconverged
/// instead; the outer EBE guard can then reject the trial.
fn skip_ode_iov_nm_fallback(model: &CompiledModel) -> bool {
    is_ode_iov(model)
}

const ODE_IOV_START_REJECT_NLL_PER_OBS: f64 = 250.0;
const ODE_IOV_START_REJECT_NLL_MIN: f64 = 1_000.0;

fn reject_ode_iov_inner_start(model: &CompiledModel, n_obs: usize, nll: f64) -> bool {
    if !is_ode_iov(model) {
        return false;
    }
    if !nll.is_finite() {
        return true;
    }
    let threshold =
        ODE_IOV_START_REJECT_NLL_MIN.max(ODE_IOV_START_REJECT_NLL_PER_OBS * n_obs.max(1) as f64);
    nll > threshold
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
    // Custom / time-varying residual-magnitude (#484/#576): η-independent, so
    // computed once per subject here — not inside `agrad`, which BFGS calls on
    // every inner step (and every line-search trial) — instead of re-walking
    // every magnitude expression on each of those calls (#486 review).
    let mult = model.ruv_obs_mult(subject, &params.theta);

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
    //
    // Enable the objective-stall convergence stop only for ODE models, whose adaptive
    // RK45 objective carries a gradient-noise floor that can sit above `tol` (#555).
    // Analytical/event-driven objectives are exact, so they keep the pure `gnorm < tol`
    // criterion and stay bit-identical to prior releases.
    let enable_stall = model.ode_spec.is_some();
    // Single gradient closure used by *both* the optimizer and the fallback's stationarity
    // check, so the two agree on convergence: the exact analytic η-gradient (Almquist 2015,
    // one provider eval per step) when in scope with a per-point FD fallback, else FD
    // throughout. Checking the fallback with a *different* (FD) gradient than the analytic
    // one the BFGS converged on mislabels weakly-identified flat-basin EBEs (#587 review).
    let use_analytic = analytic_inner_grad_supported(model, subject);
    let profile = inner_profile_enabled();
    let agrad = |e: &[f64]| -> Vec<f64> {
        if !use_analytic {
            return gradient_fd(&obj, e, n_eta);
        }
        let t0 = std::time::Instant::now();
        match analytic_eta_nll_gradient_with_schedule(
            model,
            subject,
            &params.theta,
            e,
            &params.omega,
            &params.sigma.values,
            schedule.as_ref(),
            mult.as_deref(),
        ) {
            Some(g) => {
                GRADIENT_TIMINGS.record_analytic(t0.elapsed().as_nanos() as u64);
                if profile {
                    PROFILE_INNER_ANALYTIC_GRAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                g
            }
            None => {
                let g = gradient_fd(&obj, e, n_eta);
                GRADIENT_TIMINGS.record_fd(t0.elapsed().as_nanos() as u64);
                if profile {
                    PROFILE_INNER_FD_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                g
            }
        }
    };
    let result = inner_minimize_with_grad(
        &obj,
        &agrad,
        &mut eta,
        n_eta,
        max_iter,
        tol,
        precond.as_deref(),
        stop_precond,
        enable_stall,
    );

    // If BFGS failed, fall back to Nelder-Mead. The recovery policy depends on whether the
    // objective carries a gradient-noise floor (ODE) or is exact (analytical / event-driven /
    // FREM):
    //   * ODE (#555): the "failure" is usually a *certification* failure — the adaptive RK45
    //     gradient-noise floor blocks `gnorm < tol` at a genuine mode. Keep the lower-objective
    //     of {BFGS partial, NM-from-0} so a correct, lower-objective partial is never discarded
    //     for a worse NM basin (the #555 bug). See [`argmin_inner_fallback`].
    //   * Exact objectives: there is no noise floor, so a BFGS failure is genuine
    //     non-convergence and the partial may be a non-stationary, merely-low-objective point
    //     (e.g. run out along a FREM covariate pseudo-obs flat direction). Keeping it would
    //     mis-center the FREM/IMP proposal, so recover with NM from η=0 (or the warm partial)
    //     exactly as prior releases — bit-identical for analytical/FREM fits.
    let bfgs_converged = result;
    let (nm_converged, used_fallback) = if !bfgs_converged {
        let partial = eta.clone();
        let cold = vec![0.0; n_eta];
        if enable_stall {
            let (best, ok) = argmin_inner_fallback(&obj, &partial, &cold, n_eta, max_iter, tol);
            eta = best;
            (ok, true)
        } else {
            let warm = ebe_warm_start_enabled() && partial.iter().all(|v| v.is_finite());
            eta = if warm { partial } else { cold };
            let nm_ok = nelder_mead_minimize(&obj, &mut eta, n_eta, max_iter * 5, tol);
            (nm_ok, true)
        }
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
    let analytic_jac: Option<DMatrix<f64>> = if analytic_inner_grad_supported(model, subject) {
        crate::sens::provider::subject_eta_jacobian(model, subject, &params.theta, &eta_true)
            .map(|j| DMatrix::from_row_slice(subject.obs_times.len(), n_eta, &j))
            .filter(|j| j.iter().all(|v| v.is_finite()))
    } else {
        None
    };
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
        hard_reject: false,
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

    let occ_groups = iov_occasion_groups(subject);
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

    // IOV models are not FREM, so no inner preconditioner (identity H0). Use the exact
    // analytic stacked-η gradient when the ODE IOV provider serves this model (one
    // provider eval per inner step vs ~2·n_flat+1 predictions for FD, and exact → fewer
    // steps); per-step FD fallback if a given (θ, stacked-η) is out of provider scope.
    // Covers both the ODE IOV provider (RHS-program models) and the closed-form
    // analytical IOV provider — both now expose an analytic stacked-η inner gradient
    // via `subject_eta_grad_iov`.
    // Mirror the non-IOV inner bails (#466 review #1/#2/#3): the IOV `Dual2`/`Dual1`
    // kernels share the same limitations, so route to the FD inner loop when the model
    // hits a common bail (escape hatch, `gradient = fd`, SDE, LTBS, or IIV on residual
    // error) or the subject carries survival/TTE records (whose hazard term the analytic
    // IOV gradient omits). An η-dependent `ExpressionScale` `obs_scale` is no longer a
    // common bail (#486 made the non-IOV inner serve it analytically). For IOV it is
    // now served analytically too (#575): `ode_iov_supported` admits a non-LTBS
    // `ExpressionScale` divisor, so `iov_sens_supported` is `true` and the
    // `analytic_iov_inner` path applies the per-occasion-group post-walk quotient
    // (`apply_expression_scale_iov`). A constant `ScalarScale` divisor is now analytic under
    // IOV on both engines (#486 parity — closed-form `run_obs_iov{,_eta}` divide the jet by
    // `k`; the ODE readout `apply_output_transform` already divides `p/k` in-walk over the
    // stacked dual). LTBS still routes IOV to FD via `analytic_inner_common_bail`
    // (`log_transform`). Without these guards a joint IOV + `iiv_on_ruv` / IOV + TTE /
    // `gradient = fd` fit would converge EBEs against an incomplete gradient.
    let analytic_iov_inner = crate::sens::provider::iov_sens_supported(model)
        && omega_iov_ref.is_some()
        && !analytic_inner_common_bail(model)
        && !subject_has_survival_records(subject);
    // ODE objectives carry the adaptive-solver gradient-noise floor; enable the
    // objective-stall stop only for them (see `find_ebe`).
    let enable_stall = model.ode_spec.is_some();
    // Custom / time-varying residual-magnitude (#484/#576): η-independent, so
    // computed once per subject here rather than inside `agrad` (see `find_ebe`).
    let mult = model.ruv_obs_mult(subject, &params.theta);
    // One gradient closure for both the optimizer and the fallback stationarity check
    // (analytic stacked-η IOV gradient when in scope, else FD), so they agree on
    // convergence — see the matching note in `find_ebe`.
    let agrad = |p: &[f64]| -> Vec<f64> {
        if !analytic_iov_inner {
            return gradient_fd(&obj, p, n_flat);
        }
        let omega_iov = omega_iov_ref.expect("analytic_iov_inner requires omega_iov");
        // Recover stacked_true = [η_true (= psi − mu), κ…] from the psi-space `p`; the
        // gradient is identical in psi- and η_true-space (constant `mu` shift).
        let mut stacked_true = p.to_vec();
        for (k, st) in stacked_true.iter_mut().take(n_eta).enumerate() {
            *st = p[k] - mu[k];
        }
        match analytic_eta_nll_gradient_iov(
            model,
            subject,
            &params.theta,
            &stacked_true,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k_occasions,
            mult.as_deref(),
        ) {
            Some(g) => g,
            None => gradient_fd(&obj, p, n_flat),
        }
    };

    let start_nll = obj(&x);
    let has_informative_warm_start = eta_init
        .map(|warm| warm.iter().any(|v| v.abs() > 1e-8))
        .unwrap_or(false);
    if has_informative_warm_start
        && reject_ode_iov_inner_start(model, subject.obs_times.len(), start_nll)
    {
        let bsv_eta: Vec<f64> = x[..n_eta]
            .iter()
            .zip(mu.iter())
            .map(|(p, m)| p - m)
            .collect();
        let kappas_vec: Vec<DVector<f64>> = (0..k_occasions)
            .map(|k| DVector::from_column_slice(&x[n_eta + k * n_kappa..n_eta + (k + 1) * n_kappa]))
            .collect();
        // Degenerate placeholder: the η/κ here are the (un-optimized) warm start and the
        // H-matrix is zero — folding them into an OFV would corrupt the FOCEI curvature
        // term. `hard_reject` makes the outer guard reject the whole trial rather than
        // average this in, so the placeholder is never compared as a real OFV (#603 #1/#2).
        return EbeResult {
            eta: DVector::from_column_slice(&bsv_eta),
            h_matrix: DMatrix::zeros(subject.obs_times.len(), n_eta),
            converged: false,
            used_fallback: false,
            grad_norm: 0.0,
            nll: start_nll,
            kappas: kappas_vec,
            hard_reject: true,
        };
    }

    let bfgs_converged = inner_minimize_with_grad(
        &obj,
        &agrad,
        &mut x,
        n_flat,
        max_iter,
        tol,
        None,
        None,
        enable_stall,
    );
    // On BFGS failure, recover with the same ODE-gated policy as the non-IOV `find_ebe`
    // (cold seed = prior mode `bsv_psi = μ`, κ = 0): for ODE objectives keep the
    // lower-objective of {BFGS partial, NM restart} so a correct η̂ floored above `tol` by
    // solver noise is never discarded (#555); for exact objectives recover with NM from the
    // cold seed, as prior releases, so a non-stationary low-objective partial can't be kept.
    let (nm_converged, used_fallback) = if !bfgs_converged {
        let partial = x.clone();
        let mut cold = vec![0.0; n_flat];
        cold[..n_eta].copy_from_slice(&mu);
        if skip_ode_iov_nm_fallback(model) {
            (false, false)
        } else if enable_stall {
            let (best, ok) = argmin_inner_fallback(&obj, &partial, &cold, n_flat, max_iter, tol);
            x = best;
            (ok, true)
        } else {
            let warm = ebe_warm_start_enabled() && partial.iter().all(|v| v.is_finite());
            x = if warm { partial } else { cold };
            let nm_ok = nelder_mead_minimize(&obj, &mut x, n_flat, max_iter * 5, tol);
            (nm_ok, true)
        }
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

    // H-matrix: BSV columns only (∂f/∂η_bsv with κ fixed at the EBE). The BSV block of
    // the analytic stacked-η Jacobian is exactly this, so reuse the provider when it
    // serves this subject; else the FD Jacobian.
    let h_matrix = {
        let analytic = if analytic_iov_inner {
            let mut stacked_hat = bsv_eta.clone();
            for k in &kappas_vec {
                stacked_hat.extend(k.iter().copied());
            }
            crate::sens::provider::subject_eta_grad_iov(model, subject, &params.theta, &stacked_hat)
        } else {
            None
        };
        match analytic {
            // Require one sensitivity row per observation — the indexed writes below would
            // otherwise panic and abort the fit. The provider's scope gates hold this
            // invariant today; guard it so a future mismatch degrades to FD instead of
            // crashing, mirroring the outer `subject_packed_gradient_foce_iov` check (#466
            // review round 4 #7).
            Some(sens) if sens.len() == subject.obs_times.len() => {
                let n_obs = subject.obs_times.len();
                let mut h = DMatrix::zeros(n_obs, n_eta);
                for (j, obs) in sens.iter().enumerate() {
                    for k in 0..n_eta {
                        h[(j, k)] = obs.df_deta[k];
                    }
                }
                // Match the FD path: FREM covariate pseudo-obs rows carry the exact {0,1}
                // Jacobian, which the provider's PK-prediction sensitivity does not emit
                // (#466 review #9).
                overwrite_frem_pseudo_obs_rows(&mut h, model, subject, n_eta);
                h
            }
            _ => {
                let kappas_slices: Vec<Vec<f64>> =
                    kappas_vec.iter().map(|k| k.as_slice().to_vec()).collect();
                compute_jacobian_fd_iov(model, subject, &params.theta, &bsv_eta, &kappas_slices)
            }
        }
    };

    EbeResult {
        eta: DVector::from_column_slice(&bsv_eta),
        h_matrix,
        converged: (bfgs_converged || nm_converged) && nll.is_finite(),
        used_fallback,
        grad_norm: 0.0,
        nll,
        kappas: kappas_vec,
        hard_reject: false,
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
    overwrite_frem_pseudo_obs_rows(&mut h, model, subject, n_eta);

    h
}

/// Overwrite FREM covariate pseudo-observation rows of an `n_obs × n_eta` BSV H-matrix
/// with their exact `{0, 1}` Jacobian (`∂(pseudo-obs)/∂η_k = δ_{k, eta_idx}`). Applied to
/// both the FD and the analytic IOV H-matrix so the analytic branch does not silently drop
/// the correction the FD path performs (#466 review #9). No-op when the model isn't FREM
/// or the subject carries no pseudo-obs rows.
fn overwrite_frem_pseudo_obs_rows(
    h: &mut DMatrix<f64>,
    model: &CompiledModel,
    subject: &Subject,
    n_eta: usize,
) {
    let Some(ref fc) = model.frem_config else {
        return;
    };
    if subject.fremtype.is_empty() {
        return;
    }
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

/// The model-level inner-gradient bails that are independent of which analytic inner
/// provider (non-IOV analytical, non-IOV ODE `Dual1`, or IOV) will serve the model.
/// Returns `true` when the model must use the **FD** inner gradient regardless: the
/// escape hatch / A-B toggle, an explicit `gradient = fd`, SDE diffusion, or
/// IIV on residual error (`iiv_on_ruv`, whose `exp(2·η_ruv)` variance scaling none of
/// the `Dual2`/`Dual1` kernels carry).
/// Every analytic inner path consults this so none of them can run on a model that one
/// of these reasons routes to FD — including the IOV inner loop, which previously dropped
/// these exclusions (#466 review #1/#3).
///
/// LTBS is **no longer** a blanket common bail. The plain closed-form inner provider applies
/// the `g = ln(f)` jet transform (`subject_eta_grad` → `run_obs_eta`), so it serves plain LTBS
/// analytically; the covariance step reconverges those EBEs at the tighter `cov_inner_tol`
/// ([`FitOptions::effective_cov_inner_tol`]) so the `ln`-amplified EBE noise no longer corrupts
/// the SEs. LTBS + η-dependent `ExpressionScale` stays a (narrower) common bail — its analytic
/// scale+log Jacobian is unvalidated. The other LTBS inner paths the analytic kernels don't yet
/// carry decline through their own gates — TV-cov (`subject_eta_grad_tvcov`) and closed-form IOV
/// (`iov_analytical_supported`) — so removing the blanket bail only enables the validated plain
/// path.
///
/// An eta-dependent `ExpressionScale` obs_scale is **not** a common bail: the non-IOV
/// analytical inner provider now carries the η-only quotient rule (`subject_eta_grad`
/// → `apply_expression_scale_inner`), and the ODE inner provider serves it on the static
/// walk *and* the TV-cov event-driven walk (#534/#486 — the scale is subject-static even
/// under time-varying covariates, so one post-walk quotient covers both), so both run
/// analytically. The **IOV** inner path serves it too (#486): both the closed-form
/// (`subject_eta_grad_iov_analytical` → `run_obs_iov_eta`) and the ODE
/// (`ode_subject_eta_grad_iov`) IOV inner walks apply a per-occasion-group post-walk
/// quotient, so `iov_analytical_supported` / `ode_iov_supported` now admit `ExpressionScale`
/// and the inner and outer loops stay matched. The **ODE** inner path does not consult this
/// common bail at all — it has its own inline bail list in [`analytic_inner_grad_supported`]
/// and its own per-subject scope (`ode_inner_grad_supported`, which admits exactly the
/// static-walk and TV-cov-walk `ExpressionScale` that the ODE provider actually applies).
pub(crate) fn analytic_inner_common_bail(model: &CompiledModel) -> bool {
    no_analytic_inner_forced()
        || matches!(model.gradient_method, GradientMethod::Fd)
        || model.is_sde()
        // LTBS is served analytically on the inner loop now — plain, × `ExpressionScale`
        // (the η-quotient then the `ln f` jet, `subject_eta_grad`), and × TV-cov (the
        // event-driven inner walk applies the same jet LAST). LTBS × IOV, however, stays on
        // the FD inner: the closed-form OUTER IOV gradient now serves LTBS (the `ln(f)` jet in
        // `subject_sensitivities_iov`, #486), so `iov_analytical_supported` admits
        // `log_transform` — but `run_obs_iov_eta` (the Dual1 inner walk) carries no `ln` jet,
        // so decline the inner here to keep the EBE reconvergence on FD (the IOV twin of how
        // plain/TV-cov LTBS is served on the inner but IOV is not).
        || (model.log_transform && model.n_kappa > 0)
        // Correlated residual error (`block_sigma`) is now served analytically by the
        // dense-R inner gradient (`dense_residual_inner_gradient`, #627), so it is no
        // longer a blanket bail. (An eta-dependent `ExpressionScale` is NOT a bail either.)
        // Custom residual-error magnitude (#484) alone stays analytic (#576/#486 —
        // `residual_inner_obs` threads the η-independent per-obs multiplier). Only the
        // *combination* with a correlated residual bails: the dense-R inner kernel does
        // not carry the magnitude's θ-dependence, so route it to FD (magnitude-aware).
        || (model.has_custom_ruv_magnitude() && !model.residual_correlations.is_empty())
    // `iiv_on_ruv` needs no bail here: residual-η is served analytically on both loops for
    // every combination — closed-form and ODE, plain / IOV / M3-BLOQ including the triples
    // (#474/#4b/#4c/#591/#623/#677); the scaling and the censored/quantified `η_ruv` terms
    // live in the shared, provider-agnostic gradient.
}

/// True when the subject carries survival/TTE observation records, whose hazard-likelihood
/// term neither analytic inner provider models — both the non-IOV and IOV inner gradients
/// decline such subjects (a single source so the two cannot drift; #466 review round 2).
/// Always `false` without the `survival` feature.
#[inline]
pub(crate) fn subject_has_survival_records(subject: &Subject) -> bool {
    #[cfg(feature = "survival")]
    {
        !subject.obs_records.is_empty()
    }
    #[cfg(not(feature = "survival"))]
    {
        let _ = subject;
        false
    }
}

/// Model-level half of [`analytic_inner_grad_supported`]: every gate that does
/// not depend on the subject. `build_info::gradient_method_inner` reports the
/// inner route off **this same** predicate, so the reported `gradient_method_inner`
/// cannot drift from what `find_ebe` actually runs (PR #381 review #9).
pub(crate) fn analytic_inner_grad_supported_model(model: &CompiledModel) -> bool {
    // Escape hatch, explicit `gradient = fd`, SDE, and the `iiv_on_ruv` cases that
    // force FD all revert the inner EBE gradient to FD (see `analytic_inner_common_bail`
    // for the per-reason rationale). LTBS is now served analytically on the plain
    // closed-form inner too (the `g = ln(f)` jet in `subject_eta_grad`), with the
    // covariance step reconverging at the tighter `cov_inner_tol`; the LTBS paths the
    // kernels don't carry (TV-cov, IOV, `ExpressionScale`) still decline via their own
    // gates. An eta-dependent `ExpressionScale` obs_scale is served on *both* loops
    // (#534/#486) so it is not a bail here either.
    if analytic_inner_common_bail(model) {
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
    if subject_has_survival_records(subject) {
        return false;
    }
    // ODE models use the light `Dual1` inner provider (#410) with their own
    // per-subject scope ([`ode_inner_grad_supported`]). The global escape hatches
    // plus the model-level exclusions the analytical path applies in
    // `analytic_inner_grad_supported_model` still hold here:
    //   - IIV on residual error (#409/#474): the residual-variance scaling
    //     `exp(2·η_ruv)` and the `η_ruv` variance column live in the shared
    //     `analytic_eta_nll_gradient_with_schedule` (provider-agnostic), so the
    //     light Dual1 ODE walk serves these models too. M3 BLOQ + `iiv_on_ruv`
    //     keeps FD (the censored residual-eta second derivatives are not assembled).
    //   - LTBS: the ODE `Dual1` walk shares `solve_ode_g` with the objective, so the
    //     analytic-EBE *is* the objective's own minimum — the gradient matches FD of
    //     `individual_nll` and the analytic/FD EBEs agree to integrator tolerance,
    //     leaving the covariance Hessian clean. Validated by
    //     `ode_ltbs_inner_grad_matches_fd` / `ode_ltbs_inner_ebe_matches_fd` (#474). The
    //     closed-form inner now serves LTBS too: its provider closed forms agree with
    //     `compute_predictions` only to ~1e-9, which the `g = ln(f)` wrap amplifies into
    //     the covariance Hessian, so it relies on the covariance step reconverging at the
    //     tighter `cov_inner_tol` rather than on shared code (see
    //     `FitOptions::effective_cov_inner_tol`).
    if model.ode_spec.is_some() {
        // The ODE inner path does NOT bail on LTBS or `ExpressionScale`: the `Dual1` ODE
        // walk shares `solve_ode_g` with the objective, so ODE-LTBS takes the analytic
        // inner gradient (#474). Only the escape hatch / `gradient = fd` / SDE /
        // magnitude-×-`block_sigma` cases revert here.
        if no_analytic_inner_forced()
            || matches!(model.gradient_method, GradientMethod::Fd)
            || model.is_sde()
            // Correlated residual (`block_sigma`, #627) now served analytically: the
            // dense-R inner gradient reuses the Dual1 walk's per-obs `∂f/∂η`, so no bail.
            // Custom residual-error magnitude (#484) alone stays analytic (#576/#486 —
            // `residual_inner_obs` threads the η-independent per-obs multiplier through
            // both the closed-form and Dual1 ODE inner paths). Only the *combination*
            // with a correlated residual bails: the dense-R inner kernel does not carry
            // the magnitude's θ-dependence — matching `analytic_inner_common_bail`.
            || (model.has_custom_ruv_magnitude() && !model.residual_correlations.is_empty())
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
    //
    // A `TIME`-built-in structural parameter routes through the same per-event walk
    // (#486), so it must consult `tvcov_analytical_supported` too — otherwise a TIME
    // model that the walk declines (e.g. TIME + `[initial_conditions]`) would report
    // an analytic inner here while `subject_eta_grad` returns `None`, splitting the
    // inner route from the outer. LTBS now takes the analytic inner gradient on the
    // event-driven walk too (`subject_eta_grad_tvcov` applies the `ln f` jet LAST), so
    // no `!log_transform` carve-out is needed here.
    if crate::sens::provider::subject_routes_to_event_walk(model, subject) {
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
/// `∂/∂f` of the (uncensored) Gaussian per-observation data term `½ log v + ½ ε²/v`
/// (`ε = y − f`, `v` the residual variance, `dv_df = ∂v/∂f`). Multiplying by `∂f/∂η`
/// gives that observation's contribution to the conditional-NLL η-gradient. Shared by the
/// non-IOV ([`analytic_eta_nll_gradient_with_schedule`]) and IOV
/// ([`analytic_eta_nll_gradient_iov`]) inner gradients so the two cannot silently diverge
/// (#466 review #10).
#[inline]
fn obs_gaussian_dterm_coef(y: f64, f: f64, v: f64, dv_df: f64) -> f64 {
    let eps = y - f;
    -eps / v + 0.5 * dv_df * (1.0 / v - eps * eps / (v * v))
}

/// Residual-eta (`iiv_on_ruv`) data-gradient term for a *quantified* observation:
/// `∂(½ε²/v + ½ln v)/∂η_ruv = 1 − ε²/v` (the `∂v/∂η_ruv = 2v` factor cancels the ½),
/// with `v` the `exp(2·η_ruv)`-scaled residual variance. Single source for the
/// production non-IOV and IOV inner gradients so they cannot drift (#474 review). The
/// M3-censored row uses `h·z` instead — see the call sites.
#[inline]
pub(crate) fn ruv_data_dterm(eps: f64, v: f64) -> f64 {
    1.0 - eps * eps / v
}

/// Per-observation residual data-gradient pieces shared by the non-IOV
/// ([`analytic_eta_nll_gradient_with_schedule`]) and IOV
/// ([`analytic_eta_nll_gradient_iov`]) inner gradients, so the residual-eta
/// convention (scaling + the M3-censored vs Gaussian branch) lives in one place.
/// Returns `(coef, ruv_term)`: every η gets `∂nll/∂η_k += coef·∂f/∂η_k`, and (when
/// `iiv_on_ruv` is active) the residual-eta axis gets `∂nll/∂η_ruv += ruv_term`. A
/// quantified row uses the Gaussian coef + `1 − ε²/v`; an M3-censored row the single
/// kernel eval's `h·m` (coef) + `h·z` (column). `None` on a non-positive variance.
/// `ruv_scale` is applied only when `ruv_active`, so a plain model keeps its op count.
///
/// `mult` is the observation's custom-magnitude multiplier row (#484/#576),
/// `None` reproducing the legacy unscaled variance. The magnitude is
/// η-independent, so this is the *entire* inner-loop change it needs: no new η
/// term, just the scale on `v`/`dv_df` (the direct-θ dependence is a separate,
/// outer-only gradient channel — see `sens_outer_gradient::prepare_stacked`).
#[inline]
fn residual_inner_obs(
    model: &CompiledModel,
    cmt: usize,
    y: f64,
    f: f64,
    sigma: &[f64],
    mult: Option<&[f64]>,
    ruv_scale: f64,
    ruv_active: bool,
    cens: i8,
) -> Option<(f64, f64)> {
    let mut v = match mult {
        Some(m) => model.residual_variance_at_scaled(cmt, f, sigma, Some(m)),
        None => model.residual_variance_at(cmt, f, sigma),
    };
    let mut dv_df = match mult {
        Some(m) => model.error_spec.dvar_df_scaled(cmt, f, sigma, m),
        None => model.error_spec.dvar_df(cmt, f, sigma),
    };
    if ruv_active {
        v *= ruv_scale;
        dv_df *= ruv_scale;
    }
    if !(v > 0.0) {
        return None;
    }
    let (coef, ruv_term) = if cens != 0 {
        // Signed kernel: right-censored (`cens < 0`) rows use the upper tail, so
        // `h·m` / `h·z` match `individual_nll_iov`'s `m3_logcdf` data term.
        let (h, z, m) = crate::stats::special::m3_censored_kernel(y, f, v, dv_df, cens);
        (h * m, if ruv_active { h * z } else { 0.0 })
    } else {
        let ruv_term = if ruv_active {
            ruv_data_dterm(y - f, v)
        } else {
            0.0
        };
        (obs_gaussian_dterm_coef(y, f, v, dv_df), ruv_term)
    };
    Some((coef, ruv_term))
}

/// Censored data-term f-coefficient `∂(−logΦ)/∂f = h·m`. Production computes it
/// inline (sharing the one kernel eval with the `h·z` column); retained for the
/// `m3_censored_dterm_df_matches_fd` unit test.
#[cfg(test)]
#[inline]
fn m3_censored_dterm_df(y: f64, f: f64, v: f64, dv_df: f64, cens: i8) -> f64 {
    let (h, _z, m) = crate::stats::special::m3_censored_kernel(y, f, v, dv_df, cens);
    h * m
}

/// Exact analytic `∂NLL_i/∂η` from the light first-order sensitivity provider:
/// `Σ_j (∂nll/∂f_j)·(∂f_j/∂η) + Ω⁻¹η`. `Some` only when the model is in the
/// provider's scope (returns `None` for ODE / TV-cov / oral-infusion / SS+reset /
/// LTBS subjects). A η-dependent `ExpressionScale` `obs_scale` is in scope as of
/// #486 (the quotient rule is applied to the η-block), except when combined with
/// LTBS, which still declines. Shared by the inner EBE loop and the HMC sampler so
/// both estimators use the same Dual2 gradient (replacing the retired Enzyme path).
pub(crate) fn analytic_eta_nll_gradient(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma: &[f64],
) -> Option<Vec<f64>> {
    // Custom / time-varying residual-magnitude (#484/#576): η-independent, so a
    // one-off caller like this can just compute it inline (unlike `find_ebe`'s
    // per-BFGS-step closure, which hoists it — see `analytic_eta_nll_gradient_with_schedule`).
    let mult = model.ruv_obs_mult(subject, theta);
    analytic_eta_nll_gradient_with_schedule(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma,
        None,
        mult.as_deref(),
    )
}

/// As [`analytic_eta_nll_gradient`], but reusing the per-subject `EventSchedule` the
/// inner optimizer cached once, so the TV-cov provider doesn't rebuild it every inner
/// BFGS step (#449 re-review #6). `None` rebuilds locally.
///
/// `mult` is the subject's custom-magnitude multiplier matrix (#484/#576,
/// [`CompiledModel::ruv_obs_mult`]) — the caller computes it, so a per-BFGS-step
/// closure (`find_ebe`'s `agrad`) can compute it **once** outside the loop instead
/// of re-walking every magnitude expression on every inner iteration (#486 review).
/// `None` when no magnitude is active.
#[allow(clippy::too_many_arguments)]
pub(crate) fn analytic_eta_nll_gradient_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma: &[f64],
    cached_schedule: Option<&crate::pk::event_driven::EventSchedule>,
    mult: Option<&[Vec<f64>]>,
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
    // Correlated residual (`block_sigma`, #627): the per-obs `coef·∂f/∂η` loop below
    // assumes a diagonal R. Route the dense-R generalisation here — it serves both the
    // analytical and the ODE (`Dual1`) inner path, since `sens` carries `∂f/∂η` for both.
    if !model.residual_correlations.is_empty() {
        return dense_residual_inner_gradient(model, subject, theta, eta, omega, sigma, &sens);
    }
    let n_eta = model.n_eta;
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    // IIV on residual error (`Y = IPRED + EPS·EXP(η_ruv)`, #409/#474): the residual
    // variance of every observation scales by `s = exp(2·η_ruv)`, so `v` and
    // `dv_df` carry that factor. `η_ruv` enters the likelihood only through the
    // variance (`∂f/∂η_ruv = 0`), so its gradient column is the variance term
    // `Σ_j (1 − ε²/v)`, plus the `Ω⁻¹η` prior added below — not the shared
    // `coef·∂f/∂η` loop. (M3 censoring + `iiv_on_ruv` routes to FD upstream, so the
    // residual-eta column is only ever formed on quantified rows here.)
    let ruv_idx = model.residual_error_eta;
    let ruv_active = ruv_idx.is_some();
    let ruv_scale = if ruv_active {
        model.residual_var_scale(eta)
    } else {
        1.0
    };
    let mut grad = vec![0.0_f64; n_eta];
    let mut ruv_grad = 0.0_f64;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    for (j, obs) in sens.iter().enumerate() {
        let cens = if m3 {
            subject.cens.get(j).copied().unwrap_or(0)
        } else {
            0
        };
        let (coef, ruv_term) = residual_inner_obs(
            model,
            err_keys[j],
            subject.observations[j],
            obs.f,
            sigma,
            mult.and_then(|m| m.get(j)).map(|v| v.as_slice()),
            ruv_scale,
            ruv_active,
            cens,
        )?;
        for k in 0..n_eta {
            grad[k] += coef * obs.df_deta[k];
        }
        ruv_grad += ruv_term; // 0 unless `iiv_on_ruv`
    }
    if let Some(r) = ruv_idx {
        grad[r] += ruv_grad;
    }
    // Prior: ∂/∂η ½ η'Ω⁻¹η = Ω⁻¹η.
    let eta_v = nalgebra::DVector::from_column_slice(eta);
    let prior = &omega.inv * &eta_v;
    for (k, g) in grad.iter_mut().enumerate() {
        *g += prior[k];
    }
    Some(grad)
}

/// Dense-`R` (`block_sigma`, #627) analytic inner η-gradient — the correlated-residual
/// generalisation of the per-obs `coef·∂f/∂η` loop in
/// [`analytic_eta_nll_gradient_with_schedule`]. `block_sigma` is rejected up front
/// together with M3, FREM, IOV (κ) and `iiv_on_ruv`, so this path is pure Gaussian.
///
/// For the inner NLL data term `½(rᵀR⁻¹r + log|R|)` with `r = y − f(η)`,
/// `∂r/∂η_k = −a_k`, `∂R/∂η_k = Σ_m H[m,k]·∂R/∂f_m`, `s = R⁻¹r`, `M_k = R⁻¹∂R/∂η_k`:
/// ```text
///   ∂NLL_data/∂η_k = −a_kᵀ s + ½( tr(M_k) − sᵀ ∂R/∂η_k s )
/// ```
/// plus the `Ω⁻¹η` prior. Reduces exactly to the diagonal `coef·∂f/∂η` loop when
/// `R` is diagonal. `R` / `∂R/∂f` are built exactly as
/// [`crate::stats::likelihood::foce_subject_nll_interaction_dense`] (same
/// `compute_r_matrix_with_correlations` / `compute_dr_df_matrices`, same `#484`
/// magnitude multiplier) so the inner EBE stays consistent with the marginal it
/// optimises. `sens` supplies per-obs `∂f/∂η` for both the analytical and the ODE
/// (`Dual1`) provider, so this one branch serves both inner paths.
#[allow(clippy::too_many_arguments)]
fn dense_residual_inner_gradient(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &crate::types::OmegaMatrix,
    sigma: &[f64],
    sens: &[crate::sens::provider::ObsGrad],
) -> Option<Vec<f64>> {
    use nalgebra::{DMatrix, DVector};
    let n_eta = model.n_eta;
    let eta_v = DVector::from_column_slice(eta);
    let prior = &omega.inv * &eta_v;
    let n_obs = sens.len();
    if n_obs == 0 {
        return Some(prior.as_slice().to_vec());
    }
    let ipreds: Vec<f64> = sens.iter().map(|o| o.f).collect();
    let corr = &model.residual_correlations;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    // Per-observation custom residual magnitude (#484); η-independent, matches the marginal.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let r = match ruv_mult.as_deref() {
        Some(mult) => crate::stats::residual_error::compute_r_matrix_with_correlations_scaled(
            &model.error_spec,
            &ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma,
            corr,
            mult,
        ),
        None => crate::stats::residual_error::compute_r_matrix_with_correlations(
            &model.error_spec,
            &ipreds,
            err_keys.as_ref(),
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma,
            corr,
        ),
    };
    let chol = r.clone().cholesky()?;
    let r_inv = chol.inverse();
    let residuals = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(ipreds.iter())
            .map(|(&y, &f)| y - f),
    );
    let s = chol.solve(&residuals); // R⁻¹ r
    let dr = crate::stats::residual_error::compute_dr_df_matrices(
        &model.error_spec,
        &ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        sigma,
        corr,
        ruv_mult.as_deref(),
    );
    let mut grad = vec![0.0f64; n_eta];
    for k in 0..n_eta {
        // −a_kᵀ s, where a_k = (∂f_m/∂η_k)_m is column k of H.
        let mut term_a = 0.0;
        for m in 0..n_obs {
            term_a += sens[m].df_deta[k] * s[m];
        }
        // ∂R/∂η_k = Σ_m H[m,k]·∂R/∂f_m.
        let mut dr_k = DMatrix::<f64>::zeros(n_obs, n_obs);
        for m in 0..n_obs {
            let h_mk = sens[m].df_deta[k];
            if h_mk != 0.0 {
                dr_k += h_mk * &dr[m];
            }
        }
        // tr(M_k) = tr(R⁻¹ ∂R/∂η_k) = Σ_{p,q} r_inv[p,q]·dr_k[q,p].
        let mut tr_mk = 0.0;
        for p in 0..n_obs {
            for q in 0..n_obs {
                tr_mk += r_inv[(p, q)] * dr_k[(q, p)];
            }
        }
        // sᵀ ∂R/∂η_k s.
        let quad = s.dot(&(&dr_k * &s));
        grad[k] = -term_a + 0.5 * (tr_mk - quad) + prior[k];
    }
    Some(grad)
}

/// Analytic gradient of the IOV conditional NLL (`individual_nll_iov`) w.r.t. the
/// stacked random-effects vector `[η_bsv, κ₁..κ_K]` (in `eta_true` space, i.e. the κ
/// and BSV-η values, not the psi-shifted optimiser variable). `None` when the analytic
/// inner provider can't serve this `(model, subject)` — the caller falls back to FD.
///
/// Data term: `Σ_obs coef·∂f/∂(stacked-η)` with the same `coef` as the non-IOV
/// [`analytic_eta_nll_gradient`]. Prior term: the **block-diagonal** `Σ_b⁻¹·stacked`
/// (`Σ_b = Ω_bsv ⊕ K·Ω_iov`) — `Ω_bsv⁻¹·η_bsv` on the BSV block and `Ω_iov⁻¹·κ_g` on
/// each occasion block. The BSV-η gradient equals the gradient w.r.t. the psi-space
/// optimiser variable (a constant `mu` shift drops out), and κ is unshifted, so the
/// returned vector is directly the optimiser gradient (#439 ODE IOV).
///
/// `mult` is the subject's custom-magnitude multiplier matrix (#484/#576,
/// [`CompiledModel::ruv_obs_mult`]), computed once by the caller and shared with
/// the non-IOV inner (`analytic_eta_nll_gradient_with_schedule`) — see its doc for
/// why this is a caller-supplied parameter rather than computed here.
#[allow(clippy::too_many_arguments)]
fn analytic_eta_nll_gradient_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    stacked_true: &[f64],
    omega_bsv: &crate::types::OmegaMatrix,
    omega_iov: &crate::types::OmegaMatrix,
    sigma: &[f64],
    n_eta: usize,
    n_kappa: usize,
    k_occasions: usize,
    mult: Option<&[Vec<f64>]>,
) -> Option<Vec<f64>> {
    let sens = crate::sens::provider::subject_eta_grad_iov(model, subject, theta, stacked_true)?;
    let n_stacked = n_eta + k_occasions * n_kappa;
    // IIV on residual error (`iiv_on_ruv`, #474) for IOV models: every residual variance
    // scales by `s = exp(2·η_ruv)` (η_ruv lives in the BSV block of the stacked vector),
    // so `v`/`dv_df` carry that factor and the `η_ruv` column gets the variance term
    // `Σ_j (1 − ε²/v)` — exactly the non-IOV treatment in
    // `analytic_eta_nll_gradient_with_schedule`. `residual_var_scale` returns `1.0` when
    // no `iiv_on_ruv` is declared, so a plain IOV model is unaffected. (M3-censored rows are
    // handled by the censored branch below, #580/#591.)
    // Only pay the `exp(2·η_ruv)` scaling + the `η_ruv` column when `iiv_on_ruv` is
    // active; a plain IOV model runs the original op count (no per-obs ×1.0 multiplies
    // and no residual-eta accumulation — #474 review).
    let ruv_idx = model.residual_error_eta;
    let ruv_active = ruv_idx.is_some();
    let ruv_scale = if ruv_active {
        model.residual_var_scale(stacked_true)
    } else {
        1.0
    };
    // M3 BLOQ + IOV (#580): a censored row's data term is `−logΦ(z)`, matching
    // `individual_nll_iov`'s `−2·m3_logcdf` (the inner objective `find_ebe_iov`
    // minimises). `residual_inner_obs` emits its `h·m` f-coefficient over the stacked
    // Jacobian, so the EBE minimises the same censored objective. The triple
    // M3 + IOV + `iiv_on_ruv` is analytic too (#591): on a censored row with
    // `ruv_active`, `residual_inner_obs` also returns the `h·z` residual-eta column
    // (the `η_ruv` index lives in the BSV block of the stacked vector). The ODE triple is
    // analytic as well (#486/#623) — every `iiv_on_ruv` combination is served.
    let m3 = matches!(model.bloq_method, crate::types::BloqMethod::M3);
    let mut grad = vec![0.0_f64; n_stacked];
    let mut ruv_grad = 0.0_f64;
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    for (j, obs) in sens.iter().enumerate() {
        let cens = if m3 {
            subject.cens.get(j).copied().unwrap_or(0)
        } else {
            0
        };
        // The residual logic is shared with the non-IOV inner via `residual_inner_obs`
        // so the two cannot drift (Gaussian coef + `1 − ε²/v`, or the M3-censored `h·m`).
        // The signed `cens` makes right-censored rows use the upper tail.
        let (coef, ruv_term) = residual_inner_obs(
            model,
            err_keys[j],
            subject.observations[j],
            obs.f,
            sigma,
            mult.and_then(|m| m.get(j)).map(|v| v.as_slice()),
            ruv_scale,
            ruv_active,
            cens,
        )?;
        for (p, g) in grad.iter_mut().enumerate() {
            *g += coef * obs.df_deta[p];
        }
        ruv_grad += ruv_term; // 0 unless `iiv_on_ruv`
    }
    if let Some(r) = ruv_idx {
        grad[r] += ruv_grad;
    }
    // Prior: block-diagonal Σ_b⁻¹·stacked. BSV block Ω_bsv⁻¹·η_bsv, each occasion κ
    // block Ω_iov⁻¹·κ_g (the κ-variance is shared across occasions — SAME).
    let eta_bsv = DVector::from_column_slice(&stacked_true[..n_eta]);
    let prior_bsv = &omega_bsv.inv * &eta_bsv;
    for (k, g) in grad.iter_mut().take(n_eta).enumerate() {
        *g += prior_bsv[k];
    }
    for occ in 0..k_occasions {
        let base = n_eta + occ * n_kappa;
        let kappa_g = DVector::from_column_slice(&stacked_true[base..base + n_kappa]);
        let prior_kg = &omega_iov.inv * &kappa_g;
        for c in 0..n_kappa {
            grad[base + c] += prior_kg[c];
        }
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

/// Inner EBE minimization with an externally-provided gradient (analytic
/// sensitivities or AD). Fit-scoped dispatch (dense BFGS / L-BFGS / Nelder–Mead); the
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
    enable_stall: bool,
) -> bool {
    if matches!(
        inner_optimizer_mode(),
        crate::types::InnerOptimizer::NelderMead
    ) {
        return nelder_mead_minimize(obj, x, n, max_iter, tol);
    }
    if inner_use_lbfgs(n) {
        lbfgs_core(
            obj,
            grad,
            x,
            n,
            max_iter,
            tol,
            precond,
            stop_precond,
            enable_stall,
        )
    } else {
        dense_bfgs_core(
            obj,
            grad,
            x,
            n,
            max_iter,
            tol,
            precond,
            stop_precond,
            enable_stall,
        )
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
    enable_stall: bool,
) -> bool {
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    let mut g = grad(x);
    let mut f_cur = obj(x);
    // Objective-stall convergence (see [`objective_stalled`] / [`INNER_FTOL_REL`]): for ODE
    // objectives whose gradient norm is floored above `tol` by solver noise, a search that
    // reached the mode declares convergence rather than spinning to `max_iter`. Gated on
    // `enable_stall` (ODE only) and on the gradient having *plateaued* (`best_gnorm`).
    let mut stall = 0u32;
    let mut best_gnorm = f64::INFINITY;

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
        // Has the gradient norm meaningfully improved on the best seen? (Plateau guard.)
        let gnorm_improving = gnorm < best_gnorm * (1.0 - INNER_FTOL_GNORM_PLATEAU);
        if gnorm < best_gnorm {
            best_gnorm = gnorm;
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
        // No sufficient-decrease step found: report non-convergence so the caller takes the
        // argmin Nelder–Mead fallback rather than accepting a non-stationary η̂.
        if alpha == 0.0 {
            return false;
        }
        // Stall only for ODE objectives, and only once the gradient has plateaued (no
        // longer improving) — see [`INNER_FTOL_REL`]. `objective_stalled` is always called
        // so the flat-step counter stays accurate; the plateau/ODE gates only decide
        // whether a reached count converts to convergence.
        let obj_flat = objective_stalled(f_cur, f_new, &mut stall);
        let stalled = enable_stall && obj_flat && !gnorm_improving;
        f_cur = f_new;

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        if stalled {
            return true;
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
    enable_stall: bool,
) -> bool {
    let mut h_inv = init_h_inv(n, precond);
    let mut g = grad(x);
    // Track the objective at the current iterate so the line search never has to
    // recompute `obj(x)` (one prediction walk per inner step on the hot path).
    let mut f_cur = obj(x);
    let mut first_step = true;
    // Objective-stall convergence (see [`objective_stalled`] / [`INNER_FTOL_REL`]): gated on
    // `enable_stall` (ODE only) and on the gradient having *plateaued* (`best_gnorm`), so
    // smooth analytical/FD fits stay bit-identical and a non-stationary mid-descent iterate
    // can't be accepted.
    let mut stall = 0u32;
    let mut best_gnorm = f64::INFINITY;

    for _iter in 0..max_iter {
        // `stop_precond` is `Some` only for FREM (issue #406); general fits stop
        // on the raw L2 norm so the converged EBE is independent of the `precond`
        // H0 that accelerates the search.
        let gnorm = grad_norm_metric(&g, stop_precond);
        // Plateau guard: has `gnorm` meaningfully improved on the best seen?
        let gnorm_improving = gnorm < best_gnorm * (1.0 - INNER_FTOL_GNORM_PLATEAU);
        if gnorm < best_gnorm {
            best_gnorm = gnorm;
        }

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
            // non-convergence so the caller takes the argmin Nelder–Mead fallback.
            if alpha == 0.0 {
                return false;
            }
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            let obj_flat = objective_stalled(f_cur, f_new, &mut stall);
            let stalled = enable_stall && obj_flat && !gnorm_improving;
            f_cur = f_new;
            if stalled {
                return true;
            }
            g = grad(x);
            continue;
        }

        let (alpha, f_new) = backtracking_line_search(obj, x, &d, &g, n, f_cur);
        // No sufficient-decrease step found: report non-convergence so the caller takes the
        // argmin Nelder–Mead fallback rather than accepting a non-stationary η̂.
        if alpha == 0.0 {
            return false;
        }

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        let obj_flat = objective_stalled(f_cur, f_new, &mut stall);
        let stalled = enable_stall && obj_flat && !gnorm_improving;
        f_cur = f_new;
        if stalled {
            return true;
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

/// Maximum trial steps in the backtracking line search before it gives up.
/// With quadratic interpolation a sufficient-decrease step is normally found in
/// 1–3 trials; the cap only bites on directions with no representable decrease
/// (i.e. the iterate is already at the posterior mode to machine precision).
const MAX_LINE_SEARCH_TRIALS: usize = 30;

/// Function-value stopping criterion for the inner BFGS, complementing the gradient
/// norm test (`gnorm < tol`). When the objective is computed by the adaptive RK45 ODE
/// solver, its step-pattern non-smoothness puts a noise floor on the gradient
/// (empirically ~6e-7 at the mode for a 5-η `obs_scale = V1` model) that can sit *above*
/// the inner `tol`. A BFGS that has already reached the posterior mode then sits on the
/// answer with a dead-flat objective but never satisfies `gnorm < tol`, so it spins to
/// `max_iter` and reports failure.
///
/// Declaring convergence once the *objective* has stopped improving for
/// [`INNER_STALL_LIMIT`] consecutive accepted steps short-circuits that wasted spin. Two
/// guards keep it from accepting a non-stationary iterate:
///
/// 1. **ODE-only** (`enable_stall`, set by `find_ebe`/`find_ebe_iov` for `[odes]` models).
///    Analytical / event-driven objectives are exact, reach `gnorm < tol` normally, and
///    stay bit-identical to prior releases — the stall never touches them.
/// 2. **Gradient-plateau.** The stall fires only once the gradient has *stopped
///    decreasing* — the current `gnorm` no longer improves the best seen by more than
///    [`INNER_FTOL_GNORM_PLATEAU`]. Mid-descent (including a heavily-backtracked
///    tiny-`alpha` stretch) the gradient is still shrinking, so the plateau test fails and
///    the stall cannot fire at a non-stationary point. This adapts to whatever noise floor
///    the solver tolerance produces, rather than a fixed multiple of `tol`. The plateau is
///    measured on the same `stop_precond` metric the `gnorm < tol` test uses, so FREM's
///    preconditioned stop (#406) stays authoritative.
///
/// This is a fast-path optimisation only: correctness on every `false`-on-a-converged
/// search does **not** depend on it — the inner fallback keeps the lower-objective of the
/// BFGS partial and the Nelder–Mead restart and reports convergence from a real
/// stationarity check, so a stall that never fires still yields the correct EBE. See #555.
const INNER_FTOL_REL: f64 = 1e-11;
/// Consecutive negligible-improvement steps required before [`INNER_FTOL_REL`] declares
/// convergence (a small count guards against a one-off flat step mid-descent).
const INNER_STALL_LIMIT: u32 = 3;
/// Relative gradient-norm decrease that still counts as "the gradient is improving" for the
/// plateau guard on the objective-stall stop (see [`INNER_FTOL_REL`]). While `gnorm` keeps
/// dropping by more than this fraction the search is still descending, so the stall is held
/// off; once it plateaus (no such decrease) the search is at the noise floor.
const INNER_FTOL_GNORM_PLATEAU: f64 = 1e-3;

/// True once the objective has failed to improve by more than `INNER_FTOL_REL·(1+|f|)`
/// for [`INNER_STALL_LIMIT`] consecutive accepted steps. Shared verbatim by the dense and
/// L-BFGS inner drivers so the two paths cannot drift apart on convergence (#555).
fn objective_stalled(f_old: f64, f_new: f64, stall: &mut u32) -> bool {
    if (f_old - f_new) <= INNER_FTOL_REL * (1.0 + f_old.abs()) {
        *stall += 1;
    } else {
        *stall = 0;
    }
    *stall >= INNER_STALL_LIMIT
}

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
        // No `min_obs` filter: a hard reject forces trial rejection even for a single
        // short-record subject, which the `n_unconverged` filter would otherwise drop.
        n_start_rejected: results.iter().filter(|r| r.hard_reject).count(),
    };
    let eta_hats: Vec<DVector<f64>> = results.iter().map(|r| r.eta.clone()).collect();
    let h_matrices: Vec<DMatrix<f64>> = results.iter().map(|r| r.h_matrix.clone()).collect();
    let kappas: Vec<Vec<DVector<f64>>> = results.into_iter().map(|r| r.kappas).collect();

    (eta_hats, h_matrices, stats, kappas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
            let analytic = m3_censored_dterm_df(lloq, f, v, dv_df, 1);
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

    #[test]
    fn find_ebe_uses_fd_h_matrix_when_inner_gradient_forced_fd() {
        use crate::parser::model_parser::parse_model_string;

        let mut model = parse_model_string(
            r#"
[parameters]
  theta TVCL(0.15, 0.01, 10.0)
  theta TVV(5.0, 0.1, 100.0)
  theta TVIMAX(-0.3, -10.0, 10.0)
  theta TVTI50(100.0, 1.0, 700.0)
  theta TVHILL(3.0, 0.1, 10.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.01
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  IMAX = TVIMAX
  TI50 = TVTI50
  HILL = TVHILL

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL * exp(IMAX * TIME^HILL / (TI50^HILL + TIME^HILL)) / V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  gradient = fd
"#,
        )
        .expect("parse");
        model.gradient_method = GradientMethod::Fd;

        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 200.0, 1, 9600.0, false, 0.0)],
            obs_times: vec![20.0],
            obs_raw_times: Vec::new(),
            observations: vec![12.0],
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

        let result = find_ebe(
            &model,
            &subject,
            &model.default_params,
            50,
            1e-5,
            None,
            None,
        );

        assert!(
            result.h_matrix.iter().all(|v| v.is_finite()),
            "forced-FD inner route must not consume a non-finite analytic h_matrix: {:?}",
            result.h_matrix
        );
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

    /// Closed-form `iiv_on_ruv` + M3 BLOQ (#4c): the analytic non-IOV inner
    /// η-gradient must match central FD of `individual_nll`, exercising the censored
    /// `η_ruv` data column `h·z` and the `exp(2·η_ruv)` variance scaling on the
    /// censored rows (which previously forced FD).
    #[test]
    fn analytic_inner_gradient_iiv_on_ruv_m3_matches_fd() {
        use std::cell::RefCell;
        use std::collections::HashMap;
        let mut model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n",
        )
        .expect("parse closed-form iiv_on_ruv");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(3));

        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
            obs_raw_times: Vec::new(),
            // The last two rows are below the LLOQ = 2.0 (carried in `observations`).
            observations: vec![8.0, 7.0, 5.0, 3.0, 2.0, 2.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0, 0, 1, 1],
            occasions: vec![1; 6],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };

        let theta = &model.default_params.theta;
        let omega = &model.default_params.omega;
        let sigma = &model.default_params.sigma.values;
        let eta = vec![0.12, -0.05, 0.2, 0.15]; // non-zero η_ruv

        let analytic = analytic_eta_nll_gradient(&model, &subject, theta, &eta, omega, sigma)
            .expect("analytic closed-form M3 + iiv_on_ruv inner gradient");

        let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(&subject));
        let obj = |e: &[f64]| -> f64 {
            let mut s = scratch.borrow_mut();
            individual_nll_into_with_schedule(
                &model, &subject, theta, e, omega, sigma, &mut s, None,
            )
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

    /// ODE counterpart of [`analytic_inner_gradient_m3_matches_fd_on_warfarin_bloq`]:
    /// the analytic M3 inner η-gradient produced via the **event-driven ODE
    /// sensitivity walk** (not the closed-form provider) must match a central FD of
    /// the inner objective on the warfarin BLOQ data — confirming non-IOV ODE+M3 is
    /// served analytically on the inner loop (the censored `−logΦ` coefficient rides
    /// the same provider-agnostic `apply_*_inner` path as the closed-form engine).
    #[test]
    fn analytic_inner_gradient_m3_matches_fd_on_warfarin_ode_bloq() {
        use std::cell::RefCell;
        use std::path::Path;
        let model = crate::parser::model_parser::parse_model_file(Path::new(
            "examples/warfarin_ode_bloq.ferx",
        ))
        .expect("warfarin ODE BLOQ model parses");
        assert!(
            matches!(model.bloq_method, crate::types::BloqMethod::M3),
            "model must be M3"
        );
        assert!(
            model.is_ode_based(),
            "model must be on the ODE path for this probe"
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
            .expect("analytic M3 inner gradient must be supported on ODE path");

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

    /// **Non-IOV ODE** M3 BLOQ + `iiv_on_ruv` (#486 — the last `iiv_on_ruv` holdout):
    /// the ODE counterpart of [`analytic_inner_gradient_iiv_on_ruv_m3_matches_fd`]. The
    /// censored residual-eta data column `h·z` and the `exp(2·η_ruv)` variance scaling are
    /// applied by the provider-agnostic `residual_inner_obs` over the **event-driven ODE
    /// walk's** `ObsSens` (not the closed-form provider), so the analytic inner η-gradient
    /// must match central FD of `individual_nll` — the non-IOV ODE M3 + `iiv_on_ruv` combo
    /// the inner loop now admits (#623).
    #[test]
    fn analytic_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode() {
        use std::cell::RefCell;
        use std::collections::HashMap;
        let mut model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE iiv_on_ruv");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(model.is_ode_based(), "model must be on the ODE path");

        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0],
            obs_raw_times: Vec::new(),
            // The last two rows are below the LLOQ (carried in `cens`).
            observations: vec![8.0, 7.0, 5.0, 3.0, 2.0, 2.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0, 0, 1, 1],
            occasions: vec![1; 6],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };

        let theta = &model.default_params.theta;
        let omega = &model.default_params.omega;
        let sigma = &model.default_params.sigma.values;
        let eta = vec![0.12, -0.05, 0.2, 0.15]; // non-zero η_ruv

        let analytic = analytic_eta_nll_gradient(&model, &subject, theta, &eta, omega, sigma)
            .expect("analytic non-IOV ODE M3 + iiv_on_ruv inner gradient");

        let scratch = RefCell::new(pk::EventPkParams::with_capacity_for(&subject));
        let obj = |e: &[f64]| -> f64 {
            let mut s = scratch.borrow_mut();
            individual_nll_into_with_schedule(
                &model, &subject, theta, e, omega, sigma, &mut s, None,
            )
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
            let t_dense =
                time_it(&|x| dense_bfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None, false));
            let t_lbfgs =
                time_it(&|x| lbfgs_core(&obj, &grad, x, n, 2000, 1e-8, None, None, false));
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
        let ok = dense_bfgs_core(&obj, &grad, &mut x, 2, 200, 1e-10, None, None, false);
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
            hard_reject: false,
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
                hard_reject: false,
            },
            EbeResult {
                eta: nalgebra::DVector::zeros(1),
                h_matrix: nalgebra::DMatrix::identity(1, 1),
                converged: false, // also unconverged
                used_fallback: true,
                grad_norm: 0.0,
                nll: 2.0,
                kappas: Vec::new(),
                hard_reject: false,
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

    /// #603 review #1/#2: a hard-rejected subject must be counted even with a short record,
    /// so a single one forces the outer guard to reject the trial. Mirrors the `n_start_rejected`
    /// derivation in `run_inner_loop_warm` (no `min_obs` filter, unlike `n_unconverged`).
    #[test]
    fn test_inner_loop_stats_counts_hard_reject_regardless_of_obs() {
        let make = |hard_reject: bool| EbeResult {
            eta: nalgebra::DVector::zeros(1),
            h_matrix: nalgebra::DMatrix::zeros(1, 1),
            converged: false,
            used_fallback: false,
            grad_norm: 0.0,
            nll: 1.0,
            kappas: Vec::new(),
            hard_reject,
        };
        // One hard-rejected subject with a single observation, one normal subject.
        let results = [make(true), make(false)];
        let obs_counts = [1_usize, 5_usize];
        let min_obs = 3_usize;

        // The `min_obs` filter would drop the 1-obs subject from `n_unconverged` …
        let n_unconverged = results
            .iter()
            .zip(obs_counts.iter())
            .filter(|(r, &n_obs)| !r.converged && n_obs >= min_obs.max(1))
            .count();
        assert_eq!(n_unconverged, 1); // only the 5-obs subject

        // … but `n_start_rejected` counts the hard reject regardless of obs count.
        let n_start_rejected = results.iter().filter(|r| r.hard_reject).count();
        assert_eq!(n_start_rejected, 1);
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
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp(); // CL
                    p.values[1] = theta[1] * eta[1].exp(); // V
                    p
                },
            ),
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
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            transit_ode_equivalent: None,
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

    #[test]
    fn gradient_route_summary_reports_ode_iov_analytic_route() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
        let population = Population {
            subjects: vec![Subject {
                id: "1".into(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: vec![1.0, 6.0, 25.0, 30.0],
                obs_raw_times: Vec::new(),
                observations: vec![8.0, 6.0, 7.0, 5.0],
                obs_cmts: vec![1; 4],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 4],
                occasions: vec![1, 1, 2, 2],
                dose_occasions: vec![1, 2],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            }],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let summary = gradient_route_summary(&model, &population, GradientMethod::Auto);
        assert!(
            summary.starts_with("analytic (Dual2)"),
            "ODE IOV provider should be reported as analytic, got: {summary}"
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

    /// Tier-1 follow-up to #665: a TV-cov + LTBS subject now takes the **analytic** inner
    /// gradient (the event-driven inner walk applies the `ln f` jet LAST), so the reported
    /// route is Analytic, `subject_eta_grad` returns `Some`, and no FD-fallback warning fires.
    #[test]
    fn tvcov_ltbs_reports_analytic_inner() {
        use crate::parser::model_parser::parse_model_string;
        let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma ADD_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;
        let model = parse_model_string(src).expect("parse TV-cov LTBS");
        assert!(model.log_transform);
        let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
        let tvcov_subj = || Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0, 3.0],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            // Time-varying WT → routes to the event-driven walk, which now serves LTBS.
            obs_covariates: vec![wt(70.0), wt(72.0), wt(80.0), wt(85.0), wt(90.0)],
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let subject = tvcov_subj();
        assert!(subject.has_tv_covariates(), "fixture must be TV-cov");
        // TV-cov + LTBS now takes the analytic inner gradient.
        assert_eq!(
            resolve_gradient_method(&model, &subject),
            InnerGradientMethod::Analytic,
            "TV-cov + LTBS inner gradient is analytic now"
        );
        assert!(
            crate::sens::provider::subject_eta_grad(
                &model,
                &subject,
                &model.default_params.theta,
                &[0.0; 3]
            )
            .is_some(),
            "TV-cov + LTBS inner provider must serve the subject"
        );
        // Whole population analytic → no FD-fallback warning.
        let population = Population {
            subjects: vec![tvcov_subj(), tvcov_subj()],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        assert!(
            fd_fallback_warning(&model, &population, &model.default_params.theta).is_none(),
            "analytic TV-cov + LTBS population must not warn"
        );
    }

    #[test]
    fn iov_fd_fallback_warning_reports_subject_reason() {
        // Covariate-free ODE IOV model the provider serves analytically, so the FD
        // subject below is the *only* one out of scope — a genuinely mixed
        // population (the all-FD case is suppressed, mirroring the non-IOV
        // contract, #590 review).
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
        // Analytic subject: doses and observations in the same occasions.
        let analytic_subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // FD subject: more occasion groups than the widened ODE IOV dispatch serves.
        let n_wide = crate::sens::ode_provider::MAX_ODE_IOV_AXES;
        let fd_subject = Subject {
            id: "2".into(),
            doses: (0..n_wide)
                .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
                .collect(),
            obs_times: (0..n_wide).map(|i| i as f64 * 24.0 + 1.0).collect(),
            obs_raw_times: Vec::new(),
            observations: vec![8.0; n_wide],
            obs_cmts: vec![1; n_wide],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_wide],
            occasions: (1..=n_wide as u32).collect(),
            dose_occasions: (1..=n_wide as u32).collect(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![analytic_subject, fd_subject],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let warning = fd_fallback_warning(&model, &population, &model.default_params.theta)
            .expect("mixed IOV population should warn with a reason");
        assert!(warning.contains("1 of 2"), "got: {warning}");
        assert!(
            warning.contains("ODE IOV stacked axis cap"),
            "got: {warning}"
        );
    }

    /// Uniform all-FD IOV populations are silent, matching the non-IOV contract:
    /// the `finite-difference` banner already makes a model-level fallback obvious.
    #[test]
    fn iov_fd_fallback_warning_silent_for_uniform_all_fd() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  gradient = fd\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + gradient = fd");
        let mk_subject = |id: &str| Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let population = Population {
            subjects: vec![mk_subject("1"), mk_subject("2")],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        // gradient = fd routes every subject to FD (uniform) → no warning.
        assert!(
            fd_fallback_warning(&model, &population, &model.default_params.theta).is_none(),
            "uniform all-FD population must not warn"
        );
    }

    /// Regression (#486): steady-state combined with an estimated lagtime is now analytic
    /// under IOV (the `K_SS_SEED` pre-arrival seed, shared with the non-IOV walk). Before
    /// #486 this subject declined via the `SS + lagtime` gate (#590 review); pins that the
    /// inner IOV route now admits it instead.
    #[test]
    fn iov_inner_subject_route_admits_steady_state_lagtime() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVLAG(0.5,0.01,5.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_LAG ~ 0.09\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  LAGTIME = TVLAG * exp(ETA_LAG)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  obs_scale = V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + lagtime");
        assert!(
            model.has_lagtime(),
            "model should carry a LAGTIME individual parameter"
        );
        let subject = Subject {
            id: "1".into(),
            // Steady-state bolus (ss, ii > 0) under an estimated lagtime.
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_some(),
            "SS + lagtime subject must be analytic now (#486)"
        );
    }

    /// Regression (#486): modeled-`RATE`/duration doses combined with steady-state are now
    /// analytic under IOV too (`equilibrate_ss_state_g` threads the same per-occasion
    /// `inf_eff` jet into its per-cycle split). Before #486 this subject declined via the
    /// modeled+SS screen; pins that the inner IOV route now admits it instead.
    #[test]
    fn iov_inner_subject_route_admits_modeled_dose_steady_state() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVD1(5.0,0.1,24.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_D1 ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  D1 = TVD1 * exp(ETA_D1)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + modeled D1");
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::modeled(
                0.0,
                100.0,
                1,
                true,
                24.0,
                crate::types::RateMode::ModeledDuration,
            )],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_some(),
            "modeled + SS subject must be analytic now (#486)"
        );
    }

    /// Regression (#486): a modeled dose whose `D{cmt}`/`R{cmt}` slot is undeclared (normally
    /// rejected by `check_model_data`, but defended here) routes to FD and is attributed to the
    /// missing slot — not silently mis-resolved. The base model declares no `D1`, so the
    /// `ModeledDuration` dose finds no duration slot.
    #[test]
    fn iov_fd_reason_attributes_modeled_dose_missing_slot() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV without D1");
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::modeled(
                0.0,
                100.0,
                1,
                false,
                0.0,
                crate::types::RateMode::ModeledDuration,
            )],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_none(),
            "modeled dose with missing slot must route to FD"
        );
        assert_eq!(
            iov_fd_reason(&model, &subject),
            "modeled RATE/DURATION dose with missing D/R slot"
        );
    }

    #[test]
    fn iov_fd_reason_attributes_ss_input_rate() {
        // #486: a steady-state dose + a built-in absorption forcing (`zero_order`) declines
        // to FD under IOV; `iov_fd_reason` must name that combination (not the generic
        // "outside IOV analytic scope"), mirroring `ode_iov_subject_supported`'s bail.
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVDUR(5.0,0.1,24.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_DUR ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  DUR = TVDUR * exp(ETA_DUR)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse zero_order ODE IOV");
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            dose_occasions: vec![1],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_none(),
            "SS + zero_order must route to FD under IOV"
        );
        assert_eq!(
            iov_fd_reason(&model, &subject),
            "steady-state dose + built-in absorption forcing"
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
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    // eta[0] = bsv, eta[1] = kappa (combined)
                    p.values[0] = theta[0] * eta[0].exp();
                    p.values[1] = theta[1];
                    p
                },
            ),
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
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            transit_ode_equivalent: None,
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

    /// The analytic ODE IOV inner gradient (`analytic_eta_nll_gradient_iov`) must match
    /// central finite differences of the inner objective `individual_nll_iov` over the
    /// stacked `[η_bsv, κ₁..κ_K]` vector — the gradient that now drives `find_ebe_iov`
    /// for ODE IOV models (#439 ODE IOV inner).
    #[test]
    fn analytic_iov_inner_grad_matches_fd_of_nll() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV");
        assert!(crate::sens::ode_provider::ode_iov_supported(&model));
        let subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 4.0, 7.0, 5.0, 3.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        let stacked = vec![0.10, -0.05, 0.08, -0.12];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV inner gradient");

        // Central FD of the inner objective (same NLL `find_ebe_iov` minimises).
        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-6 * (1.0 + stacked[p].abs());
            let mut sp = stacked.clone();
            sp[p] += h;
            let mut sm = stacked.clone();
            sm[p] -= h;
            let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// Closed-form twin of [`analytic_iov_inner_grad_matches_fd_of_nll`] with an η-dependent
    /// `ExpressionScale` `obs_scale = V` divisor (#486): the analytic IOV inner gradient
    /// (`analytic_eta_nll_gradient_iov`, now fed the scaled `subject_eta_grad_iov`) must match
    /// central FD of the same objective `individual_nll_iov` (which applies `obs_scale`) over
    /// the stacked `[η_bsv, κ]` vector — the gradient that drives `find_ebe_iov` for a scaled
    /// closed-form IOV model.
    #[test]
    fn analytic_iov_inner_grad_matches_fd_of_nll_closed_form_expr_scale() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + obs_scale");
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        let subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.8, 0.6, 0.4, 0.7, 0.5, 0.3],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        let stacked = vec![0.10, -0.05, 0.08, -0.12];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-6 * (1.0 + stacked[p].abs());
            let mut sp = stacked.clone();
            sp[p] += h;
            let mut sm = stacked.clone();
            sm[p] -= h;
            let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    // ----- #555 shared fixtures (two_cpt_oral_cov, η+covariate `obs_scale = V1`) -----

    /// Analytical + ODE twin of the ferx-r `two_cpt_oral_cov` model (5 η, `obs_scale = V1`
    /// with `V1 = TVV1·(WT/70)^θ·exp(ETA_V1)` — both covariate- and η-dependent). The ODE
    /// solver block is caller-supplied so a cheap fixed-η check can run tight and the inner
    /// EBE check can run loose.
    fn repro555_model_pair(ode_solver_block: &str) -> (CompiledModel, CompiledModel) {
        use crate::parser::model_parser::parse_model_string;
        let header = "[parameters]\n  theta TVCL(5.0,0.1,100.0)\n  theta TVV1(50.0,1.0,500.0)\n  theta TVQ(10.0,0.1,100.0)\n  theta TVV2(100.0,1.0,500.0)\n  theta TVKA(1.2,0.01,10.0)\n  theta THETA_WT(0.75,0.01,5.0)\n  theta THETA_CRCL(0.50,0.01,5.0)\n  omega ETA_CL ~ 0.10\n  omega ETA_V1 ~ 0.10\n  omega ETA_Q ~ 0.05\n  omega ETA_V2 ~ 0.05\n  omega ETA_KA ~ 0.15\n  sigma PROP_ERR ~ 0.02 (sd)\n[individual_parameters]\n  CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)\n  V1 = TVV1 * (WT/70)^THETA_WT * exp(ETA_V1)\n  Q = TVQ * exp(ETA_Q)\n  V2 = TVV2 * exp(ETA_V2)\n  KA = TVKA * exp(ETA_KA)\n";
        let an = parse_model_string(&format!(
            "{header}[structural_model]\n  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n[covariates]\n  WT continuous\n  CRCL continuous\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n"
        )).expect("parse analytical");
        let ode = parse_model_string(&format!(
            "{header}[structural_model]\n  ode(obs_cmt=central, states=[depot, central, periph])\n[odes]\n  d/dt(depot) = -KA * depot\n  d/dt(central) = KA*depot - (CL/V1 + Q/V1)*central + (Q/V2)*periph\n  d/dt(periph) = (Q/V1)*central - (Q/V2)*periph\n[scaling]\n  obs_scale = V1\n[covariates]\n  WT continuous\n  CRCL continuous\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n{ode_solver_block}"
        )).expect("parse ode");
        (an, ode)
    }

    /// Subject 22 of the ferx-r `two_cpt_oral_cov` dataset — the subject that diverged in
    /// #555 — inlined verbatim so the regression is self-contained (no external data file).
    fn repro555_subject22() -> Subject {
        let mut covariates = HashMap::new();
        covariates.insert("WT".to_string(), 72.1);
        covariates.insert("CRCL".to_string(), 76.7);
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0];
        let observations = vec![
            2.0190, 2.4021, 2.2985, 1.8141, 1.4699, 1.2589, 1.0804, 0.8993, 0.7054, 0.5966,
        ];
        let n = obs_times.len();
        Subject {
            id: "22".into(),
            doses: vec![DoseEvent::new(0.0, 250.0, 1, 0.0, false, 0.0)],
            obs_times,
            obs_raw_times: Vec::new(),
            observations,
            obs_cmts: vec![2; n],
            covariates,
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// #555 guard: at a *fixed* η the ODE `obs_scale = V1` form and its analytical twin
    /// must agree on the assembled FOCEI marginal (incl. log|H̃|) to integrator tolerance —
    /// the forward IPRED, the ∂f/∂η Jacobian, and the assembly are all path-independent
    /// here. (The #555 divergence lives in EBE *convergence*, not this fixed-η objective —
    /// see `repro555_ode_exprscale_ebe_finds_global_min`.) Tight ODE tol is cheap: no inner
    /// optimisation runs, only a handful of `predict`/Jacobian evaluations.
    #[test]
    fn repro555_ode_exprscale_marginal_vs_analytical() {
        use crate::stats::likelihood::foce_subject_nll_interaction;
        let (an, ode) = repro555_model_pair("  ode_reltol = 1e-11\n  ode_abstol = 1e-13\n");
        let subject = repro555_subject22();

        let marginal = |m: &CompiledModel, eta: &[f64]| -> f64 {
            let p = &m.default_params;
            let eta_v = nalgebra::DVector::from_column_slice(eta);
            let ipreds = crate::pk::compute_predictions_with_tv(m, &subject, &p.theta, eta);
            let jac = crate::sens::provider::subject_eta_jacobian(m, &subject, &p.theta, eta)
                .expect("analytic jac");
            let h = nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), m.n_eta, &jac);
            foce_subject_nll_interaction(
                &subject,
                &ipreds,
                &eta_v,
                &h,
                &p.omega,
                &p.sigma.values,
                &m.error_spec,
                m.bloq_method,
                &[],
                None,
                m.residual_error_eta,
                None, // no custom residual magnitude (#484) in this model
            )
        };

        for eta in [vec![0.0; 5], vec![0.12, -0.08, 0.05, 0.04, -0.10]] {
            let ma = marginal(&an, &eta);
            let mo = marginal(&ode, &eta);
            approx::assert_relative_eq!(ma, mo, max_relative = 1e-5, epsilon = 1e-4);
        }
    }

    /// #555 regression: on an ODE model with an η-dependent `[scaling] obs_scale = V1`,
    /// the inner EBE must reach the correct posterior mode.
    ///
    /// Root cause: the inner BFGS *reaches* the mode but its gradient norm floors above
    /// `tol` (the adaptive ODE solver's non-smoothness caps `gnorm` above `tol`), so it
    /// spun to `max_iter` and reported failure; `find_ebe` then discarded that correct η̂
    /// and overwrote it with a cold Nelder–Mead restart from η=0, which on this multimodal
    /// inner objective settled ~20 NLL units worse, inflating the FOCEI OFV by ~370 on the
    /// full dataset (the analytical twin, whose smooth objective lets BFGS satisfy
    /// `gnorm < tol`, was the correct reference). The fix is twofold: `find_ebe` now keeps
    /// the lower-objective of the BFGS partial and the NM restart (so a `false`-on-a-
    /// converged search can never regress the EBE), and the inner BFGS gained a gated
    /// objective-stall (`ftol`) stop so it converges at the mode instead of spinning. This
    /// test runs at a *moderate* ODE tolerance (not the 1e-10 of the original repro) to
    /// stay Tier-1-fast and to exercise the realistic-tolerance path where the stall may
    /// not fire and the argmin fallback is what guarantees correctness.
    #[test]
    fn repro555_ode_exprscale_ebe_finds_global_min() {
        use crate::stats::likelihood::individual_nll;
        let subject = repro555_subject22();

        // Run at the DEFAULT ODE tolerances (`ode_reltol = 1e-4`, no override) — the
        // realistic case a user actually hits, and the one where the gradient-noise floor
        // is high enough that the BFGS objective-stall may never fire, so the argmin
        // fallback is what guarantees the correct EBE. The analytical twin (whose smooth
        // objective lets BFGS satisfy `gnorm < tol`) gives the reference global minimum.
        let (an, ode) = repro555_model_pair("");

        let nll = |m: &CompiledModel, e: &[f64]| {
            let p = &m.default_params;
            individual_nll(m, &subject, &p.theta, e, &p.omega, &p.sigma.values)
        };

        let e_an = find_ebe(&an, &subject, &an.default_params, 300, 1e-7, None, None);
        let eta_an: Vec<f64> = e_an.eta.iter().copied().collect();
        let ref_nll = nll(&an, &eta_an);

        // The ODE form must reach the same global minimum (objectives are identical), and
        // must report it as a converged EBE (not a fallback that left it non-stationary).
        let e_od = find_ebe(&ode, &subject, &ode.default_params, 300, 1e-7, None, None);
        let eta_od: Vec<f64> = e_od.eta.iter().copied().collect();
        let ode_nll = nll(&ode, &eta_od);

        // Pre-fix the ODE EBE stalled ~20 NLL units high in a spurious basin; the global
        // min is now reached (to integrator tolerance) at the default ODE tolerance too.
        assert!(
            ode_nll <= ref_nll + 0.5,
            "ODE EBE stuck in a spurious basin: inner NLL {ode_nll:.4} vs analytical global min {ref_nll:.4} \
             (eta_ode={eta_od:?}, eta_an={eta_an:?})"
        );
        assert!(
            e_od.converged,
            "ODE EBE should be reported converged at the mode"
        );
    }

    /// #587 review: the shared inner-EBE fallback keeps the lower-objective **value** of the
    /// BFGS partial and the Nelder–Mead restart (the substantive #555 fix — the η̂ value fed
    /// to the FOCEI gradient), seeds NM from the partial under `ebe_warm_start`, and discards
    /// a non-finite-objective partial. Uses a bimodal 1-D objective (deep well at x=-2,
    /// f=-10; shallow well at x=+2, f=-1).
    #[test]
    fn argmin_inner_fallback_keeps_better_basin() {
        let obj = |x: &[f64]| -> f64 {
            let v = x[0];
            if v < 0.0 {
                (v + 2.0).powi(2) - 10.0
            } else {
                (v - 2.0).powi(2) - 1.0
            }
        };
        set_ebe_warm_start(false);

        // Partial in the deep (global) well, cold NM seed in the shallow well: the fallback
        // keeps the lower-objective partial rather than overwriting with the shallow NM
        // result (the old behaviour, which on this multimodal objective inflated the OFV).
        let (eta, _) = argmin_inner_fallback(&obj, &[-2.0], &[2.0], 1, 200, 1e-8);
        assert!((eta[0] + 2.0).abs() < 1e-2, "kept deep well, got {eta:?}");

        // Partial in the shallow well, cold NM seed reaches the deep well: NM wins.
        let (eta2, _) = argmin_inner_fallback(&obj, &[2.0], &[-2.0], 1, 200, 1e-8);
        assert!(
            (eta2[0] + 2.0).abs() < 1e-2,
            "NM found deeper well, got {eta2:?}"
        );

        // Non-finite partial objective → unusable → NM result is taken.
        let (eta3, _) = argmin_inner_fallback(&obj, &[f64::NAN], &[-2.0], 1, 200, 1e-8);
        assert!(
            eta3[0].is_finite(),
            "NaN partial must be discarded, got {eta3:?}"
        );

        // `ebe_warm_start` seeds the single NM from the partial (covers the warm branch):
        // from the deep well it stays there even though the cold seed is far away.
        set_ebe_warm_start(true);
        let (eta4, _) = argmin_inner_fallback(&obj, &[-2.0], &[5.0], 1, 200, 1e-8);
        assert!(
            (eta4[0] + 2.0).abs() < 1e-2,
            "warm seed held the deep well, got {eta4:?}"
        );
        set_ebe_warm_start(false);
    }

    #[test]
    fn ode_iov_skips_nelder_mead_inner_fallback() {
        use crate::parser::model_parser::parse_model_string;
        let ode_iov = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
        assert!(skip_ode_iov_nm_fallback(&ode_iov));

        let closed_form_iov = make_iov_model();
        assert!(!skip_ode_iov_nm_fallback(&closed_form_iov));
    }

    #[test]
    fn ode_iov_start_rejects_only_pathological_ode_iov_nll() {
        use crate::parser::model_parser::parse_model_string;
        let ode_iov = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
        let closed_form_iov = make_iov_model();

        assert!(!reject_ode_iov_inner_start(&ode_iov, 4, 999.0));
        assert!(reject_ode_iov_inner_start(&ode_iov, 4, 1_001.0));
        assert!(!reject_ode_iov_inner_start(&ode_iov, 20, 4_999.0));
        assert!(reject_ode_iov_inner_start(&ode_iov, 20, 5_001.0));
        assert!(reject_ode_iov_inner_start(&ode_iov, 20, f64::NAN));
        assert!(!reject_ode_iov_inner_start(
            &closed_form_iov,
            4,
            1_000_000.0
        ));
    }

    /// Closed-form IOV + `iiv_on_ruv` (#4b): the analytic stacked-η inner gradient
    /// (`analytic_eta_nll_gradient_iov`) must match central FD of the inner objective
    /// `individual_nll_iov` over `[η_bsv, η_ruv, κ₁..κ_K]` — including the `η_ruv` column
    /// (`Σ_j 1 − ε²/v`) and the `exp(2·η_ruv)` residual-variance scaling now woven into
    /// the IOV inner gradient. Proves the gate flip (`iov_analytical_supported`) ships a
    /// *correct* gradient, not just an enabled one.
    #[test]
    fn iov_iiv_on_ruv_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + iiv_on_ruv");
        // Gate flip: the closed-form IOV + iiv_on_ruv path is now analytic on both loops.
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        assert!(crate::sens::provider::iov_sens_supported(&model));
        assert!(!analytic_inner_common_bail(&model));

        let subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 4.0, 7.0, 5.0, 3.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        // Non-zero η_ruv (index 3) so the residual-variance scaling is genuinely exercised.
        let stacked = vec![0.10, -0.05, 0.08, 0.15, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV + iiv_on_ruv inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-6 * (1.0 + stacked[p].abs());
            let mut sp = stacked.clone();
            sp[p] += h;
            let mut sm = stacked.clone();
            sm[p] -= h;
            let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
        }
    }

    /// M3 BLOQ + IOV (#580): the analytic stacked-η inner gradient
    /// (`analytic_eta_nll_gradient_iov`) must match central FD of the inner objective
    /// `individual_nll_iov` over `[η_bsv, κ₁..κ_K]` when the subject carries M3-censored
    /// rows (data term `−logΦ(z)`, matching `individual_nll_iov`'s `−2·m3_logcdf`). The
    /// censored `h·m` f-coefficient rides the stacked Jacobian (κ columns included), so
    /// the EBE minimises the same censored objective. Proves the gate flip
    /// (`iov_analytical_supported` now admits M3) ships a *correct* censored gradient over
    /// the stacked layout, not just an enabled one.
    #[test]
    fn iov_m3_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3");
        model.bloq_method = crate::types::BloqMethod::M3;
        // Gate: M3 + IOV is now analytic on both loops (no `iiv_on_ruv`, so not the
        // FD-only triple). `residual_error_eta` is `None`, so `iov_analytical_supported`
        // does not early-return on M3.
        assert_eq!(model.residual_error_eta, None);
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        assert!(crate::sens::provider::iov_sens_supported(&model));
        assert!(!analytic_inner_common_bail(&model));

        let mut subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            // The last two rows (occasion 2 tail) are M3 left-censored; the `−logΦ`
            // term differentiates them.
            cens: vec![0, 0, 0, 0, 1, 1],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        // Synthesize observations from the model at a reference (η, κ), scaled to 0.85·f
        // — including the censored rows, so the carried LLOQ sits just below the
        // prediction (z ≈ −0.75, the moderate regime where the A&S `log_normal_cdf` and
        // the exact φ in `inv_mills` agree to FD precision; a deep-tail LLOQ would expose
        // only the CDF-approximation floor, not the gradient's correctness).
        let preds = crate::pk::predict_iov(
            &model,
            &subject,
            &params.theta,
            &[0.12, -0.08, 0.2],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV + M3 inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        // Richardson-extrapolated central FD: the censored `−logΦ` term has sharp
        // curvature on the occasion-2 κ axis, so plain central FD is truncation-limited
        // (~2e-4) there — Richardson removes it and validates the analytic to ~1e-7.
        for p in 0..n_stacked {
            let h = 1e-5 * (1.0 + stacked[p].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut sp = stacked.clone();
                sp[p] += hh;
                let mut sm = stacked.clone();
                sm[p] -= hh;
                (nll(&sp) - nll(&sm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// Right-censored (above-ULOQ, `CENS = -1`) regression for the analytic IOV+M3
    /// inner gradient. The objective `individual_nll_iov` scores these rows with the
    /// **upper** tail (`m3_logcdf`, `z = (f − ULOQ)/√v`); the analytic gradient must use
    /// the same tail. Before the signed `m3_censored_kernel` (review of #591) the kernel
    /// always took the lower tail, so this gradient was wrong-signed for `CENS = -1` and
    /// `find_ebe_iov` pushed the EBE the wrong way. Same model/fixture as
    /// `iov_m3_inner_grad_matches_fd` with the occasion-2 tail flipped to `CENS = -1`.
    #[test]
    fn iov_m3_right_censored_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3");
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(crate::sens::provider::iov_analytical_supported(&model));

        let mut subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            // Occasion-2 tail is M3 *right*-censored (above ULOQ): upper tail.
            cens: vec![0, 0, 0, 0, -1, -1],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        // Carry ULOQ at 0.85·f for the censored rows, so z = (f − ULOQ)/√v ≈ +0.15·f/√v
        // sits in the moderate upper-tail regime (Φ(z) well away from 0/1), where the
        // A&S `log_normal_cdf` and the exact φ in `inv_mills` agree to FD precision.
        let preds = crate::pk::predict_iov(
            &model,
            &subject,
            &params.theta,
            &[0.12, -0.08, 0.2],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV + M3 inner gradient (right-censored)");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-5 * (1.0 + stacked[p].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut sp = stacked.clone();
                sp[p] += hh;
                let mut sm = stacked.clone();
                sm[p] -= hh;
                (nll(&sp) - nll(&sm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// **ODE** M3 BLOQ + IOV (#486): the ODE counterpart of
    /// [`iov_m3_inner_grad_matches_fd`]. The analytic stacked-η inner gradient produced
    /// via the **event-driven ODE sensitivity walk** (`ode_subject_eta_grad_iov`, not the
    /// closed-form Dual1 walk) must match Richardson central FD of `individual_nll_iov`
    /// over `[η_bsv, κ₁, κ₂]` on a censored subject. Censoring is provider-agnostic —
    /// the `−logΦ` coefficient rides the same `residual_inner_obs` path keyed on
    /// `subject.cens[j]` whether the walk was closed-form or ODE — so removing the gate
    /// clause is all that was needed. Both tails (`CENS = 1` left, `CENS = -1` right).
    #[test]
    fn analytic_iov_inner_gradient_m3_matches_fd_on_ode_bloq() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  bloq_method = m3\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV + M3");
        assert!(
            matches!(model.bloq_method, crate::types::BloqMethod::M3),
            "model must be M3"
        );
        assert!(model.is_ode_based(), "must be on the ODE path");
        // After the #486 gate flip, the ODE IOV walk serves M3 analytically on the inner
        // loop (single gate — no separate M3 bail).
        assert!(crate::sens::provider::iov_sens_supported(&model));
        assert!(!analytic_inner_common_bail(&model));

        // Both tails: occasion-2 tail left-censored (CENS=1), then right-censored (CENS=-1).
        for cens_sign in [1i8, -1] {
            let mut subject = Subject {
                id: "1".into(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
                obs_raw_times: Vec::new(),
                observations: vec![0.0; 6],
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0, 0, cens_sign, cens_sign],
                occasions: vec![1, 1, 1, 2, 2, 2],
                dose_occasions: vec![1, 2],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            };
            let params = model.default_params.clone();
            // Carry the censoring limit at 0.85·f so z = (f − LIMIT)/√v sits in the
            // moderate regime where the A&S `log_normal_cdf` and the exact φ in `inv_mills`
            // agree to FD precision (a deep-tail limit would expose only the CDF floor).
            let preds = crate::pk::predict_iov(
                &model,
                &subject,
                &params.theta,
                &[0.12, -0.08, 0.2],
                &[vec![0.05], vec![-0.07]],
            );
            subject.observations = preds.iter().map(|p| p * 0.85).collect();
            let n_eta = model.n_eta;
            let n_kappa = model.n_kappa;
            let k = iov_occasion_groups(&subject).len();
            let n_stacked = n_eta + k * n_kappa;
            let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
            let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
            assert_eq!(stacked.len(), n_stacked);

            let g = analytic_eta_nll_gradient_iov(
                &model,
                &subject,
                &params.theta,
                &stacked,
                &params.omega,
                omega_iov,
                &params.sigma.values,
                n_eta,
                n_kappa,
                k,
                None,
            )
            .expect("analytic ODE IOV + M3 inner gradient");

            let nll = |s: &[f64]| -> f64 {
                let eta_t = &s[..n_eta];
                let kappas: Vec<Vec<f64>> = (0..k)
                    .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                    .collect();
                individual_nll_iov(
                    &model,
                    &subject,
                    &params.theta,
                    eta_t,
                    &kappas,
                    &params.omega,
                    Some(omega_iov),
                    &params.sigma.values,
                )
            };
            for p in 0..n_stacked {
                let h = 1e-5 * (1.0 + stacked[p].abs());
                let fd_at = |hh: f64| -> f64 {
                    let mut sp = stacked.clone();
                    sp[p] += hh;
                    let mut sm = stacked.clone();
                    sm[p] -= hh;
                    (nll(&sp) - nll(&sm)) / (2.0 * hh)
                };
                let f1 = fd_at(h);
                let f2 = fd_at(h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0;
                approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-5);
            }
        }
    }

    /// **ODE** triple M3 + IOV + `iiv_on_ruv` (#486): the ODE counterpart of
    /// [`iov_m3_iiv_on_ruv_inner_grad_matches_fd`]. The analytic stacked-η inner gradient
    /// from the **event-driven ODE walk** must match Richardson FD of `individual_nll_iov`
    /// over `[η_bsv, η_ruv, κ₁, κ₂]` when censored rows co-occur with the `exp(2·η_ruv)`
    /// residual-variance scaling. The ODE walk emits a zero `∂f/∂η_ruv` column (η_ruv is
    /// absent from CL/V/KA), so the residual-eta column comes entirely from the
    /// provider-agnostic `residual_inner_obs` term — exactly as on the closed-form path.
    /// Both tails.
    #[test]
    fn analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  bloq_method = m3\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV + M3 + iiv_on_ruv");
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(model.is_ode_based(), "must be on the ODE path");
        // The ODE triple is analytic on both loops as of #486.
        assert!(crate::sens::provider::iov_sens_supported(&model));
        assert!(!analytic_inner_common_bail(&model));

        for cens_sign in [1i8, -1] {
            let mut subject = Subject {
                id: "1".into(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
                obs_raw_times: Vec::new(),
                observations: vec![0.0; 6],
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0, 0, cens_sign, cens_sign],
                occasions: vec![1, 1, 1, 2, 2, 2],
                dose_occasions: vec![1, 2],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            };
            let params = model.default_params.clone();
            let preds = crate::pk::predict_iov(
                &model,
                &subject,
                &params.theta,
                &[0.12, -0.08, 0.2, 0.10],
                &[vec![0.05], vec![-0.07]],
            );
            subject.observations = preds.iter().map(|p| p * 0.85).collect();
            let n_eta = model.n_eta;
            let n_kappa = model.n_kappa;
            let k = iov_occasion_groups(&subject).len();
            let n_stacked = n_eta + k * n_kappa;
            let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
            // Non-zero η_ruv (index 3) so the residual-variance scaling is exercised.
            let stacked = vec![0.10, -0.05, 0.08, 0.12, 0.05, -0.07];
            assert_eq!(stacked.len(), n_stacked);

            let g = analytic_eta_nll_gradient_iov(
                &model,
                &subject,
                &params.theta,
                &stacked,
                &params.omega,
                omega_iov,
                &params.sigma.values,
                n_eta,
                n_kappa,
                k,
                None,
            )
            .expect("analytic ODE IOV + M3 + iiv_on_ruv inner gradient");

            let nll = |s: &[f64]| -> f64 {
                let eta_t = &s[..n_eta];
                let kappas: Vec<Vec<f64>> = (0..k)
                    .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                    .collect();
                individual_nll_iov(
                    &model,
                    &subject,
                    &params.theta,
                    eta_t,
                    &kappas,
                    &params.omega,
                    Some(omega_iov),
                    &params.sigma.values,
                )
            };
            for p in 0..n_stacked {
                let h = 1e-5 * (1.0 + stacked[p].abs());
                let fd_at = |hh: f64| -> f64 {
                    let mut sp = stacked.clone();
                    sp[p] += hh;
                    let mut sm = stacked.clone();
                    sm[p] -= hh;
                    (nll(&sp) - nll(&sm)) / (2.0 * hh)
                };
                let f1 = fd_at(h);
                let f2 = fd_at(h / 2.0);
                let fd = (4.0 * f2 - f1) / 3.0;
                approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-5);
            }
        }
    }

    /// The triple **M3 + IOV + `iiv_on_ruv`** (#591): the analytic stacked-η inner
    /// gradient (`analytic_eta_nll_gradient_iov`) must match Richardson FD of
    /// `individual_nll_iov` over `[η_bsv, η_ruv, κ₁..κ_K]` when censored rows co-occur with
    /// the `exp(2·η_ruv)` residual-variance scaling. `residual_inner_obs` returns the
    /// censored `(h·m, h·z)` pair (f-coefficient + residual-eta column) on a censored row
    /// under `iiv_on_ruv`, and the residual variance carries the `η_ruv` scale on every
    /// row. Proves the gate flip ships a correct *triple* inner gradient.
    #[test]
    fn iov_m3_iiv_on_ruv_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3 + iiv_on_ruv");
        model.bloq_method = crate::types::BloqMethod::M3;
        // The closed-form triple is analytic on both loops as of #591; the ODE IOV triple
        // is analytic as of #486 (see `analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode`).
        assert_eq!(model.residual_error_eta, Some(3));
        assert!(crate::sens::provider::iov_analytical_supported(&model));
        assert!(crate::sens::provider::iov_sens_supported(&model));
        assert!(!analytic_inner_common_bail(&model));

        let mut subject = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            // Occasion-2 tail rows M3 left-censored, co-occurring with iiv_on_ruv.
            cens: vec![0, 0, 0, 0, 1, 1],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: vec![1, 2],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        // Shallow censoring (≈ 0.85·f → z ≈ −0.75), the regime where the A&S CDF and the
        // exact φ in `inv_mills` agree to FD precision.
        let preds = crate::pk::predict_iov(
            &model,
            &subject,
            &params.theta,
            &[0.12, -0.08, 0.2, 0.10],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        // Non-zero η_ruv (index 3) so the residual-variance scaling is exercised.
        let stacked = vec![0.10, -0.05, 0.08, 0.12, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV + M3 + iiv_on_ruv inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-5 * (1.0 + stacked[p].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut sp = stacked.clone();
                sp[p] += hh;
                let mut sm = stacked.clone();
                sm[p] -= hh;
                (nll(&sp) - nll(&sm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// The IOV inner loop must honour the same model-level FD bails as the non-IOV inner
    /// (#466 review #1/#3): `gradient = fd` / escape hatch, IIV-on-residual-error
    /// (`iiv_on_ruv`), and LTBS all force the FD inner gradient — the shared
    /// `analytic_inner_common_bail` gate `find_ebe_iov` now consults. Without it an
    /// IOV + `iiv_on_ruv` fit would build the inner gradient on an unscaled residual
    /// variance, and `gradient = fd` would silently fail to disable the analytic inner.
    /// (`ExpressionScale` is no longer a common bail — the non-IOV analytical inner serves
    /// it via the quotient rule. The **ODE** IOV path now serves an η-dependent `obs_scale`
    /// too via the post-walk quotient (#575/#590); the **closed-form** IOV
    /// path still declines it (`iov_analytical_supported` requires `ScalingSpec::None`) —
    /// both pinned by the `iov_sens_supported` assertions below.)
    #[test]
    fn iov_inner_honours_common_bails() {
        use crate::parser::model_parser::parse_model_string;
        let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
        // Clean IOV model: no common bail → analytic inner runs.
        assert!(!analytic_inner_common_bail(&model));
        // `gradient = fd` (and the escape hatch) force FD (#466 review #3).
        model.gradient_method = GradientMethod::Fd;
        assert!(analytic_inner_common_bail(&model));
        model.gradient_method = GradientMethod::default();
        assert!(!analytic_inner_common_bail(&model));
        // IIV on residual error is NO LONGER a blanket common bail (#4b): the inner
        // gradient now carries the `exp(2·η_ruv)` scaling and the `η_ruv` variance column.
        // For this **IOV** model (`n_kappa > 0`), even the M3 triple is analytic as of #486;
        // the non-IOV ODE M3 + `iiv_on_ruv` combo is analytic as well (#623).
        model.residual_error_eta = Some(0);
        assert!(!analytic_inner_common_bail(&model));
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(!analytic_inner_common_bail(&model));
        model.bloq_method = crate::types::BloqMethod::Drop;
        model.residual_error_eta = None;
        // LTBS × IOV now takes the FD *inner* gradient via `analytic_inner_common_bail`'s
        // `log_transform && n_kappa > 0` clause (#486): the closed-form OUTER IOV gradient
        // serves LTBS (so `iov_analytical_supported` admits `log_transform`), but the Dual1
        // inner walk carries no `ln` jet, so the inner stays on FD. This model is ODE-based,
        // so its *outer* IOV gate (`ode_iov_supported`) still declines LTBS independently —
        // `iov_sens_supported` is false here for a different reason (the ODE path).
        model.log_transform = true;
        assert!(
            analytic_inner_common_bail(&model),
            "LTBS × IOV declines the analytic inner via the common bail (#486)"
        );
        assert!(
            !crate::sens::provider::iov_sens_supported(&model),
            "ODE IOV + LTBS still routes to FD via the ODE IOV support gate"
        );
        model.log_transform = false;

        // The *outer* IOV gate (`iov_sens_supported`) for this **ODE** model now admits
        // `iiv_on_ruv` and the M3 triple (#486): the ODE walk emits a zero `∂f/∂η_ruv`
        // column and the shared assembly applies the variance scaling — proven by the
        // dedicated FD-comparison tests (`analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode`
        // here and `iov_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd` in
        // `sens_outer_gradient`). FREM still routes to FD.
        assert!(crate::sens::provider::iov_sens_supported(&model));
        model.residual_error_eta = Some(0);
        assert!(
            crate::sens::provider::iov_sens_supported(&model),
            "ODE IOV + iiv_on_ruv is analytic as of #486"
        );
        model.residual_error_eta = None;
        assert!(crate::sens::provider::iov_sens_supported(&model));

        // ODE IOV + η-dependent `ExpressionScale` `obs_scale` is analytic (#575/#590):
        // the post-walk quotient carries `d(obs_scale)/d(stacked-η)`, so `ode_iov_supported`
        // (and hence `iov_sens_supported`) admits it. LTBS still declines — pinned in
        // `sens::provider::tests::ode_iov_expr_scale_supported_and_gated`.
        let iov_scaled = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + obs_scale");
        assert!(
            matches!(
                iov_scaled.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ),
            "obs_scale = 1000/V must parse as an η-dependent ExpressionScale"
        );
        assert!(
            crate::sens::provider::iov_sens_supported(&iov_scaled),
            "ODE IOV + η-dependent obs_scale is analytic via the post-walk quotient (#575/#590)"
        );

        // The CLOSED-FORM IOV path (`iov_analytical_supported`) now admits an η-dependent
        // `ExpressionScale` `obs_scale` too (#486): the closed-form event-driven walk applies
        // the same per-occasion-group post-walk quotient as the ODE path. LTBS is served on
        // the OUTER gradient now (#486) — pinned in
        // `sens::provider::tests::iov_analytical_expr_scale_supported_and_gated`.
        let iov_scaled_cf = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + obs_scale");
        assert!(matches!(
            iov_scaled_cf.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ));
        assert!(
            crate::sens::provider::iov_sens_supported(&iov_scaled_cf),
            "closed-form IOV + η-dependent obs_scale is analytic via the post-walk quotient (#486)"
        );
        let mut iov_scaled_cf_ltbs = iov_scaled_cf;
        iov_scaled_cf_ltbs.log_transform = true;
        // Closed-form IOV + obs_scale + LTBS: the OUTER gradient is served now (#486 — the
        // `ln(f)` jet applied after the in-walk scale quotient), so `iov_sens_supported` is
        // true; the inner EBE gradient still declines via `analytic_inner_common_bail`.
        assert!(
            crate::sens::provider::iov_sens_supported(&iov_scaled_cf_ltbs),
            "closed-form IOV + obs_scale + LTBS is served on the OUTER gradient (#486)"
        );
        assert!(
            analytic_inner_common_bail(&iov_scaled_cf_ltbs),
            "closed-form IOV + obs_scale + LTBS still declines the analytic inner"
        );
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

    /// IIV on residual error (#474): the closed-form inner η-gradient must match a
    /// The inner-gradient model gate accepts a closed-form `iiv_on_ruv` model AND
    /// closed-form **M3 BLOQ + `iiv_on_ruv`** (#4c — the censored × residual-eta
    /// cross-terms are assembled). The **non-IOV ODE** M3 + `iiv_on_ruv` combo is analytic
    /// as well (#623), as is the ODE *IOV* triple (#486).
    #[test]
    fn analytic_inner_grad_gate_iiv_on_ruv() {
        use crate::parser::model_parser::parse_model_string;
        let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.13,0.001,10.0)\n  theta TVV(8.0,0.1,500.0)\n  theta TVKA(1.0,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n",
        )
        .expect("parse");
        assert!(analytic_inner_grad_supported_model(&model));
        // #4c: closed-form M3 + iiv_on_ruv is now analytic (was FD).
        model.bloq_method = crate::types::BloqMethod::M3;
        assert!(analytic_inner_grad_supported_model(&model));
    }

    /// central finite difference of the production `individual_nll` (which applies
    /// the `exp(2·η_ruv)` variance scaling) at a non-zero η — including the `η_ruv`
    /// column, which the shared `coef·∂f/∂η` loop never touches (`∂f/∂η_ruv = 0`).
    #[test]
    fn analytic_eta_gradient_matches_fd_iiv_on_ruv() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.13,0.001,10.0)\n  theta TVV(8.0,0.1,500.0)\n  theta TVKA(1.0,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n",
        )
        .expect("parse");
        assert_eq!(model.residual_error_eta, Some(3));
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![2.1, 3.4, 4.0, 3.1, 1.8, 1.1, 0.4],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // A genuinely non-zero η, including the residual-error component.
        check_inner_ruv_grad(&model, &subject, &[0.20, -0.15, 0.30, 0.25]);
    }

    /// ODE counterpart of [`analytic_eta_gradient_matches_fd_iiv_on_ruv`]: the
    /// residual-variance scaling and `η_ruv` column live in the shared, provider-
    /// agnostic `analytic_eta_nll_gradient`, so the light ODE `Dual1` walk serves
    /// `iiv_on_ruv` too (#474). Verified against FD of the production `individual_nll`.
    #[test]
    fn analytic_eta_gradient_matches_fd_iiv_on_ruv_ode() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
        assert_eq!(model.residual_error_eta, Some(2));
        assert!(model.ode_spec.is_some());
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![28.0, 25.0, 20.0, 13.0, 5.5, 2.4, 0.5],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        check_inner_ruv_grad(&model, &subject, &[0.15, -0.10, 0.25]);
    }

    /// Compare the analytic inner η-gradient to a central FD of the production
    /// `individual_nll` (which scales the residual variance by `exp(2·η_ruv)`) at a
    /// non-zero η — including the `η_ruv` column that `∂f/∂η = 0` leaves to the
    /// variance term.
    fn check_inner_ruv_grad(model: &CompiledModel, subject: &Subject, eta: &[f64]) {
        let params = model.default_params.clone();
        let analytic = analytic_eta_nll_gradient(
            model,
            subject,
            &params.theta,
            eta,
            &params.omega,
            &params.sigma.values,
        )
        .expect("ruv model is in analytic inner scope");

        let nll = |e: &[f64]| {
            crate::stats::likelihood::individual_nll(
                model,
                subject,
                &params.theta,
                e,
                &params.omega,
                &params.sigma.values,
            )
        };
        for k in 0..model.n_eta {
            let h = 1e-6 * (1.0 + eta[k].abs());
            let mut ep = eta.to_vec();
            ep[k] += h;
            let mut em = eta.to_vec();
            em[k] -= h;
            let fd = (nll(&ep) - nll(&em)) / (2.0 * h);
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// Dense-`R` (`block_sigma`, #627) analytic inner η-gradient must match a central
    /// FD of the production `individual_nll`, which routes a correlated-residual model
    /// through the dense data term. The `combined(PROP,ADD)` + correlated `block_sigma`
    /// modifies the diagonal residual variance and its `∂/∂f` (the within-observation
    /// `2ρσ_iσ_j c_i c_j` cross term) — exactly the term the plain scalar `dvar_df` omits
    /// and the dense kernel (`compute_dr_df_matrices`) carries. Pins that the dense inner
    /// branch reduces correctly and stays consistent with the marginal it optimises.
    #[test]
    fn dense_residual_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse correlated block_sigma model");
        assert!(
            !model.residual_correlations.is_empty(),
            "fixture must carry a residual correlation"
        );
        assert!(
            !analytic_inner_common_bail(&model),
            "block_sigma must now take the analytic inner gradient (#627)"
        );

        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 5],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        // Nonzero residuals: predictions at η=0.1 nudged, scored at a different η.
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1],
        );
        subject.observations = preds.iter().map(|p| p * 1.15 + 0.3).collect();
        check_inner_ruv_grad(&model, &subject, &[-0.2]);
    }

    /// ODE variant of [`dense_residual_inner_grad_matches_fd`]: the dense-R inner branch
    /// reuses the light `Dual1` ODE walk's per-obs `∂f/∂η`, so the same correlated
    /// `block_sigma` model on an ODE structural model must also match FD of `individual_nll`.
    /// Exercises the ODE-branch gate flip (`analytic_inner_grad_supported`, #627).
    #[test]
    fn dense_residual_ode_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE correlated block_sigma model");
        assert!(!model.residual_correlations.is_empty());
        assert!(model.ode_spec.is_some());
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 5],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        assert!(
            analytic_inner_grad_supported(&model, &subject),
            "ODE block_sigma subject should take the analytic (Dual1) inner gradient (#627)"
        );
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1],
        );
        subject.observations = preds.iter().map(|p| p * 1.15 + 0.3).collect();
        check_inner_ruv_grad(&model, &subject, &[-0.2]);
    }

    /// `block_sigma` + η-dependent `ExpressionScale` `obs_scale` (#627 × #486): the dense
    /// inner gradient must still match FD of `individual_nll`. The scale enters only
    /// through `∂f/∂η` (quotient rule, from the provider) and the scaled prediction that
    /// `R` is built on, so the dense branch composes with it. Pins the numerical side of
    /// `expression_scale_with_correlated_residual_is_analytic_both_loops`.
    #[test]
    fn dense_expression_scale_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  omega ETA_V ~ 0.04 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse block_sigma + ExpressionScale");
        assert!(matches!(
            model.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ));
        assert!(!analytic_inner_common_bail(&model));
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 5],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1, -0.05],
        );
        subject.observations = preds.iter().map(|p| p * 1.1 + 5.0).collect();
        check_inner_ruv_grad(&model, &subject, &[-0.15, 0.12]);
    }

    /// Parse an ODE + LTBS (`log_additive`) + `iiv_on_ruv` model and build a
    /// subject with log-scale observations, shared by the two ODE-LTBS inner tests.
    fn ode_ltbs_ruv_model_and_subject() -> (CompiledModel, Subject) {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
        assert!(model.log_transform, "log_additive must set LTBS");
        assert_eq!(model.residual_error_eta, Some(2));
        // Predictions for an LTBS model are on the log scale; perturb them so the
        // residual is nonzero.
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 7],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1, -0.1, 0.0],
        );
        subject.observations = preds.iter().map(|p| p + 0.2).collect();
        (model, subject)
    }

    /// ODE + LTBS + `iiv_on_ruv`: the analytic inner η-gradient must match a central
    /// FD of the production `individual_nll` (which applies the `g = ln(f)` wrap and
    /// the `exp(2·η_ruv)` scale). Confirms the residual-eta column and the log chain
    /// compose correctly (#474).
    #[test]
    fn ode_ltbs_inner_grad_matches_fd() {
        let (model, subject) = ode_ltbs_ruv_model_and_subject();
        check_inner_ruv_grad(&model, &subject, &[0.15, -0.10, 0.20]);
    }

    /// The covariance concern that kept LTBS on the FD inner (#438): the analytic
    /// EBE must coincide with the FD (objective's-own) EBE. For ODE the Dual1 walk
    /// shares `solve_ode_g` with `individual_nll`, so they agree to integrator
    /// tolerance — leaving the covariance Hessian clean (#474).
    #[test]
    fn ode_ltbs_inner_ebe_matches_fd() {
        let (mut model, subject) = ode_ltbs_ruv_model_and_subject();
        let params = model.default_params.clone();

        model.gradient_method = GradientMethod::Auto; // analytic inner
        assert!(
            analytic_inner_grad_supported(&model, &subject),
            "ODE-LTBS subject should now take the analytic inner gradient"
        );
        let analytic = find_ebe(&model, &subject, &params, 200, 1e-10, None, None);

        model.gradient_method = GradientMethod::Fd; // force FD inner
        assert!(!analytic_inner_grad_supported(&model, &subject));
        let fd = find_ebe(&model, &subject, &params, 200, 1e-10, None, None);

        assert!(
            analytic.converged && fd.converged,
            "both EBE solves converge"
        );
        for k in 0..model.n_eta {
            approx::assert_relative_eq!(
                analytic.eta[k],
                fd.eta[k],
                max_relative = 1e-5,
                epsilon = 1e-7
            );
        }
    }

    /// #576/#486: an ODE model carrying a custom residual-error magnitude now
    /// takes the analytic inner EBE gradient too — `residual_inner_obs` (shared by
    /// the closed-form and ODE inner paths) threads the η-independent
    /// per-observation multiplier into the variance/its `f`-derivative, so the
    /// gradient stays magnitude-aware without falling back to FD. A control model
    /// without the magnitude is unaffected (same analytic route as before).
    #[test]
    fn ode_custom_magnitude_takes_analytic_inner_gradient() {
        use crate::parser::model_parser::parse_model_string;
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![5.0, 4.0, 3.0, 2.0, 1.0],
            obs_cmts: vec![1; 5],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 5],
            occasions: vec![1; 5],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };

        // Control: plain proportional ODE → analytic inner gradient.
        let mut plain = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse plain ODE");
        plain.gradient_method = GradientMethod::Auto;
        assert!(!plain.has_custom_ruv_magnitude());
        assert!(
            analytic_inner_grad_supported(&plain, &subject),
            "plain ODE model should take the analytic inner gradient"
        );

        // Same model with a TIME-varying residual magnitude → must bail to FD.
        let mut mag = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  theta RUV_LATE(1.5,0.0,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR * (if (TIME > 4.0) RUV_LATE else 1.0))\n[fit_options]\n  method = focei\n",
        )
        .expect("parse ODE + custom magnitude");
        mag.gradient_method = GradientMethod::Auto;
        assert!(
            mag.has_custom_ruv_magnitude(),
            "fixture must carry a custom residual magnitude"
        );
        assert!(
            analytic_inner_grad_supported(&mag, &subject),
            "ODE + custom magnitude should take the analytic inner gradient (#576/#486)"
        );
    }

    /// #576/#486: the closed-form analytic inner η-gradient of a custom / time-
    /// varying residual-magnitude model must match FD of the (already magnitude-
    /// aware) `individual_nll` — the magnitude is η-independent, so
    /// `residual_inner_obs` only needs the per-observation multiplier threaded
    /// into the variance/its `f`-derivative, no new η term.
    #[test]
    fn magnitude_inner_eta_gradient_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  theta RUV_LATE(1.5,0.1,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n",
        )
        .expect("parse magnitude model");
        assert!(model.has_custom_ruv_magnitude());
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 7],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let theta = vec![0.22, 11.0, 1.4, 1.6];
        let preds =
            crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &[0.1, -0.1, 0.05]);
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let mut params = model.default_params.clone();
        params.theta = theta.clone();
        let eta = [0.15_f64, -0.10, 0.20];
        let analytic = analytic_eta_nll_gradient(
            &model,
            &subject,
            &params.theta,
            &eta,
            &params.omega,
            &params.sigma.values,
        )
        .expect("magnitude model is in the analytic inner scope");
        for k in 0..model.n_eta {
            let h = 1e-6 * (1.0 + eta[k].abs());
            let mut ep = eta;
            ep[k] += h;
            let mut em = eta;
            em[k] -= h;
            let nllp = crate::stats::likelihood::individual_nll(
                &model,
                &subject,
                &params.theta,
                &ep,
                &params.omega,
                &params.sigma.values,
            );
            let nllm = crate::stats::likelihood::individual_nll(
                &model,
                &subject,
                &params.theta,
                &em,
                &params.omega,
                &params.sigma.values,
            );
            let fd = (nllp - nllm) / (2.0 * h);
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
        }
    }

    /// Gate test: `SDE` diffusion combined with a custom residual magnitude must
    /// still route the inner gradient to FD — #576/#486 relaxes the plain
    /// magnitude bail in `analytic_inner_common_bail`, but SDE stays its own,
    /// independent reason to decline (`model.is_sde()`).
    #[test]
    fn magnitude_with_sde_still_routes_inner_to_fd() {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  theta RUV_LATE(1.5,0.1,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[diffusion]\n  central ~ 0.05 FIX\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n",
        )
        .expect("parse SDE + magnitude model");
        assert!(model.is_sde(), "fixture must be an SDE model");
        assert!(model.has_custom_ruv_magnitude());
        assert!(
            analytic_inner_common_bail(&model),
            "SDE + custom magnitude must still bail the inner gradient to FD"
        );
    }

    /// Parse a plain ODE + LTBS (`log_additive`) model with **no** `iiv_on_ruv`
    /// — the exact #438 case — and a subject with log-scale observations.
    fn ode_ltbs_no_ruv_model_and_subject() -> (CompiledModel, Subject) {
        use crate::parser::model_parser::parse_model_string;
        let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
        assert!(model.log_transform, "log_additive must set LTBS");
        assert_eq!(model.residual_error_eta, None, "no iiv_on_ruv");
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 7],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1, -0.1],
        );
        subject.observations = preds.iter().map(|p| p + 0.2).collect();
        (model, subject)
    }

    /// #438 regression: PR #474 flipped plain ODE + LTBS (no `iiv_on_ruv`) onto the
    /// analytic inner. The #438 concern was that the analytic EBE could drift off the
    /// objective's own EBE and inflate the covariance Hessian (~5× SEs). For the ODE
    /// `Dual1` walk that shares `solve_ode_g` with `individual_nll` this must NOT
    /// happen — the analytic and FD EBEs must coincide to integrator tolerance.
    #[test]
    fn ode_ltbs_no_ruv_inner_ebe_matches_fd() {
        let (mut model, subject) = ode_ltbs_no_ruv_model_and_subject();
        let params = model.default_params.clone();

        model.gradient_method = GradientMethod::Auto; // analytic inner
        assert!(
            analytic_inner_grad_supported(&model, &subject),
            "plain ODE-LTBS subject should take the analytic inner gradient"
        );
        let analytic = find_ebe(&model, &subject, &params, 200, 1e-10, None, None);

        model.gradient_method = GradientMethod::Fd; // force FD inner
        assert!(!analytic_inner_grad_supported(&model, &subject));
        let fd = find_ebe(&model, &subject, &params, 200, 1e-10, None, None);

        assert!(
            analytic.converged && fd.converged,
            "both EBE solves converge"
        );
        for k in 0..model.n_eta {
            // Two independently-converged EBE solves; agreement to ~1e-4 confirms
            // no #438-style drift (which inflated SEs ~5×, i.e. ~400% off).
            approx::assert_relative_eq!(
                analytic.eta[k],
                fd.eta[k],
                max_relative = 1e-4,
                epsilon = 1e-6
            );
        }
    }

    /// ODE + LTBS + `iiv_on_ruv` with an **eta-dependent initial condition**
    /// (`init(central) = C0·V`, as in the thioguanine `run14` model). The analytic
    /// inner gradient must still match FD of `individual_nll` — confirms the init-
    /// condition η-derivative composes with the log wrap and residual-eta column.
    #[test]
    fn ode_ltbs_init_cond_inner_grad_matches_fd() {
        use crate::parser::model_parser::parse_model_string;
        let src = "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  C0 = 5.0\n[structural_model]\n  ode(states=[central])\n[odes]\n  init(central) = C0 * V\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-11\n  ode_abstol = 1e-13\n";
        let model = parse_model_string(src).expect("parse");
        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 7],
            obs_cmts: vec![1; 7],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 7],
            occasions: vec![1; 7],
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let preds = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.1, -0.1, 0.0],
        );
        subject.observations = preds.iter().map(|p| p + 0.2).collect();
        let params = model.default_params.clone();
        let eta = [0.15_f64, -0.10, 0.20];
        let analytic = analytic_eta_nll_gradient(
            &model,
            &subject,
            &params.theta,
            &eta,
            &params.omega,
            &params.sigma.values,
        )
        .expect("scope");
        for k in 0..model.n_eta {
            let h = 1e-6 * (1.0 + eta[k].abs());
            let mut ep = eta;
            ep[k] += h;
            let mut em = eta;
            em[k] -= h;
            let nllp = crate::stats::likelihood::individual_nll(
                &model,
                &subject,
                &params.theta,
                &ep,
                &params.omega,
                &params.sigma.values,
            );
            let nllm = crate::stats::likelihood::individual_nll(
                &model,
                &subject,
                &params.theta,
                &em,
                &params.omega,
                &params.sigma.values,
            );
            let fd = (nllp - nllm) / (2.0 * h);
            approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
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
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp();
                    p.values[1] = theta[1];
                    p
                },
            ),
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
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            transit_ode_equivalent: None,
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
            analytical_init: Vec::new(),
            analytic_readout: None,
            ruv_magnitude: None,
            transit_ode_equivalent: None,
            name: "noniov_mu".into(),
            has_conditional_eta_params: false,
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp();
                    p.values[1] = theta[1];
                    p
                },
            ),
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

    /// Interaction of the #486 analytic `ExpressionScale` path with correlated residual
    /// error (`block_sigma`): as of #627 the dense-R gradient serves correlated residuals
    /// on BOTH loops, and the η-dependent `obs_scale` composes with it (the scale enters
    /// only through `∂f/∂η` and the scaled prediction, both of which the dense assembly
    /// consumes). So a model with BOTH features is now analytic on both loops —
    /// `analytic_inner_common_bail` false, `analytic_outer_gradient_available` true. The
    /// diagonal control stays analytic too. Inverts the previous FD-pinning assertion.
    #[test]
    fn expression_scale_with_correlated_residual_is_analytic_both_loops() {
        use crate::parser::model_parser::parse_model_string;
        let corr = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n",
        )
        .expect("parse ExpressionScale + correlated residual");
        assert!(
            matches!(
                corr.scaling,
                crate::types::ScalingSpec::ExpressionScale { .. }
            ) && !corr.residual_correlations.is_empty(),
            "fixture must carry both an ExpressionScale obs_scale and a residual correlation"
        );
        assert!(
            !analytic_inner_common_bail(&corr),
            "block_sigma + ExpressionScale is now analytic inner (#627)"
        );
        assert!(
            crate::sens::provider::analytic_outer_gradient_available(&corr),
            "block_sigma + ExpressionScale is now analytic outer (#627)"
        );

        // Control: same obs_scale, diagonal (uncorrelated) residual → analytic on both loops.
        let diag = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n  sigma ADD_ERR ~ 0.10\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n",
        )
        .expect("parse ExpressionScale + diagonal residual");
        assert!(diag.residual_correlations.is_empty());
        assert!(!analytic_inner_common_bail(&diag));
        assert!(crate::sens::provider::analytic_outer_gradient_available(
            &diag
        ));
    }

    // `analytical_ad_unsupported` is the VESTIGIAL retired-AD classifier (not consulted by
    // live routing; the live gate is `analytic_inner_grad_supported[_model]`). It still flags
    // four genuinely out-of-scope classes (non-log-normal ETA, LTBS, conditional params, TTE)
    // and — historically — any `ExpressionScale`. The final case below pins the deliberate
    // DIVERGENCE for a differentiable `ExpressionScale`: the classifier flags it, but the live
    // gate serves it analytically (#486). This guards against a regression that re-wires the
    // classifier into routing. Build-independent; runs in the FD-only `ci` build (#278).
    #[test]
    fn analytical_ad_unsupported_flags_each_class() {
        use crate::parser::model_parser::parse_model_string;
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

        // Expression-scale obs_scale: the vestigial classifier flags any `ExpressionScale`.
        model.scaling = crate::types::ScalingSpec::ExpressionScale {
            scale_fn: Box::new(|_, _, _, _| 1.0),
            deriv: None,
        };
        assert!(analytical_ad_unsupported(&model).is_some());
        model.scaling = crate::types::ScalingSpec::ScalarScale(1000.0);
        assert!(analytical_ad_unsupported(&model).is_none());

        // DIVERGENCE pin (#486 / #534 audit): a *differentiable* η-dependent `ExpressionScale`
        // is still flagged by the vestigial classifier, but the LIVE inner gate serves it
        // analytically. If a future change re-wired `analytical_ad_unsupported` into routing,
        // it would silently send analytic ExpressionScale fits back to FD — assert both here so
        // that regression is caught.
        let scaled = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse differentiable ExpressionScale");
        assert!(matches!(
            scaled.scaling,
            crate::types::ScalingSpec::ExpressionScale { deriv: Some(_), .. }
        ));
        assert!(
            analytical_ad_unsupported(&scaled).is_some(),
            "vestigial classifier still flags any ExpressionScale"
        );
        assert!(
            analytic_inner_grad_supported_model(&scaled),
            "but the LIVE inner gate serves a differentiable ExpressionScale analytically (#486)"
        );
    }
}
