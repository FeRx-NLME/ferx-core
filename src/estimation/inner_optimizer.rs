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
    // Discrete / categorical (`[binary_model]`, #760; ordinal/Poisson/… later): the data term
    // has no Dual2 analytic eta-sensitivity (its linear-predictor closure is evaluated
    // numerically), so - exactly like TTE - it routes to the FD inner gradient. Gated on the
    // family-agnostic `has_discrete()` so a future discrete family routes here automatically
    // rather than silently falling through to the (sensitivity-free) analytic gradient.
    // (`has_discrete()` == `has_binary()` today, so this is unchanged for binary models. False
    // without `survival`.)
    if model.has_discrete() {
        return Some("discrete ([binary_model]) likelihood");
    }
    // CTMM (`[markov_model]`, #759): the matrix-exponential transition likelihood has
    // no Dual2 analytic eta-sensitivity (its generator closure is evaluated numerically),
    // so it routes to the FD inner gradient, exactly like TTE / binary.
    #[cfg(feature = "markov")]
    if model.has_ctmm() {
        return Some("CTMM ([markov_model]) transition likelihood");
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
    // #814: attribute against the *effective* model for this subject. A closed-form
    // transit/IG + IOV (or TV-cov / `TIME`) subject evaluates its objective on the ODE twin
    // (`effective_for`, #719), so its FD-fallback reason is a twin-ODE reason even though the
    // analytic primary's own `ode_spec` is `None` — the same reroute `is_ode_iov` and
    // `inner_stall_enabled` already track. `effective_for` returns `self` for a genuine ODE
    // model and for every non-rerouted subject, so this is unchanged for them.
    let eff = model.effective_for(subject);
    if eff.ode_spec.is_some() {
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
            let attr_map = eff.active_dose_attr_map();
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
            && eff
                .ode_spec
                .as_ref()
                .and_then(|o| o.rhs_program.as_ref())
                .is_some_and(|p| p.uses_time_vars())
        {
            return "steady-state dose + time-dependent ODE RHS";
        }
        // Infusion into a built-in absorption compartment (#719 gap 2) declines to FD under IOV
        // (the dual walk's `+rate` would double-count the mass the convolved `R_in_inf` already
        // delivers). `ode_iov_subject_supported` checks this *before* its SS-absorption bail, so
        // naming it here keeps the attribution from drifting to the generic catch-all below.
        if crate::sens::ode_provider::has_infusion_into_input_rate(eff, subject) {
            return "infusion into built-in absorption compartment";
        }
        // #835: a steady-state dose into a built-in absorption compartment is analytic for the
        // smooth density kernels (transit/igd/weibull/first_order) — the dual fixed point carries
        // its κ-coupled sensitivity. Only SS into a `zero_order` window and SS + an absorption
        // lagtime stay on FD; mirror `ode_iov_subject_supported`'s flipped bail via the shared
        // `CompiledModel::ss_absorption_out_of_scope` so this attribution can't drift.
        if eff.ss_absorption_out_of_scope(subject) {
            return "steady-state dose + built-in absorption forcing";
        }
        let occ_groups = iov_occasion_groups(subject);
        if occ_groups.is_empty() {
            return "no observation occasions";
        }
        let n_stacked = eff.n_eta + occ_groups.len() * eff.n_kappa;
        let m_dim = eff.n_theta + n_stacked;
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
///
/// A closed-form transit/IG model with IOV counts too (#814): `n_kappa > 0` routes every one
/// of its subjects to the ODE twin (`effective_for`, #719), so its inner EBE carries the same
/// per-vertex ODE cost as a hand-written `[odes]` IOV model even though the primary's own
/// `ode_spec` is `None`. The reroute is subject-static here (driven by `n_kappa`, not by data),
/// so the model-only classifier is exact.
fn is_ode_iov(model: &CompiledModel) -> bool {
    model.n_kappa > 0 && (model.ode_spec.is_some() || model.absorption_ode_equivalent.is_some())
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

/// Whether the inner BFGS objective-stall convergence stop should be enabled for this subject.
///
/// The stall stop (`#555`) exists to tolerate the adaptive-RK45 gradient-noise floor of an ODE
/// objective, which can sit above `tol` and block the pure `gnorm < tol` criterion. It must key
/// off the **effective** model, not the raw one: a closed-form transit/IG subject rerouted to its
/// ODE twin (TV-cov / `TIME` / IOV — `effective_for`, #719/#814) evaluates its objective on the
/// twin's RK45 integration and so carries that same noise floor, even though the primary model is
/// analytic (`ode_spec == None`). Basing the flag on `model.ode_spec` left it `false` for every
/// rerouted subject, so one that dropped to the FD inner gradient could run to `MAX_INNER` and
/// return an under-converged EBE.
///
/// The reroute considered here is subject-static (TV-cov / `TIME` / IOV via `effective_for`), so
/// a non-rerouted subject — including a constant-parameter, in-domain subject of a twin-carrying
/// model, or a purely analytic model — keeps the exact `gnorm < tol` behaviour and stays
/// bit-identical to prior releases. (`effective_for` returns `self` without building the twin for
/// those, so this adds no cost on the common path.)
fn inner_stall_enabled(model: &CompiledModel, subject: &Subject) -> bool {
    model.effective_for(subject).ode_spec.is_some()
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
/// The per-subject [`pk::event_driven::EventSchedule`] that may be built **once** and
/// reused across many evaluations at differing `eta`, or `None` when no such reuse is
/// sound. The schedule pre-computes the merged event timeline and the per-interval
/// infusion bounds; those are subject-static — and so cacheable — only when nothing
/// eta-dependent can move a baked-in break time:
///
/// - **lagtime** can be eta-dependent and the schedule bakes per-dose times in, so a
///   cached schedule goes stale as eta varies. The non-cached path
///   (`event_driven_predictions`) rebuilds it per call from the current per-dose
///   `PkParams` (which carry lagtime).
/// - **bioavailability `F`** scales a *rate*-defined infusion's duration (#419), which
///   likewise moves the baked-in window as eta varies. Duration-defined infusions keep
///   the cache (`F` scales their rate, not the window).
///
/// Only subjects that actually take the event-driven analytical path (TV covariates or
/// EVID-3/4 resets, closed-form PK) can use a schedule at all; the no-TV fast path never
/// calls `event_driven_predictions`, so `None` costs it nothing.
///
/// Shared by the inner EBE loop ([`find_ebe`]) and the AGQ node sweep
/// ([`crate::estimation::agq`]), which both hold `eta` variable over one subject — so the
/// staleness rules above cannot drift between them.
pub(crate) fn cacheable_schedule(
    model: &CompiledModel,
    subject: &Subject,
) -> Option<pk::event_driven::EventSchedule> {
    if (subject.has_tv_covariates() || subject.has_resets())
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
    }
}

/// Per-coordinate "weakly-identified random effect" detector for the guarded
/// multi-start inner EBE (#891).
///
/// A random effect whose individual objective is *flat* in its own direction has
/// a posterior that is barely tighter than the prior (high per-subject
/// shrinkage). That flatness is exactly what lets a distant, lower posterior mode
/// hide from a single warm/cold BFGS descent, so these are the coordinates worth
/// re-seeding — and, conversely, a well-informed coordinate can be skipped so the
/// multi-start cost is paid only where a missed mode is plausible.
///
/// The signal is a cheap central-difference of the inner objective at the
/// converged mode `eta`. Because `obj` already carries the prior term
/// `½·ηᵀΩ⁻¹η`, the second difference along coordinate `i` estimates the posterior
/// curvature `Hᵢᵢ = data_infoᵢ + (Ω⁻¹)ᵢᵢ`. The prior curvature `(Ω⁻¹)ᵢᵢ` is known
/// exactly, so `data_infoᵢ = Hᵢᵢ − (Ω⁻¹)ᵢᵢ`. A coordinate is flagged weakly
/// identified when the data adds *less* curvature than the prior already carries
/// (`data_infoᵢ < (Ω⁻¹)ᵢᵢ`, i.e. `Hᵢᵢ < 2·(Ω⁻¹)ᵢᵢ`) — a conditional shrinkage of
/// roughly ≳0.3. A non-finite or non-positive curvature (degenerate / genuinely
/// flat) is treated as weakly identified so a pathological coordinate is scanned
/// rather than silently skipped.
///
/// Cost: two `obj` evaluations per non-fixed coordinate, on the cold start only.
fn weakly_identified_coords(
    obj: &dyn Fn(&[f64]) -> f64,
    eta: &[f64],
    nll: f64,
    omega: &OmegaMatrix,
    n_eta: usize,
) -> Vec<bool> {
    // Prior variances below this are effectively fixed effects. `from_matrix`
    // floors a zero-variance diagonal to a 1e-8 eigenvalue (it must stay PD for
    // the cached Cholesky/inverse), so an exactly-zero test never survives; this
    // floor sits above that regularisation yet far below any genuinely free
    // random effect (a 1% CV is variance ~1e-4), so a pinned effect is skipped
    // without ever excluding a real one.
    const FIXED_VAR_FLOOR: f64 = 1e-7;
    let mut flags = vec![false; n_eta];
    if !nll.is_finite() {
        return flags;
    }
    for i in 0..n_eta {
        let var_prior = omega.matrix[(i, i)].max(0.0);
        let sd = var_prior.sqrt();
        // Fixed / effectively-zero-variance effect: it cannot move, so never
        // scan it (the prior pins it at 0 regardless of seed).
        if var_prior <= FIXED_VAR_FLOOR {
            continue;
        }
        let prior_curv = omega.inv[(i, i)];
        if !prior_curv.is_finite() || prior_curv <= 0.0 {
            continue;
        }
        // Central second difference at a one-prior-SD step — the scale at which a
        // flat coordinate is measured against its own prior.
        let h = sd;
        let mut plus = eta.to_vec();
        let mut minus = eta.to_vec();
        plus[i] += h;
        minus[i] -= h;
        let post_curv = (obj(&plus) + obj(&minus) - 2.0 * nll) / (h * h);
        // Degenerate curvature (numerical noise / genuinely flat) → scan.
        if !post_curv.is_finite() || post_curv <= 0.0 {
            flags[i] = true;
            continue;
        }
        // Weakly identified when the data curvature is below the prior curvature.
        flags[i] = post_curv < 2.0 * prior_curv;
    }
    flags
}

pub fn find_ebe(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    eta_init: Option<&[f64]>,
    mu_k: Option<&[f64]>,
    restarts: usize,
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
    let schedule = cacheable_schedule(model, subject);
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
    // Enable the objective-stall convergence stop only when the objective carries the adaptive
    // RK45 gradient-noise floor that can sit above `tol` (#555) — i.e. when the *effective*
    // model for this subject is ODE, which includes a closed-form transit/IG subject rerouted to
    // its ODE twin (TV-cov / `TIME`; #719/#814). Analytical/event-driven objectives are exact, so
    // they keep the pure `gnorm < tol` criterion and stay bit-identical to prior releases.
    let enable_stall = inner_stall_enabled(model, subject);
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

    let mut ebe_converged = bfgs_converged || nm_converged;
    let mut nll = obj(&eta);

    // ── Guarded multi-start inner EBE (`[fit_options] inner_restarts`) ──────
    // A multimodal individual objective traps a single warm-started BFGS in
    // whichever basin the start point sits in. Two families produce this:
    //   1. Saturable / reset dynamics (a high-V/low-conc basin vs a low-V/high-
    //      conc basin) on the event-driven path — subjects with system resets or
    //      time-varying covariates. These are scanned on *every* random effect.
    //   2. Weakly-identified random effects on a nonlinear readout (#891): a flat
    //      individual objective in some η direction admits a distant, lower mode
    //      that a warm/cold BFGS descent silently misses (e.g. fluconazole's
    //      poorly-identified V1, ~48% shrinkage — η̂≈+0.5 vs the global −2.0).
    //      These subjects carry no resets / TV-covariates, so family 1's gate
    //      never scanned them; we now detect the flat coordinate directly and
    //      scan only it.
    // When `inner_restarts > 0` a scanned coordinate re-solves from `inner_restarts`
    // Ω-scaled alternate seeds (`±2·sd`, `±3·sd`, …) and keeps the lowest-objective
    // mode. Gated on a COLD start (`eta_init.is_none()`): the outer loop warm-starts
    // the inner EBE from the previous iteration's mode, so once the basin is picked at
    // iteration 0 the warm start carries it forward — no re-scan every outer eval
    // (~12× slower). One multi-start per subject per fit.
    // An alternate seed replaces the warm-start mode only when it reaches a
    // meaningfully lower objective (`cand_nll + 1e-9 < nll`); a seed that merely
    // reconverges to the same basin is rejected by that guard. So a scanned
    // coordinate's EBE is unchanged from `inner_restarts = 0` unless a seed finds a
    // strictly better mode.
    if restarts > 0 && eta_init.is_none() {
        // Reset / TV-covariate subjects (family 1) scan every coordinate, exactly
        // as before — bit-identical. Everyone else (family 2) is probed per
        // coordinate and scans only the weakly-identified ones, so well-informed
        // subjects pay just `2·n_eta` cheap objective evaluations and their EBE
        // stays bit-identical to `inner_restarts = 0` (no seed solve runs).
        let scan_all = subject.has_resets() || subject.has_tv_covariates();
        let scan_coord: Vec<bool> = if scan_all {
            vec![true; n_eta]
        } else {
            weakly_identified_coords(&obj, &eta, nll, &params.omega, n_eta)
        };
        let base = eta.clone();
        for i in 0..n_eta {
            if !scan_coord[i] {
                continue;
            }
            let sd = params.omega.matrix[(i, i)].max(0.0).sqrt();
            if sd == 0.0 {
                continue;
            }
            for m in 1..=restarts {
                let step = (m as f64 + 1.0) * sd;
                for &k in &[-step, step] {
                    let mut cand = base.clone();
                    cand[i] = k;
                    let ok = inner_minimize_with_grad(
                        &obj,
                        &agrad,
                        &mut cand,
                        n_eta,
                        max_iter,
                        tol,
                        precond.as_deref(),
                        stop_precond,
                        enable_stall,
                    );
                    if cand.iter().all(|v| v.is_finite()) {
                        let cand_nll = obj(&cand);
                        if cand_nll + 1e-9 < nll {
                            nll = cand_nll;
                            eta = cand;
                            ebe_converged = ebe_converged || ok;
                        }
                    }
                }
            }
        }
    }

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
    // ODE objectives carry the adaptive-solver gradient-noise floor; enable the objective-stall
    // stop only for them — including a closed-form transit/IG + IOV subject, whose objective is
    // evaluated on the ODE twin (`effective_for`, #719/#814). See `inner_stall_enabled`/`find_ebe`.
    let enable_stall = inner_stall_enabled(model, subject);
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

/// Whether **every** non-Gaussian record on this subject is a CTMM state observation that
/// the analytic path can serve exactly (#759) — i.e. the subject's whole non-Gaussian data
/// term is `−Σ log P(Δt)[s,s']` on a time-homogeneous generator with a dual-evaluable
/// program ([`ctmm_subject_eta_grad`](crate::markov::endpoint::ctmm_subject_eta_grad)).
///
/// Strict on purpose. The inner gradient is a *sum* of term gradients, so admitting a
/// subject that also carries a TTE / binary / count record would silently drop that term's
/// contribution — a wrong gradient, not a slow one. TTE and binary have no analytic η-channel
/// at all (their `LinearPredictorFn` / `HazardParamFn` are `f64` closures outside `sens/`),
/// so such a subject keeps FD for the whole objective.
///
/// Always `false` without the `markov` feature (there are no CTMM records to serve).
fn ctmm_records_fully_analytic(model: &CompiledModel, subject: &Subject) -> bool {
    #[cfg(feature = "markov")]
    {
        use crate::types::{EndpointLikelihood, ObsRecord};
        if !model.has_ctmm() {
            return false;
        }
        // The CMTs whose endpoint is a CTMM we can differentiate exactly. An endpoint with
        // no `generator_program` — a time-inhomogeneous (drug-driven) generator, whose
        // likelihood is an occupancy ODE rather than an `expm` — is not one of them.
        let analytic_cmts: Vec<usize> = model
            .endpoints
            .iter()
            .filter_map(|(cmt, ep)| match ep {
                EndpointLikelihood::Ctmm {
                    generator_program, ..
                } if generator_program.is_some() => Some(*cmt),
                _ => None,
            })
            .collect();
        subject.obs_records.iter().all(|r| match r {
            ObsRecord::DiscreteState { cmt, .. } => analytic_cmts.contains(cmt),
            _ => false,
        })
    }
    #[cfg(not(feature = "markov"))]
    {
        let _ = (model, subject);
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
    // CTMM (#759): a subject's CTMM term is analytic only when its generator carries a
    // dual-evaluable program. Without one — a drug-driven `Q(t)`, a `(θ, η)` width past the
    // dual dispatch cap, an intensity we could not resolve — *every* subject declines at the
    // survival-record guard in `analytic_inner_grad_supported`, so reporting "analytic" here
    // would be a pure misreport.
    //
    // `analytical_supported` below does **not** catch this, which is the trap: an
    // endpoint-only model (a `[markov_model]` with no `[structural_model]`) is *still* given
    // a default `pk_model` and a vacuous `tv_fn` (`model_parser.rs`, `tv_fn = Some(..)` for
    // any non-ODE model), and its `pk_indices` are empty so the per-slot check passes
    // trivially — so the closed-form scope check waves it straight through despite there
    // being no closed form at all.
    #[cfg(feature = "markov")]
    if model.has_ctmm() && !all_ctmm_endpoints_analytic(model) {
        return false;
    }
    crate::sens::provider::analytical_supported(model)
}

/// Every `Ctmm` endpoint on the model carries a dual-evaluable generator program, i.e. its
/// transition term can be differentiated exactly rather than finite-differenced. The
/// model-level counterpart of [`ctmm_records_fully_analytic`]'s per-subject check.
#[cfg(feature = "markov")]
fn all_ctmm_endpoints_analytic(model: &CompiledModel) -> bool {
    use crate::types::EndpointLikelihood;
    model.endpoints.iter().all(|(_, ep)| match ep {
        EndpointLikelihood::Ctmm {
            generator_program, ..
        } => generator_program.is_some(),
        _ => true,
    })
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
    //
    // CTMM is the exception since #759: its transition likelihood *does* have an exact
    // η-gradient now (`Q` rebuilt over `Dual1` for `∂Q/∂η`, chained through the Van Loan
    // Fréchet derivative of `expm`), and `analytic_eta_nll_gradient_with_schedule` adds it
    // to the Gaussian block. `ctmm_records_fully_analytic` is deliberately strict — it
    // demands that *every* record on the subject be a CTMM state observation the program
    // can serve, so a subject that also carries a TTE / binary / count record (whose terms
    // remain FD-only) still declines the whole gradient rather than silently omitting them.
    if subject_has_survival_records(subject) && !ctmm_records_fully_analytic(model, subject) {
        return false;
    }
    // No Gaussian observations ⇒ no Gaussian data term, hence nothing for the PK provider
    // to serve: an **endpoint-only** fit (a `[markov_model]` with no `[structural_model]`,
    // which the parser admits) has no PK model at all, and `analytical_supported` below
    // would reject it on that basis alone. Its inner gradient is the prior plus the
    // non-Gaussian term, both of which we have exactly, so only the global escape hatches
    // apply. (`analytic_eta_nll_gradient_with_schedule` mirrors this by skipping the
    // provider for such a subject.)
    //
    // **Ordering is load-bearing:** this branch is only sound *after* the guard above. A
    // CTMM subject always carries `DiscreteState` records, so an endpoint-only CTMM whose
    // generator has no dual-evaluable program (past the axis cap, or an intensity we could
    // not resolve) is already rejected there — it can never fall through to here and claim
    // an analytic route the gradient cannot supply. Reaching this line therefore means the
    // subject's non-Gaussian term is either absent or exactly served, and the only thing
    // left is the prior. (Pinned by `program_less_endpoint_only_ctmm_stays_fd`.)
    if subject.obs_times.is_empty() {
        return !analytic_inner_common_bail(model);
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
    // The inner NLL is `½(η'Ω⁻¹η + log|Ω| + data_gauss + 2·data_nonGaussian)`, so its
    // η-gradient is a plain **sum** of term gradients. Assemble the non-Gaussian block
    // first, because it is the one that decides whether this subject can be served at all
    // (the Gaussian provider's own `None` is handled below, as before).
    //
    // CTMM (#759): exact, via `∂Q/∂(θ,η)` over `Dual1` chained through the Van Loan Fréchet
    // derivative of `expm`. The objective's `2×` on the data term and the outer `½` cancel,
    // so this enters at 1× — the same scale `ctmm_subject_eta_grad` returns.
    // TTE / binary / count records have no analytic η-channel and are refused upstream by
    // `analytic_inner_grad_supported` → `ctmm_records_fully_analytic`, so a subject reaching
    // here carries no other non-Gaussian term to omit.
    #[cfg(feature = "markov")]
    let ctmm_grad: Option<Vec<f64>> = if model.has_ctmm() {
        Some(crate::markov::endpoint::ctmm_subject_eta_grad(model, subject, theta, eta)?.1)
    } else {
        None
    };

    // No Gaussian observations ⇒ no Gaussian data term and (for an endpoint-only fit) no PK
    // model for the provider to evaluate. The gradient is prior + non-Gaussian; return it
    // directly rather than asking a provider that has nothing to compute.
    if subject.obs_times.is_empty() {
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        // `mut` is only exercised by the markov CTMM fold below.
        #[cfg_attr(not(feature = "markov"), allow(unused_mut))]
        let mut grad: Vec<f64> = (&omega.inv * &eta_v).as_slice().to_vec();
        #[cfg(feature = "markov")]
        if let Some(g) = &ctmm_grad {
            for (acc, gi) in grad.iter_mut().zip(g.iter()) {
                *acc += gi;
            }
        }
        return Some(grad);
    }

    // Light first-order provider (value + ∂f/∂η only); the inner gradient never
    // needs the second-order / θ blocks the full `subject_sensitivities` carries.
    let sens = crate::sens::provider::subject_eta_grad_with_schedule(
        model,
        subject,
        theta,
        eta,
        cached_schedule,
    )?;
    // Fold the non-Gaussian block into whichever Gaussian branch runs below.
    #[cfg(feature = "markov")]
    let add_nongaussian = |mut g: Vec<f64>| -> Vec<f64> {
        if let Some(c) = &ctmm_grad {
            for (acc, gi) in g.iter_mut().zip(c.iter()) {
                *acc += gi;
            }
        }
        g
    };
    #[cfg(not(feature = "markov"))]
    let add_nongaussian = |g: Vec<f64>| -> Vec<f64> { g };
    // Correlated residual (`block_sigma`, #627): the per-obs `coef·∂f/∂η` loop below
    // assumes a diagonal R. Route the dense-R generalisation here — it serves both the
    // analytical and the ODE (`Dual1`) inner path, since `sens` carries `∂f/∂η` for both.
    if !model.residual_correlations.is_empty() {
        return dense_residual_inner_gradient(model, subject, theta, eta, omega, sigma, &sens)
            .map(add_nongaussian);
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
    // FREM covariate pseudo-observations: the objective scores these rows against the
    // dedicated `EPSCOV` variance, not `error_spec.variance_at(f)`. The provider has
    // already corrected their `f` and `∂f/∂η` (`apply_frem_pseudo_obs_grad`); this is the
    // variance half. `R` is η-independent on such a row, so `∂R/∂f = 0` and the whole
    // residual chain collapses to `∂L/∂η_k = (−ε/R)·δ_{k,ei}`. `None` on a non-FREM model.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma,
    );
    for (j, obs) in sens.iter().enumerate() {
        if let Some(v) = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
            let coef = -(subject.observations[j] - obs.f) / v;
            for k in 0..n_eta {
                grad[k] += coef * obs.df_deta[k];
            }
            continue;
        }
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
    Some(add_nongaussian(grad))
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
            &subject.obs_l2,
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
            &subject.obs_l2,
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
        &subject.obs_l2,
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
    // Not a *finite* descent direction: nothing to do (caller falls back / stops).
    // `dg` must be finite as well as negative: a BFGS update that produced a
    // non-finite search direction gives `dg = ±inf`, and then `-dg·α²/denom`
    // below evaluates to `inf/inf = NaN`, which poisons `alpha` and makes the
    // `clamp` on the *next* trial panic (its bounds `0.1·α`/`0.5·α` become NaN).
    if !(dg < 0.0) || !dg.is_finite() {
        return (0.0, f0);
    }

    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = obj(&x_new);
        if f_new.is_finite() && f_new <= f0 + c1 * alpha * dg {
            return (alpha, f_new);
        }
        // Minimiser of the quadratic matching f0, dg (slope at 0) and f_new at
        // the current alpha. Safeguard into [0.1·α, 0.5·α] so a flat/non-convex
        // sample still makes definite progress (never larger than plain halving,
        // never a near-zero collapse). A non-finite `f_new` (an out-of-domain
        // trial η where an absorption closed form leaves its convergence region
        // and returns ±inf/NaN) carries no interpolation information, so fall
        // back to plain halving — this also keeps `alpha` finite, so the clamp
        // bounds below can never become NaN.
        let denom = 2.0 * (f_new - f0 - dg * alpha);
        let alpha_quad = if f_new.is_finite() && denom > 0.0 {
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
    overwrite_frem_pseudo_obs_rows(&mut h, model, subject, n_eta);

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
    run_inner_loop_warm(model, population, params, max_iter, tol, None, None, 0, 0)
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
#[allow(clippy::too_many_arguments)]
pub fn run_inner_loop_warm(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    prev_etas: Option<&[DVector<f64>]>,
    mu_k: Option<&[f64]>,
    min_obs: usize,
    restarts: usize,
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
            find_ebe(model, subject, params, max_iter, tol, init, mu_k, restarts)
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
#[path = "inner_optimizer_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "inner_optimizer_iov_tests.rs"]
mod iov_tests;
