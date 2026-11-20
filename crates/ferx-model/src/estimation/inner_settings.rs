//! The inner (EBE) search's settings and the scope predicates the likelihood and the adjoint
//! read to decide which subjects the analytic paths serve. Below the estimators, with no
//! dependency on them, so the model/numerics layer can name them (#3041).
use crate::pk;
use crate::types::*;

/// `cacheable_schedule` for an IOV subject (#1564): its staleness rules alone, without
/// the routing arm. An IOV subject on an event-driven model always walks the event-driven
/// funnel (`pk::predict_iov_in`), which otherwise rebuilt and re-sorted the schedule inside
/// every objective evaluation of the EBE search (10.6% of a `warfarin_iov` fit).
pub fn cacheable_schedule_iov(
    model: &CompiledModel, subject: &Subject,
) -> Option<pk::event_driven::EventSchedule> {
    schedule_stays_fresh(model, subject).then(|| subject_schedule(model, subject))
}

/// [`cacheable_schedule_iov`], built once per fit for a registered subject
/// ([`kernel_jit::prepared`](crate::parser::model_parser::rhs_regcode::kernel_jit::prepared)): it
/// is a function of the subject and the model's identity alone.
pub fn cacheable_schedule_iov_cached(
    model: &CompiledModel, subject: &Subject,
) -> std::sync::Arc<Option<pk::event_driven::EventSchedule>> {
    crate::parser::model_parser::rhs_regcode::kernel_jit::prepared(
        subject,
        ("iov-schedule", model.n_eta, model.n_kappa, model.plan_id.get()),
        || cacheable_schedule_iov(model, subject),
    )
}

/// The staleness rules of `cacheable_schedule`, the one place they live, so the IOV
/// gate ([`cacheable_schedule_iov`]) and the routed one cannot drift: an analytical,
/// event-driven, non-algebraic model, with no lagtime and no `F` reshaping a rate-defined
/// infusion's window.
pub fn schedule_stays_fresh(model: &CompiledModel, subject: &Subject) -> bool {
    model.ode_spec.is_none()
        // A compartment-free model (#811) never runs the event-driven walk — the
        // predictor short-circuits before it — and its `pk_model` is a placeholder,
        // so a schedule built from it would be a cache for a route nothing takes.
        // Time-varying covariates are the norm for such models (a per-row study or
        // arm covariate), so this is the common path, not an edge case.
        && !model.is_algebraic()
        && pk::event_driven::supports_event_driven(model.pk_model)
        && !model.has_lagtime()
        && !(model.has_bioavailability() && subject.has_rate_defined_infusion())
}

pub fn subject_schedule(
    model: &CompiledModel, subject: &Subject,
) -> pk::event_driven::EventSchedule {
    pk::event_driven::EventSchedule::for_subject(subject, model.pk_model, &subject.doses, &[])
}

/// The inner-loop settings an EBE solve runs under: `FitOptions::inner_optimizer` and
/// `FitOptions::ebe_warm_start` of the call that scheduled it. The default (`Auto`, warm
/// start off) is what a thread outside any fit-scoped call reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InnerSettings {
    pub mode: crate::types::InnerOptimizer,
    pub warm: bool,
}

impl InnerSettings {
    /// True for the settings a thread with nothing armed reads, i.e. arming this is a no-op.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// True when the subject carries survival/TTE observation records, whose hazard-likelihood
/// term neither analytic inner provider models — both the non-IOV and IOV inner gradients
/// decline such subjects (a single source so the two cannot drift; #466 review round 2).
/// Always `false` without the `survival` feature.
#[inline]
pub fn subject_has_survival_records(subject: &Subject) -> bool {
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

/// Whether `analytic_eta_nll_gradient_with_schedule` serves **every** non-Gaussian term
/// `subject` carries: none at all, or only CTMM state records and `[binary_model]` records
/// whose exact programs cover them.
///
/// That function asks this itself and declines (`None` → FD) when the answer is no, since
/// for a subject with no Gaussian rows it returns the Ω prior plus the served non-Gaussian
/// blocks and nothing else. The inner loop also asks up front, through
/// `analytic_inner_grad_supported`, so its route report matches what runs.
pub fn non_gaussian_eta_grad_served(model: &CompiledModel, subject: &Subject) -> bool {
    !subject_has_survival_records(subject) || non_gaussian_records_fully_analytic(model, subject)
}

/// Whether **every** non-Gaussian record on this subject is one the analytic path can
/// serve exactly — a CTMM state observation on an endpoint with a dual-evaluable generator
/// program (#759, [`ctmm_subject_eta_grad`](crate::markov::endpoint::ctmm_subject_eta_grad)),
/// or a binary observation on an endpoint with a dual-evaluable linear predictor (#2024,
/// [`discrete_subject_eta_grad`](crate::categorical::discrete_subject_eta_grad)).
///
/// Strict on purpose. The inner gradient is a *sum* of term gradients, so admitting a
/// subject that also carries a record whose term has no exact channel would silently drop
/// that term's contribution — a wrong gradient, not a slow one. A count record (#2877) is
/// served like a binary one, by its endpoint's `param_program`. A TTE record is served when its
/// endpoint has an exact channel ([`crate::survival::tte_subject_eta_grad`]): an analytic
/// hazard with a dual-evaluable program, or an ODE-accumulated one whose subject the dual
/// walk serves (`survival::tte_record_served`). A record without one keeps FD for the whole
/// objective.
///
/// Always `false` without the `survival` feature (there are no such records to serve).
pub fn non_gaussian_records_fully_analytic(model: &CompiledModel, subject: &Subject) -> bool {
    #[cfg(feature = "survival")]
    {
        use crate::types::{EndpointLikelihood, ObsRecord};
        // The CMTs whose endpoint term we can differentiate exactly. A binary endpoint with
        // no `lp_program`, or a CTMM endpoint with no `generator_program`, is not one of
        // them; a parsed endpoint always has its program, so that is an opaque Rust closure.
        let served = |cmt: usize| -> bool {
            match model.endpoints.get(&cmt) {
                Some(EndpointLikelihood::Binary { lp_program, .. }) => lp_program.is_some(),
                Some(EndpointLikelihood::Count { param_program, .. }) => param_program.is_some(),
                #[cfg(feature = "markov")]
                Some(EndpointLikelihood::Ctmm { generator_program, .. }) => {
                    generator_program.is_some()
                }
                #[cfg(feature = "markov")]
                Some(EndpointLikelihood::Dtmm { prob_program, .. }) => prob_program.is_some(),
                _ => false,
            }
        };
        // A subject's records sit on one or two CMTs: decide each CMT once rather than per
        // record (two endpoint-map lookups each, on every gate of every inner evaluation).
        let mut last: Option<(bool, usize, bool)> = None;
        subject.obs_records.iter().all(|r| {
            let (event, cmt) = match r {
                ObsRecord::DiscreteState { cmt, .. } | ObsRecord::Count { cmt, .. } => {
                    (false, *cmt)
                }
                ObsRecord::Event { cmt, .. } => (true, *cmt),
            };
            if let Some((e, c, ok)) = last {
                if (e, c) == (event, cmt) {
                    return ok;
                }
            }
            // A record on a CMT no endpoint claims is read by no term (every likelihood path
            // selects its records by endpoint), so it cannot leave a term unserved — counting
            // it would send the whole subject to FD over data the objective ignores.
            let ok = if !model.endpoints.contains_key(&cmt) {
                true
            } else if event {
                crate::survival::tte_record_served(model, subject, cmt)
            } else {
                served(cmt)
            };
            last = Some((event, cmt, ok));
            ok
        })
    }
    #[cfg(not(feature = "survival"))]
    {
        let _ = (model, subject);
        false
    }
}

/// Whether every non-Gaussian endpoint of `model` has an exact channel: a binary predictor
/// and a CTMM generator with dual-evaluable programs, and every TTE endpoint served
/// ([`crate::survival::tte_endpoint_served`]). Vacuously `true` with none.
pub fn non_gaussian_endpoints_served(model: &CompiledModel) -> bool {
    #[cfg(feature = "markov")]
    if model.has_ctmm() && !all_ctmm_endpoints_analytic(model) {
        return false;
    }
    #[cfg(feature = "survival")]
    {
        if model.has_discrete() && !crate::categorical::all_discrete_endpoints_analytic(model) {
            return false;
        }
        if model.has_tte()
            && !model.endpoints.iter().all(|(cmt, ep)| {
                !matches!(ep, crate::types::EndpointLikelihood::Tte { .. })
                    || crate::survival::tte_endpoint_served(model, *cmt)
            })
        {
            return false;
        }
    }
    let _ = model;
    true
}

/// Every `Ctmm` endpoint on the model carries a dual-evaluable generator program, i.e. its
/// transition term can be differentiated exactly rather than finite-differenced. The
/// model-level counterpart of [`non_gaussian_records_fully_analytic`]'s per-subject check.
#[cfg(feature = "markov")]
pub fn all_ctmm_endpoints_analytic(model: &CompiledModel) -> bool {
    use crate::types::EndpointLikelihood;
    model.endpoints.iter().all(|(_, ep)| match ep {
        EndpointLikelihood::Ctmm { generator_program, .. } => generator_program.as_ref().is_some(),
        _ => true,
    })
}

/// `FERX_NO_ANALYTIC_INNER=1` forces the FD inner gradient everywhere (A-B toggle).
/// Cached in a `OnceLock`: the value cannot change mid-run, and this is queried per
/// subject on every inner-loop entry (issue #438 review).
pub fn no_analytic_inner_forced() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("FERX_NO_ANALYTIC_INNER").map(|v| v == "1").unwrap_or(false))
}

pub fn analytic_inner_common_bail(model: &CompiledModel) -> bool {
    no_analytic_inner_forced() || GradientMethod::forced_fd(model)
    // An SDE model is served: its EKF process-noise variances are rows of the subject's
    // canonical recording beside the predictions, read by the kernel
    // (`exact_inner_gradient`, `exact_inner_hessian`) and the per-row assemblers alike.
    // LTBS is served analytically on the inner loop for every combination now — plain,
    // × `ExpressionScale` on the closed form (the η-quotient then the `ln f`,
    // `closed_form_finish`), × TV-cov (the event-driven walk applies the same transform
    // LAST), and × IOV (`record_analytic_events` applies `ltbs_log_g` last over the
    // stacked axes, #486). So there is no
    // `log_transform` clause here: the closed-form inner and outer IOV gradients now
    // differentiate the same `ln(f/s)`, closing the last inner/outer scope split.
    // Correlated residual error (`block_sigma`) is now served analytically by the
    // dense-R inner gradient (`dense_residual_inner_gradient`, #627), so it is no
    // longer a blanket bail. (An eta-dependent `ExpressionScale` is NOT a bail either.)
    // Custom residual-error magnitude (#484, `weight =`) is not a bail either, alone
    // or with a dense R: `residual_inner_obs` and the dense-R kernel both thread the
    // per-obs multiplier (#576/#627), which the η-gradient sees as a constant (#2209).
    // `iiv_on_ruv` needs no bail here: residual-η is served analytically on both loops for
    // every combination — closed-form and ODE, plain / IOV / M3-BLOQ including the triples
    // (#474/#4b/#4c/#591/#623/#677); the scaling and the censored/quantified `η_ruv` terms
    // live in the shared, provider-agnostic gradient.
}

/// Model-level half of `analytic_inner_grad_supported`: every gate that does
/// not depend on the subject. `build_info::gradient_method_inner` reports the
/// inner route off **this same** predicate, so the reported `gradient_method_inner`
/// cannot drift from what `find_ebe` actually runs (PR #381 review #9).
pub fn analytic_inner_grad_supported_model(model: &CompiledModel) -> bool {
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
    // dual-evaluable program. A parsed generator always has one (a drug-driven `Q(t)` and a
    // wide `(θ, η)` included); only an opaque Rust-closure endpoint lacks it, and then
    // *every* subject declines at the
    // survival-record guard in `analytic_inner_grad_supported`, so reporting "analytic" here
    // would be a pure misreport.
    //
    // `recorder_in_scope` below does **not** catch this (nor did the retired
    // `analytical_supported` it replaced), which is the trap: an
    // endpoint-only model (a `[markov_model]` with no `[structural_model]`) is *still* given
    // a default `pk_model` and a vacuous `tv_fn` (`model_parser.rs`, `tv_fn = Some(..)` for
    // any non-ODE model), and its `pk_indices` are empty so the per-slot check passes
    // trivially — so the closed-form scope check waves it straight through despite there
    // being no closed form at all.
    #[cfg(feature = "markov")]
    if model.has_ctmm() && !all_ctmm_endpoints_analytic(model) {
        return false;
    }
    // Binary (#2024): the same reasoning — a `[binary_model]` whose linear predictor has no
    // dual-evaluable program declines every subject carrying its records.
    #[cfg(feature = "survival")]
    if model.has_discrete() && !crate::categorical::all_discrete_endpoints_analytic(model) {
        return false;
    }
    // TTE (#1861): an endpoint with an exact η-channel (`survival::tte_endpoint_served`: an
    // analytic hazard with a dual-evaluable program, or an ODE-accumulated one) is reported
    // analytic; one without declines every subject carrying its records at the guard in
    // `analytic_inner_grad_supported`, and reporting "analytic" then would mislabel the fit.
    #[cfg(feature = "survival")]
    if model.has_tte()
        && !model.endpoints.iter().all(|(cmt, ep)| {
            !matches!(ep, crate::types::EndpointLikelihood::Tte { .. })
                || crate::survival::tte_endpoint_served(model, *cmt)
        })
    {
        return false;
    }
    // The unified recorder's scope, or an endpoint-only model (no structural model, no
    // compartments), whose inner gradient is the prior plus the exactly-served endpoint term.
    crate::sens::ode_provider::adjoint::recorder_in_scope(model)
        || (model.ode_spec.is_none()
            && !model.is_algebraic()
            && model.pk_indices.is_empty()
            && model.has_non_gaussian())
}

/// Whether the inner (per-subject EBE) loop reports an **analytic** gradient at the model
/// level: [`analytic_inner_grad_supported_model`], the unified recorder's scope behind the
/// shared escape hatches. **Single source of truth** for `build_info::gradient_method_inner`
/// and `fd_fallback_warning`'s whole-population-FD guard, so the persisted label and the
/// reconciling warning cannot drift. Best-case and model-level: a per-subject FD fallback is
/// reported by `fd_fallback_warning`.
pub fn inner_reports_analytic_model(model: &CompiledModel) -> bool {
    analytic_inner_grad_supported_model(model)
}
