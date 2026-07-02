pub mod absorption;
pub mod analytical_absorption;
pub mod event_driven;
pub mod ode_template;
pub mod one_compartment;
pub mod three_compartment;
pub mod two_compartment;

use crate::types::{CompiledModel, DoseEvent, PkModel, PkParams, ScalingSpec, Subject};

pub use one_compartment::*;
pub use three_compartment::*;
pub use two_compartment::*;

#[inline]
fn model_uses_time_builtin(model: &CompiledModel) -> bool {
    crate::parser::model_parser::compiled_model_uses_time_builtin(model)
}

#[inline]
fn pk_params_at_time(
    model: &CompiledModel,
    theta: &[f64],
    eta: &[f64],
    covariates: &std::collections::HashMap<String, f64>,
    time: f64,
) -> PkParams {
    // `pk_param_fn` sets the model-time thread-local from `time` itself, so the
    // `TIME` built-in resolves to this event time without a separate wrap.
    (model.pk_param_fn)(theta, eta, covariates, time)
}

/// Divide each prediction in-place by the scale derived from
/// `model.scaling`. The convention is **divisive** so that
/// `[scaling] obs_scale = 1000` maps mg/L → mg/mL (i.e. the user's number
/// reads as "raw / scale").
///
/// `ScalingSpec::None` is a no-op so every existing call path keeps its
/// historical behaviour. For `ExpressionScale`, a subject-static
/// `pk_param_fn` evaluation is performed so the scale expression can
/// reference individual parameters (Phase 1 limitation: subject-static
/// only — TV-cov-aware expression scales are a Phase 1.5 follow-up).
///
/// Form C (ODE `y = <expr>`) does *not* go through here — it replaces the
/// per-observation state readout inside the ODE timeline loop via
/// `OdeSpec::output_fn`, so the prediction returned to this function is
/// already in observation units.
#[inline]
pub fn apply_scaling(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    // Fast path: no scaling skips both the pk evaluation and the per-obs
    // array allocation. Keeps the hot loop allocation-free for the
    // overwhelming majority of models.
    if matches!(model.scaling, ScalingSpec::None) {
        return;
    }
    // Materialise per-observation scales via the shared
    // `ScalingSpec::build_obs_scale_array` helper so the FD path here
    // and the AD path in `inner_optimizer.rs` see identical semantics.
    // Any scaling change is felt by both paths automatically.
    //
    // Only evaluate pk when at least one ExpressionScale closure (top
    // level or nested inside PerCmt) actually needs it. ScalarScale and
    // PerCmt-with-scalar-entries skip the pk_param_fn call, which can
    // be expensive on models with parsed-expression indiv params or NN
    // forward passes. (Caught by Copilot review on PR #85.)
    let pk_owned;
    let pk_ref: &PkParams = if model.scaling.needs_pk_eval() {
        // Subject-static scale snapshot (the divisive scale is applied to the
        // whole prediction vector): typical values at t=0.
        pk_owned = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        &pk_owned
    } else {
        static DEFAULT_PK: PkParams = PkParams {
            values: [0.0; crate::types::MAX_PK_PARAMS],
        };
        &DEFAULT_PK
    };
    let scales = model.scaling.build_obs_scale_array(
        theta,
        eta,
        &subject.covariates,
        pk_ref,
        &subject.obs_cmts,
    );
    for (i, pred) in preds.iter_mut().enumerate() {
        // build_obs_scale_array already encodes invalid scales as NaN,
        // so this is purely a divide (no extra positivity check needed).
        let s = scales.get(i).copied().unwrap_or(f64::NAN);
        if s.is_finite() && s > 0.0 {
            *pred /= s;
        } else {
            *pred = f64::NAN;
        }
    }
}

/// Apply an analytic Form C output readout (`[scaling] y = <expr>`, issue #650),
/// **replacing** the built-in central-compartment concentration in `preds` with
/// the readout output, evaluated per observation.
///
/// On entry `preds[i]` is the central concentration at observation `i` (already
/// init-aware via [`add_analytical_init`]). The readout sees compartment
/// **amounts**: the central amount is `concentration × V` (so a `central / V`
/// readout recovers the concentration exactly, and an additive binding term
/// layers on top); the oral depot amount — only reconstructed when the readout's
/// state layout includes it — comes from [`analytical_state_at_times`] (slot 0).
/// Individual parameters resolve from a `pk_param_fn` snapshot; θ/η are threaded
/// (already desugared into `__ferx_ro_*` indiv params at parse for the analytic
/// sensitivity path), and covariates use the **per-observation** snapshot
/// (`subject.obs_cov(i)`) so a per-row flag like `FREE` selects the right branch.
/// On a time-varying-covariate subject (or a `TIME`-builtin model) the readout's
/// individual parameters are re-evaluated **per observation** — matching NONMEM's
/// per-record `$ERROR` and the per-observation snapshot the analytic sensitivity
/// provider differentiates (`run_obs_tvcov`), so the prediction and its gradient
/// linearise about the same point. Otherwise one subject-static snapshot drives
/// every row.
///
/// No-op unless `model.analytic_readout` is set (the common built-in-readout
/// case). Called at the single post-prediction insertion point, in place of the
/// divisive `apply_scaling` (which is `ScalingSpec::None` under Form C).
pub fn apply_analytic_readout(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    let Some(ar) = &model.analytic_readout else {
        return;
    };
    let n_states = ar.state_names.len();
    if n_states == 0 {
        return;
    }
    // Canonical order is `["central"]` (IV) or `["depot", "central"]` (oral), so
    // the central amount lands in the last slot and the depot (if any) in slot 0.
    let central_slot = n_states - 1;
    let need_depot = n_states > 1 && ar.state_names[0] == "depot";

    // Re-evaluate the readout's individual parameters per observation whenever a
    // covariate (or the `TIME` builtin) can move them between rows; otherwise a
    // single subject-static snapshot suffices.
    let per_obs = subject.has_tv_covariates()
        || crate::parser::model_parser::compiled_model_uses_time_builtin(model);
    let static_pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);

    // Depot amounts (oral, only when the readout references the depot). Reuse the
    // superposition state reconstruction; slot 0 carries the depot *amount*.
    // Superposition is invalid across EVID=3/4 resets, so a depot-referencing
    // readout on a reset subject is rejected up front by `fit`/`predict`
    // (`reject_depot_readout_with_resets`); we never reach the debug-asserted
    // `analytical_state_at_times` with a reset subject here.
    let depot_amts: Option<Vec<f64>> = if need_depot {
        let states =
            analytical_state_at_times(model.pk_model, subject, &static_pk, &subject.obs_times);
        Some(
            states
                .iter()
                .map(|s| s.first().copied().unwrap_or(0.0))
                .collect(),
        )
    } else {
        None
    };
    let mut state = vec![0.0_f64; n_states];
    for (i, pred) in preds.iter_mut().enumerate() {
        let obs_pk;
        let pk_i: &PkParams = if per_obs {
            obs_pk = (model.pk_param_fn)(
                theta,
                eta,
                subject.obs_cov(i),
                subject.obs_times.get(i).copied().unwrap_or(0.0),
            );
            &obs_pk
        } else {
            &static_pk
        };
        // Central AMOUNT = concentration × V (concentration already carries the
        // init impulse and per-event/SS dynamics).
        state[central_slot] = *pred * pk_i.v();
        if let Some(d) = &depot_amts {
            state[0] = d.get(i).copied().unwrap_or(0.0);
        }
        let cov = subject.obs_cov(i);
        let obs_cmt = subject.obs_cmts.get(i).copied().unwrap_or(0);
        *pred = ar
            .readout
            .eval(&state, &pk_i.values, theta, eta, cov, obs_cmt);
    }
}

/// Layer analytical initial-compartment amounts onto `preds` (issue #521).
///
/// For an analytical (closed-form) PK model the disposition is linear, so a
/// non-zero initial amount `A₀` in compartment `c` at t=0 is exactly the
/// impulse response of a bioavailability-bypassed bolus of `A₀` into `c`,
/// superimposed on the dose-driven prediction. `A₀` is evaluated once per
/// subject (it may depend on θ/η/covariates/individual parameters, e.g.
/// `init(central) = CONC0 * V`) and its decaying concentration contribution is
/// added at each observation time.
///
/// Called **before** `[scaling]` and the log-transform so scaling divides the
/// dose+init total — matching the ODE path, where `init(...)` seeds the state
/// that the same readout/scaling later sees.
///
/// No-op for ODE models (state seeded via `ode_spec.init_fn`) and for the
/// common case of no `[initial_conditions]` block. `preds` is parallel to
/// `subject.obs_times`.
pub fn add_analytical_init(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    if model.analytical_init.is_empty() || model.ode_spec.is_some() {
        return;
    }
    // Non-IOV path: one subject-static PK snapshot drives both the baseline
    // amount and its decay kernel. The baseline is seeded at the record start
    // (t=0), so the `TIME` built-in resolves there.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    add_analytical_init_with(model, subject, theta, eta, &pk, None, preds);
}

/// Core of [`add_analytical_init`], parameterised by the PK snapshot(s) used.
///
/// `amount_pk` evaluates the baseline amount `A₀` from the init expression
/// (it references BSV individual parameters, so callers pass the BSV-only PK
/// snapshot). `obs_decay_pk`, when `Some`, supplies the per-observation PK
/// params that drive the *decay kernel* (CL/V/KA over time) — used by the IOV
/// path so the baseline decays at each observation's occasion-specific
/// clearance instead of the population/BSV rate (issue #521 review). When
/// `None`, `amount_pk` drives the decay too (the non-IOV case).
///
/// A system reset (EVID=3/4) zeros every compartment, including the residual
/// baseline; nothing re-deposits the t=0 init afterwards. So the baseline only
/// contributes to observations strictly before the first reset — observations
/// at or after it see none of it. `obs_times` is the internal monotonic clock
/// (shared with `reset_times`), so the comparison is on the same timeline.
fn add_analytical_init_with(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    amount_pk: &PkParams,
    obs_decay_pk: Option<&[PkParams]>,
    preds: &mut [f64],
) {
    let first_reset = subject
        .reset_times
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    for init in &model.analytical_init {
        let a0 = (init.amount_fn)(theta, eta, &subject.covariates, amount_pk);
        if a0 == 0.0 || !a0.is_finite() {
            continue;
        }
        for (i, &t) in subject.obs_times.iter().enumerate() {
            // The initial amount is laid down at the subject's time origin
            // (t=0); a pre-origin observation sees none of it, and a reset
            // wipes it for every observation at/after the reset time.
            if t < 0.0 || t >= first_reset {
                continue;
            }
            let decay_pk = match obs_decay_pk {
                Some(ps) => ps.get(i).unwrap_or(amount_pk),
                None => amount_pk,
            };
            if let Some(pred) = preds.get_mut(i) {
                *pred += analytical_init_concentration(model.pk_model, init.cmt, a0, t, decay_pk);
            }
        }
    }
}

/// Closed-form central-compartment concentration at elapsed time `t` from an
/// initial amount `a0` deposited in compartment `cmt` (1-based) at t=0, for an
/// analytical model. Reuses the generic single-dose closed forms with
/// `T = f64` and `F = 1` (an initial condition is not an absorbed dose, so
/// bioavailability does not apply).
///
/// Supported compartments are the **central** compartment (IV-bolus impulse)
/// and, for oral models, the **depot** (first-order absorption of the
/// pre-loaded amount). Peripheral-compartment initial amounts need the
/// cross-compartment Green's function and are rejected at parse time, so this
/// returns 0 for any other `cmt` defensively.
fn analytical_init_concentration(
    pk_model: PkModel,
    cmt: usize,
    a0: f64,
    t: f64,
    p: &PkParams,
) -> f64 {
    // Delegate to the generic form at `T = f64`; the single formula dispatch
    // lives there so the Dual2/Dual1 sensitivity path (#524) cannot drift from
    // the f64 prediction.
    analytical_init_concentration_g::<f64>(
        pk_model,
        cmt,
        a0,
        t,
        p.cl(),
        p.v(),
        p.q(),
        p.v2(),
        p.ka(),
        p.q3(),
        p.v3(),
    )
}

/// Generic form of [`analytical_init_concentration`] over [`PkNum`]: evaluating
/// in `f64` gives the prediction; evaluating in `Dual2<N>`/`Dual1<N>` (with the
/// amount `a0` and the PK params seeded as duals) gives the exact `∂C/∂A₀`,
/// `∂C/∂(CL,V,…)` the analytic FOCE/FOCEI provider needs (#524). The amount is a
/// pre-loaded compartment quantity, so `F = 1` everywhere — an initial condition
/// is not an absorbed dose. Returns 0 for unsupported compartments defensively
/// (peripheral inits are rejected at parse time).
#[allow(clippy::too_many_arguments)]
pub(crate) fn analytical_init_concentration_g<T: crate::sens::num::PkNum>(
    pk_model: PkModel,
    cmt: usize,
    a0: T,
    t: T,
    cl: T,
    v: T,
    q: T,
    v2: T,
    ka: T,
    q3: T,
    v3: T,
) -> T {
    use crate::sens::one_cpt::{one_cpt_iv_bolus_amt_g, one_cpt_oral_amt_g};
    use crate::sens::three_cpt::{three_cpt_iv_bolus_amt_g, three_cpt_oral_amt_g};
    use crate::sens::two_cpt::{two_cpt_iv_bolus_amt_g, two_cpt_oral_amt_g};

    let one = T::from_f64(1.0);
    // Central is cmt 2 for oral models (depot is cmt 1), cmt 1 for IV models.
    let central = if pk_model.is_oral() { 2 } else { 1 };

    if cmt == central {
        // IV-bolus impulse directly into the central compartment.
        match pk_model {
            PkModel::OneCptIv | PkModel::OneCptOral | PkModel::OneCptTransit => {
                one_cpt_iv_bolus_amt_g(a0, t, cl, v)
            }
            PkModel::TwoCptIv | PkModel::TwoCptOral => two_cpt_iv_bolus_amt_g(a0, t, cl, v, q, v2),
            PkModel::ThreeCptIv | PkModel::ThreeCptOral => {
                three_cpt_iv_bolus_amt_g(a0, t, cl, v, q, v2, q3, v3)
            }
        }
    } else if pk_model.is_oral() && cmt == 1 {
        // Pre-loaded depot amount: first-order absorption into central, F=1.
        match pk_model {
            PkModel::OneCptOral => one_cpt_oral_amt_g(a0, t, cl, v, ka, one),
            PkModel::TwoCptOral => two_cpt_oral_amt_g(a0, t, cl, v, q, v2, ka, one),
            PkModel::ThreeCptOral => three_cpt_oral_amt_g(a0, t, cl, v, q, v2, q3, v3, ka, one),
            _ => T::from_f64(0.0),
        }
    } else {
        T::from_f64(0.0)
    }
}

/// Validate that every observed CMT in the population has an entry in
/// the model's `ScalingSpec::PerCmt` map (or any nested `OdeReadout::PerCmt`
/// readout for ODE models). Called once at the top of `fit()` so missing
/// entries surface as a clear parse-time-style error rather than silent
/// NaN predictions at runtime.
///
/// Returns `Ok(())` for non-PerCmt scaling. The error message names
/// every missing CMT explicitly.
pub fn validate_per_cmt_scaling(model: &CompiledModel, subjects: &[Subject]) -> Result<(), String> {
    use std::collections::BTreeSet;

    // Collect every CMT that has at least one observation in the population.
    let mut observed_cmts: BTreeSet<usize> = BTreeSet::new();
    for subj in subjects {
        for &cmt in &subj.obs_cmts {
            observed_cmts.insert(cmt);
        }
    }

    // Check ScalingSpec::PerCmt coverage (Forms A/B).
    if let ScalingSpec::PerCmt(map) = &model.scaling {
        let missing: Vec<usize> = observed_cmts
            .iter()
            .copied()
            .filter(|c| !map.contains_key(c))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "[scaling]: per-CMT scaling is missing entries for observed CMTs {:?}. \
                 Every observed CMT must have an `obs_scale[CMT=N]` (or `y[CMT=N]` for ODE) entry.",
                missing
            ));
        }
    }

    // Check OdeReadout::PerCmt coverage (Form C per-CMT).
    if let Some(ref ode) = model.ode_spec {
        if let crate::ode::OdeReadout::PerCmt(map) = &ode.readout {
            let missing: Vec<usize> = observed_cmts
                .iter()
                .copied()
                .filter(|c| !map.contains_key(c))
                .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "[scaling]: per-CMT `y[CMT=N]` Form C is missing entries for observed CMTs {:?}. \
                     Every observed CMT must have a `y[CMT=N]` entry.",
                    missing
                ));
            }
        }
    }

    Ok(())
}

/// Per-event PK parameter snapshots for one subject.
///
/// When the subject has no time-varying covariates, all entries in `dose`
/// and `obs` are equal (the single subject-static evaluation). The fast
/// superposition path detects this case via `subject.has_tv_covariates()`
/// and evaluates `pk_param_fn` only once instead of materialising the
/// vectors — see [`compute_event_pk_params`] for details.
///
/// # Reusing across calls
///
/// Hot loops (SAEM Metropolis-Hastings, BFGS gradient/objective) call
/// `compute_event_pk_params` once per evaluation. To avoid the
/// allocate-and-discard cycle, allocate one buffer per subject (
/// [`Self::with_capacity_for`]) and refill it with
/// [`compute_event_pk_params_into`]. The Vecs are reused — no new
/// allocations as long as `subject` doesn't change.
#[derive(Debug, Clone, Default)]
pub struct EventPkParams {
    /// PK params at each dose event time, parallel to `subject.doses`.
    pub dose: Vec<PkParams>,
    /// PK params at each observation event time, parallel to `subject.obs_times`.
    pub obs: Vec<PkParams>,
    /// PK params at each EVID=2 event time, parallel to
    /// `subject.pk_only_times`. Empty when the subject has no
    /// pk-only events (typical for non-TV-cov data).
    pub pk_only: Vec<PkParams>,
}

impl EventPkParams {
    /// Allocate empty `Vec`s sized to fit `subject`'s event timeline.
    /// The capacity is what matters for reuse — the subsequent
    /// [`compute_event_pk_params_into`] call resizes the `Vec`s to
    /// the exact lengths and overwrites the contents.
    pub fn with_capacity_for(subject: &Subject) -> Self {
        Self {
            dose: Vec::with_capacity(subject.doses.len()),
            obs: Vec::with_capacity(subject.obs_times.len()),
            pk_only: Vec::with_capacity(subject.pk_only_times.len()),
        }
    }
}

/// Materialise per-event PK parameters by evaluating `model.pk_param_fn`
/// at the LOCF covariate snapshot stored on each event.
///
/// When the subject has no time-varying covariates, this evaluates
/// `pk_param_fn` exactly once and clones the result into every slot — the
/// downstream PK solver still sees a uniform per-event interface, while
/// the cost stays at O(1) covariate evaluation.
///
/// Allocates fresh `Vec`s each call. For hot loops use
/// [`compute_event_pk_params_into`] with a reused [`EventPkParams`]
/// buffer.
pub fn compute_event_pk_params(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> EventPkParams {
    let mut out = EventPkParams::with_capacity_for(subject);
    compute_event_pk_params_into(model, subject, theta, eta, &mut out);
    out
}

/// Same as [`compute_event_pk_params`] but writes into a caller-owned
/// buffer. Used by SAEM's MH loop and the FOCE objective closure where
/// the buffer is allocated once per subject and reused across many
/// `eta` evaluations — cuts per-call allocator pressure to zero on the
/// TV-cov path.
///
/// Resizes `out`'s `Vec`s to the exact lengths required by `subject`,
/// then overwrites the contents. If the existing capacity already
/// fits, no allocation happens.
pub fn compute_event_pk_params_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    out: &mut EventPkParams,
) {
    out.dose.clear();
    out.obs.clear();
    out.pk_only.clear();

    if subject.has_tv_covariates() || model_uses_time_builtin(model) {
        for k in 0..subject.doses.len() {
            out.dose.push(pk_params_at_time(
                model,
                theta,
                eta,
                subject.dose_cov(k),
                subject.doses[k].time,
            ));
        }
        for j in 0..subject.obs_times.len() {
            out.obs.push(pk_params_at_time(
                model,
                theta,
                eta,
                subject.obs_cov(j),
                subject.obs_times[j],
            ));
        }
        for m in 0..subject.pk_only_times.len() {
            out.pk_only.push(pk_params_at_time(
                model,
                theta,
                eta,
                subject.pk_only_cov(m),
                subject.pk_only_times[m],
            ));
        }
    } else {
        // Reached only when the model uses neither TV covariates nor the `TIME`
        // built-in, so a single snapshot (t=0) is exact for every event.
        let p = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        // pk_only stays empty — see EventPkParams docstring.
        for _ in 0..subject.doses.len() {
            out.dose.push(p);
        }
        for _ in 0..subject.obs_times.len() {
            out.obs.push(p);
        }
    }
}

/// IOV predictions with proper per-dose occasion accounting (issue #104).
///
/// Builds per-event PK parameters carrying **each event's occasion kappa** and
/// runs the state-propagating event-driven solver once. Because that solver
/// uses the end-of-interval (current-record) parameter convention, a dose's
/// carryover into a later occasion is eliminated with the *later* occasion's
/// clearance — matching NONMEM's continuous integration with parameters that
/// switch at occasion boundaries.
///
/// This supersedes the "Option A" per-occasion superposition, which scored
/// every occasion against the subject's whole dose history using that
/// occasion's parameters, biasing the likelihood on designs with cross-occasion
/// carryover (no washout between occasions). See the `individual_nll_iov`
/// history and issue #104.
///
/// `eta_bsv` is the BSV eta (length `n_eta`); `kappas[k]` is the kappa vector
/// for the k-th occasion group in `iov_occasion_groups` order, including dose-only
/// occasions. EVID=2 rows carry no occasion label and use zero kappa.
///
/// Every `PkModel` variant has event-driven analytical support and every ODE
/// model takes the ODE path, so the dispatch below is total; an unsupported
/// model is a wiring bug and fails loud rather than silently mis-predicting.
///
/// **Occasion-dependent PK dynamics are exact**: per-event params carry the
/// occasion κ, so CL/V/KA switch correctly across occasions. `[scaling]` is also
/// applied per occasion (see the call site). One narrow case is handled at parse
/// time rather than here: a Form C ODE output expression (`y = <expr>`) that
/// references `KAPPA_*` directly would be evaluated per-observation with a single
/// eta (κ=0), so the parser rejects it for IOV models (issue #107) — reference
/// the occasion-dependent structural parameter (e.g. CL) instead. The PK state
/// is always occasion-correct (it flows through the per-event params).
/// Positivity floor applied before log-transforming a prediction under LTBS, so
/// a prediction the optimizer drives toward zero yields a large-but-finite log
/// value instead of `-inf`/`NaN`.
pub(crate) const LTBS_FLOOR: f64 = 1e-12;

/// Log-transform predictions in place when the model uses log-transform-both-sides
/// (LTBS); a no-op otherwise. Applied at every prediction sink so the effective
/// prediction is `log(f)` everywhere — FOCE linearization, residuals, IWRES/CWRES,
/// IPRED/PRED, and simulated DV all end up on the log scale (matching NONMEM's
/// `Y = LOG(F) + EPS`). The mirror transform in the analytic-sensitivity paths
/// (`sens::provider`, `sens::ode_provider`) must stay in sync.
#[inline]
pub(crate) fn apply_log_transform(model: &CompiledModel, preds: &mut [f64]) {
    if model.log_transform {
        for p in preds.iter_mut() {
            *p = ltbs_log_g(*p);
        }
    }
}

/// The LTBS log transform, generic over `T: PkNum`, so the production `f64`
/// predictor and the analytic ODE/analytical sensitivity dual walks apply the
/// *same* floor-then-log (`p.max(LTBS_FLOOR).ln()` for `T = f64`, byte-identical to
/// the original) instead of each re-deriving it — the consistency the
/// `apply_log_transform` doc demands (#438 review #4 / #451). `guard_floor` also
/// floors `NaN` to the floor (matching `f64::max`), so a transient `NaN` maps to
/// `ln(LTBS_FLOOR)` on every path; the dual callers keep their own pre-check when a
/// `NaN` readout must instead stay visible (e.g. a per-CMT miss).
#[inline]
pub(crate) fn ltbs_log_g<T: crate::sens::num::PkNum>(p: T) -> T {
    p.guard_floor(LTBS_FLOOR).ln()
}

pub fn predict_iov(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_bsv: &[f64],
    kappas: &[Vec<f64>],
) -> Vec<f64> {
    use std::collections::HashMap;
    let n_kappa = model.n_kappa;

    // occasion id -> kappa-group index (iov_occasion_groups order).
    let occ_groups = crate::stats::likelihood::iov_occasion_groups(subject);
    let mut occ_to_k: HashMap<u32, usize> = HashMap::with_capacity(occ_groups.len());
    for (k, (occ_id, _)) in occ_groups.iter().enumerate() {
        occ_to_k.insert(*occ_id, k);
    }
    let combined_for = |occ_id: u32| -> Vec<f64> {
        let mut c = Vec::with_capacity(eta_bsv.len() + n_kappa);
        c.extend_from_slice(eta_bsv);
        match occ_to_k.get(&occ_id) {
            Some(&k) if k < kappas.len() => c.extend_from_slice(&kappas[k]),
            _ => c.extend(std::iter::repeat(0.0).take(n_kappa)),
        }
        c
    };

    let dose_params: Vec<PkParams> = (0..subject.doses.len())
        .map(|d| {
            let occ = subject.dose_occasions.get(d).copied().unwrap_or(0);
            pk_params_at_time(
                model,
                theta,
                &combined_for(occ),
                subject.dose_cov(d),
                subject.doses[d].time,
            )
        })
        .collect();
    let obs_params: Vec<PkParams> = (0..subject.obs_times.len())
        .map(|j| {
            let occ = subject.occasions.get(j).copied().unwrap_or(0);
            pk_params_at_time(
                model,
                theta,
                &combined_for(occ),
                subject.obs_cov(j),
                subject.obs_times[j],
            )
        })
        .collect();
    // EVID=2 rows carry no occasion label → BSV eta with zero kappa.
    let pk_only_combined = combined_for(u32::MAX);
    let pk_only_params: Vec<PkParams> = (0..subject.pk_only_times.len())
        .map(|m| {
            pk_params_at_time(
                model,
                theta,
                &pk_only_combined,
                subject.pk_only_cov(m),
                subject.pk_only_times[m],
            )
        })
        .collect();

    let mut preds = if let Some(ref ode) = model.ode_spec {
        crate::ode::ode_predictions_event_driven(
            ode,
            subject,
            theta,
            &pk_only_combined,
            &dose_params,
            &obs_params,
            &pk_only_params,
        )
    } else if event_driven::supports_event_driven(model.pk_model) {
        // Resolve modeled-`RATE` doses (#324/#394) using each dose's per-occasion
        // PK snapshot before the analytical event-driven walker — the IOV analogue
        // of the resolve step in `compute_predictions_with_tv_into_with_schedule`.
        // (The ODE arm above resolves internally via `ode.dose_attr_map`; this arm
        // reads the analytical model's `dose_attr_map`.) Borrowed no-op when every
        // dose is already `Fixed`.
        let resolved =
            crate::ode::resolve_subject_doses_with(subject, model.active_dose_attr_map(), |k| {
                &dose_params[k].values
            });
        event_driven::event_driven_predictions(
            model.pk_model,
            &resolved,
            &dose_params,
            &obs_params,
            &pk_only_params,
        )
    } else {
        // Unreachable today: every `PkModel` variant has event-driven analytical
        // support and ODE models took the branch above. A new analytical variant
        // not wired into `supports_event_driven` lands here and fails loud —
        // far better than silently mis-predicting. (The legacy Option-A
        // superposition that used to live here dropped bioavailability F on
        // IV/infusion doses; see #327.)
        unreachable!(
            "predict_iov: model {:?} has neither an ODE spec nor event-driven \
             analytical support; wire it into `supports_event_driven`",
            model.pk_model
        )
    };

    // Analytical initial-compartment amounts (#521), layered on before scaling —
    // same insertion point as the non-IOV `compute_predictions_with_tv_*` path.
    // Needed here too because IMP (and any IOV-aware likelihood) predicts through
    // `predict_iov`; without it the baseline subjects would be mispredicted and
    // their importance weights collapse. No-op for ODE models and init-free
    // models. Init expressions reference BSV parameters, so `eta_bsv` drives the
    // baseline *amount*; the *decay* kernel uses the per-occasion `obs_params`
    // (which carry each observation's occasion kappa + TV covariates), so an
    // IOV-on-disposition baseline decays at the occasion-correct rate rather than
    // the kappa-less BSV rate (issue #521 review).
    if !model.analytical_init.is_empty() && model.ode_spec.is_none() {
        // Baseline amount is seeded at the record start (internal t=0 under the
        // subject-relative time origin), so the `TIME` built-in resolves there.
        let amount_pk = (model.pk_param_fn)(theta, eta_bsv, &subject.covariates, 0.0);
        add_analytical_init_with(
            model,
            subject,
            theta,
            eta_bsv,
            &amount_pk,
            Some(&obs_params),
            &mut preds,
        );
    }

    // `[scaling]` post-multiply, applied **per occasion** so a κ-dependent scale
    // (or a scale referencing a κ-dependent individual parameter) uses that
    // occasion's κ — matching the per-occasion prediction. `apply_scaling`
    // short-circuits on `ScalingSpec::None`, so the common case stays a cheap
    // no-op. (Form C ODE output `y = <expr>` is applied inside the ODE solver,
    // not here; see the note below for the κ=0 limitation there.)
    if !matches!(model.scaling, ScalingSpec::None) {
        let raw = preds.clone();
        for (occ_id, obs_indices) in &occ_groups {
            let combined = combined_for(*occ_id);
            let mut scaled = raw.clone();
            apply_scaling(model, subject, theta, &combined, &mut scaled);
            for &j in obs_indices {
                preds[j] = scaled[j];
            }
        }
    }
    // Analytic Form C readout (#650) under IOV: applied once with the BSV eta
    // (the readout is BSV-only — a κ reference is rejected at parse). The central
    // concentration already carries the per-occasion dynamics; the readout maps
    // it (amount = conc × V) into the observed output. No-op unless set.
    apply_analytic_readout(model, subject, theta, eta_bsv, &mut preds);
    // LTBS log-wrap. The IOV dispatch above is total (ODE or event-driven), so
    // this is the single log-wrap point for the IOV path — predictions are
    // logged exactly once.
    apply_log_transform(model, &mut preds);

    // FREM override AFTER scaling and log-transform: covariate pseudo-observations
    // are predicted as theta + eta (raw additive), regardless of the PK error model.
    apply_frem_prediction_override(model, subject, theta, eta_bsv, &mut preds);
    preds
}

/// Predict concentration at a given time for a subject, summing contributions
/// from all prior doses (superposition principle).
///
/// `pk_params.lagtime()` shifts the effective start of every dose (bolus,
/// infusion, and oral) by that amount. For infusions the duration is
/// preserved — only the start (and therefore end) of the window shifts.
///
/// # Steady state under lagtime
///
/// For a steady-state dose (`dose.ss`, `dose.ii > 0`) the periodic pulse
/// train extends infinitely into the past, so an observation between the
/// dose *record* time and the lagged dose arrival
/// (`dose.time ≤ t < dose.time + lagtime`) still sees the tail of the
/// *previous* interval — the most recent pulse landed at `t_eff − II`. We
/// recover it by wrapping the (negative) elapsed time into `[0, II)` and
/// evaluating the SS closed form there. This matches NONMEM `ALAG1` +
/// `SS=1` (verified against NONMEM 7.5 to 5 significant figures). Without
/// the wrap these early samples would read 0, which is wrong at steady
/// state.
pub fn predict_concentration(
    pk_model: PkModel,
    doses: &[DoseEvent],
    t: f64,
    pk_params: &PkParams,
) -> f64 {
    let lagtime = pk_params.lagtime();
    let mut conc = 0.0;
    for dose in doses {
        let t_eff = dose.time + lagtime;
        if t_eff <= t {
            let tau = t - t_eff;
            conc += single_dose_concentration(pk_model, dose, tau, pk_params);
        } else if dose.ss && dose.ii > 0.0 && t >= dose.time {
            // Pre-arrival steady-state tail (see the doc comment). Only for
            // observations at/after the dose *record* time: SS=1 establishes
            // steady state *at* the record, so a record cannot contribute to
            // times before itself (an SS dose later in the timeline must not
            // leak into earlier observations). Wrap the negative elapsed time
            // `t - t_eff` up into `[0, II)` by adding whole intervals — one
            // suffices for a physical lagtime < II, but the ceil keeps it
            // correct for any value.
            let raw = t - t_eff;
            let n = (-raw / dose.ii).ceil();
            let tau = raw + n * dose.ii;
            if tau >= 0.0 {
                conc += single_dose_concentration(pk_model, dose, tau, pk_params);
            }
        }
    }
    conc.max(0.0)
}

/// External bioavailability multiplier for the analytical superposition closed
/// forms, applied as a post-multiply. Only an **IV bolus** uses it: its
/// `F`-agnostic closed form is linear in the dose, so `F` post-multiplies the
/// result. Every other route returns `1.0`:
/// * oral-depot bolus forms (`*_oral_f`) bake `F` in internally;
/// * infusions carry `F` in their `(rate, duration)` via
///   [`DoseEvent::with_bioavailable_infusion`] before the closed form runs, so
///   `F` is already applied (a rate-defined window cannot be expressed as a
///   post-multiply once `F` reshapes its duration; #419).
///
/// Matches the event-driven path and NONMEM's `F1` (#327, #419).
fn route_f_scale(pk_model: PkModel, infusion: bool, p: &PkParams) -> f64 {
    if infusion || pk_model.is_oral() {
        1.0
    } else {
        p.f_bio()
    }
}

/// Concentration contribution from a single dose at elapsed time tau.
///
/// For IV variants the bolus-vs-infusion closed form is chosen per dose
/// from `dose.is_infusion()` (RATE>0 ⇒ infusion). This lets a single
/// subject mix bolus and infusion doses without changing the model
/// declaration (issue #176). Oral routes always go through the oral
/// closed form regardless of RATE.
///
/// When `dose.ss` is true and `dose.ii > 0`, the SS closed-form variant
/// is dispatched. The malformed SS configurations the closed forms
/// don't handle — `ii <= 0` and `T_inf > ii` for infusion — fall through
/// to the single-dose formula and are flagged by the data-validation
/// warning in `api.rs`.
fn single_dose_concentration(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> f64 {
    let cl = p.cl();
    let v = p.v();
    let infusion = dose.is_infusion();
    let f_scale = route_f_scale(pk_model, infusion, p);

    // Bake bioavailability `F` into the infusion `(rate, duration)` so the
    // F-agnostic closed forms see the reshaped infusion (#419). A bolus is
    // unchanged here (its `F` rides `f_scale` for IV, or is baked into the
    // `*_oral_f` form for the depot). Shadowing `dose` routes every closed-form
    // call below through the reshaped copy.
    let dose = &dose.with_bioavailable_infusion(p.f_bio());

    let raw = if dose.ss && dose.ii > 0.0 {
        match pk_model {
            // Transit rejects SS doses at parse (#386); the periodic-sum SS closed
            // form is a follow-up, so this arm is unreachable for a valid model.
            PkModel::OneCptTransit => {
                unreachable!("one_cpt_transit does not support SS doses (rejected at parse)")
            }
            PkModel::OneCptIv => {
                if infusion {
                    one_cpt_infusion_ss(dose, tau, cl, v)
                } else {
                    one_cpt_iv_bolus_ss(dose, tau, cl, v)
                }
            }
            PkModel::OneCptOral => {
                // Infusions bypass the depot — use the IV infusion SS formula,
                // matching single_dose_states and the TwoCptOral/ThreeCptOral fix.
                if infusion {
                    one_cpt_infusion_ss(dose, tau, cl, v)
                } else {
                    one_cpt_oral_f_ss(dose, tau, cl, v, p.ka(), p.f_bio())
                }
            }
            PkModel::TwoCptIv => {
                if infusion {
                    two_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_iv_bolus_ss(dose, tau, cl, v, p.q(), p.v2())
                }
            }
            PkModel::TwoCptOral => {
                // Infusions bypass the depot and enter central directly —
                // use the IV infusion SS formula, matching single_dose_states.
                if infusion {
                    two_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_oral_f_ss(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio())
                }
            }
            PkModel::ThreeCptIv => {
                if infusion {
                    three_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_iv_bolus_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                }
            }
            PkModel::ThreeCptOral => {
                // Same infusion-bypasses-depot logic as TwoCptOral.
                if infusion {
                    three_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_oral_f_ss(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    )
                }
            }
        }
    } else {
        match pk_model {
            PkModel::OneCptIv => {
                if infusion {
                    one_cpt_infusion(dose, tau, cl, v)
                } else {
                    one_cpt_iv_bolus(dose, tau, cl, v)
                }
            }
            PkModel::OneCptOral => {
                // Infusions bypass the depot — use the IV infusion formula.
                if infusion {
                    one_cpt_infusion(dose, tau, cl, v)
                } else {
                    one_cpt_oral_f(dose, tau, cl, v, p.ka(), p.f_bio())
                }
            }
            PkModel::OneCptTransit => {
                // Transit rejects infusions at parse, so only the absorbed bolus exists.
                one_cpt_transit_f(dose, tau, cl, v, p.n_transit(), p.mtt(), p.f_bio())
            }
            PkModel::TwoCptIv => {
                if infusion {
                    two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2())
                }
            }
            PkModel::TwoCptOral => {
                // Infusions bypass the depot — use the IV formula, matching single_dose_states.
                if infusion {
                    two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_oral_f(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio())
                }
            }
            PkModel::ThreeCptIv => {
                if infusion {
                    three_cpt_infusion(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                }
            }
            PkModel::ThreeCptOral => {
                // Same infusion-bypasses-depot logic as TwoCptOral.
                if infusion {
                    three_cpt_infusion(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_oral_f(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    )
                }
            }
        }
    };

    f_scale * raw
}

// --- Compartment-state helpers for `compartment_states` in SubjectResult ---

/// Returns the full state vector contribution from one dose at elapsed time `tau`.
///
/// Layout per PkModel:
///   OneCptIv:     [C_central]
///   OneCptOral:   [A_depot, C_central]
///   TwoCptIv:     [C_central, C_periph]
///   TwoCptOral:   [A_depot, C_central, C_periph]
///   ThreeCptIv:   [C_central, C_periph1, C_periph2]
///   ThreeCptOral: [A_depot, C_central, C_periph1, C_periph2]
///
/// Scaling is NOT applied — these are the raw analytical PK states.
fn single_dose_states(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> Vec<f64> {
    let cl = p.cl();
    let v = p.v();
    let infusion = dose.is_infusion();

    // Bioavailability handling mirrors `single_dose_concentration` (#419):
    // * IV bolus - states are linear in the dose, so `F` post-multiplies the raw
    //   state vector once at the end via `f_scale`;
    // * oral-depot - the depot/central closed forms bake `F` in (`f_scale == 1.0`);
    // * infusion - `F` is baked into the `(rate, duration)` below, so `f_scale`
    //   is `1.0` and the reshaped window is what the closed forms integrate.
    let f_scale = route_f_scale(pk_model, infusion, p);
    let dose = &dose.with_bioavailable_infusion(p.f_bio());

    // SS early-exit: mirrors single_dose_concentration's top-level guard.
    //
    // Peripheral and depot helpers (two_cpt_iv_peripheral, two_cpt_oral_peripheral,
    // three_cpt_iv_peripherals, three_cpt_oral_peripherals, one_cpt_oral_depot) all
    // handle SS internally via their own `dose.ss && dose.ii > 0.0` checks, so they
    // are called with the same arguments regardless of SS — the correct SS value is
    // returned automatically. Only the *central* concentration functions need explicit
    // SS dispatch here (they lack the internal guard in their non-SS variants).
    let mut state = if dose.ss && dose.ii > 0.0 {
        match pk_model {
            // Transit rejects SS doses at parse (#386) — unreachable for a valid model.
            PkModel::OneCptTransit => {
                unreachable!("one_cpt_transit does not support SS doses (rejected at parse)")
            }
            PkModel::OneCptIv => {
                let c = if infusion {
                    one_cpt_infusion_ss(dose, tau, cl, v)
                } else {
                    one_cpt_iv_bolus_ss(dose, tau, cl, v)
                };
                vec![c]
            }
            PkModel::OneCptOral => {
                // Infusions bypass the depot — treat as 1-cpt IV SS infusion,
                // consistent with single_dose_concentration and TwoCptOral/ThreeCptOral.
                if infusion {
                    let c = one_cpt_infusion_ss(dose, tau, cl, v);
                    vec![0.0, c]
                } else {
                    // one_cpt_oral_depot handles SS internally.
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central = one_cpt_oral_f_ss(dose, tau, cl, v, p.ka(), p.f_bio());
                    vec![depot, central]
                }
            }
            PkModel::TwoCptIv => {
                let central = if infusion {
                    two_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_iv_bolus_ss(dose, tau, cl, v, p.q(), p.v2())
                };
                // two_cpt_iv_peripheral handles SS internally.
                let periph = two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2());
                vec![central, periph]
            }
            PkModel::TwoCptOral => {
                if infusion {
                    // Infusions bypass depot; treat as 2-cpt IV SS infusion.
                    let c = two_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2());
                    let periph = two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2());
                    vec![0.0, c, periph]
                } else {
                    // Depot and peripheral handle SS internally.
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central =
                        two_cpt_oral_f_ss(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio());
                    let periph =
                        two_cpt_oral_peripheral(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio());
                    vec![depot, central, periph]
                }
            }
            PkModel::ThreeCptIv => {
                let central = if infusion {
                    three_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_iv_bolus_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                };
                // three_cpt_iv_peripherals handles SS internally.
                let [p1, p2] =
                    three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                vec![central, p1, p2]
            }
            PkModel::ThreeCptOral => {
                if infusion {
                    // Infusions bypass depot; treat as 3-cpt IV SS infusion.
                    let c = three_cpt_infusion_ss(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                    let [p1, p2] =
                        three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                    vec![0.0, c, p1, p2]
                } else {
                    // Depot and peripherals handle SS internally.
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central = three_cpt_oral_f_ss(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    );
                    let [p1, p2] = three_cpt_oral_peripherals(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    );
                    vec![depot, central, p1, p2]
                }
            }
        }
    } else {
        // Non-SS path.
        match pk_model {
            PkModel::OneCptIv => {
                let c = if infusion {
                    one_cpt_infusion(dose, tau, cl, v)
                } else {
                    one_cpt_iv_bolus(dose, tau, cl, v)
                };
                vec![c]
            }
            PkModel::OneCptOral => {
                // Infusions bypass the depot — treat as 1-cpt IV, matching single_dose_concentration.
                if infusion {
                    let c = one_cpt_infusion(dose, tau, cl, v);
                    vec![0.0, c]
                } else {
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central = one_cpt_oral_f(dose, tau, cl, v, p.ka(), p.f_bio());
                    vec![depot, central]
                }
            }
            PkModel::OneCptTransit => {
                // [depot = lumped in-transit mass, central]; transit rejects infusions
                // at parse, so only the absorbed bolus exists.
                let depot = one_cpt_transit_depot(dose, tau, p.n_transit(), p.mtt(), p.f_bio());
                let central =
                    one_cpt_transit_f(dose, tau, cl, v, p.n_transit(), p.mtt(), p.f_bio());
                vec![depot, central]
            }
            PkModel::TwoCptIv => {
                let central = if infusion {
                    two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2())
                } else {
                    two_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2())
                };
                let periph = two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2());
                vec![central, periph]
            }
            PkModel::TwoCptOral => {
                if infusion {
                    // Infusions bypass depot; treat as 2-cpt IV
                    let c = two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2());
                    let periph = two_cpt_iv_peripheral(dose, tau, cl, v, p.q(), p.v2());
                    vec![0.0, c, periph]
                } else {
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central =
                        two_cpt_oral_f(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio());
                    let periph =
                        two_cpt_oral_peripheral(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio());
                    vec![depot, central, periph]
                }
            }
            PkModel::ThreeCptIv => {
                let central = if infusion {
                    three_cpt_infusion(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                } else {
                    three_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
                };
                let [p1, p2] =
                    three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                vec![central, p1, p2]
            }
            PkModel::ThreeCptOral => {
                if infusion {
                    let c = three_cpt_infusion(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                    let [p1, p2] =
                        three_cpt_iv_peripherals(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3());
                    vec![0.0, c, p1, p2]
                } else {
                    let depot = one_cpt_oral_depot(dose, tau, p.ka(), p.f_bio());
                    let central = three_cpt_oral_f(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    );
                    let [p1, p2] = three_cpt_oral_peripherals(
                        dose,
                        tau,
                        cl,
                        v,
                        p.q(),
                        p.v2(),
                        p.q3(),
                        p.v3(),
                        p.ka(),
                        p.f_bio(),
                    );
                    vec![depot, central, p1, p2]
                }
            }
        }
    };

    // Post-multiply bioavailability for the IV-bolus route; a no-op
    // (`f_scale == 1.0`) for oral-depot and for infusions (whose `F` is already
    // baked into the reshaped `(rate, duration)`; #419).
    for s in &mut state {
        *s *= f_scale;
    }
    state
}

/// Compute the full compartment state vector at arbitrary `times` for an analytical model.
///
/// Returns a `Vec<Vec<f64>>` where `[k]` is the full state vector at `times[k]`.
/// Uses the same lagtime/SS superposition logic as `predict_concentration`.
/// Used by the grid-integral path in `compute_extra_output_columns` when an integrand
/// references compartment states (`uses_compartments = true`), and by
/// `predict_all_states` which is the per-observation-time convenience wrapper.
pub fn analytical_state_at_times(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
    times: &[f64],
) -> Vec<Vec<f64>> {
    debug_assert!(
        subject.reset_times.is_empty(),
        "analytical_state_at_times called on a subject with EVID=3/4 resets — \
         superposition is invalid across reset boundaries; the caller should \
         return empty states for analytical+reset subjects instead"
    );
    let lagtime = pk_params.lagtime();
    let n_states = match pk_model {
        PkModel::OneCptIv => 1,
        PkModel::OneCptOral => 2,
        PkModel::OneCptTransit => 2,
        PkModel::TwoCptIv => 2,
        PkModel::TwoCptOral => 3,
        PkModel::ThreeCptIv => 3,
        PkModel::ThreeCptOral => 4,
    };
    times
        .iter()
        .map(|&t| {
            let mut state = vec![0.0_f64; n_states];
            for dose in &subject.doses {
                let t_eff = dose.time + lagtime;
                let tau = if t_eff <= t {
                    t - t_eff
                } else if dose.ss && dose.ii > 0.0 && t >= dose.time {
                    let raw = t - t_eff;
                    let n = (-raw / dose.ii).ceil();
                    let wrapped = raw + n * dose.ii;
                    if wrapped >= 0.0 {
                        wrapped
                    } else {
                        continue;
                    }
                } else {
                    continue;
                };
                let contrib = single_dose_states(pk_model, dose, tau, pk_params);
                debug_assert_eq!(
                    contrib.len(),
                    n_states,
                    "single_dose_states returned {} values but n_states={}; \
                     update both match arms together",
                    contrib.len(),
                    n_states
                );
                for (s, v) in state.iter_mut().zip(contrib.iter()) {
                    *s += v;
                }
            }
            // Floor: states cannot be negative (amounts/concentrations).
            for s in &mut state {
                if *s < 0.0 {
                    *s = 0.0;
                }
            }
            state
        })
        .collect()
}

/// Superposition of compartment states at each observation time.
///
/// Convenience wrapper around [`analytical_state_at_times`] that uses
/// `subject.obs_times` as the time vector.
pub fn predict_all_states(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
) -> Vec<Vec<f64>> {
    analytical_state_at_times(pk_model, subject, pk_params, &subject.obs_times)
}

/// Compute predictions AND full compartment states for all observations.
/// Returns `(ipred_vec, compartment_states_vec)`.
///
/// For ODE models the states come from the same ODE integration that produces ipred
/// (single-pass for non-TV, non-reset subjects). For analytical models the states are
/// computed via superposition using the same PK params.
///
/// For subjects with resets (EVID=3/4), the reset disrupts superposition; analytical
/// states are left empty (→ NaN in `[derived]`; `W_DERIVED_CMT_RESET_ANALYTICAL`
/// explains why). ODE subjects with resets are handled correctly via the event-driven
/// solver. For IOV models, states are left empty.
pub fn compute_predictions_with_states(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let uses_time = model_uses_time_builtin(model);
    if let Some(ref ode) = model.ode_spec {
        // ODE path: both ipred and states come from a single ODE integration.
        // TV-covariate and reset subjects need the event-driven path (which also
        // handles resets as break-points). Plain subjects use the simpler function
        // that avoids the per-event PK-parameter machinery.
        let pk = pk_params_at_time(model, theta, eta, &subject.covariates, 0.0);
        let (mut ipred, states) =
            if subject.has_tv_covariates() || subject.has_resets() || uses_time {
                let mut scratch = EventPkParams::with_capacity_for(subject);
                compute_event_pk_params_into(model, subject, theta, eta, &mut scratch);
                crate::ode::ode_predictions_event_driven_with_states(
                    ode,
                    subject,
                    theta,
                    eta,
                    &scratch.dose,
                    &scratch.obs,
                    &scratch.pk_only,
                )
            } else {
                // Single-pass: one ODE integration yields both ipred and states.
                crate::ode::ode_predictions_with_states(ode, &pk.values, theta, eta, subject)
            };
        // Apply [scaling] and LTBS log-transform — the single insertion point
        // shared with compute_predictions_with_tv_into_with_schedule (lines 1153–1154).
        // Without this, ODE models with `obs_scale` or `log_transform = true` would
        // get raw, unscaled ipred in SubjectResult. For Form C (ODE output_fn) models,
        // model.scaling is ScalingSpec::None, so apply_scaling is a no-op there.
        apply_scaling(model, subject, theta, eta, &mut ipred);
        apply_log_transform(model, &mut ipred);
        (ipred, states)
    } else {
        // Analytical path: ipred via compute_predictions_with_tv (handles SS, resets,
        // TV covariates); states via predict_all_states (superposition only — valid
        // for the no-reset, no-TV case).
        let ipred = compute_predictions_with_tv(model, subject, theta, eta);
        let states = if !model.analytical_init.is_empty() {
            // [initial_conditions] baseline (#521): ipred is init-aware (the
            // closed-form baseline is layered onto the central readout), but the
            // superposition state reconstruction does not yet seed the baseline
            // amount into the compartment vectors. Reporting states without the
            // baseline would disagree with ipred. Return outer-empty → NaN
            // compartments, matching the reset/TV/IOV convention.
            // W_DERIVED_INIT_ANALYTICAL explains why.
            vec![]
        } else if subject.has_resets() {
            // Superposition is invalid across resets; return outer-empty vec so
            // compartments[i] → NaN via the .unwrap_or(&[]) fallback. Consistent
            // with the IOV convention. W_DERIVED_CMT_RESET_ANALYTICAL explains why.
            vec![]
        } else if subject.has_tv_covariates() || uses_time {
            // Superposition would use baseline pk_params while ipred honours per-event
            // TV parameters — silently wrong states. Outer-empty → NaN, consistent
            // with IOV and reset. W_DERIVED_CMT_TV_ANALYTICAL explains why.
            vec![]
        } else {
            // Reached only when !uses_time (guarded above), so t=0 is exact.
            let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
            // Resolve modeled-`RATE` doses (#394) before the superposition states.
            let resolved = crate::ode::resolve_subject_doses(
                subject,
                model.active_dose_attr_map(),
                &pk.values,
            );
            if has_oral_depot_infusion(model.pk_model, &resolved) {
                // The superposition state helper (`single_dose_states`) models an
                // oral infusion as a depot-bypassing input into central, so it
                // cannot express a zero-order input into the **depot** (#400) —
                // it would report silently-wrong compartment amounts. The
                // event-driven path (used for `ipred` above) has no states
                // variant yet, so return outer-empty → NaN compartments, matching
                // the reset/TV-analytical convention. ipred stays correct;
                // sdtab/`[derived]` compartment amounts degrade to NaN.
                vec![]
            } else {
                predict_all_states(model.pk_model, &resolved, &pk)
            }
        };
        (ipred, states)
    }
}

/// Compute predictions for all observation times of a subject.
/// Uses analytical equations for standard PK models, or delegates to ODE solver
/// when an OdeSpec is provided.
pub fn compute_predictions(pk_model: PkModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    // Defensive guard (#324/#394): modeled-RATE doses (RATE=-2 -> D{cmt}) must be
    // resolved to a concrete (`Fixed`) rate/duration *before* reaching this closed
    // form. The analytical dispatch paths do exactly that (`api::model_preds` and
    // `compute_predictions_with_tv_into_with_schedule` resolve via the model's
    // `dose_attr_map`), and the public entrypoints reject an unbacked modeled dose
    // up front (`fit()` / `ferx check` via `check_model_data`, `predict()` /
    // `simulate()` via `assert_modeled_doses_supported`). Reaching here unresolved
    // means a path forgot to resolve (e.g. a direct caller of this `pub` fn on a
    // raw `Population`). A modeled dose has `rate == 0` but reports `is_infusion()`,
    // so it would route into the infusion closed form as a 0-rate "infusion" —
    // silently 0/NaN, the exact #324 silent-bolus class. A real `assert!` (not
    // `debug_assert!`) so release builds fail loudly too; it is O(doses) and
    // dwarfed by the per-observation analytical evaluation.
    assert!(
        subject.all_doses_fixed(),
        "modeled-RATE dose reached the analytical predictor unresolved \
         (resolve via dose_attr_map, or validate with check_model_data, before predicting)"
    );
    // Dose superposition cannot express a system reset (EVID=3/4): a reset
    // zeros the compartments mid-record, which is not a sum of independent
    // dose responses. Route reset-bearing subjects through the
    // state-propagating event-driven analytical path instead, replicating
    // the (constant) `pk_params` across every event slot — the same uniform
    // fill the no-TV dispatcher branch uses.
    //
    // The same routing applies to a zero-order input into the oral **depot**
    // (cmt 1, #400): the superposition closed forms (`one_cpt_oral` etc.) treat
    // an oral dose as a depot bolus + a depot-bypassing central infusion and
    // have no depot-infusion form, so a `D{depot}` infusion would be silently
    // mishandled. The event-driven propagator implements the depot zero-order
    // forced response, so route depot-infusion subjects there too.
    if (subject.has_resets() || has_oral_depot_infusion(pk_model, subject))
        && event_driven::supports_event_driven(pk_model)
    {
        let pk_dose = vec![*pk_params; subject.doses.len()];
        let pk_obs = vec![*pk_params; subject.obs_times.len()];
        let pk_pk_only = vec![*pk_params; subject.pk_only_times.len()];
        return event_driven::event_driven_predictions(
            pk_model,
            subject,
            &pk_dose,
            &pk_obs,
            &pk_pk_only,
        );
    }
    subject
        .obs_times
        .iter()
        .map(|&t| predict_concentration(pk_model, &subject.doses, t, pk_params))
        .collect()
}

/// Whether `subject` has a zero-order infusion into the **depot** (cmt 1) of an
/// oral model (#400) — a dose into compartment 1 of `one_cpt_oral` /
/// `two_cpt_oral` / `three_cpt_oral` that [`DoseEvent::is_infusion`] reports as
/// an infusion (an explicit positive `RATE`, or a still-modeled `RATE=-2` `D1`).
///
/// Such doses have no closed form in the superposition path (which models the
/// oral depot as bolus-only), so the dispatcher routes them through the
/// event-driven propagator instead, and their per-compartment states degrade to
/// NaN. IV models and oral **central** infusions (cmt 2, handled by the
/// depot-bypass IV formula) return `false`.
///
/// Uses `is_infusion()` rather than `rate > 0` so the predicate gives the same
/// answer on the **raw** subject (modeled `RATE=-2` doses still read `rate == 0`)
/// and the **resolved** subject (where it reduces to `rate > 0`). This is the
/// single source of truth shared by the prediction dispatch, the compartment-
/// state degradation, and the `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL`
/// warning — so the three can never disagree about which subjects are affected.
pub(crate) fn has_oral_depot_infusion(pk_model: PkModel, subject: &Subject) -> bool {
    pk_model.is_oral() && subject.doses.iter().any(|d| d.cmt == 1 && d.is_infusion())
}

/// Compute predictions using ODE integration.
/// `pk_params_flat` is the flat parameter vector passed to the ODE RHS function.
/// `theta` and `eta` are forwarded to `OdeSpec::output_fn` for Form C
/// (`[scaling] y = <expr>`) — pass empty slices for ODE specs without
/// `output_fn` set.
pub fn compute_predictions_ode(
    ode_spec: &crate::ode::OdeSpec,
    subject: &Subject,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    // System resets (EVID=3/4) need the state-propagating event-driven ODE
    // walker — the plain `ode_predictions` segment loop has no reset event.
    // Replicate the (constant) params across every event slot, mirroring the
    // no-TV uniform fill used by `compute_event_pk_params`.
    if subject.has_resets() {
        let mut pk = PkParams::default();
        let n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
        pk.values[..n].copy_from_slice(&pk_params_flat[..n]);
        let pk_dose = vec![pk; subject.doses.len()];
        let pk_obs = vec![pk; subject.obs_times.len()];
        let pk_pk_only = vec![pk; subject.pk_only_times.len()];
        return crate::ode::ode_predictions_event_driven(
            ode_spec,
            subject,
            theta,
            eta,
            &pk_dose,
            &pk_obs,
            &pk_pk_only,
        );
    }
    crate::ode::ode_predictions(ode_spec, pk_params_flat, theta, eta, subject)
}

/// Top-level prediction dispatcher with time-varying-covariate awareness.
///
/// Picks the appropriate path:
///   - **No TV covariates**: evaluates `pk_param_fn` once and uses the
///     existing superposition (for analytical PK) or ODE (for ODE PK)
///     fast paths — no per-event overhead.
///   - **TV covariates + analytical PK supported by `event_driven`**:
///     materialises per-event PK parameters and calls the event-driven
///     propagator. This is the NONMEM-equivalent path.
///   - **TV covariates + analytical PK *not* supported by event_driven**
///     (oral / 3-cpt): falls back to single-snapshot superposition using
///     the *first-event* PK params, matching the pre-TV behavior. The
///     dispatcher emits no warning here — the parser should have already
///     surfaced one. Tracked for follow-up.
///   - **TV covariates + ODE PK**: per-event TV is wired through the ODE
///     segment loop (Phase 4 — until then this falls back to single
///     snapshot like the analytical-unsupported branch).
/// Apply the FREM prediction override in place: for each FREMTYPE > 0
/// observation, replace the structural PK prediction with `theta[k] + eta[m]`
/// (the covariate pseudo-observation = typical value + FREM random effect),
/// using the `(theta, eta)` indices declared by `frem_predictions`.
///
/// No-op when the model has no `[frem]` config. `eta` is the BSV eta vector
/// (the FREM etas are between-subject effects), so callers on the IOV path pass
/// the BSV slice, not the kappa-augmented vector. Single source of truth shared
/// by the analytical and IOV/SAEM prediction paths.
pub(crate) fn apply_frem_prediction_override(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    preds: &mut [f64],
) {
    if let Some(ref fc) = model.frem_config {
        for (j, ft) in subject.fremtype.iter().enumerate() {
            if *ft > 0 {
                if let Some(&(theta_idx, eta_idx)) = fc.fremtype_to_indices.get(ft) {
                    if j < preds.len() {
                        preds[j] = theta[theta_idx] + eta[eta_idx];
                    }
                }
            }
        }
    }
}

pub fn compute_predictions_with_tv(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    // Allocate-on-each-call wrapper. Hot loops should use
    // `compute_predictions_with_tv_into` instead.
    let mut scratch = EventPkParams::with_capacity_for(subject);
    compute_predictions_with_tv_into(model, subject, theta, eta, &mut scratch)
}

/// Same as [`compute_predictions_with_tv`] but writes per-event PK
/// params into a caller-owned scratch buffer. Used by hot loops (SAEM
/// MH proposals, FOCE objective evaluations) that re-evaluate the
/// same subject many times with different `eta` — allocating fresh
/// `Vec<PkParams>` on every call is the dominant allocator-pressure
/// source on TV-cov datasets.
///
/// The scratch buffer is **only used on the TV-cov analytical/ODE
/// path**; the no-TV fast path doesn't touch it. Callers can pass the
/// same scratch unconditionally — the no-TV path just ignores it.
pub fn compute_predictions_with_tv_into(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
) -> Vec<f64> {
    compute_predictions_with_tv_into_with_schedule(model, subject, theta, eta, scratch, None)
}

/// Hot-path variant that also accepts a pre-built
/// [`event_driven::EventSchedule`]. When the subject takes the TV-cov
/// event-driven analytical or ODE path, the schedule is reused on
/// every call — eliminating the per-call merged-event sort and the
/// per-interval infusion-bound construction (the dominant per-call
/// CPU cost on TV-cov datasets).
///
/// `schedule` is ignored on the no-TV fast path and on the analytical
/// fallback for models that don't support event-driven propagation.
/// Callers that don't have a schedule cached can pass `None` to fall
/// back to building one on demand.
pub fn compute_predictions_with_tv_into_with_schedule(
    model: &crate::types::CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    scratch: &mut EventPkParams,
    schedule: Option<&event_driven::EventSchedule>,
) -> Vec<f64> {
    let has_tv = subject.has_tv_covariates();
    let uses_time = model_uses_time_builtin(model);

    let mut preds = if let Some(ref ode) = model.ode_spec {
        // ODE path. Resets (EVID=3/4) need the state-propagating event-driven
        // walker too, even without time-varying covariates — the plain
        // `ode_predictions` loop has no reset event.
        if has_tv || subject.has_resets() || uses_time {
            compute_event_pk_params_into(model, subject, theta, eta, scratch);
            crate::ode::ode_predictions_event_driven(
                ode,
                subject,
                theta,
                eta,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            )
        } else {
            let pk = pk_params_at_time(model, theta, eta, &subject.covariates, 0.0);
            crate::ode::ode_predictions(ode, &pk.values, theta, eta, subject)
        }
    } else if (has_tv || subject.has_resets() || uses_time)
        && event_driven::supports_event_driven(model.pk_model)
    {
        compute_event_pk_params_into(model, subject, theta, eta, scratch);
        // Resolve modeled-`RATE` doses (#324/#394, e.g. `RATE=-2` → `D{cmt}`) to
        // concrete duration/rate using each dose's per-event PK snapshot, before the
        // event-driven walker builds its infusion bounds. Borrowed (no allocation)
        // for the all-`Fixed` common case.
        let resolved =
            crate::ode::resolve_subject_doses_with(subject, model.active_dose_attr_map(), |k| {
                &scratch.dose[k].values
            });
        // A cached `EventSchedule` was built from the *unresolved* subject, whose
        // modeled-duration infusions still read `duration == 0`; reuse it only when
        // nothing was resolved (the borrowed case). Otherwise rebuild from the
        // resolved subject — modeled duration is η-dependent, so a cached schedule
        // could not be reused across iterations anyway.
        if let (std::borrow::Cow::Borrowed(_), Some(sched)) = (&resolved, schedule) {
            // Nothing was resolved (all doses already `Fixed`) → the cached schedule
            // is valid, reuse it.
            event_driven::event_driven_predictions_with_schedule(
                model.pk_model,
                &resolved,
                sched,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            )
        } else {
            // A modeled dose was resolved (or no cache) → rebuild the schedule from
            // the resolved subject (a cache built from the unresolved subject reads
            // `duration == 0`, and modeled duration is η-dependent anyway).
            event_driven::event_driven_predictions(
                model.pk_model,
                &resolved,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            )
        }
    } else {
        // No-TV fast path (or TV with unsupported model — see docstring).
        // The fast path is time-independent; the degraded unsupported-model
        // branch matches the existing single-snapshot TV behaviour.
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
        // Resolve any modeled-`RATE` doses (#394) before the closed-form math.
        let resolved =
            crate::ode::resolve_subject_doses(subject, model.active_dose_attr_map(), &pk.values);
        compute_predictions(model.pk_model, &resolved, &pk)
    };

    // Analytical initial-compartment amounts (issue #521): layer the closed-form
    // init impulse onto the dose-driven prediction BEFORE scaling, so a
    // `[scaling]` divisor applies to the dose+init total. No-op for ODE models
    // (seeded via `init_fn`) and when no `[initial_conditions]` block is present.
    add_analytical_init(model, subject, theta, eta, &mut preds);

    // Analytic Form C readout (#650): replace the built-in central concentration
    // with `y = <expr>` before scaling/log-transform. No-op unless the model has
    // an `analytic_readout` (ODE Form C is applied inside `ode_predictions*`).
    apply_analytic_readout(model, subject, theta, eta, &mut preds);

    // `[scaling]` post-multiply. Single insertion point covers FOCE/FOCEI,
    // GN, trust-region, SAEM, and IOV — they all route through here.
    // Form C (ODE `y = <expr>`) is already applied inside `ode_predictions*`
    // via `OdeSpec::output_fn`, so `model.scaling` is `None` for those. Analytic
    // Form C sets `ScalingSpec::None` too (the readout replaces the output), so
    // `apply_scaling` is a no-op here for it.
    apply_scaling(model, subject, theta, eta, &mut preds);
    apply_log_transform(model, &mut preds);

    // FREM override AFTER log-transform and scaling: replace predictions for
    // FREMTYPE > 0 observations with theta[k] + eta[m] (raw covariate values).
    // Covariate pseudo-observations use an additive model (Y = theta + eta + eps)
    // regardless of the PK error model, so the log-transform must not apply.
    apply_frem_prediction_override(model, subject, theta, eta, &mut preds);
    preds
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn bolus_dose(time: f64, amt: f64) -> DoseEvent {
        DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
    }

    fn make_pk_params(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[0] = cl;
        p.values[1] = v;
        p
    }

    // ── analytical [initial_conditions] (issue #521) ──

    #[test]
    fn analytical_init_central_one_cpt_iv_is_bolus_decay() {
        // An initial amount A₀ in the central compartment of a 1-cpt model
        // decays as the IV-bolus impulse: C(t) = (A₀/V)·exp(-(CL/V)·t).
        let (cl, v, a0) = (2.0_f64, 10.0_f64, 500.0_f64);
        let p = make_pk_params(cl, v);
        for &t in &[0.0_f64, 1.0, 5.0, 24.0] {
            let got = analytical_init_concentration(PkModel::OneCptIv, 1, a0, t, &p);
            let expected = (a0 / v) * (-(cl / v) * t).exp();
            assert_relative_eq!(got, expected, max_relative = 1e-12);
        }
    }

    #[test]
    fn analytical_init_central_oral_uses_cmt2_and_ignores_ka() {
        // For an oral model, `central` is cmt 2 and the init is a direct IV
        // bolus into central — KA must not enter (the amount is already there,
        // not absorbed through the depot). So it matches the 1-cpt decay and is
        // independent of KA.
        let (cl, v, a0) = (3.0_f64, 20.0_f64, 250.0_f64);
        let mut p = make_pk_params(cl, v);
        p.values[crate::types::PK_IDX_KA] = 1.7;
        let mut p_other_ka = p;
        p_other_ka.values[crate::types::PK_IDX_KA] = 0.3;
        let t = 4.0;
        let got = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p);
        let got_other = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p_other_ka);
        let expected = (a0 / v) * (-(cl / v) * t).exp();
        assert_relative_eq!(got, expected, max_relative = 1e-12);
        assert_relative_eq!(got, got_other, max_relative = 1e-12);
    }

    #[test]
    fn analytical_init_depot_oral_absorbs_through_ka() {
        // An init amount pre-loaded in the depot (cmt 1) of an oral model is
        // absorbed first-order with F=1, i.e. the oral closed form with D=A₀.
        let (cl, v, ka, a0) = (3.0_f64, 20.0_f64, 1.1_f64, 100.0_f64);
        let mut p = make_pk_params(cl, v);
        p.values[crate::types::PK_IDX_KA] = ka;
        let t = 2.0;
        let got = analytical_init_concentration(PkModel::OneCptOral, 1, a0, t, &p);
        let expected = crate::sens::one_cpt::one_cpt_oral_g::<f64>(a0, t, cl, v, ka, 1.0);
        assert_relative_eq!(got, expected, max_relative = 1e-12);
        // At t=0 nothing has absorbed yet → zero central concentration.
        let at0 = analytical_init_concentration(PkModel::OneCptOral, 1, a0, 0.0, &p);
        assert_relative_eq!(at0, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form() {
        // NONMEM anchor for #327/#419. With bioavailability F1 on an IV dose,
        // NONMEM delivers F1·AMT, so for a 1-cpt model (ADVAN1/TRANS2,
        // `F1 = THETA(3)` in $PK) the closed forms are:
        //   bolus:    C(t) = F·Dose/V · exp(-k·t)
        //   infusion: rate held at R, duration scaled to T_F = F·AMT/R (#419):
        //             C(t) = R/CL · (1 − exp(-k·t))                   (t ≤ T_F)
        //             C(t) = R/CL · (1 − exp(-k·T_F)) · exp(-k·(t−T_F)) (t > T_F)
        // For a *rate-defined* infusion NONMEM keeps the rate and scales the
        // duration (total exposure still F·AMT), so there is no amplitude factor
        // on the rate. The analytical superposition path (`predict_concentration`)
        // must reproduce these.
        let (cl, v, f) = (5.0_f64, 50.0_f64, 0.4_f64);
        let k = cl / v;
        let mut pk = make_pk_params(cl, v);
        pk.values[crate::types::PK_IDX_F] = f;

        // IV bolus: F scales the amount (unchanged by #419).
        let amt = 100.0;
        let doses = vec![bolus_dose(0.0, amt)];
        for &t in &[0.25_f64, 1.0, 4.0, 12.0] {
            let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let want = f * amt / v * (-k * t).exp();
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }

        // IV infusion: R=25, raw T = AMT/R = 4 h; under F the duration scales to
        // T_F = F·AMT/R = 1.6 h with the rate held at R.
        let rate = 25.0;
        let t_inf = f * amt / rate;
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        for &t in &[1.0_f64, 4.0, 8.0] {
            let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let want = if t <= t_inf {
                rate / cl * (1.0 - (-k * t).exp())
            } else {
                rate / cl * (1.0 - (-k * t_inf).exp()) * (-k * (t - t_inf)).exp()
            };
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }
    }

    #[test]
    fn consumes_pk_slot_matches_solver() {
        // Pin `PkModel::consumes_pk_slot` — the source of truth for the parser's
        // unused-param warning (#309) — to what the analytical solvers actually
        // read: perturbing a *consumed* slot must change the predicted
        // concentration, and perturbing a *non-consumed* slot must leave it
        // bit-identical. A future variant whose closed form reads a new slot but
        // isn't reflected in `consumes_pk_slot`/`required_pk_params` fails here,
        // closing the drift gap between the parser table and the solvers.
        use crate::types::{
            PK_IDX_CL, PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V,
            PK_IDX_V2, PK_IDX_V3,
        };
        let named_slots = [
            PK_IDX_CL,
            PK_IDX_V,
            PK_IDX_Q,
            PK_IDX_V2,
            PK_IDX_KA,
            PK_IDX_F,
            PK_IDX_Q3,
            PK_IDX_V3,
            PK_IDX_LAGTIME,
        ];
        let baseline = || {
            let mut p = PkParams::default();
            p.values[PK_IDX_CL] = 1.0;
            p.values[PK_IDX_V] = 10.0;
            p.values[PK_IDX_Q] = 0.7;
            p.values[PK_IDX_V2] = 20.0;
            p.values[PK_IDX_KA] = 1.3;
            p.values[PK_IDX_F] = 0.6;
            p.values[PK_IDX_Q3] = 0.4;
            p.values[PK_IDX_V3] = 30.0;
            p.values[PK_IDX_LAGTIME] = 0.5;
            p
        };
        let doses = vec![bolus_dose(0.0, 100.0)];
        let t = 2.0;
        for model in [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ] {
            let c0 = predict_concentration(model, &doses, t, &baseline());
            assert!(
                c0 > 0.0,
                "{model:?}: baseline concentration should be positive"
            );
            for &slot in &named_slots {
                let mut perturbed = baseline();
                perturbed.values[slot] *= 5.0;
                let c1 = predict_concentration(model, &doses, t, &perturbed);
                if model.consumes_pk_slot(slot) {
                    assert!(
                        (c1 - c0).abs() > 1e-9,
                        "{model:?}: slot {slot} is marked consumed but perturbing it \
                         did not change the prediction (c0={c0}, c1={c1})"
                    );
                } else {
                    assert_eq!(
                        c1, c0,
                        "{model:?}: slot {slot} is marked unused but perturbing it \
                         changed the prediction"
                    );
                }
            }
        }
    }

    #[test]
    fn test_superposition_single_dose() {
        let doses = vec![bolus_dose(0.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let c = predict_concentration(PkModel::OneCptIv, &doses, 0.0, &pk);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_two_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(10.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let k: f64 = 10.0 / 100.0;

        // At t=10, first dose has decayed, second dose just given
        let c = predict_concentration(PkModel::OneCptIv, &doses, 10.0, &pk);
        let expected = (1000.0_f64 / 100.0) * (-k * 10.0).exp() + 1000.0 / 100.0;
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_ignores_future_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(100.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);

        // At t=5, second dose hasn't happened yet
        let c_single =
            predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 1000.0)], 5.0, &pk);
        let c_two = predict_concentration(PkModel::OneCptIv, &doses, 5.0, &pk);
        assert_relative_eq!(c_single, c_two, epsilon = 1e-12);
    }

    #[test]
    fn compute_predictions_routes_reset_subject_to_event_driven() {
        // A subject with a reset (EVID=3) must NOT use superposition:
        // `compute_predictions` should zero the state at the reset. Dose at
        // t=0, reset at t=5, obs at t=1 (positive) and t=6 (zero).
        let subj = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 6.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 2],
            obs_cmts: vec![1; 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: vec![5.0],
            cens: vec![0; 2],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIv, &subj, &pk);
        assert!(preds[0] > 0.0);
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "modeled-RATE dose reached the analytical predictor")]
    fn compute_predictions_panics_on_modeled_dose() {
        // #324 / review #2: the analytical superposition predictor must reject a
        // modeled-RATE (RATE=-2) dose loudly *in release too* — a real `assert!`,
        // not the old `debug_assert!` that compiled out and let a 0-rate "infusion"
        // silently produce 0/NaN. Direct call (bypassing the public entrypoints'
        // gates) proves the predictor itself fails fast on an unvalidated subject.
        use crate::types::RateMode;
        let mut subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 0, 1);
        subj.doses = vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            RateMode::ModeledDuration,
        )];
        let pk = make_pk_params(10.0, 100.0);
        let _ = compute_predictions(PkModel::OneCptIv, &subj, &pk);
    }

    fn make_subject_with_tv(
        cov_const: HashMap<String, f64>,
        dose_cov: Vec<HashMap<String, f64>>,
        obs_cov: Vec<HashMap<String, f64>>,
        n_doses: usize,
        n_obs: usize,
    ) -> Subject {
        Subject {
            id: "1".to_string(),
            doses: (0..n_doses).map(|i| bolus_dose(i as f64, 100.0)).collect(),
            obs_times: (0..n_obs).map(|i| 1.0 + i as f64).collect(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: cov_const,
            dose_covariates: dose_cov,
            obs_covariates: obs_cov,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Build a tiny analytical CompiledModel where CL = covariate `CR` *
    /// theta[0] (so we can prove pk_param_fn was called with the right
    /// covariate snapshot per event).
    fn cl_from_cr_model() -> crate::types::CompiledModel {
        use crate::types::{
            BloqMethod, CompiledModel, ErrorModel, GradientMethod, ModelParameters, OmegaMatrix,
            PkModel, ScalingSpec, SigmaVector,
        };
        CompiledModel {
            name: "cl_from_cr".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Additive,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Additive),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(|theta, _eta, cov, _t: f64| {
                let mut p = PkParams::default();
                let cr = cov.get("CR").copied().unwrap_or(1.0);
                p.values[crate::types::PK_IDX_CL] = theta[0] * cr;
                p.values[crate::types::PK_IDX_V] = 50.0;
                p
            }),
            n_theta: 1,
            n_eta: 0,
            n_epsilon: 1,
            n_kappa: 0,
            theta_names: vec!["TVCL".into()],
            eta_names: Vec::new(),
            kappa_names: Vec::new(),
            default_params: ModelParameters {
                theta: vec![1.0],
                theta_names: vec!["TVCL".into()],
                theta_lower: vec![0.0],
                theta_upper: vec![f64::INFINITY],
                theta_fixed: vec![false],
                omega: OmegaMatrix::from_diagonal(&[], Vec::new()),
                omega_fixed: Vec::new(),
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["EPS".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["CR".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            indiv_param_names: Vec::new(),
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
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
        }
    }

    #[test]
    fn test_event_pk_params_no_tv_evaluates_once() {
        // No TV covariates → uses subject.covariates; result is identical
        // for every event.
        let mut covs = HashMap::new();
        covs.insert("CR".to_string(), 2.0);
        let subj = make_subject_with_tv(covs, Vec::new(), Vec::new(), 2, 3);
        let model = cl_from_cr_model();
        let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
        assert_eq!(ev.dose.len(), 2);
        assert_eq!(ev.obs.len(), 3);
        // CL = theta * CR = 10 * 2 = 20 everywhere.
        for p in ev.dose.iter().chain(ev.obs.iter()) {
            assert_relative_eq!(p.cl(), 20.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_event_pk_params_tv_uses_per_event_snapshot() {
        // Per-event snapshots drive different CL at each event.
        let mk = |cr: f64| {
            let mut h = HashMap::new();
            h.insert("CR".to_string(), cr);
            h
        };
        let dose_cov = vec![mk(1.0), mk(1.5)];
        let obs_cov = vec![mk(1.0), mk(2.0)];
        let subj = make_subject_with_tv(mk(1.0), dose_cov, obs_cov, 2, 2);
        let model = cl_from_cr_model();
        let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
        assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12);
        assert_relative_eq!(ev.dose[1].cl(), 15.0, epsilon = 1e-12);
        assert_relative_eq!(ev.obs[0].cl(), 10.0, epsilon = 1e-12);
        assert_relative_eq!(ev.obs[1].cl(), 20.0, epsilon = 1e-12);
    }

    #[test]
    fn test_event_pk_params_time_builtin_uses_event_times_without_tv_covariates() {
        let model_str = "
[parameters]
  theta TVCL(10.0, 0.001, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.1

[individual_parameters]
  if (TIME > 0.5) {
    CL = (TVCL + TIME) * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }

[structural_model]
  pk one_cpt_iv(cl=CL, v=time)

[error_model]
  DV ~ additive(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(model_str)
            .expect("TIME/time built-ins parse");
        assert!(
            !model
                .referenced_covariates
                .iter()
                .any(|c| c == "TIME" || c == "time"),
            "TIME/time must not be tracked as data covariates: {:?}",
            model.referenced_covariates
        );

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 2, 2);
        assert!(!subj.has_tv_covariates(), "fixture must stay on no-TV path");

        let ev = compute_event_pk_params(&model, &subj, &model.default_params.theta, &[0.0]);

        assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12); // TIME = 0
        assert_relative_eq!(ev.dose[1].cl(), 11.0, epsilon = 1e-12); // TIME = 1
        assert_relative_eq!(ev.obs[0].cl(), 11.0, epsilon = 1e-12); // TIME = 1
        assert_relative_eq!(ev.obs[1].cl(), 12.0, epsilon = 1e-12); // TIME = 2
        assert_relative_eq!(ev.dose[0].v(), 0.0, epsilon = 1e-12); // time = 0
        assert_relative_eq!(ev.obs[1].v(), 2.0, epsilon = 1e-12); // time = 2
    }

    #[test]
    fn test_compute_predictions_length() {
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 4],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIv, &subject, &pk);
        assert_eq!(preds.len(), 4);
        // Predictions should be monotonically decreasing for IV bolus
        for i in 1..preds.len() {
            assert!(preds[i] < preds[i - 1]);
        }
    }

    fn oral_pk_params(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p
    }

    #[test]
    fn test_predict_concentration_lagtime_zero_matches_baseline() {
        // With lagtime = 0, predict_concentration must agree with the
        // pre-feature behavior at any observation time.
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let pk = oral_pk_params(2.0, 20.0, 1.5); // CL/V = 0.1, KA = 1.5
        let c0 = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk);
        // Bateman: C(t) = (F*Dose*KA) / (V*(KA-k)) * [exp(-k*t) - exp(-KA*t)]
        let k = 2.0_f64 / 20.0;
        let expected =
            (1.0 * 100.0 * 1.5) / (20.0 * (1.5 - k)) * ((-k * 2.0).exp() - (-1.5 * 2.0_f64).exp());
        assert_relative_eq!(c0, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_predict_concentration_lagtime_shifts_curve() {
        // Concentration at t with lagtime=L should equal the unlagged
        // concentration evaluated at (t - L).
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let mut pk_lag = oral_pk_params(2.0, 20.0, 1.5);
        pk_lag.values[crate::types::PK_IDX_LAGTIME] = 1.5;
        let pk_nolag = oral_pk_params(2.0, 20.0, 1.5);

        // At t = 3.5 with lag=1.5, effective elapsed time is 2.0.
        let c_lag = predict_concentration(PkModel::OneCptOral, &doses, 3.5, &pk_lag);
        let c_no = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk_nolag);
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_concentration_lagtime_before_first_obs_returns_zero() {
        // Lagtime longer than the observation window: nothing has reached
        // the system yet, so concentration is exactly 0.
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let mut pk = oral_pk_params(2.0, 20.0, 1.5);
        pk.values[crate::types::PK_IDX_LAGTIME] = 5.0;
        let c = predict_concentration(PkModel::OneCptOral, &doses, 1.0, &pk);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_infusion_with_lagtime_shifts_window() {
        // 1-cpt infusion, amt=100, rate=100 → duration = amt/rate = 1.0
        // (the `DoseEvent::new` signature is `(time, amt, cmt, rate, ss, ii)`
        // — duration is auto-computed from amt/rate; the trailing 0.0 is `ii`).
        // Starting at t_dose=2, with lagtime=0.5, the infusion effectively
        // runs 2.5..3.5.
        let dose = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
        debug_assert!(
            dose.is_infusion() && dose.duration > 0.0,
            "test must exercise the infusion branch"
        );
        let doses = vec![dose.clone()];
        let mut pk = make_pk_params(10.0, 100.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 0.5;

        // (a) Before lagged start: still zero.
        let c_pre = predict_concentration(PkModel::OneCptIv, &doses, 0.6, &pk);
        assert_eq!(c_pre, 0.0);

        // (b) Mid-infusion (lagged): pre-lag conc at tau = t - (dose.time + lag) = t - 2.5.
        // Compare against unlagged infusion at tau via the same formula.
        let pk_nolag = make_pk_params(10.0, 100.0);
        let c_lag = predict_concentration(PkModel::OneCptIv, &doses, 2.6, &pk);
        let c_no = predict_concentration(
            PkModel::OneCptIv,
            &[DoseEvent::new(0.1, 100.0, 1, 100.0, false, 0.0)],
            0.2,
            &pk_nolag,
        );
        // Both probe an elapsed time of 0.1 into a 1h infusion of rate=100.
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-10);

        // (c) Post-infusion (lagged window ends at 3.5).
        let c_post = predict_concentration(PkModel::OneCptIv, &doses, 3.6, &pk);
        let c_post_nolag = predict_concentration(
            PkModel::OneCptIv,
            &[DoseEvent::new(0.0, 100.0, 1, 100.0, false, 0.0)],
            1.1,
            &pk_nolag,
        );
        assert_relative_eq!(c_post, c_post_nolag, epsilon = 1e-10);
    }

    // --- Steady-state (SS=1) integration with predict_concentration ---

    #[test]
    fn test_ss_iv_bolus_via_predict_matches_closed_form() {
        // Sanity check: predict_concentration with an SS=1 dose dispatches to
        // the closed-form SS formula (not the single-dose response).
        let ii = 12.0_f64;
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];
        let pk = make_pk_params(10.0, 100.0);
        let k = 10.0_f64 / 100.0;
        for &t in &[0.0, 3.0, 11.9, 24.0] {
            let c = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let expected = (1000.0 / 100.0) * (-k * t).exp() / (1.0 - (-k * ii).exp());
            assert_relative_eq!(c, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ss_with_lagtime_identity_c_t_l_equals_c_t_minus_l() {
        // Acceptance criterion from issue #15: with SS+lagtime, the curve at
        // time t with lagtime L equals the unlagged SS curve at t - L. This
        // is the geometric-series identity under linear superposition.
        let ii = 12.0;
        let lagtime = 1.5;
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];

        let mut pk_lag = make_pk_params(10.0, 100.0);
        pk_lag.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let pk_nolag = make_pk_params(10.0, 100.0);

        // Sample several t > lagtime so both curves are defined.
        for &t in &[2.0, 5.0, 11.0, 13.5] {
            let c_lag = predict_concentration(PkModel::OneCptIv, &doses, t, &pk_lag);
            let c_no = predict_concentration(PkModel::OneCptIv, &doses, t - lagtime, &pk_nolag);
            assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
        }
    }

    /// NONMEM cross-check for SS + ALAG1 (issue #15). Reference PRED values
    /// were generated with NONMEM 7.5.1 (`ADVAN2 TRANS2`, `$ESTIMATION
    /// MAXEVAL=0`) for a 1-cpt oral model: CL=2, V=20, KA=1.5, ALAG1=1.5,
    /// a single `SS=1, II=24, AMT=100` dose into the depot, observed in the
    /// central compartment (`S2=V`). Control file + dataset documented in
    /// `tests/ss_lagtime_nonmem.rs`.
    ///
    /// The two earliest samples (t=0.5, 1.0 < ALAG1) exercise the
    /// previous-interval steady-state tail: NONMEM reports ~0.59 / ~0.56
    /// there, not 0. This is the analytical-path coverage of the wrap added
    /// to `predict_concentration`.
    #[test]
    fn test_ss_oral_with_lagtime_matches_nonmem() {
        let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
        let doses = vec![DoseEvent::new(0.0, amt, 1, 0.0, true, ii)];
        let mut pk = oral_pk_params(cl, v, ka);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

        // (observation time, NONMEM PRED). t < 1.5 is the previous-interval
        // tail; t >= 1.5 is the current interval; t > II decays (naked SS).
        let nonmem: &[(f64, f64)] = &[
            (0.5, 0.59069),
            (1.0, 0.56188),
            (2.0, 3.07370),
            (4.0, 4.46240),
            (8.0, 3.07540),
            (12.0, 2.06170),
            (18.0, 1.13150),
            (23.0, 0.68628),
            (25.0, 0.56188),
            (30.0, 0.34080),
        ];
        for &(t, pred) in nonmem {
            let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
            assert_relative_eq!(c, pred, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_ss_dose_does_not_contribute_before_its_record_time() {
        // SS=1 establishes steady state *at* the dose record time, so an SS
        // dose must not contribute to observations earlier than its own
        // record (the pre-arrival wrap only applies in [dose.time, t_eff)).
        // Regression guard: before the `t >= dose.time` gate, a future SS
        // record leaked a non-zero steady-state tail into earlier times.
        let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
        // SS dose recorded at t = 10 (not at the origin).
        let doses = vec![DoseEvent::new(10.0, amt, 1, 0.0, true, ii)];
        let mut pk = oral_pk_params(cl, v, ka);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

        // Strictly before the record time → no contribution.
        for &t in &[0.0, 5.0, 9.9] {
            let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
            assert_eq!(c, 0.0, "future SS dose leaked into t={t}");
        }
        // In the record-to-arrival window [10, 11.5) → previous-interval tail.
        let c_pre = predict_concentration(PkModel::OneCptOral, &doses, 10.5, &pk);
        let dose0 = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        // phase = (10.5 - 11.5) + II = 23.0
        let expected = one_cpt_oral_ss(&dose0, 23.0, cl, v, ka);
        assert_relative_eq!(c_pre, expected, max_relative = 1e-9);
    }

    #[test]
    fn test_ss_oral_via_predict_dispatches_to_ss_formula() {
        // With SS=1 and II large enough that exp(-k·II) is small, the SS
        // value should be close to but strictly above the single-dose value
        // at the same τ (the "+1" geometric-tail extra contribution).
        let ii = 24.0_f64;
        let cl = 2.0;
        let v = 20.0;
        let ka = 1.5;

        let doses_ss = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)];
        let doses_single = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let pk = oral_pk_params(cl, v, ka);

        let t = 4.0;
        let c_ss = predict_concentration(PkModel::OneCptOral, &doses_ss, t, &pk);
        let c_single = predict_concentration(PkModel::OneCptOral, &doses_single, t, &pk);
        assert!(
            c_ss > c_single,
            "SS conc {} should be greater than single-dose conc {}",
            c_ss,
            c_single
        );
        // With II = 24 and k = 0.1, KA = 1.5: exp(-k·II) ≈ 0.091, so the SS
        // tail adds ~10% to the slow term. Sanity: under 50%.
        assert!((c_ss / c_single) < 1.5);
    }

    // ── apply_scaling ───────────────────────────────────────────────────────

    /// Build the same `cl_from_cr_model` test fixture but with the
    /// requested `ScalingSpec` swapped in. Used by all apply_scaling tests
    /// so they exercise the production signature
    /// (`apply_scaling(&model, &subject, ...)`).
    fn model_with_scaling(scaling: crate::types::ScalingSpec) -> crate::types::CompiledModel {
        let mut m = cl_from_cr_model();
        m.scaling = scaling;
        m
    }

    fn one_subject_for_scaling() -> Subject {
        let mut cov = HashMap::new();
        cov.insert("CR".to_string(), 1.0);
        make_subject_with_tv(cov, Vec::new(), Vec::new(), 1, 3)
    }

    #[test]
    fn test_apply_scaling_none_is_noop() {
        let model = model_with_scaling(ScalingSpec::None);
        let subj = one_subject_for_scaling();
        let mut preds = vec![10.0, 20.0, 30.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_eq!(preds, vec![10.0, 20.0, 30.0]);
    }

    // ── analytic Form C readout (#650) ───────────────────────────────────────

    /// A degenerate `y = central / V` readout must reproduce the built-in
    /// concentration bit-for-bit (central amount = conc × V, so `/V` recovers
    /// conc), and an additive term must shift each prediction by exactly that
    /// amount — proving the readout replaces the concentration output correctly.
    #[test]
    fn analytic_form_c_readout_matches_and_shifts_concentration() {
        use crate::parser::model_parser::parse_model_string;
        let base = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
";
        let plain = parse_model_string(base).expect("plain parses");
        let ident = parse_model_string(&format!("{base}[scaling]\n  y = central / V\n"))
            .expect("identity readout parses");
        let shifted = parse_model_string(&format!("{base}[scaling]\n  y = central / V + 2.0\n"))
            .expect("shifted readout parses");
        assert!(ident.analytic_readout.is_some());

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
        let theta = [1.0, 50.0];
        let c_plain = compute_predictions_with_tv(&plain, &subj, &theta, &[0.0]);
        let c_ident = compute_predictions_with_tv(&ident, &subj, &theta, &[0.0]);
        let c_shift = compute_predictions_with_tv(&shifted, &subj, &theta, &[0.0]);

        assert_eq!(c_plain.len(), 6);
        for i in 0..c_plain.len() {
            assert_relative_eq!(c_ident[i], c_plain[i], max_relative = 1e-12);
            assert_relative_eq!(c_shift[i], c_plain[i] + 2.0, epsilon = 1e-9);
            assert!(c_plain[i] > 0.0, "sanity: positive concentration");
        }
    }

    /// The readout's covariate is read from the **per-observation** snapshot, so a
    /// per-row flag (here `SEL`) selects the branch on that row — the analytic
    /// analogue of NONMEM's per-record `$ERROR`. Rows with `SEL==0` read `central/V`
    /// (concentration); rows with `SEL==1` read twice that.
    #[test]
    fn analytic_form_c_readout_uses_per_obs_covariate() {
        use crate::parser::model_parser::parse_model_string;
        let src = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (SEL == 0) central / V else 2 * central / V
[error_model]
  DV ~ proportional(PROP)
";
        let model = parse_model_string(src).expect("parses");
        // 4 observations, alternating SEL flag per row.
        let obs_cov = vec![
            HashMap::from([("SEL".to_string(), 0.0)]),
            HashMap::from([("SEL".to_string(), 1.0)]),
            HashMap::from([("SEL".to_string(), 0.0)]),
            HashMap::from([("SEL".to_string(), 1.0)]),
        ];
        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), obs_cov, 1, 4);
        let theta = [1.0, 50.0];
        let preds = compute_predictions_with_tv(&model, &subj, &theta, &[0.0]);
        // Odd rows (SEL==1) are exactly double the even rows' readout only if the
        // underlying concentration were equal — instead compare each row to its own
        // concentration via the ratio of the SEL==1 formula (2×) applied on its conc.
        assert_relative_eq!(
            preds[1] / preds[0],
            2.0 * (preds_conc(&subj, &theta, 1) / preds_conc(&subj, &theta, 0)),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            preds[3] / preds[2],
            2.0 * (preds_conc(&subj, &theta, 3) / preds_conc(&subj, &theta, 2)),
            max_relative = 1e-9
        );
    }

    /// Helper: the plain built-in concentration at observation `i` for the fixture
    /// above (CL=1, V=50, single 100 mg IV bolus at t=0).
    fn preds_conc(subj: &Subject, theta: &[f64], i: usize) -> f64 {
        let mut pk = PkParams::default();
        pk.values[crate::types::PK_IDX_CL] = theta[0];
        pk.values[crate::types::PK_IDX_V] = theta[1];
        predict_concentration(PkModel::OneCptIv, &subj.doses, subj.obs_times[i], &pk)
    }

    /// **NONMEM-validation cross-check (#650).** An analytic Form C readout
    /// computes the same output as the *ODE* Form C readout, which is validated
    /// against NONMEM's per-record `$ERROR` (free/total protein-binding model,
    /// `docs/model-file/scaling.qmd`). On an IV 1-cpt model the analytic central
    /// amount (`concentration × V`) equals the ODE `central` state amount, so a
    /// saturable-binding total-vs-free readout `y = C + BMAX·C/(KD+C)` must give
    /// bit-for-bit-equal predictions on the analytic and ODE paths at matched
    /// parameters — inheriting the ODE path's NONMEM anchor.
    #[test]
    fn analytic_form_c_matches_ode_form_c_binding_readout() {
        use crate::parser::model_parser::parse_model_string;
        let readout = "  y = central / V + BMAX * (central / V) / (KD + central / V)\n";
        let params = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
";
        let analytic = parse_model_string(&format!(
            "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
        ))
        .expect("analytic parses");
        // Equivalent ODE: `central` is the amount, disposition -(CL/V)·central,
        // IV bolus into central (cmt 1). Same readout replaces the state readout.
        let ode = parse_model_string(&format!(
            "{params}[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
        ))
        .expect("ode parses");
        assert!(analytic.analytic_readout.is_some());
        assert!(ode.ode_spec.is_some());

        let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
        let theta = [0.2, 10.0, 3.0, 2.0];
        let eta = [0.1, -0.05];
        let a = compute_predictions_with_tv(&analytic, &subj, &theta, &eta);
        let o = compute_predictions_with_tv(&ode, &subj, &theta, &eta);
        assert_eq!(a.len(), o.len());
        for (av, ov) in a.iter().zip(o.iter()) {
            assert_relative_eq!(av, ov, max_relative = 1e-6, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_apply_log_transform_noop_when_disabled() {
        let mut model = cl_from_cr_model();
        model.log_transform = false;
        let mut preds = vec![1.0, 2.0, 4.0];
        apply_log_transform(&model, &mut preds);
        assert_eq!(preds, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_apply_log_transform_logs_with_floor() {
        let mut model = cl_from_cr_model();
        model.log_transform = true;
        let mut preds = vec![1.0, std::f64::consts::E, 0.0, -5.0];
        apply_log_transform(&model, &mut preds);
        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(preds[1], 1.0, epsilon = 1e-12);
        // Non-positive predictions are floored, not turned into -inf/NaN.
        assert_relative_eq!(preds[2], LTBS_FLOOR.ln(), epsilon = 1e-9);
        assert_relative_eq!(preds[3], LTBS_FLOOR.ln(), epsilon = 1e-9);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_ltbs_prediction_is_log_of_natural() {
        // The whole prediction pipeline (compute_predictions_with_tv) returns
        // ln(f) when LTBS is active, so log-scale predictions flow into every
        // downstream consumer (likelihood, IWRES/CWRES, IPRED, simulation).
        let subj = one_subject_for_scaling();
        let theta = [1.0];

        let mut natural_model = cl_from_cr_model();
        natural_model.log_transform = false;
        let natural = compute_predictions_with_tv(&natural_model, &subj, &theta, &[]);

        let mut ltbs_model = cl_from_cr_model();
        ltbs_model.log_transform = true;
        let logged = compute_predictions_with_tv(&ltbs_model, &subj, &theta, &[]);

        assert_eq!(natural.len(), logged.len());
        for (n, l) in natural.iter().zip(logged.iter()) {
            assert_relative_eq!(*l, n.max(LTBS_FLOOR).ln(), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_ltbs_frem_predictions_are_raw_not_logged() {
        // Regression: FREM covariate predictions must be raw (theta + eta),
        // NOT log-transformed, even when the PK error model uses log_additive.
        // The override must happen AFTER apply_log_transform so the log does
        // not corrupt covariate predictions.
        let mut model = cl_from_cr_model();
        model.log_transform = true; // LTBS (log-additive)
        model.n_eta = 2;
        model.eta_names = vec!["ETA_CL".into(), "ETA_COV".into()];
        model.default_params.omega = crate::types::OmegaMatrix::from_diagonal(
            &[0.1, 100.0],
            vec!["ETA_CL".into(), "ETA_COV".into()],
        );
        // FREM config: FREMTYPE 100 -> (theta_idx=0 [TVCL], eta_idx=1 [ETA_COV])
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize));
        model.frem_config = Some(crate::types::FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        });

        let theta = [1.0]; // TVCL = 1.0
        let eta = [0.0, 5.0]; // ETA_COV = 5.0
                              // Expected FREM prediction: theta[0] + eta[1] = 1.0 + 5.0 = 6.0 (raw, NOT logged)

        // Subject with 2 obs: FREMTYPE 0 (PK) and FREMTYPE 100 (covariate)
        let mut subj = one_subject_for_scaling(); // has 3 obs
        subj.fremtype = vec![0, 100, 0];

        let preds = compute_predictions_with_tv(&model, &subj, &theta, &eta);
        // PK rows (0, 2) should be log-transformed
        assert!(preds[0].is_finite());
        // FREM row (1) must be raw: theta[0] + eta[1] = 6.0, not ln(6.0)
        assert_relative_eq!(preds[1], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_scaling_scalar_divides_in_place() {
        let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
        let subj = one_subject_for_scaling();
        let mut preds = vec![1000.0, 2000.0, 3000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        for (got, exp) in preds.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert_relative_eq!(got, exp, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_apply_scaling_marks_bad_scale_as_nan() {
        // When a ScaleFn returns 0 / NaN / inf at runtime, apply_scaling
        // replaces every prediction with NaN so the outer NLL goes NaN
        // and the optimizer rejects the step. Loud failure mode (vs the
        // earlier "skip silently" policy, which hid mis-scaled fits).
        let bad_returns = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY];
        for &val in &bad_returns {
            let scale_fn: crate::types::ScaleFn = Box::new(move |_, _, _, _| val);
            let model = model_with_scaling(ScalingSpec::ExpressionScale {
                scale_fn,
                deriv: None,
            });
            let subj = one_subject_for_scaling();
            let mut preds = vec![1.0, 2.0, 3.0];
            apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
            for p in &preds {
                assert!(
                    p.is_nan(),
                    "bad scale ({}) must produce NaN preds, got {}",
                    val,
                    p
                );
            }
        }
    }

    #[test]
    fn test_apply_scaling_scalar_bad_value_marks_nan() {
        // Defensive guard on the ScalarScale branch (hand-constructed
        // models bypass the parser's > 0 check).
        for &k in &[0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY] {
            let model = model_with_scaling(ScalingSpec::ScalarScale(k));
            let subj = one_subject_for_scaling();
            let mut preds = vec![1.0, 2.0, 3.0];
            apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
            for p in &preds {
                assert!(
                    p.is_nan(),
                    "bad scalar scale ({}) must produce NaN preds, got {}",
                    k,
                    p
                );
            }
        }
    }

    #[test]
    fn test_apply_scaling_expression_uses_covariate() {
        // scale_fn = WT / 70 — divisive convention so preds / (WT/70).
        let scale_fn: crate::types::ScaleFn =
            Box::new(|_theta, _eta, cov: &HashMap<String, f64>, _pk: &PkParams| {
                cov.get("WT").copied().unwrap_or(70.0) / 70.0
            });
        let model = model_with_scaling(ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        });
        let mut subj = one_subject_for_scaling();
        subj.covariates.insert("WT".to_string(), 84.0); // scale = 84/70 = 1.2
        let mut preds = vec![12.0, 24.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 10.0, epsilon = 1e-12); // 12 / 1.2
        assert_relative_eq!(preds[1], 20.0, epsilon = 1e-12); // 24 / 1.2
    }

    #[test]
    fn test_apply_scaling_expression_uses_pk_param() {
        // Regression for Phase 1.5 lift: scale_fn can now read individual
        // parameters from `pk`. Here `scale = V / 10` — V is at PK slot 1
        // (PK_IDX_V) for the cl_from_cr model fixture (CL=slot 0, V=slot 1).
        // The model sets V = 50 so scale = 5 and preds get divided by 5.
        let scale_fn: crate::types::ScaleFn =
            Box::new(|_theta, _eta, _cov: &HashMap<String, f64>, pk: &PkParams| {
                pk.values[crate::types::PK_IDX_V] / 10.0
            });
        let model = model_with_scaling(ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        });
        let subj = one_subject_for_scaling();
        let mut preds = vec![100.0, 50.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        // cl_from_cr_model sets V = 50, so scale = 5.
        assert_relative_eq!(preds[0], 20.0, epsilon = 1e-12);
        assert_relative_eq!(preds[1], 10.0, epsilon = 1e-12);
    }

    // ── PerCmt dispatch ────────────────────────────────────────────────────

    #[test]
    fn test_apply_scaling_per_cmt_dispatches_per_observation() {
        // Two observations on different CMTs get scaled by different
        // factors. CMT=1 → /1000, CMT=2 → /1.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(1.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        // The helper produces 3 obs all on CMT=1; override one to CMT=2.
        assert!(subj.obs_cmts.len() >= 2);
        subj.obs_cmts[0] = 1;
        subj.obs_cmts[1] = 2;
        subj.obs_cmts[2] = 1;

        let mut preds = vec![1000.0, 50.0, 2000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12); // 1000 / 1000
        assert_relative_eq!(preds[1], 50.0, epsilon = 1e-12); // 50 / 1
        assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12); // 2000 / 1000
    }

    #[test]
    fn test_apply_scaling_per_cmt_missing_cmt_yields_nan() {
        // PerCmt map covers CMT=1 only, but obs[1] is on CMT=2.
        // Pre-fit validation catches this at fit() entry, but apply_scaling
        // is the defensive last line — must produce NaN, not silent zero.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts[0] = 1;
        subj.obs_cmts[1] = 2; // not in map → NaN
        subj.obs_cmts[2] = 1;

        let mut preds = vec![1000.0, 50.0, 2000.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12);
        assert!(preds[1].is_nan(), "missing CMT must produce NaN");
        assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_validate_per_cmt_scaling_errors_on_missing_entry() {
        // The pre-fit validation should produce a clear error naming the
        // missing CMTs rather than relying on the runtime NaN fallback.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts = vec![1, 2, 3]; // 2 and 3 missing from map

        let err = validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect_err("missing CMTs must error");
        assert!(
            err.contains("[2, 3]"),
            "error should list missing CMTs explicitly, got: {}",
            err
        );
    }

    #[test]
    fn test_validate_per_cmt_scaling_passes_when_complete() {
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(1.0));
        let model = model_with_scaling(ScalingSpec::PerCmt(map));

        let mut subj = one_subject_for_scaling();
        subj.obs_cmts = vec![1, 2, 1];

        validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect("complete PerCmt coverage must pass validation");
    }

    #[test]
    fn test_validate_per_cmt_scaling_noop_for_non_per_cmt() {
        // Non-PerCmt scaling never errors regardless of observed CMTs.
        let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
        let subj = one_subject_for_scaling();
        validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
            .expect("non-PerCmt scaling never errors");
    }

    // ── build_obs_scale_array (Phase 2.5: shared FD/AD materialiser) ───────

    #[test]
    fn test_build_obs_scale_array_none() {
        let spec = ScalingSpec::None;
        let pk = PkParams::default();
        let cov = HashMap::new();
        let obs_cmts = vec![1, 1, 2];
        let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
        assert_eq!(arr, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_build_obs_scale_array_scalar() {
        let spec = ScalingSpec::ScalarScale(1000.0);
        let pk = PkParams::default();
        let cov = HashMap::new();
        let obs_cmts = vec![1; 4];
        let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
        assert_eq!(arr, vec![1000.0; 4]);
    }

    #[test]
    fn test_build_obs_scale_array_scalar_bad_value_yields_nan() {
        // Defensive: hand-constructed ScalarScale with invalid k materialises
        // NaN — feeds the FD path's loud-failure semantic and the AD path
        // gets a Const NaN that propagates to NaN gradient too.
        for &k in &[0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let spec = ScalingSpec::ScalarScale(k);
            let pk = PkParams::default();
            let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
            assert!(arr.iter().all(|s| s.is_nan()), "bad k={} → all NaN", k);
        }
    }

    #[test]
    fn test_build_obs_scale_array_expression() {
        // Closure returns 5.0; array should be 5.0 for every obs.
        let scale_fn: crate::types::ScaleFn = Box::new(|_, _, _, _| 5.0);
        let spec = ScalingSpec::ExpressionScale {
            scale_fn,
            deriv: None,
        };
        let pk = PkParams::default();
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
        assert_eq!(arr, vec![5.0; 3]);
    }

    #[test]
    fn test_build_obs_scale_array_per_cmt_dispatches() {
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        map.insert(2, ScalingSpec::ScalarScale(2.0));
        let spec = ScalingSpec::PerCmt(map);
        let pk = PkParams::default();
        let obs_cmts = vec![1, 2, 1, 2];
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
        assert_eq!(arr, vec![1000.0, 2.0, 1000.0, 2.0]);
    }

    #[test]
    fn test_build_obs_scale_array_per_cmt_missing_cmt_is_nan() {
        // CMT=3 is observed but not in the map. The runtime fit-time
        // validation should catch this earlier; if it doesn't (hand-built
        // CompiledModel), the array materialises NaN at the missing slot.
        let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
        map.insert(1, ScalingSpec::ScalarScale(1000.0));
        let spec = ScalingSpec::PerCmt(map);
        let pk = PkParams::default();
        let obs_cmts = vec![1, 3, 1];
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
        assert_eq!(arr[0], 1000.0);
        assert!(arr[1].is_nan());
        assert_eq!(arr[2], 1000.0);
    }

    // ── Mixed bolus + infusion within a single subject (issue #176) ─────────
    //
    // Regression guard for the silent-wrong-answer bug on the superposition
    // path: before #176, `single_dose_concentration` branched on the model
    // enum (`OneCptIvBolus` vs `OneCptInfusion`), so an infusion event under
    // an IV-bolus model was treated as an instantaneous bolus of `AMT`,
    // ignoring the duration entirely. After #176 the dispatch picks per
    // dose from `dose.is_infusion()`, so both routes coexist in one record.

    #[test]
    fn test_mixed_bolus_and_infusion_in_single_subject_superposition() {
        // Same `OneCptIv` model, two doses with different administration:
        //   t=0:  bolus     AMT=100, RATE=0
        //   t=4:  infusion  AMT=500, RATE=50 → duration=10
        // The superposition prediction must equal the sum of the two
        // single-dose closed forms (one bolus, one infusion).
        let bolus = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let infusion = DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0);
        debug_assert!(!bolus.is_infusion() && infusion.is_infusion());
        debug_assert!((infusion.duration - 10.0).abs() < 1e-12);
        let doses = vec![bolus.clone(), infusion.clone()];
        let pk = make_pk_params(10.0, 100.0);

        // Probe times that fall in three regimes:
        //   t = 2.0  → only the bolus has been given
        //   t = 8.0  → both active, infusion still running
        //   t = 20.0 → infusion has ended, post-infusion decay
        for &t in &[2.0, 8.0, 20.0] {
            let combined = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
            let from_bolus = predict_concentration(PkModel::OneCptIv, &[bolus.clone()], t, &pk);
            let from_infusion =
                predict_concentration(PkModel::OneCptIv, &[infusion.clone()], t, &pk);
            assert_relative_eq!(combined, from_bolus + from_infusion, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_mixed_dose_superposition_matches_event_driven_path() {
        // Cross-check: the analytical superposition path and the
        // event-driven analytical path must agree to roundoff on a
        // mixed-administration record. Before #176 the superposition path
        // was silently wrong (bolus formula applied to the infusion dose),
        // while the event-driven path was already correct — so these two
        // would have disagreed materially.
        use crate::pk::event_driven::event_driven_predictions;
        use crate::types::Subject;

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // bolus
            DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0), // infusion (dur=10)
        ];
        let obs_times = vec![1.0_f64, 3.0, 6.0, 10.0, 14.0, 20.0];
        let pk = make_pk_params(10.0, 100.0);

        let analytical: Vec<f64> = obs_times
            .iter()
            .map(|&t| predict_concentration(PkModel::OneCptIv, &doses, t, &pk))
            .collect();

        let n_obs = obs_times.len();
        let subject = Subject {
            id: "mixed".into(),
            doses: doses.clone(),
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let pk_dose: Vec<PkParams> = vec![pk.clone(); doses.len()];
        let pk_obs: Vec<PkParams> = vec![pk.clone(); obs_times.len()];
        let event_driven =
            event_driven_predictions(PkModel::OneCptIv, &subject, &pk_dose, &pk_obs, &[]);

        for (i, (&a, &e)) in analytical.iter().zip(event_driven.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff < 1e-10,
                "mismatch at obs[{i}] t={}: superposition {a} vs event-driven {e} (|Δ|={diff})",
                obs_times[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for single_dose_states SS dispatch
    // -----------------------------------------------------------------------

    /// Bug regression: single_dose_states must use the SS formula for the
    /// central compartment when dose.ss = true.  Before the fix, oral and
    /// IV-infusion SS doses fell through to single-dose formulas, returning
    /// concentrations that were typically 5-15× too low at trough.
    ///
    /// For each model the test checks that the SS central state matches
    /// single_dose_concentration (the long-standing, validated scalar path).
    #[test]
    fn single_dose_states_central_matches_concentration_for_ss_doses() {
        use crate::pk::{
            one_cpt_infusion_ss, one_cpt_iv_bolus_ss, one_cpt_oral_f_ss, two_cpt_infusion_ss,
            two_cpt_iv_bolus_ss, two_cpt_oral_f_ss,
        };
        // --- helpers ---
        fn bolus_ss(amt: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 5.0_f64;
        let v = 80.0_f64;
        let q = 2.0_f64;
        let v2 = 40.0_f64;
        let ka = 1.0_f64;
        let f_bio = 1.0_f64;
        let ii = 24.0_f64;
        let amt = 100.0_f64;
        let rate = 50.0_f64; // 2 h infusion
        let tau = 6.0_f64;

        let mut p1iv = PkParams::default();
        p1iv.values[crate::types::PK_IDX_CL] = cl;
        p1iv.values[crate::types::PK_IDX_V] = v;

        let mut p1oral = PkParams::default();
        p1oral.values[crate::types::PK_IDX_CL] = cl;
        p1oral.values[crate::types::PK_IDX_V] = v;
        p1oral.values[crate::types::PK_IDX_KA] = ka;
        p1oral.values[crate::types::PK_IDX_F] = f_bio;

        let mut p2iv = PkParams::default();
        p2iv.values[crate::types::PK_IDX_CL] = cl;
        p2iv.values[crate::types::PK_IDX_V] = v;
        p2iv.values[crate::types::PK_IDX_Q] = q;
        p2iv.values[crate::types::PK_IDX_V2] = v2;

        let mut p2oral = PkParams::default();
        p2oral.values[crate::types::PK_IDX_CL] = cl;
        p2oral.values[crate::types::PK_IDX_V] = v;
        p2oral.values[crate::types::PK_IDX_Q] = q;
        p2oral.values[crate::types::PK_IDX_V2] = v2;
        p2oral.values[crate::types::PK_IDX_KA] = ka;
        p2oral.values[crate::types::PK_IDX_F] = f_bio;

        // OneCptIv SS bolus — central index 0
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
        let expected = one_cpt_iv_bolus_ss(&d, tau, cl, v);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "OneCptIv SS bolus: central mismatch"
        );

        // OneCptIv SS infusion — was using non-SS formula before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
        let expected = one_cpt_infusion_ss(&d, tau, cl, v);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "OneCptIv SS infusion: central mismatch"
        );

        // OneCptOral SS bolus — was using one_cpt_oral_f (non-SS) before fix
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p1oral);
        let expected = one_cpt_oral_f_ss(&d, tau, cl, v, ka, f_bio);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "OneCptOral SS: central (index 1) mismatch"
        );

        // TwoCptIv SS bolus
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
        let expected = two_cpt_iv_bolus_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "TwoCptIv SS bolus: central mismatch"
        );

        // TwoCptIv SS infusion — was using two_cpt_infusion (non-SS) before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
        let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[0], expected, max_relative = 1e-9),
            "TwoCptIv SS infusion: central mismatch"
        );

        // TwoCptOral SS bolus — was using two_cpt_oral_f (non-SS) before fix
        let d = bolus_ss(amt, ii);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
        let expected = two_cpt_oral_f_ss(&d, tau, cl, v, q, v2, ka, f_bio);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "TwoCptOral SS bolus: central (index 1) mismatch"
        );

        // TwoCptOral SS infusion central — was using two_cpt_infusion (non-SS) before fix
        let d = infusion_ss(amt, rate, ii);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
        let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
        assert!(
            approx::relative_eq!(states[1], expected, max_relative = 1e-9),
            "TwoCptOral SS infusion: central (index 1) mismatch"
        );

        // SS states must exceed single-dose states (accumulation guard).
        let d_single = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let states_ss = single_dose_states(PkModel::OneCptOral, &bolus_ss(amt, ii), tau, &p1oral);
        let states_sd = single_dose_states(PkModel::OneCptOral, &d_single, tau, &p1oral);
        assert!(
            states_ss[1] > states_sd[1],
            "SS central must be > single-dose central (accumulation): ss={} sd={}",
            states_ss[1],
            states_sd[1]
        );
    }

    /// Transit is a 2-state `[depot, central]` model: `single_dose_states` must return the
    /// lumped unabsorbed-chain "depot" amount and the central concentration for a bolus
    /// (the `[derived]` / compartment-amount path), #386.
    #[test]
    fn single_dose_states_transit_returns_depot_and_central() {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 0.5;
        p.values[crate::types::PK_IDX_V] = 10.0;
        p.values[crate::types::PK_IDX_N] = 3.0;
        p.values[crate::types::PK_IDX_MTT] = 1.5;
        let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let states = single_dose_states(PkModel::OneCptTransit, &d, 2.0, &p);
        assert_eq!(states.len(), 2, "transit exposes [depot, central]");
        assert!(
            states[0] > 0.0 && states[1] > 0.0,
            "both amounts positive mid-absorption: {states:?}"
        );
    }

    /// Regression: `single_dose_concentration` and `single_dose_states[central]` must
    /// agree for OneCptOral infusion doses.  Before the fix both functions used the
    /// oral Bateman formula for infusions (which bypass the depot), so both returned
    /// a consistent but wrong value.  After the fix both route to the IV infusion
    /// formula, keeping ipred == compartment_states[central] and matching NONMEM.
    #[test]
    fn single_dose_concentration_matches_states_for_one_cpt_oral_infusion() {
        fn infusion(amt: f64, rate: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 3.0_f64;
        let v = 50.0_f64;
        let ka = 1.2_f64; // ka present but must NOT affect infusion formula
        let f_bio = 0.9_f64;
        let ii = 12.0_f64;
        let amt = 100.0_f64;
        let rate = 50.0_f64; // 2 h infusion
        let tau = 5.0_f64;

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = f_bio;

        // Non-SS infusion: central (index 1) must equal single_dose_concentration
        let d = infusion(amt, rate);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral non-SS infusion: concentration={conc} ≠ central state={} \
             (depot is index 0, central is index 1)",
            states[1]
        );
        // Depot must be zero for an infusion (bypasses depot)
        assert_eq!(states[0], 0.0, "OneCptOral infusion: depot must be 0.0");

        // SS infusion
        let d = infusion_ss(amt, rate, ii);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );
        assert_eq!(states[0], 0.0, "OneCptOral SS infusion: depot must be 0.0");

        // Bolus must be unchanged (oral formula correct for bolus)
        let d_bolus = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let conc = single_dose_concentration(PkModel::OneCptOral, &d_bolus, tau, &p);
        let states = single_dose_states(PkModel::OneCptOral, &d_bolus, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "OneCptOral bolus: concentration={conc} ≠ central state={}",
            states[1]
        );
        assert!(states[0] > 0.0, "OneCptOral bolus: depot must be > 0");
    }

    /// Regression test: `single_dose_concentration` and `single_dose_states[central]`
    /// must agree for TwoCptOral infusion doses (both SS and non-SS).
    ///
    /// Before the fix, `single_dose_concentration` used the oral formula
    /// (`two_cpt_oral_f_ss` / `two_cpt_oral_f`) for infusion doses, while
    /// `single_dose_states` correctly used the IV infusion formula (since infusions
    /// bypass the depot).  This caused `ipred ≠ compartment_states[central]` for
    /// any TwoCptOral model with infusion doses.
    #[test]
    fn single_dose_concentration_matches_states_for_two_cpt_oral_infusion() {
        fn infusion(amt: f64, rate: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
        }
        fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
            DoseEvent::new(0.0, amt, 1, rate, true, ii)
        }

        let cl = 5.0_f64;
        let v = 80.0_f64;
        let q = 2.0_f64;
        let v2 = 40.0_f64;
        let ka = 1.2_f64;
        let f_bio = 1.0_f64;
        let ii = 24.0_f64;
        let amt = 200.0_f64;
        let rate = 100.0_f64; // 2 h infusion
        let tau = 8.0_f64;

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = f_bio;

        // Non-SS infusion: single_dose_concentration must equal central state
        let d = infusion(amt, rate);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        // central is index 1 for TwoCptOral: [depot=0, central=1, periph=2]
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral non-SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );

        // SS infusion
        let d = infusion_ss(amt, rate, ii);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral SS infusion: concentration={conc} ≠ central state={}",
            states[1]
        );

        // Bolus should be unchanged (oral formula is correct for bolus)
        let d = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
        let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
        let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
        assert!(
            approx::relative_eq!(conc, states[1], max_relative = 1e-9),
            "TwoCptOral bolus: concentration={conc} ≠ central state={}",
            states[1]
        );
    }

    /// Closes PR #327 review finding #5: bioavailability `F` must scale **every**
    /// element of the `single_dose_states` vector — depot, central, and all
    /// peripheral compartments — on every route. The analytical states are linear
    /// in the dose, so the invariant is `states(F) == F · states(F=1)`
    /// element-wise. This pins down what the central-only
    /// `single_dose_concentration` checks cannot see: a depot wrongly scaled, a
    /// scale missed on a peripheral arm, or `F` applied twice. `compartment_states`
    /// feeds `[derived]`/`[output]` expressions, so a silent error here would
    /// surface in user output.
    #[test]
    fn single_dose_states_scales_every_compartment_by_f() {
        let f = 0.4_f64;
        let tau = 6.0_f64;
        let ii = 12.0_f64;
        let amt = 100.0_f64;

        // Full structural params for up to three compartments; F set per call.
        let params = |f_bio: f64| {
            let mut p = PkParams::default();
            p.values[crate::types::PK_IDX_CL] = 3.0;
            p.values[crate::types::PK_IDX_V] = 50.0;
            p.values[crate::types::PK_IDX_Q] = 2.0;
            p.values[crate::types::PK_IDX_V2] = 40.0;
            p.values[crate::types::PK_IDX_KA] = 1.1;
            p.values[crate::types::PK_IDX_Q3] = 1.5;
            p.values[crate::types::PK_IDX_V3] = 30.0;
            p.values[crate::types::PK_IDX_F] = f_bio;
            p
        };
        let p1 = params(1.0);
        let pf = params(f);

        let models = [
            PkModel::OneCptIv,
            PkModel::OneCptOral,
            PkModel::TwoCptIv,
            PkModel::TwoCptOral,
            PkModel::ThreeCptIv,
            PkModel::ThreeCptOral,
        ];
        // Bolus into compartment 1, non-SS and SS. `F` scales the bioavailable
        // AMOUNT (`F·AMT` into central, or the oral depot load), so it scales
        // every compartment's state uniformly. Infusions are excluded here: for a
        // rate-defined infusion `F` scales the *duration* (#419), which reshapes
        // the profile rather than scaling it - that path is anchored by
        // `iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form` and the
        // cross-path equality tests instead.
        let doses = [
            ("bolus", DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)),
            ("SS bolus", DoseEvent::new(0.0, amt, 1, 0.0, true, ii)),
        ];

        for model in models {
            for (label, dose) in &doses {
                let s1 = single_dose_states(model, dose, tau, &p1);
                let sf = single_dose_states(model, dose, tau, &pf);
                assert_eq!(
                    s1.len(),
                    sf.len(),
                    "{model:?} {label}: state-vector length must not depend on F"
                );
                // Guard against a vacuous pass (all-zero states scale trivially).
                assert!(
                    s1.iter().any(|&x| x.abs() > 1e-9),
                    "{model:?} {label}: all states ~0 at F=1 — test would be vacuous"
                );
                for (i, (&a, &b)) in s1.iter().zip(sf.iter()).enumerate() {
                    assert!(
                        approx::relative_eq!(b, f * a, max_relative = 1e-9, epsilon = 1e-12),
                        "{model:?} {label}: state[{i}] = {b} at F={f}, expected {} = F·{a} \
                         — F must scale every compartment exactly once",
                        f * a
                    );
                }
            }
        }
    }
}
