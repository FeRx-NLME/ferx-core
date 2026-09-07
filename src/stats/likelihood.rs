use crate::pk;
use crate::stats::residual_error::{residual_rd, residual_rd2};
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
        // The model's declared `block_sigma` correlations. Every caller of this
        // wrapper (SAEM, HMC, VI, AGQ) holds ρ fixed at the declaration, so that
        // *is* their live value; FOCE/FOCEI, which estimate it (#847), call
        // `individual_nll_into_with_schedule` directly with the optimizer's ρ.
        &model.residual_correlations,
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
        } = r
        else {
            // Non-TTE records don't contribute hazard times; skip them.
            continue;
        };
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
    // NaN curves, which `tte_nll_from_curves` maps to the 1e20 sentinel).
    let (cum, haz) =
        crate::survival::ode_cumhaz_hazard(model, subject, chz_state, theta, eta, &times);
    // Tie the CHZ monotonicity floor to this model's *configured* solver tolerances
    // (#618). A missing ODE yields NaN curves -> sentinel regardless, so the default is
    // immaterial there.
    let tol = model
        .ode_spec
        .as_ref()
        .map(|o| crate::survival::MonoTol::from_solver(&o.effective_solver_opts()))
        .unwrap_or_default();
    tte_ode_nll_from_curves(records, &times, &cum, &haz, tol)
}

/// Per-record TTE NLL from cumulative-hazard / hazard curves already sampled at
/// `times` (sorted-unique). The lookup closures binary-search `times`, exact on the
/// f64 values that populated it. Shared by [`tte_ode_nll`] (which solves the ODE to
/// build `cum`/`haz`) and the #570 joint-PK-TTE inner-NLL share (which reads `H`/`h`
/// off the Gaussian solve), so the per-record likelihood logic lives in one place.
#[cfg(feature = "survival")]
fn tte_ode_nll_from_curves(
    records: &[ObsRecord],
    times: &[f64],
    cum: &[f64],
    haz: &[f64],
    tol: crate::survival::MonoTol,
) -> f64 {
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
    crate::survival::tte_nll_from_curves(records, cumhaz_at, hazard_at, tol)
}

/// #570: a single augmented PK+CHZ integration reused for both endpoints of a joint
/// PK-TTE subject — the scaled Gaussian predictions and the cumulative-hazard state
/// at the TTE event/censor/entry times — so the inner NLL no longer integrates the
/// same system a second time via `ode_cumhaz_hazard`.
#[cfg(feature = "survival")]
pub(crate) struct JointPkTteSolve {
    /// Scaled Gaussian predictions — bit-identical to the standalone prediction path.
    preds: Vec<f64>,
    /// Sorted-unique union of every OdeAccumulated endpoint's record times.
    times: Vec<f64>,
    /// Full ODE state at each `times[i]` — the seeded initial state for a time before the
    /// integration start (#1223), NaN only where the solve diverged.
    chz_states: Vec<Vec<f64>>,
    /// PK-parameter snapshot used for the solve — reused to evaluate `h = dCHZ/dt`.
    pk_values: Vec<f64>,
}

/// Build the shared joint PK-TTE solve, but only when it is provably equivalent to
/// the separate Gaussian + `ode_cumhaz_hazard` solves: a plain ODE model with at
/// least one OdeAccumulated hazard and none of the features that route the Gaussian
/// path away from the no-TV `ode_predictions` (time-varying covariates, EVID-3/4
/// resets, SDE diffusion, FREM pseudo-observations). Returns `None` otherwise, and the
/// caller keeps the established two-solve path.
#[cfg(feature = "survival")]
fn try_joint_pktte_shared_solve(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<JointPkTteSolve> {
    let ode = model.ode_spec.as_ref()?;
    // The share computes the Gaussian predictions via `ode_predictions_and_chz`, i.e.
    // the plain no-TV `ode_predictions` with a single t=0 PK snapshot. It is only
    // equivalent when the standalone prediction path (`compute_predictions_with_tv_*`)
    // takes that same snapshot route. Reject every case it routes elsewhere: model time
    // (`pk::model_uses_time_anywhere`) or time-varying covariates send it through the
    // per-event `ode_predictions_event_driven` (pk/mod.rs); EVID-3/4 resets also need the
    // event-driven walker; SDE adds EKF process noise; FREM rewrites the
    // pseudo-observation predictions. Each keeps the established two-solve fallback.
    //
    // The model-time clause must be the **wide** predicate, matching the routing site
    // this mirrors (`pk/mod.rs`). Asking the narrow `compiled_model_uses_time_builtin`
    // inspects only the individual-parameter program, so an `[odes]` RHS reading
    // `TAD`/`TAFD`/`T`/`TIME` stayed admitted here after #1124 rerouted the standalone
    // path — leaving the objective's Gaussian term on the dense engine while the
    // reported IPRED came from the event-driven one, which is exactly the split the
    // stated contract above exists to prevent.
    if subject.obs_records.is_empty()
        || subject.has_tv_covariates()
        || subject.has_resets()
        || crate::pk::model_uses_time_anywhere(model)
        || model.is_sde()
        || subject.fremtype.iter().any(|&f| f > 0)
    {
        return None;
    }
    let is_ode_tte = |e: &EndpointLikelihood| {
        matches!(
            e,
            EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { .. },
                ..
            }
        )
    };
    if !model.endpoints.values().any(is_ode_tte) {
        return None; // nothing to share
    }

    // Union of every OdeAccumulated endpoint's record times (event / entry / interval
    // bounds), sorted-unique — the soft sample points fed to the Gaussian solve.
    let mut times: Vec<f64> = Vec::new();
    for (cmt, endpoint) in &model.endpoints {
        if !is_ode_tte(endpoint) {
            continue;
        }
        for r in &subject.obs_records {
            let ObsRecord::Event {
                time,
                event_type,
                entry_time,
                cmt: rc,
            } = r
            else {
                // Non-TTE records don't contribute hazard times; skip them.
                continue;
            };
            if rc != cmt {
                continue;
            }
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
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("TTE times are finite"));
    times.dedup();

    // One augmented integration: Gaussian observable readout + CHZ at `times`.
    // `pk_param_fn`'s 4th arg is the evaluation time (#591); the share is guarded to
    // no-TV subjects, so a single snapshot at t=0 matches `ode_cumhaz_hazard`'s call.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let (mut preds, chz_states) =
        crate::ode::ode_predictions_and_chz(ode, &pk.values, theta, eta, subject, &times);
    // Apply exactly the post-processing the standalone no-TV ODE prediction path does
    // (compute_predictions_with_tv_into_with_schedule): init impulse, [scaling], LTBS.
    // No-op for Form-C / no-init / linear-scale models; FREM is excluded by the guard.
    crate::pk::add_analytical_init(model, subject, theta, eta, &mut preds);
    crate::pk::apply_scaling(model, subject, theta, eta, &mut preds);
    crate::pk::apply_log_transform(model, &mut preds);

    Some(JointPkTteSolve {
        preds,
        times,
        chz_states,
        pk_values: pk.values.to_vec(),
    })
}

/// TTE NLL for an OdeAccumulated endpoint, reading `H`/`h` from the shared joint solve
/// instead of integrating again. `cum = CHZ` and `haz = dCHZ/dt` are read exactly as
/// `ode_cumhaz_hazard` does (state slot `chz_state`; the bare RHS for the derivative);
/// the only difference is that the state came from the Gaussian solve's Hermite
/// read-back, equal to a dedicated clamped solve to solver tolerance. A time before the
/// integration start carries the seeded initial state on both engines (#1223), so it
/// scores `H = 0`; a NaN state means a diverged solve and stays NaN, which
/// `tte_nll_from_curves` maps to the `1e20` sentinel — matching `ode_cumhaz_hazard`.
#[cfg(feature = "survival")]
fn tte_ode_nll_from_shared(
    ode: &crate::ode::OdeSpec,
    share: &JointPkTteSolve,
    chz_state: usize,
    records: &[ObsRecord],
) -> f64 {
    let n = share.times.len();
    let mut cum = vec![f64::NAN; n];
    let mut haz = vec![f64::NAN; n];
    let mut du = vec![0.0; ode.n_states];
    for (i, &t) in share.times.iter().enumerate() {
        let st = &share.chz_states[i];
        if st.len() != ode.n_states || st.iter().any(|x| !x.is_finite()) {
            continue;
        }
        cum[i] = st[chz_state];
        (ode.rhs)(st, &share.pk_values, t, &mut du);
        haz[i] = du[chz_state];
    }
    let tol = crate::survival::MonoTol::from_solver(&ode.effective_solver_opts());
    tte_ode_nll_from_curves(records, &share.times, &cum, &haz, tol)
}

/// Dispatch a TTE endpoint's per-subject NLL on its hazard representation: the
/// closed-form [`crate::survival::tte_data_term`] for `Analytic`, or the
/// ODE-accumulated [`tte_ode_nll`] for joint PK-TTE.
#[cfg(feature = "survival")]
fn tte_endpoint_nll(
    model: &CompiledModel,
    subject: &Subject,
    hazard: &HazardSpec,
    recurrence: crate::types::TteRecurrence,
    records: &[ObsRecord],
    theta: &[f64],
    eta: &[f64],
) -> f64 {
    match hazard {
        HazardSpec::Analytic { .. } => crate::survival::tte_data_term(
            records,
            hazard,
            recurrence,
            theta,
            eta,
            &subject.covariates,
        ),
        // Recurrence is orthogonal to the ODE-accumulated path, which is Single-only
        // for now (RTTE + drug-driven hazard is rejected at parse — a later slice).
        HazardSpec::OdeAccumulated { chz_state } => {
            tte_ode_nll(model, subject, *chz_state, records, theta, eta)
        }
    }
}

/// Accumulate every non-Gaussian endpoint's data-term NLL into `acc`, applying `factor` to
/// each contribution. This is the single seam the FOCEI, SAEM θ-M-step, and IOV likelihood
/// paths share: TTE endpoints are scored by [`tte_endpoint_nll`] (an `OdeAccumulated`
/// joint-PK-TTE hazard reads `H`/`h` off the #570 shared solve when `joint_share` is `Some`);
/// the discrete families — binary today, ordinal / Poisson / negative-binomial later — by
/// [`crate::categorical::discrete_subject_nll`]. A new endpoint family is enabled by one arm
/// in those two callees, touching no dispatch site.
///
/// It accumulates **into `acc` per term** — one `factor * term` add per TTE endpoint (in
/// `model.endpoints` iteration order), then one for the discrete term — rather than returning
/// a folded sum, so a running `data_ll` that already holds the Gaussian residual term stays
/// **bit-identical** to the previously-inlined loop. Returning `-> f64` and adding
/// `factor * sum` once would reassociate the floating-point adds and perturb the FOCEI hot
/// path for multi-endpoint (competing-risks / TTE+binary) subjects.
///
/// `factor` is the site's OFV-scale weight (`2·` on the halved FOCEI/IOV `data_ll`, `1·` on
/// the raw SAEM / non-interaction NLL). `joint_share` is `None` on every path that does not
/// build the shared solve, and passing it is faithful to what each inlined:
/// - the two **IOV** sites (`individual_nll_iov`, `obs_nll_subject_into_iov`) inlined
///   `tte_data_term`, which equals `tte_endpoint_nll` here because an IOV model's hazard is
///   always `Analytic` — IOV + joint-PK-TTE is rejected at parse;
/// - the **SAEM θ-M-step** (`obs_nll_subject_from_preds`) already inlined `tte_endpoint_nll`
///   and *may* carry an `OdeAccumulated` endpoint (SAEM + joint-PK-TTE is not rejected); the
///   `_` arm re-solves it via `tte_ode_nll`, exactly as that site did before the fold.
#[cfg(feature = "survival")]
pub(crate) fn accumulate_non_gaussian_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    joint_share: Option<&JointPkTteSolve>,
    factor: f64,
    acc: &mut f64,
) {
    // Callers gate on `!subject.obs_records.is_empty()`, so an empty subject never reaches
    // here; both loops below are also natural no-ops on one.
    // TTE endpoints — one `factor * raw` add each, in endpoints iteration order.
    for (cmt, endpoint) in &model.endpoints {
        if let EndpointLikelihood::Tte {
            hazard, recurrence, ..
        } = endpoint
        {
            let records_for_cmt: Vec<crate::types::ObsRecord> = subject
                .obs_records
                .iter()
                .filter(|r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt))
                .cloned()
                .collect();
            if records_for_cmt.is_empty() {
                continue; // subject has no records for this TTE CMT
            }
            let raw = match (joint_share, hazard) {
                (Some(s), HazardSpec::OdeAccumulated { chz_state }) => tte_ode_nll_from_shared(
                    model.ode_spec.as_ref().expect("joint share ⟹ ode_spec"),
                    s,
                    *chz_state,
                    &records_for_cmt,
                ),
                _ => tte_endpoint_nll(
                    model,
                    subject,
                    hazard,
                    *recurrence,
                    &records_for_cmt,
                    theta,
                    eta,
                ),
            };
            *acc += factor * raw;
        }
    }
    // Discrete families (binary today; ordinal/Poisson/… later) — one add, matching the TTE terms.
    *acc += factor * crate::categorical::discrete_subject_nll(model, subject, theta, eta);
    // CTMM (#759): continuous-time Markov endpoint, same per-endpoint add.
    #[cfg(feature = "markov")]
    {
        *acc += factor * crate::markov::endpoint::ctmm_subject_nll(model, subject, theta, eta);
    }
}

/// Hot-path variant that additionally threads through a pre-built
/// [`pk::event_driven::EventSchedule`]. The FOCE inner-loop obj closure
/// and Jacobian build the schedule once per `find_ebe` call and reuse
/// it across all BFGS iterations.
///
/// `residual_correlations` carries the **live** `block_sigma` off-diagonals the
/// way `sigma_values` carries the live σ (#847). FOCE/FOCEI estimate ρ, so their
/// inner loop must pass the current optimizer value rather than the model's
/// declaration; an estimator that holds ρ fixed passes
/// `&model.residual_correlations`, which is then the same thing.
pub fn individual_nll_into_with_schedule(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation],
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);

    // Eta prior: eta' * Omega_inv * eta
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(omega_inv * &eta_vec));

    // #570: a joint PK-TTE subject otherwise integrates the augmented PK+CHZ system
    // twice per eval (Gaussian preds, then `ode_cumhaz_hazard` for `H`/`h`). When the
    // model qualifies, do it once: the predictions are bit-identical and CHZ is read
    // off the same solve by Hermite interpolation. `None` ⇒ the established path.
    #[cfg(feature = "survival")]
    let joint_share = try_joint_pktte_shared_solve(model, subject, theta, eta);

    // Compute individual predictions using the caller's scratch buffer
    // for per-event PK params (only consumed on the TV-cov path; ignored
    // on the no-TV fast path).
    #[cfg(feature = "survival")]
    let preds = match &joint_share {
        Some(s) => s.preds.clone(),
        None => model_predictions_into_with_schedule(model, subject, theta, eta, scratch, schedule),
    };
    #[cfg(not(feature = "survival"))]
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
    if !residual_correlations.is_empty() && !has_censored_m3 && !has_frem_rows {
        match dense_residual_data_term(
            model,
            subject,
            &preds,
            sigma_values,
            residual_correlations,
            ruv_scale,
            &p_obs,
            ruv_mult.as_deref(),
        ) {
            Some(term) => data_ll += term,
            None => return 1e20,
        }
    } else {
        for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            // #905: never score a non-Gaussian-endpoint observation as Gaussian. On
            // the endpoint-routed load path (`read_population_for`) these rows live in
            // `obs_records`, so this is a no-op; on a model-blind `read_nonmem_csv`
            // load they sit in the Gaussian Vecs with a NaN `f_pred` (the analytical
            // predictor declined them, #905) and must be skipped — a `y - NaN`
            // residual would otherwise poison the whole subject's NLL. Guarded on the
            // empty-endpoints common case so plain PK models pay nothing.
            #[cfg(feature = "survival")]
            if !model.endpoints.is_empty()
                && subject
                    .obs_cmts
                    .get(j)
                    .is_some_and(|c| model.endpoints.contains_key(c))
            {
                continue;
            }
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
                err_keys[j],
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

    // Non-Gaussian data term: −log p for each TTE endpoint plus the discrete
    // (binary/categorical) endpoints, at the 2× OFV-scale (the whole objective is halved at
    // the end). #570: an OdeAccumulated endpoint reads `H`/`h` off the shared joint solve.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        accumulate_non_gaussian_nll(
            model,
            subject,
            theta,
            eta,
            joint_share.as_ref(),
            2.0,
            &mut data_ll,
        );
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
/// Builds the residual covariance `R` from
/// [`compute_r_matrix_with_correlations`](crate::stats::residual_error::compute_r_matrix_with_correlations),
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
    residual_correlations: &[ResidualCorrelation],
    ruv_scale: f64,
    p_obs: &[f64],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Option<f64> {
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut r = crate::stats::residual_error::r_matrix_maybe_scaled(
        &model.error_spec,
        preds,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_correlations,
        ruv_mult,
    );
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
/// `residual_correlations` carries the **live** `block_sigma` off-diagonals
/// alongside the live `sigma_values` (#847). Reading them off `model` instead
/// would be wrong for any estimator that inherits a fitted rho from an earlier
/// chain stage — `method = [focei, imp]` hands IMP the FOCEI estimate, so a
/// declaration-sourced R would score a different likelihood than the one just
/// optimized.
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_into(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation],
    eta: &[f64],
    pk_scratch: &mut pk::EventPkParams,
) -> f64 {
    let preds = pk::compute_predictions_with_tv_into(model, subject, theta, eta, pk_scratch);
    obs_nll_subject_from_preds(
        model,
        subject,
        &preds,
        theta,
        sigma_values,
        residual_correlations,
        eta,
    )
}

/// Observation-only NLL from *precomputed* predictions.
///
/// Identical to [`obs_nll_subject_into`] except the caller supplies `preds`
/// (predictions are independent of `σ`), letting the SAEM M-step σ-gradient FD
/// loop reuse one ODE solve across all σ perturbations instead of re-solving per
/// element (#557). `theta` is only consumed by the TTE hazard term.
#[cfg_attr(not(feature = "survival"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_from_preds(
    model: &CompiledModel,
    subject: &Subject,
    preds: &[f64],
    theta: &[f64],
    sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation],
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut nll = 0.0;
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if !residual_correlations.is_empty() && !m3 && !has_frem_rows {
        // block_sigma + SDE is rejected up front (E_BLOCK_SIGMA_SDE_UNSUPPORTED)
        // for the SAEM M-step, so no EKF process noise enters here.
        match dense_residual_data_term(
            model,
            subject,
            preds,
            sigma_values,
            residual_correlations,
            ruv_scale,
            &[],
            ruv_mult.as_deref(),
        ) {
            Some(term) => nll += 0.5 * term,
            None => return 1e20,
        }
    } else {
        for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            // #905: a non-Gaussian-endpoint observation must never be scored as
            // Gaussian — the SAEM M-step / IS twin of the skip in
            // `individual_nll_into_with_schedule`. No-op on the endpoint-routed path
            // (those rows live in `obs_records`); on a model-blind load it keeps a NaN
            // `f` (the declined predictor) out of the sum. Guarded on empty-endpoints.
            #[cfg(feature = "survival")]
            if !model.endpoints.is_empty()
                && subject
                    .obs_cmts
                    .get(j)
                    .is_some_and(|c| model.endpoints.contains_key(c))
            {
                continue;
            }
            let frem_var = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
            // FREM covariate pseudo-observations predict a covariate *value* (any
            // real: centered/standardized/log-scale covariates are routinely ≤ 0),
            // not a concentration — do NOT clamp them to 1e-12. Clamping a negative
            // covariate prediction up to 1e-12 fabricates a huge residual and, on the
            // Rao-Blackwellised path, breaks the `obs_nll(η_c=d) ≈ const` assumption
            // (#406). Ordinary PK rows keep the positivity clamp.
            let f = if frem_var.is_some() {
                f
            } else {
                model.floor_prediction(f)
            };
            let v = match frem_var {
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
                nll += -m3_logcdf(y, f, v.sqrt(), cens);
            } else {
                nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
            }
        }
    }

    // Non-Gaussian data term at raw-NLL weight (1×) so the SAEM θ M-step gradient receives
    // the TTE hazard + discrete (binary/categorical) contributions, not just Gaussian
    // residuals. No joint-share on this path (it re-solves per endpoint).
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        accumulate_non_gaussian_nll(model, subject, theta, eta, None, 1.0, &mut nll);
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

    // The EKF is a single-pass filter over the whole subject timeline with a
    // constant PK-parameter vector (no per-event recompute), so the `TIME`
    // built-in resolves at the integration start (t=0) — the same convention as
    // the single-pass ODE prediction path. (Time-varying *ODE-RHS* `TIME` is
    // still honoured: the integrator supplies its own clock.) #610.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
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
    chol_log_det(&omega.chol)
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

/// Threshold separating a healthy CTMM likelihood / FD-Hessian entry from a
/// degeneracy-sentinel-poisoned one. The CTMM data term repels with a `1e20`
/// objective (an underflowed transition or a negative off-diagonal); a mode value
/// or FD-Hessian entry at or beyond this bound is that sentinel leaking into the
/// Laplace curvature, not a real value, and is dropped rather than folded into
/// `log|H̃|`. Sits far above any legitimate CTMM curvature and far below the `1e20`
/// sentinel. See [`foce_subject_nll`]'s CTMM FD-Hessian block (#759 review).
#[cfg(feature = "markov")]
const CTMM_FD_SENTINEL_GUARD: f64 = 1e18;

/// `residual_correlations` carries the **live** `block_sigma` off-diagonals,
/// parallel to `sigma_values` (#847): FOCE/FOCEI estimate ρ, so the marginal has
/// to be scored at the optimizer's current value, not the model's declaration.
#[allow(clippy::too_many_arguments)]
pub fn foce_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation],
    interaction: bool,
) -> f64 {
    // Individual predictions at eta_hat (per-event PK when subject has TV covariates).
    // #570: at the EBE mode the Gaussian `ipreds` solve and the at-mode TTE-term solve
    // are both at `eta_hat`; when the subject qualifies, share one augmented integration
    // (the FD-Hessian's perturbed-η TTE solves below are at different η and stay separate).
    #[cfg(feature = "survival")]
    let joint_share = try_joint_pktte_shared_solve(model, subject, theta, eta_hat.as_slice());
    #[cfg(feature = "survival")]
    let ipreds = match &joint_share {
        Some(s) => s.preds.clone(),
        None => model_predictions(model, subject, theta, eta_hat.as_slice()),
    };
    #[cfg(not(feature = "survival"))]
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
            if let EndpointLikelihood::Tte {
                hazard, recurrence, ..
            } = endpoint
            {
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
                    tte_endpoint_nll(
                        model,
                        subject,
                        hazard,
                        *recurrence,
                        &records_for_cmt,
                        theta,
                        eta_eval,
                    )
                };
                // #570: at the mode, read H/h off the shared solve when available; the
                // FD-Hessian below still calls `tte_fn` at perturbed η (intrinsic to FD).
                tte_nll_at_mode += match (joint_share.as_ref(), hazard) {
                    (Some(s), HazardSpec::OdeAccumulated { chz_state }) => tte_ode_nll_from_shared(
                        model.ode_spec.as_ref().expect("joint share ⟹ ode_spec"),
                        s,
                        *chz_state,
                        &records_for_cmt,
                    ),
                    _ => tte_fn(eta_hat.as_slice()),
                };
                if n_eta > 0 {
                    let steps = shi_step_sizes(&tte_fn, eta_hat.as_slice());
                    tte_h += data_term_hessian_fd(&tte_fn, eta_hat.as_slice(), &steps);
                }
            }
        }

        // Discrete endpoints (#760: binary; ordinal/Poisson/… later): fold their NLL at the
        // mode and FD Hessian w.r.t. η into the same accumulators, so log|H̃| includes the
        // discrete curvature. A pure-discrete subject has an empty Gaussian h_matrix; hrh then
        // comes entirely from here. `has_discrete()` gates a new family in without a change here.
        if model.has_discrete() {
            let discrete_fn = |eta_eval: &[f64]| {
                crate::categorical::discrete_subject_nll(model, subject, theta, eta_eval)
            };
            tte_nll_at_mode += discrete_fn(eta_hat.as_slice());
            if n_eta > 0 {
                let steps = shi_step_sizes(&discrete_fn, eta_hat.as_slice());
                tte_h += data_term_hessian_fd(&discrete_fn, eta_hat.as_slice(), &steps);
            }
        }

        // CTMM (#759): same treatment — fold its NLL at the mode and FD Hessian
        // w.r.t. η (the generator's η-curvature) into the accumulators.
        #[cfg(feature = "markov")]
        if model.has_ctmm() {
            let ctmm_fn = |eta_eval: &[f64]| {
                crate::markov::endpoint::ctmm_subject_nll(model, subject, theta, eta_eval)
            };
            let mode = ctmm_fn(eta_hat.as_slice());
            tte_nll_at_mode += mode;
            // A perturbed-η CTMM evaluation can hit the 1e20 degeneracy sentinel (an
            // observed transition underflowing, or a negative off-diagonal, under the
            // perturbed generator). That sentinel in a central-difference stencil yields a
            // ~1e24 curvature entry that would poison log|H̃| and hence the OFV / SEs. When
            // the mode itself is already the sentinel the subject is repelled by
            // `tte_nll_at_mode` alone; otherwise fold the FD Hessian only if every entry is
            // finite and of sane magnitude, dropping a sentinel-poisoned block. On the
            // healthy path (no sentinel) this is bit-identical to folding unconditionally.
            // See #759 review.
            if n_eta > 0 && mode < CTMM_FD_SENTINEL_GUARD {
                let steps = shi_step_sizes(&ctmm_fn, eta_hat.as_slice());
                let h_ctmm = data_term_hessian_fd(&ctmm_fn, eta_hat.as_slice(), &steps);
                if h_ctmm
                    .iter()
                    .all(|x| x.is_finite() && x.abs() < CTMM_FD_SENTINEL_GUARD)
                {
                    tte_h += h_ctmm;
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
            // FOCE/FOCEI is selected by the `interaction` flag alone — M3 no longer
            // force-promotes to interaction (mirrors the non-TTE branch below and the
            // IOV `foce_subject_nll_iov`, #367/#591), so FOCE-M3 keeps non-interaction
            // (no CᵀC) semantics here too. NOTE: the censored data term on this TTE path
            // still rides `gaussian_foce_accum` at the conditional (η̂) variance, not the
            // population-variance Sheiner–Beal `foce_subject_nll_standard` the non-TTE
            // FOCE-M3 path uses; reconciling that (a `*_standard_with_tte` censored term)
            // is a follow-up — joint analytic-`family` TTE + M3 + FOCE is the only path
            // affected.
            interaction,
            model.residual_error_eta,
            ruv_mult.as_deref(),
        );
    }

    // M3 no longer force-promotes to interaction: plain FOCE keeps a consistent
    // FOCE (Sheiner–Beal) objective for the whole subject, with the censored rows
    // entering the standard path as `−logΦ` terms (excluded from R̃). FOCEI still
    // takes the interaction path. This matches NONMEM `METHOD=1 LAPLACE` with vs
    // without INTER (FOCE-M3 and FOCEI-M3 are genuinely different optima).
    // Interaction (FOCEI, and SAEM's AIC/BIC-comparable OFV which evaluates this
    // with `interaction = true`) routes by whether the residual covariance is
    // diagonal. A correlated (`block_sigma`) model uses the dense-R interaction
    // marginal [`foce_subject_nll_interaction_dense`] (#616); the diagonal
    // accumulator [`foce_subject_nll_interaction`] would silently drop the
    // off-diagonal covariance. The non-interaction (Sheiner–Beal) branch carries
    // the dense R for both cases via `compute_r_matrix_with_correlations`.
    if interaction {
        if residual_correlations.is_empty() {
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
            foce_subject_nll_interaction_dense(
                subject,
                &ipreds,
                eta_hat,
                h_matrix,
                omega,
                sigma_values,
                &model.error_spec,
                residual_correlations,
                &p_obs,
                ruv_mult.as_deref(),
            )
        }
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
            residual_correlations,
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

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
    let mut r_matrix = crate::stats::residual_error::r_matrix_maybe_scaled(
        error_spec,
        r_eval,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_correlations,
        ruv_mult,
    );
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
    // the univariate marginal tail probability `−logΦ((LLOQ − f0)/√R̃ⱼⱼ)`, using the
    // SAME linearized-marginal moments the quantified rows use: the marginal mean
    // `f0 = f(η̂) − Hη̂` and the marginal variance `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰ⱼ` (#646).
    // This keeps FOCE a consistent Sheiner–Beal (linearized-marginal) objective —
    // matching Monolix's linearization likelihood and first-order/Tobit theory —
    // in contrast to FOCEI (`foce_subject_nll_interaction`), a *conditional* first-order
    // method whose censored term stays at the conditional `f(η̂)`/`R⁰` and instead folds
    // the censored curvature into `H̃` (see `cens_hess`, #486) — the same conditional
    // treatment NONMEM's LAPLACE M3 uses, at FOCEI Gauss-Newton order (dropping the
    // `∂²f/∂η²` term full Laplace keeps).
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
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if cens != 0 {
                // Marginal variance R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰ⱼ (Hⱼ = row j of the inner
                // Jacobian ∂f/∂η). `m3_logcdf` selects the tail from the CENS sign:
                // lower for below-LLOQ (`cens > 0`), upper for above-ULOQ (`cens < 0`);
                // `subject.observations[j]` holds the censoring limit.
                let hj = h_matrix.row(j).transpose();
                let rtilde_jj = hj.dot(&(&omega.matrix * &hj)) + r_diag[j];
                bloq_term +=
                    -2.0 * m3_logcdf(subject.observations[j], f0[j], rtilde_jj.sqrt(), cens);
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
/// With `bloq_method == M3`, censored observations leave the Gaussian residual
/// sum and contribute the matching normal-tail likelihood `−logΦ` evaluated at η̂
/// (the `bloq_term`). They also enter `H̃`/`log|H̃|` at FOCEI (Gauss–Newton) order —
/// the structural `g2·a·aᵀ` plus, under `iiv_on_ruv`, the residual-eta `C·z`/`C·m`
/// cross terms (`cens_hess`) — consistently with the quantified rows, matching
/// NONMEM's LAPLACE M3 Hessian (only the `g1·d2f` piece FOCEI drops for every row
/// is dropped here too).
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
    // H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + cens_hess + Ω⁻¹.  log|H̃| via Cholesky.
    // (`cens_hess` = 0 unless the subject has M3-censored rows.)
    let htilde = g.hrh + 0.5 * g.ctc + &g.cens_hess + &omega.inv;
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => return 1e20,
    };

    0.5 * (g.data_ll + eta_prior + omega.log_det + log_det_htilde + g.bloq_term)
}

/// FOCEI INTER per-subject −2·log marginal for a **correlated** (`block_sigma`)
/// residual covariance — the dense-`R` generalisation of
/// [`foce_subject_nll_interaction`].
///
/// The diagonal interaction path forms `R` one observation at a time and so
/// silently drops the `block_sigma` off-diagonals. This path instead carries the
/// full residual covariance `R` (from
/// [`compute_r_matrix_with_correlations`](crate::stats::residual_error::compute_r_matrix_with_correlations),
/// the same matrix the FOCE non-interaction and SAEM paths use), giving the
/// Almquist 2015 first-order conditional Hessian
///
/// ```text
///   data_ll(η̂) = rᵀ R⁻¹ r + log|R|
///   H̃          = Hᵀ R⁻¹ H + ½·B + Ω⁻¹
///   B_{kl}      = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l),  ∂R/∂η_k = Σ_m H[m,k]·∂R/∂f_m
/// ```
///
/// where `H = ∂f/∂η` (`h_matrix`) and the per-observation `∂R/∂f_m` come from
/// [`crate::stats::residual_error::compute_dr_df_matrices`].
///
/// Reduces **exactly** to [`foce_subject_nll_interaction`] when `R` is diagonal
/// (ρ = 0): `Hᵀ diag(1/V) H = Σ_m a_m a_mᵀ/V_m` (`hrh`) and, since each `∂R/∂f_m`
/// is then the single diagonal entry `∂V_m/∂f_m`, `R⁻¹ ∂R/∂η_k` is diagonal with
/// entries `c̃_{m,k} = (∂V_m/∂η_k)/V_m`, so `B = c̃ᵀc̃` (`ctc`). The dispatcher
/// keeps ρ = 0 on the diagonal path, so that function stays the bit-for-bit
/// reference and this one runs only when correlations are present.
///
/// `block_sigma` is rejected up front together with M3, FREM, IOV (κ), and
/// `iiv_on_ruv`, so none of those appear here; `p_obs` (SDE EKF process noise)
/// inflates the `R` diagonal as an η-independent term, matching the diagonal
/// interaction path (and excluded from `∂R/∂η`).
#[allow(clippy::too_many_arguments)]
pub fn foce_subject_nll_interaction_dense(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    correlations: &[ResidualCorrelation],
    p_obs: &[f64],
    // Per-observation custom residual magnitude (#484); `None` on the legacy path.
    ruv_mult: Option<&[Vec<f64>]>,
) -> f64 {
    let n_obs = subject.observations.len();
    let n_eta = eta_hat.len();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

    // Dense R at η̂ (+ SDE process noise on the diagonal), assembled exactly as
    // the data term and FOCE-standard path assemble it.
    let mut r = crate::stats::residual_error::r_matrix_maybe_scaled(
        error_spec,
        ipreds,
        err_keys.as_ref(),
        subject,
        sigma_values,
        correlations,
        ruv_mult,
    );
    for (j, v) in p_obs.iter().enumerate() {
        if j < r.nrows() {
            r[(j, j)] += *v;
        }
    }

    let chol = match r.clone().cholesky() {
        Some(c) => c,
        None => return 1e20,
    };
    let r_inv = chol.inverse();

    // data_ll = rᵀ R⁻¹ r + log|R|.
    let residuals = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(ipreds.iter())
            .map(|(&y, &f)| y - f),
    );
    let solved = chol.solve(&residuals);
    let data_ll = residuals.dot(&solved) + chol_log_det(&chol.l());

    // First-order conditional Hessian terms.
    // term1 = Hᵀ R⁻¹ H.
    let term1 = h_matrix.transpose() * &r_inv * h_matrix;

    // B_{kl} = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l). Build M_k = R⁻¹ ∂R/∂η_k once per η,
    // then B_{kl} = tr(M_k M_l) = Σ_{p,q} M_k[p,q]·M_l[q,p].
    let dr = crate::stats::residual_error::compute_dr_df_matrices(
        error_spec,
        ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        &subject.obs_l2,
        sigma_values,
        correlations,
        ruv_mult,
    );
    let mut m_mats: Vec<DMatrix<f64>> = Vec::with_capacity(n_eta);
    for k in 0..n_eta {
        let mut dr_k = DMatrix::<f64>::zeros(n_obs, n_obs);
        for (m, dr_m) in dr.iter().enumerate() {
            let h_mk = h_matrix[(m, k)];
            if h_mk != 0.0 {
                dr_k += h_mk * dr_m;
            }
        }
        m_mats.push(&r_inv * dr_k);
    }
    let mut b = DMatrix::<f64>::zeros(n_eta, n_eta);
    for k in 0..n_eta {
        for l in 0..n_eta {
            let mut t = 0.0;
            for p in 0..n_obs {
                for q in 0..n_obs {
                    t += m_mats[k][(p, q)] * m_mats[l][(q, p)];
                }
            }
            b[(k, l)] = t;
        }
    }

    let htilde = term1 + 0.5 * b + &omega.inv;
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => return 1e20,
    };
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat));

    0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde)
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
    // FOCEI adds the ½·CᵀC interaction curvature; plain FOCE omits it. The censored
    // `cens_hess` (structural `g2·a·aᵀ` + residual-eta `C·z`/`C·m`) is FOCEI-order
    // curvature — its `g2` folds in `∂v/∂f`/`∂²v/∂f²`, and the `C·z`/`C·m` terms are
    // pure `iiv_on_ruv` interaction — so it is gated to the interaction branch, exactly
    // as `½·g.ctc` is. Plain FOCE (non-interaction) leaves censored rows out of `H̃`,
    // matching `foce_subject_nll_standard`.
    let htilde = if interaction {
        hrh + 0.5 * g.ctc + &g.cens_hess + &omega.inv
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
    /// Σⱼ censored rows' contribution to `H̃` at FOCEI (Gauss-Newton) order:
    /// the structural curvature `g2·aⱼ'aⱼ` (`g2 = ∂²(−logΦ)/∂f²`) plus, under
    /// `iiv_on_ruv`, the residual-eta cross terms `C·z` (diagonal) and `C·m·a`
    /// (η_ruv coupling). This is `precise_ebe`'s censored inner-Hessian block
    /// **minus** the `g1·d2f` term FOCEI legitimately drops (that drop is the
    /// FOCEI-vs-LAPLACE difference; excluding the *rows* was an inconsistency).
    /// Added directly (unscaled) to `H̃`, so censored rows enter `log|H̃|`
    /// consistently with quantified rows. Zero when no censored rows.
    cens_hess: DMatrix<f64>,
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
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

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
        // `residual_rd` returns the bare `(R, ∂R/∂f)` pair and leaves `ruv_scale`
        // to the caller, so both carry the factor: scaling `R` by `exp(2·η_ruv)`
        // multiplies `∂R/∂f` by the same amount, and the two cancel in `c_scale`
        // below. FREM covariate rows take their variance from the override and
        // contribute `∂R/∂f = 0` (additive near-zero sigma).
        let (v_resid, dvar_df) = match frem_ov {
            Some(v) => (v, 0.0),
            None => {
                let (rv, dv) = residual_rd(error_spec, err_keys[j], f, sigma_values, mult_row(j));
                (rv * ruv_scale, dv * ruv_scale)
            }
        };
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            return None;
        }
        let r = subject.observations[j] - f;
        data_ll += r * r / v + v.ln();

        // a_j = row j of H (∂f_j/∂η); c̃_{j,k} = (∂R_j/∂η_k)/R_j.
        // For a PK eta: (∂R/∂f)·a / R.
        // For the residual eta: ∂R/∂η_ruv = 2·R, so the column is the constant 2.
        let aj = h_matrix.row(j);
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

    // Censored contributions at η̂ (ipred-based variance). Censored rows enter both
    // the data term (`bloq_term`) AND `H̃` (`cens_hess`) — consistently with quantified
    // rows — at FOCEI (Gauss-Newton) order (structural `g2·a·aᵀ` + residual-eta `C·z`/`C·m`,
    // dropping only the `g1·d2f` piece FOCEI drops for every row).
    let mut bloq_term = 0.0;
    let mut cens_hess = DMatrix::<f64>::zeros(n_eta, n_eta);
    for &j in &bloq_idx {
        let limit = subject.observations[j];
        let f = ipreds[j];
        let cmt = err_keys[j];
        // `v_resid`/`d`/`d2` all carry the same proportional-loading (`m²`) and
        // `exp(2·η_ruv)` scaling, so the censored curvature is built from mutually
        // consistent derivatives of one `v(f)`.
        let (rv, dv, d2v) = residual_rd2(error_spec, cmt, f, sigma_values, mult_row(j));
        let (v_resid, d, d2) = (rv * ruv_scale, dv * ruv_scale, d2v * ruv_scale);
        // Inflate by the EKF process-noise term exactly as the quantified rows do
        // (line above), so a censored row's `v` — in both the data term and `H̃` —
        // matches its subject's Gaussian rows under an SDE model. `p_obs` is
        // f-independent, so `d`/`d2` (∂v/∂f, ∂²v/∂f²) are unchanged.
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            return None;
        }
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        bloq_term += -2.0 * m3_logcdf(limit, f, v.sqrt(), cens);

        // Censored curvature for `H̃` (`L = −logΦ`): structural `g2·a·aᵀ` plus, under
        // `iiv_on_ruv`, the residual-eta coupling `C·z` on (rr,rr) and `C·m·a_l` on the
        // (rr,l) cross terms. Single-sourced via `m3_censored_outer` (shared with the
        // analytic outer gradient) so the two curvature copies cannot drift.
        let (_g1, g2, cz, cm) = crate::stats::special::m3_censored_outer(limit, f, v, d, d2, cens);
        let aj = h_matrix.row(j);
        for a in 0..n_eta {
            let ga = g2 * aj[a];
            for b in 0..n_eta {
                cens_hess[(a, b)] += ga * aj[b];
            }
        }
        if let Some(rr) = residual_error_eta {
            cens_hess[(rr, rr)] += cz;
            for l in 0..n_eta {
                if l != rr {
                    cens_hess[(rr, l)] += cm * aj[l];
                    cens_hess[(l, rr)] += cm * aj[l];
                }
            }
        }
    }

    Some(GaussianFoceTerms {
        data_ll,
        hrh,
        ctc,
        bloq_term,
        cens_hess,
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
/// `iov_occasion_groups`).  When `kappas` is empty, falls through to the
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
            // `block_sigma` + IOV is rejected up front, so an IOV model's
            // correlations are always the (empty) declaration (#847).
            &model.residual_correlations,
            interaction,
        );
    }

    let occ_groups = iov_occasion_groups(subject);
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
    // M3 no longer force-promotes IOV-FOCE subjects to interaction (#591, mirroring the
    // non-IOV `foce_subject_nll` change in #367/8abadbb7): plain FOCE keeps a consistent
    // FOCE (Sheiner–Beal) objective for the whole subject — the censored rows leave the
    // linearized augmented marginal and re-enter through `foce_subject_nll_standard` as the
    // marginal tail `−logΦ((LLOQ−f0)/√R̃ⱼⱼ)`, R̃ⱼⱼ = Hⱼ Σ_b Hⱼᵀ + R⁰ⱼ over the stacked
    // [η, κ] system (#646, the same linearized-marginal moments the quantified rows use).
    // FOCEI still takes the interaction path. FOCE-IOV-M3 and FOCEI-IOV-M3 are genuinely different
    // optima, matching NONMEM METHOD=1 LAPLACE with vs without INTER. (Previously a
    // censored subject was silently evaluated with η-interaction even under FOCE, mixing a
    // Sheiner–Beal marginal with a FOCEI censored term.) The censored rows are now routed
    // by the `interaction` flag alone — `foce_subject_nll_standard` carries the M3 `−logΦ`
    // term for the non-interaction path — so there is no `m3_active` gate here (the non-IOV
    // `foce_subject_nll`, including its TTE branch, drops the promotion the same way).
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
    if interaction {
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
            // `block_sigma` + IOV is rejected up front
            // (`E_BLOCK_SIGMA_IOV_UNSUPPORTED`), so this vector is empty in
            // practice and the estimated-ρ threading (#847) never reaches here.
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
    // Compute each subject's NLL in parallel, then reduce in a fixed subject
    // order. rayon's `ParallelIterator::sum` folds partial sums along split
    // boundaries that depend on the worker-thread count, and f64 addition is
    // not associative — so a bare `.sum()` yields a thread-count-dependent OFV
    // (e.g. 4 vs 15 threads). Collecting into an ordered `Vec` and summing
    // serially makes the objective bit-reproducible regardless of thread count
    // while keeping the expensive per-subject work parallel (#703).
    let per_subj: Vec<f64> = population
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
        .collect();
    per_subj.iter().sum()
}

/// Population FOCE objective: sum over all subjects
#[allow(clippy::too_many_arguments)]
pub fn foce_population_nll(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation],
    interaction: bool,
) -> f64 {
    // Deterministic reduction: collect per-subject NLLs in subject order, then
    // sum serially. A parallel `.sum()` would make the OFV depend on the
    // thread count (f64 addition is non-associative; rayon splits by worker
    // count). See `foce_population_nll_iov` and #703.
    let per_subj: Vec<f64> = population
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
                residual_correlations,
                interaction,
            )
        })
        .collect();
    per_subj.iter().sum()
}

/// Compute CWRES (Conditional Weighted Residuals) for a subject — Hooker et
/// al. (2007), NONMEM's `CWRES` table item.
///
/// The residual is `y − f0` with `f0 = f(η̂) − H·η̂`, the FOCE-linearized
/// population prediction, and its covariance is `R̃ = H·Ω·Hᵀ + R`. The
/// returned vector is the **decorrelated** residual `R̃^{-1/2}(y − f0)`, with
/// `R̃^{-1/2}` the *symmetric* inverse square root (eigendecomposition), not
/// `(y − f0)ⱼ / √R̃ⱼⱼ` and not the Cholesky factor. `R` is evaluated at
/// `r_preds` when given — `IPRED` under an interaction fit, the population
/// prediction `f(η = 0)` under FOCE without interaction, the points the
/// respective marginals use — and at `f0` otherwise.
///
/// That recipe was pinned against NONMEM 7.5.1 on a 40-subject oral dataset
/// with sizeable η (#1182): rebuilt from NONMEM's own tabled `G11..G31`,
/// `ETA`, `IPRED` and `PRED`, the symmetric square root with `R` at `IPRED`
/// reproduces its `CWRES` column to an RMS of 1e-4 (the table's rounding),
/// where the marginal standardisation ferx used before was off by an RMS of
/// 0.75 and a Cholesky decorrelation by 0.57 — differences large enough to
/// change which residual-error form a CWRES-based screen (ruvsearch's) picks.
/// A model with no η has `R̃` diagonal, so every recipe agrees there.
///
/// Censored observations get `NaN` since a weighted Gaussian residual is
/// undefined when the observed value is censored; they are left out of the
/// decorrelation.
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
    // The predictions at which `R` is evaluated — `IPRED` under an interaction
    // fit, `f(η = 0)` under FOCE — NONMEM's `CWRES`. `None` evaluates `R` at
    // the linearized `f0`.
    r_preds: Option<&[f64]>,
) -> Vec<f64> {
    let n_obs = subject.observations.len();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

    // f0 = ipred - H * eta_hat
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R_tilde, with R at the requested predictions when given (see above).
    let r_eval: &[f64] = r_preds.unwrap_or(&f0);
    let mut r_matrix = crate::stats::residual_error::r_matrix_maybe_scaled(
        error_spec,
        r_eval,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_correlations,
        ruv_mult,
    );
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

    // CWRES = R̃^{-1/2} (y − f0) over the uncensored rows, with the symmetric
    // inverse square root: R̃ = V·Λ·Vᵀ, R̃^{-1/2} = V·Λ^{-1/2}·Vᵀ. This is the
    // decorrelation NONMEM's `CWRES` applies (see the doc comment); a Cholesky
    // factor decorrelates too but yields a different vector, and the marginal
    // `(y − f0)ⱼ / √R̃ⱼⱼ` a third — the three agree only when `H·Ω·Hᵀ` has no
    // off-diagonals, i.e. with no η. Censored rows are left out of the system
    // and reported as NaN; a non-positive eigenvalue (a hand-built pathological
    // input) falls back to the marginal standardisation rather than panicking.
    let live: Vec<usize> = (0..n_obs)
        .filter(|&j| subject.cens.get(j).copied().unwrap_or(0) == 0)
        .collect();
    let mut out = vec![f64::NAN; n_obs];
    if live.is_empty() {
        return out;
    }
    let resid = DVector::from_iterator(
        live.len(),
        live.iter().map(|&j| subject.observations[j] - f0[j]),
    );
    let sub = r_tilde.select_rows(&live).select_columns(&live);
    let eig = nalgebra::SymmetricEigen::new(sub);
    if eig.eigenvalues.iter().all(|&l| l > 0.0 && l.is_finite()) {
        let vt_r = eig.eigenvectors.transpose() * &resid;
        let scaled = DVector::from_iterator(
            live.len(),
            vt_r.iter()
                .zip(eig.eigenvalues.iter())
                .map(|(v, l)| v / l.sqrt()),
        );
        let z = &eig.eigenvectors * scaled;
        for (k, &j) in live.iter().enumerate() {
            out[j] = z[k];
        }
    } else {
        for (k, &j) in live.iter().enumerate() {
            out[j] = resid[k] / r_tilde[(j, j)].max(1e-12).sqrt();
        }
    }
    out
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

/// IOV kappa groups for a subject.
///
/// Observation occasions keep the historical [`split_obs_by_occasion`] order. Dose
/// occasions that have no sampled observations are appended in first-dose order with an
/// empty observation-index list. This preserves existing stacked-kappa ordering for the
/// common case while assigning dose-only occasions their own axes, so dose carryover into
/// later observations can be differentiated analytically.
pub fn iov_occasion_groups(subject: &Subject) -> Vec<(u32, Vec<usize>)> {
    let mut groups = split_obs_by_occasion(subject);
    let mut seen: std::collections::HashSet<u32> = groups.iter().map(|(occ, _)| *occ).collect();
    for &occ in &subject.dose_occasions {
        if seen.insert(occ) {
            groups.push((occ, Vec::new()));
        }
    }
    groups
}

/// Map each occasion id to its group index `k` (the order [`iov_occasion_groups`]
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

/// The combined effect vector `[η_bsv, 0…0]` for an **EVID=2 pk-only** event — the κ=0
/// counterpart of [`iov_combined_effect`]. Pk-only breakpoints carry no occasion, so their
/// IOV κ block is held at zero (matching production `predict_iov`'s `combined_for(u32::MAX)`
/// branch). Shares the stacked-η layout with [`iov_combined_effect`] so the closed-form and
/// ODE IOV providers cannot seed the κ=0 block differently (#598 review).
pub fn iov_combined_pk_only(stacked_eta: &[f64], n_eta: usize, n_kappa: usize) -> Vec<f64> {
    let mut c = Vec::with_capacity(n_eta + n_kappa);
    c.extend_from_slice(&stacked_eta[..n_eta]);
    c.extend(std::iter::repeat(0.0).take(n_kappa));
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
/// returned by `iov_occasion_groups`).  When `kappas` is empty, falls back
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
    // #484/#1029: per-observation residual-magnitude multiplier, so the IOV
    // individual NLL (SAEM's E-step, the Bayes MH target, the IOV IS weights)
    // scores the same variance the non-IOV path and FOCE/FOCEI do.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut data_ll = 0.0;
    for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let v = match frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
            Some(vv) => vv,
            None => {
                model.residual_variance_at_scaled(
                    err_keys[j],
                    f_pred,
                    sigma_values,
                    ruv_mult.as_ref().map(|m| m[j].as_slice()),
                ) * ruv_scale
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

    // Non-Gaussian data term at 2× (the final 0.5 gives net 1.0×): TTE endpoints plus the
    // discrete (binary/categorical) term. Kappas are PK-only; the hazard param_fn uses BSV η,
    // not kappas. No joint-share on the IOV path (it never carries a joint-PK-TTE endpoint),
    // so the TTE term is analytic and `tte_endpoint_nll` matches the `tte_data_term` this
    // site inlined.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        // No CTMM term reaches this IOV path in practice: a CTMM endpoint with IOV
        // (n_kappa > 0) is rejected at fit setup (its intensities are evaluated with
        // BSV-only η), so the helper's CTMM add is a no-op here.
        accumulate_non_gaussian_nll(model, subject, theta, eta, None, 2.0, &mut data_ll);
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
            reset_covariates: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    fn make_omega(var: f64) -> OmegaMatrix {
        OmegaMatrix::from_diagonal(&[var], vec!["ETA_CL".into()])
    }

    fn make_model() -> CompiledModel {
        CompiledModel {
            covariate_model: None,
            name: "test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            residual_correlations: Vec::new(),
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp(); // CL uses combined eta[0]
                    p.values[1] = theta[1]; // V
                    p
                },
            ),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params: crate::types::ModelParameters {
                residual_correlations: Vec::new(),
                residual_correlation_fixed: Vec::new(),
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
                mixture: None,
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            kappa_weights: Vec::new(),
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
            theta_eta_linked: Vec::new(),
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
            analytic_readout: None,
            ruv_magnitude: None,
            absorption_ode_equivalent: None,
            mixture: None,
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

        let corr = model.residual_correlations.clone();
        let dense = obs_nll_subject_into(&model, &subj, &theta, &sigma, &corr, &eta, &mut scratch);
        model.residual_correlations.clear();
        let diagonal = obs_nll_subject_into(&model, &subj, &theta, &sigma, &[], &eta, &mut scratch);

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

    /// #1029 degenerate oracle: `weight = W` reproduces the hand-built
    /// study-as-subject construction — `DV/W` in the data, `PRED/W` in the
    /// `[scaling]` block, `sigma FIX 1` in `[parameters]` — exactly.
    ///
    /// The two objectives differ only by the Jacobian of the `y ↦ y/W` change of
    /// variable, `Σ_j ln W_j`. That is a data-only constant: identical for every
    /// model fitted to the same rows with the same weight column, so it cancels
    /// out of every ΔOFV, likelihood-ratio test, and AIC comparison. It appears
    /// at all only because ferx scores on the *natural* scale, which is what
    /// keeps DV / PRED / IPRED / CWRES and the VPC in reported units with no
    /// back-transformation.
    #[test]
    fn test_weighted_error_model_matches_the_hand_built_construction() {
        let src = |err: &str| {
            format!(
                "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
                 sigma ADD_ERR ~ 1.0 (variance) FIX\n[individual_parameters]\n  CL = TVCL * \
                 exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
                 v=V)\n[error_model]\n{err}\n[covariates]\n  WPSE continuous\n"
            )
        };
        let weighted = crate::parser::model_parser::parse_model_string(&src(
            "  DV ~ additive(ADD_ERR) weight = WPSE",
        ))
        .unwrap();
        let plain =
            crate::parser::model_parser::parse_model_string(&src("  DV ~ additive(ADD_ERR)"))
                .unwrap();

        // One row per trial arm, each carrying its own reported standard error.
        let w = [0.5_f64, 2.0, 1.25];
        let y = [12.0_f64, 9.0, 15.0];
        let f = [11.0_f64, 10.0, 14.0];

        let mut natural = make_simple_subject();
        natural.obs_times = vec![1.0, 2.0, 3.0];
        natural.observations = y.to_vec();
        natural.obs_cmts = vec![1; 3];
        natural.cens = vec![0; 3];
        natural.occasions = vec![1; 3];
        natural.obs_covariates = w
            .iter()
            .map(|&wi| [("WPSE".to_string(), wi)].into_iter().collect())
            .collect();

        // The hand-built twin: DV and the prediction both pre-divided by the
        // weight, scored by an unweighted additive model on the same sigma.
        let mut prescaled = natural.clone();
        prescaled.observations = y.iter().zip(w).map(|(&yi, wi)| yi / wi).collect();
        prescaled.obs_covariates = Vec::new();
        let scaled_preds: Vec<f64> = f.iter().zip(w).map(|(&fi, wi)| fi / wi).collect();

        let theta = [0.2, 10.0];
        let sigma = [0.8];
        let eta = [0.0];

        let nll_weighted =
            obs_nll_subject_from_preds(&weighted, &natural, &f, &theta, &sigma, &[], &eta);
        let nll_hand = obs_nll_subject_from_preds(
            &plain,
            &prescaled,
            &scaled_preds,
            &theta,
            &sigma,
            &[],
            &eta,
        );
        let jacobian: f64 = w.iter().map(|wi| wi.ln()).sum();

        approx::assert_relative_eq!(nll_weighted, nll_hand + jacobian, epsilon = 1e-12);
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
    fn test_iov_occasion_groups_appends_dose_only_occ() {
        let mut subj = make_simple_subject();
        subj.dose_occasions = vec![1, 3, 2, 4, 3];
        let groups = iov_occasion_groups(&subj);
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0], (1, vec![0, 1, 2]));
        assert_eq!(groups[1], (2, vec![3, 4, 5]));
        assert_eq!(groups[2], (3, Vec::new()));
        assert_eq!(groups[3], (4, Vec::new()));
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

    /// The IOV individual NLL — SAEM's E-step evaluator, the Bayes MH target,
    /// and the IOV importance-sampling weight — must apply the per-observation
    /// residual magnitude (#484/#1029) exactly as the non-IOV path does. With no
    /// kappas the two are the same likelihood, so any divergence is the
    /// magnitude going missing on one side. A weight that *varies within the
    /// subject* is what makes this bite: a frozen or dropped multiplier still
    /// matches on the first row.
    #[test]
    fn test_individual_nll_iov_applies_the_residual_magnitude() {
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(5.0)\n  theta TVV(50.0)\n  omega ETA_CL ~ 0.09\n  \
             sigma PROP_ERR ~ 0.10 (sd)\n  sigma ADD_ERR ~ 0.50 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
             DV ~ combined(PROP_ERR * (1.0 + 0.5 * WPSE), ADD_ERR) weight = WPSE\n\
             [covariates]\n  WPSE continuous\n",
        )
        .expect("weighted model parses");
        assert!(model.has_custom_ruv_magnitude());

        let mut subj = make_simple_subject();
        let snap =
            |w: f64| -> HashMap<String, f64> { [("WPSE".to_string(), w)].into_iter().collect() };
        subj.covariates = snap(0.5);
        subj.obs_covariates = (0..subj.observations.len())
            .map(|j| snap(0.5 + 0.3 * j as f64))
            .collect();

        let theta = vec![5.0, 50.0];
        let eta = vec![0.0];
        let omega = make_omega(0.09);
        let sigma = vec![0.10, 0.50];

        let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
        let iov = individual_nll_iov(&model, &subj, &theta, &eta, &[], &omega, None, &sigma);
        approx::assert_relative_eq!(base, iov, epsilon = 1e-10);

        // Guard against a vacuous pass: the magnitude must actually move the NLL.
        let mut flat = subj.clone();
        flat.obs_covariates = vec![snap(0.5); subj.observations.len()];
        let flat_nll = individual_nll(&model, &flat, &theta, &eta, &omega, &sigma);
        assert!(
            (flat_nll - base).abs() > 1e-6,
            "the per-observation weight must change the NLL, else this test proves nothing"
        );
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
        model.pk_param_fn = Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL
                p.values[1] = theta[1]; // V
                p
            },
        );
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
            &model,
            &subj,
            &theta,
            &eta_hat,
            &h_bsv,
            &omega_bsv,
            &sigma,
            &model.residual_correlations,
            false,
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

    /// The dense-`R` FOCEI interaction marginal must reduce **exactly** to the
    /// diagonal accumulator [`foce_subject_nll_interaction`] when there is no
    /// `block_sigma` correlation (ρ = 0), across a multi-η proportional model.
    /// This anchors the dense generalisation against the bit-for-bit reference.
    #[test]
    fn dense_interaction_reduces_to_diagonal_when_uncorrelated() {
        let subject = make_simple_subject(); // 6 obs, single proportional endpoint
        let sigma = vec![0.2];
        let error_spec = ErrorSpec::Single(ErrorModel::Proportional);
        let ipreds = vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0];
        let eta_hat = DVector::from_vec(vec![0.15, -0.1]);
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
        let h_matrix = DMatrix::from_row_slice(
            6,
            2,
            &[
                5.0, 1.0, //
                4.0, 1.5, //
                3.0, 2.0, //
                4.5, 1.2, //
                3.5, 1.8, //
                2.5, 2.2,
            ],
        );

        let diag = foce_subject_nll_interaction(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            BloqMethod::Drop,
            &[],
            None,
            None,
            None,
        );
        let dense = foce_subject_nll_interaction_dense(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            &[], // ρ = 0 → diagonal R, must match the reference exactly
            &[],
            None,
        );
        approx::assert_relative_eq!(dense, diag, max_relative = 1e-10);

        // The #484 magnitude path (Some(mult)) with an all-ones multiplier must
        // reproduce the bare path (modulo the documented ~1 ULP reassociation).
        let ones_mult: Vec<Vec<f64>> = vec![vec![1.0]; 6];
        let dense_mult = foce_subject_nll_interaction_dense(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            &[],
            &[],
            Some(&ones_mult),
        );
        approx::assert_relative_eq!(dense_mult, dense, max_relative = 1e-9);
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

    /// #646: FOCE (Sheiner–Beal) M3 censored rows must use the linearized-marginal
    /// variance `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰`, not the conditional residual `R⁰`. With the
    /// quantified rows' Jacobian zeroed, the quant SB marginal `R̃ = R` is
    /// Ω-independent, so the objective can only depend on Ω through the censored
    /// row's marginal variance. The old conditional censored term (√R⁰) would give
    /// an identical NLL for any Ω; the marginal term must not.
    #[test]
    fn foce_m3_censored_uses_marginal_variance() {
        let mut subject = make_simple_subject();
        subject.cens = vec![0, 0, 0, 0, 0, 1]; // last row below-LLOQ
        subject.observations[5] = 5.0; // LLOQ limit
        let sigma = vec![3.0]; // additive SD → R⁰ = 9, Ω-independent
        let error_spec = ErrorSpec::Single(ErrorModel::Additive);
        let ipreds = vec![50.0, 40.0, 30.0, 45.0, 35.0, 8.0];
        let eta_hat = DVector::from_vec(vec![0.5]);
        // Quant rows H=0 (their R̃ = R is Ω-independent); censored row H = 4 ≠ 0.
        let h_matrix = DMatrix::from_column_slice(6, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
        let call = |omega: &OmegaMatrix| {
            foce_subject_nll_standard(
                &subject,
                &ipreds,
                &eta_hat,
                &h_matrix,
                omega,
                &sigma,
                &error_spec,
                &[],
                BloqMethod::M3,
                &[],
                None,
                None,
                None,
            )
        };
        let nll_small = call(&make_omega(0.01));
        let nll_large = call(&make_omega(1.0));
        assert!(nll_small.is_finite() && nll_large.is_finite());
        // Marginal variance grows with Ω (H²·Ω = 16·Ω), widening the censored tail,
        // so the two NLLs must differ well above FP noise. (Old conditional → equal.)
        assert!(
            (nll_small - nll_large).abs() > 1e-3,
            "censored term must depend on Ω via the marginal variance (#646): \
             nll(Ω=0.01)={nll_small}, nll(Ω=1.0)={nll_large}"
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
            reset_covariates: Vec::new(),
            cens: vec![0, 0],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: vec![0, 100],
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
            None,
        );

        let pk_expected = 1.0 / (sigma[0] * eta_hat[0].exp());
        let frem_expected = 1.0 / sigma[1];
        assert_relative_eq!(cwres[0], pk_expected, epsilon = 1e-12);
        assert_relative_eq!(cwres[1], frem_expected, epsilon = 1e-12);
    }

    /// #1182: CWRES is the *decorrelated* conditional weighted residual —
    /// `R̃^{-1/2}(y − f0)` with the symmetric inverse square root — not each
    /// residual over its own marginal SD. With a nonzero `H` the two differ;
    /// the oracle is the defining identity `R̃^{1/2}·z = y − f0` (checked with
    /// the matrix square root built independently here), plus the invariant
    /// `zᵀz = rᵀR̃⁻¹r`. The covariance is built from `r_preds` when given.
    #[test]
    fn compute_cwres_decorrelates_with_the_symmetric_inverse_square_root() {
        let subject = Subject {
            id: "1".into(),
            obs_times: vec![1.0, 2.0],
            observations: vec![3.0, 5.0],
            obs_cmts: vec![1, 1],
            cens: vec![0, 0],
            ..Default::default()
        };
        // ipred = f0 + H·η̂ with η̂ = 0.5 and H = [1, 2]ᵀ → f0 = (1, 2).
        let ipreds = vec![1.5, 3.0];
        let eta_hat = DVector::from_vec(vec![0.5]);
        let h = DMatrix::from_column_slice(2, 1, &[1.0, 2.0]);
        let omega = OmegaMatrix::from_diagonal(&[0.25], vec!["ETA".into()]);
        let sigma = vec![0.5];
        let spec = ErrorSpec::Single(ErrorModel::Additive);
        let cwres = compute_cwres(
            &subject,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &spec,
            &[],
            None,
            None,
            None,
            None,
        );
        // R̃ = H Ω Hᵀ + σ²I = [[0.5, 0.5], [0.5, 1.25]]; resid = (2, 3).
        let r_tilde = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 1.25]);
        let resid = DVector::from_vec(vec![2.0, 3.0]);
        let z = DVector::from_vec(cwres.clone());
        // R̃^{1/2} via the eigendecomposition, independently of the function.
        let eig = nalgebra::SymmetricEigen::new(r_tilde.clone());
        let sqrt_l = DMatrix::from_diagonal(&eig.eigenvalues.map(f64::sqrt));
        let r_half = &eig.eigenvectors * sqrt_l * eig.eigenvectors.transpose();
        let back = r_half * &z;
        assert_relative_eq!(back[0], resid[0], epsilon = 1e-12);
        assert_relative_eq!(back[1], resid[1], epsilon = 1e-12);
        let mahalanobis = (resid.transpose() * r_tilde.try_inverse().unwrap() * &resid)[(0, 0)];
        assert_relative_eq!(z.dot(&z), mahalanobis, epsilon = 1e-12);
        // Neither the marginal standardisation nor the Cholesky factor gives
        // this vector: the marginal second row would be 3/√1.25, and the
        // Cholesky first row 2/√0.5.
        assert!((cwres[1] - 3.0 / 1.25f64.sqrt()).abs() > 0.1);
        assert!((cwres[0] - 2.0 / 0.5f64.sqrt()).abs() > 0.1);

        // A censored first row is NaN and leaves the second row standardised
        // on its own: 3/√1.25.
        let mut censored = subject.clone();
        censored.cens = vec![1, 0];
        let cw = compute_cwres(
            &censored,
            &ipreds,
            &eta_hat,
            &h,
            &omega,
            &sigma,
            &spec,
            &[],
            None,
            None,
            None,
            None,
        );
        assert!(cw[0].is_nan());
        assert_relative_eq!(cw[1], 3.0 / 1.25f64.sqrt(), epsilon = 1e-12);

        // `r_preds` moves the point R is evaluated at: on a proportional model
        // with a single row, CWRES = resid / √(HΩHᵀ + σ²·pred²).
        let one = Subject {
            id: "1".into(),
            obs_times: vec![1.0],
            observations: vec![3.0],
            obs_cmts: vec![1],
            cens: vec![0],
            ..Default::default()
        };
        let prop = ErrorSpec::Single(ErrorModel::Proportional);
        let h1 = DMatrix::from_column_slice(1, 1, &[1.0]);
        let cw = compute_cwres(
            &one,
            &ipreds[..1],
            &eta_hat,
            &h1,
            &omega,
            &sigma,
            &prop,
            &[],
            None,
            None,
            None,
            Some(&[4.0]),
        );
        assert_relative_eq!(
            cw[0],
            2.0 / (0.25_f64 + 0.25 * 16.0).sqrt(),
            epsilon = 1e-12
        );
        let cw_f0 = compute_cwres(
            &one,
            &ipreds[..1],
            &eta_hat,
            &h1,
            &omega,
            &sigma,
            &prop,
            &[],
            None,
            None,
            None,
            None,
        );
        assert_relative_eq!(
            cw_f0[0],
            2.0 / (0.25_f64 + 0.25 * 1.0).sqrt(),
            epsilon = 1e-12
        );
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
            reset_covariates: Vec::new(),
            cens: vec![0, 0],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: vec![0, 0],
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
            None,
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

        let base = obs_nll_subject_into(&model, &subj, &theta, &sigma, &[], &eta, &mut scratch);
        model.residual_error_eta = Some(0);
        let scaled = obs_nll_subject_into(&model, &subj, &theta, &sigma, &[], &eta, &mut scratch);

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
        let saem = obs_nll_subject_from_preds(
            &model,
            &subject,
            &preds,
            &p.theta,
            &p.sigma.values,
            &p.residual_correlations,
            &eta,
        );
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
            &p.residual_correlations,
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

        // #570: the shared single-solve path must agree with the dedicated two-solve
        // path it replaces — predictions bit-identical, TTE term equal to solver
        // tolerance. Use a non-zero η so the PK trajectory (and thus the hazard) is
        // non-trivial. This is what makes the fast coverage of the share *meaningful*
        // rather than merely "finite".
        let eta_nz = [0.3_f64];
        let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &eta_nz)
            .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
        let mut scratch = pk::EventPkParams::with_capacity_for(&subject);
        let preds_ref = crate::pk::compute_predictions_with_tv_into_with_schedule(
            &model,
            &subject,
            &p.theta,
            &eta_nz,
            &mut scratch,
            None,
        );
        assert_eq!(
            share.preds, preds_ref,
            "shared preds must be bit-identical to the standalone prediction path"
        );
        let (chz_state, hazard) = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: h @ HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => (*chz_state, h),
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };
        let shared_tte = tte_ode_nll_from_shared(
            model.ode_spec.as_ref().unwrap(),
            &share,
            chz_state,
            &subject.obs_records,
        );
        let dedicated_tte = tte_endpoint_nll(
            &model,
            &subject,
            hazard,
            crate::types::TteRecurrence::Single,
            &subject.obs_records,
            &p.theta,
            &eta_nz,
        );
        assert!(
            (shared_tte - dedicated_tte).abs() <= 1e-4 * dedicated_tte.abs().max(1.0),
            "shared TTE NLL {shared_tte} must match dedicated {dedicated_tte} to solver tol"
        );
    }

    /// The joint PK-TTE harness model of [`joint_pktte_ode_hazard_nll_paths_finite`],
    /// as a helper so the #1223 pre-start arms use the same PK block, hazard and error
    /// model without editing that test's fixture.
    #[cfg(feature = "survival")]
    fn joint_pktte_ode_model() -> CompiledModel {
        crate::parser::model_parser::parse_model_string(
            r"
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
",
        )
        .expect("joint PK-TTE model must parse")
    }

    /// A joint PK-TTE subject whose first record is a dose at **t = 10** — so the
    /// integration starts at 10 and a TTE time below it is genuinely pre-start.
    ///
    /// `make_simple_subject` cannot host one: it doses at `t = 0`, and `entry_time` is
    /// gated `> 0.0` ("no truncation"), so there is no positive time before its start.
    /// PK observations sit at 12 / 16 / 24, all *after* the dose, which keeps
    /// `t_last ≥ subject_integration_start` and so routes the pre-start read through the
    /// engines' pre-first-break **fill** rather than through the `k = 0` boundary visit
    /// that covers a degenerate every-time-before-the-dose timeline (#1218).
    #[cfg(feature = "survival")]
    fn late_start_joint_subject(records: Vec<ObsRecord>) -> Subject {
        let mut subject = make_simple_subject();
        subject.doses = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0)];
        subject.obs_times = vec![12.0, 16.0, 24.0];
        subject.observations = vec![50.0, 40.0, 30.0];
        subject.obs_cmts = vec![1; 3];
        subject.cens = vec![0; 3];
        subject.occasions = vec![1; 3];
        subject.obs_records = records;
        subject
    }

    /// #1223: a TTE time **at or before the subject's first record** must be scored, not
    /// repelled, and must be scored identically on all three production callers.
    ///
    /// The one-solve share (`ode_predictions_and_chz`, #570) NaN-filled a `chz_times`
    /// entry before the integration start; `tte_ode_nll_from_shared` skips a non-finite
    /// state, `tte_nll_from_curves` maps the resulting NaN `H` to its `1e20` sentinel, and
    /// the subject is repelled. The dedicated engine (`ode_dense_solve_states`) has filled
    /// the same node with the seeded state since the CTMM scorer needed it. Which engine a
    /// subject reaches is decided by `try_joint_pktte_shared_solve`'s admission list — a
    /// reset, a time-varying covariate, model time in the PK block — none of which has
    /// anything to do with when its `TENTRY` falls.
    ///
    /// Three arms, one per shape that can put a time before the start: a left-truncation
    /// `entry_time`, an interval-censored `left` (pushed unconditionally, with no `> 0`
    /// gate), and a right-censoring time itself.
    ///
    /// **What is non-degenerate here.** The seeded state of this model is all zeros (no
    /// `init(...)`), so "state == `initial_state`" is a zeros-against-zeros identity; the
    /// value that actually discriminates is the hazard, `h = H0·exp(BETA·0/V) = H0 = 0.01`
    /// exactly — non-zero, and wrong under any other state. The non-degenerate *state*
    /// check lives in the driver twin
    /// (`ode::predictions::tests::prestart_chz_time_reads_the_seeded_state_on_both_paths`),
    /// which seeds a non-zero `init(central)`.
    ///
    /// **`predict_survival` is a third caller, not a third geometry**: it routes
    /// `OdeAccumulated` straight to `crate::survival::ode_cumhaz_hazard`, the same dense
    /// solve the two-solve arm uses. It is here because it is the public entry point users
    /// read `H`/`h` from, not as independent evidence. The independent oracle is the
    /// hand-computed `H = 0`, `h = H0`.
    #[cfg(feature = "survival")]
    #[test]
    fn prestart_tte_time_is_scored_on_share_dedicated_and_predict_survival() {
        use crate::types::{EventType, ObsRecord};

        let model = joint_pktte_ode_model();
        let ode = model.ode_spec.as_ref().expect("model has an [odes] block");
        let (chz_state, hazard) = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: h @ HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => (*chz_state, h),
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };
        let p = &model.default_params;
        let theta = &p.theta;
        // `H0` carries no η, and at the seeded state `central = 0`, so `h = H0` for any η.
        // η = 0 also matches what `predict_survival` evaluates at.
        let eta = [0.0_f64];
        let tv_h0 = theta[3];
        assert_eq!(
            tv_h0, 0.01,
            "theta[3] must be TVH0 — the arms read h against it"
        );

        for (arm, t_pre, records) in [
            (
                "left truncation: Exact @20, entry 5",
                5.0_f64,
                vec![ObsRecord::Event {
                    time: 20.0,
                    event_type: EventType::Exact,
                    entry_time: 5.0,
                    cmt: 2,
                }],
            ),
            (
                "interval censoring: (5, 20]",
                5.0_f64,
                vec![ObsRecord::Event {
                    time: 20.0,
                    event_type: EventType::IntervalCensored {
                        left: 5.0,
                        right: 20.0,
                    },
                    entry_time: 0.0,
                    cmt: 2,
                }],
            ),
            (
                "right censoring at 8, before the first record",
                8.0_f64,
                vec![ObsRecord::Event {
                    time: 8.0,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: 2,
                }],
            ),
            // `left = 0.0` specifically, because the two bounds are gated differently. The
            // value also has to stay `0.0` for a second reason: it makes this arm's
            // `predict_survival(&[0.0])` a zero-*maximum* grid, which is #1218's exact shape, so
            // the arm doubles as free regression coverage for PR #1222's fix. Changing it to
            // some other pre-start value (5.0 or 8.0, matching its siblings) would look like a
            // tidy-up and would silently drop that.
            //
            // `try_joint_pktte_shared_solve` pushes `entry_time` only `if *entry_time > 0.0`
            // ("no truncation"), but pushes `left`/`right` unconditionally. So on this subject
            // a `left = 0` *is* a pre-start CHZ time that reaches the share while a
            // `TENTRY = 0` never does — the asymmetry the docs sentence ("at or before") has to
            // cover, and the one a later refactor could silently remove by adding a `> 0` gate
            // to `left` for symmetry.
            (
                "interval censoring with left = 0 (ungated bound)",
                0.0_f64,
                vec![ObsRecord::Event {
                    time: 20.0,
                    event_type: EventType::IntervalCensored {
                        left: 0.0,
                        right: 20.0,
                    },
                    entry_time: 0.0,
                    cmt: 2,
                }],
            ),
        ] {
            let subject = late_start_joint_subject(records);

            // The straddle, asserted rather than assumed: the time under test is before
            // the integration start, and the subject still has an observation at or after
            // it, so the pre-start read goes through the fill (see the helper's note).
            let start = crate::ode::subject_integration_start(&subject);
            assert!(
                t_pre < start,
                "{arm}: fixture must straddle the start — t_pre {t_pre} vs start {start}"
            );
            assert!(
                subject.obs_times.iter().any(|&t| t >= start),
                "{arm}: fixture must keep an observation at or after the start"
            );

            // Engine 1 — the #570 one-solve share.
            let share = try_joint_pktte_shared_solve(&model, &subject, theta, &eta)
                .expect("model and subject must qualify for the #570 shared solve");
            let i = share
                .times
                .iter()
                .position(|&t| t == t_pre)
                .unwrap_or_else(|| panic!("{arm}: {t_pre} must be a shared-solve CHZ time"));
            let st = &share.chz_states[i];
            // Two assertions, not one `&&`: a width mismatch is a different defect from a
            // non-finite value, and a combined message would name the wrong one (printing a
            // perfectly finite `[0.0, 0.0]` under "non-finite") and send the next reader into
            // the wrong engine.
            assert_eq!(
                st.len(),
                ode.n_states,
                "{arm}: share state has {} slots, expected {}",
                st.len(),
                ode.n_states
            );
            assert!(
                st.iter().all(|x| x.is_finite()),
                "{arm}: share left the pre-start state non-finite: {st:?}"
            );
            let seeded = ode.initial_state(&share.pk_values);
            for j in 0..ode.n_states {
                assert_eq!(
                    st[j].to_bits(),
                    seeded[j].to_bits(),
                    "{arm}: share state slot {j} = {} must be the seeded {}",
                    st[j],
                    seeded[j]
                );
            }
            // `H` and `h` read exactly as `tte_ode_nll_from_shared` reads them.
            let mut du = vec![0.0; ode.n_states];
            (ode.rhs)(st, &share.pk_values, t_pre, &mut du);
            let (share_cum, share_haz) = (st[chz_state], du[chz_state]);

            // Engine 2 — the dedicated two-solve arm.
            let (cum_ded, haz_ded) = crate::survival::ode_cumhaz_hazard(
                &model,
                &subject,
                chz_state,
                theta,
                &eta,
                &[t_pre],
            );

            // Caller 3 — the public `predict_survival` (η = 0), on the same dense engine.
            let population = crate::types::Population {
                subjects: vec![subject.clone()],
                covariate_names: Vec::new(),
                dv_column: "DV".into(),
                input_columns: Vec::new(),
                exclusions: None,
                warnings: Vec::new(),
            };
            let surv = crate::api::predict_survival(&model, &population, p, &[t_pre]);
            let row = surv
                .iter()
                .find(|r| r.cmt == 2 && r.time == t_pre)
                .unwrap_or_else(|| panic!("{arm}: predict_survival must report CMT 2 @{t_pre}"));

            // Every leg named separately, and finite before any comparison — a fold or a
            // `max` would let a NaN leg ride along on the ones that worked.
            for (leg, cum, haz) in [
                ("share (#570)", share_cum, share_haz),
                ("dedicated (ode_cumhaz_hazard)", cum_ded[0], haz_ded[0]),
                ("predict_survival", row.cum_hazard, row.hazard),
            ] {
                assert!(
                    cum.is_finite() && haz.is_finite(),
                    "{arm} / {leg}: H = {cum}, h = {haz} — must be finite"
                );
                // Nothing has acted on the system yet: `H = 0`, and `h = H0·exp(BETA·0)`
                // is `H0` bit-exactly (`exp(0.0) == 1.0`). Hand-computed, not read off
                // another engine.
                assert_eq!(cum, 0.0, "{arm} / {leg}: H({t_pre}) must be exactly 0");
                assert_eq!(haz, tv_h0, "{arm} / {leg}: h({t_pre}) must be exactly TVH0");
            }

            // And the per-record NLL agrees between the two engines that actually score.
            let shared_tte = tte_ode_nll_from_shared(ode, &share, chz_state, &subject.obs_records);
            let dedicated_tte = tte_endpoint_nll(
                &model,
                &subject,
                hazard,
                crate::types::TteRecurrence::Single,
                &subject.obs_records,
                theta,
                &eta,
            );
            assert!(
                shared_tte.is_finite() && dedicated_tte.is_finite(),
                "{arm}: TTE NLL must be finite — share {shared_tte}, dedicated {dedicated_tte}"
            );
            assert!(
                (shared_tte - dedicated_tte).abs() <= 1e-4 * dedicated_tte.abs().max(1.0),
                "{arm}: share {shared_tte} must match dedicated {dedicated_tte} to solver tol"
            );

            // The FOCEI Laplace objective, which adds the ODE-TTE **FD-Hessian** term. Without
            // this leg every assertion above is on the value path: before the fix the pre-start
            // node returned the `1e20` sentinel, so a Hessian built by perturbing η around it was
            // garbage rather than merely a wrong scalar — and a regression that reinstated the
            // sentinel only inside the perturbation would pass all the value-path checks. The
            // Tier-2 end-to-end test cannot cover this either: its model declares no omega, so no
            // Hessian term is ever formed. A zero Gaussian sensitivity matrix is fine here — the
            // prior plus the TTE Hessian keep H̃ positive-definite (same setup as
            // `joint_pktte_ode_hazard_nll_paths_finite`, which does this for `entry_time = 0`).
            let n_obs = subject.observations.len();
            let foce = foce_subject_nll(
                &model,
                &subject,
                theta,
                &DVector::from_vec(vec![0.0]),
                &DMatrix::<f64>::zeros(n_obs, 1),
                &p.omega,
                &p.sigma.values,
                &p.residual_correlations,
                true,
            );
            // `is_finite()` alone would be useless here: the TTE sentinel is `1e20`, which *is*
            // finite, so a Laplace objective built entirely on repelled records passes a
            // finiteness check. Bound it below the sentinel instead. `1e12` is ~8 orders under
            // `1e20` and ~10 orders over this fixture's actual objective (order 10), so it
            // discriminates "scored" from "repelled" without being a tolerance test.
            assert!(
                foce.is_finite() && foce.abs() < 1e12,
                "{arm}: FOCEI Laplace objective (incl. the TTE FD-Hessian) must be scored, not \
                 repelled; got {foce} (the TTE sentinel is 1e20 and is itself finite)"
            );
        }
    }

    /// #1223 positive control: the *convention* the fix adopts, as a differential pair
    /// that straddles the integration start.
    ///
    /// A pre-start `entry_time` must contribute **nothing** — `H(entry) = 0`, so the
    /// left-truncated `H(T) − H(entry)` equals the untruncated `H(T)` — while a post-start
    /// one must change the objective. Without the second leg the first is satisfied by any
    /// implementation that ignores `entry_time` entirely; without the first, by the
    /// alternative convention of starting the hazard clock at the earliest TTE time, which
    /// would make `H(entry) > 0` and shift the NLL.
    #[cfg(feature = "survival")]
    #[test]
    fn prestart_entry_time_contributes_nothing_post_start_entry_does() {
        use crate::types::{EventType, ObsRecord};

        let model = joint_pktte_ode_model();
        let p = &model.default_params;
        let eta = [0.0_f64];

        let nll_at = |entry_time: f64| -> f64 {
            let subject = late_start_joint_subject(vec![ObsRecord::Event {
                time: 20.0,
                event_type: EventType::Exact,
                entry_time,
                cmt: 2,
            }]);
            individual_nll(&model, &subject, &p.theta, &eta, &p.omega, &p.sigma.values)
        };

        let (none, pre, post) = (nll_at(0.0), nll_at(5.0), nll_at(12.0));
        assert!(
            none.is_finite() && pre.is_finite() && post.is_finite(),
            "every leg must be finite: none {none}, pre {pre}, post {post}"
        );

        // The integration starts at the dose (t = 10), so 5 is before it and 12 after —
        // the pair genuinely straddles the gate rather than sitting on one side of it.
        let subject = late_start_joint_subject(Vec::new());
        let start = crate::ode::subject_integration_start(&subject);
        assert!(
            5.0 < start && 12.0 > start,
            "the pair must straddle the integration start ({start})"
        );

        // Pre-start entry contributes nothing, and does so **bit-exactly** — measured
        // (`306.2186273580891` both ways, difference `0.0`), and bit-exact by construction
        // rather than by luck: `entry_time = 5` adds a node to the shared solve's CHZ time
        // vector, but that node is filled before the break walk and lies in no segment, and
        // the horizon `t_last = max(obs 24, chz) = 24` is unchanged either way — so the
        // break timeline, every segment and the Gaussian solve are identical. A tolerance
        // here would be strictly weaker with nothing bought for it.
        assert_eq!(
            pre.to_bits(),
            none.to_bits(),
            "a pre-start entry must contribute nothing: entry=5 {pre} vs entry=0 {none}"
        );
        // Post-start entry: the objective must move by `H(12) > 0`. Measured separation is
        // 1.363e-3 relative, so the `1e-6` floor is ~1300x below what a real contribution
        // produces — loose enough not to be a tolerance test, tight enough that "the entry
        // time was ignored" (separation 0) cannot pass.
        assert!(
            (post - none).abs() > 1e-6,
            "a post-start entry must change the objective: entry=12 {post} vs entry=0 {none}"
        );
    }

    /// #1187, at the objective: the joint PK-TTE hazard has **two** production routes and
    /// they resolve infusions differently.
    ///
    /// Arm A is the #570 shared single solve (`ode_predictions_and_chz` →
    /// `integrate_segment` → `active_infusions`). Arm B is the established two-solve
    /// fallback (`tte_endpoint_nll` → `ode_cumhaz_hazard` → `ode_dense_solve_states` →
    /// `gated_infusions`). Only arm B resolved through the unguarded helper, so for a PK
    /// block with a **built-in absorption forcing fed by a `RATE>0` dose** the two arms
    /// integrated different drug exposures and produced different cumulative hazards —
    /// a wrong *objective*, not just a wrong diagnostic.
    ///
    /// The pairing matters beyond this bug: `model_uses_time_anywhere` (#1166) decides
    /// which arm a time-dependent-hazard model takes, so a divergence here is reachable
    /// by a routing change alone. Pinning A ≡ B is what makes that routing safe.
    #[cfg(feature = "survival")]
    #[test]
    fn joint_pktte_hazard_arms_agree_for_infusion_into_absorption() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{DoseEvent, EventType, ObsRecord};

        // `first_order(ka=KA)` puts the absorption kernel *in* central (no depot state),
        // so CMT 1 is the input-rate compartment and a `RATE>0` dose there is #719 gap 2.
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
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central
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
        // Replace the bolus with an infusion into the input-rate compartment. This is the
        // one input that matters — with a bolus both arms already agree.
        let amt = subject.doses[0].amt;
        subject.doses = vec![DoseEvent::new(0.0, amt, 1, amt / 2.0, false, 0.0)];
        assert!(
            subject.doses[0].is_infusion(),
            "fixture must carry a real infusion or it tests nothing"
        );
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];

        let p = &model.default_params;
        let eta_nz = [0.3_f64]; // non-zero, so the PK trajectory (hence the hazard) is non-trivial

        let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &eta_nz)
            .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
        let (chz_state, hazard) = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: h @ HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => (*chz_state, h),
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };
        let arm_a = tte_ode_nll_from_shared(
            model.ode_spec.as_ref().unwrap(),
            &share,
            chz_state,
            &subject.obs_records,
        );
        let arm_b = tte_endpoint_nll(
            &model,
            &subject,
            hazard,
            crate::types::TteRecurrence::Single,
            &subject.obs_records,
            &p.theta,
            &eta_nz,
        );
        assert!(
            arm_a.is_finite() && arm_b.is_finite(),
            "both hazard arms must be finite (A={arm_a}, B={arm_b})"
        );
        // Non-degeneracy: a hazard that never left its baseline would let the two arms agree
        // for the wrong reason. Both inputs are read back out of the fixture rather than
        // inlined, so editing `TVH0` or the event time above cannot silently leave this
        // guard comparing against a stale baseline.
        let h0 = p.theta[3]; // TVH0 — 4th `theta` declared in `src` above
        assert!(
            (h0 - 0.01).abs() < 1e-12,
            "theta[3] is {h0}, not TVH0 = 0.01 — the `[parameters]` order in this fixture changed"
        );
        let t_event = match &subject.obs_records[0] {
            ObsRecord::Event { time, .. } => *time,
            _ => panic!("fixture must carry an exact event record"),
        };
        // Drug-free exact-event NLL is `H(t) − ln h(t)`, which at `BETA·C = 0` is
        // `H0·t − ln H0`.
        let drug_free = h0 * t_event - h0.ln();
        assert!(
            (arm_b - drug_free).abs() > 1e-2,
            "TTE term {arm_b} sits at its drug-free value {drug_free} — the infusion is not \
             reaching the hazard, so the arms would agree vacuously"
        );
        assert!(
            (arm_a - arm_b).abs() <= 1e-4 * arm_b.abs().max(1.0),
            "joint PK-TTE hazard arms disagree: shared solve {arm_a} vs dedicated {arm_b} \
             (#1187 — the dedicated arm resolves infusions through `gated_infusions`)"
        );
    }

    /// #1187 external anchor: the joint PK-TTE **cumulative hazard** under a `RATE>0` dose
    /// into a built-in absorption compartment, against NONMEM 7.6.0.
    ///
    /// [`joint_pktte_hazard_arms_agree_for_infusion_into_absorption`] pins the two ferx arms
    /// to each other; two engines agreeing is not evidence that either is right, so this
    /// pins them to a tool that has no such split — NONMEM integrates the augmented system
    /// once. `A(3)` (the CHZ compartment) is compared rather than the OFV: a likelihood
    /// model (`F_FLAG=1`) carries different additive constants in the two tools, while the
    /// cumulative hazard is constant-free.
    ///
    /// Reference: `nonmem_anchor/pktte_inf.{ctl,csv}` — `ADVAN13 TOL=9`, explicit
    /// `DEPOT/CENTRAL/CHZ` compartments, `MAXEVAL=0 POSTHOC`, every `THETA FIX` and
    /// `$OMEGA 0 FIX` so both tools evaluate at `eta = 0`. NONMEM's zero-order fill of an
    /// explicit depot followed by first-order `KA` is exactly what ferx's `first_order()`
    /// kernel plus the `R_in_inf` convolution reproduces — the equivalence already anchored
    /// by `tests/infusion_absorption_nonmem_anchor.rs`.
    ///
    /// Measured: before the fix the dedicated arm read `H(1.86) = 3.796844e0` against
    /// NONMEM's `1.2217e-1` — **31×** — while the shared arm was already correct. The
    /// hazard is exponential in concentration (`H0·exp(BETA·C)`), so the doubled input
    /// rate compounds rather than scaling linearly.
    #[cfg(feature = "survival")]
    #[test]
    fn joint_pktte_cumulative_hazard_matches_nonmem_for_infusion_into_absorption() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{DoseEvent, EventType, ObsRecord};

        // Mirrors `nonmem_anchor/pktte_inf.ctl`: CL=1, V=10, KA=1, H0=0.02, BETA=0.5.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-10
";
        let model = parse_model_string(src).expect("joint PK-TTE model must parse");

        // The NONMEM dataset, row for row: AMT=100 RATE=50 into CMT 1 (T = 2 h), PK
        // observations at 0.5…18, and the exact event at t = 1.86.
        let mut subject = make_simple_subject();
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)];
        assert!(
            subject.doses[0].is_infusion(),
            "fixture must carry a real infusion or it tests nothing"
        );
        subject.obs_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0];
        subject.observations = vec![
            4.6271, 6.2626, 7.1063, 8.0446, 6.3602, 4.8718, 3.6207, 2.0972,
        ];
        subject.obs_cmts = vec![1; 8];
        subject.cens = vec![0; 8];
        subject.occasions = vec![1; 8];
        subject.obs_records = vec![ObsRecord::Event {
            time: 1.86,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];

        let theta = &model.default_params.theta;
        let eta = [0.0_f64]; // $OMEGA 0 FIX on the NONMEM side

        let chz_state = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => *chz_state,
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };

        // NONMEM 7.6.0, `nonmem_anchor/results/pktte_inf.tab`: CHZ at TIME = 1.86.
        //
        // Printed at 9 significant figures via `FORMAT=s1PE15.8` in the control stream —
        // deliberately, and load-bearing. `$TABLE`'s default is 5 significant figures, which
        // would bank this as `1.2217e-1` and carry ±5e-6 of its **own rounding**: 4.1e-5
        // relative — a floor under any tolerance this anchor could assert, regardless of how
        // close ferx actually is. At 9 figures the reference's own contribution drops to
        // ~1e-9 and the bound below measures ferx instead of NONMEM's printf.
        const NONMEM_CHZ_AT_EVENT: f64 = 1.22168018e-1;

        // Arm B — the two-solve fallback, i.e. the arm that carried the defect.
        let (cum_dedicated, _haz) =
            crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, theta, &eta, &[1.86]);
        // Arm A — the #570 shared single solve.
        let share = try_joint_pktte_shared_solve(&model, &subject, theta, &eta)
            .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
        assert_eq!(share.times, vec![1.86], "share must cover the event time");
        let cum_shared = share.chz_states[0][chz_state];

        // Per-arm bounds, each measured rather than chosen, because the two arms are not
        // equally accurate: the dedicated arm lands at **1.41e-9** and the shared arm at
        // **3.26e-6** — a 2300x spread. Both are far inside the arms-agree tolerance and
        // neither is a defect; they integrate different timelines (the share covers the
        // union of TTE record times in one pass, the dedicated call solves to `[1.86]`
        // alone), so their local error differs. A single loose bound covering both would
        // let a real regression in the tight arm pass unnoticed, which is the whole reason
        // they are separated here.
        for (arm, got, tol) in [
            ("dedicated (ode_cumhaz_hazard)", cum_dedicated[0], 1e-8),
            ("shared (#570)", cum_shared, 1e-5),
        ] {
            let rel = (got - NONMEM_CHZ_AT_EVENT).abs() / NONMEM_CHZ_AT_EVENT;
            assert!(
                rel < tol,
                "H(1.86) on the {arm} arm: ferx {got:.9e} vs NONMEM \
                 {NONMEM_CHZ_AT_EVENT:.9e} (rel {rel:.2e}, bound {tol:.0e})"
            );
        }
    }

    /// #570 guard regression (found by an independent review of #613): a joint PK-TTE
    /// model whose `[individual_parameters]` references the `TIME` built-in must NOT
    /// take the shared single-solve path. The standalone prediction path routes such a
    /// model through the per-event, TIME-resolved `ode_predictions_event_driven`
    /// (`pk/mod.rs`), so the share's single t=0 PK snapshot would be silently wrong
    /// (~30–50% off on a time-switching parameter — no error, no NaN, just a wrong
    /// OFV). The guard must return `None` so the established two-solve path runs.
    #[cfg(feature = "survival")]
    #[test]
    fn joint_pktte_share_rejects_time_builtin_model() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        // Same joint PK-TTE shape as `joint_pktte_ode_hazard_nll_paths_finite`, but CL
        // switches on TIME, so the prediction path is genuinely time-dependent.
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
  STEP = if (TIME > 3) 1 else 0
  CL   = TVCL * (1 + STEP) * exp(ETA_CL)
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
        let model = parse_model_string(src).expect("TIME-using joint PK-TTE model must parse");
        // Sanity: the fixture really does trip the TIME built-in detector.
        assert!(
            crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
            "fixture must use the TIME built-in"
        );
        let mut subject = make_simple_subject();
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let p = &model.default_params;

        // The guard must reject the share → established two-solve fallback.
        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_none(),
            "a TIME-using joint PK-TTE model must not take the #570 shared-solve path"
        );
        // The fallback still yields a finite NLL (and η still moves it).
        let nll = individual_nll(
            &model,
            &subject,
            &p.theta,
            &[0.0],
            &p.omega,
            &p.sigma.values,
        );
        let moved = individual_nll(
            &model,
            &subject,
            &p.theta,
            &[0.3],
            &p.omega,
            &p.sigma.values,
        );
        assert!(
            nll.is_finite() && moved.is_finite() && (nll - moved).abs() > 1e-9,
            "fallback joint NLL must be finite and η-sensitive (nll={nll}, moved={moved})"
        );
    }

    /// The same rejection, for model time read in the **`[odes]` RHS** rather than
    /// in `[individual_parameters]`.
    ///
    /// The sibling above cannot pin this: its fixture puts `TIME` in an individual
    /// parameter, where the narrow `compiled_model_uses_time_builtin` and the wide
    /// `pk::model_uses_time_anywhere` both return `true`, so it passes under either
    /// and the choice of predicate is invisible to it. Here only the wide one fires,
    /// which is the whole difference — and reverting the gate to the narrow
    /// predicate left the entire suite green before this test existed.
    ///
    /// Why the rejection is required: the shared solve (#570) computes the Gaussian
    /// predictions from the plain no-TV `ode_predictions` with a single `t=0` PK
    /// snapshot, so it is only equivalent when the standalone prediction path takes
    /// that same route. Since #1124 an `[odes]` RHS reading model time routes to the
    /// event-driven walker instead, so admitting the share here would leave the
    /// objective's Gaussian term on the dense engine while the reported IPRED came
    /// from the event-driven one.
    #[test]
    #[cfg(feature = "survival")]
    fn joint_pktte_share_rejects_model_time_in_the_odes_rhs() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        // Same shape as the sibling, but the time dependence lives in the RHS and
        // the individual parameters are entirely time-independent. The hazard is
        // deliberately time-independent too, so the only thing this fixture can be
        // rejected *for* is the PK RHS. Since #1166 a `TIME`-reading hazard no
        // longer trips the predicate at all — the injected `d/dt(__chz_n)` line is
        // excluded from it — which is what `joint_pktte_share_admits_a_time_
        // dependent_hazard` below pins from the other side.
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
  d/dt(central) =  KA * depot - (CL/V) * central * (1.0 + 0.3*TAD)
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
        let model = parse_model_string(src).expect("TAD-in-RHS joint PK-TTE model must parse");

        // The discriminating preconditions. Without these the test would pass for
        // a model on which the two predicates agree, and would not pin the gate's
        // choice of predicate at all.
        assert!(
            !crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
            "the narrow predicate must NOT fire — the individual parameters are \
             time-independent"
        );
        assert!(
            crate::pk::model_uses_time_anywhere(&model),
            "…while the wide one must, via the `[odes]` RHS"
        );

        let mut subject = make_simple_subject();
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let p = &model.default_params;

        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_none(),
            "a joint PK-TTE model whose `[odes]` RHS reads model time must not take \
             the #570 shared-solve path"
        );
        // …and the two-solve fallback still produces a finite, η-sensitive NLL.
        let nll = individual_nll(
            &model,
            &subject,
            &p.theta,
            &[0.0],
            &p.omega,
            &p.sigma.values,
        );
        let moved = individual_nll(
            &model,
            &subject,
            &p.theta,
            &[0.3],
            &p.omega,
            &p.sigma.values,
        );
        assert!(
            nll.is_finite() && moved.is_finite() && (nll - moved).abs() > 1e-9,
            "fallback joint NLL must be finite and η-sensitive (nll={nll}, moved={moved})"
        );
    }

    /// A joint PK-TTE model, `{HAZ}` substituted. Autonomous PK block throughout —
    /// the hazard is the only thing that ever reads time here.
    #[cfg(feature = "survival")]
    fn joint_time_hazard_src(haz: &str) -> String {
        format!(
            r"
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
  hazard = {haz}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"
        )
    }

    /// #1166, the admission side: a Gompertz-shaped hazard — a hazard that reads
    /// `TIME` by definition — on an autonomous PK block **takes** #570's single
    /// shared solve.
    ///
    /// The sibling above pins the opposite direction (a `TAD`-reading PK RHS is
    /// still rejected), so between them the gate is pinned from both sides and
    /// neither "always admit" nor "always decline" survives.
    #[test]
    #[cfg(feature = "survival")]
    fn joint_pktte_share_admits_a_time_dependent_hazard() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        let model = parse_model_string(&joint_time_hazard_src(
            "H0 * exp(0.05*TIME) * exp(BETA * (central / V))",
        ))
        .expect("Gompertz-in-TIME joint PK-TTE model must parse");

        // The discriminating precondition: the *wide* flag still fires — the
        // augmented program does read model time, and the SS gates must keep
        // seeing that — while the predicate the share asks does not.
        let prog = model
            .ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .expect("rhs program");
        assert!(
            prog.reads_model_time(),
            "the augmented RHS does read model time (the SS gates' predicate)"
        );
        assert!(
            !crate::pk::model_uses_time_anywhere(&model),
            "…but the PK block does not, so the routing predicate must be false"
        );

        let mut subject = make_simple_subject();
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let p = &model.default_params;
        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_some(),
            "a time-dependent hazard on an autonomous PK block must take the \
             #570 shared-solve path"
        );
    }

    /// An `EVID=3/4` reset **declines** the share — the routing that
    /// `tests/pktte_prestart_entry.rs` is built on, and which nothing pinned until #1223.
    ///
    /// That file's whole structure is "A1 goes through the share, A4 (the same subject plus a
    /// reset) goes through the dedicated engine, both must agree". Admit reset subjects to the
    /// share and A4 silently becomes a second run of A1's engine: every assertion there still
    /// passes, both pinned OFVs still match, and its own guard —
    /// `(A4_OFV - A1_OFV).abs() > 1e-3` — does not notice, because it shows the two *subjects*
    /// differ, not that they took different *engines*. The "both engines agree" claim would be
    /// false while reading green.
    ///
    /// The share's own precondition is why the decline is required, not merely conventional:
    /// it computes predictions via `ode_predictions_and_chz`, i.e. the no-TV `ode_predictions`
    /// with a single `t = 0` PK snapshot, and a reset needs the event-driven walker.
    ///
    /// Measured, not argued: with `|| subject.has_resets()` disabled in the admission list, this
    /// test is the **only** one that fails. The other three admission tests stay green (they
    /// cover model time and a time-dependent hazard), and so do all three tests in
    /// `tests/pktte_prestart_entry.rs` — A4 included, whose whole premise is that it runs the
    /// dedicated engine.
    #[cfg(feature = "survival")]
    #[test]
    fn joint_pktte_share_rejects_a_subject_with_resets() {
        use crate::types::{EventType, ObsRecord};

        let model = joint_pktte_ode_model();
        let p = &model.default_params;
        let records = vec![ObsRecord::Event {
            time: 20.0,
            event_type: EventType::Exact,
            entry_time: 5.0,
            cmt: 2,
        }];

        // Without the reset the subject qualifies — the straddle that makes the next
        // assertion about the reset rather than about something else in the admission list.
        let subject = late_start_joint_subject(records.clone());
        assert!(
            !subject.has_resets(),
            "control subject must carry no resets"
        );
        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_some(),
            "the reset-free subject must qualify for the share, or this test proves nothing \
             about resets"
        );

        // With one, it must not.
        let mut reset_subject = late_start_joint_subject(records);
        reset_subject.reset_times = vec![15.0];
        assert!(
            reset_subject.has_resets(),
            "fixture must actually carry a reset"
        );
        assert!(
            try_joint_pktte_shared_solve(&model, &reset_subject, &p.theta, &[0.0]).is_none(),
            "an EVID=3/4 reset must decline the #570 share — the share takes a single t=0 PK \
             snapshot, and a reset needs the event-driven walker"
        );
    }

    /// #1166, the routing oracle: two models that compute the **same arithmetic**
    /// must land on the same engine, and therefore agree bit-for-bit.
    ///
    /// `* (1.0 + 0.0*TIME)` is the exact identity in IEEE arithmetic — `0.0*t` is
    /// `0.0` for finite `t`, `1.0 + 0.0` is `1.0`, and `h * 1.0` is `h` — and the
    /// RHS is interpreted bytecode, so nothing folds it away. Any difference in
    /// the result is therefore a difference of *engine*, not of arithmetic: before
    /// #1166 the flagged twin declined the shared solve and assembled the Gaussian
    /// half on the event-driven walk, which agreed with the dense driver only to
    /// solver tolerance (measured: 0 of 40 subjects bit-identical on the anchored
    /// fixture, worst per-subject 3.25e-4 at ferx's default tolerances).
    ///
    /// A value test could not see this — both arms are correct to tolerance — and
    /// a routing assertion alone could not either, since it would pass for a
    /// predicate that admitted the *wrong* pair too. Bit-identity is the assertion
    /// that is exactly as strong as the claim.
    #[test]
    #[cfg(feature = "survival")]
    fn identical_arithmetic_takes_the_identical_engine_bit_for_bit() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        // The pair must **straddle** the predicate under the old behaviour, or the
        // test is vacuous: with a time-reading base hazard both twins would have
        // been flagged, both would have declined, and they would agree bit-for-bit
        // whether the fix is present or not. (Found by mutating the fix away and
        // watching this test survive.) So the base hazard is autonomous and only
        // the twin's identity factor reads time.
        let plain = parse_model_string(&joint_time_hazard_src("H0 * exp(BETA * (central / V))"))
            .expect("parse");
        let twin = parse_model_string(&joint_time_hazard_src(
            "H0 * exp(BETA * (central / V)) * (1.0 + 0.0*TIME)",
        ))
        .expect("parse");
        // Pin the straddle itself, so a later predicate change that stopped the
        // twin from being *flagged at all* would show up here rather than turning
        // this back into a tautology.
        let flagged = |m: &crate::types::CompiledModel| {
            m.ode_spec
                .as_ref()
                .and_then(|o| o.rhs_program.as_ref())
                .expect("rhs program")
                .reads_model_time()
        };
        assert!(
            !flagged(&plain) && flagged(&twin),
            "the twin must trip the wide flag and the plain model must not — that \
             difference is what the fix has to make invisible"
        );

        let mut subject = make_simple_subject();
        subject.obs_records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let p = &plain.default_params;
        let q = &twin.default_params;

        for eta in [0.0, 0.3, -0.25] {
            let a = individual_nll(
                &plain,
                &subject,
                &p.theta,
                &[eta],
                &p.omega,
                &p.sigma.values,
            );
            let b = individual_nll(&twin, &subject, &q.theta, &[eta], &q.omega, &q.sigma.values);
            assert!(
                a.is_finite(),
                "the fixture must produce a real objective (eta={eta}, nll={a})"
            );
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "an arithmetically identical hazard must take the same engine \
                 (eta={eta}, plain={a}, identity-twin={b}, delta={})",
                a - b
            );
        }
    }

    /// #1218 on the fit path. `tte_ode_nll` is the two-solve arm `tte_endpoint_nll` takes when
    /// `try_joint_pktte_shared_solve` declines — time-varying covariates, EVID-3/4 resets, a
    /// model-time read, SDE, FREM. A subject whose only TTE record sits on its first dose
    /// builds `times = [0.0]`, a one-break timeline in `ode_dense_solve_states`, and until
    /// #1218 read `NaN` there, which `tte_nll_from_curves` maps to the `1e20` sentinel: the
    /// subject was repelled, not scored. It now scores `H(0) − ln h(0)` with `h(0)` the
    /// post-dose hazard — the number the multi-point grid already gave.
    ///
    /// An exact event rather than a censor, on purpose: a censor at `t = 0` scores
    /// `H(0) = 0` under the fix *and* under the wrong fix (filling the row from the seeded
    /// pre-dose state), while `−ln h(0)` sees the difference — `h(0) = H0·exp(BETA·C)` at the
    /// SS trough + pulse against the drug-free `H0`. Measured: `1e20` before, and under the
    /// seeded-state variant `−ln 0.02 = 3.91` where the post-dose row gives `−ln 25.6`.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_ode_nll_scores_an_event_on_the_first_dose_instead_of_the_sentinel() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{DoseEvent, EventType, ObsRecord};

        // #1210's drug arm, IV: CL=1, V=10, H0=0.02, BETA=0.5, hazard on central/V.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
";
        let model = parse_model_string(src).expect("joint PK-TTE model must parse");

        // An `SS=1` bolus at t = 0 (so drug is present on the incoming side of the record)
        // and nothing else: no PK rows, one exact event on the dose instant.
        let mut subject = make_simple_subject();
        subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        subject.obs_times = Vec::new();
        subject.observations = Vec::new();
        subject.obs_cmts = Vec::new();
        subject.cens = Vec::new();
        subject.occasions = Vec::new();
        subject.obs_records = vec![ObsRecord::Event {
            time: 0.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];

        let theta = &model.default_params.theta;
        let eta = [0.0_f64];
        let chz_state = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => *chz_state,
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };

        let nll = tte_ode_nll(
            &model,
            &subject,
            chz_state,
            &subject.obs_records,
            theta,
            &eta,
        );
        assert!(
            nll.is_finite() && nll < 1e20,
            "an event on the first dose is scored with the sentinel, not a likelihood: {nll}"
        );

        // The reference: the same H/h read off a grid that also asks for a later point.
        let (cum, haz) = crate::survival::ode_cumhaz_hazard(
            &model,
            &subject,
            chz_state,
            theta,
            &eta,
            &[0.0, 1.0],
        );
        assert!(
            cum[0].is_finite() && haz[0].is_finite(),
            "the multi-point reference row is not finite: {}, {}",
            cum[0],
            haz[0]
        );
        // The straddle: the post-dose hazard must be far from the drug-free `H0 = 0.02`, or
        // this test could not tell the post-dose row from a seeded one.
        assert!(
            haz[0] > 5.0 * 0.02,
            "h(0) = {} is not distinguishable from the drug-free hazard",
            haz[0]
        );
        let want = cum[0] - haz[0].ln();
        assert!(
            (nll - want).abs() < 1e-12,
            "tte_ode_nll = {nll} but H(0) - ln h(0) from the multi-point grid = {want}"
        );
    }
}
