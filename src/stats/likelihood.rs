use crate::pk;
use crate::stats::residual_error::{residual_rd, residual_rd2, ResidualCov};
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Predictions through analytical PK or the ODE solver, honouring per-event PK
/// parameters under time-varying covariates (`pk::compute_predictions_with_tv`).
/// Canonical predictions entry for FOCE/FOCEI inner-loop objectives; pass the same
/// `(theta, eta)` used elsewhere in the NLL. Allocates a fresh `EventPkParams`; hot
/// loops should use [`model_predictions_into`].
#[inline]
fn model_predictions(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Vec<f64> {
    pk::compute_predictions_with_tv(model, subject, theta, eta)
}

#[inline]
pub(crate) fn m3_logcdf(limit: f64, f: f64, sd: f64, cens: i8) -> f64 {
    let z = if cens < 0 { (f - limit) / sd } else { (limit - f) / sd };
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
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64],
) -> f64 {
    // Allocate-on-each-call wrapper — see `individual_nll_into` for
    // the scratch-aware version used by SAEM's MH loop.
    let mut scratch = pk::EventPkParams::with_capacity_for(subject);
    individual_nll_into(model, subject, theta, eta, omega, sigma_values, &mut scratch)
}

/// [`individual_nll`] with a caller-owned scratch buffer (SAEM MH hot path: one
/// buffer reused across all proposed `eta`s).
pub fn individual_nll_into(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64], scratch: &mut pk::EventPkParams,
) -> f64 {
    individual_nll_into_with_schedule_cov(
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
        model.residual_cov_theta(theta),
        scratch,
        None,
    )
}

/// TTE NLL for an ODE-accumulated hazard (joint PK-TTE, #564): solves the augmented
/// ODE at `(theta, eta)`, reads `H(t) = u[chz_state]` (and `h = u̇[chz_state]` for exact
/// events) and feeds `crate::survival::tte_nll_from_curves`. Re-solves every call.
///
/// **Test-only** since #901: the reference the dedicated arm of
/// [`tte_endpoint_nll_visit`] must equal bit for bit.
#[cfg(all(test, feature = "survival"))]
fn tte_ode_nll(
    model: &CompiledModel, subject: &Subject, chz_state: usize, records: &[ObsRecord],
    theta: &[f64], eta: &[f64],
) -> f64 {
    let curves = tte_ode_curves(model, subject, chz_state, records, theta, eta);
    tte_ode_nll_from_curves(records, &curves.times, &curves.cum, &curves.haz, curves.tol)
}

/// `H`/`h` sampled at a TTE endpoint's record times, with the monotone floor to score
/// them under, from the dedicated solve: what the dedicated arm of
/// [`tte_endpoint_nll_visit`] scores (#901).
#[cfg(feature = "survival")]
struct TteCurves {
    times: Vec<f64>,
    cum: Vec<f64>,
    haz: Vec<f64>,
    tol: crate::survival::MonoTol,
}

/// The dedicated-solve curves of `tte_ode_nll`: one `ode_cumhaz_hazard` integration
/// at `(theta, eta)`, read at the endpoint's sorted-unique record times.
#[cfg(feature = "survival")]
fn tte_ode_curves(
    model: &CompiledModel, subject: &Subject, chz_state: usize, records: &[ObsRecord],
    theta: &[f64], eta: &[f64],
) -> TteCurves {
    // Distinct times where H(t) — and h(t) for exact events — is needed.
    let mut times: Vec<f64> = Vec::new();
    for r in records {
        let ObsRecord::Event { time, event_type, entry_time, .. } = r else {
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
    TteCurves { times, cum, haz, tol }
}

/// Per-record TTE NLL from `H`/`h` curves sampled at sorted-unique `times` (lookups
/// binary-search exact f64 values). Shared by [`tte_ode_nll`] and the #570
/// joint-PK-TTE share so the per-record likelihood lives in one place.
#[cfg(all(test, feature = "survival"))]
fn tte_ode_nll_from_curves(
    records: &[ObsRecord], times: &[f64], cum: &[f64], haz: &[f64], tol: crate::survival::MonoTol,
) -> f64 {
    tte_ode_nll_from_curves_visit(records, times, cum, haz, tol, &mut |_| {})
}

/// `tte_ode_nll_from_curves` handing each scored record's term to `visit` (#901).
#[cfg(feature = "survival")]
fn tte_ode_nll_from_curves_visit(
    records: &[ObsRecord], times: &[f64], cum: &[f64], haz: &[f64], tol: crate::survival::MonoTol,
    visit: &mut impl FnMut(crate::survival::RecordTerm<f64>),
) -> f64 {
    let cumhaz_at = |t: f64| -> f64 {
        times.binary_search_by(|x| x.partial_cmp(&t).unwrap()).map_or(f64::NAN, |i| cum[i])
    };
    let hazard_at = |t: f64| -> f64 {
        times.binary_search_by(|x| x.partial_cmp(&t).unwrap()).map_or(f64::NAN, |i| haz[i])
    };
    crate::survival::tte_nll_from_curves_g(records, cumhaz_at, hazard_at, tol, visit)
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
    /// PK-parameter snapshot used for the solve.
    ///
    /// **Test-only.** The fit path uses `rhs_params` (this snapshot extended with dose
    /// anchors, #1261). The #1223 parity test needs the exact snapshot this solve used to
    /// assert the pre-start state equals `ode.initial_state(..)`.
    #[cfg(test)]
    pk_values: Vec<f64>,
    /// Extended parameter snapshots aligned to `times`, including the dose-time anchors
    /// the RHS reads when evaluating `h = dCHZ/dt` (#1261).
    rhs_params: Vec<[f64; crate::types::MAX_PK_PARAMS + 2]>,
}

/// Build the shared joint PK-TTE solve only when provably equivalent to the separate
/// Gaussian + `ode_cumhaz_hazard` solves: a plain ODE model with an OdeAccumulated
/// hazard and no TV covariates, EVID-3/4 resets, SDE diffusion or FREM. `None`
/// otherwise (caller keeps the two-solve path).
#[cfg(feature = "survival")]
fn try_joint_pktte_shared_solve(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Option<JointPkTteSolve> {
    let ode = model.ode_spec.as_ref()?;
    // Equivalent only when the standalone prediction path takes the same no-TV
    // `ode_predictions` snapshot route. Model time or TV covariates route through
    // `ode_predictions_event_driven`; EVID-3/4 need the event-driven walker; SDE adds EKF
    // noise; FREM rewrites pseudo-obs predictions. The model-time clause must be the
    // **wide** predicate matching `pk/mod.rs` routing: the narrow one missed `[odes]`
    // reading TAD/TIME after #1124.
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
        matches!(e, EndpointLikelihood::Tte { hazard: HazardSpec::OdeAccumulated { .. }, .. })
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
            let ObsRecord::Event { time, event_type, entry_time, cmt: rc } = r else {
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
    let dose_lagtimes = crate::ode::predictions::dose_lagtimes_for(subject, ode, &pk.values);
    let first_dose_time = crate::ode::predictions::earliest_dose_time(&subject.doses);
    let rhs_params = times
        .iter()
        .map(|&t| {
            crate::ode::predictions::rhs_ext_params_at(
                &subject.doses,
                &dose_lagtimes,
                first_dose_time,
                &pk.values,
                t,
            )
        })
        .collect();
    // Apply exactly the post-processing the standalone no-TV ODE prediction path does
    // (compute_predictions_with_tv_into_with_schedule): init impulse, then the sink's
    // shared tail ([scaling], LTBS, FREM override). No-op for Form-C / no-init /
    // linear-scale models; the FREM step is a no-op too, since the guard excludes FREM.
    crate::pk::add_analytical_init(model, subject, theta, eta, &mut preds);
    crate::pk::to_observation_scale(model, subject, theta, eta, &mut preds);

    Some(JointPkTteSolve {
        preds,
        times,
        chz_states,
        #[cfg(test)]
        pk_values: pk.values.to_vec(),
        rhs_params,
    })
}

/// TTE NLL for an OdeAccumulated endpoint reading `H`/`h` from the shared joint solve
/// (state slot `chz_state`; bare RHS for the derivative), equal to a dedicated solve to
/// solver tolerance. Pre-start times score `H = 0` (#1223); a NaN state maps to the
/// `1e20` sentinel, matching `ode_cumhaz_hazard`.
///
/// **Test-only** since #901: [`tte_endpoint_nll_visit`]'s shared arm must equal it.
#[cfg(all(test, feature = "survival"))]
fn tte_ode_nll_from_shared(
    ode: &crate::ode::OdeSpec, share: &JointPkTteSolve, chz_state: usize, records: &[ObsRecord],
) -> f64 {
    let (cum, haz, tol) = shared_tte_curves(ode, share, chz_state);
    tte_ode_nll_from_curves(records, &share.times, &cum, &haz, tol)
}

/// The `H`/`h` curves `tte_ode_nll_from_shared` reads off the #570 shared solve, at
/// `share.times`, with the monotone floor of the configured solver tolerances.
#[cfg(feature = "survival")]
fn shared_tte_curves(
    ode: &crate::ode::OdeSpec, share: &JointPkTteSolve, chz_state: usize,
) -> (Vec<f64>, Vec<f64>, crate::survival::MonoTol) {
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
        (ode.rhs)(st, &share.rhs_params[i], t, &mut du);
        haz[i] = du[chz_state];
    }
    let tol = crate::survival::MonoTol::from_solver(&ode.effective_solver_opts());
    (cum, haz, tol)
}

/// Dispatch a TTE endpoint's per-subject NLL on its hazard representation, re-solving any
/// ODE: the closed form of `crate::survival::tte_data_term` for `Analytic`, a dedicated
/// solve for an `OdeAccumulated` (joint PK-TTE) hazard. The
/// no-share, no-visitor case of [`tte_endpoint_nll_visit`].
#[cfg(feature = "survival")]
fn tte_endpoint_nll(
    model: &CompiledModel, subject: &Subject, hazard: &HazardSpec,
    recurrence: crate::types::TteRecurrence, records: &[ObsRecord], theta: &[f64], eta: &[f64],
) -> f64 {
    tte_endpoint_nll_visit(
        model,
        subject,
        hazard,
        recurrence,
        records,
        theta,
        eta,
        None,
        &mut |_| {},
    )
}

/// **The** routing of a TTE endpoint's per-subject likelihood onto its curves (#901).
/// `Analytic` is the closed form. `OdeAccumulated` takes, in order:
///
/// 1. the dual walk at `f64` (`survival::ode_tte_nll_g`) where in scope
///    (`ode_hazard_walk_in_scope`), so objective and gradient are one function (#2482);
/// 2. otherwise the #570 shared solve when `joint_share` is `Some`;
/// 3. otherwise (or when the walk returns `None`) the dedicated solve ([`tte_ode_curves`]).
///
/// Each scored record's term goes to `visit`; the return is their fold. Every caller
/// (objective, FD perturbations, [`non_gaussian_terms`], residuals via
/// [`tte_endpoint_record_terms`]) goes through here, so residuals and objective share
/// one engine. The ODE path is Single-only (RTTE + drug hazard rejected at parse).
#[cfg(feature = "survival")]
#[allow(clippy::too_many_arguments)]
fn tte_endpoint_nll_visit(
    model: &CompiledModel, subject: &Subject, hazard: &HazardSpec,
    recurrence: crate::types::TteRecurrence, records: &[ObsRecord], theta: &[f64], eta: &[f64],
    joint_share: Option<&JointPkTteSolve>,
    visit: &mut impl FnMut(crate::survival::RecordTerm<f64>),
) -> f64 {
    match hazard {
        HazardSpec::Analytic { family, param_fn } => {
            let params = param_fn(theta, eta, &subject.covariates);
            crate::survival::analytic_tte_nll_visit_g(records, *family, recurrence, &params, visit)
        }
        HazardSpec::OdeAccumulated { chz_state } => {
            // A subject in the walk's scope is scored off the walk, whose exact derivatives
            // the gradient takes. It declines only a non-finite trajectory (#2786: pre-start
            // times and the pre-dose `TAD` anchor are served), which any other engine would
            // integrate to the same divergence: repelled.
            if crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope(model, subject) {
                return crate::survival::ode_tte_nll_visit_g::<f64>(
                    model, subject, *chz_state, records, theta, eta, visit,
                )
                .unwrap_or(SUBJECT_SENTINEL_NLL);
            }
            match joint_share {
                Some(s) => {
                    let ode = model.ode_spec.as_ref().expect("joint share ⟹ ode_spec");
                    let (cum, haz, tol) = shared_tte_curves(ode, s, *chz_state);
                    tte_ode_nll_from_curves_visit(records, &s.times, &cum, &haz, tol, visit)
                }
                None => {
                    let c = tte_ode_curves(model, subject, *chz_state, records, theta, eta);
                    tte_ode_nll_from_curves_visit(records, &c.times, &c.cum, &c.haz, c.tol, visit)
                }
            }
        }
    }
}

/// The per-record terms one TTE endpoint scores at `eta` (#901), or `None` on the
/// `1e20` sentinel. [`tte_endpoint_nll_visit`] with a collector, so the terms fold to
/// the reported NLL bit for bit.
#[cfg(feature = "survival")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tte_endpoint_record_terms(
    model: &CompiledModel, subject: &Subject, hazard: &HazardSpec,
    recurrence: crate::types::TteRecurrence, records: &[ObsRecord], theta: &[f64], eta: &[f64],
    joint_share: Option<&JointPkTteSolve>,
) -> Option<Vec<crate::survival::RecordTerm<f64>>> {
    let mut terms = Vec::new();
    let nll = tte_endpoint_nll_visit(
        model,
        subject,
        hazard,
        recurrence,
        records,
        theta,
        eta,
        joint_share,
        &mut |t| terms.push(t),
    );
    (nll < SUBJECT_SENTINEL_NLL).then_some(terms)
}

/// Every TTE record's post-fit residuals for one subject at its EBE `eta` (#901), one
/// endpoint at a time in ascending CMT. The #570 shared solve is built once, exactly as
/// the reported objective builds it (`foce_subject_nll_cov`), and each endpoint's terms
/// come from [`tte_endpoint_record_terms`]. Empty for a subject with no TTE record.
#[cfg(feature = "survival")]
pub(crate) fn tte_diagnostics(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Vec<crate::types::TteObsDiagnostic> {
    let mut rows = Vec::new();
    if !subject.obs_records.iter().any(|r| matches!(r, ObsRecord::Event { .. })) {
        return rows;
    }
    let joint_share = try_joint_pktte_shared_solve(model, subject, theta, eta);
    let mut cmts: Vec<&usize> = model.endpoints.keys().collect();
    cmts.sort();
    for cmt in cmts {
        let EndpointLikelihood::Tte { hazard, recurrence, .. } = &model.endpoints[cmt] else {
            continue;
        };
        let records: Vec<ObsRecord> = subject
            .obs_records
            .iter()
            .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
            .cloned()
            .collect();
        if records.is_empty() {
            continue;
        }
        let terms = tte_endpoint_record_terms(
            model,
            subject,
            hazard,
            *recurrence,
            &records,
            theta,
            eta,
            joint_share.as_ref(),
        );
        rows.extend(crate::survival::tte_record_diagnostics(&records, terms.as_deref(), |t| {
            subject.param_time_at(t)
        }));
    }
    rows
}

/// [`tte_endpoint_nll`] for the endpoint on `cmt`, for tests outside this module.
#[cfg(all(test, feature = "survival"))]
pub(crate) fn tte_endpoint_nll_for_tests(
    model: &CompiledModel, subject: &Subject, cmt: usize, records: &[ObsRecord], theta: &[f64],
    eta: &[f64],
) -> f64 {
    let Some(crate::types::EndpointLikelihood::Tte { hazard, recurrence, .. }) =
        model.endpoints.get(&cmt)
    else {
        panic!("no TTE endpoint on CMT {cmt}");
    };
    tte_endpoint_nll(model, subject, hazard, *recurrence, records, theta, eta)
}

/// Accumulate every non-Gaussian endpoint's data-term NLL into `acc`, times `factor`.
/// The one seam FOCEI, SAEM θ-M-step and IOV share: TTE via [`tte_endpoint_nll`],
/// discrete families via [`crate::categorical::discrete_subject_nll`].
///
/// Adds **into `acc` per term** (TTE endpoints in ascending CMT order, #2531, then the
/// discrete term) rather than returning a sum: a single `factor * sum` add would
/// reassociate and break bit-identity of the FOCEI hot path.
///
/// `factor` is the OFV-scale weight (`2·` on halved FOCEI/IOV `data_ll`, `1·` on raw
/// SAEM NLL). `joint_share` is `None` wherever the shared solve is not built; IOV
/// hazards are always `Analytic` (IOV + joint-PK-TTE rejected at parse), SAEM may carry
/// an `OdeAccumulated` endpoint, re-solved via `tte_ode_nll`.
#[cfg(feature = "survival")]
pub(crate) fn accumulate_non_gaussian_nll(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
    joint_share: Option<&JointPkTteSolve>, factor: f64, acc: &mut f64,
) {
    // Callers gate on `!subject.obs_records.is_empty()`, so an empty subject never reaches
    // here; both loops below are also natural no-ops on one.
    // TTE endpoints — one `factor * raw` add each, in ascending CMT order.
    for (cmt, endpoint) in &model.endpoints {
        if let EndpointLikelihood::Tte { hazard, recurrence, .. } = endpoint {
            // A pooled buffer (#2549): this runs on every inner objective evaluation.
            let mut records_for_cmt: Vec<crate::types::ObsRecord> = crate::scratch::take();
            records_for_cmt.extend(
                subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned(),
            );
            if records_for_cmt.is_empty() {
                crate::scratch::give(records_for_cmt);
                continue; // subject has no records for this TTE CMT
            }
            let raw = tte_endpoint_nll_visit(
                model,
                subject,
                hazard,
                *recurrence,
                &records_for_cmt,
                theta,
                eta,
                joint_share,
                &mut |_| {},
            );
            *acc += factor * raw;
            crate::scratch::give(records_for_cmt);
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

/// Each non-Gaussian data term of one subject at `eta`, named (TTE per endpoint in
/// ascending CMT, then categorical and CTMM) — the terms
/// [`accumulate_non_gaussian_nll`] sums, kept apart so `W_NONFINITE_OBJECTIVE` can
/// name the failing one (#2267). Raw NLLs. `OdeAccumulated` hazards are re-solved
/// (agree with the #570 share to solver tolerance). Empty with no non-Gaussian record.
#[cfg(feature = "survival")]
pub(crate) fn non_gaussian_terms(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64],
) -> Vec<(String, f64)> {
    let mut terms = Vec::new();
    if subject.obs_records.is_empty() {
        return terms;
    }
    let mut cmts: Vec<&usize> = model.endpoints.keys().collect();
    cmts.sort();
    for cmt in cmts {
        if let crate::types::EndpointLikelihood::Tte { hazard, recurrence, .. } =
            &model.endpoints[cmt]
        {
            let records: Vec<ObsRecord> = subject
                .obs_records
                .iter()
                .filter(|r| matches!(r, ObsRecord::Event { cmt: c, .. } if c == cmt))
                .cloned()
                .collect();
            if records.is_empty() {
                continue;
            }
            let nll = tte_endpoint_nll(model, subject, hazard, *recurrence, &records, theta, eta);
            terms.push((format!("time-to-event on CMT {cmt}"), nll));
        }
    }
    if model.has_discrete() {
        let nll = crate::categorical::discrete_subject_nll(model, subject, theta, eta);
        terms.push(("categorical".to_string(), nll));
    }
    #[cfg(feature = "markov")]
    if model.has_ctmm() {
        let nll = crate::markov::endpoint::ctmm_subject_nll(model, subject, theta, eta);
        terms.push(("CTMM".to_string(), nll));
    }
    terms
}

/// The finite value a numerically ill-defined per-subject objective is clamped to
/// (non-finite NLL, non-PD `R`, diverged solve, unscorable term): large and finite so
/// the optimizer is repelled, not aborted. Single home of the value (#2269): guards
/// here and in survival / categorical / markov return it, and `gauss_newton`, `agq` and
/// `W_NONFINITE_OBJECTIVE` compare against it. The IOV + FREM refusal deliberately
/// uses [`IOV_FREM_REFUSAL_NLL`].
pub(crate) const SUBJECT_SENTINEL_NLL: f64 = 1e20;

/// Whether an NLL is real rather than NaN/∞ or on the [`SUBJECT_SENTINEL_NLL`] rail.
/// `<` rather than `!=` so a sum holding a sentinel also reads unusable (#2269).
pub(crate) fn is_usable_subject_nll(nll: f64) -> bool {
    nll.is_finite() && nll < SUBJECT_SENTINEL_NLL
}

/// What the IOV marginal returns for a subject with FREM covariate rows, which it
/// refuses: the augmented `b̂` and block-diagonal `Σ_b` are not set up for FREM
/// R-overrides. Named so `W_NONFINITE_OBJECTIVE` reports the refusal as such (#2267).
pub(crate) const IOV_FREM_REFUSAL_NLL: f64 = 1e18;

/// Whether the IOV marginal refuses `subject` (see [`IOV_FREM_REFUSAL_NLL`]). Only
/// consulted for a subject scored with κ.
pub(crate) fn iov_refuses_frem(model: &CompiledModel, subject: &Subject) -> bool {
    model.frem_config.is_some() && subject.fremtype.iter().any(|&ft| ft > 0)
}

/// Hot-path variant taking a pre-built [`pk::event_driven::EventSchedule`] (built once
/// per `find_ebe` call). `residual_correlations` carries the **live** `block_sigma`
/// off-diagonals as `sigma_values` carries live σ (#847): FOCE/FOCEI pass the current
/// ρ; estimators holding ρ fixed pass `&model.residual_correlations`.
pub fn individual_nll_into_with_schedule(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64], residual_correlations: &[ResidualCorrelation],
    scratch: &mut pk::EventPkParams, schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    individual_nll_into_with_schedule_cov(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        ResidualCov::block_sigma(residual_correlations),
        scratch,
        schedule,
    )
}

/// [`individual_nll_into_with_schedule`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
pub(crate) fn individual_nll_into_with_schedule_cov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64], residual_cov: ResidualCov<'_>, scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    let err_keys = model.error_spec.obs_keys(subject);
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let mut pred_recycle = Vec::new();
    let mut eta_work = DVector::zeros(eta.len());
    let mut prior_work = DVector::zeros(eta.len());
    individual_nll_into_prepared_with_schedule(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        residual_cov,
        scratch,
        schedule,
        err_keys.as_ref(),
        ruv_mult.as_deref(),
        &mut pred_recycle,
        &mut eta_work,
        &mut prior_work,
    )
}

/// Caller-owned buffers for repeated [`individual_nll`] evaluations of one subject
/// (SAEM MH loop): the prediction vector and the two `DVector`s of the η-prior
/// quadratic form, which [`individual_nll_into_with_schedule`] allocates every call.
/// Every field is overwritten before read, so reuse is bit-identical. `pk` is public
/// so callers needing a bare [`pk::EventPkParams`] borrow it.
#[derive(Default)]
pub(crate) struct IndividualNllScratch {
    pub pk: pk::EventPkParams,
    preds: Vec<f64>,
    eta_work: DVector<f64>,
    prior_work: DVector<f64>,
}

impl IndividualNllScratch {
    /// Size the η-prior work vectors. A no-op once they match, so this is free
    /// on every call after the first (`n_eta` is fixed for a whole fit).
    fn ensure_eta(&mut self, n_eta: usize) {
        if self.eta_work.len() != n_eta {
            self.eta_work = DVector::zeros(n_eta);
            self.prior_work = DVector::zeros(n_eta);
        }
    }
}

/// The η-independent inputs of one subject's individual NLL: per-observation residual
/// dispatch keys and the `#484` residual-magnitude multipliers. Built once per subject
/// per MH sweep (θ, σ fixed) instead of once per proposal.
#[derive(Default)]
pub(crate) struct IndividualNllPrep {
    err_keys: Vec<usize>,
    ruv_mult: Option<Vec<Vec<f64>>>,
    /// Test-only: the `MIXNUM` value in effect when [`Self::refresh`] ran.
    ///
    /// `ruv_obs_mult` can read the mixture class (`Expression::MixNum`), so the caller must
    /// refresh **under the same `MixtureClassGuard`** as the proposals. PR #1452 broke that
    /// ordering once; checking it where the prep is consumed makes every mixture test a
    /// detector.
    #[cfg(test)]
    mix_class_at_refresh: usize,
}

impl IndividualNllPrep {
    /// Point an existing (possibly empty) prep at a subject, reusing the key
    /// buffer's allocation. There is deliberately no `new`: every caller holds a
    /// long-lived scratch and refreshes it per subject.
    pub(crate) fn refresh(&mut self, model: &CompiledModel, subject: &Subject, theta: &[f64]) {
        self.err_keys.clear();
        self.err_keys.extend_from_slice(model.error_spec.obs_keys(subject).as_ref());
        self.ruv_mult = model.ruv_obs_mult(subject, theta);
        #[cfg(test)]
        {
            self.mix_class_at_refresh = crate::parser::model_parser::current_mixture_class();
        }
    }
}

/// Scratch-reusing, bit-identical form of [`individual_nll_into`]: same arithmetic in
/// the same order, buffers from `prep`/`scratch`. Callers hold ρ at its declaration.
pub(crate) fn individual_nll_prepared(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    prep: &IndividualNllPrep,
    // The subject's cached event-driven schedule, or `None` where reuse is
    // unsound (see `estimation::inner_optimizer::cacheable_schedule`) and the
    // predictor must rebuild it per call, as it always did.
    schedule: Option<&pk::event_driven::EventSchedule>,
    scratch: &mut IndividualNllScratch,
) -> f64 {
    #[cfg(test)]
    assert_eq!(
        prep.mix_class_at_refresh,
        crate::parser::model_parser::current_mixture_class(),
        "IndividualNllPrep was refreshed under mixture class {} but is being \
         consumed under class {}. Both `obs_keys` and `ruv_obs_mult` may read \
         MIXNUM, so the caller must refresh inside the drawn class's \
         MixtureClassGuard (#1452 review round 2).",
        prep.mix_class_at_refresh,
        crate::parser::model_parser::current_mixture_class(),
    );
    scratch.ensure_eta(eta.len());
    let IndividualNllScratch { pk, preds, eta_work, prior_work } = scratch;
    individual_nll_into_prepared_with_schedule(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        model.residual_cov_theta(theta),
        pk,
        schedule,
        &prep.err_keys,
        prep.ruv_mult.as_deref(),
        preds,
        eta_work,
        prior_work,
    )
}

/// Inner-loop form of [`individual_nll_into_with_schedule`] with subject-static
/// residual dispatch and magnitude inputs prepared by the caller, plus a
/// caller-owned prediction vector reused across objective evaluations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_into_prepared_with_schedule(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64], residual_cov: ResidualCov<'_>, scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>, err_keys: &[usize],
    ruv_mult: Option<&[Vec<f64>]>, pred_recycle: &mut Vec<f64>, eta_work: &mut DVector<f64>,
    prior_work: &mut DVector<f64>,
) -> f64 {
    individual_nll_into_prepared_with_schedule_frozen(
        model,
        subject,
        theta,
        eta,
        omega,
        sigma_values,
        residual_cov,
        scratch,
        schedule,
        err_keys,
        ruv_mult,
        pred_recycle,
        eta_work,
        prior_work,
        None,
    )
}

/// Population predictions `f(η = 0)` at which a non-interaction FOCE subject's residual
/// variance is held, or `None` for an additive (prediction-free) variance. The **one**
/// definition of `R⁰`'s evaluation point, shared by [`foce_subject_nll`] and the FOCE
/// EBE search, so both linearise around the same mode — NONMEM `METHOD=1` without
/// `INTER` (#319, #1722).
pub(crate) fn population_variance_preds(
    model: &CompiledModel, subject: &Subject, theta: &[f64], n_eta: usize,
) -> Option<Vec<f64>> {
    if !model.error_spec.has_f_dependent_variance() {
        return None;
    }
    let zeros = vec![0.0_f64; n_eta];
    Some(model_predictions(model, subject, theta, &zeros))
}

/// IOV twin of [`population_variance_preds`]: the prediction with every random
/// effect at zero (η = 0 and κ = 0 on each of the subject's `k_occasions`), the
/// point the augmented Sheiner–Beal marginal ([`foce_subject_nll_iov`]) scores
/// `R⁰` at.
pub(crate) fn population_variance_preds_iov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], n_eta: usize, n_kappa: usize,
    k_occasions: usize,
) -> Option<Vec<f64>> {
    if !model.error_spec.has_f_dependent_variance() {
        return None;
    }
    let zeros_eta = vec![0.0_f64; n_eta];
    let zero_kappas: Vec<Vec<f64>> = vec![vec![0.0_f64; n_kappa]; k_occasions];
    // The subject's layout and schedule are the fit's, built once (the schedule is the one the
    // EBE search reads; `None` where the gate declines, which `predict_iov` then builds itself).
    let layout = crate::stats::stacked_layout::StackedLayout::new_cached(
        zeros_eta.len(),
        model.n_kappa,
        subject,
    );
    let schedule =
        crate::estimation::inner_optimizer::cacheable_schedule_iov_cached(model, subject);
    Some(pk::predict_iov_in(
        model,
        subject,
        &layout,
        theta,
        &zeros_eta,
        &zero_kappas,
        &mut pk::EventPkParams::default(),
        schedule.as_ref().as_ref(),
    ))
}

/// Whether value-only predictions may come from the compiled plan
/// ([`crate::sens::ode_provider::adjoint::plan_values`]): a closed-form model or an
/// `[odes]` model on the exact linear path, without lagtime (onset arithmetic measured
/// up to 5 ulp off the native walk), where the plan is bit-equal and cheaper.
pub(crate) fn plan_value_route(model: &CompiledModel) -> bool {
    #[cfg(test)]
    if PLAN_VALUES_OFF.with(|c| c.get()) {
        return false;
    }
    plan_value_bit_equal(model)
}

/// The one predicate for "the plan's value is bit-equal to the native objective": not
/// lagged (5 ulp off, measured) and not a nonlinear / RK45 `[odes]` model. Shared by
/// [`plan_value_route`] and `find_ebe_solve`'s memo and fused line search.
pub(crate) fn plan_value_bit_equal(model: &CompiledModel) -> bool {
    !model.has_lagtime()
        && model
            .ode_spec
            .as_ref()
            .is_none_or(|o| o.effective_solver_opts().method == crate::ode::OdeMethod::Linear)
}

#[cfg(test)]
thread_local! {
    /// Test switch: the objective's value pass stays on the native predictor (a test of the
    /// native walk's own bookkeeping, e.g. its schedule builds).
    pub(crate) static PLAN_VALUES_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `preds` from the plan when [`plan_value_route`] admits the model and the plan serves the
/// point; `false` when the native predictor must answer.
fn plan_value_preds(
    model: &CompiledModel, subject: &Subject,
    layout: Option<&crate::stats::stacked_layout::StackedLayout>, theta: &[f64], eta: &[f64],
    out: &mut Vec<f64>,
) -> bool {
    plan_value_route(model)
        && crate::sens::ode_provider::adjoint::plan_values(model, subject, layout, theta, eta, out)
}

/// [`individual_nll_into_prepared_with_schedule`] with an optional frozen residual
/// variance (#1722). `Some(f₀)` scores each Gaussian row's variance at `f₀[j]` (from
/// [`population_variance_preds`]) — the non-interaction FOCE EBE objective; residual,
/// prior and non-Gaussian terms keep `f(η)`, and the `iiv_on_ruv` scale is not applied
/// (`R⁰` has every random effect zero). `None` is bit-identical to the function above.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_into_prepared_with_schedule_frozen(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], omega: &OmegaMatrix,
    sigma_values: &[f64], residual_cov: ResidualCov<'_>, scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>, err_keys: &[usize],
    ruv_mult: Option<&[Vec<f64>]>, pred_recycle: &mut Vec<f64>, eta_work: &mut DVector<f64>,
    prior_work: &mut DVector<f64>, frozen_var_preds: Option<&[f64]>,
) -> f64 {
    // Ω⁻¹ and log|Ω| are pre-computed in `OmegaMatrix::from_matrix_*`.
    // Hot-path users (FOCE inner BFGS, SAEM MH) call this 100s–1000s of
    // times per subject per outer iter — recomputing Cholesky+inverse
    // here used to dominate small-omega problems.
    if !omega.log_det.is_finite() {
        return SUBJECT_SENTINEL_NLL;
    }
    let omega_inv = &omega.inv;
    let log_det_omega = omega.log_det;
    // Eta prior: eta' * Omega_inv * eta
    eta_work.as_mut_slice().copy_from_slice(eta);
    prior_work.gemv(1.0, omega_inv, eta_work, 0.0);
    let mut eta_prior = eta_work.dot(prior_work);
    // #1716: a non-Gaussian density group's correction to the Gaussian prior; `+∞`
    // outside a Box-Cox support.
    if !model.eta_densities.is_empty() {
        eta_prior +=
            crate::stats::eta_density::correction(&model.eta_densities, eta, omega_inv, theta);
        if !eta_prior.is_finite() {
            return SUBJECT_SENTINEL_NLL;
        }
    }

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
        Some(s) => {
            pred_recycle.clear();
            pred_recycle.extend_from_slice(&s.preds);
            pred_recycle
        }
        None => {
            if !plan_value_preds(model, subject, None, theta, eta, pred_recycle) {
                let hint = std::mem::take(pred_recycle);
                *pred_recycle = pk::compute_predictions_with_tv_recycle_with_schedule(
                    model, subject, theta, eta, scratch, schedule, hint,
                );
            }
            pred_recycle
        }
    };
    #[cfg(not(feature = "survival"))]
    let preds = {
        if !plan_value_preds(model, subject, None, theta, eta, pred_recycle) {
            let hint = std::mem::take(pred_recycle);
            *pred_recycle = pk::compute_predictions_with_tv_recycle_with_schedule(
                model, subject, theta, eta, scratch, schedule, hint,
            );
        }
        pred_recycle
    };
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
    let ruv_scale = if frozen_var_preds.is_some() { 1.0 } else { model.residual_var_scale(eta) };
    let mut data_ll = 0.0;
    let has_censored_m3 = model.bloq_in_force().has_censored_row(&subject.cens);
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if residual_cov.is_dense() && !has_censored_m3 && !has_frem_rows {
        match dense_residual_data_term(
            model,
            subject,
            &preds,
            frozen_var_preds.unwrap_or(&preds[..]),
            sigma_values,
            residual_cov,
            ruv_scale,
            &p_obs,
            ruv_mult,
        ) {
            Some(term) => data_ll += term,
            None => return SUBJECT_SENTINEL_NLL,
        }
    } else {
        for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            // #905: never score a non-Gaussian-endpoint row as Gaussian. A no-op on the
            // endpoint-routed load; on a model-blind load these rows carry a NaN `f_pred` that
            // would poison the NLL.
            #[cfg(feature = "survival")]
            if !model.endpoints.is_empty()
                && subject.obs_cmts.get(j).is_some_and(|c| model.endpoints.contains_key(c))
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
            let f_var = frozen_var_preds.map_or(f_pred, |f0| f0[j]);
            // The live `block_sigma` correlations (`residual_cov.pairs`, #847), not the
            // model's declaration: this row-wise branch serves a correlated model whenever
            // it has censored or FREM rows, and its ρ must be the one being estimated.
            // Identical to `residual_variance_at_scaled` when the two coincide.
            let v_resid = match ruv_mult.and_then(|m| m.get(j)).map(Vec::as_slice) {
                Some(m) => model.error_spec.variance_at_scaled(
                    err_keys[j],
                    f_var,
                    sigma_values,
                    residual_cov.pairs,
                    m,
                ),
                None => model.error_spec.variance_at_with_correlations(
                    err_keys[j],
                    f_var,
                    sigma_values,
                    residual_cov.pairs,
                ),
            } * ruv_scale;
            let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if model.bloq_in_force().is_censored_row(cens) {
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
    // A blown-up solve (NaN/inf) maps to the finite sentinel so it sorts as worst (#97).
    if nll.is_finite() {
        nll
    } else {
        SUBJECT_SENTINEL_NLL
    }
}

/// Dense correlated-residual (block_sigma) Gaussian data term for one subject:
/// `rᵀ R⁻¹ r + log|R|` (un-halved) with `R` from
/// [`compute_r_matrix_with_correlations`](crate::stats::residual_error::compute_r_matrix_with_correlations)
/// scaled by `ruv_scale` plus EKF `p_obs` on the diagonal. `None` when `R` is not PD.
/// Shared by FOCE ([`individual_nll_into_with_schedule`]) and the SAEM M-step
/// ([`obs_nll_subject_into`]) so both score one likelihood (#557).
fn dense_residual_data_term(
    model: &CompiledModel,
    subject: &Subject,
    preds: &[f64],
    // The predictions `R` is built at: `preds` for the conditional objective, or the
    // frozen `f(η = 0)` of a non-interaction FOCE EBE search (#1722).
    var_preds: &[f64],
    sigma_values: &[f64],
    residual_cov: ResidualCov<'_>,
    ruv_scale: f64,
    p_obs: &[f64],
    ruv_mult: Option<&[Vec<f64>]>,
) -> Option<f64> {
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    let mut r = crate::stats::residual_error::r_matrix_maybe_scaled(
        &model.error_spec,
        var_preds,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_cov,
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
        subject.observations.iter().zip(preds.iter()).map(|(&y, &f)| y - f),
    );
    let solved = chol.solve(&residuals);
    Some(residuals.dot(&solved) + chol_log_det(&chol.l()))
}

/// Observation-only NLL `−log p(y_i | η, θ, σ)` for one subject (no prior / |Ω|) — the
/// SAEM M-step and IS-LL numerator term. M3-censored rows contribute normal tails.
/// `residual_correlations` carries the **live** ρ (#847): a chain like
/// `[focei, imp]` hands IMP the fitted ρ, which the declaration would not reflect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_into(
    model: &CompiledModel, subject: &Subject, theta: &[f64], sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, eta: &[f64], pk_scratch: &mut pk::EventPkParams,
) -> f64 {
    obs_nll_subject_into_with_schedule(
        model,
        subject,
        theta,
        sigma_values,
        residual_cov,
        eta,
        pk_scratch,
        None,
    )
}

/// [`obs_nll_subject_into`] with the subject's cached event-driven schedule (avoids
/// rebuilding `EventSchedule::for_subject` per SAEM M-step evaluation). `None` is the
/// uncached behaviour (see `inner_optimizer::cacheable_schedule`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_into_with_schedule(
    model: &CompiledModel, subject: &Subject, theta: &[f64], sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, eta: &[f64], pk_scratch: &mut pk::EventPkParams,
    schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    let preds = pk::compute_predictions_with_tv_into_with_schedule(
        model, subject, theta, eta, pk_scratch, schedule,
    );
    obs_nll_subject_from_preds(model, subject, &preds, theta, sigma_values, residual_cov, eta)
}

/// [`obs_nll_subject_into`] from *precomputed* `preds` (σ-independent), so the SAEM
/// σ-gradient FD loop reuses one solve (#557). `theta` feeds only the TTE hazard.
#[cfg_attr(not(feature = "survival"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn obs_nll_subject_from_preds(
    model: &CompiledModel, subject: &Subject, preds: &[f64], theta: &[f64], sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, eta: &[f64],
) -> f64 {
    obs_nll_subject_from_preds_with_semantics(
        model,
        subject,
        preds,
        theta,
        sigma_values,
        residual_cov,
        eta,
        PredNllSemantics::Marginal,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
// The conditional semantics are constructed only by `conditional_obs_nll_from_preds`, a
// test oracle since the AGQ path it served moved to the adjoint (#486); the shared body
// below still branches on them.
#[cfg_attr(not(test), allow(dead_code))]
enum PredNllSemantics {
    Marginal,
    Conditional,
    ConditionalIov,
}

#[cfg_attr(not(feature = "survival"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
fn obs_nll_subject_from_preds_with_semantics(
    model: &CompiledModel, subject: &Subject, preds: &[f64], theta: &[f64], sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, eta: &[f64], semantics: PredNllSemantics,
) -> f64 {
    let m3 = matches!(model.bloq_in_force(), BloqMethod::M3);
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
    if residual_cov.is_dense() && !m3 && !has_frem_rows {
        // block_sigma + SDE is rejected up front (E_BLOCK_SIGMA_SDE_UNSUPPORTED)
        // for the SAEM M-step, so no EKF process noise enters here.
        match dense_residual_data_term(
            model,
            subject,
            preds,
            preds,
            sigma_values,
            residual_cov,
            ruv_scale,
            &[],
            ruv_mult.as_deref(),
        ) {
            Some(term) => nll += 0.5 * term,
            None => return SUBJECT_SENTINEL_NLL,
        }
    } else {
        for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            // #905: never score a non-Gaussian-endpoint row as Gaussian (twin of the skip in
            // `individual_nll_into_with_schedule`).
            #[cfg(feature = "survival")]
            if !model.endpoints.is_empty()
                && subject.obs_cmts.get(j).is_some_and(|c| model.endpoints.contains_key(c))
            {
                continue;
            }
            let frem_var = frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x);
            if semantics == PredNllSemantics::Conditional && frem_var.is_some() {
                if let Some(ref fc) = model.frem_config {
                    let fremtype_val = subject.fremtype.get(j).copied().unwrap_or(0);
                    if let Some(&(theta_idx, eta_idx)) = fc.fremtype_to_indices.get(&fremtype_val) {
                        let frem_pred = theta[theta_idx] + eta[eta_idx];
                        let frem_sigma = sigma_values[fc.covariate_sigma_index];
                        let frem_v = (frem_sigma * frem_sigma).max(1e-12);
                        let resid = y - frem_pred;
                        nll += 0.5 * (resid * resid / frem_v + frem_v.ln());
                        continue;
                    }
                }
            }
            // Never clamp the prediction: negative values are legitimate (additive models, LTBS,
            // FREM, #406), and clamping disagreed with `individual_nll_into` (#2440). Variance
            // positivity is the variance floor's job; non-finite maps to the sentinel below.
            let v = match frem_var {
                Some(vv) if semantics == PredNllSemantics::Marginal => vv.max(1e-12),
                Some(vv) => vv,
                None if semantics == PredNllSemantics::Marginal => {
                    // Guard only a variance that cannot be scored (#2441): an absolute
                    // floor here made the objective depend on the units of DV.
                    crate::stats::residual_error::guard_variance(
                        model.residual_variance_at_scaled(
                            err_keys[j],
                            f,
                            sigma_values,
                            ruv_mult.as_ref().map(|m| m[j].as_slice()),
                        ) * ruv_scale,
                    )
                }
                None => {
                    model.residual_variance_at_scaled(
                        err_keys[j],
                        f,
                        sigma_values,
                        ruv_mult.as_ref().map(|m| m[j].as_slice()),
                    ) * ruv_scale
                }
            };
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if model.bloq_in_force().is_censored_row(cens) {
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

    // #2440: a NaN prediction yields a NaN NLL; map it to the 1e20 sentinel like
    // `individual_nll_into`.
    if nll.is_finite() {
        nll
    } else {
        SUBJECT_SENTINEL_NLL
    }
}

/// Per-observation EKF process-noise variance (`p_obs`) for an SDE model; empty when
/// `!model.is_sde()`. Also read by the level binder's rank check (#2421).
pub(crate) fn ekf_p_obs(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], sigma_values: &[f64],
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

    // Single-pass EKF with a constant PK vector, so `TIME` resolves at t=0, as in the
    // single-pass ODE prediction path (#610). ODE-RHS `TIME` still uses the solver clock.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    // An SDE model is always `ErrorSpec::Single` (Form C, needed for per-CMT errors, is
    // rejected on SDE), so `model.error_model` is exact here.
    debug_assert!(
        matches!(model.error_spec, ErrorSpec::Single(_)),
        "EKF path reached with a non-Single error spec (per-CMT + SDE should be unreachable)"
    );
    let error_model = model.error_model;

    // `diffusion_var` is passed beside the spec (it moves with θ; the spec cannot). One filter
    // solve: its Kalman `R` reads the filter's own mean, so no second ODE pass is made for it.
    let (_, p_obs) = crate::ode::ode_predictions_ekf_with_diffusion(
        ode,
        &pk.values,
        subject,
        &diffusion_var,
        |f_pred| crate::stats::residual_error::residual_variance(error_model, f_pred, sigma_values),
    );
    p_obs
}

/// Residual-variance factor from an IIV-on-RUV eta (`Y = IPRED + EPS·EXP(ETA)`):
/// `exp(2·eta[k])` for `Some(k)` in range, else `1.0`. Mirrors
/// [`CompiledModel::residual_var_scale`] for callers without the model.
#[inline]
pub(crate) fn ruv_scale_from(eta: &[f64], residual_error_eta: Option<usize>) -> f64 {
    match residual_error_eta {
        Some(k) => eta.get(k).map(|&e| (2.0 * e).exp()).unwrap_or(1.0),
        None => 1.0,
    }
}

/// FOCE per-subject negative log-likelihood. Non-interaction (standard FOCE):
/// `NLL_i = 0.5 * [(y - f0)' R̃⁻¹ (y - f0) + log|R̃|]`, `f0 = f(η̂) - H η̂`,
/// `R̃ = H Ω H' + R(f0)`.
///
/// `residual_correlations` carries the **live** `block_sigma` ρ (#847). A
/// `correlation … by COL` line (#1081) is **not** applied here; `fit()` uses the
/// in-crate form that carries it.
#[allow(clippy::too_many_arguments)]
pub fn foce_subject_nll(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation], interaction: bool,
) -> f64 {
    foce_subject_nll_cov(
        model,
        subject,
        theta,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        ResidualCov::block_sigma(residual_correlations),
        interaction,
    )
}

/// [`foce_subject_nll`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_nll_cov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, interaction: bool,
) -> f64 {
    foce_subject_nll_cov_at(
        model,
        subject,
        theta,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        residual_cov,
        interaction,
        None,
    )
}

/// [`foce_subject_nll_cov`] given the predictions at `eta_hat` when the caller holds them
/// (the inner objective's last evaluation at the mode), so the point is not predicted again.
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_nll_cov_at(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, interaction: bool, ipreds: Option<&[f64]>,
) -> f64 {
    foce_subject_nll_cov_split(
        model,
        subject,
        theta,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        residual_cov,
        interaction,
        ipreds,
        None,
    )
}

/// Where a marginal records its parts for `W_NONFINITE_OBJECTIVE` (#2267): `None` on
/// every scoring path, so the value a caller gets back is computed the same way either
/// way. Each part is `(name, nll)` on the raw NLL scale — half its share of the `2·nll`
/// contribution.
type PartSink<'a> = Option<&'a mut Vec<(String, f64)>>;

/// Record one part when a sink is attached.
fn note(parts: &mut PartSink<'_>, name: &str, nll: f64) {
    if let Some(p) = parts.as_deref_mut() {
        p.push((name.to_string(), nll));
    }
}

/// The parts of `subject`'s [`foce_subject_nll_cov`] contribution, recorded by the
/// branch that computes it (#2267). Healthy, `2·Σ nll` is the contribution to
/// rounding; when the marginal hits its sentinel, the failing part is recorded as
/// `NaN` under a name saying why and unreached parts are absent. Names:
///
/// * FOCE — `linearized residual term`, `linearized covariance log-determinant`, or
///   `linearized covariance R̃ (not positive definite)`;
/// * FOCEI — `Gaussian residual term` (or its non-finite / non-PD variants),
///   `random-effect prior`, `curvature log-determinant` or
///   `curvature (H̃ not positive definite)`;
/// * both — `censored (M3) term` when a row is censored;
/// * with TTE/categorical/CTMM records — every [`non_gaussian_terms`] name plus the
///   joint-form Gaussian, prior and curvature parts (also under FOCE).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_contribution_parts(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, interaction: bool,
) -> Vec<(String, f64)> {
    let mut parts = Vec::new();
    foce_subject_nll_cov_split(
        model,
        subject,
        theta,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        residual_cov,
        interaction,
        None,
        Some(&mut parts),
    );
    parts
}

/// A subject's non-Gaussian data terms at `eta_hat` (TTE, discrete, CTMM): summed raw
/// NLL and summed η-Hessian, each part recorded in `parts`. Folded into the FOCE/FOCEI
/// data term (`2·q`) and curvature (`H̃ + ∇²q`) by the non-IOV and IOV marginals.
#[cfg(feature = "survival")]
fn non_gaussian_mode_terms(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    joint_share: Option<&JointPkTteSolve>, parts: &mut PartSink<'_>,
) -> (f64, DMatrix<f64>) {
    use crate::types::EndpointLikelihood;
    // TTE data NLL over `model.endpoints` (a `BTreeMap`: ascending CMT, bit-identical
    // on every compile, #2531).
    let n_eta = eta_hat.len();
    let mut tte_nll_at_mode = 0.0_f64;
    let mut tte_h = DMatrix::<f64>::zeros(n_eta, n_eta);
    // Per-CMT terms for the part sink, in CMT order.
    let mut tte_parts: Vec<(usize, f64)> = Vec::new();

    for (cmt, endpoint) in &model.endpoints {
        if let EndpointLikelihood::Tte { hazard, recurrence, .. } = endpoint {
            // A pooled buffer (#2549): this runs on every inner objective evaluation.
            let mut records_for_cmt: Vec<crate::types::ObsRecord> = crate::scratch::take();
            records_for_cmt.extend(
                subject
                    .obs_records
                    .iter()
                    .filter(
                        |r| matches!(r, crate::types::ObsRecord::Event { cmt: c, .. } if c == cmt),
                    )
                    .cloned(),
            );
            if records_for_cmt.is_empty() {
                crate::scratch::give(records_for_cmt);
                continue; // subject has no records for this TTE CMT
            }
            // #570: at the mode, read H/h off the shared solve when available.
            // The shared solve only where the walk does not serve the subject: the walk's
            // own value is the one its exact derivatives differentiate. The router makes
            // that choice for every caller (#901).
            let endpoint_nll = tte_endpoint_nll_visit(
                model,
                subject,
                hazard,
                *recurrence,
                &records_for_cmt,
                theta,
                eta_hat.as_slice(),
                joint_share,
                &mut |_| {},
            );
            tte_nll_at_mode += endpoint_nll;
            if parts.is_some() {
                tte_parts.push((*cmt, endpoint_nll));
            }
            crate::scratch::give(records_for_cmt);
        }
    }

    tte_parts.sort_by_key(|&(cmt, _)| cmt);
    for (cmt, nll) in tte_parts {
        note(parts, &format!("time-to-event on CMT {cmt}"), nll);
    }

    // Discrete endpoints (#760: binary; ordinal/Poisson/… later): fold their NLL at the
    // mode and FD Hessian w.r.t. η into the same accumulators, so log|H̃| includes the
    // discrete curvature. A pure-discrete subject has an empty Gaussian h_matrix; hrh then
    // comes entirely from here. `has_discrete()` gates a new family in without a change here.
    if model.has_discrete() {
        let mode =
            crate::categorical::discrete_subject_nll(model, subject, theta, eta_hat.as_slice());
        note(parts, "categorical", mode);
        tte_nll_at_mode += mode;
    }

    // CTMM (#759): same treatment — fold its NLL at the mode and FD Hessian
    // w.r.t. η (the generator's η-curvature) into the accumulators.
    #[cfg(feature = "markov")]
    if model.has_ctmm() {
        let mode =
            crate::markov::endpoint::ctmm_subject_nll(model, subject, theta, eta_hat.as_slice());
        note(parts, "CTMM", mode);
        tte_nll_at_mode += mode;
    }
    // Exact curvature from the plan (#2786). A decline (`adjoint::HessianDecline`, a plan
    // defect) falls back to the central-difference Hessian; repelled only if that is
    // not finite either.
    if n_eta > 0 && tte_nll_at_mode < crate::survival::REPEL_SENTINEL_FLOOR {
        match crate::survival::non_gaussian_eta_jets2(model, subject, theta, eta_hat.as_slice()) {
            Some((_, _, h)) => tte_h += h,
            None => {
                // A stencil point the term repels (or cannot score) has no curvature to
                // difference — at a mode on a feasibility boundary both sides are repelled —
                // so the curvature is unmeasurable and the subject is repelled.
                let poisoned = std::cell::Cell::new(false);
                let nll = |e: &[f64]| {
                    let mut acc = 0.0;
                    accumulate_non_gaussian_nll(
                        model,
                        subject,
                        theta,
                        e,
                        joint_share,
                        1.0,
                        &mut acc,
                    );
                    if !acc.is_finite() || acc >= crate::survival::REPEL_SENTINEL_FLOOR {
                        poisoned.set(true);
                    }
                    acc
                };
                let eps: Vec<f64> = eta_hat.iter().map(|x| 1e-4 * x.abs().max(1.0)).collect();
                let h = crate::survival::data_term_hessian_fd(nll, eta_hat.as_slice(), &eps);
                if !poisoned.get() && h.iter().all(|v| v.is_finite()) {
                    tte_h += h;
                } else {
                    note(parts, "non-Gaussian curvature unmeasurable", f64::NAN);
                    tte_nll_at_mode += SUBJECT_SENTINEL_NLL;
                }
            }
        }
    }
    (tte_nll_at_mode, tte_h)
}

/// [`foce_subject_nll_cov`], recording its parts into `parts` when one is attached.
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_cov_split(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, interaction: bool, held_preds: Option<&[f64]>,
    #[cfg_attr(not(feature = "survival"), allow(unused_mut))] mut parts: PartSink<'_>,
) -> f64 {
    // Individual predictions at eta_hat (per-event PK when subject has TV covariates).
    // #570: at the EBE mode the Gaussian `ipreds` solve and the at-mode TTE-term solve
    // are both at `eta_hat`; when the subject qualifies, share one augmented integration
    // (the FD-Hessian's perturbed-η TTE solves below are at different η and stay separate).
    #[cfg(feature = "survival")]
    let joint_share = try_joint_pktte_shared_solve(model, subject, theta, eta_hat.as_slice());
    let predict = || match held_preds {
        Some(p) => p.to_vec(),
        None => model_predictions(model, subject, theta, eta_hat.as_slice()),
    };
    #[cfg(feature = "survival")]
    let ipreds = match &joint_share {
        Some(s) => s.preds.clone(),
        None => predict(),
    };
    #[cfg(not(feature = "survival"))]
    let ipreds = predict();

    // Per-observation custom residual magnitude (#484). η-independent, so one
    // evaluation here feeds both the data term and the Laplace curvature term.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // A density group's correction at η̂ (#1716), into the prior and `H̃` of every form.
    let density = crate::stats::eta_density::DensityAtMode::at(
        &model.eta_densities,
        eta_hat.as_slice(),
        &omega.inv,
        theta,
    );

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

    // TTE Laplace correction: add the TTE data term's η-Hessian (exact with a dual
    // program, else FD, #2437) to hrh so log|H̃| holds both curvatures. Pure-TTE subjects
    // get hrh entirely from it.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        let (tte_nll_at_mode, tte_h) = non_gaussian_mode_terms(
            model,
            subject,
            theta,
            eta_hat,
            joint_share.as_ref(),
            &mut parts,
        );
        return foce_subject_nll_interaction_with_tte(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            model.bloq_in_force(),
            &p_obs,
            tte_nll_at_mode,
            tte_h,
            // `interaction` alone selects FOCE/FOCEI; M3 no longer promotes (#367/#591). Known
            // gap: this TTE path's censored term rides the conditional variance, not the
            // Sheiner–Beal one the non-TTE FOCE-M3 path uses.
            interaction,
            model.residual_error_eta,
            ruv_mult.as_deref(),
            density.as_ref(),
            parts,
        );
    }
    // FOCE (no interaction) under a density: the Sheiner–Beal marginal integrates η as
    // `N(0, Ω)`, so it has no density form. It takes the conditional form FOCE takes beside
    // a non-Gaussian endpoint instead (the variance at η̂, `½·c̃ᵀc̃` and the censored
    // curvature left out of `H̃`), with no endpoint term. Fit validation refuses the
    // combinations that form does not carry (a dense residual, FREM).
    if density.is_some() && !interaction {
        let n = eta_hat.len();
        let rowwise_rows = model.bloq_in_force().has_censored_row(&subject.cens)
            || subject.fremtype.iter().any(|&ft| ft > 0);
        if residual_cov.is_dense() && !rowwise_rows {
            // A correlated residual: the dense conditional form, `½·B` left out.
            return foce_subject_nll_interaction_dense_split(
                subject,
                &ipreds,
                eta_hat,
                h_matrix,
                omega,
                sigma_values,
                &model.error_spec,
                residual_cov,
                &p_obs,
                ruv_mult.as_deref(),
                model.residual_error_eta,
                false,
                density.as_ref(),
                parts,
            );
        }
        return foce_subject_nll_interaction_with_tte(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            model.bloq_in_force(),
            &p_obs,
            0.0,
            DMatrix::zeros(n, n),
            false,
            model.residual_error_eta,
            ruv_mult.as_deref(),
            density.as_ref(),
            parts,
        );
    }

    // M3 no longer promotes to interaction: FOCE-M3 and FOCEI-M3 are different optima
    // (NONMEM `METHOD=1 LAPLACE` with/without INTER). Interaction routes on whether R
    // is diagonal: `block_sigma` uses `foce_subject_nll_interaction_dense` (#616);
    // the diagonal accumulator would drop the off-diagonals. Censored or FREM rows are
    // scored row-wise by both objectives, so `F − ½log|H̃|` stays the inner objective.
    let rowwise = model.bloq_in_force().has_censored_row(&subject.cens)
        || subject.fremtype.iter().any(|&ft| ft > 0);
    if interaction {
        if !residual_cov.is_dense() || rowwise {
            foce_subject_nll_interaction_split(
                subject,
                &ipreds,
                eta_hat,
                h_matrix,
                omega,
                sigma_values,
                &model.error_spec,
                model.bloq_in_force(),
                &p_obs,
                frem_r_override.as_deref(),
                model.residual_error_eta,
                ruv_mult.as_deref(),
                residual_cov.pairs,
                density.as_ref(),
                parts,
            )
        } else {
            foce_subject_nll_interaction_dense_split(
                subject,
                &ipreds,
                eta_hat,
                h_matrix,
                omega,
                sigma_values,
                &model.error_spec,
                residual_cov,
                &p_obs,
                ruv_mult.as_deref(),
                model.residual_error_eta,
                true,
                density.as_ref(),
                parts,
            )
        }
    } else {
        // FOCE: R at the population prediction f(η=0) (NONMEM no-interaction), not the
        // linearized f0, which can extrapolate to ≤ 0 and collapse R. Skipped for additive
        // error, where both give the same R.
        let pop_preds = population_variance_preds(model, subject, theta, eta_hat.len());
        foce_subject_nll_standard_split(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            &model.error_spec,
            residual_cov,
            model.bloq_in_force(),
            &p_obs,
            frem_r_override.as_deref(),
            pop_preds.as_deref(),
            ruv_mult.as_deref(),
            parts,
        )
    }
}

/// Overwrite the residual-variance diagonal at FREM pseudo-observation rows with the
/// overrides from [`build_frem_r_override`]; `None` entries and indices past the end
/// are left untouched.
pub fn apply_frem_r_overrides(r_diag: &mut [f64], overrides: &[Option<f64>]) {
    for (j, ov) in overrides.iter().enumerate() {
        if let (Some(v), true) = (ov, j < r_diag.len()) {
            r_diag[j] = *v;
        }
    }
}

/// The floor on a FREM covariate row's variance `σ_cov²` in
/// [`build_frem_r_override`]. Below it the objective does not depend on the
/// covariate σ, so a score of that σ must be zero there too
/// (`fixed_eta_gradient::obs_dvar_dlogsigma`, #1421).
pub(crate) const FREM_VARIANCE_FLOOR: f64 = 1e-12;

/// Per-observation R-diagonal overrides for FREM covariate pseudo-observations;
/// `None` when FREM is inactive (no config or empty fremtype).
pub fn build_frem_r_override(
    frem_config: Option<&FremConfig>, fremtype: &[u16], sigma_values: &[f64],
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
                    Some(if s * s > FREM_VARIANCE_FLOOR { s * s } else { FREM_VARIANCE_FLOOR })
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
    foce_subject_nll_standard_cov(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        ResidualCov::block_sigma(residual_correlations),
        bloq_method,
        p_obs,
        frem_r_override,
        r_pred_override,
        ruv_mult,
    )
}

/// [`foce_subject_nll_standard`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
pub(crate) fn foce_subject_nll_standard_cov(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_cov: ResidualCov<'_>,
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
    foce_subject_nll_standard_split(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        residual_cov,
        bloq_method,
        p_obs,
        frem_r_override,
        r_pred_override,
        ruv_mult,
        None,
    )
}

/// [`foce_subject_nll_standard_cov`], recording its parts into `parts` when one is
/// attached (see [`foce_subject_contribution_parts`]).
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_standard_split(
    subject: &Subject, ipreds: &[f64], eta_hat: &DVector<f64>, h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix, sigma_values: &[f64], error_spec: &ErrorSpec,
    residual_cov: ResidualCov<'_>, bloq_method: BloqMethod, p_obs: &[f64],
    frem_r_override: Option<&[Option<f64>]>, r_pred_override: Option<&[f64]>,
    ruv_mult: Option<&[Vec<f64>]>, mut parts: PartSink<'_>,
) -> f64 {
    let n_obs = subject.observations.len();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

    // f0 = ipred - H * eta_hat (linearized population prediction)
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds.iter().enumerate().map(|(j, &ip)| ip - h_eta[j]).collect();

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
        residual_cov,
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

    // M3 under FOCE: censored rows leave the Sheiner–Beal marginal and re-enter as
    // `−logΦ((LLOQ − f0)/√R̃ⱼⱼ)` with the same linearized moments (`f0 = f(η̂) − Hη̂`,
    // `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰ⱼ`, #646). FOCEI instead keeps the conditional `f(η̂)`/`R⁰` and
    // folds censored curvature into `H̃` (`cens_hess`, #486), as NONMEM LAPLACE M3.
    let m3 = bloq_method.has_censored_row(&subject.cens);
    let quant: Vec<usize> = if m3 {
        (0..n_obs)
            .filter(|&j| !bloq_method.is_censored_row(subject.cens.get(j).copied().unwrap_or(0)))
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
    let r_q = if m3 { select_symmetric_submatrix(&r_matrix, &quant) } else { r_matrix };
    let r_tilde = compute_r_tilde_with_r(&h_q, &omega.matrix, &r_q);

    let chol = match r_tilde.clone().cholesky() {
        Some(c) => c,
        None => {
            if m3 {
                note(&mut parts, "censored (M3) term", 0.5 * bloq_term);
            }
            note(&mut parts, "linearized covariance R̃ (not positive definite)", f64::NAN);
            return SUBJECT_SENTINEL_NLL;
        }
    };

    // Residuals: y - f0 over the quantified rows.
    let residuals: DVector<f64> =
        DVector::from_iterator(nq, quant.iter().map(|&j| subject.observations[j] - f0[j]));

    let solved = chol.solve(&residuals);
    let quad_form = residuals.dot(&solved);
    let log_det_r = chol_log_det(&chol.l());

    if nq > 0 {
        note(&mut parts, "linearized residual term", 0.5 * quad_form);
    }
    if m3 {
        note(&mut parts, "censored (M3) term", 0.5 * bloq_term);
    }
    if nq > 0 {
        note(&mut parts, "linearized covariance log-determinant", 0.5 * log_det_r);
    }

    0.5 * (quad_form + log_det_r + bloq_term)
}

/// Number of η columns in the inner-Jacobian `h_matrix`.
#[inline]
fn model_n_eta(h_matrix: &DMatrix<f64>) -> usize {
    h_matrix.ncols()
}

/// FOCEI INTER per-subject −2·log marginal — Almquist 2015 Laplace form (without
/// the `N·log(2π)` constant):
///
/// ```text
///   data_ll(η̂) + η̂'·Ω⁻¹·η̂ + log|Ω| + log|H̃|
/// ```
///
/// with `data_ll(η̂) = Σⱼ [(yⱼ − fⱼ)² / Rⱼ + log Rⱼ]` (R at η̂),
/// `H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹` (eq. 15), `a = ∂f/∂η` (`h_matrix`) and
/// `c̃_{j,k} = (∂Rⱼ/∂fⱼ)·a_{j,k} / Rⱼ`. The `½·c̃'·c̃` INTER correction vanishes for
/// additive error.
///
/// Matches NONMEM `METHOD=1 INTER` and nlmixr2 `focei`: reconstructing NONMEM's
/// per-subject OBJ with this formula reproduced its OFV to 0.013 of 66 539 (jasmine
/// peds vanco). The earlier Sheiner–Beal form diverged at large |η|.
///
/// Under M3, censored rows contribute `−logΦ` at η̂ and enter `H̃` at Gauss–Newton
/// order (`cens_hess`, including `iiv_on_ruv` cross terms), as NONMEM LAPLACE M3.
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
    foce_subject_nll_interaction_corr(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        bloq_method,
        p_obs,
        frem_r_override,
        residual_error_eta,
        ruv_mult,
        &[],
    )
}

/// [`foce_subject_nll_interaction`] scoring each row's variance with the live
/// `block_sigma` correlations: the row-wise FOCEI objective of a correlated model whose
/// subject has censored or FREM rows, which the inner objective scores row-wise with
/// the same correlated variance (`individual_nll_into_prepared_with_schedule_frozen`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_nll_interaction_corr(
    subject: &Subject, ipreds: &[f64], eta_hat: &DVector<f64>, h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix, sigma_values: &[f64], error_spec: &ErrorSpec, bloq_method: BloqMethod,
    p_obs: &[f64], frem_r_override: Option<&[Option<f64>]>, residual_error_eta: Option<usize>,
    ruv_mult: Option<&[Vec<f64>]>, correlations: &[ResidualCorrelation],
) -> f64 {
    foce_subject_nll_interaction_split(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        bloq_method,
        p_obs,
        frem_r_override,
        residual_error_eta,
        ruv_mult,
        correlations,
        None, // no density: the wrapper's callers are Gaussian
        None,
    )
}

/// [`foce_subject_nll_interaction_corr`], recording its parts into `parts` when one is
/// attached (see [`foce_subject_contribution_parts`]).
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_interaction_split(
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
    residual_error_eta: Option<usize>,
    ruv_mult: Option<&[Vec<f64>]>,
    correlations: &[ResidualCorrelation],
    // A density group's correction at η̂ (#1716): `c` into the prior, `½∇²c` into `H̃`.
    density: Option<&crate::stats::eta_density::DensityAtMode>,
    mut parts: PartSink<'_>,
) -> f64 {
    let prior_curv = crate::stats::eta_density::prior_curvature(&omega.inv, density);
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
        Some(correlations),
        None,
    ) else {
        note(&mut parts, RESIDUAL_VARIANCE_FAILED, f64::NAN);
        return SUBJECT_SENTINEL_NLL;
    };

    // η̂'Ω⁻¹η̂  +  log|Ω|  (both cached on OmegaMatrix).
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat)) + density.map_or(0.0, |d| d.c);
    note_gaussian_data(&mut parts, subject, bloq_method, g.data_ll, g.bloq_term);
    note(&mut parts, "random-effect prior", 0.5 * (eta_prior + omega.log_det));
    // H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + cens_hess + Ω⁻¹.  log|H̃| via Cholesky.
    // (`cens_hess` = 0 unless the subject has M3-censored rows.)
    let htilde = g.hrh + 0.5 * g.ctc + &g.cens_hess + &*prior_curv;
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => {
            note(&mut parts, CURVATURE_FAILED, f64::NAN);
            return SUBJECT_SENTINEL_NLL;
        }
    };
    note(&mut parts, "curvature log-determinant", 0.5 * log_det_htilde);

    0.5 * (g.data_ll + eta_prior + omega.log_det + log_det_htilde + g.bloq_term)
}

/// The part named when a row-wise residual variance is not finite and positive.
const RESIDUAL_VARIANCE_FAILED: &str = "Gaussian residual variance (non-finite or non-positive)";
/// The part named when `H̃` does not factor.
const CURVATURE_FAILED: &str = "curvature (H̃ not positive definite)";

/// Record the conditional Gaussian data parts: the residual term when the subject has a
/// Gaussian observation, and the censored term when the fit censors one of its rows.
fn note_gaussian_data(
    parts: &mut PartSink<'_>, subject: &Subject, bloq_method: BloqMethod, data_ll: f64,
    bloq_term: f64,
) {
    if !subject.observations.is_empty() {
        note(parts, "Gaussian residual term", 0.5 * data_ll);
    }
    if bloq_method.has_censored_row(&subject.cens) {
        note(parts, "censored (M3) term", 0.5 * bloq_term);
    }
}

/// FOCEI INTER per-subject −2·log marginal for a **correlated** (`block_sigma`)
/// residual covariance — the dense-`R` generalisation of `foce_subject_nll_interaction`,
/// which forms `R` row by row and would drop the off-diagonals. With `R` from
/// [`compute_r_matrix_with_correlations`](crate::stats::residual_error::compute_r_matrix_with_correlations):
///
/// ```text
///   data_ll(η̂) = rᵀ R⁻¹ r + log|R|
///   H̃          = Hᵀ R⁻¹ H + ½·B + Ω⁻¹
///   B_{kl}      = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l),  ∂R/∂η_k = Σ_m H[m,k]·∂R/∂f_m
/// ```
///
/// Reduces exactly to the diagonal form at ρ = 0 (`B = c̃ᵀc̃`); the dispatcher keeps
/// ρ = 0 on the diagonal path. `block_sigma` is rejected with M3, FREM, IOV and
/// `iiv_on_ruv`; `p_obs` inflates the `R` diagonal η-independently.
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
    foce_subject_nll_interaction_dense_cov(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        ResidualCov::block_sigma(correlations),
        p_obs,
        ruv_mult,
        None,
    )
}

/// [`foce_subject_nll_interaction_dense`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_nll_interaction_dense_cov(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_cov: ResidualCov<'_>,
    p_obs: &[f64],
    // Per-observation custom residual magnitude (#484); `None` on the legacy path.
    ruv_mult: Option<&[Vec<f64>]>,
    // IIV on the residual: `R` scales by `exp(2·η̂_r)` (as in the inner objective's
    // dense data term), so `∂R/∂η_r = 2R` joins `B`.
    residual_error_eta: Option<usize>,
) -> f64 {
    foce_subject_nll_interaction_dense_split(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        residual_cov,
        p_obs,
        ruv_mult,
        residual_error_eta,
        true,
        None, // no density: the wrapper's callers are Gaussian
        None,
    )
}

/// [`foce_subject_nll_interaction_dense_cov`], recording its parts into `parts` when one
/// is attached (see [`foce_subject_contribution_parts`]).
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_interaction_dense_split(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_cov: ResidualCov<'_>,
    p_obs: &[f64],
    ruv_mult: Option<&[Vec<f64>]>,
    residual_error_eta: Option<usize>,
    // FOCEI's `½·B` (the variance's η-curvature) in `H̃`, or FOCE's conditional form under a
    // density (#1716), which leaves it out.
    interaction: bool,
    // A density group's correction at η̂ (#1716): `c` into the prior, `½∇²c` into `H̃`.
    density: Option<&crate::stats::eta_density::DensityAtMode>,
    mut parts: PartSink<'_>,
) -> f64 {
    let prior_curv = crate::stats::eta_density::prior_curvature(&omega.inv, density);
    let n_obs = subject.observations.len();
    let n_eta = eta_hat.len();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);
    let ruv_scale = ruv_scale_from(eta_hat.as_slice(), residual_error_eta);

    // Dense R at η̂ (+ SDE process noise on the diagonal), assembled exactly as
    // the data term and FOCE-standard path assemble it.
    let mut r = crate::stats::residual_error::r_matrix_maybe_scaled(
        error_spec,
        ipreds,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_cov,
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

    let chol = match r.clone().cholesky() {
        Some(c) => c,
        None => {
            note(&mut parts, "Gaussian residual covariance R (not positive definite)", f64::NAN);
            return SUBJECT_SENTINEL_NLL;
        }
    };
    let r_inv = chol.inverse();

    // data_ll = rᵀ R⁻¹ r + log|R|.
    let residuals = DVector::from_iterator(
        n_obs,
        subject.observations.iter().zip(ipreds.iter()).map(|(&y, &f)| y - f),
    );
    let solved = chol.solve(&residuals);
    let data_ll = residuals.dot(&solved) + chol_log_det(&chol.l());

    // First-order conditional Hessian terms.
    // term1 = Hᵀ R⁻¹ H.
    let term1 = h_matrix.transpose() * &r_inv * h_matrix;

    // B_{kl} = tr(R⁻¹ ∂R/∂η_k R⁻¹ ∂R/∂η_l). Build M_k = R⁻¹ ∂R/∂η_k once per η,
    // then B_{kl} = tr(M_k M_l) = Σ_{p,q} M_k[p,q]·M_l[q,p].
    let dr = crate::stats::residual_error::compute_dr_df_matrices_cov(
        error_spec,
        ipreds,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        &subject.obs_l2,
        &subject.obs_rgroup,
        sigma_values,
        residual_cov,
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
        if ruv_scale != 1.0 {
            dr_k *= ruv_scale;
        }
        if Some(k) == residual_error_eta {
            // `R = exp(2η_r)·R₀ (+ P)` ⇒ `∂R/∂η_r = 2(R − P)`: the SDE process-noise
            // diagonal is η-independent, as on the diagonal path.
            let mut r_ruv = r.clone();
            for (j, v) in p_obs.iter().enumerate() {
                if j < r_ruv.nrows() {
                    r_ruv[(j, j)] -= *v;
                }
            }
            dr_k += 2.0 * r_ruv;
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

    let htilde = if interaction { term1 + 0.5 * b + &*prior_curv } else { term1 + &*prior_curv };
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat)) + density.map_or(0.0, |d| d.c);
    note(&mut parts, "Gaussian residual term", 0.5 * data_ll);
    note(&mut parts, "random-effect prior", 0.5 * (eta_prior + omega.log_det));
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => {
            note(&mut parts, CURVATURE_FAILED, f64::NAN);
            return SUBJECT_SENTINEL_NLL;
        }
    };
    note(&mut parts, "curvature log-determinant", 0.5 * log_det_htilde);

    0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde)
}

/// FOCEI NLL with the Gaussian interaction terms plus a TTE Laplace correction.
///
/// `tte_data_nll` — TTE NLL at η̂ over all TTE endpoints (scaled by 2; `data_ll` is
/// halved at the end). `tte_hessian` — η-Hessian of the raw non-Gaussian NLL (exact
/// with a dual program, else FD, #2437), added to `hrh` before `log|H̃|`.
/// `interaction = false` (FOCE) drops `½·CᵀC` from `H̃`: for mixed PK+TTE, and for FOCE's
/// conditional form under a density (#1716), which passes no endpoint term.
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
    tte_hessian: DMatrix<f64>,         // η-Hessian of the raw non-Gaussian NLL
    interaction: bool,                 // include the ½·CᵀC interaction term (FOCEI) or not (FOCE)
    residual_error_eta: Option<usize>, // IIV-on-RUV eta index (or None)
    // Per-observation custom residual magnitude (#484); `None` on the legacy
    // path. Scales the Gaussian (PK) rows' variance and f-derivative; the TTE
    // rows carry no residual variance so the multiplier never touches them.
    ruv_mult: Option<&[Vec<f64>]>,
    // The part sink (#2267); the caller has recorded the non-Gaussian data terms.
    // A density group's correction at η̂ (#1716): `c` into the prior, `½∇²c` into `H̃`.
    density: Option<&crate::stats::eta_density::DensityAtMode>,
    mut parts: PartSink<'_>,
) -> f64 {
    let prior_curv = crate::stats::eta_density::prior_curvature(&omega.inv, density);
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
        None,
        None,
    ) else {
        note(&mut parts, RESIDUAL_VARIANCE_FAILED, f64::NAN);
        return SUBJECT_SENTINEL_NLL;
    };
    note_gaussian_data(&mut parts, subject, bloq_method, g.data_ll, g.bloq_term);

    // Combine Gaussian and TTE data terms.
    // TTE NLL is scaled by 2 here to match the Gaussian data_ll convention
    // (both are halved at the end via the 0.5 factor).
    let data_ll = g.data_ll + 2.0 * tte_data_nll;
    // The part sink classifies a failed `H̃` by whether the non-Gaussian Hessian is what
    // broke it, so it keeps the Gaussian side and that Hessian's finiteness before the sum.
    let gaussian_htilde = parts.is_some().then(|| {
        let mut h = g.hrh.clone() + &*prior_curv;
        if interaction {
            h += 0.5 * &g.ctc + &g.cens_hess;
        }
        (h, tte_hessian.iter().all(|x| x.is_finite()))
    });
    // Accumulate TTE Hessian into the Gaussian Jacobian outer-product matrix.
    let hrh = g.hrh + tte_hessian;

    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat)) + density.map_or(0.0, |d| d.c);
    note(&mut parts, "random-effect prior", 0.5 * (eta_prior + omega.log_det));
    // FOCEI adds ½·CᵀC; FOCE omits it. `cens_hess` is FOCEI-order curvature (its `g2`
    // folds in ∂v/∂f, and C·z/C·m are `iiv_on_ruv` interaction), so it is gated the
    // same way; plain FOCE leaves censored rows out of `H̃`.
    let htilde = if interaction {
        hrh + 0.5 * g.ctc + &g.cens_hess + &*prior_curv
    } else {
        hrh + &*prior_curv
    };
    let log_det_htilde = match htilde.cholesky() {
        Some(c) => chol_log_det(&c.l()),
        None => {
            let name = match gaussian_htilde {
                Some((_, false)) => {
                    "non-Gaussian curvature (η-Hessian of the time-to-event, categorical or \
                     CTMM terms not finite)"
                }
                Some((h, true)) if h.clone().cholesky().is_some() => {
                    "curvature (H̃ not positive definite once the non-Gaussian Hessian is added)"
                }
                _ => CURVATURE_FAILED,
            };
            note(&mut parts, name, f64::NAN);
            return SUBJECT_SENTINEL_NLL;
        }
    };
    note(&mut parts, "curvature log-determinant", 0.5 * log_det_htilde);

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
    /// Σⱼ censored rows' contribution to `H̃` at FOCEI (Gauss-Newton) order: `g2·aⱼ'aⱼ`
    /// (`g2 = ∂²(−logΦ)/∂f²`) plus, under `iiv_on_ruv`, the `C·z` / `C·m·a` cross terms —
    /// `precise_ebe`'s censored block minus the `g1·d2f` term FOCEI drops. Added unscaled.
    cens_hess: DMatrix<f64>,
}

/// Each observation's own `−2 log L` term in the FOCEI conditional data term at
/// `ipreds`/`eta_hat` (#2266): `r²/V + ln V` (quantified) or `−2 log Φ` (censored for
/// this fit), read off [`gaussian_foce_accum`]'s sweep so every input resolves as the
/// scorer resolves it. A row whose `V` is rejected is `NaN`. Terms sum to
/// `data_ll + bloq_term` when all rows are usable; under FOCE or dense `R` they are
/// still each record's own contribution at η̂.
pub(crate) fn conditional_obs_neg2ll_terms(
    model: &CompiledModel, subject: &Subject, theta: &[f64], ipreds: &[f64], eta_hat: &[f64],
    sigma_values: &[f64], residual_cov: ResidualCov<'_>,
) -> Vec<f64> {
    let n_obs = subject.observations.len();
    if ipreds.len() != n_obs {
        return vec![f64::NAN; n_obs];
    }
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat, sigma_values)
    } else {
        Vec::new()
    };
    // The joint TTE path scores its Gaussian rows with no FREM override and the
    // legacy (uncorrelated) variance; mirror it so a joint subject's terms are its own.
    #[cfg(feature = "survival")]
    let joint_tte = !subject.obs_records.is_empty();
    #[cfg(not(feature = "survival"))]
    let joint_tte = false;
    let frem_r_override = if joint_tte {
        None
    } else {
        build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma_values)
    };
    let correlations = (!joint_tte).then_some(residual_cov.pairs);
    let mut terms = vec![f64::NAN; n_obs];
    let _ = gaussian_foce_accum(
        subject,
        ipreds,
        &DMatrix::<f64>::zeros(n_obs, 0),
        &model.error_spec,
        sigma_values,
        model.bloq_in_force(),
        &p_obs,
        0,
        frem_r_override.as_deref(),
        model.residual_error_eta,
        ruv_scale_from(eta_hat, model.residual_error_eta),
        ruv_mult.as_deref(),
        correlations,
        Some(&mut terms),
    );
    terms
}

/// Shared Gaussian accumulation loop for the FOCE/FOCEI interaction path: the
/// per-observation Hessian terms from residuals and variance derivatives. `None` if any
/// variance is non-finite or non-positive (caller returns the 1e20 sentinel).
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
    // The live `block_sigma` correlations when a correlated model's subject is scored
    // row-wise (censored or FREM rows), `None` for the legacy variance. The caller makes
    // that call (it has asked `ResidualCov::is_dense`), not an emptiness test here.
    correlations: Option<&[ResidualCorrelation]>,
    // `Some` records each row's own `−2 log L` term (`r²/V + ln V`, or `−2 log Φ` on a
    // censored row) at its index (#2266). A row with an unusable `V` is recorded as
    // `NaN` and the sweep continues instead of returning, so every row is recorded;
    // the returned terms are then not a marginal's and the caller ignores them.
    mut row_terms: Option<&mut [f64]>,
) -> Option<GaussianFoceTerms> {
    let n_obs = subject.observations.len();
    let mult_row = |j: usize| -> Option<&[f64]> { ruv_mult.map(|m| m[j].as_slice()) };
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

    // Partition observation indices into quantified vs censored (M3 only).
    let (quant_idx, bloq_idx): (Vec<usize>, Vec<usize>) = (0..n_obs)
        .partition(|&j| !bloq_method.is_censored_row(subject.cens.get(j).copied().unwrap_or(0)));

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
        // `residual_rd` leaves `ruv_scale` to the caller: scaling `R` scales `∂R/∂f` too and
        // both cancel in `c_scale`. FREM rows take the override variance and `∂R/∂f = 0`.
        let (v_resid, dvar_df) = match frem_ov {
            Some(v) => (v, 0.0),
            None => {
                let (rv, dv) = match correlations {
                    None => residual_rd(error_spec, err_keys[j], f, sigma_values, mult_row(j)),
                    Some(c) => {
                        let (v, d, _) = crate::stats::residual_error::residual_rd2_corr(
                            error_spec,
                            err_keys[j],
                            f,
                            sigma_values,
                            c,
                            mult_row(j),
                        );
                        (v, d)
                    }
                };
                (rv * ruv_scale, dv * ruv_scale)
            }
        };
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            match row_terms.as_deref_mut() {
                Some(t) => {
                    t[j] = f64::NAN;
                    continue;
                }
                None => return None,
            }
        }
        let r = subject.observations[j] - f;
        let term = r * r / v + v.ln();
        if let Some(t) = row_terms.as_deref_mut() {
            t[j] = term;
        }
        data_ll += term;

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
        let (rv, dv, d2v) = match correlations {
            None => residual_rd2(error_spec, cmt, f, sigma_values, mult_row(j)),
            Some(c) => crate::stats::residual_error::residual_rd2_corr(
                error_spec,
                cmt,
                f,
                sigma_values,
                c,
                mult_row(j),
            ),
        };
        let (v_resid, d, d2) = (rv * ruv_scale, dv * ruv_scale, d2v * ruv_scale);
        // Inflate by the EKF process-noise term exactly as the quantified rows do
        // (line above), so a censored row's `v` — in both the data term and `H̃` —
        // matches its subject's Gaussian rows under an SDE model. `p_obs` is
        // f-independent, so `d`/`d2` (∂v/∂f, ∂²v/∂f²) are unchanged.
        let v = v_resid + p_obs.get(j).copied().unwrap_or(0.0);
        if !(v.is_finite() && v > 0.0) {
            match row_terms.as_deref_mut() {
                Some(t) => {
                    t[j] = f64::NAN;
                    continue;
                }
                None => return None,
            }
        }
        let cens = subject.cens.get(j).copied().unwrap_or(0);
        let term = -2.0 * m3_logcdf(limit, f, v.sqrt(), cens);
        if let Some(t) = row_terms.as_deref_mut() {
            t[j] = term;
        }
        bloq_term += term;

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

    Some(GaussianFoceTerms { data_ll, hrh, ctc, bloq_term, cens_hess })
}

/// Conditional η-precision at `eta_hat` — the FOCEI/Laplace approximation to
/// `−∂² log p(η | y, θ) / ∂η²` (Hessian of the halved [`individual_nll`]):
///
/// ```text
///   H̃ = Hᵀ R⁻¹ H + ½·C̃ᵀC̃ + Ω⁻¹       (interaction = true)
///   H̃ = Hᵀ R⁻¹ H + Ω⁻¹                (interaction = false)
/// ```
///
/// Built from the same `gaussian_foce_accum` loop as the FOCEI objective, for the
/// SAEM f-SAEM kernel (`estimation::saem_imh`, #1449). Scope: Gaussian rows with
/// diagonal R (SDE `p_obs` added η-independently); FREM overrides, `block_sigma` and
/// IIV-on-RUV are gated out by the caller. `None` on a non-finite/non-positive variance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conditional_eta_precision(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &[f64], ipreds: &[f64],
    p_obs: &[f64], h_matrix: &DMatrix<f64>, omega: &OmegaMatrix, sigma_values: &[f64],
    interaction: bool,
) -> Option<DMatrix<f64>> {
    let n_eta = eta_hat.len();
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    let terms = gaussian_foce_accum(
        subject,
        ipreds,
        h_matrix,
        &model.error_spec,
        sigma_values,
        model.bloq_in_force(),
        p_obs,
        n_eta,
        None,
        model.residual_error_eta,
        ruv_scale_from(eta_hat, model.residual_error_eta),
        ruv_mult.as_deref(),
        None,
        None,
    )?;
    let mut htilde = terms.hrh + terms.cens_hess + &omega.inv;
    if interaction {
        htilde += terms.ctc * 0.5;
    }
    htilde.iter().all(|v| v.is_finite()).then_some(htilde)
}

/// R_tilde = H * Omega * H' + diag(r_diag)
pub(crate) fn compute_r_tilde(
    h: &DMatrix<f64>, omega: &DMatrix<f64>, r_diag: &[f64],
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
    h: &DMatrix<f64>, omega: &DMatrix<f64>, r: &DMatrix<f64>,
) -> DMatrix<f64> {
    let h_omega = h * omega;
    &h_omega * h.transpose() + r
}

fn select_symmetric_submatrix(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), indices.len(), |row, col| matrix[(indices[row], indices[col])])
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
            return SUBJECT_SENTINEL_NLL;
        }
    }
    2.0 * ld
}

/// IOV-aware FOCE per-subject NLL — a linearised marginal over the full random-effect
/// vector `b = [η, κ₁, …, κ_K]`: with `H_full = [∂f/∂η │ ∂f/∂κ₁ │ …]` and
/// `Σ_b = blkdiag(Ω_bsv, Ω_iov × K)`, evaluates `0.5·[(y−f₀)ᵀ R̃⁻¹ (y−f₀) + log|R̃|]`,
/// `R̃ = H_full Σ_b H_fullᵀ + R`.
///
/// No separate κ prior: the earlier `MAP` penalty omitted the κ Laplace determinant,
/// leaving `+0.5·K·log|Ω_iov|` → −∞ as Ω_iov → 0 (#101). K = 0 reduces exactly to
/// [`foce_subject_nll`]. `kappas[k]` is occasion k's EBE (order of
/// `iov_occasion_groups`); empty `kappas` takes the non-IOV path.
///
/// Scores `block_sigma` at its **declared** ρ; the fit uses
/// `foce_subject_nll_iov_cov` with the live one (#979).
pub fn foce_subject_nll_iov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega_bsv: &OmegaMatrix, sigma_values: &[f64], interaction: bool,
    kappas: &[DVector<f64>], omega_iov: &OmegaMatrix,
) -> f64 {
    foce_subject_nll_iov_cov(
        model,
        subject,
        theta,
        eta_hat,
        h_matrix,
        omega_bsv,
        sigma_values,
        model.residual_cov_theta(theta),
        interaction,
        kappas,
        omega_iov,
    )
}

/// [`foce_subject_nll_iov`] over an explicit residual covariance structure: the live
/// `block_sigma` off-diagonals FOCE/FOCEI estimate (#847, #979), as the non-IOV
/// [`foce_subject_nll_cov`] takes them. `residual_cov` must be the model's own
/// structure (`CompiledModel::residual_cov_at` / `residual_cov_theta`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_subject_nll_iov_cov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega_bsv: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: crate::stats::residual_error::ResidualCov<'_>, interaction: bool,
    kappas: &[DVector<f64>], omega_iov: &OmegaMatrix,
) -> f64 {
    // A nested-level subject with no κ̂ was never given its cells (#1082): refused, not
    // scored as a BSV-only subject. With cells, `iov_linearization` refuses it below.
    if kappas.is_empty() && model.has_nested_levels() {
        return SUBJECT_SENTINEL_NLL;
    }
    if kappas.is_empty() {
        return foce_subject_nll_cov(
            model,
            subject,
            theta,
            eta_hat,
            h_matrix,
            omega_bsv,
            sigma_values,
            residual_cov,
            interaction,
        );
    }

    match iov_linearization(model, subject, theta, eta_hat, h_matrix, omega_bsv, kappas, omega_iov)
    {
        Some(lin) => foce_subject_nll_iov_linearized(
            model,
            subject,
            theta,
            sigma_values,
            residual_cov,
            interaction,
            &lin,
        ),
        None => SUBJECT_SENTINEL_NLL,
    }
}

/// The linearization an IOV subject is scored and diagnosed on: joint EBE
/// `b̂ = [η̂, κ̂₁, …, κ̂_K]`, `H_full = ∂f/∂b` at `b̂`, `Σ_b = blkdiag(Ω, Ω_IOV × K)` and
/// `IPRED = f(b̂)`. Built once by [`iov_linearization`] for both the IOV marginal and
/// post-fit CWRES, so CWRES linearizes over κ as NONMEM does (#1948).
pub(crate) struct IovLinearization {
    /// `f(η̂, κ̂)` through the continuous per-occasion prediction (`pk::predict_iov`).
    pub(crate) ipreds: Vec<f64>,
    /// `[η̂, κ̂₁, …, κ̂_K]` in [`crate::stats::stacked_layout::StackedLayout`] order — or,
    /// under nested levels (#1082), the compact `[η̂, κ̂ per group of each level]` of
    /// `crate::stats::nested_levels::NestedLayout`.
    pub(crate) b_hat: DVector<f64>,
    /// `[∂f/∂η │ ∂f/∂κ₁ │ … │ ∂f/∂κ_K]`, `n_obs × n_b`, over the same `b`.
    pub(crate) h_full: DMatrix<f64>,
    /// `blkdiag(Ω, Ω_IOV, …, Ω_IOV)`, or the nested compact prior
    /// `Ω ⊕_L (I_{G_L} ⊗ Ω_L)`.
    pub(crate) sigma_b: OmegaMatrix,
    /// The number of occasion groups K.
    pub(crate) n_occasions: usize,
    /// `dim η` (the leading block of `b_hat`).
    pub(crate) n_eta: usize,
    /// `dim κ_k`, the order of `Ω_IOV`.
    pub(crate) n_iov: usize,
}

#[cfg(test)]
thread_local! {
    /// Test switch: take the nested κ columns' central difference even where the recording
    /// serves, for the test pinning the two against each other.
    pub(crate) static NESTED_KAPPA_FD_FORCE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

fn nested_kappa_fd_forced() -> bool {
    #[cfg(test)]
    return NESTED_KAPPA_FD_FORCE.with(|c| c.get());
    #[cfg(not(test))]
    false
}

/// Build the subject's [`IovLinearization`], or `None` when `kappas` does not hold one
/// width-`n_iov` vector per occasion group (never from the EBE pipeline; callers map
/// it to the sentinel or to `NaN` CWRES).
#[allow(clippy::too_many_arguments)]
pub(crate) fn iov_linearization(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>, omega_bsv: &OmegaMatrix, kappas: &[DVector<f64>],
    omega_iov: &OmegaMatrix,
) -> Option<IovLinearization> {
    // #1082: the compact layout, built once. A nested-level subject never given its cells
    // has none, and is the caller's refusal rather than a one-level linearization.
    let nested = crate::stats::nested_levels::NestedLayout::of(model, subject);
    if model.has_nested_levels() && nested.is_none() {
        return None;
    }
    let n_obs = subject.obs_times.len();
    let n_eta = eta_hat.len();
    let n_iov = omega_iov.matrix.nrows();
    let layout = crate::stats::stacked_layout::StackedLayout::new_cached(n_eta, n_iov, subject);
    if kappas.len() != layout.n_groups() || kappas.iter().any(|k| k.len() != n_iov) {
        return None;
    }
    let k_occ = layout.n_groups();
    let n_b = layout.n_stacked();

    // ipreds at the joint EBE via the continuous per-occasion prediction (cross-occasion
    // carryover, #104), plus the augmented H. BSV columns from the passed-in H; κ columns
    // from the exact stacked recording (central differences where it declines). A κ
    // column is dense across rows: κ_k also affects carryover into later occasions.
    let kappa_slices: Vec<Vec<f64>> = kappas.iter().map(|k| k.as_slice().to_vec()).collect();
    // One layout and one scratch for the 2·n_b + 1 predictions below (#2037): the layout
    // depends on the subject alone, and each prediction rewrites every scratch event.
    // The prediction pads an unlabelled record's κ to the *model's* width, which a caller
    // handing an `Ω_IOV` of another order (a hand-built fixture) does not share.
    let model_width;
    let pred_layout = if n_iov == model.n_kappa {
        &layout
    } else {
        model_width =
            crate::stats::stacked_layout::StackedLayout::new_cached(n_eta, model.n_kappa, subject);
        &model_width
    };
    let mut scratch = pk::EventPkParams::default();
    let schedule =
        crate::estimation::inner_optimizer::cacheable_schedule_iov_cached(model, subject);
    let mut predict = |k: &[Vec<f64>]| {
        pk::predict_iov_in(
            model,
            subject,
            pred_layout,
            theta,
            eta_hat.as_slice(),
            k,
            &mut scratch,
            schedule.as_ref().as_ref(),
        )
    };
    let ipreds = predict(&kappa_slices);

    // #1082: under nested levels `b` is the compact vector (one κ block per group of each
    // level), so a κ column perturbs one group's κ in every cell that shares it.
    if let Some(nested) = nested {
        let b_hat_vec = nested.compact(eta_hat.as_slice(), &kappa_slices)?;
        let n_b = nested.n_compact();
        let mut h_full = DMatrix::zeros(n_obs, n_b);
        for j in 0..n_obs {
            for c in 0..n_eta {
                h_full[(j, c)] = h_matrix[(j, c)];
            }
        }
        // The compact κ columns: exact, from the recording of the per-cell vector over the
        // axes it expands from (`exact_kappa_jacobian`, κ lanes only); central differences of the
        // prediction (2 solves per compact κ) where the recorders decline.
        let cells: Vec<f64> =
            eta_hat.iter().copied().chain(kappa_slices.iter().flatten().copied()).collect();
        let recorded = if nested_kappa_fd_forced() {
            None
        } else {
            crate::sens::ode_provider::adjoint::exact_kappa_jacobian(model, subject, theta, &cells)
                .filter(|j| {
                    j.nrows() == n_obs && j.ncols() == n_b && j.iter().all(|v| v.is_finite())
                })
        };
        if let Some(jac) = recorded {
            for j in 0..n_obs {
                for c in n_eta..n_b {
                    h_full[(j, c)] = jac[(j, c)];
                }
            }
        } else {
            const EPS: f64 = 1e-6;
            let mut bpert = b_hat_vec.clone();
            for c in n_eta..n_b {
                let orig = bpert[c];
                let step = EPS * (1.0 + orig.abs());
                bpert[c] = orig + step;
                let preds_plus = predict(&nested.expand(&bpert));
                bpert[c] = orig - step;
                let preds_minus = predict(&nested.expand(&bpert));
                bpert[c] = orig;
                let inv_2step = 1.0 / (2.0 * step);
                for j in 0..n_obs {
                    h_full[(j, c)] = (preds_plus[j] - preds_minus[j]) * inv_2step;
                }
            }
        }
        let sigma_b_mat = nested.prior(&omega_bsv.matrix, &omega_iov.matrix);
        return Some(IovLinearization {
            ipreds,
            b_hat: DVector::from_vec(b_hat_vec),
            h_full,
            sigma_b: OmegaMatrix::from_matrix(sigma_b_mat, Vec::new(), false),
            n_occasions: k_occ,
            n_eta,
            n_iov,
        });
    }

    let mut h_full = DMatrix::zeros(n_obs, n_b);
    for j in 0..n_obs {
        for c in 0..n_eta {
            h_full[(j, c)] = h_matrix[(j, c)];
        }
    }
    // Joint EBE vector b̂ = [η̂, κ̂₁, …, κ̂_K].
    let mut b_hat = DVector::zeros(n_b);
    for i in 0..n_eta {
        b_hat[i] = eta_hat[i];
    }
    for (k, kap) in kappas.iter().enumerate() {
        for ki in 0..n_iov {
            b_hat[layout.kappa_offset(k) + ki] = kap[ki];
        }
    }

    // κ columns: the exact stacked Jacobian's κ lanes from the unified recording (one
    // tape-free first-order pass, `exact_kappa_jacobian`), else central differences of
    // `predict_iov` (2·K·n_iov extra solves) for an engine no recorder serves.
    let recorded = crate::sens::ode_provider::adjoint::exact_kappa_jacobian(
        model,
        subject,
        theta,
        b_hat.as_slice(),
    )
    .filter(|j| j.nrows() == n_obs && j.ncols() == n_b);
    if let Some(jac) = recorded {
        for j in 0..n_obs {
            for c in n_eta..n_b {
                h_full[(j, c)] = jac[(j, c)];
            }
        }
    } else {
        // Reused κ buffer: perturb one element in place and restore it, rather than
        // cloning all occasions' κ twice per FD step.
        let mut kpert = kappa_slices.clone();
        const EPS: f64 = 1e-6;
        for k in 0..k_occ {
            let col_base = layout.kappa_offset(k);
            for ki in 0..n_iov {
                let orig = kpert[k][ki];
                let step = EPS * (1.0 + orig.abs());
                kpert[k][ki] = orig + step;
                let preds_plus = predict(&kpert);
                kpert[k][ki] = orig - step;
                let preds_minus = predict(&kpert);
                kpert[k][ki] = orig;
                let inv_2step = 1.0 / (2.0 * step);
                for j in 0..n_obs {
                    h_full[(j, col_base + ki)] = (preds_plus[j] - preds_minus[j]) * inv_2step;
                }
            }
        }
    }

    // Block-diagonal prior covariance Σ_b = blkdiag(Ω_bsv, Ω_iov × K).
    // `from_matrix` regularises if a sub-block is not PD, matching the
    // robustness of the non-IOV OmegaMatrix path; the standard/interaction
    // FOCE routines and `compute_cwres_cov` read only `Σ_b.matrix`.
    let sigma_b_mat = layout.prior(&omega_bsv.matrix, &omega_iov.matrix);
    let sigma_b = OmegaMatrix::from_matrix(sigma_b_mat, Vec::new(), false);

    Some(IovLinearization { ipreds, b_hat, h_full, sigma_b, n_occasions: k_occ, n_eta, n_iov })
}

/// [`foce_subject_nll_iov_cov`] on an already-built [`IovLinearization`].
pub(crate) fn foce_subject_nll_iov_linearized(
    model: &CompiledModel, subject: &Subject, theta: &[f64], sigma_values: &[f64],
    residual_cov: crate::stats::residual_error::ResidualCov<'_>, interaction: bool,
    lin: &IovLinearization,
) -> f64 {
    foce_subject_nll_iov_linearized_split(
        model,
        subject,
        theta,
        sigma_values,
        residual_cov,
        interaction,
        lin,
        None,
    )
}

/// The parts of an IOV subject's [`foce_subject_nll_iov_linearized`] contribution, as
/// [`foce_subject_contribution_parts`] records them without κ (#2267). `random-effect
/// prior` is over the stacked `[η, κ]`; curvature and `R̃` carry κ columns. A FREM
/// refusal has the one part `IOV + FREM refusal`.
pub(crate) fn foce_subject_iov_contribution_parts(
    model: &CompiledModel, subject: &Subject, theta: &[f64], sigma_values: &[f64],
    residual_cov: crate::stats::residual_error::ResidualCov<'_>, interaction: bool,
    lin: &IovLinearization,
) -> Vec<(String, f64)> {
    let mut parts = Vec::new();
    foce_subject_nll_iov_linearized_split(
        model,
        subject,
        theta,
        sigma_values,
        residual_cov,
        interaction,
        lin,
        Some(&mut parts),
    );
    parts
}

/// [`foce_subject_nll_iov_linearized`], recording its parts into `parts` when one is
/// attached.
#[allow(clippy::too_many_arguments)]
fn foce_subject_nll_iov_linearized_split(
    model: &CompiledModel, subject: &Subject, theta: &[f64], sigma_values: &[f64],
    residual_cov: crate::stats::residual_error::ResidualCov<'_>, interaction: bool,
    lin: &IovLinearization, mut parts: PartSink<'_>,
) -> f64 {
    let IovLinearization { ipreds, b_hat, h_full, sigma_b, n_occasions: k_occ, n_eta, n_iov } = lin;
    let (k_occ, n_eta, n_iov) = (*k_occ, *n_eta, *n_iov);
    let eta_hat = &b_hat.as_slice()[..n_eta];

    // κ is integrated out through R̃ like η, so no separate κ prior (it would double
    // count). M3 does not promote IOV-FOCE to interaction (#591, as #367 for non-IOV):
    // censored rows re-enter via `foce_subject_nll_standard` as
    // `−logΦ((LLOQ−f0)/√R̃ⱼⱼ)` over the stacked system (#646); FOCEI takes the interaction
    // path. Routing is by `interaction` alone, so no `m3_active` gate here.
    let p_obs_iov = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta_hat, sigma_values)
    } else {
        Vec::new()
    };
    // IOV + FREM is unsupported: the augmented b̂ vector and block-diagonal
    // Σ_b are not set up for FREM R-overrides.  Return a sentinel NLL so the
    // optimizer steers away from this region rather than silently ignoring FREM.
    if iov_refuses_frem(model, subject) {
        note(&mut parts, "IOV + FREM refusal", IOV_FREM_REFUSAL_NLL);
        return IOV_FREM_REFUSAL_NLL;
    }
    // Per-observation custom residual magnitude (#484); η/κ-independent.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // A density group's correction at b̂ (#1716). The stacked `Σ_b = Ω_BSV ⊕ Ω_IOV…` puts
    // the BSV block first, and a density reads BSV η only, so its inverse's BSV block is
    // `Ω_BSV⁻¹`.
    let density = crate::stats::eta_density::DensityAtMode::at(
        &match crate::stats::stacked_layout::StackedLayout::sens_of(model, subject) {
            Some(l) => crate::stats::eta_density::layout_groups(model, &l.kappa_blocks()),
            None => crate::stats::eta_density::stacked_groups(model, k_occ).into_owned(),
        },
        b_hat.as_slice(),
        &sigma_b.inv,
        theta,
    );
    // Non-Gaussian records (TTE, discrete) read BSV η only and enter the stacked marginal
    // over the BSV block, so the outer objective matches the one `individual_nll_iov`'s
    // EBE is the mode of.
    #[cfg(feature = "survival")]
    if !subject.obs_records.is_empty() {
        let eta_bsv = DVector::from_column_slice(eta_hat);
        let (q, hq) = non_gaussian_mode_terms(model, subject, theta, &eta_bsv, None, &mut parts);
        let n_b = b_hat.len();
        let mut h = DMatrix::<f64>::zeros(n_b, n_b);
        h.view_mut((0, 0), (n_eta, n_eta)).copy_from(&hq);
        return foce_subject_nll_interaction_with_tte(
            subject,
            ipreds,
            b_hat,
            h_full,
            sigma_b,
            sigma_values,
            &model.error_spec,
            model.bloq_in_force(),
            &p_obs_iov,
            q,
            h,
            interaction,
            model.residual_error_eta,
            ruv_mult.as_deref(),
            density.as_ref(),
            parts,
        );
    }
    // FOCE (no interaction) under a density: the conditional form, as in the non-IOV
    // dispatcher (`foce_subject_nll_cov_split`).
    if density.is_some() && !interaction {
        let n_b = b_hat.len();
        if residual_cov.is_dense() && !model.bloq_in_force().has_censored_row(&subject.cens) {
            return foce_subject_nll_interaction_dense_split(
                subject,
                &ipreds,
                &b_hat,
                &h_full,
                &sigma_b,
                sigma_values,
                &model.error_spec,
                residual_cov,
                &p_obs_iov,
                ruv_mult.as_deref(),
                model.residual_error_eta,
                false,
                density.as_ref(),
                parts,
            );
        }
        return foce_subject_nll_interaction_with_tte(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            model.bloq_in_force(),
            &p_obs_iov,
            0.0,
            DMatrix::zeros(n_b, n_b),
            false,
            model.residual_error_eta,
            ruv_mult.as_deref(),
            density.as_ref(),
            parts,
        );
    }
    // A dense `R` here is a `correlation … by` line (#1082 PR5), whose ρ is a θ, or a
    // `block_sigma` off-diagonal (#979), whose live ρ the caller passes in `residual_cov`.
    if interaction {
        // As `foce_subject_nll_cov`: dense `R` takes the dense interaction marginal over the
        // stacked `[η, κ…]` system, except censored subjects, scored row-wise both here and
        // in the IOV EBE search (`individual_nll_iov_frozen_in`).
        let rowwise = model.bloq_in_force().has_censored_row(&subject.cens);
        if !residual_cov.is_dense() || rowwise {
            foce_subject_nll_interaction_split(
                subject,
                &ipreds,
                &b_hat,
                &h_full,
                &sigma_b,
                sigma_values,
                &model.error_spec,
                model.bloq_in_force(),
                &p_obs_iov,
                None, // IOV + FREM unsupported (guarded above)
                model.residual_error_eta,
                ruv_mult.as_deref(),
                &[],
                density.as_ref(),
                parts,
            )
        } else {
            foce_subject_nll_interaction_dense_split(
                subject,
                &ipreds,
                &b_hat,
                &h_full,
                &sigma_b,
                sigma_values,
                &model.error_spec,
                residual_cov,
                &p_obs_iov,
                ruv_mult.as_deref(),
                model.residual_error_eta,
                true,
                density.as_ref(),
                parts,
            )
        }
    } else {
        // FOCE: R at f(η=0, κ=0), matching the non-IOV marginal so Ω_iov→0 reduces exactly.
        // Additive error keeps f0 (bit-identical). `kappas` has one vector per occasion.
        let pop_preds = population_variance_preds_iov(model, subject, theta, n_eta, n_iov, k_occ);
        foce_subject_nll_standard_split(
            subject,
            &ipreds,
            &b_hat,
            &h_full,
            &sigma_b,
            sigma_values,
            &model.error_spec,
            residual_cov,
            model.bloq_in_force(),
            &p_obs_iov,
            None,
            pop_preds.as_deref(),
            ruv_mult.as_deref(),
            parts,
        )
    }
}

/// Population FOCE objective with IOV: sum over all subjects using
/// `foce_subject_nll_iov`.  `kappas_per_subject[i]` holds the per-occasion
/// kappa EBEs for subject i (empty slice = no IOV for that subject).
pub fn foce_population_nll_iov(
    model: &CompiledModel, population: &Population, theta: &[f64], eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>], kappas_per_subject: &[Vec<DVector<f64>>], omega_bsv: &OmegaMatrix,
    omega_iov: &OmegaMatrix, sigma_values: &[f64], interaction: bool,
) -> f64 {
    // Per-subject NLLs in parallel, summed serially in subject order: rayon's `sum`
    // depends on thread count and f64 addition is not associative (#703).
    let per_subj: Vec<f64> = crate::par::map(population.subjects.len(), |i| {
        let subject = &population.subjects[i];
        let kappas =
            if i < kappas_per_subject.len() { kappas_per_subject[i].as_slice() } else { &[] };
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
    });
    per_subj.iter().sum()
}

/// Population FOCE objective: sum over all subjects. Takes `block_sigma` ρ only; a
/// `correlation … by COL` line (#1081) is **not** applied (`fit()` uses the in-crate
/// form that carries it).
#[allow(clippy::too_many_arguments)]
pub fn foce_population_nll(
    model: &CompiledModel, population: &Population, theta: &[f64], eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>], omega: &OmegaMatrix, sigma_values: &[f64],
    residual_correlations: &[ResidualCorrelation], interaction: bool,
) -> f64 {
    foce_population_nll_cov(
        model,
        population,
        theta,
        eta_hats,
        h_matrices,
        omega,
        sigma_values,
        ResidualCov::block_sigma(residual_correlations),
        interaction,
    )
}

/// [`foce_population_nll`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
#[allow(clippy::too_many_arguments)]
pub(crate) fn foce_population_nll_cov(
    model: &CompiledModel, population: &Population, theta: &[f64], eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>], omega: &OmegaMatrix, sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, interaction: bool,
) -> f64 {
    // Deterministic reduction: collect per-subject NLLs in subject order, then
    // sum serially. A parallel `.sum()` would make the OFV depend on the
    // thread count (f64 addition is non-associative; rayon splits by worker
    // count). See `foce_population_nll_iov` and #703.
    let per_subj: Vec<f64> = crate::par::map(population.subjects.len(), |i| {
        let subject = &population.subjects[i];
        foce_subject_nll_cov(
            model,
            subject,
            theta,
            &eta_hats[i],
            &h_matrices[i],
            omega,
            sigma_values,
            residual_cov,
            interaction,
        )
    });
    per_subj.iter().sum()
}

/// CWRES (Hooker et al. 2007, NONMEM's `CWRES`) for a subject: the **decorrelated**
/// residual `R̃^{-1/2}(y − f0)` with `f0 = f(η̂) − H·η̂`, `R̃ = H·Ω·Hᵀ + R` and the
/// *symmetric* inverse square root (eigendecomposition). `R` is at `r_preds` when given
/// (`IPRED` under interaction, `f(η = 0)` under FOCE), else at `f0`.
///
/// Pinned against NONMEM 7.5.1 (40-subject oral, #1182): reproduces its `CWRES` to RMS
/// 1e-4 (table rounding), where marginal standardisation was off by 0.75 and Cholesky
/// by 0.57.
///
/// Under IOV pass the stacked forms (`eta_hat = b̂`, `h_matrix = ∂f/∂b`,
/// `omega = blkdiag(Ω, Ω_IOV × K)`), as the fit's sdtab does (#1948).
///
/// Rows censored **for this fit** ([`BloqMethod::is_censored_row`]) get `NaN` and are
/// left out of the decorrelation; under `drop` a `CENS != 0` row stays (#1499). Takes
/// `block_sigma` ρ only; a `correlation … by COL` line (#1081) is **not** applied.
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
    // How the fit scored `CENS` rows. Only `M3` makes a flagged row censored;
    // under `Drop` the likelihood kept it as an ordinary observation and so does
    // this (#1499).
    bloq_method: BloqMethod,
) -> Vec<f64> {
    compute_cwres_cov(
        subject,
        ipreds,
        eta_hat,
        h_matrix,
        omega,
        sigma_values,
        error_spec,
        ResidualCov::block_sigma(residual_correlations),
        frem_r_override,
        residual_error_eta,
        ruv_mult,
        r_preds,
        bloq_method,
    )
}

/// [`compute_cwres`] over a resolved [`ResidualCov`] — the form every in-crate
/// caller uses (#1081).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_cwres_cov(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_spec: &ErrorSpec,
    residual_cov: ResidualCov<'_>,
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
    // How the fit scored `CENS` rows. Only `M3` makes a flagged row censored;
    // under `Drop` the likelihood kept it as an ordinary observation and so does
    // this (#1499).
    bloq_method: BloqMethod,
) -> Vec<f64> {
    let n_obs = subject.observations.len();
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = error_spec.obs_keys(subject);

    // f0 = ipred - H * eta_hat
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds.iter().enumerate().map(|(j, &ip)| ip - h_eta[j]).collect();

    // R_tilde, with R at the requested predictions when given (see above).
    let r_eval: &[f64] = r_preds.unwrap_or(&f0);
    let mut r_matrix = crate::stats::residual_error::r_matrix_maybe_scaled(
        error_spec,
        r_eval,
        err_keys.as_ref(),
        subject,
        sigma_values,
        residual_cov,
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

    // Symmetric inverse square root R̃ = V·Λ·Vᵀ, R̃^{-1/2} = V·Λ^{-1/2}·Vᵀ (NONMEM's
    // decorrelation; see the doc comment) over uncensored rows. A non-positive eigenvalue
    // falls back to marginal standardisation rather than panicking.
    let live: Vec<usize> = (0..n_obs)
        .filter(|&j| !bloq_method.is_censored_row(subject.cens.get(j).copied().unwrap_or(0)))
        .collect();
    let mut out = vec![f64::NAN; n_obs];
    if live.is_empty() {
        return out;
    }
    let resid =
        DVector::from_iterator(live.len(), live.iter().map(|&j| subject.observations[j] - f0[j]));
    let sub = r_tilde.select_rows(&live).select_columns(&live);
    let eig = nalgebra::SymmetricEigen::new(sub);
    if eig.eigenvalues.iter().all(|&l| l > 0.0 && l.is_finite()) {
        let vt_r = eig.eigenvectors.transpose() * &resid;
        let scaled = DVector::from_iterator(
            live.len(),
            vt_r.iter().zip(eig.eigenvalues.iter()).map(|(v, l)| v / l.sqrt()),
        );
        let z = &eig.eigenvectors * scaled;
        for (k, &j) in live.iter().enumerate() {
            out[j] = z[k];
        }
    } else {
        for (k, &j) in live.iter().enumerate() {
            out[j] =
                resid[k] / crate::stats::residual_error::guard_variance(r_tilde[(j, j)]).sqrt();
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
    occ_order.into_iter().map(|occ| (occ, occ_map.remove(&occ).unwrap())).collect()
}

/// IOV kappa groups for a subject: observation occasions in [`split_obs_by_occasion`]
/// order, then dose-only occasions (first-dose order), then occasions carried only by
/// pk-only rows (`EVID=2` / `MDV=1`, #1876), then only by EVID=3/4 reset rows (#1153),
/// each with an empty observation list. Each of those rows runs `$PK` under its own
/// occasion and NONMEM gives it an `ETA`, so κ = 0 is left only for unlabelled records.
pub fn iov_occasion_groups(subject: &Subject) -> Vec<(u32, Vec<usize>)> {
    let mut groups = split_obs_by_occasion(subject);
    let mut seen: std::collections::HashSet<u32> = groups.iter().map(|(occ, _)| *occ).collect();
    for &occ in subject
        .dose_occasions
        .iter()
        .chain(&subject.pk_only_occasions)
        .chain(&subject.reset_occasions)
    {
        if seen.insert(occ) {
            groups.push((occ, Vec::new()));
        }
    }
    groups
}

/// Whether any scored observation can read each [`iov_occasion_groups`] entry's κ
/// (#1911). A group owning an observation is informed. Otherwise, with
/// `carries_state = false` (`$PRED`-style) it never is; else it is informed when some
/// observation sits at or after its first record (a tie counts — conservative). A
/// reset row counts only when `reset_reseeds` (its κ reaches predictions only through
/// `init(...)`, #1153). An uninformed κ̂ is the prior mode 0 and leaves the objective
/// unchanged.
pub(crate) fn iov_group_informed(
    subject: &Subject, carries_state: bool, reset_reseeds: bool,
) -> Vec<bool> {
    let records = subject
        .doses
        .iter()
        .map(|d| d.time)
        .zip(&subject.dose_occasions)
        .chain(subject.pk_only_times.iter().copied().zip(&subject.pk_only_occasions))
        .chain(
            subject
                .reset_times
                .iter()
                .copied()
                .zip(&subject.reset_occasions)
                .filter(|_| reset_reseeds),
        );
    iov_occasion_groups(subject)
        .into_iter()
        .map(|(occ, obs_idx)| {
            if !obs_idx.is_empty() {
                return true;
            }
            if !carries_state {
                return false;
            }
            let mut times = records.clone().filter(|&(_, &o)| o == occ).map(|(t, _)| t);
            // No record that can act: the group was built only from resets that re-seed
            // nothing.
            let Some(t0) = times.next() else {
                return false;
            };
            let first = times.fold(t0, f64::min);
            // A non-finite time: keep it, the historical behaviour.
            !first.is_finite() || subject.obs_times.iter().any(|&t| t >= first)
        })
        .collect()
}

/// [`iov_group_informed`] with both flags read off `model`: state is carried unless the
/// model is compartment-free, and a reset re-seeds only on the ODE engine with an
/// `init(...)` (`OdeSpec::initial_state`; the closed-form and algebraic arms of
/// `predict_iov` build no reset snapshot at all). The one place postfit asks.
pub(crate) fn iov_group_informed_for(model: &CompiledModel, subject: &Subject) -> Vec<bool> {
    let reset_reseeds = model.ode_spec.as_ref().is_some_and(|ode| ode.init_fn.is_some());
    iov_group_informed(subject, !model.is_algebraic(), reset_reseeds)
}

/// Map each occasion id to its group index `k` (the order [`iov_occasion_groups`]
/// returns). Shared by every IOV path through `StackedLayout` (`stats/stacked_layout.rs`),
/// which the closed-form and ODE recorders both read, so the occasion→group mapping is defined once
/// (#466 review round 2).
pub fn iov_occ_to_k(occ_groups: &[(u32, Vec<usize>)]) -> std::collections::HashMap<u32, usize> {
    occ_groups.iter().enumerate().map(|(k, (occ, _))| (*occ, k)).collect()
}

/// `[η_bsv, κ_g]` for occasion group `g`, sliced from the stacked
/// `[η_bsv, κ₁..κ_K]`. The offset is owned by `stats::stacked_layout` (`kappa_offset`,
/// #1082) so closed-form and ODE IOV providers agree.
pub fn iov_combined_effect(
    stacked_eta: &[f64], n_eta: usize, n_kappa: usize, g: usize,
) -> Vec<f64> {
    let mut c = Vec::with_capacity(n_eta + n_kappa);
    c.extend_from_slice(&stacked_eta[..n_eta]);
    let base = crate::stats::stacked_layout::kappa_offset(n_eta, n_kappa, g);
    c.extend_from_slice(&stacked_eta[base..base + n_kappa]);
    c
}

/// `[η_bsv, 0…0]` — the κ=0 counterpart of [`iov_combined_effect`] for an event with
/// no resolvable occasion (matching `predict_iov`'s `combined_for(u32::MAX)` miss).
pub fn iov_combined_pk_only(stacked_eta: &[f64], n_eta: usize, n_kappa: usize) -> Vec<f64> {
    let mut c = Vec::with_capacity(n_eta + n_kappa);
    c.extend_from_slice(&stacked_eta[..n_eta]);
    c.extend(std::iter::repeat(0.0).take(n_kappa));
    c
}

/// Build a block-diagonal omega from BSV omega and K copies of IOV omega (a test oracle
/// for the stacked `[η | κ_1 … κ_K]` covariance).
#[cfg(test)]
pub(crate) fn build_block_diag_omega(
    omega_bsv: &DMatrix<f64>, omega_iov: &DMatrix<f64>, n_occasions: usize,
) -> DMatrix<f64> {
    crate::stats::stacked_layout::block_diag_stack(omega_bsv, omega_iov, n_occasions)
}

/// IOV-aware individual NLL with per-occasion kappas (`kappas[k]` in
/// `iov_occasion_groups` order; empty falls back to [`individual_nll`]).
///
/// Predictions come from [`pk::predict_iov`], which carries amounts continuously
/// across occasions, so an earlier dose decays with the later occasion's clearance as
/// in NONMEM (#104). The FD Jacobian and `foce_subject_nll_iov` use the same prediction.
pub fn individual_nll_iov(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], kappas: &[Vec<f64>],
    omega: &OmegaMatrix, omega_iov: Option<&OmegaMatrix>, sigma_values: &[f64],
) -> f64 {
    individual_nll_iov_with_scratch(
        model,
        subject,
        theta,
        eta,
        kappas,
        omega,
        omega_iov,
        sigma_values,
        &mut pk::EventPkParams::default(),
    )
}

/// Inner-loop sibling that retains per-event PK buffer capacity across trial
/// eta/kappa values. The public convenience function keeps its original API.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_iov_with_scratch<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], kappas: &[K],
    omega: &OmegaMatrix, omega_iov: Option<&OmegaMatrix>, sigma_values: &[f64],
    pk_scratch: &mut pk::EventPkParams,
) -> f64 {
    individual_nll_iov_with_scratch_and_schedule(
        model,
        subject,
        theta,
        eta,
        kappas,
        omega,
        omega_iov,
        sigma_values,
        pk_scratch,
        None,
    )
}

/// [`individual_nll_iov_with_scratch`] with the subject's cached
/// [`EventSchedule`](pk::event_driven::EventSchedule) (#1452). `None` reproduces the
/// uncached behaviour.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_iov_with_scratch_and_schedule<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], kappas: &[K],
    omega: &OmegaMatrix, omega_iov: Option<&OmegaMatrix>, sigma_values: &[f64],
    pk_scratch: &mut pk::EventPkParams, schedule: Option<&pk::event_driven::EventSchedule>,
) -> f64 {
    individual_nll_iov_frozen(
        model,
        subject,
        theta,
        eta,
        kappas,
        omega,
        omega_iov,
        sigma_values,
        pk_scratch,
        schedule,
        None,
    )
}

/// [`individual_nll_iov_with_scratch_and_schedule`] with an optional frozen residual
/// variance — IOV twin of [`individual_nll_into_prepared_with_schedule_frozen`] (#1722);
/// `None` is bit-identical to the conditional objective.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_iov_frozen<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64], kappas: &[K],
    omega: &OmegaMatrix, omega_iov: Option<&OmegaMatrix>, sigma_values: &[f64],
    pk_scratch: &mut pk::EventPkParams, schedule: Option<&pk::event_driven::EventSchedule>,
    frozen_var_preds: Option<&[f64]>,
) -> f64 {
    // #1082: a nested-level subject never given its cells has no layout, and is refused
    // rather than scored as one-level IOV or with κ dropped.
    let Some(layout) = crate::stats::stacked_layout::StackedLayout::sens_of_cached(model, subject)
    else {
        return SUBJECT_SENTINEL_NLL;
    };
    individual_nll_iov_frozen_in(
        model,
        subject,
        &layout,
        theta,
        eta,
        kappas,
        omega,
        omega_iov,
        sigma_values,
        model.residual_cov_theta(theta),
        pk_scratch,
        schedule,
        frozen_var_preds,
    )
}

/// `vᵀ·M·v` with `M·v` formed by nalgebra's own matrix–vector kernel into a pooled buffer,
/// then nalgebra's `dot`: the same kernels, in the same order, as
/// `DVector::from_column_slice(v).dot(&(m * &DVector::from_column_slice(v)))`, without the
/// two vectors that expression allocates per call (#2549 W1). Pinned bit for bit by
/// `quad_form_is_the_allocating_expression_bit_for_bit`.
pub(crate) fn quad_form(m: &DMatrix<f64>, v: &[f64]) -> f64 {
    let n = v.len();
    let mut tmp = crate::scratch::take_with_capacity(n);
    tmp.resize(n, 0.0);
    let vv = nalgebra::DVectorView::from_slice(v, n);
    let r = {
        let mut tv = nalgebra::DVectorViewMut::from_slice(&mut tmp, n);
        m.mul_to(&vv, &mut tv);
        vv.dot(&tv)
    };
    crate::scratch::give(tmp);
    r
}

/// [`individual_nll_iov_frozen`] over a caller-built `layout`
/// ([`StackedLayout::sens_of`](crate::stats::stacked_layout::StackedLayout::sens_of)),
/// built once per EBE search (#2037). `residual_cov`: live `block_sigma` ρ for the
/// FOCE/FOCEI EBE search (#979), the declaration for every other caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn individual_nll_iov_frozen_in<K: AsRef<[f64]>>(
    model: &CompiledModel, subject: &Subject, layout: &crate::stats::stacked_layout::StackedLayout,
    theta: &[f64], eta: &[f64], kappas: &[K], omega: &OmegaMatrix, omega_iov: Option<&OmegaMatrix>,
    sigma_values: &[f64], residual_cov: crate::stats::residual_error::ResidualCov<'_>,
    pk_scratch: &mut pk::EventPkParams, schedule: Option<&pk::event_driven::EventSchedule>,
    frozen_var_preds: Option<&[f64]>,
) -> f64 {
    let nested = layout.nested();
    if kappas.is_empty() {
        return individual_nll(model, subject, theta, eta, omega, sigma_values);
    }

    // BSV eta prior
    let omega_inv = &omega.inv;
    let log_det_omega = omega.log_det;
    let mut eta_prior = quad_form(omega_inv, eta);
    // #1716: the BSV η's density correction, as in the BSV objective.
    if !model.eta_densities.is_empty() {
        eta_prior +=
            crate::stats::eta_density::correction(&model.eta_densities, eta, omega_inv, theta);
        if !eta_prior.is_finite() {
            return SUBJECT_SENTINEL_NLL;
        }
    }

    // Kappa priors and IOV log-det
    let Some(iov) = omega_iov else {
        // Guaranteed unreachable: the only caller path that admits `omega_iov: None` is
        // `kappas.is_empty()`, which returns above before this point. `1e20` is a defensive
        // sentinel, not a real score — loud in debug/test builds so a future caller that
        // violates the invariant fails here rather than silently scoring a fabricated NLL.
        debug_assert!(
            false,
            "individual_nll_iov_with_scratch: omega_iov is None with kappas non-empty"
        );
        return SUBJECT_SENTINEL_NLL;
    };
    let (iov_inv, log_det_iov) = (&iov.inv, iov.log_det);

    // `Σ_k κ_kᵀΩ_IOV⁻¹κ_k + K·log|Ω_IOV|` over the K occasion groups — or, under nested
    // levels (#1082), the same form over the groups of each level, where the cells of one
    // group share its κ and count it once (`NestedLayout::kappa_prior`).
    // The two terms stay separate summands so the one-level sum keeps its association.
    let (kappa_prior, kappa_log_det) = match nested {
        Some(nested) => match nested.compact(eta, kappas) {
            Some(b) => {
                let mut kp = nested.kappa_prior(&b, iov_inv, &iov.chol);
                // #1716: the κ density groups in each level block of the compact vector.
                if !model.kappa_densities.is_empty() {
                    kp += crate::stats::eta_density::nested_kappa_correction(
                        model,
                        &nested.kappa_blocks(),
                        &b,
                        iov_inv,
                        theta,
                    );
                    if !kp.is_finite() {
                        return SUBJECT_SENTINEL_NLL;
                    }
                }
                (kp, 0.0)
            }
            None => return SUBJECT_SENTINEL_NLL,
        },
        None => {
            let mut kappa_prior = 0.0;
            for kap in kappas {
                kappa_prior += quad_form(iov_inv, kap.as_ref());
                // #1716: an occasion's κ under a density group.
                if !model.kappa_densities.is_empty() {
                    kappa_prior += crate::stats::eta_density::correction(
                        &model.kappa_densities,
                        kap.as_ref(),
                        iov_inv,
                        theta,
                    );
                }
            }
            if !kappa_prior.is_finite() {
                return SUBJECT_SENTINEL_NLL;
            }
            let k_occasions = kappas.len();
            (kappa_prior, (k_occasions as f64) * log_det_iov)
        }
    };

    // Data NLL — single continuous prediction with per-event occasion kappa
    // (proper cross-occasion carryover; issue #104).
    let mut stacked = crate::scratch::take_with_capacity(layout.n_stacked());
    stacked.extend_from_slice(eta);
    for k in kappas {
        stacked.extend_from_slice(k.as_ref());
    }
    let mut planned = Vec::new();
    let preds = if nested.is_none()
        && plan_value_preds(model, subject, Some(layout), theta, &stacked, &mut planned)
    {
        planned
    } else {
        pk::predict_iov_in(model, subject, layout, theta, eta, kappas, pk_scratch, schedule)
    };
    crate::scratch::give(stacked);
    // FREM covariate pseudo-observations use the covariate sigma (EPSCOV), not
    // the PK residual error, so the FREM etas are sampled against the right
    // variance (mirrors the FOCE paths and the non-IOV individual_nll).
    let frem_ov =
        build_frem_r_override(model.frem_config.as_ref(), &subject.fremtype, sigma_values);
    // IIV on residual error (#409): η_ruv is a BSV eta, indexed into `eta`. A frozen
    // variance is `R⁰`, taken at every random effect zero (see the doc above).
    let ruv_scale = if frozen_var_preds.is_some() { 1.0 } else { model.residual_var_scale(eta) };
    // #484/#1029: per-observation residual-magnitude multiplier, so the IOV
    // individual NLL (SAEM's E-step, the Bayes MH target, the IOV IS weights)
    // scores the same variance the non-IOV path and FOCE/FOCEI do.
    let ruv_mult = model.ruv_obs_mult(subject, theta);
    // #658: per-observation residual endpoint keys (covariate selector or CMT).
    let err_keys = model.error_spec.obs_keys(subject);
    // An SDE's EKF process-noise variance, on the BSV η (the filter reads one parameter set,
    // κ = 0): the same `V = V_resid + p_obs` the IOV marginal (`foce_subject_nll_iov`) scores,
    // so the EBE minimises the objective the outer loop differentiates.
    let p_obs = if model.is_sde() {
        ekf_p_obs(model, subject, theta, eta, sigma_values)
    } else {
        Vec::new()
    };
    let mut data_ll = 0.0;
    // A dense `R` (a `correlation … by` line, #1082 PR5, or a `block_sigma`
    // off-diagonal, #979) scores the whole subject with the non-IOV objective's data
    // term, on the same predicate: `R` at `f₀` for a FOCE EBE search
    // (`frozen_var_preds`), at the conditional predictions otherwise.
    let has_censored_m3 = model.bloq_in_force().has_censored_row(&subject.cens);
    let has_frem_rows = subject.fremtype.iter().any(|&ft| ft > 0);
    if residual_cov.is_dense() && !has_censored_m3 && !has_frem_rows {
        match dense_residual_data_term(
            model,
            subject,
            &preds,
            frozen_var_preds.unwrap_or(&preds[..]),
            sigma_values,
            residual_cov,
            ruv_scale,
            &p_obs,
            ruv_mult.as_deref(),
        ) {
            Some(term) => data_ll += term,
            None => return SUBJECT_SENTINEL_NLL,
        }
    } else {
        for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
            let v = match frem_ov.as_ref().and_then(|o| o.get(j)).and_then(|x| *x) {
                Some(vv) => vv,
                None => {
                    model.residual_variance_at_scaled(
                        err_keys[j],
                        frozen_var_preds.map_or(f_pred, |f0| f0[j]),
                        sigma_values,
                        ruv_mult.as_ref().map(|m| m[j].as_slice()),
                    ) * ruv_scale
                }
            };
            let v = v + p_obs.get(j).copied().unwrap_or(0.0);
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if model.bloq_in_force().is_censored_row(cens) {
                data_ll += -2.0 * m3_logcdf(y, f_pred, v.sqrt(), cens);
            } else {
                let resid = y - f_pred;
                data_ll += resid * resid / v + v.ln();
            }
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

    0.5 * (eta_prior + log_det_omega + kappa_prior + kappa_log_det + data_ll)
}

#[cfg(all(test, feature = "survival"))]
#[path = "non_gaussian_eta_hessian_tests.rs"]
mod non_gaussian_eta_hessian_tests;

#[cfg(all(test, feature = "survival"))]
#[path = "endpoint_order_tests.rs"]
mod endpoint_order_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The IOV marginal scores non-Gaussian records: with κ on a parameter no record reads
    /// (κ̂ = 0) it equals the NONMEM-anchored non-IOV marginal, for TTE and binary, under
    /// both interaction settings.
    #[cfg(feature = "survival")]
    #[test]
    fn the_iov_marginal_scores_the_non_gaussian_term() {
        use crate::types::{DoseEvent, EventType, ObsRecord};
        let root = env!("CARGO_MANIFEST_DIR");
        let tte = std::fs::read_to_string(format!("{root}/examples/tte_exponential.ferx")).unwrap();
        let tte_iov = tte
            .replace(
                "  omega ETA_LAMBDA ~ 0.09\n",
                "  omega ETA_LAMBDA ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n",
            )
            .replace("CL     = DUMMY_CL", "CL     = DUMMY_CL * exp(KAPPA_CL)")
            .replace("  method  = focei\n", "  method  = focei\n  iov_column = OCC\n");
        let binary =
            "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
                      theta TH0(0.3, -10.0, 10.0)\n  omega ETA_I ~ 0.3\n  sigma PROP ~ 0.05\n\
                      [individual_parameters]\n  CL = TVCL\n  V  = TVV\n\
                      [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
                      [error_model]\n  DV ~ proportional(PROP)\n\
                      [binary_model]\n  cmt   = 3\n  logit = TH0 + ETA_I\n\
                      [fit_options]\n  method = focei\n"
                .to_string();
        let binary_iov = binary
            .replace("  omega ETA_I ~ 0.3\n", "  omega ETA_I ~ 0.3\n  kappa KAPPA_CL ~ 0.04\n")
            .replace("CL = TVCL\n", "CL = TVCL * exp(KAPPA_CL)\n")
            .replace("  method = focei\n", "  method = focei\n  iov_column = OCC\n");
        let event = |time: f64, event_type: EventType, entry_time: f64| ObsRecord::Event {
            time,
            event_type,
            entry_time,
            cmt: 2,
        };
        let state = |time: f64, state: usize| ObsRecord::DiscreteState {
            time,
            raw_time: time,
            state,
            cmt: 3,
        };
        let cases = [
            (
                "tte",
                tte,
                tte_iov,
                vec![event(7.5, EventType::Exact, 0.0), event(30.0, EventType::RightCensored, 7.5)],
            ),
            (
                "binary",
                binary,
                binary_iov,
                vec![state(1.0, 1), state(5.0, 0), state(13.0, 1), state(20.0, 1)],
            ),
        ];
        for (tag, plain_src, iov_src, records) in cases {
            let plain = crate::parser::model_parser::parse_model_string(&plain_src).unwrap();
            let iov = crate::parser::model_parser::parse_model_string(&iov_src).unwrap();
            assert_eq!(iov.n_kappa, 1, "{tag}");
            // Two dose-only occasions on the (unobserved) PK compartment give two κ groups.
            let s = Subject {
                id: "1".into(),
                doses: vec![
                    DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(12.0, 10.0, 1, 0.0, false, 0.0),
                ],
                dose_occasions: vec![1, 2],
                obs_records: records,
                ..Default::default()
            };
            assert_eq!(crate::stats::stacked_layout::group_count(&s), 2);
            let theta = &plain.default_params.theta;
            let eta = DVector::from_vec(vec![0.12]);
            let h = DMatrix::<f64>::zeros(0, 1);
            let kappas = vec![DVector::zeros(1), DVector::zeros(1)];
            let no_records = Subject { obs_records: Vec::new(), ..s.clone() };
            let iov_nll = |subject: &Subject, interaction: bool| {
                foce_subject_nll_iov(
                    &iov,
                    subject,
                    theta,
                    &eta,
                    &h,
                    &iov.default_params.omega,
                    &iov.default_params.sigma.values,
                    interaction,
                    &kappas,
                    iov.default_params.omega_iov.as_ref().unwrap(),
                )
            };
            for interaction in [true, false] {
                let want = foce_subject_nll_cov(
                    &plain,
                    &s,
                    theta,
                    &eta,
                    &h,
                    &plain.default_params.omega,
                    &plain.default_params.sigma.values,
                    plain.residual_cov_with(&[], theta),
                    interaction,
                );
                let got = iov_nll(&s, interaction);
                let dropped = iov_nll(&no_records, interaction);
                assert!(want.is_finite() && got.is_finite(), "{tag}: {want} {got}");
                assert!(
                    (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                    "{tag} interaction={interaction}: IOV {got} vs non-IOV {want}"
                );
                // The term is live: without it the objective is a different number.
                assert!((dropped - want).abs() > 1e-2, "{tag}: {dropped} vs {want}");
            }
        }
    }

    /// Under nested levels the IOV linearisation's compact κ columns are exact (the
    /// per-cell recording over the compact axes), replacing two solves per compact κ:
    /// against the forced central difference they agree to the quotient's accuracy, and
    /// they differ in the bits (the exact branch ran). The η columns are the caller's.
    #[test]
    fn nested_levels_linearize_with_exact_kappa_columns() {
        let (model, pop, _) =
            crate::estimation::inner_optimizer::test_helpers::nested_levels_fixture();
        assert!(model.has_nested_levels());
        let params = &model.default_params;
        let omega_iov = params.omega_iov.as_ref().expect("an Ω_IOV");
        let mut worst = 0.0f64;
        for s in pop.subjects.iter().take(3) {
            let n_eta = model.n_eta;
            let layout = crate::stats::stacked_layout::StackedLayout::new(n_eta, model.n_kappa, s);
            let kappas: Vec<DVector<f64>> = (0..layout.n_groups())
                .map(|g| DVector::from_element(model.n_kappa, 0.01 * (g as f64 + 1.0)))
                .collect();
            let eta = DVector::from_element(n_eta, 0.04);
            let h = DMatrix::from_element(s.obs_times.len(), n_eta, 0.2);
            let run = |force: bool| {
                NESTED_KAPPA_FD_FORCE.with(|c| c.set(force));
                let l = iov_linearization(
                    &model,
                    s,
                    &params.theta,
                    &eta,
                    &h,
                    &params.omega,
                    &kappas,
                    omega_iov,
                )
                .expect("linearisation");
                NESTED_KAPPA_FD_FORCE.with(|c| c.set(false));
                l.h_full
            };
            let (he, hf) = (run(false), run(true));
            assert_eq!(he.ncols(), hf.ncols());
            assert!(
                he.iter().zip(hf.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
                "the exact κ branch did not run"
            );
            let scale = hf.iter().fold(1.0f64, |m, &v| m.max(v.abs()));
            for (a, b) in he.iter().zip(hf.iter()) {
                assert!(a.is_finite() && b.is_finite());
                worst = worst.max((a - b).abs() / scale);
            }
        }
        eprintln!("exact nested κ columns vs the central difference: worst {worst:.2e}");
        // Measured 2.2e-10: the quotient's truncation at its `1e-6` step. A column read in
        // the wrong compact order misses by O(1).
        assert!(worst < 1e-6, "{worst:.2e}");
    }
    use crate::types::{
        BloqMethod, DoseEvent, EndpointError, ErrorModel, ErrorSpec, GradientMethod, PkParams,
        ResidualCorrelation,
    };
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn make_simple_subject() -> Subject {
        Subject {
            id: "1".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            observations: vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0],
            obs_cmts: vec![1; 6],
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            ..Default::default()
        }
    }

    fn make_omega(var: f64) -> OmegaMatrix {
        OmegaMatrix::from_diagonal(&[var], vec!["ETA_CL".into()])
    }

    fn make_model() -> CompiledModel {
        CompiledModel {
            pk_param_fn: Box::new(
                |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp(); // CL uses combined eta[0]
                    p.values[1] = theta[1]; // V
                    p
                },
            ),
            gradient_method: GradientMethod::default(),
            ..crate::types::test_helpers::bare_model(crate::types::ModelParameters {
                theta_lower: vec![0.01, 1.0],
                theta_upper: vec![100.0, 500.0],
                sigma: crate::types::SigmaVector {
                    values: vec![0.05],
                    names: vec!["PROP_ERR".into()],
                },
                ..crate::types::test_helpers::bare_params(make_omega(0.09))
            })
        }
    }

    #[test]
    fn obs_nll_subject_into_uses_cross_endpoint_residual_covariance() {
        let mut model = make_model();
        model.error_spec = ErrorSpec::PerCmt(HashMap::from([
            (1, EndpointError { error_model: ErrorModel::Proportional, sigma_idx: vec![1] }),
            (2, EndpointError { error_model: ErrorModel::Proportional, sigma_idx: vec![0] }),
        ]));
        model.residual_correlations =
            vec![ResidualCorrelation { sigma_i: 0, sigma_j: 1, rho: 0.5 }];

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
        let dense = obs_nll_subject_into(
            &model,
            &subj,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&corr),
            &eta,
            &mut scratch,
        );
        model.residual_correlations.clear();
        let diagonal = obs_nll_subject_into(
            &model,
            &subj,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&[]),
            &eta,
            &mut scratch,
        );

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

    /// #1496, `m3_logcdf` (`cens < 0`): an out-of-domain `CENS` scores by its sign (`7` as
    /// `1`, `-2` as `-1`); the two tails differ here (z = ∓1).
    #[test]
    fn m3_logcdf_scores_an_out_of_domain_flag_by_its_sign() {
        let at = |cens: i8| m3_logcdf(10.0, 12.0, 2.0, cens);
        assert!(at(1).is_finite() && at(-1).is_finite());
        assert_ne!(at(1).to_bits(), at(-1).to_bits(), "the two tails differ here");
        for (flag, code) in [(7, 1), (i8::MAX, 1), (-2, -1), (i8::MIN, -1)] {
            assert_eq!(
                at(flag).to_bits(),
                at(code).to_bits(),
                "m3_logcdf: CENS={flag} must score as CENS={code}"
            );
        }
    }

    /// #2762: an M3 limit so far on the certain side that `|z| ≈ 1.9e154` (`normal_cdf`
    /// overflows) scored NaN (`warfarin_bloq.ferx`, `LLOQ = 1e300` → 2e20 sentinel). Its term
    /// is `0`; through the conditional, FOCE and FOCEI objectives the far limit must score
    /// bit-for-bit as a `±1e6` one.
    #[test]
    fn a_censored_limit_past_the_normal_cdf_overflow_scores_as_certain() {
        let theta = [5.0, 50.0];
        let eta = [0.1];
        let omega = make_omega(0.09);
        let sigma = [0.2];
        // Straddle, on the fixture row's own z (review r1 #7): the near limit stays below
        // the gate (`(z/√2)²` finite) and the far one reaches it (`(z/√2)²` overflows).
        {
            let model = make_model();
            let subject = make_simple_subject();
            let f = model_predictions(&model, &subject, &theta, &eta)[5];
            let key = model.error_spec.obs_key(&subject, 5);
            let sd = model.error_spec.variance_at(key, f, &sigma).sqrt();
            let sq = |limit: f64| ((limit - f) / sd / std::f64::consts::SQRT_2).powi(2);
            for (far, near) in [(1e300, 1e6), (-1e300, -1e6)] {
                assert!(sq(near).is_finite(), "near {near:e}: f {f}, sd {sd}");
                assert_eq!(sq(far), f64::INFINITY, "far {far:e}: f {f}, sd {sd}");
            }
        }
        let setup = |limit: f64, flag: i8| {
            let mut model = make_model();
            model.bloq_method = BloqMethod::M3;
            let mut subject = make_simple_subject();
            subject.observations[5] = limit;
            subject.cens[5] = flag;
            (model, subject)
        };
        let conditional = |limit: f64, flag: i8| {
            let (model, subject) = setup(limit, flag);
            individual_nll(&model, &subject, &theta, &eta, &omega, &sigma)
        };
        let marginal = |limit: f64, flag: i8, interaction: bool| {
            let (model, subject) = setup(limit, flag);
            let eta_hat = DVector::from_vec(eta.to_vec());
            let h = DMatrix::from_element(subject.obs_times.len(), 1, 4.0);
            foce_subject_nll(
                &model,
                &subject,
                &theta,
                &eta_hat,
                &h,
                &omega,
                &sigma,
                &[],
                interaction,
            )
        };
        let objectives: [(&str, &dyn Fn(f64, i8) -> f64); 3] = [
            ("individual_nll", &conditional),
            ("FOCE", &|l: f64, c: i8| marginal(l, c, false)),
            ("FOCEI", &|l: f64, c: i8| marginal(l, c, true)),
        ];
        for (name, nll) in objectives {
            for (far, near, flag) in [(1e300, 1e6, 1), (-1e300, -1e6, -1)] {
                let (got, want) = (nll(far, flag), nll(near, flag));
                assert!(want.is_finite(), "{name}: control at {near:e}, CENS={flag}");
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{name}: limit {far:e} (CENS={flag}) scored {got}, the certain row {want}"
                );
            }
        }
    }

    /// #1496: `W_CENS_UNEXPECTED`'s two claims checked against the objective: under `m3` an
    /// out-of-domain flag scores as its sign's code, under `drop` as a quantified row —
    /// through `individual_nll` and the FOCE/FOCEI marginals; `1`, `-1`, `0` all differ.
    #[test]
    fn an_out_of_domain_cens_flag_scores_by_sign_under_m3_and_as_quantified_under_drop() {
        let theta = [5.0, 50.0];
        let eta = [0.1];
        let omega = make_omega(0.09);
        let sigma = [0.2];
        let setup = |bloq: BloqMethod, flag: i8| {
            let mut model = make_model();
            model.bloq_method = bloq;
            let mut subject = make_simple_subject();
            // The last row, DV 25 against a prediction near 30: both tails are
            // live and differ from the quantified term.
            subject.cens[5] = flag;
            (model, subject)
        };
        let conditional = |bloq: BloqMethod, flag: i8| {
            let (model, subject) = setup(bloq, flag);
            individual_nll(&model, &subject, &theta, &eta, &omega, &sigma)
        };
        let marginal = |bloq: BloqMethod, flag: i8, interaction: bool| {
            let (model, subject) = setup(bloq, flag);
            let eta_hat = DVector::from_vec(eta.to_vec());
            let h = DMatrix::from_element(subject.obs_times.len(), 1, 4.0);
            foce_subject_nll(
                &model,
                &subject,
                &theta,
                &eta_hat,
                &h,
                &omega,
                &sigma,
                &[],
                interaction,
            )
        };
        let objectives: [(&str, &dyn Fn(BloqMethod, i8) -> f64); 3] = [
            ("individual_nll", &conditional),
            ("FOCE", &|b: BloqMethod, c: i8| marginal(b, c, false)),
            ("FOCEI", &|b: BloqMethod, c: i8| marginal(b, c, true)),
        ];
        for (name, nll) in objectives {
            let m3 = |flag: i8| nll(BloqMethod::M3, flag);
            let drop = |flag: i8| nll(BloqMethod::Drop, flag);
            for flag in [0, 1, -1] {
                assert!(m3(flag).is_finite(), "{name}: m3, CENS={flag}");
                assert!(drop(flag).is_finite(), "{name}: drop, CENS={flag}");
            }
            assert_ne!(m3(1).to_bits(), m3(-1).to_bits(), "{name}: the tails differ");
            assert_ne!(m3(1).to_bits(), m3(0).to_bits(), "{name}: the lower tail is live");
            assert_ne!(m3(-1).to_bits(), m3(0).to_bits(), "{name}: the upper tail is live");
            // "under bloq_method = m3 a positive flag is scored on the lower tail, like
            // CENS=1" — and the negative one on the upper tail, like CENS=-1.
            assert_eq!(m3(7).to_bits(), m3(1).to_bits(), "{name}: m3, 7 as 1");
            assert_eq!(m3(-2).to_bits(), m3(-1).to_bits(), "{name}: m3, -2 as -1");
            // "under bloq_method = drop the row is scored as an ordinary observation".
            for flag in [7, -2, 1, -1] {
                assert_eq!(
                    drop(flag).to_bits(),
                    drop(0).to_bits(),
                    "{name}: drop, CENS={flag} scores as the quantified row"
                );
            }
        }
    }

    #[test]
    fn prepared_inner_nll_reuse_is_bit_identical() {
        let model = make_model();
        let subject = make_simple_subject();
        let theta = [5.0, 50.0];
        let eta = [0.17];
        let omega = make_omega(0.09);
        let sigma = [0.05];
        let expected = individual_nll_into_with_schedule(
            &model,
            &subject,
            &theta,
            &eta,
            &omega,
            &sigma,
            &[],
            &mut pk::EventPkParams::default(),
            None,
        );

        let err_keys = model.error_spec.obs_keys(&subject);
        let mult = model.ruv_obs_mult(&subject, &theta);
        let mut event_scratch = pk::EventPkParams::default();
        let mut predictions = vec![-999.0; subject.obs_times.len()];
        let mut eta_work = DVector::from_element(model.n_eta, -999.0);
        let mut prior_work = DVector::from_element(model.n_eta, -999.0);
        let reused = individual_nll_into_prepared_with_schedule(
            &model,
            &subject,
            &theta,
            &eta,
            &omega,
            &sigma,
            ResidualCov::block_sigma(&[]),
            &mut event_scratch,
            None,
            err_keys.as_ref(),
            mult.as_deref(),
            &mut predictions,
            &mut eta_work,
            &mut prior_work,
        );

        assert_eq!(expected.to_bits(), reused.to_bits());
        assert_eq!(predictions.len(), subject.obs_times.len());
        assert!(predictions.iter().all(|v| v.is_finite()));
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

    /// #1029 degenerate oracle: `weight = W` reproduces the hand-built study-as-subject
    /// construction (`DV/W`, `PRED/W`, `sigma FIX 1`) up to the data-only Jacobian
    /// `Σ_j ln W_j` of `y ↦ y/W`, which cancels from every ΔOFV.
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
        natural.obs_covariates =
            w.iter().map(|&wi| [("WPSE".to_string(), wi)].into_iter().collect()).collect();

        // The hand-built twin: DV and the prediction both pre-divided by the
        // weight, scored by an unweighted additive model on the same sigma.
        let mut prescaled = natural.clone();
        prescaled.observations = y.iter().zip(w).map(|(&yi, wi)| yi / wi).collect();
        prescaled.obs_covariates = Vec::new();
        let scaled_preds: Vec<f64> = f.iter().zip(w).map(|(&fi, wi)| fi / wi).collect();

        let theta = [0.2, 10.0];
        let sigma = [0.8];
        let eta = [0.0];

        let nll_weighted = obs_nll_subject_from_preds(
            &weighted,
            &natural,
            &f,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&[]),
            &eta,
        );
        let nll_hand = obs_nll_subject_from_preds(
            &plain,
            &prescaled,
            &scaled_preds,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&[]),
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

    /// An occasion only pk-only rows carry gets a group too, after the dose-only ones and
    /// in file order; one an observation or a dose already carries does not get a second.
    #[test]
    fn test_iov_occasion_groups_appends_pk_only_occ() {
        let mut subj = make_simple_subject();
        subj.dose_occasions = vec![1, 3];
        subj.pk_only_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        subj.pk_only_occasions = vec![6, 2, 5, 3, 6, 1];
        let groups = iov_occasion_groups(&subj);
        let ids: Vec<u32> = groups.iter().map(|(occ, _)| *occ).collect();
        assert_eq!(ids, vec![1, 2, 3, 6, 5]);
        assert_eq!(groups[3], (6, Vec::new()));
        assert_eq!(groups[4], (5, Vec::new()));
    }

    /// #1153: an occasion only reset rows carry gets a group too, after the pk-only ones
    /// and in file order; one any other record already carries does not get a second.
    #[test]
    fn test_iov_occasion_groups_appends_reset_only_occ() {
        let mut subj = make_simple_subject();
        subj.dose_occasions = vec![1, 3];
        subj.pk_only_times = vec![1.0];
        subj.pk_only_occasions = vec![6];
        subj.reset_times = vec![1.5, 2.5, 3.5, 4.5];
        subj.reset_occasions = vec![9, 6, 8, 9];
        let groups = iov_occasion_groups(&subj);
        let ids: Vec<u32> = groups.iter().map(|(occ, _)| *occ).collect();
        assert_eq!(ids, vec![1, 2, 3, 6, 9, 8]);
        assert_eq!(groups[4], (9, Vec::new()));
        assert_eq!(groups[5], (8, Vec::new()));
    }

    /// #1153: a reset-only group is informed by the same first-record rule — the reset at
    /// t = 4 has observations after it (t = 4..6), the one at t = 20 has none — and never
    /// under `carries_state = false`. A reset at t = 0.5 labelled 2 changes nothing: OCC 2
    /// owns observations.
    #[test]
    fn test_iov_group_informed_reads_reset_rows() {
        let mut subj = make_simple_subject();
        subj.reset_times = vec![0.5, 4.0, 20.0];
        subj.reset_occasions = vec![2, 8, 9];
        let ids: Vec<u32> = iov_occasion_groups(&subj).iter().map(|g| g.0).collect();
        assert_eq!(ids, vec![1, 2, 8, 9]);
        assert_eq!(iov_group_informed(&subj, true, true), vec![true, true, true, false]);
        assert_eq!(iov_group_informed(&subj, false, true), vec![true, true, false, false]);
        // A reset that re-seeds nothing informs nothing, at any time (review r1 #1).
        assert_eq!(iov_group_informed(&subj, true, false), vec![true, true, false, false]);
    }

    /// #1153 review r1 #1: `iov_group_informed_for` lets a reset inform its group only on an
    /// ODE model with an `init(...)`. A closed-form model's reset reads no parameters and an
    /// init-free ODE model's reset zeros the state, so κ of a reset-only occasion moves no
    /// prediction there and must stay out of the κ shrinkage.
    #[test]
    fn test_iov_group_informed_for_reads_the_reset_reseed() {
        let ode = |init: &str| {
            format!(
                "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
                 omega ETA_CL ~ 0.09\n  kappa KAPPA ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n\
                 [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA)\n  V = TVV\n\
                 [structural_model]\n  ode(states=[central])\n\
                 [odes]\n{init}  d/dt(central) = -(CL/V) * central\n\
                 [scaling]\n  y = central / V\n\
                 [error_model]\n  DV ~ proportional(PROP_ERR)\n"
            )
        };
        let closed =
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
             omega ETA_CL ~ 0.09\n  kappa KAPPA ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA)\n  V = TVV\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n";
        let mut subj = make_simple_subject();
        subj.reset_times = vec![4.0];
        subj.reset_occasions = vec![8];
        for (src, want) in
            [(closed.to_string(), false), (ode(""), false), (ode("  init(central) = CL\n"), true)]
        {
            let m = crate::parser::model_parser::parse_model_string(&src).unwrap();
            assert_eq!(
                iov_group_informed_for(&m, &subj),
                vec![true, true, want],
                "ode {} init {}",
                m.ode_spec.is_some(),
                m.ode_spec.as_ref().is_some_and(|o| o.init_fn.is_some())
            );
        }
    }

    /// #1911: a group without observations is informed iff an observation sits at or after
    /// its first record (obs t = 1..6): dose-only at 2 yes, at 10 no; pk-only at 6 (tie)
    /// yes, at 8/12 no; dose 9 + pk-only 1 yes (earliest decides); an own-obs group is
    /// always informed. Under `carries_state = false` only observation groups remain.
    #[test]
    fn test_iov_group_informed_by_first_record_time() {
        let mut subj = make_simple_subject();
        subj.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(9.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(50.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subj.dose_occasions = vec![1, 3, 4, 7, 2];
        subj.pk_only_times = vec![6.0, 12.0, 8.0, 1.0];
        subj.pk_only_occasions = vec![5, 6, 6, 7];
        let ids: Vec<u32> = iov_occasion_groups(&subj).iter().map(|g| g.0).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 7, 5, 6]);
        assert_eq!(
            iov_group_informed(&subj, true, true),
            vec![true, true, true, false, true, true, false]
        );
        assert_eq!(
            iov_group_informed(&subj, false, true),
            vec![true, true, false, false, false, false, false]
        );
    }

    /// Without dose-only or pk-only groups every group owns an observation, so the mask
    /// is all-true and the shrinkage is the historical one (the #1911 control).
    #[test]
    fn test_iov_group_informed_all_true_for_observation_groups() {
        let subj = make_simple_subject();
        for carries_state in [true, false] {
            assert_eq!(iov_group_informed(&subj, carries_state, true), vec![true, true]);
        }
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

    /// The IOV individual NLL (SAEM E-step, Bayes MH, IOV IS weight) applies the
    /// per-observation residual magnitude (#484/#1029) as the non-IOV path does; a weight
    /// varying within the subject makes it bite.
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
        subj.obs_covariates =
            (0..subj.observations.len()).map(|j| snap(0.5 + 0.3 * j as f64)).collect();

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
        // #97: a non-finite eta (standing in for an ODE blow-up) must give the finite
        // sentinel. The analytical path scrubs NaN via `.max()`, so it enters through η'Ω⁻¹η.
        let model = make_model();
        let subj = make_simple_subject();
        let omega = make_omega(0.09);
        let nll = individual_nll(&model, &subj, &[5.0, 50.0], &[f64::INFINITY], &omega, &[0.05]);
        assert!(nll.is_finite(), "NLL must stay finite, got {nll}");
        assert_eq!(nll, SUBJECT_SENTINEL_NLL, "a non-finite NLL should map to the 1e20 sentinel");
    }

    /// #2269: each guard returns exactly `SUBJECT_SENTINEL_NLL`, and its readers
    /// (Gauss–Newton / AGQ usability, `W_NONFINITE_OBJECTIVE` "divergence sentinel" label)
    /// recognise it.
    #[test]
    fn subject_sentinel_guards_and_readers_agree() {
        let model = make_model();
        let subj = make_simple_subject();
        let omega = make_omega(0.09);
        let theta = [5.0, 50.0];
        let sigma = [0.05];

        // Guard 1: `individual_nll`'s non-finite-total fold.
        let ind = individual_nll(&model, &subj, &theta, &[f64::INFINITY], &omega, &sigma);
        // Guards 2 and 3: FOCE's and FOCEI's linearized `R̃` that is not positive
        // definite. A NaN `H` makes `R̃` NaN, which the Cholesky refuses.
        let n_obs = subj.obs_times.len();
        let h_nan = DMatrix::from_element(n_obs, 1, f64::NAN);
        let eta0 = DVector::zeros(1);
        let foce = |interaction| {
            foce_subject_nll(
                &model,
                &subj,
                &theta,
                &eta0,
                &h_nan,
                &omega,
                &sigma,
                &model.residual_correlations,
                interaction,
            )
        };
        // Guard 4: an Ω whose log-determinant is not finite (a singular Ω).
        let mut singular = omega.clone();
        singular.log_det = f64::NAN;
        let singular_omega = individual_nll(&model, &subj, &theta, &[0.0], &singular, &sigma);
        // Guard 5: `chol_log_det` on a factor with a non-positive diagonal.
        let chol = chol_log_det(&DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.5, 0.0]));
        for (guard, got) in [
            ("individual_nll", ind),
            ("FOCE R̃", foce(false)),
            ("FOCEI R̃", foce(true)),
            ("individual_nll singular Ω", singular_omega),
            ("chol_log_det", chol),
        ] {
            assert_eq!(got, SUBJECT_SENTINEL_NLL, "{guard} must return the shared sentinel");
        }

        assert!(
            !crate::stats::likelihood::is_usable_subject_nll(SUBJECT_SENTINEL_NLL),
            "the usability test must reject a subject on the rail"
        );
        assert!(
            crate::stats::likelihood::is_usable_subject_nll(SUBJECT_SENTINEL_NLL / 2.0),
            "and accept one below it, else the first assertion proves nothing"
        );
        assert_eq!(
            crate::estimation::outer_optimizer::nonfinite_objective_reason(
                2.0 * SUBJECT_SENTINEL_NLL
            ),
            "at the divergence sentinel"
        );
        assert_eq!(
            crate::estimation::outer_optimizer::nonfinite_objective_reason(
                SUBJECT_SENTINEL_NLL / 2.0
            ),
            "too large to be a fit"
        );
    }

    /// #2269: no per-subject guard spells the sentinel as a bare literal. Scans production
    /// code under `src/` (not comments, strings, `*_tests.rs` or `mod tests`).
    /// `POPULATION_OBJECTIVE_FILES` are exceptions whose `1e20`s clamp a *population*
    /// objective (or, in `io/fitrx.rs`, a formatting value). `saem.rs` uses the constant.
    #[test]
    fn no_bare_sentinel_literal_in_guard_files() {
        const POPULATION_OBJECTIVE_FILES: [&str; 7] = [
            "estimation/outer_optimizer.rs",
            "estimation/profile.rs",
            "estimation/trust_region.rs",
            "estimation/covariate_mu_ref.rs",
            "estimation/saem_mixture.rs",
            "estimation/impmap.rs",
            "io/fitrx.rs",
        ];
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).expect("read src/") {
                let path = e.expect("dir entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|x| x == "rs") {
                    out.push(path);
                }
            }
        }
        // `1e20` as a token: not `1e200`, not `x1e20`.
        fn has_literal(code: &str) -> bool {
            code.match_indices("1e20").any(|(i, _)| {
                let before = code[..i].chars().next_back();
                let after = code[i + 4..].chars().next();
                !before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
                    && !after.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
            })
        }
        const DEFINITION: &str = "pub(crate) const SUBJECT_SENTINEL_NLL: f64 = 1e20;";
        // The 1-based lines of `src` whose production code holds the literal, and
        // whether `src` holds the definition.
        fn bare_lines(src: &str) -> (Vec<usize>, bool) {
            let (mut lines, mut saw_definition, mut in_test_mod) = (Vec::new(), false, false);
            for (i, line) in src.lines().enumerate() {
                // rustfmt puts a top-level item's opening and closing at column 0.
                if line.starts_with("mod tests") || line.starts_with("pub(crate) mod tests") {
                    in_test_mod = true;
                }
                if in_test_mod {
                    // `\u{7d}` is the closing brace, escaped so this line's braces balance
                    // for the brace-counting source scans in `tests/` that blank this module.
                    in_test_mod = line != "\u{7d}";
                    continue;
                }
                if line.trim() == DEFINITION {
                    saw_definition = true;
                    continue;
                }
                let code = line.split("//").next().unwrap_or("");
                // Drop string literals, so a message may still say "1e20".
                let code: String = code.split('"').step_by(2).collect();
                if has_literal(&code) {
                    lines.push(i + 1);
                }
            }
            (lines, saw_definition)
        }
        // The matcher itself, so a scan that finds nothing is not a matcher that never
        // matches: the token rule, and which parts of a file it reads.
        assert!(has_literal("return 1e20;") && has_literal("x < 1e20"));
        assert!(!has_literal("1e200") && !has_literal("x1e20") && !has_literal("1e20_f"));
        let fixture = [
            "fn f() -> f64 \u{7b}",
            "    1e20 // a guard",
            "\u{7d}",
            "// 1e20 in a comment",
            "fn g() -> &'static str { \"1e20 in a string\" }",
            "mod tests \u{7b}",
            "    fn t() -> f64 { 1e20 }",
            "\u{7d}",
            "fn h() -> f64 { 2.0 * 1e20 }",
        ]
        .join("\n");
        assert_eq!(bare_lines(&fixture), (vec![2, 9], false));

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        walk(&root, &mut files);
        files.sort();
        let mut hits = Vec::new();
        let mut saw_definition = false;
        for path in &files {
            let rel = path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
            let is_test_file = rel.ends_with("_tests.rs") || rel.contains("/tests/");
            if is_test_file || POPULATION_OBJECTIVE_FILES.contains(&rel.as_str()) {
                continue;
            }
            let src = std::fs::read_to_string(path).expect("read source");
            let (lines, defines) = bare_lines(&src);
            saw_definition |= defines;
            hits.extend(lines.into_iter().map(|n| format!("src/{rel}:{n}")));
        }
        assert!(
            hits.is_empty(),
            "write SUBJECT_SENTINEL_NLL, not a bare 1e20 (or, for a population objective, \
             add the file to POPULATION_OBJECTIVE_FILES with a reason):\n{}",
            hits.join("\n")
        );
        // The scan must see the definition itself, else a moved file or renamed
        // constant would leave it scanning nothing; and the allowlist must name files
        // that exist, so a rename cannot silently widen it.
        assert!(saw_definition, "scan did not reach the definition");
        for f in POPULATION_OBJECTIVE_FILES {
            assert!(root.join(f).is_file(), "allowlisted {f} does not exist");
        }
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
        let borrowed: Vec<&[f64]> = kappas.iter().map(Vec::as_slice).collect();
        let reused = individual_nll_iov_with_scratch(
            &model,
            &subj,
            &theta,
            &eta,
            &borrowed,
            &omega,
            Some(&omega_iov),
            &sigma,
            &mut pk::EventPkParams::default(),
        );
        assert_eq!(iov.to_bits(), reused.to_bits());
        // Kappa prior is positive → IOV NLL should differ from base
        assert!((iov - base).abs() > 1e-6, "IOV NLL={}, base NLL={}", iov, base);
    }

    /// A model whose CL depends on both the BSV eta and the per-occasion
    /// kappa (`combined[1]`), so the kappa block genuinely enters the
    /// augmented R̃. The kappa read is defensive so the BSV-only
    /// `foce_subject_nll` path (which passes a length-1 eta) doesn't panic.
    fn make_iov_kappa_model() -> CompiledModel {
        let mut model = make_model();
        model.pk_param_fn =
            Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
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
        let kappas = vec![DVector::from_vec(vec![0.08]), DVector::from_vec(vec![-0.05])];
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

    /// FOCE+proportional: R is evaluated at a supplied `f(η=0)`, not the SB-linearized
    /// `f0`, whose zero crossing collapses `R(f0)` to the floor.
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
        // The f0 path's floored variance inflates the marginal by a clear margin (~56 here);
        // HΩH' cushions it, but it is far above FP noise.
        assert!(
            nll_f0 - nll_override > 20.0,
            "override must change the SB marginal (R evaluated at f(η=0), not f0): \
             nll_f0={nll_f0}, nll_override={nll_override}"
        );
    }

    /// #646: FOCE M3 censored rows use `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰`, not `R⁰`. With quantified
    /// rows' Jacobian zeroed, only the censored row can make the NLL depend on Ω.
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

    /// FREM r_diag merge collision: `frem_r_override` must reach the diagonal feeding R̃
    /// even though `r_pred_override` re-evaluates R afterward.
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

    /// Under additive error the Almquist `½·c̃'·c̃` correction is zero, so
    /// `foce_subject_nll_interaction` equals the hand-computed
    /// `a'·diag(1/R)·a + Ω⁻¹` Laplace value bit for bit. (FOCEI ≠ FOCE here: that identity
    /// holds only for the Sheiner–Beal form.)
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

    /// IIV on RUV (#409): a dedicated residual eta with additive error scales R by
    /// exp(2·η̂_ruv) and adds `c̃_{j,ruv}=2`, so `H̃ = 2·n_obs + 1/ω`; hand-computed, bit for bit.
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
            obs_times: vec![1.0, 2.0],
            observations: vec![11.0, 21.0],
            obs_cmts: vec![1, 1],
            cens: vec![0, 0],
            fremtype: vec![0, 100],
            ..Default::default()
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
            BloqMethod::Drop,
        );

        let pk_expected = 1.0 / (sigma[0] * eta_hat[0].exp());
        let frem_expected = 1.0 / sigma[1];
        assert_relative_eq!(cwres[0], pk_expected, epsilon = 1e-12);
        assert_relative_eq!(cwres[1], frem_expected, epsilon = 1e-12);
    }

    /// #1182: CWRES is `R̃^{-1/2}(y − f0)` with the symmetric square root, not the marginal
    /// standardisation. Oracle: `R̃^{1/2}·z = y − f0` (independent sqrt) and `zᵀz = rᵀR̃⁻¹r`.
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
            BloqMethod::Drop,
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

        // A first row censored *for this fit* (`CENS != 0` under M3) is NaN and
        // leaves the second row standardised on its own: 3/√1.25.
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
            BloqMethod::M3,
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
            BloqMethod::Drop,
        );
        assert_relative_eq!(cw[0], 2.0 / (0.25_f64 + 0.25 * 16.0).sqrt(), epsilon = 1e-12);
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
            BloqMethod::Drop,
        );
        assert_relative_eq!(cw_f0[0], 2.0 / (0.25_f64 + 0.25 * 1.0).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn compute_cwres_applies_custom_magnitude() {
        // #484 review #4: sdtab CWRES must use the per-observation magnitude.
        // With H = 0 the SB prediction f0 = ipred and R_tilde = R (diagonal), so
        // for additive error CWRES_j = (y_j − f_j) / (m_j·σ).
        let subject = Subject {
            id: "1".to_string(),
            obs_times: vec![1.0, 2.0],
            observations: vec![12.0, 22.0],
            obs_cmts: vec![1, 1],
            cens: vec![0, 0],
            fremtype: vec![0, 0],
            ..Default::default()
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
            BloqMethod::Drop,
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

        let base = obs_nll_subject_into(
            &model,
            &subj,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&[]),
            &eta,
            &mut scratch,
        );
        model.residual_error_eta = Some(0);
        let scaled = obs_nll_subject_into(
            &model,
            &subj,
            &theta,
            &sigma,
            ResidualCov::block_sigma(&[]),
            &eta,
            &mut scratch,
        );

        let s = (2.0_f64 * 0.4).exp();
        let preds = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta);
        let sig2 = sigma[0] * sigma[0];
        let mut expected = 0.0;
        for (j, &f) in preds.iter().enumerate() {
            let v = (sig2 * s).max(1e-12);
            expected += 0.5 * (v.ln() + (subj.observations[j] - f).powi(2) / v);
        }
        assert!(
            (scaled - expected).abs() < 1e-9,
            "obs_nll IIV-on-RUV scaling mismatch: got {scaled}, expected {expected}, base {base}"
        );
    }

    /// The Almquist `½·c̃'·c̃` correction is wired: additive → combined error changes the
    /// per-subject OFV non-trivially even when the data term barely shifts.
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

        let src = crate::test_models::ODE_9;
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
        assert!(inner.is_finite(), "inner joint NLL must be finite; got {inner}");

        // (2) SAEM M-step: obs_nll_subject_from_preds (Gaussian preds supplied).
        let preds = vec![30.0_f64; n_obs];
        let saem = obs_nll_subject_from_preds(
            &model,
            &subject,
            &preds,
            &p.theta,
            &p.sigma.values,
            model.residual_cov_at(p),
            &eta,
        );
        assert!(saem.is_finite(), "SAEM M-step joint NLL must be finite; got {saem}");

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
        assert!(foce.is_finite(), "FOCEI joint NLL must be finite; got {foce}");

        // The ODE hazard must actually contribute: moving η changes the inner NLL.
        let moved = individual_nll(&model, &subject, &p.theta, &[0.4], &p.omega, &p.sigma.values);
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
        crate::parser::model_parser::parse_model_string(crate::test_models::ODE_9)
            .expect("joint PK-TTE model must parse")
    }

    /// [`joint_pktte_ode_model`] without its compiled `[odes]` program: a spec the walks
    /// cannot record (`ode_hazard_walk_in_scope` requires `rhs_program`), while the share and
    /// the dedicated solve integrate the same `ode.rhs` closure. The fixture for the TTE
    /// router's non-walk arms — the one difference from the walk arm's model is the program,
    /// so the curves are the plain model's.
    #[cfg(feature = "survival")]
    fn walk_declined_joint_pktte_model() -> CompiledModel {
        let mut model = joint_pktte_ode_model();
        model.ode_spec.as_mut().expect("ODE model").rhs_program = None;
        model
    }

    /// The router's walk-arm predicate for `s` on the program-bearing model and on
    /// [`walk_declined_joint_pktte_model`]: `(true, false)` is the straddle the non-walk arms'
    /// tests stand on — the same subject, the walk taken exactly when the program is there.
    #[cfg(feature = "survival")]
    fn walk_scope_straddle(s: &Subject) -> (bool, bool) {
        let in_scope = crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope;
        (in_scope(&joint_pktte_ode_model(), s), in_scope(&walk_declined_joint_pktte_model(), s))
    }

    /// #1885's ODE fixture (the plan's M4): [`joint_pktte_ode_model`]'s subject on oral
    /// 100 at 0/12/24, so the later doses land on residual drug, with PK observations at
    /// 2/6/14/26 and one exact event at `t_event` on CMT 2. With `reset`, one EVID=3
    /// reset after every record, which declines the #570 share and sends the likelihood
    /// to the dedicated `tte_ode_nll` solve.
    #[cfg(feature = "survival")]
    fn ebe_pktte_subject(id: &str, t_event: f64, reset: bool) -> Subject {
        use crate::types::{EventType, ObsRecord};
        let mut s = make_simple_subject();
        s.id = id.to_string();
        s.doses = [0.0, 12.0, 24.0]
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect();
        s.obs_times = vec![2.0, 6.0, 14.0, 26.0];
        s.observations = vec![7.0, 6.0, 9.0, 10.0];
        s.obs_cmts = vec![1; 4];
        s.cens = vec![0; 4];
        s.occasions = vec![1; 4];
        s.obs_records = vec![ObsRecord::Event {
            time: t_event,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        if reset {
            s.reset_times = vec![60.0];
        }
        s
    }

    #[cfg(feature = "survival")]
    fn ebe_population(subjects: Vec<Subject>) -> crate::types::Population {
        crate::types::Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    }

    /// #1885 T2: for a joint PK-TTE subject scored on the **dedicated** solve,
    /// `predict_survival_at_etas_diag` on its scored times reads `H`/`h` bit for bit, and
    /// the exact-event term rebuilt from the row equals `tte_endpoint_nll`. Two subjects
    /// in reverse order catch a zero η or a zip slip; the rebuilt term catches another engine.
    #[cfg(feature = "survival")]
    #[test]
    fn ebe_survival_reads_the_dedicated_ode_solve_bit_for_bit() {
        use crate::types::{EventType, ObsRecord};
        let model = joint_pktte_ode_model();
        let p = &model.default_params;
        let (chz_state, hazard) = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: h @ HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => (*chz_state, h),
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };
        let t = 25.3;
        let cases = [("A", 0.4), ("B", -0.7)];
        let pop =
            ebe_population(cases.iter().map(|(id, _)| ebe_pktte_subject(id, t, true)).collect());
        let etas = crate::api::SuppliedEtas::from_table(
            vec!["B".into(), "A".into()],
            vec!["ETA_CL".into()],
            vec![vec![-0.7], vec![0.4]],
        )
        .unwrap();
        let out = crate::api::predict_survival_at_etas_diag(&model, &pop, p, &etas, &[t]).unwrap();
        assert_eq!(out.results.len(), 2);
        let tol = crate::survival::MonoTol::from_solver(
            &model.ode_spec.as_ref().unwrap().effective_solver_opts(),
        );
        for ((id, eta), subject) in cases.iter().zip(&pop.subjects) {
            assert!(
                try_joint_pktte_shared_solve(&model, subject, &p.theta, &[*eta]).is_none(),
                "subject {id} must be scored on the dedicated solve, not the share"
            );
            let row = out.results.iter().find(|r| r.id == *id).unwrap();
            assert!(row.cum_hazard.is_finite() && row.hazard.is_finite());
            let (cum, haz) = crate::survival::ode_cumhaz_hazard(
                &model,
                subject,
                chz_state,
                &p.theta,
                &[*eta],
                &[t],
            );
            assert_eq!(
                (row.cum_hazard.to_bits(), row.hazard.to_bits()),
                (cum[0].to_bits(), haz[0].to_bits()),
                "subject {id}: predicted ({}, {}) vs scored ({}, {})",
                row.cum_hazard,
                row.hazard,
                cum[0],
                haz[0]
            );
            let records = [ObsRecord::Event {
                time: t,
                event_type: EventType::Exact,
                entry_time: 0.0,
                cmt: 2,
            }];
            let scored = tte_endpoint_nll(
                &model,
                subject,
                hazard,
                crate::types::TteRecurrence::Single,
                &records,
                &p.theta,
                &[*eta],
            );
            assert!(scored.is_finite() && scored < 1e19, "{scored}");
            let rebuilt =
                tte_ode_nll_from_curves(&records, &[t], &[row.cum_hazard], &[row.hazard], tol);
            assert_eq!(rebuilt.to_bits(), scored.to_bits(), "subject {id}");
        }
    }

    /// #1885 T3: on the #570 **share**, the individual curve (0.5-step grid 0–48 plus Tᵢ)
    /// agrees with the share's `H(Tᵢ)` to ODE tolerance, not bitwise.
    ///
    /// Band, measured: worst relative gap 1.300e-4 (Linux aarch64 and macOS arm64); the
    /// bound is ~2.3× above it. η̂ moving `H(Tᵢ)` by more than half excludes the η = 0 curve.
    #[cfg(feature = "survival")]
    #[test]
    fn ebe_survival_agrees_with_the_share_to_solver_tolerance() {
        let model = joint_pktte_ode_model();
        let p = &model.default_params;
        let chz_state = match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte {
                hazard: HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => *chz_state,
            _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
        };
        const BAND: f64 = 3e-4;
        let mut worst = 0.0_f64;
        for t in [30.0, 25.3] {
            let mut grid: Vec<f64> = (0..=96).map(|k| 0.5 * f64::from(k)).collect();
            grid.push(t);
            for eta in [0.4, -0.7] {
                let subject = ebe_pktte_subject("1", t, false);
                let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[eta])
                    .expect("the reset-free subject must be scored on the share");
                let i = share.times.iter().position(|&s| s == t).unwrap();
                let h_share = share.chz_states[i][chz_state];
                let pop = ebe_population(vec![subject]);
                let at = |e: f64| {
                    let etas = crate::api::SuppliedEtas::from_table(
                        vec!["1".into()],
                        vec!["ETA_CL".into()],
                        vec![vec![e]],
                    )
                    .unwrap();
                    crate::api::predict_survival_at_etas_diag(&model, &pop, p, &etas, &grid)
                        .unwrap()
                        .results
                        .into_iter()
                        .find(|r| r.time == t)
                        .unwrap()
                        .cum_hazard
                };
                let (h_pred, h_zero) = (at(eta), at(0.0));
                assert!(
                    h_share.is_finite() && h_pred.is_finite() && h_zero.is_finite(),
                    "T={t} η={eta}: share {h_share}, predict {h_pred}, η=0 {h_zero}"
                );
                let rel = (h_pred - h_share).abs() / h_share;
                worst = worst.max(rel);
                eprintln!("T3 T={t} eta={eta}: share {h_share} predict {h_pred} rel {rel:.3e}");
                assert!(rel <= BAND, "T={t} η={eta}: rel {rel:e} > {BAND:e}");
                let moved = (h_pred - h_zero).abs() / h_pred.max(h_zero);
                assert!(moved >= 0.5, "T={t} η={eta}: η̂ moved H by {moved} only");
            }
        }
        eprintln!("T3 worst share-vs-predict rel gap {worst:.3e}");
    }

    /// No-TV `[odes]` plus an `OdeAccumulated` hazard reading `TAD` — the shape
    /// `try_joint_pktte_shared_solve` admits to the #570 share (its time guard does not
    /// inspect the injected `d/dt(__chz_*)` line). Shared by the two tests below.
    #[cfg(feature = "survival")]
    fn tad_hazard_two_dose_model() -> CompiledModel {
        use crate::parser::model_parser::parse_model_string;
        parse_model_string(
            r"
[parameters]
theta TVCL(1.0, 0.01, 100.0)
theta TVV(10.0, 0.1, 500.0)
theta TVH0(0.1, 0.001, 10.0)
theta TVKT(0.01, 0.0001, 1.0)
sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
CL = TVCL
V = TVV
H0 = TVH0
KT = TVKT
[structural_model]
ode(obs_cmt=central, states=[central])
[odes]
d/dt(central) = -CL / V * central
[event_model]
cmt = 3
hazard = H0 * (1.0 + KT * TAD)
[error_model]
DV ~ proportional(PROP_ERR)
",
        )
        .expect("TAD-reading ODE hazard must parse")
    }

    /// A joint PK-TTE subject whose first record is a dose at **t = 10**, so a TTE time
    /// below it is pre-start; PK obs at 12/16/24 route the pre-start read through the
    /// pre-first-break fill, not the `k = 0` boundary visit (#1218).
    /// #1261: a TTE hazard readout must receive the dose-time anchors the ODE used (a bare
    /// `PkParams::values` made TAD NaN → 1e20 sentinel).
    #[cfg(feature = "survival")]
    #[test]
    fn tte_hazard_readout_uses_the_current_tad_anchor() {
        use crate::types::{EventType, ObsRecord};

        let model = tad_hazard_two_dose_model();
        let (chz_state, hazard) = match model.endpoints.get(&3) {
            Some(EndpointLikelihood::Tte {
                hazard: h @ HazardSpec::OdeAccumulated { chz_state },
                ..
            }) => (*chz_state, h),
            _ => panic!("expected ODE-accumulated endpoint"),
        };
        let mut subject = make_simple_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.obs_times = vec![30.0];
        subject.observations = vec![0.0];
        subject.obs_cmts = vec![1];
        subject.cens = vec![0];
        subject.occasions = vec![1];
        subject.obs_records = vec![ObsRecord::Event {
            time: 30.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 3,
        }];
        let p = &model.default_params;
        let eta = [];
        let nll = tte_endpoint_nll(
            &model,
            &subject,
            hazard,
            crate::types::TteRecurrence::Single,
            &subject.obs_records,
            &p.theta,
            &eta,
        );

        // H(30) = .1 * [integral_0^12(1 + .01t) + integral_12^30(1 + .01(t - 12))] = 3.234.
        // h(30) = .1 * (1 + .01 * 18) = .118. Two doses prevent TAD and TAFD aliasing.
        // The 1e10 guard is ten orders below the old 2e20 sentinel and nine orders above
        // this roughly 5.37 objective, so it rejects the masked NaN without overfitting
        // to solver-level round-off.
        let expected = 3.234_f64 - 0.118_f64.ln();
        assert_relative_eq!(nll, expected, epsilon = 2e-4);
        assert!(
            nll < 1e10,
            "the old NaN path returns the finite 1e20 sentinel, not an infinite NLL; got {nll}"
        );

        let (cum, haz) = crate::survival::ode_cumhaz_hazard(
            &model,
            &subject,
            chz_state,
            &p.theta,
            &eta,
            &[30.0],
        );
        assert_relative_eq!(cum[0], 3.234, epsilon = 2e-4);
        assert_relative_eq!(haz[0], 0.118, epsilon = 2e-6);
    }

    /// #1261 follow-up: through the production dispatcher
    /// `individual_nll_into_with_schedule`, which routes the TTE term through
    /// `tte_ode_nll_from_shared` / `share.rhs_params[i]`. The TTE contribution is isolated
    /// by subtracting a PK-only sibling (same `[odes]`, doses, obs; no share), whose
    /// Gaussian term is bit-identical.
    #[cfg(feature = "survival")]
    #[test]
    fn hot_path_individual_nll_reads_the_shared_solve_tad_anchor() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        let model = tad_hazard_two_dose_model();
        let pk_only_model = parse_model_string(
            r"
[parameters]
theta TVCL(1.0, 0.01, 100.0)
theta TVV(10.0, 0.1, 500.0)
sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
CL = TVCL
V = TVV
[structural_model]
ode(obs_cmt=central, states=[central])
[odes]
d/dt(central) = -CL / V * central
[error_model]
DV ~ proportional(PROP_ERR)
",
        )
        .expect("PK-only sibling model must parse");

        let mut subject = make_simple_subject();
        subject.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        subject.obs_times = vec![30.0];
        subject.observations = vec![0.0];
        subject.obs_cmts = vec![1];
        subject.cens = vec![0];
        subject.occasions = vec![1];
        subject.obs_records = vec![ObsRecord::Event {
            time: 30.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 3,
        }];
        let mut pk_only_subject = subject.clone();
        pk_only_subject.obs_records = Vec::new();

        let eta = [];

        // The fixture must actually reach the shared solve — otherwise this test would
        // silently exercise the pre-#570 fallback the sibling test already covers,
        // rather than the `tte_ode_nll_from_shared` dispatch under review here.
        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &model.default_params.theta, &eta)
                .is_some(),
            "fixture must qualify for the #570 shared solve"
        );

        let mut scratch = crate::pk::EventPkParams::default();
        let p = &model.default_params;
        let nll_full = individual_nll_into_with_schedule(
            &model,
            &subject,
            &p.theta,
            &eta,
            &p.omega,
            &p.sigma.values,
            &model.residual_correlations,
            &mut scratch,
            None,
        );

        let pk_p = &pk_only_model.default_params;
        let nll_pk_only = individual_nll_into_with_schedule(
            &pk_only_model,
            &pk_only_subject,
            &pk_p.theta,
            &eta,
            &pk_p.omega,
            &pk_p.sigma.values,
            &pk_only_model.residual_correlations,
            &mut scratch,
            None,
        );

        // Same H(30) = 3.234, h(30) = .118 as the sibling test — the isolated TTE term
        // the shared solve must have contributed.
        let expected_tte = 3.234_f64 - 0.118_f64.ln();
        assert_relative_eq!(nll_full - nll_pk_only, expected_tte, epsilon = 2e-4);
        assert!(
            nll_full < 1e10,
            "the old NaN-TAD path returns the finite 1e20 sentinel through this exact \
             dispatcher; got {nll_full}"
        );
    }

    /// Construct a subject with a positive pre-start TTE time and later PK observations.
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

    /// #1223: a TTE time **at or before the first record** is scored, not repelled, and
    /// identically on all three production callers (the #570 share, the dedicated solve,
    /// `predict_survival`). Arms: left-truncation `entry_time`, interval `left`, right
    /// censoring time. The seeded state is all zeros, so the discriminating value is
    /// `h = H0 = 0.01` (the hand-computed oracle with `H = 0`); the non-zero-state check
    /// lives in `ode::predictions::tests::prestart_chz_time_reads_the_seeded_state_on_both_paths`.
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
        assert_eq!(tv_h0, 0.01, "theta[3] must be TVH0 — the arms read h against it");

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
                    event_type: EventType::IntervalCensored { left: 5.0, right: 20.0 },
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
            // `left = 0.0` deliberately: `entry_time` is pushed only if `> 0`, `left` always, so
            // this reaches the share; and `predict_survival(&[0.0])` is #1218's zero-maximum grid
            // (PR #1222 coverage). Do not "tidy" it to 5.0/8.0.
            (
                "interval censoring with left = 0 (ungated bound)",
                0.0_f64,
                vec![ObsRecord::Event {
                    time: 20.0,
                    event_type: EventType::IntervalCensored { left: 0.0, right: 20.0 },
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
            (ode.rhs)(st, &share.rhs_params[i], t_pre, &mut du);
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
            let surv = crate::api::predict_survival(&model, &population, p, &[t_pre]).unwrap();
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

            // FOCEI Laplace leg: the ODE-TTE FD-Hessian perturbs η around the pre-start node, so
            // a sentinel reinstated only inside the perturbation would pass the value checks.
            // Zero Gaussian H is fine: prior + TTE Hessian keep H̃ PD.
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
                 repelled; got {foce} (the TTE sentinel is {SUBJECT_SENTINEL_NLL:e} and is itself finite)"
            );
        }
    }

    /// #1223 positive control straddling the integration start: a pre-start `entry_time`
    /// contributes nothing (`H(entry) = 0`), a post-start one changes the objective. Rules
    /// out both "ignore entry_time" and "start the clock at the earliest TTE time".
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

        // Bit-exact by construction (measured `306.2186273580891` both ways): the pre-start
        // node is filled before the break walk and `t_last` stays 24, so every segment is
        // identical. A tolerance would be strictly weaker.
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

    /// #1187: the joint PK-TTE hazard's two production routes — the #570 share
    /// (`active_infusions`) and the two-solve fallback (`gated_infusions`) — must agree for a
    /// `RATE>0` dose into a built-in absorption forcing; before the fix they integrated
    /// different exposures. Routing (#1166) can move a model between arms, so A ≡ B matters.
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

    /// #1187 external anchor: joint PK-TTE cumulative hazard `A(3)` under a `RATE>0` dose
    /// into an absorption compartment vs NONMEM 7.6.0 (`nonmem_anchor/pktte_inf.{ctl,csv}`,
    /// `ADVAN13 TOL=9`, all fixed, η = 0). CHZ rather than OFV, which carries tool-specific
    /// constants.
    ///
    /// Measured: before the fix the dedicated arm read `H(1.86) = 3.796844e0` vs NONMEM
    /// `1.2217e-1` (31×); the shared arm was already correct.
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
        subject.observations = vec![4.6271, 6.2626, 7.1063, 8.0446, 6.3602, 4.8718, 3.6207, 2.0972];
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

        // NONMEM 7.6.0, `nonmem_anchor/results/pktte_inf.tab`: CHZ at TIME = 1.86, printed at
        // 9 significant figures (`FORMAT=s1PE15.8`): the default 5 would add 4.1e-5 relative
        // rounding of its own, a floor under any bound here.
        const NONMEM_CHZ_AT_EVENT: f64 = 1.22168018e-1;

        // Arm B — the two-solve fallback, i.e. the arm that carried the defect.
        let (cum_dedicated, _haz) =
            crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, theta, &eta, &[1.86]);
        // Arm A — the #570 shared single solve.
        let share = try_joint_pktte_shared_solve(&model, &subject, theta, &eta)
            .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
        assert_eq!(share.times, vec![1.86], "share must cover the event time");
        let cum_shared = share.chz_states[0][chz_state];

        // Per-arm bounds, measured: dedicated 1.41e-9, shared 3.26e-6 (different timelines,
        // different local error). One loose bound would hide a regression in the tight arm.
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

    /// #570 guard: a joint PK-TTE model whose `[individual_parameters]` reads `TIME` must
    /// not take the share; its single t=0 snapshot would be ~30–50% off.
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
        let nll = individual_nll(&model, &subject, &p.theta, &[0.0], &p.omega, &p.sigma.values);
        let moved = individual_nll(&model, &subject, &p.theta, &[0.3], &p.omega, &p.sigma.values);
        assert!(
            nll.is_finite() && moved.is_finite() && (nll - moved).abs() > 1e-9,
            "fallback joint NLL must be finite and η-sensitive (nll={nll}, moved={moved})"
        );
    }

    /// The same rejection for model time read in the **`[odes]` RHS**: only the wide
    /// `pk::model_uses_time_anywhere` fires here (the sibling passes under the narrow
    /// predicate too). Since #1124 such a model routes to the event-driven walker, so the
    /// share would score a different Gaussian term than the reported IPRED.
    #[test]
    #[cfg(feature = "survival")]
    fn joint_pktte_share_rejects_model_time_in_the_odes_rhs() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        // Time dependence only in the PK RHS; the hazard is time-independent so the RHS is the
        // only reason to reject. Since #1166 a `TIME`-reading hazard does not trip the
        // predicate (pinned by `joint_pktte_share_admits_a_time_dependent_hazard`).
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
        let nll = individual_nll(&model, &subject, &p.theta, &[0.0], &p.omega, &p.sigma.values);
        let moved = individual_nll(&model, &subject, &p.theta, &[0.3], &p.omega, &p.sigma.values);
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

    /// #1166: a Gompertz-shaped (`TIME`-reading) hazard on an autonomous PK block **takes**
    /// the share; with the sibling, the gate is pinned from both sides.
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
        let prog =
            model.ode_spec.as_ref().and_then(|o| o.rhs_program.as_ref()).expect("rhs program");
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

    /// An `EVID=3/4` reset **declines** the share (a reset needs the event-driven walker) —
    /// the premise of `tests/pktte_prestart_entry.rs` (A1 via share, A4 via dedicated).
    /// Measured: disabling `|| subject.has_resets()` fails only this test.
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
        assert!(!subject.has_resets(), "control subject must carry no resets");
        assert!(
            try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_some(),
            "the reset-free subject must qualify for the share, or this test proves nothing \
             about resets"
        );

        // With one, it must not.
        let mut reset_subject = late_start_joint_subject(records);
        reset_subject.reset_times = vec![15.0];
        assert!(reset_subject.has_resets(), "fixture must actually carry a reset");
        assert!(
            try_joint_pktte_shared_solve(&model, &reset_subject, &p.theta, &[0.0]).is_none(),
            "an EVID=3/4 reset must decline the #570 share — the share takes a single t=0 PK \
             snapshot, and a reset needs the event-driven walker"
        );
    }

    /// #1166 routing oracle: two models computing the **same arithmetic**
    /// (`* (1.0 + 0.0*TIME)` is an IEEE identity, not folded by the bytecode) must land on
    /// the same engine and agree bit-for-bit. Before #1166 the flagged twin took the
    /// event-driven walk (0 of 40 subjects bit-identical, worst 3.25e-4).
    #[test]
    #[cfg(feature = "survival")]
    fn identical_arithmetic_takes_the_identical_engine_bit_for_bit() {
        use crate::parser::model_parser::parse_model_string;
        use crate::types::{EventType, ObsRecord};

        // The pair must straddle the old predicate (found by mutation): the base hazard is
        // autonomous, only the twin's identity factor reads time.
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
            let a = individual_nll(&plain, &subject, &p.theta, &[eta], &p.omega, &p.sigma.values);
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

    /// #1218 on the fit path: a subject whose only TTE record sits on its first dose builds
    /// `times = [0.0]` in `tte_ode_nll`'s dedicated solve and read `NaN` (→ `1e20`). It now
    /// scores `H(0) − ln h(0)` with the post-dose hazard. An exact event, since a censor
    /// scores `H(0) = 0` under the wrong (seeded-state) fix too. Measured: `1e20` before,
    /// `−ln 0.02 = 3.91` under the seeded-state variant vs `−ln 25.6` post-dose.
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

        let nll = tte_ode_nll(&model, &subject, chz_state, &subject.obs_records, theta, &eta);
        assert!(
            nll.is_finite() && nll < SUBJECT_SENTINEL_NLL,
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
    /// #2267: the joint (TTE/categorical/CTMM) marginal records the curvature it factors
    /// (with the non-Gaussian η-Hessian) and names it when that breaks. Healthy, the parts
    /// sum to the contribution under FOCEI and FOCE (measured gap `0`, bound `1e-12·nll`).
    #[cfg(feature = "survival")]
    #[test]
    fn the_joint_marginal_names_a_curvature_the_non_gaussian_hessian_broke() {
        let subject = make_simple_subject();
        let ipreds: Vec<f64> = subject.observations.iter().map(|y| 1.05 * y).collect();
        let h = DMatrix::from_fn(6, 1, |j, _| -0.1 * ipreds[j] * subject.obs_times[j]);
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let eta = DVector::from_vec(vec![0.1]);
        let tte_nll = 2.0;
        let run = |h: &DMatrix<f64>, tte_h: f64, interaction: bool| {
            let mut parts = Vec::new();
            let nll = foce_subject_nll_interaction_with_tte(
                &subject,
                &ipreds,
                &eta,
                h,
                &omega,
                &[0.1],
                &ErrorSpec::Single(ErrorModel::Proportional),
                BloqMethod::Drop,
                &[],
                tte_nll,
                DMatrix::from_element(1, 1, tte_h),
                interaction,
                None,
                None,
                None,
                Some(&mut parts),
            );
            (nll, parts)
        };
        let names = |parts: &[(String, f64)]| -> Vec<String> {
            parts.iter().map(|(n, _)| n.clone()).collect()
        };

        for interaction in [true, false] {
            let (nll, parts) = run(&h, 3.0, interaction);
            assert_eq!(
                names(&parts),
                ["Gaussian residual term", "random-effect prior", "curvature log-determinant"],
                "interaction = {interaction}"
            );
            let summed = parts.iter().map(|(_, v)| v).sum::<f64>() + tte_nll;
            eprintln!("interaction = {interaction}: gap {:e}", (summed - nll).abs());
            assert!(
                nll.is_finite() && nll < 1e3 && (summed - nll).abs() <= 1e-12 * nll.abs(),
                "interaction = {interaction}: Σ parts + tte = {summed}, marginal = {nll}"
            );
        }

        // The straddle: each case is a sentinel of the same marginal, named for its cause.
        let broken = [
            (
                h.clone(),
                -1e6,
                "curvature (H̃ not positive definite once the non-Gaussian Hessian is added)",
            ),
            (
                h.clone(),
                f64::NAN,
                "non-Gaussian curvature (η-Hessian of the time-to-event, categorical or \
                 CTMM terms not finite)",
            ),
            (DMatrix::from_element(6, 1, f64::NAN), 3.0, CURVATURE_FAILED),
        ];
        for (h, tte_h, want) in broken {
            let (nll, parts) = run(&h, tte_h, true);
            assert_eq!(nll, SUBJECT_SENTINEL_NLL, "{want}");
            let last = parts.last().expect("a curvature part");
            assert!(last.0 == want && last.1.is_nan(), "{parts:?}");
        }
    }

    // ── #901: per-record TTE terms on each ODE arm of the router (T6 / T7) ──

    /// The two-record joint PK-TTE subject of T6 / T7 (dose at t = 10): `entry` 5 is outside
    /// the dual walk's reach, 12 inside; `ss` leaves the walk's scope; `reset` declines the share.
    #[cfg(feature = "survival")]
    fn record_terms_subject(reset: bool, ss: bool, entry: f64) -> Subject {
        let mut subject = late_start_joint_subject(vec![
            ObsRecord::Event {
                time: 20.0,
                event_type: crate::types::EventType::Exact,
                entry_time: entry,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 30.0,
                event_type: crate::types::EventType::RightCensored,
                entry_time: entry,
                cmt: 2,
            },
        ]);
        if ss {
            subject.doses = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, true, 12.0)];
        }
        if reset {
            subject.reset_times = vec![15.0];
        }
        subject
    }

    /// The fixture's `chz_state` and its single TTE endpoint's `(hazard, recurrence)`.
    #[cfg(feature = "survival")]
    fn ode_tte_endpoint(
        model: &CompiledModel,
    ) -> (usize, &HazardSpec, crate::types::TteRecurrence) {
        let EndpointLikelihood::Tte { hazard, recurrence, .. } = &model.endpoints[&2] else {
            panic!("CMT 2 is the TTE endpoint");
        };
        let HazardSpec::OdeAccumulated { chz_state } = hazard else {
            panic!("the joint fixture's hazard is ODE-accumulated");
        };
        (*chz_state, hazard, *recurrence)
    }

    /// Fold `terms` the way the survival loops fold them: `dh − log h` per point.
    #[cfg(feature = "survival")]
    fn fold_terms(terms: &[crate::survival::RecordTerm<f64>]) -> f64 {
        let mut nll = 0.0_f64;
        for t in terms {
            nll = match *t {
                crate::survival::RecordTerm::Point { dh, log_h, .. } => match log_h {
                    Some(l) => nll + (dh - l),
                    None => nll + dh,
                },
                crate::survival::RecordTerm::Interval { log_prob, .. } => nll - log_prob,
            };
        }
        nll
    }

    /// The router's terms for `s`, given `share`, with the subject-level entry point
    /// (`tte_diagnostics`, which builds its own share) checked to report the same `ΔH`.
    #[cfg(feature = "survival")]
    fn router_terms(
        model: &CompiledModel, s: &Subject, share: Option<&JointPkTteSolve>, eta: &[f64],
    ) -> Vec<crate::survival::RecordTerm<f64>> {
        let (_, hazard, recurrence) = ode_tte_endpoint(model);
        let p = &model.default_params;
        let terms = tte_endpoint_record_terms(
            model,
            s,
            hazard,
            recurrence,
            &s.obs_records,
            &p.theta,
            eta,
            share,
        )
        .expect("finite NLL ⟹ terms");
        assert_eq!(terms.len(), 2);
        let rows = tte_diagnostics(model, s, &p.theta, eta);
        let crate::survival::RecordTerm::Point { dh, .. } = terms[1] else {
            panic!("the censor is a point record");
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].coxsnell.to_bits(), dh.to_bits(), "tte_diagnostics arm");
        terms
    }

    /// T6a (#901), walk arm: an in-scope subject is scored off the dual walk (#2482) even
    /// when a #570 share exists; terms fold **bit-identically** to `survival::ode_tte_nll_g`
    /// and differ from the share's NLL. (The walk reproduces the dedicated solve bit for bit
    /// here — measured rel 0 at η ∈ {0.2, −0.4, 0, 0.7}.)
    #[cfg(feature = "survival")]
    #[test]
    fn tte_record_terms_fold_to_the_walk_nll() {
        let model = joint_pktte_ode_model();
        let p = &model.default_params;
        let (chz_state, _, _) = ode_tte_endpoint(&model);
        let eta = [0.2];
        let s = record_terms_subject(false, false, 12.0);
        assert!(
            crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope(&model, &s),
            "the plain subject must be in the walk's scope (straddle)"
        );
        let share = try_joint_pktte_shared_solve(&model, &s, &p.theta, &eta)
            .expect("…and admitted to the share, so the walk's precedence is what is tested");
        let walk = crate::survival::ode_tte_nll_g::<f64>(
            &model,
            &s,
            chz_state,
            &s.obs_records,
            &p.theta,
            &eta,
        )
        .expect("walk value");
        let terms = router_terms(&model, &s, Some(&share), &eta);
        assert_eq!(fold_terms(&terms).to_bits(), walk.to_bits());
        let shared = tte_ode_nll_from_shared(
            model.ode_spec.as_ref().unwrap(),
            &share,
            chz_state,
            &s.obs_records,
        );
        assert_ne!(
            shared.to_bits(),
            walk.to_bits(),
            "the fixture must tell the walk from the share, or the bit test is vacuous"
        );
    }

    /// T6b (#901), shared arm: without a compiled `[odes]` program the subject leaves the
    /// walk's scope while the share admits it; terms fold bit-identically to
    /// `tte_ode_nll_from_shared`.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_record_terms_fold_to_the_shared_solve_nll() {
        let model = walk_declined_joint_pktte_model();
        let p = &model.default_params;
        let (chz_state, _, _) = ode_tte_endpoint(&model);
        let eta = [0.2];
        let s = record_terms_subject(false, true, 5.0);
        assert_eq!(
            walk_scope_straddle(&s),
            (true, false),
            "only the missing program may take the subject out of the walks' scope (straddle)"
        );
        let share = try_joint_pktte_shared_solve(&model, &s, &p.theta, &eta)
            .expect("the reset-free subject must take the #570 share (straddle)");
        let nll = tte_ode_nll_from_shared(
            model.ode_spec.as_ref().unwrap(),
            &share,
            chz_state,
            &s.obs_records,
        );
        let terms = router_terms(&model, &s, Some(&share), &eta);
        assert_eq!(fold_terms(&terms).to_bits(), nll.to_bits());
        let dedicated = tte_ode_nll(&model, &s, chz_state, &s.obs_records, &p.theta, &eta);
        assert_ne!(
            dedicated.to_bits(),
            nll.to_bits(),
            "the fixture must tell the two engines apart, or the bit test above is vacuous"
        );
    }

    /// #901: on the `1e20` rail `tte_endpoint_record_terms` returns `None`, not the partial
    /// terms visited before the failing record (an RTTE interval record); the control
    /// without it returns its one term.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_record_terms_are_none_on_the_sentinel_rail() {
        use crate::types::{EventType, HazardFamily, RtteClock, TteRecurrence};
        let model = joint_pktte_ode_model(); // only read on the ODE arms
        let subject = make_simple_subject();
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn: Box::new(|_: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![0.1]),
        };
        let forward = TteRecurrence::Repeated { clock: RtteClock::Forward };
        let event =
            ObsRecord::Event { time: 4.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 };
        let interval = ObsRecord::Event {
            time: 8.0,
            event_type: EventType::IntervalCensored { left: 6.0, right: 8.0 },
            entry_time: 0.0,
            cmt: 2,
        };
        let terms = |records: &[ObsRecord]| {
            tte_endpoint_record_terms(
                &model,
                &subject,
                &hazard,
                forward,
                records,
                &[],
                &[0.0],
                None,
            )
        };
        assert_eq!(terms(std::slice::from_ref(&event)).map(|t| t.len()), Some(1));
        assert!(terms(&[event, interval]).is_none());
    }

    /// #901, competing risks: one row per cause's record, in ascending CMT (records given
    /// descending), each from its cause-specific `H_k`, never the total hazard.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_diagnostics_give_one_cause_specific_row_per_cause_record() {
        use crate::types::EventType;
        let model = crate::parser::model_parser::parse_model_string(
            "
[parameters]
  theta TVLAMBDA_A(0.10, 0.001, 10.0)
  theta TVLAMBDA_B(0.06, 0.001, 10.0)
  omega ETA_F ~ 0.25
[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)
[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
",
        )
        .expect("competing-risks model must parse");
        let mut subject = make_simple_subject();
        subject.obs_records = [(3, EventType::RightCensored), (2, EventType::Exact)]
            .into_iter()
            .map(|(cmt, event_type)| ObsRecord::Event {
                time: 4.0,
                event_type,
                entry_time: 0.0,
                cmt,
            })
            .collect();
        let eta = [0.3_f64];
        let rows = tte_diagnostics(&model, &subject, &model.default_params.theta, &eta);
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].cmt, rows[0].dv), (2, 1.0));
        assert_eq!((rows[1].cmt, rows[1].dv), (3, 0.0));
        let e = eta[0].exp();
        for (row, lambda) in rows.iter().zip([0.10, 0.06]) {
            let want = lambda * e * 4.0;
            assert!(
                (row.coxsnell - want).abs() < 1e-14,
                "CMT {}: {} vs cause-specific {want}",
                row.cmt,
                row.coxsnell
            );
            assert!((row.martingale - (row.dv - want)).abs() < 1e-14);
        }
    }

    /// #901: a TTE row's `TIME` is the time **as written** (`Subject::param_time_at`) in a
    /// reset period: written 4 at engine 104 must report 4, not 104.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_diagnostics_report_the_written_time_in_a_restarted_period() {
        use crate::types::EventType;
        let model = crate::parser::model_parser::parse_model_string(
            "
[parameters]
  theta TVLAMBDA(0.10, 0.001, 10.0)
  omega ETA ~ 0.25
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA)
",
        )
        .expect("TTE model must parse");
        let mut subject = make_simple_subject();
        subject.reset_period_times = vec![(100.0, 0.0)];
        subject.obs_records = vec![ObsRecord::Event {
            time: 104.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        assert_eq!(subject.param_time_at(104.0), 4.0, "fixture: written time is 4");
        let eta = [0.0_f64];
        let rows = tte_diagnostics(&model, &subject, &model.default_params.theta, &eta);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].time, 4.0, "TIME must be as written, not the engine 104");
        // …while ΔH stays on the engine clock the likelihood scored.
        assert!((rows[0].coxsnell - 0.10 * 104.0).abs() < 1e-12);
    }

    /// T7 (#901), dedicated arm: no `[odes]` program and an EVID-3 reset (both declines
    /// asserted), terms fold bit-identically to `tte_ode_nll`.
    #[cfg(feature = "survival")]
    #[test]
    fn tte_record_terms_fold_to_the_dedicated_solve_nll() {
        let model = walk_declined_joint_pktte_model();
        let p = &model.default_params;
        let (chz_state, _, _) = ode_tte_endpoint(&model);
        let eta = [0.2];
        let r = record_terms_subject(true, true, 5.0);
        assert_eq!(
            walk_scope_straddle(&r),
            (true, false),
            "only the missing program may take the subject out of the walks' scope (straddle)"
        );
        assert!(
            try_joint_pktte_shared_solve(&model, &r, &p.theta, &eta).is_none(),
            "the reset subject must decline the share (straddle)"
        );
        let nll = tte_ode_nll(&model, &r, chz_state, &r.obs_records, &p.theta, &eta);
        let terms = router_terms(&model, &r, None, &eta);
        assert_eq!(fold_terms(&terms).to_bits(), nll.to_bits());
    }
}

/// Conditional observation NLL from precomputed predictions with the inner/IOV raw
/// semantics (no prediction floor, no variance clamp), unlike
/// [`obs_nll_subject_from_preds`]. Kept as an independent oracle for
/// `nested_correlation_tests` since AGQ moved to the adjoint (#486).
#[cfg(test)]
#[cfg_attr(not(feature = "survival"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conditional_obs_nll_from_preds(
    model: &CompiledModel, subject: &Subject, preds: &[f64], theta: &[f64], sigma_values: &[f64],
    residual_cov: ResidualCov<'_>, eta: &[f64], iov: bool,
) -> f64 {
    let semantics =
        if iov { PredNllSemantics::ConditionalIov } else { PredNllSemantics::Conditional };
    obs_nll_subject_from_preds_with_semantics(
        model,
        subject,
        preds,
        theta,
        sigma_values,
        residual_cov,
        eta,
        semantics,
    )
}

#[cfg(test)]
mod conditional_obs_terms_tests {
    use super::*;

    /// #2266: the per-record terms sum to the FOCEI scorer's data term. At η̂ = 0, `H = 0`
    /// the scorer reduces to `½(data_ll + bloq_term)`; the SDE `p_obs` must be in both.
    #[test]
    fn conditional_terms_sum_to_the_focei_data_term_of_an_sde_model() {
        let model = crate::parser::model_parser::parse_model_string(
            r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[diffusion]
  central ~ 0.5
[error_model]
  DV ~ additive(ADD)
"#,
        )
        .expect("parse");
        assert!(model.is_sde(), "premise: an SDE model");
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0],
            observations: vec![1.6, 1.5, 1.2, 0.9],
            obs_cmts: vec![1; 4],
            cens: vec![0; 4],
            ..Default::default()
        };
        let p = &model.default_params;
        let eta = DVector::zeros(model.n_eta);
        let ipreds = model_predictions(&model, &subject, &p.theta, eta.as_slice());
        let terms = conditional_obs_neg2ll_terms(
            &model,
            &subject,
            &p.theta,
            &ipreds,
            eta.as_slice(),
            &p.sigma.values,
            model.residual_cov_at(p),
        );
        assert!(terms.iter().all(|t| t.is_finite()), "{terms:?}");
        let p_obs = ekf_p_obs(&model, &subject, &p.theta, eta.as_slice(), &p.sigma.values);
        assert!(p_obs.iter().any(|&v| v > 1e-3), "premise: the process noise is live: {p_obs:?}");
        let scorer = foce_subject_nll_cov(
            &model,
            &subject,
            &p.theta,
            &eta,
            &DMatrix::zeros(subject.observations.len(), model.n_eta),
            &p.omega,
            &p.sigma.values,
            model.residual_cov_at(p),
            true,
        );
        let sum: f64 = terms.iter().sum();
        assert!(
            (sum - 2.0 * scorer).abs() <= 1e-12 * sum.abs().max(1.0),
            "Σ terms {sum} vs scorer data term {}",
            2.0 * scorer
        );
    }
}

// ── #2440: the fixed-η objective scores a negative prediction as written ──
#[cfg(test)]
mod negative_prediction_tests {
    use super::*;

    /// The algebraic baseline-minus-Emax model of #2440: additive error, and a
    /// prediction `E0 − EMAX·t/(ET50 + t)` that goes negative after `t ≈ 0.25`.
    fn negative_prediction_model() -> CompiledModel {
        crate::parser::model_parser::parse_model_string(
            r"
[parameters]
  theta TVE0(1.0, -50.0, 100.0)
  theta TVEMAX(5.0, 0.1, 100.0)
  theta TVET50(1.0, 0.01, 100.0)
  omega ETA_E0 ~ 0.25
  sigma ADD ~ 0.25 (variance)

[individual_parameters]
  E0   = TVE0 + ETA_E0
  EMAX = TVEMAX
  ET50 = TVET50

[structural_model]
  EFF = EMAX * TIME / (ET50 + TIME)
  y   = E0 - EFF

[error_model]
  DV ~ additive(ADD)
",
        )
        .expect("#2440 negative-prediction model parses")
    }

    /// One subject at `θ = (1, 5, 1)`, `η = 0`, every prediction negative, with
    /// `y = f + r` for a fixed residual vector `r` — the fixed-parameter check of #2440.
    fn negative_prediction_subject() -> (Subject, Vec<f64>) {
        let times = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
        let resid = [0.3, -0.2, 0.1, -0.4, 0.25, 0.15];
        let f: Vec<f64> = times.iter().map(|&t| 1.0 - 5.0 * t / (1.0 + t)).collect();
        let n = times.len();
        let subject = Subject {
            id: "neg".into(),
            obs_times: times.to_vec(),
            observations: f.iter().zip(resid).map(|(f, r)| f + r).collect(),
            obs_cmts: vec![1; n],
            cens: vec![0; n],
            ..Default::default()
        };
        (subject, f)
    }

    /// #2440: every fixed-η Gaussian data term (`obs_nll_subject_into`,
    /// `obs_nll_subject_from_preds`, `gaussian_obs_nll_scaled`) scores a negative prediction
    /// as written, equal to the hand sum and to `individual_nll` minus `½ log|Ω|`. Before the
    /// fix: 80.683110594 vs hand −3.388883083. Measured error 0 (≤ 1e-15 for
    /// `individual_nll`); 1e-12 is headroom for summation order.
    #[test]
    fn fixed_eta_objective_scores_a_negative_prediction_as_written() {
        let model = negative_prediction_model();
        let (subject, f_hand) = negative_prediction_subject();
        let theta = [1.0, 5.0, 1.0];
        let sigma = [0.5_f64];
        let eta = [0.0];

        let preds = pk::compute_predictions_with_tv(&model, &subject, &theta, &eta);
        assert!(
            preds.iter().all(|&f| f < 0.0),
            "fixture must keep every prediction negative, or the floor is not exercised: {preds:?}"
        );
        for (got, want) in preds.iter().zip(&f_hand) {
            assert!((got - want).abs() < 1e-12, "prediction {got} vs hand {want}");
        }

        let s2 = sigma[0] * sigma[0];
        let hand: f64 = subject
            .observations
            .iter()
            .zip(&f_hand)
            .map(|(y, f)| 0.5 * s2.ln() + 0.5 * (y - f).powi(2) / s2)
            .sum();
        assert!(
            (hand - (-3.388883083)).abs() < 1e-9,
            "hand sum drifted from the #2440 value: {hand}"
        );

        let mut scratch = pk::EventPkParams::with_capacity_for(&subject);
        let cov = ResidualCov::block_sigma(&[]);
        let into = obs_nll_subject_into(&model, &subject, &theta, &sigma, cov, &eta, &mut scratch);
        let from_preds =
            obs_nll_subject_from_preds(&model, &subject, &preds, &theta, &sigma, cov, &eta);
        let keys = model.error_spec.obs_keys(&subject);
        let scaled = crate::stats::residual_error::gaussian_obs_nll_scaled(
            &model.error_spec,
            &keys,
            &subject.observations,
            &preds,
            &sigma,
            &[],
            1.0,
            None,
            None,
        );
        let omega = &model.default_params.omega;
        let foce =
            individual_nll(&model, &subject, &theta, &eta, omega, &sigma) - 0.5 * omega.log_det;

        for (name, got) in [
            ("obs_nll_subject_into", into),
            ("obs_nll_subject_from_preds", from_preds),
            ("gaussian_obs_nll_scaled", scaled),
            ("individual_nll − ½log|Ω|", foce),
        ] {
            assert!(
                (got - hand).abs() < 1e-12,
                "{name} = {got}, hand sum = {hand} (the 1e-12 prediction floor gives 80.683)"
            );
        }
    }

    /// #2440: `f64::max` discards NaN, so the old prediction floor scored a NaN
    /// prediction as `1e-12` — a finite, plausible NLL (≈ 0.5·log σ² + 0.5·y²/σ²,
    /// order 10 here). With the floor gone the NaN must not leak out either; it maps to
    /// the FOCE family's `1e20` sentinel. Asserts the magnitude, since a NaN-scored-as-
    /// floor regression is just as finite as a correct score.
    #[test]
    fn fixed_eta_objective_maps_a_nan_prediction_to_the_sentinel() {
        let model = negative_prediction_model();
        let (subject, f_hand) = negative_prediction_subject();
        let theta = [1.0, 5.0, 1.0];
        let sigma = [0.5_f64];
        let eta = [0.0];
        let cov = ResidualCov::block_sigma(&[]);

        let clean =
            obs_nll_subject_from_preds(&model, &subject, &f_hand, &theta, &sigma, cov, &eta);
        assert!(clean.abs() < 100.0, "clean control must score normally: {clean}");

        let mut preds = f_hand.clone();
        preds[2] = f64::NAN;
        let nll = obs_nll_subject_from_preds(&model, &subject, &preds, &theta, &sigma, cov, &eta);
        assert!(
            nll >= SUBJECT_SENTINEL_NLL,
            "a NaN prediction must hit the 1e20 sentinel, not score as a finite subject; got {nll}"
        );
    }

    /// #2440 review: `gaussian_obs_nll_scaled` — the perturbed side of the `ruv_mult`
    /// θ gradient — must map a NaN prediction to the same sentinel as
    /// `obs_nll_subject_from_preds`, not return NaN (which a finite base minus it would
    /// spread through the gradient). Clean control scores normally.
    #[test]
    fn gaussian_obs_nll_scaled_maps_a_nan_prediction_to_the_sentinel() {
        let model = negative_prediction_model();
        let (subject, f_hand) = negative_prediction_subject();
        let keys = model.error_spec.obs_keys(&subject);
        let score = |preds: &[f64]| {
            crate::stats::residual_error::gaussian_obs_nll_scaled(
                &model.error_spec,
                &keys,
                &subject.observations,
                preds,
                &[0.5],
                &[],
                1.0,
                None,
                None,
            )
        };
        let clean = score(&f_hand);
        assert!(clean.abs() < 100.0, "clean control must score normally: {clean}");
        let mut preds = f_hand.clone();
        preds[4] = f64::NAN;
        let nll = score(&preds);
        assert!(
            nll >= SUBJECT_SENTINEL_NLL,
            "NaN prediction must hit the 1e20 sentinel; got {nll}"
        );
    }
}

#[cfg(test)]
mod quad_form_tests {
    use super::quad_form;
    use nalgebra::{DMatrix, DVector};

    /// #2549 W1: `quad_form` is the allocating `v.dot(&(M * &v))` bit for bit, on dense
    /// symmetric matrices of every size the estimators use, with values spanning many
    /// magnitudes so a different summation order would show.
    #[test]
    fn quad_form_is_the_allocating_expression_bit_for_bit() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            (u - 0.5) * 10f64.powi(((state >> 3) % 7) as i32 - 3)
        };
        for n in 1..=24 {
            for _ in 0..20 {
                let a = DMatrix::from_fn(n, n, |_, _| next());
                let m = &a * a.transpose();
                let v: Vec<f64> = (0..n).map(|_| next()).collect();
                let dv = DVector::from_column_slice(&v);
                let want = dv.dot(&(&m * &dv));
                assert_eq!(quad_form(&m, &v).to_bits(), want.to_bits(), "n = {n}");
            }
        }
    }
}
