pub mod event_driven;
pub mod one_compartment;
pub mod three_compartment;
pub mod two_compartment;

use crate::types::{CompiledModel, DoseEvent, PkModel, PkParams, Subject};

pub use one_compartment::*;
pub use three_compartment::*;
pub use two_compartment::*;

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

    if subject.has_tv_covariates() {
        for k in 0..subject.doses.len() {
            out.dose
                .push((model.pk_param_fn)(theta, eta, subject.dose_cov(k)));
        }
        for j in 0..subject.obs_times.len() {
            out.obs
                .push((model.pk_param_fn)(theta, eta, subject.obs_cov(j)));
        }
        for m in 0..subject.pk_only_times.len() {
            out.pk_only
                .push((model.pk_param_fn)(theta, eta, subject.pk_only_cov(m)));
        }
    } else {
        let p = (model.pk_param_fn)(theta, eta, &subject.covariates);
        // pk_only stays empty — see EventPkParams docstring.
        for _ in 0..subject.doses.len() {
            out.dose.push(p);
        }
        for _ in 0..subject.obs_times.len() {
            out.obs.push(p);
        }
    }
}

/// Predict concentration at a given time for a subject, summing contributions
/// from all prior doses (superposition principle).
///
/// `pk_params.lagtime()` shifts the effective start of every dose (bolus,
/// infusion, and oral) by that amount. For infusions the duration is
/// preserved — only the start (and therefore end) of the window shifts.
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
        }
    }
    conc.max(0.0)
}

/// Concentration contribution from a single dose at elapsed time tau
fn single_dose_concentration(pk_model: PkModel, dose: &DoseEvent, tau: f64, p: &PkParams) -> f64 {
    let cl = p.cl();
    let v = p.v();

    match pk_model {
        PkModel::OneCptIvBolus => one_cpt_iv_bolus(dose, tau, cl, v),
        PkModel::OneCptInfusion => one_cpt_infusion(dose, tau, cl, v),
        PkModel::OneCptOral => one_cpt_oral_f(dose, tau, cl, v, p.ka(), p.f_bio()),
        PkModel::TwoCptIvBolus => two_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2()),
        PkModel::TwoCptInfusion => two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2()),
        PkModel::TwoCptOral => two_cpt_oral_f(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio()),
        PkModel::ThreeCptIvBolus => {
            three_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
        }
        PkModel::ThreeCptInfusion => {
            three_cpt_infusion(dose, tau, cl, v, p.q(), p.v2(), p.q3(), p.v3())
        }
        PkModel::ThreeCptOral => three_cpt_oral_f(
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
        ),
    }
}

/// Compute predictions for all observation times of a subject.
/// Uses analytical equations for standard PK models, or delegates to ODE solver
/// when an OdeSpec is provided.
pub fn compute_predictions(pk_model: PkModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    subject
        .obs_times
        .iter()
        .map(|&t| predict_concentration(pk_model, &subject.doses, t, pk_params))
        .collect()
}

/// Compute predictions using ODE integration.
/// `pk_params_flat` is the flat parameter vector passed to the ODE RHS function.
pub fn compute_predictions_ode(
    ode_spec: &crate::ode::OdeSpec,
    subject: &Subject,
    pk_params_flat: &[f64],
) -> Vec<f64> {
    crate::ode::ode_predictions(ode_spec, pk_params_flat, subject)
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

    // ODE path.
    if let Some(ref ode) = model.ode_spec {
        if has_tv {
            compute_event_pk_params_into(model, subject, theta, eta, scratch);
            return crate::ode::ode_predictions_event_driven(
                ode,
                subject,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            );
        }
        let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
        return crate::ode::ode_predictions(ode, &pk.values, subject);
    }

    if has_tv && event_driven::supports_event_driven(model.pk_model) {
        compute_event_pk_params_into(model, subject, theta, eta, scratch);
        if let Some(sched) = schedule {
            return event_driven::event_driven_predictions_with_schedule(
                model.pk_model,
                subject,
                sched,
                &scratch.dose,
                &scratch.obs,
                &scratch.pk_only,
            );
        }
        return event_driven::event_driven_predictions(
            model.pk_model,
            subject,
            &scratch.dose,
            &scratch.obs,
            &scratch.pk_only,
        );
    }

    // No-TV fast path (or TV with unsupported model — see docstring).
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates);
    compute_predictions(model.pk_model, subject, &pk)
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

    #[test]
    fn test_superposition_single_dose() {
        let doses = vec![bolus_dose(0.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let c = predict_concentration(PkModel::OneCptIvBolus, &doses, 0.0, &pk);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_two_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(10.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let k: f64 = 10.0 / 100.0;

        // At t=10, first dose has decayed, second dose just given
        let c = predict_concentration(PkModel::OneCptIvBolus, &doses, 10.0, &pk);
        let expected = (1000.0_f64 / 100.0) * (-k * 10.0).exp() + 1000.0 / 100.0;
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_ignores_future_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(100.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);

        // At t=5, second dose hasn't happened yet
        let c_single =
            predict_concentration(PkModel::OneCptIvBolus, &[bolus_dose(0.0, 1000.0)], 5.0, &pk);
        let c_two = predict_concentration(PkModel::OneCptIvBolus, &doses, 5.0, &pk);
        assert_relative_eq!(c_single, c_two, epsilon = 1e-12);
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
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: cov_const,
            dose_covariates: dose_cov,
            obs_covariates: obs_cov,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }
    }

    /// Build a tiny analytical CompiledModel where CL = covariate `CR` *
    /// theta[0] (so we can prove pk_param_fn was called with the right
    /// covariate snapshot per event).
    fn cl_from_cr_model() -> crate::types::CompiledModel {
        use crate::types::{
            BloqMethod, CompiledModel, ErrorModel, GradientMethod, ModelParameters, OmegaMatrix,
            PkModel, SigmaVector,
        };
        CompiledModel {
            name: "cl_from_cr".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Additive,
            pk_param_fn: Box::new(|theta, _eta, cov| {
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
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["CR".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            indiv_param_names: Vec::new(),
            theta_transform: Vec::new(),
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
    fn test_compute_predictions_length() {
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0],
            observations: vec![0.0; 4],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIvBolus, &subject, &pk);
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
        let c_pre = predict_concentration(PkModel::OneCptInfusion, &doses, 0.6, &pk);
        assert_eq!(c_pre, 0.0);

        // (b) Mid-infusion (lagged): pre-lag conc at tau = t - (dose.time + lag) = t - 2.5.
        // Compare against unlagged infusion at tau via the same formula.
        let pk_nolag = make_pk_params(10.0, 100.0);
        let c_lag = predict_concentration(PkModel::OneCptInfusion, &doses, 2.6, &pk);
        let c_no = predict_concentration(
            PkModel::OneCptInfusion,
            &[DoseEvent::new(0.1, 100.0, 1, 100.0, false, 0.0)],
            0.2,
            &pk_nolag,
        );
        // Both probe an elapsed time of 0.1 into a 1h infusion of rate=100.
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-10);

        // (c) Post-infusion (lagged window ends at 3.5).
        let c_post = predict_concentration(PkModel::OneCptInfusion, &doses, 3.6, &pk);
        let c_post_nolag = predict_concentration(
            PkModel::OneCptInfusion,
            &[DoseEvent::new(0.0, 100.0, 1, 100.0, false, 0.0)],
            1.1,
            &pk_nolag,
        );
        assert_relative_eq!(c_post, c_post_nolag, epsilon = 1e-10);
    }
}
