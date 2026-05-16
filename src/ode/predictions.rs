//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.
//!
//! Infusion doses (`rate > 0`) are handled by breaking the timeline at the
//! infusion's end time and adding `+rate` to the corresponding compartment's
//! derivative for the duration of the infusion via an RHS wrapper.

use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::{DoseEvent, PkParams, Subject, PK_IDX_LAGTIME};
use std::collections::HashMap;

/// Epsilon used to decide whether an infusion fully spans a segment.
/// Break times are constructed to coincide with infusion start/end so any
/// non-degenerate segment is either fully inside or fully outside each
/// infusion window — this tolerance only guards float-equality on the bound.
const INFUSION_EPS: f64 = 1e-12;

/// `is_infusion()` only checks `rate > 0`, but a degenerate row with
/// `rate > 0 && amt <= 0` (or NaN) yields `duration = amt/rate <= 0`
/// (or NaN). Treating those as infusions would push an infusion-end
/// break that sorts before the dose itself, and NaN would panic the
/// break-time sort. Such rows fall back to the bolus branch instead
/// (a zero/negative bolus update — visible, not silently dropped).
fn is_real_infusion(d: &DoseEvent) -> bool {
    d.is_infusion() && d.duration > 0.0 && d.duration.is_finite()
}

/// Returns `(cmt_idx_0based, rate)` for every infusion that is active
/// throughout the closed segment `[t_start, t_end]`. By construction of the
/// break-time list (every infusion start and end is a break time), each
/// infusion is either fully active or fully inactive across a segment.
///
/// `dose_lagtimes[k]` shifts dose `k`'s active window. Parallel to `doses`.
/// An empty slice means "no lagtime" (all zeros).
fn active_infusions(
    doses: &[DoseEvent],
    t_start: f64,
    t_end: f64,
    dose_lagtimes: &[f64],
) -> Vec<(usize, f64)> {
    doses
        .iter()
        .enumerate()
        .filter(|(k, d)| {
            let lag = dose_lagtimes.get(*k).copied().unwrap_or(0.0);
            is_real_infusion(d)
                && d.time + lag <= t_start + INFUSION_EPS
                && d.time + lag + d.duration >= t_end - INFUSION_EPS
        })
        .map(|(_, d)| (d.cmt.saturating_sub(1), d.rate))
        .collect()
}

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
    /// Index of the observable compartment (0-based) for DV
    pub obs_cmt_idx: usize,
    /// Per-state diagonal process-noise variances (σ²_w,i) for SDE / EKF.
    /// Length must equal `n_states` when non-empty; empty means standard ODE
    /// (no diffusion). Declared via `[diffusion]` block as `state ~ variance`,
    /// analogous to sigma/omega notation. Updated each outer iteration as
    /// diffusion thetas are re-estimated.
    pub diffusion_var: Vec<f64>,
}

/// Compute ODE-based predictions for a single subject.
///
/// `pk_params_flat` is a flat array of PK parameters passed to the RHS function.
/// Dose events are handled as state discontinuities between integration segments.
pub fn ode_predictions(ode: &OdeSpec, pk_params_flat: &[f64], subject: &Subject) -> Vec<f64> {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = OdeSolverOptions::default();

    let mut u = vec![0.0; n];
    let mut predictions = vec![f64::NAN; n_obs];

    // Lagtime shifts the effective start (and end) of every dose record.
    // Default 0.0 when not declared, so existing models behave identically.
    let lagtime = pk_params_flat.get(PK_IDX_LAGTIME).copied().unwrap_or(0.0);
    // Per-dose lagtimes for `active_infusions` — uniform for the no-TV
    // path (lagtime is constant across doses).
    let dose_lagtimes: Vec<f64> = vec![lagtime; subject.doses.len()];

    // Build obs_time → index map
    let obs_map: HashMap<u64, usize> = subject
        .obs_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t.to_bits(), i))
        .collect();

    // Break timeline at lagtime-shifted dose times — and, for infusions,
    // at lagtime-shifted infusion-end times too, so each segment is
    // either fully inside or fully outside every infusion window.
    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time + lagtime);
        if is_real_infusion(dose) {
            break_times.push(dose.time + lagtime + dose.duration);
        }
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply bolus doses at t_start. Infusions don't add to state
        // instantaneously — they're injected as a derivative term inside
        // the integrator (see `active_infusions` + wrapped RHS below).
        for dose in &subject.doses {
            if (dose.time + lagtime - t_start).abs() < 1e-12 && !is_real_infusion(dose) {
                // dose.cmt is 1-based; state indices are 0-based
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n {
                    u[cmt_idx] += dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(&obs_idx) = obs_map.get(&t_start.to_bits()) {
            predictions[obs_idx] = u[ode.obs_cmt_idx];
        }

        // Observation times in this segment (t_start < t <= t_end)
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        // Always include t_end so u is updated for next segment
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Integrate. If any infusions are active in this segment, wrap
        // the user RHS so it adds `+rate` to each infusion's compartment.
        let active = active_infusions(&subject.doses, t_start, t_end, &dose_lagtimes);
        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            (ode.rhs)(y, p, t, dy);
            for &(cmt_idx, rate) in &active {
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            }
        };
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            pk_params_flat,
            &saveat,
            &opts,
        );

        // Extract predictions and update state
        for pt in &sol {
            if let Some(&obs_idx) = obs_map.get(&pt.t.to_bits()) {
                predictions[obs_idx] = pt.u[ode.obs_cmt_idx];
            }
        }

        // State at end of segment
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    // Clamp negatives
    for p in &mut predictions {
        if *p < 0.0 || p.is_nan() {
            *p = 0.0;
        }
    }

    predictions
}

/// ODE-based predictions with per-event PK parameters (time-varying-covariate
/// aware). Walks the merged dose+obs+pk-only timeline, integrating each
/// segment `[cur_t, t_event]` with the PK params evaluated at `t_event` —
/// the NONMEM end-of-interval / current-record convention (`$PK` runs at
/// every record, then ADVAN propagates to it). A covariate that changes
/// at an event row (dose, obs, or EVID=2) is therefore consumed by the
/// segment terminating at that record.
///
/// The non-TV `ode_predictions` is preserved as a fast path; this function
/// is only invoked from the dispatcher when `subject.has_tv_covariates()`.
///
/// Infusions (`rate > 0`) break the timeline at the infusion's end and are
/// added to the wrapped RHS for any segment they fully span. The
/// infusion-end break carries no NONMEM record, so it doesn't update the
/// "current PK" used to integrate subsequent segments.
pub fn ode_predictions_event_driven(
    ode: &OdeSpec,
    subject: &Subject,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = OdeSolverOptions::default();

    let mut u = vec![0.0_f64; n];
    let mut predictions = vec![f64::NAN; n_obs];

    if n_obs == 0 {
        return predictions;
    }

    // Build merged event timeline. Tie-break at the same time:
    //   dose < pk-only < obs < infusion-end
    // — matches the analytical event-driven path for dose/pk-only/obs.
    // Infusion-end sorts last so an obs at the same time as the end of
    // an infusion is recorded with the infusion still contributing
    // (state is continuous; the ordering only affects which segments
    // include the rate in their active set on the next iteration).
    #[derive(Clone, Copy)]
    enum Kind {
        Dose,
        PkOnly,
        Obs,
        InfusionEnd,
    }
    fn kind_order(k: Kind) -> u8 {
        match k {
            Kind::Dose => 0,
            Kind::PkOnly => 1,
            Kind::Obs => 2,
            Kind::InfusionEnd => 3,
        }
    }
    let n_infusion_ends = subject.doses.iter().filter(|d| is_real_infusion(d)).count();
    let mut timeline: Vec<(f64, Kind, usize)> = Vec::with_capacity(
        subject.doses.len() + n_obs + subject.pk_only_times.len() + n_infusion_ends,
    );
    // Per-dose lagtimes from the per-event PK snapshot for the dose.
    let dose_lagtimes: Vec<f64> = pk_at_dose.iter().map(|p| p.lagtime()).collect();
    for (k, d) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[k];
        timeline.push((d.time + lag, Kind::Dose, k));
        if is_real_infusion(d) {
            timeline.push((d.time + lag + d.duration, Kind::InfusionEnd, k));
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        timeline.push((t, Kind::Obs, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        timeline.push((t, Kind::PkOnly, m));
    }
    timeline.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| kind_order(a.1).cmp(&kind_order(b.1)))
    });

    let mut cur_t = timeline[0].0;
    // Most-recent NONMEM record's PK params, used to integrate segments
    // ending at an infusion-end (which is not a record and carries no PK).
    let mut last_pk: PkParams = PkParams::default();

    for &(t_event, kind, idx) in &timeline {
        // PK params for the segment [cur_t, t_event] are evaluated AT
        // t_event (NONMEM end-of-interval / current-record convention —
        // `$PK runs at every record, then ADVAN propagates to it`).
        // Infusion-end is not a record: reuse the previous segment's PK.
        let pk_now: PkParams = match kind {
            Kind::Dose => pk_at_dose[idx],
            Kind::Obs => pk_at_obs[idx],
            Kind::PkOnly => pk_at_pk_only[idx],
            Kind::InfusionEnd => last_pk,
        };

        if t_event > cur_t {
            // Wrap the user RHS so any infusion fully spanning
            // [cur_t, t_event] contributes `+rate` to its compartment.
            let active = active_infusions(&subject.doses, cur_t, t_event, &dose_lagtimes);
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                for &(cmt_idx, rate) in &active {
                    if cmt_idx < dy.len() {
                        dy[cmt_idx] += rate;
                    }
                }
            };
            let saveat = vec![t_event];
            let sol = solve_ode(
                &wrapped_rhs,
                &u,
                (cur_t, t_event),
                &pk_now.values,
                &saveat,
                &opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }

        match kind {
            Kind::Dose => {
                let d = &subject.doses[idx];
                // Boluses: add amt to state. Infusions: no instantaneous
                // change — handled via the wrapped RHS for segments inside
                // [d.time, d.time + d.duration].
                if !is_real_infusion(d) {
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n {
                        u[cmt_idx] += d.amt;
                    }
                }
                last_pk = pk_now;
            }
            Kind::Obs => {
                let v = u[ode.obs_cmt_idx];
                predictions[idx] = if v.is_nan() || v < 0.0 { 0.0 } else { v };
                last_pk = pk_now;
            }
            Kind::PkOnly => {
                // EVID=2: $PK ran at this record but compartment state is
                // unchanged. The new pk is consumed by the next segment's
                // integration via the loop-top `pk_now` lookup.
                last_pk = pk_now;
            }
            Kind::InfusionEnd => {
                // Not a NONMEM record: no state update, no PK update —
                // only purpose is to break the timeline so the next
                // segment's `active_infusions` excludes this infusion.
            }
        }
    }

    predictions
}

/// EKF-based predictions with an explicit diffusion_var slice (bypasses
/// `ode_spec.diffusion_var`). Used by the likelihood path to supply the
/// current theta-derived diffusion variances without mutating the model.
pub fn ode_predictions_ekf_with_diffusion(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    diffusion_var: &[f64],
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    let ipred_plain = ode_predictions(ode, pk_params_flat, subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        ode.obs_cmt_idx,
        diffusion_var,
        pk_params_flat,
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

/// EKF-based predictions for a subject with an SDE model.
///
/// Wraps `solve_ekf`, handling the residual variance `r_obs` needed for the
/// Kalman update step. Returns `(ipred, p_obs)` where `p_obs[j]` is the
/// EKF state covariance at the observable compartment just before assimilating
/// observation `j`. Callers add `p_obs[j]` to the residual variance to form
/// `V_total = p_obs[j] + V_residual`.
///
/// `r_obs_fn` computes the scalar residual variance for each observation given
/// the predicted value — this feeds the Kalman update, keeping the covariance
/// estimate numerically stable. It does NOT affect the returned `p_obs` values
/// (those are pre-update, i.e. the purely process-noise contribution).
// Not currently called from outside this module — superseded by
// `ode_predictions_ekf_with_diffusion` which accepts an explicit diffusion_var.
#[allow(dead_code)]
pub fn ode_predictions_ekf(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    // Compute per-observation R for the Kalman update from a standard ODE pass.
    // Using per-observation R is correct for proportional and combined error models.
    let ipred_plain = ode_predictions(ode, pk_params_flat, subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        ode.obs_cmt_idx,
        &ode.diffusion_var,
        pk_params_flat,
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A. RHS reads CL,V from pk_params_flat.
    fn one_cpt_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            obs_cmt_idx: 0,
            diffusion_var: Vec::new(),
        }
    }

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p
    }

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        }
    }

    #[test]
    fn ode_event_driven_matches_constant_path_when_pk_constant() {
        // Equivalence: when the per-event PK params are all the same, the
        // event-driven ODE path must agree with the existing single-snapshot
        // path. This is the "no TV covariates" sanity check.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &subj);
        let event_driven = ode_predictions_event_driven(&ode, &subj, &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            // ODE solver tolerance is ~1e-4 relative — a tighter equality
            // would over-constrain RK45.
            assert_relative_eq!(*b, *e, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_event_driven_picks_up_changing_cl() {
        // Same shape as the analytical TV test: CL doubles between two doses.
        // End-of-interval / NONMEM convention — each segment uses the PK
        // params at the record being arrived at:
        //   [0, t_obs1=5]: uses pk at obs1 = pk_low → ke = 0.05
        //   [5, t_dose2=10]: uses pk at dose2 = pk_high → ke = 0.10
        //   [10, t_obs2=12]: uses pk at obs2 = pk_high → ke = 0.10
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![5.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low, pk_high];
        let pk_obs = vec![pk_low, pk_high];
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions_event_driven(&ode, &subj, &pk_dose, &pk_obs, &[]);

        // [0, 5] uses pk_low (pk at obs1): A(5) = 1000 * exp(-0.05*5) ≈ 778.80
        let a5 = 1000.0 * (-0.05f64 * 5.0).exp();
        assert_relative_eq!(preds[0], a5, epsilon = 1e-3, max_relative = 1e-4);

        // [5, 10] uses pk_high (pk at dose2): ke=0.10 for 5h.
        //   A(10⁻) = A(5) * exp(-0.10*5) = 778.80 * 0.6065 ≈ 472.37
        // After dose2: A(10⁺) = 472.37 + 1000 = 1472.37.
        // [10, 12] uses pk_high (pk at obs2): A(12) = 1472.37 * exp(-0.20) ≈ 1205.49
        let a10_minus = a5 * (-0.10f64 * 5.0).exp();
        let a10_plus = a10_minus + 1000.0;
        let a12 = a10_plus * (-0.20f64).exp();
        assert_relative_eq!(preds[1], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    /// 1-cpt oral ODE: dA1/dt = -ka·A1, dA2/dt = ka·A1 - ke·A2.
    /// Used to test infusion into the depot compartment (cmt=1).
    fn one_cpt_oral_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ka = p[crate::types::PK_IDX_KA];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ka * y[0];
                dy[1] = ka * y[0] - ke * y[1];
            }),
            n_states: 2,
            state_names: vec!["depot".into(), "central".into()],
            obs_cmt_idx: 1,
            diffusion_var: Vec::new(),
        }
    }

    #[test]
    fn ode_infusion_one_cpt_iv_matches_closed_form() {
        // 1-cpt IV infusion. Closed form during infusion:
        //   A(t) = (R/ke) · (1 - exp(-ke·t))
        // and after end-of-infusion T:
        //   A(t) = A(T) · exp(-ke·(t-T))
        // Verifies that the wrapped-RHS path produces the right shape.
        let rate = 100.0;
        let amt = 1000.0; // duration = 10 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![5.0, 10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0); // ke = 0.0625
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);

        // During infusion [0, 10]
        let a5 = (rate / ke) * (1.0 - (-ke * 5.0).exp());
        let a10 = (rate / ke) * (1.0 - (-ke * 10.0).exp());
        // After end-of-infusion
        let a15 = a10 * (-ke * 5.0).exp();
        let a20 = a10 * (-ke * 10.0).exp();

        assert_relative_eq!(preds[0], a5, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a15, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_event_driven_infusion_matches_constant_pk_path() {
        // Same infusion-only subject, run through both paths with identical
        // per-event PK params. Verifies the event-driven path's
        // InfusionEnd handling agrees with the simple-timeline path.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![3.0, 7.0, 10.0, 14.0, 20.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &subj);
        let event_driven = ode_predictions_event_driven(&ode, &subj, &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            assert_relative_eq!(*b, *e, epsilon = 1e-3, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_overlapping_infusions_sum_rates() {
        // Two infusions overlap on [2, 6] for a combined rate of 200,
        // then both end at t=6. After t=6, plain elimination.
        //   inf1: t∈[0,6], rate=100
        //   inf2: t∈[2,6], rate=100
        let doses = vec![
            DoseEvent::new(0.0, 600.0, 1, 100.0, false, 0.0),
            DoseEvent::new(2.0, 400.0, 1, 100.0, false, 0.0),
        ];
        let obs_times = vec![2.0, 4.0, 6.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);

        // [0, 2]: rate=100, A(0)=0 → A(t) = (100/ke)·(1 - exp(-ke·t))
        let a2 = (100.0_f64 / ke) * (1.0 - (-ke * 2.0).exp());
        // [2, 6]: rate=200, A0=a2
        //   A(t) = (200/ke) + (A0 - 200/ke) · exp(-ke·(t-2))
        let r_over_ke = 200.0_f64 / ke;
        let a4 = r_over_ke + (a2 - r_over_ke) * (-ke * 2.0).exp();
        let a6 = r_over_ke + (a2 - r_over_ke) * (-ke * 4.0).exp();
        // [6, ∞]: rate=0
        let a12 = a6 * (-ke * 6.0).exp();

        assert_relative_eq!(preds[0], a2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a6, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_then_bolus() {
        // Infusion [0, 10] followed by a bolus at t=15. Observation at
        // the bolus time should record state AFTER the bolus is applied,
        // matching the existing bolus convention.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0), // infusion, ends at 10
            DoseEvent::new(15.0, 500.0, 1, 0.0, false, 0.0),   // bolus
        ];
        let obs_times = vec![10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);

        let a10 = (100.0_f64 / ke) * (1.0 - (-ke * 10.0).exp());
        let a15_pre = a10 * (-ke * 5.0).exp();
        let a15_post = a15_pre + 500.0;
        let a20 = a15_post * (-ke * 5.0).exp();

        assert_relative_eq!(preds[0], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a15_post, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_into_oral_depot() {
        // Infusion into depot (cmt=1) of a 1-cpt oral model. Verifies
        // that the wrapped RHS adds `+rate` to the correct compartment
        // (depot index 0), not central (index 1). For the depot alone
        // the closed form is decoupled from ke:
        //   A1(t) during infusion = (R/ka)·(1 - exp(-ka·t))
        //   A1(t) after end T     = A1(T) · exp(-ka·(t-T))
        // Re-use the oral ODE spec but observe the depot.
        let mut ode = one_cpt_oral_ode_spec();
        ode.obs_cmt_idx = 0;

        let rate = 50.0;
        let amt = 200.0; // duration = 4 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[crate::types::PK_IDX_KA] = 1.0;
        let ka = 1.0_f64;

        let preds = ode_predictions(&ode, &pk.values, &subj);

        let depot_2 = (rate / ka) * (1.0 - (-ka * 2.0).exp());
        let depot_4 = (rate / ka) * (1.0 - (-ka * 4.0).exp());
        let depot_8 = depot_4 * (-ka * 4.0).exp();

        assert_relative_eq!(preds[0], depot_2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], depot_4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], depot_8, epsilon = 1e-2, max_relative = 1e-4);
    }

    // Degenerate input guards: `rate > 0` alone is insufficient to mark a
    // dose as an infusion — `duration = amt/rate` must also be > 0 and
    // finite. Otherwise:
    //   - `amt < 0` would push an infusion-end break time *before* the
    //     dose, scrambling the segmented integration order.
    //   - `amt = NaN` would make `partial_cmp` return None and panic
    //     the break-time sort.
    //   - In both cases, the bolus branch is skipped (because
    //     `is_infusion()` is true on rate alone), so the dose silently
    //     disappears from the prediction.
    // `is_real_infusion` falls back to the bolus path for these rows.

    #[test]
    fn ode_degenerate_zero_amt_with_positive_rate_falls_back_to_bolus() {
        // amt=0, rate>0 → duration=0. Treated as a (no-op) bolus.
        // Result must match "no dose at all".
        let doses = vec![DoseEvent::new(0.0, 0.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);

        assert_eq!(preds, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn ode_degenerate_negative_amt_with_positive_rate_does_not_break_ordering() {
        // amt<0, rate>0 → duration<0. Pre-fix, the infusion-end break time
        // would sort *before* the dose, producing nonsense segments and
        // (silently) zero output because the bolus branch was skipped.
        // Post-fix this is treated as a bolus with negative amt — at least
        // visible to the caller.
        let doses = vec![DoseEvent::new(0.0, -10.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        // Must not panic; the negative bolus update is clamped to 0 by
        // the negative-prediction guard in `ode_predictions`.
        let preds = ode_predictions(&ode, &pk.values, &subj);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn ode_degenerate_nan_amt_with_positive_rate_does_not_panic() {
        // amt=NaN, rate>0 → duration=NaN. Pre-fix, sort_by(partial_cmp).unwrap()
        // would panic on the break-time vec. Post-fix the row falls through
        // to the bolus branch and the panic is avoided.
        let doses = vec![DoseEvent::new(0.0, f64::NAN, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn ode_iv_bolus_with_lagtime_shifts_curve() {
        // 1-cpt IV bolus integrated via ODE: with lagtime=2.0 and dose at
        // t_dose=0, the central-amount state should be 0 until t=2 (the
        // lagged dose arrival), then decay as if dose-time were 2.
        // (`one_cpt_ode_spec` observes the amount A(t), not A/V.)
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 3.0, 6.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[PK_IDX_LAGTIME] = 2.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &subj);

        // At t=1, dose has not yet arrived (lagtime=2). State stays 0.
        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-10);

        // At t=3, effective elapsed time since dose is 1.0.
        // A(1) = Amt * exp(-ke * 1)
        let ke = 5.0_f64 / 80.0;
        let expected_3 = 1000.0_f64 * (-ke * 1.0).exp();
        assert_relative_eq!(preds[1], expected_3, epsilon = 1e-4, max_relative = 1e-4);

        // At t=6, effective elapsed time is 4.0.
        let expected_6 = 1000.0_f64 * (-ke * 4.0).exp();
        assert_relative_eq!(preds[2], expected_6, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_with_lagtime_shifts_break_times_and_active_window() {
        // Direct test of the ODE infusion + lagtime path that the analytical
        // superposition test alone doesn't cover. Amt=100, rate=100 ⇒
        // duration=1.0; with lagtime=0.5, the active-infusion window runs
        // [2.5, 3.5] rather than [2.0, 3.0]. Compare against an equivalent
        // unlagged dose starting at 2.5 — predictions at matched observation
        // offsets should agree to ODE tolerance.
        let dose_lag = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
        assert!(dose_lag.is_infusion() && dose_lag.duration > 0.0);
        let subj_lag = make_subject(vec![dose_lag], vec![2.0, 3.0, 4.0]);
        let mut pk_lag = pk_one(5.0, 80.0);
        pk_lag.values[PK_IDX_LAGTIME] = 0.5;

        // Reference: dose shifted at the data level, no lagtime applied.
        let dose_ref = DoseEvent::new(2.5, 100.0, 1, 100.0, false, 0.0);
        let subj_ref = make_subject(vec![dose_ref], vec![2.0, 3.0, 4.0]);
        let pk_ref = pk_one(5.0, 80.0);

        let ode = one_cpt_ode_spec();
        let preds_lag = ode_predictions(&ode, &pk_lag.values, &subj_lag);
        let preds_ref = ode_predictions(&ode, &pk_ref.values, &subj_ref);

        // Observation before lagged infusion start: zero.
        assert_relative_eq!(preds_lag[0], 0.0, epsilon = 1e-10);

        // Observations during and after the lagged infusion: must match the
        // reference where the dose was shifted at the dataset level.
        assert_relative_eq!(
            preds_lag[1],
            preds_ref[1],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        assert_relative_eq!(
            preds_lag[2],
            preds_ref[2],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }
}
