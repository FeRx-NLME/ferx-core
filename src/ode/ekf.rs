//! Extended Kalman Filter for SDE-based ODE models.
//!
//! Given an Itô SDE  dX = f(X,t,θ) dt + diag(√σ²_w) dW  the EKF propagates
//! both the mean state (via the existing RK45 solver) and the state covariance
//! matrix P between events, then applies a scalar Kalman update at each
//! observation time.
//!
//! Covariance prediction (continuous-discrete EKF):
//!   P̃ = F·P·Fᵀ + Q·Δt
//! where F = ∂f/∂X (Jacobian, computed by central FD) and Q = diag(σ²_w).
//!
//! Measurement update at obs j (scalar observation of state obs_cmt):
//!   S   = P̃[c,c] + R_j          (innovation variance)
//!   K   = P̃[:,c] / S             (Kalman gain vector)
//!   P⁺  = (I - K·eₓᵀ) · P̃      (updated covariance)
//!
//! Only P[obs_cmt, obs_cmt] is returned per observation — the caller adds it
//! to the residual variance to form V_total.

use crate::dosing::is_real_infusion;
use crate::ode::predictions::active_infusions;
use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::DoseEvent;
use nalgebra::{DMatrix, DVector};

const FD_H: f64 = 1e-5;

/// Compute the Jacobian F = ∂f/∂X at state `u` by central finite differences.
fn fd_jacobian(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u: &[f64],
    p: &[f64],
    t: f64,
    n: usize,
) -> DMatrix<f64> {
    let mut jac = DMatrix::zeros(n, n);
    let mut u_fwd = u.to_vec();
    let mut u_bwd = u.to_vec();
    let mut df_fwd = vec![0.0; n];
    let mut df_bwd = vec![0.0; n];

    for j in 0..n {
        let h = FD_H * (1.0 + u[j].abs());
        u_fwd[j] = u[j] + h;
        u_bwd[j] = u[j] - h;
        rhs(&u_fwd, p, t, &mut df_fwd);
        rhs(&u_bwd, p, t, &mut df_bwd);
        for i in 0..n {
            jac[(i, j)] = (df_fwd[i] - df_bwd[i]) / (2.0 * h);
        }
        u_fwd[j] = u[j];
        u_bwd[j] = u[j];
    }
    jac
}

/// Propagate P over a segment [t0, t1] using the linearised covariance ODE.
///
/// We use a single Euler step on the Riccati equation:
///   dP/dt = F·P + P·Fᵀ + Q
/// with F and Q evaluated at the midpoint state. For typical PK segment
/// lengths (≤ 1 h between observations) and the slow variance dynamics this
/// is accurate to O(Δt²). A Runge-Kutta covariance propagation can be added
/// later if needed for long dosing intervals.
fn propagate_covariance(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    p_mat: &DMatrix<f64>,
    u_mid: &[f64],
    params: &[f64],
    t_mid: f64,
    dt: f64,
    q_diag: &[f64],
    n: usize,
) -> DMatrix<f64> {
    let f = fd_jacobian(rhs, u_mid, params, t_mid, n);
    let q = DMatrix::from_diagonal(&DVector::from_vec(q_diag.to_vec()));
    // Euler step: P_new = P + (F·P + P·Fᵀ + Q) · Δt
    let dp = &f * p_mat + p_mat * f.transpose() + q;
    let p_new = p_mat + dp * dt;
    // Symmetrise and clamp diagonal to stay positive semi-definite
    let p_sym = (&p_new + p_new.transpose()) * 0.5;
    // Clamp diagonal to ≥ 0
    let mut p_out = p_sym;
    for i in 0..n {
        if p_out[(i, i)] < 0.0 {
            p_out[(i, i)] = 0.0;
        }
    }
    p_out
}

/// Kalman update for a scalar observation of compartment `obs_cmt`.
///
/// Returns `(P_updated, p_obs_cmt)` where `p_obs_cmt` is P[obs_cmt, obs_cmt]
/// *before* the update — the component the caller adds to residual variance.
fn kalman_update(
    p_mat: &DMatrix<f64>,
    obs_cmt: usize,
    r_obs: f64,
    n: usize,
) -> (DMatrix<f64>, f64) {
    let p_cc = p_mat[(obs_cmt, obs_cmt)];
    let s = p_cc + r_obs; // innovation variance
    let p_obs = p_cc; // returned to caller before update

    if s <= 0.0 {
        return (p_mat.clone(), p_obs);
    }

    // Gain vector K = P[:,obs_cmt] / S
    let k: DVector<f64> = p_mat.column(obs_cmt).into_owned() / s;

    // Update: P⁺ = (I - K·eₒᵀ) · P
    let mut p_new = p_mat.clone();
    for i in 0..n {
        for j in 0..n {
            p_new[(i, j)] -= k[i] * p_mat[(obs_cmt, j)];
        }
    }
    // Symmetrise
    let p_sym = (&p_new + p_new.transpose()) * 0.5;
    (p_sym, p_obs)
}

/// One observation point returned by `solve_ekf`.
#[derive(Debug, Clone)]
pub struct EkfObsPoint {
    /// Predicted mean state value at the observable compartment.
    pub ipred: f64,
    /// EKF state covariance at the observable compartment (P[obs_cmt, obs_cmt])
    /// *before* assimilating this observation. Add to residual variance for V_total.
    pub p_obs: f64,
}

/// Propagate mean and covariance through a subject's dose+obs timeline.
///
/// `rhs`, `n_states`, `obs_cmt_idx` mirror `OdeSpec`. `diffusion_var` is the
/// diagonal of Q (length == n_states). `r_obs_vec` is the per-observation
/// measurement variance R (one entry per element of `obs_times`, in the same
/// order). Using per-observation R ensures the Kalman update is correct for
/// proportional and combined error models where R depends on the predicted value.
/// The returned `p_obs` values are the pre-update EKF covariance components and
/// are not inflated by R.
///
/// Dose events are handled identically to `ode_predictions`: boluses add to
/// state; infusions inject a rate term into the wrapped RHS. Covariance is
/// reset to zero at initial time and propagated forward from there.
#[allow(clippy::too_many_arguments)]
pub fn solve_ekf(
    rhs: &(dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync),
    n_states: usize,
    obs_cmt_idx: usize,
    diffusion_var: &[f64],
    pk_params_flat: &[f64],
    dose_attr_map: &crate::types::DoseAttrMap,
    initial_state: &[f64],
    doses: &[DoseEvent],
    obs_times: &[f64],
    r_obs_vec: &[f64],
    opts: OdeSolverOptions,
) -> Vec<EkfObsPoint> {
    use std::collections::HashMap;

    let n = n_states;
    let n_obs = obs_times.len();

    // Seed the EKF mean from the model's initial compartment amounts
    // (`init(state) = expr`); zeros for models without an init block. The
    // covariance still starts at zero — init sets the deterministic mean only.
    let mut u = if initial_state.len() == n {
        initial_state.to_vec()
    } else {
        vec![0.0f64; n]
    };
    let mut p_mat = DMatrix::zeros(n, n);
    let mut results = vec![
        EkfObsPoint {
            ipred: 0.0,
            p_obs: 0.0
        };
        n_obs
    ];

    let obs_map: HashMap<u64, usize> = obs_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t.to_bits(), i))
        .collect();

    // Build break times (same logic as ode_predictions)
    let t_last = obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in doses {
        break_times.push(dose.time);
        if is_real_infusion(dose) {
            break_times.push(dose.time + dose.duration);
        }
    }
    // Bioavailability resolved per dose compartment (`Fn`; issue #369), falling
    // back to the bare `F` slot. (The EKF path does not apply lagtime.)
    let dose_f_bio: Vec<f64> = doses
        .iter()
        .map(|d| dose_attr_map.f_bio(d.cmt_raw(), pk_params_flat))
        .collect();
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    // A non-finite break time makes the subject non-finite (#1189) — see
    // `ode::predictions::timeline_has_non_finite`. `results` is prefilled with the
    // caller's default point, so overwrite it with NaN ipreds rather than returning a
    // finite-looking filter pass built on a timeline that could not be ordered.
    if crate::ode::predictions::timeline_has_non_finite(&break_times) {
        // `fill`, not a per-field loop: one spelling of the struct, so a field added to
        // `EkfObsPoint` cannot be left at its default here while the others go NaN.
        results.fill(EkfObsPoint {
            ipred: f64::NAN,
            p_obs: f64::NAN,
        });
        return results;
    }

    // Apply-once mask (#1186): an infusion end (`dose.time + duration`) is a derived
    // break that can land within `EVENT_MATCH_TOL` of a later dose's own break, and
    // this loop rescans every dose at every break. The EKF path applies no lagtime, so
    // one mask (the arrival) covers it.
    let mut applied = vec![false; doses.len()];

    // Records read *at* the current break (#1226) — sorted once, hoisted, as on the ODE
    // engines.
    let obs_index = crate::ode::predictions::RecordIndex::new(obs_times);
    let mut boundary_obs: Vec<usize> = Vec::new();
    // Assimilate-once mask — the measurement analogue of the `applied` dose mask above.
    //
    // On the ODE engines a band write is an idempotent index assignment, so "the later write
    // wins" is the whole rule. Here it is **not** a write: each match runs a `kalman_update`
    // that mutates `p_mat`, so visiting the same measurement twice assimilates it twice and
    // returns an over-confident `p_obs` plus a distorted covariance for the rest of the
    // subject. Two ways that happens, and the first predates #1226: an observation exactly on
    // an interior break is saved by the segment ending there *and* read at that break as the
    // next `t_start`; and since the band is `1e-12` wide while `break_times` dedups at
    // `1e-15`, two breaks (a #1186 infusion-end / dose collision) can both claim one sample.
    let mut assimilated = vec![false; obs_times.len()];
    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply bolus doses at t_start (infusions enter via the wrapped RHS).
        for (di, dose) in doses.iter().enumerate() {
            if applied[di] {
                continue;
            }
            if (dose.time - t_start).abs() < crate::ode::predictions::EVENT_MATCH_TOL {
                applied[di] = true;
                if !is_real_infusion(dose) {
                    let cmt_idx = dose.cmt_idx();
                    if cmt_idx < n {
                        u[cmt_idx] += dose_f_bio[di] * dose.amt;
                    }
                }
            }
        }

        // Record obs read *at* t_start (after dose application) — the whole
        // `EVENT_MATCH_TOL` band, through the same helper as the ODE engines (#1226).
        obs_index.records_at_break(t_start, &mut boundary_obs);
        for i in 0..boundary_obs.len() {
            let obs_idx = boundary_obs[i];
            let v = u[obs_cmt_idx];
            let ipred = if v.is_nan() || v < 0.0 { 0.0 } else { v };
            if assimilated[obs_idx] {
                // The **covariance** must be assimilated once; the **mean** is still the
                // post-event one. An observation exactly on this break was saved by the
                // segment that ended here, pre-dose — blocking it outright would keep that
                // pre-dose `ipred` and reintroduce the very defect #1226 fixes (measured:
                // 68.73 instead of 168.73). A bolus shifts the state deterministically and
                // leaves `p_mat` untouched, so `p_obs` is the same either side of it and
                // only the mean needs the later write.
                results[obs_idx].ipred = ipred;
                continue;
            }
            let r = r_obs_vec.get(obs_idx).copied().unwrap_or(1.0);
            let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
            p_mat = p_new;
            let point = EkfObsPoint { ipred, p_obs };
            // Records sharing this exact time are one measurement instant: they take the
            // same `ipred`/`p_obs` from this single update rather than each running their
            // own — the collapsing the old `obs_map.get(&t_start.to_bits())` did, since that
            // map keeps one index per bit pattern. They are *filled* rather than dropped:
            // the band excludes them from the segment `saveat` too, so a dropped index would
            // keep its `EkfObsPoint { ipred: 0.0, p_obs: 0.0 }` prefill and feed a finite,
            // plausible zero into the likelihood.
            let bits = obs_times[obs_idx].to_bits();
            for &j in &boundary_obs {
                if !assimilated[j] && obs_times[j].to_bits() == bits {
                    results[j] = point.clone();
                    assimilated[j] = true;
                }
            }
        }

        let mut saveat: Vec<f64> = obs_times
            .iter()
            .filter(|&&t| crate::ode::predictions::reads_in_segment(t, t_start, t_end))
            .cloned()
            .collect();
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.total_cmp(b));
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Active infusion rates for this segment (shared with the FOCEI ODE
        // path so the F·RATE / span / lag / reset semantics stay in lockstep).
        // The EKF path has no per-dose lagtimes and no system resets.
        let active = active_infusions(
            &[],
            doses,
            t_start,
            t_end,
            &[],
            &dose_f_bio,
            f64::NEG_INFINITY,
            n,
        );

        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            rhs(y, p, t, dy);
            for &(cmt_idx, rate) in &active {
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            }
        };

        // Integrate mean state
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            pk_params_flat,
            &saveat,
            &opts,
        );

        // Propagate covariance and update at obs times within this segment
        let mut t_prev = t_start;
        let mut u_prev = u.clone();

        for pt in &sol {
            let dt = pt.t - t_prev;
            if dt > 1e-15 {
                // Sub-step the Riccati ODE to keep Euler error small.
                // 0.5 h per step keeps relative error < ~3% for typical PK.
                const DT_MAX: f64 = 0.5;
                let n_steps = ((dt / DT_MAX).ceil() as usize).max(1);
                let dt_sub = dt / n_steps as f64;
                for s in 0..n_steps {
                    // Linearly interpolate state across sub-step midpoint
                    let alpha_mid = (s as f64 + 0.5) / n_steps as f64;
                    let u_mid: Vec<f64> = u_prev
                        .iter()
                        .zip(&pt.u)
                        .map(|(&a, &b)| a + alpha_mid * (b - a))
                        .collect();
                    let t_mid = t_prev + alpha_mid * dt;
                    p_mat = propagate_covariance(
                        &wrapped_rhs,
                        &p_mat,
                        &u_mid,
                        pk_params_flat,
                        t_mid,
                        dt_sub,
                        diffusion_var,
                        n,
                    );
                }
            }

            if let Some(&obs_idx) = obs_map.get(&pt.t.to_bits()) {
                // Same assimilate-once mask as the boundary read: an observation sitting
                // exactly on this segment's `t_end` is read here *and* at the next break as
                // its `t_start`, and `kalman_update` is not idempotent.
                if !assimilated[obs_idx] {
                    let r = r_obs_vec.get(obs_idx).copied().unwrap_or(1.0);
                    let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
                    p_mat = p_new;
                    let v = pt.u[obs_cmt_idx];
                    results[obs_idx] = EkfObsPoint {
                        ipred: if v.is_nan() || v < 0.0 { 0.0 } else { v },
                        p_obs,
                    };
                    assimilated[obs_idx] = true;
                }
            }

            t_prev = pt.t;
            u_prev = pt.u.clone();
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A.
    fn one_cpt_rhs(y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]) {
        let cl = p[crate::types::PK_IDX_CL];
        let v = p[crate::types::PK_IDX_V];
        let ke = if v > 0.0 { cl / v } else { 0.0 };
        dy[0] = -ke * y[0];
    }

    fn make_pk(cl: f64, v: f64) -> Vec<f64> {
        let mut p = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        p[crate::types::PK_IDX_CL] = cl;
        p[crate::types::PK_IDX_V] = v;
        // Default bioavailability to 1.0 (a raw zero-filled vector would set
        // F = 0, which after issue #122 zeroes every dose and would make
        // dose-driven comparisons vacuously pass). Mirrors PkParams::default().
        p[crate::types::PK_IDX_F] = 1.0;
        p
    }

    fn bolus_dose(amt: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)
    }

    /// With zero diffusion the EKF must return identical ipred to `ode_predictions`.
    #[test]
    fn ekf_zero_diffusion_matches_ode_predictions() {
        use crate::ode::predictions::{ode_predictions, OdeSpec};
        use crate::types::Subject;
        use std::collections::HashMap;

        let doses = vec![bolus_dose(100.0)];
        let obs_times = vec![1.0, 4.0, 8.0, 12.0];
        let pk = make_pk(5.0, 80.0);
        let diffusion_var = vec![0.0]; // zero diffusion

        let r_obs_vec: Vec<f64> = vec![0.01; obs_times.len()];
        let ekf_pts = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &diffusion_var,
            &pk,
            &Default::default(),
            &[], // no init block in test: empty seeds zero state
            &doses,
            &obs_times,
            &r_obs_vec,
            OdeSolverOptions::default(),
        );

        let subj = Subject {
            id: "1".into(),
            doses: doses.clone(),
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; obs_times.len()],
            obs_cmts: vec![1; obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let ode_spec = OdeSpec {
            chz_state_slots: Vec::new(),
            rhs: Box::new(one_cpt_rhs),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: crate::ode::OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            init_fn: None,
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
        };
        let ode_preds = ode_predictions(&ode_spec, &pk, &[], &[], &subj);

        for (ekf, &ode) in ekf_pts.iter().zip(ode_preds.iter()) {
            assert_relative_eq!(ekf.ipred, ode, epsilon = 1e-4, max_relative = 1e-4);
            assert_relative_eq!(ekf.p_obs, 0.0, epsilon = 1e-10);
        }
    }

    /// #1226 on the EKF walk: an observation **one ULP after** a dose record is read
    /// post-dose, matching `ode_predictions`.
    ///
    /// This loop rescans every dose at every break exactly as the ODE prediction engines
    /// do, and it carried the same `t > t_start + 1e-12 && t <= t_end + 1e-12` segment
    /// filter plus an exact-bit `obs_map` boundary lookup — so the sample was claimed by
    /// the *pre*-dose segment and the post-dose read missed it. The EKF path applies no
    /// lagtime, so a data time one ULP after a dose record is the reachable geometry here
    /// rather than a lagged arrival.
    ///
    /// Oracle: `ode_predictions` on the identical fixture, which is itself anchored on
    /// NONMEM in `ode/predictions_tests.rs::lag_arrival_read_1226` — plus the closed form
    /// below, so the two engines cannot agree on a wrong answer.
    ///
    /// Mutation: revert either the boundary band read or the segment filter in `solve_ekf`
    /// and `ipred` at the ULP sample drops by a whole 100 mg dose.
    #[test]
    fn ekf_reads_an_obs_one_ulp_after_a_dose_post_dose() {
        use crate::ode::predictions::{ode_predictions, OdeSpec};
        use crate::types::Subject;
        use std::collections::HashMap;

        // Two doses so the second lands with the compartment non-empty: the pre/post-dose
        // pair is 45.38 vs 145.38 rather than 0 vs 100.
        let dose_time = 8.2f64;
        let obs_1ulp = f64::from_bits(dose_time.to_bits() + 1);
        let sep = obs_1ulp - dose_time;
        assert!(
            sep > 1e-15 && sep < crate::ode::predictions::EVENT_MATCH_TOL,
            "the sample must straddle the 1e-15 dedup and EVENT_MATCH_TOL (sep {sep:.3e})"
        );
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(dose_time, 100.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![obs_1ulp, 9.0];
        let pk = make_pk(1.0, 10.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let ekf_pts = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.0], // zero diffusion, so ipred must equal `ode_predictions`
            &pk,
            &Default::default(),
            &[],
            &doses,
            &obs_times,
            &vec![0.01; obs_times.len()],
            opts,
        );

        // Closed form, independent of either engine: the first dose's residual plus the
        // second, which has already arrived.
        let ke = 0.1f64;
        let pre = 100.0 * (-ke * obs_1ulp).exp();
        let post = pre + 100.0 * (-ke * sep).exp();
        assert!(
            post - pre > 99.0,
            "the pre/post-dose pair must be a whole dose apart ({pre:.4} / {post:.4})"
        );
        assert!(
            ekf_pts[0].ipred.is_finite(),
            "the EKF returned a non-finite ipred at the ULP sample"
        );
        assert!(
            (ekf_pts[0].ipred - post).abs() < 1e-6,
            "EKF reads {:.8} one ULP after the dose — expected the post-dose {post:.8} \
             (pre-dose is {pre:.8})",
            ekf_pts[0].ipred
        );

        let subj = Subject {
            id: "1".into(),
            doses: doses.clone(),
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; obs_times.len()],
            obs_cmts: vec![1; obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let ode_spec = OdeSpec {
            chz_state_slots: Vec::new(),
            rhs: Box::new(one_cpt_rhs),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: crate::ode::OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            init_fn: None,
            solver_opts: opts,
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
        };
        let ode_preds = ode_predictions(&ode_spec, &pk, &[], &[], &subj);
        for (i, (ekf, &ode)) in ekf_pts.iter().zip(ode_preds.iter()).enumerate() {
            assert!(
                ekf.ipred.is_finite() && ode.is_finite(),
                "non-finite at obs {i}: ekf {} / ode {ode}",
                ekf.ipred
            );
            assert_relative_eq!(ekf.ipred, ode, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    /// Issue #122: the EKF dosing path must load the compartment with F·AMT
    /// (NONMEM convention). For this linear system, halving bioavailability
    /// halves every ipred.
    #[test]
    fn ekf_applies_f_bio_to_bolus_dose() {
        let obs_times = vec![1.0, 4.0, 8.0, 12.0];
        let diffusion_var = vec![0.0];
        let r_obs_vec: Vec<f64> = vec![0.01; obs_times.len()];
        let doses = vec![bolus_dose(100.0)];

        let mut pk_full = make_pk(5.0, 80.0);
        pk_full[crate::types::PK_IDX_F] = 1.0;
        let mut pk_half = make_pk(5.0, 80.0);
        pk_half[crate::types::PK_IDX_F] = 0.5;

        let run = |pk: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &diffusion_var,
                pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &r_obs_vec,
                OdeSolverOptions::default(),
            )
        };
        let full = run(&pk_full);
        let half = run(&pk_half);
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(f.ipred > 0.0, "expected positive ipred");
            assert_relative_eq!(h.ipred, 0.5 * f.ipred, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    /// #369 review #3: the EKF dose loop is a separate dose-application path, so
    /// assert it applies **per-compartment** bioavailability (`Fn`). Uses a
    /// 2-compartment accumulator (`d/dt = 0`) dosed into both compartments with
    /// bare `F = 0.5` overridden by `F2 = 0.25`; the observed amount is then the
    /// bioavailable dose for that compartment. (The EKF path applies no lagtime,
    /// so this checks `F` routing only.)
    #[test]
    fn ekf_applies_per_compartment_bioavailability() {
        fn two_cpt_zero_rhs(_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]) {
            dy[0] = 0.0;
            dy[1] = 0.0;
        }
        let mut map = crate::types::DoseAttrMap::default();
        map.insert(crate::types::DoseAttr::F, 2, 9); // F2 -> spare slot 9

        let mut pk = vec![0.0f64; crate::types::MAX_PK_PARAMS];
        pk[crate::types::PK_IDX_F] = 0.5; // bare F (compartment 1)
        pk[9] = 0.25; // F2 (compartment 2)

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0];
        let diffusion_var = vec![0.0, 0.0];
        let r_obs_vec = vec![0.01; obs_times.len()];

        let run = |obs_cmt_idx: usize| -> f64 {
            solve_ekf(
                &two_cpt_zero_rhs,
                2,
                obs_cmt_idx,
                &diffusion_var,
                &pk,
                &map,
                &[],
                &doses,
                &obs_times,
                &r_obs_vec,
                OdeSolverOptions::default(),
            )[0]
            .ipred
        };
        assert!((run(0) - 50.0).abs() < 1e-9, "cmt1 F=0.5: {}", run(0));
        assert!((run(1) - 25.0).abs() < 1e-9, "cmt2 F2=0.25: {}", run(1));
    }

    /// Linear 1D SDE: dX = -ke·X dt + σ_w dW.
    ///
    /// The variance of the conditional distribution satisfies a Riccati ODE:
    ///   dP/dt = -2·ke·P + σ²_w
    /// with P(0) = 0. The analytic solution is:
    ///   P(t) = (σ²_w / (2·ke)) · (1 - exp(-2·ke·t))
    ///
    /// Without any observations (so no Kalman updates), the EKF should
    /// reproduce this. We verify at t = 1, 4, 8, 12 h.
    #[test]
    fn ekf_variance_matches_analytic_linear_sde() {
        let cl = 5.0_f64;
        let v = 100.0_f64;
        let ke = cl / v; // 0.05 h⁻¹
        let sigma2_w = 0.04_f64; // diffusion variance on central

        let doses = vec![bolus_dose(100.0)];
        let obs_times = vec![1.0, 4.0, 8.0, 12.0];
        let pk = make_pk(cl, v);

        // Use a large R so the Kalman update barely contracts P —
        // effectively "no assimilation", so P stays near the free-drift solution.
        let r_large = 1e8_f64;
        let r_obs_vec: Vec<f64> = vec![r_large; obs_times.len()];

        let ekf_pts = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[sigma2_w],
            &pk,
            &Default::default(),
            &[], // no init block in test: empty seeds zero state
            &doses,
            &obs_times,
            &r_obs_vec,
            OdeSolverOptions::default(),
        );

        for (i, &t) in obs_times.iter().enumerate() {
            let p_analytic = (sigma2_w / (2.0 * ke)) * (1.0 - (-2.0 * ke * t).exp());
            // Euler covariance propagation introduces O(Δt²) error; 5% tolerance is adequate.
            assert_relative_eq!(ekf_pts[i].p_obs, p_analytic, max_relative = 0.05);
        }
    }

    /// With positive diffusion, p_obs must be strictly positive at all observation times.
    #[test]
    fn ekf_p_obs_positive_with_diffusion() {
        let doses = vec![bolus_dose(100.0)];
        let obs_times = vec![2.0, 6.0, 12.0];
        let pk = make_pk(5.0, 80.0);

        let r_obs_vec: Vec<f64> = vec![0.05; obs_times.len()];
        let ekf_pts = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.1],
            &pk,
            &Default::default(),
            &[], // no init block in test: empty seeds zero state
            &doses,
            &obs_times,
            &r_obs_vec,
            OdeSolverOptions::default(),
        );

        for pt in &ekf_pts {
            assert!(
                pt.p_obs > 0.0,
                "expected p_obs > 0 with diffusion, got {}",
                pt.p_obs
            );
        }
    }

    /// A measurement landing exactly on a dose break must be assimilated **once**, while its
    /// `ipred` is still the post-dose one.
    ///
    /// Unlike the ODE engines, where a band write is an idempotent index assignment, the EKF
    /// runs a `kalman_update` that mutates `p_mat`. An observation on an interior break is
    /// saved by the segment ending there *and* read at that break as the next `t_start`, so
    /// it was assimilated twice — the covariance shrank twice, giving an over-confident
    /// `p_obs` at that record and a distorted covariance for the rest of the subject.
    ///
    /// Measured on this fixture: `p_obs` at `t = 6` was **0.0437** against the correct
    /// **0.3460** (a factor of 8), and the knock-on at `t = 12` was 0.4420 against 0.4514.
    /// The defect **predates #1226** — the old exact-bit boundary lookup matched an
    /// observation exactly at `t_start` just as the band does — but the band widens which
    /// records can reach it, so the mask is part of this change.
    ///
    /// **Oracle: a bolus does not touch `p_mat`.** So nudging the dose 1e-6 h earlier —
    /// far outside the match band, but a negligible amount of propagation — must leave every
    /// `p_obs` unchanged. That differential is what a double assimilation violates by a
    /// factor of 8, and it needs no reference implementation.
    ///
    /// The split matters and is asserted both ways: gating the *whole* boundary read on the
    /// mask (rather than only the `kalman_update`) makes `ipred` read pre-dose — measured
    /// 68.73 instead of 168.73, i.e. it reintroduces the #1226 defect on the EKF.
    ///
    /// Mutations: drop the `assimilated` guard → `p_obs` at `t = 6` collapses to 0.0437;
    /// make the guard `continue` before writing `ipred` → `ipred` at `t = 6` reads 68.73.
    #[test]
    fn ekf_assimilates_a_measurement_on_a_dose_break_exactly_once() {
        let obs_times = vec![2.0, 6.0, 12.0];
        let pk = make_pk(5.0, 80.0);
        let r = vec![0.05; obs_times.len()];
        let run = |dose_time: f64| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[0.1], // positive diffusion, so p_obs is a live quantity
                &pk,
                &Default::default(),
                &[],
                &[
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(dose_time, 100.0, 1, 0.0, false, 0.0),
                ],
                &obs_times,
                &r,
                OdeSolverOptions::default(),
            )
        };
        // A: the dose lands exactly on the t=6 observation. B: 1e-6 h earlier — a separate
        // break, negligible propagation.
        let a = run(6.0);
        let b = run(6.0 - 1e-6);

        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.p_obs.is_finite() && y.p_obs.is_finite(),
                "non-finite p_obs at obs {i}: {} / {}",
                x.p_obs,
                y.p_obs
            );
            assert!(
                x.p_obs > 0.0,
                "p_obs must stay positive with diffusion; got {} at obs {i}",
                x.p_obs
            );
            let rel = (x.p_obs - y.p_obs).abs() / y.p_obs;
            assert!(
                rel < 1e-6,
                "p_obs at t={} depends on whether the dose coincides with the sample \
                 ({:.10} vs {:.10}, rel {rel:.3e}) — a bolus does not touch the covariance, \
                 so this is the measurement being assimilated more than once",
                obs_times[i],
                x.p_obs,
                y.p_obs
            );
        }

        // …and the coincident record is still read post-dose: two 100 mg boluses, so the
        // post-dose mean is a whole dose above the pre-dose one.
        let ke = 5.0f64 / 80.0;
        let pre = 100.0 * (-ke * 6.0f64).exp();
        let post = pre + 100.0;
        assert!(
            post - pre > 99.0,
            "the pre/post-dose pair must be a whole dose apart ({pre:.4} / {post:.4})"
        );
        assert!(
            a[1].ipred.is_finite() && (a[1].ipred - post).abs() < 1e-3,
            "the EKF reads {:.6} at a sample on the dose break — expected the post-dose \
             {post:.6} (pre-dose is {pre:.6})",
            a[1].ipred
        );
    }

    // ── #1186 / #1189 on the EKF walk ───────────────────────────────────────
    //
    // This loop rescans `doses` at every break exactly as the ODE prediction engines do,
    // and it pushes an infusion-end break at `dose.time + duration` — a derived time that
    // can land within `EVENT_MATCH_TOL` of a later dose's own break. Its sort was a
    // `partial_cmp(..).unwrap()` too, so a non-finite time panicked here as well.

    /// A colliding infusion end must not make the later bolus land twice, and with zero
    /// diffusion the EKF must still equal `ode_predictions` — which is itself anchored on
    /// NONMEM `break_collision_inf` in `ode/predictions_tests.rs`.
    ///
    /// Mutation: delete `ekf.rs`'s `if applied[di] { continue; }` → the bolus doubles and
    /// the closed-form assert below fires (the `ode_predictions` cross-check would too,
    /// since only one of the two engines is mutated).
    #[test]
    fn ekf_infusion_end_collision_applies_the_bolus_once() {
        use crate::ode::predictions::{ode_predictions, OdeReadout, OdeSpec};
        use crate::types::Subject;
        use std::collections::HashMap;

        // The rate whose computed duration `100/rate` lands strictly inside
        // `(1e-15, EVENT_MATCH_TOL)` of 8.2 — searched, then asserted, so the fixture
        // cannot drift out of the band it exists to test.
        let r0 = 100.0 / 8.2;
        let ulp = f64::EPSILON * r0;
        let (rate, dur) = (-200..=200)
            .filter_map(|k| {
                let r = r0 + (k as f64) * ulp;
                let d = 100.0 / r;
                let s = (d - 8.2f64).abs();
                (s > 1e-15 && s < crate::ode::predictions::EVENT_MATCH_TOL).then_some((r, d))
            })
            .next()
            .expect("no rate in the (1e-15, EVENT_MATCH_TOL) band");
        assert!(
            (dur - 8.2f64).abs() > 1e-15
                && (dur - 8.2f64).abs() < crate::ode::predictions::EVENT_MATCH_TOL,
            "the infusion end must straddle the 1e-15 dedup and EVENT_MATCH_TOL"
        );

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, rate, false, 0.0),
            DoseEvent::new(8.2, 100.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![8.2001, 12.0, 24.0];
        let pk = make_pk(1.0, 10.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let got = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.0],
            &pk,
            &crate::types::DoseAttrMap::default(),
            &[0.0],
            &doses,
            &obs_times,
            &vec![1.0; obs_times.len()],
            opts,
        );

        // Closed form: the infusion's washout plus exactly one bolus.
        let ke = 0.1_f64;
        for (i, &t) in obs_times.iter().enumerate() {
            let inf = (rate / ke) * (1.0 - (-ke * dur).exp()) * (-ke * (t - dur)).exp();
            let once = inf + 100.0 * (-ke * (t - 8.2)).exp();
            let twice = inf + 200.0 * (-ke * (t - 8.2)).exp();
            assert!(
                (once - twice).abs() > 1.0,
                "the once/twice oracles must differ at t={t} or this cannot fail"
            );
            assert!(got[i].ipred.is_finite(), "EKF non-finite at t={t}");
            assert_relative_eq!(got[i].ipred, once, max_relative = 1e-6);
        }

        // And against the objective engine at zero diffusion, so the two cannot drift.
        let ode = OdeSpec {
            chz_state_slots: Vec::new(),
            rhs: Box::new(one_cpt_rhs),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: opts,
            input_rate: Vec::new(),
            init_fn: None,
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: crate::types::DoseAttrMap::default(),
        };
        let subject = Subject {
            id: "1".into(),
            doses,
            obs_times: obs_times.clone(),
            observations: vec![1.0; obs_times.len()],
            obs_cmts: vec![1; obs_times.len()],
            covariates: HashMap::new(),
            cens: vec![0; obs_times.len()],
            occasions: vec![1; obs_times.len()],
            ..Default::default()
        };
        let want = ode_predictions(&ode, &pk, &[], &[], &subject);
        for (i, &t) in obs_times.iter().enumerate() {
            assert_relative_eq!(got[i].ipred, want[i], max_relative = 1e-8);
            let _ = t;
        }
    }

    /// A non-finite time on this walk's timeline used to panic its
    /// `partial_cmp(..).unwrap()` sort. It must now return non-finite `ipred`s — not a
    /// finite filter pass computed on a timeline that could not be ordered, which is
    /// what `total_cmp` alone would give.
    ///
    /// **Defence in depth, and the fixture says why.** The route that reaches the other
    /// engines — a non-finite *lag* — cannot reach this one: the EKF path applies no
    /// lagtime, and its only derived break is the infusion end, which
    /// [`is_real_infusion`] already refuses to emit when `duration = amt/rate` is
    /// non-finite (that dose falls back to the bolus branch, deliberately, and the
    /// timeline stays orderable — measured). So the trigger here is a non-finite *record*
    /// time, reachable through the hand-built entry `solve_ekf` exposes. The guard exists
    /// because this walk's sort is the same one, not because a fitted dataset gets here.
    ///
    /// Mutation: restore `partial_cmp(..).unwrap()` → panic; delete the
    /// `timeline_has_non_finite` guard → finite `ipred`s and the assert below fires.
    #[test]
    fn ekf_non_finite_break_is_non_finite_not_a_panic() {
        let doses = vec![
            DoseEvent::new(f64::NAN, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.2, 100.0, 1, 0.0, false, 0.0),
        ];
        let obs_times: Vec<f64> = (1..=40).map(|i| i as f64 * 0.5).collect();
        assert!(
            obs_times.len() >= 21,
            "the timeline must exceed the 21-element threshold where sort_by starts \
             detecting a non-total comparator, or this cannot fail"
        );
        let pk = make_pk(1.0, 10.0);
        let got = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.0],
            &pk,
            &crate::types::DoseAttrMap::default(),
            &[0.0],
            &doses,
            &obs_times,
            &vec![1.0; obs_times.len()],
            OdeSolverOptions::default(),
        );
        assert!(
            got.iter().all(|p| !p.ipred.is_finite()),
            "a non-finite timeline must give non-finite EKF ipreds, not a finite pass \
             with the bad dose silently dropped"
        );
    }
}
