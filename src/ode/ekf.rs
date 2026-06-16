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

use crate::ode::predictions::{active_infusions, is_real_infusion};
use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::{DoseEvent, PK_IDX_F};
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
    // Bioavailability F (slot PK_IDX_F, default 1.0) scales the amount that
    // enters the dosing compartment — NONMEM's F·AMT (bolus) / F·RATE
    // (infusion), matching the analytical and plain-ODE paths.
    let f_bio = pk_params_flat.get(PK_IDX_F).copied().unwrap_or(1.0);
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
    // Bioavailability is constant across doses on the EKF path (no per-dose
    // time-varying parameters), so the per-dose F slice is uniform.
    let dose_f_bio = vec![f_bio; doses.len()];
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply bolus doses at t_start (infusions enter via the wrapped RHS).
        for dose in doses {
            if (dose.time - t_start).abs() < 1e-12 && !is_real_infusion(dose) {
                let cmt_idx = dose.cmt.saturating_sub(1);
                if cmt_idx < n {
                    u[cmt_idx] += f_bio * dose.amt;
                }
            }
        }

        // Record obs exactly at t_start (after dose application)
        if let Some(&obs_idx) = obs_map.get(&t_start.to_bits()) {
            let r = r_obs_vec.get(obs_idx).copied().unwrap_or(1.0);
            let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
            p_mat = p_new;
            let v = u[obs_cmt_idx];
            results[obs_idx] = EkfObsPoint {
                ipred: if v.is_nan() || v < 0.0 { 0.0 } else { v },
                p_obs,
            };
        }

        let mut saveat: Vec<f64> = obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Active infusion rates for this segment (shared with the FOCEI ODE
        // path so the F·RATE / span / lag / reset semantics stay in lockstep).
        // The EKF path has no per-dose lagtimes and no system resets.
        let active = active_infusions(doses, t_start, t_end, &[], &dose_f_bio, f64::NEG_INFINITY);

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
                let r = r_obs_vec.get(obs_idx).copied().unwrap_or(1.0);
                let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
                p_mat = p_new;
                let v = pt.u[obs_cmt_idx];
                results[obs_idx] = EkfObsPoint {
                    ipred: if v.is_nan() || v < 0.0 { 0.0 } else { v },
                    p_obs,
                };
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
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let ode_spec = OdeSpec {
            rhs: Box::new(one_cpt_rhs),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: crate::ode::OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            init_fn: None,
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
        };
        let ode_preds = ode_predictions(&ode_spec, &pk, &[], &[], &subj);

        for (ekf, &ode) in ekf_pts.iter().zip(ode_preds.iter()) {
            assert_relative_eq!(ekf.ipred, ode, epsilon = 1e-4, max_relative = 1e-4);
            assert_relative_eq!(ekf.p_obs, 0.0, epsilon = 1e-10);
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
        pk_full[PK_IDX_F] = 1.0;
        let mut pk_half = make_pk(5.0, 80.0);
        pk_half[PK_IDX_F] = 0.5;

        let run = |pk: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &diffusion_var,
                pk,
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
}
