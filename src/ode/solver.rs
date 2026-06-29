//! Adaptive Runge-Kutta ODE solver (Dormand-Prince / RK45).
//!
//! This is the same family as Julia's Tsit5() — a 5th-order explicit RK method
//! with embedded 4th-order error estimate for adaptive step control.
//! Optimized for PK ODE systems (2-5 states, smooth dynamics).

/// Dormand-Prince RK45 coefficients (Butcher tableau)
const A2: f64 = 1.0 / 5.0;
const A3: f64 = 3.0 / 10.0;
const A4: f64 = 4.0 / 5.0;
const A5: f64 = 8.0 / 9.0;
// a6 = 1.0, a7 = 1.0

const B21: f64 = 1.0 / 5.0;
const B31: f64 = 3.0 / 40.0;
const B32: f64 = 9.0 / 40.0;
const B41: f64 = 44.0 / 45.0;
const B42: f64 = -56.0 / 15.0;
const B43: f64 = 32.0 / 9.0;
const B51: f64 = 19372.0 / 6561.0;
const B52: f64 = -25360.0 / 2187.0;
const B53: f64 = 64448.0 / 6561.0;
const B54: f64 = -212.0 / 729.0;
const B61: f64 = 9017.0 / 3168.0;
const B62: f64 = -355.0 / 33.0;
const B63: f64 = 46732.0 / 5247.0;
const B64: f64 = 49.0 / 176.0;
const B65: f64 = -5103.0 / 18656.0;
const B71: f64 = 35.0 / 384.0;
const B73: f64 = 500.0 / 1113.0;
const B74: f64 = 125.0 / 192.0;
const B75: f64 = -2187.0 / 6784.0;
const B76: f64 = 11.0 / 84.0;

// Error coefficients (5th order - 4th order)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Consecutive force-accepted minimum-step attempts with a **non-finite** local error
/// before an integration segment is treated as unrecoverably pathological.
///
/// The break is gated on a non-finite error norm (NaN/∞) on purpose: that is the signature
/// of a diverging trajectory whose padded-out predictions the likelihood will reject anyway.
/// A *finite* error above tolerance at `min_dt` is merely a stiff or under-resolved segment —
/// truncating it would freeze-pad the remaining save points with finite (but wrong) values
/// that the likelihood would silently accept, so those are left to run to `max_steps` as
/// before (#603 review #4).
const MAX_CONSECUTIVE_MIN_STEP_CLAMPS: usize = 64;

/// Step-shrink factor applied when the local error estimate is non-finite (NaN/∞). The
/// trajectory is diverging, so shrink toward `min_dt` instead of falling into the
/// I-controller's growth branch, which would otherwise enlarge the step on a NaN error.
const NONFINITE_ERR_SHRINK_FACTOR: f64 = 0.2;

/// ODE right-hand side function type.
/// `rhs(u, params, t) -> du/dt`  where u and du are `&[f64]` of length n_states.
pub type OdeRhsFn = Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>;

/// ODE solver options
#[derive(Debug, Clone, Copy)]
pub struct OdeSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub max_steps: usize,
    pub initial_dt: f64,
    pub min_dt: f64,
}

impl Default for OdeSolverOptions {
    fn default() -> Self {
        Self {
            abstol: 1e-6,
            reltol: 1e-4,
            max_steps: 10000,
            initial_dt: 0.1,
            min_dt: 1e-12,
        }
    }
}

/// Solution point: (time, state vector)
#[derive(Debug, Clone)]
pub struct SolPoint {
    pub t: f64,
    pub u: Vec<f64>,
}

/// Adaptive-step counters for diagnosing whether an integration is
/// rejection-dominated, accepted-small-step-dominated, or hitting the solver's
/// minimum-step escape hatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OdeSolverStats {
    /// Total RK attempts: `accepted_steps + rejected_steps`.
    pub attempted_steps: usize,
    /// Attempts that advanced `(t, u)`.
    ///
    /// Includes [`min_step_clamped_steps`](Self::min_step_clamped_steps), because
    /// the current RK45 policy accepts at `min_dt` to guarantee progress.
    pub accepted_steps: usize,
    /// Attempts rejected by the local-error test.
    pub rejected_steps: usize,
    /// Accepted attempts that failed the local-error test (`!(err_norm <= 1)`,
    /// which also catches a non-finite `err_norm`) yet advanced because
    /// `dt_eff <= min_dt`.
    pub min_step_clamped_steps: usize,
}

impl OdeSolverStats {
    /// Record one RK attempt. `accepted` is the stepper's accept decision
    /// (`err_norm <= 1.0 || dt_eff <= min_dt`); the clamp test mirrors it with
    /// `!(err_norm <= 1.0)` so a non-finite `err_norm` (a diverging RHS pinned
    /// at `min_dt`) still counts as a min-step clamp rather than a clean accept.
    #[inline]
    fn record(&mut self, accepted: bool, err_norm: f64, dt_eff: f64, min_dt: f64) {
        self.attempted_steps += 1;
        if accepted {
            self.accepted_steps += 1;
            if !(err_norm <= 1.0) && dt_eff <= min_dt {
                self.min_step_clamped_steps += 1;
            }
        } else {
            self.rejected_steps += 1;
        }
    }
}

/// Integrate an ODE system from t_start to t_end, saving at specified times.
///
/// Returns solution at each saveat time.
pub fn solve_ode(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    t_span: (f64, f64),
    params: &[f64],
    saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<SolPoint> {
    solve_ode_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode`] with optional adaptive-step instrumentation.
///
/// The counters are intentionally local to this integration segment. Higher
/// layers that split by dose/observation boundaries can aggregate across calls
/// to classify a full subject or fit.
pub fn solve_ode_with_stats(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    t_span: (f64, f64),
    params: &[f64],
    saveat: &[f64],
    opts: &OdeSolverOptions,
    mut stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPoint> {
    let n = u0.len();
    let (t0, tf) = t_span;

    if (tf - t0).abs() < 1e-15 {
        return saveat
            .iter()
            .map(|&t| SolPoint { t, u: u0.to_vec() })
            .collect();
    }

    let mut u = u0.to_vec();
    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);

    // Pre-allocate stage vectors
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut u_tmp = vec![0.0; n];
    let mut u5 = vec![0.0; n];

    let mut results: Vec<SolPoint> = Vec::with_capacity(saveat.len());
    let mut save_idx = 0;

    // FSAL (First Same As Last): in DP-RK45, k7 of an accepted step is evaluated
    // at the same (u, t) the next step's k1 would use. We carry it across via a
    // k1/k7 swap, eliminating one rhs eval per accepted step (~1 of 7 stages).
    // After a rejected step (u, t) doesn't move so k1 stays valid too; first
    // iteration has no prior k1, hence `have_k1`. ≈9% wall reduction on
    // FOCEI ODE fits with bit-identical outputs (FSAL only reuses a value that
    // would otherwise be recomputed identically).
    let mut have_k1 = false;

    // NOTE: a Gustafsson PI step-size controller was tested and rejected here.
    // While it lowers raw step-rejection rate and integrates faster, the
    // factor's dependence on err_{n-1} makes accept/reject decisions more
    // sensitive to small parameter perturbations. That raises the differential
    // noise floor of the trajectory as a function of θ, which the FOCEI FD
    // gradient cannot tolerate — BFGS line search stalled at OFV ≈ -1290 on
    // the dense-Emax PKPD benchmark vs the true -1747 with the I-controller.
    // The pure I-controller below is memoryless and gives a clean FD signal.
    // Any future revisit should condition PI on a non-FD gradient route
    // (analytical / analytic-sensitivity).
    const I_EXP: f64 = 1.0 / 5.0; // 0.20 — I-controller exponent for order p=5
    let mut consecutive_min_step_clamps = 0usize;

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            break;
        }

        // Don't overshoot tf or next saveat
        let mut dt_eff = dt.min(tf - t);
        if save_idx < saveat.len() && t + dt_eff > saveat[save_idx] + 1e-15 {
            dt_eff = (saveat[save_idx] - t).max(opts.min_dt);
        }

        // RK45 stages — k1 may be carried from previous step via FSAL.
        if !have_k1 {
            rhs(&u, params, t, &mut k1);
            have_k1 = true;
        }

        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * B21 * k1[i];
        }
        rhs(&u_tmp, params, t + A2 * dt_eff, &mut k2);

        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B31 * k1[i] + B32 * k2[i]);
        }
        rhs(&u_tmp, params, t + A3 * dt_eff, &mut k3);

        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B41 * k1[i] + B42 * k2[i] + B43 * k3[i]);
        }
        rhs(&u_tmp, params, t + A4 * dt_eff, &mut k4);

        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B51 * k1[i] + B52 * k2[i] + B53 * k3[i] + B54 * k4[i]);
        }
        rhs(&u_tmp, params, t + A5 * dt_eff, &mut k5);

        for i in 0..n {
            u_tmp[i] = u[i]
                + dt_eff * (B61 * k1[i] + B62 * k2[i] + B63 * k3[i] + B64 * k4[i] + B65 * k5[i]);
        }
        rhs(&u_tmp, params, t + dt_eff, &mut k6);

        // 5th-order solution
        for i in 0..n {
            u5[i] = u[i]
                + dt_eff * (B71 * k1[i] + B73 * k3[i] + B74 * k4[i] + B75 * k5[i] + B76 * k6[i]);
        }

        // Error estimate
        rhs(&u5, params, t + dt_eff, &mut k7);

        let mut err_norm = 0.0;
        for i in 0..n {
            let err_i = dt_eff
                * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let scale = opts.abstol + opts.reltol * u5[i].abs().max(u[i].abs());
            err_norm += (err_i / scale) * (err_i / scale);
        }
        err_norm = (err_norm / n as f64).sqrt();

        let accepted = err_norm <= 1.0 || dt_eff <= opts.min_dt;
        // Only a force-accept at `min_dt` with a *non-finite* error counts toward the
        // pathological-divergence break (#603 review #4); a finite-but-stiff clamp is left
        // to run rather than freeze-padded into a silently-accepted wrong trajectory.
        let nonfinite_min_step = accepted && dt_eff <= opts.min_dt && !err_norm.is_finite();
        if let Some(s) = stats.as_deref_mut() {
            s.record(accepted, err_norm, dt_eff, opts.min_dt);
        }
        if accepted {
            // Accept step
            t += dt_eff;
            u.copy_from_slice(&u5);

            // FSAL: k7 at (u_new, t_new) IS the next step's k1 — swap into k1.
            // Safe because k7 is dead from this point onward in this iteration.
            std::mem::swap(&mut k1, &mut k7);

            // Save at requested times
            while save_idx < saveat.len() && (t - saveat[save_idx]).abs() < 1e-12 {
                results.push(SolPoint {
                    t: saveat[save_idx],
                    u: u.clone(),
                });
                save_idx += 1;
            }

            if nonfinite_min_step {
                consecutive_min_step_clamps += 1;
                if consecutive_min_step_clamps >= MAX_CONSECUTIVE_MIN_STEP_CLAMPS {
                    break;
                }
            } else {
                consecutive_min_step_clamps = 0;
            }
        }
        // On reject: (u, t) is unchanged, so the existing k1 is still rhs(u, t)
        // for the next attempt; nothing to do.

        // Adapt step size (memoryless I-controller — see note above).
        let safety = 0.9;
        let factor = if !err_norm.is_finite() {
            NONFINITE_ERR_SHRINK_FACTOR
        } else if err_norm > 1e-15 {
            safety * err_norm.powf(-I_EXP)
        } else {
            5.0
        };
        dt = dt_eff * factor.clamp(0.2, 5.0);
        dt = dt.max(opts.min_dt);
    }

    // Fill any remaining saveat points with last state
    while save_idx < saveat.len() {
        results.push(SolPoint {
            t: saveat[save_idx],
            u: u.clone(),
        });
        save_idx += 1;
    }

    results
}

/// Outcome of [`solve_ode_until_threshold`] over a single integration span.
///
/// This is the **segment-level** primitive behind the TTE event-time sampler
/// (plan §8.8.3): the simulation entry point drives it once per dose segment
/// (mirroring [`crate::ode::ode_dense_solve_states`]), carrying the *absolute*
/// monitored threshold and the end-of-segment state across segments. The public
/// wrapper maps a terminal [`ReachedEnd`](ThresholdCrossing::ReachedEnd) (the
/// monitor never reached the threshold by the horizon) to a right-censored draw,
/// [`Crossed`](ThresholdCrossing::Crossed) to an event, and
/// [`Failed`](ThresholdCrossing::Failed) to a hard error — a failed solve must
/// never be laundered into a censored subject.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdCrossing {
    /// The monitored state reached `threshold` at this time within the span.
    Crossed(f64),
    /// Integrated cleanly to the end of the span without crossing; the caller's
    /// `u` now holds the state at `t_end` (carry it into the next segment).
    ReachedEnd,
    /// No meaningful crossing can be reported: a non-finite monitored state, a
    /// monitored state that *decreased* over a step (non-monotone accumulator ⇒ a
    /// negative rate — for TTE, a negative hazard, which invalidates the crossing
    /// argument), or the step budget exhausted before reaching `t_end`. The
    /// message names the cause.
    Failed(String),
}

/// Integrate `rhs` over `t_span`, **halting at the first time `u[monitor]` reaches
/// `threshold`** — a one-sided upward crossing, for a monitored state expected to
/// be monotone non-decreasing (a cumulative-hazard accumulator `dCHZ/dt = h ≥ 0`).
///
/// `u` is the state at `t_span.0` on entry and is advanced **in place** to the
/// state at `t_span.1` on [`ReachedEnd`](ThresholdCrossing::ReachedEnd) (so a
/// segmented caller can carry it into the next dose segment); on `Crossed` /
/// `Failed` its final value is unspecified.
///
/// Localization within the crossing step is **bracketed bisection on the cubic
/// Hermite interpolant** of the monitored component (the step endpoints plus the
/// FSAL stage derivatives `k1`, `k7`) — *not* Newton, because a Hermite cubic can
/// overshoot even when the true trajectory is monotone, whereas the accepted step
/// already proves the bracket `y0 < threshold ≤ y1`, on which bisection is
/// unconditionally robust.
///
/// This is a **separate** stepper from [`solve_ode_with_stats`] (rather than a
/// flag on it) so the FOCEI hot path stays bit-identical; the Butcher constants
/// are shared module-level items, so only the stepping loop is restated.
pub fn solve_ode_until_threshold(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u: &mut [f64],
    t_span: (f64, f64),
    params: &[f64],
    opts: &OdeSolverOptions,
    monitor: usize,
    threshold: f64,
) -> ThresholdCrossing {
    let n = u.len();
    let (t0, tf) = t_span;

    // Already at/above threshold at the span start (e.g. threshold ≈ 0 from u≈1,
    // or a non-zero initial accumulator). Catch before stepping so the loop's
    // crossing test can assume `y0 < threshold`.
    if u[monitor] >= threshold {
        return ThresholdCrossing::Crossed(t0);
    }
    if (tf - t0).abs() < 1e-15 {
        return ThresholdCrossing::ReachedEnd;
    }

    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut u_tmp = vec![0.0; n];
    let mut u5 = vec![0.0; n];

    // FSAL + memoryless I-controller, identical to `solve_ode_with_stats` (see the
    // step-control notes there); no `saveat` clamp here — this stepper saves
    // nothing, it only watches `monitor`, so its step sequence is not perturbed by
    // observation times.
    let mut have_k1 = false;
    const I_EXP: f64 = 1.0 / 5.0;

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            return ThresholdCrossing::ReachedEnd;
        }

        let dt_eff = dt.min(tf - t);

        if !have_k1 {
            rhs(u, params, t, &mut k1);
            have_k1 = true;
        }
        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * B21 * k1[i];
        }
        rhs(&u_tmp, params, t + A2 * dt_eff, &mut k2);
        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B31 * k1[i] + B32 * k2[i]);
        }
        rhs(&u_tmp, params, t + A3 * dt_eff, &mut k3);
        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B41 * k1[i] + B42 * k2[i] + B43 * k3[i]);
        }
        rhs(&u_tmp, params, t + A4 * dt_eff, &mut k4);
        for i in 0..n {
            u_tmp[i] = u[i] + dt_eff * (B51 * k1[i] + B52 * k2[i] + B53 * k3[i] + B54 * k4[i]);
        }
        rhs(&u_tmp, params, t + A5 * dt_eff, &mut k5);
        for i in 0..n {
            u_tmp[i] = u[i]
                + dt_eff * (B61 * k1[i] + B62 * k2[i] + B63 * k3[i] + B64 * k4[i] + B65 * k5[i]);
        }
        rhs(&u_tmp, params, t + dt_eff, &mut k6);
        for i in 0..n {
            u5[i] = u[i]
                + dt_eff * (B71 * k1[i] + B73 * k3[i] + B74 * k4[i] + B75 * k5[i] + B76 * k6[i]);
        }
        rhs(&u5, params, t + dt_eff, &mut k7);

        let mut err_norm = 0.0;
        for i in 0..n {
            let err_i = dt_eff
                * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let scale = opts.abstol + opts.reltol * u5[i].abs().max(u[i].abs());
            err_norm += (err_i / scale) * (err_i / scale);
        }
        err_norm = (err_norm / n as f64).sqrt();

        let accepted = err_norm <= 1.0 || dt_eff <= opts.min_dt;
        if accepted {
            // Monitored value + derivative at the step's start (y0,d0) and end
            // (y1,d1) — captured before the FSAL swap and the `u <- u5` advance.
            let y0 = u[monitor];
            let d0 = k1[monitor];
            let y1 = u5[monitor];
            let d1 = k7[monitor];

            if !y1.is_finite() {
                return ThresholdCrossing::Failed(format!(
                    "monitored state {monitor} became non-finite at t={:.6}",
                    t + dt_eff
                ));
            }
            // Scale-aware monotonicity floor: a real negative rate produces a
            // decrease ≫ this; only round-off sits below it.
            let mono_tol = opts.abstol + opts.reltol * y0.abs().max(y1.abs());
            if y1 < y0 - mono_tol {
                return ThresholdCrossing::Failed(format!(
                    "monitored state {monitor} decreased ({y0:.6} → {y1:.6}) over \
                     [{t:.6}, {:.6}]: non-monotone accumulator (negative rate / hazard)",
                    t + dt_eff
                ));
            }

            if y1 >= threshold {
                // Crossing in (t, t+dt_eff]; bisect the Hermite cubic on the bracket
                // the accepted step proves (y0 < threshold ≤ y1). 64 halvings drives
                // the bracket below machine precision of the step — cubic evals only.
                let h = dt_eff;
                let hermite = |s: f64| {
                    let s2 = s * s;
                    let s3 = s2 * s;
                    (2.0 * s3 - 3.0 * s2 + 1.0) * y0
                        + (s3 - 2.0 * s2 + s) * h * d0
                        + (-2.0 * s3 + 3.0 * s2) * y1
                        + (s3 - s2) * h * d1
                };
                let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    if hermite(mid) < threshold {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return ThresholdCrossing::Crossed(t + 0.5 * (lo + hi) * h);
            }

            t += dt_eff;
            u.copy_from_slice(&u5);
            std::mem::swap(&mut k1, &mut k7);
        }

        let safety = 0.9;
        // Finite-error branch is identical to `solve_ode_with_stats`; a *non-finite*
        // err_norm (diverging / NaN RHS) shrinks toward `min_dt` instead of growing,
        // so the step is force-accepted at `min_dt` and the non-finite guard above
        // fires with a clear cause rather than silently burning the step budget.
        let factor = if !err_norm.is_finite() {
            0.2
        } else if err_norm > 1e-15 {
            safety * err_norm.powf(-I_EXP)
        } else {
            5.0
        };
        dt = (dt_eff * factor.clamp(0.2, 5.0)).max(opts.min_dt);
    }

    ThresholdCrossing::Failed(format!(
        "step budget ({}) exhausted before reaching t_end={tf:.6} (reached t={t:.6})",
        opts.max_steps
    ))
}

/// Generic solution point for the [`solve_ode_g`] sensitivity path.
#[derive(Debug, Clone)]
pub struct SolPointG<T> {
    pub t: f64,
    pub u: Vec<T>,
}

/// [`solve_ode`] generic over the state scalar `T: PkNum`, for the analytic
/// PK-parameter sensitivity path (`T = Dual2<N>`): the same Dormand-Prince
/// stepper, but every Butcher combination is a `T` operation so the dual numbers
/// carry `∂u/∂p` and `∂²u/∂p²` through the integration. `params` holds the PK
/// parameters seeded as dual variables.
///
/// **Step-size control reads `.val()` only** — the accept/reject decision and
/// `dt` adaptation depend on values, never on derivatives, so the derivative
/// flows through a *fixed* step sequence. (Adapting on a derivative norm would
/// make the sensitivity inconsistent with the prediction.) This is the
/// derivative-carrying sibling of [`solve_ode`]; the f64 path is left untouched.
pub fn solve_ode_g<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<SolPointG<T>> {
    solve_ode_g_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode_g`] with optional adaptive-step instrumentation.
///
/// Stats are computed from value-only error control, matching the production
/// sensitivity path: derivative components never influence accept/reject or
/// step-size decisions.
pub fn solve_ode_g_with_stats<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    opts: &OdeSolverOptions,
    mut stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPointG<T>> {
    let n = u0.len();
    let (t0, tf) = t_span;
    let zero = T::from_f64(0.0);

    if (tf - t0).abs() < 1e-15 {
        return saveat
            .iter()
            .map(|&t| SolPointG { t, u: u0.to_vec() })
            .collect();
    }

    let mut u = u0.to_vec();
    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);

    let mut k1 = vec![zero; n];
    let mut k2 = vec![zero; n];
    let mut k3 = vec![zero; n];
    let mut k4 = vec![zero; n];
    let mut k5 = vec![zero; n];
    let mut k6 = vec![zero; n];
    let mut k7 = vec![zero; n];
    let mut u_tmp = vec![zero; n];
    let mut u5 = vec![zero; n];

    let mut results: Vec<SolPointG<T>> = Vec::with_capacity(saveat.len());
    let mut save_idx = 0;
    let mut have_k1 = false;
    const I_EXP: f64 = 1.0 / 5.0;
    let mut consecutive_min_step_clamps = 0usize;

    // Scalar-weighted combination `u[i] + dt·Σ wⱼ·kⱼ[i]` lifted to `T`.
    let comb = |base: T, dt_eff: f64, terms: &[(f64, T)]| -> T {
        let mut acc = base;
        for &(w, kij) in terms {
            acc = acc + kij * T::from_f64(dt_eff * w);
        }
        acc
    };

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            break;
        }

        let mut dt_eff = dt.min(tf - t);
        if save_idx < saveat.len() && t + dt_eff > saveat[save_idx] + 1e-15 {
            dt_eff = (saveat[save_idx] - t).max(opts.min_dt);
        }

        if !have_k1 {
            rhs(&u, params, t, &mut k1);
            have_k1 = true;
        }

        for i in 0..n {
            u_tmp[i] = comb(u[i], dt_eff, &[(B21, k1[i])]);
        }
        rhs(&u_tmp, params, t + A2 * dt_eff, &mut k2);

        for i in 0..n {
            u_tmp[i] = comb(u[i], dt_eff, &[(B31, k1[i]), (B32, k2[i])]);
        }
        rhs(&u_tmp, params, t + A3 * dt_eff, &mut k3);

        for i in 0..n {
            u_tmp[i] = comb(u[i], dt_eff, &[(B41, k1[i]), (B42, k2[i]), (B43, k3[i])]);
        }
        rhs(&u_tmp, params, t + A4 * dt_eff, &mut k4);

        for i in 0..n {
            u_tmp[i] = comb(
                u[i],
                dt_eff,
                &[(B51, k1[i]), (B52, k2[i]), (B53, k3[i]), (B54, k4[i])],
            );
        }
        rhs(&u_tmp, params, t + A5 * dt_eff, &mut k5);

        for i in 0..n {
            u_tmp[i] = comb(
                u[i],
                dt_eff,
                &[
                    (B61, k1[i]),
                    (B62, k2[i]),
                    (B63, k3[i]),
                    (B64, k4[i]),
                    (B65, k5[i]),
                ],
            );
        }
        rhs(&u_tmp, params, t + dt_eff, &mut k6);

        for i in 0..n {
            u5[i] = comb(
                u[i],
                dt_eff,
                &[
                    (B71, k1[i]),
                    (B73, k3[i]),
                    (B74, k4[i]),
                    (B75, k5[i]),
                    (B76, k6[i]),
                ],
            );
        }

        rhs(&u5, params, t + dt_eff, &mut k7);

        // Error norm on values only (step control must not see derivatives).
        let mut err_norm = 0.0;
        for i in 0..n {
            let err_i = dt_eff
                * (E1 * k1[i].val()
                    + E3 * k3[i].val()
                    + E4 * k4[i].val()
                    + E5 * k5[i].val()
                    + E6 * k6[i].val()
                    + E7 * k7[i].val());
            let scale = opts.abstol + opts.reltol * u5[i].val().abs().max(u[i].val().abs());
            err_norm += (err_i / scale) * (err_i / scale);
        }
        err_norm = (err_norm / n as f64).sqrt();

        let accepted = err_norm <= 1.0 || dt_eff <= opts.min_dt;
        // Mirror the scalar path: only a non-finite force-accept at `min_dt` counts toward
        // the pathological-divergence break (#603 review #4).
        let nonfinite_min_step = accepted && dt_eff <= opts.min_dt && !err_norm.is_finite();
        if let Some(s) = stats.as_deref_mut() {
            s.record(accepted, err_norm, dt_eff, opts.min_dt);
        }
        if accepted {
            t += dt_eff;
            u.copy_from_slice(&u5);
            std::mem::swap(&mut k1, &mut k7);
            while save_idx < saveat.len() && (t - saveat[save_idx]).abs() < 1e-12 {
                results.push(SolPointG {
                    t: saveat[save_idx],
                    u: u.clone(),
                });
                save_idx += 1;
            }

            if nonfinite_min_step {
                consecutive_min_step_clamps += 1;
                if consecutive_min_step_clamps >= MAX_CONSECUTIVE_MIN_STEP_CLAMPS {
                    break;
                }
            } else {
                consecutive_min_step_clamps = 0;
            }
        }

        let safety = 0.9;
        let factor = if !err_norm.is_finite() {
            NONFINITE_ERR_SHRINK_FACTOR
        } else if err_norm > 1e-15 {
            safety * err_norm.powf(-I_EXP)
        } else {
            5.0
        };
        dt = dt_eff * factor.clamp(0.2, 5.0);
        dt = dt.max(opts.min_dt);
    }

    while save_idx < saveat.len() {
        results.push(SolPointG {
            t: saveat[save_idx],
            u: u.clone(),
        });
        save_idx += 1;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exponential_decay() {
        // du/dt = -k*u, u(0) = 1.0, k = 0.1
        // Exact: u(t) = exp(-0.1*t)
        let k = 0.1;
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -k * u[0];
        };
        let saveat = vec![1.0, 5.0, 10.0, 20.0];
        let opts = OdeSolverOptions::default();
        let result = solve_ode(&rhs, &[1.0], (0.0, 20.0), &[], &saveat, &opts);

        assert_eq!(result.len(), saveat.len());
        for (sol, &t) in result.iter().zip(saveat.iter()) {
            let exact = (-k * t).exp();
            assert_relative_eq!(sol.u[0], exact, epsilon = 1e-4);
            assert_relative_eq!(sol.t, t, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_growth() {
        // du/dt = 1.0, u(0) = 0.0
        // Exact: u(t) = t
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let saveat = vec![1.0, 5.0, 10.0];
        let opts = OdeSolverOptions::default();
        let result = solve_ode(&rhs, &[0.0], (0.0, 10.0), &[], &saveat, &opts);

        for (sol, &t) in result.iter().zip(saveat.iter()) {
            assert_relative_eq!(sol.u[0], t, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_two_state_system() {
        // du1/dt = -u1, du2/dt = u1 (transfer from cpt 1 to cpt 2)
        // u1(t) = exp(-t), u2(t) = 1 - exp(-t)
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -u[0];
            du[1] = u[0];
        };
        let saveat = vec![1.0, 5.0, 10.0];
        let opts = OdeSolverOptions::default();
        let result = solve_ode(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &saveat, &opts);

        for (sol, &t) in result.iter().zip(saveat.iter()) {
            assert_relative_eq!(sol.u[0], (-t).exp(), epsilon = 1e-4);
            assert_relative_eq!(sol.u[1], 1.0 - (-t).exp(), epsilon = 1e-4);
        }
    }

    #[test]
    fn test_zero_span_returns_initial() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let saveat = vec![5.0];
        let opts = OdeSolverOptions::default();
        let result = solve_ode(&rhs, &[42.0], (5.0, 5.0), &[], &saveat, &opts);
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].u[0], 42.0, epsilon = 1e-12);
    }

    /// Primitive-level degenerate oracle: a constant accumulation rate `λ` is the
    /// ODE form of a constant hazard, whose analytic event time for `CHZ = −log u`
    /// is `t* = −log(u)/λ`. Here `dCHZ/dt = 2`, `threshold = 10` ⇒ exact crossing at
    /// `t* = 5`. The Hermite interpolant is exact on a linear trajectory, so the
    /// bisection should hit `5` to round-off.
    #[test]
    fn until_threshold_constant_rate_is_exact() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 2.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 100.0), &[], &opts, 0, 10.0) {
            ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 5.0, epsilon = 1e-9),
            other => panic!("expected Crossed(5.0), got {other:?}"),
        }
    }

    /// Drug-driven hazard with a closed form: a decaying drug `dC/dt = −0.5·C`,
    /// `C(0)=10`, feeds `dCHZ/dt = 0.1·C` ⇒ `CHZ(t) = 2·(1 − e^{−0.5t})`. Solving
    /// `CHZ = 1` gives `t* = 2·ln 2 ≈ 1.386294`. Exercises a non-zero monitor index
    /// (`CHZ` is state 1) and a genuinely curved accumulator.
    #[test]
    fn until_threshold_decaying_drug_hazard_matches_closed_form() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -0.5 * u[0]; // concentration
            du[1] = 0.1 * u[0]; // cumulative hazard
        };
        let opts = OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        };
        let mut u = [10.0, 0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1000.0), &[], &opts, 1, 1.0) {
            ThresholdCrossing::Crossed(t) => {
                assert_relative_eq!(t, 2.0 * std::f64::consts::LN_2, epsilon = 1e-5)
            }
            other => panic!("expected Crossed, got {other:?}"),
        }
    }

    /// Threshold never reached within the span ⇒ `ReachedEnd`, and `u` is advanced
    /// in place to the end-of-span state (so a segmented caller can carry it).
    #[test]
    fn until_threshold_not_reached_advances_state() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        let outcome = solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 1000.0);
        assert_eq!(outcome, ThresholdCrossing::ReachedEnd);
        assert_relative_eq!(u[0], 10.0, epsilon = 1e-6);
    }

    /// A decreasing monitored state (negative rate ⇒ negative hazard) is not a
    /// censor — it invalidates the crossing argument and must be a hard failure.
    #[test]
    fn until_threshold_negative_rate_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [5.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 100.0) {
            ThresholdCrossing::Failed(msg) => assert!(msg.contains("non-monotone"), "msg: {msg}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// Already at/above the threshold at the span start ⇒ `Crossed(t0)` without
    /// stepping (the `u ≈ 1 ⇒ threshold ≈ 0` edge, or a non-zero initial state).
    #[test]
    fn until_threshold_already_crossed_at_start() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [5.0];
        match solve_ode_until_threshold(&rhs, &mut u, (2.0, 10.0), &[], &opts, 0, 3.0) {
            ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 2.0, epsilon = 1e-12),
            other => panic!("expected Crossed(2.0), got {other:?}"),
        }
    }

    /// A non-finite trajectory cannot yield a meaningful crossing ⇒ hard failure,
    /// not a silent censor at the horizon.
    #[test]
    fn until_threshold_non_finite_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 10.0) {
            ThresholdCrossing::Failed(_) => {}
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// A zero-width span reaches its end immediately, without stepping.
    #[test]
    fn until_threshold_zero_span_reaches_end() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        assert_eq!(
            solve_ode_until_threshold(&rhs, &mut u, (5.0, 5.0), &[], &opts, 0, 10.0),
            ThresholdCrossing::ReachedEnd
        );
    }

    /// Exhausting the step budget before reaching `t_end` is a failure, not a
    /// censor — a censored draw requires a *clean* integration to the horizon.
    #[test]
    fn until_threshold_step_budget_exhausted_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions {
            max_steps: 2,
            ..OdeSolverOptions::default()
        };
        let mut u = [0.0];
        // Threshold far beyond what 2 steps can reach within the (huge) span.
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1e6), &[], &opts, 0, 1e9) {
            ThresholdCrossing::Failed(msg) => assert!(msg.contains("budget"), "msg: {msg}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// `solve_ode_g` over `Dual2` must reproduce the closed-form sensitivities of
    /// `du/dt = −k·u`, `u(0)=1` ⇒ `u(t)=e^{−kt}`, `∂u/∂k=−t·e^{−kt}`,
    /// `∂²u/∂k²=t²·e^{−kt}`.
    #[test]
    fn solve_ode_g_sensitivity_matches_closed_form() {
        use crate::sens::dual2::Dual2;
        let rhs = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = -(p[0] * u[0]);
        };
        let opts = OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        };
        let k = Dual2::<1>::var(0.1, 0);
        let u0 = [Dual2::<1>::constant(1.0)];
        let res = solve_ode_g(&rhs, &u0, (0.0, 2.0), &[k], &[2.0], &opts);
        let u = &res[0].u[0];
        let e = (-0.2_f64).exp();
        assert_relative_eq!(u.value, e, max_relative = 1e-6);
        assert_relative_eq!(u.grad[0], -2.0 * e, max_relative = 1e-5);
        assert_relative_eq!(u.hess[0][0], 4.0 * e, max_relative = 1e-5);
    }

    /// The `Dual2` integration's value must track the scalar `solve_ode`.
    #[test]
    fn solve_ode_g_value_matches_scalar() {
        use crate::sens::dual2::Dual2;
        let rhs_d = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = -(p[0] * u[0]);
        };
        let rhs_f = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -p[0] * u[0];
        };
        let opts = OdeSolverOptions::default();
        let saveat = [1.0, 5.0, 10.0];
        let rd = solve_ode_g(
            &rhs_d,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 10.0),
            &[Dual2::<1>::var(0.1, 0)],
            &saveat,
            &opts,
        );
        let rf = solve_ode(&rhs_f, &[1.0], (0.0, 10.0), &[0.1], &saveat, &opts);
        for (a, b) in rd.iter().zip(rf.iter()) {
            assert_relative_eq!(a.u[0].value, b.u[0], max_relative = 1e-9, epsilon = 1e-12);
        }
    }

    #[test]
    fn solve_ode_with_stats_counts_attempts() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -10.0 * u[0];
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(
            stats.attempted_steps,
            stats.accepted_steps + stats.rejected_steps
        );
        assert!(stats.accepted_steps > 0, "stats = {stats:?}");
        assert!(stats.rejected_steps > 0, "stats = {stats:?}");
        assert_eq!(stats.min_step_clamped_steps, 0);
    }

    #[test]
    fn solve_ode_with_stats_counts_min_step_clamped_accepts() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -100.0 * u[0];
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            min_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let _ = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, 1);
        assert_eq!(stats.accepted_steps, 1);
        assert_eq!(stats.rejected_steps, 0);
        assert_eq!(stats.min_step_clamped_steps, 1);
    }

    #[test]
    fn solve_ode_with_stats_counts_nan_blowup_as_min_step_clamped() {
        // A diverging RHS produces a non-finite err_norm. Pinned at min_dt the
        // step is force-accepted to guarantee progress; the clamp counter must
        // still flag it (`!(err_norm <= 1.0)` catches NaN, where `err_norm > 1.0`
        // would not), so a NaN-diverging integration is not mistaken for clean.
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            min_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let _ = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, 1);
        assert_eq!(stats.accepted_steps, 1);
        assert_eq!(stats.rejected_steps, 0);
        assert_eq!(stats.min_step_clamped_steps, 1);
    }

    #[test]
    fn solve_ode_breaks_repeated_min_step_clamps_before_max_steps() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-6,
            min_dt: 1e-6,
            max_steps: 10_000,
            abstol: 1e-12,
            reltol: 1e-12,
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(
            stats.min_step_clamped_steps,
            MAX_CONSECUTIVE_MIN_STEP_CLAMPS
        );
    }

    /// Regression for #603 review #4: a *finite-but-stiff* min-step clamp (large finite local
    /// error pinned at `min_dt`) must NOT trip the divergence break — only non-finite errors
    /// do. Otherwise the remaining save points get frozen-padded with finite-but-wrong values
    /// the likelihood would silently accept. High-frequency finite forcing keeps every step
    /// clamped with a finite error, so the solve runs its full `max_steps` budget (>> the
    /// clamp limit) and returns finite predictions rather than breaking at the limit.
    #[test]
    fn solve_ode_does_not_break_on_finite_stiff_min_step_clamps() {
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        // Ran the full budget — no early break at MAX_CONSECUTIVE_MIN_STEP_CLAMPS …
        assert_eq!(stats.attempted_steps, opts.max_steps);
        assert!(stats.attempted_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        // … the segment really was in the stiff min-step-clamp regime …
        assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        // … and the predictions stayed finite (not NaN-padded).
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].is_finite());
    }

    /// Dual path mirrors [`solve_ode_does_not_break_on_finite_stiff_min_step_clamps`].
    #[test]
    fn solve_ode_g_does_not_break_on_finite_stiff_min_step_clamps() {
        use crate::sens::dual2::Dual2;
        let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
            du[0] = Dual2::constant(1e6 * (t * 1e7).sin());
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_g_with_stats(
            &rhs,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, opts.max_steps);
        assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].value.is_finite());
    }

    #[test]
    fn solve_ode_g_breaks_repeated_min_step_clamps_like_scalar() {
        use crate::sens::dual2::Dual2;
        let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = Dual2::constant(f64::NAN);
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-6,
            min_dt: 1e-6,
            max_steps: 10_000,
            abstol: 1e-12,
            reltol: 1e-12,
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_g_with_stats(
            &rhs,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(
            stats.min_step_clamped_steps,
            MAX_CONSECUTIVE_MIN_STEP_CLAMPS
        );
    }

    #[test]
    fn solve_ode_g_with_stats_matches_scalar_step_pattern() {
        use crate::sens::dual2::Dual2;
        let rhs_f = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -p[0] * u[0];
        };
        let rhs_d = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = -(p[0] * u[0]);
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let saveat = [1.0];
        let mut stats_f = OdeSolverStats::default();
        let mut stats_d = OdeSolverStats::default();

        let _ = solve_ode_with_stats(
            &rhs_f,
            &[1.0],
            (0.0, 1.0),
            &[10.0],
            &saveat,
            &opts,
            Some(&mut stats_f),
        );
        let _ = solve_ode_g_with_stats(
            &rhs_d,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[Dual2::<1>::var(10.0, 0)],
            &saveat,
            &opts,
            Some(&mut stats_d),
        );

        assert_eq!(stats_d, stats_f);
    }

    #[test]
    fn test_params_passed_to_rhs() {
        // du/dt = p[0] * u, u(0) = 1
        // with p[0] = -0.5: u(t) = exp(-0.5*t)
        let rhs = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = p[0] * u[0];
        };
        let saveat = vec![2.0];
        let opts = OdeSolverOptions::default();
        let result = solve_ode(&rhs, &[1.0], (0.0, 2.0), &[-0.5], &saveat, &opts);
        assert_relative_eq!(result[0].u[0], (-1.0_f64).exp(), epsilon = 1e-4);
    }

    /// Regression guard for FSAL (First Same As Last) stage reuse.
    ///
    /// Structural rather than count-based: with FSAL the k1 of step k+1 is
    /// reused (swapped in) from the prior step's k7, so the rhs closure is
    /// **never** invoked twice in a row at the same `(u, t)`. Without FSAL,
    /// k7 of step k and k1 of step k+1 are two separate rhs calls at
    /// bit-identical `(u_new, t_new)` — an adjacent duplicate in the call
    /// sequence. Recording the `(u, t)` of every rhs call and scanning for
    /// any adjacent duplicate detects FSAL removal regardless of iteration
    /// count, controller, tolerance, or host platform.
    ///
    /// The earlier modular check `(n - 1) % 6 == 0` was unsharp: FSAL-off
    /// produces `n = 7N`, which satisfies the check whenever `N ≡ 1 (mod 6)`
    /// — a 1-in-6 silent-pass rate across the population of iteration counts
    /// the test might land on.
    #[test]
    fn test_fsal_reuses_last_stage() {
        use std::cell::RefCell;
        // Record `(u[0], t)` bit patterns at every rhs invocation. Bit
        // equality (rather than `==` on f64) sidesteps any ambiguity about
        // NaN / signed-zero corner cases — though for this smooth ODE there
        // are none.
        let calls: RefCell<Vec<(u64, u64)>> = RefCell::new(Vec::new());
        let rhs = |u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            calls.borrow_mut().push((u[0].to_bits(), t.to_bits()));
            du[0] = -0.1 * u[0];
        };
        let opts = OdeSolverOptions::default();
        let _ = solve_ode(&rhs, &[1.0], (0.0, 20.0), &[], &[20.0], &opts);
        let calls = calls.into_inner();

        assert!(
            calls.len() > 7,
            "solver did not perform multiple steps (calls = {})",
            calls.len(),
        );

        let dup_at = calls.windows(2).position(|w| w[0] == w[1]);
        assert!(
            dup_at.is_none(),
            "FSAL appears inactive: rhs called twice consecutively at the \
             same (u, t) at call index {} of {} (k7 of step k and k1 of \
             step k+1 should reuse a single evaluation).",
            dup_at.unwrap(),
            calls.len(),
        );
    }
}
