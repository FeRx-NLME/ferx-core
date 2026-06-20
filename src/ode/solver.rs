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
    // (analytical / sensitivity / autodiff).
    const I_EXP: f64 = 1.0 / 5.0; // 0.20 — I-controller exponent for order p=5

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

        if err_norm <= 1.0 || dt_eff <= opts.min_dt {
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
        }
        // On reject: (u, t) is unchanged, so the existing k1 is still rhs(u, t)
        // for the next attempt; nothing to do.

        // Adapt step size (memoryless I-controller — see note above).
        let safety = 0.9;
        let factor = if err_norm > 1e-15 {
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

        if err_norm <= 1.0 || dt_eff <= opts.min_dt {
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
        }

        let safety = 0.9;
        let factor = if err_norm > 1e-15 {
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
