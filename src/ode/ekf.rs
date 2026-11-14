//! Extended Kalman Filter for SDE-based ODE models.
//!
//! Given an Itô SDE  dX = f(X,t,θ) dt + diag(√σ²_w) dW  the EKF propagates
//! both the mean state (via the shared RK45 solver) and the state covariance
//! matrix P between events, then applies a scalar Kalman update at each
//! observation time.
//!
//! Covariance prediction (continuous-discrete EKF):
//!   P̃ = F·P·Fᵀ + Q·Δt
//! where F = ∂f/∂X is the **exact** state Jacobian (the right-hand side's forward tangent,
//! one `DualT<1, T>` pass per state) and Q = diag(σ²_w).
//!
//! Measurement update at obs j (scalar observation of state obs_cmt):
//!   S   = P̃[c,c] + R_j          (innovation variance)
//!   K   = P̃[:,c] / S             (Kalman gain vector)
//!   P⁺  = (I - K·eₓᵀ) · P̃      (updated covariance)
//!
//! Only P[obs_cmt, obs_cmt] is returned per observation — the caller adds it
//! to the residual variance to form V_total.
//!
//! The filter is written once over a number type `T` ([`solve_ekf_g`]): `T = f64` is the
//! value path, and the recording type records it into a subject's compiled plan, so the
//! mean, the covariance and the Jacobian all carry exact derivatives. Matrices are flat
//! row-major `Vec<T>`; the state is a handful of compartments.

use crate::dosing::{
    is_real_infusion, note_ss_nonconvergence_if_capped, SsStopTracker, SS_EQUILIBRATION_CYCLES,
};
use crate::ode::predictions::active_infusion_doses;
use crate::ode::solver::{solve_ode_g_with_auto_state, OdeAutoSwitchState, OdeSolverOptions};
use crate::sens::jet::DualT;
use crate::sens::num::vm::VmNum;
use crate::sens::num::BranchCmp;
use crate::types::DoseEvent;

/// A right-hand side `du = f(u, p, t)` the filter can evaluate over any number type, so the
/// same code gives the mean (`T = f64` or the recording type) and, over `DualT<1, T>`, the
/// exact Jacobian. `anchors` are the clocks the `[odes]` block reads besides `t`: the first
/// dose time (`TAFD`) and the last effective dose time (`TAD`), `NaN` where absent. The `TAD`
/// anchor is a number of the filter's type: a steady-state quiet window's is the (possibly
/// modeled) infusion window's negative.
pub(crate) trait EkfRhs {
    fn eval<U: VmNum>(&self, u: &[U], p: &[U], t: U, anchors: Anchors<U>, du: &mut [U]);

    /// The compiled `[odes]` program this right-hand side evaluates, when it is one: what the
    /// mean integration's Rosenbrock step differentiates for its exact Jacobian
    /// ([`integrate_mean`]). `None` for a host closure, which the step differences.
    fn program(&self) -> Option<&crate::parser::model_parser::OdeRhsProgram> {
        None
    }
}

/// The `(TAFD, TAD)` clock anchors of [`EkfRhs::eval`].
pub(crate) type Anchors<T> = (f64, T);

/// The compiled `[odes]` program as the filter's right-hand side.
pub(crate) struct ProgramRhs<'a>(pub &'a crate::parser::model_parser::OdeRhsProgram);

impl EkfRhs for ProgramRhs<'_> {
    fn program(&self) -> Option<&crate::parser::model_parser::OdeRhsProgram> {
        Some(self.0)
    }

    fn eval<U: VmNum>(&self, u: &[U], p: &[U], t: U, anchors: Anchors<U>, du: &mut [U]) {
        let (mut vars, mut stack) = (Vec::new(), Vec::new());
        crate::sens::ode_provider::eval_rhs_at_offset(
            self.0,
            u,
            p,
            t,
            U::from_f64(0.0),
            anchors.0,
            anchors.1,
            du,
            &mut vars,
            &mut stack,
        );
    }
}

/// `f` over `rhs` with the infusion rates `active` added, as an integrator's right-hand side.
fn forced_eval<T: VmNum, R: EkfRhs>(
    rhs: &R,
    active: &[(usize, T)],
    u: &[T],
    p: &[T],
    t: T,
    anchors: Anchors<T>,
    du: &mut [T],
) {
    rhs.eval::<T>(u, p, t, anchors, du);
    for &(cmt, rate) in active {
        if cmt < du.len() {
            du[cmt] = du[cmt] + rate;
        }
    }
}

type Mat<T> = Vec<T>;

fn zeros<T: VmNum>(n: usize) -> Mat<T> {
    vec![T::from_f64(0.0); n * n]
}

/// The Jacobian `F = ∂f/∂X` of the (forced) right-hand side at `u`: column `j` is the forward
/// tangent in direction `e_j`. A recording type records the tangent as expressions, so the
/// covariance's derivatives with respect to θ and η flow through `F` exactly.
#[allow(clippy::too_many_arguments)]
fn jacobian<T: VmNum, R: EkfRhs>(
    rhs: &R,
    active: &[(usize, T)],
    u: &[T],
    p: &[T],
    t: T,
    anchors: Anchors<T>,
    n: usize,
) -> Mat<T> {
    type D<T> = DualT<1, T>;
    let (zero, one) = (T::from_f64(0.0), T::from_f64(1.0));
    let lift = |x: T| D::<T> { v: x, g: [zero] };
    let pd: Vec<D<T>> = p.iter().map(|&x| lift(x)).collect();
    let act: Vec<(usize, D<T>)> = active.iter().map(|&(c, r)| (c, lift(r))).collect();
    let mut jac = zeros::<T>(n);
    let mut ud: Vec<D<T>> = u.iter().map(|&x| lift(x)).collect();
    let mut du = vec![lift(zero); n];
    let anchors = (anchors.0, lift(anchors.1));
    for j in 0..n {
        ud[j].g = [one];
        forced_eval::<D<T>, R>(rhs, &act, &ud, &pd, lift(t), anchors, &mut du);
        for i in 0..n {
            jac[i * n + j] = du[i].g[0];
        }
        ud[j].g = [zero];
    }
    jac
}

/// One Crank–Nicolson (Cayley) step of the Riccati equation `dP/dt = F·P + P·Fᵀ + Q` with
/// `F` evaluated at the midpoint state: `P⁺ = M⁻¹·(A·P·Aᵀ + Q·Δt)·M⁻ᵀ`, `A = I + F·Δt/2`,
/// `M = I − F·Δt/2`, symmetrised, the diagonal clamped to `≥ 0`.
///
/// Second order, and A-stable: a decaying mode contracts at every step length. The explicit
/// Euler step it replaces multiplied a state's variance by `1 − 2k·Δt` per step, which grows
/// for a rate `k > 1/Δt` (an absorption `KA > 2 h⁻¹` at [`RICCATI_DT_MAX`]): the variance of a
/// noisy fast compartment diverged, and a noiseless one's sensitivity did, overflowing the
/// adjoint through a long steady-state train. The caller sub-steps at [`RICCATI_DT_MAX`].
#[allow(clippy::too_many_arguments)]
fn propagate_covariance<T: VmNum, R: EkfRhs>(
    rhs: &R,
    active: &[(usize, T)],
    p_mat: &[T],
    u_mid: &[T],
    params: &[T],
    t_mid: T,
    anchors: Anchors<T>,
    dt: T,
    q_diag: &[T],
    n: usize,
) -> Mat<T> {
    let f = jacobian(rhs, active, u_mid, params, t_mid, anchors, n);
    let zero = T::from_f64(0.0);
    // `I ± F·Δt/2`, row-major.
    let id_pm = |s: f64| -> Mat<T> {
        (0..n * n)
            .map(|k| {
                let id = if k / n == k % n { 1.0 } else { 0.0 };
                T::from_f64(id) + T::from_f64(s * 0.5) * dt * f[k]
            })
            .collect()
    };
    let (mut m, a) = (id_pm(-1.0), id_pm(1.0));
    let mul = |x: &[T], y: &[T], yt: bool| -> Mat<T> {
        let mut z = zeros::<T>(n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let yv = if yt { y[j * n + k] } else { y[k * n + j] };
                    z[i * n + j] = z[i * n + j] + x[i * n + k] * yv;
                }
            }
        }
        z
    };
    let mut r = mul(&mul(&a, p_mat, false), &a, true);
    for i in 0..n {
        r[i * n + i] = r[i * n + i] + q_diag[i] * dt;
    }
    let mut piv = vec![zero; crate::sens::linsolve::lu_piv_len::<T>(n)];
    if !crate::sens::linsolve::lu_factor_in_place_g(&mut m, &mut piv, n) {
        return vec![T::from_f64(f64::NAN); n * n];
    }
    // `M⁻¹·X·M⁻ᵀ` as two column solves: `Y = M⁻¹X`, then `M⁻¹Yᵀ` (its transpose, and `X` is
    // symmetric).
    let solve_t = |x: &[T]| -> Mat<T> {
        let mut out = zeros::<T>(n);
        for c in 0..n {
            let mut col: Vec<T> = (0..n).map(|i| x[i * n + c]).collect();
            crate::sens::linsolve::lu_solve_in_place_g(&m, &piv, n, &mut col);
            for (i, v) in col.into_iter().enumerate() {
                out[c * n + i] = v;
            }
        }
        out
    };
    let p_new = solve_t(&solve_t(&r));
    let half = T::from_f64(0.5);
    let mut out = zeros::<T>(n);
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = (p_new[i * n + j] + p_new[j * n + i]) * half;
        }
        if out[i * n + i].br_cmp_c(BranchCmp::Lt, 0.0) {
            out[i * n + i] = zero;
        }
    }
    out
}

/// Longest sub-step of the Riccati propagation. 0.5 h per step keeps the relative
/// error under ~3% for typical PK; the walk and the steady-state run-in both sub-step at
/// this, so a steady state expanded into explicit records and one seeded from an `SS=1`
/// record propagate `P` through the same quadrature (#1260).
const RICCATI_DT_MAX: f64 = 0.5;

/// Advance `p_mat` from `(t_prev, u_prev)` to `(t, u)` along a mean trajectory the solver
/// has already produced, sub-stepping the Riccati step at [`RICCATI_DT_MAX`] with the
/// state linearly interpolated across each sub-step midpoint.
///
/// The times are numbers of the filter's type: an end that moves with the parameters (a
/// modeled or `F`-reshaped infusion window's end) carries its derivatives into every
/// sub-step's length and midpoint, so the quadrature is differentiated as it is computed. The
/// sub-step count is read on the value, the comparisons that fix it guarded.
#[allow(clippy::too_many_arguments)]
fn propagate_covariance_between<T: VmNum, R: EkfRhs>(
    rhs: &R,
    active: &[(usize, T)],
    mut p_mat: Mat<T>,
    t_prev: T,
    u_prev: &[T],
    t: T,
    u: &[T],
    params: &[T],
    anchors: Anchors<T>,
    q_diag: &[T],
    n: usize,
) -> Mat<T> {
    let dt = t - t_prev;
    if dt.br_cmp_c(BranchCmp::Le, 1e-15) {
        return p_mat;
    }
    let n_steps = ((dt.seen_value() / RICCATI_DT_MAX).ceil() as usize).max(1);
    if !dt.is_constant() {
        // The verdicts that give `n_steps`: `(n − 1)·Δ < dt ≤ n·Δ` (`dt / Δ` is exact, Δ a
        // power of two).
        let _ = dt.br_cmp_c(BranchCmp::Le, n_steps as f64 * RICCATI_DT_MAX);
        if n_steps > 1 {
            let _ = dt.br_cmp_c(BranchCmp::Gt, (n_steps - 1) as f64 * RICCATI_DT_MAX);
        }
    }
    let dt_sub = dt / T::from_f64(n_steps as f64);
    for s in 0..n_steps {
        let alpha = (s as f64 + 0.5) / n_steps as f64;
        let alpha_mid = T::from_f64(alpha);
        let u_mid: Vec<T> = u_prev
            .iter()
            .zip(u)
            .map(|(&a, &b)| a + alpha_mid * (b - a))
            .collect();
        let t_mid = t_prev + alpha_mid * dt;
        p_mat = propagate_covariance(
            rhs, active, &p_mat, &u_mid, params, t_mid, anchors, dt_sub, q_diag, n,
        );
    }
    p_mat
}

/// Kalman update for a scalar observation of compartment `obs_cmt`.
///
/// Returns `(P_updated, p_obs_cmt)` where `p_obs_cmt` is P[obs_cmt, obs_cmt]
/// *before* the update — the component the caller adds to residual variance.
fn kalman_update<T: VmNum>(p_mat: &[T], obs_cmt: usize, r_obs: T, n: usize) -> (Mat<T>, T) {
    let p_cc = p_mat[obs_cmt * n + obs_cmt];
    let s = p_cc + r_obs; // innovation variance
    if s.br_cmp_c(BranchCmp::Le, 0.0) {
        return (p_mat.to_vec(), p_cc);
    }
    // Gain K = P[:,obs_cmt] / S; update P⁺ = (I - K·eₒᵀ)·P, symmetrised.
    let k: Vec<T> = (0..n).map(|i| p_mat[i * n + obs_cmt] / s).collect();
    let mut p_new = p_mat.to_vec();
    for i in 0..n {
        for j in 0..n {
            p_new[i * n + j] = p_new[i * n + j] - k[i] * p_mat[obs_cmt * n + j];
        }
    }
    let half = T::from_f64(0.5);
    let mut out = zeros::<T>(n);
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = (p_new[i * n + j] + p_new[j * n + i]) * half;
        }
    }
    (out, p_cc)
}

/// The mean state over `span` saved at `saveat`, under the forced right-hand side. The span's
/// ends are numbers of the filter's type. A recording type records the segment over their
/// values (`PkNum::span_hint`), so an end that moves with the parameters moves the solve at run
/// time; their derivatives are the flow over their jets (`rk4_dual`, exact to every order the
/// recordings carry): the solve starts from the state flowed back by a moving start's jet, and
/// its point at a moving end is flowed on by the end's. On `f64` both are no-ops.
#[allow(clippy::too_many_arguments)]
fn integrate_mean<T: VmNum, R: EkfRhs>(
    rhs: &R,
    active: &[(usize, T)],
    u: &[T],
    span: (T, T),
    params: &[T],
    anchors: Anchors<T>,
    saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<crate::ode::solver::SolPointG<T>> {
    let f = |u: &[T], p: &[T], t: T, du: &mut [T]| forced_eval(rhs, active, u, p, t, anchors, du);
    // The step's `J` and `f_t` are the program's exact tangents at `T = f64`, as on the
    // prediction paths, not differences of `f` — the filter's covariance differentiates the
    // same `F` (`jacobian`), so mean and covariance see one Jacobian. The infusion rates added
    // by `forced_eval` are constants of the segment and move neither.
    let pj = rhs
        .program()
        .map(|prog| crate::sens::ode_provider::ProgJac::new(prog, anchors.0, anchors.1, &[]));
    let jf = |u: &[T], p: &[T], t: T, j: &mut [T], ft: &mut [T]| {
        pj.as_ref().is_some_and(|pj| pj.eval(u, p, t, j, ft))
    };
    let jr: &crate::ode::rosenbrock::JacFnG<'_, T> = &jf;
    let _guard = pj
        .is_some()
        .then(|| crate::ode::rosenbrock::RhsGuard::register_jac_g::<T>(&f, &jr));
    T::span_hint(span.0, span.1);
    let (t0, t1) = (span.0.span_value(), span.1.span_value());
    // `x` flowed over the value-zero `h` from offset `s0` past `t_v`.
    let flow = |x: &[T], t_v: f64, s0: T, h: T| -> Vec<T> {
        crate::sens::ode_provider::rk4_dual(x, s0, h, &mut |y: &[T], s: T, out: &mut [T]| {
            forced_eval(rhs, active, y, params, T::from_f64(t_v) + s, anchors, out)
        })
    };
    let u0 = match jet_part(span.0) {
        Some(d) => flow(u, t0, d, -d),
        None => u.to_vec(),
    };
    // Uncertified closure: never the exact path (see `solve_ode`).
    let mut sol = solve_ode_g_with_auto_state(
        &f,
        &u0,
        (t0, t1),
        params,
        saveat,
        &opts.for_integration(),
        &mut OdeAutoSwitchState::default(),
    );
    // Taken by the segment that recorded it; nothing else may.
    let _ = T::take_span();
    if let (Some(d), Some(last)) = (jet_part(span.1), sol.last_mut()) {
        if last.t.to_bits() == t1.to_bits() {
            last.u = flow(&last.u, t1, T::from_f64(0.0), d);
        }
    }
    sol
}

/// A time's derivatives alone (its value zero), when it moves with the inputs of a recording.
fn jet_part<T: VmNum>(t: T) -> Option<T> {
    (T::RECORDS_BRANCHES && !t.is_constant()).then(|| t - t.value_only())
}

/// A solve's saved point's time as a number of the filter's type: the span's (possibly
/// moving) `end` when it is that end, the constant otherwise.
fn point_time<T: VmNum>(t: f64, end: T) -> T {
    if t.to_bits() == end.seen_value().to_bits() {
        end
    } else {
        T::from_f64(t)
    }
}

/// Periodic steady state of the EKF's `(mean, covariance)` pair for an `SS=1` record
/// (#1260): the state the filter would hold at the record had the dose been given every
/// `II` since forever, with no observation in between.
///
/// The pulse train is expanded the way the ODE path's nonlinear fallback does it
/// (`equilibrate_ss_pk_state`): from `u = 0, P = 0`, cycles of *(apply dose; integrate
/// `II`)* on that fallback's cycle-local clock — bolus, or an `F`-reshaped active window
/// followed by a quiet one — until the concatenated `(u, P)` stops moving by [`SsStopTracker`]'s
/// criterion (read on the values, as the dual walks do) or [`SS_EQUILIBRATION_CYCLES`] is
/// spent, with the #867 non-convergence warning on a spent cap. No exact linear fixed point
/// here: a bolus leaves `P` untouched, so `P`'s cycle map is the continuous Riccati flow
/// over `II`, and the same quadrature the walk uses is what makes an explicit train and
/// this run-in agree to the stop tolerance. `P` is carried alongside `u` rather than seeded
/// after it because on a nonlinear right-hand side the Jacobian along the cycle depends on
/// the mean's own phase. A recording type records the cycle map once and the train as a
/// run-time search on the values ([`ss_train`]), differentiating the converged pair implicitly.
///
/// Cost, measured on the #1260 fixture (one objective evaluation, debug build): the run-in
/// early-stops after 16–21 cycles; 569 ms with the `SS=1` record against 996 ms with the 41
/// explicit records it stands for.
///
/// Returns the zero pair for the shapes the ODE run-in also declines: `II ≤ 0` or a compartment
/// outside the state vector.
#[allow(clippy::too_many_arguments)]
fn equilibrate_ss_ekf<T: VmNum, R: EkfRhs>(
    rhs: &R,
    n: usize,
    dose: &DoseEvent,
    f_bio: T,
    (inf_rate, t_inf): (T, T),
    params: &[T],
    q_diag: &[T],
    opts: &OdeSolverOptions,
) -> (Vec<T>, Mat<T>) {
    let zero = T::from_f64(0.0);
    let cmt_idx = dose.cmt_idx();
    if dose.ii <= 0.0 || cmt_idx >= n {
        return (vec![zero; n], zeros::<T>(n));
    }
    let is_inf = is_real_infusion(dose);
    // Overlapping pulses (`T_inf > II`): `m` earlier ones run through the whole cycle, the
    // newest for `r` (`dosing::ss_overlap_split_g`); `m = 0` otherwise.
    let (m, t_inf) = crate::dosing::ss_overlap_split_g(t_inf, dose.ii);
    let held = [(cmt_idx, inf_rate * T::from_f64(m as f64))];
    let held: &[(usize, T)] = if m > 0 { &held } else { &[] };
    // Every window opens on a clock whose origin is its cycle's pulse — `(0, II)` for a
    // bolus cycle, `(0, T_inf)` then `(0, II − T_inf)` for an infusion — the clock the ODE
    // bolus/infusion run-in integrates on (`equilibrate_ss_pk_state`'s exact solve and its
    // capped train alike), so a right-hand side reading `T` / `TIME` sees the same run-in on
    // both engines. `TAD` is anchored per window — `0` for a window opening at the pulse,
    // `−T_inf` for the quiet window that re-opens its clock after the active one — and `TAFD`
    // is left `NaN` (#1139).
    let advance = |active: &[(usize, T)],
                   u: &mut Vec<T>,
                   p_mat: Mat<T>,
                   t1: T,
                   anchors: Anchors<T>|
     -> Mat<T> {
        let sol = integrate_mean(
            rhs,
            active,
            u,
            (zero, t1),
            params,
            anchors,
            &[t1.seen_value()],
            opts,
        );
        let mut p_out = p_mat;
        let mut t_prev = zero;
        let mut u_prev = u.clone();
        for pt in &sol {
            let t = point_time(pt.t, t1);
            p_out = propagate_covariance_between(
                rhs, active, p_out, t_prev, &u_prev, t, &pt.u, params, anchors, q_diag, n,
            );
            t_prev = t;
            u_prev.clone_from(&pt.u);
        }
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        p_out
    };
    let inf_active: Vec<(usize, T)> = std::iter::once((cmt_idx, inf_rate))
        .chain(held.iter().copied())
        .collect();
    // One cycle of the train on the flat pair `[u; P]`: the dose, then its interval.
    let cycle = |z: &[T]| -> Option<Vec<T>> {
        let mut u = z[..n].to_vec();
        let mut p_mat = z[n..].to_vec();
        if is_inf {
            p_mat = advance(&inf_active, &mut u, p_mat, t_inf, (f64::NAN, zero));
            let quiet = T::from_f64(dose.ii) - t_inf;
            if quiet.br_cmp_c(BranchCmp::Gt, 0.0) {
                p_mat = advance(held, &mut u, p_mat, quiet, (f64::NAN, -t_inf));
            }
        } else {
            u[cmt_idx] = u[cmt_idx] + f_bio * T::from_f64(dose.amt);
            p_mat = advance(&[], &mut u, p_mat, T::from_f64(dose.ii), (f64::NAN, zero));
        }
        u.extend(p_mat);
        Some(u)
    };
    // A recording type (#2786) records the steady state as one call: the cycle map once, the
    // train on the values at run time (the very loop below), the derivatives implicit.
    let z = if T::RECORDS_BRANCHES {
        let solve: Box<crate::parser::model_parser::rhs_regcode::kernel_jit::SolveFn> =
            Box::new(move |maps| ss_train::<f64>(n, maps[0]));
        match T::ss_fixed_point(n + n * n, &[], &vec![0.0; n + n * n], &[&cycle], solve) {
            Some(r) => r.unwrap_or_else(|| vec![zero; n + n * n]),
            None => unreachable!("a recording type records a fixed-point call"),
        }
    } else {
        ss_train::<T>(n, &cycle).map_or_else(|| vec![zero; n + n * n], |(z, _)| z)
    };
    (z[..n].to_vec(), z[n..].to_vec())
}

/// The pulse train of [`equilibrate_ss_ekf`]: `cycle` applied from the zero pair until the
/// concatenated `(u, P)` stops moving by [`SsStopTracker`]'s criterion (read on the values) or
/// [`SS_EQUILIBRATION_CYCLES`] is spent, with the #867 non-convergence warning on a spent cap.
/// The pair and what it is for a compiled plan's derivatives (`Found`): the fixed point of
/// `cycle`, or the capped train's last iterate.
fn ss_train<T: VmNum>(
    n: usize,
    cycle: &dyn Fn(&[T]) -> Option<Vec<T>>,
) -> Option<(
    Vec<T>,
    crate::parser::model_parser::rhs_regcode::kernel_jit::Found,
)> {
    use crate::parser::model_parser::rhs_regcode::kernel_jit::Found;
    let mut z = vec![T::from_f64(0.0); n + n * n];
    let mut tracker = SsStopTracker::default();
    let mut converged = false;
    let mut cycles_run = 0;
    let mut pair = vec![0.0f64; n + n * n];
    for m in 0..SS_EQUILIBRATION_CYCLES {
        z = cycle(&z)?;
        cycles_run = m + 1;
        for (x, y) in pair.iter_mut().zip(&z) {
            *x = y.val();
        }
        if tracker.should_stop(m, &pair) {
            converged = true;
            break;
        }
    }
    let (incr_prev, incr_last, incr_mag) = tracker.recent_increments();
    note_ss_nonconvergence_if_capped(converged, incr_prev, incr_last, incr_mag);
    let found = if converged {
        Found::Fixed(0)
    } else {
        Found::Iterated {
            map: 0,
            count: cycles_run,
        }
    };
    Some((z, found))
}

/// An infusion's bioavailable `(rate, window)` (#419) as `T`s, the value
/// [`DoseEvent::bioavailable_infusion`]'s: a duration-defined dose scales its rate by `F`, a
/// rate-defined one its window. `modeled` is a modeled `RATE=-2` dose's `D` (duration-defined)
/// or `RATE=-1` dose's `R` (rate-defined), floored as `resolve_rate` floors it, so the
/// window and rate carry its derivatives; `None` reads the (resolved) dose record.
fn infusion_of<T: VmNum>(dose: &DoseEvent, f_bio: T, modeled: Option<T>) -> (T, T) {
    use crate::types::InfusionDef::{DurationDefined, RateDefined};
    let amt = T::from_f64(dose.amt);
    let (rate, dur) = match (dose.infusion_def, modeled) {
        (DurationDefined, Some(d)) => {
            let d = d.guard_floor(DoseEvent::DURATION_FLOOR);
            (amt / d, d)
        }
        (RateDefined, Some(r)) => {
            let r = r.guard_floor(DoseEvent::RATE_FLOOR);
            (r, amt / r)
        }
        (_, None) => (T::from_f64(dose.rate), T::from_f64(dose.duration)),
    };
    match dose.infusion_def {
        DurationDefined => (f_bio * rate, dur),
        RateDefined => (rate, f_bio * dur),
    }
}

/// One observation point returned by `solve_ekf_g`.
#[derive(Debug, Clone, Copy)]
pub struct EkfObsPoint<T = f64> {
    /// Predicted mean state value at the observable compartment.
    pub ipred: T,
    /// EKF state covariance at the observable compartment (P[obs_cmt, obs_cmt])
    /// *before* assimilating this observation. Add to residual variance for V_total.
    pub p_obs: T,
}

impl<T: VmNum> EkfObsPoint<T> {
    fn of(v: f64) -> Self {
        EkfObsPoint {
            ipred: T::from_f64(v),
            p_obs: T::from_f64(v),
        }
    }
}

/// The EKF mean at the observed compartment, as the prediction it is reported as.
///
/// A negative mean is clamped to `0`, as `ode_predictions` does. A `NaN` mean is **kept**
/// (#1539): it used to be rewritten to `0.0` here, which served a plausible zero for a state
/// the solver never integrated — once a state diverges the solver `NaN`-pads the rest of the
/// segment, and this was the one reader that turned that back into a finite number.
fn ekf_ipred<T: VmNum>(v: T) -> T {
    if v.br_cmp_c(BranchCmp::Lt, 0.0) {
        T::from_f64(0.0)
    } else {
        v
    }
}

/// Propagate mean and covariance through a subject's dose+obs timeline.
///
/// `rhs`, `n_states`, `obs_cmt_idx` mirror `OdeSpec`. `diffusion_var` is the
/// diagonal of Q (length == n_states). `r_of(j, ipred)` is observation `j`'s
/// measurement variance R given the filter's own prediction there, so the Kalman
/// update is correct for proportional and combined error models, where R depends on the
/// predicted value. The returned `p_obs` values are the pre-update EKF covariance
/// components and are not inflated by R. `dose_f_bio` is each dose's bioavailability
/// (`Fn`, #369) and `params` the PK parameters (both numbers of the filter's type, so their
/// derivatives flow through the mean and the covariance); the EKF path applies no lagtime.
/// `doses` are resolved (`resolve_subject_doses`); `modeled[k]`, when given, is dose `k`'s
/// modeled `D`/`R` as such a number ([`infusion_of`]), and empty for a caller on `f64`.
///
/// An infusion's window end moves with `F` (rate-defined) or its modeled `D`/`R`: it is a
/// break time of the filter's type, so the mean solve's span, the covariance quadrature's
/// sub-steps and their midpoints all carry its derivatives — the discrete filter is
/// differentiated as it is computed, with the timeline's order guarded.
///
/// Dose events are handled identically to `ode_predictions`: boluses add to
/// state; infusions inject a rate term into the wrapped RHS. Covariance is
/// reset to zero at initial time and propagated forward from there; an `SS=1`
/// record replaces both mean and covariance with their periodic steady state
/// ([`equilibrate_ss_ekf`], #1260).
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_ekf_g<T: VmNum, R: EkfRhs>(
    rhs: &R,
    n_states: usize,
    obs_cmt_idx: usize,
    diffusion_var: &[T],
    params: &[T],
    dose_f_bio: &[T],
    modeled: &[Option<T>],
    initial_state: &[T],
    doses: &[DoseEvent],
    obs_times: &[f64],
    r_of: &dyn Fn(usize, T) -> T,
    opts: OdeSolverOptions,
) -> Vec<EkfObsPoint<T>> {
    use std::collections::HashMap;
    let tf = T::from_f64;

    let n = n_states;
    let n_obs = obs_times.len();

    // Seed the EKF mean from the model's initial compartment amounts
    // (`init(state) = expr`); zeros for models without an init block. The
    // covariance still starts at zero — init sets the deterministic mean only.
    let mut u = if initial_state.len() == n {
        initial_state.to_vec()
    } else {
        vec![T::from_f64(0.0); n]
    };
    let mut p_mat = zeros::<T>(n);
    let mut results = vec![EkfObsPoint::<T>::of(0.0); n_obs];

    // The `[odes]` RHS reads TAFD and TAD from two clocks the walk anchors: the first dose
    // time for the whole subject, and the last effective dose time re-anchored per segment
    // (#1131: they were `NaN` once, through a params slice too short to hold them).
    let first_dose = crate::ode::predictions::earliest_dose_time(doses);
    let first_dose = if first_dose.is_finite() {
        first_dose
    } else {
        f64::NAN
    };
    // Every observation index at a given time, not just the last one: multiple records
    // can share a time (simultaneous PK/PD samples on different compartments), and a
    // `HashMap<u64, usize>` keeps only one of them — the rest would keep the
    // `EkfObsPoint { ipred: 0.0, p_obs: 0.0 }` prefill from above and feed a finite,
    // plausible zero into the likelihood. The boundary-read path below was hardened for
    // exactly this in #1226; the in-segment path was not, and measured on #1263
    // `obs_times = [2.0, 2.0, 8.0]` returned `ipred = 0.0` for the first of the pair
    // (#1263 review). Mirrors `predictions::build_obs_index_map`.
    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    // Build break times (same logic as ode_predictions).
    //
    // The first break is the subject's own integration start, **not** a literal `0.0`:
    // for a subject whose records begin at `t > 0` a leading `[0, t_first]` segment is
    // pure fiction, and it is not free — `propagate_covariance` runs it, so `p_mat`
    // reaches the first observation carrying `q · t_first` of process noise the ODE path
    // never accumulates (measured 13× on #1263), and an `init(state)` mean decays across
    // it before the twin has started. `doses`/`obs_times` are what this engine is given,
    // so the start is the min over them rather than `subject_integration_start`'s wider
    // event set — the caller (`ode_predictions_ekf_with_diffusion`) passes the subject's
    // full dose and observation lists, and pk-only/reset rows are not honoured here at
    // all (`W_SDE_RESET`).
    let t_first = obs_times
        .iter()
        .chain(doses.iter().map(|d| &d.time))
        .cloned()
        .fold(f64::INFINITY, f64::min);
    // `fold(0.0, max)` matches `ode_predictions` (`:3225`), then clamped up to `t_first`.
    // Without the clamp a subject with **no observations** — `t_last` folds to `0.0`
    // while `t_first` is its first dose — pushes a `0.0` back into `break_times` and
    // restores the very phantom leading segment this seed removes. Nothing is scored for
    // such a subject, so it costs only wasted integration, but it would make the code
    // contradict the comment above.
    let t_last = obs_times
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(if t_first.is_finite() { t_first } else { 0.0 });
    // Each infusion's bioavailable `(rate, window)`, its derivatives with it.
    let inf: Vec<(T, T)> = doses
        .iter()
        .enumerate()
        .map(|(k, d)| {
            if is_real_infusion(d) {
                infusion_of(d, dose_f_bio[k], modeled.get(k).copied().flatten())
            } else {
                (tf(0.0), tf(0.0))
            }
        })
        .collect();
    // `subject_integration_start`'s own dose-free fallback, spelled here because this
    // engine holds slices rather than a `&Subject`.
    let mut break_times: Vec<T> = vec![tf(if t_first.is_finite() { t_first } else { 0.0 })];
    for (k, dose) in doses.iter().enumerate() {
        break_times.push(tf(dose.time));
        if is_real_infusion(dose) {
            // `F`-reshaped window end, matching `active_infusions`' own membership test
            // (`predictions.rs:1520-1531`) and `collect_dose_break_times`. An unscaled
            // `dose.duration` puts the break past the true end, so the admission test
            // `end >= t_end - EPS` fails and the infusion is never admitted in *any*
            // segment — measured on #1263 as `ipred = 0.0` everywhere for `F = 0.5`.
            break_times.push(tf(dose.time) + inf[k].1);
            // Earlier SS pulses still running after the record (`T_inf > II`); no lagtime here.
            let (first, n) = crate::dosing::ss_residual_infusions_g(dose, tf(0.0), inf[k].1);
            // Only the stops the walk reaches (it ends at `t_last`).
            for j in crate::dosing::residual_end_indices_g(dose.time, first, dose.ii, n, t_last) {
                break_times.push(tf(dose.time) + first - tf(j as f64 * dose.ii));
            }
        }
    }
    break_times.push(tf(t_last));
    crate::sens::ode_provider::sort_times(&mut break_times);
    break_times.dedup_by(|a, b| a.br_absdiff_lt(*b, 1e-15));
    // A non-finite break time makes the subject non-finite (#1189) — see
    // `ode::predictions::times_have_non_finite`. `results` is prefilled with the
    // caller's default point, so overwrite it with NaN ipreds rather than returning a
    // finite-looking filter pass built on a timeline that could not be ordered.
    if crate::ode::predictions::abandon_non_finite_timeline(
        break_times.iter().map(|t| t.seen_value()),
        None,
    ) {
        // `fill`, not a per-field loop: one spelling of the struct, so a field added to
        // `EkfObsPoint` cannot be left at its default here while the others go NaN.
        results.fill(EkfObsPoint::<T>::of(f64::NAN));
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
        // The breaks as numbers of the filter's type (a window end moves), and their values
        // for the record reads and the infusions' membership, which the guarded order fixes.
        let (t_start_g, t_end_g) = (break_times[k], break_times[k + 1]);
        let (t_start, t_end) = (t_start_g.seen_value(), t_end_g.seen_value());

        // Records read *at* t_start — the whole `EVENT_MATCH_TOL` band, through the same
        // helper as the ODE engines (#1226). A dose matches a break symmetrically within
        // `EVENT_MATCH_TOL`, so a record the reader moved one ULP below a co-timed dose
        // (#614/#1810) can share the band with it. Such a record is read **before** the band's
        // doses are applied, as `band_read_state` does on the ODE walks (#2124): every record
        // earlier than the band's first dose takes the pre-event pair, every other one the
        // post-event pair (#2510). At the subject's first instant the two always share a break,
        // and before this the record read post-dose there (and, before an `SS=1` dose, the
        // stationary covariance).
        obs_index.records_at_break(t_start, &mut boundary_obs);
        let first_event = doses
            .iter()
            .enumerate()
            .filter(|&(di, d)| {
                !applied[di] && (d.time - t_start).abs() < crate::ode::predictions::EVENT_MATCH_TOL
            })
            .map(|(_, d)| d.time)
            .fold(f64::INFINITY, f64::min);
        // Set when an `SS=1` record at this break replaces the covariance (below).
        let mut cov_replaced = false;
        for pre_event in [true, false] {
            if !pre_event {
                // Apply bolus doses at t_start (infusions enter via the wrapped RHS).
                for (di, dose) in doses.iter().enumerate() {
                    if applied[di] {
                        continue;
                    }
                    if (dose.time - t_start).abs() < crate::ode::predictions::EVENT_MATCH_TOL {
                        applied[di] = true;
                        // An `SS=1` record replaces the pair with the periodic trough (#1260); the
                        // record's own dose then lands on it below (bolus) or through
                        // `active_infusions` (infusion), exactly as on the ODE path.
                        if dose.ss && dose.ii > 0.0 {
                            let (u_ss, p_ss) = equilibrate_ss_ekf(
                                rhs,
                                n,
                                dose,
                                dose_f_bio[di],
                                inf[di],
                                params,
                                diffusion_var,
                                &opts,
                            );
                            u.copy_from_slice(&u_ss);
                            p_mat = p_ss;
                            cov_replaced = true;
                        }
                        if !is_real_infusion(dose) {
                            let cmt_idx = dose.cmt_idx();
                            if cmt_idx < n {
                                u[cmt_idx] = u[cmt_idx] + dose_f_bio[di] * T::from_f64(dose.amt);
                            }
                        }
                    }
                }
            }
            // An `SS=1` record replaces the covariance, discarding every update before it,
            // so a post-event record that a segment ending here already assimilated against the
            // carried covariance — an observation on the break, or the later half of a pair the
            // reader split across the dose (#614) — is assimilated afresh against the steady
            // state's, and the records after it condition on that update (#2510). A bolus leaves
            // `p_mat` untouched, so without an SS record the single update stands.
            if cov_replaced {
                for &j in &boundary_obs {
                    if obs_times[j] >= first_event {
                        assimilated[j] = false;
                    }
                }
            }
            for i in 0..boundary_obs.len() {
                let obs_idx = boundary_obs[i];
                if (obs_times[obs_idx] < first_event) != pre_event {
                    continue;
                }
                let v = u[obs_cmt_idx];
                let ipred = ekf_ipred(v);
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
                let r = r_of(obs_idx, ipred);
                let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
                p_mat = p_new;
                let point = EkfObsPoint { ipred, p_obs };
                // Records sharing this exact time are one measurement instant: they take the
                // same `ipred`/`p_obs` from this single update rather than each running their
                // own. The in-segment read below collapses them the same way. They are
                // *filled* rather than dropped: the band excludes them from the segment
                // `saveat` too, so a dropped index would keep its
                // `EkfObsPoint { ipred: 0.0, p_obs: 0.0 }` prefill and feed a finite,
                // plausible zero into the likelihood.
                let bits = obs_times[obs_idx].to_bits();
                for &j in &boundary_obs {
                    if !assimilated[j] && obs_times[j].to_bits() == bits {
                        results[j] = point;
                        assimilated[j] = true;
                    }
                }
            }
        }

        let saveat = crate::ode::predictions::segment_saveat(obs_times, t_start, t_end);
        // Which records a moving end reads is the verdict of each record's comparison with it.
        for &t in obs_times {
            if !t_start_g.is_constant() {
                let _ = (tf(t) - t_start_g)
                    .br_cmp_c(BranchCmp::Ge, crate::ode::predictions::EVENT_MATCH_TOL);
            }
            if !t_end_g.is_constant() {
                let _ = tf(t).br_cmp(BranchCmp::Le, t_end_g);
            }
        }

        if t_end_g.br_absdiff_lt(t_start_g, 1e-15) {
            continue;
        }

        // Re-anchor TAD for this segment, *before* anything integrates with it. `&[]`, not a
        // zero-filled vector: this path applies no lagtime, and `tad_anchor_for` reads a
        // missing entry as zero lag.
        let anchors = (
            first_dose,
            tf(crate::ode::predictions::tad_anchor_for(doses, &[], t_start)),
        );

        // Active infusion rates for this segment (shared with the FOCEI ODE
        // path so the F·RATE / span / lag / reset semantics stay in lockstep).
        // The EKF path has no per-dose lagtimes and no system resets. Membership is read on
        // the values (the guarded order fixes it); the rate and window carry `F` and a
        // modeled `D`/`R` ([`infusion_of`]).
        let f_vals: Vec<f64> = dose_f_bio.iter().map(|f| f.seen_value()).collect();
        let active: Vec<(usize, T)> = active_infusion_doses(
            &[],
            doses,
            t_start,
            t_end,
            &[],
            &f_vals,
            f64::NEG_INFINITY,
            n,
        )
        .into_iter()
        .map(|(k, cmt, _, pulses)| {
            (
                cmt,
                if pulses == 1 {
                    inf[k].0
                } else {
                    inf[k].0 * T::from_f64(pulses as f64)
                },
            )
        })
        .collect();

        // Integrate mean state
        let sol = integrate_mean(
            rhs,
            &active,
            &u,
            (t_start_g, t_end_g),
            params,
            anchors,
            &saveat,
            &opts,
        );

        // Propagate covariance and update at obs times within this segment
        let mut t_prev = t_start_g;
        let mut u_prev = u.clone();

        for pt in &sol {
            let t_pt = point_time(pt.t, t_end_g);
            p_mat = propagate_covariance_between(
                rhs,
                &active,
                p_mat,
                t_prev,
                &u_prev,
                t_pt,
                &pt.u,
                params,
                anchors,
                diffusion_var,
                n,
            );

            // Same assimilate-once mask as the boundary read: an observation sitting
            // exactly on this segment's `t_end` is read here *and* at the next break as
            // its `t_start`, and `kalman_update` is not idempotent.
            //
            // Records sharing this exact time are **one** measurement instant and share
            // a single update, exactly as the boundary path above does — `solve_ekf`
            // has one `obs_cmt_idx`, so N records here are N reads of the same
            // compartment at the same time, and updating N times would shrink `p_mat`
            // as though N independent measurements had arrived. Before #1263 only the
            // last index of a shared time was in the map at all, so the others kept
            // their `{ ipred: 0.0, p_obs: 0.0 }` prefill.
            //
            // So are the records one ULP above it: the reader moves the first of two
            // co-timed records one ULP below their tie (#614, #1810), and the pair is still
            // one instant (`crate::types::same_instant`). Both times are in `saveat`, so the
            // earlier point takes the pair's single update and the later finds every index
            // already assimilated. No break sits between the two times, so the solver saves
            // both off one step and they share a state as well.
            let here = obs_map
                .get(&pt.t.to_bits())
                .into_iter()
                .chain(obs_map.get(&pt.t.next_up().to_bits()))
                .flatten();
            if let Some(&first) = here.clone().find(|&&j| !assimilated[j]) {
                let ipred = ekf_ipred(pt.u[obs_cmt_idx]);
                let r = r_of(first, ipred);
                let (p_new, p_obs) = kalman_update(&p_mat, obs_cmt_idx, r, n);
                p_mat = p_new;
                let point = EkfObsPoint { ipred, p_obs };
                for &j in here {
                    if !assimilated[j] {
                        results[j] = point;
                        assimilated[j] = true;
                    }
                }
            }

            t_prev = t_pt;
            u_prev.clone_from(&pt.u);
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
    use crate::sens::num::PkNum;
    use approx::assert_relative_eq;

    /// A hand-written `f64` right-hand side as an [`EkfRhs`]: the tests' closures are not
    /// generic over a number type, so their tangent is a central finite difference of the
    /// closure (a test oracle; the production path is the program's exact tangent, pinned by
    /// `exact_jacobian_matches_central_fd_on_the_compiled_program`). The anchors land in the
    /// extended-parameter slots the closures read.
    struct FdRhs<'a>(&'a (dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync));

    impl EkfRhs for FdRhs<'_> {
        fn eval<U: VmNum>(&self, u: &[U], p: &[U], t: U, anchors: Anchors<U>, du: &mut [U]) {
            let mut ext = vec![f64::NAN; crate::types::MAX_PK_PARAMS + 2];
            for (e, x) in ext.iter_mut().zip(p).take(crate::types::MAX_PK_PARAMS) {
                *e = x.val();
            }
            ext[crate::types::MAX_PK_PARAMS] = anchors.0;
            ext[crate::types::MAX_PK_PARAMS + 1] = anchors.1.val();
            let (n, t0) = (u.len(), t.val());
            let u0: Vec<f64> = u.iter().map(|x| x.val()).collect();
            let mut f0 = vec![0.0; n];
            (self.0)(&u0, &ext, t0, &mut f0);
            let mut jac = vec![0.0; n * n];
            let (mut fp, mut fm) = (vec![0.0; n], vec![0.0; n]);
            for j in 0..n {
                let h = 1e-5 * (1.0 + u0[j].abs());
                let (mut up, mut um) = (u0.clone(), u0.clone());
                up[j] += h;
                um[j] -= h;
                (self.0)(&up, &ext, t0, &mut fp);
                (self.0)(&um, &ext, t0, &mut fm);
                for i in 0..n {
                    jac[i * n + j] = (fp[i] - fm[i]) / (2.0 * h);
                }
            }
            for i in 0..n {
                let mut acc = U::from_f64(f0[i]);
                for j in 0..n {
                    acc = acc + U::from_f64(jac[i * n + j]) * (u[j] - U::from_f64(u0[j]));
                }
                du[i] = acc;
            }
        }
    }

    /// The pre-generic entry point the tests were written against: an `f64` closure, the
    /// dose-attribute map for `F`, and a precomputed per-observation `R`.
    #[allow(clippy::too_many_arguments)]
    fn solve_ekf(
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
        let dose_f: Vec<f64> = doses
            .iter()
            .map(|d| dose_attr_map.f_bio(d.cmt_raw(), pk_params_flat))
            .collect();
        solve_ekf_g::<f64, _>(
            &FdRhs(rhs),
            n_states,
            obs_cmt_idx,
            diffusion_var,
            pk_params_flat,
            &dose_f,
            &[],
            initial_state,
            doses,
            obs_times,
            &|j, _| r_obs_vec.get(j).copied().unwrap_or(1.0),
            opts,
        )
    }

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A.
    fn one_cpt_rhs(y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]) {
        let cl = p[crate::types::PK_IDX_CL];
        let v = p[crate::types::PK_IDX_V];
        let ke = if v > 0.0 { cl / v } else { 0.0 };
        dy[0] = -ke * y[0];
    }

    /// The same 1-cpt bolus with elimination modulated by a **model-time builtin**, read
    /// out of the extended-parameter slot the compiled `[odes]` RHS uses. `slot` is
    /// `MAX_PK_PARAMS` for TAFD or `MAX_PK_PARAMS + 1` for TAD; `coef` scales the term.
    ///
    /// This mirrors `parser/model_parser.rs`'s injection exactly, including the
    /// `.get(..).filter(is_finite).map_or(NAN, ..)` shape — so a short `params` slice
    /// yields `NaN` here for the same reason it does in production. Writing it any other
    /// way (e.g. indexing, or defaulting to 0.0) would make the fixture unable to
    /// reproduce the defect at all.
    fn time_reading_rhs(
        slot: usize,
        coef: f64,
    ) -> impl Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync {
        move |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            let builtin = p
                .get(slot)
                .copied()
                .filter(|x| x.is_finite())
                .map_or(f64::NAN, |anchor| t - anchor);
            dy[0] = -ke * y[0] * (1.0 + coef * builtin);
        }
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
            obs_rgroup: Vec::new(),
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
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cmt_toggles: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
            endpoint_covariates: Vec::new(),
            filled_cells: Vec::new(),
            simulated_mixnum: None,
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

    /// #1131: an `[odes]` RHS reading a model-time builtin poisons the EKF, because
    /// `solve_ekf` was handed a bare `PkParams::values` — exactly `MAX_PK_PARAMS` long —
    /// while the RHS reads TAFD/TAD from slots `MAX_PK_PARAMS` and `+1`. `.get()` returned
    /// `None`, so both builtins evaluated to `NaN` for the whole trajectory.
    ///
    /// **Assert `p_obs`, never `ipred`.** The two failed differently and only one of them
    /// can carry this test: `ipred` is clamped by `if v.is_nan() { 0.0 }` in the assimilate
    /// branch, and by the time this was measured at `579d1e96` the sdtab column was already
    /// reporting the correct value-path numbers — so an `ipred` assertion passes both
    /// before and after the fix and cannot fail. `p_obs` has no such clamp, and it is what
    /// feeds the likelihood: the defect surfaced as the `1e20` objective sentinel.
    ///
    /// No SS dose and no lagtime here — the defect needs neither, which is why #1131 is
    /// not the steady-state bug it was filed alongside.
    #[test]
    fn time_reading_rhs_gets_finite_anchors_on_the_ekf_walk() {
        for (name, slot) in [
            ("TAFD", crate::types::MAX_PK_PARAMS),
            ("TAD", crate::types::MAX_PK_PARAMS + 1),
        ] {
            // Two doses: under a single dose TAD == TAFD, so a one-dose fixture cannot
            // tell the two slots apart and would agree with an engine that plumbed only
            // one of them.
            let doses = vec![
                bolus_dose(100.0),
                DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
            ];
            let obs_times = vec![2.0, 4.0, 8.0, 24.0];
            let pk = make_pk(1.0, 20.0);
            let r_obs_vec: Vec<f64> = vec![0.01; obs_times.len()];

            let pts = solve_ekf(
                &time_reading_rhs(slot, 0.05),
                1,
                0,
                &[0.01],
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &r_obs_vec,
                OdeSolverOptions::default(),
            );

            assert_eq!(pts.len(), obs_times.len(), "{name}: wrong point count");
            for (j, pt) in pts.iter().enumerate() {
                // `is_finite` before anything else: a `NaN` compared with `<` or folded
                // through `f64::max` silently reads as "in range".
                assert!(
                    pt.p_obs.is_finite(),
                    "{name}: p_obs[{j}] = {} — the anchor slot is still absent",
                    pt.p_obs
                );
                assert!(
                    pt.p_obs > 0.0,
                    "{name}: p_obs[{j}] = {} — a non-positive variance would also be \
                     produced by an EKF that never propagated, so assert the positive",
                    pt.p_obs
                );
                assert!(
                    pt.ipred.is_finite() && pt.ipred > 0.0,
                    "{name}: ipred[{j}] = {} — drug is present at every one of these times",
                    pt.ipred
                );
            }
        }
    }

    /// The discriminator for the test above: with the modulation coefficient at zero the
    /// builtin contributes nothing, so the walk must be **bit-identical** to the plain
    /// `one_cpt_rhs`. Before the fix this failed for the opposite reason to the usual one
    /// — `0.0 * NaN` is `NaN`, so merely *mentioning* the builtin destroyed the run.
    #[test]
    fn a_zero_coefficient_model_time_term_is_bit_identical_to_omitting_it() {
        let doses = vec![
            bolus_dose(100.0),
            DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![2.0, 4.0, 8.0, 24.0];
        let pk = make_pk(1.0, 20.0);
        let r_obs_vec: Vec<f64> = vec![0.01; obs_times.len()];
        let run = |rhs: &(dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync)| {
            solve_ekf(
                rhs,
                1,
                0,
                &[0.01],
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &r_obs_vec,
                OdeSolverOptions::default(),
            )
        };
        let plain = run(&one_cpt_rhs);
        for slot in [crate::types::MAX_PK_PARAMS, crate::types::MAX_PK_PARAMS + 1] {
            let zero_coef = run(&time_reading_rhs(slot, 0.0));
            for (j, (a, b)) in zero_coef.iter().zip(plain.iter()).enumerate() {
                assert_eq!(
                    a.ipred.to_bits(),
                    b.ipred.to_bits(),
                    "slot {slot}: ipred[{j}] {} vs {}",
                    a.ipred,
                    b.ipred
                );
                assert_eq!(
                    a.p_obs.to_bits(),
                    b.p_obs.to_bits(),
                    "slot {slot}: p_obs[{j}] {} vs {}",
                    a.p_obs,
                    b.p_obs
                );
            }
        }
    }

    /// The **value** oracle for #1131, and the strongest one available: with the diffusion
    /// variance at zero the EKF's mean walk reduces to the ordinary ODE integration, so
    /// `solve_ekf`'s `ipred` must equal `ode_predictions`' — on a TAD- and a TAFD-reading
    /// RHS, the exact models the bare-slice defect destroyed.
    ///
    /// Both engines call `predictions::tad_anchor_for` after this fix, so this test cannot
    /// observe a wrong *anchor rule* shared by both — that is what the NONMEM anchors in
    /// `nonmem_anchor/` are for. What it does observe is the defect that was actually there:
    /// the EKF passing a slice too short for the RHS to read the slot at all. Deleting
    /// either the `seed_ext_params` call or the per-segment anchor write in `solve_ekf`
    /// turns the EKF column into the `0.0` clamp while the ODE column stays correct.
    ///
    /// A third reference, independent of both engines, keeps them from agreeing on a wrong
    /// answer. For `dA/dt = -ke·A·(1 + c·u)` the exponent integrates in closed form:
    /// with `u = TAD` the clock restarts at each arrival, with `u = TAFD` it does not, so
    /// the two slots have *different* closed forms after the second dose — a fixture that
    /// plumbed one slot into both would fail here.
    #[test]
    fn ekf_matches_the_ode_twin_and_a_closed_form_on_a_model_time_rhs() {
        use crate::ode::predictions::{ode_predictions, OdeSpec};
        use crate::types::Subject;
        use std::collections::HashMap;

        let (cl, v) = (1.0f64, 20.0f64);
        let ke = cl / v;
        let c = 0.05f64;
        // Two doses: after a single dose TAD ≡ TAFD, so a one-dose fixture agrees with an
        // engine that plumbed one slot into both.
        let (t2, amt) = (12.0f64, 100.0f64);
        let doses = vec![
            DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0),
            DoseEvent::new(t2, amt, 1, 0.0, false, 0.0),
        ];
        // Samples straddle the second dose, so the post-dose leg — where the two clocks
        // disagree — is actually scored.
        let obs_times = vec![2.0, 8.0, 14.0, 24.0];
        let pk = make_pk(cl, v);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };

        // ∫ (1 + c·u) over the leg, with `u` measured from `anchor`.
        let phase = |t_from: f64, t_to: f64, anchor: f64| -> f64 {
            let (a, b) = (t_from - anchor, t_to - anchor);
            (b - a) + c * (b * b - a * a) / 2.0
        };
        // `TAD` restarts at `t2`; `TAFD` keeps running from the first dose.
        let closed_form = |t: f64, restarts: bool| -> f64 {
            if t < t2 {
                amt * (-ke * phase(0.0, t, 0.0)).exp()
            } else {
                let at_t2 = amt * (-ke * phase(0.0, t2, 0.0)).exp();
                let anchor = if restarts { t2 } else { 0.0 };
                (at_t2 + amt) * (-ke * phase(t2, t, anchor)).exp()
            }
        };

        for (name, slot, restarts) in [
            ("TAFD", crate::types::MAX_PK_PARAMS, false),
            ("TAD", crate::types::MAX_PK_PARAMS + 1, true),
        ] {
            let rhs = time_reading_rhs(slot, c);
            let ekf_pts = solve_ekf(
                &rhs,
                1,
                0,
                &[0.0], // zero diffusion: the mean walk reduces to plain integration
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                opts,
            );

            let subj = Subject {
                obs_rgroup: Vec::new(),
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
                pk_only_raw_times: Vec::new(),
                reset_period_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                reset_covariates: Vec::new(),
                cmt_toggles: Vec::new(),
                cens: vec![0; obs_times.len()],
                occasions: Vec::new(),
                obs_l2: Vec::new(),
                dose_occasions: Vec::new(),
                reset_occasions: Vec::new(),
                pk_only_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: vec![],
                endpoint_covariates: Vec::new(),
                filled_cells: Vec::new(),
                simulated_mixnum: None,
            };
            let ode_spec = OdeSpec {
                chz_state_slots: Vec::new(),
                rhs: Box::new(time_reading_rhs(slot, c)),
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

            for (j, &t) in obs_times.iter().enumerate() {
                let want = closed_form(t, restarts);
                let (ekf, ode) = (ekf_pts[j].ipred, ode_preds[j]);
                // `is_finite` first: a `NaN` compared with `<` reads as "in range", and the
                // `0.0` clamp in the assimilate branch would otherwise be judged only by
                // the relative check below.
                assert!(
                    ekf.is_finite() && ode.is_finite(),
                    "{name}: non-finite at t={t} — ekf {ekf}, ode {ode}"
                );
                assert!(
                    ekf > 0.0,
                    "{name}: ekf ipred at t={t} is {ekf} — the `is_nan()` clamp reports a \
                     poisoned walk as 0.0, which is exactly the #1131 signature"
                );
                assert_relative_eq!(ekf, ode, epsilon = 1e-6, max_relative = 1e-6);
                assert_relative_eq!(ekf, want, epsilon = 1e-6, max_relative = 1e-6);
            }
        }

        // The straddle: assert the two clocks actually disagree on this fixture, so the
        // loop above cannot quietly become the same assertion twice.
        let (tad_24, tafd_24) = (closed_form(24.0, true), closed_form(24.0, false));
        assert!(
            (tad_24 - tafd_24).abs() / tad_24 > 0.1,
            "TAD and TAFD must differ materially after the second dose \
             ({tad_24:.6} vs {tafd_24:.6}) or this fixture cannot tell the slots apart"
        );
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
            obs_rgroup: Vec::new(),
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
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cmt_toggles: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
            endpoint_covariates: Vec::new(),
            filled_cells: Vec::new(),
            simulated_mixnum: None,
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
            // Measured (Crank-Nicolson Riccati step, Linux): realised error 4.95e-5 / 4.24e-5 /
            // 3.40e-5 / 2.69e-5 at t = 1/4/8/12 (the old forward-Euler step sat at 2.46% ..
            // 1.32%, a discretisation floor). Bounded at 1e-3, 20x headroom.
            assert_relative_eq!(ekf_pts[i].p_obs, p_analytic, max_relative = 1e-3);
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
    /// `times_have_non_finite` guard → finite `ipred`s and the assert below fires.
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
        // Inside a `SolverStatsScope` so the #1234 counter is observable: this walk's
        // recorder is its own site, and no other test in the tree can see it (the fit-level
        // sweep never drives the EKF). Without the scope the call is unchanged.
        let scope = crate::ode::solver::SolverStatsScope::enter();
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
        let stats = scope.collected();
        assert!(
            got.iter().all(|p| !p.ipred.is_finite()),
            "a non-finite timeline must give non-finite EKF ipreds, not a finite pass \
             with the bad dose silently dropped"
        );
        // Mutation: delete `ekf.rs`'s `abandon_non_finite_timeline` record (or route it
        // through the bare predicate again) → this fires naming the EKF walk, and no other
        // test in the tree moves.
        assert_eq!(
            stats.abandoned_non_finite_timeline, 1,
            "the EKF walk must record its abandoned walk too — it has its own guard and its \
             own return, and nothing else drives it: {stats:?}"
        );
        assert_eq!(
            stats.attempted_steps, 0,
            "EKF walk: nothing was integrated, so the step counters must stay zero — that is \
             why the abandoned counter has to exist: {stats:?}"
        );
    }

    /// **NONMEM anchor — a RATE-defined infusion under `F ≠ 1` on the EKF walk.**
    ///
    /// `F` reshapes an infusion's **duration**, not its rate (#419): NONMEM holds the data
    /// `RATE` and delivers `F·AMT` over `F·AMT/RATE`. `ekf.rs` pushed an unscaled
    /// `dose.time + dose.duration` as the window's break, so `active_infusions`'
    /// membership test (`predictions.rs:1520-1531`, `end >= t_end - EPS`) rejected the
    /// window in **every** segment and the infusion delivered zero mass — measured
    /// `ipred = 0.0` at every observation for `F = 0.5`, against `27.86 / 42.08 / 31.18`
    /// from `ode_predictions` (#1263 review).
    ///
    /// The reference is **NONMEM 7.6.0 PRED**, not ferx: `nonmem_anchor/oral_central_inf_advan2_f06.ctl`
    /// (ADVAN2 TRANS2, `CL=5, V=50, KA=1, F2=0.6, S2=V`, 100 mg at `RATE=25` into `CMT=2`,
    /// obs at `t = 1, 3, 6, 10, 16`), whose digits are also pinned in
    /// `tests/oral_central_infusion_nonmem_anchor.rs` for the analytical paths. The system
    /// is written here as the explicit two-state ODE twin so it reaches `solve_ekf`, which
    /// the analytical anchor cannot.
    ///
    /// Non-degenerate on both sides of the quantity under test: `t = 1` is **inside** the
    /// window (where `F` has not yet moved the answer — it is the same value as the `F = 1`
    /// run) and `t = 3, 6, 10, 16` are all **past** the `F`-shortened end at `2.4 h`, where
    /// an unscaled duration and a scaled one disagree. An implementation that ignored `F`
    /// on the window would match the first point and miss the other four.
    #[test]
    fn ekf_infusion_under_f_matches_nonmem() {
        // `oral_central_inf_advan2_f06.tab`, NONMEM 7.6.0 PRED at FORMAT=,1PE17.10.
        const ADVAN2_F06: [f64; 5] = [
            4.7581290982e-01,
            1.0047315645e+00,
            7.4432344989e-01,
            4.9893492919e-01,
            2.7382129479e-01,
        ];
        let (cl, v, ka, f) = (5.0f64, 50.0f64, 1.0f64, 0.6f64);
        // Two states so this is genuinely the ADVAN2 system; the depot is never dosed
        // (the infusion bypasses it into CMT=2), so `A1 ≡ 0` exactly as under NONMEM.
        let rhs = move |y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let k = p[crate::types::PK_IDX_CL] / p[crate::types::PK_IDX_V];
            dy[0] = -ka * y[0];
            dy[1] = ka * y[0] - k * y[1];
        };
        let mut pk = make_pk(cl, v);
        pk[crate::types::PK_IDX_F] = f;
        let obs_times = vec![1.0, 3.0, 6.0, 10.0, 16.0];
        // AMT=100, RATE=25 -> duration 4 h; F=0.6 shortens the window to 2.4 h.
        let doses = vec![DoseEvent::new(0.0, 100.0, 2, 25.0, false, 0.0)];
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };

        let pts = solve_ekf(
            &rhs,
            2,
            1, // observable is the central compartment
            &[0.0, 0.0],
            &pk,
            &Default::default(),
            &[],
            &doses,
            &obs_times,
            &vec![0.01; obs_times.len()],
            opts,
        );

        // Measured realised relative errors against the committed NONMEM digits:
        // 4.84e-13 / 1.10e-11 / 1.79e-13 / 2.36e-11 / 5.24e-11 at t = 1/3/6/10/16, worst
        // 5.24e-11. That worst case is at the reference's own resolution — the `$TABLE`
        // prints 11 significant figures, so ~1e-11 is the floor a comparison against it
        // can reach and a tighter bound would be pinning the last printed digit. 1e-9,
        // i.e. 19× headroom over the realised worst case.
        let mut worst = 0.0f64;
        for (j, &t) in obs_times.iter().enumerate() {
            let got = pts[j].ipred / v;
            let want = ADVAN2_F06[j];
            // `is_finite` before folding: `f64::max` discards `NaN`, so a solve that
            // diverged would otherwise leave `worst` at whatever the good rows produced.
            assert!(got.is_finite(), "non-finite EKF ipred at t={t}: {got}");
            let rel = ((got - want) / want).abs();
            worst = worst.max(rel);
            assert!(
                rel < 1e-9,
                "t={t}: EKF {got:.10e} vs NONMEM {want:.10e} (rel {rel:.3e})"
            );
        }
        assert!(
            worst > 0.0,
            "a zero worst-case error means the loop never compared anything"
        );
    }

    /// **The TAFD anchor is the first dose time, not a literal zero.**
    ///
    /// Every other fixture in this module doses at `t = 0`, where
    /// `earliest_dose_time(doses)` and `0.0` are the same number — so replacing the seed
    /// with `seed_ext_params(pk_params_flat, 0.0)` left `cargo test --lib` at
    /// **4083 passed, 0 failed** (#1263 review). This fixture's first dose is at `t = 6`,
    /// and its closed form takes the anchor from `t1` rather than hard-coding `0.0`, so
    /// that mutation moves the TAFD arm and the test fails.
    ///
    /// The `TAD` arm is carried alongside deliberately: the two anchors coincide on the
    /// first dosing interval and separate only after the second dose, so a fixture that
    /// scored the first interval alone could not tell them apart either.
    #[test]
    fn ekf_tafd_anchors_at_the_first_dose_not_at_zero() {
        let (cl, v) = (1.0f64, 20.0f64);
        let ke = cl / v;
        let c = 0.05f64;
        // First dose at t = 6, NOT at 0 — the whole point of this fixture.
        let (t1, t2, amt) = (6.0f64, 18.0f64, 100.0f64);
        let doses = vec![
            DoseEvent::new(t1, amt, 1, 0.0, false, 0.0),
            DoseEvent::new(t2, amt, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![8.0, 14.0, 20.0, 30.0];
        let pk = make_pk(cl, v);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };

        // ∫ (1 + c·u) du over the leg, with `u` measured from `anchor`.
        let phase = |t_from: f64, t_to: f64, anchor: f64| -> f64 {
            let (a, b) = (t_from - anchor, t_to - anchor);
            (b - a) + c * (b * b - a * a) / 2.0
        };
        // `TAFD` anchors at `t1` for the whole record; `TAD` re-anchors at `t2`.
        let closed_form = |t: f64, restarts: bool| -> f64 {
            if t < t2 {
                amt * (-ke * phase(t1, t, t1)).exp()
            } else {
                let at_t2 = amt * (-ke * phase(t1, t2, t1)).exp();
                let anchor = if restarts { t2 } else { t1 };
                (at_t2 + amt) * (-ke * phase(t2, t, anchor)).exp()
            }
        };

        let mut worst = 0.0f64;
        for (name, slot, restarts) in [
            ("TAFD", crate::types::MAX_PK_PARAMS, false),
            ("TAD", crate::types::MAX_PK_PARAMS + 1, true),
        ] {
            let rhs = time_reading_rhs(slot, c);
            let pts = solve_ekf(
                &rhs,
                1,
                0,
                &[0.0], // zero diffusion: the mean walk reduces to plain integration
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                opts,
            );
            for (j, &t) in obs_times.iter().enumerate() {
                let got = pts[j].ipred;
                let want = closed_form(t, restarts);
                assert!(got.is_finite(), "{name}: non-finite ipred at t={t}");
                let rel = ((got - want) / want).abs();
                worst = worst.max(rel);
                // Measured worst realised error across both arms: 5.24e-11 (TAD, t=30);
                // the TAFD arm's worst is 4.31e-11. Bounded at 1e-9 — 19× headroom over
                // the realised worst case, and the same order the `reltol = 1e-10`
                // integration above can support.
                assert!(
                    rel < 1e-9,
                    "{name} at t={t}: EKF {got} vs closed form {want} (rel {rel:.3e})"
                );
            }
        }
        assert!(worst > 0.0, "the comparison loop never ran");

        // The straddle: with the anchor mutated to a literal 0.0 the TAFD arm integrates
        // `1 + c·t` instead of `1 + c·(t − 6)`, so the two spellings must actually differ
        // here. Assert that they do, or this fixture silently becomes a tautology.
        let at_last = obs_times[obs_times.len() - 1];
        let with_true_anchor = closed_form(at_last, false);
        let with_zero_anchor = {
            let phase0 = |t_from: f64, t_to: f64| -> f64 {
                let (a, b) = (t_from, t_to);
                (b - a) + c * (b * b - a * a) / 2.0
            };
            let at_t2 = amt * (-ke * phase0(t1, t2)).exp();
            (at_t2 + amt) * (-ke * phase0(t2, at_last)).exp()
        };
        let sep = ((with_true_anchor - with_zero_anchor) / with_true_anchor).abs();
        assert!(
            sep > 0.05,
            "the t1=6 and t1=0 TAFD anchors must give materially different curves for \
             this fixture to pin the anchor at all — separation {sep:.3e}"
        );
    }

    /// **A subject whose records start after `t = 0` gets no phantom leading segment.**
    ///
    /// `break_times` was seeded with a literal `0.0` where the ODE path uses
    /// `subject_integration_start`. For a first event at `t = 24` that left a `[0, 24]`
    /// segment which `propagate_covariance` actually integrated, so `p_mat` arrived at the
    /// first observation carrying `q · 24` of process noise the ODE twin never accumulates
    /// — measured `p_obs = 6.5` against a true `0.5`, i.e. **13×** (#1263 review).
    ///
    /// The oracle is a closed form, not a twin: with `CL = 0` the drift vanishes, the
    /// Jacobian is zero and the Riccati equation is exactly `dP/dt = Q`, so
    /// `P(t) = q · (t − t_start)` with no solver error to absorb a wrong start.
    #[test]
    fn ekf_covariance_starts_at_the_first_record_not_at_zero() {
        let q = 0.25f64;
        let pk = make_pk(0.0, 20.0); // CL = 0 -> ke = 0 -> dP/dt = Q exactly
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };

        // One variable: the same subject, shifted. Both must give the same `p_obs` at the
        // same elapsed time — a start-dependent answer is the defect.
        let mut seen = Vec::new();
        for shift in [0.0f64, 24.0] {
            let doses = vec![DoseEvent::new(shift, 100.0, 1, 0.0, false, 0.0)];
            let obs_times = vec![shift + 2.0, shift + 6.0];
            let pts = solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[q],
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                opts,
            );
            let got = pts[0].p_obs;
            assert!(got.is_finite(), "shift={shift}: non-finite p_obs {got}");
            // Measured error against the closed form: 0.0 exactly at both shifts.
            let want = q * 2.0;
            assert!(
                (got - want).abs() < 1e-12,
                "shift={shift}: p_obs {got} vs closed form q·Δt {want} — a leading \
                 [0, {shift}] segment would give q·(Δt+{shift}) = {}",
                q * (2.0 + shift)
            );
            seen.push(got);
        }
        // The straddle: the two shifts must be a real experiment. `q·2 = 0.5` and
        // `q·26 = 6.5` are far apart, so a start-dependent implementation cannot pass
        // both branches above by accident.
        assert!(
            (q * 26.0 - q * 2.0).abs() > 1.0,
            "the shifted and unshifted expectations must differ for this to be a test"
        );
        assert_eq!(
            seen[0].to_bits(),
            seen[1].to_bits(),
            "the same subject shifted in time must give bit-identical covariance"
        );
    }

    /// Michaelis–Menten elimination, the nonlinear right-hand side #1260 was measured on:
    /// `VMAX = 20`, `KM = 50`, so the Riccati Jacobian `∂f/∂A = −VMAX·KM/(KM+A)²` moves with
    /// the mean's phase and a mean equilibrated wrong reaches `P`.
    fn mm_rhs(y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]) {
        dy[0] = -20.0 * y[0] / (50.0 + y[0]);
    }

    /// The `SS=1` fixture of #1260: one record at `t = 480`, `II = 12`, observed at 2, 5, 8
    /// and 11 h after it — and the explicit train it stands for, 41 ordinary doses at
    /// `0, 12, …, 480`. `ss` builds the record arm; `train` the oracle; `single` the same
    /// record with `SS=0`, which is what the filter used to compute.
    fn ss_fixture(
        amt: f64,
        rate: f64,
    ) -> (Vec<DoseEvent>, Vec<DoseEvent>, Vec<DoseEvent>, Vec<f64>) {
        let ss = vec![DoseEvent::new(480.0, amt, 1, rate, true, 12.0)];
        let single = vec![DoseEvent::new(480.0, amt, 1, rate, false, 0.0)];
        let train: Vec<DoseEvent> = (0..=40)
            .map(|k| DoseEvent::new(12.0 * k as f64, amt, 1, rate, false, 0.0))
            .collect();
        (ss, train, single, vec![482.0, 485.0, 488.0, 491.0])
    }

    /// Worst relative error over `ipred` and `p_obs`, with every value asserted finite first
    /// (a `NaN` would fold out of `max`).
    fn worst_rel(a: &[EkfObsPoint], b: &[EkfObsPoint]) -> (f64, f64) {
        let mut w = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            for v in [x.ipred, x.p_obs, y.ipred, y.p_obs] {
                assert!(v.is_finite(), "non-finite EKF point: {v}");
            }
            w.0 = w.0.max((x.ipred - y.ipred).abs() / y.ipred.abs());
            w.1 = w.1.max((x.p_obs - y.p_obs).abs() / y.p_obs.abs());
        }
        w
    }

    /// **An `SS=1` record on a nonlinear right-hand side is the explicit train** (#1260).
    ///
    /// `solve_ekf` applied the record as a single dose from a zero state with `P = 0`, so
    /// the Jacobian was linearised around a one-dose mean and `P` started accumulating at
    /// the record. Measured before the fix, against the train: `p_obs` 55 % low 2 h after
    /// the record and still 2.2–3.4 % low after the first update (11.80 vs 12.05 at
    /// `t = 485`), the mean 10–15 % low (42.63 vs 47.18); the issue's 120.5-OFV gap on
    /// this model. The ODE path reproduces the train exactly on the same
    /// model (the issue's control), so the train is the oracle here too.
    ///
    /// Regression it catches: the run-in dropped, or `P` seeded at zero after the mean is
    /// equilibrated (the mean alone would pass `ipred` and fail `p_obs`). Non-degeneracy:
    /// the `SS=0` arm is asserted to sit far from the train on *both* quantities, so the
    /// bound below cannot be met by an implementation that ignores the flag.
    #[test]
    fn ekf_ss_record_matches_the_explicit_train_on_a_nonlinear_rhs() {
        let pk = make_pk(1.0, 20.0);
        let (ss, train, single, obs_times) = ss_fixture(100.0, 0.0);
        let run = |doses: &[DoseEvent]| {
            solve_ekf(
                &mm_rhs,
                1,
                0,
                &[5.0],
                &pk,
                &Default::default(),
                &[],
                doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                OdeSolverOptions::default(),
            )
        };
        let (got, want, was) = (run(&ss), run(&train), run(&single));
        let (e_mean, e_cov) = worst_rel(&got, &want);
        // Measured: 2.7e-14 / 3.2e-12 (the run-in and the train are the same quadrature,
        // and the train has converged to the stop tolerance after 41 cycles).
        assert!(
            e_mean < 1e-9 && e_cov < 1e-9,
            "SS=1 record vs the 41-dose train: worst relative error ipred {e_mean:.3e}, \
             p_obs {e_cov:.3e}"
        );
        let (d_mean, d_cov) = worst_rel(&was, &want);
        assert!(
            d_mean > 5e-2 && d_cov > 1e-2,
            "the single-dose arm must be far from the train for this to test anything: \
             ipred {d_mean:.3e}, p_obs {d_cov:.3e} (measured 1.5e-1 / 5.5e-1)"
        );
    }

    /// **The seeded covariance is the stationary Riccati value, in closed form.**
    ///
    /// For `dA/dt = −k·A` the Riccati equation is scalar, `dP/dt = −2k·P + q`, whose
    /// stationary value `q / 2k` is also the exact fixed point of the Crank–Nicolson map the
    /// walk uses (`P ← P + (−2k·P⁺/₂ + q)·Δt`, `P⁺/₂` the average of `P` and the new `P`), for
    /// every `Δt`. A bolus leaves `P` untouched, so
    /// the periodic trough of `P` *is* `q / 2k` with no quadrature error — and once seeded
    /// there it stays there until the first Kalman update, so `p_obs` at the **first**
    /// observation after the record reads `q / 2k` to the run-in's stop tolerance. (Later
    /// observations read the regrowth after the update at `R = 0.01` collapses `P`, and are
    /// not a statement about the seed.) Before #1260 `P` was `0` at the record and read the
    /// one-step approximation of `q·(1 − e^{−2kΔ}) / 2k` — `0.185·q/2k` at `Δ = 2` here.
    ///
    /// The mean is checked at every observation against the closed-form periodic peak
    /// `D·e^{−kΔ} / (1 − e^{−k·II})`, at the solver's default tolerance.
    #[test]
    fn ekf_ss_covariance_seeds_at_the_stationary_riccati_value() {
        let (cl, v, q) = (1.0, 20.0, 0.25);
        let k = cl / v;
        let pk = make_pk(cl, v);
        let (ss, _, single, obs_times) = ss_fixture(100.0, 0.0);
        let run = |doses: &[DoseEvent]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[q],
                &pk,
                &Default::default(),
                &[],
                doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                OdeSolverOptions::default(),
            )
        };
        let got = run(&ss);
        let was = run(&single);
        let p_stationary = q / (2.0 * k);
        assert!(got[0].p_obs.is_finite() && was[0].p_obs.is_finite());
        // Measured: 2.5000000000020 against 2.5 (7.8e-13 relative).
        assert_relative_eq!(got[0].p_obs, p_stationary, max_relative = 1e-9);
        // The straddle: from `P = 0` at the record the covariance is nowhere near
        // stationary 2 h later (measured 0.4637, i.e. 0.185 of it).
        assert!(
            was[0].p_obs < 0.25 * p_stationary,
            "single-dose arm p_obs {} must sit well below q/2k = {p_stationary}",
            was[0].p_obs
        );
        for (j, pt) in got.iter().enumerate() {
            let dt = obs_times[j] - 480.0;
            let peak = 100.0 * (-k * dt).exp() / (1.0 - (-k * 12.0).exp());
            assert!(pt.ipred.is_finite());
            // Measured: 200.546448 against 200.545380 (5.3e-6 relative) at Δ = 2 — the
            // RK45 error at the default `reltol = 1e-4`, compounded over the run-in's
            // cycles by the accumulation factor `1/(1 − e^{−k·II})`. The bound is one
            // order above that; a `P`-only fix leaves the mean 55 % off.
            assert_relative_eq!(pt.ipred, peak, max_relative = 5e-5);
        }
    }

    /// **An `SS=1` infusion record is the explicit infusion train**, on a right-hand side
    /// that reads `TAD` — the shape TeunP measured at 24.1 % on #1263 (`45.60` vs `36.74`),
    /// where the walk's `TAD` anchor folded into `[0, II)` as though a pulse train existed
    /// that this engine had never built. Now it builds one: the run-in's active and quiet
    /// windows carry the cycle's own pulse as their `TAD` anchor, and the record's residual
    /// infusion is admitted by `active_infusions` as on the ODE path. `F = 0.5` on top, so
    /// the `F`-reshaped window (`bioavailable_infusion`) is exercised on both arms.
    #[test]
    fn ekf_ss_infusion_matches_the_explicit_train_on_a_tad_reading_rhs() {
        let rhs = time_reading_rhs(crate::types::MAX_PK_PARAMS + 1, 0.05);
        let mut pk = make_pk(1.0, 20.0);
        pk[crate::types::PK_IDX_F] = 0.5;
        // AMT 100 at RATE 20: a 5 h window, well inside II = 12.
        let (ss, train, single, obs_times) = ss_fixture(100.0, 20.0);
        let run = |doses: &[DoseEvent]| {
            solve_ekf(
                &rhs,
                1,
                0,
                &[2.0],
                &pk,
                &Default::default(),
                &[],
                doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                OdeSolverOptions::default(),
            )
        };
        let (got, want, was) = (run(&ss), run(&train), run(&single));
        let (e_mean, e_cov) = worst_rel(&got, &want);
        // Measured: 7.1e-13 / 9.9e-13.
        assert!(
            e_mean < 1e-9 && e_cov < 1e-9,
            "SS=1 infusion vs the explicit infusion train: worst relative error ipred \
             {e_mean:.3e}, p_obs {e_cov:.3e}"
        );
        let (d_mean, d_cov) = worst_rel(&was, &want);
        assert!(
            d_mean > 1e-1 && d_cov > 1e-1,
            "the single-infusion arm must be far from the train: ipred {d_mean:.3e}, \
             p_obs {d_cov:.3e} (measured 5.2e-1 / 7.6e-1)"
        );
    }

    /// **The run-in integrates on the ODE run-in's cycle-local clock**, so a right-hand side
    /// reading `T` / `TIME` is equilibrated identically by both engines — the objective takes
    /// `IPRED` from the ODE path and `p_obs` from this filter, and they must describe one
    /// history. For `dA/dt = −ke·A·(1 + c·T)` the cycle map on the local clock `(0, II)` is
    /// the constant `M = exp(−ke·(II + c·II²/2))`, so the periodic trough is `D·M/(1 − M)`
    /// in closed form, and the walk after the record at `t = 480` runs on the real clock:
    /// `A(480 + Δ) = (trough + D)·exp(−ke·(Δ + c·((480 + Δ)² − 480²)/2))`. Both engines
    /// are checked against that third reference, not only against each other.
    ///
    /// Regression it catches: a monotone `m·II` clock in the run-in (the first draft of
    /// #1260). Its 50th cycle integrates over `T ∈ (588, 600)`, where elimination is 31×
    /// the local-clock rate, so the trough collapses to ~0 and the seed is the single-dose
    /// state — the `SS=0` arm is asserted far from the closed form so that answer cannot
    /// pass. (A `TAD`-reading RHS cannot see the clock: its anchor moves with it, which is
    /// what `ekf_ss_infusion_matches_the_explicit_train_on_a_tad_reading_rhs` pins.)
    #[test]
    fn ekf_ss_run_in_shares_the_ode_run_in_clock_on_a_time_reading_rhs() {
        use crate::ode::predictions::{ode_predictions, OdeSpec};
        use crate::types::Subject;
        use std::collections::HashMap;

        let (cl, v, c, ii, amt) = (1.0f64, 20.0f64, 0.05f64, 12.0f64, 100.0f64);
        let ke = cl / v;
        fn t_reading_rhs(y: &[f64], p: &[f64], t: f64, dy: &mut [f64]) {
            let ke = p[crate::types::PK_IDX_CL] / p[crate::types::PK_IDX_V];
            dy[0] = -ke * y[0] * (1.0 + 0.05 * t);
        }
        let pk = make_pk(cl, v);
        let (ss, _, single, obs_times) = ss_fixture(amt, 0.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let run = |doses: &[DoseEvent]| {
            solve_ekf(
                &t_reading_rhs,
                1,
                0,
                &[0.0], // zero diffusion: the mean walk is plain integration
                &pk,
                &Default::default(),
                &[],
                doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                opts,
            )
        };
        let (got, was) = (run(&ss), run(&single));

        let subj = Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: ss.clone(),
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![0.0; obs_times.len()],
            obs_cmts: vec![1; obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cmt_toggles: Vec::new(),
            cens: vec![0; obs_times.len()],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
            endpoint_covariates: Vec::new(),
            filled_cells: Vec::new(),
            simulated_mixnum: None,
        };
        let ode_spec = OdeSpec {
            chz_state_slots: Vec::new(),
            rhs: Box::new(t_reading_rhs),
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
        let ode = ode_predictions(&ode_spec, &pk, &[], &[], &subj);

        let m = (-ke * (ii + c * ii * ii / 2.0)).exp();
        let trough = amt * m / (1.0 - m);
        for (j, &t) in obs_times.iter().enumerate() {
            let d = t - 480.0;
            let want = (trough + amt)
                * (-ke * (d + c * ((480.0 + d).powi(2) - 480.0f64.powi(2)) / 2.0)).exp();
            assert!(got[j].ipred.is_finite() && ode[j].is_finite() && was[j].ipred.is_finite());
            // Measured against the closed form: EKF 8.5e-11 … 1.5e-9, ODE 8.6e-11 … 1.5e-9
            // (15.08059537 at Δ = 2 on all three); the bound is three orders above.
            assert_relative_eq!(got[j].ipred, want, max_relative = 1e-6);
            assert_relative_eq!(ode[j], want, max_relative = 1e-6);
            // The straddle: the single-dose seed is 4.58e-1 off at every sample, and under
            // the monotone-clock mutation the EKF reads 8.16755988 at Δ = 2 — the
            // single-dose arm's 8.16755980 to 8 figures — while the ODE twin stays put.
            assert!(
                (was[j].ipred - want).abs() / want > 0.2,
                "single-dose arm {} must sit far from the periodic closed form {want}",
                was[j].ipred
            );
        }
    }

    /// **An overlapping `SS=1` infusion (`T_inf > II`) is the explicit overlapping train.**
    /// The run-in used to return a zero pair for it (the single-dose answer, as the ODE run-in
    /// did); it now replays the always-running earlier pulse as a held rate, and the walk
    /// carries it past the record (`dosing::ss_residual_infusions_g`).
    #[test]
    fn ekf_overlapping_ss_infusion_is_the_explicit_train() {
        let pk = make_pk(1.0, 20.0);
        // AMT 100 at RATE 5: a 20 h window across II = 12.
        let (ss, train, single, obs_times) = ss_fixture(100.0, 5.0);
        let run = |doses: &[DoseEvent]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[1.0],
                &pk,
                &Default::default(),
                &[],
                doses,
                &obs_times,
                &vec![0.01; obs_times.len()],
                OdeSolverOptions::default(),
            )
        };
        let (got, want, was) = (run(&ss), run(&train), run(&single));
        let (e_mean, e_cov) = worst_rel(&got, &want);
        assert!(
            e_mean < 1e-9 && e_cov < 1e-9,
            "overlapping SS=1 infusion vs the explicit train: worst relative error ipred \
             {e_mean:.3e}, p_obs {e_cov:.3e}"
        );
        let (d_mean, d_cov) = worst_rel(&was, &want);
        assert!(
            d_mean > 1e-1 && d_cov > 1e-1,
            "the single-infusion arm must be far from the train: ipred {d_mean:.3e}, \
             p_obs {d_cov:.3e}"
        );
    }

    /// **A run-in that spends its cycle budget without converging warns** through the same
    /// #867 sink the ODE run-in uses, so `fit()` surfaces it. `VMAX·II = 60 < AMT = 100`:
    /// more drug arrives per cycle than can ever be eliminated, the trough climbs a
    /// constant step per cycle and no periodic steady state exists.
    #[test]
    fn ekf_ss_run_in_without_a_steady_state_warns() {
        fn saturated_rhs(y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]) {
            dy[0] = -5.0 * y[0] / (50.0 + y[0]);
        }
        let pk = make_pk(1.0, 20.0);
        let (ss, _, _, obs_times) = ss_fixture(100.0, 0.0);
        let pts = solve_ekf(
            &saturated_rhs,
            1,
            0,
            &[1.0],
            &pk,
            &Default::default(),
            &[],
            &ss,
            &obs_times,
            &vec![0.01; obs_times.len()],
            OdeSolverOptions::default(),
        );
        assert!(pts
            .iter()
            .all(|p| p.ipred.is_finite() && p.p_obs.is_finite()));
        assert!(
            crate::dosing::last_ss_equilibration_warned(),
            "the EKF run-in must record its own non-convergence"
        );
        let warning = crate::dosing::last_ss_equilibration_warning();
        assert!(
            warning
                .as_deref()
                .is_some_and(|w| w.contains("Steady-state (SS=1) equilibration")),
            "a model with no periodic steady state must surface the #867 warning from the \
             EKF run-in; got: {warning:?}"
        );
    }

    /// **Records sharing an observation time inside a segment are all filled, from one
    /// Kalman update.**
    ///
    /// The in-segment read looked the point up in a `HashMap<u64, usize>`, which keeps one
    /// index per time — so for two records at the same instant the loser kept its
    /// `EkfObsPoint { ipred: 0.0, p_obs: 0.0 }` prefill and fed a finite, plausible zero
    /// into the likelihood. Measured: `obs_times = [2.0, 2.0, 8.0]` returned `ipred = 0.0`
    /// for index 0 and `90.4837418037` for index 1 (#1263 review). The boundary-read path
    /// was hardened for this in #1226; this is the same defect one branch over.
    ///
    /// Both halves are asserted: the duplicates are **filled**, and they share a **single**
    /// update — `solve_ekf` has one `obs_cmt_idx`, so N records at one time are N reads of
    /// the same compartment, and updating N times would shrink `p_mat` as though N
    /// independent measurements had arrived.
    #[test]
    fn ekf_duplicate_in_segment_obs_times_share_one_update() {
        let pk = make_pk(1.0, 20.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let doses = vec![bolus_dose(100.0)];
        // t = 2 is strictly inside the only segment [0, 8], so this exercises the
        // in-segment read rather than the boundary read.
        let dup = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.1],
            &pk,
            &Default::default(),
            &[],
            &doses,
            &[2.0, 2.0, 8.0],
            &[0.01; 3],
            opts,
        );
        // The single-record control: what one record at t = 2 gets.
        let single = solve_ekf(
            &one_cpt_rhs,
            1,
            0,
            &[0.1],
            &pk,
            &Default::default(),
            &[],
            &doses,
            &[2.0, 8.0],
            &[0.01; 2],
            opts,
        );

        for (i, p) in dup.iter().enumerate() {
            assert!(
                p.ipred.is_finite() && p.p_obs.is_finite(),
                "idx {i}: non-finite point {p:?}"
            );
        }
        assert!(
            dup[0].ipred > 0.0,
            "the first of two records at t=2 kept its 0.0 prefill: {:?}",
            dup[0]
        );
        assert_eq!(
            dup[0].ipred.to_bits(),
            dup[1].ipred.to_bits(),
            "records at one instant must share an ipred"
        );
        assert_eq!(
            dup[0].p_obs.to_bits(),
            dup[1].p_obs.to_bits(),
            "records at one instant must share a p_obs"
        );
        // One update, not two: the duplicate pair must leave the *later* observation with
        // exactly the covariance the single-record run gives it. Assimilating twice at
        // t = 2 would over-shrink `p_mat` and this would come out low.
        assert_eq!(
            dup[2].p_obs.to_bits(),
            single[1].p_obs.to_bits(),
            "a duplicated observation time must not assimilate twice — p_obs at t=8 is \
             {} with the duplicate and {} without",
            dup[2].p_obs,
            single[1].p_obs
        );
        // And the straddle: the two-update answer really is different, so the assertion
        // above can fail.
        let (p_new, _) = kalman_update(&[dup[0].p_obs], 0, 0.01, 1);
        assert!(
            (p_new[0] - dup[0].p_obs).abs() > 1e-9,
            "a second update at the same instant must move p_mat, or this test is vacuous"
        );
    }

    /// **A co-timed pair the reader split one ULP apart is filled, from one update.**
    ///
    /// Two observations at `TIME 4` with an EVID=2 row after the first reach the engine as
    /// `[4⁻, 4]`: the reader moves the first one ULP below the tie (#1810). The segment
    /// `saveat` deduped at `1e-15`, which merged the pair, so the record at `4` was never
    /// saved and kept its `{ ipred: 0.0, p_obs: 0.0 }` prefill, a finite zero in the
    /// likelihood. With both times saved, the in-segment read must still treat the pair as the
    /// one instant it was written as (`crate::types::same_instant`): one update, as for the
    /// unsplit `[4, 4]`, not two.
    ///
    /// Measured on macOS arm64: worst relative difference from the unsplit run 2.1e-16, over
    /// `ipred` and `p_obs` of all three records. Bound `1e-12`.
    ///
    /// Mutation: drop the one-ULP-up lookup and the pair assimilates twice — `p_obs` of the
    /// record at `4` measured 0.0097 against the unsplit 0.337. Restoring the `1e-15` dedup
    /// alone leaves this test green, since the lookup fills the record at `4` from the `4⁻`
    /// point either way. That dedup is pinned on the other engines, by
    /// `a_reader_split_tie_is_read_on_every_ode_engine` in `predictions_tests.rs`. With both
    /// reverted, the code before the fix, the record at `4` keeps its `0.0` prefill and the
    /// first assertion below fails.
    #[test]
    fn ekf_reader_split_tie_is_one_instant() {
        let pk = make_pk(1.0, 20.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let doses = vec![bolus_dose(100.0)];
        let t = 4.0f64;
        assert!(
            t - t.next_down() < 1e-15,
            "the pair must be close enough for the old 1e-15 dedup to merge it"
        );
        let run = |obs: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[0.1],
                &pk,
                &Default::default(),
                &[],
                &doses,
                obs,
                &[0.01; 3],
                opts,
            )
        };
        let split = run(&[t.next_down(), t, 8.0]);
        let unsplit = run(&[t, t, 8.0]);
        let mut worst = 0.0_f64;
        for (i, (s, u)) in split.iter().zip(&unsplit).enumerate() {
            assert!(
                s.ipred > 0.0 && s.p_obs > 0.0 && s.p_obs.is_finite(),
                "split idx {i}: {s:?} (a prefill reads 0.0)"
            );
            assert_relative_eq!(s.ipred, u.ipred, max_relative = 1e-12);
            assert_relative_eq!(s.p_obs, u.p_obs, max_relative = 1e-12);
            worst = worst
                .max((s.ipred - u.ipred).abs() / u.ipred)
                .max((s.p_obs - u.p_obs).abs() / u.p_obs);
        }
        println!("EKF reader-split tie: worst relative difference {worst:.3e}");
    }

    /// **An obs / dose / obs tie the reader split takes one update too.**
    ///
    /// Obs, bolus, obs at `TIME 4` reach the engine as `[4⁻, 4]` (#614), with a break at `4`
    /// between the pair. The `4⁻` record reads the state before the dose and the `4` record
    /// the state after it, but the pair is still one instant, so it takes one update, as the
    /// unsplit `[4, 4]` does. This tie was already filled before #2010's fix, since a break
    /// sits at `4`, but from two updates: the `4⁻` point took an in-segment update and the
    /// boundary read at `4` a second one. The one-ULP-up lookup makes it one, so an EKF fit
    /// with such a tie moves its objective.
    ///
    /// Measured on macOS arm64: `p_obs` 0.33658 / 0.0097115 / 0.33985 before the fix and
    /// 0.33658 / 0.33658 / 0.34302 after, the unsplit run's to 1.7e-16 relative. `ipred`
    /// is 81.87 / 181.87 / 148.91 either way. Bound `1e-12`.
    ///
    /// Mutation: drop the one-ULP-up lookup and the pair assimilates twice — `p_obs` of the
    /// record at `4` falls back to 0.0097 and the bit-equality below fails.
    #[test]
    fn ekf_reader_split_tie_across_a_dose_is_one_update() {
        let pk = make_pk(1.0, 20.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let t = 4.0f64;
        let dosed = vec![
            bolus_dose(100.0),
            DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0),
        ];
        let run_dosed = |obs: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[0.1],
                &pk,
                &Default::default(),
                &[],
                &dosed,
                obs,
                &[0.01; 3],
                opts,
            )
        };
        let split = run_dosed(&[t.next_down(), t, 8.0]);
        let unsplit = run_dosed(&[t, t, 8.0]);
        for (i, s) in split.iter().enumerate() {
            assert!(
                s.ipred > 0.0 && s.p_obs > 0.0 && s.p_obs.is_finite(),
                "dosed split idx {i}: {s:?} (a prefill reads 0.0)"
            );
        }
        // The means straddle the dose: `4⁻` is pre-dose, `4` post-dose (`ipred` is the
        // amount here, so the step is the dose itself), and the post-dose one is the
        // unsplit run's.
        assert_relative_eq!(split[1].ipred - split[0].ipred, 100.0, max_relative = 1e-12);
        assert_relative_eq!(split[1].ipred, unsplit[1].ipred, max_relative = 1e-12);
        // One update for the pair: both records carry its `p_obs`, the unsplit pair's, and
        // the next observation sees the covariance one update leaves.
        assert_eq!(
            split[0].p_obs.to_bits(),
            split[1].p_obs.to_bits(),
            "obs / dose / obs at one instant must share one update: p_obs {} then {}",
            split[0].p_obs,
            split[1].p_obs
        );
        assert_relative_eq!(split[1].p_obs, unsplit[1].p_obs, max_relative = 1e-12);
        assert_relative_eq!(split[2].p_obs, unsplit[2].p_obs, max_relative = 1e-12);
    }

    /// #2510: an observation the reader moved one ULP below a co-timed dose (`SS=1` or not)
    /// reads the carried pre-dose mean *and covariance*. An SS record replaces the pair with
    /// the periodic trough, so a walk that applied it before reading the `t⁻` record would hand
    /// that record the stationary covariance. The reference is the same subject without the
    /// dose (nothing acts before it), and the control is the record at `t + 4`, which must see
    /// it.
    ///
    /// Measured before the fix: at `t = 0` the record read post-dose (`ipred` 100 for the
    /// bolus, 221.64 for the SS dose, against 0), because a dose matches a break symmetrically
    /// within `EVENT_MATCH_TOL` and the first break is the record's own. `t = 4` and `8.2`
    /// already read pre-dose and are controls. Mutation: reading the band after the doses
    /// (`pre_event` arm removed) fails the `t = 0` cases.
    #[test]
    fn ekf_observation_one_ulp_before_an_ss_dose_reads_the_carried_pair() {
        let pk = make_pk(1.0, 20.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let run = |doses: &[DoseEvent], obs: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[0.1],
                &pk,
                &Default::default(),
                &[],
                doses,
                obs,
                &[0.01; 2],
                opts,
            )
        };
        // `0`: the first instant, where the record is the integration start, so the record and
        // the dose share the first break. `4` and `8.2`: a later dose with residual drug, at two
        // ULP sizes. The EKF breaks on doses and infusion ends, not on observations, so there
        // the record is a save point of the segment ending at the dose's break.
        for (t, ss) in [0.0f64, 4.0, 8.2]
            .into_iter()
            .flat_map(|t| [(t, true), (t, false)])
        {
            let history: Vec<DoseEvent> = if t > 0.0 {
                vec![bolus_dose(100.0)]
            } else {
                vec![]
            };
            let mut doses = history.clone();
            doses.push(DoseEvent::new(t, 100.0, 1, 0.0, ss, 12.0));
            let obs = [t.next_down(), t + 4.0];
            let with_ss = run(&doses, &obs);
            let without = run(&history, &obs);
            for p in &with_ss {
                // `p_obs ≥ 0`, not `> 0`: before the first instant no diffusion has acted, so
                // the carried covariance at `0⁻` is genuinely zero.
                assert!(
                    p.ipred.is_finite() && p.p_obs.is_finite() && p.p_obs >= 0.0,
                    "t = {t}, ss = {ss}: {p:?}"
                );
            }
            assert!(
                (with_ss[0].ipred - without[0].ipred).abs() <= 1e-12 * without[0].ipred.max(1.0),
                "t = {t}, ss = {ss}: ipred {} before the dose, want carried {}",
                with_ss[0].ipred,
                without[0].ipred
            );
            assert_relative_eq!(with_ss[0].p_obs, without[0].p_obs, max_relative = 1e-12);
            // Control: the SS dose reaches the later record.
            assert!(
                with_ss[1].ipred > without[1].ipred + 50.0,
                "t = {t}, ss = {ss}: the dose must reach t + 4: {} vs {}",
                with_ss[1].ipred,
                without[1].ipred
            );
        }
    }

    /// #2510 (review): a record read after an `SS=1` dose at its own break takes the steady
    /// state's covariance. The segment ending at `t` saves the records there and marks them
    /// assimilated against the carried covariance. The SS record then **replaces** `p_mat`
    /// with the periodic one, so a post-dose record must be assimilated afresh against it, and
    /// every later record conditions on that update. Before the fix only its mean was
    /// rewritten, so its `p_obs` and the covariance the next record saw were pre-dose. This
    /// held for a plain dose-then-observation tie (`unsplit`) before #2510. It also holds for
    /// an `obs / SS dose / obs` tie the reader now splits into `[t⁻, t]` (`split`). A bolus does
    /// not touch `p_mat`, which is why the non-SS pair keeps its single update
    /// (`ekf_reader_split_tie_across_a_dose_is_one_update`).
    ///
    /// The control is the same SS dose as the subject's **first** event. The SS record resets
    /// the system, so the history before it cannot reach a later read. With no segment ending
    /// at `t`, the boundary read is the record's only read. Bolus and a 2-h infusion, at `4`
    /// and `8.2`, over two histories:
    ///
    /// * a bolus at `0`, where the SS dose's own time is the break that applies it;
    /// * an infusion ending exactly at `t⁻`. That puts a break one ULP below the SS dose,
    ///   which matches it within `EVENT_MATCH_TOL`, so the dose is applied at a break that is
    ///   not its own time. A reset flag keyed on `dose.time == t_start` fails this history only.
    ///
    /// Each also has a duplicate post-dose record (`[t⁻, t, t, t + 4]`). Both copies are one
    /// instant and must share the steady state's single update, so a clear that stopped at
    /// the first matching record leaves the second stale.
    ///
    /// Measured before the fix: post-dose `p_obs` 0.3366 against the steady state's 1.0000.
    /// Mutation: dropping the re-assimilation fails the first case.
    #[test]
    fn ekf_split_tie_across_an_ss_dose_conditions_on_the_steady_state() {
        let pk = make_pk(1.0, 20.0);
        let opts = OdeSolverOptions {
            abstol: 1e-12,
            reltol: 1e-10,
            ..OdeSolverOptions::default()
        };
        let run = |doses: &[DoseEvent], obs: &[f64]| {
            solve_ekf(
                &one_cpt_rhs,
                1,
                0,
                &[0.1],
                &pk,
                &Default::default(),
                &[],
                doses,
                obs,
                &vec![0.01; obs.len()],
                opts,
            )
        };
        for t in [4.0f64, 8.2] {
            // Rate 1, so the duration is the amount exactly and the end lands on `t⁻`.
            let ends_at_t_minus = DoseEvent::new(0.0, t.next_down(), 1, 1.0, false, 0.0);
            assert_eq!(
                ends_at_t_minus.time + ends_at_t_minus.duration,
                t.next_down(),
                "the infusion must end exactly one ULP below the SS dose"
            );
            for (history, prior) in [("bolus", bolus_dose(100.0)), ("infusion", ends_at_t_minus)] {
                for rate in [0.0, 50.0] {
                    let case = format!("t = {t}, {history} history, SS rate = {rate}");
                    let doses = [prior.clone(), DoseEvent::new(t, 100.0, 1, rate, true, 12.0)];
                    let split = run(&doses, &[t.next_down(), t, t + 4.0]);
                    let unsplit = run(&doses, &[t, t + 4.0]);
                    let dup = run(&doses, &[t.next_down(), t, t, t + 4.0]);
                    // The SS record resets the system, so the history before it cannot reach
                    // a later read, and a subject whose first event is the SS dose has no
                    // segment ending at `t` to read the record early: the boundary read is its
                    // only one. One control per observation vector.
                    //
                    // Open (#2567 review): under an SS **infusion**, the covariance at `t + 4`
                    // depends on whether the dose is applied at a break on its own time or at
                    // one a ULP below it (`t = 4`, measured 0.34315 against 0.34245, ~0.2%).
                    // Which one a subject gets is decided by the other breaks (the history, or
                    // an observation at `t⁻` being the integration start), not by this fix. So
                    // for an SS infusion only the post-dose rows are compared; for an SS bolus
                    // every row is.
                    let ctl = |obs: &[f64]| run(&doses[1..], obs);
                    let (c_split, c_unsplit, c_dup) = (
                        ctl(&[t.next_down(), t, t + 4.0]),
                        ctl(&[t, t + 4.0]),
                        ctl(&[t.next_down(), t, t, t + 4.0]),
                    );
                    for (i, p) in split.iter().chain(&dup).enumerate() {
                        assert!(
                            p.ipred.is_finite() && p.p_obs.is_finite() && p.p_obs > 0.0,
                            "{case}, idx {i}: {p:?}"
                        );
                    }
                    let mut rows = vec![
                        ("split post-dose", &split[1], &c_split[1]),
                        ("unsplit post-dose", &unsplit[0], &c_unsplit[0]),
                        ("duplicate 1st post-dose", &dup[1], &c_dup[1]),
                        ("duplicate 2nd post-dose", &dup[2], &c_dup[2]),
                    ];
                    if rate == 0.0 {
                        rows.extend([
                            ("split later", &split[2], &c_split[2]),
                            ("unsplit later", &unsplit[1], &c_unsplit[1]),
                            ("duplicate later", &dup[3], &c_dup[3]),
                        ]);
                    }
                    for (name, s, c) in rows {
                        assert!(
                            (s.ipred - c.ipred).abs() <= 1e-12 * c.ipred,
                            "{case}: {name} ipred {} vs control {}",
                            s.ipred,
                            c.ipred
                        );
                        assert!(
                            (s.p_obs - c.p_obs).abs() <= 1e-12 * c.p_obs,
                            "{case}: {name} p_obs {} vs first-event SS control {}",
                            s.p_obs,
                            c.p_obs
                        );
                    }
                    // The two copies share one update, bit for bit.
                    assert_eq!(dup[1].p_obs.to_bits(), dup[2].p_obs.to_bits(), "{case}");
                    // Non-vacuity: the pre-dose record's covariance is not the steady state's,
                    // so a post-dose record keeping it would fail above.
                    assert!(
                        (split[0].p_obs - c_split[1].p_obs).abs() > 1e-6 * c_split[1].p_obs,
                        "{case}: pre {} and SS {} covariances coincide",
                        split[0].p_obs,
                        c_split[1].p_obs
                    );
                }
            }
        }
    }

    /// **The EKF covariance on a time-varying rate, against a quadrature oracle.**
    ///
    /// `time_reading_rhs_gets_finite_anchors_on_the_ekf_walk` asserts `p_obs > 0.0`. That is
    /// a real discriminator — `is_finite()` accepts `0.0` and this rejects it, so an EKF
    /// that never propagated fails — but it is a **floor**, not a bound: it passes for any
    /// positive magnitude, including an over-propagating Riccati step (#1263 review). This
    /// gives the same quantity a two-sided oracle.
    ///
    /// `dA/dt = -ke·A·(1 + c·TAFD)` is linear in the state with a time-varying rate
    /// `k(t) = ke·(1 + c·t)` for a single dose at `t = 0`, so the Riccati equation is the
    /// scalar `dP/dt = -2k(t)·P + Q` with `P(0) = 0`, whose solution is
    ///
    /// ```text
    ///   P(t) = Q · e^(-2Φ(t)) · ∫₀ᵗ e^(2Φ(s)) ds,   Φ(t) = ke·(t + c·t²/2)
    /// ```
    ///
    /// The inner integral is not elementary (it is an imaginary error function), so it is
    /// evaluated here by Simpson's rule at 20 000 intervals — arithmetic outside the filter
    /// entirely, not a second call into it. `R` is large so the Kalman update barely
    /// contracts `P` and the comparison is against the free-drift solution, the same device
    /// `ekf_variance_matches_analytic_linear_sde` uses for the constant-rate case.
    ///
    /// The `c = 0` arm is carried alongside as the degenerate control: it must reduce to the
    /// constant-rate closed form `(Q/2ke)·(1 − e^(−2·ke·t))`, and it must differ materially
    /// from the `c > 0` arm, or the fixture would not be testing the time-varying part at
    /// all.
    #[test]
    fn ekf_covariance_matches_a_quadrature_oracle_on_a_time_varying_rate() {
        let (cl, v) = (5.0f64, 100.0f64);
        let ke = cl / v; // 0.05 h⁻¹
        let q = 0.04f64;
        let obs_times = vec![1.0, 4.0, 8.0, 12.0];
        let pk = make_pk(cl, v);
        let doses = vec![bolus_dose(100.0)];
        // Large R: the update contracts P by ~Q/R per observation, i.e. negligibly, so
        // `p_obs` tracks the free-drift solution the oracle below computes.
        let r_obs_vec = vec![1e8f64; obs_times.len()];

        // ∫₀ᵗ e^(2Φ(s)) ds by Simpson's rule. Even `n`, so the composite rule applies.
        let simpson = |t: f64, c: f64| -> f64 {
            let n = 20_000usize;
            let h = t / n as f64;
            let phi = |s: f64| ke * (s + c * s * s / 2.0);
            let f = |s: f64| (2.0 * phi(s)).exp();
            let mut acc = f(0.0) + f(t);
            for i in 1..n {
                let w = if i % 2 == 0 { 2.0 } else { 4.0 };
                acc += w * f(i as f64 * h);
            }
            acc * h / 3.0
        };

        let mut worst = 0.0f64;
        let mut at_c0 = Vec::new();
        let mut at_c1 = Vec::new();
        for c in [0.0f64, 0.05] {
            let rhs = time_reading_rhs(crate::types::MAX_PK_PARAMS, c);
            let pts = solve_ekf(
                &rhs,
                1,
                0,
                &[q],
                &pk,
                &Default::default(),
                &[],
                &doses,
                &obs_times,
                &r_obs_vec,
                OdeSolverOptions::default(),
            );
            for (j, &t) in obs_times.iter().enumerate() {
                let phi_t = ke * (t + c * t * t / 2.0);
                let want = q * (-2.0 * phi_t).exp() * simpson(t, c);
                let got = pts[j].p_obs;
                // `is_finite` before folding — `f64::max` discards `NaN`, so a diverged
                // Riccati step would otherwise leave `worst` at whatever the good rows gave.
                assert!(got.is_finite(), "c={c}: non-finite p_obs at t={t}: {got}");
                let rel = ((got - want) / want).abs();
                worst = worst.max(rel);
                // Measured realised errors (Crank-Nicolson Riccati step, Linux), all eight points:
                //   c=0.00  4.953e-5 / 4.236e-5 / 3.400e-5 / 2.694e-5  at t = 1/4/8/12
                //   c=0.05  1.561e-4 / 1.540e-4 / 1.468e-4 / 1.349e-4
                // Worst 1.56e-4 (the forward-Euler step this replaced sat at 2.5e-2, its own
                // discretisation floor). Bounded at 1e-3 - 6.4x headroom - and the defect this
                // fixture exists to catch (an engine that ignores the model-time term, so
                // `c=0.05` returns the `c=0` curve) is 23% wrong at t=12, 230x the bound.
                assert!(
                    rel < 1e-3,
                    "c={c} t={t}: EKF p_obs {got:.10e} vs quadrature {want:.10e} \
                     (rel {rel:.3e})"
                );
                if c == 0.0 {
                    at_c0.push(got);
                    // The degenerate arm must also reduce to the elementary closed form,
                    // which pins the quadrature itself rather than trusting it.
                    let closed = (q / (2.0 * ke)) * (1.0 - (-2.0 * ke * t).exp());
                    assert!(
                        ((want - closed) / closed).abs() < 1e-9,
                        "the c=0 quadrature must reproduce the constant-rate closed form: \
                         {want} vs {closed}"
                    );
                } else {
                    at_c1.push(got);
                }
            }
        }
        assert!(worst > 0.0, "the comparison loop never ran");

        // The straddle: a time-varying rate must actually move `p_obs`, or every assertion
        // above is satisfied by an engine that ignores the model-time term entirely — the
        // exact failure this fixture family exists to catch.
        let sep = at_c0
            .iter()
            .zip(&at_c1)
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f64, f64::max);
        assert!(
            sep > 0.05,
            "the c=0 and c=0.05 covariance curves must differ materially — worst \
             separation {sep:.3e}"
        );
    }

    /// T8 (#1539): the EKF mean rides the same solver (`solve_ekf` → `solve_ode` →
    /// `integrate_dense_g`), so a decoupled state that diverges NaN-pads the observed mean too,
    /// rather than freezing it at its post-dose value. The EKF is a *caller* of that engine, not
    /// an independent reference — this test pins that it inherits the pad, nothing more.
    ///
    /// `central` observed, `X' = 0.5·X²` from `X(0) = 1` (pole at t = 2), one bolus of 100 at
    /// t = 12, zero diffusion. Before #1539 the mean at t = 20 and 40 was a frozen `100`; the
    /// `0·X` control gives the closed form `100·e^{−0.1(t−12)}`.
    ///
    /// Mutations that redden it: revert the solver's tail pad to the last state (a frozen
    /// `100`); rewrite a `NaN` mean to `0.0` in `ekf_ipred` again (a plausible zero).
    #[test]
    fn ekf_mean_is_nan_after_a_decoupled_state_diverges() {
        let run = |x_rate: f64| {
            let rhs = move |y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let ke = p[crate::types::PK_IDX_CL] / p[crate::types::PK_IDX_V];
                dy[0] = -ke * y[0];
                dy[1] = x_rate * y[1] * y[1];
            };
            let obs_times = [1.0, 20.0, 40.0];
            solve_ekf(
                &rhs,
                2,
                0,
                &[0.0, 0.0],
                &make_pk(5.0, 50.0),
                &Default::default(),
                &[0.0, 1.0],
                &[DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0)],
                &obs_times,
                &[0.01; 3],
                OdeSolverOptions::default(),
            )
            .iter()
            .map(|p| p.ipred)
            .collect::<Vec<f64>>()
        };
        let control = run(0.0);
        let diverged = run(0.5);
        assert_eq!(
            diverged[0].to_bits(),
            control[0].to_bits(),
            "{diverged:?} {control:?}"
        );
        for (i, t) in [(1, 20.0), (2, 40.0)] {
            let want = 100.0 * (-0.1f64 * (t - 12.0)).exp();
            assert!(
                control[i].is_finite() && (control[i] - want).abs() <= 1e-3 * want,
                "control: {control:?}"
            );
            assert!(diverged[i].is_nan(), "t={t}: {diverged:?}");
        }
    }

    // ---- exact Jacobian and recording (the EKF written once over `T`) -------------------

    /// A Michaelis–Menten SDE model (nonlinear, so its Jacobian moves along the trajectory)
    /// with a proportional error, one random effect and an estimated diffusion.
    const MM_SDE: &str = "[parameters]\n  theta TVVMAX(5.0, 0.1, 100.0)\n  theta TVKM(2.0, 0.1, 100.0)\n  \
        theta TVV(10.0, 1.0, 100.0)\n  theta TVKA(1.0, 0.1, 10.0)\n  omega ETA_V ~ 0.05\n  \
        sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  VMAX = TVVMAX\n  KM = TVKM\n  \
        V = TVV * exp(ETA_V)\n  KA = TVKA\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n\
        [odes]\n  d/dt(depot) = -KA * depot\n  d/dt(central) = KA * depot / V - VMAX * central / (KM + central)\n\
        [diffusion]\n  central ~ 0.05\n[error_model]\n  DV ~ proportional(PROP)\n";

    /// Tight tolerances, so the f64 solve is a differentiable oracle (a central difference at
    /// a relative step of 1e-4 reads solver noise of order `reltol / 1e-4`).
    fn tight() -> OdeSolverOptions {
        OdeSolverOptions {
            abstol: 1e-13,
            reltol: 1e-11,
            method: crate::ode::solver::OdeMethod::Rk45,
            ..OdeSolverOptions::default()
        }
    }

    struct Fixture {
        model: crate::types::CompiledModel,
        pk: Vec<f64>,
        slots: [usize; 4],
        doses: Vec<DoseEvent>,
        obs: Vec<f64>,
    }

    fn fixture() -> Fixture {
        let model = crate::parser::model_parser::parse_model_string(MM_SDE).expect("parses");
        let p = &model.default_params;
        let eta = vec![0.0; model.n_eta];
        let pk = (model.pk_param_fn)(&p.theta, &eta, &std::collections::HashMap::new(), 0.0)
            .values
            .to_vec();
        let slot = |name: &str| {
            let i = model
                .indiv_param_names
                .iter()
                .position(|n| n == name)
                .expect("named parameter");
            model.pk_indices[i]
        };
        let slots = [slot("VMAX"), slot("KM"), slot("V"), slot("KA")];
        Fixture {
            model,
            pk,
            slots,
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs: vec![0.5, 1.0, 2.0, 4.0, 5.5, 7.0, 9.0, 12.0],
        }
    }

    /// The exact Jacobian is the compiled program's tangent: against a central difference of
    /// the same program on a nonlinear right-hand side, at three states.
    ///
    /// Measured worst relative error 2.0e-11 (the difference's own truncation, `h² f'''`);
    /// bound 1e-9 (50× headroom). Kills a tangent that drops the `KM + central` denominator's state
    /// dependence, or one that is read at the wrong state.
    #[test]
    fn exact_jacobian_matches_central_fd_on_the_compiled_program() {
        let fx = fixture();
        let ode = fx.model.ode_spec.as_ref().unwrap();
        let rhs = ProgramRhs(ode.rhs_program.as_ref().unwrap());
        let n = 2;
        let mut worst: f64 = 0.0;
        for u in [[3.0, 1.5], [0.2, 4.0], [10.0, 0.05]] {
            let exact = jacobian::<f64, _>(&rhs, &[], &u, &fx.pk, 1.0, (f64::NAN, f64::NAN), n);
            for j in 0..n {
                let h = 1e-5 * (1.0 + u[j].abs());
                let eval = |x: &[f64; 2]| {
                    let mut du = [0.0; 2];
                    rhs.eval::<f64>(x, &fx.pk, 1.0, (f64::NAN, f64::NAN), &mut du);
                    du
                };
                let (mut up, mut um) = (u, u);
                up[j] += h;
                um[j] -= h;
                let (fp, fm) = (eval(&up), eval(&um));
                for i in 0..n {
                    let fd = (fp[i] - fm[i]) / (2.0 * h);
                    assert!(exact[i * n + j].is_finite() && fd.is_finite());
                    worst = worst.max((exact[i * n + j] - fd).abs() / (1.0 + fd.abs()));
                }
            }
        }
        eprintln!("exact-vs-FD Jacobian worst relative error: {worst:e}");
        assert!(worst < 1e-9, "worst {worst:e}");
    }

    /// What the exact Jacobian changes in the value path: `p_obs` from the program's tangent
    /// against the same filter with the old central-FD Jacobian (`FD_H = 1e-5`), default
    /// solver options, on the MM model.
    ///
    /// Measured worst relative `p_obs` difference 6.2e-11 (printed); the exact path is the
    /// better-conditioned one. Bound 1e-9 (16× headroom).
    #[test]
    fn exact_jacobian_moves_p_obs_by_the_fd_truncation_only() {
        let fx = fixture();
        let ode = fx.model.ode_spec.as_ref().unwrap();
        let q = [0.0, 0.05];
        let dose_f = vec![1.0; fx.doses.len()];
        let r = |_: usize, f: f64| 0.01 * f * f + 1e-3;
        let opts = ode.effective_solver_opts();
        let exact = solve_ekf_g::<f64, _>(
            &ProgramRhs(ode.rhs_program.as_ref().unwrap()),
            2,
            1,
            &q,
            &fx.pk,
            &dose_f,
            &[],
            &[],
            &fx.doses,
            &fx.obs,
            &r,
            opts,
        );
        let fd = solve_ekf_g::<f64, _>(
            &FdRhs(ode.rhs.as_ref()),
            2,
            1,
            &q,
            &fx.pk,
            &dose_f,
            &[],
            &[],
            &fx.doses,
            &fx.obs,
            &r,
            opts,
        );
        let mut worst: f64 = 0.0;
        for (a, b) in exact.iter().zip(&fd) {
            assert!(a.p_obs.is_finite() && b.p_obs.is_finite() && a.p_obs > 0.0);
            assert_eq!(a.ipred, b.ipred, "the mean does not read the Jacobian");
            worst = worst.max((a.p_obs - b.p_obs).abs() / b.p_obs); // nan-ok: finite asserted above
        }
        eprintln!("exact-vs-FD-Jacobian p_obs worst relative difference: {worst:e}");
        assert!(worst < 1e-9, "worst {worst:e}");
    }

    /// [`ProgramRhs`] with its program hidden: the same values, so the mean integration's
    /// Rosenbrock step has nothing to take an exact Jacobian from and differences it.
    struct HiddenProgram<'a>(ProgramRhs<'a>);

    impl EkfRhs for HiddenProgram<'_> {
        fn eval<U: VmNum>(&self, u: &[U], p: &[U], t: U, anchors: Anchors<U>, du: &mut [U]) {
            self.0.eval(u, p, t, anchors, du);
        }
    }

    /// The EKF's `f64` mean integration takes the program's exact Jacobian on a Rosenbrock
    /// method, not a difference of its closure (`integrate_mean`).
    ///
    /// Three runs of one filter on `rodas4`: the program's exact `J` (production), the same
    /// right-hand side with its program hidden (so differenced), and the exact run with the
    /// registration switched off (`EXACT_JAC_OFF`, which must reproduce the differenced run bit
    /// for bit). The exact and differenced means must differ (the registration is read) and
    /// agree to the difference's truncation. Measured: the two means differ by 3.6e-9
    /// relative (printed); bound `(1e-13, 1e-5)`. Kills `program()` returning `None` (exact ==
    /// differenced), a registration keyed to the wrong address (same), and a jet that reads
    /// the wrong anchors or state (the means then differ by far more than 1e-5).
    #[test]
    fn mean_integration_takes_the_exact_jacobian_on_a_rosenbrock_method() {
        let fx = fixture();
        let ode = fx.model.ode_spec.as_ref().unwrap();
        let prog = ode.rhs_program.as_ref().unwrap();
        let q = [0.0, 0.05];
        let dose_f = vec![1.0; fx.doses.len()];
        let r = |_: usize, f: f64| 0.01 * f * f + 1e-3;
        let mut opts = ode.effective_solver_opts();
        opts.method = crate::ode::solver::OdeMethod::Rodas4;
        let exact = {
            solve_ekf_g::<f64, _>(
                &ProgramRhs(prog),
                2,
                1,
                &q,
                &fx.pk,
                &dose_f,
                &[],
                &[],
                &fx.doses,
                &fx.obs,
                &r,
                opts.clone(),
            )
        };
        let hidden = || {
            solve_ekf_g::<f64, _>(
                &HiddenProgram(ProgramRhs(prog)),
                2,
                1,
                &q,
                &fx.pk,
                &dose_f,
                &[],
                &[],
                &fx.doses,
                &fx.obs,
                &r,
                opts.clone(),
            )
        };
        let differenced = hidden();
        crate::ode::rosenbrock::EXACT_JAC_OFF.with(|c| c.set(true));
        let off = {
            solve_ekf_g::<f64, _>(
                &ProgramRhs(prog),
                2,
                1,
                &q,
                &fx.pk,
                &dose_f,
                &[],
                &[],
                &fx.doses,
                &fx.obs,
                &r,
                opts.clone(),
            )
        };
        crate::ode::rosenbrock::EXACT_JAC_OFF.with(|c| c.set(false));
        let mut worst: f64 = 0.0;
        for ((e, d), o) in exact.iter().zip(&differenced).zip(&off) {
            assert!(e.ipred.is_finite() && d.ipred.is_finite());
            assert_eq!(
                d.ipred.to_bits(),
                o.ipred.to_bits(),
                "switch-off == differenced"
            );
            worst = worst.max((e.ipred - d.ipred).abs() / d.ipred.abs()); // nan-ok: finite asserted above
        }
        eprintln!("exact-vs-differenced Jacobian, mean relative difference: {worst:e}");
        assert!(worst > 1e-13, "the exact Jacobian was not read: {worst:e}");
        assert!(worst < 1e-5, "worst {worst:e}");
    }

    /// The filter recorded on the recording type (leaves: `VMAX`, `KM`, `V`, `KA`, the
    /// diffusion variance and the proportional `σ²` of the Kalman `R`) gives a plan whose
    /// forward reproduces the `f64` filter and whose one reverse gives the exact gradient of
    /// every `ipred_j` and `p_obs_j` — against central differences of the `f64` filter at
    /// tight solver tolerances.
    ///
    /// Measured worst relative error (printed); bound set from it with headroom. Kills a
    /// Jacobian whose tangent is dropped on the recording (`p_obs` then has no `VMAX`/`KM`/`V`
    /// derivative through `F`), a Kalman `R` held constant (no `σ²` derivative), and a plan
    /// that replays the recording point's values.
    #[test]
    fn recorded_filter_gradient_matches_central_fd() {
        use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
        use crate::sens::num::vm::Layout;
        let fx = fixture();
        let ode = fx.model.ode_spec.as_ref().unwrap();
        let rhs = ProgramRhs(ode.rhs_program.as_ref().unwrap());
        let dose_f = |_: usize| 1.0;
        let opts = tight();
        // leaves: VMAX, KM, V, KA, q, sig2
        let x0 = [5.0, 2.0, 10.0, 1.0, 0.05, 0.01];
        let n_in = x0.len();
        let run_f64 = |x: &[f64]| -> Vec<f64> {
            let mut pk = fx.pk.clone();
            for (s, v) in fx.slots.iter().zip(x) {
                pk[*s] = *v;
            }
            let f = vec![dose_f(0); fx.doses.len()];
            let r = move |_: usize, f: f64| x[5] * f * f + 1e-3;
            solve_ekf_g::<f64, _>(
                &rhs,
                2,
                1,
                &[0.0, x[4]],
                &pk,
                &f,
                &[],
                &[],
                &fx.doses,
                &fx.obs,
                &r,
                opts,
            )
            .iter()
            .flat_map(|p| [p.ipred, p.p_obs])
            .collect()
        };
        let want = run_f64(&x0);
        let n_out = want.len();

        kj::begin();
        let lv: Vec<kj::Tr> = x0
            .iter()
            .enumerate()
            .map(|(j, &v)| kj::Tr::theta(j, v))
            .collect();
        let mut params: Vec<kj::Tr> = fx.pk.iter().map(|&v| kj::Tr::from_f64(v)).collect();
        for (s, l) in fx.slots.iter().zip(&lv) {
            params[*s] = *l;
        }
        let one = kj::Tr::from_f64(1.0);
        let sig2 = lv[5];
        let r = move |_: usize, f: kj::Tr| sig2 * f * f + kj::Tr::from_f64(1e-3);
        let pts = solve_ekf_g::<kj::Tr, _>(
            &rhs,
            2,
            1,
            &[kj::Tr::from_f64(0.0), lv[4]],
            &params,
            &vec![one; fx.doses.len()],
            &[],
            &[],
            &fx.doses,
            &fx.obs,
            &r,
            opts,
        );
        let regs: Vec<u32> = pts
            .iter()
            .flat_map(|p| [p.ipred.reg(), p.p_obs.reg()])
            .collect();
        let rec = kj::end(Some(regs)).expect("the filter records");
        let plan = rec.plan(Layout::F64, 0, n_in).expect("plans");
        let mut out = vec![0.0; n_out + 1];
        let fwd = plan
            .forward_layout(Layout::F64, &[], &x0, &mut out)
            .expect("forward");
        assert_eq!(out[n_out], 1.0, "guards hold at the recording point");
        for (g, w) in out[..n_out].iter().zip(&want) {
            assert_relative_eq!(*g, *w, max_relative = 1e-10);
        }
        // One reverse, a unit seed per output.
        let mut du_bar = vec![0.0; n_out * n_out];
        for o in 0..n_out {
            du_bar[o * n_out + o] = 1.0;
        }
        let mut p_bar = vec![0.0; n_out * n_in];
        assert!(plan.reverse_from(&fwd, Layout::F64, n_out, &mut [], &mut p_bar, &du_bar));

        // A replay at another point is the filter there, not the recording point's.
        let x1 = [4.0, 3.0, 12.0, 1.4, 0.08, 0.02];
        let mut out1 = vec![0.0; n_out + 1];
        plan.forward_layout(Layout::F64, &[], &x1, &mut out1)
            .expect("forward");
        if out1[n_out] == 1.0 {
            let want1 = run_f64(&x1);
            for (g, w) in out1[..n_out].iter().zip(&want1) {
                assert_relative_eq!(*g, *w, max_relative = 1e-6);
            }
        }

        let mut worst: f64 = 0.0;
        for k in 0..n_in {
            let h = 1e-4 * x0[k].abs();
            let (mut xp, mut xm) = (x0, x0);
            xp[k] += h;
            xm[k] -= h;
            let (fp, fm) = (run_f64(&xp), run_f64(&xm));
            for o in 0..n_out {
                let fd = (fp[o] - fm[o]) / (2.0 * h);
                let ad = p_bar[o * n_in + k];
                assert!(fd.is_finite() && ad.is_finite());
                let scale = fd.abs().max(1e-3 * want[o].abs()).max(1e-12);
                worst = worst.max((ad - fd).abs() / scale); // nan-ok: finite asserted above
            }
        }
        eprintln!("recorded EKF gradient vs central FD, worst scaled error: {worst:e}");
        assert!(worst < 1e-4, "worst {worst:e}");
    }

    /// The filter's derivative in a moving infusion window's `D` (a modeled `RATE=-2` dose,
    /// handed in as `modeled`) — recorded on the recording type, one reverse — against central
    /// differences of the `f64` filter on the doses resolved at `D ± h`, for the doses at 0 and
    /// 6 (the second on residual drug, records both sides of each window end) and for an `SS=1`
    /// record (`II = 6`, the window the steady-state cycle's active/quiet split) under a
    /// `TAD`-reading right-hand side, whose quiet window's anchor is the window's negative.
    /// Each record's window-adjacent Riccati sub-interval stays below `RICCATI_DT_MAX`, so the
    /// `f64` filter has no sub-step-count jump inside the difference.
    ///
    /// Kills the window's derivative dropped from the mean solve's span, from the Riccati
    /// sub-steps' lengths and midpoints, or from the steady-state cycle's split and `TAD` anchor.
    #[test]
    fn recorded_filter_follows_a_moving_infusion_window() {
        use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
        use crate::sens::num::vm::Layout;
        // `MM_SDE` with a modeled duration `D1` on the depot.
        let with_d1 = |src: &str| {
            crate::parser::model_parser::parse_model_string(
                &src.replace(
                    "  omega ETA_V",
                    "  theta TVD1(1.9, 0.1, 10.0)\n  omega ETA_V",
                )
                .replace("  KA = TVKA\n", "  KA = TVKA\n  D1 = TVD1\n"),
            )
            .unwrap()
        };
        let plain = with_d1(MM_SDE);
        let tad = with_d1(&MM_SDE.replace(
            "VMAX * central / (KM + central)",
            "VMAX * central / (KM + central) * (1 + 0.05 * TAD)",
        ));
        let modeled = |t: f64, ss: bool| {
            DoseEvent::modeled(
                t,
                100.0,
                1,
                ss,
                6.0,
                crate::types::RateMode::ModeledDuration,
            )
        };
        let obs = [
            0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 4.0, 6.5, 7.0, 7.4, 7.8, 8.2, 8.6, 10.0,
        ];
        let opts = tight();
        let r = |_: usize, f: f64| 0.01 * f * f + 1e-3;
        let d0 = 1.9;
        let mut worst: f64 = 0.0;
        for (label, model, doses) in [
            (
                "two doses",
                &plain,
                vec![modeled(0.0, false), modeled(6.0, false)],
            ),
            ("SS", &plain, vec![modeled(0.0, true), modeled(6.0, false)]),
            (
                "SS + TAD",
                &tad,
                vec![modeled(0.0, true), modeled(6.0, false)],
            ),
        ] {
            let ode = model.ode_spec.as_ref().unwrap();
            let rhs = &ProgramRhs(ode.rhs_program.as_ref().unwrap());
            let p = &model.default_params;
            let pk0 = (model.pk_param_fn)(&p.theta, &[0.0], &Default::default(), 0.0).values;
            let slot = model.pk_indices[model
                .indiv_param_names
                .iter()
                .position(|n| n == "D1")
                .unwrap()];
            let pk_at = |d: f64| {
                let mut pk = pk0.to_vec();
                pk[slot] = d;
                pk
            };
            let resolve = |d: f64| -> Vec<DoseEvent> {
                doses
                    .iter()
                    .map(|x| x.resolve_rate(&ode.dose_attr_map, &pk_at(d)))
                    .collect()
            };
            let run_f64 = |d: f64| -> Vec<f64> {
                let f = vec![1.0; doses.len()];
                solve_ekf_g::<f64, _>(
                    rhs,
                    2,
                    1,
                    &[0.0, 0.05],
                    &pk_at(d),
                    &f,
                    &[],
                    &[],
                    &resolve(d),
                    &obs,
                    &r,
                    opts,
                )
                .iter()
                .flat_map(|p| [p.ipred, p.p_obs])
                .collect()
            };
            let want = run_f64(d0);
            let n_out = want.len();
            kj::begin();
            let leaf = kj::Tr::theta(0, d0);
            let mut params: Vec<kj::Tr> = pk_at(d0).iter().map(|&v| kj::Tr::from_f64(v)).collect();
            params[slot] = leaf;
            let rt = |_: usize, f: kj::Tr| kj::Tr::from_f64(0.01) * f * f + kj::Tr::from_f64(1e-3);
            let pts = solve_ekf_g::<kj::Tr, _>(
                rhs,
                2,
                1,
                &[kj::Tr::from_f64(0.0), kj::Tr::from_f64(0.05)],
                &params,
                &vec![kj::Tr::from_f64(1.0); doses.len()],
                &vec![Some(leaf); doses.len()],
                &[],
                &resolve(d0),
                &obs,
                &rt,
                opts,
            );
            let regs: Vec<u32> = pts
                .iter()
                .flat_map(|p| [p.ipred.reg(), p.p_obs.reg()])
                .collect();
            let rec = kj::end(Some(regs)).expect("the filter records");
            let plan = rec.plan(Layout::F64, 0, 1).expect("plans");
            let mut out = vec![0.0; n_out + 1];
            let fwd = plan
                .forward_layout(Layout::F64, &[], &[d0], &mut out)
                .expect("forward");
            assert_eq!(
                out[n_out], 1.0,
                "{label}: guards hold at the recording point"
            );
            for (g, w) in out[..n_out].iter().zip(&want) {
                assert_relative_eq!(*g, *w, max_relative = 1e-10);
            }
            let mut du_bar = vec![0.0; n_out * n_out];
            for o in 0..n_out {
                du_bar[o * n_out + o] = 1.0;
            }
            let mut p_bar = vec![0.0; n_out];
            assert!(plan.reverse_from(&fwd, Layout::F64, n_out, &mut [], &mut p_bar, &du_bar));
            let h = 1e-4 * d0;
            let (fp, fm) = (run_f64(d0 + h), run_f64(d0 - h));
            let mut moved: f64 = 0.0;
            let mut case_worst: f64 = 0.0;
            for o in 0..n_out {
                let fd = (fp[o] - fm[o]) / (2.0 * h);
                assert!(fd.is_finite() && p_bar[o].is_finite(), "{label} output {o}");
                assert!(want[o].is_finite(), "{label} output {o}");
                let scale = fd.abs().max(1e-3 * want[o].abs()).max(1e-12);
                case_worst = case_worst.max((p_bar[o] - fd).abs() / scale); // nan-ok: operands asserted finite above; scale >= 1e-12
                if o % 2 == 1 {
                    moved = moved.max(fd.abs() / want[o].abs().max(1e-12));
                }
            }
            eprintln!("{label}: worst scaled error {case_worst:e}");
            worst = worst.max(case_worst); // nan-ok: case_worst folds asserted-finite terms
                                           // The window is live in the covariance, not only in the mean.
            assert!(moved > 1e-3, "{label}: D does not reach p_obs ({moved:e})");
        }
        eprintln!("recorded EKF moving window vs central FD, worst scaled error: {worst:e}");
        // Measured: 7.5e-8 (two doses, SS), 6.0e-7 (SS + TAD) — the difference's own error.
        assert!(worst < 1e-5, "worst {worst:e}");
    }
}
