//! Generic (`PkNum`) compartment-state propagators — the `Dual2`-differentiable
//! mirror of the `f64` event-driven propagators in [`crate::pk::event_driven`].
//!
//! The closed-form *superposition* provider ([`super::provider`]) sums
//! independent per-dose contributions using a single per-subject `pk`. That works
//! when the PK parameters are constant over the subject's timeline. **IOV** and
//! **time-varying covariates** break it: the parameters *switch mid-decay* (a dose
//! given in one occasion carries over into the next and continues decaying with
//! that occasion's clearance — NONMEM #104 semantics). Representing that exactly
//! needs an *event walk* that carries the compartment **amounts** across occasion
//! boundaries and swaps the parameters at each boundary — exactly what the
//! production `f64` walker does.
//!
//! These propagators evolve the **dual** state over one inter-event interval, so
//! the walk can be run over `Dual2` and yield exact `∂(amount)/∂(η,κ)` (and second
//! order) — closed-form per segment, no numerical ODE integration. This module is
//! the foundation; the dual event walk + readout + the per-occasion block-Ω outer
//! assembly build on top (issue #367).

use super::num::PkNum;
use crate::pk::event_driven::{Event, EventKind, EventSchedule};
use crate::pk::topology::Channel;
use crate::types::{DoseEvent, PkModel, Subject};

/// 1-cpt IV/central propagator: evolve `state[0]` (central amount) over `dt` with
/// a constant input `rate` into the central compartment. Mirror of
/// [`crate::pk::event_driven::propagate_one_cpt`].
pub fn propagate_one_cpt_g<T: PkNum>(state: &mut [T], dt: T, cl: T, v: T, rate: T) {
    if v.val() <= 0.0 || cl.val() <= 0.0 {
        // Degenerate params: skip (the outer optimizer sees a poor OFV and steps
        // away), matching the production propagator.
        return;
    }
    let ke = cl / v;
    let exp_term = (-(ke * dt)).exp();
    state[0] = exp_term * state[0] + (rate / ke) * (T::from_f64(1.0) - exp_term);
}

/// 1-cpt oral propagator. `state = [A_depot, A_central]`; the depot drains into
/// the central compartment at absorption rate `ka`. `rate_central` is a constant
/// zero-order input into central (depot-bypass infusion, RATE>0 into cmt 2);
/// `rate_depot` is a constant zero-order input into the depot (RATE>0 into cmt 1,
/// #400). Both are added by linear superposition and are `0` for bolus dosing.
/// Mirror of [`crate::pk::event_driven::propagate_one_cpt_oral`], including the
/// `ka ≈ ke` L'Hôpital limit.
pub fn propagate_one_cpt_oral_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    cl: T,
    v: T,
    ka: T,
    rate_central: T,
    rate_depot: T,
) {
    if v.val() <= 0.0 || cl.val() <= 0.0 || ka.val() <= 0.0 {
        return;
    }
    let ke = cl / v;
    let dtt = dt;
    let one = T::from_f64(1.0);
    let e_ka = (-(ka * dtt)).exp();
    let e_ke = (-(ke * dtt)).exp();
    let a_d_0 = state[0];
    let a_c_0 = state[1];
    let near = (ka.val() - ke.val()).abs() < 1e-9;

    // Depot decays exponentially (decoupled).
    state[0] = a_d_0 * e_ka;

    // Central: homogeneous decay of A_c(0) plus the depot-driven Bateman term,
    // with the `ka ≈ ke` L'Hôpital fallback (branch on `.val()`).
    if near {
        state[1] = a_c_0 * e_ke + ka * a_d_0 * dtt * e_ke;
    } else {
        state[1] = a_c_0 * e_ke + (ka * a_d_0 / (ke - ka)) * (e_ka - e_ke);
    }

    // Constant infusion into central (depot bypass): forced response of the 1-cpt
    // IV propagator from a zero initial state, `(R/ke)·(1 − e_ke)`. Zero when
    // `rate_central` is zero, so applied unconditionally (derivative-safe).
    state[1] = state[1] + (rate_central / ke) * (one - e_ke);

    // Constant zero-order input into the depot (#400), added by linearity:
    //   depot:   (R/ka)·(1 − e_ka)
    //   central: (R/ke)·(1 − e_ke) − R·(e_ka − e_ke)/(ke − ka), L'Hôpital → R·dt·e_ke.
    state[0] = state[0] + (rate_depot / ka) * (one - e_ka);
    let central_forced = if near {
        rate_depot / ke * (one - e_ke) - rate_depot * dtt * e_ke
    } else {
        rate_depot / ke * (one - e_ke) - rate_depot * (e_ka - e_ke) / (ke - ka)
    };
    state[1] = state[1] + central_forced;
}

// ─── 1-cpt event-driven sensitivity walk ─────────────────────────────
//
// The propagators above evolve the dual state across *one* inter-event interval.
// This section stacks them into a full event walk — the `Dual2`-differentiable
// mirror of `event_driven::event_driven_predictions_with_schedule_impl`, but for
// the 1-cpt models only and over `PkNum`. The walk carries the dual compartment
// **amounts** across every event boundary and uses the **per-event** PK params,
// so IOV (parameters that switch at occasion boundaries, NONMEM #104) and
// time-varying covariates are exact: occasion 1's dose decays with occasion-1
// params to the boundary, then the carried-over amount continues with occasion-2
// params. Steady-state doses (`ss` + `ii > 0`) equilibrate per-event with that
// event's params, so SS composes with IOV the same way production does.
//
// Per-event PK params are passed in already seeded as `T` (e.g. `Dual2<M>` with
// each occasion's `(η, κ)`-derived params on their axes); the walk is agnostic to
// how they were seeded, which keeps it testable in isolation against FD of the
// `f64` production walk.

/// Precomputed 2-cpt disposition eigendata — the eigenvalues (α, β) and the
/// micro-rates the propagator formulas read. [`two_cpt_eigen_g`] is the **only**
/// place the 2-cpt eigenvalue solve lives; both the inline `*_g` wrappers and the
/// per-walk [`EigenCacheG`] feed the same `*_core_g` formulas. Caching this across
/// a constant-parameter walk avoids re-solving `sqrt` every interval (the cost
/// profiling flagged on infusion/reset-heavy 3-cpt fits — the 2-cpt analogue).
#[derive(Clone, Copy)]
pub struct TwoCptEigen<T: PkNum> {
    pub alpha: T,
    pub beta: T,
    pub k10: T,
    pub k12: T,
    pub k21: T,
}

/// Compute the 2-cpt eigendata, or `None` for degenerate params (the single guard
/// + eigenvalue solve). `None` ⇒ callers skip propagation, matching the old f64
/// propagators' early return.
pub fn two_cpt_eigen_g<T: PkNum>(cl: T, v1: T, q: T, v2: T) -> Option<TwoCptEigen<T>> {
    if v1.val() <= 0.0 || cl.val() <= 0.0 || v2.val() <= 0.0 || q.val() <= 0.0 {
        return None;
    }
    let (alpha, beta, k21) = crate::sens::two_cpt::macro_rates_g(cl, v1, q, v2);
    Some(TwoCptEigen {
        alpha,
        beta,
        k10: cl / v1,
        k12: q / v1,
        k21,
    })
}

/// 2-cpt IV/central propagation over `dt` from **precomputed** eigendata — the
/// single source of the 2-cpt eigenmode formula, shared by the inline wrapper and
/// the cached walk path.
pub fn propagate_two_cpt_core_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    e: &TwoCptEigen<T>,
    rate_central: T,
    rate_periph: T,
) {
    let (alpha, beta, k10, k12, k21) = (e.alpha, e.beta, e.k10, e.k12, e.k21);
    let dtt = dt;

    let denom_ss = k21 * k10;
    let (a_ss_1, a_ss_2) = if denom_ss.val() > 1e-30 {
        (
            (k21 * rate_central + k21 * rate_periph) / denom_ss,
            (k12 * rate_central + (k10 + k12) * rate_periph) / denom_ss,
        )
    } else {
        (T::from_f64(0.0), T::from_f64(0.0))
    };

    let h1_0 = state[0] - a_ss_1;
    let h2_0 = state[1] - a_ss_2;

    let denom = beta - alpha;
    let (c1, c2) = if k12.val().abs() < 1e-30 {
        (T::from_f64(0.0), T::from_f64(0.0))
    } else if denom.val().abs() < 1e-30 {
        let s_homog = h2_0 / k12;
        (s_homog * T::from_f64(0.5), s_homog * T::from_f64(0.5))
    } else {
        let s_homog = h2_0 / k12;
        let c1 = (h1_0 - s_homog * (k21 - beta)) / denom;
        let c2 = s_homog - c1;
        (c1, c2)
    };

    let e_alpha = (-(alpha * dtt)).exp();
    let e_beta = (-(beta * dtt)).exp();
    let h1_dt = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let h2_dt = (c1 * e_alpha + c2 * e_beta) * k12;

    state[0] = h1_dt + a_ss_1;
    state[1] = h2_dt + a_ss_2;
}

/// 2-cpt IV/central propagator computing its eigendata inline. Generic mirror of
/// [`crate::pk::event_driven::propagate_two_cpt`]; the cached walk path computes
/// eigendata once via [`two_cpt_eigen_g`] and calls [`propagate_two_cpt_core_g`].
#[allow(clippy::too_many_arguments)]
pub fn propagate_two_cpt_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    cl: T,
    v1: T,
    q: T,
    v2: T,
    rate_central: T,
    rate_periph: T,
) {
    if let Some(e) = two_cpt_eigen_g(cl, v1, q, v2) {
        propagate_two_cpt_core_g(state, dt, &e, rate_central, rate_periph);
    }
}

/// 2-cpt oral propagator: `state = [A_depot, A_central, A_periph]`. `rate_central`
/// is a constant zero-order input into central (depot-bypass infusion, RATE>0 into
/// cmt 2); `rate_depot` is a constant zero-order input into the depot (RATE>0 into
/// cmt 1, #400). Both are added by linear superposition and are `0` for bolus
/// dosing. Generic mirror of
/// [`crate::pk::event_driven::propagate_two_cpt_oral`].
/// 2-cpt oral propagation from **precomputed** eigendata — the single source of the
/// 2-cpt oral eigenmode + forced-response formula. `ka` is absorption; `rate_central`
/// / `rate_depot` are the #350/#400 infusion inputs.
pub fn propagate_two_cpt_oral_core_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    e: &TwoCptEigen<T>,
    ka: T,
    rate_central: T,
    rate_periph1: T,
    rate_depot: T,
) {
    let (alpha, beta, k10, k12, k21) = (e.alpha, e.beta, e.k10, e.k12, e.k21);
    let dtt = dt;

    let a_d_0 = state[0];
    let a_c_0 = state[1];
    let a_p_0 = state[2];

    let e_ka = (-(ka * dtt)).exp();
    let e_alpha = (-(alpha * dtt)).exp();
    let e_beta = (-(beta * dtt)).exp();

    state[0] = a_d_0 * e_ka;

    let denom_depot = (ka - alpha) * (ka - beta);
    let (cap_a, cap_b) = if denom_depot.val().abs() < 1e-12 {
        (T::from_f64(0.0), T::from_f64(0.0))
    } else {
        let a = ka * a_d_0 * (k21 - ka) / denom_depot;
        let b = ka * a_d_0 * k12 / denom_depot;
        (a, b)
    };

    let h_c_0 = a_c_0 - cap_a;
    let h_p_0 = a_p_0 - cap_b;

    let denom = beta - alpha;
    let (c1, c2) = if k12.val().abs() < 1e-30 || denom.val().abs() < 1e-30 {
        let kk = if k12.val().abs() < 1e-30 {
            T::from_f64(1e-30)
        } else {
            k12
        };
        let s_homog = h_p_0 / kk;
        (s_homog * T::from_f64(0.5), s_homog * T::from_f64(0.5))
    } else {
        let s_homog = h_p_0 / k12;
        let c1 = (h_c_0 - s_homog * (k21 - beta)) / denom;
        let c2 = s_homog - c1;
        (c1, c2)
    };

    let h_c_dt = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let h_p_dt = (c1 * e_alpha + c2 * e_beta) * k12;

    state[1] = h_c_dt + cap_a * e_ka;
    state[2] = h_p_dt + cap_b * e_ka;

    // Constant infusion into central (depot bypass) and/or into the **peripheral**
    // (#375): forced response of the 2-cpt IV propagator from a zero initial state.
    //
    // Both reduce to the same delegation because the depot is decoupled from the
    // disposition sub-system in the inflow direction — nothing flows *back* into
    // the depot — so a rate into central or peripheral drives exactly the
    // central/peripheral pair an IV model has, and the system is linear, so the
    // forced response superposes onto the homogeneous evolution above. The oral
    // peripheral therefore needs no new closed form: it *is* the IV one.
    //
    // `rate_*` are structurally 0 or positive (never crossing during
    // differentiation), so the `.val()` guards are derivative-safe and just skip
    // the no-infusion intervals.
    if rate_central.val() > 0.0 || rate_periph1.val() > 0.0 {
        let mut inflow = [T::from_f64(0.0), T::from_f64(0.0)];
        propagate_two_cpt_core_g(&mut inflow, dt, e, rate_central, rate_periph1);
        state[1] = state[1] + inflow[0];
        state[2] = state[2] + inflow[1];
    }

    // Constant zero-order input into the depot (#400): forced response from a zero
    // initial state = `x_ss − e^{A·dt}·x_ss`, where `x_ss` is the full steady state
    // under constant depot input R (central_ss = R/k10, periph_ss = k12·R/(k21·k10),
    // depot_ss = R/ka). `e^{A·dt}·x_ss` is this propagator's homogeneous evolution,
    // obtained by recursing once with zero rates (the `.val()` guard bounds the
    // recursion to depth 1 and is derivative-safe).
    if rate_depot.val() > 0.0 {
        let denom_ss = k21 * k10;
        let (c_ss, p_ss) = if denom_ss.val() > 1e-30 {
            (rate_depot / k10, k12 * rate_depot / denom_ss)
        } else {
            (T::from_f64(0.0), T::from_f64(0.0))
        };
        let d_ss = rate_depot / ka;
        let mut xss = [d_ss, c_ss, p_ss];
        propagate_two_cpt_oral_core_g(
            &mut xss,
            dt,
            e,
            ka,
            T::from_f64(0.0),
            T::from_f64(0.0),
            T::from_f64(0.0),
        );
        state[0] = state[0] + (d_ss - xss[0]);
        state[1] = state[1] + (c_ss - xss[1]);
        state[2] = state[2] + (p_ss - xss[2]);
    }
}

/// 2-cpt oral propagator computing its eigendata inline (the dual-walk wrapper).
#[allow(clippy::too_many_arguments)]
pub fn propagate_two_cpt_oral_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    cl: T,
    v1: T,
    q: T,
    v2: T,
    ka: T,
    rate_central: T,
    rate_periph1: T,
    rate_depot: T,
) {
    if ka.val() <= 0.0 {
        return;
    }
    if let Some(e) = two_cpt_eigen_g(cl, v1, q, v2) {
        propagate_two_cpt_oral_core_g(state, dt, &e, ka, rate_central, rate_periph1, rate_depot);
    }
}

/// One 3-cpt eigenmode's spectral data over `T` (mirror of
/// `event_driven::ThreeCptMode` / `build_three_cpt_mode`).
#[derive(Clone, Copy)]
pub struct ThreeCptModeG<T: PkNum> {
    mu: T,
    v: [T; 3],
    w: [T; 3],
    norm: T,
}

/// Precomputed 3-cpt disposition eigendata: the eigenvalues (α, β, γ), micro-rates,
/// the three eigenmodes (the homogeneous propagation basis), and the raw
/// disposition params the IV steady-state amounts read directly (kept verbatim so
/// the formula is bit-identical, not re-derived). [`three_cpt_eigen_g`] is the only
/// place the cubic `acos` eigenvalue solve lives; [`EigenCacheG`] caches it across a
/// constant-parameter walk (the cost the original per-walk cache targeted).
#[derive(Clone, Copy)]
pub struct ThreeCptEigen<T: PkNum> {
    pub alpha: T,
    pub beta: T,
    pub gamma: T,
    pub k10: T,
    pub k12: T,
    pub k13: T,
    pub k21: T,
    pub k31: T,
    pub modes: [ThreeCptModeG<T>; 3],
    pub cl: T,
    pub v1: T,
    pub v2: T,
    pub q2: T,
    pub v3: T,
    pub q3: T,
}

/// Compute the 3-cpt eigendata, or `None` for degenerate params.
pub fn three_cpt_eigen_g<T: PkNum>(
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
) -> Option<ThreeCptEigen<T>> {
    if v1.val() <= 0.0
        || cl.val() <= 0.0
        || v2.val() <= 0.0
        || q2.val() <= 0.0
        || v3.val() <= 0.0
        || q3.val() <= 0.0
    {
        return None;
    }
    let (alpha, beta, gamma, k21, k31) =
        crate::sens::three_cpt::macro_rates_three_cpt_g(cl, v1, q2, v2, q3, v3);
    let k12 = q2 / v1;
    let k13 = q3 / v1;
    let modes = [
        build_three_cpt_mode_g(alpha, k12, k13, k21, k31),
        build_three_cpt_mode_g(beta, k12, k13, k21, k31),
        build_three_cpt_mode_g(gamma, k12, k13, k21, k31),
    ];
    Some(ThreeCptEigen {
        alpha,
        beta,
        gamma,
        k10: cl / v1,
        k12,
        k13,
        k21,
        k31,
        modes,
        cl,
        v1,
        v2,
        q2,
        v3,
        q3,
    })
}

#[inline]
fn build_three_cpt_mode_g<T: PkNum>(mu: T, k12: T, k13: T, k21: T, k31: T) -> ThreeCptModeG<T> {
    let d21 = k21 - mu;
    let d31 = k31 - mu;
    let v = [d21 * d31, k12 * d31, k13 * d21];
    let w = [d21 * d31, k21 * d31, k31 * d21];
    let norm = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    ThreeCptModeG { mu, v, w, norm }
}

#[inline]
fn apply_three_cpt_mode_g<T: PkNum>(m: &ThreeCptModeG<T>, c: T, p1: T, p2: T, dt: T) -> (T, T, T) {
    if m.norm.val().abs() < 1e-30 {
        return (T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0));
    }
    let proj = m.w[0] * c + m.w[1] * p1 + m.w[2] * p2;
    let coef = proj / m.norm;
    let exp_term = (-(m.mu * dt)).exp();
    (
        coef * m.v[0] * exp_term,
        coef * m.v[1] * exp_term,
        coef * m.v[2] * exp_term,
    )
}

/// 3-cpt IV/central propagation from **precomputed** eigendata — the single source
/// of the 3-cpt eigenmode formula. Spectral decomposition along (α, β, γ) with the
/// steady-state + homogeneous pattern for constant infusion.
pub fn propagate_three_cpt_core_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    e: &ThreeCptEigen<T>,
    rate_central: T,
    rate_periph1: T,
    rate_periph2: T,
) {
    let (cl, v1, v2, q2, v3, q3) = (e.cl, e.v1, e.v2, e.q2, e.v3, e.q3);

    let r_total = rate_central + rate_periph1 + rate_periph2;
    let a_ss_c = r_total * v1 / cl;
    let a_ss_p1 =
        (rate_central + rate_periph2) * v2 / cl + rate_periph1 * (cl + q2) * v2 / (cl * q2);
    let a_ss_p2 =
        (rate_central + rate_periph1) * v3 / cl + rate_periph2 * (cl + q3) * v3 / (cl * q3);

    let h_c = state[0] - a_ss_c;
    let h_p1 = state[1] - a_ss_p1;
    let h_p2 = state[2] - a_ss_p2;

    let (ca, p1a, p2a) = apply_three_cpt_mode_g(&e.modes[0], h_c, h_p1, h_p2, dt);
    let (cb, p1b, p2b) = apply_three_cpt_mode_g(&e.modes[1], h_c, h_p1, h_p2, dt);
    let (cg, p1g, p2g) = apply_three_cpt_mode_g(&e.modes[2], h_c, h_p1, h_p2, dt);

    state[0] = ca + cb + cg + a_ss_c;
    state[1] = p1a + p1b + p1g + a_ss_p1;
    state[2] = p2a + p2b + p2g + a_ss_p2;
}

/// 3-cpt IV propagator computing its eigendata inline (the dual-walk wrapper).
/// Generic mirror of [`crate::pk::event_driven::propagate_three_cpt`].
#[allow(clippy::too_many_arguments)]
pub fn propagate_three_cpt_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
    rate_central: T,
    rate_periph1: T,
    rate_periph2: T,
) {
    if let Some(e) = three_cpt_eigen_g(cl, v1, q2, v2, q3, v3) {
        propagate_three_cpt_core_g(state, dt, &e, rate_central, rate_periph1, rate_periph2);
    }
}

/// 3-cpt oral propagator: `state = [A_depot, A_central, A_p1, A_p2]`. `rate_central`
/// is a constant zero-order input into central (depot-bypass infusion, RATE>0 into
/// cmt 2); `rate_depot` is a constant zero-order input into the depot (RATE>0 into
/// cmt 1, #400). Both are added by linear superposition and are `0` for bolus
/// dosing. Generic mirror of
/// [`crate::pk::event_driven::propagate_three_cpt_oral`].
/// 3-cpt oral propagation from **precomputed** eigendata — the single source of the
/// 3-cpt oral eigenmode + depot forced-response formula.
pub fn propagate_three_cpt_oral_core_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    e: &ThreeCptEigen<T>,
    ka: T,
    rate_central: T,
    rate_periph1: T,
    rate_periph2: T,
    rate_depot: T,
) {
    let (alpha, beta, gamma, k10, k12, k13, k21, k31) =
        (e.alpha, e.beta, e.gamma, e.k10, e.k12, e.k13, e.k21, e.k31);

    let a_d_0 = state[0];
    let a_c_0 = state[1];
    let a_p1_0 = state[2];
    let a_p2_0 = state[3];

    let e_ka = (-(ka * dt)).exp();
    state[0] = a_d_0 * e_ka;

    let denom_depot = (ka - alpha) * (ka - beta) * (ka - gamma);
    let d21 = k21 - ka;
    let d31 = k31 - ka;
    let (cap_a, cap_b, cap_c) = if denom_depot.val().abs() < 1e-12 {
        (T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0))
    } else {
        let scale = -(ka * a_d_0) / denom_depot;
        (scale * d21 * d31, scale * k12 * d31, scale * k13 * d21)
    };

    let h_c = a_c_0 - cap_a;
    let h_p1 = a_p1_0 - cap_b;
    let h_p2 = a_p2_0 - cap_c;

    let (ca, p1a, p2a) = apply_three_cpt_mode_g(&e.modes[0], h_c, h_p1, h_p2, dt);
    let (cb, p1b, p2b) = apply_three_cpt_mode_g(&e.modes[1], h_c, h_p1, h_p2, dt);
    let (cg, p1g, p2g) = apply_three_cpt_mode_g(&e.modes[2], h_c, h_p1, h_p2, dt);

    state[1] = ca + cb + cg + cap_a * e_ka;
    state[2] = p1a + p1b + p1g + cap_b * e_ka;
    state[3] = p2a + p2b + p2g + cap_c * e_ka;

    // Constant infusion into central (depot bypass) and/or into either
    // **peripheral** (#375): forced response of the 3-cpt IV propagator from a
    // zero initial state. The depot takes no inflow from the disposition
    // sub-system, so a rate into central or a peripheral drives exactly the
    // three-state system an IV model has — the oral peripherals reuse the IV
    // closed form rather than needing a new one. `.val()` guards are
    // derivative-safe (each rate is structurally 0 or positive).
    if rate_central.val() > 0.0 || rate_periph1.val() > 0.0 || rate_periph2.val() > 0.0 {
        let mut inflow = [T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0)];
        propagate_three_cpt_core_g(&mut inflow, dt, e, rate_central, rate_periph1, rate_periph2);
        state[1] = state[1] + inflow[0];
        state[2] = state[2] + inflow[1];
        state[3] = state[3] + inflow[2];
    }

    // Constant zero-order input into the depot (#400): forced response from a zero
    // initial state = `x_ss − e^{A·dt}·x_ss`. At steady state the depot holds R/ka
    // and delivers R into central (central_ss = R/k10, p1_ss = k12·R/(k21·k10),
    // p2_ss = k13·R/(k31·k10)); the homogeneous evolution is obtained by recursing
    // once with zero rates (the guard bounds it to depth 1).
    if rate_depot.val() > 0.0 {
        let (c_ss, p1_ss, p2_ss) = if k10.val() > 1e-30 && k21.val() > 1e-30 && k31.val() > 1e-30 {
            (
                rate_depot / k10,
                k12 * rate_depot / (k21 * k10),
                k13 * rate_depot / (k31 * k10),
            )
        } else {
            (T::from_f64(0.0), T::from_f64(0.0), T::from_f64(0.0))
        };
        let d_ss = rate_depot / ka;
        let mut xss = [d_ss, c_ss, p1_ss, p2_ss];
        propagate_three_cpt_oral_core_g(
            &mut xss,
            dt,
            e,
            ka,
            T::from_f64(0.0),
            T::from_f64(0.0),
            T::from_f64(0.0),
            T::from_f64(0.0),
        );
        state[0] = state[0] + (d_ss - xss[0]);
        state[1] = state[1] + (c_ss - xss[1]);
        state[2] = state[2] + (p1_ss - xss[2]);
        state[3] = state[3] + (p2_ss - xss[3]);
    }
}

/// 3-cpt oral propagator computing its eigendata inline (the dual-walk wrapper).
#[allow(clippy::too_many_arguments)]
pub fn propagate_three_cpt_oral_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    cl: T,
    v1: T,
    q2: T,
    v2: T,
    q3: T,
    v3: T,
    ka: T,
    rate_central: T,
    rate_periph1: T,
    rate_periph2: T,
    rate_depot: T,
) {
    if ka.val() <= 0.0 {
        return;
    }
    if let Some(e) = three_cpt_eigen_g(cl, v1, q2, v2, q3, v3) {
        propagate_three_cpt_oral_core_g(
            state,
            dt,
            &e,
            ka,
            rate_central,
            rate_periph1,
            rate_periph2,
            rate_depot,
        );
    }
}

/// Per-walk eigendata memo for the **f64 prediction walk**, keyed on the
/// disposition parameter values. For a constant-parameter walk every interval
/// shares the same params, so the 2-/3-cpt eigenvalue solve (`sqrt` / `acos`)
/// runs once and is reused across all intervals; a time-varying-covariate change
/// is a cache miss that recomputes transparently, so results are bit-identical
/// either way.
///
/// **`f64` only, by design.** The key is the parameter *value*, which uniquely
/// determines the eigendata for plain `f64`. A `Dual2` caller must NOT memoise
/// here: two events can share a primal value while carrying different derivative
/// seeds (IOV occasions, TV covariates), so a value key would return the wrong
/// grad/Hessian. The `Dual2` sensitivity walk recomputes eigendata inline for
/// exactly that reason; the propagators still share one formula via `*_core_g`.
#[derive(Default)]
pub struct EigenCacheG {
    two: Option<([f64; 4], Option<TwoCptEigen<f64>>)>,
    three: Option<([f64; 6], Option<ThreeCptEigen<f64>>)>,
}

impl EigenCacheG {
    /// 2-cpt eigendata for these params, computed once and reused while the param
    /// values are unchanged. `None` for degenerate params.
    pub fn two_cpt(&mut self, cl: f64, v1: f64, q: f64, v2: f64) -> Option<TwoCptEigen<f64>> {
        let key = [cl, v1, q, v2];
        if let Some((k, e)) = self.two {
            if k == key {
                return e;
            }
        }
        let e = two_cpt_eigen_g(cl, v1, q, v2);
        self.two = Some((key, e));
        e
    }

    /// 3-cpt eigendata for these params, computed once and reused while the param
    /// values are unchanged. `None` for degenerate params.
    pub fn three_cpt(
        &mut self,
        cl: f64,
        v1: f64,
        q2: f64,
        v2: f64,
        q3: f64,
        v3: f64,
    ) -> Option<ThreeCptEigen<f64>> {
        let key = [cl, v1, q2, v2, q3, v3];
        if let Some((k, e)) = self.three {
            if k == key {
                return e;
            }
        }
        let e = three_cpt_eigen_g(cl, v1, q2, v2, q3, v3);
        self.three = Some((key, e));
        e
    }
}

/// Per-event PK params for the generic walk, carrying every disposition slot the
/// 1-/2-/3-cpt propagators read. Unused slots for a given model are simply ignored.
#[derive(Clone, Copy)]
pub struct PkDual<T: PkNum> {
    pub cl: T,
    pub v: T,
    pub q: T,
    pub v2: T,
    pub ka: T,
    pub q3: T,
    pub v3: T,
    /// Bioavailability `F` (multiplies bolus amount and infusion rate).
    pub f: T,
}

/// Back-compat 1-cpt view — kept so the focused 1-cpt propagator tests and any
/// 1-cpt caller stay terse. Converts to [`PkDual`] for the shared walk.
#[derive(Clone, Copy)]
pub struct OneCptPk<T: PkNum> {
    pub cl: T,
    pub v: T,
    pub ka: T,
    pub f: T,
}

impl<T: PkNum> OneCptPk<T> {
    fn to_pk_dual(self) -> PkDual<T> {
        PkDual {
            cl: self.cl,
            v: self.v,
            q: T::from_f64(0.0),
            v2: T::from_f64(0.0),
            ka: self.ka,
            q3: T::from_f64(0.0),
            v3: T::from_f64(0.0),
            f: self.f,
        }
    }
}

/// State-vector dimension and central-compartment read-out slot for a `pk_model`.
/// Mirrors `event_driven::state_layout` (kept private there). 3-cpt rows are
/// included for the forthcoming extension; the walk currently propagates 1-/2-cpt.
#[inline]
fn state_layout_g(pk_model: PkModel) -> (usize, usize) {
    pk_model.topology().state_layout()
}

/// True when the generic walk implements this model — all six analytical 1-/2-/3-
/// cpt models. Callers (`subject_sensitivities_iov`) screen earlier; this is the
/// defensive backstop.
#[inline]
fn walk_supports(pk_model: PkModel) -> bool {
    pk_model.topology().event_walk_supported
}

#[inline]
fn pk_for_g<T: PkNum>(
    ev: Event,
    pk_at_dose: &[PkDual<T>],
    pk_at_obs: &[PkDual<T>],
    pk_at_pk_only: &[PkDual<T>],
) -> PkDual<T> {
    match ev.kind {
        EventKind::Dose => pk_at_dose[ev.orig_idx],
        EventKind::Obs => pk_at_obs[ev.orig_idx],
        EventKind::PkOnly => pk_at_pk_only[ev.orig_idx],
        EventKind::Reset => unreachable!("Reset carries no PK params"),
    }
}

/// Propagate the dual state across pre-built sub-event bounds, applying any active
/// infusion per sub-interval (central / peripheral-1, per the production
/// model→cmt map). Generic mirror of `event_driven::propagate_with_bounds` for the
/// 1-/2-cpt models.
///
/// `dose_inf_dual[k] = Some((rate_bare, dur_bare))` marks a **modeled** infusion
/// (`RATE=-1/-2` → `R{cmt}`/`D{cmt}`) whose rate and window resolve from PK params
/// (#486): the rate magnitude carries `∂rate/∂(θ,η)` and the window **end**
/// `t_start + dur_bare` is a *moving boundary*. The walk propagates the active and
/// quiet sub-intervals with the exact **dual** window lengths, so the
/// moving-boundary sensitivity flows through the closed-form flow to all orders —
/// no explicit saltation is needed (the closed-form analogue of the ODE path's
/// `inject_rate_saltation`). `rate_bare` is the F-agnostic rate (`amt/D` or `R`);
/// the walk applies `F` as the existing `pk.f * rate` post-multiply, matching
/// [`crate::types::DoseEvent::bioavailable_infusion`] for the duration-defined case.
/// Empty (or `None` entries) recover the pre-#486 fixed-window behaviour.
#[allow(clippy::too_many_arguments)]
fn propagate_bounds_g<T: PkNum>(
    state: &mut [T],
    bounds: &[f64],
    pk: &PkDual<T>,
    pk_model: PkModel,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_lag_dual: &[T],
    dose_inf_dual: &[Option<(T, T)>],
    reset_floor: f64,
) {
    // Dual clock position of dose `k`'s arrival, `τ_k = t_k + ALAG`. With an estimated
    // lagtime this boundary **moves**: `dose_lag_dual[k]` carries `∂ALAG/∂(θ,η)`, so the
    // sub-intervals on either side of the arrival pick up `±∂ALAG` in their dual lengths
    // and the closed-form flow yields the exact bolus saltation `−A·Φ(t−τ)·b` — the same
    // dual-boundary mechanism the modeled infusion *end* already uses, needing no explicit
    // saltation injection (#486). Empty `dose_lag_dual` ⇒ no lagtime in scope ⇒ a constant.
    let dose_start = |k: usize, d: &DoseEvent| -> (f64, T) {
        let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let t_start = d.time + lag;
        let dual = match dose_lag_dual.get(k) {
            Some(&lag_d) => T::from_f64(d.time) + lag_d,
            None => T::from_f64(t_start),
        };
        (t_start, dual)
    };
    // Every break whose dual position is not simply its f64 value: each post-reset dose
    // contributes its arrival `τ_k`, and an infusing dose additionally contributes its
    // window end `τ_k + D` (the modeled `D`'s jet on top of the arrival's). Doses starting
    // before a reset are skipped exactly as the rate-accumulation loop below skips them
    // (`t_start < reset_floor`), so a boundary coinciding with a reset break does not thread
    // a spurious jet into the post-reset window (#486 review #3).
    //
    // The provider declines to FD any subject whose moving boundaries are not *separable* —
    // two breaks coinciding in `f64` while carrying different jets (`moving_bounds_separable`)
    // — so at most one entry matches a given break and coincidences that survive share an
    // identical dual. The first match below is therefore exact (#486 review #2).
    //
    // Only boundaries that genuinely *move* are recorded: with no lagtime and no modeled
    // window nothing does, so this stays empty and `dual_pos` below is the same no-op scan it
    // was pre-#486 — the ordinary fixed-dose subject (the hot path) pays nothing.
    let has_lag = !dose_lag_dual.is_empty();
    let mut moving_bounds: Vec<(f64, T)> = Vec::new();
    for (k, d) in doses.iter().enumerate() {
        let (t_start, dual_start) = dose_start(k, d);
        if t_start < reset_floor {
            continue;
        }
        // The arrival moves only under a lagtime.
        if has_lag {
            moving_bounds.push((t_start, dual_start));
        }
        if d.rate > 0.0 && d.duration > 0.0 {
            // The window end moves with the arrival *and* (for a modeled `RATE=-1/-2` dose)
            // with the estimated duration. A fixed infusion under a lagtime still has a moving
            // end — it rides the arrival's jet — but a fixed infusion with no lag does not.
            let dual_end = match dose_inf_dual.get(k).copied().flatten() {
                Some((_, dur_bare)) => Some(dual_start + dur_bare),
                None if has_lag => Some(dual_start + T::from_f64(d.duration)),
                None => None,
            };
            if let Some(dual_end) = dual_end {
                moving_bounds.push((t_start + d.duration, dual_end));
            }
        }
    }
    let dual_pos = |w: f64| -> T {
        for &(val, dual) in &moving_bounds {
            if (w - val).abs() < 1e-9 {
                return dual;
            }
        }
        T::from_f64(w)
    };
    for w in bounds.windows(2) {
        if w[1] - w[0] <= 0.0 {
            continue;
        }
        let dt = dual_pos(w[1]) - dual_pos(w[0]);
        let mid = 0.5 * (w[0] + w[1]);
        let rates = active_rates_g(
            doses,
            dose_lagtimes,
            dose_inf_dual,
            pk,
            pk_model,
            reset_floor,
            mid,
            false,
        );
        apply_step_g(state, dt, pk, pk_model, rates);
    }
}

/// The infusion rates (`F·rate`, already summed per compartment) active at clock time `mid`,
/// laid out as `(central, peripheral-1, peripheral-2, depot)` per the production model→cmt map.
/// A **modeled** dose contributes its dual rate (`amt/D` or `R`), so the injected magnitude
/// carries the PK-param jet; a fixed dose contributes its concrete `d.rate`.
///
/// Shared by the interval walk ([`propagate_bounds_g`]) and the observation-on-a-moving-boundary
/// correction in [`event_driven_sens_with_doses_g`], so the two cannot drift apart on which
/// infusions are considered active at a given instant.
///
/// `strict_start` selects the window convention at the query point itself:
/// * `false` — `t_start <= mid && t_end >= mid`, inclusive at both ends, matching production.
///   This is what the interval walk wants, where `mid` is strictly inside a sub-interval.
/// * `true` — `t_start < mid`, i.e. the rates on the segment *immediately before* `mid`. The
///   observation correction wants exactly this: an infusion **ending** at `mid` is still
///   contributing (production's full-containment convention), while one **starting** at `mid`
///   is not yet, since the walk propagated into that instant under the old rate set.
#[allow(clippy::too_many_arguments)]
fn active_rates_g<T: PkNum>(
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_inf_dual: &[Option<(T, T)>],
    pk: &PkDual<T>,
    pk_model: PkModel,
    reset_floor: f64,
    mid: f64,
    strict_start: bool,
) -> (T, T, T, T) {
    let mut rate_central = T::from_f64(0.0);
    let mut rate_periph1 = T::from_f64(0.0);
    let mut rate_periph2 = T::from_f64(0.0);
    let mut rate_depot = T::from_f64(0.0);
    for (k, d) in doses.iter().enumerate() {
        let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let t_start = d.time + lag;
        let t_end = t_start + d.duration;
        if t_start < reset_floor {
            continue;
        }
        let started = if strict_start {
            t_start < mid
        } else {
            t_start <= mid
        };
        if d.rate > 0.0 && d.duration > 0.0 && started && t_end >= mid {
            let r = match dose_inf_dual.get(k).copied().flatten() {
                Some((rate_bare, _)) => pk.f * rate_bare,
                None => pk.f * T::from_f64(d.rate),
            };
            match pk_model.topology().dose_channel(d.cmt) {
                Some(Channel::Central) => rate_central = rate_central + r,
                Some(Channel::Periph1) => rate_periph1 = rate_periph1 + r,
                Some(Channel::Periph2) => rate_periph2 = rate_periph2 + r,
                Some(Channel::Depot) => rate_depot = rate_depot + r,
                // Unreachable from a validated call — `check_dose_compartments`
                // (#375) rejects an unroutable infusion before any prediction. The
                // old silent `{}` was worse than the value walk's `panic!` twin in
                // `pk::event_driven::propagate_with_bounds`: it dropped the
                // infusion from the *gradient* only, so FOCE/FOCEI would have
                // differentiated a different dosing history than it predicted.
                // Fail the same way the value walk does instead.
                None => panic!(
                    "sens PK walk: infusion into compartment {} not supported for model \
                     {:?} — should have been rejected by `check_dose_compartments`",
                    d.cmt, pk_model
                ),
            }
        }
    }
    (rate_central, rate_periph1, rate_periph2, rate_depot)
}

/// Advance the dual state by one (possibly dual-length) step under a constant rate vector.
/// `dt` may carry a jet — that is how a moving boundary's sensitivity enters — and its **value**
/// may be zero, which is exactly the zero-length sliver the observation correction uses.
fn apply_step_g<T: PkNum>(
    state: &mut [T],
    dt: T,
    pk: &PkDual<T>,
    pk_model: PkModel,
    rates: (T, T, T, T),
) {
    let (rate_central, rate_periph1, rate_periph2, rate_depot) = rates;
    {
        match pk_model {
            // See event_driven::propagate — transit / IG cannot be state-propagated; the
            // Dual2 superposition path serves their sensitivities instead (#386/#790).
            PkModel::OneCptTransit => unreachable!(
                "one_cpt_transit uses closed-form superposition, not the event-driven walk"
            ),
            PkModel::TwoCptTransit => unreachable!(
                "two_cpt_transit uses closed-form superposition, not the event-driven walk"
            ),
            PkModel::OneCptIg => {
                unreachable!("one_cpt_ig uses closed-form superposition, not the event-driven walk")
            }
            PkModel::TwoCptIg => {
                unreachable!("two_cpt_ig uses closed-form superposition, not the event-driven walk")
            }
            PkModel::OneCptIv => propagate_one_cpt_g(state, dt, pk.cl, pk.v, rate_central),
            PkModel::OneCptOral => {
                propagate_one_cpt_oral_g(state, dt, pk.cl, pk.v, pk.ka, rate_central, rate_depot)
            }
            PkModel::TwoCptIv => propagate_two_cpt_g(
                state,
                dt,
                pk.cl,
                pk.v,
                pk.q,
                pk.v2,
                rate_central,
                rate_periph1,
            ),
            PkModel::TwoCptOral => propagate_two_cpt_oral_g(
                state,
                dt,
                pk.cl,
                pk.v,
                pk.q,
                pk.v2,
                pk.ka,
                rate_central,
                rate_periph1,
                rate_depot,
            ),
            PkModel::ThreeCptIv => propagate_three_cpt_g(
                state,
                dt,
                pk.cl,
                pk.v,
                pk.q,
                pk.v2,
                pk.q3,
                pk.v3,
                rate_central,
                rate_periph1,
                rate_periph2,
            ),
            PkModel::ThreeCptOral => propagate_three_cpt_oral_g(
                state,
                dt,
                pk.cl,
                pk.v,
                pk.q,
                pk.v2,
                pk.q3,
                pk.v3,
                pk.ka,
                rate_central,
                rate_periph1,
                rate_periph2,
                rate_depot,
            ),
        }
    }
}

/// The correction an observation needs when its clock time lands exactly on a **moving
/// boundary** — a dose arrival `t_dose + ALAG`, or an infusion window end `t_dose + ALAG + D`.
///
/// Returns `Δ = boundary(p) − t` and the window convention to evaluate the sliver's rates under.
/// `Δ` has **value zero** — boundary and sample coincide at the current parameters — but a live
/// *jet*, because the boundary moves with the estimated lagtime and/or `D{cmt}`/`R{cmt}`. That jet
/// is exactly the along-the-boundary term the observation must not inherit: its own clock time
/// does not move.
///
/// The caller steps the state back over `−Δ`, i.e. it re-evolves it across the sliver that lies
/// between the sample and the boundary. Which rates act on that sliver depends on which side of
/// the boundary it is on, and that is the returned `strict_start` flag:
///
/// * **Arrival** (`strict_start = false`). The branch that defines the value here is the one where
///   the dose *has landed* (production orders `Dose` before `Obs`), so the sliver lies **after**
///   the arrival: the dose is in the state and, if it is an infusion, that infusion is running.
///   Inclusive rates. The deposited bolus stays in the state — it has had `t_obs − τ` to act.
/// * **Infusion end** (`strict_start = true`). The sliver lies **before** the end, inside the
///   window, so the ending infusion is still contributing (production's full-containment
///   convention) while an infusion merely *starting* at this instant is not.
///
/// `None` (the overwhelmingly common case) when the sample sits on no moving boundary, or when
/// nothing about the coincident boundary actually moves (a fixed window with no lagtime). The
/// caller declines subjects where *several* moving boundaries coincide with one observation, so at
/// most one applies here.
fn obs_boundary_correction<T: PkNum>(
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_lag_dual: &[T],
    dose_inf_dual: &[Option<(T, T)>],
    reset_floor: f64,
    t: f64,
) -> Option<(T, bool)> {
    let has_lag = !dose_lag_dual.is_empty();
    for (k, d) in doses.iter().enumerate() {
        let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let t_start = d.time + lag;
        if t_start < reset_floor {
            continue;
        }
        let dual_start = match dose_lag_dual.get(k) {
            Some(&lag_d) => T::from_f64(d.time) + lag_d,
            None => T::from_f64(t_start),
        };

        // (a) The sample sits on this dose's *arrival*. Only moving under a lagtime.
        if has_lag && (t_start - t).abs() < 1e-9 {
            return Some((dual_start - T::from_f64(t), false));
        }

        // (b) The sample sits on this dose's infusion *end*.
        if d.rate > 0.0 && d.duration > 0.0 && ((t_start + d.duration) - t).abs() < 1e-9 {
            let dual_end = match dose_inf_dual.get(k).copied().flatten() {
                Some((_, dur_bare)) => dual_start + dur_bare,
                None if has_lag => dual_start + T::from_f64(d.duration),
                // A fixed window with no lagtime: the end does not move, nothing to correct.
                None => continue,
            };
            return Some((dual_end - T::from_f64(t), true));
        }
    }
    None
}

/// Shared early-stop driver for the **dual** SS-equilibration loops (#519; #532 review #9/#10):
/// refresh `cur` from the value parts (`PkNum::val`) of the dual state `u`, decide the stop on
/// the *same* [`crate::dosing::ss_cycle_converged`] criterion the f64 predictor uses
/// (from cycle 1 on), and roll `cur`→`prev` by swap — no per-cycle `O(n_states)` copy. Returns
/// `true` to break. Used by the ODE provider's `equilibrate_ss_state_g` and by
/// `equilibrate_ss_g`'s singular-`I − M` fallback (here) so the dual convergence logic lives in
/// **one** place. Since #908 the analytical walk reaches this only when no periodic steady state
/// exists; its ordinary path is the exact `(I − M)⁻¹·b` solve, which has no cycles to stop.
///
/// **Why decide on the value parts, not the derivative tails** (#532 review #2/#3): the goal is
/// not a fully-converged sensitivity — it is a gradient *consistent with the value the optimizer
/// is handed*. Stopping on the value parts makes the dual break on the *same cycle* the f64
/// predictor would, so the returned dual state is exactly the AD derivative of the truncated f64
/// value: `grad = ∂(truncated value)/∂θ`, with no value/gradient mismatch to stall the outer
/// line search. The variational recursion shares the value iteration's monodromy, so the
/// derivative tails contract at the same geometric rate — but they lag the value by a constant
/// few cycles, so a small tail (~`1e-8` on a stressed scale-separated model, see
/// `ode_provider_ss_early_stop_matches_full_budget`) survives the value stop. That residual is
/// 3–4 orders below the `1e-3` gradient-validation tolerance and NONMEM's SE-matching precision,
/// so the surviving gradient is faithful for every reported purpose; chasing it to zero would
/// add a constant per-evaluation cost on the gradient hot path for no observable benefit. A
/// ULP-level disagreement between the separate f64 and dual value arithmetics can at most shift
/// the stop by ±1 cycle, i.e. by the converged increment (≤ `1e-12` relative).
pub(crate) fn ss_dual_cycle_should_stop<T: PkNum>(
    cycle: usize,
    u: &[T],
    cur: &mut Vec<f64>,
    prev: &mut Vec<f64>,
) -> bool {
    for (c, x) in cur.iter_mut().zip(u) {
        *c = x.val();
    }
    if cycle > 0
        && crate::dosing::ss_cycle_converged(cur, prev, crate::dosing::SS_EQUILIBRATION_TOL)
    {
        return true;
    }
    std::mem::swap(cur, prev);
    false
}

/// Equilibrate the dual state to its SS value for an SS=1 dose, per-event (uses
/// `pk`, the dose-event's params). Generic mirror of
/// `event_driven::equilibrate_ss_state_event_driven` for the 1-/2-cpt models;
/// overlapping SS infusions (`T_inf > II`) return the empty state, matching
/// production's reject.
///
/// **Exact, not truncated (#908)** — and it must flip in lockstep with the value walk. Solving
/// `u_ss = (I − M)⁻¹·b` over a dual `T` yields the *implicit-function* derivative
/// `∂u_ss/∂(θ,η) = (I − M)⁻¹·(∂b − ∂M·u_ss)` straight out of
/// [`crate::sens::linsolve::solve_linear_system_g`], with no hand-assembled `dM`/`db`. That is a
/// strictly better object than what the pulse train produced: the derivative of the *truncated*
/// recursion (see [`ss_dual_cycle_should_stop`], which exists to keep those two truncations
/// aligned). Were only one of the two walks converted, FOCE/FOCEI would differentiate a steady
/// state it never predicted — the exact value/gradient mismatch that stalls the outer line search.
///
/// The **window duals** (`inf`, a modeled `RATE=-1/-2` dose's `(rate_bare, dur_bare)`) belong to
/// the *forced* half only. `M` is the homogeneous monodromy: it carries `∂P/∂(θ,η)` through the
/// dual disposition params, but the infusion window is inhomogeneous forcing and enters `b`
/// alone. Threading it into the unforced call would corrupt the propagator.
fn equilibrate_ss_g<T: PkNum>(
    pk_model: PkModel,
    pk: &PkDual<T>,
    dose: &DoseEvent,
    inf: Option<(T, T)>,
) -> Vec<T> {
    let (n_states, _) = state_layout_g(pk_model);
    let mut state = vec![T::from_f64(0.0); n_states];
    if dose.ii <= 0.0 {
        return state;
    }
    // `CMT=0` = NONMEM's default dose compartment — resolved by
    // `saturating_sub(1)` like every other dose site. Mirrors the value walk's
    // `equilibrate_ss_state_event_driven`, whose identical `cmt == 0` bail
    // silently dropped an `SS=1` dose's accumulation (#375); the gradient must
    // equilibrate the same state the value does.
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= n_states {
        return state;
    }
    let is_inf = dose.rate > 0.0 && dose.duration > 0.0 && dose.duration.is_finite();
    if is_inf && dose.duration > dose.ii {
        return state;
    }
    let synthetic_dose = if is_inf {
        vec![DoseEvent::new(
            0.0, dose.amt, dose.cmt, dose.rate, false, 0.0,
        )]
    } else {
        Vec::new()
    };
    let synthetic_lag: Vec<f64> = if is_inf { vec![0.0] } else { Vec::new() };
    // #486: for a **modeled** SS infusion (`RATE=-1/-2`), thread the window dual
    // `(rate_bare, dur_bare)` so each equilibration cycle's active/quiet split carries the
    // moving infusion-end jet (`∂D`/`∂R`) through `propagate_bounds_g`'s `dual_pos`, exactly
    // as the main walk does for the current pulse — closing the closed-form SS × modeled-dose
    // gap. `None` (a fixed-window infusion) recovers the pre-#486 f64 window behaviour.
    let synthetic_inf: Vec<Option<(T, T)>> = if is_inf { vec![inf] } else { Vec::new() };
    let bounds: Vec<f64> = if is_inf {
        vec![0.0, dose.duration, dose.ii]
    } else {
        vec![0.0, dose.ii]
    };
    // One cycle from `u0`. `forced` applies this cycle's own dose — the bolus jump, or the
    // synthetic infusion window together with its modeled-window duals; unforced is the bare
    // homogeneous disposition over the *same* bounds, whose columns are the monodromy `M`.
    //
    // The SS equilibration runs in the dose's own periodic frame (the synthetic pulse sits at
    // t = 0), so its arrival is not a moving boundary: no lag duals. A subject that pairs SS with
    // a lagtime declines to FD upstream (`ss_lagtime_walk_unsupported`).
    let advance = |u0: &[T], forced: bool| -> Option<Vec<T>> {
        let mut s = u0.to_vec();
        if forced && !is_inf {
            s[cmt_idx] = s[cmt_idx] + pk.f * T::from_f64(dose.amt);
        }
        let (doses, lag, inf_dual): (&[DoseEvent], &[f64], &[Option<(T, T)>]) = if forced {
            (&synthetic_dose, &synthetic_lag, &synthetic_inf)
        } else {
            (&[], &[], &[])
        };
        propagate_bounds_g(
            &mut s,
            &bounds,
            pk,
            pk_model,
            doses,
            lag,
            &[],
            inf_dual,
            f64::NEG_INFINITY,
        );
        Some(s)
    };

    // Exact periodic steady state, `reltol = abstol = 0` — see the value walk for why.
    if let Some(u_ss) = crate::dosing::periodic_ss_fixed_point_g::<T, _, _>(
        n_states,
        dose.ii,
        0.0,
        0.0,
        |u0| advance(u0, false),
        |u0| advance(u0, true),
    ) {
        crate::dosing::record_ss_equilibration_cycles(1);
        return u_ss;
    }

    // Fallback for a singular `I − M` (no periodic steady state exists), mirroring the value
    // walk's. Shared early stop (#519, #532 review #11): the closed-form propagator equilibrates
    // the same geometric train, so the same mixed atol/rtol stop applies here too.
    let mut prev = vec![0.0_f64; n_states];
    let mut cur = vec![0.0_f64; n_states];
    let mut cycles_run = 0usize;
    for cycle in 0..crate::dosing::SS_EQUILIBRATION_CYCLES {
        if !is_inf {
            state[cmt_idx] = state[cmt_idx] + pk.f * T::from_f64(dose.amt);
        }
        propagate_bounds_g(
            &mut state,
            &bounds,
            pk,
            pk_model,
            &synthetic_dose,
            &synthetic_lag,
            &[],
            &synthetic_inf,
            f64::NEG_INFINITY,
        );
        cycles_run = cycle + 1;
        if ss_dual_cycle_should_stop(cycle, &state, &mut cur, &mut prev) {
            break;
        }
    }
    crate::dosing::record_ss_equilibration_cycles(cycles_run);
    state
}

/// Event-driven **sensitivity** walk for the 1-/2-cpt models: returns the dual
/// concentration at every observation, parallel to `subject.obs_times`. The
/// `Dual2`-differentiable mirror of
/// `event_driven::event_driven_predictions_with_schedule_impl`.
///
/// `pk_at_dose` / `pk_at_obs` / `pk_at_pk_only` are the **per-event** PK params,
/// already seeded as `T` (parallel to `subject.doses` / `obs_times` /
/// `pk_only_times`). The walk carries dual amounts across boundaries and switches
/// to each event's params, so IOV / time-varying covariates are exact; SS doses
/// equilibrate per-event. Resets (EVID 3/4) zero the dual state; infusions are
/// applied through the bounds.
///
/// The `f64` instantiation reproduces the production walk bit-for-bit (one source
/// of truth); the `Dual2` instantiation yields exact `∂(conc)/∂(seeded axes)` and
/// second order.
pub fn event_driven_sens_g<T: PkNum>(
    pk_model: PkModel,
    subject: &Subject,
    schedule: &EventSchedule,
    pk_at_dose: &[PkDual<T>],
    pk_at_obs: &[PkDual<T>],
    pk_at_pk_only: &[PkDual<T>],
) -> Vec<T> {
    // Fixed-dose entrypoint: the walk reads `subject.doses` directly and carries no
    // modeled-window duals. Modeled-`RATE` subjects use
    // [`event_driven_sens_with_doses_g`] with resolved `eff_doses` + per-dose duals.
    event_driven_sens_with_doses_g(
        pk_model,
        subject,
        schedule,
        &subject.doses,
        &[],
        &[],
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
    )
}

/// As [`event_driven_sens_g`] but with **resolved** `eff_doses` (concrete
/// `rate`/`duration`) and per-dose modeled-infusion duals `dose_inf_dual` (#486).
/// The fixed path passes `&subject.doses` + `&[]`; the modeled-`RATE=-1/-2` path
/// resolves each dose to `f64` (so the schedule's break times and the active-window
/// checks are correct) and supplies `Some((rate_bare, dur_bare))` per modeled dose
/// so the rate magnitude and the moving infusion-end boundary carry their PK-param
/// jets. `eff_doses` / `dose_inf_dual` / `dose_lag_dual` are parallel to `subject.doses`
/// (`dose_inf_dual` may be empty, treated as all-`None`).
///
/// `dose_lag_dual[k]` is dose `k`'s lagtime as a dual (#486). A model with an estimated
/// `ALAG` makes each dose *arrival* a moving boundary, exactly as `D`/`R` makes the
/// infusion *end* one; the walk threads both through the dual sub-interval lengths.
/// Empty ⇒ the model carries no lagtime and every arrival is a fixed break.
#[allow(clippy::too_many_arguments)]
pub fn event_driven_sens_with_doses_g<T: PkNum>(
    pk_model: PkModel,
    subject: &Subject,
    schedule: &EventSchedule,
    eff_doses: &[DoseEvent],
    dose_lag_dual: &[T],
    dose_inf_dual: &[Option<(T, T)>],
    pk_at_dose: &[PkDual<T>],
    pk_at_obs: &[PkDual<T>],
    pk_at_pk_only: &[PkDual<T>],
) -> Vec<T> {
    let n_obs = subject.obs_times.len();
    let mut preds = vec![T::from_f64(0.0); n_obs];
    if n_obs == 0 || schedule.events.is_empty() || !walk_supports(pk_model) {
        return preds;
    }
    let (n_states, central_slot) = state_layout_g(pk_model);

    let mut state = vec![T::from_f64(0.0); n_states];
    let mut cur_t = schedule.events[0].time;
    let mut reset_floor = f64::NEG_INFINITY;

    for (i, ev) in schedule.events.iter().enumerate() {
        if ev.kind == EventKind::Reset {
            state.iter_mut().for_each(|s| *s = T::from_f64(0.0));
            cur_t = ev.time;
            reset_floor = ev.time;
            continue;
        }
        let pk_now = pk_for_g(*ev, pk_at_dose, pk_at_obs, pk_at_pk_only);

        if ev.time > cur_t {
            let bounds = &schedule.bounds_per_interval[i - 1];
            propagate_bounds_g(
                &mut state,
                bounds,
                &pk_now,
                pk_model,
                eff_doses,
                &schedule.dose_lagtimes,
                dose_lag_dual,
                dose_inf_dual,
                reset_floor,
            );
            cur_t = ev.time;
        }

        match ev.kind {
            EventKind::Dose => {
                let d = &eff_doses[ev.orig_idx];
                if d.ss && d.ii > 0.0 {
                    // #486: forward this dose's modeled-window dual (if any) so the SS
                    // equilibration threads `∂D`/`∂R` into the trough, matching the current
                    // pulse handled by the main walk. `None` for a fixed infusion / bolus.
                    let inf = dose_inf_dual.get(ev.orig_idx).copied().flatten();
                    state = equilibrate_ss_g(pk_model, &pk_now, d, inf);
                }
                if d.rate <= 0.0 {
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n_states {
                        state[cmt_idx] = state[cmt_idx] + pk_now.f * T::from_f64(d.amt);
                    } else {
                        // Unreachable from a validated call — `check_dose_compartments`
                        // (#375) rejects `cmt > n_states` up front. Kept as a defensive
                        // guard that matches the value walk's twin in
                        // `pk::event_driven`: silently dropping the dose here would
                        // differentiate a different dosing history than was predicted.
                        panic!(
                            "sens PK walk: dose into compartment {} but model has {} states \
                             (cmt is 1-based) — should have been rejected by \
                             `check_dose_compartments`",
                            d.cmt, n_states
                        );
                    }
                }
            }
            EventKind::Obs => {
                // **Observation sitting exactly on a moving boundary** (#486) — a dose arrival
                // `t + ALAG`, or an infusion end `t + ALAG + D`.
                //
                // The walk propagates *through* such a boundary, so `state` here is the state at
                // the boundary, `x(bound(p))` — which is what every later event needs. But an
                // observation's clock time does **not** move with the parameters. Reading it here
                // differentiates along the boundary and adds a spurious `ẋ·∂bound/∂p`, giving a
                // value that is not even a subgradient in general.
                //
                // Correct it by stepping a **copy** of the state back across the zero-length
                // sliver `Δ = bound(p) − t_obs`, under the rate set of the segment immediately
                // *before* the sample (`strict_start`). `Δ` has value 0 but a live jet, so the
                // step leaves the prediction's *value* untouched and removes exactly the
                // along-the-boundary term from its derivatives — to second order, since the dual
                // carries it. The main walk's state is untouched, so later observations keep the
                // boundary sensitivity they need.
                //
                // What comes out is the **one-sided** derivative of the branch that production's
                // own event ordering uses to define the value here: `Dose` before `Obs` (the dose
                // has landed), and the infusion-end break *after* `Obs` (the infusion is still
                // contributing). That is the same convention — and the same number — the ODE
                // engine's jump/saltation sensitivities produce, which is what the ODE-twin
                // equivalence tests pin. A bolus deposited at this very instant is lifted off
                // before the step and put back after: it does not decay across a zero-length
                // window.
                let mut obs_state;
                let read = match obs_boundary_correction(
                    eff_doses,
                    &schedule.dose_lagtimes,
                    dose_lag_dual,
                    dose_inf_dual,
                    reset_floor,
                    ev.time,
                ) {
                    Some((delta, strict_start)) => {
                        obs_state = state.clone();
                        let rates = active_rates_g(
                            eff_doses,
                            &schedule.dose_lagtimes,
                            dose_inf_dual,
                            &pk_now,
                            pk_model,
                            reset_floor,
                            ev.time,
                            strict_start,
                        );
                        apply_step_g(
                            &mut obs_state,
                            T::from_f64(0.0) - delta,
                            &pk_now,
                            pk_model,
                            rates,
                        );
                        &obs_state[..]
                    }
                    None => &state[..],
                };
                let v = pk_now.v;
                let conc = if v.val() > 0.0 {
                    read[central_slot] / v
                } else {
                    T::from_f64(0.0)
                };
                // Mirror production's `conc.max(0.0)`: a negative value clamps to
                // 0, so its derivatives vanish there (consistency with the OFV).
                preds[ev.orig_idx] = if conc.val() < 0.0 {
                    T::from_f64(0.0)
                } else {
                    conc
                };
            }
            EventKind::PkOnly => {}
            EventKind::Reset => unreachable!("Reset handled before pk_for_g above"),
        }
    }

    preds
}

/// 1-cpt convenience wrapper around [`event_driven_sens_g`] (maps `oral` to the
/// `PkModel` and lifts [`OneCptPk`] to [`PkDual`]). Kept for the focused 1-cpt
/// tests and terse 1-cpt callers.
pub fn event_driven_sens_one_cpt_g<T: PkNum>(
    oral: bool,
    subject: &Subject,
    schedule: &EventSchedule,
    pk_at_dose: &[OneCptPk<T>],
    pk_at_obs: &[OneCptPk<T>],
    pk_at_pk_only: &[OneCptPk<T>],
) -> Vec<T> {
    let pk_model = if oral {
        PkModel::OneCptOral
    } else {
        PkModel::OneCptIv
    };
    let dose: Vec<PkDual<T>> = pk_at_dose.iter().map(|p| p.to_pk_dual()).collect();
    let obs: Vec<PkDual<T>> = pk_at_obs.iter().map(|p| p.to_pk_dual()).collect();
    let only: Vec<PkDual<T>> = pk_at_pk_only.iter().map(|p| p.to_pk_dual()).collect();
    event_driven_sens_g(pk_model, subject, schedule, &dose, &obs, &only)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pk::event_driven::event_driven_predictions;
    use crate::sens::dual2::Dual2;
    use crate::types::{PkModel, PkParams, PK_IDX_CL, PK_IDX_KA, PK_IDX_V};

    fn pk_of(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v;
        p.values[PK_IDX_KA] = ka;
        p
    }

    /// The `f64` instantiation of the generic propagator must reproduce the
    /// production `f64` propagator bit-for-bit (same closed form, one source of
    /// truth in disguise).
    #[test]
    fn one_cpt_g_matches_production_f64() {
        for &(cl, v, rate, dt, s0) in &[
            (3.0, 30.0, 0.0, 2.5, 10.0),
            (5.0, 40.0, 8.0, 1.0, 0.0),
            (1.2, 12.0, 2.0, 6.0, 3.5),
        ] {
            let mut s_g = [s0];
            propagate_one_cpt_g::<f64>(&mut s_g, dt, cl, v, rate);
            let mut s_p = [s0];
            crate::pk::event_driven::propagate_one_cpt(&mut s_p, dt, &pk_of(cl, v, 1.0), rate);
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
        }
    }

    #[test]
    fn one_cpt_oral_g_matches_production_f64() {
        for &(cl, v, ka, dt, ad, ac) in &[
            (3.0, 30.0, 1.5, 2.5, 50.0, 5.0),
            (1.2, 12.0, 0.8, 4.0, 20.0, 1.0),
            // ka ≈ ke (L'Hôpital): ke = cl/v = 0.1, ka = 0.1.
            (3.0, 30.0, 0.1, 3.0, 40.0, 2.0),
        ] {
            let mut s_g = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s_g, dt, cl, v, ka, 0.0, 0.0);
            let mut s_p = [ad, ac];
            crate::pk::event_driven::propagate_one_cpt_oral(
                &mut s_p,
                dt,
                &pk_of(cl, v, ka),
                0.0,
                0.0,
            );
            approx::assert_relative_eq!(s_g[0], s_p[0], max_relative = 1e-12);
            approx::assert_relative_eq!(s_g[1], s_p[1], max_relative = 1e-12);
        }
    }

    /// Central FD grad + 4-point Hessian of a 2-arg `f64` closure.
    fn fd2(p: [f64; 2], val: impl Fn([f64; 2]) -> f64) -> ([f64; 2], [[f64; 2]; 2]) {
        let h = [1e-6 * (1.0 + p[0].abs()), 1e-6 * (1.0 + p[1].abs())];
        let hh = [1e-4 * (1.0 + p[0].abs()), 1e-4 * (1.0 + p[1].abs())];
        let mut g = [0.0; 2];
        for i in 0..2 {
            let mut up = p;
            up[i] += h[i];
            let mut dn = p;
            dn[i] -= h[i];
            g[i] = (val(up) - val(dn)) / (2.0 * h[i]);
        }
        let mut he = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut pp = p;
                pp[i] += hh[i];
                pp[j] += hh[j];
                let mut pm = p;
                pm[i] += hh[i];
                pm[j] -= hh[j];
                let mut mp = p;
                mp[i] -= hh[i];
                mp[j] += hh[j];
                let mut mm = p;
                mm[i] -= hh[i];
                mm[j] -= hh[j];
                he[i][j] = (val(pp) - val(pm) - val(mp) + val(mm)) / (4.0 * hh[i] * hh[j]);
            }
        }
        (g, he)
    }

    /// The `Dual2` instantiation's `∂(amount)/∂(cl,v)` (and Hessian) must match
    /// finite differences of the `f64` propagator — the propagator differentiates
    /// the compartment amount w.r.t. the PK parameters exactly.
    #[test]
    fn one_cpt_g_dual_matches_fd() {
        let (cl, v, rate, dt, s0) = (3.0, 30.0, 4.0, 2.5, 10.0);
        let mut sd = [Dual2::<2>::constant(s0)];
        propagate_one_cpt_g::<Dual2<2>>(
            &mut sd,
            Dual2::<2>::from_f64(dt),
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(rate),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [s0];
            propagate_one_cpt_g::<f64>(&mut s, dt, p[0], p[1], rate);
            s[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[0].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[0].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }

    // ── Full 1-cpt event-driven sensitivity walk ─────────────────────

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        use std::collections::HashMap;
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
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
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    fn pk_full(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = pk_of(cl, v, ka);
        p.values[crate::types::PK_IDX_F] = 1.0;
        p
    }

    fn one_cpt_pk_f64(p: &PkParams) -> OneCptPk<f64> {
        OneCptPk {
            cl: p.cl(),
            v: p.v(),
            ka: p.ka(),
            f: p.f_bio(),
        }
    }

    /// #375: `active_rates_g` used to answer an unroutable infusion with a silent
    /// `None => {}`, dropping the dose from the **gradient** while the production
    /// value walk (`pk::event_driven::propagate_with_bounds`) panicked on the same
    /// input — so had the value path been made lenient, FOCE/FOCEI would have
    /// differentiated a different dosing history than it predicted. Both walks now
    /// fail the same way; `check_dose_compartments` makes this unreachable from a
    /// validated call, and this test is the direct-call proof that the guard is
    /// still there.
    #[test]
    #[should_panic(expected = "infusion into compartment 2 not supported")]
    fn sens_walk_panics_on_an_unroutable_infusion_like_the_value_walk() {
        // `one_cpt_iv` has a single compartment, so cmt 2 has no rate channel.
        let dose = DoseEvent::new(0.0, 100.0, 2, 20.0, false, 0.0);
        let subject = make_subject(vec![dose], vec![1.0, 4.0]);
        let pk = pk_full(5.0, 50.0, 1.0);
        let schedule = EventSchedule::for_subject(
            &subject,
            PkModel::OneCptIv,
            &subject.doses,
            &[pk.lagtime()],
        );
        let pk_g = one_cpt_pk_f64(&pk);
        let _ = event_driven_sens_one_cpt_g::<f64>(
            false,
            &subject,
            &schedule,
            &[pk_g],
            &vec![pk_g; 2],
            &[],
        );
    }

    /// The `f64` instantiation of the full event walk must reproduce the
    /// production event-driven predictions bit-for-bit across dose kinds — bolus,
    /// infusion, oral, and steady state — so the IOV/TV-cov sensitivity walk is the
    /// same closed form as the f64 predictor, only differentiable.
    #[test]
    fn event_walk_g_matches_production_f64() {
        struct Case {
            oral: bool,
            model: PkModel,
            dose: DoseEvent,
            cl: f64,
            v: f64,
            ka: f64,
        }
        let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0];
        let cases = [
            // 1-cpt IV bolus.
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                cl: 5.0,
                v: 50.0,
                ka: 1.0,
            },
            // 1-cpt IV infusion (rate=25 → 4 h).
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0),
                cl: 3.0,
                v: 40.0,
                ka: 1.0,
            },
            // 1-cpt oral bolus into depot (cmt 1).
            Case {
                oral: true,
                model: PkModel::OneCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                cl: 4.0,
                v: 30.0,
                ka: 1.2,
            },
            // 1-cpt IV bolus at steady state (II=12).
            Case {
                oral: false,
                model: PkModel::OneCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
                cl: 5.0,
                v: 50.0,
                ka: 1.0,
            },
            // 1-cpt oral bolus at steady state (II=24).
            Case {
                oral: true,
                model: PkModel::OneCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
                cl: 4.0,
                v: 30.0,
                ka: 1.2,
            },
        ];

        for (ci, c) in cases.iter().enumerate() {
            let subj = make_subject(vec![c.dose.clone()], obs.clone());
            let pk = pk_full(c.cl, c.v, c.ka);
            let prod = event_driven_predictions(c.model, &subj, &[pk], &vec![pk; obs.len()], &[]);
            let schedule = EventSchedule::for_subject(&subj, c.model, &subj.doses, &[pk.lagtime()]);
            let pk_g = one_cpt_pk_f64(&pk);
            let walk = event_driven_sens_one_cpt_g::<f64>(
                c.oral,
                &subj,
                &schedule,
                &[pk_g],
                &vec![pk_g; obs.len()],
                &[],
            );
            for (j, (&p, &w)) in prod.iter().zip(walk.iter()).enumerate() {
                approx::assert_relative_eq!(w, p, max_relative = 1e-12, epsilon = 1e-12);
                assert!(p >= 0.0, "case {ci} obs {j}: production conc negative");
            }
        }
    }

    fn pk_2cpt(cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = 1.0;
        p
    }

    fn pk_dual_2cpt_f64(p: &PkParams) -> PkDual<f64> {
        PkDual {
            cl: p.cl(),
            v: p.v(),
            q: p.q(),
            v2: p.v2(),
            ka: p.ka(),
            q3: 0.0,
            v3: 0.0,
            f: p.f_bio(),
        }
    }

    /// The generic 2-cpt walk (`event_driven_sens_g`) at `f64` must reproduce the
    /// production event-driven predictions across IV bolus, IV infusion, and oral —
    /// confirming the eigen-decomposition propagators match production bit-for-bit.
    #[test]
    fn event_walk_g_2cpt_matches_production_f64() {
        struct Case {
            model: PkModel,
            dose: DoseEvent,
        }
        let obs = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let (cl, v1, q, v2, ka) = (3.0, 30.0, 1.5, 40.0, 1.2);
        let cases = [
            Case {
                model: PkModel::TwoCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            },
            Case {
                model: PkModel::TwoCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0), // 5 h infusion
            },
            Case {
                model: PkModel::TwoCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            },
        ];
        for (ci, c) in cases.iter().enumerate() {
            let subj = make_subject(vec![c.dose.clone()], obs.clone());
            let pk = pk_2cpt(cl, v1, q, v2, ka);
            let prod = event_driven_predictions(c.model, &subj, &[pk], &vec![pk; obs.len()], &[]);
            let schedule = EventSchedule::for_subject(&subj, c.model, &subj.doses, &[0.0]);
            let pkd = pk_dual_2cpt_f64(&pk);
            let walk = event_driven_sens_g::<f64>(
                c.model,
                &subj,
                &schedule,
                &[pkd],
                &vec![pkd; obs.len()],
                &[],
            );
            for (j, (&p, &w)) in prod.iter().zip(walk.iter()).enumerate() {
                approx::assert_relative_eq!(w, p, max_relative = 1e-12, epsilon = 1e-12);
                assert!(p >= 0.0, "case {ci} obs {j}: production conc negative");
            }
        }
    }

    /// **#908 dual fallback.** The gradient twin of
    /// `pk::event_driven::tests::ss_equilibration_falls_back_when_no_steady_state_exists`. A
    /// singular `I − M` — here `CL = 0`, so every disposition eigenvalue is zero and
    /// `two_cpt_eigen_g` declines, leaving `M = I` — has no periodic steady state, so
    /// `equilibrate_ss_g`'s exact solve must decline and its capped pulse-train fallback must
    /// run. This proves the *dual* fallback still functions on genuinely-singular input; the
    /// mutation check (forcing the fallback on non-singular input) only proves the two paths
    /// disagree there, not that the fallback is correct where it is actually meant to fire.
    #[test]
    fn ss_equilibration_dual_falls_back_when_no_steady_state_exists() {
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
        // CL = 0 ⇒ ke = 0 ⇒ `two_cpt_eigen_g` returns None ⇒ propagation is the identity.
        let pkd = pk_dual_2cpt_f64(&pk_2cpt(0.0, 50.0, 3.0, 80.0, 0.0));
        let trough = equilibrate_ss_g::<f64>(PkModel::TwoCptIv, &pkd, &dose, None);
        assert_eq!(
            crate::dosing::last_ss_equilibration_cycles(),
            crate::dosing::SS_EQUILIBRATION_CYCLES,
            "a singular I − M must decline the exact solve and run the full capped train"
        );
        // Nothing is ever eliminated, so the train is a plain sum of the per-cycle boluses.
        approx::assert_relative_eq!(
            trough[0],
            crate::dosing::SS_EQUILIBRATION_CYCLES as f64 * 100.0,
            max_relative = 1e-12
        );
    }

    /// The 2-cpt walk's `Dual2` grad/Hessian (w.r.t. cl, v1) must match FD of the
    /// `f64` walk — the eigen-decomposition propagator differentiates exactly.
    #[test]
    fn event_walk_g_2cpt_dual_matches_fd() {
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![4.0];
        let subj = make_subject(vec![dose], obs.clone());
        let (cl, v1, q, v2) = (3.0, 30.0, 1.5, 40.0);
        let schedule = EventSchedule::for_subject(&subj, PkModel::TwoCptIv, &subj.doses, &[0.0]);
        let seed = |cl: Dual2<2>, v1: Dual2<2>| PkDual {
            cl,
            v: v1,
            q: Dual2::<2>::constant(q),
            v2: Dual2::<2>::constant(v2),
            ka: Dual2::<2>::constant(0.0),
            q3: Dual2::<2>::constant(0.0),
            v3: Dual2::<2>::constant(0.0),
            f: Dual2::<2>::constant(1.0),
        };
        let pk_d = seed(Dual2::var(cl, 0), Dual2::var(v1, 1));
        let walk = event_driven_sens_g::<Dual2<2>>(
            PkModel::TwoCptIv,
            &subj,
            &schedule,
            &[pk_d],
            &[pk_d],
            &[],
        );
        let out = walk[0];
        let (g, he) = fd2([cl, v1], |p| {
            let pkd = PkDual {
                cl: p[0],
                v: p[1],
                q,
                v2,
                ka: 0.0,
                q3: 0.0,
                v3: 0.0,
                f: 1.0,
            };
            event_driven_sens_g::<f64>(PkModel::TwoCptIv, &subj, &schedule, &[pkd], &[pkd], &[])[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(out.grad[i], g[i], max_relative = 1e-5, epsilon = 1e-9);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    out.hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-7
                );
            }
        }
    }

    fn pk_3cpt(cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q2;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[crate::types::PK_IDX_Q3] = q3;
        p.values[crate::types::PK_IDX_V3] = v3;
        p.values[PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = 1.0;
        p
    }

    fn pk_dual_3cpt_f64(p: &PkParams) -> PkDual<f64> {
        PkDual {
            cl: p.cl(),
            v: p.v(),
            q: p.q(),
            v2: p.v2(),
            ka: p.ka(),
            q3: p.q3(),
            v3: p.v3(),
            f: p.f_bio(),
        }
    }

    /// The generic 3-cpt walk at `f64` must reproduce the production event-driven
    /// predictions across IV bolus, IV infusion, and oral — confirming the
    /// eigenmode propagators match production bit-for-bit.
    #[test]
    fn event_walk_g_3cpt_matches_production_f64() {
        struct Case {
            model: PkModel,
            dose: DoseEvent,
        }
        let obs = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0];
        let (cl, v1, q2, v2, q3, v3, ka) = (3.0, 30.0, 2.0, 40.0, 0.8, 120.0, 1.2);
        let cases = [
            Case {
                model: PkModel::ThreeCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            },
            Case {
                model: PkModel::ThreeCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0), // 5 h infusion
            },
            Case {
                model: PkModel::ThreeCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            },
            // Steady-state doses exercise the 3-cpt `equilibrate_ss_g` per-event
            // equilibration and the SS branches of the disposition/oral propagators.
            Case {
                model: PkModel::ThreeCptIv,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
            },
            Case {
                model: PkModel::ThreeCptOral,
                dose: DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0),
            },
        ];
        for (ci, c) in cases.iter().enumerate() {
            let subj = make_subject(vec![c.dose.clone()], obs.clone());
            let pk = pk_3cpt(cl, v1, q2, v2, q3, v3, ka);
            let prod = event_driven_predictions(c.model, &subj, &[pk], &vec![pk; obs.len()], &[]);
            let schedule = EventSchedule::for_subject(&subj, c.model, &subj.doses, &[0.0]);
            let pkd = pk_dual_3cpt_f64(&pk);
            let walk = event_driven_sens_g::<f64>(
                c.model,
                &subj,
                &schedule,
                &[pkd],
                &vec![pkd; obs.len()],
                &[],
            );
            for (j, (&p, &w)) in prod.iter().zip(walk.iter()).enumerate() {
                approx::assert_relative_eq!(w, p, max_relative = 1e-10, epsilon = 1e-11);
                assert!(p >= 0.0, "case {ci} obs {j}: production conc negative");
            }
        }
    }

    /// The 3-cpt walk's `Dual2` grad/Hessian (w.r.t. cl, v1) must match FD of the
    /// `f64` walk — the eigenmode propagator differentiates exactly.
    #[test]
    fn event_walk_g_3cpt_dual_matches_fd() {
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![6.0];
        let subj = make_subject(vec![dose], obs.clone());
        let (cl, v1, q2, v2, q3, v3) = (3.0, 30.0, 2.0, 40.0, 0.8, 120.0);
        let schedule = EventSchedule::for_subject(&subj, PkModel::ThreeCptIv, &subj.doses, &[0.0]);
        let seed = |cl: Dual2<2>, v1: Dual2<2>| PkDual {
            cl,
            v: v1,
            q: Dual2::<2>::constant(q2),
            v2: Dual2::<2>::constant(v2),
            ka: Dual2::<2>::constant(0.0),
            q3: Dual2::<2>::constant(q3),
            v3: Dual2::<2>::constant(v3),
            f: Dual2::<2>::constant(1.0),
        };
        let pk_d = seed(Dual2::var(cl, 0), Dual2::var(v1, 1));
        let walk = event_driven_sens_g::<Dual2<2>>(
            PkModel::ThreeCptIv,
            &subj,
            &schedule,
            &[pk_d],
            &[pk_d],
            &[],
        );
        let out = walk[0];
        let (g, he) = fd2([cl, v1], |p| {
            let pkd = PkDual {
                cl: p[0],
                v: p[1],
                q: q2,
                v2,
                ka: 0.0,
                q3,
                v3,
                f: 1.0,
            };
            event_driven_sens_g::<f64>(PkModel::ThreeCptIv, &subj, &schedule, &[pkd], &[pkd], &[])
                [0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(out.grad[i], g[i], max_relative = 1e-5, epsilon = 1e-9);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    out.hess[i][j],
                    he[i][j],
                    max_relative = 4e-3,
                    epsilon = 1e-7
                );
            }
        }
    }

    /// Two-occasion IOV shape: the dose in occasion 1 decays with occasion-1
    /// params; the carried-over amount continues decaying with occasion-2 params
    /// after the boundary. The `f64` walk with per-event params must match the
    /// production event-driven predictor fed the same per-event params (which is
    /// exactly how `predict_iov` runs), confirming the walk handles
    /// parameter-switching mid-decay.
    #[test]
    fn event_walk_g_iov_carryover_matches_production() {
        // One dose at t=0 (occasion 1), observations spanning the boundary at
        // t=12 into occasion 2 with a different clearance.
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![1.0, 6.0, 11.0, 13.0, 18.0, 24.0];
        let subj = make_subject(vec![dose], obs.clone());

        let pk_occ1 = pk_full(5.0, 50.0, 1.0);
        let pk_occ2 = pk_full(8.0, 50.0, 1.0); // faster clearance in occasion 2
                                               // Per-event params: dose is occasion 1; obs before t=12 are occasion 1,
                                               // after are occasion 2.
        let pk_at_obs: Vec<PkParams> = obs
            .iter()
            .map(|&t| if t < 12.0 { pk_occ1 } else { pk_occ2 })
            .collect();

        let prod = event_driven_predictions(PkModel::OneCptIv, &subj, &[pk_occ1], &pk_at_obs, &[]);
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &subj.doses, &[0.0]);
        let pk_at_obs_g: Vec<OneCptPk<f64>> = pk_at_obs.iter().map(one_cpt_pk_f64).collect();
        let walk = event_driven_sens_one_cpt_g::<f64>(
            false,
            &subj,
            &schedule,
            &[one_cpt_pk_f64(&pk_occ1)],
            &pk_at_obs_g,
            &[],
        );
        for (&p, &w) in prod.iter().zip(walk.iter()) {
            approx::assert_relative_eq!(w, p, max_relative = 1e-12, epsilon = 1e-12);
        }
    }

    /// The `Dual2` walk's `∂(conc)/∂(cl, v)` (and Hessian) at one observation must
    /// match finite differences of the `f64` walk — the walk differentiates the
    /// full multi-interval prediction exactly. Single-occasion seeding (the same
    /// `(cl, v)` on every event) isolates the propagation chain.
    #[test]
    fn event_walk_g_dual_matches_fd() {
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let obs = vec![6.0];
        let subj = make_subject(vec![dose], obs.clone());
        let (cl, v, ka) = (5.0, 50.0, 1.0);

        let seed = |cl: Dual2<2>, v: Dual2<2>| OneCptPk {
            cl,
            v,
            ka: Dual2::<2>::constant(ka),
            f: Dual2::<2>::constant(1.0),
        };
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &subj.doses, &[0.0]);
        let pk_d = seed(Dual2::var(cl, 0), Dual2::var(v, 1));
        let walk =
            event_driven_sens_one_cpt_g::<Dual2<2>>(false, &subj, &schedule, &[pk_d], &[pk_d], &[]);
        let out = walk[0];

        let (g, he) = fd2([cl, v], |p| {
            let pk_g = OneCptPk {
                cl: p[0],
                v: p[1],
                ka,
                f: 1.0,
            };
            let w =
                event_driven_sens_one_cpt_g::<f64>(false, &subj, &schedule, &[pk_g], &[pk_g], &[]);
            w[0]
        });
        for i in 0..2 {
            approx::assert_relative_eq!(out.grad[i], g[i], max_relative = 1e-5, epsilon = 1e-9);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    out.hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-7
                );
            }
        }
    }

    /// Same for 1-cpt oral, validating the central-compartment amount's `∂/∂(cl,v)`
    /// through the Bateman term.
    #[test]
    fn one_cpt_oral_g_dual_matches_fd() {
        let (cl, v, ka, dt, ad, ac) = (1.2, 12.0, 0.8, 4.0, 50.0, 3.0);
        let mut sd = [Dual2::<2>::constant(ad), Dual2::<2>::constant(ac)];
        propagate_one_cpt_oral_g::<Dual2<2>>(
            &mut sd,
            Dual2::<2>::from_f64(dt),
            Dual2::var(cl, 0),
            Dual2::var(v, 1),
            Dual2::constant(ka),
            Dual2::constant(0.0),
            Dual2::constant(0.0),
        );
        let (g, he) = fd2([cl, v], |p| {
            let mut s = [ad, ac];
            propagate_one_cpt_oral_g::<f64>(&mut s, dt, p[0], p[1], ka, 0.0, 0.0);
            s[1] // central amount
        });
        for i in 0..2 {
            approx::assert_relative_eq!(sd[1].grad[i], g[i], max_relative = 1e-4, epsilon = 1e-8);
            for j in 0..2 {
                approx::assert_relative_eq!(
                    sd[1].hess[i][j],
                    he[i][j],
                    max_relative = 3e-3,
                    epsilon = 1e-6
                );
            }
        }
    }
}
