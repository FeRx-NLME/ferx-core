//! Forward-mode AD for the event-driven Jacobian d(predictions)/d(eta).
//!
//! Lives in its own module with private copies of every propagator
//! helper. A previous attempt that shared `propagate_state_ad` and
//! friends with the reverse-mode NLL in `event_driven_ad` triggered
//! `Enzyme: Cannot deduce type of phi` failures in *reverse-mode* (the
//! function that didn't change!) once the forward-mode autodiff macro
//! was added — fat-LTO inlining merges the IR of both AD-instrumented
//! call graphs and the resulting phi nodes confuse Enzyme's type
//! deduction. Sibling-module isolation keeps the two pipelines on
//! independent IR.
//!
//! The body of `predict_all_event_driven_ad` and the propagators are
//! deliberate copy-paste from `event_driven_ad`. When the upstream
//! Enzyme issue is fixed they should be merged back; until then,
//! changes to one MUST be mirrored in the other.

use crate::ad::ad_gradients::{
    PK_ID_ONE_CPT_IV, PK_ID_ONE_CPT_ORAL, PK_ID_THREE_CPT_IV, PK_ID_THREE_CPT_ORAL,
    PK_ID_TWO_CPT_IV, PK_ID_TWO_CPT_ORAL,
};
use crate::ad::event_driven_ad::{FlatEventData, FlatEventTv};
use crate::types::*;
use std::autodiff::autodiff_forward;

/// LTBS positivity floor for this forward-AD path. Mirrors
/// [`crate::pk::LTBS_FLOOR`]; kept local so Enzyme sees a plain literal.
const LTBS_FLOOR_AD: f64 = 1e-12;

// ─────────────────────────────────────────────────────────────────────
// Forward-mode AD-instrumented predict.
// ─────────────────────────────────────────────────────────────────────

#[autodiff_forward(
    predict_all_event_driven_ad_tangent,
    Dual,       // eta
    Const,      // tv_per_event
    Const,      // event_times
    Const,      // event_kinds
    Const,      // event_orig_idx_f64
    Const,      // event_reset_floor
    Const,      // dose_times
    Const,      // dose_amts
    Const,      // dose_rates
    Const,      // dose_durations
    Const,      // dose_cmts_f64
    Const,      // pk_idx_f64
    Const,      // sel_flat
    Const,      // pk_model_id_f64
    Const,      // obs_scale
    Dual        // out
)]
pub fn predict_all_event_driven_ad(
    eta: &[f64],
    tv_per_event: &[f64],
    event_times: &[f64],
    event_kinds: &[f64],
    event_orig_idx_f64: &[f64],
    // Per-event reset floor (length n_events); see sibling `event_driven_ad.rs`.
    event_reset_floor: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    dose_cmts_f64: &[f64],
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    pk_model_id_f64: f64,
    // Per-event divisor (length = event_times.len()). Non-obs slots must
    // be 1.0 — the wrapper discards them via `event_kinds`, but the AD
    // pass still writes scaled `conc` into every slot.
    obs_scale: &[f64],
    out: &mut [f64],
) {
    let n_eta = eta.len();
    let n_tv = pk_idx_f64.len();
    let n_events = event_times.len();
    let n_doses = dose_times.len();
    // +100 packs LTBS so the forward Jacobian is d log(f)/dη (see
    // `ad_gradients::predict_all_ad`).
    let ltbs = (pk_model_id_f64 as i32) >= 100;
    let pk_model_id = (pk_model_id_f64 as i32) % 100;

    let mut state0 = 0.0_f64;
    let mut state1 = 0.0_f64;
    let mut state2 = 0.0_f64;
    let mut state3 = 0.0_f64;

    let mut cur_t = if n_events > 0 { event_times[0] } else { 0.0 };

    for ev_idx in 0..n_events {
        let t_ev = event_times[ev_idx];

        // PK params at THIS event — used both for the propagation
        // [event[i-1], event[i]] (NONMEM end-of-interval / current-record
        // semantic) and for the obs read-out V if this event is an obs.
        let mut ev_cl = 0.0_f64;
        let mut ev_v = 0.0_f64;
        let mut ev_q = 0.0_f64;
        let mut ev_v2 = 0.0_f64;
        let mut ev_ka = 0.0_f64;
        let mut ev_q3 = 0.0_f64;
        let mut ev_v3 = 0.0_f64;
        // F defaults to 1.0 — see sibling `event_driven_ad.rs`.
        let mut ev_f = 1.0_f64;
        let row_off = ev_idx * n_tv;
        for i in 0..n_tv {
            let mut eta_contrib = 0.0;
            for j in 0..n_eta {
                eta_contrib += sel_flat[i * n_eta + j] * eta[j];
            }
            let val = tv_per_event[row_off + i] * eta_contrib.exp();
            let idx = pk_idx_f64[i] as usize;
            if idx == PK_IDX_CL {
                ev_cl = val;
            } else if idx == PK_IDX_V {
                ev_v = val;
            } else if idx == PK_IDX_Q {
                ev_q = val;
            } else if idx == PK_IDX_V2 {
                ev_v2 = val;
            } else if idx == PK_IDX_KA {
                ev_ka = val;
            } else if idx == PK_IDX_F {
                ev_f = val;
            } else if idx == PK_IDX_Q3 {
                ev_q3 = val;
            } else if idx == PK_IDX_V3 {
                ev_v3 = val;
            }
        }

        let (s0_new, s1_new, s2_new, s3_new) = propagate_state_jac(
            pk_model_id,
            state0,
            state1,
            state2,
            state3,
            cur_t,
            t_ev,
            ev_cl,
            ev_v,
            ev_q,
            ev_v2,
            ev_ka,
            ev_q3,
            ev_v3,
            ev_f,
            dose_times,
            dose_rates,
            dose_durations,
            dose_cmts_f64,
            n_doses,
            event_reset_floor[ev_idx],
        );
        state0 = s0_new;
        state1 = s1_new;
        state2 = s2_new;
        state3 = s3_new;

        let kind = event_kinds[ev_idx];
        let orig = event_orig_idx_f64[ev_idx] as usize;
        let dose_idx = if kind < 0.5 { orig } else { 0 };

        // Reset branch (kind=3.0): zero every compartment after propagating
        // into the reset time. `keep` is a Const mask (0 at a reset). Mirrors
        // the reverse-mode kernel in `event_driven_ad.rs`.
        let keep = if kind > 2.5 { 0.0 } else { 1.0 };
        state0 *= keep;
        state1 *= keep;
        state2 *= keep;
        state3 *= keep;

        // is_dose=0 for obs (kind=1) and pk-only (kind=2), so their
        // state0 is unchanged regardless of dose_*[dose_idx]. `ev_f`
        // applies F1 at bolus injection — see sibling `event_driven_ad.rs`.
        let is_dose = if kind < 0.5 { 1.0 } else { 0.0 };
        let is_bolus = if dose_rates[dose_idx] == 0.0 {
            1.0
        } else {
            0.0
        };
        state0 += is_dose * is_bolus * dose_amts[dose_idx] * ev_f;

        // Central-compartment slot: oral models put depot in state0 and
        // central in state1; IV models put central in state0. See the
        // matching dispatch in `event_driven_ad.rs`.
        let central_amt = if pk_model_id == PK_ID_ONE_CPT_ORAL
            || pk_model_id == PK_ID_TWO_CPT_ORAL
            || pk_model_id == PK_ID_THREE_CPT_ORAL
        {
            state1
        } else {
            state0
        };

        let v_safe = ev_v.abs() + 1e-30;
        let conc_raw = central_amt / v_safe;
        let conc_clamped = (conc_raw + conc_raw.abs()) * 0.5;
        let scaled = conc_clamped / obs_scale[ev_idx];
        // LTBS log-wrap with an explicit-comparison floor (no `f64::max`).
        let conc = if ltbs {
            let c = if scaled < LTBS_FLOOR_AD {
                LTBS_FLOOR_AD
            } else {
                scaled
            };
            c.ln()
        } else {
            scaled
        };

        // Unconditional write: every Dual output slot must be written
        // exactly once for forward-mode pointer tracking. Non-obs slots
        // (doses, pk-only) are scratch — the wrapper drops them via
        // `event_kinds`.
        out[ev_idx] = conc;

        cur_t = t_ev;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Inlined event-driven propagators for the forward-mode AD path.
//
// Copied from `event_driven_ad` with `_jac` suffix — see the
// module-level comment about AD-pass isolation. Math, AD-safety
// rationale, and inline comments are all intentionally identical.
// ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn propagate_state_jac(
    pk_model_id: i32,
    state0: f64,
    state1: f64,
    state2: f64,
    state3: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v: f64,
    q: f64,
    v2: f64,
    ka: f64,
    q3: f64,
    v3: f64,
    f_bio: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    dose_cmts_f64: &[f64],
    n_doses: usize,
    reset_floor: f64,
) -> (f64, f64, f64, f64) {
    // ID dispatch — mirrors `event_driven_ad.rs::propagate_state_ad`. Per
    // issue #176, IV bolus and infusion share a single ID; the route is
    // chosen per dose inside the propagator from RATE.
    if pk_model_id == PK_ID_ONE_CPT_IV {
        let s0 = propagate_one_cpt_jac(
            state0,
            t_from,
            t_to,
            cl,
            v,
            f_bio,
            dose_times,
            dose_rates,
            dose_durations,
            n_doses,
            reset_floor,
        );
        (s0, state1, state2, state3)
    } else if pk_model_id == PK_ID_ONE_CPT_ORAL {
        let (s0, s1) = propagate_one_cpt_oral_jac(state0, state1, t_from, t_to, cl, v, ka);
        (s0, s1, state2, state3)
    } else if pk_model_id == PK_ID_TWO_CPT_IV {
        let (s0, s1) = propagate_two_cpt_jac(
            state0,
            state1,
            t_from,
            t_to,
            cl,
            v,
            q,
            v2,
            f_bio,
            dose_times,
            dose_rates,
            dose_durations,
            dose_cmts_f64,
            n_doses,
            reset_floor,
        );
        (s0, s1, state2, state3)
    } else if pk_model_id == PK_ID_TWO_CPT_ORAL {
        let (s0, s1, s2) =
            propagate_two_cpt_oral_jac(state0, state1, state2, t_from, t_to, cl, v, q, v2, ka);
        (s0, s1, s2, state3)
    } else if pk_model_id == PK_ID_THREE_CPT_IV {
        let (s0, s1, s2) = propagate_three_cpt_jac(
            state0,
            state1,
            state2,
            t_from,
            t_to,
            cl,
            v,
            q,
            v2,
            q3,
            v3,
            f_bio,
            dose_times,
            dose_rates,
            dose_durations,
            dose_cmts_f64,
            n_doses,
            reset_floor,
        );
        (s0, s1, s2, state3)
    } else if pk_model_id == PK_ID_THREE_CPT_ORAL {
        propagate_three_cpt_oral_jac(
            state0, state1, state2, state3, t_from, t_to, cl, v, q, v2, q3, v3, ka,
        )
    } else {
        (state0, state1, state2, state3)
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_one_cpt_jac(
    state0: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v: f64,
    f_bio: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    n_doses: usize,
    reset_floor: f64,
) -> f64 {
    let v_safe = v.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let ke = cl_safe / v_safe;
    let dt = t_to - t_from;
    let exp_dt = (-ke * dt).exp();

    let mut s0 = state0 * exp_dt;

    for d in 0..n_doses {
        let s_i = dose_times[d];
        let e_i = s_i + dose_durations[d];
        let p_start = (s_i + t_from + (s_i - t_from).abs()) * 0.5;
        let p_end = (e_i + t_to - (e_i - t_to).abs()) * 0.5;
        let tau_to_raw = t_to - p_end;
        let tau_to = (tau_to_raw + tau_to_raw.abs()) * 0.5;
        let tau_total_raw = t_to - p_start;
        let diff = tau_total_raw - tau_to;
        let tau_total = tau_to + (diff + diff.abs()) * 0.5;
        let r = f_bio * dose_rates[d];
        // Infusions started before the most recent reset are off (Const mask).
        let active = if s_i < reset_floor { 0.0 } else { 1.0 };
        let contribution = (r / ke) * ((-ke * tau_to).exp() - (-ke * tau_total).exp());
        s0 += active * contribution;
    }
    s0
}

#[allow(clippy::too_many_arguments)]
fn propagate_two_cpt_jac(
    state0: f64,
    state1: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    f_bio: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    dose_cmts_f64: &[f64],
    n_doses: usize,
    reset_floor: f64,
) -> (f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let q_safe = q.abs() + 1e-30;
    let v2_safe = v2.abs() + 1e-30;
    let k10 = cl_safe / v1_safe;
    let k12 = q_safe / v1_safe;
    let k21 = q_safe / v2_safe;
    let s = k10 + k12 + k21;
    let d_eig = k10 * k21;
    let arg = s * s - 4.0 * d_eig;
    let arg_clamped = (arg + arg.abs()) * 0.5;
    let disc = arg_clamped.sqrt();
    let alpha = (s + disc) * 0.5;
    let beta = d_eig / alpha;

    let dt = t_to - t_from;
    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    let s_homog = state1 / k12;
    let denom = beta - alpha;
    let c1 = (state0 - s_homog * (k21 - beta)) / denom;
    let c2 = s_homog - c1;

    let mut s0 = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let mut s1 = (c1 * e_alpha + c2 * e_beta) * k12;

    for d in 0..n_doses {
        let s_i = dose_times[d];
        let e_i = s_i + dose_durations[d];
        let p_start = (s_i + t_from + (s_i - t_from).abs()) * 0.5;
        let p_end = (e_i + t_to - (e_i - t_to).abs()) * 0.5;
        let tau_to_raw = t_to - p_end;
        let tau_to = (tau_to_raw + tau_to_raw.abs()) * 0.5;
        let tau_total_raw = t_to - p_start;
        let diff = tau_total_raw - tau_to;
        let tau_total = tau_to + (diff + diff.abs()) * 0.5;

        let cmt = dose_cmts_f64[d] as i32;
        let r = f_bio * dose_rates[d];
        // Infusions started before the most recent reset are off (Const mask).
        let active = if s_i < reset_floor { 0.0 } else { 1.0 };
        let (a_ss_1, a_ss_2) = if cmt == 2 {
            (
                r * v1_safe / cl_safe,
                r * (cl_safe + q_safe) * v2_safe / (cl_safe * q_safe),
            )
        } else {
            (r * v1_safe / cl_safe, r * v2_safe / cl_safe)
        };
        let s_ss = a_ss_2 / k12;
        let c1_ss = (a_ss_1 - s_ss * (k21 - beta)) / denom;
        let c2_ss = s_ss - c1_ss;

        let e_a_to = (-alpha * tau_to).exp();
        let e_a_tot = (-alpha * tau_total).exp();
        let e_b_to = (-beta * tau_to).exp();
        let e_b_tot = (-beta * tau_total).exp();

        let a1_contrib =
            c1_ss * (k21 - alpha) * (e_a_to - e_a_tot) + c2_ss * (k21 - beta) * (e_b_to - e_b_tot);
        let a2_contrib = k12 * (c1_ss * (e_a_to - e_a_tot) + c2_ss * (e_b_to - e_b_tot));

        s0 += active * a1_contrib;
        s1 += active * a2_contrib;
    }
    (s0, s1)
}

#[allow(clippy::too_many_arguments)]
fn propagate_one_cpt_oral_jac(
    state0: f64,
    state1: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v: f64,
    ka: f64,
) -> (f64, f64) {
    let v_safe = v.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let ka_safe = ka.abs() + 1e-30;
    let ke = cl_safe / v_safe;
    let dt = t_to - t_from;
    let e_ka = (-ka_safe * dt).exp();
    let e_ke = (-ke * dt).exp();

    let denom = (ke - ka_safe)
        + if (ke - ka_safe).abs() < 1e-9 {
            1e-9
        } else {
            0.0
        };

    let s0 = state0 * e_ka;
    let s1 = state1 * e_ke + (ka_safe * state0 / denom) * (e_ka - e_ke);
    (s0, s1)
}

#[allow(clippy::too_many_arguments)]
fn propagate_two_cpt_oral_jac(
    state0: f64,
    state1: f64,
    state2: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
) -> (f64, f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let q_safe = q.abs() + 1e-30;
    let v2_safe = v2.abs() + 1e-30;
    let ka_safe = ka.abs() + 1e-30;
    let k10 = cl_safe / v1_safe;
    let k12 = q_safe / v1_safe;
    let k21 = q_safe / v2_safe;
    let s = k10 + k12 + k21;
    let d_eig = k10 * k21;
    let arg = s * s - 4.0 * d_eig;
    let arg_clamped = (arg + arg.abs()) * 0.5;
    let disc = arg_clamped.sqrt();
    let alpha = (s + disc) * 0.5;
    let beta = d_eig / alpha;

    let dt = t_to - t_from;
    let e_ka = (-ka_safe * dt).exp();
    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    let denom_depot = (ka_safe - alpha) * (ka_safe - beta);
    let cap_a = ka_safe * state0 * (k21 - ka_safe) / denom_depot;
    let cap_b = ka_safe * state0 * k12 / denom_depot;

    let h_c_0 = state1 - cap_a;
    let h_p_0 = state2 - cap_b;

    let s_homog = h_p_0 / k12;
    let denom = beta - alpha;
    let c1 = (h_c_0 - s_homog * (k21 - beta)) / denom;
    let c2 = s_homog - c1;

    let h_c_dt = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let h_p_dt = (c1 * e_alpha + c2 * e_beta) * k12;

    let new_s0 = state0 * e_ka;
    let new_s1 = h_c_dt + cap_a * e_ka;
    let new_s2 = h_p_dt + cap_b * e_ka;
    (new_s0, new_s1, new_s2)
}

fn macro_rates_three_jac(
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, f64, f64, f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let q2_safe = q2.abs() + 1e-30;
    let v2_safe = v2.abs() + 1e-30;
    let q3_safe = q3.abs() + 1e-30;
    let v3_safe = v3.abs() + 1e-30;
    let k10 = cl_safe / v1_safe;
    let k12 = q2_safe / v1_safe;
    let k21 = q2_safe / v2_safe;
    let k13 = q3_safe / v1_safe;
    let k31 = q3_safe / v3_safe;
    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;
    let h = s2 / 3.0;
    let p = s1 - s2 * s2 / 3.0;
    let qq = s1 * s2 / 3.0 - 2.0 * s2 * s2 * s2 / 27.0 - s0;
    let eps = 1e-30;
    let p_safe = (p + (-eps) - (p - (-eps)).abs()) * 0.5;
    let m = 2.0 * (-p_safe / 3.0).sqrt();
    let mut arg = 3.0 * qq / (p_safe * m);
    if arg < -1.0 {
        arg = -1.0;
    }
    if arg > 1.0 {
        arg = 1.0;
    }
    let phi = arg.acos() / 3.0;
    let pi23 = 2.0 * std::f64::consts::FRAC_PI_3;
    let l0 = m * phi.cos() + h;
    let l1 = m * (phi - pi23).cos() + h;
    let l2 = m * (phi - 2.0 * pi23).cos() + h;
    let alpha = if l0 >= l1 && l0 >= l2 {
        l0
    } else if l1 >= l2 {
        l1
    } else {
        l2
    };
    let gamma = if l0 <= l1 && l0 <= l2 {
        l0
    } else if l1 <= l2 {
        l1
    } else {
        l2
    };
    let beta = s2 - alpha - gamma;
    (alpha, beta, gamma, k21, k31)
}

#[allow(clippy::too_many_arguments)]
fn three_cpt_mode_jac(
    mu: f64,
    c: f64,
    p1: f64,
    p2: f64,
    k12: f64,
    k13: f64,
    k21: f64,
    k31: f64,
    dt: f64,
) -> (f64, f64, f64) {
    let d21 = k21 - mu;
    let d31 = k31 - mu;
    let v_c = d21 * d31;
    let v_p1 = k12 * d31;
    let v_p2 = k13 * d21;
    let w_c = d21 * d31;
    let w_p1 = k21 * d31;
    let w_p2 = k31 * d21;
    let norm = v_c * w_c + v_p1 * w_p1 + v_p2 * w_p2;
    let norm_safe = norm + if norm.abs() < 1e-30 { 1e-30 } else { 0.0 };
    let proj = w_c * c + w_p1 * p1 + w_p2 * p2;
    let coef = proj / norm_safe;
    let exp_term = (-mu * dt).exp();
    (
        coef * v_c * exp_term,
        coef * v_p1 * exp_term,
        coef * v_p2 * exp_term,
    )
}

#[allow(clippy::too_many_arguments)]
fn propagate_three_cpt_jac(
    state0: f64,
    state1: f64,
    state2: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    f_bio: f64,
    dose_times: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    dose_cmts_f64: &[f64],
    n_doses: usize,
    reset_floor: f64,
) -> (f64, f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let cl_safe = cl.abs() + 1e-30;
    let v2_safe = v2.abs() + 1e-30;
    let v3_safe = v3.abs() + 1e-30;
    let q_safe = q.abs() + 1e-30;
    let q3_safe = q3.abs() + 1e-30;
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_jac(cl, v1, q, v2, q3, v3);
    let k12 = q_safe / v1_safe;
    let k13 = q3_safe / v1_safe;

    let dt = t_to - t_from;

    let (ca, p1a, p2a) = three_cpt_mode_jac(alpha, state0, state1, state2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode_jac(beta, state0, state1, state2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode_jac(gamma, state0, state1, state2, k12, k13, k21, k31, dt);

    let mut s0 = ca + cb + cg;
    let mut s1 = p1a + p1b + p1g;
    let mut s2 = p2a + p2b + p2g;

    for d in 0..n_doses {
        let s_i = dose_times[d];
        let e_i = s_i + dose_durations[d];
        let p_start = (s_i + t_from + (s_i - t_from).abs()) * 0.5;
        let p_end = (e_i + t_to - (e_i - t_to).abs()) * 0.5;
        let tau_to_raw = t_to - p_end;
        let tau_to = (tau_to_raw + tau_to_raw.abs()) * 0.5;
        let tau_total_raw = t_to - p_start;
        let diff = tau_total_raw - tau_to;
        let tau_total = tau_to + (diff + diff.abs()) * 0.5;

        let cmt = dose_cmts_f64[d] as i32;
        let r = f_bio * dose_rates[d];
        // Infusions started before the most recent reset are off (Const mask).
        let active = if s_i < reset_floor { 0.0 } else { 1.0 };
        let (a_ss_c, a_ss_p1, a_ss_p2) = if cmt == 2 {
            (
                r * v1_safe / cl_safe,
                r * (cl_safe + q_safe) * v2_safe / (cl_safe * q_safe),
                r * v3_safe / cl_safe,
            )
        } else if cmt == 3 {
            (
                r * v1_safe / cl_safe,
                r * v2_safe / cl_safe,
                r * (cl_safe + q3_safe) * v3_safe / (cl_safe * q3_safe),
            )
        } else {
            (
                r * v1_safe / cl_safe,
                r * v2_safe / cl_safe,
                r * v3_safe / cl_safe,
            )
        };

        let (ca_to, p1a_to, p2a_to) =
            three_cpt_mode_jac(alpha, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (cb_to, p1b_to, p2b_to) =
            three_cpt_mode_jac(beta, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (cg_to, p1g_to, p2g_to) =
            three_cpt_mode_jac(gamma, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (ca_tot, p1a_tot, p2a_tot) = three_cpt_mode_jac(
            alpha, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );
        let (cb_tot, p1b_tot, p2b_tot) = three_cpt_mode_jac(
            beta, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );
        let (cg_tot, p1g_tot, p2g_tot) = three_cpt_mode_jac(
            gamma, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );

        s0 += active * ((ca_to - ca_tot) + (cb_to - cb_tot) + (cg_to - cg_tot));
        s1 += active * ((p1a_to - p1a_tot) + (p1b_to - p1b_tot) + (p1g_to - p1g_tot));
        s2 += active * ((p2a_to - p2a_tot) + (p2b_to - p2b_tot) + (p2g_to - p2g_tot));
    }
    (s0, s1, s2)
}

#[allow(clippy::too_many_arguments)]
fn propagate_three_cpt_oral_jac(
    state0: f64,
    state1: f64,
    state2: f64,
    state3: f64,
    t_from: f64,
    t_to: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
) -> (f64, f64, f64, f64) {
    let v1_safe = v1.abs() + 1e-30;
    let q_safe = q.abs() + 1e-30;
    let q3_safe = q3.abs() + 1e-30;
    let ka_safe = ka.abs() + 1e-30;
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_jac(cl, v1, q, v2, q3, v3);
    let k12 = q_safe / v1_safe;
    let k13 = q3_safe / v1_safe;

    let dt = t_to - t_from;
    let e_ka = (-ka_safe * dt).exp();

    let d21 = k21 - ka_safe;
    let d31 = k31 - ka_safe;
    let denom_depot = (ka_safe - alpha) * (ka_safe - beta) * (ka_safe - gamma);
    let denom_safe = denom_depot
        + if denom_depot.abs() < 1e-30 {
            1e-30
        } else {
            0.0
        };
    let scale = -ka_safe * state0 / denom_safe;
    let cap_a = scale * d21 * d31;
    let cap_b = scale * k12 * d31;
    let cap_c = scale * k13 * d21;

    let h_c = state1 - cap_a;
    let h_p1 = state2 - cap_b;
    let h_p2 = state3 - cap_c;

    let (ca, p1a, p2a) = three_cpt_mode_jac(alpha, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode_jac(beta, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode_jac(gamma, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);

    let new_s0 = state0 * e_ka;
    let new_s1 = ca + cb + cg + cap_a * e_ka;
    let new_s2 = p1a + p1b + p1g + cap_b * e_ka;
    let new_s3 = p2a + p2b + p2g + cap_c * e_ka;
    (new_s0, new_s1, new_s2, new_s3)
}

// ─── Public wrapper ────────────────────────────────────────────────

/// Forward-mode AD Jacobian d(predictions)/d(eta) for event-driven
/// subjects. Returns `n_obs × n_eta`. Calls
/// `predict_all_event_driven_ad_tangent` once per η-column, scattering
/// the obs-only tangents into the matrix via the `event_kinds` /
/// `event_orig_idx_f64` back-pointers.
#[allow(clippy::too_many_arguments)]
pub fn compute_jacobian_event_driven_ad(
    eta: &[f64],
    tv_per_event: &FlatEventTv,
    event_data: &FlatEventData,
    n_obs: usize,
    pk_model: PkModel,
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    // Per-event divisor (length = event_data.event_times.len()). The
    // caller pads non-obs entries to 1.0 since the AD pass writes scaled
    // conc into every event slot before the wrapper drops non-obs.
    obs_scale: &[f64],
    log_transform: bool,
) -> nalgebra::DMatrix<f64> {
    let n_eta = eta.len();
    // +100 packs LTBS so the forward prediction is log-wrapped, making this
    // Jacobian d log(f)/dη — consistent with the log-scale objective.
    let ltbs_offset = if log_transform { 100 } else { 0 };
    let pk_id = (crate::ad::ad_gradients::pk_model_to_id(pk_model) + ltbs_offset) as f64;
    let n_events = event_data.event_times.len();
    let mut jac = nalgebra::DMatrix::zeros(n_obs, n_eta);

    // Pad zero-length dose arrays so the kernel's masked-but-still-
    // evaluated index loads stay in-bounds for subjects without doses.
    let dose_times_padded: &[f64] = if event_data.dose_times.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_times
    };
    let dose_amts_padded: &[f64] = if event_data.dose_amts.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_amts
    };
    let dose_rates_padded: &[f64] = if event_data.dose_rates.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_rates
    };
    let dose_durations_padded: &[f64] = if event_data.dose_durations.is_empty() {
        &[0.0]
    } else {
        &event_data.dose_durations
    };
    let dose_cmts_padded: &[f64] = if event_data.dose_cmts_f64.is_empty() {
        &[1.0]
    } else {
        &event_data.dose_cmts_f64
    };

    for j in 0..n_eta {
        let mut d_eta = vec![0.0_f64; n_eta];
        d_eta[j] = 1.0;

        let mut out = vec![0.0_f64; n_events];
        let mut d_out = vec![0.0_f64; n_events];

        predict_all_event_driven_ad_tangent(
            eta,
            &d_eta,
            &tv_per_event.tv,
            &event_data.event_times,
            &event_data.event_kinds,
            &event_data.event_orig_idx_f64,
            &event_data.event_reset_floor,
            dose_times_padded,
            dose_amts_padded,
            dose_rates_padded,
            dose_durations_padded,
            dose_cmts_padded,
            pk_idx_f64,
            sel_flat,
            pk_id,
            obs_scale,
            &mut out,
            &mut d_out,
        );

        for ev in 0..n_events {
            // event_kinds: 0=dose, 1=obs, 2=pk-only. Only obs slots
            // map to a Jacobian row.
            let k = event_data.event_kinds[ev];
            if k > 0.5 && k < 1.5 {
                let obs_idx = event_data.event_orig_idx_f64[ev] as usize;
                jac[(obs_idx, j)] = d_out[ev];
            }
        }
    }

    jac
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ad::ad_gradients::pk_model_to_id;
    use crate::ad::event_driven_ad::FlatEventData;
    use crate::pk::event_driven::event_driven_predictions;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    // ── Reset (EVID=3/4) AD-vs-analytical regression ───────────────────
    //
    // The event-driven AD forward kernel must reproduce the reset-aware
    // analytical path (`pk::event_driven::event_driven_predictions`) for
    // subjects carrying system resets: state zeroed at the reset and any
    // infusion that started before the reset turned off. We drive the REAL
    // `FlatEventData::from_subject` timeline (so the reset-event insertion,
    // tie-break ordering, and `event_reset_floor` computation are all under
    // test) and feed constant PK params per event (no covariates), then
    // compare obs-slot predictions to the analytical reference.

    fn subject_with_resets(
        doses: Vec<DoseEvent>,
        obs_times: Vec<f64>,
        reset_times: Vec<f64>,
    ) -> Subject {
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
            reset_times,
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// Run the event-driven AD forward kernel for `subject` with constant
    /// PK params `values` (covering `pk_idx`), returning per-observation
    /// predictions in obs order. Uses the production `FlatEventData`.
    fn ad_event_driven_preds(
        pk_model: PkModel,
        subject: &Subject,
        pk_idx: &[usize],
        values: &[f64],
        dose_lagtimes: &[f64],
    ) -> Vec<f64> {
        let ed = FlatEventData::from_subject(subject, dose_lagtimes);
        let n_events = ed.event_times.len();
        let n_obs = subject.obs_times.len();

        let pk_idx_f64: Vec<f64> = pk_idx.iter().map(|&i| i as f64).collect();
        let n_tv = pk_idx.len();
        let eta = vec![0.0_f64]; // single eta, zero effect
        let sel_flat = vec![0.0_f64; n_tv]; // n_tv * n_eta (n_eta = 1), all zero
        let tv = build_tv_constant(pk_idx, values, n_events);
        let obs_scale = vec![1.0_f64; n_events]; // readout = central / V
        let pk_model_id = pk_model_to_id(pk_model) as f64;

        let mut out = vec![0.0_f64; n_events];
        predict_all_event_driven_ad(
            &eta,
            &tv,
            &ed.event_times,
            &ed.event_kinds,
            &ed.event_orig_idx_f64,
            &ed.event_reset_floor,
            &ed.dose_times,
            &ed.dose_amts,
            &ed.dose_rates,
            &ed.dose_durations,
            &ed.dose_cmts_f64,
            &pk_idx_f64,
            &sel_flat,
            pk_model_id,
            &obs_scale,
            &mut out,
        );

        let mut preds = vec![0.0_f64; n_obs];
        for ev in 0..n_events {
            let k = ed.event_kinds[ev];
            if k > 0.5 && k < 1.5 {
                let oi = ed.event_orig_idx_f64[ev] as usize;
                preds[oi] = out[ev];
            }
        }
        preds
    }

    fn pk_one_iv(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v;
        p
    }

    fn pk_three_iv(cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = cl;
        p.values[PK_IDX_V] = v1;
        p.values[PK_IDX_Q] = q2;
        p.values[PK_IDX_V2] = v2;
        p.values[PK_IDX_Q3] = q3;
        p.values[PK_IDX_V3] = v3;
        p
    }

    #[test]
    fn ad_reset_turns_off_ongoing_infusion_matches_analytical() {
        // 1-cpt IV: infusion 0–8 h (rate 125), reset at t=4 mid-infusion.
        // Obs at t=3 (pre-reset, > 0) and t=6 (post-reset). The reset zeros
        // the state AND turns the infusion off, so t=6 must read ~0.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 125.0, false, 0.0)];
        let obs_times = vec![3.0, 6.0];
        let subj = subject_with_resets(doses, obs_times.clone(), vec![4.0]);
        let pk = pk_one_iv(10.0, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let reference = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let ad = ad_event_driven_preds(
            PkModel::OneCptIv,
            &subj,
            &[PK_IDX_CL, PK_IDX_V],
            &[10.0, 100.0],
            &[],
        );

        assert!(reference[0] > 0.0 && ad[0] > 0.0, "pre-reset obs positive");
        assert_relative_eq!(ad[1], 0.0, epsilon = 1e-9);
        for (a, r) in ad.iter().zip(reference.iter()) {
            assert_relative_eq!(*a, *r, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn ad_three_cpt_two_occasion_reset_matches_analytical() {
        // Propofol-shaped: two infusion occasions separated by an EVID=4
        // reset on a 3-cpt IV model. Occasion 1: infusion at t=0 (rate 343,
        // dur 0.27 ≈ a 92.6 mg bolus-like push) + a maintenance infusion at
        // t=60. Reset at t=120 zeros the system; occasion 2 repeats. Every
        // obs prediction must match the reset-aware analytical path.
        let doses = vec![
            DoseEvent::new(0.0, 92.6, 1, 343.0, false, 0.0),
            DoseEvent::new(60.0, 89.4, 1, 1.49, false, 0.0),
            DoseEvent::new(120.0, 92.6, 1, 343.0, false, 0.0),
            DoseEvent::new(180.0, 89.4, 1, 1.49, false, 0.0),
        ];
        let obs_times = vec![2.0, 8.0, 30.0, 90.0, 122.0, 128.0, 150.0, 210.0];
        let subj = subject_with_resets(doses, obs_times.clone(), vec![120.0]);
        let pk = pk_three_iv(2.0, 6.0, 1.0, 25.0, 0.5, 250.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let reference =
            event_driven_predictions(PkModel::ThreeCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let ad = ad_event_driven_preds(
            PkModel::ThreeCptIv,
            &subj,
            &[
                PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2, PK_IDX_Q3, PK_IDX_V3,
            ],
            &[2.0, 6.0, 1.0, 25.0, 0.5, 250.0],
            &[],
        );

        // Occasion-2 obs (t≥120) must be positive — the reset+redose rebuilds
        // the profile — and the early occasion-2 point must NOT carry residual
        // occasion-1 drug (that's the reset/infusion-off invariant).
        assert!(ad[4] > 0.0, "post-reset redose obs positive");
        for (a, r) in ad.iter().zip(reference.iter()) {
            assert_relative_eq!(*a, *r, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    #[test]
    fn ad_reset_plus_lagtime_matches_analytical() {
        // Lagtime + reset on the same subject (the route that previously fell
        // back to FD). 2-cpt IV: bolus at t=0 with ALAG=1.5 h (so drug enters
        // at t=1.5), a long infusion at t=5 (ALAG shifts its window to
        // [6.5, 16.5], so it is still running when the reset hits), and a reset
        // at t=12 that must zero the state AND turn that infusion off. The
        // lagged dose timeline + reset/infusion-off must all match the
        // analytical path, which lags via `EventSchedule`.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0), // bolus, lagged to 1.5
            DoseEvent::new(5.0, 600.0, 1, 60.0, false, 0.0), // 10-h infusion → [6.5, 16.5]
            DoseEvent::new(12.0, 800.0, 1, 0.0, false, 0.0), // post-reset bolus, lagged to 13.5
        ];
        let obs_times = vec![1.0, 3.0, 8.0, 11.0, 13.0, 16.0, 24.0];
        let subj = subject_with_resets(doses, obs_times.clone(), vec![12.0]);

        let lag = 1.5_f64;
        let mut pk = PkParams::default();
        pk.values[PK_IDX_CL] = 4.0;
        pk.values[PK_IDX_V] = 30.0;
        pk.values[PK_IDX_Q] = 2.0;
        pk.values[PK_IDX_V2] = 40.0;
        pk.values[PK_IDX_LAGTIME] = lag;
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        // Analytical reference lags internally from `pk.lagtime()`.
        let reference = event_driven_predictions(PkModel::TwoCptIv, &subj, &pk_dose, &pk_obs, &[]);
        // AD path takes the lags explicitly (Const per-dose).
        let dose_lagtimes = vec![lag; subj.doses.len()];
        let ad = ad_event_driven_preds(
            PkModel::TwoCptIv,
            &subj,
            &[PK_IDX_CL, PK_IDX_V, PK_IDX_Q, PK_IDX_V2],
            &[4.0, 30.0, 2.0, 40.0],
            &dose_lagtimes,
        );

        // Obs at t=1.0 is before the lagged bolus entry (t=1.5) → ~0.
        assert_relative_eq!(ad[0], 0.0, epsilon = 1e-9);
        assert!(ad[1] > 0.0, "obs after lagged bolus entry positive");
        for (a, r) in ad.iter().zip(reference.iter()) {
            assert_relative_eq!(*a, *r, epsilon = 1e-7, max_relative = 1e-7);
        }
    }

    // Helper: build the minimum tv_per_event row layout the kernel needs.
    // `pk_idx_f64` declares which PK indices the row covers (and in what
    // order); each row carries one value per declared index.
    fn build_tv_constant(pk_idx: &[usize], values: &[f64], n_events: usize) -> Vec<f64> {
        assert_eq!(pk_idx.len(), values.len());
        let mut out = Vec::with_capacity(n_events * pk_idx.len());
        for _ in 0..n_events {
            out.extend_from_slice(values);
        }
        out
    }

    /// F scales bolus and infusion dose amounts linearly in the event-
    /// driven AD path. Regression for issue #16: the AD path was
    /// silently dropping `f_bio` at the dose-injection step (bolus) and
    /// in the propagator infusion contributions, so subjects with
    /// TV-covariates + F ≠ 1 got wrong AD gradients / predictions.
    /// We assert the linear-scaling invariant
    /// `pred(F=F0) == F0 * pred(F=1)` separately for an oral bolus and
    /// an IV infusion, since that catches both fix sites without
    /// requiring a hand-computed reference.
    #[test]
    fn ad_event_driven_applies_f_bio_to_oral_bolus() {
        // 1-cpt oral, single bolus at t=0, observations spread over the
        // absorption + elimination phase.
        let pk_model_id = pk_model_to_id(PkModel::OneCptOral) as f64;
        let event_times = vec![0.0_f64, 0.5, 1.0, 2.0, 4.0, 8.0];
        let event_kinds = vec![0.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0]; // dose then 5 obs
        let event_orig = vec![0.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0];
        let dose_times = vec![0.0_f64];
        let dose_amts = vec![1000.0_f64];
        let dose_rates = vec![0.0_f64]; // bolus
        let dose_durations = vec![0.0_f64];
        let dose_cmts = vec![1.0_f64];

        // No eta — single fake row to satisfy n_eta >= 1.
        let eta = vec![0.0_f64];
        // Include F in pk_idx so the per-event row carries it.
        let pk_idx = vec![PK_IDX_CL, PK_IDX_V, PK_IDX_KA, PK_IDX_F];
        let pk_idx_f64: Vec<f64> = pk_idx.iter().map(|&i| i as f64).collect();
        let n_tv = pk_idx.len();
        let n_eta = eta.len();
        let sel_flat = vec![0.0_f64; n_tv * n_eta]; // all-zero: no eta on any tv

        let cl = 5.0;
        let v = 50.0;
        let ka = 1.2;

        let run = |f: f64| -> Vec<f64> {
            let tv = build_tv_constant(&pk_idx, &[cl, v, ka, f], event_times.len());
            let mut out = vec![0.0_f64; event_times.len()];
            let obs_scale = vec![1.0_f64; event_times.len()];
            // No resets in these fixtures — floor is NEG_INFINITY everywhere.
            let event_reset_floor = vec![f64::NEG_INFINITY; event_times.len()];
            predict_all_event_driven_ad(
                &eta,
                &tv,
                &event_times,
                &event_kinds,
                &event_orig,
                &event_reset_floor,
                &dose_times,
                &dose_amts,
                &dose_rates,
                &dose_durations,
                &dose_cmts,
                &pk_idx_f64,
                &sel_flat,
                pk_model_id,
                &obs_scale,
                &mut out,
            );
            // Drop the dose slot (kind=0) — only obs slots carry preds.
            event_kinds
                .iter()
                .zip(out.iter())
                .filter(|(k, _)| **k > 0.5)
                .map(|(_, p)| *p)
                .collect()
        };

        let preds_unit = run(1.0);
        let preds_half = run(0.5);
        let preds_tall = run(2.5);

        // Sanity: at F=1 some obs has non-trivial concentration.
        assert!(preds_unit.iter().any(|p| *p > 1e-3));
        for (a, b) in preds_unit.iter().zip(preds_half.iter()) {
            assert_relative_eq!(*b, 0.5 * *a, epsilon = 1e-12, max_relative = 1e-12);
        }
        for (a, b) in preds_unit.iter().zip(preds_tall.iter()) {
            assert_relative_eq!(*b, 2.5 * *a, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn ad_event_driven_applies_f_bio_to_iv_infusion() {
        // 1-cpt IV infusion: dose split across a finite duration so the
        // propagator's per-dose `f_bio * dose_rates[d]` path runs (not the
        // main loop's bolus step).
        let pk_model_id = pk_model_to_id(PkModel::OneCptIv) as f64;
        let event_times = vec![0.0_f64, 0.5, 1.0, 2.0, 4.0, 8.0];
        let event_kinds = vec![0.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let event_orig = vec![0.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0];
        let dose_times = vec![0.0_f64];
        let dose_amts = vec![1000.0_f64];
        let dose_rates = vec![500.0_f64]; // 2-hour infusion
        let dose_durations = vec![2.0_f64];
        let dose_cmts = vec![1.0_f64];

        let eta = vec![0.0_f64];
        let pk_idx = vec![PK_IDX_CL, PK_IDX_V, PK_IDX_F];
        let pk_idx_f64: Vec<f64> = pk_idx.iter().map(|&i| i as f64).collect();
        let n_tv = pk_idx.len();
        let n_eta = eta.len();
        let sel_flat = vec![0.0_f64; n_tv * n_eta];

        let cl = 10.0;
        let v = 100.0;

        let run = |f: f64| -> Vec<f64> {
            let tv = build_tv_constant(&pk_idx, &[cl, v, f], event_times.len());
            let mut out = vec![0.0_f64; event_times.len()];
            let obs_scale = vec![1.0_f64; event_times.len()];
            // No resets in these fixtures — floor is NEG_INFINITY everywhere.
            let event_reset_floor = vec![f64::NEG_INFINITY; event_times.len()];
            predict_all_event_driven_ad(
                &eta,
                &tv,
                &event_times,
                &event_kinds,
                &event_orig,
                &event_reset_floor,
                &dose_times,
                &dose_amts,
                &dose_rates,
                &dose_durations,
                &dose_cmts,
                &pk_idx_f64,
                &sel_flat,
                pk_model_id,
                &obs_scale,
                &mut out,
            );
            event_kinds
                .iter()
                .zip(out.iter())
                .filter(|(k, _)| **k > 0.5)
                .map(|(_, p)| *p)
                .collect()
        };

        let preds_unit = run(1.0);
        let preds_half = run(0.5);
        let preds_tall = run(2.5);

        assert!(preds_unit.iter().any(|p| *p > 1e-3));
        for (a, b) in preds_unit.iter().zip(preds_half.iter()) {
            assert_relative_eq!(*b, 0.5 * *a, epsilon = 1e-12, max_relative = 1e-12);
        }
        for (a, b) in preds_unit.iter().zip(preds_tall.iter()) {
            assert_relative_eq!(*b, 2.5 * *a, epsilon = 1e-12, max_relative = 1e-12);
        }
    }
}
