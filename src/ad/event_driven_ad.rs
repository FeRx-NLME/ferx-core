//! Event-driven AD path for time-varying covariate subjects.
//!
//! Mirrors `ad_gradients.rs` but takes per-event `tv` arrays so the AD
//! path can honour per-event LOCF covariate snapshots — the analytical
//! superposition path in `ad_gradients.rs` only handles a single `tv`
//! per subject, which silently drops time-varying covariate updates.
//!
//! Initial scope (matches the non-AD event-driven path in
//! `pk::event_driven`):
//!   - 1-compartment IV bolus & infusion
//!   - 2-compartment IV bolus & infusion (dose into central / cmt=1)
//!
//! Oral and 3-cpt models still fall back to the single-snapshot AD path
//! in `ad_gradients.rs` (which is correct for them only when covariates
//! are time-constant — the dispatcher in `inner_optimizer.rs` handles
//! the fallback).
//!
//! ## AD safety
//!
//! Same constraints as `ad_gradients.rs` — see CLAUDE.md. In particular:
//!   - No `f64::max` / `f64::min` (lower to `llvm.maximumnum`/`minimumnum`
//!     intrinsics that Enzyme can't differentiate). Use the arithmetic
//!     forms `(a+b±|a-b|)/2` (`.abs()` lowers to `llvm.fabs`, which is
//!     handled).
//!   - Avoid branches on differentiated quantities; branch only on
//!     `Const` inputs (event metadata, dose properties, model IDs).
//!     Const branches get constant-folded by LLVM and don't poison the
//!     reverse-mode adjoint.

use crate::ad::ad_gradients::{
    PK_ID_ONE_CPT_IV, PK_ID_ONE_CPT_ORAL, PK_ID_THREE_CPT_IV, PK_ID_THREE_CPT_ORAL,
    PK_ID_TWO_CPT_IV, PK_ID_TWO_CPT_ORAL,
};
use crate::types::*;
use std::autodiff::autodiff_reverse;

/// LTBS positivity floor for this AD path. Mirrors [`crate::pk::LTBS_FLOOR`] /
/// `ad_gradients::LTBS_FLOOR_AD`; kept local so Enzyme sees a plain literal.
const LTBS_FLOOR_AD: f64 = 1e-12;

// ─────────────────────────────────────────────────────────────────────
// Flat data layout for one subject's event timeline.
// ─────────────────────────────────────────────────────────────────────

/// Pre-built flat arrays describing one subject's merged event timeline.
/// Constructed once per inner-loop call and re-used across every gradient
/// evaluation (eta perturbations don't change the timeline shape).
pub struct FlatEventData {
    /// Sorted merged event times.
    pub event_times: Vec<f64>,
    /// Event kind tag: 0.0 = dose, 1.0 = obs, 2.0 = pk-only, 3.0 = reset
    /// (EVID=3/4). Reset events sort first at a given time so they zero the
    /// state before a same-time dose lands (EVID=4).
    pub event_kinds: Vec<f64>,
    /// Original index back into `subject.doses` (when kind=0) or
    /// `subject.obs_times` (when kind=1). Unused for reset (kind=3) events.
    pub event_orig_idx_f64: Vec<f64>,
    /// Per-event reset floor: the time of the most recent system reset
    /// strictly *before* this event, or `f64::NEG_INFINITY` when none.
    /// Applied to the propagation interval ending at this event — infusions
    /// whose start is `< event_reset_floor[i]` are turned off (a reset stops
    /// ongoing infusions, just as it zeros the compartments). Mirrors the
    /// scalar `reset_floor` in `pk::event_driven::event_driven_predictions`.
    /// Const w.r.t. eta, so it threads through the AD kernels untouched.
    pub event_reset_floor: Vec<f64>,
    /// Dose-level arrays (parallel to `subject.doses`).
    pub dose_times: Vec<f64>,
    pub dose_amts: Vec<f64>,
    pub dose_rates: Vec<f64>,
    pub dose_durations: Vec<f64>,
    /// Per-dose compartment number (1-based, matches NONMEM).
    /// Used by the 2-/3-cpt AD propagators to route an infusion's
    /// steady-state contribution to the correct channel (central vs
    /// periph1 vs periph2). Const through the AD macros.
    pub dose_cmts_f64: Vec<f64>,
}

/// Same-time tie-break ordinal: `reset (0) < dose (1) < pk-only (2) < obs (3)`.
/// A reset must sort before a same-time dose so an EVID=4 zeros the state before
/// its own dose lands — matching `pk::event_driven::event_driven_predictions`.
/// `kind_tag` is the encoded event kind (0.0=dose, 1.0=obs, 2.0=pk-only,
/// 3.0=reset).
fn event_kind_order(kind_tag: f64) -> u8 {
    if kind_tag > 2.5 {
        0 // reset
    } else if kind_tag < 0.5 {
        1 // dose
    } else if kind_tag > 1.5 {
        2 // pk-only
    } else {
        3 // obs
    }
}

/// Merged, sorted event timeline shared by [`FlatEventData`] and [`FlatEventTv`]
/// so the two are guaranteed to order events identically (a single source for
/// the reset insertion, lag shift, and tie-break). Returns one
/// `(time, kind_tag, orig_idx)` per event, where `kind_tag` is 0.0=dose,
/// 1.0=obs, 2.0=pk-only, 3.0=reset and dose events carry the per-dose lag
/// (`doses[k].time + lag(k)`; a lagged dose may re-sort past a later obs —
/// correct, it genuinely happens later). Resets/obs/pk-only keep their record
/// times.
///
/// Panics if `dose_lagtimes` is non-empty and not length `subject.doses.len()`
/// (hard assert, matching `pk::event_driven::EventSchedule::for_subject`).
fn build_sorted_events(subject: &Subject, dose_lagtimes: &[f64]) -> Vec<(f64, f64, usize)> {
    assert!(
        dose_lagtimes.is_empty() || dose_lagtimes.len() == subject.doses.len(),
        "dose_lagtimes length {} != n_dose {}",
        dose_lagtimes.len(),
        subject.doses.len()
    );
    let lag = |k: usize| -> f64 { dose_lagtimes.get(k).copied().unwrap_or(0.0) };

    let mut events: Vec<(f64, f64, usize)> = Vec::with_capacity(
        subject.doses.len()
            + subject.obs_times.len()
            + subject.pk_only_times.len()
            + subject.reset_times.len(),
    );
    for (k, d) in subject.doses.iter().enumerate() {
        events.push((d.time + lag(k), 0.0, k));
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        events.push((t, 1.0, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        events.push((t, 2.0, m));
    }
    for (r, &t) in subject.reset_times.iter().enumerate() {
        events.push((t, 3.0, r));
    }
    events.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| event_kind_order(a.1).cmp(&event_kind_order(b.1)))
    });
    events
}

impl FlatEventData {
    /// Build the flat event timeline for `subject`.
    ///
    /// `dose_lagtimes` (length `subject.doses.len()`, or empty for no lag)
    /// shifts each dose's effective start to `doses[k].time + lag[k]` — for
    /// both its merged-timeline event (where the bolus injects and the
    /// propagation boundary lands) and the `dose_times` array the propagators
    /// use for infusion-window overlap. Resets/obs/pk-only keep their record
    /// times. Mirrors `pk::event_driven::EventSchedule::for_subject`.
    ///
    /// The lags are treated as Const w.r.t. eta in the AD kernels: exact for
    /// the usual eta-independent lagtime (a `THETA`, optionally covariate-
    /// scaled), and a documented approximation (drops `∂lag/∂η`) for the rare
    /// eta-dependent case — the marginal NLL value itself is still computed by
    /// the exact analytical path, so only the inner-loop gradient / FOCE `|H|`
    /// see the frozen lag. (The FOCEI inner loop routes eta-dependent lagtime
    /// to FD instead — see `inner_optimizer::resolve_gradient_method`.)
    pub fn from_subject(subject: &Subject, dose_lagtimes: &[f64]) -> Self {
        // Tripwire (#324 / #281 / #394): the AD kernels snapshot `d.rate`/`d.duration`
        // into flat f64 arrays below. A modeled-RATE dose (e.g. RATE=-2 -> D{cmt})
        // must never reach the AD path — `resolve_gradient_method` routes any subject
        // with a modeled dose to FD (the ODE engine has no AD path; the analytical AD
        // kernels can't carry `∂duration/∂η`), so the AD path only sees `Fixed` doses.
        // Reaching here with one would snapshot the *unresolved* rate/duration (0) and
        // yield a silently-wrong gradient — the FD-only-CI failure mode of #317.
        //
        // A real `assert!` (not `debug_assert!`): now that analytical models can be
        // BOTH AD-eligible AND carry a modeled dose (#383/#394), this path is
        // genuinely reachable, and `debug_assert!` is compiled out of `autodiff`
        // *release* builds — so only an `assert!` makes the FD-gate's invariant hold
        // across every build config (debug/release × FD/AD). It is the backstop to
        // the primary `resolve_gradient_method` gate, not the first line of defence.
        assert!(
            subject.all_doses_fixed(),
            "modeled-RATE dose reached the AD path"
        );
        let events = build_sorted_events(subject, dose_lagtimes);
        let n_events = events.len();
        let lag = |k: usize| -> f64 { dose_lagtimes.get(k).copied().unwrap_or(0.0) };

        let mut event_times = Vec::with_capacity(n_events);
        let mut event_kinds = Vec::with_capacity(n_events);
        let mut event_orig_idx_f64 = Vec::with_capacity(n_events);
        for (t, k, idx) in events {
            event_times.push(t);
            event_kinds.push(k);
            event_orig_idx_f64.push(idx as f64);
        }

        // Per-event reset floor. Walk the sorted events tracking the most
        // recent reset time; each event records the floor in effect *before*
        // it (resets at strictly earlier positions). The reset event's own
        // interval still uses the previous floor — its infusions are turned
        // off only from the next interval on, exactly as the non-AD path
        // `continue`s past a reset after setting `reset_floor = ev.time`.
        let mut event_reset_floor = Vec::with_capacity(n_events);
        let mut running_floor = f64::NEG_INFINITY;
        for i in 0..n_events {
            event_reset_floor.push(running_floor);
            if event_kinds[i] > 2.5 {
                running_floor = event_times[i];
            }
        }

        Self {
            event_times,
            event_kinds,
            event_orig_idx_f64,
            event_reset_floor,
            // Lagged start times — the propagators' infusion-window overlap
            // must use the same `time + lag` as the merged-timeline events.
            dose_times: subject
                .doses
                .iter()
                .enumerate()
                .map(|(k, d)| d.time + lag(k))
                .collect(),
            dose_amts: subject.doses.iter().map(|d| d.amt).collect(),
            dose_rates: subject.doses.iter().map(|d| d.rate).collect(),
            dose_durations: subject.doses.iter().map(|d| d.duration).collect(),
            dose_cmts_f64: subject.doses.iter().map(|d| d.cmt as f64).collect(),
        }
    }
}

/// Per-event tv arrays (length `n_events * n_tv`, row-major). Built by
/// evaluating `model.tv_fn` against each event's covariate snapshot.
/// Order must match `FlatEventData::event_*` arrays.
pub struct FlatEventTv {
    pub tv: Vec<f64>,
    pub n_tv: usize,
    pub n_events: usize,
}

impl FlatEventTv {
    /// `dose_lagtimes` must match the slice passed to
    /// [`FlatEventData::from_subject`] so the two timelines sort identically
    /// (a lagged dose may re-order relative to obs).
    pub fn from_subject(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        dose_lagtimes: &[f64],
    ) -> Self {
        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("FlatEventTv::from_subject: model.tv_fn required for AD path");

        // Same sorted timeline FlatEventData walks — built by the shared helper
        // so the per-event tv rows stay aligned with the event_* arrays.
        let events = build_sorted_events(subject, dose_lagtimes);
        let n_events = events.len();
        let n_tv = model.pk_idx_f64.len();

        let mut tv = Vec::with_capacity(n_events * n_tv);
        for (_, kind_tag, orig) in &events {
            // Covariate snapshot per event kind (0=dose, 1=obs, 2=pk-only).
            // Reset events (3) carry no per-record snapshot — use the
            // subject-static map (LOCF-correct for time-constant covariates).
            // Their PK params only drive the propagation into the reset, whose
            // result is immediately zeroed, so the exact value never reaches a
            // prediction; a valid (finite, positive) row just avoids NaN.
            let cov = if *kind_tag < 0.5 {
                subject.dose_cov(*orig)
            } else if *kind_tag < 1.5 {
                subject.obs_cov(*orig)
            } else if *kind_tag < 2.5 {
                subject.pk_only_cov(*orig)
            } else {
                &subject.covariates
            };
            let row = tv_fn(theta, cov);
            assert_eq!(row.len(), n_tv, "tv_fn returned wrong length");
            tv.extend_from_slice(&row);
        }

        Self { tv, n_tv, n_events }
    }
}

// ─────────────────────────────────────────────────────────────────────
// AD-instrumented core: reverse-mode for the gradient w.r.t. eta.
// ─────────────────────────────────────────────────────────────────────

#[autodiff_reverse(
    individual_nll_event_driven_ad_grad,
    Duplicated, // eta
    Const,      // tv_per_event
    Const,      // omega_inv_flat
    Const,      // log_det_omega
    Const,      // sigma_values
    Const,      // event_times
    Const,      // event_kinds
    Const,      // event_orig_idx_f64
    Const,      // event_reset_floor
    Const,      // dose_times
    Const,      // dose_amts
    Const,      // dose_rates
    Const,      // dose_durations
    Const,      // dose_cmts_f64
    Const,      // observations
    Const,      // cens_f64
    Const,      // pk_idx_f64
    Const,      // sel_flat
    Const,      // pk_and_err_model
    Const,      // obs_scale
    Active      // return
)]
pub fn individual_nll_event_driven_ad(
    eta: &[f64],
    tv_per_event: &[f64], // n_events * n_tv (row-major)
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    event_times: &[f64],
    event_kinds: &[f64], // 0=dose, 1=obs, 2=pk-only, 3=reset
    event_orig_idx_f64: &[f64],
    // Per-event reset floor (length n_events): time of the most recent reset
    // strictly before each event, else NEG_INFINITY. Infusions starting before
    // it are masked off in the propagators. Const w.r.t. eta.
    event_reset_floor: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    dose_cmts_f64: &[f64],
    observations: &[f64],
    cens_f64: &[f64],
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    pk_and_err_model: f64,
    // Per-event divisor (length = n_events). Non-obs slots must be 1.0
    // so the `is_obs` mask on `data_ll` doesn't trip on NaN/0 in the
    // non-obs entries.
    obs_scale: &[f64],
) -> f64 {
    let n_eta = eta.len();
    let n_tv = pk_idx_f64.len();
    let n_events = event_times.len();
    let n_doses = dose_times.len();
    // +100 packs LTBS (see `ad_gradients::individual_nll_ad`): the prediction is
    // log-wrapped and the error model is additive on the log scale.
    let ltbs = (pk_and_err_model as i32) >= 100;
    let base = (pk_and_err_model as i32) % 100;
    let pk_model_id = base / 10;
    let error_model_id = base % 10;

    // ── η prior: η' Ω⁻¹ η ───────────────────────────────────────────
    let mut eta_prior = 0.0;
    for i in 0..n_eta {
        for j in 0..n_eta {
            eta_prior += eta[i] * omega_inv_flat[i * n_eta + j] * eta[j];
        }
    }

    // ── State + walk events ─────────────────────────────────────────
    // State held as scalars (not `[f64; N]`) — array initialization emits
    // `llvm.memset`, and the `&mut [f64;N]` through propagator boundary
    // tripped Enzyme reverse-mode type deduction. Up to 4 slots cover all
    // analytical PK models (3-cpt oral: depot, central, periph1, periph2).
    let mut state0 = 0.0_f64;
    let mut state1 = 0.0_f64;
    let mut state2 = 0.0_f64;
    let mut state3 = 0.0_f64;

    let mut cur_t = if n_events > 0 { event_times[0] } else { 0.0 };

    let mut data_ll = 0.0_f64;

    for ev_idx in 0..n_events {
        let t_ev = event_times[ev_idx];

        // Compute PK params at THIS event from per-event tv row + eta.
        // NONMEM convention (end-of-interval / current-record): the
        // params at event[i] govern the propagation [event[i-1], event[i]].
        // For the first event the propagation has dt = 0, so the values
        // are unused and the loop is well-defined regardless of where
        // the params are evaluated.
        let mut ev_cl = 0.0_f64;
        let mut ev_v = 0.0_f64;
        let mut ev_q = 0.0_f64;
        let mut ev_v2 = 0.0_f64;
        let mut ev_ka = 0.0_f64;
        let mut ev_q3 = 0.0_f64;
        let mut ev_v3 = 0.0_f64;
        // F (bioavailability) defaults to 1.0 — matches PkParams::default()
        // and the analytical event-driven path. If the model doesn't
        // declare F as an individual parameter, the loop below never
        // overwrites this and the bolus / infusion scaling reduces to a
        // no-op multiplication by 1.
        let mut ev_f = 1.0_f64;
        let row_off = ev_idx * n_tv;
        for i in 0..n_tv {
            let mut eta_contrib = 0.0;
            for j in 0..n_eta {
                eta_contrib += sel_flat[i * n_eta + j] * eta[j];
            }
            let val = tv_per_event[row_off + i] * eta_contrib.exp();
            let idx = pk_idx_f64[i] as usize;
            // Const-branch fan-out (pk_idx_f64 is Const).
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

        // Always call the propagator with the current event's pk —
        // when dt = 0 the math is a no-op (exp(0) = 1, infusion
        // contributions reduce to e^x - e^x = 0). Branching on
        // `dt > 0.0` here would create a phi-node on `state` that
        // Enzyme can't type-deduce in reverse mode.
        let (s0_new, s1_new, s2_new, s3_new) = propagate_state_ad(
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

        // ── Reset branch (EVID=3/4, kind=3.0). Zero every compartment after
        // propagating into the reset time — the interval's drug is discarded,
        // matching `pk::event_driven`'s reset that zeros state and `continue`s.
        // `keep` is a Const mask (0 at a reset, 1 elsewhere) so the multiply is
        // phi-free and Enzyme-safe. is_dose / is_obs are both 0 for a reset
        // (kind=3.0), so the dose and obs branches below skip it naturally.
        let keep = if kind > 2.5 { 0.0 } else { 1.0 };
        state0 *= keep;
        state1 *= keep;
        state2 *= keep;
        state3 *= keep;
        // `orig` is an index into doses (kind=0), obs (kind=1), or
        // pk_only events (kind=2). For each side-array access we
        // clamp to a safe fallback index when the event isn't of
        // that kind — multiplied by zero downstream so the value
        // doesn't matter, only the address being valid.
        let dose_idx = if kind < 0.5 { orig } else { 0 };
        let obs_idx = if kind > 0.5 && kind < 1.5 { orig } else { 0 };

        // ── Dose branch. is_dose=0 for obs and pk-only events, so
        // their state0 is unchanged regardless of dose_amts/dose_rates
        // values. Const inputs throughout. `ev_f` applies bioavailability
        // F1 at bolus injection — mirrors the analytical event-driven
        // path (`pk::event_driven`) and the single-snapshot AD path
        // (`ad::ad_gradients`). Without it, oral / extravascular subjects
        // with F ≠ 1 silently got wrong AD gradients on TV-covariate fits.
        let is_dose = if kind < 0.5 { 1.0 } else { 0.0 };
        let is_bolus = if dose_rates[dose_idx] == 0.0 {
            1.0
        } else {
            0.0
        };
        state0 += is_dose * is_bolus * dose_amts[dose_idx] * ev_f;

        // ── Observation branch. is_obs=0 for dose and pk-only events.
        let is_obs = if kind > 0.5 && kind < 1.5 { 1.0 } else { 0.0 };

        // Central-compartment slot:
        //   IV models (1- 2- 3-cpt): central = state0
        //   Oral models (1- 2- 3-cpt): central = state1 (state0 is depot)
        // Const branch on pk_model_id constant-folds.
        let central_amt = if pk_model_id == PK_ID_ONE_CPT_ORAL
            || pk_model_id == PK_ID_TWO_CPT_ORAL
            || pk_model_id == PK_ID_THREE_CPT_ORAL
        {
            state1
        } else {
            state0
        };

        // Strictly positive divisor — handles transient ev_v ≤ 0 from
        // line-search trial steps.
        let v_safe = ev_v.abs() + 1e-30;
        let conc_raw = central_amt / v_safe;
        let conc_clamped = (conc_raw + conc_raw.abs()) * 0.5;
        let scaled = conc_clamped / obs_scale[ev_idx];
        // LTBS: compare log(prediction) to the log-scale observation. Explicit-
        // comparison floor (no `f64::max`, per CLAUDE.md).
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

        let v_resid = residual_variance_ad(error_model_id, conc, sigma_values);
        let cens_active = if cens_f64[obs_idx] > 0.5 { 1.0 } else { 0.0 };
        let resid = observations[obs_idx] - conc;
        let z = resid / v_resid.sqrt();
        let bloq_term = -2.0 * log_normal_cdf_ad(z);
        let gaussian_term = resid * resid / v_resid + v_resid.ln();
        let obs_term = cens_active * bloq_term + (1.0 - cens_active) * gaussian_term;
        // For pk-only events is_obs = 0, so no contribution to data_ll.
        // PK params for the next interval are recomputed at the top of
        // the next iteration from `tv_per_event` — no carry-over state.
        data_ll += is_obs * obs_term;

        cur_t = t_ev;
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

// Forward-mode AD path for the Jacobian d(predictions)/d(eta) lives in
// the sibling module `event_driven_ad_jac` with its own private copies
// of the propagators. Putting it in a separate module isolates the
// Enzyme-instrumented call graph: when both AD passes share the same
// helper functions, fat-LTO inlining causes phi-node IR to leak across
// the forward and reverse pipelines and reverse-mode type deduction
// breaks. Sibling-module isolation keeps the two pipelines independent.

// ─────────────────────────────────────────────────────────────────────
// Inlined event-driven propagators (AD-safe).
//
// Each propagator advances `state` from `t_from` to `t_to` under the
// linear ODE governed by (cl, v, q, v2) and adds the contributions of
// any infusions active during the interval. Infusion contribution per
// active window [p_start, p_end] uses the unified formula derived in
// the design notes:
//
//   contribution_to_central(t_to) = (r/ke) * [exp(-ke·τ_to) - exp(-ke·τ_total)]
//
// where τ_to = t_to - p_end, τ_total = t_to - p_start. Both are clamped
// to ensure τ_total ≥ τ_to ≥ 0 so that infusions outside the interval
// contribute exactly zero (formula reduces to e^x - e^x = 0).
// ─────────────────────────────────────────────────────────────────────

/// Returns `(state0_new, state1_new, state2_new, state3_new)`. State is
/// passed by value (no `&mut`) so the function returns scalars rather
/// than mutating an array — Enzyme reverse-mode handles scalar return
/// values cleanly but trips on the memset / mixed-active aliasing of
/// `&mut [f64; N]` here. Slots beyond what the model uses are returned
/// unchanged.
///
#[allow(clippy::too_many_arguments)]
fn propagate_state_ad(
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
    // Most-recent reset time strictly before this interval (or NEG_INFINITY).
    // Infusions whose start is `< reset_floor` are masked off inside the IV
    // propagators. Const w.r.t. eta; oral propagators ignore it (bolus only).
    reset_floor: f64,
) -> (f64, f64, f64, f64) {
    // Const-only branch on pk_model_id — constant-folds under LLVM. Only
    // the relevant arm contains real adjoint flow per build. `f_bio`
    // scales every active infusion's rate inside the IV propagators
    // (mirrors `pk::event_driven::propagate_with_bounds`). Oral
    // propagators take no doses internally — bolus into depot is
    // handled in the main event loop with `f_bio` applied there.
    // Per issue #176, IV bolus and infusion share one model ID (the bolus
    // vs infusion route is picked per dose inside the propagator from
    // RATE), so each IV branch below matches a single ID.
    if pk_model_id == PK_ID_ONE_CPT_IV {
        // 1-cpt IV (bolus and/or infusion): state = [central].
        let s0 = propagate_one_cpt_ad(
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
        // 1-cpt oral: state = [depot, central]. No infusion support.
        let (s0, s1) = propagate_one_cpt_oral_ad(state0, state1, t_from, t_to, cl, v, ka);
        (s0, s1, state2, state3)
    } else if pk_model_id == PK_ID_TWO_CPT_IV {
        // 2-cpt IV (bolus and/or infusion): state = [central, periph].
        let (s0, s1) = propagate_two_cpt_ad(
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
        // 2-cpt oral: state = [depot, central, periph].
        let (s0, s1, s2) =
            propagate_two_cpt_oral_ad(state0, state1, state2, t_from, t_to, cl, v, q, v2, ka);
        (s0, s1, s2, state3)
    } else if pk_model_id == PK_ID_THREE_CPT_IV {
        // 3-cpt IV (bolus and/or infusion): state = [central, periph1, periph2].
        let (s0, s1, s2) = propagate_three_cpt_ad(
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
        // 3-cpt oral: state = [depot, central, periph1, periph2].
        propagate_three_cpt_oral_ad(
            state0, state1, state2, state3, t_from, t_to, cl, v, q, v2, q3, v3, ka,
        )
    } else {
        // Unknown / unsupported — leave state unchanged.
        (state0, state1, state2, state3)
    }
}

/// 1-cpt linear propagator. Branch-free over both (cl, v) and the dose
/// loop: degenerate parameters are guarded by arithmetic clamps, and the
/// per-infusion contribution naturally evaluates to zero for bolus rows
/// (`r = 0`) and outside-the-interval infusions (`p_end ≤ p_start` makes
/// `tau_total = tau_to` so `e^{-ke·τ_to} - e^{-ke·τ_total} = 0`).
#[allow(clippy::too_many_arguments)]
fn propagate_one_cpt_ad(
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
        // Infusions that started before the most recent reset are off
        // (Const mask on dose_times[d], mirrors `reset_floor` in the non-AD
        // path). `s_i < reset_floor` strictly so an infusion starting *at*
        // the reset time (e.g. an EVID=4's own infusion) stays active.
        let active = if s_i < reset_floor { 0.0 } else { 1.0 };
        let contribution = (r / ke) * ((-ke * tau_to).exp() - (-ke * tau_total).exp());
        s0 += active * contribution;
    }
    s0
}

/// 2-cpt linear propagator. Returns `(state0_new, state1_new)`.
/// Per-channel A_ss for input b = (b1, b2):
///   A_ss[0] = (b1 + b2) · v1 / cl
///   A_ss[1] = b1 · v2 / cl + b2 · (cl + q) · v2 / (cl · q)
/// `dose_cmts_f64` routes each dose's rate to channel 1 (cmt=1) or 2 (cmt=2).
#[allow(clippy::too_many_arguments)]
fn propagate_two_cpt_ad(
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

    // Homogeneous decomposition into eigenmodes.
    let s_homog = state1 / k12;
    let denom = beta - alpha;
    let c1 = (state0 - s_homog * (k21 - beta)) / denom;
    let c2 = s_homog - c1;

    let mut s0 = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let mut s1 = (c1 * e_alpha + c2 * e_beta) * k12;

    // Infusion contributions, per dose. dose_cmts_f64 routes each
    // dose's rate to channel 1 (central) or channel 2 (peripheral).
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

        // Per-channel A_ss. `f_bio` scales the active infusion rate so a
        // dur→0 infusion limits to a bolus of amount F·AMT — same
        // convention as the bolus injection step in the event loop.
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

// ─── Oral models (AD) ──────────────────────────────────────────────

/// 1-cpt oral propagator. State = `(depot, central)`. Bolus only —
/// infusion-into-oral isn't supported (dispatcher skips this branch).
/// The L'Hôpital limit at ka == ke is handled by a small offset added
/// to `ke` so the formula stays smooth (and AD-friendly); the bias is
/// O(eps) in the answer.
#[allow(clippy::too_many_arguments)]
fn propagate_one_cpt_oral_ad(
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

    // Bateman: a_c(t) = a_c(0)·e^{-ke·t} + (ka·a_d(0)/(ke-ka))·(e^{-ka·t}-e^{-ke·t})
    // To stay AD-safe we never branch on `(ka-ke).abs() < eps`. Instead
    // add a small constant offset so the denominator is bounded away
    // from zero. Worst-case bias is O(eps) — acceptable since the
    // exact L'Hôpital limit is the analytic continuation of the same
    // formula and any sane optimizer steers away from ka = ke anyway.
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

/// 2-cpt oral propagator. State = `(depot, central, periph)`. Bolus only.
#[allow(clippy::too_many_arguments)]
fn propagate_two_cpt_oral_ad(
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

    // Particular solution amplitude (A, B) for input ka·A_d(0)·e^{-ka·t}:
    //   A = ka·A_d(0)·(k21-ka) / [(ka-α)(ka-β)]
    //   B = k12·A / (k21-ka)
    let denom_depot = (ka_safe - alpha) * (ka_safe - beta);
    let cap_a = ka_safe * state0 * (k21 - ka_safe) / denom_depot;
    let cap_b = ka_safe * state0 * k12 / denom_depot;

    // Homogeneous initial conditions = state - particular_at_t0.
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

// ─── 3-cpt models (AD) ─────────────────────────────────────────────

/// 3-cpt macro rates (α > β > γ), trigonometric (Vieta) method. AD-safe
/// — same shape as `ad_gradients::macro_rates_three_cpt_ad`.
fn macro_rates_three_ad(
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
    // p must be ≤ -ε for the cubic to have three real roots — clamp via
    // `min(p, -ε)` arithmetically: p_safe = (p + (-ε) - |p - (-ε)|) / 2 = min(p, -ε).
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
    // Branch-free max/min via explicit comparisons (CLAUDE.md: no
    // `f64::max` in AD-instrumented code).
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

/// One 3-cpt eigenmode contribution. Robust eigenvector form (no
/// division by `(k21-μ)` or `(k31-μ)`): see `pk::event_driven`'s
/// `three_cpt_mode` for the rationale.
#[allow(clippy::too_many_arguments)]
fn three_cpt_mode_ad(
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
    // Strictly-positive guard so the divide is safe even at degenerate
    // points (extremely rare for physical params).
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

/// 3-cpt linear propagator (IV bolus / infusion). Spectral decomposition
/// over (α, β, γ) eigenmodes; constant infusion into central, periph1,
/// or periph2 handled via the steady-state + homogeneous decomposition.
/// Channel-specific A_ss formulas: see the analytical
/// `propagate_three_cpt` in `pk::event_driven` for the derivation.
#[allow(clippy::too_many_arguments)]
fn propagate_three_cpt_ad(
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
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_ad(cl, v1, q, v2, q3, v3);
    let k12 = q_safe / v1_safe;
    let k13 = q3_safe / v1_safe;

    let dt = t_to - t_from;

    // Homogeneous evolution from initial state.
    let (ca, p1a, p2a) = three_cpt_mode_ad(alpha, state0, state1, state2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode_ad(beta, state0, state1, state2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode_ad(gamma, state0, state1, state2, k12, k13, k21, k31, dt);

    let mut s0 = ca + cb + cg;
    let mut s1 = p1a + p1b + p1g;
    let mut s2 = p2a + p2b + p2g;

    // Infusion contributions, per dose. The dispatcher routes by
    // dose_cmts_f64[d]: 1=central, 2=periph1, 3=periph2 (3-cpt IV).
    // Other compartments yield zero contribution (handled via Const
    // branches on dose_cmts_f64, all f64 and Const so phi-safe).
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

        // Per-channel A_ss. Const-branch on dose_cmts_f64 selects which
        // channel formula applies. `f_bio` scales the active infusion
        // rate so dur→0 limits to a bolus of amount F·AMT.
        let cmt = dose_cmts_f64[d] as i32;
        let r = f_bio * dose_rates[d];
        // Infusions started before the most recent reset are off (Const mask).
        let active = if s_i < reset_floor { 0.0 } else { 1.0 };
        let (a_ss_c, a_ss_p1, a_ss_p2) = if cmt == 2 {
            // Channel 2 (periph1).
            (
                r * v1_safe / cl_safe,
                r * (cl_safe + q_safe) * v2_safe / (cl_safe * q_safe),
                r * v3_safe / cl_safe,
            )
        } else if cmt == 3 {
            // Channel 3 (periph2).
            (
                r * v1_safe / cl_safe,
                r * v2_safe / cl_safe,
                r * (cl_safe + q3_safe) * v3_safe / (cl_safe * q3_safe),
            )
        } else {
            // Default = channel 1 (central).
            (
                r * v1_safe / cl_safe,
                r * v2_safe / cl_safe,
                r * v3_safe / cl_safe,
            )
        };

        // Mode contribution at τ_to (`_to`) and τ_total (`_tot`) — subtract.
        let (ca_to, p1a_to, p2a_to) =
            three_cpt_mode_ad(alpha, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (cb_to, p1b_to, p2b_to) =
            three_cpt_mode_ad(beta, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (cg_to, p1g_to, p2g_to) =
            three_cpt_mode_ad(gamma, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_to);
        let (ca_tot, p1a_tot, p2a_tot) = three_cpt_mode_ad(
            alpha, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );
        let (cb_tot, p1b_tot, p2b_tot) = three_cpt_mode_ad(
            beta, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );
        let (cg_tot, p1g_tot, p2g_tot) = three_cpt_mode_ad(
            gamma, a_ss_c, a_ss_p1, a_ss_p2, k12, k13, k21, k31, tau_total,
        );

        s0 += active * ((ca_to - ca_tot) + (cb_to - cb_tot) + (cg_to - cg_tot));
        s1 += active * ((p1a_to - p1a_tot) + (p1b_to - p1b_tot) + (p1g_to - p1g_tot));
        s2 += active * ((p2a_to - p2a_tot) + (p2b_to - p2b_tot) + (p2g_to - p2g_tot));
    }
    (s0, s1, s2)
}

/// 3-cpt oral propagator. State = `(depot, central, periph1, periph2)`.
/// Bolus into depot only. Depot decays independently; the (central,
/// p1, p2) subsystem follows 3-cpt homogeneous evolution from
/// `state - particular_at_t0` plus a depot-driven particular solution
/// of the form `(A, B, C)·exp(-ka·t)`.
#[allow(clippy::too_many_arguments)]
fn propagate_three_cpt_oral_ad(
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
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_ad(cl, v1, q, v2, q3, v3);
    let k12 = q_safe / v1_safe;
    let k13 = q3_safe / v1_safe;

    let dt = t_to - t_from;
    let e_ka = (-ka_safe * dt).exp();

    // Particular: X·e^{-ka·t} where (see `pk::event_driven`'s 3-cpt-oral
    // derivation) the leading negative sign comes from the cubic
    // characteristic polynomial of K.
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

    let (ca, p1a, p2a) = three_cpt_mode_ad(alpha, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode_ad(beta, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode_ad(gamma, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);

    let new_s0 = state0 * e_ka;
    let new_s1 = ca + cb + cg + cap_a * e_ka;
    let new_s2 = p1a + p1b + p1g + cap_b * e_ka;
    let new_s3 = p2a + p2b + p2g + cap_c * e_ka;
    (new_s0, new_s1, new_s2, new_s3)
}

// ─── AD-safe special functions (re-exports from ad_gradients) ───────

/// Local copy of `erf_ad` so this module compiles independently of the
/// instrumented function in `ad_gradients`. Same Abramowitz & Stegun
/// 7.1.26 polynomial; see comment there for AD-safety rationale.
fn erf_ad(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn log_normal_cdf_ad(z: f64) -> f64 {
    const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    const MIN_PROB: f64 = 1e-300;

    if z > -5.0 {
        let p = 0.5 * (1.0 + erf_ad(z * INV_SQRT_2));
        let p_floor = if p < MIN_PROB { MIN_PROB } else { p };
        p_floor.ln()
    } else {
        let log_phi = -0.5 * z * z - LOG_SQRT_2PI;
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2 - 15.0 * inv_z2 * inv_z2 * inv_z2;
        log_phi - (-z).ln() + series.ln()
    }
}

fn residual_variance_ad(error_model_id: i32, f_pred: f64, sigma: &[f64]) -> f64 {
    let v = match error_model_id {
        0 => sigma[0] * sigma[0],
        1 => {
            let fs = f_pred * sigma[0];
            fs * fs
        }
        2 => {
            let p = f_pred * sigma[0];
            p * p + sigma[1] * sigma[1]
        }
        _ => sigma[0] * sigma[0],
    };
    if v < 1e-12 {
        1e-12
    } else {
        v
    }
}

// ─── Public wrappers ────────────────────────────────────────────────

/// True when this PK model has an event-driven AD implementation in
/// this module. Caller-side dispatch in `inner_optimizer.rs` uses this
/// to decide whether to take the AD fast path or fall back to FD.
pub fn supports_event_driven_ad(pk_model: PkModel) -> bool {
    // Now covers all analytical PK models — see the `propagate_state_ad`
    // dispatch.
    matches!(
        pk_model,
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    )
}

/// Compute (NLL, ∂NLL/∂η) using the event-driven AD path with per-event
/// covariate snapshots.
///
/// Pads `observations` / `cens_f64` / `dose_*` arrays with single-element
/// placeholders when the subject has zero in either dimension. The masks
/// inside the AD kernel zero out their contribution, but the indices
/// must still be in-bounds — without padding a no-obs subject would
/// panic on `cens_f64[0]`.
#[allow(clippy::too_many_arguments)]
pub fn compute_nll_gradient_event_driven_ad(
    eta: &[f64],
    tv_per_event: &FlatEventTv,
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    event_data: &FlatEventData,
    observations: &[f64],
    cens_f64: &[f64],
    pk_model: PkModel,
    error_model: ErrorModel,
    pk_idx_f64: &[f64],
    sel_flat: &[f64],
    // Per-event divisor (length = event_data.event_times.len()). The
    // caller pads non-obs entries to 1.0 so the `is_obs` likelihood mask
    // doesn't get corrupted by NaN/0 reads.
    obs_scale: &[f64],
    log_transform: bool,
) -> (f64, Vec<f64>) {
    let n_eta = eta.len();
    let mut d_eta = vec![0.0_f64; n_eta];

    // +100 packs LTBS (see `ad_gradients::individual_nll_ad`); under LTBS the
    // error model is additive (id 0) on the log scale.
    let ltbs_offset = if log_transform { 100 } else { 0 };
    let pk_and_err = (crate::ad::ad_gradients::pk_model_to_id(pk_model) * 10
        + crate::ad::ad_gradients::error_model_to_id(error_model)
        + ltbs_offset) as f64;

    // Pad zero-length arrays so the AD kernel's masked-but-still-evaluated
    // index loads stay in-bounds. The mask multiplies the read by 0 so
    // the value doesn't matter, only the address being valid.
    let observations_padded: &[f64] = if observations.is_empty() {
        &[0.0]
    } else {
        observations
    };
    let cens_padded: &[f64] = if cens_f64.is_empty() {
        &[0.0]
    } else {
        cens_f64
    };
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

    let nll = individual_nll_event_driven_ad_grad(
        eta,
        &mut d_eta,
        &tv_per_event.tv,
        omega_inv_flat,
        log_det_omega,
        sigma_values,
        &event_data.event_times,
        &event_data.event_kinds,
        &event_data.event_orig_idx_f64,
        &event_data.event_reset_floor,
        dose_times_padded,
        dose_amts_padded,
        dose_rates_padded,
        dose_durations_padded,
        dose_cmts_padded,
        observations_padded,
        cens_padded,
        pk_idx_f64,
        sel_flat,
        pk_and_err,
        obs_scale,
        1.0,
    );

    (nll, d_eta)
}

// `compute_jacobian_event_driven_ad` lives in the sibling module
// `event_driven_ad_jac` — see top-of-file note about AD-pass isolation.
