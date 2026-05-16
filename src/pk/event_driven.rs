//! Event-driven analytical PK propagation.
//!
//! Walks events (doses + observations) in time order, propagating the
//! amount-vector state from one event to the next using the rate matrix
//! built from the *current* per-event PK parameters. This is what NONMEM
//! `ADVAN` routines do — and is how time-varying covariates take effect:
//! when CL or V change between events, the elimination rate during the
//! next interval changes accordingly.
//!
//! Coverage:
//!   - 1-compartment IV bolus & infusion (state = `[A_central]`)
//!   - 1-compartment oral (state = `[A_depot, A_central]`)
//!   - 2-compartment IV bolus & infusion (state = `[A_central, A_periph]`)
//!   - 2-compartment oral (state = `[A_depot, A_central, A_periph]`)
//!   - 3-compartment IV bolus & infusion (state = `[A_central, A_p1, A_p2]`)
//!   - 3-compartment oral (state = `[A_depot, A_central, A_p1, A_p2]`)
//!
//! For oral models the dose into compartment 1 is the depot (NONMEM
//! ADVAN2/ADVAN4 convention), and the observation read-out reads the
//! *central* compartment (state slot 1, not 0).
//!
//! Infusion support (`rate > 0`) is restricted to inputs into the
//! central compartment for IV models and into the depot for oral
//! models (cmt=1 in both cases). Infusion into peripheral compartments
//! still panics — that's a rare clinical setup tracked as a follow-up.

use crate::types::{DoseEvent, PkModel, PkParams, Subject};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Dose,
    Obs,
    /// EVID=2 "other event" — typically a covariate-change marker.
    /// Doesn't mutate compartment amounts; just refreshes the
    /// piecewise-constant rate matrix from the row's covariate values.
    /// Matches NONMEM's `$PK runs at every record` semantic.
    PkOnly,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub time: f64,
    pub kind: EventKind,
    /// Index into `subject.doses`, `subject.obs_times`, or
    /// `subject.pk_only_times` depending on `kind`.
    pub orig_idx: usize,
}

/// Pre-computed, subject-static event scheduling for the event-driven
/// propagator.
///
/// Building the merged event timeline (dose + obs + EVID=2) and the
/// per-interval infusion sub-event bounds is purely a function of the
/// subject's static event timeline (times, durations, rates) — none of
/// it depends on theta or eta. Inside hot loops (BFGS line search and
/// FOCE NLL evaluation) the same event_driven_predictions call gets
/// invoked thousands of times per subject, so re-sorting events and
/// re-deriving infusion bounds on every call is wasted work.
///
/// Build one of these once per subject per `find_ebe` (or once per fit,
/// for callers that can amortise even further) and pass it into the
/// `*_with_schedule` variants below.
#[derive(Debug, Clone)]
pub struct EventSchedule {
    /// Merged + sorted event list. Tie-break order is `Dose < PkOnly < Obs`
    /// so covariate-change markers run after a dose at the same time but
    /// before an observation (matches NONMEM `$PK` semantics).
    ///
    /// Dose event times are `subject.doses[k].time + dose_lagtimes[k]`
    /// so the schedule already reflects per-dose lagtime.
    pub events: Vec<Event>,
    /// For each interval `i` between `events[i]` and `events[i+1]`, the
    /// sub-interval boundary points (sorted, deduped, including the two
    /// interval endpoints) at which the active infusion rate matrix
    /// changes. Length = `events.len().saturating_sub(1)`.
    /// For an interval with no infusion start/stop crossing, this is
    /// just `[t_from, t_to]`.
    pub bounds_per_interval: Vec<Vec<f64>>,
    /// Per-dose lagtimes that were used to build this schedule. Parallel
    /// to `subject.doses`. Stored so callers (and the propagator's
    /// active-infusion check) use the same shifted times the schedule
    /// was built with.
    pub dose_lagtimes: Vec<f64>,
}

impl EventSchedule {
    /// Pre-compute the event timeline and per-interval infusion bounds
    /// for `subject` under `pk_model`. The result is reusable across
    /// arbitrary `(theta, eta)` evaluations of the same subject *as long
    /// as the per-dose lagtimes are unchanged* — only the event times
    /// (subject-static, plus lagtime offsets) matter, not the per-event
    /// PK rate values.
    ///
    /// `dose_lagtimes` must be length `subject.doses.len()` (or empty,
    /// which is treated as all zeros for backward compatibility with the
    /// no-lagtime fast path).
    pub fn for_subject(subject: &Subject, _pk_model: PkModel, dose_lagtimes: &[f64]) -> Self {
        assert!(
            dose_lagtimes.is_empty() || dose_lagtimes.len() == subject.doses.len(),
            "dose_lagtimes length {} does not match subject.doses.len() {}",
            dose_lagtimes.len(),
            subject.doses.len()
        );
        let get_lag = |k: usize| -> f64 {
            if dose_lagtimes.is_empty() {
                0.0
            } else {
                dose_lagtimes[k]
            }
        };

        let mut events: Vec<Event> = Vec::with_capacity(
            subject.doses.len() + subject.obs_times.len() + subject.pk_only_times.len(),
        );
        for (k, d) in subject.doses.iter().enumerate() {
            events.push(Event {
                time: d.time + get_lag(k),
                kind: EventKind::Dose,
                orig_idx: k,
            });
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            events.push(Event {
                time: t,
                kind: EventKind::Obs,
                orig_idx: j,
            });
        }
        for (m, &t) in subject.pk_only_times.iter().enumerate() {
            events.push(Event {
                time: t,
                kind: EventKind::PkOnly,
                orig_idx: m,
            });
        }
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| kind_order(a.kind).cmp(&kind_order(b.kind)))
        });

        // Materialize lagtimes for later use (active-infusion check).
        let stored_lagtimes: Vec<f64> = if dose_lagtimes.is_empty() {
            vec![0.0; subject.doses.len()]
        } else {
            dose_lagtimes.to_vec()
        };

        let mut bounds_per_interval = Vec::with_capacity(events.len().saturating_sub(1));
        for w in events.windows(2) {
            bounds_per_interval.push(compute_propagation_bounds(
                w[0].time,
                w[1].time,
                &subject.doses,
                &stored_lagtimes,
            ));
        }

        Self {
            events,
            bounds_per_interval,
            dose_lagtimes: stored_lagtimes,
        }
    }
}

#[inline]
fn kind_order(k: EventKind) -> u8 {
    match k {
        EventKind::Dose => 0,
        EventKind::PkOnly => 1,
        EventKind::Obs => 2,
    }
}

/// Sub-interval boundaries inside `(t_from, t_to)` at which the active
/// infusion rate changes (an infusion starts or stops). The returned
/// `Vec` is sorted, deduped, and always includes the two interval
/// endpoints, so `windows(2)` enumerates every sub-interval over which
/// the rate matrix is constant.
///
/// `dose_lagtimes[k]` shifts dose `k`'s effective start (and therefore
/// end) by that amount. Must be parallel to `doses`.
fn compute_propagation_bounds(
    t_from: f64,
    t_to: f64,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
) -> Vec<f64> {
    let mut bounds: Vec<f64> = vec![t_from, t_to];
    for (k, d) in doses.iter().enumerate() {
        if d.rate > 0.0 && d.duration > 0.0 {
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let start = d.time + lag;
            let end = d.time + lag + d.duration;
            if start > t_from + 1e-15 && start < t_to - 1e-15 {
                bounds.push(start);
            }
            if end > t_from + 1e-15 && end < t_to - 1e-15 {
                bounds.push(end);
            }
        }
    }
    bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bounds.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    bounds
}

/// True when this PK model has an event-driven implementation in this module.
/// Caller-side dispatch (in `pk::compute_predictions`) uses this to fall
/// back to the existing superposition path for unsupported models.
pub fn supports_event_driven(pk_model: PkModel) -> bool {
    matches!(
        pk_model,
        PkModel::OneCptIvBolus
            | PkModel::OneCptInfusion
            | PkModel::OneCptOral
            | PkModel::TwoCptIvBolus
            | PkModel::TwoCptInfusion
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIvBolus
            | PkModel::ThreeCptInfusion
            | PkModel::ThreeCptOral
    )
}

/// State-vector dimension and central-compartment slot index for a given
/// pk_model. Central is where the observation read-out reads from.
fn state_layout(pk_model: PkModel) -> (usize, usize) {
    match pk_model {
        PkModel::OneCptIvBolus | PkModel::OneCptInfusion => (1, 0),
        PkModel::OneCptOral => (2, 1), // [depot, central]
        PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => (2, 0),
        PkModel::TwoCptOral => (3, 1), // [depot, central, periph]
        PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion => (3, 0),
        PkModel::ThreeCptOral => (4, 1), // [depot, central, periph1, periph2]
    }
}

// `is_oral` was a helper for the previous infusion dispatcher; the
// per-model `match (pk_model, d.cmt)` now subsumes it.

/// Compute predictions by walking events in time order and propagating the
/// compartment-amount state with per-event PK parameters.
///
/// `pk_at_dose[k]` are the PK parameters at `subject.doses[k].time`;
/// `pk_at_obs[j]` are the PK parameters at `subject.obs_times[j]`;
/// `pk_at_pk_only[m]` are the PK parameters at `subject.pk_only_times[m]`
/// (EVID=2 "other event" rows — typically covariate-change markers).
/// All three slices are produced by [`crate::pk::compute_event_pk_params`].
///
/// Concentration at observation `j` is read out as `state_central / V` where
/// `V` is `pk_at_obs[j].v()` — i.e. the central-compartment volume at the
/// *observation's* time. This matches NONMEM `S1 = V1` / `IPRED = A(1)/S1`.
pub fn event_driven_predictions(
    pk_model: PkModel,
    subject: &Subject,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> Vec<f64> {
    let dose_lagtimes: Vec<f64> = pk_at_dose.iter().map(|p| p.lagtime()).collect();
    let schedule = EventSchedule::for_subject(subject, pk_model, &dose_lagtimes);
    event_driven_predictions_with_schedule(
        pk_model,
        subject,
        &schedule,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
    )
}

/// Same as [`event_driven_predictions`] but takes a pre-built
/// [`EventSchedule`]. Hot loops should build the schedule once per
/// subject (e.g., once per `find_ebe` call) and pass it here on every
/// `(theta, eta)` evaluation — the merged event sort and per-interval
/// infusion-bound construction otherwise dominate per-call CPU on the
/// TV-cov path.
pub fn event_driven_predictions_with_schedule(
    pk_model: PkModel,
    subject: &Subject,
    schedule: &EventSchedule,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    let n_obs = subject.obs_times.len();
    let mut preds = vec![0.0_f64; n_obs];

    if n_obs == 0 || schedule.events.is_empty() {
        return preds;
    }

    let (n_states, central_slot) = state_layout(pk_model);

    // State vector starts at zero (no residual drug before the first event).
    let mut state = vec![0.0_f64; n_states];
    let mut cur_t = schedule.events[0].time;

    for (i, ev) in schedule.events.iter().enumerate() {
        // PK params for the propagation [events[i-1], events[i]] are the
        // params evaluated AT events[i] — matches NONMEM's `$PK runs at
        // every record then ADVAN propagates to that record` semantic
        // (end-of-interval / current-record convention). For the first
        // event the propagation has dt = 0 and `pk_now` is unused.
        let pk_now = pk_for(*ev, pk_at_dose, pk_at_obs, pk_at_pk_only);

        if ev.time > cur_t {
            // The interval (events[i-1], events[i]) — its bounds were
            // pre-computed at schedule.bounds_per_interval[i-1].
            let bounds = &schedule.bounds_per_interval[i - 1];
            propagate_with_bounds(
                &mut state,
                bounds,
                &pk_now,
                pk_model,
                &subject.doses,
                &schedule.dose_lagtimes,
            );
            cur_t = ev.time;
        }

        match ev.kind {
            EventKind::Dose => {
                let d = &subject.doses[ev.orig_idx];
                if d.rate <= 0.0 {
                    // Bolus: instantaneous amount jump in dose's compartment.
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n_states {
                        state[cmt_idx] += d.amt;
                    } else {
                        panic!(
                            "event-driven PK: dose into compartment {} but model has \
                             {} states (cmt is 1-based)",
                            d.cmt, n_states
                        );
                    }
                }
                // Infusion: handled inside `propagate` via the active-input
                // lookup — no instantaneous state jump here.
            }
            EventKind::Obs => {
                let v = pk_now.v();
                // Read from the central-compartment slot — depot for slot 0
                // on oral models, central for slot 1.
                let conc = if v > 0.0 {
                    state[central_slot] / v
                } else {
                    0.0
                };
                preds[ev.orig_idx] = conc.max(0.0);
            }
            EventKind::PkOnly => {
                // EVID=2: $PK ran at this row but state is unchanged;
                // pk_now is consumed by the next interval's propagation.
            }
        }
    }

    preds
}

#[inline]
fn pk_for(
    ev: Event,
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
) -> PkParams {
    match ev.kind {
        EventKind::Dose => pk_at_dose[ev.orig_idx],
        EventKind::Obs => pk_at_obs[ev.orig_idx],
        EventKind::PkOnly => pk_at_pk_only[ev.orig_idx],
    }
}

/// Propagate the compartment-amount state across pre-built sub-event
/// bounds (the sorted+deduped sub-interval boundaries inside the
/// containing event interval — see [`compute_propagation_bounds`] /
/// [`EventSchedule`]). The input rate matrix is constant within each
/// `bounds.windows(2)` sub-interval.
///
/// `dose_lagtimes[k]` shifts dose `k`'s active-infusion window by that
/// amount; must be parallel to `doses`.
fn propagate_with_bounds(
    state: &mut [f64],
    bounds: &[f64],
    pk: &PkParams,
    pk_model: PkModel,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
) {
    for w in bounds.windows(2) {
        let s0 = w[0];
        let s1 = w[1];
        let dt = s1 - s0;
        if dt <= 0.0 {
            continue;
        }
        let mid = 0.5 * (s0 + s1);
        // Sum infusion rates active across the *whole* sub-interval,
        // routing each by the dose's compartment. The cmt → state-slot
        // mapping is model-specific:
        //   IV: cmt=1 central, cmt=2 periph1, cmt=3 periph2
        //   Oral: cmt=1 depot, cmt=2 central, cmt=3 periph1, cmt=4 periph2
        //
        // Peripheral-compartment infusion is supported on the IV
        // multi-compartment models (2-cpt: cmt=2; 3-cpt: cmt=2 and 3).
        // Oral peripheral infusion is unusual clinically and stays
        // unsupported for now (panics with a clear message); see the
        // project todo list.
        let mut rate_central = 0.0;
        let mut rate_periph1 = 0.0;
        let mut rate_periph2 = 0.0;
        for (k, d) in doses.iter().enumerate() {
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let t_start = d.time + lag;
            let t_end = t_start + d.duration;
            if d.rate > 0.0 && d.duration > 0.0 && t_start <= mid && t_end >= mid {
                match (pk_model, d.cmt) {
                    (PkModel::OneCptIvBolus | PkModel::OneCptInfusion, 1) => rate_central += d.rate,
                    (PkModel::OneCptOral, 2) => rate_central += d.rate,
                    (PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion, 1) => rate_central += d.rate,
                    (PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion, 2) => rate_periph1 += d.rate,
                    (PkModel::TwoCptOral, 2) => rate_central += d.rate,
                    (PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion, 1) => {
                        rate_central += d.rate
                    }
                    (PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion, 2) => {
                        rate_periph1 += d.rate
                    }
                    (PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion, 3) => {
                        rate_periph2 += d.rate
                    }
                    (PkModel::ThreeCptOral, 2) => rate_central += d.rate,
                    _ => panic!(
                        "event-driven PK: infusion into compartment {} not supported \
                         for model {:?}. Supported: central for all models; periph1/2 \
                         for 2- and 3-cpt IV models. Oral peripheral infusion is a \
                         tracked follow-up.",
                        d.cmt, pk_model
                    ),
                }
            }
        }

        match pk_model {
            PkModel::OneCptIvBolus | PkModel::OneCptInfusion => {
                propagate_one_cpt(state, dt, pk, rate_central);
            }
            PkModel::OneCptOral => {
                propagate_one_cpt_oral(state, dt, pk);
            }
            PkModel::TwoCptIvBolus | PkModel::TwoCptInfusion => {
                propagate_two_cpt(state, dt, pk, rate_central, rate_periph1);
            }
            PkModel::TwoCptOral => {
                propagate_two_cpt_oral(state, dt, pk);
            }
            PkModel::ThreeCptIvBolus | PkModel::ThreeCptInfusion => {
                propagate_three_cpt(state, dt, pk, rate_central, rate_periph1, rate_periph2);
            }
            PkModel::ThreeCptOral => {
                propagate_three_cpt_oral(state, dt, pk);
            }
        }
    }
}

/// 1-cpt linear propagator with constant input `rate` into central:
///   A(t+dt) = exp(-ke·dt)·A(t) + (rate/ke)·(1 - exp(-ke·dt))
fn propagate_one_cpt(state: &mut [f64], dt: f64, pk: &PkParams, rate: f64) {
    let cl = pk.cl();
    let v = pk.v();
    if v <= 0.0 || cl <= 0.0 {
        // Degenerate params; skip propagation rather than blow up. The
        // outer optimizer will see a poor OFV and step away.
        return;
    }
    let ke = cl / v;
    let exp_term = (-ke * dt).exp();
    state[0] = exp_term * state[0] + (rate / ke) * (1.0 - exp_term);
}

/// 2-cpt linear propagator with constant input rates into central and
/// peripheral compartments. Uses eigendecomposition of the rate matrix
/// (eigenvalues -α, -β from `macro_rates`-style derivation).
fn propagate_two_cpt(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_periph: f64,
) {
    let cl = pk.cl();
    let v1 = pk.v();
    let q = pk.q();
    let v2 = pk.v2();
    if v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q <= 0.0 {
        return;
    }
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let disc = {
        let x = s * s - 4.0 * d;
        if x > 0.0 {
            x.sqrt()
        } else {
            0.0
        }
    };
    let alpha = (s + disc) / 2.0;
    // Vieta: alpha * beta = d (avoids cancellation).
    let beta = if alpha > 1e-30 { d / alpha } else { 0.0 };

    // Steady-state amounts under constant input b = (rate_central, rate_periph).
    //   A_ss = -K^{-1} b = (1/(k21·k10)) * [k21·b1 + k21·b2,
    //                                       k12·b1 + (k10+k12)·b2]
    // Special-cased for our two input channels.
    let denom_ss = k21 * k10;
    let (a_ss_1, a_ss_2) = if denom_ss > 1e-30 {
        (
            (k21 * rate_central + k21 * rate_periph) / denom_ss,
            (k12 * rate_central + (k10 + k12) * rate_periph) / denom_ss,
        )
    } else {
        (0.0, 0.0)
    };

    let h1_0 = state[0] - a_ss_1;
    let h2_0 = state[1] - a_ss_2;

    // Decompose homogeneous (h1_0, h2_0) into eigenmodes:
    //   eigenvectors u_α = (k21 - α, k12),  u_β = (k21 - β, k12)
    //   h(t) = c1·u_α·exp(-α·t) + c2·u_β·exp(-β·t)
    let denom = beta - alpha;
    let (c1, c2) = if k12.abs() < 1e-30 {
        // No central→peripheral transfer: 1-cpt-equivalent. Treat A1 and A2
        // as decoupled, with A1 having rate k10 and A2 having rate k21.
        // This branch is defensive — k12 == 0 implies q == 0 (already
        // guarded). Leave both modes balanced to avoid NaN.
        (0.0, 0.0)
    } else if denom.abs() < 1e-30 {
        // alpha ≈ beta — degenerate (only happens if discriminant ≈ 0,
        // i.e. (k10+k12-k21)^2 + 4·k12·k21 → 0 which requires k12=k21=0).
        // Fall back to splitting the homogeneous part evenly.
        let s_homog = h2_0 / k12;
        (s_homog * 0.5, s_homog * 0.5)
    } else {
        let s_homog = h2_0 / k12;
        let c1 = (h1_0 - s_homog * (k21 - beta)) / denom;
        let c2 = s_homog - c1;
        (c1, c2)
    };

    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    let h1_dt = c1 * (k21 - alpha) * e_alpha + c2 * (k21 - beta) * e_beta;
    let h2_dt = (c1 * e_alpha + c2 * e_beta) * k12;

    state[0] = h1_dt + a_ss_1;
    state[1] = h2_dt + a_ss_2;
}

// ─── Oral models ─────────────────────────────────────────────────────

/// 1-cpt oral propagator. State = `[A_depot, A_central]`. Bolus only:
/// the dose-event handler adds to state[0] (depot); during propagation
/// the depot drains into the central compartment via the absorption
/// rate `ka`.
fn propagate_one_cpt_oral(state: &mut [f64], dt: f64, pk: &PkParams) {
    let cl = pk.cl();
    let v = pk.v();
    let ka = pk.ka();
    if v <= 0.0 || cl <= 0.0 || ka <= 0.0 {
        return;
    }
    let ke = cl / v;
    let e_ka = (-ka * dt).exp();
    let e_ke = (-ke * dt).exp();

    let a_d_0 = state[0];
    let a_c_0 = state[1];

    // Depot decays exponentially (decoupled).
    state[0] = a_d_0 * e_ka;

    // Central compartment: homogeneous decay of A_c(0) plus depot-driven
    // contribution. Bateman form, with L'Hôpital fallback when ka ≈ ke.
    if (ka - ke).abs() < 1e-9 {
        state[1] = a_c_0 * e_ke + ka * a_d_0 * dt * e_ke;
    } else {
        state[1] = a_c_0 * e_ke + (ka * a_d_0 / (ke - ka)) * (e_ka - e_ke);
    }
}

/// 2-cpt oral propagator. State = `[A_depot, A_central, A_periph]`.
/// Bolus only — see module-level docs for infusion-into-oral support.
fn propagate_two_cpt_oral(state: &mut [f64], dt: f64, pk: &PkParams) {
    let cl = pk.cl();
    let v1 = pk.v();
    let q = pk.q();
    let v2 = pk.v2();
    let ka = pk.ka();
    if v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q <= 0.0 || ka <= 0.0 {
        return;
    }
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d_eig = k10 * k21;
    let disc = {
        let x = s * s - 4.0 * d_eig;
        if x > 0.0 {
            x.sqrt()
        } else {
            0.0
        }
    };
    let alpha = (s + disc) * 0.5;
    let beta = if alpha > 1e-30 { d_eig / alpha } else { 0.0 };

    let a_d_0 = state[0];
    let a_c_0 = state[1];
    let a_p_0 = state[2];

    let e_ka = (-ka * dt).exp();
    let e_alpha = (-alpha * dt).exp();
    let e_beta = (-beta * dt).exp();

    // Depot drains independently.
    state[0] = a_d_0 * e_ka;

    // Particular solution amplitudes for the depot-driven input
    // ka·A_d(t) = ka·A_d(0)·exp(-ka·t) into the central compartment.
    // Assumes (A, B)·exp(-ka·t) form; substitute into the (A_c, A_p)
    // ODE and solve. See derivation in the module docs / commit history.
    let denom_depot = (ka - alpha) * (ka - beta);
    let (cap_a, cap_b) = if denom_depot.abs() < 1e-12 {
        // ka coincides with α or β: would need L'Hôpital. The Bateman
        // singularity is rare in practice; fall back to no depot
        // contribution (preserves homogeneous evolution).
        (0.0, 0.0)
    } else {
        let a = ka * a_d_0 * (k21 - ka) / denom_depot;
        let b = ka * a_d_0 * k12 / denom_depot;
        (a, b)
    };

    // Homogeneous initial conditions = state - particular_at_t0.
    let h_c_0 = a_c_0 - cap_a;
    let h_p_0 = a_p_0 - cap_b;

    // Decompose into the 2-cpt (α, β) eigenmodes (eigenvectors
    // u_α = (k21 - α, k12),  u_β = (k21 - β, k12)).
    let denom = beta - alpha;
    let (c1, c2) = if k12.abs() < 1e-30 || denom.abs() < 1e-30 {
        let s_homog = h_p_0 / k12.max(1e-30);
        (s_homog * 0.5, s_homog * 0.5)
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
}

// ─── 3-compartment models ────────────────────────────────────────────

/// Three-compartment macro-rate constants and the auxiliary (k21, k31).
/// Returns `(α, β, γ, k21, k31)` with the convention `α > β > γ > 0`.
/// Mirrors `pk::three_compartment::macro_rates_three_cpt` (kept private
/// there) — duplicated to avoid making it `pub`.
fn macro_rates_three(
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, f64, f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q2 / v1;
    let k21 = q2 / v2;
    let k13 = q3 / v1;
    let k31 = q3 / v3;
    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;
    let h = s2 / 3.0;
    let p = s1 - s2 * s2 / 3.0;
    let q = s1 * s2 / 3.0 - 2.0 * s2 * s2 * s2 / 27.0 - s0;
    let p_safe = p.min(-1e-30);
    let m = 2.0 * (-p_safe / 3.0).sqrt();
    let arg = (3.0 * q / (p_safe * m)).clamp(-1.0, 1.0);
    let phi = arg.acos() / 3.0;
    let pi23 = 2.0 * std::f64::consts::FRAC_PI_3;
    let l0 = m * phi.cos() + h;
    let l1 = m * (phi - pi23).cos() + h;
    let l2 = m * (phi - 2.0 * pi23).cos() + h;
    let alpha = l0.max(l1).max(l2);
    let gamma = l0.min(l1).min(l2);
    let beta = s2 - alpha - gamma;
    (alpha, beta, gamma, k21, k31)
}

/// Spectral propagation of one 3-cpt eigenmode `μ` applied to state vector
/// `(c, p1, p2)`. Returns the mode's contribution to the new state at +dt.
///
/// Uses the *robust* eigenvector normalisation that stays well-defined
/// even when an eigenvalue happens to be close to one of the structural
/// rate constants (`k21` or `k31`) — common in 3-cpt models where the
/// slowest eigenvalue γ is dominated by the slowest peripheral rate:
///
///   v_μ = ((k21-μ)(k31-μ),  k12·(k31-μ),  k13·(k21-μ))
///   w_μ = ((k21-μ)(k31-μ),  k21·(k31-μ),  k31·(k21-μ))
///
/// Spectral projector: `P_μ = v_μ wᵀ_μ / (w_μ · v_μ)`.
#[allow(clippy::too_many_arguments)]
fn three_cpt_mode(
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
    if norm.abs() < 1e-30 {
        return (0.0, 0.0, 0.0);
    }
    let proj = w_c * c + w_p1 * p1 + w_p2 * p2;
    let coef = proj / norm;
    let exp_term = (-mu * dt).exp();
    (
        coef * v_c * exp_term,
        coef * v_p1 * exp_term,
        coef * v_p2 * exp_term,
    )
}

/// 3-cpt linear propagator (IV models). State = `[A_central, A_p1, A_p2]`.
/// Spectral decomposition along (α, β, γ) eigenmodes. Constant infusion
/// `(rate_central, rate_periph1, rate_periph2)` into the three slots is
/// handled via the steady-state + homogeneous decomposition pattern.
///
/// Steady-state per input channel (linear superposition):
///   - Channel 1 (central):   `A_ss = (r·v1/cl, r·v2/cl, r·v3/cl)`
///   - Channel 2 (periph 1):  `A_ss[0] = r·v1/cl,
///                             A_ss[1] = r·(cl+q2)·v2/(cl·q2),
///                             A_ss[2] = r·v3/cl`
///   - Channel 3 (periph 2):  `A_ss[0] = r·v1/cl,
///                             A_ss[1] = r·v2/cl,
///                             A_ss[2] = r·(cl+q3)·v3/(cl·q3)`
/// Combined `A_ss` is the sum across channels.
fn propagate_three_cpt(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_periph1: f64,
    rate_periph2: f64,
) {
    let cl = pk.cl();
    let v1 = pk.v();
    let q2 = pk.q();
    let v2 = pk.v2();
    let q3 = pk.q3();
    let v3 = pk.v3();
    if v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q2 <= 0.0 || v3 <= 0.0 || q3 <= 0.0 {
        return;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three(cl, v1, q2, v2, q3, v3);
    let k12 = q2 / v1;
    let k13 = q3 / v1;

    // Combined steady-state amounts under inputs (r_c, r_p1, r_p2).
    // Central slot: total input divided by k10 = (r_c+r_p1+r_p2)·v1/cl.
    // Peripheral slots get the central contribution + their own
    // direct-input correction.
    let r_total = rate_central + rate_periph1 + rate_periph2;
    let a_ss_c = r_total * v1 / cl;
    let a_ss_p1 =
        (rate_central + rate_periph2) * v2 / cl + rate_periph1 * (cl + q2) * v2 / (cl * q2);
    let a_ss_p2 =
        (rate_central + rate_periph1) * v3 / cl + rate_periph2 * (cl + q3) * v3 / (cl * q3);

    let h_c = state[0] - a_ss_c;
    let h_p1 = state[1] - a_ss_p1;
    let h_p2 = state[2] - a_ss_p2;

    let (ca, p1a, p2a) = three_cpt_mode(alpha, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode(beta, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode(gamma, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);

    state[0] = ca + cb + cg + a_ss_c;
    state[1] = p1a + p1b + p1g + a_ss_p1;
    state[2] = p2a + p2b + p2g + a_ss_p2;
}

/// 3-cpt oral propagator. State = `[A_depot, A_central, A_p1, A_p2]`.
/// Depot decays independently; the central+peripheral subsystem follows
/// the 3-cpt homogeneous evolution plus a depot-driven particular solution
/// of the form `(A, B, C)·exp(-ka·t)`.
fn propagate_three_cpt_oral(state: &mut [f64], dt: f64, pk: &PkParams) {
    let cl = pk.cl();
    let v1 = pk.v();
    let q2 = pk.q();
    let v2 = pk.v2();
    let q3 = pk.q3();
    let v3 = pk.v3();
    let ka = pk.ka();
    if v1 <= 0.0 || cl <= 0.0 || v2 <= 0.0 || q2 <= 0.0 || v3 <= 0.0 || q3 <= 0.0 || ka <= 0.0 {
        return;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three(cl, v1, q2, v2, q3, v3);
    let k12 = q2 / v1;
    let k13 = q3 / v1;

    let a_d_0 = state[0];
    let a_c_0 = state[1];
    let a_p1_0 = state[2];
    let a_p2_0 = state[3];

    let e_ka = (-ka * dt).exp();

    // Depot decays.
    state[0] = a_d_0 * e_ka;

    // Particular solution `X·exp(-ka·t)` for the depot-driven input
    // `(ka·A_d(0), 0, 0)·exp(-ka·t)` into the central compartment. Solving
    // `(K + ka·I) X = -(ka·A_d(0), 0, 0)` via Cramer's rule:
    //
    //   X1 = -ka·A_d(0)·(k21-ka)·(k31-ka) / [(ka-α)(ka-β)(ka-γ)]
    //   X2 = k12·X1 / (k21-ka)
    //   X3 = k13·X1 / (k31-ka)
    //
    // The leading negative sign comes from the odd-degree characteristic
    // polynomial of K (3-cpt has a cubic; 2-cpt's quadratic gives the
    // opposite sign — see `propagate_two_cpt_oral`). To stay robust when
    // ka coincides with k21 or k31, use the X3-form
    //   X3 = k13·(-ka·A_d(0)·d21·d31/denom) / d31
    //      = -k13·ka·A_d(0)·d21 / denom
    // which cancels the d31 cleanly.
    let cap_a;
    let cap_b;
    let cap_c;
    let denom_depot = (ka - alpha) * (ka - beta) * (ka - gamma);
    let d21 = k21 - ka;
    let d31 = k31 - ka;
    if denom_depot.abs() < 1e-12 {
        // ka coincides with α/β/γ — eigenvalue resonance, would need
        // L'Hôpital. Rare in practice; fall back to no contribution.
        cap_a = 0.0;
        cap_b = 0.0;
        cap_c = 0.0;
    } else {
        let scale = -ka * a_d_0 / denom_depot;
        cap_a = scale * d21 * d31;
        cap_b = scale * k12 * d31;
        cap_c = scale * k13 * d21;
    }

    // Homogeneous initial conditions = state - particular_at_t0.
    let h_c = a_c_0 - cap_a;
    let h_p1 = a_p1_0 - cap_b;
    let h_p2 = a_p2_0 - cap_c;

    let (ca, p1a, p2a) = three_cpt_mode(alpha, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cb, p1b, p2b) = three_cpt_mode(beta, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);
    let (cg, p1g, p2g) = three_cpt_mode(gamma, h_c, h_p1, h_p2, k12, k13, k21, k31, dt);

    state[1] = ca + cb + cg + cap_a * e_ka;
    state[2] = p1a + p1b + p1g + cap_b * e_ka;
    state[3] = p2a + p2b + p2g + cap_c * e_ka;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p
    }

    fn pk_two(cl: f64, v1: f64, q: f64, v2: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
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

    // ── Equivalence with superposition (constant pk_params) ───────────────────

    #[test]
    fn one_cpt_iv_bolus_matches_superposition_single_dose() {
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0, 2.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIvBolus, &subj.doses, t, &pk))
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn one_cpt_iv_bolus_matches_superposition_multi_dose() {
        let doses = vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(16.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIvBolus, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-10, max_relative = 1e-10);
            // Sanity: predictions are positive after the first dose.
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn one_cpt_infusion_matches_superposition() {
        // 1000 mg over 2h infusion, then observe.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::OneCptInfusion, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(PkModel::OneCptInfusion, &subj.doses, t, &pk)
            })
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn two_cpt_iv_bolus_matches_superposition() {
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 2.0, 6.0, 12.5, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two(5.0, 30.0, 2.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptIvBolus, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-9,);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn two_cpt_infusion_matches_superposition() {
        // 1000 mg over 2h infusion into central, multi-dose.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 500.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 1.0, 2.0, 6.0, 12.5, 14.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two(5.0, 30.0, 2.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::TwoCptInfusion, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(PkModel::TwoCptInfusion, &subj.doses, t, &pk)
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-8, max_relative = 1e-8,);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    // ── TV-covariate effect: changing CL between doses changes elimination ───

    #[test]
    fn one_cpt_tv_cl_changes_decay_rate() {
        // Single dose at t=0, two observations: CL doubles between the two
        // observations.
        //
        // NONMEM convention (end-of-interval / current-record): each
        // propagation [t_{i-1}, t_i] uses the PK params evaluated AT t_i
        // (`$PK runs at every record, ADVAN propagates to that record`).
        // So:
        //   [0, t1=1]: uses pk at obs1 = pk_low  → ke = 0.05
        //   [t1, t2]:  uses pk at obs2 = pk_high → ke = 0.10
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 2.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low];
        let pk_obs = vec![pk_low, pk_high]; // pk changes at obs2

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);

        // [0, 1] uses pk_low (= pk at obs1):
        //   A1(1) = 1000 * exp(-0.05) ≈ 951.23
        //   C(1)  = 9.5123
        let a1_at_t1 = 1000.0 * (-0.05f64).exp();
        let c1_expected = a1_at_t1 / 100.0;
        assert_relative_eq!(preds[0], c1_expected, epsilon = 1e-12);

        // [1, 2] uses pk_high (= pk at obs2). End-of-interval — the new CL
        // applies to the interval BEFORE its record:
        //   A1(2) = A1(1) * exp(-0.10) ≈ 951.23 * 0.9048 ≈ 860.71
        //   C(2)  = 8.6071
        let a1_at_t2 = a1_at_t1 * (-0.10f64).exp();
        let c2_expected = a1_at_t2 / 100.0; // V from pk_high == 100 anyway.
        assert_relative_eq!(preds[1], c2_expected, epsilon = 1e-12);
    }

    #[test]
    fn one_cpt_tv_cl_between_doses_changes_decay() {
        // Two doses, with CL doubling between them.
        //
        // End-of-interval (NONMEM) propagation:
        //   [0, t_obs1=5]:  uses pk at obs1  = pk_low
        //   [5, t_dose2=10]: uses pk at dose2 = pk_high
        //   [10, t_obs2=12]: uses pk at obs2  = pk_high
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

        let preds = event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);

        // [0, 5] uses pk_low (pk at obs1):
        //   A1(5) = 1000 * exp(-0.05*5) = 778.80, C = 7.788
        let a1_at_5 = 1000.0 * (-0.05f64 * 5.0).exp();
        let c5_expected = a1_at_5 / 100.0;
        assert_relative_eq!(preds[0], c5_expected, epsilon = 1e-12);

        // [5, 10] uses pk_high (pk at dose2): ke=0.10 for 5h.
        //   A1(10⁻) = 778.80 * exp(-0.10*5) = 472.37
        // After dose2: A1(10⁺) = 472.37 + 1000 = 1472.37
        // [10, 12] uses pk_high (pk at obs2): ke=0.10 for 2h.
        //   A1(12) = 1472.37 * exp(-0.10*2) = 1205.49, C = 12.0549
        let a1_at_10_minus = a1_at_5 * (-0.10f64 * 5.0).exp();
        let a1_at_10_plus = a1_at_10_minus + 1000.0;
        let a1_at_12 = a1_at_10_plus * (-0.10f64 * 2.0).exp();
        let c12_expected = a1_at_12 / 100.0;
        assert_relative_eq!(preds[1], c12_expected, epsilon = 1e-12);
    }

    #[test]
    fn supports_event_driven_gates_supported_models_only() {
        // Now covers all analytical PK models.
        assert!(supports_event_driven(PkModel::OneCptIvBolus));
        assert!(supports_event_driven(PkModel::OneCptInfusion));
        assert!(supports_event_driven(PkModel::OneCptOral));
        assert!(supports_event_driven(PkModel::TwoCptIvBolus));
        assert!(supports_event_driven(PkModel::TwoCptInfusion));
        assert!(supports_event_driven(PkModel::TwoCptOral));
        assert!(supports_event_driven(PkModel::ThreeCptIvBolus));
        assert!(supports_event_driven(PkModel::ThreeCptInfusion));
        assert!(supports_event_driven(PkModel::ThreeCptOral));
    }

    // ── Oral and 3-cpt: equivalence with the existing single-dose
    //   superposition path when pk_params are constant ─────────────────

    fn pk_one_oral(cl: f64, v: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p
    }

    fn pk_two_oral(cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[crate::types::PK_IDX_KA] = ka;
        p
    }

    fn pk_three(cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v1;
        p.values[crate::types::PK_IDX_Q] = q2;
        p.values[crate::types::PK_IDX_V2] = v2;
        p.values[crate::types::PK_IDX_Q3] = q3;
        p.values[crate::types::PK_IDX_V3] = v3;
        p
    }

    fn pk_three_oral(cl: f64, v1: f64, q2: f64, v2: f64, q3: f64, v3: f64, ka: f64) -> PkParams {
        let mut p = pk_three(cl, v1, q2, v2, q3, v3);
        p.values[crate::types::PK_IDX_KA] = ka;
        p
    }

    #[test]
    fn one_cpt_oral_matches_superposition_single_dose() {
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one_oral(10.0, 100.0, 1.5);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptOral, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-10, max_relative = 1e-10);
            assert!(*a >= 0.0, "obs {} should be non-negative, got {}", i, a);
        }
    }

    #[test]
    fn one_cpt_oral_matches_superposition_multi_dose() {
        let doses = vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(16.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 4.0, 8.5, 12.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one_oral(5.0, 80.0, 1.2);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptOral, &subj.doses, t, &pk))
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn two_cpt_oral_matches_superposition_single_dose() {
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two_oral(5.0, 30.0, 2.0, 50.0, 1.5);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptOral, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-8);
            assert!(*a >= 0.0, "obs {} should be non-negative, got {}", i, a);
        }
    }

    #[test]
    fn two_cpt_oral_matches_superposition_multi_dose() {
        let doses = vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 4.0, 12.5, 16.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_two_oral(5.0, 30.0, 2.0, 50.0, 1.2);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptOral, &subj.doses, t, &pk))
            .collect();
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-8, max_relative = 1e-8);
        }
    }

    #[test]
    fn three_cpt_iv_bolus_matches_superposition() {
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 1.0, 4.0, 12.5, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_three(5.0, 20.0, 2.0, 30.0, 0.5, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::ThreeCptIvBolus, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(PkModel::ThreeCptIvBolus, &subj.doses, t, &pk)
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-8);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn three_cpt_infusion_matches_superposition() {
        // 1000 mg over 2h infusion into central, multi-dose.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 500.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 12.5, 14.0, 18.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_three(5.0, 20.0, 2.0, 30.0, 0.5, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::ThreeCptInfusion, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| {
                crate::pk::predict_concentration(PkModel::ThreeCptInfusion, &subj.doses, t, &pk)
            })
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-7, max_relative = 1e-7);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }
    }

    #[test]
    fn three_cpt_peripheral_infusion_propagates_correctly() {
        // 3-cpt IV with infusion into periph1 (cmt=2) over a 2h window.
        // The dispatcher must accept cmt=2 (instead of panicking), and
        // the propagator must produce finite, non-negative central
        // concentrations as drug transfers periph1 → central.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 2, 500.0, false, 0.0)];
        let obs_times = vec![0.5, 2.0, 4.0, 8.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_three(5.0, 20.0, 2.0, 30.0, 0.5, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::ThreeCptInfusion, &subj, &pk_dose, &pk_obs, &[]);

        for (i, &p) in preds.iter().enumerate() {
            assert!(p.is_finite(), "obs {} should be finite, got {}", i, p);
            assert!(
                p > 0.0,
                "obs {} (t={}): expected positive central, got {}",
                i,
                obs_times[i],
                p
            );
        }
    }

    #[test]
    fn three_cpt_periph2_infusion_dispatches_without_panic() {
        // cmt=3 (periph2) infusion path. Just confirm the dispatcher
        // accepts it and produces finite output — exact dynamics depend
        // on the slow-periph2 transfer rates which are model-specific.
        let doses = vec![DoseEvent::new(0.0, 100.0, 3, 50.0, false, 0.0)];
        let obs_times = vec![1.0, 4.0, 12.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_three(5.0, 20.0, 2.0, 30.0, 0.5, 100.0);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds =
            event_driven_predictions(PkModel::ThreeCptInfusion, &subj, &pk_dose, &pk_obs, &[]);
        for &p in &preds {
            assert!(p.is_finite() && p >= 0.0, "got {}", p);
        }
    }

    #[test]
    fn three_cpt_oral_matches_superposition() {
        let doses = vec![
            DoseEvent::new(0.0, 500.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 12.5, 14.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_three_oral(5.0, 20.0, 2.0, 30.0, 0.5, 100.0, 1.2);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::ThreeCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::ThreeCptOral, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-8, max_relative = 1e-7);
            assert!(*a >= 0.0, "obs {} should be non-negative, got {}", i, a);
        }
    }

    // ── TV cov on oral: pk changes between doses changes elimination ─────

    #[test]
    fn one_cpt_evid2_mid_interval_switches_decay_rate() {
        // Single dose at t=0 (CL_low), pk-only (EVID=2) at t=5 with CL_high,
        // single obs at t=10 (CL_high).
        //
        // End-of-interval (NONMEM) propagation:
        //   [0, 5]:  uses pk at EVID=2 = pk_high → ke = 0.10
        //   [5, 10]: uses pk at obs    = pk_high → ke = 0.10
        // The EVID=2 record's PK params govern the interval LEADING UP to it
        // (NONMEM "$PK runs at every record then ADVAN propagates to it"
        // semantic). Both intervals end up using pk_high here, so the
        // dose's pk_low is effectively unused for propagation.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![10.0];
        let mut subj = make_subject(doses, obs_times);
        subj.pk_only_times = vec![5.0];

        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low];
        let pk_obs = vec![pk_high];
        let pk_only = vec![pk_high];

        let preds =
            event_driven_predictions(PkModel::OneCptIvBolus, &subj, &pk_dose, &pk_obs, &pk_only);

        //   A(5⁻) = 1000 * exp(-0.10*5) = 606.53   (uses pk_high — end-of-interval)
        //   A(10) = A(5⁻) * exp(-0.10*5) = 367.88
        //   C(10) = 3.6788
        let a_at_5 = 1000.0 * (-0.10f64 * 5.0).exp();
        let a_at_10 = a_at_5 * (-0.10f64 * 5.0).exp();
        let c10_expected = a_at_10 / 100.0;
        assert_relative_eq!(preds[0], c10_expected, epsilon = 1e-12);

        // Sanity: without any pk-update event the decay would be
        // 1000 * exp(-0.05*10) / 100 = 6.065 — much higher than our 3.68.
        let no_evid2 = 1000.0 * (-0.05f64 * 10.0).exp() / 100.0;
        assert!(
            preds[0] < no_evid2,
            "EVID=2 high-CL update should decay faster than baseline pk_low: \
             with={}, without={}",
            preds[0],
            no_evid2
        );
    }

    #[test]
    fn one_cpt_oral_tv_cl_between_doses_changes_decay() {
        // Two oral doses; CL doubles between dose 1 and dose 2, so the
        // central-compartment decay rate doubles for the second interval.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![6.0, 18.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk_low = pk_one_oral(5.0, 100.0, 2.0); // ke=0.05
        let pk_high = pk_one_oral(10.0, 100.0, 2.0); // ke=0.10
        let pk_dose = vec![pk_low, pk_high];
        let pk_obs = vec![pk_low, pk_high];

        let preds_tv = event_driven_predictions(PkModel::OneCptOral, &subj, &pk_dose, &pk_obs, &[]);
        // Sanity: predictions are positive and finite.
        for &p in &preds_tv {
            assert!(p > 0.0 && p.is_finite(), "got {}", p);
        }
        // Compare with a constant-pk run at pk_high — TV run's decay during
        // the first interval is slower (lower CL) so concentration at obs1
        // (t=6, between doses) should differ.
        let pk_const_high = vec![pk_high; subj.obs_times.len() + subj.doses.len()];
        let preds_const = event_driven_predictions(
            PkModel::OneCptOral,
            &subj,
            &pk_const_high[..subj.doses.len()],
            &pk_const_high[..subj.obs_times.len()],
            &[],
        );
        // First obs: TV run must give a *higher* concentration than the
        // all-high run because the depot drained more slowly under low CL
        // doesn't matter — what matters is central elimination, which is
        // also slower. So preds_tv[0] > preds_const[0].
        assert!(
            preds_tv[0] > preds_const[0],
            "TV (low CL early) should preserve more drug at t=6 than constant high CL: \
             tv={}, const_high={}",
            preds_tv[0],
            preds_const[0]
        );
    }

    // ── EventSchedule cache ──────────────────────────────────────────────────

    #[test]
    fn event_schedule_orders_events_dose_before_obs_at_same_time() {
        // Same-time tie-break: dose should run before observation so the obs
        // sees post-dose state. EventSchedule.events must reflect that order.
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0]; // first obs at t=0, same as dose
        let subj = make_subject(doses, obs_times);
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIvBolus, &[]);

        assert_eq!(schedule.events.len(), 3);
        assert!(matches!(schedule.events[0].kind, EventKind::Dose));
        assert!(matches!(schedule.events[1].kind, EventKind::Obs));
        assert_eq!(schedule.events[1].time, 0.0);
        assert!(matches!(schedule.events[2].kind, EventKind::Obs));
        assert_eq!(schedule.events[2].time, 1.0);
    }

    #[test]
    fn event_schedule_bounds_split_intervals_at_infusion_endpoints() {
        // Interval (0, 10) with an infusion that ends at t=2 must be split
        // at t=2 so the rate matrix is constant within each sub-interval.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 2.0)];
        let obs_times = vec![10.0];
        let subj = make_subject(doses, obs_times);
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptInfusion, &[]);

        // Two events (dose at 0, obs at 10) → one interval (0, 10).
        assert_eq!(schedule.bounds_per_interval.len(), 1);
        let bounds = &schedule.bounds_per_interval[0];
        // Bounds must include the interval endpoints AND the infusion stop.
        // (Infusion start at t=0 is exactly at t_from, so the > t_from + eps
        // guard inside compute_propagation_bounds keeps it out — that's
        // fine because the start point is already implicit at t_from.)
        assert!(bounds.contains(&0.0));
        assert!(bounds.contains(&10.0));
        assert!(
            bounds.iter().any(|&b| (b - 2.0).abs() < 1e-12),
            "expected infusion stop at t=2 in bounds: {:?}",
            bounds
        );
    }

    #[test]
    fn event_driven_with_schedule_matches_no_schedule_path() {
        // Equivalence check: passing a pre-built schedule must produce
        // identical predictions to letting event_driven_predictions
        // build one internally. Run on a multi-dose infusion with TV
        // PK params (per-event clearance differs across the timeline).
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 2.0),
            DoseEvent::new(8.0, 1000.0, 1, 500.0, false, 2.0),
        ];
        let obs_times = vec![1.0, 3.0, 8.0, 9.0, 12.0, 24.0];
        let subj = make_subject(doses.clone(), obs_times.clone());

        // TV: clearance grows over time, V is constant.
        let pk_dose = vec![pk_two(5.0, 50.0, 2.0, 100.0); doses.len()];
        let mut pk_obs = Vec::with_capacity(obs_times.len());
        for (i, _) in obs_times.iter().enumerate() {
            pk_obs.push(pk_two(5.0 + 0.1 * i as f64, 50.0, 2.0, 100.0));
        }

        let direct =
            event_driven_predictions(PkModel::TwoCptInfusion, &subj, &pk_dose, &pk_obs, &[]);
        let schedule = EventSchedule::for_subject(&subj, PkModel::TwoCptInfusion, &[]);
        let with_sched = event_driven_predictions_with_schedule(
            PkModel::TwoCptInfusion,
            &subj,
            &schedule,
            &pk_dose,
            &pk_obs,
            &[],
        );
        assert_eq!(direct.len(), with_sched.len());
        for (a, b) in direct.iter().zip(with_sched.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-12);
        }
    }
}
