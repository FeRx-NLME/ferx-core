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
//! Infusion support (`rate > 0`):
//!   - IV models: into the central compartment (cmt 1), plus the
//!     peripheral compartment(s) for 2-/3-cpt IV.
//!   - Oral models: into the **central** compartment (cmt 2, a
//!     depot-bypassing infusion) AND into the **depot** (cmt 1, #400) —
//!     a zero-order release into the depot, then first-order `ka`
//!     absorption into central.
//! Infusion into an oral peripheral compartment still panics — a rare
//! clinical setup tracked as a follow-up.

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
    /// EVID=3 / EVID=4 system reset — zeros every compartment amount.
    /// For EVID=4 a `Dose` event is scheduled at the same time; the
    /// `Reset < Dose` tie-break (see [`kind_order`]) makes the reset run
    /// first so the dose lands in a freshly emptied system.
    Reset,
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub time: f64,
    pub kind: EventKind,
    /// Index into `subject.doses`, `subject.obs_times`,
    /// `subject.pk_only_times`, or `subject.reset_times` depending on `kind`.
    /// For `Reset` events the index is unused (a reset carries no per-event
    /// data — it just zeros the state).
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
    /// Merged + sorted event list. Tie-break order is
    /// `Reset < Dose < PkOnly < Obs` so a system reset zeros the state
    /// before a same-time dose lands (EVID=4), and covariate-change markers
    /// run after a dose at the same time but before an observation (matches
    /// NONMEM `$PK` semantics).
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
        for (r, &t) in subject.reset_times.iter().enumerate() {
            events.push(Event {
                time: t,
                kind: EventKind::Reset,
                orig_idx: r,
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
        // Reset sorts first so an EVID=4 (reset + dose) zeros the state
        // before its own dose lands at the same time.
        EventKind::Reset => 0,
        EventKind::Dose => 1,
        EventKind::PkOnly => 2,
        EventKind::Obs => 3,
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
        PkModel::OneCptIv
            | PkModel::OneCptOral
            | PkModel::TwoCptIv
            | PkModel::TwoCptOral
            | PkModel::ThreeCptIv
            | PkModel::ThreeCptOral
    )
}

/// State-vector dimension and central-compartment slot index for a given
/// pk_model. Central is where the observation read-out reads from.
fn state_layout(pk_model: PkModel) -> (usize, usize) {
    match pk_model {
        PkModel::OneCptIv => (1, 0),
        PkModel::OneCptOral => (2, 1), // [depot, central]
        PkModel::TwoCptIv => (2, 0),
        PkModel::TwoCptOral => (3, 1), // [depot, central, periph]
        PkModel::ThreeCptIv => (3, 0),
        PkModel::ThreeCptOral => (4, 1), // [depot, central, periph1, periph2]
    }
}

/// Number of cycles to expand for SS equilibration in the event-driven
/// analytical path. Matches the ODE-side default in `ode/predictions.rs`
/// for consistency. 50 cycles puts `exp(-50·k·II)` well below 1e-9 of
/// the SS amount for any realistic PK.
const EVENT_DRIVEN_SS_EQUILIBRATION_CYCLES: usize = 50;

/// Pre-equilibrate the event-driven analytical state to its SS value for
/// an SS=1 dose. Same scheme as `ode/predictions.rs::equilibrate_ss_state`:
/// reset state, then loop N cycles of (apply dose; propagate one II).
/// The state after the loop is the "just-before-next-pulse" SS state;
/// the caller applies the SS dose's own pulse through the normal flow.
///
/// `dose.ii > 0` and `dose.duration <= dose.ii` are required (callers
/// already guard the first; overlapping infusions are rejected).
fn equilibrate_ss_state_event_driven(
    pk_model: PkModel,
    pk: &PkParams,
    dose: &DoseEvent,
) -> Vec<f64> {
    let (n_states, _) = state_layout(pk_model);
    let mut state = vec![0.0_f64; n_states];

    if dose.ii <= 0.0 || dose.cmt == 0 {
        return state;
    }
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= n_states {
        return state;
    }
    let is_inf = dose.rate > 0.0 && dose.duration > 0.0 && dose.duration.is_finite();
    if is_inf && dose.duration > dose.ii {
        // Overlapping infusions: no closed-form pulse expansion. Caller's
        // api.rs warning fires for this case.
        return state;
    }

    // Synthetic single-dose at t=0 inside each cycle; propagate over
    // [0, T_inf, II] for infusions (so the bounds split at the infusion
    // end) or [0, II] for boluses.
    let synthetic_dose = if is_inf {
        vec![DoseEvent::new(
            0.0, dose.amt, dose.cmt, dose.rate, false, 0.0,
        )]
    } else {
        Vec::new()
    };
    let synthetic_lagtimes: Vec<f64> = if is_inf { vec![0.0] } else { Vec::new() };
    let bounds: Vec<f64> = if is_inf {
        vec![0.0, dose.duration, dose.ii]
    } else {
        vec![0.0, dose.ii]
    };

    for _ in 0..EVENT_DRIVEN_SS_EQUILIBRATION_CYCLES {
        if !is_inf {
            // Bolus pulse: instantaneous amount jump (with F).
            state[cmt_idx] += pk.bioavailable_amount(dose.amt);
        }
        propagate_with_bounds(
            &mut state,
            &bounds,
            pk,
            pk_model,
            &synthetic_dose,
            &synthetic_lagtimes,
            f64::NEG_INFINITY,
        );
    }

    state
}

/// Event-driven steady-state state at `phase` ∈ [0, II) within the dosing
/// cycle, forward from the pulse at phase 0. [`equilibrate_ss_state_event_driven`]
/// returns the pre-pulse trough (phase 0⁻ ≡ II); this advances from that
/// trough through the dose pulse and `phase` units of the cycle.
///
/// Used to recover the *previous interval's* steady-state tail for an SS
/// dose with a lagtime: observations between the dose record time and the
/// lagged arrival sit at phase `II − lagtime` … `II` (issue #15). For SS
/// infusions this assumes `phase ≥ dose.duration` (`lagtime ≤ II − T_inf`),
/// the realistic regime; overlapping infusions are rejected upstream.
fn ss_state_at_phase_event_driven(
    pk_model: PkModel,
    pk: &PkParams,
    dose: &DoseEvent,
    phase: f64,
) -> Vec<f64> {
    let mut state = equilibrate_ss_state_event_driven(pk_model, pk, dose);
    if phase <= 0.0 {
        return state;
    }
    let (n_states, _) = state_layout(pk_model);
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= n_states {
        return state;
    }

    let is_inf = dose.rate > 0.0 && dose.duration > 0.0 && dose.duration.is_finite();
    if is_inf {
        let t_inf = dose.duration;
        let synthetic_dose = vec![DoseEvent::new(
            0.0, dose.amt, dose.cmt, dose.rate, false, 0.0,
        )];
        let synthetic_lagtimes = vec![0.0];
        let bounds: Vec<f64> = if phase > t_inf {
            vec![0.0, t_inf, phase]
        } else {
            vec![0.0, phase]
        };
        propagate_with_bounds(
            &mut state,
            &bounds,
            pk,
            pk_model,
            &synthetic_dose,
            &synthetic_lagtimes,
            f64::NEG_INFINITY,
        );
    } else {
        state[cmt_idx] += pk.bioavailable_amount(dose.amt);
        propagate_with_bounds(
            &mut state,
            &[0.0, phase],
            pk,
            pk_model,
            &[],
            &[],
            f64::NEG_INFINITY,
        );
    }
    state
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
    // Defensive guard (#324/#394): modeled-RATE doses (e.g. RATE=-2 -> D{cmt})
    // must be resolved to concrete `rate`/`duration` before this event-driven
    // walker — the analytical dispatcher resolves them via the model's
    // `dose_attr_map`, and the public entrypoints reject an unbacked modeled dose
    // first (`fit()` / `ferx check` via `check_model_data`, `predict()` /
    // `simulate()` via `assert_modeled_doses_supported`). Reaching here unresolved
    // means a path forgot to resolve (e.g. a direct caller of this `pub` fn). A
    // real `assert!` (not `debug_assert!`) so release builds fail loudly too
    // instead of silently mis-handling a 0-rate "infusion"; it is O(doses) and
    // dwarfed by the per-interval event-driven evaluation.
    assert!(
        subject.all_doses_fixed(),
        "modeled-RATE dose reached the analytical predictor unresolved \
         (resolve via dose_attr_map, or validate with check_model_data, before predicting)"
    );
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
    // Most-recent system-reset time. Infusions whose window started before
    // this are no longer active (a reset turns off ongoing infusions, the
    // same way it zeros the compartments). `NEG_INFINITY` until the first
    // reset means every infusion is eligible.
    let mut reset_floor = f64::NEG_INFINITY;

    for (i, ev) in schedule.events.iter().enumerate() {
        // EVID=3 / EVID=4 reset: zero every compartment. Any drug carried
        // by the interval ending here would just be discarded, so skip the
        // propagation entirely and only advance the clock. A reset carries
        // no PK params, so this also avoids the `pk_for` lookup below.
        if ev.kind == EventKind::Reset {
            state.iter_mut().for_each(|s| *s = 0.0);
            cur_t = ev.time;
            reset_floor = ev.time;
            continue;
        }

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
                reset_floor,
            );
            cur_t = ev.time;
        }

        match ev.kind {
            EventKind::Dose => {
                let d = &subject.doses[ev.orig_idx];
                // Steady-state (SS=1): reset state and load with the SS
                // amount from the infinite-past pulse train before the SS
                // dose's own pulse is applied through the normal flow.
                // See `equilibrate_ss_state_event_driven` for the per-cycle
                // scheme. Mirrors `ode/predictions.rs::ode_predictions_*`.
                if d.ss && d.ii > 0.0 {
                    state = equilibrate_ss_state_event_driven(pk_model, &pk_now, d);
                }
                if d.rate <= 0.0 {
                    // Bolus: instantaneous amount jump in dose's compartment.
                    // Apply bioavailability F1 — mirrors the analytical
                    // *_oral_f path (`d = f_bio * dose.amt * ka / v1`). Without
                    // this, models that compute F as a function of DOSE (e.g.
                    // dose-dependent F = (DOSE/100)^γ) would silently drop the
                    // F multiplier whenever the subject had any time-varying
                    // covariate, since `compute_predictions_with_tv_into_*`
                    // routes to the event-driven path on `has_tv`. SCEN3 in the
                    // astra-testdata-simulator benchmark hits this: DAY/STIME
                    // make every subject look "TV", and predictions for
                    // high-dose subjects came out F× too small.
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n_states {
                        state[cmt_idx] += pk_now.bioavailable_amount(d.amt);
                    } else {
                        panic!(
                            "event-driven PK: dose into compartment {} but model has \
                             {} states (cmt is 1-based)",
                            d.cmt, n_states
                        );
                    }
                }
                // Infusion: handled inside `propagate` via the active-input
                // lookup — F is applied to the rate there (see
                // propagate_with_bounds), preserving the user-specified
                // duration. This matches NONMEM's `rate = AMT/DUR` convention
                // when both are supplied, and keeps F-modulated infusions
                // bit-for-bit equal to the bolus path on the limit dur→0.
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
            EventKind::Reset => unreachable!("Reset handled before pk_for above"),
        }
    }

    // SS + lagtime: previous-interval steady-state tail (issue #15). The
    // walk above seeds the SS state only at the lagged dose event, so
    // observations between the dose record time and the lagged arrival are
    // left at the empty initial state (≈0). At steady state they carry the
    // tail of the prior pulse; recompute them from the SS phase. Matches the
    // analytical (`predict_concentration`) and ODE (`ss_state_at_phase`)
    // paths, verified against NONMEM ALAG1 + SS=1.
    for (k, d) in subject.doses.iter().enumerate() {
        let lag = schedule.dose_lagtimes.get(k).copied().unwrap_or(0.0);
        if !(d.ss && d.ii > 0.0 && lag > 0.0) {
            continue;
        }
        let t_eff = d.time + lag;
        // Reconstruct the steady-state amount with the *dose-record* PK
        // snapshot and read out concentration with the *observation* V —
        // the same split the main walk uses (it equilibrates the SS dose
        // with `pk_now = pk_at_dose[k]` and divides by `pk_at_obs[j].v()`
        // at the obs event). Keeping the dose-time snapshot here is what
        // makes the pre-arrival branch continuous with the main walk at
        // t_eff: the steady-state profile is defined by the SS-record
        // params, and a full-interval propagation from that equilibrium
        // returns exactly to the trough. Equilibrating with obs-time params
        // instead would break that continuity under time-varying covariates.
        // (For pre-lag samples — within `lag` of the record — the two
        // snapshots are effectively equal anyway.)
        let pk_dose = pk_at_dose[k];
        for (j, &t_obs) in subject.obs_times.iter().enumerate() {
            if t_obs >= d.time - 1e-12 && t_obs < t_eff - 1e-12 {
                // Phase of the previous pulse (at t_eff − II) at t_obs.
                let phase = t_obs - t_eff + d.ii;
                let st = ss_state_at_phase_event_driven(pk_model, &pk_dose, d, phase);
                let v = pk_at_obs[j].v();
                preds[j] = if v > 0.0 {
                    (st[central_slot] / v).max(0.0)
                } else {
                    0.0
                };
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
        // Resets are handled by the caller before `pk_for`; they carry no
        // per-event PK snapshot.
        EventKind::Reset => unreachable!("Reset carries no PK params"),
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
///
/// `reset_floor` is the time of the most recent system reset (EVID=3/4), or
/// `f64::NEG_INFINITY` when none has happened. Infusions whose (lagged) start
/// is strictly before `reset_floor` are treated as turned off — a reset stops
/// ongoing infusions, just as it zeros the compartments.
fn propagate_with_bounds(
    state: &mut [f64],
    bounds: &[f64],
    pk: &PkParams,
    pk_model: PkModel,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    reset_floor: f64,
) {
    for w in bounds.windows(2) {
        let s0 = w[0];
        let s1 = w[1];
        let dt = s1 - s0;
        if dt <= 0.0 {
            continue;
        }
        let mid = 0.5 * (s0 + s1);
        let mut rate_central = 0.0;
        let mut rate_periph1 = 0.0;
        let mut rate_periph2 = 0.0;
        // Zero-order input into the oral **depot** (cmt 1, #400) — a zero-order
        // release into the depot followed by first-order `ka` absorption into
        // central. Distinct channel from `rate_central` (cmt 2, depot bypass).
        let mut rate_depot = 0.0;
        // F multiplies infusion rate so a dur→0 infusion limits to a bolus
        // of amount F·AMT — same convention as the EventKind::Dose arm above.
        for (k, d) in doses.iter().enumerate() {
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let t_start = d.time + lag;
            let t_end = t_start + d.duration;
            // Infusions that started before the last reset are turned off.
            if t_start < reset_floor {
                continue;
            }
            if d.rate > 0.0 && d.duration > 0.0 && t_start <= mid && t_end >= mid {
                let r = pk.bioavailable_rate(d.rate);
                match (pk_model, d.cmt) {
                    (PkModel::OneCptIv, 1) => rate_central += r,
                    (PkModel::OneCptOral, 1) => rate_depot += r,
                    (PkModel::OneCptOral, 2) => rate_central += r,
                    (PkModel::TwoCptIv, 1) => rate_central += r,
                    (PkModel::TwoCptIv, 2) => rate_periph1 += r,
                    (PkModel::TwoCptOral, 1) => rate_depot += r,
                    (PkModel::TwoCptOral, 2) => rate_central += r,
                    (PkModel::ThreeCptIv, 1) => rate_central += r,
                    (PkModel::ThreeCptIv, 2) => rate_periph1 += r,
                    (PkModel::ThreeCptIv, 3) => rate_periph2 += r,
                    (PkModel::ThreeCptOral, 1) => rate_depot += r,
                    (PkModel::ThreeCptOral, 2) => rate_central += r,
                    _ => panic!(
                        "event-driven PK: infusion into compartment {} not supported \
                         for model {:?}. Supported: central for all models; depot (cmt 1) \
                         for oral models; periph1/2 for 2- and 3-cpt IV models. Oral \
                         peripheral infusion is a tracked follow-up.",
                        d.cmt, pk_model
                    ),
                }
            }
        }

        match pk_model {
            PkModel::OneCptIv => {
                propagate_one_cpt(state, dt, pk, rate_central);
            }
            PkModel::OneCptOral => {
                propagate_one_cpt_oral(state, dt, pk, rate_central, rate_depot);
            }
            PkModel::TwoCptIv => {
                propagate_two_cpt(state, dt, pk, rate_central, rate_periph1);
            }
            PkModel::TwoCptOral => {
                propagate_two_cpt_oral(state, dt, pk, rate_central, rate_depot);
            }
            PkModel::ThreeCptIv => {
                propagate_three_cpt(state, dt, pk, rate_central, rate_periph1, rate_periph2);
            }
            PkModel::ThreeCptOral => {
                propagate_three_cpt_oral(state, dt, pk, rate_central, rate_depot);
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

/// 1-cpt oral propagator. State = `[A_depot, A_central]`. The dose-event
/// handler adds a bolus to state[0] (depot); during propagation the depot
/// drains into the central compartment via the absorption rate `ka`.
/// `rate_central` is a constant zero-order input into the central compartment
/// over this sub-interval — a depot-bypassing infusion (RATE>0 into cmt 2) —
/// added by linear superposition. `rate_depot` is a constant zero-order input
/// into the **depot** (RATE>0 into cmt 1, #400): zero-order release into the
/// depot, then first-order `ka` absorption into central.
fn propagate_one_cpt_oral(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_depot: f64,
) {
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

    // Constant infusion into central (depot bypass). By linearity its forced
    // response is the IV propagator's input term from a zero initial state, so
    // reuse `propagate_one_cpt` to avoid duplicating the closed form (#327
    // follow-up: the event-driven path previously dropped this input on oral
    // models).
    if rate_central > 0.0 {
        let mut inflow = [0.0_f64];
        propagate_one_cpt(&mut inflow, dt, pk, rate_central);
        state[1] += inflow[0];
    }

    // Constant zero-order input into the depot (#400). Forced response from a
    // zero initial state, added by linearity:
    //   depot:   A_d_forced(dt) = (R/ka)·(1 − e_ka)
    //   central: drive ka·A_d_forced = R·(1 − e^{−ka t}) convolved with the
    //            central impulse response e^{−ke t} gives
    //            R/ke·(1 − e_ke) − R·(e_ka − e_ke)/(ke − ka),
    //            with the L'Hôpital limit R·dt·e_ke as ka → ke.
    if rate_depot > 0.0 {
        state[0] += (rate_depot / ka) * (1.0 - e_ka);
        let central_forced = if (ka - ke).abs() < 1e-9 {
            rate_depot / ke * (1.0 - e_ke) - rate_depot * dt * e_ke
        } else {
            rate_depot / ke * (1.0 - e_ke) - rate_depot * (e_ka - e_ke) / (ke - ka)
        };
        state[1] += central_forced;
    }
}

/// 2-cpt oral propagator. State = `[A_depot, A_central, A_periph]`.
/// `rate_central` is a constant zero-order input into the central compartment
/// (depot-bypassing infusion, RATE>0 into cmt 2), added by linear superposition.
/// `rate_depot` is a constant zero-order input into the **depot** (RATE>0 into
/// cmt 1, #400): zero-order release into the depot, then first-order `ka`
/// absorption into central.
fn propagate_two_cpt_oral(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_depot: f64,
) {
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

    // Constant infusion into central (depot bypass) — see propagate_one_cpt_oral.
    // Reuse the 2-cpt IV propagator's forced response from a zero initial state.
    if rate_central > 0.0 {
        let mut inflow = [0.0_f64, 0.0];
        propagate_two_cpt(&mut inflow, dt, pk, rate_central, 0.0);
        state[1] += inflow[0];
        state[2] += inflow[1];
    }

    // Constant zero-order input into the depot (#400). The forced response from
    // a zero initial state is `x_ss − e^{A·dt}·x_ss`, where `x_ss` is the full
    // steady state under constant depot input R. At steady state the depot holds
    // R/ka and delivers ka·(R/ka)=R into central, so the central/peripheral SS
    // equals a 2-cpt IV infusion of rate R into central:
    //   central_ss = R/k10,  periph_ss = k12·R/(k21·k10).
    // `e^{A·dt}·x_ss` is exactly this propagator's homogeneous evolution, reused
    // by recursing with zero rates (depth 1; no further recursion).
    if rate_depot > 0.0 {
        let r = rate_depot;
        let denom_ss = k21 * k10;
        let (c_ss, p_ss) = if denom_ss > 1e-30 {
            (r / k10, k12 * r / denom_ss)
        } else {
            (0.0, 0.0)
        };
        let d_ss = r / ka;
        let mut xss = [d_ss, c_ss, p_ss];
        propagate_two_cpt_oral(&mut xss, dt, pk, 0.0, 0.0);
        state[0] += d_ss - xss[0];
        state[1] += c_ss - xss[1];
        state[2] += p_ss - xss[2];
    }
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
/// of the form `(A, B, C)·exp(-ka·t)`. `rate_central` is a constant zero-order
/// input into the central compartment (depot-bypassing infusion, RATE>0 into
/// cmt 2), added by linear superposition. `rate_depot` is a constant zero-order
/// input into the **depot** (RATE>0 into cmt 1, #400): zero-order release into
/// the depot, then first-order `ka` absorption into central.
fn propagate_three_cpt_oral(
    state: &mut [f64],
    dt: f64,
    pk: &PkParams,
    rate_central: f64,
    rate_depot: f64,
) {
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
    let k10 = cl / v1;
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

    // Constant infusion into central (depot bypass) — see propagate_one_cpt_oral.
    // Reuse the 3-cpt IV propagator's forced response from a zero initial state.
    if rate_central > 0.0 {
        let mut inflow = [0.0_f64, 0.0, 0.0];
        propagate_three_cpt(&mut inflow, dt, pk, rate_central, 0.0, 0.0);
        state[1] += inflow[0];
        state[2] += inflow[1];
        state[3] += inflow[2];
    }

    // Constant zero-order input into the depot (#400). Forced response from a
    // zero initial state = `x_ss − e^{A·dt}·x_ss`. At steady state the depot
    // holds R/ka and delivers R into central, so central/peripheral SS equals a
    // 3-cpt IV infusion of rate R into central:
    //   central_ss = R/k10,  p1_ss = k12·R/(k21·k10),  p2_ss = k13·R/(k31·k10).
    // `e^{A·dt}·x_ss` is this propagator's homogeneous evolution, reused by
    // recursing with zero rates (depth 1).
    if rate_depot > 0.0 {
        let r = rate_depot;
        let (c_ss, p1_ss, p2_ss) = if k10 > 1e-30 && k21 > 1e-30 && k31 > 1e-30 {
            (r / k10, k12 * r / (k21 * k10), k13 * r / (k31 * k10))
        } else {
            (0.0, 0.0, 0.0)
        };
        let d_ss = r / ka;
        let mut xss = [d_ss, c_ss, p1_ss, p2_ss];
        propagate_three_cpt_oral(&mut xss, dt, pk, 0.0, 0.0);
        state[0] += d_ss - xss[0];
        state[1] += c_ss - xss[1];
        state[2] += p1_ss - xss[2];
        state[3] += p2_ss - xss[3];
    }
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
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    #[test]
    #[should_panic(expected = "modeled-RATE dose reached the analytical predictor")]
    fn event_driven_predictions_panics_on_modeled_dose() {
        // #324 / review #2: the analytical event-driven predictor must reject a
        // modeled-RATE dose *loudly in release too* (a real `assert!`, not a
        // `debug_assert!`). A modeled dose has rate == 0 but reports is_infusion(),
        // so before the guard it routed into the infusion closed form as a 0-rate
        // "infusion" — the #324 silent-bolus class. Direct call (bypassing the
        // public-entrypoint gates) confirms the predictor itself fails fast.
        use crate::types::RateMode;
        let modeled = DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration);
        let subject = make_subject(vec![modeled], vec![1.0]);
        let pk = pk_one(5.0, 50.0);
        let _ = event_driven_predictions(
            PkModel::OneCptIv,
            &subject,
            std::slice::from_ref(&pk),
            std::slice::from_ref(&pk),
            &[],
        );
    }

    // ── F (bioavailability) & lagtime across RATE options (#324 follow-up) ──
    // The active (event-driven) path must apply F and lagtime correctly to
    // BOTH dose forms reachable via the RATE column: bolus (RATE=0) and
    // infusion (RATE>0). NONMEM scales the bolus *amount* and the infusion
    // *rate* by F (duration unchanged), so concentrations are linear in F; and
    // a lagtime L shifts the whole curve later by L. (Coded RATE -1/-2 never
    // reach here — they are rejected by the datareader, see #324.)

    #[test]
    fn f_bioavailability_scales_bolus_and_infusion_linearly() {
        let (cl, v, amt) = (5.0, 50.0, 100.0);
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        // rate=0 → bolus; rate=25 → infusion (duration = amt/rate = 4 h).
        for &rate in &[0.0_f64, 25.0] {
            let dose = DoseEvent::new(0.0, amt, 1, rate, false, 0.0);
            let subj = make_subject(vec![dose], obs_times.clone());

            let mut pk_full = pk_one(cl, v);
            pk_full.values[crate::types::PK_IDX_F] = 1.0;
            let mut pk_half = pk_one(cl, v);
            pk_half.values[crate::types::PK_IDX_F] = 0.5;

            let full = event_driven_predictions(
                PkModel::OneCptIv,
                &subj,
                &vec![pk_full; 1],
                &vec![pk_full; obs_times.len()],
                &[],
            );
            let half = event_driven_predictions(
                PkModel::OneCptIv,
                &subj,
                &vec![pk_half; 1],
                &vec![pk_half; obs_times.len()],
                &[],
            );

            for (j, (&f, &h)) in full.iter().zip(half.iter()).enumerate() {
                assert!(f > 0.0, "rate={rate}: expected nonzero conc at obs {j}");
                // F=0.5 must halve every concentration (linear in F).
                assert_relative_eq!(h, 0.5 * f, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn f_bioavailability_superposition_matches_event_driven_iv_bolus_and_infusion() {
        // Regression for #327. Bioavailability F must scale IV-bolus and
        // infusion doses on the analytical *superposition* path
        // (`predict_concentration`) exactly as it already does on the
        // event-driven path. Before the fix the superposition path silently
        // dropped F for these routes, so the same model gave F×-different
        // predictions for a no-TV subject (superposition) versus a TV/IOV
        // subject (event-driven). Asserts the two analytical paths agree at
        // F=0.4, including steady-state (SS) bolus and infusion. (Oral-model
        // infusions are covered separately by
        // `f_bioavailability_scales_oral_infusion_on_superposition`: the
        // event-driven path currently drops infusion input on oral models — a
        // distinct bug — so cross-path equality is not assertable there yet.)
        let f = 0.4_f64;
        let obs_times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        let with_f = |mut p: PkParams| {
            p.values[crate::types::PK_IDX_F] = f;
            p
        };

        // rate=0 → bolus; rate>0 → infusion (duration = amt/rate). IV doses go
        // to the central compartment (cmt 1); an oral-model infusion routes to
        // central (cmt 2) — the depot-bypass case.
        let bolus = |cmt: usize| DoseEvent::new(0.0, 100.0, cmt, 0.0, false, 0.0);
        let infusion = |cmt: usize| DoseEvent::new(0.0, 100.0, cmt, 25.0, false, 0.0);
        // Steady-state variants (ss=1, II=24 h). The SS closed forms are linear
        // in the dose too, so `f_scale` must apply there as well — and the
        // event-driven SS equilibration must agree.
        let ss_bolus = |cmt: usize| DoseEvent::new(0.0, 100.0, cmt, 0.0, true, 24.0);
        let ss_infusion = |cmt: usize| DoseEvent::new(0.0, 100.0, cmt, 25.0, true, 24.0);

        let cases: Vec<(&str, PkModel, PkParams, DoseEvent)> = vec![
            (
                "1cpt-iv bolus",
                PkModel::OneCptIv,
                with_f(pk_one(5.0, 50.0)),
                bolus(1),
            ),
            (
                "1cpt-iv infusion",
                PkModel::OneCptIv,
                with_f(pk_one(5.0, 50.0)),
                infusion(1),
            ),
            (
                "2cpt-iv bolus",
                PkModel::TwoCptIv,
                with_f(pk_two(5.0, 40.0, 3.0, 60.0)),
                bolus(1),
            ),
            (
                "2cpt-iv infusion",
                PkModel::TwoCptIv,
                with_f(pk_two(5.0, 40.0, 3.0, 60.0)),
                infusion(1),
            ),
            (
                "3cpt-iv bolus",
                PkModel::ThreeCptIv,
                with_f(pk_three(5.0, 40.0, 3.0, 60.0, 1.0, 120.0)),
                bolus(1),
            ),
            (
                "3cpt-iv infusion",
                PkModel::ThreeCptIv,
                with_f(pk_three(5.0, 40.0, 3.0, 60.0, 1.0, 120.0)),
                infusion(1),
            ),
            // Steady-state (F!=1) across both routes and compartment counts.
            (
                "1cpt-iv SS bolus",
                PkModel::OneCptIv,
                with_f(pk_one(5.0, 50.0)),
                ss_bolus(1),
            ),
            (
                "1cpt-iv SS infusion",
                PkModel::OneCptIv,
                with_f(pk_one(5.0, 50.0)),
                ss_infusion(1),
            ),
            (
                "2cpt-iv SS infusion",
                PkModel::TwoCptIv,
                with_f(pk_two(5.0, 40.0, 3.0, 60.0)),
                ss_infusion(1),
            ),
            (
                "3cpt-iv SS infusion",
                PkModel::ThreeCptIv,
                with_f(pk_three(5.0, 40.0, 3.0, 60.0, 1.0, 120.0)),
                ss_infusion(1),
            ),
        ];

        for (label, model, pk, dose) in cases {
            // The single-dose paths are bit-for-bit equal; the event-driven SS
            // path uses finite-cycle equilibration vs the exact analytical SS
            // closed form, so allow a looser (but still tight) tolerance there.
            // The point of the SS cases is that `F` is applied to the SS arms —
            // any `F`-drop would be a ~F× discrepancy, far above this tolerance.
            let rtol = if dose.ss { 1e-3 } else { 1e-7 };
            let subj = make_subject(vec![dose], obs_times.clone());
            let superposition: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pk))
                .collect();
            let ev = event_driven_predictions(
                model,
                &subj,
                &vec![pk; 1],
                &vec![pk; obs_times.len()],
                &[],
            );
            for (j, (&s, &e)) in superposition.iter().zip(ev.iter()).enumerate() {
                assert!(
                    e > 0.0,
                    "{label}: event-driven conc should be >0 at obs {j}"
                );
                assert_relative_eq!(s, e, epsilon = 1e-9, max_relative = rtol);
            }
        }
    }

    #[test]
    fn f_bioavailability_scales_oral_infusion_on_superposition() {
        // #327: an infusion on an oral model bypasses the depot and enters
        // central directly, so the analytical superposition path must scale it
        // by F. Cross-checking against the event-driven path is deferred —
        // that path currently drops infusion input on oral models entirely (a
        // separate bug). Here we assert F is applied and linear on the
        // superposition path, which is the path that ran F-blind before #327.
        let obs_times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0];
        // rate=25 → 4 h infusion into central (cmt 2) on a 1-cpt oral model.
        let dose = DoseEvent::new(0.0, 100.0, 2, 25.0, false, 0.0);
        let subj = make_subject(vec![dose], obs_times.to_vec());

        let mut pk_full = pk_one_oral(5.0, 50.0, 1.2);
        pk_full.values[crate::types::PK_IDX_F] = 1.0;
        let mut pk_half = pk_one_oral(5.0, 50.0, 1.2);
        pk_half.values[crate::types::PK_IDX_F] = 0.4;

        for &t in &obs_times {
            let full =
                crate::pk::predict_concentration(PkModel::OneCptOral, &subj.doses, t, &pk_full);
            let half =
                crate::pk::predict_concentration(PkModel::OneCptOral, &subj.doses, t, &pk_half);
            assert!(
                full > 0.0,
                "oral infusion should give nonzero conc at t={t}"
            );
            assert_relative_eq!(half, 0.4 * full, max_relative = 1e-12);
        }
    }

    #[test]
    fn lagtime_shifts_bolus_and_infusion_in_time() {
        let (cl, v, amt, lag) = (5.0, 50.0, 100.0, 1.5);
        // Sample strictly after the lag so the lagged curve is "on".
        let obs_times = vec![2.0, 3.0, 5.0, 9.0];
        let shifted: Vec<f64> = obs_times.iter().map(|t| t - lag).collect();
        for &rate in &[0.0_f64, 25.0] {
            // Lagged dose, sampled at t.
            let subj_lag = make_subject(
                vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)],
                obs_times.clone(),
            );
            let mut pk_lag = pk_one(cl, v);
            pk_lag.values[crate::types::PK_IDX_LAGTIME] = lag;
            let lagged = event_driven_predictions(
                PkModel::OneCptIv,
                &subj_lag,
                &vec![pk_lag; 1],
                &vec![pk_lag; obs_times.len()],
                &[],
            );

            // Same dose, no lag, sampled at t-L.
            let subj_nolag = make_subject(
                vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)],
                shifted.clone(),
            );
            let pk_nolag = pk_one(cl, v); // lagtime defaults to 0
            let unlagged = event_driven_predictions(
                PkModel::OneCptIv,
                &subj_nolag,
                &vec![pk_nolag; 1],
                &vec![pk_nolag; shifted.len()],
                &[],
            );

            for (j, (&l, &u)) in lagged.iter().zip(unlagged.iter()).enumerate() {
                assert!(u > 0.0, "rate={rate}: expected nonzero conc at obs {j}");
                // C_lagged(t) == C_unlagged(t - L).
                assert_relative_eq!(l, u, max_relative = 1e-9);
            }
        }
    }

    // ── System resets (EVID=3 / EVID=4) ───────────────────────────────────

    #[test]
    fn reset_evid3_zeros_compartments() {
        // 1-cpt IV bolus at t=0, system reset (EVID=3) at t=5. Observations
        // after the reset must read ~0: the reset emptied every compartment
        // and there is no later dose.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 6.0, 10.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![5.0];
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        assert!(preds[0] > 0.0, "pre-reset obs should be positive");
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(preds[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn reset_evid4_dose_into_emptied_system_matches_fresh_dose() {
        // Dose at t=0, then reset + dose (EVID=4) at t=10. Predictions at and
        // after t=10 must equal a single fresh 500 mg dose given at t=10 — the
        // prior drug is zeroed before the new dose lands.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![10.0, 12.0, 15.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![10.0];
        let pk = pk_one(8.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);

        let fresh = vec![DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0)];
        for (i, &t) in obs_times.iter().enumerate() {
            let expected = crate::pk::predict_concentration(PkModel::OneCptIv, &fresh, t, &pk);
            assert_relative_eq!(preds[i], expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn reset_turns_off_ongoing_infusion() {
        // Infusion 0–8 h started at t=0; a reset at t=4 (mid-infusion) zeros
        // the state AND turns the infusion off, so an obs at t=6 reads ~0.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 125.0, false, 0.0)];
        let obs_times = vec![3.0, 6.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![4.0];
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        assert!(
            preds[0] > 0.0,
            "mid-infusion pre-reset obs should be positive"
        );
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn reset_zeros_depot_for_oral_model() {
        // Oral model: the dose lands in the depot (state[0]) and observations
        // read central (state[1]). A reset must zero BOTH compartments, so an
        // EVID=4 redose reproduces the fresh-dose curve even though drug was
        // still in the depot/central at reset time.
        let mut pk = pk_one(8.0, 50.0);
        pk.values[crate::types::PK_IDX_KA] = 1.2;
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![10.5, 12.0, 16.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![10.0]; // EVID=4 at t=10

        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let preds = event_driven_predictions(PkModel::OneCptOral, &subj, &pk_dose, &pk_obs, &[]);

        let fresh = vec![DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0)];
        for (i, &t) in obs_times.iter().enumerate() {
            let expected = crate::pk::predict_concentration(PkModel::OneCptOral, &fresh, t, &pk);
            assert_relative_eq!(preds[i], expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn reset_zeros_peripheral_for_two_cpt() {
        // 2-cpt: a reset must empty the peripheral compartment too, not just
        // central. After a long first interval drug has distributed into the
        // periphery; a reset+redose must still match a lone fresh dose.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(20.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![20.5, 24.0, 32.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![20.0];
        let pk = pk_two(5.0, 30.0, 2.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::TwoCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let fresh = vec![DoseEvent::new(20.0, 1000.0, 1, 0.0, false, 0.0)];
        for (i, &t) in obs_times.iter().enumerate() {
            let expected = crate::pk::predict_concentration(PkModel::TwoCptIv, &fresh, t, &pk);
            assert_relative_eq!(preds[i], expected, epsilon = 1e-9, max_relative = 1e-9);
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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::TwoCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::TwoCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);

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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);

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
        assert!(supports_event_driven(PkModel::OneCptIv));
        assert!(supports_event_driven(PkModel::OneCptOral));
        assert!(supports_event_driven(PkModel::TwoCptIv));
        assert!(supports_event_driven(PkModel::TwoCptOral));
        assert!(supports_event_driven(PkModel::ThreeCptIv));
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
    fn event_driven_applies_f_bio_to_bolus_dose() {
        // Regression for the SCEN3 F-bioavailability bug: the event-driven
        // path was adding `dose.amt` to the depot without multiplying by F,
        // while the analytical superposition path multiplies F into the
        // amplitude (`d = f_bio * amt * ka / v1`). Models that compute F
        // as a function of dose (`F = (DOSE/100)^γ`) silently lost the F
        // factor whenever the subject had any time-varying covariate (which
        // includes "fake" TV columns like DAY/STIME), so high-dose subjects
        // came out F× too small. Verify event-driven and analytical agree
        // bit-for-bit when F ≠ 1.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());

        let mut pk = pk_two_oral(5.0, 30.0, 2.0, 50.0, 1.5);
        pk.values[crate::types::PK_IDX_F] = 2.5; // non-trivial bioavailability

        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];
        let preds = event_driven_predictions(PkModel::TwoCptOral, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::TwoCptOral, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-8);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }

        // Cross-check: predictions must scale linearly with F.
        let mut pk_unit = pk;
        pk_unit.values[crate::types::PK_IDX_F] = 1.0;
        let pk_dose_unit = vec![pk_unit; 1];
        let pk_obs_unit = vec![pk_unit; obs_times.len()];
        let preds_unit =
            event_driven_predictions(PkModel::TwoCptOral, &subj, &pk_dose_unit, &pk_obs_unit, &[]);
        for (a_f, a_1) in preds.iter().zip(preds_unit.iter()) {
            assert_relative_eq!(*a_f, 2.5 * *a_1, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    #[test]
    fn event_driven_applies_f_bio_to_infusion() {
        // Companion to the bolus test above. `propagate_with_bounds`
        // multiplies `f_bio` into the infusion rate (rate-scaled
        // convention), so a dur→0 infusion limits to a bolus of amount
        // F·AMT.
        //
        // Note: the analytical IV/Infusion paths in `pk/mod.rs` do *not*
        // apply F (only the `_f` oral variants do), so we don't compare
        // event-driven to the analytical reference here. We assert two
        // weaker invariants that the fix should satisfy:
        //   1. At F=1 the event-driven prediction matches the analytical
        //      (regression for the unrelated infusion path).
        //   2. Predictions scale linearly with F (the bolus fix's
        //      contract extended to infusions).
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 500.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());

        let mut pk = pk_one(10.0, 100.0);
        pk.values[crate::types::PK_IDX_F] = 1.0;
        let pk_dose_unit = vec![pk; 1];
        let pk_obs_unit = vec![pk; obs_times.len()];
        let preds_unit =
            event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose_unit, &pk_obs_unit, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::OneCptIv, &subj.doses, t, &pk))
            .collect();
        for (i, (a, e)) in preds_unit.iter().zip(expected.iter()).enumerate() {
            assert_relative_eq!(*a, *e, epsilon = 1e-9, max_relative = 1e-8);
            assert!(*a > 0.0, "obs {} should be positive, got {}", i, a);
        }

        let mut pk_f = pk;
        pk_f.values[crate::types::PK_IDX_F] = 0.4;
        let pk_dose_f = vec![pk_f; 1];
        let pk_obs_f = vec![pk_f; obs_times.len()];
        let preds_f =
            event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose_f, &pk_obs_f, &[]);
        for (a_f, a_1) in preds_f.iter().zip(preds_unit.iter()) {
            assert_relative_eq!(*a_f, 0.4 * *a_1, epsilon = 1e-9, max_relative = 1e-9);
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
    fn event_driven_applies_oral_infusion_into_central() {
        // Regression: the event-driven path previously DROPPED infusion input on
        // oral models — `propagate_*_oral` ignored the central input rate — so an
        // infusion into an oral model's central compartment (cmt 2, depot bypass)
        // returned ~0 instead of the correct curve. This affected TV-covariate,
        // reset, and IOV subjects on oral+infusion models.
        //
        // Three checks:
        //   1. At F=1, event-driven == the superposition reference
        //      (`predict_concentration` models an oral infusion as a depot-
        //      bypassing IV-into-central infusion — the NONMEM result). Before
        //      the fix this was ~0 vs a positive curve.
        //   2. Predictions scale linearly with F (F=0.4 → 0.4×), confirming F is
        //      applied to the infusion rate.
        //   3. Cross-path agreement at F≠1: with #327/#349 now applying F on the
        //      superposition path too, event-driven == superposition at F=0.4 —
        //      not just F=1. Guards against either path silently dropping F
        //      (the gap Ron's #351 review flagged that the F=1-only check (1)
        //      cannot catch).
        let obs_times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0];
        // rate=25 → 4 h infusion into central (cmt 2) on an oral model.
        let dose = DoseEvent::new(0.0, 100.0, 2, 25.0, false, 0.0);
        let subj = make_subject(vec![dose], obs_times.clone());

        let cases: Vec<(&str, PkModel, PkParams)> = vec![
            (
                "1cpt-oral",
                PkModel::OneCptOral,
                pk_one_oral(5.0, 50.0, 1.2),
            ),
            (
                "2cpt-oral",
                PkModel::TwoCptOral,
                pk_two_oral(5.0, 40.0, 3.0, 60.0, 1.2),
            ),
            (
                "3cpt-oral",
                PkModel::ThreeCptOral,
                pk_three_oral(5.0, 40.0, 3.0, 60.0, 1.0, 120.0, 1.2),
            ),
        ];

        for (label, model, base) in cases {
            // (1) F=1: event-driven matches the superposition reference.
            let mut pk1 = base;
            pk1.values[crate::types::PK_IDX_F] = 1.0;
            let ed = event_driven_predictions(
                model,
                &subj,
                &vec![pk1; 1],
                &vec![pk1; obs_times.len()],
                &[],
            );
            let sup: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pk1))
                .collect();
            for (j, (&e, &s)) in ed.iter().zip(sup.iter()).enumerate() {
                assert!(
                    e > 0.0,
                    "{label}: event-driven oral infusion should be >0 at obs {j} \
                     (was dropped before the fix)"
                );
                assert_relative_eq!(e, s, epsilon = 1e-9, max_relative = 1e-7);
            }

            // (2) Linear in F (the infusion rate is scaled by f_bio).
            let mut pkf = base;
            pkf.values[crate::types::PK_IDX_F] = 0.4;
            let ed_f = event_driven_predictions(
                model,
                &subj,
                &vec![pkf; 1],
                &vec![pkf; obs_times.len()],
                &[],
            );
            for (a_f, a_1) in ed_f.iter().zip(ed.iter()) {
                assert_relative_eq!(*a_f, 0.4 * *a_1, max_relative = 1e-9);
            }

            // (3) Cross-path agreement at F≠1: the superposition reference must
            // match the event-driven path at F=0.4, not only F=1. This fails iff
            // one path applies F to the infusion rate and the other does not.
            let sup_f: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pkf))
                .collect();
            for (a_f, s_f) in ed_f.iter().zip(sup_f.iter()) {
                assert_relative_eq!(*a_f, *s_f, epsilon = 1e-9, max_relative = 1e-7);
            }
        }
    }

    #[test]
    fn event_driven_applies_oral_ss_infusion_into_central() {
        // Steady-state companion to the test above. An SS (ss=1, II=24 h)
        // infusion into an oral model's central compartment must equilibrate
        // correctly — it shares `propagate_*_oral`, which previously dropped the
        // input — not return ~0. Cross-path uses a looser tolerance because the
        // event-driven path equilibrates SS with a finite cycle count vs the
        // exact analytical SS closed form; the F-linearity check is exact.
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        // SS infusion into central (cmt 2); dur = amt/rate = 4 h < II.
        let dose = DoseEvent::new(0.0, 100.0, 2, 25.0, true, 24.0);
        let subj = make_subject(vec![dose], obs_times.clone());

        let cases: Vec<(&str, PkModel, PkParams)> = vec![
            (
                "1cpt-oral SS",
                PkModel::OneCptOral,
                pk_one_oral(5.0, 50.0, 1.2),
            ),
            (
                "2cpt-oral SS",
                PkModel::TwoCptOral,
                pk_two_oral(5.0, 40.0, 3.0, 60.0, 1.2),
            ),
            (
                "3cpt-oral SS",
                PkModel::ThreeCptOral,
                pk_three_oral(5.0, 40.0, 3.0, 60.0, 1.0, 120.0, 1.2),
            ),
        ];

        for (label, model, base) in cases {
            let mut pk1 = base;
            pk1.values[crate::types::PK_IDX_F] = 1.0;
            let ed = event_driven_predictions(
                model,
                &subj,
                &vec![pk1; 1],
                &vec![pk1; obs_times.len()],
                &[],
            );
            let sup: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pk1))
                .collect();
            for (j, (&e, &s)) in ed.iter().zip(sup.iter()).enumerate() {
                assert!(
                    e > 0.0,
                    "{label}: SS oral infusion should be >0 at obs {j} (was dropped before the fix)"
                );
                assert_relative_eq!(e, s, epsilon = 1e-9, max_relative = 2e-3);
            }

            // F-linearity (exact — the infusion rate is scaled by f_bio).
            let mut pkf = base;
            pkf.values[crate::types::PK_IDX_F] = 0.4;
            let ed_f = event_driven_predictions(
                model,
                &subj,
                &vec![pkf; 1],
                &vec![pkf; obs_times.len()],
                &[],
            );
            for (a_f, a_1) in ed_f.iter().zip(ed.iter()) {
                assert_relative_eq!(*a_f, 0.4 * *a_1, max_relative = 1e-9);
            }

            // Cross-path agreement at F≠1 (looser SS tolerance, as for F=1):
            // the SS equilibration path also applies F to the infusion rate, so
            // event-driven == superposition at F=0.4, guarding the SS path
            // against silently dropping F.
            let sup_f: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pkf))
                .collect();
            for (a_f, s_f) in ed_f.iter().zip(sup_f.iter()) {
                assert_relative_eq!(*a_f, *s_f, epsilon = 1e-9, max_relative = 2e-3);
            }
        }
    }

    #[test]
    fn event_driven_oral_multi_dose_infusion_matches_superposition() {
        // Review gap: only a single oral central infusion was covered. Two
        // sequential infusions into an oral model's central compartment (cmt 2,
        // depot bypass) must accumulate across intervals on the event-driven
        // path exactly as the superposition reference does — exercising state
        // carried forward between doses (and an infusion still ongoing at an
        // observation), not just a single isolated forced response.
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 2, 25.0, false, 0.0),
            DoseEvent::new(12.0, 100.0, 2, 25.0, false, 0.0),
        ];
        // 12.5 and 14.0 fall inside the second (t=12..16) infusion.
        let obs_times = vec![0.5, 2.0, 4.0, 8.0, 12.5, 14.0, 16.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());

        let cases: Vec<(&str, PkModel, PkParams)> = vec![
            (
                "1cpt-oral",
                PkModel::OneCptOral,
                pk_one_oral(5.0, 50.0, 1.2),
            ),
            (
                "2cpt-oral",
                PkModel::TwoCptOral,
                pk_two_oral(5.0, 40.0, 3.0, 60.0, 1.2),
            ),
            (
                "3cpt-oral",
                PkModel::ThreeCptOral,
                pk_three_oral(5.0, 40.0, 3.0, 60.0, 1.0, 120.0, 1.2),
            ),
        ];

        for (label, model, pk) in cases {
            let pk_dose = vec![pk; subj.doses.len()];
            let pk_obs = vec![pk; obs_times.len()];
            let preds = event_driven_predictions(model, &subj, &pk_dose, &pk_obs, &[]);
            let expected: Vec<f64> = obs_times
                .iter()
                .map(|&t| crate::pk::predict_concentration(model, &subj.doses, t, &pk))
                .collect();
            for (j, (a, e)) in preds.iter().zip(expected.iter()).enumerate() {
                assert!(
                    *a > 0.0,
                    "{label}: obs {j} should be >0 (infusion accumulation)"
                );
                assert_relative_eq!(*a, *e, epsilon = 1e-8, max_relative = 1e-7);
            }
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

        let preds = event_driven_predictions(PkModel::ThreeCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::ThreeCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::ThreeCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let expected: Vec<f64> = obs_times
            .iter()
            .map(|&t| crate::pk::predict_concentration(PkModel::ThreeCptIv, &subj.doses, t, &pk))
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

        let preds = event_driven_predictions(PkModel::ThreeCptIv, &subj, &pk_dose, &pk_obs, &[]);

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

        let preds = event_driven_predictions(PkModel::ThreeCptIv, &subj, &pk_dose, &pk_obs, &[]);
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

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &pk_only);

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
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &[]);

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
        let schedule = EventSchedule::for_subject(&subj, PkModel::OneCptIv, &[]);

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

        let direct = event_driven_predictions(PkModel::TwoCptIv, &subj, &pk_dose, &pk_obs, &[]);
        let schedule = EventSchedule::for_subject(&subj, PkModel::TwoCptIv, &[]);
        let with_sched = event_driven_predictions_with_schedule(
            PkModel::TwoCptIv,
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

    // --- Steady-state (SS=1) tests for the event-driven path ---
    //
    // Cross-checked against the analytical SS closed forms in
    // `src/pk/one_compartment.rs::*_ss`. The event-driven path is what TV-
    // covariate subjects route through; when the per-event PK params are
    // constant the predictions must equal the no-TV closed-form result.

    #[test]
    fn event_driven_ss_iv_bolus_matches_analytical_ss() {
        use crate::pk::one_cpt_iv_bolus_ss;
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let obs_times = vec![1.0, 4.0, 8.0, 11.0, 14.0, 24.0];
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_iv_bolus_ss(&dose, t, cl, v);
            assert_relative_eq!(preds[j], expected, epsilon = 1e-9, max_relative = 1e-7);
        }
    }

    #[test]
    fn event_driven_ss_infusion_matches_analytical_ss() {
        use crate::pk::one_cpt_infusion_ss;
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let rate = 250.0; // T_inf = 4 h
        let ii = 24.0;
        let obs_times = vec![1.0, 3.5, 4.0, 8.0, 12.0, 23.0, 48.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_infusion_ss(&dose, t, cl, v);
            assert_relative_eq!(preds[j], expected, epsilon = 1e-9, max_relative = 1e-7);
        }
    }

    #[test]
    fn event_driven_ss_oral_matches_analytical_ss() {
        use crate::pk::one_cpt_oral_ss;
        let cl = 2.0;
        let v = 20.0;
        let ka = 1.5;
        let amt = 100.0;
        let ii = 24.0;
        let obs_times = vec![0.5, 1.0, 4.0, 12.0, 23.0, 48.0];
        // Oral SS: dose into the depot (cmt = 1 for the depot slot).
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one_oral(cl, v, ka);
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptOral, &subj, &pk_dose, &pk_obs, &[]);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_oral_ss(&dose, t, cl, v, ka);
            assert_relative_eq!(preds[j], expected, epsilon = 1e-9, max_relative = 1e-7);
        }
    }

    #[test]
    fn event_driven_ss_iv_bolus_with_lagtime_matches_nonmem() {
        // Event-driven-path coverage of SS + ALAG1 (issue #15). Same NONMEM
        // 7.5.1 reference as the ODE test (ADVAN1 TRANS2, MAXEVAL=0): CL=5,
        // V=80, ALAG1=2.0, single SS=1 II=12 AMT=1000 IV bolus. Control file
        // + dataset in tests/ss_lagtime_nonmem.rs.
        //
        // t=0.5,1.0,1.5 (< ALAG1=2.0) exercise the previous-interval tail
        // recomputed by `ss_state_at_phase_event_driven`; the plain walk
        // leaves them at ≈0.
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let lagtime = 2.0;
        let nonmem: &[(f64, f64)] = &[
            (0.5, 12.291),
            (1.0, 11.912),
            (1.5, 11.546),
            (2.0, 23.691),
            (3.0, 22.255),
            (6.0, 18.450),
            (11.0, 13.499),
            (13.0, 11.912),
            (18.0, 8.7153),
        ];
        let obs_times: Vec<f64> = nonmem.iter().map(|&(t, _)| t).collect();
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose], obs_times.clone());
        let mut pk = pk_one(cl, v);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let pk_dose = vec![pk; 1];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = event_driven_predictions(PkModel::OneCptIv, &subj, &pk_dose, &pk_obs, &[]);
        for (j, &(_t, pred)) in nonmem.iter().enumerate() {
            assert_relative_eq!(preds[j], pred, max_relative = 1e-4);
        }
    }

    // ── Zero-order input into the oral depot (cmt 1, #400) ────────────────
    // The analytical oral propagators gained a `rate_depot` forced response:
    // zero-order release into the depot, then first-order `ka` absorption into
    // central. Validate each against a fine fixed-step RK4 integration of the
    // same depot-infusion ODE — the ground truth the closed form must match.

    /// RK4 integrate `y' = f(t, y)` from 0 to `t_end` with `n` steps; return y(t_end).
    fn rk4<const D: usize>(
        f: impl Fn(f64, &[f64; D]) -> [f64; D],
        t_end: f64,
        n: usize,
    ) -> [f64; D] {
        let h = t_end / n as f64;
        let mut y = [0.0_f64; D];
        let mut t = 0.0;
        let add = |a: &[f64; D], b: &[f64; D], s: f64| {
            let mut o = [0.0; D];
            for i in 0..D {
                o[i] = a[i] + s * b[i];
            }
            o
        };
        for _ in 0..n {
            let k1 = f(t, &y);
            let k2 = f(t + 0.5 * h, &add(&y, &k1, 0.5 * h));
            let k3 = f(t + 0.5 * h, &add(&y, &k2, 0.5 * h));
            let k4 = f(t + h, &add(&y, &k3, h));
            for i in 0..D {
                y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            t += h;
        }
        y
    }

    #[test]
    fn one_cpt_oral_depot_infusion_matches_numerical_ode() {
        let (cl, v, ka, amt, dur) = (5.0, 50.0, 1.2, 100.0, 3.0);
        let ke = cl / v;
        let rate = amt / dur; // RATE=-2 resolves to amt/duration
        let obs_times = vec![0.5, 1.5, 3.0, 4.0, 6.0, 10.0, 16.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, false, 0.0); // cmt 1 = depot
        let subj = make_subject(vec![dose], obs_times.clone());
        let pk = pk_one_oral(cl, v, ka);

        let preds = event_driven_predictions(
            PkModel::OneCptOral,
            &subj,
            &vec![pk; 1],
            &vec![pk; obs_times.len()],
            &[],
        );

        for (j, &t) in obs_times.iter().enumerate() {
            // ODE: A_d' = R·[t<dur] − ka·A_d ;  A_c' = ka·A_d − ke·A_c
            let y = rk4::<2>(
                |tt, s| {
                    let inp = if tt < dur { rate } else { 0.0 };
                    [inp - ka * s[0], ka * s[0] - ke * s[1]]
                },
                t,
                40_000,
            );
            let expected = y[1] / v;
            assert!(expected > 0.0);
            assert_relative_eq!(preds[j], expected, max_relative = 1e-4);
        }
    }

    #[test]
    fn one_cpt_oral_depot_infusion_ka_equals_ke_branch() {
        // ka == ke hits the L'Hôpital branch of the central forced term.
        let (cl, v, amt, dur) = (5.0, 50.0, 100.0, 4.0);
        let ke = cl / v;
        let ka = ke; // force the singular branch
        let rate = amt / dur;
        let obs_times = vec![0.5, 2.0, 4.0, 7.0, 12.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, false, 0.0);
        let subj = make_subject(vec![dose], obs_times.clone());
        let pk = pk_one_oral(cl, v, ka);

        let preds = event_driven_predictions(
            PkModel::OneCptOral,
            &subj,
            &vec![pk; 1],
            &vec![pk; obs_times.len()],
            &[],
        );
        for (j, &t) in obs_times.iter().enumerate() {
            let y = rk4::<2>(
                |tt, s| {
                    let inp = if tt < dur { rate } else { 0.0 };
                    [inp - ka * s[0], ka * s[0] - ke * s[1]]
                },
                t,
                40_000,
            );
            assert_relative_eq!(preds[j], y[1] / v, max_relative = 1e-4);
        }
    }

    #[test]
    fn two_cpt_oral_depot_infusion_matches_numerical_ode() {
        let (cl, v1, q, v2, ka, amt, dur) = (4.0, 30.0, 6.0, 60.0, 0.9, 120.0, 5.0);
        let k10 = cl / v1;
        let k12 = q / v1;
        let k21 = q / v2;
        let rate = amt / dur;
        let obs_times = vec![0.5, 2.0, 5.0, 7.0, 10.0, 16.0, 24.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, false, 0.0); // depot
        let subj = make_subject(vec![dose], obs_times.clone());
        let pk = pk_two_oral(cl, v1, q, v2, ka);

        let preds = event_driven_predictions(
            PkModel::TwoCptOral,
            &subj,
            &vec![pk; 1],
            &vec![pk; obs_times.len()],
            &[],
        );
        for (j, &t) in obs_times.iter().enumerate() {
            // [A_d, A_c, A_p]
            let y = rk4::<3>(
                |tt, s| {
                    let inp = if tt < dur { rate } else { 0.0 };
                    [
                        inp - ka * s[0],
                        ka * s[0] - (k10 + k12) * s[1] + k21 * s[2],
                        k12 * s[1] - k21 * s[2],
                    ]
                },
                t,
                60_000,
            );
            assert!(y[1] / v1 > 0.0);
            assert_relative_eq!(preds[j], y[1] / v1, max_relative = 1e-4);
        }
    }

    #[test]
    fn three_cpt_oral_depot_infusion_matches_numerical_ode() {
        let (cl, v1, q2, v2, q3, v3, ka, amt, dur) =
            (4.0, 30.0, 6.0, 60.0, 3.0, 120.0, 0.8, 150.0, 6.0);
        let k10 = cl / v1;
        let k12 = q2 / v1;
        let k21 = q2 / v2;
        let k13 = q3 / v1;
        let k31 = q3 / v3;
        let rate = amt / dur;
        let obs_times = vec![0.5, 2.0, 6.0, 8.0, 12.0, 20.0, 36.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, false, 0.0); // depot
        let subj = make_subject(vec![dose], obs_times.clone());
        let pk = pk_three_oral(cl, v1, q2, v2, q3, v3, ka);

        let preds = event_driven_predictions(
            PkModel::ThreeCptOral,
            &subj,
            &vec![pk; 1],
            &vec![pk; obs_times.len()],
            &[],
        );
        for (j, &t) in obs_times.iter().enumerate() {
            // [A_d, A_c, A_p1, A_p2]
            let y = rk4::<4>(
                |tt, s| {
                    let inp = if tt < dur { rate } else { 0.0 };
                    [
                        inp - ka * s[0],
                        ka * s[0] - (k10 + k12 + k13) * s[1] + k21 * s[2] + k31 * s[3],
                        k12 * s[1] - k21 * s[2],
                        k13 * s[1] - k31 * s[3],
                    ]
                },
                t,
                80_000,
            );
            assert!(y[1] / v1 > 0.0);
            assert_relative_eq!(preds[j], y[1] / v1, max_relative = 1e-4);
        }
    }

    #[test]
    fn ss_oral_depot_infusion_matches_many_dose_accumulation() {
        // Steady-state (SS=1) zero-order input into the oral depot (#400) has no
        // closed form, so validate the event-driven SS equilibration + the depot
        // forced response together against the limit of a long explicit
        // multi-dose accumulation (which converges to steady state). This is the
        // SS analogue of the single-dose-vs-RK4 checks and the only test that
        // exercises `equilibrate_ss_state_event_driven` on a depot infusion.
        let (cl, v, ka, amt, ii, dur) = (2.0, 20.0, 1.5, 100.0, 24.0, 4.0);
        let rate = amt / dur; // 25 units/h over 4 h, repeated every 24 h
        let phases = [5.0_f64, 8.0, 12.0, 23.0]; // within-cycle, after infusion ends
        let pk = pk_one_oral(cl, v, ka);

        // SS prediction: a single SS=1 depot infusion.
        let ss_dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
        let ss_subj = make_subject(vec![ss_dose], phases.to_vec());
        let ss_preds = event_driven_predictions(
            PkModel::OneCptOral,
            &ss_subj,
            &vec![pk; 1],
            &vec![pk; phases.len()],
            &[],
        );

        // Reference: 50 explicit (non-SS) depot infusions spaced `ii`, read in the
        // last cycle. ke = CL/V = 0.1 ⇒ per-cycle carryover e^{-0.1·24} ≈ 0.09, so
        // 50 cycles leaves negligible (≈0.09^50) residual below steady state.
        let n_doses = 50usize;
        let last = (n_doses - 1) as f64 * ii;
        let acc_doses: Vec<DoseEvent> = (0..n_doses)
            .map(|k| DoseEvent::new(k as f64 * ii, amt, 1, rate, false, 0.0))
            .collect();
        let acc_times: Vec<f64> = phases.iter().map(|&p| last + p).collect();
        let acc_subj = make_subject(acc_doses, acc_times.clone());
        let acc_preds = event_driven_predictions(
            PkModel::OneCptOral,
            &acc_subj,
            &vec![pk; n_doses],
            &vec![pk; acc_times.len()],
            &[],
        );

        for (j, &p) in phases.iter().enumerate() {
            assert!(ss_preds[j] > 0.0, "phase {p}: SS pred must be nonzero");
            assert_relative_eq!(ss_preds[j], acc_preds[j], max_relative = 1e-5);
        }
    }

    #[test]
    fn oral_depot_infusion_mass_balance() {
        // A zero-order depot input delivers F·AMT into the system: at a long
        // observation the cumulative amount eliminated (∫ CL·C dt) plus the
        // amount still resident must equal AMT. Easier proxy: the AUC over a
        // very long horizon equals AMT / CL (total clearance of the full dose),
        // independent of how it was absorbed. Compare a depot infusion's AUC to
        // a depot bolus of the same AMT — both must clear the same total.
        let (cl, v, ka, amt) = (5.0, 50.0, 1.0, 100.0);
        let pk = pk_one_oral(cl, v, ka);
        // Dense grid out to a long horizon for a trapezoidal AUC.
        let obs_times: Vec<f64> = (0..=4000).map(|i| i as f64 * 0.05).collect();

        let auc = |dose: DoseEvent| -> f64 {
            let subj = make_subject(vec![dose], obs_times.clone());
            let c = event_driven_predictions(
                PkModel::OneCptOral,
                &subj,
                &vec![pk; 1],
                &vec![pk; obs_times.len()],
                &[],
            );
            obs_times
                .windows(2)
                .zip(c.windows(2))
                .map(|(t, cc)| 0.5 * (t[1] - t[0]) * (cc[0] + cc[1]))
                .sum()
        };

        let auc_inf = auc(DoseEvent::new(0.0, amt, 1, amt / 4.0, false, 0.0)); // 4 h depot infusion
        let auc_bolus = auc(DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)); // depot bolus
                                                                           // Both deliver the same AMT; AUC = AMT/CL regardless of absorption shape.
        assert_relative_eq!(auc_inf, amt / cl, max_relative = 2e-3);
        assert_relative_eq!(auc_inf, auc_bolus, max_relative = 2e-3);
    }
}
