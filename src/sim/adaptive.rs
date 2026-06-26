//! Vocabulary for state-reactive ("adaptive" / feedback) dosing simulation (#391).
//!
//! The *types* the reactive driver ([`crate::ode::predictions::ode_predictions_adaptive`])
//! and the public [`crate::api::simulate_adaptive`] entry point are built on. The
//! shapes here are deliberately the public API surface:
//!
//! - a controller is `FnMut(&ControllerCtx) -> Vec<DoseAction>`;
//! - it reads a set of declared [`MonitorSpec`] signals, **each on its own**
//!   [`ObserveMode`] (so PK can be observed latent while a safety marker is
//!   observed with assay noise — see [`ObserveMode`]);
//! - every realized dose is recorded as a [`DoseLedgerEntry`] for the Part-D
//!   output and the Part-E frozen-replay verifier.
//!
//! The engine — not the controller — resolves each monitored signal and draws any
//! assay noise, so draw order stays reproducible for the verifier (#391 S1.5).

use crate::types::DoseEvent;
use std::collections::HashMap;

/// A dosing action a controller can take at a decision time.
///
/// `Hold` and `Stop` carry no payload: `Hold` skips *this* decision (the regimen
/// continues), while `Stop` discontinues all future dosing for the subject. Only
/// `Bolus`/`Infuse` map to a [`DoseEvent`] — see [`DoseAction::to_dose_event`].
///
/// `#[non_exhaustive]`: new actions (e.g. the infusion-truncating safety-halt of
/// #391/#495) land additively without a breaking change. Within `ferx-core` the
/// driver still matches exhaustively — the attribute only forces a wildcard arm
/// in downstream crates (`ferx-r`), so adding a variant here can never silently
/// skip a code path the compiler should have flagged.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DoseAction {
    /// Instantaneous dose of `amt` into 1-based compartment `cmt`.
    Bolus { amt: f64, cmt: usize },
    /// Zero-order infusion of `amt` into `cmt` at `rate` (amount per time); the
    /// duration is `amt / rate`.
    Infuse { amt: f64, cmt: usize, rate: f64 },
    /// Skip this decision — no dose now, the regimen continues.
    Hold,
    /// Discontinue all *future* dosing for this subject. An infusion already in
    /// flight at the `Stop` decision **runs to its scheduled end** — `Stop` halts
    /// the schedule, it does not retract an active infusion. Truncating an
    /// in-flight infusion is a distinct action tracked in #495.
    Stop,
}

impl DoseAction {
    /// Convert an action issued at `time` into a concrete [`DoseEvent`], or
    /// `None` for [`DoseAction::Hold`] / [`DoseAction::Stop`] (which inject
    /// nothing). Bioavailability and lag are applied downstream by the
    /// integrator, exactly as for any `subject.doses` entry — never here.
    ///
    /// Assumes a well-formed action — call [`DoseAction::validate`] first; a
    /// malformed `Infuse`/`Bolus` (e.g. `cmt = 0`, `rate ≤ 0`) is not the
    /// concern of this pure mapping.
    pub fn to_dose_event(&self, time: f64) -> Option<DoseEvent> {
        match *self {
            DoseAction::Bolus { amt, cmt } => Some(DoseEvent::new(time, amt, cmt, 0.0, false, 0.0)),
            DoseAction::Infuse { amt, cmt, rate } => {
                Some(DoseEvent::new(time, amt, cmt, rate, false, 0.0))
            }
            DoseAction::Hold | DoseAction::Stop => None,
        }
    }

    /// `true` for [`DoseAction::Stop`] — the controller has discontinued and the
    /// driver should issue no further decisions for this subject.
    pub fn is_stop(&self) -> bool {
        matches!(self, DoseAction::Stop)
    }

    /// Reject the malformed actions a controller can produce, with a typed error,
    /// before any are applied. Guards the cases that would otherwise corrupt the
    /// integrator: compartment `0` (CMT is 1-based — `cmt - 1` would underflow a
    /// `usize`), a non-positive or non-finite infusion `rate` (which
    /// [`DoseEvent::new`] would silently turn into a zero-duration "infusion",
    /// i.e. a degenerate bolus), and a non-finite / negative `amt`. `Hold`/`Stop`
    /// are always valid. The driver (S1.3a) calls this and surfaces the error
    /// rather than letting a bad action reach the integrator.
    pub fn validate(&self) -> Result<(), String> {
        let (amt, cmt, rate) = match *self {
            DoseAction::Bolus { amt, cmt } => (amt, cmt, None),
            DoseAction::Infuse { amt, cmt, rate } => (amt, cmt, Some(rate)),
            DoseAction::Hold | DoseAction::Stop => return Ok(()),
        };
        if cmt == 0 {
            return Err("dose target compartment is 0, but CMT is 1-based".to_string());
        }
        if !amt.is_finite() || amt < 0.0 {
            return Err(format!(
                "dose amount must be finite and non-negative; got {amt}"
            ));
        }
        if let Some(rate) = rate {
            if !(rate.is_finite() && rate > 0.0) {
                return Err(format!(
                    "Infuse requires a positive, finite rate; got {rate}"
                ));
            }
        }
        Ok(())
    }
}

/// How a monitored signal is observed at a decision time.
///
/// Generalizes "clean prediction vs noisy measurement" to **latent vs realized**,
/// so it extends to non-Gaussian endpoints later (#391 Part F): for a continuous
/// endpoint `Dv` adds the endpoint's residual draw; for an ordinal/TTE endpoint it
/// becomes a *sampled* outcome rather than a Gaussian perturbation. The mode is
/// chosen **per analyte** (see [`MonitorSpec`]), e.g. PK on `Ipred`, a neutrophil
/// count driving CTCAE grading on `Dv` (the grade is defined on the measured lab).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObserveMode {
    /// Latent individual prediction — the model's value, no measurement noise.
    #[default]
    Ipred,
    /// Realized measurement — `Ipred` plus the endpoint's residual draw (the
    /// "assay noise" path; the realistic TDM / grading signal).
    Dv,
}

/// One declared monitored signal the controller can read by `name`.
///
/// `cmt` is the 1-based endpoint/compartment whose error model supplies the
/// residual when `mode == ObserveMode::Dv`; under `Ipred` it selects the latent
/// readout. Each monitor carries its **own** [`ObserveMode`], so a PK signal can
/// run on `Ipred` while a safety marker runs on `Dv` in the same simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorSpec {
    /// Label the controller reads this signal by ([`ControllerCtx::signal`]).
    pub name: String,
    /// 1-based compartment / endpoint this signal observes.
    pub cmt: usize,
    /// Latent (`Ipred`) or realized / assay-noised (`Dv`) — chosen per analyte.
    pub mode: ObserveMode,
}

impl MonitorSpec {
    /// Declare a monitored signal `name` reading endpoint `cmt` under `mode`.
    pub fn new(name: impl Into<String>, cmt: usize, mode: ObserveMode) -> Self {
        Self {
            name: name.into(),
            cmt,
            mode,
        }
    }
}

/// The value a monitored signal actually presented to the controller at a
/// decision, plus the mode it was resolved under — recorded on each
/// [`DoseLedgerEntry`] so decision-audit replay (#391 Part E) can reproduce the
/// exact inputs the controller saw.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedSignal {
    /// Matches the [`MonitorSpec::name`] that produced it.
    pub name: String,
    /// The resolved value (latent or assay-noised per the monitor's mode).
    pub value: f64,
    /// The mode it was resolved under.
    pub mode: ObserveMode,
}

/// Read-only context handed to a controller at each decision time.
///
/// The controller inspects this and returns the [`DoseAction`]s to apply. It
/// borrows the live integration state, so it is valid only for the duration of
/// the call — the driver (S1.3+) owns the timeline. `signals` holds the
/// engine-resolved value of every declared [`MonitorSpec`], each already on its
/// own [`ObserveMode`], so the controller never draws assay noise itself
/// (reproducibility is the engine's responsibility, #391 S1.5).
pub struct ControllerCtx<'a> {
    /// Decision time on the subject's clock.
    pub t: f64,
    /// Current ODE state vector (0-based compartments). The controller may
    /// compute any expression over this directly.
    pub state: &'a [f64],
    /// Subject covariates (LOCF), as the model sees them.
    pub covariates: &'a HashMap<String, f64>,
    /// Doses issued so far this simulation (the realized history).
    pub history: &'a [DoseEvent],
    /// 0-based index of this decision in the schedule.
    pub decision_index: usize,
    /// Resolved monitored signals keyed by [`MonitorSpec::name`] (latent or
    /// assay-noised per each monitor's mode).
    pub signals: &'a HashMap<String, f64>,
}

impl ControllerCtx<'_> {
    /// Value of the monitored signal `name`, or `None` if no such monitor was
    /// declared — `None` rather than a silent `0.0`, so a typo surfaces instead
    /// of quietly driving a wrong decision.
    pub fn signal(&self, name: &str) -> Option<f64> {
        self.signals.get(name).copied()
    }
}

/// One realized dose, with the provenance the Part-D ledger output and the
/// Part-E verifier need. Emitted for every dose a controller actually issues.
#[derive(Debug, Clone, PartialEq)]
pub struct DoseLedgerEntry {
    /// Subject id.
    pub subject: String,
    /// Parameter-uncertainty draw index — matches [`crate::SimulationResult::draw`]
    /// so a ledger row joins to the trajectory rows it produced.
    pub draw: usize,
    /// Replicate index within the draw — matches [`crate::SimulationResult::sim`].
    pub sim: usize,
    /// 0-based index of this dose among the subject's realized doses.
    pub dose_idx: usize,
    /// Time the dose was applied.
    pub time: f64,
    /// Nominal amount (pre-bioavailability).
    pub amt: f64,
    /// 1-based target compartment.
    pub cmt: usize,
    /// Infusion rate (`0.0` for a bolus).
    pub rate: f64,
    /// 0-based decision that produced this dose.
    pub decision_idx: usize,
    /// Human-readable tag for the action/rule that fired (e.g. `"bolus"`, or a
    /// named ladder rule once S2 lands).
    pub rule_fired: String,
    /// What the controller observed at this decision (per-analyte value + mode).
    pub observed_signals: Vec<ObservedSignal>,
    /// State immediately before / after the dose discontinuity — the inputs to
    /// the double-entry / mass-balance checks (S6). `None` when state snapshots
    /// are not retained (verification disabled), so a large run isn't charged two
    /// heap allocations per dose for data nothing consumes.
    pub pre_state: Option<Vec<f64>>,
    /// See [`DoseLedgerEntry::pre_state`].
    pub post_state: Option<Vec<f64>>,
    /// Bioavailable fraction applied to this dose.
    pub f_applied: f64,
}

/// What a controller did at one decision — the audit summary the decision log
/// records alongside the signals it observed.
///
/// The realized doses themselves live in the [`DoseLedgerEntry`] rows tagged with
/// the same `decision_idx`; this only categorizes the decision so a held / no-dose
/// decision (invisible in the dose ledger) is still on the record. `Stop` carries
/// `dosed` so a "give a final dose, then discontinue" action list (`[Bolus, Stop]`)
/// is logged faithfully rather than as a bare stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// The controller issued `n` realized dose(s) (bolus/infusion with `amt > 0`).
    /// Always `n >= 1`: a decision whose actions are all `Hold` or zero-amount is
    /// [`DecisionOutcome::Hold`], not `Dosed { n: 0 }`.
    Dosed { n: usize },
    /// No dose this decision — every action was `Hold`, an empty action list, or a
    /// zero-amount dose.
    Hold,
    /// The controller discontinued; `dosed` counts any dose(s) issued *before* the
    /// `Stop` in the same action list (`0` for a bare stop). `Stop` must be the
    /// final action — the driver rejects any action after it — so `dosed` is a
    /// faithful count, never an undercount from a silently-dropped trailing dose.
    Stop { dosed: usize },
}

/// One decision the controller made, recorded for **every** decision time —
/// including holds and no-change, which leave no [`DoseLedgerEntry`]. This is the
/// Part-D decision log and the input to the Part-E decision-audit replay (S1.6):
/// it pins exactly what the controller saw (`observed_signals`) and what it did
/// (`outcome`) at each decision, so the run can be reproduced and audited.
///
/// Reproducibility assumes the controller decides from its declared
/// `observed_signals`. [`ControllerCtx`] also exposes the raw `state`,
/// `covariates`, and dose `history`, but those are deterministically re-derivable
/// from the frozen inputs + ledger + schedule, so they are not re-stored here. The
/// signals *are* stored: under S1.3a (`Ipred` only) they are likewise re-derivable
/// and serve as a self-contained, directly-auditable record, and under S1.5's `Dv`
/// mode they pin the realized assay-noise draw, so the audit can verify what the
/// controller saw without re-running the stochastic observation.
///
/// A decision reached *after* a `Stop` is not logged — once discontinued the
/// driver issues no further decisions, so the `Stop` entry is the last record.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionLogEntry {
    /// Subject id.
    pub subject: String,
    /// Parameter-uncertainty draw index — matches [`crate::SimulationResult::draw`].
    pub draw: usize,
    /// Replicate index within the draw — matches [`crate::SimulationResult::sim`].
    pub sim: usize,
    /// 0-based index of this decision in the schedule.
    pub decision_idx: usize,
    /// Decision time on the subject's clock.
    pub time: f64,
    /// What the controller observed at this decision (per-analyte value + mode) —
    /// the exact inputs to reproduce the decision.
    pub observed_signals: Vec<ObservedSignal>,
    /// What the controller did (dose / hold / stop).
    pub outcome: DecisionOutcome,
}

/// Result of one reactive-dosing run over a single subject: the observation-time
/// predictions (same layout as [`crate::ode::predictions::ode_predictions`]), the
/// realized-dose ledger, and the per-decision log. S1.4 wraps this with the
/// per-subject/replicate orchestration and the public output schema.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveRun {
    /// Predictions at the subject's observation times (NaN where unrecorded;
    /// negatives clamped to zero, as in the static predictor).
    pub predictions: Vec<f64>,
    /// Every dose the controller actually issued, in time order.
    pub ledger: Vec<DoseLedgerEntry>,
    /// One entry per decision (incl. holds), in schedule order, up to and
    /// including any `Stop`.
    pub decisions: Vec<DecisionLogEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bolus_action_maps_to_instantaneous_dose_event() {
        let de = DoseAction::Bolus { amt: 100.0, cmt: 1 }
            .to_dose_event(5.0)
            .expect("bolus yields a dose event");
        assert_eq!(de.time, 5.0);
        assert_eq!(de.amt, 100.0);
        assert_eq!(de.cmt, 1);
        assert_eq!(de.rate, 0.0);
        assert_eq!(de.duration, 0.0);
        assert!(!de.ss);
        assert_eq!(de.ii, 0.0);
    }

    #[test]
    fn infuse_action_sets_rate_and_derived_duration() {
        let de = DoseAction::Infuse {
            amt: 100.0,
            cmt: 2,
            rate: 25.0,
        }
        .to_dose_event(0.0)
        .expect("infusion yields a dose event");
        assert_eq!(de.cmt, 2);
        assert_eq!(de.rate, 25.0);
        // DoseEvent::new derives duration = amt / rate.
        assert_eq!(de.duration, 4.0);
    }

    #[test]
    fn hold_and_stop_inject_no_dose() {
        assert!(DoseAction::Hold.to_dose_event(1.0).is_none());
        assert!(DoseAction::Stop.to_dose_event(1.0).is_none());
    }

    #[test]
    fn is_stop_only_for_stop() {
        assert!(DoseAction::Stop.is_stop());
        assert!(!DoseAction::Hold.is_stop());
        assert!(!DoseAction::Bolus { amt: 1.0, cmt: 1 }.is_stop());
    }

    #[test]
    fn observe_mode_defaults_to_ipred() {
        assert_eq!(ObserveMode::default(), ObserveMode::Ipred);
    }

    #[test]
    fn monitor_spec_new_sets_fields() {
        let m = MonitorSpec::new("ANC", 3, ObserveMode::Dv);
        assert_eq!(m.name, "ANC");
        assert_eq!(m.cmt, 3);
        assert_eq!(m.mode, ObserveMode::Dv);
    }

    #[test]
    fn controller_ctx_signal_lookup_is_some_only_for_declared() {
        let state = [1.0, 2.0];
        let covariates = HashMap::new();
        let history: Vec<DoseEvent> = Vec::new();
        let mut signals = HashMap::new();
        signals.insert("CONC".to_string(), 1.5);

        let ctx = ControllerCtx {
            t: 24.0,
            state: &state,
            covariates: &covariates,
            history: &history,
            decision_index: 0,
            signals: &signals,
        };

        assert_eq!(ctx.t, 24.0);
        assert_eq!(ctx.signal("CONC"), Some(1.5));
        assert_eq!(ctx.signal("missing"), None);
    }

    #[test]
    fn ledger_entry_and_observed_signal_round_trip() {
        let obs = ObservedSignal {
            name: "CONC".to_string(),
            value: 12.5,
            mode: ObserveMode::Dv,
        };
        let entry = DoseLedgerEntry {
            subject: "1".to_string(),
            draw: 0,
            sim: 0,
            dose_idx: 2,
            time: 48.0,
            amt: 75.0,
            cmt: 1,
            rate: 0.0,
            decision_idx: 2,
            rule_fired: "bolus".to_string(),
            observed_signals: vec![obs.clone()],
            pre_state: Some(vec![0.5, 0.1]),
            post_state: Some(vec![75.5, 0.1]),
            f_applied: 1.0,
        };
        assert_eq!(entry.clone(), entry);
        assert_eq!(entry.observed_signals[0], obs);
        let pre = entry.pre_state.as_ref().unwrap();
        let post = entry.post_state.as_ref().unwrap();
        assert_eq!(post[0] - pre[0], 75.0);
    }

    #[test]
    fn decision_log_entry_round_trips_and_outcomes_are_distinct() {
        let obs = ObservedSignal {
            name: "A".to_string(),
            value: 0.0,
            mode: ObserveMode::Ipred,
        };
        let entry = DecisionLogEntry {
            subject: "1".to_string(),
            draw: 0,
            sim: 0,
            decision_idx: 3,
            time: 72.0,
            observed_signals: vec![obs.clone()],
            outcome: DecisionOutcome::Dosed { n: 2 },
        };
        assert_eq!(entry.clone(), entry);
        assert_eq!(entry.observed_signals[0], obs);
        // The three outcome categories are distinct, and `Stop` carries the count
        // of any dose(s) issued before discontinuation.
        assert_ne!(DecisionOutcome::Hold, DecisionOutcome::Dosed { n: 0 });
        assert_ne!(
            DecisionOutcome::Stop { dosed: 0 },
            DecisionOutcome::Stop { dosed: 1 }
        );
        assert_ne!(DecisionOutcome::Hold, DecisionOutcome::Stop { dosed: 0 });
    }

    #[test]
    fn validate_rejects_zero_compartment() {
        assert!(DoseAction::Bolus { amt: 100.0, cmt: 0 }.validate().is_err());
        assert!(DoseAction::Infuse {
            amt: 100.0,
            cmt: 0,
            rate: 10.0,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn validate_rejects_nonpositive_or_nonfinite_infusion_rate() {
        for rate in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            assert!(
                DoseAction::Infuse {
                    amt: 100.0,
                    cmt: 1,
                    rate,
                }
                .validate()
                .is_err(),
                "rate {rate} should be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_negative_or_nonfinite_amount() {
        assert!(DoseAction::Bolus { amt: -1.0, cmt: 1 }.validate().is_err());
        assert!(DoseAction::Bolus {
            amt: f64::NAN,
            cmt: 1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn validate_accepts_well_formed_actions_and_holds() {
        assert!(DoseAction::Bolus { amt: 100.0, cmt: 1 }.validate().is_ok());
        assert!(DoseAction::Infuse {
            amt: 100.0,
            cmt: 2,
            rate: 25.0,
        }
        .validate()
        .is_ok());
        assert!(DoseAction::Hold.validate().is_ok());
        assert!(DoseAction::Stop.validate().is_ok());
    }
}
