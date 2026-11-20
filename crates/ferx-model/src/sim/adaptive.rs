//! Vocabulary for state-reactive ("adaptive" / feedback) dosing simulation (#391): the
//! types the reactive driver (`crate::ode::predictions::ode_predictions_adaptive`) and
//! `crate::api::simulate_adaptive` are built on. A controller is
//! `FnMut(&ControllerCtx) -> Vec<DoseAction>` reading declared [`MonitorSpec`] signals,
//! each on its own [`ObserveMode`]; every realized dose is a [`DoseLedgerEntry`]. The
//! engine, not the controller, resolves signals and draws assay noise, so draw order
//! stays reproducible for the verifier (#391 S1.5).

use crate::types::DoseEvent;
use std::collections::HashMap;

/// A dosing action a controller can take at a decision time.
///
/// `Hold` skips this decision; `Stop` discontinues all future dosing; `SetInterval`
/// changes the dosing interval from this decision on (#945). Only `Bolus`/`Infuse`
/// inject a [`DoseEvent`]. `#[non_exhaustive]` only forces a wildcard arm downstream;
/// ferx-core still matches exhaustively.
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
    /// Discontinue all future dosing. An infusion in flight runs to its scheduled end
    /// (truncating it is #495).
    Stop,
    /// Set the dosing interval from this decision on (#945). The decision schedule stays
    /// a fixed review grid: a point is due once `interval` has passed since the last
    /// consulted decision, others log [`DecisionOutcome::NotDue`]. The next due time must
    /// land on a grid point or past the last. At most one per decision, never with `Stop`.
    SetInterval { interval: f64 },
}

impl DoseAction {
    /// The [`DoseEvent`] for an action issued at `time`, or `None` for `Hold` / `Stop`.
    /// Bioavailability and lag are applied downstream. Assumes [`DoseAction::validate`]
    /// passed.
    #[cfg(any(test, feature = "test-support"))]
    pub fn to_dose_event(&self, time: f64) -> Option<DoseEvent> {
        match *self {
            DoseAction::Bolus { amt, cmt } => Some(DoseEvent::new(time, amt, cmt, 0.0, false, 0.0)),
            DoseAction::Infuse { amt, cmt, rate } => {
                Some(DoseEvent::new(time, amt, cmt, rate, false, 0.0))
            }
            DoseAction::Hold | DoseAction::Stop | DoseAction::SetInterval { .. } => None,
        }
    }

    /// `true` for [`DoseAction::Stop`] — the controller has discontinued and the
    /// driver should issue no further decisions for this subject.
    pub fn is_stop(&self) -> bool {
        matches!(self, DoseAction::Stop)
    }

    /// Reject malformed actions with a typed error before any are applied: `cmt = 0`
    /// (CMT is 1-based), a non-positive or non-finite infusion `rate` (would become a
    /// degenerate bolus), a non-finite / negative `amt`, and a `SetInterval` that is not
    /// positive and finite. `Hold`/`Stop` are always valid.
    pub fn validate(&self) -> Result<(), String> {
        let (amt, cmt, rate) = match *self {
            DoseAction::Bolus { amt, cmt } => (amt, cmt, None),
            DoseAction::Infuse { amt, cmt, rate } => (amt, cmt, Some(rate)),
            DoseAction::Hold | DoseAction::Stop => return Ok(()),
            DoseAction::SetInterval { interval } => {
                if !(interval.is_finite() && interval > 0.0) {
                    return Err(format!(
                        "SetInterval requires a positive, finite interval; got {interval}"
                    ));
                }
                return Ok(());
            }
        };
        if cmt == 0 {
            return Err("dose target compartment is 0, but CMT is 1-based".to_string());
        }
        if !amt.is_finite() || amt < 0.0 {
            return Err(format!("dose amount must be finite and non-negative; got {amt}"));
        }
        if let Some(rate) = rate {
            if !(rate.is_finite() && rate > 0.0) {
                return Err(format!("Infuse requires a positive, finite rate; got {rate}"));
            }
        }
        Ok(())
    }
}

/// How a monitored signal is observed at a decision time: latent (`Ipred`) or realized
/// (`Dv`, adding the endpoint's residual draw). Chosen per analyte (see [`MonitorSpec`]),
/// e.g. PK on `Ipred` and a lab driving CTCAE grading on `Dv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObserveMode {
    /// Latent individual prediction — the model's value, no measurement noise.
    #[default]
    Ipred,
    /// Realized measurement — `Ipred` plus the endpoint's residual draw (the
    /// "assay noise" path; the realistic TDM / grading signal).
    Dv,
}

/// One declared monitored signal the controller can read by `name`. `cmt` is the
/// 1-based endpoint whose readout (and, under `Dv`, residual error) is used. Each
/// monitor carries its own [`ObserveMode`].
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
        Self { name: name.into(), cmt, mode }
    }
}

/// A monitor paired with its optional compiled `observe` expression, so the two cannot
/// desync. `observe == None` reads the model's `cmt` readout; `Some(f)` is the
/// `[adaptive_dosing]` block's compiled signal expression (#391 S2).
pub struct AdaptiveMonitor<'a> {
    pub spec: &'a MonitorSpec,
    pub observe: Option<&'a crate::ode::OdeOutputFn>,
    /// The `[scaling]` Form A/B divisor of this monitor's readout as a function of the η
    /// in force (#2129). `None` when nothing divides it: no `[scaling]`, a Form C readout,
    /// or an `observe` expression.
    pub obs_scale: Option<&'a dyn Fn(&[f64]) -> f64>,
}

impl AdaptiveMonitor<'_> {
    /// `raw` — the readout at `spec.cmt` — in observation units under `eta`; the
    /// identity when [`Self::obs_scale`] is `None`. An invalid divisor reads `NaN`,
    /// which the driver refuses as a non-finite signal.
    pub fn scale_readout(&self, raw: f64, eta: &[f64]) -> f64 {
        match self.obs_scale {
            Some(scale) => crate::pk::divide_by_obs_scale(raw, scale(eta)),
            None => raw,
        }
    }
}

/// What a controller returns at one decision: its [`DoseAction`]s plus the label of the
/// declarative `when` rule that fired (`None` for a programmatic controller or a no-rule
/// re-issue), recorded in the dose ledger.
pub struct ControllerDecision {
    pub actions: Vec<DoseAction>,
    pub rule: Option<String>,
}

/// The value a monitored signal presented to the controller at a decision and its mode,
/// recorded on each [`DoseLedgerEntry`] for decision-audit replay (#391 Part E), next to
/// the latent value it was read from (#946).
///
/// `#[non_exhaustive]`: built only by the engine, so a field added later is not a
/// breaking change for readers.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ObservedSignal {
    /// Matches the [`MonitorSpec::name`] that produced it.
    pub name: String,
    /// The value the controller saw: the latent readout under [`ObserveMode::Ipred`],
    /// the assay-noised reading (clamped at 0 where the readout cannot be negative) under
    /// [`ObserveMode::Dv`]. Kept for decision audit; the outcome metrics do not read it.
    pub value: f64,
    /// The latent (noise-free) readout at the decision, before any assay draw or clamp.
    /// Equal to `value` bit-for-bit under [`ObserveMode::Ipred`]. The outcome metrics
    /// ([`AdaptiveSubjectMetrics::signal_min`] … `pct_readings_in_window`) are scored on it,
    /// since assay error is not part of the patient's outcome (#946).
    pub latent: f64,
    /// The mode it was resolved under.
    pub mode: ObserveMode,
}

/// Read-only context handed to a controller at each decision time, valid only for the
/// call. `signals` holds the engine-resolved value of every declared [`MonitorSpec`], so
/// the controller never draws assay noise itself (#391 S1.5).
pub struct ControllerCtx<'a> {
    /// Decision time on the subject's monotonic engine clock (#1969), which can run ahead
    /// of the written `TIME` after a reset; a monitor reading `TIME` reads the written
    /// value (#2756), so `signals` and `t` may be on different clocks.
    pub t: f64,
    /// Current ODE state vector (0-based compartments). The controller may
    /// compute any expression over this directly.
    pub state: &'a [f64],
    /// Covariates in force at the decision: the row of the latest record at or before `t`
    /// (at ties: observation, then EVID=2, then dose, then reset), else the baseline.
    /// Every monitored signal and the `F` of a dose issued here use the same row.
    pub covariates: &'a HashMap<String, f64>,
    /// Doses issued so far this simulation (the realized history).
    pub history: &'a [DoseEvent],
    /// 0-based index of this decision in the schedule.
    pub decision_index: usize,
    /// Resolved monitored signals keyed by [`MonitorSpec::name`] (latent or
    /// assay-noised per each monitor's mode).
    pub signals: &'a HashMap<String, f64>,
    /// The dosing interval in force before this decision (#945); `None` when unset
    /// (every grid point due).
    pub interval: Option<f64>,
}

impl ControllerCtx<'_> {
    /// Value of the monitored signal `name`, or `None` if undeclared (not `0.0`, so a
    /// typo surfaces).
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
    /// Parameter-uncertainty draw index — matches [`crate::sim_result::SimulationResult::draw`]
    /// so a ledger row joins to the trajectory rows it produced.
    pub draw: usize,
    /// Replicate index within the draw — matches [`crate::sim_result::SimulationResult::sim`].
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
    /// The signals at this decision, per analyte: the value the controller saw, its
    /// latent readout (which the controller did not see, #946) and the mode.
    pub observed_signals: Vec<ObservedSignal>,
    /// State immediately before / after the dose discontinuity, for the mass-balance
    /// checks (S6). `None` when snapshots are not retained.
    pub pre_state: Option<Vec<f64>>,
    /// See [`DoseLedgerEntry::pre_state`].
    pub post_state: Option<Vec<f64>>,
    /// Bioavailable fraction applied to this dose.
    pub f_applied: f64,
}

/// What a controller did at one decision, logged alongside the signals it observed so a
/// hold (invisible in the dose ledger) is on record. `Stop` carries `dosed` for a
/// `[Bolus, Stop]` list. `#[non_exhaustive]`: read it downstream through
/// [`DecisionOutcome::label`] and [`DecisionOutcome::n_dosed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "test-support"), non_exhaustive)]
pub enum DecisionOutcome {
    /// The controller issued `n >= 1` realized doses (all-`Hold`/zero-amount is `Hold`).
    Dosed { n: usize },
    /// No dose this decision — every action was `Hold`, an empty action list, or a
    /// zero-amount dose.
    Hold,
    /// The controller discontinued; `dosed` counts doses issued before the `Stop` in the
    /// same list. `Stop` must be the final action.
    Stop { dosed: usize },
    /// The grid point was not due under the current interval (#945): not consulted, no
    /// dose, but its monitors were read. Distinct from a consulted `Hold`.
    NotDue,
}

impl DecisionOutcome {
    /// The stable output label: `"dosed"`, `"hold"`, `"stop"` or `"not_due"` (ferx-r's
    /// `decisions$OUTCOME`). Exhaustive here so downstream needs no wildcard (#2581).
    pub fn label(&self) -> &'static str {
        match *self {
            DecisionOutcome::Dosed { .. } => "dosed",
            DecisionOutcome::Hold => "hold",
            DecisionOutcome::Stop { .. } => "stop",
            DecisionOutcome::NotDue => "not_due",
        }
    }

    /// The number of realized doses: `n` for `Dosed`, `dosed` for `Stop`, `0` otherwise.
    /// Exhaustive here, like [`DecisionOutcome::label`] (#2581).
    pub fn n_dosed(&self) -> usize {
        match *self {
            DecisionOutcome::Dosed { n } => n,
            DecisionOutcome::Hold => 0,
            DecisionOutcome::Stop { dosed } => dosed,
            DecisionOutcome::NotDue => 0,
        }
    }
}

/// One decision the controller made, recorded for every decision time (holds included):
/// the Part-D decision log and the input to the Part-E audit replay (S1.6). It pins what
/// the controller saw (`observed_signals`, whose `value` under `Dv` holds the realized
/// assay draw, next to the `latent` readout the controller did not see) and what it did. `state`, covariates and history are re-derivable from the
/// frozen inputs, so they are not stored. Nothing is logged after a `Stop`.
/// `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DecisionLogEntry {
    /// Subject id.
    pub subject: String,
    /// Parameter-uncertainty draw index — matches [`crate::sim_result::SimulationResult::draw`].
    pub draw: usize,
    /// Replicate index within the draw — matches [`crate::sim_result::SimulationResult::sim`].
    pub sim: usize,
    /// 0-based index of this decision in the schedule.
    pub decision_idx: usize,
    /// Decision time on the subject's clock.
    pub time: f64,
    /// The signals at this decision, per analyte: the value the controller saw (the
    /// exact input to reproduce the decision), its latent readout (#946) and the mode.
    pub observed_signals: Vec<ObservedSignal>,
    /// What the controller did (dose / hold / stop), or `NotDue`.
    pub outcome: DecisionOutcome,
    /// The dosing interval in force after this decision (#945); `None` while no
    /// interval is set, i.e. every grid point is due.
    pub interval: Option<f64>,
}

/// Result of one reactive-dosing run over a single subject: the observation-time
/// predictions (as [`crate::ode::predictions::ode_predictions`]), the dose ledger, and
/// the per-decision log.
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

/// Per-subject outcome metrics for one realized adaptive-dosing run (#391 S2.4).
///
/// One row per `(subject, draw, sim)`, the key of the other rows in
/// `AdaptiveSimulationResult`. Every field is computed by
/// `compute_subject_metrics` from that run's dose ledger and decision log alone. The
/// signal summary is over the values observed at decision times (the first monitor), a
/// decision-grid summary, `None` when no signal was recorded. `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveSubjectMetrics {
    /// Subject id (matches [`DoseLedgerEntry::subject`]).
    pub subject: String,
    /// Parameter-uncertainty draw index (matches [`crate::sim_result::SimulationResult::draw`]).
    pub draw: usize,
    /// Replicate index within the draw (matches [`crate::sim_result::SimulationResult::sim`]).
    pub sim: usize,
    /// Total nominal dose delivered — Σ `amt` over the run's ledger rows
    /// (pre-bioavailability: the prescribed amount, matching the ledger).
    pub cumulative_dose: f64,
    /// Number of realized doses (ledger rows). Holds and a bare `Stop` leave no
    /// ledger row, so this counts doses actually given, not decisions.
    pub n_doses: usize,
    /// Times the realized dose stepped up relative to the previous realized dose (by
    /// dose delta, so a re-issued clamped dose is not counted).
    pub n_increases: usize,
    /// Times the realized dose stepped **down** relative to the previous realized
    /// dose (see [`AdaptiveSubjectMetrics::n_increases`]).
    pub n_decreases: usize,
    /// Consulted decisions that issued no dose ([`DecisionOutcome::Hold`]); grid
    /// points that were not due are counted in `n_not_due`, not here.
    pub n_holds: usize,
    /// Grid points at which the controller was not consulted because no dose was
    /// due under the current dosing interval ([`DecisionOutcome::NotDue`], #945).
    pub n_not_due: usize,
    /// Whether the controller discontinued (a [`DecisionOutcome::Stop`] decision).
    pub discontinued: bool,
    /// Time of the `Stop` decision, or `None` if the run never discontinued.
    pub time_to_discontinuation: Option<f64>,
    /// Lowest latent signal ([`ObservedSignal::latent`] of the first monitor) across the
    /// decisions that recorded one, or `None` if no decision recorded a signal. Scored on
    /// the latent value, not the assay-noised reading, so a `Dv` monitor's noise moves
    /// it only through the doses it changed (#946).
    pub signal_min: Option<f64>,
    /// Highest latent signal across the decisions that recorded one (see `signal_min`).
    pub signal_max: Option<f64>,
    /// Mean latent signal across the decisions that recorded one (see `signal_min`).
    pub signal_mean: Option<f64>,
    /// Fraction of the signal-bearing **decision readings** whose latent value
    /// ([`ObservedSignal::latent`] of the first monitor) fell within `target_window`
    /// `[lo, hi]` (inclusive), a fraction in `[0, 1]`. It counts readings, not elapsed
    /// time: on an uneven grid each reading weighs the same. Decisions without a signal
    /// are excluded. `None` when no `target_window` is declared or no signal was recorded
    /// (#584). Latent, like `auc_target_attainment`, since #946, which also renamed it
    /// from `pct_time_in_window`.
    pub pct_readings_in_window: Option<f64>,
    /// Fraction of the closed inter-decision windows whose exposure (AUC of the latent
    /// signal) fell within `auc_target` `[lo, hi]` (inclusive), e.g. vancomycin AUC₂₄
    /// (#391 S2.5b). The denominator is `realized_decisions − 1`; `None` with fewer than
    /// two decisions or no `auc_target`. Integrated on the run's own trajectory with a
    /// node on each side of every discontinuity (#943; the run stays bit-identical).
    /// Scored over realized decisions only: windows after a `Stop` are not counted, since
    /// discontinuation is already its own outcome.
    pub auc_target_attainment: Option<f64>,
    /// Whether this subject's η was supplied (`AdaptiveInputs::with_etas`, #1594) rather
    /// than drawn: replicates then vary by κ and a `Dv` monitor's assay only.
    pub eta_supplied: bool,
    /// Whether a dose expression read the subject's drawn individual parameters, random
    /// effects or mixture class (`dose_knowledge = oracle`, #944).
    pub dose_oracle: bool,
}

/// `true` if any logged decision is a `Stop` that also issued a dose. The realized-window
/// AUC relies on this being false (a dose at a stop has no window to score); it is
/// currently unreachable there, so the call site `debug_assert!`s it.
pub fn run_has_dose_on_stop(decisions: &[DecisionLogEntry]) -> bool {
    decisions.iter().any(|d| matches!(d.outcome, DecisionOutcome::Stop { dosed } if dosed > 0))
}

/// Compute the [`AdaptiveSubjectMetrics`] of one `(subject, draw, sim)` run from its dose
/// `ledger` and decision log alone (#391 S2.4); pure, re-integrates nothing.
/// `window_aucs` is the per-window signal AUC series (`decisions − 1` long, `&[]` with no
/// `auc_target`). The programmatic path passes `None` / `&[]`.
pub fn compute_subject_metrics(
    subject: &str, draw: usize, sim: usize, ledger: &[DoseLedgerEntry],
    decisions: &[DecisionLogEntry], target_window: Option<(f64, f64)>,
    auc_target: Option<(f64, f64)>, window_aucs: &[f64],
) -> AdaptiveSubjectMetrics {
    let cumulative_dose: f64 = ledger.iter().map(|e| e.amt).sum();
    let n_doses = ledger.len();

    // A re-issued dose is bit-identical; a small relative tolerance separates a real
    // step from float noise in compounded percentage steps.
    let mut n_increases = 0usize;
    let mut n_decreases = 0usize;
    for w in ledger.windows(2) {
        let (prev, cur) = (w[0].amt, w[1].amt);
        let tol = 1e-9 * prev.abs().max(cur.abs()).max(1.0);
        if cur > prev + tol {
            n_increases += 1;
        } else if cur < prev - tol {
            n_decreases += 1;
        }
    }

    let n_holds = decisions.iter().filter(|d| matches!(d.outcome, DecisionOutcome::Hold)).count();
    let n_not_due =
        decisions.iter().filter(|d| matches!(d.outcome, DecisionOutcome::NotDue)).count();
    let stop = decisions.iter().find(|d| matches!(d.outcome, DecisionOutcome::Stop { .. }));
    let discontinued = stop.is_some();
    let time_to_discontinuation = stop.map(|d| d.time);

    // Over the first monitor's latent values (#946): the outcome is the patient's, and
    // assay error is not part of it. Decisions without a signal are skipped.
    let signal_vals: Vec<f64> =
        decisions.iter().filter_map(|d| d.observed_signals.first().map(|s| s.latent)).collect();
    let (signal_min, signal_max, signal_mean) = if signal_vals.is_empty() {
        (None, None, None)
    } else {
        // NaN-propagating, unlike `f64::min`/`f64::max`, which drop a NaN. Unreachable
        // through the engine (the monitor loop refuses a non-finite readout, #1578), but a
        // NaN must surface here rather than vanish if that guard ever moves.
        let nan_min = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) };
        let nan_max = |a: f64, b: f64| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) };
        let min = signal_vals.iter().copied().fold(f64::INFINITY, nan_min);
        let max = signal_vals.iter().copied().fold(f64::NEG_INFINITY, nan_max);
        let mean = signal_vals.iter().sum::<f64>() / signal_vals.len() as f64;
        (Some(min), Some(max), Some(mean))
    };
    let pct_readings_in_window = match target_window {
        // A NaN reading fails both comparisons and would count as a miss; surface it
        // instead, as the min/max fold above does (same unreachable-today guard, #1578).
        Some(_) if signal_vals.iter().any(|v| v.is_nan()) => Some(f64::NAN),
        Some((lo, hi)) if !signal_vals.is_empty() => {
            let n_in = signal_vals.iter().filter(|&&v| v >= lo && v <= hi).count();
            Some(n_in as f64 / signal_vals.len() as f64)
        }
        _ => None,
    };

    // Fraction of the integrated windows in band; empty `window_aucs` leaves `None`.
    let auc_target_attainment = match auc_target {
        Some((lo, hi)) if !window_aucs.is_empty() => {
            let n_in = window_aucs.iter().filter(|&&a| a >= lo && a <= hi).count();
            Some(n_in as f64 / window_aucs.len() as f64)
        }
        _ => None,
    };

    AdaptiveSubjectMetrics {
        subject: subject.to_string(),
        draw,
        sim,
        cumulative_dose,
        n_doses,
        n_increases,
        n_decreases,
        n_holds,
        n_not_due,
        discontinued,
        time_to_discontinuation,
        signal_min,
        signal_max,
        signal_mean,
        pct_readings_in_window,
        auc_target_attainment,
        // The caller knows where the η came from; the ledger does not.
        eta_supplied: false,
        dose_oracle: false,
    }
}

// ── Controller-assay RNG substream (#391 S1.5) ──────────────────────────────
//
// A `Dv` monitor reads IPRED + ε·√(residual variance). Each ε is drawn from its own
// substream keyed by `(subject, replicate, decision_index, analyte)`, so the noise is
// deterministic, independent of subject order, and adding a monitor or decision never
// shifts another draw.

/// Purpose tag folded into the assay seed so the controller-assay stream is
/// disjoint from any other stream derived from the same run seed (η, output).
const ASSAY_PURPOSE_SALT: u64 = 0xA55A_E155_0DE0_0001;

/// Purpose salt keeping the IOV-κ stream disjoint from the assay, η and output streams
/// of the same run seed (#701).
const KAPPA_PURPOSE_SALT: u64 = 0xCA99_0CCA_5104_0001;

/// 64-bit golden-ratio odd constant for seed mixing (same family as the
/// per-chain seeding in `estimation/bayes.rs`).
const GOLDEN64: u64 = 0x9E37_79B9_7F4A_7C15;

/// FNV-1a 64-bit hash of a string: stable across platforms and builds (unlike
/// `DefaultHasher`), for keying substreams.
pub fn stable_hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Fold one more key component into a running seed (a splitmix64 finalizer over a
/// golden-ratio-scrambled component). Order-sensitive, so distinct key tuples map
/// to distinct seeds.
pub fn combine_seed(seed: u64, component: u64) -> u64 {
    let mut z = seed ^ component.wrapping_mul(GOLDEN64);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Purpose tag of [`replicate_key`]'s folded key, so a `(draw, rep)` key cannot coincide
/// with a key any other substream derives from the same components.
const REPLICATE_KEY_SALT: u64 = 0x5EED_D2A3_0E9E_0001;

/// The replicate component of the κ and assay substream keys of cell `(sim, draw, rep)`
/// (#946): `sim` itself at `(draw, rep) = (1, 1)`, so a run without uncertainty keeps
/// its keys, else all three folded. Shared by the build loop and the snapshot verifier.
/// The mixture class is keyed by `(draw, sim)`, never `rep`.
pub fn replicate_key(sim: usize, draw: usize, rep: usize) -> usize {
    if draw == 1 && rep == 1 {
        return sim;
    }
    let k = combine_seed(
        combine_seed(combine_seed(REPLICATE_KEY_SALT, draw as u64), rep as u64),
        sim as u64,
    );
    // Truncation on a 32-bit target is still a deterministic key.
    k as usize
}

/// Which replicate a message names (#946): `sim`, and the draw first only when the run
/// draws parameter uncertainty ("draw 2, sim 3").
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReplicateTag {
    /// The parameter draw, `None` when the run draws no parameter uncertainty.
    pub draw: Option<usize>,
    pub sim: usize,
}

impl ReplicateTag {
    /// `"sim 3"`, or `"draw 2, sim 3"` under uncertainty.
    pub fn sim_label(self) -> String {
        match self.draw {
            None => format!("sim {}", self.sim),
            Some(d) => format!("draw {d}, sim {}", self.sim),
        }
    }

    /// `"replicate 3"`, or `"draw 2, replicate 3"` under uncertainty.
    pub fn replicate_label(self) -> String {
        match self.draw {
            None => format!("replicate {}", self.sim),
            Some(d) => format!("draw {d}, replicate {}", self.sim),
        }
    }
}

/// Per-(subject, replicate) base seed for the assay substream, keyed by subject id.
pub fn subject_assay_base_seed(root: u64, subject_id: &str, replicate: usize) -> u64 {
    let s = combine_seed(root ^ ASSAY_PURPOSE_SALT, stable_hash_str(subject_id));
    combine_seed(s, replicate as u64)
}

/// One standard-normal assay draw for `analyte` at `decision_index`, from a fresh RNG
/// seeded by the full key.
pub fn assay_standard_normal(base_seed: u64, decision_index: usize, analyte: &str) -> f64 {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let key =
        combine_seed(combine_seed(base_seed, decision_index as u64), stable_hash_str(analyte));
    let mut rng = rand::rngs::StdRng::seed_from_u64(key);
    Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
}

/// Per-(subject, replicate) base seed for the IOV-κ substream (#701), keyed by subject
/// id and salted apart from the assay stream. A model without IOV never calls it.
pub fn subject_kappa_base_seed(root: u64, subject_id: &str, replicate: usize) -> u64 {
    let s = combine_seed(root ^ KAPPA_PURPOSE_SALT, stable_hash_str(subject_id));
    combine_seed(s, replicate as u64)
}

/// One standard-normal draw for κ `component` on `occasion`, from a fresh RNG seeded by
/// the full key (as [`assay_standard_normal`]); the caller applies the IOV Cholesky
/// factor.
pub fn kappa_standard_normal(base_seed: u64, occasion: usize, component: usize) -> f64 {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let key = combine_seed(combine_seed(base_seed, occasion as u64), component as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(key);
    Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
}

/// Assay-noise capability for `ode_predictions_adaptive` (#391 S1.5): the residual
/// variance resolver (with `ruv_scale` folded in) and the assay base seed.
/// `resid_var(row)` is `Ok(None)` without a residual error model (the driver turns it
/// into a typed error) and `Err` on a magnitude that is not positive and finite (#2741).
pub struct AssayNoise<'a> {
    /// `row -> Ok(Some(residual variance))`, `Ok(None)` without an error model on
    /// `row.cmt`, `Err` on a bad magnitude.
    pub resid_var: &'a dyn Fn(&AssayRow) -> Result<Option<f64>, String>,
    /// Whether the variance reads `TAD` (an `[error_model]` magnitude that names it).
    /// The driver computes [`AssayRow::tad`] only then, as `tad_for` does for a data row.
    pub needs_tad: bool,
    /// Per-(subject, replicate) base seed; see [`subject_assay_base_seed`].
    pub base_seed: u64,
}

/// One `Dv` reading as the residual variance sees it (#2741): `(covariates, TIME, TAD)`
/// at a decision.
pub struct AssayRow<'a> {
    /// The monitored compartment: the endpoint key of a `CMT=N:` block.
    pub cmt: usize,
    /// The latent reading the noise is drawn around.
    pub ipred: f64,
    /// The `TIME` as written at the decision (`Subject::param_time_at`), the same
    /// value the monitor's readout reads.
    pub time: f64,
    /// Time after dose at the decision on the causal record: the base regimen plus
    /// the controller doses realized before the reading. `0.0` when
    /// [`AssayNoise::needs_tad`] is false.
    pub tad: f64,
    /// The covariate row in force at the decision.
    pub cov: &'a HashMap<String, f64>,
}

// ── Declarative `[adaptive_dosing]` block (#391 S2) ─────────────────────────
//
// The parsed AST of the file-driven controller; `observe` stays source text, compiled
// when the controller is built. `parse_adaptive_dosing_block` establishes the invariants.

/// A parsed `[adaptive_dosing]` block: a declarative, file-driven reactive
/// controller (#391 S2).
///
/// **Invariants** (guaranteed by the block parser, assumed downstream): `at` is
/// non-empty and strictly increasing; `dose_bounds.0 <= dose_bounds.1` with
/// `0 <= dose_bounds.0`; `start_dose` lies within `dose_bounds`; `confirm >= 1`;
/// `rules` is non-empty; `levels`, when present, is non-empty and strictly
/// increasing and contains `start_dose` — the dose invariants checked at parse time
/// when every dose value is a number, and per subject, before any subject runs, when
/// one is an expression (#944); and the rule ladder uses percentage
/// dose steps iff `levels` is `None` and one-level steps iff `levels` is `Some`
/// (never mixed). Exactly one signal source: `observe` (a latent expression, with
/// `with_assay_error = false`) **or** `with_assay_error = true` naming a model
/// output via `assay_cmt` — never both.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(feature = "test-support"), non_exhaustive)]
pub struct AdaptiveDosingSpec {
    /// The latent (Ipred) signal expression over states and individual parameters, e.g.
    /// `central / V`; `None` under `with_assay_error`. Rules compare `signal` to it.
    pub observe: Option<String>,
    /// Covariate names the model's `[covariates]` block declares (empty without one).
    /// `compile_observe` needs them for the `T` / `t` precedence (#1028), as a
    /// `[scaling]` Form C readout does, and they are not reachable from a `CompiledModel`.
    pub observe_declared_covariates: Vec<String>,
    /// Titrate on the assay-noised measurement of the output `assay_cmt` (value and σ
    /// from the same output) rather than a latent expression.
    pub with_assay_error: bool,
    /// 1-based compartment measured under `with_assay_error`; required iff it is set.
    pub assay_cmt: Option<usize>,
    /// Decision schedule (subject clock), expanded to explicit times at parse
    /// time — the times the controller is consulted.
    pub at: Vec<f64>,
    /// Dose issued at the first decision; seeds the controller's running dose. A
    /// number, or an expression over θ and data covariates evaluated once per subject
    /// at the first decision (#944).
    pub start_dose: DoseValue,
    /// How an emitted dose is delivered.
    pub route: AdaptiveRoute,
    /// Inclusive `(low, high)` clamp applied to every emitted dose. Each side is a
    /// number or a per-subject expression, like `start_dose` (#944).
    pub dose_bounds: (DoseValue, DoseValue),
    /// Debounce: act only after this many *consecutive* matches of the same rule.
    /// `1` acts on the first breach.
    pub confirm: u32,
    /// Discrete titration levels (oncology). When set, `increase`/`decrease` step
    /// one level; mutually exclusive with percentage steps. Each rung is a number
    /// or a per-subject expression, like `start_dose` (#944).
    pub levels: Option<Vec<DoseValue>>,
    /// The dose grid step (`round_to`, #944): every dose set is rounded to the nearest
    /// multiple (ties up), clamped to the multiples inside `dose_bounds`, and carried on.
    /// Finite and `> 0`; never with `levels`.
    pub round_to: Option<f64>,
    /// What a dose expression may know (#944): with [`DoseKnowledge::Known`] individual
    /// parameters only for supplied η; with [`DoseKnowledge::Oracle`] also drawn values.
    pub dose_knowledge: DoseKnowledge,
    /// Which covariate row a decision carries as a record (#2895). The controller reads
    /// the LOCF row either way.
    pub decision_covariates: DecisionCovariates,
    /// Optional target band `[low, high]` (inclusive, `high` may be `+∞`) for the signal.
    /// Feeds only `pct_readings_in_window` (scored on the latent signal, #946), never dosing;
    /// `None` leaves it unreported. See
    /// `AdaptiveSubjectMetrics`.
    pub target_window: Option<(f64, f64)>,
    /// Optional exposure band `[low, high]` (inclusive; `low` finite `≥ 0`, `high` may be
    /// `+∞`) for the per-window AUC. Feeds only
    /// [`AdaptiveSubjectMetrics::auc_target_attainment`] (#391 S2.5b); `None` skips the
    /// window AUC entirely.
    pub auc_target: Option<(f64, f64)>,
    /// The dosing-interval ladder (#945), strictly increasing and positive, like
    /// `levels` for the dose. A rule's `lengthen` / `shorten` steps one rung and
    /// saturates at both ends. Set together with `start_interval`, or neither.
    pub intervals: Option<Vec<f64>>,
    /// The dosing interval in force from the first decision (#945), one of `intervals`.
    /// `None` consults every grid point.
    pub start_interval: Option<f64>,
    /// The first-matching-rule ladder, in file order.
    pub rules: Vec<AdaptiveRule>,
}

/// One dose value of an `[adaptive_dosing]` block (#944): a number
/// ([`Literal`](DoseValue::Literal), checked at parse time) or an
/// [`Expr`](DoseValue::Expr) over θ and data covariates (e.g. `min(30 * WT, 4000)`),
/// evaluated per subject at the first decision. `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DoseValue {
    /// A number, the same for every subject.
    Literal(f64),
    /// An expression, evaluated per subject.
    Expr(String),
}

/// What a dose expression may know about a subject (`dose_knowledge`, #944): always θ
/// and data covariates; individual parameters, random effects and mixture class only
/// when they come from input. `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DoseKnowledge {
    /// `dose_knowledge = known` (the default): a dose expression reads a subject's
    /// individual parameters only when its η are supplied; a run with a drawn subject
    /// is refused (`E_ADAPTIVE_DOSE_KNOWLEDGE`).
    #[default]
    Known,
    /// `dose_knowledge = oracle`: a drawn subject's dose expressions read its drawn η
    /// and first-occasion κ — a perfect-information comparison arm. Each such subject
    /// has [`AdaptiveSubjectMetrics::dose_oracle`] set, and the run warns
    /// (`W_ADAPTIVE_DOSE_ORACLE`).
    Oracle,
}

/// The covariate row a decision carries as a record (`decision_covariates`, #2895): the
/// interval ending at a decision runs under its occasion κ and `TIME`, as if an `EVID=2`
/// record sat there. `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DecisionCovariates {
    /// `decision_covariates = locf` (the default): the row in force at the decision —
    /// the latest data record at or before it — which is also the row the controller
    /// reads. ferx's reader fills a blank cell the same way.
    #[default]
    Locf,
    /// `decision_covariates = nocb`: the row of the next data record at or after the
    /// decision; past the last data record, the row in force (LOCF).
    Nocb,
}

impl From<f64> for DoseValue {
    fn from(x: f64) -> Self {
        DoseValue::Literal(x)
    }
}

/// Compares a [`DoseValue::Literal`] with a number; an expression equals no number.
impl PartialEq<f64> for DoseValue {
    #[allow(clippy::float_cmp)] // a literal is compared as written
    fn eq(&self, other: &f64) -> bool {
        matches!(self, DoseValue::Literal(x) if x == other)
    }
}

impl DoseValue {
    /// The number, for a literal; `None` for an expression.
    pub fn as_literal(&self) -> Option<f64> {
        match self {
            DoseValue::Literal(x) => Some(*x),
            DoseValue::Expr(_) => None,
        }
    }

    /// The source text, for an expression; `None` for a literal.
    pub fn as_expr(&self) -> Option<&str> {
        match self {
            DoseValue::Literal(_) => None,
            DoseValue::Expr(s) => Some(s),
        }
    }

    /// As written: the number for a literal, the source for an expression.
    pub(crate) fn source(&self) -> String {
        match self {
            DoseValue::Literal(x) => x.to_string(),
            DoseValue::Expr(s) => s.clone(),
        }
    }
}

/// A `levels` ladder of numbers, as the spec holds it (test fixtures).
impl AdaptiveRule {
    /// Does the observed `signal` satisfy this rule's comparison?
    ///
    /// `Eq` uses exact `f64` equality deliberately — it is the user-written
    /// `signal == <value>` semantics; titrating on an exact equality is unusual
    /// but it is what the rule says, not a tolerance we get to invent.
    #[allow(clippy::float_cmp)]
    pub(crate) fn matches(&self, signal: f64) -> bool {
        match self.op {
            Comparison::Lt => signal < self.threshold,
            Comparison::Le => signal <= self.threshold,
            Comparison::Gt => signal > self.threshold,
            Comparison::Ge => signal >= self.threshold,
            Comparison::Eq => signal == self.threshold,
        }
    }
}

#[cfg(test)]
pub fn literal_levels(xs: &[f64]) -> Option<Vec<DoseValue>> {
    Some(xs.iter().map(|&x| DoseValue::Literal(x)).collect())
}

/// The dose values of one subject, every expression evaluated (#944): what the
/// controller runs from.
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectDosePlan {
    pub start_dose: f64,
    pub dose_bounds: (f64, f64),
    pub levels: Option<Vec<f64>>,
}

/// The first dose invariant a [`SubjectDosePlan`] breaks, in
/// [`AdaptiveDosingSpec::validate`]'s order; shared by the parse-time block and the
/// per-subject plan.
#[derive(Debug, Clone, PartialEq)]
pub enum DosePlanViolation {
    Bounds,
    StartNotFinite,
    StartOutsideBounds,
    /// `levels` is empty, not finite, or not strictly increasing; carries the reason
    /// in [`validate_increasing_finite`]'s words.
    LevelsOrder(String),
    StartNotALevel,
    /// The first rung outside the bounds: its index in `levels` and its value.
    LevelOutsideBounds(usize, f64),
    /// `round_to` is set and no multiple of it lies within `dose_bounds` (V7).
    NoGridPoint,
    /// `round_to` is set, the rules take percentage steps, and the lower bound aligns
    /// to 0 on the grid, so a dose can round to 0 and never rise again (V8).
    StuckAtZero,
}

/// The relative tolerance of the dose grid (#944), the one `check_intervals_on_every_grid`
/// and the driver's due test use: `x / s` is rarely exact (`0.3 / 0.1` is
/// `2.9999999999999996`), so a quotient this close to an integer counts as that integer.
const GRID_SNAP: f64 = 1e-9;

/// `q` as an integer when it lies within [`GRID_SNAP`] (relative, at least absolute) of one.
fn snapped_integer(q: f64) -> Option<f64> {
    let r = q.round();
    ((q - r).abs() <= GRID_SNAP * r.abs().max(1.0)).then_some(r)
}

/// The grid indices `ceil(lo / s)` and `floor(hi / s)` of the lowest and highest
/// multiples in `[lo, hi]`, snapping quotients within [`GRID_SNAP`] of an integer
/// (`0.3 / 0.1` floors to 3). A positive low never maps to index 0 (#2805). Empty when
/// the first exceeds the second.
fn grid_index_bounds(lo: f64, hi: f64, s: f64) -> (f64, f64) {
    let (ql, qh) = (lo / s, hi / s);
    let kl = snapped_integer(ql).unwrap_or_else(|| ql.ceil());
    (if lo > 0.0 { kl.max(1.0) } else { kl }, snapped_integer(qh).unwrap_or_else(|| qh.floor()))
}

/// `x` rounded to the nearest multiple of `s`, ties up: the half is nudged up by
/// `GRID_SNAP` (relative), since `x / s` misses a tie by an ulp (`0.35 / 0.1`).
pub fn round_to_grid(x: f64, s: f64) -> f64 {
    let q = x / s;
    grid_point((q + 0.5 + GRID_SNAP * q.abs().max(1.0)).floor(), s)
}

/// The `k`-th multiple of `s`: `k / n` when `1 / s` is within [`GRID_SNAP`] of an integer
/// `n >= 2` (so `7 · 0.1` is `0.7`, not `0.7000000000000001`, #2805), else `k * s`.
fn grid_point(k: f64, s: f64) -> f64 {
    match snapped_integer(1.0 / s) {
        Some(n) if n >= 2.0 => k / n,
        _ => k * s,
    }
}

/// The dose grid of one subject (#944): the step and the grid-aligned bounds
/// `[ceil(lo/s)·s, floor(hi/s)·s]`, each kept inside the written `[lo, hi]` — `k · s`
/// can miss a bound on the grid by an ulp (`3 · 0.1` is `0.30000000000000004`, above a
/// high of `0.3`), so the clamp is what keeps a dose inside `dose_bounds`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoseGrid {
    pub step: f64,
    pub low: f64,
    pub high: f64,
}

impl DoseGrid {
    /// `None` when no multiple of `step` lies within the bounds (V7, refused before any
    /// subject runs).
    pub fn new(step: f64, (lo, hi): (f64, f64)) -> Option<Self> {
        let (kl, kh) = grid_index_bounds(lo, hi, step);
        if kl > kh {
            return None;
        }
        // Each multiple kept inside `[lo, hi]`: a snapped multiple can miss its bound by an
        // ulp either way. The clamp is monotone and `kl <= kh`, so `low <= high`.
        let low = grid_point(kl, step).clamp(lo, hi);
        let high = grid_point(kh, step).clamp(lo, hi);
        Some(Self { step, low, high })
    }

    /// `x` rounded to the grid, then clamped (`hi = 1150, s = 250`: 1150 → 1250 → 1000;
    /// clamping first would give 1250).
    pub fn place(&self, x: f64) -> f64 {
        round_to_grid(x, self.step).clamp(self.low, self.high)
    }

    /// The dose one `p%` step from `d` lands on. The controller's step and the
    /// absorbed-step check (#2810) both call this, so the check cannot disagree with
    /// the engine on a tie.
    pub fn step_percent(&self, d: f64, p: f64, up: bool) -> f64 {
        self.place(percent_step(d, p, up))
    }

    /// The lowest multiple of `step` that `increase p%` raises on an unbounded grid
    /// (#2810); `None` if none up to index 2^52. Found by bisection on
    /// [`Self::step_percent`], since the relative tie nudge makes `50 / p` inexact (#2890).
    pub fn lowest_raised_by(step: f64, p: f64) -> Option<f64> {
        const MAX_INDEX: f64 = 4_503_599_627_370_496.0; // 2^52
        let open = Self { step, low: 0.0, high: f64::INFINITY };
        let raised = |k: f64| {
            let d = grid_point(k, step);
            d.is_finite() && open.step_percent(d, p, true) > d
        };
        // The first power of two that is raised, then the first index between it and
        // the power before it.
        let mut hi = 1.0;
        while !raised(hi) {
            hi *= 2.0;
            if hi > MAX_INDEX {
                return None;
            }
        }
        let mut lo = hi / 2.0;
        while hi - lo > 1.0 {
            let mid = ((lo + hi) / 2.0).floor();
            if raised(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(grid_point(hi, step))
    }
}

/// `d` after one `p%` step, before any rounding or clamp.
pub fn percent_step(d: f64, p: f64, up: bool) -> f64 {
    let factor = if up { 1.0 + p / 100.0 } else { 1.0 - p / 100.0 };
    d * factor
}

/// A dose the smallest `increase N%` cannot raise on the dose grid (#2810): the step
/// rounds back to the same dose at every decision where that rule fires.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbsorbedIncrease {
    /// The start dose, as placed on the grid.
    FromStart(f64),
    /// The start dose can rise, but the lowest dose the decrease rules bring it to
    /// cannot.
    OnceSteppedDown(f64),
}

impl AbsorbedIncrease {
    pub fn dose(self) -> f64 {
        match self {
            Self::FromStart(d) | Self::OnceSteppedDown(d) => d,
        }
    }
}

/// P4 (#944): `round_to` together with a `levels` ladder.
pub const ROUND_TO_WITH_LEVELS: &str = "[adaptive_dosing]: round_to cannot be combined \
     with levels: a levels ladder is already discrete, so write the rounded doses as the levels";

/// P3 (#944): the text for a `round_to` that is not a finite number `> 0`, as written.
pub fn round_to_message(val: &str) -> String {
    format!("[adaptive_dosing]: round_to must be a finite number > 0: {val}")
}

/// The first dose invariant `plan` breaks, if any, under `spec`'s `round_to` and rules.
pub fn dose_plan_violation(
    spec: &AdaptiveDosingSpec, plan: &SubjectDosePlan,
) -> Option<DosePlanViolation> {
    let levels: Option<Vec<Option<f64>>> =
        plan.levels.as_ref().map(|l| l.iter().copied().map(Some).collect());
    known_dose_violation(
        Some(plan.start_dose),
        Some(plan.dose_bounds.0),
        Some(plan.dose_bounds.1),
        levels.as_deref(),
        spec.grid_rule(),
    )
}

/// The first dose invariant broken among the known values (`None` = an unevaluated
/// expression), in [`AdaptiveDosingSpec::validate`]'s order. A check runs only when all
/// its values are known (#2667). The one implementation of each dose invariant.
#[allow(clippy::float_cmp)] // `start_dose ∈ levels` is exact membership
fn known_dose_violation(
    d: Option<f64>, lo: Option<f64>, hi: Option<f64>, levels: Option<&[Option<f64>]>,
    grid: Option<GridRule>,
) -> Option<DosePlanViolation> {
    // `low >= 0` and `low <= high` make a negative `high` invalid on its own.
    let bad = |x: Option<f64>| x.is_some_and(|x| !x.is_finite() || x < 0.0);
    let inverted = matches!((lo, hi), (Some(l), Some(h)) if h < l);
    if bad(lo) || bad(hi) || inverted {
        return Some(DosePlanViolation::Bounds);
    }
    let below = |x: f64| lo.is_some_and(|l| x < l);
    let above = |x: f64| hi.is_some_and(|h| x > h);
    if let Some(d) = d {
        if !d.is_finite() || d < 0.0 {
            return Some(DosePlanViolation::StartNotFinite);
        }
        if below(d) || above(d) {
            return Some(DosePlanViolation::StartOutsideBounds);
        }
    }
    if let Some(l) = levels {
        // The known rungs, in order, must increase: for strictly increasing values a
        // known pair out of order is a violation whatever lies between them.
        let known: Vec<f64> = l.iter().flatten().copied().collect();
        if l.is_empty() || !known.is_empty() {
            if let Err(e) = validate_increasing_finite(&known, "levels") {
                return Some(DosePlanViolation::LevelsOrder(e));
            }
        }
        // A known start that is no known rung may still be an expression rung strictly
        // between its known neighbours (±∞ at the ends) (#2667).
        if let Some(d) = d {
            let expression_rung_can_be_d = (0..l.len()).any(|i| {
                l[i].is_none()
                    && l[..i].iter().rev().flatten().next().is_none_or(|&p| p < d)
                    && l[i + 1..].iter().flatten().next().is_none_or(|&n| d < n)
            });
            if !known.contains(&d) && !expression_rung_can_be_d {
                return Some(DosePlanViolation::StartNotALevel);
            }
        }
        let outside = l
            .iter()
            .enumerate()
            .find_map(|(i, x)| x.filter(|&x| below(x) || above(x)).map(|x| (i, x)));
        if let Some((i, bad)) = outside {
            return Some(DosePlanViolation::LevelOutsideBounds(i, bad));
        }
    }
    // The dose grid (#944), checked last. V7: no multiple within the bounds (needs both);
    // V8: a low bound aligning to 0 (needs only the low).
    if let Some(g) = grid {
        if let (Some(l), Some(h)) = (lo, hi) {
            if DoseGrid::new(g.step, (l, h)).is_none() {
                return Some(DosePlanViolation::NoGridPoint);
            }
        }
        if let Some(l) = lo {
            if g.percent_steps && grid_index_bounds(l, l, g.step).0 == 0.0 {
                return Some(DosePlanViolation::StuckAtZero);
            }
        }
    }
    None
}

/// What the dose-grid checks (V7, V8) need from a spec with `round_to` (#944): the step,
/// and whether a rule takes a percentage step — from a running dose of 0 only those
/// can never climb again.
#[derive(Debug, Clone, Copy)]
pub struct GridRule {
    step: f64,
    percent_steps: bool,
}

impl DosePlanViolation {
    /// The parse-time text: the pre-#944 text byte for byte for an all-numeric block,
    /// else the same sentence with the expression in place of an unknown value.
    fn literal_message(&self, spec: &AdaptiveDosingSpec) -> String {
        let d = spec.start_dose.source();
        let (lo, hi) = (spec.dose_bounds.0.source(), spec.dose_bounds.1.source());
        match self {
            DosePlanViolation::Bounds => format!(
                "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: [{lo}, {hi}]"
            ),
            DosePlanViolation::StartNotFinite => {
                format!("[adaptive_dosing]: start_dose must be finite and >= 0: {d}")
            }
            DosePlanViolation::StartOutsideBounds => {
                format!("[adaptive_dosing]: start_dose {d} is outside dose_bounds [{lo}, {hi}]")
            }
            DosePlanViolation::LevelsOrder(e) => e.clone(),
            DosePlanViolation::StartNotALevel => {
                let rungs = spec.levels.as_deref().unwrap_or(&[]);
                let numbers: Option<Vec<f64>> = rungs.iter().map(DoseValue::as_literal).collect();
                match numbers {
                    // Every rung a number: the pre-#944 text.
                    Some(n) => format!(
                        "[adaptive_dosing]: start_dose {d} must be one of the declared levels {n:?}"
                    ),
                    // A mixed ladder (#2667): the rungs as written and the numbers around
                    // the start dose that no expression rung can sit between.
                    None => {
                        let start = spec.start_dose.as_literal().unwrap_or(f64::NAN);
                        let nums = rungs.iter().filter_map(DoseValue::as_literal);
                        let below = nums.clone().filter(|&x| x < start).reduce(f64::max);
                        let above = nums.filter(|&x| x > start).reduce(f64::min);
                        let gap = match (below, above) {
                            (Some(b), Some(a)) => {
                                format!("between {b} and {a}, the numbers either side of {d}")
                            }
                            (None, Some(a)) => format!("below {a}, the lowest number"),
                            (Some(b), None) => format!("above {b}, the highest number"),
                            (None, None) => "where it could".to_string(),
                        };
                        format!(
                            "[adaptive_dosing]: start_dose {d} must be one of the declared \
                             levels [{}]. It is none of the numbers, and the ladder increases, \
                             so no expression rung can equal it: none lies {gap}.",
                            rungs.iter().map(DoseValue::source).collect::<Vec<_>>().join(", ")
                        )
                    }
                }
            }
            DosePlanViolation::LevelOutsideBounds(_, bad) => {
                format!("[adaptive_dosing]: level {bad} is outside dose_bounds [{lo}, {hi}]")
            }
            DosePlanViolation::NoGridPoint => format!(
                "[adaptive_dosing]: no multiple of round_to = {} lies within dose_bounds \
                 [{lo}, {hi}]",
                spec.round_to.unwrap_or(f64::NAN)
            ),
            DosePlanViolation::StuckAtZero => {
                let s = spec.round_to.unwrap_or(f64::NAN);
                format!(
                    "[adaptive_dosing]: with round_to = {s}, dose_bounds low = {lo} lets a dose \
                     round to 0, and a running dose of 0 cannot rise again under percentage \
                     steps; set the lower bound above 0 (any value up to {s} raises it to {s})"
                )
            }
        }
    }

    /// The per-subject text (#944): the subject (and oracle replicate), each value as
    /// written and evaluated, and the covariate / subject values the expressions read.
    pub fn subject_message(
        &self, spec: &AdaptiveDosingSpec, plan: &SubjectDosePlan, subject_id: &str,
        replicate: Option<ReplicateTag>, reads: &[(String, f64)],
    ) -> String {
        let (lo, hi) = plan.dose_bounds;
        let d = plan.start_dose;
        let head = match replicate {
            None => format!("[adaptive_dosing] subject {subject_id}:"),
            Some(k) => format!("[adaptive_dosing] subject {subject_id}, {}:", k.replicate_label()),
        };
        let start = spec.start_dose.source();
        let bounds = format!("[{}, {}]", spec.dose_bounds.0.source(), spec.dose_bounds.1.source());
        let rungs = spec.levels.as_deref().unwrap_or(&[]);
        let ladder =
            format!("[{}]", rungs.iter().map(DoseValue::source).collect::<Vec<_>>().join(", "));
        let values = plan.levels.as_deref().unwrap_or(&[]);
        let body = match self {
            DosePlanViolation::Bounds => format!(
                "dose_bounds = {bounds} evaluate to [{lo}, {hi}]; they must be finite with \
                 0 <= low <= high."
            ),
            DosePlanViolation::StartNotFinite => {
                format!("start_dose = {start} evaluates to {d}; a dose must be finite and >= 0.")
            }
            DosePlanViolation::StartOutsideBounds => format!(
                "start_dose = {start} evaluates to {d}, outside dose_bounds = {bounds}, which \
                 evaluate to [{lo}, {hi}]."
            ),
            DosePlanViolation::LevelsOrder(_) => format!(
                "levels = {ladder} evaluate to {values:?}; they must be finite and strictly \
                 increasing."
            ),
            DosePlanViolation::StartNotALevel => format!(
                "start_dose = {start} evaluates to {d:?}, which is not one of levels = \
                 {ladder}, which evaluate to {values:?}."
            ),
            DosePlanViolation::LevelOutsideBounds(i, bad) => format!(
                "level {} evaluates to {bad}, outside dose_bounds = {bounds}, which evaluate \
                 to [{lo}, {hi}].",
                rungs.get(*i).map_or_else(|| bad.to_string(), DoseValue::source)
            ),
            DosePlanViolation::NoGridPoint => format!(
                "no multiple of round_to = {} lies within dose_bounds = {bounds}, which \
                 evaluate to [{lo}, {hi}].",
                spec.round_to.unwrap_or(f64::NAN)
            ),
            DosePlanViolation::StuckAtZero => {
                let s = spec.round_to.unwrap_or(f64::NAN);
                format!(
                    "with round_to = {s}, dose_bounds = {bounds} evaluate to [{lo}, {hi}]: a \
                     low of {lo} lets a dose round to 0, and a running dose of 0 cannot rise \
                     again under percentage steps; set the lower bound above 0 (any value up \
                     to {s} raises it to {s})."
                )
            }
        };
        if reads.is_empty() {
            return format!("{head} {body}");
        }
        let read = reads.iter().map(|(n, v)| format!("{n} = {v}")).collect::<Vec<_>>().join(", ");
        format!(
            "{head} {body} The dose expressions read {read}, at the first decision (t = {}).",
            spec.at[0]
        )
    }
}

/// Validate a float list is non-empty, finite and strictly increasing (`levels`, `at`),
/// shared by the block parser and [`AdaptiveDosingSpec::validate`]. `what` names it.
pub fn validate_increasing_finite(xs: &[f64], what: &str) -> Result<(), String> {
    if xs.is_empty() {
        return Err(format!("[adaptive_dosing]: {what} must be a non-empty list"));
    }
    if !xs.iter().all(|x| x.is_finite()) {
        return Err(format!("[adaptive_dosing]: {what} must all be finite"));
    }
    if xs.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!("[adaptive_dosing]: {what} must be strictly increasing"));
    }
    Ok(())
}

impl AdaptiveDosingSpec {
    /// The dose plan every subject shares when every dose value is a number; `None`
    /// when one is an expression (#944).
    pub(crate) fn literal_dose_plan(&self) -> Option<SubjectDosePlan> {
        Some(SubjectDosePlan {
            start_dose: self.start_dose.as_literal()?,
            dose_bounds: (self.dose_bounds.0.as_literal()?, self.dose_bounds.1.as_literal()?),
            levels: match &self.levels {
                Some(l) => Some(l.iter().map(DoseValue::as_literal).collect::<Option<_>>()?),
                None => None,
            },
        })
    }

    /// What the dose-grid checks need (#944): `None` without `round_to`, or with an
    /// invalid one, which `validate` refuses first (P3).
    pub(crate) fn grid_rule(&self) -> Option<GridRule> {
        let step = self.round_to.filter(|s| s.is_finite() && *s > 0.0)?;
        Some(GridRule { step, percent_steps: self.has_percent_steps() })
    }

    /// Whether a rule takes a percentage dose step (`increase N%` / `decrease N%`).
    fn has_percent_steps(&self) -> bool {
        self.rules.iter().any(|r| {
            matches!(
                r.action,
                AdaptiveAction::Increase(DoseStep::Percent(_))
                    | AdaptiveAction::Decrease(DoseStep::Percent(_))
            )
        })
    }

    /// The dose grid of a subject whose bounds evaluate to `bounds` (#944): `None`
    /// without `round_to`. A plan that passed [`dose_plan_violation`] has one.
    pub(crate) fn dose_grid(&self, bounds: (f64, f64)) -> Option<DoseGrid> {
        DoseGrid::new(self.grid_rule()?.step, bounds)
    }

    /// The smallest `increase N%` among the rules, if any. `increase 0%` is not one:
    /// it is written to keep the dose, so a dose it leaves unchanged is no surprise.
    pub(crate) fn smallest_increase_percent(&self) -> Option<f64> {
        self.rules
            .iter()
            .filter_map(|r| match r.action {
                AdaptiveAction::Increase(DoseStep::Percent(p)) => Some(p),
                _ => None,
            })
            .filter(|p| p.is_finite() && *p > 0.0)
            .reduce(f64::min)
    }

    /// The largest `decrease N%` among the rules, if any.
    fn largest_decrease_percent(&self) -> Option<f64> {
        self.rules
            .iter()
            .filter_map(|r| match r.action {
                AdaptiveAction::Decrease(DoseStep::Percent(p)) => Some(p),
                _ => None,
            })
            .filter(|p| p.is_finite() && *p > 0.0)
            .reduce(f64::max)
    }

    /// Whether the dose grid absorbs the smallest increase step at a dose this subject
    /// can reach (#2810); `None` without `round_to`, an increase rule, or absorption. The
    /// lowest reachable dose decides (`d * p / 100` grows with `d`): the placed start, or
    /// where the largest decrease stops. A dose at the aligned high bound is not reported.
    #[allow(clippy::float_cmp)] // grid doses: "unchanged" is the same multiple, bit for bit
    pub(crate) fn absorbed_increase(&self, plan: &SubjectDosePlan) -> Option<AbsorbedIncrease> {
        let grid = self.dose_grid(plan.dose_bounds)?;
        let p = self.smallest_increase_percent()?;
        let absorbed = |d: f64| d < grid.high && grid.step_percent(d, p, true) == d;
        let start = grid.place(plan.start_dose);
        if absorbed(start) {
            return Some(AbsorbedIncrease::FromStart(start));
        }
        let q = self.largest_decrease_percent()?;
        let mut floor = start;
        loop {
            let next = grid.step_percent(floor, q, false);
            // Strictly lower each pass and never below the aligned low, so this ends.
            if !(next < floor) {
                break;
            }
            floor = next;
        }
        absorbed(floor).then_some(AbsorbedIncrease::OnceSteppedDown(floor))
    }

    /// Every dose expression, with the key it is written under (#944).
    pub(crate) fn dose_expressions(&self) -> Vec<(&'static str, &str)> {
        let levels = self.levels.iter().flatten().map(|v| ("levels", v));
        [
            ("start_dose", &self.start_dose),
            ("dose_bounds", &self.dose_bounds.0),
            ("dose_bounds", &self.dose_bounds.1),
        ]
        .into_iter()
        .chain(levels)
        .filter_map(|(k, v)| v.as_expr().map(|s| (k, s)))
        .collect()
    }

    /// Enforce every invariant the block parser checks. Called by the parser and again
    /// by `compile_adaptive` as the safety net for a hand-built `pub` spec.
    pub fn validate(&self) -> Result<(), String> {
        if self.rules.is_empty() {
            return Err("[adaptive_dosing]: at least one `when … : …` rule is required".to_string());
        }
        // Dose invariants among the numeric values (#944, #2667); those involving an
        // expression run per subject in `resolve_dose_plans`. Reported at each check's
        // historical position so the same invariant is named as before #944.
        let levels: Option<Vec<Option<f64>>> =
            self.levels.as_ref().map(|l| l.iter().map(DoseValue::as_literal).collect());
        let violation = known_dose_violation(
            self.start_dose.as_literal(),
            self.dose_bounds.0.as_literal(),
            self.dose_bounds.1.as_literal(),
            levels.as_deref(),
            self.grid_rule(),
        );
        let literal_err = |v: &DosePlanViolation| v.literal_message(self);
        if let Some(
            v @ (DosePlanViolation::Bounds
            | DosePlanViolation::StartNotFinite
            | DosePlanViolation::StartOutsideBounds),
        ) = &violation
        {
            return Err(literal_err(v));
        }
        // `target_window`: low finite, low <= high, high may be +∞.
        if let Some((wlo, whi)) = self.target_window {
            if !wlo.is_finite() || whi.is_nan() || whi < wlo {
                return Err(format!(
                    "[adaptive_dosing]: target_window must be [low, high] with low finite and \
                     low <= high (high may be inf): [{wlo}, {whi}]"
                ));
            }
        }
        // `auc_target`: low finite and >= 0, low <= high, high may be +∞.
        if let Some((alo, ahi)) = self.auc_target {
            if !alo.is_finite() || alo < 0.0 || ahi.is_nan() || ahi < alo {
                return Err(format!(
                    "[adaptive_dosing]: auc_target must be [low, high] with low finite and \
                     0 <= low <= high (high may be inf): [{alo}, {ahi}]"
                ));
            }
        }
        // A non-empty, finite, strictly increasing list of non-negative times.
        validate_increasing_finite(&self.at, "`at` times")?;
        if self.at[0] < 0.0 {
            // Strictly increasing ⇒ `at[0]` is the minimum.
            return Err(format!("[adaptive_dosing]: `at` times must be >= 0: {}", self.at[0]));
        }
        if self.confirm < 1 {
            return Err(
                "[adaptive_dosing]: confirm must be >= 1 (1 acts on the first breach)".to_string()
            );
        }
        // Exactly one signal source: `observe` (latent) or `with_assay_error`.
        match (self.with_assay_error, self.observe.is_some(), self.assay_cmt.is_some()) {
            (false, true, false) => {} // Ipred: titrate on the `observe` expression.
            (true, false, true) => {}  // Dv: titrate on the noised output at `assay_cmt`.
            (false, false, _) => {
                return Err(
                    "[adaptive_dosing]: a signal is required — set `observe = <expr>` to titrate \
                     on a latent quantity, or `with_assay_error = true` with `assay_cmt = N` to \
                     titrate on a noised measurement"
                        .to_string(),
                )
            }
            (false, true, true) => {
                return Err(
                    "[adaptive_dosing]: assay_cmt is set but with_assay_error = false".to_string()
                )
            }
            (true, true, _) => {
                return Err(
                    "[adaptive_dosing]: with `with_assay_error = true` the signal is the noised \
                     measurement of the model output named by `assay_cmt`; remove `observe` (it \
                     applies only to un-noised Ipred titration)"
                        .to_string(),
                )
            }
            (true, false, false) => return Err(
                "[adaptive_dosing]: `with_assay_error = true` requires `assay_cmt = N` to name \
                     which model output is measured"
                    .to_string(),
            ),
        }

        // `round_to` (#944): finite > 0 (P3), never with `levels` (P4); before the ladder.
        if let Some(s) = self.round_to {
            if !s.is_finite() || s <= 0.0 {
                return Err(round_to_message(&s.to_string()));
            }
            if self.levels.is_some() {
                return Err(ROUND_TO_WITH_LEVELS.to_string());
            }
        }
        // Percentage vs one-level dose steps are mutually exclusive and tied to `levels`.
        let has_percent = self.has_percent_steps();
        let has_level = self.rules.iter().any(|r| {
            matches!(
                r.action,
                AdaptiveAction::Increase(DoseStep::Level)
                    | AdaptiveAction::Decrease(DoseStep::Level)
            )
        });
        match &self.levels {
            Some(_) => {
                // A descending ladder would make `increase` lower the dose.
                if let Some(v @ DosePlanViolation::LevelsOrder(_)) = &violation {
                    return Err(literal_err(v));
                }
                if has_percent {
                    return Err("[adaptive_dosing]: percentage actions (`increase N%`) are not \
                                allowed with `levels`; use bare `increase`/`decrease` to step one level"
                        .to_string());
                }
                // Start must be a rung and every rung within `dose_bounds`, or the rung
                // index decouples from the clamped dose.
                if let Some(v) = &violation {
                    return Err(literal_err(v));
                }
            }
            None => {
                if has_level {
                    return Err(
                        "[adaptive_dosing]: bare `increase`/`decrease` (a level step) requires \
                                a `levels = [...]` ladder; use `increase N%` for continuous titration"
                            .to_string(),
                    );
                }
                // V7 / V8 among the bounds written as numbers (#944); with an expression
                // bound they run per subject (`resolve_dose_plans`).
                if let Some(v @ (DosePlanViolation::NoGridPoint | DosePlanViolation::StuckAtZero)) =
                    &violation
                {
                    return Err(literal_err(v));
                }
            }
        }
        self.validate_intervals()
    }

    /// The interval ladder's invariants (#945); the on-grid check needs the `every` step
    /// and stays in the parser.
    fn validate_intervals(&self) -> Result<(), String> {
        let steps_interval = self.rules.iter().any(|r| r.interval.is_some());
        match (&self.intervals, self.start_interval) {
            (None, None) => {
                if steps_interval {
                    return Err("[adaptive_dosing]: `lengthen` / `shorten` require an \
                                `intervals = [...]` ladder and a `start_interval`"
                        .to_string());
                }
            }
            (Some(_), None) => {
                return Err(
                    "[adaptive_dosing]: `intervals` requires a `start_interval` (one of the \
                     intervals), the interval in force from the first decision"
                        .to_string(),
                )
            }
            (None, Some(_)) => {
                return Err(
                    "[adaptive_dosing]: `start_interval` requires an `intervals = [...]` ladder; \
                     for a fixed interval write `intervals = [d]` and `start_interval = d`"
                        .to_string(),
                )
            }
            (Some(l), Some(start)) => {
                validate_increasing_finite(l, "intervals")?;
                // Strictly increasing ⇒ `l[0]` is the minimum.
                if l[0] <= 0.0 {
                    return Err(format!("[adaptive_dosing]: intervals must be > 0: {}", l[0]));
                }
                if !l.contains(&start) {
                    return Err(format!(
                        "[adaptive_dosing]: start_interval {start} must be one of the declared \
                         intervals {l:?}"
                    ));
                }
            }
        }
        if let Some(r) =
            self.rules.iter().find(|r| r.interval.is_some() && r.action == AdaptiveAction::Stop)
        {
            return Err(format!(
                "[adaptive_dosing]: `stop` cannot be combined with `lengthen` / `shorten` (a \
                 stopped course has no next dose): when {}",
                r.label()
            ));
        }
        // `Maintain` is the dose effect of an interval-only rule; without the interval
        // effect the rule does nothing, and its label would name no effect at all.
        if let Some(r) =
            self.rules.iter().find(|r| r.interval.is_none() && r.action == AdaptiveAction::Maintain)
        {
            return Err(format!(
                "[adaptive_dosing]: a rule must have an effect; `Maintain` is only the dose \
                 effect of a rule that also lengthens or shortens the interval: when {}",
                r.label().trim_end()
            ));
        }
        Ok(())
    }
}

/// How an adaptive dose is delivered.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveRoute {
    /// Instantaneous dose into 1-based compartment `cmt`.
    Bolus { cmt: usize },
    /// Zero-order infusion of duration `over` into 1-based compartment `cmt`.
    Infuse { cmt: usize, over: f64 },
}

/// One rung of the ladder: `when signal <op> <threshold> : <action>`.
///
/// `#[non_exhaustive]`: new rule effects land additively.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(feature = "test-support"), non_exhaustive)]
pub struct AdaptiveRule {
    /// The comparison applied to the observed `signal`.
    pub op: Comparison,
    /// The right-hand side of the comparison.
    pub threshold: f64,
    /// What to do when this rule is the first to match.
    pub action: AdaptiveAction,
    /// The rule's interval effect, if any (#945): step the `intervals` ladder one
    /// rung. Composes with `action`, which is then [`AdaptiveAction::Maintain`]
    /// for an interval-only rule.
    pub interval: Option<IntervalStep>,
}

/// A rule's interval effect: one rung along the `intervals` ladder (#945),
/// saturating at both ends.
///
/// `#[non_exhaustive]`: new interval effects land additively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntervalStep {
    /// The next longer interval (`lengthen`).
    Lengthen,
    /// The next shorter interval (`shorten`).
    Shorten,
}

impl IntervalStep {
    pub(crate) fn label(self) -> &'static str {
        match self {
            IntervalStep::Lengthen => "lengthen",
            IntervalStep::Shorten => "shorten",
        }
    }
}

/// A signal comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    Eq,
}

/// What a matched rule does to the running dose.
///
/// `#[non_exhaustive]`: match it with a wildcard arm downstream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AdaptiveAction {
    /// Raise the dose (by a percentage, or one level if the block declares `levels`).
    Increase(DoseStep),
    /// Lower the dose (by a percentage, or one level if the block declares `levels`).
    Decrease(DoseStep),
    /// Skip this decision — no dose now, the regimen continues.
    Hold,
    /// Discontinue all future dosing.
    Stop,
    /// Re-issue the running dose unchanged: the dose effect of a rule that only
    /// changes the interval (`when signal > 15 : lengthen`, #945).
    Maintain,
}

/// Compile-time tripwire for the matches ferx-core can no longer make exhaustive (#3041).
///
/// `AdaptiveAction`, `IntervalStep` and `DecisionCovariates` are `#[non_exhaustive]` and
/// live in this crate, so ferx-core's `AdaptiveController::apply` / `step_interval`
/// (`src/sim/adaptive_control.rs`) and its decision-covariate re-derivation
/// (`src/api/adaptive.rs`) must carry a wildcard arm — which turns a new variant into a
/// runtime `unreachable!` / `Err` instead of the compile error it was before the split.
/// The exhaustive matches below restore that: a new variant fails to compile *here*.
/// Give it its arm in those ferx-core sites first, then list it here.
const _: fn(&AdaptiveAction, IntervalStep, DecisionCovariates) = |action, step, rule| {
    match action {
        AdaptiveAction::Increase(_)
        | AdaptiveAction::Decrease(_)
        | AdaptiveAction::Hold
        | AdaptiveAction::Stop
        | AdaptiveAction::Maintain => {}
    }
    match step {
        IntervalStep::Lengthen | IntervalStep::Shorten => {}
    }
    match rule {
        DecisionCovariates::Locf | DecisionCovariates::Nocb => {}
    }
};

/// The magnitude of an [`AdaptiveAction::Increase`] / [`AdaptiveAction::Decrease`].
#[derive(Debug, Clone, PartialEq)]
pub enum DoseStep {
    /// Scale by this percent, e.g. `Percent(25.0)` for `increase 25%`. Valid only
    /// when the block has no `levels` ladder.
    Percent(f64),
    /// Step one discrete level. Valid only when the block declares `levels`.
    Level,
}

impl Comparison {
    /// The operator's source symbol, for rule labels / diagnostics.
    fn symbol(self) -> &'static str {
        match self {
            Comparison::Lt => "<",
            Comparison::Le => "<=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
            Comparison::Eq => "==",
        }
    }
}

impl AdaptiveAction {
    /// The action's source-like label (`"increase 25%"`, `"hold"`, …).
    pub(crate) fn label(&self) -> String {
        match self {
            AdaptiveAction::Increase(DoseStep::Percent(p)) => format!("increase {p}%"),
            AdaptiveAction::Increase(DoseStep::Level) => "increase".to_string(),
            AdaptiveAction::Decrease(DoseStep::Percent(p)) => format!("decrease {p}%"),
            AdaptiveAction::Decrease(DoseStep::Level) => "decrease".to_string(),
            AdaptiveAction::Hold => "hold".to_string(),
            AdaptiveAction::Stop => "stop".to_string(),
            AdaptiveAction::Maintain => String::new(),
        }
    }
}

impl AdaptiveRule {
    /// A human-readable label reconstructing the rule as written
    /// (`"signal < 10 : increase 25%"`) — recorded as the dose ledger's
    /// `rule_fired` so an audit can name which rung produced each dose.
    pub(crate) fn label(&self) -> String {
        let effects: Vec<String> = [self.action.label()]
            .into_iter()
            .chain(self.interval.map(|i| i.label().to_string()))
            .filter(|e| !e.is_empty())
            .collect();
        format!("signal {} {} : {}", self.op.symbol(), self.threshold, effects.join(", "))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn bolus_action_maps_to_instantaneous_dose_event() {
        let de = DoseAction::Bolus { amt: 100.0, cmt: 1 }
            .to_dose_event(5.0)
            .expect("bolus yields a dose event");
        assert_eq!(de.time, 5.0);
        assert_eq!(de.amt, 100.0);
        assert_eq!(de.cmt_raw(), 1);
        assert_eq!(de.rate, 0.0);
        assert_eq!(de.duration, 0.0);
        assert!(!de.ss);
        assert_eq!(de.ii, 0.0);
    }

    #[test]
    fn infuse_action_sets_rate_and_derived_duration() {
        let de = DoseAction::Infuse { amt: 100.0, cmt: 2, rate: 25.0 }
            .to_dose_event(0.0)
            .expect("infusion yields a dose event");
        assert_eq!(de.cmt_raw(), 2);
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

    fn base_spec() -> AdaptiveDosingSpec {
        AdaptiveDosingSpec {
            observe: Some("central".to_string()),
            observe_declared_covariates: Vec::new(),
            with_assay_error: false,
            assay_cmt: None,
            at: vec![24.0],
            start_dose: (100.0).into(),
            route: AdaptiveRoute::Bolus { cmt: 1 },
            dose_bounds: ((0.0).into(), (400.0).into()),
            confirm: 1,
            levels: None,
            target_window: None,
            auc_target: None,
            round_to: None,
            dose_knowledge: Default::default(),
            decision_covariates: Default::default(),
            rules: vec![AdaptiveRule {
                op: Comparison::Lt,
                threshold: 10.0,
                action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
                interval: None,
            }],
            intervals: None,
            start_interval: None,
        }
    }

    // ── round_to: the dose grid (#944); closed forms from audit 1 ─────────────────

    /// Ties go up: `0.35 / 0.1` (`3.4999999999999996`) gives 0.4, `375` at `s = 250` gives
    /// 500; just below a tie goes down.
    #[test]
    fn round_to_grid_sends_ties_up_on_a_decimal_grid() {
        assert_eq!((0.35f64 / 0.1).round(), 3.0, "premise: plain round sends it down");
        assert!((round_to_grid(0.35, 0.1) - 0.4).abs() < 1e-12, "{}", round_to_grid(0.35, 0.1));
        assert_eq!(round_to_grid(375.0, 250.0).to_bits(), 500.0f64.to_bits());
        assert_eq!(round_to_grid(374.999, 250.0).to_bits(), 250.0f64.to_bits());
        assert_eq!(round_to_grid(0.0, 250.0).to_bits(), 0.0f64.to_bits());
    }

    /// A step that divides 1 gives the decimal multiple, not `k * s`: `7 * 0.1` is
    /// `0.7000000000000001`, but the grid dose for 0.7 at `s = 0.1` is `0.7` exactly, as
    /// are `0.3` (`3 * 0.1 = 0.30000000000000004`) and the multiples of 0.25. An integer
    /// step is `k * s` bit for bit, so integer grids (and D2) are unchanged (#2805 review
    /// r1 row 7).
    ///
    /// Mutation: `grid_point` back to `k * s` (the decimal cases print ulp noise).
    #[test]
    fn a_step_dividing_one_gives_the_decimal_multiple() {
        assert_eq!(7.0 * 0.1, 0.7000000000000001, "premise: k * s is noisy");
        assert_eq!(round_to_grid(0.7, 0.1).to_bits(), 0.7f64.to_bits());
        assert_eq!(round_to_grid(0.3, 0.1).to_bits(), 0.3f64.to_bits());
        assert_eq!(round_to_grid(0.35, 0.1).to_bits(), 0.4f64.to_bits());
        assert_eq!(round_to_grid(1.3, 0.25).to_bits(), 1.25f64.to_bits());
        let g = DoseGrid::new(0.1, (0.05, 0.75)).expect("grid");
        assert_eq!((g.low.to_bits(), g.high.to_bits()), (0.1f64.to_bits(), 0.7f64.to_bits()));
        for (x, k) in [(1300.0, 5.0), (375.0, 2.0), (4000.0, 16.0)] {
            assert_eq!(round_to_grid(x, 250.0).to_bits(), (k * 250.0f64).to_bits(), "{x}");
        }
    }

    /// `hi = 0.3, s = 0.1, x = 0.3 → 0.3`: `0.3 / 0.1` is `2.9999999999999996`, which a
    /// bare `floor` takes to 2 (an aligned high of 0.2). The snap takes it to 3, and the
    /// clamp keeps `3 · 0.1 = 0.30000000000000004` at the written 0.3. On the low side
    /// `2.1 / 0.3` is `7.000000000000001`, which a bare `ceil` takes to 8; the snap
    /// keeps it at 7. `[0.3, 0.3]` holds exactly one multiple, which a bare `floor`
    /// would lose (V7 on a valid block).
    ///
    /// Mutations: a bare `floor` for the high bound; a bare `ceil` for the low bound; the
    /// aligned bound not kept inside the written bounds.
    #[test]
    fn dose_grid_snaps_a_decimal_bound_onto_the_grid() {
        let g = DoseGrid::new(0.1, (0.0, 0.3)).expect("0, 0.1, 0.2 and 0.3 lie inside");
        assert_eq!(g.high.to_bits(), 0.3f64.to_bits(), "{g:?}");
        assert_eq!(g.place(0.3).to_bits(), 0.3f64.to_bits(), "{g:?}");
        assert!((2.1f64 / 0.3).ceil() == 8.0, "premise: a bare ceil overshoots");
        let g = DoseGrid::new(0.3, (2.1, 3.0)).expect("grid");
        assert!((g.low - 2.1).abs() < 1e-12, "{g:?}");
        let g = DoseGrid::new(0.1, (0.3, 0.3)).expect("0.3 is a multiple of 0.1");
        assert_eq!((g.low.to_bits(), g.high.to_bits()), (0.3f64.to_bits(), 0.3f64.to_bits()));
    }

    /// `hi = 1150, s = 250`: round then clamp gives 1000 (row 13); `lo = 1e-6` aligns to
    /// 250, so 100 becomes 250 (row 14).
    #[test]
    fn dose_grid_rounds_then_clamps_to_the_aligned_bounds() {
        let g = DoseGrid::new(250.0, (0.0, 1150.0)).expect("grid");
        assert_eq!(g.high.to_bits(), 1000.0f64.to_bits(), "{g:?}");
        assert_eq!(g.place(1150.0).to_bits(), 1000.0f64.to_bits());
        let g = DoseGrid::new(250.0, (1e-6, 4000.0)).expect("grid");
        assert_eq!(g.low.to_bits(), 250.0f64.to_bits(), "{g:?}");
        assert_eq!(g.place(100.0).to_bits(), 250.0f64.to_bits());
        assert!(DoseGrid::new(250.0, (300.0, 400.0)).is_none(), "no multiple in [300, 400]");
        // A single grid point, with a low a hair above it (within the snap): the aligned
        // bounds collapse onto the written low rather than crossing.
        let lo = 250.0 * (1.0 + 1e-12);
        let g = DoseGrid::new(250.0, (lo, 260.0)).expect("250 snaps in");
        assert!(g.low <= g.high, "{g:?}");
        assert_eq!(g.place(255.0).to_bits(), lo.to_bits(), "{g:?}");
    }

    fn gridded(round_to: f64, bounds: (f64, f64), start: f64) -> AdaptiveDosingSpec {
        AdaptiveDosingSpec {
            round_to: Some(round_to),
            dose_bounds: (bounds.0.into(), bounds.1.into()),
            start_dose: start.into(),
            ..base_spec()
        }
    }

    // ── #2810: the smallest increase step absorbed at a reachable dose (`s = 250`) ──

    /// `round_to = 250` with the rules `increase up%` (each of `ups`) and `decrease
    /// down%` (each of `downs`).
    fn stepped(bounds: (f64, f64), start: f64, ups: &[f64], downs: &[f64]) -> AdaptiveDosingSpec {
        let rule = |op, action| AdaptiveRule { op, threshold: 10.0, action, interval: None };
        let mut s = gridded(250.0, bounds, start);
        s.rules = ups
            .iter()
            .map(|&p| rule(Comparison::Lt, AdaptiveAction::Increase(DoseStep::Percent(p))))
            .chain(
                downs
                    .iter()
                    .map(|&q| rule(Comparison::Gt, AdaptiveAction::Decrease(DoseStep::Percent(q)))),
            )
            .collect();
        s
    }

    fn absorbed(spec: &AdaptiveDosingSpec) -> Option<AbsorbedIncrease> {
        spec.absorbed_increase(&spec.literal_dose_plan().expect("a block of numbers"))
    }

    /// Threshold straddle: from 500, `increase 25%` (625, a tie, up to 750) is not
    /// absorbed; `increase 24.99%` (624.95, back to 500) is.
    #[test]
    fn an_increase_just_below_half_a_step_is_absorbed_and_one_at_half_is_not() {
        let half: f64 = 250.0 / 2.0;
        let (at, below) = (25.0, 24.99);
        assert_eq!((percent_step(500.0, at, true) - 500.0).to_bits(), half.to_bits());
        let short = percent_step(500.0, below, true) - 500.0;
        assert!(short < half && short > half - 0.1, "{short}");

        let bounds = (250.0, 4000.0);
        assert_eq!(absorbed(&stepped(bounds, 500.0, &[at], &[])), None);
        assert_eq!(
            absorbed(&stepped(bounds, 500.0, &[below], &[])),
            Some(AbsorbedIncrease::FromStart(500.0))
        );
    }

    /// Under `increase 10%`: start 1000 is absorbed; start 1500 rises but `decrease 25%`
    /// brings it to 500 (absorbed), not the aligned low 250.
    #[test]
    fn the_start_dose_and_the_floor_of_the_decreases_are_told_apart() {
        let bounds = (250.0, 4000.0);
        assert_eq!(
            absorbed(&stepped(bounds, 1000.0, &[10.0], &[25.0])),
            Some(AbsorbedIncrease::FromStart(1000.0))
        );
        assert_eq!(
            absorbed(&stepped(bounds, 1500.0, &[10.0], &[25.0])),
            Some(AbsorbedIncrease::OnceSteppedDown(500.0))
        );
    }

    /// An unreachable low bound (250, where 25% is absorbed) does not warn, with or
    /// without `decrease 25%` (absorbed at 500).
    #[test]
    fn a_low_bound_no_decrease_reaches_is_not_reported() {
        let bounds = (250.0, 4000.0);
        let grid = DoseGrid::new(250.0, bounds).expect("grid");
        assert_eq!(grid.low.to_bits(), 250.0f64.to_bits());
        assert_eq!(grid.step_percent(250.0, 25.0, true).to_bits(), 250.0f64.to_bits());
        assert_eq!(absorbed(&stepped(bounds, 500.0, &[25.0], &[])), None);
        assert_eq!(absorbed(&stepped(bounds, 500.0, &[25.0], &[25.0])), None);
    }

    /// A dose at the aligned high bound is not reported; one grid point lower it is
    /// (`[250, 600]`, start 300 → 250, `examples/adaptive_vanco_mgkg_rounded.ferx`).
    #[test]
    fn a_dose_at_the_high_bound_is_not_reported() {
        assert_eq!(absorbed(&stepped((250.0, 500.0), 500.0, &[10.0], &[])), None);
        assert_eq!(
            absorbed(&stepped((250.0, 600.0), 300.0, &[25.0], &[])),
            Some(AbsorbedIncrease::FromStart(250.0))
        );
    }

    /// The smallest increase and the largest decrease decide; `increase 0%` is no step
    /// and without `round_to` there is no grid.
    #[test]
    fn the_smallest_increase_and_the_largest_decrease_are_the_ones_read() {
        let bounds = (250.0, 4000.0);
        assert_eq!(
            absorbed(&stepped(bounds, 1000.0, &[50.0, 10.0], &[])),
            Some(AbsorbedIncrease::FromStart(1000.0))
        );
        assert_eq!(absorbed(&stepped(bounds, 1000.0, &[50.0], &[])), None);
        assert_eq!(
            absorbed(&stepped(bounds, 1500.0, &[10.0], &[5.0, 25.0])),
            Some(AbsorbedIncrease::OnceSteppedDown(500.0))
        );
        assert_eq!(absorbed(&stepped(bounds, 1500.0, &[10.0], &[5.0])), None);

        assert_eq!(absorbed(&stepped(bounds, 500.0, &[0.0], &[])), None);
        let mut no_grid = stepped(bounds, 1000.0, &[10.0], &[]);
        no_grid.round_to = None;
        assert_eq!(absorbed(&no_grid), None);
    }

    /// The lowest multiple an increase raises, by hand: at `s = 250`, 25% needs
    /// `d >= 500`, 10% needs `d >= 1250`, 50% raises 250 itself (375 is a tie) and 49%
    /// does not (372.5), so it starts at 500. On a decimal grid, `0.1` under 25%: 0.1
    /// gives 0.125 (back to 0.1), 0.2 gives 0.25, a tie, up to 0.3.
    ///
    /// Small percentages (#2890 review r1 rows 1 and 2): the tie nudge is relative, so
    /// the engine raises a dose below `50 * s / p`. At `s = 1` the reviewer's bisection
    /// gave 16667 for 0.003%, 49996 for 0.001% (not 50000) and 45454546 for 1e-6% (not
    /// 50000000). `1e-310%` must return: `50 / p` is infinite there.
    ///
    /// Each value is also checked against the engine: it is raised and the multiple
    /// below it is not.
    ///
    /// Mutations: returning `50 * s / p` unrounded gives 510.2 for 49%; ties down gives
    /// 750 for 25%; starting the search at `floor(50 / p)` gives 50000 for 0.001% and
    /// does not return for 1e-310%.
    #[test]
    fn lowest_raised_by_is_the_first_multiple_the_step_moves() {
        let open = |s: f64| DoseGrid { step: s, low: 0.0, high: f64::INFINITY };
        for (s, p, want) in [
            (250.0, 25.0, Some(500.0)),
            (250.0, 10.0, Some(1250.0)),
            (250.0, 50.0, Some(250.0)),
            (250.0, 49.0, Some(500.0)),
            (0.1, 25.0, Some(0.2)),
            (1.0, 0.003, Some(16667.0)),
            (1.0, 0.001, Some(49996.0)),
            (1.0, 1e-6, Some(45_454_546.0)),
            (1.0, 1e-310, None),
        ] {
            let got = DoseGrid::lowest_raised_by(s, p).expect("a multiple is raised");
            if let Some(want) = want {
                assert_eq!(got.to_bits(), f64::to_bits(want), "s = {s}, p = {p}");
            }
            let below = got - s;
            assert!(open(s).step_percent(got, p, true) > got, "s = {s}, p = {p}: {got}");
            assert_eq!(
                open(s).step_percent(below, p, true).to_bits(),
                below.to_bits(),
                "s = {s}, p = {p}: {below}"
            );
        }
        // No finite multiple is raised: 1e308 is absorbed (1.25e308 rounds back) and
        // the next multiple is infinite.
        assert_eq!(DoseGrid::lowest_raised_by(1e308, 25.0), None);
    }

    /// P3: a `round_to` that is not finite `> 0` on a spec built in code.
    #[test]
    fn validate_refuses_a_round_to_that_is_not_positive_and_finite() {
        for (s, shown) in [(0.0, "0"), (-250.0, "-250"), (f64::NAN, "NaN"), (f64::INFINITY, "inf")]
        {
            assert_eq!(
                gridded(s, (250.0, 400.0), 250.0).validate().unwrap_err(),
                format!("[adaptive_dosing]: round_to must be a finite number > 0: {shown}")
            );
        }
    }

    /// P4: `round_to` with a valid `levels` ladder, asserted whole.
    #[test]
    fn validate_refuses_round_to_with_levels() {
        let mut s = gridded(250.0, (0.0, 400.0), 100.0);
        s.levels = literal_levels(&[100.0, 200.0]);
        s.rules[0].action = AdaptiveAction::Increase(DoseStep::Level);
        assert_eq!(
            s.validate().unwrap_err(),
            "[adaptive_dosing]: round_to cannot be combined with levels: a levels ladder is \
             already discrete, so write the rounded doses as the levels"
        );
    }

    /// V7 at parse time: no multiple of 250 in [300, 400].
    #[test]
    fn validate_refuses_bounds_holding_no_multiple_of_round_to() {
        assert_eq!(
            gridded(250.0, (300.0, 400.0), 350.0).validate().unwrap_err(),
            "[adaptive_dosing]: no multiple of round_to = 250 lies within dose_bounds [300, 400]"
        );
    }

    /// V8 straddle: `lo = 0` with percentage steps is refused (whole message); `lo = 1e-6`
    /// aligns to 250 and passes; `lo = 0` without a percentage step passes.
    #[test]
    fn validate_refuses_a_lower_bound_that_aligns_to_zero_under_percentage_steps() {
        assert_eq!(
            gridded(250.0, (0.0, 4000.0), 1000.0).validate().unwrap_err(),
            "[adaptive_dosing]: with round_to = 250, dose_bounds low = 0 lets a dose round \
             to 0, and a running dose of 0 cannot rise again under percentage steps; set the \
             lower bound above 0 (any value up to 250 raises it to 250)"
        );
        assert_eq!(gridded(250.0, (1e-6, 4000.0), 1000.0).validate(), Ok(()));
        let mut hold_only = gridded(250.0, (0.0, 4000.0), 1000.0);
        hold_only.rules[0].action = AdaptiveAction::Hold;
        assert_eq!(hold_only.validate(), Ok(()));
    }

    /// V8 refuses only a low of exactly 0: `lo = 1e-6` at `s = 1000` and `1e-7` at
    /// `s = 250` are lifted to one step (#2805).
    #[test]
    fn a_tiny_positive_low_bound_is_lifted_to_one_step_not_refused() {
        for (s, lo) in [(1000.0, 1e-6), (2000.0, 1e-6), (250.0, 1e-7)] {
            assert_eq!(gridded(s, (lo, 4000.0), 2000.0).validate(), Ok(()), "s = {s}, lo = {lo}");
            let g = DoseGrid::new(s, (lo, 4000.0)).expect("grid");
            assert_eq!(g.low.to_bits(), s.to_bits(), "s = {s}, lo = {lo}: {g:?}");
            assert_eq!(g.place(1.0).to_bits(), s.to_bits(), "s = {s}, lo = {lo}");
        }
        let e = gridded(1000.0, (0.0, 4000.0), 2000.0).validate().unwrap_err();
        assert!(e.contains("dose_bounds low = 0 lets a dose round to 0"), "{e}");
        // A positive low with no positive multiple at or below the high is V7, not a
        // grid that clamps every dose to the low.
        let e = gridded(250.0, (1e-7, 100.0), 50.0).validate().unwrap_err();
        assert!(e.contains("no multiple of round_to = 250"), "{e}");
    }

    /// V8 needs only the low bound: a numeric `0` low is refused at parse time even with
    /// an expression high.
    #[test]
    fn validate_refuses_a_zero_low_bound_beside_an_expression_high() {
        let mut s = gridded(250.0, (0.0, 4000.0), 1000.0);
        s.start_dose = DoseValue::Expr("15 * WT".into());
        s.dose_bounds.1 = DoseValue::Expr("min(30 * WT, 4000)".into());
        let e = s.validate().unwrap_err();
        assert!(e.contains("dose_bounds low = 0 lets a dose round to 0"), "{e}");
        s.dose_bounds.0 = DoseValue::Literal(250.0);
        assert_eq!(s.validate(), Ok(()));
    }

    #[test]
    fn validate_accepts_a_well_formed_spec() {
        assert!(base_spec().validate().is_ok());
    }

    #[test]
    fn validate_rejects_programmatic_spec_violations() {
        // The struct is `pub` with `pub` fields, so these never pass through the
        // block parser — `validate` (called by `compile_adaptive`) is the guard.
        let level_rule = || {
            vec![AdaptiveRule {
                op: Comparison::Lt,
                threshold: 10.0,
                action: AdaptiveAction::Increase(DoseStep::Level),
                interval: None,
            }]
        };

        // start_dose outside dose_bounds.
        let mut s = base_spec();
        s.start_dose = (500.0).into();
        assert!(s.validate().unwrap_err().contains("outside dose_bounds"));

        // no rules.
        let mut s = base_spec();
        s.rules.clear();
        assert!(s.validate().unwrap_err().contains("rule"));

        // a Level step with no `levels` ladder (a silent no-op in `step_dose`).
        let mut s = base_spec();
        s.rules = level_rule();
        assert!(s.validate().unwrap_err().contains("levels"));

        // a level rung outside dose_bounds.
        let mut s = base_spec();
        s.dose_bounds = ((0.0).into(), (120.0).into());
        s.levels = crate::sim::adaptive::literal_levels(&[100.0, 150.0]);
        s.rules = level_rule();
        assert!(s.validate().unwrap_err().contains("outside dose_bounds"));
    }

    /// `known_dose_violation` arms no other test reaches (#2667): an infinite upper bound
    /// and an empty ladder.
    #[test]
    fn validate_refuses_an_infinite_high_bound_and_an_empty_ladder() {
        let mut s = base_spec();
        s.dose_bounds = ((0.0).into(), f64::INFINITY.into());
        assert_eq!(
            s.validate().unwrap_err(),
            "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: [0, inf]"
        );
        let mut s = base_spec();
        s.levels = Some(Vec::new());
        s.rules = vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 10.0,
            action: AdaptiveAction::Increase(DoseStep::Level),
            interval: None,
        }];
        assert_eq!(s.validate().unwrap_err(), "[adaptive_dosing]: levels must be a non-empty list");
    }

    /// A numeric block breaking two invariants names the one it named before #944
    /// (#2667): each violation at its pre-#944 position.
    #[test]
    fn validate_reports_two_violations_in_the_pre_944_order() {
        // 1. A descending ladder with a percent rule: the ladder order wins.
        let mut s = base_spec();
        s.start_dose = (200.0).into();
        s.levels = crate::sim::adaptive::literal_levels(&[200.0, 100.0]);
        assert_eq!(
            s.validate().unwrap_err(),
            "[adaptive_dosing]: levels must be strictly increasing"
        );
        // 2. A start off the ladder with a percent rule: the percent refusal wins.
        let mut s = base_spec();
        s.start_dose = (150.0).into();
        s.levels = crate::sim::adaptive::literal_levels(&[100.0, 200.0]);
        assert_eq!(
            s.validate().unwrap_err(),
            "[adaptive_dosing]: percentage actions (`increase N%`) are not allowed with \
             `levels`; use bare `increase`/`decrease` to step one level"
        );
        // 3. Start outside the bounds and off the ladder, with a percent rule: the bounds
        //    check wins.
        let mut s = base_spec();
        s.start_dose = (500.0).into();
        s.levels = crate::sim::adaptive::literal_levels(&[100.0, 200.0]);
        assert_eq!(
            s.validate().unwrap_err(),
            "[adaptive_dosing]: start_dose 500 is outside dose_bounds [0, 400]"
        );
    }

    #[test]
    fn validate_rejects_schedule_and_levels_ordering() {
        // `validate` is the safety net for a hand-built `pub` spec that skipped the parser.

        // empty `at` — would otherwise run a silent no-dose simulation.
        let mut s = base_spec();
        s.at = vec![];
        assert!(s.validate().unwrap_err().contains("non-empty"));

        // unsorted `at` — would permute the decision log.
        let mut s = base_spec();
        s.at = vec![48.0, 24.0];
        assert!(s.validate().unwrap_err().contains("strictly increasing"));

        // negative decision time.
        let mut s = base_spec();
        s.at = vec![-1.0, 24.0];
        assert!(s.validate().unwrap_err().contains(">= 0"));

        // confirm = 0 (the parser requires >= 1).
        let mut s = base_spec();
        s.confirm = 0;
        assert!(s.validate().unwrap_err().contains("confirm"));

        // Descending `levels` would make `increase` lower the dose.
        let mut s = base_spec();
        s.start_dose = (200.0).into();
        s.levels = crate::sim::adaptive::literal_levels(&[200.0, 100.0, 50.0]);
        s.rules = vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 10.0,
            action: AdaptiveAction::Increase(DoseStep::Level),
            interval: None,
        }];
        assert!(s.validate().unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn validate_rejects_inverted_or_nan_target_window() {
        // `target_window`: low finite, low <= high, high may be +inf.

        // high < low (inverted).
        let mut s = base_spec();
        s.target_window = Some((20.0, 10.0));
        assert!(s.validate().unwrap_err().contains("target_window"));

        // non-finite low (NaN and +inf both rejected — low must be finite).
        let mut s = base_spec();
        s.target_window = Some((f64::NAN, 20.0));
        assert!(s.validate().unwrap_err().contains("target_window"));
        let mut s = base_spec();
        s.target_window = Some((f64::INFINITY, 20.0));
        assert!(s.validate().unwrap_err().contains("target_window"));

        // NaN high.
        let mut s = base_spec();
        s.target_window = Some((10.0, f64::NAN));
        assert!(s.validate().unwrap_err().contains("target_window"));

        // Valid bands pass: a closed band and a one-sided "at or above low".
        let mut s = base_spec();
        s.target_window = Some((10.0, 20.0));
        assert!(s.validate().is_ok());
        let mut s = base_spec();
        s.target_window = Some((10.0, f64::INFINITY));
        assert!(s.validate().is_ok());
    }

    #[test]
    fn validate_enforces_one_signal_source() {
        // `observe` and `with_assay_error` are mutually exclusive; exactly one required.
        assert!(base_spec().validate().is_ok());

        // valid Dv: no observe, with_assay_error naming the output.
        let mut s = base_spec();
        s.observe = None;
        s.with_assay_error = true;
        s.assay_cmt = Some(1);
        assert!(s.validate().is_ok());

        // no signal at all.
        let mut s = base_spec();
        s.observe = None;
        assert!(s.validate().unwrap_err().contains("a signal is required"));

        // both an observe expression and assay error.
        let mut s = base_spec();
        s.with_assay_error = true;
        s.assay_cmt = Some(1);
        assert!(s.validate().unwrap_err().contains("remove `observe`"));

        // assay error without naming the measured output.
        let mut s = base_spec();
        s.observe = None;
        s.with_assay_error = true;
        assert!(s.validate().unwrap_err().contains("requires `assay_cmt"));

        // assay_cmt set but no assay error.
        let mut s = base_spec();
        s.assay_cmt = Some(1);
        assert!(s
            .validate()
            .unwrap_err()
            .contains("assay_cmt is set but with_assay_error = false"));
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
            interval: None,
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
            latent: 11.0,
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
            latent: 0.0,
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
            interval: None,
        };
        assert_eq!(entry.clone(), entry);
        assert_eq!(entry.observed_signals[0], obs);
        // The three outcome categories are distinct, and `Stop` carries the count
        // of any dose(s) issued before discontinuation.
        assert_ne!(DecisionOutcome::Hold, DecisionOutcome::Dosed { n: 0 });
        assert_ne!(DecisionOutcome::Stop { dosed: 0 }, DecisionOutcome::Stop { dosed: 1 });
        assert_ne!(DecisionOutcome::Hold, DecisionOutcome::Stop { dosed: 0 });
    }

    // #2581: one test per variant for `label()` and `n_dosed()`; the literals are
    // ferx-r's `decisions$OUTCOME` strings.

    /// Kills: swapping the `Dosed` label with another; `Dosed`'s count returning a
    /// constant (`3` is neither `0` nor `1`).
    #[test]
    fn outcome_dosed_label_and_count() {
        let o = DecisionOutcome::Dosed { n: 3 };
        assert_eq!(o.label(), "dosed");
        assert_eq!(o.n_dosed(), 3);
    }

    /// Kills: swapping the `Hold` label with another; `Hold` reporting a dose.
    #[test]
    fn outcome_hold_label_and_count() {
        let o = DecisionOutcome::Hold;
        assert_eq!(o.label(), "hold");
        assert_eq!(o.n_dosed(), 0);
    }

    /// Kills: swapping the `Stop` label with another; `Stop`'s count made `0` (a
    /// dose-then-stop would read as a bare stop).
    #[test]
    fn outcome_stop_label_and_count() {
        let o = DecisionOutcome::Stop { dosed: 2 };
        assert_eq!(o.label(), "stop");
        assert_eq!(o.n_dosed(), 2);
    }

    /// Kills: swapping the `NotDue` label with another (or respelling it
    /// `"notdue"`); `NotDue` reporting a dose.
    #[test]
    fn outcome_not_due_label_and_count() {
        let o = DecisionOutcome::NotDue;
        assert_eq!(o.label(), "not_due");
        assert_eq!(o.n_dosed(), 0);
    }

    #[test]
    fn validate_rejects_zero_compartment() {
        assert!(DoseAction::Bolus { amt: 100.0, cmt: 0 }.validate().is_err());
        assert!(DoseAction::Infuse { amt: 100.0, cmt: 0, rate: 10.0 }.validate().is_err());
    }

    #[test]
    fn validate_rejects_nonpositive_or_nonfinite_infusion_rate() {
        for rate in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            assert!(
                DoseAction::Infuse { amt: 100.0, cmt: 1, rate }.validate().is_err(),
                "rate {rate} should be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_negative_or_nonfinite_amount() {
        assert!(DoseAction::Bolus { amt: -1.0, cmt: 1 }.validate().is_err());
        assert!(DoseAction::Bolus { amt: f64::NAN, cmt: 1 }.validate().is_err());
    }

    #[test]
    fn validate_accepts_well_formed_actions_and_holds() {
        assert!(DoseAction::Bolus { amt: 100.0, cmt: 1 }.validate().is_ok());
        assert!(DoseAction::Infuse { amt: 100.0, cmt: 2, rate: 25.0 }.validate().is_ok());
        assert!(DoseAction::Hold.validate().is_ok());
        assert!(DoseAction::Stop.validate().is_ok());
    }

    // ----- S1.5: controller-assay substream seed helpers ----------------

    #[test]
    fn stable_hash_str_is_deterministic_and_distinguishes() {
        assert_eq!(stable_hash_str("CONC"), stable_hash_str("CONC"));
        assert_ne!(stable_hash_str("CONC"), stable_hash_str("ANC"));
        // Empty string is well-defined (the FNV-1a offset basis).
        assert_eq!(stable_hash_str(""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn combine_seed_is_order_sensitive() {
        // Folding components in a different order yields a different seed, so key
        // tuples map injectively enough for substream separation.
        assert_ne!(combine_seed(combine_seed(0, 1), 2), combine_seed(combine_seed(0, 2), 1));
    }

    #[test]
    fn subject_assay_base_seed_separates_subjects_and_replicates() {
        let a = subject_assay_base_seed(42, "subj-1", 1);
        assert_eq!(a, subject_assay_base_seed(42, "subj-1", 1), "deterministic");
        assert_ne!(a, subject_assay_base_seed(42, "subj-2", 1), "subject id keys the stream");
        assert_ne!(a, subject_assay_base_seed(42, "subj-1", 2), "replicate keys the stream");
        assert_ne!(a, subject_assay_base_seed(7, "subj-1", 1), "root seed keys the stream");
    }

    #[test]
    fn assay_standard_normal_is_deterministic_and_key_separated() {
        let x = assay_standard_normal(100, 0, "A");
        assert_eq!(x, assay_standard_normal(100, 0, "A"), "deterministic");
        assert_ne!(x, assay_standard_normal(100, 1, "A"), "decision index keys the draw");
        assert_ne!(x, assay_standard_normal(100, 0, "B"), "analyte keys the draw");
        assert_ne!(x, assay_standard_normal(101, 0, "A"), "base seed keys the draw");
    }

    // ----- #701: IOV-kappa substream seed helpers -----------------------

    #[test]
    fn subject_kappa_base_seed_separates_and_is_disjoint_from_assay() {
        let a = subject_kappa_base_seed(42, "subj-1", 1);
        assert_eq!(a, subject_kappa_base_seed(42, "subj-1", 1), "deterministic");
        assert_ne!(a, subject_kappa_base_seed(42, "subj-2", 1), "subject id keys");
        assert_ne!(a, subject_kappa_base_seed(42, "subj-1", 2), "replicate keys");
        assert_ne!(a, subject_kappa_base_seed(7, "subj-1", 1), "root seed keys");
        // Disjoint from the assay stream that shares the same run root: the
        // purpose salt is what keeps enabling a `Dv` monitor from perturbing κ.
        assert_ne!(
            a,
            subject_assay_base_seed(42, "subj-1", 1),
            "kappa and assay substreams are disjoint under the same root"
        );
    }

    /// #946: `replicate_key` is `sim` at draw 1 / rep 1 and distinct across a 4 × 4 × 3
    /// grid and from every plain `sim` key.
    #[test]
    fn replicate_key_is_sim_at_draw_one_rep_one_and_distinct_elsewhere() {
        for sim in 1..=50 {
            assert_eq!(replicate_key(sim, 1, 1), sim);
        }
        let mut seen = std::collections::HashSet::new();
        for draw in 1..=4 {
            for sim in 1..=4 {
                for rep in 1..=3 {
                    let k = replicate_key(sim, draw, rep);
                    assert!(seen.insert(k), "({sim}, {draw}, {rep}) collides");
                    if (draw, rep) != (1, 1) {
                        assert!(k > 1_000_000, "({sim}, {draw}, {rep}) is a plain sim: {k}");
                    }
                }
            }
        }
    }

    #[test]
    fn kappa_standard_normal_is_deterministic_and_key_separated() {
        let x = kappa_standard_normal(100, 0, 0);
        assert_eq!(x, kappa_standard_normal(100, 0, 0), "deterministic");
        assert_ne!(x, kappa_standard_normal(100, 1, 0), "occasion keys the draw");
        assert_ne!(x, kappa_standard_normal(100, 0, 1), "component keys the draw");
        assert_ne!(x, kappa_standard_normal(101, 0, 0), "base seed keys the draw");
    }

    #[test]
    fn assay_standard_normal_spans_both_signs() {
        // The clamp (edge b) is only reachable if draws can be negative; confirm the
        // generator produces both signs across keys, with a mean near 0.
        let n = 2000;
        let draws: Vec<f64> = (0..n).map(|i| assay_standard_normal(1, i, "A")).collect();
        assert!(draws.iter().any(|&d| d < 0.0), "some draws must be negative");
        assert!(draws.iter().any(|&d| d > 0.0), "some draws must be positive");
        let mean = draws.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.1, "standard-normal draws should center near 0, got {mean}");
        // Unit variance too, so the noise scale is right (a mis-scaled generator
        // would still pass the sign/mean checks).
        let var = draws.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (var - 1.0).abs() < 0.2,
            "standard-normal draws should have ~unit variance, got {var}"
        );
    }

    // ── Per-subject metrics (#391 S2.4) ──────────────────────────────────────

    fn ledger_entry(amt: f64) -> DoseLedgerEntry {
        DoseLedgerEntry {
            subject: "S".to_string(),
            draw: 0,
            sim: 0,
            dose_idx: 0,
            time: 0.0,
            amt,
            cmt: 1,
            rate: 0.0,
            decision_idx: 0,
            rule_fired: "bolus".to_string(),
            observed_signals: Vec::new(),
            pre_state: None,
            post_state: None,
            f_applied: 1.0,
        }
    }

    fn decision(time: f64, signal: Option<f64>, outcome: DecisionOutcome) -> DecisionLogEntry {
        DecisionLogEntry {
            subject: "S".to_string(),
            draw: 0,
            sim: 0,
            decision_idx: 0,
            time,
            observed_signals: signal
                .map(|v| {
                    vec![ObservedSignal {
                        name: "signal".to_string(),
                        value: v,
                        latent: v,
                        mode: ObserveMode::Ipred,
                    }]
                })
                .unwrap_or_default(),
            outcome,
            interval: None,
        }
    }

    /// A `Dv` decision whose controller saw `value` while the latent readout was `latent`.
    fn dv_decision(time: f64, value: f64, latent: f64) -> DecisionLogEntry {
        DecisionLogEntry {
            observed_signals: vec![ObservedSignal {
                name: "signal".to_string(),
                value,
                latent,
                mode: ObserveMode::Dv,
            }],
            ..decision(time, None, DecisionOutcome::Dosed { n: 1 })
        }
    }

    #[test]
    fn metrics_are_scored_on_latent_not_on_the_noised_value_946() {
        // The noised values (8, 30, 15) and the latent ones (12, 18, 25) give different
        // answers on every metric, so scoring on `value` again reddens each line.
        let ledger = vec![ledger_entry(100.0)];
        let decisions = vec![
            dv_decision(0.0, 8.0, 12.0),
            dv_decision(24.0, 30.0, 18.0),
            dv_decision(48.0, 15.0, 25.0),
        ];
        let m =
            compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, 20.0)), None, &[]);
        assert_eq!(m.signal_min, Some(12.0), "latent min, not the noised 8");
        assert_eq!(m.signal_max, Some(25.0), "latent max, not the noised 30");
        assert_eq!(m.signal_mean, Some(55.0 / 3.0), "latent mean, not the noised 53/3");
        // Latent 12 and 18 are in [10, 20]; the noised values would put only 15 there.
        assert_eq!(m.pct_readings_in_window, Some(2.0 / 3.0));
    }

    #[test]
    fn metrics_signal_summary_and_window_propagate_nan_946() {
        // Unreachable through the engine (#1578 refuses a non-finite readout), so the
        // signal is hand-built. `f64::min`/`f64::max` would drop the NaN and report the
        // finite readings' 5 and 9 as if nothing happened; the window count would score
        // it as a miss (2/3, since the band [5, 9] holds every finite reading).
        let ledger = vec![ledger_entry(100.0)];
        for nan_at in 0..3 {
            let mut latents = [5.0, 7.0, 9.0];
            latents[nan_at] = f64::NAN;
            let decisions: Vec<_> = latents
                .iter()
                .enumerate()
                .map(|(k, &l)| dv_decision(24.0 * k as f64, 1.0, l))
                .collect();
            let m = compute_subject_metrics(
                "S",
                1,
                1,
                &ledger,
                &decisions,
                Some((5.0, 9.0)),
                None,
                &[],
            );
            assert!(
                m.pct_readings_in_window.is_some_and(f64::is_nan),
                "NaN at {nan_at}: window {m:?}"
            );
            assert!(m.signal_min.is_some_and(f64::is_nan), "NaN at {nan_at}: min {m:?}");
            assert!(m.signal_max.is_some_and(f64::is_nan), "NaN at {nan_at}: max {m:?}");
            assert!(m.signal_mean.is_some_and(f64::is_nan), "NaN at {nan_at}: mean");
        }
    }

    #[test]
    fn metrics_summarize_doses_holds_and_discontinuation() {
        // A titration that steps up once (100→125), back down once (125→100),
        // re-issues unchanged (100→100), holds once, then discontinues.
        let ledger = vec![
            ledger_entry(100.0),
            ledger_entry(125.0),
            ledger_entry(100.0),
            ledger_entry(100.0),
        ];
        let decisions = vec![
            decision(24.0, Some(8.0), DecisionOutcome::Dosed { n: 1 }),
            decision(48.0, Some(5.0), DecisionOutcome::Dosed { n: 1 }),
            decision(72.0, Some(22.0), DecisionOutcome::Hold),
            decision(96.0, Some(15.0), DecisionOutcome::Dosed { n: 1 }),
            decision(120.0, Some(45.0), DecisionOutcome::Stop { dosed: 0 }),
        ];

        let m = compute_subject_metrics("S", 1, 2, &ledger, &decisions, None, None, &[]);

        assert_eq!(m.subject, "S");
        assert_eq!((m.draw, m.sim), (1, 2));
        assert_eq!(m.cumulative_dose, 425.0);
        assert_eq!(m.n_doses, 4);
        assert_eq!(m.n_increases, 1, "100→125 is the only step up");
        assert_eq!(m.n_decreases, 1, "125→100 is the only step down; 100→100 is no change");
        assert_eq!(m.n_holds, 1);
        assert!(m.discontinued);
        assert_eq!(m.time_to_discontinuation, Some(120.0));
        assert_eq!(m.signal_min, Some(5.0));
        assert_eq!(m.signal_max, Some(45.0));
        assert_eq!(m.signal_mean, Some(19.0)); // (8+5+22+15+45)/5
        assert_eq!(m.pct_readings_in_window, None, "no target_window declared");
        assert_eq!(m.auc_target_attainment, None, "no auc_target declared");
    }

    #[test]
    fn metrics_pct_readings_in_window_counts_signals_in_band() {
        let ledger = vec![ledger_entry(100.0)];
        let decisions = vec![
            decision(0.0, Some(8.0), DecisionOutcome::Dosed { n: 1 }),
            decision(24.0, Some(5.0), DecisionOutcome::Dosed { n: 1 }),
            decision(48.0, Some(22.0), DecisionOutcome::Dosed { n: 1 }),
            decision(72.0, Some(15.0), DecisionOutcome::Dosed { n: 1 }),
            decision(96.0, Some(45.0), DecisionOutcome::Dosed { n: 1 }),
        ];
        // Two-sided band [10, 20]: only 15 is in band ⇒ 1/5.
        let band =
            compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, 20.0)), None, &[]);
        assert_eq!(band.pct_readings_in_window, Some(0.2));
        // One-sided "at or above 10" ([10, +∞]): 22, 15, 45 ⇒ 3/5.
        let one_sided = compute_subject_metrics(
            "S",
            1,
            1,
            &ledger,
            &decisions,
            Some((10.0, f64::INFINITY)),
            None,
            &[],
        );
        assert_eq!(one_sided.pct_readings_in_window, Some(0.6));
    }

    #[test]
    fn metrics_reissued_doses_are_not_changes() {
        // A fixed regimen (the degenerate controller): every dose identical ⇒ no
        // increases, no decreases — the metrics half of the degenerate oracle.
        let ledger = vec![ledger_entry(50.0), ledger_entry(50.0), ledger_entry(50.0)];
        let decisions = vec![
            decision(0.0, Some(3.0), DecisionOutcome::Dosed { n: 1 }),
            decision(24.0, Some(3.0), DecisionOutcome::Dosed { n: 1 }),
            decision(48.0, Some(3.0), DecisionOutcome::Dosed { n: 1 }),
        ];
        let m = compute_subject_metrics("S", 1, 1, &ledger, &decisions, None, None, &[]);
        assert_eq!(m.n_increases, 0);
        assert_eq!(m.n_decreases, 0);
        assert_eq!(m.n_holds, 0);
        assert!(!m.discontinued);
        assert_eq!(m.time_to_discontinuation, None);
        assert_eq!(m.cumulative_dose, 150.0);
    }

    #[test]
    fn metrics_empty_run_has_no_signal_summary() {
        // No doses, no decisions: counts are zero and the signal summary is `None`
        // (not a NaN), even when a target_window is set (no signal to score).
        let m =
            compute_subject_metrics("S", 1, 1, &[], &[], Some((10.0, 20.0)), Some((1.0, 2.0)), &[]);
        assert_eq!(m.cumulative_dose, 0.0);
        assert_eq!(m.n_doses, 0);
        assert_eq!(m.n_increases, 0);
        assert_eq!(m.n_decreases, 0);
        assert_eq!(m.n_holds, 0);
        assert!(!m.discontinued);
        assert_eq!(m.time_to_discontinuation, None);
        assert_eq!(m.signal_min, None);
        assert_eq!(m.signal_max, None);
        assert_eq!(m.signal_mean, None);
        assert_eq!(m.pct_readings_in_window, None);
        // An `auc_target` was set, but the driver integrated no windows (no run), so
        // the exposure metric is `None` rather than a 0/0 NaN.
        assert_eq!(m.auc_target_attainment, None);
    }

    #[test]
    fn metrics_signalless_decisions_are_excluded_not_counted_as_misses() {
        // A mixed decision log: the signal summary and `pct_readings_in_window` are over the
        // signal-bearing decisions only.
        let ledger = vec![ledger_entry(100.0), ledger_entry(100.0)];
        let decisions = vec![
            decision(0.0, Some(12.0), DecisionOutcome::Dosed { n: 1 }), // in [10, 20]
            decision(24.0, None, DecisionOutcome::Dosed { n: 1 }),      // no signal
            decision(48.0, Some(8.0), DecisionOutcome::Dosed { n: 1 }), // below band
            decision(72.0, None, DecisionOutcome::Hold),                // no signal
        ];
        let m =
            compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, 20.0)), None, &[]);

        // Summary spans only the two decisions that recorded a signal (12 and 8),
        // not the two signal-less ones.
        assert_eq!(m.signal_min, Some(8.0));
        assert_eq!(m.signal_max, Some(12.0));
        assert_eq!(m.signal_mean, Some(10.0));
        // Denominator = 2 signal-bearing decisions, of which one (12) is in band:
        // 1/2 = 0.5 — not 1/4 (raw decision count) and not 1/3.
        assert_eq!(m.pct_readings_in_window, Some(0.5));
        // Holds are counted over all decisions, independent of whether a signal
        // was recorded, so the signal-less Hold still registers.
        assert_eq!(m.n_holds, 1);
    }

    #[test]
    fn metrics_auc_target_attainment_counts_windows_in_band() {
        // Four windows with AUCs [380, 450, 520, 610] against the vancomycin AUC₂₄ band.
        let ledger = vec![ledger_entry(100.0)];
        let decisions = vec![decision(0.0, Some(10.0), DecisionOutcome::Dosed { n: 1 })];
        let window_aucs = [380.0, 450.0, 520.0, 610.0];

        // Two-sided [400, 600]: 450 and 520 are in band ⇒ 2/4.
        let m = compute_subject_metrics(
            "S",
            1,
            1,
            &ledger,
            &decisions,
            None,
            Some((400.0, 600.0)),
            &window_aucs,
        );
        assert_eq!(m.auc_target_attainment, Some(0.5));
        assert_eq!(m.pct_readings_in_window, None, "no target_window declared");

        // One-sided "at or above 400" ([400, +∞]): 450, 520, 610 ⇒ 3/4.
        let one_sided = compute_subject_metrics(
            "S",
            1,
            1,
            &ledger,
            &decisions,
            None,
            Some((400.0, f64::INFINITY)),
            &window_aucs,
        );
        assert_eq!(one_sided.auc_target_attainment, Some(0.75));

        // A declared band but no windows (single-decision run) ⇒ `None`, not 0/0.
        let no_windows = compute_subject_metrics(
            "S",
            1,
            1,
            &ledger,
            &decisions,
            None,
            Some((400.0, 600.0)),
            &[],
        );
        assert_eq!(no_windows.auc_target_attainment, None);

        // Windows present but no band declared ⇒ unreported (the pass would be
        // skipped upstream; the reduction still yields `None`).
        let no_band =
            compute_subject_metrics("S", 1, 1, &ledger, &decisions, None, None, &window_aucs);
        assert_eq!(no_band.auc_target_attainment, None);
    }

    #[test]
    fn run_has_dose_on_stop_flags_only_dose_carrying_stops() {
        // A declarative `Stop` is dose-free; `[Bolus, Stop]` (`dosed > 0`) is what the
        // call-site `debug_assert!` guards.
        let dose_free_stop = vec![
            decision(0.0, Some(10.0), DecisionOutcome::Dosed { n: 1 }),
            decision(24.0, Some(12.0), DecisionOutcome::Stop { dosed: 0 }),
        ];
        assert!(!run_has_dose_on_stop(&dose_free_stop));

        let dose_then_stop = vec![
            decision(0.0, Some(10.0), DecisionOutcome::Dosed { n: 1 }),
            decision(24.0, Some(12.0), DecisionOutcome::Stop { dosed: 1 }),
        ];
        assert!(run_has_dose_on_stop(&dose_then_stop));

        // No stop at all, and the empty run — both clear.
        let no_stop = vec![decision(0.0, Some(10.0), DecisionOutcome::Dosed { n: 1 })];
        assert!(!run_has_dose_on_stop(&no_stop));
        assert!(!run_has_dose_on_stop(&[]));
    }

    #[test]
    fn validate_rejects_inverted_or_negative_auc_target() {
        // A malformed `auc_target` is rejected by `validate`; `low` must also be `>= 0`.
        let mut s = base_spec();
        s.auc_target = Some((600.0, 400.0)); // inverted
        assert!(s.validate().unwrap_err().contains("auc_target"));

        let mut s = base_spec();
        s.auc_target = Some((-1.0, 600.0)); // negative low
        assert!(s.validate().unwrap_err().contains("auc_target"));

        let mut s = base_spec();
        s.auc_target = Some((f64::NAN, 600.0)); // non-finite low
        assert!(s.validate().unwrap_err().contains("auc_target"));

        let mut s = base_spec();
        s.auc_target = Some((400.0, f64::NAN)); // NaN high
        assert!(s.validate().unwrap_err().contains("auc_target"));

        // Valid: a closed band and a one-sided "at or above low".
        let mut s = base_spec();
        s.auc_target = Some((400.0, 600.0));
        assert!(s.validate().is_ok());
        let mut s = base_spec();
        s.auc_target = Some((400.0, f64::INFINITY));
        assert!(s.validate().is_ok());
    }

    /// `SetInterval` injects nothing and must carry a positive, finite interval (#945).
    #[test]
    fn set_interval_maps_to_no_dose_and_validates_its_interval() {
        assert!(DoseAction::SetInterval { interval: 24.0 }.to_dose_event(0.0).is_none());
        assert!(!DoseAction::SetInterval { interval: 24.0 }.is_stop());
        assert!(DoseAction::SetInterval { interval: 24.0 }.validate().is_ok());
        for bad in [0.0, -12.0, f64::NAN, f64::INFINITY] {
            let err = DoseAction::SetInterval { interval: bad }.validate().unwrap_err();
            assert!(err.contains("positive, finite interval"), "{bad}: {err}");
        }
    }

    /// A not-due point counts in `n_not_due`, not as a hold, and its signal feeds the
    /// summary (#945; `signal_mean` 15, not 10).
    #[test]
    fn metrics_count_not_due_apart_from_holds() {
        let decisions = vec![
            decision(0.0, Some(10.0), DecisionOutcome::Dosed { n: 1 }),
            decision(12.0, Some(20.0), DecisionOutcome::NotDue),
            decision(24.0, Some(10.0), DecisionOutcome::Hold),
            decision(36.0, Some(20.0), DecisionOutcome::NotDue),
        ];
        let m = compute_subject_metrics("S", 1, 1, &[], &decisions, None, None, &[]);
        assert_eq!(m.n_holds, 1);
        assert_eq!(m.n_not_due, 2);
        assert_eq!(m.signal_mean, Some(15.0));
    }

    // ── #945: the dosing-interval ladder's invariants ─────────────────────────────

    fn interval_base() -> AdaptiveDosingSpec {
        let mut s = base_spec();
        s.intervals = Some(vec![12.0, 24.0, 48.0]);
        s.start_interval = Some(24.0);
        s.rules[0].interval = Some(IntervalStep::Lengthen);
        s
    }

    #[test]
    fn validate_accepts_an_interval_ladder() {
        interval_base().validate().expect("well-formed ladder");
        // A ladder with no rule stepping it is a fixed interval, also valid.
        let mut s = interval_base();
        s.rules[0].interval = None;
        s.validate().expect("fixed interval");
    }

    /// Each pairing rule on its own input, its message asserted with the sentence that
    /// tells the user what to write. Deleting any one guard leaves its spec accepted.
    #[test]
    fn validate_pairs_intervals_with_start_interval_and_the_rule_steps() {
        let mut s = interval_base();
        s.start_interval = None;
        let e = s.validate().unwrap_err();
        assert!(
            e.contains("`intervals` requires a `start_interval`")
                && e.contains("in force from the first decision"),
            "{e}"
        );

        let mut s = interval_base();
        s.intervals = None;
        s.rules[0].interval = None;
        let e = s.validate().unwrap_err();
        assert!(
            e.contains("`start_interval` requires an `intervals = [...]` ladder")
                && e.contains("write `intervals = [d]` and `start_interval = d`"),
            "{e}"
        );

        let mut s = interval_base();
        s.intervals = None;
        s.start_interval = None;
        let e = s.validate().unwrap_err();
        assert!(
            e.contains(
                "`lengthen` / `shorten` require an `intervals = [...]` ladder and a \
                 `start_interval`"
            ),
            "{e}"
        );
    }

    /// A hand-built rule whose only effect is `Maintain` does nothing; the parser never
    /// writes one. It is refused and named, and the same rule with an interval effect is
    /// accepted, so the guard is on the missing effect.
    #[test]
    fn validate_rejects_a_rule_with_no_effect() {
        let mut s = interval_base();
        s.rules[0].action = AdaptiveAction::Maintain;
        s.rules[0].interval = None;
        let e = s.validate().unwrap_err();
        assert!(
            e.contains("a rule must have an effect")
                && e.contains("only the dose effect of a rule that also lengthens or shortens")
                && e.ends_with("when signal < 10 :"),
            "{e}"
        );
        s.rules[0].interval = Some(IntervalStep::Shorten);
        s.validate().expect("an interval-only rule is valid");
    }

    #[test]
    fn validate_rejects_a_malformed_interval_ladder() {
        let mut s = interval_base();
        s.intervals = Some(vec![24.0, 12.0]);
        assert!(s.validate().unwrap_err().contains("intervals must be strictly increasing"));

        let mut s = interval_base();
        s.intervals = Some(vec![0.0, 24.0]);
        s.start_interval = Some(24.0);
        let e = s.validate().unwrap_err();
        assert!(e.contains("intervals must be > 0: 0"), "{e}");

        let mut s = interval_base();
        s.start_interval = Some(36.0);
        let e = s.validate().unwrap_err();
        assert!(
            e.contains(
                "start_interval 36 must be one of the declared intervals [12.0, 24.0, 48.0]"
            ),
            "{e}"
        );
    }

    /// `stop` with an interval effect is refused and the rule is named; the same rule
    /// with `hold` instead is accepted, so the guard is on `stop`, not on the effect.
    #[test]
    fn validate_rejects_stop_with_an_interval_effect() {
        let mut s = interval_base();
        s.rules[0].action = AdaptiveAction::Stop;
        let e = s.validate().unwrap_err();
        assert!(
            e.contains(
                "`stop` cannot be combined with `lengthen` / `shorten` (a stopped course has \
                 no next dose)"
            ) && e.contains("when signal < 10 : stop, lengthen"),
            "{e}"
        );
        s.rules[0].action = AdaptiveAction::Hold;
        s.validate().expect("hold, lengthen is valid");
    }
}
