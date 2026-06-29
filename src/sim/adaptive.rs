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

/// A monitor paired with its optional compiled `observe` expression — the
/// driver's per-signal input.
///
/// Pairing the expression *with* its [`MonitorSpec`] (rather than carrying a
/// parallel `observe_exprs` slice the driver indexes by position) makes a
/// monitor/expression desync unrepresentable. `observe == None` ⇒ read the
/// model's `cmt` readout (the programmatic path); `Some(f)` ⇒ the declarative
/// `[adaptive_dosing]` block's compiled signal expression (#391 S2).
pub(crate) struct AdaptiveMonitor<'a> {
    pub spec: &'a MonitorSpec,
    pub observe: Option<&'a crate::ode::OdeOutputFn>,
}

/// What a controller returns at one decision: the dose [`DoseAction`]s to apply,
/// plus optional provenance — the label of the declarative `when` rule that
/// fired, so the dose ledger can record *why* a dose changed. `rule == None` for
/// a programmatic controller or a no-rule re-issue (the driver then records the
/// dose by its route), so the field never fabricates a rule that did not fire.
pub(crate) struct ControllerDecision {
    pub actions: Vec<DoseAction>,
    pub rule: Option<String>,
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

/// Per-subject outcome metrics for one realized adaptive-dosing run (#391 S2.4).
///
/// One row per `(subject, draw, sim)` — the same key as the trajectory, ledger,
/// and decision-log rows in [`crate::AdaptiveSimulationResult`], so a metrics row
/// joins to exactly the run it summarizes. Every field is computed by
/// [`compute_subject_metrics`] from that run's realized dose ledger and decision
/// log **alone** — no re-integration — so each number is a direct, auditable
/// function of the recorded artifacts (the same "reproduce it from the artifacts"
/// contract as the decision log).
///
/// The signal summary (`signal_min`/`max`/`mean`, `pct_time_in_window`) is over the
/// values the controller actually observed at the decision times — the troughs/peaks
/// the monitor saw, which for TDM is the clinically reported quantity. It is a
/// **decision-grid** summary, not a dense-trajectory extremum, and (when several
/// monitors are declared) summarizes the first one. When the run recorded no signal
/// at any decision, the summary fields are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSubjectMetrics {
    /// Subject id (matches [`DoseLedgerEntry::subject`]).
    pub subject: String,
    /// Parameter-uncertainty draw index (matches [`crate::SimulationResult::draw`]).
    pub draw: usize,
    /// Replicate index within the draw (matches [`crate::SimulationResult::sim`]).
    pub sim: usize,
    /// Total nominal dose delivered — Σ `amt` over the run's ledger rows
    /// (pre-bioavailability: the prescribed amount, matching the ledger).
    pub cumulative_dose: f64,
    /// Number of realized doses (ledger rows). Holds and a bare `Stop` leave no
    /// ledger row, so this counts doses actually given, not decisions.
    pub n_doses: usize,
    /// Times the realized dose stepped **up** relative to the previous realized
    /// dose. By dose-delta (not which rule fired), so it counts dose changes that
    /// actually happened — a `decrease` rule clamped at the lower bound re-issues
    /// the same dose and is not counted. `rule_fired` on the ledger remains for
    /// finer audit.
    pub n_increases: usize,
    /// Times the realized dose stepped **down** relative to the previous realized
    /// dose (see [`AdaptiveSubjectMetrics::n_increases`]).
    pub n_decreases: usize,
    /// Decisions that issued no dose ([`DecisionOutcome::Hold`]).
    pub n_holds: usize,
    /// Whether the controller discontinued (a [`DecisionOutcome::Stop`] decision).
    pub discontinued: bool,
    /// Time of the `Stop` decision, or `None` if the run never discontinued.
    pub time_to_discontinuation: Option<f64>,
    /// Lowest observed signal across the decisions that recorded one (the deepest
    /// trough the monitor saw), or `None` if no decision recorded a signal.
    pub signal_min: Option<f64>,
    /// Highest observed signal across the decisions that recorded one.
    pub signal_max: Option<f64>,
    /// Mean observed signal across the decisions that recorded one.
    pub signal_mean: Option<f64>,
    /// Fraction of the **signal-bearing** decisions whose observed value fell
    /// within the spec's `target_window` `[lo, hi]` (inclusive), in `[0, 1]`. The
    /// denominator is the count of decisions that recorded a signal — the same
    /// basis as the signal summary above — not the raw decision count: a decision
    /// with no recorded signal is neither in nor out of band, so it is excluded
    /// rather than counted as a miss. `None` when no `target_window` is declared
    /// (or no signal was recorded) — never a band guessed from the rule
    /// thresholds (#584).
    pub pct_time_in_window: Option<f64>,
}

/// Compute the [`AdaptiveSubjectMetrics`] for one realized run from its dose
/// `ledger` and decision log alone (#391 S2.4).
///
/// `ledger` and `decisions` are the artifacts of a **single** `(subject, draw,
/// sim)` run (the rows the per-subject driver emits), in emission order: `ledger`
/// in dose order and `decisions` in schedule order. The function is pure — it
/// re-integrates nothing — so it is unit-testable on hand-built rows. `target_window`
/// comes from the [`AdaptiveDosingSpec`] (the file-driven path); the programmatic
/// path passes `None`, leaving `pct_time_in_window` unreported.
pub(crate) fn compute_subject_metrics(
    subject: &str,
    draw: usize,
    sim: usize,
    ledger: &[DoseLedgerEntry],
    decisions: &[DecisionLogEntry],
    target_window: Option<(f64, f64)>,
) -> AdaptiveSubjectMetrics {
    let cumulative_dose: f64 = ledger.iter().map(|e| e.amt).sum();
    let n_doses = ledger.len();

    // Dose changes between consecutive *realized* doses. A re-issued dose
    // (unchanged, or clamped to the same bound) is bit-identical; a genuine step
    // differs by the titration increment, so a small relative tolerance separates
    // "no change" from a real step without miscounting float noise in compounded
    // percentage steps.
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

    let n_holds = decisions
        .iter()
        .filter(|d| matches!(d.outcome, DecisionOutcome::Hold))
        .count();
    let stop = decisions
        .iter()
        .find(|d| matches!(d.outcome, DecisionOutcome::Stop { .. }));
    let discontinued = stop.is_some();
    let time_to_discontinuation = stop.map(|d| d.time);

    // Signal summary over the value the controller observed at each decision (the
    // first monitor when several are declared). Decisions with no recorded signal
    // are skipped, so an empty list yields `None` rather than a NaN.
    let signal_vals: Vec<f64> = decisions
        .iter()
        .filter_map(|d| d.observed_signals.first().map(|s| s.value))
        .collect();
    let (signal_min, signal_max, signal_mean) = if signal_vals.is_empty() {
        (None, None, None)
    } else {
        let min = signal_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = signal_vals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean = signal_vals.iter().sum::<f64>() / signal_vals.len() as f64;
        (Some(min), Some(max), Some(mean))
    };
    let pct_time_in_window = match target_window {
        Some((lo, hi)) if !signal_vals.is_empty() => {
            let n_in = signal_vals.iter().filter(|&&v| v >= lo && v <= hi).count();
            Some(n_in as f64 / signal_vals.len() as f64)
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
        discontinued,
        time_to_discontinuation,
        signal_min,
        signal_max,
        signal_mean,
        pct_time_in_window,
    }
}

// ── Controller-assay RNG substream (#391 S1.5) ──────────────────────────────
//
// A DV-mode monitor observes a *realized* measurement = IPRED + ε·√(residual
// variance). Those ε draws live on their **own** per-purpose substream, separate
// from the η draws and the (currently latent) output trajectory, and each draw is
// independently keyed by `(subject, replicate, decision_index, analyte)`. Keying
// by identity rather than draw order makes the assay noise:
//   * **deterministic** — a pure function of the run seed and the key;
//   * **permutation-invariant** — subject iteration order cannot change a
//     subject's draws (Part E);
//   * **non-perturbing** — adding a monitor (a new analyte) or a decision never
//     shifts any other monitor's draws, because no draw consumes a shared stream
//     position.
// The controller-less frozen-replay verifier therefore reproduces the trajectory
// regardless of these draws (it replays realized doses, not decisions).

/// Purpose tag folded into the assay seed so the controller-assay stream is
/// disjoint from any other stream derived from the same run seed (η, output).
const ASSAY_PURPOSE_SALT: u64 = 0xA55A_E155_0DE0_0001;

/// 64-bit golden-ratio odd constant for seed mixing (same family as the
/// per-chain seeding in `estimation/bayes.rs`).
const GOLDEN64: u64 = 0x9E37_79B9_7F4A_7C15;

/// FNV-1a 64-bit hash of a string — a *stable* (cross-platform, cross-run) hash
/// for keying substreams by subject id / analyte name. `DefaultHasher` is
/// deliberately avoided because its output is not guaranteed stable across builds.
pub(crate) fn stable_hash_str(s: &str) -> u64 {
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
pub(crate) fn combine_seed(seed: u64, component: u64) -> u64 {
    let mut z = seed ^ component.wrapping_mul(GOLDEN64);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Per-(subject, replicate) base seed for the controller-assay substream, rooted
/// at the run-level `root` seed. Keyed by the subject *id* (not its loop
/// position), so the stream is permutation-invariant.
pub(crate) fn subject_assay_base_seed(root: u64, subject_id: &str, replicate: usize) -> u64 {
    let s = combine_seed(root ^ ASSAY_PURPOSE_SALT, stable_hash_str(subject_id));
    combine_seed(s, replicate as u64)
}

/// One standard-normal assay draw for monitor `analyte` at decision
/// `decision_index`, on the substream rooted at `base_seed`. A fresh RNG is seeded
/// per draw from the full key, so the draw depends on nothing else in the run
/// (the non-perturbing property).
pub(crate) fn assay_standard_normal(base_seed: u64, decision_index: usize, analyte: &str) -> f64 {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    let key = combine_seed(
        combine_seed(base_seed, decision_index as u64),
        stable_hash_str(analyte),
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(key);
    Normal::new(0.0, 1.0).unwrap().sample(&mut rng)
}

/// Assay-noise capability threaded into
/// [`crate::ode::predictions::ode_predictions_adaptive`] so it can resolve
/// DV-mode monitors (#391 S1.5). Bundles the residual-variance resolver (already
/// folded with the subject's `ruv_scale`) and the per-(subject, replicate) base
/// seed that keys the controller-assay substream.
///
/// `resid_var(cmt, ipred)` returns `Some(variance)` for a compartment that has a
/// residual error model and `None` when none is defined — the driver turns the
/// `None` into a typed error rather than fabricating a σ (S1.5 edge a).
pub(crate) struct AssayNoise<'a> {
    /// `(cmt, ipred) -> Some(residual variance incl. ruv_scale)`, or `None` when
    /// no residual error model covers `cmt`.
    pub resid_var: &'a dyn Fn(usize, f64) -> Option<f64>,
    /// Per-(subject, replicate) base seed; see [`subject_assay_base_seed`].
    pub base_seed: u64,
}

// ── Declarative `[adaptive_dosing]` block (#391 S2) ─────────────────────────
//
// The *parsed* form of the file-driven reactive controller. This is pure syntax
// (an AST): it carries no binding to the model's parameters and no runtime. The
// `observe` expression is kept as source text and compiled against the model's
// parameter context when the controller is built (S2.2) — holding the spec as
// data keeps the parser independent of the integration engine. The parser
// (`parse_adaptive_dosing_block` in `parser::model_parser`) is the single place
// that establishes the field invariants documented below.

/// A parsed `[adaptive_dosing]` block: a declarative, file-driven reactive
/// controller (#391 S2).
///
/// **Invariants** (guaranteed by the block parser, assumed downstream): `at` is
/// non-empty and strictly increasing; `dose_bounds.0 <= dose_bounds.1` with
/// `0 <= dose_bounds.0`; `start_dose` lies within `dose_bounds`; `confirm >= 1`;
/// `rules` is non-empty; `levels`, when present, is non-empty and strictly
/// increasing and contains `start_dose`; and the rule ladder uses percentage
/// dose steps iff `levels` is `None` and one-level steps iff `levels` is `Some`
/// (never mixed). Exactly one signal source: `observe` (a latent expression, with
/// `with_assay_error = false`) **or** `with_assay_error = true` naming a model
/// output via `assay_cmt` — never both.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDosingSpec {
    /// The latent (Ipred) monitored signal — a free-form expression over states +
    /// individual parameters (e.g. `central / V`), compiled against the model in
    /// S2.2. `Some` for an un-noised signal; `None` when `with_assay_error` is set
    /// (the signal is then the noised model output named by `assay_cmt`, not a
    /// re-typed expression). The `when` rules compare the keyword `signal` to it.
    pub observe: Option<String>,
    /// Titrate on the assay-noised measurement of a model output (`true`) rather
    /// than a latent expression (`false`, the default). When set, the signal's
    /// value *and* its σ both come from the output named by `assay_cmt`, so they
    /// can never be on different scales.
    pub with_assay_error: bool,
    /// 1-based compartment of the model output to measure under `with_assay_error`
    /// — its `[scaling]` readout is the signal value and its `[error_model]` σ the
    /// noise. Required iff `with_assay_error`; `None` otherwise.
    pub assay_cmt: Option<usize>,
    /// Decision schedule (subject clock), expanded to explicit times at parse
    /// time — the times the controller is consulted.
    pub at: Vec<f64>,
    /// Dose issued at the first decision; seeds the controller's running dose.
    pub start_dose: f64,
    /// How an emitted dose is delivered.
    pub route: AdaptiveRoute,
    /// Inclusive `(low, high)` clamp applied to every emitted dose.
    pub dose_bounds: (f64, f64),
    /// Debounce: act only after this many *consecutive* matches of the same rule.
    /// `1` acts on the first breach.
    pub confirm: u32,
    /// Discrete titration levels (oncology). When set, `increase`/`decrease` step
    /// one level; mutually exclusive with percentage steps.
    pub levels: Option<Vec<f64>>,
    /// Optional therapeutic target band `[low, high]` (inclusive) for the
    /// monitored signal. It feeds **only** the `pct_time_in_window` outcome metric
    /// (#391 S2.4) — it never influences dosing (the `when` rules do). `high` may
    /// be `+∞` for a one-sided "at or above `low`" target. `None` leaves
    /// `pct_time_in_window` unreported rather than guessing a band from the rule
    /// thresholds (which conflates the control law with the clinical target). See
    /// [`crate::AdaptiveSubjectMetrics`].
    pub target_window: Option<(f64, f64)>,
    /// The first-matching-rule ladder, in file order.
    pub rules: Vec<AdaptiveRule>,
}

/// Validate a `[adaptive_dosing]` float list is non-empty, all-finite, and
/// strictly increasing — the shared contract of the `levels` ladder and an
/// explicit `at` decision list. Called from BOTH the block parser and
/// [`AdaptiveDosingSpec::validate`] so the file-driven and programmatic paths
/// enforce it identically (a single source of truth, not two that can drift).
/// `what` names the list in the error (e.g. `"levels"`, `` "`at` times" ``).
pub(crate) fn validate_increasing_finite(xs: &[f64], what: &str) -> Result<(), String> {
    if xs.is_empty() {
        return Err(format!(
            "[adaptive_dosing]: {what} must be a non-empty list"
        ));
    }
    if !xs.iter().all(|x| x.is_finite()) {
        return Err(format!("[adaptive_dosing]: {what} must all be finite"));
    }
    if xs.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!(
            "[adaptive_dosing]: {what} must be strictly increasing"
        ));
    }
    Ok(())
}

impl AdaptiveDosingSpec {
    /// Enforce every invariant the block parser checks, in one place.
    ///
    /// The struct is `pub` with `pub` fields, so a programmatically-built spec can
    /// reach the controller without going through the parser. The parser calls
    /// this on the spec it assembles, and
    /// [`compile_adaptive`](crate::sim::adaptive_control::compile_adaptive) calls
    /// it again as the safety net for hand-built specs — so neither path can drive
    /// the controller with a contradiction (a `Level` step without a `levels`
    /// ladder, a `start_dose` outside `dose_bounds`, a rung outside `dose_bounds`,
    /// an empty or unsorted `at`/`levels`, a `confirm` of 0, …) the other would
    /// have rejected.
    pub fn validate(&self) -> Result<(), String> {
        if self.rules.is_empty() {
            return Err(
                "[adaptive_dosing]: at least one `when … : …` rule is required".to_string(),
            );
        }
        let (lo, hi) = self.dose_bounds;
        if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi < lo {
            return Err(format!(
                "[adaptive_dosing]: dose_bounds must be finite with 0 <= low <= high: [{lo}, {hi}]"
            ));
        }
        if !self.start_dose.is_finite() || self.start_dose < 0.0 {
            return Err(format!(
                "[adaptive_dosing]: start_dose must be finite and >= 0: {}",
                self.start_dose
            ));
        }
        if self.start_dose < lo || self.start_dose > hi {
            return Err(format!(
                "[adaptive_dosing]: start_dose {} is outside dose_bounds [{lo}, {hi}]",
                self.start_dose
            ));
        }
        // `target_window` (metrics only): low must be finite and low <= high; high
        // may be +∞ (a one-sided "at or above low" target). Checked here too, not
        // just in the parser, so a hand-built spec can't reach metrics with an
        // inverted/NaN band.
        if let Some((wlo, whi)) = self.target_window {
            if !wlo.is_finite() || whi.is_nan() || whi < wlo {
                return Err(format!(
                    "[adaptive_dosing]: target_window must be [low, high] with low finite and \
                     low <= high (high may be inf): [{wlo}, {whi}]"
                ));
            }
        }
        // The decision schedule must be a non-empty, finite, strictly-increasing
        // list of non-negative times. The parser enforces this on the way in;
        // re-check here so a hand-built spec can't silently run with no doses
        // (empty `at`) or a permuted decision log (unsorted `at`).
        validate_increasing_finite(&self.at, "`at` times")?;
        if self.at[0] < 0.0 {
            // Strictly increasing ⇒ `at[0]` is the minimum.
            return Err(format!(
                "[adaptive_dosing]: `at` times must be >= 0: {}",
                self.at[0]
            ));
        }
        if self.confirm < 1 {
            return Err(
                "[adaptive_dosing]: confirm must be >= 1 (1 acts on the first breach)".to_string(),
            );
        }
        // Exactly one signal source. `observe` is the latent (Ipred) signal — a
        // free-form expression with no measurement noise. `with_assay_error = true`
        // makes the signal the *noised measurement* of the model output named by
        // `assay_cmt`: its value and σ both come from that one output, so they can
        // never be on different scales. The two forms are mutually exclusive.
        match (
            self.with_assay_error,
            self.observe.is_some(),
            self.assay_cmt.is_some(),
        ) {
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
                    "[adaptive_dosing]: assay_cmt is set but with_assay_error = false".to_string(),
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

        // Percentage vs one-level dose steps are mutually exclusive and tied to `levels`.
        let has_percent = self.rules.iter().any(|r| {
            matches!(
                r.action,
                AdaptiveAction::Increase(DoseStep::Percent(_))
                    | AdaptiveAction::Decrease(DoseStep::Percent(_))
            )
        });
        let has_level = self.rules.iter().any(|r| {
            matches!(
                r.action,
                AdaptiveAction::Increase(DoseStep::Level)
                    | AdaptiveAction::Decrease(DoseStep::Level)
            )
        });
        match &self.levels {
            Some(l) => {
                // Non-empty, finite, strictly increasing: a `Level` step indexes
                // this ladder and assumes ascending order — on a descending list
                // `increase` would *lower* the dose. The parser checks this on the
                // way in; re-check for hand-built specs.
                validate_increasing_finite(l, "levels")?;
                if has_percent {
                    return Err("[adaptive_dosing]: percentage actions (`increase N%`) are not \
                                allowed with `levels`; use bare `increase`/`decrease` to step one level"
                        .to_string());
                }
                if !l.contains(&self.start_dose) {
                    return Err(format!(
                        "[adaptive_dosing]: start_dose {} must be one of the declared levels {l:?}",
                        self.start_dose
                    ));
                }
                // Every rung must lie within `dose_bounds`; otherwise `step_dose`
                // clamps the emitted dose while the rung index keeps advancing,
                // decoupling the decision log from the realized dose.
                if let Some(bad) = l.iter().find(|&&x| x < lo || x > hi) {
                    return Err(format!(
                        "[adaptive_dosing]: level {bad} is outside dose_bounds [{lo}, {hi}]"
                    ));
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
            }
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
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveRule {
    /// The comparison applied to the observed `signal`.
    pub op: Comparison,
    /// The right-hand side of the comparison.
    pub threshold: f64,
    /// What to do when this rule is the first to match.
    pub action: AdaptiveAction,
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
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveAction {
    /// Raise the dose (by a percentage, or one level if the block declares `levels`).
    Increase(DoseStep),
    /// Lower the dose (by a percentage, or one level if the block declares `levels`).
    Decrease(DoseStep),
    /// Skip this decision — no dose now, the regimen continues.
    Hold,
    /// Discontinue all future dosing.
    Stop,
}

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
    fn label(&self) -> String {
        match self {
            AdaptiveAction::Increase(DoseStep::Percent(p)) => format!("increase {p}%"),
            AdaptiveAction::Increase(DoseStep::Level) => "increase".to_string(),
            AdaptiveAction::Decrease(DoseStep::Percent(p)) => format!("decrease {p}%"),
            AdaptiveAction::Decrease(DoseStep::Level) => "decrease".to_string(),
            AdaptiveAction::Hold => "hold".to_string(),
            AdaptiveAction::Stop => "stop".to_string(),
        }
    }
}

impl AdaptiveRule {
    /// A human-readable label reconstructing the rule as written
    /// (`"signal < 10 : increase 25%"`) — recorded as the dose ledger's
    /// `rule_fired` so an audit can name which rung produced each dose.
    pub(crate) fn label(&self) -> String {
        format!(
            "signal {} {} : {}",
            self.op.symbol(),
            self.threshold,
            self.action.label()
        )
    }
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

    fn base_spec() -> AdaptiveDosingSpec {
        AdaptiveDosingSpec {
            observe: Some("central".to_string()),
            with_assay_error: false,
            assay_cmt: None,
            at: vec![24.0],
            start_dose: 100.0,
            route: AdaptiveRoute::Bolus { cmt: 1 },
            dose_bounds: (0.0, 400.0),
            confirm: 1,
            levels: None,
            target_window: None,
            rules: vec![AdaptiveRule {
                op: Comparison::Lt,
                threshold: 10.0,
                action: AdaptiveAction::Increase(DoseStep::Percent(25.0)),
            }],
        }
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
            }]
        };

        // start_dose outside dose_bounds.
        let mut s = base_spec();
        s.start_dose = 500.0;
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
        s.dose_bounds = (0.0, 120.0);
        s.levels = Some(vec![100.0, 150.0]);
        s.rules = level_rule();
        assert!(s.validate().unwrap_err().contains("outside dose_bounds"));
    }

    #[test]
    fn validate_rejects_schedule_and_levels_ordering() {
        // These single-field invariants are enforced by the block parser; `validate`
        // is the safety net for a hand-built `pub` spec that never saw the parser.
        // Without them a programmatic spec drives a silent wrong result through the
        // public `simulate_adaptive_from_spec`.

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

        // Descending `levels` — the regression that motivated this: a `Level` step
        // on a non-ascending ladder makes `increase` *lower* the dose. `validate`
        // must reject it rather than let the controller silently invert.
        let mut s = base_spec();
        s.start_dose = 200.0;
        s.levels = Some(vec![200.0, 100.0, 50.0]);
        s.rules = vec![AdaptiveRule {
            op: Comparison::Lt,
            threshold: 10.0,
            action: AdaptiveAction::Increase(DoseStep::Level),
        }];
        assert!(s.validate().unwrap_err().contains("strictly increasing"));
    }

    #[test]
    fn validate_rejects_inverted_or_nan_target_window() {
        // `target_window` only feeds the `pct_time_in_window` metric, but a bad band
        // must still be rejected: the parser checks it on the way in, and `validate`
        // is the safety net for a hand-built `pub` spec reaching
        // `simulate_adaptive_from_spec` without ever seeing the parser. Mirrors the
        // parser's check (low finite, low <= high, high may be +inf).

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
        // `observe` (latent) and `with_assay_error` (noised model output) are
        // mutually exclusive; exactly one is required.
        // valid Ipred: observe set, no assay error.
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
        assert_ne!(
            combine_seed(combine_seed(0, 1), 2),
            combine_seed(combine_seed(0, 2), 1)
        );
    }

    #[test]
    fn subject_assay_base_seed_separates_subjects_and_replicates() {
        let a = subject_assay_base_seed(42, "subj-1", 1);
        assert_eq!(a, subject_assay_base_seed(42, "subj-1", 1), "deterministic");
        assert_ne!(
            a,
            subject_assay_base_seed(42, "subj-2", 1),
            "subject id keys the stream"
        );
        assert_ne!(
            a,
            subject_assay_base_seed(42, "subj-1", 2),
            "replicate keys the stream"
        );
        assert_ne!(
            a,
            subject_assay_base_seed(7, "subj-1", 1),
            "root seed keys the stream"
        );
    }

    #[test]
    fn assay_standard_normal_is_deterministic_and_key_separated() {
        let x = assay_standard_normal(100, 0, "A");
        assert_eq!(x, assay_standard_normal(100, 0, "A"), "deterministic");
        assert_ne!(
            x,
            assay_standard_normal(100, 1, "A"),
            "decision index keys the draw"
        );
        assert_ne!(
            x,
            assay_standard_normal(100, 0, "B"),
            "analyte keys the draw"
        );
        assert_ne!(
            x,
            assay_standard_normal(101, 0, "A"),
            "base seed keys the draw"
        );
    }

    #[test]
    fn assay_standard_normal_spans_both_signs() {
        // The clamp (edge b) is only reachable if draws can be negative; confirm the
        // generator produces both signs across keys, with a mean near 0.
        let n = 2000;
        let draws: Vec<f64> = (0..n).map(|i| assay_standard_normal(1, i, "A")).collect();
        assert!(
            draws.iter().any(|&d| d < 0.0),
            "some draws must be negative"
        );
        assert!(
            draws.iter().any(|&d| d > 0.0),
            "some draws must be positive"
        );
        let mean = draws.iter().sum::<f64>() / n as f64;
        assert!(
            mean.abs() < 0.1,
            "standard-normal draws should center near 0, got {mean}"
        );
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
                        mode: ObserveMode::Ipred,
                    }]
                })
                .unwrap_or_default(),
            outcome,
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

        let m = compute_subject_metrics("S", 1, 2, &ledger, &decisions, None);

        assert_eq!(m.subject, "S");
        assert_eq!((m.draw, m.sim), (1, 2));
        assert_eq!(m.cumulative_dose, 425.0);
        assert_eq!(m.n_doses, 4);
        assert_eq!(m.n_increases, 1, "100→125 is the only step up");
        assert_eq!(
            m.n_decreases, 1,
            "125→100 is the only step down; 100→100 is no change"
        );
        assert_eq!(m.n_holds, 1);
        assert!(m.discontinued);
        assert_eq!(m.time_to_discontinuation, Some(120.0));
        assert_eq!(m.signal_min, Some(5.0));
        assert_eq!(m.signal_max, Some(45.0));
        assert_eq!(m.signal_mean, Some(19.0)); // (8+5+22+15+45)/5
        assert_eq!(m.pct_time_in_window, None, "no target_window declared");
    }

    #[test]
    fn metrics_pct_time_in_window_counts_signals_in_band() {
        let ledger = vec![ledger_entry(100.0)];
        let decisions = vec![
            decision(0.0, Some(8.0), DecisionOutcome::Dosed { n: 1 }),
            decision(24.0, Some(5.0), DecisionOutcome::Dosed { n: 1 }),
            decision(48.0, Some(22.0), DecisionOutcome::Dosed { n: 1 }),
            decision(72.0, Some(15.0), DecisionOutcome::Dosed { n: 1 }),
            decision(96.0, Some(45.0), DecisionOutcome::Dosed { n: 1 }),
        ];
        // Two-sided band [10, 20]: only 15 is in band ⇒ 1/5.
        let band = compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, 20.0)));
        assert_eq!(band.pct_time_in_window, Some(0.2));
        // One-sided "at or above 10" ([10, +∞]): 22, 15, 45 ⇒ 3/5.
        let one_sided =
            compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, f64::INFINITY)));
        assert_eq!(one_sided.pct_time_in_window, Some(0.6));
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
        let m = compute_subject_metrics("S", 1, 1, &ledger, &decisions, None);
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
        let m = compute_subject_metrics("S", 1, 1, &[], &[], Some((10.0, 20.0)));
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
        assert_eq!(m.pct_time_in_window, None);
    }

    #[test]
    fn metrics_signalless_decisions_are_excluded_not_counted_as_misses() {
        // A *mixed* decision log — some decisions recorded a signal, some did not.
        // `compute_subject_metrics` is a pure function whose contract admits this
        // (a decision can carry an empty `observed_signals`), even though the
        // current single-monitor engine records one at every decision. The signal
        // summary and `pct_time_in_window` are over the **signal-bearing** decisions
        // only: a measurement-less decision is neither in nor out of band, so it is
        // excluded from the denominator rather than counted as a miss.
        let ledger = vec![ledger_entry(100.0), ledger_entry(100.0)];
        let decisions = vec![
            decision(0.0, Some(12.0), DecisionOutcome::Dosed { n: 1 }), // in [10, 20]
            decision(24.0, None, DecisionOutcome::Dosed { n: 1 }),      // no signal
            decision(48.0, Some(8.0), DecisionOutcome::Dosed { n: 1 }), // below band
            decision(72.0, None, DecisionOutcome::Hold),                // no signal
        ];
        let m = compute_subject_metrics("S", 1, 1, &ledger, &decisions, Some((10.0, 20.0)));

        // Summary spans only the two decisions that recorded a signal (12 and 8),
        // not the two signal-less ones.
        assert_eq!(m.signal_min, Some(8.0));
        assert_eq!(m.signal_max, Some(12.0));
        assert_eq!(m.signal_mean, Some(10.0));
        // Denominator = 2 signal-bearing decisions, of which one (12) is in band:
        // 1/2 = 0.5 — not 1/4 (raw decision count) and not 1/3.
        assert_eq!(m.pct_time_in_window, Some(0.5));
        // Holds are counted over all decisions, independent of whether a signal
        // was recorded, so the signal-less Hold still registers.
        assert_eq!(m.n_holds, 1);
    }
}
