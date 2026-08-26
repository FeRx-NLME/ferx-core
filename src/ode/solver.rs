//! Adaptive ODE integration: the [`Stepper`] abstraction, the explicit Dormand-Prince RK45
//! stepper (the default method), and the drivers every consumer integrates through.
//!
//! RK45 is the same family as Julia's `Tsit5()` — a 5th-order explicit Runge-Kutta method with
//! an embedded 4th-order error estimate for adaptive step control, optimized here for PK ODE
//! systems (2–20 states, smooth dynamics). The stiff [`OdeMethod`] alternatives live in
//! [`super::rosenbrock`].
//!
//! # Shape of the module
//!
//! A **method** implements [`Stepper`]: attempt a step, score it, and evaluate its own
//! continuous extension inside it. A **driver** owns everything else — the step-size
//! controller, the `saveat` contract, the `min_dt` force-accept, the divergence break, soft
//! sampling, event root-finding, statistics — and is written once, generically, over both the
//! stepper and the state scalar `T: PkNum`.
//!
//! That split is what keeps the methods at parity: a consumer written against a driver gets
//! every method, and a method implementing [`Stepper`] gets every consumer. The two live
//! drivers are [`integrate_dense_g`] (saves + in-step reads; behind [`solve_ode`],
//! [`solve_ode_dense`] and [`solve_ode_g`]) and [`solve_ode_until_threshold`] (event-time
//! root-finding). [`make_stepper`] is the single place a method name selects an implementation.

use crate::sens::num::PkNum;

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

/// Consecutive force-accepted minimum-step attempts with a **non-finite** local error
/// before an integration segment is treated as unrecoverably pathological.
///
/// The break is gated on a non-finite error norm (NaN/∞) on purpose: that is the signature
/// of a diverging trajectory whose padded-out predictions the likelihood will reject anyway.
/// A *finite* error above tolerance at `min_dt` is merely a stiff or under-resolved segment —
/// truncating it would freeze-pad the remaining save points with finite (but wrong) values
/// that the likelihood would silently accept, so those are left to run to `max_steps` as
/// before (#603 review #4).
pub(crate) const MAX_CONSECUTIVE_MIN_STEP_CLAMPS: usize = 64;

/// Accepted steps between mid-segment stiffness re-probes under [`OdeMethod::Auto`] (#1080
/// Part C).
///
/// The a-priori probe reads the Jacobian at a segment's *entry* state, which is the one state
/// a binding model looks least stiff at: both factors of a `KON · C · R` term can be zero at
/// dose time and large an hour later. Re-probing along the way is what lets the driver notice,
/// and the interval is the whole cost/latency trade:
///
/// * **Cost.** A probe is `n + 1` right-hand-side evaluations plus an `O(n²)` Gershgorin bound
///   (the eigensolve only runs when that bound cannot rule stiffness out — #1080 Part B), so
///   it is a fraction of a step. Amortized over 25 accepted steps it is ~1% of the segment,
///   which is what a long benign segment pays: it is re-probed, the verdict never changes, and
///   its trajectory is bit-identical to the pinned one. A segment shorter than 25 accepted
///   steps — a dosing interval with a handful of saves, the overwhelming majority — is never
///   re-probed at all and pays nothing.
/// * **Latency.** A segment that has turned stiff takes hundreds to thousands of steps before
///   it exhausts `max_steps`, so 25 steps of delay is noise against the detection it buys.
pub(crate) const AUTO_SWITCH_PROBE_INTERVAL: usize = 25;

/// Stepper changes allowed inside one segment before the driver stops re-probing (#1080
/// Part C).
///
/// A cap rather than a hysteresis band, because the probe is a threshold test on a continuous
/// quantity: a trajectory that hovers at `|Re λ|max ≈` [`STIFF_RE_LAMBDA_THRESHOLD`] would
/// otherwise be able to rebuild its stepper on every probe. Bounding the count keeps the worst
/// case a fixed handful of rebuilds, and a segment that has already changed its mind this many
/// times has no verdict worth acting on.
///
/// [`STIFF_RE_LAMBDA_THRESHOLD`]: crate::ode::stiffness::STIFF_RE_LAMBDA_THRESHOLD
pub(crate) const MAX_AUTO_SWITCHES_PER_SEGMENT: usize = 8;

/// Step-shrink factor applied when the local error estimate is non-finite (NaN/∞). The
/// trajectory is diverging, so shrink toward `min_dt` instead of falling into the
/// I-controller's growth branch, which would otherwise enlarge the step on a NaN error.
const NONFINITE_ERR_SHRINK_FACTOR: f64 = 0.2;

/// ODE right-hand side function type.
/// `rhs(u, params, t) -> du/dt`  where u and du are `&[f64]` of length n_states.
pub type OdeRhsFn = Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>;

/// Which stepper integrates the `[odes]` block (`[fit_options] ode_method`).
///
/// The default is the explicit [`Rk45`](OdeMethod::Rk45); the three Rosenbrock entries are
/// **linearly implicit** and exist for stiff systems — fast binding / TMDD-style
/// quasi-equilibrium, Michaelis-Menten with `Km ≪ C`, long transit chains, QSP cascades —
/// where an explicit method is stability-limited and grinds down to `min_dt` (visible as
/// [`OdeSolverStats::min_step_clamped_steps`]) instead of being accuracy-limited.
///
/// Cost per step differs sharply, so this is not a free upgrade: RK45 takes 6 `f` evaluations
/// (FSAL), while a Rosenbrock step takes `n + 1` extra evaluations for the finite-difference
/// Jacobian and `∂f/∂t` plus an `O(n³)` factorization. On a non-stiff 1–3 compartment model
/// RK45 stays the faster choice; the Rosenbrock methods win only where stiffness, not
/// accuracy, is what caps the step.
///
/// `n` there is the size of the system being integrated, which is **not** always the `[odes]`
/// state count. A CTMM endpoint integrates the occupancy system `dP/dt = P·Q(t)`
/// ([`crate::markov::ctmm_inhomogeneous_transition_with_opts`]), whose state is the flattened
/// `s × s` transition matrix — so an `s`-state chain makes this an `s²`-state integration, and
/// a Rosenbrock method there builds an `s² × s²` Jacobian per step attempt. That is the right
/// behaviour for a chain whose transition rates are widely separated (the occupancy solve is
/// then genuinely stiff, which is why the intensity guard above it already talks about one),
/// but it is pure overhead for the usual well-scaled chain — so choose `ode_method` on a CTMM
/// model with the `s²` cost in mind, not the `[odes]` one.
///
/// Honoured by every `saveat` integration — predictions, the FOCE/FOCEI objective,
/// steady-state equilibration and the analytic-sensitivity walk — dispatched centrally in
/// [`solve_ode_dense`] (the sole owner of the f64 stepping loop) and [`solve_ode_g`].
///
/// Every method is a full peer: each implements [`Stepper`] — including its own continuous
/// extension — and every consumer is written once against that trait. Dense `saveat` saves,
/// in-step soft sampling (joint PK-TTE cumulative hazard, CTMM occupancy, adaptive-dosing
/// monitors), event-time root-finding and the analytic-sensitivity path therefore work for
/// all methods, with no per-method or per-feature wiring, and a method added later inherits
/// the whole set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OdeMethod {
    /// Explicit Dormand-Prince RK45 — see the module docs. The stepper
    /// [`Auto`](OdeMethod::Auto) falls back to, and what it selects on any system that is not
    /// stability-limited; name it to pin it and skip the probe.
    Rk45,
    /// `ode23s`: 3-stage, order 2(3), L-stable. Cheapest stiff option; use at crude
    /// tolerances or when the right-hand side is rough.
    Rosenbrock23,
    /// Rodas4: 6-stage, order 4(3), L-stable and stiffly accurate. The stiff workhorse at
    /// typical PK tolerances.
    Rodas4,
    /// Rodas5P: 8-stage, order 5(4), L-stable and stiffly accurate. Best at tight tolerances
    /// (`ode_reltol ≤ 1e-9`), the regime where an ODE-form OFV has to match the analytical
    /// one.
    Rodas5P,
    /// Verner 7(6): 10-stage explicit, order 7. **Not** a stiff method — the high-order option
    /// for a fit that is accuracy-limited rather than stability-limited, where step count
    /// scales as `tol^(−1/p)` and order is what pays. Worth it at tight tolerances
    /// (`ode_reltol ≤ 1e-8`); at loose ones RK45's cheaper steps win. See
    /// [`crate::ode::explicit_rk`].
    Vern7,
    /// Pick the stepper from the system itself: an a-priori Jacobian eigenvalue probe reads
    /// `|Re λ|max` at each integration's own initial state and starts a stiff method only
    /// where the system is genuinely stability-limited, keeping the explicit default
    /// everywhere else. See [`crate::ode::stiffness`] for what is measured, where, and why an
    /// unclassifiable system stays explicit.
    ///
    /// This is the *starting* decision for each integrated segment. By default,
    /// [`OdeSolverOptions::auto_switch`] re-probes periodically and may replace the active
    /// stepper between accepted steps while retaining the state integrated so far. Disable
    /// that option to keep the starting method fixed for the duration of the segment. Both
    /// modes preserve the guarantees the other variants carry (dense output, event
    /// root-finding, `Dual2` sensitivities).
    ///
    /// **The default.** A model that names no `ode_method` is probed, which is the right
    /// behaviour for the same reason the feature exists: whether a system is stiff is a
    /// property of the equations, and a user who has to know the answer in advance to get a
    /// tractable fit is being asked the wrong question. Every model that is *not* stiff keeps
    /// [`Rk45`](OdeMethod::Rk45) and pays one Jacobian per segment for the privilege; every
    /// model that is stiff stops being stability-limited without anyone having to notice.
    #[default]
    Auto,
}

impl OdeMethod {
    /// The explicit stepper [`Auto`](OdeMethod::Auto) starts from and falls back to.
    ///
    /// Deliberately a separate name from `OdeMethod::default()`, even though the two values
    /// coincided before `auto` became the default: `default()` means "what a user who said
    /// nothing gets" and this means "the method to retreat to when a stiff solve fails", and
    /// they stopped being the same thing the moment the default changed. Spelling the second
    /// as `default()` would make the guard fall back to `auto` — i.e. to the escalation it is
    /// trying to undo.
    pub const EXPLICIT_FALLBACK: OdeMethod = OdeMethod::Rk45;

    /// Parse the `[fit_options] ode_method` token (case-insensitive). Aliases: `ros23` /
    /// `ode23s` for [`Rosenbrock23`](OdeMethod::Rosenbrock23).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rk45" | "dopri5" => Some(OdeMethod::Rk45),
            "rosenbrock23" | "ros23" | "ode23s" => Some(OdeMethod::Rosenbrock23),
            "rodas4" => Some(OdeMethod::Rodas4),
            "rodas5p" | "rodas5" => Some(OdeMethod::Rodas5P),
            "vern7" | "verner7" => Some(OdeMethod::Vern7),
            "auto" => Some(OdeMethod::Auto),
            _ => None,
        }
    }

    /// Canonical token, for banners / round-tripping into a reconstructed model source.
    pub fn as_str(self) -> &'static str {
        match self {
            OdeMethod::Rk45 => "rk45",
            OdeMethod::Rosenbrock23 => "rosenbrock23",
            OdeMethod::Rodas4 => "rodas4",
            OdeMethod::Rodas5P => "rodas5p",
            OdeMethod::Vern7 => "vern7",
            OdeMethod::Auto => "auto",
        }
    }
}

/// ODE solver options
#[derive(Debug, Clone, Copy)]
pub struct OdeSolverOptions {
    pub abstol: f64,
    pub reltol: f64,
    pub max_steps: usize,
    pub initial_dt: f64,
    pub min_dt: f64,
    /// Stepper to use. Default [`OdeMethod::Rk45`]; see [`OdeMethod`] for when a stiff
    /// method pays and which paths honour it.
    pub method: OdeMethod,
    /// Give up on a segment once this many of its steps have clamped at `min_dt`, instead of
    /// grinding on to [`max_steps`](Self::max_steps) (#708, #1080 Part B).
    ///
    /// A clamp is the solver saying "I cannot meet tolerance here and am advancing anyway";
    /// once a segment has produced a handful of them it is stability-limited, and the
    /// remaining thousands of steps buy nothing but wall time. Aborting freeze-pads the
    /// segment's tail exactly as the existing unrecoverable-clamp break does — the result is
    /// no more wrong than the ground-out one, it just costs less — and bumps
    /// [`OdeSolverStats::stiff_aborted_segments`] so the fit's solver-diagnostics warning can
    /// say so.
    ///
    /// `None` (the default) grinds to `max_steps`, i.e. the pre-#1080 behaviour, bit for bit.
    /// This is deliberately **not** on by default: an early abort trades a slow-but-eventually-
    /// integrated segment for a freeze-padded one, which is a change to the objective, not
    /// only to its cost.
    ///
    /// Independent of [`OdeMethod::Auto`]. Under `auto` an escalation that trips the budget is
    /// discarded by the guard exactly as any other stalled escalation would be (the trigger is
    /// [`OdeSolverStats::stiff_min_step_clamped_steps`]` > 0`, which an abort on a stiff
    /// stepper has by construction) and the segment is re-solved explicitly.
    ///
    /// The budget is per *method*, not per segment: when [`auto_switch`](Self::auto_switch)
    /// swaps the stepper part-way, the count starts again, because what it bounds is how long
    /// one method may grind on a segment it cannot step, and the clamps behind the switch
    /// measured the method that was replaced.
    pub stiff_abort_after: Option<u32>,
    /// Let [`OdeMethod::Auto`] change stepper **inside** a segment, not only at its start
    /// (#1080 Part C). Default `true`; ignored under every named method, which is pinned.
    ///
    /// The segment-start probe is evaluated at the entry state, so it cannot see a system that
    /// becomes stiff while being integrated — a depot-absorption binding model has both
    /// factors of `KON · C · R` at zero when the dose lands and reads benign, then runs into a
    /// fast mode an hour later. Measured on that model (`|Re λ|max` 10 at entry, 7.6e3 along
    /// the trajectory), the explicit stepper exhausts `max_steps` and returns a median 143%
    /// relative error, with **zero** min-`dt` clamps and a *lower* step-rejection rate than the
    /// healthy cases — the two runtime counters #1080 proposed as triggers are both blind to
    /// it. Re-probing the Jacobian is not.
    ///
    /// So the trigger is the same discriminator the starting decision uses, re-read every
    /// [`AUTO_SWITCH_PROBE_INTERVAL`] accepted steps, and the switch happens in place: the
    /// segment keeps the state it has integrated so far and carries on with the other stepper,
    /// rather than restarting or re-solving. Setting this to `false` restores the pre-#1080
    /// Part C behaviour — one method per segment, chosen at its start.
    pub auto_switch: bool,
}

impl Default for OdeSolverOptions {
    fn default() -> Self {
        Self {
            abstol: 1e-6,
            reltol: 1e-4,
            max_steps: 10000,
            initial_dt: 0.1,
            min_dt: 1e-12,
            method: OdeMethod::Auto,
            stiff_abort_after: None,
            auto_switch: true,
        }
    }
}

/// Scale-aware tolerance band `abstol + reltol * max(|a|, |b|)`. Shared by the RK45
/// step-error control, the per-step monotonicity guard (`mono_tol`), and the TTE
/// cumulative-hazard monotonicity floor (`survival::MonoTol`), so the integrator and the
/// survival guard use one definition and cannot silently diverge (see #618).
#[inline]
pub(crate) fn scale_tol(abstol: f64, reltol: f64, a: f64, b: f64) -> f64 {
    abstol + reltol * a.abs().max(b.abs())
}

/// Solution point: (time, state vector)
#[derive(Debug, Clone)]
pub struct SolPoint {
    pub t: f64,
    pub u: Vec<f64>,
}

/// Adaptive-step counters for diagnosing whether an integration is
/// rejection-dominated, accepted-small-step-dominated, or hitting the solver's
/// minimum-step escape hatch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OdeSolverStats {
    /// Total RK attempts: `accepted_steps + rejected_steps`.
    pub attempted_steps: usize,
    /// Attempts that advanced `(t, u)`.
    ///
    /// Overlaps [`min_step_clamped_steps`](Self::min_step_clamped_steps) but does not
    /// contain it: an explicit method force-accepts at `min_dt` to guarantee progress, so
    /// those clamps are counted here too, while a linearly implicit method that cannot form
    /// a step at all is counted as a clamp *and* a rejection.
    pub accepted_steps: usize,
    /// Attempts rejected by the local-error test, plus the unusable-`W` abandonments
    /// described on [`min_step_clamped_steps`](Self::min_step_clamped_steps).
    pub rejected_steps: usize,
    /// Attempts the solver could not resolve at `min_dt` — the "I am stability-limited"
    /// signal, and the counter the `ode_method` documentation tells users to read.
    ///
    /// Two shapes reach it, and both must, or the diagnostic reads clean for exactly the
    /// failure it exists to surface:
    ///
    /// * an **explicit** method whose step failed the local-error test (`!(err_norm <= 1)`,
    ///   which also catches a non-finite `err_norm`) yet advanced anyway because
    ///   `dt_eff <= min_dt`; and
    /// * a **linearly implicit** method whose `W = I/(γh) − J` was singular at `min_dt`, so
    ///   no `u_new` existed to force-accept and the driver had to stop. The remaining
    ///   `saveat` points are then freeze-padded with the last state — finite, but wrong.
    pub min_step_clamped_steps: usize,
    /// Of [`min_step_clamped_steps`](Self::min_step_clamped_steps), the ones taken while a
    /// **stiff** stepper was the active one — i.e. the clamps that are the stiff method's own
    /// failure to form a step, rather than the explicit method's (#1080 Part C).
    ///
    /// The two are the same number for a segment that runs on one method throughout, and only
    /// diverge once a segment switches stepper part-way ([`OdeSolverOptions::auto_switch`]).
    /// There they must diverge: the escalation guard discards a stiff attempt *because the
    /// stiff method stalled*, and a segment that clamped on the explicit stepper before the
    /// re-probe escalated it has not shown that at all — throwing its result away would
    /// re-solve it with the very stepper whose clamps triggered the discard.
    pub stiff_min_step_clamped_steps: usize,
    /// Integrations [`OdeMethod::Auto`] escalated to a stiff method because the a-priori probe
    /// read `|Re λ|max ≥` [`crate::ode::stiffness::STIFF_RE_LAMBDA_THRESHOLD`] — at the
    /// segment's entry state, or at a mid-segment re-probe
    /// ([`OdeSolverOptions::auto_switch`], #1080 Part C).
    ///
    /// Both shapes count here, and deliberately so: what the counter reports is that `auto`
    /// put this segment on a stiff stepper, which is what a user reading the fit's solver
    /// warning acts on, and it is what
    /// [`auto_stiff_rejected`](Self::auto_stiff_rejected) is a subset *of*.
    /// [`auto_switched_segments`](Self::auto_switched_segments) separates the "decided later"
    /// half for anyone who needs it.
    ///
    /// Zero under every explicitly-named `ode_method`, so a non-zero value means `auto` was
    /// asked and escalated. It is a floor, not a total: [`solve_ode_until_threshold`] takes no
    /// stats, so escalations on the event-time path (TTE / adaptive dosing) are not counted
    /// here.
    pub auto_stiff_segments: usize,
    /// Integrations whose stepper was changed **in place** part-way through, because a
    /// mid-segment re-probe disagreed with the verdict the segment started on (#1080 Part C).
    ///
    /// One per segment, not per switch — a segment that escalated and later came back down
    /// counts once. A non-zero value says the segment's stiffness changed while it was being
    /// integrated, which is the ordinary shape of a binding model after a dose (the fast mode
    /// `KON · C · R` does not exist until drug and target are both present) rather than a
    /// problem in itself. It matters for reading the other counters: the steps before a switch
    /// were taken by a different method than the ones after it.
    pub auto_switched_segments: usize,
    /// Escalated integrations whose stiff solve was thrown away — it came back non-finite, or
    /// it clamped at `min_dt` and freeze-padded — and were re-solved with the explicit default,
    /// whose result is what the caller receives.
    ///
    /// A subset of [`auto_stiff_segments`](Self::auto_stiff_segments). The stiff attempt's
    /// steps stay counted in the other fields — they were really taken — so a fit that reads
    /// slow *and* shows a non-zero count here is paying for both solves. Non-zero means the
    /// probe was right that the system is stiff and wrong that this stiff method could
    /// integrate it; naming `rodas5p` (or `rosenbrock23`) explicitly is the next thing to try.
    pub auto_stiff_rejected: usize,
    /// Integrations abandoned early because their min-`dt` clamps crossed
    /// [`OdeSolverOptions::stiff_abort_after`] (#708, #1080 Part B).
    ///
    /// Always zero unless that budget is set. A non-zero value means the segment's remaining
    /// `saveat` points were freeze-padded with the last state rather than integrated — the
    /// same tail a ground-out stability-limited segment produces, reached sooner. It is the
    /// counter that distinguishes "this fit was cheap" from "this fit was cheap because it
    /// stopped integrating", so a fit that sets the budget must read it.
    ///
    /// Counts truncated **results**, not truncated attempts: an abort inside an `auto`
    /// escalation that the guard then discarded is not counted, because the caller received
    /// the explicit re-solve instead. A segment whose final step clamps but which still
    /// reaches the end of its span is likewise not counted — nothing was abandoned.
    pub stiff_aborted_segments: usize,
    /// Of [`min_step_clamped_steps`](Self::min_step_clamped_steps), the ones taken inside an
    /// `auto` escalation the guard then **discarded** — work that was really done, on a
    /// trajectory nobody received.
    ///
    /// Exists because the two halves mean different things to a user. A clamp in a *kept*
    /// solve says part of the returned trajectory was freeze-padded rather than integrated; a
    /// clamp in a discarded escalation says only that the escalation failed, which
    /// [`auto_stiff_rejected`](Self::auto_stiff_rejected) already reports and which the
    /// explicit re-solve then repaired. Since a stall *is* the rejection trigger, every
    /// rejected escalation contributes clamps here, and a diagnostic that did not separate
    /// them would tell every such fit its predictions were freeze-padded when they were not.
    pub discarded_clamped_steps: usize,
}

impl OdeSolverStats {
    /// Record one RK attempt. `accepted` is the stepper's accept decision
    /// (`err_norm <= 1.0 || dt_eff <= min_dt`); the clamp test mirrors it with
    /// `!(err_norm <= 1.0)` so a non-finite `err_norm` (a diverging RHS pinned
    /// at `min_dt`) still counts as a min-step clamp rather than a clean accept.
    #[inline]
    pub(crate) fn record(&mut self, accepted: bool, err_norm: f64, dt_eff: f64, min_dt: f64) {
        self.attempted_steps += 1;
        if accepted {
            self.accepted_steps += 1;
            if !(err_norm <= 1.0) && dt_eff <= min_dt {
                self.min_step_clamped_steps += 1;
            }
        } else {
            self.rejected_steps += 1;
        }
    }

    /// Accumulate another integration's counters into these, for a caller that splits one
    /// logical solve across more than one driver call (the `auto` finite-guard's escalated
    /// attempt and its explicit re-solve).
    #[inline]
    pub(crate) fn merge(&mut self, other: &OdeSolverStats) {
        let OdeSolverStats {
            attempted_steps,
            accepted_steps,
            rejected_steps,
            min_step_clamped_steps,
            stiff_min_step_clamped_steps,
            auto_stiff_segments,
            auto_switched_segments,
            auto_stiff_rejected,
            stiff_aborted_segments,
            discarded_clamped_steps,
        } = *other;
        self.attempted_steps += attempted_steps;
        self.accepted_steps += accepted_steps;
        self.rejected_steps += rejected_steps;
        self.min_step_clamped_steps += min_step_clamped_steps;
        self.stiff_min_step_clamped_steps += stiff_min_step_clamped_steps;
        self.auto_stiff_segments += auto_stiff_segments;
        self.auto_switched_segments += auto_switched_segments;
        self.auto_stiff_rejected += auto_stiff_rejected;
        self.stiff_aborted_segments += stiff_aborted_segments;
        self.discarded_clamped_steps += discarded_clamped_steps;
    }

    /// Record an attempt that produced no usable step at `min_dt` (a singular Rosenbrock
    /// `W`), after which the driver stops and freeze-pads the tail.
    ///
    /// This is a rejection — nothing advanced — but it is *also* a min-step clamp, and
    /// counting it only as the former is what would let a stalled-and-frozen integration
    /// present the same clean stats block as a healthy one.
    #[inline]
    pub(crate) fn record_min_step_failure(&mut self) {
        self.attempted_steps += 1;
        self.rejected_steps += 1;
        self.min_step_clamped_steps += 1;
    }
}

/// The one thing a method has to provide; everything else about integrating is shared.
///
/// A stepper owns its stage workspace and knows how to (a) attempt a step and score it, and
/// (b) evaluate its own **continuous extension** inside that step. The drivers below own the
/// step-size controller, the `saveat` contract, the `min_dt` force-accept, the divergence
/// break, soft sampling, event root-finding and the statistics — none of which is written per
/// method. A new consumer written against this trait therefore works with every method at
/// once, and a new method implementing it works with every consumer at once.
///
/// **Every comparison a driver makes reads [`PkNum::val`] only**, so at `T = Dual2` the
/// derivative rides a step sequence fixed by the value part — the property the analytic
/// FOCE/FOCEI gradient depends on.
pub(crate) trait Stepper<T: PkNum> {
    /// Attempt a step of size `dt` from `(t, u)` and return the value-only RMS error norm
    /// (`≤ 1` means "meets tolerance"), or `f64::INFINITY` when the attempt could not be
    /// scored at all. On return [`u_new`](Stepper::u_new) and
    /// [`interpolate_component`](Stepper::interpolate_component) describe this attempt.
    fn attempt(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64;

    /// Proposed state at `t + dt` for the last attempt.
    fn u_new(&self) -> &[T];

    /// Continuous extension of the last attempt: component `i` of the state at `t + θ·dt`,
    /// for `θ ∈ [0, 1]`. Must return `u_old[i]` at `θ = 0` and `u_new()[i]` at `θ = 1`, so
    /// interpolated reads agree with saved states at every step boundary.
    ///
    /// Reads only data the attempt already committed, so calling it cannot change the step
    /// sequence — a soft-sampling caller does not perturb the trajectory it samples.
    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T;

    /// Hook fired once the driver commits a step (RK45 carries `k7` into the next `k1`).
    fn on_accept(&mut self);

    /// I-controller exponent `1/(p̂+1)` for this method's embedded pair.
    fn err_exp(&self) -> f64;

    /// Tell the stepper whether a caller will read *inside* steps, before integration starts.
    ///
    /// Methods whose end-of-step derivative is a by-product (FSAL RK45) or whose continuous
    /// extension is built from stage increments (Rosenbrock) ignore this — their interpolant is
    /// free. A method that must pay an extra evaluation to interpolate (Verner 7(6)) uses it to
    /// skip that cost on the `saveat`-only path, which is nearly every call.
    fn set_dense_required(&mut self, _yes: bool) {}

    /// Whether the last attempt produced a usable [`u_new`](Stepper::u_new). False only when
    /// the step could not be formed at all (a singular Rosenbrock `W`), where even the
    /// `min_dt` force-accept must not fire because it would commit garbage.
    fn attempt_usable(&self) -> bool;
}

/// Build the stepper for `method`, sized for an `n`-state system.
///
/// This is the **only** place a method name selects an implementation. Consumers take a
/// `&mut dyn Stepper<T>`; the dynamic call happens once per step attempt, against `n + 1`
/// right-hand-side evaluations and an `O(n³)` factorization, so it does not register.
/// Every arm is spelled out rather than ending in a catch-all: adding an [`OdeMethod`] should
/// fail to compile here (and in the three sibling matches in [`super::rosenbrock`]) rather
/// than route silently into the Rosenbrock stepper and panic inside a fit's worker thread.
pub(crate) fn make_stepper<T: PkNum>(n: usize, method: OdeMethod) -> Box<dyn Stepper<T>> {
    match method {
        OdeMethod::Rk45 => Box::new(Rk45Stepper::new(n)),
        OdeMethod::Vern7 => Box::new(super::explicit_rk::ErkStepper::new(
            n,
            &super::explicit_rk::VERN7,
        )),
        stiff @ (OdeMethod::Rosenbrock23 | OdeMethod::Rodas4 | OdeMethod::Rodas5P) => {
            Box::new(super::rosenbrock::RosStepper::new(n, stiff))
        }
        // `Auto` names a *decision*, not a stepper: every driver resolves it through
        // `stiffness::resolve_method` before it gets here, so this arm is only reachable
        // from a caller that built a stepper straight from the options. Falling through to
        // the explicit default keeps such a caller integrating (correctly, if slowly on a
        // stiff system) rather than panicking inside a fit's worker thread.
        OdeMethod::Auto => Box::new(Rk45Stepper::new(n)),
    }
}

/// Cubic Hermite interpolation across one accepted step, generic over the state scalar.
/// `s ∈ [0, 1]` is the normalized position, `h` the step length, `(y0, d0)` / `(y1, d1)` the
/// value and derivative at the step's start and end — the FSAL `k1` / `k7` slots.
#[inline]
pub(crate) fn hermite_g<T: PkNum>(s: f64, h: f64, y0: T, d0: T, y1: T, d1: T) -> T {
    let s2 = s * s;
    let s3 = s2 * s;
    y0 * T::from_f64(2.0 * s3 - 3.0 * s2 + 1.0)
        + d0 * T::from_f64((s3 - 2.0 * s2 + s) * h)
        + y1 * T::from_f64(-2.0 * s3 + 3.0 * s2)
        + d1 * T::from_f64((s3 - s2) * h)
}

/// Explicit Dormand-Prince RK45 (the default method) as a [`Stepper`].
///
/// FSAL (First Same As Last): `k7` of an accepted step is evaluated at the same `(u, t)` the
/// next step's `k1` would use, so [`on_accept`](Stepper::on_accept) swaps them and saves one
/// right-hand-side evaluation per accepted step (~1 of 7 stages, ≈9% of FOCEI ODE wall time).
/// After a *rejected* step `(u, t)` has not moved, so `k1` stays valid too; the first attempt
/// has no prior `k1`, hence `have_k1`.
struct Rk45Stepper<T> {
    n: usize,
    k1: Vec<T>,
    k2: Vec<T>,
    k3: Vec<T>,
    k4: Vec<T>,
    k5: Vec<T>,
    k6: Vec<T>,
    k7: Vec<T>,
    u_tmp: Vec<T>,
    u5: Vec<T>,
    have_k1: bool,
}

impl<T: PkNum> Rk45Stepper<T> {
    fn new(n: usize) -> Self {
        let z = T::from_f64(0.0);
        Self {
            n,
            k1: vec![z; n],
            k2: vec![z; n],
            k3: vec![z; n],
            k4: vec![z; n],
            k5: vec![z; n],
            k6: vec![z; n],
            k7: vec![z; n],
            u_tmp: vec![z; n],
            u5: vec![z; n],
            have_k1: false,
        }
    }
}

impl<T: PkNum> Stepper<T> for Rk45Stepper<T> {
    // The stage loops walk `u` alongside seven stage vectors at once; zipping them would
    // obscure the Butcher tableau they transcribe — and the exact association of these
    // expressions is what keeps `T = f64` bit-identical to the pre-refactor loop.
    #[allow(clippy::needless_range_loop)]
    fn attempt(
        &mut self,
        rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
        u: &[T],
        params: &[T],
        t: f64,
        dt_eff: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        let n = self.n;
        // The Butcher combinations keep the exact association of the original f64 loop
        // (`dt·(B31·k1 + B32·k2)`, not `(dt·B31)·k1 + …`), so instantiating at `T = f64`
        // reproduces the historical trajectory bit for bit.
        //
        // The *generic* path is a deliberate last-bit change: RK45 used to exist as two
        // transcriptions that disagreed here, the f64 loop associating as above and the
        // generic one accumulating `((u + k1·(dt·B31)) + k2·(dt·B32))` term by term. Only
        // one of the two could survive being written once, and the predictor's is the one
        // worth keeping — the analytic sensitivities are now differentiating exactly the
        // trajectory `predict()` reports rather than one a few ULP away from it.
        if !self.have_k1 {
            rhs(u, params, t, &mut self.k1);
            self.have_k1 = true;
        }
        let h = T::from_f64(dt_eff);

        for i in 0..n {
            self.u_tmp[i] = u[i] + T::from_f64(dt_eff * B21) * self.k1[i];
        }
        rhs(&self.u_tmp, params, t + A2 * dt_eff, &mut self.k2);

        for i in 0..n {
            self.u_tmp[i] =
                u[i] + h * (T::from_f64(B31) * self.k1[i] + T::from_f64(B32) * self.k2[i]);
        }
        rhs(&self.u_tmp, params, t + A3 * dt_eff, &mut self.k3);

        for i in 0..n {
            self.u_tmp[i] = u[i]
                + h * (T::from_f64(B41) * self.k1[i]
                    + T::from_f64(B42) * self.k2[i]
                    + T::from_f64(B43) * self.k3[i]);
        }
        rhs(&self.u_tmp, params, t + A4 * dt_eff, &mut self.k4);

        for i in 0..n {
            self.u_tmp[i] = u[i]
                + h * (T::from_f64(B51) * self.k1[i]
                    + T::from_f64(B52) * self.k2[i]
                    + T::from_f64(B53) * self.k3[i]
                    + T::from_f64(B54) * self.k4[i]);
        }
        rhs(&self.u_tmp, params, t + A5 * dt_eff, &mut self.k5);

        for i in 0..n {
            self.u_tmp[i] = u[i]
                + h * (T::from_f64(B61) * self.k1[i]
                    + T::from_f64(B62) * self.k2[i]
                    + T::from_f64(B63) * self.k3[i]
                    + T::from_f64(B64) * self.k4[i]
                    + T::from_f64(B65) * self.k5[i]);
        }
        rhs(&self.u_tmp, params, t + dt_eff, &mut self.k6);

        // 5th-order solution.
        for i in 0..n {
            self.u5[i] = u[i]
                + h * (T::from_f64(B71) * self.k1[i]
                    + T::from_f64(B73) * self.k3[i]
                    + T::from_f64(B74) * self.k4[i]
                    + T::from_f64(B75) * self.k5[i]
                    + T::from_f64(B76) * self.k6[i]);
        }

        // Error estimate (5th − 4th). Computed on values only — the step sequence must not
        // depend on derivative components.
        rhs(&self.u5, params, t + dt_eff, &mut self.k7);
        let mut err_norm = 0.0;
        for i in 0..n {
            let err_i = dt_eff
                * (E1 * self.k1[i].val()
                    + E3 * self.k3[i].val()
                    + E4 * self.k4[i].val()
                    + E5 * self.k5[i].val()
                    + E6 * self.k6[i].val()
                    + E7 * self.k7[i].val());
            let scale = scale_tol(opts.abstol, opts.reltol, self.u5[i].val(), u[i].val());
            err_norm += (err_i / scale) * (err_i / scale);
        }
        (err_norm / n as f64).sqrt()
    }

    fn u_new(&self) -> &[T] {
        &self.u5
    }

    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T {
        hermite_g(theta, dt, u_old[i], self.k1[i], self.u5[i], self.k7[i])
    }

    fn on_accept(&mut self) {
        // `k7` (at the accepted end point) becomes the next step's `k1`; it is dead otherwise.
        std::mem::swap(&mut self.k1, &mut self.k7);
    }

    fn err_exp(&self) -> f64 {
        // 0.20 — I-controller exponent for the 5(4) pair.
        1.0 / 5.0
    }

    fn attempt_usable(&self) -> bool {
        true
    }
}

/// The shared adaptive driver: integrate `t_span`, saving the state at every `saveat` time and
/// the **interpolated** state at every `interp_at` time, with whatever [`Stepper`] is handed in.
///
/// `saveat` times clamp the step so the solver lands on them exactly; `interp_at` times are read
/// *inside* the accepted step that spans them, through the method's continuous extension, so
/// requesting them cannot change the step sequence or the `saveat` values. Both grids must be
/// sorted ascending; unreached times are filled with the final state, keeping
/// `hard.len() == saveat.len()` and `soft.len() == interp_at.len()`.
///
/// NOTE: a Gustafsson PI step-size controller was tested and rejected here. While it lowers
/// the raw step-rejection rate and integrates faster, the factor's dependence on `err_{n-1}`
/// makes accept/reject decisions more sensitive to small parameter perturbations. That raises
/// the differential noise floor of the trajectory as a function of θ, which the FOCEI FD
/// gradient cannot tolerate — BFGS line search stalled at OFV ≈ -1290 on the dense-Emax PKPD
/// benchmark vs the true -1747 with the I-controller. The pure I-controller below is memoryless
/// and gives a clean FD signal. Any future revisit should condition PI on a non-FD gradient
/// route (analytical / analytic-sensitivity).
///
/// The driver owns the stepper rather than receiving one, because under
/// [`OdeMethod::Auto`] it may **replace** it part-way through: a mid-segment re-probe that
/// disagrees with the verdict the segment started on swaps the stepper in place and carries on
/// from the state already integrated (#1080 Part C). `method` is the starting choice — the
/// segment-start probe's verdict, or the named method, which is never switched away from.
#[allow(clippy::too_many_arguments)]
fn integrate_dense_g<T: PkNum>(
    method: OdeMethod,
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    mut stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    let n = u0.len();
    let (t0, tf) = t_span;

    if (tf - t0).abs() < 1e-15 {
        let at = |times: &[f64]| -> Vec<SolPointG<T>> {
            times
                .iter()
                .map(|&t| SolPointG { t, u: u0.to_vec() })
                .collect()
        };
        return (at(saveat), at(interp_at));
    }

    let mut u = u0.to_vec();
    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);

    let mut results: Vec<SolPointG<T>> = Vec::with_capacity(saveat.len());
    let mut save_idx = 0;
    let mut interp_results: Vec<SolPointG<T>> = Vec::with_capacity(interp_at.len());
    let mut interp_idx = 0;
    let mut consecutive_min_step_clamps = 0usize;
    // Clamps produced by the *currently active* stepper. Counted locally rather than read back
    // off `stats`, because `stats` is optional and the abort budget must behave identically
    // whether or not the caller asked for counters — and reset when the stepper is swapped
    // (below), because the budget bounds how long one method is allowed to grind on a segment
    // it cannot step. Clamps the explicit stepper took before a re-probe escalated the segment
    // are that method's stability limit, not the stiff method's, and charging them to the
    // stiff phase would abort it for a limit belonging to the stepper it just replaced.
    let mut min_step_clamps = 0u32;
    // Clamps taken while a stiff stepper was active, for `stiff_min_step_clamped_steps`. Not
    // reset on a switch: it is a per-segment total, and the escalation guard needs to know
    // whether *any* stiff phase of this segment stalled.
    let mut stiff_min_step_clamps = 0usize;
    let mut method = method;
    let mut stepper = make_stepper::<T>(n, method);
    stepper.set_dense_required(!interp_at.is_empty());
    let mut err_exp = stepper.err_exp();
    // Mid-segment re-probing is `auto`'s business only: a named method is pinned, failed
    // result included (the same rule the escalation guard follows). The explicit re-solve the
    // guard runs pins its method for exactly this reason, so it cannot switch back.
    let switching = opts.method == OdeMethod::Auto && opts.auto_switch;
    let mut accepted_since_probe = 0usize;
    let mut switches = 0usize;
    // A segment that starts on a stiff method was counted as an escalation by the caller (the
    // guard needs that count before the segment runs), so a later switch must not count it
    // twice — nor may a segment that switches down and back up.
    let mut escalation_counted = method != OdeMethod::EXPLICIT_FALLBACK;

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            break;
        }

        // Don't overshoot tf or the next saveat.
        let mut dt_eff = dt.min(tf - t);
        if save_idx < saveat.len() && t + dt_eff > saveat[save_idx] + 1e-15 {
            dt_eff = (saveat[save_idx] - t).max(opts.min_dt);
        }

        let err_norm = stepper.attempt(rhs, &u, params, t, dt_eff, opts);
        let usable = stepper.attempt_usable();

        // An unusable attempt at `min_dt` is unrecoverable: the step cannot be shrunk further
        // and there is no `u_new` to force-accept. Stop and let the tail freeze-pad, the same
        // outcome a trajectory that diverges at `min_dt` gets.
        //
        // Recorded as a min-step *clamp*, not a plain rejection. The freeze-padded tail below
        // is finite and plausible, so the stats block is the only place this failure is
        // visible at all — and `min_step_clamped_steps` is precisely the counter the
        // `ode_method` docs send users to when a fit looks stability-limited.
        if !usable && dt_eff <= opts.min_dt {
            if let Some(s) = stats.as_deref_mut() {
                s.record_min_step_failure();
            }
            if method != OdeMethod::EXPLICIT_FALLBACK {
                stiff_min_step_clamps += 1;
            }
            break;
        }

        let accepted = usable && (err_norm <= 1.0 || dt_eff <= opts.min_dt);
        // Mirrors `OdeSolverStats::record`'s clamp test, so the abort budget below counts
        // exactly the steps `min_step_clamped_steps` reports.
        if accepted && !(err_norm <= 1.0) && dt_eff <= opts.min_dt {
            min_step_clamps += 1;
            if method != OdeMethod::EXPLICIT_FALLBACK {
                stiff_min_step_clamps += 1;
            }
        }
        // Only a force-accept at `min_dt` with a *non-finite* error counts toward the
        // pathological-divergence break (#603 review #4); a finite-but-stiff clamp is left to
        // run rather than freeze-padded into a silently-accepted wrong trajectory.
        let nonfinite_min_step = accepted && dt_eff <= opts.min_dt && !err_norm.is_finite();
        if let Some(s) = stats.as_deref_mut() {
            s.record(accepted, err_norm, dt_eff, opts.min_dt);
        }

        if accepted {
            // Soft sampling: read every `interp_at` time in this just-accepted step's
            // half-open span `(t, t+dt_eff]` off the method's continuous extension, *before*
            // `u` advances and the stepper recycles its stages. This only reads committed
            // step data, so it cannot affect the step sequence.
            while interp_idx < interp_at.len() && interp_at[interp_idx] <= t + dt_eff + 1e-12 {
                let ti = interp_at[interp_idx];
                let theta = ((ti - t) / dt_eff).clamp(0.0, 1.0);
                let ui: Vec<T> = (0..n)
                    .map(|j| stepper.interpolate_component(theta, &u, dt_eff, j))
                    .collect();
                interp_results.push(SolPointG { t: ti, u: ui });
                interp_idx += 1;
            }

            t += dt_eff;
            u.copy_from_slice(stepper.u_new());
            stepper.on_accept();

            while save_idx < saveat.len() && (t - saveat[save_idx]).abs() < 1e-12 {
                results.push(SolPointG {
                    t: saveat[save_idx],
                    u: u.clone(),
                });
                save_idx += 1;
            }

            if nonfinite_min_step {
                consecutive_min_step_clamps += 1;
                if consecutive_min_step_clamps >= MAX_CONSECUTIVE_MIN_STEP_CLAMPS {
                    break;
                }
            } else {
                consecutive_min_step_clamps = 0;
            }

            // Mid-segment stiffness re-probe (#1080 Part C). Read the same discriminator the
            // starting decision used, at the state the segment has actually reached, and swap
            // the stepper in place when it disagrees. Done here — after the step is committed
            // and its interpolated reads are taken — so the switch cannot disturb either the
            // trajectory behind it or the values already saved off it: the new stepper starts
            // from `(t, u)` exactly as a fresh segment would, and only the steps *after* this
            // point are taken by it.
            //
            // Reads `PkNum::val` only, like every other comparison a driver makes, so the
            // `T = f64` prediction and the `T = Dual2` sensitivity solve switch at the same
            // step and the analytic gradient keeps differentiating the reported trajectory.
            accepted_since_probe += 1;
            if switching
                && switches < MAX_AUTO_SWITCHES_PER_SEGMENT
                && accepted_since_probe >= AUTO_SWITCH_PROBE_INTERVAL
            {
                accepted_since_probe = 0;
                let verdict = super::stiffness::resolve_method(rhs, &u, params, t, opts);
                if verdict != method {
                    method = verdict;
                    stepper = make_stepper::<T>(n, method);
                    stepper.set_dense_required(!interp_at.is_empty());
                    err_exp = stepper.err_exp();
                    // Both counters are per *segment*, however many times it changes its mind.
                    // An escalation is an escalation whether the probe reached that verdict at
                    // the entry state or twenty-five steps in, so a switch onto a stiff method
                    // joins `auto_stiff_segments` too — which is what keeps the guard's "N of M
                    // escalations were discarded" count adding up. A segment that started stiff
                    // was already counted by the caller.
                    let newly_escalated =
                        !escalation_counted && method != OdeMethod::EXPLICIT_FALLBACK;
                    if let Some(s) = stats.as_deref_mut() {
                        if switches == 0 {
                            s.auto_switched_segments += 1;
                        }
                        if newly_escalated {
                            s.auto_stiff_segments += 1;
                        }
                    }
                    escalation_counted |= newly_escalated;
                    switches += 1;
                    // The new stepper gets the whole abort budget. `stiff_abort_after` bounds
                    // how long *a method* is allowed to grind on a segment it cannot step, and
                    // the method just changed; the clamps behind this point measured the one
                    // that was replaced. Bounded by `MAX_AUTO_SWITCHES_PER_SEGMENT`, so the
                    // worst case is that many budgets rather than an unbounded reprieve.
                    min_step_clamps = 0;
                }
            }
        }

        // Stiff-abort budget (#708): a segment that has clamped this many times is
        // stability-limited, and the rest of its `max_steps` allowance buys wall time, not
        // accuracy. Stop here and let the tail freeze-pad — the same outcome the unusable-
        // attempt break above produces, only cheaper to reach. Checked after the accept block
        // so the clamped step's own `saveat` values are still saved.
        //
        // `t < tf` is part of the test, not an optimization: a segment whose *last* step
        // clamps still reaches `tf` with every `saveat` saved, and breaking out there abandons
        // nothing. Counting it would report a fully-integrated segment as truncated, which is
        // the one thing this counter must never do.
        if t < tf - 1e-15
            && opts
                .stiff_abort_after
                .is_some_and(|budget| min_step_clamps >= budget.max(1))
        {
            if let Some(s) = stats.as_deref_mut() {
                s.stiff_aborted_segments += 1;
            }
            break;
        }
        // On reject: (u, t) is unchanged, so the stepper's carried state stays valid.

        // Adapt the step (memoryless I-controller — see the note above).
        let safety = 0.9;
        let factor = if !err_norm.is_finite() {
            NONFINITE_ERR_SHRINK_FACTOR
        } else if err_norm > 1e-15 {
            safety * err_norm.powf(-err_exp)
        } else {
            5.0
        };
        dt = dt_eff * factor.clamp(0.2, 5.0);
        dt = dt.max(opts.min_dt);
    }

    // Written once here rather than inside `record`, which cannot see which stepper is active.
    if let Some(s) = stats.as_deref_mut() {
        s.stiff_min_step_clamped_steps += stiff_min_step_clamps;
    }

    // Fill any remaining saveat / interp times with the last state.
    while save_idx < saveat.len() {
        results.push(SolPointG {
            t: saveat[save_idx],
            u: u.clone(),
        });
        save_idx += 1;
    }
    while interp_idx < interp_at.len() {
        interp_results.push(SolPointG {
            t: interp_at[interp_idx],
            u: u.clone(),
        });
        interp_idx += 1;
    }

    (results, interp_results)
}

/// Integrate an ODE system from `t_start` to `t_end`, returning the state at each `saveat`
/// time. Uses whichever stepper [`OdeSolverOptions::method`] selects.
pub fn solve_ode(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    t_span: (f64, f64),
    params: &[f64],
    saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<SolPoint> {
    solve_ode_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode`] with optional adaptive-step instrumentation.
///
/// The counters are intentionally local to this integration segment. Higher layers that split
/// by dose/observation boundaries can aggregate across calls to classify a full subject or fit.
pub fn solve_ode_with_stats(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    t_span: (f64, f64),
    params: &[f64],
    saveat: &[f64],
    opts: &OdeSolverOptions,
    stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPoint> {
    solve_ode_dense(rhs, u0, t_span, params, saveat, &[], opts, stats).0
}

/// [`solve_ode`] with **dense soft-sampling**: `(hard, soft)` where `hard` is the state at each
/// `saveat` time and `soft` the interpolated state at each `interp_at` time.
///
/// The soft channel reads inside an already-accepted step (#570), so a joint PK-TTE fit's
/// Gaussian predictions are identical whether or not the hazard readout is requested — for
/// every method, since the interpolation comes from the [`Stepper`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_ode_dense(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u0: &[f64],
    t_span: (f64, f64),
    params: &[f64],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPoint>, Vec<SolPoint>) {
    let (hard, soft) =
        integrate_resolved_g(rhs, u0, t_span, params, saveat, interp_at, opts, stats);
    let to_points = |v: Vec<SolPointG<f64>>| -> Vec<SolPoint> {
        v.into_iter().map(|p| SolPoint { t: p.t, u: p.u }).collect()
    };
    (to_points(hard), to_points(soft))
}

/// Outcome of [`solve_ode_until_threshold`] over a single integration span.
///
/// This is the **segment-level** primitive behind the TTE event-time sampler
/// (plan §8.8.3): the simulation entry point drives it once per dose segment
/// (mirroring [`crate::ode::ode_dense_solve_states`]), carrying the *absolute*
/// monitored threshold and the end-of-segment state across segments. The public
/// wrapper maps a terminal [`ReachedEnd`](ThresholdCrossing::ReachedEnd) (the
/// monitor never reached the threshold by the horizon) to a right-censored draw,
/// [`Crossed`](ThresholdCrossing::Crossed) to an event, and
/// [`Failed`](ThresholdCrossing::Failed) to a hard error — a failed solve must
/// never be laundered into a censored subject.
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdCrossing {
    /// The monitored state reached `threshold` at this time within the span.
    Crossed(f64),
    /// Integrated cleanly to the end of the span without crossing; the caller's
    /// `u` now holds the state at `t_end` (carry it into the next segment).
    ReachedEnd,
    /// No meaningful crossing can be reported: a non-finite monitored state, a
    /// monitored state that *decreased* over a step (non-monotone accumulator ⇒ a
    /// negative rate — for TTE, a negative hazard, which invalidates the crossing
    /// argument), or the step budget exhausted before reaching `t_end`. The
    /// message names the cause.
    Failed(String),
}

/// Integrate `rhs` over `t_span`, **halting at the first time `u[monitor]` reaches
/// `threshold`** — a one-sided upward crossing, for a monitored state expected to be monotone
/// non-decreasing (a cumulative-hazard accumulator `dCHZ/dt = h ≥ 0`).
///
/// `u` is the state at `t_span.0` on entry and is advanced **in place** to the state at
/// `t_span.1` on [`ReachedEnd`](ThresholdCrossing::ReachedEnd) (so a segmented caller can carry
/// it into the next dose segment); on `Crossed` / `Failed` its contents are unspecified.
///
/// The crossing is localized *inside* the step that brackets it, by bisecting the stepper's own
/// continuous extension — so this works for every [`OdeMethod`], and a method added later needs
/// no changes here.
pub fn solve_ode_until_threshold(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u: &mut [f64],
    t_span: (f64, f64),
    params: &[f64],
    opts: &OdeSolverOptions,
    monitor: usize,
    threshold: f64,
) -> ThresholdCrossing {
    // The explicit re-solve pins its method, which also takes mid-segment switching off it:
    // a fallback that could escalate again would be the escalation it is undoing.
    let pinned_explicit = OdeSolverOptions {
        method: OdeMethod::EXPLICIT_FALLBACK,
        ..*opts
    };
    // Both early exits below return without stepping, so probe after them rather than before.
    if u[monitor] >= threshold || (t_span.1 - t_span.0).abs() < 1e-15 {
        return until_threshold_with(
            rhs,
            u,
            t_span,
            params,
            &pinned_explicit,
            monitor,
            threshold,
            OdeMethod::EXPLICIT_FALLBACK,
        )
        .outcome;
    }
    let method = super::stiffness::resolve_method(rhs, u, params, t_span.0, opts);
    // Nothing to guard: a named method is pinned, and an `auto` run that starts explicit and
    // cannot switch later will never put a stiff stepper in play. Returning here also skips
    // the entry-state copy the guarded path needs.
    if opts.method != OdeMethod::Auto
        || (method == OdeMethod::EXPLICIT_FALLBACK && !opts.auto_switch)
    {
        return until_threshold_with(rhs, u, t_span, params, opts, monitor, threshold, method)
            .outcome;
    }
    // Escalated by the probe — at the span's start, or by a mid-segment re-probe (#1080
    // Part C) — and so the same guard `integrate_resolved_g` applies, in the shape this driver
    // reports failure in. A stiff method that cannot form a step, or that drives the monitored
    // accumulator non-finite or backwards, comes back as `Failed`; the segment is then re-run
    // explicitly from the *entry* state, since `u` is unspecified after a failure. A censored
    // draw laundered out of a diverged stiff solve would be the worst possible outcome here,
    // so the retry is not optional.
    let u_entry = u.to_vec();
    let run = until_threshold_with(rhs, u, t_span, params, opts, monitor, threshold, method);

    // Both halves of the dense driver's guard, in the shape this driver reports outcomes in.
    //
    // `Failed` is the loud half. The quiet half is a **clamp**: this driver force-accepts at
    // `min_dt` and keeps going, so a stiff method that could not form a step still returns an
    // ordinary-looking `Crossed`/`ReachedEnd` built on a trajectory it did not actually
    // integrate. That is the same freeze-pad shape (#959) the dense guard was widened for, and
    // here it means a wrong simulated event time — or, worse, a censoring laundered out of a
    // diverged solve. Nothing downstream can tell; the flag is the only trace.
    //
    // `ran_stiff` is what keeps the guard from firing on a run that never left the explicit
    // stepper: there the re-solve *is* the same solve, and repeating it would double the cost
    // of every stability-limited explicit crossing search to return the identical answer.
    if !run.ran_stiff
        || (!matches!(run.outcome, ThresholdCrossing::Failed(_)) && !run.stiff_min_step_clamped)
    {
        return run.outcome;
    }
    // `u` is unspecified after a failure and untrustworthy after a stall, so the re-solve
    // starts from the entry state rather than from wherever the escalation left it.
    u.copy_from_slice(&u_entry);
    until_threshold_with(
        rhs,
        u,
        t_span,
        params,
        &pinned_explicit,
        monitor,
        threshold,
        OdeMethod::EXPLICIT_FALLBACK,
    )
    .outcome
}

/// [`solve_ode_until_threshold`] with the stepper already chosen — the body both the probed
/// and the re-solved pass run through.
///
/// Reports `(outcome, stiff_min_step_clamped)`. The second half is what lets the caller's
/// guard see a stall: a clamped step here is force-accepted and the run continues, so a
/// stalled solve returns a perfectly ordinary-looking `Crossed`/`ReachedEnd` that no
/// inspection of the outcome alone can distinguish from a healthy one.
/// Pair a hard failure with the clamp flag. A failure is a failure whether or not the solver
/// clamped on the way there, so the flag is irrelevant to the caller in this arm — but the
/// return type is one type, and spelling it out at eight call sites is noise.
///
/// `ran_stiff` is *not* boilerplate in the same way: it tells the caller's guard whether a
/// stiff stepper was ever in play on this run — at the start or after a mid-segment switch —
/// and therefore whether re-solving explicitly can produce a different answer at all.
#[inline]
fn failed(msg: String, ran_stiff: bool) -> ThresholdRun {
    ThresholdRun {
        outcome: ThresholdCrossing::Failed(msg),
        stiff_min_step_clamped: true,
        ran_stiff,
    }
}

/// What one pass of [`until_threshold_with`] produced, for the guard above it.
struct ThresholdRun {
    outcome: ThresholdCrossing,
    /// A step was force-accepted at `min_dt` **while a stiff stepper was active**. The run
    /// continued afterwards, so this is the only trace such a step leaves in an otherwise
    /// ordinary-looking outcome.
    ///
    /// Scoped to the stiff stepper for the same reason the dense guard's
    /// [`OdeSolverStats::stiff_min_step_clamped_steps`] is: on a run that switched part-way
    /// (#1080 Part C), clamps taken by the explicit stepper *before* the escalation are not
    /// evidence that the stiff method stalled, and re-running such a crossing search with
    /// pinned explicit would move the event time — or launder a censoring — for a limit
    /// belonging to the stepper the switch replaced.
    stiff_min_step_clamped: bool,
    /// A stiff stepper integrated some part of this run — because the segment-start probe
    /// escalated, or because a mid-segment re-probe switched onto one (#1080 Part C).
    ran_stiff: bool,
}

#[allow(clippy::too_many_arguments)]
fn until_threshold_with(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]),
    u: &mut [f64],
    t_span: (f64, f64),
    params: &[f64],
    opts: &OdeSolverOptions,
    monitor: usize,
    threshold: f64,
    method: OdeMethod,
) -> ThresholdRun {
    let (t0, tf) = t_span;
    let ended = |outcome, stiff_min_step_clamped, ran_stiff| ThresholdRun {
        outcome,
        stiff_min_step_clamped,
        ran_stiff,
    };

    // Already at/above threshold at the span start (e.g. threshold ≈ 0 from u≈1, or a non-zero
    // initial accumulator). Catch before stepping so the loop's crossing test can assume
    // `y0 < threshold`.
    if u[monitor] >= threshold {
        return ended(ThresholdCrossing::Crossed(t0), false, false);
    }
    if (tf - t0).abs() < 1e-15 {
        return ended(ThresholdCrossing::ReachedEnd, false, false);
    }

    // Set when a step is force-accepted at `min_dt` (an explicit method that failed its error
    // test, or a linearly implicit one that could not form a step at all) *on a stiff stepper*.
    // The run continues afterwards, so this is the only trace such a step leaves.
    let mut stiff_min_step_clamped = false;
    // Clamps by the currently active stepper, for the abort budget; reset when the stepper is
    // swapped, for the reason `integrate_dense_g` gives at its own reset.
    let mut clamps = 0u32;
    let mut method = method;
    let mut stepper = make_stepper::<f64>(u.len(), method);
    // This driver localizes the crossing inside a step, so it always interpolates.
    stepper.set_dense_required(true);
    let mut err_exp = stepper.err_exp();
    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);
    // Mid-segment switching, exactly as the dense driver does it (#1080 Part C): `auto` only,
    // re-probed every `AUTO_SWITCH_PROBE_INTERVAL` accepted steps, capped per span. The
    // crossing search needs it at least as much as the dense path does — its answer *is* the
    // likelihood contribution (an event time, or a censoring), so a segment that turns stiff
    // and is integrated badly does not merely cost accuracy in a prediction.
    let switching = opts.method == OdeMethod::Auto && opts.auto_switch;
    let mut accepted_since_probe = 0usize;
    let mut switches = 0usize;
    let mut ran_stiff = method != OdeMethod::EXPLICIT_FALLBACK;

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            return ended(
                ThresholdCrossing::ReachedEnd,
                stiff_min_step_clamped,
                ran_stiff,
            );
        }

        let dt_eff = dt.min(tf - t);
        let err_norm = stepper.attempt(rhs, u, params, t, dt_eff, opts);
        let usable = stepper.attempt_usable();
        if !usable && dt_eff <= opts.min_dt {
            return failed(
                format!(
                    "the step at t={t:.6} could not be formed even at min_dt \
                     (singular Jacobian system)"
                ),
                ran_stiff,
            );
        }

        let accepted = usable && (err_norm <= 1.0 || dt_eff <= opts.min_dt);
        // Mirrors `OdeSolverStats::record`: a force-accept at `min_dt` whose error test did not
        // pass (`!(err_norm <= 1.0)`, which also catches a non-finite norm) is a clamp.
        if accepted && !(err_norm <= 1.0) && dt_eff <= opts.min_dt {
            stiff_min_step_clamped |= method != OdeMethod::EXPLICIT_FALLBACK;
            clamps += 1;
        }
        if accepted {
            let y0 = u[monitor];
            let y1 = stepper.u_new()[monitor];

            if !y1.is_finite() {
                return failed(
                    format!(
                        "monitored state {monitor} became non-finite at t={:.6}",
                        t + dt_eff
                    ),
                    ran_stiff,
                );
            }
            // Scale-aware monotonicity floor: a real negative rate produces a decrease ≫ this;
            // only round-off sits below it.
            let mono_tol = scale_tol(opts.abstol, opts.reltol, y0, y1);
            if y1 < y0 - mono_tol {
                return failed(
                    format!(
                        "monitored state {monitor} decreased ({y0:.6} → {y1:.6}) over \
                         [{t:.6}, {:.6}]: non-monotone accumulator (negative rate / hazard)",
                        t + dt_eff
                    ),
                    ran_stiff,
                );
            }

            if y1 >= threshold {
                // Crossing in (t, t+dt_eff]; bisect the step's continuous extension on the
                // bracket the accepted step proves (y0 < threshold ≤ y1). 64 halvings drives
                // the bracket below machine precision of the step — interpolant evals only.
                let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    if stepper.interpolate_component(mid, u, dt_eff, monitor) < threshold {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return ended(
                    ThresholdCrossing::Crossed(t + 0.5 * (lo + hi) * dt_eff),
                    stiff_min_step_clamped,
                    ran_stiff,
                );
            }

            t += dt_eff;
            u.copy_from_slice(stepper.u_new());
            stepper.on_accept();

            // Mid-segment stiffness re-probe (#1080 Part C), the dense driver's rule in this
            // driver's loop: read the same discriminator at the state just reached and swap the
            // stepper in place when it disagrees. Taken after the crossing test, so a step that
            // both advances and brackets the crossing still returns through the interpolant of
            // the stepper that produced it.
            accepted_since_probe += 1;
            if switching
                && switches < MAX_AUTO_SWITCHES_PER_SEGMENT
                && accepted_since_probe >= AUTO_SWITCH_PROBE_INTERVAL
            {
                accepted_since_probe = 0;
                let verdict = super::stiffness::resolve_method(rhs, u, params, t, opts);
                if verdict != method {
                    method = verdict;
                    stepper = make_stepper::<f64>(u.len(), method);
                    stepper.set_dense_required(true);
                    err_exp = stepper.err_exp();
                    ran_stiff |= method != OdeMethod::EXPLICIT_FALLBACK;
                    switches += 1;
                    // Fresh abort budget for the new stepper — see `integrate_dense_g`.
                    clamps = 0;
                }
            }
        }

        // Stiff-abort budget (#708), in this driver's own shape: a stability-limited crossing
        // search reports `Failed` rather than freeze-padding, because the outcome here *is*
        // the answer (an event time, or a censoring) and a padded one would be laundered into
        // the likelihood as if it had been integrated.
        //
        // Checked *after* the accept block for the same reason the dense driver checks after
        // its saves: a clamped step can still be the step that brackets the crossing, and the
        // bisection above has already returned it. Giving up on an answer the solver just
        // found would be a strictly worse outcome than the cost this budget exists to bound.
        if opts
            .stiff_abort_after
            .is_some_and(|budget| clamps >= budget.max(1))
        {
            return failed(
                format!(
                    "aborted at t={t:.6} after {clamps} min-dt clamp(s) \
                     (ode_stiff_abort_after): the segment is stability-limited"
                ),
                ran_stiff,
            );
        }

        let safety = 0.9;
        // Finite-error branch is identical to the dense driver; a *non-finite* err_norm
        // (diverging / NaN RHS, or an unusable attempt) shrinks toward `min_dt` instead of
        // growing, so the step is force-accepted at `min_dt` and the non-finite guard above
        // fires with a clear cause rather than silently burning the step budget.
        let factor = if !err_norm.is_finite() {
            NONFINITE_ERR_SHRINK_FACTOR
        } else if err_norm > 1e-15 {
            safety * err_norm.powf(-err_exp)
        } else {
            5.0
        };
        dt = (dt_eff * factor.clamp(0.2, 5.0)).max(opts.min_dt);
    }

    return failed(
        format!(
            "step budget ({}) exhausted before reaching t_end={tf:.6} (reached t={t:.6})",
            opts.max_steps
        ),
        ran_stiff,
    );
}

/// Generic solution point for the [`solve_ode_g`] sensitivity path.
#[derive(Debug, Clone)]
pub struct SolPointG<T> {
    pub t: f64,
    pub u: Vec<T>,
}

/// [`solve_ode`] generic over the state scalar `T: PkNum`, for the analytic PK-parameter
/// sensitivity path (`T = Dual2<N>`): the *same* stepper and the *same* driver as the scalar
/// path, instantiated at a dual number so the jets carry `∂u/∂p` and `∂²u/∂p²` through the
/// integration. `params` holds the PK parameters seeded as dual variables.
///
/// **Step-size control reads `.val()` only** — the accept/reject decision and `dt` adaptation
/// depend on values, never on derivatives, so the derivative flows through a *fixed* step
/// sequence. (Adapting on a derivative norm would make the sensitivity inconsistent with the
/// prediction.)
pub fn solve_ode_g<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<SolPointG<T>> {
    solve_ode_g_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode_g`] with optional adaptive-step instrumentation.
pub fn solve_ode_g_with_stats<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    opts: &OdeSolverOptions,
    stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPointG<T>> {
    solve_ode_g_dense(rhs, u0, t_span, params, saveat, &[], opts, stats).0
}

/// [`solve_ode_g`] with dense soft-sampling — the sensitivity-carrying twin of
/// [`solve_ode_dense`]. Available for every method because the interpolation comes from the
/// [`Stepper`], so an analytic-gradient consumer that needs an in-step readout (a hazard state
/// at an event time, say) has one to call.
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_g_dense<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    integrate_resolved_g(rhs, u0, t_span, params, saveat, interp_at, opts, stats)
}

thread_local! {
    /// Per-thread collector for [`SolverStatsScope`]. `None` when no scope is active, which
    /// is every production integration outside the post-fit diagnostic pass.
    static STATS_SINK: std::cell::Cell<Option<OdeSolverStats>> = const { std::cell::Cell::new(None) };
}

/// Collect [`OdeSolverStats`] from every integration this thread performs while the scope is
/// alive, without threading an `&mut` through the prediction dispatchers.
///
/// The alternative — an `Option<&mut OdeSolverStats>` parameter carried from the fit down
/// through `compute_predictions_*`, the event-driven walker, the SS equilibrator and the
/// modified-release reroute — would touch every prediction path to serve one diagnostic that
/// production never asks for. A scope instead reads the *ordinary* predictor: the post-fit
/// pass runs exactly the dispatch a user's model runs, so the counters describe the real
/// integration rather than a diagnostic re-creation of it (which is what
/// [`crate::ode::ode_predictions_with_solver_stats`] gives, and why it never acquired a
/// production caller).
///
/// Thread-local and scoped, so a scope opened on the fit thread sees that thread's work only.
/// Nesting restores the outer scope on drop.
pub(crate) struct SolverStatsScope {
    prev: Option<OdeSolverStats>,
}

impl SolverStatsScope {
    /// Begin collecting on this thread.
    pub(crate) fn enter() -> Self {
        Self {
            prev: STATS_SINK.with(|c| c.replace(Some(OdeSolverStats::default()))),
        }
    }

    /// Counters accumulated since [`enter`](Self::enter).
    pub(crate) fn collected(&self) -> OdeSolverStats {
        STATS_SINK.with(|c| c.get()).unwrap_or_default()
    }
}

impl Drop for SolverStatsScope {
    fn drop(&mut self) {
        STATS_SINK.with(|c| c.set(self.prev));
    }
}

/// Whether a [`SolverStatsScope`] is collecting on this thread.
#[inline]
fn stats_sink_active() -> bool {
    STATS_SINK.with(|c| c.get()).is_some()
}

/// Add one integration's counters to the active scope, if any.
#[inline]
fn record_to_stats_sink(stats: &OdeSolverStats) {
    STATS_SINK.with(|c| {
        if let Some(mut acc) = c.get() {
            acc.merge(stats);
            c.set(Some(acc));
        }
    });
}

/// [`integrate_resolved_g_inner`] with the thread-local [`SolverStatsScope`] tee.
///
/// Off the diagnostic path this is one `Cell` read per segment and the same call as before.
#[allow(clippy::too_many_arguments)]
fn integrate_resolved_g<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    if !stats_sink_active() {
        return integrate_resolved_g_inner(rhs, u0, t_span, params, saveat, interp_at, opts, stats);
    }
    let mut local = OdeSolverStats::default();
    let out = integrate_resolved_g_inner(
        rhs,
        u0,
        t_span,
        params,
        saveat,
        interp_at,
        opts,
        Some(&mut local),
    );
    record_to_stats_sink(&local);
    if let Some(s) = stats {
        s.merge(&local);
    }
    out
}

/// [`integrate_dense_g`] with the method resolved first, and the guard that makes
/// [`OdeMethod::Auto`] safe to escalate: the one place `auto` turns into a stepper.
///
/// Every implicit stepper in the library was measured diverging on *some* model where the
/// explicit family stayed clean (#978), and those failures are disjoint across methods — so a
/// failed escalation is a known, recoverable outcome rather than an impossible one. When one
/// happens the segment is re-solved with the explicit default and *that* result is what the
/// caller gets, which bounds `auto`'s worst case to "the explicit answer, twice the cost"
/// instead of a corrupted objective.
///
/// The guard is scoped to escalations. A named `ode_method` is honoured exactly as before,
/// failed result included: a user who asked for `rodas4` gets `rodas4`, and a fixed method that
/// silently re-solved as something else would be the worse surprise.
///
/// Generic over `T`, so the `f64` prediction and the `Dual2` sensitivity solve share one copy
/// of both the resolution and the guard. Both read `.val()` only and see the same values, so
/// they make the same decision on the same segment and the gradient keeps differentiating the
/// trajectory the predictor reports.
#[allow(clippy::too_many_arguments)]
fn integrate_resolved_g_inner<T: PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]),
    u0: &[T],
    t_span: (f64, f64),
    params: &[T],
    saveat: &[f64],
    interp_at: &[f64],
    opts: &OdeSolverOptions,
    mut stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    // A zero-length span returns `u0` at every requested time without stepping, so there is
    // no stepper to choose and no reason to pay for a Jacobian. Dosing timelines produce these
    // wherever two events share a time.
    let method = if (t_span.1 - t_span.0).abs() < 1e-15 {
        OdeMethod::EXPLICIT_FALLBACK
    } else {
        super::stiffness::resolve_method(rhs, u0, params, t_span.0, opts)
    };
    let run = |method: OdeMethod, opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>| {
        integrate_dense_g(
            method, rhs, u0, t_span, params, saveat, interp_at, opts, stats,
        )
    };
    // The explicit re-solve the guard falls back to **pins** its method, which is also what
    // takes mid-segment switching off it (`integrate_dense_g` switches only under `auto`): a
    // re-solve that could escalate again would be the escalation the guard is undoing.
    let pinned_explicit = OdeSolverOptions {
        method: OdeMethod::EXPLICIT_FALLBACK,
        ..*opts
    };

    // Not a guarded solve — a named method, or `auto` reading the system as non-stiff *and*
    // unable to change its mind later. Nothing to discard, and the caller's counters are
    // written directly by the one and only solve.
    let may_switch = opts.method == OdeMethod::Auto && opts.auto_switch;
    if opts.method != OdeMethod::Auto || (method == OdeMethod::EXPLICIT_FALLBACK && !may_switch) {
        return run(method, opts, stats);
    }

    // An escalation — at the segment's start, or reached mid-segment by a re-probe — is scored
    // on its own counters so the guard can read what *this* attempt did, rather than a total
    // the caller has been accumulating across segments.
    let escalated_at_start = method != OdeMethod::EXPLICIT_FALLBACK;
    let mut attempt = OdeSolverStats {
        auto_stiff_segments: usize::from(escalated_at_start),
        ..Default::default()
    };
    let out = run(method, opts, Some(&mut attempt));

    // Two ways an escalation goes wrong, and the second is the one that bites. A non-finite
    // trajectory is loud. A stiff method that clamps at `min_dt` is not: the driver stops and
    // freeze-pads the remaining saves with the last state, so the result comes back finite,
    // plausible, and wrong (#959) — measured at 4.8e282 on a system whose true solution had
    // already blown up. Both are the same failure to the caller, so both discard the result.
    //
    // A clamp means the *stiff* method could not form a step, which is the one thing it was
    // escalated to do; it is not the ordinary cost of a hard segment. So this trigger fires
    // where the escalation genuinely failed, not merely where it worked hard.
    //
    // The guard covers a segment that reached a stiff stepper mid-way (#1080 Part C) exactly
    // as it covers one that started on it — the failure it repairs is a property of the stiff
    // solve, not of when the decision was taken. It must *not* cover a segment that never left
    // the explicit stepper: there the fallback and the attempt are the same solve, so
    // discarding a clamped result would only buy a second copy of it.
    //
    // The stall test reads the clamps taken **on a stiff stepper**, not every clamp the segment
    // produced. On a segment that switched part-way those differ, and only the stiff ones are
    // evidence of the failure this guard repairs: a segment that clamped on the explicit
    // stepper, was escalated by a re-probe, and then integrated cleanly to the end has just
    // been rescued by the switch, and discarding it would re-solve it with pinned explicit —
    // the exact result the escalation avoided.
    let ran_stiff = attempt.auto_stiff_segments > 0;
    let stalled = attempt.stiff_min_step_clamped_steps > 0;
    let usable = !ran_stiff || (!stalled && finite_g(&out.0) && finite_g(&out.1));
    if !usable {
        attempt.auto_stiff_rejected = 1;
        // The clamps were taken, so they stay in `min_step_clamped_steps`; they are *also*
        // recorded as discarded, because the trajectory they damaged is about to be thrown
        // away and re-solved. Without this split every rejected escalation would read, to a
        // caller, as "the answer you got was freeze-padded" — the opposite of what the guard
        // just did. For the same reason a budgeted abort inside a discarded attempt is not a
        // truncated *result*: it truncated an attempt nobody receives, so it is not counted.
        attempt.discarded_clamped_steps = attempt.min_step_clamped_steps;
        attempt.stiff_aborted_segments = 0;
    }
    // The stiff attempt's steps stay counted either way — they were taken, and they cost what
    // they cost. A fit that reads slow *and* shows a rejection is paying for both solves.
    if let Some(s) = stats.as_deref_mut() {
        s.merge(&attempt);
    }
    if usable {
        return out;
    }
    run(OdeMethod::EXPLICIT_FALLBACK, &pinned_explicit, stats)
}

/// Whether every saved state is finite, reading values only (`T = Dual2` decides identically
/// to `T = f64`, which is what keeps the two paths on the same stepper).
fn finite_g<T: PkNum>(points: &[SolPointG<T>]) -> bool {
    points
        .iter()
        .all(|p| p.u.iter().all(|v| v.val().is_finite()))
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

    // ---- #570: dense soft-sampling (`solve_ode_dense`) ----

    /// The decisive #570 guarantee: requesting soft (Hermite) samples must **not**
    /// change the hard `saveat` outputs. The soft channel only reads already-
    /// accepted step data, so the adaptive step sequence — and thus every hard save
    /// — is **bit-identical** to the no-soft-sampling wrapper. Without this, the
    /// Gaussian predictions in a shared joint PK-TTE solve would move (the exact
    /// regression #570 must avoid). Asserts exact f64 equality, not `approx`.
    #[test]
    fn solve_ode_dense_soft_samples_do_not_perturb_hard_saves() {
        // Two-state system (decay + a CHZ-like accumulator) so steps actually adapt.
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -0.3 * u[0];
            du[1] = 0.3 * u[0];
        };
        let saveat = vec![1.0, 3.0, 7.0, 12.0];
        let interp_at = vec![0.5, 2.2, 4.9, 8.1, 11.3];
        let opts = OdeSolverOptions::default();

        let reference = solve_ode(&rhs, &[1.0, 0.0], (0.0, 12.0), &[], &saveat, &opts);
        let (hard, soft) = solve_ode_dense(
            &rhs,
            &[1.0, 0.0],
            (0.0, 12.0),
            &[],
            &saveat,
            &interp_at,
            &opts,
            None,
        );

        assert_eq!(hard.len(), saveat.len());
        assert_eq!(soft.len(), interp_at.len());
        for (a, b) in hard.iter().zip(reference.iter()) {
            assert_eq!(a.t, b.t);
            assert_eq!(a.u, b.u, "soft sampling perturbed a hard save");
        }
    }

    /// Soft samples reproduce the trajectory: pinned tightly against the analytic
    /// solution at small step size (the interpolant itself), and shown to match an
    /// independent clamped solve to within solver tolerance at production tolerance
    /// (the real operating point of the joint PK-TTE share — CHZ read at event
    /// times off the obs-clamped Gaussian solve instead of a dedicated TTE solve).
    #[test]
    fn solve_ode_dense_soft_samples_are_accurate() {
        // u0 = e^{-0.2 t}; u1 = 1 − e^{-0.2 t} (a cumulative accumulator).
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -0.2 * u[0];
            du[1] = 0.2 * u[0];
        };
        let saveat = vec![2.0, 6.0, 10.0];
        let interp_at = vec![1.0, 6.0, 8.5]; // between, on, and off the saveat grid

        // Tight tolerance ⇒ small steps ⇒ the Hermite read-back is near-exact, so
        // this pins the *interpolant* rather than the solver's own truncation.
        let tight = OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-9,
            ..Default::default()
        };
        let (_h, soft) = solve_ode_dense(
            &rhs,
            &[1.0, 0.0],
            (0.0, 10.0),
            &[],
            &saveat,
            &interp_at,
            &tight,
            None,
        );
        for (i, s) in soft.iter().enumerate() {
            let t = interp_at[i];
            assert_relative_eq!(s.t, t, epsilon = 1e-12);
            assert_relative_eq!(s.u[0], (-0.2 * t).exp(), epsilon = 1e-6);
            assert_relative_eq!(s.u[1], 1.0 - (-0.2 * t).exp(), epsilon = 1e-6);
        }

        // Production tolerance (reltol 1e-4): the read-back matches an independent
        // clamped solve to within solver tolerance even where the obs grid is coarse
        // (the 8.5 point sits inside a large 6→10 step). 1e-3 covers Hermite-over-a-
        // large-step error while still catching a broken interpolant (≫ 1e-3 off).
        let prod = OdeSolverOptions::default();
        let (_h2, soft2) = solve_ode_dense(
            &rhs,
            &[1.0, 0.0],
            (0.0, 10.0),
            &[],
            &saveat,
            &interp_at,
            &prod,
            None,
        );
        let clamped = solve_ode(&rhs, &[1.0, 0.0], (0.0, 10.0), &[], &interp_at, &prod);
        for (i, s) in soft2.iter().enumerate() {
            assert_relative_eq!(s.u[0], clamped[i].u[0], epsilon = 1e-3);
            assert_relative_eq!(s.u[1], clamped[i].u[1], epsilon = 1e-3);
        }
    }

    /// Edge cases: a soft point coinciding with a `saveat`/step boundary lands at
    /// `s = 1` (the exact saved state); a point at the span start clamps to `u0`.
    #[test]
    fn solve_ode_dense_soft_boundary_and_start() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -u[0];
        };
        let saveat = vec![4.0];
        let interp_at = vec![0.0, 4.0];
        let opts = OdeSolverOptions::default();
        let (hard, soft) = solve_ode_dense(
            &rhs,
            &[1.0],
            (0.0, 4.0),
            &[],
            &saveat,
            &interp_at,
            &opts,
            None,
        );
        assert_eq!(soft.len(), 2);
        assert_relative_eq!(soft[0].u[0], 1.0, epsilon = 1e-12); // t=0 → u0
        assert_eq!(soft[1].u[0], hard[0].u[0]); // t=4 boundary == hard save (exact)
    }

    /// Zero-width span: the early-return path must populate the soft channel too,
    /// keeping `soft.len() == interp_at.len()`.
    #[test]
    fn solve_ode_dense_zero_span_fills_soft() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = 1.0;
        let opts = OdeSolverOptions::default();
        let (hard, soft) = solve_ode_dense(
            &rhs,
            &[7.0],
            (3.0, 3.0),
            &[],
            &[3.0],
            &[3.0, 3.0],
            &opts,
            None,
        );
        assert_eq!(hard.len(), 1);
        assert_eq!(soft.len(), 2);
        assert!(soft.iter().all(|p| (p.u[0] - 7.0).abs() < 1e-12));
    }

    /// Primitive-level degenerate oracle: a constant accumulation rate `λ` is the
    /// ODE form of a constant hazard, whose analytic event time for `CHZ = −log u`
    /// is `t* = −log(u)/λ`. Here `dCHZ/dt = 2`, `threshold = 10` ⇒ exact crossing at
    /// `t* = 5`. The Hermite interpolant is exact on a linear trajectory, so the
    /// bisection should hit `5` to round-off.
    #[test]
    fn until_threshold_constant_rate_is_exact() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 2.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 100.0), &[], &opts, 0, 10.0) {
            ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 5.0, epsilon = 1e-9),
            other => panic!("expected Crossed(5.0), got {other:?}"),
        }
    }

    /// Drug-driven hazard with a closed form: a decaying drug `dC/dt = −0.5·C`,
    /// `C(0)=10`, feeds `dCHZ/dt = 0.1·C` ⇒ `CHZ(t) = 2·(1 − e^{−0.5t})`. Solving
    /// `CHZ = 1` gives `t* = 2·ln 2 ≈ 1.386294`. Exercises a non-zero monitor index
    /// (`CHZ` is state 1) and a genuinely curved accumulator.
    #[test]
    fn until_threshold_decaying_drug_hazard_matches_closed_form() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -0.5 * u[0]; // concentration
            du[1] = 0.1 * u[0]; // cumulative hazard
        };
        let opts = OdeSolverOptions {
            abstol: 1e-10,
            reltol: 1e-9,
            ..OdeSolverOptions::default()
        };
        let mut u = [10.0, 0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1000.0), &[], &opts, 1, 1.0) {
            ThresholdCrossing::Crossed(t) => {
                assert_relative_eq!(t, 2.0 * std::f64::consts::LN_2, epsilon = 1e-5)
            }
            other => panic!("expected Crossed, got {other:?}"),
        }
    }

    /// Threshold never reached within the span ⇒ `ReachedEnd`, and `u` is advanced
    /// in place to the end-of-span state (so a segmented caller can carry it).
    #[test]
    fn until_threshold_not_reached_advances_state() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        let outcome = solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 1000.0);
        assert_eq!(outcome, ThresholdCrossing::ReachedEnd);
        assert_relative_eq!(u[0], 10.0, epsilon = 1e-6);
    }

    /// The event-time driver honours `auto` too (#978), and has to: it is the path a
    /// joint PK-TTE fit locates event times on, and a stiff hazard system left on the explicit
    /// method exhausts its step budget and comes back `Failed` — which the TTE layer must not
    /// launder into a censored subject.
    ///
    /// `d/dt(fast) = -1e4·(fast − 1)` relaxes to 1 on a 1e-4 timescale, so the accumulator
    /// `d/dt(chz) = fast` grows at ~1 per unit and crosses 4.0 just after t = 4. RK45 needs
    /// `dt < 2e-4` for stability, so it runs out of its default 10 000 steps around t = 2 —
    /// before the crossing. (The threshold is 4 and not 2 for exactly that reason: at 2 the
    /// explicit method reaches the crossing on its last few steps and the contrast vanishes.)
    #[test]
    fn until_threshold_escalates_a_stiff_system_and_finds_the_crossing() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -1.0e4 * (u[0] - 1.0);
            du[1] = u[0];
        };

        let mut u = [0.0, 0.0];
        let auto = solve_ode_until_threshold(
            &rhs,
            &mut u,
            (0.0, 5.0),
            &[],
            &OdeSolverOptions::default(),
            1,
            4.0,
        );
        match auto {
            ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 4.0001, epsilon = 1e-3),
            other => panic!("expected a crossing under auto, got {other:?}"),
        }

        // Non-vacuity, and the reason the escalation matters here: the same system on the
        // explicit method does not merely run slower, it fails outright.
        let mut u_rk = [0.0, 0.0];
        let explicit = solve_ode_until_threshold(
            &rhs,
            &mut u_rk,
            (0.0, 5.0),
            &[],
            &OdeSolverOptions {
                method: OdeMethod::Rk45,
                ..Default::default()
            },
            1,
            4.0,
        );
        assert!(
            matches!(explicit, ThresholdCrossing::Failed(_)),
            "RK45 was expected to fail on this stiff system, got {explicit:?}"
        );
    }

    /// The event-time driver's half of the `auto` guard: an escalation that comes back
    /// `Failed` is re-run on the explicit method, from the **entry** state (`u` is unspecified
    /// after a failure, so the retry cannot reuse it).
    ///
    /// `d/dt(x) = x²` from `x₀ = 1e6` blows up at `t* = 1e-6`; both methods fail, which is the
    /// point — what is under test is that the retry *happens*, not that it rescues this
    /// system. Counting right-hand-side calls is what makes that observable: a failure the
    /// guard did not retry would cost one solve, not two.
    #[test]
    fn until_threshold_retries_a_failed_escalation_on_the_explicit_method() {
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            calls.set(calls.get() + 1);
            du[0] = u[0] * u[0];
            du[1] = u[0];
        };
        let entry = [1.0e6, 0.0];

        let mut u = entry;
        let named = solve_ode_until_threshold(
            &rhs,
            &mut u,
            (0.0, 1.0),
            &[],
            &OdeSolverOptions {
                method: OdeMethod::Rodas4,
                ..Default::default()
            },
            1,
            f64::INFINITY,
        );
        assert!(matches!(named, ThresholdCrossing::Failed(_)), "{named:?}");
        let named_calls = calls.get();

        calls.set(0);
        let mut u = entry;
        let auto = solve_ode_until_threshold(
            &rhs,
            &mut u,
            (0.0, 1.0),
            &[],
            &OdeSolverOptions::default(),
            1,
            f64::INFINITY,
        );
        assert!(matches!(auto, ThresholdCrossing::Failed(_)), "{auto:?}");
        assert!(
            calls.get() > named_calls,
            "auto spent {} right-hand-side calls against rodas4's {named_calls} — the failed \
             escalation was not retried on the explicit method",
            calls.get()
        );
    }

    /// The quiet half of the event-time guard, and the reason it cannot key on
    /// [`ThresholdCrossing::Failed`] alone.
    ///
    /// The monitored accumulator here is `d/dt(chz) = 1` — exactly linear, independent of the
    /// stiff state beside it. So when the stiff state cannot be stepped at `min_dt` the driver
    /// force-accepts, keeps going, and the monitor sails through: finite, monotone, crossing
    /// the threshold at exactly the right time. The outcome is an ordinary `Crossed` built on a
    /// trajectory the solver never actually integrated (#959's freeze-pad, on the TTE path),
    /// and no inspection of the outcome can tell. Counting right-hand-side calls is what makes
    /// the re-solve observable.
    #[test]
    fn until_threshold_retries_an_escalation_that_stalled_but_still_reported_success() {
        use std::cell::Cell;
        let calls = Cell::new(0usize);
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            calls.set(calls.get() + 1);
            du[0] = -1.0e4 * (u[0] - 1.0);
            du[1] = 1.0;
        };
        // `min_dt = 1.0` is far above what this system's fast mode admits, so every step is a
        // force-accepted clamp.
        let stalling = OdeSolverOptions {
            initial_dt: 1.0,
            min_dt: 1.0,
            ..Default::default()
        };

        let mut u = [0.0, 0.0];
        let named = solve_ode_until_threshold(
            &rhs,
            &mut u,
            (0.0, 5.0),
            &[],
            &OdeSolverOptions {
                method: OdeMethod::Rodas4,
                ..stalling
            },
            1,
            3.0,
        );
        assert!(
            matches!(named, ThresholdCrossing::Crossed(_)),
            "precondition: the stalled stiff solve must still *report success*, got {named:?}"
        );
        let named_calls = calls.get();

        calls.set(0);
        let mut u = [0.0, 0.0];
        let auto = solve_ode_until_threshold(&rhs, &mut u, (0.0, 5.0), &[], &stalling, 1, 3.0);
        assert!(matches!(auto, ThresholdCrossing::Crossed(_)), "{auto:?}");
        assert!(
            calls.get() > named_calls,
            "auto spent {} right-hand-side calls against rodas4's {named_calls} — a stalled \
             escalation that reported success was accepted instead of being re-solved",
            calls.get()
        );
    }

    /// `Auto` never reaches [`make_stepper`] through a driver — every one resolves it first —
    /// but the arm exists so a caller that builds a stepper straight from the options keeps
    /// integrating instead of panicking inside a fit's worker thread. Pin that it really is the
    /// explicit stepper and not a placeholder.
    #[test]
    fn make_stepper_falls_back_to_the_explicit_stepper_for_auto() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.3 * u[0];
        let opts = OdeSolverOptions::default();
        let u = [2.0];

        let mut from_auto = make_stepper::<f64>(1, OdeMethod::Auto);
        let mut from_rk45 = make_stepper::<f64>(1, OdeMethod::EXPLICIT_FALLBACK);
        let e_auto = from_auto.attempt(&rhs, &u, &[], 0.0, 0.1, &opts);
        let e_rk45 = from_rk45.attempt(&rhs, &u, &[], 0.0, 0.1, &opts);

        assert_eq!(e_auto, e_rk45);
        assert_eq!(from_auto.u_new(), from_rk45.u_new());
        assert_eq!(from_auto.err_exp(), from_rk45.err_exp());
    }

    /// A decreasing monitored state (negative rate ⇒ negative hazard) is not a
    /// censor — it invalidates the crossing argument and must be a hard failure.
    #[test]
    fn until_threshold_negative_rate_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [5.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 100.0) {
            ThresholdCrossing::Failed(msg) => assert!(msg.contains("non-monotone"), "msg: {msg}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// Already at/above the threshold at the span start ⇒ `Crossed(t0)` without
    /// stepping (the `u ≈ 1 ⇒ threshold ≈ 0` edge, or a non-zero initial state).
    #[test]
    fn until_threshold_already_crossed_at_start() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [5.0];
        match solve_ode_until_threshold(&rhs, &mut u, (2.0, 10.0), &[], &opts, 0, 3.0) {
            ThresholdCrossing::Crossed(t) => assert_relative_eq!(t, 2.0, epsilon = 1e-12),
            other => panic!("expected Crossed(2.0), got {other:?}"),
        }
    }

    /// A non-finite trajectory cannot yield a meaningful crossing ⇒ hard failure,
    /// not a silent censor at the horizon.
    #[test]
    fn until_threshold_non_finite_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 10.0), &[], &opts, 0, 10.0) {
            ThresholdCrossing::Failed(_) => {}
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    /// A zero-width span reaches its end immediately, without stepping.
    #[test]
    fn until_threshold_zero_span_reaches_end() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions::default();
        let mut u = [0.0];
        assert_eq!(
            solve_ode_until_threshold(&rhs, &mut u, (5.0, 5.0), &[], &opts, 0, 10.0),
            ThresholdCrossing::ReachedEnd
        );
    }

    /// Exhausting the step budget before reaching `t_end` is a failure, not a
    /// censor — a censored draw requires a *clean* integration to the horizon.
    #[test]
    fn until_threshold_step_budget_exhausted_fails() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = 1.0;
        };
        let opts = OdeSolverOptions {
            max_steps: 2,
            ..OdeSolverOptions::default()
        };
        let mut u = [0.0];
        // Threshold far beyond what 2 steps can reach within the (huge) span.
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1e6), &[], &opts, 0, 1e9) {
            ThresholdCrossing::Failed(msg) => assert!(msg.contains("budget"), "msg: {msg}"),
            other => panic!("expected Failed, got {other:?}"),
        }
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
    fn solve_ode_with_stats_counts_attempts() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -10.0 * u[0];
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(
            stats.attempted_steps,
            stats.accepted_steps + stats.rejected_steps
        );
        assert!(stats.accepted_steps > 0, "stats = {stats:?}");
        assert!(stats.rejected_steps > 0, "stats = {stats:?}");
        assert_eq!(stats.min_step_clamped_steps, 0);
    }

    #[test]
    fn solve_ode_with_stats_counts_min_step_clamped_accepts() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -100.0 * u[0];
        };
        // RK45 named rather than defaulted: `|Re λ|max = 100` here, so under the `auto`
        // default (#978) the probe escalates *and* the `min_dt = 1.0` clamp then trips the
        // guard into a second, explicit solve — which is correct behaviour and would double
        // every counter this test is asserting on. What is under test is the explicit
        // driver's clamp bookkeeping.
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            min_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            method: OdeMethod::Rk45,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let _ = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, 1);
        assert_eq!(stats.accepted_steps, 1);
        assert_eq!(stats.rejected_steps, 0);
        assert_eq!(stats.min_step_clamped_steps, 1);
    }

    #[test]
    fn solve_ode_with_stats_counts_nan_blowup_as_min_step_clamped() {
        // A diverging RHS produces a non-finite err_norm. Pinned at min_dt the
        // step is force-accepted to guarantee progress; the clamp counter must
        // still flag it (`!(err_norm <= 1.0)` catches NaN, where `err_norm > 1.0`
        // would not), so a NaN-diverging integration is not mistaken for clean.
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            min_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let _ = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, 1);
        assert_eq!(stats.accepted_steps, 1);
        assert_eq!(stats.rejected_steps, 0);
        assert_eq!(stats.min_step_clamped_steps, 1);
    }

    #[test]
    fn solve_ode_breaks_repeated_min_step_clamps_before_max_steps() {
        let rhs = |_u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = f64::NAN;
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-6,
            min_dt: 1e-6,
            max_steps: 10_000,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(
            stats.min_step_clamped_steps,
            MAX_CONSECUTIVE_MIN_STEP_CLAMPS
        );
    }

    /// Regression for #603 review #4: a *finite-but-stiff* min-step clamp (large finite local
    /// error pinned at `min_dt`) must NOT trip the divergence break — only non-finite errors
    /// do. Otherwise the remaining save points get frozen-padded with finite-but-wrong values
    /// the likelihood would silently accept. High-frequency finite forcing keeps every step
    /// clamped with a finite error, so the solve runs its full `max_steps` budget (>> the
    /// clamp limit) and returns finite predictions rather than breaking at the limit.
    #[test]
    fn solve_ode_does_not_break_on_finite_stiff_min_step_clamps() {
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        // Ran the full budget — no early break at MAX_CONSECUTIVE_MIN_STEP_CLAMPS …
        assert_eq!(stats.attempted_steps, opts.max_steps);
        assert!(stats.attempted_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        // … the segment really was in the stiff min-step-clamp regime …
        assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        // … and the predictions stayed finite (not NaN-padded).
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].is_finite());
    }

    /// #708 / #1080 Part B: with a budget set, the same stability-limited segment stops at the
    /// budget instead of grinding out its whole `max_steps` allowance.
    ///
    /// Shares the forcing and options of
    /// [`solve_ode_does_not_break_on_finite_stiff_min_step_clamps`] — a segment that clamps on
    /// every step and runs the full 200 — so the only difference measured is the budget.
    #[test]
    fn stiff_abort_after_stops_a_clamping_segment_at_the_budget() {
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let base = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let budget = 5;
        let opts = OdeSolverOptions {
            stiff_abort_after: Some(budget),
            ..base
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.min_step_clamped_steps, budget as usize);
        assert_eq!(stats.attempted_steps, budget as usize);
        assert_eq!(stats.stiff_aborted_segments, 1);
        // The tail is freeze-padded, not dropped: the `saveat` contract holds, and the value
        // is finite — the same shape the unrecoverable-clamp break already produced.
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].is_finite());

        // …and the un-budgeted run is untouched: full `max_steps`, no abort recorded.
        let mut ground_out = OdeSolverStats::default();
        solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &base,
            Some(&mut ground_out),
        );
        assert_eq!(ground_out.attempted_steps, base.max_steps);
        assert_eq!(ground_out.stiff_aborted_segments, 0);
    }

    /// A budget wider than the segment's clamp count is a no-op — the abort is a ceiling on
    /// cost, not a second convergence criterion.
    #[test]
    fn stiff_abort_after_above_the_clamp_count_changes_nothing() {
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            stiff_abort_after: Some(10_000),
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );
        assert_eq!(stats.attempted_steps, opts.max_steps);
        assert_eq!(stats.stiff_aborted_segments, 0);
    }

    /// A healthy segment never clamps, so no budget — however small — may touch it.
    #[test]
    fn stiff_abort_after_leaves_a_clean_solve_bit_identical() {
        let k = 0.1;
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -k * u[0];
        let saveat = [1.0, 5.0, 10.0];
        let base = OdeSolverOptions::default();
        let budgeted = OdeSolverOptions {
            stiff_abort_after: Some(1),
            ..base
        };
        let a = solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &saveat, &base);
        let b = solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &saveat, &budgeted);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.u[0].to_bits(), y.u[0].to_bits());
        }
    }

    /// The budget must be read from `.val()` like every other driver decision, so the `f64`
    /// prediction and the `Dual2` sensitivity solve abort on the *same* step — otherwise the
    /// gradient would differentiate a trajectory the predictor never produced.
    #[test]
    fn stiff_abort_after_aborts_identically_on_the_dual_path() {
        use crate::sens::dual2::Dual2;
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            stiff_abort_after: Some(5),
            ..OdeSolverOptions::default()
        };
        let rhs_f64 = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let rhs_dual = |_u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
            du[0] = Dual2::constant(1e6 * (t * 1e7).sin());
        };
        let mut scalar = OdeSolverStats::default();
        solve_ode_with_stats(
            &rhs_f64,
            &[1.0],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut scalar),
        );
        let mut dual = OdeSolverStats::default();
        solve_ode_g_with_stats(
            &rhs_dual,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut dual),
        );
        assert_eq!(scalar, dual);
        assert_eq!(dual.stiff_aborted_segments, 1);
    }

    /// The threshold driver bails with a named `Failed` rather than freeze-padding: its output
    /// *is* the answer (an event time, or a censoring), so a padded crossing would be laundered
    /// into the likelihood as though it had been integrated.
    #[test]
    fn stiff_abort_after_fails_the_threshold_driver_instead_of_padding() {
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            // Monotone (a hazard accumulator) but violently forced, so every step clamps.
            du[0] = 1e6 * (1.0 + (t * 1e7).sin());
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            method: OdeMethod::Rk45,
            stiff_abort_after: Some(3),
            ..OdeSolverOptions::default()
        };
        let mut u = [0.0];
        let outcome =
            solve_ode_until_threshold(&rhs, &mut u, (0.0, 1.0), &[], &opts, 0, f64::INFINITY);
        match outcome {
            ThresholdCrossing::Failed(msg) => {
                assert!(msg.contains("ode_stiff_abort_after"), "{msg}");
            }
            other => panic!("expected an abort, got {other:?}"),
        }
    }

    /// The stats scope collects every integration on the thread, nests, and leaves nothing
    /// behind — it is what the post-fit pass reads instead of threading a sink through every
    /// predictor (#1080 Part B item 2).
    #[test]
    fn the_stats_scope_collects_and_restores() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.1 * u[0];
        let opts = OdeSolverOptions::default();
        let solve = || {
            solve_ode(&rhs, &[1.0], (0.0, 10.0), &[], &[10.0], &opts);
        };

        // No scope: nothing is recorded and nothing is left set.
        solve();
        assert!(!stats_sink_active());

        let outer = SolverStatsScope::enter();
        solve();
        let after_one = outer.collected();
        assert!(after_one.attempted_steps > 0);
        {
            let inner = SolverStatsScope::enter();
            solve();
            // The inner scope starts empty rather than inheriting the outer total …
            assert!(inner.collected().attempted_steps > 0);
            assert!(inner.collected().attempted_steps < after_one.attempted_steps * 2);
        }
        // … and dropping it restores the outer one, whose own total is unchanged by the
        // integrations that ran while it was shadowed.
        assert_eq!(outer.collected(), after_one);
        solve();
        assert_eq!(
            outer.collected().attempted_steps,
            after_one.attempted_steps * 2
        );
        drop(outer);
        assert!(!stats_sink_active());
    }

    /// A caller that passes its own `&mut OdeSolverStats` keeps getting exactly that, scope or
    /// no scope: the sink is a tee, not a redirect.
    #[test]
    fn the_stats_scope_does_not_steal_a_callers_counters() {
        let rhs = |u: &[f64], _p: &[f64], _t: f64, du: &mut [f64]| du[0] = -0.1 * u[0];
        let opts = OdeSolverOptions::default();
        let mut without = OdeSolverStats::default();
        solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 10.0),
            &[],
            &[10.0],
            &opts,
            Some(&mut without),
        );

        let scope = SolverStatsScope::enter();
        let mut with = OdeSolverStats::default();
        solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 10.0),
            &[],
            &[10.0],
            &opts,
            Some(&mut with),
        );
        assert_eq!(with, without);
        assert_eq!(scope.collected(), without);
    }

    /// Review follow-up (#1080): the budget must count *abandoned* segments only. A segment
    /// whose final step clamps still reaches the end of its span with every `saveat` saved —
    /// breaking out there abandons nothing, and reporting it as truncated would make the
    /// counter useless for the one question it exists to answer.
    #[test]
    fn a_clamp_on_the_last_step_is_not_an_abort() {
        // Forcing violent enough to clamp every step, over a span that `min_dt` steps can
        // still cross: 5 steps of 1e-3 reach `tf`, so the fifth clamp lands exactly on the
        // end of the segment.
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (t * 1e7).sin();
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            stiff_abort_after: Some(5),
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_with_stats(
            &rhs,
            &[1.0],
            (0.0, 5e-3),
            &[],
            &[5e-3],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.min_step_clamped_steps, 5, "the fixture must clamp");
        assert_eq!(
            stats.stiff_aborted_segments, 0,
            "the segment reached t_end — nothing was abandoned"
        );
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].is_finite());
    }

    /// Review follow-up (#1080): a clamped step can still be the step that brackets the
    /// crossing. The budget bounds cost; throwing away an answer the solver has already found
    /// is not a cost saving, it is a worse outcome than the one being avoided.
    #[test]
    fn the_threshold_driver_keeps_a_crossing_found_on_a_clamped_step() {
        // Monotone and violently forced, so the first step clamps — and with a threshold this
        // low, that same first step crosses it.
        let rhs = |_u: &[f64], _p: &[f64], t: f64, du: &mut [f64]| {
            du[0] = 1e6 * (1.0 + (t * 1e7).sin());
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            method: OdeMethod::Rk45,
            stiff_abort_after: Some(1),
            ..OdeSolverOptions::default()
        };
        let mut u = [0.0];
        match solve_ode_until_threshold(&rhs, &mut u, (0.0, 1.0), &[], &opts, 0, 1.0) {
            ThresholdCrossing::Crossed(t) => {
                assert!(t > 0.0 && t <= 1e-3, "crossing outside the first step: {t}");
            }
            other => panic!("the crossing must survive the budget, got {other:?}"),
        }
    }
    /// Dual path mirrors [`solve_ode_does_not_break_on_finite_stiff_min_step_clamps`].
    #[test]
    fn solve_ode_g_does_not_break_on_finite_stiff_min_step_clamps() {
        use crate::sens::dual2::Dual2;
        let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], t: f64, du: &mut [Dual2<1>]| {
            du[0] = Dual2::constant(1e6 * (t * 1e7).sin());
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-3,
            min_dt: 1e-3,
            max_steps: 200,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_g_with_stats(
            &rhs,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(stats.attempted_steps, opts.max_steps);
        assert!(stats.min_step_clamped_steps > MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(result.len(), 1);
        assert!(result[0].u[0].value.is_finite());
    }

    #[test]
    fn solve_ode_g_breaks_repeated_min_step_clamps_like_scalar() {
        use crate::sens::dual2::Dual2;
        let rhs = |_u: &[Dual2<1>], _p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = Dual2::constant(f64::NAN);
        };
        let opts = OdeSolverOptions {
            initial_dt: 1e-6,
            min_dt: 1e-6,
            max_steps: 10_000,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let mut stats = OdeSolverStats::default();
        let result = solve_ode_g_with_stats(
            &rhs,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[],
            &[1.0],
            &opts,
            Some(&mut stats),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(stats.attempted_steps, MAX_CONSECUTIVE_MIN_STEP_CLAMPS);
        assert_eq!(
            stats.min_step_clamped_steps,
            MAX_CONSECUTIVE_MIN_STEP_CLAMPS
        );
    }

    #[test]
    fn solve_ode_g_with_stats_matches_scalar_step_pattern() {
        use crate::sens::dual2::Dual2;
        let rhs_f = |u: &[f64], p: &[f64], _t: f64, du: &mut [f64]| {
            du[0] = -p[0] * u[0];
        };
        let rhs_d = |u: &[Dual2<1>], p: &[Dual2<1>], _t: f64, du: &mut [Dual2<1>]| {
            du[0] = -(p[0] * u[0]);
        };
        let opts = OdeSolverOptions {
            initial_dt: 1.0,
            abstol: 1e-12,
            reltol: 1e-12,
            ..OdeSolverOptions::default()
        };
        let saveat = [1.0];
        let mut stats_f = OdeSolverStats::default();
        let mut stats_d = OdeSolverStats::default();

        let _ = solve_ode_with_stats(
            &rhs_f,
            &[1.0],
            (0.0, 1.0),
            &[10.0],
            &saveat,
            &opts,
            Some(&mut stats_f),
        );
        let _ = solve_ode_g_with_stats(
            &rhs_d,
            &[Dual2::<1>::constant(1.0)],
            (0.0, 1.0),
            &[Dual2::<1>::var(10.0, 0)],
            &saveat,
            &opts,
            Some(&mut stats_d),
        );

        assert_eq!(stats_d, stats_f);
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
