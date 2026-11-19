//! Adaptive ODE integration: the `Stepper` abstraction, the explicit Dormand-Prince RK45
//! stepper (the default method), and the drivers every consumer integrates through.
//!
//! RK45 is the same family as Julia's `Tsit5()` — a 5th-order explicit Runge-Kutta method with
//! an embedded 4th-order error estimate for adaptive step control, optimized here for PK ODE
//! systems (2–20 states, smooth dynamics). The stiff [`OdeMethod`] alternatives live in
//! `super::rosenbrock`.
//!
//! # Shape of the module
//!
//! A **method** implements `Stepper`: attempt a step, score it, and evaluate its own
//! continuous extension inside it. A **driver** owns everything else — the step-size
//! controller, the `saveat` contract, the `min_dt` force-accept, the divergence break, soft
//! sampling, event root-finding, statistics — and is written once, generically, over both the
//! stepper and the state scalar `T: PkNum`.
//!
//! That split is what keeps the methods at parity: a consumer written against a driver gets
//! every method, and a method implementing `Stepper` gets every consumer. The two live
//! drivers are `integrate_dense_g` (saves + in-step reads; behind [`solve_ode`],
//! `solve_ode_dense` and [`solve_ode_g`]) and [`solve_ode_until_threshold`] (event-time
//! root-finding). `make_stepper` is the single place a method name selects an implementation.

use crate::sens::num::NonRecRead;
use crate::sens::num::PkNum;

/// Dormand-Prince RK45 coefficients (Butcher tableau)
pub const A2: f64 = 1.0 / 5.0;
pub const A3: f64 = 3.0 / 10.0;
pub const A4: f64 = 4.0 / 5.0;
pub const A5: f64 = 8.0 / 9.0;
// a6 = 1.0, a7 = 1.0

pub const B21: f64 = 1.0 / 5.0;
pub const B31: f64 = 3.0 / 40.0;
pub const B32: f64 = 9.0 / 40.0;
pub const B41: f64 = 44.0 / 45.0;
pub const B42: f64 = -56.0 / 15.0;
pub const B43: f64 = 32.0 / 9.0;
pub const B51: f64 = 19372.0 / 6561.0;
pub const B52: f64 = -25360.0 / 2187.0;
pub const B53: f64 = 64448.0 / 6561.0;
pub const B54: f64 = -212.0 / 729.0;
pub const B61: f64 = 9017.0 / 3168.0;
pub const B62: f64 = -355.0 / 33.0;
pub const B63: f64 = 46732.0 / 5247.0;
pub const B64: f64 = 49.0 / 176.0;
pub const B65: f64 = -5103.0 / 18656.0;
pub const B71: f64 = 35.0 / 384.0;
pub const B73: f64 = 500.0 / 1113.0;
pub const B74: f64 = 125.0 / 192.0;
pub const B75: f64 = -2187.0 / 6784.0;
pub const B76: f64 = 11.0 / 84.0;

// Error coefficients (5th order - 4th order)
pub const E1: f64 = 71.0 / 57600.0;
pub const E3: f64 = -71.0 / 16695.0;
pub const E4: f64 = 71.0 / 1920.0;
pub const E5: f64 = -17253.0 / 339200.0;
pub const E6: f64 = 22.0 / 525.0;
pub const E7: f64 = -1.0 / 40.0;

/// Consecutive force-accepted minimum-step attempts with a **non-finite** local error
/// before an integration segment is treated as unrecoverably pathological.
///
/// The break is gated on a non-finite error norm (NaN/∞) on purpose: that is the signature
/// of a diverging trajectory. It used to be said that the likelihood would reject such a
/// trajectory's padded-out predictions anyway, which holds only when the observed state is
/// the one that diverged — a runaway state *decoupled* from the observed one left the
/// observed compartment frozen at a finite value that was served, fitted and verified as a
/// real prediction (#1539). So a stop on a non-finite state now pads its tail with `NaN` in
/// every component (`integrate_dense_g`, [`OdeSolverStats::diverged_segments`]).
/// A *finite* error above tolerance at `min_dt` is merely a stiff or under-resolved segment —
/// truncating it would freeze-pad the remaining save points with finite (but wrong) values
/// that the likelihood would silently accept, so those are left to run to `max_steps` as
/// before (#603 review #4).
pub const MAX_CONSECUTIVE_MIN_STEP_CLAMPS: usize = 64;

/// Accepted steps between mid-segment stiffness re-probes under [`OdeMethod::Auto`] (#1080 Part C).
///
/// The a-priori probe reads the Jacobian at a segment's *entry* state, the one state a binding
/// model looks least stiff at (both factors of a `KON · C · R` term can be zero at dose time and
/// large an hour later). Re-probing along the way lets the driver notice; the interval is the
/// cost/latency trade:
///
/// * **Cost.** A probe is `n + 1` RHS evaluations plus an `O(n²)` Gershgorin bound (the eigensolve
///   only runs when that bound cannot rule stiffness out, #1080 Part B): ~1% of the segment
///   amortized over 25 accepted steps. A long benign segment is re-probed, the verdict never
///   changes, and its trajectory is bit-identical to the pinned one; a segment shorter than 25
///   accepted steps (the overwhelming majority) is never re-probed and pays nothing.
/// * **Latency.** A segment that has turned stiff takes hundreds to thousands of steps before it
///   exhausts `max_steps`, so 25 steps of delay is noise.
///
/// **The switch is one-way** (#2845): a segment may escalate from the explicit stepper to a stiff
/// one part-way through but is never demoted back, so it changes stepper at most once. Demotion on a
/// single probe reading made `auto` carry `rk45`'s error and jitter on a TMDD model where the stiff
/// steppers are right: the probe read the relaxing tail as non-stiff, swapped to `rk45`, and
/// integrated the rest of the segment at explicit accuracy, at a step chosen by a probe counter that
/// runs across segments, so *where* it landed moved with θ and the objective jittered by a third to
/// a half of the outer optimizer's plateau window. On that fixture (`ode_reltol = 1e-6`, macOS
/// arm64, #2845) blocking only the demotion took the OFV error against a `vern7` reference from
/// -4.7e-3 to -1e-4, the same as a pinned `rodas4`. A stiff stepper integrates a non-stiff tail
/// correctly at a higher per-step cost, and the *next* segment is probed afresh, so the cost is
/// bounded by one segment. The one-way rule also replaces the per-segment switch cap #1080 Part C
/// carried against a trajectory hovering at the threshold (nothing to oscillate between).
pub const AUTO_SWITCH_PROBE_INTERVAL: usize = 25;

/// Dormand-Prince 5(4)'s real-axis stability boundary, used by Hairer's free
/// stage-based stiffness estimate. The estimate is dimensionless (`h * rho`),
/// so unlike the segment-entry Jacobian threshold it is invariant to the
/// model's choice of time unit.
pub const RK45_STIFFNESS_THRESHOLD: f64 = 3.25;

/// Consecutive stiff stage estimates required before replacing RK45. Hairer's
/// DOPRI5 detector uses the same ratification count; carrying it across event
/// intervals is important because a typical PK interval contains fewer steps.
pub const RK45_STIFFNESS_RATIFY: usize = 15;

/// Non-stiff estimates required to clear a partially accumulated stiff verdict.
pub const RK45_NONSTIFF_RESET: usize = 6;

/// State belonging to one logical subject solve rather than one event interval.
///
/// Event handling must rebuild the stepper because a dose or forcing change
/// invalidates its stages. These counters remain meaningful across that boundary:
/// they summarize accepted RK45 steps, exactly the work the detector reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct OdeAutoSwitchState {
    accepted_since_probe: usize,
    pub rk_stiff: usize,
    rk_nonstiff: usize,
}

impl OdeAutoSwitchState {
    pub fn observe_rk45(&mut self, indicator: f64) -> bool {
        if !indicator.is_nan() && indicator > RK45_STIFFNESS_THRESHOLD {
            self.rk_stiff += 1;
            self.rk_nonstiff = 0;
        } else {
            self.rk_nonstiff += 1;
            if self.rk_nonstiff >= RK45_NONSTIFF_RESET {
                self.rk_stiff = 0;
                self.rk_nonstiff = 0;
            }
        }
        self.rk_stiff >= RK45_STIFFNESS_RATIFY
    }

    pub fn reset_rk45_verdict(&mut self) {
        self.rk_stiff = 0;
        self.rk_nonstiff = 0;
    }
}

/// Step-shrink factor applied when the local error estimate is non-finite (NaN/∞). The
/// trajectory is diverging, so shrink toward `min_dt` instead of falling into the
/// I-controller's growth branch, which would otherwise enlarge the step on a NaN error.
pub const NONFINITE_ERR_SHRINK_FACTOR: f64 = 0.2;

/// ODE right-hand side function type.
/// `rhs(u, params, t) -> du/dt`  where u and du are `&[f64]` of length n_states.
pub type OdeRhsFn = Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>;

/// Which stepper integrates the `[odes]` block (`[fit_options] ode_method`).
///
/// The default is the explicit [`Rk45`](OdeMethod::Rk45); the three Rosenbrock entries are
/// **linearly implicit** and exist for stiff systems (fast binding / TMDD-style quasi-equilibrium,
/// Michaelis-Menten with `Km ≪ C`, long transit chains, QSP cascades) where an explicit method is
/// stability-limited and grinds down to `min_dt` (visible as
/// [`OdeSolverStats::min_step_clamped_steps`]) instead of being accuracy-limited.
///
/// Cost per step differs sharply: RK45 takes 6 `f` evaluations (FSAL); a Rosenbrock step adds the
/// exact `n × n` Jacobian and `∂f/∂t` (differenced, `n + 1` extra evaluations, only for a caller's
/// own closure through [`solve_ode`]) plus an `O(n³)` factorization. On a non-stiff 1-3 compartment
/// model RK45 stays faster; the Rosenbrock methods win only where stiffness, not accuracy, caps the
/// step.
///
/// `n` is the size of the system being integrated, **not** always the `[odes]` state count. A CTMM
/// endpoint integrates the occupancy system `dP/dt = P·Q(t)`
/// ([`crate::markov::ctmm_inhomogeneous_transition_with_opts`]), whose state is the flattened
/// `s × s` transition matrix: an `s`-state chain is an `s²`-state integration and a Rosenbrock method
/// builds an `s² × s²` Jacobian per step attempt. Right for a chain with widely separated rates (a
/// genuinely stiff occupancy solve), pure overhead for the usual well-scaled chain, so choose
/// `ode_method` on a CTMM model with the `s²` cost in mind.
///
/// Honoured by every `saveat` integration (predictions, the FOCE/FOCEI objective, steady-state
/// equilibration, the analytic-sensitivity walk), dispatched centrally in `solve_ode_dense` (the
/// sole owner of the f64 stepping loop) and [`solve_ode_g`]. Every method is a full peer: each
/// implements `Stepper`, including its own continuous extension, and every consumer is written once
/// against that trait, so dense saves, in-step soft sampling (joint PK-TTE cumulative hazard, CTMM
/// occupancy, adaptive-dosing monitors), event-time root-finding and the analytic-sensitivity path
/// work for all methods with no per-method wiring. Serialised as its canonical
/// [`as_str`](OdeMethod::as_str) token, the spelling a `.fitrx` bundle records (#1758).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
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
    #[serde(rename = "rodas5p")]
    Rodas5P,
    /// Verner 7(6): 10-stage explicit, order 7. **Not** a stiff method — the high-order option
    /// for a fit that is accuracy-limited rather than stability-limited, where step count
    /// scales as `tol^(−1/p)` and order is what pays. Worth it at tight tolerances
    /// (`ode_reltol ≤ 1e-8`); at loose ones RK45's cheaper steps win. See
    /// `crate::ode::explicit_rk`.
    Vern7,
    /// Pick the stepper from the system itself: an a-priori Jacobian eigenvalue probe reads `|Re λ|max`
    /// at each integration's own initial state and starts a stiff method only where the system is
    /// stability-limited, keeping the explicit default elsewhere. See [`crate::ode::stiffness`] for what
    /// is measured, where, and why an unclassifiable system stays explicit.
    ///
    /// This is the *starting* decision for each integrated segment. By default
    /// [`OdeSolverOptions::auto_switch`] re-probes periodically and may replace the active stepper
    /// between accepted steps while retaining the state integrated so far; disabling it keeps the
    /// starting method fixed for the segment. Both modes preserve dense output, event root-finding and
    /// `Dual2` sensitivities.
    ///
    /// **The default.** Whether a system is stiff is a property of the equations, and a user who must
    /// know the answer in advance to get a tractable fit is being asked the wrong question. Every
    /// non-stiff model keeps [`Rk45`](OdeMethod::Rk45) and pays one Jacobian per segment; every stiff
    /// one stops being stability-limited without anyone noticing.
    #[default]
    Auto,
    /// Exact propagation of a **linear time-invariant** right-hand side by the matrix exponential: no
    /// stepping, no tolerance, no stability limit (#852; NONMEM's ADVAN5). The propagator lives in
    /// `crate::ode::linear`.
    ///
    /// Not a stepper: each segment's `du/dt = A·u + b` is read off the RHS and advanced as
    /// `exp([[A, b], [0, 0]]·h)`, exact to roundoff. A model gets it in one of two ways: the parser
    /// selects it for [`Auto`](OdeMethod::Auto) when its `[odes]` block is *structurally* affine in the
    /// states, reads no model time and carries no absorption forcing; or the user names
    /// `ode_method = linear`, which asserts that and is refused on a model that is not (a parse error
    /// from the model file, `E_ODE_METHOD_LINEAR` from a caller's `FitOptions`). Any named integrator
    /// opts a linear model out.
    ///
    /// Only a parser-certified spec takes the exact path. The public `solve_ode*` functions take an
    /// arbitrary closure and integrate `Linear` as `Auto` (the self-check samples finitely many points,
    /// so it cannot certify a closure). A certified segment it still declines (an overflowing
    /// exponential) is integrated as `Auto` would and counted in
    /// [`OdeSolverStats::linear_declined_segments`].
    Linear,
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
            "linear" => Some(OdeMethod::Linear),
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
            OdeMethod::Linear => "linear",
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
    /// Give up on a segment once this many of its steps have clamped at `min_dt`, instead of grinding on
    /// to [`max_steps`](Self::max_steps) (#708, #1080 Part B).
    ///
    /// A clamp is the solver saying "I cannot meet tolerance here and am advancing anyway"; a segment
    /// with a handful of them is stability-limited and the remaining thousands of steps buy only wall
    /// time. Aborting freeze-pads the segment's tail as the existing unrecoverable-clamp break does (no
    /// more wrong than the ground-out one, only cheaper) and bumps
    /// [`OdeSolverStats::stiff_aborted_segments`] so the fit's solver-diagnostics warning can say so.
    ///
    /// `None` (the default) grinds to `max_steps`, the pre-#1080 behaviour bit for bit. Deliberately
    /// **not** on by default: an early abort trades a slow-but-eventually-integrated segment for a
    /// freeze-padded one, a change to the objective, not only its cost.
    ///
    /// Independent of [`OdeMethod::Auto`]: under `auto` an escalation that trips the budget is discarded
    /// by the guard like any other stalled escalation (the trigger is
    /// [`OdeSolverStats::stiff_min_step_clamped_steps`]` > 0`, which an abort on a stiff stepper has by
    /// construction) and the segment is re-solved explicitly. The budget is per *method*, not per
    /// segment: when [`auto_switch`](Self::auto_switch) swaps the stepper part-way the count restarts,
    /// since it bounds how long one method may grind on a segment it cannot step, and the clamps behind
    /// the switch measured the replaced method.
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
    /// `AUTO_SWITCH_PROBE_INTERVAL` accepted steps, and the switch happens in place: the
    /// segment keeps the state it has integrated so far and carries on with the other stepper,
    /// rather than restarting or re-solving. Setting this to `false` restores the pre-#1080
    /// Part C behaviour — one method per segment, chosen at its start.
    pub auto_switch: bool,
}

impl OdeSolverOptions {
    /// These options as a *stepping* driver should read them: [`OdeMethod::Linear`] becomes
    /// [`OdeMethod::Auto`], everything else is unchanged.
    ///
    /// `Linear` names an exact propagator, not a stepper, so wherever a segment has to be
    /// integrated after all — the exact path declined it, or a driver (the threshold search)
    /// has no exact counterpart — it is integrated exactly as the default would integrate it,
    /// stiffness probe and escalation guard included.
    pub(crate) fn for_integration(&self) -> Self {
        match self.method {
            OdeMethod::Linear => Self { method: OdeMethod::Auto, ..*self },
            _ => *self,
        }
    }
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

// ── Fit-scoped solver-option override (#1212) ───────────────────────────────────────────
//
// `OdeSpec::solver_opts` is stamped once, at parse time, by
// `CompiledModel::sync_ode_solver_opts`, and every integration entry point reads the value off
// the spec rather than off `FitOptions`. `fit` takes `&CompiledModel`, so it cannot restamp the
// spec — which is why, before #1212, a caller-supplied `FitOptions::ode_reltol` / `ode_method` /
// … reached nothing at all: the fit ran at the parse-time value and reported success. The
// override below carries those fields the last hop.

/// A per-field override of a spec's baked [`OdeSolverOptions`], armed for the duration of one
/// [`fit`](crate::fit) (#1212).
///
/// Only the fields a caller actually moved are `Some`. `FitOptions::ode_solver_override` builds
/// this by comparing against `FitOptions::default()`, whose `ode_*` values are identical to
/// [`OdeSolverOptions::default`]'s by construction (pinned by
/// `fit_option_ode_defaults_match_the_solver_defaults`), which is what makes the merge
/// one-directional: a caller who never touched a field cannot silently *loosen* a model file
/// that pinned it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OdeSolverOverride {
    pub reltol: Option<f64>,
    pub abstol: Option<f64>,
    pub max_steps: Option<usize>,
    pub method: Option<OdeMethod>,
    /// Doubly wrapped deliberately: the outer `Option` is "the caller moved this field", the
    /// inner one is the field's own `None` (no abort budget). While the default is `None` the
    /// inner `None` is unreachable from `FitOptions::ode_solver_override` — "moved" and "set to
    /// something" coincide — but collapsing the two would silently turn a future non-`None`
    /// default into a field that could no longer be cleared.
    pub stiff_abort_after: Option<Option<u32>>,
    pub auto_switch: Option<bool>,
}

impl OdeSolverOverride {
    /// True when no field was moved, i.e. arming this would be a no-op.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Reject values that the model-file parser would reject before they can
    /// reach a worker pool through the public Rust API.
    ///
    /// The domain is the one the `.fitrx` reader holds a recorded `ScoringSettings` to
    /// (`population_check::ode_tol_in_domain` and siblings), so a fit that runs records a bundle
    /// that loads (#1770). `ode_stiff_abort_after = Some(0)` is refused with them: the
    /// parser reads `0` as "off", and a run on `Some(0)` would record a value no model
    /// file can spell.
    pub fn validate(&self) -> Result<(), String> {
        use crate::estimation::population_check::{
            ode_max_steps_in_domain, ode_stiff_abort_after_in_domain, ode_tol_in_domain,
        };
        let check = || {
            for (name, value) in [("ode_reltol", self.reltol), ("ode_abstol", self.abstol)] {
                value.map_or(Ok(()), |v| ode_tol_in_domain(name, v))?;
            }
            self.max_steps.map_or(Ok(()), ode_max_steps_in_domain)?;
            self.stiff_abort_after.map_or(Ok(()), ode_stiff_abort_after_in_domain)
        };
        check().map_err(|e| e.to_string())
    }

    /// Cache-key equality uses the float representation explicitly. Validation
    /// keeps NaNs and signed zero out, while bit comparison makes the key's
    /// behavior independent of `f64`'s partial equality rules.
    pub fn same_pool_key(&self, other: &Self) -> bool {
        fn bits(value: Option<f64>) -> Option<u64> {
            value.map(f64::to_bits)
        }
        bits(self.reltol) == bits(other.reltol)
            && bits(self.abstol) == bits(other.abstol)
            && self.max_steps == other.max_steps
            && self.method == other.method
            && self.stiff_abort_after == other.stiff_abort_after
            && self.auto_switch == other.auto_switch
    }

    /// Merge onto a spec's baked options. Unset fields keep the baked value, so a model file's
    /// `[fit_options]` still wins wherever the caller expressed no opinion.
    pub fn apply_to(&self, base: OdeSolverOptions) -> OdeSolverOptions {
        OdeSolverOptions {
            reltol: self.reltol.unwrap_or(base.reltol),
            abstol: self.abstol.unwrap_or(base.abstol),
            max_steps: self.max_steps.unwrap_or(base.max_steps),
            method: self.method.unwrap_or(base.method),
            stiff_abort_after: self.stiff_abort_after.unwrap_or(base.stiff_abort_after),
            auto_switch: self.auto_switch.unwrap_or(base.auto_switch),
            ..base
        }
    }
}

// ── Where the armed override lives: one thread-local, and a pool that carries it ──
//
// The override is *per fit*, and a fit is not one thread: the inner loop fans out over subjects
// with rayon, so the workers doing the integrating must see it too. Two other shapes are wrong.
//
// A single process-global slot (the inner-loop settings' shape until #426) cannot express "this
// fit's options": `ferx-tools` runs `fit()` from a `par_iter` over replicates and `pool`
// supports concurrent callers, so a second fit's arming overwrites the first's, and a fit that
// expressed *no* opinion (almost every fit) pushes the slot back to "nothing armed" while another
// fit still runs. Both directions silently return a number at an accuracy nobody asked for (#1212).
//
// A thread-local alone fixes the arming thread and leaves every worker reading whatever some other
// fit published: the same defect one thread over, and *not* bounded in the safe direction
// (`ode_reltol` takes any positive value, a method-only override replaces no tolerance, and
// `ode_max_steps` / `ode_stiff_abort_after` have no tighter/coarser ordering to fall back on).
//
// So the override is a thread-local, and a fit that has one **runs on a pool whose workers were
// started with that same value already installed** (`pool::fit_scope_pool`, keyed by
// `pool::FitScope`). Such a worker serves only fits carrying that override, so "which fit am
// I serving?" has one answer per thread. A fit with no override uses the shared pool, whose
// workers never install one and integrate at the spec's baked options.

thread_local! {
    /// The override in force on this thread: `None` outside a fit (and on every worker of the
    /// shared pool), `Some(None)` for a fit that expressed no opinion, `Some(Some(ov))`
    /// otherwise.
    ///
    /// On a thread that arms, [`OdeSolverOverrideGuard`] restores the previous value on drop —
    /// correct here in a way it is not for a process-global, since arm and drop are the same
    /// thread's and the nesting really is LIFO. On a worker of an override pool the value is
    /// installed once at thread start and never changes.
    static LOCAL_OVERRIDE: std::cell::Cell<Option<Option<OdeSolverOverride>>> =
        const { std::cell::Cell::new(None) };
}

/// The options an integration should actually run at: `baked` with any fit-scoped override
/// applied. Read through [`OdeSpec::effective_solver_opts`](crate::ode::predictions::OdeSpec),
/// never by touching `OdeSpec::solver_opts` directly.
pub fn effective_solver_options(baked: OdeSolverOptions) -> OdeSolverOptions {
    match LOCAL_OVERRIDE.get().flatten() {
        Some(ov) => ov.apply_to(baked),
        None => baked,
    }
}

/// The override in force on this thread (see `LOCAL_OVERRIDE`): what a compiled recording's
/// ODE segment was recorded under, and must run under (#2786).
///
/// Canonical: "outside a fit" (`None`), "a fit with no opinion" (`Some(None)`) and an override
/// with no field moved all integrate at the baked options, so all three read `None` here. A
/// plan recorded on one and run on another (a shared-pool worker against the thread that
/// armed an empty override) is the same plan, and must not be refused for it.
pub fn current_ode_override() -> Option<OdeSolverOverride> {
    LOCAL_OVERRIDE.get().flatten().filter(|ov| !ov.is_empty())
}

pub fn install_worker_ode_override(ov: OdeSolverOverride) {
    LOCAL_OVERRIDE.set(Some(Some(ov)));
}

/// Arms `ov` on this thread until the returned guard drops, restoring whatever was in force
/// before — so a nested fit's opinion wins while it runs and hands the setting back on exit,
/// and an early `?` return out of the fit cannot leak the override into a later `predict`.
///
/// Arming alone reaches only this thread. A caller whose work fans out over rayon must run
/// that work through `pool::with_fit_scope`, which also puts it on a pool
/// whose workers carry the same value.
#[must_use = "the override is disarmed as soon as this guard drops"]
pub struct OdeSolverOverrideGuard {
    prev: Option<Option<OdeSolverOverride>>,
}

/// See [`OdeSolverOverrideGuard`]. An empty override arms `Some(None)`: "this fit expressed no
/// opinion" has to mean the spec's own baked value, not the enclosing fit's tolerance.
pub fn arm_ode_solver_override(ov: OdeSolverOverride) -> OdeSolverOverrideGuard {
    let prev = LOCAL_OVERRIDE.replace(Some((!ov.is_empty()).then_some(ov)));
    OdeSolverOverrideGuard { prev }
}

impl Drop for OdeSolverOverrideGuard {
    fn drop(&mut self) {
        LOCAL_OVERRIDE.set(self.prev);
    }
}

/// Scale-aware tolerance band `abstol + reltol * max(|a|, |b|)`. Shared by the RK45
/// step-error control, the per-step monotonicity guard (`mono_tol`), and the TTE
/// cumulative-hazard monotonicity floor (`survival::MonoTol`), so the integrator and the
/// survival guard use one definition and cannot silently diverge (see #618).
#[inline]
pub fn scale_tol(abstol: f64, reltol: f64, a: f64, b: f64) -> f64 {
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
    /// The switch is one-way — explicit to stiff, never back (#2845) — so a segment switches at
    /// most once and this counts escalations reached mid-segment. A non-zero value says the segment's stiffness changed while it was being
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
    /// Of [`auto_stiff_rejected`](Self::auto_stiff_rejected), the escalations rejected
    /// **only** because their derivative jets were non-finite while every saved value was
    /// finite (#1204).
    ///
    /// Structurally different from every other counter here, and the difference is the point:
    /// it can only ever be non-zero on a **dual** instantiation, because `f64` carries no
    /// jets to check ([`crate::sens::num::PkNum::jets_finite`] is `true` for it). So a
    /// segment counted here was kept by the `f64` prediction solve and discarded by the
    /// `Dual2`/`Dual2` sensitivity solve — the one place the two deliberately part company.
    /// See the escalation guard for why that asymmetry is the right trade.
    ///
    /// Non-zero means the fit's *gradient* would otherwise have been `NaN`/`inf` on those
    /// segments while its predictions looked clean. It is a repair, not a loss, but it is
    /// also a reliable sign that the trajectory is running near the edge of double precision
    /// — check the model's scaling and units before trusting the estimates.
    pub auto_stiff_rejected_jets: usize,
    /// Rejected [`OdeMethod::Auto`] escalations whose pinned-explicit re-solve also failed:
    /// it stopped before the end of the segment, or returned a saved state whose value or
    /// (on a dual instantiation) whose derivative jets were non-finite.
    ///
    /// A subset of [`auto_stiff_rejected`](Self::auto_stiff_rejected). The guard has no third
    /// method to return in this case, so the caller still receives the explicit result; this
    /// counter is the honest "both attempts failed" signal that makes that otherwise-silent
    /// outcome visible in the fit's `ode_solver` warning (#1080).
    ///
    /// Like every other `auto` counter it is a floor, not a total: [`solve_ode_until_threshold`]
    /// takes no stats, so a failure on the event-time path (TTE / adaptive dosing) is not
    /// counted here and a zero is not by itself proof that every solve in the fit was clean.
    pub auto_fallback_failed: usize,
    /// Integration attempts that stopped before reaching the end of their segment and freeze-padded the
    /// remaining output times with the last state (or with `NaN` when the stop was on a state that had
    /// gone non-finite; those are also in [`diverged_segments`](Self::diverged_segments), #1539).
    ///
    /// This counts attempts, including stiff attempts the `auto` guard later discarded; subtract
    /// [`discarded_unfinished_segments`](Self::discarded_unfinished_segments) for the unfinished
    /// trajectories the caller received.
    ///
    /// A roll-up, not a disjoint category: every segment in
    /// [`stiff_aborted_segments`](Self::stiff_aborted_segments) is also counted here (the budgeted abort
    /// only fires while `t < tf`), as is every segment that stopped because its stepper could not form
    /// a step at `min_dt`. Do not add this to those counters; the fit's `ode_solver` warning subtracts
    /// the abandoned ones first so its clauses stay disjoint. What is left is dominated by segments
    /// that exhausted [`OdeSolverOptions::max_steps`]. A floor, not a total: [`solve_ode_until_threshold`]
    /// takes no stats, so event-time segments (TTE / adaptive dosing) are not counted.
    pub unfinished_segments: usize,
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
    /// reaches the end of its span is likewise not counted — nothing was abandoned. Nor is a
    /// budgeted stop on a state that had already gone non-finite: that segment's tail is `NaN`,
    /// not freeze-padded, and it is counted in
    /// [`diverged_segments`](Self::diverged_segments) instead (#1539).
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
    /// Of [`unfinished_segments`](Self::unfinished_segments), the attempts discarded by the
    /// `auto` escalation guard before it re-solved the segment explicitly.
    pub discarded_unfinished_segments: usize,
    /// Engine walks abandoned **before integrating** because their timeline could not be ordered: a
    /// `NaN`/`inf` dose time, lagtime, route lag or infusion duration
    /// (`ode::predictions::times_have_non_finite`, #1189).
    ///
    /// The odd one out, deliberately: every other counter reports work that *happened*, so an abandoned
    /// walk leaves all at zero, which is indistinguishable from a subject with nothing to integrate.
    /// Measured on one model in a `SolverStatsScope`, a normal subject read `attempted/accepted/rejected
    /// = 7/7/0` while a non-finite timeline, a subject with no records, and a subject with a single
    /// observation at `t = 0` all read `0/0/0`, returning `[NaN, NaN, NaN, NaN]`, `[]` and `[0.0]`
    /// (#1234). This counter separates the first from the other two.
    ///
    /// Counted **once per abandoned walk**, not per subject or segment. A post-fit sweep drives more than
    /// one engine over the same subject, so one bad subject contributes more than one (measured **3** on
    /// a plain two-subject FOCEI fit: `ode_predictions` twice, `ode_predictions_with_states` once). Zero
    /// on every well-formed fit, so any non-zero value means some subject's predictions are `NaN` by
    /// construction.
    ///
    /// Recorded on all eight `f64` engine sites through `ode::predictions::abandon_non_finite_timeline`
    /// (predicate and counter are one call, so a ninth guard cannot detect an unorderable timeline
    /// without reporting it; pinned by `the_bare_timeline_predicates_are_not_called_outside_this_guard`):
    /// the four dense / event-driven prediction walks, the adaptive driver and its frozen replay, the EKF
    /// walk, and `ode_solve_until_chz_threshold`'s event-time solve.
    ///
    /// **All eight fire in production as of #1304.** Before it `SolverStatsScope::enter` appeared outside
    /// tests only in `fit()`'s post-fit pass, so the adaptive driver, its replay and the CHZ solve
    /// (reached only under `simulate()` / `simulate_adaptive()`) recorded into an inactive sink.
    /// `api::predict_diag`, `api::simulate_with_options_diag` and `api::simulate_adaptive` now open a scope
    /// (through `api::postfit::solver_stats_scope`, the single gate), so this counter reaches a user on
    /// every entry point with a warnings channel. The CHZ site was never "the one silent hole": its sole
    /// production caller (`survival::draw_ode_tte_latent`) turns `SolveFailed` into a per-subject
    /// `W_TTE_SOLVE_FAILED` warning (#2087). What is still true is narrower and lives in the *step*
    /// counters: `ode_solve_until_chz_threshold` runs on `solve_ode_until_threshold`, which never tees
    /// into `STATS_SINK`, so with a scope open its `attempted_steps` read `0` on a solve that crossed the
    /// threshold normally (#1304 item 3, whose teeing is the remaining item), and the step counters cannot
    /// serve as the "did anything integrate" discriminator there. This counter is recorded by the guard
    /// directly and is unaffected.
    pub abandoned_non_finite_timeline: usize,
    /// Segments whose integration stopped short **because a state went non-finite** — the
    /// right-hand side diverged — so the output times it did not reach were padded with `NaN`
    /// in every component rather than with the last state (#1539).
    ///
    /// A subset of [`unfinished_segments`](Self::unfinished_segments), disjoint from
    /// [`stiff_aborted_segments`](Self::stiff_aborted_segments), that counts **results**, not
    /// attempts: like `stiff_aborted_segments`, a diverged
    /// `auto` escalation the guard discarded is not counted, because the caller received the
    /// explicit re-solve instead. It also covers a segment that *entered* already non-finite
    /// (the one after the divergence), which stops the same way.
    ///
    /// It is the one unfinished shape no `ode_method` or tolerance can repair — no stepper
    /// integrates `x → ∞` — so the fit's `ode_solver` warning reports it apart and drops the
    /// solver advice when every damaged segment is one of these. Non-zero means some
    /// predictions are `NaN` by construction; check the `[odes]` right-hand side and the
    /// parameter values. Like the other step counters it is a floor:
    /// [`solve_ode_until_threshold`] takes no stats.
    pub diverged_segments: usize,
    /// Segments of a model on the exact linear path ([`OdeMethod::Linear`], #852) that the
    /// matrix-exponential propagator **declined** and that were integrated instead — its
    /// self-check found the right-hand side not affine at a checked state, the exponential
    /// overflowed, or the Padé denominator was singular.
    ///
    /// Zero on every model off that path, and on every certified-linear fit measured so far.
    /// Non-zero means `ode_linear_exact` in the fit output is `false` even though the model was
    /// routed to the exact path: some of its predictions carry integrator error. A zero-length
    /// segment is not counted — it is returned unchanged by either route.
    pub linear_declined_segments: usize,
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

    /// Steps force-accepted at `min_dt` over the error test on trajectories the caller
    /// received: a clamp in an `auto` attempt the guard discarded is not one.
    pub(crate) fn kept_clamped_steps(&self) -> usize {
        self.min_step_clamped_steps.saturating_sub(self.discarded_clamped_steps)
    }

    /// Unfinished segments among the trajectories the caller received.
    pub(crate) fn kept_unfinished_segments(&self) -> usize {
        self.unfinished_segments.saturating_sub(self.discarded_unfinished_segments)
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
            auto_stiff_rejected_jets,
            auto_fallback_failed,
            unfinished_segments,
            stiff_aborted_segments,
            discarded_clamped_steps,
            discarded_unfinished_segments,
            abandoned_non_finite_timeline,
            diverged_segments,
            linear_declined_segments,
        } = *other;
        self.attempted_steps += attempted_steps;
        self.accepted_steps += accepted_steps;
        self.rejected_steps += rejected_steps;
        self.min_step_clamped_steps += min_step_clamped_steps;
        self.stiff_min_step_clamped_steps += stiff_min_step_clamped_steps;
        self.auto_stiff_segments += auto_stiff_segments;
        self.auto_switched_segments += auto_switched_segments;
        self.auto_stiff_rejected += auto_stiff_rejected;
        self.auto_stiff_rejected_jets += auto_stiff_rejected_jets;
        self.auto_fallback_failed += auto_fallback_failed;
        self.unfinished_segments += unfinished_segments;
        self.stiff_aborted_segments += stiff_aborted_segments;
        self.discarded_clamped_steps += discarded_clamped_steps;
        self.discarded_unfinished_segments += discarded_unfinished_segments;
        self.abandoned_non_finite_timeline += abandoned_non_finite_timeline;
        self.diverged_segments += diverged_segments;
        self.linear_declined_segments += linear_declined_segments;
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

/// A right-hand side over a number type, time included: `f(u, params, t, du)`. Time is a `T`
/// so a recording type (#2786) can carry a step's stage times as inputs of the code it records;
/// at `T = f64` it is the `f64` right-hand side, unchanged.
pub type RhsG<'a, T> = dyn Fn(&[T], &[T], T, &mut [T]) + 'a;

/// A generic right-hand side from a public `f64`-time one: time handed in by value.
pub fn rhs_of_f64_time<'a, T: PkNum>(
    f: &'a dyn Fn(&[T], &[T], f64, &mut [T]),
) -> impl Fn(&[T], &[T], T, &mut [T]) + 'a {
    // Only the public `f64`-time entry points wrap a closure this way, at a time the driver
    // builds with `T::from_f64`; a recording's own bodies use the generic `RhsG` directly.
    // value-read-ok: constant time at every wrapped call, so `plain` never refuses it
    move |u, p, t, du| f(u, p, t.plain(), du)
}

/// A step attempt's clock: its start `t` and size `dt` as `f64`, what every decision reads,
/// and as `T`, what the stage arithmetic reads — constants for a number type, a recording
/// type's inputs for the step it records (#2786). Each helper is the `f64` expression the
/// steppers have always formed, so a number type's stages are unchanged bit for bit; a
/// recording type forms the same operations on its inputs.
#[derive(Clone, Copy)]
pub struct StepClock<T> {
    pub t: f64,
    pub dt: f64,
    pub ts: T,
    pub dts: T,
}

impl<T: PkNum> StepClock<T> {
    #[inline]
    pub fn new(t: f64, dt: f64) -> Self {
        StepClock { t, dt, ts: T::from_f64(t), dts: T::from_f64(dt) }
    }
    /// The step's start time.
    #[inline]
    pub fn start(&self) -> T {
        if T::RECORDS_BRANCHES {
            self.ts
        } else {
            T::from_f64(self.t)
        }
    }
    /// `t + c·dt`.
    #[inline]
    pub fn at(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            self.ts + T::from_f64(c) * self.dts
        } else {
            T::from_f64(self.t + c * self.dt)
        }
    }
    /// `t + dt`.
    #[inline]
    pub fn end(&self) -> T {
        if T::RECORDS_BRANCHES {
            self.ts + self.dts
        } else {
            T::from_f64(self.t + self.dt)
        }
    }
    /// `t + x`, `x` an `f64` offset known only as a value (a finite-difference step).
    #[inline]
    pub fn shifted(&self, x: T, xf: f64) -> T {
        if T::RECORDS_BRANCHES {
            self.ts + x
        } else {
            T::from_f64(self.t + xf)
        }
    }
    /// `dt`.
    #[inline]
    pub fn h(&self) -> T {
        if T::RECORDS_BRANCHES {
            self.dts
        } else {
            T::from_f64(self.dt)
        }
    }
    /// `dt·c`.
    #[inline]
    pub fn dt_times(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            self.dts * T::from_f64(c)
        } else {
            T::from_f64(self.dt * c)
        }
    }
    /// `c·dt`.
    #[inline]
    pub fn times_dt(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            T::from_f64(c) * self.dts
        } else {
            T::from_f64(c * self.dt)
        }
    }
    /// `c / dt`.
    #[inline]
    pub fn over_dt(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            T::from_f64(c) / self.dts
        } else {
            T::from_f64(c / self.dt)
        }
    }
    /// `dt / c`.
    #[inline]
    pub fn dt_over(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            self.dts / T::from_f64(c)
        } else {
            T::from_f64(self.dt / c)
        }
    }
    /// `1 / (c·dt)`.
    #[inline]
    pub fn inv_times_dt(&self, c: f64) -> T {
        if T::RECORDS_BRANCHES {
            T::from_f64(1.0) / (T::from_f64(c) * self.dts)
        } else {
            T::from_f64(1.0 / (c * self.dt))
        }
    }
    /// `u + dt·(Σ cⱼ·kⱼ)`, summed left to right: an explicit RK stage combination.
    #[inline]
    pub fn lincomb(&self, u: T, cs: &[f64], ks: &[T]) -> T {
        let mut acc = T::from_f64(cs[0]) * ks[0];
        for j in 1..cs.len() {
            acc = acc + T::from_f64(cs[j]) * ks[j];
        }
        u + self.h() * acc
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
pub trait Stepper<T: PkNum> {
    /// Attempt a step of size `dt` from `(t, u)` and return the value-only RMS error norm
    /// (`≤ 1` means "meets tolerance"), or `f64::INFINITY` when the attempt could not be
    /// scored at all. On return [`u_new`](Stepper::u_new) and
    /// [`interpolate_component`](Stepper::interpolate_component) describe this attempt.
    fn attempt(
        &mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], t: f64, dt: f64,
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

    /// Hook fired when the driver rejects the last attempt; no-op by default.
    fn on_reject(&mut self) {}

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

    /// Dimensionless estimate of `h * rho(J)` from stages already computed by
    /// the last attempt. Only explicit methods with a published estimator
    /// override this; returning `None` falls back to the periodic Jacobian probe.
    fn stiffness_indicator(&self, _dt: f64) -> Option<f64> {
        None
    }

    /// [`Self::attempt`] with the right-hand side's native calls already bound to this
    /// integration's parameters ([`super::predictions::DirectStep::bind`]): the same values, for
    /// a stepper that can make use of it. Any other stepper takes [`Self::attempt`].
    #[allow(clippy::too_many_arguments)]
    fn attempt_bound(
        &mut self, _bound: &super::predictions::BoundStep<'_>, rhs: &RhsG<'_, T>, u: &[T],
        params: &[T], t: f64, dt: f64, opts: &OdeSolverOptions,
    ) -> f64 {
        self.attempt(rhs, u, params, t, dt, opts)
    }

    /// The consecutive attempts of [`plain_attempts`], monomorphised per stepper so the whole
    /// loop sees the concrete step code.
    fn burst(&mut self, c: &mut Burst<'_, T>) -> Option<(f64, f64)> {
        plain_attempts(self, c)
    }
}

/// A compiled recording's ODE runtime op (#2786) runs the drivers below at `T = f64` over its
/// own steppers, which carry every derivative lane of the state beside its value: while one is
/// set on this thread, [`make_stepper`] asks it for the stepper, and the dense driver reports
/// each point it records so the op can take that point's lanes.
pub trait LaneHook {
    /// The stepper for `method` over `n` states.
    fn stepper(&self, n: usize, method: OdeMethod) -> Box<dyn Stepper<f64>>;
    /// A solve starts (again) from the segment's initial state.
    fn begin(&self);
    /// The current state was recorded: into the `saveat` list (`soft = false`) or the
    /// `interp_at` list.
    fn point(&self, soft: bool);
    /// An `interp_at` point read off the step just accepted, at `theta` of `dt`.
    fn interp(&self, theta: f64, dt: f64, i_state: usize);
    /// The interpolated point's state is complete (every component read).
    fn interp_done(&self);
    /// A point the solve did not reach, padded with the last state (`NaN` lanes when diverged).
    fn tail(&self, soft: bool, diverged: bool);
    /// How many moving switches the segment watches (#2649).
    fn switches(&self) -> usize;
    /// Steps from here on integrate with switch `k` forced to the arm of its side `pos`
    /// (`Some((k, pos))`), or as the model is written (`None`).
    fn force(&self, arm: Option<(usize, bool)>);
    /// The switch functions at `(u, t)`, on values.
    fn switch_g(&self, u: &[f64], t: f64, out: &mut [f64]);
    /// The state crossed switch `k` at the current time `t`, from the side where its function
    /// is positive when `pos`: apply the sensitivity jump to every lane of the state.
    fn jump(&self, k: usize, pos: bool, t: f64);
}

thread_local! {
    /// How many moving switches the last right-hand side evaluated on this thread has
    /// (`note_switches`); never written by a model without any.
    static SWITCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A model's right-hand side with `k` moving switches (#2649) says so at each evaluation, so an
/// integration of it, whatever wraps the model's closure, knows to watch them.
pub fn note_switches(k: usize) {
    SWITCHES.with(|c| c.set(k));
}

/// What [`note_switches`] said since the last call, which clears it.
pub fn take_switches() -> usize {
    SWITCHES.with(|c| c.replace(0))
}

/// Whether switch function value `g` is on the side where it is positive. The driver's
/// verdict, shared by the jump: `l ? r` and `l − r > 0` agree on every finite pair but where
/// `l = r` exactly, a measure-zero set the forced arms in the jump do not depend on.
#[inline]
pub fn switch_side(g: f64) -> bool {
    g > 0.0
}

/// The moving switches of one segment at `T = f64` (#2649): their functions, by the lane hook
/// of a recorded segment or the model's own switch program through `rhs`.
pub struct Switches<'a, 'r, T: PkNum> {
    k: usize,
    n: usize,
    hook: Option<std::rc::Rc<dyn LaneHook>>,
    rhs: &'a RhsG<'r, T>,
    params: &'a [T],
    ue: Vec<T>,
    du: Vec<T>,
}

impl<T: PkNum> Switches<'_, '_, T> {
    /// The functions at `(u, t)`.
    /// Non-recording: the adaptive driver decides on the value lane. A recording type never reaches it: `T::ode_segment` records the segment as a loop call (the f64 driver runs at run time) and returns before this code, except after the recording has already failed.
    fn g(&mut self, u: &[T], t: f64) -> Vec<f64> {
        let (n, k) = (self.n, self.k);
        let mut out = vec![0.0; k];
        if let Some(h) = &self.hook {
            let uv: Vec<f64> = u.iter().map(|x| x.plain()).collect();
            h.switch_g(&uv, t, &mut out);
            return out;
        }
        self.ue.clear();
        self.ue.extend_from_slice(&u[..n]);
        self.ue.resize(n + k, T::from_f64(0.0));
        self.du.clear();
        self.du.resize(n + k, T::from_f64(0.0));
        (self.rhs)(&self.ue, self.params, T::from_f64(t), &mut self.du);
        for (o, d) in out.iter_mut().zip(&self.du[n..]) {
            *o = d.plain();
        }
        out
    }
}

/// How close, in units of `EPSILON·(|t| + h)`, [`locate_switch`] brackets a crossing.
pub const SWITCH_LOCATE_ULPS: f64 = 64.0;
/// Most attempts [`locate_switch`] makes per switch; Illinois needs about ten.
pub const SWITCH_LOCATE_MAX: usize = 100;

/// Where the step from `(u, t)` over `h` first crosses a moving switch, given that switch `k`
/// changed side over it (`g_old` at its start, `g_hi` at its end): Illinois on
/// `φ(θ) = g_k(step(u, t, θ·h))`, each value a real attempt of `stepper`, so the step it ends
/// with lands on the far side of the crossing, where the right-hand side takes the other arm.
/// A second switch that crossed before the bracket's far end is located instead, in what is
/// left. Every attempt forces the located switch's arm before the crossing in all its stages
/// ([`Switches::forced_attempt`]), so the step that ends there is a flow of that arm alone.
/// Returns `θ`, the switch, its functions at the end, and the error of the attempt the stepper
/// now holds, which is the forced attempt to `θ·h`.
#[allow(clippy::too_many_arguments)]
pub fn locate_switch<T: PkNum>(
    stepper: &mut dyn Stepper<T>, sw: &mut Switches<'_, '_, T>, (u, t, h): (&[T], f64, f64),
    g_old: &[f64], mut g_hi: Vec<f64>, mut k: usize, opts: &OdeSolverOptions,
) -> (f64, usize, Vec<f64>, Option<f64>) {
    let tol = SWITCH_LOCATE_ULPS * f64::EPSILON * (t.abs() + h);
    let mut hi = 1.0_f64;
    let mut err = None;
    let mut at_hi = true;
    for _ in 0..=g_old.len() {
        let (mut lo, mut flo, mut fhi, mut side) = (0.0_f64, g_old[k], g_hi[k], 0i8);
        for _ in 0..SWITCH_LOCATE_MAX {
            if (hi - lo) * h <= tol {
                break;
            }
            let mut th = lo + (hi - lo) * (flo / (flo - fhi));
            if !(th > lo && th < hi) {
                th = 0.5 * (lo + hi);
            }
            if !(th > lo && th < hi) {
                break;
            }
            let e = sw.forced_attempt(stepper, u, t, th * h, (k, switch_side(g_old[k])), opts);
            let g = sw.g(stepper.u_new(), t + th * h);
            if switch_side(g[k]) == switch_side(g_old[k]) {
                (lo, flo) = (th, g[k]);
                if side < 0 {
                    fhi *= 0.5;
                }
                side = -1;
                at_hi = false;
            } else {
                (hi, fhi) = (th, g[k]);
                if side > 0 {
                    flo *= 0.5;
                }
                side = 1;
                at_hi = true;
                err = Some(e);
                g_hi = g;
            }
        }
        match crossed(g_old, &g_hi, Some(k)) {
            Some(j) => k = j,
            None => break,
        }
    }
    if !at_hi || err.is_none() {
        err = Some(sw.forced_attempt(stepper, u, t, hi * h, (k, switch_side(g_old[k])), opts));
    }
    (hi, k, g_hi, err)
}

/// The bias that forces a switch's arm in the switch program (`ode_switch::instrument`): past
/// any switch function's value, so `g + B` and `g − B` have the sign of `B`.
pub const SWITCH_FORCE: f64 = 1e300;

impl<T: PkNum> Switches<'_, '_, T> {
    /// `stepper`'s attempt from `(u, t)` over `dt` with switch `k` forced to the arm of its
    /// side `pos` in every stage, so a step that ends on a crossing integrates the field
    /// before it throughout, whichever side its last stages land on.
    #[allow(clippy::too_many_arguments)]
    fn forced_attempt(
        &self, stepper: &mut dyn Stepper<T>, u: &[T], t: f64, dt: f64, (k, pos): (usize, bool),
        opts: &OdeSolverOptions,
    ) -> f64 {
        if let Some(h) = &self.hook {
            h.force(Some((k, pos)));
            let e = stepper.attempt(self.rhs, u, self.params, t, dt, opts);
            h.force(None);
            return e;
        }
        let (n, kk, rhs) = (self.n, self.k, self.rhs);
        let bias = T::from_f64(if pos { SWITCH_FORCE } else { -SWITCH_FORCE });
        let forced = move |x: &[T], p: &[T], tt: T, du: &mut [T]| {
            let zero = T::from_f64(0.0);
            let mut ue = x[..n].to_vec();
            ue.resize(n + kk, zero);
            ue[n + k] = bias;
            let mut d = vec![zero; n + kk];
            rhs(&ue, p, tt, &mut d);
            du[..n].copy_from_slice(&d[..n]);
        };
        stepper.attempt(&forced, u, self.params, t, dt, opts)
    }
}

/// The first switch other than `skip` whose side differs between `a` and `b`, both finite.
pub fn crossed(a: &[f64], b: &[f64], skip: Option<usize>) -> Option<usize> {
    (0..a.len()).find(|&j| {
        Some(j) != skip
            && a[j].is_finite()
            && b[j].is_finite()
            && switch_side(a[j]) != switch_side(b[j])
    })
}

thread_local! {
    static LANE_HOOK: std::cell::RefCell<Option<std::rc::Rc<dyn LaneHook>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` with `hook` serving this thread's `f64` drivers.
pub fn with_lane_hook<R>(hook: std::rc::Rc<dyn LaneHook>, f: impl FnOnce() -> R) -> R {
    /// Restores the previous hook on drop, so a panic in `f` does not leave this thread (a
    /// persistent rayon worker, or R's main thread under a caught panic) serving a dead
    /// operation's stepper.
    struct Restore(Option<std::rc::Rc<dyn LaneHook>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            LANE_HOOK.with(|h| *h.borrow_mut() = prev);
        }
    }
    let _restore = Restore(LANE_HOOK.with(|h| h.replace(Some(hook))));
    f()
}

/// The hook serving this thread's drivers at `T = f64`, if any.
#[inline]
pub fn lane_hook<T: PkNum>() -> Option<std::rc::Rc<dyn LaneHook>> {
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f64>() {
        return None;
    }
    LANE_HOOK.with(|h| h.borrow().clone())
}

/// A stepper's step as code a recording type records, and as the state its value shadows
/// carry (#2786). A step is a *start* (what the step start needs once per `(u, t)`: RK45's
/// `k1`, a Rosenbrock Jacobian) and a *step* over `dt`; each reads `u`, the clock and the
/// carried state, and writes the carried state, with no decision of its own — every decision
/// stays in the driver, on values.
pub trait StepBody<T: PkNum>: Stepper<T> {
    /// Compute the step start.
    fn start(&mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], clock: &StepClock<T>);
    /// The step's stages over `clock` (the start computed): no error norm.
    fn step(&mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], clock: &StepClock<T>);
    /// The state the start and the step write, in a fixed order, into `out`.
    fn carried_into(&self, out: &mut Vec<T>);
    /// Set the carried state, the start included.
    fn set_carried(&mut self, v: &[T]);
}

/// A carried state ([`StepBody::carried_into`]'s order) read in place at a stride: component
/// `j` at `v[j * s]` — one lane of a compiled run's slot buffer, whose lanes are contiguous per
/// slot (#2786).
#[derive(Clone, Copy)]
pub struct Carried<'a> {
    v: &'a [f64],
    s: usize,
}

impl<'a> Carried<'a> {
    pub fn new(v: &'a [f64], s: usize) -> Self {
        Carried { v, s }
    }
    #[inline(always)]
    pub fn at(&self, j: usize) -> f64 {
        self.v[j * self.s]
    }
}

/// What the driver reads off a [`StepBody`]'s carried state — the error norm, `u_new`, the
/// continuous extension, the stiffness indicator, usability — computed straight from a
/// [`Carried`] view by the same definitions the steppers' own reads call. A compiled run
/// (#2786) decides on its value lane with these: no value stepper shadows its lanes.
#[derive(Clone, Copy)]
pub enum CarriedLayout {
    Rk45 { n: usize },
    Erk { n: usize, tab: &'static super::explicit_rk::ErkTableau, dense: bool },
    Ros { n: usize, method: OdeMethod },
}

impl CarriedLayout {
    /// The layout [`make_step_body`]`(n, method)` carries, `dense` its
    /// [`Stepper::set_dense_required`].
    pub fn of(n: usize, method: OdeMethod, dense: bool) -> Self {
        match method {
            OdeMethod::Vern7 => CarriedLayout::Erk { n, tab: &super::explicit_rk::VERN7, dense },
            OdeMethod::Rosenbrock23 | OdeMethod::Rodas4 | OdeMethod::Rodas5P => {
                CarriedLayout::Ros { n, method }
            }
            OdeMethod::Rk45 | OdeMethod::Auto | OdeMethod::Linear => CarriedLayout::Rk45 { n },
        }
    }
    /// Where [`Stepper::u_new`] starts (its `n` components are contiguous).
    pub fn u_new_at(&self) -> usize {
        match *self {
            CarriedLayout::Rk45 { n } => 8 * n,
            CarriedLayout::Erk { n, tab, .. } => super::explicit_rk::erk_u_new_at(n, tab),
            CarriedLayout::Ros { n, .. } => super::rosenbrock::ros_u_new_at(n),
        }
    }
    /// The error norm of the last attempt, on values (`∞` for an unusable one).
    pub fn err(&self, c: Carried<'_>, u: &[f64], dt: f64, opts: &OdeSolverOptions) -> f64 {
        match *self {
            CarriedLayout::Rk45 { n } => {
                rk45_err_norm(n, |b, i| c.at(b * n + i), |i| u[i], dt, opts)
            }
            CarriedLayout::Erk { n, tab, .. } => {
                let un = self.u_new_at();
                super::explicit_rk::erk_err_norm(
                    n,
                    tab,
                    |b, i| c.at(b * n + i),
                    |i| c.at(un + i),
                    |i| u[i],
                    dt,
                    opts,
                )
            }
            CarriedLayout::Ros { n, .. } => {
                if !self.usable(c) {
                    return f64::INFINITY;
                }
                let un = self.u_new_at();
                super::rosenbrock::ros_err_norm(
                    n,
                    |i| c.at(un + i),
                    |i| c.at(un + n + i),
                    |i| u[i],
                    opts,
                )
            }
        }
    }
    /// [`Stepper::attempt_usable`].
    pub fn usable(&self, c: Carried<'_>) -> bool {
        match *self {
            CarriedLayout::Ros { n, method } => {
                c.at(super::rosenbrock::ros_usable_at(n, method)) == 1.0
            }
            _ => true,
        }
    }
    /// [`Stepper::interpolate_component`], `u_old_i` the component's start value.
    pub fn interp(&self, c: Carried<'_>, theta: f64, u_old_i: f64, dt: f64, i: usize) -> f64 {
        match *self {
            CarriedLayout::Rk45 { n } => {
                hermite_g(theta, dt, u_old_i, c.at(i), c.at(8 * n + i), c.at(6 * n + i))
            }
            CarriedLayout::Erk { n, tab, .. } => {
                super::explicit_rk::erk_interp(tab, |b, i| c.at(b * n + i), theta, u_old_i, dt, i)
            }
            CarriedLayout::Ros { n, method } => {
                // The stages follow `u_new` and `err`.
                let un = self.u_new_at();
                super::rosenbrock::ros_interp(
                    method,
                    |j| c.at(un + 2 * n + j * n + i),
                    theta,
                    u_old_i,
                    c.at(un + i),
                    dt,
                )
            }
        }
    }
    /// [`Stepper::stiffness_indicator`].
    pub fn stiffness(&self, c: Carried<'_>, dt: f64) -> Option<f64> {
        match *self {
            CarriedLayout::Rk45 { n } => rk45_stiffness(n, |b, i| c.at(b * n + i), dt),
            CarriedLayout::Erk { .. } | CarriedLayout::Ros { .. } => None,
        }
    }
    /// [`Stepper::err_exp`].
    pub fn err_exp(&self) -> f64 {
        match *self {
            CarriedLayout::Rk45 { .. } => RK45_ERR_EXP,
            CarriedLayout::Erk { tab, .. } => tab.err_exp(),
            CarriedLayout::Ros { method, .. } => super::rosenbrock::method_err_exp(method),
        }
    }
    /// Whether the step start survives [`Stepper::on_accept`] (computed at the new `(u, t)`:
    /// RK45's FSAL `k7`), so the next attempt does not run [`StepBody::start`].
    pub fn start_kept_on_accept(&self) -> bool {
        match *self {
            CarriedLayout::Rk45 { .. } => true,
            CarriedLayout::Erk { tab, dense, .. } => tab.start_kept_on_accept(dense),
            CarriedLayout::Ros { .. } => super::rosenbrock::START_KEPT_ON_ACCEPT,
        }
    }
}

/// I-controller exponent for RK45's 5(4) pair.
pub const RK45_ERR_EXP: f64 = 1.0 / 5.0;

/// [`make_stepper`] as a [`StepBody`].
pub fn make_step_body<T: PkNum>(n: usize, method: OdeMethod) -> Box<dyn StepBody<T>> {
    match method {
        OdeMethod::Vern7 => {
            Box::new(super::explicit_rk::ErkStepper::new(n, &super::explicit_rk::VERN7))
        }
        stiff @ (OdeMethod::Rosenbrock23 | OdeMethod::Rodas4 | OdeMethod::Rodas5P) => {
            Box::new(super::rosenbrock::RosStepper::new(n, stiff))
        }
        OdeMethod::Rk45 | OdeMethod::Auto | OdeMethod::Linear => Box::new(Rk45Stepper::new(n)),
    }
}

/// A compiled recording's ODE segment at run time (#2786): the adaptive integration of
/// [`integrate_resolved_g_inner`] past the exact-linear path, at `f64`, under the caller's
/// [`LaneHook`].
pub fn integrate_nonlinear_f64(
    rhs: &RhsG<'_, f64>, u0: &[f64], t_span: (f64, f64), saveat: &[f64], interp_at: &[f64],
    opts: &OdeSolverOptions, auto_state: &mut OdeAutoSwitchState,
) -> (Vec<SolPointG<f64>>, Vec<SolPointG<f64>>) {
    // A compiled plan's segment (#2786) is the derivative path's integration, counted as the
    // dual sweeps' were; the run a recording makes of it is not (whether one happens depends
    // on what the plan cache holds).
    if crate::parser::model_parser::rhs_regcode::kernel_jit::recording_open() {
        return integrate_resolved_g_inner(
            rhs,
            u0,
            t_span,
            &[],
            saveat,
            interp_at,
            opts,
            None,
            auto_state,
        );
    }
    integrate_resolved_g(rhs, u0, t_span, &[], saveat, interp_at, opts, None, auto_state)
}

/// Build the stepper for `method`, sized for an `n`-state system.
///
/// This is the **only** place a method name selects an implementation. Consumers take a
/// `&mut dyn Stepper<T>`; the dynamic call happens once per step attempt, against `n + 1`
/// right-hand-side evaluations and an `O(n³)` factorization, so it does not register.
/// Every arm is spelled out rather than ending in a catch-all: adding an [`OdeMethod`] should
/// fail to compile here (and in the three sibling matches in [`super::rosenbrock`]) rather
/// than route silently into the Rosenbrock stepper and panic inside a fit's worker thread.
pub fn make_stepper<T: PkNum>(n: usize, method: OdeMethod) -> Box<dyn Stepper<T>> {
    if let Some(hook) = lane_hook::<T>() {
        let s: Box<dyn Stepper<f64>> = hook.stepper(n, method);
        // SAFETY: `lane_hook` serves `T = f64` only, so the two box types are the same.
        return unsafe { std::mem::transmute::<Box<dyn Stepper<f64>>, Box<dyn Stepper<T>>>(s) };
    }
    match method {
        OdeMethod::Rk45 => Box::new(PooledRk45::take(n)),
        OdeMethod::Vern7 => {
            Box::new(super::explicit_rk::ErkStepper::new(n, &super::explicit_rk::VERN7))
        }
        stiff @ (OdeMethod::Rosenbrock23 | OdeMethod::Rodas4 | OdeMethod::Rodas5P) => {
            Box::new(super::rosenbrock::RosStepper::new(n, stiff))
        }
        // `Auto` names a *decision*, not a stepper: every driver resolves it through
        // `stiffness::resolve_method` before it gets here, so this arm is only reachable
        // from a caller that built a stepper straight from the options. Falling through to
        // the explicit default keeps such a caller integrating (correctly, if slowly on a
        // stiff system) rather than panicking inside a fit's worker thread.
        OdeMethod::Auto => Box::new(PooledRk45::take(n)),
        // `Linear` is not a stepper either: `integrate_resolved_g_inner` serves it exactly and
        // integrates a declined segment as `Auto`, so only a caller building a stepper straight
        // from the options lands here. Same reasoning as the `Auto` arm.
        OdeMethod::Linear => Box::new(Rk45Stepper::new(n)),
    }
}

/// Cubic Hermite interpolation across one accepted step, generic over the state scalar.
/// `s ∈ [0, 1]` is the normalized position, `h` the step length, `(y0, d0)` / `(y1, d1)` the
/// value and derivative at the step's start and end — the FSAL `k1` / `k7` slots.
#[inline]
pub fn hermite_g<T: PkNum>(s: f64, h: f64, y0: T, d0: T, y1: T, d1: T) -> T {
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
pub struct Rk45Stepper<T> {
    pub n: usize,
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

/// A [`Rk45Stepper`] taken from this thread's pool and returned to it on drop (#1564). Every
/// integration segment builds a stepper, and the RK45 one holds nine state-sized vectors: on
/// an ODE fit that was the largest single source of heap allocations (a Michaelis–Menten fit
/// allocates ~2.6M times a second). A reused stepper is reset to exactly what
/// [`Rk45Stepper::new`] builds — zeroed stages, no carried `k1`, mark 0 — so the steps it
/// takes are the fresh one's, bit for bit.
pub struct PooledRk45<T: PkNum>(Option<Box<Rk45Stepper<T>>>);

thread_local! {
    pub static RK45_POOL: std::cell::RefCell<Vec<Box<dyn std::any::Any>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Most steppers a thread keeps: the drivers hold one at a time, a nested solve one more.
/// A full pool evicts its oldest entry, so steppers of a number type or state size the
/// thread no longer uses (an R session moving between models, a model search across
/// compartment counts) cannot fill it and stop the pooling of the ones it does use.
pub const RK45_POOL_MAX: usize = 8;

impl<T: PkNum> PooledRk45<T> {
    pub fn take(n: usize) -> Self {
        let pooled = RK45_POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.iter()
                .rposition(|b| b.downcast_ref::<Rk45Stepper<T>>().is_some_and(|s| s.n == n))
                .map(|i| p.remove(i))
        });
        let stepper = match pooled.and_then(|b| b.downcast::<Rk45Stepper<T>>().ok()) {
            Some(mut s) => {
                s.reset();
                s
            }
            None => Box::new(Rk45Stepper::new(n)),
        };
        PooledRk45(Some(stepper))
    }

    fn get(&self) -> &Rk45Stepper<T> {
        self.0.as_ref().expect("held until drop")
    }

    fn get_mut(&mut self) -> &mut Rk45Stepper<T> {
        self.0.as_mut().expect("held until drop")
    }
}

impl<T: PkNum> Drop for PooledRk45<T> {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            RK45_POOL.with(|p| {
                let mut p = p.borrow_mut();
                // Oldest first: `take` reuses the newest match, a full pool drops the oldest.
                if p.len() == RK45_POOL_MAX {
                    p.remove(0);
                }
                p.push(s);
            });
        }
    }
}

impl<T: PkNum> Stepper<T> for PooledRk45<T> {
    fn attempt(
        &mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], t: f64, dt: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        self.get_mut().attempt(rhs, u, params, t, dt, opts)
    }
    fn u_new(&self) -> &[T] {
        self.get().u_new()
    }
    fn interpolate_component(&self, theta: f64, u_old: &[T], dt: f64, i: usize) -> T {
        self.get().interpolate_component(theta, u_old, dt, i)
    }
    fn on_accept(&mut self) {
        self.get_mut().on_accept()
    }
    fn on_reject(&mut self) {
        self.get_mut().on_reject()
    }
    fn err_exp(&self) -> f64 {
        self.get().err_exp()
    }
    fn set_dense_required(&mut self, yes: bool) {
        self.get_mut().set_dense_required(yes)
    }
    fn attempt_usable(&self) -> bool {
        self.get().attempt_usable()
    }
    fn stiffness_indicator(&self, dt: f64) -> Option<f64> {
        self.get().stiffness_indicator(dt)
    }
    fn attempt_bound(
        &mut self, bound: &super::predictions::BoundStep<'_>, rhs: &RhsG<'_, T>, u: &[T],
        params: &[T], t: f64, dt: f64, opts: &OdeSolverOptions,
    ) -> f64 {
        self.get_mut().attempt_bound(bound, rhs, u, params, t, dt, opts)
    }
    fn burst(&mut self, c: &mut Burst<'_, T>) -> Option<(f64, f64)> {
        self.get_mut().burst(c)
    }
}

impl<T: PkNum> Rk45Stepper<T> {
    /// Back to what [`Self::new`] builds, keeping the buffers.
    fn reset(&mut self) {
        let z = T::from_f64(0.0);
        for v in [
            &mut self.k1,
            &mut self.k2,
            &mut self.k3,
            &mut self.k4,
            &mut self.k5,
            &mut self.k6,
            &mut self.k7,
            &mut self.u_tmp,
            &mut self.u5,
        ] {
            v.fill(z);
        }
        self.have_k1 = false;
    }

    pub fn new(n: usize) -> Self {
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

/// One stage's right-hand-side call, so a stepper body can be monomorphized over how it
/// reaches the model: through the closure, or straight into its machine code
/// ([`DirectCall`]), inlined.
pub trait StageRhs<T> {
    fn call(&self, u: &[T], params: &[T], t: T, du: &mut [T]);
}

impl<T> StageRhs<T> for dyn Fn(&[T], &[T], T, &mut [T]) + '_ {
    #[inline(always)]
    fn call(&self, u: &[T], params: &[T], t: T, du: &mut [T]) {
        self(u, params, t, du)
    }
}

/// A registered `f64` right-hand side called directly
/// ([`super::rosenbrock::registered_direct`]), with its closure for the calls the native
/// code declines.
pub struct DirectCall<'a, T> {
    pub direct: &'a super::predictions::DirectStep<'a>,
    pub rhs: &'a RhsG<'a, T>,
}

/// Non-recording: a `BoundStep` binds native calls to `f64` parameters, so it is only built at `T = f64`.
impl<T: PkNum> StageRhs<T> for super::predictions::BoundStep<'_> {
    #[inline(always)]
    fn call(&self, u: &[T], params: &[T], t: T, du: &mut [T]) {
        debug_assert_eq!(std::any::TypeId::of::<T>(), std::any::TypeId::of::<f64>());
        // SAFETY: a `BoundStep` is only handed out at `T = f64` (`registered_direct`).
        let (uf, pf, duf) = unsafe {
            (
                std::slice::from_raw_parts(u.as_ptr().cast::<f64>(), u.len()),
                std::slice::from_raw_parts(params.as_ptr().cast::<f64>(), params.len()),
                std::slice::from_raw_parts_mut(du.as_mut_ptr().cast::<f64>(), du.len()),
            )
        };
        self.eval(uf, pf, t.plain(), duf)
    }
}

impl<T: PkNum> StageRhs<T> for DirectCall<'_, T> {
    #[inline(always)]
    fn call(&self, u: &[T], params: &[T], t: T, du: &mut [T]) {
        super::rosenbrock::call_direct(self.direct, self.rhs, u, params, t, du)
    }
}

impl<T: PkNum> Rk45Stepper<T> {
    /// [`Stepper::attempt`] over any [`StageRhs`], so the direct path is monomorphized and
    /// inlined.
    // The stage loops walk `u` alongside seven stage vectors at once; zipping them would
    // obscure the Butcher tableau they transcribe — and the exact association of these
    // expressions is what keeps `T = f64` bit-identical to the pre-refactor loop.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    fn attempt_with<R: StageRhs<T> + ?Sized>(
        &mut self, rhs: &R, u: &[T], params: &[T], t: f64, dt_eff: f64, opts: &OdeSolverOptions,
    ) -> f64 {
        let n = self.n;
        self.attempt_len(n, rhs, u, params, t, dt_eff, opts)
    }

    /// [`Self::attempt_with`] at a state size the caller passes as a literal, so the stage
    /// loops are unrolled and index without checks; `n` must be `self.n`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn attempt_len<R: StageRhs<T> + ?Sized>(
        &mut self, n: usize, rhs: &R, u: &[T], params: &[T], t: f64, dt_eff: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        debug_assert_eq!(n, self.n);
        self.stages_len(n, rhs, u, params, &StepClock::new(t, dt_eff));
        self.err_norm_len(n, u, dt_eff, opts)
    }

    /// The step's stages and its proposed state `u5` over `clock` — the arithmetic only, no
    /// decision: what a recording type records as the step's code (#2786).
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    pub fn stages<R: StageRhs<T> + ?Sized>(
        &mut self, rhs: &R, u: &[T], params: &[T], clock: &StepClock<T>,
    ) {
        let n = self.n;
        self.stages_len(n, rhs, u, params, clock)
    }

    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    fn stages_len<R: StageRhs<T> + ?Sized>(
        &mut self, n: usize, rhs: &R, u: &[T], params: &[T], clock: &StepClock<T>,
    ) {
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
        //
        // Every buffer is cut to `n` up front, so the loops index without bounds checks
        // and the compiler keeps the stage pointers in registers across the calls.
        let Self { k1, k2, k3, k4, k5, k6, k7, u_tmp, u5, have_k1, .. } = self;
        let (k1, k2, k3, k4, k5, k6, k7) = (
            &mut k1[..n],
            &mut k2[..n],
            &mut k3[..n],
            &mut k4[..n],
            &mut k5[..n],
            &mut k6[..n],
            &mut k7[..n],
        );
        let (u_tmp, u5, u) = (&mut u_tmp[..n], &mut u5[..n], &u[..n]);
        if !*have_k1 {
            rhs.call(u, params, clock.start(), k1);
            *have_k1 = true;
        }
        let b21 = clock.dt_times(B21);
        for i in 0..n {
            u_tmp[i] = u[i] + b21 * k1[i];
        }
        rhs.call(u_tmp, params, clock.at(A2), k2);

        for i in 0..n {
            u_tmp[i] = clock.lincomb(u[i], &[B31, B32], &[k1[i], k2[i]]);
        }
        rhs.call(u_tmp, params, clock.at(A3), k3);

        for i in 0..n {
            u_tmp[i] = clock.lincomb(u[i], &[B41, B42, B43], &[k1[i], k2[i], k3[i]]);
        }
        rhs.call(u_tmp, params, clock.at(A4), k4);

        for i in 0..n {
            u_tmp[i] = clock.lincomb(u[i], &[B51, B52, B53, B54], &[k1[i], k2[i], k3[i], k4[i]]);
        }
        rhs.call(u_tmp, params, clock.at(A5), k5);

        for i in 0..n {
            u_tmp[i] = clock.lincomb(
                u[i],
                &[B61, B62, B63, B64, B65],
                &[k1[i], k2[i], k3[i], k4[i], k5[i]],
            );
        }
        rhs.call(u_tmp, params, clock.end(), k6);

        // 5th-order solution.
        for i in 0..n {
            u5[i] = clock.lincomb(
                u[i],
                &[B71, B73, B74, B75, B76],
                &[k1[i], k3[i], k4[i], k5[i], k6[i]],
            );
        }
        rhs.call(u5, params, clock.end(), k7);
    }

    /// The error estimate (5th − 4th) of the last [`Self::stages`], on values only — the
    /// step sequence must not depend on derivative components.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    /// Non-recording: the adaptive driver decides on the value lane. A recording type never reaches it: `T::ode_segment` records the segment as a loop call (the f64 driver runs at run time) and returns before this code, except after the recording has already failed.
    fn err_norm_len(&self, n: usize, u: &[T], dt_eff: f64, opts: &OdeSolverOptions) -> f64 {
        let blocks = self.blocks_n(n);
        let u = &u[..n];
        rk45_err_norm(n, |b, i| blocks[b][i].plain(), |i| u[i].plain(), dt_eff, opts)
    }

    /// Every carried block cut to `n` (the [`StepBody::carried_into`] order), so a loop over
    /// `0..n` indexes them without bounds checks.
    #[inline(always)]
    fn blocks_n(&self, n: usize) -> [&[T]; 9] {
        [
            &self.k1[..n],
            &self.k2[..n],
            &self.k3[..n],
            &self.k4[..n],
            &self.k5[..n],
            &self.k6[..n],
            &self.k7[..n],
            &self.u_tmp[..n],
            &self.u5[..n],
        ]
    }
}

/// RK45's error norm over its carried blocks, `k(b, i)` component `i` of block `b` (the
/// [`StepBody::carried_into`] order) on values: the stepper's own, and a compiled run's off its
/// value lane ([`CarriedLayout`]) — one definition, so both decide alike bit for bit.
#[inline(always)]
pub fn rk45_err_norm(
    n: usize, k: impl Fn(usize, usize) -> f64, u: impl Fn(usize) -> f64, dt_eff: f64,
    opts: &OdeSolverOptions,
) -> f64 {
    let mut err_norm = 0.0;
    for i in 0..n {
        let err_i = dt_eff
            * (E1 * k(0, i)
                + E3 * k(2, i)
                + E4 * k(3, i)
                + E5 * k(4, i)
                + E6 * k(5, i)
                + E7 * k(6, i));
        let scale = scale_tol(opts.abstol, opts.reltol, k(8, i), u(i));
        err_norm += (err_i / scale) * (err_i / scale);
    }
    (err_norm / n as f64).sqrt()
}

/// Hairer & Wanner's DOPRI5 stiffness estimate over RK45's carried blocks (as
/// [`rk45_err_norm`]). `u_tmp` is the stage-6 state and remains intact after `k6` is
/// evaluated; `u5`/`k7` are the accepted endpoint. Values only keep prediction and
/// sensitivity solves on the same path.
#[inline(always)]
pub fn rk45_stiffness(n: usize, k: impl Fn(usize, usize) -> f64, dt: f64) -> Option<f64> {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dk = k(6, i) - k(5, i);
        let du = k(8, i) - k(7, i);
        num += dk * dk;
        den += du * du;
    }
    (den > 0.0).then(|| dt.abs() * (num / den).sqrt())
}

impl<T: PkNum> StepBody<T> for Rk45Stepper<T> {
    fn start(&mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], clock: &StepClock<T>) {
        rhs(&u[..self.n], params, clock.start(), &mut self.k1[..self.n]);
        self.have_k1 = true;
    }
    fn step(&mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], clock: &StepClock<T>) {
        self.have_k1 = true;
        self.stages(rhs, u, params, clock);
    }
    fn carried_into(&self, out: &mut Vec<T>) {
        out.clear();
        for v in [
            &self.k1,
            &self.k2,
            &self.k3,
            &self.k4,
            &self.k5,
            &self.k6,
            &self.k7,
            &self.u_tmp,
            &self.u5,
        ] {
            out.extend_from_slice(v);
        }
    }
    fn set_carried(&mut self, v: &[T]) {
        let n = self.n;
        for (j, dst) in [
            &mut self.k1,
            &mut self.k2,
            &mut self.k3,
            &mut self.k4,
            &mut self.k5,
            &mut self.k6,
            &mut self.k7,
            &mut self.u_tmp,
            &mut self.u5,
        ]
        .into_iter()
        .enumerate()
        {
            dst.copy_from_slice(&v[j * n..(j + 1) * n]);
        }
        self.have_k1 = true;
    }
}

impl<T: PkNum> Stepper<T> for Rk45Stepper<T> {
    fn attempt(
        &mut self, rhs: &RhsG<'_, T>, u: &[T], params: &[T], t: f64, dt_eff: f64,
        opts: &OdeSolverOptions,
    ) -> f64 {
        // A registered `f64` right-hand side runs its machine code directly at every stage:
        // the same values, without the closures between the stepper and the model.
        if let Some(direct) = super::rosenbrock::registered_direct(rhs) {
            return self.attempt_with(&DirectCall { direct, rhs }, u, params, t, dt_eff, opts);
        }
        self.attempt_with(rhs, u, params, t, dt_eff, opts)
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
        RK45_ERR_EXP
    }

    fn attempt_usable(&self) -> bool {
        true
    }

    /// Non-recording: the adaptive driver decides on the value lane. A recording type never reaches it: `T::ode_segment` records the segment as a loop call (the f64 driver runs at run time) and returns before this code, except after the recording has already failed.
    fn stiffness_indicator(&self, dt: f64) -> Option<f64> {
        let blocks = self.blocks_n(self.n);
        rk45_stiffness(self.n, |b, i| blocks[b][i].plain(), dt)
    }

    fn attempt_bound(
        &mut self, bound: &super::predictions::BoundStep<'_>, _rhs: &RhsG<'_, T>, u: &[T],
        params: &[T], t: f64, dt_eff: f64, opts: &OdeSolverOptions,
    ) -> f64 {
        // The common small state sizes get their own copy of the step with `n` a literal.
        match self.n {
            1 => self.attempt_len(1, bound, u, params, t, dt_eff, opts),
            2 => self.attempt_len(2, bound, u, params, t, dt_eff, opts),
            3 => self.attempt_len(3, bound, u, params, t, dt_eff, opts),
            4 => self.attempt_len(4, bound, u, params, t, dt_eff, opts),
            _ => self.attempt_with(bound, u, params, t, dt_eff, opts),
        }
    }
}

/// The input and carried state of [`plain_attempts`]: one segment's step loop, lent for a run of
/// attempts. Plain fields, copied into locals at entry and written back at exit.
pub struct Burst<'a, T> {
    pub bound: &'a super::predictions::BoundStep<'a>,
    pub rhs: &'a RhsG<'a, T>,
    pub params: &'a [T],
    pub opts: &'a OdeSolverOptions,
    pub saveat: &'a [f64],
    pub interp_at: &'a [f64],
    pub interp_idx: usize,
    pub tf: f64,
    pub err_exp: f64,
    pub u: &'a mut [T],
    pub t: f64,
    pub dt: f64,
    pub steps_left: usize,
    pub save_idx: usize,
    pub results: &'a mut Vec<SolPointG<T>>,
    pub auto_state: &'a mut OdeAutoSwitchState,
    pub accepted_since_probe: usize,
    /// `auto` may still switch this segment: it is on the explicit stepper, whose stiffness
    /// stages can ratify a verdict, and the switch is one-way (#2845) — so it alone is probed,
    /// escalated and counted toward the next probe.
    pub on_explicit: bool,
    pub attempted: usize,
    pub accepted: usize,
    pub rejected: usize,
}

/// The next step's length: not past `tf`, and ending on the next `saveat` time when it falls
/// inside the step.
#[inline(always)]
pub fn next_dt_eff(dt: f64, t: f64, tf: f64, saveat: &[f64], save_idx: usize, min_dt: f64) -> f64 {
    let mut dt_eff = dt.min(tf - t);
    if save_idx < saveat.len() && t + dt_eff > saveat[save_idx] + 1e-15 {
        dt_eff = (saveat[save_idx] - t).max(min_dt);
    }
    dt_eff
}

/// Consecutive attempts of the step loop that need no decision beyond accept / reject, run in
/// one small frame so the step state stays in registers: the common accepted or rejected step
/// is committed and the controller applied exactly as [`integrate_dense_g`] does for it.
///
/// Returns at the first attempt that needs the driver — an unusable one, a `min_dt`
/// force-accept, a step that crosses an `interp_at` time, a periodic or ratified stiffness
/// verdict — *without having committed it*: `Some((dt_eff, err_norm))` is that attempt's result,
/// for the driver to finish. `None` means the segment ended or the step budget ran out.
#[inline(always)]
pub fn plain_attempts<T: PkNum, S: Stepper<T> + ?Sized>(
    s: &mut S, c: &mut Burst<'_, T>,
) -> Option<(f64, f64)> {
    let (bound, rhs, params, opts) = (c.bound, c.rhs, c.params, c.opts);
    let (saveat, interp_at, tf, err_exp) = (c.saveat, c.interp_at, c.tf, c.err_exp);
    let min_dt = opts.min_dt;
    let (mut t, mut dt, mut save_idx, mut steps) = (c.t, c.dt, c.save_idx, c.steps_left);
    let mut since_probe = c.accepted_since_probe;
    let (mut accepted, mut rejected) = (0usize, 0usize);
    let u: &mut [T] = &mut *c.u;
    let exit = loop {
        if steps == 0 || t >= tf - 1e-15 {
            break None;
        }
        steps -= 1;
        let dt_eff = next_dt_eff(dt, t, tf, saveat, save_idx, min_dt);
        let err_norm = s.attempt_bound(bound, rhs, u, params, t, dt_eff, opts);
        if !s.attempt_usable() || (!(err_norm <= 1.0) && dt_eff <= min_dt) {
            break Some((dt_eff, err_norm));
        }
        if err_norm <= 1.0 {
            let t_new = t + dt_eff;
            if (interp_idx_due(interp_at, c.interp_idx, t_new))
                || (c.on_explicit
                    && since_probe + 1 >= AUTO_SWITCH_PROBE_INTERVAL
                    && t_new < tf - 1e-15)
            {
                break Some((dt_eff, err_norm));
            }
            if c.on_explicit {
                if let Some(ind) = s.stiffness_indicator(dt_eff) {
                    let kept = *c.auto_state;
                    if c.auto_state.observe_rk45(ind) {
                        *c.auto_state = kept;
                        break Some((dt_eff, err_norm));
                    }
                }
            }
            t = t_new;
            copy_state(u, s.u_new());
            s.on_accept();
            while save_idx < saveat.len() && (t - saveat[save_idx]).abs() < 1e-12 {
                c.results.push(SolPointG { t: saveat[save_idx], u: u.to_vec() });
                save_idx += 1;
            }
            since_probe += c.on_explicit as usize;
            accepted += 1;
        } else {
            s.on_reject();
            rejected += 1;
        }
        // Adapt the step (memoryless I-controller).
        let factor = if !err_norm.is_finite() {
            NONFINITE_ERR_SHRINK_FACTOR
        } else if err_norm > 1e-15 {
            0.9 * err_norm.powf(-err_exp)
        } else {
            5.0
        };
        dt = (dt_eff * factor.clamp(0.2, 5.0)).max(min_dt);
    };
    c.t = t;
    c.dt = dt;
    c.save_idx = save_idx;
    c.steps_left = steps;
    c.accepted_since_probe = since_probe;
    c.attempted += accepted + rejected;
    c.accepted += accepted;
    c.rejected += rejected;
    exit
}

/// `dst.copy_from_slice(src)`, with the small state sizes copied inline rather than through a
/// `memcpy` call.
#[inline(always)]
pub fn copy_state<T: Copy>(dst: &mut [T], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    match dst.len() {
        1 => dst[0] = src[0],
        2 => {
            dst[0] = src[0];
            dst[1] = src[1];
        }
        3 => {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
        4 => {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
        }
        _ => dst.copy_from_slice(src),
    }
}

/// Whether the next `interp_at` time falls inside a step ending at `t_new`.
#[inline(always)]
pub fn interp_idx_due(interp_at: &[f64], idx: usize, t_new: f64) -> bool {
    idx < interp_at.len() && interp_at[idx] <= t_new + 1e-12
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
/// Non-recording: the adaptive driver decides on the value lane. A recording type never reaches it: `T::ode_segment` records the segment as a loop call (the f64 driver runs at run time) and returns before this code, except after the recording has already failed.
pub fn integrate_dense_g<T: PkNum>(
    method: OdeMethod, rhs: &RhsG<'_, T>, u0: &[T], t_span: (f64, f64), params: &[T],
    saveat: &[f64], interp_at: &[f64], opts: &OdeSolverOptions,
    mut stats: Option<&mut OdeSolverStats>, auto_state: &mut OdeAutoSwitchState,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    let n = u0.len();
    let (t0, tf) = t_span;
    let hook = lane_hook::<T>();
    if let Some(h) = &hook {
        h.begin();
    }

    if (tf - t0).abs() < 1e-15 {
        let at = |times: &[f64], soft: bool| -> Vec<SolPointG<T>> {
            times
                .iter()
                .map(|&t| {
                    if let Some(h) = &hook {
                        h.point(soft);
                    }
                    SolPointG { t, u: u0.to_vec() }
                })
                .collect()
        };
        return (at(saveat, false), at(interp_at, true));
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
    if method != OdeMethod::EXPLICIT_FALLBACK {
        auto_state.reset_rk45_verdict();
    }
    let mut stepper = make_stepper::<T>(n, method);
    stepper.set_dense_required(!interp_at.is_empty());
    let mut err_exp = stepper.err_exp();
    // Mid-segment re-probing is `auto`'s business only: a named method is pinned, failed
    // result included (the same rule the escalation guard follows). The explicit re-solve the
    // guard runs pins its method for exactly this reason, so it cannot switch back.
    let switching = opts.method == OdeMethod::Auto && opts.auto_switch;
    let mut accepted_since_probe = auto_state.accepted_since_probe;
    // The moving switches (#2649), watched on values (`T = f64`: the prediction, and a
    // recording's run over its lanes). A recorded segment's hook knows how many; otherwise the
    // model's right-hand side says so at its first evaluation (`note_switches`), and a model
    // without any never does, so its integration takes no other branch.
    let watch = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>();
    if watch {
        take_switches();
    }
    let mut sw = Switches {
        k: hook.as_ref().map_or(0, |h| h.switches()),
        n,
        hook: hook.clone(),
        rhs,
        params,
        ue: Vec::new(),
        du: Vec::new(),
    };
    let mut probe = watch && hook.is_none();
    // A registered native right-hand side, bound once to this segment's parameters: every
    // attempt then calls its machine code without re-checking what cannot have changed.
    let bound = if watch && hook.is_none() {
        super::rosenbrock::registered_direct(rhs).and_then(|d| {
            // SAFETY: `registered_direct` serves `T = f64` only.
            let pf =
                unsafe { std::slice::from_raw_parts(params.as_ptr().cast::<f64>(), params.len()) };
            d.bind(pf, n)
        })
    } else {
        None
    };
    // The switch functions at `(u, t)`, once a step has needed them.
    let mut g_old: Vec<f64> = Vec::new();

    let mut steps_left = opts.max_steps;
    while steps_left > 0 {
        // A run of plain attempts, in a frame of its own; it hands back the first one that
        // needs this driver, attempted but not committed.
        let mut pending = None;
        if let (Some(b), false, 0, 0) = (&bound, probe, sw.k, consecutive_min_step_clamps) {
            let mut c = Burst {
                bound: b,
                rhs,
                params,
                opts,
                saveat,
                interp_at,
                interp_idx,
                tf,
                err_exp,
                u: &mut u,
                t,
                dt,
                steps_left,
                save_idx,
                results: &mut results,
                auto_state: &mut *auto_state,
                accepted_since_probe,
                on_explicit: switching && method == OdeMethod::EXPLICIT_FALLBACK,
                attempted: 0,
                accepted: 0,
                rejected: 0,
            };
            pending = stepper.burst(&mut c);
            (t, dt, save_idx, steps_left) = (c.t, c.dt, c.save_idx, c.steps_left);
            accepted_since_probe = c.accepted_since_probe;
            if let Some(s) = stats.as_deref_mut() {
                s.attempted_steps += c.attempted;
                s.accepted_steps += c.accepted;
                s.rejected_steps += c.rejected;
            }
            if pending.is_none() {
                break;
            }
        } else {
            steps_left -= 1;
            if t >= tf - 1e-15 {
                break;
            }
        }

        let (dt_eff, mut err_norm) = match pending {
            Some(p) => p,
            None => {
                let dt_eff = next_dt_eff(dt, t, tf, saveat, save_idx, opts.min_dt);
                let e = match &bound {
                    Some(b) => stepper.attempt_bound(b, rhs, &u, params, t, dt_eff, opts),
                    None => stepper.attempt(rhs, &u, params, t, dt_eff, opts),
                };
                (dt_eff, e)
            }
        };
        let mut dt_eff = dt_eff;
        let usable = stepper.attempt_usable();

        // An unusable attempt at `min_dt` is unrecoverable: the step cannot be shrunk further
        // and there is no `u_new` to force-accept. Stop and let the tail pad: freeze-padded when
        // the carried state is finite, `NaN`-padded when it has already gone non-finite (#1539).
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
        if !accepted {
            stepper.on_reject();
        }
        // A moving switch the step crossed: end the step at the crossing instead, on the far
        // side of it, so the field changes arms there and the jump is applied there.
        let mut event: Option<(usize, bool)> = None;
        let mut g_end: Option<Vec<f64>> = None;
        if accepted && watch {
            if probe {
                sw.k = take_switches();
                probe = false;
            }
            if sw.k > 0 {
                if g_old.is_empty() {
                    g_old = sw.g(&u, t);
                }
                let g_new = sw.g(stepper.u_new(), t + dt_eff);
                if let Some(k) = crossed(&g_old, &g_new, None) {
                    let (theta, k, g, e) = locate_switch(
                        &mut *stepper,
                        &mut sw,
                        (&u, t, dt_eff),
                        &g_old,
                        g_new,
                        k,
                        opts,
                    );
                    dt_eff *= theta;
                    if let Some(e) = e {
                        err_norm = e;
                    }
                    event = Some((k, switch_side(g_old[k])));
                    g_end = Some(g);
                } else {
                    g_end = Some(g_new);
                }
            }
        }
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
            let stiffness_indicator = stepper.stiffness_indicator(dt_eff);
            // Soft sampling: read every `interp_at` time in this just-accepted step's
            // half-open span `(t, t+dt_eff]` off the method's continuous extension, *before*
            // `u` advances and the stepper recycles its stages. This only reads committed
            // step data, so it cannot affect the step sequence.
            while interp_idx < interp_at.len() && interp_at[interp_idx] <= t + dt_eff + 1e-12 {
                let ti = interp_at[interp_idx];
                let theta = ((ti - t) / dt_eff).clamp(0.0, 1.0);
                let ui: Vec<T> = (0..n)
                    .map(|j| {
                        if let Some(h) = &hook {
                            h.interp(theta, dt_eff, j);
                        }
                        stepper.interpolate_component(theta, &u, dt_eff, j)
                    })
                    .collect();
                if let Some(h) = &hook {
                    h.interp_done();
                }
                interp_results.push(SolPointG { t: ti, u: ui });
                interp_idx += 1;
            }

            t += dt_eff;
            u.copy_from_slice(stepper.u_new());
            stepper.on_accept();
            if let Some(g) = g_end {
                g_old = g;
            }
            if let Some((k, pos)) = event {
                // The jump of every derivative lane (the values do not move), then a fresh
                // stepper: its start reads the field on its new arm, and the jumped lanes.
                if let Some(h) = &hook {
                    h.jump(k, pos, t);
                }
                stepper = make_stepper::<T>(n, method);
                stepper.set_dense_required(!interp_at.is_empty());
            }

            while save_idx < saveat.len() && (t - saveat[save_idx]).abs() < 1e-12 {
                if let Some(h) = &hook {
                    h.point(false);
                }
                results.push(SolPointG { t: saveat[save_idx], u: u.clone() });
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
            //
            // One-way (#2845, see `AUTO_SWITCH_PROBE_INTERVAL`): only a segment
            // still on the explicit stepper is re-probed or counted toward the next probe, so a
            // stiff segment is never demoted back to `rk45` on a single probe reading.
            let on_explicit = switching && method == OdeMethod::EXPLICIT_FALLBACK;
            if on_explicit {
                accepted_since_probe += 1;
            }
            let stage_escalation = on_explicit
                && stiffness_indicator.is_some_and(|indicator| auto_state.observe_rk45(indicator));
            // A verdict ratified by the step that reaches `tf` belongs to the next interval:
            // there is no remaining span for a replacement stepper to integrate here. Leave
            // it in `auto_state` so `integrate_resolved_g_inner` can consume it at the next
            // non-zero segment instead of reporting a switch that never took a step.
            if on_explicit && t < tf - 1e-15 {
                let periodic_probe = accepted_since_probe >= AUTO_SWITCH_PROBE_INTERVAL;
                let verdict = if stage_escalation {
                    Some(super::stiffness::stiff_method_for(opts))
                } else if periodic_probe {
                    Some(super::stiffness::resolve_method(rhs, &u, params, t, opts))
                } else {
                    None
                };
                if periodic_probe {
                    accepted_since_probe = 0;
                }
                // Only an escalation can pass this filter: the segment is on the explicit
                // stepper here, and both verdicts are either it or a stiff method.
                if let Some(verdict) = verdict.filter(|&verdict| verdict != method) {
                    method = verdict;
                    stepper = make_stepper::<T>(n, method);
                    stepper.set_dense_required(!interp_at.is_empty());
                    err_exp = stepper.err_exp();
                    // An escalation is an escalation whether the probe reached that verdict at
                    // the entry state or twenty-five steps in, so a switch onto a stiff method
                    // joins `auto_stiff_segments` too — which is what keeps the guard's "N of M
                    // escalations were discarded" count adding up. A segment that started stiff
                    // was counted by the caller, and never reaches this switch.
                    if let Some(s) = stats.as_deref_mut() {
                        s.auto_switched_segments += 1;
                        s.auto_stiff_segments += 1;
                    }
                    auto_state.reset_rk45_verdict();
                    // The new stepper gets the whole abort budget. `stiff_abort_after` bounds
                    // how long *a method* is allowed to grind on a segment it cannot step, and
                    // the method just changed; the clamps behind this point measured the one
                    // that was replaced. The switch is one-way, so this is at most one extra
                    // budget per segment.
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
            && opts.stiff_abort_after.is_some_and(|budget| min_step_clamps >= budget.max(1))
        {
            // A budgeted stop on a state that has already gone non-finite is a divergence, and
            // is counted as one below (`diverged_segments`) rather than here: its tail is
            // `NaN`, not freeze-padded, so the two counters partition the stopped segments.
            if let Some(s) = stats.as_deref_mut() {
                if u.iter().all(|v| v.plain().is_finite()) {
                    s.stiff_aborted_segments += 1;
                }
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

    // A segment that stopped short with a non-finite component in its carried state diverged
    // (#1539). Read by value only, like every other driver decision, so the `f64` and dual
    // solves agree on it.
    let unfinished = t < tf - 1e-15;
    let diverged = unfinished && u.iter().any(|v| !v.plain().is_finite());
    // Written once here rather than inside `record`, which cannot see which stepper is active.
    if let Some(s) = stats.as_deref_mut() {
        s.stiff_min_step_clamped_steps += stiff_min_step_clamps;
        if unfinished {
            s.unfinished_segments += 1;
        }
        if diverged {
            s.diverged_segments += 1;
        }
    }
    auto_state.accepted_since_probe = accepted_since_probe;

    // Fill any remaining saveat / interp times the integrator did not reach. A stop for a
    // finite reason (the step budget, `stiff_abort_after`, an unusable attempt on a finite
    // state) pads with the last state, as decided in #603 / #959. A diverged stop pads with
    // `NaN` in every component, jets included (#1539): the RK stages couple every component
    // through `u`, so once one state is non-finite no component past that point is an
    // integrated value — and a decoupled, observed state frozen at a finite value is exactly
    // the prediction nothing downstream can tell from a real one.
    // Built only when a time is left to pad: a segment that reached every requested time
    // (the usual one) builds no tail.
    let tail: Vec<T> = if save_idx >= saveat.len() && interp_idx >= interp_at.len() {
        Vec::new()
    } else if diverged {
        // `v · NaN` rather than `from_f64(NaN)`: a dual constant carries zero jets, while the
        // product rule puts `NaN` into the gradient and the Hessian block as well.
        u.iter().map(|&v| v * T::from_f64(f64::NAN)).collect()
    } else {
        u.clone()
    };
    while save_idx < saveat.len() {
        if let Some(h) = &hook {
            h.tail(false, diverged);
        }
        results.push(SolPointG { t: saveat[save_idx], u: tail.clone() });
        save_idx += 1;
    }
    while interp_idx < interp_at.len() {
        if let Some(h) = &hook {
            h.tail(true, diverged);
        }
        interp_results.push(SolPointG { t: interp_at[interp_idx], u: tail.clone() });
        interp_idx += 1;
    }

    (results, interp_results)
}

/// Integrate an ODE system from `t_start` to `t_end`, returning the state at each `saveat`
/// time. Uses whichever stepper [`OdeSolverOptions::method`] selects.
///
/// [`OdeMethod::Linear`] is integrated as [`OdeMethod::Auto`] here, as on every public
/// `solve_ode*` entry point: `rhs` is an arbitrary closure no parser has certified, and the
/// exact propagator's runtime self-check samples finitely many points, so it cannot tell a
/// right-hand side that is affine from one that only agrees with an affine map where it looks
/// (a generator that varies in time between its sample points, say). The exact path is taken
/// only for a model whose `[odes]` block the parser certified (#852).
pub fn solve_ode(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64], t_span: (f64, f64), params: &[f64],
    saveat: &[f64], opts: &OdeSolverOptions,
) -> Vec<SolPoint> {
    solve_ode_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode`] for a caller holding options from
/// [`OdeSpec::effective_solver_opts`](crate::ode::predictions::OdeSpec) on the RHS that spec
/// built — the one place [`OdeMethod::Linear`] is honoured, because only there did the parser
/// certify the system linear.
pub fn solve_ode_certified(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64], t_span: (f64, f64), params: &[f64],
    saveat: &[f64], opts: &OdeSolverOptions,
) -> Vec<SolPoint> {
    solve_ode_dense(rhs, u0, t_span, params, saveat, &[], opts, None).0
}

/// [`solve_ode`] with optional adaptive-step instrumentation.
///
/// The counters are intentionally local to this integration segment. Higher layers that split
/// by dose/observation boundaries can aggregate across calls to classify a full subject or fit.
pub fn solve_ode_with_stats(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64], t_span: (f64, f64), params: &[f64],
    saveat: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPoint> {
    // Uncertified closure: never the exact path (see `solve_ode`).
    let opts = &opts.for_integration();
    solve_ode_dense(rhs, u0, t_span, params, saveat, &[], opts, stats).0
}

/// [`solve_ode`] with **dense soft-sampling**: `(hard, soft)` where `hard` is the state at each
/// `saveat` time and `soft` the interpolated state at each `interp_at` time.
///
/// The soft channel reads inside an already-accepted step (#570), so a joint PK-TTE fit's
/// Gaussian predictions are identical whether or not the hazard readout is requested — for
/// every method, since the interpolation comes from the [`Stepper`].
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_dense(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64], t_span: (f64, f64), params: &[f64],
    saveat: &[f64], interp_at: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPoint>, Vec<SolPoint>) {
    let mut auto_state = OdeAutoSwitchState::default();
    solve_ode_dense_with_auto_state(
        rhs,
        u0,
        t_span,
        params,
        saveat,
        interp_at,
        opts,
        stats,
        &mut auto_state,
    )
}

/// Dense solve that carries the cheap AutoSwitch detector across event intervals
/// belonging to the same subject. Stepper workspaces are still rebuilt because
/// an event may invalidate their stages and Jacobian.
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_dense_with_auto_state(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64], t_span: (f64, f64), params: &[f64],
    saveat: &[f64], interp_at: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
    auto_state: &mut OdeAutoSwitchState,
) -> (Vec<SolPoint>, Vec<SolPoint>) {
    let (hard, soft) =
        integrate_resolved_g(rhs, u0, t_span, params, saveat, interp_at, opts, stats, auto_state);
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
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u: &mut [f64], t_span: (f64, f64),
    params: &[f64], opts: &OdeSolverOptions, monitor: usize, threshold: f64,
) -> ThresholdCrossing {
    solve_ode_until_threshold_labelled(rhs, u, t_span, params, opts, monitor, threshold, None)
}

/// [`solve_ode_until_threshold`] with display labels for the states, used in a `Failed`
/// message in place of the bare 0-based index (#2087). `labels[k]` is printed as given, so the
/// caller decides the rendering (a backticked `[odes]` name, or a phrase for an injected
/// accumulator). `None` is the public driver, byte for byte.
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_until_threshold_labelled(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u: &mut [f64], t_span: (f64, f64),
    params: &[f64], opts: &OdeSolverOptions, monitor: usize, threshold: f64,
    labels: Option<&[String]>,
) -> ThresholdCrossing {
    // The crossing search bisects a step's continuous extension and has no exact counterpart,
    // so a linear model's search is stepped exactly as the default's is.
    let opts = &opts.for_integration();
    // The explicit re-solve pins its method, which also takes mid-segment switching off it:
    // a fallback that could escalate again would be the escalation it is undoing.
    let pinned_explicit = OdeSolverOptions { method: OdeMethod::EXPLICIT_FALLBACK, ..*opts };
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
            labels,
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
        return until_threshold_with(
            rhs, u, t_span, params, opts, monitor, threshold, method, labels,
        )
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
    let run =
        until_threshold_with(rhs, u, t_span, params, opts, monitor, threshold, method, labels);

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
        labels,
    )
    .outcome
}

/// `solve_ode_until_threshold` with the stepper already chosen — the body both the probed
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
pub fn failed(msg: String, ran_stiff: bool) -> ThresholdRun {
    ThresholdRun {
        outcome: ThresholdCrossing::Failed(msg),
        stiff_min_step_clamped: true,
        ran_stiff,
    }
}

/// What one pass of [`until_threshold_with`] produced, for the guard above it.
pub struct ThresholdRun {
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
pub fn until_threshold_with(
    rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u: &mut [f64], t_span: (f64, f64),
    params: &[f64], opts: &OdeSolverOptions, monitor: usize, threshold: f64, method: OdeMethod,
    labels: Option<&[String]>,
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
    let mut ran_stiff = method != OdeMethod::EXPLICIT_FALLBACK;

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            return ended(ThresholdCrossing::ReachedEnd, stiff_min_step_clamped, ran_stiff);
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
                        "{} became non-finite at t={:.6}",
                        monitored_label(labels, monitor),
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
                        "{} decreased ({y0:.6} → {y1:.6}) over \
                         [{t:.6}, {:.6}]: non-monotone accumulator (negative rate / hazard)",
                        monitored_label(labels, monitor),
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
            // the stepper that produced it. One-way, as there (#2845): a span already on a
            // stiff stepper is not re-probed, so it is never demoted back to `rk45`.
            accepted_since_probe += 1;
            if switching
                && method == OdeMethod::EXPLICIT_FALLBACK
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
        if opts.stiff_abort_after.is_some_and(|budget| clamps >= budget.max(1)) {
            return failed(
                with_nonfinite_state_note(
                    u,
                    t,
                    labels,
                    format!(
                        "aborted at t={t:.6} after {clamps} min-dt clamp(s) \
                         (ode_stiff_abort_after): the segment is stability-limited"
                    ),
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
        with_nonfinite_state_note(
            u,
            t,
            labels,
            format!(
                "step budget ({}) exhausted before reaching t_end={tf:.6} (reached t={t:.6})",
                opts.max_steps
            ),
        ),
        ran_stiff,
    );
}

/// Leads a step-budget / stiff-abort `Failed` message with the state(s) that went non-finite
/// (#1578). The monitored-state guard only reads `u[monitor]`, so an *unmonitored* state that
/// diverges is force-accepted at `min_dt` step after step until one of the two budgets runs
/// out, and the message would then blame the budget or stiffness for a diverged right-hand
/// side. The budget clause stays as the secondary fact. A finite `u` returns `clause`
/// unchanged, byte for byte.
///
/// With `labels` the states are named by their labels, comma-joined as the sentence's subject
/// (#2087), instead of the `state(s) <k>` index form that `None` prints.
pub fn with_nonfinite_state_note(
    u: &[f64], t: f64, labels: Option<&[String]>, clause: String,
) -> String {
    let bad: Vec<String> = u
        .iter()
        .enumerate()
        .filter(|(_, x)| !x.is_finite())
        .map(|(k, _)| state_label(labels, k))
        .collect();
    match (bad.len(), labels.is_some()) {
        (0, _) => clause,
        // A label is a full noun phrase (`` [odes] state `x` ``, "the cumulative hazard of
        // CMT=3"), so the labelled form carries no "state(s)" prefix of its own.
        (_, true) => format!("{} became non-finite before t={t:.6}; {clause}", bad.join(", ")),
        (1, false) => format!("state {} became non-finite before t={t:.6}; {clause}", bad[0]),
        (_, false) => {
            format!("states {} became non-finite before t={t:.6}; {clause}", bad.join(", "))
        }
    }
}

/// State `k` as a `Failed` message names it: its label when the caller supplied one, else the
/// 0-based index.
pub fn state_label(labels: Option<&[String]>, k: usize) -> String {
    labels.and_then(|l| l.get(k)).map_or_else(|| k.to_string(), String::clone)
}

/// The monitored state as the subject of a `Failed` sentence: `monitored state <k>` without
/// labels (the public driver's text), the bare label with them.
pub fn monitored_label(labels: Option<&[String]>, monitor: usize) -> String {
    match labels.and_then(|l| l.get(monitor)) {
        Some(label) => label.clone(),
        None => format!("monitored state {monitor}"),
    }
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
/// **Step-size control reads `.plain()` only** — the accept/reject decision and `dt` adaptation
/// depend on values, never on derivatives, so the derivative flows through a *fixed* step
/// sequence. (Adapting on a derivative norm would make the sensitivity inconsistent with the
/// prediction.)
pub fn solve_ode_g<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]), u0: &[T], t_span: (f64, f64), params: &[T],
    saveat: &[f64], opts: &OdeSolverOptions,
) -> Vec<SolPointG<T>> {
    solve_ode_g_with_stats(rhs, u0, t_span, params, saveat, opts, None)
}

/// [`solve_ode_g`] for a certified spec's options — the generic twin of
/// [`solve_ode_certified`], and like it the only `solve_ode_g*` route that honours
/// [`OdeMethod::Linear`].
pub fn solve_ode_g_certified<T: crate::sens::num::PkNum>(
    rhs: &RhsG<'_, T>, u0: &[T], t_span: (f64, f64), params: &[T], saveat: &[f64],
    opts: &OdeSolverOptions,
) -> Vec<SolPointG<T>> {
    let mut auto_state = OdeAutoSwitchState::default();
    integrate_resolved_g(rhs, u0, t_span, params, saveat, &[], opts, None, &mut auto_state).0
}

/// [`solve_ode_g`] with optional adaptive-step instrumentation.
pub fn solve_ode_g_with_stats<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]), u0: &[T], t_span: (f64, f64), params: &[T],
    saveat: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
) -> Vec<SolPointG<T>> {
    solve_ode_g_dense(rhs, u0, t_span, params, saveat, &[], opts, stats).0
}

/// Generic solve that carries AutoSwitch detector history across the event
/// intervals of one prediction/sensitivity walk.
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_g_with_auto_state<T: crate::sens::num::PkNum>(
    rhs: &RhsG<'_, T>, u0: &[T], t_span: (f64, f64), params: &[T], saveat: &[f64],
    opts: &OdeSolverOptions, auto_state: &mut OdeAutoSwitchState,
) -> Vec<SolPointG<T>> {
    integrate_resolved_g(rhs, u0, t_span, params, saveat, &[], opts, None, auto_state).0
}

/// [`solve_ode_g`] with dense soft-sampling — the sensitivity-carrying twin of
/// `solve_ode_dense`. Available for every method because the interpolation comes from the
/// `Stepper`, so an analytic-gradient consumer that needs an in-step readout (a hazard state
/// at an event time, say) has one to call.
#[allow(clippy::too_many_arguments)]
pub fn solve_ode_g_dense<T: crate::sens::num::PkNum>(
    rhs: &dyn Fn(&[T], &[T], f64, &mut [T]), u0: &[T], t_span: (f64, f64), params: &[T],
    saveat: &[f64], interp_at: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    let mut auto_state = OdeAutoSwitchState::default();
    let rhs = rhs_of_f64_time(rhs);
    integrate_resolved_g(
        &rhs,
        u0,
        t_span,
        params,
        saveat,
        interp_at,
        // Uncertified closure: never the exact path (see `solve_ode`).
        &opts.for_integration(),
        stats,
        &mut auto_state,
    )
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    /// Test seam: while set, every certified-linear segment integrated on this thread is
    /// declined by the matrix exponential and integrated instead, as a real decline would be
    /// (#2143 review). A model file cannot produce a decline that does not also diverge, and
    /// the stagnation guard's decrement has to be seen refusing one. Per thread, so a test
    /// sets it inside a one-worker pool and no other test can see it.
    pub static FORCE_LINEAR_DECLINE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
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
pub struct SolverStatsScope {
    prev: Option<OdeSolverStats>,
}

impl SolverStatsScope {
    /// Begin collecting on this thread.
    pub fn enter() -> Self {
        Self { prev: STATS_SINK.with(|c| c.replace(Some(OdeSolverStats::default()))) }
    }

    /// Counters accumulated since [`enter`](Self::enter).
    pub fn collected(&self) -> OdeSolverStats {
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
pub fn stats_sink_active() -> bool {
    STATS_SINK.with(|c| c.get()).is_some()
}

/// Add one integration's counters to the active scope, if any.
#[inline]
pub fn record_to_stats_sink(stats: &OdeSolverStats) {
    STATS_SINK.with(|c| {
        if let Some(mut acc) = c.get() {
            acc.merge(stats);
            c.set(Some(acc));
        }
    });
}

/// Note that a prediction walk was abandoned before integrating, because its timeline could
/// not be ordered ([`OdeSolverStats::abandoned_non_finite_timeline`], #1234).
///
/// Reached through [`crate::ode::predictions::abandon_non_finite_timeline`], never called
/// directly by a guard — the predicate and this record are one call so a ninth guard cannot be
/// written without the counter.
///
/// **Both channels, because the guard sits before the only place that normally tees them.**
/// A walk's counters leave by two routes: the thread-local [`SolverStatsScope`], which is how
/// production collects (`api::fit`), and the caller's own `stats` out-parameter, which is how
/// [`crate::ode::ode_predictions_with_solver_stats`] — the one public getter for an
/// `OdeSolverStats` — collects. The ordinary route into both is the tee inside
/// [`integrate_resolved_g`], and this event is precisely the one where no integration is ever
/// reached: the guard returns before a driver is called. Recording into the sink alone left
/// the public getter reporting `0` for an abandoned walk, which is the exact conflation this
/// counter exists to remove, on the surface #1234 reported it against.
///
/// Off the diagnostic path this is one `Cell` read, taken only inside a guard that has already
/// fired on a subject whose predictions are `NaN` regardless.
#[inline]
pub fn record_abandoned_non_finite_timeline(stats: Option<&mut OdeSolverStats>) {
    // One `OdeSolverStats` through the two ordinary merge paths, rather than a second
    // hand-written `STATS_SINK` read-modify-write next to `record_to_stats_sink`'s: this is
    // also what makes `merge`'s `abandoned_non_finite_timeline` line reachable, since nothing
    // else ever produces a non-scope `OdeSolverStats` carrying the field.
    let one = OdeSolverStats { abandoned_non_finite_timeline: 1, ..Default::default() };
    record_to_stats_sink(&one);
    if let Some(s) = stats {
        s.merge(&one);
    }
}

/// `n` segments of the exact linear path a compiled plan integrates instead, into the open
/// [`SolverStatsScope`] (#2786): its recording declined them, and the plan replays that.
pub fn note_linear_declines(n: usize) {
    if stats_sink_active() {
        record_to_stats_sink(&OdeSolverStats { linear_declined_segments: n, ..Default::default() });
    }
}

/// `NaN` in every component, jets included, at every `saveat` and `interp_at` time: what a
/// segment entered with a non-finite state returns without stepping (#2788). `v · NaN`, as in
/// `integrate_dense_g`'s diverged tail, so a dual's gradient and Hessian block are `NaN` too.
pub fn nan_trajectory<T: PkNum>(
    u0: &[T], saveat: &[f64], interp_at: &[f64],
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    let nan: Vec<T> = u0.iter().map(|&v| v * T::from_f64(f64::NAN)).collect();
    let at = |times: &[f64]| -> Vec<SolPointG<T>> {
        times.iter().map(|&t| SolPointG { t, u: nan.clone() }).collect()
    };
    (at(saveat), at(interp_at))
}

/// [`integrate_resolved_g_inner`] with the thread-local [`SolverStatsScope`] tee.
///
/// Off the diagnostic path this is one `Cell` read per segment and the same call as before.
#[allow(clippy::too_many_arguments)]
pub fn integrate_resolved_g<T: PkNum>(
    rhs: &RhsG<'_, T>, u0: &[T], t_span: (f64, f64), params: &[T], saveat: &[f64],
    interp_at: &[f64], opts: &OdeSolverOptions, stats: Option<&mut OdeSolverStats>,
    auto_state: &mut OdeAutoSwitchState,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    // A recording type's solve (#2786) records a plan, which happens or not by what the plan
    // cache holds: not an integration the counters describe.
    if !stats_sink_active() || T::RECORDS_BRANCHES {
        return integrate_resolved_g_inner(
            rhs, u0, t_span, params, saveat, interp_at, opts, stats, auto_state,
        );
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
        auto_state,
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
/// caller gets. When that fallback finishes cleanly, this bounds `auto` to "the explicit
/// answer, twice the cost"; when it does not, [`OdeSolverStats::auto_fallback_failed`] exposes
/// that both attempts failed instead of presenting the retry as a successful repair.
///
/// The guard is scoped to escalations. A named `ode_method` is honoured exactly as before,
/// failed result included: a user who asked for `rodas4` gets `rodas4`, and a fixed method that
/// silently re-solved as something else would be the worse surprise.
///
/// Generic over `T`, so the `f64` prediction and the `Dual2` sensitivity solve share one copy
/// of both the resolution and the guard. Every trigger but one reads `.plain()` only and sees
/// the same values, so the two make the same decision on the same segment and the gradient
/// keeps differentiating the trajectory the predictor reports.
///
/// The exception is the jet-finiteness clause (#1204), which no `f64` solve can fail. It is
/// the deliberate one-directional break in that invariant: a dual solve whose jets have
/// overflowed has nothing to differentiate, so keeping it on the predictor's stepper buys a
/// `NaN` gradient rather than consistency. Nothing else in the guard is allowed to branch on
/// a derivative.
#[allow(clippy::too_many_arguments)]
pub fn integrate_resolved_g_inner<T: PkNum>(
    rhs: &RhsG<'_, T>, u0: &[T], t_span: (f64, f64), params: &[T], saveat: &[f64],
    interp_at: &[f64], opts: &OdeSolverOptions, mut stats: Option<&mut OdeSolverStats>,
    auto_state: &mut OdeAutoSwitchState,
) -> (Vec<SolPointG<T>>, Vec<SolPointG<T>>) {
    // The driver below reads values; a recording reaches it only through `T::ode_segment`.
    // A type that records without one (a nested dual over a recording) cannot be integrated:
    // a compile error at the instantiation, not a failed recording at run time.
    const {
        assert!(
            T::ODE_DRIVER_OK,
            "the ODE driver cannot be instantiated over a nested dual of a recording"
        )
    };
    // A segment handed a non-finite state has nothing to integrate (#2788). Every stepper
    // couples the components through `u`, so its error norm is `NaN` from the first attempt:
    // the step shrinks to `min_dt`, force-accepts `MAX_CONSECUTIVE_MIN_STEP_CLAMPS` times and
    // stops diverged with a `NaN` tail (#1539) — after a stiffness probe that may have put it on
    // an implicit stepper and the escalation guard's explicit re-solve on top. The exact linear
    // path declines it at its self-check first. That is the same all-`NaN` result at the cost
    // of hundreds of RHS evaluations, and an inner line search probing an absurd η reaches it
    // on every later segment of the subject: measured on the melphalan 2-cpt model at a FOCEI
    // plateau, 64% of the declined linear segments started all-`NaN`. Return that result
    // directly, counted as the stepper counted it — an unfinished segment that diverged — so
    // the fit's `ode_solver` warning reads the same. A zero-length span is left alone: no
    // stepper runs there, and it returns `u0` as it was.
    if (t_span.1 - t_span.0).abs() >= 1e-15
        && u0.iter().any(|&v| {
            // A recordable test (`val` would fail a recording), `false` on `NaN`.
            use crate::sens::num::BranchCmp::{Gt, Lt};
            !(v.br_cmp_c(Gt, f64::NEG_INFINITY) && v.br_cmp_c(Lt, f64::INFINITY))
        })
    {
        if let Some(s) = stats.as_deref_mut() {
            s.unfinished_segments += 1;
            s.diverged_segments += 1;
        }
        return nan_trajectory(u0, saveat, interp_at);
    }
    // A certified-linear model's segment is propagated exactly (#852). Every integration entry
    // point funnels through here, so the dense segments, the steady-state run-in cycles and the
    // `Dual2`/`Dual2` sensitivity solves all take the same exact path without being threaded
    // individually. A segment it declines is integrated as `Auto` would integrate it.
    if opts.method == OdeMethod::Linear {
        if (t_span.1 - t_span.0).abs() >= 1e-15 {
            #[cfg(any(test, feature = "test-support"))]
            let declined_by_test = FORCE_LINEAR_DECLINE.get();
            #[cfg(not(any(test, feature = "test-support")))]
            let declined_by_test = false;
            if !declined_by_test {
                if let Some(out) =
                    super::linear::linear_segment_g(rhs, u0, t_span, params, saveat, interp_at)
                {
                    return out;
                }
            }
            if let Some(s) = stats.as_deref_mut() {
                s.linear_declined_segments += 1;
            }
            T::note_linear_decline();
        }
        return integrate_resolved_g_inner(
            rhs,
            u0,
            t_span,
            params,
            saveat,
            interp_at,
            &opts.for_integration(),
            stats,
            auto_state,
        );
    }
    // A recording type (#2786) records the segment as a call that runs the rest of this
    // function at run time, over steppers carrying its lanes.
    if T::RECORDS_BRANCHES {
        if let Some(out) =
            T::ode_segment(rhs, u0, t_span, params, saveat, interp_at, opts, auto_state)
        {
            return out;
        }
    }
    // A zero-length span returns `u0` at every requested time without stepping, so there is
    // no stepper to choose and no reason to pay for a Jacobian. Dosing timelines produce these
    // wherever two events share a time.
    let nonzero_span = (t_span.1 - t_span.0).abs() >= 1e-15;
    let method = if !nonzero_span {
        OdeMethod::EXPLICIT_FALLBACK
    } else if opts.method == OdeMethod::Auto
        && opts.auto_switch
        && auto_state.rk_stiff >= RK45_STIFFNESS_RATIFY
    {
        // A stage verdict may have been ratified by the final accepted step of the previous
        // event interval. The entry Jacobian is only a backstop and must not override that
        // accumulated, dimensionless evidence before a stiff stepper gets a chance to run.
        super::stiffness::stiff_method_for(opts)
    } else {
        super::stiffness::resolve_method(rhs, u0, params, t_span.0, opts)
    };
    let run = |method: OdeMethod,
               opts: &OdeSolverOptions,
               stats: Option<&mut OdeSolverStats>,
               auto_state: &mut OdeAutoSwitchState| {
        integrate_dense_g(
            method, rhs, u0, t_span, params, saveat, interp_at, opts, stats, auto_state,
        )
    };
    // The explicit re-solve the guard falls back to **pins** its method, which is also what
    // takes mid-segment switching off it (`integrate_dense_g` switches only under `auto`): a
    // re-solve that could escalate again would be the escalation the guard is undoing.
    let pinned_explicit = OdeSolverOptions { method: OdeMethod::EXPLICIT_FALLBACK, ..*opts };

    // Not a guarded solve — a named method, or `auto` reading the system as non-stiff *and*
    // unable to change its mind later. Nothing to discard, and the caller's counters are
    // written directly by the one and only solve.
    let may_switch = opts.method == OdeMethod::Auto && opts.auto_switch;
    if opts.method != OdeMethod::Auto || (method == OdeMethod::EXPLICIT_FALLBACK && !may_switch) {
        return run(method, opts, stats, auto_state);
    }

    // An escalation — at the segment's start, or reached mid-segment by a re-probe — is scored
    // on its own counters so the guard can read what *this* attempt did, rather than a total
    // the caller has been accumulating across segments.
    let escalated_at_start = method != OdeMethod::EXPLICIT_FALLBACK;
    let mut attempt = OdeSolverStats {
        auto_stiff_segments: usize::from(escalated_at_start),
        ..Default::default()
    };
    let auto_state_entry = *auto_state;
    let out = run(method, opts, Some(&mut attempt), auto_state);

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
    //
    // A third failure shape, and the one neither test above can see (#1080 review): a stiff
    // method can exhaust `max_steps` without ever reaching `min_dt`, which returns a finite,
    // freeze-padded, partially-integrated trajectory with every other counter at zero. Reject
    // it like the other two. Left unguarded it was indistinguishable from an unfinished
    // *explicit* solve, whose remedy is the opposite one — raise `max_steps` or loosen the
    // tolerances, rather than name a different stiff method.
    //
    // A fourth shape, and the only one that is invisible to the `f64` prediction solve
    // (#1204): every saved *value* is finite while the derivative jets are not. A dual's
    // jets carry higher powers of what its value carries linearly — `∂u/∂p = t·u` and
    // `∂²u/∂p² = t²·u` for `u' = p·u`, `1/x²` and `1/x³` against `recip`'s `1/x` — so a
    // trajectory driven near the top of double precision overflows its Hessian first, its
    // gradient next, and its value not at all. Measured on the repro in
    // `stiffness_tests.rs`: value `7.2e303`, gradient `1.4e306`, Hessian `NaN`, with zero
    // clamps, no unfinished segment, and a *successful* stiff solve. Every other trigger
    // here reads clean and FOCEI gets a `NaN` gradient out of a solve that reported success.
    //
    // Guarding it costs the invariant this function otherwise maintains — that the `f64`
    // and dual solves choose the same stepper — and does so knowingly, in one direction
    // only. `jets_finite` is `true` for `f64`, so the prediction path never rejects on this
    // clause and the asymmetry is confined to the case where the dual solve's result is
    // *already* unusable: the choice is between a gradient from the explicit re-solve of a
    // trajectory the predictor integrated stiffly (wrong by solver tolerance) and a `NaN`
    // (wrong by everything). The rejection is counted separately, in
    // `auto_stiff_rejected_jets`, precisely so the asymmetry is visible rather than hidden
    // inside the ordinary rejection count.
    let ran_stiff = attempt.auto_stiff_segments > 0;
    let stalled = attempt.stiff_min_step_clamped_steps > 0;
    let unfinished = attempt.unfinished_segments > 0;
    let values_finite = finite_g(&out.0) && finite_g(&out.1);
    // Scored only when the segment *started* with finite jets. Once a state's jets overflow
    // they stay overflowed for the rest of the subject's event chain, so without this every
    // downstream segment would fail the clause, pay a second solve that cannot repair
    // anything it did not break, and add another count for one overflow. The clause is about
    // a segment that lost its derivatives, not about one handed them already gone.
    let entry_jets_finite = u0.iter().all(|v| v.jets_finite());
    let jets_finite = !entry_jets_finite || (jets_finite_g(&out.0) && jets_finite_g(&out.1));
    let usable = !ran_stiff || (!stalled && !unfinished && values_finite && jets_finite);
    if !usable {
        attempt.auto_stiff_rejected = 1;
        // The jet-only case: everything the `f64` path can see was clean, and this rejection
        // exists solely because the dual jets were not. Scored apart so a reader can tell a
        // repair the prediction solve agreed with from one it did not (and would not have
        // made), which is the difference between "the stiff method failed" and "the stiff
        // method succeeded at a scale the sensitivities cannot represent".
        if !stalled && !unfinished && values_finite {
            attempt.auto_stiff_rejected_jets = 1;
        }
        // The clamps were taken, so they stay in `min_step_clamped_steps`; they are *also*
        // recorded as discarded, because the trajectory they damaged is about to be thrown
        // away and re-solved. Without this split every rejected escalation would read, to a
        // caller, as "the answer you got was freeze-padded" — the opposite of what the guard
        // just did. For the same reason a budgeted abort inside a discarded attempt is not a
        // truncated *result*: it truncated an attempt nobody receives, so it is not counted.
        attempt.discarded_clamped_steps = attempt.min_step_clamped_steps;
        attempt.discarded_unfinished_segments = attempt.unfinished_segments;
        attempt.stiff_aborted_segments = 0;
        attempt.diverged_segments = 0;
    }
    // The stiff attempt's steps stay counted either way — they were taken, and they cost what
    // they cost. A fit that reads slow *and* shows a rejection is paying for both solves.
    if let Some(s) = stats.as_deref_mut() {
        s.merge(&attempt);
    }
    if usable {
        return out;
    }
    // The discarded attempt's detector votes describe a trajectory the caller
    // will never receive. Restore the entry history before the pinned fallback.
    *auto_state = auto_state_entry;
    // Score the explicit re-solve separately too. Returning the rejected stiff attempt is no
    // better when this one fails, and a third solve would only guess at another method, so the
    // explicit result remains the deterministic fallback — but it must not be returned as if
    // the guard repaired the segment. `auto_fallback_failed` is the production-visible "both
    // attempts failed" signal requested in #1080.
    //
    // The two attempts are scored on deliberately different criteria. The stiff attempt is
    // rejected on *any* stiff `min_dt` clamp, because its clamps are recorded as discarded and
    // would otherwise leave no trace a caller could read; the fallback is scored on the
    // returned trajectory only (unfinished, or non-finite), because its clamps are kept and
    // reported directly through `min_step_clamped_steps` / `kept_clamped_steps`. A fallback
    // that clamps its way to `tf` is the ordinary stability-limited outcome that clause already
    // describes, not a second failure to re-report here.
    let mut fallback = OdeSolverStats::default();
    let fallback_out =
        run(OdeMethod::EXPLICIT_FALLBACK, &pinned_explicit, Some(&mut fallback), auto_state);
    // Scored on the same two finiteness questions as the attempt, jets included (#1204 item
    // 4): an explicit re-solve that returns finite values with `NaN` jets has not repaired
    // the gradient the guard discarded the escalation to repair, and reporting it as a
    // successful fallback would be the same silence one layer down.
    if fallback.unfinished_segments > 0
        || !finite_g(&fallback_out.0)
        || !finite_g(&fallback_out.1)
        || (entry_jets_finite
            && (!jets_finite_g(&fallback_out.0) || !jets_finite_g(&fallback_out.1)))
    {
        fallback.auto_fallback_failed = 1;
    }
    if let Some(s) = stats.as_deref_mut() {
        s.merge(&fallback);
    }
    fallback_out
}

/// Whether every saved state is finite, reading values only (`T = Dual2` decides identically
/// to `T = f64`, which is what keeps the two paths on the same stepper).
/// Non-recording: the adaptive driver decides on the value lane. A recording type never reaches it: `T::ode_segment` records the segment as a loop call (the f64 driver runs at run time) and returns before this code, except after the recording has already failed.
pub fn finite_g<T: PkNum>(points: &[SolPointG<T>]) -> bool {
    points.iter().all(|p| p.u.iter().all(|v| v.plain().is_finite()))
}

/// Whether every saved state's **derivative jets** are finite, reading the jets only.
///
/// The deliberate counterpart to [`finite_g`]: at `T = f64` this is a compile-time `true`
/// (there are no jets), so the prediction path decides exactly as it did before #1204 and
/// only a `Dual2`/`Dual2` solve can fail it. See the escalation guard for why that
/// asymmetry is wanted rather than tolerated.
pub fn jets_finite_g<T: PkNum>(points: &[SolPointG<T>]) -> bool {
    points.iter().all(|p| p.u.iter().all(|v| v.jets_finite()))
}
