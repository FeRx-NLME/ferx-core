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
    /// This is a *starting* decision per integrated segment, not a mid-step switch: the
    /// method is still fixed for the duration of one segment, so every guarantee the other
    /// variants carry (dense output, event root-finding, `Dual2` sensitivities) is unchanged.
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
    /// Integrations that [`OdeMethod::Auto`] started on a stiff method because the a-priori
    /// probe read `|Re λ|max ≥` [`crate::ode::stiffness::STIFF_RE_LAMBDA_THRESHOLD`].
    ///
    /// Zero under every explicitly-named `ode_method`, so a non-zero value means `auto` was
    /// asked and escalated. It is a floor, not a total: [`solve_ode_until_threshold`] takes no
    /// stats, so escalations on the event-time path (TTE / adaptive dosing) are not counted
    /// here.
    pub auto_stiff_segments: usize,
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
            auto_stiff_segments,
            auto_stiff_rejected,
        } = *other;
        self.attempted_steps += attempted_steps;
        self.accepted_steps += accepted_steps;
        self.rejected_steps += rejected_steps;
        self.min_step_clamped_steps += min_step_clamped_steps;
        self.auto_stiff_segments += auto_stiff_segments;
        self.auto_stiff_rejected += auto_stiff_rejected;
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
#[allow(clippy::too_many_arguments)]
fn integrate_dense_g<T: PkNum>(
    stepper: &mut dyn Stepper<T>,
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
    stepper.set_dense_required(!interp_at.is_empty());
    let err_exp = stepper.err_exp();

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
            break;
        }

        let accepted = usable && (err_norm <= 1.0 || dt_eff <= opts.min_dt);
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
    // Both early exits below return without stepping, so probe after them rather than before.
    if u[monitor] >= threshold || (t_span.1 - t_span.0).abs() < 1e-15 {
        return until_threshold_with(
            rhs,
            u,
            t_span,
            params,
            opts,
            monitor,
            threshold,
            OdeMethod::EXPLICIT_FALLBACK,
        )
        .0;
    }
    let method = super::stiffness::resolve_method(rhs, u, params, t_span.0, opts);
    if method == OdeMethod::EXPLICIT_FALLBACK || opts.method != OdeMethod::Auto {
        return until_threshold_with(rhs, u, t_span, params, opts, monitor, threshold, method).0;
    }
    // Escalated by the probe — the same guard `integrate_resolved_g` applies, in the shape
    // this driver reports failure in. A stiff method that cannot form a step, or that
    // drives the monitored accumulator non-finite or backwards, comes back as `Failed`; the
    // segment is then re-run explicitly from the *entry* state, since `u` is unspecified after
    // a failure. A censored draw laundered out of a diverged stiff solve would be the worst
    // possible outcome here, so the retry is not optional.
    let u_entry = u.to_vec();
    let (outcome, clamped) =
        until_threshold_with(rhs, u, t_span, params, opts, monitor, threshold, method);

    // Both halves of the dense driver's guard, in the shape this driver reports outcomes in.
    //
    // `Failed` is the loud half. The quiet half is a **clamp**: this driver force-accepts at
    // `min_dt` and keeps going, so a stiff method that could not form a step still returns an
    // ordinary-looking `Crossed`/`ReachedEnd` built on a trajectory it did not actually
    // integrate. That is the same freeze-pad shape (#959) the dense guard was widened for, and
    // here it means a wrong simulated event time — or, worse, a censoring laundered out of a
    // diverged solve. Nothing downstream can tell; the flag is the only trace.
    if !matches!(outcome, ThresholdCrossing::Failed(_)) && !clamped {
        return outcome;
    }
    // `u` is unspecified after a failure and untrustworthy after a stall, so the re-solve
    // starts from the entry state rather than from wherever the escalation left it.
    u.copy_from_slice(&u_entry);
    until_threshold_with(
        rhs,
        u,
        t_span,
        params,
        opts,
        monitor,
        threshold,
        OdeMethod::EXPLICIT_FALLBACK,
    )
    .0
}

/// [`solve_ode_until_threshold`] with the stepper already chosen — the body both the probed
/// and the re-solved pass run through.
///
/// Reports `(outcome, min_step_clamped)`. The second half is what lets the caller's guard see a
/// stall: a clamped step here is force-accepted and the run continues, so a stalled solve
/// returns a perfectly ordinary-looking `Crossed`/`ReachedEnd` that no inspection of the
/// outcome alone can distinguish from a healthy one.
/// Pair a hard failure with the clamp flag. A failure is a failure whether or not the solver
/// clamped on the way there, so the flag is irrelevant to the caller in this arm — but the
/// return type is one type, and spelling it out at eight call sites is noise.
#[inline]
fn failed(msg: String) -> (ThresholdCrossing, bool) {
    (ThresholdCrossing::Failed(msg), true)
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
) -> (ThresholdCrossing, bool) {
    let (t0, tf) = t_span;

    // Already at/above threshold at the span start (e.g. threshold ≈ 0 from u≈1, or a non-zero
    // initial accumulator). Catch before stepping so the loop's crossing test can assume
    // `y0 < threshold`.
    if u[monitor] >= threshold {
        return (ThresholdCrossing::Crossed(t0), false);
    }
    if (tf - t0).abs() < 1e-15 {
        return (ThresholdCrossing::ReachedEnd, false);
    }

    // Set when a step is force-accepted at `min_dt` (an explicit method that failed its error
    // test, or a linearly implicit one that could not form a step at all). The run continues
    // afterwards, so this is the only trace such a step leaves.
    let mut min_step_clamped = false;
    let mut stepper = make_stepper::<f64>(u.len(), method);
    // This driver localizes the crossing inside a step, so it always interpolates.
    stepper.set_dense_required(true);
    let err_exp = stepper.err_exp();
    let mut t = t0;
    let mut dt = opts.initial_dt.min((tf - t0) / 10.0).max(opts.min_dt);

    for _step in 0..opts.max_steps {
        if t >= tf - 1e-15 {
            return (ThresholdCrossing::ReachedEnd, min_step_clamped);
        }

        let dt_eff = dt.min(tf - t);
        let err_norm = stepper.attempt(rhs, u, params, t, dt_eff, opts);
        let usable = stepper.attempt_usable();
        if !usable && dt_eff <= opts.min_dt {
            min_step_clamped = true;
            return failed(format!(
                "the step at t={t:.6} could not be formed even at min_dt \
                 (singular Jacobian system)"
            ));
        }

        let accepted = usable && (err_norm <= 1.0 || dt_eff <= opts.min_dt);
        // Mirrors `OdeSolverStats::record`: a force-accept at `min_dt` whose error test did not
        // pass (`!(err_norm <= 1.0)`, which also catches a non-finite norm) is a clamp.
        if accepted && !(err_norm <= 1.0) && dt_eff <= opts.min_dt {
            min_step_clamped = true;
        }
        if accepted {
            let y0 = u[monitor];
            let y1 = stepper.u_new()[monitor];

            if !y1.is_finite() {
                return failed(format!(
                    "monitored state {monitor} became non-finite at t={:.6}",
                    t + dt_eff
                ));
            }
            // Scale-aware monotonicity floor: a real negative rate produces a decrease ≫ this;
            // only round-off sits below it.
            let mono_tol = scale_tol(opts.abstol, opts.reltol, y0, y1);
            if y1 < y0 - mono_tol {
                return failed(format!(
                    "monitored state {monitor} decreased ({y0:.6} → {y1:.6}) over \
                     [{t:.6}, {:.6}]: non-monotone accumulator (negative rate / hazard)",
                    t + dt_eff
                ));
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
                return (
                    ThresholdCrossing::Crossed(t + 0.5 * (lo + hi) * dt_eff),
                    min_step_clamped,
                );
            }

            t += dt_eff;
            u.copy_from_slice(stepper.u_new());
            stepper.on_accept();
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

    return failed(format!(
        "step budget ({}) exhausted before reaching t_end={tf:.6} (reached t={t:.6})",
        opts.max_steps
    ));
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
fn integrate_resolved_g<T: PkNum>(
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
    let run = |method: OdeMethod, stats: Option<&mut OdeSolverStats>| {
        let mut stepper = make_stepper::<T>(u0.len(), method);
        integrate_dense_g(
            stepper.as_mut(),
            rhs,
            u0,
            t_span,
            params,
            saveat,
            interp_at,
            opts,
            stats,
        )
    };

    // Not an escalation — a named method, or `auto` reading the system as non-stiff. Nothing
    // to guard, and the caller's counters are written directly by the one and only solve.
    if opts.method != OdeMethod::Auto || method == OdeMethod::EXPLICIT_FALLBACK {
        return run(method, stats);
    }

    // An escalation is scored on its own counters so the guard can read what *this* attempt
    // did, rather than a total the caller has been accumulating across segments.
    let mut attempt = OdeSolverStats {
        auto_stiff_segments: 1,
        ..Default::default()
    };
    let out = run(method, Some(&mut attempt));

    // Two ways an escalation goes wrong, and the second is the one that bites. A non-finite
    // trajectory is loud. A stiff method that clamps at `min_dt` is not: the driver stops and
    // freeze-pads the remaining saves with the last state, so the result comes back finite,
    // plausible, and wrong (#959) — measured at 4.8e282 on a system whose true solution had
    // already blown up. Both are the same failure to the caller, so both discard the result.
    //
    // A clamp means the *stiff* method could not form a step, which is the one thing it was
    // escalated to do; it is not the ordinary cost of a hard segment. So this trigger fires
    // where the escalation genuinely failed, not merely where it worked hard.
    let stalled = attempt.min_step_clamped_steps > 0;
    let usable = !stalled && finite_g(&out.0) && finite_g(&out.1);
    if !usable {
        attempt.auto_stiff_rejected = 1;
    }
    // The stiff attempt's steps stay counted either way — they were taken, and they cost what
    // they cost. A fit that reads slow *and* shows a rejection is paying for both solves.
    if let Some(s) = stats.as_deref_mut() {
        s.merge(&attempt);
    }
    if usable {
        return out;
    }
    run(OdeMethod::EXPLICIT_FALLBACK, stats)
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
