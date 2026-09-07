//! Neutral dose-resolution + steady-state-equilibration primitives.
//!
//! These were extracted from `ode/predictions.rs` so that `pk/`, `sens/`, and
//! `api` no longer depend *upward* on `ode/` merely to resolve modeled-`RATE`
//! doses or to share the SS-equilibration convergence tracker. Pure code motion.
//! `ode::predictions` *privately* imports the symbols it still uses (it does NOT
//! re-export them as `crate::ode::…`, so the upward dependency stays removed);
//! every other consumer reaches these through `crate::dosing::…`.
//!
//! - Dose resolution (`resolve_subject_doses{,_with}`): the #324 single source of
//!   truth for turning modeled-`RATE` (`RATE=-2` → modeled duration `D{cmt}`)
//!   doses into concrete `Fixed` doses.
//! - Exact periodic steady state (`periodic_ss_fixed_point_g`): the engine-neutral
//!   `u_ss = (I − M)⁻¹·b` solve of a linear one-cycle map, shared by the ODE
//!   input-rate path (#835) and the analytical event-driven walk (#908).
//! - SS equilibration (`SS_EQUILIBRATION_CYCLES`, `SS_EQUILIBRATION_TOL`,
//!   `ss_cycle_converged`, `SsStopTracker`, and the test-only cycle recorder):
//!   the truncated-pulse-train *fallback* policy, used when the exact solve
//!   declines, shared by the analytical, ODE, and sensitivity SS loops so their
//!   troughs can't drift apart.

use crate::types::{DoseEvent, Subject};
use std::borrow::Cow;

/// Resolve any modeled-`RATE` doses (#324, e.g. `RATE=-2` → modeled duration
/// `D{cmt}`) in `subject` to concrete (`Fixed`) doses. `pk_for_dose(k)` supplies
/// the per-dose `PkParams::values` slice used to evaluate dose `k`'s modeled
/// parameter — pass a constant closure for the no-TV-covariate paths (see
/// [`resolve_subject_doses`]) or `|k| &pk_at_dose[k].values` for the per-dose
/// event-driven path. Returns the subject **borrowed** (no allocation) when every
/// dose is already `Fixed` (the common case — see [`Subject::all_doses_fixed`]),
/// and an owned copy with resolved `doses` otherwise.
///
/// Single source of truth: every ODE entrypoint funnels its subject through this
/// (or the thin [`resolve_subject_doses`] wrapper) before building the dose
/// timeline, so the integrator and SS helpers only ever see a concrete
/// `rate`/`duration` and a coded `RATE=-2` cannot reach them unresolved.
///
/// The owned branch clones the whole `Subject`, not just `doses`, because the
/// downstream machinery ([`crate::pk::event_driven::EventSchedule::for_subject`],
/// the SS pre-equilibration, the break-time timeline) consumes a unified
/// `&Subject` and reads `obs_times` / `pk_only_times` / `reset_times` alongside
/// the resolved `doses`. Cloning only `doses` would force every one of those deep
/// helpers to take the resolved doses as a separate argument — the
/// "thread the resolved doses through every helper" design that was deliberately
/// rejected in favour of resolving once at the entrypoint. The clone is paid
/// only on the (uncommon) modeled-`RATE` path; the all-`Fixed` path is borrowed.
pub(crate) fn resolve_subject_doses_with<'a>(
    subject: &'a Subject,
    attr_map: &crate::types::DoseAttrMap,
    pk_for_dose: impl Fn(usize) -> &'a [f64],
) -> Cow<'a, Subject> {
    // Fast path: with no compartment-indexed attribute there can be no modeled
    // dose to resolve, so skip the per-dose `all_doses_fixed()` scan entirely —
    // the overwhelmingly common case (no `D{cmt}`). A modeled dose cannot reach
    // here with an empty map: it would have been rejected by the data gate first.
    if attr_map.is_empty() || subject.all_doses_fixed() {
        return Cow::Borrowed(subject);
    }
    let mut owned = subject.clone();
    for (k, d) in owned.doses.iter_mut().enumerate() {
        *d = d.resolve_rate(attr_map, pk_for_dose(k));
    }
    Cow::Owned(owned)
}

/// Resolve modeled-`RATE` doses using `params` for **every** dose — the
/// no-time-varying-covariate ODE paths, where the PK snapshot is constant across
/// doses. The event-driven / TV-covariate path calls
/// [`resolve_subject_doses_with`] directly with a per-dose closure. See
/// [`resolve_subject_doses_with`].
pub(crate) fn resolve_subject_doses<'a>(
    subject: &'a Subject,
    attr_map: &crate::types::DoseAttrMap,
    params: &'a [f64],
) -> Cow<'a, Subject> {
    resolve_subject_doses_with(subject, attr_map, |_| params)
}

/// For each slot of an ordered event timeline, the index of the **record that
/// governs the segment ending there** (#1073).
///
/// NONMEM evaluates `$PK` at every data record and then advances *to* that record,
/// so an interval is governed by the record that **terminates** it. Anything that is
/// not a record — a lagged dose arrival, an infusion end, a zero-order cutoff, a
/// per-route onset — supplies no parameters: it merely subdivides the interval its
/// enclosing record terminates, and every piece of that interval runs on that
/// record's snapshot.
///
/// `is_record(i)` reports whether timeline slot `i` is a record; the caller owns the
/// mapping from a returned index back to its own snapshot type (`PkParams`,
/// `PkDual<T>`, or a `&[T]` slice), which is why this resolves *indices* rather than
/// values — materialising a `PkDual<Dual2<N>>` per event would put ~4.7 KB × n_events
/// of copying on the FOCEI gradient hot path.
///
/// Two rules, and both must be identical in every engine — the four walks
/// (`ode::predictions`, `pk::event_driven`, `sens::propagate`, `sens::ode_provider`)
/// integrate the same subject, so a difference here is a silent cross-engine
/// divergence that `Dual2`-vs-FD parity cannot see (FD perturbs a twin's own value
/// path, so both sides move together):
///
/// 1. A record resolves to **itself**, and a non-record to the **next record ahead**
///    (one backward scan — the answer for a non-record lies ahead of it in the walk).
/// 2. Past the final record there is nothing ahead to terminate the segment, so it
///    keeps the **last record that ran** (a forward fill). Not observable through
///    predictions today — no observation follows such a boundary — but the four walks
///    previously split two-and-two between the last record and the dose row here, and
///    that becomes live the moment a trailing record, a reset, or a second pass reads
///    the state.
///
/// Returns `None` in every slot only when the timeline contains no record at all — a
/// subject with no observation, no EVID=2 row and no dose, which produces no
/// prediction.
pub(crate) fn governing_record_indices(
    n_events: usize,
    is_record: impl Fn(usize) -> bool,
) -> Vec<Option<usize>> {
    let mut acc: Vec<Option<usize>> = vec![None; n_events];
    // Rule 1: backward scan. `seen` is set *before* `acc[i]` is written, so a record
    // resolves to itself rather than to its successor.
    let mut seen: Option<usize> = None;
    for i in (0..n_events).rev() {
        if is_record(i) {
            seen = Some(i);
        }
        acc[i] = seen;
    }
    // Rule 2: forward fill the trailing tail (the only slots the backward scan left
    // `None`, since every slot at or before the final record saw it).
    let mut last: Option<usize> = None;
    for slot in acc.iter_mut() {
        match *slot {
            Some(q) => last = Some(q),
            None => *slot = last,
        }
    }
    acc
}

/// Number of dosing cycles to simulate when pre-equilibrating an SS=1
/// dose. With a typical t₁/₂/II ratio under 2 (the common clinical range)
/// this is comfortably past saturation — each additional cycle adds
/// `exp(-k·II)` of the prior decay, so by N=50 the truncation tail is
/// well below 1e-6 for any reasonable PK. The analytic-sensitivity SS
/// equilibration (`sens::ode_provider::equilibrate_ss_state_g`) reuses this
/// same constant so its trough can't drift from this f64 predictor (#473 review #11).
pub(crate) const SS_EQUILIBRATION_CYCLES: usize = 50;

/// Relative-`L∞` tolerance for the steady-state equilibration **early stop** (#519). The
/// `(apply dose; integrate II)` cycle is a geometric contraction with ratio `≈ exp(−λ·II)`;
/// once the cycle-to-cycle state change falls below this *relative* threshold, every
/// remaining cycle would move the trough by less still, so the truncation is already at f64
/// precision and we stop. Conservative (`1e-12`): the dropped tail is far below the
/// `provider`-vs-production parity tolerance, so the value is unchanged for any realistic
/// PK. Fast disposition (`λ·II ≈ 2`) converges in ~14 cycles; slow PK (`λ·II ≈ 0.1`) never
/// trips it and runs the full [`SS_EQUILIBRATION_CYCLES`] — identical to the old behaviour.
pub(crate) const SS_EQUILIBRATION_TOL: f64 = 1e-12;

/// Exact periodic steady state of a **linear** one-cycle map, `u_ss = (I − M)⁻¹·b`, generic over
/// the numeric type `T` — the engine-neutral replacement for iterating a truncated pulse train.
///
/// One dosing cycle of a linear disposition is an affine map `u ↦ M·u + b`, so its periodic fixed
/// point has a closed form. The advancement is *injected* as two "one cycle from `u0`" closures —
/// `advance_unforced` (disposition alone; builds the drift `z0` and the propagator columns of
/// `M = e^{A·II}`) and `advance_forced` (disposition + the cycle's own dose; builds `b` and drives
/// the linearity check). Each caller keeps its own propagation machinery while this body owns only
/// the `T`-typed algebra: the `I − M` assembly, the
/// [`solve_linear_system_g`](crate::sens::linsolve::solve_linear_system_g) solve, and the
/// value-part linearity verification. Run over a dual `T` it carries `∂u_ss/∂(θ,η)` (and the 2nd
/// order) through the solve automatically — the implicit-function derivative, with no
/// hand-assembled `dM`/`db`.
///
/// Two callers, distinguished only by how much numerical noise their propagation carries:
///
/// * **ODE** steady-state-into-absorption (`ode::predictions::equilibrate_ss_input_rate_g`, #835)
///   advances by RK45, so `reltol`/`abstol` are the solver's own and `vtol` below is scaled to
///   them. Its disposition may be genuinely nonlinear, which the self-check detects.
/// * **Analytical** event-driven SS (`pk::event_driven`, `sens::propagate`, #908) advances by the
///   closed-form eigenmode propagators, which are exact — it passes `reltol = abstol = 0`, leaving
///   `vtol = 1e-9·scale`, the pure floating-point rounding floor. Its disposition is *always*
///   linear (constant-coefficient), so the self-check is a can't-happen assertion there rather
///   than a real branch.
///
/// Returns `None` on a **nonlinear** map (the one-/two-cycle self-check fails), a singular `I − M`,
/// or any non-finite intermediate. A singular `I − M` means some disposition eigenvalue is zero —
/// a compartment that never empties — in which case no periodic steady state exists and the
/// caller's capped pulse-train fallback (plus [`note_ss_nonconvergence_if_capped`]) is the right
/// answer.
pub(crate) fn periodic_ss_fixed_point_g<T, FUnf, FFor>(
    n: usize,
    ii: f64,
    reltol: f64,
    abstol: f64,
    advance_unforced: FUnf,
    advance_forced: FFor,
) -> Option<Vec<T>>
where
    T: crate::sens::num::PkNum,
    FUnf: Fn(&[T]) -> Option<Vec<T>>,
    FFor: Fn(&[T]) -> Option<Vec<T>>,
{
    if !(ii > 0.0) || n == 0 {
        return None;
    }
    let zero = vec![T::from_f64(0.0); n];
    // Unforced zero-state drift (zero for a homogeneous RHS, non-zero for an affine one).
    let z0 = advance_unforced(&zero)?;
    // b: forced response over one cycle from a zero state (constant drift + the cycle's dose).
    let b = advance_forced(&zero)?;
    // I − M, row-major. Column i of M is the *homogeneous* one-cycle response of eᵢ with the
    // drift z0 removed, so an affine disposition still yields the true linear propagator.
    let mut i_minus_m = vec![T::from_f64(0.0); n * n];
    for i in 0..n {
        let mut ei = zero.clone();
        ei[i] = T::from_f64(1.0);
        let evolved = advance_unforced(&ei)?;
        for r in 0..n {
            let m_ri = evolved[r] - z0[r];
            let delta = T::from_f64(if r == i { 1.0 } else { 0.0 });
            i_minus_m[r * n + i] = delta - m_ri;
        }
    }
    let u_ss = crate::sens::linsolve::solve_linear_system_g::<T>(&i_minus_m, &b, n)?;
    if u_ss.iter().any(|x| !x.val().is_finite()) {
        return None;
    }
    // Confirm linearity on the value part: one and two forced cycles from u_ss must both return
    // u_ss (only an affine one-cycle map does). The tolerance is scaled to the caller's own
    // accuracy so the fast path is not falsely abandoned at a loose `ode_reltol`; a genuinely
    // nonlinear map diverges by O(scale) and is rejected. Two cycles (not one) rejects a weakly-
    // nonlinear map whose true fixed point merely happens to sit within `vtol` after the first.
    let scale = u_ss.iter().fold(1e-12_f64, |a, x| a.max(x.val().abs()));
    // Verification tolerance, scaled to the caller's own accuracy. For the ODE caller the
    // one-cycle residual of a genuinely *linear* map is not zero but a solver-noise floor: `b`
    // (forced from a zero state) and the check below (forced from `u_ss`) integrate the same
    // periodic `R_in` over *different* adaptive step sequences, so their forcing quadratures
    // differ by O(reltol·scale). Empirically that floor is ≈ 45·reltol (relative) — above the
    // original `32·reltol`, which therefore falsely declined even linear models at a tight
    // `ode_reltol` (e.g. 1e-10), silently forcing the fallback iteration. `256·reltol` clears the
    // floor with ~5× margin while still rejecting a genuinely nonlinear map, whose one-cycle
    // residual is O(scale) — ~1e8·reltol, eight orders above this bound (see
    // `ss_input_rate_nonlinear_disposition_falls_back_to_iteration`). The analytical caller passes
    // `reltol = abstol = 0`, so the bound degenerates to the `1e-9·scale` rounding floor its exact
    // propagators need — seven orders above their measured ~1e-16 residual and still eight orders
    // below a nonlinear map's O(scale).
    let vtol = (256.0 * reltol + 1e-9) * scale + 256.0 * abstol;
    let within = |a: &[T], b: &[T]| {
        a.iter()
            .zip(b)
            .all(|(c, u)| (c.val() - u.val()).abs() <= vtol)
    };
    let u_check = advance_forced(&u_ss)?;
    if !within(&u_check, &u_ss) {
        return None;
    }
    let u_check2 = advance_forced(&u_check)?;
    if within(&u_check2, &u_ss) {
        Some(u_ss)
    } else {
        None
    }
}

/// Whether the SS-equilibration trough has converged between two successive cycles. Shared
/// by the f64 predictor, the event-driven f64 loop, and the dual gradient path so every path
/// truncates on the *same* criterion — the dual feeds the value parts (`PkNum::val`) of its
/// state (#519), which keeps its stop cycle identical to the f64 path's, so the truncated
/// gradient is the exact derivative of the truncated value (see [`crate::sens::propagate::ss_dual_cycle_should_stop`]).
///
/// **Mixed `atol`/`rtol` test on the per-cycle *increment*** (#532 review #1): a compartment
/// is converged when its movement since the previous cycle is below `tol·|cur| + tol·max_mag`
/// — negligible both relative to itself and relative to the dominant compartment. Testing the
/// *increment* (not the magnitude) is what makes this safe in a scale-separated model: a small
/// compartment still in transit (effect-site / metabolite many orders below central) keeps the
/// loop running until it too stops moving, rather than being declared converged merely for
/// being small. The `tol·max_mag` term is the absolute floor that lets a genuinely-settled
/// near-zero compartment — where the pure relative test is ill-conditioned — pass; without it
/// the loop could never stop. Because the stop only fires once every compartment's increment
/// is below f64-relative precision, the value has reached its fixed point and the elided cycles
/// do not move it — predictions are unchanged to f64 precision, and gradients match a full
/// budget to `< 1e-6` (see `ode_provider_ss_nonlinear_fallback_early_stop_matches_full_budget`;
/// after #914 a linear disposition takes the exact fixed point, so this early stop runs only on
/// the nonlinear fallback).
///
/// A **non-finite** (`NaN`/`Inf`) compartment means the integration blew up: never report
/// convergence — don't early-exit and silently return a poisoned state; run the full cycle
/// budget exactly as the pre-#519 code did so the failure surfaces identically (#532 review
/// #4). Required because `f64::max` would otherwise *drop* a `NaN` and mask it.
pub(crate) fn ss_cycle_converged(cur: &[f64], prev: &[f64], tol: f64) -> bool {
    // Test-only escape hatch: force every path to run the full cycle budget so a test can
    // compare the early-stopped result against the fully-equilibrated one (#532 review #4).
    #[cfg(test)]
    if FORCE_FULL_SS_EQUILIBRATION.with(|c| c.get()) {
        return false;
    }
    if cur.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let max_mag = cur.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let atol = tol * max_mag;
    cur.iter()
        .zip(prev)
        .all(|(&a, &b)| (a - b).abs() <= tol * a.abs() + atol)
}

/// Rolling prev-state tracker for the f64 SS-equilibration early stop. Owns the previous
/// cycle's state so the f64 predictor and the event-driven f64 loop share one scaffold instead
/// of each re-implementing the `cycle > 0` + `copy_from_slice` dance — a later tweak missed in
/// one site would reintroduce cross-path trough drift (#532 review #6). The dual paths use the
/// generic [`crate::sens::propagate::ss_dual_cycle_should_stop`], which applies the same
/// [`ss_cycle_converged`] criterion to the value parts of the dual state.
#[derive(Default)]
pub(crate) struct SsStopTracker {
    prev: Vec<f64>,
    /// **Absolute** L∞ increments `max|cur − prev|` of the two most recent *non-converged*
    /// cycles, oldest first (`[prev_increment, last_increment]`) — the geometric-tail inputs for
    /// the #867 cycle-cap non-convergence check. Absolute (not magnitude-relative) on purpose: a
    /// *linearly diverging* trough (mean input ≥ max elimination) takes a ~constant step on a
    /// growing value, so its absolute ratio reads `ρ ≈ 1` ("not contracting") while a
    /// magnitude-relative step `B/(A+Bn)` would shrink as the trough grows and hide the
    /// divergence. Rolled only on the else branch of [`Self::should_stop`]; the dual sensitivity
    /// paths (which reuse the same stop criterion via `ss_dual_cycle_should_stop`) don't consult
    /// this.
    abs_increments: [f64; 2],
    /// L∞ magnitude `max|cur|` of the most recent non-converged cycle, used to re-express the
    /// absolute tail as a fraction of the trough actually held (and as the noise-floor scale that
    /// gates a false "no steady state" alarm on a converged model).
    last_mag: f64,
}

impl SsStopTracker {
    /// Record `cur` and report whether the trough has converged (from cycle 1 on). Returns
    /// `true` to break the equilibration loop.
    pub(crate) fn should_stop(&mut self, cycle: usize, cur: &[f64]) -> bool {
        if cycle > 0 {
            if ss_cycle_converged(cur, &self.prev, SS_EQUILIBRATION_TOL) {
                return true;
            }
            // Not converged this cycle: roll the two-deep absolute-increment history (and the
            // current magnitude) so a loop that runs out the cycle budget can estimate its
            // un-taken geometric tail and its size relative to the trough (#867).
            self.abs_increments = [self.abs_increments[1], ss_abs_increment(cur, &self.prev)];
            self.last_mag = ss_max_magnitude(cur);
        }
        self.prev.clear();
        self.prev.extend_from_slice(cur);
        false
    }

    /// The `(prev_abs, last_abs, last_mag)` of the two most recent non-converged cycles — the
    /// inputs to [`note_ss_nonconvergence_if_capped`] (#867). The absolute increments are `0.0`
    /// (and `last_mag` `0.0`) on a loop that early-stopped or ran fewer than two non-converged
    /// cycles, which that check treats as "nothing to warn about".
    pub(crate) fn recent_increments(&self) -> (f64, f64, f64) {
        (
            self.abs_increments[0],
            self.abs_increments[1],
            self.last_mag,
        )
    }
}

/// Absolute L∞ change between successive SS-equilibration cycles: `max|cur − prev|` (#867). Used
/// to estimate the geometric contraction ratio when a loop hits the cycle cap; absolute so a
/// constant-step linear divergence reads `ρ ≈ 1`. See [`SsStopTracker::abs_increments`].
fn ss_abs_increment(cur: &[f64], prev: &[f64]) -> f64 {
    cur.iter()
        .zip(prev)
        .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()))
}

/// L∞ magnitude `max|cur|` of a cycle state — the scale the absolute tail is normalised against
/// (#867). `0.0` when `cur` is all-zero (nothing has accumulated).
fn ss_max_magnitude(cur: &[f64]) -> f64 {
    cur.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

#[cfg(test)]
thread_local! {
    /// Cycles the most recent SS-equilibration call ran — a **test-only** observation of the
    /// #519 early stop, so a test can assert it fired for fast PK and ran the full budget for
    /// slow PK (#532 review #5/#6 — otherwise the stop logic ships unverified, since the loose
    /// end-value tolerances absorb a too-early exit). Set by the f64 predictor, the dual ODE /
    /// closed-form loops, and the event-driven loop.
    static LAST_SS_EQUILIBRATION_CYCLES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Branch the most recent SS equilibration took — see [`SsBranch`] for why the cycle
    /// count above is not enough.
    #[cfg(test)]
    static LAST_SS_EQUILIBRATION_BRANCH: std::cell::Cell<SsBranch> =
        const { std::cell::Cell::new(SsBranch::None) };

    /// When set, [`ss_cycle_converged`] always reports "not converged" so every path runs the
    /// full cycle budget — lets a test pin that early-stop is value-preserving vs full
    /// equilibration (#532 review #4).
    static FORCE_FULL_SS_EQUILIBRATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn record_ss_equilibration_cycles(n: usize) {
    LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.set(n));
}

/// Which SS-equilibration branch the most recent call took — a **test-only** observation,
/// alongside the cycle count above.
///
/// The branches **fall back to one another**: the exact linear fixed point declines to
/// Anderson, and Anderson declines to the capped pulse train. When all three are correct
/// they return the same trough, so a value assertion cannot tell them apart — and, worse,
/// *breaking* one silently routes to the next and the assertion stays green. That is not
/// hypothetical: two #1139 mutations (a `NaN` anchor in `equilibrate_ss_input_rate`'s
/// one-cycle advance, and a flat `0` anchor in the monotone input-rate train) left the whole
/// suite passing, because the first made the exact solve decline into a correct fallback and
/// the second sat on a branch the fixture never reached. A test that means to pin a branch's
/// clock must assert which branch ran.
///
/// **One cell per thread, shared across numeric types.** The recording sites live in
/// helpers that are generic over `T: PkNum` (`equilibrate_ss_input_rate_g`,
/// `anderson_ss_fixed_point_g`), and `sens::ode_provider` instantiates them with a dual
/// `T`. So a test that drives a `fit()` or the analytic provider observes whichever walk
/// ran *last*, f64 or dual — not necessarily the value-path run-in it means to pin. Assert
/// this only from a test that calls the f64 helpers directly, as the #1139 tests do. The
/// neighbouring cycle counter has the same property and the same caveat.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum SsBranch {
    /// No SS equilibration has completed on this thread since the last one started.
    ///
    /// Written at the top of `equilibrate_ss_pk_state`, so a call that **bails out early**
    /// (`II <= 0`, an out-of-range compartment, overlapping infusions) leaves `None` rather
    /// than the previous call's tag. That matters: without it a test asserting a branch
    /// after a bail-out would read a stale value and pass for the wrong reason — the class
    /// of un-failable assertion this enum exists to prevent.
    #[default]
    None,
    /// The exact affine fixed point `(I − M)⁻¹·b` for a **bolus or infusion** dose on a
    /// linear disposition (#914) — `equilibrate_ss_pk_state`'s `periodic_ss_fixed_point_pk`.
    Exact,
    /// The same closed form on the **input-rate** (built-in absorption) path,
    /// `equilibrate_ss_input_rate`. Distinct from [`Self::Exact`] so an assertion can say
    /// which of the two ran: they are different functions with different windows, and a
    /// test naming one of them should not pass on the other.
    InputRateExact,
    /// Anderson-accelerated iteration on the same one-cycle map, for a nonlinear
    /// disposition into a built-in absorption compartment (#867).
    Anderson,
    /// The capped explicit pulse train, on a clock local to each cycle (bolus/infusion).
    CappedTrain,
    /// The capped explicit pulse train of the input-rate path, which unlike every other
    /// window runs on a **monotone** clock `0 … n·II` and advances its `TAD` anchor per
    /// segment (#1139).
    InputRateTrain,
}

#[cfg(test)]
pub(crate) fn record_ss_equilibration_branch(b: SsBranch) {
    LAST_SS_EQUILIBRATION_BRANCH.with(|c| c.set(b));
}

/// Branch the most recent SS equilibration took (test observation; see [`SsBranch`]).
#[cfg(test)]
pub(crate) fn last_ss_equilibration_branch() -> SsBranch {
    LAST_SS_EQUILIBRATION_BRANCH.with(|c| c.get())
}

/// Cycles the most recent SS-equilibration call ran (test observation; see above).
#[cfg(test)]
pub(crate) fn last_ss_equilibration_cycles() -> usize {
    LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.get())
}

/// Run `f` with every SS-equilibration path forced to the full cycle budget (#532 review #4).
/// The reset rides a drop guard so a panic in `f` cannot leave the flag set and poison a later
/// test sharing the harness thread.
#[cfg(test)]
pub(crate) fn with_full_ss_equilibration<R>(f: impl FnOnce() -> R) -> R {
    // Re-entrant: the guard restores the PRIOR value, not an unconditional `false`, so a
    // nested call leaves the outer body running with full equilibration still forced.
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_FULL_SS_EQUILIBRATION.with(|c| c.set(self.0));
        }
    }
    let prev = FORCE_FULL_SS_EQUILIBRATION.with(|c| c.replace(true));
    let _reset = Reset(prev);
    f()
}

/// No-op in non-test builds (zero cost on the hot path).
#[cfg(not(test))]
#[inline(always)]
pub(crate) fn record_ss_equilibration_cycles(_n: usize) {}

/// The branch tag's non-test counterpart — same signature, so the call sites are
/// unconditional. [`SsBranch`] is a fieldless enum, so naming a variant costs nothing.
#[cfg(not(test))]
#[inline(always)]
pub(crate) fn record_ss_equilibration_branch(_b: SsBranch) {}

/// Relative-magnitude threshold above which a **cycle-capped** SS equilibration is reported as
/// non-converged (#867). The pulse-train equilibration
/// (`crate::ode::predictions::equilibrate_ss_state`) is a geometric contraction with per-cycle
/// carryover ratio `ρ ≈ exp(−λ·II)`. A heavily-accumulating **nonlinear** disposition (e.g.
/// Michaelis–Menten with elimination half-life ≫ dosing interval `II`) drives `ρ → 1`, so the
/// [`SS_EQUILIBRATION_CYCLES`]-capped loop stops far short of the true periodic steady state and
/// the returned trough is materially too low — silently, until now. The un-taken tail is estimated
/// from the last two cycle increments as `incr·ρ/(1−ρ)` (the closed sum of the remaining geometric
/// series); when it exceeds this fraction of the trough — or the sequence is not contracting
/// (`ρ ≥ 1`: mean input ≥ maximum elimination, so no periodic steady state exists at all) — a
/// warning is surfaced instead of returning the under-converged value silently.
///
/// This constant plays **two** roles in [`ss_equilibration_tail_warning`]: the tail threshold
/// above, and the **noise floor** below which the last step (`abs_last / mag`) is treated as
/// converged — suppressing a false "no steady state" alarm when a genuinely converged model runs
/// out the cycle budget with its increments jittering at the ODE solver's own (much looser)
/// tolerance rather than reaching the `1e-12` early-stop bit-identity.
///
/// `1e-2` (1%): the accurate low-accumulation cases sit at ~0% estimated tail while the
/// pathological cases in #867 are 38–79% low, so the threshold separates them with wide margin;
/// and 1% is ~100× the default solver `reltol = 1e-4`, so the noise floor clears the jitter with
/// room to spare. A **linear** disposition is unaffected — it takes the exact closed-form fast
/// path (`(I − M)⁻¹·b`, #835) and never reaches this iteration.
pub(crate) const SS_WARN_REL_TOL: f64 = 1e-2;

/// Process-global collector for steady-state equilibration non-convergence warnings (#867).
///
/// The capped pulse-train equilibration runs deep inside the per-subject prediction walk
/// (`ode::predictions::equilibrate_ss_state`), which rayon parallelises across subjects and whose
/// numeric signatures carry no warning channel. Threading a `&mut Vec<String>` through every
/// `ode_predictions_*` variant and its callers would be a wide, invasive change for a rare
/// diagnostic, and a thread-local would be lost on the rayon worker threads. So the message —
/// written only on the rare non-converged nonlinear branch — is deduplicated into this cross-thread
/// set and drained by the draining `api` boundaries (`fit` / `simulate_with_options_diag`) into
/// `FitResult.warnings` / `SimulationOutput.warnings` via [`take_ss_nonconvergence_warnings`].
///
/// The message is model-structural (independent of subject and of objective-eval count), so the
/// `BTreeSet` collapses the thousands of identical writes a fit produces down to one entry, and its
/// deterministic ordering keeps the surfaced list stable. This assumes one top-level `api`
/// operation per process at a time (the CLI / ferx-r usage): each draining entrypoint
/// [`clear_ss_nonconvergence_warnings`] on entry and [`take_ss_nonconvergence_warnings`] on exit to
/// bracket its own run; genuinely concurrent in-process operations would share the sink. Note that
/// bare `predict()` reaches the *write* (via `equilibrate_ss_state`) but has no warnings channel to
/// drain into — it neither clears nor takes, so its message only ever lingers until the next
/// draining entrypoint clears the sink (harmless in sequential use; the follow-up that gives
/// `predict()` a channel is #867 Option B).
fn ss_nonconvergence_sink() -> &'static std::sync::Mutex<std::collections::BTreeSet<String>> {
    static SINK: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<String>>> =
        std::sync::OnceLock::new();
    SINK.get_or_init(std::sync::Mutex::default)
}

/// Estimate the un-taken geometric tail of a cycle-capped SS equilibration and, if it is
/// non-negligible, deduplicate a non-convergence warning into the [`ss_nonconvergence_sink`]
/// (#867). `early_stopped` short-circuits (the loop converged, so there is nothing to warn about);
/// `abs_prev` / `abs_last` are the two most recent *absolute*-L∞ cycle increments and `mag` the
/// final trough magnitude ([`SsStopTracker::recent_increments`]).
pub(crate) fn note_ss_nonconvergence_if_capped(
    early_stopped: bool,
    abs_prev: f64,
    abs_last: f64,
    mag: f64,
) {
    if let Some(msg) = ss_equilibration_tail_warning(early_stopped, abs_prev, abs_last, mag) {
        if let Ok(mut set) = ss_nonconvergence_sink().lock() {
            set.insert(msg);
        }
    }
}

/// Build the #867 non-convergence warning from the last two **absolute** cycle increments and the
/// final trough magnitude, or `None` when the equilibration is trustworthy. Split out from
/// [`note_ss_nonconvergence_if_capped`] so the geometric-tail arithmetic is unit-testable without
/// touching the global sink.
///
/// `ρ = abs_last / abs_prev` is the estimated per-cycle carryover ratio (absolute, so a
/// constant-step linear divergence reads `ρ ≈ 1`); `rel_last = abs_last / mag` is the last step as
/// a fraction of the trough. The remaining relative error is the closed geometric tail
/// `rel_last·ρ/(1−ρ)`. Two guards keep this from crying wolf on a *converged* model:
///
/// * **Noise floor** — early-stop needs consecutive troughs within `SS_EQUILIBRATION_TOL` (1e-12),
///   but the ODE solver's own relative tolerance (default `reltol = 1e-4`) is orders of magnitude
///   looser, so a genuinely converged model can run out the whole cycle budget with its increments
///   jittering at the solver noise floor rather than reaching bit-identity. There `ρ` is pure
///   noise and lands `≥ 1` about half the time — which would fire the scariest "no steady state"
///   message on correct output. The `rel_last ≤ SS_WARN_REL_TOL` short-circuit suppresses that:
///   once the last step is under 1% of the trough the model has effectively converged regardless
///   of the noisy ratio. The floor is *not* applied to the contracting (`ρ < 1`) branch, where a
///   near-`ρ=1` heavy-accumulation model legitimately takes a small step yet still has a large
///   summed tail.
/// * `ρ ≥ 1` (or non-finite) means the sequence is not contracting — no periodic steady state —
///   and warns *only* once past the noise floor.
fn ss_equilibration_tail_warning(
    early_stopped: bool,
    abs_prev: f64,
    abs_last: f64,
    mag: f64,
) -> Option<String> {
    // Converged (broke out early), or nothing moving in the last cycle → trustworthy trough.
    if early_stopped || abs_last <= 0.0 {
        return None;
    }
    // A blown-up integration (non-finite state) or an all-zero trough gives no scale to normalise
    // against — don't manufacture a tail estimate from garbage. Mirrors the non-finite guard in
    // [`ss_cycle_converged`] (the failure surfaces elsewhere as a non-finite prediction / OFV).
    if !abs_last.is_finite() || !mag.is_finite() || mag <= 0.0 {
        return None;
    }
    // Last step as a fraction of the trough we hold — the noise-floor gate for the ρ ≥ 1 branch
    // below and the scale for the tail estimate.
    let rel_last = abs_last / mag;
    // Can't estimate ρ without a prior increment; a still-moving final cycle with no history is
    // too little signal to cry non-convergence on. (Only happens if the very last cycle is the
    // first non-converged one, which the cycle cap makes vanishingly unlikely.)
    if abs_prev <= 0.0 {
        return None;
    }
    let rho = abs_last / abs_prev;
    if !rho.is_finite() {
        return None;
    }
    if rho >= 1.0 {
        // Not contracting → no periodic steady state. But this is where solver-noise jitter on a
        // *converged* model (last step at the ODE-tolerance floor, so `ρ` is meaningless and lands
        // ≥ 1 about half the time) would otherwise fire the scariest message on correct output.
        // Gate it on the last step being a real fraction of the trough. Applied *here only*, never
        // to the contracting branch below — a near-`ρ=1` heavy-accumulation model legitimately
        // takes a small step yet sums to a large tail, and must still warn.
        if rel_last <= SS_WARN_REL_TOL {
            return None;
        }
        return Some(format!(
            "Steady-state (SS=1) equilibration did not converge within {SS_EQUILIBRATION_CYCLES} \
             cycles: the per-cycle carryover is not contracting (ratio ≈ {rho:.3}), indicating no \
             periodic steady state exists — the mean input rate meets or exceeds the maximum \
             elimination rate (e.g. a saturable / Michaelis–Menten disposition dosed above its \
             capacity). The returned SS trough is unreliable.",
        ));
    }
    // Remaining tail relative to the (under-converged) *current* trough — the natural scale for
    // the threshold, since that is what we hold.
    let remaining_vs_current = rel_last * rho / (1.0 - rho);
    if remaining_vs_current > SS_WARN_REL_TOL {
        // Re-express as a fraction of the *true* steady state `u_ss ≈ u_last·(1 + tail)` for the
        // human-facing "% below true SS" — the framing #867's evidence table uses (its 38% row is
        // `remaining_vs_current ≈ 0.61`, i.e. 0.61/1.61 ≈ 38%). Reporting the ÷u_last figure would
        // overstate the bias.
        let pct_below_true = 100.0 * remaining_vs_current / (1.0 + remaining_vs_current);
        return Some(format!(
            "Steady-state (SS=1) equilibration reached the {SS_EQUILIBRATION_CYCLES}-cycle cap \
             without converging (per-cycle carryover ratio ≈ {rho:.3}); the returned SS trough is \
             approximately {pct_below_true:.0}% below the true periodic steady state. This affects \
             slowly-accumulating nonlinear (e.g. Michaelis–Menten) disposition where the \
             elimination half-life greatly exceeds the dosing interval II; predictions and \
             simulations for such a model may be biased low.",
        ));
    }
    None
}

/// Serializes the tests that *read* the process-global SS non-convergence sink
/// ([`take_ss_nonconvergence_warnings`], #867) so cargo's parallel harness can't have one test
/// drain another's entry mid-read. Writers don't drain, so only readers need to coordinate.
/// Lives here rather than beside any one test module because the sink is now shared by the ODE
/// (`ode::predictions`) and analytical (`pk::event_driven`) fallbacks, which sit in different
/// files but the *same* lib-test binary — a per-file guard would not serialize them against
/// each other (#908).
#[cfg(test)]
pub(crate) static SS_WARN_SINK_READER_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Drain and return the collected SS non-convergence warnings (#867). The `api` boundary calls
/// this after its prediction pass and appends the result to `FitResult.warnings`. See
/// [`ss_nonconvergence_sink`].
pub(crate) fn take_ss_nonconvergence_warnings() -> Vec<String> {
    ss_nonconvergence_sink()
        .lock()
        .map(|mut set| std::mem::take(&mut *set).into_iter().collect())
        .unwrap_or_default()
}

/// Clear any residual SS non-convergence warnings so a top-level `api` call starts from a clean
/// sink (#867). See [`ss_nonconvergence_sink`].
pub(crate) fn clear_ss_nonconvergence_warnings() {
    if let Ok(mut set) = ss_nonconvergence_sink().lock() {
        set.clear();
    }
}

#[cfg(test)]
mod ss_warn_tests {
    use super::*;

    // Args are `(early_stopped, abs_prev, abs_last, mag)`: absolute L∞ increments of the last two
    // non-converged cycles plus the final trough magnitude. `mag = 1.0` makes `abs == rel`.

    #[test]
    fn early_stop_never_warns() {
        assert!(ss_equilibration_tail_warning(true, 0.5, 0.4, 1.0).is_none());
    }

    #[test]
    fn negligible_tail_does_not_warn() {
        // ρ = 0.1 (fast contraction), last step 5% of trough: remaining ≈ 0.05·0.1/0.9 ≈ 0.6% < 1%
        // → trustworthy (exercises the contracting branch, not the noise floor).
        assert!(ss_equilibration_tail_warning(false, 0.5, 0.05, 1.0).is_none());
    }

    #[test]
    fn slow_contraction_warns_biased_low() {
        // ρ = 0.9, last step 5% of trough → remaining ≈ 0.05·9 = 45% ≫ 1%.
        let w = ss_equilibration_tail_warning(false, 0.05 / 0.9, 0.05, 1.0).expect("should warn");
        assert!(w.contains("below the true periodic steady state"), "{w}");
    }

    #[test]
    fn non_contracting_warns_no_steady_state() {
        // Increment growing (ρ > 1), well above the noise floor: no periodic steady state.
        let w = ss_equilibration_tail_warning(false, 0.02, 0.03, 1.0).expect("should warn");
        assert!(w.contains("not contracting"), "{w}");
        assert!(w.contains("no periodic steady state"), "{w}");
    }

    #[test]
    fn constant_step_divergence_warns_no_steady_state() {
        // A linearly diverging trough takes a ~constant *absolute* step → ρ ≈ 1 → "no steady
        // state". (The pre-fix relative-increment form would have shrunk the step as the trough
        // grew, hidden the divergence, and mislabeled it "~50% below true SS".)
        let w = ss_equilibration_tail_warning(false, 0.02, 0.02, 1.0).expect("should warn");
        assert!(w.contains("no periodic steady state"), "{w}");
    }

    #[test]
    fn noise_floor_suppresses_false_no_steady_state() {
        // A *converged* model that ran the full budget: increments jitter at the solver noise
        // floor (rel step ~1e-4 ≪ 1%), and noise makes abs_last ≥ abs_prev (ρ ≥ 1). Must NOT fire
        // the "no steady state" alarm (the pre-fix code did). This is the #867-review regression.
        assert!(ss_equilibration_tail_warning(false, 1.0e-4, 1.1e-4, 1.0).is_none());
        // Same jitter with ρ < 1 is likewise trustworthy.
        assert!(ss_equilibration_tail_warning(false, 1.2e-4, 1.0e-4, 1.0).is_none());
    }

    #[test]
    fn small_step_large_tail_still_warns_near_rho_one() {
        // Heavy accumulation: the last step is only 0.5% of the trough — *below* the 1%
        // `SS_WARN_REL_TOL` noise floor — yet ρ = 0.98 sums to a large tail (remaining ≈
        // 0.005·0.98/0.02 ≈ 25%). The noise floor must NOT suppress this: it is applied only to the
        // ρ ≥ 1 branch, never to the contracting branch (regression guard for the #874-review fix,
        // where an over-broad floor silenced exactly the #867 target cases).
        let w =
            ss_equilibration_tail_warning(false, 0.005 / 0.98, 0.005, 1.0).expect("should warn");
        assert!(w.contains("below the true periodic steady state"), "{w}");
    }

    #[test]
    fn zero_increment_is_trustworthy() {
        assert!(ss_equilibration_tail_warning(false, 0.0, 0.0, 1.0).is_none());
        assert!(ss_equilibration_tail_warning(false, 0.1, 0.0, 1.0).is_none());
    }

    #[test]
    fn no_prior_increment_is_trustworthy() {
        // Moving last step (past the noise floor) but no prior increment to estimate ρ from.
        assert!(ss_equilibration_tail_warning(false, 0.0, 0.05, 1.0).is_none());
    }

    #[test]
    fn non_finite_or_zero_magnitude_does_not_warn() {
        // Blown-up state (non-finite increment or magnitude) or an all-zero trough → no scale to
        // normalise against, so no manufactured warning (mirrors ss_cycle_converged's guard).
        assert!(ss_equilibration_tail_warning(false, 1.0, f64::INFINITY, 1.0).is_none());
        assert!(ss_equilibration_tail_warning(false, 1.0, 0.05, f64::INFINITY).is_none());
        assert!(ss_equilibration_tail_warning(false, 1.0, 0.05, f64::NAN).is_none());
        assert!(ss_equilibration_tail_warning(false, 0.1, 0.05, 0.0).is_none());
    }

    #[test]
    fn increment_helpers() {
        assert_eq!(ss_max_magnitude(&[0.0, 0.0]), 0.0);
        assert_eq!(ss_max_magnitude(&[-3.0, 1.0]), 3.0);
        assert_eq!(ss_abs_increment(&[1.0, 2.0], &[1.0, 0.0]), 2.0);
        assert_eq!(ss_abs_increment(&[5.0], &[5.0]), 0.0);
    }
}

/// `is_infusion()` only checks `rate > 0`, but a degenerate row with
/// `rate > 0 && amt <= 0` (or NaN) yields `duration = amt/rate <= 0`
/// (or NaN). Treating those as infusions would push an infusion-end
/// break that sorts before the dose itself, and NaN would panic the
/// break-time sort. Such rows fall back to the bolus branch instead
/// (a zero/negative bolus update — visible, not silently dropped).
pub(crate) fn is_real_infusion(d: &DoseEvent) -> bool {
    // Tripwire (#324): every ODE entrypoint resolves modeled-RATE doses to
    // `Fixed` (via `resolve_subject_doses*`) before any infusion logic runs, so
    // a non-`Fixed` dose here means a path forgot to resolve — panic in debug /
    // tests rather than silently mis-handling it (an unresolved modeled dose has
    // `duration == 0`, so it would quietly degrade to a bolus).
    debug_assert!(d.is_fixed(), "is_real_infusion: unresolved modeled dose");
    d.is_infusion() && d.duration > 0.0 && d.duration.is_finite()
}

// ---------------------------------------------------------------------------
// Where a lagged steady-state dose loads, and what the previous cycle leaves
// running across the dose record (#1121).
// ---------------------------------------------------------------------------

/// Whether this steady-state dose's periodic state is loaded at its dose
/// **record** and flowed forward to the lagged arrival (#1121), rather than
/// equilibrated at the arrival itself.
///
/// NONMEM runs `$PK` at the dose row, loads the steady-state compartments there,
/// and then ADVANs to the lagged arrival under the record that *terminates* that
/// interval. Equilibrating at the arrival instead computes the trough throughout
/// under the dose row's snapshot. The two agree exactly under flat covariates —
/// a full-cycle propagation from phase `II − lag` returns to the trough — which
/// is why the difference stayed invisible until a covariate changed inside the
/// pre-arrival window.
///
/// `lag == 0` is excluded deliberately: record and arrival are the same instant,
/// so there is nothing to propagate, and routing it through
/// `ss_state_at_phase(…, II)` would integrate a full extra cycle and move every
/// existing non-lagged SS result by solver error, for no gain.
///
/// Every call site reads this predicate rather than re-deriving the condition, so
/// the seed and the arrival-side equilibration cannot drift into overlapping
/// (double-load) or disjoint (no-load) coverage — and the analytic twin
/// (`sens::ode_provider::integrate_tvcov_g`) shares it too, so its `K_SS_SEED`
/// timeline break fires on exactly the doses production seeds. A twin that seeded
/// on a wider or narrower set would disagree with production in *value*, which is
/// the one thing the `check_vs_production` parity tests cannot forgive.
pub(crate) fn ss_seeded_at_record(dose: &DoseEvent, lag: f64) -> bool {
    dose.ss && dose.ii > 0.0 && lag > 0.0
}

/// Cycle phase the dose **record** sits at, for a steady-state dose seeded there.
///
/// The pulse preceding the record landed at `dose.time − ss_seed_phase(…)`, so a
/// lagtime shorter than one interval puts it at `dose.time + lag − II` and the
/// record reads the previous cycle's tail at phase `II − lag`.
///
/// # `lag ≥ II` clamps rather than wraps — measured, not assumed
///
/// A lagtime of a full interval or more has no phase `II − lag`. NONMEM 7.6.0
/// **clamps** it to zero: the pulse lands on the record itself, so the record
/// carries the steady-state *peak* (`nonmem_anchor/results/ss_lag_ge_ii.tab`,
/// where an `ALAG1 = 15`, `II = 12` bolus reads `PRED(0) = 13.197 =
/// (D/V)/(1 − e^{−k·II})`, the post-pulse peak, and decays from there with no
/// intervening pulse before the real arrival at `t = 15`). It does **not** wrap
/// the phase into `[0, II)`: wrapping would put `PRED(0)` at `2.1815` instead.
///
/// Clamping is also the only continuous choice, which matters because `ALAG` is
/// routinely *estimated*: as `lag → II⁻` the phase tends to `0⁺` (the pulse has
/// just landed) and at `lag = II` it is `0` (the pulse lands here), so nothing
/// jumps as the outer optimizer walks a lagtime across the interval. Wrapping
/// would step the whole pre-arrival window by a factor of `e^{−k·II}` at that
/// crossing.
pub(crate) fn ss_seed_phase(dose: &DoseEvent, lag: f64) -> f64 {
    (dose.ii - lag).max(0.0)
}

/// End time of the previous cycle's infusion when it is **still running at the
/// dose record** of a seeded steady-state dose — `None` when it has already
/// finished (the ordinary case) or the dose is not an infusion.
///
/// The pulse before the record is at `dose.time − p` for `p = ss_seed_phase(…)`,
/// so its infusion occupies `[dose.time − p, dose.time − p + T_inf]` and crosses
/// the record whenever `p < T_inf`, i.e. `lag > II − T_inf`. The pre-arrival
/// window must then carry `+rate` on `[dose.time, dose.time + (T_inf − p)]`:
/// `ss_state_at_phase` hands back a state with the infusion mid-flight, and a
/// walk that forgot the rate would silently resume the decay early.
///
/// Confirmed against NONMEM 7.6.0 (`nonmem_anchor/results/ss_lag_infusion.tab`):
/// with `II = 12`, `T_inf = 6`, `ALAG1 = 8` the concentration *rises* from
/// `6.5468` at the record to `6.8754` at `t = 0.5` and turns over at exactly
/// `t = 2 = 8 − 12 + 6`.
///
/// Like every other infusion edge the window is `F`-reshaped (#419), so callers
/// pass the same `f_bio` they used to build the dose's own window and the two
/// edges cannot drift apart.
///
/// # `lag ≥ II` with an infusion is ferx's answer, not NONMEM's
///
/// Under the [`ss_seed_phase`] clamp this returns `dose.time + T_inf` — the
/// previous cycle's infusion starting on the record, which is what the clamped
/// phase means and is mass-balanced. NONMEM instead delivers `T_inf + (lag − II)`
/// hours of drug for that geometry (measured: an `AMT = 600`, `RATE = 100`,
/// `II = 12`, `ALAG1 = 14` dose infuses over `[0, 8]`, i.e. 800 mg), which is not
/// a physical reading of a 600 mg dose. ferx does not reproduce it; see
/// `docs/model-file/lagtime.qmd`.
pub(crate) fn ss_residual_infusion_end(dose: &DoseEvent, lag: f64, f_bio: f64) -> Option<f64> {
    if !ss_seeded_at_record(dose, lag) || !is_real_infusion(dose) {
        return None;
    }
    let (_, t_inf) = dose.bioavailable_infusion(f_bio);
    let phase = ss_seed_phase(dose, lag);
    (phase < t_inf).then(|| dose.time + (t_inf - phase))
}

/// Whether flowing a seeded steady-state dose from its record to its lagged
/// arrival lands back on the periodic **trough** — so a path that re-equilibrates
/// at the arrival instead of propagating there gets the same number.
///
/// The seed sits at phase [`ss_seed_phase`] and the arrival is `lag` later, so
/// the flowed phase is `(II − lag) + lag = II ≡ 0⁻`, the trough, for every
/// `lag ≤ II`. Past that the clamp pins the seed at phase 0 and the arrival lands
/// at phase `lag > II` — strictly further down the same decay than the trough —
/// so re-equilibrating there silently *raises* the state back to a full cycle's
/// accumulation. Measured at **+4.3 %** on an `ALAG1 = 15`, `II = 12` bolus at
/// `t = 26` (`nonmem_anchor/results/ss_lag_ge_ii`), on the two paths that take
/// the shortcut: the dense ODE predictor and the analytical superposition. Both
/// event-driven walks propagate and were already right.
///
/// The paths that re-equilibrate keep doing so while this holds, deliberately:
/// integrating a full extra cycle instead would move every existing SS+lagtime
/// result by solver error for no gain.
pub(crate) fn ss_arrival_is_trough(dose: &DoseEvent, lag: f64) -> bool {
    lag <= dose.ii
}

/// Ordering slack shared by every [`tad_referent`] caller. `1e-12` is the value all four
/// TAD folds already used before they were collapsed onto this function; it is deliberately
/// *not* [`crate::ode::predictions::EVENT_MATCH_TOL`], which answers a different question
/// (does this break time *name* this event) on a much looser scale.
const TAD_EPS: f64 = 1e-12;

/// **The** rule for what `TAD` at time `t` is measured from, for one dose — `None` when
/// this dose has no referent at `t` yet.
///
/// Every **lag-aware** `TAD` fold is this function under a `max`: the dense predictor's
/// [`crate::ode::predictions::tad_anchor_for`] (which the EKF and the event-driven walk also
/// use) and the sdtab column's `api::output_columns::tad_at_time`, via [`tad_at`]. Before
/// #1126 those were hand-written copies that disagreed on
/// `(t_dose = 120, ALAG = 3, II = 12, t = 121)`, returning `-2.0`, `NaN` and `+1.0`. The
/// referent is a property of a dose, not of an engine, so it is stated once.
///
/// **[`crate::types::Subject::data_tad`] is deliberately not folded in** (#1182/#1273). It
/// answers a different question — what the *dataset* says `TAD` is, before any model, with no
/// lagtime at all and a `same_time_counts` switch between NONMEM's record order and Pharmpy's
/// `get_doseid` grouping. Being lag-free, the pre-arrival window this function exists for is
/// empty there (record and arrival coincide), so it needs nothing from here; and this function
/// has no tie-convention parameter, so it could not serve there without growing one. Two
/// kernels, two questions — recorded rather than left to look like an oversight.
///
/// Three cases:
///
/// * **The dose has arrived** (`t ≥ dose.time + lag`). An ordinary dose is measured from its
///   own arrival. A **steady-state** dose describes a periodic pulse train, so the elapsed
///   time folds back into `[0, II)` — as though virtual pulses had landed at
///   `dose.time + lag + k·II`.
/// * **A steady-state dose's pre-arrival window** (`dose.time ≤ t < dose.time + lag`) —
///   #1126, and the case this function exists for. Since #1121 the periodic state is loaded
///   at the dose **record** and flows to the lagged arrival, so the state there is real: it
///   is the previous cycle's decaying tail. That cycle's pulse is at
///   `dose.time − ss_seed_phase(dose, lag)`, which is exactly the instant
///   [`ss_state_at_phase`](crate::ode::predictions) advances from, so `TAD` and the state it
///   describes are built from one number.
/// * **Anything earlier** — no referent from this dose.
///
/// # Why the pre-arrival branch is gated on [`ss_seeded_at_record`] and not re-derived
///
/// `dose.ss && dose.ii > 0.0 && arrival > t ≥ dose.time` already implies `lag > 0`, so
/// spelling the predicate out would be equivalent *today*. It is written as the shared
/// predicate because it is the same question: `ss_seeded_at_record` decides whether the
/// window carries the previous cycle's tail at all, and if that ever stops being true this
/// referent stops being the right one in the same edit.
///
/// # `ss_seed_phase` is not `rem_euclid`, and the difference is the whole point
///
/// The post-arrival branch's `elapsed.rem_euclid(dose.ii)` would *also* return
/// `dose.time + lag − II` for a negative `elapsed` — for `0 < lag ≤ II`. It stops agreeing at
/// `lag > II`, where [`ss_seed_phase`] **clamps** to `0` on NONMEM evidence
/// (`nonmem_anchor/results/ss_lag_ge_ii.tab`, measured in #1121) while `rem_euclid` wraps:
/// at `lag = 15, II = 12` they are `9.0` apart. Reaching the pre-arrival window by relaxing
/// the post-arrival filter — #1126's own suggested shortcut — gets that case wrong, and gets
/// it wrong silently, because the two agree everywhere a test written for `lag < II` looks.
///
/// Under the clamp `TAD` runs `0 … lag` across the window and so exceeds `II`. That is what
/// the clamp *means*: the pulse lands on the record and nothing intervenes before the real
/// arrival, so there is no shorter elapsed time to report.
#[inline]
pub(crate) fn tad_referent(dose: &DoseEvent, lag: f64, t: f64) -> Option<f64> {
    let arrival = dose.time + lag;
    if arrival <= t + TAD_EPS {
        if dose.ss && dose.ii > 0.0 {
            let elapsed = t - arrival;
            return Some(t - elapsed.rem_euclid(dose.ii));
        }
        return Some(arrival);
    }
    if ss_seeded_at_record(dose, lag) && dose.time <= t + TAD_EPS {
        return Some(dose.time - ss_seed_phase(dose, lag));
    }
    None
}

/// `TAD` at `t` — [`tad_referent`] folded over a dose list, `NaN` when no dose has a referent.
///
/// The **reporting** answer: the per-dose rule was the interesting duplication, but the fold
/// around it — `fold(NEG_INFINITY, f64::max)`, then `is_finite`, then `NaN` — was written out
/// beside it too. `api::output_columns::tad_at_time` is its caller. (`io::output`'s sdtab
/// fallback was a third copy when #1126 was written; #1273 moved that call site to
/// [`crate::types::Subject::data_tad`] instead, for the tie rule described on
/// [`tad_referent`], so it is not one of ours.)
///
/// `dose_lagtimes` may be **shorter than `doses`, including empty**; a missing entry is zero
/// lag, matching [`crate::ode::predictions::tad_anchor_for`] and `active_infusions`.
///
/// [`crate::ode::predictions::tad_anchor_for`] deliberately does **not** call this. It shares
/// the fold but not the tail: where nothing has a referent it falls back to the subject's
/// earliest lagged arrival, because a `NaN` anchor multiplies into the integrator state and
/// poisons every later prediction, while a *reported* `TAD` of "not dosed yet" is `NaN` by the
/// sdtab convention and a negative number there would be worse than blank. Two different
/// questions about the same fold, kept apart on purpose.
///
/// `f64::max` discards `NaN`, so a `Some(NaN)` referent would be silently dropped and the
/// answer taken from the remaining doses. [`tad_referent`] cannot produce one — every path to
/// `Some` returns a value built from a `dose.time`/`lag` pair that already passed a finite
/// comparison, and a non-finite pair falls through to `None` — and
/// `tad_referent_never_returns_a_nan_referent` pins that rather than leaving it to inspection.
#[inline]
pub(crate) fn tad_at(doses: &[DoseEvent], dose_lagtimes: &[f64], t: f64) -> f64 {
    let last = doses
        .iter()
        .enumerate()
        .filter_map(|(i, d)| tad_referent(d, dose_lagtimes.get(i).copied().unwrap_or(0.0), t))
        .fold(f64::NEG_INFINITY, f64::max);
    if last.is_finite() {
        t - last
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod governing_record_tests {
    use super::governing_record_indices;

    /// `true` at every index in `records`.
    fn idx(n: usize, records: &[usize]) -> Vec<Option<usize>> {
        governing_record_indices(n, |i| records.contains(&i))
    }

    #[test]
    fn a_record_governs_itself_and_a_non_record_takes_the_next_record_ahead() {
        // slots:   0=DoseRecord  1=Dose(arrival)  2=Obs  3=InfusionEnd  4=Obs
        let got = idx(5, &[0, 2, 4]);
        assert_eq!(
            got,
            vec![Some(0), Some(2), Some(2), Some(4), Some(4)],
            "a record resolves to itself; a non-record to the next record ahead"
        );
    }

    #[test]
    fn the_trailing_tail_keeps_the_last_record_that_ran() {
        // Two non-records after the final record: nothing ahead terminates their
        // segments, so both keep the last record. This is the rule the four engines
        // used to split two-and-two on — `ode/predictions` and `sens/ode_provider`
        // took the previous record, `pk/event_driven` and `sens/propagate` took the
        // dose row — and it is why the resolution lives here rather than four times
        // over. It is not observable through predictions (no observation follows such
        // a boundary), which is exactly why it needs pinning at the resolver.
        let got = idx(5, &[0, 2]);
        assert_eq!(got, vec![Some(0), Some(2), Some(2), Some(2), Some(2)]);
    }

    #[test]
    fn a_timeline_with_no_record_at_all_resolves_to_none_everywhere() {
        // A subject with no observation, no EVID=2 row and no dose row produces no
        // prediction; the callers fall back to their own seed snapshot there.
        assert_eq!(idx(3, &[]), vec![None, None, None]);
        assert_eq!(idx(0, &[]), Vec::<Option<usize>>::new());
    }

    #[test]
    fn adjacent_records_each_govern_their_own_slot() {
        // Co-timed records land as adjacent slots (the sort is stable and the
        // zero-length segment between them is skipped by the walk). Each must resolve
        // to itself, not to its neighbour — otherwise the segment arriving at a shared
        // instant would read the wrong one of the two.
        assert_eq!(idx(3, &[0, 1, 2]), vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn a_leading_non_record_takes_the_first_record_ahead() {
        // A dose arrival before any observation still runs on the record that
        // terminates its interval.
        assert_eq!(idx(3, &[2]), vec![Some(2), Some(2), Some(2)]);
    }
}

#[cfg(test)]
mod tad_referent_tests {
    use super::{ss_seed_phase, tad_referent};
    use crate::types::DoseEvent;

    /// `SS=1, II=12` bolus whose **record** is at `t = 480`.
    fn ss_dose() -> DoseEvent {
        DoseEvent::new(480.0, 100.0, 1, 0.0, true, 12.0)
    }

    /// Ordinary bolus at `t`, no steady state.
    fn plain(t: f64) -> DoseEvent {
        DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0)
    }

    /// `TAD` as a caller sees it — the folds in `ode::predictions::tad_anchor_for` and
    /// `api::output_columns::tad_at_time` both report `t - referent`, so the tests below read
    /// in the units the model does rather than in anchor times.
    fn tad(dose: &DoseEvent, lag: f64, t: f64) -> Option<f64> {
        tad_referent(dose, lag, t).map(|a| t - a)
    }

    /// The window #1126 is about: `[480, 483)` with `ALAG = 3`, `II = 12`.
    ///
    /// The pulse that produced the state there landed at `480 + 3 − 12 = 471`, so `TAD` runs
    /// `9 … 12` across the window and resets to `0` when the real dose arrives. Before #1126
    /// the fold had no candidate at all here — the sdtab column came out blank and the
    /// integrators fell back to the subject's first *arrival*, giving a negative `TAD`.
    #[test]
    fn a_seeded_ss_dose_measures_the_pre_arrival_window_from_the_previous_cycles_pulse() {
        let d = ss_dose();
        assert_eq!(
            tad(&d, 3.0, 480.0),
            Some(9.0),
            "the record sits at phase II - lag"
        );
        assert_eq!(tad(&d, 3.0, 481.0), Some(10.0));
        assert_eq!(tad(&d, 3.0, 482.0), Some(11.0));
        // The instant *before* the arrival is a full interval after the previous pulse …
        let just_before = tad(&d, 3.0, 483.0 - 1e-9).expect("inside the window");
        assert!(
            (just_before - 12.0).abs() < 1e-8,
            "TAD reaches II at the end of the window, got {just_before}"
        );
        // … and the arrival itself restarts the cycle.
        assert_eq!(tad(&d, 3.0, 483.0), Some(0.0));
        assert_eq!(
            tad(&d, 3.0, 485.0),
            Some(2.0),
            "back on the post-arrival fold"
        );
    }

    /// Before the dose **record** an `SS=1` row contributes nothing: `SS=1` establishes steady
    /// state *at* the record, so it must not leak into earlier observations. The same rule the
    /// analytical superposition states at `crate::pk::predict_concentration`.
    #[test]
    fn an_ss_record_has_no_referent_before_itself() {
        assert_eq!(tad_referent(&ss_dose(), 3.0, 479.0), None);
        assert_eq!(tad_referent(&ss_dose(), 3.0, 0.0), None);
    }

    /// **`ss_seed_phase`, not `rem_euclid`.** This is the case that separates the two, and it
    /// is a test rather than a comment because every `lag < II` fixture agrees either way.
    ///
    /// At `lag = 15, II = 12` there is no phase `II − lag`. #1121 measured NONMEM 7.6.0
    /// **clamping** it to zero — the pulse lands on the record, which then carries the
    /// steady-state peak and decays with nothing intervening before the real arrival
    /// (`nonmem_anchor/results/ss_lag_ge_ii.tab`). So `TAD` runs `0 … lag` and *exceeds* `II`,
    /// which is what the clamp means. `rem_euclid` would wrap to 471 at the record — 9.0 away,
    /// and silently, since the two agree everywhere else.
    #[test]
    fn a_lagtime_of_a_full_interval_or_more_clamps_the_phase_instead_of_wrapping() {
        let d = ss_dose();
        let (lag, ii) = (15.0, 12.0);
        assert_eq!(ss_seed_phase(&d, lag), 0.0, "the clamp itself");
        assert_eq!(tad(&d, lag, 480.0), Some(0.0));
        assert_eq!(tad(&d, lag, 485.0), Some(5.0));
        assert_eq!(
            tad(&d, lag, 494.0),
            Some(14.0),
            "TAD exceeds II under the clamp"
        );
        assert_eq!(tad(&d, lag, 495.0), Some(0.0), "the real arrival");

        // What the rejected spelling would have returned, computed here rather than
        // described: the post-arrival branch's wrap, evaluated inside the window.
        let wrapped = 480.0 - (480.0 - (480.0 + lag)).rem_euclid(ii);
        assert_eq!(wrapped, 471.0);
        assert_eq!(
            tad_referent(&d, lag, 480.0).expect("clamped referent") - wrapped,
            9.0,
            "the two spellings are 9.0 apart at lag = 15, II = 12 — a fixture with lag < II \
             cannot tell them apart"
        );
    }

    /// The scoping guard. #1126 suggested reaching the window by relaxing the post-arrival
    /// filter for SS doses; applied to *every* dose it flips an ordinary lagged regimen.
    ///
    /// 0/24 dosing with `lag = 3`: at `t = 25` the most recent **arrival** is the first dose's,
    /// at `t = 3`, so `TAD = 22`. The second dose's record is at 24 but it has not landed yet,
    /// and a relaxed filter would report `25 − 27 = −2`.
    #[test]
    fn an_ordinary_lagged_dose_keeps_its_arrival_filter() {
        let (d0, d1) = (plain(0.0), plain(24.0));
        assert_eq!(tad(&d0, 3.0, 25.0), Some(22.0));
        assert_eq!(
            tad_referent(&d1, 3.0, 25.0),
            None,
            "a non-SS dose contributes nothing before it arrives — the referent added by \
             #1126 is gated on `ss_seeded_at_record`"
        );
        assert_eq!(
            tad(&d1, 3.0, 27.0),
            Some(0.0),
            "…and everything once it does"
        );
    }

    /// `lag = 0` leaves no window to have a referent for, so the answer is the fold that stood
    /// before #1126, exactly. Pinned because the new branch must not fire here: every caller
    /// that has no lagtime concept (`ode::ekf`, the capped input-rate train) passes an empty
    /// lag slice, and this is what makes those call sites provably unchanged.
    #[test]
    fn a_zero_lagtime_is_the_pre_1126_fold_exactly() {
        let d = ss_dose();
        assert_eq!(tad(&d, 0.0, 480.0), Some(0.0));
        assert_eq!(tad(&d, 0.0, 485.0), Some(5.0));
        assert_eq!(tad(&d, 0.0, 492.0), Some(0.0), "one full interval on");
        assert_eq!(tad_referent(&d, 0.0, 479.9), None);
    }

    /// `tad_at` folds `tad_referent` and reports `NaN` where nothing has a referent — the
    /// *reporting* tail, deliberately different from `tad_anchor_for`'s first-arrival
    /// fallback. Both spellings of the fold used to be written out by hand beside each other.
    #[test]
    fn tad_at_folds_the_referents_and_reports_nan_when_none_exists() {
        use super::tad_at;
        let doses = [plain(0.0), ss_dose()];
        // Inside the SS dose's pre-arrival window the SS referent (471) outranks the
        // ordinary dose's arrival (3), so the fold must pick it rather than the max time.
        assert_eq!(tad_at(&doses, &[3.0, 3.0], 481.0), 10.0);
        // Past the SS arrival it is back on the periodic fold.
        assert_eq!(tad_at(&doses, &[3.0, 3.0], 485.0), 2.0);
        // A short slice means zero lag on the remaining doses, and `&[]` means all of them.
        assert_eq!(
            tad_at(&doses, &[3.0], 485.0),
            5.0,
            "dose 1 unlagged: pulse at 480"
        );
        assert_eq!(tad_at(&doses, &[], 485.0), 5.0);
        // Nothing has a referent yet.
        assert!(tad_at(&doses, &[3.0, 3.0], -1.0).is_nan());
        // …and a dose-free subject.
        assert!(tad_at(&[], &[], 5.0).is_nan());
    }

    /// `tad_at` folds with `f64::max`, which **discards** `NaN` — so a `Some(NaN)` referent
    /// would be silently dropped and the answer taken from the other doses.
    ///
    /// `tad_referent` cannot produce one, and this pins that rather than leaving it to
    /// inspection: every non-finite `dose.time`/`lag`/`ii` combination must come back `None`
    /// (a comparison against `NaN` is `false`, so both guards fall through), and the one
    /// infinite case that *does* pass a guard must still yield a finite referent.
    #[test]
    fn tad_referent_never_returns_a_nan_referent() {
        let mut nan_time = ss_dose();
        nan_time.time = f64::NAN;
        assert_eq!(tad_referent(&nan_time, 3.0, 481.0), None, "NaN dose time");

        let d = ss_dose();
        assert_eq!(tad_referent(&d, f64::NAN, 481.0), None, "NaN lag");
        assert!(
            tad_referent(&d, 3.0, f64::NAN).is_none(),
            "NaN observation time"
        );

        let mut nan_ii = ss_dose();
        nan_ii.ii = f64::NAN;
        // `ii > 0.0` is false for NaN, so this takes the ordinary-dose arm and never reaches
        // `rem_euclid(NaN)`.
        assert_eq!(tad_referent(&nan_ii, 3.0, 485.0), Some(483.0));

        let mut inf_ii = ss_dose();
        inf_ii.ii = f64::INFINITY;
        // `elapsed.rem_euclid(inf)` is `elapsed` for a positive elapsed, so the referent is
        // the arrival — finite, not NaN.
        assert_eq!(tad_referent(&inf_ii, 3.0, 485.0), Some(483.0));

        // The whole point: had any of the above returned `Some(NaN)`, this fold would have
        // hidden it behind the ordinary dose.
        let doses = [plain(0.0), nan_time];
        assert_eq!(super::tad_at(&doses, &[3.0, 3.0], 481.0), 478.0);
    }

    /// **A cross-PR invariant nothing else enforces.**
    ///
    /// `crate::types::Subject::time_after_dose_at_or_before` (#1182/#1273) documents itself as
    /// keeping "the same tie rule as the lag-aware `tad_at_time`" — i.e. as *this* fold under
    /// zero lag. That is a claim about code in another module, written by another change, and
    /// #1273's own test compares its two conventions against each other rather than against
    /// this one. So nothing would have gone red if #1126 had moved the tie while collapsing
    /// the folds: `tad_referent` picks `arrival <= t + TAD_EPS`, and a strict `<` there would
    /// have been just as plausible a spelling.
    ///
    /// Pinned on the row where the two data conventions are known to differ from each other
    /// (`obs@24` with `dose@0, dose@24`), so this sits exactly on the boundary rather than
    /// somewhere every spelling agrees.
    #[test]
    fn the_zero_lag_fold_breaks_a_same_time_tie_the_way_the_sdtab_fallback_does() {
        use crate::types::Subject;
        let subject = Subject {
            id: "1".into(),
            doses: vec![plain(0.0), plain(24.0)],
            obs_times: vec![0.0, 2.0, 23.9, 24.0, 30.0],
            ..Default::default()
        };
        let mine: Vec<f64> = subject
            .obs_times
            .iter()
            .map(|&t| super::tad_at(&subject.doses, &[], t))
            .collect();
        let theirs: Vec<f64> = (0..subject.obs_times.len())
            .map(|j| subject.time_after_dose_at_or_before(j))
            .collect();
        assert_eq!(
            mine, theirs,
            "the zero-lag fold and the sdtab fallback must agree on every row, including the \
             same-time tie at t = 24 where a dose record and an observation share a TIME"
        );
        // The straddle: assert the fixture actually reaches the tie, so this cannot quietly
        // become a comparison of two functions on rows where every spelling agrees.
        assert_eq!(
            mine[3], 0.0,
            "obs@24 takes the dose at 24 — NONMEM record order"
        );
        assert_eq!(
            subject.time_after_dose(3),
            24.0,
            "…and Pharmpy's `get_doseid` convention deliberately does NOT. If this stops \
             differing, the fixture no longer sits on the boundary and the assertion above \
             has stopped meaning anything"
        );

        // ### The steady-state arm, which is the half that is literally duplicated
        //
        // `Subject::data_tad`'s SS branch is `obs_t - elapsed.rem_euclid(d.ii)` — the same
        // expression as [`tad_referent`]'s, character for character. The non-SS rows above
        // do not reach it, and neither does #1273's own `subject_time_after_dose_is_the_data_tad`
        // (two plain doses). So until this block, the one genuinely copy-pasted line between
        // the two kernels was compared by nothing at all.
        //
        // The two cannot be collapsed: `data_tad` needs a tie-convention switch this function
        // has no use for, and this function needs a lag argument and the seeded pre-arrival
        // branch `data_tad` has no use for. They stay two — so the periodic fold they share
        // is pinned here instead, over a full cycle plus the wrap.
        let ss_subject = Subject {
            id: "1".into(),
            doses: vec![ss_dose()],
            obs_times: vec![480.0, 483.0, 485.0, 491.5, 492.0, 495.0],
            ..Default::default()
        };
        let ss_mine: Vec<f64> = ss_subject
            .obs_times
            .iter()
            .map(|&t| super::tad_at(&ss_subject.doses, &[], t))
            .collect();
        let ss_theirs: Vec<f64> = (0..ss_subject.obs_times.len())
            .map(|j| ss_subject.time_after_dose_at_or_before(j))
            .collect();
        assert_eq!(
            ss_mine, ss_theirs,
            "the periodic SS fold is written out in both kernels; they must not drift"
        );
        // And the fixture has to actually wrap, or it is comparing two functions on a
        // monotone ramp where any fold agrees.
        assert_eq!(ss_mine[0], 0.0, "the record itself");
        // 491.5, not 491.9: `491.9 - 480.0` is `11.899999999999977` in binary, so a
        // literal `11.9` here fails on the float and not on the fold. Halves are exact.
        assert_eq!(ss_mine[3], 11.5, "just before the next virtual pulse");
        assert_eq!(ss_mine[4], 0.0, "…and the wrap back to zero at II");
    }

    /// `SS=1` with a non-positive `II` is not a periodic train — `W_STEADY_STATE_II` warns and
    /// every engine treats the dose as an ordinary bolus. Both arms must therefore skip the SS
    /// branches, the new one included, or the pre-arrival window would be filled from an
    /// `ss_seed_phase` of `0 − lag`.
    #[test]
    fn an_ss_dose_with_no_interval_is_treated_as_an_ordinary_dose() {
        let d = DoseEvent::new(480.0, 100.0, 1, 0.0, true, 0.0);
        assert_eq!(tad_referent(&d, 3.0, 481.0), None, "not yet arrived");
        assert_eq!(tad(&d, 3.0, 483.0), Some(0.0));
        assert_eq!(
            tad(&d, 3.0, 500.0),
            Some(17.0),
            "no wrap without an interval"
        );
    }
}
