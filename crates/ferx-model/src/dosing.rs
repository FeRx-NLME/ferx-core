//! Neutral dose-resolution + steady-state-equilibration primitives.
//!
//! Kept out of `ode/` so `pk/`, `sens/` and `api` need not depend upward on it;
//! every consumer reaches these through `crate::dosing::…`.
//!
//! - Dose resolution (`resolve_subject_doses{,_with}`): modeled-`RATE` doses to
//!   concrete `Fixed` doses (#324).
//! - Exact periodic steady state (`periodic_ss_fixed_point_g`): `u_ss = (I − M)⁻¹·b`
//!   for a linear one-cycle map (ODE #835, analytical #908).
//! - SS equilibration (`SS_EQUILIBRATION_*`, `ss_cycle_converged`, `SsStopTracker`):
//!   the pulse-train fallback, shared by every SS loop so troughs can't drift apart.

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
/// Every ODE entrypoint funnels its subject through this, so a coded `RATE=-2`
/// never reaches the integrator unresolved. The owned branch clones the whole
/// `Subject` (not just `doses`) because downstream helpers read `obs_times` /
/// `pk_only_times` / `reset_times` alongside the resolved doses.
pub fn resolve_subject_doses_with<'a>(
    subject: &'a Subject, attr_map: &crate::types::DoseAttrMap,
    pk_for_dose: impl Fn(usize) -> &'a [f64],
) -> Cow<'a, Subject> {
    // Fast path: an empty map means no modeled dose (the data gate rejects one).
    let base = lower_zero_order_ss(subject, attr_map);
    if (attr_map.is_empty() && !attr_map.has_zero_order()) || base.all_doses_fixed() {
        return base;
    }
    let mut owned = base.into_owned();
    for (k, d) in owned.doses.iter_mut().enumerate() {
        *d = d.resolve_rate(attr_map, pk_for_dose(k));
    }
    Cow::Owned(owned)
}

/// Whether `d` is a plain `SS=1` bolus record into a compartment fed by one `zero_order()`
/// window and nothing else (#2786) — the doses [`lower_zero_order_ss`] rewrites.
fn is_zero_order_ss_record(attr_map: &crate::types::DoseAttrMap, d: &DoseEvent) -> bool {
    d.ss && d.ii > 0.0
        && d.amt > 0.0
        && matches!(d.rate_mode, crate::types::RateMode::Fixed)
        && !d.is_infusion()
        && attr_map.zero_order_slot(d.cmt_1based()).is_some()
}

/// Whether `d` is such a dose **after** lowering: a duration-modeled `SS=1` infusion whose
/// compartment is a pure zero-order one. A user-coded infusion into a zero-order compartment is
/// refused by validation, so this names the lowered doses and only them. Every walk asks it to
/// keep such a dose on the plain-infusion paths (no forcing onset, no `R_in_inf`).
pub fn is_lowered_zero_order_ss(attr_map: &crate::types::DoseAttrMap, d: &DoseEvent) -> bool {
    d.ss && d.ii > 0.0 && d.is_infusion() && attr_map.zero_order_slot(d.cmt_1based()).is_some()
}

/// Rewrite every `SS=1` bolus into a pure zero-order compartment as a duration-modeled
/// infusion (`RATE=-2` with the forcing's `dur` as its `D{cmt}`), #2786.
///
/// A zero-order window *is* an infusion (rate `F·amt/dur` over `[t + lag, t + lag + dur]`), and
/// the engines already carry an SS infusion exactly in every derivative order, so lowering reuses
/// that instead of a second SS train for `zero_order()`. Borrowed unless a dose is rewritten.
pub fn lower_zero_order_ss<'a>(
    subject: &'a Subject, attr_map: &crate::types::DoseAttrMap,
) -> Cow<'a, Subject> {
    if !attr_map.has_zero_order()
        || !subject.doses.iter().any(|d| is_zero_order_ss_record(attr_map, d))
    {
        return Cow::Borrowed(subject);
    }
    let mut owned = subject.clone();
    lower_zero_order_ss_in_place(&mut owned, attr_map);
    Cow::Owned(owned)
}

/// Test seam: subjects lowered whose first dose carries the marker amount of this module's fit
/// fixture, so a test counts its own fit's lowerings among the parallel suite's.
#[cfg(test)]
static MARKER_LOWERINGS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn lower_zero_order_ss_in_place(subject: &mut Subject, attr_map: &crate::types::DoseAttrMap) {
    #[cfg(test)]
    if subject.doses.first().is_some_and(|d| d.amt == 4321.5) {
        MARKER_LOWERINGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    for d in subject.doses.iter_mut().filter(|d| is_zero_order_ss_record(attr_map, d)) {
        d.rate_mode = crate::types::RateMode::ModeledDuration;
        d.infusion_def = crate::types::InfusionDef::DurationDefined;
        d.rate = 0.0;
        d.duration = 0.0;
        d.resolve_default_cmt();
    }
}

/// [`lower_zero_order_ss`] for a whole population, once where a fit binds it to its model:
/// the lowered doses are infusions no walk rewrites again, so the engines' own
/// [`lower_zero_order_ss`] on a bound subject is a scan that borrows instead of cloning the
/// subject on every walk of every inner solve. Borrowed unless a dose is rewritten.
pub fn lower_population_zero_order_ss<'a>(
    population: &'a crate::types::Population, attr_map: &crate::types::DoseAttrMap,
) -> Cow<'a, crate::types::Population> {
    let has = |s: &Subject| s.doses.iter().any(|d| is_zero_order_ss_record(attr_map, d));
    if !attr_map.has_zero_order() || !population.subjects.iter().any(has) {
        return Cow::Borrowed(population);
    }
    let mut owned = population.clone();
    for s in owned.subjects.iter_mut().filter(|s| has(s)) {
        lower_zero_order_ss_in_place(s, attr_map);
    }
    Cow::Owned(owned)
}

/// Resolve modeled-`RATE` doses using `params` for **every** dose — the
/// no-time-varying-covariate ODE paths, where the PK snapshot is constant across
/// doses. The event-driven / TV-covariate path calls
/// [`resolve_subject_doses_with`] directly with a per-dose closure. See
/// [`resolve_subject_doses_with`].
pub fn resolve_subject_doses<'a>(
    subject: &'a Subject, attr_map: &crate::types::DoseAttrMap, params: &'a [f64],
) -> Cow<'a, Subject> {
    resolve_subject_doses_with(subject, attr_map, |_| params)
}

/// For each slot of an ordered event timeline, the index of the **record that
/// governs the segment ending there** (#1073).
///
/// NONMEM evaluates `$PK` at each record and advances *to* it, so an interval is governed
/// by the record that **terminates** it; non-records (lagged arrival, infusion end, onset)
/// only subdivide it. Returns indices, not values, so callers map to their own snapshot type
/// without copying a `PkDual<Dual2<N>>` per event.
///
/// Both rules must match in all four walks (`ode::predictions`, `pk::event_driven`,
/// `sens::propagate`, `sens::ode_provider`): a difference is a cross-engine divergence that
/// FD parity cannot see.
///
/// 1. A record resolves to **itself**, a non-record to the **next record ahead**.
/// 2. Past the final record, slots keep the **last record that ran** (forward fill).
///    Not observable through predictions today (no observation follows such a boundary), but the
///    four walks once split between the last record and the dose row, which would become live
///    once a trailing record, a reset or a second pass reads the state.
///
/// `None` in every slot only when the timeline has no record at all.
pub fn governing_record_indices(
    n_events: usize, is_record: impl Fn(usize) -> bool,
) -> Vec<Option<usize>> {
    let mut acc: Vec<Option<usize>> = vec![None; n_events];
    // Rule 1: backward scan; `seen` is set before `acc[i]`, so a record maps to itself.
    let mut seen: Option<usize> = None;
    for i in (0..n_events).rev() {
        if is_record(i) {
            seen = Some(i);
        }
        acc[i] = seen;
    }
    // Rule 2: forward fill the trailing tail.
    let mut last: Option<usize> = None;
    for slot in acc.iter_mut() {
        match *slot {
            Some(q) => last = Some(q),
            None => *slot = last,
        }
    }
    acc
}

/// One of a subject's records, by kind and index into that kind's vector: `subject.doses`,
/// `subject.pk_only_times`, `subject.obs_times`. What [`first_record`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstRecord {
    Dose(usize),
    PkOnly(usize),
    Obs(usize),
}

/// The subject's **first record**, whose `$PK` snapshot seeds `init(state) = expr` (#2121).
/// `None` when the subject has no dose, EVID=2 or observation record.
///
/// NONMEM seeds `A_0` from the first record **in file order, of any kind**
/// (`nonmem_anchor/init_seed_tie`). ferx uses the walk's order instead — smallest time, then
/// kind (dose < pk-only < observation), then index — which the reader's #614/#1810 ULP moves
/// make equal to file order. The seed must use the walk's kind order (49.6 OFV on
/// `init_seed_tie_ofv_a` otherwise).
///
/// Only `subject.doses[..n_doses]` are candidates (the adaptive replay appends injected doses
/// that are not data records). Resets are not candidates: they re-seed from their own row
/// (#1133). Every engine resolves the seed through this one function (#2133).
pub fn first_record(subject: &Subject, n_doses: usize) -> Option<FirstRecord> {
    let mut best: Option<(f64, FirstRecord)> = None;
    // Kinds in walk order under a strict `<`, so the earlier kind / first row wins a tie.
    let mut consider = |t: f64, rec: FirstRecord| {
        if best.is_none_or(|(bt, _)| t < bt) {
            best = Some((t, rec));
        }
    };
    for (k, d) in subject.doses.iter().take(n_doses).enumerate() {
        consider(d.time, FirstRecord::Dose(k));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        consider(t, FirstRecord::PkOnly(m));
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        consider(t, FirstRecord::Obs(j));
    }
    best.map(|(_, rec)| rec)
}

/// Dosing cycles simulated when pre-equilibrating an SS=1 dose. For t₁/₂/II < 2 the
/// truncation tail at N=50 is well below 1e-6. Shared with the sensitivity SS loop so
/// their troughs can't drift (#473).
pub const SS_EQUILIBRATION_CYCLES: usize = 50;

/// Relative-`L∞` tolerance for the SS equilibration **early stop** (#519). The cycle is a
/// geometric contraction (`≈ exp(−λ·II)`), so once a step is below this the truncation is at
/// f64 precision. Conservative: the dropped tail is far below the provider-vs-production parity
/// tolerance, so values are unchanged for any realistic PK. `λ·II ≈ 2` stops in ~14 cycles; `λ·II ≈ 0.1` runs the full
/// [`SS_EQUILIBRATION_CYCLES`].
pub const SS_EQUILIBRATION_TOL: f64 = 1e-12;

/// Exact periodic steady state of a **linear** one-cycle map, `u_ss = (I − M)⁻¹·b`, generic over
/// the numeric type `T` — the engine-neutral replacement for iterating a truncated pulse train.
///
/// One cycle of a linear disposition is affine, `u ↦ M·u + b`. The caller injects
/// `advance_unforced` (disposition alone: drift `z0` and the columns of `M`) and
/// `advance_forced` (with the cycle's dose: `b` and the linearity check); this body owns the
/// `T`-typed algebra. Over a dual `T` the solve carries `∂u_ss/∂(θ,η)` automatically.
///
/// Callers: the ODE path (#835, RK45, so `vtol` scales with its `reltol`/`abstol`; may be
/// nonlinear) and the analytical walks (#908, exact closed-form propagators, so `reltol = abstol = 0`
/// leaves `vtol = 1e-9·scale`, the pure rounding floor; their disposition is always linear, so the
/// self-check is a can't-happen assertion there).
///
/// Returns `None` on a nonlinear map, a singular `I − M` (a zero eigenvalue: no periodic SS,
/// so the capped fallback plus [`note_ss_nonconvergence_if_capped`] applies), or a non-finite
/// intermediate.
pub fn periodic_ss_fixed_point_g<T, FUnf, FFor>(
    n: usize, ii: f64, reltol: f64, abstol: f64, advance_unforced: FUnf, advance_forced: FFor,
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
    // `|x| < ∞`, as a branch (a `NaN` fails it too): a guard on the verdict, not on the value.
    if u_ss.iter().any(|x| !x.br_abs_lt(f64::INFINITY)) {
        return None;
    }
    // Linearity check on the value part: one and two forced cycles from u_ss must both return
    // u_ss (two rejects a weakly nonlinear map that happens to pass one). Branches on a
    // difference vs a tolerance, so a recording holds while the verdict does; the solution
    // itself does not depend on the verdict. The scale is a magnitude only (`abs_hint`).
    let scale = u_ss.iter().fold(1e-12_f64, |a, x| a.max(x.abs_hint()));
    // A linear map's ODE residual is a solver-noise floor ≈ 45·reltol (different adaptive
    // steps for `b` and the check); `256·reltol` clears it ~5× (the original `32·reltol` falsely
    // declined linear models at a tight `ode_reltol`, e.g. 1e-10) while a nonlinear map's
    // O(scale) residual (~1e8·reltol) is rejected (see
    // `ss_input_rate_nonlinear_disposition_falls_back_to_iteration`). Analytical callers get
    // `1e-9·scale`, seven orders above their measured ~1e-16 residual.
    let vtol = (256.0 * reltol + 1e-9) * scale + 256.0 * abstol;
    // `|c − u| ≤ vtol` is `< next_up(vtol)`, exactly, for a finite `vtol`.
    let vtol_open = vtol.next_up();
    let within = |a: &[T], b: &[T]| a.iter().zip(b).all(|(&c, &u)| c.br_absdiff_lt(u, vtol_open));
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
/// by every f64 and dual path (the dual feeds its value parts, #519) so all stop on the same
/// cycle and the truncated gradient is the derivative of the truncated value (see
/// [`crate::sens::propagate::ss_dual_cycle_should_stop`]).
///
/// Mixed test on the per-cycle **increment** (#532): `|cur − prev| ≤ tol·|cur| + tol·max_mag`.
/// Testing the increment keeps a small compartment still in transit from passing for being
/// small; the `tol·max_mag` floor lets a settled near-zero compartment pass (without it the loop
/// could never stop). The stop fires only once every increment is below f64-relative precision,
/// so predictions are unchanged to f64 precision; gradients matched a full budget to `< 1e-6`
/// (measured pre-#2093; since #914 this early stop runs only on the nonlinear fallback).
///
/// A non-finite compartment never converges, so a blow-up runs the full budget and surfaces
/// (`f64::max` would otherwise drop a `NaN`).
pub fn ss_cycle_converged(cur: &[f64], prev: &[f64], tol: f64) -> bool {
    // Test-only: force the full cycle budget to compare against the early stop (#532).
    #[cfg(test)]
    if FORCE_FULL_SS_EQUILIBRATION.with(|c| c.get()) {
        return false;
    }
    if cur.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let max_mag = cur.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let atol = tol * max_mag;
    cur.iter().zip(prev).all(|(&a, &b)| (a - b).abs() <= tol * a.abs() + atol)
}

/// The test-only full-budget switch, for the recording type's convergence test.
#[cfg(test)]
pub fn force_full_ss_equilibration() -> bool {
    FORCE_FULL_SS_EQUILIBRATION.with(|c| c.get())
}

/// Rolling prev-state tracker for the f64 SS-equilibration early stop, shared by the f64
/// loops so they cannot drift apart (#532). The dual paths use
/// [`crate::sens::propagate::ss_dual_cycle_should_stop`] with the same criterion.
#[derive(Default)]
pub struct SsStopTracker {
    prev: Vec<f64>,
    /// **Absolute** L∞ increments of the two most recent non-converged cycles, oldest first
    /// (#867 cap check). Absolute on purpose: a linearly diverging trough takes ~constant steps,
    /// so its ratio reads `ρ ≈ 1`, while a relative step would shrink and hide the divergence.
    abs_increments: [f64; 2],
    /// L∞ magnitude of the most recent non-converged cycle (scale for the tail and noise floor).
    last_mag: f64,
}

impl SsStopTracker {
    /// Record `cur` and report whether the trough has converged (from cycle 1 on). Returns
    /// `true` to break the equilibration loop.
    pub fn should_stop(&mut self, cycle: usize, cur: &[f64]) -> bool {
        if cycle > 0 {
            if ss_cycle_converged(cur, &self.prev, SS_EQUILIBRATION_TOL) {
                return true;
            }
            // Roll the increment history for the cap check (#867).
            self.abs_increments = [self.abs_increments[1], ss_abs_increment(cur, &self.prev)];
            self.last_mag = ss_max_magnitude(cur);
        }
        self.prev.clear();
        self.prev.extend_from_slice(cur);
        false
    }

    /// `(prev_abs, last_abs, last_mag)` for [`note_ss_nonconvergence_if_capped`] (#867); all
    /// `0.0` ("nothing to warn about") after an early stop or fewer than two such cycles.
    pub fn recent_increments(&self) -> (f64, f64, f64) {
        (self.abs_increments[0], self.abs_increments[1], self.last_mag)
    }
}

/// Absolute L∞ change `max|cur − prev|` between SS cycles (#867); see
/// [`SsStopTracker::abs_increments`].
fn ss_abs_increment(cur: &[f64], prev: &[f64]) -> f64 {
    cur.iter().zip(prev).fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()))
}

/// L∞ magnitude `max|cur|` of a cycle state — the scale the absolute tail is normalised against
/// (#867). `0.0` when `cur` is all-zero (nothing has accumulated).
fn ss_max_magnitude(cur: &[f64]) -> f64 {
    cur.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
}

#[cfg(test)]
thread_local! {
    /// Cycles the most recent SS-equilibration call ran (test-only observation of the #519
    /// early stop; end-value tolerances alone would absorb a too-early exit, #532).
    static LAST_SS_EQUILIBRATION_CYCLES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// Branch the most recent SS equilibration took — see [`SsBranch`].
    #[cfg(test)]
    static LAST_SS_EQUILIBRATION_BRANCH: std::cell::Cell<SsBranch> =
        const { std::cell::Cell::new(SsBranch::None) };

    /// The warning the most recent [`note_ss_nonconvergence_if_capped`] call on this thread
    /// produced, or `None` (test-only). The process-global sink cannot be asserted on from a
    /// lib test in either direction: concurrent tests insert into it (#1289) and draining `api`
    /// entry points empty it (#2071). Reports the *last* call on this thread, so assert it only
    /// from tests that call the equilibration helpers directly; the sink path end to end is
    /// pinned by `tests/ss_absorption_nonmem_anchor.rs`.
    static LAST_SS_NONCONVERGENCE_WARNING: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };

    /// How many times this thread cleared or drained the SS sink (test-only), so a test can
    /// assert a call does **not** drain it (#2071).
    static SS_SINK_DRAINS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// When set, [`ss_cycle_converged`] never converges, forcing the full budget (#532).
    static FORCE_FULL_SS_EQUILIBRATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the most recent SS equilibration on this thread warned (test observation; see
/// [`LAST_SS_NONCONVERGENCE_WARNING`]). Use this, never the sink, in a lib test (#1289, #2071).
#[cfg(test)]
pub fn last_ss_equilibration_warned() -> bool {
    LAST_SS_NONCONVERGENCE_WARNING.with(|c| c.borrow().is_some())
}

/// The warning text the most recent SS equilibration on this thread noted, or `None` (test
/// observation; see [`LAST_SS_NONCONVERGENCE_WARNING`]). The same string
/// [`note_ss_nonconvergence_if_capped`] inserts into the sink, so a lib test can check the
/// message without racing the draining `api` entry points (#2071).
#[cfg(test)]
pub fn last_ss_equilibration_warning() -> Option<String> {
    LAST_SS_NONCONVERGENCE_WARNING.with(|c| c.borrow().clone())
}

/// How many times this thread has cleared or drained the SS non-convergence sink (test
/// observation; see [`SS_SINK_DRAINS`]).
#[cfg(test)]
pub fn ss_sink_drains_on_this_thread() -> usize {
    SS_SINK_DRAINS.with(|c| c.get())
}

/// Record the warning this SS equilibration produced, or `None` (test observation; see
/// [`LAST_SS_NONCONVERGENCE_WARNING`]).
///
/// Called with `None` at the top of every top-level equilibration, so paths that never reach
/// the capped fallback do not report a stale warning from an earlier call on this thread.
#[cfg(test)]
pub fn record_ss_nonconvergence_warning(warning: Option<&str>) {
    LAST_SS_NONCONVERGENCE_WARNING.with(|c| *c.borrow_mut() = warning.map(str::to_owned));
}

#[cfg(test)]
pub fn record_ss_equilibration_cycles(n: usize) {
    LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.set(n));
}

/// Which SS-equilibration branch the most recent call took (test-only). The branches fall
/// back to one another and return the same trough, so a value assertion cannot tell them
/// apart and a broken branch silently routes to the next (two #1139 mutations survived that
/// way): a test pinning a branch must assert which one ran. One cell per thread shared across
/// numeric types, so assert it only from tests that call the f64 helpers directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SsBranch {
    /// No SS equilibration has completed since the last one started (reset at the top of
    /// `equilibrate_ss_pk_state`, so an early bail-out leaves no stale tag).
    #[default]
    None,
    /// The exact affine fixed point `(I − M)⁻¹·b` for a bolus or infusion dose (#914).
    Exact,
    /// The same closed form on the **input-rate** (built-in absorption) path.
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
pub fn record_ss_equilibration_branch(b: SsBranch) {
    LAST_SS_EQUILIBRATION_BRANCH.with(|c| c.set(b));
}

/// Branch the most recent SS equilibration took (test observation; see [`SsBranch`]).
#[cfg(test)]
pub fn last_ss_equilibration_branch() -> SsBranch {
    LAST_SS_EQUILIBRATION_BRANCH.with(|c| c.get())
}

/// Cycles the most recent SS-equilibration call ran (test observation; see above).
#[cfg(test)]
pub fn last_ss_equilibration_cycles() -> usize {
    LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.get())
}

/// Run `f` with every SS-equilibration path forced to the full cycle budget (#532). A drop
/// guard restores the prior value, so a panic or a nested call cannot leak the flag.
#[cfg(test)]
pub fn with_full_ss_equilibration<R>(f: impl FnOnce() -> R) -> R {
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
pub fn record_ss_equilibration_cycles(_n: usize) {}

/// Non-test counterpart (same signature, so call sites are unconditional).
#[cfg(not(test))]
#[inline(always)]
pub fn record_ss_equilibration_branch(_b: SsBranch) {}

/// Non-test counterpart (same signature, so call sites are unconditional).
#[cfg(not(test))]
#[inline(always)]
pub fn record_ss_nonconvergence_warning(_warning: Option<&str>) {}

/// Relative threshold above which a **cycle-capped** SS equilibration warns (#867). A heavily
/// accumulating nonlinear disposition drives `ρ ≈ exp(−λ·II) → 1`, so the capped loop stops
/// short and the trough is too low; the tail `incr·ρ/(1−ρ)` is compared against this fraction
/// of the trough (and `ρ ≥ 1` means no periodic SS exists).
///
/// Also the **noise floor** in [`ss_equilibration_tail_warning`] below which the last step
/// counts as converged (solver jitter at `reltol` can run out the budget without reaching
/// `1e-12`). `1e-2`: accurate cases sit at ~0% tail, the #867 cases are 38–79% low, and 1% is
/// ~100× the default `reltol = 1e-4`. Linear dispositions take the exact path and never get here.
pub const SS_WARN_REL_TOL: f64 = 1e-2;

/// Per-run collector for SS non-convergence warnings (#867, #2067).
///
/// The capped equilibration runs deep in the rayon-parallel prediction walk, which has no
/// warning channel, so the (model-structural) message is deduplicated here, one `BTreeSet` per
/// run ([`run_scope`]), and drained via [`take_ss_nonconvergence_warnings`] by every `api`
/// boundary with a `warnings` channel (`fit`, `predict_diag`, the `simulate*` `_diag` family,
/// the adaptive driver; #1652). Writes outside any scope land in bucket `0`.
/// [`clear_ss_nonconvergence_warnings`] on entry and the take on exit bracket the run's set, so
/// code running inside an entry point must not call a draining one (hence
/// `api::predict_diag_undrained` in `suggest_start::sweep`). Bare `predict()` / `simulate()`
/// wrap their `_diag` forms, so they drain too and discard the note. Writers that do not drain
/// (e.g. `pk::compute_predictions_with_states`) leave their note for the next draining entry
/// point of the same run, whose clear discards it.
type SsSink = std::sync::Mutex<std::collections::HashMap<u64, std::collections::BTreeSet<String>>>;

fn ss_nonconvergence_sink() -> &'static SsSink {
    static SINK: std::sync::OnceLock<SsSink> = std::sync::OnceLock::new();
    SINK.get_or_init(std::sync::Mutex::default)
}

/// The sink bucket of the run the calling thread belongs to.
fn ss_run_key() -> u64 {
    use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
    kj::run_of(kj::current_fit())
}

/// Hold the run scope of a top-level draining operation: a thread outside any fit takes a fresh
/// run id for the operation's duration (carried by its pool tasks), a thread inside one joins it.
/// Dropping it discards what the run left in the sink.
pub struct SsRunScope(
    Option<(u64, crate::parser::model_parser::rhs_regcode::kernel_jit::FitGuard)>,
);

pub fn run_scope() -> SsRunScope {
    SsRunScope(crate::parser::model_parser::rhs_regcode::kernel_jit::open_run())
}

impl Drop for SsRunScope {
    fn drop(&mut self) {
        if let Some((id, _)) = &self.0 {
            drop_ss_run(*id);
        }
    }
}

/// Forget run `run`'s bucket (its scope ended).
pub fn drop_ss_run(run: u64) {
    if let Ok(mut m) = ss_nonconvergence_sink().lock() {
        m.remove(&run);
    }
}

/// Estimate the un-taken tail of a cycle-capped SS equilibration and, if non-negligible, note a
/// warning in the [`ss_nonconvergence_sink`] (#867). Inputs from
/// [`SsStopTracker::recent_increments`].
pub fn note_ss_nonconvergence_if_capped(
    early_stopped: bool, abs_prev: f64, abs_last: f64, mag: f64,
) {
    let warning = ss_equilibration_tail_warning(early_stopped, abs_prev, abs_last, mag);
    // Recorded on every call so a converged run clears an earlier warning (#1289).
    record_ss_nonconvergence_warning(warning.as_deref());
    if let Some(msg) = warning {
        if let Ok(mut m) = ss_nonconvergence_sink().lock() {
            m.entry(ss_run_key()).or_default().insert(msg);
        }
    }
}

/// Build the #867 non-convergence warning from the last two **absolute** cycle increments and the
/// final trough magnitude, or `None` when the equilibration is trustworthy.
///
/// `ρ = abs_last / abs_prev`; `rel_last = abs_last / mag`; the remaining error is the geometric
/// tail `rel_last·ρ/(1−ρ)`. `ρ ≥ 1` (not contracting, no periodic SS) warns only past the
/// `rel_last ≤ SS_WARN_REL_TOL` noise floor, where solver jitter makes `ρ` meaningless. The
/// floor is not applied to the `ρ < 1` branch: a near-1 ratio has a small step but a large tail.
fn ss_equilibration_tail_warning(
    early_stopped: bool, abs_prev: f64, abs_last: f64, mag: f64,
) -> Option<String> {
    if early_stopped || abs_last <= 0.0 {
        return None;
    }
    // Non-finite or all-zero state: no scale (a blow-up surfaces elsewhere as a non-finite OFV).
    if !abs_last.is_finite() || !mag.is_finite() || mag <= 0.0 {
        return None;
    }
    let rel_last = abs_last / mag;
    // No prior increment: too little signal to estimate ρ.
    if abs_prev <= 0.0 {
        return None;
    }
    let rho = abs_last / abs_prev;
    if !rho.is_finite() {
        return None;
    }
    if rho >= 1.0 {
        // Not contracting → no periodic SS, unless the step is solver noise (noise floor; this
        // branch only, see the doc comment).
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
    // Remaining tail relative to the current (under-converged) trough.
    let remaining_vs_current = rel_last * rho / (1.0 - rho);
    if remaining_vs_current > SS_WARN_REL_TOL {
        // Reported against the true SS `u_last·(1 + tail)` (#867: 0.61/1.61 ≈ 38%).
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

/// Drain and return the collected SS non-convergence warnings (#867). The `api` boundary calls
/// this after its prediction pass and appends the result to `FitResult.warnings`. See
/// [`ss_nonconvergence_sink`].
pub fn take_ss_nonconvergence_warnings() -> Vec<String> {
    #[cfg(test)]
    SS_SINK_DRAINS.with(|c| c.set(c.get() + 1));
    ss_nonconvergence_sink()
        .lock()
        .ok()
        .and_then(|mut m| m.remove(&ss_run_key()))
        .map(|set| set.into_iter().collect())
        .unwrap_or_default()
}

/// Clear any residual SS non-convergence warnings so a top-level `api` call starts from a clean
/// sink (#867). See [`ss_nonconvergence_sink`].
pub fn clear_ss_nonconvergence_warnings() {
    #[cfg(test)]
    SS_SINK_DRAINS.with(|c| c.set(c.get() + 1));
    drop_ss_run(ss_run_key());
}

#[cfg(test)]
mod ss_warn_tests {
    use super::*;

    // Args are `(early_stopped, abs_prev, abs_last, mag)`: absolute L∞ increments of the last two
    // non-converged cycles plus the final trough magnitude. `mag = 1.0` makes `abs == rel`.

    /// The per-thread observation the lib-test readers assert on (#2071) must hold exactly the
    /// text that goes into the sink, and a converged call must overwrite it with `None`.
    #[test]
    fn note_records_its_own_warning_text_per_thread() {
        // Not contracting (`ρ = 2`) and past the noise floor (last step 20% of the trough).
        let want = ss_equilibration_tail_warning(false, 1.0, 2.0, 10.0).expect("fixture warns");
        note_ss_nonconvergence_if_capped(false, 1.0, 2.0, 10.0);
        assert_eq!(last_ss_equilibration_warning().as_deref(), Some(want.as_str()));
        assert!(last_ss_equilibration_warned());
        note_ss_nonconvergence_if_capped(true, 1.0, 2.0, 10.0);
        assert_eq!(last_ss_equilibration_warning(), None);
        assert!(!last_ss_equilibration_warned());
    }

    /// #2067: two runs open at once keep their notes apart, including a note written by a pool
    /// task that carries the run's id. Each run clears on entry and takes on exit, and sees only
    /// its own. Mutation: key every write and read by one constant -> both runs read both notes.
    #[test]
    fn concurrent_runs_keep_their_ss_notes_apart() {
        use crate::parser::model_parser::rhs_regcode::kernel_jit::{current_fit, in_fit};
        let gate = std::sync::Barrier::new(2);
        let run = |mag: f64| {
            let _run = run_scope();
            clear_ss_nonconvergence_warnings();
            let fit = current_fit();
            assert_ne!(fit, 0, "a top-level run opens a scope");
            // The write comes from another thread running the run's id, as a pool task does.
            std::thread::scope(|s| {
                s.spawn(|| in_fit(fit, || note_ss_nonconvergence_if_capped(false, 1.0, 2.0, mag)))
                    .join()
                    .unwrap();
            });
            gate.wait(); // both runs have written before either reads
            (fit, take_ss_nonconvergence_warnings())
        };
        let ((fa, a), (fb, b)) = std::thread::scope(|s| {
            let ha = s.spawn(|| run(10.0));
            let hb = s.spawn(|| run(40.0));
            (ha.join().unwrap(), hb.join().unwrap())
        });
        let want = |mag: f64| {
            vec![ss_equilibration_tail_warning(false, 1.0, 2.0, mag).expect("fixture warns")]
        };
        assert_eq!(a, want(10.0));
        assert_eq!(b, want(40.0));
        // A run that ends leaves nothing behind: its bucket is gone.
        let left = {
            let run = run_scope();
            let fit = current_fit();
            note_ss_nonconvergence_if_capped(false, 1.0, 2.0, 10.0);
            assert!(ss_nonconvergence_sink().lock().unwrap().contains_key(&fit));
            drop(run);
            fit
        };
        let m = ss_nonconvergence_sink().lock().unwrap();
        assert!(![fa, fb, left].iter().any(|k| m.contains_key(k)));
    }

    /// Every clear and every take counts as a drain on the calling thread (#2071).
    #[test]
    fn clear_and_take_each_count_as_a_drain() {
        let before = ss_sink_drains_on_this_thread();
        clear_ss_nonconvergence_warnings();
        assert_eq!(ss_sink_drains_on_this_thread(), before + 1);
        let _ = take_ss_nonconvergence_warnings();
        assert_eq!(ss_sink_drains_on_this_thread(), before + 2);
    }

    /// Inputs that must not warn: early stop; negligible tail (ρ = 0.1, 5% step → 0.6%);
    /// solver-noise jitter at ~1e-4 with ρ ≥ 1 and ρ < 1 (#867 review); zero or missing
    /// increments; non-finite or zero magnitude.
    #[test]
    fn trustworthy_equilibrations_do_not_warn() {
        for (early, prev, last, mag) in [
            (true, 0.5, 0.4, 1.0),
            (false, 0.5, 0.05, 1.0),
            (false, 1.0e-4, 1.1e-4, 1.0),
            (false, 1.2e-4, 1.0e-4, 1.0),
            (false, 0.0, 0.0, 1.0),
            (false, 0.1, 0.0, 1.0),
            (false, 0.0, 0.05, 1.0),
            (false, 1.0, f64::INFINITY, 1.0),
            (false, 1.0, 0.05, f64::INFINITY),
            (false, 1.0, 0.05, f64::NAN),
            (false, 0.1, 0.05, 0.0),
        ] {
            assert!(
                ss_equilibration_tail_warning(early, prev, last, mag).is_none(),
                "{early} {prev} {last} {mag}"
            );
        }
    }

    /// Inputs that must warn, with the expected text: slow contraction (ρ = 0.9 → ~45% tail);
    /// a small step below the noise floor with ρ = 0.98 (~25% tail: the floor must not apply to
    /// the contracting branch, #874); growing and constant absolute steps (no periodic SS).
    #[test]
    fn untrustworthy_equilibrations_warn() {
        let below = "below the true periodic steady state";
        for (prev, last, want) in [
            (0.05 / 0.9, 0.05, below),
            (0.005 / 0.98, 0.005, below),
            (0.02, 0.03, "not contracting"),
            (0.02, 0.03, "no periodic steady state"),
            (0.02, 0.02, "no periodic steady state"),
        ] {
            let w = ss_equilibration_tail_warning(false, prev, last, 1.0).expect("should warn");
            assert!(w.contains(want), "{w}");
        }
    }

    #[test]
    fn increment_helpers() {
        assert_eq!(ss_max_magnitude(&[0.0, 0.0]), 0.0);
        assert_eq!(ss_max_magnitude(&[-3.0, 1.0]), 3.0);
        assert_eq!(ss_abs_increment(&[1.0, 2.0], &[1.0, 0.0]), 2.0);
        assert_eq!(ss_abs_increment(&[5.0], &[5.0]), 0.0);
    }
}

/// Whether `d` is a usable infusion. `is_infusion()` only checks `rate > 0`; a degenerate row
/// with `duration <= 0` or NaN would break the break-time sort, so it falls back to a bolus (a
/// zero/negative bolus update — visible, not silently dropped).
pub fn is_real_infusion(d: &DoseEvent) -> bool {
    // Tripwire (#324): modeled-RATE doses must be resolved before any infusion logic.
    debug_assert!(d.is_fixed(), "is_real_infusion: unresolved modeled dose");
    d.is_infusion() && d.duration > 0.0 && d.duration.is_finite()
}

/// Whether a zero-order input into this dose's compartment has a rate channel at all,
/// spelled once for every engine (#1077).
///
/// `CMT=0` is NONMEM's default dose compartment for a bolus (#899) but undefined for a
/// zero-order input. Validation rejects such a row (`E_DOSE_CMT_NOT_INFUSABLE`), so this only
/// matters for unvalidated specs, where every engine must agree: a rate-off site that skipped
/// this test moved `∂f/∂η_LAG` by `+1.8878965164` where FD reads `0.0`.
///
/// Not folded into [`is_real_infusion`]: break-time builders still break at a `CMT=0`
/// infusion's window end; only the forcing sites ask this.
#[inline]
pub fn infusion_has_rate_channel(d: &DoseEvent) -> bool {
    d.cmt_raw() != 0
}

// ---------------------------------------------------------------------------
// Where a lagged steady-state dose loads, and what the previous cycle leaves
// running across the dose record (#1121).
// ---------------------------------------------------------------------------

/// Whether this steady-state dose's periodic state is loaded at its dose
/// **record** and flowed forward to the lagged arrival (#1121), rather than
/// equilibrated at the arrival itself.
///
/// NONMEM loads SS compartments at the dose row and ADVANs to the lagged arrival under the
/// record terminating that interval; this only differs from equilibrating at the arrival when
/// a covariate changes inside the pre-arrival window. `lag == 0` is excluded: record and
/// arrival coincide, and a full extra cycle would only add solver error.
///
/// Every call site (including the analytic twin's `K_SS_SEED` break) reads this predicate and
/// the arrival side its complement [`ss_equilibrates_at_arrival`], so coverage can neither
/// overlap (double load, #1275) nor leave a gap.
pub fn ss_seeded_at_record(dose: &DoseEvent, lag: f64) -> bool {
    ss_seeded_at_record_g(dose, lag)
}

/// [`ss_seeded_at_record`] for a lag of any type: a recording type (#2786) decides it
/// through its branch API.
pub fn ss_seeded_at_record_g<G: crate::sens::num::PkNum>(dose: &DoseEvent, lag: G) -> bool {
    dose.ss && dose.ii > 0.0 && lag.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0)
}

/// Whether this steady-state dose's periodic state is equilibrated at its
/// **arrival** — the complement of [`ss_seeded_at_record`] within the `SS` doses.
///
/// A lagged SS dose's state is loaded once, at its record; its arrival adds only the pulse
/// (#1275). Re-equilibrating at the arrival would also erase any dose landing inside
/// `(record, arrival)`: up to **−54 %** against NONMEM (`nonmem_anchor/ss_arrival_flow_*`,
/// `ss_reset_tie_*`). Every value-path walk reads this one spelling.
pub fn ss_equilibrates_at_arrival(dose: &DoseEvent, lag: f64) -> bool {
    ss_equilibrates_at_arrival_g(dose, lag)
}

/// [`ss_equilibrates_at_arrival`] for a lag of any type (#2786).
pub fn ss_equilibrates_at_arrival_g<G: crate::sens::num::PkNum>(dose: &DoseEvent, lag: G) -> bool {
    dose.ss && dose.ii > 0.0 && !ss_seeded_at_record_g(dose, lag)
}

/// Cycle phase the dose **record** sits at, for a steady-state dose seeded there.
///
/// The pulse preceding the record landed at `dose.time − ss_seed_phase(…)`, so a
/// lagtime shorter than one interval puts it at `dose.time + lag − II` and the
/// record reads the previous cycle's tail at phase `II − lag`.
///
/// `lag ≥ II` **clamps** to phase 0, as NONMEM 7.6.0 does
/// (`nonmem_anchor/results/ss_lag_ge_ii.tab`: `ALAG1 = 15`, `II = 12` reads the SS peak
/// `PRED(0) = 13.197`; wrapping would give `2.1815`). Clamping is also continuous in an
/// estimated `ALAG` across `lag = II`, where wrapping would jump by `e^{−k·II}`.
pub fn ss_seed_phase(dose: &DoseEvent, lag: f64) -> f64 {
    (dose.ii - lag).max(0.0)
}

/// [`ss_seed_phase`] for a lag of any type (#2786): `II − lag` where positive, else `0`
/// (a `NaN` phase floors, as `f64::max` does).
pub fn ss_seed_phase_g<G: crate::sens::num::PkNum>(dose: &DoseEvent, lag: G) -> G {
    let p = G::from_f64(dose.ii) - lag;
    if p.br_cmp_c(crate::sens::num::BranchCmp::Gt, 0.0) {
        p
    } else {
        G::from_f64(0.0)
    }
}

/// How a steady-state infusion of length `t_inf` tiles its own interval: `m` earlier pulses
/// run through the whole cycle and the newest for `r = t_inf − m·II ∈ (0, II]`, so `m + 1`
/// pulses infuse on `[0, r)` and `m` on `[r, II)`. `m = 0` and `r = t_inf` (bit for bit) when
/// the pulses do not overlap (`t_inf ≤ II`). Every engine's one-cycle forcing, value and
/// recorded, reads this split, so an overlapping `SS=1` infusion (`T_inf > II`) is one more
/// constant rate on the cycle rather than a case of its own. Each comparison deciding `m` is a
/// recorded branch.
pub fn ss_overlap_split_g<G: crate::sens::num::PkNum>(t_inf: G, ii: f64) -> (usize, G) {
    // `m = #{k ≥ 1 : t_inf > k·II}`: the residual count of `t_inf`, less its `k = 0` term.
    let m = count_gt_multiples(t_inf, ii).saturating_sub(1);
    (m, t_inf - G::from_f64(m as f64 * ii))
}

/// The earlier steady-state pulses of an `SS=1` infusion that are **still running** when its
/// periodic state is installed — at the record for a seeded dose (`lag > 0`), else at the
/// arrival — as `(first, n)`: they stop at `dose.time + first − j·II`, `j = 0 … n − 1`.
///
/// The state installed is the cycle's phase `ψ` (`ss_seed_phase` when seeded, `II` — the
/// trough before the dose's own pulse — otherwise), so pulse `j` arrived `ψ + j·II` before the
/// record and still infuses while `ψ + j·II < T_inf`. A walk must carry `+rate` for each of them
/// after the record: the installed state has them mid-flight. Unseeded, that is the overlapping
/// case alone (`T_inf > II`); seeded, it is also the previous cycle crossing the record whenever
/// `lag > II − T_inf` (#1121, confirmed against NONMEM 7.6.0 in
/// `nonmem_anchor/results/ss_lag_infusion.tab`: `II = 12`, `T_inf = 6`, `ALAG1 = 8` rises from
/// `6.5468` at the record to `6.8754` at `t = 0.5` and turns over at `t = 2 = 8 − 12 + 6`).
/// `t_inf` is the dose's `F`-reshaped window (#419), the one its own window uses.
///
/// # `lag ≥ II` with an infusion is ferx's answer, not NONMEM's
///
/// Under the [`ss_seed_phase`] clamp the phase is 0: the previous cycle's infusion starts on
/// the record, which is what the clamped phase means and is mass-balanced. NONMEM instead
/// delivers `T_inf + (lag − II)` hours of drug for that geometry (measured: an `AMT = 600`,
/// `RATE = 100`, `II = 12`, `ALAG1 = 14` dose infuses over `[0, 8]`, i.e. 800 mg), which is
/// not a physical reading of a 600 mg dose. ferx does not reproduce it; see
/// `docs/model-file/lagtime.qmd`.
pub fn ss_residual_infusions_g<G: crate::sens::num::PkNum>(
    dose: &DoseEvent, lag: G, t_inf: G,
) -> (G, usize) {
    let psi = if ss_seeded_at_record_g(dose, lag) {
        ss_seed_phase_g(dose, lag)
    } else {
        G::from_f64(dose.ii)
    };
    let first = t_inf - psi;
    let n = if dose.ss && dose.ii > 0.0 { count_gt_multiples(first, dose.ii) } else { 0 };
    (first, n)
}

/// `#{j ≥ 0 : x > j·ii}` for `ii > 0` — what `while x > n·ii { n += 1 }` counts — in O(1)
/// (a modeled infusion rate driven toward 0 makes `x ~ 1e11` and the loop `~1e10` long). The
/// count is estimated from the value, then pinned by the two boundary comparisons the loop
/// made at its end (`x > (n−1)·ii`, `¬ x > n·ii`), recorded as branches like the loop's.
/// Saturates at `usize::MAX` for an `x` no finite count reaches.
pub fn count_gt_multiples<G: crate::sens::num::PkNum>(x: G, ii: f64) -> usize {
    use crate::sens::num::BranchCmp::Gt;
    debug_assert!(ii > 0.0, "count_gt_multiples needs an interval, got {ii}");
    if !x.br_cmp_c(Gt, 0.0) {
        return 0;
    }
    // value-read-ok: an estimate; the two comparisons recorded below pin `n`, the value is not a decision
    let v = x.probe_value();
    let mut n = if v.is_finite() { (v / ii).ceil() as usize } else { usize::MAX };
    n = n.max(1);
    // The loop's own arithmetic `n as f64 * ii`; correct the estimate by rounding.
    if n < (1usize << 52) {
        while n > 1 && v <= (n - 1) as f64 * ii {
            n -= 1;
        }
        while v > n as f64 * ii {
            n += 1;
        }
    }
    if n >= 2 {
        x.br_cmp_c(Gt, (n - 1) as f64 * ii);
    }
    x.br_cmp_c(Gt, n as f64 * ii);
    n
}

/// The stop times `t + first − j·ii`, `j = 0 … n − 1` (descending) of the residual pulses of
/// one dose, searched by index so a caller asks for the ones in a time range without
/// materialising the `n` of them.
#[derive(Clone, Copy, Debug)]
pub struct ResidualEnds {
    t: f64,
    first: f64,
    ii: f64,
    n: usize,
}

impl ResidualEnds {
    /// The ends of `n` pulses stopping at `t + first − j·ii`.
    pub fn new(t: f64, first: f64, ii: f64, n: usize) -> Self {
        Self { t, first, ii, n }
    }
    /// Pulse `j`'s stop time.
    #[inline]
    pub fn end(&self, j: usize) -> f64 {
        self.t + self.first - j as f64 * self.ii
    }
    /// Smallest `j ≤ n` whose end satisfies `pred` (`pred` monotone: false then true).
    fn first_where(&self, pred: impl Fn(f64) -> bool) -> usize {
        let (mut lo, mut hi) = (0usize, self.n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.end(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
    /// Number of pulses still running at `lo`: those with end `≥ lo`.
    pub fn count_ending_at_or_after(&self, lo: f64) -> usize {
        self.first_where(|e| e < lo)
    }
    /// Number of pulses whose end lies after `hi` — never reached by a walk that stops at
    /// `hi`, yet running (their rate counts) throughout it. They are the first `j`s.
    pub fn count_ending_after(&self, hi: f64) -> usize {
        self.first_where(|e| e <= hi)
    }
    /// `(j, end)` for the ends in `[lo, hi]`, descending.
    pub fn within(&self, lo: f64, hi: f64) -> impl Iterator<Item = (usize, f64)> {
        let me = *self;
        (me.count_ending_after(hi)..me.count_ending_at_or_after(lo)).map(move |j| (j, me.end(j)))
    }
}

/// The pulse indices `j0..n` of [`ss_residual_infusions_g`]'s `(first, n)` whose stop time
/// `t + first − j·ii` is at or before `horizon` — the ones a walk ending there can reach. The
/// split is read from the value; a recording type guards both sides of it (`end(j0−1) >
/// horizon`, `end(j0) ≤ horizon`) so the kept set is what a replay at other parameters keeps.
pub fn residual_end_indices_g<G: crate::sens::num::PkNum>(
    t: f64, first: G, ii: f64, n: usize, horizon: f64,
) -> std::ops::Range<usize> {
    use crate::sens::num::BranchCmp::{Gt, Le};
    // value-read-ok: an estimate of the split; the two comparisons recorded below pin it
    let j0 = ResidualEnds::new(t, first.probe_value(), ii, n).count_ending_after(horizon);
    let end = |j: usize| G::from_f64(t) + first - G::from_f64(j as f64 * ii);
    if j0 > 0 {
        end(j0 - 1).br_cmp_c(Gt, horizon);
    }
    if j0 < n {
        end(j0).br_cmp_c(Le, horizon);
    }
    j0..n
}

/// The number of [`ss_residual_ends`]'s pulses of `dose` still running at `lo` (end `≥ lo`),
/// estimated from values and pinned by two recorded comparisons (as
/// [`residual_end_indices_g`]).
pub fn ss_residual_count_ending_at_or_after_g<G: crate::sens::num::PkNum>(
    dose: &DoseEvent, lag: f64, dur_eff: G, lo: G,
) -> usize {
    use crate::sens::num::BranchCmp::{Ge, Lt};
    if !is_real_infusion(dose) {
        return 0;
    }
    let (first, n) = ss_residual_infusions_g(dose, G::from_f64(lag), dur_eff);
    if n == 0 {
        return 0;
    }
    // An estimate of the count; the two comparisons recorded below pin it.
    // value-read-ok: estimate only, pinned by the recorded comparisons below
    let (first_v, lo_v) = (first.probe_value(), lo.probe_value());
    let j0 = ResidualEnds::new(dose.time, first_v, dose.ii, n).count_ending_at_or_after(lo_v);
    let end = |j: usize| G::from_f64(dose.time) + first - G::from_f64(j as f64 * dose.ii);
    if j0 > 0 {
        end(j0 - 1).br_cmp(Ge, lo);
    }
    if j0 < n {
        end(j0).br_cmp(Lt, lo);
    }
    j0
}

/// [`ss_residual_infusions_g`]'s stop times, for an `F`-reshaped real infusion (else none).
pub fn ss_residual_ends(dose: &DoseEvent, lag: f64, f_bio: f64) -> ResidualEnds {
    let (first, n) = if is_real_infusion(dose) {
        ss_residual_infusions_g(dose, lag, dose.bioavailable_infusion(f_bio).1)
    } else {
        (0.0, 0)
    };
    ResidualEnds { t: dose.time, first, ii: dose.ii, n }
}

/// [`ss_residual_ends`]'s stop times lying in `[lo, hi]` — the span a caller integrates over;
/// an end outside it is never reached (the pulse still counts in the rate, through `n`).
pub fn ss_residual_infusion_ends(
    dose: &DoseEvent, lag: f64, f_bio: f64, lo: f64, hi: f64,
) -> impl Iterator<Item = f64> {
    ss_residual_ends(dose, lag, f_bio).within(lo, hi).map(|(_, e)| e)
}

/// Whether flowing a seeded steady-state dose from its record to its lagged
/// arrival lands the dose's **own** contribution back on the periodic trough — so
/// a path that writes the trough at the arrival instead of propagating there gets
/// the same number for that dose.
///
/// The flowed phase is `(II − lag) + lag = II`, the trough, for `lag ≤ II`; past that the
/// clamp lands the arrival further down the decay, and taking the trough would read **+4.3 %**
/// high (`ALAG1 = 15`, `II = 12`, `t = 26`, `nonmem_anchor/results/ss_lag_ge_ii`).
///
/// Its one reader is the additive analytic superposition (`pk/mod.rs`). It is **not** a
/// licence for a state-vector walk to re-equilibrate at the arrival (#1275); those read
/// [`ss_equilibrates_at_arrival`].
pub fn ss_arrival_is_trough(dose: &DoseEvent, lag: f64) -> bool {
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
pub fn tad_referent(dose: &DoseEvent, lag: f64, t: f64) -> Option<f64> {
    tad_referent_g(dose, lag, t)
}

/// [`tad_referent`] for a lag and a time of any type (#2786): **the** implementation, which the
/// `f64` form above is the plain instance of, so the dual walk's `TAD` anchor (a recording or a
/// jet carrying `∂TAD/∂lag = −1`) cannot drift from the value's. A recording type decides every
/// comparison through its branch API and the wrap through [`wrap_into_period_g`].
#[inline]
pub fn tad_referent_g<G: crate::sens::num::PkNum>(dose: &DoseEvent, lag: G, t: G) -> Option<G> {
    use crate::sens::num::BranchCmp::Le;
    let arrival = G::from_f64(dose.time) + lag;
    if arrival.br_cmp(Le, t + G::from_f64(TAD_EPS)) {
        if dose.ss && dose.ii > 0.0 {
            return Some(t - wrap_into_period_g(t - arrival, dose.ii));
        }
        return Some(arrival);
    }
    if ss_seeded_at_record_g(dose, lag)
        && G::from_f64(dose.time).br_cmp(Le, t + G::from_f64(TAD_EPS))
    {
        return Some(G::from_f64(dose.time) - ss_seed_phase_g(dose, lag));
    }
    None
}

/// `x.rem_euclid(ii)` (`ii > 0`) for a value of any type: the value `x − m·ii` with `m =
/// ⌊x / ii⌋`, `∂/∂x = 1` and no derivative in `m` (a whole number the value fixes). A plain
/// number takes the remainder `f64::rem_euclid` gives, bit for bit (`x − x.value_only()` is the
/// zero that keeps `x`'s jet, so the value is the remainder itself); a recording type counts
/// the periods through [`PkNum::br_floor_div`], whose two comparisons it guards.
///
/// [`PkNum::br_floor_div`]: crate::sens::num::PkNum::br_floor_div
#[inline]
pub fn wrap_into_period_g<G: crate::sens::num::PkNum>(x: G, ii: f64) -> G {
    match x.nonrec_val() {
        Some(v) => (x - x.value_only()) + G::from_f64(v.rem_euclid(ii)),
        None => x - G::from_f64(x.br_floor_div(ii) * ii),
    }
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
pub fn tad_at(doses: &[DoseEvent], dose_lagtimes: &[f64], t: f64) -> f64 {
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

/// The `SS=1` record that resets the system at `t`, as an index into `doses` (#1576).
///
/// An `SS=1` record resets the system at its record, like NONMEM: every dose that
/// precedes it in **(time, row order)** contributes nothing at or after it, and its own
/// implied pulse train contributes nothing before it. This returns the `SS=1` dose
/// (`ss && ii > 0`) with the largest `(time, index)` among those with `time ≤ t`, or
/// `None` when no such record has been reached. Dose `k` is live at `t` iff
/// [`ss_reset_live`] holds.
///
/// Row order is load-bearing, and NONMEM-measured: a bolus row *before* a co-timed
/// `SS=1` row is wiped, one *after* it superposes. `subject.doses` is stable-sorted by
/// time, so a tie keeps file order and the index is the row order. The comparison is
/// lexicographic, so the answer is also correct on an unsorted list (the adaptive shadow
/// list is base-then-controller).
///
/// Both sides of the comparison are dose **records**, never lagged arrivals, and every dose
/// kind is covered. So a caller that skips non-live doses outright — analytic superposition
/// (`crate::pk::predict_concentration`, `analytical_state_at_times`, and the dual's two
/// loops) — also cancels a lagged dose still pending at the record and stops an infusion
/// still running at it, as NONMEM does. The ODE absorption forcing skips non-live doses
/// too, so it stops an infusion into the forcing compartment; it only ever sees record =
/// arrival (`SS=1` + lag into a forcing is rejected up front). The state-propagating
/// engines (the ODE walkers, the analytic event-driven walk, and their dual twins) read it
/// through `ResetGate` at every arrival (#1588) and for every infusion and zero-order
/// window's membership (#1586). O(n), no allocation.
#[inline]
pub fn ss_reset_cutoff(doses: &[DoseEvent], t: f64) -> Option<usize> {
    ss_reset_cutoff_g(doses, t)
}

/// [`ss_reset_cutoff`] at a time of any type: a recording type (#2786) decides `record ≤ t`
/// through its branch API.
pub fn ss_reset_cutoff_g<G: crate::sens::num::PkNum>(doses: &[DoseEvent], t: G) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, d) in doses.iter().enumerate() {
        if !(d.ss && d.ii > 0.0) || G::from_f64(d.time).br_cmp(crate::sens::num::BranchCmp::Gt, t) {
            continue;
        }
        best = match best {
            Some(b) if (doses[b].time, b) > (d.time, i) => Some(b),
            _ => Some(i),
        };
    }
    best
}

/// Whether dose `k` is live under the reset `cutoff` from [`ss_reset_cutoff`]:
/// `(t_k, k) ≥ (t_s, s)` lexicographically, or no cutoff at all.
#[inline]
pub fn ss_reset_live(doses: &[DoseEvent], cutoff: Option<usize>, k: usize) -> bool {
    match cutoff {
        None => true,
        Some(s) => (doses[k].time, k) >= (doses[s].time, s),
    }
}

/// Whether a dose recorded at `record` survives the most recent EVID=3/4 reset
/// `reset_floor` (`NEG_INFINITY` when none has been reached) (#1587).
///
/// NONMEM resets the system at the reset **record** and cancels every dose recorded before
/// it — including a lagged dose whose arrival, or a lagged infusion whose window, lies after
/// the reset. So the test is on the record, never on `record + lag`. A dose recorded at the
/// reset itself is live: an EVID=4 row's own dose, or a dose row following a co-timed
/// EVID=3 row. (A dose row *preceding* a co-timed reset row reaches the engines with
/// `record < reset`: the reader shifts such a reset past it, `RESET_SEGMENT_GAP`.)
#[inline]
pub fn evid_reset_live(record: f64, reset_floor: f64) -> bool {
    record >= reset_floor
}

/// A dose list indexed once, so the closed-form event walks ask it instead of rescanning
/// every dose per interval, sub-interval, arrival and observation (#2754).
///
/// Before this index each of those queries was an `O(n_doses)` scan, so one walk over a
/// subject with `n` dose records cost `O(n²)` — on a 200-dose history the bookkeeping, not
/// the PK, was most of a fit. Every query here returns a **superset** of the doses the old
/// scan acted on, in ascending dose order, and the callers run the unchanged per-dose code
/// over that subset. A skipped dose is one the old loop provably did nothing for (no rate,
/// no break, no boundary within the match tolerance), so the walks take the same branches
/// and do the same arithmetic in the same order: values are bit-identical.
///
/// Built from one `(doses, dose_lagtimes)` pair and valid only for it. Times, `ss`/`ii`
/// and lagtimes are subject- or schedule-static; `rate`/`duration` are not (bioavailability
/// reshapes a window, a modeled dose resolves per η), which is why
/// [`crate::pk::event_driven::EventSchedule`] reuses its own index only for doses that pass
/// `rates_valid_for`, and a walk builds a fresh one otherwise.
#[derive(Debug, Clone, Default)]
pub struct DoseIndex {
    /// Number of doses indexed — a guard that the caller queries with the same list.
    n_doses: usize,
    /// The `SS=1` records (`ss && ii > 0`) as `(time, k)`, sorted lexicographically
    /// under `f64::partial_cmp` (so `-0.0` ties `0.0`, as in [`ss_reset_cutoff`]).
    ss_records: Vec<(f64, usize)>,
    /// Some `SS=1` record has a `NaN` time: [`Self::ss_cutoff`] then defers to the scan,
    /// whose answer for a `NaN` comparison is not a sort order.
    ss_scan_fallback: bool,
    /// Every dose that can carry an input rate (`duration > 0`), ascending. Both halves of
    /// the rate test — a running window and the #1121 residual of a seeded `SS=1` infusion
    /// (which requires `is_real_infusion`, so `duration > 0`) — need it.
    infusing: Vec<usize>,
    /// Every infusion break `compute_propagation_bounds` can push — window start, window
    /// end and seeded-`SS=1` residual end of each `rate > 0 && duration > 0` dose — sorted
    /// ascending, `NaN` dropped (it never passes the open-interval test).
    breaks: Vec<f64>,
    /// The residual pulses' stop times of each seeded `SS=1` infusion with any (#1121, and every
    /// overlapping pulse when `T_inf > II`), as ranges searched by index: materialising them
    /// would be one entry per pulse, and a modeled rate driven toward 0 makes that `~1e10`.
    residuals: Vec<(usize, ResidualEnds)>,
    /// `(value, k)` of every dose's arrival `t_k + lag_k`, and of every infusing dose's
    /// window end `t_k + lag_k + D_k`, sorted by value; non-finite values dropped (they
    /// never lie within a finite tolerance of anything).
    moving_points: Vec<(f64, usize)>,
}

impl DoseIndex {
    /// Index `doses` under per-dose `dose_lagtimes` (parallel to `doses`; a missing entry
    /// is a zero lag, as every walk reads it). `O(n log n)`.
    pub fn new(doses: &[DoseEvent], dose_lagtimes: &[f64]) -> Self {
        let lag_of = |k: usize| dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let mut ss_records = Vec::new();
        let mut infusing = Vec::new();
        let mut breaks = Vec::new();
        let mut residuals = Vec::new();
        let mut moving_points = Vec::with_capacity(doses.len());
        for (k, d) in doses.iter().enumerate() {
            if d.ss && d.ii > 0.0 {
                ss_records.push((d.time, k));
            }
            let lag = lag_of(k);
            // The exact expressions the walks compute, so a point here is bit-identical
            // to the value the per-dose code tests.
            let t_start = d.time + lag;
            if t_start.is_finite() {
                moving_points.push((t_start, k));
            }
            if d.duration > 0.0 {
                infusing.push(k);
                let t_end = t_start + d.duration;
                if t_end.is_finite() {
                    moving_points.push((t_end, k));
                }
            }
            if d.rate > 0.0 && d.duration > 0.0 {
                let end = d.time + lag + d.duration;
                breaks.extend([t_start, end].into_iter().filter(|b| !b.is_nan()));
                // Every earlier pulse still infusing at the record, not only the last.
                let ends = ss_residual_ends(d, lag, 1.0);
                if ends.n > 0 {
                    residuals.push((k, ends));
                }
            }
        }
        let ss_scan_fallback = ss_records.iter().any(|(t, _)| t.is_nan());
        if !ss_scan_fallback {
            // Stable, so a time tie keeps ascending `k` — the lexicographic `(time, k)`.
            ss_records.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN"));
        }
        breaks.sort_by(|a, b| a.total_cmp(b));
        moving_points.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self {
            n_doses: doses.len(),
            ss_records,
            ss_scan_fallback,
            infusing,
            breaks,
            residuals,
            moving_points,
        }
    }

    /// [`ss_reset_cutoff`]`(doses, t)`, by binary search. `doses` must be the list this
    /// index was built from (it is only read on the scan fallback).
    #[inline]
    pub fn ss_cutoff(&self, doses: &[DoseEvent], t: f64) -> Option<usize> {
        debug_assert_eq!(doses.len(), self.n_doses, "DoseIndex queried with another list");
        if self.ss_scan_fallback || t.is_nan() {
            return ss_reset_cutoff(doses, t);
        }
        // The last record with `time ≤ t` carries the largest `(time, k)` among them.
        let n = self.ss_records.partition_point(|&(time, _)| time <= t);
        n.checked_sub(1).map(|i| self.ss_records[i].1)
    }

    /// The doses that can contribute an input rate anywhere (`duration > 0`), ascending.
    #[inline]
    pub fn infusing(&self) -> &[usize] {
        &self.infusing
    }

    /// The infusion breaks `b` with `lo < b < hi`, ascending — the candidates
    /// `compute_propagation_bounds` pushes for an interval, as a slice.
    #[inline]
    pub fn breaks_within(&self, lo: f64, hi: f64) -> std::borrow::Cow<'_, [f64]> {
        let from = self.breaks.partition_point(|&b| !(b > lo));
        let len = self.breaks[from..].partition_point(|&b| b < hi);
        let own = &self.breaks[from..from + len];
        if self.residuals.is_empty() {
            // The per-interval hot path: no residual pulses, so no allocation.
            return std::borrow::Cow::Borrowed(own);
        }
        let mut out = own.to_vec();
        for (_, ends) in &self.residuals {
            out.extend(ends.within(lo, hi).map(|(_, e)| e).filter(|&e| e > lo && e < hi));
        }
        out.sort_by(|a, b| a.total_cmp(b));
        std::borrow::Cow::Owned(out)
    }

    /// Every dose with an arrival or window end within a margin of `[lo, hi]`, ascending
    /// and deduplicated: a superset of the doses that have a moving boundary within the
    /// walks' `1e-9` match tolerance of any time in `[lo, hi]`. The margin is far wider
    /// than that tolerance, so rounding in the subtraction can never drop a match.
    pub fn near_moving_points(&self, lo: f64, hi: f64) -> Vec<usize> {
        let margin = 1e-6 * (1.0 + lo.abs().max(hi.abs()));
        let (lo, hi) = (lo - margin, hi + margin);
        let from = self.moving_points.partition_point(|&(v, _)| v < lo);
        let mut ks: Vec<usize> = self.moving_points[from..]
            .iter()
            .take_while(|&&(v, _)| v <= hi)
            .map(|&(_, k)| k)
            .collect();
        // A dose whose earlier pulse stops here is a candidate too (the residual ends are
        // not materialised in `moving_points`).
        for &(k, ends) in &self.residuals {
            if ends.within(lo, hi).next().is_some() {
                ks.push(k);
            }
        }
        ks.sort_unstable();
        ks.dedup();
        ks
    }
}

/// `DoseIndex` against the `O(n)` scans it replaces (#2754). Each query must return what
/// the scan acted on — exactly for the cutoff and the breaks, a superset for the
/// candidates — on dose lists built to hit every tie the scans resolve: co-timed `SS=1`
/// records (row order decides), `-0.0` against `0.0`, a dose list out of time order (the
/// adaptive shadow list), boluses mixed with infusions, lags, and the seeded-`SS=1`
/// residual window.
#[cfg(test)]
mod dose_index_tests {
    use super::{ss_reset_cutoff, ss_residual_infusion_ends, DoseIndex};
    use crate::types::DoseEvent;

    /// A deterministic xorshift, so a failure reproduces.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Times on a coarse grid so ties are common; `-0.0` stands in for some zeros.
    fn random_doses(rng: &mut Rng, n: usize) -> (Vec<DoseEvent>, Vec<f64>) {
        let mut doses = Vec::with_capacity(n);
        let mut lags = Vec::with_capacity(n);
        for _ in 0..n {
            let mut time = rng.below(12) as f64 * 0.5;
            if time == 0.0 && rng.below(2) == 0 {
                time = -0.0;
            }
            let rate = match rng.below(3) {
                0 => 0.0,
                1 => 50.0,
                _ => 200.0,
            };
            let ss = rng.below(4) == 0;
            let ii = if ss { [0.0, 6.0, 12.0][rng.below(3) as usize] } else { 0.0 };
            doses.push(DoseEvent::new(time, 100.0, 1, rate, ss, ii));
            lags.push([0.0, 0.0, 0.25, 1.5, 7.0][rng.below(5) as usize]);
        }
        (doses, lags)
    }

    /// The probe times: every dose time and arrival, each nudged by the match tolerances
    /// either way, plus points between and beyond the grid.
    fn probes(doses: &[DoseEvent], lags: &[f64]) -> Vec<f64> {
        let mut ts = vec![-1.0, -0.0, 0.0, 0.1, 2.75, 100.0, f64::INFINITY, f64::NEG_INFINITY];
        for (d, &lag) in doses.iter().zip(lags) {
            for base in [d.time, d.time + lag, d.time + lag + d.duration] {
                for eps in [0.0, 1e-15, -1e-15, 1e-12, -1e-12, 5e-10, -5e-10, 2e-9, -2e-9] {
                    ts.push(base + eps);
                }
            }
        }
        ts
    }

    #[test]
    fn the_ss_cutoff_is_the_scans_on_every_probe() {
        let mut rng = Rng(0x2754);
        for n in [0, 1, 2, 5, 17, 40] {
            for _ in 0..30 {
                let (doses, lags) = random_doses(&mut rng, n);
                let index = DoseIndex::new(&doses, &lags);
                for t in probes(&doses, &lags).into_iter().chain([f64::NAN]) {
                    assert_eq!(
                        index.ss_cutoff(&doses, t),
                        ss_reset_cutoff(&doses, t),
                        "t = {t:?}, doses = {doses:?}"
                    );
                }
            }
        }
    }

    /// A tie between `SS=1` records is decided by row order, and the scan reads
    /// `-0.0 == 0.0` — a sort under `total_cmp` would put the `-0.0` row first and hand
    /// back the wrong record. Built by hand so both cases are certainly present.
    #[test]
    fn a_co_timed_ss_tie_is_decided_by_row_order_and_signed_zero_ties() {
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(-0.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(5.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(5.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(3.0, 100.0, 1, 0.0, true, 12.0),
        ];
        let index = DoseIndex::new(&doses, &[]);
        assert_eq!(index.ss_cutoff(&doses, 0.0), Some(1));
        assert_eq!(ss_reset_cutoff(&doses, 0.0), Some(1));
        assert_eq!(index.ss_cutoff(&doses, 4.0), Some(4));
        assert_eq!(index.ss_cutoff(&doses, 5.0), Some(3));
        assert_eq!(index.ss_cutoff(&doses, -1.0), None);
    }

    /// A `NaN` record time has no place in a sort; the index defers to the scan.
    #[test]
    fn a_nan_ss_record_time_falls_back_to_the_scan() {
        let doses = vec![
            DoseEvent::new(1.0, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(f64::NAN, 100.0, 1, 0.0, true, 12.0),
            DoseEvent::new(2.0, 100.0, 1, 0.0, true, 12.0),
        ];
        let index = DoseIndex::new(&doses, &[]);
        for t in [0.0, 1.0, 1.5, 3.0, f64::NAN] {
            assert_eq!(index.ss_cutoff(&doses, t), ss_reset_cutoff(&doses, t), "t = {t}");
        }
    }

    /// What `compute_propagation_bounds`'s per-dose loop pushed for `(lo, hi)`, sorted.
    fn breaks_by_scan(doses: &[DoseEvent], lags: &[f64], lo: f64, hi: f64) -> Vec<f64> {
        let mut out = Vec::new();
        for (k, d) in doses.iter().enumerate() {
            if d.rate > 0.0 && d.duration > 0.0 {
                let lag = lags.get(k).copied().unwrap_or(0.0);
                let start = d.time + lag;
                let end = d.time + lag + d.duration;
                let residual = ss_residual_infusion_ends(d, lag, 1.0, lo, hi);
                for b in [start, end].into_iter().chain(residual) {
                    if b > lo && b < hi {
                        out.push(b);
                    }
                }
            }
        }
        out.sort_by(|a, b| a.total_cmp(b));
        out
    }

    /// With no residual pulses `breaks_within` hands back a slice of the index and allocates
    /// nothing (the per-interval hot path); with one it owns the merged, sorted list. Dies when
    /// the borrowed arm goes back to a `Vec`, or the residual arm stops merging.
    #[test]
    fn breaks_within_borrows_unless_a_residual_pulse_is_live() {
        use std::borrow::Cow;
        let plain = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, false, 0.0)];
        let index = DoseIndex::new(&plain, &[]);
        let got = index.breaks_within(0.5, 9.0);
        assert!(matches!(got, Cow::Borrowed(_)), "no residuals: borrowed");
        assert_eq!(&*got, &[5.0], "the window end only (onset 0 is outside)");
        // An SS infusion longer than its interval: earlier pulses still run at the record.
        let ss = vec![DoseEvent::new(0.0, 100.0, 1, 20.0, true, 4.0)];
        let index = DoseIndex::new(&ss, &[]);
        let got = index.breaks_within(-1.0, 9.0);
        assert!(matches!(got, Cow::Owned(_)), "residuals: owned");
        assert!(got.windows(2).all(|w| w[0] <= w[1]), "sorted: {got:?}");
        assert!(got.len() > 2, "the residual ends are merged: {got:?}");
    }

    #[test]
    fn the_breaks_within_an_interval_are_the_scans() {
        let mut rng = Rng(0x1477);
        for n in [0, 1, 3, 12, 40] {
            for _ in 0..30 {
                let (doses, lags) = random_doses(&mut rng, n);
                let index = DoseIndex::new(&doses, &lags);
                let ts = probes(&doses, &lags);
                for _ in 0..40 {
                    let a = ts[rng.below(ts.len() as u64) as usize];
                    let b = ts[rng.below(ts.len() as u64) as usize];
                    let (lo, hi) = (a.min(b) + 1e-15, a.max(b) - 1e-15);
                    let got: Vec<u64> =
                        index.breaks_within(lo, hi).iter().map(|x| x.to_bits()).collect();
                    let want: Vec<u64> =
                        breaks_by_scan(&doses, &lags, lo, hi).iter().map(|x| x.to_bits()).collect();
                    assert_eq!(got, want, "({lo}, {hi}), doses = {doses:?}, lags = {lags:?}");
                }
            }
        }
    }

    /// Every dose the walks' `1e-9` boundary match can pick within `[lo, hi]` is a
    /// candidate, and the candidates come ascending and once each — the order the scan's
    /// first match depends on.
    #[test]
    fn the_moving_point_candidates_cover_every_match_in_dose_order() {
        let mut rng = Rng(0x486);
        for n in [0, 1, 4, 20, 40] {
            for _ in 0..30 {
                let (doses, lags) = random_doses(&mut rng, n);
                let index = DoseIndex::new(&doses, &lags);
                let ts = probes(&doses, &lags);
                for _ in 0..40 {
                    let a = ts[rng.below(ts.len() as u64) as usize];
                    let b =
                        if rng.below(2) == 0 { a } else { ts[rng.below(ts.len() as u64) as usize] };
                    let (lo, hi) = (a.min(b), a.max(b));
                    let near = index.near_moving_points(lo, hi);
                    assert!(near.windows(2).all(|w| w[0] < w[1]), "not ascending: {near:?}");
                    for (k, d) in doses.iter().enumerate() {
                        let t_start = d.time + lags[k];
                        let mut points = vec![t_start];
                        if d.duration > 0.0 {
                            points.push(t_start + d.duration);
                        }
                        // A point matches some `w ∈ [lo, hi]` iff it is within `1e-9` of
                        // the interval; probe the nearest `w`.
                        let matches = points.iter().any(|&p| {
                            let w = p.clamp(lo, hi);
                            (w - p).abs() < 1e-9
                        });
                        if matches {
                            assert!(
                                near.contains(&k),
                                "dose {k} missing for [{lo}, {hi}]: {near:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The point of the index: on a long, evenly spaced history the candidates for one
    /// inter-dose interval are a handful, not every dose (#2754's `O(n²)`).
    #[test]
    fn one_interval_of_a_long_history_sees_a_handful_of_doses() {
        let doses: Vec<DoseEvent> =
            (0..400).map(|i| DoseEvent::new(12.0 * i as f64, 100.0, 1, 0.0, false, 0.0)).collect();
        let lags = vec![0.75; doses.len()];
        let index = DoseIndex::new(&doses, &lags);
        let near = index.near_moving_points(12.0 * 200.0 + 0.75, 12.0 * 201.0 + 0.75);
        assert_eq!(near, vec![200, 201]);
        assert!(index.infusing().is_empty(), "boluses carry no rate");
        assert_eq!(index.ss_cutoff(&doses, 1e9), None);
    }
}

#[cfg(test)]
mod first_record_tests {
    use super::{first_record, FirstRecord};
    use crate::types::Subject;
    use std::io::Write;

    /// Each subject is a tie at TIME 0 between two records of the kinds in its name, in
    /// the file order its name gives, followed by an observation at TIME 2. The first row
    /// of every tie carries WT = 70 and the second WT = 140, so the first record in file
    /// order — NONMEM's `A_0` seed (`nonmem_anchor/init_seed_tie`) — is the one with
    /// WT = 70.
    const TIES: &str = "\
ID,TIME,DV,MDV,EVID,AMT,CMT,WT
1,0,1,0,0,0,1,70
1,0,1,0,0,0,1,140
1,2,1,0,0,0,1,140
2,0,0,1,2,0,1,70
2,0,0,1,2,0,1,140
2,2,1,0,0,0,1,140
3,0,0,1,1,100,1,70
3,0,0,1,1,100,1,140
3,2,1,0,0,0,1,140
4,0,0,1,2,0,1,70
4,0,1,0,0,0,1,140
4,2,1,0,0,0,1,140
5,0,1,0,0,0,1,70
5,0,0,1,2,0,1,140
5,2,1,0,0,0,1,140
6,0,0,1,1,100,1,70
6,0,1,0,0,0,1,140
6,2,1,0,0,0,1,140
7,0,1,0,0,0,1,70
7,0,0,1,1,100,1,140
7,2,1,0,0,0,1,140
8,0,0,1,1,100,1,70
8,0,0,1,2,0,1,140
8,2,1,0,0,0,1,140
9,0,0,1,2,0,1,70
9,0,0,1,1,100,1,140
9,2,1,0,0,0,1,140
";

    fn wt(s: &Subject, rec: FirstRecord) -> f64 {
        let cov = match rec {
            FirstRecord::Dose(k) => s.dose_cov(k),
            FirstRecord::PkOnly(m) => s.pk_only_cov(m),
            FirstRecord::Obs(j) => s.obs_cov(j),
        };
        cov["WT"]
    }

    fn kind(rec: FirstRecord) -> &'static str {
        match rec {
            FirstRecord::Dose(_) => "dose",
            FirstRecord::PkOnly(_) => "pk-only",
            FirstRecord::Obs(_) => "obs",
        }
    }

    /// #2121: the selector returns the first record in file order for every pair of kinds
    /// and both file orders, on the reader's own output (so after its #614/#1810 ULP
    /// moves, which are what turn file order into the walk's order).
    #[test]
    fn first_record_is_first_in_file_order_for_every_kind_pair() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(TIES.as_bytes()).unwrap();
        let pop = crate::read_nonmem_csv(f.path(), None, None).unwrap();
        // (ID, kind of the first row in file order).
        let want = [
            ("1", "obs"),
            ("2", "pk-only"),
            ("3", "dose"),
            ("4", "pk-only"),
            ("5", "obs"),
            ("6", "dose"),
            ("7", "obs"),
            ("8", "dose"),
            ("9", "pk-only"),
        ];
        for (id, want_kind) in want {
            let s = pop.subjects.iter().find(|s| s.id == id).expect(id);
            let rec = first_record(s, s.doses.len()).expect("records");
            assert_eq!(
                (kind(rec), wt(s, rec)),
                (want_kind, 70.0),
                "ID {id}: the seed must be the tie's first row in file order"
            );
        }
        // The straddle: each mixed pair is in the file in both orders, and the two orders
        // pick records of different kinds. A rule that ignored file order (a fixed kind
        // preference, or last-wins) cannot satisfy both halves of a pair.
        for (a, b) in [("4", "5"), ("6", "7"), ("8", "9")] {
            let k = |id: &str| {
                let s = pop.subjects.iter().find(|s| s.id == id).unwrap();
                kind(first_record(s, s.doses.len()).unwrap())
            };
            assert_ne!(k(a), k(b), "IDs {a}/{b}: the two file orders pick one kind");
        }
    }

    /// `n_doses` caps the dose candidates (the adaptive replay's appended controller doses
    /// are not records), and a subject with no record has no seed.
    #[test]
    fn first_record_skips_doses_past_n_doses_and_is_none_without_records() {
        let mut s = Subject { id: "1".into(), obs_times: vec![1.0], ..Default::default() };
        s.doses.push(crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0));
        assert_eq!(first_record(&s, 1), Some(FirstRecord::Dose(0)));
        assert_eq!(first_record(&s, 0), Some(FirstRecord::Obs(0)));
        let empty = Subject::default();
        assert_eq!(first_record(&empty, 0), None);
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
        assert_eq!(tad(&d, 3.0, 480.0), Some(9.0), "the record sits at phase II - lag");
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
        assert_eq!(tad(&d, 3.0, 485.0), Some(2.0), "back on the post-arrival fold");
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
        assert_eq!(tad(&d, lag, 494.0), Some(14.0), "TAD exceeds II under the clamp");
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
        assert_eq!(tad(&d1, 3.0, 27.0), Some(0.0), "…and everything once it does");
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
        assert_eq!(tad_at(&doses, &[3.0], 485.0), 5.0, "dose 1 unlagged: pulse at 480");
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
        assert!(tad_referent(&d, 3.0, f64::NAN).is_none(), "NaN observation time");

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
        let mine: Vec<f64> =
            subject.obs_times.iter().map(|&t| super::tad_at(&subject.doses, &[], t)).collect();
        let theirs: Vec<f64> =
            (0..subject.obs_times.len()).map(|j| subject.time_after_dose_at_or_before(j)).collect();
        assert_eq!(
            mine, theirs,
            "the zero-lag fold and the sdtab fallback must agree on every row, including the \
             same-time tie at t = 24 where a dose record and an observation share a TIME"
        );
        // The straddle: assert the fixture actually reaches the tie, so this cannot quietly
        // become a comparison of two functions on rows where every spelling agrees.
        assert_eq!(mine[3], 0.0, "obs@24 takes the dose at 24 — NONMEM record order");
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
        assert_eq!(tad(&d, 3.0, 500.0), Some(17.0), "no wrap without an interval");
    }
}

#[cfg(test)]
mod infusion_rate_channel_tests {
    use super::*;

    /// The convention, stated once: `CMT=0` is the default dose *bolus* compartment
    /// (`cmt_idx() == 0`, same state as `CMT=1`), and it is **not** a zero-order input
    /// target. So the two resolved accessors deliberately disagree with this predicate on
    /// `CMT=0`, and that disagreement is the whole content of the function — a version
    /// that returned `d.cmt_idx() < n` or `d.cmt_1based() >= 1` would be `true`
    /// everywhere and silently re-open #1077 on all six call sites at once.
    #[test]
    fn only_cmt_zero_lacks_a_rate_channel() {
        let cmt0 = DoseEvent::new(0.0, 100.0, 0, 40.0, false, 0.0);
        let cmt1 = DoseEvent::new(0.0, 100.0, 1, 40.0, false, 0.0);
        let cmt3 = DoseEvent::new(0.0, 100.0, 3, 40.0, false, 0.0);

        assert!(!infusion_has_rate_channel(&cmt0));
        assert!(infusion_has_rate_channel(&cmt1));
        assert!(infusion_has_rate_channel(&cmt3));

        // `CMT=0` and `CMT=1` are the same *state*: the predicate is about the dose's
        // spelling, not about where the compartment is, which is why it cannot be derived
        // from either resolved accessor.
        assert_eq!(cmt0.cmt_idx(), cmt1.cmt_idx());
        assert_eq!(cmt0.cmt_1based(), cmt1.cmt_1based());
    }

    /// Orthogonal to `is_real_infusion`, on purpose: the break-time builders ask *that*
    /// about every infusion — including a `CMT=0` one, whose window end must still break
    /// the timeline for the segment structure to line up — and only the forcing sites ask
    /// this. Folding the two would delete a break and change segment geometry.
    #[test]
    fn it_is_independent_of_whether_the_row_is_a_real_infusion() {
        let bolus_cmt0 = DoseEvent::new(0.0, 100.0, 0, 0.0, false, 0.0);
        let inf_cmt0 = DoseEvent::new(0.0, 100.0, 0, 40.0, false, 0.0);
        assert!(!is_real_infusion(&bolus_cmt0) && !infusion_has_rate_channel(&bolus_cmt0));
        assert!(is_real_infusion(&inf_cmt0) && !infusion_has_rate_channel(&inf_cmt0));
    }
}

#[cfg(test)]
mod ss_reset_tests {
    use super::*;

    fn bolus(t: f64) -> DoseEvent {
        DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0)
    }
    fn ss(t: f64) -> DoseEvent {
        DoseEvent::new(t, 100.0, 1, 0.0, true, 12.0)
    }
    /// Indices live at `t`, the observable of the rule.
    fn live(doses: &[DoseEvent], t: f64) -> Vec<usize> {
        let c = ss_reset_cutoff(doses, t);
        (0..doses.len()).filter(|&k| ss_reset_live(doses, c, k)).collect()
    }

    #[test]
    fn with_no_ss_record_every_dose_is_live() {
        let doses = [bolus(0.0), bolus(12.0), bolus(24.0)];
        assert_eq!(ss_reset_cutoff(&doses, 100.0), None);
        assert_eq!(live(&doses, 100.0), vec![0, 1, 2]);
        // `SS=1` with `II = 0` is not a periodic record and resets nothing.
        let ii0 = [bolus(0.0), DoseEvent::new(10.0, 100.0, 1, 0.0, true, 0.0)];
        assert_eq!(ss_reset_cutoff(&ii0, 100.0), None);
    }

    #[test]
    fn an_ss_first_dose_and_later_non_ss_doses_all_stay_live() {
        // Row 4's shape: `SS=1` first, a non-SS continuation superposes on it.
        let doses = [ss(0.0), bolus(120.0)];
        assert_eq!(ss_reset_cutoff(&doses, 130.0), Some(0));
        assert_eq!(live(&doses, 130.0), vec![0, 1]);
    }

    #[test]
    fn an_ss_record_is_not_reached_before_its_time() {
        // Row 5's shape: before the record the bolus is the only thing live; at and after
        // it the bolus is dead.
        let doses = [bolus(0.0), ss(24.0)];
        assert_eq!(ss_reset_cutoff(&doses, 23.999), None);
        assert_eq!(live(&doses, 23.999), vec![0, 1]);
        assert_eq!(ss_reset_cutoff(&doses, 24.0), Some(1));
        assert_eq!(live(&doses, 24.0), vec![1]);
        assert_eq!(live(&doses, 50.0), vec![1]);
    }

    /// Both sides of the tie in one test (rows 51/52), so a predicate stuck on either
    /// branch reddens it: strict-time (`t_k > t_s`) wipes row 52's bolus, and a
    /// time-only `≥` keeps row 51's.
    #[test]
    fn a_co_timed_dose_is_wiped_before_the_ss_row_and_live_after_it() {
        let row51 = [bolus(10.0), ss(10.0)];
        assert_eq!(live(&row51, 10.0), vec![1], "row 51: bolus row BEFORE the SS row is wiped");
        let row52 = [ss(10.0), bolus(10.0)];
        assert_eq!(live(&row52, 10.0), vec![0, 1], "row 52: bolus row AFTER the SS row superposes");
    }

    #[test]
    fn the_latest_of_two_ss_records_governs() {
        // Row 41 (TDM "SS on every visit"): only the latest reached record is live.
        let doses = [ss(0.0), ss(12.0), ss(24.0)];
        assert_eq!(ss_reset_cutoff(&doses, 13.0), Some(1));
        assert_eq!(live(&doses, 13.0), vec![1, 2]);
        assert_eq!(live(&doses, 30.0), vec![2]);
    }

    #[test]
    fn an_unsorted_list_is_ordered_by_time_then_index() {
        // Base-then-controller shadow list: the later record sits earlier in the vector.
        let doses = [bolus(0.0), ss(48.0), ss(24.0), bolus(30.0)];
        assert_eq!(ss_reset_cutoff(&doses, 40.0), Some(2));
        assert_eq!(live(&doses, 40.0), vec![1, 2, 3]);
        assert_eq!(ss_reset_cutoff(&doses, 60.0), Some(1));
        assert_eq!(live(&doses, 60.0), vec![1]);
    }
}

#[cfg(test)]
mod evid_reset_tests {
    use super::evid_reset_live;

    /// #1587: the EVID=3/4 reset test is on the dose **record**. The arrival is not an
    /// argument at all, so a lagged dose recorded before the reset cannot survive it — the
    /// pair `(9, 10)` is the issue's case (a lag-2 arrival at 11). Both sides of the gate,
    /// and its edge: a dose recorded at the reset (an EVID=4 row's own dose) is live.
    #[test]
    fn a_dose_is_live_iff_recorded_at_or_after_the_reset() {
        assert!(evid_reset_live(9.0, f64::NEG_INFINITY), "no reset reached");
        assert!(!evid_reset_live(9.0, 10.0), "recorded before the reset");
        assert!(evid_reset_live(10.0, 10.0), "recorded at the reset (EVID=4)");
        assert!(evid_reset_live(10.001, 10.0), "recorded after the reset");
    }
}

#[cfg(test)]
mod residual_pulse_count_tests {
    use super::*;

    /// The loop `count_gt_multiples` replaced, as the reference.
    fn loop_count(x: f64, ii: f64) -> usize {
        let mut n = 0usize;
        while x > n as f64 * ii {
            n += 1;
        }
        n
    }

    /// The count is the loop's, for every `x` the loop could reach: a sweep through each
    /// boundary `k·II` (exactly on it, one ulp either side) and off it, at several `II`.
    #[test]
    fn count_matches_the_loop_through_every_boundary() {
        for ii in [12.0, 7.3, 0.1, 1.0 / 3.0] {
            for k in 1..60usize {
                let b = k as f64 * ii;
                for x in [
                    b,
                    f64::from_bits(b.to_bits() + 1),
                    f64::from_bits(b.to_bits() - 1),
                    b + 0.5 * ii,
                    b - 0.5 * ii,
                ] {
                    assert_eq!(
                        count_gt_multiples(x, ii),
                        loop_count(x, ii),
                        "x = {x:e}, II = {ii}"
                    );
                }
            }
        }
        assert_eq!(count_gt_multiples(-3.0, 12.0), 0);
        assert_eq!(count_gt_multiples(0.0, 12.0), 0);
    }

    /// `T_inf ∈ {6, 30, 1e6, 1e11, 1e300}` at `II = 12`. Integer `T_inf` against exact integer
    /// arithmetic (`n = ⌈x/II⌉` in `u128`); `30` and `6` against the loop; `1e300` saturates —
    /// no finite count reaches it — instead of looping or allocating.
    #[test]
    fn count_for_large_windows_is_exact_integer_arithmetic() {
        for x in [6.0f64, 30.0, 1e6, 1e11] {
            let exact = ((x as u128) + 11) / 12;
            assert_eq!(count_gt_multiples(x, 12.0) as u128, exact, "x = {x:e}");
        }
        assert_eq!(count_gt_multiples(6.0, 12.0), loop_count(6.0, 12.0));
        assert_eq!(count_gt_multiples(30.0, 12.0), loop_count(30.0, 12.0));
        assert_eq!(count_gt_multiples(1e6, 12.0), loop_count(1e6, 12.0));
        assert_eq!(count_gt_multiples(1e300, 12.0), usize::MAX);
        // And through the dose entry point: `RATE = 1e-9` makes `T_inf = 1e11`, `n ~ 8.3e9`.
        let d = DoseEvent::new(0.0, 100.0, 1, 1e-9, true, 12.0);
        let (first, n) = ss_residual_infusions_g(&d, 0.0, d.duration);
        assert_eq!(first, d.duration - 12.0);
        assert_eq!(n as u128, (((d.duration - 12.0) as u128) + 11) / 12);
    }

    /// The ends listed for a time range are exactly the loop's materialised ends that lie in
    /// it, in the same (descending) order, and the running count is the loop's.
    #[test]
    fn ends_in_a_range_are_the_loops_ends_in_that_range() {
        let ends_of = |d: &DoseEvent, lag: f64| -> Vec<f64> {
            let (first, _) = ss_residual_infusions_g(d, lag, d.duration);
            let n = loop_count(first, d.ii);
            (0..n).map(|j| d.time + first - j as f64 * d.ii).collect()
        };
        // T_inf = 20 over II = 8, unseeded and seeded (lag 3); and T_inf = 250 over II = 12.
        for (d, lag) in [
            (DoseEvent::new(5.0, 1000.0, 1, 50.0, true, 8.0), 0.0),
            (DoseEvent::new(5.0, 1000.0, 1, 50.0, true, 8.0), 3.0),
            (DoseEvent::new(0.0, 1000.0, 1, 4.0, true, 12.0), 0.0),
        ] {
            let all = ends_of(&d, lag);
            assert!(!all.is_empty());
            for (lo, hi) in [
                (f64::NEG_INFINITY, f64::INFINITY),
                (0.0, 40.0),
                (all[all.len() - 1], all[0]),
                (all[all.len() - 1] + 1e-9, all[0] - 1e-9),
                (30.0, 20.0),
                (1e9, 2e9),
                (-5.0, -1.0),
            ] {
                let want: Vec<f64> = all.iter().copied().filter(|&e| e >= lo && e <= hi).collect();
                let got: Vec<f64> = ss_residual_infusion_ends(&d, lag, 1.0, lo, hi).collect();
                assert_eq!(got, want, "dose {d:?} lag {lag} range [{lo}, {hi}]");
            }
            for lo in [f64::NEG_INFINITY, 0.0, 12.0, 25.5, all[0], all[0] + 1.0] {
                let want = all.iter().filter(|&&e| e >= lo).count();
                assert_eq!(
                    ss_residual_ends(&d, lag, 1.0).count_ending_at_or_after(lo),
                    want,
                    "count at {lo}"
                );
            }
        }
    }

    /// The case the cap rejected: `n ~ 8e9` pulses, every one of them running, and none
    /// materialised — the in-horizon list is empty, the far count is exact.
    #[test]
    fn astronomical_pulse_count_is_not_materialised() {
        let d = DoseEvent::new(0.0, 100.0, 1, 1e-9, true, 12.0);
        let re = ss_residual_ends(&d, 0.0, 1.0);
        let (_, n) = ss_residual_infusions_g(&d, 0.0, d.duration);
        assert!(n > 8_000_000_000);
        // `first ≡ 4 (mod 12)`: the ends in `(0, 100]` are 4, 16, …, 100 — the last 9 of the
        // `n`. Every other pulse ends beyond 100: counted, not listed.
        let near: Vec<f64> = ss_residual_infusion_ends(&d, 0.0, 1.0, 0.0, 100.0).collect();
        assert_eq!(near.len(), 9);
        assert_eq!((near[0], near[8]), (100.0, 4.0));
        assert_eq!(re.count_ending_after(100.0), n - 9);
        assert_eq!(re.count_ending_at_or_after(0.0), n);
        assert_eq!(re.count_ending_at_or_after(100.0), n - 8);
    }
}

#[cfg(test)]
mod zero_order_binding_tests {
    //! A fit lowers an `SS=1` dose into a `zero_order()` window once, where it binds the data,
    //! not by a clone of the subject in every walk (#2786 review).
    use super::*;
    use std::path::Path;
    use std::sync::atomic::Ordering::Relaxed;

    const DATA: &str = "ID,TIME,EVID,MDV,DV,AMT,RATE,CMT,II,SS
1,0,1,1,0,4321.5,0,1,12,1
1,2,0,0,80.0,0,0,1,0,0
1,6,0,0,55.0,0,0,1,0,0
1,10,0,0,40.0,0,0,1,0,0
2,0,1,1,0,4321.5,0,1,12,1
2,2,0,0,90.0,0,0,1,0,0
2,6,0,0,60.0,0,0,1,0,0
2,10,0,0,45.0,0,0,1,0,0
";

    fn fixture(amt: &str) -> (crate::types::CompiledModel, crate::types::Population) {
        let model = crate::parser::model_parser::parse_model_file(Path::new(
            "examples/zero_order_absorption.ferx",
        ))
        .expect("model");
        let dir = tempfile::tempdir().expect("tmp");
        let csv = dir.path().join("d.csv");
        std::fs::write(&csv, DATA.replace("4321.5", amt)).expect("write");
        let pop = crate::read_nonmem_csv(&csv, None, None).expect("data");
        (model, pop)
    }

    /// Mutation: return `Cow::Borrowed` from `lower_population_zero_order_ss` -> the bound
    /// population is not lowered and every walk clones its subject.
    #[test]
    fn a_bound_population_is_lowered_once_and_its_walks_borrow() {
        let (model, pop) = fixture("4321.25");
        let map = model.active_dose_attr_map();
        assert!(
            matches!(lower_zero_order_ss(&pop.subjects[0], map), Cow::Owned(_)),
            "fixture: the data as read is rewritten by every walk"
        );
        let bound = lower_population_zero_order_ss(&pop, map);
        assert!(matches!(bound, Cow::Owned(_)));
        for s in &bound.subjects {
            assert!(matches!(lower_zero_order_ss(s, map), Cow::Borrowed(_)));
            assert!(is_lowered_zero_order_ss(map, &s.doses[0]));
        }
        assert!(matches!(lower_population_zero_order_ss(&bound, map), Cow::Borrowed(_)));
        // The data as read is untouched: the lowering is the fit's own.
        assert!(!pop.subjects[0].doses[0].is_infusion());
    }

    /// A whole fit lowers each affected subject exactly once. Mutation: drop the binding in
    /// `fit_inner` -> every walk lowers (thousands of lowerings, not 2).
    #[test]
    fn a_fit_lowers_each_subject_once() {
        let (model, pop) = fixture("4321.5");
        let o = crate::types::FitOptions {
            method: crate::types::EstimationMethod::FoceI,
            outer_maxiter: 2,
            threads: Some(1),
            verbose: false,
            ..crate::types::FitOptions::default()
        };
        let before = MARKER_LOWERINGS.load(Relaxed);
        crate::api::fit(&model, &pop, &model.default_params, &o).expect("fit");
        let n = MARKER_LOWERINGS.load(Relaxed) - before;
        assert_eq!(n, pop.subjects.len(), "lowerings per fit");
    }
}

#[cfg(test)]
mod tad_referent_g_tests {
    use super::{tad_referent, tad_referent_g, wrap_into_period_g, TAD_EPS};
    use crate::sens::dual2::Dual2;
    use crate::types::DoseEvent;

    /// The rule as `tad_referent` was written before it became the `f64` instance of
    /// [`tad_referent_g`] -- the reference the instance must still equal bit for bit.
    fn reference(dose: &DoseEvent, lag: f64, t: f64) -> Option<f64> {
        let arrival = dose.time + lag;
        if arrival <= t + TAD_EPS {
            if dose.ss && dose.ii > 0.0 {
                let elapsed = t - arrival;
                return Some(t - elapsed.rem_euclid(dose.ii));
            }
            return Some(arrival);
        }
        if super::ss_seeded_at_record(dose, lag) && dose.time <= t + TAD_EPS {
            return Some(dose.time - super::ss_seed_phase(dose, lag));
        }
        None
    }

    /// Both ends of the walk's anchor fold: the `f64` instance is the old rule bit for bit (on
    /// dyadic and non-dyadic `II`, before the record, in the seeded window, past the arrival
    /// and at later cycles, with `lag` below, at and above `II`), and over `Dual2` the value is
    /// the same number with the lag's jet: `∂referent/∂lag = +1` where the referent follows the
    /// arrival or the seed's previous pulse (`d.time - (II - lag)`), `0` where the phase is
    /// clamped (`lag >= II`), and `∂/∂t = 0` for a wrapped steady-state referent. The derivative
    /// is checked against a central difference of the `f64` rule, away from its kinks.
    /// Mutations: no wrap at later cycles (`elapsed` for `wrap_into_period_g`) -> the bit
    /// comparison and the later-cycle `∂/∂t` fail; `Lt` for the clamp in `ss_seed_phase_g` ->
    /// the `lag >= II` jet row fails.
    #[test]
    fn the_generic_referent_is_the_f64_rule_and_carries_the_lag_jet() {
        for ii in [12.0, 7.3, 0.1 + 0.2] {
            for ss in [true, false] {
                for lag in [0.0, 0.4, 3.0, 11.0, 12.5, 15.0] {
                    let d = DoseEvent::new(5.0, 100.0, 1, 0.0, ss, if ss { ii } else { 0.0 });
                    for k in 0..60 {
                        let t = 4.7 + 0.9 * k as f64 + 0.013 * (k % 7) as f64;
                        let old = reference(&d, lag, t);
                        assert_eq!(
                            tad_referent(&d, lag, t).map(f64::to_bits),
                            old.map(f64::to_bits),
                            "f64 instance: ii {ii} ss {ss} lag {lag} t {t}"
                        );
                        let got =
                            tad_referent_g(&d, Dual2::<2>::var(lag, 0), Dual2::<2>::var(t, 1));
                        assert_eq!(
                            got.map(|g| g.value.to_bits()),
                            old.map(f64::to_bits),
                            "Dual2 value: ii {ii} ss {ss} lag {lag} t {t}"
                        );
                        let Some(g) = got else { continue };
                        // The finite differences must not straddle a kink of the rule.
                        let h = 1e-6;
                        let near = |a: f64, b: f64| (a - b).abs() < 10.0 * h;
                        let arrival = d.time + lag;
                        let kink = near(t, arrival)
                            || near(t, d.time)
                            || (ss
                                && ((t - arrival) / ii - ((t - arrival) / ii).round()).abs()
                                    < 1e-5)
                            || near(lag, ii);
                        if kink {
                            continue;
                        }
                        let fd_lag = (reference(&d, lag + h, t).unwrap()
                            - reference(&d, lag - h, t).unwrap())
                            / (2.0 * h);
                        let fd_t = (reference(&d, lag, t + h).unwrap()
                            - reference(&d, lag, t - h).unwrap())
                            / (2.0 * h);
                        assert!(
                            (g.grad[0] - fd_lag).abs() < 1e-6 && (g.grad[1] - fd_t).abs() < 1e-6,
                            "jet: ii {ii} ss {ss} lag {lag} t {t}: {:?} vs ({fd_lag}, {fd_t})",
                            g.grad
                        );
                    }
                }
            }
        }
    }

    /// The wrap alone: a plain number takes `f64::rem_euclid`'s remainder bit for bit, and a
    /// negative or sub-period value wraps the same way.
    #[test]
    fn the_period_wrap_is_rem_euclid_bit_for_bit() {
        for ii in [12.0, 7.3, 0.3] {
            for x in [-30.5, -0.2, 0.0, 0.2, 7.3, 11.99, 12.0, 25.7, 479.123] {
                assert_eq!(
                    wrap_into_period_g(x, ii).to_bits(),
                    f64::rem_euclid(x, ii).to_bits(),
                    "x {x} ii {ii}"
                );
            }
        }
    }
}
