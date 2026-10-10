//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.
//!
//! Infusion doses (`rate > 0`) are handled by breaking the timeline at the
//! infusion's end time and adding `+rate` to the corresponding compartment's
//! derivative for the duration of the infusion via an RHS wrapper.

use crate::ode::solver::{
    solve_ode, solve_ode_dense_with_auto_state, OdeAutoSwitchState, OdeSolverOptions,
    OdeSolverStats,
};
use crate::pk::absorption::{InputRateForcing, PreparedInputRate};
use crate::sim::adaptive::{
    assay_standard_normal, AdaptiveMonitor, AdaptiveRun, AssayNoise, ControllerCtx,
    ControllerDecision, DecisionLogEntry, DecisionOutcome, DoseAction, DoseLedgerEntry,
    ObserveMode, ObservedSignal,
};
// `MonitorSpec` is named only by the `#[cfg(test)]` cmt-only wrapper and the
// driver's unit tests (production pairs it inside `AdaptiveMonitor`).
#[cfg(test)]
use crate::sim::adaptive::MonitorSpec;
use crate::types::{DoseEvent, PkParams, Subject};
use std::collections::HashMap;

/// Epsilon used to decide whether an infusion fully spans a segment.
/// Break times are constructed to coincide with infusion start/end so any
/// non-degenerate segment is either fully inside or fully outside each
/// infusion window — this tolerance only guards float-equality on the bound.
/// `pub(crate)` so the analytic-sensitivity walks reuse the same value rather
/// than hard-coding a parallel literal (#472 review [7]).
pub(crate) const INFUSION_EPS: f64 = 1e-12;

/// Tolerance for matching a break time to the **event** it stands for — a dose
/// arrival (`dose.time + lag`), an SS dose's record-time seed, or a system-reset
/// time (EVID=3/4) — on every engine that resolves its events by rescanning the
/// timeline (#716, #1186). Each such time is pushed into `break_times`, which is
/// then deduped at `1e-15`, so an event merged into a sub-`1e-15` neighbour is
/// still applied at that representative break rather than dropped.
///
/// **Invariant: `dedup (1e-15) ≤ EVENT_MATCH_TOL`, and a dose fires at the first
/// break within `EVENT_MATCH_TOL` and at no other** — enforced by the
/// `seed_applied` / `applied` masks the rescanning loops carry, not by the
/// tolerance. #1186: a *derived* break (a route onset `dose.time + lag_cmt +
/// lag_route`, an infusion end `dose.time + amt/rate`) is a multi-term float sum,
/// so it routinely lands 1–2 ULP from another dose's own break — past the `1e-15`
/// dedup and well inside this match. Every break in that gap used to re-apply the
/// dose, doubling a bolus (144.04 → 244.04 on the #1186 fixture) or pushing an
/// infusion into `active_infusions` twice. No pair of tolerances fixes that: with
/// dedup `D` and match `M`, `D ≥ 2M` permits zero applications and `D ≤ M` permits
/// two — only an apply-once mask gives exactly one. Widening `D` instead would also
/// re-segment every engine and trip the adaptive exact-bit guards (#700).
///
/// One value on every engine, deliberately: it used to be `1e-12` on the objective
/// path and `1e-10` on the sdtab / dense / hazard paths, a 100× asymmetry that let
/// the same dataset double a dose in sdtab, the joint PK-TTE hazard, `[derived]`
/// integrals and `simulate()` while the OFV was correct. Tightening the two `1e-10`s
/// is safe under the same argument that bounds the mask: a dose's own break is
/// pushed from the identical expression, so the distance is zero.
///
/// Same magnitude as [`INFUSION_EPS`], which stays separate — that is a containment
/// epsilon on an infusion *window*, a different role.
///
/// # The recording rule (#1226)
///
/// The same tolerance decides where a *record* — an observation, a `saveat` grid point,
/// a soft (CHZ) sample — is read, and there it is deliberately **one-sided**. For a break
/// `t_k` and its successor `t_{k+1}`:
///
///  - **band** — `t_k ≤ t < t_k + EVENT_MATCH_TOL` is read **at `t_k`, after** that
///    break's events (reset → dose → read), from the post-event state `u(t_k⁺)`; see
///    [`reads_at_break`]. The shortcut error against the true `u(t)` is `≤ |f|·1e-12`,
///    below every solver tolerance, and it is measured rather than argued in
///    `lag_arrival_read_1226::band_read_is_continuous_across_the_tolerance_edge`.
///  - **segment** — `t_k + EVENT_MATCH_TOL ≤ t ≤ t_{k+1}` is recorded off the
///    integration of `(t_k, t_{k+1}]`; see [`reads_in_segment`].
///
/// Not symmetric: a record within tolerance *before* a break stays on the pre-event
/// side, because a dose record strictly earlier than an observation record is applied
/// first and one strictly later is not — NONMEM's ordering, anchored on both sides by
/// `nonmem_anchor/lag_arrival_read_{before,after}_advan{1,13}`.
///
/// The old segment bound was `t <= t_end + 1e-12`, which handed a within-tolerance-
/// **after** record to the segment *ending* at the break — i.e. to the state before the
/// dose was applied. #1226: an estimated `ALAG` whose arrival landed `1.3e-13` short of a
/// sample made `ode_predictions`, `ode_predictions_with_states` and
/// `ode_dense_solve_states` read that subject drug-free at a post-dose sample (45.38
/// against NONMEM's 145.38), moving the OFV and not only a diagnostic. The mirror sign
/// was wrong in the opposite direction on one site only — the #570 shared solve's CHZ
/// boundary read used a *symmetric* `abs() < 1e-12`, so a hazard time `1.8e-15` **before**
/// an arrival was overwritten with the post-dose state.
///
/// The two predicates partition `[t_k, t_{k+1}]` whenever adjacent breaks are at least
/// `EVENT_MATCH_TOL` apart. When two breaks are closer than that, both bands can claim a
/// time; the loops visit breaks in ascending order, so the **later** write wins, which is
/// the right answer (the latest break at or before `t`).
pub(crate) const EVENT_MATCH_TOL: f64 = 1e-12;

/// True when a record time `t` is read **at** the break `t_break` — from the post-event
/// state `u(t_break⁺)` — rather than off the integration that follows it.
///
/// One-sided: the band is the half-open `[t_break, t_break + EVENT_MATCH_TOL)`. See
/// [`EVENT_MATCH_TOL`] for why the *before* side must stay on the pre-event state.
///
/// Written as a **difference**, `t - t_break < TOL`, never as the sum `t < t_break + TOL`.
/// `EVENT_MATCH_TOL` is absolute, so the sum rounds back to `t_break` once `ulp(t_break)`
/// exceeds it and the half-open band becomes **empty** — the predicate then fails even at
/// bit equality, which the exact-bit `obs_map` lookups it replaced always matched. Measured:
///
/// | `t_break` | `ulp` | band width, sum form | difference form |
/// |---|---|---|---|
/// | `8.2` | 1.776e-15 | 563 ulp | 563 ulp |
/// | `1000.0` | 1.137e-13 | 9 ulp | 9 ulp |
/// | `8192.0` | 1.819e-12 | 1 ulp | 1 ulp |
/// | `16384.0` | 3.638e-12 | **0 ulp — matches nothing** | 1 ulp (bit equality) |
/// | `17520.0` | 3.638e-12 | **0 ulp** | 1 ulp |
///
/// `17520` is hours in two years, so this is an ordinary study timescale, not a corner.
/// The difference form degrades gracefully: as `ulp` grows past the tolerance the band
/// narrows to exactly the bit-equal set and never below it. That is the same shape as the
/// dose-application predicate (`(dose.time - t_start).abs() < EVENT_MATCH_TOL`), which is
/// why that one never had this failure mode.
#[inline]
pub(crate) fn reads_at_break(t: f64, t_break: f64) -> bool {
    t >= t_break && t - t_break < EVENT_MATCH_TOL
}

/// True when a record time `t` is recorded off the integration of `(t_start, t_end]` —
/// the complement of [`reads_at_break`]`(t, t_start)` on `[t_start, t_end]`.
///
/// The upper bound is **exact**: a time up to `EVENT_MATCH_TOL` past `t_end` belongs to
/// `t_end`'s own band, on the next iteration, after that break's events are applied.
///
/// The **upper** bound's exactness is hygiene rather than the fix, and that was verified by
/// running the mutation rather than argued: restoring `t <= t_end + 1e-12` here, in
/// `ode_predictions_with_states`, in `ode_dense_solve_states` or in `ode/ekf.rs` leaves the
/// whole suite green (re-measured against the current suite: 4228 pass). These engines have
/// no first-write-wins guard, so the band read at the next break simply overwrites the
/// pre-event value the wider bound let through — the band reads are the fix. It is kept
/// because it stops a record being written twice and stops the solver being handed a
/// `saveat` point outside its own span; the only place where an equivalent slack is
/// load-bearing is `sens/ode_provider.rs`, whose `recorded[j]` mask *is* first-write-wins
/// and which therefore needed an explicit overwrite instead.
///
/// The **lower** bound is a different matter and is not hygiene: written as a sum it admits
/// `t == t_start` into this segment's own `saveat` once `t_start + TOL == t_start`, and
/// `lag_arrival_read_1226::the_band_still_matches_bit_equality_at_large_times` fails on that
/// spelling. The difference form makes `t > t_start` structural, so the contract holds by
/// construction rather than by a tolerance being small enough.
///
/// The lower bound is a **difference** for the reason [`reads_at_break`] gives, and that
/// also keeps it strictly greater than `t_start` at every magnitude: `t - t_start >= TOL`
/// implies `t > t_start`, so the `(t_start, t_end]` contract the surrounding `saveat`
/// comments state holds by construction. The sum form `t >= t_start + TOL` did not — once
/// `t_start + TOL == t_start` it admits `t == t_start` into that segment's own `saveat`,
/// handing `solve_ode` a save point equal to `t0`, outside its span.
#[inline]
pub(crate) fn reads_in_segment(t: f64, t_start: f64, t_end: f64) -> bool {
    t - t_start >= EVENT_MATCH_TOL && t <= t_end
}

/// A subject's record times sorted once, for resolving which records each break claims.
///
/// Built per subject and queried per break, so the whole break loop costs
/// `O(n log n) + n_breaks·(log n + k)` instead of the `O(n_breaks · n)` a rescan per break
/// costs. Measured over a full subject walk with `n_breaks == n_records`, this spelling
/// against the linear scan it replaced:
///
/// | records | 5 | 10 | 20 | 50 | 100 | 1000 | 5000 |
/// |---|---|---|---|---|---|---|---|
/// | speedup | 0.20× | 0.35× | 1.03× | 2.27× | 3.91× | 16.0× | 102.8× |
///
/// Below ~20 records the sort does not pay for itself, but the absolute cost there is
/// +140 ns per subject against an ODE solve of tens of microseconds. Above it the win is
/// the one that matters: [`ode_dense_solve_states`] is driven by a *grid* — a hazard
/// timeline, a `[derived]` integral, an AUC or `predict_survival` horizon — which routinely
/// runs to hundreds or thousands of points.
///
/// This is also the one spelling of the band rule. `sens/ode_provider.rs` already resolved
/// its boundary records by sorted binary search (#438 review); having the ODE engines rescan
/// linearly meant the same rule was written twice, in two shapes that could drift.
pub(crate) struct RecordIndex {
    sorted: Vec<(f64, usize)>,
}

impl RecordIndex {
    /// Sort `(time, index)` ascending by [`f64::total_cmp`] — a total order, so a `NaN`
    /// record time sorts last and cannot make the sort panic (#1189).
    pub(crate) fn new(times: &[f64]) -> Self {
        let mut sorted: Vec<(f64, usize)> = times.iter().copied().zip(0..).collect();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { sorted }
    }

    /// The sorted `(time, index)` pairs, for callers that need a different window over the
    /// same order (the provider's tolerant `record_at` catch-all, #410).
    pub(crate) fn sorted(&self) -> &[(f64, usize)] {
        &self.sorted
    }

    /// Indices of the records [`reads_at_break`] assigns to `t_break`, written into `out`
    /// (cleared first) so the caller can hoist one allocation out of its break loop.
    ///
    /// This **replaces** the exact-bit `obs_map.get(&t_start.to_bits())` boundary lookups the
    /// rescanning engines used to do — the band is a superset of the exact hit, and keeping
    /// both would record a band member twice. `obs_map` stays for matching the solver's
    /// returned save points, whose bits are the `saveat` entries' own.
    ///
    /// Indices come back in **time order** (index order within one time), not in the
    /// caller's original record order. Every consumer but one writes by index and does not
    /// care; `ode/ekf.rs` assimilates sequentially, and taking the earlier measurement first
    /// is the order it should have.
    pub(crate) fn records_at_break(&self, t_break: f64, out: &mut Vec<usize>) {
        out.clear();
        // `t < t_break` is false for `NaN`, so the NaN tail sits in the upper partition and
        // is rejected by `reads_at_break` on the first comparison.
        let lo = self.sorted.partition_point(|&(t, _)| t < t_break);
        for &(t, j) in &self.sorted[lo..] {
            if !reads_at_break(t, t_break) {
                break;
            }
            out.push(j);
        }
    }

    /// Whether any record is read at `t_break` — the same question [`Self::records_at_break`]
    /// answers, without materialising the indices.
    pub(crate) fn any_at_break(&self, t_break: f64) -> bool {
        let lo = self.sorted.partition_point(|&(t, _)| t < t_break);
        self.sorted
            .get(lo)
            .is_some_and(|&(t, _)| reads_at_break(t, t_break))
    }
}

/// True when a subject's integration timeline carries a non-finite entry (#1189).
///
/// A `NaN` compartment lag (`ALAG`) or route lag makes `dose.time + lag` — and every
/// derived break built from it — `NaN`, and the same holds for an infusion end
/// `amt/rate` when `rate` is `NaN`. Two things must then happen, and neither is
/// automatic:
///
///  - the **sort must not panic**. Every timeline sort here uses [`f64::total_cmp`],
///    a total order that puts `NaN` last. `partial_cmp(..).unwrap()` panicked outright
///    and deterministically. `partial_cmp(..).unwrap_or(Ordering::Equal)` — the spelling
///    three call sites used *as* the NaN-safe fix — is no better: it is not a total order
///    either, and Rust's `sort_by` detects that and panics ("user-provided comparison
///    function does not correctly implement a total order"). That detection is
///    **opportunistic**, not a threshold: measured on this toolchain it fires for 30 and
///    84 events of the analytical walk's own `Event` type but not for 24, 40 or 60, and
///    adding one `usize` field to the element flips shapes either way. So the old
///    spelling was neither safe nor reliably loud — which is the worst of both.
///  - the **subject must come back non-finite**. `total_cmp` alone is a *silent wrong
///    number*: the `NaN`-lagged dose simply never matches a break, so it is never
///    applied and the remaining trajectory is finite — a drug-free subject reported as
///    a valid prediction. Every builder therefore checks this and returns its own
///    engine's non-finite outcome, which the estimation guards already handle
///    (`inner_optimizer`'s and `likelihood`'s `!is_finite()` arms; the TTE half maps it
///    to the `1e20` sentinel).
///
/// The front door is `check_dose_attr_finiteness` (`E_DOSE_ATTR_NONFINITE`), which
/// rejects a non-finite `ALAG`/`F` at typical values before the fit starts; this guard
/// is for the mid-fit θ/η excursion that no init-time check can see.
#[inline]
pub(crate) fn timeline_has_non_finite(break_times: &[f64]) -> bool {
    times_have_non_finite(break_times.iter().copied())
}

/// [`timeline_has_non_finite`] for a walk whose timeline is not a `&[f64]` — the two
/// event-driven engines carry `(time, kind, idx)` tuples. Takes an iterator so those
/// call sites share this one definition instead of open-coding `!is_finite()`, and
/// without allocating a temporary `Vec` on a per-subject hot path.
#[inline]
pub(crate) fn times_have_non_finite(mut times: impl Iterator<Item = f64>) -> bool {
    times.any(|t| !t.is_finite())
}

/// **The engine guard: `true` when this walk must be abandoned, and the abandonment recorded.**
///
/// The predicate and the counter are deliberately one call (#1234). Nothing about
/// [`timeline_has_non_finite`] makes recording structural, and the counter is only worth
/// having if every abandoning engine bumps it: a ninth guard written as a bare predicate would
/// compile, run, and put the diagnostic back to `0/0/0` for that walk — indistinguishable from
/// a subject there was nothing to integrate for, which is the whole defect. Going through here
/// makes forgetting impossible rather than conventional, and
/// `the_bare_timeline_predicates_are_not_called_outside_this_guard` fails if a new call site
/// takes the bare spelling.
///
/// `times` is an iterator so the dense builders (`break_times.iter().copied()`) and the
/// event-driven ones (`timeline.iter().map(|e| e.0)`) share one definition; `stats` is the
/// caller's out-parameter where it has one — `Some` only in
/// `ode_predictions_with_extra_breaks_and_stats`, whose public wrapper
/// [`ode_predictions_with_solver_stats`] returns it. See
/// [`crate::ode::solver::record_abandoned_non_finite_timeline`] for why both channels are fed.
///
/// **Deliberately not used by the two `sens/ode_provider.rs` walks**, which carry the same
/// predicate and do *not* record: their sweep is collected in its own scope from which
/// `fit_inner` copies one unrelated field, so a gradient-solve event deposited here would be
/// discarded or fire a prediction-shaped warning clause. That exclusion is the guard test's
/// only allowance, and it is stated there.
#[inline]
pub(crate) fn abandon_non_finite_timeline(
    times: impl Iterator<Item = f64>,
    stats: Option<&mut crate::ode::solver::OdeSolverStats>,
) -> bool {
    if times_have_non_finite(times) {
        crate::ode::solver::record_abandoned_non_finite_timeline(stats);
        true
    } else {
        false
    }
}

// Dose resolution + SS-equilibration primitives moved to `crate::dosing` (a neutral
// leaf module) so pk/sens/api don't depend upward on ode/. A PRIVATE import (NOT a
// `pub(crate) use` re-export) so these do not leak back out as `crate::ode::…` — the
// upward dependency this move removed stays removed. The ode-internal resolve callers
// + `equilibrate_ss_state` use the bare names; the `#[cfg(test)] mod tests` picks them
// up via `use super::*`. Test-only symbols (`ss_cycle_converged`, `SS_EQUILIBRATION_TOL`,
// `last_ss_equilibration_cycles`, `with_full_ss_equilibration`) are referenced directly
// as `crate::dosing::…` by the tests, so they are not imported here.
use crate::dosing::{
    is_real_infusion, note_ss_nonconvergence_if_capped, record_ss_equilibration_cycles,
    resolve_subject_doses, resolve_subject_doses_with, ss_equilibrates_at_arrival,
    ss_residual_infusion_end, ss_seed_phase, ss_seeded_at_record, SsStopTracker,
    SS_EQUILIBRATION_CYCLES,
};

/// Relative floor for truncating the steady-state **input-rate periodic sum** (#719). An
/// `SS=1` dose into a built-in absorption compartment stands for an infinite past pulse
/// train, so its appearance rate at time `t` is `Σ_{j≥0} R_in(tad + j·II)` — the tail of
/// every prior pulse still arriving (see [`add_prepared_input_rate_forcing`]). The absorption
/// density is eventually monotone-decreasing, so once a term falls below this fraction of the
/// running sum the remaining tail is spent and the sum stops (hard-capped at
/// [`crate::dosing::SS_EQUILIBRATION_CYCLES`] so a pathologically slow absorption — mode ≫ II
/// — still terminates, matching the trough's own cycle budget). Conservative (`1e-10`): the
/// dropped tail is far below the provider-vs-production parity tolerance. (Kept in
/// `ode/predictions` — its only consumers — rather than in the neutral `dosing` module.)
const SS_TAIL_REL_FLOOR: f64 = 1e-10;

/// The time at which a subject's integration begins: the earliest event on the
/// subject's timeline (first dose, observation, PK-only sample, or reset).
///
/// The dense/static drivers seed their `break_times` here rather than at a fixed
/// `t = 0`. This mirrors NONMEM (and the event-driven walk, which already starts
/// at `timeline[0]`): the initial state is applied at the first record, so a
/// dataset whose TIME column starts off-zero is *not* integrated over a phantom
/// `[0, first_record]` window. TIME stays on the raw data clock everywhere — no
/// per-subject origin shift (#573).
pub(crate) fn subject_integration_start(subject: &Subject) -> f64 {
    let mut t0 = f64::INFINITY;
    for &t in &subject.obs_times {
        t0 = t0.min(t);
    }
    for d in &subject.doses {
        t0 = t0.min(d.time);
    }
    for &t in &subject.pk_only_times {
        t0 = t0.min(t);
    }
    for &t in &subject.reset_times {
        t0 = t0.min(t);
    }
    // No events at all → fall back to the historical t = 0 start.
    if t0.is_finite() {
        t0
    } else {
        0.0
    }
}

/// Fill every requested sample time that falls **before the first break** with the seeded
/// initial state `u`.
///
/// Nothing has acted on the system before the first event, so that *is* the state there. Both
/// engines that read states at caller-supplied times need this and must agree on it: the
/// dedicated [`ode_dense_solve_states`] (whose `saveat` may hold a CTMM observation recorded
/// before the first dose) and the #570 one-solve share
/// [`ode_predictions_and_chz`] (whose `chz_times` may hold a left-truncation `TENTRY` or an
/// interval-censored `left`). Left as `NaN`, such a node is read as a diverged solve — the CTMM
/// scorer's finiteness guard rejects the subject, and the TTE likelihood maps it to the `1e20`
/// sentinel.
///
/// **This function exists because the two engines drifted (#1223).** They carried separate
/// copies of this loop; the dense one grew the fill for the CTMM scorer and the share's kept its
/// `NaN`, so whether a joint PK-TTE subject was scored or repelled depended on which engine
/// `try_joint_pktte_shared_solve` admitted it to. Keep it one function: a comment claiming two
/// copies are twins is what failed last time.
///
/// Keyed on the caller's **first break**, not on [`subject_integration_start`]: both engines fold
/// a terminal horizon (`max(0, …)`) into the timeline before sorting, so a timeline whose every
/// sample precedes the first dose puts the first break *below* the start, and there the node is
/// read at the `k = 0` boundary visit instead — the same seeded state by the other mechanism
/// (#1218). Strict `<` (with the shared `1e-12`), so a time *on* the first break still reads at
/// that boundary visit, post-dose.
///
/// The scan is unconditional rather than a `take_while` over a sorted slice: `chz_times` is
/// sorted-unique by the share's caller contract, but [`ode_dense_solve_states`] is `pub` and its
/// `saveat` carries no such guarantee, so an early exit would be wrong there. One shared
/// implementation is worth more than the micro-optimisation on one of the two callers.
///
/// **Preconditions**, asserted in debug because extracting this loop is what removed the local
/// context that made them self-evident — it used to sit a few lines under the allocation it was
/// paired with, and now lives thousands of lines from one of its two callers:
///
/// * `states.len() == times.len()`. `states` is indexed by an enumerate over `times`, so a short
///   `states` panics out of bounds naming neither slice, and a long one silently leaves its tail
///   unconsidered.
/// * `u.len() == states[i].len()` (i.e. `ode.n_states`) — a short `u` would write rows of the
///   wrong width for every downstream `st[chz_state]` read.
/// * `first_break` is finite. Both callers run `timeline_has_non_finite` first and return early;
///   a `NaN` here would make every comparison false and fill nothing, silently.
fn fill_prestart_states(
    times: &[f64],
    states: &mut [Vec<f64>],
    first_break: Option<f64>,
    u: &[f64],
) {
    debug_assert_eq!(
        times.len(),
        states.len(),
        "fill_prestart_states: one state row per requested time"
    );
    debug_assert!(
        first_break.is_none_or(|b| b.is_finite()),
        "fill_prestart_states: callers guard `timeline_has_non_finite` before this point"
    );
    let Some(first_break) = first_break else {
        return;
    };
    for (i, &t) in times.iter().enumerate() {
        if t < first_break - 1e-12 {
            debug_assert_eq!(
                u.len(),
                states[i].len(),
                "fill_prestart_states: seeded state is not the system's width"
            );
            states[i] = u.to_vec();
        }
    }
}

/// Tighten the ODE tolerance used for the SS **fixed-point equilibration** (#867). The value error
/// of the periodic-SS trough is the one-cycle residual amplified by `1/(1−ρ)`, and a heavily-
/// accumulating disposition has `ρ → 1`, so the per-cycle integration must be tighter than the
/// model's *prediction* tolerance for the trough to be accurate (and for the Anderson stop not to
/// false-fire on a still-drifting no-steady-state iterate). Floors `reltol`/`abstol` at
/// `1e-9`/`1e-12` — a no-op when the model already integrates tighter. Cheap: equilibration is a
/// one-time setup per evaluation, separate from the forward walk (which keeps the model tolerance).
///
/// Also raises `max_steps` for the equilibration: at the tighter `reltol` one `II` cycle needs more
/// adaptive steps, and `solve_ode` *silently returns the partial (under-integrated) state* on
/// step-budget exhaustion (`solver.rs`, "Fill any remaining saveat points with last state"). A
/// truncated one-cycle map `P` would hand the Anderson fixed point a wrong operator with no error
/// signal, so give the tightened integration enough headroom (`≥ 200_000`) that a realistic PK
/// cycle completes rather than truncating.
pub(crate) fn ss_equilibration_opts(opts: &OdeSolverOptions) -> OdeSolverOptions {
    let mut o = *opts;
    o.reltol = o.reltol.min(1e-9);
    o.abstol = o.abstol.min(1e-12);
    o.max_steps = o.max_steps.max(200_000);
    o
}

/// The row restriction that makes an accumulator-carrying system solvable as its PK
/// sub-problem: drop the `d/dt(__chz_<cmt>)` rows, solve, put them back at zero.
///
/// Two call sites need exactly this — [`periodic_ss_fixed_point_pk`] (delegating to
/// [`crate::dosing::periodic_ss_fixed_point_g`]) and [`equilibrate_ss_input_rate`]'s joint
/// branch (delegating to [`equilibrate_ss_input_rate_g`]) — and they cannot share a delegate.
/// They share this instead, because the invariant is the subtle half of #1210: an accumulator
/// row embeds as `0.0` and projects out, so the reduced one-cycle map is the PK propagator and
/// `I − M` is no longer singular.
struct ChzProjection {
    n: usize,
    pk_rows: Vec<usize>,
}

impl ChzProjection {
    fn new(chz: &[usize], n: usize) -> Self {
        Self {
            n,
            pk_rows: (0..n).filter(|i| !chz.contains(i)).collect(),
        }
    }

    /// Size of the reduced system. Zero means the spec is all accumulator and no PK.
    fn n_pk_rows(&self) -> usize {
        self.pk_rows.len()
    }

    /// Reduced vector → full-length state, accumulator rows left at zero.
    fn embed(&self, reduced: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.n];
        for (k, &row) in self.pk_rows.iter().enumerate() {
            full[row] = reduced[k];
        }
        full
    }

    /// Full-length state → reduced vector.
    fn project(&self, full: &[f64]) -> Vec<f64> {
        self.pk_rows.iter().map(|&row| full[row]).collect()
    }
}

/// Hold the injected cumulative-hazard accumulators still for one derivative evaluation.
///
/// Steady-state equilibration is a statement about the **PK** sub-system: it asks what the
/// compartments look like after an infinite past of identical dosing intervals. A
/// `d/dt(__chz_<cmt>)` row has no such state — it is a pure integrator, so it just counts up,
/// and cycling it along with the compartments is what put the run-in's own hazard into `H(0)`
/// (#1210).
///
/// **What this mask is and is not load-bearing for**, measured by mutation rather than argued:
/// it is *not* what makes `H` correct — [`restore_chz`] overwrites the accumulator row on the
/// way out unconditionally, and `[odes]` may not read `__chz_*` (rejected at parse time), so
/// the PK rows cannot see the row either. Removing the mask changes no hazard directly.
///
/// It earns its place on the path where the accumulator is *read back*: when the exact fixed
/// point declines — a nonlinear PK block — the capped pulse train runs and [`SsStopTracker`]
/// judges convergence on the **whole** state vector. An unmasked accumulator grows by
/// `hazard × II` every cycle and never settles, so the train can never early-stop: it burns all
/// [`SS_EQUILIBRATION_CYCLES`] and then reports a #867 non-convergence for a PK block that
/// converged long before. That is held by
/// `a_nonlinear_joint_model_judges_convergence_on_the_pk_rows`, which asserts the cycle count
/// and dies at the 50-cycle cap when [`equilibrate_ss_pk_state`]'s mask is dropped.
///
/// **The copy in [`ss_state_at_phase_pk`] is deliberately kept although no test can hold it.**
/// Measured: dropping it kills nothing, because the wrapper's [`restore_chz`] overwrites the
/// row on exit, and the phase advance spans at most one interval, so the row it would grow is
/// bounded by `hazard × II` rather than by 50 times that. The only channel left is the
/// integrator's error norm — a monotonically growing row would steer the PK step sizes — which
/// is a tolerance-level effect and the wrong thing to pin in a test. It stays for uniformity
/// (all three SS paths mask, so a reader who finds one unmasked does not have to re-derive
/// why) and because deleting it would leave [`restore_chz`] as the *sole* thing carrying
/// correctness on that path.
///
/// A no-op (an empty loop) for every model without an `[event_model]`.
#[inline]
fn mask_chz(chz: &[usize], dy: &mut [f64]) {
    for &slot in chz {
        // A slot outside the state vector means `chz_state_slots` disagrees with `n_states`.
        // Skipping silently would leave the row unmasked; assert in debug so an inconsistent
        // spec is loud rather than quietly reinstating the #1210 behaviour.
        debug_assert!(
            slot < dy.len(),
            "chz slot {slot} is outside a {}-state system",
            dy.len()
        );
        if slot < dy.len() {
            dy[slot] = 0.0;
        }
    }
}

/// The accumulator values an SS equilibration must hand back untouched (#1210), read off the
/// state the caller is about to overwrite. Parallel to `ode.chz_state_slots`.
///
/// The rule is *preserve*, not *zero*: for a first SS dose at the start of the record the value
/// is 0 and the two agree, but a second SS dose at `t = 48` must keep the hazard accrued over
/// `[0, 48)` rather than discard it. Zeroing there would throw away `H(48⁻)`.
///
/// Allocation-free (`Vec::new()` does not allocate) whenever the model has no accumulators.
#[inline]
fn chz_snapshot(ode: &crate::ode::OdeSpec, u: &[f64]) -> Vec<f64> {
    ode.chz_state_slots
        .iter()
        .map(|&slot| u.get(slot).copied().unwrap_or(0.0))
        .collect()
}

/// Write a [`chz_snapshot`] back into an equilibrated state. The single chokepoint for
/// #1210's rule — every early return of the equilibration passes through it, so a bail-out
/// path (`ii <= 0`, an out-of-range dose compartment, overlapping infusions) cannot silently
/// reset the accumulator either.
#[inline]
fn restore_chz(ode: &crate::ode::OdeSpec, u: &mut [f64], chz_before: &[f64]) {
    // Both fallbacks below — skipping an out-of-range slot, and substituting `0.0` for a short
    // `chz_before` — degrade into *zeroing* the accumulator, which is precisely the behaviour
    // #1210 rejects and the one outcome no first-SS-dose test can tell from correct. Assert in
    // debug so a spec/snapshot mismatch fails loudly instead of reinstating the bug.
    debug_assert_eq!(
        chz_before.len(),
        ode.chz_state_slots.len(),
        "chz snapshot length does not match the spec's accumulator slots"
    );
    for (k, &slot) in ode.chz_state_slots.iter().enumerate() {
        debug_assert!(
            slot < u.len(),
            "chz slot {slot} is outside a {}-state system",
            u.len()
        );
        if slot < u.len() {
            u[slot] = chz_before.get(k).copied().unwrap_or(0.0);
        }
    }
}

/// [`crate::dosing::periodic_ss_fixed_point_g`] restricted to the PK sub-system.
///
/// The exact solve inverts `I − M` for the one-cycle propagator `M`. Under [`mask_chz`] an
/// accumulator row's one-cycle map is the *identity*, so that row of `I − M` is all zeros and
/// the system is singular — for **every** joint PK-TTE model, whatever its PK block looks
/// like. Masking alone would therefore leave #1210's fixtures on the capped 50-cycle pulse
/// train (the accumulator never stops growing, so `SsStopTracker` never sees convergence), and
/// liable to the spurious #867 non-convergence warning that follows from capping. Whether that
/// warning actually fires is a further question — it rides `note_ss_nonconvergence_if_capped`'s
/// geometric-tail test, so a capped run does not always produce one.
///
/// Projecting the accumulator rows out restores the PK propagator, and a linear PK block gets
/// its handful of solves back. The returned vector is full-length with the accumulator rows
/// left at zero; the caller's [`restore_chz`] fills them.
fn periodic_ss_fixed_point_pk<FUnf, FFor>(
    chz: &[usize],
    n: usize,
    ii: f64,
    reltol: f64,
    abstol: f64,
    advance_unforced: FUnf,
    advance_forced: FFor,
) -> Option<Vec<f64>>
where
    FUnf: Fn(&[f64]) -> Option<Vec<f64>>,
    FFor: Fn(&[f64]) -> Option<Vec<f64>>,
{
    if chz.is_empty() {
        return crate::dosing::periodic_ss_fixed_point_g::<f64, _, _>(
            n,
            ii,
            reltol,
            abstol,
            advance_unforced,
            advance_forced,
        );
    }
    let proj = ChzProjection::new(chz, n);
    if proj.n_pk_rows() == 0 {
        return None;
    }
    let u_red = crate::dosing::periodic_ss_fixed_point_g::<f64, _, _>(
        proj.n_pk_rows(),
        ii,
        reltol,
        abstol,
        |r| advance_unforced(&proj.embed(r)).map(|f| proj.project(&f)),
        |r| advance_forced(&proj.embed(r)).map(|f| proj.project(&f)),
    )?;
    Some(proj.embed(&u_red))
}

/// Extended parameters for **one window** of a steady-state run-in (#1139).
///
/// The compiled `[odes]` right-hand side reads `TAD` out of `params[MAX_PK_PARAMS + 1]`
/// (`parser/model_parser.rs`, the model-time closure) as `t − anchor`, and `TAFD` out of
/// `params[MAX_PK_PARAMS]`. Every SS-equilibration call site hands `solve_ode` a bare
/// `PkParams::values`, which is exactly `MAX_PK_PARAMS` long, so both `.get()` calls
/// returned `None`, the RHS injected `NaN`, and `0.0 * NaN = NaN` poisoned the whole
/// run-in — merely *mentioning* `TAD` turned an otherwise ordinary `SS=1` fit into a
/// non-finite objective.
///
/// `pulse_at` is the **local-clock** time of the pulse this window measures `TAD` from,
/// in the same units the window's own `solve_ode` span uses. The run-in does not run on
/// the subject's clock — it expands a periodic train on a clock private to each window —
/// so this is not a `tad_anchor_for` question and the walk's own `ext_params` must not be
/// threaded in here. Three shapes, all of them live:
///
/// * a window opening **at** the pulse → `0.0`. That is `(0, II)` for the exact solve's
///   propagator probes and its forced bolus cycle, `(0, II)` again for the capped train's
///   bolus cycle, `(0, T_inf)` for an infusion's **active** window on both of those
///   branches, and `(0, II)` for the input-rate path's one-cycle advance;
/// * the **quiet** window an infusion cycle re-opens at local `0` after `T_inf` of active
///   rate → `−T_inf`, so `TAD` continues at `T_inf … II` rather than restarting at zero;
/// * the monotone `(m·II, (m+1)·II)` segments of the capped input-rate pulse train, whose
///   pulses sit at local `0, II, 2·II, …` → `m·II`, taken from [`tad_anchor_for`] over
///   that train's own dose list rather than re-spelled. A flat `0.0` there reads
///   `TAD = m·II + τ`, wrong by up to `SS_EQUILIBRATION_CYCLES − 1 = 49` whole intervals.
///
/// **`TAFD` is deliberately left `NaN`**, which reproduces today's answer bit-for-bit for a
/// `TAFD`-reading model. Anchoring it at the run-in's own origin would hand back a finite,
/// plausible number for a quantity that has no periodic steady state at all — measured, its
/// explicit train diverges 0.294 per doubling — i.e. it would silently redefine `TAFD` as
/// `TAD` inside the run-in, which is the very defect class this function removes. `TAFD`,
/// `T` and `TIME` under `SS=1` are #1139's other half and are handled separately.
///
/// [`ss_state_at_phase_pk`]'s three windows call this too since #1126 — the phase advance
/// is a run-in window like any other, opening at its cycle's pulse (`0.0`) with the same
/// `−T_inf` quiet window for an infusion. It stayed on the bare slice for one release
/// because anchoring it *alone* converts a loud `NaN` into a number 2.7 % wrong on every
/// observation; see the note at the top of that function for why the other half is the
/// walk's anchor and not this one.
#[inline]
pub(crate) fn ss_run_in_params(
    pk_params_flat: &[f64],
    pulse_at: f64,
) -> [f64; crate::types::MAX_PK_PARAMS + 2] {
    // `seed_ext_params` copies `min(len, MAX_PK_PARAMS)` and leaves the rest `NaN`, so a
    // short slice would turn what used to be an index panic inside the RHS into a silent
    // `NaN` `CL` — the loud-to-silent conversion this whole change exists to reverse.
    // Every production caller passes a `PkParams::values`, which is exactly the right
    // length; this keeps that contract loud if one ever does not.
    debug_assert!(
        pk_params_flat.len() >= crate::types::MAX_PK_PARAMS,
        "SS run-in handed {} params, needs at least {}",
        pk_params_flat.len(),
        crate::types::MAX_PK_PARAMS
    );
    let mut ext = seed_ext_params(pk_params_flat, f64::NAN);
    ext[crate::types::MAX_PK_PARAMS + 1] = pulse_at;
    ext
}

/// Periodic steady-state trough for an `SS=1` dose into a built-in absorption input-rate
/// compartment (#719; nonlinear solve #867).
///
/// The system is `du/dt = f(u) + R_in(t)`, where `R_in` is the periodic absorption forcing
/// (period `II`, the superposed pulse train — see [`add_prepared_input_rate_forcing`]'s SS branch).
/// The steady-state trough is the fixed point `u = P(u)` of the one-cycle Poincaré map
/// `P(u₀)` = "integrate one `II` cycle under `R_in` from `u₀`".
///
/// For a **linear** disposition that fixed point is a closed form —
/// `u_ss = (I − M)⁻¹·b`, `M = e^{A·II}`, `b` one forced cycle from a zero state — costing
/// `n_states + 3` ODE solves ([`crate::dosing::periodic_ss_fixed_point_g`]). For a **nonlinear**
/// disposition (its self-check declines) the same fixed point is found by an Anderson-accelerated
/// iteration on the identical `P` ([`anderson_ss_fixed_point_g`]) — a bounded handful of one-cycle
/// solves, unlike the plain pulse train's `O(1/(1−ρ))`. Delegates both to
/// [`equilibrate_ss_input_rate_g`].
///
/// Returns `None` — so the caller falls back to the capped pulse train + #867 warning — only when
/// *neither* converges: a singular `I − M`, a non-finite intermediate, or `ρ ≥ 1` (mean input ≥
/// maximum elimination, so no periodic steady state exists).
fn equilibrate_ss_input_rate(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    f_bio: f64,
    opts: &OdeSolverOptions,
    prepared: &PreparedForcings,
) -> Option<Vec<f64>> {
    let n = ode.n_states;
    let ii = dose.ii;
    if !(ii > 0.0) || n == 0 {
        return None;
    }

    // Forced one-cycle RHS: the disposition plus the periodic absorption `R_in` of a single
    // local SS pulse at t = 0 (its SS branch superposes the prior-pulse tails). Reused for `b`,
    // the fixed-point verification, and the Anderson iteration's `P`. `prepared` is built once by
    // the caller (`equilibrate_ss_state`) and passed in so the fallback pulse-train iteration
    // doesn't redo the same prep on a `None` return.
    let local_ss = [DoseEvent::new(0.0, dose.amt, dose.cmt_raw(), 0.0, true, ii)];
    let local_f_bio = [f_bio];
    let no_lag: [f64; 0] = [];
    let no_zero: [(usize, f64); 0] = [];
    let forced_rhs = wrap_rhs_with_forcings(
        ode,
        &local_ss,
        &no_lag,
        &local_f_bio,
        f64::NEG_INFINITY,
        // The one cycle `[0, II]` is a single segment starting at the local pulse's own
        // record, so the `SS=1` gate is a no-op here: the cutoff is `local_ss` itself.
        0.0,
        prepared,
        InfusionInput::Spanning(Vec::new()),
        &no_zero,
    );

    // Advance a state one cycle `[0, II]` under `rhs`. The equilibration integrates at a tightened
    // tolerance (`ss_equilibration_opts`) so the fixed-point trough is accurate even when `ρ → 1`
    // amplifies the per-cycle solver noise — the forward walk keeps the model tolerance.
    let eq_opts = ss_equilibration_opts(opts);
    // The cycle's pulse sits at local `0`, so `TAD = t` across the whole window (#1139).
    let ext = ss_run_in_params(pk_params_flat, 0.0);
    let advance = |rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64]| -> Option<Vec<f64>> {
        solve_ode(rhs, u0, (0.0, ii), &ext, &[ii], &eq_opts)
            .last()
            .map(|p| p.u.clone())
    };
    let chz = &ode.chz_state_slots[..];
    if chz.is_empty() {
        return equilibrate_ss_input_rate_g::<f64, _, _>(
            n,
            ii,
            eq_opts.reltol,
            eq_opts.abstol,
            |u0| advance(ode.rhs.as_ref(), u0),
            |u0| advance(&forced_rhs, u0),
        );
    }

    // Joint PK-TTE (#1210). Two things have to happen for the run-in, and neither is enough
    // alone: the accumulator's derivative is held at zero (`mask_chz`) so the equilibration
    // cannot bank the run-in's hazard, and its row is projected out of the one-cycle map so
    // `I − M` is the PK propagator rather than a singular matrix. Both the exact solve and the
    // Anderson fallback inside `equilibrate_ss_input_rate_g` then work on the PK sub-system.
    // The returned vector is full-length with the accumulator rows at zero; the caller
    // restores their record values.
    let masked_unforced = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
        (ode.rhs)(y, p, t, dy);
        mask_chz(chz, dy);
    };
    let masked_forced = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
        forced_rhs(y, p, t, dy);
        mask_chz(chz, dy);
    };
    // Mirror the `n == 0` guard the non-joint path gets above: a system that is *all*
    // accumulator has no PK sub-problem, and an empty reduced solve would hand back an
    // all-zero state for the caller to mistake for an equilibrated trough.
    let proj = ChzProjection::new(chz, n);
    if proj.n_pk_rows() == 0 {
        return None;
    }
    let u_red = equilibrate_ss_input_rate_g::<f64, _, _>(
        proj.n_pk_rows(),
        ii,
        eq_opts.reltol,
        eq_opts.abstol,
        |r| advance(&masked_unforced, &proj.embed(r)).map(|f| proj.project(&f)),
        |r| advance(&masked_forced, &proj.embed(r)).map(|f| proj.project(&f)),
    )?;
    Some(proj.embed(&u_red))
}

/// Bounded iteration budget for the Anderson-accelerated nonlinear periodic-SS solve (#867).
/// Anderson converges geometrically-fast (in the number of *distinct decay modes*, not in
/// `1/(1−ρ)`) even as the per-cycle carryover ratio `ρ → 1`, so a few dozen one-cycle solves
/// suffice for any `ρ < 1` that admits a steady state. The cap is only a backstop for `ρ ≥ 1`
/// (mean input ≥ maximum elimination → no periodic steady state), which falls through to the
/// capped pulse train and the #867 non-convergence warning.
const SS_ANDERSON_MAX_ITERS: usize = 80;

/// Anderson-acceleration depth: how many past residual differences are mixed. A handful covers the
/// low-dimensional PK compartment count (typically 1–3 states); more would only add near-parallel
/// columns the Tikhonov damping discards.
const SS_ANDERSON_WINDOW: usize = 5;

/// One dual Newton correction of a value-converged Anderson iterate that snaps the **derivative**
/// jets to the exact implicit-function derivative of the fixed point (#867).
///
/// Anderson's accelerated iterate has the converged *value* but a *derivative* that is a by-product
/// of the (possibly large) extrapolation coefficients — for a slowly-contracting map (`ρ → 1`) that
/// derivative can be several percent off, and a plain fixed-point cleanup would only contract it at
/// the map's own rate `ρ`, so it does not scale. Instead take one Newton step of `F(u) = P(u) − u`
/// from the converged value `u*`:
///
/// ```text
///   u_new = u* + (I − J_P)⁻¹ · (P(u*) − u*),    J_P = ∂P/∂u |_{u*}.
/// ```
///
/// Carried over the dual `T`, the artifact derivative `du*` of `u*` **cancels** in the step
/// (`du_new = du* − (I − J_P)⁻¹[(I − J_P)·du* − P_θ] = (I − J_P)⁻¹ P_θ`), leaving exactly the
/// implicit derivative `∂u*/∂(θ,η)` — the same object #835's linear closed form obtains from its
/// `(I − M)⁻¹` dual solve, here linearised at the nonlinear fixed point. `J_P` is a finite-
/// difference state-Jacobian (`n + 1` one-cycle solves), so the cost is `O(n)` regardless of how
/// slowly the model accumulates. Returns `None` on a singular `I − J_P` or a failed solve, so the
/// caller keeps the value-only result.
fn newton_ss_derivative_correction_g<T, FFor>(
    u_star: &[T],
    n: usize,
    advance_forced: &FFor,
) -> Option<Vec<T>>
where
    T: crate::sens::num::PkNum,
    FFor: Fn(&[T]) -> Option<Vec<T>>,
{
    let g0 = advance_forced(u_star)?; // P(u*)
    let residual: Vec<T> = (0..n).map(|i| g0[i] - u_star[i]).collect();
    // FD state-Jacobian `J_P` at `u*`. The perturbation is a *real* bump on the state value (it
    // carries no jet), so each column's dual parts are `∂J_P/∂(θ,η)` — exactly what the dual solve
    // below needs to propagate the 2nd-order derivative. Relative step, floored for a near-zero
    // trough; one-sided (reusing `g0`) keeps it to `n + 1` solves.
    let scale = u_star.iter().fold(1e-8_f64, |m, x| m.max(x.val().abs()));
    let eps = 1e-5 * scale;
    let mut i_minus_j = vec![T::from_f64(0.0); n * n];
    for i in 0..n {
        let mut up = u_star.to_vec();
        up[i] = up[i] + T::from_f64(eps);
        let gi = advance_forced(&up)?;
        for r in 0..n {
            let j_ri = (gi[r] - g0[r]) / T::from_f64(eps);
            let delta_ri = T::from_f64(if r == i { 1.0 } else { 0.0 });
            i_minus_j[r * n + i] = delta_ri - j_ri;
        }
    }
    let step = crate::sens::linsolve::solve_linear_system_g::<T>(&i_minus_j, &residual, n)?;
    let u_new: Vec<T> = (0..n).map(|i| u_star[i] + step[i]).collect();
    if u_new.iter().any(|x| !x.val().is_finite()) {
        return None;
    }
    Some(u_new)
}

/// Solve the nonlinear periodic steady state `u* = P(u*)` for an SS-into-absorption dose on a
/// **nonlinear** disposition (#867), where `P = advance_forced` integrates one `II` cycle under the
/// periodic absorption forcing `R_in` — the *same* stationary Poincaré map the linear closed form
/// `u_ss = (I − M)⁻¹·b` inverts exactly, here solved by [Anderson acceleration] for a disposition
/// whose one-cycle map is not affine. This replaces the plain pulse-train iteration, which is a
/// geometric contraction costing `O(1/(1−ρ))` cycles and so silently under-converges when a
/// saturable disposition accumulates heavily (`ρ → 1`); Anderson reaches the same fixed point in a
/// bounded handful of one-cycle solves, cheap enough for the fit hot path.
///
/// Generic over `T`: run over a dual it carries `∂u*/∂(θ,η)` (and the 2nd order) through the same
/// recursion. The mixing coefficients `γ` are chosen to annihilate the **value** residual and then
/// applied to the whole `T` state, so the converged derivative is the implicit-function derivative
/// of the converged value — the analytic gradient, with no hand-assembled `dM`/`db`, exactly as the
/// linear fixed point obtains it from the dual linear solve (verified against FD in
/// `ss_input_rate_nonlinear_dual_gradient_matches_fd`).
///
/// Returns `None` — caller falls back to the capped pulse train + warning — when it fails to
/// contract within [`SS_ANDERSON_MAX_ITERS`] (a non-finite iterate, or `ρ ≥ 1`: no periodic SS).
///
/// [Anderson acceleration]: https://doi.org/10.1137/10078356X
fn anderson_ss_fixed_point_g<T, FFor>(
    n: usize,
    ii: f64,
    reltol: f64,
    abstol: f64,
    advance_forced: &FFor,
) -> Option<Vec<T>>
where
    T: crate::sens::num::PkNum,
    FFor: Fn(&[T]) -> Option<Vec<T>>,
{
    if !(ii > 0.0) || n == 0 {
        return None;
    }
    // Each `P` evaluation carries O(reltol) adaptive-quadrature noise, so the value part cannot be
    // driven below a small multiple of `reltol`; target that floor (with an abstol cushion).
    let conv_tol = (8.0 * reltol).max(1e-12);
    let zero = vec![T::from_f64(0.0); n];
    // Seed with one forced cycle from a zero state (the linear `b` — the single-period response
    // ignoring accumulation): finite, cheap, and in the basin of any disposition that admits a
    // steady state.
    let mut u = advance_forced(&zero)?;
    // Divergence ceiling: a genuine periodic SS is at most `≈ 1/(1−ρ)` times this single-period
    // response, so any iterate a huge factor beyond it means the map is not contracting (no SS) —
    // and, crucially, Anderson can extrapolate a *divergent* map to a spurious near-stationary
    // point (a huge value where a saturated RHS barely moves), whose small *relative* residual
    // would otherwise false-trip the convergence test. Bail so the caller falls to the capped
    // pulse train + #867 warning instead of returning garbage (e.g. a huge negative "trough").
    let seed_mag = u.iter().fold(1.0_f64, |m, x| m.max(x.val().abs()));
    // Tie the ceiling to the accuracy at which a genuine fixed point can still be
    // distinguished from solver noise. The previous fixed `1e8` factor contradicted
    // the seed-scale bound below: at the default reltol it admitted an over-capacity
    // Anderson extrapolate more than 200,000 seeds away, where integration error hid
    // the positive per-cycle surplus and produced a false steady state (#867).
    let distinguishable_growth = 1.0 / reltol.sqrt().clamp(1e-8, 1.0);
    let diverged_ceiling = distinguishable_growth * seed_mag;
    // Seed-scale residual bound (#867). The `conv_tol·max_mag` test above is *relative to the
    // current iterate*, so once Anderson inflates a divergent (no-SS, over-capacity) map to a huge
    // value the real per-cycle surplus `Δ = (mean input − max elimination)·II` — an `O(input)`
    // quantity, NOT solver noise — hides beneath it and false-trips convergence (returning a huge
    // or even negative "trough"). A *genuine* fixed point's residual is only solver noise
    // (`≈ reltol·magnitude`), so it also clears a bound anchored to the SEED: `√reltol` leaves ample
    // headroom for a legitimately huge deep-accumulation SS (up to `≈ seed/√reltol`) while an
    // `O(input)` surplus fails it. Combined with a non-negativity check (compartment amounts cannot
    // be negative) at the acceptance point below, this rejects the spurious inflation.
    let seed_residual_bound = reltol.sqrt().max(1e-7) * seed_mag + abstol;
    let mut u_hist: Vec<Vec<T>> = Vec::with_capacity(SS_ANDERSON_WINDOW + 1);
    let mut g_hist: Vec<Vec<T>> = Vec::with_capacity(SS_ANDERSON_WINDOW + 1);
    // Previous iterate, to confirm the sequence has *settled* (see the step test below).
    let mut u_prev: Option<Vec<T>> = None;
    for iter in 0..SS_ANDERSON_MAX_ITERS {
        let g = advance_forced(&u)?;
        if g.iter()
            .any(|x| !x.val().is_finite() || x.val().abs() > diverged_ceiling)
        {
            return None;
        }
        // Relative-L∞ residual of the one-cycle map on the value parts.
        let (mut max_res, mut max_mag) = (0.0_f64, 0.0_f64);
        for (gi, ui) in g.iter().zip(&u) {
            max_res = max_res.max((gi.val() - ui.val()).abs());
            max_mag = max_mag.max(gi.val().abs());
        }
        let tol = conv_tol * max_mag + abstol;
        // Convergence needs BOTH a small residual (`u` is a fixed point of `P`) AND a settled
        // iterate (`u` barely moved since the previous step). The step test is what stops a
        // *divergent* map from false-converging: Anderson can hurl the iterate to a huge value
        // where a saturated RHS is nearly stationary — its residual is small *relative* to that
        // inflated magnitude, but it was reached by an enormous jump, so the step is not small.
        let settled_step = match &u_prev {
            Some(p) => {
                u.iter()
                    .zip(p)
                    .fold(0.0_f64, |m, (a, b)| m.max((a.val() - b.val()).abs()))
                    <= tol
            }
            None => false, // the seed is not, on its own, evidence of convergence
        };
        if max_res <= tol && settled_step {
            // The magnitude-relative test flagged a candidate fixed point — but confirm it is a
            // *genuine* periodic SS, not a spurious no-steady-state inflation (#867): the residual
            // must also be small on the seed scale, and a compartment amount cannot be negative. A
            // candidate that satisfies the relative test yet fails either is an over-capacity map
            // Anderson extrapolated to garbage; there is no periodic SS, so decline (→ caller's
            // capped pulse train + #867 warning) rather than return it.
            let nonneg = g.iter().all(|x| x.val() >= -(tol + abstol));
            if max_res > seed_residual_bound || !nonneg {
                return None;
            }
            // Value converged. One dual Newton step snaps the derivative jets to the exact
            // implicit-function derivative (the Anderson iterate's derivative is an extrapolation
            // by-product). On a singular `I − J_P` — the `ρ ≈ 1` degenerate boundary, where the
            // value would barely have converged anyway — fall back to the value-converged image `g`
            // (which then carries the artifact derivative; a `ρ ≈ 1` corner not reached by any
            // genuinely-contracting model).
            record_ss_equilibration_cycles(iter + 1);
            crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::Anderson);
            return newton_ss_derivative_correction_g(&g, n, advance_forced).or(Some(g));
        }
        u_hist.push(u.clone());
        g_hist.push(g.clone());
        if u_hist.len() > SS_ANDERSON_WINDOW + 1 {
            u_hist.remove(0);
            g_hist.remove(0);
        }
        u_prev = Some(u.clone());
        u = anderson_combine::<T>(&u_hist, &g_hist, n);
    }
    None
}

/// One Anderson-acceleration mixing step (β = 1). Given the retained iterate/image history
/// (`u_hist[i]`, `g_hist[i] = P(u_hist[i])`), form the residual differences `ΔF` on the **value**
/// parts, solve the small least-squares `γ = argmin‖f_last − ΔF·γ‖` via the Tikhonov-damped normal
/// equations ([`solve_linear_system_g`](crate::sens::linsolve::solve_linear_system_g) over `f64`),
/// and return `g_last − ΔG·γ` in `T` arithmetic so a dual state's derivative rides the same
/// combination. Reduces to a plain Picard step (`g_last`) with a single history point or a singular
/// least-squares.
fn anderson_combine<T: crate::sens::num::PkNum>(
    u_hist: &[Vec<T>],
    g_hist: &[Vec<T>],
    n: usize,
) -> Vec<T> {
    let k = u_hist.len();
    let g_last = &g_hist[k - 1];
    if k < 2 {
        return g_last.clone(); // Picard
    }
    let m = k - 1; // difference-column count
                   // Residual value parts f_i = g_i − u_i, then columns ΔF_j = f_{j+1} − f_j (n × m).
    let f: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..n)
                .map(|r| g_hist[i][r].val() - u_hist[i][r].val())
                .collect()
        })
        .collect();
    let df = |r: usize, j: usize| f[j + 1][r] - f[j][r];
    // Normal equations A = ΔFᵀΔF (m × m), rhs = ΔFᵀ f_last.
    let mut a = vec![0.0_f64; m * m];
    let mut rhs = vec![0.0_f64; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for r in 0..n {
                s += df(r, i) * df(r, j);
            }
            a[i * m + j] = s;
        }
        let mut s = 0.0;
        for r in 0..n {
            s += df(r, i) * f[k - 1][r];
        }
        rhs[i] = s;
    }
    // Tikhonov floor stabilises a rank-deficient history (near-parallel difference columns).
    let diag_max = (0..m).fold(0.0_f64, |mx, i| mx.max(a[i * m + i]));
    let lambda = 1e-12 * diag_max.max(1.0);
    for i in 0..m {
        a[i * m + i] += lambda;
    }
    let gamma = match crate::sens::linsolve::solve_linear_system_g::<f64>(&a, &rhs, m) {
        Some(g) => g,
        None => return g_last.clone(), // singular → Picard
    };
    // u_next = g_last − Σ_j γ_j (g_{j+1} − g_j)   [T arithmetic threads the dual jets].
    let mut u_next = g_last.clone();
    for j in 0..m {
        let gj = T::from_f64(gamma[j]);
        for r in 0..n {
            u_next[r] = u_next[r] - gj * (g_hist[j + 1][r] - g_hist[j][r]);
        }
    }
    u_next
}

/// Periodic steady-state trough for an SS-into-absorption dose, generic over `T` (#867). Tries the
/// **linear** closed form [`crate::dosing::periodic_ss_fixed_point_g`] first (exact, one linear
/// solve); on a nonlinear disposition — where its self-check declines — falls to the
/// [Anderson-accelerated][`anderson_ss_fixed_point_g`] solve of the same `u = P(u)` fixed point.
/// Both share the injected one-cycle propagators, so a caller assembles its solver/forcings once.
/// Returns `None` only when *neither* converges (`ρ ≥ 1`: no periodic steady state), leaving the
/// caller's capped pulse-train fallback to run and the #867 warning to fire.
pub(crate) fn equilibrate_ss_input_rate_g<T, FUnf, FFor>(
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
    // A top-level entry in its own right: `sens::ode_provider` calls this directly for the dual
    // SS equilibration, not only through `equilibrate_ss_pk_state`. Neither arm below ever
    // reaches `note_ss_nonconvergence_if_capped`, so clear the warning observation here too or
    // that caller reads an earlier capped run's `true` (#1289, PR #1392 review).
    crate::dosing::record_ss_nonconvergence_warned(false);
    // `&F` still implements `Fn` when `F: Fn`, so borrowing lets the linear attempt and the
    // Anderson fallback share the same two closures without moving them.
    if let Some(u_ss) = crate::dosing::periodic_ss_fixed_point_g::<T, _, _>(
        n,
        ii,
        reltol,
        abstol,
        &advance_unforced,
        &advance_forced,
    ) {
        record_ss_equilibration_cycles(1);
        crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::InputRateExact);
        return Some(u_ss);
    }
    anderson_ss_fixed_point_g::<T, _>(n, ii, reltol, abstol, &advance_forced)
}

/// Pre-equilibrate the ODE state to its steady-state value for an SS=1
/// dose with interval `dose.ii`. NONMEM SS=1 semantics: at the time of
/// the SS dose, the compartments are loaded with the steady-state
/// amounts from an infinite-past pulse train.
///
/// For a **linear** disposition the periodic steady state is the exact affine
/// fixed point `u_ss = (I − M)⁻¹·b` — the same closed form the analytical walk
/// (#908) and the input-rate branch (#835) use — solved via
/// [`periodic_ss_fixed_point_g`](crate::dosing::periodic_ss_fixed_point_g) in a
/// handful of one-cycle integrations (#914). A **nonlinear** RHS
/// (Michaelis–Menten, …) fails that solve's linearity self-check and falls back
/// to numerically expanding the train: starting from a zero state, simulate
/// [`SS_EQUILIBRATION_CYCLES`] cycles of `(apply dose; integrate for II)`, with a
/// #867 non-convergence warning if the cap is hit without converging.
/// Either way the returned state equals the "just-before-next-pulse" SS state;
/// the caller then applies the SS dose itself through the normal flow,
/// recovering the at-pulse SS amount.
///
/// `dose.ii > 0` and `dose.cmt` valid are required (callers guard this).
/// For SS infusions (`is_real_infusion(dose)`), each cycle integrates a
/// `dose.duration`-long active-infusion window followed by a
/// `(II - duration)`-long quiet window. The SS form requires
/// `dose.duration <= dose.ii` (non-overlapping); overlapping pulses
/// would need a different equilibration scheme and are out of scope —
/// the existing api.rs warning fires for those.
///
/// `chz_before` carries the injected cumulative-hazard accumulators' values *at the record
/// this dose sits on* (from [`chz_snapshot`] of the caller's current state). Those rows are
/// held still through the run-in and handed back unchanged: an equilibration is a statement
/// about the PK compartments, and the hazard clock runs on record time, not on the infinite
/// past the run-in stands in for (#1210).
fn equilibrate_ss_state(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    opts: &OdeSolverOptions,
    chz_before: &[f64],
) -> Vec<f64> {
    let mut u = equilibrate_ss_pk_state(ode, pk_params_flat, dose, opts);
    restore_chz(ode, &mut u, chz_before);
    u
}

/// The PK-only body of [`equilibrate_ss_state`]. Every integration here runs under
/// [`mask_chz`], so the accumulator rows do not move; they come back at zero and the
/// caller-facing wrapper writes the record values over them. Kept separate so that single
/// restore is the one exit — including from the three early bail-outs below.
fn equilibrate_ss_pk_state(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    // Clear the branch tag up front so the early returns below leave `None` rather than
    // the previous call's value — see `crate::dosing::SsBranch`. Every completing path
    // overwrites it; every bail-out is then honestly reported as "no branch ran".
    crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::None);
    // Same reasoning for the warning observation (#1289): the exact affine fixed point and the
    // input-rate closed form below return without ever reaching `note_ss_nonconvergence_if_capped`,
    // so without this an earlier capped run's `true` would be reported as *this* call's answer.
    crate::dosing::record_ss_nonconvergence_warned(false);
    let n = ode.n_states;
    let chz = &ode.chz_state_slots[..];
    // The model's own RHS with the accumulator derivatives held at zero. Everything the
    // equilibration integrates goes through this instead of `ode.rhs` — the exact solve's
    // propagator probes, the infusion windows, and the capped pulse-train fallback alike.
    let base_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
        (ode.rhs)(y, p, t, dy);
        mask_chz(chz, dy);
    };
    let mut u = vec![0.0; n];

    if dose.ii <= 0.0 {
        return u;
    }
    // `CMT=0` equilibrates compartment 1 — NONMEM's default dose compartment —
    // like every other dose site (#899). This used to bail out here and return
    // an unequilibrated all-zero state, so an `SS=1` dose written `CMT=0`
    // produced the *single-dose* curve on the ODE engine while the analytical
    // engine (fixed in #375) produced the accumulated steady state. That was the
    // cross-engine disagreement recorded in `CHANGELOG.md`; this closes it.
    let cmt_idx = dose.cmt_idx();
    if cmt_idx >= n {
        return u;
    }

    // Bioavailability F scales the amount that actually enters the dosing
    // compartment — NONMEM's convention (F·AMT for a bolus, F·RATE for an
    // infusion). Resolved per dose compartment (`Fn`; issue #369), falling back
    // to the bare `PK_IDX_F` slot. Matches the analytical path
    // (`equilibrate_ss_state_event_driven`).
    let f_bio = ode.dose_attr_map.f_bio(dose.cmt_raw(), pk_params_flat);

    let is_inf = is_real_infusion(dose);
    // Mode-aware bioavailability (#419): a rate-defined infusion keeps its rate
    // and `F` scales the duration; a duration-defined infusion (`RATE=-2`) keeps
    // its duration and `F` scales the rate. Total input is `F·AMT` either way.
    let (inf_rate, t_inf) = dose.bioavailable_infusion(f_bio);
    if is_inf && t_inf > dose.ii {
        // Overlapping infusions; no closed-form / simple equilibration.
        return u;
    }

    // Steady-state into a built-in absorption input-rate compartment (#719): the dose does
    // not enter as an instantaneous bolus — it drives the compartment through the absorption
    // kernel `R_in(tad)` (transit/igd/weibull/first_order). Equilibrate by integrating an
    // explicit periodic pulse train (a non-SS pulse per cycle at local times 0, II, 2II, …)
    // through the *same* input-rate forcing the forward walk uses. Because the whole train
    // stays present and `R_in` is re-evaluated by absolute age every RHS call, each pulse's
    // absorption keeps contributing across cycle boundaries — an absorption tail longer than
    // II is not truncated (unlike the bolus loop, which would lose it). The state at the final
    // pre-pulse trough is the SS carryover (disposition + any depot amount); the current
    // pulse and the prior pulses' still-arriving tails are then superposed by the forward
    // `add_prepared_input_rate_forcing` periodic sum, which is disjoint from this trough. SS
    // *infusion* into an absorption compartment is out of scope here (gap 2, #719) — this
    // branch is bolus-record SS only.
    if !is_inf && input_rate_consumes_cmt(ode, dose.cmt_raw()) {
        // Periodic-SS solve for the fixed point `u = P(u)` of the one-cycle map: a *linear*
        // disposition has the closed form `u_ss = (I − M)⁻¹ b` (a handful of solves); a *nonlinear*
        // one is found by an Anderson-accelerated iteration on the same `P`, also in a bounded
        // handful of one-cycle solves (`equilibrate_ss_input_rate`, records its own cycle count).
        // Only `ρ ≥ 1` (no periodic steady state) returns `None` and falls through to the capped
        // pulse-train iteration + #867 warning below. `prepared` is built once and reused by both
        // the solve and the fallback.
        let prepared = prepare_input_rates(ode, pk_params_flat);
        if let Some(u_ss) =
            equilibrate_ss_input_rate(ode, pk_params_flat, dose, f_bio, opts, &prepared)
        {
            return u_ss;
        }
        let n_pulses = SS_EQUILIBRATION_CYCLES;
        let local_doses: Vec<DoseEvent> = (0..n_pulses)
            .map(|m| {
                DoseEvent::new(
                    m as f64 * dose.ii,
                    dose.amt,
                    dose.cmt_raw(),
                    0.0,
                    false,
                    0.0,
                )
            })
            .collect();
        let local_f_bio = vec![f_bio; n_pulses];
        // One empty slice serves both callees: `wrap_rhs_with_forcings` has always read its
        // lag slice with `.get(k)`, and since #1263's review `tad_anchor_for` does too
        // (`lag_at`), so `&[]` means "no lag on any of these synthetic pulses" to each of
        // them. This used to need a second, zero-filled `Vec<f64>` of length `n_pulses`
        // purely to satisfy an index that would otherwise panic; `ss_monotone_run_in_anchor_is_the_same_with_an_empty_lag_slice`
        // pins that the two spellings agree, so the allocation cannot creep back in on the
        // strength of a misremembered contract.
        let no_lag: [f64; 0] = [];
        let no_zero_order: [(usize, f64); 0] = [];
        let wrapped_raw = wrap_rhs_with_forcings(
            ode,
            &local_doses,
            &no_lag,
            &local_f_bio,
            f64::NEG_INFINITY,
            // Every synthetic pulse is non-SS, so there is no `SS=1` record to reset at and
            // the segment start is immaterial; one closure serves every cycle.
            f64::NEG_INFINITY,
            &prepared,
            InfusionInput::Spanning(Vec::new()),
            &no_zero_order,
        );
        // The forcing wrapper builds on `ode.rhs`, so the mask goes on the outside of it.
        let wrapped = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            wrapped_raw(y, p, t, dy);
            mask_chz(chz, dy);
        };
        let mut tracker = SsStopTracker::default();
        let mut cycles_run = 0usize;
        let mut early_stopped = false;
        for m in 0..n_pulses {
            let seg_start = m as f64 * dose.ii;
            let seg_end = seg_start + dose.ii;
            // Unlike every other run-in window this train runs on a *monotone* clock
            // `0 … n_pulses·II`, so the `TAD` anchor advances with it: the pulse governing
            // segment `m` is `local_doses[m]` at `m·II`. Taken from the shared rule over
            // this train's own dose list rather than re-spelled — a flat `0.0` would read
            // `TAD = m·II + τ`, wrong by up to 49 whole dosing intervals (#1139).
            let ext = ss_run_in_params(
                pk_params_flat,
                tad_anchor_for(&local_doses, &no_lag, seg_start),
            );
            let sol = solve_ode(&wrapped, &u, (seg_start, seg_end), &ext, &[seg_end], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cycles_run = m + 1;
            if tracker.should_stop(m, &u) {
                early_stopped = true;
                break;
            }
        }
        record_ss_equilibration_cycles(cycles_run);
        crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::InputRateTrain);
        // If the pulse train hit the cycle cap without converging, the returned trough may be
        // materially below the true periodic steady state — surface a warning instead of silently
        // under-reporting it (#867). Only a *nonlinear* disposition reaches this fallback (the
        // linear closed form above returns early), so this is exactly the saturable
        // heavy-accumulation case; a fast-contracting model early-stops and is left alone.
        let (incr_prev, incr_last, incr_mag) = tracker.recent_increments();
        note_ss_nonconvergence_if_capped(early_stopped, incr_prev, incr_last, incr_mag);
        return u;
    }

    // #914: exact periodic steady state for a LINEAR disposition — the affine fixed point
    // `u_ss = (I − M)⁻¹·b` the analytical walk (#908) and the input-rate branch (#835) already
    // use, replacing the truncated pulse train below. `advance_forced` integrates one cycle
    // *with* the dose (a bolus pulse, or an active-infusion window + quiet window);
    // `advance_unforced` integrates one `II` of disposition alone — the propagator `M`. For an
    // infusion the active and quiet windows share the same homogeneous propagator (the constant
    // `+RATE` forcing has zero state-Jacobian), so a full-`II` unforced decay reconstructs `M`
    // exactly — the window length enters only through `b`. A genuinely nonlinear RHS
    // (Michaelis–Menten, …) fails the linearity self-check, returning `None` → the capped pulse
    // train below, now with the #867 non-convergence warning wired in (gap 2). The exact solve
    // integrates at the tightened `ss_equilibration_opts` tolerance (a handful of one-cycle
    // solves, so it is cheap) and passes those tolerances to the linearity check, mirroring the
    // input-rate path.
    let eq_opts = ss_equilibration_opts(opts);
    // Both closures integrate the cycle `[0, II]` on a clock whose origin *is* the pulse, so
    // the RHS must read `TAD = t` throughout (#1139). The propagator probe carries the same
    // anchor as the forced cycle deliberately: `periodic_ss_fixed_point_g` subtracts one from
    // the other to build `M`, and that decomposition is only the true one-cycle map when both
    // legs see the same clock. A state-linear but time-varying RHS still passes its linearity
    // self-check, so this fixture takes the exact solve rather than the train below — which it
    // could not do while the anchor was absent and every probe came back `NaN`.
    let ext_cycle = ss_run_in_params(pk_params_flat, 0.0);
    // The quiet window's own params are built inside each infusion arm below, not here: a
    // bolus has no quiet window (`bioavailable_infusion` returns `t_inf = 0`), and hoisting
    // it would either allocate an array anchored at `-0.0` that nothing reads, or make it an
    // `Option` whose `expect` asserts an invariant two guards away. Binding it where it is
    // used costs one 1 KB stack array per infusion cycle — against that cycle's ODE solve —
    // and removes the invariant rather than documenting it.
    let advance_unforced = |u0: &[f64]| -> Option<Vec<f64>> {
        solve_ode(
            &base_rhs,
            u0,
            (0.0, dose.ii),
            &ext_cycle,
            &[dose.ii],
            &eq_opts,
        )
        .last()
        .map(|p| p.u.clone())
    };
    let advance_forced = |u0: &[f64]| -> Option<Vec<f64>> {
        if is_inf {
            // Active-infusion window then quiet window — the same one-cycle body the fallback
            // loop runs, as a pure function of `u0`.
            let rate = inf_rate;
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                base_rhs(y, p, t, dy);
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            };
            let mut y = solve_ode(
                &wrapped_rhs,
                u0,
                (0.0, t_inf),
                &ext_cycle,
                &[t_inf],
                &eq_opts,
            )
            .last()
            .map(|p| p.u.clone())?;
            let quiet = dose.ii - t_inf;
            if quiet > 0.0 {
                // Pulse at local `−T_inf`: this window re-opens its clock at `0` but sits
                // `T_inf` after the cycle's pulse. See `ss_run_in_params`.
                let ext_quiet = ss_run_in_params(pk_params_flat, -t_inf);
                y = solve_ode(&base_rhs, &y, (0.0, quiet), &ext_quiet, &[quiet], &eq_opts)
                    .last()
                    .map(|p| p.u.clone())?;
            }
            Some(y)
        } else {
            let mut y = u0.to_vec();
            y[cmt_idx] += f_bio * dose.amt;
            solve_ode(
                &base_rhs,
                &y,
                (0.0, dose.ii),
                &ext_cycle,
                &[dose.ii],
                &eq_opts,
            )
            .last()
            .map(|p| p.u.clone())
        }
    };
    if let Some(u_ss) = periodic_ss_fixed_point_pk(
        chz,
        n,
        dose.ii,
        eq_opts.reltol,
        eq_opts.abstol,
        advance_unforced,
        advance_forced,
    ) {
        record_ss_equilibration_cycles(1);
        crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::Exact);
        return u_ss;
    }

    // Nonlinear disposition (or singular `I − M`): fall back to the capped pulse train, at the
    // model tolerance `opts` (not the tightened `eq_opts`), matching the prior behaviour and the
    // input-rate fallback. Early stop once the trough stops moving (#519): the shared tracker
    // holds the previous cycle's state and, from cycle 1 on, breaks when the increment is below
    // the mixed atol/rtol criterion (#532 review #6 — one scaffold across the f64 paths).
    let mut tracker = SsStopTracker::default();
    let mut cycles_run = 0usize;
    let mut early_stopped = false;
    for cycle in 0..SS_EQUILIBRATION_CYCLES {
        if is_inf {
            // Active-infusion window: wrapped RHS injects rate into the
            // dosing compartment.
            let rate = inf_rate;
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                base_rhs(y, p, t, dy);
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            };
            let sol = solve_ode(&wrapped_rhs, &u, (0.0, t_inf), &ext_cycle, &[t_inf], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            // Quiet window from end-of-infusion to end-of-cycle. Its local clock restarts
            // at `0`, so its `TAD` anchor is `−T_inf`, not `0` — see `ss_run_in_params`.
            let quiet = dose.ii - t_inf;
            if quiet > 0.0 {
                let ext_quiet = ss_run_in_params(pk_params_flat, -t_inf);
                let sol = solve_ode(&base_rhs, &u, (0.0, quiet), &ext_quiet, &[quiet], opts);
                if let Some(last) = sol.last() {
                    u.copy_from_slice(&last.u);
                }
            }
        } else {
            // Bolus pulse + decay for one cycle.
            //
            // NOTE: this applies the SS dose as an instantaneous bolus and does
            // not route it through an input-rate forcing (`R_in`). That is correct
            // only because SS dosing into a built-in absorption (e.g. transit())
            // compartment is rejected upstream by `E_ABSORPTION_SS`
            // (`api::check_absorption_dosing`). When SS + input-rate is supported
            // (a later phase of `plans/absorption-models.md`), this pulse must be
            // suppressed for an input-rate compartment and `R_in` integrated over
            // the cycle instead.
            u[cmt_idx] += f_bio * dose.amt;
            let sol = solve_ode(&base_rhs, &u, (0.0, dose.ii), &ext_cycle, &[dose.ii], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
        cycles_run = cycle + 1;
        if tracker.should_stop(cycle, &u) {
            early_stopped = true;
            break;
        }
    }
    record_ss_equilibration_cycles(cycles_run);
    crate::dosing::record_ss_equilibration_branch(crate::dosing::SsBranch::CappedTrain);
    // Gap 2 (#914): a capped ordinary bolus/infusion equilibration was silent (the warning was
    // wired only into the input-rate branch). Only a *nonlinear* disposition reaches this
    // fallback — the linear closed form above returned early — so this is exactly the saturable
    // heavy-accumulation / no-steady-state case #867 warns about.
    let (incr_prev, incr_last, incr_mag) = tracker.recent_increments();
    note_ss_nonconvergence_if_capped(early_stopped, incr_prev, incr_last, incr_mag);

    u
}

/// Steady-state ODE state at `phase` ∈ [0, II) within the dosing cycle,
/// measured forward from the pulse at phase 0. [`equilibrate_ss_state`]
/// returns the pre-pulse trough (phase 0⁻ ≡ II); this advances from that
/// trough through the dose pulse and `phase` units of the cycle.
///
/// Used to seed the *previous interval's* steady-state tail when an SS dose
/// has a lagtime: observations between the dose record time and the lagged
/// arrival sit at phase `II − lagtime` … `II`, decaying from the prior
/// pulse. Without this seed those samples would read the (empty) initial
/// state. See [`ode_predictions`] for placement and issue #15.
///
/// `phase == 0` is the instant *after* the pulse — a bolus is already in the
/// compartment and an infusion has delivered nothing yet — which is what the
/// [`crate::dosing::ss_seed_phase`] clamp hands back for `lagtime ≥ II`.
/// Returning the bare (pre-pulse) trough there would be off by a whole cycle of
/// decay; NONMEM reads the peak. Note the asymmetry is only apparent: `phase`
/// runs over `[0, II]` where `0` is post-pulse and `II ≡ 0⁻` is pre-pulse.
///
/// For an SS **infusion** with `phase < T_inf` the prior infusion has not
/// finished by `phase`, so the returned state is mid-flight and the caller must
/// carry `+rate` forward for another `T_inf − phase` — see
/// [`crate::dosing::ss_residual_infusion_end`], which is where that window and
/// this one are kept consistent. (Overlapping infusions, `T_inf > II`, are
/// rejected upstream.)
///
/// `chz_before` is [`equilibrate_ss_state`]'s: the accumulator values at the record. The phase
/// advance is still part of the run-in — it reconstructs the *previous* interval's tail, which
/// happened before the record began — so it too integrates under [`mask_chz`]. Without that,
/// a lagged SS dose banked another `phase` worth of hazard on top of the equilibration's, which
/// is where #1210's `ALAG1 = 2` arm got its extra `0.2` and its non-monotone `H`.
fn ss_state_at_phase(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    phase: f64,
    opts: &OdeSolverOptions,
    chz_before: &[f64],
) -> Vec<f64> {
    let mut u = ss_state_at_phase_pk(ode, pk_params_flat, dose, phase, opts);
    restore_chz(ode, &mut u, chz_before);
    u
}

/// The PK-only body of [`ss_state_at_phase`] — see that function. Every integration runs under
/// [`mask_chz`]; the accumulator rows come back at zero for the wrapper to fill.
fn ss_state_at_phase_pk(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    phase: f64,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    // ### The phase advance is a run-in window too, and it is the pre-arrival one
    //
    // Like [`equilibrate_ss_pk_state`]'s windows (#1139), the three `solve_ode` calls below
    // hand the RHS an extended array via [`ss_run_in_params`] rather than a bare
    // `PkParams::values`, so a `TAD`-reading RHS reads a real anchor instead of `NaN`.
    // Until #1126 they did not, deliberately: this function is reachable only through
    // `ss_seeded_at_record` (`lag > 0`), and anchoring it *alone* replaced a loud `NaN`
    // with a plausible number **2.733 % high at every post-arrival observation** — the seed
    // it returns was already right, but the walk then integrated `[t_dose, t_dose + lag)`
    // under a `TAD` anchored at the *first arrival*, and #1121 flows that state to the
    // arrival rather than re-equilibrating there, so the error multiplied into the whole
    // subject by a uniform 1.0273332950. That measurement is why the two halves ship
    // together: [`crate::dosing::tad_referent`] gives the walk the previous cycle's pulse,
    // and this function stops handing it `NaN` to carry.
    //
    // Both windows measure from the **pulse this advance starts at**, which sits at local
    // `0`: `equilibrate_ss_pk_state` returns the pre-pulse trough and the bolus (or the
    // infusion's first `T_inf`) lands at the origin of the spans below. An infusion's quiet
    // window re-opens at local `0` a further `T_inf` after that pulse, so it anchors at
    // `−T_inf`, exactly as the sibling's group-B windows do.
    //
    // Note for anyone asserting `crate::dosing::last_ss_equilibration_branch()` after this
    // function: the tag it leaves belongs to the `equilibrate_ss_pk_state` call below, not to
    // the phase advance, which records nothing of its own.
    let chz = &ode.chz_state_slots[..];
    let base_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
        (ode.rhs)(y, p, t, dy);
        mask_chz(chz, dy);
    };
    let mut u = equilibrate_ss_pk_state(ode, pk_params_flat, dose, opts);
    let cmt_idx = dose.cmt_idx();
    if cmt_idx >= u.len() {
        return u;
    }
    // Bioavailability scales the amount entering the dosing compartment,
    // resolved per dose compartment (`Fn`; see `equilibrate_ss_state`).
    let f_bio = ode.dose_attr_map.f_bio(dose.cmt_raw(), pk_params_flat);
    if phase <= 0.0 {
        // Post-pulse, pre-flow. An infusion delivers over time and so has
        // nothing to add here; the caller's residual window carries it.
        if !is_real_infusion(dose) {
            u[cmt_idx] += f_bio * dose.amt;
        }
        return u;
    }

    if is_real_infusion(dose) {
        // Mode-aware bioavailability (#419): see `equilibrate_ss_state`.
        let (rate, t_inf) = dose.bioavailable_infusion(f_bio);
        let active = phase.min(t_inf);
        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            base_rhs(y, p, t, dy);
            if cmt_idx < dy.len() {
                dy[cmt_idx] += rate;
            }
        };
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (0.0, active),
            &ss_run_in_params(pk_params_flat, 0.0),
            &[active],
            opts,
        );
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        if phase > t_inf {
            let quiet = phase - t_inf;
            let sol = solve_ode(
                &base_rhs,
                &u,
                (0.0, quiet),
                &ss_run_in_params(pk_params_flat, -t_inf),
                &[quiet],
                opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
    } else {
        // Instantaneous SS bolus (no `R_in` routing) — sound only because SS into
        // an input-rate compartment is rejected upstream by `E_ABSORPTION_SS`;
        // see the matching note in `equilibrate_ss_state`.
        u[cmt_idx] += f_bio * dose.amt;
        let sol = solve_ode(
            &base_rhs,
            &u,
            (0.0, phase),
            &ss_run_in_params(pk_params_flat, 0.0),
            &[phase],
            opts,
        );
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }
    u
}

/// Returns `(cmt_idx_0based, rate)` for every infusion that is active
/// throughout the closed segment `[t_start, t_end]`. By construction of the
/// break-time list (every infusion start and end is a break time), each
/// infusion is either fully active or fully inactive across a segment.
///
/// `dose_lagtimes[k]` shifts dose `k`'s active window. Parallel to `doses`.
/// An empty slice means "no lagtime" (all zeros).
///
/// `dose_f_bio[k]` is the bioavailability F applied to dose `k`'s infusion under
/// the mode-aware rule (#419): a rate-defined infusion (`RATE>0`, `RATE=-1`)
/// keeps its rate and `F` scales the active window to `F·AMT/rate`; a
/// duration-defined infusion (`RATE=-2`) keeps its window and `F` scales the rate.
/// Parallel to `doses`; a missing entry defaults to 1.0. The caller's break-time
/// list must split at the same `F`-scaled infusion ends so each segment is fully
/// active or inactive.
pub(crate) fn active_infusions(
    input_rate: &[crate::pk::absorption::InputRateForcing],
    doses: &[DoseEvent],
    t_start: f64,
    t_end: f64,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    reset_floor: f64,
    n_states: usize,
) -> Vec<(usize, f64)> {
    // Both resets reached by this segment, built once (#1586): an infusion recorded before
    // an `SS=1` record or an EVID=3/4 reset is off, whether its window is running across the
    // record, pending behind a lag, or the #1121 residual of an earlier `SS=1` infusion.
    let gate = ResetGate::at_segment(doses, reset_floor, t_start);
    doses
        .iter()
        .enumerate()
        .filter_map(|(k, d)| {
            // The one membership rule, shared with `gated_infusions` (#1196 step 3):
            // real infusion, in-range compartment, not fed by a built-in absorption
            // forcing (whose mass arrives through `R_in_inf` instead, #719 gap 2).
            if !infusion_contributes(input_rate, d, n_states) {
                return None;
            }
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let f_bio = dose_f_bio.get(k).copied().unwrap_or(1.0);
            // `F`-reshaped rate and window (#419).
            let (rate_eff, dur_eff) = d.bioavailable_infusion(f_bio);
            let start = d.time + lag;
            let end = start + dur_eff;
            // Infusions recorded before the most recent reset — EVID=3/4 (#1587) or a
            // reached `SS=1` record (#1586) — are turned off, the same way the reset zeros
            // the compartments. Keyed on the record, so a lagged window opening after the
            // reset is off too.
            let live = gate.live(doses, k);
            if live && start <= t_start + INFUSION_EPS && end >= t_end - INFUSION_EPS {
                return Some((d.cmt_idx(), rate_eff));
            }
            // A seeded steady-state infusion (#1121) whose *previous* cycle is
            // still running at the dose record keeps delivering across the
            // pre-arrival window, on `[d.time, ss_residual_infusion_end]`. That
            // window belongs to no `DoseEvent` — it is the tail of the periodic
            // fiction `ss_state_at_phase` handed back mid-flight — so it is
            // admitted here rather than by the `start`/`end` test above, which
            // only knows about the dose's own arrival. Reset-aware on the record
            // time for the same reason the real window is: an EVID=3/4 between
            // the record and the arrival zeros the seeded state, and a rate that
            // survived it would refill a compartment the reset just emptied — and a later
            // `SS=1` record stops it the same way (#1586).
            let residual_end = ss_residual_infusion_end(d, lag, f_bio)?;
            (live && d.time <= t_start + INFUSION_EPS && residual_end >= t_end - INFUSION_EPS)
                .then_some((d.cmt_idx(), rate_eff))
        })
        .collect()
}

/// One dose's zero-order absorption window — `(cmt_idx, rate, w_start, w_end, k)`, `k` the
/// dose's index in the subject's dose list (the reset gate needs the row, not just the
/// record time: a ZO row before a co-timed `SS=1` row is reset, one after it is not, #1586),
/// the constant `rate = F·amt/dur` delivered over
/// `[w_start, w_end] = [time+lag, time+lag+dur]`. The tuple shape mirrors
/// [`gated_infusions`].
///
/// `dur`/`F`/`lag` are **dose-time** attributes (fixed when the dose is given), so
/// the window and its rate are built from **one** PK snapshot per dose — the
/// per-dose `pk_at_dose[k]` on the event-driven path, the single subject snapshot
/// `pk_params_flat` on the dense paths — and that one snapshot is the invariant
/// that keeps `∫R_in = F·amt` exact even under time-varying covariates:
/// re-deriving the rate from the *running* (mid-window) snapshot would let it
/// drift and silently break mass balance. The event-driven path materialises the
/// windows once and reuses them across segments; the dense paths re-derive them
/// per segment, but always from that same fixed snapshot, so every segment sees
/// byte-identical edges and rate (the cost is a small, often-empty `Vec`).
type ZeroOrderWindow = (usize, f64, f64, f64, usize);

/// Build the per-dose [`ZeroOrderWindow`]s for a subject. `dur_frac_for_dose`
/// yields the floored `dur` **and pathway fraction `frac`** for dose `k` from *its*
/// PK snapshot — a single subject snapshot on the dense paths, the per-dose
/// `pk_at_dose[k]` on the time-varying / event-driven path — so the window edges and
/// rate stay consistent with that snapshot wherever it is also read (the break
/// placement and the per-segment filter share this one source). Doses not feeding a
/// `zero_order` forcing contribute no window.
///
/// The constant window rate is `F·amt·frac/dur`. `frac` is `1` for an unfractioned
/// `zero_order(...)` term (the single-pathway `zero_order`/`sequential` case), and
/// the declared pathway fraction for a `FR*zero_order(...)` term in a `mixed` model
/// (#505) — a linear multiplier on the rate, so the window machinery (break times,
/// full-containment filter, reset turn-off) is otherwise untouched and the mass the
/// window delivers is `rate·dur = F·amt·frac`.
fn zero_order_windows(
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    dur_frac_for_dose: impl Fn(usize, &DoseEvent) -> Option<(f64, f64, f64)>,
) -> Vec<ZeroOrderWindow> {
    let mut out = Vec::new();
    for (k, d) in doses.iter().enumerate() {
        let Some((dur, frac, route_lag)) = dur_frac_for_dose(k, d) else {
            continue;
        };
        let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let f_bio = dose_f_bio.get(k).copied().unwrap_or(1.0);
        // The window opens at `d.time + lag_cmt + lag_route`: the dose's compartment
        // lagtime plus this zero-order route's own delay (`zero_order(..., lag=L)`,
        // `0` for an unlagged route). The full-containment break at `w_end` shifts
        // with it (via `zero_order_dur_and_lag_for_dose`), keeping every segment
        // fully inside or outside the window (#504 mass-exactness) under a route lag.
        let w_start = d.time + lag + route_lag;
        out.push((
            d.cmt_idx(),
            f_bio * d.amt * frac / dur,
            w_start,
            w_start + dur,
            k,
        ));
    }
    out
}

/// The per-segment **constant** zero-order rates whose window fully contains the
/// closed segment `[t_start, t_end]` — the artifact-free analogue of
/// [`active_infusions`] for `zero_order(dur)` forcings (#504).
///
/// A zero-order input delivers a constant `F·amt·frac/dur` over its window (`frac`
/// = 1 for a single-pathway `zero_order`; the pathway fraction for a `mixed`
/// `FR*zero_order`, #505). Evaluating
/// the hard `tad ≤ dur` cutoff **pointwise** inside RK45 mis-resolves the step: the
/// post-cutoff segment's left endpoint (`t = dur`) still reads the in-window rate,
/// so the adaptive solver's first stage there over-counts a sliver of mass.
/// Delivering it as a per-segment constant — like an infusion — sidesteps that: a
/// window is included **only if it fully contains the segment** (`w_start ≤ t_start`
/// and `w_end ≥ t_end`), so the post-cutoff segment (whose right end is past
/// `w_end`) is correctly excluded. The break-time list splits at **both** `w_start`
/// and `w_end` (see [`push_zero_order_break_times`]) so every segment is fully inside
/// or outside each window — the invariant this test relies on, exactly as
/// [`active_infusions`] relies on it for infusion windows. Both edges matter and the
/// filter is two-sided: bracketing only `w_end` leaves the segment straddling
/// `w_start` failing containment, which drops the rate for the entire window rather
/// than mis-resolving an edge (#1171). The segment's [`ResetGate`] over `doses` turns off
/// the window of every dose **recorded** before the most recent reset — EVID=3/4 at
/// `reset_floor` (#1587) or an `SS=1` record the segment has reached (#1586) — wherever its
/// lagged window opens.
fn active_zero_order_inputs(
    windows: &[ZeroOrderWindow],
    doses: &[DoseEvent],
    t_start: f64,
    t_end: f64,
    reset_floor: f64,
) -> Vec<(usize, f64)> {
    // Most subjects have no zero-order window: skip the gate's O(n) cutoff scan for them.
    if windows.is_empty() {
        return Vec::new();
    }
    let gate = ResetGate::at_segment(doses, reset_floor, t_start);
    windows
        .iter()
        .filter(|&&(_, _, w_start, w_end, k)| {
            gate.live(doses, k)
                && w_start <= t_start + INFUSION_EPS
                && w_end >= t_end - INFUSION_EPS
        })
        .map(|&(cmt, rate, _, _, _)| (cmt, rate))
        .collect()
}

/// The floored zero-order duration `dur` **and pathway fraction `frac`** for
/// `dose`, if `dose` feeds a `zero_order(dur)` forcing (positive amount into that
/// forcing's compartment); else `None`. The window length is read through
/// [`PreparedInputRate`] (so it is floored identically to the `R_in` evaluation) and
/// `frac` through [`InputRateForcing::frac`] (`1` for an unfractioned term); used by
/// the [`zero_order_windows`] `dur_frac_for_dose` closures.
///
/// `find_map` resolves **one** zero-order forcing per dose-compartment — the
/// `mixed` model has exactly one (alongside a `first_order` on the same
/// compartment), and the parser (`build_ode_spec`) rejects `> 1` zero-order term
/// on a compartment (biphasic zero-order, #505), so this single-forcing lookup
/// never under-delivers.
fn zero_order_dur_and_frac_for_dose(
    ode: &OdeSpec,
    dose: &DoseEvent,
    pk_params: &[f64],
) -> Option<(f64, f64, f64)> {
    if dose.amt <= 0.0 {
        return None;
    }
    ode.input_rate.iter().find_map(|f| {
        // `f.cmt` is 0-based; match it against the dose's 0-based target index so a `CMT=0` dose
        // (the default dose compartment == compartment 1) resolves its zero-order window instead of
        // missing it. The bare `f.cmt + 1 == dose.cmt` this replaced never matched `CMT=0`, so the
        // bolus was suppressed (`input_rate_consumes_cmt` normalises) yet no window opened — the
        // mass was silently dropped (#899, #913 review).
        if f.kind == crate::pk::absorption::InputRateKind::ZeroOrder && f.cmt == dose.cmt_idx() {
            match f.prepare(pk_params) {
                PreparedInputRate::ZeroOrder { dur, .. } => {
                    Some((dur, f.frac(pk_params), f.route_lag(pk_params)))
                }
                _ => None,
            }
        } else {
            None
        }
    })
}

/// The floored zero-order duration `dur` **and per-route lag** for `dose` (ignoring
/// the pathway fraction) — used by the event-driven timeline's cutoff break, which
/// needs the window *edge* `d.time + lag_cmt + lag_route + dur`, not the rate. A thin
/// projection of [`zero_order_dur_and_frac_for_dose`] so the two never disagree on
/// which forcing / `dur` / `lag_route` a dose resolves to.
fn zero_order_dur_and_lag_for_dose(
    ode: &OdeSpec,
    dose: &DoseEvent,
    pk_params: &[f64],
) -> Option<(f64, f64)> {
    zero_order_dur_and_frac_for_dose(ode, dose, pk_params)
        .map(|(dur, _, route_lag)| (dur, route_lag))
}

/// Does any forcing in `input_rate` feed the **0-based** state `cmt`?
///
/// The single spelling of the input-rate membership rule. Every consumer — the bolus
/// suppression ([`input_rate_consumes_cmt`]) and both infusion resolvers
/// ([`active_infusions`], [`gated_infusions`]) — asks this one question, because a dose
/// into such a compartment is delivered by `R_in` over time and its instantaneous
/// contribution must be suppressed exactly once. #1187 was this rule existing twice and
/// the copies disagreeing; #1196 tracks folding the remaining duplication.
///
/// Takes the **slice**, not an [`OdeSpec`], so a caller that applies no forcing can pass
/// `&[]` and keep its plain contribution (see [`active_infusions`]' EKF caller).
#[inline]
pub(crate) fn forcing_consumes_cmt(
    input_rate: &[crate::pk::absorption::InputRateForcing],
    cmt: usize,
) -> bool {
    input_rate.iter().any(|f| f.cmt == cmt)
}

/// The single spelling of "this infusion contributes a plain `+rate` to the RHS"
/// (#1196 step 3) — the membership rule both infusion resolvers ask, so a term added
/// to one can no longer drift from the other (#1187 was that drift).
///
/// Four conditions, in the order the two resolvers used to spell them separately:
/// it must be a real infusion; `CMT=0` and a compartment past the state vector are
/// dropped (`check_dose_compartments` rejects both since #899, so this is only
/// reachable from a hand-built [`OdeSpec`], where dropping beats panicking inside the
/// integration loop); and a dose into a built-in absorption compartment is suppressed
/// because its mass arrives through the convolved `R_in_inf` instead (#719 gap 2).
///
/// `input_rate` is the forcing slice the caller will **actually apply**, not
/// necessarily the spec's: a caller that applies no forcing (the EKF path) passes
/// `&[]` and keeps its plain `+rate`. Hard-wiring the spec would suppress a rate
/// nothing replaces.
///
/// **Two effects inside [`active_infusions`], not one.** The compartment tests are new
/// to that resolver (they were `gated_infusions`-only before #1196 step 3), and they
/// gate its `ss_residual_infusion_end` branch as well as its plain `+rate` branch — so a
/// `CMT=0` or out-of-range **`SS=1`** infusion now loses its previous-cycle residual
/// window too, not just its rate. Both are unreachable from a validated call:
/// `check_dose_compartments` rejects a `CMT=0` infusion (`E_DOSE_CMT_NOT_INFUSABLE`) and
/// any `cmt > n_states`, and `check_absorption_dosing` rejects an SS infusion into an
/// absorption compartment (`E_ABSORPTION_SS_INFUSION`). Recorded because "confined to
/// the plain `+rate`" would understate the change for a hand-built [`OdeSpec`].
#[inline]
pub(crate) fn infusion_contributes(
    input_rate: &[crate::pk::absorption::InputRateForcing],
    d: &DoseEvent,
    n_states: usize,
) -> bool {
    // The `CMT=0` half is [`crate::dosing::infusion_has_rate_channel`] rather than a
    // local `cmt_raw() == 0`, so this resolver and the analytic sensitivity walk cannot
    // spell it differently — which is exactly what they did through #1077.
    if !is_real_infusion(d) || !crate::dosing::infusion_has_rate_channel(d) {
        return false;
    }
    // One binding, so the range test and the forcing test provably ask about the same
    // compartment — the property this shared predicate exists to guarantee.
    let cmt = d.cmt_idx();
    cmt < n_states && !forcing_consumes_cmt(input_rate, cmt)
}

/// True if a built-in absorption input-rate forcing (transit/etc.) feeds the
/// compartment `cmt_1based` (the data file's 1-based CMT). A dose into such a
/// compartment delivers its mass via `R_in(tad)` integrated over time
/// (`∫R_in dt = F·amt`), so its instantaneous **bolus must be suppressed** to
/// avoid double-counting the dose — the dose feeds the input-rate function, not
/// the state directly (see `plans/absorption-models.md`).
///
/// The spec-reading form of [`forcing_consumes_cmt`], for callers that always apply the
/// model's own forcings.
#[inline]
pub(crate) fn input_rate_consumes_cmt(ode: &OdeSpec, cmt_1based: usize) -> bool {
    forcing_consumes_cmt(&ode.input_rate, cmt_1based.saturating_sub(1))
}

/// Push the hard-cutoff break times — each window's end `w_end` — for the
/// subject's precomputed zero-order windows (#504) onto a dense-path `break_times`
/// list.
///
/// A zero-order input delivers a constant rate over `[w_start, w_end]` then stops —
/// step discontinuities at *both* edges that the smooth densities (transit/igd/weibull)
/// don't have. Without a break there, the adaptive RK45 steps across the edge and
/// mis-resolves the absorbed mass, so the timeline must break at both for every
/// zero-order window — `w_end` mirroring the infusion-end break, `w_start` mirroring
/// the infusion start.
///
/// **Both edges, because [`active_zero_order_inputs`] tests both.** Its filter is
/// `w_start <= t_start && w_end >= t_end`, so a segment straddling an unbracketed
/// `w_start` fails full containment and the constant rate is dropped for the *whole*
/// window — #1171, where two builders pushed only `w_end` and a model whose sole
/// input was a lagged `zero_order` read exactly `0.0` everywhere. Emitting both from
/// the same [`ZeroOrderWindow`] the filter reads is what makes the segment edges and
/// the containment boundary unable to drift apart; pushing one of the two made that
/// claim false for every caller that did not *also* remember
/// [`push_route_lag_break_times`]. Keep it that way: a new break-time builder must
/// get correct zero-order segmentation from this call alone.
///
/// `w_start` is a no-op for an unlagged route (it coincides with the dose's own
/// `d.time + lag_cmt` break and dedups away). Doses turned off by a later reset still
/// get a harmless extra break (over-segmentation only). No-op for the common model
/// with no zero-order window.
fn push_zero_order_break_times(break_times: &mut Vec<f64>, windows: &[ZeroOrderWindow]) {
    break_times.extend(
        windows
            .iter()
            .flat_map(|&(_, _, w_start, w_end, _)| [w_start, w_end]),
    );
}

/// Push a break at every per-route absorption onset `d.time + lag_cmt + lag_route`
/// (`fn(..., lag=L)`) — for each input-rate forcing carrying a `lag_slot`, over every
/// positive-amount dose feeding that forcing's compartment. `route_lag_of(forcing, k)`
/// reads the forcing's lag from the PK snapshot dose `k` is evaluated under (a single
/// subject snapshot on the dense path, so the index is ignored there; the per-dose
/// `pk_at_dose[k]` for a caller that has one, #1505). A route lag delays that
/// route's onset PAST the dose's `d.time + lag_cmt` break, so without this break the
/// smooth routes' onset kink is unresolved and a lagged `zero_order` window's start is
/// unbracketed (never fully contained in a segment → no mass delivered). A no-op when
/// no forcing carries a lag (the `filter` yields nothing), so the common case is free.
fn push_route_lag_break_times(
    break_times: &mut Vec<f64>,
    ode: &OdeSpec,
    subject: &Subject,
    dose_lagtimes: &[f64],
    route_lag_of: impl Fn(&crate::pk::absorption::InputRateForcing, usize) -> f64,
) {
    for forcing in ode.input_rate.iter().filter(|f| f.lag_slot.is_some()) {
        for (k, d) in subject.doses.iter().enumerate() {
            if d.amt > 0.0 && d.cmt_idx() == forcing.cmt {
                break_times.push(
                    d.time
                        + dose_lagtimes.get(k).copied().unwrap_or(0.0)
                        + route_lag_of(forcing, k),
                );
            }
        }
    }
}

/// How a segment's infusions are injected as a `+rate` derivative term in the
/// wrapped RHS. The two shapes mirror how the two families of ODE paths break
/// their timelines:
///
/// - [`InfusionInput::Spanning`]: a constant `(cmt_idx, rate)` list added on
///   every RHS evaluation. The prediction paths split the timeline at every
///   dose/infusion-end, so within a segment each active infusion spans the whole
///   interval — see [`active_infusions`].
/// - [`InfusionInput::Gated`]: `(cmt_idx, rate, t_start, t_end)` tuples, each
///   active only for `t ∈ [t_start − ε, t_end + ε)`. The dense/simulate paths do
///   **not** split at infusion edges, so an infusion can start or end inside a
///   segment and must be gated on the integration time.
///
/// In both cases `rate` already folds in bioavailability (`F·RATE`).
enum InfusionInput {
    Spanning(Vec<(usize, f64)>),
    Gated(Vec<(usize, f64, f64, f64)>),
}

/// Resolve the dense-path infusion list (`(dose_idx, t_start, t_end)`) into the
/// `(cmt_idx, F·rate, t_start, t_end)` tuples the seam's [`InfusionInput::Gated`]
/// branch injects. Doses with `CMT=0` (no compartment) or a compartment beyond
/// the state vector are dropped — the same guard the dense paths applied per RHS
/// evaluation before the seam, lifted out to once per segment.
///
/// **The `InfusionInput::Spanning` twin of [`active_infusions`], and it must drop
/// exactly what that drops.** Both feed the same `wrap_rhs_with_forcings` seam, which
/// adds the resolved `+rate` *alongside* `add_prepared_input_rate_forcing`'s convolved
/// `R_in_inf`. So an infusion into a built-in absorption compartment has to be
/// suppressed here for the same reason it is suppressed there — omitting it delivered
/// the mass twice on every gated engine (#1187: exactly `2·F·amt` in the accumulator,
/// and up to 214× in a readout, because the stray `+rate` lands in the compartment
/// *directly* instead of feeding the kernel).
///
/// `input_rate` is the forcing slice the caller will **actually apply**, which need not be
/// the spec's. Both call sites here pass `&ode.input_rate`, because that is what their own
/// `prepare_input_rates(ode, …)` builds `prepared` from — the suppression and the
/// replacement therefore cover the same compartments by construction. Taking the slice
/// rather than reading the spec through [`input_rate_consumes_cmt`] is what keeps that
/// true: a caller that applies no forcing passes `&[]` and keeps its plain `+rate`, as
/// [`active_infusions`]' EKF caller does, and hard-wiring the spec would suppress a rate
/// nothing replaces.
///
/// `gate` is the segment's [`ResetGate`], and it is this list's **one** reset test
/// (#1586): a window whose dose record precedes an `SS=1` record or an EVID=3/4 reset the
/// segment has reached is dropped here, whether it was registered before the reset,
/// co-timed in an earlier row, or behind a lag. The walkers register every arrival's
/// window unconditionally and never clear the list at a reset, so there is no second gate
/// for this one to disagree with.
fn gated_infusions(
    input_rate: &[crate::pk::absorption::InputRateForcing],
    active: &[(usize, f64, f64)],
    doses: &[DoseEvent],
    dose_f_bio: &[f64],
    n_states: usize,
    gate: &ResetGate,
) -> Vec<(usize, f64, f64, f64)> {
    active
        .iter()
        .filter_map(|&(di, t_start_inf, t_end_inf)| {
            if !gate.live(doses, di) {
                return None;
            }
            let dose = &doses[di];
            // The one membership rule, shared with `active_infusions` (#1196 step 3):
            // real infusion, in-range compartment (`CMT=0` and `cmt > n_states` are
            // unreachable from a validated call since #899 but stay dropped for
            // hand-built `OdeSpec`s — dropping beats panicking inside the integration
            // loop), and not fed by a built-in absorption forcing (#719 gap 2 / #1187).
            if !infusion_contributes(input_rate, dose, n_states) {
                return None;
            }
            let cmt = dose.cmt_idx();
            // Mode-aware bioavailability rate (#419); the `(t_start_inf, t_end_inf)`
            // window already carries the `F`-scaled duration from the caller's
            // break-time list.
            let (rate_eff, _) = dose.bioavailable_infusion(dose_f_bio[di]);
            Some((cmt, rate_eff, t_start_inf, t_end_inf))
        })
        .collect()
}

/// One dose's built-in absorption forcing, read at that dose's own record (#1569): the
/// kernel constants ([`PreparedInputRate`]: `n`/`mtt`, `mat`/`cv2`, `td`/`β`, `ka`) and the
/// forcing's two per-dose multipliers — its pathway fraction `FR` (#388) and its own route
/// lag (`fn(..., lag=L)`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DoseInputRate<T = f64> {
    pub(crate) prep: PreparedInputRate<T>,
    pub(crate) frac: T,
    pub(crate) route_lag: T,
}

impl<T: crate::sens::num::PkNum> DoseInputRate<T> {
    fn read(
        forcing: &InputRateForcing,
        params: &[T],
        prep: &impl Fn(&InputRateForcing, &[T]) -> PreparedInputRate<T>,
    ) -> Self {
        Self {
            prep: prep(forcing, params),
            frac: forcing.frac(params),
            route_lag: forcing.route_lag(params),
        }
    }
}

/// The built-in absorption forcings a forcing loop applies: one [`DoseInputRate`] per forcing
/// (parallel to `ode.input_rate`) and dose.
///
/// **Every absorption parameter is read at the dose record (#1569)**, the way `F` and the
/// compartment lag always were. A built-in forcing is a *per-dose* density — dose `d`
/// delivers `F·D·frac·R_in(t − t_d − lag − lag_route)` with `∫₀^∞ R_in = 1` — and that
/// integral is 1 only while each dose is absorbed through one density. Until #1569 the loop
/// rebuilt the kernel, the fraction and the route lag per segment from the segment's
/// snapshot, so an `MTT` moved mid-absorption by IOV or a time-varying covariate spliced two
/// densities at the same time-since-dose and created or destroyed drug: a transit dose
/// absorbed 0.504–1.424 of its mass under one `MTT` switch
/// (`an_in_flight_dose_absorbs_exactly_its_mass_when_its_kernel_parameter_switches`). The
/// disposition is untouched: it acts on the carried amounts, where the segment's snapshot is
/// exact.
///
/// Built once per subject evaluation, not per segment or RHS call.
pub(crate) struct PreparedForcings<T = f64> {
    /// `None`: one snapshot serves every dose — one row per forcing. The parameter-static
    /// engines, and the SS run-ins, whose synthetic pulse trains carry no records of their
    /// own. `Some(n)`: dose `k` of forcing `f` is row `f·n + k`.
    per_dose: Option<usize>,
    rows: Vec<DoseInputRate<T>>,
}

impl<T: crate::sens::num::PkNum> PreparedForcings<T> {
    /// Every dose reads the one snapshot `params`.
    pub(crate) fn shared(
        ode: &OdeSpec,
        params: &[T],
        prep: impl Fn(&InputRateForcing, &[T]) -> PreparedInputRate<T>,
    ) -> Self {
        Self {
            per_dose: None,
            rows: ode
                .input_rate
                .iter()
                .map(|f| DoseInputRate::read(f, params, &prep))
                .collect(),
        }
    }

    /// Dose `k` reads its own record's snapshot, `dose_params(k)`.
    pub(crate) fn per_dose<'p>(
        ode: &OdeSpec,
        n_doses: usize,
        dose_params: impl Fn(usize) -> &'p [T],
        prep: impl Fn(&InputRateForcing, &[T]) -> PreparedInputRate<T>,
    ) -> Self
    where
        T: 'p,
    {
        let mut rows = Vec::with_capacity(ode.input_rate.len() * n_doses);
        for f in &ode.input_rate {
            rows.extend((0..n_doses).map(|k| DoseInputRate::read(f, dose_params(k), &prep)));
        }
        Self {
            per_dose: Some(n_doses),
            rows,
        }
    }

    /// No forcing to apply — the model has no built-in absorption, or no dose carries one.
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Forcing `forcing`'s kernel for dose `dose`.
    #[inline]
    pub(crate) fn get(&self, forcing: usize, dose: usize) -> &DoseInputRate<T> {
        match self.per_dose {
            None => &self.rows[forcing],
            Some(n) => &self.rows[forcing * n + dose],
        }
    }

    /// The fraction that scales forcing `forcing`'s whole dose sum, when one snapshot serves
    /// every dose. Factoring it out of the sum keeps the parameter-static engines'
    /// arithmetic bit-for-bit what it was; per dose it cannot factor.
    #[inline]
    fn common_frac(&self, forcing: usize) -> Option<T> {
        match self.per_dose {
            None => Some(self.rows[forcing].frac),
            Some(_) => None,
        }
    }
}

/// The `f64` [`PreparedForcings`] for the one snapshot `params`.
fn prepare_input_rates(ode: &OdeSpec, params: &[f64]) -> PreparedForcings {
    PreparedForcings::shared(ode, params, InputRateForcing::prepare)
}

/// The steady-state periodic-sum forcing of a **single** SS dose into a built-in
/// absorption compartment (#719): `Σ_{j≥0} R_in(tad + j·II)`, the still-in-flight
/// absorption of the infinite past pulse train the one `SS=1` record stands for.
/// `R_in` is the stateless, dose-scaled per-dose kernel and the absorption chain
/// is linear, so the appearance rate at the evaluation time is the superposition
/// of every prior pulse's tail. (The *already-absorbed* mass of those pulses —
/// now distributing/clearing — is seeded separately as the initial state by
/// [`equilibrate_ss_state`]'s trough, disjoint from this sum, so there is no
/// double count.)
///
/// Past the density's mode the terms decrease monotonically, so the sum truncates
/// once a tail term is negligible **relative to this dose's own running total**
/// (`local`), hard-capped at [`SS_EQUILIBRATION_CYCLES`] (the trough's budget).
/// Keeping the break total *local* is load-bearing: the caller superposes this
/// over every dose into the compartment, and comparing each SS tail term against a
/// shared cross-dose accumulator would let an unrelated (e.g. large run-in) dose
/// inflate the threshold and truncate this train's small pre-mode leading terms
/// prematurely. Extracting the sum here makes that cross-dose contamination
/// structurally impossible — the function has no access to the outer accumulator.
/// Generic over `T: PkNum` so the one implementation serves the `f64` predictor
/// and the `Dual*` sensitivity walk identically.
#[inline]
fn ss_periodic_forcing<T: crate::sens::num::PkNum>(
    prep: &PreparedInputRate<T>,
    tad: T,
    ii: T,
    dose_mass: T,
) -> T {
    let mut local = T::from_f64(0.0);
    let mut j = 0usize;
    while j < SS_EQUILIBRATION_CYCLES {
        let tad_j = tad + ii * T::from_f64(j as f64);
        if tad_j.val() > 0.0 {
            let term = prep.rate(tad_j, dose_mass);
            local = local + term;
            if j >= 1
                && local.val().abs() > 0.0
                && term.val().abs() <= SS_TAIL_REL_FLOOR * local.val().abs()
            {
                break;
            }
        }
        j += 1;
    }
    local
}

/// The `SS=1` reset (#1576) of one integration segment, built once per segment and read
/// by [`add_prepared_input_rate_forcing`] on every RHS evaluation.
///
/// A dose that precedes the latest `SS=1` record reached by the segment start in (time,
/// row order) contributes nothing ([`crate::dosing::ss_reset_cutoff`]), and an `SS=1`
/// dose's implied pulse train contributes nothing before its own record. Both are gated
/// **per segment, not pointwise**: the gate changes exactly at a dose record, which is
/// always a break, and Dormand–Prince evaluates its last stage at the right end of the
/// step, so a pointwise `t ≥ t_s` gate would feed the next regimen's tails into the last
/// stage of the segment that *ends* at `t_s` (the same reason `reset_floor` is per
/// segment). The record is the dose row's own time: `SS=1` + lag into a forcing is
/// rejected up front.
///
/// "Reached" is **within [`EVENT_MATCH_TOL`]**, the tolerance the walkers already fire a
/// dose at a break with. It cannot be exact: breaks are deduplicated at `1e-15` keeping
/// the *lower* value, so a segment can start 1 ulp below the record it belongs to (an
/// infusion ending at `0.1 + 0.7 = 0.7999999999999999` merges with an `SS=1` record at
/// `0.8`). The walker still equilibrates the record's state there, and an exact
/// comparison read the record as not reached for the whole segment — train off, earlier
/// doses live (PR #1589 review finding 1: −51 % at t = 6). [`Self::at_segment`] is the
/// only constructor, so no caller can apply the gate without the tolerance.
///
/// It also carries the **EVID=3/4 reset** (#1587), keyed the same way: on the dose
/// *record*, never its lagged arrival. NONMEM cancels a dose whose record precedes a reset
/// even when `record + ALAG` lands after it (measured, `nonmem_anchor/evid_reset_lag`), so
/// a dose is dead once the most recent EVID=3/4 reset reached by the segment is later than
/// its record. The caller passes that reset as `reset_floor` — the floor every walker
/// already tracks, `NEG_INFINITY` before the first reset — because only the walker knows
/// which resets it has visited. A dose recorded *at* the floor is live: an EVID=4 row's
/// own dose, and a dose row after a co-timed EVID=3 row (the reader shifts a reset that
/// lands at or before an earlier dose past it, `RESET_SEGMENT_GAP`, so a dose row
/// *before* a co-timed reset reaches the engines with `record < reset`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResetGate {
    cutoff: Option<usize>,
    reached: f64,
    reset_floor: f64,
}

impl ResetGate {
    /// The gate for a segment starting at `t_seg` over `doses`, under the EVID=3/4
    /// `reset_floor` the walker has reached there. `NEG_INFINITY` for `t_seg` reaches no
    /// `SS=1` record (the synthetic non-SS pulse trains, which have none to reach); for
    /// `reset_floor` it means no EVID=3/4 reset.
    #[inline]
    pub(crate) fn at_segment(doses: &[DoseEvent], reset_floor: f64, t_seg: f64) -> Self {
        let reached = t_seg + EVENT_MATCH_TOL;
        Self {
            cutoff: crate::dosing::ss_reset_cutoff(doses, reached),
            reached,
            reset_floor,
        }
    }

    /// Whether dose `k` is live: not reset by an `SS=1` record this segment has reached,
    /// and not recorded before the EVID=3/4 reset in force.
    ///
    /// Also the arrival gate of every state engine (#1588, #1587): a dose whose record
    /// precedes a reset reached at its arrival break changes nothing there — no bolus
    /// jump, no arrival re-equilibration, and on the dual walks no lag saltation. The
    /// absorption forcing reads it per segment, and every infusion and zero-order window's
    /// membership reads it too (#1586): a window recorded before a reached reset is off,
    /// running, pending behind a lag, or the #1121 residual of an earlier `SS=1` infusion.
    #[inline]
    pub(crate) fn live(&self, doses: &[DoseEvent], k: usize) -> bool {
        crate::dosing::ss_reset_live(doses, self.cutoff, k)
            && crate::dosing::evid_reset_live(doses[k].time, self.reset_floor)
    }

    /// Whether this segment has reached `d`'s record, so its implied pulse train is on.
    #[inline]
    fn record_reached(&self, d: &DoseEvent) -> bool {
        d.time <= self.reached
    }
}

/// Add every built-in absorption input-rate forcing into `dy` at integration
/// time `t`. For each forcing, sums `frac·R_in(tad)` over all doses targeting its
/// compartment (Savic superposition), with `tad = t − (dose.time + lag + lag_route)`
/// and dose mass `F·amt`. `R_in = 0` for `tad ≤ 0`, so future doses contribute
/// nothing. This is the input-rate analogue of the `+rate` infusion injection in the
/// wrapped RHS.
///
/// `gate` carries both resets of the segment being integrated ([`ResetGate`]): the `SS=1`
/// record (#1576) and the EVID=3/4 floor, keyed on the dose record (#1587) — a lagged dose
/// recorded before the reset contributes nothing even when its input would start after it.
///
/// Each dose is absorbed through the kernel, fraction and route lag of **its own dose
/// record** (`prepared.get(forcing, dose)`, see [`PreparedForcings`]), never the
/// current segment's: with IOV or a time-varying covariate on an absorption parameter
/// that is what keeps each dose's delivered mass at exactly `F·amt·frac` (#1569).
///
/// Generic over the numeric type `T: PkNum` so the **single** superposition loop
/// serves both the production `f64` predictor (`T = f64`, byte-identical to the
/// original) and the analytic ODE sensitivity provider's dual walk (`T = Dual*`),
/// instead of `sens/ode_provider.rs` hand-maintaining a second copy (#430 review
/// #4 / #451). The two dual callers each feed one branch live: the TV-cov
/// event-driven walk (`integrate_tvcov_g`) passes the tracked dual `dose_lagtimes`
/// for an in-scope estimated lagtime (#486), and the static walk (`integrate_g`)
/// passes a gate built from the tracked `reset_floor` for an in-scope EVID 3/4 reset (#486).
/// `integrate_g` still passes `dose_lagtimes = &[]` (its gate excludes lagtime
/// subjects, which always route to the TV-cov walk instead).
#[inline]
#[allow(clippy::too_many_arguments)] // mirrors the dose context threaded into the RHS wrappers
pub(crate) fn add_prepared_input_rate_forcing<T: crate::sens::num::PkNum>(
    ode: &OdeSpec,
    prepared: &PreparedForcings<T>,
    doses: &[DoseEvent],
    dose_lagtimes: &[T],
    dose_f_bio: &[T],
    gate: ResetGate,
    t: f64,
    dy: &mut [T],
) {
    debug_assert!(
        prepared.per_dose.is_none_or(|n| n == doses.len()),
        "per-dose forcings built for {:?} doses, applied to {}",
        prepared.per_dose,
        doses.len()
    );
    for (fi, forcing) in ode.input_rate.iter().enumerate() {
        if forcing.cmt >= dy.len() {
            continue;
        }
        // Zero-order is delivered as a per-segment constant (`active_zero_order_inputs`,
        // routed through the wrapper's spanning channel), not pointwise: its hard
        // `tad ≤ dur` cutoff would otherwise let the post-cutoff segment's left
        // endpoint over-count a sliver of mass (#504). Skip it here; the smooth
        // densities (transit/igd/weibull) stay on this exact pointwise path.
        if forcing.kind == crate::pk::absorption::InputRateKind::ZeroOrder {
            continue;
        }
        // Pathway fraction (#388): a `FR*fn(...)` term scales its dose's `R_in` by the
        // declared fraction `FR`; `frac = 1` for an unfractioned single-pathway forcing, so
        // this is a no-op there. The multiplier flows linearly, so for `T = Dual2` it
        // carries the exact `∂R_in/∂frac` sensitivity.
        let common_frac = prepared.common_frac(fi);
        let mut acc = T::from_f64(0.0);
        for (k, d) in doses.iter().enumerate() {
            if d.cmt_idx() != forcing.cmt || !gate.live(doses, k) {
                continue;
            }
            let dose_rate = prepared.get(fi, k);
            // `dose_lagtimes[k]` (`T`, not `f64`) carries the exact `∂t_eff/∂lag = 1`
            // sensitivity when the caller's lag is itself an estimated parameter (an
            // event-driven walk with an in-scope lagtime, #486) — `T::from_f64(0.0)`
            // (zero jet) for every other caller (production `f64`, or a dual walk with
            // no lagtime), so `tad` below reduces to the pre-#486 constant-boundary
            // computation there. The gating comparisons use `.val()` (the boundary
            // itself never needs a jet — see `rate_at_zero`'s jump for that).
            let lag = dose_lagtimes.get(k).copied().unwrap_or(T::from_f64(0.0));
            // Per-route absorption delay (`fn(..., lag=L)`): an offset ON TOP of the
            // dose's compartment lag, so each parallel / mixed pathway can switch on at
            // its own time; `0` for an unlagged forcing (the common case), a no-op there.
            // Since #859 a `first_order` per-route lag is analytic on the `Dual2`
            // event-driven walk: this continuous `∂R_in/∂lag_route` shift flows here, and
            // the onset discontinuity is supplied separately as the `K_ROUTE_ONSET` rate-on
            // saltation.
            let t_eff = T::from_f64(d.time) + lag + dose_rate.route_lag;
            let tad = T::from_f64(t) - t_eff;
            let dose_mass =
                dose_f_bio.get(k).copied().unwrap_or(T::from_f64(1.0)) * T::from_f64(d.amt);
            let prep = &dose_rate.prep;
            let rate = if d.ss && d.ii > 0.0 {
                // The implied train starts at the record: nothing before it (#1576).
                if !gate.record_reached(d) {
                    continue;
                }
                ss_periodic_forcing(prep, tad, T::from_f64(d.ii), dose_mass)
            } else if d.is_infusion() {
                // Infusion (RATE>0) into a built-in absorption compartment (#719 gap 2): the
                // dose is a *zero-order source* feeding the kernel — its mass is delivered at a
                // constant rate over the infusion window, so `R_in` is the convolution of the
                // kernel with that rectangle, `(dose/T)·[G(tad) − G(tad − T)]` (mass-exact).
                // The bioavailable window `T` (#419: rate-defined → `F·amt/rate`, duration-defined
                // → the duration) is a *fixed* boundary here — the analytic sensitivity of the
                // window under an estimated `F` (rate-defined case) is gated to FD upstream. The
                // dose's plain `+rate` injection is suppressed for this compartment
                // (`active_infusions` skips input-rate cmts), so there is no double count.
                let f_bio_k = dose_f_bio.get(k).copied().unwrap_or(T::from_f64(1.0));
                let window = d.bioavailable_infusion(f_bio_k.val()).1;
                prep.rate_infused(tad, dose_mass, T::from_f64(window))
            } else {
                if tad.val() <= 0.0 {
                    continue;
                }
                prep.rate(tad, dose_mass)
            };
            acc = match common_frac {
                Some(_) => acc + rate,
                None => acc + dose_rate.frac * rate,
            };
        }
        dy[forcing.cmt] = match common_frac {
            Some(frac) => dy[forcing.cmt] + acc * frac,
            None => dy[forcing.cmt] + acc,
        };
    }
}

/// The single seam that wraps a model's user RHS with the two dose-driven
/// forcing terms shared by **all** ODE integration paths: the infusion `+rate`
/// injection and the built-in absorption input-rate forcing (`R_in`,
/// transit/etc.).
///
/// Before this seam each path hand-copied `(ode.rhs)(…)` + the infusion loop +
/// `add_input_rate_forcing(…)` into its own closure; a new path or absorption
/// model had to replicate it in every one, and an omission silently dropped the
/// forcing (#322 #6). Routing every path through here removes the copy-paste.
///
/// `reset_floor` is threaded per call and **intentionally differs** by path: the
/// two non-reset paths (`ode_predictions`, `ode_predictions_with_states`) pass
/// `f64::NEG_INFINITY` because the dispatcher routes reset subjects to the
/// event-driven walker; the two reset-aware paths pass a real floor. `t_seg` is the
/// start of the segment the closure integrates; with `reset_floor` it builds the segment's
/// [`ResetGate`], which carries both resets into the forcing (see
/// [`add_prepared_input_rate_forcing`]). `prepared`
/// holds each dose's absorption forcing, read at its dose record ([`PreparedForcings`]).
#[allow(clippy::too_many_arguments)] // each is a distinct slice of dose/forcing context
fn wrap_rhs_with_forcings<'a>(
    ode: &'a OdeSpec,
    doses: &'a [DoseEvent],
    dose_lagtimes: &'a [f64],
    dose_f_bio: &'a [f64],
    reset_floor: f64,
    t_seg: f64,
    prepared: &'a PreparedForcings,
    infusions: InfusionInput,
    zero_order: &'a [(usize, f64)],
) -> impl Fn(&[f64], &[f64], f64, &mut [f64]) + 'a {
    // Built once per closure: `t_seg` and `reset_floor` are fixed for the segment it
    // integrates.
    let gate = ResetGate::at_segment(doses, reset_floor, t_seg);
    move |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
        (ode.rhs)(y, p, t, dy);
        // Zero-order absorption (#504): a constant rate per *segment*, injected the
        // same way as a spanning infusion (independent of the infusion gating
        // shape). The caller passes only the windows that fully contain this
        // segment (`active_zero_order_inputs`), so there is no time gate here.
        for &(cmt_idx, rate) in zero_order {
            if cmt_idx < dy.len() {
                dy[cmt_idx] += rate;
            }
        }
        match &infusions {
            InfusionInput::Spanning(active) => {
                for &(cmt_idx, rate) in active {
                    if cmt_idx < dy.len() {
                        dy[cmt_idx] += rate;
                    }
                }
            }
            InfusionInput::Gated(active) => {
                for &(cmt_idx, rate, t_start_inf, t_end_inf) in active {
                    // +ε on the upper bound (not −ε) so the infusion is active
                    // right up to t_end_inf — the dynamic gate must not cut off
                    // the last sub-step.
                    if t >= t_start_inf - INFUSION_EPS
                        && t < t_end_inf + INFUSION_EPS
                        && cmt_idx < dy.len()
                    {
                        dy[cmt_idx] += rate;
                    }
                }
            }
        }
        if !prepared.is_empty() {
            add_prepared_input_rate_forcing(
                ode,
                prepared,
                doses,
                dose_lagtimes,
                dose_f_bio,
                gate,
                t,
                dy,
            );
        }
    }
}

/// Function that computes the observable from
/// `(state, pk_params_flat, theta, eta, covariates)`. Used by `[scaling]
/// y = <expr>` (Form C) to replace the default `u[obs_cmt_idx]` readout
/// with an arbitrary expression over states + individual parameters +
/// thetas + etas + covariates. Callers that don't have theta/eta in scope
/// (e.g. the EKF path, which never sets a Single/PerCmt readout) may pass
/// empty slices.
pub type OdeOutputFn =
    Box<dyn Fn(&[f64], &[f64], &[f64], &[f64], &HashMap<String, f64>) -> f64 + Send + Sync>;

/// How an ODE model's observable is read at each observation event.
///
/// Replaces the earlier mutually-exclusive `(obs_cmt_idx, output_fn)` pair
/// with a single enum that scales naturally to per-CMT (multi-analyte)
/// dispatch.
pub enum OdeReadout {
    /// Default: read `state[obs_cmt_idx]` (0-based into the state vector)
    /// for every observation regardless of its CMT. The canonical
    /// single-output ODE shape.
    ObsCmt(usize),
    /// Form C uniform: `[scaling] y = <expr>` — a single output_fn
    /// replaces the state-index readout for every observation.
    Single(OdeOutputFn),
    /// Form C per-CMT: `[scaling] y[CMT=N] = <expr>` for each observed
    /// CMT. Key is the 1-based CMT index from the data file (matches
    /// `subject.obs_cmts[i]`, which is `usize`). Fit-time validation
    /// enforces that every observed CMT has an entry; missing entries
    /// fall through to NaN at runtime as a defensive guard.
    PerCmt(HashMap<usize, PerCmtReadout>),
}

/// One per-CMT Form-C readout (`y[CMT=N] = <expr>`): the f64 closure the production
/// predictor calls, plus the optional `PkNum`-differentiable program the analytic
/// sensitivity provider evaluates over `Dual2`/`Dual1` (issue #439). `program` is
/// `None` for hand-constructed readouts that bypass the parser — those keep the f64
/// FD path (the dual provider declines them).
pub struct PerCmtReadout {
    pub out_fn: OdeOutputFn,
    pub program: Option<crate::parser::model_parser::OdeOutputProgram>,
}

impl OdeReadout {
    /// Evaluate the readout at one observation given the compartment `state`
    /// vector, the flat PK-parameter slice, θ/η, the covariate snapshot, the
    /// observation's 1-based CMT, and the observation `time`. Shared by the ODE
    /// predictor ([`read_observable`]) and the analytic Form C path
    /// (`pk::apply_analytic_readout`, #650) so the two dispatch/NaN-guard
    /// conventions cannot drift. A `PerCmt` map miss (or an out-of-range
    /// `ObsCmt`) yields `NaN` — the loud guard that propagates to a NaN OFV
    /// rather than silently mis-reading, since parser + fit-time validation
    /// already guarantee every observed CMT has an entry.
    ///
    /// `time` seeds the model-time thread-local for the duration of the Form C
    /// arms, so a `[scaling] y = <expr>` readout that references the `TIME` / `T`
    /// built-in resolves `Op::PushTime` to *this* observation's time (#1028).
    /// Without the guard the readout ran outside any [`ModelTimeGuard`] — the
    /// integrator's guard is dropped before the readout — so `TIME` silently read
    /// the `0.0` default and collapsed the whole structural prediction. The
    /// `ObsCmt` arm reads a state slot directly and skips the guard entirely, so
    /// the overwhelmingly common built-in readout pays nothing. The analytic and
    /// dual-walk readout sites (`sens::provider::apply_readout_jet`,
    /// `sens::ode_provider::resolve_obs_readout`) enter the matching guard, so
    /// FD and analytic sensitivities linearise the same expression.
    #[inline]
    pub(crate) fn eval(
        &self,
        state: &[f64],
        pk_params_flat: &[f64],
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
        obs_cmt: usize,
        time: f64,
    ) -> f64 {
        match self {
            OdeReadout::ObsCmt(idx) => state[*idx],
            OdeReadout::Single(out_fn) => {
                let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(time);
                out_fn(state, pk_params_flat, theta, eta, covariates)
            }
            OdeReadout::PerCmt(map) => match map.get(&obs_cmt) {
                Some(r) => {
                    let _time_guard = crate::parser::model_parser::ModelTimeGuard::enter(time);
                    (r.out_fn)(state, pk_params_flat, theta, eta, covariates)
                }
                None => f64::NAN,
            },
        }
    }
}

/// Read the observable value at observation `obs_idx`.
///
/// `subject.obs_cmts[obs_idx]` selects the per-CMT readout when
/// `OdeReadout::PerCmt` is in use; the simpler variants ignore it. `time` is the
/// observation time a `TIME`-referencing Form C readout resolves against (#1028)
/// — see [`OdeReadout::eval`].
#[inline]
fn read_observable(
    ode: &OdeSpec,
    u: &[f64],
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    obs_cmt: usize,
    time: f64,
) -> f64 {
    ode.readout
        .eval(u, pk_params_flat, theta, eta, covariates, obs_cmt, time)
}

/// Record `read_observable` into `predictions[obs_idx]` for every observation
/// sharing a break/save time — the state-independent obs-recording idiom copied
/// across the dense drivers. `pk`/`eta` are the (constant) snapshot for these
/// observations; the per-observation TV/IOV variants (which pick `pk`/`eta` per
/// `obs_idx`) stay inline. When `states` is `Some`, the compartment state `u` is
/// cloned into `states[obs_idx]` too (the `_with_states` driver).
#[inline]
fn record_observations(
    ode: &OdeSpec,
    obs_idxs: &[usize],
    u: &[f64],
    pk: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    predictions: &mut [f64],
    mut states: Option<&mut [Vec<f64>]>,
) {
    for &obs_idx in obs_idxs {
        let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
        // The readout's `TIME` is the *user* clock (`readout_time`), not the shifted
        // integrator timeline — the `$ERROR` convention the rest of the per-record
        // objects use. The two differ only under stacked reset occasions (#1028).
        let t_obs = subject.readout_time(obs_idx);
        predictions[obs_idx] =
            read_observable(ode, u, pk, theta, eta, subject.obs_cov(obs_idx), cmt, t_obs);
        if let Some(states) = states.as_deref_mut() {
            states[obs_idx] = u.to_vec();
        }
    }
}

/// Clamp negative predictions to zero (ODE solver overshoot guard) — the shared
/// epilogue of the dense drivers. NaN is intentionally NOT clamped (it survives
/// `< 0.0` per IEEE 754) so it propagates to a NaN OFV.
///
/// A no-op unless the readout is a bare state ([`OdeReadout::clamps_negative`]):
/// the overshoot guard is a statement about a compartment amount, not about an
/// arbitrary Form C `[scaling]` expression, which is often legitimately signed
/// (#1020).
#[inline]
fn clamp_negative_predictions(readout: &OdeReadout, predictions: &mut [f64]) {
    if !readout.clamps_negative() {
        return;
    }
    for p in predictions.iter_mut() {
        if *p < 0.0 {
            *p = 0.0;
        }
    }
}

/// TAD anchor for `ext_params[MAX_PK_PARAMS + 1]`: the latest referent any of the
/// subject's doses has at `t_start`, per [`crate::dosing::tad_referent`] — which is
/// where the SS pulse-train fold and the seeded pre-arrival window (#1126) are
/// defined, once, for every engine.
///
/// Before any dose has a referent — the window a lagged *non*-SS first dose opens —
/// it falls back to the subject's **earliest lagged arrival**, exactly as the two
/// production ODE predictors both now do (this is their one implementation). The two
/// production ODE predictors are selected per subject on `has_resets()`, so a
/// divergence here would make two subjects of the same model and the same data
/// shape behave differently: one finite, its neighbour NaN — and a NaN anchor
/// multiplies into the state (`0.0 * NaN`) and poisons every prediction of an
/// `[odes]` RHS reading `TAD`, turning a finite fit into the 1e20 sentinel.
///
/// One value per subject, so — unlike anchoring at `t_start` — it cannot make the
/// answer depend on where records happen to fall. What `TAD` *means* before an
/// ordinary dose has arrived remains a convention; this only guarantees it is
/// finite and mesh-independent, and identical across both predictors. A **seeded
/// steady-state** dose's pre-arrival window is not that case — there the periodic
/// fiction does have a prior pulse and the referent is determined, which is what
/// [`crate::dosing::tad_referent`] returns (#1126).
///
/// Returns NaN only for a **dose-free** subject, where `TAD` has no referent at all
/// (the pre-existing answer, and the sdtab convention, for that case).
#[inline]
fn tad_anchor(subject: &Subject, dose_lagtimes: &[f64], t_start: f64) -> f64 {
    tad_anchor_for(&subject.doses, dose_lagtimes, t_start)
}

/// The dose-list body of [`tad_anchor`]. Split out so an engine that holds only a
/// `&[DoseEvent]` — the EKF (`crate::ode::ekf::solve_ekf`), which has no `Subject` —
/// anchors `TAD` by the same rule as the two ODE predictors instead of growing yet
/// another spelling of it (#1131).
///
/// **This is now the fold, and [`crate::dosing::tad_referent`] is the rule.** There used
/// to be four hand-written copies of the per-dose arithmetic — this function, an inline
/// duplicate further down this file, `api::output_columns::tad_at_time`, and the
/// lag-ignoring sdtab fallback in `io::output` — and three of them disagreed on
/// `(t_dose = 120, ALAG = 3, II = 12, t = 121)`, returning `-2.0`, `NaN` and `+1.0`.
/// #1126 collapsed the **lag-aware** ones onto one function, because "add the pre-arrival
/// referent" spelled four times is four chances to spell it differently.
///
/// Two are deliberately outside that, and both are recorded rather than left implicit:
/// [`crate::types::Subject::data_tad`] (#1182/#1273) answers the lag-free *data* question
/// with its own tie-convention switch — see the note on [`crate::dosing::tad_referent`];
/// and the dual walk's anchors in `sens::ode_provider` (segment start, and the pre/post
/// sides of a saltation) are a different shape (a running `max` over arrivals, not a
/// per-segment re-fold) and unreachable for this model class behind the
/// `has_ss && reads_model_time` FD gate — see #1272, which is where that is tracked.
///
/// `dose_lagtimes` may be **shorter than `doses`, including empty** — a missing entry is
/// zero lag, matching [`active_infusions`] and `api::output_columns::tad_at_time`. An
/// engine with no lagtime concept passes `&[]`.
///
/// **The `ss` branch describes a periodic pulse train.** It folds the elapsed time into
/// `[0, II)` as though virtual doses had arrived at `t_dose + k·II`, so a caller whose
/// state was *not* built from such a train gets an anchor its own dose history does not
/// justify — measured on #1263 as a 24.1% `ipred` divergence for one `SS=1` infusion
/// whose end break lands past a virtual pulse, back when the EKF applied the record as a
/// single dose (since #1260 it expands the train too, `ode::ekf::equilibrate_ss_ekf`).
/// Since #1126 that also covers the pre-arrival window of a *seeded* SS dose, whose state
/// comes from `ss_state_at_phase`; a caller that does not perform that seed must not
/// consume this anchor there either. A caller that does not equilibrate an `SS` dose at
/// all must warn rather than quietly consume it.
#[inline]
pub(crate) fn tad_anchor_for(doses: &[DoseEvent], dose_lagtimes: &[f64], t_start: f64) -> f64 {
    // `.get(..).unwrap_or(0.0)`, not `dose_lagtimes[i]`: a short slice means "no lag on
    // the remaining doses", which is what `active_infusions` (`:1520`) and
    // `api::output_columns::tad_at_time` already spell, and it lets a caller with no
    // lagtime concept at all pass `&[]` instead of allocating a zero-filled vector whose
    // only job is to satisfy a length invariant enforced by a panic (#1263 review).
    let lag_at = |i: usize| dose_lagtimes.get(i).copied().unwrap_or(0.0);
    let last_dose_eff = doses
        .iter()
        .enumerate()
        .filter_map(|(i, d)| crate::dosing::tad_referent(d, lag_at(i), t_start))
        .fold(f64::NEG_INFINITY, f64::max);
    if last_dose_eff.is_finite() {
        return last_dose_eff;
    }
    // No dose has arrived yet. `fold` over an empty dose list leaves `+∞`, which is
    // the dose-free case and must read NaN rather than propagate as an infinite
    // anchor.
    let first_arrival = doses
        .iter()
        .enumerate()
        .map(|(i, d)| d.time + lag_at(i))
        .fold(f64::INFINITY, f64::min);
    if first_arrival.is_finite() {
        first_arrival
    } else {
        f64::NAN
    }
}

/// Per dose-compartment lagtime / bioavailability vectors (`Fn`/`ALAGn`; issue
/// #369, with fallback to the bare `lagtime`/`F` slots). Uniform on the no-TV
/// dense path, where every dose reads the same `pk_params_flat`.
#[inline]
fn subject_dose_attrs(
    subject: &Subject,
    ode: &OdeSpec,
    pk_params_flat: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    subject_dose_attrs_with(subject, ode, |_| pk_params_flat)
}

/// [`subject_dose_attrs`] with a per-dose PK snapshot: dose `k`'s lag and `F` are read
/// from `pk_for_dose(k)`. The dense path's uniform-snapshot form above is the
/// `|_| pk_params_flat` instance of this.
#[inline]
fn subject_dose_attrs_with<'a>(
    subject: &Subject,
    ode: &OdeSpec,
    pk_for_dose: impl Fn(usize) -> &'a [f64],
) -> (Vec<f64>, Vec<f64>) {
    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .enumerate()
        .map(|(k, d)| ode.dose_attr_map.lagtime(d.cmt_raw(), pk_for_dose(k)))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .enumerate()
        .map(|(k, d)| ode.dose_attr_map.f_bio(d.cmt_raw(), pk_for_dose(k)))
        .collect();
    (dose_lagtimes, dose_f_bio)
}

/// Per-dose lagtimes only — the half of [`subject_dose_attrs`] that
/// [`rhs_ext_params_at`] needs. Callers that read `H`/`h` at many `times` for the same
/// subject (the TTE hazard readouts, #1261) compute this once and reuse it, rather than
/// re-walking the dose list — and recomputing the unused `F`/bioavailability half — on
/// every timepoint.
#[cfg(feature = "survival")]
#[inline]
pub(crate) fn dose_lagtimes_for(
    subject: &Subject,
    ode: &OdeSpec,
    pk_params_flat: &[f64],
) -> Vec<f64> {
    dose_lagtimes_reading(subject, ode, DoseReads::Shared(pk_params_flat))
}

/// [`dose_lagtimes_for`] with dose `k`'s lag read through `dose_reads` — the lags the
/// dense walk ran under when it was handed the same `dose_reads` (#1575), so a `TAD` the
/// hazard RHS reads afterwards is anchored where the walk's arrivals landed.
#[cfg(feature = "survival")]
#[inline]
pub(crate) fn dose_lagtimes_reading(
    subject: &Subject,
    ode: &OdeSpec,
    dose_reads: DoseReads<'_>,
) -> Vec<f64> {
    subject
        .doses
        .iter()
        .enumerate()
        .map(|(k, d)| ode.dose_attr_map.lagtime(d.cmt_raw(), dose_reads.at(k)))
        .collect()
}

/// Build the extended parameter slice required for a standalone RHS read at `t`, from
/// a dose-lagtime vector and first-dose time the caller has already computed (via
/// [`dose_lagtimes_for`] / [`earliest_dose_time`]) — once per subject, not once per `t`.
///
/// Integrating prediction paths keep this slice live and update its TAD anchor at
/// each segment boundary. Readout-only callers (the TTE hazard derivative) do not
/// own that walk, so they must derive both anchors from the same dose-time helpers
/// instead of passing a bare [`PkParams::values`](crate::types::PkParams::values)
/// array to the RHS (#1261).
#[cfg(feature = "survival")]
#[inline]
pub(crate) fn rhs_ext_params_at(
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    first_dose_time: f64,
    pk_params_flat: &[f64],
    t: f64,
) -> [f64; crate::types::MAX_PK_PARAMS + 2] {
    let mut ext_params = seed_ext_params(pk_params_flat, first_dose_time);
    ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor_for(doses, dose_lagtimes, t);
    ext_params
}

/// Earliest dose record time, or `+∞` when there are no doses.
///
/// Takes the dose list rather than the `Subject` so the EKF
/// (`crate::ode::ekf::solve_ekf`), which has no `Subject`, seeds its TAFD anchor from this
/// function instead of re-spelling the fold — the same reason [`tad_anchor_for`] exists.
/// `seed_ext_params` maps the dose-free `+∞` to `NaN`, so callers pass this straight through.
#[inline]
pub(crate) fn earliest_dose_time(doses: &[DoseEvent]) -> f64 {
    doses.iter().map(|d| d.time).fold(f64::INFINITY, f64::min)
}

/// #1151 / #1535: the reactive driver's refusal of a **dose-clock window with no referent**.
///
/// `TAD` and `TAFD` are anchored at a dose. In a *causal* run the driver's shadow subject
/// carries only the doses realized so far, so before the first one it is dose-free and both
/// anchors are `NaN` ([`tad_anchor_for`]'s empty fold, and the `NaN` seed of
/// `ext_params[MAX_PK_PARAMS]` that [`update_tafd_anchor`] lowers at the first realized
/// dose). The static engines have no such window — `predict()` sees the whole record, so its
/// fallback returns the first *future* arrival and `TAD` comes back finite and negative —
/// which is exactly the peek a controller has not earned: the dose anchoring it has not been
/// decided yet, and may never be. (With no dose anywhere in the record `predict()` returns
/// `NaN` there too, measured — so the anchoring claim is conditional on the record carrying
/// one, and the message says so.)
///
/// Returns `Some(message)` when the segment `(t_start, t_end]` must be refused: **a dose-clock
/// slot the `[odes]` RHS reads is unanchored (`NaN`) there, and the RHS's derivative over the
/// segment depends on that slot's value** — measured by [`unanchored_clock_dependence`], which
/// evaluates the RHS directly and re-solves nothing. Called after [`integrate_segment`], so
/// `ext_params` holds the anchors the segment actually ran under and `u` its advanced state. A
/// zero-length segment needs no special case: [`integrate_segment`] returns before touching
/// either, so `u` is whatever the previous segment left.
///
/// That asks the harm question directly — would the answer change if the clock had a value? —
/// and it replaced three approximations of it, each falsified by a measurement:
///
/// - **The parse walk alone** (#1534 review, round 1). `pk_reads_tad()` is a *syntactic* walk
///   (`stmts_read_slots`) that recurses into `if` arms and into conditions, so it is true for a
///   `TAD` in a branch the window never takes, and for one that only feeds a condition on an
///   empty compartment — both finite and verifier-clean. It stays as a pre-filter: a slot the
///   program never reads cannot move a derivative, so the walk cannot cause a false negative,
///   and it keeps the probe off every model that reads no dose clock.
/// - **The outcome** (#1534 review, rounds 2–3). Refusing only a segment that integrated to a
///   non-finite state missed every `NaN` consumed by a *comparison*. `min`/`max` desugar to
///   `if (a <= b) …`, and an ordering comparison against `NaN` is always false, so
///   `min(TAD, 24)` silently yielded 24 and `if (TAD < 5)` silently took its else arm: the
///   state stayed finite, and the run returned a number that depended on which way the
///   comparison fell. The counterfactual re-solve that confirmed causation on the non-finite
///   path also took its evidence from the frozen tail the solver pads a diverged segment with
///   (#1539), so it would have gone silent once that tail is fixed.
/// - **A "slot was loaded" flag** (#1535's proposal). A load refuses `if (TAD > 5) {…} else
///   {…}` on an empty compartment, where both arms give the same zero derivative, and it puts a
///   branch on every RHS evaluation in the process.
///
/// The message opens with what happened, and there are two openers. A segment whose state went
/// non-finite **and** whose derivative the unanchored clock itself made non-finite keeps the
/// #1151 one ("integrated to a non-finite state"). Every other refused segment — a clock
/// consumed by a comparison, including one that only chose a branch while some other state
/// diverged on its own — is told that its derivative changes with the clock's value, which is
/// true of all of them; the non-finite opener would pin on the clock a symptom it did not
/// cause.
///
/// Keyed **per spelling**, not on [`OdeRhsProgram::pk_reads_model_time`]: `T`/`TIME` are the
/// integration axis and are always anchored, so a `TIME`-reading RHS over a dose-free base
/// integrates correctly and must not be refused. The dependence test backs that up — such a RHS
/// reads no clock slot, so no anchor can move it, and keying the candidates on
/// `pk_reads_model_time` alone changes no verdict (measured, #1535) — which leaves the
/// per-spelling keys two jobs: naming the spelling, and keeping the probe off a RHS that reads
/// neither. (The fixture that pins the `TIME` case starts from an empty compartment, so what it
/// can see is "not `NaN`", not the value of `TIME` in the window —
/// `adaptive_time_reading_rhs_is_not_refused_before_the_first_dose`.)
///
/// **And what nothing here sees**: once any state goes non-finite the solve stops advancing
/// *every* state, so a run that survives this function can still return frozen, finite, wrong
/// numbers — `predict()` included, which is why the frozen-replay verifier agrees with the
/// driver on them. That is an engine defect, filed as #1539, and it is the reason a silent
/// row of the message table is not the same as a correct one.
fn unanchored_dose_clock_error(
    ode: &OdeSpec,
    subject: &Subject,
    u_start: &[f64],
    u: &[f64],
    ext_params: &[f64],
    t_start: f64,
    t_end: f64,
) -> Option<String> {
    let prog = ode.rhs_program.as_ref()?;
    // The anchors exactly as the segment received them: `integrate_segment` wrote the `TAD`
    // slot itself, and the `TAFD` slot is the one it inherited. Reading the quantities the
    // RHS was handed — rather than re-deriving "is the shadow dose-free?" — keeps the
    // diagnostic gated on its own mechanism, and costs no second fold of the dose list.
    let tad_unanchored = prog.pk_reads_tad() && ext_params[TAD_ANCHOR_SLOT].is_nan();
    let tafd_unanchored = prog.pk_reads_tafd() && ext_params[TAFD_ANCHOR_SLOT].is_nan();
    if !tad_unanchored && !tafd_unanchored {
        return None;
    }
    let dependence = unanchored_clock_dependence(
        ode,
        ext_params,
        tad_unanchored,
        tafd_unanchored,
        u_start,
        u,
        t_start,
        t_end,
    )?;
    let slot = match (dependence.tad, dependence.tafd) {
        (true, true) => "`TAD` and `TAFD`",
        (true, false) => "`TAD`",
        _ => "`TAFD`",
    };
    // The reads taken off this segment, resolved with the same predicate
    // [`integrate_segment`] builds its `saveat` from — minus `t_end`'s own band. A record
    // there IS sampled pre-dose into `saveat`, but the value finally reported for it is the
    // POST-dose boundary read the next break takes (pinned by
    // `degenerate_oracle_tad_rhs_observation_on_a_decision_boundary_is_bit_identical`), so
    // naming it as "read off that segment" points the reader at the wrong record. Such a
    // record is poisoned the way the no-observation branch below describes: through the
    // state carried into the dose (#1534 review, row 9).
    let first_obs = subject
        .obs_times
        .iter()
        .copied()
        .filter(|&t| reads_in_segment(t, t_start, t_end) && (t_end - t).abs() > EVENT_MATCH_TOL)
        .fold(f64::INFINITY, f64::min);
    let reads = if first_obs.is_finite() {
        format!("The observation at t={first_obs} is read off that segment.")
    } else {
        "No observation is read off that segment, but the state integrated there carries \
         into every later read."
            .to_string()
    };
    // Both conjuncts, or the non-finite opener would be false of a state that stayed finite,
    // or would blame the clock for a divergence it did not cause (see the doc comment).
    let opener = if dependence.nan_reached_derivative && !u.iter().all(|x| x.is_finite()) {
        format!(
            "ode_predictions_adaptive: the segment ({t_start}, {t_end}] integrated to a \
             non-finite state, and the [odes] RHS reads {slot}, which has no referent there: no \
             dose has been given in this run."
        )
    } else {
        format!(
            "ode_predictions_adaptive: over the segment ({t_start}, {t_end}] the [odes] RHS \
             reads {slot}, which has no referent there: no dose has been given in this run. \
             The derivative it computes there changes when that clock is anchored, so the \
             trajectory depends on a dose time this run does not have. A comparison against \
             an unanchored clock does not fail, it answers one way — `NaN < 5` is false — \
             which is how an `if`, `min` or `max` on it picks a side without producing a \
             `NaN`."
        )
    };
    Some(format!(
        "{opener} {reads} Where the record contains a dose, the \
         static engines anchor such a window at the first one — a dose a reactive run has \
         not decided yet, and may never decide — so this driver refuses the window rather \
         than read one out of the future. Give the subject a pre-scheduled base regimen, or \
         have the controller dose at a decision at or before the subject's first record; a \
         controller that never doses leaves a model reading {slot} unanchored for the whole \
         run."
    ))
}

/// Intervals in the grid [`unanchored_clock_dependence`] evaluates a segment's RHS on:
/// `CLOCK_PROBE_INTERVALS + 1` evenly spaced times spanning `[t_start, t_end]`, both ends
/// included (#1535).
const CLOCK_PROBE_INTERVALS: usize = 16;

/// The `ext_params` slot the `[odes]` RHS reads the `TAFD` anchor (the first dose time) from.
const TAFD_ANCHOR_SLOT: usize = crate::types::MAX_PK_PARAMS;

/// The `ext_params` slot the `[odes]` RHS reads the `TAD` anchor (the last dose time) from.
const TAD_ANCHOR_SLOT: usize = crate::types::MAX_PK_PARAMS + 1;

/// What [`unanchored_clock_dependence`] measured on a segment whose derivative depends on an
/// unanchored dose clock (#1535).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClockDependence {
    /// The message names `TAD`.
    tad: bool,
    /// The message names `TAFD`.
    tafd: bool,
    /// Some derivative component was non-finite under the unanchored clock and finite under an
    /// anchored one: the clock's `NaN` reached the arithmetic, as in `(1 + KT·TAD)`, rather
    /// than being consumed by a comparison. It selects the message's opener, never the verdict.
    nan_reached_derivative: bool,
}

/// #1535: does the `[odes]` RHS's derivative over `[t_start, t_end]` depend on the value of an
/// unanchored dose clock? `None` when it does not.
///
/// `tad` / `tafd` mark the candidate slots: `NaN` in `ext_params`, and read by the program. The
/// raw RHS — without the dose forcings, which read no clock — is evaluated at
/// [`CLOCK_PROBE_INTERVALS`]` + 1` evenly spaced times spanning the segment, at `u_start` and
/// at the end state `u`, each only when every component is finite: once with `ext_params`
/// exactly as the segment ran, and once with every candidate slot anchored at `t_end`. The
/// segment depends on the clock iff some derivative component differs. Each of these choices is
/// pinned by a test:
///
/// - **One anchor: the window's end** (#1570 review, row 1). A window reaches this probe only
///   when no dose has landed by `t_start`, so on the realized schedule the first dose — and with
///   it the static engines' anchor — comes at `t_end` or later, and their clock is ≤ 0 across
///   the window. A dose landing right at `t_end` is the earliest anchor a schedule can give; it
///   puts `TAD` in `[-L, 0]` for a window of length `L`. An anchor at `t_start` would put it in
///   `[0, L]`, where no schedule can: it refused `if (TAD > 5)` and `max(TAD, 0)`, whose
///   unanchored arm is the arm every `TAD ≤ 0` takes, so those runs already equal `predict()`
///   (measured bit-identical, and to 4.3e-16). The one shape only it could see — a RHS that is
///   non-finite on all of `[-L, 0]`, such as `TAD^(-0.5)` — is non-finite in `predict()` too.
/// - **Only states whose every component is finite** (#1570 review, row 2). `u` is probed as
///   well as `u_start` because an empty compartment at the start can hide a comparison that the
///   state filled during the segment makes visible. But once any component goes non-finite the
///   solver pads every state with its last value (#1539), so the finite components of such a
///   state are that pad, not an integrated one: `u` after this segment diverged, or `u_start`
///   after an earlier segment did — a divergence that is not this window's clock's. Neither is
///   probed, which also keeps the verdict the same whatever the pad holds.
/// - **The grid, not the endpoints.** A condition can hold only inside the window — `TAD`
///   between −8 and −6 on a 12 h one — so the ends alone would miss it.
/// - **Compared only where the anchored derivative is finite.** A component whose derivative is
///   non-finite under the anchor too — the anchor overflows the RHS by itself (`exp(-TAD)` at
///   `TAD ≈ −800`), or the RHS is non-finite for every clock a schedule can give
///   (`TAD^(-0.5)`) — is not something an anchor would fix. Then by value, with
///   IEEE `!=`: `NaN` differs from every finite value, while `-0.0 == 0.0`, since a zero
///   derivative is zero whatever its sign — and on an empty compartment
///   `-k·0·(1 − 0.1·min(TAD, 24))` does flip that sign between the unanchored clock and the
///   anchored one.
/// - **All candidate slots take the anchor together for the verdict.** Both clocks are
///   unanchored by the same fact, and a reactive run would anchor both at its first dose.
///   Anchoring one while the other stays `NaN` would leave `(1 + TAD + TAFD)` non-finite under
///   the anchor, where the comparison skips it — a false negative. The *naming* is per slot, and
///   is worked out only once the segment is found dependent: with both unanchored, a slot is
///   named when its `NaN` alone, the other anchored, moves the anchored derivative; when only
///   the pair does (`if (TAD < 5 || TAFD < 5)`), both are named.
///
/// Its limits, stated rather than hidden — measured on #1570, filed as #1572, and left to the
/// default-on frozen-replay verifier until then:
///
/// - A first dose later than `t_end` puts the clock further below zero than the anchor reaches,
///   so a condition that switches only there is not seen: `if (TAD < -20)` over 12 h windows,
///   with the first dose at 36, runs 23.5× off `predict()` at its worst read.
/// - It samples 17 times and pairs the window's start and end states with each. A dependence
///   between two sample times, or one that shows only at states inside the window, is missed;
///   and a pairing the trajectory never reaches can be refused although the run matches
///   `predict()` — measured with a condition on both the state and `TIME`.
///
/// Cost: `2 × 17 × 2` RHS evaluations on a segment with an unanchored clock the program reads,
/// plus `2 × 17 × 3` to name the slots when both are candidates and the segment is refused.
/// Nothing on segments with an anchored clock, and nothing on the RHS hot path.
#[allow(clippy::too_many_arguments)]
fn unanchored_clock_dependence(
    ode: &OdeSpec,
    ext_params: &[f64],
    tad: bool,
    tafd: bool,
    u_start: &[f64],
    u: &[f64],
    t_start: f64,
    t_end: f64,
) -> Option<ClockDependence> {
    // A fixed order: `alone[j]` below is indexed like this list.
    let candidates: Vec<usize> = [(tad, TAD_ANCHOR_SLOT), (tafd, TAFD_ANCHOR_SLOT)]
        .into_iter()
        .filter_map(|(read, slot)| read.then_some(slot))
        .collect();
    // The one comparison the rule makes, per derivative component — see the doc comment.
    let moves = |ran: f64, anchored: f64| anchored.is_finite() && ran != anchored;

    let n = u_start.len();
    let (mut du_ran, mut du_anchored) = (vec![0.0; n], vec![0.0; n]);
    // Every candidate slot at the window's end: the earliest anchor a schedule can give.
    let mut anchored = ext_params.to_vec();
    for &slot in &candidates {
        anchored[slot] = t_end;
    }
    let states: Vec<&[f64]> = [u_start, u]
        .into_iter()
        .filter(|state| state.iter().all(|x| x.is_finite()))
        .collect();
    let times = (0..=CLOCK_PROBE_INTERVALS)
        .map(|k| t_start + (t_end - t_start) * (k as f64 / CLOCK_PROBE_INTERVALS as f64));
    let mut depends = false;
    let mut nan_reached_derivative = false;
    for &state in &states {
        for t in times.clone() {
            (ode.rhs)(state, ext_params, t, &mut du_ran);
            (ode.rhs)(state, &anchored, t, &mut du_anchored);
            for (&ran, &anch) in du_ran.iter().zip(&du_anchored) {
                if moves(ran, anch) {
                    depends = true;
                    nan_reached_derivative |= !ran.is_finite();
                }
            }
        }
    }
    if !depends {
        return None;
    }
    // The naming, worked out only now that the segment is refused (#1570 review, row 6): with
    // both clocks unanchored, which one's `NaN` alone — the other anchored — moves the anchored
    // derivative?
    let mut alone = [false; 2];
    if candidates.len() > 1 {
        let mut alone_unanchored = anchored.clone();
        for &state in &states {
            for t in times.clone() {
                (ode.rhs)(state, &anchored, t, &mut du_anchored);
                for (j, &slot) in candidates.iter().enumerate() {
                    alone_unanchored.copy_from_slice(&anchored);
                    alone_unanchored[slot] = f64::NAN;
                    (ode.rhs)(state, &alone_unanchored, t, &mut du_ran);
                    alone[j] |= du_ran
                        .iter()
                        .zip(&du_anchored)
                        .any(|(&ran, &anch)| moves(ran, anch));
                }
            }
        }
    }
    // One candidate is named by the verdict itself. Two are named by what each does alone —
    // unless only the pair moves the derivative, which names both.
    let named = |slot: usize| match candidates.iter().position(|&s| s == slot) {
        None => false,
        Some(_) if candidates.len() == 1 || !alone.iter().any(|&a| a) => true,
        Some(j) => alone[j],
    };
    Some(ClockDependence {
        tad: named(TAD_ANCHOR_SLOT),
        tafd: named(TAFD_ANCHOR_SLOT),
        nan_reached_derivative,
    })
}

/// Lower the reactive driver's TAFD anchor (`ext_params[MAX_PK_PARAMS]`) to `t` if `t`
/// precedes the current anchor, or set it when none is (`NaN`). #934: a base regimen
/// pre-seeds the anchor to the earliest *base* dose, but a controller dose scheduled
/// *before* the earliest base dose is the true first dose — so the anchor must be
/// `min(earliest base, first controller dose)`, matching the static frozen-replay
/// verifier's `earliest_dose_time` over the merged (base ∪ ledger) list. Called at each
/// realized controller dose; the ascending break walk means only the first can lower a
/// finite base-dose seed.
fn update_tafd_anchor(ext_params: &mut [f64], t: f64) {
    let slot = &mut ext_params[crate::types::MAX_PK_PARAMS];
    if !slot.is_finite() || t < *slot {
        *slot = t;
    }
}

/// Seed the extended-parameter array for the ODE RHS: slots `0..MAX_PK_PARAMS`
/// hold the PK snapshot; slot `MAX_PK_PARAMS` carries the TAFD anchor (the first
/// dose time, NaN when there are no doses so the RHS injects NaN rather than `-∞`);
/// slot `MAX_PK_PARAMS + 1` (TAD) is left NaN for the per-segment update.
#[inline]
pub(crate) fn seed_ext_params(
    pk_params_flat: &[f64],
    first_dose_time: f64,
) -> [f64; crate::types::MAX_PK_PARAMS + 2] {
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };
    ext_params
}

/// Map each time (by bit pattern) to *all* its indices. Multiple observations can
/// share a time (e.g. simultaneous PK/PD samples on different CMTs), so each time
/// maps to every index — recording only one would leave the others at their
/// initial NaN.
#[inline]
fn build_obs_index_map(times: &[f64]) -> HashMap<u64, Vec<usize>> {
    let mut map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in times.iter().enumerate() {
        map.entry(t.to_bits()).or_default().push(i);
    }
    map
}

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
    /// State slots of the **injected** joint-PK-TTE `d/dt(__chz_<cmt>)` cumulative-hazard
    /// accumulators. Empty for every model without an `[event_model]`, and empty in a build
    /// without the `survival` feature.
    ///
    /// These rows are not compartments of the PK system: each is a pure integrator with no
    /// elimination term, so it has no steady state and must be held out of anything that
    /// assumes one. `[fit_options]`-visible consequence: an `SS=1` dose equilibrates the PK
    /// rows only and hands the accumulator back at its pre-record value (#1210).
    ///
    /// Carried by slot rather than by name for the reason #1166 records: without `survival`
    /// there is no reserved-name guard, so a user may legally declare a state called
    /// `__chz_1`, and a name-prefix filter would then silently treat that user's own state as
    /// an accumulator.
    pub chz_state_slots: Vec<usize>,
    /// How the per-observation observable is computed. Replaces the
    /// earlier `(obs_cmt_idx, output_fn)` pair — see [`OdeReadout`].
    pub readout: OdeReadout,
    /// Per-state diagonal process-noise variances (σ²_w,i) for SDE / EKF.
    /// Length must equal `n_states` when non-empty; empty means standard ODE
    /// (no diffusion). Declared via `[diffusion]` block as `state ~ variance`,
    /// analogous to sigma/omega notation. Updated each outer iteration as
    /// diffusion thetas are re-estimated.
    pub diffusion_var: Vec<f64>,
    /// Optional per-subject initial compartment amounts. Declared in the
    /// `[odes]` block as `init(state) = <expr>`; the expression may reference
    /// individual parameters (so it folds in theta/eta/covariates via the
    /// individual-parameter layer, exactly like the RHS). Given the flat
    /// individual-parameter vector (`PkParams.values`), returns the full
    /// `n_states`-length initial-amount vector — the init value for declared
    /// states and `0.0` for the rest. `None` when no `init(...)` is declared,
    /// in which case every compartment starts at zero (the historical default).
    /// A system reset (EVID=3/4) re-applies this on the ODE event-driven path.
    #[allow(clippy::type_complexity)]
    pub init_fn: Option<Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>>,
    /// RK45 solver tolerances used to integrate this system. Defaults to
    /// `OdeSolverOptions::default()` (reltol 1e-4 / abstol 1e-6); overridden
    /// from the model's `[fit_options]` (`ode_reltol` / `ode_abstol` /
    /// `ode_max_steps`) and call-time `settings` via
    /// [`crate::types::CompiledModel::sync_ode_solver_opts`]. Carried on the spec so every
    /// integration entry point (`ode_predictions*`, EKF) uses the configured
    /// accuracy without threading options through each call.
    pub solver_opts: OdeSolverOptions,
    /// Built-in absorption input-rate forcing terms (design A,
    /// `plans/absorption-models.md`). Each adds `R_in(tad)` into its compartment
    /// during integration, superposed over doses — the same RHS-wrapper layer
    /// that injects `+rate` for infusions. Empty for models with no built-in
    /// `transit()`/etc. input-rate term (the historical default).
    pub input_rate: Vec<crate::pk::absorption::InputRateForcing>,
    /// Compiled RHS program for the analytic-sensitivity path (issue #367,
    /// Option A): lets the sensitivity provider evaluate the same RHS over
    /// `Dual2<N>` to obtain exact PK-parameter derivatives. `None` for
    /// hand-built specs (tests, EKF) and any model outside the ODE-sensitivity
    /// scope gate; those fall back to the gradient-free path.
    pub rhs_program: Option<crate::parser::model_parser::OdeRhsProgram>,
    /// Compiled Form C readout (`[scaling] y = <expr>`) for the analytic-
    /// sensitivity path (issue #367): lets the provider evaluate the scaled
    /// observable (e.g. `central / V1`) over `Dual2<N>`. `None` for `ObsCmt`
    /// readouts (read the state directly), per-CMT Form C, and hand-built specs.
    pub readout_program: Option<crate::parser::model_parser::OdeOutputProgram>,
    /// Compiled `[individual_parameters]` program for the analytic-sensitivity
    /// η/θ chain (issue #367): lets the provider compute `∂p/∂η`, `∂p/∂θ`
    /// **analytically** over `Dual2`, instead of finite-differencing `pk_param_fn`.
    /// Attached after `[individual_parameters]` is parsed; `None` for hand-built
    /// specs.
    pub indiv_param_program: Option<crate::parser::model_parser::IndivParamProgram>,
    /// Compartment-indexed dose attributes (NONMEM `Fn`/`ALAGn`). Maps
    /// `(attribute, 1-based compartment) -> PkParams slot` for any `F{c}` /
    /// `ALAG{c}` / `LAGTIME{c}` individual parameter the model declares;
    /// resolves bioavailability / lag **per dose compartment** instead of from
    /// the single `PK_IDX_F` / `PK_IDX_LAGTIME` slot (issue #369). Empty for the
    /// common bare-`F`/`lagtime` model, where every lookup falls through to the
    /// reserved slot (i.e. the historical single-value behaviour).
    pub dose_attr_map: crate::types::DoseAttrMap,
}

impl OdeSpec {
    /// The solver options this spec is actually integrated at: its baked
    /// [`solver_opts`](Self::solver_opts) with any fit-scoped override merged in (#1212).
    ///
    /// **Every integration path must read this, not the field.** `solver_opts` is stamped at
    /// parse time and `fit` takes `&CompiledModel`, so a call-time `FitOptions::ode_reltol` /
    /// `ode_method` / … has no other way to reach the integrator; a site that reads the field
    /// directly silently runs at the parse-time value instead. Outside a fit (`predict`, a
    /// hand-built spec) nothing is armed and this returns the field unchanged.
    pub(crate) fn effective_solver_opts(&self) -> crate::ode::OdeSolverOptions {
        crate::ode::solver::effective_solver_options(self.solver_opts)
    }

    /// Initial compartment-amount vector for a subject, given the flat
    /// individual-parameter vector `params` (`PkParams.values`). Returns the
    /// `init(...)` expression values where declared and `0.0` elsewhere; when
    /// no `init(...)` is declared this is all zeros — the historical default.
    /// Used to seed the integrator at the start of a record and to re-seed it
    /// after an EVID=3/4 reset.
    pub fn initial_state(&self, params: &[f64]) -> Vec<f64> {
        match &self.init_fn {
            Some(f) => f(params),
            None => vec![0.0; self.n_states],
        }
    }

    /// True when any built-in absorption input-rate forcing carries a per-route
    /// lag (`fn(..., lag=L)`, #859). The single predicate shared by the sensitivity
    /// gates (`ode_analytical_supported`'s subject variants), the IOV gate
    /// (`ode_iov_supported`), the event-driven walk (`integrate_tvcov_g`), the
    /// initial-point diagnostics (`api::check_absorption_dosing`), and
    /// `crate::types::CompiledModel::has_route_absorption_lag` — so the "does this
    /// model have a route lag?" test cannot drift between them.
    pub fn has_route_lag(&self) -> bool {
        self.input_rate.iter().any(|f| f.lag_slot.is_some())
    }

    /// Convenience accessor: returns the canonical `obs_cmt_idx` when the
    /// readout is the default `ObsCmt` variant. Used by EKF (which requires
    /// a single observable compartment) and by callers that need to know
    /// whether the readout is "Phase 1 simple" vs "Form C custom".
    pub fn obs_cmt_idx(&self) -> Option<usize> {
        match &self.readout {
            OdeReadout::ObsCmt(idx) => Some(*idx),
            OdeReadout::Single(_) | OdeReadout::PerCmt(_) => None,
        }
    }
}

impl OdeReadout {
    /// Returns true when this readout cannot be paired with `gradient = ad`.
    ///
    /// Both Form C variants (`Single` and `PerCmt`) call arbitrary
    /// user-defined closures at each observation. The analytical AD entry
    /// points take only a single `Const f64` scale and cannot evaluate
    /// closures over theta/eta — there's no AD path for Form C. At runtime
    /// `model.tv_fn` is `None` for any ODE model anyway, so AD silently
    /// falls back to FD. The parse-time guard surfaces that fallback as a
    /// clear error rather than silently demoting the user's `gradient = ad`
    /// choice.
    pub fn requires_fd(&self) -> bool {
        match self {
            OdeReadout::ObsCmt(_) => false,
            OdeReadout::Single(_) | OdeReadout::PerCmt(_) => true,
        }
    }

    /// Whether a negative prediction from this readout is a solver artefact that
    /// should be clamped to zero (#1020).
    ///
    /// True only for the bare-state readout [`OdeReadout::ObsCmt`], where
    /// non-negativity is a *physical* property of the quantity: a compartment
    /// amount / concentration cannot go below zero, so a negative value is RK
    /// overshoot and clamping it is the ODE analogue of the analytical path's
    /// `conc.max(0.0)`.
    ///
    /// False for both Form C `[scaling]` variants. `y = <expr>` is an arbitrary
    /// user expression with no non-negativity guarantee — a change from baseline,
    /// a z-score, a difference from comparator, or the `sqrt(N) * logit(p)`
    /// transform used by model-based meta-analysis are all legitimately negative.
    /// Clamping those silently returned `0` for every genuinely negative
    /// prediction. This also matches the analytical Form C path
    /// (`pk::apply_analytic_readout` / the `sens` providers), which clamps the
    /// *concentration* fed into the readout but never the readout's output.
    #[inline]
    pub fn clamps_negative(&self) -> bool {
        match self {
            OdeReadout::ObsCmt(_) => true,
            OdeReadout::Single(_) | OdeReadout::PerCmt(_) => false,
        }
    }
}

/// Compute ODE-based predictions for a single subject.
///
/// `pk_params_flat` is a flat array of PK parameters passed to the RHS function.
/// `theta` and `eta` are forwarded to `OdeSpec::output_fn` for Form C
/// (`[scaling] y = <expr>`); pass empty slices when no Form C is configured.
/// Integrate one timeline segment `(t_start, t_end]` of the plain ODE path.
///
/// Builds the segment's `saveat`, sets the per-segment TAD anchor on
/// `ext_params`, integrates the forcing-wrapped RHS from the carried state `u`,
/// records every observation landing in the half-open interval, and advances
/// `u` in place to `t_end` so the caller can continue with the next segment.
///
/// The left-boundary discontinuities (SS pre-seed, bolus jumps) and the
/// observation recorded exactly at `t_start` are applied by the caller *before*
/// this call — this function owns only the integration of the open interval,
/// which is the piece a reactive (state-dependent) driver reuses unchanged
/// (#391 S1.2). Behaviour is identical to the inline segment body it replaced.
#[allow(clippy::too_many_arguments)]
fn integrate_segment(
    ode: &OdeSpec,
    u: &mut [f64],
    t_start: f64,
    t_end: f64,
    subject: &Subject,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    // Most-recent system-reset time (EVID=3/4) at or before `t_start`, or
    // `f64::NEG_INFINITY` when none applies. Doses / infusions / zero-order
    // windows started before it are turned off (the reset zeroed the
    // compartments, so their still-arriving tails must stop too), mirroring
    // `ode_predictions_event_driven`. The non-reset callers (`ode_predictions`
    // and the reset-free dense/replay paths) pass `NEG_INFINITY`, so their
    // forcing set is unchanged (#716).
    reset_floor: f64,
    ext_params: &mut [f64],
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    obs_map: &HashMap<u64, Vec<usize>>,
    predictions: &mut [f64],
    stats: Option<&mut OdeSolverStats>,
    auto_state: &mut OdeAutoSwitchState,
    // #570: soft (Hermite-interpolated) sample times within this segment — e.g. TTE
    // event/censor times — read off the *same* integration as the observations,
    // without clamping the step sequence. The returned observation predictions and
    // the advanced `u` are therefore bit-identical to a `chz_times = &[]` call.
    // Must be sorted ascending and lie in `(t_start, t_end]`; the caller filters.
    chz_times: &[f64],
) -> Vec<Vec<f64>> {
    let opts = ode.effective_solver_opts();

    // Observation times recorded off this segment's integration. `reads_in_segment` is
    // the half-open `(t_start, t_end]` minus `t_start`'s own band: the upper bound is
    // **exact**, so a time up to `EVENT_MATCH_TOL` past `t_end` belongs to `t_end`'s band
    // on the next iteration, post-event, rather than to this pre-event integration (#1226).
    let mut saveat: Vec<f64> = subject
        .obs_times
        .iter()
        .filter(|&&t| reads_in_segment(t, t_start, t_end))
        .cloned()
        .collect();
    // Always include t_end so u is updated for next segment
    if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
        saveat.push(t_end);
    }
    saveat.sort_by(|a, b| a.total_cmp(b));
    saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    if (t_end - t_start).abs() < 1e-15 {
        return Vec::new();
    }

    // Update TAD anchor (slot MAX_PK_PARAMS+1): last effective dose time
    // before this segment, SS-aware (gives TAD = t - last_dose_eff).
    ext_params[TAD_ANCHOR_SLOT] = tad_anchor(subject, dose_lagtimes, t_start);

    // Integrate. If any infusions are active in this segment, wrap
    // the user RHS so it adds `+rate` to each infusion's compartment.
    // `reset_floor` turns off infusions started before the most recent
    // system reset (EVID=3/4). The plain dense path (`ode_predictions`) never
    // sees reset subjects — the dispatcher routes those to
    // `ode_predictions_event_driven` — and passes `NEG_INFINITY`, so its
    // active set is unchanged; the reactive driver and its reset-aware replay
    // pass a real floor (#716).
    let active = active_infusions(
        &ode.input_rate,
        &subject.doses,
        t_start,
        t_end,
        dose_lagtimes,
        dose_f_bio,
        reset_floor,
        ode.n_states,
    );
    // Zero-order absorption windows fully covering this segment (#504): constant
    // `F·amt/dur` injected like a spanning infusion. The dense path has a single
    // subject snapshot (`pk_params_flat`), so the windows are the same every
    // segment and consistent with `ode_predictions`' break placement. (Empty for
    // the common model / a non-zero_order subject — e.g. the adaptive caller.)
    let zo_windows = zero_order_windows(&subject.doses, dose_lagtimes, dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    let zero_order =
        active_zero_order_inputs(&zo_windows, &subject.doses, t_start, t_end, reset_floor);
    // Hoist the input-rate constants (ln Γ, KTR, …) once per segment; the PK
    // snapshot `ext_params` is constant across the integration (#322 #7).
    let prepared = prepare_input_rates(ode, ext_params);
    let wrapped_rhs = wrap_rhs_with_forcings(
        ode,
        &subject.doses,
        dose_lagtimes,
        dose_f_bio,
        reset_floor,
        t_start,
        &prepared,
        InfusionInput::Spanning(active),
        &zero_order,
    );
    let (sol, soft) = solve_ode_dense_with_auto_state(
        &wrapped_rhs,
        u,
        (t_start, t_end),
        ext_params,
        &saveat,
        chz_times,
        &opts,
        stats,
        auto_state,
    );

    // Extract predictions and update state
    for pt in &sol {
        if let Some(obs_idxs) = obs_map.get(&pt.t.to_bits()) {
            record_observations(
                ode,
                obs_idxs,
                &pt.u,
                pk_params_flat,
                theta,
                eta,
                subject,
                predictions,
                None,
            );
        }
    }

    // State at end of segment
    if let Some(last) = sol.last() {
        u.copy_from_slice(&last.u);
    }

    // #570: full interpolated state at each requested soft time, in `chz_times`
    // order. Empty (just an empty Vec, no heap alloc) on the `chz_times = &[]` hot
    // path, so existing callers ignore a no-op return.
    soft.into_iter().map(|p| p.u).collect()
}

/// Dose events are handled as state discontinuities between integration segments.
pub fn ode_predictions(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
) -> Vec<f64> {
    ode_predictions_with_extra_breaks_and_stats(
        ode,
        pk_params_flat,
        theta,
        eta,
        subject,
        &[],
        None,
        &[],
    )
    .0
}

/// #570: one augmented-ODE integration yielding **both** the Gaussian predictions
/// and the cumulative-hazard state at `chz_times` (a joint PK-TTE subject's
/// event/censor/entry times), so the joint fit no longer integrates the augmented
/// system a second time to read `H`/`h`.
///
/// The predictions are **bit-identical** to [`ode_predictions`] — the observation
/// `saveat` (which clamps the step sequence) is untouched; the CHZ states are read
/// by in-step cubic Hermite interpolation, which does not perturb the steps.
/// `chz_times` must be **sorted ascending and unique**. Returns `(ipred, chz_states)`
/// where `chz_states[i]` is the full ODE state at `chz_times[i]`. A time before the
/// integration start reads the **seeded initial state** — nothing has acted on the system
/// yet — exactly as the dedicated `ode_dense_solve_states` path does (#1223). NaN survives
/// only where a solve diverged, which the TTE NLL maps to its `1e20` sentinel. `ipred` is
/// the raw observable readout; callers apply `[scaling]` / log-transform exactly as for
/// `ode_predictions`.
///
/// Gated on `survival` — its only consumer is the joint PK-TTE fit path, so the
/// default build neither compiles nor flags it.
#[cfg(feature = "survival")]
pub(crate) fn ode_predictions_and_chz(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    chz_times: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    ode_predictions_with_extra_breaks_and_stats(
        ode,
        pk_params_flat,
        theta,
        eta,
        subject,
        &[],
        None,
        chz_times,
    )
}

/// [`ode_predictions`] plus aggregate RK45 step counters across all integration
/// segments in this subject.
///
/// This is an opt-in diagnostic path: production predictions call
/// [`ode_predictions`] and pay no stats plumbing. The integration segmentation,
/// dose handling, forcing wrapper, and readout logic are otherwise identical,
/// so the returned counters classify the same RK45 work the production
/// predictor performs.
pub fn ode_predictions_with_solver_stats(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
) -> (Vec<f64>, OdeSolverStats) {
    let mut stats = OdeSolverStats::default();
    let (predictions, _chz) = ode_predictions_with_extra_breaks_and_stats(
        ode,
        pk_params_flat,
        theta,
        eta,
        subject,
        &[],
        Some(&mut stats),
        &[],
    );
    (predictions, stats)
}

/// [`ode_predictions`] with additional, dose-free segment break points seeded
/// into the integration timeline.
///
/// Each `extra_break` only *splits* an integration interval — the integrator
/// restarts there with the carried state, but no dose, observation, or state
/// change is applied (the TAFD/TAD anchors, derived from `subject.doses`, are
/// untouched). On the smooth models we integrate the result is invariant to
/// where a no-event break falls only up to the adaptive solver's own error
/// control, so this is the lever the frozen-schedule replay verifier
/// ([`verify_adaptive_frozen_replay`]) uses to reproduce the reactive driver's
/// segment structure exactly: the driver restarts at *every* decision time
/// (including holds and post-`Stop` no-ops), so replaying with those same
/// decision times as breaks makes the two engines share `integrate_segment`
/// over identical segments — turning the comparison bit-aligned rather than
/// merely tolerance-close.
pub(crate) fn ode_predictions_with_extra_breaks(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    extra_breaks: &[f64],
) -> Vec<f64> {
    ode_predictions_with_extra_breaks_and_stats(
        ode,
        pk_params_flat,
        theta,
        eta,
        subject,
        extra_breaks,
        None,
        &[],
    )
    .0
}

/// Push every break time a pre-scheduled dose list contributes to `break_times`:
/// the lagtime-shifted dose time, a real infusion's F-scaled end, the SS+lagtime
/// record time (issue #15), per-route absorption-lag onsets, and zero-order window
/// ends. Factored out of [`ode_predictions_with_extra_breaks_and_stats`] so the
/// reactive driver's pre-scheduled base regimen (#702) builds the **identical**
/// segmentation — a hand-copied second walk would silently drift (cf. #798, where
/// three parallel break-walking loops diverged on dose handling).
///
/// `pk_for_dose(k)` is the PK snapshot dose `k`'s route lag and zero-order window are read
/// from. Every prediction engine that calls this has one subject-wide snapshot and passes
/// `|_| pk_params_flat`; the covariance step's kink bound ([`dose_break_times_per_dose`],
/// #1505) passes the per-record snapshot so a time-varying-covariate subject's events land
/// where its own event-driven walk puts them.
fn collect_dose_break_times<'a>(
    break_times: &mut Vec<f64>,
    ode: &OdeSpec,
    subject: &Subject,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    pk_for_dose: impl Fn(usize) -> &'a [f64],
) {
    for (i, dose) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        break_times.push(dose.time + lag);
        if is_real_infusion(dose) {
            // F-scaled infusion end (#419): a rate-defined infusion's window is
            // `F·duration`. Must match `active_infusions`'s window so each segment
            // is fully inside or outside every infusion.
            let (_, dur_eff) = dose.bioavailable_infusion(dose_f_bio[i]);
            break_times.push(dose.time + lag + dur_eff);
        }
        // SS + lagtime: break at the dose *record* time too, so we can seed the
        // previous-interval steady-state tail there before the lagged pulse arrives.
        if ss_seeded_at_record(dose, lag) {
            break_times.push(dose.time);
        }
        // End of the *previous* cycle's infusion when it is still running at the
        // dose record of a seeded SS dose (#1121) — a segment boundary for the
        // same reason the real infusion end is one.
        if let Some(residual_end) = ss_residual_infusion_end(dose, lag, dose_f_bio[i]) {
            break_times.push(residual_end);
        }
    }
    // Per-route absorption lag (`fn(..., lag=L)`): a route with its own lag switches
    // on past the dose's compartment-lag break, so add a break at each route onset.
    push_route_lag_break_times(break_times, ode, subject, dose_lagtimes, |f, k| {
        f.route_lag(pk_for_dose(k))
    });
    // Zero-order windows (#504): break at each window end so segments align with the
    // cutoff (the same windows `integrate_segment` recomputes for the injection).
    let zo_windows = zero_order_windows(&subject.doses, dose_lagtimes, dose_f_bio, |k, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_for_dose(k))
    });
    push_zero_order_break_times(break_times, &zo_windows);
}

/// Every timeline break the subject's pre-scheduled doses contribute under the per-dose PK
/// snapshots `pk_for_dose(k)`, in the deterministic per-dose order
/// [`collect_dose_break_times`] pushes them — **unsorted and undeduplicated**, so two calls
/// at nearby parameter points can be compared entry by entry.
///
/// This is the moving-event enumerator behind the covariance step's kink bound (#1505):
/// every entry that depends on an estimated quantity — a lagged arrival `t + ALAG`, an
/// infusion end `t + ALAG + F·dur`, a per-route onset `t + ALAG + lag_route`, a
/// `zero_order` window edge — moves when that parameter is perturbed, and an entry that does
/// not (a dose record time under `SS`, an unlagged arrival) comes back bit-identical, so the
/// caller finds the moving ones by differencing the two lists rather than by re-spelling
/// which break depends on what. Reusing the engines' own builder is what keeps the bound
/// honest: a new kind of moving break added there is seen here without a second edit.
///
/// Modeled-`RATE` doses (`RATE=-1/-2`) are read as coded — the covariance scope declines
/// them before this runs (`ode_analytical_supported`'s `all_doses_fixed` gate), and a
/// caller that admits them must resolve the doses first
/// ([`crate::dosing::resolve_subject_doses`]).
pub(crate) fn dose_break_times_per_dose<'a>(
    ode: &OdeSpec,
    subject: &Subject,
    pk_for_dose: impl Fn(usize) -> &'a [f64] + Copy,
) -> Vec<f64> {
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs_with(subject, ode, pk_for_dose);
    let mut out = Vec::with_capacity(2 * subject.doses.len());
    collect_dose_break_times(
        &mut out,
        ode,
        subject,
        &dose_lagtimes,
        &dose_f_bio,
        pk_for_dose,
    );
    out
}

/// Re-seed the state `u` for every pre-scheduled **steady-state** dose landing at
/// `t_start`: the SS+lagtime tail seed at the record time, then SS equilibration at the
/// lag-shifted arrival. Both overwrite `u` (they represent "the state the patient is in",
/// not an additive event). This is the half of the dose-application pass that establishes
/// the *observed* reality — the SS trough — so the reactive driver (#933) runs it BEFORE
/// the decision hook, letting the controller read the pre-dose SS trough. A non-SS (plain
/// bolus / infusion) dose does nothing here. Split out of the combined
/// [`apply_prescheduled_doses_at`] so the driver can interpose the decision hook between
/// the state re-seed and the bolus jump ([`apply_prescheduled_boluses_at`]).
///
/// **Apply-once (#1186).** A dose has up to two distinct events on a lagged SS record —
/// the pre-arrival seed at `dose.time` and the arrival at `dose.time + lag` — so the walk
/// carries one mask per event, both indexed by dose position. `seed_applied` is checked
/// and set here; `applied` (the arrival) is only *checked* here, because the arrival's
/// last sub-step is the bolus jump in [`apply_prescheduled_boluses_at`], which is what
/// sets it — that ordering is what lets the adaptive driver split the arrival around its
/// decision hook (reseed → hook → boluses) and still mark the event exactly once.
#[allow(clippy::too_many_arguments)] // two apply-once masks on top of the dose/PK context
fn reseed_prescheduled_states_at(
    u: &mut [f64],
    ode: &OdeSpec,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    pk_params_flat: &[f64],
    t_start: f64,
    reset_floor: f64,
    opts: &OdeSolverOptions,
    seed_applied: &mut [bool],
    applied: &[bool],
) {
    debug_assert!(seed_applied.len() >= doses.len() && applied.len() >= doses.len());
    // SS + lagtime: at the dose record time (strictly before the lagged arrival) seed
    // the previous interval's steady-state tail so pre-lag observations don't read the
    // empty initial state. Phase II−lagtime is where the prior pulse has decayed to.
    for (i, dose) in doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        if seed_applied[i] {
            continue;
        }
        if ss_seeded_at_record(dose, lag) && (dose.time - t_start).abs() < EVENT_MATCH_TOL {
            seed_applied[i] = true;
            let chz_before = chz_snapshot(ode, u);
            u.copy_from_slice(&ss_state_at_phase(
                ode,
                pk_params_flat,
                dose,
                ss_seed_phase(dose, lag),
                opts,
                &chz_before,
            ));
        }
    }
    // The resets reached at this break (#1588, #1587): a dose whose record precedes the
    // `SS=1` record in (time, row order), or the EVID=3/4 reset in force, does not
    // re-equilibrate at its arrival.
    let gate = ResetGate::at_segment(doses, reset_floor, t_start);
    for (i, dose) in doses.iter().enumerate() {
        // The arrival is one event: this equilibration and the bolus jump in
        // `apply_prescheduled_boluses_at`. Its mask is set there (the last sub-step),
        // so this only reads it.
        if applied[i] {
            continue;
        }
        if (dose.time + dose_lagtimes[i] - t_start).abs() >= EVENT_MATCH_TOL {
            continue;
        }
        if !gate.live(doses, i) {
            continue;
        }
        // A dose seeded at its record (above, #1121) has its state flowed here and adds
        // only the pulse: re-equilibrating would replace the whole state with the
        // periodic trough and erase any dose that landed inside the pre-arrival window
        // (#1275). Only an unseeded (`lag = 0`) SS dose equilibrates at its arrival.
        if ss_equilibrates_at_arrival(dose, dose_lagtimes[i]) {
            let chz_before = chz_snapshot(ode, u);
            u.copy_from_slice(&equilibrate_ss_state(
                ode,
                pk_params_flat,
                dose,
                opts,
                &chz_before,
            ));
        }
    }
}

/// Apply the bolus amount jump (`F·AMT`) of every pre-scheduled dose landing at `t_start`,
/// in dose-list order. A real infusion (or a dose into a built-in input-rate compartment)
/// adds nothing here — it is injected as a `+rate` derivative by `active_infusions` inside
/// `integrate_segment`; an SS dose's jump is applied here too (on top of the trough seeded
/// by [`reseed_prescheduled_states_at`]). This is the *additive event* half of the pass, so
/// the reactive driver (#933) runs it AFTER the decision hook, over the full growing
/// `shadow` dose list — base then controller-injected — so every bolus at `t_start` is
/// applied in one dose-list-ordered pass, matching the frozen-replay verifier bit-for-bit.
fn apply_prescheduled_boluses_at(
    u: &mut [f64],
    ode: &OdeSpec,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    t_start: f64,
    reset_floor: f64,
    applied: &mut [bool],
) {
    debug_assert!(applied.len() >= doses.len());
    // A bolus whose record precedes the `SS=1` record reached at this break, in (time, row
    // order), is wiped by that reset (#1588): a co-timed row *before* the `SS=1` row, and a
    // lagged dose still pending at it. So is one recorded before the EVID=3/4 reset in force
    // (#1587), however late its lagged arrival. It is still marked applied, so the arrival
    // closes.
    let gate = ResetGate::at_segment(doses, reset_floor, t_start);
    for (i, dose) in doses.iter().enumerate() {
        if applied[i] {
            continue;
        }
        if (dose.time + dose_lagtimes[i] - t_start).abs() >= EVENT_MATCH_TOL {
            continue;
        }
        // Set for EVERY dose matched at this break, whichever branch below fires
        // (bolus, input-rate-suppressed, or infusion) — this is the last sub-step of
        // the arrival event, so marking it here closes the whole arrival (#1186).
        applied[i] = true;
        if !is_real_infusion(dose)
            && !input_rate_consumes_cmt(ode, dose.cmt_raw())
            && gate.live(doses, i)
        {
            // dose.cmt is 1-based; state indices are 0-based. A dose into a built-in
            // input-rate compartment (transit/etc.) is delivered as R_in over time by
            // the wrapped RHS — not as a bolus — so it's skipped to avoid double-count.
            // `cmt_idx` maps CMT=0 → compartment 1 (NONMEM default, #899).
            let cmt_idx = dose.cmt_idx();
            // Unreachable from a validated call (`check_dose_compartments` rejects
            // `cmt > n_states` since #899); kept as a bound for hand-built `OdeSpec`s.
            if cmt_idx < ode.n_states {
                u[cmt_idx] += dose_f_bio[i] * dose.amt;
            }
        }
    }
}

/// Apply every pre-scheduled dose landing at `t_start` to the state `u`, in the order the
/// static engine uses: the SS+lagtime tail seed, SS equilibration, then the bolus amount
/// jump (F·AMT) — i.e. [`reseed_prescheduled_states_at`] followed by
/// [`apply_prescheduled_boluses_at`]. Factored out of
/// [`ode_predictions_with_extra_breaks_and_stats`] so the static engine and the reactive
/// driver's base regimen (#702) apply base doses identically (cf. #798 drift). `doses` /
/// `dose_lagtimes` / `dose_f_bio` are parallel and cover only the pre-scheduled doses. The
/// reactive driver calls the two halves separately (interposing the decision hook, #933);
/// every other caller wants the combined pass. Behavior-preserving vs the original single
/// loop for every real regimen — the only reorder is two distinct SS records at the *same*
/// instant (clinically nonsensical: one cannot be at two steady states at once), which no
/// dataset or test carries.
#[allow(clippy::too_many_arguments)] // each is a distinct slice of dose/PK context
fn apply_prescheduled_doses_at(
    u: &mut [f64],
    ode: &OdeSpec,
    doses: &[DoseEvent],
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    pk_params_flat: &[f64],
    t_start: f64,
    reset_floor: f64,
    opts: &OdeSolverOptions,
    // Apply-once masks (#1186), owned by the walk and threaded through both halves.
    seed_applied: &mut [bool],
    applied: &mut [bool],
) {
    reseed_prescheduled_states_at(
        u,
        ode,
        doses,
        dose_lagtimes,
        pk_params_flat,
        t_start,
        reset_floor,
        opts,
        seed_applied,
        applied,
    );
    apply_prescheduled_boluses_at(
        u,
        ode,
        doses,
        dose_lagtimes,
        dose_f_bio,
        t_start,
        reset_floor,
        applied,
    );
}

fn ode_predictions_with_extra_breaks_and_stats(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    extra_breaks: &[f64],
    mut stats: Option<&mut OdeSolverStats>,
    // #570: soft (Hermite-interpolated) sample times — e.g. TTE event/censor times —
    // read off this same Gaussian integration. Sorted ascending. Empty for every
    // ipred-only caller, in which case the second return value is empty and the
    // predictions are bit-identical to before.
    chz_times: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.effective_solver_opts();
    // #570: full state at each `chz_times[i]`, pre-filled NaN so a soft time no segment
    // covered is visibly *unset* rather than silently zero. Times before the first break
    // are overwritten with the seeded state below (#1223), the same fill the dedicated
    // `ode_dense_solve_states` path applies — so a surviving NaN means one thing on both
    // engines: a diverged solve, which the TTE NLL maps to its 1e20 sentinel.
    // `chz_times` is sorted-unique (caller contract), enabling the binary search that
    // maps each segment's soft samples back to their global slot.
    let mut chz_states: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; chz_times.len()];

    // Seed compartments from `init(state) = expr` (zeros when none declared).
    let mut u = ode.initial_state(pk_params_flat);
    let mut predictions = vec![f64::NAN; n_obs];

    // Resolve modeled-RATE doses to concrete (`Fixed`) doses ONCE, before
    // building the timeline/forcing: `resolve_subject_doses` is the single source
    // of truth (#324), so every `subject.doses` read below sees a concrete
    // rate/duration and a coded RATE=-2 (modeled duration `D{cmt}`) cannot reach
    // the integrator unresolved. Borrowed (no clone) for the common all-`Fixed`
    // dataset; parameters are constant across doses on this no-TV path.
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    // Lagtime shifts the effective start (and end) of every dose record; F
    // scales the amount entering the compartment (NONMEM's F·AMT bolus / F·RATE
    // infusion). Both default (lag 0.0, F 1.0) when not declared, so existing
    // models behave identically. Resolved **per dose compartment** so a model
    // with `Fn`/`ALAGn` (issue #369) applies the right value to each route; the
    // common bare-`F`/`lagtime` model gets a uniform vector.
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    // Extended params: slots 0..MAX_PK_PARAMS hold the PK parameters; slots
    // MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry TAFD/TAD anchors for the ODE RHS.
    let first_dose_time = earliest_dose_time(&subject.doses);
    let mut ext_params = seed_ext_params(pk_params_flat, first_dose_time);

    // Build obs_time → indices map. Multiple observations can share a time
    // (e.g. simultaneous PK/PD samples on different CMTs), so each time maps to
    // *all* its observation indices — recording only one would leave the others
    // at their initial NaN.
    let obs_map = build_obs_index_map(&subject.obs_times);

    // Break timeline at lagtime-shifted dose times — and, for infusions,
    // at lagtime-shifted infusion-end times too, so each segment is
    // either fully inside or fully outside every infusion window.
    // #570: also reach any soft (TTE) time past the last observation. This only
    // *appends* a final segment after the last obs — earlier breaks and every
    // observation prediction are untouched, so ipred stays bit-identical.
    let t_last = subject
        .obs_times
        .iter()
        .chain(chz_times.iter())
        .cloned()
        .fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![subject_integration_start(subject)];
    // Every pre-scheduled dose's breaks — the lag-shifted dose time, a real infusion's
    // F-scaled end, the SS+lag record time, per-route absorption-lag onsets, and
    // zero-order window ends. Shared with the reactive driver's base regimen (#702) so
    // the static engine and the frozen-replay verifier segment identically (#798).
    collect_dose_break_times(
        &mut break_times,
        ode,
        subject,
        &dose_lagtimes,
        &dose_f_bio,
        |_| pk_params_flat,
    );
    break_times.push(t_last);
    // System-reset times (EVID=3/4): each is a segment boundary where the state
    // zeros. Empty for every non-reset subject — so the dispatcher's reset-free
    // callers (`ode_predictions` et al.) are byte-identical — and non-empty only
    // on the reset-aware adaptive frozen-replay constant path (#716), which drives
    // this engine with a reset-carrying static subject.
    break_times.extend(subject.reset_times.iter().copied());
    // No-event break points (e.g. the reactive driver's decision times) — they
    // only re-segment the integration, never change state. Drop non-positive /
    // non-finite entries (0.0 is already present; the timeline starts at 0).
    break_times.extend(
        extra_breaks
            .iter()
            .copied()
            .filter(|b| b.is_finite() && *b > 0.0),
    );
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // A non-finite break time makes the whole subject non-finite (#1189) — see
    // [`timeline_has_non_finite`]. `predictions` and `chz_states` are already
    // NaN-prefilled, so returning them here is exactly that outcome.
    //
    // Ordered **before** the #1223 fill below, deliberately: on a broken timeline every record
    // must be repelled, and filling first would hand a pre-start `TENTRY` the seeded state — a
    // scored `H = 0` — on a subject whose integration never happened.
    if abandon_non_finite_timeline(break_times.iter().copied(), stats.as_deref_mut()) {
        return (predictions, chz_states);
    }

    // #1223: a soft (CHZ) time earlier than the first break — a left-truncation `TENTRY`, or an
    // interval-censored `left`, before the subject's first dose or observation. No segment covers
    // it, so without this it keeps its NaN prefill and the TTE likelihood repels the subject with
    // its `1e20` sentinel. One function with `ode_dense_solve_states`, which is the point: see
    // [`fill_prestart_states`] for why there is no second copy to keep in step.
    fill_prestart_states(chz_times, &mut chz_states, break_times.first().copied(), &u);

    // Most-recent system-reset time; `NEG_INFINITY` until the first reset is
    // crossed. Threaded into `integrate_segment` so infusions / zero-order windows
    // opened before the reset stop contributing (mirrors `ode_predictions_event_driven`).
    // Detected in the loop by matching a break against `reset_times` within
    // `EVENT_MATCH_TOL` (not an exact-bit lookup): `reset_times` are added to
    // `break_times` above, so even one merged into a sub-1e-15 neighbour by the dedup is
    // applied at that representative break rather than dropped. Empty `reset_times` (every
    // non-adaptive caller — the dispatcher routes reset subjects elsewhere) makes this a
    // no-op, so those paths stay byte-identical.
    let mut reset_floor = f64::NEG_INFINITY;

    // Apply-once masks (#1186), one entry per dose: `seed_applied` for the SS
    // record-time seed, `applied` for the arrival (equilibrate + bolus). A *derived*
    // break — a route onset, an infusion end — is a multi-term float sum that can land
    // inside `EVENT_MATCH_TOL` of another dose's own break, and every such break used to
    // re-apply that dose. See [`EVENT_MATCH_TOL`] for why no tolerance pair fixes this.
    let mut seed_applied = vec![false; subject.doses.len()];
    let mut applied = vec![false; subject.doses.len()];

    // Walk every break as a left boundary — bound `0..len`, not the old `0..len-1`
    // (#731) — so a dose / observation / CHZ landing on the final break is applied and
    // read post-dose, matching the reactive driver (`ode_predictions_adaptive_impl`)
    // and the per-event replay (`adaptive_frozen_replay_tv`). `break_times` always
    // holds the integration start, so even a degenerate single-instant timeline is a
    // 1-element vector the loop runs once (recording the record from the initial
    // post-dose state, integration skipped by the `k + 1 < len` guard below). The
    // timeline is deduped at 1e-15, so no two adjacent breaks are equal and no break is
    // ever visited twice.
    //
    // Hoisted out of the loop: the records read *at* the current break (#1226). The index
    // is sorted once per subject and binary-searched per break; the buffer is one allocation
    // per subject rather than one per break.
    let obs_index = RecordIndex::new(&subject.obs_times);
    let mut boundary_obs: Vec<usize> = Vec::new();
    let mut auto_state = OdeAutoSwitchState::default();
    for k in 0..break_times.len() {
        let t_start = break_times[k];

        // System reset (EVID=3/4) at t_start: zero the compartments (or re-seed
        // `init(state)=expr`) and record the reset time so infusions / zero-order
        // windows opened earlier stop contributing. Applied BEFORE the dose passes
        // and the observation read below, so a reset sorts ahead of a dose or obs
        // at the same instant — the exact ordering `ode_predictions_event_driven`
        // uses (Reset < Dose < Obs). No-op when the subject carries no resets.
        if subject
            .reset_times
            .iter()
            .any(|&rt| (rt - t_start).abs() < EVENT_MATCH_TOL)
        {
            u = ode.initial_state(pk_params_flat);
            reset_floor = t_start;
        }

        // Apply every pre-scheduled dose landing at t_start — SS tail seed, SS
        // equilibration, then the bolus F·AMT jump — in a single shared pass. This is
        // the exact pass the reactive driver's base regimen (#702) reuses, so the
        // static engine and the frozen-replay verifier apply base doses identically
        // (#798). Infusions add nothing here (injected as a derivative by
        // `active_infusions` below); a reset above already sorted ahead of this.
        apply_prescheduled_doses_at(
            &mut u,
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            pk_params_flat,
            t_start,
            reset_floor,
            &opts,
            &mut seed_applied,
            &mut applied,
        );

        // Record observations read *at* t_start (after the reset/dose passes above) —
        // its whole band `[t_start, t_start + EVENT_MATCH_TOL)`, not just the exact bits
        // (#1226). A lagged arrival is a multi-term float sum, so an observation whose
        // time is nominally the arrival routinely misses it by a few ULP; the old
        // exact-bit lookup handed those to the *preceding* segment, i.e. to the state
        // before the dose was applied.
        obs_index.records_at_break(t_start, &mut boundary_obs);
        record_observations(
            ode,
            &boundary_obs,
            &u,
            pk_params_flat,
            theta,
            eta,
            subject,
            &mut predictions,
            None,
        );

        // #570: a soft (CHZ) time coinciding with this segment's *left* boundary is
        // read here, as the post-dose / initial state `u` — the exact analogue of the
        // observation-at-`t_start` read just above, and of how the dedicated
        // `ode_dense_solve_states` records a `saveat` at a break (post-dose `u`, see its
        // `t_start` handler). `integrate_segment` integrates the *open* interval
        // `(t_start, t_end]`, so without this a CHZ time equal to the integration start
        // (e.g. an interval-censored `left = 0`, or an event at the first dose time)
        // would never be read → NaN → the TTE `1e20` sentinel; and one equal to an
        // *interior* dose time would be read pre-dose. For an interior break this
        // overwrites the previous segment's `t_end` soft sample with the post-dose state
        // — matching the dedicated path, whose next-segment `t_start` handler does the
        // same. `reads_in_segment` below excludes this band, so a soft time is never
        // written twice within one iteration.
        //
        // **One-sided** (#1226). This was a symmetric `(t - t_start).abs() < 1e-12`, the
        // only site in the repo that read the *before* side at a break: a hazard time
        // 1.8e-15 earlier than a lagged arrival was overwritten with the post-dose state
        // while the dedicated `ode_dense_solve_states` kept the pre-dose one, breaking
        // the #570 "shared solve ≡ dedicated path" invariant on the mirror geometry.
        // NONMEM applies a dose record strictly *later* than an observation record
        // second (`nonmem_anchor/lag_arrival_read_after_advan{1,13}`), so before the
        // break is pre-event.
        for (gi, &t) in chz_times.iter().enumerate() {
            if reads_at_break(t, t_start) {
                chz_states[gi] = u.clone();
            }
        }

        // Integrate the open interval `(t_start, t_end]` to the next break, if there is
        // one, recording observations inside it and advancing `u` to `t_end`. The final
        // break time has no successor: its left-boundary discontinuities and `t_start`
        // observation were applied above, but there is nothing left to integrate.
        // Visiting that final break as a left boundary — rather than stopping the loop
        // one short of it (the old `0..len-1` bound, #731) — is what applies a dose
        // landing at the maximum time and reads a coincident observation post-dose,
        // matching the reactive driver (`ode_predictions_adaptive_impl`) and the
        // per-event replay (`adaptive_frozen_replay_tv`), both of which walk every break.
        // `integrate_segment` owns only the integration — the piece a reactive
        // (state-dependent) driver reuses unchanged (#391 S1.2).
        if k + 1 < break_times.len() {
            let t_end = break_times[k + 1];
            // #570: soft (TTE) times recorded off this segment's integration — the
            // complement of the `t_start` band handled above, with an exact `t_end` upper
            // bound so a time inside `t_end`'s own band is read there instead (#1226).
            // `chz_times` is sorted, so this slice is too.
            let seg_chz: Vec<f64> = chz_times
                .iter()
                .copied()
                .filter(|&t| reads_in_segment(t, t_start, t_end))
                .collect();
            let soft = integrate_segment(
                ode,
                &mut u,
                t_start,
                t_end,
                subject,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
                &mut ext_params,
                pk_params_flat,
                theta,
                eta,
                &obs_map,
                &mut predictions,
                stats.as_deref_mut(),
                &mut auto_state,
                &seg_chz,
            );
            // Place each soft sample at its global `chz_times` index (NaN slots left for
            // any time no segment covered).
            for (t, state) in seg_chz.iter().zip(soft) {
                if let Ok(gi) = chz_times.binary_search_by(|x| x.total_cmp(t)) {
                    chz_states[gi] = state;
                }
            }
        }
    }

    // Clamp negative predictions to zero (ODE solver overshoot guard).
    // NaN intentionally NOT clamped — it propagates to a NaN OFV so the
    // outer optimizer rejects the step, matching the analytical path's
    // `conc.max(0.0)` semantic (NaN survives `.max(0.0)` per IEEE 754).
    // This is also what surfaces a missing `OdeReadout::PerCmt` entry as
    // a loud failure rather than a silent zero. (Pre-Phase-2 the clamp
    // included NaN; Copilot's review of #84 caught the inconsistency.)
    clamp_negative_predictions(&ode.readout, &mut predictions);

    (predictions, chz_states)
}

/// Insert a dynamically-discovered break time — an infusion end the reactive
/// driver only learns once the controller issues the infusion — into the sorted
/// `breaks` timeline, collapsing near-duplicates within the **same** `1e-15`
/// tolerance the static timeline uses (see [`ode_predictions`]).
///
/// A break within `1e-15` of an existing one is dropped, so two cases match the
/// static engine's deduped segmentation rather than spuriously re-segmenting:
///  - an infusion that ends *exactly* at a later decision time, and
///  - a degenerate sub-`1e-15`-duration infusion that ends at its own start
///    (collapsing with the decision break — a no-op, mirroring the static
///    engine's `is_real_infusion` `duration > 0` guard).
///
/// Because an infusion end is always strictly after the decision that issued it,
/// the insertion point is always *after* the driver's current position, so a
/// just-issued end never disturbs an already-processed break.
fn insert_break(breaks: &mut Vec<f64>, t: f64) {
    let pos = breaks.partition_point(|&b| b < t);
    if pos < breaks.len() && (breaks[pos] - t).abs() < 1e-15 {
        return;
    }
    if pos > 0 && (t - breaks[pos - 1]).abs() < 1e-15 {
        return;
    }
    breaks.insert(pos, t);
}

/// Out-of-scope-compartment guards shared by the bolus and infusion decision
/// branches of [`ode_predictions_adaptive`]. A controller dose into compartment
/// `cmt` (1-based) is a typed error — never a silent wrong answer — when the
/// compartment is:
///  - **out of range** (`cmt > n_states`);
///  - **fed by a built-in input-rate (absorption) function** — the dose would be
///    double-counted: the trusted static engine delivers it as `R_in` through the
///    wrapped RHS (`input_rate_consumes_cmt`), yet the same forcing is rebuilt
///    from `shadow.doses` here; or
///  - **lagged** — a lag time would be applied with zero delay yet excluded from
///    its own TAD anchor inside `integrate_segment` (whose filter is
///    `d.time + lag <= t_start`).
///
/// On success returns the per-compartment bioavailability `F`, which both
/// branches need (the bolus to scale its state jump, the infusion its window).
/// Single source of truth so the two branches cannot drift the eligibility
/// contract apart.
fn reject_unsupported_dose_compartment(
    ode: &OdeSpec,
    cmt: usize,
    n_states: usize,
    pk_params_flat: &[f64],
    decision_index: usize,
) -> Result<f64, String> {
    if cmt > n_states {
        return Err(format!(
            "decision {decision_index}: dose into compartment {cmt} but the model has \
             {n_states} state(s)"
        ));
    }
    if input_rate_consumes_cmt(ode, cmt) {
        return Err(format!(
            "decision {decision_index}: compartment {cmt} is fed by a built-in input-rate \
             (absorption) function; controller dosing into an input-rate compartment is not \
             supported"
        ));
    }
    let lag = ode.dose_attr_map.lagtime(cmt, pk_params_flat);
    if lag != 0.0 {
        return Err(format!(
            "decision {decision_index}: compartment {cmt} declares a dose lag time ({lag}); \
             lagged controller dosing is not supported"
        ));
    }
    Ok(ode.dose_attr_map.f_bio(cmt, pk_params_flat))
}

/// The data records on the per-event (time-varying) adaptive walk — base dose rows,
/// EVID=2 pk-only rows, observations and EVID=3/4 resets — indexed by time bits and
/// listed in one sorted `times` vector. Empty on the constant-covariate path, where every
/// segment reads the same frozen snapshot and the resolution is a no-op.
///
/// **One record set, two lookups** (#1148), differing only in tie order at a shared
/// instant, which is `ode_predictions_event_driven`'s `kind_order`
/// (`Reset < DoseRecord < PkOnly < Obs`):
///
///   * [`Self::at`] — the **first** record at an instant: the one that terminates, and so
///     governs, the segment arriving there (#1073).
///   * [`Self::in_force`] — the **last** record at or before an instant: the one a
///     decision there reads (covariates, readout PK, an injected dose's `F`).
///
/// A controller-injected dose is not a data record and is never indexed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AdaptiveRecord {
    /// Index into `subject.reset_times` / `event_pk.reset`.
    Reset(usize),
    /// Index into the base regimen (`subject.doses[..n_base]` / `event_pk.dose`).
    Dose(usize),
    /// Index into `subject.pk_only_times` / `event_pk.pk_only`.
    PkOnly(usize),
    /// Index into `subject.obs_times` / `event_pk.obs`.
    Obs(usize),
}

impl AdaptiveRecord {
    /// This record's own per-event PK snapshot.
    fn pk(self, event_pk: &crate::pk::EventPkParams) -> PkParams {
        match self {
            AdaptiveRecord::Reset(r) => event_pk.reset[r],
            AdaptiveRecord::Dose(k) => event_pk.dose[k],
            AdaptiveRecord::PkOnly(m) => event_pk.pk_only[m],
            AdaptiveRecord::Obs(j) => event_pk.obs[j],
        }
    }

    /// This record's own covariate row.
    fn cov(self, subject: &Subject) -> &HashMap<String, f64> {
        match self {
            AdaptiveRecord::Reset(r) => subject.reset_cov(r),
            AdaptiveRecord::Dose(k) => subject.dose_cov(k),
            AdaptiveRecord::PkOnly(m) => subject.pk_only_cov(m),
            AdaptiveRecord::Obs(j) => subject.obs_cov(j),
        }
    }
}

#[derive(Default)]
pub(crate) struct AdaptiveRecordIndex {
    /// Every record time, sorted ascending and deduped — the lookups' search space.
    times: Vec<f64>,
    /// Time bits -> the LAST reset at that instant (the one whose re-seed survives, as
    /// the driver's `rposition` match applies it).
    reset: HashMap<u64, usize>,
    /// Time bits -> index into the base regimen.
    dose: HashMap<u64, usize>,
    /// Time bits -> index into `pk_only_times`.
    pk_only: HashMap<u64, usize>,
    /// Time bits -> first index into `obs_times` at that instant.
    obs: HashMap<u64, usize>,
}

impl AdaptiveRecordIndex {
    /// Build from `subject`'s record grid. The base regimen is `subject.doses[..n_base]`:
    /// the driver's and the IOV build's subject carries only base doses (`n_base =
    /// doses.len()`), while the frozen replay's appends the ledger's controller doses after
    /// them. `resolve_subject_doses_with` maps doses in place, so the indices stay parallel
    /// to `event_pk.dose` and `subject.dose_covariates`.
    pub(crate) fn new(subject: &Subject, n_base: usize) -> Self {
        let mut idx = AdaptiveRecordIndex::default();
        for (r, &t) in subject.reset_times.iter().enumerate() {
            idx.reset.insert(t.to_bits(), r);
            idx.times.push(t);
        }
        for (k, d) in subject.doses.iter().take(n_base).enumerate() {
            idx.dose.entry(d.time.to_bits()).or_insert(k);
            idx.times.push(d.time);
        }
        for (m, &t) in subject.pk_only_times.iter().enumerate() {
            idx.pk_only.entry(t.to_bits()).or_insert(m);
            idx.times.push(t);
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            idx.obs.entry(t.to_bits()).or_insert(j);
            idx.times.push(t);
        }
        idx.times.sort_by(|a, b| a.total_cmp(b));
        idx.times.dedup_by(|a, b| a.to_bits() == b.to_bits());
        idx
    }

    /// The **first** record sitting exactly at `t` in `kind_order`
    /// (`Reset < DoseRecord < PkOnly < Obs`), or `None` at a break that is not a record — a
    /// dose arrival, an infusion end, a zero-order cutoff, a decision. The segment arriving
    /// at `t` terminates at this record, so it is the **governing** lookup.
    fn at(&self, t: f64) -> Option<AdaptiveRecord> {
        let bits = t.to_bits();
        if let Some(&r) = self.reset.get(&bits) {
            return Some(AdaptiveRecord::Reset(r));
        }
        if let Some(&k) = self.dose.get(&bits) {
            return Some(AdaptiveRecord::Dose(k));
        }
        if let Some(&m) = self.pk_only.get(&bits) {
            return Some(AdaptiveRecord::PkOnly(m));
        }
        self.obs.get(&bits).copied().map(AdaptiveRecord::Obs)
    }

    /// The **last** record sitting exactly at `t` in `kind_order` — the reverse of
    /// [`Self::at`]. Processing order at an instant is `Reset`, then the dose, EVID=2 and
    /// observation rows, so the last one processed is what is in force after the instant.
    fn last_at(&self, t: f64) -> Option<AdaptiveRecord> {
        let bits = t.to_bits();
        if let Some(&j) = self.obs.get(&bits) {
            return Some(AdaptiveRecord::Obs(j));
        }
        if let Some(&m) = self.pk_only.get(&bits) {
            return Some(AdaptiveRecord::PkOnly(m));
        }
        if let Some(&k) = self.dose.get(&bits) {
            return Some(AdaptiveRecord::Dose(k));
        }
        self.reset.get(&bits).copied().map(AdaptiveRecord::Reset)
    }

    /// The record **in force** at `t` (#1148): the latest record at or before `t`, within
    /// [`EVENT_MATCH_TOL`], the last in `kind_order` at that instant. `None` when no record
    /// precedes `t` — the caller then reads the subject's baseline covariates and t=0
    /// snapshot (never the first record *ahead*, which is the state seed's rule).
    ///
    /// This is the one resolution behind everything a decision reads: `ctx.covariates`,
    /// the readout PK, and an injected dose's `F`; and the LOCF carry (`last_pk`) advances
    /// through it too. Resets are records here (#1133): a decision after a reset reads the
    /// reset row, as `$PK` ran on it.
    pub(crate) fn in_force(&self, t: f64) -> Option<AdaptiveRecord> {
        let i = self.times.partition_point(|&r| r <= t + EVENT_MATCH_TOL);
        i.checked_sub(1).and_then(|i| self.last_at(self.times[i]))
    }

    /// The PK the LOCF carry (`last_pk`) takes after the walk integrates into `t_end`: the
    /// snapshot of the LAST record at `t_end` in processing order, or `None` when `t_end`
    /// is not a record and the carry stays put. Shared by the driver and its frozen replay,
    /// so the two cannot advance the carry differently.
    fn carried_pk_at(&self, t_end: f64, event_pk: &crate::pk::EventPkParams) -> Option<PkParams> {
        self.last_at(t_end).map(|rec| rec.pk(event_pk))
    }

    /// Time of the record that GOVERNS the segment ending at `t_end` (#1073): `t_end`
    /// itself when a record sits there, otherwise the **next record ahead** — a boundary
    /// that is not a data record (a lagged dose arrival, an infusion end, a zero-order
    /// cutoff, a per-route onset, a decision break) supplies no parameters and merely
    /// subdivides the interval that record terminates. `None` past the final record:
    /// nothing ahead terminates the segment, so the caller keeps the last record that ran
    /// — the trailing rule the static engines share through
    /// [`crate::dosing::governing_record_indices`].
    ///
    /// Observation, EVID=2 and decision times are bit-identical to their break times —
    /// the `#700` survival guard fails loudly otherwise — so the search needs no
    /// tolerance.
    ///
    /// **Reset** times are the exception: the walk matches a reset to its break within
    /// `EVENT_MATCH_TOL`, because the 1e-15 dedup can merge a reset into a neighbouring
    /// break with different bits. Then the segment ending at that break may resolve past
    /// the reset rather than to it. That is harmless: the reset is applied at the start of
    /// the break it was merged into, so it overwrites that segment's state before any
    /// decision or observation reads it, and a decision earlier in the interval still finds
    /// the reset's exact time here as the next record ahead.
    ///
    /// Base **dose** rows are deliberately not added to that guard, because their
    /// commonest collision is legitimate: a dose row co-timed with an observation. The
    /// reader nudges such an observation one ULP earlier to carry file order, and
    /// `break_times`' 1e-15 dedup then merges the pair onto the observation. The
    /// segment ending at that break resolves to the observation here — which is exactly
    /// what production does, since its timeline has the same `Obs`-then-`DoseRecord`
    /// order and the interval between them integrates nothing. A guard would reject
    /// ordinary datasets to protect a sub-ULP case that is already correct.
    fn governing(&self, t_end: f64) -> Option<f64> {
        self.times
            .get(self.times.partition_point(|&r| r < t_end))
            .copied()
    }
}

/// PK governing the segment ENDING at `t_end` on the per-event adaptive walk (#1073).
///
/// **An EVID=3/4 reset is a record here** (#1148), and so governs the segment that ends at
/// it with no record in between. The dense engine differs on purpose, and the asymmetry is
/// recorded at both sites: `ode_predictions_event_driven`'s `is_record` excludes
/// `Kind::Reset` because there the segment ending at a reset is **discarded** — the re-seed
/// overwrites the state before any readout — so which record governs it is unobservable.
/// On the adaptive walk it is not: a **decision** between the last record and the reset
/// reads the state integrated over that segment (`ctx.state`, every monitored signal). The
/// governing rule is "the record that terminates the interval", and since #1133 a reset row
/// is a record (NONMEM runs `$PK` on it; `nonmem_anchor/reset_init_snapshot_J`), so the
/// segment runs under the reset row's snapshot — not under the record *after* the reset,
/// which the walk would otherwise skip ahead to. No prediction moves: the reset still
/// overwrites that state before any observation is read.
///
/// This is the reactive twin of the static engines' end-of-interval resolution, and it
/// is what makes the **degenerate oracle** hold: a controller re-emitting a fixed
/// regimen must equal `simulate()` on that regimen, and `simulate()` routes to
/// `ode_predictions_event_driven`, which governs each segment by the record that
/// terminates it. Carrying the previous record forward here instead was measured at
/// **11 %** on an infusion window ending between two records under a changing
/// covariate.
///
/// Note this is *not* [`AdaptiveRecordIndex::in_force`]: that one answers "which record is
/// in force **at** this instant" for a decision, where the LOCF carry-forward is the
/// causally correct answer — a controller cannot read a covariate that has not been
/// recorded yet.
fn governing_segment_pk_at(
    t_end: f64,
    records: &AdaptiveRecordIndex,
    event_pk: &crate::pk::EventPkParams,
    last_pk: PkParams,
) -> PkParams {
    match records.governing(t_end).and_then(|t| records.at(t)) {
        Some(rec) => rec.pk(event_pk),
        // Past the final record. (`governing` only ever returns a time drawn from the record
        // grid, so `at` cannot miss.)
        None => last_pk,
    }
}

/// Per-record occasion (decision window) on the IOV path (#701), parallel to each record
/// vector of an [`AdaptiveRecordIndex`]. Empty on the non-IOV path.
#[derive(Default)]
struct AdaptiveRecordOcc {
    reset: Vec<Option<usize>>,
    dose: Vec<Option<usize>>,
    pk_only: Vec<Option<usize>>,
    obs: Vec<Option<usize>>,
}

impl AdaptiveRecordOcc {
    /// Every record's occasion from the decision schedule — exactly as the
    /// occasion-aware `event_pk` was built in `run_adaptive_population`.
    fn new(subject: &Subject, n_base: usize, decision_times: &[f64]) -> Self {
        let occ = |t: f64| crate::pk::occasion_of(decision_times, t);
        AdaptiveRecordOcc {
            reset: subject.reset_times.iter().map(|&t| occ(t)).collect(),
            dose: subject
                .doses
                .iter()
                .take(n_base)
                .map(|d| occ(d.time))
                .collect(),
            pk_only: subject.pk_only_times.iter().map(|&t| occ(t)).collect(),
            obs: subject.obs_times.iter().map(|&t| occ(t)).collect(),
        }
    }

    fn of(&self, rec: AdaptiveRecord) -> Option<usize> {
        let (v, i) = match rec {
            AdaptiveRecord::Reset(r) => (&self.reset, r),
            AdaptiveRecord::Dose(k) => (&self.dose, k),
            AdaptiveRecord::PkOnly(m) => (&self.pk_only, m),
            AdaptiveRecord::Obs(j) => (&self.obs, j),
        };
        v.get(i).copied().flatten()
    }
}

/// Occasion twin of [`governing_segment_pk_at`] (#701): the same record, so the eta
/// threaded into the segment carries the same occasion's κ as its PK snapshot.
fn governing_segment_occ_at(
    t_end: f64,
    records: &AdaptiveRecordIndex,
    occ: &AdaptiveRecordOcc,
    last_occ: Option<usize>,
) -> Option<usize> {
    match records.governing(t_end).and_then(|t| records.at(t)) {
        Some(rec) => occ.of(rec),
        None => last_occ,
    }
}

/// PK snapshot to seed the per-event (time-varying) adaptive walk from: the
/// earliest obs / pk-only record's snapshot, mirroring
/// [`ode_predictions_event_driven`]'s init so a covariate-dependent
/// `init(state)=expr` is seeded correctly (#700). Falls back to `fallback` when
/// the subject carries **no** record (e.g. a `TIME`-in-PK subject driven purely by
/// decision times) — the caller passes the t=0 baseline PK there, never a
/// zero-PK default. Shared by the driver and `adaptive_frozen_replay_tv` so the two
/// seed identically and stay bit-aligned; when at least one record exists the
/// fallback is unused, so a differing fallback between the two is harmless.
fn earliest_record_pk(
    subject: &Subject,
    event_pk: &crate::pk::EventPkParams,
    fallback: PkParams,
) -> PkParams {
    let mut best: Option<(f64, PkParams)> = None;
    for (j, &t) in subject.obs_times.iter().enumerate() {
        if best.map_or(true, |(bt, _)| t < bt) {
            best = Some((t, event_pk.obs[j]));
        }
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        if best.map_or(true, |(bt, _)| t < bt) {
            best = Some((t, event_pk.pk_only[m]));
        }
    }
    best.map(|(_, p)| p).unwrap_or(fallback)
}

/// Covariate map a **decision** at time `t` reads on the per-event (time-varying)
/// adaptive path: the row of the record in force at `t`
/// ([`AdaptiveRecordIndex::in_force`] — the latest dose / EVID=2 / obs / EVID=3-4 reset row
/// at or before `t`, the last of a co-timed group in processing order), else `baseline`
/// for a decision before the first record (#700, #1148). The constant-covariate path never
/// calls this.
///
/// The driver resolves its readout PK and an injected dose's `F` from the **same**
/// `in_force` record, so the covariates a controller reads and the parameters behind its
/// signals cannot come from different rows. `run_adaptive_population` calls this for
/// each IOV `decision_pk[g]` snapshot, and `verify_adaptive_snapshots` re-derives it the
/// same way — which is why it is `pub(crate)`, and also why that verifier pins the
/// plumbing, not this rule: it shares the resolver.
pub(crate) fn locf_decision_cov<'a>(
    records: &AdaptiveRecordIndex,
    t: f64,
    subject: &'a Subject,
    baseline: &'a HashMap<String, f64>,
) -> &'a HashMap<String, f64> {
    records.in_force(t).map_or(baseline, |rec| rec.cov(subject))
}

/// The eta to evaluate model expressions with for occasion `occ` (#701): the
/// per-window `[η_bsv | κ_g]` from `eta_occ` when IOV is active and a window is
/// open, else the fixed baseline `eta` (non-IOV runs, and the pre-first-decision
/// window where κ = 0). `read_observable` / `integrate_segment` take `eta`
/// directly — their `[derived]` / observation / ODE-RHS expressions can reference κ,
/// not only the PK params — so the whole eta, not just the PK snapshot, must be
/// occasion-correct.
fn eta_for<'a>(eta_occ: Option<&'a [Vec<f64>]>, eta: &'a [f64], occ: Option<usize>) -> &'a [f64] {
    match (eta_occ, occ) {
        (Some(eo), Some(g)) => eo[g].as_slice(),
        _ => eta,
    }
}

/// Reactive ("adaptive" / feedback) ODE prediction over a single subject (#391
/// S1.3). Walks a fixed `decision_times` schedule, and at each decision lets
/// `controller` read the current state (through the declared `monitors`) and
/// return the [`DoseAction`]s to apply, then carries on integrating with the
/// **same** trusted per-segment engine ([`integrate_segment`]) the static
/// predictor uses.
///
/// Scope of this cut — everything outside it is a typed error, never a silent
/// wrong answer:
/// - **Bolus / Infuse / Hold / Stop** are handled. A zero-amount bolus or
///   infusion is treated as `Hold` (no realized dose recorded). An `Infuse`
///   injects `+rate` over its F-scaled window: its end is inserted as a break
///   (via [`insert_break`]) so each segment is fully inside or outside the
///   window — the invariant [`active_infusions`] relies on (S1.3b). `Stop`
///   discontinues *future* decisions only; an infusion already in flight
///   completes its delivery (a committed dose is not retracted — a true safety
///   halt is a separate, explicit action, tracked as a follow-up).
/// - **Monitors resolve per-mode (S1.5).** `ObserveMode::Ipred` reads the latent
///   state; `ObserveMode::Dv` adds the endpoint's residual draw — `IPRED +
///   ε·√(residual variance)`, clamped at 0 — on the controller-assay substream
///   carried in `assay` (keyed `(subject, replicate, decision, analyte)`). A `Dv`
///   monitor with `assay = None`, or on a compartment with no `[error_model]`, is
///   a typed error (never a fabricated σ). The all-`Ipred` path draws nothing, so
///   it is byte-identical regardless of `assay`.
/// - **Pre-scheduled base regimen (#702, #930).** The base subject MAY carry pre-scheduled
///   doses — a loading / maintenance regimen, including steady-state (`SS=1`) — which
///   are integrated and augmented by the controller's decisions. Base doses occupy the
///   leading `0..n_base` slots of the growing `shadow` dose list, are seeded through the
///   same static-engine break/apply helpers, and appear in the controller's
///   `ctx.history`. Supported on constant-covariate models, and (since #930) on
///   time-varying-covariate models for plain bolus / infusion base doses — each base
///   dose's F is resolved from its own covariate snapshot (`event_pk.dose[k]`). Still a
///   typed error: a base regimen combined with IOV (#931) or system resets (#932), and —
///   under a time-varying covariate — an SS / lagged / input-rate / modeled-rate base
///   dose (a #930 follow-up). A dose-free base subject is the special case `n_base == 0`
///   and is byte-identical to before.
/// - **No lagged or input-rate (absorption) dosing.** Controller dosing into a
///   compartment with a dose lag time, or one fed by a built-in input-rate
///   function, is a typed error (the TAD-anchor and double-count subtleties are
///   deferred, as for the bolus path).
/// - `max_decisions` bounds the schedule (runaway guard); every action is run
///   through [`DoseAction::validate`] before it can reach the integrator.
///
/// The observe-then-dose order is pre-dose (the controller sees the trough at the
/// decision time, then doses). The TAFD anchor is set at the first realized dose,
/// so a TAFD-using model integrated over a segment strictly *before* its first
/// dose would see `NaN` rather than the static predictor's first-dose anchor —
/// immaterial for a controller-driven regimen (no dose ⇒ TAFD undefined).
///
/// Verified contract (see tests): a *state-independent* controller reproduces
/// [`ode_predictions`] on the same realized doses exactly — for boluses *and*
/// infusions — anchoring the reactive bookkeeping to the trusted static engine.
/// The bit-exactness holds when the realized schedule keeps the two engines'
/// segment structure aligned: a dose is realized at every decision (so a held
/// decision does not introduce a break the static dose-list lacks) and the last
/// observation is the global maximum (so neither engine breaks at an interior
/// observation, and the adaptive `t_last = max(obs ∪ decisions)` coincides with
/// the static `t_last = max(obs)`). Outside those conditions a phantom decision
/// break only restarts the integrator on a no-event segment, so predictions are
/// unaffected on the smooth models tested; genuinely reactive/hold regimens are
/// therefore pinned against the closed form instead.
// The cmt-only adaptive driver entry used by the driver's own unit tests: wraps
// each [`MonitorSpec`] into an [`AdaptiveMonitor`] with no compiled `observe`
// expression (every signal resolves via its `cmt`) and adapts a plain
// `Vec<DoseAction>` controller to the engine's [`ControllerDecision`] contract
// (rule provenance is the declarative path's, so `None` here). `#[cfg(test)]`:
// production goes through `_impl` directly — both public entry points supply
// expression-backed monitors and rule-aware controllers — so this is test-only
// scaffolding, not dead production code (#391).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ode_predictions_adaptive(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    decision_times: &[f64],
    monitors: &[MonitorSpec],
    controller: &mut dyn FnMut(&ControllerCtx) -> Vec<DoseAction>,
    max_decisions: usize,
    // Assay-noise capability for `Dv` monitors (#391 S1.5). `None` ⇒ Ipred-only;
    // a `Dv` monitor then errors at its first decision.
    assay: Option<&AssayNoise>,
) -> Result<AdaptiveRun, String> {
    let mons: Vec<AdaptiveMonitor> = monitors
        .iter()
        .map(|spec| AdaptiveMonitor {
            spec,
            observe: None,
        })
        .collect();
    let mut decide = |ctx: &ControllerCtx| ControllerDecision {
        actions: controller(ctx),
        rule: None,
    };
    ode_predictions_adaptive_impl(
        ode,
        pk_params_flat,
        None,
        None,
        None,
        theta,
        eta,
        subject,
        decision_times,
        &mons,
        &mut decide,
        max_decisions,
        assay,
    )
}

/// The core reactive driver. Each [`AdaptiveMonitor`] carries its own optional
/// compiled `observe` expression: `Some(f)` takes the monitor's **latent** value
/// from `f` (the engine-resolved signal for a declarative `[adaptive_dosing]`
/// block, #391 S2), `None` reads `read_observable(cmt)` (the programmatic path,
/// byte-for-byte unchanged). `Dv` still draws its σ from the monitor's `cmt`.
///
/// The controller returns a [`ControllerDecision`] — the dose actions plus the
/// optional label of the `when` rule that fired, recorded as each dose row's
/// `rule_fired`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ode_predictions_adaptive_impl(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    // Per-event PK for the base subject (#700). `Some` ⇒ time-varying-covariate /
    // `TIME`-built-in path: PK is resolved per segment from these snapshots
    // (`event_pk.obs[j]` / `event_pk.pk_only[m]`) instead of the frozen
    // `pk_params_flat`, and observation / pk-only times become segment breaks.
    // `None` ⇒ the constant-covariate path, byte-identical to before (`pk_params_flat`
    // threads through every segment). `event_pk.dose[k]` carries each pre-scheduled base
    // dose's per-dose PK — the covariate at its administration time — used to resolve that
    // base dose's bioavailability F on the TV path (#930); empty on the dose-free path.
    // Controller-injected doses take their PK from the carried-forward (LOCF) snapshot at
    // injection time.
    event_pk: Option<&crate::pk::EventPkParams>,
    // Per-decision occasion PK + per-window eta for the IOV path (#701). `Some` ⇒
    // draw-time κ varies by decision window: `decision_pk[g]` is the PK at decision g
    // under occasion g's κ (drives the pre-dose readout, the injected dose's F, and
    // the LOCF carry into the following segment), and `eta_occ[g]` is the full
    // `[η_bsv | κ_g]` threaded into `read_observable` / `integrate_segment` for every
    // event in window g. `None` on both ⇒ no IOV, byte-identical to the pre-#701 path
    // (the fixed `eta` is used throughout). IOV implies `event_pk = Some` (κ makes PK
    // per-occasion, so obs / pk-only records carry their occasion snapshot).
    decision_pk: Option<&[PkParams]>,
    eta_occ: Option<&[Vec<f64>]>,
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    decision_times: &[f64],
    monitors: &[AdaptiveMonitor],
    controller: &mut dyn FnMut(&ControllerCtx) -> ControllerDecision,
    max_decisions: usize,
    assay: Option<&AssayNoise>,
) -> Result<AdaptiveRun, String> {
    let n = ode.n_states;
    let tv = event_pk.is_some();
    let iov = eta_occ.is_some();

    // --- Preconditions (typed errors, never silent) ----------------------
    // #702/#930/#931/#932: a pre-scheduled base regimen (loading / maintenance dose) IS
    // supported on the constant-covariate path (the controller augments it), on the time-
    // varying-covariate path (#930), and — since #931 — under inter-occasion variability (each
    // base dose's F resolved from its own occasion-κ / covariate snapshot just below). Since
    // #932 it composes with system resets (EVID=3/4) on the CONSTANT-covariate path: the reset
    // zeros the state and lowers `reset_floor` (which already turns off base infusions opened
    // before it — they live in `shadow.doses`, gated by `active_infusions`), and both the
    // reset-aware static verifier (`ode_predictions_with_extra_breaks`) and the reference
    // event-driven engine apply Reset < Dose identically, so the degenerate oracle holds.
    // (An EVID=4 reset+dose row records BOTH a reset and a dose, so its dose is just a base
    // dose landing at the reset instant — zeroed, then re-applied — reaching this path.)
    //
    // Base × reset UNDER a time-varying covariate or IOV (`tv`) stays a typed error: the
    // per-event-PK replay (`adaptive_frozen_replay_tv`) is itself reset-aware, but its
    // composition with a base regimen across a reset is not yet oracle-verified against the
    // reference, so loud-fail rather than risk a silent mis-integration (a #932 follow-up).
    if !subject.doses.is_empty() && subject.has_resets() && tv {
        return Err(
            "ode_predictions_adaptive: a pre-scheduled base regimen combined with system \
             resets (EVID=3/4) is not yet supported under time-varying covariates or \
             inter-occasion variability; #932 supports base × reset on the constant-covariate \
             path only (time-varying-covariate / IOV base × reset is a follow-up)"
                .to_string(),
        );
    }
    if decision_times.len() > max_decisions {
        return Err(format!(
            "decision schedule has {} points, exceeding max_decisions = {} (runaway guard); \
             raise `max_decisions` in the simulate options if the schedule is intentional",
            decision_times.len(),
            max_decisions
        ));
    }
    for am in monitors {
        let m = am.spec;
        if m.cmt == 0 || m.cmt > n {
            return Err(format!(
                "monitor '{}' observes compartment {} but the model has {} state(s)",
                m.name, m.cmt, n
            ));
        }
    }

    // #702: resolve any pre-scheduled base regimen to concrete rate/duration (#324) and
    // capture its per-dose lagtime / bioavailability, exactly as the static engine does.
    // On the (common) dose-free path `resolve_subject_doses` borrows `subject` unchanged,
    // `n_base == 0`, and both vectors are empty — so every base-regimen branch below is a
    // no-op and the reactive path stays byte-identical. When base doses ARE present, only a
    // base × reset subject UNDER a time-varying covariate / IOV is rejected upstream; the
    // constant-covariate path governs both the base and controller doses with this single
    // frozen `pk_params_flat`, while the
    // per-event-PK path (a time-varying covariate #930 and/or IOV #931) overwrites each base
    // dose's F per-occasion from `event_pk.dose[k]` in the block ~40 lines below.
    let resolved_base = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let (base_lagtimes, mut base_f_bio) = subject_dose_attrs(&resolved_base, ode, pk_params_flat);
    let n_base = resolved_base.doses.len();

    // #930/#931: when the driver runs with per-event PK snapshots (`event_pk` is `Some`) —
    // because the model has a time-varying covariate (#930) and/or inter-occasion variability
    // (#931) — resolve each base dose's bioavailability F from its *own* snapshot
    // (`event_pk.dose[k]`: the covariate active at, and the occasion κ of, the dose's
    // administration time) instead of the t=0 `pk_params_flat` the constant path uses. This is
    // symmetric with a controller-injected dose, whose F is fixed from the driver's per-decision
    // LOCF snapshot at injection. `compute_event_pk_params_{into,iov}` populates `event_pk.dose`
    // parallel to the base doses; base × reset is the only base combo rejected above, and it never
    // reaches here.
    //
    // Scope: only a plain FIXED bolus / real-infusion base dose into a PLAIN compartment is
    // supported here. A base dose that is steady-state, lagged, fed by a built-in input-rate
    // (transit / zero-order absorption) function, or carries a modeled (coded) RATE additionally
    // needs its per-dose SS / lag / input-rate / rate-resolution bookkeeping threaded through the
    // hand-rolled frozen-replay engine, which this does not yet do — reject loudly (a narrower
    // follow-up), never a silent snapshot-frozen integration of the delivered dose. (`is_fixed`
    // is tested on the *original* `subject.doses`, before the `resolve_subject_doses` above
    // collapses a coded RATE to `Fixed`.) The default-on frozen-replay verifier is the backstop:
    // even were one of these to slip the guard, the driver and replay would diverge and it would
    // `Err` rather than return a wrong answer.
    if tv && n_base > 0 {
        let ev = event_pk.expect("tv ⇒ event_pk is Some");
        for (k, dose) in resolved_base.doses.iter().enumerate() {
            if !subject.doses[k].is_fixed()
                || dose.ss
                || base_lagtimes[k] != 0.0
                || input_rate_consumes_cmt(ode, dose.cmt_raw())
            {
                return Err(format!(
                    "ode_predictions_adaptive: a pre-scheduled base dose (index {k}) combined with \
                     time-varying covariates or inter-occasion variability must be a plain fixed \
                     bolus or infusion into a plain compartment; steady-state, lagged, built-in \
                     input-rate (transit / zero-order absorption), and modeled-RATE base doses \
                     under a time-varying covariate or IOV are a #930/#931 follow-up"
                ));
            }
            base_f_bio[k] = ode.dose_attr_map.f_bio(dose.cmt_raw(), &ev.dose[k].values);
        }
    }

    // --- Running state ---------------------------------------------------
    let n_obs = subject.obs_times.len();

    // Per-event PK seed for the TV path (#700): the snapshot at the subject's
    // earliest record (obs or pk-only), mirroring `ode_predictions_event_driven`'s
    // init so a covariate-dependent `init(state)=expr` is seeded correctly. A
    // record-free subject (e.g. a `TIME`-in-PK subject driven purely by decision
    // times) falls back to the t=0 baseline `pk_params_flat`, never a zero-PK
    // default (which would integrate CL=V=0 → NaN). Only read when `tv`; the
    // constant path seeds `u` from `pk_params_flat` as before.
    let init_pk: PkParams = match event_pk {
        Some(ev) => {
            let mut base = PkParams::default();
            let m = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
            base.values[..m].copy_from_slice(&pk_params_flat[..m]);
            earliest_record_pk(subject, ev, base)
        }
        None => PkParams::default(),
    };
    // Most-recent real record's PK, carried forward (LOCF) across non-record
    // breaks and updated as the loop crosses obs / pk-only records. Unused (`!tv`).
    let mut last_pk: PkParams = init_pk;
    // IOV twin of `last_pk` (#701): the occasion (decision window) of the most-recent
    // record / decision crossed, carried forward for non-record breaks. Starts at the
    // baseline window (`None`, κ = 0) and advances as the loop crosses decisions and
    // records. Unused (`!iov`).
    let mut last_occ: Option<usize> = None;

    let mut u = if tv {
        ode.initial_state(&init_pk.values)
    } else {
        ode.initial_state(pk_params_flat)
    };
    let mut predictions = vec![f64::NAN; n_obs];
    let mut ledger: Vec<DoseLedgerEntry> = Vec::new();
    let mut decisions: Vec<DecisionLogEntry> = Vec::new();

    // Shadow subject: seeded with the resolved pre-scheduled base regimen (#702; empty
    // on the dose-free path, where `into_owned` just clones `subject`) and then grows as
    // the controller issues realized doses (the #324 pattern). Base doses occupy indices
    // `0..n_base`; injected doses append after. `integrate_segment` reads `shadow.doses`
    // for the TAD anchor and the infusion forcings.
    let mut shadow = resolved_base.into_owned();
    // Bioavailability `F` per dose, parallel to `shadow.doses`. Pre-seeded with the base
    // regimen's F (#702) so the vector stays index-aligned with `shadow.doses` as the
    // controller appends realized doses — the #1 alignment trap. Each injected dose's F
    // is captured at its injection time (from the LOCF PK there): a delivered dose's F is
    // fixed when it is given — later covariate drift must not retroactively rescale it —
    // so segments read F from here rather than re-resolving from the segment PK. On the
    // constant path every F equals `f_bio(cmt, pk_params_flat)`, so the per-segment
    // infusion window is byte-identical to the static engine (empty on the dose-free path).
    let mut injected_f: Vec<f64> = base_f_bio.clone();

    // Extended params: PK params + TAFD/TAD anchors. TAD is set per segment inside
    // `integrate_segment`. TAFD (slot MAX_PK_PARAMS) anchors at the earliest dose: a
    // pre-scheduled base dose when the regimen carries one (#702, mirroring the static
    // engine's `earliest_dose_time` seed), else NaN. `update_tafd_anchor` then LOWERS it at
    // each realized controller dose, so the anchor is `min(earliest base, first controller
    // dose)` — the true global earliest, matching the verifier when a controller dose
    // precedes the earliest base dose (#934; the ascending walk means only the first
    // controller dose can lower a finite base seed).
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if n_base > 0 {
        earliest_dose_time(&shadow.doses)
    } else {
        f64::NAN
    };

    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in shadow.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    // #1073 / #1148: the data records on this walk — base dose rows, EVID=2 pk-only rows,
    // observations and EVID=3/4 resets. `records.at` governs segments; `records.in_force`
    // resolves what a decision reads. Empty on the constant path, where both are no-ops.
    let records = if tv {
        AdaptiveRecordIndex::new(subject, n_base)
    } else {
        AdaptiveRecordIndex::default()
    };
    // Per-record occasion (decision window) for the IOV path (#701), resolved from the
    // decision schedule exactly as the occasion-aware `event_pk` was built in
    // `run_adaptive_population`, so a governed segment threads its record's κ. Empty on
    // the non-IOV path.
    let record_occ = if iov {
        AdaptiveRecordOcc::new(subject, n_base, decision_times)
    } else {
        AdaptiveRecordOcc::default()
    };
    // The t=0 baseline snapshot, read by a decision that precedes every record (#1148).
    let baseline_pk: PkParams = {
        let mut base = PkParams::default();
        let m = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
        base.values[..m].copy_from_slice(&pk_params_flat[..m]);
        base
    };

    // Decision time -> 0-based index, for the in-loop hook.
    let mut decision_index_of: HashMap<u64, usize> = HashMap::new();
    for (i, &t) in decision_times.iter().enumerate() {
        decision_index_of.entry(t.to_bits()).or_insert(i);
    }

    // Break timeline, seeded with the points known up front: 0, every decision,
    // and the last time. Infusion ends are *not* known here — the controller
    // discovers them as it issues infusions — so they are inserted into this
    // (sorted) list dynamically inside the loop (see `insert_break`), which is why
    // the walk below is a `while` over a growing `Vec` rather than a fixed range.
    // With no infusions issued the timeline never grows, so the bolus-only path is
    // byte-identical to before.
    //
    // On the constant-covariate path observations are deliberately NOT break points
    // — they are recorded via `saveat` *inside* a segment, exactly as
    // `ode_predictions` does; breaking at each one would reinitialize the integrator
    // and perturb the step sequence, so the segment structure (and the bit-exact
    // match to the static engine on the same realized doses) is preserved. On the
    // time-varying path (#700) the covariate — hence CL/V/KA — changes only at
    // obs / pk-only records, so those times MUST become segment boundaries for the
    // per-event PK to stay piecewise-constant; the frozen-replay static engine adds
    // the identical breaks, so the two still share `integrate_segment` over
    // identical segments and stay bit-aligned.
    let mut t_last = shadow
        .obs_times
        .iter()
        .chain(decision_times.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    // On the TV path a trailing pk-only (EVID=2) record can be the latest event and
    // becomes a break below, so the horizon must reach it too. Gated by `tv` so the
    // constant path — where pk-only rows are neither breaks nor PK-changing — keeps
    // its exact prior horizon (byte-identical, the shipped canary).
    if tv {
        t_last = shadow.pk_only_times.iter().cloned().fold(t_last, f64::max);
    }
    // #936: the base record's origin — its first obs / pk-only / dose / reset, the same
    // `subject_integration_start` the static engine seeds at (NONMEM's first-record
    // convention). `shadow.doses` is still only the base regimen here. The run's true origin
    // is `min(t_base0, first realized controller dose)`, which is exactly the frozen-replay
    // subject's (base ∪ ledger) start; the walk below holds the state at `init` until then.
    //
    // The seed only has to be a first break at or before the origin, and it must NOT put
    // `t_base0` on the timeline up front when a decision precedes it: once an earlier
    // controller dose is the origin, the static engine has no break at `t_base0` on the
    // constant path (obs are `saveat` points), and an extra one perturbs the step sequence
    // (measured: seeding with `t_base0` alone reddens 8 static-oracle tests). `t_base0` is
    // instead inserted by the walk only if it reaches it un-started (below). Given that, the
    // seed's VALUE below the first decision is immaterial — `0.0` is an equivalent mutation,
    // since a decision is already a break and nothing before the origin is integrated — so
    // the fold is simply "the first decision, or `t_base0` when none precedes it".
    let t_base0 = subject_integration_start(&shadow);
    let t_seed = decision_times.iter().cloned().fold(t_base0, f64::min);
    let mut break_times: Vec<f64> = vec![t_seed, t_last];
    break_times.extend(decision_times.iter().cloned());
    if tv {
        break_times.extend(shadow.obs_times.iter().cloned());
        break_times.extend(shadow.pk_only_times.iter().cloned());
    }
    // System-reset times (EVID=3/4, #716): each is a segment boundary where the state
    // zeros. Since #932 a base regimen reaches here alongside its resets on the constant-
    // covariate path — including an EVID=4 (reset+dose) row, whose dose is a base dose
    // landing at the reset instant (Reset < Dose). Only base × reset UNDER a time-varying
    // covariate / IOV is rejected by the guard above. Empty for a reset-free subject, so
    // the bolus-only path stays byte-identical.
    break_times.extend(subject.reset_times.iter().copied());
    // #702/#930: fold in the pre-scheduled base regimen's breaks via the shared builder so the
    // reactive segmentation matches the static engine's exactly (the frozen-replay oracle).
    // `shadow.doses` here is precisely the base regimen — controller doses are appended
    // later, in the loop — so this passes only the base doses. No-op on the dose-free path.
    // Runs on the constant AND (since #930) the time-varying path: there the base doses'
    // infusion-end breaks are computed from the covariate-resolved `base_f_bio` set above and
    // fold in alongside the obs / pk-only covariate breaks (the dedup below merges any
    // coincident times). base × IOV / reset stays rejected upstream.
    if n_base > 0 {
        collect_dose_break_times(
            &mut break_times,
            ode,
            &shadow,
            &base_lagtimes,
            &base_f_bio,
            |_| pk_params_flat,
        );
        // #1073: a base dose's own **record** is a parameter source — NONMEM runs `$PK`
        // at the dose row and ADVANs to it — so the segment ending there must end there.
        // `collect_dose_break_times` emits only the lag-shifted *arrival* (plus the SS
        // record-time seed), which coincides with the row exactly when `ALAG = 0`; under
        // a lagtime the row needs its own break. Gated on `tv` because only there can two
        // records carry different parameters: on the constant path every snapshot is
        // `pk_params_flat`, so an extra break would change the segmentation without
        // changing the answer, and the byte-identical constant-path canaries would move
        // for nothing. The replay (`adaptive_frozen_replay_tv`) already breaks at every
        // `d.time`.
        if tv {
            break_times.extend(shadow.doses.iter().take(n_base).map(|d| d.time));
        }
    }
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // A non-finite break time makes the subject unsolvable (#1189) — see
    // [`timeline_has_non_finite`]. This driver already has a typed error channel, so
    // it uses that rather than returning a NaN run the caller must re-diagnose. Placed
    // before the #700 exact-bit guards below, whose message would otherwise name the
    // wrong cause for a `NaN` time.
    if abandon_non_finite_timeline(break_times.iter().copied(), None) {
        return Err(
            "ode_predictions_adaptive: a non-finite break time (NaN/infinite dose lagtime, \
             route lag, or infusion duration) — the subject's timeline cannot be ordered"
                .to_string(),
        );
    }

    // #700 review guard: the tolerance dedup above can merge two break times within
    // 1e-15 that are not bit-identical, but `AdaptiveRecordIndex` / `decision_index_of` /
    // `obs_map` resolve records by *exact* bits. If a decision, observation, or
    // pk-only time were merged into a different representative, its exact-bit lookup
    // would silently miss — dropping a per-event PK snapshot or a dose decision, and
    // the frozen-replay verifier (which shares this dedup) could not catch it. Fail
    // loudly instead, honoring this module's "never a silent wrong answer" contract.
    // Integer-hour grids (every test / example) are bit-exact and never trip this.
    if tv {
        let surviving: std::collections::HashSet<u64> =
            break_times.iter().map(|t| t.to_bits()).collect();
        if let Some(t) = decision_times
            .iter()
            .chain(shadow.obs_times.iter())
            .chain(shadow.pk_only_times.iter())
            .copied()
            .find(|t| !surviving.contains(&t.to_bits()))
        {
            return Err(format!(
                "adaptive-dosing (time-varying) event time {t} lies within 1e-15 of another \
                 break time but is not bit-identical, so its per-event PK / decision lookup \
                 would be silently dropped. Align decision, observation, and EVID=2 times to \
                 identical values (integer-valued time grids are unaffected)."
            ));
        }
    }

    // #716/#702: on the constant-covariate path (no tv guard above) decisions are still
    // looked up by exact bits (`decision_index_of`). Adding reset times OR a pre-scheduled
    // base regimen's dose/infusion-end breaks to `break_times` introduces a new collision
    // source: a break within 1e-15 of a decision could make the dedup keep that break's
    // representative and silently drop the decision. Guard it — the same exact-bit contract
    // the tv guard enforces, scoped to the one lookup that matters here. No-op without
    // resets and without a base regimen, so the reset-free dose-free path is unchanged.
    if !tv && (!subject.reset_times.is_empty() || n_base > 0) {
        let surviving: std::collections::HashSet<u64> =
            break_times.iter().map(|t| t.to_bits()).collect();
        if let Some(t) = decision_times
            .iter()
            .copied()
            .find(|t| !surviving.contains(&t.to_bits()))
        {
            return Err(format!(
                "adaptive-dosing decision time {t} lies within 1e-15 of a system-reset or \
                 base-regimen (dose / infusion-end) break time but is not bit-identical, so its \
                 decision lookup would be silently dropped. Align decision, reset (EVID=3), and \
                 base-dose times to identical values (integer-valued time grids are unaffected)."
            ));
        }
    }

    // Running reset floor (`NEG_INFINITY` until the first reset is crossed), threaded
    // into `integrate_segment` so controller-issued infusions / zero-order windows
    // opened before a reset stop contributing — mirroring `ode_predictions_event_driven`.
    // A reset is detected in the loop by matching a break against `reset_times` within
    // the timeline tolerance (`EVENT_MATCH_TOL`), NOT by an exact-bit lookup: `reset_times`
    // are added to `break_times` above, and a reset merged into a sub-1e-15 neighbour by
    // the dedup is then still applied at that representative break (correct to
    // floating-point precision) rather than silently dropped. Resets are coarse episode
    // boundaries, so a tolerance match cannot alias two distinct resets. (Decisions /
    // observations still use exact-bit lookups — they key `HashMap`s — hence the #700
    // survival guard above; resets need no such guard.)
    let mut reset_floor = f64::NEG_INFINITY;

    // Apply-once masks (#1186), parallel to the *growing* `shadow.doses`: base doses
    // occupy `0..n_base` and controller-injected doses append after, so both vectors are
    // `resize`d to `shadow.doses.len()` before each pass. `reseed_prescheduled_states_at`
    // sees only the `..n_base` prefix, so the indices agree with the full-list pass in
    // `apply_prescheduled_boluses_at`.
    let mut seed_applied = vec![false; shadow.doses.len()];
    let mut applied = vec![false; shadow.doses.len()];

    let mut stopped = false;

    // Records read *at* the current break (#1226) — sorted once, hoisted so the walk
    // allocates once.
    let obs_index = RecordIndex::new(&shadow.obs_times);
    let mut boundary_obs: Vec<usize> = Vec::new();
    let mut k = 0;
    let mut auto_state = OdeAutoSwitchState::default();
    // #936: false until the walk reaches the origin — the base record's start `t_base0`, or
    // an earlier realized controller dose (set at the `update_tafd_anchor` call sites). Before
    // it the state stays at `init` and no segment is integrated, so a decision there reads
    // the seeded state, as the static subject (base ∪ ledger) would have it.
    let mut started = false;
    while k < break_times.len() {
        let t_start = break_times[k];
        if t_start >= t_base0 - EVENT_MATCH_TOL {
            started = true;
        }

        // #701 (IOV): a decision break opens its occasion window `g`. Set the LOCF PK
        // + occasion to occasion g's snapshot so the pre-dose readout, the injected
        // dose's F, and any following non-record segment all use occasion g's
        // parameters — not the previous occasion carried in `last_pk`/`last_occ`.
        // Unconditional (every decision break, including holds and post-`Stop`, so
        // every event reads its window's κ). The readout below then reads this snapshot.
        if let (Some(dp), Some(&g)) = (decision_pk, decision_index_of.get(&t_start.to_bits())) {
            last_pk = dp[g];
            last_occ = Some(g);
        }

        // What a decision at `t_start` reads (#1148): ONE record — the one in force there
        // (`records.in_force`) — supplies the covariates (`decision_cov` below), the readout
        // PK, and the PK behind the `F` of any dose injected here. With no record at or
        // before `t_start`, the t=0 baseline. Under IOV the readout is `decision_pk[g]`
        // (set into `last_pk` above): the same record's covariates, under the κ of the
        // DECISION's window. The constant path reads the frozen `pk_params_flat`.
        let in_force = records.in_force(t_start);
        let readout_pk = match event_pk {
            Some(_) if decision_pk.is_some() => last_pk,
            Some(ev) => in_force.map_or(baseline_pk, |rec| rec.pk(ev)),
            None => PkParams::default(),
        };
        let pk_readout: &[f64] = if tv {
            &readout_pk.values
        } else {
            pk_params_flat
        };
        // Eta to evaluate the pre-dose readouts with — paired to `readout_pk`'s
        // occasion (#701): at a decision, `last_occ` is that decision's window, so this is
        // `[η_bsv | κ_g]`. Byte-identical to `eta` on the non-IOV path.
        let readout_eta = eta_for(eta_occ, eta, if iov { last_occ } else { None });

        // System reset (EVID=3) at t_start (#716): zero the compartments (or
        // re-seed `init(state)=expr`) and record the reset floor so infusions /
        // zero-order windows opened before it stop contributing. Applied BEFORE
        // the decision hook reads `u` and before the coincident observation is
        // recorded, so a reset sorts ahead of a dose or obs at the same instant —
        // the ordering `ode_predictions_event_driven` uses (Reset < Dose < Obs).
        // Runs regardless of `stopped`, so a reset after a `Stop` still zeros the
        // state for later observations. No-op for a reset-free subject.
        //
        // The seed reads the RESET ROW'S OWN snapshot (`event_pk.reset[r]`), not the
        // decision-time LOCF carry (#1133): an EVID=3/4 row is a NONMEM data record, so
        // `$PK` runs at it and a covariate-driven `init(...)` restarts on that row's
        // covariates. This is deliberately *not* `pk_readout` — that one is LOCF because a
        // controller must not read a covariate no record has reported yet, which is a
        // statement about the decision hook, not about the state the reset restores. Falls
        // back to `pk_readout` only when no snapshot exists (the constant path, where
        // `event_pk` is `None` and every candidate agrees).
        if let Some(r) = subject
            .reset_times
            .iter()
            // `rposition`, not `position`: if two reset rows ever land within
            // `EVENT_MATCH_TOL` of each other, the dense engine pushes one timeline entry
            // per reset and applies them in order, so the LAST one's seed is the state that
            // survives. Matching that here keeps the adaptive driver, its replay and
            // `predict()` on one answer rather than splitting the degenerate oracle (#1133).
            .rposition(|&rt| (rt - t_start).abs() < EVENT_MATCH_TOL)
        {
            // `tv` is `event_pk.is_some()`, so this is the constant path (`pk_readout` is
            // the frozen `pk_params_flat`, where every candidate snapshot agrees) or the
            // per-event one. The index is asserted rather than defaulted: a short `reset`
            // vector would silently restore the pre-#1133 LOCF carry, which is the defect
            // itself, and `ode_predictions_event_driven` fails loudly on the same
            // condition.
            let seed_pk: &[f64] = match event_pk {
                Some(ev) => {
                    assert_eq!(
                        ev.reset.len(),
                        subject.reset_times.len(),
                        "event_pk.reset must be parallel to subject.reset_times (#1133)"
                    );
                    &ev.reset[r].values
                }
                None => pk_readout,
            };
            u = ode.initial_state(seed_pk);
            reset_floor = t_start;
        }

        // #702/#933: re-seed any pre-scheduled base *steady-state* state landing at
        // t_start — SS equilibration + SS+lag tail — BEFORE the decision hook. The SS
        // trough IS the observed pre-dose reality, so the controller reads it. A base
        // dose's *bolus* jump (F·AMT) is NOT applied here: it is deferred to the shared
        // bolus pass AFTER the hook, so a base bolus coincident with a decision is observed
        // pre-dose (the true trough), symmetric with the controller's own doses (#933 —
        // previously the base bolus landed here, before the hook, and the controller read
        // the post-dose peak). No-op on the dose-free path and for non-SS base doses;
        // constant path only (`pk_params_flat` is the frozen snapshot). Since #932 a reset MAY
        // intervene on the constant path — it zeroed `u` earlier in this same break iteration
        // (Reset < Dose). Correct regardless: this reseed fires only at an SS base dose's own
        // landing time and `copy_from_slice`s the SS equilibrium/tail, re-establishing steady
        // state independent of prior state, so a just-applied reset is correctly superseded; the
        // static engine applies reset-then-reseed in the same order.
        if n_base > 0 {
            // Masks cover the whole (growing) dose list; this pass reads the `..n_base`
            // prefix, so a slice of the same length keeps the indices aligned (#1186).
            seed_applied.resize(shadow.doses.len(), false);
            applied.resize(shadow.doses.len(), false);
            reseed_prescheduled_states_at(
                &mut u,
                ode,
                &shadow.doses[..n_base],
                &base_lagtimes,
                pk_params_flat,
                t_start,
                reset_floor,
                &ode.effective_solver_opts(),
                &mut seed_applied[..n_base],
                &applied[..n_base],
            );
        }

        // --- Decision hook: observe (pre-dose trough) -> decide -> dose. ---
        if !stopped {
            if let Some(&decision_index) = decision_index_of.get(&t_start.to_bits()) {
                // Covariate snapshot in effect at the decision time. On the TV path (#700,
                // #1148) it is the row of the SAME in-force record `readout_pk` came from, so
                // an `observe` / `[scaling]` expression that references a time-varying
                // covariate directly reads the row the PK was built from. The constant path
                // keeps the subject-static map (byte-identical; `obs_cov` equals it there).
                let decision_cov = if tv {
                    in_force.map_or(&shadow.covariates, |rec| rec.cov(subject))
                } else {
                    match obs_map
                        .get(&t_start.to_bits())
                        .and_then(|idxs| idxs.first())
                    {
                        Some(&i) => shadow.obs_cov(i),
                        None => &shadow.covariates,
                    }
                };
                // Resolve each monitored signal at the current (pre-dose) state.
                let mut signals: HashMap<String, f64> = HashMap::new();
                let mut observed: Vec<ObservedSignal> = Vec::with_capacity(monitors.len());
                for am in monitors.iter() {
                    let m = am.spec;
                    // A declarative `[adaptive_dosing]` block (S2) supplies a
                    // compiled `observe` expression for the latent value; absent
                    // one (the programmatic path), read the model's cmt readout.
                    let latent = match am.observe {
                        // `[adaptive_dosing] observe` compiles through the same
                        // `build_y_output_fn` as a `[scaling]` Form C readout, so it can
                        // reference the `TIME` / `T` built-in — enter this decision's time
                        // so it resolves there rather than to the thread-local default
                        // (#1028). The `None` arm's `read_observable` guards itself.
                        Some(f) => {
                            let _time_guard =
                                crate::parser::model_parser::ModelTimeGuard::enter(t_start);
                            f(&u, pk_readout, theta, readout_eta, decision_cov)
                        }
                        None => read_observable(
                            ode,
                            &u,
                            pk_readout,
                            theta,
                            readout_eta,
                            decision_cov,
                            m.cmt,
                            t_start,
                        ),
                    };
                    // Resolve the monitored signal on its own mode: Ipred is the
                    // latent readout; Dv adds the endpoint's assay residual draw on
                    // the controller-assay substream (#391 S1.5).
                    let value = match m.mode {
                        ObserveMode::Ipred => latent,
                        ObserveMode::Dv => {
                            let a = assay.ok_or_else(|| {
                                format!(
                                    "decision {decision_index} at t={t_start}: monitor '{}' \
                                     requests DV (assay-noised) observation but no assay-noise \
                                     capability was supplied (Ipred-only run)",
                                    m.name
                                )
                            })?;
                            // Scale-correct by construction: under `Dv` the
                            // declarative path compiles no `observe` expression, so
                            // `latent` here is the model's own readout for this
                            // monitor's `cmt` (`am.observe == None` ⇒ `read_observable`
                            // above) and σ is `residual_variance_at(cmt, latent)` — both
                            // come from the same model output, so the noised signal is
                            // always on the error model's scale (#391 S2).
                            //
                            // Edge (a): a DV monitor on a compartment with no
                            // residual error model is a typed error, not a guessed σ.
                            let var = (a.resid_var)(m.cmt, latent).ok_or_else(|| {
                                format!(
                                    "decision {decision_index} at t={t_start}: monitor '{}' \
                                     requests DV observation on compartment {} but no [error_model] \
                                     defines residual error there",
                                    m.name, m.cmt
                                )
                            })?;
                            // `has_residual_error_for_cmt` (the gate behind `resid_var`)
                            // requires `sigma` to cover the model's σ indices, so a `Some`
                            // here is panic-free and structurally finite — no downstream
                            // finiteness guard. Value-pathology (a NaN/∞ in `sigma`, a
                            // diverged IPRED) is whole-sim garbage-in, out of scope here.
                            let eps = assay_standard_normal(a.base_seed, decision_index, &m.name);
                            let noised = latent + var.sqrt() * eps;
                            // Edge (b): an assay cannot read below zero; clamp the
                            // noised value at 0 (BLQ-blinding is deferred to Part F).
                            // Gated on the same predicate as the prediction path
                            // (#1039): "cannot read below zero" is a statement about a
                            // compartment amount / concentration, not about a Form C
                            // `[scaling]` readout, which is an arbitrary expression
                            // (change from baseline, z-score, `sqrt(N)*logit(p)`) and is
                            // legitimately signed. Without the gate the *same* model
                            // read `mode = ipred` correctly and `mode = dv` floored at 0,
                            // so a controller thresholding a signed signal silently saw
                            // `0` over the whole negative region.
                            if ode.readout.clamps_negative() {
                                noised.max(0.0)
                            } else {
                                noised
                            }
                        }
                    };
                    signals.insert(m.name.clone(), value);
                    observed.push(ObservedSignal {
                        name: m.name.clone(),
                        value,
                        mode: m.mode,
                    });
                }

                let decision = {
                    let ctx = ControllerCtx {
                        t: t_start,
                        state: &u,
                        covariates: decision_cov,
                        history: &shadow.doses,
                        decision_index,
                        signals: &signals,
                    };
                    controller(&ctx)
                };
                // The `when` rule that produced these actions (declarative path);
                // `None` for a re-issue or a programmatic controller, in which case
                // the ledger records the dose by its route below.
                let rule_fired = decision.rule;
                let actions = decision.actions;

                // Validate the whole action list up front — before any action is
                // applied — and require `Stop` to be the final action. A malformed
                // action anywhere (not only one before the first `Stop`) is a typed
                // error, and a controller that issues actions *after* discontinuing
                // (`[Stop, …]`) is rejected rather than silently truncated, so the
                // decision log can never disagree with the ledger about what ran.
                for (j, action) in actions.iter().enumerate() {
                    action
                        .validate()
                        .map_err(|e| format!("decision {decision_index} at t={t_start}: {e}"))?;
                    if action.is_stop() && j + 1 < actions.len() {
                        return Err(format!(
                            "decision {decision_index} at t={t_start}: Stop must be the final \
                             action, but {} action(s) follow it",
                            actions.len() - j - 1
                        ));
                    }
                }

                // Count realized doses this decision so the log can categorize the
                // outcome (a held / zero-amount decision leaves no ledger row).
                let mut n_dosed = 0usize;
                for action in actions {
                    match action {
                        DoseAction::Bolus { amt, cmt } => {
                            // A zero-amount bolus is a no-op; don't record an empty dose.
                            if amt == 0.0 {
                                continue;
                            }
                            // Out-of-range / input-rate / lagged compartments are typed errors
                            // (never a silent wrong answer) — see the shared guard for why.
                            let f = reject_unsupported_dose_compartment(
                                ode,
                                cmt,
                                n,
                                pk_readout,
                                decision_index,
                            )?;
                            // The bolus jump `u[cmt-1] += f·amt` is NOT applied here; it is
                            // deferred to the shared `apply_prescheduled_boluses_at` pass after
                            // this hook, so every bolus at t_start — base then controller — is
                            // applied in ONE dose-list-ordered pass (matching the frozen-replay
                            // verifier's accumulation order bit-for-bit), and the controller read
                            // the true pre-dose trough above (#933). Recording it in `shadow.doses`
                            // + `injected_f` here is what that pass then applies.
                            update_tafd_anchor(&mut ext_params, t_start);
                            started = true;
                            shadow
                                .doses
                                .push(DoseEvent::new(t_start, amt, cmt, 0.0, false, 0.0));
                            injected_f.push(f);
                            ledger.push(DoseLedgerEntry {
                                subject: shadow.id.clone(),
                                draw: 0,
                                sim: 0,
                                dose_idx: ledger.len(),
                                time: t_start,
                                amt,
                                cmt,
                                rate: 0.0,
                                decision_idx: decision_index,
                                rule_fired: rule_fired
                                    .clone()
                                    .unwrap_or_else(|| "bolus".to_string()),
                                observed_signals: observed.clone(),
                                pre_state: None,
                                post_state: None,
                                f_applied: f,
                            });
                            n_dosed += 1;
                        }
                        DoseAction::Infuse { amt, cmt, rate } => {
                            // A zero-amount infusion is a no-op; don't record an empty dose.
                            if amt == 0.0 {
                                continue;
                            }
                            // Same out-of-scope guards as the bolus path (and for the same
                            // reasons) — see the shared guard. A lagged compartment additionally
                            // shifts the infusion window out of step with its own TAD anchor.
                            let f = reject_unsupported_dose_compartment(
                                ode,
                                cmt,
                                n,
                                pk_readout,
                                decision_index,
                            )?;
                            // Unlike a bolus, an infusion adds nothing to `u` here: it is injected
                            // as a `+rate` derivative term over its window by the next
                            // `integrate_segment` (which reads `shadow.doses` via
                            // `active_infusions`). All this branch must do is make every infusion
                            // *edge* a break so each segment is fully inside or outside the window.
                            // The start (this decision) is already a break; insert the F-scaled
                            // end. `bioavailable_infusion` is the SAME mode-aware window (#419) the
                            // static engine and `active_infusions` use, so the adaptive timeline
                            // reproduces the static segmentation exactly (the degenerate oracle).
                            let dose = DoseEvent::new(t_start, amt, cmt, rate, false, 0.0);
                            let (_, dur_eff) = dose.bioavailable_infusion(f);
                            insert_break(&mut break_times, t_start + dur_eff);
                            update_tafd_anchor(&mut ext_params, t_start);
                            started = true;
                            shadow.doses.push(dose);
                            injected_f.push(f);
                            ledger.push(DoseLedgerEntry {
                                subject: shadow.id.clone(),
                                draw: 0,
                                sim: 0,
                                dose_idx: ledger.len(),
                                time: t_start,
                                amt,
                                cmt,
                                rate,
                                decision_idx: decision_index,
                                rule_fired: rule_fired
                                    .clone()
                                    .unwrap_or_else(|| "infuse".to_string()),
                                observed_signals: observed.clone(),
                                pre_state: None,
                                post_state: None,
                                f_applied: f,
                            });
                            n_dosed += 1;
                        }
                        DoseAction::Hold => {}
                        DoseAction::Stop => {
                            stopped = true;
                            break;
                        }
                    }
                }

                // Log every decision — including holds and no-change, which leave
                // no ledger row. `stopped` was false on entry to this hook (it gates
                // the hook), so its truth here means the `Stop` fired this decision.
                // `observed` is moved in (the ledger rows above already cloned it).
                let outcome = if stopped {
                    DecisionOutcome::Stop { dosed: n_dosed }
                } else if n_dosed > 0 {
                    DecisionOutcome::Dosed { n: n_dosed }
                } else {
                    DecisionOutcome::Hold
                };
                decisions.push(DecisionLogEntry {
                    subject: shadow.id.clone(),
                    draw: 0,
                    sim: 0,
                    decision_idx: decision_index,
                    time: t_start,
                    observed_signals: observed,
                    outcome,
                });
            }
        }

        // #702/#933: apply every bolus landing at t_start — base doses (slots `0..n_base`)
        // then controller-injected (`n_base..`), in dose-list order — in ONE shared pass, so
        // the reactive accumulation order matches the frozen-replay verifier's merged
        // (base ∪ ledger) list bit-for-bit. The SS state was re-seeded before the hook; this
        // adds the F·AMT jump for both plain and SS boluses. Runs regardless of `stopped`: a
        // pre-scheduled base bolus past a controller `Stop` still lands — the base regimen is
        // the patient's standing prescription, independent of the controller (the verifier
        // replays it too; #702 Finding 4). `injected_f` is F parallel to `shadow.doses` (base
        // F pre-seeded, injected F captured at injection); `dose_lagtimes` carries the base
        // lagtimes then 0 for injected doses. On the dose-free path `shadow.doses` is empty
        // until the first controller dose, so this is a no-op there and — once it fires —
        // byte-identical to the in-hook `u[cmt-1] += f·amt` it replaces (same F·AMT, same
        // dose-list order). `dose_lagtimes` is reused by `integrate_segment` below.
        let mut dose_lagtimes: Vec<f64> = base_lagtimes.clone();
        dose_lagtimes.resize(shadow.doses.len(), 0.0);
        // Grow the apply-once masks over any dose the hook just injected (#1186); an
        // injected dose starts unapplied and is marked by this very pass.
        applied.resize(shadow.doses.len(), false);
        seed_applied.resize(shadow.doses.len(), false);
        apply_prescheduled_boluses_at(
            &mut u,
            ode,
            &shadow.doses,
            &dose_lagtimes,
            &injected_f,
            t_start,
            reset_floor,
            &mut applied,
        );

        // Record the observations read *at* t_start (post-dose), mirroring
        // `ode_predictions`' left-boundary recording — its whole `EVENT_MATCH_TOL` band,
        // through the same [`RecordIndex::records_at_break`] the static engine and the frozen
        // replay call, so the three cannot drift (#1226).
        {
            obs_index.records_at_break(t_start, &mut boundary_obs);
            for &obs_idx in &boundary_obs {
                let cmt = shadow.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                // On the TV path each observation reads with its own per-event PK
                // snapshot (`event_pk.obs[obs_idx]`), consistent with the record-at-
                // `t_start` PK that propagated the state into this boundary; the
                // frozen snapshot on the constant path.
                let obs_pk: &[f64] = match event_pk {
                    Some(ev) => &ev.obs[obs_idx].values,
                    None => pk_params_flat,
                };
                // Eta paired to this observation's occasion (#701), consistent with
                // its per-occasion `event_pk.obs` snapshot; baseline `eta` otherwise.
                let obs_eta = eta_for(
                    eta_occ,
                    eta,
                    if iov { record_occ.obs[obs_idx] } else { None },
                );
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    obs_pk,
                    theta,
                    obs_eta,
                    shadow.obs_cov(obs_idx),
                    cmt,
                    // The readout's `TIME` is the user clock, not `t_start` — the
                    // integrator break this observation was keyed to. `obs_map` keys off
                    // `shadow.obs_times`, so `t_start` is the shifted monotonic timeline
                    // (and, thanks to the reader's pre-dose trough nudge, 1 ULP off the
                    // data value even with no resets). Using it here would give
                    // `simulate_adaptive` a different `TIME` than `predict()`/`fit()` for
                    // the same record, and break the frozen-schedule replay oracle's
                    // bit-equality against the static engine (#1028).
                    shadow.readout_time(obs_idx),
                );
            }
        }

        // Integrate the open interval `(t_start, t_end]` to the next break, if
        // there is one. The final break time (== `t_last`) has no successor: its
        // decision hook and left-boundary observation were applied above, but
        // there is nothing left to integrate. Processing that last break — rather
        // than stopping the loop one short of it — is what lets a decision
        // scheduled at the maximum time still fire: its dose reaches the `ledger`
        // and any coincident observation is recorded post-dose.
        // #936: still before the origin with no dose issued here — make `t_base0` the next
        // break so the walk starts integrating exactly there, as the static engine does.
        if !started {
            insert_break(&mut break_times, t_base0);
        }
        if k + 1 < break_times.len() {
            let t_end = break_times[k + 1];

            // PK governing the segment `(t_start, t_end]`: the record that TERMINATES
            // it (NONMEM end-of-interval convention) — `t_end` itself when a record sits
            // there, else the next record ahead (#1073) — on the TV path (#700), the
            // frozen snapshot otherwise. On the TV path it is written
            // into `ext_params`'s PK slots (leaving the TAFD/TAD anchors intact) for
            // the ODE RHS and passed through as the readout PK for any observation
            // `integrate_segment` records internally.
            let seg_pk = match event_pk {
                Some(ev) => governing_segment_pk_at(t_end, &records, ev, last_pk),
                None => PkParams::default(),
            };
            // Occasion governing this segment (#701) — the twin of `seg_pk`'s
            // end-of-interval resolution, so the eta threaded into `integrate_segment`
            // carries the same occasion's κ as `seg_pk`.
            let seg_occ = if iov {
                governing_segment_occ_at(t_end, &records, &record_occ, last_occ)
            } else {
                None
            };
            let seg_eta = eta_for(eta_occ, eta, seg_occ);
            let seg_pk_values: &[f64] = if tv { &seg_pk.values } else { pk_params_flat };
            if tv {
                ext_params[..crate::types::MAX_PK_PARAMS]
                    .copy_from_slice(&seg_pk.values[..crate::types::MAX_PK_PARAMS]);
            }

            // Per-dose lagtimes for the segment (computed once above for the bolus pass and
            // reused here). Base doses (indices `0..n_base`, #702) carry their resolved
            // lagtime; controller-injected doses (`n_base..`) are lag-0 (a nonzero lag is
            // rejected at injection). Dose F comes from `injected_f` — base F pre-seeded,
            // injected F captured at injection time (LOCF PK) — so a later covariate change
            // can't retroactively rescale a delivered dose. Infusions are delivered by
            // `integrate_segment`'s `active_infusions` over any segment they fully span (the
            // base + dynamic injected infusion-end breaks guarantee full containment). On the
            // dose-free path this is all-zeros and `injected_f` empty, byte-identical to before.
            //
            // The pre-segment state is kept for the #1151 / #1535 dose-clock check below, which
            // evaluates the RHS at it. One `n_states` copy per segment.
            // #936: nothing evolves before the origin (see `started`).
            if started {
                let u_start = u.clone();
                integrate_segment(
                    ode,
                    &mut u,
                    t_start,
                    t_end,
                    &shadow,
                    &dose_lagtimes,
                    &injected_f,
                    reset_floor,
                    &mut ext_params,
                    seg_pk_values,
                    theta,
                    seg_eta,
                    &obs_map,
                    &mut predictions,
                    None,
                    &mut auto_state,
                    &[],
                );

                // #1151 / #1535: a segment whose dose clock had no referent — a window before the
                // first realized (or base) dose on a `TAD`/`TAFD`-reading RHS — is refused when
                // the RHS's derivative there depends on that clock's value: through arithmetic
                // (`0.0 * NaN` reaches the *state*, so every later read inherits it) or through a
                // comparison that silently picks a side. Refused here rather than left to the
                // (default-on, but opt-out) frozen-replay verifier, whose message names the
                // symptom and not the cause.
                //
                // Placed AFTER the solve: the helper probes the RHS at the segment's end state as
                // well as its start, and `ext_params` now carries the anchors this segment ran
                // under, so it re-folds nothing. It evaluates the RHS directly and re-solves
                // nothing, so when it returns `None` the walk continues exactly as it would have
                // without it.
                if let Some(msg) = unanchored_dose_clock_error(
                    ode,
                    &shadow,
                    &u_start,
                    &u,
                    &ext_params,
                    t_start,
                    t_end,
                ) {
                    return Err(msg);
                }
            }

            // Advance the LOCF carry: after integrating into `t_end`, the record in force
            // there — **if `t_end` is one** — is the most-recent PK. `last_occ` advances in
            // lockstep (#701) so the next non-record segment reads this occasion's κ.
            //
            // It must be the record AT `t_end`, NOT `seg_pk`. Since #1073 the segment
            // resolution looks FORWARD at a non-record break, so assigning `seg_pk`
            // here would move the carry onto a record the walk has not reached yet. It is
            // also what keeps a break that is not a parameter source (a dose arrival, an
            // infusion end, a zero-order cutoff, a decision) from disturbing the carry at
            // all — matching every other engine, where only a record updates `last_pk` /
            // `last_params`. An EVID=3/4 reset is a record (#1133, #1148); of a co-timed
            // group the LAST in processing order is the one left in force (`last_at`).
            if tv {
                if let Some(pk) = event_pk.and_then(|ev| records.carried_pk_at(t_end, ev)) {
                    last_pk = pk;
                    last_occ = seg_occ;
                }
            }
        }

        k += 1;
    }

    // Clamp negative predictions to zero, matching the static predictor.
    clamp_negative_predictions(&ode.readout, &mut predictions);

    Ok(AdaptiveRun {
        predictions,
        ledger,
        decisions,
    })
}

/// Frozen-schedule replay verifier — the Part-E backbone of #391, default-on in
/// [`crate::api::simulate_adaptive`].
///
/// Rebuild the *static* dose schedule from a reactive run's realized `ledger`,
/// integrate it through the trusted static engine ([`ode_predictions`]) on the
/// same `eta`, and check the reactive trajectory against it. The reactive driver
/// (which re-plans break times as the controller acts) and `ode_predictions`
/// (which plans up front) are different code, so agreement proves **the driver
/// applied every realized dose identically to the static engine** — cleanly
/// separating dose-bookkeeping correctness from controller logic (the latter is
/// captured in the ledger). A divergence localizes a bug to dose application.
///
/// The replay reproduces the reactive driver's **segment structure**, so the
/// check sits at the solver's true round-off floor rather than a held-decision
/// slack. The driver restarts the integrator at *every* decision time (holds and
/// post-`Stop` no-ops included); a naive static replay breaks only at realized
/// doses, so a held decision used to perturb the adaptive RK45 step sequence at
/// the solver's error level and forced a wide (×100) tolerance. Here the
/// `decision_times` are fed back in as no-op breaks
/// ([`ode_predictions_with_extra_breaks`]), so both engines walk the same
/// segments through the same `integrate_segment` — agreement is bit-aligned, and
/// the bound is a small multiple of the solver tolerance, tight enough to catch a
/// sub-percent bookkeeping error (a dropped dose, wrong compartment, or
/// double-applied `F` moves a prediction by O(dose), i.e. tens of percent) while
/// staying clear of pure floating-point accumulation. A default-on verifier must
/// never false-positive on a legitimate run; the exact double-entry / mass-
/// balance bookkeeping checks are S6.
///
/// `decision_times` is the full schedule the run was driven from (not just the
/// realized-dose times) — post-`Stop` decisions are not in `run.decisions` but
/// the driver still breaks at them, so the realized ledger alone cannot
/// reconstruct the segmentation.
///
/// `base_subject` is the subject the run was driven from — its pre-scheduled base
/// regimen (doses / reset_times, if any) survives on it (#702/#932); its observation
/// grid (and any covariates) carry over, and the realized ledger doses are appended
/// after the base doses. The ledger stores nominal `amt`/`rate`
/// (pre-bioavailability), exactly as a `subject.doses` entry, so `F`/lag re-apply
/// downstream identically.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_adaptive_frozen_replay(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    event_pk: Option<&crate::pk::EventPkParams>,
    // Per-decision occasion PK + per-window eta for the IOV path (#701); see
    // `ode_predictions_adaptive_impl`. Reused directly from the driver so the replay
    // applies the identical per-occasion κ over the identical windows — bit-aligned.
    decision_pk: Option<&[PkParams]>,
    eta_occ: Option<&[Vec<f64>]>,
    theta: &[f64],
    eta: &[f64],
    base_subject: &Subject,
    decision_times: &[f64],
    run: &AdaptiveRun,
) -> Result<(), String> {
    // #702/#930: keep the pre-scheduled base regimen (its SS / II / lagtime survive on
    // `base_subject.doses`) and APPEND the controller's realized doses from the ledger,
    // mirroring the reactive driver's `shadow` (base doses first, injected after). The
    // static engine resolves + applies both exactly as the driver did, so the replay
    // stays bit-aligned. On the dose-free path `base_subject.doses` is empty, so this is
    // the prior ledger-only rebuild — byte-identical. A base regimen reaches the constant
    // path (`event_pk` is `None`) and, since #930/#931, the per-event-PK path too (a
    // time-varying covariate and/or IOV); only base × reset is still rejected upstream. On
    // the per-event-PK branch the base doses' F is recomputed from each dose's own snapshot
    // — the covariate active at, and the occasion κ of, its administration time (below) —
    // matching the driver.
    let mut static_subject = base_subject.clone();
    static_subject.doses.extend(
        run.ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
    );

    // On the time-varying path (#700) the static replay must resolve PK per event
    // exactly as the reactive driver did — a single frozen snapshot would diverge.
    // The driver's `event_pk` is reused directly: its obs / pk-only snapshots depend
    // only on the (unchanged) observation grid and covariates, not on the doses, so
    // they are identical for the ledger-rebuilt static subject; each dose's realized
    // F is taken from the ledger. `adaptive_frozen_replay_tv` shares `AdaptiveRecordIndex`
    // + `integrate_segment` with the driver, so the two stay bit-aligned. The
    // constant path keeps the general single-snapshot engine.
    let static_preds = match event_pk {
        Some(ev) => {
            // #930/#931: a base regimen can now ride the per-event-PK path (TV covariate
            // and/or IOV). `static_subject.doses` is the base doses (indices `0..n_base`)
            // followed by the ledger's injected doses, so `dose_f` must be the base doses' F —
            // recomputed here from each base dose's own snapshot (`ev.dose[k]`: its
            // administration-time covariate and occasion κ), identically to the driver —
            // followed by the ledger's realized `f_applied`. On the dose-free base path the
            // base slice is empty, so this is the prior ledger-only vector, byte-identical.
            let mut dose_f: Vec<f64> = base_subject
                .doses
                .iter()
                .enumerate()
                .map(|(k, d)| ode.dose_attr_map.f_bio(d.cmt_raw(), &ev.dose[k].values))
                .collect();
            dose_f.extend(run.ledger.iter().map(|e| e.f_applied));
            adaptive_frozen_replay_tv(
                ode,
                pk_params_flat,
                ev,
                decision_pk,
                eta_occ,
                theta,
                eta,
                &static_subject,
                &dose_f,
                decision_times,
            )
        }
        None => ode_predictions_with_extra_breaks(
            ode,
            pk_params_flat,
            theta,
            eta,
            &static_subject,
            decision_times,
        ),
    };

    if static_preds.len() != run.predictions.len() {
        return Err(format!(
            "frozen replay produced {} prediction(s) but the reactive run has {}",
            static_preds.len(),
            run.predictions.len()
        ));
    }

    // Segment structures now match, so the slack is bounded by floating-point
    // accumulation across the shared integration, not by where holds fall. A
    // small multiple of the solver's own error control covers that while still
    // flagging any sub-percent dose-bookkeeping divergence.
    const REPLAY_TOL_FACTOR: f64 = 8.0;
    // Both engines integrate at the fit-scoped options (#1212), so the agreement band is
    // derived from those and not from the spec's parse-time field.
    let replay_opts = ode.effective_solver_opts();
    let rel_tol = (REPLAY_TOL_FACTOR * replay_opts.reltol).max(1e-9);
    let abs_tol = (REPLAY_TOL_FACTOR * replay_opts.abstol).max(1e-12);
    for (j, (got, want)) in run.predictions.iter().zip(static_preds.iter()).enumerate() {
        // `NaN == NaN` is **not** agreement (#1539). Both engines share the solver, so a
        // state that went non-finite NaN-pads the same rows in both, and counting that as a
        // match would certify a run whose predictions were never integrated. A NaN row is one
        // the replay cannot confirm, whatever produced it; a NaN-vs-finite split falls through
        // to the comparison below, which fails it as a divergence.
        if got.is_nan() && want.is_nan() {
            return Err(format!(
                "prediction {j} is NaN in both the reactive run and the frozen-schedule \
                 replay, so the replay cannot confirm it. A NaN prediction usually means an \
                 ODE state became non-finite (the [odes] right-hand side diverged): check the \
                 right-hand side and the parameter values"
            ));
        }
        let diff = (got - want).abs();
        let tol = abs_tol + rel_tol * want.abs();
        if !(diff <= tol) {
            return Err(format!(
                "prediction {j} diverges from the frozen-schedule replay: \
                 reactive={got}, static={want}, |Δ|={diff} > tol={tol}"
            ));
        }
    }
    Ok(())
}

/// Static, up-front frozen-replay engine for the time-varying-covariate adaptive
/// path (#700). Rebuilds the trajectory from the realized ledger the way the
/// reactive driver did, but plans the entire break timeline **up front** from the
/// frozen ledger (rather than discovering it reactively) — so agreement with the
/// reactive run still proves the driver's dose bookkeeping. It shares
/// [`AdaptiveRecordIndex`] and [`integrate_segment`] with the driver and adds the same
/// obs / pk-only breaks, so the two walk identical segments with identical
/// per-event PK and stay **bit-aligned**. Verifier-only; the constant-covariate
/// path keeps the general single-snapshot [`ode_predictions_with_extra_breaks`].
///
/// `dose_f[i]` is the bioavailability of `subject.doses[i]`, parallel to that list:
/// each pre-scheduled base dose's F (recomputed by the caller from the dose's own
/// covariate snapshot, #930) followed by each ledger dose's realized `f_applied`. A
/// delivered dose's F is taken as given rather than re-derived — F correctness is
/// pinned separately by the degenerate oracle against `ode_predictions_event_driven`.
///
/// IOV (#701): when `eta_occ` / `decision_pk` are `Some`, the replay threads the
/// **same** per-window eta and per-decision occasion PK as the driver — occasion is
/// resolved from `extra_breaks` (the decision schedule) exactly as the driver
/// resolves it, so the two apply identical per-occasion κ over identical windows and
/// stay bit-aligned.
#[allow(clippy::too_many_arguments)]
fn adaptive_frozen_replay_tv(
    ode: &OdeSpec,
    // The driver's frozen t=0 PK snapshot. Used ONLY to place the dose-driven break times
    // (#1188) — route-lag onsets, `zero_order` window edges — from the identical snapshot
    // the driver placed its own from; every integration below reads its segment's own
    // per-event snapshot, never this one.
    pk_params_flat: &[f64],
    event_pk: &crate::pk::EventPkParams,
    decision_pk: Option<&[PkParams]>,
    eta_occ: Option<&[Vec<f64>]>,
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    dose_f: &[f64],
    extra_breaks: &[f64],
) -> Vec<f64> {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let iov = eta_occ.is_some();

    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

    // #1073 / #1148: the records that can govern a segment, mirroring the driver's index
    // (and occasions, #701, resolved from the decision schedule `extra_breaks`).
    // `event_pk.dose` covers the **base** regimen only — `subject.doses` here is the base
    // doses followed by the ledger's controller doses, and a controller dose is not a data
    // record, so it supplies no parameters. `break_times` above already breaks at every
    // `d.time`, so every dose row is reachable as a segment end.
    let n_base = event_pk.dose.len().min(subject.doses.len());
    let records = AdaptiveRecordIndex::new(subject, n_base);
    let record_occ = if iov {
        AdaptiveRecordOcc::new(subject, n_base, extra_breaks)
    } else {
        AdaptiveRecordOcc::default()
    };

    let mut decision_index_of: HashMap<u64, usize> = HashMap::new();
    if iov {
        for (i, &t) in extra_breaks.iter().enumerate() {
            decision_index_of.entry(t.to_bits()).or_insert(i);
        }
    }
    let mut last_occ: Option<usize> = None;

    // Seed PK / state from the earliest record, mirroring the driver's init via the
    // shared `earliest_record_pk` so the two seed identically. A record-free subject
    // yields empty `predictions` here (nothing to verify), so the `default()`
    // fallback — which the driver seeds from the t=0 baseline instead — is moot.
    let init_pk: PkParams = earliest_record_pk(subject, event_pk, PkParams::default());
    let mut last_pk = init_pk;
    let mut u = ode.initial_state(&init_pk.values);
    let mut predictions = vec![f64::NAN; n_obs];

    // Injected doses carry no lag (a nonzero lag is rejected at injection); a *base* dose
    // under a time-varying covariate / IOV is rejected upstream unless its compartment lag
    // is zero (`ode_predictions_adaptive_impl`'s #930/#931 base-dose scope guard), so the
    // driver's own `dose_lagtimes` — `base_lagtimes` followed by zeros — is all-zeros here
    // too. Declared before the break-time build because that build reads it (below).
    let dose_lagtimes = vec![0.0; subject.doses.len()];

    // NB: PK slots are left NaN here (unlike `seed_ext_params`) — the replay
    // overwrites them per-segment from each event's own snapshot before integrating.
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let first_dose_time = earliest_dose_time(&subject.doses);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // The same break set the reactive driver visited: 0, the last time, every dose
    // time, F-scaled infusion ends, obs / pk-only records, and the decision breaks.
    let t_last = subject
        .obs_times
        .iter()
        .chain(extra_breaks.iter())
        // Match the driver: a trailing pk-only record extends the horizon too.
        .chain(subject.pk_only_times.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    // #936: start where the static engine starts — this subject is base ∪ ledger, so its
    // first record or realized dose is the reactive run's origin too.
    let mut break_times: Vec<f64> = vec![subject_integration_start(subject), t_last];
    // Every break a dose list contributes, through the SAME builder the reactive driver
    // folds its base regimen in with (#1188). The hand-rolled loop this replaces pushed
    // only `d.time` and a real infusion's F-scaled end, so it emitted neither a per-route
    // absorption onset (`push_route_lag_break_times`) nor a `zero_order` window's edges
    // (`push_zero_order_break_times`) — and `integrate_segment`'s `active_zero_order_inputs`
    // admits a window's constant rate only for a segment the window *fully contains*. With
    // neither edge bracketed the rate was dropped for every segment that straddled one, so
    // the replay under-delivered the absorbed mass (measured on a lagged `zero_order(dur=2,
    // lag=1.5)` two-dose subject: 0.0 against the static engine's 24.385 at the first
    // in-window sample) and the verifier reported a dose-bookkeeping mismatch that was its
    // own artifact. The two engines are required to agree bit for bit, so they share the
    // builder rather than keeping two copies of the rule (#1171/#1174 fixed two other
    // copies; this was the fifth).
    //
    // `pk_params_flat` is the driver's frozen t=0 snapshot, not a per-segment one, for one
    // reason only: it is the snapshot the driver placed its own base-regimen edges from
    // (`ode_predictions_adaptive_impl`'s `collect_dose_break_times` call), and bit-alignment
    // with the driver is the property this engine exists to establish. It is NOT a claim
    // that the frozen snapshot is the physically right one to place an edge from — if a
    // route lag or `dur` ever becomes covariate-dependent on this path, BOTH engines place
    // their break from the frozen snapshot while `integrate_segment` recomputes the window
    // for containment from the SEGMENT's snapshot, so the edge and the containment boundary
    // move apart and mass is dropped. That inconsistency is pre-existing, symmetric across
    // the two engines (so this verifier cannot see it), and latent: the #930/#931 base-dose
    // guard refuses an input-rate dose under a time-varying covariate, which is the only way
    // to reach it. It belongs to whichever change lifts that guard — recorded here because
    // no test distinguishes the two snapshots today (every fixture's per-event PK is
    // constant, so they are equal; swapping this argument for `init_pk.values` leaves the
    // whole lib suite green, measured).
    //
    // **Asymmetry, recorded here rather than shared away.** The driver runs this builder
    // over its *base* regimen only (controller doses are appended reactively and get an
    // infusion-end break from `insert_break` at injection); this runs it over base ∪
    // ledger. That is wider only for a controller dose into an input-rate compartment,
    // which `reject_unsupported_dose_compartment` refuses, so today the two sets coincide.
    // A future widening of that guard must add the route-lag / zero-order edges at the
    // injection site too, or the pair stops being bit-aligned.
    collect_dose_break_times(
        &mut break_times,
        ode,
        subject,
        &dose_lagtimes,
        dose_f,
        |_| pk_params_flat,
    );
    break_times.extend(subject.obs_times.iter().cloned());
    break_times.extend(subject.pk_only_times.iter().cloned());
    // System-reset times (EVID=3, #716): the same reset breaks the reactive driver
    // added, so the replay zeros the state at the identical instants and stays
    // aligned. Empty for a reset-free subject.
    break_times.extend(subject.reset_times.iter().copied());
    break_times.extend(
        extra_breaks
            .iter()
            .copied()
            .filter(|b| b.is_finite() && *b > 0.0),
    );
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    // A non-finite break time makes the subject non-finite (#1189); `predictions` is
    // NaN-prefilled, matching what the driver this verifies now reports as an `Err`.
    if abandon_non_finite_timeline(break_times.iter().copied(), None) {
        return predictions;
    }
    if break_times.len() < 2 {
        break_times.push(break_times[0]);
    }
    // Running reset floor, mirroring the driver. Detected by the same
    // `EVENT_MATCH_TOL` tolerance match as the driver (resets are added to
    // `break_times` above), so a reset merged into a sub-1e-15 neighbour is still
    // applied at that representative break.
    let mut reset_floor = f64::NEG_INFINITY;

    // Apply-once mask (#1186). This walk is lag-free (every dose lands at `d.time`),
    // so there is no separate SS record-time seed and one mask covers it — but a
    // *derived* break can still land within `EVENT_MATCH_TOL` of a dose time here, and
    // this verifier must stay bit-aligned with the driver it checks.
    let mut applied = vec![false; subject.doses.len()];

    // Records read *at* the current break (#1226) — sorted once, hoisted, as in the driver.
    let obs_index = RecordIndex::new(&subject.obs_times);
    let mut boundary_obs: Vec<usize> = Vec::new();
    let mut auto_state = OdeAutoSwitchState::default();
    for k in 0..break_times.len() {
        let t_start = break_times[k];

        // #701: open a decision window here, mirroring the driver — set the LOCF PK +
        // occasion to this decision's occasion snapshot so the coincident observation,
        // and any following non-record segment, read occasion g's κ.
        if let (Some(dp), Some(&g)) = (decision_pk, decision_index_of.get(&t_start.to_bits())) {
            last_pk = dp[g];
            last_occ = Some(g);
        }

        // System reset (EVID=3) at t_start (#716): zero the state (or re-seed
        // `init(state)=expr`) and record the reset floor — before the boluses and
        // observation below, matching the driver's Reset < Dose < Obs ordering. No-op for
        // a reset-free subject.
        //
        // The seed reads the reset ROW's own snapshot (#1133), the same source the driver
        // uses, so the replay stays bit-aligned with it. Falls back to the LOCF carry only
        // when no reset snapshot exists.
        if let Some(r) = subject
            .reset_times
            .iter()
            // `rposition`, not `position`: if two reset rows ever land within
            // `EVENT_MATCH_TOL` of each other, the dense engine pushes one timeline entry
            // per reset and applies them in order, so the LAST one's seed is the state that
            // survives. Matching that here keeps the adaptive driver, its replay and
            // `predict()` on one answer rather than splitting the degenerate oracle (#1133).
            .rposition(|&rt| (rt - t_start).abs() < EVENT_MATCH_TOL)
        {
            // Indexed, not defaulted — see the driver's matching assert. The two used to
            // fall back to *different* quantities (`pk_readout` there, `last_pk` here), so
            // a length bug would have split the driver from the verifier that exists to
            // check it.
            assert_eq!(
                event_pk.reset.len(),
                subject.reset_times.len(),
                "event_pk.reset must be parallel to subject.reset_times (#1133)"
            );
            u = ode.initial_state(&event_pk.reset[r].values);
            reset_floor = t_start;
        }

        // Apply boluses landing at t_start (lag 0) with their realized F — at EVERY
        // break, including the last. The driver's while-loop processes the final
        // break too (so a decision at the maximum time still doses and its coincident
        // observation is read post-dose); the replay must match, or a dose landing at
        // the last time is silently dropped here (the frozen-replay verifier caught
        // exactly this). Infusions add nothing here — `integrate_segment`'s
        // `active_infusions` delivers them over every segment they span.
        for (i, d) in subject.doses.iter().enumerate() {
            if applied[i] {
                continue;
            }
            if (d.time - t_start).abs() >= EVENT_MATCH_TOL {
                continue;
            }
            applied[i] = true;
            if !is_real_infusion(d) && !input_rate_consumes_cmt(ode, d.cmt_raw()) {
                let cmt_idx = d.cmt_idx();
                if cmt_idx < n {
                    u[cmt_idx] += dose_f[i] * d.amt;
                }
            }
        }

        // Record obs read *at* the left boundary (post-dose) with each observation's own
        // per-event PK (consistent with the state propagated into this boundary). Same
        // [`RecordIndex::records_at_break`] band as the reactive driver this verifier replays,
        // which is what keeps the pair bit-identical (#1028, #1226).
        {
            obs_index.records_at_break(t_start, &mut boundary_obs);
            for &obs_idx in &boundary_obs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                let obs_eta = eta_for(
                    eta_occ,
                    eta,
                    if iov { record_occ.obs[obs_idx] } else { None },
                );
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    &event_pk.obs[obs_idx].values,
                    theta,
                    obs_eta,
                    subject.obs_cov(obs_idx),
                    cmt,
                    // User clock, not the integrator break — see the matching note in
                    // `ode_predictions_adaptive_impl`. This is the replay verifier, so it
                    // is the one path that *must* agree with the static engine bit for
                    // bit (#1028).
                    subject.readout_time(obs_idx),
                );
            }
        }

        // Integrate `(t_start, t_end]` to the next break, if any. The final break has
        // no successor — its dose + observation were applied above.
        if k + 1 < break_times.len() {
            let t_end = break_times[k + 1];
            // Segment PK = the record that TERMINATES `(t_start, t_end]` — itself when
            // a record sits at `t_end`, else the next record ahead (#1073) — via the
            // identical `governing_segment_pk_at` the driver used, so the two stay
            // bit-aligned.
            let seg_pk = governing_segment_pk_at(t_end, &records, event_pk, last_pk);
            // Occasion twin of `seg_pk` (#701), so the threaded eta carries the same
            // occasion's κ — the identical resolution the driver used.
            let seg_occ = if iov {
                governing_segment_occ_at(t_end, &records, &record_occ, last_occ)
            } else {
                None
            };
            let seg_eta = eta_for(eta_occ, eta, seg_occ);
            ext_params[..crate::types::MAX_PK_PARAMS]
                .copy_from_slice(&seg_pk.values[..crate::types::MAX_PK_PARAMS]);

            integrate_segment(
                ode,
                &mut u,
                t_start,
                t_end,
                subject,
                &dose_lagtimes,
                dose_f,
                reset_floor,
                &mut ext_params,
                &seg_pk.values,
                theta,
                seg_eta,
                &obs_map,
                &mut predictions,
                None,
                &mut auto_state,
                &[],
            );

            // Advance the LOCF carry only at an actual record — the identical rule the
            // driver uses, and for the identical reason: since #1073 `seg_pk` looks
            // FORWARD at a non-record break, so carrying it would move `last_pk` onto a
            // record this walk has not reached. The two must agree here or the replay
            // stops being bit-aligned with the run it is verifying — which is why both call
            // the one `carried_pk_at` (#1148 review): past the final record no prediction
            // reads this carry, so a replay-only copy could drift with no test able to see it.
            if let Some(pk) = records.carried_pk_at(t_end, event_pk) {
                last_pk = pk;
                last_occ = seg_occ;
            }
        }
    }

    clamp_negative_predictions(&ode.readout, &mut predictions);
    predictions
}

/// Number of trapezoid panels per inter-decision window for the metrics-only
/// signal-AUC (#391 S2.5b). A fixed *subdivision count* (unit-agnostic — not a step
/// in time units), generous enough that the trapezoid discretization error on a
/// smooth PK curve sits well below the cross-engine solver agreement. This is the
/// AUC machinery's **own** grid: it deliberately does not touch the reactive
/// driver's `saveat`, because the stepper clamps `dt` to land on each save point
/// (`solver.rs`), so adding points there would perturb the bit-aligned trajectory
/// and the default-on frozen-replay verifier.
const ADAPTIVE_AUC_PANELS: usize = 128;

/// Per-(inter-decision)-window AUC of the **latent** monitored signal — the input
/// to the `auc_target_attainment` metric (#391 S2.5b).
///
/// Metrics-only: the exposure never feeds the controller (the `when` rules titrate
/// on the point `signal`), so it is computed here — *after* the reactive run, from
/// the realized `ledger` — rather than inline in the hot loop. Like
/// [`verify_adaptive_frozen_replay`] it rebuilds the static dose schedule from the
/// run's `ledger` and replays it through the trusted dense-state engine
/// ([`ode_dense_solve_states`]); each window is integrated on its **own** uniform
/// sub-grid and reduced with the shared trapezoid rule ([`crate::api::trapezoid`]).
///
/// **Window convention — left-closed / right-open.** Each window
/// `[decision_times[k], decision_times[k+1]]` includes the dose at its left edge
/// `a` (the post-dose state there is the true start of this window's exposure) but
/// **not** the dose at its right edge `b`, which belongs to the next window.
/// [`ode_dense_solve_states`] saves the *post-dose* state at a save point that
/// coincides with a dose time, so the windows cannot share one grid + one solve:
/// that folds the next window's dose into this window's right endpoint — a spurious
/// jump of ≈ ½·Δsignal·(window ⁄ panels) for an instantaneous (bolus) dose (an
/// infusion delivers ≈0 at its start instant and is unaffected, but the convention
/// must be correct for both). So each window is solved against a static subject that
/// keeps only the doses **before** `b` (`time < b`), leaving `b` a plain pre-dose
/// decay point. Controller doses sit on the decision grid (window edges), so for them
/// `time < b` is exactly "at or before `a`" and the window is integrated exactly. A
/// pre-scheduled base maintenance dose (#702) may instead land strictly inside
/// `(a, b)`; it is kept — dropping it would under-count this window's exposure — and
/// is integrated exactly for an infusion / absorption input (whose signal stays
/// continuous), with only the same ≈ ½·Δsignal·(window ⁄ panels) node-placement error
/// as above for the rarer instantaneous bolus into the monitored compartment.
///
/// **Cost — `O(m²)`, deliberately.** This is one dense solve per window, and
/// because [`ode_dense_solve_states`] always starts from `t = 0` (it cannot resume
/// from a saved state), window `k` re-integrates `[0, decision_times[k+1]]` — so the
/// pass is quadratic in the decision count `m` (`1 + 2 + … + (m−1)`), versus `O(m)`
/// for a single shared solve. That is an accepted trade for correctness: the pass
/// runs **only** when `auc_target` is declared and **only after** the reactive run
/// (a per-(subject, replicate) reporting step, never inside the fit/inner loop), so
/// for the intended TDM scale (tens of decisions, a microsecond each) the quadratic
/// factor is negligible. Collapsing it back to `O(m)` would require a solver entry
/// point that resumes from a mid-trajectory state, or a dense readout of the
/// *pre-dose* value at a dose instant — both larger changes to the shared engine,
/// left as a follow-up rather than bundled into the boundary fix.
///
/// Returns one AUC per **closed** window `[decision_times[k], decision_times[k+1]]`
/// (length `decision_times.len() − 1`; empty for a single decision — there is no
/// window to integrate over). The signal is the latent readout the driver itself
/// would resolve: the compiled `observe` expression when present (the `Ipred`
/// path), else the model's `monitor_cmt` readout (the `Dv` path's underlying
/// latent — the AUC is always over the un-noised signal, never the assay draw).
///
/// `base_subject` carries the run's covariates **and** any pre-scheduled base regimen
/// (#702 — a loading / maintenance dose on `subject.doses`): each window is integrated
/// against those base doses (kept via clone, so their `SS`/lagtime/infusion attributes
/// carry over) plus the controller's realized ledger doses, restricted per the window
/// composition below. On the dose-free path `base_subject.doses` is empty, so this is
/// the prior ledger-only rebuild. This pass uses a **single** PK snapshot
/// (`pk_params_flat`), exact only for constant-covariate subjects — a time-varying (or
/// TIME-in-PK) or IOV subject with an `auc_target` is rejected upstream in
/// `run_adaptive_population` (#700/#701), so it never reaches here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adaptive_window_signal_aucs(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    base_subject: &Subject,
    decision_times: &[f64],
    ledger: &[DoseLedgerEntry],
    observe: Option<&OdeOutputFn>,
    monitor_cmt: usize,
) -> Vec<f64> {
    let m = decision_times.len();
    if m < 2 {
        return Vec::new();
    }

    let panels = ADAPTIVE_AUC_PANELS;
    // BSV-only ⇒ the subject's static covariate snapshot applies at every grid time.
    let cov = &base_subject.covariates;

    // One closed window at a time (see "Window convention" above): integrate
    // `[a, b]` against a static subject carrying the base regimen and ledger doses
    // that can influence this window — everything before the right edge `b` (see the
    // per-window composition below) — so the dose at `b` (the next window's) never
    // folds into this window's endpoint.
    (0..m - 1)
        .map(|k| {
            let (a, b) = (decision_times[k], decision_times[k + 1]);

            // Compose the window's static regimen exactly as
            // [`verify_adaptive_frozen_replay`] does: the pre-scheduled base regimen
            // (#702) — CLONED, so each base dose's `SS`/`II`/lagtime/modeled-`RATE`/
            // infusion attributes survive (a `DoseEvent::new` rebuild would silently
            // reset them to a plain `Fixed` bolus) — followed by the controller's
            // realized ledger doses, then restricted to the doses that can influence
            // this window. A dose strictly after the right edge `b` cannot affect
            // `[a, b]` (causality); the dose *at* `b` is the next window's left-edge
            // dose, whose post-dose jump would otherwise fold into this window's right
            // endpoint ([`ode_dense_solve_states`] saves the post-dose state at a save
            // point on a dose time). Both are dropped by `time < b`; doses at or before
            // `a` (state setup) and any base maintenance dose strictly inside `(a, b)`
            // are kept. Controller doses sit on the decision grid (= window edges), so
            // for them `time < b` is byte-identical to the old `<= a` — the pure-
            // controller path is unchanged; only a base regimen (previously dropped by
            // the `sub.doses = ledger` overwrite) is now integrated. The `1e-9` guards
            // float equality at `b`, far below any real decision spacing.
            let mut sub = base_subject.clone();
            sub.doses.extend(
                ledger
                    .iter()
                    .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0)),
            );
            sub.doses.retain(|d| d.time < b - 1e-9);

            // The window's own uniform sub-grid: `panels + 1` points, `grid[0] == a`
            // (post-dose) and `grid[panels] == b` (pre-dose decay).
            let span = b - a;
            let grid: Vec<f64> = (0..=panels)
                .map(|i| a + span * (i as f64) / (panels as f64))
                .collect();

            let states = ode_dense_solve_states(ode, pk_params_flat, theta, eta, &sub, &grid);

            // Latent signal at each grid point (the same readout the driver resolves
            // at a decision), then trapezoid the window.
            let pts: Vec<(f64, f64)> = states
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    let s = match observe {
                        // Same `TIME` guard as the decision-time monitor read above
                        // (#1028), at this grid point's own time.
                        Some(f) => {
                            let _time_guard =
                                crate::parser::model_parser::ModelTimeGuard::enter(grid[i]);
                            f(u, pk_params_flat, theta, eta, cov)
                        }
                        None => read_observable(
                            ode,
                            u,
                            pk_params_flat,
                            theta,
                            eta,
                            cov,
                            monitor_cmt,
                            grid[i],
                        ),
                    };
                    (grid[i], s)
                })
                .collect();
            crate::api::trapezoid(&pts)
        })
        .collect()
}

/// ODE-based predictions with per-event PK parameters (time-varying-covariate
/// aware). Walks the merged dose+obs+pk-only timeline, integrating each
/// segment `[cur_t, t_event]` with the PK params evaluated at `t_event` —
/// the NONMEM end-of-interval / current-record convention (`$PK` runs at
/// every record, then ADVAN propagates to it). A covariate that changes
/// at an event row (dose, obs, or EVID=2) is therefore consumed by the
/// segment terminating at that record.
///
/// The non-TV `ode_predictions` is preserved as a fast path; this function
/// is only invoked from the dispatcher when `subject.has_tv_covariates()`.
///
/// Infusions (`rate > 0`) break the timeline at the infusion's end and are
/// added to the wrapped RHS for any segment they fully span. The
/// infusion-end break carries no NONMEM record, so it doesn't update the
/// "current PK" used to integrate subsequent segments.
pub fn ode_predictions_event_driven(
    ode: &OdeSpec,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
    pk_at_reset: &[PkParams],
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());
    assert_eq!(pk_at_reset.len(), subject.reset_times.len());

    // Resolve modeled-RATE doses to concrete (`Fixed`) doses once (#324), each
    // with its own per-dose PK snapshot `pk_at_dose[k]` (this is the event-driven
    // / time-varying-covariate path). Borrowed (no clone) for the common
    // all-`Fixed` dataset. Single source of truth — see `resolve_subject_doses`.
    let resolved =
        resolve_subject_doses_with(subject, &ode.dose_attr_map, |k| &pk_at_dose[k].values);
    let subject: &Subject = &resolved;

    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.effective_solver_opts();

    // First-dose time anchor for TAFD injection via extended params.
    // fold yields INFINITY when there are no doses; convert to NaN so the ODE
    // RHS injects NaN for TAFD (consistent with sdtab) rather than -∞.
    let first_dose_time_ed = {
        let t = subject
            .doses
            .iter()
            .map(|d| d.time)
            .fold(f64::INFINITY, f64::min);
        if t.is_finite() {
            t
        } else {
            f64::NAN
        }
    };

    // Seed compartments from `init(state) = expr` (zeros when none declared).
    // The init expression folds covariates/eta in via the individual-parameter
    // layer, so evaluate it with the snapshot from the subject's *first record*
    // — the smallest record time across dose / obs / pk-only. Selecting by
    // event kind would wrongly prefer a later dose over an earlier observation
    // when covariates are time-varying (e.g. a pre-dose baseline obs at t=0).
    // Raw record times are used (not lagtime-shifted) since `$PK` order follows
    // the record, not the absorption delay.
    let init_pk: Option<PkParams> = {
        let mut best: Option<(f64, PkParams)> = None;
        let mut consider = |t: f64, p: &PkParams| {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, *p));
            }
        };
        for (k, d) in subject.doses.iter().enumerate() {
            consider(d.time, &pk_at_dose[k]);
        }
        for (j, &t) in subject.obs_times.iter().enumerate() {
            consider(t, &pk_at_obs[j]);
        }
        for (m, &t) in subject.pk_only_times.iter().enumerate() {
            consider(t, &pk_at_pk_only[m]);
        }
        best.map(|(_, p)| p)
    };
    let mut u = match &init_pk {
        Some(p) => ode.initial_state(&p.values),
        None => vec![0.0_f64; n],
    };
    let mut predictions = vec![f64::NAN; n_obs];

    if n_obs == 0 {
        return predictions;
    }

    // Build merged event timeline. Tie-break at the same time:
    //   dose-record < dose-arrival < pk-only < obs < infusion-end
    // — matches the analytical event-driven path for dose/pk-only/obs.
    // Infusion-end sorts last so an obs at the same time as the end of
    // an infusion is recorded with the infusion still contributing
    // (state is continuous; the ordering only affects which segments
    // include the rate in their active set on the next iteration).
    //
    // `DoseRecord` and `Dose` are the two halves of one dose (#1073): the NONMEM
    // *record* sits at `d.time` and is where `$PK` runs, while the state jump
    // happens at the lagged arrival `d.time + ALAG`. With no lagtime they
    // coincide and `DoseRecord` sorts first, so the parameters are in force
    // before the dose lands — bit-identical to the single-event form this
    // replaced. `Dose` keeps its rank ahead of `Obs` so an observation landing
    // exactly on an arrival still reads the post-dose state.
    // No `PartialEq`/`Eq`: every classification goes through `is_record` (or a
    // `matches!`), so there is no `==` that could bypass the predicate and drift
    // from it when a variant is added.
    #[derive(Clone, Copy)]
    enum Kind {
        Reset,
        DoseRecord,
        Dose,
        PkOnly,
        Obs,
        InfusionEnd,
    }
    fn kind_order(k: Kind) -> u8 {
        match k {
            // Reset sorts first so EVID=4 (reset + dose) zeros the state
            // before its own dose lands at the same time.
            Kind::Reset => 0,
            Kind::DoseRecord => 1,
            Kind::Dose => 2,
            Kind::PkOnly => 3,
            Kind::Obs => 4,
            Kind::InfusionEnd => 5,
        }
    }
    /// Whether a timeline entry is a NONMEM **data record** — an event `$PK` runs
    /// at, and therefore a source of segment parameters (#1073).
    ///
    /// A lagged dose *arrival* is not one (its `DoseRecord` at `d.time` is), and
    /// neither is an infusion end, a zero-order cutoff, or a per-route onset.
    ///
    /// `Reset` is a real record — NONMEM runs `$PK` at an EVID=3/4 row — but it is
    /// excluded here because this predicate answers "which record governs the
    /// segment *terminating* at index i", and a reset terminates nothing
    /// observable: the state it would hand on is overwritten by the re-seed.
    /// Admitting it would only change which snapshot the discarded segment ran
    /// on. Where the reset row's `$PK` genuinely matters — the `init(...)`
    /// re-seed — it is read directly from `pk_at_reset` (#1133).
    ///
    /// The adaptive walk deliberately differs (#1148): there a decision can read the
    /// state of the segment ending at a reset, so `AdaptiveRecordIndex` counts the reset
    /// as a record for governance too. Both choices produce the same predictions; see
    /// `governing_segment_pk_at`.
    fn is_record(k: Kind) -> bool {
        matches!(k, Kind::DoseRecord | Kind::PkOnly | Kind::Obs)
    }
    let n_infusion_ends = subject.doses.iter().filter(|d| is_real_infusion(d)).count();
    let mut timeline: Vec<(f64, Kind, usize)> = Vec::with_capacity(
        2 * subject.doses.len()
            + n_obs
            + subject.pk_only_times.len()
            + subject.reset_times.len()
            + n_infusion_ends,
    );
    for (r, &t) in subject.reset_times.iter().enumerate() {
        timeline.push((t, Kind::Reset, r));
    }
    // Per-dose lagtime / bioavailability from each dose's PK snapshot, resolved
    // per dose compartment (`Fn`/`ALAGn`; issue #369) with fallback to the bare
    // `lagtime`/`F` slots. The per-event snapshot also captures variation from
    // time-varying covariates.
    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .zip(pk_at_dose.iter())
        .map(|(d, p)| ode.dose_attr_map.lagtime(d.cmt_raw(), &p.values))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .zip(pk_at_dose.iter())
        .map(|(d, p)| ode.dose_attr_map.f_bio(d.cmt_raw(), &p.values))
        .collect();
    for (k, d) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[k];
        // The dose *record* at its own time — always pushed, lagtime or not
        // (#1073). NONMEM runs `$PK` at the dose row and ADVANs to it, so the
        // dose row's snapshot governs the segment that ENDS there and nothing
        // after it; the interval from here to the lagged arrival belongs to the
        // next record. Skipping this push when `lag == 0` would be wrong in the
        // opposite direction — the arrival is not a parameter source any more, so
        // without the record the segment ending at `d.time` would look forward
        // past the dose to the following record. The zero-length segment that
        // results when `lag == 0` costs nothing (`if t_event > cur_t`).
        timeline.push((d.time, Kind::DoseRecord, k));
        timeline.push((d.time + lag, Kind::Dose, k));
        if is_real_infusion(d) {
            // F-scaled infusion end (#419): rate-defined -> F·duration window.
            let (_, dur_eff) = d.bioavailable_infusion(dose_f_bio[k]);
            timeline.push((d.time + lag + dur_eff, Kind::InfusionEnd, k));
        }
        // End of the *previous* cycle's infusion for a seeded SS dose (#1121),
        // when it is still running at the dose record. Same no-op `InfusionEnd`
        // break an ordinary infusion end gets, and for the same reason: without
        // it a segment could straddle the edge and `active_infusions`'
        // full-containment test would drop the rate over the whole segment.
        if let Some(residual_end) = ss_residual_infusion_end(d, lag, dose_f_bio[k]) {
            timeline.push((residual_end, Kind::InfusionEnd, k));
        }
        // Zero-order absorption cutoff (#504): a dose feeding a `zero_order(dur)`
        // compartment delivers a constant rate over `(0, dur]`, so break at the
        // window end `d.time+lag_cmt+lag_route+dur` exactly like an infusion end (no
        // record, no state change — just a segment boundary so
        // `active_zero_order_inputs`'s full-containment test sees each segment fully
        // inside or outside). The route lag shifts this edge in lock-step with the
        // window `w_start` built by `zero_order_windows` from the same helper.
        if let Some((dur, route_lag)) =
            zero_order_dur_and_lag_for_dose(ode, d, &pk_at_dose[k].values)
        {
            timeline.push((d.time + lag + route_lag + dur, Kind::InfusionEnd, k));
        }
        // Per-route absorption onset (`fn(..., lag=L)`): each route with its own lag
        // switches on at `d.time + lag_cmt + lag_route`, past the `Kind::Dose` break at
        // `d.time + lag_cmt` above — break there (a pure segment boundary, same
        // `Kind::InfusionEnd` no-op the zero-order cutoff uses) so the smooth routes'
        // onset kink resolves exactly and a lagged zero-order window's START is
        // bracketed. No-op for unlagged forcings.
        for forcing in ode.input_rate.iter().filter(|f| f.lag_slot.is_some()) {
            if d.amt > 0.0 && d.cmt_idx() == forcing.cmt {
                let route_lag = forcing.route_lag(&pk_at_dose[k].values);
                timeline.push((d.time + lag + route_lag, Kind::InfusionEnd, k));
            }
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        timeline.push((t, Kind::Obs, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        timeline.push((t, Kind::PkOnly, m));
    }
    timeline.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| kind_order(a.1).cmp(&kind_order(b.1)))
    });
    // A non-finite event time makes the subject non-finite (#1189). This engine
    // dispatches typed events by index and so never re-applies a dose, but a `NaN` time
    // still sorts to the end and its event is silently never reached — the same silent
    // drop the dense engines get, reported the same way (`predictions` is NaN-prefilled).
    if abandon_non_finite_timeline(timeline.iter().map(|e| e.0), None) {
        return predictions;
    }

    // Zero-order windows (#504) read from each dose's **own** PK snapshot
    // (`pk_at_dose[k]`) — the same per-dose source as the timeline cutoff above, so
    // the window edge `w_end` and the per-segment containment test below agree, and
    // the constant rate `F·amt/dur` is fixed at dose time (mass-exact even when
    // `dur` rides a time-varying covariate, where a per-segment recompute would
    // drift). Precomputed once here, then filtered per segment in the loop.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |k, d| {
        zero_order_dur_and_frac_for_dose(ode, d, &pk_at_dose[k].values)
    });
    // The smooth absorption kernels (transit / igd / weibull / first_order) take the same
    // per-dose source (#1569): each dose is absorbed through the kernel, pathway fraction
    // and route lag of its own record, so its delivered mass stays `F·amt·frac` when IOV
    // or a time-varying covariate moves an absorption parameter mid-absorption. Built once
    // here; the segment loop below only chooses the disposition.
    let prepared = PreparedForcings::per_dose(
        ode,
        subject.doses.len(),
        |k| &pk_at_dose[k].values[..],
        InputRateForcing::prepare,
    );

    // Parameters for the segment ENDING at each timeline entry (#1073).
    //
    // NONMEM evaluates `$PK` at every record and then ADVANs *to* that record, so
    // a segment is governed by the record that TERMINATES it. An entry that is not
    // a record — a lagged dose arrival, an infusion end, a zero-order cutoff, a
    // per-route onset — supplies no parameters of its own: it merely subdivides
    // the interval its enclosing record terminates, and every piece of that
    // interval runs on that record's snapshot.
    //
    // Resolved by the shared [`crate::dosing::governing_record_indices`] rule rather
    // than at the point of use, because the answer for a non-record lies *ahead* of
    // it in the walk. All four engines call that one helper so the resolution — and
    // in particular its trailing-tail rule — cannot drift between them.
    //
    // Reusing `last_pk` for these — the previous record — is what this replaced,
    // and it is wrong by exactly one record: measured against NONMEM 7.6.0 it puts
    // a 4.2 % error on the predictions after an infusion end that falls between
    // two records under a changing covariate, and 14.9 OFV on a lagged second
    // dose whose arrival crosses one.
    let governing_record =
        crate::dosing::governing_record_indices(timeline.len(), |i| is_record(timeline[i].1));
    let record_pk_at = |q: usize| -> PkParams {
        let (_, kind, idx) = timeline[q];
        match kind {
            Kind::DoseRecord => pk_at_dose[idx],
            Kind::PkOnly => pk_at_pk_only[idx],
            // Unreachable while `is_record` excludes `Reset`, and spelled out anyway so the
            // two cannot drift: admitting `Reset` there without this arm would send a reset
            // index into `pk_at_obs` — a wrong snapshot, or a panic when the subject has
            // more resets than observations (#1133).
            Kind::Reset => pk_at_reset[idx],
            _ => pk_at_obs[idx],
        }
    };

    let mut cur_t = timeline[0].0;
    // Most-recent NONMEM record's PK params, used to integrate segments
    // ending at an infusion-end (which is not a record and carries no PK).
    // Seed last_pk with the first record's snapshot (not zeroed defaults) so a
    // reset that is itself the first event — e.g. an EVID=4 reset+dose at t=0 —
    // re-applies init from real parameters rather than zeros. Updated as
    // dose/obs/pk-only records are processed.
    let mut last_pk: PkParams = init_pk.unwrap_or_default();
    // Most-recent system-reset time (EVID=3/4); `NEG_INFINITY` until the
    // first reset. Infusions started before it are no longer active.
    let mut reset_floor = f64::NEG_INFINITY;
    let mut auto_state = OdeAutoSwitchState::default();

    for (i, &(t_event, kind, idx)) in timeline.iter().enumerate() {
        // PK params for the segment [cur_t, t_event] are evaluated AT the record
        // that TERMINATES the interval (NONMEM end-of-interval / current-record
        // convention — `$PK runs at every record, then ADVAN propagates to it`).
        // For a record that is itself; for a non-record it is the next record
        // ahead (#1073), which `governing_record` resolved above.
        let pk_now: PkParams = match kind {
            // The segment ending at a reset is DISCARDED — the reset arm below
            // overwrites `u` before any readout — so whichever record governs it
            // cannot reach a prediction, and `last_pk` here is arithmetic that
            // never leaves the loop. The reset's own snapshot does matter, but
            // only to the re-seed, which reads `pk_at_reset[idx]` directly
            // (#1133). Keeping this arm explicit stops the reset falling into the
            // `_` branch and taking the *next* record ahead, which would be the
            // wrong answer if this value ever became observable.
            Kind::Reset => last_pk,
            // `None` only for a subject with no record anywhere in its timeline,
            // which produces no prediction; `last_pk` is the only snapshot that
            // exists there.
            _ => governing_record[i].map_or(last_pk, &record_pk_at),
        };

        if t_event > cur_t {
            // Build extended params for this segment: slots 0..MAX_PK_PARAMS
            // are pk_now.values; slots MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry
            // the TAFD/TAD anchors for TIME/TAFD/TAD injection in the ODE RHS.
            //
            // `tad_anchor_for`, not a fold written out here (#1126). This walk carried
            // its own copy of the arithmetic until then — same rule, spelled twice, on
            // the two production ODE predictors that are selected *per subject* on
            // `has_resets()`. So a divergence between them would make two subjects of the
            // same model and the same data shape read different `TAD`s, and the copies
            // had to be edited in lockstep to add the seeded-SS pre-arrival referent.
            // It is bit-identical to what stood here: the fold shifts each dose by its
            // own resolved lag, and the pre-any-arrival fallback is `min_k(d.time + lag_k)`,
            // the same subject-wide earliest lagged arrival this walk used to compute for
            // itself in a separate binding above the loop (now deleted with the copy).
            //
            // Two properties of that fallback are load-bearing, and both were learned the
            // hard way:
            //
            //   * **Finite.** A NaN anchor multiplies into the state (`0.0 * NaN = NaN`)
            //     and poisons every later prediction of any `[odes]` RHS reading `TAD`,
            //     turning a finite fit into the 1e20 objective sentinel.
            //   * **Segment-invariant.** The anchor is recomputed per segment, so anchoring
            //     at `cur_t` restarts `TAD` at zero at each record inside the pre-arrival
            //     window — a sawtooth whose shape depends on the sampling mesh, not on the
            //     model. Measured: two subjects identical but for one extra observation in
            //     that window diverged by 4.2e-4 at *every* later time, the error injected
            //     once and then carried multiplicatively. A prediction must not move
            //     because someone took an extra sample.
            let last_dose_eff_ed = tad_anchor_for(&subject.doses, &dose_lagtimes, cur_t);
            let mut ext_params_ed = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
            ext_params_ed[..crate::types::MAX_PK_PARAMS]
                .copy_from_slice(&pk_now.values[..crate::types::MAX_PK_PARAMS]);
            ext_params_ed[crate::types::MAX_PK_PARAMS] = first_dose_time_ed;
            ext_params_ed[crate::types::MAX_PK_PARAMS + 1] = last_dose_eff_ed;

            // Wrap the user RHS so any infusion fully spanning
            // [cur_t, t_event] contributes `+rate` to its compartment.
            let active = active_infusions(
                &ode.input_rate,
                &subject.doses,
                cur_t,
                t_event,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
                ode.n_states,
            );
            // Zero-order absorption windows covering [cur_t, t_event] (#504),
            // reset-aware via the same `reset_floor` (a window opened pre-reset
            // is off). Constant `F·amt/dur`, injected like a spanning infusion.
            // `zo_windows` is precomputed once from the per-dose `pk_at_dose`
            // snapshots (below), the same source as the timeline's cutoff break —
            // so the window edge and the containment boundary can't drift apart,
            // and the constant rate is fixed at dose time (mass-exact under
            // time-varying covariates).
            let zero_order =
                active_zero_order_inputs(&zo_windows, &subject.doses, cur_t, t_event, reset_floor);
            let wrapped_rhs = wrap_rhs_with_forcings(
                ode,
                &subject.doses,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
                cur_t,
                &prepared,
                InfusionInput::Spanning(active),
                &zero_order,
            );
            let saveat = vec![t_event];
            let (sol, _) = solve_ode_dense_with_auto_state(
                &wrapped_rhs,
                &u,
                (cur_t, t_event),
                &ext_params_ed,
                &saveat,
                &[],
                &opts,
                None,
                &mut auto_state,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }

        match kind {
            Kind::DoseRecord => {
                // The dose row itself: a NONMEM record, so `$PK` ran here and this
                // snapshot becomes current. No state change — that happens at the
                // lagged arrival below (#1073).
                last_pk = pk_now;
                // …with one exception: a *steady-state* dose carrying a lagtime
                // loads its compartments HERE, at the record, not at the arrival
                // (#1121). NONMEM runs `$PK` at the dose row, fills the
                // compartments with the periodic solution, and then ADVANs to the
                // lagged arrival under the record that terminates that interval.
                // Equilibrating at the arrival instead — which is what this walk
                // did — computes the trough throughout under the dose row's
                // snapshot, so the pre-arrival window gets the wrong elimination
                // whenever a covariate changes inside it.
                //
                // Phase `II − lag` is where the *previous* cycle's pulse (at
                // `d.time + lag − II`) has decayed to by the record time. The
                // snapshot is the dose row's own, never `pk_now`: like `F`, `ALAG`
                // and `D{n}`, the steady state is a property of the record that
                // declares it, and `pk_now` is the *next* record's after #1073.
                // From here the walk's ordinary integration carries the state to
                // the arrival, where only the pulse is applied.
                let d = &subject.doses[idx];
                if ss_seeded_at_record(d, dose_lagtimes[idx]) {
                    let chz_before = chz_snapshot(ode, &u);
                    u = ss_state_at_phase(
                        ode,
                        &pk_at_dose[idx].values,
                        d,
                        ss_seed_phase(d, dose_lagtimes[idx]),
                        &opts,
                        &chz_before,
                    );
                }
            }
            Kind::Dose => {
                let d = &subject.doses[idx];
                // Dose *attributes* are properties of the dose row, so they read
                // that row's own snapshot (`pk_at_dose[idx]`) and never `pk_now`,
                // which after #1073 is the NEXT record's. Before the split the two
                // were the same object and the distinction did not show.
                let dose_pk = &pk_at_dose[idx];
                // A dose whose record precedes the `SS=1` record reached here, in
                // (time, row order), was wiped by that reset (#1588): no arrival at all.
                // The seed at a lagged `SS=1` record sorts before every co-timed arrival
                // (`DoseRecord < Dose`), so list order alone cannot express this. Nor
                // does a dose recorded before the EVID=3/4 reset in force arrive
                // (#1587); `Reset < Dose` has already raised `reset_floor` for a
                // co-timed reset.
                if !ResetGate::at_segment(&subject.doses, reset_floor, cur_t)
                    .live(&subject.doses, idx)
                {
                    continue;
                }
                // Steady-state (SS=1) dose: reset state and load with the
                // SS amount from the infinite-past pulse train before the
                // SS dose's own pulse is applied below. See
                // `equilibrate_ss_state` for the per-cycle scheme.
                //
                // Skipped when the trough was already seeded at the dose record
                // and flowed here (#1121) — re-equilibrating would discard that
                // propagation and restore the defect. The two branches read the
                // same predicate, so they cannot both fire or both skip.
                if ss_equilibrates_at_arrival(d, dose_lagtimes[idx]) {
                    let chz_before = chz_snapshot(ode, &u);
                    u = equilibrate_ss_state(ode, &dose_pk.values, d, &opts, &chz_before);
                }
                // Boluses: add amt to state. Infusions: no instantaneous
                // change — handled via the wrapped RHS for segments inside
                // [d.time, d.time + d.duration]. A dose into a built-in
                // input-rate compartment (transit/etc.) is delivered as R_in
                // over time by the wrapped RHS, so it's skipped here too.
                if !is_real_infusion(d) && !input_rate_consumes_cmt(ode, d.cmt_raw()) {
                    let cmt_idx = d.cmt_idx();
                    if cmt_idx < n {
                        // Bioavailability resolved per dose compartment (`Fn`),
                        // precomputed from `pk_at_dose` alongside the lagtimes.
                        u[cmt_idx] += dose_f_bio[idx] * d.amt;
                    }
                }
                // The arrival is not a record: it must NOT become `last_pk`.
            }
            Kind::Obs => {
                let cmt = subject.obs_cmts.get(idx).copied().unwrap_or(0);
                let v = read_observable(
                    ode,
                    &u,
                    &pk_now.values,
                    theta,
                    eta,
                    subject.obs_cov(idx),
                    cmt,
                    // User-clock `TIME` for the readout — see `record_observations`.
                    subject.readout_time(idx),
                );
                // Clamp negative readouts (ODE solver overshoot guard);
                // let NaN through so a missing `OdeReadout::PerCmt` entry
                // (or any other genuine NaN) surfaces as a NaN OFV
                // rather than a silent zero. See the corresponding note
                // in `ode_predictions`. Bare-state readouts only — a Form C
                // `[scaling]` expression may legitimately be negative (#1020).
                predictions[idx] = if v < 0.0 && ode.readout.clamps_negative() {
                    0.0
                } else {
                    v
                };
                last_pk = pk_now;
            }
            Kind::PkOnly => {
                // EVID=2: $PK ran at this record but compartment state is
                // unchanged. The new pk is consumed by the next segment's
                // integration via the loop-top `pk_now` lookup.
                last_pk = pk_now;
            }
            Kind::InfusionEnd => {
                // Not a NONMEM record: no state update, no PK update —
                // only purpose is to break the timeline so the next
                // segment's `active_infusions` excludes this infusion.
            }
            Kind::Reset => {
                // EVID=3 / EVID=4: reset the system. Compartments with an
                // `init(state) = expr` return to their initial value; all
                // others go to zero (a reset starts a fresh episode from
                // baseline). With no init declared this zeros everything.
                //
                // The seed is evaluated at the RESET ROW'S OWN snapshot
                // (`pk_at_reset[idx]`), not the previous record's (#1133). An
                // EVID=3/4 row is a NONMEM data record: `$PK` runs at it, so a
                // covariate-driven `init(...)` restarts the episode on this
                // row's covariates. Measured against NONMEM 7.6.0
                // (`nonmem_anchor/reset_init_snapshot_*.ctl`): carrying the
                // previous record forward put the whole post-reset trajectory a
                // factor of two out on a `WT` that doubles at the reset. It is
                // also not the *next* record ahead — the resolution #1073 uses
                // for a non-record boundary — which anchor C separates by giving
                // the following record a third `WT` and getting NONMEM's arm-A
                // answer back unchanged.
                //
                // For EVID=4 the dose at this same time follows (Reset sorts
                // before Dose), so it lands on the re-seeded state. Record the
                // reset time so infusions started earlier stop contributing.
                u = ode.initial_state(&pk_at_reset[idx].values);
                reset_floor = t_event;
            }
        }
    }

    predictions
}

/// EKF-based predictions with an explicit diffusion_var slice (bypasses
/// `ode_spec.diffusion_var`). Used by the likelihood path to supply the
/// current theta-derived diffusion variances without mutating the model.
pub fn ode_predictions_ekf_with_diffusion(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    diffusion_var: &[f64],
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    // Resolve modeled-RATE doses once (#324). This resolve is load-bearing for the
    // `solve_ekf` call below, which reads `subject.doses` directly and so needs
    // concrete rate/duration; it cannot be dropped in favour of the resolve inside
    // `ode_predictions` (that one is internal and not visible here). The
    // `ode_predictions` call then re-checks an already-`Fixed` subject — a cheap
    // `all_doses_fixed()` scan that returns `Cow::Borrowed` (no second clone). The
    // clone happens at most once, only on the modeled-`RATE` path.
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    // EKF path: parser rejects SDE + Form C, so output_fn is always None
    // here and theta/eta would never be consulted. Pass empty slices.
    let ipred_plain = ode_predictions(ode, pk_params_flat, &[], &[], subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        // EKF/SDE path requires a single observable compartment index for
        // the Kalman update. Parser-side validation rejects SDE models that
        // use Form C `y = <expr>`; so `obs_cmt_idx` is always `Some` here.
        ode.obs_cmt_idx()
            .expect("EKF requires obs_cmt_idx; SDE + [scaling] y = ... is not supported"),
        diffusion_var,
        pk_params_flat,
        &ode.dose_attr_map,
        &ode.initial_state(pk_params_flat),
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
        ode.effective_solver_opts(),
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

/// EKF-based predictions for a subject with an SDE model.
///
/// Wraps `solve_ekf`, handling the residual variance `r_obs` needed for the
/// Kalman update step. Returns `(ipred, p_obs)` where `p_obs[j]` is the
/// EKF state covariance at the observable compartment just before assimilating
/// observation `j`. Callers add `p_obs[j]` to the residual variance to form
/// `V_total = p_obs[j] + V_residual`.
///
/// `r_obs_fn` computes the scalar residual variance for each observation given
/// the predicted value — this feeds the Kalman update, keeping the covariance
/// estimate numerically stable. It does NOT affect the returned `p_obs` values
/// (those are pre-update, i.e. the purely process-noise contribution).
// Not currently called from outside this module — superseded by
// `ode_predictions_ekf_with_diffusion` which accepts an explicit diffusion_var.
#[allow(dead_code)]
pub fn ode_predictions_ekf(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    r_obs_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    use crate::ode::ekf::solve_ekf;

    // Resolve modeled-RATE doses once (#324). Load-bearing for the `solve_ekf`
    // call below (it reads `subject.doses` directly); the later `ode_predictions`
    // call re-checks an already-`Fixed` subject (cheap scan, `Cow::Borrowed`, no
    // second clone). See `ode_predictions_ekf_with_diffusion` for the rationale.
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    // Compute per-observation R for the Kalman update from a standard ODE pass.
    // Using per-observation R is correct for proportional and combined error models.
    // EKF path: parser rejects SDE + Form C, so output_fn is always None
    // here and theta/eta would never be consulted. Pass empty slices.
    let ipred_plain = ode_predictions(ode, pk_params_flat, &[], &[], subject);
    let r_obs_vec: Vec<f64> = ipred_plain
        .iter()
        .map(|&f| {
            let v = r_obs_fn(f);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();

    let pts = solve_ekf(
        ode.rhs.as_ref(),
        ode.n_states,
        ode.obs_cmt_idx()
            .expect("EKF requires obs_cmt_idx; SDE + [scaling] y = ... is not supported"),
        &ode.diffusion_var,
        pk_params_flat,
        &ode.dose_attr_map,
        &ode.initial_state(pk_params_flat),
        &subject.doses,
        &subject.obs_times,
        &r_obs_vec,
        ode.effective_solver_opts(),
    );

    let ipreds: Vec<f64> = pts.iter().map(|p| p.ipred).collect();
    let p_obs: Vec<f64> = pts.iter().map(|p| p.p_obs).collect();
    (ipreds, p_obs)
}

/// Like [`ode_predictions`] but also returns the raw ODE state vector at every
/// observation time. Returns `(ipred_vec, compartment_states)` where
/// `compartment_states[j]` is `u[0..n_states]` at observation `j`.
///
/// The estimation hot path uses [`ode_predictions`] (no allocation overhead);
/// this variant is called once post-fit to populate `SubjectResult::compartment_states`.
///
/// # KEEP-IN-SYNC with [`ode_predictions`]
///
/// This function is a near-copy of `ode_predictions` with the single addition of
/// `states[obs_idx] = u.clone()` / `states[obs_idx] = pt.u.clone()` at every
/// observation capture site. Any change to dose-event handling, SS logic,
/// infusion tracking, break-time construction, or `read_observable` calls in
/// `ode_predictions` **must be mirrored here**. Search for the parallel line in
/// `ode_predictions` and apply the same change.
///
/// This note is not enough on its own: the inline break-time builder below drifted
/// from `collect_dose_break_times` anyway, losing the per-route absorption onset
/// and with it every lagged `zero_order` window (#1171). The end-to-end guard is
/// `ode::predictions::tests::route_lagged_zero_order_reaches_every_dense_engine`,
/// which checks all three dense engines against a closed-form ramp.
///
/// # Precondition
///
/// The caller **must not** pass a subject that has EVID=3/4 resets
/// (`subject.reset_times` non-empty) or time-varying covariates
/// (`subject.has_tv_covariates()`).  For those subjects
/// `compute_predictions_with_states` routes through
/// `ode_predictions_event_driven_with_states`, which handles resets correctly.
/// Calling this function directly on a reset subject would produce incorrect
/// states because the re-seed events are absent from the break-time list.
pub fn ode_predictions_with_states(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.effective_solver_opts();

    let mut u = ode.initial_state(pk_params_flat);
    let mut predictions = vec![f64::NAN; n_obs];
    let mut states: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; n_obs];

    // Resolve modeled-RATE doses once (#324) before building the timeline so the
    // states pass sees concrete rate/duration; borrowed for all-`Fixed`.
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    // Per dose-compartment bioavailability / lag (`Fn`/`ALAGn`; issue #369),
    // falling back to the bare `PK_IDX_F`/`PK_IDX_LAGTIME` slots. Uniform on
    // this no-TV path, where every dose reads the same `pk_params_flat`.
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    let first_dose_time = earliest_dose_time(&subject.doses);
    let mut ext_params = seed_ext_params(pk_params_flat, first_dose_time);

    let obs_map = build_obs_index_map(&subject.obs_times);

    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![subject_integration_start(subject)];
    for (i, dose) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        break_times.push(dose.time + lag);
        if is_real_infusion(dose) {
            // F-scaled infusion end (#419): rate-defined -> F·duration window.
            let (_, dur_eff) = dose.bioavailable_infusion(dose_f_bio[i]);
            break_times.push(dose.time + lag + dur_eff);
        }
        if ss_seeded_at_record(dose, lag) {
            break_times.push(dose.time);
        }
        // End of the *previous* cycle's infusion when it is still running at the
        // dose record of a seeded SS dose (#1121) — a segment boundary for the
        // same reason the real infusion end is one.
        if let Some(residual_end) = ss_residual_infusion_end(dose, lag, dose_f_bio[i]) {
            break_times.push(residual_end);
        }
    }
    // Per-route absorption onset (`fn(..., lag=L)`), the same call
    // `collect_dose_break_times` makes (#1171). Zero-order no longer depends on this —
    // `push_zero_order_break_times` brackets its own `w_start` — but the *smooth*
    // kernels still do: `first_order` is `dose·ka·exp(-ka·tad)` with a hard `0` for
    // `tad <= 0`, i.e. a step at the onset, and `weibull` (β < 1) and `transit` (n = 0)
    // likewise. `sens/ode_provider.rs` emits `K_ROUTE_ONSET` for every lagged kind and
    // is pinned bit-identical to this break (#859), so it stays unconditional.
    push_route_lag_break_times(&mut break_times, ode, subject, &dose_lagtimes, |f, _| {
        f.route_lag(pk_params_flat)
    });
    // Zero-order windows for this subject (#504): the dense paths have a single
    // PK snapshot, so the per-dose `dur`/`F`/`lag` come from `pk_params_flat`.
    // Break at each window end so segments align with the cutoff, and reuse the
    // same windows for the per-segment constant-rate injection below.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    push_zero_order_break_times(&mut break_times, &zo_windows);
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    // A non-finite break time makes the subject non-finite (#1189); both outputs are
    // NaN-prefilled, so this returns exactly that.
    if abandon_non_finite_timeline(break_times.iter().copied(), None) {
        return (predictions, states);
    }

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

    // Apply-once masks (#1186) — the `Gated` twin of the objective path's pair. The
    // arrival branch below both jumps the state and pushes into `active_infusions`, so
    // a re-application here doubled an infusion's *rate* for its whole window
    // (148.30 against NONMEM's 99.5249857 on the #1186 infusion fixture).
    let mut seed_applied = vec![false; subject.doses.len()];
    let mut applied = vec![false; subject.doses.len()];

    // Records read *at* the current break (#1226) — sorted once, hoisted, as on the
    // objective path.
    let obs_index = RecordIndex::new(&subject.obs_times);
    let mut boundary_obs: Vec<usize> = Vec::new();
    for k in 0..break_times.len() {
        let t_start = break_times[k];

        // SS + lagtime: at the dose *record* time (strictly before the lagged pulse
        // arrives) seed the previous interval's steady-state tail, exactly mirroring
        // the separate pre-pass in `ode_predictions` (lines 479-485).
        for (i, dose) in subject.doses.iter().enumerate() {
            let lag = dose_lagtimes[i];
            if seed_applied[i] {
                continue;
            }
            if ss_seeded_at_record(dose, lag) && (dose.time - t_start).abs() < EVENT_MATCH_TOL {
                seed_applied[i] = true;
                let chz_before = chz_snapshot(ode, &u);
                u = ss_state_at_phase(
                    ode,
                    pk_params_flat,
                    dose,
                    ss_seed_phase(dose, lag),
                    &opts,
                    &chz_before,
                );
                if let Some(residual_end) = ss_residual_infusion_end(dose, lag, dose_f_bio[i]) {
                    // The previous cycle's infusion is still running at the record
                    // and stops inside the pre-arrival window (#1121). Registered
                    // like any other window so `gated_infusions` injects `+rate`
                    // over exactly `[dose.time, residual_end]`; without it the walk
                    // resumes the decay early and the whole window reads low.
                    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                    active_infusions.push((i, dose.time, residual_end));
                }
            }
        }

        // Apply boluses and SS doses at t_eff = dose.time + lagtime.
        // A dose whose record precedes the `SS=1` record reached here, in (time, row
        // order), was wiped by that reset (#1588): no equilibration, no bolus, and its
        // infusion window is off (#1586, `gated_infusions`). No EVID=3/4 floor: this walk is
        // never handed a reset subject (`compute_predictions_with_states` routes one to
        // `ode_predictions_event_driven_with_states`).
        let gate = ResetGate::at_segment(&subject.doses, f64::NEG_INFINITY, t_start);
        for (dose_idx, dose) in subject.doses.iter().enumerate() {
            if applied[dose_idx] {
                continue;
            }
            let t_eff = dose.time + dose_lagtimes[dose_idx];
            if (t_eff - t_start).abs() < EVENT_MATCH_TOL {
                // Marked for every matched dose, whichever branch fires below — the
                // arrival is one event (equilibrate + bolus + infusion push) (#1186).
                applied[dose_idx] = true;
                let f = dose_f_bio[dose_idx];
                let live = gate.live(&subject.doses, dose_idx);
                if live && ss_equilibrates_at_arrival(dose, dose_lagtimes[dose_idx]) {
                    // Only an unseeded (`lag = 0`) SS dose equilibrates at its
                    // arrival. A lagged one was seeded at its record above and has
                    // flowed here; overwriting it would erase a dose that landed
                    // inside the pre-arrival window (#1275).
                    let chz_before = chz_snapshot(ode, &u);
                    u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts, &chz_before);
                }
                if !is_real_infusion(dose) {
                    if live && !input_rate_consumes_cmt(ode, dose.cmt_raw()) {
                        // dose.cmt is 1-based; `CMT=0` is NONMEM's default dose
                        // compartment and resolves to compartment 1, like every
                        // other dose site on both engines (#899). This used to
                        // skip the dose entirely, disagreeing with the two
                        // event-driven drivers on the same dataset.
                        let cmt = dose.cmt_idx();
                        if cmt < n {
                            u[cmt] += dose.amt * f;
                        }
                    }
                    // else: the dose feeds a built-in input-rate function
                    // (transit/etc.) and is delivered as R_in over time by the
                    // wrapped RHS below — no bolus here (would double-count).
                } else {
                    // F-scaled infusion end (#419), matching the break-time list.
                    let (_, dur_eff) = dose.bioavailable_infusion(f);
                    let end_t = t_eff + dur_eff;
                    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                    active_infusions.push((dose_idx, t_eff, end_t));
                }
            }
        }

        // Handle obs read *at* t_start (after dose) — the whole `EVENT_MATCH_TOL` band,
        // through the same helper as the objective path (#1226).
        obs_index.records_at_break(t_start, &mut boundary_obs);
        record_observations(
            ode,
            &boundary_obs,
            &u,
            pk_params_flat,
            theta,
            eta,
            subject,
            &mut predictions,
            Some(states.as_mut_slice()),
        );

        // #731: integrate the open interval `(t_start, t_end]` to the next break, if
        // there is one. The final break has no successor — its dose was applied and its
        // observation read post-dose above, as a left boundary, with nothing left to
        // integrate. Mirrors `ode_predictions`' `0..len` + `k + 1 < len` shape (and the
        // matching fix in `ode_dense_solve_states`); this doc says any dose-event change
        // in `ode_predictions` must be mirrored here.
        if k + 1 >= break_times.len() {
            continue;
        }
        let t_end = break_times[k + 1];

        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .cloned()
            .filter(|&t| reads_in_segment(t, t_start, t_end))
            .collect();
        // Always include t_end so u is advanced to segment end, even when there
        // are no observations in the segment (e.g. two doses with no obs between
        // them). Without this, solve_ode returns an empty solution and u is not
        // updated, leaving the wrong (undecayed) state for the next segment.
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        // Mirror ode_predictions lines 530-531: sort + dedup so solve_ode's
        // linear save_idx cursor works correctly even if obs_times contains
        // duplicate entries or arrives out of order.
        saveat.sort_by(|a, b| a.total_cmp(b));
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // TAD anchor: last effective dose time before this segment, SS-aware.
        // For SS doses, rem_euclid maps the elapsed time back into [0, II) so
        // TAD stays within one dosing interval — matching ode_predictions.
        ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor(subject, &dose_lagtimes, t_start);

        active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
        // Resolve each active infusion to (cmt_idx, F·rate, t_start, t_end) for
        // the time-gated injection inside the seam (CMT=0 / out-of-range dropped).
        let gated = gated_infusions(
            &ode.input_rate,
            &active_infusions,
            &subject.doses,
            &dose_f_bio,
            n,
            &gate,
        );
        // Zero-order absorption windows covering this segment (#504): constant
        // `F·amt/dur` injected alongside the gated infusions (empty otherwise).
        let zero_order = active_zero_order_inputs(
            &zo_windows,
            &subject.doses,
            t_start,
            t_end,
            f64::NEG_INFINITY,
        );
        // Hoist the input-rate constants once per segment (#322 #7).
        let prepared = prepare_input_rates(ode, &ext_params);
        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            f64::NEG_INFINITY,
            t_start,
            &prepared,
            InfusionInput::Gated(gated),
            &zero_order,
        );

        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            &ext_params,
            &saveat,
            &opts,
        );

        for pt in &sol {
            if let Some(obs_idxs) = obs_map.get(&pt.t.to_bits()) {
                record_observations(
                    ode,
                    obs_idxs,
                    &pt.u,
                    pk_params_flat,
                    theta,
                    eta,
                    subject,
                    &mut predictions,
                    Some(states.as_mut_slice()),
                );
            }
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    clamp_negative_predictions(&ode.readout, &mut predictions);

    (predictions, states)
}

/// Like [`ode_predictions_event_driven`] but also returns the raw ODE state
/// at every observation time. Returns `(ipred_vec, compartment_states)`.
///
/// Called post-fit for TV-covariate ODE models to populate
/// `SubjectResult::compartment_states`.
///
/// # Approximation for TV-covariate subjects
///
/// `ipred` is exact (the event-driven path uses per-event PK parameters). The
/// compartment `states`, however, are derived from a second pass of the dense walk
/// behind [`ode_dense_solve_states`], whose **disposition** (CL, V, etc.) and `init()`
/// are the first observation's PK parameters held fixed for the entire timeline. Every
/// dose-record quantity — absorption kernel, pathway fraction, route lag, zero-order
/// `dur`, compartment lag, `F`, `D{n}`/`R{n}` and the SS run-in — is read at its own
/// dose record (`pk_at_dose`), as `ipred` reads it (#1575). For subjects whose
/// disposition genuinely varies between observations the states will be approximate.
/// `fit()` emits `W_DERIVED_CMT_TV_ODE` to alert users to this limitation. For
/// reset-only subjects (no TV covariates) `pk_at_obs` is uniformly filled, so using the
/// first entry is exact.
pub fn ode_predictions_event_driven_with_states(
    ode: &OdeSpec,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
    pk_at_reset: &[PkParams],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Re-use the standard path to get ipred, then do a second pass to
    // extract states. The event-driven function is already complex enough
    // that duplicating it would be error-prone; a second pass is acceptable
    // because this is post-fit only.
    let ipreds = ode_predictions_event_driven(
        ode,
        subject,
        theta,
        eta,
        pk_at_dose,
        pk_at_obs,
        pk_at_pk_only,
        pk_at_reset,
    );

    // Second pass: extract the full ODE state at each obs time via the dense walk
    // (`ode_dense_solve_states_reading`), which integrates under one disposition snapshot.
    //
    // Every dose-record quantity — absorption kernel, pathway fraction, route lag,
    // zero-order `dur`, compartment lag, `F`, `D{n}`/`R{n}`, the SS run-in — is read at
    // its own dose record (`pk_at_dose`), exactly as the event-driven pass above reads
    // it (#1575). Only the disposition (CL/V/etc.) and `init()` come from the
    // first-observation snapshot.
    //
    // For subjects with EVID=3/4 resets but *no* TV covariates, `pk_at_obs` is
    // uniformly filled (every entry identical), so using `first()` is exact.
    //
    // For subjects with genuine TV covariates, `pk_at_obs` varies per timepoint.
    // Holding the disposition at `first()` is an approximation: the compartment state
    // trajectory eliminates under the first-observation snapshot, while `ipreds`
    // correctly reflect per-event covariate snapshots. The caller
    // (`compute_predictions_with_states`) is the approximate path; `fit()` emits
    // W_DERIVED_CMT_TV_ODE when TV covariates are present so users know.
    //
    // A future improvement: duplicate the event-driven loop to capture `u` at each
    // obs time directly — exact states, but ~2× the integration work post-fit.
    let pk_flat = &pk_at_obs
        .first()
        .map(|p| p.values)
        .unwrap_or([0.0; crate::types::MAX_PK_PARAMS]);
    let states = ode_dense_solve_states_reading(
        ode,
        pk_flat,
        DoseReads::PerDose(pk_at_dose),
        subject,
        &subject.obs_times,
    );

    (ipreds, states)
}

/// Build the sorted, deduped dose-segment break times for a subject — the points
/// where the integrator must stop and re-apply boundary events (dose pulses, lags,
/// infusion ends, SS-record seeds, EVID-3/4 resets, per-route absorption onsets,
/// zero-order windows). `terminal` is the final break: the last `saveat` for the
/// dense solve, or the horizon for the event-time search. Shared by
/// [`ode_dense_solve_states`] and [`ode_solve_until_chz_threshold`] so the two
/// segment the timeline identically (a divergence here would make a simulated event
/// time inconsistent with the fitted hazard).
///
/// On the breaks the two share — dose arrivals, infusion ends, SS seeds, route
/// onsets, zero-order edges — it must agree with [`collect_dose_break_times`], the
/// prediction engines' builder; `route_lagged_zero_order_break_builders_agree`
/// asserts that on a route-lagged subject, after this one silently lost the
/// route-onset break (#1171). The lists are **not** equal in general: this one also
/// seeds `subject_integration_start`, pushes `subject.reset_times` and appends
/// `terminal`. Those three are this builder's own, and the reset push in particular
/// is load-bearing (#1133) — do not delete it to "restore agreement".
fn build_segment_break_times(
    ode: &OdeSpec,
    dose_reads: DoseReads<'_>,
    subject: &Subject,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    zo_windows: &[ZeroOrderWindow],
    terminal: f64,
) -> Vec<f64> {
    // Integration starts at the subject's first event, not a phantom t=0 (#573) —
    // shared by the dense fit path and the TTE event-time search so both segment
    // the timeline identically.
    let mut break_times: Vec<f64> = vec![subject_integration_start(subject)];
    for (i, dose) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        break_times.push(dose.time + lag);
        if is_real_infusion(dose) {
            // F-scaled infusion end (#419): rate-defined -> F·duration window.
            let (_, dur_eff) = dose.bioavailable_infusion(dose_f_bio[i]);
            break_times.push(dose.time + lag + dur_eff);
        }
        if ss_seeded_at_record(dose, lag) {
            break_times.push(dose.time);
        }
        // End of the *previous* cycle's infusion when it is still running at the
        // dose record of a seeded SS dose (#1121) — a segment boundary for the
        // same reason the real infusion end is one.
        if let Some(residual_end) = ss_residual_infusion_end(dose, lag, dose_f_bio[i]) {
            break_times.push(residual_end);
        }
    }
    // EVID=3/4 resets must be break-points so the re-seed happens at the exact boundary.
    for &rt in &subject.reset_times {
        break_times.push(rt);
    }
    // Per-route absorption onset (`fn(..., lag=L)`), the same call
    // `collect_dose_break_times` makes (#1171). Zero-order no longer depends on this —
    // `push_zero_order_break_times` brackets its own `w_start` — but the *smooth*
    // kernels still do: `first_order` is `dose·ka·exp(-ka·tad)` with a hard `0` for
    // `tad <= 0`, i.e. a step at the onset, and `weibull` (β < 1) and `transit` (n = 0)
    // likewise. `sens/ode_provider.rs` emits `K_ROUTE_ONSET` for every lagged kind and
    // is pinned bit-identical to this break (#859), so it stays unconditional.
    // Read at dose `k`'s own record, like the compartment lag above (#1575).
    push_route_lag_break_times(&mut break_times, ode, subject, dose_lagtimes, |f, k| {
        f.route_lag(dose_reads.at(k))
    });
    push_zero_order_break_times(&mut break_times, zo_windows);
    break_times.push(terminal);
    break_times.sort_by(|a, b| a.total_cmp(b));
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    break_times
}

/// Where the dense walks ([`ode_dense_solve_states`], [`ode_solve_until_chz_threshold`])
/// read each **dose-record quantity** (#1575): the absorption kernel, pathway fraction and
/// route lag of a built-in input rate, the zero-order `dur`, the compartment lag and `F`,
/// `D{n}`/`R{n}`, and the SS run-in of an `SS=1` dose.
///
/// The disposition — the parameters the `[odes]` RHS integrates under, and `init()` — is a
/// separate argument and stays one snapshot. These quantities are properties of the dose
/// record, so the event-driven engine reads every one of them at `pk_at_dose[k]`; a pass
/// that read them at its one disposition snapshot instead gave a time-varying covariate on
/// a dose quantity whatever value it held at that snapshot, a whole-dose error invisible
/// to the dose-record domain check.
#[derive(Clone, Copy)]
pub(crate) enum DoseReads<'a> {
    /// One snapshot serves every dose: the parameter-static callers, where it is exact.
    Shared(&'a [f64]),
    /// Dose `k` reads its own record's snapshot; one entry per dose of the subject.
    PerDose(&'a [PkParams]),
}

impl<'a> DoseReads<'a> {
    /// Dose `k`'s snapshot.
    #[inline]
    fn at(self, k: usize) -> &'a [f64] {
        match self {
            DoseReads::Shared(p) => p,
            DoseReads::PerDose(per_dose) => &per_dose[k].values,
        }
    }

    /// The input-rate forcings for `n_doses` doses, built once per subject. `Shared` keeps
    /// [`PreparedForcings::shared`]'s factored fraction, so a parameter-static caller's
    /// arithmetic is bit-for-bit what it was.
    fn prepare(self, ode: &OdeSpec, n_doses: usize) -> PreparedForcings {
        match self {
            DoseReads::Shared(p) => prepare_input_rates(ode, p),
            DoseReads::PerDose(per_dose) => {
                debug_assert_eq!(per_dose.len(), n_doses);
                PreparedForcings::per_dose(
                    ode,
                    n_doses,
                    |k| &per_dose[k].values[..],
                    InputRateForcing::prepare,
                )
            }
        }
    }
}

/// Owned per-segment forcings produced by [`apply_segment_boundary`]: everything
/// `wrap_rhs_with_forcings` needs for one dose segment besides the per-subject
/// [`PreparedForcings`], returned by value so the caller can build (and borrow into) the
/// wrapped RHS without a dangling borrow.
struct SegmentForcings {
    reset_floor: f64,
    gated: Vec<(usize, f64, f64, f64)>,
    zero_order: Vec<(usize, f64)>,
}

/// Apply a dose segment's boundary events and resolve its forcings — the shared
/// core of the per-segment loop used by both [`ode_dense_solve_states`] (the
/// fit-path dense solve) and [`ode_solve_until_chz_threshold`] (the TTE event-time
/// search), so the two cannot drift. Mutates `u` (EVID-3/4 reset re-seed, SS-lag
/// seeding, bolus additions), `active_infusions` (activation + expiry), and
/// `ext_params` (the TAD anchor slot), then returns this `[t_start, t_end)`
/// segment's forcings for the caller to build the wrapped RHS and integrate.
///
/// `seed_applied` / `applied` are the walk's apply-once masks (#1186), owned by the
/// caller for the same reason `active_infusions` is: they are walk state, not segment
/// state, and a dose must fire at the first break within [`EVENT_MATCH_TOL`] and at no
/// other.
///
/// `disposition` re-seeds `init()` at an EVID-3/4 reset; an `SS=1` dose's run-in reads its
/// own record through `dose_reads`, as the event-driven engine's does (#1575).
#[allow(clippy::too_many_arguments)]
fn apply_segment_boundary(
    ode: &OdeSpec,
    subject: &Subject,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    zo_windows: &[ZeroOrderWindow],
    disposition: &[f64],
    dose_reads: DoseReads<'_>,
    n: usize,
    opts: &OdeSolverOptions,
    t_start: f64,
    t_end: f64,
    u: &mut Vec<f64>,
    active_infusions: &mut Vec<(usize, f64, f64)>,
    ext_params: &mut [f64],
    seed_applied: &mut [bool],
    applied: &mut [bool],
) -> SegmentForcings {
    debug_assert!(
        seed_applied.len() >= subject.doses.len() && applied.len() >= subject.doses.len()
    );
    // EVID=3/4 reset: re-seed compartments before processing doses at this time.
    // Resets sort before doses at the same time (mirroring Kind::Reset < Kind::Dose).
    for &rt in &subject.reset_times {
        if (rt - t_start).abs() < EVENT_MATCH_TOL {
            *u = ode.initial_state(disposition);
            break;
        }
    }

    // SS + lagtime: at the dose *record* time (before the lagged pulse arrives)
    // seed the previous interval's steady-state tail, mirroring ode_predictions.
    for (i, dose) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        if seed_applied[i] {
            continue;
        }
        if ss_seeded_at_record(dose, lag) && (dose.time - t_start).abs() < EVENT_MATCH_TOL {
            seed_applied[i] = true;
            let chz_before = chz_snapshot(ode, u);
            *u = ss_state_at_phase(
                ode,
                dose_reads.at(i),
                dose,
                ss_seed_phase(dose, lag),
                opts,
                &chz_before,
            );
            if let Some(residual_end) = ss_residual_infusion_end(dose, lag, dose_f_bio[i]) {
                // The previous cycle's infusion is still running at the record
                // and stops inside the pre-arrival window (#1121). Registered
                // like any other window so `gated_infusions` injects `+rate`
                // over exactly `[dose.time, residual_end]`; without it the walk
                // resumes the decay early and the whole window reads low.
                active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                active_infusions.push((i, dose.time, residual_end));
            }
        }
    }

    // The most recent EVID=3/4 reset at or before this segment (the one just applied
    // above, if any).
    let reset_floor = subject
        .reset_times
        .iter()
        .cloned()
        .filter(|&rt| rt <= t_start + 1e-12)
        .fold(f64::NEG_INFINITY, f64::max);

    // The resets reached here (#1588, #1587), as in `ode_predictions_event_driven`: a dose
    // whose record precedes the `SS=1` record or the EVID=3/4 floor neither equilibrates
    // nor jumps, and its infusion window is off from this segment on (#1586) — dropped by
    // `gated_infusions` under this same gate.
    let gate = ResetGate::at_segment(&subject.doses, reset_floor, t_start);
    for (dose_idx, dose) in subject.doses.iter().enumerate() {
        if applied[dose_idx] {
            continue;
        }
        let t_eff = dose.time + dose_lagtimes[dose_idx];
        if (t_eff - t_start).abs() < EVENT_MATCH_TOL {
            // One arrival event: equilibrate + bolus + infusion push (#1186).
            applied[dose_idx] = true;
            let f = dose_f_bio[dose_idx];
            let live = gate.live(&subject.doses, dose_idx);
            if live && ss_equilibrates_at_arrival(dose, dose_lagtimes[dose_idx]) {
                // Only an unseeded (`lag = 0`) SS dose equilibrates at its arrival;
                // a lagged one was seeded at its record above and flows here, so a
                // dose inside the pre-arrival window survives (#1275).
                let chz_before = chz_snapshot(ode, u);
                *u = equilibrate_ss_state(ode, dose_reads.at(dose_idx), dose, opts, &chz_before);
            }
            if !is_real_infusion(dose) {
                if live && !input_rate_consumes_cmt(ode, dose.cmt_raw()) {
                    // dose.cmt is 1-based; `CMT=0` is NONMEM's default dose
                    // compartment and resolves to compartment 1, like every
                    // other dose site on both engines (#899).
                    let cmt = dose.cmt_idx();
                    if cmt < n {
                        u[cmt] += dose.amt * f;
                    }
                }
                // else: the dose feeds a built-in input-rate function
                // (transit/etc.) and is delivered as R_in over time by the
                // wrapped RHS below — no bolus here (would double-count).
            } else {
                // F-scaled infusion end (#419), matching the break-time list. Registered
                // live or not: `gated_infusions` drops a reset window per segment (#1586).
                let (_, dur_eff) = dose.bioavailable_infusion(f);
                let end_t = t_eff + dur_eff;
                active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
                active_infusions.push((dose_idx, t_eff, end_t));
            }
        }
    }

    // TAD anchor: SS-aware, matching ode_predictions (rem_euclid wraps the elapsed
    // time back into [0, II)).
    ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor(subject, dose_lagtimes, t_start);

    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
    // Resolve to (cmt_idx, F·rate, t_start, t_end) for the seam's time-gated
    // injection (CMT=0 / out-of-range dropped).
    let gated = gated_infusions(
        &ode.input_rate,
        active_infusions,
        &subject.doses,
        dose_f_bio,
        n,
        &gate,
    );

    // `reset_floor` (above) also turns off, for the input-rate forcing, every dose recorded
    // before the reset — mirroring how the reset clears `active_infusions` and re-seeds `u`.

    // Zero-order absorption windows covering this segment (#504): constant
    // `F·amt/dur`, reset-aware via the same `reset_floor` (a window opened
    // pre-reset is off), injected alongside the gated infusions.
    let zero_order =
        active_zero_order_inputs(zo_windows, &subject.doses, t_start, t_end, reset_floor);

    SegmentForcings {
        reset_floor,
        gated,
        zero_order,
    }
}

/// Run the ODE solver with an arbitrary set of `saveat` time points and
/// return the full state vector at each requested time.
///
/// This is used by the grid-based integral path in `compute_extra_output_columns`
/// when the integrand references compartment states. The result is only needed
/// post-fit (never on the estimation hot path).
///
/// Dose events (boluses, infusions, SS) are handled identically to
/// [`ode_predictions`]. Subject observation times are ignored; only `saveat`
/// times are returned.
///
/// Every quantity — the disposition, `init()` and each dose's absorption, lag, `F`,
/// `D{n}`/`R{n}` and SS run-in — is read from the one snapshot `pk_params_flat`, which is
/// exact for a parameter-static subject. The crate's callers that reach a subject whose
/// dose records differ from that snapshot read each dose-record quantity at its own record
/// instead (#1575).
pub fn ode_dense_solve_states(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    saveat: &[f64],
) -> Vec<Vec<f64>> {
    // `theta` and `eta` are accepted for API symmetry with sibling ODE functions
    // (e.g. `ode_predictions_with_states`) but are not consumed here: this
    // function returns the raw ODE state vector `u` without applying any
    // `output_fn` / Form-C scaling. A future extension that returns scaled
    // observables alongside states would use them. Suppress the unused warning.
    let _ = (theta, eta);
    ode_dense_solve_states_reading(
        ode,
        pk_params_flat,
        DoseReads::Shared(pk_params_flat),
        subject,
        saveat,
    )
}

/// [`ode_dense_solve_states`] with the dose-record quantities read through `dose_reads`
/// (#1575) and the disposition and `init()` from `disposition`. `DoseReads::Shared` of the
/// disposition is [`ode_dense_solve_states`] itself, bit for bit.
pub(crate) fn ode_dense_solve_states_reading(
    ode: &OdeSpec,
    disposition: &[f64],
    dose_reads: DoseReads<'_>,
    subject: &Subject,
    saveat: &[f64],
) -> Vec<Vec<f64>> {
    if saveat.is_empty() {
        return vec![];
    }
    let n = ode.n_states;
    let opts = ode.effective_solver_opts();

    let mut u = ode.initial_state(disposition);
    let mut result: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; saveat.len()];

    // Resolve modeled-RATE doses once (#324), each at its own record, before building the
    // timeline so the states pass sees concrete rate/duration; borrowed for all-`Fixed`.
    let resolved = resolve_subject_doses_with(subject, &ode.dose_attr_map, |k| dose_reads.at(k));
    let subject: &Subject = &resolved;

    // Per dose-compartment bioavailability / lag (`Fn`/`ALAGn`; issue #369),
    // falling back to the bare `PK_IDX_F`/`PK_IDX_LAGTIME` slots, read at each dose's
    // record.
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs_with(subject, ode, |k| dose_reads.at(k));

    let first_dose_time = earliest_dose_time(&subject.doses);
    let mut ext_params = seed_ext_params(disposition, first_dose_time);

    // Build saveat → index map for fast lookup.
    let saveat_map = build_obs_index_map(saveat);

    let t_last = saveat.iter().cloned().fold(0.0f64, f64::max);
    // Zero-order absorption windows for this subject (#504), `dur` read at each dose's
    // record. Reused for both the segment break points and the per-segment constant-rate
    // injection.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |k, d| {
        zero_order_dur_and_frac_for_dose(ode, d, dose_reads.at(k))
    });
    // The input-rate forcings — kernel, pathway fraction, route lag — once per subject.
    let prepared = dose_reads.prepare(ode, subject.doses.len());
    let break_times = build_segment_break_times(
        ode,
        dose_reads,
        subject,
        &dose_lagtimes,
        &dose_f_bio,
        &zo_windows,
        t_last,
    );

    // A non-finite break time makes the subject non-finite (#1189); `result` is
    // NaN-prefilled, so the caller's finiteness guard sees a diverged solve rather than
    // a plausible-looking trajectory with the NaN-lagged dose silently missing.
    if abandon_non_finite_timeline(break_times.iter().copied(), None) {
        return result;
    }

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();
    // Apply-once masks (#1186), owned here and threaded into `apply_segment_boundary`
    // exactly like `active_infusions` — walk state, not segment state.
    let mut seed_applied = vec![false; subject.doses.len()];
    let mut applied = vec![false; subject.doses.len()];

    // Saveat nodes earlier than the first integrated segment (e.g. a discrete-state CTMM
    // observation recorded before the first dose, whose times the segment timeline — built from
    // doses/obs_times/pk-only/resets, not `obs_records` — does not cover). No-op for the usual
    // case where every saveat is at or after the first event. One function with the #570
    // one-solve share; see [`fill_prestart_states`] for why there is no second copy to keep in
    // step (#1223).
    fill_prestart_states(saveat, &mut result, break_times.first().copied(), &u);

    // Walk every break as a **left boundary** — bound `0..len`, the walk
    // `ode_predictions` and `ode_predictions_with_states` use (#731) — so a dose
    // landing on the final break is applied and its `saveat` read post-dose, and a
    // one-break timeline is visited exactly once (#1218: every `saveat` at or before
    // the first event puts the horizon `t_last` on the integration start, so the
    // timeline is a single instant). The integration half runs only while a next
    // break exists; on the last break the boundary visit is all there is.
    //
    // History, because both defects lived in this loop's shape: it was a
    // `windows(2)` walk that saw the final break only as a segment *end*, patched by
    // a post-loop re-visit for #731 that was guarded to `len >= 2` — "a single-instant
    // `saveat` keeps its prior behaviour", and the prior behaviour was the `f64::NAN`
    // prefill, which `predict_survival(&[0.0])` and the event-driven `[derived]` state
    // path returned as a silent non-answer. The `0..len` walk has no special case to
    // guard. The pre-first-event prefill above is deliberately *not* widened to cover
    // the instant: it holds the seeded, pre-dose state, and a drug-driven hazard read
    // off it is wrong in a way that looks finite.
    //
    // The last break's visit is skipped when no `saveat` sits on it: `u`, `ext_params`
    // and `active_infusions` are dead after this loop, so the SS equilibration it
    // would run for a grid entirely before the first event (`[-1.0]` with a dose at
    // `0`, where the `0.0`-seeded horizon fold still puts the dose on the timeline)
    // has no reader. Unobservable on every other timeline: `t_last` is a `saveat`
    // whenever any point is non-negative.
    // Grid points read *at* the current break (#1226) — sorted once, hoisted, as on the
    // other engines. This is the engine where the index earns its keep: `saveat` is a grid
    // (hazard timeline, `[derived]` integral, AUC, `predict_survival` horizon) and routinely
    // runs to hundreds or thousands of points.
    let saveat_index = RecordIndex::new(saveat);
    let mut boundary_saveat: Vec<usize> = Vec::new();
    for k in 0..break_times.len() {
        let t_start = break_times[k];
        let next = break_times.get(k + 1).copied();
        // The band, not the exact bits: a grid point inside the final break's band is a
        // reader for that break's visit, so the skip must ask the same question the read
        // below does or it can break out one iteration too early (#1226).
        if next.is_none() && !saveat_index.any_at_break(t_start) {
            break;
        }
        let t_end = next.unwrap_or(t_start);

        let forcings = apply_segment_boundary(
            ode,
            subject,
            &dose_lagtimes,
            &dose_f_bio,
            &zo_windows,
            disposition,
            dose_reads,
            n,
            &opts,
            t_start,
            t_end,
            &mut u,
            &mut active_infusions,
            &mut ext_params,
            &mut seed_applied,
            &mut applied,
        );

        // Saveat points read *at* t_start (after dose, matching ode_predictions
        // convention) — the whole `EVENT_MATCH_TOL` band, through the same helper (#1226).
        // `u` here is the post-dose state; `apply_segment_boundary` set ext_params and
        // resolved forcings but did not touch `u` after the dose pulses.
        saveat_index.records_at_break(t_start, &mut boundary_saveat);
        for &i in &boundary_saveat {
            result[i] = u.clone();
        }

        let Some(t_end) = next else {
            break;
        };

        let mut seg_saveat: Vec<f64> = saveat
            .iter()
            .cloned()
            .filter(|&t| reads_in_segment(t, t_start, t_end))
            .collect();
        // Always include t_end so u advances through empty segments (e.g. two
        // consecutive doses with no saveat points between them).
        if seg_saveat.is_empty() || (seg_saveat.last().unwrap() - t_end).abs() > 1e-12 {
            seg_saveat.push(t_end);
        }
        // Mirror ode_predictions lines 530-531 (and the same fix applied to
        // ode_predictions_with_states): sort + dedup so solve_ode's linear
        // save_idx cursor works correctly for duplicate / out-of-order times.
        seg_saveat.sort_by(|a, b| a.total_cmp(b));
        seg_saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            forcings.reset_floor,
            t_start,
            &prepared,
            InfusionInput::Gated(forcings.gated),
            &forcings.zero_order,
        );

        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (t_start, t_end),
            &ext_params,
            &seg_saveat,
            &opts,
        );

        for pt in &sol {
            if let Some(idxs) = saveat_map.get(&pt.t.to_bits()) {
                for &i in idxs {
                    result[i] = pt.u.clone();
                }
            }
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    result
}

/// Whole-horizon outcome of the drug-driven TTE event-time search (plan §8.8.3,
/// wrapper level). Maps from the per-segment
/// [`crate::ode::solver::ThresholdCrossing`]: a `Crossed` in any dose segment ⇒
/// [`Crossed`](ThresholdOutcome::Crossed); every segment reaching its end up to
/// `horizon` ⇒ [`CensoredAtHorizon`](ThresholdOutcome::CensoredAtHorizon); any
/// segment failing ⇒ [`SolveFailed`](ThresholdOutcome::SolveFailed) — a failed
/// solve is **never** reported as a censored subject.
#[cfg(feature = "survival")]
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdOutcome {
    /// The cumulative hazard reached `−log(u)` (an event) at this time.
    Crossed(f64),
    /// Integrated cleanly to `horizon` without the hazard reaching the threshold:
    /// the draw is administratively right-censored at `horizon`.
    CensoredAtHorizon,
    /// The integration cannot yield a meaningful event time (non-monotone /
    /// non-finite hazard, or step budget exhausted). The message names the cause.
    SolveFailed(String),
}

/// Integrate a subject's augmented ODE from `0` to `horizon`, applying doses /
/// infusions / EVID-3 resets via the **same break-time segmentation as
/// [`ode_dense_solve_states`]**, and halt at the first time `u[chz_state]` reaches
/// `threshold` (the cumulative-hazard accumulator hitting `−log u`). This is the
/// segmented driver behind drug-driven TTE event-time sampling (plan §8.8.3): the
/// CHZ accumulator runs continuously across dose boundaries (it is *not* reset),
/// and the absolute `threshold` is held across segments.
///
/// `horizon` must be finite — a drug-driven hazard can vanish and never fire, so an
/// unbounded search is ill-posed; the `simulate` layer enforces this before calling.
///
/// **Why this mirrors `ode_dense_solve_states` and not `integrate_segment`:** the
/// fit-path cumulative hazard is computed by `ode_dense_solve_states` (via
/// `survival::ode_cumhaz_hazard`), which uses the `Gated` infusion strategy and the
/// inline segment loop. Simulation must reproduce *that* orchestration so a
/// simulated event time is consistent with the hazard the fit integrated. The
/// physics is the shared helpers (`resolve_subject_doses`, `ss_state_at_phase`,
/// `equilibrate_ss_state`, `gated_infusions`, `zero_order_windows`,
/// `prepare_input_rates`, `wrap_rhs_with_forcings`); only the segment *loop* is
/// restated, and it is pinned against drift by the `until_chz_threshold` parity
/// test (the crossing time it returns must satisfy `CHZ_dense(t) ≈ threshold`).
///
/// One deliberate difference from the dense walk: the final break is visited only as
/// a segment *end*, never as a left boundary (#731 / #1218). A dose landing exactly on
/// `horizon` cannot move the accumulator before `horizon`, and a one-break timeline —
/// `horizon` at the integration start — is a censored draw, not a solve; so there is
/// no post-dose state to read here and nothing for the parity test to miss.
///
/// `disposition` and `dose_reads` are read exactly as [`ode_dense_solve_states_reading`]
/// reads them (#1575): the disposition and `init()` from one snapshot, every dose-record
/// quantity through `dose_reads`.
#[cfg(feature = "survival")]
pub(crate) fn ode_solve_until_chz_threshold(
    ode: &OdeSpec,
    disposition: &[f64],
    dose_reads: DoseReads<'_>,
    subject: &Subject,
    chz_state: usize,
    threshold: f64,
    horizon: f64,
) -> ThresholdOutcome {
    use crate::ode::solver::{solve_ode_until_threshold, ThresholdCrossing};

    let n = ode.n_states;
    let opts = ode.effective_solver_opts();
    let mut u = ode.initial_state(disposition);

    // Resolve modeled-RATE doses once, exactly as the dense path (#324).
    let resolved = resolve_subject_doses_with(subject, &ode.dose_attr_map, |k| dose_reads.at(k));
    let subject: &Subject = &resolved;

    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs_with(subject, ode, |k| dose_reads.at(k));

    let first_dose_time = earliest_dose_time(&subject.doses);
    let mut ext_params = seed_ext_params(disposition, first_dose_time);

    // Zero-order windows, reused for the break points and the per-segment injection
    // (same as the dense path). The terminal break is the horizon; doses scheduled
    // after it are dropped — they can never bring an event forward.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |k, d| {
        zero_order_dur_and_frac_for_dose(ode, d, dose_reads.at(k))
    });
    let prepared = dose_reads.prepare(ode, subject.doses.len());
    let mut break_times = build_segment_break_times(
        ode,
        dose_reads,
        subject,
        &dose_lagtimes,
        &dose_f_bio,
        &zo_windows,
        horizon,
    );
    // A non-finite break time makes the subject unsolvable (#1189). This engine has a
    // typed failure, so it uses it rather than reporting a NaN crossing time.
    //
    // **Before the `retain` below, deliberately.** `NaN <= horizon + 1e-15` and
    // `inf <= horizon + 1e-15` are both `false`, so the horizon filter *removes* exactly
    // the entries this guard exists to catch. Ordered the other way the guard is dead
    // code: the walk would proceed on a timeline the bad dose had been deleted from,
    // never apply it, and return a finite crossing time — the silent-wrong-number
    // outcome, on the one engine whose typed failure was supposed to make it loud.
    if abandon_non_finite_timeline(break_times.iter().copied(), None) {
        return ThresholdOutcome::SolveFailed("non-finite break time".to_string());
    }
    break_times.retain(|&t| t <= horizon + 1e-15);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();
    // Apply-once masks (#1186) — same ownership as the dense solve's pair.
    let mut seed_applied = vec![false; subject.doses.len()];
    let mut applied = vec![false; subject.doses.len()];

    for w in break_times.windows(2) {
        let (t_start, t_end) = (w[0], w[1]);
        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Same per-segment boundary handling as the fit-path dense solve — shared so
        // a simulated event time is consistent with the fitted hazard. (A full EVID-3
        // reset would zero CHZ; the `simulate` layer asserts ODE-TTE subjects carry
        // none — selective per-state reset is Phase 3, §8.8.6.)
        let forcings = apply_segment_boundary(
            ode,
            subject,
            &dose_lagtimes,
            &dose_f_bio,
            &zo_windows,
            disposition,
            dose_reads,
            n,
            &opts,
            t_start,
            t_end,
            &mut u,
            &mut active_infusions,
            &mut ext_params,
            &mut seed_applied,
            &mut applied,
        );

        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            forcings.reset_floor,
            t_start,
            &prepared,
            InfusionInput::Gated(forcings.gated),
            &forcings.zero_order,
        );

        // The absolute CHZ threshold is held across segments — `u[chz_state]`
        // accumulates continuously, so a crossing in any segment is the event.
        match solve_ode_until_threshold(
            &wrapped_rhs,
            &mut u,
            (t_start, t_end),
            &ext_params,
            &opts,
            chz_state,
            threshold,
        ) {
            ThresholdCrossing::Crossed(t) => return ThresholdOutcome::Crossed(t),
            ThresholdCrossing::ReachedEnd => {} // u advanced; carry into next segment
            ThresholdCrossing::Failed(why) => return ThresholdOutcome::SolveFailed(why),
        }
    }

    ThresholdOutcome::CensoredAtHorizon
}

#[cfg(test)]
#[path = "predictions_tests.rs"]
mod tests;
