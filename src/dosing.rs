//! Neutral dose-resolution + steady-state-equilibration primitives.
//!
//! These were extracted from `ode/predictions.rs` so that `pk/`, `sens/`, and
//! `api` no longer depend *upward* on `ode/` merely to resolve modeled-`RATE`
//! doses or to share the SS-equilibration convergence tracker. Pure code motion:
//! `ode::predictions` re-exports every symbol below, so all historical
//! `crate::ode::{resolve_subject_doses, predictions::*}` paths keep resolving.
//!
//! - Dose resolution (`resolve_subject_doses{,_with}`): the #324 single source of
//!   truth for turning modeled-`RATE` (`RATE=-2` → modeled duration `D{cmt}`)
//!   doses into concrete `Fixed` doses.
//! - SS equilibration (`SS_EQUILIBRATION_CYCLES`, `SS_EQUILIBRATION_TOL`,
//!   `ss_cycle_converged`, `SsStopTracker`, and the test-only cycle recorder):
//!   the shared convergence policy the analytical, ODE, and sensitivity SS loops
//!   all reuse so their troughs can't drift apart.

use crate::types::Subject;
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

/// Relative floor for truncating the steady-state **input-rate periodic sum** (#719). An
/// `SS=1` dose into a built-in absorption compartment stands for an infinite past pulse
/// train, so its appearance rate at time `t` is `Σ_{j≥0} R_in(tad + j·II)` — the tail of
/// every prior pulse still arriving (see [`add_prepared_input_rate_forcing`]). The absorption
/// density is eventually monotone-decreasing, so once a term falls below this fraction of the
/// running sum the remaining tail is spent and the sum stops (hard-capped at
/// [`SS_EQUILIBRATION_CYCLES`] so a pathologically slow absorption — mode ≫ II — still
/// terminates, matching the trough's own cycle budget). Conservative (`1e-10`): the dropped
/// tail is far below the provider-vs-production parity tolerance.
pub(crate) const SS_TAIL_REL_FLOOR: f64 = 1e-10;

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
/// budget to `< 1e-6` (see `ode_provider_ss_early_stop_matches_full_budget`).
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
}

impl SsStopTracker {
    /// Record `cur` and report whether the trough has converged (from cycle 1 on). Returns
    /// `true` to break the equilibration loop.
    pub(crate) fn should_stop(&mut self, cycle: usize, cur: &[f64]) -> bool {
        if cycle > 0 && ss_cycle_converged(cur, &self.prev, SS_EQUILIBRATION_TOL) {
            return true;
        }
        self.prev.clear();
        self.prev.extend_from_slice(cur);
        false
    }
}

#[cfg(test)]
thread_local! {
    /// Cycles the most recent SS-equilibration call ran — a **test-only** observation of the
    /// #519 early stop, so a test can assert it fired for fast PK and ran the full budget for
    /// slow PK (#532 review #5/#6 — otherwise the stop logic ships unverified, since the loose
    /// end-value tolerances absorb a too-early exit). Set by the f64 predictor, the dual ODE /
    /// closed-form loops, and the event-driven loop.
    static LAST_SS_EQUILIBRATION_CYCLES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// When set, [`ss_cycle_converged`] always reports "not converged" so every path runs the
    /// full cycle budget — lets a test pin that early-stop is value-preserving vs full
    /// equilibration (#532 review #4).
    static FORCE_FULL_SS_EQUILIBRATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn record_ss_equilibration_cycles(n: usize) {
    LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.set(n));
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
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_FULL_SS_EQUILIBRATION.with(|c| c.set(false));
        }
    }
    FORCE_FULL_SS_EQUILIBRATION.with(|c| c.set(true));
    let _reset = Reset;
    f()
}

/// No-op in non-test builds (zero cost on the hot path).
#[cfg(not(test))]
#[inline(always)]
pub(crate) fn record_ss_equilibration_cycles(_n: usize) {}
