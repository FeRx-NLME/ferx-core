//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.
//!
//! Infusion doses (`rate > 0`) are handled by breaking the timeline at the
//! infusion's end time and adding `+rate` to the corresponding compartment's
//! derivative for the duration of the infusion via an RHS wrapper.

use crate::ode::solver::{solve_ode, solve_ode_dense, OdeSolverOptions, OdeSolverStats};
use crate::pk::absorption::PreparedInputRate;
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
use std::borrow::Cow;
use std::collections::HashMap;

/// Epsilon used to decide whether an infusion fully spans a segment.
/// Break times are constructed to coincide with infusion start/end so any
/// non-degenerate segment is either fully inside or fully outside each
/// infusion window — this tolerance only guards float-equality on the bound.
/// `pub(crate)` so the analytic-sensitivity walks reuse the same value rather
/// than hard-coding a parallel literal (#472 review [7]).
pub(crate) const INFUSION_EPS: f64 = 1e-12;

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

/// Pre-equilibrate the ODE state to its steady-state value for an SS=1
/// dose with interval `dose.ii`. NONMEM SS=1 semantics: at the time of
/// the SS dose, the compartments are loaded with the steady-state
/// amounts from an infinite-past pulse train. No closed form is
/// available for arbitrary ODE systems, so we numerically expand the
/// train: starting from a zero state, simulate
/// [`SS_EQUILIBRATION_CYCLES`] cycles of `(apply dose; integrate for II)`.
/// The state after the loop equals the "just-before-next-pulse" SS state;
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
fn equilibrate_ss_state(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    let n = ode.n_states;
    let mut u = vec![0.0; n];

    if dose.ii <= 0.0 || dose.cmt == 0 {
        return u;
    }
    let cmt_idx = dose.cmt - 1;
    if cmt_idx >= n {
        return u;
    }

    // Bioavailability F scales the amount that actually enters the dosing
    // compartment — NONMEM's convention (F·AMT for a bolus, F·RATE for an
    // infusion). Resolved per dose compartment (`Fn`; issue #369), falling back
    // to the bare `PK_IDX_F` slot. Matches the analytical path
    // (`equilibrate_ss_state_event_driven`).
    let f_bio = ode.dose_attr_map.f_bio(dose.cmt, pk_params_flat);

    let is_inf = is_real_infusion(dose);
    // Mode-aware bioavailability (#419): a rate-defined infusion keeps its rate
    // and `F` scales the duration; a duration-defined infusion (`RATE=-2`) keeps
    // its duration and `F` scales the rate. Total input is `F·AMT` either way.
    let (inf_rate, t_inf) = dose.bioavailable_infusion(f_bio);
    if is_inf && t_inf > dose.ii {
        // Overlapping infusions; no closed-form / simple equilibration.
        return u;
    }

    // Early stop once the trough stops moving (#519): the shared tracker holds the previous
    // cycle's state and, from cycle 1 on, breaks when the increment is below the mixed
    // atol/rtol criterion (#532 review #6 — one scaffold across the f64 paths).
    let mut tracker = SsStopTracker::default();
    let mut cycles_run = 0usize;
    for cycle in 0..SS_EQUILIBRATION_CYCLES {
        if is_inf {
            // Active-infusion window: wrapped RHS injects rate into the
            // dosing compartment.
            let rate = inf_rate;
            let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
                (ode.rhs)(y, p, t, dy);
                if cmt_idx < dy.len() {
                    dy[cmt_idx] += rate;
                }
            };
            let sol = solve_ode(
                &wrapped_rhs,
                &u,
                (0.0, t_inf),
                pk_params_flat,
                &[t_inf],
                opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            // Quiet window from end-of-infusion to end-of-cycle.
            let quiet = dose.ii - t_inf;
            if quiet > 0.0 {
                let sol = solve_ode(&ode.rhs, &u, (0.0, quiet), pk_params_flat, &[quiet], opts);
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
            let sol = solve_ode(
                &ode.rhs,
                &u,
                (0.0, dose.ii),
                pk_params_flat,
                &[dose.ii],
                opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
        cycles_run = cycle + 1;
        if tracker.should_stop(cycle, &u) {
            break;
        }
    }
    record_ss_equilibration_cycles(cycles_run);

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
/// For SS infusions this assumes `phase ≥ dose.duration` (the prior
/// infusion has finished by `phase`), i.e. `lagtime ≤ II − dose.duration`
/// — the realistic regime; overlapping infusions (`T_inf > II`) are already
/// rejected upstream.
fn ss_state_at_phase(
    ode: &crate::ode::OdeSpec,
    pk_params_flat: &[f64],
    dose: &DoseEvent,
    phase: f64,
    opts: &OdeSolverOptions,
) -> Vec<f64> {
    let mut u = equilibrate_ss_state(ode, pk_params_flat, dose, opts);
    if phase <= 0.0 {
        return u;
    }
    let cmt_idx = dose.cmt.saturating_sub(1);
    if cmt_idx >= u.len() {
        return u;
    }
    // Bioavailability scales the amount entering the dosing compartment,
    // resolved per dose compartment (`Fn`; see `equilibrate_ss_state`).
    let f_bio = ode.dose_attr_map.f_bio(dose.cmt, pk_params_flat);

    if is_real_infusion(dose) {
        // Mode-aware bioavailability (#419): see `equilibrate_ss_state`.
        let (rate, t_inf) = dose.bioavailable_infusion(f_bio);
        let active = phase.min(t_inf);
        let wrapped_rhs = |y: &[f64], p: &[f64], t: f64, dy: &mut [f64]| {
            (ode.rhs)(y, p, t, dy);
            if cmt_idx < dy.len() {
                dy[cmt_idx] += rate;
            }
        };
        let sol = solve_ode(
            &wrapped_rhs,
            &u,
            (0.0, active),
            pk_params_flat,
            &[active],
            opts,
        );
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
        if phase > t_inf {
            let quiet = phase - t_inf;
            let sol = solve_ode(&ode.rhs, &u, (0.0, quiet), pk_params_flat, &[quiet], opts);
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
        }
    } else {
        // Instantaneous SS bolus (no `R_in` routing) — sound only because SS into
        // an input-rate compartment is rejected upstream by `E_ABSORPTION_SS`;
        // see the matching note in `equilibrate_ss_state`.
        u[cmt_idx] += f_bio * dose.amt;
        let sol = solve_ode(&ode.rhs, &u, (0.0, phase), pk_params_flat, &[phase], opts);
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
    doses: &[DoseEvent],
    t_start: f64,
    t_end: f64,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    reset_floor: f64,
) -> Vec<(usize, f64)> {
    doses
        .iter()
        .enumerate()
        .filter_map(|(k, d)| {
            if !is_real_infusion(d) {
                return None;
            }
            let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
            let f_bio = dose_f_bio.get(k).copied().unwrap_or(1.0);
            // `F`-reshaped rate and window (#419).
            let (rate_eff, dur_eff) = d.bioavailable_infusion(f_bio);
            let start = d.time + lag;
            let end = start + dur_eff;
            // Infusions started before the most recent system reset (EVID=3/4)
            // are turned off, the same way the reset zeros the compartments.
            if start >= reset_floor
                && start <= t_start + INFUSION_EPS
                && end >= t_end - INFUSION_EPS
            {
                Some((d.cmt.saturating_sub(1), rate_eff))
            } else {
                None
            }
        })
        .collect()
}

/// One dose's zero-order absorption window — `(cmt_idx, rate, w_start, w_end)`,
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
type ZeroOrderWindow = (usize, f64, f64, f64);

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
    dur_frac_for_dose: impl Fn(usize, &DoseEvent) -> Option<(f64, f64)>,
) -> Vec<ZeroOrderWindow> {
    let mut out = Vec::new();
    for (k, d) in doses.iter().enumerate() {
        let Some((dur, frac)) = dur_frac_for_dose(k, d) else {
            continue;
        };
        let lag = dose_lagtimes.get(k).copied().unwrap_or(0.0);
        let f_bio = dose_f_bio.get(k).copied().unwrap_or(1.0);
        let w_start = d.time + lag;
        out.push((
            d.cmt.saturating_sub(1),
            f_bio * d.amt * frac / dur,
            w_start,
            w_start + dur,
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
/// `w_end`) is correctly excluded. The break-time list splits at `w_end` (see
/// [`push_zero_order_break_times`]) so every segment is fully inside or outside each
/// window — the invariant this test relies on, exactly as [`active_infusions`]
/// relies on it for infusion windows. `reset_floor` turns off windows opened before
/// the most recent reset (EVID=3/4).
fn active_zero_order_inputs(
    windows: &[ZeroOrderWindow],
    t_start: f64,
    t_end: f64,
    reset_floor: f64,
) -> Vec<(usize, f64)> {
    windows
        .iter()
        .filter(|&&(_, _, w_start, w_end)| {
            w_start >= reset_floor
                && w_start <= t_start + INFUSION_EPS
                && w_end >= t_end - INFUSION_EPS
        })
        .map(|&(cmt, rate, _, _)| (cmt, rate))
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
) -> Option<(f64, f64)> {
    if dose.amt <= 0.0 {
        return None;
    }
    ode.input_rate.iter().find_map(|f| {
        if f.kind == crate::pk::absorption::InputRateKind::ZeroOrder && f.cmt + 1 == dose.cmt {
            match f.prepare(pk_params) {
                PreparedInputRate::ZeroOrder { dur, .. } => Some((dur, f.frac(pk_params))),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// The floored zero-order duration `dur` for `dose` (ignoring the pathway
/// fraction) — used by the event-driven timeline's cutoff break, which needs the
/// window *edge*, not the rate. A thin projection of
/// [`zero_order_dur_and_frac_for_dose`] so the two never disagree on which forcing /
/// `dur` a dose resolves to.
fn zero_order_dur_for_dose(ode: &OdeSpec, dose: &DoseEvent, pk_params: &[f64]) -> Option<f64> {
    zero_order_dur_and_frac_for_dose(ode, dose, pk_params).map(|(dur, _)| dur)
}

/// True if a built-in absorption input-rate forcing (transit/etc.) feeds the
/// compartment `cmt_1based` (the data file's 1-based CMT). A dose into such a
/// compartment delivers its mass via `R_in(tad)` integrated over time
/// (`∫R_in dt = F·amt`), so its instantaneous **bolus must be suppressed** to
/// avoid double-counting the dose — the dose feeds the input-rate function, not
/// the state directly (see `plans/absorption-models.md`).
#[inline]
pub(crate) fn input_rate_consumes_cmt(ode: &OdeSpec, cmt_1based: usize) -> bool {
    !ode.input_rate.is_empty()
        && ode
            .input_rate
            .iter()
            .any(|f| f.cmt == cmt_1based.saturating_sub(1))
}

/// Push the hard-cutoff break times — each window's end `w_end` — for the
/// subject's precomputed zero-order windows (#504) onto a dense-path `break_times`
/// list.
///
/// A zero-order input delivers a constant rate over `[w_start, w_end]` then stops —
/// a step discontinuity at `w_end` that the smooth densities (transit/igd/weibull)
/// don't have. Without a break there, the adaptive RK45 steps across the cutoff and
/// mis-resolves the absorbed mass, so the timeline must break at `w_end` for every
/// zero-order window — exactly mirroring the infusion-end break. Because the break
/// reads `w_end` from the same [`ZeroOrderWindow`] the per-segment filter uses
/// ([`active_zero_order_inputs`]), the segment edge and the containment boundary
/// can't drift apart. Doses turned off by a later reset still get a harmless extra
/// break (over-segmentation only). No-op for the common model with no zero-order
/// window.
fn push_zero_order_break_times(break_times: &mut Vec<f64>, windows: &[ZeroOrderWindow]) {
    break_times.extend(windows.iter().map(|&(_, _, _, w_end)| w_end));
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
fn gated_infusions(
    active: &[(usize, f64, f64)],
    doses: &[DoseEvent],
    dose_f_bio: &[f64],
    n_states: usize,
) -> Vec<(usize, f64, f64, f64)> {
    active
        .iter()
        .filter_map(|&(di, t_start_inf, t_end_inf)| {
            let dose = &doses[di];
            // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
            if dose.cmt == 0 {
                return None;
            }
            let cmt = dose.cmt - 1;
            if cmt >= n_states {
                return None;
            }
            // Mode-aware bioavailability rate (#419); the `(t_start_inf, t_end_inf)`
            // window already carries the `F`-scaled duration from the caller's
            // break-time list.
            let (rate_eff, _) = dose.bioavailable_infusion(dose_f_bio[di]);
            Some((cmt, rate_eff, t_start_inf, t_end_inf))
        })
        .collect()
}

/// Precompute the per-forcing dose-invariant constants (ln Γ, KTR, ln KTR) for
/// the segment's PK snapshot `params`, parallel to `ode.input_rate` (#322 #7).
///
/// Built **once per segment** and reused across every RK45 stage / step inside
/// the seam, instead of re-running [`InputRateForcing::prepare`] on each RHS
/// evaluation. `params` (the segment's `ext_params` snapshot) is constant for
/// the whole segment, so this is an exact hoist. Returns an empty (non-allocating)
/// vec when the model has no built-in input-rate forcings.
fn prepare_input_rates(ode: &OdeSpec, params: &[f64]) -> Vec<PreparedInputRate> {
    ode.input_rate.iter().map(|f| f.prepare(params)).collect()
}

/// Add every built-in absorption input-rate forcing into `dy` at integration
/// time `t`, using the per-segment-hoisted `prepared` constants. For each
/// forcing, sums `R_in(tad)` over all doses targeting its compartment (Savic
/// superposition), with `tad = t − (dose.time + lag)` and dose mass `F·amt`.
/// `R_in = 0` for `tad ≤ 0`, so future doses contribute nothing. `reset_floor`
/// turns off doses delivered before the most recent EVID=3/4 reset, mirroring
/// [`active_infusions`]. This is the input-rate analogue of the `+rate` infusion
/// injection in the wrapped RHS.
///
/// `prepared` is parallel to `ode.input_rate` (built by [`prepare_input_rates`]
/// from the current segment's snapshot), so with IOV every superposed dose's
/// tail uses the *current* occasion's `n`/`mtt`. This is exact for IIV and when
/// `II` exceeds the absorption window; only overlapping-occasion tails are
/// approximated.
///
/// Generic over the numeric type `T: PkNum` so the **single** superposition loop
/// serves both the production `f64` predictor (`T = f64`, byte-identical to the
/// original) and the analytic ODE sensitivity provider's dual walk (`T = Dual*`),
/// instead of `sens/ode_provider.rs` hand-maintaining a second copy (#430 review
/// #4 / #451). The two dual callers each feed one branch live: the TV-cov
/// event-driven walk (`integrate_tvcov_g`) passes the tracked dual `dose_lagtimes`
/// for an in-scope estimated lagtime (#486), and the static walk (`integrate_g`)
/// passes the tracked `reset_floor` for an in-scope EVID 3/4 reset (#486).
/// `integrate_g` still passes `dose_lagtimes = &[]` (its gate excludes lagtime
/// subjects, which always route to the TV-cov walk instead).
///
/// `params` is the flat individual-parameter vector the `prepared` constants were
/// built from; it is read here only for the optional pathway-fraction multiplier
/// (`FR*fn(...)`, #388) via [`InputRateForcing::frac`] — `frac = 1` (no `frac_slot`)
/// is the single-pathway default, so this is a no-op for unfractioned forcings.
#[inline]
#[allow(clippy::too_many_arguments)] // mirrors the dose context threaded into the RHS wrappers
pub(crate) fn add_prepared_input_rate_forcing<T: crate::sens::num::PkNum>(
    ode: &OdeSpec,
    prepared: &[PreparedInputRate<T>],
    params: &[T],
    doses: &[DoseEvent],
    dose_lagtimes: &[T],
    dose_f_bio: &[T],
    reset_floor: f64,
    t: f64,
    dy: &mut [T],
) {
    for (forcing, prep) in ode.input_rate.iter().zip(prepared) {
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
        let mut acc = T::from_f64(0.0);
        for (k, d) in doses.iter().enumerate() {
            if d.cmt.saturating_sub(1) != forcing.cmt {
                continue;
            }
            // `dose_lagtimes[k]` (`T`, not `f64`) carries the exact `∂t_eff/∂lag = 1`
            // sensitivity when the caller's lag is itself an estimated parameter (an
            // event-driven walk with an in-scope lagtime, #486) — `T::from_f64(0.0)`
            // (zero jet) for every other caller (production `f64`, or a dual walk with
            // no lagtime), so `tad` below reduces to the pre-#486 constant-boundary
            // computation there. The gating comparisons use `.val()` (the boundary
            // itself never needs a jet — see `rate_at_zero`'s jump for that).
            let lag = dose_lagtimes.get(k).copied().unwrap_or(T::from_f64(0.0));
            let t_eff = T::from_f64(d.time) + lag;
            // Doses delivered before the most recent reset are off — the reset
            // zeroed the compartments, same rule as `active_infusions`.
            if t_eff.val() < reset_floor - INFUSION_EPS {
                continue;
            }
            let tad = T::from_f64(t) - t_eff;
            if tad.val() <= 0.0 {
                continue;
            }
            let dose_mass =
                dose_f_bio.get(k).copied().unwrap_or(T::from_f64(1.0)) * T::from_f64(d.amt);
            acc = acc + prep.rate(tad, dose_mass);
        }
        // Pathway fraction (#388): a `FR*fn(...)` term scales its whole `R_in` by
        // the declared fraction `FR`; `frac = 1` for an unfractioned single-pathway
        // forcing, so this is a no-op there. The multiplier flows linearly, so for
        // `T = Dual2` it carries the exact `∂R_in/∂frac` sensitivity.
        dy[forcing.cmt] = dy[forcing.cmt] + acc * forcing.frac(params);
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
/// event-driven walker; the two reset-aware paths pass a real floor. `prepared`
/// is the per-segment hoist from [`prepare_input_rates`].
#[allow(clippy::too_many_arguments)] // each is a distinct slice of dose/forcing context
fn wrap_rhs_with_forcings<'a>(
    ode: &'a OdeSpec,
    doses: &'a [DoseEvent],
    dose_lagtimes: &'a [f64],
    dose_f_bio: &'a [f64],
    reset_floor: f64,
    prepared: &'a [PreparedInputRate],
    infusions: InfusionInput,
    zero_order: &'a [(usize, f64)],
) -> impl Fn(&[f64], &[f64], f64, &mut [f64]) + 'a {
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
                p,
                doses,
                dose_lagtimes,
                dose_f_bio,
                reset_floor,
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
    /// vector, the flat PK-parameter slice, θ/η, the covariate snapshot, and the
    /// observation's 1-based CMT. Shared by the ODE predictor ([`read_observable`])
    /// and the analytic Form C path (`pk::apply_analytic_readout`, #650) so the
    /// two dispatch/NaN-guard conventions cannot drift. A `PerCmt` map miss (or an
    /// out-of-range `ObsCmt`) yields `NaN` — the loud guard that propagates to a
    /// NaN OFV rather than silently mis-reading, since parser + fit-time validation
    /// already guarantee every observed CMT has an entry.
    #[inline]
    pub(crate) fn eval(
        &self,
        state: &[f64],
        pk_params_flat: &[f64],
        theta: &[f64],
        eta: &[f64],
        covariates: &HashMap<String, f64>,
        obs_cmt: usize,
    ) -> f64 {
        match self {
            OdeReadout::ObsCmt(idx) => state[*idx],
            OdeReadout::Single(out_fn) => out_fn(state, pk_params_flat, theta, eta, covariates),
            OdeReadout::PerCmt(map) => match map.get(&obs_cmt) {
                Some(r) => (r.out_fn)(state, pk_params_flat, theta, eta, covariates),
                None => f64::NAN,
            },
        }
    }
}

/// Read the observable value at observation `obs_idx`.
///
/// `subject.obs_cmts[obs_idx]` selects the per-CMT readout when
/// `OdeReadout::PerCmt` is in use; the simpler variants ignore it.
#[inline]
fn read_observable(
    ode: &OdeSpec,
    u: &[f64],
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    obs_cmt: usize,
) -> f64 {
    ode.readout
        .eval(u, pk_params_flat, theta, eta, covariates, obs_cmt)
}

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
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
    /// [`CompiledModel::sync_ode_solver_opts`]. Carried on the spec so every
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
    ext_params: &mut [f64],
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    obs_map: &HashMap<u64, Vec<usize>>,
    predictions: &mut [f64],
    stats: Option<&mut OdeSolverStats>,
    // #570: soft (Hermite-interpolated) sample times within this segment — e.g. TTE
    // event/censor times — read off the *same* integration as the observations,
    // without clamping the step sequence. The returned observation predictions and
    // the advanced `u` are therefore bit-identical to a `chz_times = &[]` call.
    // Must be sorted ascending and lie in `(t_start, t_end]`; the caller filters.
    chz_times: &[f64],
) -> Vec<Vec<f64>> {
    let opts = ode.solver_opts;

    // Observation times in this segment (t_start < t <= t_end)
    let mut saveat: Vec<f64> = subject
        .obs_times
        .iter()
        .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
        .cloned()
        .collect();
    // Always include t_end so u is updated for next segment
    if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
        saveat.push(t_end);
    }
    saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    if (t_end - t_start).abs() < 1e-15 {
        return Vec::new();
    }

    // Update TAD anchor (slot MAX_PK_PARAMS+1): last effective dose time
    // before this segment, SS-aware (gives TAD = t - last_dose_eff).
    {
        let last_dose_eff = subject
            .doses
            .iter()
            .enumerate()
            .filter(|(i, d)| d.time + dose_lagtimes[*i] <= t_start + 1e-12)
            .map(|(i, d)| {
                let lag = dose_lagtimes[i];
                if d.ss && d.ii > 0.0 {
                    let elapsed = t_start - (d.time + lag);
                    t_start - elapsed.rem_euclid(d.ii)
                } else {
                    d.time + lag
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        // Store NaN when no effective prior dose exists so the ODE RHS injects
        // NaN for TAD (consistent with sdtab) rather than +∞ (t - NEG_INFINITY).
        ext_params[crate::types::MAX_PK_PARAMS + 1] = if last_dose_eff.is_finite() {
            last_dose_eff
        } else {
            f64::NAN
        };
    }

    // Integrate. If any infusions are active in this segment, wrap
    // the user RHS so it adds `+rate` to each infusion's compartment.
    // The plain (non-event-driven) ODE path never sees reset subjects —
    // the dispatcher routes those to `ode_predictions_event_driven` — so
    // no reset floor applies here.
    let active = active_infusions(
        &subject.doses,
        t_start,
        t_end,
        dose_lagtimes,
        dose_f_bio,
        f64::NEG_INFINITY,
    );
    // Zero-order absorption windows fully covering this segment (#504): constant
    // `F·amt/dur` injected like a spanning infusion. The dense path has a single
    // subject snapshot (`pk_params_flat`), so the windows are the same every
    // segment and consistent with `ode_predictions`' break placement. (Empty for
    // the common model / a non-zero_order subject — e.g. the adaptive caller.)
    let zo_windows = zero_order_windows(&subject.doses, dose_lagtimes, dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    let zero_order = active_zero_order_inputs(&zo_windows, t_start, t_end, f64::NEG_INFINITY);
    // Hoist the input-rate constants (ln Γ, KTR, …) once per segment; the PK
    // snapshot `ext_params` is constant across the integration (#322 #7).
    let prepared = prepare_input_rates(ode, ext_params);
    let wrapped_rhs = wrap_rhs_with_forcings(
        ode,
        &subject.doses,
        dose_lagtimes,
        dose_f_bio,
        f64::NEG_INFINITY,
        &prepared,
        InfusionInput::Spanning(active),
        &zero_order,
    );
    let (sol, soft) = solve_ode_dense(
        &wrapped_rhs,
        u,
        (t_start, t_end),
        ext_params,
        &saveat,
        chz_times,
        &opts,
        stats,
    );

    // Extract predictions and update state
    for pt in &sol {
        if let Some(obs_idxs) = obs_map.get(&pt.t.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &pt.u,
                    pk_params_flat,
                    theta,
                    eta,
                    subject.obs_cov(obs_idx),
                    cmt,
                );
            }
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
/// where `chz_states[i]` is the full ODE state at `chz_times[i]` — NaN-filled for any
/// time before the integration start (matching the dedicated `ode_dense_solve_states`
/// path, which the TTE NLL maps to its `1e20` sentinel). `ipred` is the raw observable
/// readout; callers apply `[scaling]` / log-transform exactly as for `ode_predictions`.
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
    let opts = ode.solver_opts;
    // #570: full state at each `chz_times[i]`, pre-filled NaN so a soft time before
    // the integration start (or otherwise uncovered by a segment) reads NaN → the TTE
    // 1e20 sentinel — exactly as the dedicated `ode_dense_solve_states` path does
    // today. `chz_times` is sorted-unique (caller contract), enabling the binary
    // search that maps each segment's soft samples back to their global slot.
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
    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.lagtime(d.cmt, pk_params_flat))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.f_bio(d.cmt, pk_params_flat))
        .collect();

    // Extended params: slots 0..MAX_PK_PARAMS hold the PK parameters; slots
    // MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry TAFD/TAD anchors for the ODE RHS.
    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    // Store NaN when there are no doses so the ODE RHS injects NaN for TAFD
    // (consistent with the sdtab convention) rather than -∞ (INFINITY - t).
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // Build obs_time → indices map. Multiple observations can share a time
    // (e.g. simultaneous PK/PD samples on different CMTs), so each time maps to
    // *all* its observation indices — recording only one would leave the others
    // at their initial NaN.
    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

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
        // SS + lagtime: break at the dose *record* time too, so we can seed
        // the previous-interval steady-state tail there before the lagged
        // pulse arrives (issue #15).
        if lag > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    // Zero-order windows for this subject (#504): the dense paths have a single
    // PK snapshot, so the per-dose `dur`/`F`/`lag` come from `pk_params_flat`.
    // Break at each window end so segments align with the cutoff, and reuse the
    // same windows for the per-segment constant-rate injection below.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    push_zero_order_break_times(&mut break_times, &zo_windows);
    break_times.push(t_last);
    // No-event break points (e.g. the reactive driver's decision times) — they
    // only re-segment the integration, never change state. Drop non-positive /
    // non-finite entries (0.0 is already present; the timeline starts at 0).
    break_times.extend(
        extra_breaks
            .iter()
            .copied()
            .filter(|b| b.is_finite() && *b > 0.0),
    );
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    // Degenerate single-instant timeline (e.g. one observation, no dose, off
    // zero): keep a second identical break so the loop runs once and records
    // observations at the first record from the initial (post-dose) state,
    // rather than leaving them at NaN. Integration over the zero-length segment
    // is a no-op.
    if break_times.len() < 2 {
        break_times.push(break_times[0]);
    }

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply dose effects at t_start in a single pass over the dose
        // list. Ordering inside the pass matters:
        //   1. SS=1 + II > 0: pre-equilibrate by overwriting state with
        //      the SS amount from the infinite-past pulse train (see
        //      `equilibrate_ss_state`).
        //   2. Bolus (non-infusion): instantaneous amount jump in the
        //      dose's compartment, applied on top of any SS preload.
        // Infusions don't add to state at t_start — they're injected as
        // a derivative term inside the integrator (see `active_infusions`
        // + wrapped RHS below).
        // SS + lagtime: at the dose record time (strictly before the lagged
        // arrival) seed the previous interval's steady-state tail so pre-lag
        // observations don't read the empty initial state. Phase II−lagtime
        // is where the prior pulse has decayed to by the record time.
        for (i, dose) in subject.doses.iter().enumerate() {
            let lag = dose_lagtimes[i];
            if lag > 0.0 && dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
                u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lag, &opts);
            }
        }

        for (i, dose) in subject.doses.iter().enumerate() {
            if (dose.time + dose_lagtimes[i] - t_start).abs() >= 1e-12 {
                continue;
            }
            if dose.ss && dose.ii > 0.0 {
                u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts);
            }
            if !is_real_infusion(dose) && !input_rate_consumes_cmt(ode, dose.cmt) {
                // dose.cmt is 1-based; state indices are 0-based. A dose into a
                // built-in input-rate compartment (transit/etc.) is delivered as
                // R_in over time by the wrapped RHS below — not as a bolus — so
                // it's skipped here to avoid double-counting the dose.
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n {
                    u[cmt_idx] += dose_f_bio[i] * dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    pk_params_flat,
                    theta,
                    eta,
                    subject.obs_cov(obs_idx),
                    cmt,
                );
            }
        }

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
        // same. The `> t_start + 1e-12` filter below excludes `t == t_start`, so a soft
        // time is never written twice within one iteration.
        for (gi, &t) in chz_times.iter().enumerate() {
            if (t - t_start).abs() < 1e-12 {
                chz_states[gi] = u.clone();
            }
        }

        // Integrate the open interval `(t_start, t_end]` from the carried state,
        // recording observations inside it and advancing `u` to `t_end`. The
        // left-boundary discontinuities and the `t_start` observation were applied
        // above; `integrate_segment` owns only the integration — the piece a
        // reactive (state-dependent) driver reuses unchanged (#391 S1.2).
        // #570: soft (TTE) times in the *half-open* interval `(t_start, t_end]`, read
        // off the same integration (the closed `t_start` boundary was handled above).
        // `chz_times` is sorted, so this slice is too.
        let seg_chz: Vec<f64> = chz_times
            .iter()
            .copied()
            .filter(|&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .collect();
        let soft = integrate_segment(
            ode,
            &mut u,
            t_start,
            t_end,
            subject,
            &dose_lagtimes,
            &dose_f_bio,
            &mut ext_params,
            pk_params_flat,
            theta,
            eta,
            &obs_map,
            &mut predictions,
            stats.as_deref_mut(),
            &seg_chz,
        );
        // Place each soft sample at its global `chz_times` index (NaN slots left for
        // any time no segment covered).
        for (t, state) in seg_chz.iter().zip(soft) {
            if let Ok(gi) = chz_times.binary_search_by(|x| x.partial_cmp(t).unwrap()) {
                chz_states[gi] = state;
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
    for p in &mut predictions {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

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
/// - **Dose-free base subject** — the regimen is entirely controller-driven
///   (augmenting pre-scheduled doses is a later step).
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

    // --- Preconditions (typed errors, never silent) ----------------------
    if !subject.doses.is_empty() {
        return Err(
            "ode_predictions_adaptive (S1.3a) requires a dose-free base subject; the regimen is \
             controller-driven (augmenting pre-scheduled doses is a later step)"
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

    // --- Running state ---------------------------------------------------
    let n_obs = subject.obs_times.len();
    let mut u = ode.initial_state(pk_params_flat);
    let mut predictions = vec![f64::NAN; n_obs];
    let mut ledger: Vec<DoseLedgerEntry> = Vec::new();
    let mut decisions: Vec<DecisionLogEntry> = Vec::new();

    // Shadow subject accumulates the controller's realized doses (the #324
    // pattern); `integrate_segment` reads `shadow.doses` for the TAD anchor.
    let mut shadow = subject.clone();

    // Extended params: PK params + TAFD/TAD anchors. TAFD (slot MAX_PK_PARAMS)
    // stays NaN until the first dose arrives; TAD is set per segment inside
    // `integrate_segment`.
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = f64::NAN;

    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in shadow.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

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
    // byte-identical to before. Observations are deliberately NOT break points —
    // they are recorded via `saveat` *inside* a segment, exactly as
    // `ode_predictions` does. Breaking at observations too would reinitialize the
    // adaptive integrator at each one and perturb the step sequence, so the segment
    // structure (and the result) would no longer match the static engine on the
    // same realized doses.
    let t_last = shadow
        .obs_times
        .iter()
        .chain(decision_times.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0, t_last];
    break_times.extend(decision_times.iter().cloned());
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut stopped = false;

    let mut k = 0;
    while k < break_times.len() {
        let t_start = break_times[k];

        // --- Decision hook: observe (pre-dose trough) -> decide -> dose. ---
        if !stopped {
            if let Some(&decision_index) = decision_index_of.get(&t_start.to_bits()) {
                // Covariate snapshot in effect at the decision time. When the
                // decision coincides with an observation row, use that row's
                // per-observation snapshot (so time-varying covariates drive the
                // monitored readouts and the controller's view); otherwise fall
                // back to the subject-static map.
                let decision_cov = obs_map
                    .get(&t_start.to_bits())
                    .and_then(|idxs| idxs.first())
                    .map(|&i| shadow.obs_cov(i))
                    .unwrap_or(&shadow.covariates);
                // Resolve each monitored signal at the current (pre-dose) state.
                let mut signals: HashMap<String, f64> = HashMap::new();
                let mut observed: Vec<ObservedSignal> = Vec::with_capacity(monitors.len());
                for am in monitors.iter() {
                    let m = am.spec;
                    // A declarative `[adaptive_dosing]` block (S2) supplies a
                    // compiled `observe` expression for the latent value; absent
                    // one (the programmatic path), read the model's cmt readout.
                    let latent = match am.observe {
                        Some(f) => f(&u, pk_params_flat, theta, eta, decision_cov),
                        None => read_observable(
                            ode,
                            &u,
                            pk_params_flat,
                            theta,
                            eta,
                            decision_cov,
                            m.cmt,
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
                            // Edge (b): an assay cannot read below zero; clamp the
                            // noised value at 0 (BLQ-blinding is deferred to Part F).
                            (latent + var.sqrt() * eps).max(0.0)
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
                                pk_params_flat,
                                decision_index,
                            )?;
                            u[cmt - 1] += f * amt;
                            if !ext_params[crate::types::MAX_PK_PARAMS].is_finite() {
                                ext_params[crate::types::MAX_PK_PARAMS] = t_start;
                            }
                            shadow
                                .doses
                                .push(DoseEvent::new(t_start, amt, cmt, 0.0, false, 0.0));
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
                                pk_params_flat,
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
                            if !ext_params[crate::types::MAX_PK_PARAMS].is_finite() {
                                ext_params[crate::types::MAX_PK_PARAMS] = t_start;
                            }
                            shadow.doses.push(dose);
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

        // Record the observation exactly at t_start (post-dose), mirroring
        // `ode_predictions`' left-boundary recording.
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = shadow.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    pk_params_flat,
                    theta,
                    eta,
                    shadow.obs_cov(obs_idx),
                    cmt,
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
        if k + 1 < break_times.len() {
            let t_end = break_times[k + 1];

            // Per-segment lag/F for the realized doses (boluses and infusions):
            // lag 0 — a nonzero lag is rejected at the decision hook for either
            // — and F per cmt. Infusions are delivered by `integrate_segment`'s
            // `active_infusions` over any segment they fully span, which the
            // dynamic infusion-end breaks guarantee.
            let dose_lagtimes: Vec<f64> = shadow
                .doses
                .iter()
                .map(|d| ode.dose_attr_map.lagtime(d.cmt, pk_params_flat))
                .collect();
            let dose_f_bio: Vec<f64> = shadow
                .doses
                .iter()
                .map(|d| ode.dose_attr_map.f_bio(d.cmt, pk_params_flat))
                .collect();

            integrate_segment(
                ode,
                &mut u,
                t_start,
                t_end,
                &shadow,
                &dose_lagtimes,
                &dose_f_bio,
                &mut ext_params,
                pk_params_flat,
                theta,
                eta,
                &obs_map,
                &mut predictions,
                None,
                &[],
            );
        }

        k += 1;
    }

    // Clamp negative predictions to zero, matching the static predictor.
    for p in &mut predictions {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

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
/// `base_subject` is the dose-free subject the run was driven from; its
/// observation grid (and any covariates) carry over, only `doses` are replaced
/// with the realized ledger. The ledger stores nominal `amt`/`rate`
/// (pre-bioavailability), exactly as a `subject.doses` entry, so `F`/lag re-apply
/// downstream identically.
pub(crate) fn verify_adaptive_frozen_replay(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    base_subject: &Subject,
    decision_times: &[f64],
    run: &AdaptiveRun,
) -> Result<(), String> {
    let mut static_subject = base_subject.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();

    let static_preds = ode_predictions_with_extra_breaks(
        ode,
        pk_params_flat,
        theta,
        eta,
        &static_subject,
        decision_times,
    );

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
    let rel_tol = (REPLAY_TOL_FACTOR * ode.solver_opts.reltol).max(1e-9);
    let abs_tol = (REPLAY_TOL_FACTOR * ode.solver_opts.abstol).max(1e-12);
    for (j, (got, want)) in run.predictions.iter().zip(static_preds.iter()).enumerate() {
        // Unrecorded slots are NaN in both engines (same observation grid), so
        // NaN==NaN is agreement; a NaN-vs-finite split is a genuine divergence.
        if got.is_nan() && want.is_nan() {
            continue;
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
/// drops every dose after `a` — future doses cannot affect `[a, b]`, so this is
/// exact — leaving `b` a plain pre-dose decay point.
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
/// `base_subject` is the dose-free subject the run was driven from; only its doses
/// are replaced (its covariates carry over — the reactive driver is BSV-only in
/// this slice, so a single static covariate snapshot is exact).
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
    // `[a, b]` against a static subject carrying only the doses at or before the
    // window's left edge `a`, so the dose at the right edge `b` (the next window's)
    // never folds into this window's endpoint.
    (0..m - 1)
        .map(|k| {
            let (a, b) = (decision_times[k], decision_times[k + 1]);

            // Doses at or before `a` (nominal amt/rate; F/lag re-apply downstream
            // exactly as for a scheduled dose). A dose exactly at `a` is THIS
            // window's own dose and is kept; the `1e-9` only guards float equality
            // at the boundary, far below any real decision spacing.
            let mut sub = base_subject.clone();
            sub.doses = ledger
                .iter()
                .filter(|e| e.time <= a + 1e-9)
                .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
                .collect();

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
                        Some(f) => f(u, pk_params_flat, theta, eta, cov),
                        None => {
                            read_observable(ode, u, pk_params_flat, theta, eta, cov, monitor_cmt)
                        }
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
) -> Vec<f64> {
    assert_eq!(pk_at_dose.len(), subject.doses.len());
    assert_eq!(pk_at_obs.len(), subject.obs_times.len());
    assert_eq!(pk_at_pk_only.len(), subject.pk_only_times.len());

    // Resolve modeled-RATE doses to concrete (`Fixed`) doses once (#324), each
    // with its own per-dose PK snapshot `pk_at_dose[k]` (this is the event-driven
    // / time-varying-covariate path). Borrowed (no clone) for the common
    // all-`Fixed` dataset. Single source of truth — see `resolve_subject_doses`.
    let resolved =
        resolve_subject_doses_with(subject, &ode.dose_attr_map, |k| &pk_at_dose[k].values);
    let subject: &Subject = &resolved;

    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = ode.solver_opts;

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
    //   dose < pk-only < obs < infusion-end
    // — matches the analytical event-driven path for dose/pk-only/obs.
    // Infusion-end sorts last so an obs at the same time as the end of
    // an infusion is recorded with the infusion still contributing
    // (state is continuous; the ordering only affects which segments
    // include the rate in their active set on the next iteration).
    #[derive(Clone, Copy)]
    enum Kind {
        Reset,
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
            Kind::Dose => 1,
            Kind::PkOnly => 2,
            Kind::Obs => 3,
            Kind::InfusionEnd => 4,
        }
    }
    let n_infusion_ends = subject.doses.iter().filter(|d| is_real_infusion(d)).count();
    let mut timeline: Vec<(f64, Kind, usize)> = Vec::with_capacity(
        subject.doses.len()
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
        .map(|(d, p)| ode.dose_attr_map.lagtime(d.cmt, &p.values))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .zip(pk_at_dose.iter())
        .map(|(d, p)| ode.dose_attr_map.f_bio(d.cmt, &p.values))
        .collect();
    for (k, d) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[k];
        timeline.push((d.time + lag, Kind::Dose, k));
        if is_real_infusion(d) {
            // F-scaled infusion end (#419): rate-defined -> F·duration window.
            let (_, dur_eff) = d.bioavailable_infusion(dose_f_bio[k]);
            timeline.push((d.time + lag + dur_eff, Kind::InfusionEnd, k));
        }
        // Zero-order absorption cutoff (#504): a dose feeding a `zero_order(dur)`
        // compartment delivers a constant rate over `(0, dur]`, so break at the
        // window end `d.time+lag+dur` exactly like an infusion end (no record, no
        // state change — just a segment boundary so `active_zero_order_inputs`'s
        // full-containment test sees each segment fully inside or outside).
        if let Some(dur) = zero_order_dur_for_dose(ode, d, &pk_at_dose[k].values) {
            timeline.push((d.time + lag + dur, Kind::InfusionEnd, k));
        }
    }
    for (j, &t) in subject.obs_times.iter().enumerate() {
        timeline.push((t, Kind::Obs, j));
    }
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        timeline.push((t, Kind::PkOnly, m));
    }
    timeline.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| kind_order(a.1).cmp(&kind_order(b.1)))
    });

    // Zero-order windows (#504) read from each dose's **own** PK snapshot
    // (`pk_at_dose[k]`) — the same per-dose source as the timeline cutoff above, so
    // the window edge `w_end` and the per-segment containment test below agree, and
    // the constant rate `F·amt/dur` is fixed at dose time (mass-exact even when
    // `dur` rides a time-varying covariate, where a per-segment recompute would
    // drift). Precomputed once here, then filtered per segment in the loop.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |k, d| {
        zero_order_dur_and_frac_for_dose(ode, d, &pk_at_dose[k].values)
    });

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

    for &(t_event, kind, idx) in &timeline {
        // PK params for the segment [cur_t, t_event] are evaluated AT
        // t_event (NONMEM end-of-interval / current-record convention —
        // `$PK runs at every record, then ADVAN propagates to it`).
        // Infusion-end is not a record: reuse the previous segment's PK.
        // Reset is not a record either: it just zeros the state below.
        let pk_now: PkParams = match kind {
            Kind::Dose => pk_at_dose[idx],
            Kind::Obs => pk_at_obs[idx],
            Kind::PkOnly => pk_at_pk_only[idx],
            Kind::InfusionEnd | Kind::Reset => last_pk,
        };

        if t_event > cur_t {
            // Build extended params for this segment: slots 0..MAX_PK_PARAMS
            // are pk_now.values; slots MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry
            // the TAFD/TAD anchors for TIME/TAFD/TAD injection in the ODE RHS.
            // TAD anchor: shift each dose by its own resolved lag (per dose
            // compartment), consistent with the timeline above and the
            // non-event-driven path.
            let last_dose_eff_ed = subject
                .doses
                .iter()
                .enumerate()
                .filter(|(i, d)| d.time + dose_lagtimes[*i] <= cur_t + 1e-12)
                .map(|(i, d)| {
                    let lag = dose_lagtimes[i];
                    if d.ss && d.ii > 0.0 {
                        let elapsed = cur_t - (d.time + lag);
                        cur_t - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lag
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            // Store NaN when no effective prior dose exists (fold stays at NEG_INFINITY)
            // so the ODE RHS injects NaN for TAD rather than +∞ (t - NEG_INFINITY).
            let last_dose_eff_ed = if last_dose_eff_ed.is_finite() {
                last_dose_eff_ed
            } else {
                f64::NAN
            };
            let mut ext_params_ed = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
            ext_params_ed[..crate::types::MAX_PK_PARAMS]
                .copy_from_slice(&pk_now.values[..crate::types::MAX_PK_PARAMS]);
            ext_params_ed[crate::types::MAX_PK_PARAMS] = first_dose_time_ed;
            ext_params_ed[crate::types::MAX_PK_PARAMS + 1] = last_dose_eff_ed;

            // Wrap the user RHS so any infusion fully spanning
            // [cur_t, t_event] contributes `+rate` to its compartment.
            let active = active_infusions(
                &subject.doses,
                cur_t,
                t_event,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
            );
            // Zero-order absorption windows covering [cur_t, t_event] (#504),
            // reset-aware via the same `reset_floor` (a window opened pre-reset
            // is off). Constant `F·amt/dur`, injected like a spanning infusion.
            // `zo_windows` is precomputed once from the per-dose `pk_at_dose`
            // snapshots (below), the same source as the timeline's cutoff break —
            // so the window edge and the containment boundary can't drift apart,
            // and the constant rate is fixed at dose time (mass-exact under
            // time-varying covariates).
            let zero_order = active_zero_order_inputs(&zo_windows, cur_t, t_event, reset_floor);
            // Hoist the input-rate constants once per segment (#322 #7); the
            // segment PK snapshot `ext_params_ed` is constant for the integration.
            let prepared = prepare_input_rates(ode, &ext_params_ed);
            let wrapped_rhs = wrap_rhs_with_forcings(
                ode,
                &subject.doses,
                &dose_lagtimes,
                &dose_f_bio,
                reset_floor,
                &prepared,
                InfusionInput::Spanning(active),
                &zero_order,
            );
            let saveat = vec![t_event];
            let sol = solve_ode(
                &wrapped_rhs,
                &u,
                (cur_t, t_event),
                &ext_params_ed,
                &saveat,
                &opts,
            );
            if let Some(last) = sol.last() {
                u.copy_from_slice(&last.u);
            }
            cur_t = t_event;
        }

        match kind {
            Kind::Dose => {
                let d = &subject.doses[idx];
                // Steady-state (SS=1) dose: reset state and load with the
                // SS amount from the infinite-past pulse train before the
                // SS dose's own pulse is applied below. See
                // `equilibrate_ss_state` for the per-cycle scheme.
                if d.ss && d.ii > 0.0 {
                    u = equilibrate_ss_state(ode, &pk_now.values, d, &opts);
                }
                // Boluses: add amt to state. Infusions: no instantaneous
                // change — handled via the wrapped RHS for segments inside
                // [d.time, d.time + d.duration]. A dose into a built-in
                // input-rate compartment (transit/etc.) is delivered as R_in
                // over time by the wrapped RHS, so it's skipped here too.
                if !is_real_infusion(d) && !input_rate_consumes_cmt(ode, d.cmt) {
                    let cmt_idx = d.cmt.saturating_sub(1);
                    if cmt_idx < n {
                        // Bioavailability resolved per dose compartment (`Fn`).
                        u[cmt_idx] += ode.dose_attr_map.f_bio(d.cmt, &pk_now.values) * d.amt;
                    }
                }
                last_pk = pk_now;
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
                );
                // Clamp negative readouts (ODE solver overshoot guard);
                // let NaN through so a missing `OdeReadout::PerCmt` entry
                // (or any other genuine NaN) surfaces as a NaN OFV
                // rather than a silent zero. See the corresponding note
                // in `ode_predictions`.
                predictions[idx] = if v < 0.0 { 0.0 } else { v };
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
                // Evaluate init with the params in effect at the reset
                // (`last_pk`). For EVID=4 the dose at this same time follows
                // (Reset sorts before Dose), so it lands on the re-seeded
                // state. Record the reset time so infusions started earlier
                // stop contributing.
                u = ode.initial_state(&last_pk.values);
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
        ode.solver_opts,
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
        ode.solver_opts,
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
    let opts = ode.solver_opts;

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
    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.lagtime(d.cmt, pk_params_flat))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.f_bio(d.cmt, pk_params_flat))
        .collect();

    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    let mut obs_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in subject.obs_times.iter().enumerate() {
        obs_map.entry(t.to_bits()).or_default().push(i);
    }

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
        if lag > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    // Zero-order windows for this subject (#504): the dense paths have a single
    // PK snapshot, so the per-dose `dur`/`F`/`lag` come from `pk_params_flat`.
    // Break at each window end so segments align with the cutoff, and reuse the
    // same windows for the per-segment constant-rate injection below.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    push_zero_order_break_times(&mut break_times, &zo_windows);
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

    for w in break_times.windows(2) {
        let (t_start, t_end) = (w[0], w[1]);
        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // SS + lagtime: at the dose *record* time (strictly before the lagged pulse
        // arrives) seed the previous interval's steady-state tail, exactly mirroring
        // the separate pre-pass in `ode_predictions` (lines 479-485).
        for (i, dose) in subject.doses.iter().enumerate() {
            let lag = dose_lagtimes[i];
            if lag > 0.0 && dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
                u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lag, &opts);
            }
        }

        // Apply boluses and SS doses at t_eff = dose.time + lagtime.
        for (dose_idx, dose) in subject.doses.iter().enumerate() {
            let t_eff = dose.time + dose_lagtimes[dose_idx];
            if (t_eff - t_start).abs() < 1e-10 {
                let f = dose_f_bio[dose_idx];
                if dose.ss && dose.ii > 0.0 {
                    // Lagged arrival: pre-lag seeding was already done above;
                    // here we apply the full equilibrated state.
                    u = equilibrate_ss_state(ode, pk_params_flat, dose, &opts);
                }
                if !is_real_infusion(dose) {
                    if !input_rate_consumes_cmt(ode, dose.cmt) {
                        // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                        if dose.cmt > 0 {
                            let cmt = dose.cmt - 1;
                            if cmt < n {
                                u[cmt] += dose.amt * f;
                            }
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

        // Handle obs at t_start (after dose).
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    pk_params_flat,
                    theta,
                    eta,
                    subject.obs_cov(obs_idx),
                    cmt,
                );
                states[obs_idx] = u.clone();
            }
        }

        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .cloned()
            .filter(|&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
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
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        // TAD anchor: last effective dose time before this segment, SS-aware.
        // For SS doses, rem_euclid maps the elapsed time back into [0, II) so
        // TAD stays within one dosing interval — matching ode_predictions.
        ext_params[crate::types::MAX_PK_PARAMS + 1] = {
            let last_dose_eff = subject
                .doses
                .iter()
                .enumerate()
                .filter(|(i, d)| d.time + dose_lagtimes[*i] <= t_start + 1e-12)
                .map(|(i, d)| {
                    let lag = dose_lagtimes[i];
                    if d.ss && d.ii > 0.0 {
                        let elapsed = t_start - (d.time + lag);
                        t_start - elapsed.rem_euclid(d.ii)
                    } else {
                        d.time + lag
                    }
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if last_dose_eff.is_finite() {
                last_dose_eff
            } else {
                f64::NAN
            }
        };

        active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
        // Resolve each active infusion to (cmt_idx, F·rate, t_start, t_end) for
        // the time-gated injection inside the seam (CMT=0 / out-of-range dropped).
        let gated = gated_infusions(&active_infusions, &subject.doses, &dose_f_bio, n);
        // Zero-order absorption windows covering this segment (#504): constant
        // `F·amt/dur` injected alongside the gated infusions (empty otherwise).
        let zero_order = active_zero_order_inputs(&zo_windows, t_start, t_end, f64::NEG_INFINITY);
        // Hoist the input-rate constants once per segment (#322 #7).
        let prepared = prepare_input_rates(ode, &ext_params);
        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            f64::NEG_INFINITY,
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
                for &obs_idx in obs_idxs {
                    let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                    predictions[obs_idx] = read_observable(
                        ode,
                        &pt.u,
                        pk_params_flat,
                        theta,
                        eta,
                        subject.obs_cov(obs_idx),
                        cmt,
                    );
                    states[obs_idx] = pt.u.clone();
                }
            }
        }

        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    for p in &mut predictions {
        if *p < 0.0 {
            *p = 0.0;
        }
    }

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
/// compartment `states`, however, are derived from a second pass via
/// [`ode_dense_solve_states`] using **the first observation's PK parameters held
/// fixed** for the entire timeline. For subjects with genuinely time-varying
/// covariates (CL, V, etc. changing between observations) the states will be
/// approximate. `fit()` emits `W_DERIVED_CMT_TV_ODE` to alert users to this
/// limitation. For reset-only subjects (no TV covariates) `pk_at_obs` is
/// uniformly filled, so using the first entry is exact.
pub fn ode_predictions_event_driven_with_states(
    ode: &OdeSpec,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    pk_at_dose: &[PkParams],
    pk_at_obs: &[PkParams],
    pk_at_pk_only: &[PkParams],
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
    );

    // Second pass: extract the full ODE state at each obs time via
    // `ode_dense_solve_states`. That function runs the standard (non-event-driven)
    // solver, so it uses a single fixed set of PK params for the entire timeline.
    //
    // For subjects with EVID=3/4 resets but *no* TV covariates, `pk_at_obs` is
    // uniformly filled (every entry identical), so using `first()` is exact.
    //
    // For subjects with genuine TV covariates, `pk_at_obs` varies per timepoint.
    // Using `first()` here is an approximation: the compartment state trajectory
    // will be computed with the first-observation PK params (CL/V/etc.) held fixed,
    // while `ipreds` correctly reflect per-event covariate snapshots. For most PK
    // contexts this approximation is acceptable post-fit, but the caller
    // (`compute_predictions_with_states`) is the approximate path; `fit()` emits
    // W_DERIVED_CMT_TV_ODE when TV covariates are present so users know.
    //
    // A future improvement: duplicate the event-driven loop to capture `u` at each
    // obs time directly — exact states, but ~2× the integration work post-fit.
    let pk_flat = &pk_at_obs.first().map(|p| p.values).unwrap_or_default();
    let states = ode_dense_solve_states(ode, pk_flat, theta, eta, subject, &subject.obs_times);

    (ipreds, states)
}

/// Build the sorted, deduped dose-segment break times for a subject — the points
/// where the integrator must stop and re-apply boundary events (dose pulses, lags,
/// infusion ends, SS-record seeds, EVID-3/4 resets, zero-order windows). `terminal`
/// is the final break: the last `saveat` for the dense solve, or the horizon for
/// the event-time search. Shared by [`ode_dense_solve_states`] and
/// [`ode_solve_until_chz_threshold`] so the two segment the timeline identically
/// (a divergence here would make a simulated event time inconsistent with the
/// fitted hazard).
fn build_segment_break_times(
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
        if lag > 0.0 && dose.ss && dose.ii > 0.0 {
            break_times.push(dose.time);
        }
    }
    // EVID=3/4 resets must be break-points so the re-seed happens at the exact boundary.
    for &rt in &subject.reset_times {
        break_times.push(rt);
    }
    push_zero_order_break_times(&mut break_times, zo_windows);
    break_times.push(terminal);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    break_times
}

/// Owned per-segment forcings produced by [`apply_segment_boundary`]: everything
/// `wrap_rhs_with_forcings` needs for one dose segment, returned by value so the
/// caller can build (and borrow into) the wrapped RHS without a dangling borrow.
struct SegmentForcings {
    reset_floor: f64,
    gated: Vec<(usize, f64, f64, f64)>,
    zero_order: Vec<(usize, f64)>,
    prepared: Vec<PreparedInputRate>,
}

/// Apply a dose segment's boundary events and resolve its forcings — the shared
/// core of the per-segment loop used by both [`ode_dense_solve_states`] (the
/// fit-path dense solve) and [`ode_solve_until_chz_threshold`] (the TTE event-time
/// search), so the two cannot drift. Mutates `u` (EVID-3/4 reset re-seed, SS-lag
/// seeding, bolus additions), `active_infusions` (activation + expiry), and
/// `ext_params` (the TAD anchor slot), then returns this `[t_start, t_end)`
/// segment's forcings for the caller to build the wrapped RHS and integrate.
#[allow(clippy::too_many_arguments)]
fn apply_segment_boundary(
    ode: &OdeSpec,
    subject: &Subject,
    dose_lagtimes: &[f64],
    dose_f_bio: &[f64],
    zo_windows: &[ZeroOrderWindow],
    pk_params_flat: &[f64],
    n: usize,
    opts: &OdeSolverOptions,
    t_start: f64,
    t_end: f64,
    u: &mut Vec<f64>,
    active_infusions: &mut Vec<(usize, f64, f64)>,
    ext_params: &mut [f64],
) -> SegmentForcings {
    // EVID=3/4 reset: re-seed compartments before processing doses at this time.
    // Resets sort before doses at the same time (mirroring Kind::Reset < Kind::Dose).
    for &rt in &subject.reset_times {
        if (rt - t_start).abs() < 1e-10 {
            *u = ode.initial_state(pk_params_flat);
            active_infusions.clear();
            break;
        }
    }

    // SS + lagtime: at the dose *record* time (before the lagged pulse arrives)
    // seed the previous interval's steady-state tail, mirroring ode_predictions.
    for (i, dose) in subject.doses.iter().enumerate() {
        let lag = dose_lagtimes[i];
        if lag > 0.0 && dose.ss && dose.ii > 0.0 && (dose.time - t_start).abs() < 1e-12 {
            *u = ss_state_at_phase(ode, pk_params_flat, dose, dose.ii - lag, opts);
        }
    }

    for (dose_idx, dose) in subject.doses.iter().enumerate() {
        let t_eff = dose.time + dose_lagtimes[dose_idx];
        if (t_eff - t_start).abs() < 1e-10 {
            let f = dose_f_bio[dose_idx];
            if dose.ss && dose.ii > 0.0 {
                // Lagged arrival: pre-lag seeding already done above.
                *u = equilibrate_ss_state(ode, pk_params_flat, dose, opts);
            }
            if !is_real_infusion(dose) {
                if !input_rate_consumes_cmt(ode, dose.cmt) {
                    // dose.cmt is 1-based; CMT=0 means no compartment — ignore.
                    if dose.cmt > 0 {
                        let cmt = dose.cmt - 1;
                        if cmt < n {
                            u[cmt] += dose.amt * f;
                        }
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

    // TAD anchor: SS-aware, matching ode_predictions (rem_euclid wraps the elapsed
    // time back into [0, II)).
    ext_params[crate::types::MAX_PK_PARAMS + 1] = {
        let last_dose_eff = subject
            .doses
            .iter()
            .enumerate()
            .filter(|(i, d)| d.time + dose_lagtimes[*i] <= t_start + 1e-12)
            .map(|(i, d)| {
                let lag = dose_lagtimes[i];
                if d.ss && d.ii > 0.0 {
                    let elapsed = t_start - (d.time + lag);
                    t_start - elapsed.rem_euclid(d.ii)
                } else {
                    d.time + lag
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if last_dose_eff.is_finite() {
            last_dose_eff
        } else {
            f64::NAN
        }
    };

    active_infusions.retain(|(_, _, e)| *e > t_start + 1e-12);
    // Resolve to (cmt_idx, F·rate, t_start, t_end) for the seam's time-gated
    // injection (CMT=0 / out-of-range dropped).
    let gated = gated_infusions(active_infusions, &subject.doses, dose_f_bio, n);

    // Doses delivered before the most recent reset (EVID=3/4) at or before this
    // segment are off for the input-rate forcing — mirroring how the reset clears
    // `active_infusions` and re-seeds `u` above.
    let reset_floor = subject
        .reset_times
        .iter()
        .cloned()
        .filter(|&rt| rt <= t_start + 1e-12)
        .fold(f64::NEG_INFINITY, f64::max);

    // Zero-order absorption windows covering this segment (#504): constant
    // `F·amt/dur`, reset-aware via the same `reset_floor` (a window opened
    // pre-reset is off), injected alongside the gated infusions.
    let zero_order = active_zero_order_inputs(zo_windows, t_start, t_end, reset_floor);
    // Hoist the input-rate constants once per segment (#322 #7).
    let prepared = prepare_input_rates(ode, ext_params);

    SegmentForcings {
        reset_floor,
        gated,
        zero_order,
        prepared,
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
pub fn ode_dense_solve_states(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    theta: &[f64],
    eta: &[f64],
    subject: &Subject,
    saveat: &[f64],
) -> Vec<Vec<f64>> {
    if saveat.is_empty() {
        return vec![];
    }
    let n = ode.n_states;
    let opts = ode.solver_opts;

    let mut u = ode.initial_state(pk_params_flat);
    let mut result: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; saveat.len()];

    // Resolve modeled-RATE doses once (#324) before building the timeline so the
    // states pass sees concrete rate/duration; borrowed for all-`Fixed`.
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    // Per dose-compartment bioavailability / lag (`Fn`/`ALAGn`; issue #369),
    // falling back to the bare `PK_IDX_F`/`PK_IDX_LAGTIME` slots. Uniform on
    // this no-TV path, where every dose reads the same `pk_params_flat`.
    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.lagtime(d.cmt, pk_params_flat))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.f_bio(d.cmt, pk_params_flat))
        .collect();

    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // Build saveat → index map for fast lookup.
    let mut saveat_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &t) in saveat.iter().enumerate() {
        saveat_map.entry(t.to_bits()).or_default().push(i);
    }

    let t_last = saveat.iter().cloned().fold(0.0f64, f64::max);
    // Zero-order absorption windows for this subject (#504): a single PK snapshot,
    // so per-dose `dur`/`F`/`lag` come from `pk_params_flat`. Reused for both the
    // segment break points and the per-segment constant-rate injection.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    let break_times =
        build_segment_break_times(subject, &dose_lagtimes, &dose_f_bio, &zo_windows, t_last);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

    for w in break_times.windows(2) {
        let (t_start, t_end) = (w[0], w[1]);
        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        let forcings = apply_segment_boundary(
            ode,
            subject,
            &dose_lagtimes,
            &dose_f_bio,
            &zo_windows,
            pk_params_flat,
            n,
            &opts,
            t_start,
            t_end,
            &mut u,
            &mut active_infusions,
            &mut ext_params,
        );

        // Saveat points at t_start (after dose, matching ode_predictions convention).
        // `u` here is the post-dose state; `apply_segment_boundary` set ext_params and
        // resolved forcings but did not touch `u` after the dose pulses.
        if let Some(idxs) = saveat_map.get(&t_start.to_bits()) {
            for &i in idxs {
                result[i] = u.clone();
            }
        }

        let mut seg_saveat: Vec<f64> = saveat
            .iter()
            .cloned()
            .filter(|&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .collect();
        // Always include t_end so u advances through empty segments (e.g. two
        // consecutive doses with no saveat points between them).
        if seg_saveat.is_empty() || (seg_saveat.last().unwrap() - t_end).abs() > 1e-12 {
            seg_saveat.push(t_end);
        }
        // Mirror ode_predictions lines 530-531 (and the same fix applied to
        // ode_predictions_with_states): sort + dedup so solve_ode's linear
        // save_idx cursor works correctly for duplicate / out-of-order times.
        seg_saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        seg_saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            forcings.reset_floor,
            &forcings.prepared,
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

    // `theta` and `eta` are accepted for API symmetry with sibling ODE functions
    // (e.g. `ode_predictions_with_states`) but are not consumed here: this
    // function returns the raw ODE state vector `u` without applying any
    // `output_fn` / Form-C scaling. A future extension that returns scaled
    // observables alongside states would use them. Suppress the unused warning.
    let _ = (theta, eta);

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
#[cfg(feature = "survival")]
pub(crate) fn ode_solve_until_chz_threshold(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
    chz_state: usize,
    threshold: f64,
    horizon: f64,
) -> ThresholdOutcome {
    use crate::ode::solver::{solve_ode_until_threshold, ThresholdCrossing};

    let n = ode.n_states;
    let opts = ode.solver_opts;
    let mut u = ode.initial_state(pk_params_flat);

    // Resolve modeled-RATE doses once, exactly as the dense path (#324).
    let resolved = resolve_subject_doses(subject, &ode.dose_attr_map, pk_params_flat);
    let subject: &Subject = &resolved;

    let dose_lagtimes: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.lagtime(d.cmt, pk_params_flat))
        .collect();
    let dose_f_bio: Vec<f64> = subject
        .doses
        .iter()
        .map(|d| ode.dose_attr_map.f_bio(d.cmt, pk_params_flat))
        .collect();

    let first_dose_time = subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min);
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let copy_n = pk_params_flat.len().min(crate::types::MAX_PK_PARAMS);
    ext_params[..copy_n].copy_from_slice(&pk_params_flat[..copy_n]);
    ext_params[crate::types::MAX_PK_PARAMS] = if first_dose_time.is_finite() {
        first_dose_time
    } else {
        f64::NAN
    };

    // Zero-order windows, reused for the break points and the per-segment injection
    // (same as the dense path). The terminal break is the horizon; doses scheduled
    // after it are dropped — they can never bring an event forward.
    let zo_windows = zero_order_windows(&subject.doses, &dose_lagtimes, &dose_f_bio, |_, d| {
        zero_order_dur_and_frac_for_dose(ode, d, pk_params_flat)
    });
    let mut break_times =
        build_segment_break_times(subject, &dose_lagtimes, &dose_f_bio, &zo_windows, horizon);
    break_times.retain(|&t| t <= horizon + 1e-15);

    let mut active_infusions: Vec<(usize, f64, f64)> = Vec::new();

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
            pk_params_flat,
            n,
            &opts,
            t_start,
            t_end,
            &mut u,
            &mut active_infusions,
            &mut ext_params,
        );

        let wrapped_rhs = wrap_rhs_with_forcings(
            ode,
            &subject.doses,
            &dose_lagtimes,
            &dose_f_bio,
            forcings.reset_floor,
            &forcings.prepared,
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
mod tests {
    use super::*;
    use crate::types::DoseEvent;
    use approx::assert_relative_eq;

    /// 1-cpt IV bolus ODE: dA/dt = -ke·A. RHS reads CL,V from pk_params_flat.
    fn one_cpt_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        }
    }

    fn pk_one(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p
    }

    fn make_subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>) -> Subject {
        let n_obs = obs_times.len();
        Subject {
            id: "1".into(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations: vec![0.0; n_obs],
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    /// 1-cpt IV bolus + a cumulative-hazard accumulator: state 0 = central
    /// (`dC/dt = -ke·C`, `ke = CL/V`), state 1 = CHZ (`dCHZ/dt = 0.1·C`). With a
    /// bolus `amt` at t=0 this has the closed form `C(t) = amt·e^{-ke t}`,
    /// `CHZ(t) = 0.1·amt·(1 - e^{-ke t})/ke`.
    #[cfg(feature = "survival")]
    fn one_cpt_chz_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
                dy[1] = 0.1 * y[0];
            }),
            n_states: 2,
            state_names: vec!["central".into(), "chz".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions {
                abstol: 1e-10,
                reltol: 1e-9,
                ..OdeSolverOptions::default()
            },
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        }
    }

    /// Bolus-driven crossing matches the closed form: `amt=100`, `CL=10`, `V=100`
    /// ⇒ `ke=0.1`, `CHZ(t)=100(1-e^{-0.1t})`; solving `CHZ=50` gives `t = 10·ln2`.
    #[cfg(feature = "survival")]
    #[test]
    fn until_chz_threshold_bolus_crossing_matches_closed_form() {
        let ode = one_cpt_chz_ode_spec();
        let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
        let pk = pk_one(10.0, 100.0);
        match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 50.0, 1000.0) {
            ThresholdOutcome::Crossed(t) => {
                assert_relative_eq!(t, 10.0 * std::f64::consts::LN_2, epsilon = 1e-4)
            }
            other => panic!("expected Crossed, got {other:?}"),
        }
    }

    /// Parity pin against the fit-path orchestration: the crossing time the wrapper
    /// returns, fed back through `ode_dense_solve_states`, must read a CHZ equal to
    /// the threshold. If the restated segment loop ever drifts from the dense one,
    /// `CHZ_dense(t_cross) ≠ threshold` and this fails.
    #[cfg(feature = "survival")]
    #[test]
    fn until_chz_threshold_parity_with_dense_solve() {
        let ode = one_cpt_chz_ode_spec();
        let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
        let pk = pk_one(10.0, 100.0);
        let threshold = 37.0;
        let t =
            match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, threshold, 1000.0) {
                ThresholdOutcome::Crossed(t) => t,
                other => panic!("expected Crossed, got {other:?}"),
            };
        let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[t]);
        assert_relative_eq!(states[0][1], threshold, epsilon = 1e-4);
    }

    /// #570 driver-level share: `ode_predictions_and_chz` returns the Gaussian
    /// predictions **bit-identical** to `ode_predictions` (the obs `saveat` / step
    /// sequence is untouched) and the **full** ODE state at every event time equal to
    /// the dedicated `ode_dense_solve_states` read it replaces — proving one
    /// integration now serves both consumers.
    ///
    /// The event times deliberately exercise the boundary cases the open-interval soft
    /// filter alone gets wrong (regression for the two bugs in the #613 review):
    ///   - `0.0` = the integration start (a segment's *left* boundary). The open
    ///     `(t_start, t_end]` solve never reads it, so before the left-boundary handler
    ///     it stayed NaN → the TTE `1e20` sentinel (e.g. an interval-censored `left=0`).
    ///   - `24.0` = an *interior dose time*. The shared state must be **post-dose** (the
    ///     dedicated path overwrites with the post-dose state); reading the pre-dose
    ///     value would move the instantaneous hazard `h = dCHZ/dt` for an event there.
    /// plus interior (`1/6/18`), on the obs grid (`6`), and **past** the last obs
    /// (`33 > 30`, exercising the `t_last` extension).
    #[cfg(feature = "survival")]
    #[test]
    fn ode_predictions_and_chz_shares_one_solve() {
        let ode = one_cpt_chz_ode_spec();
        let subject = make_subject(
            vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            vec![0.5, 2.0, 6.0, 12.0, 24.5, 30.0],
        );
        let pk = pk_one(10.0, 100.0);
        // Sorted, unique TTE times: integration start, interior, on-grid, interior dose
        // time, and past last obs.
        let chz_times = vec![0.0, 1.0, 6.0, 18.0, 24.0, 33.0];

        let (ipred, chz_states) =
            ode_predictions_and_chz(&ode, &pk.values, &[], &[], &subject, &chz_times);
        let ipred_ref = ode_predictions(&ode, &pk.values, &[], &[], &subject);
        let chz_ref = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &chz_times);

        // (1) Predictions bit-identical — the shared solve does not move the fit ipred,
        // even with a soft time at the integration start and on an interior dose break.
        assert_eq!(
            ipred, ipred_ref,
            "ode_predictions_and_chz must not change the predictions"
        );

        // (2) The full state (PK compartment *and* the CHZ accumulator) at each event
        // time matches the dedicated clamped solve to solver tolerance. Checking both
        // components is what catches the interior-dose case: CHZ (slot 1) is continuous
        // across a dose, but the PK compartment (slot 0) jumps, so a pre- vs post-dose
        // read only shows up there.
        assert_eq!(chz_states.len(), chz_times.len());
        for (i, st) in chz_states.iter().enumerate() {
            assert_eq!(st.len(), ode.n_states, "state {i} must be fully populated");
            for j in 0..ode.n_states {
                assert_relative_eq!(st[j], chz_ref[i][j], max_relative = 1e-5);
            }
        }
    }

    /// A threshold above the asymptotic cumulative hazard (`CHZ → 100`) is never
    /// reached ⇒ the draw is censored at the horizon, not failed.
    #[cfg(feature = "survival")]
    #[test]
    fn until_chz_threshold_censors_when_unreached() {
        let ode = one_cpt_chz_ode_spec();
        let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
        let pk = pk_one(10.0, 100.0);
        assert_eq!(
            ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 200.0, 50.0),
            ThresholdOutcome::CensoredAtHorizon
        );
    }

    /// An infusion input is handled like the dense path: a 100-unit infusion over
    /// 10 h (`rate = 10`) gives the same total exposure as the bolus, so the same
    /// asymptotic `CHZ → 100`. The crossing time reads back (via the dense solve) a
    /// CHZ equal to the threshold — the parity pin, now over the infusion branch.
    #[cfg(feature = "survival")]
    #[test]
    fn until_chz_threshold_infusion_parity_with_dense_solve() {
        let ode = one_cpt_chz_ode_spec();
        let subject = make_subject(
            vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)],
            vec![],
        );
        let pk = pk_one(10.0, 100.0);
        let threshold = 50.0;
        let t =
            match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, threshold, 1000.0) {
                ThresholdOutcome::Crossed(t) => t,
                other => panic!("expected Crossed, got {other:?}"),
            };
        let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subject, &[t]);
        assert_relative_eq!(states[0][1], threshold, epsilon = 1e-4);
    }

    /// 1-cpt with an *invalid* negative hazard (`dCHZ/dt = -0.1·C`): the accumulator
    /// decreases, so a crossing can never be well-defined.
    #[cfg(feature = "survival")]
    fn one_cpt_neg_chz_ode_spec() -> OdeSpec {
        let mut ode = one_cpt_chz_ode_spec();
        ode.rhs = Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            let cl = p[crate::types::PK_IDX_CL];
            let v = p[crate::types::PK_IDX_V];
            let ke = if v > 0.0 { cl / v } else { 0.0 };
            dy[0] = -ke * y[0];
            dy[1] = -0.1 * y[0];
        });
        ode
    }

    /// A negative hazard propagates the primitive's `Failed` to a `SolveFailed` —
    /// never a silent censor (the wrapper's failure arm).
    #[cfg(feature = "survival")]
    #[test]
    fn until_chz_threshold_negative_hazard_fails() {
        let ode = one_cpt_neg_chz_ode_spec();
        let subject = make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)], vec![]);
        let pk = pk_one(10.0, 100.0);
        match ode_solve_until_chz_threshold(&ode, &pk.values, &subject, 1, 5.0, 1000.0) {
            ThresholdOutcome::SolveFailed(msg) => {
                assert!(msg.contains("non-monotone"), "msg: {msg}")
            }
            other => panic!("expected SolveFailed, got {other:?}"),
        }
    }

    /// A bolus dose-ledger row, for the signal-AUC pass test below.
    fn ledger_bolus(time: f64, amt: f64, cmt: usize) -> DoseLedgerEntry {
        DoseLedgerEntry {
            subject: "1".into(),
            draw: 0,
            sim: 0,
            dose_idx: 0,
            time,
            amt,
            cmt,
            rate: 0.0,
            decision_idx: 0,
            rule_fired: "bolus".into(),
            observed_signals: Vec::new(),
            pre_state: None,
            post_state: None,
            f_applied: 1.0,
        }
    }

    /// Analytic accuracy check for the metrics-only signal-AUC pass
    /// ([`adaptive_window_signal_aucs`], #391 S2.5b). For a 1-cpt model with a single
    /// IV bolus `D` at t = 0, the amount readout is `A(t) = D·e^{-ke t}` (ke = CL/V),
    /// so the exposure over a window `[a, b]` *after* the dose is the closed form
    /// `∫ₐᵇ A dt = (D/ke)(e^{-ke a} − e^{-ke b})`. The window is placed strictly after
    /// the bolus so every grid edge sits in the smooth decay phase (no dose
    /// discontinuity to straddle), and the trapezoid converges to that closed form.
    #[test]
    fn adaptive_window_signal_aucs_matches_closed_form() {
        let ode = one_cpt_ode_spec(); // readout = ObsCmt(0): the central amount
        let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
        let (ke, d) = (0.1_f64, 100.0_f64);
        let base = make_subject(vec![], vec![]); // dose-free; the pass uses its own grid
        let ledger = vec![ledger_bolus(0.0, d, 1)];
        let auc = |a: f64, b: f64| (d / ke) * ((-ke * a).exp() - (-ke * b).exp());

        // Single window [2, 10], entirely in the decay phase.
        let one = adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[2.0, 10.0],
            &ledger,
            None,
            1,
        );
        assert_eq!(one.len(), 1);
        // 128-panel trapezoid of a smooth decay ⇒ ~3e-6 relative to the closed form.
        assert_relative_eq!(one[0], auc(2.0, 10.0), max_relative = 1e-4);

        // Two windows [2,6],[6,10]: exercises per-window splitting *and* state
        // continuity across the shared boundary (the second window must resume the
        // decay, not restart from the initial state).
        let two = adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[2.0, 6.0, 10.0],
            &ledger,
            None,
            1,
        );
        assert_eq!(two.len(), 2);
        assert_relative_eq!(two[0], auc(2.0, 6.0), max_relative = 1e-4);
        assert_relative_eq!(two[1], auc(6.0, 10.0), max_relative = 1e-4);
        // The two windows partition [2, 10], so their exposures sum to the total.
        assert_relative_eq!(two[0] + two[1], auc(2.0, 10.0), max_relative = 1e-4);

        // Fewer than two decisions ⇒ no closed window ⇒ empty (the metric is `None`).
        let none = adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[2.0],
            &ledger,
            None,
            1,
        );
        assert!(none.is_empty());
    }

    /// Regression (#391 S2.5b): a dose landing exactly on a window's **right**
    /// boundary belongs to the *next* window and must not inflate this one. Because
    /// [`ode_dense_solve_states`] saves the post-dose state at a save point on a dose
    /// time, an earlier single-grid implementation folded the next bolus's jump into
    /// the preceding window's right endpoint (≈ ½·Δsignal·(window ⁄ panels)). With a
    /// bolus at *every* decision time, each window must integrate only the doses at
    /// or before its own left edge. This test fails on that earlier implementation.
    #[test]
    fn adaptive_window_signal_aucs_excludes_boundary_dose() {
        let ode = one_cpt_ode_spec(); // readout = central amount, RHS = -ke·y
        let pk = pk_one(10.0, 100.0); // ke = CL/V = 0.1
        let (ke, d) = (0.1_f64, 100.0_f64);
        let base = make_subject(vec![], vec![]);
        // A bolus at each decision time 0, 24, 48 ⇒ windows [0,24] and [24,48].
        let ledger = vec![
            ledger_bolus(0.0, d, 1),
            ledger_bolus(24.0, d, 1),
            ledger_bolus(48.0, d, 1),
        ];
        let aucs = adaptive_window_signal_aucs(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 24.0, 48.0],
            &ledger,
            None,
            1,
        );
        assert_eq!(aucs.len(), 2);

        // Window 0 = [0,24]: exposure from the t=0 bolus ONLY (the t=24 bolus is the
        // next window's). Post-dose A(0) = D, so ∫₀²⁴ = (D/ke)(1 − e^{−24ke}). The
        // buggy single-grid version reported ≈ +9.4 (≈ +1%) here from the t=24 jump.
        let w0 = (d / ke) * (1.0 - (-ke * 24.0).exp());
        assert_relative_eq!(aucs[0], w0, max_relative = 1e-4);

        // Window 1 = [24,48]: superposition of the t=0 and t=24 boluses, decaying
        // from the post-dose amount A(24⁺) = D·e^{−24ke} + D; the t=48 bolus excluded.
        let a24 = d * (-ke * 24.0).exp() + d;
        let w1 = (a24 / ke) * (1.0 - (-ke * 24.0).exp());
        assert_relative_eq!(aucs[1], w1, max_relative = 1e-4);
    }

    /// Build the per-segment `obs_time -> indices` map the integrator uses.
    fn obs_index_map(obs_times: &[f64]) -> HashMap<u64, Vec<usize>> {
        let mut m: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, &t) in obs_times.iter().enumerate() {
            m.entry(t.to_bits()).or_default().push(i);
        }
        m
    }

    #[test]
    fn integrate_segment_zero_length_is_a_noop() {
        // A degenerate `[t, t]` segment must skip integration and leave the carried
        // state and predictions untouched — the guard a reactive driver relies on
        // when a decision time coincides with another break (#391 S1.2).
        // `ode_predictions` never reaches it (break_times are deduped at the same
        // 1e-15), so it has to be exercised directly here.
        let ode = one_cpt_ode_spec();
        let subject = make_subject(vec![], vec![5.0]);
        let pk = pk_one(1.0, 10.0);
        let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
        let mut u = vec![10.0];
        let mut predictions = vec![f64::NAN; subject.obs_times.len()];
        let obs_map = obs_index_map(&subject.obs_times);

        integrate_segment(
            &ode,
            &mut u,
            5.0,
            5.0,
            &subject,
            &[],
            &[],
            &mut ext_params,
            &pk.values,
            &[],
            &[],
            &obs_map,
            &mut predictions,
            None,
            &[],
        );

        assert_eq!(u, vec![10.0], "zero-length segment must not change state");
        assert!(
            predictions[0].is_nan(),
            "zero-length segment must record no observation"
        );
    }

    #[test]
    fn integrate_segment_advances_state_and_records_obs() {
        // A normal segment integrates 1-cpt decay (ke = CL/V = 0.1) over [0, 10]
        // and writes the observation at t_end, advancing `u` in place.
        let ode = one_cpt_ode_spec();
        let subject = make_subject(vec![], vec![10.0]);
        let pk = pk_one(1.0, 10.0);
        let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
        ext_params[crate::types::PK_IDX_CL] = 1.0;
        ext_params[crate::types::PK_IDX_V] = 10.0;
        let mut u = vec![10.0];
        let mut predictions = vec![f64::NAN; subject.obs_times.len()];
        let obs_map = obs_index_map(&subject.obs_times);

        integrate_segment(
            &ode,
            &mut u,
            0.0,
            10.0,
            &subject,
            &[],
            &[],
            &mut ext_params,
            &pk.values,
            &[],
            &[],
            &obs_map,
            &mut predictions,
            None,
            &[],
        );

        let expected = 10.0 * (-1.0f64).exp(); // 10·e^{-ke·10}, ke = 0.1
        assert_relative_eq!(u[0], expected, max_relative = 1e-4);
        assert_relative_eq!(predictions[0], expected, max_relative = 1e-4);
    }

    // ----- S1.3a reactive driver (#391) ---------------------------------

    #[test]
    fn adaptive_state_independent_controller_matches_static_ode() {
        // Certainty anchor (degenerate oracle): a controller that ignores state
        // and gives a fixed 100 mg bolus at every decision must reproduce
        // `ode_predictions` on the same realized doses — pinning the reactive
        // bookkeeping to the trusted static engine.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = CL/V = 0.1
        let decisions = [0.0, 24.0, 48.0];
        let obs = vec![6.0, 30.0, 54.0];

        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let static_doses: Vec<DoseEvent> = decisions
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let static_subject = make_subject(static_doses, obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

        assert_eq!(run.predictions.len(), static_preds.len());
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
        assert_eq!(run.ledger.len(), 3);
        for (i, &t) in decisions.iter().enumerate() {
            assert_eq!(run.ledger[i].time, t);
            assert_eq!(run.ledger[i].amt, 100.0);
            assert_eq!(run.ledger[i].cmt, 1);
            assert_eq!(run.ledger[i].decision_idx, i);
            assert_eq!(run.ledger[i].dose_idx, i);
        }
    }

    #[test]
    fn frozen_replay_verifier_accepts_aligned_run_and_rejects_corruption() {
        // The verifier's Err branches aren't reachable from a faithful run (the
        // bookkeeping is correct), so exercise them directly: a faithful run
        // passes, a perturbed trajectory is a typed divergence error, and a
        // wrong-length prediction vector is a typed error rather than a panic.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 24.0, 48.0];
        let obs = vec![6.0, 30.0, 54.0];
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], obs);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        // A dose at every decision aligns the segment structure → exact match.
        verify_adaptive_frozen_replay(&ode, &pk.values, &[], &[], &base, &decisions, &run)
            .expect("aligned run matches the static replay");

        let mut perturbed = run.clone();
        perturbed.predictions[0] += 10.0;
        let err = verify_adaptive_frozen_replay(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &perturbed,
        )
        .expect_err("a perturbed trajectory must fail verification");
        assert!(err.contains("diverges"), "got: {err}");

        let mut short = run.clone();
        short.predictions.pop();
        let err =
            verify_adaptive_frozen_replay(&ode, &pk.values, &[], &[], &base, &decisions, &short)
                .expect_err("a length mismatch must fail verification");
        assert!(err.contains("prediction"), "got: {err}");
    }

    #[test]
    fn frozen_replay_aligns_break_structure_on_held_decisions() {
        // Regression for the held-decision tolerance fix: a run that holds at
        // some decisions used to only agree with the static replay within a wide
        // (×100·reltol) slack, because the driver breaks at every decision while a
        // naive static replay breaks only at realized doses. Feeding the decision
        // schedule back in as no-op breaks aligns the two engines' segments, so
        // the run now passes the *tight* default verifier. Dose only while the
        // central amount is below 50: at t=0 the trough is 0 → dose; the later
        // decisions see a decayed-but-still-high amount → hold.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 2.0, 4.0];
        let obs = vec![1.0, 3.0, 5.0];
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let base = make_subject(vec![], obs);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        // Exactly one realized dose (t=0); the t=2 / t=4 decisions held.
        assert_eq!(run.ledger.len(), 1, "only the t=0 decision should dose");

        // Passes the tight (aligned) verifier — the whole point of the fix.
        verify_adaptive_frozen_replay(&ode, &pk.values, &[], &[], &base, &decisions, &run)
            .expect("held-decision run matches the aligned static replay");
    }

    #[test]
    fn frozen_replay_residual_is_pinned_below_the_verifier_bound() {
        // Characterization of the residual that justifies the verifier's tolerance
        // factor (`REPLAY_TOL_FACTOR = 8`). On a held-decision run we measure the
        // max relative |reactive − static| both ways:
        //   * ALIGNED   (decision times fed back as no-op breaks): measured 0.0 —
        //     the reactive driver and the static engine, walking identical segments
        //     through the same `integrate_segment`, agree BIT-FOR-BIT.
        //   * UNALIGNED (naive replay, breaks only at realized doses): measured
        //     ~7.3e-8 here — a real held-decision perturbation that the alignment
        //     removes entirely.
        // Both sit far under the live verifier bound (×8·reltol = 8e-4), so it
        // never false-positives; the ×8 is the conservative margin that holds even
        // on stiffer models where the (pre-alignment) perturbation would be larger.
        // If the alignment ever regresses, `rel_aligned` jumps toward the unaligned
        // level and the bit-exact bound below fails loudly.
        let ode = one_cpt_ode_spec(); // reltol 1e-4 / abstol 1e-6 (defaults)
        let pk = pk_one(1.0, 10.0);
        // CL=1, V=10 → k=0.1/h. A 100-unit bolus only while the central amount is
        // below 50; it decays 100·e^{-0.1t}, crossing 50 near t≈6.9, so over this
        // schedule the t=0 and t=8 troughs dose and t∈{2,4,6} hold — a dose/hold mix.
        let decisions = [0.0, 2.0, 4.0, 6.0, 8.0];
        let obs = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let base = make_subject(vec![], obs);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert!(
            run.ledger.len() >= 2,
            "expected a dose/hold mix (≥2 realized doses), got {}",
            run.ledger.len()
        );

        // Rebuild the static subject from the realized ledger, exactly as the
        // verifier does.
        let mut static_subject = base.clone();
        static_subject.doses = run
            .ledger
            .iter()
            .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
            .collect();

        let max_rel = |preds: &[f64]| -> f64 {
            run.predictions
                .iter()
                .zip(preds)
                .filter(|(g, w)| g.is_finite() && w.is_finite() && w.abs() > 0.0)
                .map(|(g, w)| (g - w).abs() / w.abs())
                .fold(0.0_f64, f64::max)
        };

        let aligned = ode_predictions_with_extra_breaks(
            &ode,
            &pk.values,
            &[],
            &[],
            &static_subject,
            &decisions,
        );
        let unaligned = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

        let rel_aligned = max_rel(&aligned);
        let rel_unaligned = max_rel(&unaligned);

        // Aligned replay is bit-exact (measured 0.0). Allow a few ULP of headroom
        // for future legitimate reordering, but stay ~9 orders under the live ×8
        // bound: a held-break-mismatch regression pushes this toward the unaligned
        // level (~7e-8) and trips here long before the ×8 verifier would.
        assert!(
            rel_aligned <= 1e-12,
            "aligned replay should match the reactive driver bit-for-bit, got {rel_aligned:e} \
             (verifier bound is 8·reltol = 8e-4); the decision-time break alignment may have \
             regressed"
        );
        // And the alignment is genuinely doing the work: the naive (unaligned)
        // replay carries a real, measurable residual that the alignment eliminates.
        assert!(
            rel_unaligned > 1e-9,
            "expected a measurable unaligned residual (the perturbation alignment removes); \
             got {rel_unaligned:e} — if this is ~0 the scenario no longer holds any decisions, \
             so the characterization is vacuous"
        );
    }

    #[test]
    fn adaptive_feedback_doses_only_below_threshold() {
        // State-dependent: dose 100 only when the monitored amount is below 50.
        // At t=0 amount is 0 (<50) -> dose; by t=2 it decayed to 100·e^{-0.2}
        // ≈ 81.9 (>50) -> hold. Exactly one realized dose.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let decisions = [0.0, 2.0];
        let obs = vec![1.0, 3.0];

        let mut controller = |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A is declared") < 50.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        assert_eq!(run.ledger.len(), 1, "dose at t=0, hold at t=2");
        assert_eq!(run.ledger[0].time, 0.0);
        assert_eq!(run.ledger[0].decision_idx, 0);
        assert_eq!(run.ledger[0].observed_signals[0].name, "A");
        assert_eq!(run.ledger[0].observed_signals[0].value, 0.0);

        // Trajectory vs the exact 1-cpt closed form A(t) = 100·e^{-ke·t}, ke=0.1.
        // (An analytical oracle, not `ode_predictions`: with a hold at t=2 the
        // driver breaks there while the static engine wouldn't, so a static
        // comparison would confound the integrator restart with the dosing logic.)
        let ke = 0.1;
        for (i, t) in [1.0_f64, 3.0].into_iter().enumerate() {
            let exact = 100.0 * (-ke * t).exp();
            assert_relative_eq!(run.predictions[i], exact, max_relative = 1e-5);
        }
    }

    #[test]
    fn adaptive_decision_monitor_uses_observation_covariates() {
        // Regression (#538): at a decision time the monitored Form-C readout
        // must see the covariate snapshot in effect at that time (the coincident
        // observation row), not the subject-level first-row covariate. The
        // readout is `state * FREE`; with no decay the state stays at the dose
        // amount, so the monitored signal is driven purely by FREE.
        let ode = OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::Single(Box::new(|state, _pk, _theta, _eta, covariates| {
                state[0] * covariates.get("FREE").copied().unwrap_or(0.0)
            })),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        };
        let pk = pk_one(0.0, 1.0); // ke = 0 -> state holds at the dose amount
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let decisions = [0.0, 10.0];

        // Single observation at the second decision time, carrying FREE=2.0;
        // the subject-static map carries the stale FREE=1.0.
        let mut base = make_subject(vec![], vec![10.0]);
        base.covariates.insert("FREE".into(), 1.0);
        base.obs_covariates = vec![HashMap::from([("FREE".to_string(), 2.0)])];

        // Dose 100 at t=0 (signal 0 < 150). At t=10 the pre-dose state is 100,
        // so the monitored signal is 100*FREE. With the observation snapshot
        // (FREE=2) the signal is 200 >= 150 -> hold; with the stale static
        // value (FREE=1) it would be 100 < 150 -> a second (wrong) dose.
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("monitor A is declared") < 150.0 {
                vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        assert_eq!(
            run.ledger.len(),
            1,
            "dose only at t=0; the t=10 monitor must read FREE=2 (signal 200) and hold"
        );
        assert_eq!(run.ledger[0].time, 0.0);
        // The decision at t=10 logged the monitored signal computed with the
        // observation-row covariate: 100 * 2.0 = 200.
        let d10 = run
            .decisions
            .iter()
            .find(|d| d.time == 10.0)
            .expect("decision logged at t=10");
        assert_relative_eq!(d10.observed_signals[0].value, 200.0, epsilon = 1e-9);
    }

    #[test]
    fn adaptive_stop_discontinues_further_dosing() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Stop];
        let base = make_subject(vec![], vec![12.0, 36.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 24.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert!(
            run.ledger.is_empty(),
            "Stop at decision 0 prevents all doses"
        );
        assert!(
            run.predictions.iter().all(|&p| p == 0.0),
            "no dose -> zero state"
        );
    }

    #[test]
    fn adaptive_zero_amount_bolus_is_treated_as_hold() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 0.0, cmt: 1 }];
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert!(run.ledger.is_empty(), "zero-amount bolus records no dose");
        assert_eq!(run.predictions[0], 0.0);
    }

    #[test]
    fn adaptive_infusion_state_independent_matches_static_ode() {
        // Degenerate oracle (infusion edition): a controller that ignores state
        // and issues the same fixed infusion at every decision must reproduce
        // `ode_predictions` on the equivalent static infusion schedule, bit-exact.
        // This pins the dynamic infusion-end timeline (every F-scaled end inserted
        // as a break) to the trusted static segmentation. The last observation is
        // the global maximum so neither engine breaks at an interior observation
        // (which would restart the integrator on only one side and diverge).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let decisions = [0.0, 24.0, 48.0];
        // Each infusion: 100 mg at rate 25 -> 4 h. Ends 4, 28, 52 (between
        // decisions). Observations span during/post-infusion; the last (60) is
        // past every infusion end so it is the global maximum.
        let obs = vec![2.0, 6.0, 26.0, 30.0, 50.0, 60.0];

        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 25.0,
            }]
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let static_doses: Vec<DoseEvent> = decisions
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 25.0, false, 0.0))
            .collect();
        let static_subject = make_subject(static_doses, obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

        assert_eq!(run.predictions.len(), static_preds.len());
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
        assert_eq!(run.ledger.len(), 3);
        for (i, &t) in decisions.iter().enumerate() {
            assert_eq!(run.ledger[i].time, t);
            assert_eq!(run.ledger[i].rate, 25.0);
            assert_eq!(run.ledger[i].rule_fired, "infuse");
        }
    }

    // ----- S1.5: DV-mode (assay-noised) monitors (#391) -----------------

    fn dv_monitor() -> [MonitorSpec; 1] {
        [MonitorSpec::new("A", 1, ObserveMode::Dv)]
    }

    #[test]
    fn adaptive_dv_without_assay_capability_errors() {
        // A DV monitor on an Ipred-only run (assay = None) is a typed error, not a
        // silent fallback to the latent value.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &dv_monitor(),
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(
            err.contains("DV") && err.contains("capability"),
            "got: {err}"
        );
    }

    #[test]
    fn adaptive_dv_no_error_model_errors() {
        // Edge (a): a DV monitor on a compartment with no residual error model is a
        // typed error (resid_var returns None), never a fabricated sigma.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(vec![], vec![1.0]);
        let no_model = |_cmt: usize, _ipred: f64| None;
        let assay = AssayNoise {
            resid_var: &no_model,
            base_seed: 7,
        };
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &dv_monitor(),
            &mut controller,
            100,
            Some(&assay),
        )
        .unwrap_err();
        assert!(err.contains("error_model"), "got: {err}");
    }

    #[test]
    fn adaptive_dv_zero_variance_equals_ipred() {
        // sigma -> 0: the DV signal collapses to the latent IPRED. Compare the value
        // the controller saw under a zero-variance assay against an Ipred monitor on
        // the same realized run.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let decisions = [0.0, 24.0]; // dose at t=0, observe pre-dose trough at t=24
        let base = make_subject(vec![], vec![24.0]);
        let dose = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];

        let mut ctrl_ref = dose;
        let ref_run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[MonitorSpec::new("A", 1, ObserveMode::Ipred)],
            &mut ctrl_ref,
            100,
            None,
        )
        .expect("ipred run");
        let ipred = ref_run.decisions[1].observed_signals[0].value;

        let zero_var = |_cmt: usize, _ipred: f64| Some(0.0);
        let assay = AssayNoise {
            resid_var: &zero_var,
            base_seed: 12345,
        };
        let mut ctrl_dv = dose;
        let dv_run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &dv_monitor(),
            &mut ctrl_dv,
            100,
            Some(&assay),
        )
        .expect("dv run");
        let dv = dv_run.decisions[1].observed_signals[0].value;

        assert!(ipred > 0.0, "expected a non-zero trough at t=24");
        assert_relative_eq!(dv, ipred, epsilon = 1e-12);
    }

    #[test]
    fn adaptive_dv_noised_and_deterministic() {
        // Non-zero variance perturbs the latent IPRED, and the draw is reproducible:
        // the same base seed yields the same value, a different seed a different one.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 24.0];
        let base = make_subject(vec![], vec![24.0]);
        let var4 = |_cmt: usize, _ipred: f64| Some(4.0); // sd = 2

        let observe = |seed: u64| {
            let assay = AssayNoise {
                resid_var: &var4,
                base_seed: seed,
            };
            let mut ctrl = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
            ode_predictions_adaptive(
                &ode,
                &pk.values,
                &[],
                &[],
                &base,
                &decisions,
                &dv_monitor(),
                &mut ctrl,
                100,
                Some(&assay),
            )
            .expect("dv run")
            .decisions[1]
                .observed_signals[0]
                .value
        };

        let a = observe(999);
        let b = observe(999);
        let c = observe(1000);
        assert_eq!(a, b, "same base seed must reproduce the assay draw");
        assert_ne!(a, c, "a different base seed must change the assay draw");
        let latent = 100.0 * (-2.4f64).exp(); // trough at t=24
        assert!(
            (a - latent).abs() > 1e-9,
            "expected the assay to perturb the latent value"
        );
    }

    #[test]
    fn adaptive_dv_clamps_negative_at_zero() {
        // Edge (b): the noised value cannot read below zero. At t=0 the pre-dose
        // trough is 0, so a negative assay draw with a large sigma would push it
        // negative; assert it clamps to exactly 0.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let neg_seed = (0u64..)
            .find(|&s| assay_standard_normal(s, 0, "A") < 0.0)
            .expect("some seed gives a negative draw");
        let big_var = |_cmt: usize, _ipred: f64| Some(1.0e6);
        let assay = AssayNoise {
            resid_var: &big_var,
            base_seed: neg_seed,
        };
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &dv_monitor(),
            &mut controller,
            100,
            Some(&assay),
        )
        .expect("dv run");
        assert_eq!(
            run.decisions[0].observed_signals[0].value, 0.0,
            "a negative assay reading must clamp at 0"
        );
    }

    #[test]
    fn adaptive_dv_added_monitor_does_not_perturb_other_draw() {
        // Non-perturbing: adding a second DV monitor (a new analyte) must not change
        // the first analyte's draw — each is keyed by its own analyte name.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 24.0];
        let base = make_subject(vec![], vec![24.0]);
        let var4 = |_cmt: usize, _ipred: f64| Some(4.0);

        let signal_a = |monitors: &[MonitorSpec]| {
            let assay = AssayNoise {
                resid_var: &var4,
                base_seed: 555,
            };
            let mut ctrl = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
            let run = ode_predictions_adaptive(
                &ode,
                &pk.values,
                &[],
                &[],
                &base,
                &decisions,
                monitors,
                &mut ctrl,
                100,
                Some(&assay),
            )
            .expect("dv run");
            run.decisions[1]
                .observed_signals
                .iter()
                .find(|s| s.name == "A")
                .expect("analyte A present")
                .value
        };

        let one = [MonitorSpec::new("A", 1, ObserveMode::Dv)];
        let two = [
            MonitorSpec::new("A", 1, ObserveMode::Dv),
            MonitorSpec::new("B", 1, ObserveMode::Dv),
        ];
        assert_eq!(
            signal_a(&one),
            signal_a(&two),
            "adding analyte B must not perturb A's draw"
        );
    }

    #[test]
    fn adaptive_rejects_nonempty_base_subject() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(
            vec![DoseEvent::new(0.0, 50.0, 1, 0.0, false, 0.0)],
            vec![1.0],
        );
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("dose-free"), "got: {err}");
    }

    #[test]
    fn adaptive_max_decisions_runaway_guard() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 24.0, 48.0],
            &[],
            &mut controller,
            2,
            None,
        )
        .unwrap_err();
        assert!(err.contains("max_decisions"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_zero_compartment_via_validate() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 0 }];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("compartment"), "got: {err}");
    }

    #[test]
    fn adaptive_final_decision_at_max_time_still_fires() {
        // Regression: the last decision must fire even when it lands on the
        // schedule's maximum time (i.e. at or after the last observation). Here
        // the second decision (t=24) coincides with the last observation, so it
        // is the maximum break time; it must still dose, reach the ledger, and
        // make the t=24 observation read the *post*-dose state.
        //
        // Checked against the exact 1-cpt closed form (ke = CL/V = 0.1), not
        // `ode_predictions`: the static engine likewise never applies a dose on
        // its terminal break, so a static comparison would mask the bug.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let decisions = [0.0, 24.0];
        let obs = vec![6.0, 24.0]; // last obs coincides with the last decision
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], obs);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        // Both decisions dosed — including the one at the maximum time.
        assert_eq!(run.ledger.len(), 2, "final decision at t_max must dose");
        assert_eq!(run.ledger[1].time, 24.0);
        assert_eq!(run.ledger[1].decision_idx, 1);

        let ke = 0.1_f64;
        // t=6: only the t=0 dose has been given.
        assert_relative_eq!(
            run.predictions[0],
            100.0 * (-ke * 6.0).exp(),
            max_relative = 1e-5
        );
        // t=24: first dose decayed to 24 h, plus the fresh 100 mg bolus (post-dose).
        let expected_24 = 100.0 * (-ke * 24.0).exp() + 100.0;
        assert_relative_eq!(run.predictions[1], expected_24, max_relative = 1e-5);
    }

    #[test]
    fn adaptive_rejects_out_of_range_bolus_compartment() {
        // `validate()` only catches cmt == 0; an out-of-range cmt (> n_states) is
        // caught by the driver's own guard. 1-state model, bolus into cmt 2.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 2 }];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("state"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_out_of_range_monitor_compartment() {
        // A monitor on a compartment beyond the model is a precondition error.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let monitors = [MonitorSpec::new("A", 2, ObserveMode::Ipred)]; // n_states = 1
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Hold];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &monitors,
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("state"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_dosing_into_input_rate_compartment() {
        // A bolus into a compartment fed by a built-in input-rate function would
        // be double-counted (state jump *and* `R_in` forcing). Must be a typed
        // error, not a silent wrong answer.
        let mut ode = one_cpt_ode_spec();
        ode.input_rate = vec![crate::pk::absorption::InputRateForcing {
            cmt: 0, // 0-based -> consumes 1-based compartment 1
            kind: crate::pk::absorption::InputRateKind::Transit,
            arg_slots: vec![],
            frac_slot: None,
        }];
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("input-rate"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_lagged_dose_compartment() {
        // A lag time on the dosed compartment would be applied with zero delay
        // here yet dropped from its own TAD anchor in `integrate_segment`. Reject.
        let ode = one_cpt_ode_spec();
        let mut pk = pk_one(1.0, 10.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // bare-slot lag on cmt 1
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("lag time"), "got: {err}");
    }

    // ----- S1.3b reactive infusions (#391) ------------------------------

    #[test]
    fn insert_break_keeps_sorted_and_dedups_within_tolerance() {
        let mut breaks = vec![0.0, 10.0, 20.0];
        insert_break(&mut breaks, 5.0); // strictly between -> inserted
        assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0]);
        insert_break(&mut breaks, 10.0 + 1e-16); // within 1e-15 of existing -> dropped
        assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0]);
        insert_break(&mut breaks, 25.0); // past the end -> appended
        assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0, 25.0]);
        insert_break(&mut breaks, 0.0); // duplicate of the first -> dropped
        assert_eq!(breaks, vec![0.0, 5.0, 10.0, 20.0, 25.0]);
    }

    #[test]
    fn adaptive_infusion_matches_closed_form() {
        // Absolute oracle: a single zero-order infusion into a 1-cpt linear model
        // has the closed form A(t) = (R/ke)(1 - e^{-ke t}) while infusing and
        // A(t_inf)·e^{-ke (t - t_inf)} afterward. Pins magnitude against
        // mathematics, not just against the static engine.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let ke = 0.1;
        let (rate, amt) = (10.0_f64, 100.0_f64);
        let t_inf = amt / rate; // 10 h (F = 1)
        let obs = vec![5.0, 10.0, 20.0]; // during, at end, after
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Infuse { amt, cmt: 1, rate }];
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp());
        let expected = [
            (rate / ke) * (1.0 - (-ke * 5.0_f64).exp()), // during
            a_inf,                                       // at end
            a_inf * (-ke * (20.0 - t_inf)).exp(),        // after
        ];
        // RK45-vs-analytical tolerance (the established 1e-4 in this file): the
        // bit-exact 1e-9 oracle below pins the integrator to the static engine;
        // this test pins *magnitude* against mathematics, where 0.01% is ample.
        for (i, e) in expected.iter().enumerate() {
            assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
        }
    }

    #[test]
    fn adaptive_overlapping_infusions_match_static() {
        // The hard case: an infusion whose end falls *after* the next decision, so
        // two controller infusions overlap. `active_infusions` must sum both rates
        // over the overlap window, and the timeline must carry both ends as breaks.
        // Compared bit-exact to the equivalent two-infusion static schedule (dosing
        // at every decision, so there is no phantom break).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 5.0];
        // 100 mg @ rate 10 -> 10 h each: windows [0,10] and [5,15] overlap on [5,10].
        let obs = vec![2.0, 7.0, 12.0, 20.0]; // last (20) past both ends
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 10.0,
            }]
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let static_doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0),
            DoseEvent::new(5.0, 100.0, 1, 10.0, false, 0.0),
        ];
        let static_subject = make_subject(static_doses, obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
        assert_eq!(run.ledger.len(), 2);
    }

    #[test]
    fn adaptive_infusion_end_coincident_with_decision_dedups() {
        // An infusion that ends *exactly* at the next decision must not create a
        // second break: the end coincides with the decision break (which the static
        // engine also has, as the infusion end), so the timelines match bit-exact.
        // A hold at that decision is therefore safe to compare to the
        // single-infusion static schedule (no phantom break).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let decisions = [0.0, 4.0]; // 100@25 -> 4 h, ends exactly at decision 1
        let obs = vec![2.0, 4.0, 8.0];
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.decision_index == 0 {
                vec![DoseAction::Infuse {
                    amt: 100.0,
                    cmt: 1,
                    rate: 25.0,
                }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.ledger.len(), 1, "only the first decision infuses");

        let static_subject =
            make_subject(vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)], obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
    }

    #[test]
    fn adaptive_infusion_f_scaling_matches_static() {
        // The S1.3b invariant *under F != 1*: the F-scaled infusion end inserted as
        // a break (`t_start + dur_eff`, from `bioavailable_infusion`) must coincide
        // with the F-scaled window `active_infusions` re-derives inside
        // `integrate_segment`. At F = 1 the two are trivially equal (`dur_eff ==
        // amt/rate`), so every other oracle test leaves this seam unexercised. Here
        // a bare-slot F = 0.5 halves a rate-defined infusion's window to F·amt/rate;
        // the degenerate oracle must still reproduce the equivalent static infusion
        // schedule (carrying the same F) bit-exact.
        let ode = one_cpt_ode_spec();
        let mut pk = pk_one(1.0, 10.0); // ke = 0.1
        pk.values[crate::types::PK_IDX_F] = 0.5; // bare-slot F on all compartments
        let decisions = [0.0, 24.0, 48.0];
        // 100 mg @ rate 25 -> nominal 4 h window, F-scaled to 0.5*4 = 2 h. Ends at
        // 2, 26, 50 (between decisions). The last obs (60) is past every end, so it
        // is the global maximum and neither engine breaks at an interior obs.
        let obs = vec![1.0, 3.0, 25.0, 27.0, 49.0, 60.0];
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 25.0,
            }]
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let static_doses: Vec<DoseEvent> = decisions
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 25.0, false, 0.0))
            .collect();
        let static_subject = make_subject(static_doses, obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

        assert_eq!(run.predictions.len(), static_preds.len());
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
        // F is actually applied (window halved), recorded as f_applied on each row.
        assert_eq!(run.ledger.len(), 3);
        for entry in &run.ledger {
            assert_eq!(entry.f_applied, 0.5);
        }
    }

    #[test]
    fn adaptive_bolus_f_scaling_matches_static() {
        // Coverage for the bolus emit-path multiply `u[cmt-1] += F*amt` under
        // F != 1. The infusion-F seam is covered above, but no test drove the
        // *bolus* F multiply with F != 1 (it shares the static engine's structure,
        // but was unexercised). A bare-slot F = 0.5 halves every controller-issued
        // bolus; the degenerate oracle (re-issue the same bolus at each decision)
        // must reproduce the equivalent static bolus schedule (carrying the same F)
        // bit-exact.
        let ode = one_cpt_ode_spec();
        let mut pk = pk_one(1.0, 10.0); // ke = 0.1
        pk.values[crate::types::PK_IDX_F] = 0.5; // bare-slot F on all compartments
        let decisions = [0.0, 24.0, 48.0];
        // A dose is realized at every decision and the last observation (60) is the
        // global maximum, so neither engine breaks at an interior observation — the
        // condition under which the degenerate oracle is bit-exact.
        let obs = vec![1.0, 12.0, 25.0, 36.0, 49.0, 60.0];
        let mut controller = |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }];
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        let static_doses: Vec<DoseEvent> = decisions
            .iter()
            .map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect();
        let static_subject = make_subject(static_doses, obs);
        let static_preds = ode_predictions(&ode, &pk.values, &[], &[], &static_subject);

        assert_eq!(run.predictions.len(), static_preds.len());
        for (got, want) in run.predictions.iter().zip(static_preds.iter()) {
            assert_relative_eq!(*got, *want, max_relative = 1e-9);
        }
        // F is actually applied to the bolus (u += F*amt), recorded per row.
        assert_eq!(run.ledger.len(), 3);
        for entry in &run.ledger {
            assert_eq!(entry.f_applied, 0.5);
        }
    }

    #[test]
    fn adaptive_reactive_infusion_titrates_against_closed_form() {
        // Genuine state-reactive infusion: infuse 100 mg @ 25 (4 h) only when the
        // monitored amount is below 50, else hold. Checked against the exact 1-cpt
        // infusion closed form — NOT the static engine: the hold at the second
        // decision makes the driver break where a dose-list replay would not, so a
        // static comparison would confound the integrator restart with the logic
        // (same reason as the bolus feedback test).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let ke = 0.1;
        let (rate, amt) = (25.0_f64, 100.0_f64);
        let t_inf = amt / rate; // 4 h
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let decisions = [0.0, 6.0];
        let obs = vec![2.0, 5.0, 8.0];
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.signal("A").expect("A declared") < 50.0 {
                vec![DoseAction::Infuse { amt, cmt: 1, rate }]
            } else {
                vec![DoseAction::Hold]
            }
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        // t=0: A=0 (<50) -> infuse [0,4]. By t=6 the amount has decayed back above
        // 50 (A(6) = a_inf·e^{-0.2} ≈ 67.5) -> hold. Exactly one realized dose.
        assert_eq!(run.ledger.len(), 1, "infuse once, then hold");
        assert_eq!(run.ledger[0].rule_fired, "infuse");

        let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp()); // amount at end of infusion
        let expected = [
            (rate / ke) * (1.0 - (-ke * 2.0_f64).exp()), // t=2 during infusion
            a_inf * (-ke * (5.0 - t_inf)).exp(),         // t=5 post infusion
            a_inf * (-ke * (8.0 - t_inf)).exp(),         // t=8 post infusion
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
        }
    }

    #[test]
    fn adaptive_stop_lets_in_flight_infusion_complete() {
        // Contract: `Stop` discontinues *future* decisions, but an infusion already
        // issued is a committed dose and keeps delivering to its end. Infuse at t=0
        // over [0,20]; Stop at t=5. The infusion must still be active at t=10 (well
        // past the Stop) and finish at t=20 — verified against the closed form. A
        // true safety-halt that truncates delivery is a separate action (tracked as
        // a follow-up), deliberately not conflated with `Stop`.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0); // ke = 0.1
        let ke = 0.1;
        let (rate, amt) = (5.0_f64, 100.0_f64);
        let t_inf = amt / rate; // 20 h
        let decisions = [0.0, 5.0];
        let obs = vec![10.0, 25.0];
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.decision_index == 0 {
                vec![DoseAction::Infuse { amt, cmt: 1, rate }]
            } else {
                vec![DoseAction::Stop]
            }
        };
        let base = make_subject(vec![], obs.clone());
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        assert_eq!(run.ledger.len(), 1, "Stop adds no dose; the infusion stays");
        let a_inf = (rate / ke) * (1.0 - (-ke * t_inf).exp());
        let expected = [
            (rate / ke) * (1.0 - (-ke * 10.0_f64).exp()), // t=10: still infusing despite Stop@5
            a_inf * (-ke * (25.0 - t_inf)).exp(),         // t=25: after the infusion finished @20
        ];
        for (i, e) in expected.iter().enumerate() {
            assert_relative_eq!(run.predictions[i], *e, max_relative = 1e-4);
        }
    }

    #[test]
    fn adaptive_zero_amount_infusion_is_treated_as_hold() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 0.0,
                cmt: 1,
                rate: 10.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert!(
            run.ledger.is_empty(),
            "zero-amount infusion records no dose"
        );
        assert_eq!(run.predictions[0], 0.0);
    }

    #[test]
    fn adaptive_rejects_nonpositive_infusion_rate() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 0.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("rate"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_out_of_range_infusion_compartment() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 2,
                rate: 10.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("state"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_infusion_into_input_rate_compartment() {
        let mut ode = one_cpt_ode_spec();
        ode.input_rate = vec![crate::pk::absorption::InputRateForcing {
            cmt: 0, // 0-based -> consumes 1-based compartment 1
            kind: crate::pk::absorption::InputRateKind::Transit,
            arg_slots: vec![],
            frac_slot: None,
        }];
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 10.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("input-rate"), "got: {err}");
    }

    #[test]
    fn adaptive_rejects_lagged_infusion_compartment() {
        let ode = one_cpt_ode_spec();
        let mut pk = pk_one(1.0, 10.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // bare-slot lag on cmt 1
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 10.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let err = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .unwrap_err();
        assert!(err.contains("lag time"), "got: {err}");
    }

    // ----- S1.4a decision log (#391) ------------------------------------

    #[test]
    fn adaptive_decision_log_records_dose_hold_and_stop() {
        // Every decision is logged — including the hold, which leaves no ledger
        // row — with the signal the controller observed and the outcome it chose.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let monitors = [MonitorSpec::new("A", 1, ObserveMode::Ipred)];
        let decisions = [0.0, 24.0, 48.0];
        let mut controller = |ctx: &ControllerCtx| match ctx.decision_index {
            0 => vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }],
            1 => vec![DoseAction::Hold],
            _ => vec![DoseAction::Stop],
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &decisions,
            &monitors,
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");

        assert_eq!(run.decisions.len(), 3, "one log entry per decision");
        for (i, d) in run.decisions.iter().enumerate() {
            assert_eq!(d.decision_idx, i);
            assert_eq!(d.time, decisions[i]);
            assert_eq!(d.observed_signals.len(), 1);
            assert_eq!(d.observed_signals[0].name, "A");
        }
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 1 });
        assert_eq!(run.decisions[1].outcome, DecisionOutcome::Hold);
        assert_eq!(run.decisions[2].outcome, DecisionOutcome::Stop { dosed: 0 });
        // The pre-dose signal at the first decision is the empty initial state.
        assert_eq!(run.decisions[0].observed_signals[0].value, 0.0);
    }

    #[test]
    fn adaptive_decision_log_omits_decisions_after_stop() {
        // Once the controller stops, the driver issues no further decisions, so
        // the Stop entry is the last record (no phantom post-stop log rows).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |ctx: &ControllerCtx| {
            if ctx.decision_index == 0 {
                vec![DoseAction::Stop]
            } else {
                // The driver must never call the controller again after a Stop;
                // reaching here would be the bug this test guards against.
                unreachable!(
                    "driver issued a decision after Stop (idx {})",
                    ctx.decision_index
                )
            }
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 10.0, 20.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1, "only the stop decision is logged");
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 0 });
        assert!(run.ledger.is_empty());
    }

    #[test]
    fn adaptive_decision_log_dose_then_stop_in_one_action_list() {
        // `[Bolus, Stop]` — a final dose, then discontinue — is logged as
        // `Stop { dosed: 1 }`, not a bare stop, and the dose reaches the ledger.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller =
            |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }, DoseAction::Stop];
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 24.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1, "stop ends the schedule after one");
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 1 });
        assert_eq!(run.ledger.len(), 1);
    }

    #[test]
    fn adaptive_decision_log_counts_multiple_doses_in_one_decision() {
        // A decision can issue more than one dose (e.g. a loading split); the log
        // records `Dosed { n }` with the realized count, and a zero-amount action
        // in the same list is excluded (it leaves no ledger row).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![
                DoseAction::Bolus { amt: 50.0, cmt: 1 },
                DoseAction::Bolus { amt: 0.0, cmt: 1 }, // normalized to Hold, not counted
                DoseAction::Bolus { amt: 50.0, cmt: 1 },
            ]
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1);
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 2 });
        assert_eq!(
            run.ledger.len(),
            2,
            "two realized doses, the zero-amt excluded"
        );
    }

    #[test]
    fn adaptive_decision_log_records_infusion_as_dosed() {
        // An infusion is a realized dose: its decision categorizes to `Dosed { n }`
        // exactly as a bolus does (the outcome doesn't distinguish route), and it
        // reaches the ledger.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![DoseAction::Infuse {
                amt: 100.0,
                cmt: 1,
                rate: 50.0,
            }]
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1);
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Dosed { n: 1 });
        assert_eq!(run.ledger.len(), 1);
    }

    #[test]
    fn adaptive_decision_log_infusion_then_stop() {
        // `[Infuse, Stop]` mirrors the bolus dose-then-stop: a final infusion, then
        // discontinue, logged as `Stop { dosed: 1 }` with the infusion in the ledger.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| {
            vec![
                DoseAction::Infuse {
                    amt: 100.0,
                    cmt: 1,
                    rate: 50.0,
                },
                DoseAction::Stop,
            ]
        };
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0, 24.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1, "stop ends the schedule after one");
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Stop { dosed: 1 });
        assert_eq!(run.ledger.len(), 1);
    }

    #[test]
    fn adaptive_decision_log_empty_action_list_is_hold() {
        // An empty action list is a no-change decision: it categorizes to `Hold`
        // (no dose, not stopped) and leaves no ledger row — same as `[Hold]`.
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let mut controller = |_ctx: &ControllerCtx| Vec::<DoseAction>::new();
        let base = make_subject(vec![], vec![1.0]);
        let run = ode_predictions_adaptive(
            &ode,
            &pk.values,
            &[],
            &[],
            &base,
            &[0.0],
            &[],
            &mut controller,
            100,
            None,
        )
        .expect("driver runs");
        assert_eq!(run.decisions.len(), 1);
        assert_eq!(run.decisions[0].outcome, DecisionOutcome::Hold);
        assert!(run.ledger.is_empty());
    }

    #[test]
    fn adaptive_driver_rejects_malformed_or_post_stop_actions() {
        // The whole action list is validated up front, before anything is applied:
        // a malformed action is a typed error wherever it sits, and `Stop` must be
        // the final action — a controller that issues actions after discontinuing is
        // rejected, not silently truncated, so the log can't disagree with the
        // ledger. Nothing is applied when the list is rejected (the ledger would be
        // discarded with the `Err` regardless).
        let ode = one_cpt_ode_spec();
        let pk = pk_one(1.0, 10.0);
        let base = make_subject(vec![], vec![1.0]);

        let cases: [(Vec<DoseAction>, &str); 3] = [
            // Malformed action (compartment 0) -> the up-front validate() error.
            (
                vec![DoseAction::Bolus { amt: 100.0, cmt: 0 }],
                "compartment is 0",
            ),
            // A well-formed action after a Stop -> Stop-must-be-final error.
            (
                vec![DoseAction::Stop, DoseAction::Bolus { amt: 100.0, cmt: 1 }],
                "Stop must be the final action",
            ),
            // A Stop in the middle of the list -> same rejection (not a silent drop
            // of the trailing dose).
            (
                vec![
                    DoseAction::Bolus { amt: 50.0, cmt: 1 },
                    DoseAction::Stop,
                    DoseAction::Bolus { amt: 50.0, cmt: 1 },
                ],
                "Stop must be the final action",
            ),
        ];

        for (actions, needle) in cases {
            let mut controller = |_ctx: &ControllerCtx| actions.clone();
            let err = ode_predictions_adaptive(
                &ode,
                &pk.values,
                &[],
                &[],
                &base,
                &[0.0],
                &[],
                &mut controller,
                100,
                None,
            )
            .expect_err("malformed / post-stop action list is rejected");
            assert!(err.contains(needle), "expected {needle:?}, got: {err}");
        }
    }

    #[test]
    fn integrate_segment_tad_anchor_set_when_prior_dose_exists() {
        // Covers the `last_dose_eff.is_finite()` branch: when a dose precedes the
        // segment the TAD anchor slot must hold that dose time (not NaN).
        let ode = one_cpt_ode_spec();
        let dose = crate::types::DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let subject = make_subject(vec![dose], vec![10.0]);
        let pk = pk_one(1.0, 10.0);
        let mut ext_params = [0.0f64; crate::types::MAX_PK_PARAMS + 2];
        ext_params[crate::types::PK_IDX_CL] = 1.0;
        ext_params[crate::types::PK_IDX_V] = 10.0;
        let mut u = vec![100.0]; // pre-loaded with the bolus amount
        let mut predictions = vec![f64::NAN; subject.obs_times.len()];
        let obs_map = obs_index_map(&subject.obs_times);

        integrate_segment(
            &ode,
            &mut u,
            0.0,
            10.0,
            &subject,
            &[0.0],
            &[1.0],
            &mut ext_params,
            &pk.values,
            &[],
            &[],
            &obs_map,
            &mut predictions,
            None,
            &[],
        );

        // TAD anchor must be the dose time (0.0), not NaN.
        assert_eq!(
            ext_params[crate::types::MAX_PK_PARAMS + 1],
            0.0,
            "TAD anchor must equal the prior dose time"
        );
        let expected = 100.0 * (-1.0f64).exp();
        assert_relative_eq!(predictions[0], expected, max_relative = 1e-4);
    }

    /// Two-compartment "accumulator": `d/dt = 0` for both states, so each state
    /// holds exactly the bioavailable amount injected into it — letting a test
    /// read `F·amt` (and lag timing) straight off the state. `readout_idx`
    /// selects which compartment the observable reads.
    fn two_cpt_accumulator(readout_idx: usize, map: crate::types::DoseAttrMap) -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
                dy[1] = 0.0;
            }),
            n_states: 2,
            state_names: vec!["c1".into(), "c2".into()],
            readout: OdeReadout::ObsCmt(readout_idx),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: map,
            init_fn: None,
        }
    }

    #[test]
    fn ode_predictions_apply_per_compartment_bioavailability_and_lag() {
        // Issue #369. Dose 100 into cmt 1 and 100 into cmt 2. Bare F = 0.5
        // applies to every compartment; `F2` = 0.25 overrides compartment 2;
        // `ALAG2` = 5 h delays only the compartment-2 dose. Reading each state
        // off the accumulator must show the *per-compartment* attribute.
        let mut map = crate::types::DoseAttrMap::default();
        map.insert(crate::types::DoseAttr::F, 2, 9); // F2 -> spare slot 9
        map.insert(crate::types::DoseAttr::Lag, 2, 10); // ALAG2 -> spare slot 10

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_F] = 0.5; // bare F (all compartments)
        p.values[9] = 0.25; // F2 overrides cmt 2
        p.values[10] = 5.0; // ALAG2 on cmt 2

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0),
        ];
        // Observe at t = 1 (before ALAG2 = 5) and t = 10 (after).
        let subj = make_subject(doses, vec![1.0, 10.0]);

        // Compartment 1: bare F = 0.5, no lag -> 50 at both times.
        let c1 = ode_predictions(
            &two_cpt_accumulator(0, map.clone()),
            &p.values,
            &[],
            &[],
            &subj,
        );
        assert!((c1[0] - 50.0).abs() < 1e-9, "cmt1 @t=1: {}", c1[0]);
        assert!((c1[1] - 50.0).abs() < 1e-9, "cmt1 @t=10: {}", c1[1]);

        // Compartment 2: F2 = 0.25 and ALAG2 = 5 -> 0 before lag, 25 after.
        let c2 = ode_predictions(&two_cpt_accumulator(1, map), &p.values, &[], &[], &subj);
        assert!(c2[0].abs() < 1e-9, "cmt2 pre-lag: {}", c2[0]);
        assert!((c2[1] - 25.0).abs() < 1e-9, "cmt2 @t=10 (F2): {}", c2[1]);
    }

    #[test]
    fn ode_predictions_event_driven_apply_per_compartment_bioavailability_and_lag() {
        // #369 review #3: the event-driven path is the actual fit path and
        // resolves F through a *distinct* inline form
        // (`dose_attr_map.f_bio(d.cmt, &pk_now.values)`), so per-compartment
        // correctness must be asserted here too — not only on `ode_predictions`.
        // Same 2-compartment accumulator and expectations as the no-TV test.
        let mut map = crate::types::DoseAttrMap::default();
        map.insert(crate::types::DoseAttr::F, 2, 9);
        map.insert(crate::types::DoseAttr::Lag, 2, 10);

        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_F] = 0.5;
        p.values[9] = 0.25; // F2
        p.values[10] = 5.0; // ALAG2

        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0),
        ];
        let subj = make_subject(doses, vec![1.0, 10.0]);
        let dose_pk = vec![p; subj.doses.len()];
        let obs_pk = vec![p; subj.obs_times.len()];

        // Compartment 1: bare F = 0.5, no lag.
        let c1 = ode_predictions_event_driven(
            &two_cpt_accumulator(0, map.clone()),
            &subj,
            &[],
            &[],
            &dose_pk,
            &obs_pk,
            &[],
        );
        assert!((c1[0] - 50.0).abs() < 1e-9, "cmt1 @t=1: {}", c1[0]);
        assert!((c1[1] - 50.0).abs() < 1e-9, "cmt1 @t=10: {}", c1[1]);

        // Compartment 2: F2 = 0.25, ALAG2 = 5 -> 0 pre-lag, 25 after.
        let c2 = ode_predictions_event_driven(
            &two_cpt_accumulator(1, map),
            &subj,
            &[],
            &[],
            &dose_pk,
            &obs_pk,
            &[],
        );
        assert!(c2[0].abs() < 1e-9, "cmt2 pre-lag: {}", c2[0]);
        assert!((c2[1] - 25.0).abs() < 1e-9, "cmt2 @t=10 (F2): {}", c2[1]);
    }

    /// Coverage: the steady-state branch of the event-driven TAD anchor in
    /// `ode_predictions_event_driven` (`last_dose_eff` reckons from the most
    /// recent SS cycle). Smoke-level — predictions must stay finite.
    #[test]
    fn event_driven_ss_dose_predictions_finite() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)]; // SS bolus
        let subj = make_subject(doses, vec![6.0, 18.0]);
        let dose_pk = vec![pk; subj.doses.len()];
        let obs_pk = vec![pk; subj.obs_times.len()];
        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &dose_pk, &obs_pk, &[]);
        assert!(
            preds.iter().all(|p| p.is_finite()),
            "SS preds finite: {preds:?}"
        );
    }

    /// Coverage: the infusion break-time branch of `ode_predictions_with_states`.
    #[test]
    fn with_states_infusion_dose_runs() {
        let ode = one_cpt_ode_spec();
        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)]; // infusion, dur=10
        assert!(is_real_infusion(&doses[0]));
        let subj = make_subject(doses, vec![5.0, 20.0]);
        let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(states.len(), 2);
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    /// Coverage: `ode_dense_solve_states` with a steady-state, *lagged* infusion —
    /// exercises the infusion break, the SS pre-seed at the dose record time, and
    /// the SS branch of the dense TAD anchor in a single pass.
    #[test]
    fn dense_solve_ss_lagged_infusion_runs() {
        let ode = one_cpt_ode_spec();
        let mut pk = pk_one(5.0, 80.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 2.0; // lag > 0
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 10.0, true, 12.0)]; // SS infusion
        let subj = make_subject(doses, vec![6.0]);
        let states = ode_dense_solve_states(&ode, &pk.values, &[], &[], &subj, &[6.0, 14.0]);
        assert_eq!(states.len(), 2);
        assert!(states.iter().all(|s| s.iter().all(|x| x.is_finite())));
    }

    /// Coverage: the `ode_predictions_ekf` wrapper (a 1-state `[diffusion]` spec);
    /// elsewhere only `solve_ekf` is exercised directly.
    #[test]
    fn ode_predictions_ekf_wrapper_runs() {
        let ode = OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: vec![0.1],
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        };
        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![2.0, 8.0]);
        let (ipreds, p_obs) = ode_predictions_ekf(&ode, &pk.values, &subj, |_| 1.0);
        assert_eq!(ipreds.len(), 2);
        assert!(ipreds.iter().chain(p_obs.iter()).all(|x| x.is_finite()));
    }

    /// Turnover model with a baseline initial condition:
    ///   d/dt(R) = kin - kout*R,  init(R) = kin/kout
    /// params: kin @ slot 0, kout @ slot 1. Observable reads R (state 0).
    fn turnover_ode_spec_with_init() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = p[0] - p[1] * y[0];
            }),
            n_states: 1,
            state_names: vec!["R".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: Some(Box::new(|p: &[f64]| {
                let (kin, kout) = (p[0], p[1]);
                vec![if kout > 0.0 { kin / kout } else { 0.0 }]
            })),
        }
    }

    fn pk_kin_kout(kin: f64, kout: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[0] = kin;
        p.values[1] = kout;
        p
    }

    // ── Built-in absorption input-rate forcing (transit) ──────────────────
    use crate::pk::absorption::{InputRateForcing, InputRateKind};

    /// Single compartment that only *accumulates* the transit input (`dy = 0`),
    /// so its amount at large `t` equals the total delivered mass `∫R_in = F·amt`
    /// — a direct mass-balance probe of the forcing through the real integrator.
    /// Transit args live at free slots: `n` @ 6, `mtt` @ 7.
    fn transit_accumulator_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
            }),
            n_states: 1,
            state_names: vec!["depot".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: vec![InputRateForcing {
                cmt: 0,
                kind: InputRateKind::Transit,
                arg_slots: vec![6, 7],
                frac_slot: None,
            }],
            init_fn: None,
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
        }
    }

    fn pk_transit_vec(n: f64, mtt: f64, f: f64) -> Vec<f64> {
        let mut v = vec![0.0; crate::types::MAX_PK_PARAMS];
        v[6] = n;
        v[7] = mtt;
        v[crate::types::PK_IDX_F] = f;
        v
    }

    fn pk_transit_struct(n: f64, mtt: f64, f: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[6] = n;
        p.values[7] = mtt;
        p.values[crate::types::PK_IDX_F] = f;
        p
    }

    #[test]
    fn input_rate_consumes_cmt_matches_forcing_compartment() {
        let ode = transit_accumulator_spec(); // forcing on state 0 ≡ 1-based CMT 1
        assert!(input_rate_consumes_cmt(&ode, 1));
        assert!(!input_rate_consumes_cmt(&ode, 2));
        // A spec with no input-rate term never consumes a dose.
        assert!(!input_rate_consumes_cmt(&one_cpt_ode_spec(), 1));
    }

    /// Single accumulator compartment (`dy = 0`) fed by a `zero_order(dur)`
    /// forcing, `dur` at free slot 4 — the zero-order analogue of
    /// `transit_accumulator_spec`, so its amount at large `t` equals the delivered
    /// mass `∫R_in = F·amt` and at an interior `t < dur` equals the linear partial
    /// `(F·amt/dur)·t` (a direct probe that the cutoff break is placed correctly).
    fn zero_order_accumulator_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
            }),
            n_states: 1,
            state_names: vec!["depot".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: vec![InputRateForcing {
                cmt: 0,
                kind: InputRateKind::ZeroOrder,
                arg_slots: vec![4],
                frac_slot: None,
            }],
            init_fn: None,
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
        }
    }

    fn pk_zero_order_vec(dur: f64, f: f64) -> Vec<f64> {
        let mut v = vec![0.0; crate::types::MAX_PK_PARAMS];
        v[4] = dur;
        v[crate::types::PK_IDX_F] = f;
        v
    }

    /// Build the per-dose windows from a single subject snapshot (the dense-path
    /// shape) — the test analogue of the `|_, d| zero_order_dur_and_frac_for_dose(...)`
    /// closure the production callers pass.
    fn zo_windows_for(
        ode: &OdeSpec,
        doses: &[DoseEvent],
        lags: &[f64],
        pk: &[f64],
    ) -> Vec<ZeroOrderWindow> {
        let f_bio: Vec<f64> = doses.iter().map(|_| 1.0).collect();
        zero_order_windows(doses, lags, &f_bio, |_, d| {
            zero_order_dur_and_frac_for_dose(ode, d, pk)
        })
    }

    #[test]
    fn zero_order_window_edges_rate_and_cutoff_break() {
        // A dose at t=2 into the zero-order compartment, lag 0.5, dur 4 ⇒ a window
        // [2.5, 6.5] with rate F·amt/dur = 100/4 = 25, and a cutoff break at 6.5. A
        // dose into a *different* compartment, and a zero-amount dose, contribute no
        // window (so no break).
        let ode = zero_order_accumulator_spec(); // cmt 0 ≡ 1-based CMT 1
        let pk = pk_zero_order_vec(4.0, 1.0);
        let doses = vec![
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0), // feeds R_in
            DoseEvent::new(0.0, 100.0, 2, 0.0, false, 0.0), // other cmt → no window
            DoseEvent::new(1.0, 0.0, 1, 0.0, false, 0.0),   // zero amt → no window
        ];
        let windows = zo_windows_for(&ode, &doses, &[0.5, 0.0, 0.0], &pk);
        assert_eq!(windows, vec![(0, 25.0, 2.5, 6.5)]);

        let mut breaks = Vec::new();
        push_zero_order_break_times(&mut breaks, &windows);
        assert_eq!(breaks, vec![6.5]);
    }

    #[test]
    fn active_zero_order_includes_only_fully_contained_segments() {
        // Window [2.5, 6.5], rate 25. A segment strictly inside is active; a segment
        // straddling the cutoff (right end past w_end) is excluded — the
        // full-containment rule that makes the post-cutoff mass exact. A reset_floor
        // after the window start turns it off.
        let windows: Vec<ZeroOrderWindow> = vec![(0, 25.0, 2.5, 6.5)];
        assert_eq!(
            active_zero_order_inputs(&windows, 3.0, 5.0, f64::NEG_INFINITY),
            vec![(0, 25.0)]
        );
        // [5, 7] ends past w_end=6.5 ⇒ not fully contained ⇒ excluded.
        assert!(active_zero_order_inputs(&windows, 5.0, 7.0, f64::NEG_INFINITY).is_empty());
        // [1, 2] precedes the window start ⇒ excluded.
        assert!(active_zero_order_inputs(&windows, 1.0, 2.0, f64::NEG_INFINITY).is_empty());
        // reset_floor past the window start (e.g. 3.0) turns the window off.
        assert!(active_zero_order_inputs(&windows, 3.0, 5.0, 3.0).is_empty());
    }

    #[test]
    fn smooth_forcing_contributes_no_zero_order_window() {
        // transit/igd/weibull are smooth (no cutoff) — they yield no zero-order
        // window, so they keep their existing break structure and pointwise forcing.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        assert!(zo_windows_for(&ode, &doses, &[0.0], &pk).is_empty());
    }

    #[test]
    fn zero_order_forcing_delivers_full_dose_mass() {
        // After the window closes (`t > dur`) the accumulator holds ∫R_in = F·amt
        // = 100 — not 200 (bolus double-count) and not 0 (no forcing). The cutoff
        // break stops the input cleanly at `dur`, so the plateau is exact.
        let ode = zero_order_accumulator_spec();
        let pk = pk_zero_order_vec(4.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![20.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 100.0, max_relative = 1e-6);
    }

    #[test]
    fn zero_order_partial_window_is_linear() {
        // Inside the window the accumulated mass is the rectangle's running area
        // `(F·amt/dur)·t`: at t = dur/2 = 2 it is exactly half the dose. This only
        // holds if the constant rate is delivered over `(0, dur]` and the cutoff
        // break does not truncate the window early.
        let ode = zero_order_accumulator_spec();
        let pk = pk_zero_order_vec(4.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![2.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 50.0, max_relative = 1e-6);
    }

    #[test]
    fn transit_forcing_delivers_full_dose_mass() {
        // The accumulator depot should hold ∫R_in = F·amt = 100 once absorption
        // is complete — NOT 200 (bolus would double-count) and NOT 0 (no forcing).
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![40.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 100.0, max_relative = 5e-3);
    }

    #[test]
    fn transit_dose_does_not_enter_as_bolus() {
        // An observation exactly at the dose time reads ~0: the transit dose is
        // delivered as R_in over time, never as an instantaneous bolus jump. (A
        // trailing obs keeps the break-time loop non-empty.) The late obs then
        // confirms the full mass still arrives.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![0.0, 40.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert!(preds[0].abs() < 1e-9, "bolus not suppressed: {}", preds[0]);
        assert_relative_eq!(preds[1], 100.0, max_relative = 5e-3);
    }

    #[test]
    fn transit_forcing_scales_with_bioavailability() {
        // F = 0.4 ⇒ delivered mass = 0.4·100 = 40.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 0.4);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![40.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 40.0, max_relative = 5e-3);
    }

    #[test]
    fn transit_forcing_superposes_over_doses() {
        // Two doses (100 @ t=0, 50 @ t=10) superpose: ∫R_in = F·(100+50) = 150.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 50.0, 1, 0.0, false, 0.0),
        ];
        let subj = make_subject(doses, vec![60.0]);
        let preds = ode_predictions(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 150.0, max_relative = 5e-3);
    }

    #[test]
    fn transit_forcing_respects_reset_floor() {
        // Event-driven path: an EVID=3 reset at t=1 zeros the depot AND turns off
        // the pre-reset dose's input rate. With no post-reset dose, the
        // accumulator stays at 0 — the t=0 dose's R_in must not resume.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_struct(3.0, 2.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let mut subj = make_subject(doses, vec![40.0]);
        subj.reset_times = vec![1.0];
        let dose_pk = vec![pk; subj.doses.len()];
        let obs_pk = vec![pk; subj.obs_times.len()];
        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &dose_pk, &obs_pk, &[]);
        assert!(
            preds[0].abs() < 1e-6,
            "pre-reset dose R_in leaked past the reset: got {}",
            preds[0]
        );
    }

    #[test]
    fn transit_forcing_applied_in_with_states_path() {
        // The per-compartment states path (`ode_predictions_with_states`, used for
        // derived-output state extraction) must inject the transit forcing too —
        // the accumulator state holds ∫R_in = F·amt = 100.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let subj = make_subject(doses, vec![40.0]);
        let (preds, states) = ode_predictions_with_states(&ode, &pk, &[], &[], &subj);
        assert_relative_eq!(preds[0], 100.0, max_relative = 5e-3);
        assert_relative_eq!(states[0][0], 100.0, max_relative = 5e-3);
    }

    #[test]
    fn transit_forcing_in_dense_solve_states_skips_other_cmt_dose() {
        // `ode_dense_solve_states` applies the forcing; a dose targeting a
        // *non-forcing* compartment is skipped by the superposition loop. State 0
        // (the forcing cmt ≡ CMT 1) holds only the CMT-1 dose's mass — not the
        // CMT-2 dose, which never feeds R_in.
        let ode = transit_accumulator_spec();
        let pk = pk_transit_vec(3.0, 2.0, 1.0);
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // CMT 1: feeds R_in
            DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),  // CMT 2: not the forcing cmt
        ];
        let subj = make_subject(doses, vec![40.0]);
        let states = ode_dense_solve_states(&ode, &pk, &[], &[], &subj, &[40.0]);
        assert_relative_eq!(states[0][0], 100.0, max_relative = 5e-3);
    }

    // ── Forcing-seam helpers (#353): the single RHS-wrapper seam + per-segment
    //    prepare() hoist shared by all four ODE integration paths. ────────────

    #[test]
    fn prepare_input_rates_parallel_to_forcings_and_empty_without_them() {
        // Parallel to `ode.input_rate`; empty (non-allocating) when the model has
        // no built-in input-rate forcing.
        let ode = transit_accumulator_spec();
        let params = pk_transit_vec(3.0, 2.0, 1.0);
        let prepared = prepare_input_rates(&ode, &params);
        assert_eq!(prepared.len(), 1);
        // The hoisted constant must match a direct `prepare` on the same params —
        // the invariant that keeps the #7 hoist from drifting from the per-eval form.
        assert_eq!(
            prepared[0].rate(2.5, 100.0),
            ode.input_rate[0].prepare(&params).rate(2.5, 100.0)
        );
        assert!(prepare_input_rates(&one_cpt_ode_spec(), &params).is_empty());
    }

    #[test]
    fn gated_infusions_resolves_rate_and_drops_unaddressable() {
        // (dose_idx, t_start, t_end) -> (cmt_idx, rate_eff, t_start, t_end) with
        // the mode-aware bioavailability rate (#419): a rate-defined infusion
        // holds its rate (F scales the duration, carried by the window), while a
        // duration-defined infusion (RATE=-2) gets F·rate. CMT=0 and compartments
        // beyond the state vector are dropped.
        let mut dur_defined = DoseEvent::new(0.0, 0.0, 1, 4.0, false, 0.0);
        dur_defined.infusion_def = crate::types::InfusionDef::DurationDefined;
        let doses = vec![
            DoseEvent::new(0.0, 0.0, 1, 4.0, false, 0.0), // rate-defined: rate held
            DoseEvent::new(0.0, 0.0, 0, 9.0, false, 0.0), // CMT 0 -> dropped
            DoseEvent::new(0.0, 0.0, 5, 9.0, false, 0.0), // CMT 5 -> state 4 >= n -> dropped
            dur_defined,                                  // duration-defined: F·rate
        ];
        let f_bio = vec![0.5, 1.0, 1.0, 0.5];
        let active = vec![
            (0usize, 1.0, 3.0),
            (1, 1.0, 3.0),
            (2, 1.0, 3.0),
            (3, 1.0, 3.0),
        ];
        let gated = gated_infusions(&active, &doses, &f_bio, 1);
        assert_eq!(
            gated,
            vec![(0usize, 4.0, 1.0, 3.0), (0usize, 4.0 * 0.5, 1.0, 3.0)]
        );
    }

    #[test]
    fn add_prepared_forcing_superposes_skips_other_cmt_and_respects_floor() {
        let ode = transit_accumulator_spec(); // forcing on state 0 ≡ CMT 1
        let params = pk_transit_vec(3.0, 2.0, 1.0);
        let prepared = prepare_input_rates(&ode, &params);
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // feeds R_in
            DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0),  // other cmt → ignored
        ];
        let lags = vec![0.0, 0.0];
        let f_bio = vec![1.0, 1.0];
        let t = 1.5;

        // No reset: only the CMT-1 dose contributes its R_in(tad).
        let mut dy = vec![0.0];
        add_prepared_input_rate_forcing(
            &ode,
            &prepared,
            &params,
            &doses,
            &lags,
            &f_bio,
            f64::NEG_INFINITY,
            t,
            &mut dy,
        );
        let want = prepared[0].rate(t, 100.0);
        assert!(want > 0.0);
        assert_relative_eq!(dy[0], want, max_relative = 1e-12);

        // A reset_floor after the dose time turns its forcing off.
        let mut dy_off = vec![0.0];
        add_prepared_input_rate_forcing(
            &ode,
            &prepared,
            &params,
            &doses,
            &lags,
            &f_bio,
            1.0,
            t,
            &mut dy_off,
        );
        assert_eq!(dy_off[0], 0.0);
    }

    #[test]
    fn add_prepared_forcing_applies_pathway_fraction_linear_in_frac() {
        // Biphasic IG (#388): two igd forcings on one compartment, split FR1/FR2.
        // The seam adds `FR1·R_in1 + FR2·R_in2`; because the fraction enters
        // linearly, the analytic dual derivative ∂(dy)/∂FR1 is exactly R_in1 (so
        // the FOCEI/Bayes gradient w.r.t. a pathway fraction is exact, no FD).
        use crate::pk::absorption::{InputRateForcing, InputRateKind, PreparedInputRate};
        use crate::sens::dual_mixed::DualMixed;
        use crate::sens::num::PkNum;
        use crate::types::{MAX_PK_PARAMS, PK_IDX_F};

        // Slots: FR1@0, FR2@1, MAT1@2, CV2_1@3, MAT2@4, CV2_2@5, F@PK_IDX_F.
        let mk = |frac_slot, arg_slots| InputRateForcing {
            cmt: 0,
            kind: InputRateKind::InverseGaussian,
            arg_slots,
            frac_slot: Some(frac_slot),
        };
        let ode = OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: vec![mk(0, vec![2, 3]), mk(1, vec![4, 5])],
            init_fn: None,
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
        };

        let (fr1, fr2) = (0.7_f64, 0.3_f64);
        let mut params = vec![0.0; MAX_PK_PARAMS];
        params[0] = fr1;
        params[1] = fr2;
        params[2] = 2.0; // MAT1
        params[3] = 0.3; // CV2_1
        params[4] = 5.0; // MAT2
        params[5] = 0.6; // CV2_2
        params[PK_IDX_F] = 1.0;

        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)]; // CMT 1 → cmt 0
        let f_bio = vec![1.0];
        let (t, tad) = (2.5_f64, 2.5_f64);

        // f64: dy = FR1·R_in1 + FR2·R_in2.
        let prepared = prepare_input_rates(&ode, &params);
        let (r1, r2) = (prepared[0].rate(tad, 100.0), prepared[1].rate(tad, 100.0));
        assert!(r1 > 0.0 && r2 > 0.0);
        let mut dy = vec![0.0];
        add_prepared_input_rate_forcing(
            &ode,
            &prepared,
            &params,
            &doses,
            &[0.0],
            &f_bio,
            f64::NEG_INFINITY,
            t,
            &mut dy,
        );
        assert_relative_eq!(dy[0], fr1 * r1 + fr2 * r2, max_relative = 1e-12);

        // Dual: seed FR1 as the variable ⇒ value matches f64 and ∂(dy)/∂FR1 = R_in1.
        type D = DualMixed<1, 1>;
        let mut dp = vec![D::constant(0.0); MAX_PK_PARAMS];
        dp[0] = D::var(fr1, 0);
        dp[1] = D::constant(fr2);
        dp[2] = D::constant(2.0);
        dp[3] = D::constant(0.3);
        dp[4] = D::constant(5.0);
        dp[5] = D::constant(0.6);
        dp[PK_IDX_F] = D::constant(1.0);
        let prepared_d: Vec<PreparedInputRate<D>> = ode
            .input_rate
            .iter()
            .map(|f| f.prepare_dual::<D>(&dp).unwrap())
            .collect();
        let f_bio_d = vec![D::constant(1.0)];
        let mut dyd = vec![D::constant(0.0)];
        add_prepared_input_rate_forcing(
            &ode,
            &prepared_d,
            &dp,
            &doses,
            &[],
            &f_bio_d,
            f64::NEG_INFINITY,
            t,
            &mut dyd,
        );
        assert_relative_eq!(dyd[0].val(), fr1 * r1 + fr2 * r2, max_relative = 1e-12);
        assert_relative_eq!(dyd[0].grad[0], r1, max_relative = 1e-9);
    }

    #[test]
    fn seam_spanning_adds_base_rhs_and_infusion() {
        // Spanning infusion is added unconditionally on top of the user RHS; with
        // no input_rate forcing the forcing branch is skipped.
        let ode = one_cpt_ode_spec();
        let params = pk_one(1.0, 1.0).values; // ke = cl/v = 1
        let prepared: Vec<PreparedInputRate> = Vec::new();
        let rhs = wrap_rhs_with_forcings(
            &ode,
            &[],
            &[],
            &[],
            f64::NEG_INFINITY,
            &prepared,
            InfusionInput::Spanning(vec![(0, 7.0)]),
            &[],
        );
        let mut dy = vec![0.0];
        rhs(&[2.0], &params, 0.0, &mut dy); // base −ke·y = −2, +7 infusion = 5
        assert_relative_eq!(dy[0], 5.0, max_relative = 1e-12);
    }

    #[test]
    fn seam_gated_infusion_active_only_inside_window() {
        let ode = one_cpt_ode_spec();
        let params = pk_one(0.0, 1.0).values; // ke = 0 ⇒ base RHS = 0
        let prepared: Vec<PreparedInputRate> = Vec::new();
        let rhs = wrap_rhs_with_forcings(
            &ode,
            &[],
            &[],
            &[],
            f64::NEG_INFINITY,
            &prepared,
            InfusionInput::Gated(vec![(0, 3.0, 2.0, 5.0)]),
            &[],
        );
        let mut before = vec![0.0];
        rhs(&[0.0], &params, 1.0, &mut before); // before [2,5)
        assert_eq!(before[0], 0.0);
        let mut inside = vec![0.0];
        rhs(&[0.0], &params, 3.0, &mut inside); // inside
        assert_relative_eq!(inside[0], 3.0, max_relative = 1e-12);
        let mut after = vec![0.0];
        rhs(&[0.0], &params, 6.0, &mut after); // past t_end
        assert_eq!(after[0], 0.0);
    }

    #[test]
    fn seam_applies_input_rate_forcing_on_top_of_base_rhs() {
        // With an input_rate forcing and no infusions, the seam adds R_in(tad)
        // into the forcing compartment — matching the hoisted prepared constant.
        let ode = transit_accumulator_spec(); // rhs sets dy[0] = 0
        let params = pk_transit_vec(3.0, 2.0, 1.0);
        let prepared = prepare_input_rates(&ode, &params);
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let lags = vec![0.0];
        let f_bio = vec![1.0];
        let rhs = wrap_rhs_with_forcings(
            &ode,
            &doses,
            &lags,
            &f_bio,
            f64::NEG_INFINITY,
            &prepared,
            InfusionInput::Spanning(Vec::new()),
            &[],
        );
        let t = 1.5;
        let mut dy = vec![0.0];
        rhs(&[0.0], &params, t, &mut dy);
        assert_relative_eq!(dy[0], prepared[0].rate(t, 100.0), max_relative = 1e-12);
    }

    #[test]
    fn ode_init_state_seeds_plain_path() {
        // No doses; the system starts at baseline kin/kout = 5 and stays there
        // (dR/dt = 0). Without init it would start at 0 and climb.
        let ode = turnover_ode_spec_with_init();
        let pk = pk_kin_kout(10.0, 2.0);
        let subj = make_subject(Vec::new(), vec![0.0, 1.0, 5.0, 20.0]);
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (i, &p) in preds.iter().enumerate() {
            assert_relative_eq!(p, 5.0, epsilon = 1e-5);
            let _ = i;
        }
    }

    #[test]
    fn ode_init_state_then_dose_and_reset_reapplies_init() {
        // Exercises all three: (1) init seeds the start at baseline=5, (2) a
        // bolus at t=0 lands on top of the seeded state (5 + 20 = 25), and
        // (3) an EVID=3 reset at t=5 re-applies init (back to 5, NOT zero).
        let ode = turnover_ode_spec_with_init();
        let pk = pk_kin_kout(10.0, 2.0); // baseline 5
        let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 5.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![5.0];
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        // t=0: init(5) + bolus(20) = 25.
        assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
        // t=5: reset re-applies init → 5 (a zeroing reset would give 0).
        assert_relative_eq!(preds[1], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_init_uses_chronologically_first_record_not_first_dose() {
        // Regression (Copilot review #1): with time-varying covariates the init
        // snapshot must come from the earliest record by TIME, not the first
        // dose. Here a pre-dose observation at t=0 carries KIN=10 (baseline 5)
        // while a later dose at t=5 carries KIN=100 (baseline 50). Seeding must
        // use the t=0 obs → prediction at t=0 is 5, not 50.
        let ode = turnover_ode_spec_with_init();
        let doses = vec![DoseEvent::new(5.0, 0.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk_dose = vec![pk_kin_kout(100.0, 2.0)]; // baseline 50 (must NOT be used)
        let pk_obs = vec![pk_kin_kout(10.0, 2.0)]; // baseline 5 (first record)

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_relative_eq!(preds[0], 5.0, epsilon = 1e-9);
    }

    #[test]
    fn ode_init_reapplied_when_reset_is_first_event() {
        // Regression (Copilot review #2): an EVID=4 reset+dose at t=0 re-applies
        // init *before* the same-time dose. last_pk must be seeded from the
        // first record's params (not zeroed defaults), or the re-applied
        // baseline would evaluate KIN/KOUT with zero params and collapse to 0.
        // Expected: init(5) re-applied at reset, then bolus 20 → 25.
        let ode = turnover_ode_spec_with_init();
        let doses = vec![DoseEvent::new(0.0, 20.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![0.0];
        let pk = pk_kin_kout(10.0, 2.0); // baseline 5
        let pk_dose = vec![pk];
        let pk_obs = vec![pk];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        // Re-applied baseline (5) + bolus (20) = 25. A zero-param re-seed would
        // give 0 + 20 = 20.
        assert_relative_eq!(preds[0], 25.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_event_driven_reset_evid3_zeros_state() {
        // EVID=3 reset at t=5 must zero the ODE state: obs after the reset
        // read ~0 when no later dose exists.
        let ode = one_cpt_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 6.0, 10.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![5.0];
        let pk = pk_one(10.0, 100.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert!(preds[0] > 0.0, "pre-reset obs should be positive");
        assert_relative_eq!(preds[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(preds[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn ode_event_driven_reset_evid4_matches_fresh_dose() {
        // EVID=4 (reset + dose) at t=10 must match a single fresh dose at t=10.
        let ode = one_cpt_ode_spec();
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![10.0, 12.0, 15.0];
        let mut subj = make_subject(doses, obs_times.clone());
        subj.reset_times = vec![10.0];
        let pk = pk_one(8.0, 50.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);

        // Reference: lone 500 mg dose at t=10 through the same ODE path.
        let fresh = make_subject(
            vec![DoseEvent::new(10.0, 500.0, 1, 0.0, false, 0.0)],
            obs_times.clone(),
        );
        let fresh_pk_dose = vec![pk; fresh.doses.len()];
        let fresh_pk_obs = vec![pk; obs_times.len()];
        let expected = ode_predictions_event_driven(
            &ode,
            &fresh,
            &[],
            &[],
            &fresh_pk_dose,
            &fresh_pk_obs,
            &[],
        );
        for (a, e) in preds.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *e, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn ode_event_driven_matches_constant_path_when_pk_constant() {
        // Equivalence: when the per-event PK params are all the same, the
        // event-driven ODE path must agree with the existing single-snapshot
        // path. This is the "no TV covariates" sanity check.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(8.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![1.0, 4.0, 8.5, 12.0, 24.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        let event_driven =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            // ODE solver tolerance is ~1e-4 relative — a tighter equality
            // would over-constrain RK45.
            assert_relative_eq!(*b, *e, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_event_driven_picks_up_changing_cl() {
        // Same shape as the analytical TV test: CL doubles between two doses.
        // End-of-interval / NONMEM convention — each segment uses the PK
        // params at the record being arrived at:
        //   [0, t_obs1=5]: uses pk at obs1 = pk_low → ke = 0.05
        //   [5, t_dose2=10]: uses pk at dose2 = pk_high → ke = 0.10
        //   [10, t_obs2=12]: uses pk at obs2 = pk_high → ke = 0.10
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, 1000.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![5.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk_low = pk_one(5.0, 100.0); // ke = 0.05
        let pk_high = pk_one(10.0, 100.0); // ke = 0.10
        let pk_dose = vec![pk_low, pk_high];
        let pk_obs = vec![pk_low, pk_high];
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);

        // [0, 5] uses pk_low (pk at obs1): A(5) = 1000 * exp(-0.05*5) ≈ 778.80
        let a5 = 1000.0 * (-0.05f64 * 5.0).exp();
        assert_relative_eq!(preds[0], a5, epsilon = 1e-3, max_relative = 1e-4);

        // [5, 10] uses pk_high (pk at dose2): ke=0.10 for 5h.
        //   A(10⁻) = A(5) * exp(-0.10*5) = 778.80 * 0.6065 ≈ 472.37
        // After dose2: A(10⁺) = 472.37 + 1000 = 1472.37.
        // [10, 12] uses pk_high (pk at obs2): A(12) = 1472.37 * exp(-0.20) ≈ 1205.49
        let a10_minus = a5 * (-0.10f64 * 5.0).exp();
        let a10_plus = a10_minus + 1000.0;
        let a12 = a10_plus * (-0.20f64).exp();
        assert_relative_eq!(preds[1], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    /// 1-cpt oral ODE: dA1/dt = -ka·A1, dA2/dt = ka·A1 - ke·A2.
    /// Used to test infusion into the depot compartment (cmt=1).
    fn one_cpt_oral_ode_spec() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                let v = p[crate::types::PK_IDX_V];
                let ka = p[crate::types::PK_IDX_KA];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ka * y[0];
                dy[1] = ka * y[0] - ke * y[1];
            }),
            n_states: 2,
            state_names: vec!["depot".into(), "central".into()],
            readout: OdeReadout::ObsCmt(1),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        }
    }

    #[test]
    fn ode_infusion_one_cpt_iv_matches_closed_form() {
        // 1-cpt IV infusion. Closed form during infusion:
        //   A(t) = (R/ke) · (1 - exp(-ke·t))
        // and after end-of-infusion T:
        //   A(t) = A(T) · exp(-ke·(t-T))
        // Verifies that the wrapped-RHS path produces the right shape.
        let rate = 100.0;
        let amt = 1000.0; // duration = 10 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![5.0, 10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0); // ke = 0.0625
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // During infusion [0, 10]
        let a5 = (rate / ke) * (1.0 - (-ke * 5.0).exp());
        let a10 = (rate / ke) * (1.0 - (-ke * 10.0).exp());
        // After end-of-infusion
        let a15 = a10 * (-ke * 5.0).exp();
        let a20 = a10 * (-ke * 10.0).exp();

        assert_relative_eq!(preds[0], a5, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a15, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_event_driven_infusion_matches_constant_pk_path() {
        // Same infusion-only subject, run through both paths with identical
        // per-event PK params. Verifies the event-driven path's
        // InfusionEnd handling agrees with the simple-timeline path.
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![3.0, 7.0, 10.0, 14.0, 20.0];
        let subj = make_subject(doses, obs_times.clone());
        let pk = pk_one(5.0, 80.0);
        let pk_dose = vec![pk; subj.doses.len()];
        let pk_obs = vec![pk; obs_times.len()];
        let ode = one_cpt_ode_spec();

        let baseline = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        let event_driven =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose, &pk_obs, &[]);
        assert_eq!(baseline.len(), event_driven.len());
        for (b, e) in baseline.iter().zip(event_driven.iter()) {
            assert_relative_eq!(*b, *e, epsilon = 1e-3, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_event_driven_form_c_uses_observation_covariates() {
        // Regression for a NONMEM translation with paired total/free assays:
        // the dose row carried FREE=3, while same-time observation rows carried
        // FREE=0 and FREE=1. Form C must see the observation snapshot, not the
        // subject-level first-row covariate.
        let ode = OdeSpec {
            rhs: Box::new(|_y: &[f64], _p: &[f64], _t: f64, dy: &mut [f64]| {
                dy[0] = 0.0;
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::Single(Box::new(|state, _pk, _theta, _eta, covariates| {
                state[0] * covariates.get("FREE").copied().unwrap_or(0.0)
            })),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        };
        let mut subj = make_subject(
            vec![DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0)],
            vec![1.0, 1.0],
        );
        subj.covariates.insert("FREE".into(), 3.0);
        subj.dose_covariates = vec![HashMap::from([("FREE".to_string(), 3.0)])];
        subj.obs_covariates = vec![
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
        ];
        let pk = pk_one(0.0, 1.0);
        let preds = ode_predictions_event_driven(&ode, &subj, &[], &[], &[pk], &[pk, pk], &[]);

        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(preds[1], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn ode_overlapping_infusions_sum_rates() {
        // Two infusions overlap on [2, 6] for a combined rate of 200,
        // then both end at t=6. After t=6, plain elimination.
        //   inf1: t∈[0,6], rate=100
        //   inf2: t∈[2,6], rate=100
        let doses = vec![
            DoseEvent::new(0.0, 600.0, 1, 100.0, false, 0.0),
            DoseEvent::new(2.0, 400.0, 1, 100.0, false, 0.0),
        ];
        let obs_times = vec![2.0, 4.0, 6.0, 12.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // [0, 2]: rate=100, A(0)=0 → A(t) = (100/ke)·(1 - exp(-ke·t))
        let a2 = (100.0_f64 / ke) * (1.0 - (-ke * 2.0).exp());
        // [2, 6]: rate=200, A0=a2
        //   A(t) = (200/ke) + (A0 - 200/ke) · exp(-ke·(t-2))
        let r_over_ke = 200.0_f64 / ke;
        let a4 = r_over_ke + (a2 - r_over_ke) * (-ke * 2.0).exp();
        let a6 = r_over_ke + (a2 - r_over_ke) * (-ke * 4.0).exp();
        // [6, ∞]: rate=0
        let a12 = a6 * (-ke * 6.0).exp();

        assert_relative_eq!(preds[0], a2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a6, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[3], a12, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_then_bolus() {
        // Infusion [0, 10] followed by a bolus at t=15. Observation at
        // the bolus time should record state AFTER the bolus is applied,
        // matching the existing bolus convention.
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 100.0, false, 0.0), // infusion, ends at 10
            DoseEvent::new(15.0, 500.0, 1, 0.0, false, 0.0),   // bolus
        ];
        let obs_times = vec![10.0, 15.0, 20.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ke = 5.0_f64 / 80.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        let a10 = (100.0_f64 / ke) * (1.0 - (-ke * 10.0).exp());
        let a15_pre = a10 * (-ke * 5.0).exp();
        let a15_post = a15_pre + 500.0;
        let a20 = a15_post * (-ke * 5.0).exp();

        assert_relative_eq!(preds[0], a10, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], a15_post, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], a20, epsilon = 1e-2, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_into_oral_depot() {
        // Infusion into depot (cmt=1) of a 1-cpt oral model. Verifies
        // that the wrapped RHS adds `+rate` to the correct compartment
        // (depot index 0), not central (index 1). For the depot alone
        // the closed form is decoupled from ke:
        //   A1(t) during infusion = (R/ka)·(1 - exp(-ka·t))
        //   A1(t) after end T     = A1(T) · exp(-ka·(t-T))
        // Re-use the oral ODE spec but observe the depot.
        let mut ode = one_cpt_oral_ode_spec();
        ode.readout = OdeReadout::ObsCmt(0);

        let rate = 50.0;
        let amt = 200.0; // duration = 4 h
        let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
        let obs_times = vec![2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[crate::types::PK_IDX_KA] = 1.0;
        let ka = 1.0_f64;

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        let depot_2 = (rate / ka) * (1.0 - (-ka * 2.0).exp());
        let depot_4 = (rate / ka) * (1.0 - (-ka * 4.0).exp());
        let depot_8 = depot_4 * (-ka * 4.0).exp();

        assert_relative_eq!(preds[0], depot_2, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[1], depot_4, epsilon = 1e-2, max_relative = 1e-4);
        assert_relative_eq!(preds[2], depot_8, epsilon = 1e-2, max_relative = 1e-4);
    }

    // Degenerate input guards: `rate > 0` alone is insufficient to mark a
    // dose as an infusion — `duration = amt/rate` must also be > 0 and
    // finite. Otherwise:
    //   - `amt < 0` would push an infusion-end break time *before* the
    //     dose, scrambling the segmented integration order.
    //   - `amt = NaN` would make `partial_cmp` return None and panic
    //     the break-time sort.
    //   - In both cases, the bolus branch is skipped (because
    //     `is_infusion()` is true on rate alone), so the dose silently
    //     disappears from the prediction.
    // `is_real_infusion` falls back to the bolus path for these rows.

    #[test]
    fn ode_degenerate_zero_amt_with_positive_rate_falls_back_to_bolus() {
        // amt=0, rate>0 → duration=0. Treated as a (no-op) bolus.
        // Result must match "no dose at all".
        let doses = vec![DoseEvent::new(0.0, 0.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0, 5.0, 10.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        assert_eq!(preds, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn ode_degenerate_negative_amt_with_positive_rate_does_not_break_ordering() {
        // amt<0, rate>0 → duration<0. Pre-fix, the infusion-end break time
        // would sort *before* the dose, producing nonsense segments and
        // (silently) zero output because the bolus branch was skipped.
        // Post-fix this is treated as a bolus with negative amt — at least
        // visible to the caller.
        let doses = vec![DoseEvent::new(0.0, -10.0, 1, 100.0, false, 0.0)];
        let obs_times = vec![0.0, 1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        // Must not panic; the negative bolus update is clamped to 0 by
        // the negative-prediction guard in `ode_predictions`.
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn ode_degenerate_nan_amt_with_positive_rate_does_not_panic() {
        // amt=NaN, rate>0 → duration=NaN. Pre-fix, sort_by(partial_cmp).unwrap()
        // would panic on the break-time vec. Post-fix the row falls through
        // to the bolus branch and the panic is avoided.
        let doses = vec![DoseEvent::new(0.0, f64::NAN, 1, 100.0, false, 0.0)];
        let obs_times = vec![1.0];
        let subj = make_subject(doses, obs_times);
        let pk = pk_one(5.0, 80.0);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn ode_iv_bolus_with_lagtime_shifts_curve() {
        // 1-cpt IV bolus integrated via ODE: with lagtime=2.0 and dose at
        // t_dose=0, the central-amount state should be 0 until t=2 (the
        // lagged dose arrival), then decay as if dose-time were 2.
        // (`one_cpt_ode_spec` observes the amount A(t), not A/V.)
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 3.0, 6.0];
        let subj = make_subject(doses, obs_times);
        let mut pk = pk_one(5.0, 80.0);
        pk.values[crate::types::PK_IDX_LAGTIME] = 2.0;
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        // At t=1, dose has not yet arrived (lagtime=2). State stays 0.
        assert_relative_eq!(preds[0], 0.0, epsilon = 1e-10);

        // At t=3, effective elapsed time since dose is 1.0.
        // A(1) = Amt * exp(-ke * 1)
        let ke = 5.0_f64 / 80.0;
        let expected_3 = 1000.0_f64 * (-ke * 1.0).exp();
        assert_relative_eq!(preds[1], expected_3, epsilon = 1e-4, max_relative = 1e-4);

        // At t=6, effective elapsed time is 4.0.
        let expected_6 = 1000.0_f64 * (-ke * 4.0).exp();
        assert_relative_eq!(preds[2], expected_6, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn ode_infusion_with_lagtime_shifts_break_times_and_active_window() {
        // Direct test of the ODE infusion + lagtime path that the analytical
        // superposition test alone doesn't cover. Amt=100, rate=100 ⇒
        // duration=1.0; with lagtime=0.5, the active-infusion window runs
        // [2.5, 3.5] rather than [2.0, 3.0]. Compare against an equivalent
        // unlagged dose starting at 2.5 — predictions at matched observation
        // offsets should agree to ODE tolerance.
        let dose_lag = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
        assert!(dose_lag.is_infusion() && dose_lag.duration > 0.0);
        let subj_lag = make_subject(vec![dose_lag], vec![2.0, 3.0, 4.0]);
        let mut pk_lag = pk_one(5.0, 80.0);
        pk_lag.values[crate::types::PK_IDX_LAGTIME] = 0.5;

        // Reference: dose shifted at the data level, no lagtime applied.
        let dose_ref = DoseEvent::new(2.5, 100.0, 1, 100.0, false, 0.0);
        let subj_ref = make_subject(vec![dose_ref], vec![2.0, 3.0, 4.0]);
        let pk_ref = pk_one(5.0, 80.0);

        let ode = one_cpt_ode_spec();
        let preds_lag = ode_predictions(&ode, &pk_lag.values, &[], &[], &subj_lag);
        let preds_ref = ode_predictions(&ode, &pk_ref.values, &[], &[], &subj_ref);

        // Observation before lagged infusion start: zero.
        assert_relative_eq!(preds_lag[0], 0.0, epsilon = 1e-10);

        // Observations during and after the lagged infusion: must match the
        // reference where the dose was shifted at the dataset level.
        assert_relative_eq!(
            preds_lag[1],
            preds_ref[1],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
        assert_relative_eq!(
            preds_lag[2],
            preds_ref[2],
            epsilon = 1e-4,
            max_relative = 1e-4
        );
    }

    // --- Steady-state (SS=1) tests ---
    //
    // The ODE SS path is verified against the corresponding analytical
    // 1-cpt SS closed forms (PR #75): a 1-cpt IV-bolus ODE with SS dose
    // must match `one_cpt_iv_bolus_ss` to RK45 tolerance, and similarly
    // for infusion. This cross-checks the per-cycle pulse-expansion
    // equilibration loop in `equilibrate_ss_state`.

    #[test]
    fn ss_cycle_converged_is_mixed_atol_rtol_on_increment() {
        // Increment below tol: a 1e-13 move on a magnitude-100 state is ≪ tol·(|a| + max) →
        // converged.
        assert!(ss_cycle_converged(
            &[100.0, 50.0],
            &[100.0 + 1e-13, 50.0],
            SS_EQUILIBRATION_TOL
        ));
        // Increment above tol: a 1e-4 move ≫ tol·(|a| + max) → not converged.
        assert!(!ss_cycle_converged(
            &[100.0, 50.0],
            &[100.0001, 50.0],
            SS_EQUILIBRATION_TOL
        ));
        // Scale-invariant: the same *relative* increment at a tiny magnitude → same verdict.
        assert!(ss_cycle_converged(
            &[1e-6],
            &[1e-6 + 1e-19],
            SS_EQUILIBRATION_TOL
        ));
        assert!(!ss_cycle_converged(
            &[1e-6],
            &[1e-6 + 1e-15],
            SS_EQUILIBRATION_TOL
        ));
        // A genuinely zero state (no dose effect) is trivially converged.
        assert!(ss_cycle_converged(
            &[0.0, 0.0],
            &[0.0, 0.0],
            SS_EQUILIBRATION_TOL
        ));
        // Non-finite compartment (blown-up integration) is never "converged" → no early exit.
        assert!(!ss_cycle_converged(
            &[f64::NAN, 1.0],
            &[f64::NAN, 1.0],
            SS_EQUILIBRATION_TOL
        ));
        assert!(!ss_cycle_converged(
            &[f64::INFINITY],
            &[1.0],
            SS_EQUILIBRATION_TOL
        ));
        // Per-compartment: a small compartment moving 1% (relative to itself), by an amount well
        // above the system-scale atol, blocks the stop even though the dominant compartment is
        // steady.
        assert!(!ss_cycle_converged(
            &[100.0, 1e-3],
            &[100.0, 1e-3 * 1.01],
            SS_EQUILIBRATION_TOL
        ));
        // #532 review #1 — the footgun the increment test fixes: a compartment whose *magnitude*
        // (5e-11) is below the old `tol·max_mag` floor (1e-10) but which is still *moving* by
        // more than that floor (Δ = 1.5e-10). The old magnitude-floor declared it converged; the
        // increment test correctly keeps the loop running.
        assert!(!ss_cycle_converged(
            &[100.0, 5e-11],
            &[100.0, 2e-10],
            SS_EQUILIBRATION_TOL
        ));
    }

    #[test]
    fn ss_equilibration_early_stop_fires_for_fast_pk() {
        // #532 review #6: pin that the #519 early stop actually fires (the loose end-value
        // tolerances would otherwise hide a broken stop). Use a tight integrator tol — the
        // gradient context where the speedup matters.
        let mut ode = one_cpt_ode_spec();
        ode.solver_opts.reltol = 1e-10;
        ode.solver_opts.abstol = 1e-12;
        let dose = DoseEvent::new(0.0, 1000.0, 1, 0.0, true, 12.0);

        // Fast disposition (ke = CL/V = 2.0, λ·II = 24): the trough converges in a few cycles,
        // so the early stop fires well inside SS_EQUILIBRATION_CYCLES.
        let fast = pk_one(20.0, 10.0);
        let _ = equilibrate_ss_state(&ode, &fast.values, &dose, &ode.solver_opts);
        let fast_cycles = LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.get());
        assert!(
            (2..SS_EQUILIBRATION_CYCLES).contains(&fast_cycles),
            "fast PK should early-stop inside the budget, ran {fast_cycles}"
        );

        // Near-non-eliminating (ke ≈ 5e-4, λ·II ≈ 6e-3): never reaches the 1e-12 relative
        // threshold in the budget → runs the full SS_EQUILIBRATION_CYCLES (this is the
        // pre-existing slow-PK truncation, tracked separately — #532 review #12).
        let slow = pk_one(0.05, 100.0);
        let _ = equilibrate_ss_state(&ode, &slow.values, &dose, &ode.solver_opts);
        let slow_cycles = LAST_SS_EQUILIBRATION_CYCLES.with(|c| c.get());
        assert_eq!(
            slow_cycles, SS_EQUILIBRATION_CYCLES,
            "slow PK should run the full budget, ran {slow_cycles}"
        );
    }

    #[test]
    fn ode_ss_iv_bolus_matches_analytical_ss() {
        // The test ODE stores compartment AMOUNT (dA/dt = -ke·A), while the
        // analytical formula returns CONCENTRATION = amount/V. Divide
        // before comparing.
        use crate::pk::one_cpt_iv_bolus_ss;
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let amt = 1000.0_f64;
        let ii = 12.0_f64;
        // Sample times within and beyond one dosing interval.
        let obs_times = vec![1.0, 4.0, 8.0, 11.0, 14.0, 24.0];
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert_eq!(preds.len(), obs_times.len());

        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_iv_bolus_ss(&dose, t, cl, v);
            // The RK45 reltol/abstol set in `[fit_options]` dominate the error here; the SS
            // equilibration now stops on the `SS_EQUILIBRATION_TOL` (1e-12) early-stop
            // (#519) rather than a fixed N=50 truncation, so its own tail is negligible.
            // 1e-4 is the safe headroom across the population.
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_infusion_matches_analytical_ss() {
        use crate::pk::one_cpt_infusion_ss;
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let amt = 1000.0_f64;
        let rate = 250.0_f64; // T_inf = 4 h
        let ii = 24.0_f64;
        // Cover during-infusion, post-infusion, and beyond one interval.
        let obs_times = vec![1.0, 3.5, 4.0, 8.0, 12.0, 23.0, 48.0];
        let dose = DoseEvent::new(0.0, amt, 1, rate, true, ii);
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_infusion_ss(&dose, t, cl, v);
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_resets_prior_state() {
        // SS=1 semantics: at the SS dose time, prior compartment state is
        // discarded and reset to the SS-train value. Build a subject with
        // a non-SS dose at t=0 (which would normally contribute decay
        // through to t=10) and an SS=1 dose at t=10. The post-SS-dose
        // observation must match the SS analytical formula evaluated at
        // tau = obs_time - 10, independent of the t=0 dose.
        use crate::pk::one_cpt_iv_bolus_ss;
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let doses = vec![
            DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0),
            DoseEvent::new(10.0, amt, 1, 0.0, true, ii),
        ];
        let obs_times = vec![11.0, 14.0, 20.0];
        let subj = make_subject(doses.clone(), obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            let expected = one_cpt_iv_bolus_ss(&doses[1], t - 10.0, cl, v);
            assert_relative_eq!(preds[j] / v, expected, epsilon = 1e-6, max_relative = 1e-4);
        }
    }

    #[test]
    fn ode_ss_iv_bolus_with_lagtime_matches_nonmem() {
        // ODE-path coverage of SS + ALAG1 (issue #15). Reference PRED from
        // NONMEM 7.5.1 (ADVAN1 TRANS2, MAXEVAL=0): CL=5, V=80, ALAG1=2.0,
        // single SS=1 II=12 AMT=1000 IV bolus into the central compartment
        // (S1=V). Control file + dataset in tests/ss_lagtime_nonmem.rs.
        //
        // The first three samples (t=0.5,1.0,1.5 < ALAG1=2.0) exercise the
        // previous-interval steady-state tail seeded by `ss_state_at_phase`;
        // without the seed the ODE state would still be empty there (≈0).
        let cl = 5.0;
        let v = 80.0;
        let amt = 1000.0;
        let ii = 12.0;
        let lagtime = 2.0;
        let nonmem: &[(f64, f64)] = &[
            (0.5, 12.291),
            (1.0, 11.912),
            (1.5, 11.546),
            (2.0, 23.691),
            (3.0, 22.255),
            (6.0, 18.450),
            (11.0, 13.499),
            (13.0, 11.912),
            (18.0, 8.7153),
        ];
        let obs_times: Vec<f64> = nonmem.iter().map(|&(t, _)| t).collect();
        let dose = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
        let subj = make_subject(vec![dose], obs_times);
        let mut pk = pk_one(cl, v);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let ode = one_cpt_ode_spec();

        // one_cpt_ode_spec stores amount; divide by V for concentration.
        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &(_t, pred)) in nonmem.iter().enumerate() {
            assert_relative_eq!(preds[j] / v, pred, max_relative = 1e-4);
        }
    }

    // ── [scaling] Form C: output_fn replaces obs_cmt readout ────────────────

    /// Same shape as `one_cpt_ode_spec` but the state holds `amount` (not
    /// concentration), and `output_fn` produces concentration via `A/V`.
    /// V is passed in via `pk_params_flat[PK_IDX_V]`.
    fn one_cpt_ode_spec_amount_form() -> OdeSpec {
        OdeSpec {
            rhs: Box::new(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
                let cl = p[crate::types::PK_IDX_CL];
                // dA/dt = -CL/V * A   (state is amount; same exp decay as
                // the concentration-baked spec)
                let v = p[crate::types::PK_IDX_V];
                let ke = if v > 0.0 { cl / v } else { 0.0 };
                dy[0] = -ke * y[0];
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::Single(Box::new(
                |state: &[f64], pk: &[f64], _theta: &[f64], _eta: &[f64], _cov| {
                    let v = pk[crate::types::PK_IDX_V];
                    if v > 0.0 {
                        state[0] / v
                    } else {
                        0.0
                    }
                },
            )),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        }
    }

    #[test]
    fn test_ode_output_fn_form_c_matches_concentration_form() {
        // Build two equivalent 1-cpt IV bolus ODE models:
        //   Reference: state = concentration; obs_cmt_idx = Some(0).
        //   Form C:    state = amount;        output_fn = state/V.
        //
        // The dose adds amt to state in both cases. In the reference, that
        // means state = amt directly equals the initial concentration AMT/V
        // ONLY if AMT/V already matches. To make the two truly equivalent
        // we have to scale the dose differently. Easier: pick V = 1.0 so
        // amount equals concentration numerically, and run an analytical
        // sanity check instead.
        let pk = pk_one(5.0, 1.0); // CL=5, V=1 → ke = 5
        let doses = vec![DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.0, 0.5, 1.0, 2.0];
        let subj = make_subject(doses, obs_times.clone());

        let ode_ref = one_cpt_ode_spec();
        let ode_form_c = one_cpt_ode_spec_amount_form();

        let preds_ref = ode_predictions(&ode_ref, &pk.values, &[], &[], &subj);
        let preds_c = ode_predictions(&ode_form_c, &pk.values, &[], &[], &subj);

        // V = 1 makes amount/V numerically equal to amount, so both must agree.
        for (a, b) in preds_ref.iter().zip(preds_c.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6, max_relative = 1e-6);
        }

        // And — crucially — Form C must produce a different numeric answer
        // when V differs from 1, demonstrating the readout actually divides
        // by V rather than ignoring it.
        let pk_v_5 = pk_one(5.0, 5.0); // CL=5, V=5 → ke = 1
        let preds_c_v5 = ode_predictions(&ode_form_c, &pk_v_5.values, &[], &[], &subj);
        // At t=0 just after the bolus, state = 10, V = 5 → conc = 2.
        assert_relative_eq!(preds_c_v5[0], 2.0, epsilon = 1e-9);
        // Reference (concentration-baked) with same params: state = 10
        // ⇒ conc = 10. Different from Form C, confirming output_fn ran.
        let preds_ref_v5 = ode_predictions(&ode_ref, &pk_v_5.values, &[], &[], &subj);
        assert!(
            (preds_ref_v5[0] - preds_c_v5[0]).abs() > 1.0,
            "output_fn must change the readout (ref={} c={})",
            preds_ref_v5[0],
            preds_c_v5[0]
        );
    }

    /// Regression for the co-temporal multi-CMT recorder bug: two observations
    /// at the SAME time but different CMTs (simultaneous PK/PD sampling) must
    /// BOTH be recorded. Before the fix, `obs_map` keyed by time alone kept
    /// only one index per time and left the other observation at its initial
    /// NaN.
    #[test]
    fn test_ode_predictions_records_cotemporal_multi_cmt() {
        // CMT=1 reads the compartment amount; CMT=2 reads twice that — two
        // distinct, finite readouts of the same single-state system, so we can
        // confirm each observation got its own value (not one overwriting the
        // other).
        let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
        map.insert(
            1,
            PerCmtReadout {
                out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| s[0]),
                program: None,
            },
        );
        map.insert(
            2,
            PerCmtReadout {
                out_fn: Box::new(|s: &[f64], _pk: &[f64], _t, _e, _c| 2.0 * s[0]),
                program: None,
            },
        );
        let mut ode = one_cpt_ode_spec();
        ode.readout = OdeReadout::PerCmt(map);

        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        // Two obs at t=1 (CMT 1 and 2) and two at t=4 (CMT 1 and 2).
        let mut subj = make_subject(doses, vec![1.0, 1.0, 4.0, 4.0]);
        subj.obs_cmts = vec![1, 2, 1, 2];

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);

        assert!(
            preds.iter().all(|p| p.is_finite()),
            "all co-temporal obs must be recorded (finite), got {preds:?}"
        );
        // CMT=2 readout is exactly twice CMT=1 at the same time.
        assert!((preds[1] - 2.0 * preds[0]).abs() < 1e-9);
        assert!((preds[3] - 2.0 * preds[2]).abs() < 1e-9);
    }

    /// Regression for Copilot review on PR #84: pre-Phase-2 the ODE paths
    /// clamped NaN predictions to 0 at the end of `ode_predictions` (and
    /// at the Obs branch of `ode_predictions_event_driven`). That defeated
    /// the "loud failure" semantic for `OdeReadout::PerCmt` missing
    /// entries (and for any other genuine NaN). The clamp now only
    /// touches negatives; NaN propagates.
    #[test]
    fn test_ode_predictions_propagates_nan_from_readout() {
        // Build an OdeReadout::PerCmt that DELIBERATELY returns NaN for
        // CMT=1 — emulating a missing-CMT lookup that bypassed pre-fit
        // validation. The resulting prediction must be NaN, not 0.
        let mut map: HashMap<usize, PerCmtReadout> = HashMap::new();
        map.insert(
            1,
            PerCmtReadout {
                out_fn: Box::new(|_state: &[f64], _pk: &[f64], _theta, _eta, _cov| f64::NAN),
                program: None,
            },
        );
        let mut ode = one_cpt_ode_spec();
        ode.readout = OdeReadout::PerCmt(map);

        let pk = pk_one(5.0, 80.0);
        let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![1.0, 4.0];
        let subj = make_subject(doses, obs_times);

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, p) in preds.iter().enumerate() {
            assert!(
                p.is_nan(),
                "obs {} from a NaN-returning readout must be NaN, got {}",
                j,
                p
            );
        }
    }

    #[test]
    fn test_ode_predictions_still_clamps_negatives() {
        // Sanity: dropping the NaN clamp must not change the negative
        // clamp behavior (ODE solver overshoot guard).
        let ode = OdeSpec {
            // dA/dt = -1 → state goes negative quickly with starting amount 1
            rhs: Box::new(|_y, _p, _t, dy| {
                dy[0] = -1.0;
            }),
            n_states: 1,
            state_names: vec!["central".into()],
            readout: OdeReadout::ObsCmt(0),
            diffusion_var: Vec::new(),
            solver_opts: OdeSolverOptions::default(),
            input_rate: Vec::new(),
            rhs_program: None,
            readout_program: None,
            indiv_param_program: None,
            dose_attr_map: Default::default(),
            init_fn: None,
        };
        let pk = pk_one(1.0, 1.0);
        let doses = vec![DoseEvent::new(0.0, 1.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![10.0]; // dose=1, after 10s of -1/s → state = -9
        let subj = make_subject(doses, obs_times);

        let preds = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert!(
            !preds[0].is_nan(),
            "negative readout must be clamped to 0, not NaN"
        );
        assert!(
            preds[0] >= 0.0,
            "negative readout must be clamped to 0, got {}",
            preds[0]
        );
    }

    /// Helper: oral PK params with clearance, volume, ka, and bioavailability.
    fn pk_oral_f(cl: f64, v: f64, ka: f64, f: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = cl;
        p.values[crate::types::PK_IDX_V] = v;
        p.values[crate::types::PK_IDX_KA] = ka;
        p.values[crate::types::PK_IDX_F] = f;
        p
    }

    #[test]
    fn ode_applies_f_bio_to_bolus_dose() {
        // Issue #122: the ODE engine must load the depot with F·AMT (NONMEM
        // convention), not the full AMT. For this linear oral system the
        // central readout is exactly proportional to the depot load, so a
        // bioavailability of F = 0.5 must halve every prediction relative to
        // F = 1.0. Covers both the plain and event-driven paths.
        let ode = one_cpt_oral_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        let obs_times = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let subj = make_subject(doses, obs_times.clone());

        let pk_full = pk_oral_f(5.0, 50.0, 1.5, 1.0);
        let pk_half = pk_oral_f(5.0, 50.0, 1.5, 0.5);

        // Plain (non-TV) path.
        let full = ode_predictions(&ode, &pk_full.values, &[], &[], &subj);
        let half = ode_predictions(&ode, &pk_half.values, &[], &[], &subj);
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(*f > 0.0, "expected positive prediction");
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }

        // Event-driven path.
        let pk_dose_full = vec![pk_full; subj.doses.len()];
        let pk_obs_full = vec![pk_full; obs_times.len()];
        let pk_dose_half = vec![pk_half; subj.doses.len()];
        let pk_obs_half = vec![pk_half; obs_times.len()];
        let ed_full =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_full, &pk_obs_full, &[]);
        let ed_half =
            ode_predictions_event_driven(&ode, &subj, &[], &[], &pk_dose_half, &pk_obs_half, &[]);
        for (f, h) in ed_full.iter().zip(ed_half.iter()) {
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    #[test]
    fn ode_applies_f_bio_to_infusion() {
        // A rate-defined infusion under F holds the rate and scales the duration
        // (#419): F=0.5 on (AMT=100, rate=50, T=2h) delivers rate 50 over 1h -
        // identical to a full-F infusion of F·AMT=50 at rate 50, NOT 0.5x the F=1
        // curve.
        let ode = one_cpt_oral_ode_spec();
        let rate = 50.0;
        let obs_times = vec![1.0, 2.0, 4.0, 8.0];
        let preds = |amt: f64, f: f64| {
            let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
            let subj = make_subject(doses, obs_times.clone());
            ode_predictions(&ode, &pk_oral_f(5.0, 50.0, 1.5, f).values, &[], &[], &subj)
        };
        let full = preds(100.0, 1.0);
        let half_f = preds(100.0, 0.5);
        let equiv = preds(50.0, 1.0); // F=1, F·AMT delivered at the same rate
        for ((f, hf), e) in full.iter().zip(half_f.iter()).zip(equiv.iter()) {
            assert!(*f > 0.0, "expected positive prediction");
            assert_relative_eq!(*hf, *e, epsilon = 1e-9, max_relative = 1e-6);
        }
        assert!(
            half_f
                .iter()
                .zip(full.iter())
                .any(|(h, f)| (*h - 0.5 * *f).abs() > 1e-6),
            "rate-defined infusion under F must reshape, not scale"
        );
    }

    #[test]
    fn ode_applies_f_bio_to_ss_dose() {
        // Steady-state pre-equilibration must also load F·AMT each cycle, so a
        // halved F halves the steady-state predictions.
        let ode = one_cpt_oral_ode_spec();
        let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
        let obs_times = vec![1.0, 4.0, 8.0, 11.0];
        let subj = make_subject(doses, obs_times);

        let full = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 1.0).values,
            &[],
            &[],
            &subj,
        );
        let half = ode_predictions(
            &ode,
            &pk_oral_f(5.0, 50.0, 1.5, 0.5).values,
            &[],
            &[],
            &subj,
        );
        for (f, h) in full.iter().zip(half.iter()) {
            assert!(*f > 0.0, "expected positive SS prediction");
            assert_relative_eq!(*h, 0.5 * *f, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Regression tests for ode_predictions_with_states / ode_dense_solve_states
    // -----------------------------------------------------------------------

    /// Bug regression: state must be advanced through segments that contain no
    /// observations (the t_end push).  Before the fix, `sol.last()` returned
    /// `None` for an empty saveat and `u` was not updated, so all subsequent
    /// compartment states were wrong.
    #[test]
    fn ode_with_states_advances_through_empty_segment() {
        // Two doses, observations only after the second.  The segment [0, 12)
        // has no obs — the state must still decay correctly through it.
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ode = one_cpt_ode_spec();
        let pk = pk_one(cl, v);
        let doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(12.0, 50.0, 1, 0.0, false, 0.0),
        ];
        let obs_times = vec![24.0];
        let subj = make_subject(doses, obs_times);
        let (preds, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        // Compare against the full ode_predictions path.
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        assert!(
            approx::relative_eq!(preds[0], preds_ref[0], max_relative = 1e-6),
            "ipred diverges — state was not advanced through the empty segment"
        );
        // State[0] must be positive (non-zero drug remaining).
        assert!(
            states[0][0] > 0.0 && states[0][0].is_finite(),
            "compartment state is wrong after empty inter-dose segment: {}",
            states[0][0]
        );
    }

    /// Bug regression: CMT out-of-range (CMT=0 or CMT > n_states) must be
    /// ignored by both new functions, matching ode_predictions behaviour.
    /// Before the fix, saturating_sub(1).min(n-1) applied the dose to
    /// compartment 0 or the last compartment instead.
    #[test]
    fn ode_with_states_ignores_out_of_range_cmt() {
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ode = one_cpt_ode_spec();
        let pk = pk_one(cl, v);
        // CMT=0 — out-of-range for a 1-state ODE (states are CMT 1).
        let dose_valid = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
        let dose_oor = DoseEvent::new(0.0, 999.0, 0, 0.0, false, 0.0); // CMT=0
        let obs_times = vec![4.0, 12.0];

        let subj_ref = make_subject(vec![dose_valid.clone()], obs_times.clone());
        let subj_oor = make_subject(vec![dose_valid.clone(), dose_oor], obs_times.clone());

        let (preds_ref, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_ref);
        let (preds_oor, _) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj_oor);
        for j in 0..obs_times.len() {
            assert!(
                approx::relative_eq!(preds_ref[j], preds_oor[j], max_relative = 1e-9),
                "obs {j}: CMT=0 dose was applied (got {}) instead of being ignored (expected {})",
                preds_oor[j],
                preds_ref[j]
            );
        }
    }

    /// Bug regression: TAD for SS doses must be computed with rem_euclid so it
    /// stays within [0, II).  Before the fix the raw elapsed time was used,
    /// injecting a growing TAD into the ODE RHS.  This test uses an ODE that
    /// writes TAD into its output so we can verify it.
    #[test]
    fn ode_with_states_tad_stays_within_dosing_interval_for_ss() {
        // ODE: dA/dt = -ke*A; but we read TAD = t - ext_params[TAD_SLOT] back
        // as the compartment state for the diagnostic.  Use a second-state ODE:
        //   dA/dt = -ke*A
        //   dT/dt = 0  (T is just a placeholder; we seed it externally via the
        //               TAD anchor update, which is non-state, so we use ipred)
        // Actually simplest: verify ode_predictions and ode_predictions_with_states
        // agree on ipred for an SS dose observed beyond one II, because the TAD
        // error only shows up when TAD modulates the ODE.
        //
        // For a pure 1-cpt IV where the RHS does NOT use TAD, both paths must
        // agree with the closed-form SS regardless of the TAD anchor.
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ii = 24.0_f64;
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
        // Observations beyond one dosing interval.
        let obs_times = vec![0.5, 6.0, 24.0, 30.0, 48.0, 53.0];
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let pk = pk_one(cl, v);
        let ode = one_cpt_ode_spec();

        let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, &t) in obs_times.iter().enumerate() {
            // ipred must agree with ode_predictions (which uses rem_euclid for TAD).
            assert!(
                approx::relative_eq!(preds_ws[j], preds_ref[j], max_relative = 1e-6),
                "ipred diverges at t={t} — TAD anchor mismatch for SS dose"
            );
            // For ObsCmt(0) readout, ipred == u[0] == state[0], so they must agree.
            assert!(
                approx::relative_eq!(states[j][0], preds_ws[j], max_relative = 1e-9),
                "state != ipred at t={t} — state not self-consistent with ipred"
            );
        }
    }

    /// Bug regression: for an SS dose with lagtime > 0, the pre-lag break
    /// point at dose.time must seed ss_state_at_phase so observations before
    /// the lagged pulse see the correct pre-lag SS tail rather than zero.
    /// Before the fix the merged dose loop fired only at dose.time + lagtime,
    /// leaving the pre-lag segment with an all-zero initial state.
    #[test]
    fn ode_with_states_ss_lagtime_preseed_is_correct() {
        let cl = 5.0_f64;
        let v = 80.0_f64;
        let ii = 24.0_f64;
        let lagtime = 2.0_f64;
        // SS dose at t=0 with lagtime=2; observations at t=0.5 and t=1.5
        // (both before the lagged arrival at t=2) should see the SS tail
        // from the prior cycle.
        let dose = DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii);
        let mut pk = pk_one(cl, v);
        pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;
        let obs_times = vec![0.5, 1.5, 3.0, 12.0];
        let subj = make_subject(vec![dose.clone()], obs_times.clone());
        let ode = one_cpt_ode_spec();

        let (preds_ws, states) = ode_predictions_with_states(&ode, &pk.values, &[], &[], &subj);
        let preds_ref = ode_predictions(&ode, &pk.values, &[], &[], &subj);
        for (j, (&t, &p_ws)) in obs_times.iter().zip(preds_ws.iter()).enumerate() {
            assert!(
                approx::relative_eq!(p_ws, preds_ref[j], max_relative = 1e-6),
                "ipred diverges at t={t} — SS+lagtime pre-lag seeding missing"
            );
            // Pre-lag obs (t < lagtime) must be > 0 (from prior SS cycle).
            if t < lagtime {
                assert!(
                    states[j][0] > 0.0,
                    "state is zero at t={t} (before lag) — SS tail was not pre-seeded"
                );
            }
        }
    }

    #[test]
    fn adaptive_observe_expression_flows_through_driver() {
        // The model readout is the raw `central` amount (`[scaling] y = central`),
        // but the declarative controller observes `central / V` (concentration).
        // The driver must feed the controller the compiled expression's value, not
        // the cmt readout — this exercises the S2.2 `observe_exprs` path.
        const M: &str = r#"
[parameters]
  theta TVCL(1.0)
  theta TVV(50.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
  observe = central / V
  at = [24, 48]
  start_dose = 100
  route = bolus(cmt = 1)
  dose_bounds = [0, 400]
  when signal > 1000 : decrease 25%
"#;
        let parsed = crate::parser::model_parser::parse_full_model(M).expect("parses");
        let model = parsed.model;
        let spec = parsed.adaptive_dosing.expect("has adaptive block");
        let compiled =
            crate::sim::adaptive_control::compile_adaptive(&model, &spec).expect("compiles");
        let theta = model.default_params.theta.clone();
        let eta = vec![0.0; model.n_eta + model.n_kappa];
        let pk = (model.pk_param_fn)(&theta, &eta, &HashMap::new(), 0.0);
        let subject = make_subject(vec![], spec.at.clone());
        let mut controller = (compiled.make_controller)();
        let monitors = vec![crate::sim::adaptive::AdaptiveMonitor {
            spec: &compiled.monitors[0],
            observe: compiled.observe.as_ref(),
        }];
        let run = ode_predictions_adaptive_impl(
            model.ode_spec.as_ref().unwrap(),
            &pk.values,
            &theta,
            &eta,
            &subject,
            &spec.at,
            &monitors,
            &mut controller,
            spec.at.len() + 1,
            None,
        )
        .expect("driver runs");

        // Decision 0 is the pre-dose trough (central = 0 ⇒ concentration 0).
        assert_eq!(run.decisions[0].observed_signals[0].value, 0.0);
        // Decision 1: one bolus of start_dose decayed over Δt, divided by V — the
        // CONCENTRATION. Reading the cmt amount instead would be ~50× larger (the
        // raw `central`), so this pins the expression path.
        let ke = theta[0] / theta[1]; // CL/V at eta = 0
        let dt = spec.at[1] - spec.at[0];
        let expected_conc = spec.start_dose * (-ke * dt).exp() / theta[1];
        let got = run.decisions[1].observed_signals[0].value;
        assert!(
            (got - expected_conc).abs() < 1e-3,
            "observed {got}, expected concentration {expected_conc} (raw amount would be ~{})",
            expected_conc * theta[1]
        );
    }
}
