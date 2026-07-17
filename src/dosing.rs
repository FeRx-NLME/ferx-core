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
