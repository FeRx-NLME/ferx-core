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

// Dose resolution + SS-equilibration primitives moved to `crate::dosing` (a neutral
// leaf module) so pk/sens/api don't depend upward on ode/. A PRIVATE import (NOT a
// `pub(crate) use` re-export) so these do not leak back out as `crate::ode::…` — the
// upward dependency this move removed stays removed. The ode-internal resolve callers
// + `equilibrate_ss_state` use the bare names; the `#[cfg(test)] mod tests` picks them
// up via `use super::*`. Test-only symbols (`ss_cycle_converged`, `SS_EQUILIBRATION_TOL`,
// `last_ss_equilibration_cycles`, `with_full_ss_equilibration`) are referenced directly
// as `crate::dosing::…` by the tests, so they are not imported here.
use crate::dosing::{
    note_ss_nonconvergence_if_capped, record_ss_equilibration_cycles, resolve_subject_doses,
    resolve_subject_doses_with, SsStopTracker, SS_EQUILIBRATION_CYCLES,
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
/// `n_states + 3` ODE solves ([`equilibrate_ss_input_rate_fixed_point_g`]). For a **nonlinear**
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
    prepared: &[PreparedInputRate],
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
    let local_ss = [DoseEvent::new(0.0, dose.amt, dose.cmt, 0.0, true, ii)];
    let local_f_bio = [f_bio];
    let no_lag: [f64; 0] = [];
    let no_zero: [(usize, f64); 0] = [];
    let forced_rhs = wrap_rhs_with_forcings(
        ode,
        &local_ss,
        &no_lag,
        &local_f_bio,
        f64::NEG_INFINITY,
        prepared,
        InfusionInput::Spanning(Vec::new()),
        &no_zero,
    );

    // Advance a state one cycle `[0, II]` under `rhs`. The equilibration integrates at a tightened
    // tolerance (`ss_equilibration_opts`) so the fixed-point trough is accurate even when `ρ → 1`
    // amplifies the per-cycle solver noise — the forward walk keeps the model tolerance.
    let eq_opts = ss_equilibration_opts(opts);
    let advance = |rhs: &dyn Fn(&[f64], &[f64], f64, &mut [f64]), u0: &[f64]| -> Option<Vec<f64>> {
        solve_ode(rhs, u0, (0.0, ii), pk_params_flat, &[ii], &eq_opts)
            .last()
            .map(|p| p.u.clone())
    };
    equilibrate_ss_input_rate_g::<f64, _, _>(
        n,
        ii,
        eq_opts.reltol,
        eq_opts.abstol,
        |u0| advance(ode.rhs.as_ref(), u0),
        |u0| advance(&forced_rhs, u0),
    )
}

/// Numeric-generic core of [`equilibrate_ss_input_rate`], shared by the production
/// `f64` predictor and the analytic-sensitivity walk (`T = Dual1/Dual2`, #835) so the fixed-point
/// formula lives in exactly one place.
///
/// The ODE advancement is *injected* as two "one cycle from `u0`" closures — `advance_unforced`
/// (disposition alone; builds the drift `z0` and the propagator columns of `M = e^{A·II}`) and
/// `advance_forced` (disposition + periodic `R_in`; builds `b` and drives the linearity check).
/// Each caller keeps its own solver and forcing assembly (production `solve_ode` +
/// `wrap_rhs_with_forcings`; the dual walk `solve_ode_g::<T>` + `add_prepared_input_rate_forcing`),
/// while this body owns only the `T`-typed algebra: the `I − M` assembly, the
/// [`solve_linear_system_g`](crate::sens::linsolve::solve_linear_system_g) solve of
/// `u_ss = (I − M)⁻¹·b`, and the value-part linearity verification. Run over a dual `T` it thus
/// carries `∂u_ss/∂(θ,η)` (and the 2nd order) through the solve automatically — the implicit-
/// function derivative, with no hand-assembled `dM`/`db`.
///
/// Returns `None` — the caller ([`equilibrate_ss_input_rate_g`]) then tries the Anderson solve —
/// on a **nonlinear** disposition (the one-/two-cycle self-check fails), a singular `I − M`, or any
/// non-finite intermediate. (Only when Anderson *also* declines does the pulse-train fallback run.)
pub(crate) fn equilibrate_ss_input_rate_fixed_point_g<T, FUnf, FFor>(
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
    // b: forced response over one cycle from a zero state (constant drift + periodic R_in).
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
    // u_ss (only an affine one-cycle map does). The tolerance is scaled to the solver's own
    // accuracy so the fast path is not falsely abandoned at a loose `ode_reltol`; a genuinely
    // nonlinear map diverges by O(scale) and is rejected. Two cycles (not one) rejects a weakly-
    // nonlinear map whose true fixed point merely happens to sit within `vtol` after the first.
    let scale = u_ss.iter().fold(1e-12_f64, |a, x| a.max(x.val().abs()));
    // Verification tolerance, scaled to the solver's own accuracy. The one-cycle residual of a
    // genuinely *linear* map is not zero but a solver-noise floor: `b` (forced from a zero state)
    // and the check below (forced from `u_ss`) integrate the same periodic `R_in` over *different*
    // adaptive step sequences, so their forcing quadratures differ by O(reltol·scale). Empirically
    // that floor is ≈ 45·reltol (relative) — above the original `32·reltol`, which therefore
    // falsely declined even linear models at a tight `ode_reltol` (e.g. 1e-10), silently forcing
    // the fallback iteration. `256·reltol` clears the floor with ~5× margin while still rejecting a
    // genuinely nonlinear map, whose one-cycle residual is O(scale) — ~1e8·reltol, eight orders
    // above this bound (see `ss_input_rate_nonlinear_disposition_falls_back_to_iteration`).
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
    let diverged_ceiling = 1e8 * seed_mag;
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
/// **linear** closed form [`equilibrate_ss_input_rate_fixed_point_g`] first (exact, one linear
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
    // `&F` still implements `Fn` when `F: Fn`, so borrowing lets the linear attempt and the
    // Anderson fallback share the same two closures without moving them.
    if let Some(u_ss) = equilibrate_ss_input_rate_fixed_point_g::<T, _, _>(
        n,
        ii,
        reltol,
        abstol,
        &advance_unforced,
        &advance_forced,
    ) {
        record_ss_equilibration_cycles(1);
        return Some(u_ss);
    }
    anderson_ss_fixed_point_g::<T, _>(n, ii, reltol, abstol, &advance_forced)
}

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

    if dose.ii <= 0.0 {
        return u;
    }
    // `CMT=0` equilibrates compartment 1 — NONMEM's default dose compartment —
    // like every other dose site (#899). This used to bail out here and return
    // an unequilibrated all-zero state, so an `SS=1` dose written `CMT=0`
    // produced the *single-dose* curve on the ODE engine while the analytical
    // engine (fixed in #375) produced the accumulated steady state. That was the
    // cross-engine disagreement recorded in `CHANGELOG.md`; this closes it.
    let cmt_idx = dose.cmt.saturating_sub(1);
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
    if !is_inf && input_rate_consumes_cmt(ode, dose.cmt) {
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
            .map(|m| DoseEvent::new(m as f64 * dose.ii, dose.amt, dose.cmt, 0.0, false, 0.0))
            .collect();
        let local_f_bio = vec![f_bio; n_pulses];
        let no_lag: [f64; 0] = [];
        let no_zero_order: [(usize, f64); 0] = [];
        let wrapped = wrap_rhs_with_forcings(
            ode,
            &local_doses,
            &no_lag,
            &local_f_bio,
            f64::NEG_INFINITY,
            &prepared,
            InfusionInput::Spanning(Vec::new()),
            &no_zero_order,
        );
        let mut tracker = SsStopTracker::default();
        let mut cycles_run = 0usize;
        let mut early_stopped = false;
        for m in 0..n_pulses {
            let seg_start = m as f64 * dose.ii;
            let seg_end = seg_start + dose.ii;
            let sol = solve_ode(
                &wrapped,
                &u,
                (seg_start, seg_end),
                pk_params_flat,
                &[seg_end],
                opts,
            );
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
        // If the pulse train hit the cycle cap without converging, the returned trough may be
        // materially below the true periodic steady state — surface a warning instead of silently
        // under-reporting it (#867). Only a *nonlinear* disposition reaches this fallback (the
        // linear closed form above returns early), so this is exactly the saturable
        // heavy-accumulation case; a fast-contracting model early-stops and is left alone.
        let (incr_prev, incr_last, incr_mag) = tracker.recent_increments();
        note_ss_nonconvergence_if_capped(early_stopped, incr_prev, incr_last, incr_mag);
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
    input_rate: &[crate::pk::absorption::InputRateForcing],
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
            // Infusion into a built-in absorption compartment (#719 gap 2): the dose is a
            // zero-order source *feeding the kernel*, delivered through the convolved input rate
            // `R_in_inf` (`add_prepared_input_rate_forcing`), NOT injected directly. Suppress the
            // plain `+rate` here so the mass is not double-counted. (`input_rate` is empty on the
            // EKF path and on models with no built-in absorption, so this is then a no-op.)
            if input_rate.iter().any(|f| f.cmt == d.cmt.saturating_sub(1)) {
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

/// Push a break at every per-route absorption onset `d.time + lag_cmt + lag_route`
/// (`fn(..., lag=L)`) — for each input-rate forcing carrying a `lag_slot`, over every
/// positive-amount dose feeding that forcing's compartment. `route_lag_of` reads the
/// forcing's lag from the caller's PK snapshot (a single subject snapshot on the dense
/// path; per-forcing over the event-driven walk's snapshot). A route lag delays that
/// route's onset PAST the dose's `d.time + lag_cmt` break, so without this break the
/// smooth routes' onset kink is unresolved and a lagged `zero_order` window's start is
/// unbracketed (never fully contained in a segment → no mass delivered). A no-op when
/// no forcing carries a lag (the `filter` yields nothing), so the common case is free.
fn push_route_lag_break_times(
    break_times: &mut Vec<f64>,
    ode: &OdeSpec,
    subject: &Subject,
    dose_lagtimes: &[f64],
    route_lag_of: impl Fn(&crate::pk::absorption::InputRateForcing) -> f64,
) {
    for forcing in ode.input_rate.iter().filter(|f| f.lag_slot.is_some()) {
        let route_lag = route_lag_of(forcing);
        for (k, d) in subject.doses.iter().enumerate() {
            if d.amt > 0.0 && d.cmt.saturating_sub(1) == forcing.cmt {
                break_times.push(d.time + dose_lagtimes.get(k).copied().unwrap_or(0.0) + route_lag);
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
            // Both arms are unreachable from a validated call since #899:
            // `check_dose_compartments` rejects an infusion with `CMT=0` (the
            // default dose compartment is defined for a bolus, not for a
            // zero-order input — the analytical engine has rejected it since
            // #375) and rejects `cmt > n_states` on ODE models. Kept because
            // `gated_infusions` is also reachable from hand-built `OdeSpec`s
            // that run no validation; dropping an infusion is preferable to
            // panicking inside the integration loop.
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
        // Per-route absorption delay (`fn(..., lag=L)`): an offset ON TOP of the
        // dose's compartment lag, so each parallel / mixed pathway can switch on at
        // its own time. Dose-invariant (a property of the forcing, not the dose), so
        // hoisted out of the per-dose loop; `0` for an unlagged forcing (the common
        // case), a no-op there. Since #859 a `first_order` per-route lag is analytic on
        // the `Dual2` event-driven walk: this continuous `∂R_in/∂lag_route` shift flows
        // here, and the onset discontinuity is supplied separately as the `K_ROUTE_ONSET`
        // rate-on saltation. Other kernels' route lags stay FD-gated (`zero_order`/
        // `transit`/`igd` pending their slices; `weibull`'s divergent onset permanently).
        let route_lag = forcing.route_lag(params);
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
            let t_eff = T::from_f64(d.time) + lag + route_lag;
            // Doses delivered before the most recent reset are off — the reset
            // zeroed the compartments, same rule as `active_infusions`.
            if t_eff.val() < reset_floor - INFUSION_EPS {
                continue;
            }
            let tad = T::from_f64(t) - t_eff;
            let dose_mass =
                dose_f_bio.get(k).copied().unwrap_or(T::from_f64(1.0)) * T::from_f64(d.amt);
            if d.ss && d.ii > 0.0 {
                acc = acc + ss_periodic_forcing(prep, tad, T::from_f64(d.ii), dose_mass);
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
                acc = acc + prep.rate_infused(tad, dose_mass, T::from_f64(window));
            } else {
                if tad.val() <= 0.0 {
                    continue;
                }
                acc = acc + prep.rate(tad, dose_mass);
            }
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
        predictions[obs_idx] =
            read_observable(ode, u, pk, theta, eta, subject.obs_cov(obs_idx), cmt);
        if let Some(states) = states.as_deref_mut() {
            states[obs_idx] = u.to_vec();
        }
    }
}

/// Clamp negative predictions to zero (ODE solver overshoot guard) — the shared
/// epilogue of the dense drivers. NaN is intentionally NOT clamped (it survives
/// `< 0.0` per IEEE 754) so it propagates to a NaN OFV.
#[inline]
fn clamp_negative_predictions(predictions: &mut [f64]) {
    for p in predictions.iter_mut() {
        if *p < 0.0 {
            *p = 0.0;
        }
    }
}

/// TAD anchor for `ext_params[MAX_PK_PARAMS + 1]`: the last effective dose time at
/// or before `t_start`, SS-aware (`rem_euclid` wraps the elapsed time back into
/// `[0, II)` so TAD stays within one dosing interval). Returns NaN when no
/// effective prior dose exists, so the ODE RHS injects NaN for TAD (matching the
/// sdtab convention) rather than `+∞`.
#[inline]
fn tad_anchor(subject: &Subject, dose_lagtimes: &[f64], t_start: f64) -> f64 {
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
    (dose_lagtimes, dose_f_bio)
}

/// Earliest dose record time, or `+∞` when the subject has no doses.
#[inline]
fn earliest_dose_time(subject: &Subject) -> f64 {
    subject
        .doses
        .iter()
        .map(|d| d.time)
        .fold(f64::INFINITY, f64::min)
}

/// Seed the extended-parameter array for the ODE RHS: slots `0..MAX_PK_PARAMS`
/// hold the PK snapshot; slot `MAX_PK_PARAMS` carries the TAFD anchor (the first
/// dose time, NaN when there are no doses so the RHS injects NaN rather than `-∞`);
/// slot `MAX_PK_PARAMS + 1` (TAD) is left NaN for the per-segment update.
#[inline]
fn seed_ext_params(
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

    /// True when any built-in absorption input-rate forcing carries a per-route
    /// lag (`fn(..., lag=L)`, #859). The single predicate shared by the sensitivity
    /// gates (`ode_analytical_supported`'s subject variants), the IOV gate
    /// (`ode_iov_supported`), the event-driven walk (`integrate_tvcov_g`), the
    /// initial-point diagnostics (`api::check_absorption_dosing`), and
    /// [`crate::types::CompiledModel::has_route_absorption_lag`] — so the "does this
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
    ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor(subject, dose_lagtimes, t_start);

    // Integrate. If any infusions are active in this segment, wrap
    // the user RHS so it adds `+rate` to each infusion's compartment.
    // The plain (non-event-driven) ODE path never sees reset subjects —
    // the dispatcher routes those to `ode_predictions_event_driven` — so
    // no reset floor applies here.
    let active = active_infusions(
        &ode.input_rate,
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
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    // Extended params: slots 0..MAX_PK_PARAMS hold the PK parameters; slots
    // MAX_PK_PARAMS and MAX_PK_PARAMS+1 carry TAFD/TAD anchors for the ODE RHS.
    let first_dose_time = earliest_dose_time(subject);
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
    // Per-route absorption lag (`fn(..., lag=L)`): a route with its own lag switches
    // on at `d.time + lag_cmt + lag_route`, PAST the dose's `d.time + lag_cmt` break
    // above — so add a break at each route onset. This resolves the smooth routes'
    // onset kink (`R_in` jumping 0 → `ka·dose` etc.) exactly, and brackets a lagged
    // `zero_order` window's START (its end is broken via the route-lag-shifted
    // `zo_windows` below). Without it the adaptive solver smears the kink and a lagged
    // zero-order window is never fully contained in a segment (delivering nothing).
    // A no-op for unlagged forcings (`lag_slot = None`), the common case.
    push_route_lag_break_times(&mut break_times, ode, subject, &dose_lagtimes, |f| {
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

    // Walk every break as a left boundary — bound `0..len`, not the old `0..len-1`
    // (#731) — so a dose / observation / CHZ landing on the final break is applied and
    // read post-dose, matching the reactive driver (`ode_predictions_adaptive_impl`)
    // and the per-event replay (`adaptive_frozen_replay_tv`). `break_times` always
    // holds the integration start, so even a degenerate single-instant timeline is a
    // 1-element vector the loop runs once (recording the record from the initial
    // post-dose state, integration skipped by the `k + 1 < len` guard below). The
    // timeline is deduped at 1e-15, so no two adjacent breaks are equal and no break is
    // ever visited twice.
    for k in 0..break_times.len() {
        let t_start = break_times[k];

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
                //
                // `saturating_sub` maps `CMT=0` to compartment 1, NONMEM's
                // default dose compartment, matching the analytical engine and
                // the other ODE drivers (#899). It was a bare `- 1` until then,
                // which underflowed on `CMT=0`: a debug build panicked with
                // "attempt to subtract with overflow" and a release build
                // wrapped to `usize::MAX`, failed the bound below, and dropped
                // the dose in silence.
                let cmt_idx = dose.cmt.saturating_sub(1);
                // Unreachable from a validated call — `check_dose_compartments`
                // rejects `cmt > n_states` on ODE models too since #899. Kept as
                // a bound because `ode_predictions*` is also reachable from
                // hand-built `OdeSpec`s (EKF, unit tests) that run no validation.
                if cmt_idx < n {
                    u[cmt_idx] += dose_f_bio[i] * dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            record_observations(
                ode,
                obs_idxs,
                &u,
                pk_params_flat,
                theta,
                eta,
                subject,
                &mut predictions,
                None,
            );
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
    }

    // Clamp negative predictions to zero (ODE solver overshoot guard).
    // NaN intentionally NOT clamped — it propagates to a NaN OFV so the
    // outer optimizer rejects the step, matching the analytical path's
    // `conc.max(0.0)` semantic (NaN survives `.max(0.0)` per IEEE 754).
    // This is also what surfaces a missing `OdeReadout::PerCmt` entry as
    // a loud failure rather than a silent zero. (Pre-Phase-2 the clamp
    // included NaN; Copilot's review of #84 caught the inconsistency.)
    clamp_negative_predictions(&mut predictions);

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

/// PK snapshot governing a segment boundary at time `t`, for the per-event
/// (time-varying-covariate / `TIME`-built-in) adaptive path (#700). Mirrors the
/// NONMEM end-of-interval convention of [`ode_predictions_event_driven`]: a real
/// record — an observation or EVID=2 (pk-only) row — at `t` contributes its own
/// per-event snapshot; any other break (a decision-only time or an infusion end,
/// neither a data record) carries the previous record's PK forward (LOCF). Shared
/// by the reactive driver and its frozen-replay static engine so the two resolve
/// PK identically and stay bit-aligned.
fn segment_pk_at(
    t: f64,
    obs_map: &HashMap<u64, Vec<usize>>,
    pk_only_map: &HashMap<u64, usize>,
    event_pk: &crate::pk::EventPkParams,
    last_pk: PkParams,
) -> PkParams {
    if let Some(&j) = obs_map.get(&t.to_bits()).and_then(|idxs| idxs.first()) {
        return event_pk.obs[j];
    }
    if let Some(&m) = pk_only_map.get(&t.to_bits()) {
        return event_pk.pk_only[m];
    }
    last_pk
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

/// LOCF covariate map governing a **decision** at time `t` on the per-event
/// (time-varying) adaptive path (#700). The covariate of the most-recent obs /
/// pk-only record at or before `t`, mirroring the LOCF PK carry (`pk_readout`), so
/// a monitored signal (or a `[scaling]` / `observe` expression) that references a
/// time-varying covariate *directly* — not only through a resolved PK parameter —
/// sees the covariate active at the decision, not the frozen t=0 baseline. An obs
/// record wins a tie with a pk-only record at the same time (matching
/// [`segment_pk_at`]). Falls back to `baseline` only for a decision that precedes
/// the first record. The constant-covariate path never calls this.
///
/// Also called by `run_adaptive_population` to resolve the covariate for each IOV
/// `decision_pk[g]` snapshot (#701), so the precomputed decision PK and this driver's
/// live `decision_cov` share **one** covariate rule and cannot silently diverge — the
/// reason this is `pub(crate)` rather than private.
pub(crate) fn locf_decision_cov<'a>(
    t: f64,
    subject: &'a Subject,
    baseline: &'a HashMap<String, f64>,
) -> &'a HashMap<String, f64> {
    let mut best: Option<(f64, &'a HashMap<String, f64>)> = None;
    for (j, &rt) in subject.obs_times.iter().enumerate() {
        if rt <= t + 1e-12 && best.map_or(true, |(bt, _)| rt >= bt) {
            if let Some(cov) = subject.obs_covariates.get(j) {
                best = Some((rt, cov));
            }
        }
    }
    for (m, &rt) in subject.pk_only_times.iter().enumerate() {
        // Strict `>` so an obs record at the same time keeps priority.
        if rt <= t + 1e-12 && best.map_or(true, |(bt, _)| rt > bt) {
            if let Some(cov) = subject.pk_only_covariates.get(m) {
                best = Some((rt, cov));
            }
        }
    }
    best.map(|(_, c)| c).unwrap_or(baseline)
}

/// Occasion (decision window) governing the segment ending at `t` — the IOV twin of
/// [`segment_pk_at`] (#701). Deliberately the **same** end-of-interval / LOCF
/// structure, so the per-window eta threaded into `integrate_segment` always agrees
/// with the occasion of the PK `segment_pk_at` returns for the same `t`: the
/// record-at-`t`'s occasion (obs / pk-only), else the carried-forward `last_occ`.
/// `None` = the baseline window (before the first decision), whose κ is zero.
fn segment_occ_at(
    t: f64,
    obs_map: &HashMap<u64, Vec<usize>>,
    obs_occ: &[Option<usize>],
    pk_only_map: &HashMap<u64, usize>,
    pk_only_occ: &[Option<usize>],
    last_occ: Option<usize>,
) -> Option<usize> {
    if let Some(&j) = obs_map.get(&t.to_bits()).and_then(|idxs| idxs.first()) {
        return obs_occ[j];
    }
    if let Some(&m) = pk_only_map.get(&t.to_bits()) {
        return pk_only_occ[m];
    }
    last_occ
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
    // threads through every segment). The base subject is dose-free, so `event_pk.dose`
    // is unused; controller-injected doses take their PK from the carried-forward
    // (LOCF) snapshot at injection time.
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

    // Shadow subject accumulates the controller's realized doses (the #324
    // pattern); `integrate_segment` reads `shadow.doses` for the TAD anchor.
    let mut shadow = subject.clone();
    // Bioavailability `F` captured at each injected dose's injection time (from the
    // LOCF PK there), parallel to `shadow.doses`. A delivered dose's F is fixed when
    // it is given — later covariate drift must not retroactively rescale it — so
    // segments read F from here rather than re-resolving from the segment PK. On the
    // constant path this equals `f_bio(cmt, pk_params_flat)` for every dose, so the
    // per-segment infusion window is byte-identical to before.
    let mut injected_f: Vec<f64> = Vec::new();

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
    // Time -> pk-only (EVID=2) event index, for the per-event PK resolver; empty
    // on the constant path.
    let mut pk_only_map: HashMap<u64, usize> = HashMap::new();
    if tv {
        for (m, &t) in subject.pk_only_times.iter().enumerate() {
            pk_only_map.entry(t.to_bits()).or_insert(m);
        }
    }

    // Per-record occasion (decision window) for the IOV path (#701), parallel to
    // `obs_times` / `pk_only_times`; empty on the non-IOV path. Resolved from the
    // decision schedule exactly as the occasion-aware `event_pk` was built in
    // `run_adaptive_population`, so `segment_occ_at` and the precomputed PK snapshots
    // agree on each record's occasion.
    let (obs_occ, pk_only_occ): (Vec<Option<usize>>, Vec<Option<usize>>) = if iov {
        (
            subject
                .obs_times
                .iter()
                .map(|&t| crate::pk::occasion_of(decision_times, t))
                .collect(),
            subject
                .pk_only_times
                .iter()
                .map(|&t| crate::pk::occasion_of(decision_times, t))
                .collect(),
        )
    } else {
        (Vec::new(), Vec::new())
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
    let mut break_times: Vec<f64> = vec![0.0, t_last];
    break_times.extend(decision_times.iter().cloned());
    if tv {
        break_times.extend(shadow.obs_times.iter().cloned());
        break_times.extend(shadow.pk_only_times.iter().cloned());
    }
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // #700 review guard: the tolerance dedup above can merge two break times within
    // 1e-15 that are not bit-identical, but `segment_pk_at` / `decision_index_of` /
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

    let mut stopped = false;

    let mut k = 0;
    while k < break_times.len() {
        let t_start = break_times[k];

        // #701 (IOV): a decision break opens its occasion window `g`. Set the LOCF PK
        // + occasion to occasion g's snapshot so the pre-dose readout, the injected
        // dose's F, and any following non-record segment all use occasion g's
        // parameters — not the previous occasion carried in `last_pk`/`last_occ`.
        // Unconditional (every decision break, including holds and post-`Stop`, so
        // every event reads its window's κ). `segment_pk_at`/`segment_occ_at` below
        // then return this snapshot for a decision that is not itself a record.
        if let (Some(dp), Some(&g)) = (decision_pk, decision_index_of.get(&t_start.to_bits())) {
            last_pk = dp[g];
            last_occ = Some(g);
        }

        // PK snapshot in effect at this break's left boundary `t_start`: the record
        // there (obs / pk-only) or the LOCF carry-forward (`last_pk`) on the TV path
        // (#700), the frozen `pk_params_flat` on the constant path. Drives the
        // decision-time readouts and the bioavailability of any dose injected here —
        // a dose's F is fixed at injection (LOCF of the covariate to the decision).
        let readout_pk = match event_pk {
            Some(ev) => segment_pk_at(t_start, &obs_map, &pk_only_map, ev, last_pk),
            None => PkParams::default(),
        };
        let pk_readout: &[f64] = if tv {
            &readout_pk.values
        } else {
            pk_params_flat
        };
        // Eta to evaluate the pre-dose readouts with — paired to `readout_pk`'s
        // occasion (#701): occasion g's `[η_bsv | κ_g]` at a decision, else the fixed
        // baseline `eta`. Byte-identical to `eta` on the non-IOV path.
        let readout_eta = eta_for(
            eta_occ,
            eta,
            if iov {
                segment_occ_at(
                    t_start,
                    &obs_map,
                    &obs_occ,
                    &pk_only_map,
                    &pk_only_occ,
                    last_occ,
                )
            } else {
                None
            },
        );

        // --- Decision hook: observe (pre-dose trough) -> decide -> dose. ---
        if !stopped {
            if let Some(&decision_index) = decision_index_of.get(&t_start.to_bits()) {
                // Covariate snapshot in effect at the decision time. When the
                // decision coincides with an observation row, use that row's
                // per-observation snapshot. Otherwise, on the TV path (#700), carry
                // the covariate of the most-recent record forward (LOCF) so it stays
                // consistent with `pk_readout` (also LOCF) — a decision that lands
                // between records must NOT read the frozen t=0 covariate, or an
                // `observe` / `[scaling]` expression that references a time-varying
                // covariate directly would drive the controller off a stale value.
                // The constant path keeps the subject-static map (byte-identical).
                let decision_cov = match obs_map
                    .get(&t_start.to_bits())
                    .and_then(|idxs| idxs.first())
                {
                    Some(&i) => shadow.obs_cov(i),
                    None if tv => locf_decision_cov(t_start, subject, &shadow.covariates),
                    None => &shadow.covariates,
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
                        Some(f) => f(&u, pk_readout, theta, readout_eta, decision_cov),
                        None => read_observable(
                            ode,
                            &u,
                            pk_readout,
                            theta,
                            readout_eta,
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
                                pk_readout,
                                decision_index,
                            )?;
                            u[cmt - 1] += f * amt;
                            if !ext_params[crate::types::MAX_PK_PARAMS].is_finite() {
                                ext_params[crate::types::MAX_PK_PARAMS] = t_start;
                            }
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
                            if !ext_params[crate::types::MAX_PK_PARAMS].is_finite() {
                                ext_params[crate::types::MAX_PK_PARAMS] = t_start;
                            }
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

        // Record the observation exactly at t_start (post-dose), mirroring
        // `ode_predictions`' left-boundary recording.
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
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
                let obs_eta = eta_for(eta_occ, eta, if iov { obs_occ[obs_idx] } else { None });
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    obs_pk,
                    theta,
                    obs_eta,
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

            // PK governing the segment `(t_start, t_end]`: the record at `t_end`
            // (NONMEM end-of-interval convention) or the LOCF carry-forward on the TV
            // path (#700), the frozen snapshot otherwise. On the TV path it is written
            // into `ext_params`'s PK slots (leaving the TAFD/TAD anchors intact) for
            // the ODE RHS and passed through as the readout PK for any observation
            // `integrate_segment` records internally.
            let seg_pk = match event_pk {
                Some(ev) => segment_pk_at(t_end, &obs_map, &pk_only_map, ev, last_pk),
                None => PkParams::default(),
            };
            // Occasion governing this segment (#701) — the twin of `seg_pk`'s
            // end-of-interval / LOCF resolution, so the eta threaded into
            // `integrate_segment` carries the same occasion's κ as `seg_pk`.
            let seg_occ = if iov {
                segment_occ_at(
                    t_end,
                    &obs_map,
                    &obs_occ,
                    &pk_only_map,
                    &pk_only_occ,
                    last_occ,
                )
            } else {
                None
            };
            let seg_eta = eta_for(eta_occ, eta, seg_occ);
            let seg_pk_values: &[f64] = if tv { &seg_pk.values } else { pk_params_flat };
            if tv {
                ext_params[..crate::types::MAX_PK_PARAMS]
                    .copy_from_slice(&seg_pk.values[..crate::types::MAX_PK_PARAMS]);
            }

            // Realized doses are all controller-injected: lag is 0 (a nonzero lag is
            // rejected at injection) and F was captured at injection time in
            // `injected_f` (LOCF PK), so a later covariate change can't retroactively
            // rescale a delivered dose. Infusions are delivered by `integrate_segment`'s
            // `active_infusions` over any segment they fully span (the dynamic
            // infusion-end breaks guarantee full containment). On the constant path
            // `injected_f == f_bio(cmt, pk_params_flat)` for every dose, byte-identical.
            let dose_lagtimes: Vec<f64> = vec![0.0; shadow.doses.len()];

            integrate_segment(
                ode,
                &mut u,
                t_start,
                t_end,
                &shadow,
                &dose_lagtimes,
                &injected_f,
                &mut ext_params,
                seg_pk_values,
                theta,
                seg_eta,
                &obs_map,
                &mut predictions,
                None,
                &[],
            );

            // Advance the LOCF carry: after integrating into `t_end`, the record
            // there (if any) is the most-recent PK. `segment_pk_at` returns `last_pk`
            // unchanged for a non-record break, so this is a no-op in that case.
            // `last_occ` advances in lockstep (#701) so the next non-record segment
            // reads this occasion's κ.
            if tv {
                last_pk = seg_pk;
                last_occ = seg_occ;
            }
        }

        k += 1;
    }

    // Clamp negative predictions to zero, matching the static predictor.
    clamp_negative_predictions(&mut predictions);

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
    let mut static_subject = base_subject.clone();
    static_subject.doses = run
        .ledger
        .iter()
        .map(|e| DoseEvent::new(e.time, e.amt, e.cmt, e.rate, false, 0.0))
        .collect();

    // On the time-varying path (#700) the static replay must resolve PK per event
    // exactly as the reactive driver did — a single frozen snapshot would diverge.
    // The driver's `event_pk` is reused directly: its obs / pk-only snapshots depend
    // only on the (unchanged) observation grid and covariates, not on the doses, so
    // they are identical for the ledger-rebuilt static subject; each dose's realized
    // F is taken from the ledger. `adaptive_frozen_replay_tv` shares `segment_pk_at`
    // + `integrate_segment` with the driver, so the two stay bit-aligned. The
    // constant path keeps the general single-snapshot engine.
    let static_preds = match event_pk {
        Some(ev) => {
            let dose_f: Vec<f64> = run.ledger.iter().map(|e| e.f_applied).collect();
            adaptive_frozen_replay_tv(
                ode,
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

/// Static, up-front frozen-replay engine for the time-varying-covariate adaptive
/// path (#700). Rebuilds the trajectory from the realized ledger the way the
/// reactive driver did, but plans the entire break timeline **up front** from the
/// frozen ledger (rather than discovering it reactively) — so agreement with the
/// reactive run still proves the driver's dose bookkeeping. It shares
/// [`segment_pk_at`] and [`integrate_segment`] with the driver and adds the same
/// obs / pk-only breaks, so the two walk identical segments with identical
/// per-event PK and stay **bit-aligned**. Verifier-only; the constant-covariate
/// path keeps the general single-snapshot [`ode_predictions_with_extra_breaks`].
///
/// `dose_f[i]` is the realized bioavailability of ledger dose `i` (the driver's
/// `f_applied`), so a delivered dose's F is taken as given rather than re-derived —
/// F correctness is pinned separately by the degenerate oracle against
/// `ode_predictions_event_driven`.
///
/// IOV (#701): when `eta_occ` / `decision_pk` are `Some`, the replay threads the
/// **same** per-window eta and per-decision occasion PK as the driver — occasion is
/// resolved from `extra_breaks` (the decision schedule) exactly as the driver
/// resolves it, so the two apply identical per-occasion κ over identical windows and
/// stay bit-aligned.
#[allow(clippy::too_many_arguments)]
fn adaptive_frozen_replay_tv(
    ode: &OdeSpec,
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
    let mut pk_only_map: HashMap<u64, usize> = HashMap::new();
    for (m, &t) in subject.pk_only_times.iter().enumerate() {
        pk_only_map.entry(t.to_bits()).or_insert(m);
    }

    // Occasion bookkeeping for the IOV path (#701), mirroring the driver: per-record
    // occasion resolved from the decision schedule (`extra_breaks`), a decision-time
    // → index map to open windows at decision breaks, and the LOCF `last_occ` carry.
    let (obs_occ, pk_only_occ): (Vec<Option<usize>>, Vec<Option<usize>>) = if iov {
        (
            subject
                .obs_times
                .iter()
                .map(|&t| crate::pk::occasion_of(extra_breaks, t))
                .collect(),
            subject
                .pk_only_times
                .iter()
                .map(|&t| crate::pk::occasion_of(extra_breaks, t))
                .collect(),
        )
    } else {
        (Vec::new(), Vec::new())
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

    // Injected doses carry no lag (a nonzero lag is rejected at injection).
    let dose_lagtimes = vec![0.0; subject.doses.len()];

    // NB: PK slots are left NaN here (unlike `seed_ext_params`) — the replay
    // overwrites them per-segment from each event's own snapshot before integrating.
    let mut ext_params = [f64::NAN; crate::types::MAX_PK_PARAMS + 2];
    let first_dose_time = earliest_dose_time(subject);
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
    let mut break_times: Vec<f64> = vec![0.0, t_last];
    for (i, d) in subject.doses.iter().enumerate() {
        break_times.push(d.time);
        if is_real_infusion(d) {
            let (_, dur_eff) = d.bioavailable_infusion(dose_f[i]);
            break_times.push(d.time + dur_eff);
        }
    }
    break_times.extend(subject.obs_times.iter().cloned());
    break_times.extend(subject.pk_only_times.iter().cloned());
    break_times.extend(
        extra_breaks
            .iter()
            .copied()
            .filter(|b| b.is_finite() && *b > 0.0),
    );
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    if break_times.len() < 2 {
        break_times.push(break_times[0]);
    }

    for k in 0..break_times.len() {
        let t_start = break_times[k];

        // #701: open a decision window here, mirroring the driver — set the LOCF PK +
        // occasion to this decision's occasion snapshot so the coincident observation,
        // and any following non-record segment, read occasion g's κ.
        if let (Some(dp), Some(&g)) = (decision_pk, decision_index_of.get(&t_start.to_bits())) {
            last_pk = dp[g];
            last_occ = Some(g);
        }

        // Apply boluses landing at t_start (lag 0) with their realized F — at EVERY
        // break, including the last. The driver's while-loop processes the final
        // break too (so a decision at the maximum time still doses and its coincident
        // observation is read post-dose); the replay must match, or a dose landing at
        // the last time is silently dropped here (the frozen-replay verifier caught
        // exactly this). Infusions add nothing here — `integrate_segment`'s
        // `active_infusions` delivers them over every segment they span.
        for (i, d) in subject.doses.iter().enumerate() {
            if (d.time - t_start).abs() >= 1e-12 {
                continue;
            }
            if !is_real_infusion(d) && !input_rate_consumes_cmt(ode, d.cmt) {
                let cmt_idx = d.cmt.saturating_sub(1);
                if cmt_idx < n {
                    u[cmt_idx] += dose_f[i] * d.amt;
                }
            }
        }

        // Record obs at the left boundary (post-dose) with each observation's own
        // per-event PK (consistent with the state propagated into this boundary).
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            for &obs_idx in obs_idxs {
                let cmt = subject.obs_cmts.get(obs_idx).copied().unwrap_or(0);
                let obs_eta = eta_for(eta_occ, eta, if iov { obs_occ[obs_idx] } else { None });
                predictions[obs_idx] = read_observable(
                    ode,
                    &u,
                    &event_pk.obs[obs_idx].values,
                    theta,
                    obs_eta,
                    subject.obs_cov(obs_idx),
                    cmt,
                );
            }
        }

        // Integrate `(t_start, t_end]` to the next break, if any. The final break has
        // no successor — its dose + observation were applied above.
        if k + 1 < break_times.len() {
            let t_end = break_times[k + 1];
            // Segment PK = record at t_end (NONMEM end-of-interval) or LOCF carry —
            // the identical `segment_pk_at` the driver used.
            let seg_pk = segment_pk_at(t_end, &obs_map, &pk_only_map, event_pk, last_pk);
            // Occasion twin of `seg_pk` (#701), so the threaded eta carries the same
            // occasion's κ — the identical `segment_occ_at` the driver used.
            let seg_occ = if iov {
                segment_occ_at(
                    t_end,
                    &obs_map,
                    &obs_occ,
                    &pk_only_map,
                    &pk_only_occ,
                    last_occ,
                )
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
                &mut ext_params,
                &seg_pk.values,
                theta,
                seg_eta,
                &obs_map,
                &mut predictions,
                None,
                &[],
            );

            last_pk = seg_pk;
            last_occ = seg_occ;
        }
    }

    clamp_negative_predictions(&mut predictions);
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
/// are replaced (its covariates carry over). This pass uses a **single** PK snapshot
/// (`pk_params_flat`), exact only for constant-covariate subjects — a time-varying
/// (or TIME-in-PK) subject with an `auc_target` is rejected upstream in
/// `run_adaptive_population` (#700), so it never reaches here.
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
            if d.amt > 0.0 && d.cmt.saturating_sub(1) == forcing.cmt {
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
                &ode.input_rate,
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
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    let first_dose_time = earliest_dose_time(subject);
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

    for k in 0..break_times.len() {
        let t_start = break_times[k];

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
                        // dose.cmt is 1-based; `CMT=0` is NONMEM's default dose
                        // compartment and resolves to compartment 1, like every
                        // other dose site on both engines (#899). This used to
                        // skip the dose entirely, disagreeing with the two
                        // event-driven drivers on the same dataset.
                        let cmt = dose.cmt.saturating_sub(1);
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

        // Handle obs at t_start (after dose).
        if let Some(obs_idxs) = obs_map.get(&t_start.to_bits()) {
            record_observations(
                ode,
                obs_idxs,
                &u,
                pk_params_flat,
                theta,
                eta,
                subject,
                &mut predictions,
                Some(states.as_mut_slice()),
            );
        }

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
        ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor(subject, &dose_lagtimes, t_start);

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

    clamp_negative_predictions(&mut predictions);

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
    let pk_flat = &pk_at_obs
        .first()
        .map(|p| p.values)
        .unwrap_or([0.0; crate::types::MAX_PK_PARAMS]);
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
                    // dose.cmt is 1-based; `CMT=0` is NONMEM's default dose
                    // compartment and resolves to compartment 1, like every
                    // other dose site on both engines (#899).
                    let cmt = dose.cmt.saturating_sub(1);
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

    // TAD anchor: SS-aware, matching ode_predictions (rem_euclid wraps the elapsed
    // time back into [0, II)).
    ext_params[crate::types::MAX_PK_PARAMS + 1] = tad_anchor(subject, dose_lagtimes, t_start);

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
    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    let first_dose_time = earliest_dose_time(subject);
    let mut ext_params = seed_ext_params(pk_params_flat, first_dose_time);

    // Build saveat → index map for fast lookup.
    let saveat_map = build_obs_index_map(saveat);

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

    // Saveat nodes earlier than the first integrated segment (e.g. a discrete-state
    // CTMM observation recorded before the first dose, whose times the segment
    // timeline — built from doses/obs_times/pk-only/resets, not `obs_records` — does
    // not cover). Nothing has acted on the system before the first event, so the state
    // there is the seeded initial state `u`; fill it so these nodes are not left as
    // `NaN`, which a downstream finiteness guard (the inhomogeneous CTMM scorer) would
    // otherwise read as a diverged solve and use to wrongly repel a valid subject. No-op
    // for the usual case where every saveat is at or after the first event.
    if let Some(&first_start) = break_times.first() {
        for (i, &t) in saveat.iter().enumerate() {
            if t < first_start - 1e-12 {
                result[i] = u.clone();
            }
        }
    }

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

    // #731: the `windows(2)` loop visits the final break `t_last` only as a segment
    // *end*, so a dose landing exactly on it is never applied and a `saveat` there
    // reads the pre-dose state — the same terminal-break bug fixed on the
    // constant-parameter engine (`ode_predictions_and_chz`). Visit `t_last` once more
    // as a left boundary (dose applied via the shared `apply_segment_boundary`, then
    // the `saveat` re-read post-dose) so the two paths agree and #570's
    // one-solve == two-solve equivalence holds at a dose-on-`t_last`. When no dose
    // lands there this re-reads the identical carried `u`, so it stays byte-identical
    // everywhere else. Guarded to a timeline the loop actually ran (`len >= 2`); a
    // single-instant `saveat` keeps its prior behaviour.
    if break_times.len() >= 2 {
        let t_last_break = break_times[break_times.len() - 1];
        let _ = apply_segment_boundary(
            ode,
            subject,
            &dose_lagtimes,
            &dose_f_bio,
            &zo_windows,
            pk_params_flat,
            n,
            &opts,
            t_last_break,
            t_last_break,
            &mut u,
            &mut active_infusions,
            &mut ext_params,
        );
        if let Some(idxs) = saveat_map.get(&t_last_break.to_bits()) {
            for &i in idxs {
                result[i] = u.clone();
            }
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

    let (dose_lagtimes, dose_f_bio) = subject_dose_attrs(subject, ode, pk_params_flat);

    let first_dose_time = earliest_dose_time(subject);
    let mut ext_params = seed_ext_params(pk_params_flat, first_dose_time);

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
#[path = "predictions_tests.rs"]
mod tests;
