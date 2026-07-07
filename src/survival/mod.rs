// TTE / survival non-Gaussian endpoint support — Phase 1.
//
// Public interface:
//   tte_data_term         — negative log-likelihood for a TTE subject's records
//   data_term_hessian_fd  — 4-point FD Hessian of any scalar eta-function
//   shi_step_sizes        — adaptive Shi (2021) step-size vector for FD Hessian
//   simulate_tte          — draw TTE event times (administratively right-censored at
//                           each subject's observation window) into a SimulationResult vec
//
// See plans/tte-survival-markov.md §3.1, §2.3, §9.3, §8.8.2.

pub mod parametric;

pub use parametric::{
    cum_hazard, hazard_and_cum_hazard, mean_survival, median_survival,
    sample_conditional_event_time, sample_event_time,
};

use nalgebra::DMatrix;
use rand::RngExt;
use std::collections::HashMap;

use crate::types::{
    EndpointLikelihood, EventType, HazardFamily, HazardSpec, ObsRecord, RtteClock, SimOutcome,
    TteRecurrence,
};

// ─────────────────────────────────────────────────────────────────────────────
//  TTE data term
// ─────────────────────────────────────────────────────────────────────────────

/// Round-off tolerance for the cumulative-hazard monotonicity guard in
/// [`tte_nll_from_curves`]. A cumulative hazard is non-decreasing, so a negative
/// increment `H(b) - H(a) < 0` is ill-posed - but each `H` read carries solver
/// round-off, so an increment counts as a genuine violation only past
/// `abstol + reltol * |H|`. This mirrors the ODE integrator's own per-step
/// monotonicity tolerance (`mono_tol` in `src/ode/solver.rs`).
///
/// ODE-accumulated callers pass the *configured* solver tolerances
/// ([`MonoTol::from_solver`]) so a user-tightened `ode_reltol` tightens the guard in
/// lockstep; the analytic closed-form path passes a tight fixed floor
/// ([`MonoTol::analytic`]) since it carries only f64 round-off, not solver error.
#[derive(Clone, Copy, Debug)]
pub struct MonoTol {
    pub reltol: f64,
    pub abstol: f64,
}

impl MonoTol {
    /// Tight floor for closed-form hazard families. Their `H(t)` is evaluated in closed
    /// form (no ODE solve), so the only round-off is f64 arithmetic - a few ULPs, i.e.
    /// ~1e-15 relative. Valid families are monotone by construction, so this floor mainly
    /// guards against sign-flipped parameters. `1e-9` is used for *both* terms, giving a
    /// floor `1e-9 + 1e-9*|H|` that scales with `H` (so it does not over-reject a large
    /// closed-form `H`) yet sits ~6 orders above the true round-off - tight enough to
    /// still catch any genuine negative increment, loose enough never to fire on a
    /// legitimate fit.
    pub fn analytic() -> Self {
        Self {
            reltol: 1e-9,
            abstol: 1e-9,
        }
    }

    /// Floor tied to the ODE solver's effective tolerances.
    pub fn from_solver(opts: &crate::ode::OdeSolverOptions) -> Self {
        Self {
            reltol: opts.reltol,
            abstol: opts.abstol,
        }
    }
}

impl Default for MonoTol {
    /// The standard floor at the solver's default tolerances (reltol 1e-4, abstol 1e-6),
    /// delegating to [`crate::ode::OdeSolverOptions::default`] so the two stay in sync.
    fn default() -> Self {
        Self::from_solver(&crate::ode::OdeSolverOptions::default())
    }
}

/// Negative log-likelihood contribution of a TTE endpoint for one subject.
///
/// Handles all three EventType variants and left truncation (entry_time > 0).
///
/// Formula (§3.1 of plan):
///   RightCensored:   H(T) − H(entry)
///   Exact:           H(T) − H(entry) − log h(T)
///   IntervalCensored: −log [ exp(−(H(left)−H(entry))) − exp(−(H(right)−H(entry))) ]
///
/// Returns 1e20 as a sentinel when the likelihood is numerically ill-defined
/// (e.g. negative interval probability, non-positive hazard for an exact event).
pub fn tte_data_term(
    records: &[ObsRecord],
    hazard: &HazardSpec,
    recurrence: TteRecurrence,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
) -> f64 {
    match hazard {
        HazardSpec::Analytic { family, param_fn } => {
            let params = param_fn(theta, eta, covariates);
            let cumhaz_at = |t| cum_hazard(*family, t, &params);
            let hazard_at = |t| hazard_and_cum_hazard(*family, t, &params).0;
            // Analytic families carry only f64 round-off (no ODE solver error), so both
            // recurrence paths use the tight analytic monotone floor (#633).
            match recurrence {
                // Standard TTE / competing risks: per-record independent terms.
                TteRecurrence::Single => {
                    tte_nll_from_curves(records, cumhaz_at, hazard_at, MonoTol::analytic())
                }
                // Clock-forward RTTE (Andersen–Gill): the cumulative hazard is integrated
                // once across the subject's whole risk window (§3.3), so the records couple.
                TteRecurrence::Repeated {
                    clock: RtteClock::Forward,
                } => {
                    rtte_forward_nll_from_curves(records, cumhaz_at, hazard_at, MonoTol::analytic())
                }
                // Clock-reset (gap-time / renewal) RTTE: the hazard clock resets at each
                // event, so each inter-event gap is an independent contribution evaluated
                // on its own duration (§3.3). For an analytic family this needs no ODE —
                // the closures are evaluated at the gap `Δ_k`, not absolute time.
                TteRecurrence::Repeated {
                    clock: RtteClock::Reset,
                } => rtte_reset_nll_from_curves(records, cumhaz_at, hazard_at, MonoTol::analytic()),
            }
        }
        // ODE-accumulated hazards are routed through `tte_endpoint_nll` → `tte_ode_nll`
        // (which reads H(t)/h(t) from the integrated CHZ state); this closed-form entry
        // point is analytic-only and returns the ill-defined sentinel if mis-called. Kept
        // as an explicit, testable arm (not `unreachable!`/`debug_assert!`) so the dispatch
        // contract is covered and behaves identically in debug and ci-test builds.
        HazardSpec::OdeAccumulated { .. } => 1e20,
    }
}

/// Per-record TTE negative log-likelihood from cumulative-hazard / hazard curves.
///
/// Shared by the analytic-family path ([`tte_data_term`]) and the ODE-accumulated
/// path (joint PK-TTE); they differ only in *where* the curves come from.
/// `cumhaz_at(t)` returns `H(t)`; `hazard_at(t)` returns the instantaneous hazard
/// `h(t)`. Handles all three [`EventType`] variants and left truncation
/// (entry_time > 0):
///   RightCensored:    H(T) − H(entry)
///   Exact:            H(T) − H(entry) − log h(T)
///   IntervalCensored: −log [ exp(−(H(left)−H(entry))) − exp(−(H(right)−H(entry))) ]
///
/// Returns 1e20 as a sentinel when the likelihood is numerically ill-defined
/// (e.g. negative interval probability, non-positive hazard for an exact event,
/// or a **non-monotone cumulative hazard** — `H(b) < H(a)` for `b ≥ a`, which a
/// negative drug-driven hazard produces and which would imply `S = exp(−ΔH) > 1`).
pub fn tte_nll_from_curves(
    records: &[ObsRecord],
    cumhaz_at: impl Fn(f64) -> f64,
    hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    let mut nll = 0.0_f64;

    // A cumulative hazard is non-decreasing: `H(b) ≥ H(a)` for `b ≥ a`. A drug-driven
    // `[odes]` hazard is a free user expression with no `h ≥ 0` constraint, so a
    // sign-flipped / non-monotone hazard can make an increment negative — `S = exp(−ΔH)
    // > 1`, ill-posed. Treat an increment as a genuine violation only past the solver's
    // own round-off band `abstol + reltol·|H|` (see `MonoTol`), so a near-zero hazard's
    // quadrature noise on a flat `H` is tolerated while a larger negative step is not.
    // The simulation path hard-errors on the same non-monotone CHZ; here, with an
    // optimizer to steer, it folds into the shared 1e20 sentinel.
    // #618: the previous fixed `1e-3·|H|` term was 10× looser than the default `reltol`
    // and ignored a user-tightened `ode_reltol`, so a genuine negative step up to ~0.1%·H
    // slipped past as round-off; tying the floor to `tol` closes that gap.
    let monotone_violation = |hi: f64, lo: f64| cumhaz_monotone_violation(hi, lo, tol);

    for record in records {
        let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            ..
        } = record;

        let h_entry = if *entry_time > 0.0 {
            cumhaz_at(*entry_time)
        } else {
            0.0
        };

        match event_type {
            EventType::RightCensored => {
                let h_t = cumhaz_at(*time);
                if monotone_violation(h_t, h_entry) {
                    return 1e20;
                }
                nll += h_t - h_entry;
            }
            EventType::Exact => {
                let h_val = hazard_at(*time);
                if h_val <= 0.0 {
                    return 1e20;
                }
                let h_t = cumhaz_at(*time);
                if monotone_violation(h_t, h_entry) {
                    return 1e20;
                }
                nll += h_t - h_entry - h_val.ln();
            }
            EventType::IntervalCensored { left, right } => {
                let h_l = cumhaz_at(*left);
                let h_r = cumhaz_at(*right);
                // H(left) ≥ H(entry) and H(right) > H(left): a non-monotone CHZ
                // (negative hazard) violates either and is ill-posed.
                if monotone_violation(h_l, h_entry) {
                    return 1e20;
                }
                let a = h_l - h_entry; // H(left) − H(entry) ≥ 0
                let delta = h_r - h_l; // H(right) − H(left) > 0 for a proper interval
                if delta <= 0.0 {
                    // Degenerate: right ≤ left, or hazard non-monotone (bad params).
                    return 1e20;
                }
                // log P(left < T ≤ right | T > entry) = −a + log(1 − exp(−delta))
                // Use expm1 for numerical precision when delta is small (tight interval).
                // Computing exp(-a) − exp(-b) in probability space would lose significant
                // digits for small Δ or large hazards — the log-domain form avoids that.
                let log_prob = -a + (-delta).exp_m1().abs().ln();
                if !log_prob.is_finite() {
                    return 1e20;
                }
                nll -= log_prob;
            }
        }
    }

    if nll.is_finite() {
        nll
    } else {
        1e20
    }
}

/// Round-off-tolerant non-monotonicity test for a cumulative-hazard increment
/// `H(hi) − H(lo)` (`hi ≥ lo` in time). `H` is non-decreasing, but a drug-driven
/// `[odes]` hazard is a free user expression with no `h ≥ 0` constraint, so a
/// sign-flipped hazard can drive an increment negative — `S = exp(−ΔH) > 1`, which
/// is ill-posed. Treat it as a genuine violation only past the solver's own round-off
/// band `abstol + reltol·|H|` (see [`MonoTol`], #633). Shared by
/// [`tte_nll_from_curves`] and [`rtte_forward_nll_from_curves`] so the floor lives in
/// one place (see #618 re: decoupling it from a fixed `|H|` term).
#[inline]
fn cumhaz_monotone_violation(hi: f64, lo: f64, tol: MonoTol) -> bool {
    hi - lo < -crate::ode::scale_tol(tol.abstol, tol.reltol, hi, lo)
}

/// Clock-forward (Andersen–Gill) **recurrent-event** (RTTE) negative log-likelihood
/// from cumulative-hazard / hazard curves:
///
/// ```text
///   −log L = H(T) − H(entry) − Σ_k log h(t_k)
/// ```
///
/// Unlike [`tte_nll_from_curves`], which treats each record as an independent
/// single-event term, the cumulative hazard here is integrated **once** across the
/// subject's whole risk window (§3.3 of `plans/tte-survival-markov.md`): each record
/// uses the *previous* record's time as the lower integration limit, so the per-record
/// `H`-increments telescope to `H(T) − H(entry)`. Summing independent single-event
/// terms instead would over-count the cumulative hazard by `Σ_k H(t_k)` — the classic
/// "treat recurrent events as independent observations" error.
///
/// Preconditions (enforced upstream by `api::check_rtte_records` at the fit boundary):
///   * `records` are in **nondecreasing, finite `time`** order (the last is typically
///     the administrative right-censor at the horizon `T ≥ t_K`);
///   * left truncation uses the **first** record's `entry_time` as the initial lower
///     limit; per-record `entry_time` on later records is ignored (the previous event
///     is the lower limit);
///   * `IntervalCensored` is not supported for RTTE. Interval censoring is a *data*
///     property (a DV=0→DV=2 pair) the parser cannot see, so it is rejected at the fit
///     boundary; the `1e20` sentinel here is defense-in-depth for a direct caller.
///
/// Returns the `1e20` sentinel on a non-positive hazard at an event, a non-monotone
/// cumulative hazard, or a non-finite total — matching [`tte_nll_from_curves`].
pub fn rtte_forward_nll_from_curves(
    records: &[ObsRecord],
    cumhaz_at: impl Fn(f64) -> f64,
    hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    let mut nll = 0.0_f64;
    // Running lower integration limit as a cumulative hazard `H(lo)`. Seeded from the
    // subject's left-truncation entry on the first record, then advanced to each
    // record's `H(time)` so the increments telescope.
    let mut h_lo = 0.0_f64;
    let mut seeded = false;

    for record in records {
        let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            ..
        } = record;

        if !seeded {
            h_lo = if *entry_time > 0.0 {
                cumhaz_at(*entry_time)
            } else {
                0.0
            };
            seeded = true;
        }

        match event_type {
            EventType::RightCensored => {
                let h_t = cumhaz_at(*time);
                if cumhaz_monotone_violation(h_t, h_lo, tol) {
                    return 1e20;
                }
                nll += h_t - h_lo;
                h_lo = h_t;
            }
            EventType::Exact => {
                let h_val = hazard_at(*time);
                if h_val <= 0.0 {
                    return 1e20;
                }
                let h_t = cumhaz_at(*time);
                if cumhaz_monotone_violation(h_t, h_lo, tol) {
                    return 1e20;
                }
                nll += (h_t - h_lo) - h_val.ln();
                h_lo = h_t;
            }
            // RTTE + interval censoring is unsupported (rejected at the fit boundary by
            // `api::check_rtte_records`, since DV-driven censoring is invisible to the
            // parser); this sentinel is defense-in-depth for a direct caller.
            EventType::IntervalCensored { .. } => return 1e20,
        }
    }

    if nll.is_finite() {
        nll
    } else {
        1e20
    }
}

/// Clock-reset (gap-time / renewal) **recurrent-event** (RTTE) negative log-likelihood
/// from cumulative-hazard / hazard curves:
///
/// ```text
///   −log L = Σ_k [ H(Δ_k) − log h(Δ_k) ]  +  H(Δ_censor)
/// ```
///
/// The hazard clock **resets to 0 at each event**, so each inter-event gap
/// `Δ_k = t_k − t_{k−1}` (with `t_0` = the subject's `entry_time`) is an independent
/// single-event contribution evaluated on its own *duration* — `cumhaz_at`/`hazard_at`
/// are called at `Δ_k`, not at absolute time. This is the renewal-process form (§3.3);
/// for a time-homogeneous hazard (exponential) it coincides with the clock-forward
/// likelihood, and differs once the hazard varies with time (Weibull/Gompertz).
///
/// Preconditions match [`rtte_forward_nll_from_curves`]: records nondecreasing in `time`
/// (enforced at data load), `IntervalCensored` unsupported (folds to the `1e20`
/// sentinel). Returns the sentinel on a non-positive hazard at an event, a non-monotone
/// cumulative hazard on a gap, or a non-finite total.
pub fn rtte_reset_nll_from_curves(
    records: &[ObsRecord],
    cumhaz_at: impl Fn(f64) -> f64,
    hazard_at: impl Fn(f64) -> f64,
    tol: MonoTol,
) -> f64 {
    let mut nll = 0.0_f64;
    // Start of the current gap: the previous record's time, or the subject's entry on
    // the first record.
    let mut prev_time: Option<f64> = None;

    for record in records {
        let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            ..
        } = record;

        let gap = time - prev_time.unwrap_or(*entry_time);
        // A negative gap means out-of-order records (guarded at data load); the H(Δ) with
        // Δ<0 is meaningless, so fail closed.
        if gap < 0.0 {
            return 1e20;
        }

        match event_type {
            EventType::RightCensored => {
                let h_gap = cumhaz_at(gap); // H(Δ) from a reset clock (lower limit 0)
                if cumhaz_monotone_violation(h_gap, 0.0, tol) {
                    return 1e20;
                }
                nll += h_gap;
            }
            EventType::Exact => {
                let h_val = hazard_at(gap);
                if h_val <= 0.0 {
                    return 1e20;
                }
                let h_gap = cumhaz_at(gap);
                if cumhaz_monotone_violation(h_gap, 0.0, tol) {
                    return 1e20;
                }
                nll += h_gap - h_val.ln();
            }
            // RTTE + interval censoring is unsupported (rejected at parse); sentinel.
            EventType::IntervalCensored { .. } => return 1e20,
        }
        prev_time = Some(*time);
    }

    if nll.is_finite() {
        nll
    } else {
        1e20
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FD Hessian and Shi step sizes
// ─────────────────────────────────────────────────────────────────────────────

/// 4-point central-stencil finite-difference Hessian of `eval` at `eta_hat`.
///
/// Cost: 2·n·(n+1) evaluations (n=1→4, n=2→12, n=4→40).
///
/// `eps[j]` is the step size for dimension j; use `shi_step_sizes` to compute them.
///
/// The (j,k) entry is:
///   (f(η+sj·ej+sk·ek) − f(η+sj·ej−sk·ek) − f(η−sj·ej+sk·ek) + f(η−sj·ej−sk·ek))
///   ─────────────────────────────────────────────────────────────────────────────────
///                              4 · sj · sk
///
/// For j==k this reduces to the standard central-difference second derivative with step 2·sj.
pub fn data_term_hessian_fd(
    eval: impl Fn(&[f64]) -> f64,
    eta_hat: &[f64],
    eps: &[f64],
) -> DMatrix<f64> {
    let n = eta_hat.len();
    let mut h = DMatrix::zeros(n, n);

    let perturb = |j: usize, dj: f64, k: usize, dk: f64| -> f64 {
        let mut e = eta_hat.to_vec();
        e[j] += dj * eps[j];
        e[k] += dk * eps[k];
        eval(&e)
    };

    for j in 0..n {
        for k in 0..=j {
            let entry =
                (perturb(j, 1.0, k, 1.0) - perturb(j, 1.0, k, -1.0) - perturb(j, -1.0, k, 1.0)
                    + perturb(j, -1.0, k, -1.0))
                    / (4.0 * eps[j] * eps[k]);
            h[(j, k)] = entry;
            h[(k, j)] = entry;
        }
    }
    h
}

/// Shi (2021) adaptive step sizes for FD Hessian.
///
/// Computes the central-difference gradient of `eval` at `eta_hat` (2·n evals),
/// takes the harmonic mean of gradient component norms, then scales by ε^(1/3).
/// Returns a per-dimension step vector — each component scaled by the harmonic mean.
///
/// Falls back to a fixed 1e-4 per dimension when all gradient components are near zero.
pub fn shi_step_sizes(eval: impl Fn(&[f64]) -> f64, eta_hat: &[f64]) -> Vec<f64> {
    let n = eta_hat.len();
    let base_step = 1e-5_f64; // forward-difference step for gradient norms
    let scale = f64::EPSILON.powf(1.0 / 3.0);

    let mut grad_norms = Vec::with_capacity(n);
    for j in 0..n {
        let mut e_fwd = eta_hat.to_vec();
        let mut e_bwd = eta_hat.to_vec();
        e_fwd[j] += base_step;
        e_bwd[j] -= base_step;
        let g_j = (eval(&e_fwd) - eval(&e_bwd)) / (2.0 * base_step);
        grad_norms.push(g_j.abs().max(1e-10));
    }

    // Harmonic mean of gradient norms; then apply Shi (2021) eq. (3.4):
    //   h_opt ≈ (harmonic_norm)^(1/3) · ε_mach^(1/3)
    let n_f = n as f64;
    let inv_sum: f64 = grad_norms.iter().map(|g| 1.0 / g).sum();
    let harmonic = if inv_sum > 0.0 { n_f / inv_sum } else { 1e-4 };
    let step = (harmonic.powf(1.0 / 3.0) * scale).max(1e-6).min(0.1);

    vec![step; n]
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Draw one latent event time (uncensored) for a hazard family. Consumes exactly
/// one uniform from `rng`. `entry_time > 0` draws conditionally on survival past
/// entry (left truncation, §3.6). Returns `f64::MAX` for degenerate / improper
/// cases (`λ = 0`; a Gompertz with `γ < 0` whose survival never reaches the drawn
/// quantile) — that value never wins a competing-risks `min` and censors at the
/// window in `draw_tte_outcome`.
fn draw_tte_latent<R: rand::Rng>(
    family: HazardFamily,
    params: &[f64],
    entry_time: f64,
    rng: &mut R,
) -> f64 {
    // Open01 samples from (0, 1) exclusive, avoiding the u=0 edge case that
    // would send -ln(u) to +∞.
    let u: f64 = rng.sample(rand::distr::Open01);
    if entry_time > 0.0 {
        sample_conditional_event_time(family, params, entry_time, u)
    } else {
        sample_event_time(family, params, u)
    }
}

/// Draw one TTE outcome within the observation window `window`.
///
/// Returns `(time, observed)`:
/// - an **event** at the drawn time when it falls before `window`
///   (`observed = true`, `time = t_event`);
/// - **administrative right-censoring** at `window` when the drawn event time
///   reaches the window (`observed = false`, `time = window`).
///
/// The draw comes from [`draw_tte_latent`], whose `f64::MAX` degenerate / improper
/// sentinel must not be reported as an event: an event is recorded only for a draw
/// strictly below the window **and** below that sentinel, so a degenerate draw is
/// censored even when `window` is unbounded (`+∞`, an event record carrying no
/// administrative horizon; see [`observation_window`]).
fn draw_tte_outcome<R: rand::Rng>(
    family: HazardFamily,
    params: &[f64],
    entry_time: f64,
    window: f64,
    rng: &mut R,
) -> (f64, bool) {
    let t_event = draw_tte_latent(family, params, entry_time, rng);
    // `t_event < f64::MAX` rejects the samplers' degenerate / improper sentinel;
    // without it an unbounded window would mis-report that sentinel as an
    // observed event at `f64::MAX`.
    if t_event < window && t_event < f64::MAX {
        (t_event, true)
    } else {
        (window, false)
    }
}

/// The administrative right-censoring horizon for a TTE record, or `+∞` when the
/// record carries none.
///
/// Only a `RightCensored` record marks an administrative horizon: the subject
/// was event-free through `time`, at which point observation ended, so a
/// simulated draw reaching `time` is genuinely censored there. An `Exact` or
/// `IntervalCensored` record instead marks an *event* — its `time` field is the
/// realized event time (or interval upper bound), **not** a horizon. Censoring a
/// re-simulated draw at a realized event time would truncate the simulated
/// event-time distribution at the data's own event times (a re-simulation / VPC
/// bias), so those records draw uncensored (`+∞`). Administrative censoring for
/// such a design is supplied either by the design itself (right-censored
/// template rows, as the SSE tests do) or by the forthcoming
/// `[simulation] horizon` (Phase 2).
fn observation_window(event_type: &EventType, time: f64) -> f64 {
    match event_type {
        EventType::RightCensored => time,
        EventType::Exact | EventType::IntervalCensored { .. } => f64::INFINITY,
    }
}

/// Resolve a competing-risks outcome from each cause's latent event time and the
/// subject's administrative `window`. Returns `(winning cause index, observed
/// time, event_observed)`: the earliest latent time wins, and an event is
/// observed only if it precedes the window — otherwise the subject is censored at
/// the window (the winning index is still the earliest cause but `event = false`).
/// `total_cmp` makes ties and `f64::MAX` sentinels deterministic (lowest index).
/// The `t_star < f64::MAX` guard prevents a degenerate / improper draw (the
/// samplers' sentinel) from being reported as a spurious event when the window is
/// unbounded (`+∞`, e.g. every cause is an event record carrying no horizon).
fn resolve_competing_risks(latents: &[f64], window: f64) -> (usize, f64, bool) {
    let (win_idx, &t_star) = latents
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .expect("at least one cause");
    let event = t_star < window && t_star < f64::MAX;
    (win_idx, if event { t_star } else { window }, event)
}

/// Destructure a TTE endpoint and evaluate its hazard parameters at the given
/// `(theta, eta, covariates)`. Returns `None` for any non-`Tte` endpoint so
/// callers can `filter_map`/`?` over `model.endpoints` or a looked-up CMT.
///
/// This centralises the `EndpointLikelihood::Tte { HazardSpec::Analytic {
/// family, param_fn } }` destructure and `param_fn` evaluation shared by
/// `simulate_tte` (per `obs_record`) and `predict_survival` (per endpoint), so a
/// new `HazardSpec`/`HazardFamily` variant only has to be handled here. Callers
/// supply their own trailing per-cause fields (window/entry time vs. summary
/// statistics).
pub(crate) fn tte_cause_params(
    endpoint: &EndpointLikelihood,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
) -> Option<(HazardFamily, Vec<f64>)> {
    let EndpointLikelihood::Tte { hazard, .. } = endpoint else {
        return None;
    };
    // Analytic families expose closed-form cause params; the ODE-accumulated hazard
    // does not (its H/h come from the integrated CHZ state), so it has no analytic
    // cause params here. Simulation of ODE-accumulated TTE is guarded at the api
    // layer (Slice 2.2 root-finder); this simply reports "no analytic cause".
    let HazardSpec::Analytic { family, param_fn } = hazard else {
        return None;
    };
    Some((*family, param_fn(theta, eta, covariates)))
}

/// Solve a subject's augmented ODE at `(theta, eta)` and read the cumulative hazard
/// `H(t) = u[chz_state]` and instantaneous hazard `h(t) = u̇[chz_state]` at each `times`
/// point (joint PK-TTE, #564). `times` must be the exact f64 values the caller will look
/// up. Returns `(H, h)` aligned to `times`; an entry is NaN where the solve did not reach
/// that time (or there is no ODE / the solve length disagrees). Shared by the TTE
/// likelihood (`stats::likelihood::tte_ode_nll`) and `api::predict_survival` so the
/// H/h-extraction contract lives in one place; one reusable derivative buffer serves all
/// `h(t)` evaluations.
#[cfg(feature = "survival")]
pub(crate) fn ode_cumhaz_hazard(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    chz_state: usize,
    theta: &[f64],
    eta: &[f64],
    times: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = times.len();
    let (mut cum, mut haz) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    let Some(ode) = model.ode_spec.as_ref() else {
        return (cum, haz);
    };
    // Single-pass hazard integration with a constant PK-parameter vector. A
    // time-dependent hazard RHS (e.g. a Weibull `TIME` term) is honoured by the
    // integrator clock; the individual-parameter snapshot resolves the `TIME`
    // built-in at the integration start (t=0). #610.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let states = crate::ode::ode_dense_solve_states(ode, &pk.values, theta, eta, subject, times);
    if states.len() != n {
        return (cum, haz);
    }
    // `h(t)` is the cumulative-hazard derivative: the bare hazard RHS at the integrated
    // state (dose forcings touch PK compartments, not CHZ). One buffer, reused — only the
    // `chz_state` slot is read, and the RHS always writes it.
    let mut du = vec![0.0; ode.n_states];
    for (i, &t) in times.iter().enumerate() {
        cum[i] = states[i][chz_state];
        (ode.rhs)(&states[i], &pk.values, t, &mut du);
        haz[i] = du[chz_state];
    }
    (cum, haz)
}

/// Draw one **ODE-accumulated** (drug-driven) TTE latent event time for a single
/// cause: sample `u ~ U(0,1)` and root-find the augmented ODE for the first time
/// the cumulative hazard reaches `−log u` — the event-time analogue of the analytic
/// inverse-CDF in [`draw_tte_latent`]. Returns the crossing time, or `f64::MAX` (the
/// shared "did not fire" sentinel) when the hazard does not reach the threshold by
/// `horizon`, so the caller's window logic censors it exactly as for an analytic
/// cause.
///
/// `horizon` must be `Some` and finite — the `simulate` entry points validate this
/// (a drug-driven hazard can vanish, so there is no implicit window). Left
/// truncation (`entry_time > 0`) for an ODE hazard is a deferred follow-up and is
/// rejected up front, so the search starts at 0. A non-finite / non-monotone
/// (negative) hazard cannot yield a meaningful event time: rather than silently
/// censoring a broken model it **panics** — the fit path has an optimizer to steer
/// away via a sentinel NLL, but simulation does not.
#[cfg(feature = "survival")]
fn draw_ode_tte_latent<R: rand::Rng>(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
    chz_state: usize,
    horizon: Option<f64>,
    cmt: usize,
    rng: &mut R,
) -> f64 {
    let horizon = horizon.expect(
        "ODE-accumulated TTE simulation requires a finite horizon; the simulate entry \
         points validate this before sampling",
    );
    let u: f64 = rng.sample(rand::distr::Open01);
    let threshold = -u.ln();
    // Single-pass hazard integration (see `ode_accumulated_hazard`): the hazard
    // RHS `TIME` is honoured by the integrator clock; the PK-parameter snapshot
    // resolves the `TIME` built-in at the integration start (t=0). #610.
    let pk = (model.pk_param_fn)(theta, eta, &subject.covariates, 0.0);
    let ode = model
        .ode_spec
        .as_ref()
        .expect("ODE-accumulated hazard requires an [odes] block");
    match crate::ode::ode_solve_until_chz_threshold(
        ode, &pk.values, subject, chz_state, threshold, horizon,
    ) {
        crate::ode::ThresholdOutcome::Crossed(t) => t,
        crate::ode::ThresholdOutcome::CensoredAtHorizon => f64::MAX,
        crate::ode::ThresholdOutcome::SolveFailed(why) => panic!(
            "ODE-accumulated TTE simulation failed for subject '{}' (CMT={cmt}): {why}",
            subject.id
        ),
    }
}

/// Draw TTE event/censoring outcomes for a subject and append them to `results`.
///
/// Called from `api::simulate_inner_with_draw` after the Gaussian path. Each
/// `ObsRecord::Event` carries the subject's observation window in its `time`
/// field.
///
/// - **Single endpoint:** the drawn event is administratively right-censored at
///   the window when it would occur later, so simulated data reproduce the
///   design's censoring pattern rather than every draw being an event.
/// - **Competing risks** (≥2 TTE records, one per cause CMT — §3.6): a latent
///   time is drawn for each cause; the **earliest** is the observed event (its
///   CMT, `observed = true`) and every other cause is right-censored at that same
///   time (`observed = false`). The subject's administrative horizon is shared
///   across causes: a record carrying an observed event (`Exact`) imposes none
///   (+∞), so only an all-`RightCensored` subject is censored, at its common
///   window (the `max` of the per-cause windows — see the match arm). One row per
///   cause CMT is emitted, giving the cause-specific layout the data reader expects.
///
/// `horizon` (`[simulation] horizon`, #522), when `Some(h)`, **overrides** the
/// per-record [`observation_window`] for every cause: `h` becomes the shared
/// administrative censoring window regardless of each record's `event_type`. This
/// is what a competing-risks VPC needs — re-simulating event-bearing data
/// (`Exact` records, which on their own draw unbounded) then censors at the
/// *planned* study end `h` rather than at the data's own observed event times.
/// `None` keeps the per-record window. The override changes only the censoring
/// comparison, never the number of uniforms drawn, so the RNG sequence (and the
/// SSE characterization tests) are unaffected.
// Same flat (model, subject, params, replicate indices, RNG, output sink) shape
// as the sole caller `emit_subject_rows`, which carries this same allow: the args
// are heterogeneous with no cohesive struct to bundle, and splitting them would
// only diverge from that sibling.
#[allow(clippy::too_many_arguments)]
pub fn simulate_tte<R: rand::Rng>(
    model: &crate::types::CompiledModel,
    subject: &crate::types::Subject,
    theta: &[f64],
    eta: &[f64],
    draw: usize,
    sim: usize,
    horizon: Option<f64>,
    rng: &mut R,
    results: &mut Vec<crate::api::SimulationResult>,
) {
    // Each TTE cause (a record routed to a `Tte` endpoint) carries its observation
    // window and the kind of hazard that draws its latent event time. A subject may
    // also carry non-TTE records; those are skipped here.
    enum CauseKind {
        Analytic {
            family: HazardFamily,
            params: Vec<f64>,
            entry_time: f64,
        },
        Ode {
            chz_state: usize,
        },
    }

    let mut causes: Vec<(usize, f64, CauseKind)> = Vec::new(); // (cmt, window, kind)
    for record in &subject.obs_records {
        let ObsRecord::Event {
            cmt,
            entry_time,
            time,
            event_type,
        } = record;
        let Some(EndpointLikelihood::Tte { hazard, .. }) = model.endpoints.get(cmt) else {
            continue;
        };
        // With an explicit `horizon`, every cause shares it as the administrative
        // window (#522); otherwise only a right-censored record marks a horizon and
        // an event record draws uncensored (+∞) — see `observation_window`.
        let window = horizon.unwrap_or_else(|| observation_window(event_type, *time));
        let kind = match hazard {
            HazardSpec::Analytic { family, param_fn } => CauseKind::Analytic {
                family: *family,
                params: param_fn(theta, eta, &subject.covariates),
                entry_time: *entry_time,
            },
            HazardSpec::OdeAccumulated { chz_state } => CauseKind::Ode {
                chz_state: *chz_state,
            },
        };
        causes.push((*cmt, window, kind));
    }

    let push = |cmt: usize, time: f64, observed: bool, results: &mut Vec<_>| {
        results.push(crate::api::SimulationResult {
            draw,
            sim,
            id: subject.id.clone(),
            time,
            cmt,
            ipred: f64::NAN,
            outcome: SimOutcome::Event { time, observed },
        });
    };

    match causes.as_slice() {
        [] => {}
        // Single endpoint: administrative censoring at its own window. Analytic
        // stays on `draw_tte_outcome` (byte-identical RNG to the pre-ODE path); an
        // ODE hazard root-finds its latent, then applies the same window rule.
        [(cmt, window, kind)] => {
            let (t, observed) = match kind {
                CauseKind::Analytic {
                    family,
                    params,
                    entry_time,
                } => draw_tte_outcome(*family, params, *entry_time, *window, rng),
                CauseKind::Ode { chz_state } => {
                    let latent = draw_ode_tte_latent(
                        model, subject, theta, eta, *chz_state, horizon, *cmt, rng,
                    );
                    let observed = latent < *window && latent < f64::MAX;
                    (if observed { latent } else { *window }, observed)
                }
            };
            push(*cmt, t, observed, results);
        }
        // Competing risks: earliest latent event wins; the rest censor at that time.
        // Draw in record order so the RNG sequence stays deterministic across the mix.
        _ => {
            let latents: Vec<f64> = causes
                .iter()
                .map(|(cmt, _, kind)| match kind {
                    CauseKind::Analytic {
                        family,
                        params,
                        entry_time,
                    } => draw_tte_latent(*family, params, *entry_time, rng),
                    CauseKind::Ode { chz_state } => draw_ode_tte_latent(
                        model, subject, theta, eta, *chz_state, horizon, *cmt, rng,
                    ),
                })
                .collect();
            // Shared administrative horizon across causes. `observation_window`
            // already maps an event-bearing record (Exact / IntervalCensored) to
            // +∞ — it carries no horizon — so a subject with any observed cause
            // must draw uncensored (the #494 anti-truncation rule), and only an
            // all-`RightCensored` subject is censored, at its common window.
            // Folding with `max` realises exactly that: any +∞ wins ⇒ unbounded;
            // otherwise the (shared) censoring time. `min` would instead re-censor
            // a re-simulated event-bearing subject at its own observed event time
            // (the sibling cause's `RightCensored` window = the event time),
            // re-introducing the VPC bias #494 removed for single endpoints.
            let window = causes
                .iter()
                .map(|(_, w, _)| *w)
                .fold(f64::NEG_INFINITY, f64::max);
            let (win_idx, obs_time, event) = resolve_competing_risks(&latents, window);
            for (i, (cmt, ..)) in causes.iter().enumerate() {
                push(*cmt, obs_time, event && i == win_idx, results);
            }
        }
    }
}

/// Cause-specific cumulative incidence functions (CIF) from per-cause cumulative
/// hazards on a shared, **ascending** time grid (`cum_hazards[k][i] = H_k(t_i)`).
///
/// Uses the discrete Aalen–Johansen / actuarial allocation: at each grid step the
/// drop in all-cause survival `ΔS = S_all(t_{i−1}) − S_all(t_i)` is split across
/// causes in proportion to their cumulative-hazard increment share
/// `ΔH_k / Σ_j ΔH_j` over that step. This guarantees the exact invariant
/// `Σ_k F_k(t) + S_all(t) = 1` at every grid point (telescoping), reduces to
/// `F = 1 − S` for a single cause, and is **exact** (grid-independent) for
/// constant hazards. Because `ΔS` and `Σ_j ΔH_j` are derived from the *same*
/// cumulative hazards, `ΔS > 0 ⟺ Σ_j ΔH_j > 0`: the degenerate-node guard can
/// only skip a step whose `ΔS` is already 0, so the invariant never breaks.
///
/// `S_all(0) = 1` is assumed (parametric cumulative hazards vanish at `t = 0`), so
/// the running cumulative hazard starts at 0 and the lower integration limit is 0.
/// The grid must be ascending (the caller sorts it) so every increment is ≥ 0.
///
/// Returns `(cif[k][i], s_all[i])` where `s_all[i] = exp(−Σ_k H_k(t_i))`.
pub(crate) fn cif_curves(cum_hazards: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_cause = cum_hazards.len();
    let n_grid = cum_hazards.first().map_or(0, |h| h.len());

    let s_all: Vec<f64> = (0..n_grid)
        .map(|i| {
            let h_tot: f64 = (0..n_cause).map(|k| cum_hazards[k][i]).sum();
            (-h_tot).exp()
        })
        .collect();

    let mut cif = vec![vec![0.0_f64; n_grid]; n_cause];
    let mut acc = vec![0.0_f64; n_cause];
    let mut cum_prev = vec![0.0_f64; n_cause]; // H_k(0) = 0
    let mut s_prev = 1.0_f64; // S_all(0)
    for i in 0..n_grid {
        // ΔS_i ≥ 0, and the per-cause cumulative-hazard increment ΔH_k ≥ 0 on an
        // ascending grid. `dh_tot` and `drop` are both functions of the same
        // cumulative hazards, so `drop > 0 ⟺ dh_tot > 0`; the guard therefore only
        // skips a node whose ΔS is 0 (e.g. t = 0) — never one carrying mass.
        let drop = (s_prev - s_all[i]).max(0.0);
        let dh: Vec<f64> = (0..n_cause)
            .map(|k| (cum_hazards[k][i] - cum_prev[k]).max(0.0))
            .collect();
        let dh_tot: f64 = dh.iter().sum();
        if drop > 0.0 && dh_tot > 0.0 && dh_tot.is_finite() {
            for k in 0..n_cause {
                acc[k] += drop * dh[k] / dh_tot;
            }
        }
        for k in 0..n_cause {
            cif[k][i] = acc[k];
            cum_prev[k] = cum_hazards[k][i];
        }
        s_prev = s_all[i];
    }

    (cif, s_all)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests for FD Hessian accuracy
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fd_hessian_matches_analytic_quadratic() {
        // f(η) = a·η₀² + b·η₁² + c·η₀·η₁
        // Hessian = [[2a, c], [c, 2b]] — exact, no approximation error.
        let a = 3.0_f64;
        let b = 2.0_f64;
        let c = 1.5_f64;
        let eval = move |e: &[f64]| a * e[0] * e[0] + b * e[1] * e[1] + c * e[0] * e[1];
        let eta = &[0.1, -0.2];
        let eps = &[1e-4, 1e-4];
        let h = data_term_hessian_fd(eval, eta, eps);
        assert_abs_diff_eq!(h[(0, 0)], 2.0 * a, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(1, 1)], 2.0 * b, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(0, 1)], c, epsilon = 1e-6);
        assert_abs_diff_eq!(h[(1, 0)], c, epsilon = 1e-6);
    }

    #[test]
    fn fd_hessian_scalar_eta() {
        // f(η) = η² / 2 → f''(η) = 1.0
        let eval = |e: &[f64]| 0.5 * e[0] * e[0];
        let eta = &[0.5];
        let eps = &[1e-4];
        let h = data_term_hessian_fd(eval, eta, eps);
        assert_abs_diff_eq!(h[(0, 0)], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn tte_data_term_right_censored_exponential() {
        use crate::types::HazardFamily;
        // Simple: lambda=0.1, T=10, entry=0 → H(T) = 1.0
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let theta = &[0.1_f64];
        let eta = &[0.0_f64];
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, theta, eta, &cov);
        // -log L = H(T) = 0.1 * 10 = 1.0
        assert_abs_diff_eq!(nll, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn tte_cause_params_evaluates_tte_and_skips_non_tte() {
        use crate::types::{EndpointError, ErrorModel, HazardFamily, HazardParamFn};
        let theta = [0.1_f64];
        let eta = [0.0_f64];
        let cov = HashMap::new();

        // A `Tte` endpoint yields `Some((family, evaluated params))`.
        let param_fn: HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let tte = EndpointLikelihood::Tte {
            hazard: HazardSpec::Analytic {
                family: HazardFamily::Exponential,
                param_fn,
            },
            recurrence: TteRecurrence::Single,
        };
        let (family, params) =
            tte_cause_params(&tte, &theta, &eta, &cov).expect("Tte endpoint must yield Some");
        assert_eq!(family, HazardFamily::Exponential);
        assert_eq!(params, vec![0.1]);

        // A non-`Tte` (Gaussian) endpoint takes the `None` branch, letting callers
        // `filter_map`/`continue` over a mixed endpoint map.
        let gaussian = EndpointLikelihood::Gaussian(EndpointError {
            error_model: ErrorModel::Additive,
            sigma_idx: vec![],
        });
        assert!(tte_cause_params(&gaussian, &theta, &eta, &cov).is_none());

        // An ODE-accumulated hazard has no closed-form cause params either (its H/h come
        // from the integrated CHZ state), so it also takes the `None` branch.
        let ode = EndpointLikelihood::Tte {
            hazard: HazardSpec::OdeAccumulated { chz_state: 2 },
            recurrence: TteRecurrence::Single,
        };
        assert!(tte_cause_params(&ode, &theta, &eta, &cov).is_none());
    }

    #[test]
    fn tte_data_term_ode_accumulated_is_analytic_only_sentinel() {
        // `tte_data_term` is the closed-form (Analytic) entry point. ODE-accumulated
        // hazards are dispatched elsewhere (tte_endpoint_nll → tte_ode_nll), so calling
        // this with an OdeAccumulated hazard returns the ill-defined sentinel rather than
        // a real likelihood — documenting (and covering) the dispatch contract.
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let hazard = HazardSpec::OdeAccumulated { chz_state: 2 };
        let cov = HashMap::new();
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &cov,
        );
        assert_eq!(nll, 1e20);
    }

    #[test]
    fn tte_nll_from_curves_degenerate_interval_returns_sentinel() {
        use crate::types::HazardFamily;
        // An interval with right ≤ left gives a non-positive Δ = H(right) − H(left); the
        // shared per-record path returns the 1e20 sentinel (covers the degenerate guard).
        let records = vec![ObsRecord::Event {
            time: 0.0,
            event_type: EventType::IntervalCensored {
                left: 10.0,
                right: 5.0,
            },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let cov = HashMap::new();
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &cov,
        );
        assert_eq!(nll, 1e20);
    }

    /// A right-censored record whose cumulative hazard *decreased* (`H(T) < H(entry)`,
    /// i.e. a negative / non-monotone drug-driven hazard) is ill-posed — `S(T) =
    /// exp(−ΔH) > 1`. It must hit the 1e20 sentinel, not contribute a spurious
    /// *negative* NLL that pulls the optimizer toward the negative-hazard region (the
    /// simulation path hard-errors on the same non-monotone CHZ; #564 Slice 2.2 review).
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_censored() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 3,
        }];
        // CHZ is negative by t=5 (entry H=0): a clear monotonicity violation.
        let nll = tte_nll_from_curves(&records, |_t| -0.5, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, 1e20,
            "non-monotone CHZ on a censor must be sentinel-guarded"
        );
    }

    /// The guard also covers an exact event with a non-monotone cumulative hazard:
    /// a positive instantaneous `h(T)` is not enough — the accumulated `H` must be
    /// non-decreasing too.
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_exact() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 3,
        }];
        let nll = tte_nll_from_curves(&records, |_t| -0.5, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, 1e20,
            "non-monotone CHZ on an exact event must be sentinel-guarded"
        );
    }

    /// The monotonicity guard tolerates ODE quadrature round-off on a (near-)flat
    /// cumulative hazard: a tiny negative dip within the relative floor is NOT
    /// rejected, so a legitimate `h ≈ 0` model keeps a finite likelihood (guards the
    /// floor against false positives that would derail a valid fit).
    #[test]
    fn tte_nll_from_curves_tolerates_flat_hazard_roundoff() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 3,
        }];
        // H(5) = −1e−9, below the 1e−6 absolute floor ⇒ accepted, not sentinel.
        let nll = tte_nll_from_curves(&records, |_t| -1e-9, |_t| 0.0, MonoTol::default());
        assert!(
            nll.abs() < 1e-6,
            "round-off dip must stay finite, got {nll}"
        );
    }

    /// And for an interval-censored record whose cumulative hazard is non-monotone
    /// *before* the interval (`H(left) < H(entry)`): the guard fires before the
    /// `delta` check, so this exercises the interval arm's monotonicity branch
    /// (distinct from the degenerate `right ≤ left` case above).
    #[test]
    fn tte_nll_from_curves_rejects_non_monotone_interval() {
        let records = vec![ObsRecord::Event {
            time: 0.0,
            event_type: EventType::IntervalCensored {
                left: 5.0,
                right: 10.0,
            },
            entry_time: 2.0,
            cmt: 3,
        }];
        // H(entry=2) = 1.0 but H(left=5) = 0.2 < H(entry): non-monotone ⇒ sentinel.
        let nll = tte_nll_from_curves(
            &records,
            |t| if t < 3.0 { 1.0 } else { 0.2 },
            |_t| 0.1,
            MonoTol::default(),
        );
        assert_eq!(
            nll, 1e20,
            "non-monotone CHZ before an interval must be sentinel-guarded"
        );
    }

    /// #618: with a large accumulated `H`, a *genuine* negative increment (here 0.05% of
    /// `H` - between the solver `reltol` and the old `1e-3` term) must hit the sentinel.
    /// The previous magnitude-relative `1e-3·|H|` floor accepted it as round-off and let
    /// the optimizer see a spurious finite (negative) NLL.
    #[test]
    fn tte_nll_from_curves_rejects_genuine_negative_increment_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        // H(entry=2) = 100, H(5) = 99.95 ⇒ ΔH = −0.05 = −0.05% of H. Old floor
        // 1e-3·100 = 0.1 accepted it; new floor reltol·100 = 0.01 rejects it.
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.95 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, 1e20,
            "genuine negative increment on a large H must be sentinel-guarded"
        );
    }

    /// And the exact-event arm: a positive instantaneous `h(T)` does not excuse a
    /// genuinely negative accumulated increment on a large `H` (#618).
    #[test]
    fn tte_nll_from_curves_exact_rejects_genuine_negative_increment_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::Exact,
            entry_time: 2.0,
            cmt: 3,
        }];
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.95 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert_eq!(
            nll, 1e20,
            "exact-event genuine negative increment must be sentinel-guarded"
        );
    }

    /// The tightened floor must NOT over-reject: a dip *within* the solver's round-off
    /// band (ΔH = −0.005 on H = 100, i.e. 0.005% - below the reltol·100 = 0.01 floor)
    /// stays finite, so a legitimate fit is not derailed by quadrature noise.
    #[test]
    fn tte_nll_from_curves_tolerates_solver_band_dip_large_h() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        let cumhaz = |t: f64| if t <= 2.0 { 100.0 } else { 99.995 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert!(
            nll.is_finite() && nll < 1e20,
            "within-band round-off dip must stay finite, got {nll}"
        );
    }

    /// The analytic path uses a tight fixed floor ([`MonoTol::analytic`]): a small
    /// negative increment that the ODE-sized default floor would swallow is still
    /// rejected, so a sign-flipped closed-form hazard cannot sneak `S > 1` through.
    #[test]
    fn tte_nll_from_curves_analytic_floor_rejects_small_dip() {
        let records = vec![ObsRecord::Event {
            time: 5.0,
            event_type: EventType::RightCensored,
            entry_time: 2.0,
            cmt: 3,
        }];
        // ΔH = −1e−5 on H ≈ 1: above the analytic ~1e−9 floor (rejected) but below the
        // default ODE floor 1e−6 + reltol·1 ≈ 1e−4 (which tolerates it as round-off).
        let cumhaz = |t: f64| if t <= 2.0 { 1.0 } else { 1.0 - 1e-5 };
        let nll = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::analytic());
        assert_eq!(
            nll, 1e20,
            "analytic floor must reject a small genuine negative dip"
        );
        let nll_default = tte_nll_from_curves(&records, cumhaz, |_t| 0.1, MonoTol::default());
        assert!(
            nll_default.is_finite() && nll_default < 1e20,
            "same dip is within the ODE-sized default floor, got {nll_default}"
        );
    }

    #[test]
    fn tte_data_term_exact_event_exponential() {
        use crate::types::HazardFamily;
        // lambda=0.1, T=10, exact event → -log L = H(T) - log h(T) = 1.0 - log(0.1) = 1.0 + 2.303
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let theta = &[0.1_f64];
        let eta = &[0.0_f64];
        let cov = HashMap::new();
        let nll = tte_data_term(&records, &hazard, TteRecurrence::Single, theta, eta, &cov);
        let expected = 0.1 * 10.0 - (0.1_f64).ln(); // H - log h
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-10);
    }

    #[test]
    fn tte_data_term_left_truncation() {
        use crate::types::HazardFamily;
        // Exponential, entry=5, T=10, right-censored → H(T)-H(entry) = 0.1*(10-5) = 0.5
        let records = vec![ObsRecord::Event {
            time: 10.0,
            event_type: EventType::RightCensored,
            entry_time: 5.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _eta: &[f64], _cov: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &HashMap::new(),
        );
        assert_abs_diff_eq!(nll, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn tte_data_term_interval_censored_exponential() {
        use crate::types::HazardFamily;
        // Exponential, lambda=0.2, interval (3, 5), entry=0.
        // H(3)=0.6, H(5)=1.0, delta=0.4
        // log_prob = −0.6 + log(1 − exp(−0.4)) = −0.6 + log(0.32968...) = −1.70881...
        // nll = 1.70881...
        let records = vec![ObsRecord::Event {
            time: 5.0, // right bound (time field = right for interval-censored)
            event_type: EventType::IntervalCensored {
                left: 3.0,
                right: 5.0,
            },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.2],
            &[0.0],
            &HashMap::new(),
        );
        let expected = -(-(0.6_f64) + ((-0.4_f64).exp_m1().abs().ln()));
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-10);
        assert!(nll > 0.0 && nll.is_finite());
    }

    #[test]
    fn tte_data_term_interval_censored_tight_interval() {
        use crate::types::HazardFamily;
        // Tight interval: left=10.0, right=10.0001, lambda=0.1.
        // Old probability-space subtraction would lose ~4 significant digits here.
        // The expm1 form should stay finite and positive.
        let records = vec![ObsRecord::Event {
            time: 10.0001,
            event_type: EventType::IntervalCensored {
                left: 10.0,
                right: 10.0001,
            },
            entry_time: 0.0,
            cmt: 2,
        }];
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let nll = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Single,
            &[0.1],
            &[0.0],
            &HashMap::new(),
        );
        // NLL must be finite and positive; exact value verified via log-domain formula.
        assert!(nll.is_finite() && nll > 0.0, "nll = {nll}");
        let delta = 0.1_f64 * 0.0001; // H(right) - H(left)
        let a = 0.1_f64 * 10.0; // H(left) - H(entry)
        let expected = a - ((-delta).exp_m1().abs().ln());
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-8);
    }

    // ── RTTE clock-forward: telescoping cumulative hazard ──

    // Three records for one subject at constant hazard h = λ = 0.05: events at
    // t = 5 and t = 10, administrative right-censor at t = 30. Curves are closed
    // form: H(t) = λt, h(t) = λ.
    fn rtte_constant_hazard_records() -> Vec<ObsRecord> {
        vec![
            ObsRecord::Event {
                time: 5.0,
                event_type: EventType::Exact,
                entry_time: 0.0,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 10.0,
                event_type: EventType::Exact,
                entry_time: 0.0,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 30.0,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            },
        ]
    }

    #[test]
    fn rtte_forward_telescopes_cumulative_hazard() {
        // Clock-forward RTTE (Andersen–Gill) integrates the cumulative hazard ONCE to
        // the final time: −log L = H(T) − Σ_k log h(t_k) = 30λ − 2·log λ. Summing
        // independent single-event terms (as the standard-TTE per-record accumulator
        // does) would instead give Σ_k H(t_k) + H(T) − 2·log λ = 45λ − 2·log λ, an
        // over-count of Σ_k H(t_k) = 15λ. This test pins the correct telescoping.
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;

        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_rtte = 30.0 * lambda - 2.0 * lambda.ln(); // H(T) − Σ log h
        assert_abs_diff_eq!(rtte, expected_rtte, epsilon = 1e-12);

        // Contrast: the single-event per-record accumulator over-counts by Σ H(t_k).
        let independent = tte_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_independent = 45.0 * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(independent, expected_independent, epsilon = 1e-12);
        assert!(
            (independent - rtte - 15.0 * lambda).abs() < 1e-12,
            "the over-count must be exactly Σ_k H(t_k) = 15λ (rtte={rtte}, indep={independent})"
        );
    }

    #[test]
    fn rtte_forward_single_event_matches_standard_tte() {
        // With exactly one event record, the recurrent and single-event likelihoods
        // coincide (nothing to telescope) — the RTTE path must not diverge for K=1.
        let lambda = 0.05_f64;
        let records = vec![ObsRecord::Event {
            time: 7.0,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        }];
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let single = tte_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(rtte, single, epsilon = 1e-12);
    }

    #[test]
    fn rtte_forward_left_truncation_uses_first_entry() {
        // Delayed entry at t = 2 on the first record: the lower integration limit is
        // H(entry), so −log L = H(T) − H(entry) − Σ log h = (30−2)λ − 2·log λ.
        let lambda = 0.05_f64;
        // Same event/censor times as `rtte_constant_hazard_records`, but the subject
        // enters the risk set at t = 2 (left truncation carried on the first record).
        let records = vec![
            ObsRecord::Event {
                time: 5.0,
                event_type: EventType::Exact,
                entry_time: 2.0,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 10.0,
                event_type: EventType::Exact,
                entry_time: 2.0,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 30.0,
                event_type: EventType::RightCensored,
                entry_time: 2.0,
                cmt: 2,
            },
        ];
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let rtte =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected = (30.0 - 2.0) * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(rtte, expected, epsilon = 1e-12);
    }

    #[test]
    fn rtte_forward_rejects_non_monotone_hazard() {
        // A non-monotone (negative-hazard) cumulative hazard makes an increment
        // negative — S = exp(−ΔH) > 1, ill-posed. Must fold into the 1e20 sentinel,
        // exactly like the single-event path.
        let records = rtte_constant_hazard_records();
        // Cumulative hazard that decreases after t = 6 (negative hazard region).
        let cumhaz_at = |t: f64| if t < 6.0 { t } else { 12.0 - t };
        let hazard_at = |_t: f64| 0.1;
        let nll = rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_eq!(nll, 1e20, "non-monotone CHZ must return the sentinel");
    }

    #[test]
    fn tte_data_term_dispatches_rtte_forward() {
        use crate::types::{HazardFamily, RtteClock, TteRecurrence};
        // Route a repeated-event subject through the public `tte_data_term` entry with
        // an analytic exponential family and `Repeated { Forward }`; the result must
        // match the telescoping curve helper (i.e. the dispatch is wired).
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let via_dispatch = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Repeated {
                clock: RtteClock::Forward,
            },
            &[lambda],
            &[0.0],
            &HashMap::new(),
        );
        let expected = 30.0 * lambda - 2.0 * lambda.ln();
        assert_abs_diff_eq!(via_dispatch, expected, epsilon = 1e-12);
    }

    // ── RTTE clock-reset (gap time): renewal likelihood ──

    #[test]
    fn rtte_reset_equals_forward_for_exponential() {
        // A constant hazard is memoryless, so clock-reset (gap time) and clock-forward
        // (total time) must give the SAME likelihood — the sharpest check that the reset
        // gap bookkeeping is right.
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| lambda * t;
        let hazard_at = |_t: f64| lambda;
        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        assert_abs_diff_eq!(reset, forward, epsilon = 1e-12);
        // And both equal the closed form 30λ − 2·log λ.
        assert_abs_diff_eq!(reset, 30.0 * lambda - 2.0 * lambda.ln(), epsilon = 1e-12);
    }

    #[test]
    fn rtte_reset_weibull_gap_time_closed_form() {
        // Weibull scale=10, shape=2: H(t) = (t/10)^2, h(t) = 0.2·(t/10). Events at 5 and
        // 10, censor at 30. Clock-RESET evaluates each gap from 0: gaps are 5, 5, 20.
        //   event  (Δ=5):  H(5)  − log h(5)  = 0.25 − log(0.1)
        //   event  (Δ=5):  H(5)  − log h(5)  = 0.25 − log(0.1)   (clock reset ⇒ same as above)
        //   censor (Δ=20): H(20)             = 4.0
        // A time-varying hazard makes this DIFFER from clock-forward (which would use the
        // absolute event times), so the two must not coincide here.
        let records = rtte_constant_hazard_records();
        let cumhaz_at = |t: f64| (t / 10.0).powi(2);
        let hazard_at = |t: f64| 0.2 * (t / 10.0);

        let reset = rtte_reset_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_reset = 2.0 * (0.25 - 0.1_f64.ln()) + 4.0;
        assert_abs_diff_eq!(reset, expected_reset, epsilon = 1e-12);

        let forward =
            rtte_forward_nll_from_curves(&records, cumhaz_at, hazard_at, MonoTol::analytic());
        let expected_forward = 9.0 - 0.1_f64.ln() - 0.2_f64.ln(); // H(30) − log h(5) − log h(10)
        assert_abs_diff_eq!(forward, expected_forward, epsilon = 1e-12);
        assert!(
            (reset - forward).abs() > 1.0,
            "reset ({reset}) and forward ({forward}) must differ for a time-varying hazard"
        );
    }

    #[test]
    fn tte_data_term_dispatches_rtte_reset() {
        use crate::types::{HazardFamily, RtteClock, TteRecurrence};
        // Route a repeated-event subject through `tte_data_term` with `Repeated { Reset }`
        // and an exponential family; must match the reset curve helper (dispatch wired).
        let lambda = 0.05_f64;
        let records = rtte_constant_hazard_records();
        let param_fn: crate::types::HazardParamFn =
            Box::new(|theta: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![theta[0]]);
        let hazard = HazardSpec::Analytic {
            family: HazardFamily::Exponential,
            param_fn,
        };
        let via_dispatch = tte_data_term(
            &records,
            &hazard,
            TteRecurrence::Repeated {
                clock: RtteClock::Reset,
            },
            &[lambda],
            &[0.0],
            &HashMap::new(),
        );
        // Exponential ⇒ reset == forward == 30λ − 2·log λ.
        assert_abs_diff_eq!(
            via_dispatch,
            30.0 * lambda - 2.0 * lambda.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn rtte_reset_folds_unsupported_inputs_to_sentinel() {
        // Every ill-posed input to the gap-time renewal likelihood must fold into the
        // 1e20 sentinel (a finite-but-poison objective the optimizer steers away from),
        // never a finite silently-wrong contribution — matching the forward and
        // single-event paths. This is the reset counterpart to the forward slice's
        // `rtte_forward_rejects_non_monotone_hazard`, exercising each guard return in
        // `rtte_reset_nll_from_curves` (Δ<0, h≤0, non-monotone H on either arm, interval).
        let exact = |t: f64| ObsRecord::Event {
            time: t,
            event_type: EventType::Exact,
            entry_time: 0.0,
            cmt: 2,
        };
        let censor = |t: f64| ObsRecord::Event {
            time: t,
            event_type: EventType::RightCensored,
            entry_time: 0.0,
            cmt: 2,
        };
        let tol = MonoTol::analytic();
        let lin = |t: f64| 0.1 * t; // proper increasing H(Δ)
        let good_h = |_t: f64| 0.1_f64;
        // Decreasing after t = 3, so H(Δ) < 0 on a gap of 5 (non-monotone / negative hazard).
        let dec = |t: f64| if t < 3.0 { t } else { 1.0 - t };

        // Non-positive hazard at an Exact event → sentinel (Exact arm, h ≤ 0).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(5.0)], lin, |_t| 0.0, tol),
            1e20
        );
        // Non-monotone cumulative hazard on an Exact gap → sentinel (Exact arm).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(5.0)], dec, good_h, tol),
            1e20
        );
        // Non-monotone cumulative hazard on a censored gap → sentinel (RightCensored arm).
        assert_eq!(
            rtte_reset_nll_from_curves(&[censor(5.0)], dec, good_h, tol),
            1e20
        );
        // Out-of-order records → negative gap → sentinel (Δ < 0 guard).
        assert_eq!(
            rtte_reset_nll_from_curves(&[exact(10.0), exact(4.0)], lin, good_h, tol),
            1e20
        );
        // Interval-censored record is unsupported for RTTE → sentinel.
        let interval = ObsRecord::Event {
            time: 5.0,
            event_type: EventType::IntervalCensored {
                left: 3.0,
                right: 5.0,
            },
            entry_time: 0.0,
            cmt: 2,
        };
        assert_eq!(
            rtte_reset_nll_from_curves(&[interval], lin, good_h, tol),
            1e20
        );
    }

    // ── draw_tte_outcome: administrative censoring at the observation window ──

    #[test]
    fn draw_tte_censoring_fraction_matches_survival() {
        use rand::SeedableRng;
        // Exponential λ=0.1 over a window τ=10 ⇒ P(censored) = S(τ) = exp(−λτ) ≈ 0.3679.
        // 20 000 draws; proportion SE ≈ 0.0034 ⇒ a 0.02 tolerance is ~6σ.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let lambda = 0.1_f64;
        let window = 10.0_f64;
        let n = 20_000;
        let mut censored = 0usize;
        for _ in 0..n {
            let (t, observed) =
                draw_tte_outcome(HazardFamily::Exponential, &[lambda], 0.0, window, &mut rng);
            if observed {
                assert!(
                    t > 0.0 && t < window,
                    "event time must lie in (0, window): {t}"
                );
            } else {
                assert_eq!(t, window, "censored time must equal the window");
                censored += 1;
            }
        }
        let frac = censored as f64 / n as f64;
        let expected = (-lambda * window).exp();
        assert!(
            (frac - expected).abs() < 0.02,
            "censoring fraction {frac} should track S(τ) = {expected}"
        );
    }

    #[test]
    fn draw_tte_infinite_window_never_censors() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..1000 {
            let (t, observed) = draw_tte_outcome(
                HazardFamily::Exponential,
                &[1.0],
                0.0,
                f64::INFINITY,
                &mut rng,
            );
            assert!(observed, "no censoring is possible with an infinite window");
            assert!(
                t.is_finite() && t > 0.0,
                "event time must be finite positive: {t}"
            );
        }
    }

    #[test]
    fn draw_tte_degenerate_zero_hazard_censors_at_window() {
        use rand::SeedableRng;
        // λ=0 ⇒ sampler returns f64::MAX; with a finite window the draw must
        // censor at the window rather than emit a spurious event at f64::MAX.
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let window = 12.5_f64;
        for _ in 0..100 {
            let (t, observed) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.0], 0.0, window, &mut rng);
            assert!(!observed, "zero hazard can never produce an event");
            assert_eq!(t, window);
        }
    }

    #[test]
    fn draw_tte_left_truncation_respects_entry_and_window() {
        use rand::SeedableRng;
        // entry=5, window=8, λ=0.2 (memoryless): P(event before window)
        // = 1 − exp(−0.2·3) ≈ 0.45 ⇒ both outcomes appear in 5 000 draws.
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let (entry, window) = (5.0_f64, 8.0_f64);
        let (mut saw_event, mut saw_censor) = (false, false);
        for _ in 0..5000 {
            let (t, observed) =
                draw_tte_outcome(HazardFamily::Exponential, &[0.2], entry, window, &mut rng);
            if observed {
                assert!(
                    t > entry && t < window,
                    "conditional event in (entry, window): {t}"
                );
                saw_event = true;
            } else {
                assert_eq!(t, window);
                saw_censor = true;
            }
        }
        assert!(
            saw_event && saw_censor,
            "expected a mix of events and censored draws"
        );
    }

    #[test]
    fn draw_tte_unbounded_window_degenerate_does_not_emit_spurious_event() {
        use rand::SeedableRng;
        // λ=0 ⇒ sampler returns f64::MAX. With an unbounded window (an event
        // record carries no administrative horizon, so `observation_window`
        // returns +∞) a bare `t_event < window` test would mis-report that
        // sentinel as an observed event at f64::MAX. The `t_event < f64::MAX`
        // guard must classify it as censored (no event) instead.
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for _ in 0..100 {
            let (t, observed) = draw_tte_outcome(
                HazardFamily::Exponential,
                &[0.0],
                0.0,
                f64::INFINITY,
                &mut rng,
            );
            assert!(
                !observed,
                "a degenerate (no-event) draw must never be reported as an observed event"
            );
            assert_eq!(t, f64::INFINITY, "censored at the (unbounded) window");
        }
    }

    // ── resolve_competing_risks: earliest cause wins, else administrative censor ──

    #[test]
    fn resolve_competing_risks_earliest_wins() {
        // Cause 1 (index 1) is earliest at t=3 < window=10 ⇒ observed event.
        let (idx, t, event) = resolve_competing_risks(&[5.0, 3.0, 8.0], 10.0);
        assert_eq!(idx, 1);
        assert_eq!(t, 3.0);
        assert!(event);
    }

    #[test]
    fn resolve_competing_risks_all_after_window_censors() {
        // No cause fires before the window ⇒ censored at the window, no event.
        let (idx, t, event) = resolve_competing_risks(&[12.0, 15.0], 10.0);
        assert_eq!(idx, 0, "winner is still the earliest latent");
        assert_eq!(t, 10.0);
        assert!(!event);
    }

    #[test]
    fn resolve_competing_risks_max_sentinel_never_wins() {
        // f64::MAX (degenerate cause) must not win over a real event.
        let (idx, t, event) = resolve_competing_risks(&[f64::MAX, 4.0], 10.0);
        assert_eq!(idx, 1);
        assert_eq!(t, 4.0);
        assert!(event);
    }

    #[test]
    fn resolve_competing_risks_tie_picks_lowest_index() {
        let (idx, t, event) = resolve_competing_risks(&[3.0, 3.0], 10.0);
        assert_eq!(idx, 0, "ties resolve to the lowest cause index");
        assert_eq!(t, 3.0);
        assert!(event);
    }

    // ── cif_curves: cause-specific cumulative incidence ──────────────────────

    #[test]
    fn cif_two_cause_exponential_closed_form() {
        // Constant hazards λ1=0.1, λ2=0.3 ⇒ closed form (grid-independent):
        //   F_k(t) = (λ_k/Σλ)·(1 − exp(−Σλ·t)),  S_all(t) = exp(−Σλ·t).
        let (l1, l2) = (0.1_f64, 0.3_f64);
        let grid = [0.0, 2.0, 5.0, 13.0, 40.0];
        let cum: Vec<Vec<f64>> = vec![
            grid.iter().map(|&t| l1 * t).collect(),
            grid.iter().map(|&t| l2 * t).collect(),
        ];

        let (cif, s_all) = cif_curves(&cum);
        let lsum = l1 + l2;
        for (i, &t) in grid.iter().enumerate() {
            let s_ref = (-lsum * t).exp();
            assert_abs_diff_eq!(s_all[i], s_ref, epsilon = 1e-12);
            assert_abs_diff_eq!(cif[0][i], (l1 / lsum) * (1.0 - s_ref), epsilon = 1e-12);
            assert_abs_diff_eq!(cif[1][i], (l2 / lsum) * (1.0 - s_ref), epsilon = 1e-12);
            // Invariant must hold exactly at every grid point.
            assert_abs_diff_eq!(cif[0][i] + cif[1][i] + s_all[i], 1.0, epsilon = 1e-12);
        }
        // t = 0 ⇒ no incidence, full survival.
        assert_eq!(cif[0][0], 0.0);
        assert_eq!(cif[1][0], 0.0);
        assert_eq!(s_all[0], 1.0);
    }

    #[test]
    fn cif_invariant_holds_for_mixed_families() {
        // Weibull-like increasing hazard + a constant hazard on a coarse,
        // irregular grid. The per-cause CIF is only approximate here, but the
        // partition invariant Σ_k F_k + S_all = 1 must be exact (telescoping).
        let grid = [1.0_f64, 2.5, 4.0, 9.0, 20.0];
        // cause 0: Weibull scale=8, shape=1.7 ⇒ H=(t/8)^1.7
        let (sc, sh) = (8.0_f64, 1.7_f64);
        let cum0: Vec<f64> = grid.iter().map(|&t| (t / sc).powf(sh)).collect();
        // cause 1: constant λ=0.05 ⇒ H=0.05·t
        let cum1: Vec<f64> = grid.iter().map(|&t| 0.05 * t).collect();

        let (cif, s_all) = cif_curves(&[cum0, cum1]);
        for i in 0..grid.len() {
            assert_abs_diff_eq!(cif[0][i] + cif[1][i] + s_all[i], 1.0, epsilon = 1e-12);
            assert!(cif[0][i] >= 0.0 && cif[1][i] >= 0.0, "CIF non-negative");
            if i > 0 {
                assert!(cif[0][i] >= cif[0][i - 1] - 1e-15, "CIF non-decreasing");
                assert!(cif[1][i] >= cif[1][i - 1] - 1e-15, "CIF non-decreasing");
            }
        }
    }

    #[test]
    fn cif_time_varying_matches_independent_quadrature() {
        // Numeric anchor for the *time-varying* per-cause CIF, which the partition
        // invariant alone does not pin down (the constant-hazard values are pinned
        // by `cif_two_cause_exponential_closed_form`). Two increasing hazards —
        // Weibull + Gompertz — are compared against an INDEPENDENT high-resolution
        // composite-Simpson integration of the defining integral
        //   F_k(t) = ∫₀ᵗ h_k(u)·S_all(u) du,   S_all(u) = exp(−Σ_j H_j(u)),
        // which uses a different numerical scheme (direct integrand quadrature)
        // than `cif_curves`' actuarial ΔS-allocation, so agreement is a genuine
        // cross-check rather than a restatement. This is the closed-form-free
        // analogue of an R `cmprsk::cuminc` cross-check (see docs/estimation/tte.qmd).
        let (sc, sh) = (10.0_f64, 1.6_f64); // cause 0: Weibull  H=(t/sc)^sh
        let (alpha, gamma) = (0.03_f64, 0.08_f64); // cause 1: Gompertz H=(α/γ)(e^{γt}−1)
        let h0 = |t: f64| (sh / sc) * (t / sc).powf(sh - 1.0);
        let cumh0 = |t: f64| (t / sc).powf(sh);
        let h1 = |t: f64| alpha * (gamma * t).exp();
        let cumh1 = |t: f64| (alpha / gamma) * ((gamma * t).exp() - 1.0);
        let s_all = |t: f64| (-(cumh0(t) + cumh1(t))).exp();

        // Independent reference: composite Simpson (N even) of ∫₀ᵗ h_k·S_all.
        let ref_cif = |hk: &dyn Fn(f64) -> f64, t: f64| -> f64 {
            let n = 60_000usize;
            let dt = t / n as f64;
            let mut acc = 0.0;
            for i in 0..=n {
                let u = i as f64 * dt;
                let w = if i == 0 || i == n {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                acc += w * hk(u) * s_all(u);
            }
            acc * dt / 3.0
        };

        // `cif_curves` on a fine grid over [0, 15] (dt = 0.0025).
        let n_grid = 6000usize;
        let grid: Vec<f64> = (0..=n_grid)
            .map(|i| i as f64 * 15.0 / n_grid as f64)
            .collect();
        let chz = vec![
            grid.iter().map(|&t| cumh0(t)).collect::<Vec<_>>(),
            grid.iter().map(|&t| cumh1(t)).collect::<Vec<_>>(),
        ];
        let (cif, s) = cif_curves(&chz);

        // Check at five interior/boundary points (exact grid nodes).
        for &idx in &[1200usize, 2400, 3600, 4800, 6000] {
            let t = grid[idx];
            assert_abs_diff_eq!(s[idx], s_all(t), epsilon = 1e-12);
            assert_abs_diff_eq!(cif[0][idx], ref_cif(&h0, t), epsilon = 1e-3);
            assert_abs_diff_eq!(cif[1][idx], ref_cif(&h1, t), epsilon = 1e-3);
            // Sanity: still a valid partition.
            assert_abs_diff_eq!(cif[0][idx] + cif[1][idx] + s[idx], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn observation_window_only_right_censored_has_a_horizon() {
        // A right-censored record's `time` IS the administrative horizon.
        assert_eq!(observation_window(&EventType::RightCensored, 12.0), 12.0);
        // Event records carry no administrative horizon: their `time` is an
        // event time / interval bound, so they must draw uncensored (+∞) rather
        // than truncate at a realized event time.
        assert_eq!(observation_window(&EventType::Exact, 12.0), f64::INFINITY);
        assert_eq!(
            observation_window(
                &EventType::IntervalCensored {
                    left: 1.0,
                    right: 5.0
                },
                5.0
            ),
            f64::INFINITY
        );
    }

    #[test]
    fn cif_single_cause_is_one_minus_survival() {
        // One cause ⇒ CIF = 1 − S_all = 1 − S, invariant trivially holds.
        let grid = [0.0, 1.0, 4.0, 12.0];
        let lam = 0.2_f64;
        let cum = vec![grid.iter().map(|&t| lam * t).collect::<Vec<_>>()];
        let (cif, s_all) = cif_curves(&cum);
        for i in 0..grid.len() {
            assert_abs_diff_eq!(cif[0][i], 1.0 - s_all[i], epsilon = 1e-12);
        }
    }
}
