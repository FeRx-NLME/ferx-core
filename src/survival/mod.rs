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
    EndpointLikelihood, EventType, HazardFamily, HazardSpec, ObsRecord, SimOutcome,
};

// ─────────────────────────────────────────────────────────────────────────────
//  TTE data term
// ─────────────────────────────────────────────────────────────────────────────

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
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
) -> f64 {
    let HazardSpec::Analytic { family, param_fn } = hazard;
    let params = param_fn(theta, eta, covariates);

    let mut nll = 0.0_f64;

    for record in records {
        let ObsRecord::Event {
            time,
            event_type,
            entry_time,
            ..
        } = record;

        let h_entry = if *entry_time > 0.0 {
            cum_hazard(*family, *entry_time, &params)
        } else {
            0.0
        };

        match event_type {
            EventType::RightCensored => {
                let h_t = cum_hazard(*family, *time, &params);
                nll += h_t - h_entry;
            }
            EventType::Exact => {
                let (h_val, h_t) = hazard_and_cum_hazard(*family, *time, &params);
                if h_val <= 0.0 {
                    return 1e20;
                }
                nll += h_t - h_entry - h_val.ln();
            }
            EventType::IntervalCensored { left, right } => {
                let h_l = cum_hazard(*family, *left, &params);
                let h_r = cum_hazard(*family, *right, &params);
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
    // Gather this subject's TTE causes — records routed to a `Tte` endpoint —
    // with each cause's drawn-parameter vector, entry time and window. (A subject
    // may also carry non-TTE records; those are skipped here.)
    let mut causes: Vec<(usize, HazardFamily, Vec<f64>, f64, f64)> = Vec::new();
    for record in &subject.obs_records {
        let ObsRecord::Event {
            cmt,
            entry_time,
            time,
            event_type,
        } = record;
        let Some(EndpointLikelihood::Tte { hazard }) = model.endpoints.get(cmt) else {
            continue;
        };
        let HazardSpec::Analytic { family, param_fn } = hazard;
        let params = param_fn(theta, eta, &subject.covariates);
        // With an explicit `horizon`, every cause shares it as the administrative
        // window (#522); otherwise only a right-censored record marks a horizon and
        // an event record draws uncensored (+∞) — see `observation_window`.
        let window = horizon.unwrap_or_else(|| observation_window(event_type, *time));
        causes.push((*cmt, *family, params, *entry_time, window));
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
        // Single endpoint: administrative censoring at its own window. (Draws one
        // uniform, preserving the RNG sequence of the pre-competing-risks path.)
        [(cmt, family, params, entry_time, window)] => {
            let (t, observed) = draw_tte_outcome(*family, params, *entry_time, *window, rng);
            push(*cmt, t, observed, results);
        }
        // Competing risks: earliest latent event wins; the rest censor at that
        // time. Draw in record order so the RNG sequence stays deterministic.
        _ => {
            let latents: Vec<f64> = causes
                .iter()
                .map(|(_, family, params, entry_time, _)| {
                    draw_tte_latent(*family, params, *entry_time, rng)
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
                .map(|(_, _, _, _, w)| *w)
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
        let nll = tte_data_term(&records, &hazard, theta, eta, &cov);
        // -log L = H(T) = 0.1 * 10 = 1.0
        assert_abs_diff_eq!(nll, 1.0, epsilon = 1e-12);
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
        let nll = tte_data_term(&records, &hazard, theta, eta, &cov);
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
        let nll = tte_data_term(&records, &hazard, &[0.1], &[0.0], &HashMap::new());
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
        let nll = tte_data_term(&records, &hazard, &[0.2], &[0.0], &HashMap::new());
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
        let nll = tte_data_term(&records, &hazard, &[0.1], &[0.0], &HashMap::new());
        // NLL must be finite and positive; exact value verified via log-domain formula.
        assert!(nll.is_finite() && nll > 0.0, "nll = {nll}");
        let delta = 0.1_f64 * 0.0001; // H(right) - H(left)
        let a = 0.1_f64 * 10.0; // H(left) - H(entry)
        let expected = a - ((-delta).exp_m1().abs().ln());
        assert_abs_diff_eq!(nll, expected, epsilon = 1e-8);
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
