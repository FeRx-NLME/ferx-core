//! IMPMAP — Importance Sampling assisted by Mode A Posteriori (NONMEM
//! `METHOD=IMPMAP`).
//!
//! A Monte-Carlo EM (MCEM) **estimator**. Each iteration:
//!
//! 1. **E-step part A (MAP):** re-evaluate every subject's conditional mode η̂ᵢ
//!    and first-order-variance Hessian `Hᵢ = Jᵢᵀ Rᵢ⁻¹ Jᵢ + Ω⁻¹` at the current
//!    parameters (the FOCE/ITS inner loop). This re-centering each iteration —
//!    rather than only on the first, as plain `IMP` does — is what makes IMPMAP
//!    robust on high-dimensional, rich-data problems where IMP can stall.
//! 2. **E-step part B (IS):** draw `K` importance samples ηᵢₖ from a proposal
//!    centred at η̂ᵢ with scale `Σᵢ = Hᵢ⁻¹` (multivariate normal by default;
//!    Student-t with `impmap_proposal_df`), with self-normalized weights w̃ᵢₖ.
//! 3. **M-step:** update parameters from the importance-weighted complete-data
//!    expectation:
//!    - **Ω** closed form: `Ω = (1/N) Σᵢ Σₖ w̃ᵢₖ ηᵢₖ ηᵢₖᵀ`.
//!    - **θ, σ** by maximizing the weighted observation likelihood
//!      `Σᵢ Σₖ w̃ᵢₖ log p(yᵢ | ηᵢₖ, θ, σ)` (derivative-free NLopt BOBYQA in
//!      packed log-space, warm-started from the previous iteration).
//!
//! The reported estimate is the running mean of the parameter vector over the
//! final `impmap_averaging` iterations (Monte-Carlo variance reduction). The
//! returned [`OuterResult`] carries the final EBEs / Jacobians and a FOCE
//! Laplace `ofv` for AIC/BIC comparability, identical in shape to SAEM's, so the
//! covariance step and chained-stage handoff in `api.rs` need no special casing.
//!
//! ## Inter-occasion variability (#2301)
//!
//! An IOV subject samples its stacked `[η, κ₁..κ_K]` from one joint proposal
//! against the prior `Ω ⊕ I_K ⊗ Ω_iov`, and Ω_iov takes the closed-form update
//! `Σᵢ Σ_g E_w[κᵢg κᵢgᵀ] / Σᵢ Kᵢ` (`omega_iov_mstep`). Nested variability
//! levels and IOV under a `[mixture]` are still refused.
//!
//! ## Scope
//!
//! SDE / `[diffusion]` models are refused for the same reason `IMP` refuses
//! them. Use FOCE / FOCEI for those.

use crate::estimation::covariate_mu_ref::{resolve_covariate_mu_groups, GroupStepInput};
use crate::estimation::fixed_eta_gradient::obs_nll_subject_into_iov_with_schedule;
use crate::estimation::importance_sampling::{
    build_joint_omega_inv, compute_joint_posterior_hessian, compute_posterior_hessian,
    find_optimal_iscale, find_optimal_iscale_with, subject_is_draws, subject_is_draws_with,
    SubjectDraws, MIX_ESS_PMIX_FLOOR,
};
use crate::estimation::inner_optimizer::{
    build_schedule_cache, cacheable_schedule, cacheable_schedule_iov, find_ebe, EbeResult,
    InnerLoopStats,
};
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, pack_params, theta_packs_log};
use crate::estimation::saem::{
    classify_mixture_mu_ref_pairs, classify_mu_ref_pairs, floor_omega_diagonal,
    get_mixture_mu_ref_pairs, mixture_mu_ref_means, mstep_initial_step,
    mu_ref_pairs_for_cov_groups, MixtureMuRefPairs, MuRefPairs, MSTEP_FTOL_REL,
};
use crate::pk::event_driven::EventSchedule;
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into_with_schedule;
use crate::stats::stacked_layout::{kappa_range, stacked_width};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Positive-definite floor for free Ω diagonals (matches the SAEM constant).
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Adaptive-sampling (`auto`) target for the **per-subject** objective Monte-Carlo
/// standard error, in −2 log L units: `total_MC_SE / √N`. NONMEM's `AUTO` `STDOBJ`
/// targets the *total* objective SE, which grows as √N, so a large but perfectly
/// well-sampled dataset would ramp purely because it has more subjects. The
/// per-subject normalization makes the criterion N-independent (it tracks the
/// average per-subject sampling adequacy, which is what biases the M-step). The
/// value `0.05` reproduces NONMEM's effective stopping point on the FREM workshop
/// model (its `STDOBJ = 1.0` total target at N≈475 ⇒ 1.0/√475 ≈ 0.046): the count
/// still ramps 300→~10000 there, while a rich low-dim fit (e.g. warfarin, total
/// SE ≈ 0.07) never trips it.
const AUTO_STDOBJ_TARGET: f64 = 0.05;

/// Hard cap on the adaptive per-subject sample count (NONMEM `ISAMPEND` default).
const AUTO_SAMPLES_MAX: usize = 10_000;

/// Iterations in each of the two windows [`objective_settled`] compares. Five
/// puts the first possible raise at iteration 10, and a window mean carries
/// `√(1/5)` of one iteration's noise.
const AUTO_SETTLE_WINDOW: usize = 5;

/// Headroom on the sample count a raise aims for, over the count the current
/// per-subject SE says would just meet the target. The SE of one iteration is
/// itself noisy, so an exact aim would land on the target and raise again.
const AUTO_RAISE_HEADROOM: f64 = 1.1;

/// How far `SE²·K` may grow past its value at the first raise before [`raises_pay`]
/// refuses further raises.
const AUTO_SCALE_GROWTH_LIMIT: f64 = 2.0;

/// Whether the MCEM objective has stopped moving, as far as Monte-Carlo noise
/// can tell: the means of the last two [`AUTO_SETTLE_WINDOW`]-iteration windows
/// differ by no more than two standard errors of that difference, taken from the
/// current total objective SE (#2687).
///
/// This is the ascent-based MCEM idea (Caffo, Jank & Jones 2005): more draws pay
/// once the change in the objective is lost in the noise, not while a cold start
/// is descending by tens of units an iteration. The test uses the *estimated* SE
/// on purpose. A fit whose objective wanders far beyond that SE, as cefepime does
/// at ±20 against a total SE near 3, is not held to be settled, and gets its
/// samples when [`auto_hold_ends`] says so.
fn objective_settled(objectives: &[f64], total_mc_se: f64) -> bool {
    let w = AUTO_SETTLE_WINDOW;
    if objectives.len() < 2 * w {
        return false;
    }
    let (older, recent) = objectives[objectives.len() - 2 * w..].split_at(w);
    let mean = |x: &[f64]| x.iter().sum::<f64>() / w as f64;
    // Written so a `NaN` on either side reads as "not settled".
    (mean(recent) - mean(older)).abs() <= 2.0 * total_mc_se * (2.0 / w as f64).sqrt()
}

/// The iteration after which `auto` may raise K whether or not the objective has
/// settled: the one before the averaging window, whose iterations make up the
/// reported estimate, so a fit that never settles still has them sampled to the
/// target. The hold covers at most the first three quarters of the run, so
/// `averaging = 1` does not leave the raise one iteration to work with, and at
/// least the first half, so an averaging window as long as a short run does not
/// switch the hold off (#2687).
fn auto_hold_ends(n_iter: usize, n_avg: usize) -> usize {
    n_iter - n_avg.clamp(n_iter / 4, n_iter / 2)
}

/// The per-subject SE `auto` acts on at a sample count of `k`: the median of
/// `SE²·K` over the last [`AUTO_SETTLE_WINDOW`] iterations, which `SE² ∝ 1/K` makes
/// comparable across a change of K, rescaled to `k`.
///
/// One iteration's SE is not enough to act on. It is dominated by the few subjects
/// with the worst ESS, and on cefepime it swung between 0.018 and 0.094 from one
/// iteration to the next, which halved K on one iteration and doubled it on the
/// following one (#2687).
fn windowed_se(se_times_k: &[(f64, usize)], k: usize) -> f64 {
    let tail = &se_times_k[se_times_k.len().saturating_sub(AUTO_SETTLE_WINDOW)..];
    let mut scaled: Vec<f64> = tail.iter().map(|&(se, n)| se * se * n as f64).collect();
    scaled.sort_by(f64::total_cmp);
    match scaled.get(scaled.len() / 2) {
        Some(&median) => (median / k as f64).sqrt(),
        None => f64::NAN,
    }
}

/// Whether raising K is still doing what a raise assumes.
///
/// `SE² ∝ 1/K` makes `SE²·K` constant, so a raise sized from it lands on the
/// target. Once that product has grown by [`AUTO_SCALE_GROWTH_LIMIT`] since the
/// first raise the assumption has failed, and following the SE only walks K to the
/// cap: on cefepime the SE was 0.071 at K = 1000 and 0.083 at K = 6912, and K = 10000
/// cost ten times as much per iteration for no lower SE (#2687). The limit leaves
/// a run whose SE does scale untouched, since its product stays where it was and its
/// target K sits below the largest K this allows, about `2.2 ×` the one the first
/// raise asked for.
///
/// The baseline is the window before K first left `k_start`. It holds across a later
/// lowering, and a product that falls back under the limit allows raises again.
fn raises_pay(se_history: &[(f64, usize)], k_start: usize) -> bool {
    let Some(first_raise) = se_history.iter().position(|&(_, k)| k > k_start) else {
        return true;
    };
    let scale_at = |history: &[(f64, usize)]| windowed_se(history, 1).powi(2);
    scale_at(se_history) < AUTO_SCALE_GROWTH_LIMIT * scale_at(&se_history[..first_raise])
}

/// The sample count for the next iteration under `auto` (#2687).
///
/// Above [`AUTO_STDOBJ_TARGET`], and only when `may_raise`, K rises to what the
/// SE asks for (`SE² ∝ 1/K`) with [`AUTO_RAISE_HEADROOM`], at most doubling and at
/// most [`AUTO_SAMPLES_MAX`]; sizing the step keeps an SE that only just misses
/// the target from doubling the cost of every later iteration. Below half the
/// target K halves, but never below the configured `k_start`. Halving a count
/// that sits under half the target predicts an SE of `0.71 ×` target, so on a
/// stable [`windowed_se`] a lowering does not call for a raise.
fn next_sample_count(k: usize, k_start: usize, se_per_subject: f64, may_raise: bool) -> usize {
    if se_per_subject > AUTO_STDOBJ_TARGET {
        if !may_raise {
            return k;
        }
        let ratio = se_per_subject / AUTO_STDOBJ_TARGET;
        let wanted = (k as f64 * ratio * ratio * AUTO_RAISE_HEADROOM).ceil() as usize;
        wanted.min(k.saturating_mul(2)).min(AUTO_SAMPLES_MAX).max(k)
    } else if se_per_subject < 0.5 * AUTO_STDOBJ_TARGET {
        (k / 2).max(k_start).min(k)
    } else {
        k
    }
}

/// The closing advice of the under-sampling warning. Under `auto`, a final count
/// below the cap means either [`raises_pay`] stopped the ramp or the run ran out of iterations rather than out of samples, which the hold on
/// raising makes a real case (#2687).
fn under_sampling_advice(
    auto: bool,
    final_k: usize,
    se_history: &[(f64, usize)],
    k_start: usize,
    sample_opt: &str,
    auto_opt: &str,
    only_left_out_noisy: bool,
) -> String {
    if auto && only_left_out_noisy {
        format!(
            "`{auto_opt}` met the MC-SE target on the subjects it samples for; the remaining \
             noise comes from the subject(s) it left out, named below, so a larger \
             `{sample_opt}` is unlikely to help."
        )
    } else if auto && final_k >= AUTO_SAMPLES_MAX {
        format!(
            "`{auto_opt}` is enabled but the sample count {final_k} (cap {AUTO_SAMPLES_MAX}) \
             was not enough to reach the MC-SE target."
        )
    } else if auto && !raises_pay(se_history, k_start) {
        format!(
            "`{auto_opt}` stopped raising the sample count at {final_k}, below the cap of \
             {AUTO_SAMPLES_MAX}, because extra samples were not lowering the MC SE as they \
             should. A larger `{sample_opt}` is unlikely to help."
        )
    } else if auto {
        format!(
            "`{auto_opt}` is enabled but the run ended at {final_k} samples, below the cap of \
             {AUTO_SAMPLES_MAX}, before the count could reach the MC-SE target. Raise \
             `{sample_opt}` or give the fit more iterations."
        )
    } else {
        format!(
            "Raise `{sample_opt}` (high-dimensional / FREM models typically need several \
             thousand) or set `{auto_opt} = true` to ramp it automatically."
        )
    }
}

/// One `auto` decision after an iteration: the next K, and the [`windowed_se`] it
/// was made on. `se_history` ends with the iteration just run, whose K is current;
/// `hold_over` is whether [`auto_hold_ends`] has passed.
fn auto_step(
    objectives: &[f64],
    se_history: &[(f64, usize)],
    k_start: usize,
    n_subjects: usize,
    hold_over: bool,
) -> (usize, f64) {
    let k = se_history.last().map_or(k_start, |&(_, k)| k);
    let se = windowed_se(se_history, k);
    let total_se = se * (n_subjects.max(1) as f64).sqrt();
    let may_raise =
        (hold_over || objective_settled(objectives, total_se)) && raises_pay(se_history, k_start);
    (next_sample_count(k, k_start, se, may_raise), se)
}

/// Effective sample size below which a subject's importance weights are
/// degenerate: one draw carries the weight, or the sampler failed outright.
const DEGENERATE_ESS: f64 = 2.0;

/// Whether a subject with ESS fraction `f` at `k` draws is degenerate
/// (ESS < [`DEGENERATE_ESS`]). Written so a `NaN` fraction reads as degenerate.
fn is_degenerate(f: f64, k: usize) -> bool {
    !(f * k as f64 >= DEGENERATE_ESS)
}

/// Subjects whose degenerate weights more draws have already failed to fix, so
/// `auto` leaves them out of the sample-count decision (#2690).
///
/// A degenerate subject's variance term is about 1 at any K, so one of them keeps
/// the per-subject SE at `2/√N` or more, above the target for any N below 1600, and
/// would ramp every subject to the cap. Not every degenerate subject is hopeless: a
/// proposal that only undersamples the posterior recovers as K grows. So a subject
/// stays in the decision until, within one unbroken run of degenerate iterations, it
/// has been degenerate at a K above the lowest K of that run: a raise was tried on it
/// and did not help. A lowering is no such test, so a subject degenerate first at a
/// high K and then at a lower one stays in. A failed subject (`f = 0`, which is not a
/// sampling problem) is left out at once.
#[derive(Debug, Default)]
struct StuckSubjects {
    /// Per subject, its current degenerate run: the lowest K of the run, and
    /// whether a raise above that K has already failed to help it.
    runs: Vec<Option<(usize, bool)>>,
}

impl StuckSubjects {
    /// Record one iteration's ESS fractions at `k` draws and return, per subject,
    /// whether it is now left out of the ramp.
    fn update(&mut self, ess_fracs: &[f64], k: usize) -> Vec<bool> {
        self.runs.resize(ess_fracs.len(), None);
        ess_fracs
            .iter()
            .zip(self.runs.iter_mut())
            .map(|(&f, run)| {
                if !is_degenerate(f, k) {
                    *run = None;
                    return false;
                }
                let (lo, raised) =
                    run.map_or((k, false), |(lo, raised)| (lo.min(k), raised || k > lo));
                *run = Some((lo, raised));
                !(f > 0.0) || raised
            })
            .collect()
    }
}

/// The per-subject SE history `auto` acts on, rebuilt from each iteration's ESS
/// fractions over the subjects not flagged in `stuck` now (#2690).
///
/// [`windowed_se`] takes a median over several iterations, so excluding a subject
/// only from the newest entry would leave it driving raises from the older ones
/// until the window turned over, which on a one-draw subject among 100 is three
/// doublings toward the cap. [`raises_pay`] compares against its own baseline the
/// same way, so the whole history is rebuilt on the current set.
fn ramp_se_history(ess_history: &[(Vec<f64>, usize)], stuck: &[bool]) -> Vec<(f64, usize)> {
    ess_history
        .iter()
        .map(|(fracs, k)| (ramp_se_per_subject(fracs, *k, stuck).0, *k))
        .collect()
}

/// The per-subject objective MC SE over the subjects `auto` acts on, and how many
/// of them there are: those not flagged in `stuck` (#2690).
fn ramp_se_per_subject(ess_fracs: &[f64], k: usize, stuck: &[bool]) -> (f64, usize) {
    let (var, n) = ess_fracs
        .iter()
        .zip(stuck)
        .filter(|(_, &s)| !s)
        .fold((0.0, 0usize), |(v, n), (&f, _)| {
            (v + log_marginal_mc_variance(f, k), n + 1)
        });
    (2.0 * var.sqrt() / (n.max(1) as f64).sqrt(), n)
}

/// The lowest-ESS subjects, as `ID (ESS e)`, for the verbose log of a change of K:
/// it shows which subjects drive a ramp (#2690).
fn lowest_ess_summary(ids: &[&str], ess_fracs: &[f64], k: usize, n: usize) -> String {
    let mut order: Vec<usize> = (0..ess_fracs.len()).collect();
    order.sort_by(|&a, &b| ess_fracs[a].total_cmp(&ess_fracs[b]));
    order
        .iter()
        .take(n)
        .map(|&i| format!("{} (ESS {:.1})", ids[i], ess_fracs[i] * k as f64))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A list of subject IDs for a warning, truncated after `max` with a count of the
/// rest.
fn id_list(ids: &[&str], max: usize) -> String {
    let shown = ids[..ids.len().min(max)].join(", ");
    match ids.len().saturating_sub(max) {
        0 => shown,
        rest => format!("{shown} and {rest} more"),
    }
}

/// The under-sampling warning's sentence on degenerate subjects that are not named as
/// left out of the ramp. More draws may still recover one whose proposal only
/// undersamples the posterior, so it makes no claim either way (#2690 review).
fn collapse_sentence(n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    format!(
        " {n} subject(s) had a degenerate proposal (ESS < {DEGENERATE_ESS}: one draw carries \
         the weight, or the sampler failed), whose moments are unreliable at this sample \
         count — check their EBE/Hessian quality."
    )
}

/// The under-sampling warning's sentence naming the subjects `auto` left out of the
/// sample-count decision; empty when there are none or `auto` is off (#2690).
fn stuck_subjects_sentence(auto: bool, auto_opt: &str, ids: &[&str]) -> String {
    if !auto || ids.is_empty() {
        return String::new();
    }
    format!(
        " `{auto_opt}` left {} subject(s) out of the sample-count decision because more \
         draws did not lift their ESS above {DEGENERATE_ESS}: {}. Their fix is a better \
         proposal, not more samples.",
        ids.len(),
        id_list(ids, 10)
    )
}

/// Refuse an unusable wall-clock limit rather than let a `NaN` never trip it.
fn check_max_time(label: &str, key: &str, limit: Option<f64>) -> Result<(), String> {
    match limit {
        Some(s) if !(s.is_finite() && s > 0.0) => Err(format!(
            "{label}: `{key}` must be a positive number of seconds, got {s}"
        )),
        _ => Ok(()),
    }
}

/// Seconds elapsed once the wall-clock limit has passed; `None` while there is
/// time left, or when no limit is set.
fn time_limit_reached(started: std::time::Instant, limit: Option<f64>) -> Option<f64> {
    let elapsed = started.elapsed().as_secs_f64();
    limit.filter(|&s| elapsed >= s).map(|_| elapsed)
}

/// The warning that explains a fit stopped by `imp_max_time` / `impmap_max_time`.
/// `n_acc` is how many iterations of the averaging window were reached, which
/// decides what the reported estimate is.
fn time_limit_warning(
    label: &str,
    key: &str,
    elapsed: f64,
    iterations_done: usize,
    n_iter: usize,
    n_acc: usize,
) -> String {
    let estimate = if n_acc > 0 {
        format!("the mean of the {n_acc} averaging iteration(s) reached")
    } else {
        "the last iteration's parameters, unaveraged".to_string()
    };
    format!(
        "{label}: stopped by `{key}` after {elapsed:.1} s, at iteration {iterations_done} of \
         {n_iter}. The reported estimate is {estimate}. The fit did not finish its schedule, \
         so it is reported as not converged."
    )
}

/// Absolute lower bound on the IMP proposal-covariance diagonal — a numerical
/// guard against a literally-zero (degenerate-ESS) variance, NOT a statistical
/// floor. It must stay well below any real conditional variance: with rich data
/// the conditional posterior of η is legitimately tiny (orders of magnitude below
/// the prior Ω), and flooring it against Ω would make the proposal far too broad
/// and collapse the ESS (the very rich-data failure mode that motivates IMPMAP).
const IMP_PROPOSAL_COV_FLOOR: f64 = 1e-10;

/// Objective ceiling beyond which an estimation run is declared diverged rather
/// than converged (issue #528). A collapsed-weight runaway pins θ to the
/// parameter bounds and the final FOCE-Laplace OFV blows up to ~1e35 — a finite
/// value that `is_finite()` would otherwise accept. Real −2logL objectives scale
/// with the number of observations and reach at most ~1e6–1e7 even for very
/// large cohorts, so 1e15 leaves ~8 orders of head-room above any legitimate fit
/// while closing the gap that let a 1e12–1e29 runaway pass as converged.
pub(crate) const OFV_DIVERGENCE_CAP: f64 = 1e15;

/// Whether an estimation run's final objective signals convergence rather than a
/// runaway. A non-finite OFV is an outright divergence; a finite-but-enormous
/// one (≥ [`OFV_DIVERGENCE_CAP`]) is the bounded blowup of a collapsed-weight
/// run, which `is_finite()` alone would wave through (issue #528). Shared by the
/// IMP/IMPMAP MCEM core and SAEM (both can run away to a finite blowup).
pub(crate) fn objective_converged(ofv: f64) -> bool {
    ofv.is_finite() && ofv.abs() < OFV_DIVERGENCE_CAP
}

/// The MCEM family's convergence gate: #1303's shared
/// [`gate_converged_on_objective`](crate::estimation::outer_optimizer::gate_converged_on_objective)
/// composed with [`objective_converged`]'s #528 runaway cap, demoting `converged`
/// and returning the reason.
///
/// **This is not a redundant second gate.** Neither predicate subsumes the
/// other, and the two witnesses are disjoint — both measured, and both pinned by
/// `the_two_mcem_objective_rules_reject_different_runs`:
///
/// | `ofv` | `ofv_is_valid` (#1303) | `objective_converged` (#528) |
/// |---|---|---|
/// | `5e14` | **false** (≥ `DIVERGENCE_OFV` = 1e14) | true (`abs` < 1e15) |
/// | `-1e16` | true (the cutoff is deliberately **one-sided**: a large negative −2 log L is legitimate) | **false** |
///
/// So deleting either leaves a class of diverged MCEM run reported as converged.
/// Before #1303 the three MCEM exits applied the second rule alone and demoted
/// **silently** — a caller got `converged: false` with nothing in `warnings`
/// saying which of the many reasons a fit can fail it was.
pub(crate) fn gate_converged_on_mcem_objective(converged: &mut bool, ofv: f64) -> Option<String> {
    use crate::estimation::outer_optimizer::{
        gate_converged_on_objective, NONFINITE_OBJECTIVE_TOKEN,
    };
    // The shared rule first, so a `NaN` / sentinel run gets the message every
    // other estimator gives it rather than a Monte-Carlo-specific one.
    if let Some(w) = gate_converged_on_objective(converged, ofv) {
        return Some(w);
    }
    // Same shape as the shared gate: a pure predicate on the objective, not on
    // the verdict coming in, so a run already failed for another reason still
    // learns its objective is unusable.
    if objective_converged(ofv) {
        return None;
    }
    *converged = false;
    Some(format!(
        "{NONFINITE_OBJECTIVE_TOKEN}: the Monte-Carlo EM objective at the final estimates \
         ({ofv:?}) is beyond any physically meaningful −2 log L, so the run did not converge \
         and is reported converged: false. This is the signature of an importance-weight \
         collapse: the weighted M-step walks THETA out to the parameter bounds and the final \
         Laplace objective blows up to a large but finite value, which is why `is_finite()` \
         does not catch it. Raise the sample count (`imp_isample`), start from better initial \
         estimates, or chain a FOCE/FOCEI stage first."
    ))
}

/// How far above the lowest iteration objective, as a fraction of the objective's
/// scale, a final marginal may sit before [`mcem_objective_rise`] calls the run
/// collapsed (#2652).
///
/// An EM iteration should not raise the objective, so the only legitimate gap is
/// Monte-Carlo noise. Measured on Linux aarch64 (glibc 2.41, release without LTO) over
/// the seven models of the 2026-11 IMP-vs-NONMEM evaluation, K = 1000, `[focei, imp]`
/// and cold `imp`, reading every iteration's objective. With #2650, #2651 and #2653 in,
/// all eleven runs are healthy and the widest gap is 0.13 % of the scale (thioguanine:
/// lowest -250.89, marginal -250.57). Collapses need the tree before #2650: there the
/// narrowest is 50.7 % (cold cefepime: lowest 4300.70, marginal 6480.15). 10 % sits 77×
/// above the first and 5× below the second. macOS arm64, measured on its own before
/// #2650, gave a narrowest collapse of 28 %, also above the bound.
pub(crate) const MCEM_RISE_TOLERANCE: f64 = 0.10;

/// The rise of the final marginal `marginal` over `lowest`, the lowest objective the
/// MCEM iterations reached, when it is too large to be Monte-Carlo noise (#2652);
/// `None` otherwise.
///
/// A collapse on the first recentre can sit below the 1e14 / 1e15 cutoffs of
/// [`gate_converged_on_mcem_objective`] for good: before #2650, on Linux, cefepime read
/// 4313.3 at iteration 1, 4.4e7 at iteration 2, and ended at a marginal of 13,354.7
/// against NONMEM's 4282.9.
///
/// * The baseline is the **lowest** iteration, not the first, because a cold start
///   makes the first a poor reference: infliximab read 3087.4, reached 703.1, blew up
///   to 5e5 and ended at 1545.6, below where it began.
/// * The scale is `max(|lowest|, n_obs)`, because −2 log L has no natural zero: an
///   objective near 0 would make a purely relative test fire on noise.
pub(crate) fn mcem_objective_rise(marginal: f64, lowest: f64, n_obs: usize) -> Option<f64> {
    let rise = marginal - lowest;
    (rise > MCEM_RISE_TOLERANCE * lowest.abs().max(n_obs as f64)).then_some(rise)
}

/// Fold one iteration's objective into the lowest seen so far. A non-finite value is
/// skipped explicitly: `f64::min` would do it silently for `NaN`, and for `-inf` it
/// would keep a baseline nothing can rise above.
fn lowest_objective(lowest: Option<f64>, objective: f64) -> Option<f64> {
    if !objective.is_finite() {
        return lowest;
    }
    Some(lowest.map_or(objective, |l| l.min(objective)))
}

/// Iterations in each of the two trailing windows [`McemObjectiveTrail::settling`]
/// compares (#2686). A run shorter than three windows uses a third of its length, so
/// the first third, where a cold start descends, is never read.
pub(crate) const MCEM_SETTLE_WINDOW: usize = 50;

/// Fewest iterations per window [`McemObjectiveTrail::settling`] will judge, so a run
/// needs three times this many.
pub(crate) const MCEM_SETTLE_MIN_WINDOW: usize = 10;

/// The fraction of a window below its floor (#2686). See
/// [`McemObjectiveTrail::settling`] for why a floor, and not a spread.
const MCEM_FLOOR_QUANTILE: f64 = 0.10;

/// Smallest gap, in −2 log L units, that [`McemObjectiveTrail::settling`] treats as
/// material: 3.84, the 5% point of χ²₁. A gap under one likelihood-ratio step cannot
/// change a model comparison. It also spares a slow finish: cold infliximab's floor was
/// still falling by 1.78, 23 SEs, and its marginal ended within 1 of the best.
pub(crate) const MCEM_SETTLE_MIN_GAP: f64 = 3.84;

/// How many Monte-Carlo SEs a gap must also exceed in
/// [`McemObjectiveTrail::settling`], so a noisy objective has to show more.
pub(crate) const MCEM_SETTLE_NOISE_RATIO: f64 = 4.0;

/// The per-iteration record of an MCEM run, published so `fit()` can test whether the
/// objective settled once the final marginal is known (#2686).
#[derive(Debug, Clone)]
pub(crate) struct McemObjectiveTrail {
    /// −2 log L of each iteration.
    pub(crate) objectives: Vec<f64>,
    /// Monte-Carlo SE of each entry of `objectives`.
    pub(crate) mc_se: Vec<f64>,
    /// `"IMP"` or `"IMPMAP"`, as the run's other warnings spell it.
    pub(crate) label: String,
    pub(crate) iterations_option: &'static str,
    pub(crate) samples_option: &'static str,
    pub(crate) proposal_option: &'static str,
}

/// The final marginal [`McemObjectiveTrail::settling`] compares the floor with. Its SE
/// is its own: the final pass draws the configured sample count, which `imp_auto` can
/// leave far below what the iterations ramped to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FinalMarginal {
    pub(crate) value: f64,
    pub(crate) mc_se: f64,
}

/// What [`McemObjectiveTrail::settling`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum McemSettling {
    /// Too few iterations to judge. The verdict stays as it was.
    TooShort,
    /// Enough iterations, but too few of them finite to read a window. `usable` is the
    /// count in the worse of the two windows, each `window` iterations long.
    Unusable {
        window: usize,
        usable: usize,
    },
    Settled,
    /// The objective's floor was still falling at the iteration limit.
    Falling {
        window: usize,
        previous_floor: f64,
        floor: f64,
        previous_mc_se: f64,
        mc_se: f64,
    },
    /// The iterations stayed well above the marginal of their own final estimates.
    AboveMarginal {
        window: usize,
        floor: f64,
        marginal: FinalMarginal,
        mc_se: f64,
    },
}

/// The floor of a window of iteration objectives: its lower 10% point. Takes the
/// window's finite values, so a non-finite iteration is dropped, as [`lowest_objective`]
/// drops it.
fn objective_floor(mut finite: Vec<f64>) -> f64 {
    finite.sort_by(f64::total_cmp);
    finite[(MCEM_FLOOR_QUANTILE * finite.len() as f64) as usize]
}

/// The finite entries of `window`.
fn finite_values(window: &[f64]) -> Vec<f64> {
    window.iter().copied().filter(|v| v.is_finite()).collect()
}

/// How far the objective of a window with MC SE `previous_mc_se` sits above that of one
/// with `mc_se`, from sampling alone. `log p̂` runs low by half its variance, so −2 log L
/// runs high by `Σᵢ Var(log p̂ᵢ) = se²/4`. It counts only against the earlier window:
/// that is the direction `imp_auto` ramps, and the direction that fakes a fall.
fn sampling_bias_excess(previous_mc_se: f64, mc_se: f64) -> f64 {
    ((previous_mc_se * previous_mc_se - mc_se * mc_se) / 4.0).max(0.0)
}

impl McemObjectiveTrail {
    /// Whether the objective had settled by the end of the run (#2686). IMP and IMPMAP
    /// run a fixed schedule, so every run ends at the iteration limit.
    ///
    /// Reads the floor of the last window, and asks two things of it. A gap counts when
    /// it exceeds both [`MCEM_SETTLE_MIN_GAP`] and [`MCEM_SETTLE_NOISE_RATIO`] MC SEs of
    /// what it compares. Each side brings its own SE, and they combine as independent
    /// estimates: the two windows' for two floors, the last window's and the marginal's
    /// for a floor and a marginal.
    ///
    /// * **Is it still falling?** Against the floor of the window before. This is the
    ///   run that needs more iterations. `imp_auto` can leave that window on fewer
    ///   samples, which raises its objective with no change in the estimates, so
    ///   [`sampling_bias_excess`] comes off the fall first.
    /// * **Is it above the marginal?** `marginal` is scored at the final estimates with
    ///   a fresh proposal. Iterations that never come near it were not sampling the
    ///   posterior, so nothing in the trace shows the run reached an optimum. A marginal
    ///   without a finite value and SE cannot convict a run of this.
    ///
    /// **The floor, not the spread.** A failed proposal misses posterior mass, so it
    /// only ever raises an iteration's objective. A normal proposal makes these
    /// excursions on a run whose estimates are fine: thioguanine at seed 777 climbed to
    /// −132.8 inside the window, and still ended with a marginal of −249.82 against a
    /// best of −250.58. Its robust spread (13.4) is twice that of vancomycin at 1004.72,
    /// which is 8 short of the optimum (6.0), so no bound on spread separates them.
    ///
    /// Measured on Linux aarch64 (glibc 2.41, release without LTO), K = 1000 and 200
    /// iterations, on the six models of the 2026-11 IMP-vs-NONMEM evaluation: `[focei,
    /// imp]`, cold `imp`, and cold `imp` with `imp_proposal_df = normal` and
    /// `imp_averaging = 1`, the last at three seeds. Also cold `impmap` on warfarin,
    /// thioguanine and propofol, and both drivers on a two-class mixture.
    ///
    /// | Gap | Widest on a healthy run | On a run short of the optimum |
    /// |---|---|---|
    /// | floor fell | 1.78 (infliximab) | 30.5 (vancomycin at 1004.72, best 996.75) |
    /// | floor above marginal | 3.86 at 1.5 SEs (chained cefepime) | 42.4 and 69.2 (cefepime at 4288.06 and 4298.55, best 4277) |
    ///
    /// The noise term decides that 3.86, which the minimum alone would convict. It sits
    /// on a good estimate: cefepime's low-ESS subjects keep its objective noisy.
    ///
    /// Those runs fix K. `imp_auto` holds its raises until the objective settles or
    /// [`auto_hold_ends`], which at 200 iterations and an averaging window of 50 is
    /// iteration 150, the line between the two windows. Cold on the same six models
    /// with `imp_auto` on, under both proposals, eleven runs finished. K moved inside
    /// the windows on three of them (vancomycin, down at iteration 108; propofol, 152 to
    /// 158; thioguanine, up to 192). All eleven kept `converged: true` with no settling warning, the
    /// widest fall being 2.16 (vancomycin, marginal 996.67). None had an SE near the 16
    /// where [`sampling_bias_excess`] outruns the noise bound, so that term rests on its
    /// derivation.
    pub(crate) fn settling(&self, marginal: Option<FinalMarginal>) -> McemSettling {
        use crate::estimation::vi::run::median_and_mad;
        let n = self.objectives.len();
        let window = MCEM_SETTLE_WINDOW.min(n / 3);
        if window < MCEM_SETTLE_MIN_WINDOW {
            return McemSettling::TooShort;
        }
        let start = n - window;
        let previous = start - window..start;
        let last = start..n;
        let columns = [
            finite_values(&self.objectives[previous.clone()]),
            finite_values(&self.objectives[last.clone()]),
            finite_values(&self.mc_se[previous]),
            finite_values(&self.mc_se[last]),
        ];
        let usable = columns.iter().map(Vec::len).min().unwrap_or(0);
        if usable < MCEM_SETTLE_MIN_WINDOW {
            return McemSettling::Unusable { window, usable };
        }
        let [previous_objectives, objectives, previous_se, se] = columns;
        let (Some((previous_mc_se, _)), Some((mc_se, _))) =
            (median_and_mad(&previous_se), median_and_mad(&se))
        else {
            return McemSettling::Unusable { window, usable };
        };
        let previous_floor = objective_floor(previous_objectives);
        let floor = objective_floor(objectives);
        let material = |se: f64| MCEM_SETTLE_MIN_GAP.max(MCEM_SETTLE_NOISE_RATIO * se);
        let fall = previous_floor - floor - sampling_bias_excess(previous_mc_se, mc_se);
        if fall > material(mc_se.hypot(previous_mc_se)) {
            return McemSettling::Falling {
                window,
                previous_floor,
                floor,
                previous_mc_se,
                mc_se,
            };
        }
        // Filtered first: `f64::max` would discard a `NaN` SE and leave the bare minimum.
        match marginal.filter(|m| m.value.is_finite() && m.mc_se.is_finite()) {
            Some(marginal) if floor - marginal.value > material(mc_se.hypot(marginal.mc_se)) => {
                McemSettling::AboveMarginal {
                    window,
                    floor,
                    marginal,
                    mc_se,
                }
            }
            _ => McemSettling::Settled,
        }
    }
}

/// `Var(log p̂)` of one subject's importance-sampling marginal (Geweke, from the
/// self-normalized ESS): `(1/ESS_frac − 1)/K`. A failed subject (`ESS_frac = 0`)
/// gets a finite fallback of 1.
///
/// A subject whose single draw dominates has `ESS_frac = 1/K` and a term of
/// `(K − 1)/K ≈ 1` whatever K is, the same as the fallback (#2690).
fn log_marginal_mc_variance(ess_fraction: f64, k_samples: usize) -> f64 {
    if ess_fraction > 1e-6 {
        ((1.0 / ess_fraction) - 1.0).max(0.0) / k_samples as f64
    } else {
        1.0
    }
}

/// [`log_marginal_mc_variance`] of a mixture subject's `log Σ_c π_c p̂_c`. By the delta
/// method class `c` enters with its responsibility squared, so a class the subject does
/// not belong to adds nothing however poor its proposal.
fn mixture_log_marginal_mc_variance(
    responsibilities: &[f64],
    ess_fractions: impl Iterator<Item = f64>,
    k_samples: usize,
) -> f64 {
    responsibilities
        .iter()
        .zip(ess_fractions)
        .map(|(r, ess)| r * r * log_marginal_mc_variance(ess, k_samples))
        .sum()
}

/// How each MCEM iteration positions the per-subject importance-sampling
/// proposal — the one piece that distinguishes IMP from IMPMAP. Everything else
/// (M-step, sufficient statistics, averaging, ESS diagnostics, final objective)
/// is shared by [`run_mcem`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProposalRecenter {
    /// IMPMAP (NONMEM `METHOD=IMPMAP`): re-run the MAP inner loop **every**
    /// iteration; proposal centered at the conditional mode with
    /// first-order-variance scale `(JᵀR⁻¹J + Ω⁻¹)⁻¹`.
    Map,
    /// IMP (NONMEM `METHOD=IMP`): run the MAP inner loop on the **first**
    /// iteration only (to seed the proposal); thereafter center at the previous
    /// iteration's weighted posterior mean with scale = previous weighted
    /// posterior covariance `Ŝ − m̂m̂ᵀ`.
    SampleMoments,
}

impl ProposalRecenter {
    /// Bundle a finished run's per-iteration record with the option names its warning
    /// should cite.
    fn objective_trail(
        self,
        objectives: Vec<f64>,
        mc_se: Vec<f64>,
        label: &str,
    ) -> McemObjectiveTrail {
        let (iterations_option, samples_option, proposal_option) = match self {
            ProposalRecenter::Map => ("impmap_iterations", "impmap_samples", "impmap_proposal_df"),
            ProposalRecenter::SampleMoments => ("imp_iterations", "imp_samples", "imp_proposal_df"),
        };
        McemObjectiveTrail {
            objectives,
            mc_se,
            label: label.to_string(),
            iterations_option,
            samples_option,
            proposal_option,
        }
    }

    /// The estimation method this recentering strategy implements, so the shared
    /// MCEM core can ask `EstimationMethod::requires_random_effects` about the
    /// method actually running rather than restating the membership rule (#1050).
    fn method(self) -> EstimationMethod {
        match self {
            ProposalRecenter::Map => EstimationMethod::Impmap,
            ProposalRecenter::SampleMoments => EstimationMethod::Imp,
        }
    }

    /// The `[fit_options]` key of this method's wall-clock limit.
    fn max_time_key(self) -> &'static str {
        match self {
            ProposalRecenter::Map => "impmap_max_time",
            ProposalRecenter::SampleMoments => "imp_max_time",
        }
    }
}

/// Convert a weighted posterior covariance `Cov` into the proposal precision
/// `Σ⁻¹ = Cov⁻¹` that [`subject_is_draws`]/`build_proposal` expects (it forms the
/// proposal scale as `(Σ⁻¹ + λI)⁻¹`). Used only by the IMP (`SampleMoments`)
/// recenter path.
///
/// The raw weighted sample covariance is unbounded above and makes the adaptive
/// proposal unstable: a heavy-tailed outlier inflates it without limit, and then
/// the prior term `−½ηᵀΩ⁻¹η` of the resulting far samples explodes the −2 log L
/// and the next Ω M-step. We therefore **cap** the proposal-covariance diagonal
/// at the prior `Ωᵢᵢ` — the conditional variance of a well-identified η is
/// bounded above by its prior variance. A capped dimension's covariances are
/// rescaled by the same factor, so the correlations survive: `D C D` stays
/// positive semi-definite, whereas capping the diagonal alone leaves a strongly
/// correlated covariance indefinite (#2650). A non-finite variance carries no
/// usable correlation, so its covariances are dropped. The diagonal is floored
/// only at a tiny absolute value to avoid a singular matrix (NOT at a fraction of
/// Ω — see [`IMP_PROPOSAL_COV_FLOOR`]). If the result is still not
/// Cholesky-invertible the prior precision `Ω⁻¹` is returned — a broad but valid
/// proposal. It has to be returned here: a zero matrix does not reach
/// `build_proposal`'s Ω fallback, because the jitter turns it into a
/// positive-definite `1e−10·I`, a proposal SD of 1e5 (#2650). A zero matrix remains
/// only for an Ω that is itself not positive-definite.
fn covariance_to_proposal_hessian(
    cov: &DMatrix<f64>,
    omega: &DMatrix<f64>,
    floor: f64,
) -> DMatrix<f64> {
    let n = cov.nrows();
    let mut c = cov.clone();
    let mut sd_scale = vec![1.0_f64; n];
    for i in 0..n {
        let hi = omega[(i, i)].max(floor);
        let v = c[(i, i)];
        if !v.is_finite() {
            c[(i, i)] = hi;
            sd_scale[i] = 0.0;
        } else if v > hi {
            c[(i, i)] = hi;
            sd_scale[i] = (hi / v).sqrt();
        } else if v < floor {
            c[(i, i)] = floor;
        }
    }
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Assign rather than multiply on a dropped dimension: `0 · NaN` is NaN.
            let s = sd_scale[i] * sd_scale[j];
            c[(i, j)] = if s == 0.0 { 0.0 } else { c[(i, j)] * s };
        }
    }
    c.cholesky()
        .or_else(|| omega.clone().cholesky())
        .map_or_else(|| DMatrix::zeros(n, n), |ch| ch.inverse())
}

/// Translate one subject's importance draws by `−shift` in η-space: every
/// sampled η, the weighted mean, and the second moment, so the weights and the
/// weighted covariance `S − m mᵀ` are unchanged.
///
/// The closed-form mu-ref step adds `δ_e` to the packed θ (= mu) anchoring η_e,
/// so the individual parameter `φ = mu + η` is unchanged only if the posterior of
/// η_e moves by `−δ_e`. IMP (`SampleMoments`) centres the next iteration's
/// proposal on the stored weighted mean, so leaving it stale puts the proposal
/// `δ_e` off the posterior it is meant to sample — from a cold start several
/// posterior SDs on the first iteration, which is what collapsed every subject's
/// ESS on iteration 2 (#1365). Each #619 group step applies the same
/// translation with its own per-subject shift `Δ_i`.
///
/// The samples move too, not only the moments: later in the same iteration a
/// #619 group scores these draws against a θ that already carries the shift
/// (the samples on the numerical engine, the mean on the exact one), and it must
/// see each sample at its unchanged `φ`.
fn shift_draws(d: &mut SubjectDraws, shift: &[f64]) {
    for eta in d.etas.iter_mut() {
        for (e, &s) in eta.iter_mut().zip(shift.iter()) {
            *e -= s;
        }
    }
    let m_old = DVector::from_column_slice(&d.mean);
    for (m, &s) in d.mean.iter_mut().zip(shift.iter()) {
        *m -= s;
    }
    let m_new = DVector::from_column_slice(&d.mean);
    d.second_moment = &d.second_moment - &m_old * m_old.transpose() + &m_new * m_new.transpose();
}

/// Per-θ mask of "is the anchor of a detected mu-reference", indexed by θ
/// position — the proxy both the #406 IMP/IMPMAP advisory
/// ([`non_fixed_unanchored_thetas`]) and the #1011 SAEM M-step damping gate use
/// for "this θ carries an ETA", read from here so the two cannot drift apart.
///
/// **It is mu-reference *detection*, not a scan of ETA usage.** `true` means the
/// parser paired this θ with an η it can move by a closed-form shift — the
/// `TVX * exp(ETA_X)` and logit forms, plus a class θ whose class-aware shift is
/// active this fit. An η attached in a form the detector does not recognise (an
/// additive `X = TVX + ETA_X`, a covariate model whose typical value is not
/// log-linear in one θ, #619) reads as `false` even though an η exists.
///
/// That is the right side to err on for both consumers, because both are about
/// the *channel that moves the θ*, not about η bookkeeping: an un-anchored θ has
/// no closed-form shift and is moved solely by the numerical / importance-
/// weighted M-step, which is exactly the biased channel #406 and #1011 are
/// about. What the advisories must *not* do is call every un-anchored θ
/// "ETA-less": [`partition_by_eta`] separates the θ that share a value with an
/// η in an unrecognised form (a Box-Cox ETA, its shape θ — where removing the
/// transform is the wrong advice) from the ones with no η at all (#1715).
///
/// A MIXNUM-switched typical value carries an ETA in every class arm, so its
/// class thetas must not be counted as fixed-effect-only (#996) — they are
/// mu-referenced per class, not estimated by the weighted M-step alone.
///
/// The caller passes the thetas whose class-aware shift is *actually active*
/// this fit rather than everything the parser detected: the shift is skipped
/// under `mu_referencing = false`, for identity-packed θ, and for weak-IIV η,
/// and in each of those cases the class θ really is estimated by the weighted
/// M-step alone — exactly the situation these consumers exist for (#996
/// review). Pass `&[]` where no class-aware shift runs.
pub(crate) fn theta_is_mu_ref_anchor_mask(
    model: &CompiledModel,
    cov_group_thetas: &[usize],
    class_mu_ref_thetas: &[usize],
) -> Vec<bool> {
    use std::collections::HashSet;
    let mut with_eta: HashSet<&str> = model
        .mu_refs
        .values()
        .map(|m| m.theta_name.as_str())
        .collect();
    // Every theta of a covariate mu-reference group (#619) is an anchor: the
    // group step moves it through its eta's population of individual values.
    // Like `class_mu_ref_thetas`, these are the thetas of the groups the run
    // *resolved*, not the ones the parser detected —
    // `resolve_covariate_mu_groups` drops a group for weak IIV, a conflicting
    // single-anchor pair or an all-FIXed theta list, and for a dropped group
    // the theta really is estimated by the weighted M-step alone, which is the
    // situation these consumers exist for (#918 review).
    with_eta.extend(
        cov_group_thetas
            .iter()
            .filter_map(|&t| model.theta_names.get(t).map(String::as_str)),
    );
    with_eta.extend(
        class_mu_ref_thetas
            .iter()
            .filter_map(|&t| model.theta_names.get(t).map(String::as_str)),
    );
    model
        .theta_names
        .iter()
        .map(|name| with_eta.contains(name.as_str()))
        .collect()
}

/// Names of non-fixed thetas that do **not** anchor a mu-reference this fit, so
/// no closed-form shift moves them. Under IMP/IMPMAP such parameters are
/// estimated solely through the importance-weighted θ M-step, which carries an
/// IS-weight bias for weakly-identified parameters and can converge to the wrong
/// value (issue #406: the FREM `FRD1` absorption-fraction drove to 0.90 vs a
/// FOCEI/NONMEM value of ~0.4); under SAEM they ride the η-frozen numerical
/// M-step.
///
/// "Not anchored" is not "no ETA" (#1715): a θ can share a value with an η in a
/// form the shift cannot use — a Box-Cox / power transform of the η, a shape
/// θ inside that transform, a covariate coefficient. Split the list with
/// [`partition_by_eta`] before telling the user to attach an ETA.
///
/// See [`theta_is_mu_ref_anchor_mask`] for the exact anchor predicate and for
/// what `cov_group_thetas` and `class_mu_ref_thetas` must contain.
pub(crate) fn non_fixed_unanchored_thetas(
    model: &CompiledModel,
    theta_fixed: &[bool],
    cov_group_thetas: &[usize],
    class_mu_ref_thetas: &[usize],
) -> Vec<String> {
    let anchored = theta_is_mu_ref_anchor_mask(model, cov_group_thetas, class_mu_ref_thetas);
    model
        .theta_names
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            !theta_fixed.get(*i).copied().unwrap_or(false)
                && !anchored.get(*i).copied().unwrap_or(false)
        })
        .map(|(_, name)| name.clone())
        .collect()
}

/// Split θ names into `(with_eta, without_eta)` (#1715): the first enter an
/// individual parameter that carries a random effect, the second share a value
/// with no η at all. Order within each half is preserved.
///
/// "Carries" is [`CompiledModel::theta_eta_linked`], the #1177 Delattre class:
/// transitive through intermediate assignments and `if` conditions, κ
/// included, classified after `[covariate_model]` desugaring. It is empty on a
/// hand-built fixture, which then lands every name in `without_eta` — the
/// pre-#1715 wording.
///
/// Applied to [`non_fixed_unanchored_thetas`] it separates the two advisories
/// the old single "NO associated ETA" list conflated: a θ with no η, where
/// attaching one is the remedy, and a θ whose η is there but not in a
/// mu-referenceable form, where it is not — removing a Box-Cox transform the
/// user chose on purpose would change the model, not fix the fit.
pub(crate) fn partition_by_eta(
    model: &CompiledModel,
    names: Vec<String>,
) -> (Vec<String>, Vec<String>) {
    names.into_iter().partition(|n| {
        model
            .theta_names
            .iter()
            .position(|t| t == n)
            .is_some_and(|t| model.theta_eta_linked.get(t).copied().unwrap_or(false))
    })
}

/// The IMP/IMPMAP advisory for a non-fixed θ the closed-form mu-reference shift
/// cannot move, split by whether it carries an ETA (#1715). Empty when both
/// lists are. `label` prefixes each message (`IMPMAP`, `FREM conversion`, …).
pub(crate) fn imp_unanchored_theta_warnings(
    label: &str,
    with_eta: &[String],
    without_eta: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    if !without_eta.is_empty() {
        out.push(format!(
            "{label}: estimated parameter(s) [{}] have NO associated ETA. NONMEM's \
             IMP/IMPMAP require every estimated parameter to carry a random effect; a \
             fixed-effect-only parameter is estimated solely through the importance-weighted \
             M-step, which is biased for weakly-identified parameters and may converge to the \
             wrong value. STRONGLY add an ETA to each (e.g. `P = TVP * exp(ETA_P)` with a small, \
             optionally FIX, omega — ferx applies mu-referencing automatically), or hold the \
             parameter FIX, or use FOCEI.",
            without_eta.join(", ")
        ));
    }
    if !with_eta.is_empty() {
        out.push(format!(
            "{label}: estimated parameter(s) [{}] enter an individual parameter that carries \
             an ETA, but not as a mu-referenced typical value. The closed-form mu-reference \
             shift needs `log P = log TVP + ETA_P`, linear in the ETA; a transformed ETA \
             (Box-Cox, power), a parameter inside that transform, one scaling the ETA, or a \
             covariate coefficient leaves no closed-form shift. They are therefore estimated \
             solely through the importance-weighted M-step, which is biased for \
             weakly-identified parameters and may converge to the wrong value. This is \
             expected for such a model and no reason to remove the transform: cross-check the \
             fit against FOCEI, or hold the parameter FIX.",
            with_eta.join(", ")
        ));
    }
    out
}

/// Split the class-aware mu-ref θ into those the closed-form shift can move this
/// iteration (`shift[t].is_some()`) and those it cannot (#996 review).
///
/// Only the first group may be pinned out of the weighted M-step. A θ whose
/// classes drew no responsibility mass has no shift to apply, and pinning it
/// anyway would freeze it for the remainder of the fit — neither M-stepped nor
/// shifted, with no path back, since θ never moves so the responsibilities never
/// recover. Those are returned in the second group and left free.
fn split_mu_ref_pins(pinned: &[usize], shift: &[Option<f64>]) -> (Vec<usize>, Vec<usize>) {
    pinned
        .iter()
        .copied()
        .partition(|&t| shift.get(t).copied().flatten().is_some())
}

/// One subject's multi-start MAP: the warm-start (or cold-start) solve, a cold
/// re-solve from η = 0 whenever the baseline was warm-started, plus `mceta`
/// additional random starting points drawn from N(0, Ω) via the supplied
/// Cholesky factor. The start with the lowest NLL wins; `omega_chol = None` (or
/// `mceta == 0`) degrades to the best of the warm and cold solves.
///
/// The cold re-solve is not optional (#1629). The MCEM loop warm-starts each
/// iteration's MAP from the previous iteration's mode, and the parameters move
/// far between early iterations, so a mode that was global at the initial
/// estimates can stop being global while the warm start keeps returning it. On
/// `two_cpt_oral_cov` one subject's iteration-1 mode is the low-V1/low-KA basin
/// (η_V1 ≈ −1.0, η_KA ≈ −0.8); from iteration 2 on the η ≈ 0 basin is 8.4 NLL
/// units lower, but warm-starting held the subject in the stale one for all 200
/// iterations — the proposal sat off the posterior, the reported IS −2 log L
/// read ~17 units high, and that subject's ηs inflated ω²_V1 and ω²_KA. This is
/// the MCEM counterpart of FOCEI's cold-vs-warm re-solve at its final point
/// (#833). A cold solve that only reconverges to the warm mode is rejected by
/// the strict-improvement guard, so the warm mode is kept on ties.
///
/// Shared by the single-population MAP sweep ([`run_map_multistart`]) and the
/// mixture MCEM E-step, which runs the same multi-start *per class* inside the
/// subject's rayon task (so `MIXNUM` stays correct on the worker thread).
#[allow(clippy::too_many_arguments)]
fn find_ebe_multistart(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    inner_maxiter: usize,
    inner_tol: f64,
    warm: Option<&[f64]>,
    mu: Option<&[f64]>,
    inner_restarts: usize,
    mceta: usize,
    omega_chol: Option<&DMatrix<f64>>,
    subj_seed: u64,
) -> EbeResult {
    // Baseline: warm-start (or cold-start from η = 0).
    let mut best = find_ebe(
        model,
        subject,
        params,
        inner_maxiter,
        inner_tol,
        warm,
        mu,
        inner_restarts,
    );
    if warm.is_some() {
        let cold = find_ebe(
            model,
            subject,
            params,
            inner_maxiter,
            inner_tol,
            None,
            mu,
            inner_restarts,
        );
        if cold.nll + 1e-9 < best.nll {
            best = cold;
        }
    }
    let Some(l_omega) = omega_chol else {
        return best;
    };
    let l_iov = kappa_start_chol(model, subject, params, mceta);
    let kappa_draw = l_iov
        .as_ref()
        .map(|l| (l, crate::stats::stacked_layout::group_count(subject)));
    let mut rng = StdRng::seed_from_u64(subj_seed);
    for _start in 0..mceta {
        let eta_slice = random_start(&mut rng, l_omega, kappa_draw);

        let candidate = find_ebe(
            model,
            subject,
            params,
            inner_maxiter,
            inner_tol,
            Some(&eta_slice),
            mu,
            inner_restarts,
        );
        if candidate.nll < best.nll {
            best = candidate;
        }
    }
    best
}

/// The Cholesky factor of Ω_iov an IOV subject's `mceta` random starts draw κ
/// from, or `None` for a subject that samples η alone (no IOV, no occasions, no
/// restarts, or an Ω_iov that is not positive definite).
fn kappa_start_chol(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    mceta: usize,
) -> Option<DMatrix<f64>> {
    if mceta == 0 || model.n_kappa == 0 || subject.occasions.is_empty() {
        return None;
    }
    params
        .omega_iov
        .as_ref()
        .and_then(|o| o.matrix.clone().cholesky())
        .map(|c| c.l())
}

/// One `mceta` random start: η = L_Ω·z and, for an IOV subject, each of its
/// `n_occ` κ blocks = L_iov·z, stacked as the `[η, κ₁..κ_K]` the IOV inner solve
/// starts from (#2301). Without `kappa`, η alone — the draw sequence is then
/// exactly the pre-IOV one.
fn random_start(
    rng: &mut StdRng,
    l_omega: &DMatrix<f64>,
    kappa: Option<(&DMatrix<f64>, usize)>,
) -> Vec<f64> {
    let mut draw = |l: &DMatrix<f64>| -> Vec<f64> {
        let z: Vec<f64> = (0..l.nrows()).map(|_| StandardNormal.sample(rng)).collect();
        (l * DVector::from_vec(z)).iter().copied().collect()
    };
    let mut start = draw(l_omega);
    if let Some((l_iov, n_occ)) = kappa {
        for _ in 0..n_occ {
            start.extend(draw(l_iov));
        }
    }
    start
}

/// Multi-start MAP: for each subject, run `find_ebe` with the warm-start (or
/// cold-start), a cold re-solve when it was warm-started, and then `mceta`
/// additional random starting points drawn from N(0, Ω) — with each occasion's κ
/// drawn from N(0, Ω_iov) for an IOV subject (see [`random_start`]). The start with the
/// lowest NLL wins. When `mceta == 0` this degrades to the best of the warm and
/// cold solves (see [`find_ebe_multistart`] for why the cold one is kept).
///
/// Returns `(eta_hats, h_matrices, kappas, stats)`. `kappas[i]` holds subject
/// `i`'s per-occasion κ̂ (empty without IOV); `prev_etas` may carry the stacked
/// `[η̂, κ̂₁..κ̂_K]` of the previous sweep, which warm-starts κ too (#2301).
#[allow(clippy::too_many_arguments)]
fn run_map_multistart(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    inner_maxiter: usize,
    inner_tol: f64,
    prev_etas: Option<&[DVector<f64>]>,
    mu_k: &[f64],
    mceta: usize,
    seed: u64,
    iteration: usize,
) -> (
    Vec<DVector<f64>>,
    Vec<DMatrix<f64>>,
    Vec<Vec<DVector<f64>>>,
    InnerLoopStats,
) {
    // Cholesky of Ω for drawing random starts (computed once, outside the
    // per-subject parallel loop).
    let omega_chol = if mceta > 0 {
        params.omega.matrix.clone().cholesky().map(|c| c.l())
    } else {
        None
    };

    let results: Vec<EbeResult> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let warm = prev_etas.map(|pe| pe[i].as_slice());
            // Deterministic per-subject, per-iteration seed, separated from IS draws.
            let subj_seed = seed
                .wrapping_add(i as u64)
                .wrapping_add((iteration as u64) << 32)
                .wrapping_add(0x4D43_4554_4100u64);
            find_ebe_multistart(
                model,
                subject,
                params,
                inner_maxiter,
                inner_tol,
                warm,
                Some(mu_k),
                0,
                mceta,
                omega_chol.as_ref(),
                subj_seed,
            )
        })
        .collect();

    let stats = InnerLoopStats {
        n_unconverged: results.iter().filter(|r| !r.converged).count(),
        n_fallback: results.iter().filter(|r| r.used_fallback).count(),
        n_nm_polished: results.iter().filter(|r| r.nm_polished).count(),
        n_start_rejected: results.iter().filter(|r| r.hard_reject).count(),
    };
    let eta_hats: Vec<DVector<f64>> = results.iter().map(|r| r.eta.clone()).collect();
    let h_matrices: Vec<DMatrix<f64>> = results.iter().map(|r| r.h_matrix.clone()).collect();
    let kappas: Vec<Vec<DVector<f64>>> = results.into_iter().map(|r| r.kappas).collect();

    (eta_hats, h_matrices, kappas, stats)
}

/// The importance-sampling marginal −2 log L at a finished stage's estimates, as NONMEM
/// reports it. `fit()` publishes it, and a direct [`run_imp`] / [`run_impmap`] call reads
/// it for its verdict. `options` are the stage's. A mixture uses its own evaluator.
pub(crate) fn evaluate_final_marginal(
    model: &CompiledModel,
    population: &Population,
    result: &OuterResult,
    options: &FitOptions,
    method: EstimationMethod,
) -> Result<crate::types::ImportanceSamplingResult, String> {
    use crate::estimation::importance_sampling::{
        run_importance_sampling, run_importance_sampling_mixture,
    };
    debug_assert!(matches!(
        method,
        EstimationMethod::Imp | EstimationMethod::Impmap
    ));
    let mut marg_opts = options.clone();
    // `run_importance_sampling` reads the `imp_*` knobs; for IMPMAP map the
    // `impmap_*` knobs onto them so the final eval mirrors the method's
    // own sample count / proposal df / seed.
    if method == EstimationMethod::Impmap {
        marg_opts.imp_samples = options.impmap_samples;
        marg_opts.imp_seed = options.impmap_seed;
        marg_opts.imp_low_ess_threshold = options.impmap_low_ess_threshold;
        // A Gaussian IMPMAP proposal (`impmap_proposal_df = ∞`, opt-in)
        // is one the IS evaluator could sample (#2651), but its light
        // tails leave the weights unbounded. The marginal is
        // proposal-independent in expectation, so evaluate under a
        // finite-t proposal instead (heavier tails ⇒ bounded weights).
        // The default `impmap_proposal_df = 4` passes through unchanged.
        let df = options.impmap_proposal_df;
        marg_opts.imp_proposal_df = if df.is_finite() && df >= 1.0 { df } else { 5.0 };
    }
    if model.mixture.is_some() {
        run_importance_sampling_mixture(model, population, &result.params, &marg_opts)
    } else {
        run_importance_sampling(
            model,
            population,
            &result.params,
            &result.eta_hats,
            &result.h_matrices,
            &result.kappas,
            &marg_opts,
        )
    }
}

/// Give a direct [`run_imp`] / [`run_impmap`] caller the settling verdict `fit()` gives
/// its stages (#2686), which needs the final marginal a stage does not compute. If the
/// marginal fails, the falling-floor test still runs and a warning says so.
fn apply_settling_verdict(
    mut result: OuterResult,
    model: &CompiledModel,
    population: &Population,
    options: &FitOptions,
    method: EstimationMethod,
) -> OuterResult {
    use crate::api::postfit::mcem_settling_warning;
    let marginal = match evaluate_final_marginal(model, population, &result, options, method) {
        Ok(is) => Some(FinalMarginal {
            value: is.minus2_log_likelihood,
            mc_se: is.mc_standard_error,
        }),
        Err(e) => {
            result
                .warnings
                .push(format!("marginal −2 log L eval skipped: {e}"));
            None
        }
    };
    let trail = result.mcem_objective_trail.take();
    if let Some((msg, _)) = mcem_settling_warning(&mut result.converged, trail.as_ref(), marginal) {
        result.warnings.push(msg);
    }
    result.mcem_objective_trail = trail;
    result
}

/// Run IMPMAP (NONMEM `METHOD=IMPMAP`), as a complete estimation. `warm_etas`, when
/// supplied, seed the first MAP inner loop; otherwise it cold-starts from η = 0.
///
/// `converged` includes the settling verdict, so this also scores the final marginal.
pub fn run_impmap(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let result = run_impmap_stage(model, population, init_params, warm_etas, options)?;
    Ok(apply_settling_verdict(
        result,
        model,
        population,
        options,
        EstimationMethod::Impmap,
    ))
}

/// One IMPMAP stage of a `fit()`, which scores the marginal and gives the settling
/// verdict itself. Thin wrapper over the shared MCEM core with mode re-centering on
/// every iteration; resolves the `impmap_*` options.
pub(crate) fn run_impmap_stage(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let nu = options.impmap_proposal_df;
    run_mcem(
        model,
        population,
        init_params,
        warm_etas,
        options,
        ProposalRecenter::Map,
        "IMPMAP",
        "impmap_proposal_df",
        options.impmap_iterations,
        options.impmap_samples,
        nu,
        options.impmap_averaging,
        options.impmap_seed.unwrap_or(12345),
        options.impmap_low_ess_threshold,
        options.impmap_mceta,
        // Sobol now drives every proposal form — MVN, Student-t, and the
        // defensive mixture (#1965) — so there is no combination to warn about.
        options.impmap_sobol,
        options.impmap_trace,
        options.impmap_auto,
        options.impmap_max_time,
    )
}

/// Run IMP as an estimator (NONMEM `METHOD=IMP`), as a complete estimation. The
/// evaluation-only `imp_eval_only` path lives in `importance_sampling.rs`.
///
/// `converged` includes the settling verdict, so this also scores the final marginal.
pub fn run_imp(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let result = run_imp_stage(model, population, init_params, warm_etas, options)?;
    Ok(apply_settling_verdict(
        result,
        model,
        population,
        options,
        EstimationMethod::Imp,
    ))
}

/// One IMP stage of a `fit()`, which scores the marginal and gives the settling verdict
/// itself. Thin wrapper over the shared MCEM core with sample-moment re-centering
/// (conditional mode found only on the first iteration); resolves the `imp_*` options.
pub(crate) fn run_imp_stage(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    run_mcem(
        model,
        population,
        init_params,
        warm_etas,
        options,
        ProposalRecenter::SampleMoments,
        "IMP",
        "imp_proposal_df",
        options.imp_iterations,
        options.imp_samples,
        options.imp_proposal_df,
        options.imp_averaging,
        options.imp_seed.unwrap_or(12345),
        options.imp_low_ess_threshold,
        0, // mceta: no multi-start MAP for IMP
        options.imp_sobol,
        false, // collect_trace: IMP has no trace option
        options.imp_auto,
        options.imp_max_time,
    )
}

/// Refuse a Sobol configuration the quasi-random sampler cannot serve, instead
/// of letting `Sobol::new` panic or drawing from a wrong t (#1965):
///
/// - a draw spends `n_eta` coordinates on the normals, one more on the
///   Student-t scale when ν is finite, and one more on the mixture component
///   when the defensive mixture is on — at most `SOBOL_MAX_DIMS` in total;
/// - the t scale comes from `chi_squared_inv_cdf`, validated only up to
///   `CHI_SQ_INV_MAX_NU`; a t with more degrees of freedom is a normal for
///   every practical purpose, so the message points there.
fn check_sobol_config(
    label: &str,
    df_key: &str,
    n_eta: usize,
    use_sobol: bool,
    nu: f64,
    mixture: bool,
) -> Result<(), String> {
    use crate::estimation::importance_sampling::SOBOL_MAX_DIMS;
    use crate::stats::special::CHI_SQ_INV_MAX_NU;
    if !use_sobol {
        return Ok(());
    }
    let dims = n_eta + usize::from(nu.is_finite()) + usize::from(mixture);
    if dims > SOBOL_MAX_DIMS {
        return Err(format!(
            "{label}: Sobol draws need {dims} quasi-random coordinates here ({n_eta} random \
             effects{}{}) but support at most {SOBOL_MAX_DIMS}; turn off `impmap_sobol` / \
             `imp_sobol`.",
            if nu.is_finite() {
                " + the Student-t scale"
            } else {
                ""
            },
            if mixture {
                " + the defensive-mixture component"
            } else {
                ""
            },
        ));
    }
    if nu.is_finite() && nu > CHI_SQ_INV_MAX_NU {
        return Err(format!(
            "{label}: Sobol draws support a Student-t proposal with at most \
             {CHI_SQ_INV_MAX_NU} degrees of freedom (`{df_key}` = {nu}); a t that wide is \
             a normal in practice — set `{df_key} = normal`."
        ));
    }
    Ok(())
}

/// Shared Monte-Carlo EM core for IMP and IMPMAP. The `recenter` strategy is the
/// only behavioural difference; `label`/`df_key` tag warnings and verbose output.
#[allow(clippy::too_many_arguments)]
fn run_mcem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
    recenter: ProposalRecenter,
    label: &str,
    df_key: &str,
    n_iter_opt: usize,
    k_opt: usize,
    nu: f64,
    n_avg_opt: usize,
    seed: u64,
    threshold: f64,
    mceta: usize,
    use_sobol: bool,
    collect_trace: bool,
    auto: bool,
    max_time: Option<f64>,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();
    let started = std::time::Instant::now();
    let max_time_key = recenter.max_time_key();

    // ---- Validation ----
    check_max_time(label, max_time_key, max_time)?;
    check_sobol_config(
        label,
        df_key,
        n_eta,
        use_sobol,
        nu,
        options.imp_defensive_alpha > 0.0,
    )?;
    // Mixture (#985): estimating IMP/IMPMAP runs a class-partitioned MCEM
    // (per-class IS E-step + responsibility-weighted M-step). Delegate to the
    // mixture core, which reuses this function's option resolution via the same
    // arguments.
    if model.mixture.is_some() {
        return run_mcem_mixture(
            model,
            population,
            init_params,
            warm_etas,
            options,
            recenter,
            label,
            df_key,
            n_iter_opt,
            k_opt,
            nu,
            n_avg_opt,
            seed,
            threshold,
            mceta,
            use_sobol,
            collect_trace,
            auto,
            max_time,
        );
    }
    if recenter.method().requires_random_effects() && n_eta == 0 {
        return Err(format!(
            "{label} requires at least one random effect (n_eta = 0). \
             Use FOCE/FOCEI for fixed-effects-only models."
        ));
    }
    if model.is_sde() {
        return Err(format!(
            "{label} is not yet supported for SDE / [diffusion] models \
             (the EKF process-noise variance is not threaded through the IS \
             observation likelihood). Use FOCE / FOCEI instead."
        ));
    }
    // IOV (#2301): each IOV subject samples its stacked `[η, κ₁..κ_K]` jointly.
    // Nested variability levels (#1082) lay κ out per cell of several levels,
    // which the joint proposal below does not model.
    if model.n_kappa > 0 && model.has_nested_levels() {
        return Err(format!(
            "{label} does not yet support nested variability levels (#1082). Use FOCE or \
             FOCEI for this model."
        ));
    }
    let omega_iov_init: Option<&OmegaMatrix> = if model.n_kappa > 0 {
        match init_params.omega_iov.as_ref() {
            Some(iov) if iov.log_det.is_finite() => Some(iov),
            Some(_) => {
                return Err(format!(
                    "{label}: initial Ω_iov log-determinant is not finite — check the κ \
                     declarations."
                ))
            }
            None => {
                return Err(format!(
                    "{label}: the model declares κ (IOV) but no Ω_iov was supplied."
                ))
            }
        }
    } else {
        None
    };
    if !init_params.omega.log_det.is_finite() {
        return Err(format!(
            "{label}: initial Ω log-determinant is not finite — check the \
             [parameters] Ω block."
        ));
    }

    let n_iter = n_iter_opt.max(1);
    // `k_samples` is mutable so the adaptive (`auto`) path can move it between
    // iterations when the objective is too Monte-Carlo-noisy (NONMEM `AUTO`).
    let k_start = k_opt.max(2);
    let mut k_samples = k_start;
    // `INFINITY` selects the multivariate-normal proposal; anything else must be
    // a valid Student-t DoF (>= 1). Guard here so a programmatic caller that
    // bypasses the parser's range check can't reach the `ChiSquared::new(nu)`
    // panic in `subject_is_draws`, and a NaN can't pass for a normal proposal by
    // being non-finite. Mirrors `run_importance_sampling`.
    if nu.is_nan() || nu < 1.0 {
        return Err(format!(
            "{label}: {df_key} must be >= 1.0 (or +inf for a normal proposal), got {nu}"
        ));
    }
    let n_avg = n_avg_opt.min(n_iter);
    let auto_hold_end = auto_hold_ends(n_iter, n_avg);
    let verbose = options.verbose;
    let cancel = &options.cancel;

    if verbose {
        let prop = if nu.is_finite() {
            format!("t_{nu}")
        } else {
            "normal".to_string()
        };
        let recenter_desc = match recenter {
            ProposalRecenter::Map => "MAP recenter/iter",
            ProposalRecenter::SampleMoments => "sample-moment recenter",
        };
        let mceta_msg = if mceta > 0 {
            format!(", MCETA={}", mceta)
        } else {
            String::new()
        };
        eprintln!(
            "{}: {} subjects, {} ETAs, {} iters, K={}/subject, {} proposal, {}, seed={}{}",
            label, n_subjects, n_eta, n_iter, k_samples, prop, recenter_desc, seed, mceta_msg
        );
    }

    // ---- Packing scaffolding (mirrors SAEM) ----
    // Per-theta packing: log for `theta_lower >= 0`, identity otherwise (so
    // covariate exponents with negative lower bounds are not pinned to ~0).
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };

    let mut log_theta: Vec<f64> = (0..n_theta)
        .map(|i| pack_theta(i, init_params.theta[i]))
        .collect();
    let mut log_sigma: Vec<f64> = init_params
        .sigma
        .values
        .iter()
        .map(|&s| s.max(1e-10).ln())
        .collect();

    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: lower == upper == packed value.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

    // Closed-form mu-referencing M-step: shift `packed θ += mean(η)` for eligible
    // mu-ref pairs (log-packed lognormal, identity-packed logit — #918), with those
    // θ pinned out of the NLopt weighted M-step (which then fits
    // only σ and non-mu-ref θ). This is the EM-correct typical-value update for
    // mu-referenced random effects, NOT an optional refinement: without it θ and the
    // η mean are confounded over the fixed importance samples, so θ stays at its
    // start and Ω inflates to absorb the misfit. It is therefore applied whenever
    // eligible pairs exist, independent of `options.mu_referencing` (which only
    // governs inner-loop `compute_mu_k` centering, a separate concern). NONMEM's
    // EM methods likewise require mu-referencing.
    let mut warnings: Vec<String> = Vec::new();

    // Same eligibility rule as SAEM: a pair qualifies when the packed theta *is*
    // the mu (log-packed lognormal, identity-packed logit — see
    // `classify_mu_ref_pairs`), which is exactly what the `log_theta[t] +=
    // eta_bar[e]` shift below assumes.
    let mu_ref_split = classify_mu_ref_pairs(model, &init_params.theta_lower);
    // What a #619 covariate group may not collide with — every declared anchor
    // pair, including the ones the shared-anchor rule just dropped.
    let cov_group_conflicts = mu_ref_pairs_for_cov_groups(&mu_ref_split);
    let MuRefPairs {
        eligible: mu_ref_pairs,
        // SAEM-only (#2540): this M-step steps the packed θ, which is not an
        // additive anchor's mu when the θ is log-packed.
        additive: _,
        identity_packed_log,
        log_packed_logit,
        shared_theta,
        shared_pairs: _,
    } = mu_ref_split;
    // Multi-theta (covariate) mu-references (#619): a group supersedes the
    // single-anchor pair on its own eta, and its thetas leave the packing
    // advisories below (the group step moves them, packing or not).
    let (cov_mu_groups, cov_mu_notes) = resolve_covariate_mu_groups(
        model,
        population,
        &cov_group_conflicts,
        &init_params.theta_fixed,
        &init_params.omega.matrix,
    );
    for n in &cov_mu_notes {
        warnings.push(format!("{label}: {n}"));
    }
    // Every member's eta (#1620): see the SAEM twin in `run_saem`.
    let cov_group_etas: Vec<usize> =
        crate::estimation::covariate_mu_ref::group_etas(&cov_mu_groups);
    let cov_group_thetas: Vec<usize> = cov_mu_groups
        .iter()
        .flat_map(|g| g.theta_idx.iter().copied())
        .collect();

    // STRONG warning: any estimated (non-fixed) θ the closed-form mu-ref shift
    // cannot move is handled only by the weighted M-step, which is biased for
    // such parameters under importance sampling. One with no ETA at all needs
    // one attached; one whose ETA is there in a non-mu-referenceable form
    // (#1715) needs a cross-check instead.
    //
    // Assembled *after* the groups resolve, and from the resolved list: a
    // theta whose #619 group was dropped (weak IIV, a conflicting anchor, an
    // all-FIXed list) has no group moving it, so it belongs in this warning
    // (#918 review).
    let (thetas_with_eta, thetas_without_eta) = partition_by_eta(
        model,
        non_fixed_unanchored_thetas(model, &init_params.theta_fixed, &cov_group_thetas, &[]),
    );
    warnings.extend(imp_unanchored_theta_warnings(
        label,
        &thetas_with_eta,
        &thetas_without_eta,
    ));

    let mu_ref_pairs: Vec<(usize, usize)> = mu_ref_pairs
        .into_iter()
        .filter(|(_t, e)| !cov_group_etas.contains(e))
        .collect();
    let identity_packed_log: Vec<usize> = identity_packed_log
        .into_iter()
        .filter(|t| !cov_group_thetas.contains(t))
        .collect();
    let log_packed_logit: Vec<usize> = log_packed_logit
        .into_iter()
        .filter(|t| !cov_group_thetas.contains(t))
        .collect();
    let theta_name_list = |idx: &[usize]| -> String {
        let mut n: Vec<&str> = idx
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        n.sort_unstable();
        n.dedup();
        n.join(", ")
    };
    // A lognormal θ declared with a negative lower bound is packed on the
    // identity scale, so `log_theta[t]` holds the raw value and the additive
    // `+= mean(η)` shift is not its EM optimum (it would apply `θ += mean(η)`
    // where the closed form means `θ *= exp(mean(η))`). Route those to the
    // weighted M-step, the same guard the mixture path applies (#996 review).
    if !identity_packed_log.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but declared with a negative \
             lower bound, so they are packed on the identity scale; the closed-form mu-ref \
             shift does not apply and they are estimated by the importance-weighted M-step \
             instead (#996).",
            theta_name_list(&identity_packed_log)
        ));
    }
    // Mirror image for a logit-scale theta (#918): `inv_logit(THETA + ETA)` has
    // mu scale θ, but a non-negative lower bound makes it log-packed.
    if !log_packed_logit.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are logit-mu-referenced but declared with a \
             non-negative lower bound, so they are packed on the log scale; the closed-form \
             mu-ref shift does not apply and they are estimated by the importance-weighted \
             M-step instead. Declare the logit-scale theta with a negative lower bound to use \
             the closed-form update (#918).",
            theta_name_list(&log_packed_logit)
        ));
    }
    // A θ anchoring two ETAs has no single closed-form shift (see
    // `classify_mu_ref_pairs`); both its pairs are dropped, so it is estimated by
    // the weighted M-step alone.
    // A covariate group that took both pairs in moves the θ instead (#1630).
    let shared_theta: Vec<usize> = shared_theta
        .into_iter()
        .filter(|t| !cov_group_thetas.contains(t))
        .collect();
    if !shared_theta.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are the mu-reference anchor of more than one ETA, so \
             the closed-form mean shift has no single well-defined value for them; they are \
             estimated by the importance-weighted M-step instead — the channel that is biased \
             for weakly-identified parameters (#406). Give each ETA its own typical value if \
             the closed-form update is wanted.",
            theta_name_list(&shared_theta)
        ));
    }
    // A closed-form-eligible typical value is updated only through the
    // `packed θ += mean(η)` shift. When its paired η carries negligible IIV (a tiny,
    // often `FIX`ed ω — e.g. a structural parameter given a dummy random effect
    // so it can be mu-referenced), that population mean is ≈ 0 and the typical
    // value would be frozen at its initial value (#411). Route those pairs to the
    // weighted-likelihood M-step instead (the same channel that estimates σ and
    // non-mu-ref θ), where the data can actually move them. Decided once from the
    // initial Ω so a parameter never toggles channels between iterations.
    const WEAK_IIV_VAR: f64 = 1e-3;
    let weak_mu_ref: std::collections::HashSet<usize> = mu_ref_pairs
        .iter()
        .filter(|&&(_t, e)| init_params.omega.matrix[(e, e)] < WEAK_IIV_VAR)
        .map(|&(t, _e)| t)
        .collect();
    if !weak_mu_ref.is_empty() {
        let weak_list: Vec<usize> = weak_mu_ref.iter().copied().collect();
        warnings.push(format!(
            "{label}: typical value(s) {} are mu-referenced but their random effect has \
             negligible variance (ω < {WEAK_IIV_VAR:.0e}); the mu-ref mean-shift carries no \
             information, so they are estimated through the weighted M-step instead.",
            theta_name_list(&weak_list)
        ));
    }
    // Closed-form mu-ref shift is used as long as at least one pair has real IIV.
    let use_closed_form = mu_ref_pairs
        .iter()
        .any(|&(t, _e)| !weak_mu_ref.contains(&t));
    if !use_closed_form && cov_mu_groups.is_empty() {
        // No closed-form-eligible parameter: every typical value goes through the
        // weighted M-step, which cannot resolve the θ/η-mean confounding on its own.
        // Flag it — estimates may be unreliable (see the docs caveat).
        warnings.push(format!(
            "{label}: no closed-form mu-referenced parameters found (e.g. \
             `CL = TVCL*exp(ETA)`, or `F = inv_logit(LOGIT_F + ETA)` with LOGIT_F declared \
             on the logit scale); typical-value estimation relies on the weighted M-step \
             alone and may converge poorly. Prefer such a parameterization, or use FOCEI."
        ));
    }

    // ---- Iteration state ----
    let mut theta_cur = init_params.theta.clone();
    let mut sigma_cur = init_params.sigma.values.clone();
    let mut omega_mat = init_params.omega.matrix.clone();
    // IOV (#2301): Ω_iov, and each subject's number of κ blocks — the occasion
    // groups its stacked `[η, κ₁..κ_K]` is sampled over. A subject with none
    // (or a model without IOV) samples η alone, exactly as before.
    let n_kappa = model.n_kappa;
    let mut omega_iov_mat: Option<DMatrix<f64>> = omega_iov_init.map(|o| o.matrix.clone());
    let n_occ_of: Vec<usize> = population
        .subjects
        .iter()
        .map(|s| {
            if n_kappa > 0 && !s.occasions.is_empty() {
                crate::stats::stacked_layout::group_count(s)
            } else {
                0
            }
        })
        .collect();
    let schedules = mcem_schedule_cache(model, population, &n_occ_of);
    // MAP warm starts, stacked `[η̂, κ̂₁..κ̂_K]` for an IOV subject so κ is
    // warm-started too. A chained stage's η-only warm start leaves κ at 0.
    let mut prev_etas: Option<Vec<DVector<f64>>> = warm_etas.map(|e| e.to_vec());
    // Previous iteration's per-subject weighted draws — the proposal source for
    // the IMP (`SampleMoments`) recenter path on iterations 2+. `None` on the
    // first iteration (and always for IMPMAP, which never reads it).
    let mut prev_draws: Option<Vec<SubjectDraws>> = None;

    // Running mean of parameters over the final `n_avg` iterations.
    let mut acc_theta = vec![0.0f64; n_theta];
    let mut acc_sigma = vec![0.0f64; n_sigma];
    let mut acc_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut acc_omega_iov = DMatrix::<f64>::zeros(n_kappa, n_kappa);
    let mut n_acc = 0usize;

    let mut last_eta_hats: Vec<DVector<f64>> = Vec::new();

    // ---- FREM Rao-Blackwellisation (issue #406) ----
    // For FREM models, integrate the covariate etas analytically and importance
    // sample only the PK etas (a well-conditioned low-dim problem with near-unit
    // ESS) instead of all n_eta etas (~1–2% ESS). Partition is model-static;
    // per-subject covariate deviations are computed inside the E-step. `None` for
    // non-FREM models → the full-dimensional path is used unchanged.
    let frem_rb: Option<(Vec<usize>, Vec<usize>)> = if !options.frem_rao_blackwell {
        None
    } else {
        model
            .frem_config
            .as_ref()
            // The Rao-Blackwell partition is over η alone; an IOV model samples
            // the joint (η, κ) vector instead, as `run_importance_sampling` does.
            .filter(|_| n_kappa == 0)
            .map(|fc| crate::estimation::importance_sampling::frem_pk_cov_partition(fc, n_eta))
            .filter(|(pk, cov)| !pk.is_empty() && !cov.is_empty())
    };

    // ---- Trace: collect per-iteration parameters (analogous to NONMEM .ext) ----
    let mut trace_rows: Vec<ImpmapTraceRow> = if collect_trace {
        Vec::with_capacity(n_iter + 2)
    } else {
        Vec::new()
    };

    // ESS state from the final completed E-step, used to flag importance-sampling
    // moment bias (which scales as 1/(K·ESS) and is severe in high dimensions).
    let mut final_ess_median = 1.0_f64;
    let mut final_n_collapsed = 0usize;
    // Subjects `auto` left out of the sample-count decision at the final E-step.
    let mut final_stuck_ids: Vec<&str> = Vec::new();
    let mut stuck_subjects = StuckSubjects::default();
    let subject_ids: Vec<&str> = population.subjects.iter().map(|s| s.id.as_str()).collect();
    // Effective (possibly ramped) sample count and the final objective MC SE,
    // for the post-fit under-sampling warning.
    let mut final_k_samples = k_samples;
    let mut final_mc_se = 0.0_f64;
    // Iterations on which each #619 group had no admissible step, so neither it
    // nor the weighted M-step moved that group's thetas. Reported once after the
    // loop — silence here reads as a converged estimate of a theta that never
    // left its initial value (#918 review).
    let mut cov_group_skipped = vec![0usize; cov_mu_groups.len()];

    // The lowest −2 log L any iteration reached, for `fit()`'s rise check (#2652).
    let mut lowest_iteration_objective: Option<f64> = None;
    // Every iteration's −2 log L and total MC SE, and its (per-subject SE, K) for `auto`.
    let mut iteration_objectives: Vec<f64> = Vec::with_capacity(n_iter);
    let mut iteration_mc_se: Vec<f64> = Vec::with_capacity(n_iter);
    let mut iteration_se: Vec<(f64, usize)> = Vec::with_capacity(n_iter);
    // Each iteration's ESS fractions and K, so `iteration_se` can be rebuilt
    // when the set of subjects `auto` acts on changes (#2690).
    let mut ess_history: Vec<(Vec<f64>, usize)> = Vec::new();
    let mut prev_stuck: Vec<bool> = vec![false; population.subjects.len()];
    // Iterations actually run, and the seconds at which `max_time` stopped them.
    let mut iterations_done = 0usize;
    let mut stopped_after: Option<f64> = None;
    let mut mstep_motion = MstepMotion::new(n_theta + n_sigma);
    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(cancel) {
            if verbose {
                eprintln!("{}: cancelled at iteration {}", label, k);
            }
            break;
        }

        // Assemble current params for the inner loop / E-step.
        let omega_k = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        let params_k = ModelParameters {
            theta: theta_cur.clone(),
            theta_names: init_params.theta_names.clone(),
            theta_lower: init_params.theta_lower.clone(),
            theta_upper: init_params.theta_upper.clone(),
            theta_fixed: init_params.theta_fixed.clone(),
            omega: omega_k,
            omega_fixed: init_params.omega_fixed.clone(),
            sigma: SigmaVector {
                values: sigma_cur.clone(),
                names: init_params.sigma.names.clone(),
            },
            sigma_fixed: init_params.sigma_fixed.clone(),
            // IMP/IMPMAP does not update the `block_sigma` off-diagonals
            // (#847); carry them so the sampled residual `R` keeps its
            // declared correlation structure.
            residual_correlations: init_params.residual_correlations.clone(),
            residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
            omega_iov: omega_iov_snapshot(init_params, omega_iov_mat.as_ref()),
            kappa_fixed: init_params.kappa_fixed.clone(),
            mixture: None,
        };

        // ---- E-step A: position the proposal ----
        // IMPMAP (`Map`) re-runs the MAP inner loop every iteration. IMP
        // (`SampleMoments`) runs it only on the first iteration — when
        // `prev_draws` is still `None` — to seed the proposal, then recenters
        // from the previous iteration's weighted moments inside the draws loop.
        let run_inner = recenter == ProposalRecenter::Map || prev_draws.is_none();
        let (eta_hats, h_matrices, kappa_hats) = if run_inner {
            let mu_k = compute_mu_k(model, &params_k.theta, options.mu_referencing);
            let (e, h, kap, _stats) = run_map_multistart(
                model,
                population,
                &params_k,
                options.inner_maxiter,
                options.inner_tol,
                prev_etas.as_deref(),
                &mu_k,
                mceta,
                seed,
                k,
            );
            check_kappa_blocks(population, &kap, &n_occ_of, label)?;
            (e, h, kap)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        let omega_inv = params_k.omega.inv.clone();
        let log_det_omega = params_k.omega.log_det;

        // ---- E-step B: importance sampling around each mode ----
        let iscale_min = options.iscale_min;
        let iscale_max = options.iscale_max;
        let defensive_alpha = options.imp_defensive_alpha;
        // Build the defensive-mixture broad component q_broad = N(0, Ω) once per
        // MCEM iteration (Ω changes each iteration but is shared across subjects)
        // and pass it read-only into the parallel subject loop, avoiding a
        // redundant per-subject Cholesky of Ω⁻¹ (issue #528). `None` when the
        // mixture is inactive. The FREM RB path builds its own component.
        let defensive_mixture = crate::estimation::importance_sampling::DefensiveMixture::new(
            &omega_inv,
            n_eta,
            defensive_alpha,
        );
        // The IOV twin (#2301): the joint prior `Ω ⊕ I_K ⊗ Ω_iov` and its defensive
        // component depend only on the occasion count K, so build one per distinct
        // K per iteration and share it read-only across subjects.
        let joint_priors = params_k
            .omega_iov
            .as_ref()
            .map(|iov| JointPriors::build(&params_k.omega, iov, &n_occ_of, defensive_alpha));
        let mut draws: Vec<_> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                // Poll per subject: the inner-loop MAP + IS draws below are the
                // dominant per-iteration cost, so without this a cancel set
                // mid-iteration is not seen until the next iteration's top-of-loop
                // check (line ~257) — minutes on a large dataset. Mirrors
                // `run_importance_sampling`. The driver breaks right after the
                // collect, so the placeholder draws never reach the M-step.
                if crate::cancel::is_cancelled(cancel) {
                    return SubjectDraws::cancelled(n_eta);
                }
                let subj_seed = seed.wrapping_add(i as u64).wrapping_add((k as u64) << 32);

                // IOV subject (#2301): sample the stacked `b = [η, κ₁..κ_K]` from one
                // joint proposal against the block-diagonal prior `Ω ⊕ I_K ⊗ Ω_iov`,
                // the construction `run_importance_sampling` already uses for the
                // final objective. Every downstream moment is then over `b`.
                let n_occ = n_occ_of[i];
                if n_occ > 0 {
                    let prior = joint_priors
                        .as_ref()
                        .expect("Ω_iov present for an IOV model (checked on entry)")
                        .get(n_occ);
                    let n_b = stacked_width(n_eta, n_kappa, n_occ);
                    let omega_joint_inv = &prior.inv;
                    let log_det_joint = prior.log_det;
                    let (center, h_post) = if run_inner {
                        let h = compute_joint_posterior_hessian(
                            model,
                            subject,
                            &params_k.theta,
                            &eta_hats[i],
                            &kappa_hats[i],
                            &params_k.sigma.values,
                            &h_matrices[i],
                            omega_joint_inv,
                            n_eta,
                            n_kappa,
                            n_occ,
                            scratch,
                        );
                        (stack_eta_kappa(&eta_hats[i], &kappa_hats[i]), h)
                    } else {
                        let pd = &prev_draws.as_ref().expect("prev_draws set when !run_inner")[i];
                        let center = DVector::from_row_slice(&pd.mean);
                        let cov = &pd.second_moment - &center * center.transpose();
                        let h = covariance_to_proposal_hessian(
                            &cov,
                            &prior.matrix,
                            IMP_PROPOSAL_COV_FLOOR,
                        );
                        (center, h)
                    };
                    let theta = &params_k.theta;
                    let sigma = &params_k.sigma.values;
                    let obs_nll = |b: &[f64], sc: &mut EventPkParams| {
                        draw_obs_nll(model, subject, theta, sigma, b, schedules[i].as_ref(), sc)
                    };
                    let iscale = find_optimal_iscale_with(
                        &obs_nll,
                        &center,
                        &h_post,
                        omega_joint_inv,
                        log_det_joint,
                        n_b,
                        nu,
                        subj_seed,
                        scratch,
                        iscale_min,
                        iscale_max,
                    );
                    return subject_is_draws_with(
                        &obs_nll,
                        &center,
                        &h_post,
                        omega_joint_inv,
                        log_det_joint,
                        n_b,
                        k_samples,
                        nu,
                        subj_seed,
                        scratch,
                        iscale,
                        use_sobol,
                        prior.mixture.as_ref(),
                    );
                }

                let (center, h_post) = if run_inner {
                    // Proposal centred at the conditional mode with
                    // first-order-variance (Sheiner–Beal posterior) scale.
                    let h_post = compute_posterior_hessian(
                        model,
                        subject,
                        &params_k.theta,
                        &eta_hats[i],
                        &params_k.sigma.values,
                        &h_matrices[i],
                        &omega_inv,
                        n_eta,
                        scratch,
                    );
                    (eta_hats[i].clone(), h_post)
                } else {
                    // IMP, iterations 2+: centre at the previous iteration's
                    // weighted posterior mean m̂, scale at the previous weighted
                    // posterior covariance Ŝ − m̂m̂ᵀ (passed as its inverse).
                    let pd = &prev_draws.as_ref().expect("prev_draws set when !run_inner")[i];
                    let center = DVector::from_row_slice(&pd.mean);
                    let cov = &pd.second_moment - &center * center.transpose();
                    let h_post = covariance_to_proposal_hessian(
                        &cov,
                        &params_k.omega.matrix,
                        IMP_PROPOSAL_COV_FLOOR,
                    );
                    (center, h_post)
                };

                // FREM: Rao-Blackwellised low-dimensional PK sampling. The
                // conditional PK proposal is usually well matched, but a
                // per-subject ISCALE pilot search still rescues subjects where
                // `h_pp` is a poor curvature estimate (sparse PK data) — see
                // `find_optimal_iscale_frem_rb` (issue #406 follow-up).
                if let Some((ref pk_idx, ref cov_idx)) = frem_rb {
                    if let Some(fc) = model.frem_config.as_ref() {
                        if let Some((sampled, observed, d)) =
                            crate::estimation::importance_sampling::subject_frem_partition(
                                subject,
                                &params_k.theta,
                                fc,
                                pk_idx,
                                cov_idx,
                            )
                        {
                            let rb_iscale =
                                crate::estimation::importance_sampling::find_optimal_iscale_frem_rb(
                                    model,
                                    subject,
                                    &params_k.theta,
                                    &params_k.sigma.values,
                                    &center,
                                    &h_post,
                                    &omega_inv,
                                    &params_k.omega.matrix,
                                    &sampled,
                                    &observed,
                                    &d,
                                    n_eta,
                                    nu,
                                    subj_seed,
                                    scratch,
                                    iscale_min,
                                    iscale_max,
                                );
                            if let Some(rb) =
                                crate::estimation::importance_sampling::subject_is_draws_frem_rb(
                                    model,
                                    subject,
                                    &params_k.theta,
                                    &params_k.sigma.values,
                                    &center,
                                    &h_post,
                                    &omega_inv,
                                    &params_k.omega.matrix,
                                    &sampled,
                                    &observed,
                                    &d,
                                    n_eta,
                                    k_samples,
                                    nu,
                                    subj_seed,
                                    scratch,
                                    rb_iscale,
                                    use_sobol,
                                    defensive_alpha,
                                )
                            {
                                return rb;
                            }
                        }
                    }
                }

                let iscale = find_optimal_iscale(
                    model,
                    subject,
                    &params_k.theta,
                    &params_k.sigma.values,
                    &center,
                    &h_post,
                    &omega_inv,
                    log_det_omega,
                    n_eta,
                    nu,
                    subj_seed,
                    scratch,
                    iscale_min,
                    iscale_max,
                    schedules[i].as_ref(),
                );
                subject_is_draws(
                    model,
                    subject,
                    &params_k.theta,
                    &params_k.sigma.values,
                    &center,
                    &h_post,
                    &omega_inv,
                    log_det_omega,
                    n_eta,
                    k_samples,
                    nu,
                    subj_seed,
                    scratch,
                    iscale,
                    use_sobol,
                    defensive_mixture.as_ref(),
                    schedules[i].as_ref(),
                )
            })
            .collect();

        // If a cancel was observed inside the E-step, the `draws` are placeholders;
        // break before the M-steps consume them. The post-loop check returns Err.
        if crate::cancel::is_cancelled(cancel) {
            if verbose {
                eprintln!("{}: cancelled during E-step at iteration {}", label, k);
            }
            break;
        }

        // ESS diagnostics + marginal log-likelihood for the trace.
        let mut ll = 0.0f64;
        let mut n_low_ess = 0usize;
        let mut ess_fracs: Vec<f64> = Vec::with_capacity(draws.len());
        for d in &draws {
            ll += d.log_marginal;
            if d.ess_fraction < threshold {
                n_low_ess += 1;
            }
            ess_fracs.push(d.ess_fraction);
        }
        let minus2ll = -2.0 * ll;
        lowest_iteration_objective = lowest_objective(lowest_iteration_objective, minus2ll);
        // Objective Monte-Carlo standard error from the self-normalized ESS
        // (Geweke: Var(log p̂ᵢ) ≈ (1/ESS_frac − 1)/K; degenerate subjects get a
        // finite fallback). −2 log L scales this by 2.
        let var_ll: f64 = ess_fracs
            .iter()
            .map(|&f| log_marginal_mc_variance(f, k_samples))
            .sum();
        let mc_se = 2.0 * var_ll.sqrt();
        iteration_objectives.push(minus2ll);
        iteration_mc_se.push(mc_se);
        // Per-subject objective SE (N-independent): see AUTO_STDOBJ_TARGET.
        let mc_se_per_subject = mc_se / (ess_fracs.len().max(1) as f64).sqrt();
        final_mc_se = mc_se_per_subject;
        final_k_samples = k_samples;
        // The ramp acts on the SE without the subjects more draws have not helped,
        // so one of them cannot send every subject to the cap (#2690).
        let stuck = stuck_subjects.update(&ess_fracs, k_samples);
        // Adaptive sampling (NONMEM `AUTO`): move K so the M-step is driven by
        // signal, not noise. Raising waits for the objective to settle, because
        // early in a run the proposal lags the parameters and the SE reflects that
        // lag, not what the converged fit needs (#2687).
        if auto {
            let (ramp_se, n_ramp) = ramp_se_per_subject(&ess_fracs, k_samples, &stuck);
            ess_history.push((ess_fracs.clone(), k_samples));
            if stuck == prev_stuck {
                iteration_se.push((ramp_se, k_samples));
            } else {
                iteration_se = ramp_se_history(&ess_history, &stuck);
            }
            prev_stuck.clone_from(&stuck);
            let (new_k, se) = auto_step(
                &iteration_objectives,
                &iteration_se,
                k_start,
                n_ramp,
                k >= auto_hold_end,
            );
            if verbose && new_k != k_samples {
                let verb = if new_k > k_samples {
                    "raising"
                } else {
                    "lowering"
                };
                let n_degenerate = ess_fracs
                    .iter()
                    .filter(|&&f| is_degenerate(f, k_samples))
                    .count();
                eprintln!(
                    "{label}: auto-sampling — iteration {k}, per-subject objective MC SE \
                     {se:.3} (target {AUTO_STDOBJ_TARGET:.2}), {verb} K {k_samples} → {new_k}; \
                     {n_degenerate} subject(s) at ESS < {DEGENERATE_ESS}, {} left out; \
                     lowest ESS: {}",
                    ess_fracs.len() - n_ramp,
                    lowest_ess_summary(&subject_ids, &ess_fracs, k_samples, 5)
                );
            }
            k_samples = new_k;
        }
        // Track the final E-step's ESS health for the post-fit bias warning.
        // `f ≥ 1/K` whenever any draw has weight, so `f ≤ 1e-6` alone would only
        // count failed subjects; ESS < 2 also counts a one-draw subject (#2690).
        final_n_collapsed = ess_fracs
            .iter()
            .filter(|&&f| is_degenerate(f, final_k_samples))
            .count();
        final_stuck_ids = subject_ids
            .iter()
            .zip(&stuck)
            .filter(|(_, &s)| s)
            .map(|(&id, _)| id)
            .collect();
        ess_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        final_ess_median = ess_fracs.get(ess_fracs.len() / 2).copied().unwrap_or(1.0);

        // Record this iteration's parameters for the trace (opt-in).
        if collect_trace {
            trace_rows.push(ImpmapTraceRow {
                iteration: k as i64,
                theta: theta_cur.clone(),
                omega_lower_tri: lower_triangle(&omega_mat),
                sigma: sigma_cur.clone(),
                ofv: minus2ll,
            });
        }

        // ---- M-step Ω: weighted second moment, structurally masked + floored ----
        // The η block of each subject's second moment: the whole matrix for an
        // η-only subject, its leading `n_eta × n_eta` block for an IOV subject.
        let mut new_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
        for d in &draws {
            new_omega += d.second_moment.view((0, 0), (n_eta, n_eta));
        }
        new_omega /= n_subjects as f64;
        for i in 0..n_eta {
            for j in 0..n_eta {
                if !init_params.omega.free_mask[(i, j)] {
                    new_omega[(i, j)] = 0.0;
                }
                let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    new_omega[(i, j)] = init_params.omega.matrix[(i, j)];
                }
            }
        }
        floor_omega_diagonal(&mut new_omega, &init_params.omega_fixed, OMEGA_DIAG_FLOOR);
        omega_mat = new_omega;

        // ---- M-step Ω_iov (#2301): weighted κ second moment over every occasion ----
        if let (Some(cur), Some(template)) = (omega_iov_mat.as_mut(), omega_iov_init) {
            if let Some(m) =
                omega_iov_mstep(&draws, &n_occ_of, n_eta, template, &init_params.kappa_fixed)
            {
                *cur = m;
            }
        }

        // ---- M-step σ + non-mu-ref θ: maximize weighted observation likelihood ----
        // Pin the log-mu-ref θ (handled by the closed-form shift below) so NLopt
        // optimizes only σ and any non-mu-ref θ, using the θ_old-centered samples.
        let mut mstep_theta_lower = log_theta_lower.clone();
        let mut mstep_theta_upper = log_theta_upper.clone();
        if use_closed_form {
            for &(t, _e) in &mu_ref_pairs {
                // Weak-IIV mu-ref θ are estimated by this M-step, not the shift,
                // so leave their bounds free here.
                if weak_mu_ref.contains(&t) {
                    continue;
                }
                mstep_theta_lower[t] = log_theta[t];
                mstep_theta_upper[t] = log_theta[t];
            }
        }
        // Covariate mu-reference thetas take the group step below (#619) — but
        // only pin the ones whose group step can actually run from here. A
        // group whose starting point makes some subject's mu non-finite returns
        // `None` from both engines, and the pin is set *before* the weighted
        // M-step, so pinning it anyway freezes those thetas for the whole
        // iteration with nothing left to move them (#918 review).
        let theta_now_premstep: Vec<f64> = (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    log_theta[i].exp()
                } else {
                    log_theta[i]
                }
            })
            .collect();
        let group_can_step: Vec<bool> = cov_mu_groups
            .iter()
            .map(|g| g.can_step(&theta_now_premstep, population, &init_params.theta_fixed))
            .collect();
        for (group, &can) in cov_mu_groups.iter().zip(group_can_step.iter()) {
            if !can {
                continue;
            }
            for &t in &group.theta_idx {
                if !init_params.theta_fixed.get(t).copied().unwrap_or(false) {
                    mstep_theta_lower[t] = log_theta[t];
                    mstep_theta_upper[t] = log_theta[t];
                }
            }
        }
        let mstep_maxiter: u32 = if k <= n_iter / 2 { 4 } else { 8 };
        let mstep_before = [log_theta.as_slice(), log_sigma.as_slice()].concat();
        let mstep = theta_sigma_weighted_mstep(
            model,
            population,
            &draws,
            &schedules,
            &log_theta,
            &log_sigma,
            &mstep_theta_lower,
            &mstep_theta_upper,
            &log_sigma_lower,
            &log_sigma_upper,
            n_theta,
            n_sigma,
            mstep_maxiter,
            &theta_packs_log_mask,
        );
        mstep_motion.record(
            &mstep_before,
            &[mstep_theta_lower.as_slice(), log_sigma_lower.as_slice()].concat(),
            &[mstep_theta_upper.as_slice(), log_sigma_upper.as_slice()].concat(),
            &mstep,
        );
        log_theta = mstep.log_theta;
        log_sigma = mstep.log_sigma;

        // ---- Closed-form mu-ref θ shift: packed θ (= mu) += population mean(η) ----
        if use_closed_form {
            let mut eta_bar = vec![0.0f64; n_eta];
            for d in &draws {
                for (acc, &m) in eta_bar.iter_mut().zip(d.mean.iter()) {
                    *acc += m;
                }
            }
            for acc in eta_bar.iter_mut() {
                *acc /= n_subjects as f64;
            }
            // The shift actually applied per η (after the bound clamp), for the
            // IMP proposal recentre below.
            let mut applied = vec![0.0f64; n_eta];
            for &(t, e) in &mu_ref_pairs {
                // Weak-IIV mu-ref θ were already updated by the weighted M-step;
                // their η-mean shift is ≈ 0 and uninformative — skip it.
                if weak_mu_ref.contains(&t) {
                    continue;
                }
                let before = log_theta[t];
                log_theta[t] =
                    (log_theta[t] + eta_bar[e]).clamp(log_theta_lower[t], log_theta_upper[t]);
                applied[e] = log_theta[t] - before;
            }
            // Keep φ fixed across the shift (#1365); see `shift_draws`. The
            // next IMP proposal is centred on these moments, and the #619
            // numerical group step below scores these samples at the shifted θ
            // — under IMPMAP too, which is why this is not gated on `recenter`.
            // Ω above was taken from the pre-shift draws, as before.
            if applied.iter().any(|&a| a != 0.0) {
                for d in draws.iter_mut() {
                    shift_draws(d, &applied);
                }
            }
        }

        // ---- Covariate mu-reference groups (#619): φ-frozen M-step ----
        // The importance-weighted prior term is quadratic in φ, so each
        // subject's posterior mean η is a sufficient input for the exact
        // engine; the numerical engine (time-varying covariates) re-weights
        // the data term over the draws with every sample's eta shifted by the
        // amount that keeps its φ fixed. No SA blend here — MCEM applies the
        // full M-step, as the shift above does.
        if !cov_mu_groups.is_empty() {
            let unpack_all = |lt: &[f64]| -> Vec<f64> {
                (0..n_theta)
                    .map(|i| {
                        if theta_packs_log_mask[i] {
                            lt[i].exp()
                        } else {
                            lt[i]
                        }
                    })
                    .collect()
            };
            let pack_one = |i: usize, v: f64| -> f64 {
                if theta_packs_log_mask[i] {
                    v.max(1e-10).ln()
                } else {
                    v
                }
            };
            let sigma_now: Vec<f64> = log_sigma.iter().map(|s| s.exp()).collect();
            for (gi, group) in cov_mu_groups.iter().enumerate() {
                // Not pinned above, so the weighted M-step already owned these
                // thetas this iteration; running the group step on top would be
                // a second M-step from a point it declared inadmissible.
                if !group_can_step[gi] {
                    cov_group_skipped[gi] += 1;
                    continue;
                }
                let theta_now = unpack_all(&log_theta);
                // Read per group, not once before the loop: an earlier group's
                // step re-centred the draws below, and the exact engine folds
                // every other eta's mean into its target through Ω's cross terms.
                // The η part only: an IOV subject's mean is stacked [η, κ₁, …] (#2301).
                let eta_means: Vec<Vec<f64>> =
                    draws.iter().map(|d| d.mean[..n_eta].to_vec()).collect();
                let input = GroupStepInput {
                    theta: &theta_now,
                    theta_lower: &init_params.theta_lower,
                    theta_upper: &init_params.theta_upper,
                    theta_fixed: &init_params.theta_fixed,
                    theta_packs_log: &theta_packs_log_mask,
                    omega: &omega_mat,
                    etas: &eta_means,
                };
                let solved = if group.needs_data_term {
                    let data = |th: &[f64], shifts: &[Vec<f64>]| -> f64 {
                        let per_subj: Vec<f64> = population
                            .subjects
                            .par_iter()
                            .zip(draws.par_iter())
                            .enumerate()
                            .map_init(EventPkParams::default, |scratch, (i, (subject, d))| {
                                let mut s = 0.0f64;
                                for (w, eta) in d.weights.iter().zip(d.etas.iter()) {
                                    if *w == 0.0 {
                                        continue;
                                    }
                                    let mut e2 = eta.clone();
                                    group.shift_eta(&mut e2, shifts, i);
                                    s += w * draw_obs_nll(
                                        model,
                                        subject,
                                        th,
                                        &sigma_now,
                                        &e2,
                                        schedules[i].as_ref(),
                                        scratch,
                                    );
                                }
                                s
                            })
                            .collect();
                        per_subj.iter().sum()
                    };
                    group.solve_numerical(population, &input, mstep_maxiter, &data)
                } else {
                    group.solve_exact(population, &input)
                };
                // `can_step` cleared this group before the pin, so a `None`
                // here means the solve became inadmissible partway — the
                // thetas are pinned and nothing else will move them this
                // iteration, which is worth the same report as a skipped step.
                let Some(theta_star) = solved else {
                    cov_group_skipped[gi] += 1;
                    continue;
                };
                for &t in &group.theta_idx {
                    if init_params.theta_fixed.get(t).copied().unwrap_or(false) {
                        continue;
                    }
                    log_theta[t] =
                        pack_one(t, theta_star[t]).clamp(log_theta_lower[t], log_theta_upper[t]);
                }
                // Keep φ_i fixed across the step, in the samples as well as the
                // moments, for IMP and IMPMAP alike: a later group in this loop
                // scores these draws (`d.etas` on the numerical engine, `d.mean`
                // on the exact one) at a θ that already carries this group's move
                // (#1365 review). IMP (`SampleMoments`) also centres the
                // next iteration's proposal on this iteration's weighted posterior
                // mean of η; the group step just moved each subject's mu by
                // Δ_i = g(A_i(θ_new)) − g(A_i(θ_old)), so the posterior of η has
                // moved by −Δ_i while that stored mean has not. Left stale, the
                // proposal sits several posterior SDs off, the ESS collapses and
                // the next weighted mean is biased toward the stale centre — the
                // step then re-fires on the bias and runs away (measured: the
                // renal slope climbed 0.041 → 0.10 in five iterations, then
                // collapsed to its bound). Shifting the mean by −Δ_i is the exact
                // φ-space bookkeeping; the second moment follows so the proposal
                // covariance `S − m mᵀ` is unchanged. This is the per-subject twin
                // of SAEM's `e[eta_idx] -= delta` re-centring, and like it covers
                // every member's eta of a joint group (#1620). Each subject moves
                // by one vector, so the samples, mean and second moment translate
                // together through `shift_draws`.
                let theta_new = unpack_all(&log_theta);
                let deltas = group.recentre_deltas(&theta_now, &theta_new, population);
                for (i, d) in draws.iter_mut().enumerate() {
                    // `recentre_eta` is the one place the per-member delta is
                    // read (a non-finite entry leaves that eta alone); applied to
                    // zero it yields −Δ_i, the shift `shift_draws` subtracts.
                    let mut neg_shift = vec![0.0f64; n_eta];
                    if !group.recentre_eta(&mut neg_shift, &deltas, i) {
                        continue;
                    }
                    let shift: Vec<f64> = neg_shift.iter().map(|v| -v).collect();
                    shift_draws(d, &shift);
                }
            }
        }

        theta_cur = (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    log_theta[i].exp()
                } else {
                    log_theta[i]
                }
            })
            .collect();
        sigma_cur = log_sigma.iter().map(|&s| s.exp()).collect();

        // Warm-start next iteration's inner loop from this iteration's modes —
        // only when we actually ran the inner loop this iteration (IMP skips it
        // on iterations 2+, leaving the iter-1 modes in place for the final EBE
        // pass).
        if run_inner {
            // Stacked `[η̂, κ̂]` for an IOV subject, so κ is warm-started too.
            let warm: Vec<DVector<f64>> = eta_hats
                .iter()
                .zip(kappa_hats.iter())
                .map(|(e, kap)| stack_eta_kappa(e, kap))
                .collect();
            prev_etas = Some(warm.clone());
            last_eta_hats = warm;
        }

        // IMP recenters the next iteration's proposal from these draws; IMPMAP
        // never reads them, so retain only for `SampleMoments` to avoid holding
        // K·n_subjects samples for the MAP path.
        if recenter == ProposalRecenter::SampleMoments {
            prev_draws = Some(draws);
        }

        // ---- Parameter averaging over the final n_avg iterations ----
        if k > n_iter - n_avg {
            for i in 0..n_theta {
                acc_theta[i] += theta_cur[i];
            }
            for i in 0..n_sigma {
                acc_sigma[i] += sigma_cur[i];
            }
            acc_omega += &omega_mat;
            if let Some(m) = omega_iov_mat.as_ref() {
                acc_omega_iov += m;
            }
            n_acc += 1;
        }

        if verbose && (k <= 5 || k % 10 == 0 || k == n_iter) {
            eprintln!(
                "  iter {:4}: -2logL(IS) = {:.4}  (low-ESS subjects: {})",
                k, minus2ll, n_low_ess
            );
        }

        iterations_done = k;
        if k < n_iter {
            stopped_after = time_limit_reached(started, max_time);
            if stopped_after.is_some() {
                break;
            }
        }
    }

    if crate::cancel::is_cancelled(cancel) {
        return Err("cancelled by user".to_string());
    }
    if let Some(elapsed) = stopped_after {
        warnings.push(time_limit_warning(
            label,
            max_time_key,
            elapsed,
            iterations_done,
            n_iter,
            n_acc,
        ));
    }

    // ---- Importance-sampling bias warning ----
    // Self-normalized IS moment estimates carry a finite-sample bias that drives
    // the M-step. For a fixed proposal quality it scales as ≈ 1/K, but the
    // constant grows with the sampling dimension, so a sample count that is ample
    // in low dimensions becomes badly under-sampled for a high-dimensional (e.g.
    // FREM) model. Empirically the absorption typical value on the FREM workshop
    // model (13 ETAs) drifts from ~2.6 (FOCEI/NONMEM) to ~4.6 at K=300 and
    // recovers toward ~2.8 at K=6000 — while the *median* ESS stays ~0.8 the
    // whole time, so ESS alone does not flag it. Use the per-dimension sample
    // density K / (sampled dimension) as the trigger (≥ ~100 keeps the bias small here), plus
    // any fully-collapsed subjects. Skip for eval-only IS (no M-step to bias).
    // Triggers, using the *effective* (possibly auto-ramped) sample count:
    //  - too few samples for the dimension (a fast proxy that catches the common
    //    case before the MC SE is even trustworthy), or
    //  - the objective is still MC-noisy at the end (the direct measure — also
    //    fires when `auto` ramped to the cap and could not reach the target), or
    //  - a subject's proposal fully collapsed.
    // The dimension heuristic only applies when `auto` is off — with `auto` on,
    // a low starting count is fine because it ramps when (and only when) the
    // objective is actually noisy, so the direct MC-SE check is the only trigger.
    // The dimension is the widest sampled vector: an IOV subject samples its
    // stacked `[η, κ₁..κ_K]`, and the Ω_iov M-step is biased by under-sampling
    // just as the Ω one is (#2301).
    let max_occ = n_occ_of.iter().copied().max().unwrap_or(0);
    let sampled_dim = crate::stats::stacked_layout::stacked_width(n_eta, n_kappa, max_occ);
    let under_sampled = !auto && n_eta > 0 && final_k_samples < MIN_SAMPLES_PER_DIM * sampled_dim;
    let noisy = final_mc_se > 2.0 * AUTO_STDOBJ_TARGET;
    if !options.imp_eval_only && (under_sampled || noisy || final_n_collapsed > 0) {
        let (sample_opt, auto_opt) = if recenter == ProposalRecenter::Map {
            ("impmap_samples", "impmap_auto")
        } else {
            ("imp_samples", "imp_auto")
        };
        // Under `auto` the left-out subjects get their own sentence, which says
        // more samples will not help them; the collapse sentence counts the rest,
        // which a larger count may still recover (#2690 review).
        let named = if auto { final_stuck_ids.len() } else { 0 };
        let collapse = collapse_sentence(final_n_collapsed.saturating_sub(named));
        let stuck = stuck_subjects_sentence(auto, auto_opt, &final_stuck_ids);
        // The noise is all theirs when the subjects `auto` acts on met the target.
        let only_left_out_noisy = named > 0
            && iteration_se
                .last()
                .is_some_and(|&(se, _)| se <= AUTO_STDOBJ_TARGET);
        let advice = under_sampling_advice(
            auto,
            final_k_samples,
            &iteration_se,
            k_start,
            sample_opt,
            auto_opt,
            only_left_out_noisy,
        );
        warnings.push(format!(
            "{label}: {} importance samples for {} give a noisy objective \
             (per-subject MC SE = {:.3}, target {:.2}; median ESS/K = {:.2}). The weighted M-step \
             moments then carry a finite-sample bias — typical-value and Ω estimates may be off, \
             and it shrinks only as the sample count grows. {}{}{}",
            final_k_samples,
            sampled_dimension_phrase(n_eta, n_kappa, max_occ),
            final_mc_se,
            AUTO_STDOBJ_TARGET,
            final_ess_median,
            advice,
            collapse,
            stuck
        ));
    }

    // A #619 group that could not step on some iterations left its thetas to
    // the weighted M-step on exactly those iterations, and to neither channel
    // when the step failed after the pin. Either way the estimate is not the
    // one the group was meant to produce, so say so rather than return a
    // plausible-looking number (#918 review).
    for (gi, &skipped) in cov_group_skipped.iter().enumerate() {
        if skipped == 0 {
            continue;
        }
        let group = &cov_mu_groups[gi];
        let names: Vec<&str> = group
            .theta_idx
            .iter()
            .filter_map(|&t| model.theta_names.get(t).map(String::as_str))
            .collect();
        let (what, typical_was) = group.skipped_step_phrases();
        warnings.push(format!(
            "{label}: the {what} had no admissible step on {} of {} iteration(s) — \
             {typical_was} not finite for at least one subject at the current θ (an additive \
             typical value can go ≤ 0 for a low-covariate subject). {} were then moved by the \
             importance-weighted M-step alone, or not at all; check them against a FOCEI fit \
             and consider bounding the covariate slope (#619).",
            skipped,
            iterations_done,
            names.join(", ")
        ));
    }

    // ---- Final (averaged) parameters ----
    let (final_theta, final_sigma, final_omega_mat) = if n_acc > 0 {
        let t: Vec<f64> = acc_theta.iter().map(|&v| v / n_acc as f64).collect();
        let s: Vec<f64> = acc_sigma.iter().map(|&v| v / n_acc as f64).collect();
        let o = acc_omega / n_acc as f64;
        (t, s, o)
    } else {
        (theta_cur.clone(), sigma_cur.clone(), omega_mat.clone())
    };
    let final_omega_iov_mat: Option<DMatrix<f64>> = if n_acc > 0 {
        omega_iov_mat
            .as_ref()
            .map(|_| &acc_omega_iov / n_acc as f64)
    } else {
        omega_iov_mat.clone()
    };

    let final_omega = OmegaMatrix::from_matrix(
        final_omega_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    let final_params = ModelParameters {
        theta: final_theta,
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: final_sigma,
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        // IMP/IMPMAP does not update the `block_sigma` off-diagonals
        // (#847); carry them so the sampled residual `R` keeps its
        // declared correlation structure.
        residual_correlations: init_params.residual_correlations.clone(),
        residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
        omega_iov: omega_iov_snapshot(init_params, final_omega_iov_mat.as_ref()),
        kappa_fixed: init_params.kappa_fixed.clone(),
        mixture: None,
    };

    // ---- Final EBEs (warm-started) + FOCE Laplace OFV for comparability ----
    let warm = if last_eta_hats.is_empty() {
        None
    } else {
        Some(last_eta_hats.as_slice())
    };
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, final_kappas, _stats) = run_map_multistart(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        warm,
        &final_mu_k,
        mceta,
        seed,
        n_iter + 1, // distinct iteration index for final EBEs
    );
    check_kappa_blocks(population, &final_kappas, &n_occ_of, label)?;

    let ofv = 2.0
        * pop_nll(
            model,
            population,
            &final_params,
            &eta_hats,
            &h_matrices,
            &final_kappas,
            options.interaction,
        );

    // ---- Covariance step ----
    let packed = pack_params(&final_params);
    let out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
        method: covariance_method,
        newton_decrement,
        non_stationary_optimum: _,
    } = out;
    warnings.extend(cov_warnings);

    // ---- Finalize trace ----
    let impmap_trace = if collect_trace {
        // Append final (averaged) estimate row.
        trace_rows.push(ImpmapTraceRow {
            iteration: -1_000_000_000,
            theta: final_params.theta.clone(),
            omega_lower_tri: lower_triangle(&final_params.omega.matrix),
            sigma: final_params.sigma.values.clone(),
            ofv,
        });
        // Append SE row when the covariance step succeeded.
        if let Some(ref cov) = covariance_matrix {
            let se: Vec<f64> = (0..cov.nrows()).map(|i| cov[(i, i)].sqrt()).collect();
            // Unpack SEs into theta / omega-LT / sigma segments, mirroring
            // pack_params layout: [theta..., cholesky-omega..., sigma...].
            let n_free_theta = final_params.theta.len();
            let n_omega_lt = lower_triangle(&final_params.omega.matrix).len();
            let n_free_sigma = final_params.sigma.values.len();
            let se_theta: Vec<f64> = se.iter().take(n_free_theta).copied().collect();
            let se_omega: Vec<f64> = se
                .iter()
                .skip(n_free_theta)
                .take(n_omega_lt)
                .copied()
                .collect();
            let se_sigma: Vec<f64> = se
                .iter()
                .skip(n_free_theta + n_omega_lt)
                .take(n_free_sigma)
                .copied()
                .collect();
            trace_rows.push(ImpmapTraceRow {
                iteration: -1_000_000_001,
                theta: se_theta,
                omega_lower_tri: se_omega,
                sigma: se_sigma,
                ofv: 0.0,
            });
        }

        // Build column names following NONMEM convention.
        let theta_names: Vec<String> = (1..=n_theta).map(|i| format!("THETA{i}")).collect();
        let omega_names: Vec<String> = {
            let mut names = Vec::new();
            for i in 0..n_eta {
                for j in 0..=i {
                    names.push(format!("OMEGA({},{})", i + 1, j + 1));
                }
            }
            names
        };
        let sigma_names: Vec<String> = (1..=n_sigma).map(|i| format!("SIGMA({i},{i})")).collect();

        Some(ImpmapTrace {
            rows: trace_rows,
            theta_names,
            omega_names,
            sigma_names,
        })
    } else {
        None
    };

    // IMP and IMPMAP run a fixed iteration schedule, so there is no stopping test
    // to report. The verdict here rests on a sane final objective, and `fit()` adds
    // the checks that need the marginal (#2652, #2686). A non-finite OFV means the MCEM
    // diverged; but a *runaway* (importance weights collapse → weighted
    // M-step walks θ to the bounds) produces a finite-but-enormous OFV
    // (~1e35), which `is_finite()` alone would wave through as converged and
    // could then win multi-start selection (issue #528).
    //
    // #1303: both rules now live in `gate_converged_on_mcem_objective`, which also
    // *says* which one fired — the verdict used to be a bare boolean with nothing
    // in `warnings` to explain it.
    let mut converged = stopped_after.is_none();
    if let Some(w) = gate_converged_on_mcem_objective(&mut converged, ofv) {
        warnings.push(w);
    }
    warnings.extend(mstep_warnings(
        &mut converged,
        label,
        &mstep_motion,
        model,
        stopped_after.is_some(),
    ));

    if verbose {
        eprintln!("{} completed. Final OFV (Laplace) = {:.4}", label, ofv);
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged,
        n_iterations: iterations_done,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        total_ebe_nm_polished: 0,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal,
        impmap_trace,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        left_init: None,
        stationary_start: false,
        mixture_posteriors: None,
        vi: None,
        start_eval: None,
        mcem_lowest_objective: lowest_iteration_objective,
        // A run `max_time` stopped did not reach the limit the settling verdict is about,
        // and already says why it is not converged.
        mcem_objective_trail: stopped_after
            .is_none()
            .then(|| recenter.objective_trail(iteration_objectives, iteration_mc_se, label)),
        newton_decrement,
        optimizer_stop: None,
        eval_budget_spent: false,
    })
}

/// Estimating IMP / IMPMAP under a mixture model (#985): a class-partitioned
/// MCEM. Each iteration importance-samples each subject's η *within every class*
/// (under a [`MixtureClassGuard`], so `MIXNUM` resolves to that class), forms the
/// deterministic class responsibilities `PMIX_ik ∝ p_ik · L_ik`, then runs the
/// responsibility-weighted M-steps:
///
/// * mixing coefficients — from the responsibilities (reusing SAEM's mixing
///   M-step with `r̄ = PMIX`);
/// * Ω (class-shared) — `Σ_i Σ_k PMIX_ik · secondmoment_ik / N`;
/// * θ / σ — the responsibility- and importance-weighted observation M-step
///   ([`theta_sigma_weighted_mstep_mixture`]), which recovers the class-switched
///   typical values from each class's own weighted samples.
///
/// Ω/σ are class-shared (per-class overrides are rejected); IOV, FREM and SDE are
/// not supported here (as for the single-population MCEM). The final OFV,
/// per-subject `MIXEST`/`PMIX`, and EBEs come from the K-fold marginal
/// (`mixture_ofv`) at the estimates, matching FOCEI/SAEM.
#[allow(clippy::too_many_arguments)]
fn run_mcem_mixture(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
    recenter: ProposalRecenter,
    label: &str,
    df_key: &str,
    n_iter_opt: usize,
    k_opt: usize,
    nu: f64,
    n_avg_opt: usize,
    seed: u64,
    threshold: f64,
    mceta: usize,
    use_sobol: bool,
    collect_trace: bool,
    auto: bool,
    max_time: Option<f64>,
) -> Result<OuterResult, String> {
    use crate::estimation::importance_sampling::{
        compute_posterior_hessian, subject_is_draws, DefensiveMixture,
    };
    use crate::estimation::mixture::combine_subject;
    use crate::parser::model_parser::{eval_mixing_log_probs, MixtureClassGuard};

    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();
    let spec = model
        .mixture
        .as_ref()
        .expect("run_mcem_mixture on non-mixture model");
    let mp0 = init_params
        .mixture
        .as_ref()
        .ok_or_else(|| format!("{label} mixture requires params.mixture (per-class Ω/Σ)"))?;

    // ---- Validation (mixture scope) ----
    if recenter.method().requires_random_effects() && n_eta == 0 {
        return Err(format!(
            "{label} requires at least one random effect (n_eta = 0)."
        ));
    }
    if model.n_kappa > 0 {
        return Err(format!(
            "{label} for a mixture does not yet support inter-occasion variability (IOV); use \
             SAEM or FOCE/FOCEI (#985)."
        ));
    }
    if model.frem_config.is_some() {
        return Err(format!(
            "{label} for a mixture does not support FREM models (#985)."
        ));
    }
    if model.is_sde() {
        return Err(format!(
            "{label} for a mixture does not support SDE / [diffusion] models."
        ));
    }
    if !mp0.omega_override_addr.is_empty() || !mp0.sigma_override_addr.is_empty() {
        return Err(format!(
            "{label} for a mixture does not yet support per-class Omega/Sigma overrides \
             (omega(k)/sigma(k)); Omega/Sigma are class-shared. Use FOCE/FOCEI for per-class \
             overrides (#985)."
        ));
    }
    if nu.is_finite() && nu < 1.0 {
        return Err(format!(
            "{label}: {df_key} must be >= 1.0 (or +inf), got {nu}"
        ));
    }

    let started = std::time::Instant::now();
    let n_iter = n_iter_opt.max(1);
    let k_samples = k_opt.max(2);
    let n_avg = n_avg_opt.min(n_iter);
    let verbose = options.verbose;
    let cancel = &options.cancel;
    let n_classes = spec.n_classes;

    // ---- Packing scaffolding (mirrors run_mcem / SAEM) ----
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };
    let mut log_theta: Vec<f64> = (0..n_theta)
        .map(|i| pack_theta(i, init_params.theta[i]))
        .collect();
    let mut log_sigma: Vec<f64> = init_params
        .sigma
        .values
        .iter()
        .map(|&s| s.max(1e-10).ln())
        .collect();
    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }
    let unpack_theta = |i: usize, v: f64| -> f64 {
        if theta_packs_log_mask[i] {
            v.exp()
        } else {
            v
        }
    };

    // Mixing state: detects the mixing thetas and drives the mixing M-step from
    // the deterministic responsibilities (no stochastic averaging — `rbar` is set
    // to the current `PMIX` each iteration). The closed-form mu-ref θ shift is not
    // used for a mixture (the class-switched typical value can't pair to one η);
    // every θ flows through the responsibility-weighted M-step.
    let mut mix =
        crate::estimation::saem_mixture::SaemMixture::build(model, init_params, population);

    // Reject a theta that drives both the mixing expression and a structural
    // typical value. Like SAEM (`run_saem`), this MCEM runs *separated* M-steps:
    // `theta_sigma_weighted_mstep_mixture` fits such a theta from the weighted
    // observation likelihood and `mstep_mixing` then overwrites it from the
    // responsibilities, silently discarding the structural estimate. FOCEI's
    // joint marginal handles the shared parameter, so route there (#992 review).
    let overlap = crate::estimation::saem_mixture::mixing_structural_overlap(
        model,
        init_params,
        population,
        &mix.mixing_theta_idx,
    );
    if !overlap.is_empty() {
        let names: Vec<String> = overlap
            .iter()
            .map(|&j| {
                init_params
                    .theta_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("theta[{j}]"))
            })
            .collect();
        return Err(format!(
            "{label} cannot fit a mixture where a mixing-coefficient theta also drives the \
             structural model: {} appear(s) in both the [mixture] mixing expression and an \
             [individual_parameters] typical value. The weighted θ/σ M-step and the mixing \
             M-step run separately, so a shared parameter would be double-owned. Split it into \
             two thetas (one for structure, one for mixing), or fit with FOCE/FOCEI (whose \
             joint marginal handles the shared parameter) (#985).",
            names.join(", ")
        ));
    }

    let mut warnings: Vec<String> = Vec::new();

    // The mixture MCEM never uses the closed-form log-mu-ref θ shift (a
    // class-switched typical value cannot be paired to a single η), so *every* θ
    // is estimated through the importance-weighted M-step alone — the channel
    // that is biased for weakly-identified parameters. The single-population
    // `run_mcem` pushes the equivalent warnings before this delegation point;
    // repeat them here so a mixture fit is not silently unwarned (#992 review).
    let mixing_names: std::collections::HashSet<&str> = mix
        .mixing_theta_idx
        .iter()
        .filter_map(|&j| model.theta_names.get(j).map(String::as_str))
        .collect();
    // ---- Class-aware closed-form mu-ref θ shift (#996) ----
    //
    // MCEM over a mixture importance-samples η *within every class*, so a
    // per-class posterior mean η̄_ic exists directly and the EM update for the
    // class-switched typical value is the responsibility-weighted
    //     log θ_k += (Σ_i r_ic η̄_ic) / (Σ_i r_ic),   r_ic = PMIX_ic
    // (`mixture_mu_ref_means`). A class-*shared* typical value repeats the same
    // theta in every class slot, so it collapses to the pooled `mean_i(η̄_i)`
    // shift the single-population `run_mcem` applies. Those θ are pinned out of
    // the weighted M-step below, exactly as in `run_mcem`.
    //
    // Disabled by `mu_referencing = false`, which restores the pre-#996 behaviour
    // (every θ through the weighted M-step alone). Note this differs from the
    // single-population IMP path, where the closed-form shift is EM-mandatory and
    // applied regardless of that option.
    const WEAK_IIV_VAR: f64 = 1e-3;
    // Every θ the parser identified as a class-aware anchor, *before* the
    // filters below decide which shifts actually run. Used to tell an
    // anchor-with-a-disabled-shift apart from a genuinely ETA-less θ in the #406
    // advisory (#996 review).
    let all_class_anchors: Vec<usize> = get_mixture_mu_ref_pairs(model)
        .iter()
        .flat_map(|p| p.theta_idx.iter().copied())
        .collect();
    // A θ whose packed scale is not its mu scale — an identity-packed lognormal
    // anchor (`theta_lower < 0`), or a log-packed logit anchor (#918) — cannot
    // take the additive shift, so `classify_mixture_mu_ref_pairs` routes it to
    // the weighted M-step. The two dropped lists get different advisories
    // because they name opposite bounds to change.
    let MixtureMuRefPairs {
        eligible: mut mix_mu_ref_pairs,
        identity_packed_log: identity_packed,
        log_packed_logit: log_packed_logit_mix,
    } = if options.mu_referencing {
        classify_mixture_mu_ref_pairs(model, &theta_packs_log_mask)
    } else {
        MixtureMuRefPairs {
            eligible: Vec::new(),
            identity_packed_log: Vec::new(),
            log_packed_logit: Vec::new(),
        }
    };
    // A η with negligible IIV carries no mean-shift information, so its typical
    // values would be frozen at their inits (#411) — same guard as `run_mcem`,
    // applied per pair (all of that η's class θ share the one ω).
    let weak: Vec<usize> = mix_mu_ref_pairs
        .iter()
        .filter(|p| init_params.omega.matrix[(p.eta_idx, p.eta_idx)] < WEAK_IIV_VAR)
        .flat_map(|p| p.theta_idx.iter().copied())
        .collect();
    mix_mu_ref_pairs.retain(|p| init_params.omega.matrix[(p.eta_idx, p.eta_idx)] >= WEAK_IIV_VAR);
    let name_list = |idx: Vec<usize>| -> String {
        let mut n: Vec<&str> = idx
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        n.sort_unstable();
        n.dedup();
        n.join(", ")
    };
    if !identity_packed.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but declared with a negative \
             lower bound, so they are packed on the identity scale; the closed-form mu-ref \
             shift does not apply and they are estimated by the importance-weighted M-step \
             instead (#996).",
            name_list(identity_packed)
        ));
    }
    if !log_packed_logit_mix.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are logit-mu-referenced but declared with a \
             non-negative lower bound, so they are packed on the log scale; the closed-form \
             mu-ref shift does not apply and they are estimated by the importance-weighted \
             M-step instead. Declare the logit-scale theta with a negative lower bound to use \
             the closed-form update (#918).",
            name_list(log_packed_logit_mix)
        ));
    }
    if !weak.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but their random effect has \
             negligible variance (ω < {WEAK_IIV_VAR:.0e}); the mu-ref mean-shift carries no \
             information, so they are estimated through the weighted M-step instead.",
            name_list(weak)
        ));
    }
    // Thetas the closed-form shift owns — pinned out of the weighted M-step each
    // iteration. A FIXed θ is left to the regular fixed-bounds path.
    let mu_ref_pinned: Vec<usize> = {
        let mut v: Vec<usize> = mix_mu_ref_pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .filter(|&t| !init_params.theta_fixed.get(t).copied().unwrap_or(false))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    // Mixing thetas are excluded: they are estimated by `mstep_mixing` from the
    // class responsibilities, not by the weighted M-step, so an ETA on them is
    // neither expected nor useful. The class θ that the *active* shift owns
    // (`mu_ref_pinned`, after the mu_referencing / identity-pack / weak-IIV
    // filters above) do carry an ETA and are excluded too; the ones those filters
    // dropped are genuinely M-step-only and stay flagged (#996 review).
    //
    // The flagged set is split in two: a class-aware anchor whose shift the
    // filters above switched off does carry an ETA in the model text — telling
    // the user to add one would be wrong — but it is still estimated by the
    // weighted M-step alone, so it earns its own message.
    let anchor_names: std::collections::HashSet<&str> = all_class_anchors
        .iter()
        .filter_map(|&t| model.theta_names.get(t).map(String::as_str))
        .collect();
    let (shift_disabled, unanchored): (Vec<String>, Vec<String>) =
        // `&[]` for the covariate groups: `run_mcem_mixture` builds none — the
        // class draw would have to enter the group's prior term — so no theta
        // is anchored by one here.
        non_fixed_unanchored_thetas(model, &init_params.theta_fixed, &[], &mu_ref_pinned)
            .into_iter()
            .filter(|n| !mixing_names.contains(n.as_str()))
            .partition(|n| anchor_names.contains(n.as_str()));
    let (thetas_with_eta, thetas_without_eta) = partition_by_eta(model, unanchored);
    warnings.extend(imp_unanchored_theta_warnings(
        label,
        &thetas_with_eta,
        &thetas_without_eta,
    ));
    if !shift_disabled.is_empty() {
        warnings.push(format!(
            "{label}: class typical value(s) [{}] carry an ETA, but their class-aware mu-ref \
             shift is disabled (see the preceding warning, or `mu_referencing = false`), so they \
             are estimated by the importance-weighted M-step alone — the channel that is biased \
             for weakly-identified parameters (#406). Re-enable the shift, or cross-check \
             against a FOCEI fit.",
            shift_disabled.join(", ")
        ));
    }
    if mix_mu_ref_pairs.is_empty() {
        warnings.push(format!(
            "{label}: no class-aware mu-referencing was applied, so every typical value is \
             estimated through the importance-weighted M-step alone and may converge poorly. \
             Write each class-switched typical value as a MIXNUM chain of \
             `THETA_k * exp(ETA)` terms on one ETA, or cross-check against a FOCEI fit (#996)."
        ));
    }
    if collect_trace {
        warnings.push(format!(
            "{label}: iteration trace collection (impmap_trace) is not implemented for mixture \
             models; no trace is reported (#985)."
        ));
    }
    if auto {
        warnings.push(format!(
            "{label}: adaptive sampling (auto) is not implemented for mixture models; the sample \
             count stays at K={k_samples} for every iteration (#985)."
        ));
    }

    let schedules = build_schedule_cache(model, population);
    let mut theta_cur = init_params.theta.clone();
    let mut sigma_cur = init_params.sigma.values.clone();
    let mut omega_mat = init_params.omega.matrix.clone();

    // Previous iteration's per-class weighted draws (for the IMP `SampleMoments`
    // recenter, which rebuilds the proposal from their second moments). `None` on
    // iteration 1 and — as in `run_mcem` — never retained on the IMPMAP (`Map`)
    // path, which re-runs the MAP solve and only needs the previous weighted
    // means as warm starts. Holding the full `K · n_subjects · n_classes` sample
    // sets there would cost `K×` the memory for nothing (#992 review).
    let mut prev_draws: Option<Vec<Vec<SubjectDraws>>> = None;
    // Previous iteration's per-class weighted posterior means, `[class][subject]`
    // — the warm start for the next iteration's inner solve on both paths.
    let mut prev_means: Option<Vec<Vec<Vec<f64>>>> = None;
    let warm0: Option<Vec<DVector<f64>>> = warm_etas.map(|e| e.to_vec());

    // Running average of the estimates over the final `n_avg` iterations.
    let mut acc_theta = vec![0.0f64; n_theta];
    let mut acc_sigma = vec![0.0f64; n_sigma];
    let mut acc_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut n_acc = 0usize;
    // Class-aware mu-ref θ that were unpinned on at least one iteration because
    // their classes drew no responsibility mass (#996 review). A collapsed class
    // is a modelling problem, so report it once at the end of the fit.
    let mut empty_class_theta: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();

    if verbose {
        eprintln!(
            "{label} (mixture): {n_subjects} subjects × {n_classes} classes, {n_eta} ETAs, \
             {n_iter} iters, K={k_samples}/subject/class, seed={seed}"
        );
    }

    // The lowest −2 log L any iteration reached, for `fit()`'s rise check (#2652).
    let mut lowest_iteration_objective: Option<f64> = None;
    let mut iteration_objectives: Vec<f64> = Vec::with_capacity(n_iter);
    let mut iteration_mc_se: Vec<f64> = Vec::with_capacity(n_iter);
    // Iterations actually run, and the seconds at which `max_time` stopped them.
    let mut iterations_done = 0usize;
    let mut stopped_after: Option<f64> = None;
    let mut mstep_motion = MstepMotion::new(n_theta + n_sigma);
    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(cancel) {
            break;
        }
        // Shared Ω/σ view for this iteration (identical across classes; the class
        // enters only through the `MIXNUM` guard on the structural model).
        let omega_k = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        let omega_inv = omega_k.inv.clone();
        let log_det_omega = omega_k.log_det;
        let defensive = DefensiveMixture::new(&omega_inv, n_eta, options.imp_defensive_alpha);
        let params_iter = ModelParameters {
            theta: theta_cur.clone(),
            theta_names: init_params.theta_names.clone(),
            theta_lower: init_params.theta_lower.clone(),
            theta_upper: init_params.theta_upper.clone(),
            theta_fixed: init_params.theta_fixed.clone(),
            omega: omega_k.clone(),
            omega_fixed: init_params.omega_fixed.clone(),
            sigma: SigmaVector {
                values: sigma_cur.clone(),
                names: init_params.sigma.names.clone(),
            },
            sigma_fixed: init_params.sigma_fixed.clone(),
            // IMP/IMPMAP does not update the `block_sigma` off-diagonals
            // (#847); carry them so the sampled residual `R` keeps its
            // declared correlation structure.
            residual_correlations: init_params.residual_correlations.clone(),
            residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
            omega_iov: None,
            kappa_fixed: init_params.kappa_fixed.clone(),
            mixture: None,
        };
        let run_inner = recenter == ProposalRecenter::Map || prev_draws.is_none();
        let prev = prev_draws.as_ref();
        let prev_mean = prev_means.as_ref();
        // Cholesky of the shared Ω for the MCETA random restarts (computed once
        // per iteration, outside the parallel loop). `None` when MCETA is off.
        let omega_chol = if mceta > 0 && run_inner {
            omega_mat.clone().cholesky().map(|c| c.l())
        } else {
            None
        };

        // ---- E-step: per subject, per class IS draws + class responsibilities ----
        // Parallel over subjects; the class loop runs inside each worker so the
        // `MIXNUM` guard is entered on the same thread as the solve/draws.
        #[allow(clippy::type_complexity)]
        let per_subject: Vec<(Vec<SubjectDraws>, Vec<f64>, f64)> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                // Poll per subject: one subject's E-step is `n_classes` inner
                // solves plus `n_classes · K` importance draws — the dominant
                // per-iteration cost — so without this a cancel set mid-sweep is
                // not observed until the whole subject × class loop finishes.
                // Mirrors `run_mcem`; the driver breaks right after the collect,
                // so the placeholder draws never reach an M-step.
                if crate::cancel::is_cancelled(cancel) {
                    let class_draws: Vec<SubjectDraws> = (0..n_classes)
                        .map(|_| SubjectDraws::cancelled(n_eta))
                        .collect();
                    return (class_draws, vec![1.0 / n_classes as f64; n_classes], 0.0);
                }
                let logp = eval_mixing_log_probs(spec, &theta_cur, &subject.covariates);
                let mut class_draws: Vec<SubjectDraws> = Vec::with_capacity(n_classes);
                let mut nll = vec![0.0f64; n_classes];
                for c in 0..n_classes {
                    let _g = MixtureClassGuard::enter(c + 1);
                    let (center, h_post) = if run_inner {
                        let warm_i = prev_mean
                            .map(|pm| DVector::from_row_slice(&pm[c][i]))
                            .or_else(|| warm0.as_ref().map(|w| w[i].clone()));
                        let ebe = find_ebe_multistart(
                            model,
                            subject,
                            &params_iter,
                            options.inner_maxiter,
                            options.inner_tol,
                            warm_i.as_ref().map(|v| v.as_slice()),
                            None,
                            options.inner_restarts,
                            mceta,
                            omega_chol.as_ref(),
                            seed.wrapping_add(i as u64)
                                .wrapping_add((k as u64) << 32)
                                .wrapping_add((c as u64) << 48)
                                .wrapping_add(0x4D43_4554_4100u64),
                        );
                        let h_post = compute_posterior_hessian(
                            model,
                            subject,
                            &params_iter.theta,
                            &ebe.eta,
                            &params_iter.sigma.values,
                            &ebe.h_matrix,
                            &omega_inv,
                            n_eta,
                            scratch,
                        );
                        (ebe.eta, h_post)
                    } else {
                        let pd = &prev.expect("prev set when !run_inner")[c][i];
                        let center = DVector::from_row_slice(&pd.mean);
                        let cov = &pd.second_moment - &center * center.transpose();
                        let h_post = covariance_to_proposal_hessian(
                            &cov,
                            &omega_mat,
                            IMP_PROPOSAL_COV_FLOOR,
                        );
                        (center, h_post)
                    };
                    let subj_seed = seed
                        .wrapping_add(i as u64)
                        .wrapping_add((k as u64) << 32)
                        .wrapping_add((c as u64) << 48);
                    // Per-subject, per-class ISCALE pilot search — the same guard
                    // the single-population E-step runs (#528/#961): it rescues
                    // subjects whose posterior Hessian is a poor curvature
                    // estimate, whose IS weights would otherwise collapse and
                    // drive the weighted M-step to the bounds. Skipping it is
                    // materially riskier under a mixture, where an off-class
                    // proposal is a poor fit by construction (#992 review).
                    let iscale = find_optimal_iscale(
                        model,
                        subject,
                        &params_iter.theta,
                        &params_iter.sigma.values,
                        &center,
                        &h_post,
                        &omega_inv,
                        log_det_omega,
                        n_eta,
                        nu,
                        subj_seed,
                        scratch,
                        options.iscale_min,
                        options.iscale_max,
                        schedules[i].as_ref(),
                    );
                    let d = subject_is_draws(
                        model,
                        subject,
                        &params_iter.theta,
                        &params_iter.sigma.values,
                        &center,
                        &h_post,
                        &omega_inv,
                        log_det_omega,
                        n_eta,
                        k_samples,
                        nu,
                        subj_seed,
                        scratch,
                        iscale,
                        use_sobol,
                        defensive.as_ref(),
                        schedules[i].as_ref(),
                    );
                    nll[c] = -d.log_marginal;
                    class_draws.push(d);
                }
                let (contribution, pmix_i, _mixest) = combine_subject(&logp, &nll);
                (class_draws, pmix_i, contribution)
            })
            .collect();

        if crate::cancel::is_cancelled(cancel) {
            break;
        }

        // Reorganize into [class][subject] draws + [subject][class] responsibilities.
        let mut draws_by_class: Vec<Vec<SubjectDraws>> = (0..n_classes)
            .map(|_| Vec::with_capacity(n_subjects))
            .collect();
        let mut pmix: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
        let mut ofv = 0.0f64;
        for (class_draws, pmix_i, contribution) in per_subject {
            for (c, d) in class_draws.into_iter().enumerate() {
                draws_by_class[c].push(d);
            }
            pmix.push(pmix_i);
            ofv += contribution;
        }
        lowest_iteration_objective = lowest_objective(lowest_iteration_objective, ofv);
        let var_ll: f64 = (0..n_subjects)
            .map(|i| {
                let ess = (0..n_classes).map(|c| draws_by_class[c][i].ess_fraction);
                mixture_log_marginal_mc_variance(&pmix[i], ess, k_samples)
            })
            .sum();
        iteration_objectives.push(ofv);
        iteration_mc_se.push(2.0 * var_ll.sqrt());
        if verbose {
            // Per-subject ESS: the worst per-class proposal among the classes the
            // subject actually contributes to (a collapsed proposal in a class
            // with PMIX ≈ 0 does not degrade that subject's marginal).
            let n_low_ess = (0..n_subjects)
                .filter(|&i| {
                    let worst = (0..n_classes)
                        .filter(|&c| pmix[i][c] > MIX_ESS_PMIX_FLOOR)
                        .map(|c| draws_by_class[c][i].ess_fraction)
                        .fold(f64::INFINITY, f64::min);
                    worst.is_finite() && worst < threshold
                })
                .count();
            eprintln!(
                "{label} (mixture) iter {k}: marginal −2 log L = {ofv:.4} \
                 (low-ESS subjects: {n_low_ess})"
            );
        }

        // ---- M-step ----
        // (a) θ / σ from the responsibility- and importance-weighted obs NLL.
        // The mixing thetas are pinned out of this M-step: they do not enter
        // `obs_nll_subject_into` at all, so leaving them free only inflates the
        // BOBYQA dimension with degenerate directions and can park them at an
        // arbitrary point that `mstep_mixing` (4–8 evals) then has to recover
        // from (#992 review). `mstep_mixing` below is their sole owner.
        let mut mstep_theta_lower = log_theta_lower.clone();
        let mut mstep_theta_upper = log_theta_upper.clone();
        for &j in &mix.mixing_theta_idx {
            mstep_theta_lower[j] = log_theta[j];
            mstep_theta_upper[j] = log_theta[j];
        }
        // Class-aware closed-form mu-ref θ shift (#996), computed *before* the
        // M-step because the pinning depends on which θ it can actually move this
        // iteration. `mixture_mu_ref_means` returns `None` for a θ whose classes
        // drew no responsibility mass at all; pinning such a θ anyway would
        // freeze it for the rest of the fit — neither M-stepped nor shifted, and
        // with no path back, since θ never moves so the responsibilities never
        // recover (#996 review). Leave those free and let the weighted M-step own
        // them this iteration instead.
        //
        // Both inputs (`pmix`, `draws_by_class`) come from the E-step above, so
        // these are the same means the shift used when it ran after the M-step.
        let mut mu_ref_shift: Vec<Option<f64>> = vec![None; n_theta];
        for pair in &mix_mu_ref_pairs {
            let eta_idx = pair.eta_idx;
            let means = mixture_mu_ref_means(n_theta, &pair.theta_idx, &pmix, |i, c| {
                draws_by_class[c][i].mean[eta_idx]
            });
            for (t, mean_eta) in means.iter().enumerate() {
                let Some(mean_eta) = mean_eta else { continue };
                if init_params.theta_fixed.get(t).copied().unwrap_or(false) {
                    continue;
                }
                // `get_mixture_mu_ref_pairs` guarantees each θ is claimed by at
                // most one pair, so this never overwrites another η's shift.
                mu_ref_shift[t] = Some(*mean_eta);
            }
        }
        // Pin the class-aware mu-ref θ the shift can move: it is their sole owner,
        // so NLopt optimizes only σ and the remaining θ (#996).
        let (pin_now, unpinned) = split_mu_ref_pins(&mu_ref_pinned, &mu_ref_shift);
        for t in pin_now {
            mstep_theta_lower[t] = log_theta[t];
            mstep_theta_upper[t] = log_theta[t];
        }
        empty_class_theta.extend(unpinned);
        let mstep_maxiter: u32 = if k <= n_iter / 2 { 4 } else { 8 };
        let mstep_before = [log_theta.as_slice(), log_sigma.as_slice()].concat();
        let mstep = theta_sigma_weighted_mstep_mixture(
            model,
            population,
            &draws_by_class,
            &schedules,
            &pmix,
            &log_theta,
            &log_sigma,
            &mstep_theta_lower,
            &mstep_theta_upper,
            &log_sigma_lower,
            &log_sigma_upper,
            n_theta,
            n_sigma,
            mstep_maxiter,
            &theta_packs_log_mask,
        );
        mstep_motion.record(
            &mstep_before,
            &[mstep_theta_lower.as_slice(), log_sigma_lower.as_slice()].concat(),
            &[mstep_theta_upper.as_slice(), log_sigma_upper.as_slice()].concat(),
            &mstep,
        );
        log_theta = mstep.log_theta;
        log_sigma = mstep.log_sigma;

        // (a2) Apply the class-aware closed-form mu-ref θ shift computed above:
        //      log θ_k += (Σ_i r_ic η̄_ic) / (Σ_i r_ic) over the classes θ_k serves.
        // Ω below is the weighted second moment about zero of these same draws,
        // which is what the single-population `run_mcem` also uses, so the
        // draws are re-centred only after it, for the next IMP proposal.
        let mut applied_shift = vec![0.0f64; n_theta];
        for (t, shift) in mu_ref_shift.iter().enumerate() {
            let Some(mean_eta) = shift else { continue };
            let before = log_theta[t];
            log_theta[t] = (log_theta[t] + mean_eta).clamp(log_theta_lower[t], log_theta_upper[t]);
            applied_shift[t] = log_theta[t] - before;
        }

        theta_cur = (0..n_theta)
            .map(|i| unpack_theta(i, log_theta[i]))
            .collect();
        sigma_cur = log_sigma.iter().map(|&v| v.exp()).collect();

        // (b) mixing coefficients from the class responsibilities.
        mix.rbar = pmix.clone();
        crate::estimation::saem_mixture::mstep_mixing(
            model,
            population,
            &mix,
            &mut theta_cur,
            &init_params.theta_lower,
            &init_params.theta_upper,
            mstep_maxiter,
        );
        for &j in &mix.mixing_theta_idx {
            log_theta[j] = if theta_packs_log_mask[j] {
                theta_cur[j].max(1e-12).ln()
            } else {
                theta_cur[j]
            };
        }

        // (c) Ω (class-shared): responsibility-weighted second moment, masked/floored.
        let mut new_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
        for (c, class) in draws_by_class.iter().enumerate() {
            for (i, d) in class.iter().enumerate() {
                new_omega += pmix[i][c] * &d.second_moment;
            }
        }
        new_omega /= n_subjects as f64;
        for a in 0..n_eta {
            for b in 0..n_eta {
                if !init_params.omega.free_mask[(a, b)] {
                    new_omega[(a, b)] = 0.0;
                }
                let fa = init_params.omega_fixed.get(a).copied().unwrap_or(false);
                let fb = init_params.omega_fixed.get(b).copied().unwrap_or(false);
                if fa || fb {
                    new_omega[(a, b)] = init_params.omega.matrix[(a, b)];
                }
            }
        }
        floor_omega_diagonal(&mut new_omega, &init_params.omega_fixed, OMEGA_DIAG_FLOOR);
        omega_mat = new_omega;

        // Accumulate the final `n_avg` iterations for the reported point estimate.
        if k > n_iter - n_avg {
            for (a, &v) in acc_theta.iter_mut().zip(&theta_cur) {
                *a += v;
            }
            for (a, &v) in acc_sigma.iter_mut().zip(&sigma_cur) {
                *a += v;
            }
            acc_omega += &omega_mat;
            n_acc += 1;
        }

        prev_means = Some(
            draws_by_class
                .iter()
                .map(|class| class.iter().map(|d| d.mean.clone()).collect())
                .collect(),
        );
        prev_draws = if recenter == ProposalRecenter::SampleMoments {
            // Keep φ fixed across the class-aware shift (#1365): class `c`'s
            // posterior of η moved by minus the shift of the θ serving `c`.
            // (`prev_means` above only warm-starts the inner solve, which IMP
            // runs on iteration 1 alone.)
            let mut draws_by_class = draws_by_class;
            for (c, class) in draws_by_class.iter_mut().enumerate() {
                let mut shift = vec![0.0f64; n_eta];
                for pair in &mix_mu_ref_pairs {
                    if let Some(&t) = pair.theta_idx.get(c) {
                        if t < n_theta {
                            shift[pair.eta_idx] = applied_shift[t];
                        }
                    }
                }
                if shift.iter().any(|&a| a != 0.0) {
                    for d in class.iter_mut() {
                        shift_draws(d, &shift);
                    }
                }
            }
            Some(draws_by_class)
        } else {
            None
        };

        iterations_done = k;
        if k < n_iter {
            stopped_after = time_limit_reached(started, max_time);
            if stopped_after.is_some() {
                break;
            }
        }
    }

    // A cancel observed inside (or at the top of) the loop breaks out with
    // parameters from a truncated MCEM. Return before the K-class marginal pass
    // and the covariance step — together the most expensive part of the run —
    // rather than paying for them and handing back a bogus `Ok` (#992 review).
    if crate::cancel::is_cancelled(cancel) {
        return Err("cancelled by user".to_string());
    }
    if let Some(elapsed) = stopped_after {
        warnings.push(time_limit_warning(
            label,
            recenter.max_time_key(),
            elapsed,
            iterations_done,
            n_iter,
            n_acc,
        ));
    }

    if !empty_class_theta.is_empty() {
        let names: Vec<usize> = empty_class_theta.iter().copied().collect();
        warnings.push(format!(
            "{label}: the class(es) served by typical value(s) {} received no responsibility \
             mass on at least one iteration, so the class-aware mu-ref shift could not be \
             computed for them; they were estimated by the importance-weighted M-step on those \
             iterations. A class that collapses this way is usually over-specified — check the \
             mixing estimate and consider fewer classes (#996).",
            name_list(names)
        ));
    }

    // ---- Point estimate: average over the final n_avg iterations ----
    let (final_theta, final_sigma, final_omega_mat) = if n_acc > 0 {
        (
            acc_theta
                .iter()
                .map(|&v| v / n_acc as f64)
                .collect::<Vec<_>>(),
            acc_sigma
                .iter()
                .map(|&v| v / n_acc as f64)
                .collect::<Vec<_>>(),
            acc_omega / n_acc as f64,
        )
    } else {
        (theta_cur.clone(), sigma_cur.clone(), omega_mat.clone())
    };
    let final_omega = OmegaMatrix::from_matrix(
        final_omega_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    // Rebuild the per-class Ω/Σ as the (shared) final estimates so the marginal
    // post-loop pass and covariance step see the mixture structure.
    let final_mixture = init_params.mixture.as_ref().map(|m| {
        let mut mm = m.clone();
        for o in mm.omega.iter_mut() {
            *o = final_omega.clone();
        }
        for s in mm.sigma.iter_mut() {
            s.values = final_sigma.clone();
        }
        mm
    });
    let final_params = ModelParameters {
        theta: final_theta,
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: final_sigma,
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        // IMP/IMPMAP does not update the `block_sigma` off-diagonals
        // (#847); carry them so the sampled residual `R` keeps its
        // declared correlation structure.
        residual_correlations: init_params.residual_correlations.clone(),
        residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
        omega_iov: None,
        kappa_fixed: init_params.kappa_fixed.clone(),
        mixture: final_mixture,
    };

    // ---- Final EBEs / OFV / posteriors via the K-fold marginal ----
    let meval =
        crate::estimation::mixture::mixture_ofv(model, population, &final_params, options, None);
    let eta_hats = meval.mixest_etas;
    let h_matrices = meval.mixest_h_mats;
    let final_kappas = meval.mixest_kappas;
    let ofv = meval.ofv;
    let mixture_posteriors = Some(crate::estimation::outer_optimizer::MixturePosteriors {
        pmix: meval.pmix,
        mixest: meval.mixest,
        n_fd_h_redone: meval.n_fd_h_redone,
        class_nll: meval.class_nll,
    });

    // ---- Covariance step (mixture-aware) ----
    let packed = pack_params(&final_params);
    let cov_out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
        method: covariance_method,
        newton_decrement,
        non_stationary_optimum: _,
    } = cov_out;
    warnings.extend(cov_warnings);

    // #1303 / #528 — the same two rules as the single-population exit above.
    let mut converged = stopped_after.is_none();
    if let Some(w) = gate_converged_on_mcem_objective(&mut converged, ofv) {
        warnings.push(w);
    }
    warnings.extend(mstep_warnings(
        &mut converged,
        label,
        &mstep_motion,
        model,
        stopped_after.is_some(),
    ));

    if verbose {
        eprintln!("{label} (mixture) completed. Final marginal OFV = {ofv:.4}");
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged,
        n_iterations: iterations_done,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        total_ebe_nm_polished: 0,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        left_init: None,
        stationary_start: false,
        mixture_posteriors,
        vi: None,
        start_eval: None,
        mcem_lowest_objective: lowest_iteration_objective,
        // A run `max_time` stopped did not reach the limit the settling verdict is about,
        // and already says why it is not converged.
        mcem_objective_trail: stopped_after
            .is_none()
            .then(|| recenter.objective_trail(iteration_objectives, iteration_mc_se, label)),
        newton_decrement,
        optimizer_stop: None,
        eval_budget_spent: false,
    })
}

/// Below this many importance samples per sampled dimension (with `auto` off),
/// the IMP / IMPMAP M-step moments are flagged as under-sampled (see `run_mcem`).
const MIN_SAMPLES_PER_DIM: usize = 100;

/// The model the IMP / IMPMAP sample-count warning names: `n_eta` alone, or for
/// IOV the widest stacked `[η, κ₁..κ_K]` a subject samples (#2301).
fn sampled_dimension_phrase(n_eta: usize, n_kappa: usize, max_occ: usize) -> String {
    if n_kappa == 0 || max_occ == 0 {
        return format!("a {n_eta}-ETA model");
    }
    format!(
        "a {n_eta}-ETA model with {n_kappa} κ on each of up to {max_occ} occasions \
         ({} sampled dimensions per subject)",
        crate::stats::stacked_layout::stacked_width(n_eta, n_kappa, max_occ)
    )
}

/// The run's current Ω_iov as an [`OmegaMatrix`] carrying the declared
/// structural mask (as SAEM does: a free off-diagonal that happens to be 0 must
/// not be read as structural), or the template unchanged without IOV.
fn omega_iov_snapshot(
    init_params: &ModelParameters,
    current: Option<&DMatrix<f64>>,
) -> Option<OmegaMatrix> {
    match (init_params.omega_iov.as_ref(), current) {
        (Some(t), Some(m)) => Some(OmegaMatrix::from_matrix_with_mask(
            m.clone(),
            t.eta_names.clone(),
            t.diagonal,
            t.free_mask.clone(),
        )),
        _ => init_params.omega_iov.clone(),
    }
}

/// The MAP sweep's κ̂ must have one block per occasion group the joint proposal
/// samples (`n_occ_of`), or the stacked mode would be mis-assembled. The same
/// guard `run_importance_sampling` applies, here on every sweep.
fn check_kappa_blocks(
    population: &Population,
    kappas: &[Vec<DVector<f64>>],
    n_occ_of: &[usize],
    label: &str,
) -> Result<(), String> {
    for (i, subject) in population.subjects.iter().enumerate() {
        let got = kappas.get(i).map_or(0, Vec::len);
        if got != n_occ_of[i] {
            return Err(format!(
                "{label}: subject {} has {got} κ occasion block(s) from the MAP step but {} \
                 occasion group(s) in the data; the joint (η, κ) proposal needs them to match.",
                subject.id, n_occ_of[i],
            ));
        }
    }
    Ok(())
}

/// `[η, κ₁..κ_K]` as one vector — the layout the joint proposal samples, the
/// IOV inner loop warm-starts from, and [`draw_obs_nll`] reads. Just `η` when
/// `kappas` is empty.
fn stack_eta_kappa(eta: &DVector<f64>, kappas: &[DVector<f64>]) -> DVector<f64> {
    let mut v: Vec<f64> = eta.iter().copied().collect();
    for k in kappas {
        v.extend(k.iter().copied());
    }
    DVector::from_vec(v)
}

/// The joint prior `Ω ⊕ I_K ⊗ Ω_iov` of an IOV subject with `K` occasions, the
/// pieces of it the E-step reads, and its defensive component (#2301).
pub(crate) struct JointPrior {
    /// `(Ω ⊕ I_K ⊗ Ω_iov)⁻¹`.
    pub(crate) inv: DMatrix<f64>,
    /// `log det Ω + K · log det Ω_iov`.
    pub(crate) log_det: f64,
    /// `Ω ⊕ I_K ⊗ Ω_iov`, the IMP proposal-covariance cap.
    matrix: DMatrix<f64>,
    pub(crate) mixture: Option<crate::estimation::importance_sampling::DefensiveMixture>,
}

/// One [`JointPrior`] per distinct occasion count in the population, built once
/// per MCEM iteration — the IOV twin of sharing the η-only defensive component
/// across subjects (#528).
pub(crate) struct JointPriors(std::collections::BTreeMap<usize, JointPrior>);

impl JointPriors {
    pub(crate) fn build(
        omega: &OmegaMatrix,
        iov: &OmegaMatrix,
        n_occ_of: &[usize],
        alpha: f64,
    ) -> Self {
        let (n_eta, n_kappa) = (omega.matrix.nrows(), iov.matrix.nrows());
        let mut map = std::collections::BTreeMap::new();
        for &n_occ in n_occ_of.iter().filter(|&&k| k > 0) {
            map.entry(n_occ).or_insert_with(|| {
                let inv = build_joint_omega_inv(&omega.inv, &iov.inv, n_eta, n_kappa, n_occ);
                let mixture = crate::estimation::importance_sampling::DefensiveMixture::new(
                    &inv,
                    stacked_width(n_eta, n_kappa, n_occ),
                    alpha,
                );
                JointPrior {
                    inv,
                    log_det: omega.log_det + n_occ as f64 * iov.log_det,
                    matrix: crate::stats::stacked_layout::block_diag_stack(
                        &omega.matrix,
                        &iov.matrix,
                        n_occ,
                    ),
                    mixture,
                }
            });
        }
        Self(map)
    }

    pub(crate) fn get(&self, n_occ: usize) -> &JointPrior {
        &self.0[&n_occ]
    }
}

thread_local! {
    /// Reused per-occasion κ split for [`draw_obs_nll`].
    static KAPPA_SPLIT: std::cell::RefCell<Vec<Vec<f64>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Split the stacked κ part of a sample into `buf[..n_occ]`, one `n_kappa`
/// block per occasion, reusing `buf`'s allocations; returns `n_occ`. Entries
/// past `n_occ` are left over from a wider subject and must not be read.
fn split_kappas(kappa_part: &[f64], n_kappa: usize, buf: &mut Vec<Vec<f64>>) -> usize {
    let n_occ = kappa_part.len() / n_kappa;
    if buf.len() < n_occ {
        buf.resize_with(n_occ, Vec::new);
    }
    for (dst, src) in buf.iter_mut().zip(kappa_part.chunks(n_kappa)) {
        dst.clear();
        dst.extend_from_slice(src);
    }
    n_occ
}

/// `−log p(yᵢ | b, θ, σ)` at one importance sample `b` of a subject. `b` is η
/// alone (scored exactly as before #2301), or an IOV subject's stacked
/// `[η, κ₁..κ_K]`, scored with per-occasion predictions — the observation NLL
/// SAEM's IOV M-step uses.
pub(crate) fn draw_obs_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    b: &[f64],
    schedule: Option<&EventSchedule>,
    scratch: &mut EventPkParams,
) -> f64 {
    let n_eta = model.n_eta;
    if b.len() > n_eta {
        // Split κ into a per-thread buffer rather than `n_occ` fresh `Vec`s per
        // sample: this runs K times per subject per iteration.
        KAPPA_SPLIT.with(|buf| {
            let mut buf = buf.borrow_mut();
            let n_occ = split_kappas(&b[n_eta..], model.n_kappa, &mut buf);
            obs_nll_subject_into_iov_with_schedule(
                model,
                subject,
                theta,
                sigma,
                &b[..n_eta],
                &buf[..n_occ],
                scratch,
                schedule,
            )
        })
    } else {
        obs_nll_subject_into_with_schedule(
            model,
            subject,
            theta,
            sigma,
            model.residual_cov_theta(theta),
            b,
            scratch,
            schedule,
        )
    }
}

/// Each subject's cached event schedule for the MCEM scoring loops, `None` where
/// reuse is unsound. The E-step and the M-steps score every draw, so a subject on
/// the event-driven walk would otherwise rebuild its schedule K times per
/// evaluation; on vancomycin and infliximab (time-varying covariates) that rebuild
/// was the largest single cost of a fixed-K run (#2687). An IOV subject walks the
/// IOV funnel whatever its covariates, hence its own gate.
fn mcem_schedule_cache(
    model: &CompiledModel,
    population: &Population,
    n_occ_of: &[usize],
) -> Vec<Option<EventSchedule>> {
    population
        .subjects
        .iter()
        .zip(n_occ_of)
        .map(|(subject, &n_occ)| {
            if n_occ > 0 {
                cacheable_schedule_iov(model, subject)
            } else {
                cacheable_schedule(model, subject)
            }
        })
        .collect()
}

/// Closed-form Ω_iov M-step (#2301): the importance-weighted second moment of
/// κ pooled over every occasion of every IOV subject,
/// `Ω_iov = Σᵢ Σ_g E_w[κᵢg κᵢgᵀ] / Σᵢ Kᵢ`, read off the κ diagonal blocks of
/// each subject's stacked second moment. Then the Ω treatment: structural
/// off-diagonals zeroed, `FIX` rows/columns restored from `template`, free
/// diagonal floored. `None` when no subject has an occasion (nothing to
/// estimate from — Ω_iov is left where it was).
fn omega_iov_mstep(
    draws: &[SubjectDraws],
    n_occ_of: &[usize],
    n_eta: usize,
    template: &OmegaMatrix,
    kappa_fixed: &[bool],
) -> Option<DMatrix<f64>> {
    let n_kappa = template.matrix.nrows();
    let mut m = DMatrix::<f64>::zeros(n_kappa, n_kappa);
    let mut n_blocks = 0usize;
    for (d, &n_occ) in draws.iter().zip(n_occ_of) {
        for g in 0..n_occ {
            let r = kappa_range(n_eta, n_kappa, g);
            m += d.second_moment.view((r.start, r.start), (n_kappa, n_kappa));
        }
        n_blocks += n_occ;
    }
    if n_blocks == 0 {
        return None;
    }
    m /= n_blocks as f64;
    for i in 0..n_kappa {
        for j in 0..n_kappa {
            if !template.free_mask[(i, j)] {
                m[(i, j)] = 0.0;
            }
            let fi = kappa_fixed.get(i).copied().unwrap_or(false);
            let fj = kappa_fixed.get(j).copied().unwrap_or(false);
            if fi || fj {
                m[(i, j)] = template.matrix[(i, j)];
            }
        }
    }
    floor_omega_diagonal(&mut m, kappa_fixed, OMEGA_DIAG_FLOOR);
    Some(m)
}

/// Extract the lower triangle of a square matrix in row-major order:
/// `(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), …`
fn lower_triangle(m: &DMatrix<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            out.push(m[(i, j)]);
        }
    }
    out
}

/// The value the weighted θ/σ M-step minimizes at natural-space `theta` /
/// `sigma`: `Σᵢ Σₖ w̃ᵢₖ · obs_nll(yᵢ | bᵢₖ, θ, σ)`, or `1e20` when that is not
/// finite. Each sample `bᵢₖ` is scored by [`draw_obs_nll`], so an IOV subject's
/// samples are scored with their own κ (#2301).
fn weighted_mstep_objective(
    model: &CompiledModel,
    population: &Population,
    draws: &[SubjectDraws],
    schedules: &[Option<EventSchedule>],
    theta: &[f64],
    sigma: &[f64],
) -> f64 {
    // Reduce in subject order (collect then serial sum) so the objective is
    // reproducible regardless of rayon worker count — a parallel `.sum()`
    // folds along thread-count-dependent split points and f64 addition is
    // non-associative (#703).
    let per_subj: Vec<f64> = population
        .subjects
        .par_iter()
        .zip(draws.par_iter())
        .zip(schedules.par_iter())
        .map_init(
            EventPkParams::default,
            |scratch, ((subject, d), schedule)| {
                let mut s = 0.0f64;
                for (w, b) in d.weights.iter().zip(d.etas.iter()) {
                    if *w == 0.0 {
                        continue;
                    }
                    s += w * draw_obs_nll(
                        model,
                        subject,
                        theta,
                        sigma,
                        b,
                        schedule.as_ref(),
                        scratch,
                    );
                }
                s
            },
        )
        .collect();
    let val: f64 = per_subj.iter().sum();
    if val.is_finite() {
        val
    } else {
        1e20
    }
}

/// Stable token of [`gate_converged_on_mstep_motion`].
pub(crate) const MCEM_MSTEP_FROZEN_TOKEN: &str = "W_MCEM_MSTEP_FROZEN";

/// Stable token of [`gate_converged_on_mstep_failures`].
pub(crate) const MCEM_MSTEP_FAILED_TOKEN: &str = "W_MCEM_MSTEP_FAILED";

/// Smallest displacement from its start, in packed units, that counts as a coordinate of
/// the weighted M-step having moved (#2718). The reported freeze left coordinates equal
/// to their start to 6 decimals.
const MSTEP_MOVED_TOL: f64 = 1e-6;

/// What the weighted θ/σ M-step did over a run (#2718).
///
/// Nothing else can see a frozen coordinate, because Ω keeps the objective falling and
/// #2686's settling test reads the objective. Not moving is not enough to call one frozen:
/// a coordinate that starts at its optimum has no reason to move. So it also has to be
/// [`improvable_coordinates`] after the last solve that owned it.
struct MstepMotion {
    /// Each coordinate's value before the first solve that owned it; `None` while pinned.
    start: Vec<Option<f64>>,
    moved: Vec<bool>,
    /// Whether a step along the coordinate alone beat the latest solve that owned it.
    improvable: Vec<bool>,
    solves: usize,
    failed: usize,
    last_failure: Option<nlopt::FailState>,
}

impl MstepMotion {
    fn new(n: usize) -> Self {
        MstepMotion {
            start: vec![None; n],
            moved: vec![false; n],
            improvable: vec![false; n],
            solves: 0,
            failed: 0,
            last_failure: None,
        }
    }

    /// Fold in one solve. `before` is the packed vector it started from and `lower` /
    /// `upper` its bounds, θ then σ; a coordinate with `upper == lower` was pinned.
    fn record(&mut self, before: &[f64], lower: &[f64], upper: &[f64], step: &WeightedMstep) {
        self.solves += 1;
        if let Some(state) = step.failure {
            self.failed += 1;
            self.last_failure = Some(state);
        }
        let after = step.log_theta.iter().chain(&step.log_sigma);
        for (i, &x) in after.enumerate() {
            if upper[i] <= lower[i] {
                continue;
            }
            let start = *self.start[i].get_or_insert(before[i]);
            self.moved[i] |= (x - start).abs() > MSTEP_MOVED_TOL;
            self.improvable[i] = step.improvable[i];
        }
    }

    /// The coordinates a solve owned, never moved, and could still have improved, in
    /// θ-then-σ order.
    fn frozen(&self) -> Vec<usize> {
        (0..self.start.len())
            .filter(|&i| self.start[i].is_some() && !self.moved[i] && self.improvable[i])
            .collect()
    }
}

/// Demote `converged` when the weighted M-step left a parameter it owns at its start
/// although moving it would have paid (#2718), and say why.
fn gate_converged_on_mstep_motion(
    converged: &mut bool,
    label: &str,
    motion: &MstepMotion,
    model: &CompiledModel,
) -> Option<String> {
    let frozen = motion.frozen();
    if frozen.is_empty() {
        return None;
    }
    *converged = false;
    let names: Vec<&str> = frozen
        .iter()
        .map(|&i| {
            model
                .theta_names
                .iter()
                .chain(&model.default_params.sigma.names)
                .nth(i)
                .map_or("?", String::as_str)
        })
        .collect();
    Some(format!(
        "{MCEM_MSTEP_FROZEN_TOKEN}: {label}: [{}] never left the starting values in {} \
         weighted M-step(s), though a step along each alone still lowers −2 log L by more \
         than {MCEM_SETTLE_MIN_GAP}, so the fit is reported converged: false. No other step of the run updates them, so \
         these are starting values and not estimates. Start closer, mu-reference them so \
         the closed-form step owns them, or chain a FOCE/FOCEI stage first.",
        names.join(", "),
        motion.solves,
    ))
}

/// Demote `converged` when NLopt failed any weighted M-step (#2718). Independent of
/// whether the coordinates moved, because an earlier solve moving them says nothing about
/// a later one. BOBYQA has no failure a sound problem reaches: past roundoff, which
/// [`mstep_failure`] lets through, it only reports bad arguments, memory and a forced
/// stop. So one is enough, and there is no count to calibrate.
fn gate_converged_on_mstep_failures(
    converged: &mut bool,
    label: &str,
    motion: &MstepMotion,
) -> Option<String> {
    let state = motion.last_failure?;
    *converged = false;
    Some(format!(
        "{MCEM_MSTEP_FAILED_TOKEN}: {label}: NLopt failed {} of {} weighted M-step(s) \
         ({state:?}), so the fit is reported converged: false. A failed solve can leave θ \
         and σ where they were for that iteration. This is a solver fault, not a property \
         of the data: please report it.",
        motion.failed, motion.solves,
    ))
}

/// Everything the weighted M-step has to report at the end of a run. A run `max_time`
/// stopped is not judged on motion: it may have had too few solves to tell, and it
/// already reports why it is not converged. A failure stands either way.
fn mstep_warnings(
    converged: &mut bool,
    label: &str,
    motion: &MstepMotion,
    model: &CompiledModel,
    stopped: bool,
) -> Vec<String> {
    let failed = gate_converged_on_mstep_failures(converged, label, motion);
    let frozen = if stopped {
        None
    } else {
        gate_converged_on_mstep_motion(converged, label, motion, model)
    };
    failed.into_iter().chain(frozen).collect()
}

/// For each coordinate of the solve's result `x` that did not move from `start`, whether a
/// step along it alone, inside its bounds, gains a likelihood-ratio step (#2718):
/// [`MCEM_SETTLE_MIN_GAP`] on −2 log L, so half of it on `obj`, which is −log L.
///
/// The solve's own tolerance is too low a bar. On `impmap_mstep_tests`'s fixture the
/// additive σ, which its data barely inform, read as improvable at 1e-7 relative from a
/// converged solve's result. The step is the solve's first step, so a coordinate within
/// about half of it of its optimum reads `false`, as does one the objective is flat in, and
/// one on a bound with its optimum outside. Moved and pinned coordinates read `false`.
fn improvable_coordinates<F>(
    obj: &F,
    start: &[f64],
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Vec<bool>
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
{
    let unmoved = |i: usize| upper[i] > lower[i] && (x[i] - start[i]).abs() <= MSTEP_MOVED_TOL;
    let mut improvable = vec![false; x.len()];
    if !(0..x.len()).any(unmoved) {
        return improvable;
    }
    let at = obj(x, None, &mut ());
    let mut probe = x.to_vec();
    for i in (0..x.len()).filter(|&i| unmoved(i)) {
        let step = mstep_initial_step(lower[i], upper[i]);
        for candidate in [(x[i] - step).max(lower[i]), (x[i] + step).min(upper[i])] {
            probe[i] = candidate;
            improvable[i] |=
                candidate != x[i] && 2.0 * (at - obj(&probe, None, &mut ())) > MCEM_SETTLE_MIN_GAP;
        }
        probe[i] = x[i];
    }
    improvable
}

/// What one weighted θ/σ M-step returned.
struct WeightedMstep {
    log_theta: Vec<f64>,
    log_sigma: Vec<f64>,
    /// The NLopt failure that ended the solve, if it was one that can leave the start in
    /// place. `RoundoffLimited` is not: BOBYQA reports it at the end of a solve that
    /// has run out of precision, with its best point in hand.
    failure: Option<nlopt::FailState>,
    /// [`improvable_coordinates`] at the returned point, θ then σ.
    improvable: Vec<bool>,
}

#[cfg(test)]
thread_local! {
    /// Fail every weighted M-step on this thread after this many more, for a driver test
    /// of the gates: a working BOBYQA moves every free coordinate, even one the objective
    /// is flat in.
    static FAIL_MSTEP_SOLVES_AFTER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Whether [`FAIL_MSTEP_SOLVES_AFTER`] fails this solve.
#[cfg(test)]
fn mstep_solve_forced_to_fail() -> bool {
    let Some(left) = FAIL_MSTEP_SOLVES_AFTER.get() else {
        return false;
    };
    FAIL_MSTEP_SOLVES_AFTER.set(Some(left.saturating_sub(1)));
    left == 0
}

#[cfg(not(test))]
fn mstep_solve_forced_to_fail() -> bool {
    false
}

/// The failure an NLopt outcome reports, under [`WeightedMstep::failure`]'s meaning.
fn mstep_failure(
    outcome: &Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>,
) -> Option<nlopt::FailState> {
    match outcome {
        Ok(_) | Err((nlopt::FailState::RoundoffLimited, _)) => None,
        Err((state, _)) => Some(*state),
    }
}

/// Run one weighted θ/σ M-step: BOBYQA on the packed vector `x`, θ then σ, from a local
/// first design ([`mstep_initial_step`]) to [`MSTEP_FTOL_REL`] (#2718).
///
/// These are the settings #1415 gave SAEM. Before them this solve took NLopt's default
/// first design, a quarter of the bound width and so about 5 units on (−10, 10), and
/// stopped at `ftol_rel = 1e-4`, about 0.15 on an objective near 1480. On a 14-coordinate
/// model every M-step then ended `FtolReached` with 2–3 coordinates moved. The first step
/// is the main lever; loosening the tolerance alone moved 4–6.
///
/// The mixture M-step shares this configuration. SAEM's mixture arm keeps the old one
/// (`MSTEP_MIXTURE_FTOL_REL`) because it solves on hard class assignments. This solve is
/// weighted by the soft responsibilities, so it is an ordinary EM step and a converged
/// solve is the right target.
fn solve_weighted_mstep<F>(
    obj: F,
    mut x: Vec<f64>,
    lower: &[f64],
    upper: &[f64],
    n_theta: usize,
    maxiter: u32,
) -> WeightedMstep
where
    F: Fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64,
{
    let n = x.len();
    let start = x.clone();
    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Bobyqa,
        n,
        &obj,
        nlopt::Target::Minimize,
        (),
    );
    opt.set_lower_bounds(lower).unwrap();
    opt.set_upper_bounds(upper).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(MSTEP_FTOL_REL).unwrap();
    let initial_step: Vec<f64> = lower
        .iter()
        .zip(upper)
        .map(|(&lo, &hi)| mstep_initial_step(lo, hi))
        .collect();
    opt.set_initial_step(&initial_step).unwrap();

    let failure = if mstep_solve_forced_to_fail() {
        Some(nlopt::FailState::Failure)
    } else {
        mstep_failure(&opt.optimize(&mut x))
    };
    // A configuration NLopt rejects is a bug here, not a property of the fit. A release
    // build reports it through `MstepMotion`.
    debug_assert!(
        !matches!(failure, Some(nlopt::FailState::InvalidArgs)),
        "IMP weighted M-step: NLopt rejected the problem (InvalidArgs) — check the initial \
         step against the bounds"
    );
    let improvable = improvable_coordinates(&obj, &start, &x, lower, upper);
    let log_sigma = x.split_off(n_theta);
    WeightedMstep {
        log_theta: x,
        log_sigma,
        failure,
        improvable,
    }
}

/// Weighted θ/σ M-step: minimize the importance-weighted observation NLL
/// `Σᵢ Σₖ w̃ᵢₖ · obs_nll(yᵢ | ηᵢₖ, θ, σ)` over the per-subject sample sets, using
/// derivative-free NLopt BOBYQA in packed log-space, warm-started from the
/// current parameters. Mirrors SAEM's `theta_sigma_mstep_light` but sums over
/// the `K` weighted samples per subject instead of a single EBE.
#[allow(clippy::too_many_arguments)]
fn theta_sigma_weighted_mstep(
    model: &CompiledModel,
    population: &Population,
    draws: &[crate::estimation::importance_sampling::SubjectDraws],
    schedules: &[Option<EventSchedule>],
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    theta_packs_log_mask: &[bool],
) -> WeightedMstep {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    // Weighted observation NLL, parallel over subjects. Each subject contributes
    // Σₖ w̃ₖ · obs_nll(yᵢ | ηᵢₖ, θ, σ).
    let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        weighted_mstep_objective(model, population, draws, schedules, &th, &sg)
    };

    solve_weighted_mstep(obj, x, &lower, &upper, n_theta, maxiter)
}

/// Mixture (#985) weighted θ/σ M-step. Minimises the class-responsibility- and
/// importance-weighted observation NLL
/// `Σᵢ Σ_c PMIX_ic · Σₖ w̃_ick · obs_nll(yᵢ | η_ick, θ, σ | MIXNUM = c)` over the
/// per-(subject,class) sample sets. Each class's samples were drawn under that
/// class's proposal, and its `obs_nll` runs under a [`MixtureClassGuard`] on the
/// worker thread, so a class-switched typical value (`if MIXNUM == c …`) is
/// estimated from its own class's weighted samples. `draws[c][i]` is class `c`'s
/// draws for subject `i`; `pmix[i][c]` is the class responsibility.
#[allow(clippy::too_many_arguments)]
fn theta_sigma_weighted_mstep_mixture(
    model: &CompiledModel,
    population: &Population,
    draws: &[Vec<crate::estimation::importance_sampling::SubjectDraws>],
    schedules: &[Option<EventSchedule>],
    pmix: &[Vec<f64>],
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    theta_packs_log_mask: &[bool],
) -> WeightedMstep {
    use crate::parser::model_parser::MixtureClassGuard;
    let n = n_theta + n_sigma;
    let n_classes = draws.len();

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);
    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);
    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let per_subj: Vec<f64> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                let mut s = 0.0f64;
                for c in 0..n_classes {
                    let p = pmix[i][c];
                    if p == 0.0 {
                        continue;
                    }
                    let _g = MixtureClassGuard::enter(c + 1);
                    let d = &draws[c][i];
                    let mut sc = 0.0f64;
                    for (w, eta) in d.weights.iter().zip(d.etas.iter()) {
                        if *w == 0.0 {
                            continue;
                        }
                        sc += w * obs_nll_subject_into_with_schedule(
                            model,
                            subject,
                            &th,
                            &sg,
                            model.residual_cov_theta(&th),
                            eta,
                            scratch,
                            schedules[i].as_ref(),
                        );
                    }
                    s += p * sc;
                }
                s
            })
            .collect();
        let val: f64 = per_subj.iter().sum();
        if val.is_finite() {
            val
        } else {
            1e20
        }
    };

    solve_weighted_mstep(obj, x, &lower, &upper, n_theta, maxiter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MCEM core's run-time n_eta = 0 guard asks `requires_random_effects`
    /// of `recenter.method()` (#1050), so each strategy must name the method it
    /// implements — and both must be members, or that guard silently disappears.
    #[test]
    fn proposal_recenter_names_its_method_and_both_require_random_effects() {
        assert_eq!(ProposalRecenter::Map.method(), EstimationMethod::Impmap);
        assert_eq!(
            ProposalRecenter::SampleMoments.method(),
            EstimationMethod::Imp
        );
        for r in [ProposalRecenter::Map, ProposalRecenter::SampleMoments] {
            assert!(r.method().requires_random_effects());
        }
    }

    /// #2588 (M11): the MAP sweep's `n_nm_polished` counts the subjects whose returned EBE
    /// the certificate moved under `inner_optimizer = nelder_mead`. Neither caller of
    /// [`run_map_multistart`] reads its stats today (`_stats` at both sites), so no fit can
    /// observe this count; only a direct call can pin it. The oracle is the same solve per
    /// subject on the calling thread: with no warm start and `mceta = 0` the sweep's EBE is
    /// `find_ebe` from η = 0. At warfarin's `inner_tol = 1e-5` every NM stop needs the move
    /// (#2571), so the count is non-zero; under the default optimizer it is 0.
    #[test]
    fn the_map_sweep_counts_the_moved_subjects_under_nelder_mead() {
        use crate::estimation::inner_optimizer::{arm_inner_settings, InnerSettings};
        use crate::types::{FitOptions, InnerOptimizer};
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let p = &model.default_params;
        let (max_iter, tol) = (200, 1e-5);
        let mu_k = compute_mu_k(&model, &p.theta, true);
        let sweep = |mode| {
            let opts = FitOptions {
                inner_optimizer: mode,
                ..Default::default()
            };
            crate::api::with_fit_scope(&opts, || {
                run_map_multistart(&model, &pop, p, max_iter, tol, None, &mu_k, 0, 1, 0).3
            })
            .expect("the scope builds")
        };
        let expected = {
            let _armed = arm_inner_settings(InnerSettings {
                mode: InnerOptimizer::NelderMead,
                warm: false,
            });
            pop.subjects
                .iter()
                .filter(|s| find_ebe(&model, s, p, max_iter, tol, None, Some(&mu_k), 0).nm_polished)
                .count()
        };
        let nm = sweep(InnerOptimizer::NelderMead);
        eprintln!(
            "IMPMAP sweep under nelder_mead: {} of {} moved",
            nm.n_nm_polished,
            pop.subjects.len()
        );
        assert!(expected > 0, "fixture: no subject moved");
        assert_eq!(nm.n_nm_polished, expected, "{nm:?}");
        assert_eq!(sweep(InnerOptimizer::Auto).n_nm_polished, 0);
    }

    // ── #996 end-to-end helpers: a tiny 2-class mixture dataset ──────────
    //
    // Two weight groups with well-separated clearances, so the class draw is
    // stable and a handful of iterations is enough to exercise the M-step.

    /// Small 1-cpt IV dataset, `n_per` subjects at each of two weights.
    fn mix996_csv(n_per: usize) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (g, &wt) in [60.0_f64, 90.0].iter().enumerate() {
            let cl = if g == 0 { 1.0 } else { 3.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
                for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0].iter().enumerate() {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                    let dv = c * (1.0 + 0.02 * ((sid + ti) as f64).sin());
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    /// `TVP` anchors `ETA_CL` and `ETA_V` both — no single closed-form shift — and
    /// `{q}` decides whether a covariate group on `ETA_Q` also reads it.
    fn shared_anchor_model_1630(q: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVP(5.0, 0.01, 100.0)
  theta TH_Q(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_Q ~ 0.09
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVP * exp(ETA_CL)
  V  = TVP * 10.0 * exp(ETA_V)
  Q  = {q}

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V, q=Q, v2=50.0)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("parses")
    }

    fn mix996_pop(n_per: usize) -> Population {
        use std::io::Write;
        let csv = mix996_csv(n_per);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// Two-class mixture. `v_expr` picks whether V carries a class-shared
    /// mu-referenced ETA (`TVV * exp(ETA_V)`) or none (`TVV`).
    fn mix996_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("mixture model parses")
    }

    #[test]
    fn impmap_mixture_applies_class_aware_mu_ref_shift() {
        // The MIXNUM-switched clearances are mu-referenced per class, so the
        // "no class-aware mu-referencing" fallback warning must be absent.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("no class-aware mu-referencing was applied")),
            "class-aware shift should be active, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn impmap_mixture_mu_referencing_off_restores_numerical_mstep() {
        // The degenerate-oracle switch: `mu_referencing = false` must reproduce
        // the pre-#996 behaviour, where every θ goes through the weighted M-step.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("no class-aware mu-referencing was applied")),
            "expected the fallback warning with mu_referencing off, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn split_mu_ref_pins_leaves_a_mass_less_class_theta_free() {
        // theta 1's classes drew no responsibility mass this iteration, so the
        // shift is `None`: it must be handed back to the M-step, not pinned
        // (pinning it with no shift freezes it for the rest of the fit).
        let shift = vec![Some(0.2), None, Some(-0.1)];
        let (pinned, free) = split_mu_ref_pins(&[0, 1, 2], &shift);
        assert_eq!(pinned, vec![0, 2]);
        assert_eq!(free, vec![1]);
    }

    #[test]
    fn split_mu_ref_pins_pins_every_theta_when_all_classes_have_mass() {
        let shift = vec![Some(0.2), Some(0.0)];
        let (pinned, free) = split_mu_ref_pins(&[0, 1], &shift);
        assert_eq!(pinned, vec![0, 1]);
        assert!(free.is_empty());
    }

    #[test]
    fn imp_single_population_identity_packed_mu_ref_routes_to_weighted_mstep() {
        // A θ declared with a negative lower bound is packed on the identity
        // scale, so `log_theta[t]` is the raw θ and the closed-form shift would
        // apply `θ += mean(η)` where the EM optimum is `θ *= exp(mean(η))`. The
        // single-population IMP path must apply the same guard the mixture path
        // does (#996 review) — warn and route the θ to the weighted M-step.
        let src = r"
[parameters]
  theta TVCL(1.0, -5.0, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 2;
        opts.imp_samples = 40;
        opts.imp_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("packed on the identity scale"))
            .unwrap_or_else(|| {
                panic!("expected the identity-pack warning, got {:?}", res.warnings)
            });
        assert!(hit.contains("TVCL"), "TVCL named in the warning: {hit}");
        assert!(
            !hit.contains("TVV"),
            "TVV is log-packed and must not be listed: {hit}"
        );
    }

    /// The IMP twin of SAEM's `saem_shared_anchor_warning_drops_a_theta_a_group_took_in`:
    /// a theta a covariate group took both pairs of in (#1630) is moved by the
    /// group, so the shared-anchor warning must not name it. The control side
    /// is `imp_single_population_shared_anchor_routes_to_weighted_mstep` below.
    #[test]
    fn imp_shared_anchor_warning_drops_a_theta_a_group_took_in() {
        let model = shared_anchor_model_1630("TVP * TH_Q * exp(ETA_Q)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 2;
        opts.imp_samples = 40;
        opts.imp_seed = Some(1630);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w
                    .contains("ETA_CL (anchored on TVP) and ETA_V (anchored on TVP) share TVP")),
            "premise: the group took both pairs in, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("mu-reference anchor of more than one ETA")),
            "{:?}",
            res.warnings
        );
    }

    #[test]
    fn imp_single_population_shared_anchor_routes_to_weighted_mstep() {
        // Two ETAs mu-referenced to the same θ: the closed form shifts a packed θ
        // by *one* η mean and pins it, so there is no single well-defined update
        // — both pairs are dropped and TVP is named (codex review of #1375).
        let src = r"
[parameters]
  theta TVP(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVP * exp(ETA_CL)
  V  = TVP * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 2;
        opts.imp_samples = 40;
        opts.imp_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("mu-reference anchor of more than one ETA"))
            .unwrap_or_else(|| {
                panic!("expected the shared-anchor warning, got {:?}", res.warnings)
            });
        assert!(hit.contains("TVP"), "TVP named in the warning: {hit}");
        // …and the fit still falls back cleanly: with no eligible pair left, the
        // weighted M-step owns every θ.
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("no closed-form mu-referenced parameters found")),
            "expected the no-closed-form fallback warning, got {:?}",
            res.warnings
        );
    }

    /// The eligible twin: a mixture whose **only** mu-reference is a class-shared
    /// logit anchor, identity-packed as the closed form wants. The class-switched
    /// clearances carry no ETA, so the class-aware pair set is empty unless the
    /// logit anchor is in it — which is exactly what the "no class-aware
    /// mu-referencing was applied" fallback warning reports. Before the codex
    /// review of #1375 the mixture pair builder filtered on `log_transformed()`
    /// and this warning fired (#918).
    #[test]
    fn impmap_mixture_takes_a_class_shared_logit_anchor() {
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(-0.405465, -10.0, 10.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_F ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 else TVCL2
  V  = TVV
  F  = inv_logit(LOGIT_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("no class-aware mu-referencing was applied")),
            "the logit anchor is an eligible class-shared pair, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the log scale")),
            "LOGIT_F is identity-packed; no packing advisory expected, got {:?}",
            res.warnings
        );
    }

    /// A **mixture** whose class-shared anchor is logit-scale but declared with a
    /// non-negative lower bound: log-packed, so the class-aware shift cannot run
    /// and the #918 advisory names it. Before the codex review of #1375 a logit
    /// anchor never reached the mixture pair set at all, so neither this warning
    /// nor its eligible twin existed.
    #[test]
    fn impmap_mixture_log_packed_logit_anchor_warns() {
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(0.405465, 0.0, 10.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_F ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV
  F  = inv_logit(LOGIT_F + ETA_F)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V, f=F)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("packed on the log scale"))
            .unwrap_or_else(|| {
                panic!("expected the #918 packing advisory, got {:?}", res.warnings)
            });
        assert!(hit.contains("LOGIT_F"), "LOGIT_F named: {hit}");
    }

    #[test]
    fn imp_log_packed_mu_ref_keeps_the_closed_form_shift() {
        // Control for the test above: with a non-negative lower bound the same
        // model keeps the closed-form shift and emits no identity-pack warning.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 2;
        opts.imp_samples = 40;
        opts.imp_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the identity scale")),
            "no identity-pack warning expected, got {:?}",
            res.warnings
        );
    }

    /// Mirror image of the identity-packed case for a *logit* mu-ref (#918):
    /// `inv_logit(LOGIT_F + ETA_F)` has mu scale θ, but a non-negative lower
    /// bound makes `LOGIT_F` log-packed, so the closed-form shift does not apply.
    /// The θ must route to the weighted M-step with the packing advisory naming
    /// it, while the ordinary log-packed lognormal `TVCL` stays unlisted.
    #[test]
    fn imp_log_packed_logit_mu_ref_routes_to_weighted_mstep() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta LOGIT_F(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  F  = inv_logit(LOGIT_F + ETA_F)
  CL = TVCL * exp(ETA_CL) * F
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        assert_eq!(
            model.mu_refs.get("ETA_F").map(|m| m.transform),
            Some(crate::types::MuTransform::Logit),
            "fixture must be detected as a logit mu-ref"
        );
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 2;
        opts.imp_samples = 40;
        opts.imp_auto = false;
        opts.imp_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("packed on the log scale"))
            .unwrap_or_else(|| {
                panic!("expected the #918 packing advisory, got {:?}", res.warnings)
            });
        assert!(
            hit.contains("LOGIT_F"),
            "LOGIT_F named in the advisory: {hit}"
        );
        assert!(
            !hit.contains("TVCL"),
            "TVCL is a log-packed lognormal, not listed: {hit}"
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("packed on the identity scale")),
            "no identity-pack advisory for this model, got {:?}",
            res.warnings
        );
    }

    /// Eight subjects with CRCL 40..140, CL generated from the additive renal
    /// model (`CL = 5 + (CRCL-90)*0.05`, no IIV in the data) so a covariate
    /// group has real between-subject structure to fit in a handful of
    /// iterations. `V = 50`.
    fn covmuref_csv(with_v_eta: bool) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,CRCL\n");
        for (i, crcl) in [40.0_f64, 55.0, 70.0, 85.0, 95.0, 110.0, 125.0, 140.0]
            .iter()
            .enumerate()
        {
            let id = i + 1;
            let cl = 5.0 + (crcl - 90.0) * 0.05;
            let v = if with_v_eta {
                50.0 * (0.1 * ((id as f64) - 4.5) / 4.5).exp()
            } else {
                50.0
            };
            s.push_str(&format!("{id},0,0,100,1,1,{crcl}\n"));
            for (ti, t) in [1.0_f64, 4.0, 8.0, 16.0, 24.0].iter().enumerate() {
                let c = 100.0 / v * (-(cl / v) * t).exp();
                let dv = c * (1.0 + 0.03 * ((id + ti) as f64).sin());
                s.push_str(&format!("{id},{t},{dv:.6},0,0,1,{crcl}\n"));
            }
        }
        s
    }

    fn covmuref_pop(with_v_eta: bool) -> Population {
        use std::io::Write;
        let csv = covmuref_csv(with_v_eta);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["CRCL"]), None).unwrap()
    }

    /// The #619 additive form with the covariate group as the **only**
    /// eta-bearing parameter (`V` carries no eta), so a closed-form channel
    /// exists if and only if the group is active.
    const COVMUREF_ONLY_MODEL: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.02, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// #619 under IMP: the group's thetas are pinned out of the weighted M-step
    /// and moved by the group step; the "no closed-form mu-referenced
    /// parameters" advisory (which would fire for this model without the
    /// group) stays silent, and the slope moves toward the data-generating 0.05.
    #[test]
    fn imp_covariate_mu_ref_group_replaces_the_weighted_mstep_for_its_thetas() {
        let model =
            crate::parser::model_parser::parse_model_string(COVMUREF_ONLY_MODEL).expect("parses");
        assert_eq!(model.covariate_mu_refs.len(), 1);
        let pop = covmuref_pop(false);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 6;
        opts.imp_samples = 60;
        opts.imp_auto = false;
        opts.imp_seed = Some(619);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("no closed-form mu-referenced parameters found")),
            "the group is a closed-form channel, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("covariate mu-reference on")),
            "no group should be dropped, got {:?}",
            res.warnings
        );
        let slope = res.theta[1];
        assert!(
            slope > 0.02,
            "TH_CRCL must move from its 0.02 start: {slope}"
        );
    }

    /// [`COVMUREF_ONLY_MODEL`] started where the additive typical value is
    /// negative for the lowest-CRCL subject: `TVCL + (40 − 90)·TH_CRCL =
    /// 4 − 50·0.2 = −6`, so the lognormal `mu` is `NaN` for that subject and
    /// both group engines decline.
    const COVMUREF_INADMISSIBLE_START: &str = r"
[parameters]
  theta TVCL(4.0, 0.0, 100.0)
  theta TH_CRCL(0.2, 0.0, 5.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.01 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    /// A group that cannot step must not have its thetas pinned out of the
    /// weighted M-step (#918 review).
    ///
    /// IMP sets the pin bounds *before* the M-step and only learns the solve
    /// returned `None` afterwards, so pinning unconditionally left `TVCL` and
    /// `TH_CRCL` frozen at their initial values for the whole fit — with no
    /// warning, under a converged-looking result. The regression this catches
    /// is exactly that: both thetas still bit-equal to their starts.
    #[test]
    fn imp_does_not_freeze_a_group_that_cannot_step() {
        let model = crate::parser::model_parser::parse_model_string(COVMUREF_INADMISSIBLE_START)
            .expect("parses");
        let pop = covmuref_pop(false);
        // Premise: the start really is inadmissible, so the fixture reaches the
        // branch under test rather than passing on a healthy group.
        let omega = model.default_params.omega.matrix.clone();
        let (groups, _) = crate::estimation::covariate_mu_ref::resolve_covariate_mu_groups(
            &model,
            &pop,
            &[],
            &model.default_params.theta_fixed,
            &omega,
        );
        assert_eq!(groups.len(), 1, "premise: a group is resolved");
        assert!(
            !groups[0].can_step(
                &model.default_params.theta,
                &pop,
                &model.default_params.theta_fixed
            ),
            "premise: the group cannot step from this start"
        );

        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Imp;
        opts.imp_iterations = 4;
        opts.imp_samples = 60;
        opts.imp_auto = false;
        opts.imp_seed = Some(918);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMP Ok");

        let tvcl0 = model.default_params.theta[0];
        let th0 = model.default_params.theta[1];
        assert!(
            (res.theta[0] - tvcl0).abs() > 1e-9 || (res.theta[1] - th0).abs() > 1e-9,
            "both group thetas frozen at their starts ({tvcl0}, {th0}) — the pin was set for a \
             group that could not step: got ({}, {})",
            res.theta[0],
            res.theta[1]
        );
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("had no admissible step")),
            "the skipped step must be reported, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn impmap_mixture_flags_class_anchors_when_the_shift_is_disabled() {
        // With `mu_referencing = false` the class thetas are estimated by the
        // weighted M-step alone — exactly the #406 situation — so the advisory
        // must fire. It must use the anchor-specific wording, not "NO associated
        // ETA": these thetas do carry an ETA in the model text (#996 review).
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        let hit = res
            .warnings
            .iter()
            .find(|w| w.contains("class-aware mu-ref shift is disabled"))
            .unwrap_or_else(|| panic!("expected the #406 advisory, got {:?}", res.warnings));
        assert!(
            hit.contains("TVCL1") && hit.contains("TVCL2"),
            "both class anchors listed: {hit}"
        );
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("NO associated ETA") && w.contains("TVCL1")),
            "class anchors must not be reported as ETA-less, got {:?}",
            res.warnings
        );
    }

    /// #1715 end to end through IMPMAP's `fit()`: a Box-Cox typical value and
    /// its shape θ are reported as carrying a non-mu-referenceable ETA, and only
    /// the genuinely ETA-less `TVV` gets the "add an ETA" advice.
    #[test]
    fn impmap_boxcox_typical_value_is_not_reported_as_eta_less() {
        let model = crate::parser::model_parser::parse_model_string(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta LAMBDA(0.5, 0.05, 2.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  ETAT = (exp(ETA_CL)^LAMBDA - 1) / LAMBDA
  CL   = TVCL * exp(ETAT)
  V    = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
",
        )
        .expect("model parses");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(1715);
        opts.run_covariance_step = false;
        let ws = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("IMPMAP Ok")
            .warnings;
        let no_eta = ws
            .iter()
            .find(|w| w.contains("NO associated ETA"))
            .unwrap_or_else(|| panic!("expected the ETA-less advisory, got {ws:?}"));
        assert!(no_eta.contains("[TVV]"), "only TVV is ETA-less: {no_eta}");
        let with_eta = ws
            .iter()
            .find(|w| w.contains(IMP_WITH_ETA_SENTENCES[0]))
            .unwrap_or_else(|| panic!("expected the transformed-ETA advisory, got {ws:?}"));
        assert!(with_eta.contains("[TVCL, LAMBDA]"), "{with_eta}");
    }

    /// The same split on the mixture (`run_mcem_mixture`) site: a Box-Cox `V`
    /// shared by both classes carries `ETA_V` but no mu-reference, so `TVV` is
    /// named in the transformed-ETA advisory and not as ETA-less (#1715).
    #[test]
    fn impmap_mixture_reports_a_boxcox_typical_value_as_carrying_an_eta() {
        let model = mix996_model("TVV * exp((exp(ETA_V)^0.5 - 1) / 0.5)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(1715);
        opts.run_covariance_step = false;
        let ws = crate::api::fit(&model, &pop, &model.default_params, &opts)
            .expect("IMPMAP Ok")
            .warnings;
        assert!(
            !ws.iter().any(|w| w.contains("NO associated ETA")),
            "TVV carries ETA_V, got {ws:?}"
        );
        let with_eta = ws
            .iter()
            .find(|w| w.contains(IMP_WITH_ETA_SENTENCES[0]))
            .unwrap_or_else(|| panic!("expected the transformed-ETA advisory, got {ws:?}"));
        assert!(with_eta.contains("[TVV]"), "{with_eta}");
    }

    #[test]
    fn impmap_mixture_active_shift_suppresses_the_class_anchor_advisory() {
        // The other side of the switch: with the shift active the class thetas
        // are owned by the closed form, so neither advisory fires for them.
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("class-aware mu-ref shift is disabled")),
            "shift is active — no advisory expected, got {:?}",
            res.warnings
        );
        assert!(
            !res.warnings.iter().any(|w| w.contains("NO associated ETA")),
            "every theta is anchored, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn impmap_mixture_weak_iiv_mu_ref_routes_to_weighted_mstep() {
        // A near-zero ω on the mu-ref η carries no mean-shift information, so the
        // class θ must fall back to the weighted M-step with a warning (#411).
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 1e-6 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("negligible variance") && w.contains("TVCL1")),
            "expected the weak-IIV warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn objective_converged_rejects_runaway_and_nonfinite() {
        // A normal objective converges.
        assert!(objective_converged(-249.23));
        assert!(objective_converged(0.0));
        // A legitimately large objective for a huge cohort still converges —
        // 1e15 leaves head-room above any real −2logL.
        assert!(objective_converged(1.0e7));
        // Non-finite is divergence.
        assert!(!objective_converged(f64::NAN));
        assert!(!objective_converged(f64::INFINITY));
        assert!(!objective_converged(f64::NEG_INFINITY));
        // The bounded blowup (issue #528): finite but enormous → not converged,
        // so a collapsed-weight runaway can't masquerade as a good fit or win
        // multi-start selection. The lowered cap also rejects a less-extreme
        // 1e20 runaway that the old 1e30 cap waved through.
        assert!(!objective_converged(1e35));
        assert!(!objective_converged(1e20));
        assert!(!objective_converged(OFV_DIVERGENCE_CAP));
    }

    /// #2652: every row is a run from the 2026-11 evaluation (lowest iteration, final
    /// objective, observations), so the bound is pinned to what separated them.
    #[test]
    fn mcem_objective_rise_separates_the_measured_collapses_from_healthy_runs() {
        // Linux aarch64 (glibc 2.41): (lowest iteration, final marginal, n_obs). See
        // `MCEM_RISE_TOLERANCE` for the setup. The collapses are from the tree before
        // #2650, which removed their cause.
        for (lowest, end, n_obs) in [
            (4313.3406, 13354.7091, 702), // cefepime, warm: the issue's fit
            (4300.6956, 6480.1505, 702),  // cefepime, cold: the narrowest collapse
            (661.4346, 1589.4539, 183),   // infliximab, warm
            (703.0588, 1545.6446, 183),   // infliximab, cold: ends below its 3087.4 start
            (998.0913, 3814.4033, 274),   // vancomycin, warm
            (-79.8594, 2126.9567, 132),   // thioguanine, cold
        ] {
            assert!(
                mcem_objective_rise(end, lowest, n_obs).is_some(),
                "a collapse passed: {lowest} → {end}"
            );
        }
        for (lowest, end, n_obs) in [
            (4276.0932, 4279.2241, 702),    // cefepime, cold
            (4279.0421, 4281.7779, 702),    // cefepime, warm
            (665.3862, 665.4009, 183),      // infliximab, cold
            (660.8735, 661.1218, 183),      // infliximab, warm
            (995.9150, 996.7820, 274),      // vancomycin, warm
            (5963.0649, 5966.0838, 1231),   // pembrolizumab, warm
            (-3054.2406, -3053.8571, 1006), // propofol, warm
            (-250.8932, -250.5679, 132),    // thioguanine, cold: the widest gap
            (-250.8932, -250.5775, 132),    // thioguanine, warm
            (-286.1552, -286.0387, 110),    // warfarin, cold
            (-286.1552, -286.0393, 110),    // warfarin, warm
        ] {
            assert_eq!(
                mcem_objective_rise(end, lowest, n_obs),
                None,
                "a healthy run fired: {lowest} → {end}"
            );
        }
    }

    /// The scale's two halves, each on the input only it decides. Dropping `abs()` makes
    /// a negative objective's tolerance negative, so a tiny rise fires; dropping the
    /// `n_obs` floor makes noise around zero a 1000 % rise.
    #[test]
    fn mcem_objective_rise_scales_on_the_magnitude_and_the_observation_count() {
        assert_eq!(mcem_objective_rise(-990.0, -1000.0, 10), None);
        assert_eq!(mcem_objective_rise(-850.0, -1000.0, 10), Some(150.0));
        assert_eq!(mcem_objective_rise(5.0, 0.4, 100), None);
        assert!(mcem_objective_rise(11.0, 0.4, 100).is_some());
        // A fall is never a rise, however large.
        assert_eq!(mcem_objective_rise(-5000.0, 4000.0, 100), None);
    }

    /// A non-finite iteration must neither become the baseline nor erase it. `-inf`
    /// is the case `f64::min` gets wrong: nothing could ever rise above it.
    #[test]
    fn lowest_objective_keeps_the_lowest_finite_iteration() {
        assert_eq!(lowest_objective(None, f64::NAN), None);
        assert_eq!(lowest_objective(None, f64::INFINITY), None);
        assert_eq!(lowest_objective(None, 7.0), Some(7.0));
        assert_eq!(lowest_objective(Some(7.0), 9.0), Some(7.0));
        assert_eq!(lowest_objective(Some(7.0), 5.0), Some(5.0));
        assert_eq!(lowest_objective(Some(7.0), f64::NAN), Some(7.0));
        assert_eq!(lowest_objective(Some(7.0), f64::NEG_INFINITY), Some(7.0));
    }

    /// The last 100 iteration objectives of four Linux runs, to one decimal (#2686).
    /// `THIOGUANINE_NORMAL_S777` and both `*_NORMAL` runs are cold, with
    /// `imp_proposal_df = normal` and `imp_averaging = 1`; `CEFEPIME_CHAINED` is
    /// `[focei, imp]`.
    const THIOGUANINE_NORMAL_S777: [f64; 100] = [
        -244.1, -239.6, -244.7, -250.1, -250.3, -245.2, -249.6, -250.3, -249.7, -250.1, -251.0,
        -245.0, -249.1, -250.3, -249.9, -247.9, -243.0, -248.1, -250.3, -251.0, -249.7, -250.0,
        -250.5, -242.5, -235.5, -248.5, -245.9, -252.1, -230.2, -236.3, -231.7, -247.7, -246.9,
        -249.8, -249.8, -249.7, -250.2, -249.6, -250.1, -241.9, -219.7, -210.6, -203.1, -193.6,
        -198.6, -174.1, -168.2, -152.0, -154.0, -140.1, -132.8, -138.9, -156.8, -171.7, -166.4,
        -185.8, -175.3, -166.4, -184.0, -194.6, -207.6, -226.2, -214.4, -192.2, -205.9, -219.9,
        -225.0, -234.8, -241.2, -234.3, -239.5, -241.1, -243.0, -231.7, -244.7, -247.5, -250.4,
        -228.3, -248.6, -250.3, -249.9, -249.9, -249.5, -235.9, -250.9, -249.8, -250.1, -249.6,
        -246.4, -250.1, -249.5, -250.4, -251.3, -249.7, -242.5, -250.3, -249.6, -249.4, -249.7,
        -240.3,
    ];
    const VANCOMYCIN_NORMAL: [f64; 100] = [
        1082.2, 1065.9, 1080.5, 1073.9, 1073.0, 1085.9, 1088.0, 1083.3, 1090.1, 1083.7, 1071.6,
        1072.7, 1069.6, 1066.2, 1088.0, 1084.2, 1080.3, 1079.9, 1089.2, 1073.5, 1073.4, 1059.9,
        1070.9, 1081.7, 1087.3, 1087.4, 1076.7, 1057.2, 1053.6, 1061.8, 1051.1, 1047.9, 1044.0,
        1034.1, 1039.0, 1046.4, 1050.2, 1052.8, 1045.6, 1034.7, 1050.5, 1048.8, 1048.4, 1056.7,
        1036.0, 1040.9, 1034.3, 1035.2, 1031.7, 1041.1, 1034.8, 1044.9, 1040.5, 1047.2, 1034.8,
        1025.4, 1025.9, 1032.5, 1026.1, 1019.5, 1016.8, 1015.5, 1014.2, 1013.4, 1013.1, 1013.2,
        1012.6, 1011.7, 1011.6, 1010.9, 1011.1, 1010.6, 1010.4, 1010.2, 1010.9, 1009.5, 1017.0,
        1008.8, 1009.7, 1008.7, 1007.5, 1008.7, 1008.7, 1008.1, 1007.2, 1007.1, 1005.5, 1006.5,
        1006.5, 1005.9, 1006.0, 1005.9, 1005.4, 1005.3, 1005.0, 1005.0, 1005.7, 1013.7, 1005.5,
        1004.9,
    ];
    const CEFEPIME_NORMAL: [f64; 100] = [
        4320.0, 4334.1, 4327.5, 4375.1, 4328.1, 4318.2, 4323.3, 4361.0, 4355.0, 4348.9, 4339.6,
        4340.6, 4387.1, 4410.3, 4410.0, 4396.3, 4384.4, 4387.2, 4392.9, 4410.2, 4403.9, 4404.8,
        4371.3, 4339.6, 4319.8, 4318.7, 4362.7, 4380.7, 4362.1, 4368.2, 4375.1, 4357.9, 4384.1,
        4385.5, 4336.3, 4341.1, 4367.3, 4361.1, 4356.2, 4334.8, 4335.0, 4344.9, 4391.7, 4359.9,
        4346.5, 4380.1, 4342.3, 4340.2, 4340.6, 4338.2, 4344.4, 4367.5, 4356.7, 4361.1, 4356.7,
        4331.9, 4337.3, 4341.0, 4330.4, 4321.5, 4352.0, 4322.1, 4353.5, 4348.0, 4365.4, 4348.0,
        4316.9, 4348.6, 4319.8, 4311.9, 4362.3, 4366.8, 4377.4, 4394.0, 4421.7, 4405.3, 4373.7,
        4397.4, 4355.7, 4376.3, 4415.6, 4431.8, 4444.7, 4419.2, 4426.9, 4403.2, 4393.9, 4439.8,
        4413.7, 4392.6, 4370.5, 4379.0, 4362.3, 4408.1, 4439.1, 4442.3, 4470.3, 4423.3, 4408.5,
        4398.6,
    ];
    const CEFEPIME_CHAINED: [f64; 100] = [
        4282.0, 4283.1, 4282.0, 4279.9, 4294.0, 4283.2, 4328.1, 4305.3, 4300.7, 4306.5, 4292.6,
        4294.0, 4283.5, 4282.8, 4281.1, 4280.6, 4280.5, 4284.6, 4281.6, 4282.7, 4282.0, 4284.2,
        4286.7, 4280.7, 4282.2, 4277.6, 4285.5, 4279.3, 4277.2, 4277.4, 4281.9, 4281.4, 4283.9,
        4282.2, 4286.5, 4279.7, 4288.7, 4280.5, 4270.2, 4278.8, 4279.7, 4283.4, 4277.6, 4300.6,
        4289.9, 4301.2, 4311.1, 4307.9, 4307.7, 4299.2, 4281.5, 4292.5, 4292.4, 4278.6, 4280.4,
        4289.0, 4310.3, 4277.8, 4277.3, 4294.1, 4277.4, 4281.1, 4286.0, 4276.8, 4280.4, 4278.7,
        4277.6, 4289.7, 4282.4, 4281.0, 4278.8, 4275.0, 4277.7, 4283.9, 4310.5, 4290.2, 4294.1,
        4277.0, 4313.9, 4291.0, 4309.4, 4295.5, 4278.5, 4275.6, 4304.8, 4289.8, 4287.7, 4280.0,
        4292.3, 4297.0, 4312.6, 4290.1, 4285.6, 4279.1, 4279.1, 4284.8, 4292.2, 4279.2, 4297.5,
        4294.4,
    ];

    /// A 150-iteration trail ending in `tail`. The first 50 entries are absurd on
    /// purpose: at that length the test never reads the first third.
    fn trail_ending_in(tail: &[f64], mc_se: f64) -> McemObjectiveTrail {
        let mut objectives = vec![1e9; 150 - tail.len()];
        objectives.extend_from_slice(tail);
        ProposalRecenter::SampleMoments.objective_trail(objectives, vec![mc_se; 150], "IMP")
    }

    /// A marginal with no noise of its own, for tests about the floor.
    fn exact(value: f64) -> Option<FinalMarginal> {
        Some(FinalMarginal { value, mc_se: 0.0 })
    }

    /// Two windows of 50, at the two floors given. Most of each window sits 10 above its
    /// floor, so the floor is what the test has to read.
    fn two_windows(previous_floor: f64, floor: f64, mc_se: f64) -> McemObjectiveTrail {
        let window = |f: f64| {
            let mut v = vec![f + 10.0; 44];
            v.extend_from_slice(&[f; 6]);
            v
        };
        trail_ending_in(&[window(previous_floor), window(floor)].concat(), mc_se)
    }

    /// The two runs NONMEM's evaluator scored short of the optimum are the two this
    /// fires on, each by its own arm. The other two are the hard negative controls: the
    /// thioguanine run swings by more than either, and chained cefepime is noisy.
    #[test]
    fn settling_separates_the_measured_runs() {
        let settling = |tail: &[f64], se, value, mc_se| {
            trail_ending_in(tail, se).settling(Some(FinalMarginal { value, mc_se }))
        };
        assert!(matches!(
            settling(&VANCOMYCIN_NORMAL, 0.29, 1004.72, 0.23),
            McemSettling::Falling { window: 50, .. }
        ));
        assert!(matches!(
            settling(&CEFEPIME_NORMAL, 2.54, 4288.06, 1.09),
            McemSettling::AboveMarginal { window: 50, .. }
        ));
        assert_eq!(
            settling(&THIOGUANINE_NORMAL_S777, 0.69, -249.82, 0.23),
            McemSettling::Settled
        );
        assert_eq!(
            settling(&CEFEPIME_CHAINED, 2.24, 4277.29, 1.26),
            McemSettling::Settled
        );
    }

    /// Cefepime's trace alone says nothing, which is why the test sits in `fit()`: its
    /// floor is flat, so only the marginal shows the run is short of the optimum.
    #[test]
    fn a_plateau_needs_the_marginal_to_be_seen() {
        let trail = trail_ending_in(&CEFEPIME_NORMAL, 2.54);
        assert_eq!(trail.settling(None), McemSettling::Settled);
    }

    #[test]
    fn a_gap_must_clear_both_the_minimum_and_the_noise() {
        // Falling arm. 3.0 is 30 SEs and under the minimum, 5.0 clears both, and with
        // an SE of 2 the 5.0 is under 4 SEs.
        assert_eq!(
            two_windows(103.0, 100.0, 0.1).settling(None),
            McemSettling::Settled
        );
        assert!(matches!(
            two_windows(105.0, 100.0, 0.1).settling(None),
            McemSettling::Falling { .. }
        ));
        assert_eq!(
            two_windows(105.0, 100.0, 2.0).settling(None),
            McemSettling::Settled
        );
        // Marginal arm, the same three cases.
        let flat = |se| two_windows(100.0, 100.0, se);
        assert_eq!(flat(0.1).settling(exact(97.0)), McemSettling::Settled);
        assert!(matches!(
            flat(0.1).settling(exact(95.0)),
            McemSettling::AboveMarginal { .. }
        ));
        assert_eq!(flat(2.0).settling(exact(95.0)), McemSettling::Settled);
    }

    /// The reported values, each distinct so that swapping two fails.
    #[test]
    fn settling_reports_the_floors_it_compared() {
        let mut falling = two_windows(120.0, 100.0, 0.5);
        falling.mc_se[50..100].fill(0.75);
        assert_eq!(
            falling.settling(None),
            McemSettling::Falling {
                window: 50,
                previous_floor: 120.0,
                floor: 100.0,
                previous_mc_se: 0.75,
                mc_se: 0.5,
            }
        );
        assert_eq!(
            two_windows(100.0, 100.0, 0.5).settling(Some(FinalMarginal {
                value: 90.0,
                mc_se: 0.25,
            })),
            McemSettling::AboveMarginal {
                window: 50,
                floor: 100.0,
                marginal: FinalMarginal {
                    value: 90.0,
                    mc_se: 0.25,
                },
                mc_se: 0.5,
            }
        );
    }

    /// A rising floor is not "still falling", and a marginal above the floor is not
    /// "floor above the marginal": `W_MCEM_OBJECTIVE_ROSE` owns that direction.
    #[test]
    fn settling_is_one_sided() {
        assert_eq!(
            two_windows(100.0, 120.0, 0.1).settling(None),
            McemSettling::Settled
        );
        assert_eq!(
            two_windows(100.0, 100.0, 0.1).settling(exact(120.0)),
            McemSettling::Settled
        );
    }

    /// Five of 50 iterations can sit anywhere below the floor without moving it, and any
    /// number above it. The sixth lowest value is the one read.
    #[test]
    fn the_floor_ignores_excursions_on_either_side() {
        let mut w = vec![100.0; 50];
        assert_eq!(objective_floor(w.clone()), 100.0);
        w[..5].fill(50.0);
        assert_eq!(objective_floor(w.clone()), 100.0);
        w[5] = 50.0;
        assert_eq!(objective_floor(w.clone()), 50.0);
        let mut w = vec![100.0; 50];
        w[..44].fill(1e6);
        assert_eq!(objective_floor(w.clone()), 100.0);
    }

    /// A run needs three windows of ten, and a shorter schedule reads a third of itself.
    #[test]
    fn settling_needs_three_windows_and_skips_the_first_third() {
        let trail = |n: usize| {
            let third = n / 3;
            let mut o = vec![1e9; n - 2 * third];
            o.extend(vec![110.0; third]);
            o.extend(vec![100.0; third]);
            ProposalRecenter::SampleMoments.objective_trail(o, vec![0.1; n], "IMP")
        };
        assert_eq!(trail(29).settling(None), McemSettling::TooShort);
        assert!(matches!(
            trail(30).settling(None),
            McemSettling::Falling { window: 10, .. }
        ));
        assert!(matches!(
            trail(90).settling(None),
            McemSettling::Falling { window: 30, .. }
        ));
        assert!(matches!(
            trail(150).settling(None),
            McemSettling::Falling { window: 50, .. }
        ));
        // The window stops growing at 50: both of them now sit in the flat last third.
        assert_eq!(trail(600).settling(None), McemSettling::Settled);
    }

    /// A non-finite iteration is dropped rather than read as a floor. With too few left
    /// the run cannot be judged, and a marginal that is not a number convicts nothing:
    /// `W_NONFINITE_MARGINAL` reports it.
    #[test]
    fn settling_survives_non_finite_values() {
        let mut t = two_windows(120.0, 100.0, 0.1);
        t.objectives[120] = f64::NEG_INFINITY;
        t.objectives[121] = f64::NAN;
        assert!(matches!(
            t.settling(None),
            McemSettling::Falling { floor, .. } if floor == 100.0
        ));
        let flat = two_windows(100.0, 100.0, 0.1);
        for marginal in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert_eq!(flat.settling(exact(marginal)), McemSettling::Settled);
        }
        // A marginal whose own SE is unusable cannot convict either, even at a gap of 50.
        for mc_se in [f64::NAN, f64::INFINITY] {
            let marginal = Some(FinalMarginal { value: 50.0, mc_se });
            assert_eq!(flat.settling(marginal), McemSettling::Settled);
        }
    }

    /// Each of the four columns the test reads can leave it unable to read a window, and
    /// that is `Unusable`, not `TooShort`: the run was long enough. `usable` counts the
    /// worse window, and ten finite values are enough.
    #[test]
    fn settling_tells_an_unusable_trace_from_a_short_one() {
        let columns: [(fn(&mut McemObjectiveTrail) -> &mut Vec<f64>, usize); 4] = [
            (|t| &mut t.objectives, 50),
            (|t| &mut t.objectives, 100),
            (|t| &mut t.mc_se, 50),
            (|t| &mut t.mc_se, 100),
        ];
        for (column, start) in columns {
            for (poison, expected) in [(f64::NAN, 9), (f64::INFINITY, 9)] {
                let mut t = two_windows(100.0, 100.0, 0.1);
                column(&mut t)[start..start + 41].fill(poison);
                assert_eq!(
                    t.settling(None),
                    McemSettling::Unusable {
                        window: 50,
                        usable: expected
                    },
                    "column at {start}"
                );
            }
            let mut t = two_windows(100.0, 100.0, 0.1);
            column(&mut t)[start..start + 40].fill(f64::NAN);
            assert_eq!(t.settling(None), McemSettling::Settled, "column at {start}");
        }
    }

    /// The first third is not read, so nothing there can make a trace unusable.
    #[test]
    fn settling_ignores_non_finite_values_before_the_windows() {
        let mut t = two_windows(100.0, 100.0, 0.1);
        t.objectives[..50].fill(f64::NAN);
        t.mc_se[..50].fill(f64::NAN);
        assert_eq!(t.settling(None), McemSettling::Settled);
    }

    /// Each window's noise is its own median, so one bad iteration's SE cannot excuse a
    /// falling floor, and iterations before the windows do not count.
    #[test]
    fn settling_reads_the_noise_of_each_window_only() {
        let mut t = two_windows(120.0, 100.0, 0.5);
        t.mc_se[50..100].fill(0.75);
        t.mc_se[149] = 1e6;
        t.mc_se[99] = 1e6;
        t.mc_se[..50].fill(1e6);
        assert!(matches!(
            t.settling(None),
            McemSettling::Falling { previous_mc_se, mc_se, .. }
                if previous_mc_se == 0.75 && mc_se == 0.5
        ));
    }

    /// The case `imp_auto` makes: the earlier window sampled less, so its floor is the
    /// noisy one and the last window's SE alone would convict. A gap of 20 clears 4 of
    /// the last window's SEs (0.4) and not 4 of the earlier window's (6).
    #[test]
    fn settling_weighs_the_earlier_windows_noise() {
        let ramped = |previous_se: f64| {
            let mut t = two_windows(120.0, 100.0, 0.1);
            t.mc_se[50..100].fill(previous_se);
            t
        };
        assert_eq!(ramped(6.0).settling(None), McemSettling::Settled);
        assert!(matches!(
            ramped(0.1).settling(None),
            McemSettling::Falling { .. }
        ));
        // The SEs combine, so a gap clearing each alone can fall short of both: 4 · 3 = 12
        // and 4 · 3√2 = 16.97, with no bias between equal SEs.
        let both = |gap: f64| two_windows(100.0 + gap, 100.0, 3.0).settling(None);
        assert_eq!(both(16.9), McemSettling::Settled);
        assert!(matches!(both(17.0), McemSettling::Falling { .. }));
    }

    /// Fewer samples raise the objective by `se²/4` with no change in the estimates,
    /// which outruns the noise bound once the SE passes 16: at 20 the excess is 100 and
    /// the bound 80, so a fall has to pass 180.
    #[test]
    fn settling_takes_the_sampling_bias_off_the_fall() {
        let ramped = |fall: f64| {
            let mut t = two_windows(100.0 + fall, 100.0, 0.0);
            t.mc_se[50..100].fill(20.0);
            t.settling(None)
        };
        assert_eq!(ramped(90.0), McemSettling::Settled);
        assert_eq!(ramped(179.9), McemSettling::Settled);
        assert!(matches!(ramped(180.1), McemSettling::Falling { .. }));
    }

    /// `se²/4`, on the earlier window only: a noisier last window is no reason to think
    /// the earlier one sat high, and must not lift the fall either.
    #[test]
    fn sampling_bias_counts_only_against_the_earlier_window() {
        assert_eq!(sampling_bias_excess(4.0, 0.0), 4.0);
        assert_eq!(sampling_bias_excess(4.0, 2.0), 3.0);
        assert_eq!(sampling_bias_excess(2.0, 2.0), 0.0);
        assert_eq!(sampling_bias_excess(0.0, 4.0), 0.0);
    }

    /// The final pass draws the configured count while `imp_auto` ramps the iterations,
    /// so the marginal can be the noisier of the two. The gap here is 20: far past 4 of
    /// the window's SEs, and within 4 of the marginal's. The SEs combine, so a gap that
    /// clears each alone (5.0 > 4 · 1) can still fall short of both (4 · √2 = 5.66).
    #[test]
    fn a_noisy_marginal_cannot_convict_a_settled_trace() {
        let flat = |se| two_windows(100.0, 100.0, se);
        let marginal = |value, mc_se| Some(FinalMarginal { value, mc_se });
        assert!(matches!(
            flat(0.1).settling(marginal(80.0, 0.1)),
            McemSettling::AboveMarginal { .. }
        ));
        assert_eq!(
            flat(0.1).settling(marginal(80.0, 6.0)),
            McemSettling::Settled
        );
        assert_eq!(
            flat(1.0).settling(marginal(95.0, 1.0)),
            McemSettling::Settled
        );
        assert!(matches!(
            flat(1.0).settling(marginal(94.0, 1.0)),
            McemSettling::AboveMarginal { .. }
        ));
    }

    /// The marginal's SE judges only the marginal: two floors share no such noise.
    #[test]
    fn a_noisy_marginal_does_not_excuse_a_falling_floor() {
        let noisy = Some(FinalMarginal {
            value: 100.0,
            mc_se: 50.0,
        });
        assert!(matches!(
            two_windows(120.0, 100.0, 0.1).settling(noisy),
            McemSettling::Falling { .. }
        ));
    }

    #[test]
    fn log_marginal_mc_variance_is_gewekes_form_with_a_finite_fallback() {
        assert_eq!(log_marginal_mc_variance(0.5, 100), 0.01);
        assert_eq!(log_marginal_mc_variance(1.0, 100), 0.0);
        assert_eq!(log_marginal_mc_variance(0.0, 100), 1.0);
    }

    /// Weighted by the squared responsibility, so a subject settled in one class gets
    /// that class's variance and the other's proposal is irrelevant.
    #[test]
    fn a_mixture_subject_pays_for_its_classes_by_responsibility() {
        let var =
            |r: &[f64], ess: &[f64]| mixture_log_marginal_mc_variance(r, ess.iter().copied(), 100);
        assert_eq!(var(&[1.0, 0.0], &[0.5, 0.0]), 0.01);
        assert_eq!(var(&[0.5, 0.5], &[0.5, 0.5]), 0.005);
        assert_eq!(var(&[0.5, 0.5], &[0.5, 1.0]), 0.0025);
    }

    /// Each driver's record has one row per iteration, and cites its own options.
    #[test]
    fn each_mcem_driver_publishes_its_trail() {
        let check = |res: &OuterResult, n: usize, label: &str, prefix: &str| {
            let t = res.mcem_objective_trail.as_ref().expect("trail published");
            assert_eq!(t.objectives.len(), n);
            assert_eq!(t.mc_se.len(), n);
            assert!(t.objectives.iter().all(|v| v.is_finite()), "{t:?}");
            assert!(t.mc_se.iter().all(|v| v.is_finite()), "{t:?}");
            assert!(t.mc_se.iter().any(|&v| v > 0.0), "{t:?}");
            assert_eq!(t.label, label);
            assert_eq!(t.iterations_option, format!("{prefix}_iterations"));
            assert_eq!(t.samples_option, format!("{prefix}_samples"));
            assert_eq!(t.proposal_option, format!("{prefix}_proposal_df"));
        };
        let model = iov_model();
        let pop = iov_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 3;
        opts.impmap_samples = 120;
        opts.impmap_seed = Some(11);
        opts.imp_iterations = 4;
        opts.imp_samples = 100;
        opts.imp_seed = Some(11);
        opts.run_covariance_step = false;
        let params = &model.default_params;
        let impmap = run_impmap(&model, &pop, params, None, &opts).expect("impmap");
        check(&impmap, 3, "IMPMAP", "impmap");
        let imp = run_imp(&model, &pop, params, None, &opts).expect("imp");
        check(&imp, 4, "IMP", "imp");

        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let params = &model.default_params;
        let mixture = run_impmap(&model, &pop, params, None, &opts).expect("mixture impmap");
        check(&mixture, 3, "IMPMAP", "impmap");
        // The wrapper reaches the mixture too: too short to test, and saying so.
        let told = |w: &String| w.starts_with("W_MCEM_SETTLING_UNTESTED: IMPMAP: 3 iteration(s)");
        assert!(mixture.warnings.iter().any(told), "{:?}", mixture.warnings);
        let marginal =
            evaluate_final_marginal(&model, &pop, &mixture, &opts, EstimationMethod::Impmap)
                .expect("mixture marginal");
        assert!(marginal.minus2_log_likelihood.is_finite());
        assert_eq!(marginal.n_samples, opts.impmap_samples);
    }

    /// The baseline `fit()` reads is the lowest iteration row, and not simply the first:
    /// the premise asserts this fixture's objective falls, or the two would coincide.
    #[test]
    fn run_mcem_publishes_the_lowest_iteration_objective() {
        let model = iov_model();
        let pop = iov_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 4;
        opts.impmap_samples = 200;
        opts.impmap_seed = Some(11);
        opts.impmap_trace = true;
        opts.run_covariance_step = false;
        let res = run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap");
        let trace = res.impmap_trace.as_ref().expect("impmap_trace requested");
        let rows: Vec<f64> = trace
            .rows
            .iter()
            .filter(|r| r.iteration > 0)
            .map(|r| r.ofv)
            .collect();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|v| v.is_finite()), "rows {rows:?}");
        let lowest = rows.iter().copied().fold(f64::INFINITY, f64::min); // nan-ok: asserted finite above
        assert!(
            lowest < rows[0],
            "premise: the objective falls, rows {rows:?}"
        );
        assert_eq!(res.mcem_lowest_objective, Some(lowest));
    }

    /// #1303: the MCEM exits apply **two** objective rules, and this pins that
    /// neither is redundant — the trap being that two conditions rejecting the
    /// same inputs cover for each other, so deleting either leaves the suite
    /// green and the gate has quietly stopped gating.
    ///
    /// The witnesses are disjoint and measured, not reasoned:
    ///
    /// * `5e14` — inside #528's `abs() < 1e15` cap, past #1303's one-sided
    ///   `DIVERGENCE_OFV = 1e14` sentinel cutoff.
    /// * `-1e16` — past #528's cap, inside the sentinel cutoff, which is
    ///   deliberately one-sided because a large negative −2 log L is legitimate.
    ///
    /// Mutation (run): delete the `gate_converged_on_objective` call from
    /// `gate_converged_on_mcem_objective` → the `5e14` half fires. Delete the
    /// `objective_converged` half → the `-1e16` half fires. Neither mutation is
    /// caught by the other's witness.
    #[test]
    fn the_two_mcem_objective_rules_reject_different_runs() {
        use crate::estimation::outer_optimizer::{ofv_is_valid, NONFINITE_OBJECTIVE_TOKEN};

        // The witnesses separate the two rules — assert that, or the rest of this
        // test is two copies of one check.
        assert!(objective_converged(5e14) && !ofv_is_valid(5e14));
        assert!(!objective_converged(-1e16) && ofv_is_valid(-1e16));

        for ofv in [5e14, -1e16, f64::NAN, 2e20] {
            let mut converged = true;
            let msg = gate_converged_on_mcem_objective(&mut converged, ofv)
                .unwrap_or_else(|| panic!("ofv = {ofv:?} must demote an MCEM verdict"));
            assert!(!converged, "ofv = {ofv:?}");
            assert!(
                msg.contains(NONFINITE_OBJECTIVE_TOKEN),
                "every demotion carries the classifier's token: {msg}"
            );
        }

        // ...and a real MCEM objective is left alone.
        let mut converged = true;
        assert!(gate_converged_on_mcem_objective(&mut converged, -249.23).is_none());
        assert!(converged);
    }

    #[test]
    fn flags_non_fixed_theta_without_eta() {
        // CL/V have ETAs (mu-referenced); FRAC is an estimated theta with no ETA;
        // TVFIX is FIX. Only FRAC should be flagged.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.0)
  theta TVV(10.0, 0.0)
  theta FRAC(0.5, 0.0, 1.0)
  theta TVFIX(2.0, FIX)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = FRAC + TVFIX

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let fixed = &model.default_params.theta_fixed;
        let flagged = non_fixed_unanchored_thetas(&model, fixed, &[], &[]);
        assert!(
            flagged.contains(&"FRAC".to_string()),
            "FRAC flagged: {flagged:?}"
        );
        assert!(!flagged.contains(&"TVCL".to_string()), "TVCL has ETA");
        assert!(!flagged.contains(&"TVV".to_string()), "TVV has ETA");
        assert!(!flagged.contains(&"TVFIX".to_string()), "TVFIX is FIX");
    }

    /// One required fragment per sentence of the IMP/IMPMAP transformed-ETA
    /// advisory, so deleting any sentence fails
    /// [`imp_transformed_eta_advisory_text`].
    const IMP_WITH_ETA_SENTENCES: [&str; 4] = [
        "enter an individual parameter that carries an ETA, but not as a mu-referenced typical value",
        "needs `log P = log TVP + ETA_P`, linear in the ETA; a transformed ETA",
        "estimated solely through the importance-weighted M-step",
        "no reason to remove the transform: cross-check the fit against FOCEI, or hold the parameter FIX",
    ];

    /// #1715 on the IMP/IMPMAP side: a θ whose ETA is there in a
    /// non-mu-referenceable form gets its own message, without the "STRONGLY
    /// add an ETA" advice; an ETA-less θ keeps the pre-#1715 text. Every cell of
    /// (with-ETA list, ETA-less list) is asserted.
    #[test]
    fn imp_transformed_eta_advisory_text() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(imp_unanchored_theta_warnings("IMPMAP", &[], &[]).is_empty());

        let ws = imp_unanchored_theta_warnings("IMPMAP", &names(&["TVCL", "LAMBDA"]), &[]);
        assert_eq!(ws.len(), 1, "{ws:?}");
        let w = &ws[0];
        assert!(
            w.starts_with("IMPMAP: estimated parameter(s) [TVCL, LAMBDA] "),
            "{w}"
        );
        for frag in IMP_WITH_ETA_SENTENCES {
            assert!(w.contains(frag), "missing {frag:?} in: {w}");
        }
        assert!(!w.contains("NO associated ETA"), "{w}");
        assert!(!w.contains("STRONGLY add an ETA"), "{w}");

        let ws = imp_unanchored_theta_warnings("IMP", &[], &names(&["TVKA"]));
        assert_eq!(ws.len(), 1, "{ws:?}");
        let w = &ws[0];
        assert!(
            w.starts_with("IMP: estimated parameter(s) [TVKA] have NO associated ETA"),
            "{w}"
        );
        assert!(w.contains("STRONGLY add an ETA"), "{w}");
        assert!(!w.contains(IMP_WITH_ETA_SENTENCES[0]), "{w}");

        let ws = imp_unanchored_theta_warnings("IMP", &names(&["LAMBDA"]), &names(&["TVKA"]));
        assert_eq!(ws.len(), 2, "{ws:?}");
        let no_eta = ws.iter().find(|w| w.contains("NO associated ETA")).unwrap();
        let with_eta = ws
            .iter()
            .find(|w| w.contains(IMP_WITH_ETA_SENTENCES[0]))
            .unwrap();
        assert!(
            no_eta.contains("[TVKA]") && !no_eta.contains("LAMBDA"),
            "{no_eta}"
        );
        assert!(
            with_eta.contains("[LAMBDA]") && !with_eta.contains("TVKA"),
            "{with_eta}"
        );
    }

    /// The covariate exponent `TH_WT` of the #1415 fixture shape shares `CL` with
    /// `ETA_CL` but anchors no mu-reference (a local assigned before an `if` with
    /// no `else` and again in its arm defeats #619 detection; the `if`/`else`
    /// form is a group since #1422), so it lands in the carries-an-ETA list, not the ETA-less one
    /// whose advice is "add an ETA" (#1715).
    #[test]
    fn a_covariate_exponent_sharing_cl_with_its_eta_is_classified_as_carrying_one() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TH_WT(0.75, -2.0, 3.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.04

[individual_parameters]
  WTN = 1.0
  if (WT > 0) {
    WTN = WT / 70.0
  }
  CL = TVCL * WTN ^ TH_WT * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        assert!(
            model.covariate_mu_refs.is_empty(),
            "premise: the conditional local defeats #619 detection, so TH_WT is un-anchored"
        );
        let (with_eta, without_eta) = partition_by_eta(
            &model,
            non_fixed_unanchored_thetas(&model, &model.default_params.theta_fixed, &[], &[]),
        );
        assert_eq!(with_eta, ["TH_WT"]);
        assert!(without_eta.is_empty(), "{without_eta:?}");
    }

    #[test]
    fn no_flag_when_all_thetas_have_eta() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.0)
  theta TVV(10.0, 0.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let fixed = &model.default_params.theta_fixed;
        let flagged = non_fixed_unanchored_thetas(&model, fixed, &[], &[]);
        assert!(flagged.is_empty(), "expected no flags, got {flagged:?}");
    }

    /// The #619 anchor set comes from the caller's **resolved** group list, not
    /// from `model.covariate_mu_refs` (#918 review). A group the resolver drops
    /// — weak IIV, a conflicting single-anchor pair, an all-FIXed theta list —
    /// moves nothing, so its thetas really are M-step-only and must stay in the
    /// advisory. Both arms are asserted on one model, so the difference is the
    /// argument and nothing else.
    #[test]
    fn mu_ref_anchor_mask_reads_the_resolved_groups_not_the_parsed_ones() {
        let src = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        assert_eq!(
            model.covariate_mu_refs.len(),
            1,
            "premise: the parser found a group, so reading it off the model would differ"
        );
        let names = &model.theta_names;
        let tvcl = names.iter().position(|n| n == "TVCL").unwrap();
        let th = names.iter().position(|n| n == "TH_CRCL").unwrap();

        // No group resolved: neither group theta is an anchor.
        let dropped = theta_is_mu_ref_anchor_mask(&model, &[], &[]);
        assert!(!dropped[tvcl] && !dropped[th], "got {dropped:?}");
        let flagged =
            non_fixed_unanchored_thetas(&model, &model.default_params.theta_fixed, &[], &[]);
        assert!(
            flagged.contains(&"TVCL".to_string()) && flagged.contains(&"TH_CRCL".to_string()),
            "a dropped group leaves its thetas M-step-only: {flagged:?}"
        );

        // Group resolved: both are anchors and neither is flagged.
        let live = theta_is_mu_ref_anchor_mask(&model, &[tvcl, th], &[]);
        assert!(live[tvcl] && live[th], "got {live:?}");
        let flagged = non_fixed_unanchored_thetas(
            &model,
            &model.default_params.theta_fixed,
            &[tvcl, th],
            &[],
        );
        assert!(flagged.is_empty(), "expected no flags, got {flagged:?}");
    }

    #[test]
    fn shift_draws_moves_samples_and_mean_and_keeps_the_covariance() {
        // #1365: after the closed-form mu-ref step adds δ to the packed θ, the
        // stored draws must move by −δ on that η — every sample as well as the
        // moments — and leave the weights and the weighted covariance S − m mᵀ
        // untouched, including its cross terms with an η that did not move.
        let etas = vec![vec![0.1, -0.5], vec![0.7, 0.3], vec![0.4, -0.4]];
        let weights = vec![0.25, 0.25, 0.5];
        let mut mean = vec![0.0; 2];
        let mut s2 = DMatrix::<f64>::zeros(2, 2);
        for (eta, &w) in etas.iter().zip(&weights) {
            let v = DVector::from_column_slice(eta);
            for (a, &x) in mean.iter_mut().zip(eta) {
                *a += w * x;
            }
            s2 += w * &v * v.transpose();
        }
        let m = DVector::from_column_slice(&mean);
        let cov = &s2 - &m * m.transpose();
        let mut d = SubjectDraws::cancelled(2);
        d.etas = etas.clone();
        d.weights = weights.clone();
        d.mean = mean.clone();
        d.second_moment = s2;

        shift_draws(&mut d, &[0.35, 0.0]);

        // The samples moved with the moments: the shifted mean is still the
        // weighted mean of the shifted samples.
        for (k, (got, was)) in d.etas.iter().zip(&etas).enumerate() {
            assert!(
                (got[0] - (was[0] - 0.35)).abs() < 1e-15,
                "sample {k} η₀ = {}",
                got[0]
            );
            assert_eq!(got[1], was[1], "sample {k}: an unshifted η keeps its value");
        }
        assert_eq!(d.weights, weights, "weights are unchanged");
        let wmean: f64 = d.etas.iter().zip(&d.weights).map(|(e, w)| w * e[0]).sum();
        assert!(
            (wmean - d.mean[0]).abs() < 1e-15,
            "{wmean} vs {}",
            d.mean[0]
        );
        assert!(
            (d.mean[0] - (mean[0] - 0.35)).abs() < 1e-15,
            "mean[0] = {}",
            d.mean[0]
        );
        assert_eq!(d.mean[1], mean[1], "an unshifted η keeps its mean");
        let m_new = DVector::from_column_slice(&d.mean);
        let cov_new = &d.second_moment - &m_new * m_new.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cov_new[(i, j)] - cov[(i, j)]).abs() < 1e-15,
                    "covariance must be preserved at ({i},{j}): {} vs {}",
                    cov_new[(i, j)],
                    cov[(i, j)]
                );
            }
        }
    }

    #[test]
    fn covariance_to_proposal_hessian_inverts_an_in_bounds_covariance() {
        // A covariance comfortably inside [floor, Ωii] passes through unclamped,
        // so the returned precision is its exact inverse.
        let cov = DMatrix::from_row_slice(2, 2, &[0.25, 0.05, 0.05, 0.16]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[10.0, 10.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        let recovered = h.clone().try_inverse().expect("h must be invertible");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (recovered[(i, j)] - cov[(i, j)]).abs() < 1e-9,
                    "inverse-of-inverse must recover cov at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn covariance_to_proposal_hessian_floors_collapsed_diagonal() {
        // A zero-variance dimension (collapsed ESS) is floored to a tiny absolute
        // value rather than inverting to a near-delta proposal.
        let cov = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.2]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "floored result must be finite"
        );
        let expected = 1.0 / IMP_PROPOSAL_COV_FLOOR;
        assert!(
            (h[(0, 0)] - expected).abs() / expected < 1e-9,
            "floored precision should be ~1/floor, got {}",
            h[(0, 0)]
        );
    }

    /// Correlation between dimensions `i` and `j` of a covariance matrix.
    fn corr(c: &DMatrix<f64>, i: usize, j: usize) -> f64 {
        c[(i, j)] / (c[(i, i)] * c[(j, j)]).sqrt()
    }

    #[test]
    fn covariance_to_proposal_hessian_keeps_a_capped_correlated_covariance_pd() {
        // #2650. Both variances sit just above Ωii and the correlation is about
        // −0.98, so capping the diagonal alone gives
        // 0.034·0.031 − 0.0335² < 0: indefinite, a failed Cholesky, a zero matrix.
        let cov = DMatrix::from_row_slice(2, 2, &[0.0362, -0.0335, -0.0335, 0.0322]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[0.034, 0.031]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        let capped = h
            .try_inverse()
            .expect("capped covariance must stay invertible");
        assert!((capped[(0, 0)] - 0.034).abs() < 1e-12);
        assert!((capped[(1, 1)] - 0.031).abs() < 1e-12);
        assert!(
            (corr(&capped, 0, 1) - corr(&cov, 0, 1)).abs() < 1e-9,
            "the cap must keep the correlation: {} vs {}",
            corr(&capped, 0, 1),
            corr(&cov, 0, 1)
        );
    }

    #[test]
    fn covariance_to_proposal_hessian_rescales_only_what_the_cap_touched() {
        // Dimension 0 is capped, 1 and 2 are not. Every correlation survives, and
        // the covariance between the two uncapped dimensions is left exactly alone,
        // so a scale applied to all off-diagonals, or squared, cannot pass.
        let cov = DMatrix::from_row_slice(3, 3, &[4.0, 0.3, 0.2, 0.3, 0.25, 0.05, 0.2, 0.05, 0.16]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 10.0, 10.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        let capped = h.try_inverse().expect("must stay invertible");
        assert!((capped[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((capped[(1, 2)] - 0.05).abs() < 1e-9);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                (corr(&capped, i, j) - corr(&cov, i, j)).abs() < 1e-9,
                "correlation ({i},{j}) must survive the cap"
            );
        }
    }

    #[test]
    fn covariance_to_proposal_hessian_drops_covariances_of_a_non_finite_variance() {
        // A NaN variance usually arrives with NaN covariances. They must not reach
        // the Cholesky, or a dimension that could be capped costs the whole matrix.
        let cov = DMatrix::from_row_slice(2, 2, &[f64::NAN, f64::NAN, f64::NAN, 0.2]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[0.5, 1.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!((h[(0, 0)] - 2.0).abs() < 1e-9, "got {}", h[(0, 0)]);
        assert!((h[(1, 1)] - 5.0).abs() < 1e-9, "got {}", h[(1, 1)]);
        assert_eq!(h[(0, 1)], 0.0);
    }

    #[test]
    fn covariance_to_proposal_hessian_returns_the_prior_precision_when_indefinite() {
        // An in-bounds but indefinite covariance gets past the cap and fails the
        // Cholesky. The answer is Ω⁻¹ itself, not a zero matrix for `build_proposal`
        // to jitter into a proposal of SD 1e5 (#2650).
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let omega = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 4.0]);
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        let want = omega.clone().try_inverse().expect("Ω is invertible");
        assert!(
            (&h - &want).abs().max() < 1e-12,
            "got {h}, want Ω⁻¹ = {want}"
        );
    }

    #[test]
    fn covariance_to_proposal_hessian_caps_exploding_diagonal() {
        // A heavy-tailed-outlier-inflated covariance is capped at Ωii so the
        // proposal can't drift broader than the prior.
        let cov = DMatrix::from_row_slice(2, 2, &[1e14, 0.0, 0.0, 1e12]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[0.2, 0.3]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!((h[(0, 0)] - 1.0 / 0.2).abs() / (1.0 / 0.2) < 1e-9);
        assert!((h[(1, 1)] - 1.0 / 0.3).abs() / (1.0 / 0.3) < 1e-9);
    }

    #[test]
    fn covariance_to_proposal_hessian_is_zero_only_when_omega_is_not_pd() {
        // With no usable covariance and no usable Ω there is nothing left to return.
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 5.0, 5.0, 1.0]);
        let omega = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!(
            h.iter().all(|&v| v == 0.0),
            "a non-PD Ω must yield the zero matrix, got {h}"
        );
    }

    // ── Estimating IMP/IMPMAP under a mixture (#985) ──

    const MIX_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    fn mix_pop() -> Population {
        use std::io::Write;
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        for (sid, &(cl, wt)) in [(1.0_f64, 60.0_f64), (3.0, 90.0)].iter().enumerate() {
            let id = sid + 1;
            csv.push_str(&format!("{id},0,0,100,1,1,{wt}\n"));
            for t in [0.5_f64, 1.0, 2.0, 4.0] {
                let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                csv.push_str(&format!("{id},{t},{c:.5},0,0,1,{wt}\n"));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// A short estimating IMPMAP (Map recenter) run on a mixture returns Ok,
    /// carries the mixture through, and populates per-subject MIXEST + posteriors.
    #[test]
    fn impmap_mixture_short_run_populates_posteriors() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 2;
        opts.impmap_samples = 200;
        opts.impmap_seed = Some(1);
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        assert!(res.ofv.is_finite());
        assert!(res.mixture_posteriors.is_some(), "mixture posteriors set");
        let mp = res.mixture_posteriors.unwrap();
        assert_eq!(mp.mixest.len(), pop.subjects.len());
        assert_eq!(mp.pmix.len(), pop.subjects.len());
    }

    /// The IMP (SampleMoments recenter) estimating path also runs on a mixture.
    #[test]
    fn imp_mixture_short_run_ok() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.imp_iterations = 2;
        opts.imp_samples = 200;
        opts.imp_seed = Some(1);
        opts.run_covariance_step = false;
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("imp mixture");
        assert!(res.ofv.is_finite());
        assert!(res.mixture_posteriors.is_some());
        // The mixture loop keeps its own objective, so it records its own baseline (#2652).
        assert!(res.mcem_lowest_objective.is_some_and(f64::is_finite));
    }

    /// Estimating IMP/IMPMAP rejects a mixture with per-class Ω/Σ overrides.
    #[test]
    fn estimating_imp_rejects_per_class_overrides() {
        const OV: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.15

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(OV).unwrap();
        let pop = mix_pop();
        let opts = FitOptions::default();
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("per-class overrides must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.contains("per-class Omega/Sigma overrides"),
            "got: {err}"
        );
    }
    /// A theta that drives both the mixing expression and a structural typical
    /// value is rejected up front: the weighted θ/σ M-step would fit it from the
    /// observation likelihood and `mstep_mixing` would then overwrite it,
    /// silently discarding the structural estimate (#992 review). SAEM rejects
    /// the same shape; the estimating IMP/IMPMAP MCEM must too.
    #[test]
    fn estimating_imp_rejects_mixing_theta_that_drives_structure() {
        const SHARED: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta MIXL(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = 10.0 * MIXL

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(SHARED).unwrap();
        let pop = mix_pop();
        let opts = FitOptions::default();
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("a mixing theta driving the structural model must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.contains("also drives the structural model") && err.contains("MIXL"),
            "got: {err}"
        );
    }

    /// The mixture MCEM delegates before `run_mcem` builds its mu-ref warnings,
    /// so it must push its own no-ETA warning. It names `TVV` (a bare typical
    /// value) but neither the mixing theta `MIXL` (which `mstep_mixing` owns)
    /// nor the class-switched `TVCL1`/`TVCL2`, which carry `ETA_CL` in every
    /// class arm and are class-aware mu-referenced (#996).
    #[test]
    fn mixture_mcem_warns_about_weighted_mstep_only_thetas() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 1;
        opts.impmap_samples = 50;
        opts.impmap_seed = Some(1);
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        let joined = res.warnings.join("\n");
        assert!(
            joined.contains("have NO associated ETA") && joined.contains("TVV"),
            "no-ETA warning missing: {joined}"
        );
        assert!(
            !joined.contains("MIXL"),
            "the mixing theta must not be flagged as a missing-ETA parameter: {joined}"
        );
        assert!(
            !joined.contains("TVCL1") && !joined.contains("TVCL2"),
            "class-aware mu-referenced thetas must not be flagged as ETA-less: {joined}"
        );
        assert!(
            !joined.contains("no class-aware mu-referencing was applied"),
            "the class-aware shift is available for this model: {joined}"
        );
    }

    /// `impmap_trace` and `impmap_auto` are not implemented for mixtures; they
    /// must warn rather than be silently ignored (#992 review).
    #[test]
    fn mixture_mcem_warns_that_trace_and_auto_are_ignored() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 1;
        opts.impmap_samples = 50;
        opts.impmap_seed = Some(1);
        opts.impmap_trace = true;
        opts.impmap_auto = true;
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        let joined = res.warnings.join("\n");
        assert!(
            joined.contains("trace collection") && joined.contains("adaptive sampling"),
            "trace/auto no-op warnings missing: {joined}"
        );
        assert!(res.impmap_trace.is_none(), "no trace is produced");
    }

    /// A cancel set before the run returns `Err` instead of falling through to
    /// the K-class marginal pass and the covariance step — the most expensive
    /// part of the run — with parameters from a truncated MCEM (#992 review).
    #[test]
    fn mixture_mcem_cancel_returns_err() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 5;
        opts.impmap_samples = 200;
        opts.impmap_seed = Some(1);
        let flag = crate::cancel::CancelFlag::new();
        flag.cancel();
        opts.cancel = Some(flag);
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("a cancelled mixture MCEM must not return Ok"),
            Err(e) => e,
        };
        assert!(err.contains("cancelled by user"), "got: {err}");
    }

    // ── Stale warm-start basin under MCEM (#1629) ──

    /// `examples/two_cpt_oral_cov.ferx` with V1's own weight exponent, at the
    /// FOCEI estimates on `data/two_cpt_oral_cov.csv`.
    const TWO_CPT_COV_SEP: &str = r"
[parameters]
  theta TVCL(4.943001, 0.1, 100.0)
  theta TVV1(48.562858, 1.0, 500.0)
  theta TVQ(9.575645, 0.1, 100.0)
  theta TVV2(94.166322, 1.0, 500.0)
  theta TVKA(1.256269, 0.01, 10.0)
  theta THETA_WT(0.729548, 0.01, 5.0)
  theta THETA_CRCL(0.555389, 0.01, 5.0)
  theta THETA_WT_V1(0.577427, 0.01, 5.0)
  omega ETA_CL ~ 0.107337
  omega ETA_V1 ~ 0.106107
  omega ETA_Q  ~ 0.051339
  omega ETA_V2 ~ 0.037946
  omega ETA_KA ~ 0.174963
  sigma PROP_ERR ~ 0.019619 (sd)

[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * (WT / 70)^THETA_WT_V1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

    /// Subject 6 of `data/two_cpt_oral_cov.csv`, verbatim.
    fn subject6_pop() -> Population {
        use std::io::Write;
        let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT,CRCL
6,0,.,1,250,1,0,1,71.2,129.5
6,0.5,2.6485,0,.,1,0,0,71.2,129.5
6,1,3.3507,0,.,1,0,0,71.2,129.5
6,2,3.2617,0,.,1,0,0,71.2,129.5
6,4,1.9015,0,.,1,0,0,71.2,129.5
6,6,1.1540,0,.,1,0,0,71.2,129.5
6,8,0.7876,0,.,1,0,0,71.2,129.5
6,12,0.4953,0,.,1,0,0,71.2,129.5
6,24,0.3088,0,.,1,0,0,71.2,129.5
6,36,0.2083,0,.,1,0,0,71.2,129.5
6,48,0.1423,0,.,1,0,0,71.2,129.5
";
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::read_nonmem_csv(f.path(), Some(&["WT", "CRCL"]), None).unwrap()
    }

    /// The MCEM MAP step must not inherit a stale basin from its warm start
    /// (#1629). Subject 6's iteration-1 mode under the default initial estimates
    /// is the low-V1/low-KA basin; warm-starting every later iteration from it
    /// held the subject there for a whole 200-iteration IMPMAP fit, ~8 NLL
    /// units above its real mode, which read as a ~17-unit −2 log L gap and an
    /// inflated ω²_V1 / ω²_KA. Warm-started from that basin at the FOCEI
    /// estimates, `find_ebe_multistart` must return the cold-start mode.
    ///
    /// The first two assertions are the straddle: the warm start alone really is
    /// trapped and the cold start really is not, so the third can only pass
    /// through the cold re-solve (deleting it fails it). Measured: warm-only
    /// NLL −30.886 (η_V1 −1.34, η_KA −1.12), cold −44.194 (η_V1 −0.07, η_KA
    /// +0.17) — 13.3 apart, against the 5.0 straddle bound.
    #[test]
    fn find_ebe_multistart_escapes_stale_warm_basin() {
        let model = crate::parser::model_parser::parse_model_string(TWO_CPT_COV_SEP).unwrap();
        let pop = subject6_pop();
        let params = &model.default_params;
        let subject = &pop.subjects[0];
        // Subject 6's η̂ held over 200 IMPMAP iterations before the fix.
        let stale = [0.293, -1.304, 0.024, 0.323, -1.076];

        let warm = find_ebe(&model, subject, params, 200, 1e-6, Some(&stale), None, 0);
        let cold = find_ebe(&model, subject, params, 200, 1e-6, None, None, 0);
        assert!(
            warm.nll.is_finite() && cold.nll.is_finite(),
            "both solves must be finite: warm {} cold {}",
            warm.nll,
            cold.nll
        );
        assert!(
            warm.nll > cold.nll + 5.0,
            "fixture no longer straddles the trap: the warm start from the stale basin \
             (nll {}) must stay well above the cold mode (nll {})",
            warm.nll,
            cold.nll
        );

        let ms = find_ebe_multistart(
            &model,
            subject,
            params,
            200,
            1e-6,
            Some(&stale),
            None,
            0,
            0,
            None,
            7,
        );
        assert!(
            (ms.nll - cold.nll).abs() < 1e-9,
            "a warm-started MAP must keep the cold-start mode when it is lower: got nll {} \
             (warm-only {}, cold {})",
            ms.nll,
            warm.nll,
            cold.nll
        );
    }

    /// `find_ebe_multistart` degrades to the plain warm-start solve when no Ω
    /// Cholesky is supplied, and never returns a worse start than the baseline
    /// when MCETA restarts are requested.
    #[test]
    fn find_ebe_multistart_never_worse_than_baseline() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let params = &model.default_params;
        let subject = &pop.subjects[0];

        let baseline =
            find_ebe_multistart(&model, subject, params, 50, 1e-4, None, None, 0, 0, None, 7);
        let direct = find_ebe(&model, subject, params, 50, 1e-4, None, None, 0);
        assert!(
            (baseline.nll - direct.nll).abs() < 1e-12,
            "no Cholesky must reproduce find_ebe exactly: {} vs {}",
            baseline.nll,
            direct.nll
        );

        let chol = params.omega.matrix.clone().cholesky().unwrap().l();
        let multi = find_ebe_multistart(
            &model,
            subject,
            params,
            50,
            1e-4,
            None,
            None,
            0,
            4,
            Some(&chol),
            7,
        );
        assert!(
            multi.nll <= baseline.nll + 1e-12,
            "MCETA restarts must not return a worse mode: {} vs {}",
            multi.nll,
            baseline.nll
        );
    }

    /// The Sobol config guard (#1965), each boundary on its own: the
    /// coordinate count is exactly `n_eta` + (t) + (mixture), so every
    /// proposal form is admitted up to the table's 1000 coordinates and refused
    /// one beyond; and the t degrees of freedom stop at `CHI_SQ_INV_MAX_NU`.
    /// The messages name the count, the limit and the switch to flip.
    #[test]
    fn sobol_config_guard_counts_the_coordinates_each_form_uses() {
        let inf = f64::INFINITY;
        for (t, mix, n_ok) in [
            (false, false, 1000),
            (true, false, 999),
            (false, true, 999),
            (true, true, 998),
        ] {
            let nu = if t { 4.0 } else { inf };
            let tag = format!("t={t} mixture={mix}");
            assert!(
                check_sobol_config("IMPMAP", "impmap_proposal_df", n_ok, true, nu, mix).is_ok(),
                "{tag}"
            );
            let e = check_sobol_config("IMPMAP", "impmap_proposal_df", n_ok + 1, true, nu, mix)
                .unwrap_err();
            assert!(
                e.contains("1001") && e.contains("1000") && e.contains("impmap_sobol"),
                "{tag}: {e}"
            );
        }
        assert!(
            check_sobol_config("IMPMAP", "impmap_proposal_df", 5000, false, 4.0, true).is_ok(),
            "no limit without Sobol"
        );

        let nu_max = crate::stats::special::CHI_SQ_INV_MAX_NU;
        assert!(check_sobol_config("IMP", "imp_proposal_df", 3, true, nu_max, false).is_ok());
        let e =
            check_sobol_config("IMP", "imp_proposal_df", 3, true, nu_max * 2.0, false).unwrap_err();
        assert!(
            e.contains("imp_proposal_df = normal") && e.contains("2000"),
            "{e}"
        );
        assert!(check_sobol_config("IMP", "imp_proposal_df", 3, false, 1e9, false).is_ok());
    }

    /// The weighted θ/σ M-step honours a pinned coordinate (`lower == upper`),
    /// which is how the mixture MCEM keeps the mixing thetas — absent from the
    /// observation likelihood — out of the BOBYQA search (#992 review).
    #[test]
    fn weighted_mstep_mixture_honours_pinned_theta_bounds() {
        use crate::estimation::importance_sampling::SubjectDraws;
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let params = &model.default_params;
        let n_theta = params.theta.len();
        let n_sigma = params.sigma.values.len();
        let n_eta = model.n_eta;

        // One (degenerate, single-sample) draw set per class per subject.
        let draws_by_class: Vec<Vec<SubjectDraws>> = (0..2)
            .map(|_| {
                pop.subjects
                    .iter()
                    .map(|_| {
                        let mut d = SubjectDraws::cancelled(n_eta);
                        d.etas = vec![vec![0.0; n_eta]];
                        d.weights = vec![1.0];
                        d
                    })
                    .collect()
            })
            .collect();
        let pmix: Vec<Vec<f64>> = pop.subjects.iter().map(|_| vec![0.5, 0.5]).collect();

        let log_theta: Vec<f64> = params.theta.iter().map(|&t| t.max(1e-10).ln()).collect();
        let log_sigma: Vec<f64> = params
            .sigma
            .values
            .iter()
            .map(|&s| s.max(1e-10).ln())
            .collect();
        let mask = vec![true; n_theta];
        let mut lower = vec![-10.0; n_theta];
        let mut upper = vec![10.0; n_theta];
        // Pin theta 0 exactly, as the mixing-theta pin does.
        lower[0] = log_theta[0];
        upper[0] = log_theta[0];

        let out_theta = theta_sigma_weighted_mstep_mixture(
            &model,
            &pop,
            &draws_by_class,
            &vec![None; pop.subjects.len()],
            &pmix,
            &log_theta,
            &log_sigma,
            &lower,
            &upper,
            &vec![-8.0; n_sigma],
            &vec![5.0; n_sigma],
            n_theta,
            n_sigma,
            8,
            &mask,
        )
        .log_theta;
        assert!(
            (out_theta[0] - log_theta[0]).abs() < 1e-12,
            "pinned theta must not move: {} vs {}",
            out_theta[0],
            log_theta[0]
        );
    }

    // ── Inter-occasion variability (#2301) ──

    fn iov_model() -> CompiledModel {
        crate::parser::model_parser::parse_full_model(include_str!(
            "../../examples/warfarin_iov.ferx"
        ))
        .expect("warfarin_iov parses")
        .model
    }

    fn iov_pop() -> Population {
        crate::read_nonmem_csv(
            std::path::Path::new("data/warfarin_iov.csv"),
            None,
            Some("OCC"),
        )
        .expect("warfarin_iov.csv reads")
    }

    /// A draw set carrying only the given stacked second moment (the Ω_iov
    /// M-step reads nothing else).
    fn draws_with_second_moment(m: DMatrix<f64>) -> SubjectDraws {
        let mut d = SubjectDraws::cancelled(m.nrows());
        d.second_moment = m;
        d
    }

    /// A symmetric `n × n` matrix with distinct entries, so a block read from
    /// the wrong offset cannot coincide with the right one.
    fn distinct_spd(n: usize, seed: f64) -> DMatrix<f64> {
        DMatrix::from_fn(n, n, |i, j| {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            if i == j {
                1.0 + seed + 0.1 * a
            } else {
                0.01 * (seed + a + 3.0 * b)
            }
        })
    }

    /// `Ω_iov = Σᵢ Σ_g E_w[κκᵀ] / Σᵢ Kᵢ`: pooled over every occasion block of
    /// every subject (here 2 + 1 occasions; the η-only subject contributes
    /// none), read from the κ blocks — not the η block or an off-diagonal
    /// η/κ block — and divided by the occasion count, not the subject count.
    #[test]
    fn omega_iov_mstep_pools_kappa_blocks_over_every_occasion() {
        let (n_eta, n_kappa) = (1usize, 2usize);
        let a = distinct_spd(n_eta + 2 * n_kappa, 0.0); // 2 occasions
        let b = distinct_spd(n_eta + n_kappa, 5.0); // 1 occasion
        let c = distinct_spd(n_eta, 9.0); // no occasion: η only
        let draws = vec![
            draws_with_second_moment(a.clone()),
            draws_with_second_moment(b.clone()),
            draws_with_second_moment(c),
        ];
        let n_occ_of = [2usize, 1, 0];
        let want = (a.view((1, 1), (2, 2)).into_owned()
            + a.view((3, 3), (2, 2)).into_owned()
            + b.view((1, 1), (2, 2)).into_owned())
            / 3.0;

        let names = vec!["K1".to_string(), "K2".to_string()];
        let full = OmegaMatrix::from_matrix_with_mask(
            DMatrix::from_diagonal_element(2, 2, 0.04),
            names.clone(),
            false,
            DMatrix::from_element(2, 2, true),
        );
        let got = omega_iov_mstep(&draws, &n_occ_of, n_eta, &full, &[false, false])
            .expect("three occasions to estimate from");
        assert!((&got - &want).abs().max() < 1e-15, "got {got}, want {want}");

        // A diagonal template zeroes the off-diagonal; a FIXed κ keeps its
        // declared row and column.
        let diag = OmegaMatrix::from_matrix_with_mask(
            DMatrix::from_diagonal_element(2, 2, 0.04),
            names,
            true,
            DMatrix::from_fn(2, 2, |i, j| i == j),
        );
        let got = omega_iov_mstep(&draws, &n_occ_of, n_eta, &diag, &[false, true]).unwrap();
        assert_eq!(got[(0, 1)], 0.0);
        assert_eq!(got[(1, 0)], 0.0);
        assert_eq!(got[(1, 1)], 0.04, "FIXed κ keeps its declared variance");
        assert!((got[(0, 0)] - want[(0, 0)]).abs() < 1e-15);

        // A free diagonal that collapses is floored, so Ω_iov stays PD.
        let tiny = vec![draws_with_second_moment(DMatrix::from_diagonal_element(
            n_eta + n_kappa,
            n_eta + n_kappa,
            1e-12,
        ))];
        let got = omega_iov_mstep(&tiny, &[1], n_eta, &full, &[false, false]).unwrap();
        assert_eq!(got[(0, 0)], OMEGA_DIAG_FLOOR);

        // No occasion anywhere: nothing to estimate from.
        let none = vec![draws_with_second_moment(DMatrix::identity(n_eta, n_eta))];
        assert!(omega_iov_mstep(&none, &[0], n_eta, &full, &[false, false]).is_none());
    }

    /// An IOV subject's stacked sample is scored with per-occasion predictions:
    /// at κ = 0 that is the η-only likelihood, and a non-zero κ moves it. An
    /// η-only sample is scored exactly as before #2301. Either half alone is
    /// satisfied by a scorer that ignores κ.
    #[test]
    fn draw_obs_nll_scores_kappa_on_iov_samples_only() {
        let model = iov_model();
        let pop = iov_pop();
        let subject = &pop.subjects[0];
        let n_occ = crate::stats::stacked_layout::group_count(subject);
        assert_eq!(n_occ, 2, "fixture subject has two occasions");
        let theta = &model.default_params.theta;
        let sigma = &model.default_params.sigma.values;
        let mut scratch = EventPkParams::default();
        let eta = [0.1, -0.05, 0.2];

        let plain = obs_nll_subject_into_with_schedule(
            &model,
            subject,
            theta,
            sigma,
            model.residual_cov_theta(theta),
            &eta,
            &mut scratch,
            None,
        );
        let eta_only = draw_obs_nll(&model, subject, theta, sigma, &eta, None, &mut scratch);
        assert_eq!(eta_only, plain, "an η-only sample is scored unchanged");

        let mut b = eta.to_vec();
        b.extend([0.0, 0.0]);
        let at_zero = draw_obs_nll(&model, subject, theta, sigma, &b, None, &mut scratch);
        assert!(
            (at_zero - plain).abs() < 1e-9 * plain.abs().max(1.0),
            "κ = 0 must reproduce the η-only NLL: {at_zero} vs {plain}"
        );

        b[eta.len() + 1] = 0.5; // κ on the second occasion only
        let moved = draw_obs_nll(&model, subject, theta, sigma, &b, None, &mut scratch);
        assert!(
            (moved - at_zero).abs() > 1e-3,
            "a non-zero κ must change the score: {moved} vs {at_zero}"
        );
    }

    /// The θ/σ M-step objective scores each IOV sample with its own κ: the same
    /// draws with every κ zeroed must give a different value, while the η part
    /// is held fixed. A M-step that dropped κ (scoring `b[..n_eta]`) would
    /// return the zeroed value for both.
    #[test]
    fn weighted_mstep_objective_reads_each_samples_kappa() {
        let model = iov_model();
        let pop = iov_pop();
        let n_eta = model.n_eta;
        let theta = &model.default_params.theta;
        let sigma = &model.default_params.sigma.values;
        let draws_with = |kappa: f64| -> Vec<SubjectDraws> {
            pop.subjects
                .iter()
                .map(|s| {
                    let n_occ = crate::stats::stacked_layout::group_count(s);
                    let b = |sign: f64| {
                        let mut v = vec![0.05 * sign; n_eta];
                        v.extend((0..n_occ).map(|g| sign * kappa * (g as f64 + 1.0)));
                        v
                    };
                    let mut d = SubjectDraws::cancelled(n_eta);
                    d.etas = vec![b(1.0), b(-1.0)];
                    d.weights = vec![0.4, 0.6];
                    d
                })
                .collect()
        };
        let no_schedules = vec![None; pop.subjects.len()];
        let live =
            weighted_mstep_objective(&model, &pop, &draws_with(0.3), &no_schedules, theta, sigma);
        let zero =
            weighted_mstep_objective(&model, &pop, &draws_with(0.0), &no_schedules, theta, sigma);
        assert!(
            live.is_finite() && live < 1e19,
            "objective must be real: {live}"
        );
        assert!(
            zero.is_finite() && zero < 1e19,
            "objective must be real: {zero}"
        );
        assert!(
            (live - zero).abs() > 1e-2,
            "κ must enter the M-step objective: {live} vs {zero} at κ = 0"
        );
    }

    /// End to end on warfarin IOV: IMPMAP and IMP both return Ω_iov, move it
    /// off its start, and report per-occasion κ̂ for every subject.
    #[test]
    fn imp_and_impmap_estimate_an_iov_model() {
        let model = iov_model();
        let pop = iov_pop();
        let params = model.default_params.clone();
        let start = params.omega_iov.as_ref().expect("IOV model").matrix[(0, 0)];

        let mut opts = FitOptions::default();
        opts.impmap_iterations = 3;
        opts.impmap_samples = 100;
        opts.impmap_seed = Some(3);
        opts.imp_iterations = 3;
        opts.imp_samples = 100;
        opts.imp_seed = Some(3);
        opts.run_covariance_step = false;

        for (name, res) in [
            ("impmap", run_impmap(&model, &pop, &params, None, &opts)),
            ("imp", run_imp(&model, &pop, &params, None, &opts)),
        ] {
            let res = res.unwrap_or_else(|e| panic!("{name} on an IOV model: {e}"));
            assert!(res.ofv.is_finite(), "{name}: ofv {}", res.ofv);
            let iov = res
                .params
                .omega_iov
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: Ω_iov missing from the result"));
            let est = iov.matrix[(0, 0)];
            assert!(est.is_finite() && est > 0.0, "{name}: Ω_iov = {est}");
            assert!(
                (est - start).abs() > 1e-4,
                "{name}: Ω_iov never left its start ({start} → {est})"
            );
            assert_eq!(res.kappas.len(), pop.subjects.len());
            for (i, k) in res.kappas.iter().enumerate() {
                assert_eq!(k.len(), 2, "{name}: subject {i} κ̂ blocks");
            }
        }
    }

    /// #2301's degenerate oracle: with Ω_iov FIXed at ~0 the κ draws carry no
    /// variance, so an IOV fit must reproduce the same model without `[iov]`.
    /// Unlike the κ = 0 scorer check, this runs the joint proposal, the stacked
    /// moments and the η-block slice of the Ω M-step.
    #[test]
    fn iov_with_vanishing_omega_iov_matches_the_eta_only_fit() {
        let src = include_str!("../../examples/warfarin_iov.ferx");
        let parse = |s: &str| {
            crate::parser::model_parser::parse_full_model(s)
                .expect("parses")
                .model
        };
        let degenerate = parse(&src.replace("kappa KAPPA_CL ~ 0.01", "kappa KAPPA_CL ~ 1e-8 FIX"));
        let eta_only = parse(
            &src.replace("  kappa KAPPA_CL ~ 0.01\n", "")
                .replace(" + KAPPA_CL", ""),
        );
        assert_eq!(degenerate.n_kappa, 1);
        assert_eq!(eta_only.n_kappa, 0);
        let pop = iov_pop();

        let mut opts = FitOptions::default();
        opts.impmap_iterations = 5;
        opts.impmap_samples = 300;
        opts.impmap_seed = Some(7);
        opts.imp_iterations = 5;
        opts.imp_samples = 300;
        opts.imp_seed = Some(7);
        opts.run_covariance_step = false;

        type Runner = fn(
            &CompiledModel,
            &Population,
            &ModelParameters,
            Option<&[DVector<f64>]>,
            &FitOptions,
        ) -> Result<OuterResult, String>;
        for (name, run) in [("impmap", run_impmap as Runner), ("imp", run_imp as Runner)] {
            let a = run(&degenerate, &pop, &degenerate.default_params, None, &opts)
                .unwrap_or_else(|e| panic!("{name} degenerate IOV: {e}"));
            let b = run(&eta_only, &pop, &eta_only.default_params, None, &opts)
                .unwrap_or_else(|e| panic!("{name} η-only: {e}"));
            let mut worst = 0.0f64;
            let rel = |x: f64, y: f64| (x - y).abs() / y.abs().max(1e-12);
            for (x, y) in a.params.theta.iter().zip(&b.params.theta) {
                assert!(x.is_finite() && y.is_finite());
                worst = worst.max(rel(*x, *y));
            }
            for i in 0..eta_only.n_eta {
                let (x, y) = (a.params.omega.matrix[(i, i)], b.params.omega.matrix[(i, i)]);
                assert!(x.is_finite() && y.is_finite());
                worst = worst.max(rel(x, y));
            }
            for (x, y) in a.params.sigma.values.iter().zip(&b.params.sigma.values) {
                assert!(x.is_finite() && y.is_finite());
                worst = worst.max(rel(*x, *y));
            }
            eprintln!(
                "{name}: worst relative gap {worst:.3e}; θ {:?} vs {:?}; Ω {:?} vs {:?}",
                a.params.theta,
                b.params.theta,
                a.params.omega.matrix.diagonal().as_slice(),
                b.params.omega.matrix.diagonal().as_slice(),
            );
            // Measured (macOS arm64): worst 1.6e-2 (impmap) and 2.6e-2 (imp), the
            // Monte-Carlo gap between two different random streams (the stacked
            // draw is wider). 0.1 is ~4× the larger.
            assert!(worst < 0.1, "{name}: worst relative gap {worst:.3e}");
        }
    }

    /// `run_importance_sampling`'s joint (η, κ) estimate at `params`, with 2000
    /// samples around the EBEs found there — the separate estimator the IMPMAP
    /// trace is checked against below.
    fn joint_is_at(
        model: &CompiledModel,
        pop: &Population,
        params: &ModelParameters,
    ) -> ImportanceSamplingResult {
        // Adapting routes through the E-step's own draw code; this check needs
        // `subject_is_estimate_joint`, the separate estimator.
        joint_is_with(model, pop, params, 0, 0.0)
    }

    fn joint_is_with(
        model: &CompiledModel,
        pop: &Population,
        params: &ModelParameters,
        passes: usize,
        alpha: f64,
    ) -> ImportanceSamplingResult {
        let mut opts = FitOptions::default();
        opts.imp_samples = 2000;
        opts.imp_seed = Some(5);
        opts.imp_adapt_passes = passes;
        opts.imp_defensive_alpha = alpha;
        let mu_k = compute_mu_k(model, &params.theta, opts.mu_referencing);
        let (eta_hats, h_matrices, kappas, _) = run_map_multistart(
            model,
            pop,
            params,
            opts.inner_maxiter,
            opts.inner_tol,
            None,
            &mu_k,
            0,
            11,
            1,
        );
        crate::estimation::importance_sampling::run_importance_sampling(
            model,
            pop,
            params,
            &eta_hats,
            &h_matrices,
            &kappas,
            &opts,
        )
        .expect("joint IS on an IOV model")
    }

    /// `run_importance_sampling` at `params` with 2000 samples around the EBEs
    /// found there, after `tweak` has adjusted the evaluator's options.
    fn is_at_with(
        model: &CompiledModel,
        pop: &Population,
        params: &ModelParameters,
        tweak: impl Fn(&mut FitOptions),
    ) -> ImportanceSamplingResult {
        let mut opts = FitOptions::default();
        opts.imp_samples = 2000;
        opts.imp_seed = Some(5);
        // The single-proposal evaluator: what the kernels under test run with no
        // adaptive pilots (#2653) in front of them.
        opts.imp_adapt_passes = 0;
        tweak(&mut opts);
        let mu_k = compute_mu_k(model, &params.theta, opts.mu_referencing);
        let (eta_hats, h_matrices, kappas, _) = run_map_multistart(
            model,
            pop,
            params,
            opts.inner_maxiter,
            opts.inner_tol,
            None,
            &mu_k,
            0,
            11,
            1,
        );
        crate::estimation::importance_sampling::run_importance_sampling(
            model,
            pop,
            params,
            &eta_hats,
            &h_matrices,
            &kappas,
            &opts,
        )
        .expect("IS evaluation")
    }

    /// warfarin_iov with its κ removed: the same data, scored by the η-only
    /// evaluator kernel.
    fn eta_only_model() -> CompiledModel {
        let src = include_str!("../../examples/warfarin_iov.ferx")
            .replace("  kappa KAPPA_CL ~ 0.01\n", "")
            .replace(" + KAPPA_CL", "");
        let model = crate::parser::model_parser::parse_full_model(&src)
            .expect("parses")
            .model;
        assert_eq!(model.n_kappa, 0, "the κ edit must take");
        model
    }

    /// `imp_eval_only` with `imp_proposal_df = normal` (parsed to `f64::INFINITY`)
    /// is a documented setting. The evaluator once carried its own Student-t-only
    /// sampler, whose density constant at ν = ∞ is `lnΓ(∞) − lnΓ(∞) = NaN`, so the
    /// readout was not a number. Both kernels are covered — η-only, and the joint
    /// (η, κ) one an IOV subject takes — each against its own t₅ readout, which
    /// estimates the same marginal from a different proposal.
    #[test]
    fn imp_eval_with_a_normal_proposal_agrees_with_the_student_t_readout() {
        let pop = iov_pop();
        for (name, model) in [("eta-only", eta_only_model()), ("iov", iov_model())] {
            let params = model.default_params.clone();
            let t5 = is_at_with(&model, &pop, &params, |o| o.imp_proposal_df = 5.0);
            let mvn = is_at_with(&model, &pop, &params, |o| o.imp_proposal_df = f64::INFINITY);
            assert!(
                mvn.minus2_log_likelihood.is_finite() && mvn.mc_standard_error.is_finite(),
                "{name}: normal-proposal readout {} ± {}",
                mvn.minus2_log_likelihood,
                mvn.mc_standard_error
            );
            assert!(t5.minus2_log_likelihood.is_finite(), "{name}: t5 readout");
            let gap = (mvn.minus2_log_likelihood - t5.minus2_log_likelihood).abs();
            eprintln!(
                "{name}: normal {} ± {} (ess_min {}), t5 {} ± {}; gap {gap}",
                mvn.minus2_log_likelihood,
                mvn.mc_standard_error,
                mvn.ess_min,
                t5.minus2_log_likelihood,
                t5.mc_standard_error
            );
            // A ν = ∞ quietly mapped to a finite df would make the "normal" readout
            // the t₅ one: gap 0, inside the bound. The normal branch draws no χ², so
            // the two streams always differ.
            assert_ne!(
                mvn.minus2_log_likelihood.to_bits(),
                t5.minus2_log_likelihood.to_bits(),
                "{name}: the normal readout is the t5 one — ν = ∞ never reached the sampler"
            );
            assert!(mvn.ess_min > 0.0, "{name}: ess_min {}", mvn.ess_min);
            assert!(gap < NORMAL_VS_T5_GAP_BOUND, "{name}: gap {gap}");
        }
    }

    /// Normal vs t₅ readouts of one marginal, at warfarin_iov's start values.
    /// Measured (Linux x86_64, K = 2000): 0.319 (η-only) and 0.495 (IOV), 3–4
    /// combined MC SEs: at the start values the light-tailed normal proposal
    /// under-covers the posterior, which biases its readout up and makes its
    /// ESS-based SE optimistic (at SAEM's estimates the CLI gap is 0.01–0.05). 1.0
    /// is 2× the worst; the pre-fix normal readout was not finite at all.
    const NORMAL_VS_T5_GAP_BOUND: f64 = 1.0;

    /// α = 0.2 vs α = 0 on warfarin_iov at the start values. Measured (Linux
    /// x86_64, K = 2000): 0.059, against MC SEs of 0.10 / 0.13. 0.15 is 2.5× that,
    /// and still well below a mis-scored mixture (the `assert_ne` alone passes on
    /// the changed RNG stream, so this bound is what checks the mixture density).
    const ALPHA_GAP_BOUND: f64 = 0.15;

    /// `imp_defensive_alpha` reaches the joint (η, κ) evaluator. The IOV kernel
    /// once had its own sampler that took no mixture, so α changed nothing for an
    /// IOV subject. α > 0 must move the readout (a different proposal) and stay
    /// within Monte-Carlo error of α = 0 (the same marginal).
    #[test]
    fn imp_eval_defensive_mixture_reaches_iov_subjects() {
        let model = iov_model();
        let pop = iov_pop();
        let params = model.default_params.clone();
        let plain = is_at_with(&model, &pop, &params, |_| {});
        let mixed = is_at_with(&model, &pop, &params, |o| o.imp_defensive_alpha = 0.2);
        let gap = (mixed.minus2_log_likelihood - plain.minus2_log_likelihood).abs();
        eprintln!(
            "α=0 {} ± {}, α=0.2 {} ± {}; gap {gap}",
            plain.minus2_log_likelihood,
            plain.mc_standard_error,
            mixed.minus2_log_likelihood,
            mixed.mc_standard_error
        );
        assert_ne!(
            mixed.minus2_log_likelihood.to_bits(),
            plain.minus2_log_likelihood.to_bits(),
            "α = 0.2 left the IOV readout unchanged"
        );
        assert!(gap < ALPHA_GAP_BOUND, "gap {gap}");
    }

    /// The adaptive and the single-proposal evaluation of a joint (η, κ) subject
    /// draw from different proposals, so their marginals must agree within MC
    /// error. Since #2627 both share the sampler and the per-draw scorer, so this
    /// pins the adaptation's wiring, not a second likelihood implementation.
    #[test]
    fn adapted_joint_marginal_agrees_with_the_single_proposal_estimator() {
        let model = iov_model();
        let pop = iov_pop();
        let params = &model.default_params;
        crate::estimation::importance_sampling::assert_adaptive_evaluation_is_wired(
            |passes, alpha| joint_is_with(&model, &pop, params, passes, alpha),
        );
    }

    /// The trace −2LL of an IOV fit is the joint (η, κ) marginal. Iteration 1's
    /// E-step runs at the start values, so its row must agree with
    /// `run_importance_sampling` at the same parameters. The two share the sampler,
    /// the per-draw scorer and the joint prior (`JointPriors`), so what this pins
    /// is the rest of the wiring: each side's own MAP solve, proposal Hessian and
    /// (E-step) ISCALE search must land on the same marginal. The prior's
    /// `K · log det Ω_iov` term is pinned directly by
    /// `joint_priors_match_the_per_subject_construction_for_each_occasion_count`;
    /// the `m1_shift` check below only confirms the fixture would make it visible.
    #[test]
    fn impmap_iov_trace_ofv_matches_the_joint_is_marginal() {
        let model = iov_model();
        let pop = iov_pop();
        let params = model.default_params.clone();

        let mut opts = FitOptions::default();
        opts.impmap_iterations = 1;
        opts.impmap_samples = 2000;
        opts.impmap_seed = Some(11);
        opts.impmap_trace = true;
        opts.run_covariance_step = false;

        let res = run_impmap(&model, &pop, &params, None, &opts).expect("impmap on an IOV model");
        let trace = res.impmap_trace.as_ref().expect("impmap_trace requested");
        let row = trace
            .rows
            .iter()
            .find(|r| r.iteration == 1)
            .expect("iteration 1 row");
        assert!(row.ofv.is_finite(), "trace ofv {}", row.ofv);

        let is = joint_is_at(&model, &pop, &params);
        assert!(is.minus2_log_likelihood.is_finite());

        let iov = params.omega_iov.as_ref().expect("IOV model");
        let n_occ_total: usize = pop
            .subjects
            .iter()
            .map(crate::stats::stacked_layout::group_count)
            .sum();
        let m1_shift = n_occ_total as f64 * iov.log_det.abs();
        let gap = (row.ofv - is.minus2_log_likelihood).abs();
        eprintln!(
            "trace {} vs IS {} (gap {gap}, IS mcse {}); dropping the κ prior's log-det moves it by {m1_shift}",
            row.ofv, is.minus2_log_likelihood, is.mc_standard_error
        );
        // Measured (macOS arm64, 2000 samples each): gap 0.029 against an IS MC SE
        // of 0.10; dropping the κ log-det moves the trace by 92. 0.5 is ~5 MC SE.
        assert!(
            gap < 0.5,
            "trace {} vs IS {}: gap {gap}",
            row.ofv,
            is.minus2_log_likelihood
        );
        assert!(
            m1_shift > 10.0,
            "the fixture must make the κ log-det term visible"
        );
    }

    #[test]
    fn sampled_dimension_phrase_names_the_stacked_kappa_dimensions_only_for_iov() {
        let iov = sampled_dimension_phrase(3, 1, 2);
        assert_eq!(
            iov,
            "a 3-ETA model with 1 κ on each of up to 2 occasions (5 sampled dimensions per subject)"
        );
        // No IOV, or IOV declared but no subject has occasions: η alone.
        assert_eq!(sampled_dimension_phrase(3, 0, 0), "a 3-ETA model");
        assert_eq!(sampled_dimension_phrase(3, 1, 0), "a 3-ETA model");
    }

    /// The under-sampling trigger counts the stacked `[η, κ₁..κ_K]` width: at
    /// K = 400 with `auto` off, warfarin_iov (3 η + 2 κ = 5 dimensions, needing
    /// 500) warns and names 5 dimensions, while the same model without `[iov]`
    /// (3 dimensions, needing 300) does not. Under the old `100 · n_eta` rule
    /// neither warned, so the IOV leg straddles the change.
    #[test]
    fn imp_sample_count_warning_counts_the_stacked_kappa_dimensions() {
        let src = include_str!("../../examples/warfarin_iov.ferx");
        let parse = |s: &str| {
            crate::parser::model_parser::parse_full_model(s)
                .expect("parses")
                .model
        };
        let iov = parse(src);
        let eta_only = parse(
            &src.replace("  kappa KAPPA_CL ~ 0.01\n", "")
                .replace(" + KAPPA_CL", ""),
        );
        let pop = iov_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 2;
        opts.impmap_samples = 400;
        opts.impmap_auto = false;
        opts.impmap_seed = Some(3);
        opts.run_covariance_step = false;
        assert!(400 < MIN_SAMPLES_PER_DIM * 5 && 400 >= MIN_SAMPLES_PER_DIM * 3);

        let sample_warning = |model: &CompiledModel| -> Option<String> {
            let res =
                run_impmap(model, &pop, &model.default_params, None, &opts).expect("impmap runs");
            res.warnings
                .into_iter()
                .find(|w| w.contains("importance samples for"))
        };
        let w =
            sample_warning(&iov).expect("an IOV fit at K = 400 is under-sampled for 5 dimensions");
        assert!(
            w.contains("400 importance samples for a 3-ETA model with 1 κ on each of up to 2 occasions (5 sampled dimensions per subject)"),
            "{w}"
        );
        assert_eq!(
            sample_warning(&eta_only),
            None,
            "3 dimensions at K = 400 is enough"
        );
    }

    /// An `mceta` random start for an IOV subject covers κ (#2301): it is the
    /// stacked `[η, κ₁..κ_K]`, η from N(0, Ω) and every occasion's κ from
    /// N(0, Ω_iov) — not κ restarted at 0. Without κ it is the pre-IOV draw.
    #[test]
    fn random_start_draws_each_occasions_kappa_from_omega_iov() {
        let omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.04]);
        let omega_iov = DMatrix::from_row_slice(1, 1, &[0.25]);
        let l_omega = omega.clone().cholesky().unwrap().l();
        let l_iov = omega_iov.clone().cholesky().unwrap().l();

        // η alone: exactly L_Ω·z from the same stream.
        let mut a = StdRng::seed_from_u64(9);
        let mut b = StdRng::seed_from_u64(9);
        let z: Vec<f64> = (0..2).map(|_| StandardNormal.sample(&mut b)).collect();
        let want: Vec<f64> = (&l_omega * DVector::from_vec(z)).iter().copied().collect();
        assert_eq!(random_start(&mut a, &l_omega, None), want);

        // Stacked: 2 η + 3 occasions × 1 κ, with the declared variances.
        let n = 20_000;
        let mut rng = StdRng::seed_from_u64(9);
        let mut sum2 = [0.0f64; 5];
        for _ in 0..n {
            let s = random_start(&mut rng, &l_omega, Some((&l_iov, 3)));
            assert_eq!(s.len(), 5);
            for (acc, v) in sum2.iter_mut().zip(&s) {
                *acc += v * v;
            }
        }
        let var: Vec<f64> = sum2.iter().map(|s| s / n as f64).collect();
        let want = [0.09, 0.04, 0.25, 0.25, 0.25];
        for (k, (got, w)) in var.iter().zip(want).enumerate() {
            // 20 000 draws: relative SE of a variance is sqrt(2/n) = 1%; 5% bound.
            assert!(
                (got / w - 1.0).abs() < 0.05,
                "coordinate {k}: var {got} vs {w}"
            );
        }
    }

    /// The κ draw is wired in for exactly the subjects whose inner solve is the
    /// stacked one: an IOV subject with restarts requested.
    #[test]
    fn kappa_start_chol_is_set_only_for_iov_subjects_with_restarts() {
        let model = iov_model();
        let pop = iov_pop();
        let params = &model.default_params;
        let subject = &pop.subjects[0];
        let l = kappa_start_chol(&model, subject, params, 3).expect("IOV subject, mceta = 3");
        let var = params.omega_iov.as_ref().unwrap().matrix[(0, 0)];
        assert!((l[(0, 0)] * l[(0, 0)] - var).abs() < 1e-15);
        assert!(
            kappa_start_chol(&model, subject, params, 0).is_none(),
            "no restarts"
        );

        let src = include_str!("../../examples/warfarin_iov.ferx");
        let eta_only = crate::parser::model_parser::parse_full_model(
            &src.replace("  kappa KAPPA_CL ~ 0.01\n", "")
                .replace(" + KAPPA_CL", ""),
        )
        .expect("parses")
        .model;
        assert!(kappa_start_chol(&eta_only, subject, &eta_only.default_params, 3).is_none());
    }

    /// With κ in the restarts, an IOV subject's multistart still returns a full
    /// set of κ̂ blocks and never a worse mode than the baseline start.
    #[test]
    fn find_ebe_multistart_on_an_iov_subject_is_never_worse_than_baseline() {
        let model = iov_model();
        let pop = iov_pop();
        let params = &model.default_params;
        let chol = params.omega.matrix.clone().cholesky().unwrap().l();
        for subject in pop.subjects.iter().take(4) {
            let n_occ = crate::stats::stacked_layout::group_count(subject);
            let base =
                find_ebe_multistart(&model, subject, params, 50, 1e-4, None, None, 0, 0, None, 7);
            let multi = find_ebe_multistart(
                &model,
                subject,
                params,
                50,
                1e-4,
                None,
                None,
                0,
                4,
                Some(&chol),
                7,
            );
            assert!(base.nll.is_finite() && multi.nll.is_finite());
            assert!(
                multi.nll <= base.nll + 1e-12,
                "{} vs {}",
                multi.nll,
                base.nll
            );
            assert_eq!(multi.kappas.len(), n_occ, "subject {}", subject.id);
        }
    }

    /// `find_ebe_multistart` hands the κ draw to the IOV inner solve. With
    /// `inner_maxiter = 0` the solve returns (essentially) its start, so the
    /// winning restart's κ̂ is its drawn κ — a draw from N(0, Ω_iov = 0.01), not
    /// the 0 (or the solver's 2.5e-4 nudge) every start had before #2301's
    /// review. Measured: the winners' |κ| reach 0.095; 0.005 is 20× the nudge.
    #[test]
    fn find_ebe_multistart_restarts_carry_their_kappa_draw_into_the_iov_solve() {
        let model = iov_model();
        let pop = iov_pop();
        let params = &model.default_params;
        let chol = params.omega.matrix.clone().cholesky().unwrap().l();
        let mut largest = 0.0f64;
        for subject in &pop.subjects {
            let multi = find_ebe_multistart(
                &model,
                subject,
                params,
                0,
                1e-4,
                None,
                None,
                0,
                4,
                Some(&chol),
                7,
            );
            for k in multi.kappas.iter().flat_map(|b| b.iter()) {
                assert!(k.is_finite());
                largest = largest.max(k.abs());
            }
        }
        assert!(
            largest > 0.005,
            "no restart carried a drawn κ: max |κ̂| = {largest}"
        );
    }

    /// The per-K cache holds exactly what each IOV subject built for itself
    /// before it existed: the joint inverse, log-det and prior matrix of its own
    /// occasion count — bit for bit — and one entry per distinct K > 0.
    #[test]
    fn joint_priors_match_the_per_subject_construction_for_each_occasion_count() {
        let omega = OmegaMatrix::from_matrix(
            DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.04]),
            vec!["A".into(), "B".into()],
            false,
        );
        let iov = OmegaMatrix::from_matrix(
            DMatrix::from_row_slice(1, 1, &[0.25]),
            vec!["K".into()],
            true,
        );
        let priors = JointPriors::build(&omega, &iov, &[2, 0, 1, 2, 3], 0.1);
        assert_eq!(priors.0.keys().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        for k in [1usize, 2, 3] {
            let p = priors.get(k);
            assert_eq!(
                p.inv,
                build_joint_omega_inv(&omega.inv, &iov.inv, 2, 1, k),
                "K = {k}"
            );
            assert_eq!(p.log_det, omega.log_det + k as f64 * iov.log_det, "K = {k}");
            assert_eq!(
                p.matrix,
                crate::stats::stacked_layout::block_diag_stack(&omega.matrix, &iov.matrix, k),
                "K = {k}"
            );
            assert_eq!(p.inv.nrows(), 2 + k, "K = {k}");
            assert!(
                p.mixture.is_some(),
                "α = 0.1 builds the defensive component"
            );
        }
    }

    /// The reused κ split must score every sample exactly as fresh `Vec`s did,
    /// including a narrower subject scored right after a wider one on the same
    /// thread. (Passing the whole buffer, stale blocks included, is an
    /// equivalent mutation today — the scorer reads one block per occasion —
    /// so `split_kappas`'s own test pins the slice contract.)
    #[test]
    fn draw_obs_nll_with_a_reused_kappa_buffer_matches_fresh_vecs() {
        let model = iov_model();
        let pop = iov_pop();
        let theta = &model.default_params.theta;
        let sigma = &model.default_params.sigma.values;
        let mut scratch = EventPkParams::default();
        let fresh = |subject: &Subject, b: &[f64], scratch: &mut EventPkParams| {
            let kappas: Vec<Vec<f64>> = b[3..].chunks(1).map(<[f64]>::to_vec).collect();
            crate::estimation::fixed_eta_gradient::obs_nll_subject_into_iov(
                &model,
                subject,
                theta,
                sigma,
                &b[..3],
                &kappas,
                scratch,
            )
        };
        // warfarin_iov subjects all have two occasions; a copy with every record
        // on its first occasion has one.
        let mut subjects: Vec<Subject> = Vec::new();
        for s in pop.subjects.iter().take(4) {
            let first = s.occasions[0];
            let mut narrow = s.clone();
            narrow.occasions.iter_mut().for_each(|o| *o = first);
            narrow.dose_occasions.iter_mut().for_each(|o| *o = first);
            narrow.pk_only_occasions.iter_mut().for_each(|o| *o = first);
            narrow.reset_occasions.iter_mut().for_each(|o| *o = first);
            subjects.push(s.clone());
            subjects.push(narrow);
        }
        let widths: Vec<usize> = subjects
            .iter()
            .map(crate::stats::stacked_layout::group_count)
            .collect();
        assert!(
            widths.windows(2).any(|w| w[0] > w[1]),
            "the fixture must score a narrower subject after a wider one: {widths:?}"
        );
        for (n, subject) in subjects.iter().enumerate() {
            let mut b = vec![0.1, -0.05, 0.2];
            b.extend(
                (0..widths[n])
                    .map(|g| 0.03 * (g as f64 + 1.0) * if n % 4 < 2 { 1.0 } else { -1.0 }),
            );
            let got = draw_obs_nll(&model, subject, theta, sigma, &b, None, &mut scratch);
            assert!(got.is_finite());
            assert_eq!(
                got,
                fresh(subject, &b, &mut scratch),
                "subject {} (call {n})",
                subject.id
            );
        }
    }

    /// `split_kappas` reuses the buffer's allocations: after a wider sample the
    /// first `n_occ` blocks hold exactly the narrower sample's κ (each block
    /// cleared before refilling), and the returned count excludes the rest.
    #[test]
    fn split_kappas_refills_the_reused_blocks_and_counts_only_this_sample() {
        let mut buf = Vec::new();
        assert_eq!(
            split_kappas(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, &mut buf),
            3
        );
        assert_eq!(buf, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(split_kappas(&[7.0, 8.0], 2, &mut buf), 1);
        assert_eq!(buf[..1], [vec![7.0, 8.0]]);
        assert_eq!(buf.len(), 3, "the wider sample's allocations are kept");
    }

    // ---- `auto` sample count and wall-clock limit (#2687) ----

    #[test]
    fn objective_settled_needs_two_full_windows() {
        let flat = vec![-250.0; 2 * AUTO_SETTLE_WINDOW - 1];
        assert!(!objective_settled(&flat, 1.0));
        let flat = vec![-250.0; 2 * AUTO_SETTLE_WINDOW];
        assert!(objective_settled(&flat, 1.0));
    }

    #[test]
    fn objective_settled_compares_the_last_two_windows_to_the_mc_se() {
        // Thioguanine's iterations 1-10: a descent of hundreds against a total SE
        // near 2.8 (0.5 per subject, 31 subjects).
        let cold = [
            65.7, -86.8, -145.2, -177.6, -212.4, -231.0, -240.0, -245.0, -247.0, -248.4,
        ];
        assert!(!objective_settled(&cold, 2.8));
        // Only the last 2w iterations count: the same descent followed by a plateau.
        let mut settled = cold.to_vec();
        settled.extend([-250.3, -250.1, -250.4, -250.2, -250.3]);
        settled.extend([-250.2, -250.4, -250.1, -250.3, -250.3]);
        assert!(objective_settled(&settled, 0.33));
        // The bound on a window-mean step of 1: 2·SE·√(2/5) straddles it at SE ≈ 0.79.
        let step: Vec<f64> = [[0.0; 5], [1.0; 5]].concat();
        assert!(!objective_settled(&step, 0.78));
        assert!(objective_settled(&step, 0.80));
        // A rise is movement as well.
        let rise: Vec<f64> = [[1.0; 5], [0.0; 5]].concat();
        assert!(!objective_settled(&rise, 0.78));
    }

    #[test]
    fn objective_settled_reads_a_nan_as_unsettled() {
        let mut o = vec![1.0; 2 * AUTO_SETTLE_WINDOW];
        assert!(!objective_settled(&o, f64::NAN));
        o[9] = f64::NAN;
        assert!(!objective_settled(&o, 1.0));
    }

    #[test]
    fn auto_hold_ends_before_the_averaging_window_within_its_bounds() {
        assert_eq!(auto_hold_ends(200, 50), 150, "the default: the window");
        assert_eq!(auto_hold_ends(200, 1), 150, "at most 3/4 of the run");
        assert_eq!(auto_hold_ends(200, 150), 100, "at least 1/2 of the run");
        assert_eq!(auto_hold_ends(6, 6), 3);
        assert_eq!(auto_hold_ends(1, 1), 1);
    }

    #[test]
    fn windowed_se_takes_the_median_over_the_window() {
        // A spike in either direction does not move it.
        let spike = [
            (0.05, 1000),
            (0.05, 1000),
            (0.5, 1000),
            (0.05, 1000),
            (0.001, 1000),
        ];
        assert!((windowed_se(&spike, 1000) - 0.05).abs() < 1e-12);
        // Only the last window counts.
        let old = [
            (9.0, 1000),
            (0.05, 1000),
            (0.05, 1000),
            (0.05, 1000),
            (0.05, 1000),
            (0.05, 1000),
        ];
        assert!((windowed_se(&old, 1000) - 0.05).abs() < 1e-12);
        // A short history still answers, and an empty one reads as no signal.
        assert!((windowed_se(&[(0.2, 100)], 100) - 0.2).abs() < 1e-12);
        assert!(windowed_se(&[], 100).is_nan());
    }

    #[test]
    fn windowed_se_rescales_across_a_change_of_k() {
        // The same per-draw variance, sampled at K and then 4K: the SE halves, and
        // the window reads one value at each count.
        let h = [
            (0.2, 1000),
            (0.2, 1000),
            (0.1, 4000),
            (0.1, 4000),
            (0.1, 4000),
        ];
        assert!((windowed_se(&h, 4000) - 0.1).abs() < 1e-12);
        assert!((windowed_se(&h, 1000) - 0.2).abs() < 1e-12);
    }

    /// Cefepime's trace at iterations 30-35: neither the one quiet iteration among
    /// noisy ones nor the one noisy iteration among quiet ones moves K. Read off the
    /// last iteration's SE, the first would halve K and the second double it.
    #[test]
    fn auto_step_does_not_act_on_a_single_iteration() {
        let flat = vec![4290.0; 10];
        let k = 4000;
        let quiet = [(0.04, k), (0.04, k), (0.04, k), (0.04, k), (0.02, k)];
        assert_eq!(auto_step(&flat, &quiet, 1000, 458, true).0, k);
        let noisy = [(0.04, k), (0.04, k), (0.04, k), (0.04, k), (0.09, k)];
        assert_eq!(auto_step(&flat, &noisy, 1000, 458, true).0, k);
    }

    /// The hold, through the same decision the loop makes: above the target K waits
    /// for a settled objective or the end of the hold, and takes either.
    #[test]
    fn auto_step_raises_once_settled_or_once_the_hold_is_over() {
        let k = 1000;
        let se = [(0.1, k); 5];
        let descending: Vec<f64> = (0..10).map(|i| 1000.0 - 50.0 * i as f64).collect();
        let settled = vec![500.0; 10];
        assert_eq!(auto_step(&descending, &se, k, 100, false).0, k);
        assert_eq!(auto_step(&settled, &se, k, 100, false).0, 2 * k);
        assert_eq!(auto_step(&descending, &se, k, 100, true).0, 2 * k);
    }

    /// The settled test reads the total SE, √N times the per-subject one: this step
    /// of 1 is inside the noise of 100 subjects at 0.1 and outside that of one.
    #[test]
    fn auto_step_scales_the_settled_test_by_the_subject_count() {
        let k = 1000;
        let se = [(0.1, k); 5];
        let step: Vec<f64> = [[0.0; 5], [1.0; 5]].concat();
        assert_eq!(auto_step(&step, &se, k, 100, false).0, 2 * k);
        assert_eq!(auto_step(&step, &se, k, 1, false).0, k);
    }

    #[test]
    fn is_degenerate_is_ess_below_two() {
        assert!(is_degenerate(1.0 / 300.0, 300));
        assert!(is_degenerate(0.0, 300));
        assert!(is_degenerate(f64::NAN, 300));
        assert!(!is_degenerate(2.0 / 300.0, 300));
        assert!(!is_degenerate(0.5, 300));
    }

    /// The mechanism of #2690: a one-draw subject's term is about 1 at every K, the
    /// same as a failed subject's fallback, while a healthy subject's falls as 1/K.
    #[test]
    fn a_one_draw_subjects_variance_term_does_not_fall_with_k() {
        for k in [300, 1000, AUTO_SAMPLES_MAX] {
            let term = log_marginal_mc_variance(1.0 / k as f64, k);
            assert!((term - 1.0).abs() <= 1.0 / 300.0, "k = {k}: {term}");
        }
        assert_eq!(log_marginal_mc_variance(0.0, 1000), 1.0);
        assert!((log_marginal_mc_variance(0.5, 1000) - 1e-3).abs() < 1e-12);
        assert!((log_marginal_mc_variance(0.5, 2000) - 5e-4).abs() < 1e-12);
    }

    #[test]
    fn a_degenerate_subject_is_left_out_once_a_change_of_k_has_not_helped() {
        let mut t = StuckSubjects::default();
        let one = |k: usize| 1.0 / k as f64;
        // One-draw at the start: it may only be undersampled, so it stays in.
        assert_eq!(t.update(&[0.5, one(300)], 300), [false, false]);
        // Still one-draw at the same K: no evidence yet that draws do not help.
        assert_eq!(t.update(&[0.5, one(300)], 300), [false, false]);
        // Still one-draw after a raise: left out.
        assert_eq!(t.update(&[0.5, one(600)], 600), [false, true]);
        // And it stays out after a lowering.
        assert_eq!(t.update(&[0.5, one(300)], 300), [false, true]);
        assert_eq!(t.update(&[0.5, one(600)], 600), [false, true]);
        // A recovery ends the run, so it must fail at two counts again.
        assert_eq!(t.update(&[0.5, 0.2], 300), [false, false]);
        assert_eq!(t.update(&[0.5, one(300)], 300), [false, false]);
        assert_eq!(t.update(&[0.5, one(600)], 600), [false, true]);
    }

    /// A lowering is not a test of whether more draws help: a subject degenerate
    /// first at a high K and then at a lower one stays in, and is left out only
    /// once K rises past the lowest K of its run.
    #[test]
    fn a_lowering_does_not_mark_a_degenerate_subject_stuck() {
        let mut t = StuckSubjects::default();
        let one = |k: usize| 1.0 / k as f64;
        assert_eq!(t.update(&[one(4000)], 4000), [false]);
        assert_eq!(t.update(&[one(2000)], 2000), [false]);
        assert_eq!(t.update(&[one(1000)], 1000), [false]);
        // Back up to a K it has been degenerate at before, but above the run's
        // lowest: that raise did not help.
        assert_eq!(t.update(&[one(2000)], 2000), [true]);
    }

    /// Once a subject is left out, the window `auto` reads must lose it too. With
    /// the stale entries the median still includes the one-draw subject and K
    /// doubles; rebuilt on the current set it holds. Both sides are asserted.
    #[test]
    fn leaving_a_subject_out_rebuilds_the_se_window() {
        let k = 1000;
        let mut fracs = vec![0.8; 100];
        fracs.push(1.0 / k as f64);
        let history: Vec<(Vec<f64>, usize)> = vec![(fracs.clone(), k); 5];
        let flat = vec![4000.0; 10];
        let mut stuck = vec![false; 101];
        let stale_with_newest = {
            let mut h = ramp_se_history(&history[..4], &stuck);
            stuck[100] = true;
            h.push((ramp_se_per_subject(&fracs, k, &stuck).0, k));
            h
        };
        assert_eq!(auto_step(&flat, &stale_with_newest, k, 100, true).0, 2 * k);
        let rebuilt = ramp_se_history(&history, &stuck);
        assert!(rebuilt.iter().all(|&(se, _)| se < AUTO_STDOBJ_TARGET));
        assert_eq!(auto_step(&flat, &rebuilt, k, 100, true).0, k);
    }

    #[test]
    fn a_failed_subject_is_left_out_at_once() {
        let mut t = StuckSubjects::default();
        assert_eq!(t.update(&[0.0, f64::NAN, 0.5], 300), [true, true, false]);
    }

    /// 100 well-sampled subjects and one that draws cannot fix: with it the SE sits
    /// at `2/√101 ≈ 0.2`, four times the target, and would ramp all 101 to the cap;
    /// without it the SE is under the target. Both sides of the straddle are asserted.
    #[test]
    fn a_stuck_subject_does_not_set_the_ramp_se() {
        let k = 1000;
        let mut fracs = vec![0.8; 100];
        fracs.push(1.0 / k as f64);
        let mut stuck = vec![false; 101];
        let (with, n_with) = ramp_se_per_subject(&fracs, k, &stuck);
        assert_eq!(n_with, 101);
        assert!(with > 2.0 / (101f64).sqrt() - 1e-3, "{with}");
        stuck[100] = true;
        let (without, n_without) = ramp_se_per_subject(&fracs, k, &stuck);
        assert_eq!(n_without, 100);
        // Each healthy term is (1/0.8 − 1)/1000 = 2.5e-4, so the SE is 2·√2.5e-4.
        assert!(
            (without - 2.0 * (2.5e-4f64).sqrt()).abs() < 1e-12,
            "{without}"
        );
        assert!(without < AUTO_STDOBJ_TARGET && with > AUTO_STDOBJ_TARGET);
        // With every subject left out there is nothing to raise for.
        assert_eq!(ramp_se_per_subject(&fracs, k, &[true; 101]), (0.0, 0));
    }

    #[test]
    fn lowest_ess_summary_lists_the_worst_subjects_first() {
        let ids = ["a", "b", "c"];
        assert_eq!(
            lowest_ess_summary(&ids, &[0.5, 0.001, 0.1], 1000, 2),
            "b (ESS 1.0), c (ESS 100.0)"
        );
    }

    #[test]
    fn stuck_subjects_sentence_names_the_subjects_under_auto_only() {
        let ids: Vec<String> = (1..=12).map(|i| format!("S{i}")).collect();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        let s = stuck_subjects_sentence(true, "imp_auto", &ids);
        assert!(s.contains("`imp_auto` left 12 subject(s)"), "{s}");
        assert!(s.contains("S1, S2") && s.contains("S10 and 2 more"), "{s}");
        assert!(!s.contains("S11"), "{s}");
        assert!(s.contains("better proposal, not more samples"), "{s}");
        assert_eq!(stuck_subjects_sentence(false, "imp_auto", &ids), "");
        assert_eq!(stuck_subjects_sentence(true, "imp_auto", &[]), "");
        assert_eq!(id_list(&["x", "y"], 10), "x, y");
    }

    #[test]
    fn next_sample_count_raises_only_when_allowed() {
        assert_eq!(next_sample_count(1000, 1000, 0.5, false), 1000);
        assert_eq!(next_sample_count(1000, 1000, 0.5, true), 2000);
    }

    #[test]
    fn next_sample_count_sizes_the_raise_to_the_se() {
        // Cefepime's `0.050 > 0.05`: an SE that only just misses the target used
        // to double K. SE² ∝ 1/K asks for 2 % more, and the headroom adds 10 %.
        assert_eq!(next_sample_count(4000, 1000, 0.0505, true), 4489);
        // Doubling and the cap both still bound the step.
        assert_eq!(next_sample_count(4000, 1000, 0.09, true), 8000);
        assert_eq!(next_sample_count(8000, 1000, 0.09, true), AUTO_SAMPLES_MAX);
        assert_eq!(
            next_sample_count(AUTO_SAMPLES_MAX, 1000, 0.09, true),
            AUTO_SAMPLES_MAX
        );
        // A configured count above the cap is the user's, and `auto` keeps it.
        assert_eq!(next_sample_count(20_000, 20_000, 0.09, true), 20_000);
    }

    #[test]
    fn next_sample_count_lowers_to_no_less_than_the_start() {
        assert_eq!(next_sample_count(4000, 1000, 0.02, false), 2000);
        assert_eq!(next_sample_count(1500, 1000, 0.02, true), 1000);
        assert_eq!(next_sample_count(1000, 1000, 0.02, true), 1000);
    }

    #[test]
    fn next_sample_count_holds_between_half_the_target_and_the_target() {
        for se in [0.5 * AUTO_STDOBJ_TARGET, 0.04, AUTO_STDOBJ_TARGET, f64::NAN] {
            assert_eq!(next_sample_count(4000, 1000, se, true), 4000, "se = {se}");
        }
    }

    #[test]
    fn under_sampling_advice_names_what_ran_out() {
        let advice = |auto, k, stopped| {
            let mut history = at_scale(5.0, 1000, 5);
            history.extend(at_scale(if stopped { 11.0 } else { 5.0 }, 2000, 5));
            under_sampling_advice(auto, k, &history, 1000, "imp_samples", "imp_auto", false)
        };
        let at_cap = advice(true, AUTO_SAMPLES_MAX, false);
        assert!(at_cap.contains("was not enough") && !at_cap.contains("below the cap"));
        // Reaching the cap is the story even if the last raises were refused.
        assert_eq!(advice(true, AUTO_SAMPLES_MAX, true), at_cap);
        let stopped = advice(true, 2000, true);
        assert!(
            stopped.contains("stopped raising the sample count at 2000")
                && stopped.contains("not lowering the MC SE")
                && stopped.contains("larger `imp_samples` is unlikely to help")
                && !stopped.contains("more iterations"),
            "{stopped}"
        );
        // The flag only means something under `auto`.
        assert_eq!(advice(false, 2000, true), advice(false, 2000, false));
        let short = advice(true, 2000, false);
        assert!(
            short.contains("ended at 2000 samples")
                && short.contains("`imp_samples`")
                && short.contains("more iterations")
                && !short.contains("was not enough"),
            "{short}"
        );
        let fixed = advice(false, 2000, false);
        assert!(fixed.contains("`imp_auto = true`") && !fixed.contains("is enabled"));
    }

    /// When only the left-out subjects are noisy, the advice must not ask for more
    /// samples or iterations, whatever else the history says; without them it does.
    #[test]
    fn under_sampling_advice_does_not_ask_for_samples_left_out_subjects_cannot_use() {
        let h = at_scale(5.0, 1000, 5);
        for k in [1000, AUTO_SAMPLES_MAX] {
            let left_out =
                under_sampling_advice(true, k, &h, 1000, "imp_samples", "imp_auto", true);
            assert!(
                left_out.contains("met the MC-SE target")
                    && left_out.contains("left out")
                    && left_out.contains("larger `imp_samples` is unlikely to help")
                    && !left_out.contains("more iterations")
                    && !left_out.contains("was not enough"),
                "{left_out}"
            );
        }
        let ordinary =
            under_sampling_advice(true, 1000, &h, 1000, "imp_samples", "imp_auto", false);
        assert!(ordinary.contains("more iterations") && !ordinary.contains("left out"));
        // Without `auto` nothing was left out of any ramp.
        let fixed = under_sampling_advice(false, 1000, &h, 1000, "imp_samples", "imp_auto", true);
        assert!(fixed.contains("`imp_auto = true`") && !fixed.contains("left out"));
    }

    #[test]
    fn collapse_sentence_makes_no_claim_about_more_samples() {
        assert_eq!(collapse_sentence(0), "");
        let s = collapse_sentence(3);
        assert!(s.contains("3 subject(s)") && s.contains("ESS < 2"), "{s}");
        assert!(s.contains("at this sample count"), "{s}");
        assert!(
            !s.contains("regardless") && !s.contains("not more samples"),
            "{s}"
        );
    }

    /// `n` iterations at `k` with a constant `SE²·K` of `scale`.
    fn at_scale(scale: f64, k: usize, n: usize) -> Vec<(f64, usize)> {
        vec![((scale / k as f64).sqrt(), k); n]
    }

    #[test]
    fn raises_pay_while_the_scale_stays_under_the_limit() {
        // No raise yet: nothing to compare against, whatever the SE.
        assert!(raises_pay(&at_scale(50.0, 1000, 5), 1000));
        let mut h = at_scale(5.0, 1000, 5);
        // SE² ∝ 1/K holds: the SE fell with the raise.
        h.extend(at_scale(5.0, 2000, 5));
        assert!(raises_pay(&h, 1000));
        h.extend(at_scale(9.9, 2000, 5));
        assert!(raises_pay(&h, 1000));
        h.extend(at_scale(10.1, 2000, 5));
        assert!(!raises_pay(&h, 1000));
    }

    #[test]
    fn raises_pay_measures_from_the_first_raise_and_keeps_that_baseline() {
        // Scale 5 before the first raise, 8 after, then K back at the start. The
        // baseline stays 5, so 11 is over the limit, and against the 8 of the second
        // visit to K = 2000 it would not be.
        let mut h = at_scale(5.0, 1000, 5);
        h.extend(at_scale(8.0, 2000, 5));
        h.extend(at_scale(8.0, 1000, 5));
        h.extend(at_scale(8.0, 2000, 5));
        assert!(raises_pay(&h, 1000));
        h.extend(at_scale(11.0, 2000, 5));
        assert!(!raises_pay(&h, 1000));
        // The product falling back reopens raising.
        h.extend(at_scale(6.0, 2000, 5));
        assert!(raises_pay(&h, 1000));
    }

    #[test]
    fn raises_pay_baseline_is_a_window_not_the_last_iteration() {
        // The last iteration before the raise spiked to 100; the window median is 5.
        let mut h = at_scale(5.0, 1000, 4);
        h.extend(at_scale(100.0, 1000, 1));
        h.extend(at_scale(11.0, 2000, 5));
        assert!(!raises_pay(&h, 1000));
    }

    #[test]
    fn auto_step_stops_raising_when_the_se_does_not_fall_with_k() {
        // The cefepime shape: SE 0.071 at K = 1000 and still 0.072 at K = 2000.
        let flat = vec![4276.0; 20];
        let mut h = vec![(0.071, 1000); 5];
        h.extend(vec![(0.072, 2000); 5]);
        assert_eq!(auto_step(&flat, &h, 1000, 458, true).0, 2000);
        // The twin whose SE did fall with K, still over target, keeps ramping.
        let mut paying = vec![(0.071, 1000); 5];
        paying.extend(vec![(0.0515, 2000); 5]);
        assert!(auto_step(&flat, &paying, 1000, 458, true).0 > 2000);
    }

    #[test]
    fn auto_step_still_lowers_after_raises_stopped() {
        let flat = vec![4276.0; 20];
        // Scale 1.25 at K = 500, then 4.84 at the cap: raises stopped, SE under half
        // the target.
        let mut h = vec![(0.05, 500); 5];
        h.extend(vec![(0.022, AUTO_SAMPLES_MAX); 5]);
        assert!(!raises_pay(&h, 500));
        assert_eq!(auto_step(&flat, &h, 500, 458, true).0, AUTO_SAMPLES_MAX / 2);
    }

    /// `WT` changes within each subject, so scoring takes the event-driven walk,
    /// and σ is free, so the weighted M-step runs.
    fn tv_fixture() -> (CompiledModel, Population) {
        use std::io::Write;
        let model = crate::parser::model_parser::parse_full_model(
            r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04

[individual_parameters]
  CL = TVCL * (WT / 70.0)^0.75 * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
",
        )
        .expect("model parses")
        .model;
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        for id in 1..=6 {
            let cl = 0.8 + 0.1 * id as f64;
            csv.push_str(&format!("{id},0,0,100,1,1,60\n"));
            for (j, t) in [0.5_f64, 1.0, 2.0, 4.0, 8.0].iter().enumerate() {
                let dv = 10.0 * (-(cl / 10.0) * t).exp() * (1.0 + 0.03 * ((id + j) as f64).sin());
                csv.push_str(&format!("{id},{t},{dv:.5},0,0,1,{}\n", 60 + 5 * j));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let pop = crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap();
        (model, pop)
    }

    fn tv_imp_options(iterations: usize) -> FitOptions {
        FitOptions {
            method: EstimationMethod::Imp,
            imp_iterations: iterations,
            imp_samples: 30,
            imp_averaging: 1,
            imp_auto: false,
            imp_seed: Some(2687),
            run_covariance_step: false,
            ..FitOptions::default()
        }
    }

    #[test]
    fn a_cached_schedule_scores_a_draw_as_the_rebuilt_one_does() {
        let (model, pop) = tv_fixture();
        let schedules = mcem_schedule_cache(&model, &pop, &vec![0; pop.subjects.len()]);
        assert!(schedules.iter().all(Option::is_some), "premise: cacheable");
        let theta = &model.default_params.theta;
        let sigma = &model.default_params.sigma.values;
        let mut scratch = EventPkParams::default();
        for (subject, schedule) in pop.subjects.iter().zip(&schedules) {
            let eta = [0.2, -0.1];
            let cached = draw_obs_nll(
                &model,
                subject,
                theta,
                sigma,
                &eta,
                schedule.as_ref(),
                &mut scratch,
            );
            let rebuilt = draw_obs_nll(&model, subject, theta, sigma, &eta, None, &mut scratch);
            assert!(cached.is_finite());
            assert_eq!(cached.to_bits(), rebuilt.to_bits());
        }
    }

    #[test]
    fn mcem_schedule_cache_uses_the_iov_gate_for_a_subject_with_occasions() {
        let model = iov_model();
        let pop = iov_pop();
        let n = pop.subjects.len();
        assert!(
            pop.subjects.iter().all(|s| !s.has_tv_covariates()),
            "premise: nothing but the IOV gate admits these subjects"
        );
        let with_occasions = mcem_schedule_cache(&model, &pop, &vec![2; n]);
        assert!(with_occasions.iter().all(Option::is_some));
        let without = mcem_schedule_cache(&model, &pop, &vec![0; n]);
        assert!(without.iter().all(Option::is_none));
    }

    /// The E-step pilot, the E-step draws and the M-step all run every iteration,
    /// so a rebuild in any of them makes the count grow with `imp_iterations`.
    /// Equality of the fits cannot see that (#1452).
    #[test]
    fn imp_schedule_builds_do_not_grow_with_iterations() {
        use crate::pk::event_driven::{reset_schedule_build_count, schedule_build_count};
        let (model, pop) = tv_fixture();
        let count = |iterations: usize| -> u64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            pool.install(|| {
                reset_schedule_build_count();
                let opts = tv_imp_options(iterations);
                run_imp(&model, &pop, &model.default_params, None, &opts).expect("IMP Ok");
                schedule_build_count()
            })
        };
        let (short, long) = (count(2), count(6));
        assert!(
            long - short < pop.subjects.len() as u64,
            "schedule builds grew from {short} to {long} over 4 extra iterations"
        );
    }

    /// The same cold start with the hold in force: 12 iterations from K = 8, with
    /// the hold ending at iteration 9, leave room for three raises. Raising from
    /// iteration 1 would reach the cap of 10000.
    #[test]
    fn imp_auto_holds_a_cold_start_and_raises_when_the_hold_ends() {
        let (model, pop) = tv_fixture();
        let opts = FitOptions {
            imp_iterations: 12,
            imp_averaging: 3,
            imp_samples: 8,
            imp_auto: true,
            ..tv_imp_options(12)
        };
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("IMP Ok");
        let warning = res
            .warnings
            .iter()
            .find(|w| w.contains("give a noisy objective"))
            .unwrap_or_else(|| panic!("no under-sampling warning: {:?}", res.warnings));
        let k: usize = warning
            .strip_prefix("IMP: ")
            .and_then(|w| w.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no K in {warning}"));
        assert!(k > 8 && k <= 64, "final K = {k}");
        assert!(warning.contains("below the cap"), "{warning}");
    }

    #[test]
    fn imp_max_time_stops_cleanly_and_reports_not_converged() {
        let (model, pop) = tv_fixture();
        let opts = FitOptions {
            imp_max_time: Some(1e-9),
            ..tv_imp_options(5)
        };
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("IMP Ok");
        assert_eq!(res.n_iterations, 1);
        assert!(!res.converged);
        assert!(res.ofv.is_finite());
        let warning = res
            .warnings
            .iter()
            .find(|w| w.contains("stopped by `imp_max_time`"))
            .unwrap_or_else(|| panic!("no stop warning: {:?}", res.warnings));
        assert!(warning.contains("at iteration 1 of 5"), "{warning}");
        assert!(warning.contains("unaveraged"), "{warning}");
        assert!(warning.contains("reported as not converged"), "{warning}");
        // Its stop was the clock, so the settling verdict has nothing to say about it.
        assert!(res.mcem_objective_trail.is_none());

        let classified = crate::types::classify_warning(warning);
        assert_eq!(classified.severity, WarningSeverity::Warning);
        assert_eq!(classified.category, WarningCode::Convergence);
    }

    /// What the user reads: the published verdict of a stopped fit, on its own and
    /// as the last stage of a chain, whose FOCEI stage the limit must not touch.
    #[test]
    fn a_fit_stopped_by_imp_max_time_publishes_not_converged() {
        let (model, pop) = tv_fixture();
        for methods in [vec![], vec![EstimationMethod::FoceI, EstimationMethod::Imp]] {
            let opts = FitOptions {
                methods: methods.clone(),
                imp_max_time: Some(1e-9),
                ..tv_imp_options(5)
            };
            let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("fit Ok");
            assert!(!res.converged, "{methods:?}");
            assert!(res.ofv.is_finite(), "{methods:?}");
            assert!(
                res.warnings
                    .iter()
                    .any(|w| w.contains("stopped by `imp_max_time`")),
                "{methods:?}: {:?}",
                res.warnings
            );
            // One iteration is under the 30 the settling test needs, but the clock
            // stopped the run: raising `imp_iterations` is not the advice it needs.
            assert!(
                !res.warnings.iter().any(|w| w.contains("W_MCEM_")),
                "{methods:?}: {:?}",
                res.warnings
            );
        }
    }

    #[test]
    fn imp_max_time_reports_the_averaged_estimate_inside_the_window() {
        let (model, pop) = tv_fixture();
        let opts = FitOptions {
            imp_max_time: Some(1e-9),
            imp_averaging: 3,
            ..tv_imp_options(3)
        };
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("IMP Ok");
        assert!(
            res.warnings.iter().any(|w| w
                .contains("the mean of the 1 averaging iteration(s) reached")
                && !w.contains("unaveraged")),
            "{:?}",
            res.warnings
        );
    }

    /// The other side of the gate: a limit that does not bind leaves the schedule
    /// and the verdict alone, and one that binds on the last iteration stopped
    /// nothing.
    #[test]
    fn imp_max_time_that_stops_nothing_changes_nothing() {
        let (model, pop) = tv_fixture();
        let plain = run_imp(
            &model,
            &pop,
            &model.default_params,
            None,
            &tv_imp_options(3),
        )
        .expect("IMP Ok");
        let roomy = FitOptions {
            imp_max_time: Some(1e9),
            ..tv_imp_options(3)
        };
        let roomy = run_imp(&model, &pop, &model.default_params, None, &roomy).expect("IMP Ok");
        assert_eq!(roomy.n_iterations, 3);
        assert!(roomy.converged && plain.converged);
        assert_eq!(roomy.ofv.to_bits(), plain.ofv.to_bits());

        let last = FitOptions {
            imp_max_time: Some(1e-9),
            ..tv_imp_options(1)
        };
        let last = run_imp(&model, &pop, &model.default_params, None, &last).expect("IMP Ok");
        assert!(last.converged, "{:?}", last.warnings);
        assert!(!last.warnings.iter().any(|w| w.contains("_max_time")));
    }

    #[test]
    fn impmap_max_time_is_its_own_key() {
        let (model, pop) = tv_fixture();
        let opts = FitOptions {
            method: EstimationMethod::Impmap,
            impmap_iterations: 4,
            impmap_samples: 30,
            impmap_auto: false,
            impmap_max_time: Some(1e-9),
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let res = run_impmap(&model, &pop, &model.default_params, None, &opts).expect("Ok");
        assert_eq!(res.n_iterations, 1);
        assert!(res
            .warnings
            .iter()
            .any(|w| w.starts_with("IMPMAP: stopped by `impmap_max_time`")));
    }

    #[test]
    fn imp_max_time_stops_a_mixture_fit() {
        let model = mix996_model("TVV * exp(ETA_V)");
        let pop = mix996_pop(3);
        let opts = FitOptions {
            imp_max_time: Some(1e-9),
            ..tv_imp_options(4)
        };
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("IMP Ok");
        assert_eq!(res.n_iterations, 1);
        assert!(!res.converged);
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("stopped by `imp_max_time`")),
            "{:?}",
            res.warnings
        );
        assert!(res.mcem_objective_trail.is_none());
    }

    #[test]
    fn a_max_time_that_cannot_trip_is_refused() {
        let (model, pop) = tv_fixture();
        for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let opts = FitOptions {
                imp_max_time: Some(bad),
                ..tv_imp_options(2)
            };
            let err = run_imp(&model, &pop, &model.default_params, None, &opts)
                .err()
                .unwrap_or_else(|| panic!("{bad} must be refused"));
            assert!(err.contains("`imp_max_time`"), "{err}");
        }
    }
}

#[cfg(test)]
#[path = "impmap_mstep_tests.rs"]
mod mstep_tests;
