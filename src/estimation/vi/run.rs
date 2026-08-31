//! The VI driver: `run_vi`.
//!
//! Structure of one iteration:
//!
//! 1. Evaluate `−ELBO` and its gradients ([`population_neg_elbo`]).
//! 2. Adam-step the packed population vector `x` and every subject's `φᵢ`, then
//!    project `x` back into the declared parameter box ([`compute_bounds`]).
//! 3. Replace `Ω` with its closed-form maximizer (unless `vi_omega_update = adam`).
//! 4. Once inside the averaging window, fold `x` and `{φᵢ}` into a Polyak mean.
//!
//! There is no inner loop and no EBE solve: `φ` persists across iterations the way
//! an EBE warm-start does, but is optimized rather than re-solved.
//!
//! Afterwards: one warm-started inner-loop pass for the conditional modes and `H`,
//! an optional Laplace OFV, and the ordinary FD-of-OFV covariance step at the VI
//! estimate — so a VI fit reports standard errors like any other method.
//!
//! # Stopping
//!
//! `vi_iters` is a **ceiling, not a budget**. The objective is a Monte-Carlo estimate,
//! so a per-iteration improvement test would stop on noise; the run therefore stops on
//! the same windowed moving-average predicate ([`trace_has_settled`]) that decides the
//! reported `converged` flag, checked every `CONVERGENCE_CHECK_INTERVAL` iterations.
//! Tying both to one predicate means "ran to completion" and "reported converged"
//! cannot disagree.
//!
//! Detection does not stop the run on the spot: the reported estimate is a Polyak mean,
//! so a full averaging window is collected *after* settling before breaking. Otherwise
//! early stopping would simply trade under-convergence for a noisy last iterate.

use nalgebra::DVector;

use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_bounds, compute_mu_k, pack_params, unpack_params,
};
use crate::types::{
    CompiledModel, FitOptions, ModelParameters, Population, ViFamily, ViFinalOfv, ViOmegaUpdate,
    ViResult, ViSigmaUpdate,
};

use super::adam::{averaging_start, AdamConfig, AdamState, PolyakAverager};
use super::elbo::{
    closed_form_omega, closed_form_sigma, closed_form_sigma_support, population_neg_elbo,
    stacked_prior, unsupported_data_term_reason, ElboConfig, ElboTightness, Families, PackedLayout,
};
use super::family::{FullRank, MeanField, VariationalFamily};

/// Default seed when `vi_seed` is unset, so a fit is reproducible across
/// invocations without the user having to pin one.
pub const DEFAULT_VI_SEED: u64 = 20_240_704;

/// Fraction of the run used as each half of the convergence moving average.
const CONVERGENCE_WINDOW_FRACTION: f64 = 0.1;

/// Relative tolerance the moving-average change must fall under for the run to be
/// reported as settled.
const CONVERGENCE_REL_TOL: f64 = 1e-4;

/// How often the early-stopping check runs. The predicate compares two windowed means,
/// so testing it every iteration would buy nothing and cost a scan of the trace each
/// time.
const CONVERGENCE_CHECK_INTERVAL: usize = 100;

/// Polyak averaging window to collect *after* the objective has settled, when the user
/// has not pinned one with `vi_avg_last`. Mirrors [`averaging_start`]'s default
/// fraction, applied to the run length so far rather than to `vi_iters` — which under
/// early stopping is a ceiling nobody reached.
fn averaging_window_for(n_so_far: usize) -> usize {
    n_so_far
        .saturating_sub(averaging_start(n_so_far, None))
        .max(1)
}

/// How many standard errors the two window means may differ by and still count as
/// settled. At `2.0` a genuinely flat trace passes ~95% of the time, so the check is
/// not tripped up by ordinary sampling variation.
const SETTLE_Z: f64 = 2.0;

/// Smallest comparison window the settling test will use, regardless of how short the
/// trace is.
///
/// [`CONVERGENCE_WINDOW_FRACTION`] alone makes the window proportional to the trace, so
/// early in a run it is tiny — 12 samples at iteration 125 — and the standard error is
/// correspondingly huge. Any drift then hides inside the noise band and the run stops
/// almost immediately: on warfarin the fraction-only rule declared convergence at
/// iteration 125, roughly 20 000 short of the actual plateau. A floor on the window is
/// what makes the noise estimate meaningful.
const SETTLE_MIN_WINDOW: usize = 500;

/// Consecutive settled checks required before the run stops.
///
/// The test is a hypothesis test on noisy data, so it fires spuriously some fraction of
/// the time by construction. Requiring it to hold across `SETTLE_PATIENCE` consecutive
/// checks (i.e. over `SETTLE_PATIENCE * CONVERGENCE_CHECK_INTERVAL` iterations) makes an
/// isolated lucky window insufficient. Calibrated on a 40 000-iteration warfarin trace
/// whose objective plateaus near iteration 20 000: at patience 3 every combination of
/// window floor and `z` stops at ~22 900, while at patience 1 the looser combinations
/// stop in the first few hundred iterations.
const SETTLE_PATIENCE: usize = 3;

/// Comparison window for a trace of length `n`: a fixed fraction of the run, floored at
/// [`SETTLE_MIN_WINDOW`] so the variance estimate has enough samples to mean anything.
fn settle_window(n: usize) -> usize {
    (((n as f64) * CONVERGENCE_WINDOW_FRACTION).round() as usize).max(SETTLE_MIN_WINDOW)
}

/// Relative movement below which a parameter block mean counts as settled.
///
/// Compared per coordinate against `|mean| + PARAM_SETTLE_FLOOR`, so a coordinate sitting
/// near zero is judged on an absolute scale rather than blowing the ratio up.
const PARAM_SETTLE_REL_TOL: f64 = 1e-3;

/// Absolute floor in the relative-change denominator above.
const PARAM_SETTLE_FLOOR: f64 = 1e-6;

/// Largest relative change between two block means of the packed parameter vector.
///
/// This is the question a user actually asks — *have the estimates stopped moving?* — as
/// opposed to the one [`trace_has_settled`] asks, which is whether the noisy objective
/// has stopped creeping. The two come apart badly on a parameter space with flat
/// directions: a neural-network weight vector has exact permutation and layer-scale
/// symmetries, so an unregularised fit drifts along those ridges indefinitely while the
/// likelihood, the estimates that matter, and the predictions all stop changing. The
/// objective test correctly reports "still moving" forever; this one does not.
///
/// Block means rather than raw iterates, for the same reason `trace_has_settled` averages
/// both of its windows: a single Adam iterate is a noisy draw.
fn max_relative_change(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (y.abs().max(x.abs()) + PARAM_SETTLE_FLOOR))
        .fold(0.0f64, f64::max)
}

/// Whether *both* halves of the optimization have stopped moving between two blocks.
///
/// The parameter-stability criterion is sufficient on its own to stop a VI run, so it has
/// to be asked about everything the run is fitting. `x` — the packed population vector —
/// is only half of that: `φ`, the variational parameters, is the other half, is never
/// FIXed, and is what a VI fit is usually *for*. Measuring `x` alone lets a chain in which
/// the population coordinates settle first (or are nearly all pinned) stop the run while
/// the per-subject posteriors are still moving, and return them as converged.
///
/// Both halves are judged on the same window and the same tolerance. `max_relative_change`
/// is already a max over coordinates, so the max of the two calls is the same question
/// asked over the concatenated vector — written as two only because `x` and `φ` are stored
/// separately.
fn estimates_have_settled(x: &[f64], prev_x: &[f64], phi: &[f64], prev_phi: &[f64]) -> bool {
    max_relative_change(x, prev_x).max(max_relative_change(phi, prev_phi)) < PARAM_SETTLE_REL_TOL
}

/// Whether the parameter-stability criterion means anything for this model.
///
/// [`max_relative_change`] asks whether the population estimates have stopped moving. When every
/// packed coordinate is FIXed or a structural zero, `x` is constant *by construction* — so the
/// criterion answers "settled" from its first comparison, no matter what the fit is doing, and
/// stops the run. It does not see `φ`, which in that configuration is the only thing being
/// optimized and is nowhere near converged.
///
/// Measured on warfarin with `θ`, `Ω` and `σ` all FIXed at a known-good estimate: the fit stopped
/// after **500 iterations** reporting `converged: true` with `elbo_tightness_ratio: 78` (the
/// implausibility threshold is 25) and `−2·ELBO = +2026`, on a model that reaches `−286` when `φ`
/// is allowed to converge. The objective trace had fallen from ~21 000 to ~2 900 across those 500
/// iterations and was still dropping steeply, so the *objective* criterion had correctly reported
/// "still moving" — the parameter criterion overrode it, because either is sufficient.
///
/// Pinning every population parameter and fitting only `q` is a legitimate thing to ask for (it is
/// how you read per-subject posteriors at a fixed estimate), so the fix is to recognise the
/// criterion as vacuous rather than to forbid the configuration. With no free population
/// coordinate the objective is the only convergence signal there is, and it is used alone.
///
/// The criterion now also requires `φ` to have settled, which independently defeats the
/// all-FIXed case (a moving `φ` resets the counter). This guard is kept as the explicit,
/// zero-tolerance statement of it: an `x` that cannot move must never *contribute* evidence
/// of convergence, whatever `φ` happens to be doing on a given window.
fn param_criterion_applies(template: &ModelParameters) -> bool {
    let fixed = crate::estimation::parameterization::packed_fixed_mask(template);
    let structural = crate::estimation::parameterization::omega_structural_zero_mask(template);
    fixed.iter().zip(structural.iter()).any(|(&f, &z)| !f && !z)
}

/// Whether the tail of the objective trace has stopped moving.
///
/// Compares the mean of the last `window` values against the mean of the `window`
/// before it. Averaging both sides is the point: single-iteration deltas are
/// dominated by Monte-Carlo noise and would report convergence at random.
///
/// # Why the threshold is noise-aware, not purely relative
///
/// The ELBO is a Monte-Carlo estimate, so the difference between two window means is
/// itself a random variable. Judging it against a *relative* tolerance
/// (`rel_tol·(1 + |mean|)`) asks the wrong question: it scales with the objective's
/// magnitude, which has nothing to do with how noisy the trace is. On a real fit that
/// threshold is unreachable — a plateaued warfarin run still shows window-to-window
/// differences orders of magnitude above `1e-4·(1 + 286)`, so the run is reported
/// unsettled forever and early stopping never fires.
///
/// The right comparison is against the **standard error of the difference**,
/// `√(s²_recent/w + s²_prior/w)`, estimated from the trace's own within-window
/// variance. That is scale-free, adapts automatically to `vi_mc_samples`, and asks the
/// question that matters: *is the remaining drift distinguishable from noise?* If it is
/// not, more iterations at this noise level cannot resolve it — the fix would be more
/// Monte-Carlo draws, not more epochs.
///
/// `rel_tol` is retained as an additive **floor** so the criterion is never stricter
/// than the original one: a noiseless trace (`s² = 0`) still settles on the relative
/// test alone, which is what the synthetic-trace unit tests exercise.
pub fn trace_has_settled(trace: &[f64], window: usize, rel_tol: f64) -> bool {
    if window == 0 || trace.len() < 2 * window {
        return false;
    }
    let n = trace.len();
    let recent = &trace[n - window..];
    let prior = &trace[n - 2 * window..n - window];

    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let m_recent = mean(recent);
    let m_prior = mean(prior);
    if !m_recent.is_finite() || !m_prior.is_finite() {
        return false;
    }

    // Unbiased within-window variance; a 1-wide window carries no variance information,
    // in which case this degenerates to the plain relative test below.
    let var = |s: &[f64], m: f64| -> f64 {
        if s.len() < 2 {
            return 0.0;
        }
        s.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (s.len() - 1) as f64
    };
    let se = ((var(recent, m_recent) + var(prior, m_prior)) / window as f64).sqrt();
    if !se.is_finite() {
        return false;
    }

    let tol = SETTLE_Z * se + rel_tol * (1.0 + m_recent.abs());
    (m_prior - m_recent).abs() <= tol
}

/// Blocks the trace tail is split into for the systematic-drift test below.
const DRIFT_BLOCKS: usize = 10;

/// Fraction of consecutive block-to-block steps that must move the same way for the
/// remaining drift to count as systematic rather than noise.
const DRIFT_SIGN_FRACTION: f64 = 0.8;

/// Whether the objective is still moving *systematically* despite [`trace_has_settled`].
///
/// # Why this exists
///
/// `trace_has_settled` stops the run when the drift between two windows is no longer
/// distinguishable from Monte-Carlo noise. At a low `vi_mc_samples` the noise floor is
/// high, so that becomes true while real drift remains — the run stops short and reports
/// `converged: true` at a visibly worse point. Measured on warfarin at the default 8
/// draws: `σ = 0.014150` against `0.010565` from both AGQ (`n_agq = 9`) and FOCEI, 34 %
/// high, with an OFV 11.3 units short of the reference. Because per-subject posterior
/// width scales with `σ²`, every variational covariance reported there was ~1.75× too
/// wide. Raising the draws resolves it (`32` → −284.6, `128` → −285.4 against AGQ's
/// −285.977); so does lowering `vi_lr`. Starting from a fitted FOCEI point does not.
///
/// # Why a sign test
///
/// The failure is *below* the per-window noise the settling test measures, so no single
/// window comparison can see it — that is the definition of the noise floor. What a
/// noise-limited stop and a genuine plateau differ in is not the size of each step but
/// its *sign*: noise alternates, real drift does not. Splitting the tail into
/// [`DRIFT_BLOCKS`] block means and counting how many consecutive steps move the same
/// direction extracts a trend far below the amplitude of one step, at no cost — the
/// trace is already in hand.
///
/// Deliberately one-sided and conservative: it fires only when at least
/// [`DRIFT_SIGN_FRACTION`] of the steps agree, and a fair coin clears that bar on
/// 9 steps under 5 % of the time. A run that has genuinely plateaued alternates and
/// does not trip it.
pub(crate) fn trace_still_drifting(trace: &[f64], window: usize) -> bool {
    let tail = 2 * window;
    if window == 0 || trace.len() < tail || tail < DRIFT_BLOCKS * 2 {
        return false;
    }
    let t = &trace[trace.len() - tail..];
    let per = t.len() / DRIFT_BLOCKS;
    let means: Vec<f64> = (0..DRIFT_BLOCKS)
        .map(|b| {
            let seg = &t[b * per..(b + 1) * per];
            seg.iter().sum::<f64>() / seg.len() as f64
        })
        .collect();
    if means.iter().any(|m| !m.is_finite()) {
        return false;
    }
    let steps = means.windows(2).map(|w| w[1] - w[0]);
    let (down, total) = steps.fold((0usize, 0usize), |(d, n), s| {
        (d + usize::from(s < 0.0), n + 1)
    });
    total > 0 && (down as f64) / (total as f64) >= DRIFT_SIGN_FRACTION
}

/// The `vi_kl = analytic` fallback warning, or `None` when nothing fell back.
///
/// Split out so the message is unit-testable without a variational family that lacks
/// a closed-form KL: every family shipped today has one, so `build_family` cannot
/// produce the condition and it is otherwise reachable only by driving `φ` to a
/// degenerate Cholesky factor. Mirrors `should_run_sir_fallback`, split out for the
/// same reason (#264).
pub(crate) fn kl_fallback_warning(
    n_fell_back: usize,
    n_subjects: usize,
    family: &str,
) -> Option<String> {
    (n_fell_back > 0).then(|| {
        format!(
            "VI: vi_kl = analytic was requested but variational family '{family}' has no \
             closed-form KL, so the KL was estimated by Monte Carlo for {n_fell_back} of \
             {n_subjects} subjects. The estimator is unbiased but noisier; raise \
             vi_mc_samples, or set vi_kl = mc to silence this."
        )
    })
}

/// Demote a fit whose final ELBO diagnostic says the optimizer stopped in a bad basin.
///
/// Kept separate from [`ElboTightness::is_implausible`] because that method diagnoses the
/// bound, while this one owns the user-visible consequence: a flat objective in the wrong
/// basin is not convergence.
fn bad_basin_warning(converged: &mut bool, tightness: &ElboTightness) -> Option<String> {
    if !tightness.is_implausible() {
        return None;
    }

    *converged = false;
    Some(format!(
        "W_VI_BAD_BASIN: VI: the ELBO is not a usable bound at this estimate — the data term \
         sits {:.0}x further above its value at the variational means than a posterior-shaped \
         q would put it ({:.3e} against an expected {:.3e}). The optimizer has almost \
         certainly settled in a bad basin. Reported converged: false. Re-run from better \
         starting values (for a [covariate_nn] model, declare `init` so the network starts at \
         plausible parameter values), or chain `method = [focei, vi]` to start VI from a \
         fitted point.",
        tightness.ratio(),
        tightness.excess,
        tightness.expected
    ))
}

/// Whether a generic iteration-budget warning adds information after quality checks.
fn needs_iteration_budget_warning(
    converged: bool,
    noise_floor_stop: bool,
    bad_basin_stop: bool,
) -> bool {
    !converged && !noise_floor_stop && !bad_basin_stop
}

fn build_family(kind: ViFamily, d: usize) -> Box<dyn VariationalFamily> {
    match kind {
        ViFamily::FullRank => Box::new(FullRank::new(d)),
        ViFamily::MeanField => Box::new(MeanField::new(d)),
    }
}

/// One family per subject, sized to that subject's stacked vector
/// `[η, κ₁ … κ_{K_i}]`.
///
/// Without IOV every `K_i` is zero and every family has dimension `n_eta`, so this is N
/// copies of the same thing — cheap (a family is two `usize`s) and it keeps one code
/// path rather than branching the whole driver on whether the model has kappas.
fn build_families(
    kind: ViFamily,
    n_eta: usize,
    n_kappa: usize,
    k_occasions: &[usize],
) -> Vec<Box<dyn VariationalFamily>> {
    k_occasions
        .iter()
        .map(|&k| build_family(kind, n_eta + k * n_kappa))
        .collect()
}

/// Zero the gradient at coordinates the model declares FIXed.
///
/// Adam's step for a zero gradient is exactly zero (`m` and `v` both stay at 0,
/// giving `0/(0+ε)`), so zeroing here is sufficient to pin a coordinate — no
/// projection or re-clamping is needed afterwards.
fn zero_fixed_coords(grad: &mut [f64], template: &ModelParameters, layout: &PackedLayout) {
    for (i, &fixed) in template.theta_fixed.iter().enumerate() {
        if fixed {
            grad[i] = 0.0;
        }
    }
    for (k, &fixed) in template.sigma_fixed.iter().enumerate() {
        if fixed {
            grad[layout.sigma_start() + k] = 0.0;
        }
    }
    // FIXed kappas pin their whole row and column of `Ω_iov`, matching the `fi || fj`
    // rule `closed_form_omega` applies in Ω space and `packed_fixed_mask` applies here.
    if let Some(iov) = template.omega_iov.as_ref() {
        let is_fixed = |k: usize| template.kappa_fixed.get(k).copied().unwrap_or(false);
        for (slot, (i, j)) in
            crate::estimation::parameterization::lower_tri_iter(iov.dim(), iov.diagonal).enumerate()
        {
            if is_fixed(i) || is_fixed(j) {
                grad[layout.omega_iov_start() + slot] = 0.0;
            }
        }
    }
}

/// Write `omega`'s Cholesky factor into the `Ω` block of the packed vector,
/// in `pack_params` order and transform (diagonal as `log`).
///
/// Deliberately *not* implemented as `unpack_params` → replace `Ω` →
/// `pack_params`: that round-trips `θ` and `σ` through `exp(ln(·))` on **every**
/// iteration, and the 1-ULP error each time accumulates over a thousand
/// iterations — visibly so on a FIXed parameter, which is supposed to hold its
/// declared value exactly. Touching only the `Ω` slots leaves the rest bit-exact.
fn write_omega_block(
    x: &mut [f64],
    omega: &crate::types::OmegaMatrix,
    diagonal: bool,
    start: usize,
) {
    let l = &omega.chol;
    for (slot, (i, j)) in
        crate::estimation::parameterization::lower_tri_iter(omega.dim(), diagonal).enumerate()
    {
        x[start + slot] = if i == j {
            l[(i, j)].max(1e-10).ln()
        } else {
            l[(i, j)]
        };
    }
}

/// Write both `Ω` blocks of the closed-form maximizer into the packed vector.
///
/// `Ω_iov` is `None` for a model without IOV, in which case only the BSV block moves and
/// the (empty) IOV range is untouched.
fn write_closed_form_omegas(
    x: &mut [f64],
    omega: &crate::types::OmegaMatrix,
    omega_iov: Option<&crate::types::OmegaMatrix>,
    template: &ModelParameters,
    layout: &PackedLayout,
) {
    write_omega_block(x, omega, template.omega.diagonal, layout.omega_start());
    if let (Some(iov), Some(tmpl)) = (omega_iov, template.omega_iov.as_ref()) {
        write_omega_block(x, iov, tmpl.diagonal, layout.omega_iov_start());
    }
}

/// Restore parameters the model declares FIXed to their exact declared values.
///
/// The optimizer never moves them (their gradients are zeroed), but the single
/// `pack`/`unpack` round trip between the initial estimates and the reported ones
/// is not bit-exact — `exp(ln(x)) != x` in general. A FIXed parameter that comes
/// back one ULP off its declared value is a small lie, and one that shows up in
/// output diffs, so undo it.
fn restore_fixed(params: &mut ModelParameters, template: &ModelParameters) {
    for (i, &fixed) in template.theta_fixed.iter().enumerate() {
        if fixed {
            params.theta[i] = template.theta[i];
        }
    }
    for (k, &fixed) in template.sigma_fixed.iter().enumerate() {
        if fixed {
            params.sigma.values[k] = template.sigma.values[k];
        }
    }
}

/// Run variational inference, stopping once the objective has settled or `vi_iters`
/// iterations have elapsed — whichever comes first.
pub fn run_vi(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    if let Some(reason) = unsupported_data_term_reason(model) {
        return Err(reason);
    }
    let n_subjects = population.subjects.len();
    if n_subjects == 0 {
        return Err("VI requires at least one subject".to_string());
    }

    let n_eta = model.n_eta;
    let layout = PackedLayout::new(init_params);
    let n_iters = options.vi_iters.max(1);

    let cfg = ElboConfig {
        n_mc_samples: options.vi_mc_samples.max(1),
        eta_grad: options.vi_eta_grad,
        kl: options.vi_kl,
        seed: options.vi_seed.unwrap_or(DEFAULT_VI_SEED),
    };
    let adam_cfg = AdamConfig {
        lr: options.vi_lr,
        grad_clip: options.vi_grad_clip,
        ..Default::default()
    };

    // The same packed box every other estimator optimizes inside: `θ` from the
    // model file's declared bounds, and the `Ω`/`σ` runaway guards. Adam knows
    // nothing about the box NLopt is handed for FOCE/FOCEI, and unlike SAEM's
    // M-step VI has no optimizer to hand it to — so `θ (1.0, 0.1, 10.0)` is
    // enforced by projecting `x` after each step, or it is not enforced at all.
    let bounds = compute_bounds(init_params);

    // Occasions per subject, from the data. Fixed for the run, so compute once: this is
    // what sets each subject's stacked dimension and the pooled `Ω_iov` denominator.
    let k_occasions: Vec<usize> = population
        .subjects
        .iter()
        .map(|s| super::elbo::subject_k_occasions(model, s))
        .collect();

    // One family per subject, sized to `n_eta + K_i * n_kappa`.
    let families = build_families(options.vi_family, n_eta, model.n_kappa, &k_occasions);
    let fams = Families::PerSubject(&families);

    // Start q at the prior and Θ at the model file's initial estimates: no
    // randomization, so the fit is reproducible by default. The initial estimates
    // are projected too, so `x` is inside the box from the first evaluation
    // onwards rather than only after the first step.
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    // Each subject's `q` starts at *its own* prior: the block-diagonal
    // `Σ_b = Ω ⊕ Ω_iov^{⊗K_i}`, which is `Ω` itself without IOV. Starting at the prior
    // rather than at a random point keeps the fit reproducible and makes the first
    // iteration's KL exactly zero.
    let mut phis: Vec<Vec<f64>> = (0..n_subjects)
        .map(|i| {
            let prior = stacked_prior(
                &init_params.omega,
                init_params.omega_iov.as_ref(),
                k_occasions[i],
            );
            families[i].init(&prior)
        })
        .collect();
    let mut adam_x = AdamState::new(x.len());
    let mut adam_phi: Vec<AdamState> = (0..n_subjects)
        .map(|i| AdamState::new(families[i].n_params()))
        .collect();

    // Recomputed if the run settles early (see `stop_at` below), so the Polyak window
    // always sits on the *end* of whatever run actually happened.
    let mut avg_start = averaging_start(n_iters, options.vi_avg_last);
    // Iteration count at which to stop. `n_iters` is a **ceiling**, not a budget: the
    // loop breaks as soon as the objective has settled and a full averaging window has
    // been collected after that point.
    let mut stop_at = n_iters;
    let mut settled_at: Option<usize> = None;
    let mut consecutive_settled = 0usize;
    // Parameter-stability tracking: a running block mean of `x`, compared against the
    // previous block every `CONVERGENCE_CHECK_INTERVAL` iterations.
    let mut block_acc = vec![0.0f64; x.len()];
    let mut block_n = 0usize;
    let mut prev_block: Option<Vec<f64>> = None;
    let mut consecutive_param_settled = 0usize;
    let mut param_settled = false;
    // The same block mean over the *variational* parameters, flattened across subjects.
    // `x` alone is not enough to certify convergence: `φ` is half the optimization, it is
    // never FIXed, and a population vector that settles first (or is all but pinned) would
    // otherwise stop a run whose per-subject posteriors are still moving — and those
    // posteriors are what a VI fit is for. Both must be still for this criterion to fire.
    let phi_len: usize = (0..n_subjects).map(|i| families[i].n_params()).sum();
    let mut phi_block_acc = vec![0.0f64; phi_len];
    let mut prev_phi_block: Option<Vec<f64>> = None;
    let mut avg_x = PolyakAverager::new(x.len());
    let mut avg_phi: Vec<PolyakAverager> = (0..n_subjects)
        .map(|i| PolyakAverager::new(families[i].n_params()))
        .collect();

    let mut warnings: Vec<String> = Vec::new();

    // Resolve the σ route once: support is a property of the model's error structure, not
    // of the iterate, so probing it here keeps the loop branch-free and makes "asked for
    // the closed form, had to Adam-step" a single decision to report.
    let (sigma_mstep, sigma_fallback) = match options.vi_sigma_update {
        ViSigmaUpdate::Adam => (None, None),
        ViSigmaUpdate::ClosedForm => match closed_form_sigma_support(model, init_params) {
            Ok(k) => (Some(k), None),
            Err(reason) => (None, Some(reason)),
        },
    };
    if let Some(reason) = sigma_fallback {
        warnings.push(format!(
            "VI: vi_sigma_update = closed_form does not apply to this model ({reason}), so σ is \
             stepped by Adam instead. Adam on σ is sensitive to vi_lr — if the fit lands with a σ \
             well above a FOCEI/AGQ fit of the same data, lower vi_lr."
        ));
    }
    // The stationarity condition is summed over every observation entering the data term.
    let n_obs_total = population.n_obs();
    let param_criterion_live = param_criterion_applies(init_params);
    if !param_criterion_live {
        warnings.push(
            "VI: every population parameter is FIXed, so only `q` is being optimized. Convergence \
             is judged on the objective alone — the parameter-stability test cannot say anything \
             about a vector that never moves."
                .to_string(),
        );
    }
    let mut trace: Vec<f64> = Vec::with_capacity(n_iters);
    let mut n_fd_subjects = 0usize;
    let mut n_kl_fallback_subjects = 0usize;
    let verbose = options.verbose;

    for iter in 0..n_iters {
        let eval = population_neg_elbo(
            model,
            population,
            init_params,
            &x,
            fams,
            &phis,
            &cfg,
            iter as u64,
        )?;
        // `neg_elbo` is already `−ELBO`, so `−2·ELBO` is `2·neg_elbo`. Reported on
        // the OFV scale: an upper bound on `−2 log p(y)` that *decreases* as the
        // fit improves, so a trace reads the way every other objective here does.
        trace.push(2.0 * eval.neg_elbo);
        n_fd_subjects = eval.n_fd_subjects;
        n_kl_fallback_subjects = eval.n_kl_fallback_subjects;

        if !eval.neg_elbo.is_finite() {
            return Err(format!(
                "VI objective became non-finite at iteration {iter}; try a smaller vi_lr \
                 or check the model's initial estimates"
            ));
        }

        let mut grad_x = eval.grad_x.clone();
        zero_fixed_coords(&mut grad_x, init_params, &layout);
        if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
            // Ω is set analytically below; stepping it here as well would apply the
            // same information twice and fight the closed form.
            for g in grad_x[layout.omega_start()..layout.sigma_start()].iter_mut() {
                *g = 0.0;
            }
            // Same for `Ω_iov`, which the closed form also sets. An empty range without
            // IOV.
            for g in grad_x[layout.omega_iov_start()..layout.total()].iter_mut() {
                *g = 0.0;
            }
        }
        // σ, like Ω, is replaced by its exact maximizer rather than stepped. Compute it from
        // this iteration's gradient *before* zeroing that coordinate, then apply it after the
        // Adam step so the write is not undone. Stepping and solving the same coordinate would
        // apply the same information twice, exactly as for Ω above.
        let sigma_next = sigma_mstep.and_then(|k| {
            let slot = layout.sigma_start() + k;
            closed_form_sigma(x[slot].exp(), grad_x[slot], n_obs_total)
        });
        if let Some(k) = sigma_mstep {
            grad_x[layout.sigma_start() + k] = 0.0;
        }
        adam_x.step(&mut x, &grad_x, &adam_cfg);
        if let (Some(k), Some(next)) = (sigma_mstep, sigma_next) {
            // Packed as `ln σ` (see `pack_params`). Written before `clamp_to_bounds` so the
            // maximizer still picks up the σ runaway guard every other estimator gets.
            x[layout.sigma_start() + k] = next.ln();
        }
        // Project *before* the closed-form Ω is written below, not after. The two
        // blocks want different treatment: `θ`/`σ` are stepped by Adam and need the
        // box, whereas a closed-form `Ω` is already a valid covariance by
        // construction (`floor_omega_diagonal`), and clipping it would leave `x`'s
        // Ω block disagreeing with the `final_params.omega` computed from the same
        // maximizer. Under `vi_omega_update = adam` the Ω block *is* an Adam step,
        // and this is where it picks up the same runaway guard FOCE/FOCEI gives it.
        //
        // Adam's moments are deliberately not reset when a coordinate hits a bound:
        // a gradient that keeps pushing outward accumulates in `m`, so the
        // coordinate lags slightly before leaving the bound once the gradient
        // reverses. That is the ordinary trade-off of projected Adam, and preferable
        // to discarding the curvature estimate whenever a coordinate touches an edge.
        clamp_to_bounds(&mut x, &bounds);

        for (i, phi) in phis.iter_mut().enumerate() {
            adam_phi[i].step(phi, &eval.grad_phi[i], &adam_cfg);
        }

        if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
            let (omega, omega_iov) = closed_form_omega(fams, &phis, &k_occasions, init_params);
            write_closed_form_omegas(&mut x, &omega, omega_iov.as_ref(), init_params, &layout);
        }

        if iter >= avg_start {
            avg_x.accumulate(&x);
            for (i, phi) in phis.iter().enumerate() {
                avg_phi[i].accumulate(phi);
            }
        }

        if verbose && (iter % 100 == 0 || iter + 1 == stop_at) {
            eprintln!(
                "VI iter {:>5}  -2*ELBO = {:.4}",
                iter,
                trace[trace.len() - 1]
            );
        }

        // ---- Early stopping -------------------------------------------------
        //
        // `vi_iters` is a ceiling. Checking the same settled-ness predicate the final
        // `converged` flag uses means "ran to completion" and "reported converged"
        // cannot disagree, and it keeps an easy fit at a second or two while leaving
        // head-room for a hard one.
        //
        // Detecting settling is not enough to stop *immediately*: the reported estimate
        // is a Polyak mean, so a full averaging window still has to be collected —
        // otherwise early stopping would trade under-convergence for a noisy last
        // iterate. So the first detection schedules the stop, it does not perform it.
        //
        // The rewrite is skipped once averaging is already under way (`iter >= avg_start`,
        // i.e. settling was detected very late): the window is already being collected
        // on the original schedule and moving it would mix pre- and post-settling
        // iterates into one mean.
        for (a, v) in block_acc.iter_mut().zip(x.iter()) {
            *a += *v;
        }
        for (a, v) in phi_block_acc
            .iter_mut()
            .zip(phis.iter().flat_map(|p| p.iter()))
        {
            *a += *v;
        }
        block_n += 1;

        if settled_at.is_none() && iter < avg_start && (iter + 1) % CONVERGENCE_CHECK_INTERVAL == 0
        {
            if trace_has_settled(&trace, settle_window(trace.len()), CONVERGENCE_REL_TOL) {
                consecutive_settled += 1;
            } else {
                // Must be *consecutive*: one settled window followed by a moving one
                // means the run had not settled, so the count starts over.
                consecutive_settled = 0;
            }

            // Parameter stability, on the same cadence and with the same patience.
            let block: Vec<f64> = block_acc.iter().map(|a| a / block_n as f64).collect();
            let phi_block: Vec<f64> = phi_block_acc.iter().map(|a| a / block_n as f64).collect();
            if let (Some(prev), Some(prev_phi)) = (prev_block.as_ref(), prev_phi_block.as_ref()) {
                if estimates_have_settled(&block, prev, &phi_block, prev_phi) {
                    consecutive_param_settled += 1;
                } else {
                    consecutive_param_settled = 0;
                }
            }
            prev_block = Some(block);
            prev_phi_block = Some(phi_block);
            block_acc.iter_mut().for_each(|a| *a = 0.0);
            phi_block_acc.iter_mut().for_each(|a| *a = 0.0);
            block_n = 0;
            // Guarded: with no free population coordinate this criterion is vacuous and must
            // not certify convergence. See `param_criterion_applies`.
            if consecutive_param_settled >= SETTLE_PATIENCE && param_criterion_live {
                param_settled = true;
            }

            // Either criterion is sufficient. The objective test is the stricter one on a
            // well-conditioned model and is what the existing behaviour is pinned to; the
            // parameter test is what rescues a flat parameter space, where the objective
            // never settles but the estimates have.
            if consecutive_settled >= SETTLE_PATIENCE || param_settled {
                let avg_window = options
                    .vi_avg_last
                    .unwrap_or_else(|| averaging_window_for(iter + 1))
                    .max(1);
                let candidate = iter + 1 + avg_window;
                if candidate < n_iters {
                    settled_at = Some(iter);
                    avg_start = iter + 1;
                    stop_at = candidate;
                }
            }
        }
        if iter + 1 >= stop_at {
            break;
        }
    }
    let n_iters_run = trace.len();

    // The reported estimate is the Polyak mean, not the last iterate. Averaging
    // projected iterates of a convex box cannot leave it, so this clamp only
    // absorbs the rounding of the mean itself — but it makes "the reported θ is
    // inside its declared bounds" hold unconditionally rather than by argument.
    let mut x_final = avg_x.mean().unwrap_or_else(|| x.clone());
    clamp_to_bounds(&mut x_final, &bounds);
    let phis_final: Vec<Vec<f64>> = avg_phi
        .iter()
        .zip(phis.iter())
        .map(|(a, last)| a.mean().unwrap_or_else(|| last.clone()))
        .collect();
    let mut final_params = unpack_params(&x_final, init_params);
    // σ needs no recompute here, unlike Ω. Ω is a function of `phis_final`, so it has to be
    // rebuilt at the averaged φ; σ lives *in* `x`, where every iterate written above was
    // already an exact maximizer, so the Polyak mean of `x` carries the mean of those
    // maximizers — which is what `vi_avg_last` exists to report. Re-solving here would also
    // need a gradient at the averaged point, and applying it after `final_eval` would leave
    // the reported ELBO evaluated at a σ the fit no longer holds.
    if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
        let (omega, omega_iov) = closed_form_omega(fams, &phis_final, &k_occasions, init_params);
        final_params.omega = omega;
        if omega_iov.is_some() {
            final_params.omega_iov = omega_iov;
        }
    }
    restore_fixed(&mut final_params, init_params);

    // Re-evaluate at the averaged point with more draws: this number is reported,
    // so it should carry less Monte-Carlo noise than a training iteration.
    let report_cfg = ElboConfig {
        n_mc_samples: (cfg.n_mc_samples * 32).max(32),
        ..cfg.clone()
    };
    let final_eval = population_neg_elbo(
        model,
        population,
        init_params,
        &pack_params(&final_params),
        fams,
        &phis_final,
        &report_cfg,
        n_iters as u64,
    )?;

    // The variational moments — VI's own output, reported on `ViResult`.
    //
    // Under IOV `μ` spans the stacked vector, so the reported `η` moments are its BSV
    // head and the per-occasion `κ` means are read off the blocks behind it. `warm` keeps
    // only the `η` part: it seeds the EBE search below, whose own `κ` solve is separate.
    let mut eta_means: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
    let mut eta_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_subjects);
    let mut kappa_means: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_subjects);
    let mut warm: Vec<DVector<f64>> = Vec::with_capacity(n_subjects);
    let n_kappa = model.n_kappa;
    for (i, phi) in phis_final.iter().enumerate() {
        let (mu, s) = families[i].moments(phi);
        eta_means.push(mu.iter().take(n_eta).copied().collect());
        eta_covs.push(
            (0..n_eta)
                .map(|a| (0..n_eta).map(|b| s[(a, b)]).collect())
                .collect(),
        );
        kappa_means.push(
            (0..k_occasions[i])
                .map(|g| {
                    let off = n_eta + g * n_kappa;
                    (0..n_kappa).map(|c| mu[off + c]).collect()
                })
                .collect(),
        );
        warm.push(DVector::from_iterator(
            n_eta,
            mu.iter().take(n_eta).copied(),
        ));
    }

    // Downstream diagnostics — CWRES, IWRES, shrinkage, sdtab — are all defined in
    // terms of the conditional **mode** and the `n_obs × n_eta` sensitivity matrix
    // `∂f/∂η` (`OuterResult::h_matrices`, NONMEM's "H"). Reporting variational
    // means as `eta_hat` while sourcing `H` elsewhere would make those diagnostics
    // mutually inconsistent, so one inner-loop pass at the converged estimate
    // produces both. It is cheap: warm-starting from `μ` lands the EBE search
    // essentially on top of its answer.
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _, kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm),
        Some(&final_mu_k),
        0,
        0,
    );

    // Judged on the run that actually happened, not on the `vi_iters` ceiling — under
    // early stopping those differ, and windowing a 900-iteration trace as though it were
    // 25 000 long would make `trace_has_settled` return `false` for every early stop.
    let mut converged = settled_at.is_some()
        || param_settled
        || trace_has_settled(&trace, settle_window(n_iters_run), CONVERGENCE_REL_TOL);

    // A stop the noise floor caused rather than the optimum. `trace_has_settled` asks
    // whether the remaining drift is distinguishable from Monte-Carlo noise; at a low
    // `vi_mc_samples` the answer turns "no" while the objective is still falling, and the
    // fit is reported successful well short of the reference (see `trace_still_drifting`).
    // The parameter criterion is exempt: it fires on a flat direction the objective can
    // never settle on, which is the case it exists for, and there the trend is real but
    // irrelevant. A warning either way — the numbers are not wrong, they are unfinished —
    // but `converged` must not say otherwise.
    let mut noise_floor_stop = false;
    if converged && !param_settled && trace_still_drifting(&trace, settle_window(n_iters_run)) {
        converged = false;
        noise_floor_stop = true;
        warnings.push(format!(
            "VI: the objective stopped because its drift fell below the Monte-Carlo noise \
             floor at vi_mc_samples = {}, not because it reached the optimum — the ELBO trace \
             is still falling systematically. Reported converged: false. Raise vi_mc_samples \
             (32 recovers the FOCEI/AGQ reference on warfarin, where the default 8 leaves sigma \
             34% high and the OFV 11.3 units short), or lower vi_lr.",
            options.vi_mc_samples
        ));
    }
    // Is the bound we are about to report actually tight? A stuck optimizer produces a
    // flat trace and stable parameters — indistinguishable, to the convergence test, from
    // a converged one. This is the check that tells them apart, and it is the difference
    // between reporting a bad fit as successful and saying so.
    let tightness = super::elbo::elbo_tightness(
        model,
        population,
        &final_params,
        final_eval.data_term,
        &eta_means,
        &kappa_means,
    );
    let bad_basin_stop = if let Some(warning) = bad_basin_warning(&mut converged, &tightness) {
        warnings.push(warning);
        true
    } else {
        false
    };

    // Skipped when either quality check just demoted `converged`: those warnings already
    // say what happened, and more precisely — "increase vi_iters" is wrong advice for a
    // noise-limited stop or a fit trapped in a bad basin.
    if needs_iteration_budget_warning(converged, noise_floor_stop, bad_basin_stop) {
        warnings.push(format!(
            "VI: neither the objective nor the parameter estimates had settled after \
             {n_iters_run} iterations (see vi.elbo_trace). Increase vi_iters, or lower \
             vi_lr if the trace is oscillating."
        ));
    }
    if n_fd_subjects > 0 {
        warnings.push(format!(
            "VI: {n_fd_subjects} of {n_subjects} subjects used finite-difference \
             eta-gradients because the analytic provider declined them. The fit is correct \
             but much slower than it needs to be."
        ));
    }
    if let Some(w) = kl_fallback_warning(n_kl_fallback_subjects, n_subjects, families[0].label()) {
        warnings.push(w);
    }

    // The ELBO is a lower bound, so it is never reported as the OFV. See
    // `ViFinalOfv` for why, and for how to obtain a real marginal likelihood.
    let ofv = match options.vi_final_ofv {
        ViFinalOfv::None => {
            warnings.push(
                "VI: `ofv` is NaN because the ELBO is a lower bound on the log likelihood, not \
                 a −2 log L, and is not comparable with a FOCE/SAEM OFV. Set \
                 `vi_final_ofv = laplace`, or chain `methods = vi, imp` with \
                 `imp_eval_only = true`, to evaluate a genuine marginal likelihood at the VI \
                 estimate. The bound itself is on `vi.neg_two_elbo`."
                    .to_string(),
            );
            f64::NAN
        }
        // Reuses the EBEs and sensitivities already converged above, so requesting
        // an OFV costs only the objective evaluation itself.
        ViFinalOfv::Laplace => {
            2.0 * pop_nll(
                model,
                population,
                &final_params,
                &eta_hats,
                &h_matrices,
                &kappas,
                options.interaction,
            )
        }
    };

    // ---- Covariance step ----
    //
    // The same FD-of-OFV Hessian every other estimator uses, evaluated at the VI
    // estimate with the EBEs and `H` already converged above — so it is a *Laplace*
    // covariance at the VI point, directly comparable with the one a FOCE/FOCEI or
    // SAEM fit reports. Deliberately **not** built from `vi.eta_covs`: those are
    // per-subject *posterior* variances, and variational posteriors are known to
    // understate them, so they are not a route to population standard errors.
    //
    // It runs independently of `vi_final_ofv`: the covariance is the curvature of
    // the Laplace objective, not of the ELBO, so it is well-defined even when the
    // reported `ofv` is deliberately left `NaN`. `run_covariance_step` gates itself
    // on `options.run_covariance_step`, which `fit_inner` clears on every
    // non-terminal stage of a chain — so a `methods = vi, focei` run pays for it
    // once, at the end.
    let packed_final = pack_params(&final_params);
    let cov_out = crate::estimation::covariance::run_covariance_step(
        &packed_final,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
    } = cov_out;
    warnings.extend(cov_warnings);

    let vi_result = ViResult {
        neg_two_elbo: 2.0 * final_eval.neg_elbo,
        data_term: final_eval.data_term,
        kl_term: final_eval.kl_term,
        n_iterations: n_iters_run,
        converged,
        family: families[0].label().to_string(),
        n_mc_samples: cfg.n_mc_samples,
        // What actually ran, not what was asked for: a family with no closed-form KL
        // is sampled even under `vi_kl = analytic`.
        kl: if n_kl_fallback_subjects > 0 || options.vi_kl == crate::types::ViKl::Mc {
            "mc"
        } else {
            "analytic"
        }
        .to_string(),
        n_kl_fallback_subjects,
        elbo_trace: trace,
        eta_means,
        eta_covs,
        kappa_means,
        n_fd_subjects,
        elbo_tightness_ratio: tightness.ratio(),
        // Set by `fit_inner`, which is the only place that can see the rest of the chain;
        // `run_vi` has been handed a blanked per-stage option set and knows nothing of it.
        superseded_by: None,
    };

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged,
        n_iterations: n_iters_run,
        eta_hats,
        h_matrices,
        kappas,
        covariance_matrix,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient: None,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        // The exact packed vector the covariance step above ran at, so a later
        // standalone `run_covariance` reproduces it bit-for-bit instead of
        // re-deriving `chol(Ω)` from `L·Lᵀ` (#816 follow-up).
        packed_estimate: Some(packed_final),
        vi: Some(vi_result),
        mixture_posteriors: None,
    })
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
