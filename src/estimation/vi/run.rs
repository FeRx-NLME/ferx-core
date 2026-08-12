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
//! reported `converged` flag, checked every [`CONVERGENCE_CHECK_INTERVAL`] iterations.
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
    ViResult,
};

use super::adam::{averaging_start, AdamConfig, AdamState, PolyakAverager};
use super::elbo::{
    closed_form_omega, population_neg_elbo, stacked_prior, unsupported_data_term_reason,
    ElboConfig, Families, PackedLayout,
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
    let mut avg_x = PolyakAverager::new(x.len());
    let mut avg_phi: Vec<PolyakAverager> = (0..n_subjects)
        .map(|i| PolyakAverager::new(families[i].n_params()))
        .collect();

    let mut warnings: Vec<String> = Vec::new();
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
        adam_x.step(&mut x, &grad_x, &adam_cfg);
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
        if settled_at.is_none() && iter < avg_start && (iter + 1) % CONVERGENCE_CHECK_INTERVAL == 0
        {
            if trace_has_settled(&trace, settle_window(trace.len()), CONVERGENCE_REL_TOL) {
                consecutive_settled += 1;
            } else {
                // Must be *consecutive*: one settled window followed by a moving one
                // means the run had not settled, so the count starts over.
                consecutive_settled = 0;
            }
            if consecutive_settled >= SETTLE_PATIENCE {
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
    let converged = settled_at.is_some()
        || trace_has_settled(&trace, settle_window(n_iters_run), CONVERGENCE_REL_TOL);
    if !converged {
        warnings.push(format!(
            "VI: the objective was still moving at the end of {n_iters_run} iterations \
             (see vi.elbo_trace). Increase vi_iters, or lower vi_lr if the trace is oscillating."
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
