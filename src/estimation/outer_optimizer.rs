use crate::estimation::inner_optimizer::{find_ebe, run_inner_loop_warm, InnerLoopStats};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{foce_population_nll, foce_population_nll_iov};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
// `SymmetricEigen` is used only by this module's `#[cfg(test)]` code (the non-PD
// fallback tests); gate the import so a non-test build doesn't flag it unused.
#[cfg(test)]
use nalgebra::SymmetricEigen;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// The covariance/SE subsystem moved to `estimation::covariance` (refactor T4). The
// re-export exists only so this module's `#[cfg(test)] mod tests` can reach
// `compute_covariance` / `CovarianceStepResult` via `super::*`; no non-test code uses
// the historical `outer_optimizer::{..}` path, so gate it to avoid an unused-import
// warning in a non-test build.
#[cfg(test)]
pub(crate) use crate::estimation::covariance::{compute_covariance, CovarianceStepResult};

/// Result of outer optimization
/// Per-subject mixture posteriors lifted out of a converged `[mixture]` fit
/// (#977 Phase 5), threaded onto `SubjectResult.pmix` / `.mixest` in postfit.
pub struct MixturePosteriors {
    /// `PMIX_ik` — posterior class-membership probabilities per subject (each of
    /// length `K`), in subject order.
    pub pmix: Vec<Vec<f64>>,
    /// `MIXEST_i` — argmax-posterior class per subject, **0-based** (converted to
    /// the 1-based NONMEM convention when written onto `SubjectResult`).
    pub mixest: Vec<usize>,
}

pub struct OuterResult {
    pub params: ModelParameters,
    pub ofv: f64,
    pub converged: bool,
    pub n_iterations: usize,
    pub eta_hats: Vec<DVector<f64>>,
    pub h_matrices: Vec<DMatrix<f64>>,
    /// Per-occasion kappa EBEs for each subject. Empty vecs when `n_kappa == 0`.
    pub kappas: Vec<Vec<DVector<f64>>>,
    pub covariance_matrix: Option<DMatrix<f64>>,
    /// Wall-clock time spent inside this stage's covariance-step block
    /// (`compute_covariance` / SIR-fallback construction), in seconds.
    /// `0.0` when `run_covariance_step` was false for this stage.
    pub covariance_wall_time_secs: f64,
    pub warnings: Vec<String>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    /// Number of subjects that used HMC at least once during the SAEM E-step.
    /// `None` when `n_leapfrog = 0` (MH-only run) or for non-SAEM methods.
    pub saem_n_subjects_hmc: Option<usize>,
    pub ebe_convergence_warnings: u32,
    pub max_unconverged_subjects: u32,
    pub total_ebe_fallbacks: u32,
    /// Gradient at the best-OFV parameter point in packed space (log-theta,
    /// Cholesky-omega, log-sigma). `Some` for NLopt gradient-based runs
    /// (SLSQP, L-BFGS, MMA) when at least one gradient-requesting iteration
    /// improved the OFV; `None` for BOBYQA, built-in BFGS, GN, and SAEM.
    pub final_gradient: Option<Vec<f64>>,
    /// Fallback proposal covariance for the SIR sampler, set when the FD
    /// Hessian is non-PD. Built from the `|eigenvalue|`-rectified free-block
    /// Hessian, inflated 4×, and embedded into the full packed parameter space.
    /// `None` when the Hessian succeeded or the covariance step was skipped.
    pub sir_fallback_proposal: Option<DMatrix<f64>>,
    /// Per-iteration parameter trace from IMPMAP. `None` for all other methods.
    pub impmap_trace: Option<crate::types::ImpmapTrace>,
    /// Posterior summaries + diagnostics from a Bayesian (`method=bayes`) run.
    /// `Some` only for `EstimationMethod::Bayes`; `None` for all point
    /// estimators. Carried here so the chain dispatch can lift it onto
    /// `FitResult.bayes` through the generic OuterResult → FitResult path.
    pub bayes: Option<crate::types::BayesResult>,
    /// Per-subject conditional distribution of the random effects, estimated by
    /// the post-fit SAEM conditional-distribution pass. `Some` only when
    /// `method = saem` and `saem_conddist = true`; `None` for every other
    /// estimator and for SAEM runs that did not request the pass (#257).
    pub cond_dist: Option<CondDist>,
    /// The optimizer's **exact** final packed parameter vector (log-theta,
    /// Cholesky-omega lower triangle, log-sigma, over the free parameters) — the
    /// same vector this stage's inline covariance step used. `Some` for every
    /// packed-Cholesky-space optimizer — BOBYQA/SLSQP/MMA (NLopt), the hand-rolled
    /// BFGS, the trust region, and Gauss-Newton (pure and hybrid) — including
    /// `outer_maxiter = 0` evaluation. `None` for SAEM and importance-sampling,
    /// whose covariance step rebuilds `omega` from the reported matrix (so its
    /// Cholesky factor re-decomposes identically on both the inline and
    /// `run_covariance` paths and already agrees), and for Bayes (no Hessian
    /// covariance step).
    ///
    /// Carried so `run_covariance` can reproduce the inline FD-Hessian bit-for-bit
    /// by reusing this exact Cholesky factor instead of re-decomposing `omega`
    /// (`omega → chol` is not the round-trip inverse of the stored `L·Lᵀ`, and the
    /// FD Hessian amplifies the difference on ill-conditioned ω directions).
    pub packed_estimate: Option<Vec<f64>>,
    /// Per-subject mixture posteriors from the final mixture eval. `Some` only for
    /// a converged `[mixture]` fit (#977); `None` for every non-mixture path.
    pub mixture_posteriors: Option<MixturePosteriors>,
}

/// Run the outer optimization loop (population parameter estimation).
pub fn optimize_population(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    // `outer_maxiter == 0` requests an evaluation-only run (NONMEM `MAXEVAL=0`):
    // report the objective at the *initial* parameters with no minimisation. This
    // must short-circuit before any optimizer is constructed, because NLopt's
    // `set_maxeval(0)` means "no limit", not "zero evaluations" — so the gradient
    // NLopt path (`NloptLbfgs`/`Slsqp`/`Mma`) would otherwise run a *full fit* on
    // a `maxiter = 0` request, making the reported OFV an optimizer- and
    // platform-dependent converged value rather than the deterministic θ₀ check
    // callers expect (see #562: ferx-r's `settings = list(maxiter = 0)` silently
    // ran to convergence, so `two_cpt_oral_cov_ode`'s "init" OFV diverged ~534
    // from its analytical sibling on x86 Linux). BOBYQA and the built-in BFGS loop
    // already honour 0, but routing every optimizer through one eval-only path
    // keeps the semantics uniform.
    if options.outer_maxiter == 0 {
        return evaluate_at_initial_params(model, population, init_params, options);
    }
    // Resolve `auto` once, here, so the concrete optimizer flows through the rest
    // of the outer loop (optimize_nlopt re-reads `options.optimizer` for its own
    // branching). Every other variant is returned unchanged, so this is a no-op
    // unless the user left the default `auto` in place.
    // Mixture models (#977). BOBYQA (derivative-free) is the default and safe
    // choice — robust against the mixture's label-switching multimodality. Since
    // Phase 4 an analytic posterior-weighted outer gradient exists, so a user who
    // explicitly picks an NLopt *gradient* optimizer (SLSQP / L-BFGS / MMA) is
    // honoured — those route through `optimize_nlopt`, whose objective closure
    // branches to `mixture_gradient`. Every other choice (including `auto`, and
    // the built-in BFGS / trust-region paths, which do not carry the mixture
    // objective) falls back to BOBYQA.
    let mut optimizer_downgrade_warning: Vec<String> = Vec::new();
    let resolved = if init_params.mixture.is_some() {
        match options.optimizer {
            Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma => options.optimizer,
            // `Auto` is the mixture default and downgrades silently by design;
            // any *explicitly* chosen optimizer that the mixture path can't drive
            // (built-in BFGS/L-BFGS, trust-region, Gauss-Newton — none carry the
            // mixture objective) is run under BOBYQA instead, so say so rather than
            // dropping the choice invisibly.
            Optimizer::Auto => Optimizer::Bobyqa,
            other => {
                optimizer_downgrade_warning.push(format!(
                    "Mixture models are optimized with BOBYQA or an NLopt gradient method \
                     (SLSQP / L-BFGS / MMA); the requested {other:?} optimizer does not carry \
                     the mixture objective and was replaced by BOBYQA."
                ));
                Optimizer::Bobyqa
            }
        }
    } else {
        options.optimizer.resolve_auto(model, options.interaction)
    };
    let owned_opts;
    let options = if resolved == options.optimizer {
        options
    } else {
        owned_opts = FitOptions {
            optimizer: resolved,
            ..options.clone()
        };
        &owned_opts
    };
    // Pre-flight flat-theta guard (#826): a non-fixed theta whose outer gradient is
    // identically ~0 at the initial estimate never reaches the objective (typically
    // unmapped / dropped from the structural or scaling model). Left in the optimized
    // vector it gives a zero search direction that makes gradient NLopt return
    // `Failure` on eval 1, pinning *every* parameter at its initial value. Freeze such
    // thetas (treat as FIX) and warn, so the remaining parameters optimize normally.
    // Skip the flat-theta pre-flight for mixture models: it probes the single-
    // population outer gradient, which is not the mixture objective's gradient and
    // would mis-freeze the mixing-logit thetas.
    let frozen_params;
    let (init_params, preflight_warnings) = if init_params.mixture.is_some() {
        (init_params, Vec::new())
    } else {
        match freeze_flat_thetas(model, population, init_params, options) {
            Some((fp, w)) => {
                frozen_params = fp;
                (&frozen_params, w)
            }
            None => (init_params, Vec::new()),
        }
    };

    let mut result = match resolved {
        // `Auto` is resolved away above; group it with the NLopt path defensively.
        Optimizer::Slsqp
        | Optimizer::NloptLbfgs
        | Optimizer::Mma
        | Optimizer::Bobyqa
        | Optimizer::Auto => optimize_nlopt(model, population, init_params, options),
        Optimizer::Bfgs | Optimizer::Lbfgs => {
            optimize_bfgs(model, population, init_params, options)
        }
        Optimizer::TrustRegion => crate::estimation::trust_region::optimize_trust_region(
            model,
            population,
            init_params,
            options,
        ),
    };
    // Surface the optimizer-downgrade and freeze warnings ahead of the optimizer's
    // own (they explain the substituted optimizer / why a parameter was held fixed,
    // which the reader wants before any convergence notes).
    if !optimizer_downgrade_warning.is_empty() || !preflight_warnings.is_empty() {
        let mut w = optimizer_downgrade_warning;
        w.extend(preflight_warnings);
        w.append(&mut result.warnings);
        result.warnings = w;
    }
    result
}

/// Pre-flight flat-theta detection (#826). Computes the outer gradient at the
/// initial estimate and flags any non-fixed theta whose gradient is negligible
/// relative to the largest theta gradient, then **confirms** each candidate with a
/// perturbation probe — only a theta that leaves the reconverged objective exactly
/// unchanged when moved is truly unmapped and gets frozen. (A near-zero *initial*
/// gradient alone is not sufficient: an identifiable theta can have a
/// coincidentally-tiny gradient at the start point, and freezing it there biases
/// the whole fit — see the probe comment below.)
///
/// Returns `None` when nothing is flat (the common case; the caller keeps the
/// borrowed `init_params` untouched), or `Some((frozen, warnings))` with a
/// modified clone whose `theta_fixed` marks the flat thetas so the rest of the
/// pipeline treats them as FIX — the graceful-degradation the issue asks for
/// instead of the whole fit dying on an eval-1 `Failure`.
fn freeze_flat_thetas(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Option<(ModelParameters, Vec<String>)> {
    let n_theta = init_params.theta.len();
    // Nothing to freeze if every theta is already fixed.
    if (0..n_theta).all(|i| init_params.theta_fixed.get(i).copied().unwrap_or(false)) {
        return None;
    }

    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let params = unpack_params(&x, init_params);
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    // One cold inner solve, then one outer-gradient eval — the same pair the first
    // optimizer iteration would run, so the pre-flight costs roughly one iteration.
    let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let cold_etas = vec![DVector::zeros(n_eta); n_subj];
    let (ehs, hms, _stats, kappas) = run_inner_loop_warm(
        model,
        population,
        &params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cold_etas),
        Some(&mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
    );
    let mut grad_eval_idx = 0usize;
    let grad = population_gradient(
        &x,
        n_subj,
        init_params,
        model,
        population,
        &ehs,
        &hms,
        &kappas,
        &bounds,
        options,
        &mut grad_eval_idx,
    );

    // Thetas are the first `n_theta` packed coordinates (see `pack_params`), so
    // `grad[i]` is d(OFV)/d(coord_i) for theta `i`. A structurally flat theta has an
    // *identically* zero analytic sensitivity, so its gradient is ~0 to machine
    // precision — freeze only that near-zero case, never a merely weakly-identified
    // param (small-but-nonzero; covariance flags those as high RSE).
    const FLAT_ABS: f64 = 1e-8;
    const FLAT_REL: f64 = 1e-6;
    let is_free = |i: usize| !init_params.theta_fixed.get(i).copied().unwrap_or(false);
    let g_at = |i: usize| grad.get(i).copied().unwrap_or(0.0).abs();
    // Reference scale for "negligible". If the whole theta gradient is ~0 the
    // objective is globally flat (no data reaches it) — a different pathology, so
    // bail rather than freeze every parameter.
    let g_max = (0..n_theta)
        .filter(|&i| is_free(i))
        .map(g_at)
        .fold(0.0_f64, f64::max);
    if g_max <= FLAT_ABS {
        return None;
    }

    let candidates: Vec<usize> = (0..n_theta)
        .filter(|&i| is_free(i) && g_at(i) <= FLAT_ABS && g_at(i) <= FLAT_REL * g_max)
        .collect();
    if candidates.is_empty() {
        return None;
    }

    // Confirm structural flatness with a perturbation probe before freezing. A
    // near-zero gradient at the *initial* estimate is necessary but not sufficient
    // for "unmapped": a genuinely-identifiable theta can have a coincidentally-tiny
    // gradient there — e.g. an event-model hazard baseline `H0` evaluated at
    // `BETA = 0`, where `hazard = H0·exp(0)` is momentarily flat in the coupling
    // term, and whose ODE-path outer gradient is finite-differenced (#826 froze
    // such an `H0` at its wrong initial value and biased the whole joint PK-TTE
    // fit). A truly unmapped theta leaves the reconverged objective *exactly*
    // unchanged when moved; an identifiable one moves it. Only freeze the former —
    // freezing an identifiable theta pins it at its initial value and biases every
    // other estimate. The probe costs one inner solve per candidate (candidates
    // are rare), on top of the pre-flight gradient eval.
    let base_ofv = 2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
    let reconverged_ofv = |theta_i: usize, value: f64| -> f64 {
        let mut p = params.clone();
        p.theta[theta_i] = value;
        let mu = compute_mu_k(model, &p.theta, options.mu_referencing);
        let (e, h, _s, k) = run_inner_loop_warm(
            model,
            population,
            &p,
            options.inner_maxiter,
            options.inner_tol,
            Some(&cold_etas),
            Some(&mu),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        2.0 * pop_nll_opts(model, population, &p, &e, &h, &k, options)
    };
    // True ⇒ moving this theta changes the objective ⇒ it is identifiable, not flat.
    let moves_objective = |i: usize| -> bool {
        let ti = params.theta[i];
        let (lo, hi) = (init_params.theta_lower[i], init_params.theta_upper[i]);
        let delta = (ti.abs() * 0.5).max((hi - lo).abs() * 0.1).max(1e-2);
        let mut probe = ti + delta;
        if !(probe < hi) {
            probe = ti - delta;
        }
        if !(probe > lo) {
            probe = 0.5 * (lo + hi);
        }
        // Could not build a distinct in-bounds probe — do not freeze (safe side).
        if (probe - ti).abs() <= 1e-12 {
            return true;
        }
        (reconverged_ofv(i, probe) - base_ofv).abs() > 1e-6 * (1.0 + base_ofv.abs())
    };
    let flat: Vec<usize> = candidates
        .into_iter()
        .filter(|&i| !moves_objective(i))
        .collect();
    if flat.is_empty() {
        return None;
    }

    let mut frozen = init_params.clone();
    let mut warnings = Vec::new();
    for &i in &flat {
        // Pin the FIX at the *clamped* value the gradient was actually evaluated at
        // (`params.theta`, not the raw `init_params.theta`): an out-of-bounds initial
        // theta must not be frozen — nor reported — outside its declared bounds, and
        // the printed value must match the point the flatness was decided at.
        frozen.theta[i] = params.theta[i];
        frozen.theta_fixed[i] = true;
        let name = init_params
            .theta_names
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or("<theta>");
        warnings.push(format!(
            "[parameters] `{name}` has no effect on the objective (gradient ≈ 0 at the \
             initial estimate) — it is likely computed but never used (unmapped, or dropped \
             from the structural / scaling model). Freezing it at its initial value ({val}) \
             so the remaining parameters can be estimated; map or remove `{name}` to silence \
             this.",
            val = params.theta[i],
        ));
    }
    Some((frozen, warnings))
}

/// Evaluate the population objective at the initial parameters without running
/// the outer optimizer (`outer_maxiter == 0`, NONMEM `MAXEVAL=0` semantics).
///
/// Runs one inner EBE solve per subject from a cold start (η = 0), reports the
/// FOCE/FOCEI objective `2 · pop_nll` at the initial parameters, and — when
/// requested — the covariance step at that point. `converged` is `false` (no
/// minimisation was attempted) and `n_iterations` is 0; the generic
/// "did not converge" warning is intentionally *not* emitted, since an eval-only
/// run is a request, not a failure. This is the single eval-only entry every
/// outer optimizer routes through, so `maxiter = 0` is deterministic regardless
/// of which optimizer `auto` would have picked.
fn evaluate_at_initial_params(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let params = unpack_params(&x, init_params);

    // Mixture (#977 Phase 3): the eval-only OFV is the K-fold log-sum-exp at the
    // initial parameters; EBEs reported are the MIXEST class per subject.
    let is_mixture = params.mixture.is_some();
    let (eta_hats, h_matrices, kappas, ofv, mixture_posteriors) = if is_mixture {
        let m = crate::estimation::mixture::mixture_ofv(model, population, &params, options, None);
        // Carry PMIX/MIXEST like the converged path so an eval-only mixture run
        // still emits the PMIX_*/MIXEST sdtab columns (#977).
        let posteriors = MixturePosteriors {
            pmix: m.pmix,
            mixest: m.mixest,
        };
        (
            m.mixest_etas,
            m.mixest_h_mats,
            Vec::new(),
            m.ofv,
            Some(posteriors),
        )
    } else {
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        // Genuine cold start: an eval-only run has no warm EBE history, so pass
        // `None` (not `Some(zeros)`). Both seed the inner search at η = 0, but `None`
        // is what marks this as a cold start to the guarded multi-start inner EBE
        // (`inner_restarts`), so a subject with a multimodal individual posterior —
        // system resets / TV-covariates, or a weakly-identified random effect (#891)
        // — is re-seeded here instead of silently reporting a sub-optimal mode. This
        // is the scenario #891's evidence is drawn from (NONMEM `MAXEVAL=0`).
        let (eta_hats, h_matrices, _, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            None,
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        let ofv = 2.0
            * pop_nll_opts(
                model,
                population,
                &params,
                &eta_hats,
                &h_matrices,
                &kappas,
                options,
            );
        (eta_hats, h_matrices, kappas, ofv, None)
    };

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, ofv);
        eprintln!("outer_maxiter = 0: evaluation only, no optimization performed.");
    }

    let mut warnings = Vec::new();
    // Mixture (#983 Phase 6): the covariance step now builds its FD Hessian on the
    // K-fold mixture OFV (`compute_covariance` branches on `template.mixture`), so
    // it runs for mixtures exactly like the single-population path.
    let (covariance_matrix, covariance_wall_time_secs, sir_fallback_proposal) = {
        let out = crate::estimation::covariance::run_covariance_step(
            &x,
            init_params,
            model,
            population,
            &eta_hats,
            &h_matrices,
            &kappas,
            options,
            options.verbose.then_some("Computing covariance matrix..."),
        );
        let crate::estimation::covariance::CovStepOutcome {
            matrix,
            wall_time_secs,
            warnings: cov_warnings,
            sir_fallback_proposal,
        } = out;
        warnings.extend(cov_warnings);
        (matrix, wall_time_secs, sir_fallback_proposal)
    };

    OuterResult {
        // Evaluation-only (`outer_maxiter = 0`): no optimizer ran, but the eval
        // still packs the init in Cholesky space and the inline covariance step
        // above used `&x` as its FD center. Carry that exact vector so a later
        // `run_covariance` reproduces this step bit-for-bit — the re-decomposition
        // fallback (`chol(L·Lᵀ) ≠ L`) would otherwise diverge on an ill-conditioned
        // init omega just as it does for a converged fit (#816 follow-up).
        packed_estimate: Some(x.clone()),
        mixture_posteriors,
        params,
        ofv,
        converged: false,
        n_iterations: 0,
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
    }
}

/// Warm-started variant: starts from given EBEs and H-matrices instead of zeros.
/// Used by the Gauss-Newton hybrid to polish from the GN result.
pub fn optimize_population_warm(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
    warm_etas: &[DVector<f64>],
    warm_h_mats: &[DMatrix<f64>],
) -> OuterResult {
    // For now, delegate to the standard path — the inner loop warm-starts
    // from the provided EBEs automatically via the NloptState initialization.
    // TODO: pass warm_etas into the NLopt state directly for tighter coupling.
    let _ = (warm_etas, warm_h_mats);
    optimize_population(model, population, init_params, options)
}

// ═══════════════════════════════════════════════════════════════════════════
//  NLopt-based outer optimizer (matches Julia's NLopt path exactly)
// ═══════════════════════════════════════════════════════════════════════════

/// Identity-Hessian first-step overshoot guard for the scaled gradient.
///
/// NLopt LD_SLSQP and LD_LBFGS both start each fit with their quasi-Newton
/// Hessian set to identity, so the first search direction is the unconstrained
/// `d = -∇f`, projected onto the box bounds. When |∇f|∞ is several times larger
/// than the bound width — which is what the AD/analytical FOCE gradient added in
/// PR #48 looks like on standard PK models (≈ 10²–10³ in scaled log/Cholesky
/// space) — that first step pins every component to a corner of the box and the
/// OFV explodes. The two algorithms then dead-end differently but with the same
/// outcome: SLSQP's QP stays stuck at that projected corner, while L-BFGS's line
/// search cannot find a decrease along the overshot direction and fails on eval
/// 1. Either way theta stays byte-identical to init for the rest of the budget.
/// See issue #55 (SLSQP) and #960 (the same first-step overshoot on the
/// analytic-gradient NLopt L-BFGS `Auto` default, which left warfarin FOCEI stuck
/// at its initial estimates).
///
/// This helper rescales `g` in place by a single scalar so that no component
/// of the identity-Hessian Newton step exceeds its per-dimension step budget,
/// where the budget is `clamp(half_width, 0.1, 1.0)` in scaled space. The
/// [0.1, 1.0] clamp keeps the cap effective on very narrow bounds (where
/// half-width alone would paralyse it — notably fixed parameters with
/// half-width 0) and on very wide log/Cholesky bounds (40+ units on some
/// omega/sigma dims, where an uncapped budget would let the gradient
/// magnitude through unchanged). For non-fixed parameters with `half_width <
/// 0.1` the post-cap step can exceed half-width by a small constant, which
/// is benign because the dimension itself is narrow.
///
/// The rescale is uniform across components, so the descent direction is
/// unchanged.
///
/// Returns true if the cap fired (gradient was rescaled), false otherwise.
/// Applied on **every** SLSQP gradient eval, but on L-BFGS **only the first**
/// (see [`should_cap_gradient`]): SLSQP re-solves its QP from the current
/// Hessian each step so a uniform cap is harmless, whereas L-BFGS builds its
/// Hessian from successive `(s, y)` gradient-difference pairs — capping past the
/// first eval would corrupt that curvature (the regression noted in #960). Only
/// the opening `H₀ = I` step needs taming; once real curvature accumulates the
/// L-BFGS line search safeguards itself. MMA has its own trust-region-style
/// safeguards and BOBYQA is derivative-free, so neither is capped.
pub(crate) fn cap_scaled_gradient(g: &mut [f64], lower_s: &[f64], upper_s: &[f64]) -> bool {
    debug_assert_eq!(g.len(), lower_s.len());
    debug_assert_eq!(g.len(), upper_s.len());
    let mut worst_ratio = 0.0_f64;
    for i in 0..g.len() {
        let budget = ((upper_s[i] - lower_s[i]).abs() * 0.5).clamp(0.1, 1.0);
        let ratio = g[i].abs() / budget;
        if ratio > worst_ratio {
            worst_ratio = ratio;
        }
    }
    if worst_ratio > 1.0 {
        for gi in g.iter_mut() {
            *gi /= worst_ratio;
        }
        true
    } else {
        false
    }
}

/// Whether the identity-Hessian overshoot cap ([`cap_scaled_gradient`]) should
/// fire on this gradient eval.
///
/// `n_grad_evals` is the running count of gradient evaluations *including this
/// one* (`population_gradient` increments it before returning), so the first
/// gradient eval is `n_grad_evals == 1`.
///
/// - **SLSQP** — cap every eval. Its QP re-solves from the current quasi-Newton
///   Hessian each step, so rescaling the gradient never corrupts stored
///   curvature (issue #55).
/// - **L-BFGS** — cap only the first eval. Its Hessian is reconstructed from the
///   `(s, y)` pairs formed by successive gradient differences; a blanket cap
///   would perturb `y` on every step and corrupt that curvature (which is why a
///   uniform cap regressed well-behaved L-BFGS fits — #960). Only the opening
///   `H₀ = I` step overshoots, so taming eval 1 alone lets the fit leave init
///   while leaving every later `(s, y)` pair intact.
/// - **MMA / BOBYQA** and everything else — never cap here (MMA has its own
///   safeguards; BOBYQA is derivative-free).
pub(crate) fn should_cap_gradient(algo: nlopt::Algorithm, n_grad_evals: usize) -> bool {
    match algo {
        nlopt::Algorithm::Slsqp => true,
        nlopt::Algorithm::Lbfgs => n_grad_evals == 1,
        _ => false,
    }
}

/// The population objective the outer loop actually minimises: [`pop_nll`] (FOCE/FOCEI),
/// or the AGQ marginal when the stage's method is `agq`.
///
/// **Every** production site that needs "the objective for *this* fit" must call this, not
/// `pop_nll` — the objective closures, the reconverged-FD gradient, and the covariance
/// stencil alike. An AGQ fit whose covariance step differenced the *FOCE* objective would
/// report standard errors for a likelihood it never optimised.
///
/// Non-AGQ fits forward to `pop_nll` with identical arguments, so their OFV is unchanged
/// bit for bit.
pub(crate) fn pop_nll_opts(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> f64 {
    if let Some(n_nodes) = options.agq_nodes() {
        // AGQ integrates over whatever random effects the subject has: η alone, or the
        // stacked (η, κ₁..κ_K) under IOV — the joint marginal, not the η-only one. The modes
        // are the ones the shared inner loop already converged (`find_ebe_iov` returns the
        // joint mode); AGQ does not re-optimise them, it lays its grid around them.
        // `h_matrices` (the ∂f/∂η Jacobian) is a FOCE artefact AGQ has no use for — it
        // finite-differences the true posterior Hessian instead. See `crate::estimation::agq`.
        return crate::estimation::agq::agq_population_nll(
            model,
            population,
            params,
            eta_hats,
            kappas,
            n_nodes,
            options.hessian_anchor(),
        );
    }
    pop_nll(
        model,
        population,
        params,
        eta_hats,
        h_matrices,
        kappas,
        options.interaction,
    )
}

/// Dispatch to the IOV-aware or standard population NLL based on model.n_kappa.
/// `kappas` is ignored (may be empty) when `model.n_kappa == 0`.
pub(crate) fn pop_nll(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> f64 {
    if model.n_kappa > 0 {
        if let Some(ref iov) = params.omega_iov {
            return foce_population_nll_iov(
                model,
                population,
                &params.theta,
                eta_hats,
                h_matrices,
                kappas,
                &params.omega,
                iov,
                &params.sigma.values,
                interaction,
            );
        }
    }
    foce_population_nll(
        model,
        population,
        &params.theta,
        eta_hats,
        h_matrices,
        &params.omega,
        &params.sigma.values,
        interaction,
    )
}

/// State passed through NLopt's user-data mechanism
struct NloptState {
    cached_etas: Vec<DVector<f64>>,
    cached_h_mats: Vec<DMatrix<f64>>,
    /// Mixture per-class EBE warm-start cache `[class][subject]` (#977 Phase 3).
    /// Empty for non-mixture models.
    cached_etas_by_class: Vec<Vec<DVector<f64>>>,
    best_ofv: f64,
    n_evals: usize,
    /// Count of gradient evaluations so far. Distinct from `n_evals` (which
    /// also counts objective-only line-search probes); drives the
    /// `reconverge_gradient_interval` schedule.
    n_grad_evals: usize,
    /// Previous parameter vector — used to compute step_norm for the trace.
    prev_x: Vec<f64>,
    last_improvement_eval: usize,
    best_at_last_improvement: f64,
    /// Sticky once latched — subsequent evals return `best_ofv` with zero
    /// gradient so SLSQP/L-BFGS xtol/ftol fires in microseconds instead
    /// of grinding through `maxeval` at full inner-loop cost.
    stagnation_stopped: bool,
}

/// Latches `stagnation_stopped` once recent evals show no OFV progress.
///
/// Without this, SLSQP on poorly-identified (e.g. γ-bearing) FOCEI
/// problems can spend 30+ min at a numerically-flat OFV before its
/// xtol/ftol criteria fire.
///
/// `enabled = false` disables the guard entirely: never latches and never
/// reports stagnation, so the optimizer runs to its own termination
/// criterion (or to `outer_maxiter`).
/// Whether the EBE convergence guard rejects this outer trial. Two independent triggers:
///
/// 1. **Hard reject** — any subject was rejected at its inner start (a pathological ODE+IOV
///    warm-start NLL). Its returned `(η, H)` is a degenerate placeholder, so the trial must
///    be rejected regardless of `max_unconverged_frac` or the `min_obs` filter, otherwise a
///    zero H-matrix would corrupt an *accepted* OFV (#603 review #1/#2).
/// 2. **Too many unconverged** — the fraction of (long-enough-record) subjects whose inner
///    optimizer failed exceeds `max_unconverged_frac` and the OFV is finite.
///
/// A negative `max_unconverged_frac` disables the fraction trigger (but never the hard
/// reject). Centralising the predicate keeps the five evaluation sites in this module from
/// drifting (#603 review #8).
fn ebe_guard_rejects(
    stats: &InnerLoopStats,
    n_subj: usize,
    raw_ofv: f64,
    max_unconverged_frac: f64,
) -> bool {
    if stats.n_start_rejected > 0 {
        return true;
    }
    let frac = stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
    raw_ofv.is_finite() && frac > max_unconverged_frac && max_unconverged_frac >= 0.0
}

/// Objective value for a guard-rejected outer step, **consistent** with the center-push
/// gradient `g[i] = 100·(xs[i] − c[i])` (`c` = scaled bound midpoint) returned alongside it.
///
/// NLopt's gradient line search (More-Thuente) reconciles `f` and `∇f`: it interpolates on
/// both, so the returned objective must integrate the returned gradient. The historical
/// pairing — flat `f = 1e20` with a non-zero center-push gradient — violates that
/// (`∇(const) = 0 ≠ center-push`), and on a stiff objective whose **first** optimizer step
/// overshoots straight into the EBE guard (ODE + `iiv_on_ruv`: a large step diverges the
/// inner EBEs and overflows the `exp(2·η_ruv)` marginal) the line search fails on iteration
/// one, before any curvature is built. Returning the quadratic bowl `BASE + 50·Σ(xs − c)²`
/// — whose gradient is exactly the `100·(xs − c)` center-push — lets the line search
/// backtrack to a feasible step. `BASE` is a wall far above any feasible OFV yet low enough
/// that the quadratic term stays f64-resolvable (the old `1e20` swamped it). Only the
/// gradient optimizers need this; derivative-free BOBYQA keeps the flat `1e20` wall.
fn guard_penalty_value(xs: &[f64], lower_s: &[f64], upper_s: &[f64]) -> f64 {
    const BASE: f64 = 1e12;
    let pen: f64 = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let c = (lower_s[i] + upper_s[i]) / 2.0;
            let d = x - c;
            d * d
        })
        .sum();
    BASE + 50.0 * pen
}

fn detect_stagnation(state: &mut NloptState, n: usize, enabled: bool) -> bool {
    if !enabled {
        return false;
    }
    if state.stagnation_stopped {
        return true;
    }
    // Tied to the FD-gradient cost: 3*(n+1) evals = 3 attempted descent
    // steps with their gradient probes. Minimum of 50 evals so very-small
    // problems still get a real chance before we declare stagnation.
    let stagnation_window: usize = (3 * (n + 1)).max(50);
    // Absolute OFV improvement below this is treated as noise. Matches
    // typical FOCE EBE-loop precision (~1e-3 OFV units) — see
    // `inner_tol` default and Sheiner–Beal linearisation comment in
    // [types.rs:959].
    const STAGNATION_THRESHOLD: f64 = 1e-3;

    let improved = (state.best_at_last_improvement - state.best_ofv) > STAGNATION_THRESHOLD;
    if improved {
        state.last_improvement_eval = state.n_evals;
        state.best_at_last_improvement = state.best_ofv;
        false
    } else if state.n_evals.saturating_sub(state.last_improvement_eval) >= stagnation_window {
        state.stagnation_stopped = true;
        true
    } else {
        false
    }
}

fn new_nlopt_state(n_subj: usize, n_eta: usize, x0: &[f64]) -> NloptState {
    NloptState {
        cached_etas: vec![DVector::zeros(n_eta); n_subj],
        cached_h_mats: Vec::new(),
        cached_etas_by_class: Vec::new(),
        best_ofv: f64::INFINITY,
        n_evals: 0,
        n_grad_evals: 0,
        prev_x: x0.to_vec(),
        last_improvement_eval: 0,
        best_at_last_improvement: f64::INFINITY,
        stagnation_stopped: false,
    }
}

/// Run NLopt CRS2-LM (Controlled Random Search with Local Mutation) as a
/// gradient-free global pre-search before the local optimizer. Returns
/// the best point found in the same scaled coordinate system as the
/// caller's `x0` / `lower_s` / `upper_s`. Falls back with `Err(...)`
/// when the NLopt build doesn't ship CRS2-LM (a clear-message failure
/// is more useful than the local optimizer silently using the original
/// `x0`).
///
/// CRS2-LM is a population-based algorithm: it maintains a pool of
/// `population_size` candidate points (NLopt's default is `10*(n+1)`),
/// repeatedly drawing new candidates inside the simplex of the best-so-far
/// points and mutating one at a time. It needs explicit bounds (which
/// the FOCE outer-loop space provides) and is generally insensitive to
/// the initial point — useful precisely when our initial point lies in
/// a bad basin.
fn run_global_presearch(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
    scale: &[f64],
    lower_s: &[f64],
    upper_s: &[f64],
    x0: &[f64],
) -> Result<Vec<f64>, String> {
    let n = x0.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    // Probe CRS2-LM availability — some NLopt builds (notably the
    // minimal one in the homebrew nlopt-rs crate) ship without it.
    // Catch the panic so we surface a useful warning instead of
    // crashing the fit.
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fn dummy(_x: &[f64], _g: Option<&mut [f64]>, _d: &mut ()) -> f64 {
            0.0
        }
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            n,
            dummy,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if probe.is_err() {
        return Err(
            "NLopt CRS2-LM not available in this build — install a full \
             NLopt (brew install nlopt / apt install libnlopt-dev) and rebuild"
                .into(),
        );
    }

    let n_evals = Arc::new(AtomicUsize::new(0));
    let n_evals_cl = Arc::clone(&n_evals);
    let verbose = options.verbose;

    // Helper: evaluate the FOCE OFV at a single point in scaled space,
    // independent of any NLopt state. Used to compute the user's initial
    // OFV up-front (for the keep-best-of-(user, CRS2-LM) compare below).
    let eval_at_scaled = |xs: &[f64]| -> f64 {
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let cached_zero = vec![DVector::zeros(n_eta); n_subj];
        let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&cached_zero),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        let nll = pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
        let raw = 2.0 * nll;
        let guarded = ebe_guard_rejects(&ebe_stats, n_subj, raw, options.max_unconverged_frac);
        if !raw.is_finite() || guarded {
            1e20
        } else {
            raw
        }
    };

    let initial_ofv = eval_at_scaled(x0);
    if options.verbose {
        eprintln!(
            "Initial OFV at user-supplied parameters: {:.6} (used as fallback if global \
             pre-search doesn't beat it)",
            initial_ofv,
        );
    }

    let pre_state = new_nlopt_state(n_subj, n_eta, x0);

    let pre_objective = |xs: &[f64], _grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        if crate::cancel::is_cancelled(&options.cancel) {
            return 1e20;
        }
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);

        let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&state.cached_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );

        let nll = pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
        let raw_ofv = 2.0 * nll;

        let ebe_guard =
            ebe_guard_rejects(&ebe_stats, n_subj, raw_ofv, options.max_unconverged_frac);
        let ofv = if ebe_guard {
            1e20
        } else if raw_ofv.is_finite() {
            raw_ofv
        } else {
            1e20
        };

        // CRS2-LM samples globally, so warm-starting EBEs with the
        // best-so-far cached etas is mostly noise-amplifying — keep
        // them at zeros for the next eval. The local optimizer that
        // follows starts from a sensible point and warm-starts cleanly.
        state.cached_etas = vec![DVector::zeros(n_eta); n_subj];
        state.cached_h_mats = hms;
        state.n_evals += 1;
        n_evals_cl.fetch_add(1, Ordering::Relaxed);

        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                eprintln!(
                    "Global pre-search eval {:>4}: OFV = {:.6}",
                    state.n_evals, ofv
                );
            }
        }

        ofv
    };

    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Crs2Lm,
        n,
        pre_objective,
        nlopt::Target::Minimize,
        pre_state,
    );
    opt.set_lower_bounds(lower_s)
        .map_err(|e| format!("CRS2-LM lower bounds: {:?}", e))?;
    opt.set_upper_bounds(upper_s)
        .map_err(|e| format!("CRS2-LM upper bounds: {:?}", e))?;

    // Default budget: 30 * (n + 1) — modest budget that's enough to
    // probe a few candidate basins without dominating the wall time of
    // the subsequent local refine. Users with hard-to-find optima can
    // bump `global_maxeval` to e.g. 200*(n+1) for a thorough sweep.
    let max_eval = if options.global_maxeval > 0 {
        options.global_maxeval as u32
    } else {
        30 * (n as u32 + 1)
    };
    opt.set_maxeval(max_eval)
        .map_err(|e| format!("CRS2-LM maxeval: {:?}", e))?;

    if options.verbose {
        eprintln!(
            "Starting NLopt CRS2-LM global pre-search ({} parameters, max {} evals)...",
            n, max_eval
        );
    }

    let mut x_pre = x0.to_vec();
    let pre_ofv = match opt.optimize(&mut x_pre) {
        Ok((status, ofv)) => {
            if options.verbose {
                eprintln!(
                    "Global pre-search finished: {:?}, best OFV = {:.6} after {} evals",
                    status,
                    ofv,
                    n_evals.load(Ordering::Relaxed),
                );
            }
            ofv
        }
        Err((fail, ofv)) => {
            if options.verbose {
                eprintln!(
                    "Global pre-search stopped: {:?}, best OFV = {:.6} after {} evals",
                    fail,
                    ofv,
                    n_evals.load(Ordering::Relaxed),
                );
            }
            ofv
        }
    };

    // Keep whichever is better between the user-supplied initials and
    // CRS2-LM's best point. CRS2-LM ignores the starting point and
    // samples freely in [lower, upper], so for already-good inits its
    // best point is often *worse* than where we started — handing that
    // to the local optimizer would actively regress the fit. The
    // initial-OFV evaluation above is one extra inner-loop pass, cheap
    // insurance against that case.
    if initial_ofv.is_finite() && initial_ofv <= pre_ofv {
        if options.verbose {
            eprintln!(
                "Global pre-search did not beat user-supplied initials \
                 ({:.4} vs {:.4}); keeping user initials for local optimisation.",
                pre_ofv, initial_ofv,
            );
        }
        Ok(x0.to_vec())
    } else {
        Ok(x_pre)
    }
}

/// nlmixr2-style `rescale2` preconditioner: scale each packed param by its
/// bounds half-range `(hi−lo)/2`, so every coordinate spans ~2 units in scaled
/// space (the optimizer sees comparable per-parameter search ranges → similar
/// gradient/step magnitudes). This is value/bounds-based normalization (what
/// nlmixr2's `normType="rescale2"` does), not curvature-based — it worked where
/// the BHHH-diagonal preconditioner did not. Fixed params (lo==hi) and
/// degenerate ranges fall back to 1.0. Selected by
/// `parameter_scaling = rescale2` (see [`ParameterScaling::Rescale2`]).
fn compute_rescale2_scale(bounds: &PackedBounds) -> Vec<f64> {
    (0..bounds.lower.len())
        .map(|k| {
            let hw = (bounds.upper[k] - bounds.lower[k]).abs() * 0.5;
            if hw.is_finite() && hw > 1e-6 {
                hw
            } else {
                1.0
            }
        })
        .collect()
}

/// Resolve [`ParameterScaling::Auto`] to a concrete strategy. `Auto` applies
/// `Abs` (per-coordinate magnitude scaling, normalise by |packed value|) to the
/// gradient-based optimizers (`Bfgs`, `Lbfgs`, `NloptLbfgs`, `Slsqp`) and `None`
/// otherwise. `Abs` is the correct preconditioner for a quasi-Newton / SQP step:
/// it presents O(1) coordinates so the optimizer's first move is well-scaled.
///
/// This replaces the earlier `Rescale2` (bound-half-width) scaling, which is the
/// *wrong* preconditioner for gradient optimizers — bound width is unrelated to
/// curvature, so it drove L-BFGS into a parameter bound on ill-conditioned fits
/// (warfarin FOCEI −286→−243; tvcov to a +166 local min with TVV pinned at its
/// lower bound) and froze SLSQP's first step on two_cpt_oral_cov (−1026, no move
/// from init). `Abs` recovers the correct optimum in every one of those cases
/// (warfarin −286, tvcov −188.6 at truth, two_cpt_oral_cov −1165) and preserves
/// SLSQP's warfarin_iov cold-start win (OFV 307.8, the #335 case).
///
/// The derivative-free default `Bobyqa` is left unscaled — any per-coordinate
/// scaling distorts its trust-region quadratic model and regresses multi-cpt / PD
/// fits (e.g. emax_pkpd −36.8→−13.5, three_cpt_iv −730.6→−715.9). `Mma` /
/// `TrustRegion` are left to the unscaled (legacy `scale_params` / IOV-auto)
/// branch. Non-`Auto` values pass through unchanged.
fn resolve_scaling(ps: ParameterScaling, opt: Optimizer) -> ParameterScaling {
    match ps {
        ParameterScaling::Auto => match opt {
            // Gradient-based optimizers condition best with magnitude scaling
            // (`Abs` = normalise by |packed value|). `Rescale2` (bound-half-width)
            // is the wrong preconditioner: it drives L-BFGS into a bound on
            // ill-conditioned fits (warfarin −286→−243, tvcov to a local min) and
            // freezes SLSQP's first step on two_cpt_oral_cov (−1026, no move).
            // `Abs` recovers the correct optimum for both (warfarin −286, tvcov
            // −188.6 at truth, two_cpt_oral_cov −1165) while preserving SLSQP's
            // warfarin_iov cold-start win (OFV 307.8, the #335 case).
            Optimizer::Bfgs | Optimizer::Lbfgs | Optimizer::NloptLbfgs | Optimizer::Slsqp => {
                ParameterScaling::Abs
            }
            _ => ParameterScaling::None,
        },
        other => other,
    }
}

/// Resolve the derivative-free `bobyqa` outer optimizer's `ftol_rel` stop tolerance.
///
/// `override_ftol` (`[fit_options] outer_ftol`) wins when set. Otherwise auto-select:
/// `1e-8` for a **pure non-Gaussian** model (TTE or the Phase-4 categorical family,
/// #760) — its data objective is evaluated *exactly*, so the looser historical `1e-6`
/// stopped BOBYQA short of the optimum on the near-flat frailty/random-effect-ω² ridge
/// (#469: a Weibull shape-frailty read 0.204 vs the NONMEM/nlmixr2 0.175 consensus;
/// `1e-8` lands 0.176) — and `1e-6` for everything else. The floor is deliberate: on a
/// **noisy** objective (ODE solver error, or an FD-inner FOCE model such as LTBS) `1e-8`
/// is unreachable, so BOBYQA would grind toward its maxeval budget instead of converging
/// (≈3× the evaluations on an ODE fit). A non-Gaussian endpoint carried on an ODE
/// disposition (`is_ode` true) therefore keeps `1e-6`.
fn resolve_outer_ftol(is_non_gaussian: bool, is_ode: bool, override_ftol: Option<f64>) -> f64 {
    override_ftol.unwrap_or(if is_non_gaussian && !is_ode {
        1e-8
    } else {
        1e-6
    })
}

/// Absolute OFV improvement below which a step counts as "no significant
/// progress" for the plateau tracker. Matches the stagnation guard's
/// `STAGNATION_THRESHOLD` (both key off the ~1e-3 FOCE EBE-loop precision).
const PLATEAU_OFV_THRESHOLD: f64 = 1e-3;

/// Minimum number of consecutive flat tail evals (no improvement above
/// `PLATEAU_OFV_THRESHOLD`) for a bare NLopt `Failure`/`ForcedStop` to be
/// reclassified as convergence-at-a-plateau (issue #751). A genuine early stall
/// (e.g. the SS-oral fit quits after ~5 evals still plunging) never accumulates
/// a flat tail and stays `converged=false`. Chosen below the shortest observed
/// good-fit tail (npde ≈ 8, schnider ≈ 19) yet well above the zero-length tail
/// of a real stall.
const PLATEAU_MIN_FLAT_EVALS: usize = 5;

/// Relative tolerance for the best-seen ↔ final-inner-loop OFV self-consistency
/// guard. A converged fit's EBE fixpoint is reproducible: re-running the inner
/// loop cold at the restored best point returns the same OFV the optimizer saw
/// warm-started. A large positive gap (the SS-oral fit: best-seen 83.3 vs cold
/// 121.4) means the "optimum" was a warm-start artifact — not converged — so it
/// is rejected even if the OFV trace looked flat.
const PLATEAU_CONSISTENCY_REL_TOL: f64 = 1e-3;

/// Classify a bare NLopt `Failure`/`ForcedStop` as convergence-at-a-plateau
/// (issue #751). Every eval index and count here is measured over *feasible*
/// (unguarded) evals only — guarded/penalty evals are excluded entirely, so
/// neither the progress test nor the flat-tail length can be padded by boundary
/// thrashing. Returns `true` only when all three hold:
///   - **progress**: the last significant OFV improvement landed on a feasible
///     eval *after* the first (`last_sig_feasible_eval >= 2`; the count is
///     1-based over feasible evals, so `0` means no feasible eval was ever seen).
///     Feasible eval 1 merely establishes the baseline objective (INF → OFV₀); a
///     fit whose last significant improvement is that same first feasible eval
///     never descended at all — it stalled at the start (NLopt's L-BFGS first
///     step overshoots and its line search fails on e.g. warfarin FOCEI, leaving
///     the fit pinned at the initial estimates). Counting over *feasible* evals
///     is also what stops a guard-rejected eval 1 from faking progress: the first
///     feasible point is feasible-eval 1 (the baseline) whether or not earlier
///     evals were guard-penalised, so a "significant improvement" there is not
///     descent. That is a failed start, not a converged plateau, even though the
///     objective is then "flat" for the remaining probes;
///   - **plateau**: the flat tail (feasible evals since the last improvement
///     above `PLATEAU_OFV_THRESHOLD`, = `feasible_evals − last_sig_feasible_eval`)
///     is at least `PLATEAU_MIN_FLAT_EVALS` — a genuine mid-descent stall has
///     none; and
///   - **consistency**: the cold-restart `final_ofv` is not materially *worse*
///     than `best_seen_ofv` (a large positive gap exposes a warm-start-only
///     "optimum"). A cold restart that ties or improves is fine.
/// Pulled out as a pure fn so the decision is unit-testable without driving a
/// full NLopt fit.
fn failure_is_converged_plateau(
    feasible_evals: usize,
    last_sig_feasible_eval: usize,
    best_seen_ofv: Option<f64>,
    final_ofv: f64,
) -> bool {
    let made_progress = last_sig_feasible_eval >= 2;
    let flat_tail = feasible_evals.saturating_sub(last_sig_feasible_eval);
    let plateaued = flat_tail >= PLATEAU_MIN_FLAT_EVALS;
    let consistent = best_seen_ofv
        .is_none_or(|best| final_ofv <= best + PLATEAU_CONSISTENCY_REL_TOL * (1.0 + best.abs()));
    made_progress && plateaued && consistent
}

fn optimize_nlopt(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x0 = pack_params(init_params);
    clamp_to_bounds(&mut x0, &bounds);
    let n = x0.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();

    // Per-element scale factors: present O(1) coordinates to NLopt.
    //
    // `compute_scale` normalises by |packed value|, which gives O(1)
    // scaled coords for log-packed thetas (CL, V, KA — log-magnitude
    // is typically > 0.1) and a 1.0 fallback for everything near zero.
    // For identity-packed thetas (those with `theta_lower < 0`,
    // typically small covariate effects like THETA_AGE_CL = -0.01)
    // this places the scaled value near zero, and SLSQP's BFGS-flavored
    // Hessian estimate handles wildly different scaled magnitudes
    // poorly — observed regression: SAD_SCEN1 FOCEI took 510+ evals
    // (40+ min) vs ~90 evals (~5 min) with scaling off. Auto-disable
    // scaling whenever any identity-packed theta is present, so the
    // optimizer runs in the natural (mixed) packed space where
    // BFGS's own scale-adaptation works correctly.
    let has_identity_theta = init_params.theta_lower.iter().any(|&lo| lo < 0.0);
    // IOV + SLSQP: auto-enable per-coordinate scaling (issue #101 rec #2). IOV
    // models pack disparate-magnitude parameters (block-diagonal omega plus the
    // kappa block), and SLSQP's uniform gradient cap (`cap_scaled_gradient`,
    // applied on every SLSQP eval — L-BFGS is capped only on its first) otherwise
    // rescales the whole gradient by
    // the worst (theta) component, starving the omega/omega_iov step so the
    // variance components stay pinned at their initial values. Scaling presents
    // O(1) coordinates so the cap no longer starves them. The #99 regression
    // that made scaling default-off was on non-IOV models and other algorithms
    // (notably MMA, which scaling hurts here), so scope the auto-enable to the
    // IOV + SLSQP combination that actually needs it.
    //
    // Scope note: as of #155 the default outer optimizer is `Bobyqa`, not
    // `Slsqp` — so default-IOV fits no longer hit this branch. BOBYQA is
    // gradient-free and doesn't suffer the `cap_scaled_gradient` starvation that
    // motivates the scaling here, so leaving it disabled on the default path is
    // intentional. This auto-enable now only fires for an explicit
    // `optimizer = slsqp` on IOV models (the path it was originally written for).
    let auto_scale_iov = model.n_kappa > 0 && matches!(options.optimizer, Optimizer::Slsqp);
    let scale: Vec<f64> = match resolve_scaling(options.parameter_scaling, options.optimizer) {
        ParameterScaling::Rescale2 => compute_rescale2_scale(&bounds),
        // Magnitude scaling, but disabled when an identity-packed theta is present
        // (covariate effects with `theta_lower < 0`): `compute_scale` leaves those
        // small coordinates near their raw value while log-packed θ become O(1), and
        // the resulting wildly-mixed scaled magnitudes hurt the quasi-Newton/SQP
        // Hessian estimate (observed: SAD_SCEN1 FOCEI 510+ evals vs ~90 unscaled).
        // Falling back to the natural mixed space keeps that protection.
        ParameterScaling::Abs => {
            if has_identity_theta {
                vec![1.0; n]
            } else {
                compute_scale(&x0)
            }
        }
        // `Auto` is resolved away by `resolve_scaling`; group with `None`.
        ParameterScaling::None | ParameterScaling::Auto => {
            if (options.scale_params || auto_scale_iov) && !has_identity_theta {
                compute_scale(&x0)
            } else {
                vec![1.0; n]
            }
        }
    };
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    // Scale x0 into optimizer space: xs[i] = x[i] / scale[i].
    for i in 0..n {
        x0[i] /= scale[i];
    }

    // Optional gradient-free global pre-search (NLopt CRS2-LM). Samples
    // within the parameter bounds and lets the local optimizer pick up
    // from the best point found — useful for poorly-identified models
    // where the local optimizer can land in a degenerate basin from a
    // far-from-truth start. The pre-search runs the same FOCE objective
    // as the main optimizer (no shortcuts), so each global eval is a
    // full inner-loop pass; budget is `global_maxeval` (default
    // `200 * (n_params + 1)` when 0).
    if options.global_search {
        let pre_x = run_global_presearch(
            model,
            population,
            init_params,
            options,
            &scale,
            &lower_s,
            &upper_s,
            &x0,
        );
        match pre_x {
            Ok(best_x) => x0 = best_x,
            Err(e) => warnings.push(format!("global_search disabled: {}", e)),
        }
    }

    let state = new_nlopt_state(n_subj, n_eta, &x0);

    // External counter mirrors state.n_evals — nlopt doesn't hand `state`
    // back after `opt.optimize()`, so we need an Arc to read the final
    // count for reporting. Keep both in sync inside the objective closure.
    let n_evals_outer = Arc::new(AtomicUsize::new(0));
    let n_evals_cl = Arc::clone(&n_evals_outer);

    // Best-seen accumulator (issue #59). NLopt returns the last evaluated
    // point, not the best one — when the stagnation guard short-circuits
    // by returning `best_ofv` with zero gradient, the optimizer can drift
    // a step or two off the true minimum before its xtol/ftol fires. We
    // track the best (xs, ofv) externally and restore x0 to it after
    // optimize() returns, before the final inner loop and covariance step.
    let best_seen: Arc<Mutex<Option<(Vec<f64>, f64)>>> = Arc::new(Mutex::new(None));
    let best_seen_cl = Arc::clone(&best_seen);

    let last_gradient: Arc<Mutex<Option<Vec<f64>>>> = Arc::new(Mutex::new(None));
    let last_gradient_cl = Arc::clone(&last_gradient);

    // Externalised OFV-plateau tracker counting *feasible* (unguarded) evals
    // only: `(baseline_ofv, last_sig_feasible_eval, feasible_evals)`.
    // `feasible_evals` is a 1-based running count of unguarded evals;
    // `last_sig_feasible_eval` is the feasible-eval index at which the feasible
    // best OFV last improved by more than `PLATEAU_OFV_THRESHOLD` over
    // `baseline_ofv`. Guarded evals are excluded entirely so (a) a guard-penalty
    // value (which pollutes `state.best_ofv`) can never seed a fake "improvement"
    // — feasible eval 1 only establishes the baseline, real progress must land on
    // a later feasible eval — and (b) a tail of guard-rejected boundary probes
    // cannot pad the plateau length (both the index and the count are feasible-
    // only, so `flat_tail = feasible_evals − last_sig_feasible_eval` measures flat
    // *feasible* evals). Distinct from (and independent of) the stagnation guard's
    // own bookkeeping so it works even when that guard is disabled. Read after
    // `optimize()` to tell a plateaued optimum from a genuine early stall (#751).
    let plateau_tracker: Arc<Mutex<(f64, usize, usize)>> =
        Arc::new(Mutex::new((f64::INFINITY, 0, 0)));
    let plateau_tracker_cl = Arc::clone(&plateau_tracker);

    // EBE stats accumulator: tracks worst unconverged count and total fallbacks.
    #[derive(Default)]
    struct EbeAccum {
        max_unconverged: usize,
        total_fallback: usize,
        n_convergence_warnings: usize,
    }
    let ebe_accum: Arc<Mutex<EbeAccum>> = Arc::new(Mutex::new(EbeAccum::default()));
    let ebe_accum_cl = Arc::clone(&ebe_accum);

    // Select NLopt algorithm. `optimize_population` resolves `Auto` to a concrete
    // optimizer before dispatching here, so `Auto` should never reach this match;
    // map it to BOBYQA (auto's FD fallback) rather than the catch-all SLSQP so a
    // future bypass degrades to the safe derivative-free path, not a silent SLSQP.
    let algo = match options.optimizer {
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::NloptLbfgs => nlopt::Algorithm::Lbfgs,
        Optimizer::Mma => nlopt::Algorithm::Mma,
        Optimizer::Bobyqa | Optimizer::Auto => nlopt::Algorithm::Bobyqa,
        _ => nlopt::Algorithm::Slsqp,
    };

    let verbose = options.verbose;

    // NLopt objective: receives xs (scaled), unscales before running inner loop.
    // Gradient: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i] (chain rule).
    let objective = |xs: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        // Cooperative cancellation: short-circuit cheaply so NLopt burns through
        // its remaining iteration budget in microseconds instead of minutes.
        if crate::cancel::is_cancelled(&options.cancel) {
            if let Some(g) = grad {
                for gi in g.iter_mut() {
                    *gi = 0.0;
                }
            }
            return 1e20;
        }
        // Stagnation guard: once latched, every subsequent eval returns
        // `best_ofv` with zero gradient. SLSQP / L-BFGS see a stationary
        // point and terminate via xtol_rel within a couple of evals,
        // instead of grinding through the remaining maxeval budget at
        // full inner-loop cost. See `detect_stagnation` doc comment for
        // the trigger criterion.
        if state.stagnation_stopped {
            if let Some(g) = grad {
                for gi in g.iter_mut() {
                    *gi = 0.0;
                }
            }
            state.n_evals += 1;
            n_evals_cl.fetch_add(1, Ordering::Relaxed);
            return state.best_ofv;
        }
        // Unscale from optimizer space to real (log/Cholesky) space.
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);

        // Mixture models (#977): K-fold log-sum-exp objective with a per-class
        // serial inner solve. The per-eval `kappas` slot stays empty: on the mixture
        // path the gradient reads `mixeval` (not these kappas), so the MIXEST-class κ
        // are only needed at the final inner loop below, where they *are* carried. The
        // MIXEST-class EBEs stand in for the warm-start / trace. The full `MixtureEval`
        // is kept in `mixeval` for the analytic gradient below.
        let mut mixeval: Option<crate::estimation::mixture::MixtureEval> = None;
        let (ehs, hms, ebe_stats, kappas, raw_ofv) = if params.mixture.is_some() {
            let warm = (!state.cached_etas_by_class.is_empty())
                .then_some(state.cached_etas_by_class.as_slice());
            let mut m =
                crate::estimation::mixture::mixture_ofv(model, population, &params, options, warm);
            let stats = InnerLoopStats {
                n_unconverged: m.ebe_stats.n_unconverged,
                n_fallback: m.ebe_stats.n_fallback,
                n_start_rejected: m.ebe_stats.n_start_rejected,
            };
            let ofv = m.ofv;
            // A derivative-free eval (`grad` is `None` — e.g. BOBYQA, the default)
            // never touches `mixeval` or the analytic gradient, so avoid the full
            // per-class EBE cache clone: move `etas_by_class` straight into the
            // warm-start cache and the MIXEST EBEs into the result. When a gradient
            // *is* requested the analytic path reads `m.etas_by_class`, so it must
            // stay intact and the cache takes a clone.
            if grad.is_some() {
                state.cached_etas_by_class = m.etas_by_class.clone();
                let out = (
                    m.mixest_etas.clone(),
                    m.mixest_h_mats.clone(),
                    stats,
                    Vec::new(),
                    ofv,
                );
                mixeval = Some(m);
                out
            } else {
                state.cached_etas_by_class = std::mem::take(&mut m.etas_by_class);
                (
                    std::mem::take(&mut m.mixest_etas),
                    std::mem::take(&mut m.mixest_h_mats),
                    stats,
                    Vec::new(),
                    ofv,
                )
            }
        } else {
            let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
            let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
                model,
                population,
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&state.cached_etas),
                Some(&mu_k),
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
            );
            let nll = pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
            (ehs, hms, ebe_stats, kappas, 2.0 * nll)
        };

        // EBE convergence guard: reject step when too many subjects unconverged or any
        // subject was hard-rejected at its inner start.
        let ebe_guard_triggered =
            ebe_guard_rejects(&ebe_stats, n_subj, raw_ofv, options.max_unconverged_frac);
        {
            let mut acc = ebe_accum_cl.lock().unwrap();
            if acc.max_unconverged < ebe_stats.n_unconverged {
                acc.max_unconverged = ebe_stats.n_unconverged;
            }
            acc.total_fallback += ebe_stats.n_fallback;
            if ebe_guard_triggered {
                acc.n_convergence_warnings += 1;
            }
        }

        // Guard-rejected (EBE guard or non-finite OFV). For the gradient optimizers return
        // a quadratic penalty consistent with the center-push gradient below, so NLopt's
        // line search can backtrack instead of failing on a first-step overshoot (#486; see
        // `guard_penalty_value`). Derivative-free BOBYQA (`grad` is `None`) keeps the wall.
        let guarded = ebe_guard_triggered || !raw_ofv.is_finite();
        let ofv = if guarded {
            if grad.is_some() {
                guard_penalty_value(xs, &lower_s, &upper_s)
            } else {
                1e20
            }
        } else {
            raw_ofv
        };

        // Compute gradient if requested (central FD with fixed EBEs)
        let mut grad_norm_for_trace: Option<f64> = None;
        // Per-coordinate scaled gradient for the trace (#640); only populated
        // for a genuine OFV gradient (not the guard-penalty push), so it is
        // present exactly when `grad_norm_for_trace` is.
        let mut grad_vec_for_trace: Option<Vec<f64>> = None;
        if let Some(g) = grad {
            // A rejected or non-finite point has no useful population gradient: steepest
            // ascent toward bounds center nudges the optimizer back. Fall through (no early
            // return) so the eval is still traced and prev_x / stagnation stay correct,
            // just skipping the expensive population gradient (#603 review #5).
            if guarded {
                for i in 0..g.len() {
                    let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                    g[i] = 100.0 * (xs[i] - center_s);
                }
            } else {
                // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x); then scale for optimizer space.
                // Mixture (#977 Phase 4): analytic posterior-weighted gradient,
                // FD fallback when out of analytic scope.
                let grad_raw = if let Some(mev) = &mixeval {
                    crate::estimation::mixture::mixture_gradient(
                        model, population, &params, options, mev,
                    )
                    .unwrap_or_else(|| {
                        crate::estimation::mixture::mixture_gradient_fd(
                            model,
                            population,
                            &x,
                            init_params,
                            options,
                        )
                    })
                } else {
                    population_gradient(
                        &x,
                        n_subj,
                        init_params,
                        model,
                        population,
                        &ehs,
                        &hms,
                        &kappas,
                        &bounds,
                        options,
                        &mut state.n_grad_evals,
                    )
                };
                let mut sq = 0.0_f64;
                for k in 0..g.len() {
                    let gi = if grad_raw[k].is_finite() {
                        grad_raw[k] * scale[k]
                    } else {
                        0.0
                    };
                    g[k] = gi;
                    sq += gi * gi;
                }
                grad_norm_for_trace = Some(sq.sqrt());
                // Clone the scaled gradient for the trace only when a trace is
                // open — this runs on every gradient eval (the hot path), so an
                // unconditional `to_vec` would waste an allocation per eval on
                // the default trace-off path (#640 review). Snapshot before the
                // SLSQP cap below so it matches `grad_norm_for_trace`.
                if crate::estimation::trace::is_active() {
                    grad_vec_for_trace = Some(g.to_vec());
                }
                if should_cap_gradient(algo, state.n_grad_evals) {
                    cap_scaled_gradient(g, &lower_s, &upper_s);
                }
                // Gate on the global best (same tracker as the `best_seen` update
                // below) so `last_gradient` always reflects the best point seen.
                {
                    let global_best = best_seen_cl
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|(_, o)| *o)
                        .unwrap_or(f64::INFINITY);
                    if ofv < global_best {
                        *last_gradient_cl.lock().unwrap() = Some(grad_raw.clone());
                    }
                }
            }
        }

        // Update state
        state.cached_etas = ehs;
        state.cached_h_mats = hms;
        state.n_evals += 1;
        n_evals_cl.fetch_add(1, Ordering::Relaxed);
        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                eprintln!("Eval {:>4}: OFV = {:.6}", state.n_evals, ofv);
            }
        }
        // Record the *feasible*-eval index at which the feasible best OFV last
        // improved significantly (> `PLATEAU_OFV_THRESHOLD` below the last
        // recorded baseline). The gap between this and the feasible-eval count is
        // the flat-tail length — the plateau signal read after `optimize()`
        // (#751). Guarded evals are skipped entirely and do not advance the
        // feasible counter: their `guard_penalty_value` leaks into
        // `state.best_ofv`, so counting them would let a guard→feasible transition
        // masquerade as descent *and* let a tail of boundary probes pad the
        // plateau length. Feasible eval 1 sets the baseline; genuine progress must
        // land on a later feasible eval. Independent of the stagnation guard so it
        // is populated even when that guard is off. `ofv == raw_ofv` here.
        if !guarded {
            let mut pt = plateau_tracker_cl.lock().unwrap();
            pt.2 += 1; // one more feasible eval (1-based count / index)
            let feasible_idx = pt.2;
            if feasible_idx == 1 {
                // First feasible eval: establish the baseline objective. Not
                // "progress" — `last_sig_feasible_eval == 1` here.
                pt.0 = ofv;
                pt.1 = feasible_idx;
            } else if pt.0 - ofv > PLATEAU_OFV_THRESHOLD {
                pt.0 = ofv;
                pt.1 = feasible_idx;
            }
        }
        // `best_seen` tracks the global minimum across the whole run so the
        // final restore (issue #59) lands on the true best point, even when the
        // optimizer drifts away from it before terminating.
        {
            let mut bs = best_seen_cl.lock().unwrap();
            if bs.as_ref().is_none_or(|(_, prev)| ofv < *prev) {
                *bs = Some((xs.to_vec(), ofv));
            }
        }
        // After updating best_ofv, check whether we've stalled. If yes,
        // `stagnation_stopped` is latched and the early-return at the
        // top of the closure trips on the next eval.
        if detect_stagnation(state, n, options.stagnation_guard) && verbose {
            eprintln!(
                "Eval {:>4}: stopping early — OFV has converged (no improvement \
                 above 1e-3 in last window). This is normal convergence behaviour, \
                 not an error: further evaluations are unlikely to find a better \
                 solution.",
                state.n_evals,
            );
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm = {
                let sq: f64 = xs
                    .iter()
                    .zip(&state.prev_x)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let n = sq.sqrt();
                if n > 0.0 {
                    Some(n)
                } else {
                    None
                }
            };
            let method_str = match options.method {
                EstimationMethod::FoceI => "focei",
                _ => "foce",
            };
            let optimizer_str = match algo {
                nlopt::Algorithm::Bobyqa => "bobyqa",
                nlopt::Algorithm::Mma => "mma",
                nlopt::Algorithm::Lbfgs => "nlopt_lbfgs",
                _ => "slsqp",
            };
            let values = crate::estimation::parameterization::coordinate_values(&params);
            crate::estimation::trace::write_foce(
                state.n_evals,
                method_str,
                ofv,
                grad_norm_for_trace,
                step_norm,
                optimizer_str,
                Some(ebe_stats.n_unconverged),
                Some(ebe_stats.n_fallback),
                &values,
                grad_vec_for_trace.as_deref(),
            );
        }

        // Checkpoint (#755): pack the current estimates only when a write is due.
        // The objective runs once per eval, so gate on `is_due` to avoid a
        // per-eval allocation on the (default) no-checkpoint-due path.
        if crate::io::checkpoint::is_due() {
            let packed = crate::estimation::parameterization::pack_params(&params);
            crate::io::checkpoint::maybe_write(state.n_evals, ofv, &packed);
        }

        state.prev_x = xs.to_vec();

        ofv
    };

    // Create NLopt optimizer with state (operates in scaled xs space)
    let mut opt = nlopt::Nlopt::new(algo, n, objective, nlopt::Target::Minimize, state);
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    if matches!(algo, nlopt::Algorithm::Bobyqa) {
        // BOBYQA is derivative-free: each eval is one objective call, not
        // n+1 (gradient methods FD the gradient inside one outer iter).
        // Give it enough headroom to triangulate a quadratic in n-D and
        // still make real trust-region progress: 40 evals/param baseline
        // plus the outer_maxiter budget. The setup phase alone costs
        // 2n+1 evals before any movement.
        let bobyqa_maxeval =
            (options.outer_maxiter as u32).saturating_mul(n as u32 + 1) + 40 * (n as u32 + 1);
        opt.set_maxeval(bobyqa_maxeval).unwrap();
        // BOBYQA's xtol_rel controls rho_end / rho_start — i.e. how much
        // it must shrink the trust radius to declare success. 1e-12 is
        // unreachable in any realistic budget and forces MaxevalReached
        // at an arbitrary interim point; the default xtol 1e-4 in scaled
        // log-space is a ~0.01% move in the natural-scale parameter, which
        // is plenty tight for NLME work.
        opt.set_xtol_rel(options.outer_xtol).unwrap();
        // ftol_rel is the objective-change stop; see `resolve_outer_ftol` for the
        // `None` auto-selection (1e-8 pure non-Gaussian, 1e-6 otherwise) and #469 rationale.
        let ftol = resolve_outer_ftol(
            model.has_non_gaussian(),
            model.is_ode_based(),
            options.outer_ftol,
        );
        opt.set_ftol_rel(ftol).unwrap();
        // NLopt's default rhobeg is 25% of the bound-width — huge in our
        // log-space packing (theta bounds can span 40+ log units), so the
        // initial 2n+1 interpolation probes land in regions where the EBE
        // inner loop fails and the OFV gets clamped to 1e20, poisoning the
        // quadratic model. 0.5 in scaled space is a ~1.6× move on the
        // natural parameter scale — small enough to stay feasible at
        // start, large enough to see real OFV signal.
        let init_step: Vec<f64> = (0..n)
            .map(|i| {
                let half_width = (upper_s[i] - lower_s[i]).abs() * 0.5;
                0.5_f64.min(half_width.max(1e-6))
            })
            .collect();
        opt.set_initial_step(&init_step).unwrap();
    } else {
        opt.set_maxeval(options.outer_maxiter as u32 * (n as u32 + 1))
            .unwrap();
        if options.agq_nodes().is_some() {
            // AGQ's gradient is exact but **finite-difference-limited**: the grid-response
            // term and the posterior Hessian are both central differences, so the gradient
            // carries a noise floor (~1e-4 relative). The 1e-12 stops below are therefore
            // *unreachable* for it — and unreachable stops are not harmless. L-BFGS keeps
            // stepping until the true gradient drops under that floor, at which point the
            // search direction is noise, the line search cannot find a decrease, and NLopt
            // returns a bare `NLOPT_FAILURE`. The fit is fine (the engine restores the
            // best-seen point) but it is reported as *not converged*, which is a lie about a
            // result that has been flat to 8 significant figures for 15 evaluations.
            //
            // So stop AGQ where its objective actually settles — the same reachable
            // objective-change / step-size criteria BOBYQA gets — rather than chasing a
            // gradient norm the gradient cannot deliver. FOCE/FOCEI keep the 1e-12 stops:
            // their gradient is analytic to ~1e-11 and they *do* reach `XtolReached`.
            opt.set_xtol_rel(options.outer_xtol).unwrap();
            let ftol = resolve_outer_ftol(
                model.has_non_gaussian(),
                model.is_ode_based(),
                options.outer_ftol,
            );
            opt.set_ftol_rel(ftol).unwrap();
        } else {
            // FOCE objective is noisy from EBE re-estimation; let maxeval be the primary
            // stopping criterion and rely on the analytic gradient to drive |g| down.
            opt.set_xtol_rel(1e-12).unwrap();
            opt.set_ftol_rel(1e-12).unwrap();
        }
    }

    if options.verbose {
        eprintln!(
            "Starting NLopt {:?} optimization ({} parameters)...",
            algo, n
        );
    }

    // Run optimization
    let result = opt.optimize(&mut x0);

    // `max_eval_reached` distinguishes a spent evaluation budget from other
    // non-convergence: it gets its own warning ("increase maxiter") rather than
    // the generic "did not converge" message.
    let mut max_eval_reached = false;
    // A bare NLopt `Failure`/`ForcedStop` is ambiguous: it is returned both by a
    // genuine mid-descent stall *and* by the analytic-gradient L-BFGS default
    // (#639) settling onto a plateaued optimum whose ∇ has dropped below the
    // floor its line search can beat. We defer that verdict — see
    // `stationarity_check_pending` — and resolve it below on the OFV trace: the
    // feasible-eval plateau length plus a cold-restart self-consistency check at
    // the restored best point. (The analytic gradient norm is deliberately *not*
    // used — it still reads O(1) at these genuine optima; the resolution block
    // explains why.)
    let mut stationarity_check_pending = false;
    let mut converged = match &result {
        Ok((status, _)) => {
            if options.verbose {
                eprintln!("NLopt finished: {:?}", status);
            }
            max_eval_reached = matches!(status, nlopt::SuccessState::MaxEvalReached);
            matches!(
                status,
                nlopt::SuccessState::Success
                    | nlopt::SuccessState::FtolReached
                    | nlopt::SuccessState::XtolReached
                    | nlopt::SuccessState::StopValReached
            )
        }
        Err((fail, _)) => {
            if options.verbose {
                eprintln!("NLopt stopped: {:?}", fail);
            }
            match fail {
                nlopt::FailState::RoundoffLimited => true,
                nlopt::FailState::Failure | nlopt::FailState::ForcedStop => {
                    stationarity_check_pending = true;
                    false
                }
                _ => false,
            }
        }
    };

    drop(opt);

    // A spent evaluation budget gets a targeted "increase maxiter" warning; every
    // other non-convergence falls through to the generic "did not converge"
    // warning below. There is no automatic second optimization — a user who wants
    // SLSQP sets `optimizer = slsqp` as the primary (issue #657).
    if max_eval_reached {
        warnings.push(format!(
            "Outer optimization hit the evaluation budget (maxiter = {}) before \
             converging; increase maxiter for a tighter fit.",
            options.outer_maxiter,
        ));
        if options.verbose {
            eprintln!(
                "NLopt hit the evaluation budget (maxiter = {}) without converging — \
                 increase maxiter for a tighter fit.",
                options.outer_maxiter,
            );
        }
    }

    // Restore the best-seen point (issue #59). NLopt returns the last
    // evaluated `x0`, not the best-seen one — when the stagnation guard
    // short-circuits, the last few evals return `best_ofv` with zero
    // gradient and the optimizer can drift off the true minimum before
    // termination. Replacing `x0` with the best-seen xs guarantees the
    // final inner loop and covariance step run at the actual minimum.
    let mut best_seen_ofv: Option<f64> = None;
    if let Some((best_xs, best_ofv)) = best_seen.lock().unwrap().clone() {
        if best_xs.len() == n {
            x0.copy_from_slice(&best_xs);
            best_seen_ofv = Some(best_ofv);
            if options.verbose {
                eprintln!(
                    "Restored best-seen point (OFV = {:.6}) for final inner loop \
                     and covariance step.",
                    best_ofv,
                );
            }
        }
    }

    // Unscale x0 back from optimizer space to real (log/Cholesky) space.
    for i in 0..n {
        x0[i] *= scale[i];
    }

    let final_params = unpack_params(&x0, init_params);
    let final_is_mixture = final_params.mixture.is_some();

    // Final inner loop at converged parameters. Mixture (#977 Phase 3): the OFV
    // is the K-fold log-sum-exp and the reported EBEs are the MIXEST class.
    let (final_ehs, final_hms, final_kappas, final_ofv, final_mixture_posteriors) =
        if final_is_mixture {
            let m = crate::estimation::mixture::mixture_ofv(
                model,
                population,
                &final_params,
                options,
                None,
            );
            // #985: carry the MIXEST class's per-occasion κ̂ into postfit. Empty for a
            // non-IOV mixture, so the downstream `kappas.is_empty()` branches (sdtab
            // IPRED/IWRES/CWRES, per-subject OFV, κ shrinkage, `.fitrx` `ebe_kappas`)
            // behave exactly as before for non-IOV models, and reflect the IOV the fit
            // actually used for an IOV mixture instead of κ = 0.
            (
                m.mixest_etas,
                m.mixest_h_mats,
                m.mixest_kappas,
                m.ofv,
                Some(MixturePosteriors {
                    pmix: m.pmix,
                    mixest: m.mixest,
                }),
            )
        } else {
            let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
            let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm(
                model,
                population,
                &final_params,
                options.inner_maxiter,
                options.inner_tol,
                None,
                Some(&final_mu_k),
                options.min_obs_for_convergence_check as usize,
                options.inner_restarts,
            );
            let final_nll = pop_nll_opts(
                model,
                population,
                &final_params,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            );
            (final_ehs, final_hms, final_kappas, 2.0 * final_nll, None)
        };

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // Resolve a deferred `Failure`/`ForcedStop` verdict (see
    // `stationarity_check_pending`). NLopt's analytic-gradient L-BFGS default
    // (#639) returns a bare `NLOPT_FAILURE` *at* a plateaued optimum — its line
    // search can no longer beat an OFV already flat to ~8 significant figures —
    // and the raw enum then libels a finished fit as `converged=false`. That
    // both fails the honest convergence tests and tags the point non-stationary
    // right before the FD-of-OFV covariance step, whose R-matrix is only
    // well-conditioned at a true minimum (issue #751).
    //
    // The analytic gradient norm is *not* a usable stationarity proxy here: at
    // these genuine optima it still reads O(1) (npde ≈ 1.8, schnider ≈ 0.05)
    // because the best-point EBEs the outer gradient reuses differ slightly from
    // the cold-restart `final_ehs`, and because weakly-identified directions
    // carry a large scaled ∂OFV/∂x at a flat OFV. Decide on the OFV trace
    // instead — the quantity that actually defines convergence for a noisy FOCE
    // objective:
    //   (a) plateau — the best OFV has not improved by more than
    //       `PLATEAU_OFV_THRESHOLD` for at least `PLATEAU_MIN_FLAT_EVALS` evals
    //       (a real stall, e.g. SS-oral quitting after ~5 evals still plunging,
    //       has no flat tail); and
    //   (b) self-consistency — re-running the inner loop cold at the restored
    //       best point reproduces the best-seen OFV (the SS-oral stall's
    //       best-seen 83.3 vs cold 121.4 exposes a warm-start artifact).
    // Both must hold; a genuine mid-descent stall fails at least one, so this
    // never papers over non-convergence.
    if stationarity_check_pending {
        let (_, last_sig_feasible_eval, feasible_evals) = *plateau_tracker.lock().unwrap();
        if failure_is_converged_plateau(
            feasible_evals,
            last_sig_feasible_eval,
            best_seen_ofv,
            final_ofv,
        ) {
            converged = true;
        }
        if options.verbose {
            let flat_tail = feasible_evals.saturating_sub(last_sig_feasible_eval);
            eprintln!(
                "Plateau check: flat_tail = {} feasible evals (min {}), feasible_evals = {}, \
                 best-seen {:?} vs final {:.6} → converged = {}",
                flat_tail,
                PLATEAU_MIN_FLAT_EVALS,
                feasible_evals,
                best_seen_ofv,
                final_ofv,
                converged,
            );
        }
    }

    // Covariance step (skip if user cancelled — it's expensive and the result
    // will be discarded by the top-level fit() anyway).
    // Mixture (#983 Phase 6): `compute_covariance` builds the FD Hessian on the
    // K-fold mixture OFV when `template.mixture` is set, so the step runs for
    // mixtures too. The `final_ehs`/`final_hms` handed in are the MIXEST-class
    // EBEs; the mixture branch reconverges per class internally and does not use
    // them as a warm start.
    let (covariance_matrix, covariance_wall_time_secs, sir_fallback_proposal) = {
        let out = crate::estimation::covariance::run_covariance_step(
            &x0,
            init_params,
            model,
            population,
            &final_ehs,
            &final_hms,
            &final_kappas,
            options,
            options.verbose.then_some("Computing covariance matrix..."),
        );
        let crate::estimation::covariance::CovStepOutcome {
            matrix,
            wall_time_secs,
            warnings: cov_warnings,
            sir_fallback_proposal,
        } = out;
        warnings.extend(cov_warnings);
        (matrix, wall_time_secs, sir_fallback_proposal)
    };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }

    let final_gradient = last_gradient.lock().unwrap().clone();

    let ebe_final = ebe_accum.lock().unwrap();
    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        // NLopt doesn't expose an "iteration" count (BOBYQA/SLSQP don't have
        // iterations in the textbook sense), so report the number of
        // objective-function evaluations instead — the only monotone
        // progress counter NLopt exposes, and the quantity most users
        // actually care about ("how much work did the fit do").
        n_iterations: n_evals_outer.load(Ordering::Relaxed),
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        ebe_convergence_warnings: ebe_final.n_convergence_warnings as u32,
        max_unconverged_subjects: ebe_final.max_unconverged as u32,
        total_ebe_fallbacks: ebe_final.total_fallback as u32,
        final_gradient,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        // The exact packed vector this stage's inline covariance step used (#816
        // follow-up): reused by `run_covariance` to avoid re-decomposing omega.
        packed_estimate: Some(x0.clone()),
        mixture_posteriors: final_mixture_posteriors,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hand-rolled BFGS outer optimizer (legacy fallback)
// ═══════════════════════════════════════════════════════════════════════════

fn optimize_bfgs(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let n = x.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();
    let mut cached_etas: Vec<DVector<f64>> = vec![DVector::zeros(n_eta); n_subj];

    // Closures operating on unscaled real (log/Cholesky) space.
    let ofv_at_fixed = |x: &[f64],
                        eta_hats: &[DVector<f64>],
                        h_matrices: &[DMatrix<f64>],
                        kappas: &[Vec<DVector<f64>>]|
     -> f64 {
        let params = unpack_params(x, init_params);
        2.0 * pop_nll_opts(
            model, population, &params, eta_hats, h_matrices, kappas, options,
        )
    };

    let f_only = |x: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        let ofv = 2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
        if ofv.is_finite() {
            ofv
        } else {
            1e20
        }
    };

    let fdfg = |x: &[f64],
                prev_etas: &[DVector<f64>],
                grad_eval_idx: &mut usize|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        let ofv = ofv_at_fixed(x, &ehs, &hms, &kappas);
        // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x).
        let g = population_gradient(
            x,
            n_subj,
            init_params,
            model,
            population,
            &ehs,
            &hms,
            &kappas,
            &bounds,
            options,
            grad_eval_idx,
        );
        let f = if ofv.is_finite() { ofv } else { 1e20 };
        (f, g, ehs, hms)
    };

    // Per-element scale factors for the BFGS outer loop.
    let scale: Vec<f64> = match resolve_scaling(options.parameter_scaling, options.optimizer) {
        ParameterScaling::Rescale2 => compute_rescale2_scale(&bounds),
        ParameterScaling::Abs => compute_scale(&x),
        ParameterScaling::None | ParameterScaling::Auto => {
            if options.scale_params {
                compute_scale(&x)
            } else {
                vec![1.0; n]
            }
        }
    };
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    let bounds_s = PackedBounds {
        lower: lower_s,
        upper: upper_s,
    };

    // Wrappers that operate in scaled space; unscale before calling base closures.
    let fdfg_s = |xs: &[f64],
                  prev_etas: &[DVector<f64>],
                  grad_eval_idx: &mut usize|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let (f, g_r, ehs, hms) = fdfg(&x_r, prev_etas, grad_eval_idx);
        let g_s: Vec<f64> = (0..n).map(|i| g_r[i] * scale[i]).collect();
        (f, g_s, ehs, hms)
    };

    let f_only_s = |xs: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        f_only(&x_r, prev_etas)
    };

    // Scale initial x into optimizer space.
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();

    // Gradient-evaluation counter driving the reconverge schedule; advanced
    // inside `population_gradient` so it counts actual gradient evals (not
    // outer iterations or objective-only line-search probes).
    let mut grad_eval_idx = 0usize;
    let (mut f_val, mut g, ehs, _) = fdfg_s(&xs, &cached_etas, &mut grad_eval_idx);
    cached_etas = ehs;

    // EBE warm-start predictor (Almquist Eq. 48): extrapolate each subject's EBE
    // to the next outer point via dη̂/dx, so the inner solve starts closer and
    // needs fewer iterations. dη̂/dx is interaction-independent (shared inner
    // objective), so it engages for both FOCE and FOCEI on analytical models;
    // set FERX_EBE_PREDICTOR=0 to disable (A/B timing). When the Jacobian is
    // unavailable it degrades to plain warm-start from prior η̂.
    let use_predictor = crate::sens::provider::sens_supported(model)
        && std::env::var("FERX_EBE_PREDICTOR")
            .map(|v| v != "0")
            .unwrap_or(true);
    let mut x_anchor_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
    let mut last_jac: Option<Vec<Vec<DVector<f64>>>> = if use_predictor {
        crate::estimation::sens_outer_gradient::population_eta_dx(
            model,
            population,
            init_params,
            &x_anchor_real,
            &cached_etas,
        )
    } else {
        None
    };

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, f_val);
    }

    // Two outer Hessian strategies share this loop: `Optimizer::Lbfgs` uses a
    // limited-memory L-BFGS two-loop recursion over the last `LBFGS_MEMORY`
    // curvature pairs (no dense matrix); `Optimizer::Bfgs` keeps the full inverse
    // Hessian `h_inv`. Both consume the same analytic gradient and Eq. 48 warm
    // EBEs below.
    let use_lbfgs = matches!(options.optimizer, Optimizer::Lbfgs);
    const LBFGS_MEMORY: usize = 10;
    let mut s_hist: Vec<DVector<f64>> = Vec::new();
    let mut y_hist: Vec<DVector<f64>> = Vec::new();
    let mut h_inv = DMatrix::<f64>::identity(n, n);
    let mut converged = false;
    let mut n_iterations = 0;
    let mut stall_count = 0;

    for iter in 1..=options.outer_maxiter {
        n_iterations = iter;

        if crate::cancel::is_cancelled(&options.cancel) {
            warnings.push("cancelled by user".to_string());
            break;
        }

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Snapshot the scaled gradient that `g_norm` is taken from, before the
        // step overwrites `g` with `g_new`. The trace's `grad:*` columns log
        // this vector so `sqrt(Σ gᵢ²) == grad_norm` holds (#640). Like the
        // existing `grad_norm` column, it reflects the pre-step point.
        let g_for_trace: Vec<f64> = g.to_vec();
        if g_norm < options.outer_gtol {
            if options.verbose {
                eprintln!("Converged at iteration {} (|g| = {:.2e})", iter, g_norm);
            }
            converged = true;
            break;
        }

        let mut d: Vec<f64> = if use_lbfgs {
            lbfgs_two_loop(&g, &s_hist, &y_hist)
        } else {
            let g_vec = DVector::from_column_slice(&g);
            (-&h_inv * &g_vec).iter().copied().collect()
        };

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 || !dg.is_finite() {
            // Non-descent direction: discard curvature memory and take steepest
            // descent (L-BFGS clears its history; dense BFGS resets `h_inv`).
            d = g.iter().map(|gi| -gi).collect();
            s_hist.clear();
            y_hist.clear();
            h_inv = DMatrix::identity(n, n);
        }

        let alpha =
            backtracking_line_search_warm(&xs, &d, &g, f_val, &bounds_s, &cached_etas, &f_only_s);

        if alpha < 1e-18 {
            stall_count += 1;
            if stall_count >= 10 {
                if options.verbose {
                    eprintln!("Stopping: line search stalled at iteration {}", iter);
                }
                break;
            }
            s_hist.clear();
            y_hist.clear();
            h_inv = DMatrix::identity(n, n);
            continue;
        }
        stall_count = 0;

        let xs_old = xs.clone();
        for i in 0..n {
            xs[i] = (xs[i] + alpha * d[i]).clamp(bounds_s.lower[i], bounds_s.upper[i]);
        }

        // Eq. 48: predict the accepted point's EBEs from the anchor before the
        // inner solve; falls back to plain warm-start when no Jacobian.
        let x_new_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let warm: Vec<DVector<f64>> = match &last_jac {
            Some(jac) => crate::estimation::sens_outer_gradient::predict_warm_etas(
                &cached_etas,
                jac,
                &x_anchor_real,
                &x_new_real,
            ),
            None => cached_etas.clone(),
        };
        let (f_new, g_new, ehs, _) = fdfg_s(&xs, &warm, &mut grad_eval_idx);
        cached_etas = ehs;
        if use_predictor {
            last_jac = crate::estimation::sens_outer_gradient::population_eta_dx(
                model,
                population,
                init_params,
                &x_new_real,
                &cached_etas,
            );
            x_anchor_real = x_new_real;
        }

        if use_lbfgs {
            // Push the new curvature pair (s = Δx, y = Δg) with the same `s·y > 0`
            // filter `bfgs_update` uses, capping the history at `LBFGS_MEMORY`.
            let s = DVector::from_iterator(n, (0..n).map(|i| xs[i] - xs_old[i]));
            let y = DVector::from_iterator(n, (0..n).map(|i| g_new[i] - g[i]));
            if s.dot(&y) > 1e-12 {
                s_hist.push(s);
                y_hist.push(y);
                if s_hist.len() > LBFGS_MEMORY {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
            }
        } else {
            bfgs_update(&mut h_inv, &xs, &xs_old, &g_new, &g, n);
        }

        let prev_ofv = f_val;
        f_val = f_new;
        g = g_new;

        if options.verbose && (iter % 10 == 0 || iter <= 5) {
            eprintln!(
                "Iter {:>4}: OFV = {:.6}  |g| = {:.2e}  alpha = {:.2e}",
                iter, f_val, g_norm, alpha
            );
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm: f64 = (0..n)
                .map(|i| (xs[i] - xs_old[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let method_str = match options.method {
                EstimationMethod::FoceI => "focei",
                _ => "foce",
            };
            let optimizer_str = match options.optimizer {
                Optimizer::Lbfgs => "lbfgs",
                _ => "bfgs",
            };
            // Recompute the real (unscaled) point rather than reuse
            // `x_new_real`, which may already have been moved into the
            // predictor's anchor above.
            let x_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            let values = crate::estimation::parameterization::coordinate_values(&unpack_params(
                &x_real,
                init_params,
            ));
            crate::estimation::trace::write_foce(
                iter,
                method_str,
                f_val,
                Some(g_norm),
                Some(step_norm),
                optimizer_str,
                None,
                None,
                &values,
                Some(&g_for_trace),
            );
        }

        // Checkpoint (#755): recompute the unscaled packed point when a write is
        // due (the trace's `x_real` is scoped to the trace block above).
        if crate::io::checkpoint::is_due() {
            let x_real: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            crate::io::checkpoint::maybe_write(iter, f_val, &x_real);
        }

        let rel_change = (f_val - prev_ofv).abs() / (f_val.abs() + 1.0);
        if rel_change < 1e-8 && g_norm < 0.1 {
            if options.verbose {
                eprintln!(
                    "Converged at iteration {} (rel OFV change: {:.2e}, |g| = {:.2e})",
                    iter, rel_change, g_norm
                );
            }
            converged = true;
            break;
        }
    }

    // Unscale xs back to real (log/Cholesky) space for unpacking and covariance.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let final_params = unpack_params(&x_final, init_params);
    let bfgs_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cached_etas),
        Some(&bfgs_final_mu_k),
        options.min_obs_for_convergence_check as usize,
        options.inner_restarts,
    );
    let final_ofv = ofv_at_fixed(&x_final, &final_ehs, &final_hms, &final_kappas);

    let out = crate::estimation::covariance::run_covariance_step(
        &x_final,
        init_params,
        model,
        population,
        &final_ehs,
        &final_hms,
        &final_kappas,
        options,
        options.verbose.then_some("Computing covariance matrix..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
    } = out;
    warnings.extend(cov_warnings);

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }

    OuterResult {
        // The exact packed vector this stage's inline covariance step used (#816
        // follow-up): reused by `run_covariance` to avoid re-decomposing omega.
        packed_estimate: Some(x_final.clone()),
        mixture_posteriors: None,
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
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
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Central-FD `d(OFV)/d(x)` that **re-converges the EBEs at every perturbed
/// point** (warm-started from `warm_etas`), rather than holding them fixed.
///
/// For IOV models the variance components — especially `omega_iov` — are
/// weakly identified, and the EBE response dominates their gradient: raising
/// `omega_iov` un-shrinks the per-occasion kappas and improves the fit, an
/// effect the fixed-EBE gradient ([`ad_population_gradient`]) misses entirely.
/// The result is that gradient optimizers leave `omega_iov` pinned at its
/// initial value while derivative-free methods (which re-solve the EBEs at
/// each trial point) move it freely. Re-converging the inner loop inside the
/// FD stencil restores the correct descent direction. See issue #101 rec #2.
///
/// This costs `2·n_free` inner-loop solves per gradient, so it is gated to IOV
/// models (`model.n_kappa > 0`); the non-IOV path keeps the cheap analytical
/// fixed-EBE gradient, which already converges OMEGA correctly (issue #99).
#[allow(clippy::too_many_arguments)]
fn reconverged_fd_gradient(
    x: &[f64],
    init_params: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    warm_etas: &[DVector<f64>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let n_subj = population.subjects.len();
    let fixed = packed_fixed_mask(init_params);

    // OFV at a packed point, re-solving the inner loop (warm-started). Matches
    // the objective closure's definition: 2·pop_nll, guarded to 1e20 on
    // non-finite or excess EBE non-convergence.
    let eval = |xv: &[f64]| -> f64 {
        let params = unpack_params(xv, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(warm_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
        );
        let raw = 2.0 * pop_nll_opts(model, population, &params, &ehs, &hms, &kappas, options);
        if !raw.is_finite()
            || ebe_guard_rejects(&ebe_stats, n_subj, raw, options.max_unconverged_frac)
        {
            1e20
        } else {
            raw
        }
    };
    // Same bounded central-difference policy as the per-subject reconverged-FD gradients —
    // shared so the `eps`/clamp/`is_finite`-drop convention can't drift (#466 review round 4 #8).
    central_diff_packed(x, &fixed, bounds, eval)
}

/// Bounded central-difference of a packed-space scalar `eval`, skipping fixed
/// coordinates and dropping non-finite differences to zero. Shared by the non-IOV and
/// IOV per-subject reconverged-FD gradients so the two cannot drift (#466 review round 2).
fn central_diff_packed(
    x: &[f64],
    fixed: &[bool],
    bounds: &PackedBounds,
    eval: impl Fn(&[f64]) -> f64,
) -> Vec<f64> {
    let n = x.len();
    let eps = 1e-4;
    let mut grad = vec![0.0_f64; n];
    let mut xw = x.to_vec();
    for k in 0..n {
        if fixed[k] {
            continue;
        }
        let h = eps * (1.0 + x[k].abs());
        let xp = (x[k] + h).min(bounds.upper[k]);
        let xm = (x[k] - h).max(bounds.lower[k]);
        let denom = xp - xm;
        if denom.abs() < 1e-16 {
            continue;
        }
        xw[k] = xp;
        let fp = eval(&xw);
        xw[k] = xm;
        let fm = eval(&xw);
        xw[k] = x[k];
        let d = (fp - fm) / denom;
        if d.is_finite() {
            grad[k] = d;
        }
    }
    grad
}

/// Central-FD per-subject packed gradient `dᵢ = d(nllᵢ)/dx` that **re-converges
/// that one subject's EBE** (warm-started) at every perturbed point. The single-
/// subject analog of [`reconverged_fd_gradient`], used to fill the handful of
/// subjects the analytic provider can't handle (SS+reset, time-varying
/// covariates, modeled-duration doses, EVID=2 reset) inside the otherwise-exact
/// analytic population gradient. Because the EBEs are re-solved at each ±h, the
/// Ω/σ EBE-response is included — the term the θ-only fixed-EBE fallback drops,
/// whose absence stalled the gradient optimizers (focei-slsqp-fixed-ebe-gradient-bias).
/// Returns `d(nllᵢ)/dx` (length `x.len()`); the caller scales by 2 and zeroes
/// fixed coordinates, matching the analytic per-subject convention.
#[allow(clippy::too_many_arguments)]
fn subject_reconverged_fd_gradient(
    x: &[f64],
    init_params: &ModelParameters,
    model: &CompiledModel,
    subject: &Subject,
    warm_eta: &DVector<f64>,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let fixed = packed_fixed_mask(init_params);
    // Subject marginal NLL at a packed point, re-solving this subject's EBE
    // (warm-started from `warm_eta`). Mirrors the objective's per-subject term
    // (`foce_subject_nll`, summed by `pop_nll`); non-finite → NaN so the central
    // difference drops to zero for that coordinate.
    let eval = |xv: &[f64]| -> f64 {
        let params = unpack_params(xv, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let ebe = find_ebe(
            model,
            subject,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(warm_eta.as_slice()),
            Some(&mu_k),
            0,
        );
        crate::stats::likelihood::foce_subject_nll(
            model,
            subject,
            &params.theta,
            &ebe.eta,
            &ebe.h_matrix,
            &params.omega,
            &params.sigma.values,
            options.interaction,
        )
    };
    central_diff_packed(x, &fixed, bounds, eval)
}

/// Per-subject reconverged-FD packed gradient for an **IOV** subject — the IOV analogue of
/// [`subject_reconverged_fd_gradient`], used to salvage subjects outside the analytic IOV
/// scope without dropping the whole population to FD (#466 review round 2). `find_ebe`
/// dispatches to the IOV joint (η_bsv, κ) EBE for `n_kappa > 0`, and the marginal uses the
/// IOV objective `foce_subject_nll_iov` (the same one `pop_nll` sums).
fn subject_reconverged_fd_gradient_iov(
    x: &[f64],
    init_params: &ModelParameters,
    model: &CompiledModel,
    subject: &Subject,
    warm_eta: &DVector<f64>,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let fixed = packed_fixed_mask(init_params);
    let eval = |xv: &[f64]| -> f64 {
        let params = unpack_params(xv, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let ebe = find_ebe(
            model,
            subject,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(warm_eta.as_slice()),
            Some(&mu_k),
            0,
        );
        crate::stats::likelihood::foce_subject_nll_iov(
            model,
            subject,
            &params.theta,
            &ebe.eta,
            &ebe.h_matrix,
            &params.omega,
            &params.sigma.values,
            options.interaction,
            &ebe.kappas,
            params
                .omega_iov
                .as_ref()
                .expect("IOV model (n_kappa > 0) has omega_iov"),
        )
    };
    central_diff_packed(x, &fixed, bounds, eval)
}

/// Non-IOV population gradient assembled **per subject**: the exact analytic
/// (Almquist) gradient — including the EBE response on every θ/Ω/σ block — for
/// every subject inside the provider's scope, and a per-subject
/// [`subject_reconverged_fd_gradient`] for each subject outside it (or whose
/// analytic gradient came back non-finite). This replaces the all-or-nothing
/// [`population_gradient_sens`]: previously a single out-of-scope subject forced
/// the whole population onto the θ-only fixed-EBE gradient, whose biased Ω/σ
/// block left the variance components pinned at their start and stalled
/// SLSQP/L-BFGS/MMA above the derivative-free optimum
/// (focei-slsqp-fixed-ebe-gradient-bias). Returns the packed `2·Σᵢ dᵢ` with
/// fixed coordinates zeroed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_sens_mixed(
    x: &[f64],
    init_params: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    ehs: &[DVector<f64>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let np = x.len();
    let per_sub = crate::estimation::sens_outer_gradient::per_subject_packed_gradients(
        model,
        population,
        init_params,
        x,
        ehs,
        options.interaction,
        None,
    );
    let filled: Vec<Vec<f64>> = per_sub
        .into_par_iter()
        .enumerate()
        .map(|(i, gi)| match gi {
            // Keep the exact analytic gradient for in-scope, finite subjects.
            Some(g) if g.iter().all(|v| v.is_finite()) => g,
            // Out-of-scope (or non-finite analytic) → reconverged per-subject FD.
            _ => subject_reconverged_fd_gradient(
                x,
                init_params,
                model,
                &population.subjects[i],
                &ehs[i],
                bounds,
                options,
            ),
        })
        .collect();
    let mut grad = vec![0.0f64; np];
    for gi in &filled {
        for k in 0..np {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(init_params);
    for k in 0..np {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    grad
}

/// **IOV** population gradient assembled **per subject** — the IOV analogue of
/// [`population_gradient_sens_mixed`]: the exact analytic stacked-η / block-Ω gradient for
/// every in-scope subject, and a per-subject reconverged-FD gradient
/// ([`subject_reconverged_fd_gradient_iov`]) for any out-of-scope (or non-finite) one.
/// Replaces the former all-or-nothing IOV outer gradient, which dropped the *whole*
/// population to FD on the first out-of-scope subject — so a single infusion / steady-state
/// / wide-axis subject no longer forces the entire fit onto FD (#466 review round 2).
/// Returns the packed `2·Σᵢ dᵢ` with fixed coordinates zeroed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_gradient_sens_iov_mixed(
    x: &[f64],
    init_params: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    ehs: &[DVector<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let np = x.len();
    let per_sub = crate::estimation::sens_outer_gradient::per_subject_packed_gradients_iov(
        model,
        population,
        init_params,
        x,
        ehs,
        kappas,
        options.interaction,
    );
    let filled: Vec<Vec<f64>> = per_sub
        .into_par_iter()
        .enumerate()
        .map(|(i, gi)| match gi {
            Some(g) if g.iter().all(|v| v.is_finite()) => g,
            _ => subject_reconverged_fd_gradient_iov(
                x,
                init_params,
                model,
                &population.subjects[i],
                &ehs[i],
                bounds,
                options,
            ),
        })
        .collect();
    let mut grad = vec![0.0f64; np];
    for gi in &filled {
        for k in 0..np {
            grad[k] += 2.0 * gi[k];
        }
    }
    let fixed = packed_fixed_mask(init_params);
    for k in 0..np {
        if fixed[k] {
            grad[k] = 0.0;
        }
    }
    grad
}

/// Compute `d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x)` by summing per-subject
/// gradients in parallel.  ETAs are fixed at their current EBE values.
///
/// `kappas` must have length `n_subj`; each `kappas[i]` is the IOV kappa
/// vector for subject `i` (empty for non-IOV models).
fn ad_population_gradient(
    x: &[f64],
    n_subj: usize,
    init_params: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    ehs: &[DVector<f64>],
    hms: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    debug_assert_eq!(ehs.len(), n_subj);
    debug_assert_eq!(hms.len(), n_subj);
    debug_assert_eq!(kappas.len(), n_subj);
    let np = x.len();
    // For FOCEI (interaction), add the `log|H̃|` EBE-response term `t_i` (the
    // #274/#289 Δ) the fixed-η̂ analytic gradient drops, so slsqp/L-BFGS see the
    // full marginal gradient and reach the true minimum instead of stalling
    // above it. Reuses the Laplace cache the gradient just formed (one extra
    // n_eta×n_eta solve per subject); θ-block (mu-ref) only, zero for additive
    // error. IOV routes through the reconverged-FD gradient, not here, so this
    // only affects non-IOV FOCEI gradient steps.
    let per_subj: Vec<Vec<f64>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            let (_, mut gi, cache) =
                crate::estimation::gauss_newton::subject_nll_pop_grad_with_cache(
                    x,
                    init_params,
                    model,
                    population,
                    i,
                    &ehs[i],
                    &hms[i],
                    kappas[i].as_slice(),
                    bounds,
                    options,
                );
            if let Some(c) = cache.as_ref() {
                if let Some(t) = crate::estimation::gauss_newton::subject_eta_response_correction(
                    Some(c),
                    x,
                    init_params,
                    model,
                    population,
                    i,
                    &ehs[i],
                    &hms[i],
                    bounds,
                    options,
                ) {
                    for (g, ti) in gi.iter_mut().zip(t.iter()) {
                        *g += *ti;
                    }
                }
            }
            gi
        })
        .collect();
    assemble_population_gradient(&per_subj, np)
}

/// Assemble the covariance-step population gradient `2·Σᵢ gᵢ` from per-subject
/// gradients, summing over subjects in index order. Both the parallel
/// [`ad_population_gradient`] and the serial per-point gradient inside
/// [`compute_covariance`] route their reduction through here, so there is a
/// single summation order — which is what keeps the flattened (#256) covariance
/// bit-identical to the pre-flatten serial stencil for FOCE. `np` is the packed
/// parameter count; each `gᵢ` has length `np`.
fn assemble_population_gradient(per_subj: &[Vec<f64>], np: usize) -> Vec<f64> {
    (0..np)
        .map(|k| per_subj.iter().map(|gi| gi[k]).sum::<f64>() * 2.0)
        .collect()
}

/// Whether gradient evaluation number `grad_idx` (0-based, per optimization
/// run) should use the expensive reconverged path on a **non-IOV** model.
///
/// Driven by `reconverge_gradient_interval`: `0` disables it entirely; `N`
/// fires on evals `0, N, 2N, …`. The `interval != 0` guard also short-circuits
/// the modulo, so a `0` interval can never divide by zero. IOV models
/// reconverge unconditionally and never consult this.
fn reconverge_this_eval(options: &FitOptions, grad_idx: usize) -> bool {
    let interval = options.reconverge_gradient_interval;
    interval != 0 && grad_idx % interval == 0
}

/// `FERX_SENS_CHECK=1` enables the per-eval analytic-vs-reconverged-FD outer
/// gradient cross-check in [`population_gradient`] (off by default — it doubles
/// the gradient cost, so it is a CI/diagnostic backstop, not a production path).
fn sens_check_enabled() -> bool {
    std::env::var("FERX_SENS_CHECK")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Population gradient dispatcher. IOV models (`n_kappa > 0`) and M3-censored
/// models use the EBE-reconverging FD gradient — their weakly-identified variance
/// components / non-Gaussian censored rows need it — and everything else uses the
/// cheap analytical fixed-EBE gradient unless the `reconverge_gradient_interval`
/// schedule opts this evaluation into the reconverged path.
///
/// `grad_eval_idx` is the caller's count of gradient evaluations so far; this
/// function reads it to apply the schedule and then advances it. Owning the
/// counter here keeps every optimizer path (NLopt objective, NLopt fallback,
/// BFGS) on one definition of "gradient evaluation" — they can't drift apart in
/// how they count or pick the gradient.
#[allow(clippy::too_many_arguments)]
fn population_gradient(
    x: &[f64],
    n_subj: usize,
    init_params: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    ehs: &[DVector<f64>],
    hms: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
    grad_eval_idx: &mut usize,
) -> Vec<f64> {
    let reconverge = reconverge_this_eval(options, *grad_eval_idx);
    *grad_eval_idx += 1;
    // AGQ minimises a *different* objective (the quadrature marginal, not the FOCE/Laplace
    // one), so every analytic and fixed-EBE gradient below — all of them closed forms of
    // the FOCE marginal — is simply the gradient of the wrong function. Feeding one to the
    // outer optimizer would not fail loudly; it would converge, smoothly, to the FOCE
    // optimum while reporting AGQ OFVs. AGQ has its own gradient.
    if let Some(n_nodes) = options.agq_nodes() {
        // Preferred: AGQ's own exact gradient — the analytic posterior-weighted score over
        // the nodes (Fisher identity) plus the grid-response term — which needs no inner
        // re-solve, against the FD path's `2·n_free` *full population objective*
        // re-evaluations. Exact at every `n_agq`. See `estimation::agq`.
        // `reconverge_gradient_interval` is honoured here too: it is the documented escape
        // hatch onto the numeric path, so it must override the analytic gradient for AGQ
        // exactly as it does for FOCE/FOCEI below.
        // `agq_population_gradient` is the analytic gradient of the quadrature objective for
        // **either** anchor: the fixed-node score is anchor-independent, and the grid-response
        // term differences whichever Hessian scales the grid (exact for `laplace`,
        // Gauss-Newton for `focei` — `anchor` selects it, matching the objective).
        if !reconverge && crate::estimation::agq::analytic_gradient_available(model) {
            let params = unpack_params(x, init_params);
            if let Some(mut g) = crate::estimation::agq::agq_population_gradient(
                model,
                population,
                &params,
                init_params,
                x,
                ehs,
                kappas,
                n_nodes,
                options.hessian_anchor(),
            ) {
                // Fixed coordinates carry no gradient, matching the analytic FOCE path.
                let fixed = packed_fixed_mask(init_params);
                for (i, gi) in g.iter_mut().enumerate() {
                    if fixed[i] {
                        *gi = 0.0;
                    }
                }
                return g;
            }
        }
        // Fallback (always correct, just slower): central-difference the real objective,
        // re-solving the inner loop at each perturbed point so the response of η̂ to the
        // population parameters is captured too.
        return reconverged_fd_gradient(x, init_params, model, population, ehs, bounds, options);
    }
    // M3-censored models now have an exact analytic censored gradient on both the
    // FOCEI (`subject_packed_gradient` + `prepare`'s M3 branch) and the FOCE
    // (`subject_packed_gradient_foce`, censored rows excluded from R̃ and added as
    // `−logΦ`) paths, so M3 takes the analytic path like any other fit.
    let force_reconverge = reconverge;
    // Analytic-sensitivity gradient (Almquist 2015 Eq. 23, closed form via the
    // `sens` provider): the exact marginal FOCEI gradient including the Eq. 46
    // EBE response on every θ/Ω/σ block — no fixed-EBE bias, no FD noise, so it
    // supersedes both branches below where it applies. Gated to the supported
    // analytical PK scope (1-/2-/3-cpt); `population_gradient_sens` returns `None`
    // (→ the existing FD/Laplace path) if any subject is outside provider scope.
    // FOCEI uses the Almquist Laplace marginal (R at f(η̂), ½c̃ᵀc̃ in H̃); plain
    // FOCE uses the Sheiner–Beal linearized marginal (R̃ = JΩJᵀ + R⁰). Both have
    // exact closed-form gradients here, sharing the same EBE/inner-Hessian core.
    //
    // `reconverge` (driven by `reconverge_gradient_interval`) overrides the
    // analytic path: it is the documented opt-out / escape hatch (PR #381 review
    // findings #6/#7). Setting `reconverge_gradient_interval = 1` forces the
    // reconverged-FD gradient on every eval even for analytical models — so the
    // numeric fallback remains available if the analytic gradient is ever
    // suspect, and the setting is honoured rather than silently ignored.
    // IOV-analytical models route to the dedicated stacked-η / block-Ω assembly
    // (both FOCEI and FOCE — see the interaction branch below). Their gradient
    // needs the per-occasion κ̂ alongside the BSV EBEs, so it is dispatched
    // separately from the non-IOV `sens_supported` path.
    // Covers both the closed-form analytical IOV provider and the ODE IOV provider
    // (RHS-program models); both produce the stacked-η / block-Ω assembly the IOV
    // gradient entry points consume (#439 ODE IOV).
    let iov_analytic = crate::sens::provider::iov_sens_supported(model);
    // `gradient = fd` forces the numeric path for the outer gradient too (the inner
    // EBE gradient honours it via `analytic_inner_grad_supported`), so the option
    // fully disables the analytic sensitivities rather than only the inner half.
    // `analytic_outer_gradient_available` is the shared predicate that
    // `Optimizer::resolve_auto` and `build_info::gradient_method_outer` also use,
    // so the `auto` optimizer cannot pick a gradient-based optimizer while this
    // gate falls through to FD (#490 review).
    if !force_reconverge && crate::sens::provider::analytic_outer_gradient_available(model) {
        let g = if iov_analytic {
            // Per-subject: exact analytic for in-scope subjects, per-subject reconverged-FD
            // for out-of-scope ones — always `Some`, mirroring the non-IOV mixed path. A
            // single out-of-scope subject no longer drops the whole population to FD (and
            // so the reported `gradient_method` stays accurate) (#466 review round 2).
            Some(population_gradient_sens_iov_mixed(
                x,
                init_params,
                model,
                population,
                ehs,
                kappas,
                bounds,
                options,
            ))
        } else {
            // Non-IOV: assemble per subject — exact analytic for in-scope
            // subjects, per-subject reconverged-FD for the few out-of-scope ones.
            // Always `Some`; the finiteness backstop below still guards it. This
            // is the fix for focei-slsqp-fixed-ebe-gradient-bias: one out-of-scope
            // subject no longer drops the whole population to the biased θ-only
            // fixed-EBE fallback (`ad_population_gradient`).
            Some(population_gradient_sens_mixed(
                x,
                init_params,
                model,
                population,
                ehs,
                bounds,
                options,
            ))
        };
        if let Some(g) = g {
            // Always-on finiteness backstop: a non-finite analytic component (the
            // class PR #381 review finding #3 warns about — a degenerate acos /
            // singular eigenvalue producing NaN) would poison the optimizer. Rather
            // than return it, fall through to the numeric path. Cheap (a scan of a
            // length-`np` vector) and reliable, unlike a mid-run magnitude compare
            // to reconverged-FD: with loosely-converged EBEs the analytic and
            // reconverged-FD gradients legitimately differ away from the optimum
            // (they agree to ~1e-11 only at convergence — see the unit tests), so a
            // value-tolerance assert here cries wolf. With FERX_SENS_CHECK=1 the
            // divergence is additionally reported for diagnosis.
            if g.iter().all(|v| v.is_finite()) {
                if sens_check_enabled() {
                    let fd = reconverged_fd_gradient(
                        x,
                        init_params,
                        model,
                        population,
                        ehs,
                        bounds,
                        options,
                    );
                    let max_abs = g
                        .iter()
                        .chain(fd.iter())
                        .fold(1e-8_f64, |m, v| m.max(v.abs()));
                    let max_diff = g
                        .iter()
                        .zip(fd.iter())
                        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                    eprintln!(
                        "[FERX_SENS_CHECK] analytic vs reconverged-FD outer gradient: \
                         max abs diff {max_diff:.3e}, rel {:.2e} (interaction={})",
                        max_diff / max_abs,
                        options.interaction
                    );
                }
                return g;
            } else if options.verbose {
                eprintln!(
                    "warning: non-finite analytic outer gradient — falling back to the numeric path"
                );
            }
        }
    }
    // IOV models always reconverge the inner EBE solution inside the gradient.
    // For non-IOV models the default is the fixed-EBE analytical/AD gradient,
    // which is far cheaper but omits the response of (η̂, H) to the population
    // parameters — an omission that stalls SLSQP well above the derivative-free
    // optimum on ill-conditioned fits. The `reconverge_gradient_interval`
    // schedule (via `reconverge_this_eval`) opts a non-IOV fit into the
    // reconverged path (see focei-slsqp-fixed-ebe-gradient-bias).
    if model.n_kappa > 0 || force_reconverge {
        reconverged_fd_gradient(x, init_params, model, population, ehs, bounds, options)
    } else {
        ad_population_gradient(
            x,
            n_subj,
            init_params,
            model,
            population,
            ehs,
            hms,
            kappas,
            bounds,
            options,
        )
    }
}

fn bfgs_update(
    h_inv: &mut DMatrix<f64>,
    x_new: &[f64],
    x_old: &[f64],
    g_new: &[f64],
    g_old: &[f64],
    n: usize,
) {
    let s: Vec<f64> = (0..n).map(|i| x_new[i] - x_old[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| g_new[i] - g_old[i]).collect();
    let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
    if sy > 1e-12 {
        let rho = 1.0 / sy;
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let eye = DMatrix::<f64>::identity(n, n);
        let rs_yt = rho * &s_vec * y_vec.transpose();
        let ry_st = rho * &y_vec * s_vec.transpose();
        let rss = rho * &s_vec * s_vec.transpose();
        *h_inv = (&eye - &rs_yt) * &*h_inv * (&eye - &ry_st) + rss;
    } else {
        *h_inv = DMatrix::identity(n, n);
    }
}

/// Limited-memory L-BFGS search direction `d = −H∇f` via the two-loop recursion
/// (Nocedal & Wright Alg. 7.4), using the most recent `(s, y)` history pairs
/// (newest last). The implicit inverse-Hessian seed is `γ·I` with
/// `γ = (sₖ·yₖ)/(yₖ·yₖ)` from the newest pair (Barzilai–Borwein scaling) — the
/// standard choice that keeps the step well-scaled without ever forming the
/// dense `n×n` matrix `bfgs_update` maintains. With no history it returns plain
/// steepest descent `−∇f`. Curvature filtering (`s·y > 0`) is enforced by the
/// caller before a pair is pushed, so every stored `ρᵢ = 1/(yᵢ·sᵢ)` is finite.
fn lbfgs_two_loop(g: &[f64], s_hist: &[DVector<f64>], y_hist: &[DVector<f64>]) -> Vec<f64> {
    let m = s_hist.len();
    debug_assert_eq!(m, y_hist.len());
    let mut q = DVector::from_column_slice(g);
    if m == 0 {
        return (-q).iter().copied().collect();
    }
    let rho: Vec<f64> = (0..m).map(|i| 1.0 / y_hist[i].dot(&s_hist[i])).collect();
    let mut alpha = vec![0.0f64; m];
    // First loop: newest → oldest.
    for i in (0..m).rev() {
        alpha[i] = rho[i] * s_hist[i].dot(&q);
        q -= alpha[i] * &y_hist[i];
    }
    // Seed with γ·I from the newest pair.
    let last = m - 1;
    let gamma = s_hist[last].dot(&y_hist[last]) / y_hist[last].dot(&y_hist[last]);
    let mut r = gamma * q;
    // Second loop: oldest → newest.
    for i in 0..m {
        let beta = rho[i] * y_hist[i].dot(&r);
        r += (alpha[i] - beta) * &s_hist[i];
    }
    (-r).iter().copied().collect()
}

fn backtracking_line_search_warm(
    x: &[f64],
    d: &[f64],
    g: &[f64],
    f0: f64,
    bounds: &PackedBounds,
    prev_etas: &[DVector<f64>],
    f_only: &dyn Fn(&[f64], &[DVector<f64>]) -> f64,
) -> f64 {
    let c1 = 1e-4;
    let n = x.len();
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
    if dg >= 0.0 {
        return 0.0;
    }

    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..30 {
        for i in 0..n {
            x_new[i] = (x[i] + alpha * d[i]).clamp(bounds.lower[i], bounds.upper[i]);
        }
        let f_new = f_only(&x_new, prev_etas);
        if f_new <= f0 + c1 * alpha * dg {
            return alpha;
        }
        alpha *= 0.5;
        if alpha < 1e-18 {
            return 0.0;
        }
    }
    0.0
}

/// Analytic covariance-step gradient with ETAs/H fixed: `2·pop_nll` with no
/// omega-prior add-back (both the SB and Laplace marginals already carry Ω —
/// #243/#249). The production stencil inlines a serial variant (plus the #274 Δ
/// correction); this thin wrapper over [`ad_population_gradient`] is retained for
/// the gradient-consistency tests that finite-difference the fixed-EBE objective.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn covariance_gradient(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Vec<f64> {
    let n_subj = population.subjects.len();
    ad_population_gradient(
        x, n_subj, template, model, population, eta_hats, h_matrices, kappas, bounds, options,
    )
}

#[cfg(test)]
#[path = "outer_optimizer_tests.rs"]
mod tests;
