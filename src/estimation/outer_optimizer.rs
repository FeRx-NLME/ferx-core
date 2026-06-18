use crate::estimation::inner_optimizer::{find_ebe, run_inner_loop_warm};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{foce_population_nll, foce_population_nll_iov};
use crate::types::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Result of outer optimization
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
}

/// Run the outer optimization loop (population parameter estimation).
pub fn optimize_population(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    match options.optimizer {
        Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma | Optimizer::Bobyqa => {
            optimize_nlopt(model, population, init_params, options)
        }
        Optimizer::Bfgs | Optimizer::Lbfgs => {
            optimize_bfgs(model, population, init_params, options)
        }
        Optimizer::TrustRegion => crate::estimation::trust_region::optimize_trust_region(
            model,
            population,
            init_params,
            options,
        ),
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

/// SLSQP overshoot guard for the scaled gradient.
///
/// NLopt LD_SLSQP starts each fit with its quasi-Newton Hessian set to
/// identity, so the QP that produces its first step has an unconstrained
/// solution d = -∇f, projected onto the box bounds. When |∇f|∞ is several
/// times larger than the bound width — which is what the AD/analytical
/// FOCE gradient added in PR #48 looks like on standard PK models (≈ 10²–10³
/// in scaled log/Cholesky space) — the projected step pins every component
/// to a corner of the box. The OFV at the corner explodes and SLSQP cannot
/// recover; theta stays byte-identical to init for the rest of the budget.
/// See issue #55.
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
/// LBFGS/MMA have line-search-style safeguards and BOBYQA is derivative-free,
/// so this is only applied on the SLSQP path.
pub(crate) fn cap_slsqp_gradient(g: &mut [f64], lower_s: &[f64], upper_s: &[f64]) -> bool {
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
        );
        let nll = pop_nll(
            model,
            population,
            &params,
            &ehs,
            &hms,
            &kappas,
            options.interaction,
        );
        let raw = 2.0 * nll;
        let frac = ebe_stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
        let guarded = raw.is_finite()
            && frac > options.max_unconverged_frac
            && options.max_unconverged_frac >= 0.0;
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
        );

        let nll = pop_nll(
            model,
            population,
            &params,
            &ehs,
            &hms,
            &kappas,
            options.interaction,
        );
        let raw_ofv = 2.0 * nll;

        let unconverged_frac = ebe_stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
        let ebe_guard = raw_ofv.is_finite()
            && unconverged_frac > options.max_unconverged_frac
            && options.max_unconverged_frac >= 0.0;
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
/// `Rescale2` to the gradient-based optimizers that benefit (`Bfgs`, `Lbfgs`,
/// `NloptLbfgs`, `Slsqp`) and `None` otherwise — critically, the derivative-free
/// default `Bobyqa` is left unscaled because `Rescale2` distorts its trust-region
/// quadratic model and regresses multi-cpt / PD fits (e.g. emax_pkpd −36.8→−13.5,
/// three_cpt_iv −730.6→−715.9). `Slsqp` is included because the bound-half-width
/// rescaling fixes its cold-start convergence — e.g. pure FOCEI/SLSQP on
/// warfarin_iov reaches OFV 307.84 from the cold default start instead of
/// stalling at 343.5 (#335). `Mma`/`TrustRegion` are left to the unscaled (legacy
/// `scale_params` / IOV-auto) branch. Non-`Auto` values pass through unchanged.
fn resolve_scaling(ps: ParameterScaling, opt: Optimizer) -> ParameterScaling {
    match ps {
        ParameterScaling::Auto => match opt {
            Optimizer::Bfgs | Optimizer::Lbfgs | Optimizer::NloptLbfgs | Optimizer::Slsqp => {
                ParameterScaling::Rescale2
            }
            _ => ParameterScaling::None,
        },
        other => other,
    }
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
    // kappa block), and SLSQP's uniform gradient cap (`cap_slsqp_gradient`,
    // applied only on the SLSQP path) otherwise rescales the whole gradient by
    // the worst (theta) component, starving the omega/omega_iov step so the
    // variance components stay pinned at their initial values. Scaling presents
    // O(1) coordinates so the cap no longer starves them. The #99 regression
    // that made scaling default-off was on non-IOV models and other algorithms
    // (notably MMA, which scaling hurts here), so scope the auto-enable to the
    // IOV + SLSQP combination that actually needs it.
    //
    // Scope note: as of #155 the default outer optimizer is `Bobyqa`, not
    // `Slsqp` — so default-IOV fits no longer hit this branch. BOBYQA is
    // gradient-free and doesn't suffer the `cap_slsqp_gradient` starvation that
    // motivates the scaling here, so leaving it disabled on the default path is
    // intentional. This auto-enable now only fires for an explicit
    // `optimizer = slsqp` on IOV models (the path it was originally written for).
    let auto_scale_iov = model.n_kappa > 0 && matches!(options.optimizer, Optimizer::Slsqp);
    let scale: Vec<f64> = match resolve_scaling(options.parameter_scaling, options.optimizer) {
        ParameterScaling::Rescale2 => compute_rescale2_scale(&bounds),
        ParameterScaling::Abs => compute_scale(&x0),
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

    // EBE stats accumulator: tracks worst unconverged count and total fallbacks.
    #[derive(Default)]
    struct EbeAccum {
        max_unconverged: usize,
        total_fallback: usize,
        n_convergence_warnings: usize,
    }
    let ebe_accum: Arc<Mutex<EbeAccum>> = Arc::new(Mutex::new(EbeAccum::default()));
    let ebe_accum_cl = Arc::clone(&ebe_accum);

    // Select NLopt algorithm
    let algo = match options.optimizer {
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::NloptLbfgs => nlopt::Algorithm::Lbfgs,
        Optimizer::Mma => nlopt::Algorithm::Mma,
        Optimizer::Bobyqa => nlopt::Algorithm::Bobyqa,
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
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);

        // Run inner loop (warm-started)
        let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&state.cached_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
        );

        // Compute OFV with fixed EBEs
        let nll = pop_nll(
            model,
            population,
            &params,
            &ehs,
            &hms,
            &kappas,
            options.interaction,
        );
        let raw_ofv = 2.0 * nll;

        // EBE convergence guard: reject step when too many subjects unconverged.
        let unconverged_frac = ebe_stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
        let ebe_guard_triggered = raw_ofv.is_finite()
            && unconverged_frac > options.max_unconverged_frac
            && options.max_unconverged_frac >= 0.0;
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

        let ofv = if ebe_guard_triggered {
            1e20
        } else if raw_ofv.is_finite() {
            raw_ofv
        } else {
            1e20
        };

        // Compute gradient if requested (central FD with fixed EBEs)
        let mut grad_norm_for_trace: Option<f64> = None;
        if let Some(g) = grad {
            // If OFV is non-finite, gradient is meaningless — use steepest ascent
            // toward center of bounds to nudge optimizer back
            if !raw_ofv.is_finite() {
                for i in 0..g.len() {
                    let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                    g[i] = 100.0 * (xs[i] - center_s);
                }
                state.n_evals += 1;
                n_evals_cl.fetch_add(1, Ordering::Relaxed);
                return ofv;
            }
            // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x); then scale for optimizer space.
            let grad_raw = population_gradient(
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
            );
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
            if matches!(algo, nlopt::Algorithm::Slsqp) {
                cap_slsqp_gradient(g, &lower_s, &upper_s);
            }
            // Gate on the global best (same reason as the `best_seen` update
            // below): `state.best_ofv` resets to INFINITY when the SLSQP
            // fallback starts, so using it here would let the fallback's first
            // eval overwrite a better gradient found by the primary run.
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
        // `best_seen` is global across the primary run and any SLSQP fallback.
        // Gate on the externally-tracked best (not `state.best_ofv`, which
        // resets to INFINITY when the fallback starts fresh) so the first
        // fallback eval at the drifted post-primary x0 can't clobber a
        // better point found earlier.
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
            crate::estimation::trace::write_foce(
                state.n_evals,
                method_str,
                ofv,
                grad_norm_for_trace,
                step_norm,
                optimizer_str,
                Some(ebe_stats.n_unconverged),
                Some(ebe_stats.n_fallback),
            );
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
        // at an arbitrary interim point; 1e-4 in scaled log-space is a
        // ~0.01% move in the natural-scale parameter, which is plenty
        // tight for NLME work.
        opt.set_xtol_rel(1e-4).unwrap();
        opt.set_ftol_rel(1e-6).unwrap();
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
        // Use very loose tolerances — FOCE objective is noisy from EBE re-estimation.
        // Let maxeval be the primary stopping criterion.
        opt.set_xtol_rel(1e-12).unwrap();
        opt.set_ftol_rel(1e-12).unwrap();
    }

    if options.verbose {
        eprintln!(
            "Starting NLopt {:?} optimization ({} parameters)...",
            algo, n
        );
    }

    // Run optimization
    let result = opt.optimize(&mut x0);

    let (mut converged, first_algo) = match &result {
        Ok((status, _)) => {
            if options.verbose {
                eprintln!("NLopt finished: {:?}", status);
            }
            (
                matches!(
                    status,
                    nlopt::SuccessState::Success
                        | nlopt::SuccessState::FtolReached
                        | nlopt::SuccessState::XtolReached
                        | nlopt::SuccessState::StopValReached
                ),
                algo,
            )
        }
        Err((fail, _)) => {
            if options.verbose {
                eprintln!("NLopt stopped: {:?}", fail);
            }
            (matches!(fail, nlopt::FailState::RoundoffLimited), algo)
        }
    };

    drop(opt);

    // Fallback: if L-BFGS failed, retry with SLSQP from current best point.
    // Skip the fallback if the user cancelled — no point burning more cycles.
    let already_slsqp = matches!(first_algo, nlopt::Algorithm::Slsqp);
    let cancelled = crate::cancel::is_cancelled(&options.cancel);
    if !converged && !already_slsqp && !cancelled {
        if options.verbose {
            eprintln!("Retrying with NLopt SLSQP from current point...");
        }

        let state2 = new_nlopt_state(n_subj, n_eta, &x0);

        let n_evals_cl2 = Arc::clone(&n_evals_outer);
        let ebe_accum_cl2 = Arc::clone(&ebe_accum);
        let best_seen_cl2 = Arc::clone(&best_seen);
        let last_gradient_cl2 = Arc::clone(&last_gradient);
        // SLSQP fallback also operates in scaled xs space (same scale as primary opt).
        let objective2 = |xs: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
            if crate::cancel::is_cancelled(&options.cancel) {
                if let Some(g) = grad {
                    for gi in g.iter_mut() {
                        *gi = 0.0;
                    }
                }
                return 1e20;
            }
            // Stagnation guard — see primary closure for rationale.
            if state.stagnation_stopped {
                if let Some(g) = grad {
                    for gi in g.iter_mut() {
                        *gi = 0.0;
                    }
                }
                state.n_evals += 1;
                n_evals_cl2.fetch_add(1, Ordering::Relaxed);
                return state.best_ofv;
            }
            let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            let params = unpack_params(&x, init_params);
            let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
            let (ehs, hms, ebe_stats2, kappas) = run_inner_loop_warm(
                model,
                population,
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&state.cached_etas),
                Some(&mu_k),
                options.min_obs_for_convergence_check as usize,
            );
            let nll = pop_nll(
                model,
                population,
                &params,
                &ehs,
                &hms,
                &kappas,
                options.interaction,
            );
            let raw_ofv = 2.0 * nll;

            let unconverged_frac2 = ebe_stats2.n_unconverged as f64 / (n_subj as f64).max(1.0);
            let ebe_guard2 = raw_ofv.is_finite()
                && unconverged_frac2 > options.max_unconverged_frac
                && options.max_unconverged_frac >= 0.0;
            {
                let mut acc = ebe_accum_cl2.lock().unwrap();
                if acc.max_unconverged < ebe_stats2.n_unconverged {
                    acc.max_unconverged = ebe_stats2.n_unconverged;
                }
                acc.total_fallback += ebe_stats2.n_fallback;
                if ebe_guard2 {
                    acc.n_convergence_warnings += 1;
                }
            }
            let ofv = if ebe_guard2 {
                1e20
            } else if raw_ofv.is_finite() {
                raw_ofv
            } else {
                1e20
            };

            let mut grad_norm_for_trace: Option<f64> = None;
            if let Some(g) = grad {
                if !raw_ofv.is_finite() {
                    for i in 0..g.len() {
                        let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                        g[i] = 100.0 * (xs[i] - center_s);
                    }
                    state.n_evals += 1;
                    n_evals_cl2.fetch_add(1, Ordering::Relaxed);
                    return ofv;
                }
                // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x); then scale for optimizer space.
                let grad_raw = population_gradient(
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
                );
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
                // SLSQP overshoot guard (issue #55) — this fallback
                // closure is unconditionally SLSQP.
                cap_slsqp_gradient(g, &lower_s, &upper_s);
                // See `best_seen` comment in the primary closure — gate on the
                // global accumulator, not `state.best_ofv` which is fresh here.
                {
                    let global_best = best_seen_cl2
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|(_, o)| *o)
                        .unwrap_or(f64::INFINITY);
                    if ofv < global_best {
                        *last_gradient_cl2.lock().unwrap() = Some(grad_raw.clone());
                    }
                }
            }

            state.cached_etas = ehs;
            state.cached_h_mats = hms;
            state.n_evals += 1;
            n_evals_cl2.fetch_add(1, Ordering::Relaxed);
            if ofv < state.best_ofv {
                state.best_ofv = ofv;
                if verbose {
                    eprintln!("Eval {:>4}: OFV = {:.6} (SLSQP)", state.n_evals, ofv);
                }
            }
            // See `best_seen` comment in the primary closure — gate on the
            // global accumulator, not `state.best_ofv` which is fresh here.
            {
                let mut bs = best_seen_cl2.lock().unwrap();
                if bs.as_ref().is_none_or(|(_, prev)| ofv < *prev) {
                    *bs = Some((xs.to_vec(), ofv));
                }
            }
            if detect_stagnation(state, n, options.stagnation_guard) && verbose {
                eprintln!(
                    "Eval {:>4}: SLSQP fallback stopping early — OFV has converged \
                     (no improvement above 1e-3 in last window). This is normal \
                     convergence behaviour, not an error: further evaluations are \
                     unlikely to find a better solution.",
                    state.n_evals,
                );
            }

            // Optimizer trace (SLSQP fallback, step_norm in scaled space)
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
                crate::estimation::trace::write_foce(
                    state.n_evals,
                    method_str,
                    ofv,
                    grad_norm_for_trace,
                    step_norm,
                    "slsqp",
                    Some(ebe_stats2.n_unconverged),
                    Some(ebe_stats2.n_fallback),
                );
            }
            state.prev_x = xs.to_vec();

            ofv
        };

        let mut opt2 = nlopt::Nlopt::new(
            nlopt::Algorithm::Slsqp,
            n,
            objective2,
            nlopt::Target::Minimize,
            state2,
        );
        opt2.set_lower_bounds(&lower_s).unwrap();
        opt2.set_upper_bounds(&upper_s).unwrap();
        opt2.set_maxeval(options.outer_maxiter as u32 * (n as u32 + 1))
            .unwrap();
        opt2.set_xtol_rel(1e-12).unwrap();
        opt2.set_ftol_rel(1e-12).unwrap();

        let result2 = opt2.optimize(&mut x0);
        converged = match &result2 {
            Ok((status, _)) => {
                if options.verbose {
                    eprintln!("NLopt SLSQP finished: {:?}", status);
                }
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
                    eprintln!("NLopt SLSQP stopped: {:?}", fail);
                }
                matches!(fail, nlopt::FailState::RoundoffLimited)
            }
        };
        drop(opt2);
    }

    // Restore the best-seen point (issue #59). NLopt returns the last
    // evaluated `x0`, not the best-seen one — when the stagnation guard
    // short-circuits, the last few evals return `best_ofv` with zero
    // gradient and the optimizer can drift off the true minimum before
    // termination. Replacing `x0` with the best-seen xs guarantees the
    // final inner loop and covariance step run at the actual minimum.
    if let Some((best_xs, best_ofv)) = best_seen.lock().unwrap().clone() {
        if best_xs.len() == n {
            x0.copy_from_slice(&best_xs);
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
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);

    // Final inner loop at converged parameters
    let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&final_mu_k),
        options.min_obs_for_convergence_check as usize,
    );

    let final_nll = pop_nll(
        model,
        population,
        &final_params,
        &final_ehs,
        &final_hms,
        &final_kappas,
        options.interaction,
    );
    let final_ofv = 2.0 * final_nll;

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // Covariance step (skip if user cancelled — it's expensive and the result
    // will be discarded by the top-level fit() anyway).
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            match compute_covariance(
                &x0,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            ) {
                CovarianceStepResult::Success(out) => {
                    warnings.extend(out.warnings);
                    Some(out.matrix)
                }
                CovarianceStepResult::Unusable(msg) => {
                    warnings.push(msg);
                    None
                }
                CovarianceStepResult::FailedNonPd {
                    reason,
                    fallback_proposal,
                } => {
                    warnings.push(reason);
                    sir_fallback_proposal = Some(fallback_proposal);
                    None
                }
            }
        } else {
            None
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
        2.0 * pop_nll(
            model,
            population,
            &params,
            eta_hats,
            h_matrices,
            kappas,
            options.interaction,
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
        );
        let ofv = 2.0
            * pop_nll(
                model,
                population,
                &params,
                &ehs,
                &hms,
                &kappas,
                options.interaction,
            );
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

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, f_val);
    }

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
        if g_norm < options.outer_gtol {
            if options.verbose {
                eprintln!("Converged at iteration {} (|g| = {:.2e})", iter, g_norm);
            }
            converged = true;
            break;
        }

        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let mut d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 || !dg.is_finite() {
            d = g.iter().map(|gi| -gi).collect();
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
            h_inv = DMatrix::identity(n, n);
            continue;
        }
        stall_count = 0;

        let xs_old = xs.clone();
        for i in 0..n {
            xs[i] = (xs[i] + alpha * d[i]).clamp(bounds_s.lower[i], bounds_s.upper[i]);
        }

        let (f_new, g_new, ehs, _) = fdfg_s(&xs, &cached_etas, &mut grad_eval_idx);
        cached_etas = ehs;

        bfgs_update(&mut h_inv, &xs, &xs_old, &g_new, &g, n);

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
            crate::estimation::trace::write_foce(
                iter,
                method_str,
                f_val,
                Some(g_norm),
                Some(step_norm),
                optimizer_str,
                None,
                None,
            );
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
    );
    let final_ofv = ofv_at_fixed(&x_final, &final_ehs, &final_hms, &final_kappas);

    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            match compute_covariance(
                &x_final,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            ) {
                CovarianceStepResult::Success(out) => {
                    warnings.extend(out.warnings);
                    Some(out.matrix)
                }
                CovarianceStepResult::Unusable(msg) => {
                    warnings.push(msg);
                    None
                }
                CovarianceStepResult::FailedNonPd {
                    reason,
                    fallback_proposal,
                } => {
                    warnings.push(reason);
                    sir_fallback_proposal = Some(fallback_proposal);
                    None
                }
            }
        } else {
            None
        };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
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
    let n = x.len();
    let n_subj = population.subjects.len();
    let fixed = packed_fixed_mask(init_params);
    let eps = 1e-4;

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
        );
        let raw = 2.0
            * pop_nll(
                model,
                population,
                &params,
                &ehs,
                &hms,
                &kappas,
                options.interaction,
            );
        let frac = ebe_stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
        if !raw.is_finite()
            || (options.max_unconverged_frac >= 0.0 && frac > options.max_unconverged_frac)
        {
            1e20
        } else {
            raw
        }
    };

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

/// Population gradient dispatcher. IOV models (`n_kappa > 0`) use the
/// EBE-reconverging FD gradient — their weakly-identified variance components
/// need it (issue #101 rec #2) — and everything else uses the cheap analytical
/// fixed-EBE gradient unless the `reconverge_gradient_interval` schedule opts
/// this evaluation into the reconverged path.
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
    // IOV models always reconverge the inner EBE solution inside the gradient.
    // For non-IOV models the default is the fixed-EBE analytical/AD gradient,
    // which is far cheaper but omits the response of (η̂, H) to the population
    // parameters — an omission that stalls SLSQP well above the derivative-free
    // optimum on ill-conditioned fits. The `reconverge_gradient_interval`
    // schedule (via `reconverge_this_eval`) opts a non-IOV fit into the
    // reconverged path (see focei-slsqp-fixed-ebe-gradient-bias).
    if model.n_kappa > 0 || reconverge {
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

/// Outcome of the FD covariance step. `matrix` is the n×n covariance with FIX
/// rows/cols zeroed; `warnings` carries non-fatal notes (regularisation applied,
/// off-diagonal FD stencil failures, etc.). Empty when everything was clean.
pub(crate) struct CovarianceOutput {
    pub matrix: DMatrix<f64>,
    pub warnings: Vec<String>,
}

/// Return type of [`compute_covariance`].
pub(crate) enum CovarianceStepResult {
    /// Covariance computed (possibly with non-fatal warnings).
    Success(CovarianceOutput),
    /// Structurally unusable. Carries a complete user-facing warning message
    /// (already ends with "SE estimates not available.").
    Unusable(String),
    /// FD Hessian symmetrised free-block has no positive eigenvalues — cannot
    /// be inverted. Carries the warning message and a ready-to-use fallback
    /// proposal covariance (full packed space, zeros for FIX params) built
    /// from `|eigenvalue|`-rectified Hessian, inflated 4×.
    FailedNonPd {
        reason: String,
        fallback_proposal: DMatrix<f64>,
    },
}

/// Human-readable label for the packed parameter at position `packed_idx`.
/// E.g. `"theta[CL]"`, `"omega[ETA_1, ETA_2]"`, `"sigma[1]"`.
///
/// Uses names from `template` directly (`theta_names`, `omega.eta_names`) rather
/// than from the `CompiledModel`, so the label is correct even when a test
/// constructs a `ModelParameters` whose dimensions differ from the test model.
fn packed_param_label(packed_idx: usize, template: &ModelParameters) -> String {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_omega = if template.omega.diagonal {
        n_eta
    } else {
        n_eta * (n_eta + 1) / 2
    };
    let n_sigma = template.sigma.values.len();
    let n_iov = template.omega_iov.as_ref().map_or(0, |m| {
        let d = m.dim();
        if m.diagonal {
            d
        } else {
            d * (d + 1) / 2
        }
    });

    if packed_idx < n_theta {
        let name = template
            .theta_names
            .get(packed_idx)
            .map(String::as_str)
            .unwrap_or("?");
        format!("theta[{}]", name)
    } else if packed_idx < n_theta + n_omega {
        let omega_idx = packed_idx - n_theta;
        let (row, col) = if template.omega.diagonal {
            (omega_idx, omega_idx)
        } else {
            let mut cnt = 0usize;
            let mut found = false;
            let mut res = (0, 0);
            'search: for c in 0..n_eta {
                for r in c..n_eta {
                    if cnt == omega_idx {
                        res = (r, c);
                        found = true;
                        break 'search;
                    }
                    cnt += 1;
                }
            }
            debug_assert!(
                found,
                "unreachable: omega_idx={omega_idx} >= n_omega={n_omega}"
            );
            res
        };
        let nr = template
            .omega
            .eta_names
            .get(row)
            .map(String::as_str)
            .unwrap_or("?");
        let nc = template
            .omega
            .eta_names
            .get(col)
            .map(String::as_str)
            .unwrap_or("?");
        format!("omega[{}, {}]", nr, nc)
    } else if packed_idx < n_theta + n_omega + n_sigma {
        let idx = packed_idx - n_theta - n_omega + 1;
        format!("sigma[{}]", idx)
    } else if packed_idx < n_theta + n_omega + n_sigma + n_iov {
        let idx = packed_idx - n_theta - n_omega - n_sigma + 1;
        format!("kappa[{}]", idx)
    } else {
        format!("packed[{}]", packed_idx)
    }
}

/// Format a single eigenvalue for display: `"0"`, fixed-4, or scientific-3.
///
/// The exact-zero branch handles rank-deficient inputs (e.g. a parameter block
/// that is entirely FIX) where `SymmetricEigen` returns eigenvalue `0.0` exactly.
/// Any non-zero value — even 1e-300 — uses fixed or scientific notation instead.
fn fmt_eig(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".to_string()
    } else if abs >= 1e-4 && abs < 1e5 {
        format!("{:.4}", v)
    } else {
        format!("{:.3e}", v)
    }
}

/// Eigenvalues of `sym` sorted descending. Returns `None` if any eigenvalue is non-finite.
fn extract_eigenvalues(sym: &DMatrix<f64>) -> Option<Vec<f64>> {
    let eig = SymmetricEigen::new(sym.clone());
    if eig.eigenvalues.iter().any(|l| !l.is_finite()) {
        return None;
    }
    let mut eigvals: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigvals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Some(eigvals)
}

/// Format a diagnostic warning for a non-positive-definite covariance Hessian.
fn format_non_pd_warning(eigvals: &[f64]) -> String {
    let fmt = eigvals
        .iter()
        .map(|&v| fmt_eig(v))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Covariance step: Hessian is not positive definite. \
         Eigenvalues: [{}]. SE estimates not available.",
        fmt
    )
}

/// Largest condition number permitted for the non-PD fallback proposal. The
/// eigenvalue magnitudes are floored at `λ_max_abs / COND` so a near-zero
/// curvature direction can't blow its proposal variance up without bound (see
/// [`build_non_pd_fallback_proposal`]).
const FALLBACK_PROPOSAL_MAX_COND: f64 = 1e8;

/// Build a SIR proposal covariance for the non-PD-Hessian fallback path.
///
/// This is the standard eigenvalue-modification heuristic: the symmetrised
/// free-block Hessian has at least one non-positive eigenvalue, so it cannot be
/// inverted into a covariance directly. We take each eigenvalue's *magnitude*
/// `|λ_i|` as the curvature in that direction, and use `inflation / |λ_i|` as the
/// corresponding proposal variance (`inflation`× wider than the inverted
/// absolute Hessian).
///
/// The magnitudes are floored **relative to the largest** at
/// `|λ|_max / FALLBACK_PROPOSAL_MAX_COND` rather than at a fixed absolute value.
/// A fixed floor (e.g. `1e-10`) is not scale-invariant: on a well-scaled Hessian
/// a near-zero eigenvalue would yield a proposal variance of `inflation / 1e-10`
/// ≈ 1e10, scattering every SIR draw far outside the parameter bounds so the
/// fallback degenerates to "all samples had invalid weights". The relative floor
/// caps the proposal's condition number at `FALLBACK_PROPOSAL_MAX_COND`, keeping
/// the draws in a usable range while still giving the weakly-identified
/// directions the widest proposal.
///
/// `inflation = 4.0` is the recommended default: heavier tails account for the
/// uncertainty introduced by the non-PD correction.
///
/// The result is embedded into the full packed-parameter covariance (zeros for
/// FIX parameters) and explicitly symmetrised, since the eigen-reconstruction
/// `V·diag·Vᵀ` can leave sub-ULP asymmetry that a downstream Cholesky rejects.
fn build_non_pd_fallback_proposal(
    hess_free_sym: &DMatrix<f64>,
    free_idx: &[usize],
    n_full: usize,
    inflation: f64,
) -> DMatrix<f64> {
    let eig = SymmetricEigen::new(hess_free_sym.clone());
    // Largest absolute eigenvalue anchors the relative floor. Guard the
    // all-zero block (max_abs == 0) with a tiny absolute fallback so the floor
    // stays positive and we never divide by zero.
    let max_abs = eig
        .eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let floor = (max_abs / FALLBACK_PROPOSAL_MAX_COND).max(1e-10);
    // Proposal covariance eigenvalues: inflation / max(|λ_i|, floor).
    let inv_eigs: DVector<f64> = eig.eigenvalues.map(|v| inflation / v.abs().max(floor));
    // Reconstruct: C_free = V * diag(inv_eigs) * V^T, then symmetrise to remove
    // any floating-point asymmetry from the matrix products.
    let cov_free_raw =
        &eig.eigenvectors * DMatrix::from_diagonal(&inv_eigs) * eig.eigenvectors.transpose();
    let cov_free = (&cov_free_raw + cov_free_raw.transpose()) * 0.5;
    // Embed free block into full n×n (FIX rows/cols stay zero).
    let mut cov = DMatrix::zeros(n_full, n_full);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            cov[(i, j)] = cov_free[(a, b)];
        }
    }
    cov
}

/// Choose a finite-difference step that keeps all free-parameter diagonal
/// stencils finite, starting from `initial_eps` and halving up to
/// `MAX_HALVINGS` times.
///
/// Returns `(chosen_eps, n_halvings)`. If every halving fails (all stencils
/// still non-finite at `initial_eps / 2^MAX_HALVINGS`), returns the final
/// eps anyway — the FD loop will detect and report the remaining failures.
///
/// The probe is on the scalar-OFV second-difference stencil
/// `(f₊ − 2·f₀ + f₋)/h²`, which is the exact stencil the IOV Hessian path uses.
/// The non-IOV path instead assembles the Hessian from central differences of
/// the analytical population gradient, so the OFV probe is a deliberate *proxy*
/// there: it shares the same underlying model evaluations (an OFV overflow at a
/// perturbation implies the gradient overflows too), is far cheaper than probing
/// the gradient, and the gradient FD loop carries its own `is_finite()` guard as
/// a backstop for the rare case the two disagree.
fn select_fd_step<F: Fn(&[f64]) -> f64>(
    x_hat: &[f64],
    free_idx: &[usize],
    initial_eps: f64,
    f0: f64,
    ofv: &F,
) -> (f64, usize) {
    const MAX_HALVINGS: usize = 8;
    let mut eps = initial_eps;
    let mut x = x_hat.to_vec();
    for halvings in 0..MAX_HALVINGS {
        let all_ok = free_idx.iter().all(|&i| {
            let hi = eps * (1.0 + x_hat[i].abs());
            x[i] = x_hat[i] + hi;
            let fp = ofv(&x);
            x[i] = x_hat[i] - hi;
            let fm = ofv(&x);
            x[i] = x_hat[i]; // always restore before returning
                             // Mirror the diagonal stencil the FD loop actually computes —
                             // (fp - 2·f0 + fm) / hi² — including the division. A finite
                             // numerator can still overflow once divided by a tiny hi², and the
                             // FD loop rejects on the quotient, so accepting the step here on the
                             // numerator alone would hand back an eps the loop then rejects.
            let h_ii = (fp - 2.0 * f0 + fm) / (hi * hi);
            h_ii.is_finite()
        });
        if all_ok {
            return (eps, halvings);
        }
        eps *= 0.5;
    }
    (eps, MAX_HALVINGS)
}

/// Combine the observed-information inverse `r_inv = R⁻¹` (already `2·H_ofv⁻¹`)
/// and the score cross-product `S` into the covariance estimator selected by
/// `method`:
///   - `Hessian`      → `R⁻¹`            (model-based; `S` ignored)
///   - `CrossProduct` → `S⁻¹`            (empirical information)
///   - `Sandwich`     → `R⁻¹ S R⁻¹`      (Huber–White, robust)
///
/// Returns `None` only for `CrossProduct`, when `S` is not strictly
/// positive-definite — singular *or* merely rank-deficient (fewer subjects than
/// free parameters, or collinear scores). Unlike the Hessian path, a
/// rank-deficient `S` is **rejected** rather than eigenvalue-floored: `S⁻¹` of a
/// regularised `S` would silently report finite-but-fictitious SEs in the
/// unidentified directions, so the cross-product estimator requires a full-rank
/// `S`. `Sandwich` never inverts `S`, so it stays defined even when `S` is
/// rank-deficient.
fn combine_covariance(
    method: CovarianceMethod,
    r_inv: DMatrix<f64>,
    s: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    match method {
        CovarianceMethod::Hessian => Some(r_inv),
        CovarianceMethod::Sandwich => Some(&r_inv * s * &r_inv),
        // Accept S⁻¹ only when S is full-rank (no eigenvalues clipped); a
        // rank-deficient or indefinite S yields `None`.
        CovarianceMethod::CrossProduct => match invert_psd_with_floor(s) {
            Some(inv) if inv.n_clipped == 0 => Some(inv.inverse),
            _ => None,
        },
    }
}

/// Assemble the per-subject score cross-product `S = Σᵢ gᵢgᵢᵀ` over the free
/// parameter block, where `gᵢ = ∂(−logLᵢ)/∂θ` is subject `i`'s contribution to
/// the population score (the same per-subject gradient the Gauss–Newton optimizer
/// uses for its BHHH step). `S` is NONMEM's `S` matrix; combined with the
/// observed-information `R` it yields the `S⁻¹` and `R⁻¹SR⁻¹` covariance forms.
///
/// The result is `n_free × n_free`, ordered to match `free_idx`. Caller embeds it
/// (or its inverse) back into the full packed space.
#[allow(clippy::too_many_arguments)]
fn assemble_score_cross_product(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
    free_idx: &[usize],
) -> DMatrix<f64> {
    let n_free = free_idx.len();
    let n_subj = population.subjects.len();

    // Per-subject scores in parallel (mirrors `build_gn_system`).
    //
    // The score cross-product evaluates the per-subject gradient directly at x̂.
    // Unlike the FD-built R-matrix — which reconverges η̂ at every perturbed point
    // and so captures the `log|H̃|` EBE-response `½·∂log|H̃|/∂η̂·dη̂/dθ` — the raw
    // analytic gradient holds η̂ fixed and drops it. Add it back here (the #274
    // `tᵢ` term, in −logL units; `point_grad` adds `2·tᵢ` to the −2logL gradient)
    // so the score matches how NONMEM differences the individual objective with
    // its conditional estimate responding to θ. This is what makes the FOCEI
    // S/RSR match NONMEM (warfarin RSR ≈ 1.8% with it, ≈ 5% without); the
    // alternative `∂a/∂θ` "a-response" was tested and is NOT what NONMEM's S
    // carries (it holds the model sensitivities `a` fixed at the linearization).
    // FOCE (`!interaction`) uses the Sheiner–Beal gradient, which has no `log|H̃|`
    // term — applying this Laplace-form `tᵢ` to FOCE was tested and over-corrects
    // (warfarin FOCE RSR 1.3% → 9.8% vs NONMEM), so the correction is FOCEI-only.
    let scores: Vec<Vec<f64>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            let kap_i = if i < kappas.len() {
                kappas[i].as_slice()
            } else {
                &[]
            };
            let (_, mut gi) = crate::estimation::gauss_newton::subject_nll_pop_grad(
                x_hat,
                template,
                model,
                population,
                i,
                &eta_hats[i],
                &h_matrices[i],
                kap_i,
                bounds,
                options,
            );
            if options.interaction {
                if let Some(ti) = crate::estimation::gauss_newton::subject_eta_response_correction(
                    None,
                    x_hat,
                    template,
                    model,
                    population,
                    i,
                    &eta_hats[i],
                    &h_matrices[i],
                    bounds,
                    options,
                ) {
                    for (g, t) in gi.iter_mut().zip(ti.iter()) {
                        *g += *t;
                    }
                }
            }
            gi
        })
        .collect();

    let mut s = DMatrix::zeros(n_free, n_free);
    for gi in &scores {
        let gi_free = DVector::from_iterator(n_free, free_idx.iter().map(|&k| gi[k]));
        s.ger(1.0, &gi_free, &gi_free, 1.0); // s += gi_free * gi_freeᵀ (full outer product)
    }
    s
}

/// Compute the parameter covariance matrix at convergence (the R-matrix:
/// inverse observed Fisher information).
///
/// The Hessian is built by finite differences that **reconverge the inner EBE
/// loop at every perturbed point** — matching how NONMEM's `$COVARIANCE` step
/// works. Holding the EBEs fixed (the previous behaviour) gives a Hessian with
/// the wrong curvature, indefinite even on well-conditioned surfaces like
/// warfarin, which forced eigenvalue clipping (#129) and inflated the SEs.
///
/// Two stencils:
/// - **non-IOV**: central FD of the analytical population gradient (issue #209),
///   `H[:,k] ≈ (g(x̂+hₖeₖ) − g(x̂−hₖeₖ)) / 2hₖ` — `2·n_free` gradient evaluations.
///   The θ part reuses H-matrix columns for mu-referenced parameters (issue #196).
/// - **IOV**: second differences of the reconverged OFV (the kappa block has no
///   fixed-EBE analytical gradient).
///
/// The returned covariance is `2·H⁻¹`: the objective is `−2·logL`, so its Hessian
/// is twice the observed information.
///
/// Returns [`CovarianceStepResult::Unusable`] when the FD Hessian is structurally
/// unusable (non-finite or zero-diagonal entries, or eigenvalues that diverge to
/// NaN/Inf so no proposal can be built). When the symmetrised free-block Hessian
/// is near-singular or has negative eigenvalues — a common FD noise artefact on
/// well-conditioned surfaces (see issue #129) — it is regularised by clipping
/// eigenvalues to a small positive floor before inversion, and the returned
/// `warning` records what was done. When the Hessian has finite eigenvalues but
/// no positive curvature at all (all eigenvalues ≤ 0), returns
/// [`CovarianceStepResult::FailedNonPd`], carrying the eigenvalue list formatted
/// as a warning together with an `|eigenvalue|`-rectified proposal covariance the
/// caller can hand to SIR when `covariance_fallback = sir`.
///
/// The estimator assembled from the Hessian `R` is selected by
/// [`FitOptions::covariance_method`] — `R⁻¹` (default), the score cross-product
/// `S⁻¹`, or the sandwich `R⁻¹SR⁻¹` (see [`assemble_score_cross_product`]).
pub(crate) fn compute_covariance(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> CovarianceStepResult {
    let n = x_hat.len();
    let initial_eps = options.fd_hessian_step;
    if initial_eps <= 0.0 || !initial_eps.is_finite() {
        return CovarianceStepResult::Unusable(format!(
            "Covariance step failed: fd_hessian_step must be positive and finite, got {}. \
             SE estimates not available.",
            initial_eps
        ));
    }
    let bounds = compute_bounds(template);

    // `h_matrices` (the H from the fit) is intentionally unused: the covariance
    // step reconverges the EBEs at every perturbed point and recomputes H there.
    // It stays in the signature for symmetry with `eta_hats` (the reconvergence
    // warm-start) and with the other optimizers' call sites.
    let _ = h_matrices;
    let n_subj_cov = population.subjects.len();

    // Re-solve the inner EBE loop at a packed point, warm-started from the
    // converged EBEs, serially over subjects. NONMEM reconverges the conditional
    // estimates at every perturbed point in its covariance step; holding η̂/H
    // fixed gives a Hessian with the wrong curvature — indefinite even on
    // warfarin, which previously forced eigenvalue clipping (#129) and inflated
    // the SEs.
    //
    // This single helper is the reconvergence used by all three covariance paths
    // — the base-OFV evaluation, the non-IOV gradient-FD `point_grad`, and the
    // IOV scalar-FD `serial_ofv` — so they cannot drift apart (#298). It is
    // serial (not the parallel `run_inner_loop_warm`) because the covariance step
    // parallelises over perturbed POINTS, not subjects; nested parallelism is
    // what #256 removed. `find_ebe` is deterministic per subject, so the
    // per-subject EBEs are bit-identical to the parallel loop.
    let reconverge_point = |xv: &[f64]| -> (
        ModelParameters,
        Vec<DVector<f64>>,
        Vec<DMatrix<f64>>,
        Vec<Vec<DVector<f64>>>,
    ) {
        let params = unpack_params(xv, template);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let mut ehs = Vec::with_capacity(n_subj_cov);
        let mut hms = Vec::with_capacity(n_subj_cov);
        let mut kaps = Vec::with_capacity(n_subj_cov);
        for i in 0..n_subj_cov {
            let ebe = find_ebe(
                model,
                &population.subjects[i],
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(eta_hats[i].as_slice()),
                Some(&mu_k),
            );
            ehs.push(ebe.eta);
            hms.push(ebe.h_matrix);
            kaps.push(ebe.kappas);
        }
        (params, ehs, hms, kaps)
    };

    // Covariance OFV = −2·logL at a reconverged point. For FOCEI the per-subject
    // marginal already carries ηᵀΩ⁻¹η + log|Ω|; for FOCE we add that prior here.
    let ofv = |xv: &[f64]| -> f64 {
        let (params, ehs, hms, kaps) = reconverge_point(xv);
        let foce_nll = pop_nll(
            model,
            population,
            &params,
            &ehs,
            &hms,
            &kaps,
            options.interaction,
        );
        // Covariance OFV = −2·logL = 2·pop_nll for both FOCE and FOCEI.
        //
        // FOCE uses the Sheiner–Beal linearised marginal `(y−f₀)ᵀR̃⁻¹(y−f₀) +
        // log|R̃|` with R̃ = HΩHᵀ + R. By Woodbury that marginal *already* carries
        // the Ω penalty (it equals the conditional form including η̂ᵀΩ⁻¹η̂ +
        // log|Ω|), so its Ω-curvature is complete. An earlier version added the
        // η̂ᵀΩ⁻¹η̂ + log|Ω| prior here for the FOCE branch, which double-counted Ω
        // and flattened the Ω-block curvature — the source of the ~31%-low FOCE
        // omega SEs (issue #243). FOCEI's Almquist–Laplace marginal likewise
        // carries the prior internally. So neither method needs an add-back.
        2.0 * foce_nll
    };

    let base_ofv = ofv(x_hat);
    if !base_ofv.is_finite() {
        // Diagnose: check Omega conditioning to distinguish Omega collapse from
        // a model-evaluation overflow/underflow.
        let params_at = unpack_params(x_hat, template);
        let reason = match extract_eigenvalues(&params_at.omega.matrix) {
            Some(ref ev) if ev.last().copied().unwrap_or(1.0) <= 1e-8 => {
                let min_eig = ev.last().copied().unwrap_or(f64::NAN);
                // Distinguish truly negative eigenvalues from tiny-positive (near-singular).
                let descriptor = if min_eig < 0.0 {
                    "not positive definite"
                } else {
                    "near-singular"
                };
                format!(
                    "Covariance step failed: Omega matrix is {} at convergence \
                     (min eigenvalue = {}; eigenvalues: [{}]). \
                     SE estimates not available.",
                    descriptor,
                    fmt_eig(min_eig),
                    ev.iter()
                        .map(|&v| fmt_eig(v))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            _ => "Covariance step failed: base OFV is non-finite at convergence \
                  (likely numerical overflow or underflow in model evaluation). \
                  SE estimates not available."
                .to_string(),
        };
        if options.verbose {
            eprintln!("  {}", reason);
        }
        return CovarianceStepResult::Unusable(reason);
    }

    // FIX parameters contribute no information — skip their FD stencils and,
    // after inverting the Hessian of the free block, leave their covariance
    // rows/cols at zero (→ SE = 0 downstream).
    let fixed_mask = packed_fixed_mask(template);
    // Structural-zero Ω off-diagonals (the cross-block elements of a mixed
    // block+diagonal Ω, where `free_mask[(i,j)] == false`) are not estimated
    // parameters — the analytical population gradient zeroes them, so their
    // Hessian diagonal is flat. Exclude them from the free set exactly like FIX
    // parameters; otherwise the ill-conditioning guard below rejects the entire
    // covariance step. (Before #243 the omega-prior add-back iterated all
    // lower-triangle entries and gave these a spurious non-zero curvature, which
    // masked the issue for the FOCE path; FOCEI never had that mask.)
    let structural_zero = omega_structural_zero_mask(template);
    let free_idx: Vec<usize> = (0..n)
        .filter(|&i| !fixed_mask[i] && !structural_zero[i])
        .collect();

    let f0 = base_ofv;

    // Adaptively select the FD step: halve up to 8× until all free-parameter
    // diagonal stencils are finite. Most models use the initial step; halving
    // only kicks in when the OFV overflows at the default perturbation size.
    let (eps, n_halvings) = select_fd_step(x_hat, &free_idx, initial_eps, f0, &ofv);
    if options.verbose && n_halvings > 0 {
        eprintln!(
            "  [covariance] Adaptive FD step: reduced {:.3e} → {:.3e} ({} halving{})",
            initial_eps,
            eps,
            n_halvings,
            if n_halvings == 1 { "" } else { "s" }
        );
    }

    let mut hess = DMatrix::zeros(n, n);
    let is_iov = kappas.iter().any(|k| !k.is_empty());
    // Route non-interaction FOCE with f-dependent error (proportional/combined)
    // through the OFV second-difference stencil (the IOV path), which builds
    // the true Hessian of the actual marginal. The analytical SB gradient is an
    // envelope approximation with no EBE-response Δ (that correction exists only
    // for FOCEI, #274), so its central-FD Hessian comes out indefinite on the
    // f-dependent FOCE surface. Additive FOCE keeps the cheap analytical path
    // (the Δ vanishes for f-independent variance, and it already matches NONMEM).
    // Route through the OFV second-difference Hessian when: (a) f-dependent FOCE
    // (the analytical SB gradient comes out indefinite there), or (b) the user
    // opts in via `covariance_ofv_hessian`. The latter trades speed for an R
    // that recomputes `a = ∂f/∂η` at every perturbed point, capturing the
    // `∂a/∂θ` curvature the analytical stencil drops — which removes the
    // weakly-identified-θ SE bias (e.g. warfarin TVKA ~9% high vs a Richardson
    // FD-of-OFV ground truth).
    let force_ofv_hessian = (!options.interaction && model.error_spec.has_f_dependent_variance())
        || options.covariance_ofv_hessian;
    let use_analytical = !is_iov && !force_ofv_hessian;

    // Track FD failures at source so diagnostics name the right cause (a NaN/Inf
    // stencil result is not a genuine zero curvature). HashSet for O(1) ops.
    let mut fd_diag_nan: HashSet<usize> = HashSet::new();
    let mut fd_offdiag_nan: HashSet<usize> = HashSet::new();

    if use_analytical {
        // Issue #209 + #256 + #274: central FD of the analytical population
        // gradient, as one flat `par_iter` over the 2·n_free perturbed points.
        //   H[:,k] ≈ (g(x̂ + hₖ·eₖ) − g(x̂ − hₖ·eₖ)) / 2hₖ
        // `point_grad` reconverges the EBEs serially at each perturbed point, so
        // the curvature includes the EBE response (and the determinant curvature).
        //
        // #256: the work-list is point-level, not the per-subject `par_iter` the
        // gradient used to fan out into. Each point runs its subjects serially, so
        // there is no nested parallelism, and the parallel width (2·n_free)
        // saturates the pool even when n_subj < n_cores — removing the fork/join
        // overhead of firing 4·n_free rayon barriers in series (~9–11× faster).
        //
        // #274: for FOCEI the per-point gradient adds the dropped `log|H̃|`
        // EBE-response term `2·Σᵢ tᵢ` (`subject_eta_response_correction`). The
        // fixed-η̂ analytic gradient invokes the envelope theorem, which zeros only
        // the inner objective — not `log|H̃|` — so without this term the non-IOV
        // FD Hessian omits the determinant EBE-response curvature `Δ` that the IOV
        // scalar-OFV stencil captures. Adding it makes the two stencils consistent
        // and recovers ∇²(−2logL). Mu-ref θ block only; vanishes for additive error.
        // Count subject-points where the FOCEI Δ correction was skipped because
        // the Laplace gradient fell back to FD (non-PD H̃) — those contributions
        // keep the pre-#274 fixed-η̂ curvature, so a non-zero count is surfaced as
        // a diagnostic (#298).
        let delta_skips = std::sync::atomic::AtomicUsize::new(0);
        let point_grad = |xv: &[f64]| -> Vec<f64> {
            let (_, ehs, hms, _) = reconverge_point(xv);
            let np = xv.len();
            // Gradient of `2·pop_nll` (no omega-prior add-back; both the SB and
            // Laplace marginals already carry Ω — issue #243/#249).
            //
            // Build the per-subject gradients serially (subjects are serial inside
            // each point — the #256 flatten parallelises over points, not subjects)
            // and reduce through `assemble_population_gradient`, the same reduction
            // `ad_population_gradient` uses — so the summation order matches and the
            // FOCE covariance stays bit-identical to the pre-#256 serial stencil.
            // The Δ correction below is kept as a separate loop (NOT fused): summing
            // `2·tᵢ` after `2·Σ gᵢ` preserves that reduction order exactly.
            //
            // `subject_nll_pop_grad_with_cache` also hands back the per-subject
            // Laplace intermediates (when this subject took the FOCEI analytical
            // path); the Δ loop below reuses them so it does not recompute the
            // predictions or re-factorise H̃.
            let mut grads: Vec<Vec<f64>> = Vec::with_capacity(n_subj_cov);
            let mut caches: Vec<Option<crate::estimation::gauss_newton::LaplaceGradCache>> =
                Vec::with_capacity(n_subj_cov);
            for i in 0..n_subj_cov {
                let (_, gi, ci) = crate::estimation::gauss_newton::subject_nll_pop_grad_with_cache(
                    xv,
                    template,
                    model,
                    population,
                    i,
                    &ehs[i],
                    &hms[i],
                    &[],
                    &bounds,
                    options,
                );
                grads.push(gi);
                caches.push(ci);
            }
            let mut g = assemble_population_gradient(&grads, np);
            // #274 Δ correction (FOCEI only); summed in subject order to match.
            if options.interaction {
                for i in 0..n_subj_cov {
                    match crate::estimation::gauss_newton::subject_eta_response_correction(
                        caches[i].as_ref(),
                        xv,
                        template,
                        model,
                        population,
                        i,
                        &ehs[i],
                        &hms[i],
                        &bounds,
                        options,
                    ) {
                        Some(ti) => {
                            for (gk, tk) in g.iter_mut().zip(ti.iter()) {
                                *gk += 2.0 * *tk;
                            }
                        }
                        None => {
                            delta_skips.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            g
        };

        // One (k, hₖ) spec per free parameter; the two perturbed points (±hₖ) are
        // kept adjacent so `chunks_exact(2)` re-pairs (g₊, g₋) structurally — the
        // pairing is no longer a positional `2·pair` index that a reordering of
        // the point build could silently desync (#298).
        let specs: Vec<(usize, f64)> = free_idx
            .iter()
            .map(|&k| (k, eps * (1.0 + x_hat[k].abs())))
            .collect();
        let pts: Vec<Vec<f64>> = specs
            .iter()
            .flat_map(|&(k, hk)| {
                let mut x_p = x_hat.to_vec();
                x_p[k] += hk;
                let mut x_m = x_hat.to_vec();
                x_m[k] -= hk;
                [x_p, x_m]
            })
            .collect();
        let point_grads: Vec<Vec<f64>> = pts.par_iter().map(|xv| point_grad(xv)).collect();
        for (&(k, hk), pair) in specs.iter().zip(point_grads.chunks_exact(2)) {
            let (g_p, g_m) = (&pair[0], &pair[1]);
            for &j in &free_idx {
                let h_jk = (g_p[j] - g_m[j]) / (2.0 * hk);
                if h_jk.is_finite() {
                    hess[(j, k)] = h_jk;
                } else if j == k {
                    fd_diag_nan.insert(k);
                } else {
                    fd_offdiag_nan.insert(k);
                    fd_offdiag_nan.insert(j);
                }
            }
        }
        let skipped = delta_skips.load(std::sync::atomic::Ordering::Relaxed);
        if options.interaction && skipped > 0 && options.verbose {
            eprintln!(
                "  [covariance] log|H̃| EBE-response correction skipped at {} subject-point(s) \
                 where the Laplace gradient fell back to FD (non-PD H̃); those contributions \
                 retain the pre-#274 fixed-η̂ curvature.",
                skipped
            );
        }
        // Symmetrise: each column is differenced independently, so H[j,k] and
        // H[k,j] can differ slightly; average before inversion.
        for &i in &free_idx {
            for &j in &free_idx {
                if j > i {
                    let avg = (hess[(i, j)] + hess[(j, i)]) * 0.5;
                    hess[(i, j)] = avg;
                    hess[(j, i)] = avg;
                }
            }
        }
    } else {
        // Reconverged-OFV second-difference Hessian (3-point diagonal, 4-point
        // off-diagonal), reconverging the EBEs at each perturbed point. Taken
        // when the analytical fixed-EBE gradient does not cover the true marginal
        // curvature: (a) IOV — no analytical gradient covers the kappa block; or
        // (b) `force_ofv_hessian` — non-IOV FOCE with f-dependent error, whose SB
        // gradient lacks the EBE-response Δ and yields an indefinite analytical
        // Hessian. `pop_nll` dispatches on the kappa count, so this stencil is
        // correct for both the IOV (joint η, κ) and the non-IOV (η-only) cases.
        //
        // #256: flattened to one `par_iter` over all ~2·n_free² perturbed OFV
        // points (subjects iterated serially inside `serial_ofv`) instead of the
        // old serial loop that fired a per-subject `par_iter` at every point —
        // removing the fork/join overhead of firing a rayon barrier per point.
        // Bit-identical to the serial stencil: each point's OFV is the same
        // `2·pop_nll` at the same per-subject `find_ebe`, and the difference
        // formulas/assembly are unchanged; only the scheduling differs.
        let f0 = base_ofv;
        let serial_ofv = |xv: &[f64]| -> f64 {
            let (params, ehs, hms, kaps) = reconverge_point(xv);
            2.0 * pop_nll(
                model,
                population,
                &params,
                &ehs,
                &hms,
                &kaps,
                options.interaction,
            )
        };

        let nf = free_idx.len();
        let hsteps: Vec<f64> = free_idx
            .iter()
            .map(|&i| eps * (1.0 + x_hat[i].abs()))
            .collect();
        // Flat list of perturbation SPECS (not materialised x-vectors): 2 per
        // diagonal (±hᵢ), then 4 per (a<b) off-diagonal pair. Each par_iter task
        // clones `x_hat` once and applies its spec, so only ~n_threads vectors are
        // live at a time instead of all ~2·nf² perturbed points held resident for
        // the whole reduction (the pre-#298 O(nf²·np) footprint) (#298).
        #[derive(Clone, Copy)]
        enum Pert {
            Single {
                i: usize,
                di: f64,
            },
            Pair {
                i: usize,
                di: f64,
                j: usize,
                dj: f64,
            },
        }
        let mut specs: Vec<Pert> = Vec::with_capacity(2 * nf + 2 * nf * nf);
        for a in 0..nf {
            let (i, hi) = (free_idx[a], hsteps[a]);
            specs.push(Pert::Single { i, di: hi });
            specs.push(Pert::Single { i, di: -hi });
        }
        let n_diag = specs.len();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for a in 0..nf {
            for b in (a + 1)..nf {
                let (i, j) = (free_idx[a], free_idx[b]);
                let (hi, hj) = (hsteps[a], hsteps[b]);
                for (si, sj) in [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
                    specs.push(Pert::Pair {
                        i,
                        di: si * hi,
                        j,
                        dj: sj * hj,
                    });
                }
                pairs.push((a, b));
            }
        }
        let vals: Vec<f64> = specs
            .par_iter()
            .map(|p| {
                let mut xv = x_hat.to_vec();
                match *p {
                    Pert::Single { i, di } => xv[i] += di,
                    Pert::Pair { i, di, j, dj } => {
                        xv[i] += di;
                        xv[j] += dj;
                    }
                }
                serial_ofv(&xv)
            })
            .collect();
        // Diagonal: (f(x+h) − 2f(x) + f(x−h)) / h².
        for a in 0..nf {
            let i = free_idx[a];
            let hi = hsteps[a];
            let h_ii = (vals[2 * a] - 2.0 * f0 + vals[2 * a + 1]) / (hi * hi);
            if h_ii.is_finite() {
                hess[(i, i)] = h_ii;
            } else {
                fd_diag_nan.insert(i);
            }
        }
        // Off-diagonal: (f++ − f+− − f−+ + f−−) / (4 hᵢ hⱼ).
        let mut off = n_diag;
        for &(a, b) in &pairs {
            let (i, j) = (free_idx[a], free_idx[b]);
            let (hi, hj) = (hsteps[a], hsteps[b]);
            let (fpp, fpm, fmm, fmp) = (vals[off], vals[off + 1], vals[off + 2], vals[off + 3]);
            off += 4;
            let h_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            if h_ij.is_finite() {
                hess[(i, j)] = h_ij;
                hess[(j, i)] = h_ij;
            } else {
                fd_offdiag_nan.insert(i);
                fd_offdiag_nan.insert(j);
            }
        }
    }

    // Diagnose fatal Hessian problems. Use the FD-failure trackers for accurate
    // cause labels — post-hoc checks on `hess` would always read 0 (finite) because
    // non-finite FD results are never stored (only the zero initialisation remains).
    let mut problem_params: Vec<String> = Vec::new();
    for &i in &free_idx {
        let diag = hess[(i, i)];
        if fd_diag_nan.contains(&i) {
            // Diagonal FD stencil overflowed; zero stored value does not mean flat
            // objective. Adjust fd_hessian_step or check for model overflow.
            problem_params.push(format!(
                "{} (FD stencil non-finite; model may overflow at perturbation — \
                 try tuning fd_hessian_step)",
                packed_param_label(i, template)
            ));
        } else if diag.abs() < 1e-30 {
            // Genuine flat objective: the FD stencil succeeded but returned ~0 curvature.
            problem_params.push(format!(
                "{} (zero diagonal — flat objective)",
                packed_param_label(i, template)
            ));
        }
    }

    if !problem_params.is_empty() {
        let reason = format!(
            "Covariance step failed: Hessian has ill-conditioned entries for the following \
             parameter(s) — {}. SE estimates not available.",
            problem_params.join("; ")
        );
        if options.verbose {
            eprintln!("  {}", reason);
        }
        return CovarianceStepResult::Unusable(reason);
    }

    // Build the reduced Hessian over free indices, invert, then embed back
    // into the full n×n covariance matrix (FIX rows/cols stay zero).
    let n_free = free_idx.len();
    if n_free == 0 {
        // Nothing to estimate — return an all-zero covariance so downstream
        // SE extraction reports zeros (all params FIX).
        return CovarianceStepResult::Success(CovarianceOutput {
            matrix: DMatrix::zeros(n, n),
            warnings: vec![],
        });
    }
    let mut hess_free = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            hess_free[(a, b)] = hess[(i, j)];
        }
    }
    let hess_free_sym = (&hess_free + hess_free.transpose()) * 0.5;

    let inv = match invert_psd_with_floor(&hess_free_sym) {
        Some(inv) => inv,
        None => {
            // `invert_psd_with_floor` returns None in two distinct cases, and we
            // must not conflate them: (a) every eigenvalue is finite but the
            // spectrum has no positive curvature (a genuine non-PD Hessian — a
            // SIR fallback is meaningful here), or (b) the eigendecomposition
            // itself diverged and produced a non-finite eigenvalue (the Hessian
            // contains NaN/Inf — no usable proposal can be built).
            //
            // `extract_eigenvalues` returns None for exactly case (b). Building a
            // fallback proposal there would re-run the same divergent
            // decomposition and embed NaN eigenvectors into the proposal
            // covariance, which SIR would then silently turn into NaN samples.
            // So only build the proposal when the eigenvalues are finite.
            match extract_eigenvalues(&hess_free_sym) {
                Some(eigvals) => {
                    let fallback_proposal =
                        build_non_pd_fallback_proposal(&hess_free_sym, &free_idx, n, 4.0);
                    return CovarianceStepResult::FailedNonPd {
                        reason: format_non_pd_warning(&eigvals),
                        fallback_proposal,
                    };
                }
                None => {
                    return CovarianceStepResult::Unusable(
                        "Covariance step failed: could not compute eigenvalues of the \
                         FD Hessian (Hessian may contain NaN or Inf). \
                         SE estimates not available."
                            .to_string(),
                    );
                }
            }
        }
    };
    // The FD Hessian is of the OFV = −2·logL. The asymptotic covariance is the
    // inverse observed Fisher information R = Hessian of −logL = ½·H_ofv, so
    // R⁻¹ = 2·H_ofv⁻¹. Without this factor every SE is 1/√2 too small.
    let r_inv = inv.inverse * 2.0;

    // Select the covariance estimator (NONMEM `$COV MATRIX=`). `R⁻¹` is the
    // model-based default; `S⁻¹` and `R⁻¹SR⁻¹` additionally need the per-subject
    // score cross-product `S = Σᵢ gᵢgᵢᵀ`. `S` is on the −logL scale
    // (`gᵢ = ∂(−logLᵢ)/∂θ`, no factor of 2), matching `R = ½·H_ofv`.
    // Anchored against NONMEM `$COV MATRIX=S`/`RSR` for both FOCEI (#266) and
    // FOCE (no-INTER) (#250): all SEs within ~10% of NONMEM.
    let cov_free = if options.covariance_method == CovarianceMethod::Hessian {
        r_inv
    } else {
        let s_free = assemble_score_cross_product(
            x_hat, template, model, population, eta_hats, h_matrices, kappas, &bounds, options,
            &free_idx,
        );
        match combine_covariance(options.covariance_method, r_inv, &s_free) {
            Some(c) => c,
            None => {
                return CovarianceStepResult::Unusable(
                    "Covariance step failed: the score cross-product matrix S is singular or \
                     rank-deficient (covariance_method = s); typically fewer subjects than free \
                     parameters, or collinear per-subject scores. Use covariance_method = r or \
                     rsr. SE estimates not available."
                        .to_string(),
                );
            }
        }
    };

    let mut cov = DMatrix::zeros(n, n);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            cov[(i, j)] = cov_free[(a, b)];
        }
    }

    let mut cov_warnings: Vec<String> = Vec::new();

    // The Hessian eigenvalue-floor warning is about `R`. It is relevant only when
    // the returned covariance actually uses `R⁻¹` (Hessian and sandwich); the
    // cross-product path returns `S⁻¹` (with a full-rank `S` guaranteed above), so
    // a clipped `R` there would be a misleading note about a matrix it didn't use.
    if inv.n_clipped > 0 && options.covariance_method != CovarianceMethod::CrossProduct {
        let pct = inv.n_clipped * 100 / n_free.max(1);
        // Informal thresholds: ≤33 % clipped → minor concern; 34–50 % → caution; >50 % → unreliable.
        // Note: integer truncation means the boundary moves in steps of 1/n_free; for small
        // n_free adjacent clipped counts can jump directly from "minor" to "severe".
        let (severity, interp) = match pct {
            0..=33 => ("minor", "Standard errors are likely reliable."),
            34..=50 => (
                "moderate",
                "Standard errors should be interpreted with caution; \
                 consider SIR-based confidence intervals.",
            ),
            _ => (
                "severe",
                "Standard errors are likely unreliable; \
                 SIR-based confidence intervals are recommended.",
            ),
        };
        let msg = format!(
            "Covariance step regularized: eigenvalue floor applied to FD Hessian \
             ({} of {} free-block eigenvalues clipped; min eig = {:.3e}, floor = {:.3e}; \
             severity: {}). {}",
            inv.n_clipped, n_free, inv.min_eigenvalue, inv.floor, severity, interp
        );
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    } else if options.verbose {
        eprintln!("  Covariance step successful");
    }

    // Soft warning: cross-partial FD stencils that returned NaN/Inf were stored as 0,
    // so off-diagonal correlation is missing for these parameters. SEs for the named
    // parameters may be over-optimistic (correlation with other parameters is absent).
    if !fd_offdiag_nan.is_empty() {
        // Sort by packed index so the warning message is deterministic regardless
        // of HashSet iteration order.
        let mut sorted_idx: Vec<usize> = fd_offdiag_nan.iter().cloned().collect();
        sorted_idx.sort_unstable();
        let names: Vec<String> = sorted_idx
            .iter()
            .map(|&i| packed_param_label(i, template))
            .collect();
        let msg = format!(
            "Covariance step: off-diagonal FD stencil(s) non-finite for {}. \
             Cross-partial correlation set to 0; SE for these parameter(s) \
             may be over-optimistic. Try tuning fd_hessian_step.",
            names.join(", ")
        );
        if options.verbose {
            eprintln!("  {}", msg);
        }
        cov_warnings.push(msg);
    }

    CovarianceStepResult::Success(CovarianceOutput {
        matrix: cov,
        warnings: cov_warnings,
    })
}

/// Result of [`invert_psd_with_floor`].
pub(crate) struct RegularizedInverse {
    pub inverse: DMatrix<f64>,
    /// Smallest eigenvalue of the input matrix (before clipping). `f64::INFINITY`
    /// for 0×0 matrices.
    pub min_eigenvalue: f64,
    /// Floor used for clipping. Same shape rules as `min_eigenvalue`.
    pub floor: f64,
    /// How many eigenvalues fell below the floor and were clipped.
    pub n_clipped: usize,
}

/// Invert a symmetric matrix by clipping eigenvalues to a small positive floor.
///
/// This is the regularised replacement for `try_inverse() + neg-diag check` on
/// the FD Hessian. The previous code rejected the entire covariance step on a
/// single negative diagonal of the raw inverse — which on a well-conditioned
/// surface (FOCE/FOCEI converges cleanly to the same OFV across optimizers) is
/// almost always an FD-noise artefact rather than real ill-conditioning. The
/// floor leaves PD inputs untouched (`n_clipped == 0`, exact inverse) and
/// recovers a PD inverse on near-singular or marginally-indefinite inputs.
///
/// Floor: `max(max_eig * 1e-10, 1e-12)`. Anchoring to `max_eig` keeps the
/// regularisation scale-equivariant; the absolute floor handles the edge case
/// where the whole spectrum is tiny.
///
/// Returns `None` only when the eigendecomposition fails or every eigenvalue
/// is non-finite or non-positive — i.e. the Hessian carries no usable
/// curvature information at all, in which case regularisation cannot help.
pub(crate) fn invert_psd_with_floor(sym: &DMatrix<f64>) -> Option<RegularizedInverse> {
    let n = sym.nrows();
    debug_assert_eq!(
        n,
        sym.ncols(),
        "invert_psd_with_floor requires square input"
    );
    if n == 0 {
        return Some(RegularizedInverse {
            inverse: DMatrix::zeros(0, 0),
            min_eigenvalue: f64::INFINITY,
            floor: f64::INFINITY,
            n_clipped: 0,
        });
    }

    // Symmetric eigendecomposition: H = Q Λ Qᵀ ⇒ H⁻¹ = Q Λ⁻¹ Qᵀ. Inverting via
    // the eigendecomposition lets us clip non-positive Λ entries before
    // forming Λ⁻¹, which is what `try_inverse` cannot do.
    let eig = SymmetricEigen::new(sym.clone());
    let q = &eig.eigenvectors;
    let lambdas = &eig.eigenvalues;

    let mut max_eig = f64::NEG_INFINITY;
    for i in 0..n {
        let l = lambdas[i];
        if !l.is_finite() {
            return None;
        }
        if l > max_eig {
            max_eig = l;
        }
    }
    if !max_eig.is_finite() || max_eig <= 0.0 {
        // Spectrum is entirely ≤ 0 — no positive curvature anywhere; this is
        // a genuinely degenerate Hessian, not FD noise. Flag as failure so the
        // caller can report "Covariance step failed" rather than silently
        // returning a meaningless matrix.
        return None;
    }

    let floor = (max_eig * 1e-10).max(1e-12);
    let mut min_eig = f64::INFINITY;
    let mut n_clipped = 0;
    let mut inv_lambdas = DVector::zeros(n);
    for i in 0..n {
        let l = lambdas[i];
        if l < min_eig {
            min_eig = l;
        }
        let l_clipped = if l < floor {
            n_clipped += 1;
            floor
        } else {
            l
        };
        inv_lambdas[i] = 1.0 / l_clipped;
    }

    // cov = Q diag(1/λ) Qᵀ — scale columns of Q by 1/λ, then multiply by Qᵀ.
    let mut q_scaled = q.clone();
    for j in 0..n {
        let s = inv_lambdas[j];
        for i in 0..n {
            q_scaled[(i, j)] *= s;
        }
    }
    let mut inverse = &q_scaled * q.transpose();
    // Eigendecomposition + reconstruction is symmetric in exact arithmetic but
    // not in floating point; symmetrise so downstream consumers (e.g. SIR
    // proposal Cholesky) see a numerically symmetric matrix.
    let inv_t = inverse.transpose();
    inverse = (&inverse + &inv_t) * 0.5;

    Some(RegularizedInverse {
        inverse,
        min_eigenvalue: min_eig,
        floor,
        n_clipped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parameterization::{compute_bounds, pack_params};

    /// `resolve_scaling` maps `Auto` to `Rescale2` for the gradient-based
    /// optimizers that benefit (incl. `Slsqp` — the #335 cold-start fix) and to
    /// `None` for the derivative-free `Bobyqa` default (and `Mma`/`TrustRegion`);
    /// explicit non-`Auto` values pass through unchanged. Guards the #341/#335
    /// default-scaling routing.
    #[test]
    fn resolve_scaling_routes_auto_by_optimizer() {
        use crate::types::ParameterScaling::{Abs, Auto, None as PsNone, Rescale2};
        for opt in [
            Optimizer::Bfgs,
            Optimizer::Lbfgs,
            Optimizer::NloptLbfgs,
            Optimizer::Slsqp,
        ] {
            assert_eq!(
                resolve_scaling(Auto, opt),
                Rescale2,
                "{opt:?} should be Rescale2 under Auto"
            );
        }
        for opt in [Optimizer::Bobyqa, Optimizer::Mma, Optimizer::TrustRegion] {
            assert_eq!(
                resolve_scaling(Auto, opt),
                PsNone,
                "{opt:?} should be unscaled under Auto"
            );
        }
        // Explicit values pass through regardless of optimizer.
        assert_eq!(resolve_scaling(Rescale2, Optimizer::Bobyqa), Rescale2);
        assert_eq!(resolve_scaling(PsNone, Optimizer::Bfgs), PsNone);
        assert_eq!(resolve_scaling(Abs, Optimizer::Slsqp), Abs);
    }

    // ── invert_psd_with_floor: regularised PD inversion ──────────────────────

    /// A genuinely PD matrix is inverted unchanged (n_clipped == 0) and the
    /// result satisfies `H · H⁻¹ ≈ I` to high precision.
    #[test]
    fn test_invert_psd_with_floor_pd_matrix_unchanged() {
        // 3×3 SPD: build as Lᵀ·L with L lower-triangular so eigenvalues are O(1).
        let l = DMatrix::from_row_slice(3, 3, &[2.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.3, 0.2, 1.2]);
        let h = l.transpose() * &l;
        let r = invert_psd_with_floor(&h).expect("PD input inverts");
        assert_eq!(r.n_clipped, 0, "PD input should not trigger clipping");
        assert!(r.min_eigenvalue > 0.0);

        let prod = &h * &r.inverse;
        let eye = DMatrix::<f64>::identity(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (prod[(i, j)] - eye[(i, j)]).abs() < 1e-9,
                    "H·H⁻¹ deviates at ({i},{j}): {:.3e}",
                    (prod[(i, j)] - eye[(i, j)]).abs()
                );
            }
        }
    }

    /// Helper: assert two matrices agree element-wise.
    #[cfg(test)]
    fn assert_mat_close(a: &DMatrix<f64>, b: &DMatrix<f64>, tol: f64, ctx: &str) {
        assert_eq!(a.shape(), b.shape(), "{ctx}: shape mismatch");
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[(i, j)] - b[(i, j)]).abs() < tol,
                    "{ctx}: ({i},{j}) {:.6e} vs {:.6e}",
                    a[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    /// Information-matrix equality: when `S = R`, all three estimators collapse
    /// to the model-based `R⁻¹` (`R⁻¹SR⁻¹ = R⁻¹RR⁻¹ = R⁻¹`, and `S⁻¹ = R⁻¹`). This
    /// is the asymptotic behaviour at the MLE of a correctly-specified model.
    #[test]
    fn test_combine_covariance_collapses_when_s_equals_r() {
        let l = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.5, 1.3]);
        let r = l.transpose() * &l; // SPD
        let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
        for m in [
            CovarianceMethod::Hessian,
            CovarianceMethod::CrossProduct,
            CovarianceMethod::Sandwich,
        ] {
            let cov = combine_covariance(m, r_inv.clone(), &r)
                .unwrap_or_else(|| panic!("{m:?} should produce a covariance"));
            assert_mat_close(&cov, &r_inv, 1e-9, &format!("{m:?} with S=R"));
        }
    }

    /// With `S ≠ R`, the sandwich is exactly `R⁻¹ S R⁻¹`.
    #[test]
    fn test_combine_covariance_sandwich_matches_explicit_product() {
        let l = DMatrix::from_row_slice(2, 2, &[1.7, 0.0, 0.3, 1.1]);
        let r = l.transpose() * &l;
        let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
        let s = DMatrix::from_row_slice(2, 2, &[3.0, 0.4, 0.4, 2.0]);
        let sandwich =
            combine_covariance(CovarianceMethod::Sandwich, r_inv.clone(), &s).expect("sandwich");
        let expected = &r_inv * &s * &r_inv;
        assert_mat_close(&sandwich, &expected, 1e-12, "sandwich = R⁻¹SR⁻¹");
        // Sandwich must stay symmetric (S and R⁻¹ are symmetric).
        assert_mat_close(
            &sandwich,
            &sandwich.transpose(),
            1e-12,
            "sandwich symmetric",
        );
    }

    /// A rank-deficient `S` (here a single score's outer product) is singular, so
    /// `S⁻¹` (cross-product) is unavailable — but the sandwich, which never
    /// inverts `S`, is still defined.
    #[test]
    fn test_combine_covariance_singular_s() {
        let l = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.2, 1.0]);
        let r = l.transpose() * &l;
        let r_inv = invert_psd_with_floor(&r).expect("R PD").inverse;
        let g = DVector::from_column_slice(&[1.0, 2.0]);
        let s_rank1 = &g * g.transpose(); // rank-1, singular 2×2
        assert!(
            combine_covariance(CovarianceMethod::CrossProduct, r_inv.clone(), &s_rank1).is_none(),
            "S⁻¹ must report singular S"
        );
        assert!(
            combine_covariance(CovarianceMethod::Sandwich, r_inv, &s_rank1).is_some(),
            "sandwich must tolerate rank-deficient S"
        );
    }

    /// A near-singular symmetric matrix with one tiny-negative eigenvalue
    /// (the exact failure mode reported in issue #129) is regularised: the
    /// helper flags the clip and returns a PD inverse with positive
    /// diagonals — what the old code rejected as "negative diagonal".
    #[test]
    fn test_invert_psd_with_floor_clips_negative_eigenvalue() {
        // H = Q diag(λ) Qᵀ with λ = [1.0, 0.5, -1e-9]. The tiny-negative
        // eigenvalue is the kind of FD noise the issue calls out.
        let q = {
            // Any orthogonal 3×3 will do. Use a Householder reflector built
            // from v = (1, 1, 1)/√3:  Q = I - 2 v vᵀ.
            let v = DVector::from_column_slice(&[1.0, 1.0, 1.0]) / (3.0_f64).sqrt();
            let mut m = DMatrix::<f64>::identity(3, 3);
            m -= 2.0 * &v * v.transpose();
            m
        };
        let lambdas = DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 0.5, -1e-9]));
        let h = &q * lambdas * q.transpose();

        let r = invert_psd_with_floor(&h).expect("near-PD input must regularise");
        assert_eq!(r.n_clipped, 1, "exactly one eigenvalue should clip");
        assert!(
            r.min_eigenvalue < 0.0 && r.min_eigenvalue.abs() < 1e-6,
            "min_eigenvalue should record the raw (pre-clip) value: {:.3e}",
            r.min_eigenvalue
        );

        // Inverse is PD ⇒ all diagonal entries positive. This is the assertion
        // the old neg-diag check used to fail on for the warfarin FD Hessian.
        for i in 0..3 {
            assert!(
                r.inverse[(i, i)] > 0.0,
                "regularised inverse diag[{i}] = {:.3e} should be positive",
                r.inverse[(i, i)]
            );
        }
        // Inverse is also numerically symmetric.
        for i in 0..3 {
            for j in i + 1..3 {
                assert!(
                    (r.inverse[(i, j)] - r.inverse[(j, i)]).abs() < 1e-12,
                    "inverse not symmetric at ({i},{j})",
                );
            }
        }
    }

    /// Hopelessly indefinite input (all eigenvalues ≤ 0) returns None. Because
    /// the eigenvalues are finite, `compute_covariance` surfaces this as
    /// `CovarianceStepResult::FailedNonPd` — carrying the eigenvalue-list warning
    /// and a usable fallback proposal — rather than `Unusable`.
    #[test]
    fn test_invert_psd_with_floor_rejects_negative_definite() {
        let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        assert!(invert_psd_with_floor(&h).is_none());
        // The eigenvalues are finite, so the fallback path is taken (not Unusable):
        // extract_eigenvalues succeeds and the proposal is all-finite and PD.
        let eigs = extract_eigenvalues(&h).expect("finite eigenvalues");
        assert!(eigs.iter().all(|e| e.is_finite()));
        let proposal = build_non_pd_fallback_proposal(&h, &[0, 1], 2, 4.0);
        assert!(
            proposal.iter().all(|v| v.is_finite()),
            "fallback proposal must be finite for a finite non-PD Hessian"
        );
    }

    /// extract_eigenvalues returns eigenvalues sorted descending and returns None
    /// for inputs with non-finite entries.
    #[test]
    fn test_extract_eigenvalues_sorts_descending() {
        let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        let ev = extract_eigenvalues(&h).expect("finite eigenvalues for this input");
        assert_eq!(ev.len(), 2);
        assert!(ev[0] >= ev[1], "eigenvalues must be sorted descending");
        assert!(
            ev.iter().all(|&e| e < 0.0),
            "both eigenvalues must be negative"
        );
    }

    /// format_non_pd_warning produces a message with the expected structure and
    /// includes the eigenvalue list.
    #[test]
    fn test_format_non_pd_warning_structure() {
        let ev = vec![8.4, 2.1, 0.3, -0.01];
        let msg = format_non_pd_warning(&ev);
        assert!(
            msg.contains("Hessian is not positive definite"),
            "message must flag non-PD Hessian"
        );
        assert!(
            msg.contains("Eigenvalues:"),
            "message must include eigenvalue list"
        );
        assert!(
            msg.contains("SE estimates not available"),
            "message must indicate SEs are unavailable"
        );
        // Most-negative eigenvalue appears in the output.
        assert!(
            msg.contains("-0.0100"),
            "negative eigenvalue must appear: {msg}"
        );
    }

    /// extract_eigenvalues returns None when the matrix contains a NaN entry.
    #[test]
    fn test_extract_eigenvalues_none_on_nan() {
        let mut h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        h[(0, 0)] = f64::NAN;
        assert!(
            extract_eigenvalues(&h).is_none(),
            "NaN entry must cause None return"
        );
    }

    /// packed_param_label decodes the lower-triangular block-omega index correctly.
    /// Packing order (column-major lower triangle): (0,0), (1,0), (1,1).
    /// With n_theta=2, packed_idx=3 → omega_idx=1 → (row=1, col=0).
    #[test]
    fn test_packed_param_label_block_omega() {
        use crate::types::{OmegaMatrix, SigmaVector};
        let mut mat = DMatrix::zeros(2, 2);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.04;
        mat[(0, 1)] = 0.01;
        mat[(1, 0)] = 0.01;
        let free_mask = DMatrix::from_element(2, 2, true);
        let omega = OmegaMatrix::from_matrix_with_mask(
            mat,
            vec!["ETA_CL".into(), "ETA_V".into()],
            false,
            free_mask,
        );
        let template = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false, false, false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };

        // n_theta=2, so: idx=2 → omega[ETA_CL, ETA_CL], idx=3 → omega[ETA_V, ETA_CL] (off-diag),
        // idx=4 → omega[ETA_V, ETA_V].
        let label_diag = packed_param_label(2, &template);
        assert_eq!(label_diag, "omega[ETA_CL, ETA_CL]", "diagonal 0,0");

        let label_off = packed_param_label(3, &template);
        assert_eq!(label_off, "omega[ETA_V, ETA_CL]", "off-diagonal 1,0");

        let label_diag2 = packed_param_label(4, &template);
        assert_eq!(label_diag2, "omega[ETA_V, ETA_V]", "diagonal 1,1");
    }

    /// `format_non_pd_warning` is a pure formatting function: it embeds whatever
    /// eigenvalue list it receives, regardless of sign. This test exercises the
    /// fixed-4 and scientific-3 branches of `fmt_eig` without needing an actual
    /// non-PD Hessian.
    ///
    /// Note: in production `format_non_pd_warning` is only reached when
    /// `invert_psd_with_floor` returns `None` (all eigenvalues ≤ 0), so the
    /// "all-positive" input below is a formatter unit test, not a semantic one.
    #[test]
    fn test_format_non_pd_warning_all_positive() {
        let ev = vec![5.0, 0.01, 1e-9];
        let msg = format_non_pd_warning(&ev);
        assert!(
            msg.contains("Hessian is not positive definite"),
            "message must flag non-PD Hessian even for all-positive inputs: {msg}"
        );
        assert!(
            msg.contains("5.0000"),
            "largest eigenvalue in output: {msg}"
        );
        assert!(msg.contains("0.0100"), "medium eigenvalue in output: {msg}");
        // Tiny eigenvalue formatted in scientific notation.
        assert!(msg.contains("e-"), "tiny eigenvalue in scientific: {msg}");
    }

    /// packed_param_label — sigma[1] and sigma[2] paths (1-indexed by convention).
    #[test]
    fn test_packed_param_label_sigma() {
        use crate::types::SigmaVector;
        // n_theta=1 (diagonal omega), n_omega=1 (diagonal), n_sigma=2
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["CL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega: crate::types::OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1, 0.2],
                names: vec!["ADD".into(), "PROP".into()],
            },
            sigma_fixed: vec![false, false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        // packed layout: [theta(0), omega(1), sigma(2), sigma(3)]
        assert_eq!(packed_param_label(2, &template), "sigma[1]");
        assert_eq!(packed_param_label(3, &template), "sigma[2]");
    }

    /// packed_param_label — kappa[1] path (IOV diagonal omega).
    #[test]
    fn test_packed_param_label_kappa() {
        use crate::types::SigmaVector;
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["CL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega: crate::types::OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(crate::types::OmegaMatrix::from_diagonal(
                &[0.02],
                vec!["KAPPA_CL".into()],
            )),
            kappa_fixed: vec![false],
        };
        // packed layout: [theta(0), omega(1), sigma(2), kappa(3)]
        assert_eq!(packed_param_label(3, &template), "kappa[1]");
    }

    /// invert_psd_with_floor severity thresholds: 1-of-3 clipped → pct=33 → "minor".
    #[test]
    fn test_regularization_severity_minor() {
        // Build a matrix with exactly one eigenvalue near-zero so exactly 1 of 3 is clipped.
        // Diagonal 3×3: eigenvalues are the diagonal entries.
        let h = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            1.0, 1.0, 1e-20, // this one will be clipped
        ]));
        let r = invert_psd_with_floor(&h).expect("should succeed");
        assert_eq!(r.n_clipped, 1, "exactly one eigenvalue should be clipped");
        // pct = 1*100/3 = 33 → "minor" threshold
        let pct = r.n_clipped * 100 / 3;
        assert_eq!(pct, 33, "33% → minor severity bucket");
    }

    /// fd_hessian_step = 0.0 triggers Unusable early return in compute_covariance.
    #[test]
    fn test_compute_covariance_invalid_eps() {
        use crate::types::{FitOptions, OmegaMatrix, SigmaVector};
        let model = make_model();
        let population = make_population(1);
        let template = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["CL".into(), "V".into()],
            theta_lower: vec![0.1, 1.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false, false],
            omega: OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        let x_hat: Vec<f64> = vec![
            5.0_f64.ln(),
            50.0_f64.ln(),
            0.04_f64.sqrt().ln(),
            0.1_f64.ln(),
        ];
        let eta_hats = vec![nalgebra::DVector::zeros(1)];
        let h_mats = vec![DMatrix::zeros(1, 1)];
        let kappas = vec![vec![]];
        let mut opts = FitOptions::default();
        opts.fd_hessian_step = 0.0;

        let result = compute_covariance(
            &x_hat,
            &template,
            &model,
            &population,
            &eta_hats,
            &h_mats,
            &kappas,
            &opts,
        );
        assert!(
            matches!(result, CovarianceStepResult::Unusable(_)),
            "eps=0.0 must return Unusable"
        );
        if let CovarianceStepResult::Unusable(msg) = result {
            assert!(
                msg.contains("fd_hessian_step"),
                "message names the option: {msg}"
            );
        }
    }

    /// Empty matrix is a valid zero-dimensional input (all-FIX parameter
    /// case). Returns a 0×0 inverse without clipping.
    #[test]
    fn test_invert_psd_with_floor_empty() {
        let r =
            invert_psd_with_floor(&DMatrix::<f64>::zeros(0, 0)).expect("0×0 input must succeed");
        assert_eq!(r.inverse.nrows(), 0);
        assert_eq!(r.n_clipped, 0);
    }

    // ── detect_stagnation: NLopt stagnation-guard unit tests ─────────────────

    /// Build a minimal `NloptState` for `detect_stagnation` unit tests.
    /// The non-stagnation fields (`cached_etas`, `cached_h_mats`, `prev_x`)
    /// are left empty because `detect_stagnation` reads only the
    /// `*_evals` / `*_improvement` / `stagnation_stopped` fields.
    fn fresh_state() -> NloptState {
        NloptState {
            cached_etas: Vec::new(),
            cached_h_mats: Vec::new(),
            best_ofv: 0.0,
            n_evals: 0,
            n_grad_evals: 0,
            prev_x: Vec::new(),
            last_improvement_eval: 0,
            best_at_last_improvement: f64::INFINITY,
            stagnation_stopped: false,
        }
    }

    /// `detect_stagnation(enabled=false)` is a no-op: it never latches, never
    /// fires, even after a window of zero improvement.  Replaces the
    /// end-to-end `stagnation_guard_toggle_runs_to_natural_termination` test
    /// (removed from `tests/new_optimizers.rs`), which became unreliable
    /// after SLSQP's own xtol fires before the guard window elapses on the
    /// warfarin example (both guard-on and guard-off now exit at exactly
    /// 100 evals via NLopt `XtolReached`, so the e2e toggle comparison no
    /// longer discriminates).
    #[test]
    fn test_detect_stagnation_disabled_never_fires() {
        let mut state = fresh_state();
        state.best_ofv = -100.0;
        state.best_at_last_improvement = -100.0;
        // Far past the stagnation window (n=7 → window = max(3*8, 50) = 50)
        // with zero improvement: still must not fire when disabled.
        for n_evals in 0..200 {
            state.n_evals = n_evals;
            assert!(
                !detect_stagnation(&mut state, 7, false),
                "enabled=false must never fire (n_evals={n_evals})"
            );
        }
        assert!(
            !state.stagnation_stopped,
            "disabled path must not latch `stagnation_stopped`"
        );
    }

    /// `detect_stagnation(enabled=true)` fires once `n_evals - last_improvement`
    /// reaches the stagnation window and latches sticky thereafter.
    #[test]
    fn test_detect_stagnation_enabled_fires_at_window_and_latches() {
        let mut state = fresh_state();
        state.best_ofv = -100.0;
        state.best_at_last_improvement = -100.0; // identical → no improvement
        state.last_improvement_eval = 0;

        let n = 7usize;
        let window = (3 * (n + 1)).max(50); // = 50

        // Within the window, no firing.
        for n_evals in 1..window {
            state.n_evals = n_evals;
            assert!(
                !detect_stagnation(&mut state, n, true),
                "must not fire inside window (n_evals={n_evals}, window={window})"
            );
            assert!(!state.stagnation_stopped);
        }

        // At the window, fires and latches.
        state.n_evals = window;
        assert!(
            detect_stagnation(&mut state, n, true),
            "must fire at window (n_evals={window})"
        );
        assert!(
            state.stagnation_stopped,
            "first firing must latch `stagnation_stopped`"
        );

        // Latched: subsequent calls keep returning `true` without re-checking
        // the window arithmetic.  Drop n_evals well below the window to prove
        // the short-circuit is on `stagnation_stopped`, not on the counter.
        state.n_evals = 1;
        assert!(
            detect_stagnation(&mut state, n, true),
            "latched state must stay sticky-true regardless of n_evals"
        );
    }

    /// `detect_stagnation` resets the improvement counter when OFV moves down
    /// by more than the 1e-3 threshold — so a long run of fruitful descent
    /// never triggers the guard.
    #[test]
    fn test_detect_stagnation_resets_on_improvement() {
        let mut state = fresh_state();
        state.best_ofv = -100.0;
        state.best_at_last_improvement = -100.0;
        state.last_improvement_eval = 0;

        let n = 7usize;
        // Walk almost up to the window with zero improvement…
        state.n_evals = 49;
        assert!(!detect_stagnation(&mut state, n, true));

        // …then improve OFV by > 1e-3.  Improvement must reset the
        // last-improvement counter so the next 50 evals start fresh.
        state.best_ofv = -100.5;
        state.n_evals = 50;
        assert!(
            !detect_stagnation(&mut state, n, true),
            "improvement must reset the counter"
        );
        assert_eq!(
            state.last_improvement_eval, 50,
            "last_improvement_eval must advance to the improving eval"
        );
        assert_eq!(
            state.best_at_last_improvement, -100.5,
            "best_at_last_improvement must update to the new best"
        );

        // Now we need another full window of zero improvement before firing.
        state.n_evals = 99;
        assert!(!detect_stagnation(&mut state, n, true));
        state.n_evals = 100;
        assert!(detect_stagnation(&mut state, n, true));
    }

    /// Improvement *below* the 1e-3 threshold counts as stagnation — the
    /// guard is deliberately noise-tolerant.  Without this, OFV noise of a
    /// few ULPs would constantly reset the counter and the guard would
    /// never fire.
    #[test]
    fn test_detect_stagnation_subthreshold_improvement_does_not_reset() {
        let mut state = fresh_state();
        state.best_ofv = -100.0;
        state.best_at_last_improvement = -100.0;
        state.last_improvement_eval = 0;

        let n = 7usize;
        // Improve OFV by 5e-4 — below the 1e-3 threshold.  Counter must
        // NOT reset.
        state.best_ofv = -100.0005;
        state.n_evals = 25;
        assert!(!detect_stagnation(&mut state, n, true));
        assert_eq!(
            state.last_improvement_eval, 0,
            "sub-threshold improvement must not advance the counter"
        );

        // 50 evals after the original last_improvement_eval (= 0), it fires.
        state.n_evals = 50;
        assert!(detect_stagnation(&mut state, n, true));
    }

    #[test]
    fn test_reconverge_this_eval_schedule() {
        let mut opts = FitOptions::default();

        // Interval 0 (the default): never reconverge, and never a
        // divide-by-zero from the modulo (the `!= 0` guard short-circuits).
        opts.reconverge_gradient_interval = 0;
        for idx in 0..7 {
            assert!(!reconverge_this_eval(&opts, idx), "idx {idx}");
        }

        // Interval 1: every eval reconverges (the always-on case).
        opts.reconverge_gradient_interval = 1;
        for idx in 0..7 {
            assert!(reconverge_this_eval(&opts, idx), "idx {idx}");
        }

        // Interval 5: reconverge only on 0, 5, 10, …
        opts.reconverge_gradient_interval = 5;
        let got: Vec<usize> = (0..12)
            .filter(|&i| reconverge_this_eval(&opts, i))
            .collect();
        assert_eq!(got, vec![0, 5, 10]);
    }

    use crate::types::{
        BloqMethod, CompiledModel, DoseEvent, ErrorModel, FitOptions, GradientMethod,
        ModelParameters, OmegaMatrix, PkModel, PkParams, Population, SigmaVector, Subject,
    };
    use nalgebra::DVector;
    use std::collections::HashMap;

    fn make_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        CompiledModel {
            name: "outer_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
        }
    }

    fn make_population(n_subj: usize) -> Population {
        let subjects = (0..n_subj)
            .map(|_| Subject {
                id: "S1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 4.0, 8.0],
                obs_raw_times: Vec::new(),
                observations: vec![25.0, 15.0, 9.0],
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: vec![1, 1, 1],
                dose_occasions: vec![1],
                fremtype: Vec::new(),
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn check_gradient(model: &CompiledModel, population: &Population, n_eta: usize) {
        let template = &model.default_params;
        let n_subj = population.subjects.len();
        let n_obs = population.subjects[0].observations.len();

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        // FOCE non-interaction: the AD/analytical population-gradient path is the
        // Sheiner–Beal-derived closed form for the SB NLL. FOCEI INTER now uses
        // the Almquist Laplace NLL, whose gradient takes the FD fallback in
        // `subject_nll_pop_grad`; under INTER this test would be vacuous
        // (FD-vs-FD). Set interaction=false to exercise the analytical path.
        let mut options = FitOptions::default();
        options.interaction = false;

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        // Use a non-zero H-matrix so r_tilde = R + H·Ω·Hᵀ depends on Ω and
        // the omega/off-diagonal Cholesky gradients are non-trivially exercised.
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

        let ad_grad = ad_population_gradient(
            &x,
            n_subj,
            template,
            model,
            population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            &options,
        );

        let ofv_at = |xp: &[f64]| -> f64 {
            let p = unpack_params(xp, template);
            2.0 * pop_nll(
                model,
                population,
                &p,
                &eta_hats,
                &h_matrices,
                &kappas,
                options.interaction,
            )
        };
        let eps = 1e-4;
        let fd_grad: Vec<f64> = (0..n)
            .map(|j| {
                let h = eps * (1.0 + x[j].abs());
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[j] += h;
                xm[j] -= h;
                (ofv_at(&xp) - ofv_at(&xm)) / (2.0 * h)
            })
            .collect();

        for j in 0..n {
            let tol = 1e-4 * (1.0 + fd_grad[j].abs());
            assert!(
                (ad_grad[j] - fd_grad[j]).abs() < tol,
                "grad[{j}]: AD={:.6e}, FD={:.6e}",
                ad_grad[j],
                fd_grad[j],
            );
        }
    }

    /// IIV (diagonal omega, 1 ETA): analytical path.
    #[test]
    fn test_outer_ad_gradient_iiv() {
        check_gradient(&make_model(), &make_population(3), 1);
    }

    /// Block omega (2×2 with off-diagonal): tests Cholesky-param gradient.
    #[test]
    fn test_outer_ad_gradient_block_omega() {
        use crate::types::{OmegaMatrix, PkParams};
        // 2-ETA model: CL and V both random with correlation.
        // Build 2×2 omega with variance 0.04 on diagonal and covariance 0.01.
        let mut mat = nalgebra::DMatrix::zeros(2, 2);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.04;
        mat[(0, 1)] = 0.01;
        mat[(1, 0)] = 0.01;
        let free_mask = nalgebra::DMatrix::from_element(2, 2, true);
        let omega = OmegaMatrix::from_matrix_with_mask(
            mat,
            vec!["ETA_CL".into(), "ETA_V".into()],
            false,
            free_mask,
        );
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false, false, false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "block_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1] * eta[1].exp();
                p
            }),
            n_theta: 2,
            n_eta: 2,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into(), "ETA_V".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false; 2],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0, 1],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
            frem_config: None,
        };
        check_gradient(&model, &make_population(3), 2);
    }

    /// With omega_iov=None but non-empty kappas, subject_nll_pop_grad falls
    /// through to central FD (no analytical IOV path without omega_iov).
    /// The population-sum must still match the population-level FD reference.
    ///
    /// Note: with omega_iov=None the IOV NLL formula is not exercised here —
    /// this test covers the FD *code path*, not IOV NLL correctness.
    #[test]
    fn test_outer_ad_gradient_fd_fallback_path() {
        // `subject_nll_at` only enters the IOV NLL branch when kappas is
        // non-empty AND omega_iov is Some. With omega_iov=None and non-empty
        // kappas the function falls through to standard FOCE; without
        // omega_iov the dispatch in subject_nll_pop_grad also falls through to
        // central FD — the code path this test exercises.
        let model = make_model();

        let template = &model.default_params;
        let n_subj = 3;
        let n_eta = 1;
        let n_obs = 3;
        let population = make_population(n_subj);

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        let options = FitOptions::default();

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::zeros(n_obs, n_eta))
            .collect();
        // Non-empty kappas trigger the FD fallback path.
        let kappas: Vec<Vec<DVector<f64>>> = (0..n_subj).map(|_| vec![DVector::zeros(1)]).collect();

        let ad_grad = ad_population_gradient(
            &x,
            n_subj,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            &options,
        );

        let ofv_at = |xp: &[f64]| -> f64 {
            let p = unpack_params(xp, template);
            2.0 * pop_nll(
                &model,
                &population,
                &p,
                &eta_hats,
                &h_matrices,
                &kappas,
                options.interaction,
            )
        };
        let eps = 1e-4;
        let fd_grad: Vec<f64> = (0..n)
            .map(|j| {
                let h = eps * (1.0 + x[j].abs());
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[j] += h;
                xm[j] -= h;
                (ofv_at(&xp) - ofv_at(&xm)) / (2.0 * h)
            })
            .collect();

        for j in 0..n {
            let tol = 1e-3 * (1.0 + fd_grad[j].abs());
            assert!(
                (ad_grad[j] - fd_grad[j]).abs() < tol,
                "IOV grad[{j}]: AD={:.6e}, FD={:.6e}",
                ad_grad[j],
                fd_grad[j],
            );
        }
    }

    // ── covariance_gradient (issue #209 / #243) ─────────────────────────────

    /// `covariance_gradient` (FOCE path, interaction=false) must match FD of
    /// `ofv_fixed = 2·pop_nll`. The Sheiner–Beal marginal already carries the Ω
    /// penalty via R̃ = HΩHᵀ + R, so there is no separate omega-prior add-back
    /// (issue #243 — adding one double-counted Ω and under-stated the FOCE
    /// omega SEs).
    #[test]
    fn test_covariance_gradient_foce_matches_fd_ofv_fixed() {
        let model = make_model();
        let template = &model.default_params;
        let population = make_population(3);
        let n_subj = 3;
        let n_obs = 3;
        let n_eta = 1;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        let mut options = FitOptions::default();
        options.interaction = false; // FOCE: Sheiner–Beal marginal

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

        // FOCE ofv_fixed = 2·pop_nll (Ω penalty already inside the SB marginal).
        let ofv_fixed = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, template);
            2.0 * pop_nll(
                &model,
                &population,
                &p,
                &eta_hats,
                &h_matrices,
                &kappas,
                false, // FOCE
            )
        };

        let grad = covariance_gradient(
            &x,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            &options,
        );

        let eps = 1e-4;
        for j in 0..n {
            let h = eps * (1.0 + x[j].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd = (ofv_fixed(&xp) - ofv_fixed(&xm)) / (2.0 * h);
            let tol = 1e-3 * (1.0 + fd.abs());
            assert!(
                (grad[j] - fd).abs() < tol,
                "covariance_gradient FOCE [{j}]: grad={:.6e}, FD_ofv={:.6e}",
                grad[j],
                fd,
            );
        }
    }

    /// `covariance_gradient` (FOCEI path, interaction=true) must match FD of
    /// `2·pop_nll` only — pop_nll already contains ηᵀΩ⁻¹η + log|Ω| per subject.
    #[test]
    fn test_covariance_gradient_focei_matches_fd_2pop_nll() {
        let model = make_model();
        let template = &model.default_params;
        let population = make_population(3);
        let n_subj = 3;
        let n_obs = 3;
        let n_eta = 1;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        let mut options = FitOptions::default();
        options.interaction = true; // FOCEI: omega prior inside pop_nll

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

        // FOCEI ofv_fixed = 2·pop_nll (omega prior already inside)
        let ofv_fixed_focei = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, template);
            2.0 * pop_nll(
                &model,
                &population,
                &p,
                &eta_hats,
                &h_matrices,
                &kappas,
                true, // FOCEI
            )
        };

        let grad = covariance_gradient(
            &x,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            &options,
        );

        let eps = 1e-4;
        for j in 0..n {
            let h = eps * (1.0 + x[j].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd = (ofv_fixed_focei(&xp) - ofv_fixed_focei(&xm)) / (2.0 * h);
            let tol = 1e-3 * (1.0 + fd.abs());
            assert!(
                (grad[j] - fd).abs() < tol,
                "covariance_gradient FOCEI [{j}]: grad={:.6e}, FD_ofv={:.6e}",
                grad[j],
                fd,
            );
        }
    }

    /// End-to-end guard for `compute_covariance` (#209 factor-of-2, #256 flatten,
    /// #274 Δ correction): the reconverging point-flatten gradient-FD covariance
    /// must (a) compute without regularization on a well-conditioned surface,
    /// (b) be positive-definite, and (c) equal `2·H⁻¹` of an *independently*
    /// reconverged scalar-FD Hessian of the same FOCEI objective. Because the
    /// model has proportional error, the reference (a second difference of the
    /// reconverged OFV) carries the `log|H̃|` EBE-response curvature `Δ`; the
    /// gradient-FD path only matches it because the #274 correction adds `Δ` back —
    /// so this also guards the Δ correction. A missing factor of two would be ~29%
    /// off (caught by the 15% band); a broken reconvergence would diverge wildly.
    #[test]
    fn test_compute_covariance_reconverged_matches_scalar_fd_with_factor_two() {
        let model = make_model();
        // Put the model at a near-optimum: set observations to the η=0
        // predictions of the 1-cpt IV model (CL=5, V=50, dose=100):
        // conc(t) = (100/50)·exp(−(5/50)·t) at t = 1, 4, 8.
        let mut population = make_population(8);
        for s in &mut population.subjects {
            s.observations = vec![1.80967, 1.34064, 0.89866];
        }
        // Fix ω and σ so the free block is the θ Hessian, which is positive
        // definite at this near-optimum (ω/σ would otherwise be pulled toward
        // their boundaries by the noise-free residuals, an artefact of the
        // synthetic data, not of the covariance code).
        let mut template = model.default_params.clone();
        template.omega_fixed = vec![true];
        template.sigma_fixed = vec![true];
        let template = &template;

        let n_subj = 8;
        let n_eta = 1;
        let n_obs = 3;
        let x = pack_params(template);
        let n = x.len();
        let mut options = FitOptions::default();
        options.interaction = true;

        // Warm-start EBEs (compute_covariance reconverges from these; the passed
        // h_matrices are intentionally ignored and recomputed).
        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
            .map(|_| DMatrix::from_element(n_obs, n_eta, 0.1))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

        let out = match compute_covariance(
            &x,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &options,
        ) {
            CovarianceStepResult::Success(out) => out,
            CovarianceStepResult::Unusable(msg) => {
                panic!("covariance must compute on the synthetic 1-cpt model: {msg}")
            }
            CovarianceStepResult::FailedNonPd { reason, .. } => {
                panic!("covariance must be PD on synthetic 1-cpt model: {reason}")
            }
        };

        // (a) No eigenvalue clipping on this well-conditioned surface.
        assert!(
            out.warnings.is_empty(),
            "unexpected covariance regularization: {:?}",
            out.warnings
        );

        let fixed = packed_fixed_mask(template);
        let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();

        // (b) Positive-definite: every free diagonal is positive and finite.
        for &i in &free_idx {
            let v = out.matrix[(i, i)];
            assert!(
                v.is_finite() && v > 0.0,
                "covariance diagonal [{i}] = {v} is not positive-finite"
            );
        }

        // (c) Independent reference: 2·inv(reconverged scalar-FD Hessian).
        // Mirrors the production `ofv` closure (interaction=true → 2·pop_nll,
        // reconverging EBEs from the same warm start).
        let ofv = |xv: &[f64]| -> f64 {
            let params = unpack_params(xv, template);
            let mu_k = compute_mu_k(&model, &params.theta, options.mu_referencing);
            let (ehs, hms, _s, kaps) = run_inner_loop_warm(
                &model,
                &population,
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&eta_hats),
                Some(&mu_k),
                options.min_obs_for_convergence_check as usize,
            );
            2.0 * pop_nll(&model, &population, &params, &ehs, &hms, &kaps, true)
        };

        let eps = 1e-2;
        let f0 = ofv(&x);
        let nf = free_idx.len();
        let mut h = DMatrix::zeros(nf, nf);
        let mut x_ij = x.clone();
        for (a, &i) in free_idx.iter().enumerate() {
            let hi = eps * (1.0 + x[i].abs());
            x_ij[i] = x[i] + hi;
            let fp = ofv(&x_ij);
            x_ij[i] = x[i] - hi;
            let fm = ofv(&x_ij);
            x_ij[i] = x[i];
            h[(a, a)] = (fp - 2.0 * f0 + fm) / (hi * hi);
            for (b, &j) in free_idx.iter().enumerate() {
                if j <= i {
                    continue;
                }
                let hj = eps * (1.0 + x[j].abs());
                x_ij[i] = x[i] + hi;
                x_ij[j] = x[j] + hj;
                let fpp = ofv(&x_ij);
                x_ij[j] = x[j] - hj;
                let fpm = ofv(&x_ij);
                x_ij[i] = x[i] - hi;
                let fmm = ofv(&x_ij);
                x_ij[j] = x[j] + hj;
                let fmp = ofv(&x_ij);
                x_ij[i] = x[i];
                x_ij[j] = x[j];
                let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                h[(a, b)] = v;
                h[(b, a)] = v;
            }
        }
        let h_sym = (&h + h.transpose()) * 0.5;
        let ref_cov = invert_psd_with_floor(&h_sym)
            .expect("reference Hessian inverts")
            .inverse
            * 2.0;

        // SE (sqrt of diagonal) must agree within 15%: catches a missing factor
        // of two (~29%) and any reconvergence/scale break, while tolerating the
        // gradient-FD-vs-scalar-FD truncation difference at eps=1e-2.
        for (a, &i) in free_idx.iter().enumerate() {
            let se_prod = out.matrix[(i, i)].sqrt();
            let se_ref = ref_cov[(a, a)].sqrt();
            let rel = (se_prod - se_ref).abs() / se_ref;
            assert!(
                rel < 0.15,
                "SE[{i}]: compute_covariance {se_prod:.6e} vs scalar-FD reference {se_ref:.6e} (rel {:.1}%)",
                rel * 100.0
            );
        }
    }

    /// Coverage + smoke guard for the **IOV** covariance branch (#256 flatten +
    /// the #298 perturbation-spec memory rewrite): an IOV model routes through
    /// the scalar-`OFV`-2nd-difference `serial_ofv` stencil — subjects
    /// reconverged via the shared `reconverge_point`, points built from the
    /// lightweight `Pert` specs rather than materialised x-vectors. ω/κ/σ are
    /// fixed so the free block is the θ Hessian (positive-definite at the
    /// near-optimum where observations equal the η=κ=0 predictions); the test
    /// asserts the branch runs and returns positive-finite θ SEs.
    #[test]
    fn test_compute_covariance_iov_runs_and_is_pd() {
        // 1-cpt IV, CL = θ₀·exp(η); IOV κ on CL. Predictions at η=κ=0:
        // conc(t) = (100/50)·exp(−(5/50)·t) = 2·exp(−0.1·t).
        let preds: Vec<f64> = (1..=6).map(|t| 2.0 * (-0.1 * t as f64).exp()).collect();

        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![true], // fix ω/κ/σ → free block is the θ Hessian
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![true],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![true],
        };
        let model = CompiledModel {
            frem_config: None,
            name: "iov_cov_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            dose_attr_map: Default::default(),
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        };

        let n_subj = 6;
        let subjects = (0..n_subj)
            .map(|_| Subject {
                fremtype: Vec::new(),
                id: "S".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                obs_raw_times: Vec::new(),
                observations: preds.clone(),
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 6],
                occasions: vec![1, 1, 1, 2, 2, 2],
                dose_occasions: vec![1],
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        let population = Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let template = &model.default_params;
        let x = pack_params(template);
        let n = x.len();
        let mut options = FitOptions::default();
        options.interaction = true;

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(1)).collect();
        let h_matrices: Vec<DMatrix<f64>> = (0..n_subj)
            .map(|_| DMatrix::from_element(6, 1, 0.1))
            .collect();
        // Non-empty per-occasion kappas → is_iov = true → exercises the IOV
        // scalar-FD stencil (serial_ofv + Pert specs + reconverge_point kaps).
        let kappas: Vec<Vec<DVector<f64>>> = (0..n_subj)
            .map(|_| vec![DVector::zeros(1), DVector::zeros(1)])
            .collect();

        let out = match compute_covariance(
            &x,
            template,
            &model,
            &population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &options,
        ) {
            CovarianceStepResult::Success(out) => out,
            CovarianceStepResult::Unusable(msg) => panic!("IOV covariance unusable: {msg}"),
            CovarianceStepResult::FailedNonPd { reason, .. } => {
                panic!("IOV covariance not PD: {reason}")
            }
        };

        let fixed = packed_fixed_mask(template);
        let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed[i]).collect();
        assert!(!free_idx.is_empty(), "θ block must be free");
        for &i in &free_idx {
            let v = out.matrix[(i, i)];
            assert!(
                v.is_finite() && v > 0.0,
                "IOV covariance diagonal [{i}] = {v} is not positive-finite"
            );
        }
    }

    // ── SLSQP overshoot guard tests (issue #55) ────────────────────────────
    //
    // NLopt LD_SLSQP starts every fit with its quasi-Newton Hessian set to
    // identity; the QP's unconstrained first step is therefore d = -∇f. The
    // AD/analytical FOCE gradient introduced in PR #48 has inf-norm ≈ 10²–10³
    // on standard PK models, while the scaled bound width is ≈ 3–9, so the
    // projected step lands at a corner of the box and the OFV explodes. The
    // `cap_slsqp_gradient` helper rescales `g` by a single scalar so the
    // would-be Newton step fits inside the box on every dimension.

    /// Cap fires when the gradient inf-norm exceeds the per-dimension
    /// step budget, and the cap is a uniform rescale (preserves direction
    /// and relative magnitudes between components).
    #[test]
    fn test_cap_slsqp_gradient_uniformly_rescales_when_huge() {
        // Bounds chosen so each dimension's budget = clamp(half-width, 0.1, 1.0).
        //   i=0: width=2.0 → budget = clamp(1.0, …) = 1.0
        //   i=1: width=4.0 → budget = clamp(2.0, …) = 1.0 (clamped to 1.0)
        //   i=2: width=0.2 → budget = clamp(0.1, …) = 0.1 (clamped to 0.1)
        let lower = vec![-1.0, -2.0, -0.1];
        let upper = vec![1.0, 2.0, 0.1];

        // Gradient with inf-norm 200 at the third component → worst_ratio = 200/0.1 = 2000.
        let mut g = vec![10.0, 100.0, 200.0];
        let g_before = g.clone();
        let fired = cap_slsqp_gradient(&mut g, &lower, &upper);
        assert!(fired, "cap should have fired for huge gradient");

        // Direction preserved: g[i] / g_before[i] is the same scalar across i.
        let scalar0 = g[0] / g_before[0];
        let scalar1 = g[1] / g_before[1];
        let scalar2 = g[2] / g_before[2];
        assert!(
            (scalar0 - scalar1).abs() < 1e-12 && (scalar1 - scalar2).abs() < 1e-12,
            "cap should be a uniform rescale: scalars {scalar0}, {scalar1}, {scalar2}",
        );

        // Inf-norm relative to per-dim budget should be exactly 1.0 after capping
        // (the dimension that drove the rescale is now at its budget).
        let after_inf_ratio = (g[0].abs() / 1.0)
            .max(g[1].abs() / 1.0)
            .max(g[2].abs() / 0.1);
        assert!(
            (after_inf_ratio - 1.0).abs() < 1e-12,
            "post-cap inf-norm ratio should equal 1.0, got {after_inf_ratio}",
        );
    }

    /// Cap is a no-op when the gradient is already within budget — preserves
    /// SLSQP convergence behaviour once it's in the basin of the optimum.
    #[test]
    fn test_cap_slsqp_gradient_noop_when_within_budget() {
        let lower = vec![-1.0, -2.0];
        let upper = vec![1.0, 2.0];
        // Per-dim budgets are both clamped to 1.0; gradient inf-norm = 0.5 < 1.0.
        let mut g = vec![0.5, -0.3];
        let g_before = g.clone();
        let fired = cap_slsqp_gradient(&mut g, &lower, &upper);
        assert!(!fired, "cap should not fire for in-budget gradient");
        assert_eq!(g, g_before, "in-budget gradient must be untouched");
    }

    /// Even when one dimension has very wide bounds (a typical pattern in
    /// log-Cholesky omega/sigma packing, where bounds span 10+ units), the
    /// budget is clamped to 1.0 so the cap still fires.
    #[test]
    fn test_cap_slsqp_gradient_clamps_wide_bounds_to_unit_budget() {
        // Wide bounds: half-width = 5 → budget clamped to 1.0.
        let lower = vec![-10.0, -10.0];
        let upper = vec![10.0, 10.0];
        let mut g = vec![5.0, 0.0];
        let fired = cap_slsqp_gradient(&mut g, &lower, &upper);
        assert!(fired, "cap should fire: budget clamped to 1.0, |g_max| = 5");
        // Worst ratio = 5/1 = 5 → divide all by 5 → g[0] becomes 1.0.
        assert!(
            (g[0] - 1.0).abs() < 1e-12,
            "g[0] post-cap should be 1.0, got {}",
            g[0]
        );
        assert_eq!(g[1], 0.0);
    }

    /// Regression test for the original issue #55 symptom: SLSQP optimizing
    /// a multi-theta mu-referenced FOCEI fit terminated with theta byte-
    /// identical to init. The cap doesn't restore SLSQP to LBFGS's optimum
    /// (the QP is still less aggressive than a line-search method on this
    /// objective), but it does guarantee meaningful movement and a real OFV
    /// improvement — the failure mode of "looks converged, didn't run".
    ///
    /// Gated under `slow-tests` because it calls fit() to convergence.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_slsqp_moves_on_mu_referenced_two_cpt_oral_cov() {
        use crate::api::fit_from_files;
        use crate::types::{EstimationMethod, FitOptions, Optimizer};

        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            optimizer: Optimizer::Slsqp,
            outer_maxiter: 200,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let model_path = "examples/two_cpt_oral_cov.ferx";
        let data_path = "data/two_cpt_oral_cov.csv";
        let result =
            fit_from_files(model_path, data_path, None, Some(opts)).expect("fit should succeed");

        // Initial theta from the .ferx file: [4.0, 40.0, 8.0, 80.0, 1.0, 0.6, 0.3].
        let init = [4.0, 40.0, 8.0, 80.0, 1.0, 0.6, 0.3];
        let max_rel_delta = result
            .theta
            .iter()
            .zip(init.iter())
            .map(|(t, i)| ((t - i) / i).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_rel_delta > 0.01,
            "SLSQP didn't move (max relative theta change = {:.4e}); \
             this is the issue #55 byte-identical-theta regression.\n\
             theta = {:?}\ninit  = {:?}",
            max_rel_delta,
            result.theta,
            init,
        );

        // OFV at init on this model + data is around -1040; LBFGS finds
        // ≈ -1198. SLSQP-with-cap reaches ≈ -1182. Assert at least a
        // 100-unit OFV improvement so we catch silent regressions where
        // SLSQP only moves by a hair.
        assert!(
            result.ofv < -1140.0,
            "SLSQP OFV = {:.2} is too close to init (-1040); cap may be \
             overly aggressive and throttling convergence.",
            result.ofv,
        );
    }

    // ── built-in BFGS optimizer (non-NLopt path) ─────────────────────────────

    /// Drives the `Optimizer::Bfgs` branch of `optimize_population` for a few
    /// outer iterations. Exercises `optimize_bfgs`, `bfgs_update`, and
    /// `backtracking_line_search_warm` end-to-end on a tiny 1-cpt IV problem
    /// without running to convergence (fast, Tier-1).
    #[test]
    fn bfgs_optimizer_runs_and_improves_ofv() {
        use crate::types::{EstimationMethod, FitOptions, Optimizer};
        let model = make_model();
        let population = make_population(4);
        let opts = FitOptions {
            method: EstimationMethod::Foce,
            optimizer: Optimizer::Bfgs,
            outer_maxiter: 5,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };

        // OFV at the initial point, for an improvement comparison.
        let init_ofv = {
            let init = optimize_population(
                &model,
                &population,
                &model.default_params,
                &FitOptions {
                    optimizer: Optimizer::Bfgs,
                    outer_maxiter: 0,
                    run_covariance_step: false,
                    ..opts.clone()
                },
            );
            init.ofv
        };

        let result = optimize_population(&model, &population, &model.default_params, &opts);
        assert!(result.ofv.is_finite(), "BFGS produced non-finite OFV");
        assert_eq!(result.eta_hats.len(), population.subjects.len());
        // Built-in BFGS does not export a final gradient (NLopt-only field).
        assert!(result.final_gradient.is_none());
        // A handful of iterations should not make the OFV worse.
        assert!(
            result.ofv <= init_ofv + 1e-6,
            "BFGS worsened OFV: init={init_ofv:.4} final={:.4}",
            result.ofv
        );
    }

    // ── global pre-search (CRS2-LM) ──────────────────────────────────────────

    /// Exercises the `run_global_presearch` branch. CRS2-LM may be absent from
    /// the linked NLopt build; in that case the pre-search returns `Err` and a
    /// `global_search disabled` warning is recorded. Either way the run must
    /// finish with a finite OFV — this test just ensures the branch is taken
    /// and handled gracefully (small `global_maxeval` keeps it fast).
    #[test]
    fn global_presearch_branch_runs_or_warns() {
        use crate::types::{EstimationMethod, FitOptions, Optimizer};
        let model = make_model();
        let population = make_population(4);
        let opts = FitOptions {
            method: EstimationMethod::Foce,
            optimizer: Optimizer::Bobyqa,
            outer_maxiter: 3,
            global_search: true,
            global_maxeval: 8,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let result = optimize_population(&model, &population, &model.default_params, &opts);
        assert!(
            result.ofv.is_finite(),
            "presearch run produced non-finite OFV"
        );
    }

    // ── Covariance Hessian throughput benchmark (issue #209) ─────────────────
    //
    // Run with:  cargo test --lib --no-default-features --features ci \
    //              bench_cov_hessian -- --ignored --nocapture

    /// Measures wall time for the gradient-FD Hessian (new path, issue #209) vs
    /// the legacy scalar-FD Hessian (reconstructed inline) on the same setup.
    ///
    /// n_free = 4 (2 theta + 1 omega + 1 sigma).
    /// Old: ~2·n_free² = 32 OFV evaluations.
    /// New: n_free+1 = 5 gradient evaluations.
    #[test]
    #[ignore = "benchmark: run with -- --ignored --nocapture"]
    fn bench_cov_hessian_throughput() {
        use std::time::Instant;

        let model = make_model();
        let template = &model.default_params;
        let population = make_population(30);
        let n_subj = 30;
        let n_obs = 3;
        let n_eta = 1;
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        let options = FitOptions::default();

        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::from_element(n_obs, n_eta, 0.1))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];

        let fixed_mask = packed_fixed_mask(template);
        let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed_mask[i]).collect();
        let eps = 1e-2;

        // ── Scalar-FD Hessian (old path, reconstructed inline) ────────────
        let ofv_at = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, template);
            let foce = pop_nll(
                &model,
                &population,
                &p,
                &eta_hats,
                &h_matrices,
                &kappas,
                options.interaction,
            );
            let omega_inv = p.omega.matrix.clone().cholesky().unwrap().inverse();
            let n_e = p.omega.dim();
            let log_det = 2.0 * (0..n_e).map(|i| p.omega.chol[(i, i)].ln()).sum::<f64>();
            let om_terms: f64 = eta_hats
                .iter()
                .map(|eta| eta.dot(&(&omega_inv * eta)) + log_det)
                .sum();
            2.0 * foce + om_terms
        };
        let f0 = ofv_at(&x);

        const REPS: u32 = 20;
        let t0 = Instant::now();
        for _ in 0..REPS {
            let mut hess = DMatrix::zeros(n, n);
            let mut xij = x.clone();
            for &i in &free_idx {
                let hi = eps * (1.0 + x[i].abs());
                xij[i] = x[i] + hi;
                let fp = ofv_at(&xij);
                xij[i] = x[i] - hi;
                let fm = ofv_at(&xij);
                xij[i] = x[i];
                if ((fp - 2.0 * f0 + fm) / (hi * hi)).is_finite() {
                    hess[(i, i)] = (fp - 2.0 * f0 + fm) / (hi * hi);
                }
                for &j in &free_idx {
                    if j <= i {
                        continue;
                    }
                    let hj = eps * (1.0 + x[j].abs());
                    xij[i] = x[i] + hi;
                    xij[j] = x[j] + hj;
                    let fpp = ofv_at(&xij);
                    xij[j] = x[j] - hj;
                    let fpm = ofv_at(&xij);
                    xij[i] = x[i] - hi;
                    let fmm = ofv_at(&xij);
                    xij[j] = x[j] + hj;
                    let fmp = ofv_at(&xij);
                    xij[i] = x[i];
                    xij[j] = x[j];
                    let v = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
                    if v.is_finite() {
                        hess[(i, j)] = v;
                        hess[(j, i)] = v;
                    }
                }
            }
            std::hint::black_box(hess);
        }
        let scalar_ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

        // ── Gradient-FD Hessian (new path) ───────────────────────────────
        let t1 = Instant::now();
        for _ in 0..REPS {
            let mut hess = DMatrix::zeros(n, n);
            let g0 = covariance_gradient(
                &x,
                template,
                &model,
                &population,
                &eta_hats,
                &h_matrices,
                &kappas,
                &bounds,
                &options,
            );
            for &k in &free_idx {
                let hk = eps * (1.0 + x[k].abs());
                let mut xp = x.clone();
                xp[k] += hk;
                let gp = covariance_gradient(
                    &xp,
                    template,
                    &model,
                    &population,
                    &eta_hats,
                    &h_matrices,
                    &kappas,
                    &bounds,
                    &options,
                );
                for &j in &free_idx {
                    let v = (gp[j] - g0[j]) / hk;
                    if v.is_finite() {
                        hess[(j, k)] = v;
                    }
                }
            }
            std::hint::black_box(hess);
        }
        let grad_ms = t1.elapsed().as_secs_f64() * 1000.0 / REPS as f64;

        println!(
            "\n── Covariance Hessian throughput (n_free={}, n_subj={}) ──────────",
            free_idx.len(),
            n_subj
        );
        println!("  scalar-FD (old): {:.2}ms/Hessian", scalar_ms);
        println!("  gradient-FD (new): {:.2}ms/Hessian", grad_ms);
        println!("  speedup: {:.1}×", scalar_ms / grad_ms);
    }

    // ── build_non_pd_fallback_proposal ───────────────────────────────────────

    /// Diagonal 2×2 Hessian with one negative eigenvalue (-2) and one positive
    /// (4). The proposal covariance should have eigenvalues inflation / |λ_i|,
    /// inflated by factor 4: so 4/2 = 2.0 and 4/4 = 1.0.
    #[test]
    fn build_fallback_proposal_is_pd_and_inflated() {
        let hess = DMatrix::from_row_slice(2, 2, &[-2.0_f64, 0.0, 0.0, 4.0]);
        let free_idx = [0usize, 1];
        let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 2, 4.0);
        // Result must be symmetric PD.
        assert!(proposal[(0, 0)] > 0.0, "diagonal must be positive");
        assert!(proposal[(1, 1)] > 0.0, "diagonal must be positive");
        assert!(
            (proposal[(0, 1)] - proposal[(1, 0)]).abs() < 1e-12,
            "must be symmetric"
        );
        // Eigenvalues of the proposal should be inflation / |original eigenvalue|.
        let eig = SymmetricEigen::new(proposal.clone());
        let mut evs: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Expected: [4/4, 4/2] = [1.0, 2.0]
        assert!(
            (evs[0] - 1.0).abs() < 1e-10,
            "smaller eigenvalue should be 1.0: {:?}",
            evs
        );
        assert!(
            (evs[1] - 2.0).abs() < 1e-10,
            "larger eigenvalue should be 2.0: {:?}",
            evs
        );
    }

    /// Fixed parameter (index 2 absent from free_idx) stays zero in full matrix.
    #[test]
    fn build_fallback_proposal_zeros_fixed_params() {
        let hess = DMatrix::from_row_slice(1, 1, &[2.0_f64]);
        let free_idx = [0usize];
        let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 3, 4.0);
        assert_eq!(proposal.nrows(), 3);
        assert_eq!(proposal.ncols(), 3);
        assert!(
            proposal[(0, 0)] > 0.0,
            "free param row/col must be non-zero"
        );
        assert_eq!(proposal[(1, 1)], 0.0, "fixed param row/col must be zero");
        assert_eq!(proposal[(2, 2)], 0.0, "fixed param row/col must be zero");
    }

    /// A near-zero eigenvalue must be floored *relative* to the largest, capping
    /// the proposal's condition number at `FALLBACK_PROPOSAL_MAX_COND`. With the
    /// old absolute `1e-10` floor a 1e-9 eigenvalue would give a variance of
    /// 4/1e-9 = 4e9 (and a 1e12 condition number) — far enough to scatter every
    /// SIR draw out of bounds. The relative floor caps it at 4/(λ_max/1e8).
    #[test]
    fn build_fallback_proposal_caps_condition_number() {
        // diag(1000, 1e-9): one well-determined direction, one near-flat.
        let hess = DMatrix::from_row_slice(2, 2, &[1000.0_f64, 0.0, 0.0, 1e-9]);
        let free_idx = [0usize, 1];
        let proposal = build_non_pd_fallback_proposal(&hess, &free_idx, 2, 4.0);
        let eig = SymmetricEigen::new(proposal.clone());
        let max_var = eig.eigenvalues.iter().cloned().fold(f64::MIN, f64::max);
        let min_var = eig.eigenvalues.iter().cloned().fold(f64::MAX, f64::min);
        // floor = 1000 / 1e8 = 1e-5 ⇒ largest variance = 4 / 1e-5 = 4e5,
        // well below the un-floored 4e9.
        assert!(
            max_var < 1e6,
            "near-zero direction variance must be capped by the relative floor, got {max_var:e}"
        );
        // Condition number of the proposal must not exceed the cap (allow a
        // little slack for the inflation/eigen round-trip).
        assert!(
            max_var / min_var <= FALLBACK_PROPOSAL_MAX_COND * 1.01,
            "proposal condition number {} exceeds cap {}",
            max_var / min_var,
            FALLBACK_PROPOSAL_MAX_COND
        );
    }

    // ── select_fd_step ───────────────────────────────────────────────────────

    /// When all stencils are finite from the start, no halvings occur and the
    /// initial step is returned unchanged.
    #[test]
    fn select_fd_step_no_halving_needed() {
        let ofv = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x_hat = [1.0f64, 2.0];
        let free_idx = [0usize, 1];
        let f0 = ofv(&x_hat);
        let (eps, halvings) = select_fd_step(&x_hat, &free_idx, 0.01, f0, &ofv);
        assert_eq!(eps, 0.01, "step should be unchanged");
        assert_eq!(halvings, 0, "no halvings expected");
    }

    /// When the initial step causes overflow (NaN stencils), the function halves
    /// until stencils are finite and returns the reduced step.
    #[test]
    fn select_fd_step_halves_on_overflow() {
        // Returns NaN whenever |x[0]| >= 0.5 — simulates model overflow.
        let ofv = |x: &[f64]| {
            if x[0].abs() >= 0.5 {
                f64::NAN
            } else {
                x[0] * x[0]
            }
        };
        let x_hat = [0.0f64];
        let free_idx = [0usize];
        let f0 = 0.0f64;
        // initial_eps=1.0 → hi=1.0 → x=1.0 ≥ 0.5 → NaN → halve
        // eps=0.5 → hi=0.5 → x=0.5 ≥ 0.5 → NaN → halve
        // eps=0.25 → hi=0.25 → x=0.25 < 0.5 → 0.0625 → OK
        let (eps, halvings) = select_fd_step(&x_hat, &free_idx, 1.0, f0, &ofv);
        assert_eq!(eps, 0.25, "should have halved twice");
        assert_eq!(halvings, 2);
        // Verify the chosen step actually produces finite stencils.
        let hi = eps * (1.0 + x_hat[0].abs());
        let fp = ofv(&[x_hat[0] + hi]);
        let fm = ofv(&[x_hat[0] - hi]);
        assert!(fp.is_finite() && fm.is_finite());
    }

    /// Empty free_idx — vacuously all OK, returns initial eps without halvings.
    #[test]
    fn select_fd_step_empty_free_idx() {
        let ofv = |_x: &[f64]| f64::NAN; // would fail any real stencil
        let (eps, halvings) = select_fd_step(&[1.0], &[], 0.01, 0.0, &ofv);
        assert_eq!(eps, 0.01);
        assert_eq!(halvings, 0);
    }

    /// Regression: a stencil whose *numerator* (fp − 2·f0 + fm) is finite but
    /// whose *quotient* (÷ hi²) overflows must not be accepted at halvings == 0.
    /// The old numerator-only check declared this step usable, then the FD loop —
    /// which divides — rejected the diagonal and the covariance step failed
    /// without ever halving. select_fd_step now applies the same quotient the FD
    /// loop does, so it recognises the step as unusable (and exhausts its
    /// halvings rather than falsely reporting success on the first try).
    #[test]
    fn select_fd_step_rejects_finite_numerator_infinite_quotient() {
        // f0 = 0; any non-zero perturbation returns 1e200, so the numerator is a
        // finite 2e200 but hi² is ~1e-200, making the quotient overflow to +inf.
        let ofv = |x: &[f64]| if x[0] == 0.0 { 0.0 } else { 1e200 };
        let x_hat = [0.0f64];
        let free_idx = [0usize];
        let f0 = 0.0f64;
        let initial_eps = 1e-100;
        // Numerator is finite (the old check would accept immediately) …
        let hi = initial_eps * (1.0 + x_hat[0].abs());
        let numerator = ofv(&[hi]) - 2.0 * f0 + ofv(&[-hi]);
        assert!(
            numerator.is_finite(),
            "test setup: numerator must be finite"
        );
        assert!(
            !(numerator / (hi * hi)).is_finite(),
            "test setup: quotient must overflow"
        );
        // … but the quotient overflows, so the step is not accepted on the first
        // pass: halvings > 0 (smaller steps can't rescue this pathological case,
        // so it exhausts the budget — the point is it did not return 0).
        let (_eps, halvings) = select_fd_step(&x_hat, &free_idx, initial_eps, f0, &ofv);
        assert!(
            halvings > 0,
            "finite-numerator/infinite-quotient step must not be accepted at halvings == 0"
        );
    }
}
