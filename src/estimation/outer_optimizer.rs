use crate::estimation::gauss_newton::subject_nll_pop_grad;
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{foce_population_nll, foce_population_nll_iov};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
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
    pub ebe_convergence_warnings: u32,
    pub max_unconverged_subjects: u32,
    pub total_ebe_fallbacks: u32,
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
fn detect_stagnation(state: &mut NloptState, n: usize) -> bool {
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
    let scale: Vec<f64> = if options.scale_params && !has_identity_theta {
        compute_scale(&x0)
    } else {
        vec![1.0; n]
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
            let grad_raw = ad_population_gradient(
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
        // After updating best_ofv, check whether we've stalled. If yes,
        // `stagnation_stopped` is latched and the early-return at the
        // top of the closure trips on the next eval.
        if detect_stagnation(state, n) && verbose {
            eprintln!(
                "Eval {:>4}: stagnation guard triggered (no improvement \
                 below 1e-3 in last window); next eval will short-circuit",
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
                let grad_raw = ad_population_gradient(
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
            if detect_stagnation(state, n) && verbose {
                eprintln!(
                    "Eval {:>4}: stagnation guard triggered in SLSQP fallback",
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
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            compute_covariance(
                &x0,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            )
        } else {
            None
        };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }
    if covariance_matrix.is_none() && options.run_covariance_step {
        warnings.push("Covariance step failed".to_string());
    }

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
        ebe_convergence_warnings: ebe_final.n_convergence_warnings as u32,
        max_unconverged_subjects: ebe_final.max_unconverged as u32,
        total_ebe_fallbacks: ebe_final.total_fallback as u32,
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
                prev_etas: &[DVector<f64>]|
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
        // d(OFV)/d(x) = 2 · Σᵢ d(NLL_i)/d(x)
        let g = ad_population_gradient(
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
        );
        let f = if ofv.is_finite() { ofv } else { 1e20 };
        (f, g, ehs, hms)
    };

    // Per-element scale factors for the BFGS outer loop.
    let scale: Vec<f64> = if options.scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    let bounds_s = PackedBounds {
        lower: lower_s,
        upper: upper_s,
    };

    // Wrappers that operate in scaled space; unscale before calling base closures.
    let fdfg_s = |xs: &[f64],
                  prev_etas: &[DVector<f64>]|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let (f, g_r, ehs, hms) = fdfg(&x_r, prev_etas);
        let g_s: Vec<f64> = (0..n).map(|i| g_r[i] * scale[i]).collect();
        (f, g_s, ehs, hms)
    };

    let f_only_s = |xs: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        f_only(&x_r, prev_etas)
    };

    // Scale initial x into optimizer space.
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();

    let (mut f_val, mut g, ehs, _) = fdfg_s(&xs, &cached_etas);
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

        let (f_new, g_new, ehs, _) = fdfg_s(&xs, &cached_etas);
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

    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            compute_covariance(
                &x_final,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            )
        } else {
            None
        };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }
    if covariance_matrix.is_none()
        && options.run_covariance_step
        && !crate::cancel::is_cancelled(&options.cancel)
    {
        warnings.push("Covariance step failed".to_string());
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
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared utilities
// ═══════════════════════════════════════════════════════════════════════════

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
    let per_subj: Vec<Vec<f64>> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            subject_nll_pop_grad(
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
            )
            .1
        })
        .collect();
    (0..np)
        .map(|k| per_subj.iter().map(|gi| gi[k]).sum::<f64>() * 2.0)
        .collect()
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

/// Compute covariance matrix via finite-difference Hessian at convergence.
pub(crate) fn compute_covariance(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> Option<DMatrix<f64>> {
    let n = x_hat.len();
    let eps = 1e-2; // large step for FD Hessian on log-scale parameters

    // OFV for covariance step: includes explicit Omega terms (log|Omega| + eta'*Omega_inv*eta)
    // so the Hessian is sensitive to Omega parameters.
    // This matches Julia's foce_population_nll_diff.
    let ofv_fixed = |x: &[f64]| -> f64 {
        let params = unpack_params(x, template);
        let foce_nll = pop_nll(
            model,
            population,
            &params,
            eta_hats,
            h_matrices,
            kappas,
            options.interaction,
        );

        // Add explicit Omega prior terms for each subject
        let n_subj = eta_hats.len();
        let n_eta = if n_subj > 0 { eta_hats[0].len() } else { 0 };

        let omega_inv = match params.omega.matrix.clone().cholesky() {
            Some(c) => c.inverse(),
            None => return 1e20,
        };
        let log_det_omega = {
            let mut ld = 0.0;
            for i in 0..n_eta {
                let lii = params.omega.chol[(i, i)];
                if lii > 0.0 {
                    ld += lii.ln();
                } else {
                    return 1e20;
                }
            }
            2.0 * ld
        };

        let mut omega_terms = 0.0;
        for eta in eta_hats {
            omega_terms += eta.dot(&(&omega_inv * eta)) + log_det_omega;
        }

        2.0 * foce_nll + omega_terms
    };

    let base_ofv = ofv_fixed(x_hat);
    if !base_ofv.is_finite() {
        if options.verbose {
            eprintln!("  Covariance failed: base OFV is non-finite");
        }
        return None;
    }

    // FIX parameters contribute no information — skip their FD stencils and,
    // after inverting the Hessian of the free block, leave their covariance
    // rows/cols at zero (→ SE = 0 downstream).
    let fixed_mask = packed_fixed_mask(template);
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed_mask[i]).collect();

    let mut hess = DMatrix::zeros(n, n);
    let mut x_ij = x_hat.to_vec();

    let f0 = base_ofv;

    for &i in &free_idx {
        let hi = eps * (1.0 + x_hat[i].abs());

        // Diagonal: 3-point formula  (f(x+h) - 2f(x) + f(x-h)) / h^2
        x_ij[i] = x_hat[i] + hi;
        let fp = ofv_fixed(&x_ij);
        x_ij[i] = x_hat[i] - hi;
        let fm = ofv_fixed(&x_ij);
        x_ij[i] = x_hat[i];

        let h_ii = (fp - 2.0 * f0 + fm) / (hi * hi);
        if h_ii.is_finite() {
            hess[(i, i)] = h_ii;
        }

        // Off-diagonal: 4-point stencil (over free indices only)
        for &j in &free_idx {
            if j <= i {
                continue;
            }
            let hj = eps * (1.0 + x_hat[j].abs());

            x_ij[i] = x_hat[i] + hi;
            x_ij[j] = x_hat[j] + hj;
            let fpp = ofv_fixed(&x_ij);

            x_ij[j] = x_hat[j] - hj;
            let fpm = ofv_fixed(&x_ij);

            x_ij[i] = x_hat[i] - hi;
            let fmm = ofv_fixed(&x_ij);

            x_ij[j] = x_hat[j] + hj;
            let fmp = ofv_fixed(&x_ij);

            x_ij[i] = x_hat[i];
            x_ij[j] = x_hat[j];

            let h_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            if h_ij.is_finite() {
                hess[(i, j)] = h_ij;
                hess[(j, i)] = h_ij;
            }
        }
    }

    // Check for non-finite or zero Hessian entries *in the free block*. Rows
    // and columns of FIX parameters are intentionally zero.
    let mut n_nonfinite = 0;
    let mut n_zero = 0;
    for &i in &free_idx {
        if hess[(i, i)].abs() < 1e-30 {
            n_zero += 1;
        }
        for &j in &free_idx {
            if !hess[(i, j)].is_finite() {
                n_nonfinite += 1;
            }
        }
    }

    if n_nonfinite > 0 || n_zero > 0 {
        if options.verbose {
            eprintln!(
                "  Covariance failed: Hessian has {} non-finite, {} zero-diagonal entries",
                n_nonfinite, n_zero
            );
        }
        return None;
    }

    // Build the reduced Hessian over free indices, invert, then embed back
    // into the full n×n covariance matrix (FIX rows/cols stay zero).
    let n_free = free_idx.len();
    if n_free == 0 {
        // Nothing to estimate — return an all-zero covariance so downstream
        // SE extraction reports zeros (all params FIX).
        return Some(DMatrix::zeros(n, n));
    }
    let mut hess_free = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            hess_free[(a, b)] = hess[(i, j)];
        }
    }
    let hess_free_sym = (&hess_free + hess_free.transpose()) * 0.5;
    match hess_free_sym.try_inverse() {
        Some(cov_free) => {
            let neg_diag: Vec<usize> = (0..n_free).filter(|&a| cov_free[(a, a)] <= 0.0).collect();
            if !neg_diag.is_empty() {
                if options.verbose {
                    eprintln!(
                        "  Covariance failed: negative diagonal in free-block at {:?}",
                        neg_diag
                    );
                }
                return None;
            }
            let mut cov = DMatrix::zeros(n, n);
            for (a, &i) in free_idx.iter().enumerate() {
                for (b, &j) in free_idx.iter().enumerate() {
                    cov[(i, j)] = cov_free[(a, b)];
                }
            }
            if options.verbose {
                eprintln!("  Covariance step successful");
            }
            Some(cov)
        }
        None => {
            if options.verbose {
                eprintln!("  Covariance failed: Hessian not invertible");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::parameterization::{compute_bounds, pack_params};
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
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
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
            default_params,
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
        }
    }

    fn make_population(n_subj: usize) -> Population {
        let subjects = (0..n_subj)
            .map(|_| Subject {
                id: "S1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 4.0, 8.0],
                observations: vec![25.0, 15.0, 9.0],
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: vec![1, 1, 1],
                dose_occasions: vec![1],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    fn check_gradient(model: &CompiledModel, population: &Population, n_eta: usize) {
        let template = &model.default_params;
        let n_subj = population.subjects.len();
        let n_obs = population.subjects[0].observations.len();

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();
        let options = FitOptions::default();

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
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
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
            default_params,
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0, 1],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
        };
        check_gradient(&model, &make_population(3), 2);
    }

    /// Non-empty kappas force the central-FD fallback in subject_nll_pop_grad
    /// (can_use_analytical = false); the population-sum must still match the
    /// population-level FD reference.
    ///
    /// Note: with omega_iov=None the IOV NLL formula is not exercised here —
    /// this test covers the FD *code path*, not IOV NLL correctness.
    #[test]
    fn test_outer_ad_gradient_fd_fallback_path() {
        // `subject_nll_at` only enters the IOV NLL branch when kappas is
        // non-empty AND omega_iov is Some. With omega_iov=None and non-empty
        // kappas the function falls through to standard FOCE, but
        // subject_nll_pop_grad still takes the per-subject central-FD path
        // because kappas.is_empty() is false — the path we want to exercise.
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
}
