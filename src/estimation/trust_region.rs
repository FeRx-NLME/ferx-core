use argmin::core::{
    CostFunction, Error, Executor, Gradient, Hessian, IterState, OptimizationResult, Problem,
    Solver, State, TerminationReason, TerminationStatus, KV,
};
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::gauss_newton::subject_nll_pop_grad;
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{pop_nll_opts, OuterResult};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_bounds, compute_mu_k, pack_params, unpack_params, PackedBounds,
};
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

/// Per-call cache for per-subject NLL gradients.
/// Avoids recomputing the inner loop and AD gradients when `gradient()` and
/// `hessian()` are called with the same parameter vector in the same TR iteration.
struct GradCache {
    x: Vec<f64>,
    etas: Vec<DVector<f64>>,
    h_mats: Vec<DMatrix<f64>>,
    per_subj_grads: Vec<Vec<f64>>,
}

struct FoceiProblem<'a> {
    model: &'a CompiledModel,
    population: &'a Population,
    options: &'a FitOptions,
    init_params: &'a ModelParameters,
    bounds: PackedBounds,
    cached_etas: std::sync::Mutex<Vec<DVector<f64>>>,
    grad_cache: std::sync::Mutex<Option<GradCache>>,
}

impl FoceiProblem<'_> {
    fn run_inner(&self, x: &[f64]) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let params = unpack_params(x, self.init_params);
        let warm = self.cached_etas.lock().unwrap().clone();
        let warm_ref = if warm.is_empty() {
            None
        } else {
            Some(warm.as_slice())
        };
        let mu_k = compute_mu_k(self.model, &params.theta, self.options.mu_referencing);
        let (etas, h_mats, _, _kappas) = run_inner_loop_warm(
            self.model,
            self.population,
            &params,
            self.options.inner_maxiter,
            self.options.inner_tol,
            warm_ref,
            Some(&mu_k),
            self.options.min_obs_for_convergence_check as usize,
            self.options.inner_restarts,
        );
        *self.cached_etas.lock().unwrap() = etas.clone();
        (etas, h_mats)
    }

    fn ofv_fixed(&self, x: &[f64], etas: &[DVector<f64>], h_mats: &[DMatrix<f64>]) -> f64 {
        let params = unpack_params(x, self.init_params);
        let nll = pop_nll_opts(
            self.model,
            self.population,
            &params,
            etas,
            h_mats,
            &[], // trust_region doesn't support IOV yet; kappas empty
            self.options,
        );
        let raw = 2.0 * nll;
        if raw.is_finite() {
            raw
        } else {
            1e20
        }
    }

    /// Compute per-subject NLL gradients via `subject_nll_pop_grad`, caching the
    /// result so that `hessian()` can reuse it without a second inner-loop solve.
    ///
    /// Three cache states (keyed by `x` equality and sentinel field):
    ///   Full hit:    `c.x == x` and `!c.per_subj_grads.is_empty()` → return everything cached.
    ///   Partial hit: `c.x == x` and `c.per_subj_grads.is_empty()`  → EBEs warm (from `cost()`),
    ///                                                                   run AD pass only.
    ///   Miss:        `c.x != x` or cache is `None`                 → full inner solve + AD.
    fn compute_ad_grads(&self, x: &[f64]) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<f64>>) {
        let maybe_warm: Option<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)> = {
            let cache = self.grad_cache.lock().unwrap();
            if let Some(ref c) = *cache {
                if c.x == x {
                    if !c.per_subj_grads.is_empty() {
                        // Full hit: EBEs and AD gradients both cached.
                        return (c.etas.clone(), c.h_mats.clone(), c.per_subj_grads.clone());
                    }
                    // Partial hit: EBEs ready from cost(), AD not yet done.
                    Some((c.etas.clone(), c.h_mats.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Use warm EBEs on partial hit; run inner solve on miss.
        let (etas, h_mats) = maybe_warm.unwrap_or_else(|| self.run_inner(x));
        let n_subj = self.population.subjects.len();

        let per_subj: Vec<Vec<f64>> = (0..n_subj)
            .into_par_iter()
            .map(|i| {
                subject_nll_pop_grad(
                    x,
                    self.init_params,
                    self.model,
                    self.population,
                    i,
                    &etas[i],
                    &h_mats[i],
                    &[], // IOV not yet supported in trust_region path
                    &self.bounds,
                    self.options,
                )
                .1
            })
            .collect();

        *self.grad_cache.lock().unwrap() = Some(GradCache {
            x: x.to_vec(),
            etas: etas.clone(),
            h_mats: h_mats.clone(),
            per_subj_grads: per_subj.clone(),
        });

        (etas, h_mats, per_subj)
    }
}

// Use Vec<f64> / Vec<Vec<f64>> as the argmin param/gradient/hessian types.
// argmin-math provides trait impls for Vec natively, avoiding nalgebra version conflicts.

impl CostFunction for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        let (etas, h_mats) = self.run_inner(p);
        let ofv = self.ofv_fixed(p, &etas, &h_mats);
        // Pre-warm the gradient cache with EBEs so that a subsequent
        // gradient() call on the same x skips the redundant run_inner().
        // per_subj_grads: vec![] is the sentinel for "EBEs ready, AD pending".
        *self.grad_cache.lock().unwrap() = Some(GradCache {
            x: p.clone(),
            etas,
            h_mats,
            per_subj_grads: vec![],
        });
        Ok(ofv)
    }
}

impl Gradient for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let (_, _, per_subj) = self.compute_ad_grads(p);
        let n = p.len();
        let mut g = vec![0.0_f64; n];
        for gi in &per_subj {
            for k in 0..n {
                g[k] += 2.0 * gi[k];
            }
        }
        Ok(g)
    }
}

impl Hessian for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, p: &Vec<f64>) -> Result<Vec<Vec<f64>>, Error> {
        let (_, _, per_subj) = self.compute_ad_grads(p);
        let n = p.len();
        // BHHH approximation: H ≈ 4 Σ gᵢgᵢᵀ  (factor 4 because OFV = 2*NLL,
        // so grad(OFV) = 2*gᵢ and the outer product scales by 4).
        let mut h = vec![vec![0.0_f64; n]; n];
        for gi in &per_subj {
            for i in 0..n {
                for j in 0..n {
                    h[i][j] += 4.0 * gi[i] * gi[j];
                }
            }
        }
        Ok(h)
    }
}

/// Steihaug truncated-CG trust-region subproblem solver.
///
/// Finds an approximate minimiser of the quadratic model `½ δᵀ H δ + gᵀ δ`
/// subject to `‖δ‖ ≤ trust_radius` (Nocedal & Wright, Algorithm 7.2).
/// `H` must be symmetric; it works best when `H` is positive semi-definite
/// (e.g. the BHHH approximation), but handles zero and negative curvature by
/// terminating at the trust-region boundary.
///
/// Returns the step `δ` satisfying the trust constraint.
pub(crate) fn solve_trust_region_subproblem(
    g: &DVector<f64>,
    h: &DMatrix<f64>,
    trust_radius: f64,
    max_iters: usize,
) -> DVector<f64> {
    let n = g.len();

    let mut p = DVector::zeros(n);
    let mut r = g.clone();
    let mut d = -g.clone();
    let r0_norm = r.norm();

    if r0_norm < 1e-16 {
        return p; // gradient is zero — no step needed
    }

    // N&W Algorithm 7.2 forcing sequence: ε = min(0.5, √‖r₀‖).
    // At 1e-10 the criterion never fires within the small CG budget; this
    // tighter value lets a raised budget benefit from early termination.
    let eps_rel = r0_norm.sqrt().min(0.5);

    for _ in 0..max_iters {
        let hd = h * &d;
        let d_hd = d.dot(&hd);

        if d_hd <= 0.0 {
            // Zero or negative curvature along d: step to the TR boundary.
            return boundary_step(&p, &d, trust_radius);
        }

        let r_sq = r.dot(&r);
        let alpha = r_sq / d_hd;
        let p_new = &p + alpha * &d;

        if p_new.norm() >= trust_radius {
            // Step would exit the trust region: clip to boundary.
            return boundary_step(&p, &d, trust_radius);
        }

        let r_new = &r + alpha * &hd;

        if r_new.norm() < eps_rel * r0_norm {
            return p_new; // residual converged
        }

        let beta = r_new.dot(&r_new) / r_sq;
        d = -&r_new + beta * &d;
        p = p_new;
        r = r_new;
    }

    p
}

/// Find τ ≥ 0 such that ‖p + τd‖ = delta, i.e. the boundary intersection
/// along d from p (which must lie inside the ball).
///
/// Solves: ‖d‖² τ² + 2(p·d) τ + (‖p‖² − Δ²) = 0, taking the positive root.
fn boundary_step(p: &DVector<f64>, d: &DVector<f64>, delta: f64) -> DVector<f64> {
    let d_sq = d.dot(d);
    let pd = p.dot(d);
    let p_sq = p.dot(p);

    if d_sq < 1e-30 {
        // d is negligible — return p clamped to the boundary (or as-is if inside).
        let p_norm = p_sq.sqrt();
        return if p_norm > 1e-30 {
            p * (delta / p_norm)
        } else {
            p.clone()
        };
    }

    // disc = (p·d)² − ‖d‖²(‖p‖² − Δ²) ≥ 0 since p is inside the ball.
    let disc = pd * pd - d_sq * (p_sq - delta * delta);
    let disc_clamped = if disc > 0.0 { disc } else { 0.0 };
    let tau = (-pd + disc_clamped.sqrt()) / d_sq;
    p + tau * d
}

/// Size-adaptive Steihaug CG budget: `ceil(sqrt(n_params)).clamp(5, n_params)`.
/// Avoids the fixed-50 default that wastes CG iterations when n_params ≤ 15.
pub(crate) fn adaptive_steihaug_budget(n_params: usize) -> usize {
    let base = (n_params as f64).sqrt().ceil() as usize;
    base.clamp(5, n_params.max(5))
}

/// The concrete argmin state this optimizer drives: packed `Vec<f64>` parameters,
/// a `Vec<f64>` outer gradient and the dense BHHH `Vec<Vec<f64>>` Hessian.
type TrState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

/// Consecutive iterations without measurable progress (in both the objective and
/// the parameter vector) before the trust region declares convergence.
///
/// argmin's `TrustRegion` has **no** convergence criterion of its own — its
/// `terminate` returns `NotTerminated` unconditionally, so before #1000 the only
/// reachable stop was `MaxItersReached` and *every* run was reported converged,
/// including one that had simply exhausted `outer_maxiter` thousands of OFV units
/// short of the optimum.
///
/// The criterion is no-progress rather than a stationary-gradient test, because
/// the outer gradient here is evaluated at **fixed** EBEs and is not a reliable
/// stationarity measure on an ODE model: on `one_cpt_iv_ode` the gradient norm at
/// the optimum the default optimizer converges to (OFV −387.0737) is 43.2, while
/// a *worse* point 0.73 OFV units away (−386.3451) carries 4.0 — non-monotone in
/// solution quality, so no threshold can separate the two. Objective and step
/// size are the noise-robust measures, and they are what the NLopt outer
/// optimizers already stop on (`outer_ftol` / `outer_xtol`).
///
/// A run of rejected steps is *not* by itself convergence: a trust region shrinks
/// its radius after each rejection, so a handful of consecutive rejections is
/// ordinary mid-descent behaviour. The count is set well above that transient and
/// far below the thousands of frozen iterations a converged run spends waiting
/// for `outer_maxiter`.
const TRUST_REGION_NO_PROGRESS_ITERS: u32 = 20;

/// Relative objective tolerance for the no-progress test, used when
/// `[fit_options] outer_ftol` is unset. Matches the NLopt outer default
/// (see `resolve_outer_ftol`).
const TRUST_REGION_DEFAULT_FTOL: f64 = 1e-6;

/// L2 norm of a packed vector (the outer gradient, a step, or the parameter
/// vector itself); `f64::INFINITY` if any entry is non-finite, so a blown-up
/// value can never read as small.
fn l2_norm(v: &[f64]) -> f64 {
    let norm = v.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm.is_finite() {
        norm
    } else {
        f64::INFINITY
    }
}

/// Whether one iteration made measurable progress: either the objective improved
/// by more than `ftol · (1 + |cost|)`, or the parameter vector moved by more than
/// `xtol · (1 + ‖x‖)`. Both are relative, so neither depends on the scale of the
/// dataset or the parameterization.
fn made_progress(
    prev_cost: f64,
    cost: f64,
    prev_x: &[f64],
    x: &[f64],
    ftol: f64,
    xtol: f64,
) -> bool {
    let improved = prev_cost - cost;
    if improved.is_nan() || improved > ftol * (1.0 + cost.abs()) {
        // NaN counts as progress: it must never be mistaken for a settled fit.
        return true;
    }
    if prev_x.len() != x.len() {
        return true;
    }
    let step = l2_norm(
        &prev_x
            .iter()
            .zip(x)
            .map(|(p, c)| c - p)
            .collect::<Vec<f64>>(),
    );
    let x_norm = l2_norm(x);
    step > xtol * (1.0 + x_norm)
}

/// argmin's `TrustRegion` with the convergence stop it lacks.
///
/// Delegates `init` / `next_iter` to the wrapped solver and adds a `terminate`
/// that stops once the optimizer can no longer make progress at the requested
/// precision — see [`TRUST_REGION_NO_PROGRESS_ITERS`]. The executor checks
/// `terminate` *before* the `MaxItersReached` test, so a run that settles on its
/// final permitted iteration is still reported as converged.
struct TrustRegionWithStop {
    inner: TrustRegion<Steihaug<Vec<f64>, f64>, f64>,
    ftol: f64,
    xtol: f64,
    /// Objective and parameters as of the previous `terminate` call.
    prev: Option<(f64, Vec<f64>)>,
    /// Consecutive iterations with no measurable progress.
    stalled_iters: u32,
    /// Whether the run has ever made progress. A trust region that rejects from
    /// the very first iteration has not converged — it never started.
    ever_progressed: bool,
    /// Verbose diagnostics for the iteration trace.
    verbose: bool,
}

impl TrustRegionWithStop {
    fn new(inner: TrustRegion<Steihaug<Vec<f64>, f64>, f64>, options: &FitOptions) -> Self {
        Self {
            inner,
            ftol: options.outer_ftol.unwrap_or(TRUST_REGION_DEFAULT_FTOL),
            xtol: options.outer_xtol,
            prev: None,
            stalled_iters: 0,
            ever_progressed: false,
            verbose: options.verbose,
        }
    }

    /// Record one optimizer iteration; `true` once the run has settled — no
    /// measurable progress for [`TRUST_REGION_NO_PROGRESS_ITERS`] consecutive
    /// iterations, having made progress at some point first. The second half
    /// matters: a trust region that rejects its very first steps has not
    /// converged, it has failed to start.
    fn note_iteration(&mut self, iter: u64, cost: f64, x: Vec<f64>) -> bool {
        if let Some((prev_cost, prev_x)) = self.prev.take() {
            if made_progress(prev_cost, cost, &prev_x, &x, self.ftol, self.xtol) {
                self.ever_progressed = true;
                self.stalled_iters = 0;
            } else {
                self.stalled_iters += 1;
            }
        }
        if self.verbose {
            eprintln!(
                "  TR iter {:>4}: OFV = {:.6}  (stalled {} / {})",
                iter, cost, self.stalled_iters, TRUST_REGION_NO_PROGRESS_ITERS
            );
        }
        self.prev = Some((cost, x));
        self.ever_progressed && self.stalled_iters >= TRUST_REGION_NO_PROGRESS_ITERS
    }
}

impl<'a> Solver<FoceiProblem<'a>, TrState> for TrustRegionWithStop {
    fn name(&self) -> &str {
        "Trust region"
    }

    fn init(
        &mut self,
        problem: &mut Problem<FoceiProblem<'a>>,
        state: TrState,
    ) -> Result<(TrState, Option<KV>), Error> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<FoceiProblem<'a>>,
        state: TrState,
    ) -> Result<(TrState, Option<KV>), Error> {
        self.inner.next_iter(problem, state)
    }

    fn terminate(&mut self, state: &TrState) -> TerminationStatus {
        let x = match state.get_param() {
            Some(p) => p.clone(),
            None => return TerminationStatus::NotTerminated,
        };
        if self.note_iteration(state.get_iter(), state.get_cost(), x) {
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}

/// Map an argmin termination status onto `(converged, warning)`.
///
/// Before #1000 every `Ok(_)` from the executor was reported as converged,
/// including the `MaxItersReached` stop that argmin returns for a run which
/// simply exhausted `outer_maxiter` — so a fit that had moved 8 000–11 000 OFV
/// units short of the optimum was labelled `Converged: YES` and its standard
/// errors were computed at a non-stationary point.
///
/// Only a solver-side stop counts as convergence: `SolverConverged` (our
/// stationary-gradient criterion) and `TargetCostReached` (a caller-set target
/// cost — unreachable here, since we never set one, but honest if it ever is).
/// Everything else — budget exhaustion, Ctrl-C, timeout, a solver bail-out — is
/// reported as not converged with a warning naming the reason.
fn classify_termination(status: &TerminationStatus, max_iters: u64) -> (bool, Option<String>) {
    match status {
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
        | TerminationStatus::Terminated(TerminationReason::TargetCostReached) => (true, None),
        TerminationStatus::Terminated(TerminationReason::MaxItersReached) => (
            false,
            Some(format!(
                "Trust-region did not converge: reached outer_maxiter ({max_iters}). \
                 The estimates and any standard errors are reported at a non-stationary \
                 point; raise maxiter or improve the starting values."
            )),
        ),
        TerminationStatus::Terminated(reason) => (
            false,
            Some(format!("Trust-region did not converge: {reason}")),
        ),
        // Unreachable: the executor only returns once the state is terminated.
        TerminationStatus::NotTerminated => (
            false,
            Some(
                "Trust-region did not converge: solver stopped without a termination reason"
                    .to_string(),
            ),
        ),
    }
}

pub fn optimize_trust_region(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x0 = pack_params(init_params);
    clamp_to_bounds(&mut x0, &bounds);

    let mut warnings = Vec::new();

    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let problem = FoceiProblem {
        model,
        population,
        options,
        init_params,
        bounds,
        cached_etas: std::sync::Mutex::new(vec![DVector::zeros(n_eta); n_subj]),
        grad_cache: std::sync::Mutex::new(None),
    };

    if options.verbose {
        eprintln!(
            "Starting trust-region optimization ({} parameters)...",
            x0.len()
        );
    }

    let cg_budget = options
        .steihaug_max_iters
        .unwrap_or_else(|| adaptive_steihaug_budget(x0.len()));

    let subproblem = Steihaug::new().with_max_iters(cg_budget as u64);
    let solver = TrustRegionWithStop::new(
        TrustRegion::new(subproblem)
            .with_radius(1.0)
            .expect("trust region radius must be positive")
            .with_max_radius(10.0)
            .expect("trust region max radius must be positive"),
        options,
    );

    let max_iters = options.outer_maxiter as u64;
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(x0.clone()).max_iters(max_iters))
        .run();

    let (converged, mut best_x, final_gradient) = match result {
        Ok(res) => {
            let (converged, warning) =
                classify_termination(res.state().get_termination_status(), max_iters);
            if options.verbose {
                eprintln!(
                    "Trust-region finished: {} iters, {}",
                    res.state().get_iter(),
                    res.state().get_termination_status()
                );
            }
            let vec = res
                .state()
                .get_best_param()
                .cloned()
                .unwrap_or_else(|| x0.clone());
            // Gradient at the returned point, for `FitResult.final_gradient` and
            // so the non-convergence warning can quote how far from stationary
            // the fit stopped. Reuses the executor's problem, and with it the
            // gradient cache: free when the last evaluated point is the one
            // returned, and otherwise (the final trial step was rejected, so the
            // cache holds that rejected point) one extra gradient evaluation per
            // fit — not per iteration.
            let OptimizationResult {
                problem: mut prob, ..
            } = res;
            let grad = prob.gradient(&vec).ok();
            if let Some(mut w) = warning {
                if let Some(g) = grad.as_deref() {
                    w.push_str(&format!(" Final ‖∂OFV/∂x‖ = {:.3e}.", l2_norm(g)));
                }
                warnings.push(w);
            }
            (converged, vec, grad)
        }
        Err(e) => {
            if options.verbose {
                eprintln!("Trust-region stopped: {}", e);
            }
            warnings.push(format!("Trust-region did not converge: {}", e));
            (false, x0.clone(), None)
        }
    };

    clamp_to_bounds(&mut best_x, &compute_bounds(init_params));

    let final_params = unpack_params(&best_x, init_params);
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

    let final_ofv = 2.0
        * pop_nll_opts(
            model,
            population,
            &final_params,
            &final_ehs,
            &final_hms,
            &final_kappas,
            options,
        );

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // A settled run whose objective is not finite has not converged to anything:
    // the trial objective is clamped to a large sentinel when it blows up, so the
    // no-progress rule can otherwise "settle" on that sentinel.
    let converged = converged && final_ofv.is_finite();
    if !converged && warnings.is_empty() {
        warnings.push(format!(
            "Trust-region did not converge: final OFV is not finite ({final_ofv})"
        ));
    }

    let out = crate::estimation::covariance::run_covariance_step(
        &best_x,
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

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: 0,
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
        final_gradient,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        // The exact packed vector this stage's inline covariance step used
        // (`compute_covariance(&best_x, …)` above). The trust-region optimizer
        // works in packed Cholesky space, so `best_x`'s omega block is the exact
        // factor `L`; carrying it lets `run_covariance` reproduce this covariance
        // step bit-for-bit instead of re-decomposing `omega` (#816 follow-up).
        packed_estimate: Some(best_x.clone()),
        mixture_posteriors: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_steihaug_budget() {
        // Typical NLME: 7 params → ceil(sqrt(7))=3, clamped to 5.
        assert_eq!(adaptive_steihaug_budget(7), 5);
        // Medium model: 16 params → ceil(sqrt(16))=4, clamped to 5.
        assert_eq!(adaptive_steihaug_budget(16), 5);
        // Larger model: 25 params → ceil(sqrt(25))=5.
        assert_eq!(adaptive_steihaug_budget(25), 5);
        // Growth visible: 50 params → ceil(sqrt(50))=8.
        assert_eq!(adaptive_steihaug_budget(50), 8);
        // Very large: 100 params → ceil(sqrt(100))=10.
        assert_eq!(adaptive_steihaug_budget(100), 10);
        // Budget never exceeds n_params.
        assert!(adaptive_steihaug_budget(4) <= 4.max(5));
    }

    /// #816 follow-up regression: the trust-region optimizer works in packed
    /// Cholesky space, so its covariance step uses the optimizer's exact factor
    /// `L`. It must carry `packed_estimate` so a standalone `run_covariance`
    /// reproduces the inline covariance bit-for-bit. Fails if the return regresses
    /// to `None` (which would silently drop the standalone step back to the
    /// re-decomposition fallback that diverges on ill-conditioned ω).
    #[test]
    fn trust_region_carries_packed_estimate() {
        use crate::estimation::parameterization::packed_len;
        use crate::io::datareader::read_nonmem_csv;
        use crate::parser::model_parser::parse_model_file;
        use std::path::Path;

        let model = parse_model_file(Path::new("examples/warfarin.ferx"))
            .expect("warfarin model must parse");
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");
        // One outer step, no covariance step: only assert propagation, stay fast.
        let opts = FitOptions {
            outer_maxiter: 1,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let res = optimize_trust_region(&model, &population, &model.default_params, &opts);
        let packed = res
            .packed_estimate
            .expect("trust_region must carry packed_estimate");
        assert_eq!(packed.len(), packed_len(&model.default_params));
    }

    /// Verify the dynamic cache-state contract between `cost()` and `compute_ad_grads()`:
    ///
    /// 1. `cost(x)` writes a partial sentinel (`per_subj_grads.is_empty()`).
    /// 2. `compute_ad_grads(x)` on the same x upgrades to a full entry
    ///    (`!per_subj_grads.is_empty()`).
    /// 3. `compute_ad_grads(x)` on a *different* x (miss path) also produces a
    ///    full entry — the fallback still works without a preceding `cost()`.
    #[test]
    fn test_grad_cache_sentinel_invariant() {
        use crate::estimation::parameterization::{clamp_to_bounds, compute_bounds, pack_params};
        use crate::io::datareader::read_nonmem_csv;
        use crate::parser::model_parser::parse_model_file;
        use argmin::core::CostFunction;
        use std::path::Path;

        let model = parse_model_file(Path::new("examples/warfarin.ferx"))
            .expect("warfarin model must parse");
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");
        let options = FitOptions::default();
        let bounds = compute_bounds(&model.default_params);
        let mut x0 = pack_params(&model.default_params);
        clamp_to_bounds(&mut x0, &bounds);
        let n_subj = population.subjects.len();
        let n_eta = model.n_eta;

        let problem = FoceiProblem {
            model: &model,
            population: &population,
            options: &options,
            init_params: &model.default_params,
            bounds,
            cached_etas: std::sync::Mutex::new(vec![nalgebra::DVector::zeros(n_eta); n_subj]),
            grad_cache: std::sync::Mutex::new(None),
        };

        // 1. Before cost(): cache is None.
        assert!(
            problem.grad_cache.lock().unwrap().is_none(),
            "cache must be empty before any call"
        );

        // 2. After cost(x0): partial sentinel written — x matches, per_subj_grads empty.
        let _ = problem.cost(&x0).expect("cost() must not fail");
        {
            let cache = problem.grad_cache.lock().unwrap();
            let c = cache.as_ref().expect("cost() must populate grad_cache");
            assert_eq!(c.x, x0, "cost() must write the current x into grad_cache");
            assert!(
                c.per_subj_grads.is_empty(),
                "cost() must write the partial sentinel (empty per_subj_grads)"
            );
        }

        // 3. After compute_ad_grads(x0): full entry — same x, per_subj_grads populated.
        let _ = problem.compute_ad_grads(&x0);
        {
            let cache = problem.grad_cache.lock().unwrap();
            let c = cache
                .as_ref()
                .expect("compute_ad_grads() must populate grad_cache");
            assert_eq!(
                c.x, x0,
                "grad_cache x must still match x0 after full AD pass"
            );
            assert!(
                !c.per_subj_grads.is_empty(),
                "compute_ad_grads() must upgrade sentinel to a full entry"
            );
            assert_eq!(
                c.per_subj_grads.len(),
                n_subj,
                "per_subj_grads must have one entry per subject"
            );
        }

        // 4. compute_ad_grads on a different x (miss path) must still produce a full entry.
        let x_other: Vec<f64> = x0.iter().map(|v| v + 0.01).collect();
        let _ = problem.compute_ad_grads(&x_other);
        {
            let cache = problem.grad_cache.lock().unwrap();
            let c = cache.as_ref().expect("miss path must populate grad_cache");
            assert_eq!(c.x, x_other, "miss path must write x_other into grad_cache");
            assert!(
                !c.per_subj_grads.is_empty(),
                "miss path must produce a full entry without a preceding cost() call"
            );
        }
    }

    /// TR subproblem must respect the trust radius for a range of radii.
    #[test]
    fn test_solve_trust_region_subproblem_respects_radius() {
        let g = DVector::from_vec(vec![1.0, -2.0]);
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]); // PD
        for &delta in &[0.1_f64, 0.5, 1.0, 5.0] {
            let step = solve_trust_region_subproblem(&g, &h, delta, 20);
            assert!(
                step.norm() <= delta * (1.0 + 1e-8),
                "‖step‖ = {:.6} > Δ = {} (violation)",
                step.norm(),
                delta
            );
        }
    }

    /// The step must decrease the quadratic model q(δ) = ½ δᵀ H δ + gᵀ δ < 0.
    #[test]
    fn test_solve_trust_region_subproblem_improves_quadratic_model() {
        let g = DVector::from_vec(vec![1.0, -2.0]);
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let delta = solve_trust_region_subproblem(&g, &h, 1.0, 20);
        let q = 0.5 * delta.dot(&(h * &delta)) + g.dot(&delta);
        assert!(q < 0.0, "quadratic model must decrease: q = {:.6}", q);
    }

    /// With a Hessian that has a negative eigenvalue, the step must reach the
    /// TR boundary (Steihaug terminates at the boundary on negative curvature).
    #[test]
    fn test_solve_trust_region_subproblem_negative_curvature() {
        // H = diag(-1, 2) has a negative eigenvalue along e₁.
        // g must point along e₁ so the initial CG direction d = -g = [-1, 0]
        // immediately encounters d·Hd = -1 < 0 and triggers the boundary step.
        let g = DVector::from_vec(vec![1.0, 0.0]);
        let h = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 2.0]);
        let trust_radius = 1.0;
        let step = solve_trust_region_subproblem(&g, &h, trust_radius, 20);
        // Step must reach the boundary, not panic.
        assert!(
            (step.norm() - trust_radius).abs() < 1e-8,
            "negative curvature: ‖step‖ = {:.8} should equal Δ = {}",
            step.norm(),
            trust_radius
        );
    }

    /// `boundary_step` with d ≈ 0: the function must return a point on the sphere
    /// (‖result‖ = delta) rather than panicking or producing a degenerate step.
    /// This edge case is unreachable from the normal Steihaug flow but the guard
    /// (`d_sq < 1e-30`) is present and must be tested independently.
    #[test]
    fn test_boundary_step_near_zero_d() {
        // p is inside the ball; d is effectively zero.
        let p = DVector::from_vec(vec![0.3, 0.4]); // ‖p‖ = 0.5
        let d = DVector::from_vec(vec![0.0, 0.0]);
        let delta = 1.0;
        let result = boundary_step(&p, &d, delta);
        // With d ≈ 0 the function projects p onto the sphere.
        let result_norm = result.norm();
        assert!(
            (result_norm - delta).abs() < 1e-10,
            "boundary_step(d≈0): ‖result‖ = {result_norm:.8} should equal Δ = {delta}"
        );
    }

    #[test]
    fn test_steihaug_budget_option_none_uses_adaptive() {
        let options = FitOptions::default();
        assert!(options.steihaug_max_iters.is_none());
        // Simulate what optimize_trust_region does for n_params = 8.
        let budget = options
            .steihaug_max_iters
            .unwrap_or_else(|| adaptive_steihaug_budget(8));
        assert_eq!(budget, 5); // ceil(sqrt(8))=3, clamped to 5
    }

    #[test]
    fn test_steihaug_budget_option_some_pins_value() {
        let mut options = FitOptions::default();
        options.steihaug_max_iters = Some(20);
        let budget = options
            .steihaug_max_iters
            .unwrap_or_else(|| adaptive_steihaug_budget(8));
        assert_eq!(budget, 20);
    }

    // --- #1000: convergence reporting -----------------------------------------

    /// The whole defect in #1000 is this mapping: argmin returns `Ok(_)` for
    /// *every* normal termination, so treating the `Result` discriminant as the
    /// convergence verdict reported a budget-exhausted run as converged.
    #[test]
    fn classify_termination_only_calls_solver_side_stops_converged() {
        let (conv, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::SolverConverged),
            300,
        );
        assert!(conv);
        assert!(warn.is_none());

        let (conv, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::TargetCostReached),
            300,
        );
        assert!(conv, "a caller-set target cost is a deliberate stop");
        assert!(warn.is_none());

        // The regression: MaxItersReached must not read as convergence, and must
        // name the budget it exhausted.
        let (conv, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::MaxItersReached),
            300,
        );
        assert!(!conv, "budget exhaustion is not convergence");
        let warn = warn.expect("MaxItersReached must warn");
        assert!(warn.contains("reached outer_maxiter (300)"), "{warn}");

        for reason in [
            TerminationReason::Interrupt,
            TerminationReason::Timeout,
            TerminationReason::SolverExit("bail".to_string()),
        ] {
            let (conv, warn) = classify_termination(&TerminationStatus::Terminated(reason), 300);
            assert!(!conv);
            assert!(warn.is_some());
        }

        let (conv, warn) = classify_termination(&TerminationStatus::NotTerminated, 300);
        assert!(!conv);
        assert!(warn.is_some());
    }

    /// The no-progress test that stands in for argmin's missing convergence
    /// criterion. Both limbs are relative, so neither depends on the scale of the
    /// dataset or the parameterization.
    #[test]
    fn made_progress_reads_objective_and_step() {
        let ftol = 1e-6;
        let xtol = 1e-4;
        let x = vec![1.0, 2.0];

        // Objective improvement above / below ftol * (1 + |cost|).
        assert!(made_progress(-100.0, -100.5, &x, &x, ftol, xtol));
        assert!(!made_progress(-100.0, -100.000_001, &x, &x, ftol, xtol));

        // A step larger than xtol * (1 + ||x||) is progress even with a flat
        // objective — the optimizer is still moving.
        let moved = vec![1.1, 2.0];
        assert!(made_progress(-100.0, -100.0, &x, &moved, ftol, xtol));
        let nudged = vec![1.0 + 1e-9, 2.0];
        assert!(!made_progress(-100.0, -100.0, &x, &nudged, ftol, xtol));

        // A rejected trust-region step leaves both untouched: no progress.
        assert!(!made_progress(-100.0, -100.0, &x, &x, ftol, xtol));

        // A cost *increase* is not progress by the objective limb (the step limb
        // still decides), and a NaN objective never reads as a settled fit.
        assert!(!made_progress(-100.0, -99.0, &x, &x, ftol, xtol));
        assert!(made_progress(f64::NAN, -100.0, &x, &x, ftol, xtol));
    }

    /// #1000 regression: a fit given a budget it cannot finish in must report
    /// `converged = false` and say why. Two outer iterations on warfarin is
    /// nowhere near the point it settles at (iteration 59), so this is
    /// deterministic without being a convergence test.
    #[test]
    fn trust_region_maxiter_exhaustion_is_not_converged() {
        use crate::io::datareader::read_nonmem_csv;
        use crate::parser::model_parser::parse_model_file;
        use std::path::Path;

        let model = parse_model_file(Path::new("examples/warfarin.ferx"))
            .expect("warfarin model must parse");
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");
        let opts = FitOptions {
            outer_maxiter: 2,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let res = optimize_trust_region(&model, &population, &model.default_params, &opts);
        assert!(
            !res.converged,
            "a run that exhausted outer_maxiter must not report converged"
        );
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("reached outer_maxiter (2)")),
            "missing the budget-exhaustion warning: {:?}",
            res.warnings
        );
        // The same run must now carry the gradient it stopped at, which the
        // trust-region path previously left unset.
        assert!(
            res.final_gradient.is_some(),
            "trust_region must report final_gradient"
        );
    }

    fn stall_tracker() -> TrustRegionWithStop {
        TrustRegionWithStop::new(
            TrustRegion::new(Steihaug::new()).with_radius(1.0).unwrap(),
            &FitOptions {
                verbose: false,
                ..FitOptions::default()
            },
        )
    }

    /// The other half of #1000: with argmin's `TrustRegion` never terminating on
    /// its own, a run only earns `Converged: YES` from this no-progress rule, so
    /// it has to actually fire. Pins the exact iteration it fires on.
    #[test]
    fn note_iteration_declares_convergence_after_a_settled_run() {
        let mut tracker = stall_tracker();
        let x = vec![0.5, -0.25];

        // Descending: each step improves the objective well above ftol.
        for k in 0..5 {
            let cost = 100.0 - 10.0 * k as f64;
            assert!(
                !tracker.note_iteration(k, cost, x.clone()),
                "a descending run must not read as converged"
            );
        }

        // Frozen: the trust region now rejects every step (identical cost and
        // parameters), which is what a settled run looks like.
        // The first frozen call still registers the drop from the descent, so a
        // settled window of N stalled iterations takes N + 1 calls.
        let settled = 50.0;
        for k in 0..TRUST_REGION_NO_PROGRESS_ITERS {
            assert!(
                !tracker.note_iteration(10 + k as u64, settled, x.clone()),
                "must not declare convergence at only {} stalled iterations",
                k + 1
            );
        }
        assert!(
            tracker.note_iteration(100, settled, x.clone()),
            "must converge on the {TRUST_REGION_NO_PROGRESS_ITERS}th stalled iteration"
        );
    }

    /// A trust region that rejects from the very first iteration never started —
    /// reporting that as convergence would recreate #1000 in a different guise.
    #[test]
    fn note_iteration_never_converges_without_progress_first() {
        let mut tracker = stall_tracker();
        let x = vec![1.0, 1.0];
        for k in 0..(TRUST_REGION_NO_PROGRESS_ITERS * 3) {
            assert!(
                !tracker.note_iteration(k as u64, 42.0, x.clone()),
                "a run that never progressed must never read as converged"
            );
        }
    }

    /// A single improving step resets the stall count: a fit that pauses, moves,
    /// then pauses again must serve the full stall window from the later pause.
    #[test]
    fn note_iteration_resets_the_stall_count_on_progress() {
        let mut tracker = stall_tracker();
        let x = vec![2.0];
        tracker.note_iteration(0, 100.0, x.clone());
        tracker.note_iteration(1, 90.0, x.clone()); // progress
        for k in 0..(TRUST_REGION_NO_PROGRESS_ITERS - 1) {
            assert!(!tracker.note_iteration(2 + k as u64, 90.0, x.clone()));
        }
        // One more improving step, then the window restarts from zero.
        assert!(!tracker.note_iteration(50, 80.0, x.clone()));
        for k in 0..(TRUST_REGION_NO_PROGRESS_ITERS - 1) {
            assert!(
                !tracker.note_iteration(51 + k as u64, 80.0, x.clone()),
                "the stall window must restart after progress"
            );
        }
        assert!(tracker.note_iteration(100, 80.0, x));
    }
}
