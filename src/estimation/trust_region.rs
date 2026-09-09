use argmin::core::{
    CostFunction, Error, Executor, Gradient, Hessian, IterState, OptimizationResult, Problem,
    Solver, State, TerminationReason, TerminationStatus, KV,
};
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::gauss_newton::subject_nll_pop_grad_with_cache;
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{
    ofv_is_valid, pop_nll_opts, resolve_outer_ftol, OuterResult,
};
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
    /// Raw per-subject score `gᵢ` — the BHHH Hessian's outer-product input.
    per_subj_grads: Vec<Vec<f64>>,
    /// Per-subject FOCEI EBE-response correction `tᵢ` (`∂log|H̃|/∂η · dη̂/dθ`),
    /// all-zero under FOCE / additive error. Kept alongside `gᵢ` rather than
    /// folded into it: the gradient wants `gᵢ + tᵢ`, the BHHH Hessian must stay
    /// on the raw score (see [`Gradient`] / [`Hessian`] below).
    per_subj_corrections: Vec<Vec<f64>>,
}

struct FoceiProblem<'a> {
    model: &'a CompiledModel,
    population: &'a Population,
    options: &'a FitOptions,
    init_params: &'a ModelParameters,
    bounds: PackedBounds,
    cached_etas: std::sync::Mutex<Vec<DVector<f64>>>,
    grad_cache: std::sync::Mutex<Option<GradCache>>,
    /// Covariate-NN (DCM) regularizer. No-op when both λ are 0. Added to the
    /// optimizer objective (`cost`/`ofv_fixed`), gradient and Hessian, not to
    /// the final reported OFV, which reuses a clean `pop_nll_opts`.
    nn_reg: crate::estimation::nn_reg::NnRegularizer,
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
        // Penalized objective fed to the optimizer (unregularized fits unchanged).
        let raw = 2.0 * nll + self.nn_reg.penalty_value(&params.theta);
        if raw.is_finite() {
            raw
        } else {
            1e20
        }
    }

    /// Compute per-subject NLL gradients via `subject_nll_pop_grad_with_cache`,
    /// caching the result so that `hessian()` can reuse it without a second
    /// inner-loop solve.
    ///
    /// Returns `(etas, h_mats, gᵢ, tᵢ)`: the raw score and the FOCEI EBE-response
    /// correction separately, because they have different consumers — see the
    /// [`Gradient`] and [`Hessian`] impls.
    ///
    /// Three cache states (keyed by `x` equality and sentinel field):
    ///   Full hit:    `c.x == x` and `!c.per_subj_grads.is_empty()` → return everything cached.
    ///   Partial hit: `c.x == x` and `c.per_subj_grads.is_empty()`  → EBEs warm (from `cost()`),
    ///                                                                   run AD pass only.
    ///   Miss:        `c.x != x` or cache is `None`                 → full inner solve + AD.
    #[allow(clippy::type_complexity)]
    fn compute_ad_grads(
        &self,
        x: &[f64],
    ) -> (
        Vec<DVector<f64>>,
        Vec<DMatrix<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
    ) {
        let maybe_warm: Option<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)> = {
            let cache = self.grad_cache.lock().unwrap();
            if let Some(ref c) = *cache {
                if c.x == x {
                    if !c.per_subj_grads.is_empty() {
                        // Full hit: EBEs and AD gradients both cached.
                        return (
                            c.etas.clone(),
                            c.h_mats.clone(),
                            c.per_subj_grads.clone(),
                            c.per_subj_corrections.clone(),
                        );
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
        let n = x.len();

        let per_subj: Vec<(Vec<f64>, Vec<f64>)> = (0..n_subj)
            .into_par_iter()
            .map(|i| {
                let (_, gi, cache) = subject_nll_pop_grad_with_cache(
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
                );
                let ti = cache
                    .as_ref()
                    .map(|c| c.gn_theta_correction.clone())
                    .unwrap_or_else(|| vec![0.0; n]);
                (gi, ti)
            })
            .collect();

        let (grads, corrections): (Vec<Vec<f64>>, Vec<Vec<f64>>) = per_subj.into_iter().unzip();

        *self.grad_cache.lock().unwrap() = Some(GradCache {
            x: x.to_vec(),
            etas: etas.clone(),
            h_mats: h_mats.clone(),
            per_subj_grads: grads.clone(),
            per_subj_corrections: corrections.clone(),
        });

        (etas, h_mats, grads, corrections)
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
            per_subj_corrections: vec![],
        });
        Ok(ofv)
    }
}

impl Gradient for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// `grad(OFV) = 2 · Σᵢ (gᵢ + tᵢ)` — the score plus the FOCEI `log|H̃|`
    /// EBE-response curvature.
    ///
    /// `tᵢ` is not optional polish: under `interaction` the fixed-η̂ analytic
    /// gradient drops it (the #274/#289 Δ — the envelope theorem zeros the inner
    /// objective but not `log|H̃|`), and a gradient optimizer descending on the
    /// truncated score stalls above the minimum (warfarin FOCEI −276.6 instead of
    /// −286.0, `gauss_newton::build_gn_system`). It is identically zero under
    /// FOCE and for additive error, so those paths are unchanged bit-for-bit.
    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let (_, _, per_subj, corrections) = self.compute_ad_grads(p);
        let n = p.len();
        let mut g = vec![0.0_f64; n];
        for (gi, ti) in per_subj.iter().zip(&corrections) {
            for k in 0..n {
                g[k] += 2.0 * (gi[k] + ti[k]);
            }
        }
        // NN penalty gradient. NN weights are identity-packed, so a natural-space
        // weight coordinate is a packed coordinate and the raw-weight gradient
        // maps 1:1 into `g`. Its curvature goes into `hessian()` below.
        let params = unpack_params(p, self.init_params);
        self.nn_reg.add_packed_gradient(&params.theta, &mut g);
        Ok(g)
    }
}

impl Hessian for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, p: &Vec<f64>) -> Result<Vec<Vec<f64>>, Error> {
        let (_, _, per_subj, _) = self.compute_ad_grads(p);
        let n = p.len();
        // BHHH approximation: H ≈ 4 Σ gᵢgᵢᵀ  (factor 4 because OFV = 2*NLL,
        // so grad(OFV) = 2*gᵢ and the outer product scales by 4).
        //
        // Raw score only — the FOCEI `tᵢ` correction the gradient carries is a
        // *curvature* term, not part of the score, and folding it into the outer
        // product corrupts the Hessian (same rule as `build_gn_system`).
        let mut h = vec![vec![0.0_f64; n]; n];
        for gi in &per_subj {
            for i in 0..n {
                for j in 0..n {
                    h[i][j] += 4.0 * gi[i] * gi[j];
                }
            }
        }
        // Covariate-NN penalty curvature: exact `2λ` on the L2 weight diagonal
        // and the Gauss–Newton `2λ·Σ ∂C/∂w ∂C/∂wᵀ` for the curvature term. The
        // gradient carries ∇P, so a quadratic model without this would predict
        // zero curvature along every NN-weight direction under a large λ, ρ
        // would read poor on each step and the radius would shrink for no
        // reason. Both pieces are PSD, so the BHHH model stays PSD.
        let params = unpack_params(p, self.init_params);
        self.nn_reg
            .add_packed_hessian(&params.theta, &mut |i, j, v| h[i][j] += v);
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
///
/// Because progress is an **or** (`Δf > ftol` *or* `‖Δx‖ > xtol`) and a rejected
/// trust-region step moves neither the parameters nor the objective, 20
/// consecutive no-progress iterations means 20 consecutive *rejections* — the
/// radius has shrunk by ≈`0.25²⁰ ≈ 1e-12` from the last accepted step. That is
/// what the count actually tests, and it is why it does not need re-tuning per
/// model.
///
/// Reaching it costs at least `TRUST_REGION_NO_PROGRESS_ITERS + 1` outer
/// iterations, so a fit given a smaller `maxiter` can never report converged.
/// [`classify_termination`] says so explicitly rather than blaming the fit.
const TRUST_REGION_NO_PROGRESS_ITERS: u32 = 20;

/// L2 norm of a packed vector (the outer gradient, a step, or the parameter
/// vector itself); `f64::INFINITY` if any entry is non-finite, so a blown-up
/// value can never read as small.
fn l2_norm(v: &[f64]) -> f64 {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
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
    /// `model` only feeds [`resolve_outer_ftol`], which auto-selects `1e-8` for a
    /// pure non-Gaussian (TTE / categorical) non-ODE objective and `1e-6`
    /// otherwise. Hardcoding `1e-6` here would leave `trust_region` stopping 100×
    /// looser than every other optimizer on exactly the near-flat frailty-ω²
    /// ridge #469 tightened it for.
    fn new(
        inner: TrustRegion<Steihaug<Vec<f64>, f64>, f64>,
        model: &CompiledModel,
        options: &FitOptions,
    ) -> Self {
        Self {
            inner,
            ftol: resolve_outer_ftol(
                model.has_non_gaussian(),
                model.is_ode_based(),
                options.outer_ftol,
            ),
            xtol: options.outer_xtol,
            prev: None,
            stalled_iters: 0,
            ever_progressed: false,
            verbose: options.verbose,
        }
    }

    /// `true` once the run has stalled for a full window **without ever having
    /// made progress** — a trust region that rejects from its very first step has
    /// not converged, it failed to start. Distinct from the converged stop so the
    /// fit can bail out here instead of paying out the whole `maxiter` budget on
    /// a frozen state, and so the warning can name the real cause (the starting
    /// values) rather than telling the user to raise a budget that is not the
    /// problem.
    fn stalled_without_ever_progressing(&self) -> bool {
        !self.ever_progressed && self.stalled_iters >= TRUST_REGION_NO_PROGRESS_ITERS
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
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if self.stalled_without_ever_progressing() {
            return TerminationStatus::Terminated(TerminationReason::SolverExit(
                TRUST_REGION_NO_INITIAL_PROGRESS.to_string(),
            ));
        }
        TerminationStatus::NotTerminated
    }
}

/// `SolverExit` reason for a run that rejected every step from iteration 0.
/// Matched by [`classify_termination`] so the warning names the starting values
/// instead of the iteration budget.
const TRUST_REGION_NO_INITIAL_PROGRESS: &str = "made no progress from the starting values";

/// Map an argmin termination status onto `(converged, warning)`.
///
/// Before #1000 every `Ok(_)` from the executor was reported as converged,
/// including the `MaxItersReached` stop that argmin returns for a run which
/// simply exhausted `outer_maxiter` — so a fit that had moved 8 000–11 000 OFV
/// units short of the optimum was labelled `Converged: YES` and its standard
/// errors were computed at a non-stationary point.
///
/// Only a solver-side stop counts as convergence: `SolverConverged` (the
/// no-progress criterion of [`TRUST_REGION_NO_PROGRESS_ITERS`] — *not* a
/// gradient-stationarity test, see that constant for why) and `TargetCostReached`
/// (a caller-set target cost — unreachable here, since we never set one, but
/// honest if it ever is). Everything else — budget exhaustion, Ctrl-C, timeout, a
/// solver bail-out — is reported as not converged with a warning naming the
/// reason.
///
/// The `maxiter` message names the `[fit_options]` key (`maxiter`), not the
/// internal `FitOptions` field, and calls out the case where the budget was too
/// small for the stopping rule to fire *at all* rather than blaming the fit.
fn classify_termination(status: &TerminationStatus, max_iters: u64) -> (bool, Option<String>) {
    match status {
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
        | TerminationStatus::Terminated(TerminationReason::TargetCostReached) => (true, None),
        TerminationStatus::Terminated(TerminationReason::MaxItersReached) => {
            let min_for_convergence = u64::from(TRUST_REGION_NO_PROGRESS_ITERS) + 1;
            let remedy = if max_iters < min_for_convergence {
                format!(
                    "settling cannot even be demonstrated below maxiter = {min_for_convergence} \
                     (the stopping rule needs {TRUST_REGION_NO_PROGRESS_ITERS} consecutive \
                     rejected steps), so raise maxiter above that before reading anything into \
                     this verdict."
                )
            } else {
                "raise maxiter or improve the starting values.".to_string()
            };
            (
                false,
                Some(format!(
                    "Trust-region did not converge: reached the iteration budget \
                     (maxiter = {max_iters}). The estimates and any standard errors are \
                     reported at a non-stationary point; {remedy}"
                )),
            )
        }
        TerminationStatus::Terminated(TerminationReason::SolverExit(reason))
            if reason == TRUST_REGION_NO_INITIAL_PROGRESS =>
        {
            (
                false,
                Some(format!(
                    "Trust-region did not converge: {reason} — every step was rejected from the \
                     first iteration, so the fit never left them. Check the starting values \
                     (`inits_from_nca` helps) and that the model is identifiable on this data."
                )),
            )
        }
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

    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let nn_reg_active = nn_reg.is_active();
    let problem = FoceiProblem {
        model,
        population,
        options,
        init_params,
        bounds,
        cached_etas: std::sync::Mutex::new(vec![DVector::zeros(n_eta); n_subj]),
        grad_cache: std::sync::Mutex::new(None),
        nn_reg,
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
        model,
        options,
    );

    let max_iters = options.outer_maxiter as u64;
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(x0.clone()).max_iters(max_iters))
        .run();

    let (converged, best_x, n_iterations, final_gradient) = match result {
        Ok(res) => {
            let (converged, warning) =
                classify_termination(res.state().get_termination_status(), max_iters);
            let n_iterations = res.state().get_iter() as usize;
            if options.verbose {
                eprintln!(
                    "Trust-region finished: {} iters, {}",
                    n_iterations,
                    res.state().get_termination_status()
                );
            }
            let mut vec = res
                .state()
                .get_best_param()
                .cloned()
                .unwrap_or_else(|| x0.clone());
            // Clamp *before* the gradient: the trust region is unconstrained and
            // can walk a theta past its bound, and every downstream consumer
            // (`unpack_params`, the covariance step, `packed_estimate`) sees the
            // clamped vector. Taking the gradient at the raw point would report a
            // ‖∂OFV/∂x‖ that belongs to a different parameter vector than the
            // estimates it is printed next to.
            clamp_to_bounds(&mut vec, &compute_bounds(init_params));
            // Gradient at the returned point, for `FitResult.final_gradient` and
            // so the non-convergence warning can quote how far from stationary
            // the fit stopped. Reuses the executor's problem, and with it the
            // gradient cache — but note that on the *converged* path the run
            // stopped precisely because its last steps were rejected, so `cost()`
            // has already overwritten the cache with the rejected trial point and
            // this misses. Budget one extra gradient evaluation per fit; it is
            // per-fit, not per-iteration.
            let OptimizationResult {
                problem: mut prob, ..
            } = res;
            let grad = prob.gradient(&vec).ok();
            if let Some(mut w) = warning {
                if let Some(g) = grad.as_deref() {
                    // Under covariate-NN regularization this gradient is of the
                    // penalized objective the optimizer minimised, not of the
                    // reported OFV — label it as such.
                    let label = if nn_reg_active {
                        "∂(OFV + NN penalty)/∂x"
                    } else {
                        "∂OFV/∂x"
                    };
                    w.push_str(&format!(" Final ‖{label}‖ = {:.3e}.", l2_norm(g)));
                }
                warnings.push(w);
            }
            (converged, vec, n_iterations, grad)
        }
        Err(e) => {
            if options.verbose {
                eprintln!("Trust-region stopped: {}", e);
            }
            warnings.push(format!("Trust-region did not converge: {}", e));
            (false, x0.clone(), 0, None)
        }
    };

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

    // A settled run whose objective is not a real population OFV has not
    // converged to anything: `ofv_fixed` clamps a blown-up objective to a ~1e20
    // sentinel, and the no-progress rule can settle on that sentinel or on any
    // other diverged value. `is_finite()` is *not* enough — the sentinel is
    // finite — so this uses the same validity cutoff the multi-start ranking
    // applies (`DIVERGENCE_OFV`).
    let converged = converged && ofv_is_valid(final_ofv);
    if !converged && warnings.is_empty() {
        warnings.push(format!(
            "Trust-region did not converge: the final OFV ({final_ofv:.4e}) is not a valid \
             population objective — the fit diverged."
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
        // Outer trust-region iterations actually run. This is the number the
        // convergence verdict is about — `io/output.rs` prints it right next to
        // the "reached the iteration budget (maxiter = N)" warning, so a
        // hardcoded 0 contradicted the message beside it.
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
        left_init: None,
        mixture_posteriors: None,
        vi: None,
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
            nn_reg: crate::estimation::nn_reg::NnRegularizer::build(&model, &population, &options),
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
        // Names the `[fit_options]` key the user can actually set — `maxiter`,
        // not the internal `outer_maxiter` field.
        assert!(warn.contains("maxiter = 300"), "{warn}");
        assert!(!warn.contains("outer_maxiter"), "{warn}");
        assert!(warn.contains("raise maxiter"), "{warn}");

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

    /// Reaching the no-progress stop costs at least
    /// `TRUST_REGION_NO_PROGRESS_ITERS + 1` outer iterations, so a smaller budget
    /// makes `Converged: NO` structural rather than a verdict on the fit. The
    /// warning has to say so instead of telling the user to "improve the starting
    /// values" for a run that was never allowed to demonstrate settling.
    #[test]
    fn classify_termination_flags_a_budget_too_small_to_ever_converge() {
        let floor = u64::from(TRUST_REGION_NO_PROGRESS_ITERS) + 1;

        let (_, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::MaxItersReached),
            floor - 1,
        );
        let warn = warn.expect("MaxItersReached must warn");
        assert!(
            warn.contains(&format!(
                "cannot even be demonstrated below maxiter = {floor}"
            )),
            "{warn}"
        );
        assert!(!warn.contains("improve the starting values"), "{warn}");

        // At the floor and above it is an ordinary budget verdict again.
        let (_, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::MaxItersReached),
            floor,
        );
        let warn = warn.expect("MaxItersReached must warn");
        assert!(warn.contains("improve the starting values"), "{warn}");
        assert!(!warn.contains("cannot even be demonstrated"), "{warn}");
    }

    /// A run that rejected every step from iteration 0 gets its own reason, so
    /// the warning points at the starting values rather than at a budget that was
    /// never the constraint.
    #[test]
    fn classify_termination_names_the_starting_values_when_nothing_ever_moved() {
        let (conv, warn) = classify_termination(
            &TerminationStatus::Terminated(TerminationReason::SolverExit(
                TRUST_REGION_NO_INITIAL_PROGRESS.to_string(),
            )),
            500,
        );
        assert!(!conv, "a run that never started has not converged");
        let warn = warn.expect("the no-initial-progress exit must warn");
        assert!(warn.contains(TRUST_REGION_NO_INITIAL_PROGRESS), "{warn}");
        // Not just argmin's `Display` echo of the exit string — the arm has to
        // add the remedy, which is what distinguishes it from the generic
        // solver-bail-out branch.
        assert!(warn.contains("`inits_from_nca` helps"), "{warn}");
        assert!(warn.contains("every step was rejected"), "{warn}");
        assert!(
            !warn.contains("maxiter = 500"),
            "the budget is not the cause here: {warn}"
        );
    }

    /// Analytic-gradient ↔ central-FD parity for the vector the trust region
    /// descends on.
    ///
    /// `Gradient::gradient` must equal `d/dx` of `CostFunction::cost` — the OFV
    /// *after* the inner loop has re-solved the EBEs at `x`, i.e. the marginal.
    /// Under `interaction` the fixed-η̂ analytic score `2·Σ gᵢ` alone does **not**:
    /// it drops the `log|H̃|` EBE-response term `tᵢ` (#274/#289), and a gradient
    /// optimizer descending on it stalls above the minimum. Dropping `tᵢ` here
    /// fails this test by orders of magnitude on the σ and Ω coordinates.
    ///
    /// Each evaluation gets a fresh `FoceiProblem` so `cost` is a pure function of
    /// `x` (no warm-started EBE carry-over between FD probes).
    #[test]
    fn gradient_matches_central_fd_of_the_marginal_cost() {
        use crate::io::datareader::read_nonmem_csv;
        use std::path::Path;

        let model = warfarin_model();
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");

        for interaction in [false, true] {
            let options = FitOptions {
                interaction,
                // The FD baseline is only as good as the EBE fixpoint under it.
                inner_tol: 1e-10,
                inner_maxiter: 200,
                verbose: false,
                ..FitOptions::default()
            };
            let init = &model.default_params;
            let bounds = compute_bounds(init);
            let mut x = pack_params(init);
            clamp_to_bounds(&mut x, &bounds);

            let fresh = || FoceiProblem {
                model: &model,
                population: &population,
                options: &options,
                init_params: init,
                bounds: compute_bounds(init),
                cached_etas: std::sync::Mutex::new(vec![
                    DVector::zeros(model.n_eta);
                    population.subjects.len()
                ]),
                grad_cache: std::sync::Mutex::new(None),
                nn_reg: crate::estimation::nn_reg::NnRegularizer::build(
                    &model,
                    &population,
                    &options,
                ),
            };

            let analytic = fresh().gradient(&x).expect("gradient must evaluate");

            let h = 1e-5;
            for k in 0..x.len() {
                let mut up = x.clone();
                up[k] += h;
                let mut dn = x.clone();
                dn[k] -= h;
                let fd = (fresh().cost(&up).unwrap() - fresh().cost(&dn).unwrap()) / (2.0 * h);
                let scale = fd.abs().max(analytic[k].abs()).max(1.0);
                assert!(
                    (analytic[k] - fd).abs() / scale < 5e-3,
                    "interaction={interaction} coord {k}: analytic {} vs central FD {}",
                    analytic[k],
                    fd
                );
            }
        }
    }

    /// The **assembled** packed gradient — likelihood part plus the covariate-NN
    /// penalty spliced in at the NN-weight coordinates — against central FD of
    /// the penalized `cost` at λ > 0, on the DCM fixture. The per-kernel FD
    /// checks in `nn::` pin each penalty against its own value; this is the seam
    /// they cannot see: that the penalty gradient lands at the right *packed*
    /// index and in the right space (identity-packed weights, no scale factor),
    /// as CLAUDE.md asks of "the provider that assembles them". The warfarin
    /// test above builds its regularizer from `FitOptions::default()` (λ = 0),
    /// so without this the new arm was exercised only as a no-op.
    ///
    /// Also pins the Hessian's penalty block: the exact `2λ` L2 diagonal and the
    /// Gauss–Newton curvature term must match central FD of the assembled
    /// gradient along the NN-weight coordinates (the BHHH likelihood part is an
    /// approximation, so only the *difference* between λ > 0 and λ = 0 is
    /// compared, which isolates the penalty's contribution).
    #[cfg(feature = "nn")]
    #[test]
    fn assembled_gradient_and_hessian_match_fd_of_penalized_cost_on_dcm() {
        use crate::io::datareader::read_nonmem_csv;
        use crate::nn::CovariateMapper;
        use crate::parser::model_parser::parse_full_model;
        use std::path::Path;

        let parsed = parse_full_model(include_str!(
            "../../tests/fixtures/two_cpt_dcm_regularized.ferx"
        ))
        .expect("DCM fixture parses");
        let model = parsed.model;
        let population = read_nonmem_csv(
            Path::new("data/two_cpt_oral_cov.csv"),
            Some(&["WT", "CRCL"]),
            None,
        )
        .expect("dataset loads");

        let mk_options = |l2: f64, smooth: f64| FitOptions {
            inner_tol: 1e-10,
            inner_maxiter: 200,
            verbose: false,
            nn_l2_lambda: l2,
            nn_smooth_lambda: smooth,
            ..parsed.fit_options.clone()
        };
        let options = mk_options(3e-2, 2e-1);
        let init = &model.default_params;
        let bounds = compute_bounds(init);
        let mut x = pack_params(init);
        clamp_to_bounds(&mut x, &bounds);

        fn fresh<'a>(
            model: &'a CompiledModel,
            population: &'a Population,
            init: &'a ModelParameters,
            options: &'a FitOptions,
        ) -> FoceiProblem<'a> {
            FoceiProblem {
                model,
                population,
                options,
                init_params: init,
                bounds: compute_bounds(init),
                cached_etas: std::sync::Mutex::new(vec![
                    DVector::zeros(model.n_eta);
                    population.subjects.len()
                ]),
                grad_cache: std::sync::Mutex::new(None),
                nn_reg: crate::estimation::nn_reg::NnRegularizer::build(model, population, options),
            }
        }
        assert!(fresh(&model, &population, init, &options)
            .nn_reg
            .is_active());

        let nn = &model.covariate_nns[0];
        let (w_lo, w_hi) = (nn.weights_offset, nn.weights_offset + nn.mapper.n_weights());
        let zero = mk_options(0.0, 0.0);

        // Gradient seam: [∇cost(λ) − ∇cost(0)] against central FD of
        // [cost(λ) − cost(0)], over *every* packed coordinate. At a fixed `x` the
        // likelihood part is identical in both, so the difference is exactly the
        // penalty as the optimizer sees it in packed space — which is the seam
        // under test (index placement, no stray scale factor), free of the
        // EBE-fixpoint FD noise the DCM likelihood carries. The likelihood part
        // of the assembled gradient is pinned by the warfarin test above.
        let grad_at = |o: &FitOptions, p: &Vec<f64>| -> Vec<f64> {
            fresh(&model, &population, init, o)
                .gradient(p)
                .expect("gradient must evaluate")
        };
        let pen_cost_at = |p: &Vec<f64>| -> f64 {
            fresh(&model, &population, init, &options).cost(p).unwrap()
                - fresh(&model, &population, init, &zero).cost(p).unwrap()
        };
        let (gl, g0) = (grad_at(&options, &x), grad_at(&zero, &x));
        let pen_grad: Vec<f64> = gl.iter().zip(&g0).map(|(a, b)| a - b).collect();
        assert!(
            pen_grad[w_lo..w_hi].iter().any(|g| g.abs() > 1e-6),
            "penalty gradient must be live on the NN block"
        );
        let h = 1e-6;
        for k in 0..x.len() {
            let mut up = x.clone();
            up[k] += h;
            let mut dn = x.clone();
            dn[k] -= h;
            let fd = (pen_cost_at(&up) - pen_cost_at(&dn)) / (2.0 * h);
            // The likelihood cancels exactly in value but its rayon reduction
            // order is not bit-stable across evals; ~1e-12 of noise on a cost of
            // O(1e3) divided by 2h is ~1e-6 absolute, so the comparison floors the
            // scale at 1e-3 and asks for 5e-3 relative — a misplaced index or a
            // stray scale factor still shows up as an O(1) mismatch.
            let scale = fd.abs().max(pen_grad[k].abs()).max(1e-3);
            assert!(
                (pen_grad[k] - fd).abs() / scale < 5e-3,
                "coord {k} (nn weight: {}): penalty grad {} vs central FD {}",
                (w_lo..w_hi).contains(&k),
                pen_grad[k],
                fd
            );
            if !(w_lo..w_hi).contains(&k) {
                assert_eq!(pen_grad[k], 0.0, "penalty gradient leaked to coord {k}");
            }
        }

        // Hessian seam: H(λ) − H(0) must be exactly the regularizer's own packed
        // Hessian (index placement into the `Vec<Vec<f64>>` the trust region
        // consumes), confined to the NN block. The kernel itself — Gauss–Newton
        // for the curvature term, exact for L2 — is pinned against FD of the
        // penalty gradient in `nn::regularization_tests`, where it is cheap;
        // here the point is that `hessian()` carries it at all, at the right
        // indices, and adds nothing anywhere else.
        let h_pen: Vec<Vec<f64>> = {
            let hl = fresh(&model, &population, init, &options)
                .hessian(&x)
                .unwrap();
            let h0 = fresh(&model, &population, init, &zero).hessian(&x).unwrap();
            hl.iter()
                .zip(&h0)
                .map(|(a, b)| a.iter().zip(b).map(|(p, q)| p - q).collect())
                .collect()
        };
        let mut expected = vec![vec![0.0; x.len()]; x.len()];
        let theta = unpack_params(&x, init).theta;
        crate::estimation::nn_reg::NnRegularizer::build(&model, &population, &options)
            .add_packed_hessian(&theta, &mut |i, j, v| expected[i][j] += v);
        assert!(
            (w_lo..w_hi).any(|k| expected[k][k] > 0.0),
            "penalty Hessian must be live on the NN block"
        );
        for k in 0..x.len() {
            for j in 0..x.len() {
                if !((w_lo..w_hi).contains(&k) && (w_lo..w_hi).contains(&j)) {
                    assert_eq!(h_pen[k][j], 0.0, "penalty Hessian leaked to ({k}, {j})");
                }
                assert!(
                    (h_pen[k][j] - expected[k][j]).abs() <= 1e-9 * (1.0 + expected[k][j].abs()),
                    "penalty Hessian ({k}, {j}): hessian() diff {} vs regularizer {}",
                    h_pen[k][j],
                    expected[k][j]
                );
            }
        }
        // …and with the curvature term off, the L2 block is exact: `2λ` on weight
        // entries, `0` on biases, nothing off-diagonal.
        let l2_only = mk_options(3e-2, 0.0);
        let hl2: Vec<Vec<f64>> = {
            let hl = fresh(&model, &population, init, &l2_only)
                .hessian(&x)
                .unwrap();
            let h0 = fresh(&model, &population, init, &zero).hessian(&x).unwrap();
            hl.iter()
                .zip(&h0)
                .map(|(a, b)| a.iter().zip(b).map(|(p, q)| p - q).collect())
                .collect()
        };
        let weight_idx: std::collections::HashSet<usize> = nn
            .mapper
            .mlp()
            .weight_param_indices()
            .into_iter()
            .map(|i| w_lo + i)
            .collect();
        for k in 0..x.len() {
            for j in 0..x.len() {
                let expect = if k == j && weight_idx.contains(&k) {
                    2.0 * 3e-2
                } else {
                    0.0
                };
                // 1e-8 absolute: the BHHH likelihood part cancels between the two
                // `hessian()` calls only up to rayon reduction-order noise (~1e-12
                // on entries of O(1e3)); the L2 term itself is exact.
                assert!(
                    (hl2[k][j] - expect).abs() < 1e-8,
                    "L2 Hessian ({k}, {j}): {} vs {expect}",
                    hl2[k][j]
                );
            }
        }
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
            res.warnings.iter().any(|w| w.contains("maxiter = 2")),
            "missing the budget-exhaustion warning: {:?}",
            res.warnings
        );
        // The same run must now carry the gradient it stopped at, which the
        // trust-region path previously left unset.
        assert!(
            res.final_gradient.is_some(),
            "trust_region must report final_gradient"
        );
        // ...and the iteration count it stopped at, which was hardcoded to 0
        // right next to the warning that names the budget.
        assert_eq!(
            res.n_iterations, 2,
            "the reported iteration count must be the budget it exhausted"
        );
    }

    fn warfarin_model() -> crate::types::CompiledModel {
        crate::parser::model_parser::parse_model_file(std::path::Path::new(
            "examples/warfarin.ferx",
        ))
        .expect("warfarin model must parse")
    }

    fn stall_tracker() -> TrustRegionWithStop {
        TrustRegionWithStop::new(
            TrustRegion::new(Steihaug::new()).with_radius(1.0).unwrap(),
            &warfarin_model(),
            &FitOptions {
                verbose: false,
                ..FitOptions::default()
            },
        )
    }

    /// The no-progress test must run on the same objective tolerance every other
    /// outer optimizer resolves, not a hardcoded `1e-6`: `resolve_outer_ftol`
    /// tightens to `1e-8` for a pure non-Gaussian (TTE) non-ODE objective, and
    /// that tightening is the #469 fix for the near-flat frailty-ω² ridge.
    /// Hardcoding would have left `trust_region` stopping 100× looser than
    /// `bobyqa` on exactly that model, and reporting `Converged: YES` for it.
    #[test]
    fn stall_tolerance_follows_resolve_outer_ftol() {
        let tracker = |model: &crate::types::CompiledModel, override_ftol| {
            TrustRegionWithStop::new(
                TrustRegion::new(Steihaug::new()).with_radius(1.0).unwrap(),
                model,
                &FitOptions {
                    outer_ftol: override_ftol,
                    ..FitOptions::default()
                },
            )
        };

        let gaussian = warfarin_model();
        assert!(!gaussian.has_non_gaussian());
        assert_eq!(tracker(&gaussian, None).ftol, 1e-6);

        // An explicit `[fit_options] outer_ftol` still wins.
        assert_eq!(tracker(&gaussian, Some(1e-10)).ftol, 1e-10);

        // The step limb is the user's `outer_xtol` verbatim.
        assert_eq!(
            tracker(&gaussian, None).xtol,
            FitOptions::default().outer_xtol
        );

        // The branch that matters — a pure-TTE objective gets the #469 `1e-8` —
        // needs a model only the `survival` build can parse.
        #[cfg(feature = "survival")]
        {
            let tte = crate::parser::model_parser::parse_model_file(std::path::Path::new(
                "examples/tte_weibull.ferx",
            ))
            .expect("tte_weibull model must parse");
            assert!(tte.has_non_gaussian() && !tte.is_ode_based());
            assert_eq!(
                tracker(&tte, None).ftol,
                1e-8,
                "a pure-TTE fit must inherit the #469 tightening"
            );
            assert_eq!(tracker(&tte, Some(1e-5)).ftol, 1e-5);
        }
    }

    /// The three verdicts `Solver::terminate` can return, driven through the trait
    /// method itself rather than through the tracker.
    ///
    /// The tracker tests below pin `note_iteration` and
    /// `stalled_without_ever_progressing` in isolation; this pins the wiring —
    /// that a settled run reports `SolverConverged`, a never-started one reports
    /// the [`TRUST_REGION_NO_INITIAL_PROGRESS`] `SolverExit` (the branch that
    /// lets the fit bail out instead of paying out `maxiter`), and everything
    /// before either threshold reports `NotTerminated` so the executor keeps
    /// going.
    #[test]
    fn terminate_maps_the_tracker_onto_argmin_statuses() {
        let state_at = |iter: u64, cost: f64, x: Vec<f64>| -> TrState {
            let mut st: TrState = IterState::new().param(x).cost(cost);
            for _ in 0..iter {
                st.increment_iter();
            }
            st
        };
        let terminated_as = |status: &TerminationStatus| -> Option<String> {
            match status {
                TerminationStatus::Terminated(r) => Some(format!("{r}")),
                TerminationStatus::NotTerminated => None,
            }
        };

        // Settled: descend once, then freeze for a full window.
        let mut settled = stall_tracker();
        let x = vec![0.5, -0.25];
        assert!(matches!(
            settled.terminate(&state_at(0, 100.0, x.clone())),
            TerminationStatus::NotTerminated
        ));
        // Call k = 0 is the drop from 100 to 50 (progress, resetting the window);
        // k = 1..N-1 are stalls, so the window closes on the call after the loop.
        for k in 0..u64::from(TRUST_REGION_NO_PROGRESS_ITERS) {
            assert!(
                matches!(
                    settled.terminate(&state_at(k + 1, 50.0, x.clone())),
                    TerminationStatus::NotTerminated
                ),
                "terminated early at iteration {}",
                k + 1
            );
        }
        let status = settled.terminate(&state_at(99, 50.0, x.clone()));
        assert!(
            matches!(
                status,
                TerminationStatus::Terminated(TerminationReason::SolverConverged)
            ),
            "a settled run must report SolverConverged, got {status}"
        );

        // Never started: frozen from iteration 0, so the *other* exit fires and
        // `classify_termination` turns it into the starting-values warning.
        let mut frozen = stall_tracker();
        let y = vec![1.0, 1.0];
        // Call k = 0 only seeds `prev`, so the window closes one call later.
        for k in 0..u64::from(TRUST_REGION_NO_PROGRESS_ITERS) {
            assert!(
                matches!(
                    frozen.terminate(&state_at(k, 42.0, y.clone())),
                    TerminationStatus::NotTerminated
                ),
                "bailed out after only {k} frozen iterations"
            );
        }
        let status = frozen.terminate(&state_at(99, 42.0, y.clone()));
        let reason = terminated_as(&status).expect("a fully frozen run must terminate");
        assert!(
            reason.contains(TRUST_REGION_NO_INITIAL_PROGRESS),
            "expected the no-initial-progress exit, got {reason}"
        );
        // ...and never `SolverConverged`, which would recreate #1000.
        assert!(!matches!(
            status,
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        ));
        let (converged, warn) = classify_termination(&status, 500);
        assert!(!converged);
        assert!(warn.expect("must warn").contains("starting values"));

        // A state with no parameter vector cannot be judged; the executor keeps going.
        let mut empty = stall_tracker();
        let bare: TrState = IterState::new();
        assert!(matches!(
            empty.terminate(&bare),
            TerminationStatus::NotTerminated
        ));
    }

    /// A run that rejects every step from iteration 0 must bail out on its own
    /// rather than paying out the whole `maxiter` budget on a frozen state. The
    /// converged verdict stays false either way (see
    /// `note_iteration_never_converges_without_progress_first`); this pins the
    /// *early exit* that keeps it from costing 500 inner-loop solves.
    #[test]
    fn stalled_without_ever_progressing_bails_out_of_a_frozen_run() {
        let mut tracker = stall_tracker();
        let x = vec![1.0, 1.0];

        for k in 0..TRUST_REGION_NO_PROGRESS_ITERS {
            assert!(!tracker.note_iteration(k as u64, 42.0, x.clone()));
            assert!(
                !tracker.stalled_without_ever_progressing(),
                "bailed out after only {k} frozen iterations"
            );
        }
        // Call k = 0 only seeded `prev`, so the window closes one call later.
        assert!(!tracker.note_iteration(
            u64::from(TRUST_REGION_NO_PROGRESS_ITERS),
            42.0,
            x.clone()
        ));
        assert!(
            tracker.stalled_without_ever_progressing(),
            "a fully frozen run must bail out instead of burning the budget"
        );

        // A run that did move is never diverted onto this exit, however long it
        // then sits still — that is the converged stop's business.
        let mut moving = stall_tracker();
        moving.note_iteration(0, 100.0, vec![1.0]);
        moving.note_iteration(1, 90.0, vec![1.0]);
        for k in 0..(TRUST_REGION_NO_PROGRESS_ITERS * 2) {
            moving.note_iteration(2 + k as u64, 90.0, vec![1.0]);
            assert!(
                !moving.stalled_without_ever_progressing(),
                "a run that progressed must not take the never-started exit"
            );
        }
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
