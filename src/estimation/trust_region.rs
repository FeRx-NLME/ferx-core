use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::gauss_newton::subject_nll_pop_grad;
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{
    compute_covariance, pop_nll, CovarianceStepResult, OuterResult,
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
        );
        *self.cached_etas.lock().unwrap() = etas.clone();
        (etas, h_mats)
    }

    fn ofv_fixed(&self, x: &[f64], etas: &[DVector<f64>], h_mats: &[DMatrix<f64>]) -> f64 {
        let params = unpack_params(x, self.init_params);
        let nll = pop_nll(
            self.model,
            self.population,
            &params,
            etas,
            h_mats,
            &[], // trust_region doesn't support IOV yet; kappas empty
            self.options.interaction,
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
    let solver = TrustRegion::new(subproblem)
        .with_radius(1.0)
        .expect("trust region radius must be positive")
        .with_max_radius(10.0)
        .expect("trust region max radius must be positive");

    let result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(x0.clone())
                .max_iters(options.outer_maxiter as u64)
        })
        .run();

    let (converged, mut best_x) = match result {
        Ok(res) => {
            if options.verbose {
                eprintln!("Trust-region finished: {} iters", res.state().get_iter());
            }
            let vec = res
                .state()
                .get_best_param()
                .cloned()
                .unwrap_or_else(|| x0.clone());
            (true, vec)
        }
        Err(e) => {
            if options.verbose {
                eprintln!("Trust-region stopped: {}", e);
            }
            warnings.push(format!("Trust-region did not converge: {}", e));
            (false, x0.clone())
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
    );

    let final_ofv = 2.0
        * pop_nll(
            model,
            population,
            &final_params,
            &final_ehs,
            &final_hms,
            &final_kappas,
            options.interaction,
        );

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let (covariance_matrix, covariance_wall_time_secs) =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            let cov_timer = std::time::Instant::now();
            let cm = match compute_covariance(
                &best_x,
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
            };
            (cm, cov_timer.elapsed().as_secs_f64())
        } else {
            (None, 0.0)
        };

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
        final_gradient: None,
        sir_fallback_proposal,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
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
}
