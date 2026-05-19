use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::gauss_newton::subject_nll_pop_grad;
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, pop_nll, OuterResult};
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
    fn compute_ad_grads(&self, x: &[f64]) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<f64>>) {
        // Return cached result if x matches.
        {
            let cache = self.grad_cache.lock().unwrap();
            if let Some(ref c) = *cache {
                if c.x == x {
                    return (c.etas.clone(), c.h_mats.clone(), c.per_subj_grads.clone());
                }
            }
        }

        let (etas, h_mats) = self.run_inner(x);
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
        Ok(self.ofv_fixed(p, &etas, &h_mats))
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

/// Size-adaptive Steihaug CG budget: `ceil(sqrt(n_params)).clamp(5, n_params)`.
/// Avoids the fixed-50 default that wastes CG iterations when n_params ≤ 15.
fn adaptive_steihaug_budget(n_params: usize) -> usize {
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

    let covariance_matrix = if options.run_covariance_step {
        if options.verbose {
            eprintln!("Computing covariance matrix...");
        }
        compute_covariance(
            &best_x,
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

    if covariance_matrix.is_none() && options.run_covariance_step {
        warnings.push("Covariance step failed".to_string());
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: 0,
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
