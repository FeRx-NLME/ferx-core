use nalgebra::{DMatrix, DVector};

use crate::estimation::gauss_newton::{ebe_response_correction, subject_nll_pop_grad_with_cache};
use crate::estimation::inner_optimizer::{
    run_inner_loop_warm_map, InnerHessianSeed, InnerSolvePolicy, PrevEtas,
};
use crate::estimation::outer_optimizer::{
    final_ebe_check, gate_converged_on_objective, pop_nll_opts, score_final_ebes, OuterResult,
};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_mu_k, pack_with_bounds, unpack_params, PackedBounds, PackedStart,
};
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

/// Per-call cache for per-subject NLL gradients: a BHHH iteration's gradient and curvature
/// ([`FoceiProblem::likelihood_gradient`], [`FoceiProblem::bhhh`]) share one inner solve and
/// one sweep.
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
    /// What `cached_etas` are: `ModesHere` while they are converged modes at the start
    /// point, not yet consumed — the next inner pass takes them as warm starts whatever
    /// their values (`InnerSolvePolicy::init_is_mode`, #2076) — and `Solved` once a pass
    /// wrote them, so an exactly-zero EBE is not re-solved as a cold start.
    cached_etas_kind: std::sync::Mutex<PrevEtas>,
    grad_cache: std::sync::Mutex<Option<GradCache>>,
    /// Covariate-NN (DCM) regularizer. No-op when both λ are 0. Added to the
    /// optimizer objective (`cost`/`ofv_fixed`), gradient and Hessian, not to
    /// the final reported OFV, which reuses a clean `pop_nll_opts`.
    nn_reg: crate::estimation::nn_reg::NnRegularizer,
    /// Parameter priors (#254). Same contract as `nn_reg`: in the optimizer
    /// objective, gradient and Hessian; out of the reported OFV.
    priors: crate::estimation::priors::PriorSet,
}

/// The trust region (#2975): a dense, scaled trust-region Newton on the exact Hessian. Each iteration
/// forms the exact gradient and Hessian of the penalized OFV at the incumbent (EBEs, and κ̂ under
/// IOV, re-solved), measures the step in the norm `‖D s‖` with `D = √diag|H|` kept at its running
/// maximum (Moré's scaling: the curvature scale of every coordinate), makes the scaled Hessian
/// saddle-free (`|H|`), fixes the coordinates held at a bound by the gradient, solves the
/// trust-region subproblem exactly on the eigenbasis, projects onto the box and accepts on the
/// actual-to-predicted ratio. Where the exact Hessian cannot be formed at an iterate, BHHH from
/// the exact per-subject gradients stands in for that iteration. Stops when the exact Newton
/// decrement is at most [`TR_EXACT_DECREMENT_TOL`].
fn exact_trust_region(
    problem: &FoceiProblem<'_>,
    x0: &[f64],
    bounds: &PackedBounds,
    options: &FitOptions,
    budget_spent: &mut bool,
) -> (bool, Vec<f64>, usize, Option<Vec<f64>>, HeldEbes) {
    use crate::estimation::outer_optimizer::{joint_warm_starts, outer_cache_warm_starts};
    let n = x0.len();
    let model = problem.model;
    let population = problem.population;
    let axes: Vec<usize> = (0..n)
        .filter(|&k| bounds.lower[k] < bounds.upper[k])
        .collect();
    let n_subj = population.subjects.len();
    // The inner solve at `x` from `warm` (η̂, then each κ̂ cell): the penalized objective and the
    // modes.
    let eval = |x: &[f64], warm: &[DVector<f64>]| {
        let params = unpack_params(x, problem.init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (etas, hms, _, kappas, _) = run_inner_loop_warm_map(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(warm),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
            options.inner_restarts,
            InnerSolvePolicy {
                seed: InnerHessianSeed::for_options(options),
                ..InnerSolvePolicy::for_stage(options, model)
            }
            .with_prev_etas(PrevEtas::Solved),
            None,
            |_, _| (),
        );
        let nll = pop_nll_opts(model, population, &params, &etas, &hms, &kappas, options);
        let raw =
            2.0 * nll + problem.nn_reg.penalty_value(&params.theta) + problem.priors.penalty(x);
        let f = if raw.is_finite() { raw } else { 1e20 };
        (f, etas, hms, kappas)
    };
    let mut x = x0.to_vec();
    let start_warm = problem.cached_etas.lock().unwrap().clone();
    let start_warm = if start_warm.len() == n_subj {
        start_warm
    } else {
        vec![DVector::zeros(model.n_eta); n_subj]
    };
    let (mut f, mut etas, mut hms, mut kappas) = eval(&x, &start_warm);
    let mut radius = 1.0_f64;
    let mut d_scale = vec![0.0_f64; axes.len()];
    let mut converged = false;
    let mut iters = 0usize;
    let mut last_g: Option<Vec<f64>> = None;
    let at = |b: f64, v: f64| b.is_finite() && (v - b).abs() <= 1e-10 * (1.0 + b.abs());
    for it in 0..options.outer_maxiter {
        if crate::cancel::is_cancelled(&options.cancel) {
            break;
        }
        iters = it + 1;
        // The exact Hessian re-solves the joint [η, κ] modes; the trial's inner solve starts
        // from the warm starts the NLopt objective uses (κ̂ restarted at 0 on ODE + IOV, #2508).
        let warm = joint_warm_starts(&etas, &kappas);
        let trial_warm = outer_cache_warm_starts(model, &etas, &kappas);
        let params = unpack_params(&x, problem.init_params);
        let mut pg = vec![0.0; n];
        let mut ph = DMatrix::<f64>::zeros(n, n);
        problem.nn_reg.add_packed_gradient(&params.theta, &mut pg);
        problem
            .nn_reg
            .add_packed_hessian(&params.theta, &mut |i, j, v| ph[(i, j)] += v);
        problem.priors.add_gradient(&x, &mut pg);
        problem.priors.add_hessian(&mut |i, j, v| ph[(i, j)] += v);
        let exact = crate::estimation::covariance::exact_candidate_hessian(
            model,
            problem.population,
            problem.init_params,
            &x,
            &warm,
            options,
            &axes,
        );
        let (g_ax, h_ax): (Vec<f64>, DMatrix<f64>) = match exact {
            Some(gh) => gh,
            None if model.n_kappa == 0 => {
                // BHHH from the exact per-subject gradients, for this iteration only.
                *problem.cached_etas.lock().unwrap() = etas.clone();
                let g = problem.likelihood_gradient(&x);
                let h = problem.bhhh(&x);
                if options.verbose {
                    eprintln!("  TR iter {it:>3}: exact Hessian unavailable, BHHH this iteration");
                }
                (
                    axes.iter().map(|&k| g[k]).collect(),
                    DMatrix::from_fn(axes.len(), axes.len(), |i, j| h[(axes[i], axes[j])]),
                )
            }
            None => break,
        };
        let g_full: Vec<f64> = (0..n)
            .map(|k| axes.iter().position(|&a| a == k).map_or(0.0, |i| g_ax[i]) + pg[k])
            .collect();
        last_g = Some(g_full.clone());
        let fr: Vec<usize> = (0..axes.len())
            .filter(|&i| {
                let k = axes[i];
                !(at(bounds.lower[k], x[k]) && g_full[k] > 0.0)
                    && !(at(bounds.upper[k], x[k]) && g_full[k] < 0.0)
            })
            .collect();
        if fr.is_empty() {
            converged = true;
            break;
        }
        let m = fr.len();
        let h = DMatrix::from_fn(m, m, |i, j| {
            h_ax[(fr[i], fr[j])] + ph[(axes[fr[i]], axes[fr[j]])]
        });
        if h.iter().chain(g_full.iter()).any(|v| !v.is_finite()) {
            break;
        }
        // Moré's scaling: `D` at its running maximum of `√|H_aa|`.
        for (i, &fi) in fr.iter().enumerate() {
            let hd = h[(i, i)].abs().sqrt();
            if hd.is_finite() && hd > d_scale[fi] {
                d_scale[fi] = hd;
            }
        }
        let dmax = fr.iter().map(|&fi| d_scale[fi]).fold(0.0_f64, f64::max);
        if dmax <= 0.0 {
            break;
        }
        let d: Vec<f64> = fr.iter().map(|&fi| d_scale[fi].max(1e-8 * dmax)).collect();
        let gy = DVector::from_iterator(m, (0..m).map(|i| g_full[axes[fr[i]]] / d[i]));
        let hy = DMatrix::from_fn(m, m, |i, j| h[(i, j)] / (d[i] * d[j]));
        let Some(sub) = saddle_free_subproblem(&gy, hy, radius) else {
            break;
        };
        let decrement = sub.decrement;
        if options.verbose {
            eprintln!(
                "  TR iter {it:>3}: OFV = {f:.6}  decrement {decrement:.3e}  radius {radius:.3e}  free {m}/{}",
                axes.len()
            );
        }
        if decrement <= TR_EXACT_DECREMENT_TOL {
            converged = true;
            break;
        }
        let py = &sub.step;
        let mut xn = x.clone();
        for (i, &fi) in fr.iter().enumerate() {
            let k = axes[fi];
            xn[k] = (x[k] + py[i] / d[i]).clamp(bounds.lower[k], bounds.upper[k]);
        }
        // The step actually taken, in the scaled norm, and the model's prediction for it.
        let sy =
            DVector::from_iterator(m, (0..m).map(|i| (xn[axes[fr[i]]] - x[axes[fr[i]]]) * d[i]));
        let pred = -(gy.dot(&sy) + 0.5 * sy.dot(&(&sub.habs * &sy)));
        let (fnew, en, hn, kn) = eval(&xn, &trial_warm);
        let rho = if pred > 0.0 { (f - fnew) / pred } else { -1.0 };
        let snorm = sy.norm();
        if fnew < f && rho > 1e-4 {
            x = xn;
            f = fnew;
            etas = en;
            hms = hn;
            kappas = kn;
            if rho > 0.75 && snorm >= 0.99 * radius {
                radius = (2.0 * radius).min(1e3);
            } else if rho < 0.25 {
                radius = 0.25 * snorm.max(1e-12);
            }
        } else {
            radius = 0.25 * snorm.max(1e-12);
        }
        if radius < 1e-10 {
            break;
        }
    }
    *problem.cached_etas.lock().unwrap() = etas.clone();
    if iters == options.outer_maxiter && !converged {
        *budget_spent = true;
    }
    let held = HeldEbes {
        etas,
        h_mats: hms,
        kappas,
    };
    (converged, x, iters, last_g, held)
}

/// The exact solution of one trust-region subproblem (#2975), in scaled coordinates.
pub(crate) struct Subproblem {
    /// The step: the minimiser of `gᵀs + ½ sᵀ|H|s` over `‖s‖ ≤ radius`.
    pub(crate) step: DVector<f64>,
    /// The Newton decrement `½ gᵀ|H|⁻¹g`: the drop the unconstrained step predicts.
    pub(crate) decrement: f64,
    /// `|H|`: `H` with its eigenvalues made absolute (floored at `1e-10·max|λ|`).
    pub(crate) habs: DMatrix<f64>,
}

/// Solve `min gᵀs + ½ sᵀ|H|s` subject to `‖s‖ ≤ radius` exactly, on `H`'s eigenbasis (#2975):
/// the Newton step `−|H|⁻¹g` when it fits, else `−(|H| + μI)⁻¹g` with `μ > 0` found by
/// bisection so the step lies on the boundary. Saddle-free: a negative eigenvalue counts by its
/// magnitude, so the step always descends. `None` when `H` is zero or not finite.
pub(crate) fn saddle_free_subproblem(
    g: &DVector<f64>,
    h: DMatrix<f64>,
    radius: f64,
) -> Option<Subproblem> {
    let m = g.len();
    if g.iter().chain(h.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let eig = nalgebra::SymmetricEigen::new(h);
    let lmax = eig.eigenvalues.iter().fold(0.0_f64, |a, l| a.max(l.abs()));
    if lmax <= 0.0 {
        return None;
    }
    let lam: Vec<f64> = eig
        .eigenvalues
        .iter()
        .map(|l| l.abs().max(1e-10 * lmax))
        .collect();
    let along = eig.eigenvectors.transpose() * g;
    let decrement: f64 = (0..m).map(|j| 0.5 * along[j] * along[j] / lam[j]).sum();
    let norm_at = |mu: f64| {
        (0..m)
            .map(|j| (along[j] / (lam[j] + mu)).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mu = if norm_at(0.0) <= radius {
        0.0
    } else {
        let (mut lo, mut hi) = (0.0_f64, g.norm() / radius);
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if norm_at(mid) > radius {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    };
    let coef = DVector::from_iterator(m, (0..m).map(|j| -along[j] / (lam[j] + mu)));
    let habs = &eig.eigenvectors
        * DMatrix::from_diagonal(&DVector::from_vec(lam))
        * eig.eigenvectors.transpose();
    Some(Subproblem {
        step: &eig.eigenvectors * coef,
        decrement,
        habs,
    })
}

/// The exact Newton decrement at or below which the exact-Hessian trust region stops (#2975).
const TR_EXACT_DECREMENT_TOL: f64 = 1e-6;

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
        // `run_inner_loop_warm_seeded`'s solve, with the #2076 certificate carried through.
        // Every pass's modes become the cache below.
        let prev = std::mem::replace(
            &mut *self.cached_etas_kind.lock().unwrap(),
            PrevEtas::Solved,
        );
        let (etas, h_mats, _, _kappas, _) = run_inner_loop_warm_map(
            self.model,
            self.population,
            &params,
            self.options.inner_maxiter,
            self.options.inner_tol,
            warm_ref,
            Some(&mu_k),
            self.options.min_obs_for_convergence_check as usize,
            self.options.inner_restarts,
            InnerSolvePolicy {
                seed: InnerHessianSeed::for_options(self.options),
                ..InnerSolvePolicy::for_stage(self.options, self.model)
            }
            .with_prev_etas(prev),
            None,
            |_, _| (),
        );
        *self.cached_etas.lock().unwrap() = etas.clone();
        (etas, h_mats)
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

        let per_subj: Vec<(Vec<f64>, Vec<f64>)> = crate::par::map(n_subj, |i| {
            let (nll, gi, cache) = subject_nll_pop_grad_with_cache(
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
            let ti = ebe_response_correction(
                x,
                self.init_params,
                self.model,
                self.population,
                i,
                &etas[i],
                &[],
                None,
                nll,
                &gi,
                cache.as_ref(),
                &self.bounds,
                self.options,
            );
            (gi, ti)
        });

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

impl FoceiProblem<'_> {
    /// `grad(OFV) = 2 · Σᵢ (gᵢ + tᵢ)` — the score plus the FOCEI `log|H̃|`
    /// EBE-response curvature.
    ///
    /// `tᵢ` is not optional polish: under `interaction` the fixed-η̂ analytic
    /// gradient drops it (the #274/#289 Δ — the envelope theorem zeros the inner
    /// objective but not `log|H̃|`), and a gradient optimizer descending on the
    /// truncated score stalls above the minimum (warfarin FOCEI −276.6 instead of
    /// −286.0, `gauss_newton::build_gn_system`). It is identically zero under
    /// FOCE and for additive error, so those paths are unchanged bit-for-bit.
    fn likelihood_gradient(&self, p: &[f64]) -> Vec<f64> {
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
        // Parameter priors (#254) are defined on the packed vector directly, so
        // `p` is already the right space.
        self.priors.add_gradient(p, &mut g);
        g
    }

    /// The BHHH curvature `4 Σ gᵢgᵢᵀ` with the penalties' exact curvature: what an iteration
    /// whose exact Hessian cannot be formed models the objective with (#2975).
    fn bhhh(&self, p: &[f64]) -> DMatrix<f64> {
        let (_, _, per_subj, _) = self.compute_ad_grads(p);
        let n = p.len();
        // BHHH approximation: H ≈ 4 Σ gᵢgᵢᵀ  (factor 4 because OFV = 2*NLL,
        // so grad(OFV) = 2*gᵢ and the outer product scales by 4).
        //
        // Raw score only — the FOCEI `tᵢ` correction the gradient carries is a
        // *curvature* term, not part of the score, and folding it into the outer
        // product corrupts the Hessian (same rule as `build_gn_system`).
        let mut h = DMatrix::<f64>::zeros(n, n);
        for gi in &per_subj {
            for i in 0..n {
                for j in 0..n {
                    h[(i, j)] += 4.0 * gi[i] * gi[j];
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
            .add_packed_hessian(&params.theta, &mut |i, j, v| h[(i, j)] += v);
        // Parameter-prior curvature (#254): the exact `2/s²` diagonal, not an
        // approximation. Same reason it has to be here as the NN term — the
        // gradient carries the prior's pull, so a quadratic model without its
        // curvature would read every priored direction as flat and shrink the
        // radius on steps that were fine. PSD, so the BHHH model stays PSD.
        self.priors.add_hessian(&mut |i, j, v| h[(i, j)] += v);
        h
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

/// The modes the trust region's best accepted point was scored with (#2097 review): the
/// counterpart of the NLopt leg's `IncumbentSolve`, held rather than re-solved, since a
/// re-solve at that point could land in another mode on a start-dependent surface.
struct HeldEbes {
    etas: Vec<DVector<f64>>,
    h_mats: Vec<DMatrix<f64>>,
    /// Each subject's κ̂ cells (empty without IOV).
    kappas: Vec<Vec<DVector<f64>>>,
}

pub fn optimize_trust_region(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    optimize_trust_region_seeded(model, population, init_params, options, None)
}

/// [`optimize_trust_region`] with the first evaluation's EBEs supplied: `first_eval_etas`
/// are the EBEs already solved at the packed, clamped `init_params` (the flat-theta
/// pre-flight's, #2076), so the first inner solve starts at that mode instead of solving
/// it cold a second time. `None` is the cold start.
pub(crate) fn optimize_trust_region_seeded(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
    first_eval_etas: Option<&[DVector<f64>]>,
) -> OuterResult {
    let PackedStart {
        packed: mut x0,
        bounds,
        ..
    } = pack_with_bounds(init_params);
    clamp_to_bounds(&mut x0, &bounds);

    let mut warnings = Vec::new();

    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let nn_reg = crate::estimation::nn_reg::NnRegularizer::build(model, population, options);
    let nn_reg_active = nn_reg.is_active();
    // The loop needs the box beside `problem`, which owns one. Two `Vec<f64>` copies,
    // against a second `compute_bounds` walk of the same unchanged template (#1252).
    let post_run_bounds = PackedBounds {
        lower: bounds.lower.clone(),
        upper: bounds.upper.clone(),
    };
    let problem = FoceiProblem {
        model,
        population,
        options,
        init_params,
        bounds,
        cached_etas: std::sync::Mutex::new(match first_eval_etas {
            Some(etas) if etas.len() == n_subj => etas.to_vec(),
            _ => vec![DVector::zeros(n_eta); n_subj],
        }),
        cached_etas_kind: std::sync::Mutex::new(
            if first_eval_etas.is_some_and(|e| e.len() == n_subj) {
                PrevEtas::ModesHere
            } else {
                PrevEtas::Placeholder
            },
        ),
        grad_cache: std::sync::Mutex::new(None),
        priors: crate::estimation::outer_optimizer::build_prior_set(model, init_params),
        nn_reg,
    };

    if options.verbose {
        eprintln!(
            "Starting trust-region optimization ({} parameters)...",
            x0.len()
        );
    }

    let mut tr_budget_spent = false;
    let (converged, best_x, n_iterations, final_gradient, held) = exact_trust_region(
        &problem,
        &x0,
        &post_run_bounds,
        options,
        &mut tr_budget_spent,
    );
    if crate::cancel::is_cancelled(&options.cancel) {
        warnings.push("Trust-region did not converge: cancelled by user".to_string());
    } else if !converged {
        let label = if nn_reg_active {
            "∂(OFV + NN penalty)/∂x"
        } else {
            "∂OFV/∂x"
        };
        let reason = if tr_budget_spent {
            format!(
                "reached the iteration budget (maxiter = {}) before the exact Newton decrement \
                 fell to {TR_EXACT_DECREMENT_TOL:e}",
                options.outer_maxiter
            )
        } else {
            "stopped with a trust radius too small to make progress".to_string()
        };
        let g = final_gradient.as_deref().map_or(f64::NAN, l2_norm);
        warnings.push(format!(
            "Trust-region did not converge: it {reason}. Final ‖{label}‖ = {g:.3e}."
        ));
    }
    if options.verbose {
        eprintln!("Trust-region finished: {n_iterations} iters, converged = {converged}");
    }

    let final_params = unpack_params(&best_x, init_params);
    // The final EBEs, on the rules the NLopt leg uses (#2097): `final_ebe_check` re-solves
    // cold and reports that solve unless it missed the optimizer's own EBEs (#833), then the
    // best of the cold, a re-solve seeded from them, and they themselves — so a fit whose cold
    // solve reproduces its optimizer's EBEs is unchanged — and raises the #833, #1565 and
    // #2013 `W_EBE_START_DEPENDENT` forms. The held modes are always the reported point's:
    // every iterate is inside the box.
    let mut converged = converged;
    let incumbent = score_final_ebes(
        model,
        population,
        &final_params,
        (held.etas, held.h_mats, held.kappas),
        options,
    );
    let (reported, ebe_warnings) = final_ebe_check(
        model,
        population,
        &final_params,
        incumbent,
        true,
        options,
        &mut converged,
    );
    warnings.extend(ebe_warnings);
    let (final_ehs, final_hms, final_kappas, final_ofv) = reported;

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // A settled run whose objective is not a real population OFV has not
    // converged to anything: `ofv_fixed` clamps a blown-up objective to a ~1e20
    // sentinel, and the no-progress rule can settle on that sentinel or on any
    // other diverged value. `is_finite()` is *not* enough — the sentinel is
    // finite — so this uses the same validity cutoff the multi-start ranking
    // applies (`DIVERGENCE_OFV`).
    //
    // #1303 moved the demotion itself into the shared gate, so this engine, the
    // NLopt and built-in descents, Gauss–Newton, the MCEM family and `fit()`'s
    // own assembly all apply one rule and emit one message. The local
    // `warnings.is_empty()` guard the reason used to sit behind went with it: a
    // trust region that both stalled *and* landed on a `NaN` said only the first,
    // and the second is the one that invalidates every derived quantity.
    if let Some(w) = gate_converged_on_objective(&mut converged, final_ofv) {
        warnings.push(w);
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
        method: covariance_method,
        newton_decrement,
        non_stationary_optimum,
    } = out;
    warnings.extend(cov_warnings);
    // #2785: a trust region that stopped on a small step is not thereby at a minimum.
    crate::estimation::covariance::demote_non_stationary_optimum(
        non_stationary_optimum,
        &mut converged,
        &mut warnings,
    );

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
        final_gradient_source: final_gradient.as_ref().map(|_| "optimizer".to_string()),
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
        stationary_start: false,
        mixture_posteriors: None,
        vi: None,
        start_eval: None,
        mcem_lowest_objective: None,
        mcem_objective_trail: None,
        newton_decrement,
        optimizer_stop: None,
        eval_budget_spent: tr_budget_spent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The objective, gradient and BHHH curvature the trust region models, in the shapes
    /// the tests below compare: the penalized OFV at `p` with its EBEs re-solved warm.
    impl FoceiProblem<'_> {
        fn cost(&self, p: &[f64]) -> Result<f64, ()> {
            let (etas, h_mats) = self.run_inner(p);
            let params = unpack_params(p, self.init_params);
            let nll = pop_nll_opts(
                self.model,
                self.population,
                &params,
                &etas,
                &h_mats,
                &[],
                self.options,
            );
            let raw = 2.0 * nll + self.nn_reg.penalty_value(&params.theta) + self.priors.penalty(p);
            Ok(if raw.is_finite() { raw } else { 1e20 })
        }

        fn gradient(&self, p: &[f64]) -> Result<Vec<f64>, ()> {
            Ok(self.likelihood_gradient(p))
        }

        #[cfg(feature = "nn")]
        fn hessian(&self, p: &[f64]) -> Result<Vec<Vec<f64>>, ()> {
            let h = self.bhhh(p);
            Ok((0..h.nrows())
                .map(|i| (0..h.ncols()).map(|j| h[(i, j)]).collect())
                .collect())
        }
    }
    // Production code goes through `pack_with_bounds` (#1252); these two are
    // still the clearest way for a test to build a box or a packed vector on
    // its own, so they are imported here rather than at module scope.
    use crate::estimation::parameterization::{compute_bounds, pack_params};

    /// The exact subproblem (#2975). Inside the radius it is the Newton step `−|H|⁻¹g` and the
    /// decrement is `½gᵀ|H|⁻¹g`; outside it the step lies on the boundary and lowers the
    /// saddle-free model below the steepest-descent (Cauchy) step of the same length; a negative
    /// eigenvalue counts by its magnitude, so the step descends where plain Newton would climb.
    /// Mutations: a wrong bisection bracket leaves the step off the boundary; dropping `.abs()`
    /// makes the indefinite row ascend; a `½` dropped from the decrement fails the first row.
    #[test]
    fn the_subproblem_is_solved_exactly_and_saddle_free() {
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let g = DVector::from_vec(vec![1.0, -2.0]);
        let newton = -h.clone().try_inverse().unwrap() * &g;
        let inside = saddle_free_subproblem(&g, h.clone(), 10.0).unwrap();
        assert!((&inside.step - &newton).norm() < 1e-12, "{:?}", inside.step);
        let want = 0.5 * g.dot(&(h.clone().try_inverse().unwrap() * &g));
        assert!((inside.decrement - want).abs() < 1e-12 * want);

        let r = 0.1 * newton.norm();
        let edge = saddle_free_subproblem(&g, h.clone(), r).unwrap();
        assert!(
            (edge.step.norm() - r).abs() < 1e-9 * r,
            "{}",
            edge.step.norm()
        );
        let model = |s: &DVector<f64>| g.dot(s) + 0.5 * s.dot(&(&h * s));
        let cauchy = -&g * (r / g.norm());
        assert!(model(&edge.step) < model(&cauchy));

        let indefinite = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, -1.0]);
        let gi = DVector::from_vec(vec![0.5, 0.5]);
        let sub = saddle_free_subproblem(&gi, indefinite, 10.0).unwrap();
        assert!(gi.dot(&sub.step) < 0.0, "must descend: {:?}", sub.step);
        assert!(
            (sub.step[1] + 0.5).abs() < 1e-12,
            "|−1| = 1 on the second axis"
        );

        assert!(saddle_free_subproblem(&g, DMatrix::zeros(2, 2), 1.0).is_none());
        assert!(saddle_free_subproblem(&DVector::from_vec(vec![f64::NAN, 0.0]), h, 1.0).is_none());
    }

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

    /// #1778, the trust-region call site: on a TTE model with η, whose fixed-EBE score
    /// drops `∂_η log|H̃| · dη̂/dθ`, `gradient()` must be the total derivative `2·Σ dᵢ` (the
    /// exact adjoint) at the EBEs `compute_ad_grads` solved. Dies if `compute_ad_grads`
    /// stops calling `gauss_newton::ebe_response_correction` (each `tᵢ` falls back to zero
    /// and the gradient to the biased `2·Σ gᵢ`).
    #[cfg(feature = "survival")]
    #[test]
    fn gradient_is_the_total_derivative_on_a_tte_model_with_eta() {
        use crate::estimation::parameterization::{clamp_to_bounds, compute_bounds, pack_params};
        use argmin::core::Gradient;

        let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
        let model = crate::parser::model_parser::parse_model_string(&src).expect("parse");
        assert!(crate::estimation::gauss_newton::fixed_ebe_score_drops_ebe_response(&model));
        let (population, _) = crate::api::read_population_for(
            &model,
            &None,
            "data/tte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        let options = FitOptions::default();
        let bounds = compute_bounds(&model.default_params);
        let mut x0 = pack_params(&model.default_params);
        clamp_to_bounds(&mut x0, &bounds);
        let n_subj = population.subjects.len();
        let problem = FoceiProblem {
            model: &model,
            population: &population,
            options: &options,
            init_params: &model.default_params,
            bounds,
            cached_etas: std::sync::Mutex::new(vec![nalgebra::DVector::zeros(model.n_eta); n_subj]),
            cached_etas_kind: std::sync::Mutex::new(PrevEtas::Placeholder),
            grad_cache: std::sync::Mutex::new(None),
            nn_reg: crate::estimation::nn_reg::NnRegularizer::build(&model, &population, &options),
            priors: crate::estimation::outer_optimizer::build_prior_set(
                &model,
                &model.default_params,
            ),
        };

        let (etas, _, per_subj, _) = problem.compute_ad_grads(&x0);
        let g = problem.gradient(&x0).expect("gradient");
        // `dᵢ`: the exact total derivative (the adjoint at the mode), and the reconverged
        // difference as a loose cross-check of it.
        let mut want = vec![0.0_f64; x0.len()];
        let mut fd = vec![0.0_f64; x0.len()];
        let mut raw = vec![0.0_f64; x0.len()];
        let pt = crate::sens::ode_provider::adjoint::PackedPoint::new(&model.default_params, &x0)
            .unwrap();
        let fixed = crate::estimation::parameterization::packed_fixed_mask(&model.default_params);
        for (i, subject) in population.subjects.iter().enumerate() {
            let di = crate::sens::ode_provider::adjoint::subject_adjoint_gradient_at(
                &model,
                subject,
                &pt,
                etas[i].as_slice(),
                crate::sens::ode_provider::adjoint::Objective::Focei,
            )
            .expect("in the adjoint's scope");
            let fi = crate::estimation::outer_optimizer::subject_reconverged_fd_gradient(
                &x0,
                &model.default_params,
                &model,
                subject,
                &etas[i],
                &problem.bounds,
                &options,
            );
            for k in 0..x0.len() {
                if !fixed[k] {
                    want[k] += 2.0 * di[k];
                }
                fd[k] += 2.0 * fi[k];
                raw[k] += 2.0 * per_subj[i][k];
            }
        }
        assert!(
            g.iter()
                .chain(&want)
                .chain(&fd)
                .chain(&raw)
                .all(|v| v.is_finite()),
            "every side must be finite: gradient {g:?}, exact {want:?}, reconverged {fd:?}, raw {raw:?}"
        );
        let vs_fd = want
            .iter()
            .zip(&fd)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        // The reconverged difference's own error at its step and the inner tolerance.
        assert!(vs_fd < 1e-4, "exact vs reconverged {vs_fd:.3e}");
        let off = g
            .iter()
            .zip(&want)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let raw_gap = raw
            .iter()
            .zip(&want)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("[#1778] trust-region: ‖g − 2Σd‖∞ = {off:.3e}, ‖2Σg − 2Σd‖∞ = {raw_gap:.3e}");
        // Each `gᵢ + (dᵢ − gᵢ)` rounds back to `dᵢ`: 1e-10 allows for summation-order
        // rounding. The raw gap was measured at 2.435e-1; 5e-2 is 4.9× under it.
        assert!(
            off < 1e-10,
            "the trust-region gradient must be the total derivative, off by {off:.3e}"
        );
        assert!(
            raw_gap > 5e-2,
            "fixture precondition: the raw score must be visibly biased, gap {raw_gap:.3e}"
        );
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

    // --- #1000: convergence reporting -----------------------------------------

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
                cached_etas_kind: std::sync::Mutex::new(PrevEtas::Placeholder),
                grad_cache: std::sync::Mutex::new(None),
                priors: crate::estimation::outer_optimizer::build_prior_set(&model, init),
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
    /// as AGENTS.md asks of "the provider that assembles them". The warfarin
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

        // The bare start: its output-layer weights are live, so the curvature
        // penalty and its gradient are not ≈ 0 (see `dcm_fixture_without_init`).
        let parsed =
            parse_full_model(&crate::nn::dcm_fixture_without_init()).expect("DCM fixture parses");
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
                cached_etas_kind: std::sync::Mutex::new(PrevEtas::Placeholder),
                grad_cache: std::sync::Mutex::new(None),
                nn_reg: crate::estimation::nn_reg::NnRegularizer::build(model, population, options),
                priors: crate::estimation::outer_optimizer::build_prior_set(model, init),
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
        // #2799: the typed form of the same fact.
        assert!(res.eval_budget_spent, "MaxItersReached is a spent budget");
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

    /// #574: the trust region polls the cancel flag at every iteration boundary and
    /// skips the post-run gradient pass. Without the `terminate` check a pre-cancelled
    /// run walks on (every trial scored 1e20) until the no-initial-progress stop, whose
    /// warning names the starting values instead of the cancel.
    #[test]
    fn a_cancelled_trust_region_stops_at_the_next_iteration() {
        use crate::io::datareader::read_nonmem_csv;
        use std::path::Path;

        let model = warfarin_model();
        let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data must load");
        let flag = crate::cancel::CancelFlag::new();
        flag.cancel();
        let opts = FitOptions {
            outer_maxiter: 50,
            run_covariance_step: false,
            verbose: false,
            cancel: Some(flag),
            ..FitOptions::default()
        };
        let res = optimize_trust_region(&model, &population, &model.default_params, &opts);
        assert!(!res.converged);
        assert!(
            res.warnings.iter().any(|w| w.contains("cancelled by user")),
            "the stop must name the cancel: {:?}",
            res.warnings
        );
        assert!(res.n_iterations <= 1, "ran {} iterations", res.n_iterations);
        assert!(
            res.final_gradient.is_none(),
            "a cancelled run paid a gradient pass"
        );
    }

    fn warfarin_model() -> crate::types::CompiledModel {
        crate::parser::model_parser::parse_model_file(std::path::Path::new(
            "examples/warfarin.ferx",
        ))
        .expect("warfarin model must parse")
    }
}
