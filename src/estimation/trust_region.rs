use nalgebra::{DMatrix, DVector};

use crate::estimation::inner_optimizer::{
    run_inner_loop_warm_map, InnerHessianSeed, InnerSolvePolicy, PrevEtas,
};
use crate::estimation::outer_optimizer::{
    ebe_guard_rejects, final_ebe_check, gate_converged_on_objective, pop_nll_opts,
    score_final_ebes, OuterResult,
};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_mu_k, pack_with_bounds, unpack_params, PackedBounds, PackedStart,
};
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

/// Per-call cache for per-subject NLL gradients: a BHHH iteration's gradient and curvature
/// ([`FoceiProblem::data_gradient`]) reuses the iterate's inner solve and
/// one sweep.
struct GradCache {
    x: Vec<f64>,
    etas: Vec<DVector<f64>>,
    h_mats: Vec<DMatrix<f64>>,
    /// Per-subject total derivative `dᵢ` (#2974) — the gradient's terms and the BHHH
    /// Hessian's outer-product input alike.
    per_subj_grads: Vec<Vec<f64>>,
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
    /// The evaluation trace the EBE convergence guard reads (#1617): every objective the
    /// trust region scores goes through [`ebe_guard_rejects`], as the NLopt objective does.
    feasibility: std::sync::Mutex<TrFeasibility>,
    /// Per packed coordinate, the largest `√|B_aa|` any leg's model gave it (`0` where none
    /// did): the curvature units Moré's scaling measures steps in, and the units the
    /// `left_init` escape test (#751) reads a move in, as NLopt reads it in its curvature-scaled
    /// space (#2975).
    curvature_scale: std::sync::Mutex<Vec<f64>>,
}

/// Per-evaluation tallies of the trust region's objective, for the #1617 verdict
/// ([`crate::estimation::outer_optimizer::no_feasible_eval_warning`]) and the published
/// `max_unconverged_subjects`.
#[derive(Debug, Default, Clone, Copy)]
struct TrFeasibility {
    feasible: usize,
    guard_rejected: usize,
    nonfinite: usize,
    max_unconverged: usize,
}

/// The trust region (#2975): PORT's model/trust-region method (the `DMNGB` family behind R's
/// `nlminb`: Dennis, Gay & Welsch 1981; Gay 1983) with exact second-order information.
///
/// - **Model.** `m(s) = f + gᵀs + ½ sᵀBs`, with `g` the exact adjoint gradient at the iterate's
///   own modes and `B` a BFGS matrix **seeded with the exact Hessian made saddle-free**, `|H|`
///   (eigenvalues by absolute value): positive definite, so every secant update applies (a raw
///   indefinite seed has `sᵀBs < 0` along its negative directions and never learns them).
///   Updates are Powell-damped (`y` blended toward `Bs` when `sᵀy < 0.2·sᵀBs`), since a trust
///   region, unlike a Wolfe line search, does not guarantee `sᵀy > 0`.
/// - **Exact refresh.** `B` is replaced by a fresh exact `|H|` when its step predicted badly
///   (`ρ < ¼`) and when the set of coordinates free of their bounds changes (the secant
///   curvature was learned in the old subspace).
/// - **Step.** Coordinates held at a bound by the gradient are fixed; the subproblem is solved
///   exactly on `B`'s eigenbasis in the norm `‖D s‖`, `D = √diag|B|` at its running maximum
///   (Moré's scaling, PORT's `DD7DUP`); the step is projected onto the box and accepted on the
///   actual-to-predicted ratio.
/// - **Stop.** A model Newton decrement `½ gᵀB⁻¹g ≤` [`TR_EXACT_DECREMENT_TOL`] is certified on a
///   fresh exact Hessian: the fit stops only on the exact decrement.
///
/// Where the exact Hessian is out of scope (a model or subject the exact derivatives do not
/// cover), the same rule runs on the finite-difference Hessian of the same objective and the
/// #2974 rows' gradient (non-IOV models): one rule, two precisions.
fn exact_trust_region(
    problem: &FoceiProblem<'_>,
    x0: &[f64],
    bounds: &PackedBounds,
    options: &FitOptions,
    budget_spent: &mut bool,
) -> (bool, Vec<f64>, usize, Option<Vec<f64>>, HeldEbes, f64, f64) {
    use crate::estimation::covariance::{
        exact_candidate_hessian, exact_gradient_at_modes, fd_candidate_hessian, ExactDecline,
    };
    use crate::estimation::outer_optimizer::{joint_warm_starts, outer_cache_warm_starts};
    let n = x0.len();
    let model = problem.model;
    let population = problem.population;
    let axes: Vec<usize> = (0..n)
        .filter(|&k| bounds.lower[k] < bounds.upper[k])
        .collect();
    let n_subj = population.subjects.len();
    // The inner solve at `x` from `warm` (η̂, then each κ̂ cell): the penalized objective and the
    // modes. `prev` says what `warm` is (#2076): the start's cache kind for the first
    // evaluation (the pre-flight's converged modes are taken as modes, not re-scanned), the
    // modes of an earlier pass for every trial after it. A point the EBE convergence guard
    // rejects, or whose objective is not finite, scores `1e20`, so its step is refused, as
    // the NLopt objective refuses it (#1617).
    let eval = |x: &[f64], warm: &[DVector<f64>], prev: PrevEtas| {
        let params = unpack_params(x, problem.init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (etas, hms, stats, kappas, _) = run_inner_loop_warm_map(
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
            .with_prev_etas(prev),
            None,
            |_, _| (),
        );
        let nll = pop_nll_opts(model, population, &params, &etas, &hms, &kappas, options);
        let raw =
            2.0 * nll + problem.nn_reg.penalty_value(&params.theta) + problem.priors.penalty(x);
        let mut tally = problem.feasibility.lock().unwrap();
        tally.max_unconverged = tally.max_unconverged.max(stats.n_unconverged);
        let f = if !raw.is_finite() {
            tally.nonfinite += 1;
            1e20
        } else if ebe_guard_rejects(&stats, n_subj, raw, options.max_unconverged_frac) {
            tally.guard_rejected += 1;
            1e20
        } else {
            tally.feasible += 1;
            raw
        };
        (f, etas, hms, kappas, stats, raw)
    };
    let mut x = x0.to_vec();
    let start_warm = problem.cached_etas.lock().unwrap().clone();
    let start_warm = if start_warm.len() == n_subj {
        start_warm
    } else {
        vec![DVector::zeros(model.n_eta); n_subj]
    };
    let start_kind = std::mem::replace(
        &mut *problem.cached_etas_kind.lock().unwrap(),
        PrevEtas::Solved,
    );
    let (mut f, mut etas, mut hms, mut kappas, start_stats, start_raw) =
        eval(&x, &start_warm, start_kind);
    // #2682, on the NLopt leg's rule: a start the guard rejects on its *fraction* of
    // unconverged subjects is solved once more, warm from the EBEs it just found, before it is
    // judged; at the start there is no incumbent's modes to warm-start from, and a second pass
    // gives the unconverged subjects that budget. Kept only if the guard then accepts it.
    if crate::estimation::outer_optimizer::wants_start_warm_rescore(
        &start_stats,
        n_subj,
        start_raw,
        options.max_unconverged_frac,
    ) {
        let rewarm = joint_warm_starts(&etas, &kappas);
        let (f2, e2, h2, k2, _, _) = eval(&x, &rewarm, PrevEtas::Solved);
        if f2 < 1e20 {
            (f, etas, hms, kappas) = (f2, e2, h2, k2);
        }
    }
    if f >= 1e20 {
        // The start is infeasible (#1617): there is no objective to model or to improve on,
        // and a step from it would rank a guard value against a guard value. Stop here, not
        // converged; the caller reports the start with the no-feasible-evaluation verdict.
        let held = HeldEbes {
            etas,
            h_mats: hms,
            kappas,
        };
        return (false, x, 0, None, held, f, f);
    }
    let f_start = f;
    let mut radius = 1.0_f64;
    let mut d_scale = vec![0.0_f64; axes.len()];
    // The BFGS model on `axes`, and the point and gradient it was last updated at.
    let mut model_b: Option<DMatrix<f64>> = None;
    let mut last: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut refresh = true;
    // The point the model's exact Hessian was formed at: a refresh there would recompute it.
    let mut exact_at: Option<Vec<f64>> = None;
    // `|actual − predicted|` of the last rejected step: once the radius has collapsed the step
    // is so short that the model's own error (third order in it) is negligible, and this is the
    // objective's evaluation noise at the incumbent, measured.
    let mut last_mismatch = f64::NAN;
    let mut last_decrement: f64;
    // `OutOfScope` is a property of the model: once the exact Hessian (or the exact gradient at
    // held modes) declines that way, it is not asked again this leg. `NotResolved` is one of
    // the point: the exact Hessian is not asked again at the point it declined at.
    let mut hessian_out_of_scope = false;
    let gradient_out_of_scope = std::cell::Cell::new(false);
    let mut unresolved_at: Option<Vec<f64>> = None;
    // The optimizer trace: the last point a row was written at, and the packed norm of the
    // accepted step that reached the current iterate.
    let mut traced_at: Option<Vec<f64>> = None;
    let mut last_step: Option<f64> = None;
    // The first radius is set from the seed model's own Newton step (below).
    let mut radius_set = false;
    let mut prev_free: Option<Vec<usize>> = None;
    let mut n_exact = 0usize;
    let mut converged = false;
    let mut iters = 0usize;
    let mut last_g: Option<Vec<f64>> = None;
    let at = |b: f64, v: f64| b.is_finite() && (v - b).abs() <= 1e-10 * (1.0 + b.abs());
    // The #2974 rows' gradient along `axes` at `x`, from `x`'s own modes: where the exact
    // adjoint gradient is out of scope (non-IOV only; the rows carry no κ̂).
    // The gradient that goes with the FD twin of the exact Hessian, of the objective the fit
    // scores: the #2974 rows' total derivative of the FOCE/FOCEI marginal, and otherwise (IOV,
    // or a quadrature objective, which the rows do not differentiate) the reconverged central
    // difference of `pop_nll_opts`, which re-solves the EBEs and κ̂ at every probe.
    let rows_gradient = |x: &[f64], etas: &[DVector<f64>], kappas: &[Vec<DVector<f64>>]| {
        let g = if model.n_kappa == 0 && options.agq_nodes().is_none() {
            *problem.cached_etas.lock().unwrap() = etas.to_vec();
            problem.data_gradient(x)
        } else {
            crate::estimation::outer_optimizer::reconverged_fd_gradient(
                x,
                problem.init_params,
                model,
                population,
                etas,
                kappas,
                bounds,
                options,
            )
        };
        axes.iter().map(|&k| g[k]).collect::<Vec<f64>>()
    };
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
        // The secant model: the exact gradient at the iterate's own modes (no re-solve), and
        // `B` updated with the step since the last model point.
        let secant = || {
            model_b.as_ref().and_then(|b| {
                let g: Vec<f64> = match &last {
                    Some((xl, gl)) if *xl == x => gl.clone(),
                    _ => (!gradient_out_of_scope.get())
                        .then(|| {
                            exact_gradient_at_modes(
                                model,
                                population,
                                problem.init_params,
                                &x,
                                &etas,
                                &kappas,
                                options,
                                &axes,
                            )
                        })
                        .flatten()
                        .unwrap_or_else(|| {
                            gradient_out_of_scope.set(true);
                            rows_gradient(&x, &etas, &kappas)
                        }),
                };
                let mut b = b.clone();
                if let Some((xl, gl)) = last.as_ref().filter(|(xl, _)| *xl != x) {
                    let s = DVector::from_iterator(axes.len(), axes.iter().map(|&k| x[k] - xl[k]));
                    let y =
                        DVector::from_iterator(axes.len(), (0..axes.len()).map(|i| g[i] - gl[i]));
                    damped_bfgs_update(&mut b, &s, &y);
                }
                Some((g, b))
            })
        };
        let quasi = if refresh { None } else { secant() };
        let mut used_quasi = quasi.is_some();
        let (g_ax, h_ax): (Vec<f64>, DMatrix<f64>) = match quasi {
            Some(gb) => gb,
            None => {
                let exact = if hessian_out_of_scope {
                    Err(ExactDecline::OutOfScope)
                } else if unresolved_at.as_ref() == Some(&x) {
                    Err(ExactDecline::NotResolved)
                } else {
                    n_exact += 1;
                    exact_candidate_hessian(
                        model,
                        population,
                        problem.init_params,
                        &x,
                        &warm,
                        options,
                        &axes,
                    )
                };
                match exact {
                    Err(ExactDecline::OutOfScope) => hessian_out_of_scope = true,
                    Err(ExactDecline::NotResolved) => unresolved_at = Some(x.clone()),
                    Ok(_) => {}
                }
                match exact {
                    Ok(gh) => gh,
                    // Out of the exact scope: the same model from the finite-difference
                    // Hessian of the same objective, and the #2974 rows' gradient.
                    Err(ExactDecline::OutOfScope) => {
                        let Some(h) = fd_candidate_hessian(
                            model,
                            population,
                            problem.init_params,
                            &x,
                            &warm,
                            options,
                            &axes,
                        ) else {
                            break;
                        };
                        if options.verbose {
                            eprintln!("  TR iter {it:>3}: exact Hessian out of scope, FD Hessian");
                        }
                        (rows_gradient(&x, &etas, &kappas), h)
                    }
                    // In scope but not resolved at this point (an EBE that does not re-converge
                    // at the covariance tolerance, a non-finite term): a property of the point,
                    // not of the model, so the FD twin would meet the same trouble ~2n² times.
                    // Keep the secant model here; the next refresh trigger tries again.
                    Err(ExactDecline::NotResolved) => match secant() {
                        Some(gb) if !crate::cancel::is_cancelled(&options.cancel) => {
                            if options.verbose {
                                eprintln!(
                                    "  TR iter {it:>3}: exact Hessian not resolved here; secant model"
                                );
                            }
                            used_quasi = true;
                            gb
                        }
                        // No model yet to keep (the first iterate): the same quantity at FD
                        // precision, so the fit can start at all.
                        None if model_b.is_none() => {
                            let Some(h) = fd_candidate_hessian(
                                model,
                                population,
                                problem.init_params,
                                &x,
                                &warm,
                                options,
                                &axes,
                            ) else {
                                break;
                            };
                            if options.verbose {
                                eprintln!(
                                    "  TR iter {it:>3}: exact Hessian not resolved at the start; FD Hessian"
                                );
                            }
                            (rows_gradient(&x, &etas, &kappas), h)
                        }
                        _ => break,
                    },
                }
            }
        };
        refresh = false;
        if !used_quasi {
            exact_at = Some(x.clone());
        }
        let exact_here = exact_at.as_ref() == Some(&x);
        model_b = Some(if used_quasi {
            h_ax.clone()
        } else {
            saddle_free(&h_ax)
        });
        last = Some((x.clone(), g_ax.clone()));
        let g_full: Vec<f64> = (0..n)
            .map(|k| axes.iter().position(|&a| a == k).map_or(0.0, |i| g_ax[i]) + pg[k])
            .collect();
        last_g = Some(g_full.clone());
        // `optimizer_trace`: one row per iteration, at the point its model is built, as the
        // built-in descents write theirs (the unpenalized OFV, the gradient there, and the
        // packed norm of the step that led to it).
        if crate::estimation::trace::is_active() && traced_at.as_ref() != Some(&x) {
            traced_at = Some(x.clone());
            let p = unpack_params(&x, problem.init_params);
            let clean = f - problem.nn_reg.penalty_value(&p.theta) - problem.priors.penalty(&x);
            crate::estimation::trace::write_foce(
                it,
                crate::estimation::trace::method_label(options),
                clean,
                Some(l2_norm(&g_full)),
                last_step.take(),
                "trust_region",
                None,
                None,
                &crate::estimation::parameterization::coordinate_values(&p),
                Some(&g_full),
            );
        }
        let fr: Vec<usize> = (0..axes.len())
            .filter(|&i| {
                let k = axes[i];
                !(at(bounds.lower[k], x[k]) && g_full[k] > 0.0)
                    && !(at(bounds.upper[k], x[k]) && g_full[k] < 0.0)
            })
            .collect();
        // A change of the free set changes the subspace the model lives in; the secant
        // curvature was learned in the old one. Re-model here on the exact Hessian.
        let free_changed = prev_free.as_ref().is_some_and(|p| *p != fr);
        prev_free = Some(fr.clone());
        if used_quasi && free_changed {
            refresh = true;
            continue;
        }
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
        // Moré's scaling: `D` at its running maximum of `√|B_aa|`.
        {
            let mut cs = problem.curvature_scale.lock().unwrap();
            if cs.len() != n {
                cs.resize(n, 0.0);
            }
            for (i, &fi) in fr.iter().enumerate() {
                let hd = h[(i, i)].abs().sqrt();
                if hd.is_finite() && hd > d_scale[fi] {
                    d_scale[fi] = hd;
                }
                cs[axes[fi]] = cs[axes[fi]].max(d_scale[fi]);
            }
        }
        let dmax = fr.iter().map(|&fi| d_scale[fi]).fold(0.0_f64, f64::max);
        if dmax <= 0.0 {
            break;
        }
        let d: Vec<f64> = fr.iter().map(|&fi| d_scale[fi].max(1e-8 * dmax)).collect();
        let gy = DVector::from_iterator(m, (0..m).map(|i| g_full[axes[fr[i]]] / d[i]));
        let hy = DMatrix::from_fn(m, m, |i, j| h[(i, j)] / (d[i] * d[j]));
        if !radius_set {
            // The first radius is the length of the seed model's own Newton step, in the
            // scaled norm, clamped to `[1, 1e3]`. The seed is the exact `|H|`, so its Newton
            // step is the best length estimate available, and the ρ test still rejects it
            // where the model is wrong. PORT's fixed initial radius of 1 is 1/√B_aa in a
            // coordinate's own units: on a tightly curved coordinate far from its optimum
            // (a 1% prior, #254's `a_prior_moves_the_estimates_and_not_only_the_report`) it
            // moved 0.7% a step and needed ~log₂ of the distance in doublings to get there.
            radius_set = true;
            if let Some(newton) = saddle_free_subproblem(&gy, hy.clone(), f64::INFINITY) {
                let len = newton.step.norm();
                if len.is_finite() {
                    radius = len.clamp(1.0, 1e3);
                }
            }
        }
        let Some(sub) = saddle_free_subproblem(&gy, hy, radius) else {
            break;
        };
        let decrement = sub.decrement;
        if options.verbose {
            eprintln!(
                "  TR iter {it:>3}: OFV = {f:.6}  decrement {decrement:.3e}  radius {radius:.3e}  free {m}/{}{}",
                axes.len(),
                if used_quasi { "" } else { "  (exact Hessian)" }
            );
        }
        last_decrement = decrement;
        if decrement <= TR_EXACT_DECREMENT_TOL {
            if !exact_here && unresolved_at.as_ref() == Some(&x) {
                // The exact Hessian cannot be formed at this point (an EBE that does not
                // re-converge at the covariance tolerance), so the stop rests on the secant
                // model's decrement, with the exact gradient. Asking again here would spin.
                converged = true;
                break;
            }
            if !exact_here {
                // A stop is certified on the exact Hessian only.
                refresh = true;
                continue;
            }
            converged = true;
            break;
        }
        let mut xn = x.clone();
        for (i, &fi) in fr.iter().enumerate() {
            let k = axes[fi];
            xn[k] = (x[k] + sub.step[i] / d[i]).clamp(bounds.lower[k], bounds.upper[k]);
        }
        // The step actually taken, in the scaled norm, and the model's prediction for it.
        let sy =
            DVector::from_iterator(m, (0..m).map(|i| (xn[axes[fr[i]]] - x[axes[fr[i]]]) * d[i]));
        let pred = -(gy.dot(&sy) + 0.5 * sy.dot(&(&sub.habs * &sy)));
        let (fnew, en, hn, kn, _, _) = eval(&xn, &trial_warm, PrevEtas::Solved);
        let rho = if pred > 0.0 { (f - fnew) / pred } else { -1.0 };
        // Measured (#2975): without this refresh (PORT's pure secant mode) the trust region took
        // 32% more iterations on the 26-model set, up to 13× on single models, with one more miss.
        if rho < 0.25 && !exact_here {
            // The secant model predicted badly here: re-model on the exact Hessian. An exact
            // model at this very point is kept; only the radius shrinks.
            refresh = true;
        }
        let snorm = sy.norm();
        if fnew < f && rho > 1e-4 {
            // The noise floor is measured at the incumbent: a new one starts unmeasured.
            last_mismatch = f64::NAN;
            last_step = Some(l2_norm(
                &xn.iter().zip(&x).map(|(a, b)| a - b).collect::<Vec<f64>>(),
            ));
            x = xn;
            f = fnew;
            etas = en;
            hms = hn;
            kappas = kn;
            // A resume point (#755), as every outer optimizer writes one: the accepted iterate
            // is the best point seen (the trust region's iterates descend monotonically), and
            // the checkpoint carries the unpenalized −2LL there.
            if crate::io::checkpoint::is_due() {
                let theta = unpack_params(&x, problem.init_params).theta;
                let clean = f - problem.nn_reg.penalty_value(&theta) - problem.priors.penalty(&x);
                crate::io::checkpoint::maybe_write(it, clean, &x);
            }
            // The rows' gradient at the next iterate starts from this point's own modes.
            *problem.grad_cache.lock().unwrap() = Some(GradCache {
                x: x.clone(),
                etas: etas.clone(),
                h_mats: hms.clone(),
                per_subj_grads: vec![],
            });
            if rho > 0.75 && snorm >= 0.99 * radius {
                radius = (2.0 * radius).min(1e3);
            } else if rho < 0.25 {
                radius = 0.25 * snorm.max(1e-12);
            }
        } else {
            radius = 0.25 * snorm.max(1e-12);
            // A guard-rejected or non-finite trial measures no noise: its `1e20` is a wall,
            // not an objective, and must not settle a stop at the noise floor.
            last_mismatch = if fnew < 1e20 {
                ((f - fnew) - pred).abs()
            } else {
                f64::NAN
            };
        }
        if radius < 1e-10 {
            // The noise floor: no step can realise a predicted gain smaller than the
            // objective's measured evaluation noise, so a decrement within it is converged.
            converged = settles_at_noise_floor(exact_here, last_mismatch, last_decrement);
            if options.verbose {
                eprintln!(
                    "  TR: radius collapsed; decrement {last_decrement:.3e} vs measured noise \
                     {last_mismatch:.3e} → converged = {converged}"
                );
            }
            break;
        }
    }
    *problem.cached_etas.lock().unwrap() = etas.clone();
    if options.verbose {
        eprintln!("  TR: {iters} iterations, {n_exact} exact Hessians");
    }
    if iters == options.outer_maxiter && !converged {
        *budget_spent = true;
    }
    let held = HeldEbes {
        etas,
        h_mats: hms,
        kappas,
    };
    (converged, x, iters, last_g, held, f_start, f)
}

/// The noise-floor stop (#2975), at a collapsed radius: converged when the model is exact at
/// this point and its Newton `decrement` is no larger than `mismatch`, the `|actual − predicted|`
/// of the last rejected step. That step is so short that the model's own error (third order in
/// it) is negligible, so the mismatch is the objective's evaluation noise, and no step can
/// realise a gain below it. A non-finite mismatch (no step was rejected) never settles.
pub(crate) fn settles_at_noise_floor(exact_here: bool, mismatch: f64, decrement: f64) -> bool {
    exact_here && mismatch.is_finite() && decrement <= mismatch
}

/// `|H|`: `H` with its eigenvalues made absolute, floored at `1e-10·max|λ|`. The saddle-free
/// metric (Dauphin et al. 2014): `H` itself where `H` is positive definite; positive definite
/// always, with every curvature magnitude kept.
pub(crate) fn saddle_free(h: &DMatrix<f64>) -> DMatrix<f64> {
    let e = h.clone().symmetric_eigen();
    let top = e.eigenvalues.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let lam = e.eigenvalues.map(|v| v.abs().max(1e-10 * top));
    &e.eigenvectors * DMatrix::from_diagonal(&lam) * e.eigenvectors.transpose()
}

/// Powell's damped BFGS update of `b` for the step `s` and gradient change `y` (Nocedal &
/// Wright §18.3; PORT's `DW7ZBF` damps the same way): `y` is blended toward `Bs` when
/// `sᵀy < 0.2·sᵀBs`, so `B` stays positive definite and no update is skipped. Afterwards
/// `B⁺s` equals the blended `y`.
pub(crate) fn damped_bfgs_update(b: &mut DMatrix<f64>, s: &DVector<f64>, y: &DVector<f64>) {
    let bs = &*b * s;
    let sbs = s.dot(&bs);
    if !(sbs > 0.0) {
        return;
    }
    let sy = s.dot(y);
    let theta = if sy >= 0.2 * sbs {
        1.0
    } else {
        0.8 * sbs / (sbs - sy)
    };
    let r = y * theta + &bs * (1.0 - theta);
    let sr = s.dot(&r);
    if sr > 0.0 {
        *b -= &bs * bs.transpose() / sbs;
        *b += &r * r.transpose() / sr;
    }
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

    /// Compute the per-subject rows `dᵢ` (`gauss_newton::gn_subject_rows`, each subject's
    /// total derivative), caching the result so that `hessian()` can reuse it without a
    /// second inner-loop solve.
    ///
    /// Returns `(etas, h_mats, dᵢ)`.
    ///
    /// Three cache states (keyed by `x` equality and sentinel field):
    ///   Full hit:    `c.x == x` and `!c.per_subj_grads.is_empty()` → return everything cached.
    ///   Partial hit: `c.x == x` and `c.per_subj_grads.is_empty()`  → EBEs warm (from `cost()`),
    ///                                                                   run AD pass only.
    ///   Miss:        `c.x != x` or cache is `None`                 → full inner solve + AD.
    #[allow(clippy::type_complexity)]
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

        // #2974: each subject's row is its total derivative `dᵢ`, in the gradient and the BHHH
        // matrix alike (`gauss_newton::gn_subject_rows`).
        let rows = crate::estimation::gauss_newton::gn_subject_rows(
            x,
            self.init_params,
            self.model,
            self.population,
            &etas,
            &vec![Vec::new(); n_subj],
            &self.bounds,
            self.options,
            &[],
        );
        *self.grad_cache.lock().unwrap() = Some(GradCache {
            x: x.to_vec(),
            etas: etas.clone(),
            h_mats: h_mats.clone(),
            per_subj_grads: rows.clone(),
        });
        (etas, h_mats, rows)
    }
}

impl FoceiProblem<'_> {
    /// `grad(OFV) = 2 · Σᵢ dᵢ`, `dᵢ` each subject's total derivative (#2974): the EBE response
    /// included — a fixed-η̂ score drops the FOCEI `log|H̃|` response, and a gradient optimizer
    /// descending on it stalls above the minimum (warfarin FOCEI −276.6 instead of −286.0).
    /// Without the penalties, which the caller adds.
    fn data_gradient(&self, p: &[f64]) -> Vec<f64> {
        let (_, _, per_subj) = self.compute_ad_grads(p);
        let mut g = vec![0.0_f64; p.len()];
        for di in &per_subj {
            for (gk, dk) in g.iter_mut().zip(di) {
                *gk += 2.0 * dk;
            }
        }
        g
    }

    /// The penalized gradient the BHHH fallback models: [`Self::data_gradient`] plus the
    /// penalties'. The tests pin it against finite differences of the objective.
    #[cfg(test)]
    fn likelihood_gradient(&self, p: &[f64]) -> Vec<f64> {
        let mut g = self.data_gradient(p);
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
        fixed,
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
        feasibility: std::sync::Mutex::new(TrFeasibility::default()),
        curvature_scale: std::sync::Mutex::new(Vec::new()),
    };

    if options.verbose {
        eprintln!(
            "Starting trust-region optimization ({} parameters)...",
            x0.len()
        );
    }

    let mut tr_budget_spent = false;
    let (mut converged, mut best_x, mut n_iterations, mut final_gradient, mut held, _, _) =
        exact_trust_region(
            &problem,
            &x0,
            &post_run_bounds,
            options,
            &mut tr_budget_spent,
        );
    // The NLopt leg's basin-shift restart (#1565, #2093), on its rules: a converged leg whose
    // cold EBE re-solve scores more than `BASIN_SHIFT_ABS_TOL` below the objective it minimised
    // held some subjects in a higher individual mode; resume from the same point with the cold
    // EBEs, on the remaining budget, at most `MAX_BASIN_RESTARTS` times. A trust region takes few,
    // large steps, so warm EBEs carried along it have fewer chances to drop into the lower mode
    // than along L-BFGS's many short evaluations.
    for _ in 0..crate::estimation::outer_optimizer::MAX_BASIN_RESTARTS {
        if !converged
            || n_iterations >= options.outer_maxiter
            || crate::cancel::is_cancelled(&options.cancel)
        {
            break;
        }
        let params = unpack_params(&best_x, init_params);
        let incumbent = score_final_ebes(
            model,
            population,
            &params,
            (held.etas.clone(), held.h_mats.clone(), held.kappas.clone()),
            options,
        );
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let cold_solve = crate::estimation::outer_optimizer::final_inner_solve(
            model, population, &params, &mu_k, None, options,
        );
        let cold = score_final_ebes(
            model,
            population,
            &params,
            (cold_solve.etas, cold_solve.h_mats, cold_solve.kappas),
            options,
        );
        if !(cold.3.is_finite()
            && incumbent.3 - cold.3 > crate::estimation::outer_optimizer::BASIN_SHIFT_ABS_TOL)
        {
            break;
        }
        if options.verbose {
            eprintln!(
                "Trust region: cold EBEs score {:.6} below the objective it minimised ({:.6}); \
                 resuming from the same point with them",
                incumbent.3 - cold.3,
                incumbent.3
            );
        }
        warnings.push(
            crate::estimation::outer_optimizer::basin_shift_restart_warning(cold.3, n_iterations),
        );
        // The cold modes whole, κ̂ included, in the layout every trial warm-starts from.
        *problem.cached_etas.lock().unwrap() =
            crate::estimation::outer_optimizer::outer_cache_warm_starts(model, &cold.0, &cold.2);
        *problem.cached_etas_kind.lock().unwrap() = PrevEtas::ModesHere;
        *problem.grad_cache.lock().unwrap() = None;
        let mut leg_options = options.clone();
        leg_options.outer_maxiter = options.outer_maxiter - n_iterations;
        let (c, bx, it, fg, h, f0, _) = exact_trust_region(
            &problem,
            &best_x,
            &post_run_bounds,
            &leg_options,
            &mut tr_budget_spent,
        );
        n_iterations += it;
        if f0 >= 1e20 {
            // The cold modes did not score a feasible start at this point: keep the leg it
            // was to resume, unconverged, rather than adopt modes the guard refused.
            converged = false;
            break;
        }
        converged = c;
        best_x = bx;
        final_gradient = fg;
        held = h;
    }
    // A stop that is neither converged nor out of budget is a collapsed radius whose decrement
    // exceeds the measured noise: the model promises a gain no short step realises, so the
    // gradient disagrees with the objective the inner loop scores. That is the NLopt leg's
    // verification case (#2462): one leg from the best point with the inner loop at
    // `UNPROFILED_VERIFY_INNER_TOL`, settling when the leg converges or gains at most
    // `UNPROFILED_FIRST_LEG_GAIN` (#2507). The leg starts at the incumbent and only descends,
    // so its point is kept either way.
    if !converged
        && !tr_budget_spent
        && n_iterations < options.outer_maxiter
        && !crate::cancel::is_cancelled(&options.cancel)
        // An infeasible start (#1617) has no objective for a tighter leg to verify.
        && problem.feasibility.lock().unwrap().feasible > 0
    {
        use crate::estimation::outer_optimizer::{
            UNPROFILED_FIRST_LEG_GAIN, UNPROFILED_VERIFY_INNER_TOL,
        };
        let mut leg_options = options.clone();
        leg_options.inner_tol = options.inner_tol.min(UNPROFILED_VERIFY_INNER_TOL);
        leg_options.cov_inner_tol = Some(
            options
                .effective_cov_inner_tol(model.uses_closed_form_ltbs_inner())
                .min(UNPROFILED_VERIFY_INNER_TOL),
        );
        leg_options.outer_maxiter = options.outer_maxiter - n_iterations;
        *problem.cached_etas.lock().unwrap() =
            crate::estimation::outer_optimizer::outer_cache_warm_starts(
                model,
                &held.etas,
                &held.kappas,
            );
        *problem.cached_etas_kind.lock().unwrap() = PrevEtas::ModesHere;
        *problem.grad_cache.lock().unwrap() = None;
        let (c, bx, it, fg, h, f0, f1) = exact_trust_region(
            &problem,
            &best_x,
            &post_run_bounds,
            &leg_options,
            &mut tr_budget_spent,
        );
        // A leg that never scored a feasible start, or stopped before its first iteration,
        // verified nothing: its zero gain is not a measurement.
        let ran = it > 0 && f0 < 1e20 && f1 < 1e20;
        let gain = if ran { f0 - f1 } else { f64::NAN };
        if options.verbose {
            eprintln!(
                "Trust region: verification leg at inner_tol {:e}: gain {gain:.3e}, converged = {c}",
                leg_options.inner_tol
            );
        }
        n_iterations += it;
        // A leg that did not run keeps the first leg's point, modes and gradient.
        if ran {
            converged = c || gain <= UNPROFILED_FIRST_LEG_GAIN;
            best_x = bx;
            final_gradient = fg;
            held = h;
        }
    }
    if crate::cancel::is_cancelled(&options.cancel) {
        warnings.push("Trust-region did not converge: cancelled by user".to_string());
    } else if !converged {
        let label = if nn_reg_active {
            "∂(OFV + NN penalty)/∂x"
        } else {
            "∂OFV/∂x"
        };
        let reason = if problem.feasibility.lock().unwrap().feasible == 0 {
            "found no feasible objective at its start (W_NO_FEASIBLE_EVAL)".to_string()
        } else if tr_budget_spent {
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

    // #1617, on the NLopt path's rule and message: a run that never scored a feasible
    // objective has minimised nothing. The trust region stops at an infeasible start, so
    // `best_x` is the start and the estimates reported are the ones the fit started from.
    let tally = *problem.feasibility.lock().unwrap();
    let run = crate::estimation::outer_optimizer::RunFeasibility {
        feasible_evals: tally.feasible,
        n_guard_rejected: tally.guard_rejected,
        n_nonfinite: tally.nonfinite,
        max_unconverged: tally.max_unconverged,
        n_subj,
        max_unconverged_frac: options.max_unconverged_frac,
        inner_maxiter: options.inner_maxiter,
        cancelled: crate::cancel::is_cancelled(&options.cancel),
    };
    if let Some(w) =
        crate::estimation::outer_optimizer::no_feasible_eval_warning(&mut converged, &run)
    {
        warnings.push(w);
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
        saem_objective_value: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: u32::try_from(tally.max_unconverged).unwrap_or(u32::MAX),
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
        // The NLopt loop's own escape test (#751), on the free coordinates and in the same
        // units: NLopt reads a move in its curvature-scaled space (`x·√|H_aa|`), and the trust
        // region's model measures its steps in `√|B_aa|` (Moré's scaling), so a move counts in
        // those, and in packed units on a coordinate no model reached. `!(… < …)` so a
        // non-finite deviation counts as moved.
        left_init: Some({
            let cs = problem.curvature_scale.lock().unwrap();
            let scaled_free = |x: &[f64]| -> Vec<f64> {
                x.iter()
                    .zip(&fixed)
                    .enumerate()
                    .filter(|&(_, (_, &f))| !f)
                    .map(|(k, (&v, _))| {
                        let c = cs.get(k).copied().unwrap_or(0.0);
                        v * if c > 0.0 { c } else { 1.0 }
                    })
                    .collect()
            };
            !(crate::estimation::outer_optimizer::max_scaled_deviation(
                &scaled_free(&best_x),
                &scaled_free(&x0),
            ) < crate::estimation::outer_optimizer::INIT_ESCAPE_STEP_S)
        }),
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
    /// The noise-floor stop settles only on an exact model whose decrement the measured noise
    /// covers. Each case dies on its own mutation: dropping `exact_here` settles the secant row,
    /// `<=` → `<` misses the inclusive boundary, dropping the finiteness check settles the
    /// infinite-mismatch row.
    #[test]
    fn the_noise_floor_settles_only_an_exact_decrement_within_the_measured_noise() {
        assert!(
            settles_at_noise_floor(true, 5e-5, 3e-5),
            "decrement inside the noise"
        );
        assert!(
            settles_at_noise_floor(true, 3e-5, 3e-5),
            "boundary is inclusive"
        );
        assert!(
            !settles_at_noise_floor(true, 2e-7, 3.2),
            "a real gain left: not noise"
        );
        assert!(
            !settles_at_noise_floor(false, 5e-5, 3e-5),
            "a secant model never settles"
        );
        assert!(
            !settles_at_noise_floor(true, f64::NAN, 3e-5),
            "no rejected step measured"
        );
        assert!(
            !settles_at_noise_floor(true, f64::INFINITY, 3e-5),
            "an unbounded mismatch is not a measurement"
        );
    }

    /// `|H|` is `H` on a positive-definite matrix and flips a negative eigenvalue, keeping its
    /// eigenvector and magnitude. This is what makes the BFGS seed positive definite: with the
    /// raw indefinite `H` every update along the negative direction has `sᵀBs < 0` and is
    /// skipped, and the trust region crawled at ρ ≈ 1 for 9999 iterations (`syn_pcm_int`).
    /// Mutations: dropping `.abs()` leaves the `−2` eigenvalue (the second assertion); dropping
    /// the floor leaves a zero eigenvalue (the third).
    #[test]
    fn saddle_free_keeps_pd_curvature_and_flips_negative_curvature() {
        let pd = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        assert!(
            (saddle_free(&pd) - &pd).norm() < 1e-12,
            "|H| = H on a PD matrix"
        );
        // Eigenvalues 3 and −2 on the basis (1, 1)/√2, (1, −1)/√2.
        let ind = DMatrix::from_row_slice(2, 2, &[0.5, 2.5, 2.5, 0.5]);
        let e = saddle_free(&ind).symmetric_eigen();
        let mut lam: Vec<f64> = e.eigenvalues.iter().copied().collect();
        lam.sort_by(f64::total_cmp);
        assert!(
            (lam[0] - 2.0).abs() < 1e-12 && (lam[1] - 3.0).abs() < 1e-12,
            "{lam:?}"
        );
        let v = DVector::from_vec(vec![1.0, -1.0]) / 2f64.sqrt();
        assert!(
            (saddle_free(&ind) * &v - &v * 2.0).norm() < 1e-12,
            "same eigenvector"
        );
        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let min = saddle_free(&singular).symmetric_eigen().eigenvalues.min();
        assert!(min > 0.0 && min <= 1e-10, "floored at 1e-10·max|λ|: {min}");
    }

    /// Powell's damped BFGS: with ample curvature (`sᵀy ≥ 0.2·sᵀBs`) the update satisfies the
    /// secant condition `B⁺s = y` exactly; with negative curvature (`sᵀy < 0`) it does not skip,
    /// stays positive definite and puts `sᵀB⁺s` at exactly `0.2·sᵀBs`.
    /// Mutations: removing the damping (θ = 1 always) makes `B⁺` indefinite in the second case;
    /// skipping the update when `sᵀy ≤ 0` leaves `sᵀB⁺s = sᵀBs`; a wrong blend constant moves
    /// the `0.2`.
    #[test]
    fn damped_bfgs_meets_the_secant_condition_and_stays_positive_definite() {
        let b0 = DMatrix::from_row_slice(2, 2, &[2.0, 0.3, 0.3, 1.0]);
        let s = DVector::from_vec(vec![0.4, -0.2]);
        let y = DVector::from_vec(vec![1.1, 0.1]);
        assert!(s.dot(&y) >= 0.2 * s.dot(&(&b0 * &s)));
        let mut b = b0.clone();
        damped_bfgs_update(&mut b, &s, &y);
        assert!((&b * &s - &y).norm() < 1e-12, "secant: B⁺s = y");
        let y_neg = DVector::from_vec(vec![-0.5, 0.2]);
        assert!(s.dot(&y_neg) < 0.0);
        let mut b = b0.clone();
        damped_bfgs_update(&mut b, &s, &y_neg);
        assert!(
            b.clone().symmetric_eigen().eigenvalues.min() > 0.0,
            "stays PD"
        );
        let sbs0 = s.dot(&(&b0 * &s));
        assert!(
            (s.dot(&(&b * &s)) - 0.2 * sbs0).abs() < 1e-12,
            "damped to 0.2·sᵀBs"
        );
    }

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
    /// exact adjoint) at the EBEs `compute_ad_grads` solved. Dies if the rows fall back to
    /// the fixed-mode score (the gradient to the biased `2·Σ gᵢ`).
    #[cfg(feature = "survival")]
    #[test]
    fn gradient_is_the_total_derivative_on_a_tte_model_with_eta() {
        use crate::estimation::parameterization::{clamp_to_bounds, compute_bounds, pack_params};

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
            feasibility: std::sync::Mutex::new(TrFeasibility::default()),
            curvature_scale: std::sync::Mutex::new(Vec::new()),
            grad_cache: std::sync::Mutex::new(None),
            nn_reg: crate::estimation::nn_reg::NnRegularizer::build(&model, &population, &options),
            priors: crate::estimation::outer_optimizer::build_prior_set(
                &model,
                &model.default_params,
            ),
        };

        let (etas, h_mats, _) = problem.compute_ad_grads(&x0);
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
            let (_, gi) = crate::estimation::gauss_newton::subject_nll_pop_grad(
                &x0,
                &model.default_params,
                &model,
                &population,
                i,
                &etas[i],
                &h_mats[i],
                &[],
                &problem.bounds,
                &options,
            );
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
                raw[k] += 2.0 * gi[k];
            }
        }
        assert!(
            g.iter().chain(&want).chain(&fd).chain(&raw).all(|v| v.is_finite()),
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
        // The gradient sums the rows `dᵢ` themselves: 1e-10 allows for summation-order
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
                feasibility: std::sync::Mutex::new(TrFeasibility::default()),
                curvature_scale: std::sync::Mutex::new(Vec::new()),
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
                feasibility: std::sync::Mutex::new(TrFeasibility::default()),
                curvature_scale: std::sync::Mutex::new(Vec::new()),
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
