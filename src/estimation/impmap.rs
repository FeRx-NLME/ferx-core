//! IMPMAP — Importance Sampling assisted by Mode A Posteriori (NONMEM
//! `METHOD=IMPMAP`).
//!
//! A Monte-Carlo EM (MCEM) **estimator**. Each iteration:
//!
//! 1. **E-step part A (MAP):** re-evaluate every subject's conditional mode η̂ᵢ
//!    and first-order-variance Hessian `Hᵢ = Jᵢᵀ Rᵢ⁻¹ Jᵢ + Ω⁻¹` at the current
//!    parameters (the FOCE/ITS inner loop). This re-centering each iteration —
//!    rather than only on the first, as plain `IMP` does — is what makes IMPMAP
//!    robust on high-dimensional, rich-data problems where IMP can stall.
//! 2. **E-step part B (IS):** draw `K` importance samples ηᵢₖ from a proposal
//!    centred at η̂ᵢ with scale `Σᵢ = Hᵢ⁻¹` (multivariate normal by default;
//!    Student-t with `impmap_proposal_df`), with self-normalized weights w̃ᵢₖ.
//! 3. **M-step:** update parameters from the importance-weighted complete-data
//!    expectation:
//!    - **Ω** closed form: `Ω = (1/N) Σᵢ Σₖ w̃ᵢₖ ηᵢₖ ηᵢₖᵀ`.
//!    - **θ, σ** by maximizing the weighted observation likelihood
//!      `Σᵢ Σₖ w̃ᵢₖ log p(yᵢ | ηᵢₖ, θ, σ)` (derivative-free NLopt BOBYQA in
//!      packed log-space, warm-started from the previous iteration).
//!
//! The reported estimate is the running mean of the parameter vector over the
//! final `impmap_averaging` iterations (Monte-Carlo variance reduction). The
//! returned [`OuterResult`] carries the final EBEs / Jacobians and a FOCE
//! Laplace `ofv` for AIC/BIC comparability, identical in shape to SAEM's, so the
//! covariance step and chained-stage handoff in `api.rs` need no special casing.
//!
//! ## Scope (v1)
//!
//! Inter-occasion variability (`κ` / `[iov]`) is **not yet supported** by the
//! IMPMAP M-step (the κ sufficient statistics and Ω_iov update are a follow-up);
//! such models are refused up front. SDE / `[diffusion]` models are refused for
//! the same reason `IMP` refuses them. Use SAEM or FOCEI for those.

use crate::estimation::em_common::{
    floor_omega_diagonal, get_mu_ref_pairs, unpack_thetas, PackedThetaSigma, OMEGA_DIAG_FLOOR,
};
use crate::estimation::importance_sampling::{compute_posterior_hessian, subject_is_draws};
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{
    compute_covariance, pop_nll, CovarianceStepResult, OuterResult,
};
use crate::estimation::parameterization::{compute_mu_k, pack_params};
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Run IMPMAP. `warm_etas`, when supplied by a preceding chain stage, seed the
/// first MAP inner loop; otherwise the inner loop cold-starts from η = 0.
pub fn run_impmap(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // ---- Validation ----
    if n_eta == 0 {
        return Err("IMPMAP requires at least one random effect (n_eta = 0). \
             Use FOCE/FOCEI for fixed-effects-only models."
            .to_string());
    }
    if model.is_sde() {
        return Err("IMPMAP is not yet supported for SDE / [diffusion] models \
             (the EKF process-noise variance is not threaded through the IS \
             observation likelihood). Use FOCE / FOCEI instead."
            .to_string());
    }
    if model.n_kappa > 0 {
        return Err(
            "IMPMAP does not yet support inter-occasion variability (κ / [iov]); \
             the IOV M-step is a planned follow-up. Use SAEM or FOCEI for IOV models."
                .to_string(),
        );
    }
    if !init_params.omega.log_det.is_finite() {
        return Err(
            "IMPMAP: initial Ω log-determinant is not finite — check the \
             [parameters] Ω block."
                .to_string(),
        );
    }

    let n_iter = options.impmap_iterations.max(1);
    let k_samples = options.impmap_samples.max(2);
    // `INFINITY` selects the multivariate-normal proposal; any finite value must
    // be a valid Student-t DoF (>= 1). Guard here so a programmatic caller that
    // bypasses the parser's range check can't reach the `ChiSquared::new(nu)`
    // panic in `subject_is_draws`. Mirrors `run_importance_sampling`.
    let nu = options.impmap_proposal_df;
    if nu.is_finite() && nu < 1.0 {
        return Err(format!(
            "IMPMAP: impmap_proposal_df must be >= 1.0 (or +inf for a normal proposal), got {nu}"
        ));
    }
    let n_avg = options.impmap_averaging.min(n_iter);
    let seed = options.impmap_seed.unwrap_or(12345);
    let threshold = options.impmap_low_ess_threshold;
    let verbose = options.verbose;
    let cancel = &options.cancel;

    if verbose {
        let prop = if nu.is_finite() {
            format!("t_{nu}")
        } else {
            "normal".to_string()
        };
        eprintln!(
            "IMPMAP: {} subjects, {} ETAs, {} iters, K={}/subject, {} proposal, seed={}",
            n_subjects, n_eta, n_iter, k_samples, prop, seed
        );
    }

    // ---- Packing scaffolding (shared with SAEM via em_common) ----
    let PackedThetaSigma {
        theta_packs_log_mask,
        mut log_theta,
        mut log_sigma,
        log_theta_lower,
        log_theta_upper,
        log_sigma_lower,
        log_sigma_upper,
    } = PackedThetaSigma::new(init_params);

    // Closed-form mu-referencing M-step: shift `log(θ) += mean(η)` for log-mu-ref
    // pairs, with those θ pinned out of the NLopt weighted M-step (which then fits
    // only σ and non-mu-ref θ). This is the EM-correct typical-value update for
    // log-normal random effects, NOT an optional refinement: without it θ and the
    // η mean are confounded over the fixed importance samples, so θ stays at its
    // start and Ω inflates to absorb the misfit. It is therefore applied whenever
    // log-mu-ref pairs exist, independent of `options.mu_referencing` (which only
    // governs inner-loop `compute_mu_k` centering, a separate concern). NONMEM's
    // EM methods likewise require mu-referencing.
    let mut warnings: Vec<String> = Vec::new();
    let mu_ref_pairs = get_mu_ref_pairs(model);
    let use_closed_form = !mu_ref_pairs.is_empty();
    if !use_closed_form {
        // No log-mu-ref parameter: every typical value goes through the weighted
        // M-step, which cannot resolve the θ/η-mean confounding on its own. Flag
        // it — estimates may be unreliable (see the docs caveat).
        warnings.push(
            "IMPMAP: no log-mu-referenced parameters found (e.g. `CL = TVCL*exp(ETA)`); \
             typical-value estimation relies on the weighted M-step alone and may converge \
             poorly. Prefer a log-mu-referenced parameterization, or use FOCEI."
                .to_string(),
        );
    }

    // ---- Iteration state ----
    let mut theta_cur = init_params.theta.clone();
    let mut sigma_cur = init_params.sigma.values.clone();
    let mut omega_mat = init_params.omega.matrix.clone();
    let mut prev_etas: Option<Vec<DVector<f64>>> = warm_etas.map(|e| e.to_vec());

    // Running mean of parameters over the final `n_avg` iterations.
    let mut acc_theta = vec![0.0f64; n_theta];
    let mut acc_sigma = vec![0.0f64; n_sigma];
    let mut acc_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut n_acc = 0usize;

    // Reusable per-iteration parameter bundle. Only `theta`, `sigma.values`, and
    // `omega` change between iterations; the invariant metadata (names, bounds,
    // fixed masks) is cloned once here and left untouched, instead of rebuilding
    // the full struct every iteration.
    let mut params_k = ModelParameters {
        theta: theta_cur.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        ),
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: sigma_cur.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: None,
        kappa_fixed: init_params.kappa_fixed.clone(),
    };

    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(cancel) {
            if verbose {
                eprintln!("IMPMAP: cancelled at iteration {}", k);
            }
            break;
        }

        // Refresh only the three mutated pieces for this iteration (the rest of
        // `params_k` is invariant — see its construction above).
        params_k.theta.clone_from(&theta_cur);
        params_k.sigma.values.clone_from(&sigma_cur);
        params_k.omega = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );

        // ---- E-step A: MAP recenter (conditional mode + Jacobian) ----
        let mu_k = compute_mu_k(model, &params_k.theta, options.mu_referencing);
        let (eta_hats, h_matrices, _stats, _kappas) = run_inner_loop_warm(
            model,
            population,
            &params_k,
            options.inner_maxiter,
            options.inner_tol,
            prev_etas.as_deref(),
            Some(&mu_k),
            0,
        );

        // Borrow rather than clone: `params_k` is only read for the rest of this
        // iteration, so the n_eta×n_eta inverse need not be copied per iteration.
        let omega_inv = &params_k.omega.inv;
        let log_det_omega = params_k.omega.log_det;

        // ---- E-step B: importance sampling around each mode ----
        let draws: Vec<_> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                let h_post = compute_posterior_hessian(
                    model,
                    subject,
                    &params_k.theta,
                    &eta_hats[i],
                    &params_k.sigma.values,
                    &h_matrices[i],
                    omega_inv,
                    n_eta,
                    scratch,
                );
                subject_is_draws(
                    model,
                    subject,
                    &params_k.theta,
                    &params_k.sigma.values,
                    &eta_hats[i],
                    &h_post,
                    omega_inv,
                    log_det_omega,
                    n_eta,
                    k_samples,
                    nu,
                    seed.wrapping_add(i as u64).wrapping_add((k as u64) << 32),
                    scratch,
                )
            })
            .collect();

        // ESS diagnostics + marginal log-likelihood for the trace.
        let mut ll = 0.0f64;
        let mut n_low_ess = 0usize;
        for d in &draws {
            ll += d.log_marginal;
            if d.ess_fraction < threshold {
                n_low_ess += 1;
            }
        }
        let minus2ll = -2.0 * ll;

        // ---- M-step Ω: weighted second moment, structurally masked + floored ----
        let mut new_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
        for d in &draws {
            new_omega += &d.second_moment;
        }
        new_omega /= n_subjects as f64;
        for i in 0..n_eta {
            for j in 0..n_eta {
                if !init_params.omega.free_mask[(i, j)] {
                    new_omega[(i, j)] = 0.0;
                }
                let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    new_omega[(i, j)] = init_params.omega.matrix[(i, j)];
                }
            }
        }
        floor_omega_diagonal(&mut new_omega, &init_params.omega_fixed, OMEGA_DIAG_FLOOR);
        omega_mat = new_omega;

        // ---- M-step σ + non-mu-ref θ: maximize weighted observation likelihood ----
        // Pin the log-mu-ref θ (handled by the closed-form shift below) so NLopt
        // optimizes only σ and any non-mu-ref θ, using the θ_old-centered samples.
        let mut mstep_theta_lower = log_theta_lower.clone();
        let mut mstep_theta_upper = log_theta_upper.clone();
        if use_closed_form {
            for &(t, _e) in &mu_ref_pairs {
                mstep_theta_lower[t] = log_theta[t];
                mstep_theta_upper[t] = log_theta[t];
            }
        }
        let mstep_maxiter: u32 = if k <= n_iter / 2 { 4 } else { 8 };
        let (new_log_theta, new_log_sigma) = theta_sigma_weighted_mstep(
            model,
            population,
            &draws,
            &log_theta,
            &log_sigma,
            &mstep_theta_lower,
            &mstep_theta_upper,
            &log_sigma_lower,
            &log_sigma_upper,
            n_theta,
            n_sigma,
            mstep_maxiter,
            &theta_packs_log_mask,
        );
        log_theta = new_log_theta;
        log_sigma = new_log_sigma;

        // ---- Closed-form mu-ref θ shift: log(θ) += population mean(η) ----
        if use_closed_form {
            let mut eta_bar = vec![0.0f64; n_eta];
            for d in &draws {
                for (acc, &m) in eta_bar.iter_mut().zip(d.mean.iter()) {
                    *acc += m;
                }
            }
            for acc in eta_bar.iter_mut() {
                *acc /= n_subjects as f64;
            }
            for &(t, e) in &mu_ref_pairs {
                log_theta[t] =
                    (log_theta[t] + eta_bar[e]).clamp(log_theta_lower[t], log_theta_upper[t]);
            }
        }

        theta_cur = unpack_thetas(&log_theta, &theta_packs_log_mask);
        sigma_cur = log_sigma.iter().map(|&s| s.exp()).collect();

        // Warm-start next iteration's inner loop from this iteration's modes.
        // `prev_etas` also carries the final modes out for the post-loop EBE pass.
        prev_etas = Some(eta_hats);

        // ---- Parameter averaging over the final n_avg iterations ----
        if k > n_iter - n_avg {
            for i in 0..n_theta {
                acc_theta[i] += theta_cur[i];
            }
            for i in 0..n_sigma {
                acc_sigma[i] += sigma_cur[i];
            }
            acc_omega += &omega_mat;
            n_acc += 1;
        }

        if verbose && (k <= 5 || k % 10 == 0 || k == n_iter) {
            eprintln!(
                "  iter {:4}: -2logL(IS) = {:.4}  (low-ESS subjects: {})",
                k, minus2ll, n_low_ess
            );
        }
    }

    if crate::cancel::is_cancelled(cancel) {
        return Err("cancelled by user".to_string());
    }

    // ---- Final (averaged) parameters ----
    let (final_theta, final_sigma, final_omega_mat) = if n_acc > 0 {
        let t: Vec<f64> = acc_theta.iter().map(|&v| v / n_acc as f64).collect();
        let s: Vec<f64> = acc_sigma.iter().map(|&v| v / n_acc as f64).collect();
        let o = acc_omega / n_acc as f64;
        (t, s, o)
    } else {
        (theta_cur.clone(), sigma_cur.clone(), omega_mat.clone())
    };

    // Reuse the per-iteration bundle for the final params — only the three
    // mutated pieces differ from the invariant metadata already in `params_k`.
    let mut final_params = params_k;
    final_params.theta = final_theta;
    final_params.sigma.values = final_sigma;
    final_params.omega = OmegaMatrix::from_matrix(
        final_omega_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );

    // ---- Final EBEs (warm-started) + FOCE Laplace OFV for comparability ----
    // `prev_etas` holds the final iteration's modes. `n_iter = ...max(1)` (above)
    // guarantees the loop body ran at least once and overwrote `prev_etas`, so
    // this is never the caller's stale seed (or `None`). If a future change ever
    // lets the body be skipped, this assert flags that the warm start below would
    // silently fall back to etas computed under the *initial* params.
    debug_assert!(
        n_iter >= 1,
        "IMPMAP final warm-start assumes the iteration loop ran at least once"
    );
    let warm = prev_etas.as_deref();
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _stats, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        warm,
        Some(&final_mu_k),
        0,
    );

    let ofv = 2.0
        * pop_nll(
            model,
            population,
            &final_params,
            &eta_hats,
            &h_matrices,
            &final_kappas,
            options.interaction,
        );

    // ---- Covariance step ----
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            let packed = pack_params(&final_params);
            match compute_covariance(
                &packed,
                &final_params,
                model,
                population,
                &eta_hats,
                &h_matrices,
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

    if verbose {
        eprintln!("IMPMAP completed. Final OFV (Laplace) = {:.4}", ofv);
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        // IMPMAP runs a fixed iteration schedule (no parameter-stabilization
        // stopping test yet), so the only convergence signal we can honestly
        // report is a finite final objective — a non-finite OFV means the MCEM
        // diverged. Matches SAEM's `converged: ofv.is_finite()`; importantly it
        // keeps a diverged run from being preferred in multi-start selection.
        converged: ofv.is_finite(),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
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
    })
}

/// Weighted θ/σ M-step: minimize the importance-weighted observation NLL
/// `Σᵢ Σₖ w̃ᵢₖ · obs_nll(yᵢ | ηᵢₖ, θ, σ)` over the per-subject sample sets, using
/// derivative-free NLopt BOBYQA in packed log-space, warm-started from the
/// current parameters. Mirrors SAEM's `theta_sigma_mstep_light` but sums over
/// the `K` weighted samples per subject instead of a single EBE.
#[allow(clippy::too_many_arguments)]
fn theta_sigma_weighted_mstep(
    model: &CompiledModel,
    population: &Population,
    draws: &[crate::estimation::importance_sampling::SubjectDraws],
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
    theta_packs_log_mask: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    // Weighted observation NLL, parallel over subjects. Each subject contributes
    // Σₖ w̃ₖ · obs_nll(yᵢ | ηᵢₖ, θ, σ).
    let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta], theta_packs_log_mask);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let val: f64 = population
            .subjects
            .par_iter()
            .zip(draws.par_iter())
            .map_init(EventPkParams::default, |scratch, (subject, d)| {
                let mut s = 0.0f64;
                for (w, eta) in d.weights.iter().zip(d.samples()) {
                    if *w == 0.0 {
                        continue;
                    }
                    s += w * obs_nll_subject_into(model, subject, &th, &sg, eta, scratch);
                }
                s
            })
            .sum();
        if val.is_finite() {
            val
        } else {
            1e20
        }
    };

    let mut opt = nlopt::Nlopt::new(
        nlopt::Algorithm::Bobyqa,
        n,
        obj,
        nlopt::Target::Minimize,
        (),
    );
    opt.set_lower_bounds(&lower).unwrap();
    opt.set_upper_bounds(&upper).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(1e-4).unwrap();

    let mut xs = x.clone();
    match opt.optimize(&mut xs) {
        Ok(_) | Err(_) => {}
    }

    let log_theta_new = xs[..n_theta].to_vec();
    let log_sigma_new = xs[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}
