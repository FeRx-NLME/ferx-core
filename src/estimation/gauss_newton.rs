/// Gauss-Newton optimizer for FOCE estimation.
///
/// Instead of the standard approach (separate inner/outer loops with first-order
/// gradient methods), this uses a coupled Gauss-Newton step that exploits the
/// nonlinear-least-squares structure of the FOCE objective:
///
///   OFV = sum_i [ r_i^T W_i^{-1} r_i + log|W_i| ]
///
/// where r_i are the weighted residuals and W_i = R_tilde_i is the linearized
/// covariance for subject i. The Gauss-Newton approximation uses J^T W^{-1} J
/// as the approximate Hessian (dropping second-derivative terms), giving
/// quadratic convergence near the minimum.
///
/// This approach mirrors NONMEM's modified Gauss-Newton algorithm and typically
/// converges in 10-30 iterations vs 100+ for first-order methods.
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::pop_nll;
use crate::estimation::outer_optimizer::{compute_covariance, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{foce_subject_nll_interaction, foce_subject_nll_standard};
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Run FOCE estimation using a Gauss-Newton optimizer.
///
/// Returns the same `OuterResult` as `optimize_population`.
pub fn run_foce_gn(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let n_subj = population.subjects.len();
    let _n_eta = model.n_eta;
    let verbose = options.verbose;
    let maxiter = options.outer_maxiter;
    let mut lambda = options.gn_lambda; // LM damping factor

    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let n_packed = x.len();
    let fixed_mask = packed_fixed_mask(init_params);

    // Scaling: computed once from initial x; x itself stays in real packed space
    // throughout the GN loop. Scaling only affects the linear system solve so
    // the Hessian is better conditioned when log-space values differ in magnitude.
    let gn_scale: Vec<f64> = if options.scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n_packed]
    };

    let mut warnings = Vec::new();

    // BHHH Information-matrix approximation degrades as the censoring fraction
    // grows — each BLOQ row contributes less Fisher information than its
    // Gaussian counterpart, biasing the outer-product Hessian small-sample.
    if matches!(model.bloq_method, BloqMethod::M3)
        && population.subjects.iter().any(|s| s.has_bloq())
    {
        warnings.push(
            "Gauss-Newton (BHHH) approximation may be inaccurate with M3 BLOQ handling; \
             consider method=foce_i for heavy BLOQ fractions (>20%)."
                .to_string(),
        );
    }

    if verbose {
        eprintln!("Starting FOCE Gauss-Newton estimation...");
        eprintln!("  {} subjects, {} observations", n_subj, population.n_obs());
        eprintln!("  {} packed parameters, lambda={:.4}", n_packed, lambda);
    }

    // Initial inner loop
    let params = unpack_params(&x, init_params);
    let init_mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let (mut eta_hats, mut h_matrices, _, mut kappas) = run_inner_loop_warm(
        model,
        population,
        &params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&init_mu_k),
        options.min_obs_for_convergence_check as usize,
    );

    let mut ofv = 2.0
        * pop_nll(
            model,
            population,
            &params,
            &eta_hats,
            &h_matrices,
            &kappas,
            options.interaction,
        );

    if verbose {
        eprintln!("  GN iter {:>3}: OFV = {:.6}", 0, ofv);
    }

    let mut converged = false;

    for iter in 1..=maxiter {
        if crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("  GN iter {:>3}: cancelled by user", iter);
            }
            warnings.push("cancelled by user".to_string());
            break;
        }

        // ---- Build the BHHH system ----
        // Gradient + outer-product Hessian approximation
        let (mut grad, mut h_bhhh) = build_gn_system(
            &x,
            init_params,
            model,
            population,
            &eta_hats,
            &h_matrices,
            &kappas,
            &bounds,
            options,
        );

        // Zero gradient rows / BHHH rows & cols for FIX parameters, and set
        // their diagonal to 1. The clamp at step-application keeps x[i] at its
        // pinned value; this form guarantees the Cholesky solve gives
        // `delta[i] = 0` exactly (rather than relying on the clamp to hide a
        // large, meaningless step).
        for i in 0..n_packed {
            if fixed_mask[i] {
                grad[i] = 0.0;
                for j in 0..n_packed {
                    h_bhhh[(i, j)] = 0.0;
                    h_bhhh[(j, i)] = 0.0;
                }
                h_bhhh[(i, i)] = 1.0;
            }
        }

        // ---- Scale the linear system (better Hessian conditioning) ----
        // g_s[i] = g[i] * scale[i],  H_s[i,j] = H[i,j] * scale[i] * scale[j]
        // Solve H_s_lm * delta_s = -g_s, then delta[i] = delta_s[i] * scale[i].
        // With identity scale (scale_params=false) this is a no-op.
        let grad_s: DVector<f64> =
            DVector::from_iterator(n_packed, (0..n_packed).map(|i| grad[i] * gn_scale[i]));
        let mut h_s = DMatrix::zeros(n_packed, n_packed);
        for i in 0..n_packed {
            for j in 0..n_packed {
                h_s[(i, j)] = h_bhhh[(i, j)] * gn_scale[i] * gn_scale[j];
            }
        }

        // ---- Levenberg-Marquardt damping (in scaled space) ----
        let mut h_s_lm = h_s.clone();
        for i in 0..n_packed {
            h_s_lm[(i, i)] += lambda * h_s[(i, i)].max(1e-8);
        }

        // ---- Solve for step: H_s_lm * delta_s = -grad_s ----
        let neg_grad_s = -&grad_s;
        let chol = h_s_lm.clone().cholesky();
        let delta_s = match chol {
            Some(c) => c.solve(&neg_grad_s),
            None => {
                // Fall back to regularized pseudo-inverse
                if verbose {
                    eprintln!("  GN iter {:>3}: Hessian singular, increasing lambda", iter);
                }
                lambda *= 10.0;
                continue;
            }
        };
        // Convert step back to real (unscaled) space
        let delta =
            DVector::from_iterator(n_packed, (0..n_packed).map(|i| delta_s[i] * gn_scale[i]));

        // ---- Line search with backtracking ----
        let mut alpha = 1.0;
        let mut x_new = x.clone();
        let mut ofv_new = f64::INFINITY;
        let mut eta_new = eta_hats.clone();
        let mut h_new = h_matrices.clone();

        for _ls in 0..15 {
            // Take step
            for i in 0..n_packed {
                x_new[i] = (x[i] + alpha * delta[i]).clamp(bounds.lower[i], bounds.upper[i]);
            }

            let params_try = unpack_params(&x_new, init_params);

            // Re-estimate EBEs at new parameters (warm-started)
            let ls_mu_k = compute_mu_k(model, &params_try.theta, options.mu_referencing);
            let (eh, hm, _, kap_new) = run_inner_loop_warm(
                model,
                population,
                &params_try,
                options.inner_maxiter,
                options.inner_tol,
                Some(&eta_new),
                Some(&ls_mu_k),
                options.min_obs_for_convergence_check as usize,
            );

            let ofv_try = 2.0
                * pop_nll(
                    model,
                    population,
                    &params_try,
                    &eh,
                    &hm,
                    &kap_new,
                    options.interaction,
                );

            if ofv_try.is_finite() && ofv_try < ofv {
                ofv_new = ofv_try;
                eta_new = eh;
                h_new = hm;
                kappas = kap_new;
                break;
            }

            alpha *= 0.5;
        }

        if ofv_new >= ofv {
            // Step failed — increase damping and retry
            lambda *= 10.0;

            // Trace: rejected step
            if crate::estimation::trace::is_active() {
                let (gn_method, gn_phase) = gn_trace_method_phase(options.method);
                crate::estimation::trace::write_gn(
                    iter, gn_method, gn_phase, ofv, lambda, 0.0, false, None, None,
                );
            }

            if lambda > 1e6 {
                if verbose {
                    eprintln!("  GN iter {:>3}: lambda too large, stopping", iter);
                }
                warnings.push("Gauss-Newton: lambda exceeded threshold".to_string());
                break;
            }
            if verbose {
                eprintln!(
                    "  GN iter {:>3}: step rejected, lambda -> {:.4}",
                    iter, lambda
                );
            }
            continue;
        }

        // ---- Accept step ----
        let ofv_change = (ofv - ofv_new).abs();
        let rel_change = ofv_change / ofv.abs().max(1.0);

        x = x_new;
        let prev_ofv = ofv;
        ofv = ofv_new;
        eta_hats = eta_new;
        h_matrices = h_new;
        // kappas already updated in the line-search block on accept

        // Decrease damping on success
        lambda = (lambda * 0.3).max(1e-6);

        // Trace: accepted step
        if crate::estimation::trace::is_active() {
            let (gn_method, gn_phase) = gn_trace_method_phase(options.method);
            crate::estimation::trace::write_gn(
                iter,
                gn_method,
                gn_phase,
                ofv,
                lambda,
                ofv - prev_ofv,
                true,
                None,
                None,
            );
        }

        if verbose {
            eprintln!(
                "  GN iter {:>3}: OFV = {:.6}  (delta={:.2e}, lambda={:.4})",
                iter, ofv, ofv_change, lambda
            );
        }

        // Check convergence
        if rel_change < 1e-6 && iter > 3 {
            converged = true;
            if verbose {
                eprintln!("  Converged: relative OFV change = {:.2e}", rel_change);
            }
            break;
        }
    }

    if !converged {
        warnings.push("Gauss-Newton: max iterations reached without convergence".to_string());
    }

    let gn_ofv = ofv;
    let do_polish = matches!(options.method, EstimationMethod::FoceGnHybrid);

    // ---- Optional hybrid: polish with FOCEI from GN result ----
    if do_polish && verbose {
        eprintln!("GN phase done (OFV={:.4}). Polishing with FOCEI...", ofv);
    }

    let gn_params = unpack_params(&x, init_params);

    if !do_polish {
        // Pure GN — skip FOCEI polish, go directly to covariance step
        let covariance_matrix =
            if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
                if verbose {
                    eprintln!("Running covariance step...");
                }
                let cov = compute_covariance(
                    &x,
                    &gn_params,
                    model,
                    population,
                    &eta_hats,
                    &h_matrices,
                    &kappas,
                    options,
                );
                if cov.is_none() {
                    warnings.push("Covariance step failed".to_string());
                }
                cov
            } else {
                None
            };

        if verbose {
            eprintln!("FOCE-GN completed. Final OFV = {:.4}", ofv);
        }

        return OuterResult {
            params: gn_params,
            ofv,
            converged,
            n_iterations: maxiter,
            eta_hats,
            h_matrices,
            kappas,
            covariance_matrix,
            warnings,
            saem_mu_ref_m_step_evals_saved: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
        };
    }

    // Build FitOptions for the FOCEI polish: short maxiter, warm-started from GN
    let mut polish_options = options.clone();
    polish_options.method = EstimationMethod::Foce;
    polish_options.outer_maxiter = 100; // short polish
    polish_options.global_search = false;
    polish_options.run_covariance_step = false; // defer to after polish

    // Tell the trace that the following NLopt rows belong to the focei polish
    // phase of gn_hybrid, not a standalone foce/focei run.
    crate::estimation::trace::set_overrides(Some("gn_hybrid"), Some("focei"));
    let polish_result = crate::estimation::outer_optimizer::optimize_population_warm(
        model,
        population,
        &gn_params,
        &polish_options,
        &eta_hats,
        &h_matrices,
    );
    crate::estimation::trace::set_overrides(None, None);

    let final_ofv;
    let final_params;
    let final_etas;
    let final_h_mats;
    let final_kappas;

    if polish_result.ofv < gn_ofv {
        if verbose {
            eprintln!(
                "  FOCEI polish improved OFV: {:.4} -> {:.4}",
                gn_ofv, polish_result.ofv
            );
        }
        final_ofv = polish_result.ofv;
        final_params = polish_result.params;
        final_etas = polish_result.eta_hats;
        final_h_mats = polish_result.h_matrices;
        final_kappas = polish_result.kappas;
        converged = polish_result.converged || converged;
    } else {
        if verbose {
            eprintln!("  FOCEI polish did not improve (GN result kept)");
        }
        final_ofv = gn_ofv;
        final_params = gn_params;
        final_etas = eta_hats;
        final_h_mats = h_matrices;
        final_kappas = kappas;
    }

    // ---- Covariance step ----
    let covariance_matrix = if options.run_covariance_step {
        if verbose {
            eprintln!("Running covariance step...");
        }
        let packed = pack_params(&final_params);
        let cov = compute_covariance(
            &packed,
            &final_params,
            model,
            population,
            &final_etas,
            &final_h_mats,
            &final_kappas,
            options,
        );
        if cov.is_none() {
            warnings.push("Covariance step failed".to_string());
        }
        cov
    } else {
        None
    };

    if verbose {
        eprintln!("FOCE-GN completed. Final OFV = {:.4}", final_ofv);
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: maxiter,
        eta_hats: final_etas,
        h_matrices: final_h_mats,
        kappas: final_kappas,
        covariance_matrix,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
    }
}

/// Returns the (method, phase) strings for GN trace rows.
/// gn_hybrid rows use method="gn_hybrid" and phase="gn" during the GN loop.
/// Pure gn rows use method="gn" and phase="".
fn gn_trace_method_phase(method: EstimationMethod) -> (&'static str, &'static str) {
    match method {
        EstimationMethod::FoceGnHybrid => ("gn_hybrid", "gn"),
        _ => ("gn", ""),
    }
}

/// Build the Gauss-Newton linear system using the gradient and approximate Hessian
/// of the FOCE population objective.
///
/// The gradient is computed via central FD of the total OFV w.r.t. packed params.
/// The approximate Hessian uses the outer product of per-subject gradients (BHHH):
///   H_bhhh = sum_i g_i g_i^T
/// where g_i = d(2*nll_i)/d(x) is the per-subject OFV gradient.
///
/// This is the Berndt-Hall-Hall-Hausman (BHHH) approximation, which is equivalent
/// to Gauss-Newton for the FOCE log-likelihood and is what NONMEM uses internally.
fn build_gn_system(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> (DVector<f64>, DMatrix<f64>) {
    let n = x.len();
    let n_subj = population.subjects.len();

    // Compute per-subject NLL at current point
    let params = unpack_params(x, template);
    let _nll_base: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let kap_i = if i < kappas.len() {
                kappas[i].as_slice()
            } else {
                &[]
            };
            subject_nll_at(
                model,
                population,
                i,
                &params,
                &eta_hats[i],
                &h_matrices[i],
                kap_i,
                options,
            )
        })
        .collect();

    // Compute per-subject gradient via central FD
    // g_i[j] = d(nll_i)/d(x_j) for each subject i, parameter j
    let eps = 1e-4;
    let mut per_subj_grad: Vec<Vec<f64>> = vec![vec![0.0; n]; n_subj];
    let mut x_work = x.to_vec();
    let fixed_mask = packed_fixed_mask(template);

    for j in 0..n {
        // Skip FD evaluation for FIX parameters — the gradient is identically
        // zero there and the `unpack_params + per-subject NLL` passes are
        // expensive.
        if fixed_mask[j] {
            continue;
        }
        let h = eps * (1.0 + x[j].abs());
        let xj_plus = (x[j] + h).min(bounds.upper[j]);
        let xj_minus = (x[j] - h).max(bounds.lower[j]);
        let actual_2h = xj_plus - xj_minus;
        if actual_2h.abs() < 1e-16 {
            continue;
        }

        x_work[j] = xj_plus;
        let params_plus = unpack_params(&x_work, template);
        let nll_plus: Vec<f64> = population
            .subjects
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let kap_i = if i < kappas.len() {
                    kappas[i].as_slice()
                } else {
                    &[]
                };
                subject_nll_at(
                    model,
                    population,
                    i,
                    &params_plus,
                    &eta_hats[i],
                    &h_matrices[i],
                    kap_i,
                    options,
                )
            })
            .collect();

        x_work[j] = xj_minus;
        let params_minus = unpack_params(&x_work, template);
        let nll_minus: Vec<f64> = population
            .subjects
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let kap_i = if i < kappas.len() {
                    kappas[i].as_slice()
                } else {
                    &[]
                };
                subject_nll_at(
                    model,
                    population,
                    i,
                    &params_minus,
                    &eta_hats[i],
                    &h_matrices[i],
                    kap_i,
                    options,
                )
            })
            .collect();

        x_work[j] = x[j];

        for i in 0..n_subj {
            let deriv = (nll_plus[i] - nll_minus[i]) / actual_2h;
            per_subj_grad[i][j] = if deriv.is_finite() { deriv } else { 0.0 };
        }
    }

    // Total gradient: g = sum_i g_i (scaled by 2 for OFV = 2*NLL)
    let mut grad = DVector::zeros(n);
    for i in 0..n_subj {
        for j in 0..n {
            grad[j] += 2.0 * per_subj_grad[i][j];
        }
    }

    // BHHH approximate Hessian: H = sum_i (2*g_i)(2*g_i)^T = 4 * sum_i g_i g_i^T
    // But for the Newton step H*delta = -grad, we can factor out the 4:
    // Use H_bhhh = sum_i g_i g_i^T, and solve (H_bhhh * delta) = -(grad/4)...
    // Actually, let's just use the properly scaled version.
    //
    // For OFV = 2 * sum_i nll_i:
    //   grad(OFV) = 2 * sum_i grad_i
    //   H_bhhh(OFV) ≈ 4 * sum_i grad_i grad_i^T
    //
    // Newton step: delta = -H^{-1} grad = -(4 sum g_i g_i^T)^{-1} (2 sum g_i)
    //            = -0.5 * (sum g_i g_i^T)^{-1} (sum g_i)
    //
    // We return (grad_total, H_total) where grad_total = 2*sum(g_i) and
    // H_total = 4*sum(g_i g_i^T) so the caller solves H*delta = -grad directly.

    let mut h_bhhh = DMatrix::zeros(n, n);
    for i in 0..n_subj {
        let gi = DVector::from_column_slice(&per_subj_grad[i]);
        h_bhhh += 4.0 * &gi * gi.transpose();
    }

    (grad, h_bhhh)
}

/// Compute FOCE NLL for a single subject at given parameters with fixed EBEs.
///
/// `kappas` contains per-occasion kappa EBEs (empty when no IOV).  When
/// non-empty, delegates to `foce_subject_nll_iov` which builds per-occasion
/// predictions using `[bsv_eta, kappa_k]` and adds kappa priors.
fn subject_nll_at(
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    params: &ModelParameters,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    kappas: &[DVector<f64>],
    options: &FitOptions,
) -> f64 {
    use crate::stats::likelihood::foce_subject_nll_iov;
    let subject = &population.subjects[subj_idx];

    if !kappas.is_empty() {
        if let Some(ref iov) = params.omega_iov {
            return foce_subject_nll_iov(
                model,
                subject,
                &params.theta,
                eta_hat,
                h_matrix,
                &params.omega,
                &params.sigma.values,
                options.interaction,
                kappas,
                iov,
            );
        }
    }

    let ipreds =
        crate::pk::compute_predictions_with_tv(model, subject, &params.theta, eta_hat.as_slice());

    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();

    if options.interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            &params.omega,
            &params.sigma.values,
            model.error_model,
            model.bloq_method,
            &[],
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            &params.omega,
            &params.sigma.values,
            model.error_model,
            model.bloq_method,
            &[],
        )
    }
}
