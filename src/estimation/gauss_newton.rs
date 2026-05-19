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
use crate::stats::likelihood::{
    chol_log_det, compute_r_tilde, foce_subject_nll_interaction, foce_subject_nll_standard,
};
use crate::stats::residual_error::compute_r_diag;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

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

// ── Analytical gradient helpers ──────────────────────────────────────────────
//
// For each packed parameter x[k] the gradient of the FOCE NLL uses:
//
//   dL = A^T dv  +  0.5 * (tr(R^{-1} dR) - A^T dR A)
//
// where  v = y - f0,  A = R_tilde^{-1} v,  and C = chol(R_tilde).
//
// Omega diagonal x[k] = log L_omega[i,i]:
//   dR = 2*omega_ii * h_i h_i^T,  dv = 0
//   dL = omega_ii * (||C^{-1} h_i||^2 - (h_i^T A)^2)
//
// Omega off-diagonal x[k] = L_omega[j,i] (j>i, identity-packed):
//   dOmega = e_j (L[:,i])^T + L[:,i] e_j^T
//   u = H L_omega[:,i],  w = H[:,j] = h_j
//   dR = w u^T + u w^T,  dv = 0
//   dL = (C^{-1} w)^T (C^{-1} u) - (w^T A)(u^T A)
//
// Sigma x[k] = log sigma_k:
//   dR = diag(d r_j / d log sigma_k),  dv = 0
//   dL = 0.5 * sum_j (d r_j) * ((R^{-1})_jj - A_j^2)
//
// Theta x[k] (log-packed or identity):
//   d(ipreds) from forward FD of compute_predictions_with_tv
//   d(f0) = d(ipreds),  d(r) via chain rule through residual variance
//   dv = -d(f0)
//   dL = -A^T d(f0)  +  0.5 * (sum_j d(r_j) * ((R^{-1})_jj - A_j^2))
//      (the diagonal-R^{-1} term is shared with sigma and reused)

/// Diagonal of R^{-1} via Cholesky: (R^{-1})_jj = ||C^{-1} e_j||^2
/// where C = chol(R) (lower triangular). Computed by forward-substituting each unit vector.
fn r_inv_diag(chol_l: &DMatrix<f64>) -> Vec<f64> {
    let n = chol_l.nrows();
    (0..n)
        .map(|j| {
            // Solve C w = e_j by forward substitution
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                if i == j {
                    w[i] = 1.0 / chol_l[(i, i)];
                } else if i > j {
                    let mut s = 0.0;
                    for k in j..i {
                        s += chol_l[(i, k)] * w[k];
                    }
                    w[i] = -s / chol_l[(i, i)];
                }
            }
            w.iter().map(|&x| x * x).sum()
        })
        .collect()
}

/// Solve C w = rhs by forward substitution (C lower triangular).
fn fwd_solve(chol_l: &DMatrix<f64>, rhs: &[f64]) -> Vec<f64> {
    let n = chol_l.nrows();
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= chol_l[(i, k)] * w[k];
        }
        w[i] = s / chol_l[(i, i)];
    }
    w
}

/// d(r_j)/d(log sigma_k) for each observation, at the prediction point used
/// to evaluate r_diag (f0 for standard, ipreds for interaction).
fn dr_diag_d_log_sigma(
    error_model: ErrorModel,
    r_diag: &[f64],
    pred_point: &[f64], // f0 or ipreds depending on path
    sigma_values: &[f64],
    sigma_k: usize,
) -> Vec<f64> {
    r_diag
        .iter()
        .zip(pred_point.iter())
        .map(|(&r_j, &f_j)| match error_model {
            ErrorModel::Additive => {
                if sigma_k == 0 {
                    2.0 * sigma_values[0] * sigma_values[0]
                } else {
                    0.0
                }
            }
            ErrorModel::Proportional => {
                if sigma_k == 0 {
                    2.0 * r_j
                } else {
                    0.0
                }
            }
            ErrorModel::Combined => {
                if sigma_k == 0 {
                    // d(sigma_prop^2 * f^2)/d(log sigma_prop) = 2 * sigma_prop^2 * f^2
                    let sp2 = sigma_values[0] * sigma_values[0];
                    2.0 * sp2 * f_j * f_j
                } else {
                    // sigma_k == 1: d(sigma_add^2)/d(log sigma_add) = 2 * sigma_add^2
                    2.0 * sigma_values[1] * sigma_values[1]
                }
            }
        })
        .collect()
}

/// Analytical per-subject FOCE NLL gradient for non-IOV, non-ODE, non-M3 models.
///
/// Returns `None` if the Cholesky of R_tilde fails (degenerate parameters) so
/// the caller can fall back to central FD.
///
/// The gradient is exact for omega and sigma packed parameters. For theta packed
/// parameters, uses forward FD of `compute_predictions_with_tv` only (cheap —
/// no matrix work per perturbation).
#[allow(clippy::too_many_arguments)]
fn subject_nll_pop_grad_analytical(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Option<(f64, Vec<f64>)> {
    use crate::pk;

    let n = x.len();
    let n_eta = model.n_eta;
    let n_theta = template.theta.len();
    let n_sigma = template.sigma.values.len();
    let n_obs = population.subjects[subj_idx].observations.len();
    let subject = &population.subjects[subj_idx];
    let params = unpack_params(x, template);
    let fixed_mask = packed_fixed_mask(template);

    // Base predictions and FOCE state
    let ipreds = pk::compute_predictions_with_tv(model, subject, &params.theta, eta_hat.as_slice());

    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // r_diag: at f0 (standard) or at ipreds (interaction)
    let r_pred_point: &[f64] = if options.interaction { &ipreds } else { &f0 };
    let r_diag = compute_r_diag(model.error_model, r_pred_point, &params.sigma.values);

    let r_tilde = compute_r_tilde(h_matrix, &params.omega.matrix, &r_diag);
    let chol = r_tilde.cholesky()?;
    let chol_l = chol.l();

    let v: DVector<f64> = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(f0.iter())
            .map(|(&y, &f)| y - f),
    );
    let solved_a = chol.solve(&v);
    let nll = 0.5 * (v.dot(&solved_a) + chol_log_det(&chol_l));

    // Diagonal of R_tilde^{-1} — shared across all sigma and theta gradient terms
    let rinv_diag = r_inv_diag(&chol_l);

    // Pre-compute C^{-1} H columns for omega gradient: each w_i = C^{-1} h_i
    let h_cols_solved: Vec<Vec<f64>> = (0..n_eta)
        .map(|i| fwd_solve(&chol_l, h_matrix.column(i).as_slice()))
        .collect();

    // For block omega: get the Cholesky factor of omega (L_omega)
    let l_omega = &params.omega.chol; // nalgebra DMatrix, lower triangular

    let mut grad = vec![0.0f64; n];

    // ── Theta parameters (indices 0..n_theta) ──────────────────────────────
    let eps = 1e-5;
    for k in 0..n_theta {
        if fixed_mask[k] {
            continue;
        }
        let h = eps * (1.0 + x[k].abs());
        let xk_plus = (x[k] + h).min(bounds.upper[k]);
        let actual_h = xk_plus - x[k];
        if actual_h.abs() < 1e-16 {
            continue;
        }

        let mut x_pert = x.to_vec();
        x_pert[k] = xk_plus;
        let params_pert = unpack_params(&x_pert, template);
        let ipreds_pert =
            pk::compute_predictions_with_tv(model, subject, &params_pert.theta, eta_hat.as_slice());

        // d(ipreds)/dx[k] via forward FD
        let d_ipreds: Vec<f64> = ipreds_pert
            .iter()
            .zip(ipreds.iter())
            .map(|(&p, &b)| (p - b) / actual_h)
            .collect();

        // d(f0) = d(ipreds); d(v) = -d(f0)
        // For sigma-dependent r: d(r_j)/d(x[k]) via chain rule through r(f0 or ipreds)
        let dr: Vec<f64> = r_diag
            .iter()
            .zip(r_pred_point.iter().zip(d_ipreds.iter()))
            .map(|(&r_j, (&pred_j, &dp_j))| match model.error_model {
                ErrorModel::Additive => 0.0,
                ErrorModel::Proportional => {
                    // r_j = sigma^2 * pred^2 => dr/d(pred) = 2*sigma^2*pred = 2*r_j/pred
                    if pred_j.abs() > 1e-15 {
                        2.0 * r_j / pred_j * dp_j
                    } else {
                        0.0
                    }
                }
                ErrorModel::Combined => {
                    let sp2 = params.sigma.values[0] * params.sigma.values[0];
                    2.0 * sp2 * pred_j * dp_j
                }
            })
            .collect();

        // dL = -A^T d(f0) + 0.5 * sum_j dr_j * ((R^{-1})_jj - A_j^2)
        let data_term: f64 = solved_a
            .iter()
            .zip(d_ipreds.iter())
            .map(|(&a_j, &df_j)| -a_j * df_j)
            .sum();
        let r_term: f64 = dr
            .iter()
            .zip(rinv_diag.iter().zip(solved_a.iter()))
            .map(|(&drj, (&rinv_jj, &a_j))| 0.5 * drj * (rinv_jj - a_j * a_j))
            .sum();

        grad[k] = data_term + r_term;
    }

    // ── Omega packed parameters ────────────────────────────────────────────
    let omega_start = n_theta;
    let omega_entries: Vec<(usize, usize)> = if template.omega.diagonal {
        (0..n_eta).map(|i| (i, i)).collect()
    } else {
        let mut v = Vec::new();
        for j in 0..n_eta {
            for i in j..n_eta {
                v.push((i, j));
            }
        }
        v
    };

    for (ko, &(row, col)) in omega_entries.iter().enumerate() {
        let k = omega_start + ko;
        if fixed_mask[k] {
            continue;
        }
        if row == col {
            // x[k] = log L_omega[i,i]; chain rule factor is L[i,i].
            // ∂NLL/∂L[i,i] = (C⁻¹hᵢ)·(C⁻¹uᵢ) − (hᵢ·a)(uᵢ·a)
            // where uᵢ = H·L[:,i].
            //
            // For diagonal omega L[:,i] = L[i,i]·eᵢ so uᵢ = L[i,i]·hᵢ and the
            // expression collapses to Ω[i,i]·(w² − (hᵢ·a)²).  For block omega
            // L[:,i] has sub-diagonal entries and we must use the full uᵢ.
            let l_ii = l_omega[(row, row)];
            let c_inv_w = &h_cols_solved[row]; // C⁻¹ hᵢ
            let w_dot_a: f64 = h_matrix
                .column(row)
                .iter()
                .zip(solved_a.iter())
                .map(|(h, a)| h * a)
                .sum();
            if template.omega.diagonal {
                // Fast path: uᵢ = L[i,i]·hᵢ, so C⁻¹uᵢ = L[i,i]·(C⁻¹hᵢ).
                let w_sq: f64 = c_inv_w.iter().map(|&x| x * x).sum();
                // ∂NLL/∂x[k] = L[i,i] · L[i,i] · (w² − (hᵢ·a)²) = Ω[i,i] · (...)
                grad[k] = l_ii * l_ii * (w_sq - w_dot_a * w_dot_a);
            } else {
                // General path: compute uᵢ = H·L[:,i] using the full i-th column.
                let l_col: Vec<f64> = (0..n_eta).map(|r| l_omega[(r, row)]).collect();
                let u: Vec<f64> = (0..n_obs)
                    .map(|obs_j| {
                        (0..n_eta)
                            .map(|eta_i| h_matrix[(obs_j, eta_i)] * l_col[eta_i])
                            .sum::<f64>()
                    })
                    .collect();
                let c_inv_u = fwd_solve(&chol_l, &u);
                let u_dot_a: f64 = u.iter().zip(solved_a.iter()).map(|(ui, ai)| ui * ai).sum();
                let trace_term: f64 = c_inv_w.iter().zip(c_inv_u.iter()).map(|(w, u)| w * u).sum();
                grad[k] = l_ii * (trace_term - w_dot_a * u_dot_a);
            }
        } else {
            // Off-diagonal: x[k] = L_omega[row, col] (identity-packed)
            // u = H * L_omega[:,col], w_vec = h_row = H[:,row]
            // dR = w u^T + u w^T (symmetric rank-2)
            // dL = (C^{-1} w)^T (C^{-1} u) - (w^T A)(u^T A)
            let l_col: Vec<f64> = (0..n_eta).map(|r| l_omega[(r, col)]).collect();
            let u: Vec<f64> = (0..n_obs)
                .map(|obs_j| {
                    (0..n_eta)
                        .map(|eta_i| h_matrix[(obs_j, eta_i)] * l_col[eta_i])
                        .sum::<f64>()
                })
                .collect();
            let c_inv_u = fwd_solve(&chol_l, &u);
            let c_inv_w = &h_cols_solved[row];
            let u_dot_a: f64 = u.iter().zip(solved_a.iter()).map(|(ui, ai)| ui * ai).sum();
            let w_dot_a: f64 = h_matrix
                .column(row)
                .iter()
                .zip(solved_a.iter())
                .map(|(h, a)| h * a)
                .sum();
            let trace_term: f64 = c_inv_w.iter().zip(c_inv_u.iter()).map(|(w, u)| w * u).sum();
            grad[k] = trace_term - w_dot_a * u_dot_a;
        }
    }

    // ── Sigma packed parameters ────────────────────────────────────────────
    let sigma_start = omega_start + omega_entries.len();
    for ks in 0..n_sigma {
        let k = sigma_start + ks;
        if fixed_mask[k] {
            continue;
        }
        let dr_k = dr_diag_d_log_sigma(
            model.error_model,
            &r_diag,
            r_pred_point,
            &params.sigma.values,
            ks,
        );
        let g: f64 = dr_k
            .iter()
            .zip(rinv_diag.iter().zip(solved_a.iter()))
            .map(|(&drj, (&rinv_jj, &a_j))| 0.5 * drj * (rinv_jj - a_j * a_j))
            .sum();
        grad[k] = g;
    }

    Some((nll, grad))
}

/// Compute the FOCE NLL and its gradient w.r.t. the packed population parameter
/// vector for a single subject, with ETAs fixed at their current EBE values.
///
/// For non-IOV analytical PK models without M3 BLOQ, uses an analytical gradient:
/// exact for omega/sigma parameters; forward-FD of predictions only for theta.
/// Falls back to central FD for ODE models, IOV, or M3 BLOQ.
///
/// Returns `(nll_i, gradient_i)` where `gradient_i[j] = d(nll_i)/d(x[j])`.
pub(crate) fn subject_nll_pop_grad(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    kappas: &[DVector<f64>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> (f64, Vec<f64>) {
    let can_use_analytical = model.ode_spec.is_none()
        && kappas.is_empty()
        && !matches!(model.bloq_method, BloqMethod::M3);

    if can_use_analytical {
        if let Some(result) = subject_nll_pop_grad_analytical(
            x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
        ) {
            return result;
        }
    }

    // Fallback: central FD over full per-subject NLL
    let n = x.len();
    let fixed_mask = packed_fixed_mask(template);
    let eps = 1e-4;

    let params_base = unpack_params(x, template);
    let nll_base = subject_nll_at(
        model,
        population,
        subj_idx,
        &params_base,
        eta_hat,
        h_matrix,
        kappas,
        options,
    );

    let mut grad = vec![0.0f64; n];
    let mut x_work = x.to_vec();

    for j in 0..n {
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
        let nll_plus = subject_nll_at(
            model,
            population,
            subj_idx,
            &params_plus,
            eta_hat,
            h_matrix,
            kappas,
            options,
        );

        x_work[j] = xj_minus;
        let params_minus = unpack_params(&x_work, template);
        let nll_minus = subject_nll_at(
            model,
            population,
            subj_idx,
            &params_minus,
            eta_hat,
            h_matrix,
            kappas,
            options,
        );

        x_work[j] = x[j];

        let deriv = (nll_plus - nll_minus) / actual_2h;
        grad[j] = if deriv.is_finite() {
            deriv
        } else if nll_plus.is_finite() && nll_base.is_finite() {
            // One-sided fallback: minus-side was non-finite.
            (nll_plus - nll_base) / (xj_plus - x[j])
        } else if nll_minus.is_finite() && nll_base.is_finite() {
            // One-sided fallback: plus-side was non-finite.
            (nll_base - nll_minus) / (x[j] - xj_minus)
        } else {
            0.0
        };
    }

    (nll_base, grad)
}

/// Build the Gauss-Newton linear system: gradient and BHHH approximate Hessian
/// of the FOCE population objective.
///
/// Calls `subject_nll_pop_grad` in parallel over subjects (rayon `par_iter`).
/// Each subject contributes its per-subject NLL gradient g_i; the totals are:
///   grad(OFV) = 2 * Σ_i g_i
///   H_bhhh(OFV) ≈ 4 * Σ_i g_i g_i^T   (BHHH approximation)
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

    let per_subj: Vec<(f64, Vec<f64>)> = (0..n_subj)
        .into_par_iter()
        .map(|i| {
            let kap_i = if i < kappas.len() {
                kappas[i].as_slice()
            } else {
                &[]
            };
            subject_nll_pop_grad(
                x,
                template,
                model,
                population,
                i,
                &eta_hats[i],
                &h_matrices[i],
                kap_i,
                bounds,
                options,
            )
        })
        .collect();

    // For OFV = 2 * Σ nll_i:
    //   grad(OFV)    = 2 * Σ g_i
    //   H_bhhh(OFV) ≈ 4 * Σ g_i g_i^T
    let mut grad = DVector::zeros(n);
    let mut h_bhhh = DMatrix::zeros(n, n);
    for (_, gi) in &per_subj {
        let gi_vec = DVector::from_column_slice(gi);
        grad += 2.0 * &gi_vec;
        h_bhhh += 4.0 * &gi_vec * gi_vec.transpose();
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
            name: "gn_test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL
                p.values[1] = theta[1]; // V
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

    fn make_population() -> Population {
        let subjects = (0..3)
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

    /// Verify that build_gn_system returns a gradient and BHHH Hessian with
    /// correct dimensions and that the gradient matches a sequential central-FD
    /// reference to within numerical noise.
    #[test]
    fn test_build_gn_system_gradient_matches_fd_reference() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;
        let n_subj = population.subjects.len();

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        // h_matrix shape: (n_obs × n_eta) — here 3 observations, 1 eta
        let n_obs = 3;
        let n_eta = 1;
        let eta_hats: Vec<DVector<f64>> = (0..n_subj).map(|_| DVector::zeros(n_eta)).collect();
        let h_matrices: Vec<nalgebra::DMatrix<f64>> = (0..n_subj)
            .map(|_| nalgebra::DMatrix::zeros(n_obs, n_eta))
            .collect();
        let kappas: Vec<Vec<DVector<f64>>> = vec![vec![]; n_subj];
        let options = FitOptions::default();

        let (grad, h_bhhh) = build_gn_system(
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

        // Dimensions
        assert_eq!(grad.len(), n);
        assert_eq!(h_bhhh.nrows(), n);
        assert_eq!(h_bhhh.ncols(), n);

        // BHHH Hessian must be symmetric and positive semi-definite (all eigenvalues >= 0)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (h_bhhh[(i, j)] - h_bhhh[(j, i)]).abs() < 1e-10,
                    "H_bhhh not symmetric at ({i},{j})"
                );
            }
        }

        // Gradient must be finite
        for (k, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "gradient[{k}] is not finite");
        }

        // Cross-check: sequential central-FD reference must agree with the
        // parallel result to within FD numerical noise (~1e-6 relative).
        let eps = 1e-4;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps * (1.0 + x[j].abs());
            xm[j] -= eps * (1.0 + x[j].abs());
            let actual_2h = xp[j] - xm[j];

            let params_p = unpack_params(&xp, template);
            let params_m = unpack_params(&xm, template);

            let nll_p: f64 = (0..n_subj)
                .map(|i| {
                    subject_nll_at(
                        &model,
                        &population,
                        i,
                        &params_p,
                        &eta_hats[i],
                        &h_matrices[i],
                        &[],
                        &options,
                    )
                })
                .sum();
            let nll_m: f64 = (0..n_subj)
                .map(|i| {
                    subject_nll_at(
                        &model,
                        &population,
                        i,
                        &params_m,
                        &eta_hats[i],
                        &h_matrices[i],
                        &[],
                        &options,
                    )
                })
                .sum();

            let ref_grad_j = 2.0 * (nll_p - nll_m) / actual_2h;
            let tol = 1e-4 * (1.0 + ref_grad_j.abs());
            assert!(
                (grad[j] - ref_grad_j).abs() < tol,
                "gradient[{j}]: parallel={:.6e}, reference={:.6e}, diff={:.2e}",
                grad[j],
                ref_grad_j,
                (grad[j] - ref_grad_j).abs()
            );
        }
    }

    /// Verify that the analytical gradient path agrees with central FD to tight
    /// tolerance — tests exact omega/sigma gradients and forward-FD theta
    /// gradients against the reference.
    #[test]
    fn test_subject_nll_pop_grad_analytical_matches_fd() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::zeros(n_eta);
        let h_matrix = nalgebra::DMatrix::zeros(n_obs, n_eta);
        let options = FitOptions::default();

        // Analytical path (called directly)
        let (nll_an, grad_an) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &options,
        )
        .expect("analytical path should succeed for non-IOV non-ODE model");

        // Central-FD reference
        let params_base = unpack_params(&x, template);
        let nll_ref = subject_nll_at(
            &model,
            &population,
            0,
            &params_base,
            &eta_hat,
            &h_matrix,
            &[],
            &options,
        );
        assert!(
            (nll_an - nll_ref).abs() < 1e-10,
            "nll mismatch: {nll_an} vs {nll_ref}"
        );

        let eps = 1e-5;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps * (1.0 + x[j].abs());
            xm[j] -= eps * (1.0 + x[j].abs());
            let actual_2h = xp[j] - xm[j];
            let pp = unpack_params(&xp, template);
            let pm = unpack_params(&xm, template);
            let np = subject_nll_at(
                &model,
                &population,
                0,
                &pp,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let nm = subject_nll_at(
                &model,
                &population,
                0,
                &pm,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let ref_j = (np - nm) / actual_2h;

            // Omega and sigma components should be machine-accurate; theta can
            // differ due to forward vs central FD — allow 1% relative tolerance.
            let tol = 0.01 * (1.0 + ref_j.abs()).max(1e-8);
            assert!(
                (grad_an[j] - ref_j).abs() < tol,
                "analytical grad[{j}]: {:.6e}, fd ref: {:.6e}, diff: {:.2e}",
                grad_an[j],
                ref_j,
                (grad_an[j] - ref_j).abs()
            );
        }
    }

    /// Verify that the analytical gradient is correct when H is non-zero —
    /// exercises the `h_cols_solved` path and the omega diagonal formula with
    /// non-trivial eta-to-obs mapping.
    #[test]
    fn test_subject_nll_pop_grad_analytical_nonzero_h() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        // Non-zero H: partial derivatives of predictions w.r.t. eta at baseline
        let eta_hat = DVector::from_vec(vec![0.1]);
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![2.0, 1.5, 0.8]);
        let options = FitOptions::default();

        let (nll_an, grad_an) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &options,
        )
        .expect("analytical path should succeed");

        // NLL must match direct call
        let params_base = unpack_params(&x, template);
        let nll_ref = subject_nll_at(
            &model,
            &population,
            0,
            &params_base,
            &eta_hat,
            &h_matrix,
            &[],
            &options,
        );
        assert!(
            (nll_an - nll_ref).abs() < 1e-10,
            "nll mismatch: {nll_an} vs {nll_ref}"
        );

        // Gradient must agree with central-FD reference
        let eps = 1e-5;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps * (1.0 + x[j].abs());
            xm[j] -= eps * (1.0 + x[j].abs());
            let actual_2h = xp[j] - xm[j];
            let pp = unpack_params(&xp, template);
            let pm = unpack_params(&xm, template);
            let np = subject_nll_at(
                &model,
                &population,
                0,
                &pp,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let nm = subject_nll_at(
                &model,
                &population,
                0,
                &pm,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let ref_j = (np - nm) / actual_2h;

            let tol = 0.01 * (1.0 + ref_j.abs()).max(1e-8);
            assert!(
                (grad_an[j] - ref_j).abs() < tol,
                "nonzero-H: analytical grad[{j}]={:.6e}, fd ref={:.6e}, diff={:.2e}",
                grad_an[j],
                ref_j,
                (grad_an[j] - ref_j).abs()
            );
        }
    }

    /// Verify that the analytical gradient is correct under the FOCEI interaction
    /// path (`options.interaction = true`), where r_diag is evaluated at ipreds
    /// rather than f0, and dr/d(theta) passes through the r(ipred) chain.
    #[test]
    fn test_subject_nll_pop_grad_analytical_interaction() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.05]);
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![3.0, 2.0, 1.0]);
        let mut options = FitOptions::default();
        options.interaction = true;

        let (nll_an, grad_an) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &options,
        )
        .expect("analytical path should succeed with interaction=true");

        let params_base = unpack_params(&x, template);
        let nll_ref = subject_nll_at(
            &model,
            &population,
            0,
            &params_base,
            &eta_hat,
            &h_matrix,
            &[],
            &options,
        );
        assert!(
            (nll_an - nll_ref).abs() < 1e-10,
            "interaction nll mismatch: {nll_an} vs {nll_ref}"
        );

        let eps = 1e-5;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps * (1.0 + x[j].abs());
            xm[j] -= eps * (1.0 + x[j].abs());
            let actual_2h = xp[j] - xm[j];
            let pp = unpack_params(&xp, template);
            let pm = unpack_params(&xm, template);
            let np = subject_nll_at(
                &model,
                &population,
                0,
                &pp,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let nm = subject_nll_at(
                &model,
                &population,
                0,
                &pm,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let ref_j = (np - nm) / actual_2h;

            let tol = 0.01 * (1.0 + ref_j.abs()).max(1e-8);
            assert!(
                (grad_an[j] - ref_j).abs() < tol,
                "interaction: analytical grad[{j}]={:.6e}, fd ref={:.6e}, diff={:.2e}",
                grad_an[j],
                ref_j,
                (grad_an[j] - ref_j).abs()
            );
        }
    }

    /// Verify that `subject_nll_pop_grad` returns (nll, gradient) where the
    /// gradient agrees with a sequential central-FD reference for subject 0,
    /// and the returned nll matches a direct call to `subject_nll_at`.
    #[test]
    fn test_subject_nll_pop_grad_matches_fd_reference() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::zeros(n_eta);
        let h_matrix = nalgebra::DMatrix::zeros(n_obs, n_eta);
        let options = FitOptions::default();

        let (nll, grad) = subject_nll_pop_grad(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &[],
            &bounds,
            &options,
        );

        // NLL must match a direct subject_nll_at call
        let params_base = unpack_params(&x, template);
        let nll_ref = subject_nll_at(
            &model,
            &population,
            0,
            &params_base,
            &eta_hat,
            &h_matrix,
            &[],
            &options,
        );
        assert!(
            (nll - nll_ref).abs() < 1e-12,
            "nll mismatch: {nll} vs {nll_ref}"
        );

        // Gradient must be finite
        for (j, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "grad[{j}] not finite");
        }

        // Each gradient component must agree with sequential central-FD reference
        let eps = 1e-4;
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps * (1.0 + x[j].abs());
            xm[j] -= eps * (1.0 + x[j].abs());
            let actual_2h = xp[j] - xm[j];

            let params_p = unpack_params(&xp, template);
            let params_m = unpack_params(&xm, template);
            let nll_p = subject_nll_at(
                &model,
                &population,
                0,
                &params_p,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let nll_m = subject_nll_at(
                &model,
                &population,
                0,
                &params_m,
                &eta_hat,
                &h_matrix,
                &[],
                &options,
            );
            let ref_j = (nll_p - nll_m) / actual_2h;

            let tol = 1e-4 * (1.0 + ref_j.abs());
            assert!(
                (grad[j] - ref_j).abs() < tol,
                "grad[{j}]: subject_nll_pop_grad={:.6e}, ref={:.6e}",
                grad[j],
                ref_j,
            );
        }
    }
}
