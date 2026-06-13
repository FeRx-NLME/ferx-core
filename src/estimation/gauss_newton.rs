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
use crate::estimation::outer_optimizer::{compute_covariance, CovarianceStepResult, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::estimation::trust_region::{adaptive_steihaug_budget, solve_trust_region_subproblem};
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
    let mut trust_radius: f64 = 1.0; // TR initial radius (in scaled space)
    let delta_max: f64 = 10.0; // TR maximum radius

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
        eprintln!(
            "  {} packed parameters, initial trust radius={:.4}",
            n_packed, trust_radius
        );
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

        // ---- Trust-region subproblem (Steihaug truncated CG) ----
        let cg_budget = adaptive_steihaug_budget(n_packed);
        let delta_s = solve_trust_region_subproblem(&grad_s, &h_s, trust_radius, cg_budget);

        // Predicted decrease in the scaled quadratic model: -gᵀδ - ½ δᵀHδ
        let h_s_delta_s = &h_s * &delta_s;
        let pred_reduction = -grad_s.dot(&delta_s) - 0.5 * delta_s.dot(&h_s_delta_s);

        if pred_reduction < 1e-10 {
            if grad_s.norm() < 1e-8 {
                // Gradient is genuinely zero — true stationary point.
                converged = true;
                if verbose {
                    eprintln!(
                        "  GN iter {:>3}: predicted reduction negligible, converged",
                        iter
                    );
                }
                break;
            } else {
                // Degenerate BHHH: near-zero quadratic improvement despite a
                // non-zero gradient. Shrink the trust radius and retry.
                trust_radius /= 4.0;
                if trust_radius < 1e-10 {
                    if verbose {
                        eprintln!(
                            "  GN iter {:>3}: degenerate BHHH Hessian, trust radius collapsed",
                            iter
                        );
                    }
                    warnings.push(
                        "Gauss-Newton: degenerate BHHH Hessian, trust radius collapsed".to_string(),
                    );
                    break;
                }
                if verbose {
                    eprintln!(
                        "  GN iter {:>3}: degenerate BHHH, shrinking radius to {:.4e}",
                        iter, trust_radius
                    );
                }
                continue;
            }
        }

        // Proposed new point (in unscaled parameter space)
        let delta =
            DVector::from_iterator(n_packed, (0..n_packed).map(|i| delta_s[i] * gn_scale[i]));
        let mut x_try = x.clone();
        for i in 0..n_packed {
            x_try[i] = (x[i] + delta[i]).clamp(bounds.lower[i], bounds.upper[i]);
        }

        let params_try = unpack_params(&x_try, init_params);
        let try_mu_k = compute_mu_k(model, &params_try.theta, options.mu_referencing);
        let (eta_try, h_try, _, kap_try) = run_inner_loop_warm(
            model,
            population,
            &params_try,
            options.inner_maxiter,
            options.inner_tol,
            Some(&eta_hats),
            Some(&try_mu_k),
            options.min_obs_for_convergence_check as usize,
        );
        let ofv_try = 2.0
            * pop_nll(
                model,
                population,
                &params_try,
                &eta_try,
                &h_try,
                &kap_try,
                options.interaction,
            );

        // TR ratio: actual OFV decrease vs quadratic model decrease.
        // rho < 0 or non-finite OFV → reject.
        let rho = if ofv_try.is_finite() {
            (ofv - ofv_try) / pred_reduction
        } else {
            -1.0
        };

        let radius_before = trust_radius;
        let (new_radius, accepted) = update_trust_radius(rho, trust_radius, delta_max);
        trust_radius = new_radius;

        if !accepted {
            // Trace: rejected step — use radius_before so the column records
            // the radius that was active when the step was attempted, consistent
            // with accepted-step rows (which also log the post-accept radius).
            if crate::estimation::trace::is_active() {
                let (gn_method, gn_phase) = gn_trace_method_phase(options.method);
                crate::estimation::trace::write_gn(
                    iter,
                    gn_method,
                    gn_phase,
                    ofv,
                    radius_before,
                    0.0,
                    false,
                    None,
                    None,
                );
            }

            if trust_radius < 1e-10 {
                if verbose {
                    eprintln!("  GN iter {:>3}: trust radius collapsed, stopping", iter);
                }
                warnings.push("Gauss-Newton: trust radius collapsed".to_string());
                break;
            }
            if verbose {
                eprintln!(
                    "  GN iter {:>3}: step rejected (rho={:.3}), radius -> {:.4e}",
                    iter, rho, trust_radius
                );
            }
            continue;
        }

        // ---- Accept step ----
        let ofv_change = (ofv - ofv_try).abs();
        let rel_change = ofv_change / ofv.abs().max(1.0);

        x = x_try;
        let prev_ofv = ofv;
        ofv = ofv_try;
        eta_hats = eta_try;
        h_matrices = h_try;
        kappas = kap_try;

        // Trace: accepted step (lm_lambda column carries trust_radius for GN-TR)
        if crate::estimation::trace::is_active() {
            let (gn_method, gn_phase) = gn_trace_method_phase(options.method);
            crate::estimation::trace::write_gn(
                iter,
                gn_method,
                gn_phase,
                ofv,
                trust_radius,
                ofv - prev_ofv,
                true,
                None,
                None,
            );
        }

        if verbose {
            eprintln!(
                "  GN iter {:>3}: OFV = {:.6}  (delta={:.2e}, radius={:.4})",
                iter, ofv, ofv_change, trust_radius
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

    // Recompute gradient at the final accepted x so the stored value is always
    // at the converged point (mid-loop capture would be off by one step).
    let (grad_final, _) = build_gn_system(
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
    let mut final_gradient: Option<Vec<f64>> = Some(grad_final.as_slice().to_vec());

    let gn_ofv = ofv;
    let do_polish = matches!(options.method, EstimationMethod::FoceGnHybrid);

    // ---- Optional hybrid: polish with FOCEI from GN result ----
    if do_polish && verbose {
        eprintln!("GN phase done (OFV={:.4}). Polishing with FOCEI...", ofv);
    }

    let gn_params = unpack_params(&x, init_params);

    if !do_polish {
        // Pure GN — skip FOCEI polish, go directly to covariance step
        let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
        let covariance_matrix =
            if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
                if verbose {
                    eprintln!("Running covariance step...");
                }
                match compute_covariance(
                    &x,
                    &gn_params,
                    model,
                    population,
                    &eta_hats,
                    &h_matrices,
                    &kappas,
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
            saem_n_subjects_hmc: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            final_gradient,
            sir_fallback_proposal,
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
        final_gradient = polish_result.final_gradient.or(final_gradient);
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
    let mut sir_fallback_proposal: Option<DMatrix<f64>> = None;
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("Running covariance step...");
            }
            let packed = pack_params(&final_params);
            match compute_covariance(
                &packed,
                &final_params,
                model,
                population,
                &final_etas,
                &final_h_mats,
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
        saem_n_subjects_hmc: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient,
        sir_fallback_proposal,
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

/// Update the trust radius based on the TR ratio ρ = actual / predicted reduction.
///
/// Returns `(new_radius, accepted)`:
/// - ρ > 0.75: expand radius (× 2, capped at delta_max), accept step.
/// - 0.25 ≤ ρ ≤ 0.75: keep radius, accept step.
/// - ρ < 0.25: shrink radius (÷ 4), reject step.
fn update_trust_radius(rho: f64, current: f64, delta_max: f64) -> (f64, bool) {
    if rho < 0.25 {
        (current / 4.0, false)
    } else if rho > 0.75 {
        ((current * 2.0).min(delta_max), true)
    } else {
        (current, true)
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

/// Returns the index into `model.eta_names` of the ETA log-mu-referenced to
/// theta `k`, or `None` if no such pairing exists.
///
/// Only log-transformed pairs (`MuRef::log_transformed == true`) qualify.
/// Additive pairs (`PARAM = THETA + ETA`) do not: ferx packs all thetas as
/// `x_k = log(THETA)`, so ∂f/∂x_k = THETA · ∂f/∂η ≠ H[:,j] there.
fn mu_ref_eta_index(model: &CompiledModel, template: &ModelParameters, k: usize) -> Option<usize> {
    let theta_name = template.theta_names.get(k)?;
    let (eta_name, _) = model
        .mu_refs
        .iter()
        .find(|(_, mu_ref)| mu_ref.log_transformed && mu_ref.theta_name == *theta_name)?;
    model.eta_names.iter().position(|n| n == eta_name)
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

    // This analytical-gradient path dispatches the dr/dsigma terms on the
    // single `model.error_model`. It is only ever entered for analytical PK
    // models (`subject_nll_pop_grad` gates on `ode_spec.is_none()`), and
    // per-CMT error models are ODE-only — so the error spec is always Single
    // here and the representative error model is exact. The assert locks that
    // invariant against future drift; ODE / per-CMT models take the FD path,
    // which dispatches through `error_spec`.
    debug_assert!(
        matches!(model.error_spec, ErrorSpec::Single(_)),
        "analytical GN gradient reached with a non-Single error spec"
    );
    // The SB analytical gradient is consistent with `foce_subject_nll_standard`
    // (the FOCE-without-interaction marginal). Under `options.interaction` the
    // outer optimiser minimises the Almquist Laplace marginal instead — a
    // different NLL whose gradient does not match this function — so the
    // dispatcher routes interaction=true to `subject_nll_pop_grad_analytical_laplace`.
    // Lock the invariant against direct callers that might bypass the dispatcher.
    debug_assert!(
        !options.interaction,
        "subject_nll_pop_grad_analytical (SB form) called with options.interaction=true; \
         use subject_nll_pop_grad_analytical_laplace for the Almquist Laplace gradient"
    );

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

    // FOCE (no interaction): evaluate R at the population prediction f(η=0),
    // matching `foce_subject_nll_standard`. f0 = f(η̂) − H·η̂ can cross zero on
    // a nonlinear model and make R̃ ill-conditioned; f(η=0) is always sensible.
    // Additive error is f-independent → keep f0 (bit-identical, no extra eval).
    let use_pop_var = model.error_spec.has_f_dependent_variance();
    let zeros_eta = vec![0.0_f64; n_eta];
    let pop_preds: Vec<f64> = if use_pop_var {
        pk::compute_predictions_with_tv(model, subject, &params.theta, &zeros_eta)
    } else {
        Vec::new()
    };
    let r_pred_point: &[f64] = if use_pop_var { &pop_preds } else { &f0 };
    let r_diag = compute_r_diag(
        &model.error_spec,
        r_pred_point,
        &subject.obs_cmts,
        &params.sigma.values,
    );

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
    'theta: for k in 0..n_theta {
        if fixed_mask[k] {
            continue;
        }

        // d(ipreds)/dx_k: mu-ref shortcut reads H[:,j] (already computed);
        // FD fallback perturbs theta and re-evaluates predictions.
        let d_ipreds: Vec<f64> = 'fd: {
            if options.mu_referencing {
                if let Some(j) = mu_ref_eta_index(model, template, k) {
                    break 'fd h_matrix.column(j).iter().copied().collect();
                }
            }
            let h = eps * (1.0 + x[k].abs());
            let xk_plus = (x[k] + h).min(bounds.upper[k]);
            let actual_h = xk_plus - x[k];
            if actual_h.abs() < 1e-16 {
                continue 'theta;
            }
            let mut x_pert = x.to_vec();
            x_pert[k] = xk_plus;
            let params_pert = unpack_params(&x_pert, template);
            pk::compute_predictions_with_tv(model, subject, &params_pert.theta, eta_hat.as_slice())
                .iter()
                .zip(ipreds.iter())
                .map(|(&p, &b)| (p - b) / actual_h)
                .collect()
        };

        // d(f(η=0))/dx_k for the variance chain rule, when R is evaluated at the
        // population prediction (f-dependent error). No mu-ref shortcut: the
        // H-column gives ∂f/∂η at η̂, not the θ-derivative of f at η=0, so this
        // is always FD. Cheap (one extra prediction eval per free θ) and only on
        // the f-dependent path. For additive error `dr_j == 0`, so it is unused.
        let d_pop_preds: Vec<f64> = if use_pop_var {
            let h = eps * (1.0 + x[k].abs());
            let xk_plus = (x[k] + h).min(bounds.upper[k]);
            let actual_h = xk_plus - x[k];
            if actual_h.abs() < 1e-16 {
                vec![0.0; n_obs]
            } else {
                let mut x_pert = x.to_vec();
                x_pert[k] = xk_plus;
                let params_pert = unpack_params(&x_pert, template);
                pk::compute_predictions_with_tv(model, subject, &params_pert.theta, &zeros_eta)
                    .iter()
                    .zip(pop_preds.iter())
                    .map(|(&p, &b)| (p - b) / actual_h)
                    .collect()
            }
        } else {
            Vec::new()
        };

        // Variance-point derivative: d(f(η=0)) on the f-dependent path, else
        // d(f0)=d(ipreds). The mean term below always uses d(f0)=d_ipreds.
        let d_var_pred: &[f64] = if use_pop_var { &d_pop_preds } else { &d_ipreds };

        // d(f0) = d(ipreds); d(v) = -d(f0)
        // For sigma-dependent r: d(r_j)/d(x[k]) via chain rule through r at r_pred_point
        let dr: Vec<f64> = r_diag
            .iter()
            .zip(r_pred_point.iter().zip(d_var_pred.iter()))
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
    let free_mask = &template.omega.free_mask;

    for (ko, &(row, col)) in omega_entries.iter().enumerate() {
        let k = omega_start + ko;
        if fixed_mask[k] {
            continue;
        }
        // Structural zero (cross-block off-diagonal in a multi-block_omega
        // declaration): same reasoning as the Laplace path.
        if !free_mask[(row, col)] {
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

/// Analytical per-subject FOCEI **INTER** NLL gradient — Almquist 2015 Laplace
/// form. Same structure as [`subject_nll_pop_grad_analytical`] but matches the
/// NLL `foce_subject_nll_interaction` is computing under `options.interaction`:
///
/// ```text
///   NLL_i = 0.5 · [ data_ll(η̂) + η̂'·Ω⁻¹·η̂ + log|Ω| + log|H̃| ]
/// ```
/// with
/// ```text
///   data_ll(η̂) = Σⱼ [(yⱼ−fⱼ)²/Rⱼ + log Rⱼ]      (R evaluated at η̂)
///   H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹
///   c̃_{j,k} = (∂Rⱼ/∂fⱼ)·a_{j,k} / Rⱼ
/// ```
///
/// Like the SB path, η̂ and H (= a) are held fixed under all parameter
/// perturbations — the "fixed-EBE" gradient. The chain rule then closes
/// cheaply on small (n_eta × n_eta) Hessian matrices:
///   - **θ_k**: 1 forward-FD call on `compute_predictions_with_tv` per θ_k,
///     then `0.5·Σⱼ (αⱼ + βⱼ·qⱼ) · ∂fⱼ/∂θ_k` where
///     αⱼ = −2·errⱼ/Rⱼ + dⱼ·(Rⱼ − errⱼ²)/Rⱼ²    (data_ll piece)
///     βⱼ = −dⱼ/Rⱼ² + dⱼ·d2ⱼ/Rⱼ² − dⱼ³/Rⱼ³     (log|H̃| piece via chain on R, c̃)
///     qⱼ = aⱼ'·H̃⁻¹·aⱼ                          (pre-computed once)
///     with dⱼ = ∂R/∂f, d2ⱼ = ∂²R/∂f².
///   - **Ω** (Cholesky-packed): closed form using
///     z = Ω⁻¹·η̂   and   G = Ω⁻¹·H̃⁻¹·Ω⁻¹.
///     For diagonal x_k = log L_kk:
///       ∂NLL/∂x_k = −L_kk·z_k·(v_k'·z) + 1 − L_kk·(G·v_k)_k
///     For off-diagonal x_k = L[i,j] (block Ω only):
///       ∂NLL/∂x_k = −z_i·(v_j'·z) − (G·v_j)_i
///     where v_k = L[:,k] is the k-th column of the Ω Cholesky factor.
///   - **σ_s**: closed form using ∂R/∂log σ_s and ∂d/∂log σ_s on the same
///     scalar `qⱼ` reservoir.
///
/// **Cost** is the same order as the SB analytical path: one prediction call
/// per non-fixed θ, then a single per-obs sweep for the inner accumulators.
/// The n_eta × n_eta matrix ops (Cholesky, inverse, G product) are
/// constant-cost on n_eta ≲ 10 — far cheaper than the n_obs × n_obs Cholesky
/// the SB path uses.
///
/// Returns `None` (caller falls back to central FD) when `H̃` is not PD.
#[allow(clippy::too_many_arguments)]
fn subject_nll_pop_grad_analytical_laplace(
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
    subject_nll_pop_grad_analytical_laplace_cached(
        x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
    )
    .map(|(nll, grad, _)| (nll, grad))
}

/// Per-subject Laplace intermediates the FOCEI θ/Ω/σ gradient already forms,
/// captured so the #274 covariance EBE-response correction can reuse them
/// instead of recomputing the predictions and re-factorising `H̃`. Every field
/// is evaluated at the same `(η̂, parameter)` point as the gradient that
/// produced it, so a correction built from the cache is bit-identical to one
/// that recomputes from scratch. See [`subject_eta_response_correction`].
pub(crate) struct LaplaceGradCache {
    /// Per-observation residual variance `Rⱼ`.
    pub r_diag: Vec<f64>,
    /// Per-observation `dⱼ = ∂R/∂f`.
    pub d_vec: Vec<f64>,
    /// Per-observation `d2ⱼ = ∂²R/∂f²`.
    pub d2_vec: Vec<f64>,
    /// `G = a'diag(1/R)a` (the `hrh` accumulator — `H̃` without the `½c̃'c̃` and
    /// `Ω⁻¹` terms).
    pub hrh: DMatrix<f64>,
    /// `H̃⁻¹`.
    pub htilde_inv: DMatrix<f64>,
    /// Per-observation `qⱼ = aⱼ'H̃⁻¹aⱼ`.
    pub q: Vec<f64>,
}

/// As [`subject_nll_pop_grad_analytical_laplace`], but also returns the
/// [`LaplaceGradCache`] of reusable per-subject intermediates.
#[allow(clippy::too_many_arguments)]
fn subject_nll_pop_grad_analytical_laplace_cached(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Option<(f64, Vec<f64>, LaplaceGradCache)> {
    use crate::pk;

    // The Almquist Laplace gradient now supports both `ErrorSpec::Single` and
    // `ErrorSpec::PerCmt`. The per-CMT routing flows through `dvar_df`,
    // `dvar_dlogsigma`, `d2var_df2`, and `variance_at` — every variance-related
    // call site already takes `subject.obs_cmts[j]`. The caller still gates on
    // analytical PK and excludes M3 (`can_use_analytical` in
    // `subject_nll_pop_grad`).
    // Sibling debug_assert to the SB path: this function computes the Almquist
    // Laplace gradient, which is only consistent with the NLL the outer FOCE
    // loop minimises when `options.interaction == true`. The dispatcher already
    // enforces this; the assert guards direct callers.
    debug_assert!(
        options.interaction,
        "subject_nll_pop_grad_analytical_laplace called with options.interaction=false; \
         use subject_nll_pop_grad_analytical for the Sheiner–Beal gradient"
    );
    // `options` is accepted for signature symmetry with the SB path and to
    // future-proof against new `FitOptions` fields that influence the Laplace
    // NLL (e.g. a regulariser, a robust-variance toggle). The current
    // implementation only reads `options.interaction` via the assert above.
    let _ = options;

    let n = x.len();
    let n_eta = model.n_eta;
    let n_theta = template.theta.len();
    let n_sigma = template.sigma.values.len();
    let n_obs = population.subjects[subj_idx].observations.len();
    let subject = &population.subjects[subj_idx];
    let params = unpack_params(x, template);
    let fixed_mask = packed_fixed_mask(template);
    let omega = &params.omega;
    let sigma_values = &params.sigma.values;
    let error_spec = &model.error_spec;

    // ── Base quantities at the current parameter point ───────────────────────
    // ipreds, residuals, R, d = ∂R/∂f, d2 = ∂²R/∂f² per observation.
    let ipreds = pk::compute_predictions_with_tv(model, subject, &params.theta, eta_hat.as_slice());
    if ipreds.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let err: Vec<f64> = (0..n_obs)
        .map(|j| subject.observations[j] - ipreds[j])
        .collect();
    let r_diag: Vec<f64> = (0..n_obs)
        .map(|j| error_spec.variance_at(subject.obs_cmts[j], ipreds[j], sigma_values))
        .collect();
    if r_diag.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return None;
    }
    let d_vec: Vec<f64> = (0..n_obs)
        .map(|j| error_spec.dvar_df(subject.obs_cmts[j], ipreds[j], sigma_values))
        .collect();
    // ∂²R/∂f² per observation. f-independent for additive/proportional/combined
    // (0 for additive, 2·σ_prop² otherwise), but the *per-CMT* value can differ
    // across observations: e.g. an Emax PK/PD model with proportional error on
    // PK (CMT=2) and additive on PD (CMT=3) needs `d2 = 2·σ_prop²` at PK obs
    // and `d2 = 0` at PD obs. Dispatch through `error_spec.d2var_df2` so the
    // PerCmt case picks up each endpoint's own contribution; `Single` ignores
    // the cmt argument and returns the scalar value uniformly. The β_j chain
    // at line ~1095 then reads `d2_vec[j]` per obs.
    let d2_vec: Vec<f64> = (0..n_obs)
        .map(|j| error_spec.d2var_df2(subject.obs_cmts[j], sigma_values))
        .collect();

    // Conditional Hessian H̃ = a'·diag(1/R)·a + ½·c̃'·c̃ + Ω⁻¹.
    // c̃_{j,k} = dⱼ·a_{j,k}/Rⱼ; we only need the symmetric outer products,
    // which collapse cleanly into the (n_eta × n_eta) accumulator.
    let mut hrh = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut ctc = DMatrix::<f64>::zeros(n_eta, n_eta);
    for j in 0..n_obs {
        let aj = h_matrix.row(j);
        let inv_r = 1.0 / r_diag[j];
        let c_scale = d_vec[j] * inv_r;
        let cs2 = c_scale * c_scale;
        for a in 0..n_eta {
            let aa = aj[a];
            for b in 0..n_eta {
                let outer = aa * aj[b];
                hrh[(a, b)] += outer * inv_r;
                ctc[(a, b)] += outer * cs2;
            }
        }
    }
    let htilde = &hrh + 0.5 * &ctc + &omega.inv;
    let htilde_chol = htilde.clone().cholesky()?;
    let log_det_htilde = chol_log_det(&htilde_chol.l());
    let htilde_inv = htilde_chol.inverse();

    // Per-obs scalar qⱼ = aⱼ'·H̃⁻¹·aⱼ — central reservoir for the log|H̃|
    // chain rule. n_eta² flops per obs.
    let mut q = vec![0.0f64; n_obs];
    for j in 0..n_obs {
        let aj = h_matrix.row(j);
        let mut s = 0.0;
        for a in 0..n_eta {
            for b in 0..n_eta {
                s += aj[a] * htilde_inv[(a, b)] * aj[b];
            }
        }
        q[j] = s;
    }

    // ── NLL at this parameter point ──────────────────────────────────────────
    let mut data_ll = 0.0_f64;
    for j in 0..n_obs {
        let r = r_diag[j];
        let e = err[j];
        data_ll += e * e / r + r.ln();
    }
    let eta_prior = eta_hat.dot(&(&omega.inv * eta_hat));
    let log_det_omega = omega.log_det;
    let nll = 0.5 * (data_ll + eta_prior + log_det_omega + log_det_htilde);

    // ── Theta gradient (forward FD on predictions; closed-form chain rest) ──
    // Per-obs scalar coeff combines data_ll and log|H̃| contributions:
    //   per_j = αⱼ + βⱼ·qⱼ
    //   αⱼ   = −2·errⱼ/Rⱼ + dⱼ·(Rⱼ − errⱼ²)/Rⱼ²
    //   βⱼ   = −dⱼ/Rⱼ² + dⱼ·d2ⱼ/Rⱼ² − dⱼ³/Rⱼ³
    let mut theta_per_j = vec![0.0f64; n_obs];
    for j in 0..n_obs {
        let r = r_diag[j];
        let inv_r = 1.0 / r;
        let inv_r2 = inv_r * inv_r;
        let d = d_vec[j];
        let d2 = d2_vec[j]; // per-obs / per-CMT; constant in f for the current error models
        let e = err[j];
        let alpha_j = -2.0 * e * inv_r + d * (r - e * e) * inv_r2;
        let beta_j = logdet_htilde_beta(d, d2, inv_r);
        theta_per_j[j] = alpha_j + beta_j * q[j];
    }

    let mut grad = vec![0.0f64; n];
    let eps = 1e-5;
    for k in 0..n_theta {
        if fixed_mask[k] {
            continue;
        }
        // mu-ref shortcut: d(ipreds)/dx_k == H[:,j] for a paired theta/eta.
        if options.mu_referencing {
            if let Some(j) = mu_ref_eta_index(model, template, k) {
                let s: f64 = (0..n_obs)
                    .map(|obs| theta_per_j[obs] * h_matrix[(obs, j)])
                    .sum();
                grad[k] = 0.5 * s;
                continue;
            }
        }
        // FD fallback for non-mu-referenced thetas.
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
        if ipreds_pert.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let s: f64 = (0..n_obs)
            .map(|j| {
                let df_j = (ipreds_pert[j] - ipreds[j]) / actual_h;
                theta_per_j[j] * df_j
            })
            .sum();
        grad[k] = 0.5 * s;
    }

    // ── Omega gradient (closed-form chain rule through Ω⁻¹) ─────────────────
    // z = Ω⁻¹·η̂;  G = Ω⁻¹·H̃⁻¹·Ω⁻¹ — both (n_eta × n_eta) operations.
    let z: DVector<f64> = &omega.inv * eta_hat;
    let g_mat: DMatrix<f64> = &omega.inv * &htilde_inv * &omega.inv;

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
    let l_omega = &omega.chol;
    let free_mask = &template.omega.free_mask;

    for (ko, &(row, col)) in omega_entries.iter().enumerate() {
        let k = omega_start + ko;
        if fixed_mask[k] {
            continue;
        }
        // Structural zero (cross-block off-diagonal in a multi-block_omega
        // declaration): the model declares L[row, col] ≡ 0, so its gradient
        // is zero by construction. Skipping here prevents the outer
        // optimiser from pulling these slots away from zero on the strength
        // of an in-block-only chain rule.
        if !free_mask[(row, col)] {
            continue;
        }
        // v = L[:,col]
        let v_vec: Vec<f64> = (0..n_eta).map(|r| l_omega[(r, col)]).collect();
        let v_dot_z: f64 = v_vec.iter().zip(z.iter()).map(|(a, b)| a * b).sum();
        // (G·v)_row
        let mut gv_row = 0.0_f64;
        for c in 0..n_eta {
            gv_row += g_mat[(row, c)] * v_vec[c];
        }
        if row == col {
            // Diagonal: x_k = log L_kk. NLL has a 0.5·[…] outer factor; the
            // η'Ω⁻¹η, log|Ω|, and log|H̃| contributions each carry an inner
            // factor of 2 that cancels with the outer 0.5, so the RHS is the
            // *full* ∂NLL/∂x_k (not half of it).
            //   ∂NLL/∂x_k = -L_kk·z_k·(v'z) + 1 - L_kk·(G·v)_k
            let l_kk = l_omega[(row, row)];
            grad[k] = -l_kk * z[row] * v_dot_z + 1.0 - l_kk * gv_row;
        } else {
            // Off-diagonal: x_k = L[i,j] (i > j). log|Ω| contribution is 0
            // because L's off-diagonals do not enter ∏ L_ii. Same 0.5/2
            // cancellation as the diagonal case → RHS is the full ∂NLL/∂x_k.
            //   ∂NLL/∂x_k = -z_i·(v'z) - (G·v)_i
            grad[k] = -z[row] * v_dot_z - gv_row;
        }
    }

    // ── Sigma gradient (closed-form chain rule through R and c̃) ────────────
    // Per sigma index s (∈ flat sigma vector):
    //   ∂R/∂log σ_s, ∂d/∂log σ_s per obs (from `dvar_dlogsigma` and the
    //   error-model dispatch — `dvar_dlogsigma` is already the right hook
    //   for ∂R/∂log σ; ∂d/∂log σ for the proportional component is 2·d
    //   (combined or proportional), and 0 for additive — see the analytical
    //   SB path's `dr_diag_d_log_sigma` for the matching idiom).
    //
    // Resolve SigmaType through `error_spec` (the same dispatcher every other
    // variance call uses), not `model.error_model`, so this stays internally
    // consistent under any future refactor that lets the two diverge.
    let sigma_types = error_spec.sigma_types(n_sigma);
    let sigma_start = omega_start + omega_entries.len();
    for ks in 0..n_sigma {
        let k = sigma_start + ks;
        if fixed_mask[k] {
            continue;
        }
        let dr_per_obs: Vec<f64> = (0..n_obs)
            .map(|j| error_spec.dvar_dlogsigma(subject.obs_cmts[j], ks, ipreds[j], sigma_values))
            .collect();
        // ∂d/∂log σ_s. For proportional and combined, d = 2·σ_prop²·f, so
        // ∂d/∂log σ_prop = 2·d and ∂d/∂log σ_add = 0. For additive d = 0
        // identically.
        let dd_factor = match sigma_types.get(ks).copied() {
            Some(SigmaType::Proportional) => 2.0,
            _ => 0.0,
        };

        let mut s_acc = 0.0_f64;
        for j in 0..n_obs {
            let r = r_diag[j];
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let inv_r3 = inv_r2 * inv_r;
            let d = d_vec[j];
            let e = err[j];
            let dr = dr_per_obs[j];
            let dd = d * dd_factor;
            // data_ll piece: ∂R·(1/R − err²/R²) = ∂R·(R − err²)/R²
            let data_term = dr * (r - e * e) * inv_r2;
            // log|H̃| piece via chain on R and on c̃:
            //   γⱼ = -∂R/R² + d·∂d/R² - d²·∂R/R³
            let gamma_j = -dr * inv_r2 + d * dd * inv_r2 - d * d * dr * inv_r3;
            s_acc += data_term + gamma_j * q[j];
        }
        grad[k] = 0.5 * s_acc;
    }

    // Late finiteness guard: the `r > 0` admit at line 970 accepts arbitrarily
    // small positive r (e.g. 1e-160 → inv_r³ overflows to +∞), and an
    // ill-conditioned H̃ that just barely passes Cholesky can produce an
    // ~1e300 inverse that overflows q[j] downstream. In either case the
    // assembled NLL or gradient picks up ±∞/NaN and would silently poison
    // SLSQP. Bail to the FD fallback instead — `subject_nll_pop_grad` will
    // central-FD over `subject_nll_at`, which clamps non-finite NLLs to the
    // 1e20 sentinel and then takes the one-sided FD fork on either side.
    if !nll.is_finite() || grad.iter().any(|g| !g.is_finite()) {
        return None;
    }

    let cache = LaplaceGradCache {
        r_diag,
        d_vec,
        d2_vec,
        hrh,
        htilde_inv,
        q,
    };
    Some((nll, grad, cache))
}

/// Per-observation coefficient of the `qⱼ = aⱼ'H̃⁻¹aⱼ` reservoir in the `log|H̃|`
/// chain rule along the prediction axis: `βⱼ = −dⱼ/Rⱼ² + dⱼ·d2ⱼ/Rⱼ² − dⱼ³/Rⱼ³`,
/// where `dⱼ = ∂R/∂f` and `d2ⱼ = ∂²R/∂f²`. Shared by the Laplace θ-gradient
/// (`∂log|H̃|/∂θ` via `df/dθ`) and the #274 EBE-response correction
/// (`∂log|H̃|/∂η` via `df/dη = a`), so the formula lives in one place.
#[inline]
fn logdet_htilde_beta(d: f64, d2: f64, inv_r: f64) -> f64 {
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    -d * inv_r2 + d * d2 * inv_r2 - d * d * d * inv_r3
}

/// Leading-order estimate of the per-subject `log|H̃|` EBE-response gradient term
/// that the fixed-η̂ analytic Laplace gradient drops.
///
/// The analytic Laplace gradient computes `∂NLL_i/∂θ` holding η̂ fixed and
/// invoking the envelope theorem — which zeros only the *inner* objective
/// (`data_ll + η'Ω⁻¹η`), NOT `log|H̃|`. The true total gradient therefore carries
/// an extra term:
/// ```text
///   dΦ_i/dθ = ∂NLL_i/∂θ + (∂NLL_i/∂η)·(dη̂_i/dθ),   ∂NLL_i/∂η = ½ ∂log|H̃_i|/∂η
///   t_i[k]  = ½ (∂log|H̃_i|/∂η)' · (dη̂_i/dθ_k)
/// ```
/// Adding `t_i` back to the covariance-step gradient (only) makes the central FD
/// of that gradient recover the full marginal Hessian `∇²(−2logL)` — including the
/// EBE-response cross-curvature `Δ = d/dθ[½ ∂log|H̃|/∂η · dη̂/dθ]` that the non-IOV
/// stencil otherwise omits (the term the IOV scalar-OFV-2nd-difference captures).
///
/// For a mu-referenced `θ_k ↔ η_{j'}` every factor is already formed by the
/// Laplace gradient, so this reduces to a few `n_eta × n_eta` products:
/// ```text
///   gη_i[m] = Σ_j β_j q_j a_{j,m}        (= ∂log|H̃_i|/∂η_m, a-fixed / Fisher approx)
///   G_i     = a'diag(1/R)a   (= hrh),    dη̂_i/dθ_k = −H̃_i⁻¹ G_i[:,j']   (IFT, GN)
///   t_i[k]  = −½ ( G_i · H̃_i⁻¹ · gη_i )_{j'}
/// ```
/// Returns `None` on an ill-conditioned point (caller then skips the correction).
/// Covers the **mu-ref θ block** (`m_k = G[:,j']`). The σ SE is corrected
/// indirectly through the θ/σ off-diagonals (matrix coupling), so no σ-direct term
/// is needed. The ω block (closed form `m_k = ∂Ω⁻¹/∂x_k·η̂`) is deferred: its
/// `z_i·(H̃⁻¹g^η)_i` form lacks the θ block's `G·H̃⁻¹ ≈ I` cancellation, so it is
/// sensitive to the leading-order `g^η` (dropped `∂²f/∂η²`) and overshoots
/// large-IIV components — see issue #274.
///
/// `cache` carries the per-subject Laplace intermediates (`R`, `d`, `d2`, `G`,
/// `H̃⁻¹`, `q`) the FOCEI gradient already formed at this point. The covariance
/// step passes `Some(..)` so the correction does not recompute the predictions
/// or re-factorise `H̃`. Pass `None` to have the correction re-derive them
/// itself (used by the unit test and any FD-fallback subject).
pub(crate) fn subject_eta_response_correction(
    cache: Option<&LaplaceGradCache>,
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    bounds: &PackedBounds,
    options: &FitOptions,
) -> Option<Vec<f64>> {
    let n = x.len();
    let n_eta = model.n_eta;
    let n_theta = template.theta.len();
    let n_obs = population.subjects[subj_idx].observations.len();
    if n_eta == 0 || n_obs == 0 {
        return Some(vec![0.0; n]);
    }

    // Reuse the gradient's per-subject intermediates when the covariance step
    // already formed them (the common path); otherwise re-derive them by running
    // the cached Laplace gradient here. Either way the intermediates are the same
    // `R/d/d2/G/H̃⁻¹/q` the gradient uses, so the correction is unchanged.
    let owned;
    let c: &LaplaceGradCache = match cache {
        Some(c) => c,
        None => {
            let (_, _, computed) = subject_nll_pop_grad_analytical_laplace_cached(
                x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
            )?;
            owned = computed;
            &owned
        }
    };

    // gη_m = Σ_j β_j q_j a_{j,m}: the η-gradient of log|H̃| under the same a-fixed
    // chain the θ-gradient uses (q_j = a_j'H̃⁻¹a_j; β_j the log|H̃| coefficient).
    let mut g_eta = DVector::<f64>::zeros(n_eta);
    for j in 0..n_obs {
        let aj = h_matrix.row(j);
        let beta_j = logdet_htilde_beta(c.d_vec[j], c.d2_vec[j], 1.0 / c.r_diag[j]);
        let coef = beta_j * c.q[j];
        for m in 0..n_eta {
            g_eta[m] += coef * aj[m];
        }
    }

    // u = G·H̃⁻¹·gη;  t_i[k] = −½ u_{j'} for mu-ref θ_k↔η_{j'}. A non-finite entry
    // can only reach the result through a mapped `u[jp]`, which the final
    // finiteness check below catches (returning `None` so the subject contributes
    // 0) — no separate intermediate guard needed.
    let u = &c.hrh * (&c.htilde_inv * &g_eta);
    let fixed_mask = packed_fixed_mask(template);
    let mut t = vec![0.0f64; n];

    // ── θ block (mu-ref): dη̂/dθ_k = −H̃⁻¹ G[:,j'],  t_i[k] = −½ u_{j'}. ──
    if options.mu_referencing {
        for k in 0..n_theta {
            if fixed_mask[k] {
                continue;
            }
            if let Some(jp) = mu_ref_eta_index(model, template, k) {
                t[k] = -0.5 * u[jp];
            }
        }
    }

    if t.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(t)
}

/// Compute the FOCE NLL and its gradient w.r.t. the packed population parameter
/// vector for a single subject, with ETAs fixed at their current EBE values.
///
/// Two analytical paths with different scope (see `sb_ok` / `laplace_ok` below):
///   - **Almquist Laplace (INTER)**: omega/sigma exact, θ via forward-FD of
///     predictions. Runs for any model except M3 BLOQ and IOV — ODE models and
///     per-CMT (`ErrorSpec::PerCmt`) error specs are both supported.
///   - **Sheiner–Beal (non-INTER)**: same θ axis, but the chain rule still
///     assumes a single error spec and `ipreds = f₀ + a·η̂`, so this branch
///     additionally requires analytical PK and `ErrorSpec::Single`.
///
/// In all other cases — M3 BLOQ, IOV, or the SB path's extra restrictions —
/// the dispatcher falls back to central FD over `subject_nll_at`. The Laplace
/// branch can also bail to FD per call when `H̃` fails Cholesky.
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
    // IOV uses the FD path below: the per-subject NLL is now the proper
    // augmented marginal (`foce_subject_nll_iov`, issue #101) whose gradient
    // w.r.t. Ω_iov flows through R̃, not a closed-form κ prior — so FD over
    // `subject_nll_at` stays exactly consistent with the objective the outer
    // FOCE loop minimises. (The old analytical κ-prior gradient matched the
    // since-removed MAP-penalty objective and would now disagree.)
    // Two analytical paths with different scope:
    //   - Sheiner–Beal (no INTER): analytical PK only, single error spec — the
    //     SB chain rule still assumes `ipreds = f₀ + a·η̂` (a linear surface)
    //     and a single (∂R/∂σ_k, ∂d/∂σ_k) per obs, both untouched here.
    //   - Almquist Laplace (INTER): every variance call already dispatches
    //     through `error_spec` (so per-CMT works), and the θ axis is a
    //     forward-FD on `pk::compute_predictions_with_tv` — which itself
    //     dispatches to the ODE solver for ODE models. So Laplace can run on
    //     ODE + PerCmt models too; the only blockers are M3 and IOV.
    let common_ok = !matches!(model.bloq_method, BloqMethod::M3) && kappas.is_empty();
    let sb_ok =
        common_ok && model.ode_spec.is_none() && matches!(model.error_spec, ErrorSpec::Single(_));
    let laplace_ok = common_ok;

    if (options.interaction && laplace_ok) || (!options.interaction && sb_ok) {
        // Dispatch to the form whose gradient is exactly consistent with the
        // NLL `foce_subject_nll` is computing for this subject:
        //   - `options.interaction == true`  → Almquist 2015 Laplace
        //     (`data_ll + η̂'Ω⁻¹η̂ + log|Ω| + log|H̃|`, with H̃ carrying the
        //      `½·c̃'·c̃` INTER correction). See
        //      `subject_nll_pop_grad_analytical_laplace`.
        //   - `options.interaction == false` → Sheiner–Beal linearised marginal
        //     (`(y - f₀)' R̃⁻¹ (y - f₀) + log|R̃|` at R(f₀)). See
        //      `subject_nll_pop_grad_analytical`.
        let result = if options.interaction {
            subject_nll_pop_grad_analytical_laplace(
                x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
            )
        } else {
            subject_nll_pop_grad_analytical(
                x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
            )
        };
        if let Some(result) = result {
            return result;
        }
    }

    // Fallback: central FD over full per-subject NLL.
    //
    // `subject_nll_at` (via `foce_subject_nll_interaction` / `foce_subject_nll_standard`)
    // returns the `1e20` sentinel from `stats::likelihood` for ill-conditioned
    // states (non-PD R̃ / H̃, non-finite intermediate NLL). Central-FD'ing
    // across that sentinel — or differencing two finite values where one is
    // the sentinel — would push a ~1e24/h gradient component into the outer
    // optimiser. Map the sentinel onto +∞ before differencing so the existing
    // one-sided / zero-gradient fork handles it the same as a NaN.
    let n = x.len();
    let fixed_mask = packed_fixed_mask(template);
    let eps = 1e-4;
    // The sentinel is the largest finite NLL likelihood.rs ever returns
    // (~1e20). Anything ≥ that bound is treated as "ill-conditioned" and
    // hidden from the FD difference; using `>=` keeps us robust to a future
    // sentinel bump.
    const NLL_SENTINEL_THRESHOLD: f64 = 1e20;
    fn mask_sentinel(nll: f64) -> f64 {
        if nll.is_finite() && nll < NLL_SENTINEL_THRESHOLD {
            nll
        } else {
            f64::INFINITY
        }
    }

    let params_base = unpack_params(x, template);
    let nll_base_raw = subject_nll_at(
        model,
        population,
        subj_idx,
        &params_base,
        eta_hat,
        h_matrix,
        kappas,
        options,
    );
    let nll_base_masked = mask_sentinel(nll_base_raw);

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
        let nll_plus = mask_sentinel(subject_nll_at(
            model,
            population,
            subj_idx,
            &params_plus,
            eta_hat,
            h_matrix,
            kappas,
            options,
        ));

        x_work[j] = xj_minus;
        let params_minus = unpack_params(&x_work, template);
        let nll_minus = mask_sentinel(subject_nll_at(
            model,
            population,
            subj_idx,
            &params_minus,
            eta_hat,
            h_matrix,
            kappas,
            options,
        ));

        x_work[j] = x[j];

        let deriv = (nll_plus - nll_minus) / actual_2h;
        grad[j] = if deriv.is_finite() {
            deriv
        } else if nll_plus.is_finite() && nll_base_masked.is_finite() {
            // One-sided fallback: minus-side was non-finite or sentinel.
            (nll_plus - nll_base_masked) / (xj_plus - x[j])
        } else if nll_minus.is_finite() && nll_base_masked.is_finite() {
            // One-sided fallback: plus-side was non-finite or sentinel.
            (nll_base_masked - nll_minus) / (x[j] - xj_minus)
        } else {
            // Both sides ill-conditioned: gradient is undefined here. Returning
            // 0 lets the outer optimiser step elsewhere instead of stalling on
            // a ±1e24/h spike. NLL itself stays at the raw (unmasked) sentinel
            // so the outer line search still knows the move was infeasible.
            0.0
        };
    }

    (nll_base_raw, grad)
}

/// [`subject_nll_pop_grad`] that additionally returns the [`LaplaceGradCache`]
/// when this subject took the FOCEI Laplace analytical path — letting the
/// covariance step's #274 EBE-response correction reuse the predictions and `H̃`
/// factorisation rather than recomputing them. The cache is `None` for FOCE, for
/// the M3/IOV/FD-fallback path, and when the Laplace gradient bails (non-PD `H̃`);
/// in every `None` case the returned `(nll, grad)` is exactly what
/// [`subject_nll_pop_grad`] returns, so callers can treat this as a drop-in.
///
/// On the rare Laplace bail the analytical path is attempted twice (once here,
/// once inside the delegated `subject_nll_pop_grad`); this only happens on an
/// ill-conditioned point that was going to fall back to FD anyway.
#[allow(clippy::too_many_arguments)]
pub(crate) fn subject_nll_pop_grad_with_cache(
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
) -> (f64, Vec<f64>, Option<LaplaceGradCache>) {
    let laplace_ok = !matches!(model.bloq_method, BloqMethod::M3) && kappas.is_empty();
    if options.interaction && laplace_ok {
        if let Some((nll, grad, cache)) = subject_nll_pop_grad_analytical_laplace_cached(
            x, template, model, population, subj_idx, eta_hat, h_matrix, bounds, options,
        ) {
            return (nll, grad, Some(cache));
        }
    }
    let (nll, grad) = subject_nll_pop_grad(
        x, template, model, population, subj_idx, eta_hat, h_matrix, kappas, bounds, options,
    );
    (nll, grad, None)
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
            &model.error_spec,
            model.bloq_method,
            &[],
        )
    } else {
        // FOCE (no interaction): evaluate R at the population prediction f(η=0)
        // for f-dependent error, consistent with the marginal in likelihood.rs
        // and with `subject_nll_pop_grad_analytical` (so GN's NLL matches its
        // gradient). Additive error keeps f0 (bit-identical).
        let pop_preds: Option<Vec<f64>> = if model.error_spec.has_f_dependent_variance() {
            let zeros = vec![0.0_f64; eta_hat.len()];
            Some(crate::pk::compute_predictions_with_tv(
                model,
                subject,
                &params.theta,
                &zeros,
            ))
        } else {
            None
        };
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            &params.omega,
            &params.sigma.values,
            &model.error_spec,
            model.bloq_method,
            &[],
            pop_preds.as_deref(),
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
    use nalgebra::{DMatrix, DVector};
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
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
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
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
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
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        }
    }

    fn make_population() -> Population {
        let subjects = (0..3)
            .map(|_| Subject {
                id: "S1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0, 4.0, 8.0],
                obs_raw_times: Vec::new(),
                observations: vec![25.0, 15.0, 9.0],
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: vec![1, 1, 1],
                dose_occasions: vec![1],
                #[cfg(feature = "survival")]
                obs_records: vec![],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// The `log|H̃|` EBE-response correction (#274) must vanish for additive error:
    /// `∂R/∂f = 0` ⇒ `β_j = 0` ⇒ `g^η = 0` ⇒ every `t_i[k] = 0`. This is the Δ=0
    /// control that guarantees the correction is inert when there is no η-σ
    /// interaction (additive residual variance), so additive-error SEs are
    /// unchanged. Contrast: the proportional model gives a non-zero correction.
    #[test]
    fn eta_response_correction_zero_for_additive_nonzero_for_proportional() {
        let pop = make_population();
        let eta_hat = DVector::from_vec(vec![0.15]);
        // Plausible Jacobian dpred/dη (3 obs × 1 eta) — sign/scale immaterial to
        // the additive-zero claim (β_j = 0 regardless).
        let h = DMatrix::from_column_slice(3, 1, &[-0.20, -0.50, -0.60]);
        let mut opts = FitOptions::default();
        opts.interaction = true;
        opts.mu_referencing = true;

        // mu-reference TVCL ↔ ETA_CL (log-transformed) so the θ block is active.
        let mu = || {
            let mut m = HashMap::new();
            m.insert(
                "ETA_CL".to_string(),
                crate::types::MuRef {
                    theta_name: "TVCL".to_string(),
                    log_transformed: true,
                },
            );
            m
        };

        // Proportional: correction is active (some |t| > 0).
        let mut model_p = make_model();
        model_p.mu_refs = mu();
        let template_p = model_p.default_params.clone();
        let xp = pack_params(&template_p);
        let bounds_p = compute_bounds(&template_p);
        // `None` cache: exercise the re-derive path (the covariance step passes a
        // cache, but the unit test checks the standalone correction).
        let tp = subject_eta_response_correction(
            None,
            &xp,
            &template_p,
            &model_p,
            &pop,
            0,
            &eta_hat,
            &h,
            &bounds_p,
            &opts,
        )
        .expect("correction computes (proportional)");
        assert!(
            tp.iter().any(|&v| v.abs() > 0.0),
            "proportional error must give a non-zero EBE-response correction"
        );

        // Additive: correction is exactly zero everywhere (even with the mu-ref).
        let mut model_a = make_model();
        model_a.mu_refs = mu();
        model_a.error_model = ErrorModel::Additive;
        model_a.error_spec = crate::types::ErrorSpec::Single(ErrorModel::Additive);
        let template_a = model_a.default_params.clone();
        let xa = pack_params(&template_a);
        let bounds_a = compute_bounds(&template_a);
        let ta = subject_eta_response_correction(
            None,
            &xa,
            &template_a,
            &model_a,
            &pop,
            0,
            &eta_hat,
            &h,
            &bounds_a,
            &opts,
        )
        .expect("correction computes (additive)");
        for (k, &v) in ta.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "additive error must give zero correction at packed param {k}"
            );
        }
    }

    /// update_trust_radius: rho > 0.75 expands the radius (× 2) and accepts.
    #[test]
    fn test_update_trust_radius_expands_on_good_step() {
        let (radius, accepted) = update_trust_radius(0.9, 1.0, 10.0);
        assert!(accepted, "rho=0.9 must accept");
        assert!(
            (radius - 2.0).abs() < 1e-12,
            "radius must double: got {radius}"
        );
    }

    /// update_trust_radius: rho > 0.75 caps at delta_max.
    #[test]
    fn test_update_trust_radius_capped_at_delta_max() {
        let (radius, accepted) = update_trust_radius(0.9, 6.0, 10.0);
        assert!(accepted);
        assert!(
            (radius - 10.0).abs() < 1e-12,
            "radius must be capped at 10: got {radius}"
        );
    }

    /// update_trust_radius: rho < 0.25 shrinks (÷ 4) and rejects.
    #[test]
    fn test_update_trust_radius_shrinks_and_rejects_on_poor_step() {
        let (radius, accepted) = update_trust_radius(0.1, 1.0, 10.0);
        assert!(!accepted, "rho=0.1 must reject");
        assert!(
            (radius - 0.25).abs() < 1e-12,
            "radius must quarter: got {radius}"
        );
    }

    /// update_trust_radius: 0.25 ≤ rho ≤ 0.75 keeps radius and accepts.
    #[test]
    fn test_update_trust_radius_keeps_radius_on_moderate_step() {
        let (radius, accepted) = update_trust_radius(0.5, 1.0, 10.0);
        assert!(accepted, "rho=0.5 must accept");
        assert!(
            (radius - 1.0).abs() < 1e-12,
            "radius must stay at 1.0: got {radius}"
        );
    }

    /// update_trust_radius: rho = 0.25 exactly falls into the "keep, accept" branch
    /// (boundary of the strict `< 0.25` shrink condition).
    #[test]
    fn test_update_trust_radius_boundary_rho_0_25() {
        let (radius, accepted) = update_trust_radius(0.25, 1.0, 10.0);
        assert!(accepted, "rho=0.25 must accept (boundary of shrink branch)");
        assert!(
            (radius - 1.0).abs() < 1e-12,
            "radius must stay at 1.0 for rho=0.25: got {radius}"
        );
    }

    /// update_trust_radius: rho = 0.75 exactly falls into the "keep, accept" branch
    /// (boundary of the strict `> 0.75` expand condition).
    #[test]
    fn test_update_trust_radius_boundary_rho_0_75() {
        let (radius, accepted) = update_trust_radius(0.75, 1.0, 10.0);
        assert!(accepted, "rho=0.75 must accept (boundary of expand branch)");
        assert!(
            (radius - 1.0).abs() < 1e-12,
            "radius must stay at 1.0 for rho=0.75: got {radius}"
        );
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
        // SB path requires non-interaction; default FitOptions has interaction=true.
        let mut options = FitOptions::default();
        options.interaction = false;

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
        // FOCE non-interaction: the analytical-gradient path was derived for the
        // Sheiner–Beal NLL and only applies to FOCE-non-INTER (`foce_subject_nll_standard`).
        // FOCEI INTER now uses the Almquist Laplace NLL whose gradient takes the
        // FD fallback in `subject_nll_pop_grad`; analytical-vs-FD comparisons
        // under INTER would be vacuous (FD-vs-FD) so we set interaction=false.
        let mut options = FitOptions::default();
        options.interaction = false;

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

    /// (Previously verified the analytical gradient under `interaction=true`;
    /// that bridge is gone since FOCEI INTER now uses the Almquist Laplace
    /// NLL whose gradient takes the FD fallback in `subject_nll_pop_grad`.
    /// We keep the test with `interaction=false` so the analytical SB-derived
    /// path still gets a Cholesky/H-aware check at non-trivial `eta`/`H`.)
    #[test]
    fn test_subject_nll_pop_grad_analytical_at_nontrivial_eta() {
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
        options.interaction = false;

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
        // FOCE non-interaction — see `test_subject_nll_pop_grad_analytical_nonzero_h`.
        let mut options = FitOptions::default();
        options.interaction = false;

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

    /// FOCEI INTER (Almquist Laplace) analytical pop-gradient — checks NLL
    /// matches `foce_subject_nll_interaction` and each component agrees with a
    /// central-FD reference to the same `1e-4 · (1 + |ref|)` tolerance the
    /// SB-path test uses. Exercises non-trivial `eta_hat` and `H` so the
    /// c̃'·c̃ INTER correction is actually carrying weight in `H̃` (the
    /// algebra would simplify to the SB-equivalent if both were zero).
    #[test]
    fn test_subject_nll_pop_grad_analytical_laplace_matches_fd() {
        let model = make_model();
        let population = make_population();
        let template = &model.default_params;

        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.07]);
        // Non-zero H so a'·diag(1/R)·a contributes; combined error so c̃'·c̃
        // contributes too (model.error_model is Proportional by default in
        // `make_model`, which gives non-zero d_j and thus non-zero c̃).
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![2.5, 1.8, 1.2]);
        let mut options = FitOptions::default();
        options.interaction = true;

        // Call the analytical function directly (not through `subject_nll_pop_grad`)
        // so a silent fall-through to the FD path — which would also match the FD
        // reference and pass the test — is impossible. `.expect` surfaces any
        // future fixture regression that drives H̃ non-PD or trips the late
        // finiteness guard.
        let (nll, grad) = subject_nll_pop_grad_analytical_laplace(
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
        .expect("analytical Laplace path must succeed on this fixture");

        // NLL must match a direct subject_nll_at call (which routes through
        // foce_subject_nll_interaction = Almquist Laplace under interaction).
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
            "Laplace nll mismatch: {nll} vs {nll_ref}"
        );

        for (j, g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "Laplace grad[{j}] not finite");
        }

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
                "Laplace grad[{j}]: analytical={:.6e}, FD-ref={:.6e}, diff={:.3e}, tol={:.3e}",
                grad[j],
                ref_j,
                grad[j] - ref_j,
                tol,
            );
        }
    }

    /// Almquist Laplace analytical gradient — combined error variant. The
    /// `½·c̃'·c̃` INTER correction has both σ_prop and σ_add contributions
    /// here (σ_add zeroes the dd/dlogσ piece but still flows through R), so
    /// this is a stricter check on the σ-gradient code path than the
    /// Proportional-only `make_model`. Sigma values are chosen so
    /// `σ_prop²·f² ≈ σ_add²` at the test predictions — that way both
    /// branches of the dd_factor switch carry weight (a dd_factor sign
    /// error on the Proportional slot would shift the σ_prop gradient
    /// well above the 1e-4·(1+|ref|) tolerance).
    #[test]
    fn test_subject_nll_pop_grad_analytical_laplace_combined_error() {
        let mut model = make_model();
        model.error_model = ErrorModel::Combined;
        model.error_spec = ErrorSpec::Single(ErrorModel::Combined);
        // Combined needs two sigmas — replace template's single-σ vector and
        // keep model.n_epsilon / model.sigma_init_as_sd in sync to avoid any
        // downstream code path that reads those fields tripping on a length
        // mismatch with `template.sigma.values`.
        let mut template = model.default_params.clone();
        template.sigma = SigmaVector {
            values: vec![0.3, 0.3],
            names: vec!["PROP_ERR".into(), "ADD_ERR".into()],
        };
        template.sigma_fixed = vec![false, false];
        model.default_params = template.clone();
        model.n_epsilon = 2;
        model.sigma_init_as_sd = vec![false, false];
        let template = &model.default_params;

        let population = make_population();
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.05]);
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![2.0, 1.5, 1.0]);
        let mut options = FitOptions::default();
        options.interaction = true;

        // Direct analytical call — see sibling test for why we bypass the dispatcher.
        let (nll, grad) = subject_nll_pop_grad_analytical_laplace(
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
        .expect("analytical Laplace path must succeed on this fixture");

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
            "Laplace (combined) nll mismatch: {nll} vs {nll_ref}"
        );

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
                "Laplace combined grad[{j}]: analytical={:.6e}, FD-ref={:.6e}, diff={:.3e}, tol={:.3e}",
                grad[j],
                ref_j,
                grad[j] - ref_j,
                tol,
            );
        }
    }

    /// Almquist Laplace analytical gradient — **multi-eta block omega**. This
    /// is the test the diagonal+n_eta=1 fixtures above cannot exercise:
    ///   - `template.omega.diagonal == false` routes through the off-diagonal
    ///     `omega_entries` branch (column-major lower triangle) and the
    ///     `-z[row]·v_dot_z - gv_row` formula at the off-diagonal slot.
    ///   - `n_eta == 2` makes every `for a in 0..n_eta { for b in 0..n_eta }`
    ///     loop in `hrh`, `ctc`, `q_j`, and the `n_eta × n_eta` matrix
    ///     products do real work (a transposed index would now bite).
    /// FD-validate every gradient component to the same 1e-4·(1+|ref|)
    /// tolerance the diagonal tests use.
    #[test]
    fn test_subject_nll_pop_grad_analytical_laplace_block_omega() {
        // Build a 2-eta block-omega fixture: ETA on CL and on V, correlated.
        let omega_matrix = {
            let mut m = nalgebra::DMatrix::<f64>::zeros(2, 2);
            m[(0, 0)] = 0.09;
            m[(1, 1)] = 0.04;
            // ρ ≈ 0.5 correlation: cov = 0.5·√(0.09·0.04) = 0.03
            m[(0, 1)] = 0.03;
            m[(1, 0)] = 0.03;
            m
        };
        let omega =
            OmegaMatrix::from_matrix(omega_matrix, vec!["ETA_CL".into(), "ETA_V".into()], false);

        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false, false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "gn_block_omega_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp(); // CL · exp(ETA_CL)
                p.values[1] = theta[1] * eta[1].exp(); // V  · exp(ETA_V)
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
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false, false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
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
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        };

        let template = &model.default_params;
        let population = make_population();
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        // Block-omega packed layout is 3 entries: log L[0,0], L[1,0], log L[1,1].
        // n = 2 thetas + 3 omega + 1 sigma = 6.
        assert_eq!(n, 6, "block-omega packed layout should be 6 entries");

        let n_obs = 3;
        let n_eta = 2;
        let eta_hat = DVector::from_vec(vec![0.07, -0.04]);
        // Non-zero H so both etas enter every observation (otherwise an
        // off-diagonal index error in the (a,b) loops would silently produce
        // a zero contribution and pass the test).
        let h_matrix = nalgebra::DMatrix::from_vec(
            n_obs,
            n_eta,
            vec![2.5, 1.8, 1.2, /* col 1 */ 1.3, 0.9, 0.5],
        );
        let mut options = FitOptions::default();
        options.interaction = true;

        // Direct analytical call — see sibling test for why we bypass the dispatcher.
        let (nll, grad) = subject_nll_pop_grad_analytical_laplace(
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
        .expect("analytical Laplace path must succeed on this fixture");

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
            "Laplace block-omega nll mismatch: {nll} vs {nll_ref}"
        );

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
                "Laplace block-omega grad[{j}]: analytical={:.6e}, FD-ref={:.6e}, diff={:.3e}, tol={:.3e}",
                grad[j],
                ref_j,
                grad[j] - ref_j,
                tol,
            );
        }
    }

    /// Almquist Laplace analytical gradient — **Additive error**. Exercises
    /// the `d2_scalar = 0`, `d_vec ≡ 0`, `dd_factor = 0` branches that the
    /// Proportional and Combined tests above cannot hit. With d ≡ 0 the
    /// `½·c̃'·c̃` correction collapses to zero (so H̃ = a'·diag(1/R)·a +
    /// Ω⁻¹) and the σ gradient reduces to its data_ll-only form — a
    /// regression in those zero arms (e.g. wiring `dd_factor = 2` to an
    /// Additive slot) would shift `grad[σ]` well above the FD tolerance.
    #[test]
    fn test_subject_nll_pop_grad_analytical_laplace_additive_error() {
        let mut model = make_model();
        model.error_model = ErrorModel::Additive;
        model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
        // Bigger σ_add so R is well-conditioned at the test observations
        // (the default make_model observations span 9..25, so σ_add ≈ 2
        // gives R ≈ 4 — finite gradients without R → 0 risk).
        let mut template = model.default_params.clone();
        template.sigma = SigmaVector {
            values: vec![2.0],
            names: vec!["ADD_ERR".into()],
        };
        template.sigma_fixed = vec![false];
        model.default_params = template.clone();
        let template = &model.default_params;

        let population = make_population();
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n = x.len();

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.06]);
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![2.2, 1.6, 1.0]);
        let mut options = FitOptions::default();
        options.interaction = true;

        // Direct analytical call — see sibling test for why we bypass the dispatcher.
        let (nll, grad) = subject_nll_pop_grad_analytical_laplace(
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
        .expect("analytical Laplace path must succeed on this fixture");

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
            "Laplace (additive) nll mismatch: {nll} vs {nll_ref}"
        );

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
                "Laplace additive grad[{j}]: analytical={:.6e}, FD-ref={:.6e}, diff={:.3e}, tol={:.3e}",
                grad[j],
                ref_j,
                grad[j] - ref_j,
                tol,
            );
        }
    }

    // ── IOV analytical gradient tests ─────────────────────────────────────

    fn make_iov_model_gn() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.01, 1.0],
            theta_upper: vec![100.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
        };
        CompiledModel {
            name: "iov_gn_test".into(),
            pk_model: PkModel::OneCptIv,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                // eta[0] = bsv_eta, eta[1] = kappa (combined vector from inner optimizer)
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
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
            has_conditional_eta_params: false,
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
            derived_exprs: vec![],
            output_columns: vec![],
            #[cfg(feature = "survival")]
            endpoints: std::collections::HashMap::new(),
        }
    }

    fn make_iov_population_gn() -> Population {
        let subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![40.0, 32.0, 25.0, 38.0, 22.0, 14.0],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            dose_occasions: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        Population {
            subjects: vec![subject],
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// `subject_nll_pop_grad` on an IOV model returns a finite NLL and a
    /// gradient consistent with central FD of the per-subject marginal NLL.
    ///
    /// IOV takes the FD path (the closed-form κ-prior gradient was removed with
    /// the MAP-penalty objective in issue #101), so this is an FD-vs-FD
    /// self-consistency check on the augmented-marginal `foce_subject_nll_iov`.
    #[test]
    fn test_iov_grad_path_matches_fd_in_subject_nll_pop_grad() {
        let model = make_iov_model_gn();
        let population = make_iov_population_gn();
        let template = &model.default_params;
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let options = FitOptions {
            interaction: false,
            ..FitOptions::default()
        };

        use crate::estimation::inner_optimizer::run_inner_loop_warm;
        let (eta_hats, h_mats, _, kappas_all) =
            run_inner_loop_warm(&model, &population, template, 200, 1e-5, None, None, 0);

        let (nll_an, grad_an) = subject_nll_pop_grad(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hats[0],
            &h_mats[0],
            &kappas_all[0],
            &bounds,
            &options,
        );
        assert!(nll_an.is_finite(), "IOV grad path must return finite NLL");
        assert!(
            grad_an.iter().all(|g| g.is_finite()),
            "IOV grad path must return all-finite gradient"
        );

        // Confirm gradient is close to central FD of the same NLL.
        use crate::stats::likelihood::foce_subject_nll_iov;
        let subject = &population.subjects[0];
        let nll_at = |xv: &[f64]| -> f64 {
            let p = unpack_params(xv, template);
            if let Some(ref iov) = p.omega_iov {
                foce_subject_nll_iov(
                    &model,
                    subject,
                    &p.theta,
                    &eta_hats[0],
                    &h_mats[0],
                    &p.omega,
                    &p.sigma.values,
                    options.interaction,
                    &kappas_all[0],
                    iov,
                )
            } else {
                f64::NAN
            }
        };
        let eps = 1e-4;
        let n = x.len();
        for j in 0..n {
            let h_step = eps * (1.0 + x[j].abs());
            let mut xp = x.clone();
            xp[j] = (x[j] + h_step).min(bounds.upper[j]);
            let mut xm = x.clone();
            xm[j] = (x[j] - h_step).max(bounds.lower[j]);
            let actual_2h = xp[j] - xm[j];
            let fd_j = (nll_at(&xp) - nll_at(&xm)) / actual_2h;
            let tol = 1e-4 * (1.0 + fd_j.abs());
            assert!(
                (grad_an[j] - fd_j).abs() < tol,
                "IOV dispatch grad[{j}]: analytical={:.6e}, FD={:.6e}",
                grad_an[j],
                fd_j,
            );
        }
    }

    /// Microbenchmark: mu-ref shortcut vs forward-FD, both SB and Laplace paths.
    ///
    /// Run with:
    ///   cargo test --lib --no-default-features --features ci --release \
    ///     bench_mu_ref_gradient_throughput -- --nocapture --ignored
    ///
    /// Prints ns/call and speedup. Not a correctness test — always passes.
    #[test]
    #[ignore = "benchmark: run explicitly with --nocapture --ignored"]
    fn bench_mu_ref_gradient_throughput() {
        use crate::types::MuRef;
        use std::time::Instant;

        const N_SUBJ: usize = 32;
        const N_ITER: u32 = 5_000;
        const N_OBS: usize = 11;
        const N_ETA: usize = 3;

        // Build a 3-theta fully mu-referenced warfarin-like model.
        fn make_bench_model(with_mu_refs: bool) -> CompiledModel {
            let omega = OmegaMatrix::from_diagonal(
                &[0.09, 0.04, 0.30],
                vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
            );
            let default_params = ModelParameters {
                theta: vec![0.2, 10.0, 1.5],
                theta_names: vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
                theta_lower: vec![0.001, 0.1, 0.01],
                theta_upper: vec![10.0, 500.0, 50.0],
                theta_fixed: vec![false; 3],
                omega,
                omega_fixed: vec![false; 3],
                sigma: SigmaVector {
                    values: vec![0.02],
                    names: vec!["PROP_ERR".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            };
            let mu_refs = if with_mu_refs {
                let mut m = HashMap::new();
                m.insert(
                    "ETA_CL".into(),
                    MuRef {
                        theta_name: "TVCL".into(),
                        log_transformed: true,
                    },
                );
                m.insert(
                    "ETA_V".into(),
                    MuRef {
                        theta_name: "TVV".into(),
                        log_transformed: true,
                    },
                );
                m.insert(
                    "ETA_KA".into(),
                    MuRef {
                        theta_name: "TVKA".into(),
                        log_transformed: true,
                    },
                );
                m
            } else {
                HashMap::new()
            };
            CompiledModel {
                name: "bench".into(),
                pk_model: PkModel::OneCptOral,
                error_model: ErrorModel::Proportional,
                error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
                pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                    let mut p = PkParams::default();
                    p.values[0] = theta[0] * eta[0].exp();
                    p.values[1] = theta[1] * eta[1].exp();
                    p.values[4] = theta[2] * eta[2].exp();
                    p
                }),
                n_theta: 3,
                n_eta: N_ETA,
                n_epsilon: 1,
                n_kappa: 0,
                kappa_names: Vec::new(),
                theta_names: vec!["TVCL".into(), "TVV".into(), "TVKA".into()],
                eta_names: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
                indiv_param_names: vec!["CL".into(), "V".into(), "KA".into()],
                indiv_param_partials: crate::types::IndivParamPartials::empty(),
                default_params,
                omega_init_as_sd: vec![false; 3],
                sigma_init_as_sd: vec![false],
                kappa_init_as_sd: Vec::new(),
                mu_refs,
                kappa_mu_refs: HashMap::new(),
                tv_fn: None,
                pk_indices: vec![0, 1, 4],
                eta_map: vec![0, 1, 2],
                pk_idx_f64: vec![0.0, 1.0, 4.0],
                sel_flat: vec![1.0, 0.0, 0.0],
                ode_spec: None,
                diffusion_theta_start: None,
                diffusion_state_indices: Vec::new(),
                bloq_method: BloqMethod::Drop,
                referenced_covariates: Vec::new(),
                gradient_method: GradientMethod::Fd,
                parse_warnings: Vec::new(),
                has_conditional_eta_params: false,
                eta_param_info: Vec::new(),
                theta_transform: Vec::new(),
                #[cfg(feature = "nn")]
                covariate_nns: Vec::new(),
                scaling: crate::types::ScalingSpec::None,
                log_transform: false,
                dv_pre_logged: false,
                derived_exprs: vec![],
                output_columns: vec![],
                #[cfg(feature = "survival")]
                endpoints: std::collections::HashMap::new(),
            }
        }

        let population = {
            let subjects = (0..N_SUBJ)
                .map(|_| Subject {
                    id: "S1".into(),
                    doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0],
                    obs_raw_times: Vec::new(),
                    observations: vec![25.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2],
                    obs_cmts: vec![1; N_OBS],
                    covariates: HashMap::new(),
                    dose_covariates: Vec::new(),
                    obs_covariates: Vec::new(),
                    pk_only_times: Vec::new(),
                    pk_only_covariates: Vec::new(),
                    reset_times: Vec::new(),
                    cens: vec![0; N_OBS],
                    occasions: vec![1; N_OBS],
                    dose_occasions: vec![1],
                    #[cfg(feature = "survival")]
                    obs_records: vec![],
                })
                .collect();
            Population {
                subjects,
                covariate_names: Vec::new(),
                dv_column: "DV".to_string(),
                input_columns: vec![],
                exclusions: None,
                warnings: vec![],
            }
        };

        let model_fd = make_bench_model(false);
        let model_mu = make_bench_model(true);
        let template = &model_fd.default_params;
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let eta_hat = DVector::zeros(N_ETA);
        let kappas: Vec<DVector<f64>> = vec![];
        let h_vals: Vec<f64> = (0..N_OBS * N_ETA).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let h_matrix = nalgebra::DMatrix::from_vec(N_OBS, N_ETA, h_vals);

        let run = |model: &CompiledModel, interaction: bool, mu_on: bool| -> f64 {
            let mut opts = FitOptions::default();
            opts.interaction = interaction;
            opts.mu_referencing = mu_on;
            let t0 = Instant::now();
            for _ in 0..N_ITER {
                for si in 0..N_SUBJ {
                    let _ = subject_nll_pop_grad(
                        &x,
                        template,
                        model,
                        &population,
                        si,
                        &eta_hat,
                        &h_matrix,
                        &kappas,
                        &bounds,
                        &opts,
                    );
                }
            }
            t0.elapsed().as_nanos() as f64 / (N_ITER as f64 * N_SUBJ as f64)
        };

        // Cost of a single prediction solve (reference)
        let ns_pred = {
            let params = unpack_params(&x, template);
            let eta = [0.0f64; 3];
            let t0 = Instant::now();
            for _ in 0..100_000 {
                let _ = crate::pk::compute_predictions_with_tv(
                    &model_fd,
                    &population.subjects[0],
                    &params.theta,
                    &eta,
                );
            }
            t0.elapsed().as_nanos() as f64 / 100_000.0
        };

        let ns_sb_fd = run(&model_fd, false, false);
        let ns_sb_mu = run(&model_mu, false, true);
        let ns_lp_fd = run(&model_fd, true, false);
        let ns_lp_mu = run(&model_mu, true, true);

        println!("\nMu-ref gradient shortcut — {N_SUBJ} subjects, {N_OBS} obs, 3 mu-ref thetas, {N_ITER} iters");
        println!("  1 prediction solve = {ns_pred:.0} ns");
        println!("  FD solves saved per call = 3 (all thetas mu-referenced)");
        println!();
        println!("  Path             FD (ns/call)   mu-ref (ns/call)   speedup");
        println!("  {}", "-".repeat(60));
        println!(
            "  FOCE  (SB)       {ns_sb_fd:>9.0}        {ns_sb_mu:>9.0}       {:.2}x",
            ns_sb_fd / ns_sb_mu
        );
        println!(
            "  FOCEI (Laplace)  {ns_lp_fd:>9.0}        {ns_lp_mu:>9.0}       {:.2}x",
            ns_lp_fd / ns_lp_mu
        );
        println!();
        println!(
            "  Expected saving/call from skipping 3 FD solves: ~{:.0} ns  ({:.0}% of FD cost)",
            3.0 * ns_pred,
            100.0 * 3.0 * ns_pred / ns_sb_fd.max(1.0)
        );
    }

    /// Verify that the mu-ref gradient shortcut (H-column read) gives the same
    /// theta gradient as forward FD for both the SB and Laplace paths.
    ///
    /// The test model has CL = TVCL * exp(ETA_CL) — a log mu-referenced pair.
    /// H[:,0] is computed numerically from the eta Jacobian; mathematically it
    /// equals ∂f/∂(log TVCL) exactly, so the shortcut and FD should agree to
    /// within FD error (~1e-5 relative).
    #[test]
    fn test_mu_ref_gradient_shortcut() {
        use crate::types::MuRef;

        let mut model = make_model();
        // Register the mu-referencing: ETA_CL is paired with TVCL (log-transformed).
        model.mu_refs.insert(
            "ETA_CL".into(),
            MuRef {
                theta_name: "TVCL".into(),
                log_transformed: true,
            },
        );

        let population = make_population();
        let template = &model.default_params;
        let x = pack_params(template);
        let bounds = compute_bounds(template);

        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.05]);

        // Compute H numerically: ∂f/∂η_0 via forward FD on eta.
        let params = crate::estimation::parameterization::unpack_params(&x, template);
        let ipreds_base = crate::pk::compute_predictions_with_tv(
            &model,
            &population.subjects[0],
            &params.theta,
            eta_hat.as_slice(),
        );
        let h_step = 1e-6;
        let eta_pert = DVector::from_vec(vec![eta_hat[0] + h_step]);
        let ipreds_pert_eta = crate::pk::compute_predictions_with_tv(
            &model,
            &population.subjects[0],
            &params.theta,
            eta_pert.as_slice(),
        );
        let h_col: Vec<f64> = ipreds_pert_eta
            .iter()
            .zip(ipreds_base.iter())
            .map(|(&p, &b)| (p - b) / h_step)
            .collect();
        let h_matrix = nalgebra::DMatrix::from_column_slice(n_obs, n_eta, &h_col);

        // SB path (interaction = false)
        let mut opts_sb = FitOptions::default();
        opts_sb.interaction = false;

        opts_sb.mu_referencing = true;
        let (_, grad_muref_sb) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts_sb,
        )
        .expect("mu-ref SB path should succeed");

        opts_sb.mu_referencing = false;
        let (_, grad_fd_sb) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts_sb,
        )
        .expect("FD SB path should succeed");

        let n = x.len();
        for j in 0..n {
            let tol = 1e-4 * (1.0 + grad_fd_sb[j].abs());
            assert!(
                (grad_muref_sb[j] - grad_fd_sb[j]).abs() < tol,
                "SB grad[{j}]: mu-ref={:.6e}, FD={:.6e}, diff={:.2e}",
                grad_muref_sb[j],
                grad_fd_sb[j],
                (grad_muref_sb[j] - grad_fd_sb[j]).abs()
            );
        }

        // Laplace path (interaction = true)
        let mut opts_lap = FitOptions::default();
        opts_lap.interaction = true;

        opts_lap.mu_referencing = true;
        let (_, grad_muref_lap) = subject_nll_pop_grad_analytical_laplace(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts_lap,
        )
        .expect("mu-ref Laplace path should succeed");

        opts_lap.mu_referencing = false;
        let (_, grad_fd_lap) = subject_nll_pop_grad_analytical_laplace(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts_lap,
        )
        .expect("FD Laplace path should succeed");

        for j in 0..n {
            let tol = 1e-4 * (1.0 + grad_fd_lap[j].abs());
            assert!(
                (grad_muref_lap[j] - grad_fd_lap[j]).abs() < tol,
                "Laplace grad[{j}]: mu-ref={:.6e}, FD={:.6e}, diff={:.2e}",
                grad_muref_lap[j],
                grad_fd_lap[j],
                (grad_muref_lap[j] - grad_fd_lap[j]).abs()
            );
        }
    }

    /// Additive mu-refs (log_transformed: false) must NOT use the H-column
    /// shortcut. Ferx packs all thetas as log(THETA), so for an additive pair
    /// PARAM = THETA + ETA the identity ∂f/∂x_k = H[:,j] does not hold:
    ///   ∂f/∂x_k = ∂f/∂(log THETA) = THETA · ∂f/∂PARAM
    ///   ∂f/∂η   =                         1 · ∂f/∂PARAM   (≠ above unless THETA=1)
    ///
    /// This test registers an additive mu-ref and verifies that the gradient
    /// with mu_referencing=true still equals the FD gradient (i.e. the shortcut
    /// was skipped and FD was used as the fallback).
    #[test]
    fn test_mu_ref_additive_skipped() {
        use crate::types::MuRef;

        let mut model = make_model();
        // Register an additive (non-log) mu-ref for TVCL.
        model.mu_refs.insert(
            "ETA_CL".into(),
            MuRef {
                theta_name: "TVCL".into(),
                log_transformed: false,
            },
        );

        let population = make_population();
        let template = &model.default_params;
        let x = pack_params(template);
        let bounds = compute_bounds(template);
        let n_obs = 3;
        let n_eta = 1;
        let eta_hat = DVector::from_vec(vec![0.05]);
        let h_matrix = nalgebra::DMatrix::from_vec(n_obs, n_eta, vec![2.0, 1.5, 0.8]);

        let mut opts = FitOptions::default();
        opts.interaction = false;

        // Both should use FD (shortcut skipped for additive), so results must match.
        opts.mu_referencing = true;
        let (_, grad_on) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts,
        )
        .expect("should succeed");

        opts.mu_referencing = false;
        let (_, grad_off) = subject_nll_pop_grad_analytical(
            &x,
            template,
            &model,
            &population,
            0,
            &eta_hat,
            &h_matrix,
            &bounds,
            &opts,
        )
        .expect("should succeed");

        let n = x.len();
        for j in 0..n {
            assert!(
                (grad_on[j] - grad_off[j]).abs() < 1e-10,
                "additive mu-ref: grad[{j}] differed (shortcut was wrongly applied): \
                 on={:.6e}, off={:.6e}",
                grad_on[j],
                grad_off[j]
            );
        }
    }
}
