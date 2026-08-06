//! The VI driver: `run_vi`.
//!
//! Structure of one iteration:
//!
//! 1. Evaluate `−ELBO` and its gradients ([`population_neg_elbo`]).
//! 2. Adam-step the packed population vector `x` and every subject's `φᵢ`.
//! 3. Replace `Ω` with its closed-form maximizer (unless `vi_omega_update = adam`).
//! 4. Once inside the averaging window, fold `x` and `{φᵢ}` into a Polyak mean.
//!
//! There is no inner loop and no EBE solve: `φ` persists across iterations the way
//! an EBE warm-start does, but is optimized rather than re-solved.
//!
//! # Why the full budget always runs
//!
//! The objective is a Monte-Carlo estimate, so a per-iteration improvement test
//! would stop on noise. VI therefore runs all `vi_iters` iterations and reports
//! whether the objective had *in fact* settled, judged on a moving average.
//! Users control cost through `vi_iters`, not through a tolerance.

use nalgebra::DVector;

use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, pack_params, unpack_params};
use crate::types::{
    CompiledModel, FitOptions, ModelParameters, Population, ViFamily, ViFinalOfv, ViOmegaUpdate,
    ViResult,
};

use super::adam::{averaging_start, AdamConfig, AdamState, PolyakAverager};
use super::elbo::{
    closed_form_omega, population_neg_elbo, unsupported_data_term_reason, ElboConfig, PackedLayout,
};
use super::family::{FullRank, MeanField, VariationalFamily};

/// Default seed when `vi_seed` is unset, so a fit is reproducible across
/// invocations without the user having to pin one.
pub const DEFAULT_VI_SEED: u64 = 20_240_704;

/// Fraction of the run used as each half of the convergence moving average.
const CONVERGENCE_WINDOW_FRACTION: f64 = 0.1;

/// Relative tolerance the moving-average change must fall under for the run to be
/// reported as settled.
const CONVERGENCE_REL_TOL: f64 = 1e-4;

/// Whether the tail of the objective trace has stopped moving.
///
/// Compares the mean of the last `window` values against the mean of the `window`
/// before it. Averaging both sides is the point: single-iteration deltas are
/// dominated by Monte-Carlo noise and would report convergence at random.
pub fn trace_has_settled(trace: &[f64], window: usize, rel_tol: f64) -> bool {
    if window == 0 || trace.len() < 2 * window {
        return false;
    }
    let n = trace.len();
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let recent = mean(&trace[n - window..]);
    let prior = mean(&trace[n - 2 * window..n - window]);
    if !recent.is_finite() || !prior.is_finite() {
        return false;
    }
    (prior - recent).abs() <= rel_tol * (1.0 + recent.abs())
}

fn build_family(kind: ViFamily, n_eta: usize) -> Box<dyn VariationalFamily> {
    match kind {
        ViFamily::FullRank => Box::new(FullRank::new(n_eta)),
        ViFamily::MeanField => Box::new(MeanField::new(n_eta)),
    }
}

/// Zero the gradient at coordinates the model declares FIXed.
///
/// Adam's step for a zero gradient is exactly zero (`m` and `v` both stay at 0,
/// giving `0/(0+ε)`), so zeroing here is sufficient to pin a coordinate — no
/// projection or re-clamping is needed afterwards.
fn zero_fixed_coords(grad: &mut [f64], template: &ModelParameters, layout: &PackedLayout) {
    for (i, &fixed) in template.theta_fixed.iter().enumerate() {
        if fixed {
            grad[i] = 0.0;
        }
    }
    for (k, &fixed) in template.sigma_fixed.iter().enumerate() {
        if fixed {
            grad[layout.sigma_start() + k] = 0.0;
        }
    }
}

/// Write `omega`'s Cholesky factor into the `Ω` block of the packed vector,
/// in `pack_params` order and transform (diagonal as `log`).
///
/// Deliberately *not* implemented as `unpack_params` → replace `Ω` →
/// `pack_params`: that round-trips `θ` and `σ` through `exp(ln(·))` on **every**
/// iteration, and the 1-ULP error each time accumulates over a thousand
/// iterations — visibly so on a FIXed parameter, which is supposed to hold its
/// declared value exactly. Touching only the `Ω` slots leaves the rest bit-exact.
fn write_omega_block(
    x: &mut [f64],
    omega: &crate::types::OmegaMatrix,
    template: &ModelParameters,
    layout: &PackedLayout,
) {
    let l = &omega.chol;
    for (slot, (i, j)) in
        crate::estimation::parameterization::lower_tri_iter(omega.dim(), template.omega.diagonal)
            .enumerate()
    {
        x[layout.omega_start() + slot] = if i == j {
            l[(i, j)].max(1e-10).ln()
        } else {
            l[(i, j)]
        };
    }
}

/// Restore parameters the model declares FIXed to their exact declared values.
///
/// The optimizer never moves them (their gradients are zeroed), but the single
/// `pack`/`unpack` round trip between the initial estimates and the reported ones
/// is not bit-exact — `exp(ln(x)) != x` in general. A FIXed parameter that comes
/// back one ULP off its declared value is a small lie, and one that shows up in
/// output diffs, so undo it.
fn restore_fixed(params: &mut ModelParameters, template: &ModelParameters) {
    for (i, &fixed) in template.theta_fixed.iter().enumerate() {
        if fixed {
            params.theta[i] = template.theta[i];
        }
    }
    for (k, &fixed) in template.sigma_fixed.iter().enumerate() {
        if fixed {
            params.sigma.values[k] = template.sigma.values[k];
        }
    }
}

/// Run variational inference to a fixed iteration budget.
pub fn run_vi(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    if let Some(reason) = unsupported_data_term_reason(model) {
        return Err(reason);
    }
    let n_subjects = population.subjects.len();
    if n_subjects == 0 {
        return Err("VI requires at least one subject".to_string());
    }

    let n_eta = model.n_eta;
    let family = build_family(options.vi_family, n_eta);
    let layout = PackedLayout::new(init_params);
    let n_iters = options.vi_iters.max(1);

    let cfg = ElboConfig {
        n_mc_samples: options.vi_mc_samples.max(1),
        eta_grad: options.vi_eta_grad,
        seed: options.vi_seed.unwrap_or(DEFAULT_VI_SEED),
    };
    let adam_cfg = AdamConfig {
        lr: options.vi_lr,
        ..Default::default()
    };

    // Start q at the prior and Θ at the model file's initial estimates: no
    // randomization, so the fit is reproducible by default.
    let mut x = pack_params(init_params);
    let mut phis: Vec<Vec<f64>> = (0..n_subjects)
        .map(|_| family.init(&init_params.omega))
        .collect();
    let mut adam_x = AdamState::new(x.len());
    let mut adam_phi: Vec<AdamState> = (0..n_subjects)
        .map(|_| AdamState::new(family.n_params()))
        .collect();

    let avg_start = averaging_start(n_iters, options.vi_avg_last);
    let mut avg_x = PolyakAverager::new(x.len());
    let mut avg_phi: Vec<PolyakAverager> = (0..n_subjects)
        .map(|_| PolyakAverager::new(family.n_params()))
        .collect();

    let mut warnings: Vec<String> = Vec::new();
    let mut trace: Vec<f64> = Vec::with_capacity(n_iters);
    let mut n_fd_subjects = 0usize;
    let verbose = options.verbose;

    for iter in 0..n_iters {
        let eval = population_neg_elbo(
            model,
            population,
            init_params,
            &x,
            family.as_ref(),
            &phis,
            &cfg,
            iter as u64,
        )?;
        // `neg_elbo` is already `−ELBO`, so `−2·ELBO` is `2·neg_elbo`. Reported on
        // the OFV scale: an upper bound on `−2 log p(y)` that *decreases* as the
        // fit improves, so a trace reads the way every other objective here does.
        trace.push(2.0 * eval.neg_elbo);
        n_fd_subjects = eval.n_fd_subjects;

        if !eval.neg_elbo.is_finite() {
            return Err(format!(
                "VI objective became non-finite at iteration {iter}; try a smaller vi_lr \
                 or check the model's initial estimates"
            ));
        }

        let mut grad_x = eval.grad_x.clone();
        zero_fixed_coords(&mut grad_x, init_params, &layout);
        if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
            // Ω is set analytically below; stepping it here as well would apply the
            // same information twice and fight the closed form.
            for g in grad_x[layout.omega_start()..layout.sigma_start()].iter_mut() {
                *g = 0.0;
            }
        }
        adam_x.step(&mut x, &grad_x, &adam_cfg);

        for (i, phi) in phis.iter_mut().enumerate() {
            adam_phi[i].step(phi, &eval.grad_phi[i], &adam_cfg);
        }

        if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
            let omega = closed_form_omega(family.as_ref(), &phis, init_params);
            write_omega_block(&mut x, &omega, init_params, &layout);
        }

        if iter >= avg_start {
            avg_x.accumulate(&x);
            for (i, phi) in phis.iter().enumerate() {
                avg_phi[i].accumulate(phi);
            }
        }

        if verbose && (iter % 100 == 0 || iter + 1 == n_iters) {
            eprintln!(
                "VI iter {:>5}  -2*ELBO = {:.4}",
                iter,
                trace[trace.len() - 1]
            );
        }
    }

    // The reported estimate is the Polyak mean, not the last iterate.
    let x_final = avg_x.mean().unwrap_or_else(|| x.clone());
    let phis_final: Vec<Vec<f64>> = avg_phi
        .iter()
        .zip(phis.iter())
        .map(|(a, last)| a.mean().unwrap_or_else(|| last.clone()))
        .collect();
    let mut final_params = unpack_params(&x_final, init_params);
    if options.vi_omega_update == ViOmegaUpdate::ClosedForm {
        final_params.omega = closed_form_omega(family.as_ref(), &phis_final, init_params);
    }
    restore_fixed(&mut final_params, init_params);

    // Re-evaluate at the averaged point with more draws: this number is reported,
    // so it should carry less Monte-Carlo noise than a training iteration.
    let report_cfg = ElboConfig {
        n_mc_samples: (cfg.n_mc_samples * 32).max(32),
        ..cfg.clone()
    };
    let final_eval = population_neg_elbo(
        model,
        population,
        init_params,
        &pack_params(&final_params),
        family.as_ref(),
        &phis_final,
        &report_cfg,
        n_iters as u64,
    )?;

    // The variational moments — VI's own output, reported on `ViResult`.
    let mut eta_means: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
    let mut eta_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_subjects);
    let mut warm: Vec<DVector<f64>> = Vec::with_capacity(n_subjects);
    for phi in &phis_final {
        let (mu, s) = family.moments(phi);
        eta_means.push(mu.iter().copied().collect());
        eta_covs.push(
            (0..n_eta)
                .map(|i| (0..n_eta).map(|j| s[(i, j)]).collect())
                .collect(),
        );
        warm.push(mu);
    }

    // Downstream diagnostics — CWRES, IWRES, shrinkage, sdtab — are all defined in
    // terms of the conditional **mode** and the `n_obs × n_eta` sensitivity matrix
    // `∂f/∂η` (`OuterResult::h_matrices`, NONMEM's "H"). Reporting variational
    // means as `eta_hat` while sourcing `H` elsewhere would make those diagnostics
    // mutually inconsistent, so one inner-loop pass at the converged estimate
    // produces both. It is cheap: warm-starting from `μ` lands the EBE search
    // essentially on top of its answer.
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _, kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm),
        Some(&final_mu_k),
        0,
        0,
    );

    let converged = trace_has_settled(
        &trace,
        ((n_iters as f64) * CONVERGENCE_WINDOW_FRACTION).round() as usize,
        CONVERGENCE_REL_TOL,
    );
    if !converged {
        warnings.push(format!(
            "VI: the objective was still moving at the end of {n_iters} iterations \
             (see vi.elbo_trace). Increase vi_iters, or lower vi_lr if the trace is oscillating."
        ));
    }
    if n_fd_subjects > 0 {
        warnings.push(format!(
            "VI: {n_fd_subjects} of {n_subjects} subjects used finite-difference \
             eta-gradients because the analytic provider declined them. The fit is correct \
             but much slower than it needs to be."
        ));
    }

    // The ELBO is a lower bound, so it is never reported as the OFV. See
    // `ViFinalOfv` for why, and for how to obtain a real marginal likelihood.
    let ofv = match options.vi_final_ofv {
        ViFinalOfv::None => {
            warnings.push(
                "VI: `ofv` is NaN because the ELBO is a lower bound on the log likelihood, not \
                 a −2 log L, and is not comparable with a FOCE/SAEM OFV. Set \
                 `vi_final_ofv = laplace`, or chain `methods = vi, imp` with \
                 `imp_eval_only = true`, to evaluate a genuine marginal likelihood at the VI \
                 estimate. The bound itself is on `vi.neg_two_elbo`."
                    .to_string(),
            );
            f64::NAN
        }
        // Reuses the EBEs and sensitivities already converged above, so requesting
        // an OFV costs only the objective evaluation itself.
        ViFinalOfv::Laplace => {
            2.0 * pop_nll(
                model,
                population,
                &final_params,
                &eta_hats,
                &h_matrices,
                &kappas,
                options.interaction,
            )
        }
    };

    let vi_result = ViResult {
        neg_two_elbo: 2.0 * final_eval.neg_elbo,
        data_term: final_eval.data_term,
        kl_term: final_eval.kl_term,
        n_iterations: n_iters,
        converged,
        family: family.label().to_string(),
        n_mc_samples: cfg.n_mc_samples,
        elbo_trace: trace,
        eta_means,
        eta_covs,
        n_fd_subjects,
    };

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged,
        n_iterations: n_iters,
        eta_hats,
        h_matrices,
        kappas,
        covariance_matrix: None,
        covariance_wall_time_secs: 0.0,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient: None,
        sir_fallback_proposal: None,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        vi: Some(vi_result),
        mixture_posteriors: None,
    })
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
