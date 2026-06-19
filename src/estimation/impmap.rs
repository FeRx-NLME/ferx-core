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

use crate::estimation::importance_sampling::{
    compute_posterior_hessian, find_optimal_iscale, subject_is_draws, SubjectDraws,
};
use crate::estimation::inner_optimizer::{find_ebe, EbeResult, InnerLoopStats};
use crate::estimation::outer_optimizer::{
    compute_covariance, pop_nll, CovarianceStepResult, OuterResult,
};
use crate::estimation::parameterization::{compute_mu_k, pack_params, theta_packs_log};
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Floor the free Ω diagonal to keep the proposal/prior positive-definite.
/// Mirrors SAEM's `floor_omega_diagonal`: FIX-ed diagonals are left untouched.
fn floor_omega_diagonal(omega_mat: &mut DMatrix<f64>, omega_fixed: &[bool], floor: f64) {
    for i in 0..omega_mat.nrows() {
        if omega_fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        if omega_mat[(i, i)] < floor {
            omega_mat[(i, i)] = floor;
        }
    }
}

/// Positive-definite floor for free Ω diagonals (matches the SAEM constant).
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Absolute lower bound on the IMP proposal-covariance diagonal — a numerical
/// guard against a literally-zero (degenerate-ESS) variance, NOT a statistical
/// floor. It must stay well below any real conditional variance: with rich data
/// the conditional posterior of η is legitimately tiny (orders of magnitude below
/// the prior Ω), and flooring it against Ω would make the proposal far too broad
/// and collapse the ESS (the very rich-data failure mode that motivates IMPMAP).
const IMP_PROPOSAL_COV_FLOOR: f64 = 1e-10;

/// How each MCEM iteration positions the per-subject importance-sampling
/// proposal — the one piece that distinguishes IMP from IMPMAP. Everything else
/// (M-step, sufficient statistics, averaging, ESS diagnostics, final objective)
/// is shared by [`run_mcem`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProposalRecenter {
    /// IMPMAP (NONMEM `METHOD=IMPMAP`): re-run the MAP inner loop **every**
    /// iteration; proposal centered at the conditional mode with
    /// first-order-variance scale `(JᵀR⁻¹J + Ω⁻¹)⁻¹`.
    Map,
    /// IMP (NONMEM `METHOD=IMP`): run the MAP inner loop on the **first**
    /// iteration only (to seed the proposal); thereafter center at the previous
    /// iteration's weighted posterior mean with scale = previous weighted
    /// posterior covariance `Ŝ − m̂m̂ᵀ`.
    SampleMoments,
}

/// Convert a weighted posterior covariance `Cov` into the proposal precision
/// `Σ⁻¹ = Cov⁻¹` that [`subject_is_draws`]/`build_proposal` expects (it forms the
/// proposal scale as `(Σ⁻¹ + λI)⁻¹`). Used only by the IMP (`SampleMoments`)
/// recenter path.
///
/// The raw weighted sample covariance is unbounded above and makes the adaptive
/// proposal unstable: a heavy-tailed outlier inflates it without limit, and then
/// the prior term `−½ηᵀΩ⁻¹η` of the resulting far samples explodes the −2 log L
/// and the next Ω M-step. We therefore **cap** the proposal-covariance diagonal
/// at the prior `Ωᵢᵢ` — the conditional variance of a well-identified η is
/// bounded above by its prior variance. The diagonal is floored only at a tiny
/// absolute value to avoid a singular matrix (NOT at a fraction of Ω — see
/// [`IMP_PROPOSAL_COV_FLOOR`]). If the result is still not Cholesky-invertible a
/// zero matrix is returned, which makes `build_proposal` take its Ω fallback — a
/// broad but valid proposal.
fn covariance_to_proposal_hessian(
    cov: &DMatrix<f64>,
    omega: &DMatrix<f64>,
    floor: f64,
) -> DMatrix<f64> {
    let n = cov.nrows();
    let mut c = cov.clone();
    for i in 0..n {
        let hi = omega[(i, i)].max(floor);
        let v = c[(i, i)];
        if !v.is_finite() || v > hi {
            c[(i, i)] = hi;
        } else if v < floor {
            c[(i, i)] = floor;
        }
    }
    match c.cholesky() {
        Some(ch) => ch.inverse(),
        None => DMatrix::zeros(n, n),
    }
}

/// Log-transformed mu-referencing pairs `(theta_idx, eta_idx)`. For these the
/// typical value satisfies `log(P_i) = log(θ) + η_i`, so the EM M-step shifts
/// `log(θ) += mean(η)` in closed form — without it θ and the η mean are
/// confounded and the variance Ω absorbs the misfit instead. Mirrors SAEM's
/// `get_mu_ref_pairs`.
fn mu_ref_log_pairs(model: &CompiledModel) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !mu_ref.log_transformed {
                continue;
            }
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                pairs.push((theta_idx, eta_idx));
            }
        }
    }
    pairs
}

/// Multi-start MAP: for each subject, run `find_ebe` with the warm-start (or
/// cold-start) and then `mceta` additional random starting points drawn from
/// N(0, Ω). The start with the lowest NLL wins. When `mceta == 0` this
/// degrades to a single warm-start — identical to the previous behaviour.
///
/// Returns `(eta_hats, h_matrices, stats)`. Kappas are always empty because
/// IMPMAP refuses IOV models.
#[allow(clippy::too_many_arguments)]
fn run_map_multistart(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    inner_maxiter: usize,
    inner_tol: f64,
    prev_etas: Option<&[DVector<f64>]>,
    mu_k: &[f64],
    mceta: usize,
    seed: u64,
    iteration: usize,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, InnerLoopStats) {
    let n_eta = model.n_eta;

    // Cholesky of Ω for drawing random starts (computed once, outside the
    // per-subject parallel loop).
    let omega_chol = if mceta > 0 {
        params.omega.matrix.clone().cholesky().map(|c| c.l())
    } else {
        None
    };

    let results: Vec<EbeResult> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let warm = prev_etas.map(|pe| pe[i].as_slice());
            let mu = Some(mu_k);

            // Baseline: warm-start (or cold-start from η = 0).
            let mut best = find_ebe(model, subject, params, inner_maxiter, inner_tol, warm, mu);

            if let Some(ref l_omega) = omega_chol {
                // Deterministic per-subject, per-iteration seed, separated from IS draws.
                let subj_seed = seed
                    .wrapping_add(i as u64)
                    .wrapping_add((iteration as u64) << 32)
                    .wrapping_add(0x4D43_4554_4100u64);
                let mut rng = StdRng::seed_from_u64(subj_seed);

                for _start in 0..mceta {
                    // Draw z ~ N(0, I), compute eta_start = L_Ω · z.
                    let z: Vec<f64> = (0..n_eta)
                        .map(|_| StandardNormal.sample(&mut rng))
                        .collect();
                    let z_dv = DVector::from_vec(z);
                    let eta_start = l_omega * &z_dv;
                    let eta_slice: Vec<f64> = eta_start.iter().copied().collect();

                    let candidate = find_ebe(
                        model,
                        subject,
                        params,
                        inner_maxiter,
                        inner_tol,
                        Some(&eta_slice),
                        mu,
                    );
                    if candidate.nll < best.nll {
                        best = candidate;
                    }
                }
            }

            best
        })
        .collect();

    let stats = InnerLoopStats {
        n_unconverged: results.iter().filter(|r| !r.converged).count(),
        n_fallback: results.iter().filter(|r| r.used_fallback).count(),
    };
    let eta_hats: Vec<DVector<f64>> = results.iter().map(|r| r.eta.clone()).collect();
    let h_matrices: Vec<DMatrix<f64>> = results.iter().map(|r| r.h_matrix.clone()).collect();

    (eta_hats, h_matrices, stats)
}

/// Run IMPMAP. `warm_etas`, when supplied by a preceding chain stage, seed the
/// first MAP inner loop; otherwise the inner loop cold-starts from η = 0.
/// Run IMPMAP (NONMEM `METHOD=IMPMAP`). Thin wrapper over the shared MCEM core
/// with mode re-centering on every iteration; resolves the `impmap_*` options.
pub fn run_impmap(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let nu = options.impmap_proposal_df;
    run_mcem(
        model,
        population,
        init_params,
        warm_etas,
        options,
        ProposalRecenter::Map,
        "IMPMAP",
        "impmap_proposal_df",
        options.impmap_iterations,
        options.impmap_samples,
        nu,
        options.impmap_averaging,
        options.impmap_seed.unwrap_or(12345),
        options.impmap_low_ess_threshold,
        options.impmap_mceta,
        options.impmap_sobol && nu.is_infinite(),
        options.impmap_trace,
    )
}

/// Run IMP as an estimator (NONMEM `METHOD=IMP`). Thin wrapper over the shared
/// MCEM core with sample-moment re-centering (conditional mode found only on the
/// first iteration); resolves the `is_*` options. The evaluation-only
/// `is_eval_only` path lives in `importance_sampling.rs`.
pub fn run_imp(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    run_mcem(
        model,
        population,
        init_params,
        warm_etas,
        options,
        ProposalRecenter::SampleMoments,
        "IMP",
        "is_proposal_df",
        options.is_iterations,
        options.is_samples,
        options.is_proposal_df,
        options.is_averaging,
        options.is_seed.unwrap_or(12345),
        options.is_low_ess_threshold,
        0,     // mceta: no multi-start MAP for IMP
        false, // use_sobol: IMP has no Sobol option
        false, // collect_trace: IMP has no trace option
    )
}

/// Shared Monte-Carlo EM core for IMP and IMPMAP. The `recenter` strategy is the
/// only behavioural difference; `label`/`df_key` tag warnings and verbose output.
#[allow(clippy::too_many_arguments)]
fn run_mcem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    warm_etas: Option<&[DVector<f64>]>,
    options: &FitOptions,
    recenter: ProposalRecenter,
    label: &str,
    df_key: &str,
    n_iter_opt: usize,
    k_opt: usize,
    nu: f64,
    n_avg_opt: usize,
    seed: u64,
    threshold: f64,
    mceta: usize,
    use_sobol: bool,
    collect_trace: bool,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // ---- Validation ----
    if n_eta == 0 {
        return Err(format!(
            "{label} requires at least one random effect (n_eta = 0). \
             Use FOCE/FOCEI for fixed-effects-only models."
        ));
    }
    if model.is_sde() {
        return Err(format!(
            "{label} is not yet supported for SDE / [diffusion] models \
             (the EKF process-noise variance is not threaded through the IS \
             observation likelihood). Use FOCE / FOCEI instead."
        ));
    }
    if model.n_kappa > 0 {
        return Err(format!(
            "{label} does not yet support inter-occasion variability (κ / [iov]); \
             the IOV M-step is a planned follow-up. Use SAEM or FOCEI for IOV models."
        ));
    }
    if !init_params.omega.log_det.is_finite() {
        return Err(format!(
            "{label}: initial Ω log-determinant is not finite — check the \
             [parameters] Ω block."
        ));
    }

    let n_iter = n_iter_opt.max(1);
    let k_samples = k_opt.max(2);
    // `INFINITY` selects the multivariate-normal proposal; any finite value must
    // be a valid Student-t DoF (>= 1). Guard here so a programmatic caller that
    // bypasses the parser's range check can't reach the `ChiSquared::new(nu)`
    // panic in `subject_is_draws`. Mirrors `run_importance_sampling`.
    if nu.is_finite() && nu < 1.0 {
        return Err(format!(
            "{label}: {df_key} must be >= 1.0 (or +inf for a normal proposal), got {nu}"
        ));
    }
    let n_avg = n_avg_opt.min(n_iter);
    let verbose = options.verbose;
    let cancel = &options.cancel;

    if verbose {
        let prop = if nu.is_finite() {
            format!("t_{nu}")
        } else {
            "normal".to_string()
        };
        let recenter_desc = match recenter {
            ProposalRecenter::Map => "MAP recenter/iter",
            ProposalRecenter::SampleMoments => "sample-moment recenter",
        };
        let mceta_msg = if mceta > 0 {
            format!(", MCETA={}", mceta)
        } else {
            String::new()
        };
        eprintln!(
            "{}: {} subjects, {} ETAs, {} iters, K={}/subject, {} proposal, {}, seed={}{}",
            label, n_subjects, n_eta, n_iter, k_samples, prop, recenter_desc, seed, mceta_msg
        );
    }

    // ---- Packing scaffolding (mirrors SAEM) ----
    // Per-theta packing: log for `theta_lower >= 0`, identity otherwise (so
    // covariate exponents with negative lower bounds are not pinned to ~0).
    let theta_packs_log_mask: Vec<bool> = init_params
        .theta_lower
        .iter()
        .map(|&lo| theta_packs_log(lo))
        .collect();
    let pack_theta = |i: usize, t: f64| -> f64 {
        if theta_packs_log_mask[i] {
            t.max(1e-10).ln()
        } else {
            t
        }
    };

    let mut log_theta: Vec<f64> = (0..n_theta)
        .map(|i| pack_theta(i, init_params.theta[i]))
        .collect();
    let mut log_sigma: Vec<f64> = init_params
        .sigma
        .values
        .iter()
        .map(|&s| s.max(1e-10).ln())
        .collect();

    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i]
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).ln()
            } else {
                init_params.theta_upper[i]
            }
        })
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: lower == upper == packed value.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

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
    let mu_ref_pairs = mu_ref_log_pairs(model);
    let use_closed_form = !mu_ref_pairs.is_empty();
    if !use_closed_form {
        // No log-mu-ref parameter: every typical value goes through the weighted
        // M-step, which cannot resolve the θ/η-mean confounding on its own. Flag
        // it — estimates may be unreliable (see the docs caveat).
        warnings.push(format!(
            "{label}: no log-mu-referenced parameters found (e.g. `CL = TVCL*exp(ETA)`); \
             typical-value estimation relies on the weighted M-step alone and may converge \
             poorly. Prefer a log-mu-referenced parameterization, or use FOCEI."
        ));
    }

    // ---- Iteration state ----
    let mut theta_cur = init_params.theta.clone();
    let mut sigma_cur = init_params.sigma.values.clone();
    let mut omega_mat = init_params.omega.matrix.clone();
    let mut prev_etas: Option<Vec<DVector<f64>>> = warm_etas.map(|e| e.to_vec());
    // Previous iteration's per-subject weighted draws — the proposal source for
    // the IMP (`SampleMoments`) recenter path on iterations 2+. `None` on the
    // first iteration (and always for IMPMAP, which never reads it).
    let mut prev_draws: Option<Vec<SubjectDraws>> = None;

    // Running mean of parameters over the final `n_avg` iterations.
    let mut acc_theta = vec![0.0f64; n_theta];
    let mut acc_sigma = vec![0.0f64; n_sigma];
    let mut acc_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut n_acc = 0usize;

    let mut last_eta_hats: Vec<DVector<f64>> = Vec::new();

    // ---- Trace: collect per-iteration parameters (analogous to NONMEM .ext) ----
    let mut trace_rows: Vec<ImpmapTraceRow> = if collect_trace {
        Vec::with_capacity(n_iter + 2)
    } else {
        Vec::new()
    };

    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(cancel) {
            if verbose {
                eprintln!("{}: cancelled at iteration {}", label, k);
            }
            break;
        }

        // Assemble current params for the inner loop / E-step.
        let omega_k = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        let params_k = ModelParameters {
            theta: theta_cur.clone(),
            theta_names: init_params.theta_names.clone(),
            theta_lower: init_params.theta_lower.clone(),
            theta_upper: init_params.theta_upper.clone(),
            theta_fixed: init_params.theta_fixed.clone(),
            omega: omega_k,
            omega_fixed: init_params.omega_fixed.clone(),
            sigma: SigmaVector {
                values: sigma_cur.clone(),
                names: init_params.sigma.names.clone(),
            },
            sigma_fixed: init_params.sigma_fixed.clone(),
            omega_iov: None,
            kappa_fixed: init_params.kappa_fixed.clone(),
        };

        // ---- E-step A: position the proposal ----
        // IMPMAP (`Map`) re-runs the MAP inner loop every iteration. IMP
        // (`SampleMoments`) runs it only on the first iteration — when
        // `prev_draws` is still `None` — to seed the proposal, then recenters
        // from the previous iteration's weighted moments inside the draws loop.
        let run_inner = recenter == ProposalRecenter::Map || prev_draws.is_none();
        let (eta_hats, h_matrices) = if run_inner {
            let mu_k = compute_mu_k(model, &params_k.theta, options.mu_referencing);
            let (e, h, _stats) = run_map_multistart(
                model,
                population,
                &params_k,
                options.inner_maxiter,
                options.inner_tol,
                prev_etas.as_deref(),
                &mu_k,
                mceta,
                seed,
                k,
            );
            (e, h)
        } else {
            (Vec::new(), Vec::new())
        };

        let omega_inv = params_k.omega.inv.clone();
        let log_det_omega = params_k.omega.log_det;

        // ---- E-step B: importance sampling around each mode ----
        let iscale_min = options.iscale_min;
        let iscale_max = options.iscale_max;
        let draws: Vec<_> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                // Poll per subject: the inner-loop MAP + IS draws below are the
                // dominant per-iteration cost, so without this a cancel set
                // mid-iteration is not seen until the next iteration's top-of-loop
                // check (line ~257) — minutes on a large dataset. Mirrors
                // `run_importance_sampling`. The driver breaks right after the
                // collect, so the placeholder draws never reach the M-step.
                if crate::cancel::is_cancelled(cancel) {
                    return SubjectDraws::cancelled(n_eta);
                }
                let (center, h_post) = if run_inner {
                    // Proposal centred at the conditional mode with
                    // first-order-variance (Sheiner–Beal posterior) scale.
                    let h_post = compute_posterior_hessian(
                        model,
                        subject,
                        &params_k.theta,
                        &eta_hats[i],
                        &params_k.sigma.values,
                        &h_matrices[i],
                        &omega_inv,
                        n_eta,
                        scratch,
                    );
                    (eta_hats[i].clone(), h_post)
                } else {
                    // IMP, iterations 2+: centre at the previous iteration's
                    // weighted posterior mean m̂, scale at the previous weighted
                    // posterior covariance Ŝ − m̂m̂ᵀ (passed as its inverse).
                    let pd = &prev_draws.as_ref().expect("prev_draws set when !run_inner")[i];
                    let center = DVector::from_row_slice(&pd.mean);
                    let cov = &pd.second_moment - &center * center.transpose();
                    let h_post = covariance_to_proposal_hessian(
                        &cov,
                        &params_k.omega.matrix,
                        IMP_PROPOSAL_COV_FLOOR,
                    );
                    (center, h_post)
                };
                let subj_seed = seed.wrapping_add(i as u64).wrapping_add((k as u64) << 32);
                let iscale = find_optimal_iscale(
                    model,
                    subject,
                    &params_k.theta,
                    &params_k.sigma.values,
                    &center,
                    &h_post,
                    &omega_inv,
                    log_det_omega,
                    n_eta,
                    nu,
                    subj_seed,
                    scratch,
                    iscale_min,
                    iscale_max,
                );
                subject_is_draws(
                    model,
                    subject,
                    &params_k.theta,
                    &params_k.sigma.values,
                    &center,
                    &h_post,
                    &omega_inv,
                    log_det_omega,
                    n_eta,
                    k_samples,
                    nu,
                    subj_seed,
                    scratch,
                    iscale,
                    use_sobol,
                )
            })
            .collect();

        // If a cancel was observed inside the E-step, the `draws` are placeholders;
        // break before the M-steps consume them. The post-loop check returns Err.
        if crate::cancel::is_cancelled(cancel) {
            if verbose {
                eprintln!("{}: cancelled during E-step at iteration {}", label, k);
            }
            break;
        }

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

        // Record this iteration's parameters for the trace (opt-in).
        if collect_trace {
            trace_rows.push(ImpmapTraceRow {
                iteration: k as i64,
                theta: theta_cur.clone(),
                omega_lower_tri: lower_triangle(&omega_mat),
                sigma: sigma_cur.clone(),
                ofv: minus2ll,
            });
        }

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

        theta_cur = (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    log_theta[i].exp()
                } else {
                    log_theta[i]
                }
            })
            .collect();
        sigma_cur = log_sigma.iter().map(|&s| s.exp()).collect();

        // Warm-start next iteration's inner loop from this iteration's modes —
        // only when we actually ran the inner loop this iteration (IMP skips it
        // on iterations 2+, leaving the iter-1 modes in place for the final EBE
        // pass).
        if run_inner {
            prev_etas = Some(eta_hats.clone());
            last_eta_hats = eta_hats;
        }

        // IMP recenters the next iteration's proposal from these draws; IMPMAP
        // never reads them, so retain only for `SampleMoments` to avoid holding
        // K·n_subjects samples for the MAP path.
        if recenter == ProposalRecenter::SampleMoments {
            prev_draws = Some(draws);
        }

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

    let final_omega = OmegaMatrix::from_matrix(
        final_omega_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    let final_params = ModelParameters {
        theta: final_theta,
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: final_sigma,
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: None,
        kappa_fixed: init_params.kappa_fixed.clone(),
    };

    // ---- Final EBEs (warm-started) + FOCE Laplace OFV for comparability ----
    let warm = if last_eta_hats.is_empty() {
        None
    } else {
        Some(last_eta_hats.as_slice())
    };
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _stats) = run_map_multistart(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        warm,
        &final_mu_k,
        mceta,
        seed,
        n_iter + 1, // distinct iteration index for final EBEs
    );
    let final_kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new(); n_subjects];

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

    // ---- Finalize trace ----
    let impmap_trace = if collect_trace {
        // Append final (averaged) estimate row.
        trace_rows.push(ImpmapTraceRow {
            iteration: -1_000_000_000,
            theta: final_params.theta.clone(),
            omega_lower_tri: lower_triangle(&final_params.omega.matrix),
            sigma: final_params.sigma.values.clone(),
            ofv,
        });
        // Append SE row when the covariance step succeeded.
        if let Some(ref cov) = covariance_matrix {
            let se: Vec<f64> = (0..cov.nrows()).map(|i| cov[(i, i)].sqrt()).collect();
            // Unpack SEs into theta / omega-LT / sigma segments, mirroring
            // pack_params layout: [theta..., cholesky-omega..., sigma...].
            let n_free_theta = final_params.theta.len();
            let n_omega_lt = lower_triangle(&final_params.omega.matrix).len();
            let n_free_sigma = final_params.sigma.values.len();
            let se_theta: Vec<f64> = se.iter().take(n_free_theta).copied().collect();
            let se_omega: Vec<f64> = se
                .iter()
                .skip(n_free_theta)
                .take(n_omega_lt)
                .copied()
                .collect();
            let se_sigma: Vec<f64> = se
                .iter()
                .skip(n_free_theta + n_omega_lt)
                .take(n_free_sigma)
                .copied()
                .collect();
            trace_rows.push(ImpmapTraceRow {
                iteration: -1_000_000_001,
                theta: se_theta,
                omega_lower_tri: se_omega,
                sigma: se_sigma,
                ofv: 0.0,
            });
        }

        // Build column names following NONMEM convention.
        let theta_names: Vec<String> = (1..=n_theta).map(|i| format!("THETA{i}")).collect();
        let omega_names: Vec<String> = {
            let mut names = Vec::new();
            for i in 0..n_eta {
                for j in 0..=i {
                    names.push(format!("OMEGA({},{})", i + 1, j + 1));
                }
            }
            names
        };
        let sigma_names: Vec<String> = (1..=n_sigma).map(|i| format!("SIGMA({i},{i})")).collect();

        Some(ImpmapTrace {
            rows: trace_rows,
            theta_names,
            omega_names,
            sigma_names,
        })
    } else {
        None
    };

    if verbose {
        eprintln!("{} completed. Final OFV (Laplace) = {:.4}", label, ofv);
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
        impmap_trace,
        bayes: None,
    })
}

/// Extract the lower triangle of a square matrix in row-major order:
/// `(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), …`
fn lower_triangle(m: &DMatrix<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            out.push(m[(i, j)]);
        }
    }
    out
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

    let unpack_thetas = |packed: &[f64]| -> Vec<f64> {
        (0..n_theta)
            .map(|i| {
                if theta_packs_log_mask[i] {
                    packed[i].exp()
                } else {
                    packed[i]
                }
            })
            .collect()
    };

    // Weighted observation NLL, parallel over subjects. Each subject contributes
    // Σₖ w̃ₖ · obs_nll(yᵢ | ηᵢₖ, θ, σ).
    let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let val: f64 = population
            .subjects
            .par_iter()
            .zip(draws.par_iter())
            .map_init(EventPkParams::default, |scratch, (subject, d)| {
                let mut s = 0.0f64;
                for (w, eta) in d.weights.iter().zip(d.etas.iter()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_to_proposal_hessian_inverts_an_in_bounds_covariance() {
        // A covariance comfortably inside [floor, Ωii] passes through unclamped,
        // so the returned precision is its exact inverse.
        let cov = DMatrix::from_row_slice(2, 2, &[0.25, 0.05, 0.05, 0.16]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[10.0, 10.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        let recovered = h.clone().try_inverse().expect("h must be invertible");
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (recovered[(i, j)] - cov[(i, j)]).abs() < 1e-9,
                    "inverse-of-inverse must recover cov at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn covariance_to_proposal_hessian_floors_collapsed_diagonal() {
        // A zero-variance dimension (collapsed ESS) is floored to a tiny absolute
        // value rather than inverting to a near-delta proposal.
        let cov = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.2]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "floored result must be finite"
        );
        let expected = 1.0 / IMP_PROPOSAL_COV_FLOOR;
        assert!(
            (h[(0, 0)] - expected).abs() / expected < 1e-9,
            "floored precision should be ~1/floor, got {}",
            h[(0, 0)]
        );
    }

    #[test]
    fn covariance_to_proposal_hessian_caps_exploding_diagonal() {
        // A heavy-tailed-outlier-inflated covariance is capped at Ωii so the
        // proposal can't drift broader than the prior.
        let cov = DMatrix::from_row_slice(2, 2, &[1e14, 0.0, 0.0, 1e12]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[0.2, 0.3]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!((h[(0, 0)] - 1.0 / 0.2).abs() / (1.0 / 0.2) < 1e-9);
        assert!((h[(1, 1)] - 1.0 / 0.3).abs() / (1.0 / 0.3) < 1e-9);
    }

    #[test]
    fn covariance_to_proposal_hessian_falls_back_on_non_pd() {
        // An indefinite covariance is not Cholesky-invertible → zero matrix,
        // signalling `build_proposal` to use its Ω fallback.
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 5.0, 5.0, 1.0]);
        let omega = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0]));
        let h = covariance_to_proposal_hessian(&cov, &omega, IMP_PROPOSAL_COV_FLOOR);
        assert!(
            h.iter().all(|&v| v == 0.0),
            "non-PD covariance must yield the zero fallback"
        );
    }
}
