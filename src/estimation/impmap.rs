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
    MIX_ESS_PMIX_FLOOR,
};
use crate::estimation::inner_optimizer::{find_ebe, EbeResult, InnerLoopStats};
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, pack_params, theta_packs_log};
use crate::estimation::saem::{
    floor_omega_diagonal, get_mixture_mu_ref_pairs, get_mu_ref_pairs, mixture_mu_ref_means,
    MixtureMuRefPair,
};
use crate::pk::EventPkParams;
use crate::stats::likelihood::obs_nll_subject_into;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Positive-definite floor for free Ω diagonals (matches the SAEM constant).
const OMEGA_DIAG_FLOOR: f64 = 1e-6;

/// Adaptive-sampling (`auto`) target for the **per-subject** objective Monte-Carlo
/// standard error, in −2 log L units: `total_MC_SE / √N`. NONMEM's `AUTO` `STDOBJ`
/// targets the *total* objective SE, which grows as √N, so a large but perfectly
/// well-sampled dataset would ramp purely because it has more subjects. The
/// per-subject normalization makes the criterion N-independent (it tracks the
/// average per-subject sampling adequacy, which is what biases the M-step). The
/// value `0.05` reproduces NONMEM's effective stopping point on the FREM workshop
/// model (its `STDOBJ = 1.0` total target at N≈475 ⇒ 1.0/√475 ≈ 0.046): the count
/// still ramps 300→~10000 there, while a rich low-dim fit (e.g. warfarin, total
/// SE ≈ 0.07) never trips it.
const AUTO_STDOBJ_TARGET: f64 = 0.05;

/// Hard cap on the adaptive per-subject sample count (NONMEM `ISAMPEND` default).
const AUTO_SAMPLES_MAX: usize = 10_000;

/// Absolute lower bound on the IMP proposal-covariance diagonal — a numerical
/// guard against a literally-zero (degenerate-ESS) variance, NOT a statistical
/// floor. It must stay well below any real conditional variance: with rich data
/// the conditional posterior of η is legitimately tiny (orders of magnitude below
/// the prior Ω), and flooring it against Ω would make the proposal far too broad
/// and collapse the ESS (the very rich-data failure mode that motivates IMPMAP).
const IMP_PROPOSAL_COV_FLOOR: f64 = 1e-10;

/// Objective ceiling beyond which an estimation run is declared diverged rather
/// than converged (issue #528). A collapsed-weight runaway pins θ to the
/// parameter bounds and the final FOCE-Laplace OFV blows up to ~1e35 — a finite
/// value that `is_finite()` would otherwise accept. Real −2logL objectives scale
/// with the number of observations and reach at most ~1e6–1e7 even for very
/// large cohorts, so 1e15 leaves ~8 orders of head-room above any legitimate fit
/// while closing the gap that let a 1e12–1e29 runaway pass as converged.
pub(crate) const OFV_DIVERGENCE_CAP: f64 = 1e15;

/// Whether an estimation run's final objective signals convergence rather than a
/// runaway. A non-finite OFV is an outright divergence; a finite-but-enormous
/// one (≥ [`OFV_DIVERGENCE_CAP`]) is the bounded blowup of a collapsed-weight
/// run, which `is_finite()` alone would wave through (issue #528). Shared by the
/// IMP/IMPMAP MCEM core and SAEM (both can run away to a finite blowup).
pub(crate) fn objective_converged(ofv: f64) -> bool {
    ofv.is_finite() && ofv.abs() < OFV_DIVERGENCE_CAP
}

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

/// Names of non-fixed thetas that have **no associated ETA** (are not the target
/// of any mu-reference). Under IMP/IMPMAP such fixed-effect-only parameters are
/// estimated solely through the importance-weighted θ M-step, which carries an
/// IS-weight bias for weakly-identified parameters and can converge to the wrong
/// value (issue #406: the FREM `FRD1` absorption-fraction drove to 0.90 vs a
/// FOCEI/NONMEM value of ~0.4). NONMEM's IMP methods require every estimated
/// parameter to carry a random effect; ferx applies mu-referencing
/// automatically, so the user only needs to add the ETA.
pub(crate) fn non_fixed_thetas_without_eta(
    model: &CompiledModel,
    theta_fixed: &[bool],
) -> Vec<String> {
    use std::collections::HashSet;
    let mut with_eta: HashSet<&str> = model
        .mu_refs
        .values()
        .map(|m| m.theta_name.as_str())
        .collect();
    // A MIXNUM-switched typical value carries an ETA in every class arm, so its
    // class thetas must not be flagged as fixed-effect-only (#996) — they are
    // mu-referenced per class, not estimated by the weighted M-step alone.
    if let Some(spec) = model.mixture.as_ref() {
        with_eta.extend(
            spec.mu_refs
                .iter()
                .flat_map(|m| m.theta_names.iter().map(String::as_str)),
        );
    }
    model
        .theta_names
        .iter()
        .enumerate()
        .filter(|(i, name)| {
            !theta_fixed.get(*i).copied().unwrap_or(false) && !with_eta.contains(name.as_str())
        })
        .map(|(_, name)| name.clone())
        .collect()
}

/// One subject's multi-start MAP: the warm-start (or cold-start) solve plus
/// `mceta` additional random starting points drawn from N(0, Ω) via the supplied
/// Cholesky factor. The start with the lowest NLL wins; `omega_chol = None` (or
/// `mceta == 0`) degrades to the single warm-start solve.
///
/// Shared by the single-population MAP sweep ([`run_map_multistart`]) and the
/// mixture MCEM E-step, which runs the same multi-start *per class* inside the
/// subject's rayon task (so `MIXNUM` stays correct on the worker thread).
#[allow(clippy::too_many_arguments)]
fn find_ebe_multistart(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    inner_maxiter: usize,
    inner_tol: f64,
    warm: Option<&[f64]>,
    mu: Option<&[f64]>,
    inner_restarts: usize,
    mceta: usize,
    omega_chol: Option<&DMatrix<f64>>,
    subj_seed: u64,
) -> EbeResult {
    // Baseline: warm-start (or cold-start from η = 0).
    let mut best = find_ebe(
        model,
        subject,
        params,
        inner_maxiter,
        inner_tol,
        warm,
        mu,
        inner_restarts,
    );
    let Some(l_omega) = omega_chol else {
        return best;
    };
    let n_eta = l_omega.nrows();
    let mut rng = StdRng::seed_from_u64(subj_seed);
    for _start in 0..mceta {
        // Draw z ~ N(0, I), compute eta_start = L_Ω · z.
        let z: Vec<f64> = (0..n_eta)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let eta_start = l_omega * DVector::from_vec(z);
        let eta_slice: Vec<f64> = eta_start.iter().copied().collect();

        let candidate = find_ebe(
            model,
            subject,
            params,
            inner_maxiter,
            inner_tol,
            Some(&eta_slice),
            mu,
            inner_restarts,
        );
        if candidate.nll < best.nll {
            best = candidate;
        }
    }
    best
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
            // Deterministic per-subject, per-iteration seed, separated from IS draws.
            let subj_seed = seed
                .wrapping_add(i as u64)
                .wrapping_add((iteration as u64) << 32)
                .wrapping_add(0x4D43_4554_4100u64);
            find_ebe_multistart(
                model,
                subject,
                params,
                inner_maxiter,
                inner_tol,
                warm,
                Some(mu_k),
                0,
                mceta,
                omega_chol.as_ref(),
                subj_seed,
            )
        })
        .collect();

    let stats = InnerLoopStats {
        n_unconverged: results.iter().filter(|r| !r.converged).count(),
        n_fallback: results.iter().filter(|r| r.used_fallback).count(),
        n_start_rejected: results.iter().filter(|r| r.hard_reject).count(),
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
    // Sobol quasi-random draws are only implemented for the multivariate-normal
    // (`impmap_proposal_df = normal`) proposal; the Student-t default makes the
    // option a silent no-op, so flag the mismatch rather than ignore it.
    let use_sobol = options.impmap_sobol && nu.is_infinite();
    let mut result = run_mcem(
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
        use_sobol,
        options.impmap_trace,
        options.impmap_auto,
    )?;
    if options.impmap_sobol && !use_sobol {
        result.warnings.push(
            "IMPMAP: `impmap_sobol = true` is ignored because the proposal is a Student-t \
             (`impmap_proposal_df` is finite). Sobol draws apply only to the multivariate-normal \
             proposal — set `impmap_proposal_df = normal` to use them."
                .to_string(),
        );
    }
    // Sobol is also silently disabled when the defensive mixture is active: the
    // broad/narrow per-sample branch breaks the quasi-random sequence, so the
    // sampler falls back to plain Monte Carlo. Surface that rather than ignore it
    // (issue #528).
    if use_sobol && options.imp_defensive_alpha > 0.0 {
        result.warnings.push(
            "IMPMAP: `impmap_sobol = true` is ignored because the defensive mixture is active \
             (`imp_defensive_alpha > 0`). Quasi-random draws are disabled when sampling from the \
             broad/narrow mixture — set `imp_defensive_alpha = 0` to use Sobol draws."
                .to_string(),
        );
    }
    Ok(result)
}

/// Run IMP as an estimator (NONMEM `METHOD=IMP`). Thin wrapper over the shared
/// MCEM core with sample-moment re-centering (conditional mode found only on the
/// first iteration); resolves the `imp_*` options. The evaluation-only
/// `imp_eval_only` path lives in `importance_sampling.rs`.
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
        "imp_proposal_df",
        options.imp_iterations,
        options.imp_samples,
        options.imp_proposal_df,
        options.imp_averaging,
        options.imp_seed.unwrap_or(12345),
        options.imp_low_ess_threshold,
        0,     // mceta: no multi-start MAP for IMP
        false, // use_sobol: IMP has no Sobol option
        false, // collect_trace: IMP has no trace option
        options.imp_auto,
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
    auto: bool,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // ---- Validation ----
    // Mixture (#985): estimating IMP/IMPMAP runs a class-partitioned MCEM
    // (per-class IS E-step + responsibility-weighted M-step). Delegate to the
    // mixture core, which reuses this function's option resolution via the same
    // arguments.
    if model.mixture.is_some() {
        return run_mcem_mixture(
            model,
            population,
            init_params,
            warm_etas,
            options,
            recenter,
            label,
            df_key,
            n_iter_opt,
            k_opt,
            nu,
            n_avg_opt,
            seed,
            threshold,
            mceta,
            use_sobol,
            collect_trace,
            auto,
        );
    }
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
    // `k_samples` is mutable so the adaptive (`auto`) path can ramp it up between
    // iterations when the objective is too Monte-Carlo-noisy (NONMEM `AUTO`).
    let mut k_samples = k_opt.max(2);
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

    // STRONG warning: any estimated (non-fixed) θ without an associated ETA is
    // handled only by the weighted M-step, which is biased for such parameters
    // under importance sampling. ferx mu-references automatically, so the user
    // only needs to attach a random effect.
    let thetas_without_eta = non_fixed_thetas_without_eta(model, &init_params.theta_fixed);
    if !thetas_without_eta.is_empty() {
        warnings.push(format!(
            "{label}: estimated parameter(s) [{}] have NO associated ETA. NONMEM's \
             IMP/IMPMAP require every estimated parameter to carry a random effect; a \
             fixed-effect-only parameter is estimated solely through the importance-weighted \
             M-step, which is biased for weakly-identified parameters and may converge to the \
             wrong value. STRONGLY add an ETA to each (e.g. `P = TVP * exp(ETA_P)` with a small, \
             optionally FIX, omega — ferx applies mu-referencing automatically), or hold the \
             parameter FIX, or use FOCEI.",
            thetas_without_eta.join(", ")
        ));
    }

    let mu_ref_pairs = get_mu_ref_pairs(model);
    // A log-mu-referenced typical value is updated only through the closed-form
    // `log θ += mean(η)` shift. When its paired η carries negligible IIV (a tiny,
    // often `FIX`ed ω — e.g. a structural parameter given a dummy random effect
    // so it can be mu-referenced), that population mean is ≈ 0 and the typical
    // value would be frozen at its initial value (#411). Route those pairs to the
    // weighted-likelihood M-step instead (the same channel that estimates σ and
    // non-mu-ref θ), where the data can actually move them. Decided once from the
    // initial Ω so a parameter never toggles channels between iterations.
    const WEAK_IIV_VAR: f64 = 1e-3;
    let weak_mu_ref: std::collections::HashSet<usize> = mu_ref_pairs
        .iter()
        .filter(|&&(_t, e)| init_params.omega.matrix[(e, e)] < WEAK_IIV_VAR)
        .map(|&(t, _e)| t)
        .collect();
    if !weak_mu_ref.is_empty() {
        let mut names: Vec<&str> = weak_mu_ref
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        names.sort_unstable();
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but their random effect has \
             negligible variance (ω < {WEAK_IIV_VAR:.0e}); the mu-ref mean-shift carries no \
             information, so they are estimated through the weighted M-step instead.",
            names.join(", ")
        ));
    }
    // Closed-form mu-ref shift is used as long as at least one pair has real IIV.
    let use_closed_form = mu_ref_pairs
        .iter()
        .any(|&(t, _e)| !weak_mu_ref.contains(&t));
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

    // ---- FREM Rao-Blackwellisation (issue #406) ----
    // For FREM models, integrate the covariate etas analytically and importance
    // sample only the PK etas (a well-conditioned low-dim problem with near-unit
    // ESS) instead of all n_eta etas (~1–2% ESS). Partition is model-static;
    // per-subject covariate deviations are computed inside the E-step. `None` for
    // non-FREM models → the full-dimensional path is used unchanged.
    let frem_rb: Option<(Vec<usize>, Vec<usize>)> = if !options.frem_rao_blackwell {
        None
    } else {
        model
            .frem_config
            .as_ref()
            .map(|fc| crate::estimation::importance_sampling::frem_pk_cov_partition(fc, n_eta))
            .filter(|(pk, cov)| !pk.is_empty() && !cov.is_empty())
    };

    // ---- Trace: collect per-iteration parameters (analogous to NONMEM .ext) ----
    let mut trace_rows: Vec<ImpmapTraceRow> = if collect_trace {
        Vec::with_capacity(n_iter + 2)
    } else {
        Vec::new()
    };

    // ESS state from the final completed E-step, used to flag importance-sampling
    // moment bias (which scales as 1/(K·ESS) and is severe in high dimensions).
    let mut final_ess_median = 1.0_f64;
    let mut final_n_collapsed = 0usize;
    // Effective (possibly ramped) sample count and the final objective MC SE,
    // for the post-fit under-sampling warning.
    let mut final_k_samples = k_samples;
    let mut final_mc_se = 0.0_f64;

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
            mixture: None,
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
        let defensive_alpha = options.imp_defensive_alpha;
        // Build the defensive-mixture broad component q_broad = N(0, Ω) once per
        // MCEM iteration (Ω changes each iteration but is shared across subjects)
        // and pass it read-only into the parallel subject loop, avoiding a
        // redundant per-subject Cholesky of Ω⁻¹ (issue #528). `None` when the
        // mixture is inactive. The FREM RB path builds its own component.
        let defensive_mixture = crate::estimation::importance_sampling::DefensiveMixture::new(
            &omega_inv,
            n_eta,
            defensive_alpha,
        );
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

                // FREM: Rao-Blackwellised low-dimensional PK sampling. The
                // conditional PK proposal is usually well matched, but a
                // per-subject ISCALE pilot search still rescues subjects where
                // `h_pp` is a poor curvature estimate (sparse PK data) — see
                // `find_optimal_iscale_frem_rb` (issue #406 follow-up).
                if let Some((ref pk_idx, ref cov_idx)) = frem_rb {
                    if let Some(fc) = model.frem_config.as_ref() {
                        if let Some((sampled, observed, d)) =
                            crate::estimation::importance_sampling::subject_frem_partition(
                                subject,
                                &params_k.theta,
                                fc,
                                pk_idx,
                                cov_idx,
                            )
                        {
                            let rb_iscale =
                                crate::estimation::importance_sampling::find_optimal_iscale_frem_rb(
                                    model,
                                    subject,
                                    &params_k.theta,
                                    &params_k.sigma.values,
                                    &center,
                                    &h_post,
                                    &omega_inv,
                                    &params_k.omega.matrix,
                                    &sampled,
                                    &observed,
                                    &d,
                                    n_eta,
                                    nu,
                                    subj_seed,
                                    scratch,
                                    iscale_min,
                                    iscale_max,
                                );
                            if let Some(rb) =
                                crate::estimation::importance_sampling::subject_is_draws_frem_rb(
                                    model,
                                    subject,
                                    &params_k.theta,
                                    &params_k.sigma.values,
                                    &center,
                                    &h_post,
                                    &omega_inv,
                                    &params_k.omega.matrix,
                                    &sampled,
                                    &observed,
                                    &d,
                                    n_eta,
                                    k_samples,
                                    nu,
                                    subj_seed,
                                    scratch,
                                    rb_iscale,
                                    use_sobol,
                                    defensive_alpha,
                                )
                            {
                                return rb;
                            }
                        }
                    }
                }

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
                    defensive_mixture.as_ref(),
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
        let mut ess_fracs: Vec<f64> = Vec::with_capacity(draws.len());
        for d in &draws {
            ll += d.log_marginal;
            if d.ess_fraction < threshold {
                n_low_ess += 1;
            }
            ess_fracs.push(d.ess_fraction);
        }
        let minus2ll = -2.0 * ll;
        // Objective Monte-Carlo standard error from the self-normalized ESS
        // (Geweke: Var(log p̂ᵢ) ≈ (1/ESS_frac − 1)/K; degenerate subjects get a
        // finite fallback). −2 log L scales this by 2.
        let var_ll: f64 = ess_fracs
            .iter()
            .map(|&f| {
                if f > 1e-6 {
                    ((1.0 / f) - 1.0).max(0.0) / k_samples as f64
                } else {
                    1.0
                }
            })
            .sum();
        let mc_se = 2.0 * var_ll.sqrt();
        // Per-subject objective SE (N-independent): see AUTO_STDOBJ_TARGET.
        let mc_se_per_subject = mc_se / (ess_fracs.len().max(1) as f64).sqrt();
        final_mc_se = mc_se_per_subject;
        final_k_samples = k_samples;
        // Adaptive sampling (NONMEM `AUTO`): ramp K up (×2, capped) while the
        // objective is too MC-noisy, so the M-step is driven by signal not noise.
        if auto && mc_se_per_subject > AUTO_STDOBJ_TARGET && k_samples < AUTO_SAMPLES_MAX {
            let new_k = (k_samples.saturating_mul(2)).min(AUTO_SAMPLES_MAX);
            if verbose {
                eprintln!(
                    "{label}: auto-sampling — per-subject objective MC SE {mc_se_per_subject:.3} \
                     > {AUTO_STDOBJ_TARGET:.2}, raising K {k_samples} → {new_k}"
                );
            }
            k_samples = new_k;
        }
        // Track the final E-step's ESS health for the post-fit bias warning.
        final_n_collapsed = ess_fracs.iter().filter(|&&f| f <= 1e-6).count();
        ess_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        final_ess_median = ess_fracs.get(ess_fracs.len() / 2).copied().unwrap_or(1.0);

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
                // Weak-IIV mu-ref θ are estimated by this M-step, not the shift,
                // so leave their bounds free here.
                if weak_mu_ref.contains(&t) {
                    continue;
                }
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
                // Weak-IIV mu-ref θ were already updated by the weighted M-step;
                // their η-mean shift is ≈ 0 and uninformative — skip it.
                if weak_mu_ref.contains(&t) {
                    continue;
                }
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

    // ---- Importance-sampling bias warning ----
    // Self-normalized IS moment estimates carry a finite-sample bias that drives
    // the M-step. For a fixed proposal quality it scales as ≈ 1/K, but the
    // constant grows with the sampling dimension, so a sample count that is ample
    // in low dimensions becomes badly under-sampled for a high-dimensional (e.g.
    // FREM) model. Empirically the absorption typical value on the FREM workshop
    // model (13 ETAs) drifts from ~2.6 (FOCEI/NONMEM) to ~4.6 at K=300 and
    // recovers toward ~2.8 at K=6000 — while the *median* ESS stays ~0.8 the
    // whole time, so ESS alone does not flag it. Use the per-dimension sample
    // density K / n_eta as the trigger (≥ ~100 keeps the bias small here), plus
    // any fully-collapsed subjects. Skip for eval-only IS (no M-step to bias).
    // Triggers, using the *effective* (possibly auto-ramped) sample count:
    //  - too few samples for the dimension (a fast proxy that catches the common
    //    case before the MC SE is even trustworthy), or
    //  - the objective is still MC-noisy at the end (the direct measure — also
    //    fires when `auto` ramped to the cap and could not reach the target), or
    //  - a subject's proposal fully collapsed.
    const MIN_SAMPLES_PER_ETA: usize = 100;
    // The dimension heuristic only applies when `auto` is off — with `auto` on,
    // a low starting count is fine because it ramps when (and only when) the
    // objective is actually noisy, so the direct MC-SE check is the only trigger.
    let under_sampled = !auto && n_eta > 0 && final_k_samples < MIN_SAMPLES_PER_ETA * n_eta;
    let noisy = final_mc_se > 2.0 * AUTO_STDOBJ_TARGET;
    if !options.imp_eval_only && (under_sampled || noisy || final_n_collapsed > 0) {
        let (sample_opt, auto_opt) = if recenter == ProposalRecenter::Map {
            ("impmap_samples", "impmap_auto")
        } else {
            ("imp_samples", "imp_auto")
        };
        let collapse = if final_n_collapsed > 0 {
            format!(
                " {} subject(s) had a fully collapsed proposal (ESS ≈ 0), whose moments \
                 are unreliable regardless of sample count — check their EBE/Hessian quality.",
                final_n_collapsed
            )
        } else {
            String::new()
        };
        let advice = if auto {
            format!(
                "`{auto_opt}` is enabled but the sample count {} (cap {AUTO_SAMPLES_MAX}) \
                 was not enough to reach the MC-SE target.",
                final_k_samples
            )
        } else {
            format!(
                "Raise `{sample_opt}` (high-dimensional / FREM models typically need several \
                 thousand) or set `{auto_opt} = true` to ramp it automatically."
            )
        };
        warnings.push(format!(
            "{label}: {} importance samples for a {}-ETA model give a noisy objective \
             (per-subject MC SE = {:.3}, target {:.2}; median ESS/K = {:.2}). The weighted M-step \
             moments then carry a finite-sample bias — typical-value and Ω estimates may be off, \
             and it shrinks only as the sample count grows. {}{}",
            final_k_samples,
            n_eta,
            final_mc_se,
            AUTO_STDOBJ_TARGET,
            final_ess_median,
            advice,
            collapse
        ));
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
        mixture: None,
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
    let packed = pack_params(&final_params);
    let out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
    } = out;
    warnings.extend(cov_warnings);

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
        // report is a sane final objective. A non-finite OFV means the MCEM
        // diverged; but a *runaway* (importance weights collapse → weighted
        // M-step walks θ to the bounds) produces a finite-but-enormous OFV
        // (~1e35), which `is_finite()` alone would wave through as converged and
        // could then win multi-start selection (issue #528). Treat any objective
        // beyond a generous physical ceiling as non-converged too — real −2logL
        // values are at most thousands. Matches SAEM's `converged: ofv.is_finite()`
        // in spirit while catching the bounded blowup.
        converged: objective_converged(ofv),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
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
        impmap_trace,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        mixture_posteriors: None,
    })
}

/// Estimating IMP / IMPMAP under a mixture model (#985): a class-partitioned
/// MCEM. Each iteration importance-samples each subject's η *within every class*
/// (under a [`MixtureClassGuard`], so `MIXNUM` resolves to that class), forms the
/// deterministic class responsibilities `PMIX_ik ∝ p_ik · L_ik`, then runs the
/// responsibility-weighted M-steps:
///
/// * mixing coefficients — from the responsibilities (reusing SAEM's mixing
///   M-step with `r̄ = PMIX`);
/// * Ω (class-shared) — `Σ_i Σ_k PMIX_ik · secondmoment_ik / N`;
/// * θ / σ — the responsibility- and importance-weighted observation M-step
///   ([`theta_sigma_weighted_mstep_mixture`]), which recovers the class-switched
///   typical values from each class's own weighted samples.
///
/// Ω/σ are class-shared (per-class overrides are rejected); IOV, FREM and SDE are
/// not supported here (as for the single-population MCEM). The final OFV,
/// per-subject `MIXEST`/`PMIX`, and EBEs come from the K-fold marginal
/// (`mixture_ofv`) at the estimates, matching FOCEI/SAEM.
#[allow(clippy::too_many_arguments)]
fn run_mcem_mixture(
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
    auto: bool,
) -> Result<OuterResult, String> {
    use crate::estimation::importance_sampling::{
        compute_posterior_hessian, subject_is_draws, DefensiveMixture,
    };
    use crate::estimation::mixture::combine_subject;
    use crate::parser::model_parser::{eval_mixing_log_probs, MixtureClassGuard};

    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();
    let spec = model
        .mixture
        .as_ref()
        .expect("run_mcem_mixture on non-mixture model");
    let mp0 = init_params
        .mixture
        .as_ref()
        .ok_or_else(|| format!("{label} mixture requires params.mixture (per-class Ω/Σ)"))?;

    // ---- Validation (mixture scope) ----
    if n_eta == 0 {
        return Err(format!(
            "{label} requires at least one random effect (n_eta = 0)."
        ));
    }
    if model.n_kappa > 0 {
        return Err(format!(
            "{label} for a mixture does not yet support inter-occasion variability (IOV); use \
             SAEM or FOCE/FOCEI (#985)."
        ));
    }
    if model.frem_config.is_some() {
        return Err(format!(
            "{label} for a mixture does not support FREM models (#985)."
        ));
    }
    if model.is_sde() {
        return Err(format!(
            "{label} for a mixture does not support SDE / [diffusion] models."
        ));
    }
    if !mp0.omega_override_addr.is_empty() || !mp0.sigma_override_addr.is_empty() {
        return Err(format!(
            "{label} for a mixture does not yet support per-class Omega/Sigma overrides \
             (omega(k)/sigma(k)); Omega/Sigma are class-shared. Use FOCE/FOCEI for per-class \
             overrides (#985)."
        ));
    }
    if nu.is_finite() && nu < 1.0 {
        return Err(format!(
            "{label}: {df_key} must be >= 1.0 (or +inf), got {nu}"
        ));
    }

    let n_iter = n_iter_opt.max(1);
    let k_samples = k_opt.max(2);
    let n_avg = n_avg_opt.min(n_iter);
    let verbose = options.verbose;
    let cancel = &options.cancel;
    let n_classes = spec.n_classes;

    // ---- Packing scaffolding (mirrors run_mcem / SAEM) ----
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
    let unpack_theta = |i: usize, v: f64| -> f64 {
        if theta_packs_log_mask[i] {
            v.exp()
        } else {
            v
        }
    };

    // Mixing state: detects the mixing thetas and drives the mixing M-step from
    // the deterministic responsibilities (no stochastic averaging — `rbar` is set
    // to the current `PMIX` each iteration). The closed-form mu-ref θ shift is not
    // used for a mixture (the class-switched typical value can't pair to one η);
    // every θ flows through the responsibility-weighted M-step.
    let mut mix =
        crate::estimation::saem_mixture::SaemMixture::build(model, init_params, population);

    // Reject a theta that drives both the mixing expression and a structural
    // typical value. Like SAEM (`run_saem`), this MCEM runs *separated* M-steps:
    // `theta_sigma_weighted_mstep_mixture` fits such a theta from the weighted
    // observation likelihood and `mstep_mixing` then overwrites it from the
    // responsibilities, silently discarding the structural estimate. FOCEI's
    // joint marginal handles the shared parameter, so route there (#992 review).
    let overlap = crate::estimation::saem_mixture::mixing_structural_overlap(
        model,
        init_params,
        population,
        &mix.mixing_theta_idx,
    );
    if !overlap.is_empty() {
        let names: Vec<String> = overlap
            .iter()
            .map(|&j| {
                init_params
                    .theta_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("theta[{j}]"))
            })
            .collect();
        return Err(format!(
            "{label} cannot fit a mixture where a mixing-coefficient theta also drives the \
             structural model: {} appear(s) in both the [mixture] mixing expression and an \
             [individual_parameters] typical value. The weighted θ/σ M-step and the mixing \
             M-step run separately, so a shared parameter would be double-owned. Split it into \
             two thetas (one for structure, one for mixing), or fit with FOCE/FOCEI (whose \
             joint marginal handles the shared parameter) (#985).",
            names.join(", ")
        ));
    }

    let mut warnings: Vec<String> = Vec::new();

    // The mixture MCEM never uses the closed-form log-mu-ref θ shift (a
    // class-switched typical value cannot be paired to a single η), so *every* θ
    // is estimated through the importance-weighted M-step alone — the channel
    // that is biased for weakly-identified parameters. The single-population
    // `run_mcem` pushes the equivalent warnings before this delegation point;
    // repeat them here so a mixture fit is not silently unwarned (#992 review).
    let mixing_names: std::collections::HashSet<&str> = mix
        .mixing_theta_idx
        .iter()
        .filter_map(|&j| model.theta_names.get(j).map(String::as_str))
        .collect();
    // Mixing thetas are excluded: they are estimated by `mstep_mixing` from the
    // class responsibilities, not by the weighted M-step, so an ETA on them is
    // neither expected nor useful.
    let thetas_without_eta: Vec<String> =
        non_fixed_thetas_without_eta(model, &init_params.theta_fixed)
            .into_iter()
            .filter(|n| !mixing_names.contains(n.as_str()))
            .collect();
    if !thetas_without_eta.is_empty() {
        warnings.push(format!(
            "{label}: estimated parameter(s) [{}] have NO associated ETA. NONMEM's \
             IMP/IMPMAP require every estimated parameter to carry a random effect; a \
             fixed-effect-only parameter is estimated solely through the importance-weighted \
             M-step, which is biased for weakly-identified parameters and may converge to the \
             wrong value. STRONGLY add an ETA to each (e.g. `P = TVP * exp(ETA_P)` with a small, \
             optionally FIX, omega), or hold the parameter FIX, or use FOCEI.",
            thetas_without_eta.join(", ")
        ));
    }
    // ---- Class-aware closed-form mu-ref θ shift (#996) ----
    //
    // MCEM over a mixture importance-samples η *within every class*, so a
    // per-class posterior mean η̄_ic exists directly and the EM update for the
    // class-switched typical value is the responsibility-weighted
    //     log θ_k += (Σ_i r_ic η̄_ic) / (Σ_i r_ic),   r_ic = PMIX_ic
    // (`mixture_mu_ref_means`). A class-*shared* typical value repeats the same
    // theta in every class slot, so it collapses to the pooled `mean_i(η̄_i)`
    // shift the single-population `run_mcem` applies. Those θ are pinned out of
    // the weighted M-step below, exactly as in `run_mcem`.
    //
    // Disabled by `mu_referencing = false`, which restores the pre-#996 behaviour
    // (every θ through the weighted M-step alone). Note this differs from the
    // single-population IMP path, where the closed-form shift is EM-mandatory and
    // applied regardless of that option.
    const WEAK_IIV_VAR: f64 = 1e-3;
    let mut mix_mu_ref_pairs: Vec<MixtureMuRefPair> = if options.mu_referencing {
        get_mixture_mu_ref_pairs(model)
    } else {
        Vec::new()
    };
    // An identity-packed θ (`theta_lower < 0`) is not on the log scale, so the
    // additive shift is not its EM optimum — route it to the weighted M-step.
    let identity_packed: Vec<usize> = mix_mu_ref_pairs
        .iter()
        .flat_map(|p| p.theta_idx.iter().copied())
        .filter(|&t| !theta_packs_log_mask[t])
        .collect();
    mix_mu_ref_pairs.retain(|p| p.theta_idx.iter().all(|&t| theta_packs_log_mask[t]));
    // A η with negligible IIV carries no mean-shift information, so its typical
    // values would be frozen at their inits (#411) — same guard as `run_mcem`,
    // applied per pair (all of that η's class θ share the one ω).
    let weak: Vec<usize> = mix_mu_ref_pairs
        .iter()
        .filter(|p| init_params.omega.matrix[(p.eta_idx, p.eta_idx)] < WEAK_IIV_VAR)
        .flat_map(|p| p.theta_idx.iter().copied())
        .collect();
    mix_mu_ref_pairs.retain(|p| init_params.omega.matrix[(p.eta_idx, p.eta_idx)] >= WEAK_IIV_VAR);
    let name_list = |idx: Vec<usize>| -> String {
        let mut n: Vec<&str> = idx
            .iter()
            .map(|&t| model.theta_names.get(t).map(String::as_str).unwrap_or("?"))
            .collect();
        n.sort_unstable();
        n.dedup();
        n.join(", ")
    };
    if !identity_packed.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but declared with a negative \
             lower bound, so they are packed on the identity scale; the closed-form mu-ref \
             shift does not apply and they are estimated by the importance-weighted M-step \
             instead (#996).",
            name_list(identity_packed)
        ));
    }
    if !weak.is_empty() {
        warnings.push(format!(
            "{label}: typical value(s) {} are log-mu-referenced but their random effect has \
             negligible variance (ω < {WEAK_IIV_VAR:.0e}); the mu-ref mean-shift carries no \
             information, so they are estimated through the weighted M-step instead.",
            name_list(weak)
        ));
    }
    // Thetas the closed-form shift owns — pinned out of the weighted M-step each
    // iteration. A FIXed θ is left to the regular fixed-bounds path.
    let mu_ref_pinned: Vec<usize> = {
        let mut v: Vec<usize> = mix_mu_ref_pairs
            .iter()
            .flat_map(|p| p.theta_idx.iter().copied())
            .filter(|&t| !init_params.theta_fixed.get(t).copied().unwrap_or(false))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if mix_mu_ref_pairs.is_empty() {
        warnings.push(format!(
            "{label}: no class-aware mu-referencing was applied, so every typical value is \
             estimated through the importance-weighted M-step alone and may converge poorly. \
             Write each class-switched typical value as a MIXNUM chain of \
             `THETA_k * exp(ETA)` terms on one ETA, or cross-check against a FOCEI fit (#996)."
        ));
    }
    if collect_trace {
        warnings.push(format!(
            "{label}: iteration trace collection (impmap_trace) is not implemented for mixture \
             models; no trace is reported (#985)."
        ));
    }
    if auto {
        warnings.push(format!(
            "{label}: adaptive sampling (auto) is not implemented for mixture models; the sample \
             count stays at K={k_samples} for every iteration (#985)."
        ));
    }

    let mut theta_cur = init_params.theta.clone();
    let mut sigma_cur = init_params.sigma.values.clone();
    let mut omega_mat = init_params.omega.matrix.clone();

    // Previous iteration's per-class weighted draws (for the IMP `SampleMoments`
    // recenter, which rebuilds the proposal from their second moments). `None` on
    // iteration 1 and — as in `run_mcem` — never retained on the IMPMAP (`Map`)
    // path, which re-runs the MAP solve and only needs the previous weighted
    // means as warm starts. Holding the full `K · n_subjects · n_classes` sample
    // sets there would cost `K×` the memory for nothing (#992 review).
    let mut prev_draws: Option<Vec<Vec<SubjectDraws>>> = None;
    // Previous iteration's per-class weighted posterior means, `[class][subject]`
    // — the warm start for the next iteration's inner solve on both paths.
    let mut prev_means: Option<Vec<Vec<Vec<f64>>>> = None;
    let warm0: Option<Vec<DVector<f64>>> = warm_etas.map(|e| e.to_vec());

    // Running average of the estimates over the final `n_avg` iterations.
    let mut acc_theta = vec![0.0f64; n_theta];
    let mut acc_sigma = vec![0.0f64; n_sigma];
    let mut acc_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
    let mut n_acc = 0usize;

    if verbose {
        eprintln!(
            "{label} (mixture): {n_subjects} subjects × {n_classes} classes, {n_eta} ETAs, \
             {n_iter} iters, K={k_samples}/subject/class, seed={seed}"
        );
    }

    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(cancel) {
            break;
        }
        // Shared Ω/σ view for this iteration (identical across classes; the class
        // enters only through the `MIXNUM` guard on the structural model).
        let omega_k = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        let omega_inv = omega_k.inv.clone();
        let log_det_omega = omega_k.log_det;
        let defensive = DefensiveMixture::new(&omega_inv, n_eta, options.imp_defensive_alpha);
        let params_iter = ModelParameters {
            theta: theta_cur.clone(),
            theta_names: init_params.theta_names.clone(),
            theta_lower: init_params.theta_lower.clone(),
            theta_upper: init_params.theta_upper.clone(),
            theta_fixed: init_params.theta_fixed.clone(),
            omega: omega_k.clone(),
            omega_fixed: init_params.omega_fixed.clone(),
            sigma: SigmaVector {
                values: sigma_cur.clone(),
                names: init_params.sigma.names.clone(),
            },
            sigma_fixed: init_params.sigma_fixed.clone(),
            omega_iov: None,
            kappa_fixed: init_params.kappa_fixed.clone(),
            mixture: None,
        };
        let run_inner = recenter == ProposalRecenter::Map || prev_draws.is_none();
        let prev = prev_draws.as_ref();
        let prev_mean = prev_means.as_ref();
        // Cholesky of the shared Ω for the MCETA random restarts (computed once
        // per iteration, outside the parallel loop). `None` when MCETA is off.
        let omega_chol = if mceta > 0 && run_inner {
            omega_mat.clone().cholesky().map(|c| c.l())
        } else {
            None
        };

        // ---- E-step: per subject, per class IS draws + class responsibilities ----
        // Parallel over subjects; the class loop runs inside each worker so the
        // `MIXNUM` guard is entered on the same thread as the solve/draws.
        #[allow(clippy::type_complexity)]
        let per_subject: Vec<(Vec<SubjectDraws>, Vec<f64>, f64)> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                // Poll per subject: one subject's E-step is `n_classes` inner
                // solves plus `n_classes · K` importance draws — the dominant
                // per-iteration cost — so without this a cancel set mid-sweep is
                // not observed until the whole subject × class loop finishes.
                // Mirrors `run_mcem`; the driver breaks right after the collect,
                // so the placeholder draws never reach an M-step.
                if crate::cancel::is_cancelled(cancel) {
                    let class_draws: Vec<SubjectDraws> = (0..n_classes)
                        .map(|_| SubjectDraws::cancelled(n_eta))
                        .collect();
                    return (class_draws, vec![1.0 / n_classes as f64; n_classes], 0.0);
                }
                let logp = eval_mixing_log_probs(spec, &theta_cur, &subject.covariates);
                let mut class_draws: Vec<SubjectDraws> = Vec::with_capacity(n_classes);
                let mut nll = vec![0.0f64; n_classes];
                for c in 0..n_classes {
                    let _g = MixtureClassGuard::enter(c + 1);
                    let (center, h_post) = if run_inner {
                        let warm_i = prev_mean
                            .map(|pm| DVector::from_row_slice(&pm[c][i]))
                            .or_else(|| warm0.as_ref().map(|w| w[i].clone()));
                        let ebe = find_ebe_multistart(
                            model,
                            subject,
                            &params_iter,
                            options.inner_maxiter,
                            options.inner_tol,
                            warm_i.as_ref().map(|v| v.as_slice()),
                            None,
                            options.inner_restarts,
                            mceta,
                            omega_chol.as_ref(),
                            seed.wrapping_add(i as u64)
                                .wrapping_add((k as u64) << 32)
                                .wrapping_add((c as u64) << 48)
                                .wrapping_add(0x4D43_4554_4100u64),
                        );
                        let h_post = compute_posterior_hessian(
                            model,
                            subject,
                            &params_iter.theta,
                            &ebe.eta,
                            &params_iter.sigma.values,
                            &ebe.h_matrix,
                            &omega_inv,
                            n_eta,
                            scratch,
                        );
                        (ebe.eta, h_post)
                    } else {
                        let pd = &prev.expect("prev set when !run_inner")[c][i];
                        let center = DVector::from_row_slice(&pd.mean);
                        let cov = &pd.second_moment - &center * center.transpose();
                        let h_post = covariance_to_proposal_hessian(
                            &cov,
                            &omega_mat,
                            IMP_PROPOSAL_COV_FLOOR,
                        );
                        (center, h_post)
                    };
                    let subj_seed = seed
                        .wrapping_add(i as u64)
                        .wrapping_add((k as u64) << 32)
                        .wrapping_add((c as u64) << 48);
                    // Per-subject, per-class ISCALE pilot search — the same guard
                    // the single-population E-step runs (#528/#961): it rescues
                    // subjects whose posterior Hessian is a poor curvature
                    // estimate, whose IS weights would otherwise collapse and
                    // drive the weighted M-step to the bounds. Skipping it is
                    // materially riskier under a mixture, where an off-class
                    // proposal is a poor fit by construction (#992 review).
                    let iscale = find_optimal_iscale(
                        model,
                        subject,
                        &params_iter.theta,
                        &params_iter.sigma.values,
                        &center,
                        &h_post,
                        &omega_inv,
                        log_det_omega,
                        n_eta,
                        nu,
                        subj_seed,
                        scratch,
                        options.iscale_min,
                        options.iscale_max,
                    );
                    let d = subject_is_draws(
                        model,
                        subject,
                        &params_iter.theta,
                        &params_iter.sigma.values,
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
                        defensive.as_ref(),
                    );
                    nll[c] = -d.log_marginal;
                    class_draws.push(d);
                }
                let (contribution, pmix_i, _mixest) = combine_subject(&logp, &nll);
                (class_draws, pmix_i, contribution)
            })
            .collect();

        if crate::cancel::is_cancelled(cancel) {
            break;
        }

        // Reorganize into [class][subject] draws + [subject][class] responsibilities.
        let mut draws_by_class: Vec<Vec<SubjectDraws>> = (0..n_classes)
            .map(|_| Vec::with_capacity(n_subjects))
            .collect();
        let mut pmix: Vec<Vec<f64>> = Vec::with_capacity(n_subjects);
        let mut ofv = 0.0f64;
        for (class_draws, pmix_i, contribution) in per_subject {
            for (c, d) in class_draws.into_iter().enumerate() {
                draws_by_class[c].push(d);
            }
            pmix.push(pmix_i);
            ofv += contribution;
        }
        if verbose {
            // Per-subject ESS: the worst per-class proposal among the classes the
            // subject actually contributes to (a collapsed proposal in a class
            // with PMIX ≈ 0 does not degrade that subject's marginal).
            let n_low_ess = (0..n_subjects)
                .filter(|&i| {
                    let worst = (0..n_classes)
                        .filter(|&c| pmix[i][c] > MIX_ESS_PMIX_FLOOR)
                        .map(|c| draws_by_class[c][i].ess_fraction)
                        .fold(f64::INFINITY, f64::min);
                    worst.is_finite() && worst < threshold
                })
                .count();
            eprintln!(
                "{label} (mixture) iter {k}: marginal −2 log L = {ofv:.4} \
                 (low-ESS subjects: {n_low_ess})"
            );
        }

        // ---- M-step ----
        // (a) θ / σ from the responsibility- and importance-weighted obs NLL.
        // The mixing thetas are pinned out of this M-step: they do not enter
        // `obs_nll_subject_into` at all, so leaving them free only inflates the
        // BOBYQA dimension with degenerate directions and can park them at an
        // arbitrary point that `mstep_mixing` (4–8 evals) then has to recover
        // from (#992 review). `mstep_mixing` below is their sole owner.
        let mut mstep_theta_lower = log_theta_lower.clone();
        let mut mstep_theta_upper = log_theta_upper.clone();
        for &j in &mix.mixing_theta_idx {
            mstep_theta_lower[j] = log_theta[j];
            mstep_theta_upper[j] = log_theta[j];
        }
        // Pin the class-aware mu-ref θ too: the closed-form shift below is their
        // sole owner, so NLopt optimizes only σ and the remaining θ (#996).
        for &t in &mu_ref_pinned {
            mstep_theta_lower[t] = log_theta[t];
            mstep_theta_upper[t] = log_theta[t];
        }
        let mstep_maxiter: u32 = if k <= n_iter / 2 { 4 } else { 8 };
        let (new_log_theta, new_log_sigma) = theta_sigma_weighted_mstep_mixture(
            model,
            population,
            &draws_by_class,
            &pmix,
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

        // (a2) Class-aware closed-form mu-ref θ shift (#996):
        //      log θ_k += (Σ_i r_ic η̄_ic) / (Σ_i r_ic) over the classes θ_k serves.
        // The draws are not re-centred: Ω below is the weighted second moment
        // about zero of these same draws, which is what the single-population
        // `run_mcem` also uses (its shift likewise leaves the draws alone), and
        // the next E-step re-solves at the updated θ.
        for pair in &mix_mu_ref_pairs {
            let eta_idx = pair.eta_idx;
            let means = mixture_mu_ref_means(n_theta, &pair.theta_idx, &pmix, |i, c| {
                draws_by_class[c][i].mean[eta_idx]
            });
            for (t, mean_eta) in means.iter().enumerate() {
                // `None` means no responsibility mass reached any class this θ
                // serves — hold it and let the next E-step move it.
                let Some(mean_eta) = mean_eta else { continue };
                if init_params.theta_fixed.get(t).copied().unwrap_or(false) {
                    continue;
                }
                log_theta[t] =
                    (log_theta[t] + mean_eta).clamp(log_theta_lower[t], log_theta_upper[t]);
            }
        }

        theta_cur = (0..n_theta)
            .map(|i| unpack_theta(i, log_theta[i]))
            .collect();
        sigma_cur = log_sigma.iter().map(|&v| v.exp()).collect();

        // (b) mixing coefficients from the class responsibilities.
        mix.rbar = pmix.clone();
        crate::estimation::saem_mixture::mstep_mixing(
            model,
            population,
            &mix,
            &mut theta_cur,
            &init_params.theta_lower,
            &init_params.theta_upper,
            mstep_maxiter,
        );
        for &j in &mix.mixing_theta_idx {
            log_theta[j] = if theta_packs_log_mask[j] {
                theta_cur[j].max(1e-12).ln()
            } else {
                theta_cur[j]
            };
        }

        // (c) Ω (class-shared): responsibility-weighted second moment, masked/floored.
        let mut new_omega = DMatrix::<f64>::zeros(n_eta, n_eta);
        for (c, class) in draws_by_class.iter().enumerate() {
            for (i, d) in class.iter().enumerate() {
                new_omega += pmix[i][c] * &d.second_moment;
            }
        }
        new_omega /= n_subjects as f64;
        for a in 0..n_eta {
            for b in 0..n_eta {
                if !init_params.omega.free_mask[(a, b)] {
                    new_omega[(a, b)] = 0.0;
                }
                let fa = init_params.omega_fixed.get(a).copied().unwrap_or(false);
                let fb = init_params.omega_fixed.get(b).copied().unwrap_or(false);
                if fa || fb {
                    new_omega[(a, b)] = init_params.omega.matrix[(a, b)];
                }
            }
        }
        floor_omega_diagonal(&mut new_omega, &init_params.omega_fixed, OMEGA_DIAG_FLOOR);
        omega_mat = new_omega;

        // Accumulate the final `n_avg` iterations for the reported point estimate.
        if k > n_iter - n_avg {
            for (a, &v) in acc_theta.iter_mut().zip(&theta_cur) {
                *a += v;
            }
            for (a, &v) in acc_sigma.iter_mut().zip(&sigma_cur) {
                *a += v;
            }
            acc_omega += &omega_mat;
            n_acc += 1;
        }

        prev_means = Some(
            draws_by_class
                .iter()
                .map(|class| class.iter().map(|d| d.mean.clone()).collect())
                .collect(),
        );
        prev_draws = if recenter == ProposalRecenter::SampleMoments {
            Some(draws_by_class)
        } else {
            None
        };
    }

    // A cancel observed inside (or at the top of) the loop breaks out with
    // parameters from a truncated MCEM. Return before the K-class marginal pass
    // and the covariance step — together the most expensive part of the run —
    // rather than paying for them and handing back a bogus `Ok` (#992 review).
    if crate::cancel::is_cancelled(cancel) {
        return Err("cancelled by user".to_string());
    }

    // ---- Point estimate: average over the final n_avg iterations ----
    let (final_theta, final_sigma, final_omega_mat) = if n_acc > 0 {
        (
            acc_theta
                .iter()
                .map(|&v| v / n_acc as f64)
                .collect::<Vec<_>>(),
            acc_sigma
                .iter()
                .map(|&v| v / n_acc as f64)
                .collect::<Vec<_>>(),
            acc_omega / n_acc as f64,
        )
    } else {
        (theta_cur.clone(), sigma_cur.clone(), omega_mat.clone())
    };
    let final_omega = OmegaMatrix::from_matrix(
        final_omega_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    // Rebuild the per-class Ω/Σ as the (shared) final estimates so the marginal
    // post-loop pass and covariance step see the mixture structure.
    let final_mixture = init_params.mixture.as_ref().map(|m| {
        let mut mm = m.clone();
        for o in mm.omega.iter_mut() {
            *o = final_omega.clone();
        }
        for s in mm.sigma.iter_mut() {
            s.values = final_sigma.clone();
        }
        mm
    });
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
        mixture: final_mixture,
    };

    // ---- Final EBEs / OFV / posteriors via the K-fold marginal ----
    let meval =
        crate::estimation::mixture::mixture_ofv(model, population, &final_params, options, None);
    let eta_hats = meval.mixest_etas;
    let h_matrices = meval.mixest_h_mats;
    let final_kappas = meval.mixest_kappas;
    let ofv = meval.ofv;
    let mixture_posteriors = Some(crate::estimation::outer_optimizer::MixturePosteriors {
        pmix: meval.pmix,
        mixest: meval.mixest,
    });

    // ---- Covariance step (mixture-aware) ----
    let packed = pack_params(&final_params);
    let cov_out = crate::estimation::covariance::run_covariance_step(
        &packed,
        &final_params,
        model,
        population,
        &eta_hats,
        &h_matrices,
        &final_kappas,
        options,
        verbose.then_some("Running covariance step..."),
    );
    let crate::estimation::covariance::CovStepOutcome {
        matrix: covariance_matrix,
        wall_time_secs: covariance_wall_time_secs,
        warnings: cov_warnings,
        sir_fallback_proposal,
    } = cov_out;
    warnings.extend(cov_warnings);

    if verbose {
        eprintln!("{label} (mixture) completed. Final marginal OFV = {ofv:.4}");
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged: objective_converged(ofv),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
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
        packed_estimate: None,
        mixture_posteriors,
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
        // Reduce in subject order (collect then serial sum) so the objective is
        // reproducible regardless of rayon worker count — a parallel `.sum()`
        // folds along thread-count-dependent split points and f64 addition is
        // non-associative (#703).
        let per_subj: Vec<f64> = population
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
            .collect();
        let val: f64 = per_subj.iter().sum();
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

/// Mixture (#985) weighted θ/σ M-step. Minimises the class-responsibility- and
/// importance-weighted observation NLL
/// `Σᵢ Σ_c PMIX_ic · Σₖ w̃_ick · obs_nll(yᵢ | η_ick, θ, σ | MIXNUM = c)` over the
/// per-(subject,class) sample sets. Each class's samples were drawn under that
/// class's proposal, and its `obs_nll` runs under a [`MixtureClassGuard`] on the
/// worker thread, so a class-switched typical value (`if MIXNUM == c …`) is
/// estimated from its own class's weighted samples. `draws[c][i]` is class `c`'s
/// draws for subject `i`; `pmix[i][c]` is the class responsibility.
#[allow(clippy::too_many_arguments)]
fn theta_sigma_weighted_mstep_mixture(
    model: &CompiledModel,
    population: &Population,
    draws: &[Vec<crate::estimation::importance_sampling::SubjectDraws>],
    pmix: &[Vec<f64>],
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
    use crate::parser::model_parser::MixtureClassGuard;
    let n = n_theta + n_sigma;
    let n_classes = draws.len();

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

    let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = unpack_thetas(&xv[..n_theta]);
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let per_subj: Vec<f64> = population
            .subjects
            .par_iter()
            .enumerate()
            .map_init(EventPkParams::default, |scratch, (i, subject)| {
                let mut s = 0.0f64;
                for c in 0..n_classes {
                    let p = pmix[i][c];
                    if p == 0.0 {
                        continue;
                    }
                    let _g = MixtureClassGuard::enter(c + 1);
                    let d = &draws[c][i];
                    let mut sc = 0.0f64;
                    for (w, eta) in d.weights.iter().zip(d.etas.iter()) {
                        if *w == 0.0 {
                            continue;
                        }
                        sc += w * obs_nll_subject_into(model, subject, &th, &sg, eta, scratch);
                    }
                    s += p * sc;
                }
                s
            })
            .collect();
        let val: f64 = per_subj.iter().sum();
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
    let _ = opt.optimize(&mut xs);
    (xs[..n_theta].to_vec(), xs[n_theta..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── #996 end-to-end helpers: a tiny 2-class mixture dataset ──────────
    //
    // Two weight groups with well-separated clearances, so the class draw is
    // stable and a handful of iterations is enough to exercise the M-step.

    /// Small 1-cpt IV dataset, `n_per` subjects at each of two weights.
    fn mix996_csv(n_per: usize) -> String {
        let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        let mut sid = 0;
        for (g, &wt) in [60.0_f64, 90.0].iter().enumerate() {
            let cl = if g == 0 { 1.0 } else { 3.0 };
            for _ in 0..n_per {
                sid += 1;
                s.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
                for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0].iter().enumerate() {
                    let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                    let dv = c * (1.0 + 0.02 * ((sid + ti) as f64).sin());
                    s.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
                }
            }
        }
        s
    }

    fn mix996_pop(n_per: usize) -> Population {
        use std::io::Write;
        let csv = mix996_csv(n_per);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::io::datareader::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// Two-class mixture. `v_expr` picks whether V carries a class-shared
    /// mu-referenced ETA (`TVV * exp(ETA_V)`) or none (`TVV`).
    fn mix996_model(v_expr: &str) -> CompiledModel {
        let src = format!(
            r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = {v_expr}

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
        );
        crate::parser::model_parser::parse_model_string(&src).expect("mixture model parses")
    }

    #[test]
    fn impmap_mixture_applies_class_aware_mu_ref_shift() {
        // The MIXNUM-switched clearances are mu-referenced per class, so the
        // "no class-aware mu-referencing" fallback warning must be absent.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            !res.warnings
                .iter()
                .any(|w| w.contains("no class-aware mu-referencing was applied")),
            "class-aware shift should be active, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn impmap_mixture_mu_referencing_off_restores_numerical_mstep() {
        // The degenerate-oracle switch: `mu_referencing = false` must reproduce
        // the pre-#996 behaviour, where every θ goes through the weighted M-step.
        let model = mix996_model("TVV");
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        opts.mu_referencing = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("no class-aware mu-referencing was applied")),
            "expected the fallback warning with mu_referencing off, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn impmap_mixture_weak_iiv_mu_ref_routes_to_weighted_mstep() {
        // A near-zero ω on the mu-ref η carries no mean-shift information, so the
        // class θ must fall back to the weighted M-step with a warning (#411).
        let src = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 1e-6 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let pop = mix996_pop(3);
        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Impmap;
        opts.impmap_iterations = 2;
        opts.impmap_samples = 40;
        opts.impmap_seed = Some(996);
        opts.run_covariance_step = false;
        let res = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP Ok");
        assert!(
            res.warnings
                .iter()
                .any(|w| w.contains("negligible variance") && w.contains("TVCL1")),
            "expected the weak-IIV warning, got {:?}",
            res.warnings
        );
    }

    #[test]
    fn objective_converged_rejects_runaway_and_nonfinite() {
        // A normal objective converges.
        assert!(objective_converged(-249.23));
        assert!(objective_converged(0.0));
        // A legitimately large objective for a huge cohort still converges —
        // 1e15 leaves head-room above any real −2logL.
        assert!(objective_converged(1.0e7));
        // Non-finite is divergence.
        assert!(!objective_converged(f64::NAN));
        assert!(!objective_converged(f64::INFINITY));
        assert!(!objective_converged(f64::NEG_INFINITY));
        // The bounded blowup (issue #528): finite but enormous → not converged,
        // so a collapsed-weight runaway can't masquerade as a good fit or win
        // multi-start selection. The lowered cap also rejects a less-extreme
        // 1e20 runaway that the old 1e30 cap waved through.
        assert!(!objective_converged(1e35));
        assert!(!objective_converged(1e20));
        assert!(!objective_converged(OFV_DIVERGENCE_CAP));
    }

    #[test]
    fn flags_non_fixed_theta_without_eta() {
        // CL/V have ETAs (mu-referenced); FRAC is an estimated theta with no ETA;
        // TVFIX is FIX. Only FRAC should be flagged.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.0)
  theta TVV(10.0, 0.0)
  theta FRAC(0.5, 0.0, 1.0)
  theta TVFIX(2.0, FIX)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = FRAC + TVFIX

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let fixed = &model.default_params.theta_fixed;
        let flagged = non_fixed_thetas_without_eta(&model, fixed);
        assert!(
            flagged.contains(&"FRAC".to_string()),
            "FRAC flagged: {flagged:?}"
        );
        assert!(!flagged.contains(&"TVCL".to_string()), "TVCL has ETA");
        assert!(!flagged.contains(&"TVV".to_string()), "TVV has ETA");
        assert!(!flagged.contains(&"TVFIX".to_string()), "TVFIX is FIX");
    }

    #[test]
    fn no_flag_when_all_thetas_have_eta() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.0)
  theta TVV(10.0, 0.0)
  omega ETA_CL ~ 0.1
  omega ETA_V  ~ 0.1
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
        let model = crate::parser::model_parser::parse_model_string(src).unwrap();
        let fixed = &model.default_params.theta_fixed;
        let flagged = non_fixed_thetas_without_eta(&model, fixed);
        assert!(flagged.is_empty(), "expected no flags, got {flagged:?}");
    }

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

    // ── Estimating IMP/IMPMAP under a mixture (#985) ──

    const MIX_MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

    fn mix_pop() -> Population {
        use std::io::Write;
        let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
        for (sid, &(cl, wt)) in [(1.0_f64, 60.0_f64), (3.0, 90.0)].iter().enumerate() {
            let id = sid + 1;
            csv.push_str(&format!("{id},0,0,100,1,1,{wt}\n"));
            for t in [0.5_f64, 1.0, 2.0, 4.0] {
                let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                csv.push_str(&format!("{id},{t},{c:.5},0,0,1,{wt}\n"));
            }
        }
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        crate::read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
    }

    /// A short estimating IMPMAP (Map recenter) run on a mixture returns Ok,
    /// carries the mixture through, and populates per-subject MIXEST + posteriors.
    #[test]
    fn impmap_mixture_short_run_populates_posteriors() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 2;
        opts.impmap_samples = 200;
        opts.impmap_seed = Some(1);
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        assert!(res.ofv.is_finite());
        assert!(res.mixture_posteriors.is_some(), "mixture posteriors set");
        let mp = res.mixture_posteriors.unwrap();
        assert_eq!(mp.mixest.len(), pop.subjects.len());
        assert_eq!(mp.pmix.len(), pop.subjects.len());
    }

    /// The IMP (SampleMoments recenter) estimating path also runs on a mixture.
    #[test]
    fn imp_mixture_short_run_ok() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.imp_iterations = 2;
        opts.imp_samples = 200;
        opts.imp_seed = Some(1);
        opts.run_covariance_step = false;
        let res = run_imp(&model, &pop, &model.default_params, None, &opts).expect("imp mixture");
        assert!(res.ofv.is_finite());
        assert!(res.mixture_posteriors.is_some());
    }

    /// Estimating IMP/IMPMAP rejects a mixture with per-class Ω/Σ overrides.
    #[test]
    fn estimating_imp_rejects_per_class_overrides() {
        const OV: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.15

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(OV).unwrap();
        let pop = mix_pop();
        let opts = FitOptions::default();
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("per-class overrides must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.contains("per-class Omega/Sigma overrides"),
            "got: {err}"
        );
    }
    /// A theta that drives both the mixing expression and a structural typical
    /// value is rejected up front: the weighted θ/σ M-step would fit it from the
    /// observation likelihood and `mstep_mixing` would then overwrite it,
    /// silently discarding the structural estimate (#992 review). SAEM rejects
    /// the same shape; the estimating IMP/IMPMAP MCEM must too.
    #[test]
    fn estimating_imp_rejects_mixing_theta_that_drives_structure() {
        const SHARED: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta MIXL(0.5, 0.01, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = 10.0 * MIXL

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
        let model = crate::parser::model_parser::parse_model_string(SHARED).unwrap();
        let pop = mix_pop();
        let opts = FitOptions::default();
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("a mixing theta driving the structural model must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.contains("also drives the structural model") && err.contains("MIXL"),
            "got: {err}"
        );
    }

    /// The mixture MCEM delegates before `run_mcem` builds its mu-ref warnings,
    /// so it must push its own no-ETA warning. It names `TVV` (a bare typical
    /// value) but neither the mixing theta `MIXL` (which `mstep_mixing` owns)
    /// nor the class-switched `TVCL1`/`TVCL2`, which carry `ETA_CL` in every
    /// class arm and are class-aware mu-referenced (#996).
    #[test]
    fn mixture_mcem_warns_about_weighted_mstep_only_thetas() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 1;
        opts.impmap_samples = 50;
        opts.impmap_seed = Some(1);
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        let joined = res.warnings.join("\n");
        assert!(
            joined.contains("have NO associated ETA") && joined.contains("TVV"),
            "no-ETA warning missing: {joined}"
        );
        assert!(
            !joined.contains("MIXL"),
            "the mixing theta must not be flagged as a missing-ETA parameter: {joined}"
        );
        assert!(
            !joined.contains("TVCL1") && !joined.contains("TVCL2"),
            "class-aware mu-referenced thetas must not be flagged as ETA-less: {joined}"
        );
        assert!(
            !joined.contains("no class-aware mu-referencing was applied"),
            "the class-aware shift is available for this model: {joined}"
        );
    }

    /// `impmap_trace` and `impmap_auto` are not implemented for mixtures; they
    /// must warn rather than be silently ignored (#992 review).
    #[test]
    fn mixture_mcem_warns_that_trace_and_auto_are_ignored() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 1;
        opts.impmap_samples = 50;
        opts.impmap_seed = Some(1);
        opts.impmap_trace = true;
        opts.impmap_auto = true;
        opts.run_covariance_step = false;
        let res =
            run_impmap(&model, &pop, &model.default_params, None, &opts).expect("impmap mixture");
        let joined = res.warnings.join("\n");
        assert!(
            joined.contains("trace collection") && joined.contains("adaptive sampling"),
            "trace/auto no-op warnings missing: {joined}"
        );
        assert!(res.impmap_trace.is_none(), "no trace is produced");
    }

    /// A cancel set before the run returns `Err` instead of falling through to
    /// the K-class marginal pass and the covariance step — the most expensive
    /// part of the run — with parameters from a truncated MCEM (#992 review).
    #[test]
    fn mixture_mcem_cancel_returns_err() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let mut opts = FitOptions::default();
        opts.impmap_iterations = 5;
        opts.impmap_samples = 200;
        opts.impmap_seed = Some(1);
        let flag = crate::cancel::CancelFlag::new();
        flag.cancel();
        opts.cancel = Some(flag);
        let err = match run_impmap(&model, &pop, &model.default_params, None, &opts) {
            Ok(_) => panic!("a cancelled mixture MCEM must not return Ok"),
            Err(e) => e,
        };
        assert!(err.contains("cancelled by user"), "got: {err}");
    }

    /// `find_ebe_multistart` degrades to the plain warm-start solve when no Ω
    /// Cholesky is supplied, and never returns a worse start than the baseline
    /// when MCETA restarts are requested.
    #[test]
    fn find_ebe_multistart_never_worse_than_baseline() {
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let params = &model.default_params;
        let subject = &pop.subjects[0];

        let baseline =
            find_ebe_multistart(&model, subject, params, 50, 1e-4, None, None, 0, 0, None, 7);
        let direct = find_ebe(&model, subject, params, 50, 1e-4, None, None, 0);
        assert!(
            (baseline.nll - direct.nll).abs() < 1e-12,
            "no Cholesky must reproduce find_ebe exactly: {} vs {}",
            baseline.nll,
            direct.nll
        );

        let chol = params.omega.matrix.clone().cholesky().unwrap().l();
        let multi = find_ebe_multistart(
            &model,
            subject,
            params,
            50,
            1e-4,
            None,
            None,
            0,
            4,
            Some(&chol),
            7,
        );
        assert!(
            multi.nll <= baseline.nll + 1e-12,
            "MCETA restarts must not return a worse mode: {} vs {}",
            multi.nll,
            baseline.nll
        );
    }

    /// The weighted θ/σ M-step honours a pinned coordinate (`lower == upper`),
    /// which is how the mixture MCEM keeps the mixing thetas — absent from the
    /// observation likelihood — out of the BOBYQA search (#992 review).
    #[test]
    fn weighted_mstep_mixture_honours_pinned_theta_bounds() {
        use crate::estimation::importance_sampling::SubjectDraws;
        let model = crate::parser::model_parser::parse_model_string(MIX_MODEL).unwrap();
        let pop = mix_pop();
        let params = &model.default_params;
        let n_theta = params.theta.len();
        let n_sigma = params.sigma.values.len();
        let n_eta = model.n_eta;

        // One (degenerate, single-sample) draw set per class per subject.
        let draws_by_class: Vec<Vec<SubjectDraws>> = (0..2)
            .map(|_| {
                pop.subjects
                    .iter()
                    .map(|_| {
                        let mut d = SubjectDraws::cancelled(n_eta);
                        d.etas = vec![vec![0.0; n_eta]];
                        d.weights = vec![1.0];
                        d
                    })
                    .collect()
            })
            .collect();
        let pmix: Vec<Vec<f64>> = pop.subjects.iter().map(|_| vec![0.5, 0.5]).collect();

        let log_theta: Vec<f64> = params.theta.iter().map(|&t| t.max(1e-10).ln()).collect();
        let log_sigma: Vec<f64> = params
            .sigma
            .values
            .iter()
            .map(|&s| s.max(1e-10).ln())
            .collect();
        let mask = vec![true; n_theta];
        let mut lower = vec![-10.0; n_theta];
        let mut upper = vec![10.0; n_theta];
        // Pin theta 0 exactly, as the mixing-theta pin does.
        lower[0] = log_theta[0];
        upper[0] = log_theta[0];

        let (out_theta, _out_sigma) = theta_sigma_weighted_mstep_mixture(
            &model,
            &pop,
            &draws_by_class,
            &pmix,
            &log_theta,
            &log_sigma,
            &lower,
            &upper,
            &vec![-8.0; n_sigma],
            &vec![5.0; n_sigma],
            n_theta,
            n_sigma,
            8,
            &mask,
        );
        assert!(
            (out_theta[0] - log_theta[0]).abs() < 1e-12,
            "pinned theta must not move: {} vs {}",
            out_theta[0],
            log_theta[0]
        );
    }
}
