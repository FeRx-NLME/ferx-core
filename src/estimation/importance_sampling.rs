//! Importance-sampling marginal log-likelihood (Monolix / NONMEM `$EST METHOD=IMP`).
//!
//! For each subject i, given the EBE η̂ᵢ and Hessian Hᵢ from the inner loop,
//! draw K samples η_ik from a Student-t proposal centred at η̂ᵢ with scale
//! Σᵢ = (Hᵢ + λI)⁻¹ and DoF ν, then estimate
//!
//!   log p(yᵢ | θ) ≈ logsumexp_k [log p(yᵢ | η_ik, θ) + log p(η_ik | θ) − log q(η_ik)] − log K
//!
//! Summed across subjects this gives `−2 log L_IS = −2 Σᵢ log p(yᵢ | θ)`,
//! a lower-bias estimate of the marginal likelihood than the FOCE/Laplace
//! approximation when individual posteriors of η are non-Gaussian
//! (sparse-data PK, strong nonlinearity).
//!
//! The kernel is rayon-parallel over subjects; per-subject RNGs are seeded
//! from `options.is_seed.wrapping_add(i as u64)` so the result is
//! deterministic for a given seed.
//!
//! ## IOV (v2: joint sampling)
//!
//! For models with inter-occasion variability, we perform joint sampling of
//! (η, κ₁, …, κ_O) where O is the number of occasions. The proposal is built
//! from the full (n_eta + n_kappa × n_occasions) posterior Hessian, and the
//! IS weights include both the η and κ priors. This makes the IS -2LL directly
//! comparable to FOCE and NONMEM's `$EST METHOD=IMP LAPLACIAN=1`.

use crate::pk::{compute_predictions_with_tv_into, predict_iov, EventPkParams};
use crate::stats::likelihood::{obs_nll_subject_into, split_obs_by_occasion};
use crate::stats::residual_error::compute_r_diag;
use crate::stats::special::{ln_gamma, log_normal_cdf};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{ChiSquared, Distribution, StandardNormal};
use rayon::prelude::*;

/// `2π` as `f64`.  Used in the Gaussian-prior log-density.
const TWO_PI: f64 = std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// Inverse normal CDF (Acklam rational approximation)
// ---------------------------------------------------------------------------

/// Inverse normal CDF (probit function): given u ∈ (0, 1), returns z such that
/// Φ(z) = u. Uses the Acklam rational approximation with full f64 precision
/// (~1.15e-9 relative error over the entire range).
///
/// Used to transform uniform Sobol quasi-random points to N(0,1) draws.
fn inv_normal_cdf(u: f64) -> f64 {
    let u = u.clamp(1e-15, 1.0 - 1e-15);

    // Acklam (2003) rational approximation coefficients
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if u < P_LOW {
        // Lower tail
        let q = (-2.0 * u.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if u <= P_HIGH {
        // Central region
        let q = u - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail (symmetry)
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Generate `k_samples` quasi-random N(0,1) vectors of dimension `d` using
/// Sobol sequences with Cranley-Patterson randomization.
///
/// Returns a Vec of k_samples vectors, each of length d.
fn sobol_normal_draws(d: usize, k_samples: usize, seed: u64) -> Vec<Vec<f64>> {
    use sobol::params::JoeKuoD6;
    use sobol::Sobol;

    // Cranley-Patterson rotation: shift Sobol points by a uniform random vector
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0x534F_424F_4C00_0000u64));
    let shift: Vec<f64> = (0..d).map(|_| rand::Rng::gen::<f64>(&mut rng)).collect();

    let params = JoeKuoD6::minimal(); // supports up to 100 dims
    let sobol_seq = Sobol::<f64>::new(d, &params);

    sobol_seq
        .take(k_samples)
        .map(|point| {
            point
                .iter()
                .zip(shift.iter())
                .map(|(&u, &s)| {
                    let u_shifted = (u + s) % 1.0;
                    inv_normal_cdf(u_shifted)
                })
                .collect()
        })
        .collect()
}

/// Estimate `−2 log L` by importance sampling. See module docstring for the
/// algorithm and IOV caveats.
pub fn run_importance_sampling(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> Result<ImportanceSamplingResult, String> {
    let n_subjects = population.subjects.len();
    if eta_hats.len() != n_subjects || h_matrices.len() != n_subjects {
        return Err(format!(
            "IS: eta_hats ({}) / h_matrices ({}) length must match n_subjects ({})",
            eta_hats.len(),
            h_matrices.len(),
            n_subjects
        ));
    }

    // n_eta == 0: model has no random effects, so p(y|θ) = ∏ p(yⱼ|θ) directly
    // (no integration needed). IS is meaningless here — the marginal collapses
    // to the obs likelihood. Refuse rather than silently returning 0.
    if model.n_eta == 0 {
        return Err("Importance sampling requires at least one random effect. \
             With n_eta = 0 the marginal likelihood is just the observation likelihood — \
             read `FitResult.ofv` directly (no IS needed)."
            .to_string());
    }

    // SDE / EKF likelihood inflates the residual variance with per-observation
    // process-noise (see `individual_nll_into_with_schedule`). Our IS obs-NLL
    // path (`obs_nll_subject_into`) does not thread that through yet, so an
    // SDE model would silently report a wrong −2 log L. Refuse upfront — the
    // user can still get the Laplace OFV via FOCE / FOCEI.
    if model.is_sde() {
        return Err(
            "Importance sampling is not yet supported for SDE / [diffusion] models. \
             The EKF process-noise variance is not included in the IS observation likelihood, \
             so the marginal would be biased. Use FOCE / FOCEI for the Laplace OFV instead."
                .to_string(),
        );
    }

    // Defensive: every Jacobian must have shape (n_obs_i × n_eta). The only
    // path in the current inner loop that violates this (degenerate-Ω
    // early-out, `inner_optimizer.rs`) is already caught globally by the
    // omega.log_det check below, but a future inner-loop path that fails
    // per-subject would otherwise panic deep inside `compute_posterior_hessian`.
    // Cheap to verify here; refuse with a clear message if violated.
    for (i, j) in h_matrices.iter().enumerate() {
        let expected_rows = population.subjects[i].observations.len();
        if j.ncols() != model.n_eta || j.nrows() != expected_rows {
            return Err(format!(
                "IS: subject {} has Jacobian shape ({}×{}); expected ({}×{}). \
                 The preceding estimator likely failed to compute EBEs for this subject \
                 — fix the upstream convergence issue (tighter `inner_tol`, more `outer_maxiter`) \
                 before running IMP.",
                population.subjects[i].id,
                j.nrows(),
                j.ncols(),
                expected_rows,
                model.n_eta,
            ));
        }
    }

    // Defensive: the joint (η, κ) path builds a mode vector of size
    // `n_eta + n_occ·n_iov`, filling the κ blocks from the per-occasion EBEs in
    // `kappas[i]`. The two notions of "number of occasions" — the EBE count and
    // `split_obs_by_occasion(subject).len()` — must agree, or the fill loop
    // would index out of bounds (κ too long) or silently leave occasions at
    // κ = 0 (κ too short). Verify once up front so the parallel loop can index
    // freely. Subjects with no κ EBEs fall through to the η-only path.
    if model.n_kappa > 0 {
        for (i, subject) in population.subjects.iter().enumerate() {
            let kap_len = kappas.get(i).map(|v| v.len()).unwrap_or(0);
            if kap_len == 0 {
                continue;
            }
            let n_occ = split_obs_by_occasion(subject).len();
            if kap_len != n_occ {
                return Err(format!(
                    "IS: subject {} has {} κ occasion block(s) but {} observation \
                     occasion(s); the joint (η, κ) proposal requires these to match. \
                     This usually means the preceding estimator produced κ EBEs on a \
                     different occasion grouping than the data.",
                    subject.id, kap_len, n_occ,
                ));
            }
        }
    }

    let n_eta = model.n_eta;
    let k_samples = options.is_samples;
    let nu = options.is_proposal_df;
    let seed = options.is_seed.unwrap_or(42);
    let threshold = options.is_low_ess_threshold;
    let cancel = &options.cancel;

    if k_samples < 2 {
        return Err(format!("IS: is_samples must be >= 2, got {}", k_samples));
    }
    if nu < 1.0 {
        return Err(format!("IS: is_proposal_df must be >= 1.0, got {}", nu));
    }

    let kappa_treatment = if model.n_kappa > 0 {
        KappaTreatment::Marginalized
    } else {
        KappaTreatment::NotApplicable
    };

    // Pre-compute Ω⁻¹ and log|Ω| once (shared across all subjects + samples).
    let omega = &params.omega;
    if !omega.log_det.is_finite() {
        return Err("IS: Ω log-determinant is not finite — cannot evaluate η prior".into());
    }
    let omega_inv = omega.inv.clone();
    let log_det_omega = omega.log_det;

    // For IOV models, pre-compute Ω_iov⁻¹ and log|Ω_iov| for the joint proposal.
    // An IOV model (`n_kappa > 0`) must carry Ω_iov; enforce that here so the
    // per-subject joint path can rely on it without `Option` handling (and
    // `unwrap`) in the hot loop.
    let (omega_iov_inv, log_det_omega_iov) = if let Some(ref iov) = params.omega_iov {
        if !iov.log_det.is_finite() {
            return Err("IS: Ω_iov log-determinant is not finite — cannot evaluate κ prior".into());
        }
        (Some(iov.inv.clone()), Some(iov.log_det))
    } else if model.n_kappa > 0 {
        return Err(
            "IS: model declares κ (IOV) but params.omega_iov is missing — \
                    cannot build the joint (η, κ) proposal"
                .into(),
        );
    } else {
        (None, None)
    };

    if options.verbose {
        eprintln!(
            "Importance sampling: {} subjects, K={} per subject, t_{} proposal, seed={}",
            n_subjects, k_samples, nu, seed
        );
    }

    let per_subject: Vec<SubjectIsOutput> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |scratch, (i, subject)| {
            if crate::cancel::is_cancelled(cancel) {
                return SubjectIsOutput::cancelled(subject.id.clone());
            }
            let eta_hat = &eta_hats[i];
            let kap = kappas.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            let subj_seed = seed.wrapping_add(i as u64);

            if model.n_kappa > 0 && !kap.is_empty() {
                // Joint (eta, kappa) sampling for IOV models. Ω_iov is guaranteed
                // present by the up-front check in this function, so bind the
                // references once instead of unwrapping in the hot path.
                let omega_iov_inv = omega_iov_inv
                    .as_ref()
                    .expect("omega_iov present for IOV model (checked in run_importance_sampling)");
                let log_det_omega_iov = log_det_omega_iov
                    .expect("omega_iov present for IOV model (checked in run_importance_sampling)");

                let occ_groups = split_obs_by_occasion(subject);
                let n_occ = occ_groups.len();
                let n_iov = model.n_kappa;
                let n_b = n_eta + n_occ * n_iov;

                // Joint prior precision (block-diagonal: Ω_bsv⁻¹ plus one Ω_iov⁻¹
                // block per occasion). Built once and shared by the Hessian
                // assembly, the proposal, and the per-draw prior quadratic form.
                let omega_joint_inv =
                    build_joint_omega_inv(&omega_inv, omega_iov_inv, n_eta, n_iov, n_occ);
                let log_det_omega_joint = log_det_omega + n_occ as f64 * log_det_omega_iov;

                // Build joint posterior Hessian via FD
                let h_joint = compute_joint_posterior_hessian(
                    model,
                    subject,
                    &params.theta,
                    eta_hat,
                    kap,
                    &params.sigma.values,
                    &h_matrices[i],
                    &omega_joint_inv,
                    n_eta,
                    n_iov,
                    n_occ,
                    scratch,
                );

                // Build joint mode vector [eta_hat, kappa_1, ..., kappa_K]
                let mut mode_joint = vec![0.0_f64; n_b];
                for j in 0..n_eta {
                    mode_joint[j] = eta_hat[j];
                }
                for (k, kappa_occ) in kap.iter().enumerate() {
                    for ki in 0..n_iov {
                        mode_joint[n_eta + k * n_iov + ki] = kappa_occ[ki];
                    }
                }

                subject_is_estimate_joint(
                    model,
                    subject,
                    &params.theta,
                    &params.sigma.values,
                    &mode_joint,
                    &h_joint,
                    &omega_joint_inv,
                    log_det_omega_joint,
                    n_b,
                    n_eta,
                    n_iov,
                    n_occ,
                    k_samples,
                    nu,
                    subj_seed,
                    scratch,
                )
            } else {
                // Non-IOV path: eta-only sampling
                let jacobian = &h_matrices[i];
                let h_post = compute_posterior_hessian(
                    model,
                    subject,
                    &params.theta,
                    eta_hat,
                    &params.sigma.values,
                    jacobian,
                    &omega_inv,
                    n_eta,
                    scratch,
                );
                subject_is_estimate(
                    model,
                    subject,
                    &params.theta,
                    &params.sigma.values,
                    eta_hat,
                    &h_post,
                    &omega_inv,
                    log_det_omega,
                    n_eta,
                    k_samples,
                    nu,
                    subj_seed,
                    scratch,
                    1.0, // iscale: no adaptive scaling for IMP EONLY
                )
            }
        })
        .collect();

    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    // Reduce across subjects.
    let mut ll = 0.0_f64;
    let mut var_ll = 0.0_f64;
    let mut ess_fracs: Vec<f64> = Vec::with_capacity(n_subjects);
    let mut low_ess: Vec<(String, f64)> = Vec::new();
    for (i, out) in per_subject.iter().enumerate() {
        ll += out.log_marginal;
        var_ll += out.var_log_marginal;
        ess_fracs.push(out.ess_fraction);
        if out.ess_fraction < threshold {
            low_ess.push((population.subjects[i].id.clone(), out.ess_fraction));
        }
    }
    let minus2_ll = -2.0 * ll;
    // Var(−2 LL) = 4 · Var(LL) = 4 · Σᵢ Var(log p̂(yᵢ)).
    let mc_se = 2.0 * var_ll.max(0.0).sqrt();

    ess_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ess_min = *ess_fracs.first().unwrap_or(&0.0);
    let ess_median = if ess_fracs.is_empty() {
        0.0
    } else {
        let mid = ess_fracs.len() / 2;
        if ess_fracs.len().is_multiple_of(2) {
            0.5 * (ess_fracs[mid - 1] + ess_fracs[mid])
        } else {
            ess_fracs[mid]
        }
    };

    if options.verbose {
        eprintln!(
            "IS done. −2 log L = {:.4} ± {:.4} (ess_min/K = {:.3}, ess_med/K = {:.3}, low_ess = {})",
            minus2_ll,
            mc_se,
            ess_min,
            ess_median,
            low_ess.len()
        );
    }

    Ok(ImportanceSamplingResult {
        minus2_log_likelihood: minus2_ll,
        mc_standard_error: mc_se,
        low_ess_subjects: low_ess,
        n_samples: k_samples,
        proposal_df: nu,
        ess_min,
        ess_median,
        kappa_treatment,
    })
}

// ---------------------------------------------------------------------------
// Per-subject estimate
// ---------------------------------------------------------------------------

struct SubjectIsOutput {
    /// log p̂(yᵢ | θ) — the IS-estimated marginal log-likelihood for subject i.
    log_marginal: f64,
    /// Per-subject contribution to `Var(LL_IS) = Σᵢ Var(log p̂(yᵢ))`.
    /// Used to build the population-level MC SE.
    var_log_marginal: f64,
    /// Normalised effective sample size, ESS/K. 1.0 = ideal; near 0 = degenerate.
    ess_fraction: f64,
}

impl SubjectIsOutput {
    fn cancelled(_id: String) -> Self {
        Self {
            log_marginal: 0.0,
            var_log_marginal: 0.0,
            ess_fraction: 0.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn subject_is_estimate(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta_hat: &DVector<f64>,
    h: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    log_det_omega: f64,
    d: usize,
    k_samples: usize,
    nu: f64,
    seed: u64,
    scratch: &mut EventPkParams,
    iscale: f64,
) -> SubjectIsOutput {
    let mut rng = StdRng::seed_from_u64(seed);

    // Build proposal: t_ν(η̂, Σ) with Σ = (H + λI)⁻¹.
    let proposal = match build_proposal(h, omega_inv, d) {
        Some(p) => p,
        None => {
            // Even the omega-scale fallback failed (n_eta = 0?). Treat as a
            // degenerate proposal with all weight at the prior mode.
            return SubjectIsOutput {
                log_marginal: 0.0,
                var_log_marginal: 0.0,
                ess_fraction: 0.0,
            };
        }
    };

    let normal = StandardNormal;
    let chi_sq = ChiSquared::new(nu).expect("ChiSquared requires nu > 0; checked by caller");

    // Constant pieces of log q (Student-t) and log p_η (Gaussian) pulled out
    // of the per-sample loop. ISCALE adjusts the proposal: Σ_prop = iscale² · Σ.
    let half_d = 0.5 * d as f64;
    let iscale_log_adj = -(d as f64) * iscale.ln();
    let inv_iscale_sq = 1.0 / (iscale * iscale);
    let log_t_const = ln_gamma(0.5 * (nu + d as f64))
        - ln_gamma(0.5 * nu)
        - half_d * (nu * std::f64::consts::PI).ln()
        + 0.5 * proposal.log_det_inv_scale
        + iscale_log_adj;
    let log_p_eta_const = -half_d * TWO_PI.ln() - 0.5 * log_det_omega;

    // Preallocate every per-sample buffer once. With K typically in the
    // thousands per subject and n_subjects in the hundreds, repeated
    // allocation of `z`, `eta_sample`, `diff` and the implicit `DVector`
    // for the prior quadratic form would dominate the inner loop.
    let mut log_w: Vec<f64> = Vec::with_capacity(k_samples);
    let mut z = vec![0.0_f64; d];
    let mut eta_sample = vec![0.0_f64; d];
    let mut diff = vec![0.0_f64; d];

    for _ in 0..k_samples {
        // Draw z ~ N(0, I_d) and c ~ χ²_ν; build η = η̂ + iscale·sqrt(ν/c) · L_Σ z.
        for zi in z.iter_mut() {
            *zi = normal.sample(&mut rng);
        }
        let c: f64 = chi_sq.sample(&mut rng).max(1e-300);
        let scale = (nu / c).sqrt() * iscale;
        // L_Σ z via the precomputed factor.
        proposal.apply_l_sigma(&z, &mut eta_sample, scale);
        for (j, e) in eta_sample.iter_mut().enumerate() {
            *e += eta_hat[j];
        }

        let obs_nll = obs_nll_subject_into(model, subject, theta, sigma, &eta_sample, scratch);
        let log_p_y = -obs_nll;

        // log p(η | θ): multivariate-normal quadratic form `η' Ω⁻¹ η`,
        // computed allocation-free on slices. For d ≈ 2–10 the O(d²) double
        // loop is far cheaper than building a `DVector` and a fresh
        // `omega_inv * eta` allocation per sample.
        let mut quad_form = 0.0_f64;
        for i in 0..d {
            let mut row = 0.0_f64;
            for j in 0..d {
                row += omega_inv[(i, j)] * eta_sample[j];
            }
            quad_form += row * eta_sample[i];
        }
        let log_p_eta = log_p_eta_const - 0.5 * quad_form;

        // log q(η): multivariate t at (η̂, iscale²·Σ, ν).
        for (k, d_slot) in diff.iter_mut().enumerate() {
            *d_slot = eta_sample[k] - eta_hat[k];
        }
        let mahal = proposal.mahalanobis(&diff);
        let log_q = log_t_const - 0.5 * (nu + d as f64) * (1.0 + inv_iscale_sq * mahal / nu).ln();

        log_w.push(log_p_y + log_p_eta - log_q);
    }

    // logsumexp + ESS.
    let (lse, weights_norm) = logsumexp_with_normalised(&log_w);
    let log_marginal = lse - (k_samples as f64).ln();
    let ess = if weights_norm.is_empty() {
        0.0
    } else {
        let sum_sq: f64 = weights_norm.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    };
    let ess_fraction = ess / (k_samples as f64);

    // Asymptotic variance of log p̂(yᵢ) for a self-normalised IS estimator:
    //   Var(log p̂) ≈ (K · Σ wₖ² − 1) / K = (1/ESS_fraction − 1) / K
    // (Geweke 1989; equivalent to the standard "1/ESS − 1/K" relation.)
    let var_log_marginal = if ess_fraction > 0.0 {
        (1.0 / ess_fraction - 1.0) / (k_samples as f64)
    } else {
        // Degenerate — treat the per-subject estimate as having undefined SE.
        // Inflate by a finite-but-large number so the overall MC SE flags it
        // without producing a NaN that contaminates the sum.
        1.0
    };

    SubjectIsOutput {
        log_marginal,
        var_log_marginal,
        ess_fraction,
    }
}

// ---------------------------------------------------------------------------
// IMPMAP: MAP-centered draws with retained samples + weighted moments
// ---------------------------------------------------------------------------

/// Per-subject output of one IMPMAP E-step: the retained importance samples and
/// their normalized weights (consumed by the θ/σ M-step), the weighted posterior
/// moments (the Ω sufficient statistic and conditional mean), and the same
/// marginal-LL / ESS diagnostics the `Imp` kernel reports.
pub(crate) struct SubjectDraws {
    /// IS-estimated marginal log-likelihood `log p̂(yᵢ | θ)` (for the OFV trace).
    pub log_marginal: f64,
    /// Normalized effective sample size ESS/K (proposal-quality diagnostic).
    pub ess_fraction: f64,
    /// The `K` sampled η vectors (each length `d`).
    pub etas: Vec<Vec<f64>>,
    /// Self-normalized importance weights `w̃ᵢₖ` (length `K`, sums to 1).
    pub weights: Vec<f64>,
    /// Weighted posterior mean `Σₖ w̃ᵢₖ ηᵢₖ` (length `d`) — drives the closed-form
    /// mu-referencing θ shift.
    pub mean: Vec<f64>,
    /// Weighted second moment `Σₖ w̃ᵢₖ ηᵢₖ ηᵢₖᵀ` (d×d) — the per-subject Ω
    /// sufficient statistic.
    pub second_moment: DMatrix<f64>,
}

impl SubjectDraws {
    /// Cheap placeholder returned when a cancellation is observed mid-E-step, so
    /// the rayon task exits immediately instead of running the inner loop /
    /// importance sampling. The IMPMAP driver breaks out of the iteration right
    /// after the `par_iter` collect, so these benign values never feed a real
    /// M-step.
    pub(crate) fn cancelled(n_eta: usize) -> Self {
        SubjectDraws {
            log_marginal: 0.0,
            ess_fraction: 1.0,
            etas: Vec::new(),
            weights: Vec::new(),
            mean: vec![0.0; n_eta],
            second_moment: DMatrix::zeros(n_eta, n_eta),
        }
    }
}

/// Draw `K` importance samples for one subject from a proposal centered at the
/// conditional mode `η̂` with first-order-variance scale `Σ = (H + λI)⁻¹`, and
/// return the retained samples, self-normalized weights, and weighted second
/// moment for the IMPMAP M-step.
///
/// `nu = f64::INFINITY` selects a multivariate-normal proposal (NONMEM IMPMAP
/// default); a finite `nu ≥ 1` selects a multivariate Student-t. The marginal-LL
/// and weight math is otherwise identical to [`subject_is_estimate`].
///
/// `iscale` scales the proposal standard deviation: Σ_prop = iscale² · H⁻¹.
/// Use 1.0 for unscaled (default). NONMEM ISCALE range is typically [0.1, 10.0].
///
/// `use_sobol`: when `true` **and** the proposal is MVN (ν = ∞), replace
/// pseudo-random N(0,I) draws with Sobol quasi-random sequences
/// (Cranley-Patterson randomized).  Falls back to pseudo-random for Student-t.
#[allow(clippy::too_many_arguments)]
pub(crate) fn subject_is_draws(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta_hat: &DVector<f64>,
    h: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    log_det_omega: f64,
    d: usize,
    k_samples: usize,
    nu: f64,
    seed: u64,
    scratch: &mut EventPkParams,
    iscale: f64,
    use_sobol: bool,
) -> SubjectDraws {
    let mvn = !nu.is_finite();
    let mut rng = StdRng::seed_from_u64(seed);

    let proposal = match build_proposal(h, omega_inv, d) {
        Some(p) => p,
        None => {
            return SubjectDraws {
                log_marginal: 0.0,
                ess_fraction: 0.0,
                etas: Vec::new(),
                weights: Vec::new(),
                mean: vec![0.0; d],
                second_moment: DMatrix::zeros(d, d),
            };
        }
    };

    let normal = StandardNormal;
    // Student-t scale mixing variable; unused for the MVN branch.
    let chi_sq = if mvn {
        None
    } else {
        Some(ChiSquared::new(nu).expect("ChiSquared requires nu > 0; checked by caller"))
    };

    let half_d = 0.5 * d as f64;
    let log_p_eta_const = -half_d * TWO_PI.ln() - 0.5 * log_det_omega;

    // Constant term of log q(η) with ISCALE adjustment.
    // MVN: −d/2·log(2π) + ½log|Σ⁻¹| − d·log(iscale)
    //   where the −d·log(iscale) accounts for det(s²Σ) = s^{2d}·det(Σ).
    // Student-t: Γ-ratio + ½log|Σ⁻¹| − d·log(iscale) − d/2·log(ν·π).
    let iscale_log_adj = -(d as f64) * iscale.ln();
    let inv_iscale_sq = 1.0 / (iscale * iscale);
    let log_q_const = if mvn {
        -half_d * TWO_PI.ln() + 0.5 * proposal.log_det_inv_scale + iscale_log_adj
    } else {
        ln_gamma(0.5 * (nu + d as f64)) - ln_gamma(0.5 * nu)
            + 0.5 * proposal.log_det_inv_scale
            + iscale_log_adj
            - half_d * (nu * std::f64::consts::PI).ln()
    };

    let mut log_w: Vec<f64> = Vec::with_capacity(k_samples);
    let mut etas: Vec<Vec<f64>> = Vec::with_capacity(k_samples);
    let mut z = vec![0.0_f64; d];
    let mut diff = vec![0.0_f64; d];

    // Pre-generate Sobol quasi-random draws if requested and MVN.
    let sobol_draws = if use_sobol && mvn {
        Some(sobol_normal_draws(d, k_samples, seed))
    } else {
        None
    };

    for sample_idx in 0..k_samples {
        if let Some(ref qr) = sobol_draws {
            // Sobol quasi-random N(0,I) draws (MVN only)
            z.copy_from_slice(&qr[sample_idx]);
        } else {
            for zi in z.iter_mut() {
                *zi = normal.sample(&mut rng);
            }
        }
        let scale = match &chi_sq {
            Some(c) => {
                let cc: f64 = c.sample(&mut rng).max(1e-300);
                (nu / cc).sqrt() * iscale
            }
            None => iscale,
        };
        let mut eta_sample = vec![0.0_f64; d];
        proposal.apply_l_sigma(&z, &mut eta_sample, scale);
        for (j, e) in eta_sample.iter_mut().enumerate() {
            *e += eta_hat[j];
        }

        let obs_nll = obs_nll_subject_into(model, subject, theta, sigma, &eta_sample, scratch);
        let log_p_y = -obs_nll;

        let mut quad_form = 0.0_f64;
        for i in 0..d {
            let mut row = 0.0_f64;
            for j in 0..d {
                row += omega_inv[(i, j)] * eta_sample[j];
            }
            quad_form += row * eta_sample[i];
        }
        let log_p_eta = log_p_eta_const - 0.5 * quad_form;

        for (k, d_slot) in diff.iter_mut().enumerate() {
            *d_slot = eta_sample[k] - eta_hat[k];
        }
        let mahal = proposal.mahalanobis(&diff);
        let log_q = if mvn {
            log_q_const - 0.5 * inv_iscale_sq * mahal
        } else {
            log_q_const - 0.5 * (nu + d as f64) * (1.0 + inv_iscale_sq * mahal / nu).ln()
        };

        log_w.push(log_p_y + log_p_eta - log_q);
        etas.push(eta_sample);
    }

    let (lse, weights) = logsumexp_with_normalised(&log_w);
    let log_marginal = lse - (k_samples as f64).ln();
    let ess = {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    };
    let ess_fraction = ess / (k_samples as f64);

    // Weighted first and second moments Σₖ w̃ₖ ηₖ and Σₖ w̃ₖ ηₖ ηₖᵀ.
    let mut mean = vec![0.0_f64; d];
    let mut second_moment = DMatrix::<f64>::zeros(d, d);
    for (w, eta) in weights.iter().zip(etas.iter()) {
        for i in 0..d {
            mean[i] += w * eta[i];
            for j in 0..d {
                second_moment[(i, j)] += w * eta[i] * eta[j];
            }
        }
    }

    SubjectDraws {
        log_marginal,
        ess_fraction,
        etas,
        weights,
        mean,
        second_moment,
    }
}

/// Find the optimal ISCALE for a subject via pilot draws.
///
/// Tries a grid of log-spaced scale factors in `[iscale_min, iscale_max]`
/// using `n_pilot` draws each, and returns the scale that maximises ESS.
/// Returns 1.0 if ISCALE is disabled (min >= max or both == 1.0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_optimal_iscale(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta_hat: &DVector<f64>,
    h: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    log_det_omega: f64,
    d: usize,
    nu: f64,
    seed: u64,
    scratch: &mut EventPkParams,
    iscale_min: f64,
    iscale_max: f64,
) -> f64 {
    if iscale_min >= iscale_max || (iscale_min == 1.0 && iscale_max == 1.0) {
        return 1.0;
    }
    // Grid of 7 log-spaced scale factors from iscale_min to iscale_max
    let n_grid = 7;
    let n_pilot = 50;
    let log_min = iscale_min.ln();
    let log_max = iscale_max.ln();
    let mut best_scale = 1.0_f64;
    let mut best_ess = f64::NEG_INFINITY;

    for g in 0..n_grid {
        let frac = g as f64 / (n_grid - 1) as f64;
        let scale = (log_min + frac * (log_max - log_min)).exp();
        // Use a different seed per scale to avoid correlations
        let pilot_seed = seed
            .wrapping_add(0x4953_4341_4C45_0000u64)
            .wrapping_add(g as u64);
        let draws = subject_is_draws(
            model,
            subject,
            theta,
            sigma,
            eta_hat,
            h,
            omega_inv,
            log_det_omega,
            d,
            n_pilot,
            nu,
            pilot_seed,
            scratch,
            scale,
            false, // pilot draws don't need Sobol
        );
        if draws.ess_fraction > best_ess {
            best_ess = draws.ess_fraction;
            best_scale = scale;
        }
    }
    best_scale
}

// ---------------------------------------------------------------------------
// Joint (eta, kappa) sampling for IOV models
// ---------------------------------------------------------------------------

/// Build the joint posterior Hessian for (eta, kappa) via finite differences.
///
/// The joint vector is b = [eta, kappa_1, ..., kappa_K] where K is n_occasions.
/// The Hessian is H_post = J' R^{-1} J + Omega_joint^{-1} where J is the
/// Jacobian of predictions w.r.t. b.
#[allow(clippy::too_many_arguments)]
fn compute_joint_posterior_hessian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    kappas: &[DVector<f64>],
    sigma: &[f64],
    jacobian_eta: &DMatrix<f64>,
    omega_joint_inv: &DMatrix<f64>,
    n_eta: usize,
    n_iov: usize,
    n_occ: usize,
    _scratch: &mut EventPkParams,
) -> DMatrix<f64> {
    let n_obs = subject.observations.len();
    let n_b = n_eta + n_occ * n_iov;

    if n_obs == 0 {
        // No observations: posterior = prior
        return omega_joint_inv.clone();
    }

    // Compute predictions at the joint mode using predict_iov for proper cross-occasion carryover
    let kappa_slices: Vec<Vec<f64>> = kappas.iter().map(|k| k.as_slice().to_vec()).collect();
    let ipreds = predict_iov(model, subject, theta, eta_hat.as_slice(), &kappa_slices);

    // Compute residual variance with FREM overrides
    let mut r_diag = compute_r_diag(&model.error_spec, &ipreds, &subject.obs_cmts, sigma);
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma,
    );
    if let Some(ref overrides) = frem_ov {
        crate::stats::likelihood::apply_frem_r_overrides(&mut r_diag, overrides);
    }

    // Build the full Jacobian via FD for kappa columns
    // eta columns come from jacobian_eta (already computed by inner loop)
    let mut j_full = DMatrix::zeros(n_obs, n_b);

    // Copy eta columns
    for j in 0..n_obs {
        for c in 0..n_eta {
            j_full[(j, c)] = jacobian_eta[(j, c)];
        }
    }

    // FD for kappa columns using predict_iov for proper cross-occasion carryover
    const EPS: f64 = 1e-6;
    let mut kap_perturbed: Vec<Vec<f64>> = kappas.iter().map(|k| k.as_slice().to_vec()).collect();

    for k in 0..n_occ {
        let col_base = n_eta + k * n_iov;
        for ki in 0..n_iov {
            let orig = kap_perturbed[k][ki];
            let step = EPS * (1.0 + orig.abs());

            // Forward perturbation
            kap_perturbed[k][ki] = orig + step;
            let preds_plus = predict_iov(model, subject, theta, eta_hat.as_slice(), &kap_perturbed);

            // Backward perturbation
            kap_perturbed[k][ki] = orig - step;
            let preds_minus =
                predict_iov(model, subject, theta, eta_hat.as_slice(), &kap_perturbed);

            // Restore
            kap_perturbed[k][ki] = orig;

            let inv_2step = 1.0 / (2.0 * step);
            for j in 0..n_obs {
                j_full[(j, col_base + ki)] = (preds_plus[j] - preds_minus[j]) * inv_2step;
            }
        }
    }

    // Build H_post = J' R^{-1} J + Omega_joint^{-1}
    let mut h_post = omega_joint_inv.clone();

    for j in 0..n_obs {
        let rj = r_diag[j].max(1e-12);
        for a in 0..n_b {
            let ja = j_full[(j, a)];
            for b in 0..n_b {
                h_post[(a, b)] += ja * j_full[(j, b)] / rj;
            }
        }
    }

    h_post
}

/// Build the joint prior precision matrix (block-diagonal: Omega_bsv^{-1} + K copies of Omega_iov^{-1}).
fn build_joint_omega_inv(
    omega_inv: &DMatrix<f64>,
    omega_iov_inv: &DMatrix<f64>,
    n_eta: usize,
    n_iov: usize,
    n_occ: usize,
) -> DMatrix<f64> {
    let n_b = n_eta + n_occ * n_iov;
    let mut m = DMatrix::zeros(n_b, n_b);

    // BSV block
    for i in 0..n_eta {
        for j in 0..n_eta {
            m[(i, j)] = omega_inv[(i, j)];
        }
    }

    // K copies of IOV block
    for k in 0..n_occ {
        let offset = n_eta + k * n_iov;
        for i in 0..n_iov {
            for j in 0..n_iov {
                m[(offset + i, offset + j)] = omega_iov_inv[(i, j)];
            }
        }
    }

    m
}

/// Joint (eta, kappa) IS estimate for IOV models.
#[allow(clippy::too_many_arguments)]
fn subject_is_estimate_joint(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    mode_joint: &[f64],
    h_joint: &DMatrix<f64>,
    omega_joint_inv: &DMatrix<f64>,
    log_det_omega_joint: f64,
    n_b: usize,
    n_eta: usize,
    n_iov: usize,
    n_occ: usize,
    k_samples: usize,
    nu: f64,
    seed: u64,
    _scratch: &mut EventPkParams,
) -> SubjectIsOutput {
    let mut rng = StdRng::seed_from_u64(seed);

    // Build joint proposal
    let proposal = match build_proposal(h_joint, omega_joint_inv, n_b) {
        Some(p) => p,
        None => {
            return SubjectIsOutput {
                log_marginal: 0.0,
                var_log_marginal: 0.0,
                ess_fraction: 0.0,
            };
        }
    };

    let normal = StandardNormal;
    let chi_sq = ChiSquared::new(nu).expect("ChiSquared requires nu > 0; checked by caller");

    let half_d = 0.5 * n_b as f64;
    let log_t_const = ln_gamma(0.5 * (nu + n_b as f64))
        - ln_gamma(0.5 * nu)
        - half_d * (nu * std::f64::consts::PI).ln()
        + 0.5 * proposal.log_det_inv_scale;
    let log_p_joint_const = -half_d * TWO_PI.ln() - 0.5 * log_det_omega_joint;

    let mut log_w: Vec<f64> = Vec::with_capacity(k_samples);
    let mut z = vec![0.0_f64; n_b];
    let mut sample_joint = vec![0.0_f64; n_b];
    let mut diff = vec![0.0_f64; n_b];
    // Reused across draws: overwriting in place avoids K allocations of the
    // outer `Vec` plus its per-occasion κ vectors (a hot-loop allocator cost
    // at K in the thousands).
    let mut kappas_sampled: Vec<Vec<f64>> = (0..n_occ).map(|_| vec![0.0_f64; n_iov]).collect();

    for _ in 0..k_samples {
        // Draw from joint proposal
        for zi in z.iter_mut() {
            *zi = normal.sample(&mut rng);
        }
        let c: f64 = chi_sq.sample(&mut rng).max(1e-300);
        let scale = (nu / c).sqrt();
        proposal.apply_l_sigma(&z, &mut sample_joint, scale);
        for (j, e) in sample_joint.iter_mut().enumerate() {
            *e += mode_joint[j];
        }

        // Split into eta and kappa parts (κ buffer reused across draws).
        let eta_sample = &sample_joint[..n_eta];
        for (k, kappa_occ) in kappas_sampled.iter_mut().enumerate() {
            let start = n_eta + k * n_iov;
            kappa_occ.copy_from_slice(&sample_joint[start..start + n_iov]);
        }

        // Compute obs NLL with sampled eta and kappa using predict_iov
        let ipreds = predict_iov(model, subject, theta, eta_sample, &kappas_sampled);

        let m3 = matches!(model.bloq_method, BloqMethod::M3);
        let mut obs_nll = 0.0_f64;
        for (j, (&y, &f)) in subject.observations.iter().zip(ipreds.iter()).enumerate() {
            let f = f.max(1e-12);
            let v = model
                .residual_variance_at(subject.obs_cmts[j], f, sigma)
                .max(1e-12);
            if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                let z = (y - f) / v.sqrt();
                obs_nll += -log_normal_cdf(z);
            } else {
                obs_nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
            }
        }
        let log_p_y = -obs_nll;

        // Compute joint prior: eta' Omega^{-1} eta + sum_k kappa_k' Omega_iov^{-1} kappa_k
        let mut quad_form = 0.0_f64;
        for i in 0..n_b {
            let mut row = 0.0_f64;
            for j in 0..n_b {
                row += omega_joint_inv[(i, j)] * sample_joint[j];
            }
            quad_form += row * sample_joint[i];
        }
        let log_p_joint = log_p_joint_const - 0.5 * quad_form;

        // Compute proposal log-density
        for (k, d_slot) in diff.iter_mut().enumerate() {
            *d_slot = sample_joint[k] - mode_joint[k];
        }
        let mahal = proposal.mahalanobis(&diff);
        let log_q = log_t_const - 0.5 * (nu + n_b as f64) * (1.0 + mahal / nu).ln();

        log_w.push(log_p_y + log_p_joint - log_q);
    }

    // logsumexp + ESS
    let (lse, weights_norm) = logsumexp_with_normalised(&log_w);
    let log_marginal = lse - (k_samples as f64).ln();
    let ess = if weights_norm.is_empty() {
        0.0
    } else {
        let sum_sq: f64 = weights_norm.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    };
    let ess_fraction = ess / (k_samples as f64);

    let var_log_marginal = if ess_fraction > 0.0 {
        (1.0 / ess_fraction - 1.0) / (k_samples as f64)
    } else {
        1.0
    };

    SubjectIsOutput {
        log_marginal,
        var_log_marginal,
        ess_fraction,
    }
}

// ---------------------------------------------------------------------------
// Proposal construction (regularised Hessian, Cholesky factors)
// ---------------------------------------------------------------------------

struct Proposal {
    /// Lower-triangular L such that L L' = H_reg = Σ⁻¹.
    /// Used both to apply Σ^{1/2} when sampling (L'⁻¹ z) and to evaluate the
    /// Mahalanobis term (‖L'·diff‖²) when scoring q(η).
    chol_h: DMatrix<f64>,
    /// log|H_reg| = 2 · Σ log L_ii — used for the log|Σ| = −log|H_reg| piece
    /// of the Student-t log-density.
    log_det_inv_scale: f64,
    d: usize,
}

impl Proposal {
    /// Apply `scale · L_Σ z` into `out`, where `L_Σ` is the Cholesky factor of
    /// Σ = H⁻¹. Implementation: `L_Σ = L^{-T}` for the L from `H = L L^T`, so
    /// `L_Σ z = L^{-T} z` — one back-substitution.
    fn apply_l_sigma(&self, z: &[f64], out: &mut [f64], scale: f64) {
        // Back-solve L^T x = z for x (i.e. `out`).
        // L is lower-triangular; L^T is upper-triangular.
        let l = &self.chol_h;
        for i in (0..self.d).rev() {
            let mut s = z[i];
            for j in (i + 1)..self.d {
                s -= l[(j, i)] * out[j];
            }
            out[i] = s / l[(i, i)];
        }
        if scale != 1.0 {
            for x in out.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Compute `diff' H_reg diff` via `‖L' diff‖²`.
    fn mahalanobis(&self, diff: &[f64]) -> f64 {
        let l = &self.chol_h;
        let mut sum = 0.0;
        for i in 0..self.d {
            let mut s = 0.0;
            for j in i..self.d {
                s += l[(j, i)] * diff[j];
            }
            sum += s * s;
        }
        sum
    }
}

/// Build the IS proposal scale matrix from a per-subject inner-loop Hessian.
///
/// Strategy: Cholesky(H + λI), with λ = max(1e−6 · trace(H)/d, 1e−10). If H
/// is so degenerate that even the jittered matrix isn't positive-definite,
/// fall back to Σ = Ω (the prior covariance) — a broad proposal that won't
/// give a sharp likelihood estimate but stays well-defined.
///
/// Returns `None` only when `d == 0`.
fn build_proposal(h: &DMatrix<f64>, omega_inv: &DMatrix<f64>, d: usize) -> Option<Proposal> {
    if d == 0 {
        return None;
    }
    debug_assert_eq!(h.nrows(), d);
    debug_assert_eq!(h.ncols(), d);

    let trace = (0..d).map(|i| h[(i, i)]).sum::<f64>();
    let lambda = (1e-6 * trace / d as f64).max(1e-10);
    let mut h_reg = h.clone();
    for i in 0..d {
        h_reg[(i, i)] += lambda;
    }
    if let Some(chol) = h_reg.clone().cholesky() {
        let l = chol.l();
        let log_det = 2.0 * (0..d).map(|i| l[(i, i)].ln()).sum::<f64>();
        return Some(Proposal {
            chol_h: l,
            log_det_inv_scale: log_det,
            d,
        });
    }

    // Fallback: Σ = Ω (proposal scale = prior covariance). To keep the same
    // (L L' = Σ⁻¹) interface we Cholesky-factor Ω⁻¹ instead.
    let omega_inv_chol = omega_inv.clone().cholesky()?;
    let l = omega_inv_chol.l();
    let log_det = 2.0 * (0..d).map(|i| l[(i, i)].ln()).sum::<f64>();
    Some(Proposal {
        chol_h: l,
        log_det_inv_scale: log_det,
        d,
    })
}

// ---------------------------------------------------------------------------
// Numerical helpers
// ---------------------------------------------------------------------------

/// Numerically stable `log Σ exp(xᵢ)` plus the normalised weights `wᵢ`.
fn logsumexp_with_normalised(xs: &[f64]) -> (f64, Vec<f64>) {
    if xs.is_empty() {
        return (f64::NEG_INFINITY, Vec::new());
    }
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return (m, vec![0.0; xs.len()]);
    }
    let mut sum = 0.0;
    let mut shifted: Vec<f64> = Vec::with_capacity(xs.len());
    for &x in xs {
        let s = (x - m).exp();
        shifted.push(s);
        sum += s;
    }
    let lse = m + sum.ln();
    let weights: Vec<f64> = if sum > 0.0 {
        shifted.iter().map(|&s| s / sum).collect()
    } else {
        vec![0.0; xs.len()]
    };
    (lse, weights)
}

/// Sheiner–Beal posterior-Hessian approximation at η̂.
///
/// `H_post ≈ J' R⁻¹ J + Ω⁻¹` where J is the Jacobian `df/dη` (the misleadingly-
/// named `h_matrix` stored on `OuterResult`) and R is the diagonal residual
/// variance evaluated at η̂. This is the matrix the IS proposal scales against
/// — it's the Laplace covariance's inverse.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_posterior_hessian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    sigma: &[f64],
    jacobian: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    n_eta: usize,
    scratch: &mut EventPkParams,
) -> DMatrix<f64> {
    let n_obs = subject.observations.len();
    if n_obs == 0 {
        // Subject has no observations — posterior == prior; Hessian = Ω⁻¹.
        return omega_inv.clone();
    }
    let ipreds =
        compute_predictions_with_tv_into(model, subject, theta, eta_hat.as_slice(), scratch);
    let mut r_diag = compute_r_diag(&model.error_spec, &ipreds, &subject.obs_cmts, sigma);
    // Apply FREM R-diagonal overrides: covariate pseudo-observations use EPSCOV²
    // instead of the PK error model variance.
    let frem_ov = crate::stats::likelihood::build_frem_r_override(
        model.frem_config.as_ref(),
        &subject.fremtype,
        sigma,
    );
    if let Some(ref overrides) = frem_ov {
        crate::stats::likelihood::apply_frem_r_overrides(&mut r_diag, overrides);
    }
    let mut h_post = omega_inv.clone();
    for j in 0..n_obs {
        let rj = r_diag[j].max(1e-12);
        for a in 0..n_eta {
            let ja = jacobian[(j, a)];
            for b in 0..n_eta {
                h_post[(a, b)] += ja * jacobian[(j, b)] / rj;
            }
        }
    }
    h_post
}

// ---------------------------------------------------------------------------
// Tier 1 unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_draws_cancelled_is_benign_and_correctly_shaped() {
        // The placeholder returned when the IMPMAP E-step observes a cancel:
        // zero samples/weights so it can't bias an (already-skipped) M-step, a
        // zero second moment of the right dimension, and an ESS that does not
        // count as low-ESS.
        let n_eta = 3;
        let d = SubjectDraws::cancelled(n_eta);
        assert_eq!(d.log_marginal, 0.0);
        assert_eq!(d.ess_fraction, 1.0);
        assert!(d.etas.is_empty());
        assert!(d.weights.is_empty());
        assert_eq!(d.mean, vec![0.0; n_eta]);
        assert_eq!(d.second_moment.nrows(), n_eta);
        assert_eq!(d.second_moment.ncols(), n_eta);
        assert!(d.second_moment.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn logsumexp_handles_extreme_spread() {
        // log[exp(1000) + exp(1001)] = 1001 + log(1 + exp(-1)) ≈ 1001.3133
        let (lse, w) = logsumexp_with_normalised(&[1000.0, 1001.0]);
        assert!((lse - (1001.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-10);
        // Normalised weights should sum to 1 and reflect the relative
        // log-weight: w[1] / w[0] = exp(1) ≈ 2.718.
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        let ratio = w[1] / w[0];
        assert!((ratio - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn logsumexp_returns_neginf_on_empty_input() {
        let (lse, w) = logsumexp_with_normalised(&[]);
        assert_eq!(lse, f64::NEG_INFINITY);
        assert!(w.is_empty());
    }

    #[test]
    fn logsumexp_handles_all_neginf() {
        let (lse, w) = logsumexp_with_normalised(&[f64::NEG_INFINITY; 3]);
        assert_eq!(lse, f64::NEG_INFINITY);
        assert_eq!(w, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_proposal_handles_well_conditioned_h() {
        // H = diag(2, 4) — well conditioned.
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 4.0]);
        let omega_inv = DMatrix::identity(2, 2);
        let p = build_proposal(&h, &omega_inv, 2).unwrap();
        // Σ = H⁻¹ = diag(0.5, 0.25), so log|Σ⁻¹| ≈ log 8.
        // (λ jitter is ~1e-6·trace/d = 3e-6 — negligible at this precision.)
        assert!((p.log_det_inv_scale - 8.0_f64.ln()).abs() < 1e-4);
    }

    #[test]
    fn build_proposal_falls_back_when_h_is_singular() {
        // H = zero matrix — not positive-definite even with default jitter.
        // Fallback uses Ω⁻¹'s Cholesky.
        let h = DMatrix::zeros(2, 2);
        let omega_inv = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let p = build_proposal(&h, &omega_inv, 2);
        assert!(p.is_some(), "fallback to Ω-scale proposal should succeed");
    }

    #[test]
    fn proposal_mahalanobis_matches_direct_quadratic_form() {
        // H = [[4, 1], [1, 3]], diff = [0.5, -0.2].
        // diff' H diff = 4·0.25 + 2·1·0.5·(-0.2) + 3·0.04 = 1.0 − 0.2 + 0.12 = 0.92.
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let omega_inv = DMatrix::identity(2, 2);
        let p = build_proposal(&h, &omega_inv, 2).unwrap();
        let diff = vec![0.5, -0.2];
        let m = p.mahalanobis(&diff);
        // Allow a touch of slack for the jitter.
        assert!((m - 0.92).abs() < 1e-3, "mahalanobis = {} vs 0.92", m);
    }

    #[test]
    fn proposal_apply_l_sigma_round_trips_with_mahalanobis() {
        // For x = L_Σ z, x' H x should equal z' L^{-1} H L^{-T} z = z' z = ||z||².
        // Set scale = 1 so the t expansion factor doesn't enter.
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let omega_inv = DMatrix::identity(2, 2);
        let p = build_proposal(&h, &omega_inv, 2).unwrap();
        let z = vec![0.7, -0.4];
        let mut x = vec![0.0, 0.0];
        p.apply_l_sigma(&z, &mut x, 1.0);
        let m = p.mahalanobis(&x);
        let z_sq: f64 = z.iter().map(|v| v * v).sum();
        // Jitter tilts this slightly; loose tolerance is fine here.
        assert!((m - z_sq).abs() < 1e-3, "m={} vs ||z||²={}", m, z_sq);
    }

    #[test]
    fn build_joint_omega_inv_block_diagonal_structure() {
        // 1 BSV param, 2 IOV params, 3 occasions → 7×7 matrix.
        // BSV block: [[2.0]], IOV block: [[3.0, 0.5], [0.5, 4.0]]
        let omega_inv = DMatrix::from_row_slice(1, 1, &[2.0]);
        let omega_iov_inv = DMatrix::from_row_slice(2, 2, &[3.0, 0.5, 0.5, 4.0]);
        let m = build_joint_omega_inv(&omega_inv, &omega_iov_inv, 1, 2, 3);

        assert_eq!(m.nrows(), 7);
        assert_eq!(m.ncols(), 7);

        // BSV block (top-left 1×1)
        assert_eq!(m[(0, 0)], 2.0);

        // Three IOV blocks on the diagonal at offsets 1, 3, 5
        for occ in 0..3 {
            let off = 1 + occ * 2;
            assert_eq!(m[(off, off)], 3.0, "occ {occ} diag[0,0]");
            assert_eq!(m[(off, off + 1)], 0.5, "occ {occ} off-diag");
            assert_eq!(m[(off + 1, off)], 0.5, "occ {occ} off-diag sym");
            assert_eq!(m[(off + 1, off + 1)], 4.0, "occ {occ} diag[1,1]");
        }

        // Cross-block entries must be zero
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(1, 3)], 0.0);
        assert_eq!(m[(3, 5)], 0.0);
    }

    #[test]
    fn weights_invariant_under_log_constant_shift() {
        // Shifting every log-weight by the same constant must not change the
        // normalised weights (and hence ESS).
        let xs = vec![0.1, -0.5, 2.3, -1.2, 0.8];
        let (_, w1) = logsumexp_with_normalised(&xs);
        let shifted: Vec<f64> = xs.iter().map(|x| x + 17.4).collect();
        let (_, w2) = logsumexp_with_normalised(&shifted);
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// Linear-Gaussian sanity check: when the model is exactly Gaussian and
    /// the proposal is correctly scaled, the IS estimate of log Σ wₖ should
    /// concentrate around the analytic marginal. We don't drive a full PK
    /// model here — instead we exercise the IS-weight bookkeeping on a
    /// known integrand by injecting hand-built log-weights through
    /// logsumexp + ESS, and verify the per-subject SE formula matches the
    /// known closed form for equal weights.
    #[test]
    fn equal_weights_imply_full_ess_and_zero_variance() {
        // K equal log-weights → wₖ = 1/K, ESS = K, variance estimate = 0.
        let k = 100;
        let log_w = vec![0.0_f64; k];
        let (lse, w) = logsumexp_with_normalised(&log_w);
        let log_marginal = lse - (k as f64).ln();
        // Σ exp(0) = K, so log Σ = log K, log_marginal = 0.
        assert!(log_marginal.abs() < 1e-12);
        let ess: f64 = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
        assert!((ess - k as f64).abs() < 1e-10);
        let ess_fraction = ess / (k as f64);
        let var = (1.0 / ess_fraction - 1.0) / (k as f64);
        assert!(var.abs() < 1e-12);
    }

    #[test]
    fn inv_normal_cdf_known_quantiles() {
        // Φ⁻¹(0.5) = 0
        assert!(inv_normal_cdf(0.5).abs() < 1e-8);
        // Φ⁻¹(0.975) ≈ 1.96
        let q975 = inv_normal_cdf(0.975);
        assert!((q975 - 1.96).abs() < 0.05, "Φ⁻¹(0.975) = {q975}");
        // Φ⁻¹(0.025) ≈ -1.96
        let q025 = inv_normal_cdf(0.025);
        assert!((q025 + 1.96).abs() < 0.05, "Φ⁻¹(0.025) = {q025}");
        // Φ⁻¹(0.84) ≈ 1.0
        let q84 = inv_normal_cdf(0.84);
        assert!((q84 - 1.0).abs() < 0.05, "Φ⁻¹(0.84) = {q84}");
    }

    #[test]
    fn sobol_draws_have_correct_shape_and_near_zero_mean() {
        let d = 5;
        let k = 1000;
        let draws = sobol_normal_draws(d, k, 42);
        assert_eq!(draws.len(), k);
        assert_eq!(draws[0].len(), d);
        // Mean of each dimension should be near zero for a large sample
        for dim in 0..d {
            let mean: f64 = draws.iter().map(|v| v[dim]).sum::<f64>() / k as f64;
            assert!(mean.abs() < 0.15, "dim {dim} mean = {mean}, expected ~0");
        }
    }
}
