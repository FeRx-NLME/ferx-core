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
//! from `options.imp_seed.wrapping_add(i as u64)` so the result is
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
use crate::stats::likelihood::{iov_occasion_groups, m3_logcdf, obs_nll_subject_into};
use crate::stats::residual_error::compute_r_diag;
use crate::stats::special::ln_gamma;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::RngExt;
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
    let shift: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();

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
    // `iov_occasion_groups(subject).len()` — must agree, or the fill loop
    // would index out of bounds (κ too long) or silently leave occasions at
    // κ = 0 (κ too short). Verify once up front so the parallel loop can index
    // freely. Subjects with no κ EBEs fall through to the η-only path.
    if model.n_kappa > 0 {
        for (i, subject) in population.subjects.iter().enumerate() {
            let kap_len = kappas.get(i).map(|v| v.len()).unwrap_or(0);
            if kap_len == 0 {
                continue;
            }
            let n_occ = iov_occasion_groups(subject).len();
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
    let k_samples = options.imp_samples;
    let nu = options.imp_proposal_df;
    let seed = options.imp_seed.unwrap_or(42);
    let threshold = options.imp_low_ess_threshold;
    let defensive_alpha = options.imp_defensive_alpha;
    let cancel = &options.cancel;

    if k_samples < 2 {
        return Err(format!("IS: imp_samples must be >= 2, got {}", k_samples));
    }
    if nu < 1.0 {
        return Err(format!("IS: imp_proposal_df must be >= 1.0, got {}", nu));
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

    // Build the defensive-mixture broad component q_broad = N(0, Ω) once and
    // share it read-only across the parallel subject loop (issue #528). It
    // depends only on Ω, so a per-subject rebuild would redundantly re-Cholesky
    // the same Ω⁻¹. `None` when the mixture is inactive (`alpha == 0`). The FREM
    // Rao-Blackwell path builds its own (conditional-prior) component per subject.
    let defensive_mixture = DefensiveMixture::new(&omega_inv, n_eta, defensive_alpha);

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

    // FREM: Rao-Blackwellise (integrate covariate etas, sample only PK etas) when
    // a clean PK/cov partition exists. `None` → full-dimensional IS (unchanged).
    let frem_rb: Option<(Vec<usize>, Vec<usize>)> = if !options.frem_rao_blackwell {
        None
    } else {
        model
            .frem_config
            .as_ref()
            .filter(|_| model.n_kappa == 0)
            .map(|fc| frem_pk_cov_partition(fc, n_eta))
            .filter(|(pk, cov)| !pk.is_empty() && !cov.is_empty())
    };

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

                let occ_groups = iov_occasion_groups(subject);
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
                // FREM Rao-Blackwellised estimate (low-dim PK IS). Reuse the
                // draws routine and keep only its marginal-LL / ESS diagnostics.
                if let Some((ref pk_idx, ref cov_idx)) = frem_rb {
                    if let Some(fc) = model.frem_config.as_ref() {
                        if let Some((sampled, observed, d)) =
                            subject_frem_partition(subject, &params.theta, fc, pk_idx, cov_idx)
                        {
                            if let Some(rb) = subject_is_draws_frem_rb(
                                model,
                                subject,
                                &params.theta,
                                &params.sigma.values,
                                eta_hat,
                                &h_post,
                                &omega_inv,
                                &params.omega.matrix,
                                &sampled,
                                &observed,
                                &d,
                                n_eta,
                                k_samples,
                                nu,
                                subj_seed,
                                scratch,
                                1.0,
                                false,
                                defensive_alpha,
                            ) {
                                let ess_fraction = rb.ess_fraction;
                                let var_log_marginal = if ess_fraction > 0.0 {
                                    (1.0 / ess_fraction - 1.0) / (k_samples as f64)
                                } else {
                                    1.0
                                };
                                return SubjectIsOutput {
                                    log_marginal: rb.log_marginal,
                                    var_log_marginal,
                                    ess_fraction,
                                };
                            }
                        }
                    }
                }
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
                    defensive_mixture.as_ref(),
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
    mixture: Option<&DefensiveMixture>,
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

    // Defensive-mixture component q_broad = N(0, Ω); see `subject_is_draws` and
    // issue #528. `mixture == None` reproduces the single-proposal estimator
    // exactly. The broad component is built once by the caller and shared.

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
        let from_broad = mixture.is_some_and(|m| m.draws_broad(&mut rng));
        if from_broad {
            // Defensive draw η ~ N(0, Ω): centred at the prior mean (0), not η̂.
            mixture.expect("mixture set when from_broad").sample_broad(
                &mut rng,
                &mut z,
                &mut eta_sample,
            );
        } else {
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
        let log_q_narrow =
            log_t_const - 0.5 * (nu + d as f64) * (1.0 + inv_iscale_sq * mahal / nu).ln();
        // Defensive mixture q = (1−α)·q_narrow + α·N(0,Ω); the broad component's
        // log-density is the prior `log_p_eta`. Scored for every sample.
        let log_q = match mixture {
            Some(m) => m.log_q(log_q_narrow, log_p_eta),
            None => log_q_narrow,
        };

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

// ---------------------------------------------------------------------------
// Rao-Blackwellised FREM importance sampling (issue #406)
// ---------------------------------------------------------------------------
//
// On a FREM model the covariate pseudo-observations pin their etas to the
// (essentially noise-free, EPSCOV²≈0) data value `dₖ = cov_obsₖ − TVₖ`, while
// the PK observations depend only on the PK etas. Sampling all n_eta etas by
// importance sampling is then a high-dimensional, extreme multi-scale problem
// with ~1–2% ESS even when the mode is correct. Instead we integrate the
// covariate etas analytically (Rao-Blackwellisation): fix η_c = d and importance
// sample **only the n_pk PK etas** from the conditional posterior
//
//   p(η_p | y_pk, d) ∝ p_pk(y_pk | η_p) · N(η_p; μ_{p|c}, Σ_{p|c}),
//
// where the conditional Gaussian prior comes directly from the joint precision
// P = Ω⁻¹ partitioned into PK/cov blocks:
//
//   Σ_{p|c}⁻¹ = P_pp,            μ_{p|c} = −P_pp⁻¹ P_pc d.
//
// The PK proposal is the pp-block of the full posterior Hessian
// `h_post_pp = J_pk' R⁻¹ J_pk + P_pp` (the covariate rows have zero PK Jacobian,
// so this block is exactly the conditional PK posterior precision). The IS is
// then a well-conditioned n_pk-dimensional problem with near-unit ESS.
//
// Reconstructed for the M-step: η_c = d is exact, so the per-subject Ω
// sufficient statistic is `[[Σ w η_p η_p', (Σ w η_p) d'], [d (Σ w η_p)', d d']]`
// — the cc-block `d d'` summed over subjects is precisely the FREM covariate
// sample covariance. The covariate data marginal `log N(d; 0, Ω_cc + R)` is
// added to each subject's marginal log-likelihood as a closed-form constant.

/// PK vs covariate eta index partition for a FREM model. Covariate etas are the
/// eta indices that appear as FREMTYPE pseudo-observation targets; PK etas are
/// the rest. Returns `(pk_idx, cov_idx)`, each ascending.
pub(crate) fn frem_pk_cov_partition(fc: &FremConfig, n_eta: usize) -> (Vec<usize>, Vec<usize>) {
    let mut is_cov = vec![false; n_eta];
    for &(_t, e) in fc.fremtype_to_indices.values() {
        if e < n_eta {
            is_cov[e] = true;
        }
    }
    let pk_idx = (0..n_eta).filter(|&i| !is_cov[i]).collect();
    let cov_idx = (0..n_eta).filter(|&i| is_cov[i]).collect();
    (pk_idx, cov_idx)
}

/// Per-subject Rao-Blackwell partition for a FREM model with possibly-missing
/// covariates. Splits the covariate etas into:
///   - **observed** (a FREMTYPE pseudo-observation row exists): pinned at their
///     data value `dₖ = cov_obsₖ − TVₖ` and integrated analytically, and
///   - **missing** (no pseudo-obs row — the FREM data omits rows for missing
///     covariate values): latent, so they are sampled together with the PK etas
///     (the omega correlation structure imputes them, matching NONMEM).
///
/// Returns `(sampled_idx, observed_idx, d)` where `sampled_idx = pk_idx ∪
/// missing-covariate etas` (ascending) and `observed_idx` / `d` are aligned. The
/// precision-form conditional prior in [`subject_is_draws_frem_rb`] is exact for
/// this split because *all* of `observed_idx` is conditioned on and *everything
/// else* is sampled. Returns `None` when no covariate is observed for the
/// subject, or when any covariate has more than one pseudo-obs row (→ caller
/// falls back to full-dimensional IS).
pub(crate) fn subject_frem_partition(
    subject: &Subject,
    theta: &[f64],
    fc: &FremConfig,
    pk_idx: &[usize],
    cov_idx: &[usize],
) -> Option<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    // eta_idx → (fremtype, theta_idx)
    let mut eta_to_ft: std::collections::HashMap<usize, (u16, usize)> =
        std::collections::HashMap::new();
    for (&ft, &(t, e)) in fc.fremtype_to_indices.iter() {
        eta_to_ft.insert(e, (ft, t));
    }
    let mut observed = Vec::with_capacity(cov_idx.len());
    let mut d = Vec::with_capacity(cov_idx.len());
    let mut missing = Vec::new();
    for &e in cov_idx {
        match eta_to_ft.get(&e) {
            Some(&(ft, t)) => {
                // The Rao-Blackwell marginal pins this covariate eta at a single
                // deviation `d` and subtracts a single `0.5·ln(R)` constant per
                // covariate. A covariate with >1 pseudo-obs row (a time-varying
                // covariate, or a duplicate) would have its extra rows scored by
                // `obs_nll` at the pinned eta with a non-zero residual that the
                // constant does not cancel, silently corrupting the weights. The
                // RB split is only exact for one row per FREMTYPE — bail to the
                // full-dimensional sampler (which scores every row consistently).
                if subject.fremtype.iter().filter(|&&x| x == ft).count() > 1 {
                    return None;
                }
                match subject.fremtype.iter().position(|&x| x == ft) {
                    Some(row) => match subject.observations.get(row) {
                        Some(&obs) => {
                            observed.push(e);
                            d.push(obs - theta.get(t).copied().unwrap_or(0.0));
                        }
                        None => missing.push(e),
                    },
                    None => missing.push(e),
                }
            }
            None => missing.push(e),
        }
    }
    if observed.is_empty() {
        return None;
    }
    let mut sampled: Vec<usize> = pk_idx.iter().chain(missing.iter()).copied().collect();
    sampled.sort_unstable();
    Some((sampled, observed, d))
}

/// Extract the sub-matrix `m[rows, cols]`.
fn submatrix(m: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(rows.len(), cols.len());
    for (a, &r) in rows.iter().enumerate() {
        for (b, &c) in cols.iter().enumerate() {
            out[(a, b)] = m[(r, c)];
        }
    }
    out
}

/// Rao-Blackwellised FREM E-step for one subject. See the module section above.
///
/// `pk_idx` is the **sampled** eta set (PK etas plus any missing-covariate etas,
/// from [`subject_frem_partition`]) and `cov_idx` / `d` are the **observed**
/// covariate etas pinned at their data deviations. The precision-form conditional
/// prior is exact for this split: all of `cov_idx` is conditioned on and
/// everything in `pk_idx` is sampled. Returns `None` if the conditioning is
/// degenerate (caller then falls back to the full-dimensional [`subject_is_draws`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn subject_is_draws_frem_rb(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma: &[f64],
    eta_hat: &DVector<f64>,
    h_post: &DMatrix<f64>,
    omega_inv: &DMatrix<f64>,
    omega_matrix: &DMatrix<f64>,
    pk_idx: &[usize],
    cov_idx: &[usize],
    d: &[f64],
    n_eta: usize,
    k_samples: usize,
    nu: f64,
    seed: u64,
    scratch: &mut EventPkParams,
    iscale: f64,
    use_sobol: bool,
    defensive_alpha: f64,
) -> Option<SubjectDraws> {
    let np = pk_idx.len();
    let nc = cov_idx.len();
    if np == 0 || nc == 0 || d.len() != nc {
        return None;
    }
    let fc = model.frem_config.as_ref()?;
    let mvn = !nu.is_finite();
    let mut rng = StdRng::seed_from_u64(seed);

    // Conditional PK prior: precision P_pp, mean μ = −P_pp⁻¹ P_pc d.
    let p_pp = submatrix(omega_inv, pk_idx, pk_idx);
    let p_pc = submatrix(omega_inv, pk_idx, cov_idx);
    let p_pp_chol = p_pp.clone().cholesky()?;
    let logdet_p_pp = 2.0 * (0..np).map(|i| p_pp_chol.l()[(i, i)].ln()).sum::<f64>();
    let dvec = DVector::from_column_slice(d);
    let mu_pc = p_pp_chol.solve(&(-(&p_pc * &dvec))); // np

    // PK proposal: pp-block of the full posterior Hessian (= conditional PK
    // posterior precision), centred at the PK sub-vector of the mode.
    let h_pp = submatrix(h_post, pk_idx, pk_idx);
    let proposal = build_proposal(&h_pp, &p_pp, np)?;
    let eta_hat_pk: Vec<f64> = pk_idx.iter().map(|&i| eta_hat[i]).collect();

    // Defensive-mixture component for the RB path (issue #528). The covering
    // distribution is the *conditional* PK prior N(μ_pc, P_pp⁻¹) — the exact
    // analogue of N(0, Ω) in the full-dimensional sampler — so the broad density
    // at any η_p equals `log_prior` below. `defensive_alpha == 0` (or a non-PD
    // P_pp) leaves the legacy single-proposal RB sampler and its RNG stream
    // untouched. Sobol QMC is disabled while the mixture is active.
    let mixture = if defensive_alpha > 0.0 {
        build_broad_proposal(&p_pp, np)
            .map(|broad| DefensiveMixture::from_proposal(broad, defensive_alpha))
    } else {
        None
    };

    // Covariate data marginal log p(d) = log N(d; 0, Ω_cc + R), R = EPSCOV²·I.
    let r_cov = {
        let s = sigma[fc.covariate_sigma_index];
        (s * s).max(1e-12)
    };
    let mut occ = submatrix(omega_matrix, cov_idx, cov_idx);
    for i in 0..nc {
        occ[(i, i)] += r_cov;
    }
    let occ_chol = occ.cholesky()?;
    let logdet_occ = 2.0 * (0..nc).map(|i| occ_chol.l()[(i, i)].ln()).sum::<f64>();
    let occ_inv_d = occ_chol.solve(&dvec);
    let quad_cov = dvec.dot(&occ_inv_d);
    // Covariate data marginal log N(d; 0, Ω_cc + R). The covariate pseudo-obs
    // are *observations*, so their 2π normalizer (nc·ln2π) must be dropped to
    // match the rest of the OFV, which omits the per-obs 2π constant (see
    // `obs_nll_subject_into`) — i.e. NONMEM's "OBJECTIVE FUNCTION WITHOUT
    // CONSTANT" convention. Including it inflated the RB marginal by
    // Σ nc·ln2π (≈ n_covariate_obs · ln2π).
    let log_p_d = -0.5 * (logdet_occ + quad_cov);

    // Constants. The covariate observation rows contribute a fixed
    // 0.5·Σ ln(R) to obs_nll (their residual is ≈0 at η_c = d); subtract it so
    // `log p_pk` is the PK-only observation log-likelihood.
    let cov_obs_const = 0.5 * nc as f64 * r_cov.ln();
    let half_np = 0.5 * np as f64;
    let log_prior_const = -half_np * TWO_PI.ln() + 0.5 * logdet_p_pp;
    let iscale_log_adj = -(np as f64) * iscale.ln();
    let inv_iscale_sq = 1.0 / (iscale * iscale);
    let log_q_const = if mvn {
        -half_np * TWO_PI.ln() + 0.5 * proposal.log_det_inv_scale + iscale_log_adj
    } else {
        ln_gamma(0.5 * (nu + np as f64)) - ln_gamma(0.5 * nu)
            + 0.5 * proposal.log_det_inv_scale
            + iscale_log_adj
            - half_np * (nu * std::f64::consts::PI).ln()
    };

    let normal = StandardNormal;
    let chi_sq = if mvn {
        None
    } else {
        Some(ChiSquared::new(nu).expect("ChiSquared requires nu > 0; checked by caller"))
    };
    let sobol_draws = if use_sobol && mvn && mixture.is_none() {
        Some(sobol_normal_draws(np, k_samples, seed))
    } else {
        None
    };

    let mut log_w: Vec<f64> = Vec::with_capacity(k_samples);
    let mut etas: Vec<Vec<f64>> = Vec::with_capacity(k_samples);
    let mut eta_p_samples: Vec<Vec<f64>> = Vec::with_capacity(k_samples);
    let mut z = vec![0.0_f64; np];
    let mut eta_p = vec![0.0_f64; np];
    let mut diff_q = vec![0.0_f64; np];

    for s_idx in 0..k_samples {
        let from_broad = mixture.as_ref().is_some_and(|m| m.draws_broad(&mut rng));
        if from_broad {
            // Defensive draw η_p ~ N(μ_pc, P_pp⁻¹): the conditional PK prior.
            let m = mixture.as_ref().expect("mixture set when from_broad");
            m.sample_broad(&mut rng, &mut z, &mut eta_p);
            for (j, e) in eta_p.iter_mut().enumerate() {
                *e += mu_pc[j];
            }
        } else {
            if let Some(ref qr) = sobol_draws {
                z.copy_from_slice(&qr[s_idx]);
            } else {
                for zi in z.iter_mut() {
                    *zi = normal.sample(&mut rng);
                }
            }
            let scale = match &chi_sq {
                Some(c) => (nu / c.sample(&mut rng).max(1e-300)).sqrt() * iscale,
                None => iscale,
            };
            proposal.apply_l_sigma(&z, &mut eta_p, scale);
            for (j, e) in eta_p.iter_mut().enumerate() {
                *e += eta_hat_pk[j];
            }
        }

        // Reconstruct full η: PK sample at pk_idx, fixed d at cov_idx.
        let mut full_eta = vec![0.0_f64; n_eta];
        for (a, &i) in pk_idx.iter().enumerate() {
            full_eta[i] = eta_p[a];
        }
        for (b, &i) in cov_idx.iter().enumerate() {
            full_eta[i] = d[b];
        }

        let obs_nll = obs_nll_subject_into(model, subject, theta, sigma, &full_eta, scratch);
        let log_p_y = -(obs_nll - cov_obs_const); // PK-only obs log-likelihood

        // Conditional prior log N(η_p; μ, P_pp): (η_p−μ)' P_pp (η_p−μ).
        let mut quad = 0.0_f64;
        for i in 0..np {
            let mut row = 0.0_f64;
            for j in 0..np {
                row += p_pp[(i, j)] * (eta_p[j] - mu_pc[j]);
            }
            quad += row * (eta_p[i] - mu_pc[i]);
        }
        let log_prior = log_prior_const - 0.5 * quad;

        // Proposal log q over η_p (centred at η̂_pk).
        for (j, dq) in diff_q.iter_mut().enumerate() {
            *dq = eta_p[j] - eta_hat_pk[j];
        }
        let mahal = proposal.mahalanobis(&diff_q);
        let log_q_narrow = if mvn {
            log_q_const - 0.5 * inv_iscale_sq * mahal
        } else {
            log_q_const - 0.5 * (nu + np as f64) * (1.0 + inv_iscale_sq * mahal / nu).ln()
        };
        // Mixture denominator q = (1−α)·q_narrow + α·q_cond_prior; the broad
        // component's log-density is the conditional prior `log_prior`.
        let log_q = match mixture.as_ref() {
            Some(m) => m.log_q(log_q_narrow, log_prior),
            None => log_q_narrow,
        };

        log_w.push(log_p_y + log_prior - log_q);
        eta_p_samples.push(eta_p.clone());
        etas.push(full_eta);
    }

    let (lse, weights) = logsumexp_with_normalised(&log_w);
    let log_marginal = lse - (k_samples as f64).ln() + log_p_d;
    let ess = {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    };
    let ess_fraction = ess / (k_samples as f64);

    // Reconstruct full-η weighted moments. η_p moments from the samples; η_c = d
    // exact, so the cc-block is d d' and the cross-block is (Σ w η_p) d'.
    let mut mean_p = vec![0.0_f64; np];
    let mut sm_pp = DMatrix::<f64>::zeros(np, np);
    for (w, ep) in weights.iter().zip(eta_p_samples.iter()) {
        for i in 0..np {
            mean_p[i] += w * ep[i];
            for j in 0..np {
                sm_pp[(i, j)] += w * ep[i] * ep[j];
            }
        }
    }
    let mut mean = vec![0.0_f64; n_eta];
    let mut second_moment = DMatrix::<f64>::zeros(n_eta, n_eta);
    for (a, &i) in pk_idx.iter().enumerate() {
        mean[i] = mean_p[a];
    }
    for (b, &i) in cov_idx.iter().enumerate() {
        mean[i] = d[b];
    }
    for (a, &i) in pk_idx.iter().enumerate() {
        for (b, &j) in pk_idx.iter().enumerate() {
            second_moment[(i, j)] = sm_pp[(a, b)];
        }
    }
    for (a, &i) in pk_idx.iter().enumerate() {
        for (b, &j) in cov_idx.iter().enumerate() {
            let v = mean_p[a] * d[b];
            second_moment[(i, j)] = v;
            second_moment[(j, i)] = v;
        }
    }
    for (a, &i) in cov_idx.iter().enumerate() {
        for (b, &j) in cov_idx.iter().enumerate() {
            second_moment[(i, j)] = d[a] * d[b];
        }
    }

    Some(SubjectDraws {
        log_marginal,
        ess_fraction,
        etas,
        weights,
        mean,
        second_moment,
    })
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
    mixture: Option<&DefensiveMixture>,
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

    // Defensive-mixture component q_broad = N(0, Ω) (issue #528). Drawing an
    // `alpha` fraction of samples from the prior — and scoring every sample
    // under the full mixture density q = (1−α)·q_narrow + α·q_broad — bounds the
    // importance weight of any sample by `p(y|η)/α` (since q ≥ α·q_broad and
    // q_broad is the prior). That stops a weakly-identified subject (e.g. an
    // analytical `[initial_conditions]` baseline whose V cancels in the
    // amplitude) from contributing a single dominant sample that hijacks the
    // importance-weighted M-step. It bounds the weights, not the raw ESS — a
    // sharp interior likelihood spike can still keep ESS low — but it is enough
    // to keep the population estimates identifiable. `mixture == None` reproduces
    // the pre-#528 single-proposal sampler exactly. The broad component (which
    // depends only on Ω) is built once by the caller and shared across subjects.
    // Quasi-random (Sobol) draws are disabled when the mixture is active — the
    // branch structure breaks the QMC sequence.

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

    // Pre-generate Sobol quasi-random draws if requested and MVN. Disabled when
    // the defensive mixture is active (the per-sample component branch breaks
    // the quasi-random sequence).
    let sobol_draws = if use_sobol && mvn && mixture.is_none() {
        Some(sobol_normal_draws(d, k_samples, seed))
    } else {
        None
    };

    for sample_idx in 0..k_samples {
        let mut eta_sample = vec![0.0_f64; d];
        let from_broad = mixture.is_some_and(|m| m.draws_broad(&mut rng));
        if from_broad {
            // Defensive draw η ~ N(0, Ω): centred at the prior mean (0), not η̂.
            mixture.expect("mixture set when from_broad").sample_broad(
                &mut rng,
                &mut z,
                &mut eta_sample,
            );
        } else {
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
            proposal.apply_l_sigma(&z, &mut eta_sample, scale);
            for (j, e) in eta_sample.iter_mut().enumerate() {
                *e += eta_hat[j];
            }
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
        let log_q_narrow = if mvn {
            log_q_const - 0.5 * inv_iscale_sq * mahal
        } else {
            log_q_const - 0.5 * (nu + d as f64) * (1.0 + inv_iscale_sq * mahal / nu).ln()
        };
        // Mixture denominator: q = (1−α)·q_narrow + α·q_broad, with the broad
        // component's log-density equal to the Gaussian prior `log_p_eta`
        // (q_broad = N(0,Ω)). Evaluated for every sample (balance heuristic),
        // not just the component it was drawn from.
        let log_q = match mixture {
            Some(m) => m.log_q(log_q_narrow, log_p_eta),
            None => log_q_narrow,
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
            None,  // tune the narrow proposal alone; defensive mixing applies at the real draw
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
    // IIV on residual error (#409): scale PK residual variance by exp(2·η̂_ruv).
    let ruv_scale = model.residual_var_scale(eta_hat.as_slice());
    if ruv_scale != 1.0 {
        for v in r_diag.iter_mut() {
            *v *= ruv_scale;
        }
    }
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

    // IIV on residual error (#409): η_ruv is a BSV eta (index < n_eta) whose
    // prediction-Jacobian column is zero; add its data curvature so the joint IS
    // proposal tightens around η̂_ruv. See `add_ruv_posterior_curvature`.
    add_ruv_posterior_curvature(&mut h_post, model, subject, &ipreds, &r_diag, n_eta);

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
        // IIV on residual error (#409): scale by exp(2·η_ruv) for this draw's eta.
        let ruv_scale = model.residual_var_scale(eta_sample);
        let mut obs_nll = 0.0_f64;
        for (j, (&y, &f)) in subject.observations.iter().zip(ipreds.iter()).enumerate() {
            let f = f.max(1e-12);
            let v =
                (model.residual_variance_at(subject.obs_cmts[j], f, sigma) * ruv_scale).max(1e-12);
            let cens = subject.cens.get(j).copied().unwrap_or(0);
            if m3 && cens != 0 {
                obs_nll += -m3_logcdf(y, f, v.sqrt(), cens);
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
/// Strategy: Cholesky(H + Λ) with a **per-dimension** relative jitter
/// `Λᵢᵢ = max(1e−6·|Hᵢᵢ|, 1e−10)`. A single global `λ = 1e−6·trace(H)/d`
/// (the previous strategy) is dominated by the sharpest dimensions: on a FREM
/// model the covariate pseudo-obs dims (~1e6) and near-fixed dims (~1e10) push
/// `λ` to ~1e3, which then swamps the well-conditioned PK dims (~1e2) and
/// collapses their proposal width 5–10×, hurting ESS even when the mode is
/// correct (issue #406). The per-dimension form regularizes each dimension
/// proportionally to its own curvature, leaving every dimension's proposal
/// width intact. If H is so degenerate that even the jittered matrix isn't
/// positive-definite, fall back to Σ = Ω (the prior covariance) — a broad
/// proposal that won't give a sharp likelihood estimate but stays well-defined.
///
/// Returns `None` only when `d == 0`.
fn build_proposal(h: &DMatrix<f64>, omega_inv: &DMatrix<f64>, d: usize) -> Option<Proposal> {
    if d == 0 {
        return None;
    }
    debug_assert_eq!(h.nrows(), d);
    debug_assert_eq!(h.ncols(), d);

    let mut h_reg = h.clone();
    for i in 0..d {
        h_reg[(i, i)] += (1e-6 * h[(i, i)].abs()).max(1e-10);
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
    build_broad_proposal(omega_inv, d)
}

/// Build the prior-covariance proposal `Σ = Ω` (centred at 0) used as the
/// defensive-mixture component (issue #528). Factoring `Ω⁻¹ = L Lᵀ` gives the
/// same `(L L' = Σ⁻¹)` interface as [`build_proposal`], so `apply_l_sigma`
/// draws `N(0, Ω)` and `mahalanobis` scores `η' Ω⁻¹ η`. Because the conditional
/// posterior `p(η|y) ∝ p(η)·p(y|η)` has support contained in the prior's, this
/// component is guaranteed to cover the posterior — bounding each importance
/// weight by `p(y|η)/α` regardless of how poorly the narrow proposal is centred
/// or scaled, so no single sample can dominate the weighted M-step. Returns
/// `None` only when `Ω⁻¹` is not positive-definite (then the mixture degrades to
/// the narrow proposal alone).
fn build_broad_proposal(omega_inv: &DMatrix<f64>, d: usize) -> Option<Proposal> {
    if d == 0 {
        return None;
    }
    let omega_inv_chol = omega_inv.clone().cholesky()?;
    let l = omega_inv_chol.l();
    let log_det = 2.0 * (0..d).map(|i| l[(i, i)].ln()).sum::<f64>();
    Some(Proposal {
        chol_h: l,
        log_det_inv_scale: log_det,
        d,
    })
}

/// Defensive-mixture proposal component and precomputed log-mixing weights
/// (issue #528). The proposal is `q = (1−α)·q_narrow + α·q_broad`; this struct
/// owns the broad covering component `q_broad` and the per-component log-weights
/// so the math lives in **one** place shared by `subject_is_estimate` and
/// `subject_is_draws` (rather than being copy-pasted into each).
///
/// `q_broad` depends only on Ω (not on the subject), so the caller builds this
/// **once** before the parallel subject loop and shares it read-only across
/// subjects — avoiding a redundant per-subject Cholesky of Ω⁻¹. A `None` mixture
/// means "inactive" (`alpha == 0` or a non-PD Ω) and reproduces the pre-#528
/// single-proposal sampler exactly, including its RNG stream.
pub(crate) struct DefensiveMixture {
    /// Broad covering component `q_broad = N(0, Ω)`.
    broad: Proposal,
    /// Mixing fraction α ∈ (0, 1): probability a given draw comes from `q_broad`.
    alpha: f64,
    ln_alpha: f64,
    ln_one_minus_alpha: f64,
}

impl DefensiveMixture {
    /// Build the `N(0, Ω)` defensive component from Ω⁻¹. Returns `None` (no
    /// mixture) when `alpha <= 0` or Ω⁻¹ is not positive-definite — in which case
    /// the caller degrades to the narrow proposal alone.
    pub(crate) fn new(omega_inv: &DMatrix<f64>, d: usize, alpha: f64) -> Option<Self> {
        if alpha <= 0.0 {
            return None;
        }
        let broad = build_broad_proposal(omega_inv, d)?;
        Some(Self {
            broad,
            alpha,
            ln_alpha: alpha.ln(),
            ln_one_minus_alpha: (1.0 - alpha).ln(),
        })
    }

    /// Build a defensive component from a pre-factored broad proposal (used by the
    /// FREM Rao-Blackwell path, whose covering distribution is the conditional PK
    /// prior `N(μ, P_pp⁻¹)` rather than `N(0, Ω)`).
    fn from_proposal(broad: Proposal, alpha: f64) -> Self {
        Self {
            broad,
            alpha,
            ln_alpha: alpha.ln(),
            ln_one_minus_alpha: (1.0 - alpha).ln(),
        }
    }

    /// Decide whether this draw comes from the broad component. Consumes exactly
    /// one uniform from `rng` — kept on the `Option` so callers preserve the
    /// legacy (no-mixture) RNG stream by short-circuiting before any draw.
    fn draws_broad(&self, rng: &mut StdRng) -> bool {
        rng.random::<f64>() < self.alpha
    }

    /// Draw `η ~ q_broad` into `out` (centred at 0; the FREM RB caller shifts to
    /// μ afterwards), using scratch buffer `z`.
    fn sample_broad(&self, rng: &mut StdRng, z: &mut [f64], out: &mut [f64]) {
        let normal = StandardNormal;
        for zi in z.iter_mut() {
            *zi = normal.sample(rng);
        }
        self.broad.apply_l_sigma(z, out, 1.0);
    }

    /// Mixture log-density `log[(1−α)·q_narrow + α·q_broad]` (balance heuristic —
    /// scored for every sample regardless of which component produced it).
    /// `log_q_broad` is the broad component's log-density at this sample (the
    /// Gaussian prior for the `N(0,Ω)` component, or the conditional prior for the
    /// FREM RB component).
    fn log_q(&self, log_q_narrow: f64, log_q_broad: f64) -> f64 {
        logsumexp2(
            self.ln_one_minus_alpha + log_q_narrow,
            self.ln_alpha + log_q_broad,
        )
    }
}

// ---------------------------------------------------------------------------
// Numerical helpers
// ---------------------------------------------------------------------------

/// Stable `log(eᵃ + eᵇ)` for the two-component defensive-mixture denominator.
fn logsumexp2(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Numerically stable `log Σ exp(xᵢ)` plus the normalised weights `wᵢ`.
fn logsumexp_with_normalised(xs: &[f64]) -> (f64, Vec<f64>) {
    if xs.is_empty() {
        return (f64::NEG_INFINITY, Vec::new());
    }

    let n_pos_inf = xs
        .iter()
        .filter(|x| x.is_infinite() && x.is_sign_positive())
        .count();
    if n_pos_inf > 0 {
        let w = 1.0 / n_pos_inf as f64;
        let weights = xs
            .iter()
            .map(|x| {
                if x.is_infinite() && x.is_sign_positive() {
                    w
                } else {
                    0.0
                }
            })
            .collect();
        return (f64::INFINITY, weights);
    }

    let m = xs
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return (f64::NEG_INFINITY, vec![0.0; xs.len()]);
    }
    let mut sum = 0.0;
    let mut shifted: Vec<f64> = Vec::with_capacity(xs.len());
    for &x in xs {
        let s = if x.is_finite() { (x - m).exp() } else { 0.0 };
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
    // block_sigma: build the proposal precision from the FULL residual covariance,
    // `H_post = Ω⁻¹ + Jᵀ R⁻¹ J`, so the Student-t proposal reflects the correlated
    // posterior shape (a diagonal-R proposal still yields correct importance
    // weights — those use the dense data term — but inflates the weight variance /
    // lowers ESS). block_sigma is rejected with FREM and iiv_on_ruv, so no R
    // overrides or residual-eta curvature apply. A non-PD R falls through to the
    // diagonal approximation below.
    if !model.residual_correlations.is_empty() {
        let r = crate::stats::residual_error::compute_r_matrix_with_correlations(
            &model.error_spec,
            &ipreds,
            &subject.obs_cmts,
            &subject.obs_times,
            &subject.obs_raw_times,
            &subject.occasions,
            sigma,
            &model.residual_correlations,
        );
        if let Some(chol) = r.cholesky() {
            let r_inv = chol.inverse();
            return omega_inv + jacobian.transpose() * &r_inv * jacobian;
        }
    }
    let mut r_diag = compute_r_diag(&model.error_spec, &ipreds, &subject.obs_cmts, sigma);
    // IIV on residual error (#409): scale the PK residual variance at the mode by
    // exp(2·η̂_ruv) so the Laplace proposal precision reflects the per-subject
    // residual SD. FREM rows are overwritten below with their own variance.
    let ruv_scale = model.residual_var_scale(eta_hat.as_slice());
    if ruv_scale != 1.0 {
        for v in r_diag.iter_mut() {
            *v *= ruv_scale;
        }
    }
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
    add_ruv_posterior_curvature(&mut h_post, model, subject, &ipreds, &r_diag, n_eta);
    h_post
}

/// IIV on residual error (#409): the prediction `f` does not depend on η_ruv, so
/// its Jacobian column is identically zero and the `JᵀR⁻¹J` loop leaves the
/// η_ruv diagonal of `H_post` at the prior precision `Ω⁻¹` alone. The IS
/// proposal built from that Hessian would then sample η_ruv from its prior
/// rather than the (much tighter) data-informed posterior, inflating the
/// importance-weight variance on that axis.
///
/// Add the exact second derivative of the per-subject objective in η_ruv. With
/// `R_j = R₀ⱼ·exp(2·η_ruv)`, the data term `0.5·Σⱼ[(y−f)²/Rⱼ + ln Rⱼ]` has
/// `∂²/∂η_ruv² = Σⱼ 2·(y−f)²/Rⱼ` (the `ln Rⱼ` term is linear in η_ruv, so it
/// drops). FREM covariate rows carry no PK residual and are excluded. Off-
/// diagonal η_ruv couplings are left at the Gauss-Newton level (zero), matching
/// how the structural-eta block already drops `∂R/∂η` cross terms.
fn add_ruv_posterior_curvature(
    h_post: &mut DMatrix<f64>,
    model: &CompiledModel,
    subject: &Subject,
    ipreds: &[f64],
    r_diag: &[f64],
    n_eta: usize,
) {
    let Some(k) = model.residual_error_eta else {
        return;
    };
    if k >= n_eta {
        return;
    }
    let mut curv = 0.0;
    for (j, &f) in ipreds.iter().enumerate() {
        if subject.fremtype.get(j).copied().unwrap_or(0) != 0 {
            continue; // FREM covariate pseudo-observation: no PK residual.
        }
        let rj = r_diag[j].max(1e-12);
        let res = subject.observations[j] - f;
        curv += 2.0 * res * res / rj;
    }
    h_post[(k, k)] += curv;
}

// ---------------------------------------------------------------------------
// Tier 1 unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// IIV on residual error (#409): `add_ruv_posterior_curvature` injects the
    /// η_ruv data curvature `Σⱼ 2·res²/Rⱼ` into the proposal Hessian diagonal
    /// (PK rows only — FREM covariate rows are skipped), and is a no-op when no
    /// `residual_error_eta` is set. Without this the IS proposal samples η_ruv
    /// from the prior, since its prediction-Jacobian column is identically zero.
    #[test]
    fn ruv_posterior_curvature_adds_data_term_and_skips_frem() {
        use crate::types::{DoseEvent, GradientMethod, Subject};
        use std::collections::HashMap;

        let mut subject = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0],
            obs_raw_times: Vec::new(),
            observations: vec![10.0, 8.0, 0.0],
            obs_cmts: vec![1; 3],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; 3],
            occasions: vec![1, 1, 1],
            dose_occasions: Vec::new(),
            // 3rd row is a FREM covariate pseudo-observation.
            fremtype: vec![0, 0, 5],
            #[cfg(feature = "survival")]
            obs_records: vec![],
        };
        let ipreds = vec![9.0, 7.0, 99.0];
        let r_diag = vec![4.0, 2.0, 1.0];
        let n_eta = 2;

        let mut model = crate::types::test_helpers::analytical_model(GradientMethod::Auto);

        // No residual-error eta → no-op.
        model.residual_error_eta = None;
        let mut h = DMatrix::<f64>::zeros(n_eta, n_eta);
        add_ruv_posterior_curvature(&mut h, &model, &subject, &ipreds, &r_diag, n_eta);
        assert!(
            h.iter().all(|&v| v == 0.0),
            "no residual eta must be a no-op"
        );

        // η_ruv at index 1: diagonal gains Σ 2·res²/R over PK rows only.
        // 2·(10−9)²/4 + 2·(8−7)²/2 = 0.5 + 1.0 = 1.5 (FREM row 3 excluded).
        model.residual_error_eta = Some(1);
        add_ruv_posterior_curvature(&mut h, &model, &subject, &ipreds, &r_diag, n_eta);
        assert!((h[(1, 1)] - 1.5).abs() < 1e-12, "got {}", h[(1, 1)]);
        assert_eq!(h[(0, 0)], 0.0, "structural diagonal untouched");
        assert_eq!(h[(0, 1)], 0.0, "no off-diagonal coupling added");

        // Out-of-range index is a safe no-op (defensive).
        let mut h2 = DMatrix::<f64>::zeros(n_eta, n_eta);
        model.residual_error_eta = Some(5);
        add_ruv_posterior_curvature(&mut h2, &model, &subject, &ipreds, &r_diag, n_eta);
        assert!(
            h2.iter().all(|&v| v == 0.0),
            "out-of-range k must be a no-op"
        );

        // Including a FREM row only (no PK rows) yields zero curvature.
        subject.fremtype = vec![1, 1, 1];
        let mut h3 = DMatrix::<f64>::zeros(n_eta, n_eta);
        model.residual_error_eta = Some(1);
        add_ruv_posterior_curvature(&mut h3, &model, &subject, &ipreds, &r_diag, n_eta);
        assert_eq!(h3[(1, 1)], 0.0, "all-FREM subject adds no curvature");
    }

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
    fn logsumexp2_matches_direct_and_handles_neginf() {
        // Matches the naive log(eᵃ + eᵇ) in the well-scaled regime.
        let a = -1.3_f64;
        let b = 2.7_f64;
        let want = (a.exp() + b.exp()).ln();
        assert!((logsumexp2(a, b) - want).abs() < 1e-12);
        // One degenerate component collapses to the other.
        assert!((logsumexp2(f64::NEG_INFINITY, 5.0) - 5.0).abs() < 1e-12);
        assert!((logsumexp2(5.0, f64::NEG_INFINITY) - 5.0).abs() < 1e-12);
        // Both degenerate → −∞ (no NaN).
        assert_eq!(
            logsumexp2(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        // Stable for large magnitudes (no overflow).
        let big = logsumexp2(1000.0, 1001.0);
        assert!((big - (1001.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-9);
    }

    #[test]
    fn build_broad_proposal_factors_prior_and_scores_quadratic_form() {
        // Ω⁻¹ = diag(4) ⇒ Ω = diag(0.25). The broad proposal draws N(0, Ω) and
        // scores ‖·‖²_{Ω⁻¹}; check both against closed forms (issue #528).
        let omega_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0]));
        let p = build_broad_proposal(&omega_inv, 2).expect("PD Ω⁻¹ must factor");
        // log|Ω⁻¹| = log(4·9) = log 36.
        assert!((p.log_det_inv_scale - 36.0_f64.ln()).abs() < 1e-12);
        // mahalanobis(diff) == diff' Ω⁻¹ diff = 4·d0² + 9·d1².
        let diff = [0.5, -2.0];
        let want = 4.0 * 0.25 + 9.0 * 4.0;
        assert!((p.mahalanobis(&diff) - want).abs() < 1e-12);
        // apply_l_sigma maps N(0,I) → N(0,Ω): unit input on dim 0 ⇒ sd 0.5.
        let mut out = [0.0_f64; 2];
        p.apply_l_sigma(&[1.0, 0.0], &mut out, 1.0);
        assert!((out[0].abs() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn build_broad_proposal_rejects_degenerate_inputs() {
        // d == 0 → None (no random effects).
        assert!(build_broad_proposal(&DMatrix::zeros(0, 0), 0).is_none());
        // Non-PD Ω⁻¹ (zero matrix) → None, so the mixture degrades to narrow-only.
        assert!(build_broad_proposal(&DMatrix::zeros(2, 2), 2).is_none());
    }

    #[test]
    fn defensive_mixture_inactive_for_nonpositive_alpha() {
        let omega_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0]));
        // alpha <= 0 → no mixture (the caller passes None into the sampler).
        assert!(DefensiveMixture::new(&omega_inv, 2, 0.0).is_none());
        assert!(DefensiveMixture::new(&omega_inv, 2, -0.1).is_none());
        // A non-PD Ω⁻¹ also yields None (degrade to narrow-only).
        assert!(DefensiveMixture::new(&DMatrix::zeros(2, 2), 2, 0.1).is_none());
    }

    #[test]
    fn defensive_mixture_scores_balance_heuristic_and_samples_broad() {
        let omega_inv = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0]));
        let m = DefensiveMixture::new(&omega_inv, 2, 0.25).expect("PD Ω⁻¹ → mixture");
        // log_q is the two-component balance-heuristic denominator.
        let (log_q_narrow, log_q_broad) = (-3.0, -1.0);
        let expected = logsumexp2(0.75_f64.ln() + log_q_narrow, 0.25_f64.ln() + log_q_broad);
        assert!((m.log_q(log_q_narrow, log_q_broad) - expected).abs() < 1e-12);
        // draws_broad consumes one uniform and decides by the α threshold; with a
        // fixed seed the outcome is deterministic. sample_broad maps N(0,I)→N(0,Ω):
        // a unit input on dim 0 ⇒ sd 0.5 (Ω = diag(0.25, 1/9)).
        let mut rng = StdRng::seed_from_u64(7);
        let _ = m.draws_broad(&mut rng);
        let mut z = [1.0_f64, 0.0];
        let mut out = [0.0_f64; 2];
        // sample_broad refills z from rng; force the factor check via apply_l_sigma.
        m.broad.apply_l_sigma(&z, &mut out, 1.0);
        assert!((out[0].abs() - 0.5).abs() < 1e-12);
        // Exercise the rng path too (just assert it writes finite values).
        m.sample_broad(&mut rng, &mut z, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
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
    fn logsumexp_ignores_nan_log_weights() {
        let (lse, w) = logsumexp_with_normalised(&[f64::NAN, 2.0, 3.0, f64::NEG_INFINITY]);
        assert!((lse - (3.0 + (1.0 + (-1.0_f64).exp()).ln())).abs() < 1e-10);
        assert_eq!(w[0], 0.0);
        assert!(w[1] > 0.0);
        assert!(w[2] > w[1]);
        assert_eq!(w[3], 0.0);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn logsumexp_handles_positive_infinite_log_weights_without_nan() {
        let (lse, w) = logsumexp_with_normalised(&[1.0, f64::INFINITY, f64::NAN, f64::INFINITY]);
        assert_eq!(lse, f64::INFINITY);
        assert_eq!(w, vec![0.0, 0.5, 0.0, 0.5]);
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
    fn frem_partition_splits_pk_and_cov_etas() {
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (5usize, 1usize)); // cov eta at index 1
        map.insert(200u16, (6usize, 3usize)); // cov eta at index 3
        let fc = FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        };
        let (pk, cov) = frem_pk_cov_partition(&fc, 4);
        assert_eq!(pk, vec![0, 2]);
        assert_eq!(cov, vec![1, 3]);
    }

    /// Build a Subject carrying only the fields `subject_frem_partition` reads.
    fn frem_subject(fremtype: Vec<u16>, observations: Vec<f64>) -> Subject {
        Subject {
            id: "t".into(),
            doses: Vec::new(),
            obs_times: Vec::new(),
            obs_raw_times: Vec::new(),
            observations,
            obs_cmts: Vec::new(),
            covariates: std::collections::HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: Vec::new(),
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
            fremtype,
            #[cfg(feature = "survival")]
            obs_records: Vec::new(),
        }
    }

    #[test]
    fn frem_partition_bails_on_duplicate_covariate_rows() {
        // One eta (idx 1) mapped to FREMTYPE 100; pk = {0}, cov = {1}.
        let mut map = std::collections::HashMap::new();
        map.insert(100u16, (0usize, 1usize)); // (theta_idx, eta_idx)
        let fc = FremConfig {
            fremtype_to_indices: map,
            covariate_sigma_index: 0,
        };
        let theta = [0.0_f64];
        // Single covariate row → RB split is exact → Some.
        let single = frem_subject(vec![0, 100], vec![5.0, 1.2]);
        assert!(subject_frem_partition(&single, &theta, &fc, &[0], &[1]).is_some());
        // Two rows of the same FREMTYPE (time-varying / duplicate) → the
        // cov_obs_const cancellation breaks → must bail to full-dim IS (None).
        let dup = frem_subject(vec![100, 0, 100], vec![1.2, 5.0, 1.3]);
        assert!(
            subject_frem_partition(&dup, &theta, &fc, &[0], &[1]).is_none(),
            "duplicate covariate rows must disable the RB partition"
        );
    }

    #[test]
    fn rb_conditional_prior_matches_covariance_form() {
        // The RB draws path takes the conditional PK prior from the joint
        // precision: precision = P_pp, mean = −P_pp⁻¹ P_pc d. Verify this equals
        // the textbook covariance-form conditioning Σ = Ω_pp − Ω_pc Ω_cc⁻¹ Ω_cp,
        // μ = Ω_pc Ω_cc⁻¹ d. (pk = {0,1}, cov = {2}.)
        let omega = DMatrix::from_row_slice(3, 3, &[2.0, 0.3, 0.5, 0.3, 1.0, 0.2, 0.5, 0.2, 3.0]);
        let p = omega.clone().try_inverse().unwrap();
        let pk = [0usize, 1];
        let cov = [2usize];
        let d = DVector::from_column_slice(&[0.7]);

        // Precision form (what subject_is_draws_frem_rb uses).
        let p_pp = submatrix(&p, &pk, &pk);
        let p_pc = submatrix(&p, &pk, &cov);
        let p_pp_inv = p_pp.clone().try_inverse().unwrap();
        let mu_prec = &p_pp_inv * &(-(&p_pc * &d));
        let sigma_prec = p_pp_inv;

        // Covariance form (textbook).
        let o_pp = submatrix(&omega, &pk, &pk);
        let o_pc = submatrix(&omega, &pk, &cov);
        let o_cc = submatrix(&omega, &cov, &cov);
        let o_cc_inv = o_cc.try_inverse().unwrap();
        let mu_cov = &o_pc * &o_cc_inv * &d;
        let sigma_cov = &o_pp - &o_pc * &o_cc_inv * o_pc.transpose();

        for i in 0..2 {
            assert!((mu_prec[i] - mu_cov[i]).abs() < 1e-10, "mu[{i}]");
            for j in 0..2 {
                assert!(
                    (sigma_prec[(i, j)] - sigma_cov[(i, j)]).abs() < 1e-10,
                    "sigma[{i},{j}]"
                );
            }
        }
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
