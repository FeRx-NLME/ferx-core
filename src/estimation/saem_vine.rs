//! SAEM with regular-vine copula + zero-mean flexible marginals (`omega_dist = vine`, #225).
//!
//! Separated cleanly from [`crate::estimation::saem`] so the `OmegaDist::Gaussian`
//! codepath remains 100% untouched bit-for-bit.
//!
//! Statistical architecture:
//! - **Zero-mean marginals**: every marginal family (`Normal`, `StudentT`, `SkewNormal`,
//!   `SkewT`) satisfies `E[eta_j] == 0` analytically, preserving location identifiability
//!   with `theta`.
//! - **Three-phase schedule**:
//!   1. *Gaussian warm-up* (`1 <= k <= K_select`): updates `Omega_equiv` via Robbins-Monro
//!      SA after `omega_burnin`, buffering the trailing `M <= 25` exploration `eta` draws
//!      (re-centred whenever mu-referencing moves `theta`).
//!   2. *Family & R-vine selection* (`k == K_select`): selects per-coordinate marginals
//!      (when `vine_marginals = auto`) and the Dissmann R-vine tree structure + rotated
//!      pair-copula families (`0/90/180/270` deg) by weighted BIC scaled to `N` subjects.
//!   3. *Robbins-Monro score/information SA* (`K_select < k <= K_1 + K_2`): updates all
//!      marginal and pair-copula parameters via stochastic approximation on their scores
//!      and observed Fisher information (`avg_score -> 0` as `gamma_k -> 0`).
//! - **Phi-frozen vine-score mu-referencing**: steps mu-referenced `theta` along the
//!   exact descent direction of `(1/N) sum_i prior_nll(eta_i - delta * e_j)` with
//!   backtracking and immediate `eta` re-centring.
//! - **Multi-start Vine MAP EBEs**: optimizes `obs_nll + prior_nll_vine` from multiple
//!   starting points per subject, checks posterior Hessian positive-definiteness and
//!   secondary local modes (`W_VINE_MULTIMODAL_EBE`), and populates `(n_obs_i x n_eta)`
//!   observation sensitivity Jacobians for downstream CWRES and Vine QMC IS.

use crate::estimation::inner_optimizer::{cacheable_schedule, ebe_h_matrix_of};
use crate::estimation::mstep_newton::NewtonTally;
use crate::estimation::outer_optimizer::{pop_nll, OuterResult};
use crate::estimation::parameterization::{
    pack_params, packed_fixed_mask, theta_packs_log, unpack_params,
};
use crate::estimation::saem::{
    cholesky_perturbation_into, classify_mu_ref_pairs, componentwise_sweeps, damp_mstep,
    damp_mstep_sigma_variance, mh_steps_report, resolve_mstep_optimizer, resolve_n_mh_steps,
    saem_final_ofv_report, sanitize_mstep_damping, sanitize_scale_deadband, sigma_mstep_sa_steps,
    theta_sigma_mstep_light, MhScratch, SAEM_DEFAULT_SEED, SAEM_OMEGA_DIAG_FLOOR,
};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{
    individual_nll_iov_with_scratch_and_schedule, obs_nll_subject_into_with_schedule,
};
use crate::stats::random_effects::RandomEffectDistribution;
use crate::stats::vine_copula::VineCopulaOmega;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

const VINE_SELECT_BUFFER_MAX: usize = 25;
const MSTEP_DEFAULT_DAMPING: f64 = 0.25;

/// Construct an `OmegaMatrix` whose Gaussian prior contribution `0.5 * (eta' inv eta + log_det)`
/// is identically `0.0`, allowing `MhScratch::score` and `individual_nll_iov_with_scratch_and_schedule`
/// to evaluate all observation channels (including JIT plans, survival, categorical, M3, IOV kappa)
/// while we add the exact `vine.prior_nll(eta)` separately.
pub(crate) fn flat_eta_omega(n_eta: usize, eta_names: &[String]) -> OmegaMatrix {
    let mut om = OmegaMatrix::from_diagonal(&vec![1.0; n_eta], eta_names.to_vec());
    om.inv.fill(0.0);
    om.log_det = 0.0;
    om
}

/// Evaluate exact posterior NLL `obs_nll + vine.prior_nll(eta)` (+ kappa prior when IOV).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn score_subject_vine(
    mh_scratch: &mut MhScratch, model: &CompiledModel, subject: &Subject, theta: &[f64],
    eta: &[f64], flat_omega: &OmegaMatrix, vine: &VineCopulaOmega, sigma_values: &[f64],
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> f64 {
    let obs_plus_kappa = if let Some((kappas, omega_iov)) = kappas_opt {
        individual_nll_iov_with_scratch_and_schedule(
            model,
            subject,
            theta,
            eta,
            kappas,
            flat_omega,
            Some(omega_iov),
            sigma_values,
            mh_scratch.pk(),
            schedule,
        )
    } else {
        mh_scratch.score(model, subject, theta, eta, flat_omega, sigma_values, schedule)
    };
    obs_plus_kappa + vine.prior_nll(eta)
}

/// Symmetric block random-walk MH steps under the exact `VineCopulaOmega` prior.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps_vine(
    eta: &mut [f64], nll_current: f64, subject: &Subject, model: &CompiledModel, theta: &[f64],
    flat_omega: &OmegaMatrix, vine: &VineCopulaOmega, sigma_values: &[f64], step_scale: f64,
    rng: &mut impl Rng, n_steps: usize, scratch: &mut MhScratch,
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (usize, f64) {
    let n_eta = eta.len();
    let mut nll = nll_current;
    let mut n_accepted = 0usize;
    let chol = vine.proposal_chol();
    let mut z = DVector::<f64>::zeros(n_eta);
    let mut pert = DVector::<f64>::zeros(n_eta);
    let mut eta_prop = vec![0.0_f64; n_eta];

    for _ in 0..n_steps {
        for slot in z.iter_mut() {
            *slot = rng.sample(StandardNormal);
        }
        cholesky_perturbation_into(chol, &z, &mut pert);
        for j in 0..n_eta {
            eta_prop[j] = if vine.is_zero_fixed(j) { 0.0 } else { eta[j] + step_scale * pert[j] };
        }
        let nll_prop = score_subject_vine(
            scratch,
            model,
            subject,
            theta,
            &eta_prop,
            flat_omega,
            vine,
            sigma_values,
            schedule,
            kappas_opt,
        );
        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            eta.copy_from_slice(&eta_prop);
            nll = nll_prop;
            n_accepted += 1;
        }
    }
    (n_accepted, nll)
}

/// Componentwise Metropolis-within-Gibbs sweeps under the exact `VineCopulaOmega` prior.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_steps_componentwise_vine(
    eta: &mut [f64], nll_current: f64, subject: &Subject, model: &CompiledModel, theta: &[f64],
    flat_omega: &OmegaMatrix, vine: &VineCopulaOmega, sigma_values: &[f64], step_scales: &[f64],
    cw_sd: &[f64], rng: &mut impl Rng, n_sweeps: usize, scratch: &mut MhScratch,
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
    kappas_opt: Option<(&[Vec<f64>], &OmegaMatrix)>,
) -> (Vec<usize>, usize, f64) {
    let n_eta = eta.len();
    let mut nll = nll_current;
    let mut per_eta_accepted = vec![0usize; n_eta];

    for _ in 0..n_sweeps {
        for j in 0..n_eta {
            if vine.is_zero_fixed(j) {
                continue;
            }
            let z: f64 = rng.sample(StandardNormal);
            let old_j = eta[j];
            eta[j] = old_j + step_scales[j] * cw_sd[j] * z;
            let nll_prop = score_subject_vine(
                scratch,
                model,
                subject,
                theta,
                eta,
                flat_omega,
                vine,
                sigma_values,
                schedule,
                kappas_opt,
            );
            let log_u: f64 = rng.random::<f64>().ln();
            if log_u < nll - nll_prop {
                nll = nll_prop;
                per_eta_accepted[j] += 1;
            } else {
                eta[j] = old_j;
            }
        }
    }
    (per_eta_accepted, n_sweeps * n_eta, nll)
}

/// Per-occasion kappa MH sweeps holding `vine.prior_nll(eta)` fixed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mh_kappa_steps_vine(
    kappas: &mut [Vec<f64>], nll_current: f64, subject: &Subject, model: &CompiledModel,
    theta: &[f64], eta: &[f64], flat_omega: &OmegaMatrix, vine_prior_nll: f64,
    omega_iov: &OmegaMatrix, sigma_values: &[f64], step_scale: f64, rng: &mut impl Rng,
    schedule: Option<&crate::pk::event_driven::EventSchedule>, pk_scratch: &mut EventPkParams,
) -> (usize, usize, f64) {
    let n_occ = kappas.len();
    let n_kappa = model.n_kappa;
    if n_occ == 0 || n_kappa == 0 {
        return (0, 0, nll_current);
    }
    let mut nll = nll_current;
    let mut accepted = 0usize;
    let mut proposed = 0usize;
    let mut z = DVector::<f64>::zeros(n_kappa);
    let mut pert = DVector::<f64>::zeros(n_kappa);
    let mut old_occ = vec![0.0_f64; n_kappa];

    for occ in 0..n_occ {
        for slot in z.iter_mut() {
            *slot = rng.sample(StandardNormal);
        }
        cholesky_perturbation_into(&omega_iov.chol, &z, &mut pert);
        old_occ.copy_from_slice(&kappas[occ]);
        for m in 0..n_kappa {
            kappas[occ][m] = old_occ[m] + step_scale * pert[m];
        }
        let nll_prop = individual_nll_iov_with_scratch_and_schedule(
            model,
            subject,
            theta,
            eta,
            kappas,
            flat_omega,
            Some(omega_iov),
            sigma_values,
            pk_scratch,
            schedule,
        ) + vine_prior_nll;
        proposed += 1;
        let log_u: f64 = rng.random::<f64>().ln();
        if log_u < nll - nll_prop {
            nll = nll_prop;
            accepted += 1;
        } else {
            kappas[occ].copy_from_slice(&old_occ);
        }
    }
    (accepted, proposed, nll)
}

/// Radical inverse in prime base `b` for deterministic low-discrepancy multi-start points.
fn halton_scalar(mut index: usize, base: usize) -> f64 {
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    let inv_b = 1.0 / (base as f64);
    while index > 0 {
        f *= inv_b;
        r += f * ((index % base) as f64);
        index /= base;
    }
    r.clamp(0.02, 0.98)
}

/// Damped BFGS optimizer for one subject's Vine MAP EBE (`obs_nll + vine.prior_nll`).
#[allow(clippy::too_many_arguments)]
fn solve_vine_map_from_start(
    start: &[f64], active_dims: &[usize], full_len: usize, max_iter: usize, tol: f64,
    eval_nll: &mut dyn FnMut(&[f64]) -> f64, eval_grad: &mut dyn FnMut(&[f64], &mut [f64]),
) -> (Vec<f64>, f64) {
    let d = active_dims.len();
    let mut x_full = start.to_vec();
    let f0 = eval_nll(&x_full);
    if d == 0 || !f0.is_finite() {
        return (x_full, f0);
    }

    let mut x = vec![0.0_f64; d];
    for (m, &idx) in active_dims.iter().enumerate() {
        x[m] = start[idx];
    }
    let mut f = f0;
    let mut g_full = vec![0.0_f64; full_len];
    eval_grad(&x_full, &mut g_full);
    let mut g: Vec<f64> = active_dims.iter().map(|&idx| g_full[idx]).collect();
    let mut inv_h = DMatrix::<f64>::identity(d, d);
    let mut x_trial_full = x_full.clone();

    for _ in 0..max_iter {
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !gnorm.is_finite() || gnorm <= tol {
            break;
        }
        let g_vec = DVector::from_column_slice(&g);
        let mut dir = -&inv_h * &g_vec;
        let slope = dir.dot(&g_vec);
        if !(slope < 0.0) {
            inv_h = DMatrix::identity(d, d);
            dir = -&g_vec;
        }
        // Cap step norm to avoid jumping into numerical overflow regions of exp(eta).
        let dir_norm = dir.norm();
        if dir_norm > 2.5 {
            dir *= 2.5 / dir_norm;
        }
        let slope_used = dir.dot(&g_vec);

        let mut alpha = 1.0_f64;
        let mut accepted = false;
        let mut f_new = f;
        for _ in 0..14 {
            for (m, &idx) in active_dims.iter().enumerate() {
                x_trial_full[idx] = (x[m] + alpha * dir[m]).clamp(-12.0, 12.0);
            }
            let ft = eval_nll(&x_trial_full);
            if ft.is_finite() && ft <= f + 1e-4 * alpha * slope_used {
                f_new = ft;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }

        eval_grad(&x_trial_full, &mut g_full);
        let g_new: Vec<f64> = active_dims.iter().map(|&idx| g_full[idx]).collect();
        let s = DVector::from_fn(d, |m, _| x_trial_full[active_dims[m]] - x[m]);
        let y = DVector::from_fn(d, |m, _| g_new[m] - g[m]);
        let sy = s.dot(&y);
        let hy = &inv_h * &y;
        let yhy = y.dot(&hy);
        if sy > 1e-12 && sy.is_finite() && yhy.is_finite() {
            let rho = 1.0 / sy;
            let i_mat = DMatrix::<f64>::identity(d, d);
            let left = &i_mat - rho * (&s * y.transpose());
            let right = &i_mat - rho * (&y * s.transpose());
            inv_h = left * &inv_h * right + rho * (&s * s.transpose());
        }

        for (m, &idx) in active_dims.iter().enumerate() {
            x[m] = x_trial_full[idx];
        }
        x_full.copy_from_slice(&x_trial_full);
        f = f_new;
        g = g_new;
        if s.norm() <= tol * 0.1 {
            break;
        }
    }
    (x_full, f)
}

struct SubjectVineEbe {
    eta_hat: DVector<f64>,
    kappas: Vec<DVector<f64>>,
    h_matrix: DMatrix<f64>,
    multimodal_or_non_pd: bool,
}

/// Multi-start Vine MAP EBE solver + multimodality / Hessian PD diagnostics across all subjects.
fn compute_vine_map_ebes(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    vine: &VineCopulaOmega, warm_etas: &[Vec<f64>], warm_kappas: &[Vec<Vec<f64>>],
    options: &FitOptions,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, Vec<Vec<DVector<f64>>>, Option<String>) {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let flat_omega = flat_eta_omega(n_eta, &params.omega.eta_names);
    let active_eta: Vec<usize> = (0..n_eta).filter(|&j| !vine.is_zero_fixed(j)).collect();
    let primes = [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    let max_iter = options.inner_maxiter.max(60);
    let tol = options.inner_tol.clamp(1e-8, 1e-4);

    let solved: Vec<SubjectVineEbe> =
        crate::par::map_init(n_subjects, MhScratch::default, |mh_scratch, i| {
            let subject = &population.subjects[i];
            mh_scratch.begin_subject(model, subject, &params.theta, n_eta);
            let n_occ = if n_kappa > 0 { warm_kappas.get(i).map_or(0, |ks| ks.len()) } else { 0 };
            let schedule = if n_occ == 0 { cacheable_schedule(model, subject) } else { None };
            let full_len = n_eta + n_occ * n_kappa;
            let mut active_dims = active_eta.clone();
            if n_occ > 0 {
                for occ in 0..n_occ {
                    for m in 0..n_kappa {
                        let fixed = params.kappa_fixed.get(m).copied().unwrap_or(false);
                        if !fixed {
                            active_dims.push(n_eta + occ * n_kappa + m);
                        }
                    }
                }
            }

            // Build starts: [SAEM chain state, zero, 4 low-discrepancy vine draws].
            let mut starts: Vec<Vec<f64>> = Vec::with_capacity(6);
            let mut warm = vec![0.0_f64; full_len];
            warm[..n_eta].copy_from_slice(&warm_etas[i]);
            for j in (0..n_eta).filter(|&j| vine.is_zero_fixed(j)) {
                warm[j] = 0.0;
            }
            if n_occ > 0 {
                for (occ, ko) in warm_kappas[i].iter().enumerate() {
                    warm[n_eta + occ * n_kappa..n_eta + (occ + 1) * n_kappa].copy_from_slice(ko);
                }
            }
            starts.push(warm);
            starts.push(vec![0.0_f64; full_len]);
            for s_idx in 1..=4 {
                let u: Vec<f64> = (0..n_eta)
                    .map(|j| halton_scalar(i * 7 + s_idx, primes[j % primes.len()]))
                    .collect();
                let mut v = vec![0.0_f64; full_len];
                let e = vine.draw_eta_from_uniforms(&u);
                v[..n_eta].copy_from_slice(&e);
                starts.push(v);
            }

            let eval_at = |x: &[f64], sc: &mut MhScratch| -> f64 {
                if n_occ > 0 {
                    let iov = params.omega_iov.as_ref().expect("omega_iov present for n_occ > 0");
                    let kaps: Vec<&[f64]> = (0..n_occ)
                        .map(|occ| &x[n_eta + occ * n_kappa..n_eta + (occ + 1) * n_kappa])
                        .collect();
                    individual_nll_iov_with_scratch_and_schedule(
                        model,
                        subject,
                        &params.theta,
                        &x[..n_eta],
                        &kaps,
                        &flat_omega,
                        Some(iov),
                        &params.sigma.values,
                        sc.pk(),
                        None,
                    ) + vine.prior_nll(&x[..n_eta])
                } else {
                    sc.score(
                        model,
                        subject,
                        &params.theta,
                        &x[..n_eta],
                        &flat_omega,
                        &params.sigma.values,
                        schedule.as_ref(),
                    ) + vine.prior_nll(&x[..n_eta])
                }
            };

            // Screen the 4 Halton starts and optimize [warm, zero, best_halton_1, best_halton_2].
            let mut halton_scored: Vec<(f64, Vec<f64>)> = starts
                .drain(2..)
                .map(|st| {
                    let v = eval_at(&st, mh_scratch);
                    (v, st)
                })
                .collect();
            halton_scored
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, st) in halton_scored.into_iter().take(2) {
                starts.push(st);
            }

            let mut modes: Vec<(Vec<f64>, f64)> = Vec::with_capacity(starts.len());
            for st in &starts {
                let (sol, nll) = {
                    let mut nll_fn = |x: &[f64]| eval_at(x, mh_scratch);
                    // Split closures by borrowing mh_scratch inside each call sequentially.
                    let _ = &mut nll_fn;
                    solve_vine_map_single_scratch(
                        st,
                        &active_dims,
                        full_len,
                        n_eta,
                        n_occ,
                        n_kappa,
                        max_iter,
                        tol,
                        model,
                        subject,
                        params,
                        &flat_omega,
                        vine,
                        schedule.as_ref(),
                        mh_scratch,
                    )
                };
                if nll.is_finite() {
                    modes.push((sol, nll));
                }
            }

            modes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let (best_x, best_nll) =
                modes.first().cloned().unwrap_or_else(|| (vec![0.0_f64; full_len], 0.0));

            // Check secondary modes & posterior Hessian positive-definiteness on scored subjects.
            let has_obs = !subject.observations.is_empty() || {
                #[cfg(feature = "survival")]
                {
                    !subject.obs_records.is_empty()
                }
                #[cfg(not(feature = "survival"))]
                {
                    false
                }
            };
            let mut multimodal_or_non_pd = false;
            if has_obs && !active_eta.is_empty() {
                for (alt_x, alt_nll) in modes.iter().skip(1) {
                    if alt_nll - best_nll <= 3.0 {
                        let dist_sq: f64 = active_eta
                            .iter()
                            .map(|&j| {
                                let d = alt_x[j] - best_x[j];
                                d * d
                            })
                            .sum();
                        if dist_sq.sqrt() > 0.15 {
                            multimodal_or_non_pd = true;
                            break;
                        }
                    }
                }
                if !multimodal_or_non_pd {
                    // Check FD posterior Hessian PD on active BSV coordinates at `best_x`.
                    let d_act = active_eta.len();
                    let mut h_mat = DMatrix::<f64>::zeros(d_act, d_act);
                    let eps = 1e-4_f64;
                    let mut xp = best_x.clone();
                    for (a, &ja) in active_eta.iter().enumerate() {
                        let step_a = eps * (1.0 + best_x[ja].abs());
                        xp[ja] = best_x[ja] + step_a;
                        let fp = eval_at(&xp, mh_scratch);
                        xp[ja] = best_x[ja] - step_a;
                        let fm = eval_at(&xp, mh_scratch);
                        xp[ja] = best_x[ja];
                        h_mat[(a, a)] = (fp - 2.0 * best_nll + fm) / (step_a * step_a);
                        for (b, &jb) in active_eta.iter().enumerate().skip(a + 1) {
                            let step_b = eps * (1.0 + best_x[jb].abs());
                            xp[ja] = best_x[ja] + step_a;
                            xp[jb] = best_x[jb] + step_b;
                            let fpp = eval_at(&xp, mh_scratch);
                            xp[jb] = best_x[jb] - step_b;
                            let fpm = eval_at(&xp, mh_scratch);
                            xp[ja] = best_x[ja] - step_a;
                            let fmm = eval_at(&xp, mh_scratch);
                            xp[jb] = best_x[jb] + step_b;
                            let fmp = eval_at(&xp, mh_scratch);
                            xp[ja] = best_x[ja];
                            xp[jb] = best_x[jb];
                            let h_ab = (fpp - fpm - fmp + fmm) / (4.0 * step_a * step_b);
                            h_mat[(a, b)] = h_ab;
                            h_mat[(b, a)] = h_ab;
                        }
                    }
                    let min_eig = h_mat
                        .symmetric_eigen()
                        .eigenvalues
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    if !(min_eig > 1e-6) {
                        multimodal_or_non_pd = true;
                    }
                }
            }

            let eta_hat = DVector::from_column_slice(&best_x[..n_eta]);
            let kappas: Vec<DVector<f64>> = (0..n_occ)
                .map(|occ| {
                    DVector::from_column_slice(
                        &best_x[n_eta + occ * n_kappa..n_eta + (occ + 1) * n_kappa],
                    )
                })
                .collect();
            let h_matrix = ebe_h_matrix_of(model, subject, params, &best_x[..n_eta], &kappas);

            SubjectVineEbe { eta_hat, kappas, h_matrix, multimodal_or_non_pd }
        });

    let mut flagged_ids = Vec::new();
    let mut eta_hats = Vec::with_capacity(n_subjects);
    let mut h_matrices = Vec::with_capacity(n_subjects);
    let mut final_kappas = Vec::with_capacity(n_subjects);

    for (i, s) in solved.into_iter().enumerate() {
        if s.multimodal_or_non_pd {
            flagged_ids.push(population.subjects[i].id.clone());
        }
        eta_hats.push(s.eta_hat);
        h_matrices.push(s.h_matrix);
        final_kappas.push(s.kappas);
    }

    let warning = if !flagged_ids.is_empty() {
        let shown: Vec<&str> = flagged_ids.iter().take(10).map(String::as_str).collect();
        let more = if flagged_ids.len() > 10 {
            format!(" (+{} more)", flagged_ids.len() - 10)
        } else {
            String::new()
        };
        Some(format!(
            "W_VINE_MULTIMODAL_EBE: {} of {} subject(s) exhibited secondary local posterior \
             modes (within 3 NLL units) or a non-positive-definite posterior Hessian under the \
             fitted vine prior [subjects: {}{}]. The lowest-NLL mode was selected; prefer \
             simulation/QMC diagnostics (NPDE, `conddist = true`) over point-EBE summaries.",
            flagged_ids.len(),
            n_subjects,
            shown.join(", "),
            more
        ))
    } else {
        None
    };

    (eta_hats, h_matrices, final_kappas, warning)
}

#[allow(clippy::too_many_arguments)]
fn solve_vine_map_single_scratch(
    start: &[f64], active_dims: &[usize], full_len: usize, n_eta: usize, n_occ: usize,
    n_kappa: usize, max_iter: usize, tol: f64, model: &CompiledModel, subject: &Subject,
    params: &ModelParameters, flat_omega: &OmegaMatrix, vine: &VineCopulaOmega,
    schedule: Option<&crate::pk::event_driven::EventSchedule>, mh_scratch: &mut MhScratch,
) -> (Vec<f64>, f64) {
    let cell = std::cell::RefCell::new(mh_scratch);
    let eval_point = |x: &[f64]| -> f64 {
        let mut sc = cell.borrow_mut();
        if n_occ > 0 {
            let iov = params.omega_iov.as_ref().expect("omega_iov present");
            let kaps: Vec<&[f64]> = (0..n_occ)
                .map(|occ| &x[n_eta + occ * n_kappa..n_eta + (occ + 1) * n_kappa])
                .collect();
            individual_nll_iov_with_scratch_and_schedule(
                model,
                subject,
                &params.theta,
                &x[..n_eta],
                &kaps,
                flat_omega,
                Some(iov),
                &params.sigma.values,
                sc.pk(),
                None,
            ) + vine.prior_nll(&x[..n_eta])
        } else {
            sc.score(
                model,
                subject,
                &params.theta,
                &x[..n_eta],
                flat_omega,
                &params.sigma.values,
                schedule,
            ) + vine.prior_nll(&x[..n_eta])
        }
    };
    let mut nll_fn = |x: &[f64]| eval_point(x);
    let mut grad_fn = |x: &[f64], out: &mut [f64]| {
        out.fill(0.0);
        // Exact analytic + interior-u gradient for the vine prior, plus central FD for obs_nll!
        let pg = vine.prior_nll_grad(&x[..n_eta]);
        let mut work = x.to_vec();
        for &idx in active_dims {
            let h = 1e-5 * (1.0 + x[idx].abs());
            work[idx] = x[idx] + h;
            let fp = if idx < n_eta {
                eval_point(&work) - vine.prior_nll(&work[..n_eta])
            } else {
                eval_point(&work)
            };
            work[idx] = x[idx] - h;
            let fm = if idx < n_eta {
                eval_point(&work) - vine.prior_nll(&work[..n_eta])
            } else {
                eval_point(&work)
            };
            work[idx] = x[idx];
            let obs_g = (fp - fm) / (2.0 * h);
            let total_g = if idx < n_eta { obs_g + pg[idx] } else { obs_g };
            out[idx] = if total_g.is_finite() { total_g } else { 0.0 };
        }
    };
    solve_vine_map_from_start(
        start,
        active_dims,
        full_len,
        max_iter,
        tol,
        &mut nll_fn,
        &mut grad_fn,
    )
}

/// Louis / BHHH empirical score covariance `S^-1` on the free `(theta, sigma)` coordinates
/// under the exact vine marginal likelihood, embedded into the full packed covariance matrix.
fn compute_vine_louis_covariance(
    model: &CompiledModel, population: &Population, params: &ModelParameters,
    eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>], kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> (Option<DMatrix<f64>>, Option<CovarianceMethod>, f64, Vec<String>) {
    if !options.run_covariance_step || crate::cancel::is_cancelled(&options.cancel) {
        return (None, None, 0.0, Vec::new());
    }
    let start = Instant::now();
    let mut held_template = params.clone();
    // Gaussian `omega` is replaced by `vine_dist` (whose marginal + copula score blocks are
    // appended explicitly below so cross-information with `(theta, sigma, omega_iov)` is preserved).
    held_template.omega_fixed.fill(true);
    held_template.residual_correlation_fixed.fill(true);
    let packed = pack_params(params);
    let mask = packed_fixed_mask(&held_template);
    let n_packed = packed.len();
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let n_eta = model.n_eta;
    let n_iov = model.n_kappa;

    let omega_packed_len = if params.omega.diagonal { n_eta } else { n_eta * (n_eta + 1) / 2 };
    let sigma_offset = n_theta + omega_packed_len;
    let res_corr_offset = sigma_offset + n_sigma;
    let iov_offset = res_corr_offset + params.residual_correlations.len();
    let free_indices: Vec<usize> =
        (0..n_packed).filter(|&p| !mask.get(p).copied().unwrap_or(true)).collect();
    if free_indices.is_empty() {
        return (
            Some(DMatrix::zeros(n_packed, n_packed)),
            Some(CovarianceMethod::CrossProduct),
            start.elapsed().as_secs_f64(),
            Vec::new(),
        );
    }

    let vine_ref = params.vine_dist.as_deref();
    let n_vine_free = vine_ref.map_or(0, VineCopulaOmega::n_free_params);
    let n_free = free_indices.len();
    let n_total = n_free + n_vine_free;

    // Pre-unpack perturbed `omega_iov` matrices outside the parallel subject loop so IOV score
    // columns cost zero matrix decompositions per subject/draw.
    let iov_perturbations: Vec<Option<(OmegaMatrix, OmegaMatrix, f64)>> = free_indices
        .iter()
        .map(|&p_idx| {
            if p_idx < iov_offset || params.omega_iov.is_none() {
                return None;
            }
            let h = 1e-5 * (1.0 + packed[p_idx].abs());
            let mut p_up = packed.clone();
            let mut p_dn = packed.clone();
            p_up[p_idx] += h;
            p_dn[p_idx] -= h;
            let om_up = unpack_params(&p_up, &held_template).omega_iov?;
            let om_dn = unpack_params(&p_dn, &held_template).omega_iov?;
            Some((om_up, om_dn, h))
        })
        .collect();

    let k_cov = 200usize;
    let seed = options.imp_seed.or(options.saem_seed).unwrap_or(42) ^ 0x51CE_C07A;
    let packs_log: Vec<bool> = params.theta_lower.iter().map(|&lo| theta_packs_log(lo)).collect();

    let scores: Vec<Option<DVector<f64>>> =
        crate::par::map_init(population.subjects.len(), EventPkParams::default, |scratch, i| {
            let subject = &population.subjects[i];
            if subject.observations.is_empty() {
                #[cfg(feature = "survival")]
                if subject.obs_records.is_empty() {
                    return None;
                }
                #[cfg(not(feature = "survival"))]
                return None;
            }
            let kap = kappas.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            let subj_seed = crate::rng::stream_seed(seed, &[i as u64]);
            let (draws, weights) =
                crate::estimation::importance_sampling::vine_posterior_weighted_draws(
                    model,
                    subject,
                    params,
                    &eta_hats[i],
                    &h_matrices[i],
                    kap,
                    k_cov,
                    subj_seed,
                    scratch,
                )?;
            let n_occ = if n_iov > 0 && !kap.is_empty() {
                crate::stats::stacked_layout::group_count(subject)
            } else {
                0
            };
            let schedule = if n_occ == 0 { cacheable_schedule(model, subject) } else { None };

            let eval_obs = |th: &[f64], sg: &[f64], b: &[f64], sc: &mut EventPkParams| -> f64 {
                if n_occ > 0 {
                    crate::estimation::impmap::draw_obs_nll(model, subject, th, sg, b, None, sc)
                } else {
                    obs_nll_subject_into_with_schedule(
                        model,
                        subject,
                        th,
                        sg,
                        model.residual_cov_theta(th),
                        &b[..n_eta],
                        sc,
                        schedule.as_ref(),
                    )
                }
            };

            let eval_iov_log_p = |om: &OmegaMatrix, b: &[f64]| -> f64 {
                if n_occ == 0 || n_iov == 0 || !om.log_det.is_finite() {
                    return 0.0;
                }
                let mut quad = 0.0_f64;
                for o in 0..n_occ {
                    let base = n_eta + o * n_iov;
                    for r in 0..n_iov {
                        let mut row = 0.0;
                        for c in 0..n_iov {
                            row += om.inv[(r, c)] * b[base + c];
                        }
                        quad += row * b[base + r];
                    }
                }
                -0.5 * (n_occ as f64) * om.log_det - 0.5 * quad
            };

            let mut s_i = DVector::<f64>::zeros(n_total);
            let mut th_work = params.theta.clone();
            let mut sg_work = params.sigma.values.clone();

            for (f_pos, &p_idx) in free_indices.iter().enumerate() {
                let h = 1e-5 * (1.0 + packed[p_idx].abs());
                let mut acc = 0.0_f64;
                if p_idx < n_theta {
                    let orig = th_work[p_idx];
                    let p0 = packed[p_idx];
                    let th_p = if packs_log[p_idx] { (p0 + h).exp() } else { p0 + h };
                    let th_m = if packs_log[p_idx] { (p0 - h).exp() } else { p0 - h };
                    for (b, &w) in draws.iter().zip(weights.iter()) {
                        if w <= 1e-15 {
                            continue;
                        }
                        th_work[p_idx] = th_p;
                        let np = eval_obs(&th_work, &sg_work, b, scratch);
                        th_work[p_idx] = th_m;
                        let nm = eval_obs(&th_work, &sg_work, b, scratch);
                        let deriv = (np - nm) / (2.0 * h);
                        if deriv.is_finite() {
                            acc -= w * deriv;
                        }
                    }
                    th_work[p_idx] = orig;
                } else if p_idx < res_corr_offset {
                    let s_idx = p_idx - sigma_offset;
                    let orig = sg_work[s_idx];
                    let p0 = packed[p_idx];
                    let sg_p = (p0 + h).exp();
                    let sg_m = (p0 - h).exp();
                    for (b, &w) in draws.iter().zip(weights.iter()) {
                        if w <= 1e-15 {
                            continue;
                        }
                        sg_work[s_idx] = sg_p;
                        let np = eval_obs(&th_work, &sg_work, b, scratch);
                        sg_work[s_idx] = sg_m;
                        let nm = eval_obs(&th_work, &sg_work, b, scratch);
                        let deriv = (np - nm) / (2.0 * h);
                        if deriv.is_finite() {
                            acc -= w * deriv;
                        }
                    }
                    sg_work[s_idx] = orig;
                } else if let Some((om_up, om_dn, h_iov)) = &iov_perturbations[f_pos] {
                    for (b, &w) in draws.iter().zip(weights.iter()) {
                        if w <= 1e-15 {
                            continue;
                        }
                        let lp = eval_iov_log_p(om_up, b);
                        let lm = eval_iov_log_p(om_dn, b);
                        let deriv = (lp - lm) / (2.0 * *h_iov);
                        if deriv.is_finite() {
                            acc += w * deriv;
                        }
                    }
                }
                s_i[f_pos] = acc;
            }

            if let Some(vine) = vine_ref {
                if n_vine_free > 0 {
                    for (b, &w) in draws.iter().zip(weights.iter()) {
                        if w <= 1e-15 {
                            continue;
                        }
                        let vs = vine.parameter_score(&b[..n_eta]);
                        for (q, &val) in vs.iter().enumerate() {
                            s_i[n_free + q] += w * val;
                        }
                    }
                }
            }
            Some(s_i)
        });

    let mut s_full = DMatrix::<f64>::zeros(n_total, n_total);
    let mut n_contrib = 0usize;
    for s_opt in scores.into_iter().flatten() {
        s_full += &s_opt * s_opt.transpose();
        n_contrib += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    if n_contrib < n_free {
        return (
            None,
            None,
            elapsed,
            vec![format!(
                "Covariance step failed: empirical vine score matrix S has only {n_contrib} \
                 contributing subjects for {n_free} free fixed-effect parameters."
            )],
        );
    }

    let mut cov_notes = Vec::new();
    let inv_free =
        if n_vine_free > 0 && n_contrib > n_total {
            // Unconditional joint BHHH inverse: invert the full `(packed, vine)` cross-product S and
            // extract the `(packed, packed)` Schur-complement block so `(theta, sigma, omega_iov)` SEs
            // reflect co-estimation uncertainty with the marginal and pair-copula parameters.
            let eig = s_full.symmetric_eigen();
            let max_ev = eig.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
            if !(max_ev > 1e-12) {
                return (
                None,
                None,
                elapsed,
                vec!["Covariance step failed: empirical vine score matrix S is zero or non-finite."
                    .to_string()],
            );
            }
            let floor = (1e-8 * max_ev).max(1e-10);
            let mut inv_all = DMatrix::<f64>::zeros(n_total, n_total);
            for k in 0..n_total {
                let lam = eig.eigenvalues[k].max(floor);
                let col = eig.eigenvectors.column(k);
                inv_all += (1.0 / lam) * (&col * col.transpose());
            }
            inv_all.view((0, 0), (n_free, n_free)).into_owned()
        } else {
            if n_vine_free > 0 {
                cov_notes.push(format!(
                "SAEM vine covariance: {n_contrib} informative subjects cannot jointly span all \
                 {n_total} model + vine parameters ({n_free} fixed/residual/IOV + {n_vine_free} \
                 vine); reporting standard errors conditional on the fitted vine parameters."
            ));
            }
            let s_mat = s_full.view((0, 0), (n_free, n_free)).into_owned();
            let eig = s_mat.symmetric_eigen();
            let max_ev = eig.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
            if !(max_ev > 1e-12) {
                return (
                None,
                None,
                elapsed,
                vec!["Covariance step failed: empirical vine score matrix S is zero or non-finite."
                    .to_string()],
            );
            }
            let floor = (1e-8 * max_ev).max(1e-10);
            let mut inv_sub = DMatrix::<f64>::zeros(n_free, n_free);
            for k in 0..n_free {
                let lam = eig.eigenvalues[k].max(floor);
                let col = eig.eigenvectors.column(k);
                inv_sub += (1.0 / lam) * (&col * col.transpose());
            }
            inv_sub
        };

    let mut full_cov = DMatrix::<f64>::zeros(n_packed, n_packed);
    for (a, &pa) in free_indices.iter().enumerate() {
        for (b, &pb) in free_indices.iter().enumerate() {
            full_cov[(pa, pb)] = inv_free[(a, b)];
        }
    }
    (Some(full_cov), Some(CovarianceMethod::CrossProduct), elapsed, cov_notes)
}

/// Execute the SAEM-Copula (`omega_dist = vine`) algorithm and return an [`OuterResult`].
pub(crate) fn run_saem_vine(
    model: &CompiledModel, population: &Population, init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    if EstimationMethod::Saem.requires_random_effects() && n_eta == 0 {
        return Err("SAEM requires at least one random effect (n_eta = 0). \
             Use FOCE/FOCEI for fixed-effects-only models."
            .to_string());
    }
    if model.mixture.is_some() {
        return Err("`omega_dist = vine` is incompatible with `[mixture]` models.".to_string());
    }

    let n_kappa = model.n_kappa;
    let k1 = options.saem_n_exploration;
    if k1 == 0 {
        return Err("E_OMEGA_DIST_OPTIONS: `omega_dist = vine` requires `n_exploration >= 1` \
             (vine marginal and pair-copula BIC selection requires exploration draws)."
            .to_string());
    }
    let k2 = options.saem_n_convergence;
    let n_iter = k1 + k2;
    let omega_burnin = options.saem_omega_burnin.min(k1);
    let k_select = omega_burnin.max(k1 / 2).clamp(1, k1);

    let n_mh_steps =
        resolve_n_mh_steps(options.saem_n_mh_steps, population.n_obs(), n_subjects, n_eta);
    let n_cw_sweeps = componentwise_sweeps(n_mh_steps, n_eta);
    let adapt_interval = options.saem_adapt_interval.max(1);
    let rm_scale_adaptation =
        matches!(options.saem_scale_adaptation, ScaleAdaptation::RobbinsMonro);
    let verbose = options.verbose;
    let master_seed = options.saem_seed.unwrap_or(SAEM_DEFAULT_SEED);

    let mut warnings = Vec::new();
    let scale_deadband = match sanitize_scale_deadband(options.saem_scale_deadband) {
        None => options.saem_scale_deadband,
        Some(fixed) => {
            let (lo, hi) =
                options.saem_scale_deadband.expect("only a Some band can be substituted");
            let became = match fixed {
                None => "no dead band (the Robbins-Monro step runs every iteration)".to_string(),
                Some((a, b)) => format!("{a},{b}"),
            };
            warnings.push(format!(
                "SAEM: `scale_deadband` = {lo},{hi} is not a usable acceptance window — using \
                 {became} instead (#1449)."
            ));
            fixed
        }
    };

    if options.saem_n_leapfrog > 0 {
        // Substring "HMC is unavailable" lets `classify_warning` tag Info/gradient_fallback.
        warnings.push(
            "saem_n_leapfrog > 0 but HMC is unavailable with omega_dist = vine \
             (requires a Gaussian prior gradient); falling back to Metropolis-Hastings"
                .to_string(),
        );
    }

    let raw_damping = options.saem_mstep_damping.unwrap_or(MSTEP_DEFAULT_DAMPING);
    let mstep_damping = match sanitize_mstep_damping(raw_damping) {
        None => raw_damping,
        Some(fixed) => {
            warnings.push(format!(
                "SAEM: `mstep_damping` = {raw_damping} is outside (0, 1]; clamped to {fixed}."
            ));
            fixed
        }
    };

    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();
    let theta_packs_log_mask: Vec<bool> =
        init_params.theta_lower.iter().map(|&lo| theta_packs_log(lo)).collect();
    let pack_theta = |i: usize, v: f64| -> f64 {
        if theta_packs_log_mask[i] {
            v.max(1e-10).ln()
        } else {
            v
        }
    };
    let unpack_theta = |i: usize, pv: f64| -> f64 {
        if theta_packs_log_mask[i] {
            pv.exp()
        } else {
            pv
        }
    };

    let mut log_theta: Vec<f64> =
        (0..n_theta).map(|i| pack_theta(i, init_params.theta[i])).collect();
    let mut log_theta_lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
                log_theta[i]
            } else if theta_packs_log_mask[i] {
                init_params.theta_lower[i].max(1e-10).ln()
            } else {
                init_params.theta_lower[i].max(-1e9)
            }
        })
        .collect();
    let mut log_theta_upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
                log_theta[i]
            } else if theta_packs_log_mask[i] {
                init_params.theta_upper[i].min(1e9).max(1e-10).ln()
            } else {
                init_params.theta_upper[i].min(1e9)
            }
        })
        .collect();
    for i in 0..n_theta {
        if log_theta_lower[i] > log_theta_upper[i] {
            std::mem::swap(&mut log_theta_lower[i], &mut log_theta_upper[i]);
        }
        log_theta[i] = log_theta[i].clamp(log_theta_lower[i], log_theta_upper[i]);
    }

    let mut log_sigma: Vec<f64> =
        init_params.sigma.values.iter().map(|&s| s.max(1e-8).ln()).collect();
    let log_sigma_lower: Vec<f64> = (0..n_sigma)
        .map(|i| {
            if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
                log_sigma[i]
            } else {
                -8.0
            }
        })
        .collect();
    let log_sigma_upper: Vec<f64> = (0..n_sigma)
        .map(|i| {
            if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
                log_sigma[i]
            } else {
                5.0
            }
        })
        .collect();

    let mu_info = classify_mu_ref_pairs(model, &init_params.theta_lower);
    let mu_ref_pairs: Vec<(usize, usize, bool)> = if options.mu_referencing {
        let mut pairs: Vec<(usize, usize, bool)> =
            mu_info.eligible.iter().map(|&(th, et)| (th, et, false)).collect();
        pairs.extend(mu_info.additive.iter().map(|&(th, et)| (th, et, true)));
        pairs
    } else {
        Vec::new()
    };
    let use_mu_ref = options.mu_referencing && !mu_ref_pairs.is_empty();
    let mut mstep_evals_saved: u64 = 0;

    let allow_non_gaussian_marginals = options.vine_marginals == VineMarginals::Auto;
    let mut dist = VineCopulaOmega::from_init_params(init_params, allow_non_gaussian_marginals);
    let flat_omega = flat_eta_omega(n_eta, &init_params.omega.eta_names);

    let mut omega_iov_cur = init_params.omega_iov.clone();
    let mut s2_iov = init_params
        .omega_iov
        .as_ref()
        .map(|om| om.matrix.clone())
        .unwrap_or_else(|| DMatrix::zeros(n_kappa, n_kappa));

    let schedules: Vec<Option<crate::pk::event_driven::EventSchedule>> =
        population.subjects.iter().map(|s| cacheable_schedule(model, s)).collect();

    let mut etas: Vec<Vec<f64>> = vec![vec![0.0_f64; n_eta]; n_subjects];
    let mut kappas: Vec<Vec<Vec<f64>>> = population
        .subjects
        .iter()
        .map(|s| {
            if n_kappa > 0 {
                let n_occ = crate::stats::stacked_layout::group_count(s);
                vec![vec![0.0_f64; n_kappa]; n_occ]
            } else {
                Vec::new()
            }
        })
        .collect();

    let mut step_block = vec![0.3_f64; n_subjects];
    let mut step_cw = vec![vec![1.0_f64; n_eta]; n_subjects];
    let mut step_kappa = vec![0.3_f64; n_subjects];
    let mut acc_b_win = vec![0usize; n_subjects];
    let mut prop_b_win = vec![0usize; n_subjects];
    let mut acc_c_win = vec![vec![0usize; n_eta]; n_subjects];
    let mut prop_c_win = vec![0usize; n_subjects];
    let mut acc_k_win = vec![0usize; n_subjects];
    let mut prop_k_win = vec![0usize; n_subjects];

    let mut tail_accepted = 0usize;
    let mut tail_proposed = 0usize;

    let mut select_buffer: VecDeque<Vec<Vec<f64>>> =
        VecDeque::with_capacity(VINE_SELECT_BUFFER_MAX);

    // Robbins-Monro score & curvature accumulators for phi-frozen vine mu-referencing.
    let mut mu_avg_grad = vec![0.0_f64; mu_ref_pairs.len()];
    let mut mu_avg_curv = vec![1.0_f64; mu_ref_pairs.len()];

    let theta_free_any = (0..n_theta).any(|i| log_theta_lower[i] < log_theta_upper[i]);
    let (mstep_optimizer, mstep_opt_warn) = resolve_mstep_optimizer(
        model,
        options.mstep_optimizer,
        false,
        false,
        theta_free_any,
        false,
        "SAEM",
    );
    warnings.extend(mstep_opt_warn);
    let newton_tally = NewtonTally::default();

    if verbose {
        eprintln!(
            "SAEM (vine copula, marginals={}): {} subjects, {} ETAs, {} total iter ({} explore + {} converge, select at iter {})",
            options.vine_marginals.label(),
            n_subjects,
            n_eta,
            n_iter,
            k1,
            k2,
            k_select
        );
        eprintln!("{}", mh_steps_report(options.saem_n_mh_steps, n_mh_steps, n_cw_sweeps));
    }

    struct SubjStepOut {
        eta: Vec<f64>,
        kappa: Vec<Vec<f64>>,
        acc_b: usize,
        prop_b: usize,
        acc_c: Vec<usize>,
        prop_c_per_eta: usize,
        acc_k: usize,
        prop_k: usize,
    }

    let mid_k2_iter = (k2 > 0).then(|| k1 + (k2 / 2).max(1));
    let mut mid_phase2_state: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;

    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }

        let theta_cur: Vec<f64> = (0..n_theta).map(|i| unpack_theta(i, log_theta[i])).collect();
        let sigma_cur: Vec<f64> = log_sigma.iter().map(|&v| v.exp()).collect();
        let cw_sd: Vec<f64> =
            dist.marginal_sds().into_iter().map(|s| s.max(SAEM_OMEGA_DIAG_FLOOR.sqrt())).collect();
        let iov_ref = omega_iov_cur.as_ref();

        // ---- Step 1: Parallel two-kernel MH E-step under the exact Vine/Warmup prior ----
        let step_outs: Vec<SubjStepOut> =
            crate::par::map_init(n_subjects, MhScratch::default, |mh_scratch, i| {
                let subject = &population.subjects[i];
                let mut rng = StdRng::seed_from_u64(crate::rng::stream_seed(
                    master_seed,
                    &[k as u64, i as u64],
                ));
                mh_scratch.begin_subject(model, subject, &theta_cur, n_eta);
                let sched = schedules[i].as_ref();

                let mut eta_i = etas[i].clone();
                let mut kap_i = kappas[i].clone();
                let mut nll = score_subject_vine(
                    mh_scratch,
                    model,
                    subject,
                    &theta_cur,
                    &eta_i,
                    &flat_omega,
                    &dist,
                    &sigma_cur,
                    sched,
                    iov_ref.map(|iov| (kap_i.as_slice(), iov)),
                );

                let (acc_b, nll_b) = mh_steps_vine(
                    &mut eta_i,
                    nll,
                    subject,
                    model,
                    &theta_cur,
                    &flat_omega,
                    &dist,
                    &sigma_cur,
                    step_block[i],
                    &mut rng,
                    n_mh_steps,
                    mh_scratch,
                    sched,
                    iov_ref.map(|iov| (kap_i.as_slice(), iov)),
                );
                nll = nll_b;

                let (acc_c, prop_c_per_eta) = if n_cw_sweeps > 0 {
                    let (per_eta, _, nll_c) = mh_steps_componentwise_vine(
                        &mut eta_i,
                        nll,
                        subject,
                        model,
                        &theta_cur,
                        &flat_omega,
                        &dist,
                        &sigma_cur,
                        &step_cw[i],
                        &cw_sd,
                        &mut rng,
                        n_cw_sweeps,
                        mh_scratch,
                        sched,
                        iov_ref.map(|iov| (kap_i.as_slice(), iov)),
                    );
                    nll = nll_c;
                    (per_eta, n_cw_sweeps)
                } else {
                    (vec![0usize; n_eta], 0usize)
                };

                let (acc_k, prop_k) = if n_kappa > 0 {
                    if let Some(iov) = iov_ref {
                        let vp = dist.prior_nll(&eta_i);
                        let (ak, pk, _nll_k) = mh_kappa_steps_vine(
                            &mut kap_i,
                            nll,
                            subject,
                            model,
                            &theta_cur,
                            &eta_i,
                            &flat_omega,
                            vp,
                            iov,
                            &sigma_cur,
                            step_kappa[i],
                            &mut rng,
                            sched,
                            mh_scratch.pk(),
                        );
                        (ak, pk)
                    } else {
                        (0, 0)
                    }
                } else {
                    (0, 0)
                };

                SubjStepOut {
                    eta: eta_i,
                    kappa: kap_i,
                    acc_b,
                    prop_b: n_mh_steps,
                    acc_c,
                    prop_c_per_eta,
                    acc_k,
                    prop_k,
                }
            });

        for (i, out) in step_outs.into_iter().enumerate() {
            etas[i] = out.eta;
            kappas[i] = out.kappa;
            acc_b_win[i] += out.acc_b;
            prop_b_win[i] += out.prop_b;
            for j in 0..n_eta {
                acc_c_win[i][j] += out.acc_c[j];
            }
            prop_c_win[i] += out.prop_c_per_eta;
            acc_k_win[i] += out.acc_k;
            prop_k_win[i] += out.prop_k;

            if k > omega_burnin {
                let cw_acc_sum: usize = out.acc_c.iter().sum();
                let active_cnt = (0..n_eta).filter(|&j| !dist.is_zero_fixed(j)).count();
                tail_accepted += out.acc_b + cw_acc_sum;
                tail_proposed += out.prop_b + out.prop_c_per_eta * active_cnt;
            }

            if rm_scale_adaptation {
                let rm_gain = 0.5 / (k as f64).powf(0.6);
                let rate_b = out.acc_b as f64 / out.prop_b.max(1) as f64;
                let in_band = scale_deadband.is_some_and(|(lo, hi)| rate_b >= lo && rate_b <= hi);
                if !in_band {
                    step_block[i] =
                        (step_block[i] * (rm_gain * (rate_b - 0.40)).exp()).clamp(0.01, 5.0);
                }
                if out.prop_c_per_eta > 0 {
                    for j in 0..n_eta {
                        if dist.is_zero_fixed(j) {
                            continue;
                        }
                        let rate_c = out.acc_c[j] as f64 / out.prop_c_per_eta as f64;
                        let cw_in_band =
                            scale_deadband.is_some_and(|(lo, hi)| rate_c >= lo && rate_c <= hi);
                        if !cw_in_band {
                            step_cw[i][j] = (step_cw[i][j] * (rm_gain * (rate_c - 0.44)).exp())
                                .clamp(1e-6, 5.0);
                        }
                    }
                }
            }
        }

        if !rm_scale_adaptation && k % adapt_interval == 0 {
            for i in 0..n_subjects {
                let rate_b = acc_b_win[i] as f64 / prop_b_win[i].max(1) as f64;
                step_block[i] = if rate_b > 0.40 {
                    (step_block[i] * 1.1).min(5.0)
                } else {
                    (step_block[i] * 0.9).max(0.01)
                };
                acc_b_win[i] = 0;
                prop_b_win[i] = 0;

                if prop_c_win[i] > 0 {
                    for j in 0..n_eta {
                        if dist.is_zero_fixed(j) {
                            continue;
                        }
                        let rate_c = acc_c_win[i][j] as f64 / prop_c_win[i] as f64;
                        step_cw[i][j] = if rate_c > 0.44 {
                            (step_cw[i][j] * 1.1).min(5.0)
                        } else {
                            (step_cw[i][j] * 0.9).max(1e-6)
                        };
                        acc_c_win[i][j] = 0;
                    }
                    prop_c_win[i] = 0;
                }
            }
        }
        if n_kappa > 0 && k % adapt_interval == 0 {
            for i in 0..n_subjects {
                if prop_k_win[i] > 0 {
                    let rate_k = acc_k_win[i] as f64 / prop_k_win[i] as f64;
                    step_kappa[i] = if rate_k > 0.40 {
                        (step_kappa[i] * 1.1).min(5.0)
                    } else {
                        (step_kappa[i] * 0.9).max(0.01)
                    };
                    acc_k_win[i] = 0;
                    prop_k_win[i] = 0;
                }
            }
        }

        // ---- Step 2: SA Step-Size Schedule ----
        let gamma = if k <= k1 { 1.0 } else { 1.0 / (k - k1) as f64 };
        let gamma_omega = if k <= k1 {
            (1.0 / (k.saturating_sub(omega_burnin)).max(1) as f64).clamp(0.15, 0.50)
        } else {
            gamma
        };
        let gamma_mstep = if k <= k1 { mstep_damping } else { gamma };

        // ---- Step 3: Phi-frozen Mu-Referencing BEFORE updating zero-mean marginals/copulas ----
        // Critical order: shifting `theta` and re-centring `etas[i]` FIRST ensures `etas`
        // are centred around 0 before the zero-mean marginal & copula M-step scores them!
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        let mut temp_theta_lower = log_theta_lower.clone();
        let mut temp_theta_upper = log_theta_upper.clone();
        let mstep_maxiter = if k <= k1 { 3 } else { 5 };

        if use_mu_ref {
            let sds = dist.marginal_sds();
            for (p_idx, &(theta_idx, eta_idx, additive)) in mu_ref_pairs.iter().enumerate() {
                if init_params.theta_fixed.get(theta_idx).copied().unwrap_or(false) {
                    continue;
                }
                let raw_delta = if !dist.is_selected() {
                    // Gaussian warm-up: exact closed-form sample mean shift.
                    let mean_eta =
                        etas.iter().map(|e| e[eta_idx]).sum::<f64>() / (n_subjects.max(1) as f64);
                    gamma * mean_eta
                } else {
                    // Post-selection: exact phi-frozen Robbins-Monro Newton step minimizing
                    // g(delta) = (1/N) sum_i prior_nll(eta_i - delta * e_{eta_idx}).
                    let sd_j = sds.get(eta_idx).copied().unwrap_or(0.3).max(0.02);
                    let h = 1e-4 * sd_j;
                    let inv_n = 1.0 / (n_subjects.max(1) as f64);
                    let mut g0 = 0.0_f64;
                    let mut gp = 0.0_f64;
                    let mut gm = 0.0_f64;
                    let mut work = vec![0.0_f64; n_eta];
                    for e in &etas {
                        g0 += dist.prior_nll(e);
                        work.copy_from_slice(e);
                        work[eta_idx] = e[eta_idx] - h;
                        gp += dist.prior_nll(&work);
                        work[eta_idx] = e[eta_idx] + h;
                        gm += dist.prior_nll(&work);
                    }
                    g0 *= inv_n;
                    gp *= inv_n;
                    gm *= inv_n;
                    let deriv = (gp - gm) / (2.0 * h);
                    let curv_raw = (gp - 2.0 * g0 + gm) / (h * h);
                    let min_curv = 0.25 / (sd_j * sd_j);
                    let curv = if curv_raw.is_finite() {
                        curv_raw.max(min_curv)
                    } else {
                        1.0 / (sd_j * sd_j)
                    };
                    let mu_gain = if k <= k1 { 0.35 } else { gamma };
                    mu_avg_grad[p_idx] = (1.0 - mu_gain) * mu_avg_grad[p_idx] + mu_gain * deriv;
                    mu_avg_curv[p_idx] = (1.0 - mu_gain) * mu_avg_curv[p_idx] + mu_gain * curv;
                    let step_cap = 0.5 * sd_j;
                    let newton = (-mu_avg_grad[p_idx] / mu_avg_curv[p_idx].max(min_curv))
                        .clamp(-step_cap, step_cap);
                    let scale = if k <= k1 { 0.5 } else { 1.0 };
                    scale * newton
                };

                if raw_delta.is_finite() && raw_delta.abs() > 0.0 {
                    let before = log_theta[theta_idx];
                    let realised_eta_shift = if additive {
                        let th_before = unpack_theta(theta_idx, before);
                        let th_after = (th_before + raw_delta).clamp(
                            init_params.theta_lower[theta_idx],
                            init_params.theta_upper[theta_idx],
                        );
                        log_theta[theta_idx] = pack_theta(theta_idx, th_after)
                            .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                        unpack_theta(theta_idx, log_theta[theta_idx]) - th_before
                    } else {
                        log_theta[theta_idx] = (before + raw_delta)
                            .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                        log_theta[theta_idx] - before
                    };

                    if realised_eta_shift.abs() > 0.0 {
                        for e in &mut etas {
                            e[eta_idx] -= realised_eta_shift;
                        }
                        // Keep all buffered Phase-1a snapshots in the current theta frame!
                        for snap in &mut select_buffer {
                            for e in snap {
                                e[eta_idx] -= realised_eta_shift;
                            }
                        }
                    }
                }
                if run_mstep {
                    temp_theta_lower[theta_idx] = log_theta[theta_idx];
                    temp_theta_upper[theta_idx] = log_theta[theta_idx];
                    mstep_evals_saved += 2 * mstep_maxiter as u64;
                }
            }
        }

        // ---- Step 4: Three-Phase Vine / Marginals / IOV M-Step ----
        if k <= k_select {
            if k > omega_burnin {
                dist.warmup_gaussian_update(&etas, gamma_omega);
            }
            if k + VINE_SELECT_BUFFER_MAX > k_select {
                if select_buffer.len() == VINE_SELECT_BUFFER_MAX {
                    select_buffer.pop_front();
                }
                select_buffer.push_back(etas.clone());
            }
            if k == k_select {
                if select_buffer.is_empty() {
                    select_buffer.push_back(etas.clone());
                }
                let m_draws = select_buffer.len();
                let mut pooled = Vec::with_capacity(m_draws * n_subjects);
                for snap in &select_buffer {
                    pooled.extend(snap.iter().cloned());
                }
                dist.select_structure_and_families(&pooled, n_subjects, 1.0 / (m_draws as f64));
                select_buffer.clear();
                if verbose {
                    let summary = dist.to_fit_params(&init_params.omega.eta_names);
                    let m_desc: Vec<String> = summary
                        .marginals
                        .iter()
                        .map(|m| format!("{}:{}", m.eta, m.family))
                        .collect();
                    let p_desc: Vec<String> = summary
                        .trees
                        .iter()
                        .flat_map(|t| t.pairs.iter())
                        .map(|p| format!("({})->{}", p.label, p.copula.family))
                        .collect();
                    eprintln!(
                        "SAEM vine selected at iter {k}: marginals=[{}], pairs=[{}]",
                        m_desc.join(", "),
                        p_desc.join(", ")
                    );
                }
            }
        } else if k > omega_burnin {
            let gamma_vine = if k <= k1 { gamma_omega.min(0.25) } else { gamma };
            dist.mstep_update(&etas, gamma_vine);
        }

        // IOV Omega_iov SA update (when model declares kappa).
        if n_kappa > 0 && k > omega_burnin {
            if let Some(iov_template) = init_params.omega_iov.as_ref() {
                let mut kappa_outer = DMatrix::<f64>::zeros(n_kappa, n_kappa);
                let mut n_total_occ = 0usize;
                for subj_k in &kappas {
                    for kv in subj_k {
                        let v = DVector::from_column_slice(kv);
                        kappa_outer += &v * v.transpose();
                        n_total_occ += 1;
                    }
                }
                if n_total_occ > 0 {
                    kappa_outer /= n_total_occ as f64;
                    s2_iov = (1.0 - gamma_omega) * &s2_iov + gamma_omega * &kappa_outer;
                }
                let mut mat = s2_iov.clone();
                for r in 0..n_kappa {
                    for c in 0..n_kappa {
                        if !iov_template.free_mask[(r, c)] {
                            mat[(r, c)] = 0.0;
                        }
                        let fr = init_params.kappa_fixed.get(r).copied().unwrap_or(false);
                        let fc = init_params.kappa_fixed.get(c).copied().unwrap_or(false);
                        if fr || fc {
                            mat[(r, c)] = iov_template.matrix[(r, c)];
                        }
                    }
                    if mat[(r, r)] < 1e-8 {
                        mat[(r, r)] = 1e-8;
                    }
                }
                omega_iov_cur = Some(OmegaMatrix::from_matrix_with_mask(
                    mat,
                    iov_template.eta_names.clone(),
                    iov_template.diagonal,
                    iov_template.free_mask.clone(),
                ));
            }
        }

        // ---- Step 5: Numerical M-step for non-pinned theta + sigma ----
        if run_mstep {
            let gamma_sigma = sigma_mstep_sa_steps(gamma, gamma_mstep, k <= k1, None, n_sigma);
            let kappas_for_mstep = if n_kappa > 0 { Some(kappas.as_slice()) } else { None };
            let any_free_num_theta =
                (0..n_theta).any(|i| temp_theta_lower[i] < temp_theta_upper[i]);
            let any_free_sigma = (0..n_sigma).any(|i| log_sigma_lower[i] < log_sigma_upper[i]);
            if any_free_num_theta || any_free_sigma {
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &etas,
                    kappas_for_mstep,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                    &theta_packs_log_mask,
                    None,
                    &schedules,
                    &[],
                    mstep_optimizer,
                    Some(&newton_tally),
                );
                damp_mstep(&mut log_theta, &theta_new, gamma_mstep);
                damp_mstep_sigma_variance(&mut log_sigma, &sigma_new, &gamma_sigma);
            }
        }

        if mid_k2_iter == Some(k) {
            mid_phase2_state = Some((log_theta.clone(), log_sigma.clone(), dist.pack_free_psi()));
        }
    }

    // Assemble final ModelParameters carrying `omega_equiv` AND `Arc<VineCopulaOmega>`.
    let final_theta: Vec<f64> = (0..n_theta).map(|i| unpack_theta(i, log_theta[i])).collect();
    let final_sigma: Vec<f64> = log_sigma.iter().map(|&v| v.exp()).collect();
    let final_vine_psi = dist.pack_free_psi();
    let vine_summary = dist.to_fit_params(&init_params.omega.eta_names);
    let omega_equiv = dist.to_omega_matrix().clone();
    let vine_arc = Arc::new(dist);

    let final_params = ModelParameters {
        theta: final_theta,
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: omega_equiv,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector { values: final_sigma, names: init_params.sigma.names.clone() },
        sigma_fixed: init_params.sigma_fixed.clone(),
        residual_correlations: init_params.residual_correlations.clone(),
        residual_correlation_fixed: init_params.residual_correlation_fixed.clone(),
        omega_iov: omega_iov_cur,
        kappa_fixed: init_params.kappa_fixed.clone(),
        mixture: None,
        vine_dist: Some(Arc::clone(&vine_arc)),
    };

    // ---- Post-SAEM Multi-Start Vine MAP EBEs + Sensitivity Jacobians ----
    let (eta_hats, h_matrices, final_kappas, mm_warning) =
        compute_vine_map_ebes(model, population, &final_params, &vine_arc, &etas, &kappas, options);
    if let Some(w) = mm_warning {
        warnings.push(w);
    }

    // Gaussian-equivalent FOCE/Laplace OFV baseline (stored as `SaemObjective::foce_ofv`;
    // `fit.rs` replaces `ofv` and `vine_corrected_ofv` with the exact Vine QMC IS `-2 log L`).
    let resolved_interaction =
        crate::types::interaction_for(EstimationMethod::Saem, options.interaction);
    let foce_ofv = 2.0
        * pop_nll(
            model,
            population,
            &final_params,
            &eta_hats,
            &h_matrices,
            &final_kappas,
            resolved_interaction,
        );
    if verbose {
        eprintln!("{}", saem_final_ofv_report(foce_ofv));
    }

    // ---- Louis Empirical Score Covariance Step on Free Fixed Effects (theta, sigma) ----
    let (covariance_matrix, covariance_method, covariance_wall_time_secs, cov_warnings) =
        compute_vine_louis_covariance(
            model,
            population,
            &final_params,
            &eta_hats,
            &h_matrices,
            &final_kappas,
            options,
        );
    warnings.extend(cov_warnings);
    warnings.extend(newton_tally.warning("SAEM"));

    // ---- Optional Conditional-Distribution Pass (`conddist = true`) ----
    let cond_dist = if options.saem_conddist {
        if verbose {
            eprintln!(
                "Running SAEM vine conditional-distribution pass ({} samples/subject, {} burn-in)...",
                options.saem_conddist_nsamp, options.saem_conddist_burnin
            );
        }
        Some(crate::estimation::saem_conddist::run_conditional_distribution(
            model,
            population,
            &final_params,
            &eta_hats,
            &final_kappas,
            options,
        ))
    } else {
        None
    };

    let saem_mh_accept_tail =
        (tail_proposed > 0).then_some(tail_accepted as f64 / tail_proposed as f64);

    let mut converged = true;
    if k2 == 0 {
        converged = false;
        warnings.push(
            "SAEM vine: `n_convergence = 0` skips the stochastic-approximation accumulation \
             phase; reporting converged = false."
                .to_string(),
        );
    } else if tail_proposed > 0 && tail_accepted == 0 {
        converged = false;
        warnings.push(
            "SAEM vine: 0 Metropolis–Hastings proposals accepted during the convergence phase; \
             reporting converged = false."
                .to_string(),
        );
    }
    if final_params.theta.iter().any(|x| !x.is_finite())
        || final_params.sigma.values.iter().any(|x| !x.is_finite())
        || final_vine_psi.iter().any(|x| !x.is_finite())
    {
        converged = false;
        warnings.push(
            "SAEM vine: non-finite parameter estimate at termination; reporting converged = false."
                .to_string(),
        );
    }
    if k2 >= 20 {
        if let Some((mid_th, mid_sg, mid_vp)) = &mid_phase2_state {
            let mut max_drift = 0.0_f64;
            for (i, (&a, &b)) in log_theta.iter().zip(mid_th.iter()).enumerate() {
                if !init_params.theta_fixed.get(i).copied().unwrap_or(false) {
                    max_drift = max_drift.max((a - b).abs());
                }
            }
            for (i, (&a, &b)) in log_sigma.iter().zip(mid_sg.iter()).enumerate() {
                if !init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
                    max_drift = max_drift.max((a - b).abs());
                }
            }
            if final_vine_psi.len() == mid_vp.len() {
                for (&a, &b) in final_vine_psi.iter().zip(mid_vp.iter()) {
                    max_drift = max_drift.max((a - b).abs());
                }
            }
            // Under `1/k` Robbins–Monro averaging, a >1.0 unconstrained jump across the second half
            // of the convergence phase (>2.7x fold move) indicates an unsettled runaway trajectory.
            if !max_drift.is_finite() || max_drift > 1.0 {
                converged = false;
                warnings.push(format!(
                    "SAEM vine: convergence-phase parameter trajectory did not settle (max second-half \
                     unconstrained drift = {max_drift:.3} > 1.0); reporting converged = false."
                ));
            }
        }
    }
    if let Some(w) =
        crate::estimation::impmap::gate_converged_on_mcem_objective(&mut converged, foce_ofv)
    {
        warnings.push(w);
    }

    Ok(OuterResult {
        ofv: foce_ofv,
        converged,
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        covariance_method,
        covariance_wall_time_secs,
        warnings,
        saem_mu_ref_m_step_evals_saved: use_mu_ref.then_some(mstep_evals_saved),
        saem_objective_value: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail,
        mstep_optimizer_used: Some(mstep_optimizer.as_str().to_string()),
        cond_dist,
        vine_params: Some(vine_summary),
        vine_corrected_ofv: None,
        ..OuterResult::from_params(final_params)
    })
}
