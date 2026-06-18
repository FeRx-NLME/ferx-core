//! Full MCMC Bayesian estimation — Path A: Gibbs-within-HMC
//! (`EstimationMethod::Bayes`, NONMEM `METHOD=BAYES` parity).
//!
//! Draws from the joint posterior `p(θ, Ω, Σ, {ηᵢ} | y)` by alternating:
//!   1. a per-subject **η block** — reuses the SAEM HMC (`hmc::hmc_step`) /
//!      block-MH (`saem::mh_steps`) kernels, sampling `ηᵢ | θ, Ω, Σ, y`;
//!   2. a **population block** — conjugate full-conditional draws of `θ, Ω, Σ`
//!      from the same sufficient statistics the SAEM M-step already forms.
//!
//! This file currently provides the conjugate-draw primitives for block (2).
//! The sweep loop and estimator entry point land in a follow-up (see
//! ferx-core#380, Phase 2).
//!
//! ## Conjugate draws
//!
//! `rand_distr` 0.4 ships `Gamma` and `ChiSquared` but **not** `InverseGamma`
//! or `Wishart`, so both are built here:
//!   - inverse-gamma via `1 / Gamma` ([`inverse_gamma_draw`]);
//!   - inverse-Wishart via the Bartlett decomposition of a Wishart draw, then
//!     matrix inversion ([`inverse_wishart_draw`], [`wishart_draw`]).

use crate::estimation::outer_optimizer::OuterResult;
use crate::estimation::saem::{mh_kappa_steps, mh_steps};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{
    individual_nll_into_with_schedule, individual_nll_iov, split_obs_by_occasion,
};
use crate::types::{
    BayesResult, CompiledModel, FitOptions, ModelParameters, OmegaMatrix, Population, SigmaVector,
    Subject,
};
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{ChiSquared, Distribution, Gamma, StandardNormal};

/// Draw from an inverse-gamma distribution `InvGamma(shape, scale)` with the
/// standard (Wikipedia) parameterization: density `∝ x^(−shape−1) exp(−scale/x)`,
/// mean `scale / (shape − 1)` for `shape > 1`.
///
/// Uses the identity `X ~ Gamma(shape, rate = scale) ⟹ 1/X ~ InvGamma(shape, scale)`.
/// `rand_distr::Gamma` is parameterized by (shape, *scale* = 1/rate), so we pass
/// `1/scale` as the Gamma scale.
///
/// Posterior use: for a Normal residual with `n` observations, scatter
/// `SS = Σ (y−f)²`, and conjugate prior `InvGamma(a₀, b₀)`, the full conditional
/// of the residual variance is `InvGamma(a₀ + n/2, b₀ + SS/2)`.
// Used by the Ω block for the independent per-variance full conditional of a
// diagonal Ω; also the basis for any future conjugate-σ draw (ferx-core#380).
pub fn inverse_gamma_draw(shape: f64, scale: f64, rng: &mut impl Rng) -> f64 {
    debug_assert!(
        shape > 0.0 && scale > 0.0,
        "InvGamma params must be positive"
    );
    // Gamma(shape, scale = 1/rate); we want rate = `scale`, so Gamma scale = 1/scale.
    let gamma = Gamma::new(shape, 1.0 / scale).expect("valid Gamma parameters");
    let g = gamma.sample(rng);
    1.0 / g
}

/// Draw from a Wishart distribution `Wishart(df, V)` with `df` degrees of freedom
/// and scale matrix `V = scale_chol · scale_cholᵀ` (pass the **lower** Cholesky
/// factor of `V`). Returns a `p×p` symmetric positive-definite matrix.
///
/// Bartlett decomposition: build a lower-triangular `A` with
/// `A[i,i] = sqrt(χ²_{df − i})` and `A[i,j] = N(0,1)` for `i > j`; then
/// `W = (L A)(L A)ᵀ ~ Wishart(df, L Lᵀ)`.
///
/// Requires `df > p − 1` so every diagonal chi-squared has positive df.
pub fn wishart_draw(df: f64, scale_chol: &DMatrix<f64>, rng: &mut impl Rng) -> DMatrix<f64> {
    let p = scale_chol.nrows();
    debug_assert_eq!(p, scale_chol.ncols(), "scale Cholesky must be square");
    debug_assert!(df > (p as f64) - 1.0, "Wishart df must exceed p − 1");

    let mut a = DMatrix::<f64>::zeros(p, p);
    for i in 0..p {
        let dfi = df - i as f64;
        let chi = ChiSquared::new(dfi).expect("valid chi-squared df");
        a[(i, i)] = chi.sample(rng).sqrt();
        for j in 0..i {
            a[(i, j)] = rng.sample(StandardNormal);
        }
    }
    let m = scale_chol * a; // lower-triangular × lower-triangular = lower-triangular
    &m * m.transpose()
}

/// Draw from an inverse-Wishart distribution `InvWishart(df, psi)` with `df`
/// degrees of freedom and scale matrix `psi`. Mean is `psi / (df − p − 1)` for
/// `df > p + 1`.
///
/// Sampled as `Σ = W⁻¹` where `W ~ Wishart(df, psi⁻¹)`. Returns `None` if `psi`
/// (or the realized `W`) is not invertible — the caller should fall back to the
/// previous Ω draw in that (rare, degenerate) case.
///
/// Posterior use: for `N` random-effect vectors with scatter `S = Σ ηᵢηᵢᵀ` and
/// conjugate prior `InvWishart(ν₀, Λ₀)`, the full conditional of Ω is
/// `InvWishart(ν₀ + N, Λ₀ + S)`.
pub fn inverse_wishart_draw(
    df: f64,
    psi: &DMatrix<f64>,
    rng: &mut impl Rng,
) -> Option<DMatrix<f64>> {
    let psi_inv = psi.clone().try_inverse()?;
    // Symmetrize to kill the asymmetry that try_inverse can introduce, so the
    // Cholesky sees an exactly-symmetric matrix.
    let psi_inv = 0.5 * (&psi_inv + psi_inv.transpose());
    let chol = Cholesky::new(psi_inv)?;
    let w = wishart_draw(df, &chol.l(), rng);
    let sigma = w.try_inverse()?;
    Some(0.5 * (&sigma + sigma.transpose()))
}

// ---------------------------------------------------------------------------
// Posterior summaries
// ---------------------------------------------------------------------------
// The generic MCMC diagnostics (split-R̂, ESS, quantile) live in
// `crate::stats::convergence` so any sampler can share them; only the
// Bayes-specific `PosteriorSummary` assembly stays here.

use crate::stats::convergence::{ess_bulk, ess_tail, mean, quantile_sorted, split_rhat};
use crate::types::PosteriorSummary;

/// Build a [`PosteriorSummary`] for one parameter from its per-chain draws.
pub fn summarize_param(name: &str, chains: &[Vec<f64>]) -> PosteriorSummary {
    let mut all: Vec<f64> = chains.iter().flatten().copied().collect();
    let mean_v = mean(&all);
    let sd_v = if all.len() > 1 {
        (all.iter().map(|x| (x - mean_v).powi(2)).sum::<f64>() / (all.len() as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let bulk = ess_bulk(chains);
    let tail = ess_tail(chains);
    PosteriorSummary {
        name: name.to_string(),
        mean: mean_v,
        sd: sd_v,
        q025: quantile_sorted(&all, 0.025),
        median: quantile_sorted(&all, 0.5),
        q975: quantile_sorted(&all, 0.975),
        rhat: split_rhat(chains),
        ess_bulk: bulk,
        ess_tail: tail,
        // MCSE of the posterior mean uses the bulk ESS.
        mcse: if bulk > 0.0 {
            sd_v / bulk.sqrt()
        } else {
            f64::NAN
        },
    }
}

// ---------------------------------------------------------------------------
// Gibbs-within-HMC sampler
// ---------------------------------------------------------------------------

/// Weakly-informative prior SD (on the unconstrained, log-where-positive scale)
/// for the population θ / σ random-walk block. Broad ⇒ near-flat.
const POP_PRIOR_SD: f64 = 10.0;
/// Floor for variances / scales to keep logs and Cholesky factors finite.
const TINY: f64 = 1e-12;
/// Positive-definite floor for the BSV Ω diagonal after an inverse-Wishart draw
/// (matches SAEM's `SAEM_OMEGA_DIAG_FLOOR`).
const OMEGA_DIAG_FLOOR: f64 = 1e-6;
/// Positive-definite floor for the IOV Ω_iov diagonal (matches SAEM's IOV floor;
/// looser than BSV because per-occasion variances are smaller).
const OMEGA_IOV_DIAG_FLOOR: f64 = 1e-8;
/// Max split-R̂ at or below which the fit is reported converged. Matches the
/// guidance in `docs/src/estimation/bayes.md` (Vehtari et al. 2021); used for
/// both the `converged` flag and the non-convergence warning.
const RHAT_CONVERGENCE_THRESHOLD: f64 = 1.01;

/// One coordinate of the population random-walk block.
#[derive(Clone, Copy)]
enum PopCoord {
    Theta { idx: usize, log: bool },
    Sigma { idx: usize },
}

/// Per-subject negative log-likelihood, routing to the IOV-aware kernel when the
/// subject carries per-occasion kappas (`kappas` non-empty) and the plain kernel
/// otherwise. Centralizes the IOV/non-IOV branch so every sweep site stays
/// consistent.
#[allow(clippy::too_many_arguments)]
fn subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    kappas: &[Vec<f64>],
    omega: &OmegaMatrix,
    omega_iov: Option<&OmegaMatrix>,
    sigma: &[f64],
    scratch: &mut EventPkParams,
    schedule: Option<&crate::pk::event_driven::EventSchedule>,
) -> f64 {
    if kappas.is_empty() {
        individual_nll_into_with_schedule(
            model, subject, theta, eta, omega, sigma, scratch, schedule,
        )
    } else {
        individual_nll_iov(model, subject, theta, eta, kappas, omega, omega_iov, sigma)
    }
}

/// Re-impose a covariance matrix's structural template onto a fresh draw `m`:
/// zero structurally-absent entries (`free_mask` false), restore FIX-ed
/// rows/columns from `original`, and floor the diagonal for positive-
/// definiteness. Shared by the Ω and Ω_iov inverse-Wishart blocks.
fn impose_omega_structure(
    m: &mut DMatrix<f64>,
    free_mask: &nalgebra::DMatrix<bool>,
    fixed: &[bool],
    original: &DMatrix<f64>,
    diag_floor: f64,
) {
    let n = m.nrows();
    for i in 0..n {
        for j in 0..n {
            if !free_mask[(i, j)] {
                m[(i, j)] = 0.0;
            }
            let fi = fixed.get(i).copied().unwrap_or(false);
            let fj = fixed.get(j).copied().unwrap_or(false);
            if fi || fj {
                m[(i, j)] = original[(i, j)];
            }
        }
    }
    for i in 0..n {
        if m[(i, i)] < diag_floor {
            m[(i, i)] = diag_floor;
        }
    }
}

/// Full MCMC Bayesian estimation entry point (Path A: Gibbs-within-HMC).
///
/// Per sweep, per chain:
///   1. η block — `mh_steps` (block random-walk preconditioned by `chol(Ω)`),
///      sampling `ηᵢ | θ, Ω, σ, κ, y` for each subject;
///   1b. κ block (IOV models) — `mh_kappa_steps` sampling each per-occasion
///      `κᵢₖ | η, θ, Ω, Ω_iov, y`;
///   2. Ω block — conjugate inverse-Wishart draw from `S = Σ ηᵢηᵢᵀ`, with the
///      structural `free_mask` / fixed entries re-imposed;
///   2c. Ω_iov block (IOV) — conjugate inverse-Wishart draw from `Σᵢ Σₖ κᵢₖκᵢₖᵀ`;
///   2b. mu-ref θ block — exact Gaussian full conditional (when every η is a
///      non-fixed log mu-ref); 3. remaining (θ, σ) — componentwise + adaptive
///      joint Metropolis, objective `Σᵢ individual_nll` (η, Ω, κ held fixed).
///
/// IOV (per-occasion κ) is supported for **zero-mean kappas** (`κ ~ N(0, Ω_iov)`,
/// the `exp(η + κ)` form); kappa mu-references are rejected.
///
/// Returns an [`OuterResult`] whose point estimate is the posterior mean and
/// whose [`OuterResult::bayes`] carries the posterior summaries + diagnostics.
pub fn run_bayes(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_kappa = model.n_kappa;
    // IOV (per-occasion kappa) is supported for zero-mean kappas only — i.e.
    // κ ~ N(0, Ω_iov) added to an existing mu-reference (`exp(η + κ)`), the
    // common form. Kappas that anchor to their own θ (kappa mu-refs) would
    // need a θ-block extension; reject those for now.
    if n_kappa > 0 {
        if init_params.omega_iov.is_none() {
            return Err(
                "Bayesian estimation: model declares kappa but init has no omega_iov".to_string(),
            );
        }
        if !model.kappa_mu_refs.is_empty() {
            return Err(
                "Bayesian estimation (method = bayes) supports zero-mean IOV kappas only \
                 (κ ~ N(0, Ω_iov)); kappa mu-references are not yet supported"
                    .to_string(),
            );
        }
    }

    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_theta = model.n_theta;
    let n_sigma = init_params.sigma.values.len();
    if n_eta == 0 {
        return Err("Bayesian estimation requires at least one random effect (eta)".to_string());
    }

    let n_warmup = options.bayes_warmup;
    let n_sample = options.bayes_iters;
    let thin = options.bayes_thin.max(1);
    let n_chains = options.bayes_chains.max(1);
    let n_eta_mh = options.saem_n_mh_steps.max(1);
    let master_seed = options.bayes_seed.unwrap_or(0x6E_61_6D_63_62_61_79_65); // "bayesnam"
    let verbose = options.verbose;

    // Mu-referenced (log) θ↔η map: mu_pairs[eta_idx] = Some(theta_idx) when that
    // η is the log-deviation of θ (`P_i = θ·exp(η_i)`). When EVERY η is a
    // non-fixed log mu-ref, the whole θ-mean vector is drawn from its exact
    // Gaussian full conditional (the hierarchical-normal Gibbs move) instead of
    // the random-walk block — without it the RW barely moves θ (the data pins θ
    // at fixed η) and the chains do not mix.
    let mut mu_pairs: Vec<Option<usize>> = vec![None; n_eta];
    for (ei, ename) in model.eta_names.iter().enumerate() {
        if let Some(mr) = model.mu_refs.get(ename) {
            if mr.log_transformed {
                if let Some(ti) = model.theta_names.iter().position(|t| t == &mr.theta_name) {
                    mu_pairs[ei] = Some(ti);
                }
            }
        }
    }
    let full_mu_ref = n_eta > 0
        && (0..n_eta).all(|j| match mu_pairs[j] {
            Some(ti) => !init_params.theta_fixed.get(ti).copied().unwrap_or(false),
            None => false,
        });
    let conjugate_theta: std::collections::HashSet<usize> = if full_mu_ref {
        mu_pairs.iter().filter_map(|&o| o).collect()
    } else {
        std::collections::HashSet::new()
    };
    // The conjugate Gaussian θ move requires EVERY η to be a non-fixed log
    // mu-reference. When only some are (partial mu-referencing), the conjugate
    // move is disabled and those θ fall back to the random-walk block, which
    // mixes poorly (the data pins θ at fixed η). Flag it so a high R̂ has an
    // actionable cause rather than looking like a generic non-convergence.
    let n_log_mu_ref = (0..n_eta)
        .filter(|&j| match mu_pairs[j] {
            Some(ti) => !init_params.theta_fixed.get(ti).copied().unwrap_or(false),
            None => false,
        })
        .count();
    let partial_mu_ref = n_log_mu_ref > 0 && !full_mu_ref;

    // ----- population RW coordinates (free θ not handled conjugately, then σ) -----
    let mut pop_coords: Vec<PopCoord> = Vec::new();
    for j in 0..n_theta {
        if init_params.theta_fixed.get(j).copied().unwrap_or(false) || conjugate_theta.contains(&j)
        {
            continue;
        }
        let log = init_params.theta_lower.get(j).copied().unwrap_or(0.0) >= 0.0;
        pop_coords.push(PopCoord::Theta { idx: j, log });
    }
    for k in 0..n_sigma {
        if !init_params.sigma_fixed.get(k).copied().unwrap_or(false) {
            pop_coords.push(PopCoord::Sigma { idx: k });
        }
    }

    // ----- recorded-parameter layout: θ (all), Ω free lower-tri, σ (all) -----
    let mut omega_coords: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_eta {
        for j in 0..=i {
            if init_params.omega.free_mask[(i, j)] {
                omega_coords.push((i, j));
            }
        }
    }
    let mut param_names: Vec<String> = Vec::new();
    param_names.extend(init_params.theta_names.iter().cloned());
    for &(i, j) in &omega_coords {
        param_names.push(format!("OMEGA({},{})", i + 1, j + 1));
    }
    param_names.extend(init_params.sigma.names.iter().cloned());
    // Ω_iov (per-occasion kappa covariance) free lower-triangle, appended last.
    let mut omega_iov_coords: Vec<(usize, usize)> = Vec::new();
    if let Some(oi) = init_params.omega_iov.as_ref() {
        for i in 0..n_kappa {
            for j in 0..=i {
                if oi.free_mask[(i, j)] {
                    omega_iov_coords.push((i, j));
                }
            }
        }
    }
    for &(i, j) in &omega_iov_coords {
        param_names.push(format!("OMEGA_IOV({},{})", i + 1, j + 1));
    }
    let n_params = param_names.len();

    // Prior scale Λ₀ and df ν₀ for the Ω inverse-Wishart full conditional.
    let omega_all_fixed =
        (0..n_eta).all(|i| init_params.omega_fixed.get(i).copied().unwrap_or(false));
    let lambda0 = init_params.omega.matrix.clone();
    let nu0 = n_eta as f64 + 2.0;
    // Same inverse-Wishart prior for Ω_iov (kappa covariance).
    let omega_iov_all_fixed =
        (0..n_kappa).all(|i| init_params.kappa_fixed.get(i).copied().unwrap_or(false));
    let lambda0_iov = init_params.omega_iov.as_ref().map(|o| o.matrix.clone());
    let nu0_iov = n_kappa as f64 + 2.0;

    // Per-chain recorded draws: draws_by_chain[c][param] = Vec over retained sweeps.
    let mut draws_by_chain: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_chains);
    // Posterior-mean η accumulation (across all chains' retained draws).
    let mut eta_sum: Vec<DVector<f64>> = (0..n_subjects).map(|_| DVector::zeros(n_eta)).collect();
    let mut eta_record_count: u64 = 0;

    // HMC eta-block routing (autodiff builds only; opt-in via n_leapfrog > 0,
    // analytical-PK subjects). Default n_leapfrog = 0 keeps the MH kernel.
    #[cfg(feature = "autodiff")]
    let n_leapfrog = options.saem_n_leapfrog;
    // HMC is BSV-only (the AD gradient + kernel are kappa-unaware), so IOV
    // models always use the MH eta kernel.
    #[cfg(feature = "autodiff")]
    let using_hmc =
        n_leapfrog > 0 && model.ode_spec.is_none() && model.tv_fn.is_some() && n_kappa == 0;

    // Post-warmup HMC divergences across all chains (only the autodiff HMC
    // η-kernel can produce these; the MH kernel never mutates it, hence the
    // allow on non-autodiff builds).
    #[allow(unused_mut)]
    let mut n_divergent_total = 0u64;

    if verbose {
        eprintln!(
            "Starting Bayesian estimation (Gibbs-within-HMC): {} chain(s), \
             {} warmup + {} sampling sweeps{}",
            n_chains,
            n_warmup,
            n_sample,
            if n_kappa > 0 { ", IOV" } else { "" }
        );
    }
    // Print ~10 progress lines per chain.
    let progress_every = ((n_warmup + n_sample) / 10).max(1);

    // Pre-build each subject's event schedule once (it depends only on the
    // subject's doses + the PK model, not on θ/η/σ) and reuse it across every
    // NLL evaluation. Without this `subject_nll` rebuilds the dose/infusion
    // schedule on every call — O(subjects · n_pop · sweeps · chains) times.
    // Gating mirrors the FOCE inner loop (`inner_optimizer.rs`): only analytical
    // event-driven subjects with TV covariates or resets, and only when no
    // (possibly η-dependent) lagtime would make a baked-in schedule stale.
    let schedules: Vec<Option<crate::pk::event_driven::EventSchedule>> = population
        .subjects
        .iter()
        .map(|subject| {
            if (subject.has_tv_covariates() || subject.has_resets())
                && model.ode_spec.is_none()
                && crate::pk::event_driven::supports_event_driven(model.pk_model)
                && !model.has_lagtime()
            {
                Some(crate::pk::event_driven::EventSchedule::for_subject(
                    subject,
                    model.pk_model,
                    &[],
                ))
            } else {
                None
            }
        })
        .collect();

    'chains: for chain in 0..n_chains {
        let mut rng = StdRng::seed_from_u64(master_seed.wrapping_add(chain as u64 * 0x9E3779B9));
        let mut scratch = EventPkParams::default();

        // Chain state.
        let mut theta = init_params.theta.clone();
        let mut sigma = init_params.sigma.values.clone();
        let mut omega_mat = init_params.omega.matrix.clone();
        let mut omega_cur = OmegaMatrix::from_matrix(
            omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        let mut etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];

        // IOV state: per-subject, per-occasion kappa vectors (empty when no
        // IOV, which routes `subject_nll` to the plain kernel). One occasion
        // list per subject from the OCC column.
        let mut kappas: Vec<Vec<Vec<f64>>> = (0..n_subjects)
            .map(|i| {
                let n_occ = if n_kappa > 0 {
                    split_obs_by_occasion(&population.subjects[i]).len()
                } else {
                    0
                };
                vec![vec![0.0; n_kappa]; n_occ]
            })
            .collect();
        let mut omega_iov_cur: Option<OmegaMatrix> = init_params.omega_iov.clone();
        // Per-subject kappa-MH step scale (adapted in warmup).
        let mut kappa_scale = 0.6_f64;
        let mut acc_kappa = 0u64;
        let mut prop_kappa = 0u64;

        // Unconstrained population vector + its prior centre.
        let pack = |theta: &[f64], sigma: &[f64]| -> Vec<f64> {
            pop_coords
                .iter()
                .map(|c| match *c {
                    PopCoord::Theta { idx, log } => {
                        if log {
                            theta[idx].max(TINY).ln()
                        } else {
                            theta[idx]
                        }
                    }
                    PopCoord::Sigma { idx } => sigma[idx].max(TINY).ln(),
                })
                .collect()
        };
        let u0 = pack(&theta, &sigma);

        // Per-coordinate random-walk step sizes for the (θ, σ) block. A single
        // shared scalar mixed badly for parameters on very different scales or
        // with very different identifiability (e.g. a weakly-identified 3-cpt
        // peripheral volume vs a well-determined clearance); each coordinate now
        // adapts its own scale and is updated componentwise.
        let n_pop = pop_coords.len();
        let mut rw_scales = vec![0.1_f64; n_pop];
        let mut acc_pop = vec![0u64; n_pop];
        let mut prop_pop = vec![0u64; n_pop];
        let mut eta_scale = 0.6_f64;
        let mut acc_eta = 0u64;
        let mut prop_eta = 0u64;
        // Cumulative η-accept counters for the progress display only (the
        // adaptation counters above are reset every window, so they read ~0 right
        // after a reset).
        let mut acc_eta_disp = 0u64;
        let mut prop_eta_disp = 0u64;

        // Adaptive-covariance (Haario 2001) proposal for the (θ,σ) block. A
        // Welford-accumulated covariance of the unconstrained pop vector seeds a
        // JOINT proposal that moves along parameter correlations the
        // per-coordinate scales cannot (e.g. the V3↔Q3 ridge in a 3-cpt model).
        // Componentwise runs during the bootstrap phase (first half of warmup);
        // once the covariance is well-conditioned the sampler switches to the
        // joint proposal, frozen at the end of warmup so the sampling phase is
        // non-adaptive (valid MCMC).
        let mut u_mean = vec![0.0_f64; n_pop];
        let mut u_m2 = DMatrix::<f64>::zeros(n_pop, n_pop);
        let mut n_cov = 0usize;
        let mut prop_chol: Option<DMatrix<f64>> = None;
        let mut joint_scale = 1.0_f64;
        let mut joint_acc = 0u64;
        let mut joint_prop = 0u64;
        let bootstrap_end = (n_warmup / 2).max(1);
        let am_base = 2.38 * 2.38 / (n_pop.max(1) as f64); // Haario optimal scaling

        let mut chain_draws: Vec<Vec<f64>> = vec![Vec::new(); n_params];

        let total_sweeps = n_warmup + n_sample;
        for sweep in 0..total_sweeps {
            // Cooperative cancel: a Gibbs sweep (the η/κ/pop/Ω moves below) is the
            // dominant per-chain cost, so poll the flag at the sweep boundary so an
            // interrupt set from the host (e.g. R) takes effect within one sweep
            // rather than running every chain to completion. `break 'chains` drops
            // straight to the post-loop check, which returns Err before the
            // (now-partial) draws reach the summary stage.
            if crate::cancel::is_cancelled(&options.cancel) {
                if verbose {
                    eprintln!("Bayes: cancelled at chain {} sweep {}", chain, sweep);
                }
                break 'chains;
            }

            // (re)compute the per-subject NLL at the current (θ, Ω, σ, η, κ).
            let mut nll: Vec<f64> = (0..n_subjects)
                .map(|i| {
                    subject_nll(
                        model,
                        &population.subjects[i],
                        &theta,
                        &etas[i],
                        &kappas[i],
                        &omega_cur,
                        omega_iov_cur.as_ref(),
                        &sigma,
                        &mut scratch,
                        schedules[i].as_ref(),
                    )
                })
                .collect();

            // ---- 1. η block ----
            // HMC (gradient-guided) when available + opt-in (n_leapfrog > 0 on
            // an autodiff build, analytical-PK subject); otherwise the
            // chol(Ω)-preconditioned block random walk. Same routing as SAEM.
            for i in 0..n_subjects {
                #[cfg(feature = "autodiff")]
                let did_hmc = if using_hmc {
                    if let Some((new_eta, new_nll, accepted, divergent)) =
                        crate::estimation::hmc::hmc_step(
                            &population.subjects[i],
                            &etas[i],
                            nll[i],
                            model,
                            &theta,
                            &omega_cur,
                            &sigma,
                            eta_scale,
                            n_leapfrog,
                            &mut rng,
                        )
                    {
                        etas[i] = new_eta;
                        nll[i] = new_nll;
                        acc_eta += accepted as u64;
                        prop_eta += 1;
                        acc_eta_disp += accepted as u64;
                        prop_eta_disp += 1;
                        // Count post-warmup divergences for the diagnostic.
                        if sweep >= n_warmup && divergent {
                            n_divergent_total += 1;
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                #[cfg(not(feature = "autodiff"))]
                let did_hmc = false;

                if !did_hmc {
                    // IOV: sample η | κ (kappas held fixed) via the IOV-aware NLL.
                    let kappas_opt = omega_iov_cur.as_ref().map(|oi| (kappas[i].as_slice(), oi));
                    let (na, nll_new) = mh_steps(
                        &mut etas[i],
                        nll[i],
                        &population.subjects[i],
                        model,
                        &theta,
                        &omega_cur,
                        &sigma,
                        eta_scale,
                        &mut rng,
                        n_eta_mh,
                        &mut scratch,
                        kappas_opt,
                    );
                    nll[i] = nll_new;
                    acc_eta += na as u64;
                    prop_eta += n_eta_mh as u64;
                    acc_eta_disp += na as u64;
                    prop_eta_disp += n_eta_mh as u64;
                }
            }

            // ---- 1b. κ block: sample κ_ik | η, θ, Ω, Ω_iov, data (η fixed) ----
            if n_kappa > 0 && !omega_iov_all_fixed {
                if let Some(ref oi) = omega_iov_cur {
                    for i in 0..n_subjects {
                        let (na, np, nll_new) = mh_kappa_steps(
                            &mut kappas[i],
                            nll[i],
                            &population.subjects[i],
                            model,
                            &theta,
                            &etas[i],
                            &omega_cur,
                            oi,
                            &sigma,
                            kappa_scale,
                            &mut rng,
                        );
                        nll[i] = nll_new;
                        acc_kappa += na as u64;
                        prop_kappa += np as u64;
                    }
                }
            }

            // ---- 2. Ω block (conjugate full conditional) ----
            if !omega_all_fixed {
                if init_params.omega.diagonal {
                    // Diagonal Ω: each variance has an INDEPENDENT inverse-gamma
                    // full conditional. Drawing a dense inverse-Wishart and then
                    // zeroing the off-diagonals is wrong — the marginal of an IW
                    // diagonal element carries df ν0−p+1, so the variance
                    // posteriors are mis-scaled (bias grows with η-dimension and
                    // small N). Using the IW diagonal-marginal IG((ν0−p+1)/2,
                    // Λ0_jj/2) as the per-variance prior keeps the implied prior
                    // identical to the dense path while giving each variance the
                    // full N "observations" (η_ij ~ N(0, ω_j²)).
                    let a0 = (nu0 - n_eta as f64 + 1.0) / 2.0;
                    for j in 0..n_eta {
                        if init_params.omega_fixed.get(j).copied().unwrap_or(false)
                            || !init_params.omega.free_mask[(j, j)]
                        {
                            continue; // FIX-ed or structurally-absent variance.
                        }
                        let ss: f64 = etas.iter().map(|e| e[j] * e[j]).sum();
                        let shape = a0 + n_subjects as f64 / 2.0;
                        let scale = (lambda0[(j, j)] / 2.0 + ss / 2.0).max(TINY);
                        omega_mat[(j, j)] =
                            inverse_gamma_draw(shape, scale, &mut rng).max(OMEGA_DIAG_FLOOR);
                    }
                    omega_cur = OmegaMatrix::from_matrix(
                        omega_mat.clone(),
                        init_params.omega.eta_names.clone(),
                        init_params.omega.diagonal,
                    );
                } else {
                    let mut s = DMatrix::<f64>::zeros(n_eta, n_eta);
                    for e in &etas {
                        let ev = DVector::from_column_slice(e);
                        s += &ev * ev.transpose();
                    }
                    let psi_post = &lambda0 + s;
                    if let Some(draw) =
                        inverse_wishart_draw(nu0 + n_subjects as f64, &psi_post, &mut rng)
                    {
                        let mut m = draw;
                        impose_omega_structure(
                            &mut m,
                            &init_params.omega.free_mask,
                            &init_params.omega_fixed,
                            &init_params.omega.matrix,
                            OMEGA_DIAG_FLOOR,
                        );
                        omega_mat = m;
                        omega_cur = OmegaMatrix::from_matrix(
                            omega_mat.clone(),
                            init_params.omega.eta_names.clone(),
                            init_params.omega.diagonal,
                        );
                    }
                }
            }

            // ---- 2c. Ω_iov block (conjugate inverse-Wishart over kappas) ----
            // Posterior IW(ν0 + N_occ, Λ0_iov + Σᵢ Σₖ κᵢₖκᵢₖᵀ); the structural
            // template (free_mask, fixed rows, OMEGA_IOV_DIAG_FLOOR) is re-imposed
            // via impose_omega_structure, the same helper the Ω block uses.
            if n_kappa > 0 && !omega_iov_all_fixed {
                if let (Some(oi_ref), Some(lam)) =
                    (init_params.omega_iov.as_ref(), lambda0_iov.as_ref())
                {
                    let mut s = DMatrix::<f64>::zeros(n_kappa, n_kappa);
                    let mut n_occ_total = 0usize;
                    for ks in &kappas {
                        for kap in ks {
                            let kv = DVector::from_column_slice(kap);
                            s += &kv * kv.transpose();
                            n_occ_total += 1;
                        }
                    }
                    let psi_post = lam + s;
                    if let Some(draw) =
                        inverse_wishart_draw(nu0_iov + n_occ_total as f64, &psi_post, &mut rng)
                    {
                        let mut m = draw;
                        impose_omega_structure(
                            &mut m,
                            &oi_ref.free_mask,
                            &init_params.kappa_fixed,
                            &oi_ref.matrix,
                            OMEGA_IOV_DIAG_FLOOR,
                        );
                        omega_iov_cur = Some(OmegaMatrix::from_matrix_with_mask(
                            m,
                            oi_ref.eta_names.clone(),
                            oi_ref.diagonal,
                            oi_ref.free_mask.clone(),
                        ));
                    }
                }
            }

            // ---- 2b. mu-ref θ block (exact Gaussian full conditional) ----
            // For P_i = θ·exp(η_i) with η ~ N(0, Ω), the population mean
            // μ = log θ has full conditional μ ~ N(μ_old + η̄, Ω/N). Draw the
            // shift s = η̄ + chol(Ω/N)·z, set θ ← θ·exp(s), and re-centre
            // η_i ← η_i − s so each individual parameter logφ_i = μ + η_i is
            // unchanged (the data likelihood is invariant; only the η-prior
            // moves). This is an always-accepted Gibbs move and is what makes
            // the chains mix.
            if full_mu_ref {
                let mut eta_bar = vec![0.0; n_eta];
                for e in &etas {
                    for j in 0..n_eta {
                        eta_bar[j] += e[j];
                    }
                }
                for v in eta_bar.iter_mut() {
                    *v /= n_subjects as f64;
                }
                let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
                let lz = &omega_cur.chol * DVector::from_column_slice(&z);
                let inv_sqrt_n = 1.0 / (n_subjects as f64).sqrt();
                let s: Vec<f64> = (0..n_eta)
                    .map(|j| eta_bar[j] + inv_sqrt_n * lz[j])
                    .collect();
                // The η re-centering must subtract the shift *actually applied* to
                // log θ, not the raw drawn shift `s`. When a θ bound clamps the
                // move, the applied log-shift is smaller than `s[j]`; subtracting
                // the full `s[j]` would break the logφ_i = log θ + η_i invariance,
                // silently changing the data likelihood with no MH correction.
                let mut s_applied = s.clone();
                for j in 0..n_eta {
                    if let Some(ti) = mu_pairs[j] {
                        let lo = init_params.theta_lower.get(ti).copied().unwrap_or(f64::MIN);
                        let hi = init_params.theta_upper.get(ti).copied().unwrap_or(f64::MAX);
                        // Clamp in log space (equivalent for positive θ) so the
                        // applied shift is exact when no bound is active.
                        let lo_ln = if lo > 0.0 { lo.ln() } else { f64::NEG_INFINITY };
                        let hi_ln = if hi > 0.0 && hi.is_finite() {
                            hi.ln()
                        } else {
                            f64::INFINITY
                        };
                        let old_ln = theta[ti].max(TINY).ln();
                        let new_ln = (old_ln + s[j]).clamp(lo_ln, hi_ln);
                        theta[ti] = new_ln.exp();
                        s_applied[j] = new_ln - old_ln;
                    }
                }
                for e in etas.iter_mut() {
                    for j in 0..n_eta {
                        e[j] -= s_applied[j];
                    }
                }
            }

            // Refresh the cached per-subject NLL before the (θ,σ) block uses it
            // as the Metropolis baseline. Blocks 2/2c/2b drew a new Ω / Ω_iov and
            // (for mu-ref) re-centered (θ, η), so the cached `nll` still carries
            // the pre-draw / pre-recenter η-prior, κ-prior, and log|Ω| terms.
            // Block 3 recomputes only the *proposal* NLL, so a stale baseline
            // leaves those terms uncancelled — a constant per-sweep offset δ on
            // every θ/σ accept that biases σ and non-mu-ref θ. Recompute so the
            // ratio is exact. (Unconditional when block 3 runs: cheaper to always
            // refresh than to track which of the three blocks fired.)
            if n_pop > 0 {
                for i in 0..n_subjects {
                    nll[i] = subject_nll(
                        model,
                        &population.subjects[i],
                        &theta,
                        &etas[i],
                        &kappas[i],
                        &omega_cur,
                        omega_iov_cur.as_ref(),
                        &sigma,
                        &mut scratch,
                        schedules[i].as_ref(),
                    );
                }
            }

            // ---- 3. (θ, σ) block ----
            // Moving one θ/σ changes every subject's likelihood, so each move
            // recomputes the full per-subject NLL; η and Ω are fixed, so their
            // prior terms cancel in the ratio. Two kernels: componentwise random
            // walk during the bootstrap phase, then a joint adaptive-covariance
            // proposal (once `prop_chol` is built) that moves along correlations.
            if n_pop > 0 {
                let inv_var = 1.0 / (POP_PRIOR_SD * POP_PRIOR_SD);

                // (a) Componentwise random walk — one coordinate at a time, each
                // with its own adaptive scale. ALWAYS runs, so it carries mixing
                // regardless of whether the joint proposal exists or is well
                // scaled (a mixture kernel: the joint move below can only help).
                for c in 0..n_pop {
                    let (idx, log, is_theta) = match pop_coords[c] {
                        PopCoord::Theta { idx, log } => (idx, log, true),
                        PopCoord::Sigma { idx } => (idx, true, false),
                    };
                    let u_old = if is_theta {
                        if log {
                            theta[idx].max(TINY).ln()
                        } else {
                            theta[idx]
                        }
                    } else {
                        sigma[idx].max(TINY).ln()
                    };
                    let u_new = u_old + rw_scales[c] * rng.sample::<f64, _>(StandardNormal);

                    // Mutate the single coordinate in place (no full θ/σ vector
                    // clone per move) and restore it on reject.
                    let old_val = if is_theta { theta[idx] } else { sigma[idx] };
                    if is_theta {
                        theta[idx] = if log { u_new.exp() } else { u_new };
                    } else {
                        sigma[idx] = u_new.exp();
                    }
                    let nll_prop: Vec<f64> = (0..n_subjects)
                        .map(|i| {
                            subject_nll(
                                model,
                                &population.subjects[i],
                                &theta,
                                &etas[i],
                                &kappas[i],
                                &omega_cur,
                                omega_iov_cur.as_ref(),
                                &sigma,
                                &mut scratch,
                                schedules[i].as_ref(),
                            )
                        })
                        .collect();
                    let sum_cur: f64 = nll.iter().sum();
                    let sum_prop: f64 = nll_prop.iter().sum();
                    let d_nlp = 0.5 * ((u_new - u0[c]).powi(2) - (u_old - u0[c]).powi(2)) * inv_var;
                    prop_pop[c] += 1;
                    if rng.gen::<f64>().ln() < (sum_cur - sum_prop) - d_nlp {
                        nll = nll_prop;
                        acc_pop[c] += 1;
                    } else if is_theta {
                        theta[idx] = old_val;
                    } else {
                        sigma[idx] = old_val;
                    }
                }

                // (b) Joint adaptive-covariance (Haario) proposal — an ADDITIONAL
                // move, available once `prop_chol` is built. Proposes all
                // coordinates together along the estimated posterior covariance,
                // catching correlations the per-coordinate walk cannot (e.g. a
                // V3↔Q3 ridge). u' = u + √joint_scale · L z.
                if let Some(ref l) = prop_chol {
                    let u_cur = pack(&theta, &sigma);
                    let z = DVector::from_iterator(
                        n_pop,
                        (0..n_pop).map(|_| rng.sample::<f64, _>(StandardNormal)),
                    );
                    let step = joint_scale.sqrt() * (l * z);
                    let u_prop: Vec<f64> = (0..n_pop).map(|c| u_cur[c] + step[c]).collect();
                    let mut theta_prop = theta.clone();
                    let mut sigma_prop = sigma.clone();
                    for (c, &up) in pop_coords.iter().zip(&u_prop) {
                        match *c {
                            PopCoord::Theta { idx, log } => {
                                theta_prop[idx] = if log { up.exp() } else { up };
                            }
                            PopCoord::Sigma { idx } => sigma_prop[idx] = up.exp(),
                        }
                    }
                    let nll_prop: Vec<f64> = (0..n_subjects)
                        .map(|i| {
                            subject_nll(
                                model,
                                &population.subjects[i],
                                &theta_prop,
                                &etas[i],
                                &kappas[i],
                                &omega_cur,
                                omega_iov_cur.as_ref(),
                                &sigma_prop,
                                &mut scratch,
                                schedules[i].as_ref(),
                            )
                        })
                        .collect();
                    let sum_cur: f64 = nll.iter().sum();
                    let sum_prop: f64 = nll_prop.iter().sum();
                    let mut d_nlp = 0.0;
                    for c in 0..n_pop {
                        d_nlp += 0.5
                            * ((u_prop[c] - u0[c]).powi(2) - (u_cur[c] - u0[c]).powi(2))
                            * inv_var;
                    }
                    joint_prop += 1;
                    if rng.gen::<f64>().ln() < (sum_cur - sum_prop) - d_nlp {
                        theta = theta_prop;
                        sigma = sigma_prop;
                        nll = nll_prop;
                        joint_acc += 1;
                    }
                }

                // Welford update of the pop-vector covariance (warmup only; the
                // proposal is frozen for the sampling phase). The estimate is
                // built from the componentwise-driven exploration, which keeps
                // moving even when the joint proposal is poorly scaled.
                if sweep < n_warmup {
                    let u_now = pack(&theta, &sigma);
                    n_cov += 1;
                    let nc = n_cov as f64;
                    // delta vs the OLD mean, then update the mean, then delta2 vs
                    // the NEW mean — Welford's covariance recurrence.
                    let delta: Vec<f64> = (0..n_pop).map(|c| u_now[c] - u_mean[c]).collect();
                    for c in 0..n_pop {
                        u_mean[c] += delta[c] / nc;
                    }
                    let delta2: Vec<f64> = (0..n_pop).map(|c| u_now[c] - u_mean[c]).collect();
                    for i in 0..n_pop {
                        for j in 0..n_pop {
                            u_m2[(i, j)] += delta[i] * delta2[j];
                        }
                    }
                }
            }

            // ---- warmup adaptation of the step sizes ----
            if sweep < n_warmup && (sweep + 1) % 50 == 0 {
                // Componentwise scales (bootstrap kernel).
                for c in 0..n_pop {
                    if prop_pop[c] > 0 {
                        let r = acc_pop[c] as f64 / prop_pop[c] as f64;
                        rw_scales[c] *= (r - 0.234).exp();
                        rw_scales[c] = rw_scales[c].clamp(1e-4, 100.0);
                        acc_pop[c] = 0;
                        prop_pop[c] = 0;
                    }
                }
                // Global scale of the joint Haario proposal (≈0.234 target).
                if joint_prop > 0 {
                    let r = joint_acc as f64 / joint_prop as f64;
                    joint_scale *= (r - 0.234).exp();
                    joint_scale = joint_scale.clamp(1e-4, 1e4);
                    joint_acc = 0;
                    joint_prop = 0;
                }
                // (Re)build the joint proposal Cholesky once past the bootstrap
                // phase and with enough samples for a well-conditioned estimate.
                if n_pop > 0 && sweep + 1 >= bootstrap_end && n_cov > 2 * n_pop {
                    let cov = &u_m2 / ((n_cov - 1) as f64);
                    let mut p = am_base * cov;
                    for i in 0..n_pop {
                        p[(i, i)] += 1e-9; // regularize against a singular estimate
                    }
                    if let Some(ch) = Cholesky::new(p) {
                        prop_chol = Some(ch.l());
                    }
                }
                if prop_eta > 0 {
                    let r = acc_eta as f64 / prop_eta as f64;
                    eta_scale *= (r - 0.234).exp();
                    eta_scale = eta_scale.clamp(1e-4, 100.0);
                }
                acc_eta = 0;
                prop_eta = 0;
                if prop_kappa > 0 {
                    let r = acc_kappa as f64 / prop_kappa as f64;
                    kappa_scale *= (r - 0.234).exp();
                    kappa_scale = kappa_scale.clamp(1e-4, 100.0);
                }
                acc_kappa = 0;
                prop_kappa = 0;
            }

            // ---- record retained draws ----
            if sweep >= n_warmup && (sweep - n_warmup) % thin == 0 {
                let mut p = 0;
                for &t in &theta {
                    chain_draws[p].push(t);
                    p += 1;
                }
                for &(i, j) in &omega_coords {
                    chain_draws[p].push(omega_mat[(i, j)]);
                    p += 1;
                }
                for &s in &sigma {
                    chain_draws[p].push(s);
                    p += 1;
                }
                if let Some(ref oi) = omega_iov_cur {
                    for &(i, j) in &omega_iov_coords {
                        chain_draws[p].push(oi.matrix[(i, j)]);
                        p += 1;
                    }
                }
                for i in 0..n_subjects {
                    eta_sum[i] += DVector::from_column_slice(&etas[i]);
                }
                eta_record_count += 1;
            }

            if verbose && (sweep + 1) % progress_every == 0 {
                let phase = if sweep < n_warmup { "warmup" } else { "sample" };
                // η-accept over the sweeps since the last progress line (a recent
                // window, not cumulative — so it tracks the adapted rate).
                let eta_acc = if prop_eta_disp > 0 {
                    100.0 * acc_eta_disp as f64 / prop_eta_disp as f64
                } else {
                    0.0
                };
                acc_eta_disp = 0;
                prop_eta_disp = 0;
                eprintln!(
                    "  Bayes chain {}/{}  sweep {:>5}/{} [{}]  η-accept≈{:.0}%",
                    chain + 1,
                    n_chains,
                    sweep + 1,
                    total_sweeps,
                    phase,
                    eta_acc
                );
            }
        }

        draws_by_chain.push(chain_draws);
    }

    // A cancel observed inside the sweep loop drops here via `break 'chains` with
    // partial/empty draws; bail before the summary stage indexes into them.
    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    if verbose {
        eprintln!("Bayes sampling complete; computing posterior summaries + EBEs...");
    }

    // ----- summaries -----
    let summaries: Vec<PosteriorSummary> = (0..n_params)
        .map(|p| {
            let chains: Vec<Vec<f64>> = draws_by_chain.iter().map(|c| c[p].clone()).collect();
            summarize_param(&param_names[p], &chains)
        })
        .collect();
    let max_rhat = summaries
        .iter()
        .map(|s| s.rhat)
        .filter(|r| r.is_finite())
        .fold(0.0_f64, f64::max);
    let n_draws_per_chain = draws_by_chain.first().map(|c| c[0].len()).unwrap_or(0);

    // ----- posterior-mean point estimate -----
    let mean_of = |name_pred: &dyn Fn(usize) -> bool| -> Vec<f64> {
        (0..n_params)
            .filter(|&p| name_pred(p))
            .map(|p| {
                let all: Vec<f64> = draws_by_chain
                    .iter()
                    .flat_map(|c| c[p].iter().copied())
                    .collect();
                all.iter().sum::<f64>() / all.len().max(1) as f64
            })
            .collect()
    };
    let sig_start = n_theta + omega_coords.len();
    let theta_mean = mean_of(&|p| p < n_theta);
    let omega_entries_mean = mean_of(&|p| p >= n_theta && p < sig_start);
    let sigma_mean = mean_of(&|p| p >= sig_start && p < sig_start + n_sigma);
    let omega_iov_entries_mean = mean_of(&|p| p >= sig_start + n_sigma);

    let mut omega_mean_mat = init_params.omega.matrix.clone();
    for (slot, &(i, j)) in omega_coords.iter().enumerate() {
        omega_mean_mat[(i, j)] = omega_entries_mean[slot];
        omega_mean_mat[(j, i)] = omega_entries_mean[slot];
    }
    let omega_mean = OmegaMatrix::from_matrix(
        omega_mean_mat,
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );

    // Posterior-mean Ω_iov (kappa covariance), reassembled from the recorded
    // free entries; preserves the structural free_mask.
    let omega_iov_mean = init_params.omega_iov.as_ref().map(|oi_ref| {
        let mut m = oi_ref.matrix.clone();
        for (slot, &(i, j)) in omega_iov_coords.iter().enumerate() {
            m[(i, j)] = omega_iov_entries_mean[slot];
            m[(j, i)] = omega_iov_entries_mean[slot];
        }
        OmegaMatrix::from_matrix_with_mask(
            m,
            oi_ref.eta_names.clone(),
            oi_ref.diagonal,
            oi_ref.free_mask.clone(),
        )
    });

    let mean_params = ModelParameters {
        theta: theta_mean.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        theta_fixed: init_params.theta_fixed.clone(),
        omega: omega_mean.clone(),
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: sigma_mean.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: omega_iov_mean.clone(),
        kappa_fixed: init_params.kappa_fixed.clone(),
    };

    // Final EBEs + sensitivity (H) matrices at the posterior mean, warm-started
    // from the posterior-mean η. Mirrors the SAEM post-loop pass; gives the
    // correctly-shaped (n_obs × n_eta) H matrices that CWRES/shrinkage need and
    // keeps the reported EBEs consistent with the point-estimate params.
    let warm_etas: Vec<DVector<f64>> = (0..n_subjects)
        .map(|i| {
            if eta_record_count > 0 {
                &eta_sum[i] / eta_record_count as f64
            } else {
                DVector::zeros(n_eta)
            }
        })
        .collect();
    let (eta_hats, h_matrices, _inner_stats, kappas) =
        crate::estimation::inner_optimizer::run_inner_loop_warm(
            model,
            population,
            &mean_params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&warm_etas),
            None,
            0,
        );

    // OFV at the posterior mean (2·Σ individual_nll, IOV-aware). NOTE: this is
    // the posterior-mean joint NLL ×2, NOT a FOCE/Laplace marginal OFV — it is
    // reported for a rough AIC-style comparison only.
    let kappas_mean: Vec<Vec<Vec<f64>>> = kappas
        .iter()
        .map(|ks| ks.iter().map(|k| k.iter().copied().collect()).collect())
        .collect();
    let mut scratch = EventPkParams::default();
    let ofv = 2.0
        * (0..n_subjects)
            .map(|i| {
                subject_nll(
                    model,
                    &population.subjects[i],
                    &theta_mean,
                    eta_hats[i].as_slice(),
                    &kappas_mean[i],
                    &omega_mean,
                    omega_iov_mean.as_ref(),
                    &sigma_mean,
                    &mut scratch,
                    schedules[i].as_ref(),
                )
            })
            .sum::<f64>();

    let bayes = BayesResult {
        summaries,
        n_chains,
        n_warmup,
        n_draws_per_chain,
        // Post-warmup HMC divergences (0 on the MH eta path, which has no
        // divergence concept).
        n_divergent: n_divergent_total as usize,
        max_rhat,
        draws: None,
    };

    let mut warnings = Vec::new();
    if max_rhat > RHAT_CONVERGENCE_THRESHOLD {
        warnings.push(format!(
            "Bayes: max split-R-hat = {max_rhat:.3} (> {RHAT_CONVERGENCE_THRESHOLD}) — chains may \
             not have converged; increase bayes_warmup / bayes_iters."
        ));
    }
    if n_chains < 2 {
        warnings.push(
            "Bayes: bayes_chains = 1 — split-R̂ is computed by halving a single chain, so it \
             cannot detect between-chain non-convergence; max_rhat near 1 here is weak evidence \
             of convergence. Use bayes_chains >= 2."
                .to_string(),
        );
    }
    if partial_mu_ref {
        warnings.push(
            "Bayes: only some random effects are log mu-referenced — the conjugate θ Gibbs \
             move is disabled and those θ are sampled by random walk, which mixes slowly. \
             For best mixing, log mu-reference every η (P_i = θ·exp(η_i))."
                .to_string(),
        );
    }

    Ok(OuterResult {
        params: mean_params,
        ofv,
        converged: max_rhat.is_finite() && max_rhat < RHAT_CONVERGENCE_THRESHOLD,
        n_iterations: n_warmup + n_sample,
        eta_hats,
        h_matrices,
        kappas,
        covariance_matrix: None,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        final_gradient: None,
        sir_fallback_proposal: None,
        bayes: Some(bayes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// InvGamma(a, b) has mean b/(a−1). Check the sample mean converges.
    #[test]
    fn test_inverse_gamma_mean() {
        let mut rng = StdRng::seed_from_u64(1);
        let (a, b) = (5.0_f64, 4.0_f64); // mean = 4/4 = 1.0
        let n = 50_000;
        let mean: f64 = (0..n)
            .map(|_| inverse_gamma_draw(a, b, &mut rng))
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.03,
            "InvGamma mean = {mean}, expected ~1.0"
        );
    }

    /// InvGamma(a, b) has variance b²/((a−1)²(a−2)) for a > 2.
    #[test]
    fn test_inverse_gamma_variance() {
        let mut rng = StdRng::seed_from_u64(2);
        let (a, b) = (5.0_f64, 4.0_f64);
        let expected_var = b * b / ((a - 1.0).powi(2) * (a - 2.0)); // 16/(16·3) = 1/3
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| inverse_gamma_draw(a, b, &mut rng)).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(
            (var - expected_var).abs() < 0.02,
            "InvGamma var = {var}, expected ~{expected_var}"
        );
    }

    /// Same seed ⟹ identical draw (reproducibility).
    #[test]
    fn test_inverse_gamma_seed_determinism() {
        let mut r1 = StdRng::seed_from_u64(42);
        let mut r2 = StdRng::seed_from_u64(42);
        for _ in 0..10 {
            assert_eq!(
                inverse_gamma_draw(3.0, 2.0, &mut r1),
                inverse_gamma_draw(3.0, 2.0, &mut r2)
            );
        }
    }

    /// InvWishart(df, Ψ) has mean Ψ/(df − p − 1). Check element-wise convergence
    /// and that every individual draw is symmetric positive-definite.
    #[test]
    fn test_inverse_wishart_mean_and_pd() {
        let mut rng = StdRng::seed_from_u64(7);
        let p = 2;
        let psi = DMatrix::<f64>::identity(p, p) * 2.0;
        let df = 10.0_f64;
        let denom = df - p as f64 - 1.0; // 7
        let expected_diag = 2.0 / denom; // ~0.2857

        let n = 30_000;
        let mut acc = DMatrix::<f64>::zeros(p, p);
        for _ in 0..n {
            let s = inverse_wishart_draw(df, &psi, &mut rng).expect("IW draw");
            // symmetry
            assert!((s[(0, 1)] - s[(1, 0)]).abs() < 1e-10);
            // positive-definite: Cholesky succeeds
            assert!(Cholesky::new(s.clone()).is_some(), "IW draw not PD: {s}");
            acc += s;
        }
        acc /= n as f64;
        assert!(
            (acc[(0, 0)] - expected_diag).abs() < 0.01,
            "IW mean[0,0] = {}, expected ~{expected_diag}",
            acc[(0, 0)]
        );
        assert!(
            acc[(0, 1)].abs() < 0.01,
            "IW mean off-diagonal should be ~0"
        );
    }

    /// 1-D consistency: InvWishart(df, [s]) ≡ InvGamma(df/2, s/2). Both must
    /// reproduce the same mean s/(df−2).
    #[test]
    fn test_inverse_wishart_1d_matches_inverse_gamma() {
        let mut rng = StdRng::seed_from_u64(11);
        let df = 8.0_f64;
        let s = 3.0_f64;
        let psi = DMatrix::from_row_slice(1, 1, &[s]);
        let expected = s / (df - 2.0); // 3/6 = 0.5

        let n = 60_000;
        let iw_mean: f64 = (0..n)
            .map(|_| inverse_wishart_draw(df, &psi, &mut rng).unwrap()[(0, 0)])
            .sum::<f64>()
            / n as f64;
        let ig_mean: f64 = (0..n)
            .map(|_| inverse_gamma_draw(df / 2.0, s / 2.0, &mut rng))
            .sum::<f64>()
            / n as f64;

        assert!((iw_mean - expected).abs() < 0.02, "IW 1-D mean = {iw_mean}");
        assert!((ig_mean - expected).abs() < 0.02, "IG mean = {ig_mean}");
        assert!((iw_mean - ig_mean).abs() < 0.03, "IW(1-D) and IG disagree");
    }

    /// Wishart(df, I) has mean df·I. Sanity check the Bartlett builder directly.
    #[test]
    fn test_wishart_mean() {
        let mut rng = StdRng::seed_from_u64(3);
        let p = 3;
        let l = DMatrix::<f64>::identity(p, p); // V = I
        let df = 12.0_f64;
        let n = 20_000;
        let mut acc = DMatrix::<f64>::zeros(p, p);
        for _ in 0..n {
            acc += wishart_draw(df, &l, &mut rng);
        }
        acc /= n as f64;
        for i in 0..p {
            assert!(
                (acc[(i, i)] - df).abs() < 0.2,
                "Wishart mean diag[{i}] = {}, expected ~{df}",
                acc[(i, i)]
            );
        }
    }

    fn iid_normal_chains(m: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_summarize_param_normal() {
        let chains = iid_normal_chains(4, 2000, 999);
        let s = summarize_param("X", &chains);
        assert_eq!(s.name, "X");
        assert!(s.mean.abs() < 0.1, "mean ~0, got {}", s.mean);
        assert!((s.sd - 1.0).abs() < 0.1, "sd ~1, got {}", s.sd);
        assert!(s.q025 < s.median && s.median < s.q975);
        assert!((s.q025 + 1.96).abs() < 0.2, "q025 ~ -1.96, got {}", s.q025);
        assert!(s.rhat < 1.02);
        assert!(s.mcse > 0.0 && s.mcse < 0.1);
    }

    /// Partial mu-referencing (not every η has a non-fixed log mu-ref θ)
    /// disables the conjugate θ move; the run must warn so a high R̂ has an
    /// actionable cause. Fixing one of warfarin's three mu-ref θ (TVKA) makes
    /// the all-η-mu-ref condition false while leaving the other two as live
    /// log mu-refs — the partial case.
    #[test]
    fn run_bayes_warns_on_partial_mu_ref() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let mut params = model.default_params.clone();
        params.theta_fixed[2] = true; // fix TVKA → its mu-ref pair is excluded

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 20;
        opts.bayes_iters = 40;
        opts.bayes_chains = 1;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 3;

        let res = run_bayes(&model, &pop, &params, &opts).expect("bayes runs");
        assert!(
            res.warnings.iter().any(|w| w.contains("mu-referenced")),
            "expected a partial-mu-ref warning, got: {:?}",
            res.warnings
        );
    }

    /// Exercises the multi-coordinate componentwise (θ,σ) block. The
    /// bioavailability model has several free non-mu-ref θ — TVV / TVKA have no
    /// η, THETA_F's η is logit (not log) — so the conjugate move is disabled and
    /// all four θ plus σ are sampled componentwise. Asserts the path runs and
    /// recovers the simulation truth (TVCL≈5, TVV≈50, TVKA≈1.5, THETA_F≈0.7);
    /// R̂ is deliberately NOT asserted tightly — these params are correlated and
    /// mix slowly under a random walk (the adaptive-covariance proposal is the
    /// follow-up), but the posterior MEANS are unbiased.
    #[test]
    fn run_bayes_multi_coord_nonmuref_block() {
        use std::path::Path;
        let model = crate::parser::model_parser::parse_model_file(Path::new(
            "examples/bioavailability.ferx",
        ))
        .expect("bioavailability model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/bioavailability_oral.csv"), None, None)
            .expect("bioavailability data loads");
        let params = model.default_params.clone();

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 1500;
        opts.bayes_iters = 1500;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 10;

        let res = run_bayes(&model, &pop, &params, &opts).expect("bayes runs");
        // Partial mu-referencing (only ETA_CL is a log mu-ref) ⇒ all θ go to the
        // componentwise block; the run must warn.
        assert!(
            res.warnings.iter().any(|w| w.contains("mu-referenced")),
            "expected partial-mu-ref warning, got: {:?}",
            res.warnings
        );
        let b = res.bayes.as_ref().expect("BayesResult present");
        let get = |n: &str| b.summaries.iter().find(|s| s.name == n).expect(n);
        for s in &b.summaries {
            assert!(
                s.mean.is_finite() && s.sd.is_finite() && s.sd >= 0.0,
                "{}",
                s.name
            );
            assert!(
                s.q025 <= s.median && s.median <= s.q975,
                "{} quantiles",
                s.name
            );
        }
        // Sanity-floor bounds only. This model is strongly correlated
        // (CL/V/F/KA trade off) and mixes slowly, so the mean estimate at these
        // settings is noisy — the test guards that the multi-coordinate block
        // runs and returns physically plausible values, not tight convergence
        // (truth is TVCL≈5, TVV≈50, TVKA≈1.5, THETA_F≈0.7).
        assert!(
            (2.0..9.0).contains(&get("TVCL").mean),
            "TVCL {}",
            get("TVCL").mean
        );
        assert!(
            (25.0..75.0).contains(&get("TVV").mean),
            "TVV {}",
            get("TVV").mean
        );
        assert!(
            (0.8..2.5).contains(&get("TVKA").mean),
            "TVKA {}",
            get("TVKA").mean
        );
        assert!(
            (0.4..0.95).contains(&get("THETA_F").mean),
            "THETA_F {}",
            get("THETA_F").mean
        );
    }

    /// End-to-end smoke test: short Bayes run on the bundled warfarin model.
    /// Asserts the sampler produces finite, well-ordered posterior summaries
    /// and a populated BayesResult. Short chains ⇒ no convergence assertion
    /// beyond finiteness.
    #[test]
    fn run_bayes_warfarin_smoke() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let params = model.default_params.clone();

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 40;
        opts.bayes_iters = 80;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 4; // keep the eta block cheap for a smoke test

        let res = run_bayes(&model, &pop, &params, &opts).expect("bayes runs");
        let bayes = res.bayes.as_ref().expect("BayesResult present");

        assert_eq!(bayes.n_chains, 2);
        assert_eq!(bayes.n_warmup, 40);
        assert!(bayes.n_draws_per_chain >= 1);
        assert!(!bayes.summaries.is_empty(), "expected posterior summaries");
        for s in &bayes.summaries {
            assert!(s.mean.is_finite(), "{}: mean not finite", s.name);
            assert!(s.sd.is_finite() && s.sd >= 0.0, "{}: bad sd", s.name);
            assert!(
                s.q025 <= s.median && s.median <= s.q975,
                "{}: quantiles out of order",
                s.name
            );
            assert!(s.rhat.is_finite(), "{}: R-hat not finite", s.name);
        }
        assert!(res.ofv.is_finite(), "OFV not finite");
        assert!(bayes.max_rhat.is_finite());
        assert_eq!(res.eta_hats.len(), pop.subjects.len());
    }

    /// A cancel flag set before the run aborts at the first sweep boundary and
    /// returns Err("cancelled by user") instead of running all chains to
    /// completion (regression for #393: a Bayes run could not be stopped).
    #[test]
    fn run_bayes_cancel_returns_err() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let params = model.default_params.clone();

        let cancel = crate::cancel::CancelFlag::new();
        cancel.cancel(); // already requested before the loop starts

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 1000;
        opts.bayes_iters = 1000;
        opts.bayes_chains = 4;
        opts.bayes_seed = Some(1);
        opts.cancel = Some(cancel);

        match run_bayes(&model, &pop, &params, &opts) {
            Err(e) => assert_eq!(e, "cancelled by user"),
            Ok(_) => panic!("expected cancel to abort the run"),
        }
    }

    /// Regression for the mu-ref θ bound clamp: with a tight `theta_upper` on a
    /// log mu-ref θ, the conjugate Gibbs shift repeatedly hits the bound. The fix
    /// subtracts the *actually applied* log-shift from η (not the raw drawn
    /// shift), preserving logφ_i = log θ + η_i, and the clamp keeps θ inside the
    /// bound. Asserts the whole posterior for TVCL stays at/below the active upper
    /// bound and the summaries remain finite (the old code subtracted the full
    /// shift, breaking the invariance and corrupting the likelihood). Also
    /// exercises the otherwise-uncovered clamp branch.
    #[test]
    fn run_bayes_respects_active_theta_bound() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let mut params = model.default_params.clone();
        // Truth is TVCL ≈ 0.133; pin the upper bound just below it so the mu-ref
        // shift is clamped on most sweeps.
        let bound = 0.12_f64;
        params.theta[0] = 0.11; // start inside the bound
        params.theta_upper[0] = bound;

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 100;
        opts.bayes_iters = 200;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 4;

        let res = run_bayes(&model, &pop, &params, &opts).expect("bayes runs");
        let b = res.bayes.as_ref().expect("BayesResult present");
        let tvcl = b.summaries.iter().find(|s| s.name == "TVCL").expect("TVCL");
        // The entire recorded posterior must respect the active upper bound.
        assert!(
            tvcl.q975 <= bound + 1e-9,
            "TVCL q975 {} exceeded active upper bound {bound}",
            tvcl.q975
        );
        assert!(tvcl.mean.is_finite() && tvcl.mean > 0.0 && tvcl.mean <= bound + 1e-9);
        for s in &b.summaries {
            assert!(s.mean.is_finite() && s.sd.is_finite(), "{}", s.name);
        }
    }

    /// IOV (per-occasion kappa) end-to-end: the warfarin IOV model has a
    /// zero-mean KAPPA_CL. Asserts the sampler runs the kappa block + Ω_iov
    /// draw, surfaces an OMEGA_IOV posterior on $bayes, and recovers a sane
    /// fit (TVCL≈0.13, finite kappa variance, mixed chains).
    #[test]
    fn run_bayes_iov_warfarin() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin_iov.ferx"))
                .expect("warfarin_iov model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, None)
            .expect("warfarin_iov data loads");

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 600;
        opts.bayes_iters = 600;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 6;

        let res = run_bayes(&model, &pop, &model.default_params, &opts).expect("IOV bayes runs");
        let b = res.bayes.as_ref().expect("BayesResult present");
        let get = |n: &str| b.summaries.iter().find(|s| s.name == n).expect(n);

        // The IOV variance is surfaced as a posterior parameter.
        let kiov = get("OMEGA_IOV(1,1)");
        assert!(
            kiov.mean.is_finite() && kiov.mean > 0.0,
            "IOV var {}",
            kiov.mean
        );
        assert!(kiov.q025 <= kiov.median && kiov.median <= kiov.q975);
        // Population fit is sane.
        assert!(
            (0.10..0.18).contains(&get("TVCL").mean),
            "TVCL {}",
            get("TVCL").mean
        );
        assert!(b.max_rhat.is_finite());
        // Per-occasion kappa EBEs are returned (one occasion list per subject).
        assert_eq!(res.kappas.len(), pop.subjects.len());
    }

    /// HMC eta-block end-to-end (autodiff only). With `saem_n_leapfrog > 0` on an
    /// analytical-PK model with no IOV, `run_bayes` routes the η block through the
    /// gradient-guided `hmc_step` instead of the random-walk kernel (the
    /// `#[cfg(feature = "autodiff")]` branch at the top of the sweep). The default
    /// (non-autodiff) coverage build compiles that branch out, so without this
    /// test the Bayes→HMC routing has zero coverage in any CI job. Asserts the
    /// HMC path yields finite, well-ordered summaries and a sane warfarin fit.
    #[test]
    #[cfg(feature = "autodiff")]
    fn run_bayes_warfarin_hmc_eta_block() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let params = model.default_params.clone();

        // Routing precondition for the HMC η block: analytical PK, no IOV.
        assert!(model.ode_spec.is_none() && model.tv_fn.is_some() && model.n_kappa == 0);

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 200;
        opts.bayes_iters = 200;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_leapfrog = 3; // > 0 ⇒ HMC η block (vs random-walk default)

        let res = run_bayes(&model, &pop, &params, &opts).expect("HMC bayes runs");
        let bayes = res.bayes.as_ref().expect("BayesResult present");

        assert!(!bayes.summaries.is_empty(), "expected posterior summaries");
        for s in &bayes.summaries {
            assert!(s.mean.is_finite(), "{}: mean not finite", s.name);
            assert!(s.sd.is_finite() && s.sd >= 0.0, "{}: bad sd", s.name);
            assert!(
                s.q025 <= s.median && s.median <= s.q975,
                "{}: quantiles out of order",
                s.name
            );
            assert!(s.rhat.is_finite(), "{}: R-hat not finite", s.name);
        }
        // Sane population recovery from the HMC-sampled etas.
        let tvcl = bayes
            .summaries
            .iter()
            .find(|s| s.name == "TVCL")
            .expect("TVCL summary");
        assert!((0.10..0.20).contains(&tvcl.mean), "TVCL {}", tvcl.mean);
        assert!(res.ofv.is_finite(), "OFV not finite");
        assert!(bayes.max_rhat.is_finite());
        assert_eq!(res.eta_hats.len(), pop.subjects.len());
    }

    #[test]
    #[ignore = "exploratory: prints FOCEI vs Bayes posterior means"]
    fn bayes_vs_focei_print() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("parse");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");

        let mut fopts = FitOptions::default();
        fopts.method = crate::types::EstimationMethod::FoceI;
        fopts.run_covariance_step = false;
        let f = crate::api::fit(&model, &pop, &model.default_params, &fopts).expect("focei");
        eprintln!("FOCEI theta = {:?}", f.theta);
        eprintln!("FOCEI omega diag = {:?}", f.omega.diagonal());
        eprintln!("FOCEI sigma = {:?}", f.sigma);

        let mut bopts = FitOptions::default();
        bopts.method = crate::types::EstimationMethod::Bayes;
        bopts.run_covariance_step = false;
        bopts.bayes_warmup = 1000;
        bopts.bayes_iters = 2000;
        bopts.bayes_chains = 4;
        bopts.bayes_seed = Some(1);
        bopts.saem_n_mh_steps = 10;
        let b = crate::api::fit(&model, &pop, &model.default_params, &bopts).expect("bayes");
        let br = b.bayes.as_ref().unwrap();
        for s in &br.summaries {
            eprintln!(
                "BAYES {:>12}: mean={:.4} sd={:.4} [{:.4}, {:.4}] Rhat={:.3} ESS={:.0}",
                s.name, s.mean, s.sd, s.q025, s.q975, s.rhat, s.ess_bulk
            );
        }
        eprintln!("BAYES max_rhat = {:.4}", br.max_rhat);
    }

    /// Accuracy + mixing regression on the bundled warfarin model. The
    /// posterior means must land near the FOCEI point estimate
    /// (TVCL≈0.133, TVV≈7.74, TVKA≈0.82; PROP_ERR var≈0.0106) and the chains
    /// must mix (max split-R̂ < 1.05). Ω posterior means run a little above the
    /// FOCEI MLE (inverse-Wishart posterior-mean bias at N=10 subjects), so
    /// their bounds are deliberately loose.
    #[test]
    fn run_bayes_warfarin_accuracy() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");

        let mut opts = FitOptions::default();
        opts.bayes_warmup = 400;
        opts.bayes_iters = 800;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(1);
        opts.saem_n_mh_steps = 10;

        let res = run_bayes(&model, &pop, &model.default_params, &opts).expect("bayes runs");
        let b = res.bayes.as_ref().unwrap();
        let get = |name: &str| -> &PosteriorSummary {
            b.summaries.iter().find(|s| s.name == name).expect(name)
        };

        let tvcl = get("TVCL");
        let tvv = get("TVV");
        let tvka = get("TVKA");
        let prop = get("PROP_ERR");

        assert!(
            b.max_rhat < 1.05,
            "chains did not mix: max R-hat = {}",
            b.max_rhat
        );
        assert!(
            (0.11..0.16).contains(&tvcl.mean),
            "TVCL posterior mean {} off (FOCEI ~0.133)",
            tvcl.mean
        );
        assert!(
            (6.5..9.0).contains(&tvv.mean),
            "TVV mean {} off (~7.74)",
            tvv.mean
        );
        assert!(
            (0.6..1.1).contains(&tvka.mean),
            "TVKA mean {} off (~0.82)",
            tvka.mean
        );
        assert!(
            (0.006..0.016).contains(&prop.mean),
            "PROP_ERR mean {} off (~0.0106)",
            prop.mean
        );
        // Thetas should be well-mixed (conjugate block ⇒ high ESS).
        for s in [tvcl, tvv, tvka] {
            assert!(s.ess_bulk > 200.0, "{} ESS too low: {}", s.name, s.ess_bulk);
        }
    }

    /// Full dispatch path: `fit` with `method = bayes` must route to run_bayes
    /// and surface the posterior on `FitResult.bayes`.
    #[test]
    fn fit_dispatch_bayes_populates_fitresult() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");

        let mut opts = FitOptions::default();
        opts.method = crate::types::EstimationMethod::Bayes;
        opts.run_covariance_step = false;
        opts.bayes_warmup = 20;
        opts.bayes_iters = 40;
        opts.bayes_chains = 2;
        opts.bayes_seed = Some(2);
        opts.saem_n_mh_steps = 3;

        let fitres = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
        assert_eq!(fitres.method, crate::types::EstimationMethod::Bayes);
        let b = fitres
            .bayes
            .as_ref()
            .expect("FitResult.bayes set by dispatch");
        assert!(!b.summaries.is_empty());
        assert!(b.max_rhat.is_finite());
    }
}
