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
use crate::estimation::saem::mh_steps;
use crate::pk::EventPkParams;
use crate::stats::likelihood::individual_nll_into;
use crate::types::{
    BayesResult, CompiledModel, FitOptions, ModelParameters, OmegaMatrix, Population, SigmaVector,
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
// Not yet used by run_bayes (σ is currently drawn by the RW-MH population block);
// retained for the conjugate-σ optimization (ferx-core#380, Phase 2b).
#[allow(dead_code)]
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
// Posterior summaries & convergence diagnostics
// ---------------------------------------------------------------------------

use crate::types::PosteriorSummary;

/// Type-7 (linear-interpolation) quantile of an already-sorted slice.
/// `q ∈ [0, 1]`. Empty input returns NaN.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Split-R̂ (Gelman et al. / Vehtari et al. 2021) across `chains` of equal-ish
/// length. Each chain is split in half, giving `2·M` sub-chains of length `n`;
/// R̂ = √(v̂ar⁺ / W). Values near 1.0 indicate mixing; `> 1.01` flags
/// non-convergence. Returns NaN if there are fewer than 2 usable sub-chains or
/// `n < 2`.
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    // Split each chain in half (drop the middle element when odd).
    let mut subs: Vec<&[f64]> = Vec::with_capacity(chains.len() * 2);
    for c in chains {
        let half = c.len() / 2;
        if half < 2 {
            continue;
        }
        subs.push(&c[..half]);
        subs.push(&c[c.len() - half..]);
    }
    let m = subs.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = subs[0].len();
    let means: Vec<f64> = subs.iter().map(|s| mean(s)).collect();
    let vars: Vec<f64> = subs.iter().map(|s| sample_var(s)).collect();
    let grand = mean(&means);

    // Between-chain variance B (per draw) and within-chain variance W.
    let b = n as f64 / (m as f64 - 1.0) * means.iter().map(|mj| (mj - grand).powi(2)).sum::<f64>();
    let w = vars.iter().sum::<f64>() / m as f64;
    if w <= 0.0 {
        return f64::NAN;
    }
    let var_plus = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
    (var_plus / w).sqrt()
}

/// Effective sample size via the combined multi-chain autocorrelation with
/// Geyer's initial-positive / initial-monotone truncation (Vehtari et al.
/// 2021, eq. 10–11). `chains` are equal-length. Returns the total draw count
/// when the chains are essentially uncorrelated, less when autocorrelated.
pub fn effective_sample_size(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m == 0 {
        return 0.0;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return (m * n) as f64;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let vars: Vec<f64> = chains.iter().map(|c| sample_var(c)).collect();
    let grand = mean(&means);
    let w = vars.iter().sum::<f64>() / m as f64;

    // With a single chain there is no between-chain term; the marginal variance
    // estimate is just W. With ≥2 chains use the standard B/W combination.
    let var_plus = if m == 1 {
        w
    } else {
        let b =
            n as f64 / (m as f64 - 1.0) * means.iter().map(|mj| (mj - grand).powi(2)).sum::<f64>();
        (n as f64 - 1.0) / n as f64 * w + b / n as f64
    };
    if !(var_plus > 0.0) {
        return (m * n) as f64;
    }

    // Combined autocorrelation at each lag: ρ_t = 1 − (W − mean_m acov_m(t)) / var⁺.
    let max_lag = n - 1;
    let mut rho = vec![0.0_f64; max_lag + 1];
    for (t, rho_t) in rho.iter_mut().enumerate() {
        let mean_acov: f64 = chains
            .iter()
            .zip(&means)
            .map(|(c, &mu)| autocov(c, t, mu))
            .sum::<f64>()
            / m as f64;
        *rho_t = 1.0 - (w - mean_acov) / var_plus;
    }

    // Geyer initial-positive sequence: sum paired autocorrelations Ρ_k =
    // ρ_{2k} + ρ_{2k+1} while positive, enforcing a monotone non-increasing cap.
    let mut tau = 1.0; // ρ_0 = 1 contributes once via the 1 + 2Σ form below.
    let mut prev_pair = f64::INFINITY;
    let mut k = 1;
    while 2 * k + 1 <= max_lag {
        let mut pair = rho[2 * k] + rho[2 * k + 1];
        if pair < 0.0 {
            break;
        }
        // Initial-monotone: never let a pair exceed the previous one.
        if pair > prev_pair {
            pair = prev_pair;
        }
        prev_pair = pair;
        tau += 2.0 * pair;
        k += 1;
    }
    // ρ_1 is added once (the k=0 pair's ρ_1 half); include it explicitly.
    tau += 2.0 * rho[1].max(0.0);

    let ess = (m * n) as f64 / tau.max(1.0);
    ess.min((m * n) as f64)
}

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
    let ess = effective_sample_size(chains);
    PosteriorSummary {
        name: name.to_string(),
        mean: mean_v,
        sd: sd_v,
        q025: quantile_sorted(&all, 0.025),
        median: quantile_sorted(&all, 0.5),
        q975: quantile_sorted(&all, 0.975),
        rhat: split_rhat(chains),
        // Bulk/tail ESS distinction (rank-normalized, folded) is a follow-up;
        // the single autocorrelation-based estimate is reported for both for now.
        ess_bulk: ess,
        ess_tail: ess,
        mcse: if ess > 0.0 {
            sd_v / ess.sqrt()
        } else {
            f64::NAN
        },
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample variance with denominator `n − 1`.
fn sample_var(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Biased (divide-by-N) lag-`t` autocovariance of `xs` about supplied mean `mu`.
fn autocov(xs: &[f64], t: usize, mu: f64) -> f64 {
    let n = xs.len();
    if t >= n {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..(n - t) {
        s += (xs[i] - mu) * (xs[i + t] - mu);
    }
    s / n as f64
}

// ---------------------------------------------------------------------------
// Gibbs-within-HMC sampler
// ---------------------------------------------------------------------------

/// Weakly-informative prior SD (on the unconstrained, log-where-positive scale)
/// for the population θ / σ random-walk block. Broad ⇒ near-flat.
const POP_PRIOR_SD: f64 = 10.0;
/// Floor for variances / scales to keep logs and Cholesky factors finite.
const TINY: f64 = 1e-12;

/// One coordinate of the population random-walk block.
#[derive(Clone, Copy)]
enum PopCoord {
    Theta { idx: usize, log: bool },
    Sigma { idx: usize },
}

/// Full MCMC Bayesian estimation entry point (Path A: Gibbs-within-HMC).
///
/// First cut: **BSV-only** (no IOV / `omega_iov`). Per sweep, per chain:
///   1. η block — `mh_steps` (block random-walk preconditioned by `chol(Ω)`),
///      sampling `ηᵢ | θ, Ω, σ, y` for each subject;
///   2. Ω block — conjugate inverse-Wishart draw from `S = Σ ηᵢηᵢᵀ`, with the
///      structural `free_mask` / fixed entries re-imposed;
///   3. (θ, σ) block — random-walk Metropolis in unconstrained space
///      (log where the lower bound is ≥ 0), objective `Σᵢ individual_nll` with η
///      and Ω held fixed (the η-prior term is then constant and cancels).
///
/// Returns an [`OuterResult`] whose point estimate is the posterior mean and
/// whose [`OuterResult::bayes`] carries the posterior summaries + diagnostics.
pub fn run_bayes(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    if init_params.omega_iov.is_some() || !model.kappa_names.is_empty() {
        return Err(
            "Bayesian estimation (method = bayes) does not yet support IOV (kappa) models \
             — BSV-only in this first cut (see ferx-core#380)"
                .to_string(),
        );
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
    let n_params = param_names.len();

    // Prior scale Λ₀ and df ν₀ for the Ω inverse-Wishart full conditional.
    let omega_all_fixed =
        (0..n_eta).all(|i| init_params.omega_fixed.get(i).copied().unwrap_or(false));
    let lambda0 = init_params.omega.matrix.clone();
    let nu0 = n_eta as f64 + 2.0;

    // Per-chain recorded draws: draws_by_chain[c][param] = Vec over retained sweeps.
    let mut draws_by_chain: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_chains);
    // Posterior-mean η accumulation (across all chains' retained draws).
    let mut eta_sum: Vec<DVector<f64>> = (0..n_subjects).map(|_| DVector::zeros(n_eta)).collect();
    let mut eta_record_count: u64 = 0;

    for chain in 0..n_chains {
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

        let mut rw_scale = 0.1_f64;
        let mut eta_scale = 0.6_f64;
        let mut acc_pop = 0u64;
        let mut prop_pop = 0u64;
        let mut acc_eta = 0u64;
        let mut prop_eta = 0u64;

        let mut chain_draws: Vec<Vec<f64>> = vec![Vec::new(); n_params];

        let total_sweeps = n_warmup + n_sample;
        for sweep in 0..total_sweeps {
            // (re)compute the per-subject NLL at the current (θ, Ω, σ, η).
            let mut nll: Vec<f64> = (0..n_subjects)
                .map(|i| {
                    individual_nll_into(
                        model,
                        &population.subjects[i],
                        &theta,
                        &etas[i],
                        &omega_cur,
                        &sigma,
                        &mut scratch,
                    )
                })
                .collect();

            // ---- 1. η block ----
            for i in 0..n_subjects {
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
                    None,
                );
                nll[i] = nll_new;
                acc_eta += na as u64;
                prop_eta += n_eta_mh as u64;
            }

            // ---- 2. Ω block (conjugate inverse-Wishart) ----
            if !omega_all_fixed {
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
                    // Re-impose structural zeros and fixed rows/cols, then floor.
                    for i in 0..n_eta {
                        for j in 0..n_eta {
                            if !init_params.omega.free_mask[(i, j)] {
                                m[(i, j)] = 0.0;
                            }
                            let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                            let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                            if fi || fj {
                                m[(i, j)] = init_params.omega.matrix[(i, j)];
                            }
                        }
                    }
                    for i in 0..n_eta {
                        if m[(i, i)] < TINY {
                            m[(i, i)] = TINY;
                        }
                    }
                    omega_mat = m;
                    omega_cur = OmegaMatrix::from_matrix(
                        omega_mat.clone(),
                        init_params.omega.eta_names.clone(),
                        init_params.omega.diagonal,
                    );
                    // η-block NLLs are now stale w.r.t. Ω; they are recomputed at
                    // the top of the next sweep, and the (θ,σ) block below
                    // recomputes its own proposal NLLs, so no refresh needed here.
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
                for j in 0..n_eta {
                    if let Some(ti) = mu_pairs[j] {
                        let lo = init_params.theta_lower.get(ti).copied().unwrap_or(f64::MIN);
                        let hi = init_params.theta_upper.get(ti).copied().unwrap_or(f64::MAX);
                        theta[ti] = (theta[ti].max(TINY).ln() + s[j]).exp().clamp(lo, hi);
                    }
                }
                for e in etas.iter_mut() {
                    for j in 0..n_eta {
                        e[j] -= s[j];
                    }
                }
            }

            // ---- 3. (θ, σ) block (random-walk Metropolis) ----
            if !pop_coords.is_empty() {
                let u_cur = pack(&theta, &sigma);
                let u_prop: Vec<f64> = u_cur
                    .iter()
                    .map(|&u| u + rw_scale * rng.sample::<f64, _>(StandardNormal))
                    .collect();
                let mut theta_prop = theta.clone();
                let mut sigma_prop = sigma.clone();
                for (c, &up) in pop_coords.iter().zip(&u_prop) {
                    match *c {
                        PopCoord::Theta { idx, log } => {
                            theta_prop[idx] = if log { up.exp() } else { up };
                        }
                        PopCoord::Sigma { idx } => {
                            sigma_prop[idx] = up.exp();
                        }
                    }
                }
                // Proposal population NLL (η, Ω fixed ⇒ η-prior term cancels).
                let nll_prop: Vec<f64> = (0..n_subjects)
                    .map(|i| {
                        individual_nll_into(
                            model,
                            &population.subjects[i],
                            &theta_prop,
                            &etas[i],
                            &omega_cur,
                            &sigma_prop,
                            &mut scratch,
                        )
                    })
                    .collect();
                let sum_cur: f64 = nll.iter().sum();
                let sum_prop: f64 = nll_prop.iter().sum();
                let nlp_cur = neg_log_prior(&u_cur, &u0);
                let nlp_prop = neg_log_prior(&u_prop, &u0);
                let log_alpha = (sum_cur + nlp_cur) - (sum_prop + nlp_prop);
                prop_pop += 1;
                if rng.gen::<f64>().ln() < log_alpha {
                    theta = theta_prop;
                    sigma = sigma_prop;
                    nll = nll_prop;
                    acc_pop += 1;
                }
            }

            // ---- warmup adaptation of the two step sizes ----
            if sweep < n_warmup && (sweep + 1) % 50 == 0 {
                if prop_pop > 0 {
                    let r = acc_pop as f64 / prop_pop as f64;
                    rw_scale *= ((r - 0.234) * 1.0).exp();
                    rw_scale = rw_scale.clamp(1e-4, 100.0);
                }
                if prop_eta > 0 {
                    let r = acc_eta as f64 / prop_eta as f64;
                    eta_scale *= ((r - 0.234) * 1.0).exp();
                    eta_scale = eta_scale.clamp(1e-4, 100.0);
                }
                acc_pop = 0;
                prop_pop = 0;
                acc_eta = 0;
                prop_eta = 0;
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
                for i in 0..n_subjects {
                    eta_sum[i] += DVector::from_column_slice(&etas[i]);
                }
                eta_record_count += 1;
            }
        }

        draws_by_chain.push(chain_draws);
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
    let theta_mean = mean_of(&|p| p < n_theta);
    let omega_entries_mean = mean_of(&|p| p >= n_theta && p < n_theta + omega_coords.len());
    let sigma_mean = mean_of(&|p| p >= n_theta + omega_coords.len());

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
        omega_iov: None,
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

    // OFV at the posterior mean (2·Σ individual_nll). NOTE: this is the
    // posterior-mean joint NLL ×2, NOT a FOCE/Laplace marginal OFV — it is
    // reported for a rough AIC-style comparison only.
    let mut scratch = EventPkParams::default();
    let ofv = 2.0
        * (0..n_subjects)
            .map(|i| {
                individual_nll_into(
                    model,
                    &population.subjects[i],
                    &theta_mean,
                    eta_hats[i].as_slice(),
                    &omega_mean,
                    &sigma_mean,
                    &mut scratch,
                )
            })
            .sum::<f64>();

    let bayes = BayesResult {
        summaries,
        n_chains,
        n_warmup,
        n_draws_per_chain,
        n_divergent: 0, // MH eta block has no divergence concept; HMC count TBD
        max_rhat,
        draws: None,
    };

    let warnings = if max_rhat > 1.1 {
        vec![format!(
            "Bayes: max split-R-hat = {max_rhat:.3} (> 1.1) — chains may not have converged; \
             increase bayes_warmup / bayes_iters."
        )]
    } else {
        Vec::new()
    };

    Ok(OuterResult {
        params: mean_params,
        ofv,
        converged: max_rhat.is_finite() && max_rhat < 1.1,
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

/// Negative log of the (unnormalized) Gaussian population prior on the
/// unconstrained vector `u`, centred at `u0` with SD [`POP_PRIOR_SD`].
fn neg_log_prior(u: &[f64], u0: &[f64]) -> f64 {
    u.iter()
        .zip(u0)
        .map(|(&x, &m)| 0.5 * ((x - m) / POP_PRIOR_SD).powi(2))
        .sum()
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
    fn test_quantile_sorted() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&s, 0.0), 1.0);
        assert_eq!(quantile_sorted(&s, 1.0), 5.0);
        assert_eq!(quantile_sorted(&s, 0.5), 3.0);
        assert!((quantile_sorted(&s, 0.25) - 2.0).abs() < 1e-12);
        assert!(quantile_sorted(&[], 0.5).is_nan());
    }

    #[test]
    fn test_split_rhat_mixed_is_near_one() {
        let chains = iid_normal_chains(4, 1000, 100);
        let rhat = split_rhat(&chains);
        assert!(rhat < 1.02, "R-hat for iid chains should be ~1, got {rhat}");
    }

    #[test]
    fn test_split_rhat_unmixed_is_large() {
        // Two chains with very different means → poor mixing → large R-hat.
        let mut chains = iid_normal_chains(2, 1000, 7);
        for x in chains[0].iter_mut() {
            *x += 10.0; // shift one chain far away
        }
        let rhat = split_rhat(&chains);
        assert!(
            rhat > 1.5,
            "R-hat for separated chains should be large, got {rhat}"
        );
    }

    #[test]
    fn test_ess_iid_near_total() {
        let m = 4;
        let n = 1000;
        let chains = iid_normal_chains(m, n, 314);
        let ess = effective_sample_size(&chains);
        let total = (m * n) as f64;
        assert!(
            ess > 0.6 * total && ess <= total,
            "iid ESS should be near total {total}, got {ess}"
        );
    }

    #[test]
    fn test_ess_autocorrelated_is_reduced() {
        // AR(1) with phi = 0.8 → strong positive autocorrelation → ESS ≪ N.
        let mut rng = StdRng::seed_from_u64(55);
        let phi = 0.8_f64;
        let n = 4000;
        let mut x = 0.0_f64;
        let chain: Vec<f64> = (0..n)
            .map(|_| {
                let eps: f64 = rng.sample(StandardNormal);
                x = phi * x + eps;
                x
            })
            .collect();
        let ess = effective_sample_size(&[chain]);
        assert!(
            ess < 0.4 * n as f64,
            "AR(1) phi=0.8 ESS should be well below {n}, got {ess}"
        );
        assert!(ess > 1.0, "ESS should stay positive, got {ess}");
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

    /// IOV models are not supported in the first cut — must error clearly
    /// rather than silently mis-sample.
    #[test]
    fn run_bayes_rejects_iov() {
        use std::path::Path;
        let model =
            crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
                .expect("warfarin model parses");
        let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
            .expect("warfarin data loads");
        let mut params = model.default_params.clone();
        // Fake an IOV omega to trip the guard.
        params.omega_iov = Some(params.omega.clone());
        let opts = FitOptions::default();
        let err = run_bayes(&model, &pop, &params, &opts)
            .err()
            .expect("IOV model should be rejected");
        assert!(err.contains("IOV"), "expected IOV rejection, got: {err}");
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
