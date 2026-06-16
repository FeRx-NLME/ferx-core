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

// TODO(ferx-core#380, Phase 2): remove once the sweep loop consumes these.
#![allow(dead_code)]

use nalgebra::{Cholesky, DMatrix};
use rand::Rng;
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
}
