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
}
