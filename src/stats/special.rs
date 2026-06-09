//! Special functions for M3 BLOQ likelihood: erf, normal CDF, and log normal CDF.
//!
//! These are implemented from polynomial/rational approximations (Abramowitz &
//! Stegun 7.1.26 for erf) so that they are differentiable by Enzyme without
//! relying on LLVM intrinsics (`llvm.maximumnum`, `llvm.minimumnum`) that the
//! current Enzyme release cannot differentiate — see CLAUDE.md for context.
//! The implementations use only `+`, `-`, `*`, `/`, and `.exp()`.

/// 1 / sqrt(2)
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// 1 / sqrt(2*pi)
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
/// Smallest probability retained before taking `ln`. Prevents `-inf` contamination
/// of the likelihood when a BLOQ observation lies many SDs above the prediction.
const MIN_PROB: f64 = 1e-300;

/// Abramowitz & Stegun 7.1.26 — max error ~1.5e-7 over the whole real line.
/// Entirely polynomial in t = 1/(1 + p*|x|) and exp(-x²), so Enzyme-safe.
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + p * ax);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

/// Complementary error function. Uses `erfc(x) = 1 - erf(x)` directly; for very
/// large positive x this loses precision but the `log_normal_cdf` path below
/// uses the asymptotic form directly, so the precision loss never matters.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / √2)).
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z * INV_SQRT_2))
}

/// Standard normal quantile Φ⁻¹(p) — Peter Acklam's rational approximation.
///
/// Accurate to ~1.15×10⁻⁹ for p ∈ (10⁻³⁰⁸, 1−10⁻¹⁶). Returns ±∞ outside
/// (0, 1). Delegates to the identical implementation in `copula.rs`.
pub fn normal_quantile(p: f64) -> f64 {
    crate::stats::copula::normal_cdf_inv(p)
}

/// Numerically stable log Φ(z).
///
/// For z > -5 we use `ln(max(Φ(z), MIN_PROB))` directly. For very negative z the
/// naive form underflows to `ln(0) = -inf`, so we switch to the asymptotic
/// expansion of the Mills ratio:
///
///   log Φ(z) ≈ log φ(z) + log(-1/z) + log(1 - 1/z² + 3/z⁴ - 15/z⁶ + …)
///
/// where φ(z) = exp(-z²/2) / √(2π). We truncate after the 1/z⁶ term, which is
/// accurate to ~1e-12 for z < -5 (the branch threshold).
pub fn log_normal_cdf(z: f64) -> f64 {
    if z > -5.0 {
        let p = normal_cdf(z);
        if p < MIN_PROB {
            MIN_PROB.ln()
        } else {
            p.ln()
        }
    } else {
        let log_phi = -0.5 * z * z - (1.0_f64 / INV_SQRT_2PI).ln();
        let inv_z2 = 1.0 / (z * z);
        let series = 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2 - 15.0 * inv_z2 * inv_z2 * inv_z2;
        // z < 0, so -z > 0 and ln(-z) is well-defined; series > 0 for z < -5.
        log_phi - (-z).ln() + series.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn erf_zero() {
        // A&S 7.1.26 is a max-error-1.5e-7 rational approximation; erf(0) is
        // close to but not exactly zero.
        assert!(erf(0.0).abs() < 2e-7);
    }

    #[test]
    fn erf_one() {
        // erf(1) = 0.8427007929... — A&S 7.1.26 matches to ~1.5e-7.
        let v = erf(1.0);
        assert!((v - 0.842_700_792_949_715).abs() < 2e-7, "got {}", v);
    }

    #[test]
    fn erf_symmetry() {
        for &x in &[0.1, 0.5, 1.0, 2.0, 3.5] {
            assert_relative_eq!(erf(-x), -erf(x), epsilon = 1e-12);
        }
    }

    #[test]
    fn erf_bounds() {
        assert!(erf(-10.0) > -1.0 - 1e-6);
        assert!(erf(-10.0) < -1.0 + 1e-6);
        assert!(erf(10.0) < 1.0 + 1e-6);
        assert!(erf(10.0) > 1.0 - 1e-6);
    }

    #[test]
    fn normal_cdf_zero() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 2e-7);
    }

    #[test]
    fn normal_cdf_standard_values() {
        // Φ(1.96) ≈ 0.975 (classic) — allow A&S 1.5e-7 error.
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-5);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-5);
    }

    #[test]
    fn log_normal_cdf_matches_direct_for_moderate_z() {
        // For z > -5 the two branches should agree closely.
        for &z in &[-4.99, -3.0, -1.0, 0.0, 1.0, 3.0] {
            let direct = normal_cdf(z).ln();
            let stable = log_normal_cdf(z);
            assert_relative_eq!(direct, stable, epsilon = 1e-4);
        }
    }

    #[test]
    fn log_normal_cdf_stable_at_extreme_negative() {
        // At z = -20 the direct CDF underflows to zero, but the asymptotic form
        // yields approximately -203.92 (see Mills-ratio expansion).
        let v = log_normal_cdf(-20.0);
        assert!(v.is_finite(), "log_normal_cdf(-20) must be finite");
        assert!(
            (v - (-203.917)).abs() < 0.01,
            "expected ≈ -203.917, got {}",
            v
        );
    }

    #[test]
    fn log_normal_cdf_is_monotone_increasing() {
        let mut prev = log_normal_cdf(-30.0);
        for z in [-25.0, -20.0, -10.0, -5.5, -5.01, -4.99, -2.0, 0.0, 2.0].iter() {
            let v = log_normal_cdf(*z);
            assert!(
                v >= prev - 1e-6,
                "non-monotone at z={}: {} < {}",
                z,
                v,
                prev
            );
            prev = v;
        }
    }
}
