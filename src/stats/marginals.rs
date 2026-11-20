//! Zero-mean parametric marginal distributions for SAEM-copula random effects.
//!
//! Every random-effect marginal in an NLME model must have location anchored so
//! that `E[eta_j] == 0` identically; otherwise `log(TVP_j) + eta_j` (or
//! `TVP_j + eta_j`) carries an unidentifiable translation ridge between fixed
//! effects `theta` and the marginal location parameter.
//!
//! Supported zero-mean families:
//! - [`MarginalDist::Normal`]: `eta ~ N(0, sd^2)` (`k = 1`).
//! - [`MarginalDist::StudentT`]: symmetric zero-mean Student-t with scale `s > 0`
//!   and `nu > 2.05` (`k = 2`, `Var(eta) = s^2 * nu / (nu - 2)`).
//! - [`MarginalDist::SkewNormal`]: Azzalini skew-normal `SN(xi, omega, alpha)`
//!   with `xi = -omega * delta * sqrt(2 / pi)` (`k = 2`, `E[eta] == 0`).
//! - [`MarginalDist::SkewT`]: Azzalini-Capitanio skew-t `ST(xi, omega, alpha, nu)`
//!   with `nu > 2.10` and `xi = -omega * delta * b_nu` (`k = 3`, `E[eta] == 0`).

use crate::stats::special::{inv_mills, ln_gamma, log_normal_cdf, normal_inv_cdf, pnorm};
use serde::{Deserialize, Serialize};

/// Floor on marginal scale/standard-deviation parameters to prevent singular densities.
pub(crate) const MARGINAL_SCALE_FLOOR: f64 = 1e-3;
/// Ceiling on marginal scale/standard-deviation parameters.
const MARGINAL_SCALE_CEIL: f64 = 50.0;
/// Maximum absolute Azzalini slant parameter `|alpha|` (`|delta| <= 0.9965`).
const MAX_ABS_ALPHA: f64 = 12.0;
/// Minimum degrees of freedom above `2.0` so mean and variance exist and stay well-conditioned.
const NU_SHIFT: f64 = 2.10;
/// Maximum degrees of freedom before a t/skew-t marginal is indistinguishable from Gaussian.
const MAX_NU: f64 = 120.0;

/// 16-point Gauss-Legendre nodes on `[-1, 1]` (non-negative half; symmetric).
const GL16_X: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

/// 16-point Gauss-Legendre weights on `[-1, 1]` paired with [`GL16_X`].
const GL16_W: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_58,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_74,
    0.124_628_971_255_533_88,
    0.095_158_511_682_492_79,
    0.062_253_523_938_647_894,
    0.027_152_459_411_754_096,
];

/// High-precision standard normal quantile `Phi^-1(p)` via Acklam + one Halley refinement
/// against [`pnorm`].
pub(crate) fn probit_exact(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let z0 = normal_inv_cdf(p);
    if !z0.is_finite() {
        return z0;
    }
    let pdf = (-0.5 * z0 * z0).exp() * (1.0 / std::f64::consts::TAU.sqrt());
    if pdf <= 1e-300 {
        return z0;
    }
    let err = pnorm(z0) - p;
    let u = err / pdf;
    // Halley step using phi'(z)/phi(z) = -z.
    let step = u / (1.0 - 0.5 * z0 * u).max(0.5);
    z0 - step
}

/// Modified Lentz continued fraction for the regularised incomplete beta `I_x(a, b)`.
fn betai_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 3e-15;
    const FPMIN: f64 = 1e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAXIT {
        let mf = m as f64;
        let m2 = 2.0 * mf;

        let mut aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Log of the lower-tail regularised incomplete beta `ln I_x(a, b)` when `x <= (a+1)/(a+b+2)`.
fn log_betai_direct(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let log_front = a * x.ln() + b * (-x).ln_1p() - lbeta;
    let cf = betai_cf(a, b, x).max(1e-300);
    log_front + cf.ln() - a.ln()
}

/// Regularised incomplete beta `I_x(a, b)` for `x in [0, 1]`.
pub(crate) fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        1.0 - log_betai_direct(b, a, 1.0 - x).exp()
    } else {
        log_betai_direct(a, b, x).exp()
    }
}

/// Standard Student-t log-PDF `ln t_nu(x)`.
#[inline]
pub(crate) fn student_t_log_pdf(x: f64, nu: f64) -> f64 {
    ln_gamma(0.5 * (nu + 1.0))
        - ln_gamma(0.5 * nu)
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - 0.5 * (nu + 1.0) * (x * x / nu).ln_1p()
}

/// Standard Student-t CDF `T_nu(x)`.
#[inline]
pub(crate) fn student_t_cdf(x: f64, nu: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    let x2 = x * x;
    if x2 < nu {
        // Avoid catastrophic cancellation in `1 - nu / (nu + x^2)` near x = 0:
        // I_z(nu/2, 1/2) = 1 - I_{1-z}(1/2, nu/2) with 1 - z = x^2 / (nu + x^2).
        let w = x2 / (nu + x2);
        let half_ib = 0.5 * betai(0.5, 0.5 * nu, w);
        if x >= 0.0 {
            (0.5 + half_ib).clamp(0.0, 1.0)
        } else {
            (0.5 - half_ib).clamp(0.0, 1.0)
        }
    } else {
        let z = nu / (nu + x2);
        let half_ib = 0.5 * betai(0.5 * nu, 0.5, z);
        if x >= 0.0 {
            (1.0 - half_ib).clamp(0.0, 1.0)
        } else {
            half_ib.clamp(0.0, 1.0)
        }
    }
}

/// Numerically stable `ln T_nu(x)` without lower-tail underflow.
#[inline]
pub(crate) fn log_student_t_cdf(x: f64, nu: f64) -> f64 {
    if x >= 0.0 {
        student_t_cdf(x, nu).max(1e-300).ln()
    } else {
        let z = nu / (nu + x * x);
        let a = 0.5 * nu;
        let b = 0.5;
        if z <= (a + 1.0) / (a + b + 2.0) {
            -std::f64::consts::LN_2 + log_betai_direct(a, b, z)
        } else {
            student_t_cdf(x, nu).max(1e-300).ln()
        }
    }
}

/// Standard Student-t quantile `T_nu^-1(p)` via safeguarded Newton-Halley.
pub(crate) fn student_t_quantile(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-16 {
        return 0.0;
    }
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p_low = p.min(1.0 - p);

    // Cornish-Fisher / power-tail seed so extreme tails start near the root.
    let z = -probit_exact(p_low);
    let g1 = (z * z * z + z) / (4.0 * nu);
    let mut x = (z + g1).max(0.0);
    if p_low < 1e-4 {
        // Asymptotic tail: T_nu(-x) ~ c_nu * x^-nu / nu.
        let log_c =
            ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * std::f64::consts::PI.ln()
                + 0.5 * nu * nu.ln();
        let tail_est = ((log_c - nu.ln() - p_low.ln()) / nu).exp();
        if tail_est.is_finite() && tail_est > 0.0 {
            x = tail_est;
        }
    }

    // Establish a bracket on x >= 0 for F(-x) = p_low (monotone decreasing in x).
    let mut lo = 0.0_f64;
    let mut hi = x.max(1.0);
    while student_t_cdf(-hi, nu) > p_low && hi < 1e12 {
        lo = hi;
        hi *= 2.0;
    }
    x = x.clamp(lo, hi);

    for _ in 0..64 {
        let fx = student_t_cdf(-x, nu) - p_low;
        if fx.abs() <= 1e-15 * p_low.max(1e-15) {
            break;
        }
        if fx > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let pdf = student_t_log_pdf(x, nu).exp();
        // d/dx [student_t_cdf(-x, nu) - p_low] = -pdf.
        let u = if pdf > 1e-300 { -fx / pdf } else { f64::NAN };
        // Halley correction: f'(x)/f(x) for -t_nu(x) is -(nu+1)x/(nu+x^2).
        let ratio = -(nu + 1.0) * x / (nu + x * x);
        let cand = if u.is_finite() { x - u / (1.0 - 0.5 * u * ratio).max(0.25) } else { f64::NAN };
        let next = if cand.is_finite() && cand > lo && cand < hi { cand } else { 0.5 * (lo + hi) };
        if (next - x).abs() <= 1e-13 * (1.0 + x) {
            x = next;
            break;
        }
        x = next;
    }
    sign * x
}

/// Twice Owen's T function `2 * T(z, alpha) = (1/pi) * int_0^alpha exp(-0.5 * z^2 * (1 + t^2)) / (1 + t^2) dt`
/// computed by 16-point Gauss-Legendre quadrature on `[0, alpha]`.
fn two_owens_t(z: f64, alpha: f64) -> f64 {
    if alpha == 0.0 || !z.is_finite() {
        return 0.0;
    }
    let half_a = 0.5 * alpha;
    let z2 = z * z;
    if z2 > 1500.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..8 {
        let dx = half_a * GL16_X[i];
        let t_pos = half_a + dx;
        let t_neg = half_a - dx;
        let f_pos = (-0.5 * z2 * (1.0 + t_pos * t_pos)).exp() / (1.0 + t_pos * t_pos);
        let f_neg = (-0.5 * z2 * (1.0 + t_neg * t_neg)).exp() / (1.0 + t_neg * t_neg);
        sum += GL16_W[i] * (f_pos + f_neg);
    }
    (half_a * sum) * std::f64::consts::FRAC_1_PI
}

/// Skew-t CDF correction `(1/pi) * int_0^alpha (1 + (1 + t^2) z^2 / nu)^(-nu/2) / (1 + t^2) dt`
/// computed by 16-point Gauss-Legendre quadrature on `[0, alpha]`.
fn skew_t_cdf_integral(z: f64, alpha: f64, nu: f64) -> f64 {
    if alpha == 0.0 || !z.is_finite() {
        return 0.0;
    }
    let half_a = 0.5 * alpha;
    let z2_over_nu = z * z / nu;
    let neg_half_nu = -0.5 * nu;
    let mut sum = 0.0;
    for i in 0..8 {
        let dx = half_a * GL16_X[i];
        let t_pos = half_a + dx;
        let t_neg = half_a - dx;
        let s_pos = 1.0 + t_pos * t_pos;
        let s_neg = 1.0 + t_neg * t_neg;
        let f_pos = (neg_half_nu * (s_pos * z2_over_nu).ln_1p()).exp() / s_pos;
        let f_neg = (neg_half_nu * (s_neg * z2_over_nu).ln_1p()).exp() / s_neg;
        sum += GL16_W[i] * (f_pos + f_neg);
    }
    (half_a * sum) * std::f64::consts::FRAC_1_PI
}

#[inline]
fn skew_t_b_nu(nu: f64) -> f64 {
    (nu / std::f64::consts::PI).sqrt() * (ln_gamma(0.5 * (nu - 1.0)) - ln_gamma(0.5 * nu)).exp()
}

/// Zero-mean univariate marginal distribution for a single random effect `eta_j`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarginalDist {
    /// Zero-mean Gaussian `N(0, sd^2)`.
    Normal {
        /// Marginal standard deviation `sd > 0`.
        sd: f64,
    },
    /// Symmetric zero-mean Student-t with scale `scale > 0` and `nu > 2.10`.
    StudentT {
        /// Scale parameter `s > 0` (`SD = s * sqrt(nu / (nu - 2))`).
        scale: f64,
        /// Degrees of freedom `nu > 2.10`.
        nu: f64,
    },
    /// Zero-mean Azzalini skew-normal `SN(xi, omega, alpha)` with `xi = -omega * delta * sqrt(2/pi)`.
    SkewNormal {
        /// Scale parameter `omega > 0`.
        omega: f64,
        /// Slant / skewness parameter `alpha`.
        alpha: f64,
    },
    /// Zero-mean Azzalini skew-t `ST(xi, omega, alpha, nu)` with `xi = -omega * delta * b_nu`.
    SkewT {
        /// Scale parameter `omega > 0`.
        omega: f64,
        /// Slant / skewness parameter `alpha`.
        alpha: f64,
        /// Degrees of freedom `nu > 2.10`.
        nu: f64,
    },
}

/// Serializable summary of one fitted marginal distribution for reporting and reconstruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginalSummary {
    /// ETA parameter name (e.g. `"ETA_CL"`).
    pub eta: String,
    /// Canonical family label: `"normal"`, `"student_t"`, `"skew_normal"`, or `"skew_t"`.
    pub family: String,
    /// Analytical mean (identically `0.0` by zero-mean identifiability constraint).
    pub mean: f64,
    /// Marginal standard deviation `sqrt(Var(eta))`.
    pub sd: f64,
    /// Analytical skewness `E[eta^3] / sd^3` when finite (`nu > 3` for t-based families).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skewness: Option<f64>,
    /// Analytical excess kurtosis `E[eta^4] / sd^4 - 3` when finite (`nu > 4` for t-based families).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excess_kurtosis: Option<f64>,
    /// Family-specific natural parameters in canonical order.
    pub params: Vec<(String, f64)>,
    /// Approximate standard errors paired with `params`.
    #[serde(default)]
    pub se: Vec<(String, f64)>,
}

impl MarginalDist {
    /// Construct a zero-mean Gaussian marginal with clamped `sd`.
    pub fn normal(sd: f64) -> Self {
        Self::Normal { sd: sd.clamp(MARGINAL_SCALE_FLOOR, MARGINAL_SCALE_CEIL) }
    }

    /// Construct a zero-mean Student-t marginal with clamped `scale` and `nu`.
    pub fn student_t(scale: f64, nu: f64) -> Self {
        Self::StudentT {
            scale: scale.clamp(MARGINAL_SCALE_FLOOR, MARGINAL_SCALE_CEIL),
            nu: nu.clamp(NU_SHIFT + 0.05, MAX_NU),
        }
    }

    /// Construct a zero-mean Azzalini skew-normal marginal.
    pub fn skew_normal(omega: f64, alpha: f64) -> Self {
        Self::SkewNormal {
            omega: omega.clamp(MARGINAL_SCALE_FLOOR, MARGINAL_SCALE_CEIL),
            alpha: alpha.clamp(-MAX_ABS_ALPHA, MAX_ABS_ALPHA),
        }
    }

    /// Construct a zero-mean Azzalini skew-t marginal.
    pub fn skew_t(omega: f64, alpha: f64, nu: f64) -> Self {
        Self::SkewT {
            omega: omega.clamp(MARGINAL_SCALE_FLOOR, MARGINAL_SCALE_CEIL),
            alpha: alpha.clamp(-MAX_ABS_ALPHA, MAX_ABS_ALPHA),
            nu: nu.clamp(NU_SHIFT + 0.05, MAX_NU),
        }
    }

    /// Canonical family identifier.
    pub fn family_name(&self) -> &'static str {
        match self {
            Self::Normal { .. } => "normal",
            Self::StudentT { .. } => "student_t",
            Self::SkewNormal { .. } => "skew_normal",
            Self::SkewT { .. } => "skew_t",
        }
    }

    /// Number of free scalar parameters estimated for this marginal family.
    pub fn n_params(&self) -> usize {
        match self {
            Self::Normal { .. } => 1,
            Self::StudentT { .. } | Self::SkewNormal { .. } => 2,
            Self::SkewT { .. } => 3,
        }
    }

    /// Analytical marginal standard deviation `sqrt(Var(eta))`.
    pub fn sd(&self) -> f64 {
        match *self {
            Self::Normal { sd } => sd,
            Self::StudentT { scale, nu } => scale * (nu / (nu - 2.0)).sqrt(),
            Self::SkewNormal { omega, alpha } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let mu_z = delta * std::f64::consts::FRAC_2_PI.sqrt();
                omega * (1.0 - mu_z * mu_z).max(1e-12).sqrt()
            }
            Self::SkewT { omega, alpha, nu } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let mu_z = delta * skew_t_b_nu(nu);
                let var_z = (nu / (nu - 2.0) - mu_z * mu_z).max(1e-12);
                omega * var_z.sqrt()
            }
        }
    }

    /// Analytical skewness when the third moment exists.
    pub fn skewness(&self) -> Option<f64> {
        match *self {
            Self::Normal { .. } | Self::StudentT { .. } => Some(0.0),
            Self::SkewNormal { alpha, .. } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let mu_z = delta * std::f64::consts::FRAC_2_PI.sqrt();
                let v = (1.0 - mu_z * mu_z).max(1e-12);
                Some(0.5 * (4.0 - std::f64::consts::PI) * mu_z.powi(3) / v.powf(1.5))
            }
            Self::SkewT { alpha, nu, .. } => {
                if nu <= 3.05 {
                    return None;
                }
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let b = skew_t_b_nu(nu);
                let mu_z = delta * b;
                let ez2 = nu / (nu - 2.0);
                let ez3 = (nu / (nu - 3.0)) * b * delta * (3.0 - delta * delta);
                let var_z = (ez2 - mu_z * mu_z).max(1e-12);
                let mu3 = ez3 - 3.0 * mu_z * ez2 + 2.0 * mu_z.powi(3);
                Some(mu3 / var_z.powf(1.5))
            }
        }
    }

    /// Analytical excess kurtosis when the fourth moment exists.
    pub fn excess_kurtosis(&self) -> Option<f64> {
        match *self {
            Self::Normal { .. } => Some(0.0),
            Self::StudentT { nu, .. } => (nu > 4.05).then(|| 6.0 / (nu - 4.0)),
            Self::SkewNormal { alpha, .. } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let mu_z = delta * std::f64::consts::FRAC_2_PI.sqrt();
                let v = (1.0 - mu_z * mu_z).max(1e-12);
                Some(2.0 * (std::f64::consts::PI - 3.0) * mu_z.powi(4) / (v * v))
            }
            Self::SkewT { alpha, nu, .. } => {
                if nu <= 4.05 {
                    return None;
                }
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                let b = skew_t_b_nu(nu);
                let mu_z = delta * b;
                let ez2 = nu / (nu - 2.0);
                let ez3 = (nu / (nu - 3.0)) * b * delta * (3.0 - delta * delta);
                let ez4 = 3.0 * nu * nu / ((nu - 2.0) * (nu - 4.0));
                let var_z = (ez2 - mu_z * mu_z).max(1e-12);
                let mu4 = ez4 - 4.0 * mu_z * ez3 + 6.0 * mu_z * mu_z * ez2 - 3.0 * mu_z.powi(4);
                Some(mu4 / (var_z * var_z) - 3.0)
            }
        }
    }

    #[inline]
    fn location(&self) -> f64 {
        match *self {
            Self::Normal { .. } | Self::StudentT { .. } => 0.0,
            Self::SkewNormal { omega, alpha } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                -omega * delta * std::f64::consts::FRAC_2_PI.sqrt()
            }
            Self::SkewT { omega, alpha, nu } => {
                let delta = alpha / (1.0 + alpha * alpha).sqrt();
                -omega * delta * skew_t_b_nu(nu)
            }
        }
    }

    /// Log marginal density `ln f(eta)`.
    pub fn log_pdf(&self, eta: f64) -> f64 {
        if !eta.is_finite() {
            return f64::NEG_INFINITY;
        }
        match *self {
            Self::Normal { sd } => {
                let z = eta / sd;
                -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - sd.ln()
            }
            Self::StudentT { scale, nu } => student_t_log_pdf(eta / scale, nu) - scale.ln(),
            Self::SkewNormal { omega, alpha } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                std::f64::consts::LN_2 - omega.ln() - 0.5 * std::f64::consts::TAU.ln() - 0.5 * z * z
                    + log_normal_cdf(alpha * z)
            }
            Self::SkewT { omega, alpha, nu } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                let r = ((nu + 1.0) / (nu + z * z)).sqrt();
                let w = alpha * z * r;
                std::f64::consts::LN_2 - omega.ln()
                    + student_t_log_pdf(z, nu)
                    + log_student_t_cdf(w, nu + 1.0)
            }
        }
    }

    /// Marginal density `f(eta) = exp(log_pdf(eta))`.
    #[inline]
    pub fn pdf(&self, eta: f64) -> f64 {
        self.log_pdf(eta).exp()
    }

    /// Exact analytic derivative `d/d(eta) [ln f(eta)]`.
    pub fn d_log_pdf_d_eta(&self, eta: f64) -> f64 {
        match *self {
            Self::Normal { sd } => -eta / (sd * sd),
            Self::StudentT { scale, nu } => {
                let z = eta / scale;
                -((nu + 1.0) * z) / (scale * (nu + z * z))
            }
            Self::SkewNormal { omega, alpha } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                (-z + alpha * inv_mills(alpha * z)) / omega
            }
            Self::SkewT { omega, alpha, nu } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                let denom = nu + z * z;
                let r = ((nu + 1.0) / denom).sqrt();
                let w = alpha * z * r;
                let log_ratio = student_t_log_pdf(w, nu + 1.0) - log_student_t_cdf(w, nu + 1.0);
                let ratio = if log_ratio < 700.0 { log_ratio.exp() } else { 0.0 };
                let dw_dz = (alpha * nu / denom) * r;
                (-(nu + 1.0) * z / denom + ratio * dw_dz) / omega
            }
        }
    }

    /// Marginal CDF `F(eta) in [0, 1]`.
    pub fn cdf(&self, eta: f64) -> f64 {
        if !eta.is_finite() {
            return if eta.is_sign_negative() { 0.0 } else { 1.0 };
        }
        match *self {
            Self::Normal { sd } => pnorm(eta / sd).clamp(0.0, 1.0),
            Self::StudentT { scale, nu } => student_t_cdf(eta / scale, nu),
            Self::SkewNormal { omega, alpha } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                (pnorm(z) - two_owens_t(z, alpha)).clamp(0.0, 1.0)
            }
            Self::SkewT { omega, alpha, nu } => {
                let xi = self.location();
                let z = (eta - xi) / omega;
                (student_t_cdf(z, nu) - skew_t_cdf_integral(z, alpha, nu)).clamp(0.0, 1.0)
            }
        }
    }

    /// Marginal quantile `F^-1(u)` for `u in (0, 1)`.
    pub fn quantile(&self, u: f64) -> f64 {
        if u <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if u >= 1.0 {
            return f64::INFINITY;
        }
        let u_c = u.clamp(1e-15, 1.0 - 1e-15);
        match *self {
            Self::Normal { sd } => sd * probit_exact(u_c),
            Self::StudentT { scale, nu } => scale * student_t_quantile(u_c, nu),
            Self::SkewNormal { .. } | Self::SkewT { .. } => {
                let sd = self.sd().max(MARGINAL_SCALE_FLOOR);
                let mut x = match *self {
                    Self::SkewT { nu, .. } => sd * student_t_quantile(u_c, nu),
                    _ => sd * probit_exact(u_c),
                };
                if !x.is_finite() {
                    x = 0.0;
                }
                let mut step = sd.max(0.5);
                let mut lo = x - step;
                let mut hi = x + step;
                for _ in 0..80 {
                    if self.cdf(lo) <= u_c {
                        break;
                    }
                    hi = lo;
                    step *= 2.0;
                    lo -= step;
                }
                step = sd.max(0.5);
                for _ in 0..80 {
                    if self.cdf(hi) >= u_c {
                        break;
                    }
                    lo = hi;
                    step *= 2.0;
                    hi += step;
                }
                x = x.clamp(lo, hi);
                for _ in 0..64 {
                    let fx = self.cdf(x) - u_c;
                    if fx.abs() <= 1e-14 {
                        break;
                    }
                    if fx < 0.0 {
                        lo = x;
                    } else {
                        hi = x;
                    }
                    let pdf = self.pdf(x);
                    let u_step = if pdf > 1e-300 { fx / pdf } else { f64::NAN };
                    let dlogf = self.d_log_pdf_d_eta(x);
                    let cand = if u_step.is_finite() && dlogf.is_finite() {
                        x - u_step / (1.0 - 0.5 * u_step * dlogf).clamp(0.25, 4.0)
                    } else {
                        f64::NAN
                    };
                    let next = if cand.is_finite() && cand > lo && cand < hi {
                        cand
                    } else {
                        0.5 * (lo + hi)
                    };
                    if (next - x).abs() <= 1e-13 * (1.0 + x.abs()) {
                        x = next;
                        break;
                    }
                    x = next;
                }
                x
            }
        }
    }

    /// Pack parameters onto unconstrained coordinates `psi in R^p`.
    pub fn pack_psi(&self) -> Vec<f64> {
        match *self {
            Self::Normal { sd } => vec![sd.ln()],
            Self::StudentT { scale, nu } => vec![scale.ln(), (nu - NU_SHIFT).max(1e-3).ln()],
            Self::SkewNormal { omega, alpha } => vec![omega.ln(), alpha],
            Self::SkewT { omega, alpha, nu } => {
                vec![omega.ln(), alpha, (nu - NU_SHIFT).max(1e-3).ln()]
            }
        }
    }

    /// Unpack unconstrained coordinates `psi` into the same family variant.
    pub fn from_psi(&self, psi: &[f64]) -> Self {
        let clamp_scale =
            |ls: f64| ls.clamp(MARGINAL_SCALE_FLOOR.ln(), MARGINAL_SCALE_CEIL.ln()).exp();
        let clamp_nu = |ln_dnu: f64| {
            (NU_SHIFT + ln_dnu.clamp(-3.0, (MAX_NU - NU_SHIFT).ln()).exp())
                .clamp(NU_SHIFT + 0.05, MAX_NU)
        };
        match self {
            Self::Normal { .. } => Self::normal(clamp_scale(psi[0])),
            Self::StudentT { .. } => Self::student_t(clamp_scale(psi[0]), clamp_nu(psi[1])),
            Self::SkewNormal { .. } => {
                Self::skew_normal(clamp_scale(psi[0]), psi[1].clamp(-MAX_ABS_ALPHA, MAX_ABS_ALPHA))
            }
            Self::SkewT { .. } => Self::skew_t(
                clamp_scale(psi[0]),
                psi[1].clamp(-MAX_ABS_ALPHA, MAX_ABS_ALPHA),
                clamp_nu(psi[2]),
            ),
        }
    }

    /// Named natural parameters in canonical order.
    pub fn named_params(&self) -> Vec<(String, f64)> {
        match *self {
            Self::Normal { sd } => vec![("sd".into(), sd)],
            Self::StudentT { scale, nu } => vec![("scale".into(), scale), ("nu".into(), nu)],
            Self::SkewNormal { omega, alpha } => {
                vec![("omega".into(), omega), ("alpha".into(), alpha)]
            }
            Self::SkewT { omega, alpha, nu } => {
                vec![("omega".into(), omega), ("alpha".into(), alpha), ("nu".into(), nu)]
            }
        }
    }

    /// Reconstruct a [`MarginalDist`] from a [`MarginalSummary`].
    pub fn from_summary(summary: &MarginalSummary) -> Self {
        let get = |key: &str| -> Option<f64> {
            summary.params.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
        };
        match summary.family.as_str() {
            "normal" => Self::normal(get("sd").unwrap_or(summary.sd)),
            "student_t" => match (get("scale"), get("nu")) {
                (Some(s), Some(nu)) => Self::student_t(s, nu),
                _ => Self::normal(summary.sd),
            },
            "skew_normal" => match (get("omega"), get("alpha")) {
                (Some(w), Some(a)) => Self::skew_normal(w, a),
                _ => Self::normal(summary.sd),
            },
            "skew_t" => match (get("omega"), get("alpha"), get("nu")) {
                (Some(w), Some(a), Some(nu)) => Self::skew_t(w, a, nu),
                _ => Self::normal(summary.sd),
            },
            _ => Self::normal(summary.sd),
        }
    }

    /// Build a [`MarginalSummary`] including delta-method SEs computed from the full
    /// `p x p` observed Fisher information on `samples` (weighted by `sample_weight`).
    pub fn to_summary(
        &self, eta_name: &str, samples: &[f64], sample_weight: f64,
    ) -> MarginalSummary {
        let params = self.named_params();
        let se = self.observed_se(samples, sample_weight);
        MarginalSummary {
            eta: eta_name.to_string(),
            family: self.family_name().to_string(),
            mean: 0.0,
            sd: self.sd(),
            skewness: self.skewness(),
            excess_kurtosis: self.excess_kurtosis(),
            params,
            se,
        }
    }

    /// Weighted log-likelihood `weight * sum_i ln f(samples[i])`.
    pub fn weighted_log_lik(&self, samples: &[f64], weight: f64) -> f64 {
        weight * samples.iter().map(|&x| self.log_pdf(x)).sum::<f64>()
    }

    /// Score and positive-definite observed Fisher information on unconstrained `psi`.
    pub fn score_and_obs_info(&self, samples: &[f64], weight: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let psi = self.pack_psi();
        let p = psi.len();
        let h = 1e-4_f64;
        let ll0 = self.weighted_log_lik(samples, weight);

        let mut grad = vec![0.0; p];
        let mut neg_hess = vec![vec![0.0; p]; p];
        let mut ll_plus = vec![0.0; p];
        let mut ll_minus = vec![0.0; p];

        for j in 0..p {
            let mut p_up = psi.clone();
            let mut p_dn = psi.clone();
            p_up[j] += h;
            p_dn[j] -= h;
            ll_plus[j] = self.from_psi(&p_up).weighted_log_lik(samples, weight);
            ll_minus[j] = self.from_psi(&p_dn).weighted_log_lik(samples, weight);
            grad[j] = (ll_plus[j] - ll_minus[j]) / (2.0 * h);
            let diag = -(ll_plus[j] - 2.0 * ll0 + ll_minus[j]) / (h * h);
            neg_hess[j][j] = if diag.is_finite() { diag.max(1e-4) } else { 1.0 };
        }

        for i in 0..p {
            for j in (i + 1)..p {
                let mut pp = psi.clone();
                let mut pm = psi.clone();
                let mut mp = psi.clone();
                let mut mm = psi.clone();
                pp[i] += h;
                pp[j] += h;
                pm[i] += h;
                pm[j] -= h;
                mp[i] -= h;
                mp[j] += h;
                mm[i] -= h;
                mm[j] -= h;
                let v_pp = self.from_psi(&pp).weighted_log_lik(samples, weight);
                let v_pm = self.from_psi(&pm).weighted_log_lik(samples, weight);
                let v_mp = self.from_psi(&mp).weighted_log_lik(samples, weight);
                let v_mm = self.from_psi(&mm).weighted_log_lik(samples, weight);
                let cross = -(v_pp - v_pm - v_mp + v_mm) / (4.0 * h * h);
                let cross = if cross.is_finite() { cross } else { 0.0 };
                neg_hess[i][j] = cross;
                neg_hess[j][i] = cross;
            }
        }

        // Enforce positive-definiteness via small Levenberg diagonal damping if needed.
        make_spd_small(&mut neg_hess);
        (grad, neg_hess)
    }

    /// Approximate delta-method SEs on natural parameters from the full `p x p` inverse
    /// observed Fisher information.
    pub fn observed_se(&self, samples: &[f64], weight: f64) -> Vec<(String, f64)> {
        let names: Vec<String> = self.named_params().into_iter().map(|(n, _)| n).collect();
        if samples.len() < 4 || weight <= 0.0 {
            return names.into_iter().map(|n| (n, f64::NAN)).collect();
        }
        let (_, info) = self.score_and_obs_info(samples, weight);
        let Some(cov_psi) = invert_small(&info) else {
            return names.into_iter().map(|n| (n, f64::NAN)).collect();
        };
        let jac_diag: Vec<f64> = match *self {
            Self::Normal { sd } => vec![sd],
            Self::StudentT { scale, nu } => vec![scale, (nu - NU_SHIFT).max(1e-3)],
            Self::SkewNormal { omega, .. } => vec![omega, 1.0],
            Self::SkewT { omega, nu, .. } => vec![omega, 1.0, (nu - NU_SHIFT).max(1e-3)],
        };
        names
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                let v = cov_psi[i][i];
                let se = if v.is_finite() && v > 0.0 {
                    (jac_diag[i] * v.sqrt()).abs()
                } else {
                    f64::NAN
                };
                (name, se)
            })
            .collect()
    }
}

/// Robbins-Monro score & observed-information stochastic approximation state for one marginal.
#[derive(Debug, Clone)]
pub struct MarginalSaState {
    avg_score: Vec<f64>,
    avg_info: Vec<Vec<f64>>,
    initialized: bool,
}

impl MarginalSaState {
    /// Create an uninitialized SA state.
    pub fn new() -> Self {
        Self { avg_score: Vec::new(), avg_info: Vec::new(), initialized: false }
    }

    /// Reset state when the marginal family changes at `k = K_select`.
    pub fn reset(&mut self) {
        self.avg_score.clear();
        self.avg_info.clear();
        self.initialized = false;
    }

    /// Apply one Robbins-Monro score & Fisher-information stochastic approximation update.
    ///
    /// For `Normal`, the exact sufficient-statistic variance update
    /// `s2_k = (1 - gamma) * s2_{k-1} + gamma * mean(eta^2)` is equivalent and closed-form;
    /// for all non-exponential families (`StudentT`, `SkewNormal`, `SkewT`), `avg_score -> 0`
    /// solves `E_{eta | y}[grad_psi ln f(eta; psi)] = 0` without single-draw MLE Jensen bias.
    pub fn step(&mut self, dist: &mut MarginalDist, samples: &[f64], gamma: f64) {
        if samples.is_empty() || gamma <= 0.0 {
            return;
        }
        if let MarginalDist::Normal { sd } = *dist {
            let mean_sq = samples.iter().map(|&x| x * x).sum::<f64>() / samples.len() as f64;
            let var_new = (1.0 - gamma) * (sd * sd) + gamma * mean_sq;
            *dist = MarginalDist::normal(var_new.max(0.0).sqrt());
            return;
        }

        let (g, info) = dist.score_and_obs_info(samples, 1.0);
        let p = g.len();
        if !self.initialized || self.avg_score.len() != p {
            self.avg_score = g.iter().map(|&v| gamma * v).collect();
            self.avg_info = info;
            self.initialized = true;
        } else {
            for i in 0..p {
                self.avg_score[i] = (1.0 - gamma) * self.avg_score[i] + gamma * g[i];
                for j in 0..p {
                    self.avg_info[i][j] = (1.0 - gamma) * self.avg_info[i][j] + gamma * info[i][j];
                }
            }
        }
        make_spd_small(&mut self.avg_info);
        if let Some(mut delta) = solve_small(&self.avg_info, &self.avg_score) {
            // Trust-region cap on unconstrained step so exploration outliers never jump wildly.
            let max_step = 0.25_f64;
            let norm_inf = delta.iter().fold(0.0_f64, |acc, &d| acc.max(d.abs()));
            if norm_inf > max_step {
                let scale = max_step / norm_inf;
                for d in &mut delta {
                    *d *= scale;
                }
            }
            let mut psi = dist.pack_psi();
            for i in 0..p {
                psi[i] += delta[i];
            }
            *dist = dist.from_psi(&psi);
        }
    }
}

impl Default for MarginalSaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Fit a specific marginal family template by damped Newton/line-search MLE on pooled samples.
pub fn fit_marginal_weighted(template: MarginalDist, samples: &[f64], weight: f64) -> MarginalDist {
    if samples.len() < 4 || weight <= 0.0 {
        return template;
    }
    let sample_sd = (samples.iter().map(|&x| x * x).sum::<f64>() / samples.len() as f64)
        .sqrt()
        .clamp(MARGINAL_SCALE_FLOOR, MARGINAL_SCALE_CEIL);

    if matches!(template, MarginalDist::Normal { .. }) {
        return MarginalDist::normal(sample_sd);
    }

    // Method-of-moments sample skewness to seed asymmetric families cleanly on both signs.
    let m3 = samples.iter().map(|&x| (x / sample_sd).powi(3)).sum::<f64>() / samples.len() as f64;
    let alpha_seed = (m3 * 2.5).clamp(-4.0, 4.0);

    let starts: Vec<MarginalDist> = match template {
        MarginalDist::Normal { .. } => vec![MarginalDist::normal(sample_sd)],
        MarginalDist::StudentT { .. } => vec![
            MarginalDist::student_t(sample_sd * 0.85, 5.0),
            MarginalDist::student_t(sample_sd * 0.95, 15.0),
        ],
        MarginalDist::SkewNormal { .. } => vec![
            MarginalDist::skew_normal(sample_sd * 1.15, alpha_seed),
            MarginalDist::skew_normal(sample_sd, 0.0),
        ],
        MarginalDist::SkewT { .. } => vec![
            MarginalDist::skew_t(sample_sd, alpha_seed, 6.0),
            MarginalDist::skew_t(sample_sd * 0.9, 0.0, 6.0),
        ],
    };

    let mut best = starts[0];
    let mut best_ll = best.weighted_log_lik(samples, weight);

    for start in starts {
        let mut cur = start;
        let mut cur_ll = cur.weighted_log_lik(samples, weight);
        for _ in 0..35 {
            let (g, mut info) = cur.score_and_obs_info(samples, weight);
            let g_norm = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if g_norm < 1e-5 {
                break;
            }
            for i in 0..info.len() {
                info[i][i] += 1e-3;
            }
            let Some(dir) = solve_small(&info, &g) else {
                break;
            };
            let psi = cur.pack_psi();
            let mut step = 1.0_f64;
            let norm_inf = dir.iter().fold(0.0_f64, |a, &d| a.max(d.abs()));
            if norm_inf > 1.5 {
                step = 1.5 / norm_inf;
            }
            let mut improved = false;
            for _ in 0..10 {
                let cand_psi: Vec<f64> =
                    psi.iter().zip(&dir).map(|(&x, &d)| x + step * d).collect();
                let cand = cur.from_psi(&cand_psi);
                let cand_ll = cand.weighted_log_lik(samples, weight);
                if cand_ll.is_finite() && cand_ll > cur_ll + 1e-9 {
                    cur = cand;
                    cur_ll = cand_ll;
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            if !improved {
                break;
            }
        }
        if cur_ll > best_ll {
            best = cur;
            best_ll = cur_ll;
        }
    }
    best
}

/// Select the best zero-mean marginal family by BIC on `samples` weighted by `weight`,
/// with penalty `k * ln(n_subjects)`.
///
/// When `allow_flexible == false` (or `n_subjects < 12`), returns [`MarginalDist::Normal`].
pub fn select_marginal_bic(
    samples: &[f64], weight: f64, n_subjects: usize, allow_flexible: bool,
) -> MarginalDist {
    let normal = fit_marginal_weighted(MarginalDist::normal(0.3), samples, weight);
    if !allow_flexible || n_subjects < 12 || samples.len() < 12 {
        return normal;
    }

    let ln_n = (n_subjects as f64).ln();
    let bic = |m: &MarginalDist| -> f64 {
        let ll = m.weighted_log_lik(samples, weight);
        if ll.is_finite() {
            -2.0 * ll + (m.n_params() as f64) * ln_n
        } else {
            f64::INFINITY
        }
    };

    let candidates = [
        normal,
        fit_marginal_weighted(MarginalDist::student_t(0.3, 6.0), samples, weight),
        fit_marginal_weighted(MarginalDist::skew_normal(0.3, 0.0), samples, weight),
        fit_marginal_weighted(MarginalDist::skew_t(0.3, 0.0, 6.0), samples, weight),
    ];

    let mut best = normal;
    let mut best_bic = bic(&normal);
    for cand in candidates.into_iter().skip(1) {
        let b = bic(&cand);
        if b + 1e-6 < best_bic {
            best = cand;
            best_bic = b;
        }
    }
    best
}

// ── Tiny dense linear algebra for p in {1, 2, 3} ──────────────────────────────

pub(crate) fn make_spd_small(a: &mut [Vec<f64>]) {
    let n = a.len();
    for i in 0..n {
        if !a[i][i].is_finite() || a[i][i] < 1e-6 {
            a[i][i] = 1e-4;
        }
    }
    for _ in 0..8 {
        if cholesky_small(a).is_some() {
            return;
        }
        let diag_scale = (0..n).map(|i| a[i][i].abs()).fold(1e-3_f64, f64::max);
        for (i, row) in a.iter_mut().enumerate() {
            row[i] += 0.1 * diag_scale;
        }
    }
}

fn cholesky_small(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for (lik, ljk) in l[i].iter().zip(&l[j]).take(j) {
                sum -= lik * ljk;
            }
            if i == j {
                if !sum.is_finite() || sum <= 1e-14 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

pub(crate) fn solve_small(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let l = cholesky_small(a)?;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * y[j];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i];
    }
    Some(x)
}

pub(crate) fn invert_small(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = solve_small(a, &e)?;
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn test_distributions() -> Vec<MarginalDist> {
        vec![
            MarginalDist::normal(0.35),
            MarginalDist::student_t(0.28, 3.5),
            MarginalDist::student_t(0.32, 8.0),
            MarginalDist::skew_normal(0.45, 3.5),
            MarginalDist::skew_normal(0.40, -2.8),
            MarginalDist::skew_t(0.38, 3.0, 4.5),
            MarginalDist::skew_t(0.34, -2.5, 6.0),
        ]
    }

    #[test]
    fn all_marginals_integrate_to_one_have_zero_mean_and_exact_variance() {
        // Composite Simpson / midpoint rule over [-L, L] with L = 40 * sd.
        for dist in test_distributions() {
            let sd = dist.sd();
            let bound = 45.0 * sd;
            let n = 20_000_usize;
            let dx = (2.0 * bound) / n as f64;
            let mut m0 = 0.0;
            let mut m1 = 0.0;
            let mut m2 = 0.0;
            for i in 0..n {
                let x = -bound + (i as f64 + 0.5) * dx;
                let p = dist.pdf(x);
                m0 += p;
                m1 += x * p;
                m2 += x * x * p;
            }
            m0 *= dx;
            m1 *= dx;
            m2 *= dx;
            assert!((m0 - 1.0).abs() < 1e-4, "{dist:?}: integral = {m0} (expected 1.0)");
            assert!(m1.abs() < 2e-4, "{dist:?}: mean = {m1} (expected identically 0.0)");
            assert!(
                (m2.sqrt() - sd).abs() / sd < 5e-3,
                "{dist:?}: numerical sd = {}, analytical sd = {sd}",
                m2.sqrt()
            );
        }
    }

    #[test]
    fn cdf_derivative_matches_pdf_and_log_pdf_derivative_matches_fd() {
        let h = 1e-6;
        for dist in test_distributions() {
            for &x in &[-1.2, -0.4, -0.05, 0.0, 0.15, 0.7, 1.5] {
                let fd_cdf = (dist.cdf(x + h) - dist.cdf(x - h)) / (2.0 * h);
                let pdf = dist.pdf(x);
                assert!(
                    (fd_cdf - pdf).abs() < 1e-7 * (1.0 + pdf),
                    "{dist:?} at x={x}: dCDF/dx={fd_cdf} vs pdf={pdf}"
                );

                let fd_dlog = (dist.log_pdf(x + h) - dist.log_pdf(x - h)) / (2.0 * h);
                let ana_dlog = dist.d_log_pdf_d_eta(x);
                assert!(
                    (fd_dlog - ana_dlog).abs() < 1e-6 * (1.0 + ana_dlog.abs()),
                    "{dist:?} at x={x}: FD d_log_pdf={fd_dlog} vs analytic={ana_dlog}"
                );
            }
        }
    }

    #[test]
    fn quantile_inverts_cdf_across_interior_and_tails() {
        for dist in test_distributions() {
            for &u in &[1e-5, 1e-3, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0 - 1e-3, 1.0 - 1e-5] {
                let x = dist.quantile(u);
                let u_back = dist.cdf(x);
                assert_relative_eq!(u_back, u, epsilon = 1e-10, max_relative = 1e-8);
            }
        }
    }

    #[test]
    fn bic_selection_picks_skew_t_or_skew_normal_on_skewed_data_and_normal_on_gaussian() {
        let mut rng = StdRng::seed_from_u64(2026);
        let gauss_truth = MarginalDist::normal(0.35);
        let gauss_samples: Vec<f64> = (0..300)
            .map(|_| gauss_truth.quantile(rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6)))
            .collect();
        let sel_g = select_marginal_bic(&gauss_samples, 1.0, 300, true);
        assert_eq!(sel_g.family_name(), "normal");

        let skew_truth = MarginalDist::skew_t(0.45, 5.0, 4.5);
        let skew_samples: Vec<f64> = (0..300)
            .map(|_| skew_truth.quantile(rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6)))
            .collect();
        let sel_s = select_marginal_bic(&skew_samples, 1.0, 300, true);
        assert_ne!(
            sel_s.family_name(),
            "normal",
            "expected non-Gaussian selection on strongly skewed heavy-tailed data"
        );
        assert!((sel_s.sd() - skew_truth.sd()).abs() / skew_truth.sd() < 0.20);
    }
}
