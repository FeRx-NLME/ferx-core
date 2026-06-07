/// Bivariate copula families for the vine-copula SAEM variant.
///
/// Each family provides:
/// - `log_density(u, v)` — log c(u,v;θ), the bivariate copula log-density.
/// - `h(u, v)` — h-function h(u|v;θ) = ∂C(u,v;θ)/∂v, the conditional CDF
///   of U given V=v. Used in the h-function recursion for vine density
///   evaluation (Aas et al., 2009).
/// - `h_inv(p, v)` — inverse h-function: the value of u such that h(u|v) = p.
/// - `fit(u, v)` — maximum-likelihood parameter estimate given pseudo-
///   observations (u_i, v_i) ∈ (0,1)².
///
/// # Families implemented
///
/// | Family         | Parameter range    | Tail dependence          |
/// |----------------|--------------------|--------------------------|
/// | Gaussian       | ρ ∈ (−1, 1)        | none                     |
/// | Student-t      | ρ ∈ (−1, 1), ν > 2 | symmetric (both tails)   |
/// | Clayton        | θ > 0              | lower tail               |
/// | Gumbel         | θ ≥ 1              | upper tail               |
/// | Frank          | θ ≠ 0              | symmetric (none in tail) |
///
/// # References
/// - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). Pair-copula
///   constructions of multiple dependence. Insurance: Mathematics and
///   Economics 44:182–198.
/// - Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.

// ── Special functions ─────────────────────────────────────────────────────────

/// Standard normal CDF Φ(x) via erf, matching `stats::special::normal_cdf`.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    crate::stats::special::normal_cdf(x)
}

/// Standard normal quantile Φ⁻¹(p) — Peter Acklam's rational approximation.
///
/// Accurate to ~1.15×10⁻⁹ for p ∈ (10⁻³⁰⁸, 1−10⁻¹⁶). Returns ±∞ outside
/// (0, 1). Reference: P. J. Acklam (2003), as refined by Voutier (2010).
pub fn normal_cdf_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Log-gamma function via Lanczos approximation (g=7, n=9; ~15 sig. figs.).
fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let xm1 = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (xm1 + (i + 1) as f64);
    }
    let t = xm1 + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + sum.ln()
}

/// Regularised incomplete beta function I_x(a, b) via continued fractions
/// (Lentz method; Numerical Recipes §6.4). Accurate to ~1e-12 for x ∈ [0,1].
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the symmetry relation I_x(a,b) = 1 - I_{1-x}(b,a) for numerical
    // stability when x > (a+1)/(a+b+2).
    let (x_cf, a_cf, b_cf, flip) = if x > (a + 1.0) / (a + b + 2.0) {
        (1.0 - x, b, a, true)
    } else {
        (x, a, b, false)
    };

    // log of the leading factor: x^a (1-x)^b / Beta(a,b)
    let lbeta = lgamma(a_cf) + lgamma(b_cf) - lgamma(a_cf + b_cf);
    let log_front = a_cf * x_cf.ln() + b_cf * (1.0 - x_cf).ln() - lbeta;

    // Evaluate the continued fraction by Lentz's method.
    let cf = betai_cf(a_cf, b_cf, x_cf);
    let val = (log_front + cf.ln()).exp() / a_cf;

    if flip {
        1.0 - val
    } else {
        val
    }
}

/// Modified Lentz continued-fraction evaluation for incomplete beta.
/// Returns the value of cf(a,b,x) such that I_x(a,b) = e^{log_front} * cf / a.
fn betai_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
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
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
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

        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
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

/// Student-t log-PDF with ν degrees of freedom: log f_ν(x).
fn student_t_log_pdf(x: f64, nu: f64) -> f64 {
    lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - ((nu + 1.0) / 2.0) * (1.0 + x * x / nu).ln()
}

/// Student-t CDF F_ν(x) = I_{ν/(ν+x²)}(ν/2, 1/2) / 2 for x ≤ 0.
pub fn student_t_cdf(x: f64, nu: f64) -> f64 {
    let z = nu / (nu + x * x);
    let ib = betai(nu / 2.0, 0.5, z) / 2.0;
    if x >= 0.0 {
        1.0 - ib
    } else {
        ib
    }
}

/// Inverse Student-t CDF F_ν⁻¹(p) via Newton-Raphson seeded from the normal
/// quantile. Convergence is typically 3–5 iterations.
pub fn student_t_cdf_inv(p: f64, nu: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Start from the normal approximation.
    let mut x = normal_cdf_inv(p);
    for _ in 0..50 {
        let fx = student_t_cdf(x, nu) - p;
        let fpx = (student_t_log_pdf(x, nu)).exp();
        if fpx < 1e-300 {
            break;
        }
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < 1e-12 * (1.0 + x.abs()) {
            break;
        }
    }
    x
}

// ── Scalar optimizer (Brent's method, minimisation) ───────────────────────────

/// Find the minimum of `f` on `[lo, hi]` to tolerance `tol` using Brent's
/// method. Returns `(x_min, f_min)`. Fails silently (returns midpoint) if
/// `f` is constant or non-finite everywhere.
fn brent_minimize<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, tol: f64) -> (f64, f64) {
    const GOLD: f64 = 0.381_966_011_250_105; // 1 − 1/φ
    const ZEPS: f64 = 1e-10;
    const ITMAX: usize = 500;

    let mut a = lo;
    let mut b = hi;
    let mut x = a + GOLD * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for _ in 0..ITMAX {
        let midpoint = 0.5 * (a + b);
        let tol1 = tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;

        if (x - midpoint).abs() <= tol2 - 0.5 * (b - a) {
            return (x, fx);
        }

        let mut use_parabola = false;
        let mut p_par = 0.0_f64;
        let mut q_par = 0.0_f64;

        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q2 = 2.0 * (q - r);
            let (p2, q3) = if q2 > 0.0 { (-p, q2) } else { (p, -q2) };
            if p2.abs() < (0.5 * q3 * e).abs() && p2 > q3 * (a - x) && p2 < q3 * (b - x) {
                p_par = p2 / q3;
                q_par = 0.0;
                if (x + p_par - a) < tol2 || (b - x - p_par) < tol2 {
                    p_par = if midpoint >= x { tol1 } else { -tol1 };
                }
                use_parabola = true;
            }
        }

        let u = if use_parabola {
            e = d;
            d = p_par + q_par; // q_par == 0 here, just p_par
            x + d
        } else {
            e = if x >= midpoint { a - x } else { b - x };
            d = GOLD * e;
            if (x + d - a).abs() >= tol1 {
                x + d
            } else {
                x + if d > 0.0 { tol1 } else { -tol1 }
            }
        };

        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || (w - x).abs() < ZEPS {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < ZEPS || (v - w).abs() < ZEPS {
                v = u;
                fv = fu;
            }
        }
    }
    (x, fx)
}

// ── BivariateCopula trait ─────────────────────────────────────────────────────

/// Bivariate copula: density, conditional CDF (h-function), its inverse, and MLE.
pub trait BivariateCopula: Send + Sync {
    /// Log copula density log c(u, v; params). Returns `f64::NEG_INFINITY` if
    /// (u, v) is outside (0, 1)².
    fn log_density(&self, u: f64, v: f64) -> f64;

    /// H-function h(u|v) = ∂C(u,v)/∂v — conditional CDF of U given V=v.
    ///
    /// Used in the sequential recursion for vine density evaluation.
    fn h(&self, u: f64, v: f64) -> f64;

    /// Inverse h-function: the unique u ∈ (0,1) such that h(u|v) = p.
    fn h_inv(&self, p: f64, v: f64) -> f64;
}

// ── Gaussian copula ───────────────────────────────────────────────────────────

/// Bivariate Gaussian copula with linear correlation ρ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianCopula {
    /// Correlation: ρ ∈ (−1, 1).
    pub rho: f64,
}

impl GaussianCopula {
    pub fn new(rho: f64) -> Self {
        assert!(rho.abs() < 1.0, "Gaussian copula requires |rho| < 1");
        Self { rho }
    }

    /// MLE of ρ given pseudo-observations (u, v). Returns `Err` if fewer than
    /// 2 pairs are provided.
    pub fn fit(u: &[f64], v: &[f64]) -> Result<Self, String> {
        if u.len() < 2 || u.len() != v.len() {
            return Err("Gaussian copula fit requires ≥ 2 matched pairs".into());
        }
        // Negative log-likelihood as a function of rho.
        let neg_ll = |rho: f64| {
            if rho.abs() >= 1.0 {
                return f64::INFINITY;
            }
            let c = GaussianCopula { rho };
            -u.iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| c.log_density(ui, vi))
                .sum::<f64>()
        };
        let (rho_hat, _) = brent_minimize(neg_ll, -0.9999, 0.9999, 1e-8);
        Ok(Self::new(rho_hat.clamp(-0.9999, 0.9999)))
    }
}

impl BivariateCopula for GaussianCopula {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let rho = self.rho;
        let r2 = 1.0 - rho * rho;
        let x = normal_cdf_inv(u);
        let y = normal_cdf_inv(v);
        -0.5 * (r2.ln() + (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / r2)
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        let x = normal_cdf_inv(u);
        let y = normal_cdf_inv(v);
        normal_cdf((x - self.rho * y) / (1.0 - self.rho * self.rho).sqrt())
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        let y = normal_cdf_inv(v);
        normal_cdf(normal_cdf_inv(p) * (1.0 - self.rho * self.rho).sqrt() + self.rho * y)
    }
}

// ── Student-t copula ──────────────────────────────────────────────────────────

/// Bivariate Student-t copula with correlation ρ and ν degrees of freedom.
///
/// MLE optimises ρ for fixed ν. The default ν=4 gives moderate tail heaviness
/// typical of PK η distributions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTCopula {
    /// Correlation: ρ ∈ (−1, 1).
    pub rho: f64,
    /// Degrees of freedom: ν > 2.
    pub nu: f64,
}

impl StudentTCopula {
    pub fn new(rho: f64, nu: f64) -> Self {
        assert!(rho.abs() < 1.0, "Student-t copula requires |rho| < 1");
        assert!(nu > 2.0, "Student-t copula requires nu > 2");
        Self { rho, nu }
    }

    /// MLE of ρ for fixed `nu`. Returns `Err` if fewer than 2 pairs provided.
    pub fn fit_fixed_nu(u: &[f64], v: &[f64], nu: f64) -> Result<Self, String> {
        if u.len() < 2 || u.len() != v.len() {
            return Err("Student-t copula fit requires ≥ 2 matched pairs".into());
        }
        let neg_ll = |rho: f64| {
            if rho.abs() >= 1.0 {
                return f64::INFINITY;
            }
            let c = StudentTCopula { rho, nu };
            -u.iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| c.log_density(ui, vi))
                .sum::<f64>()
        };
        let (rho_hat, _) = brent_minimize(neg_ll, -0.9999, 0.9999, 1e-8);
        Ok(Self::new(rho_hat.clamp(-0.9999, 0.9999), nu))
    }

    /// MLE of both ρ and ν via coordinate descent (alternating 1-D Brent).
    /// Starts from ν=4; converges in 5–10 rounds for typical PK data.
    pub fn fit(u: &[f64], v: &[f64]) -> Result<Self, String> {
        if u.len() < 3 || u.len() != v.len() {
            return Err("Student-t copula joint fit requires ≥ 3 matched pairs".into());
        }
        let mut rho = 0.0_f64;
        let mut nu = 4.0_f64;

        for _ in 0..20 {
            // Optimise ρ with fixed ν.
            let nu_c = nu;
            let neg_ll_rho = |r: f64| {
                if r.abs() >= 1.0 {
                    return f64::INFINITY;
                }
                let c = StudentTCopula { rho: r, nu: nu_c };
                -u.iter()
                    .zip(v.iter())
                    .map(|(&ui, &vi)| c.log_density(ui, vi))
                    .sum::<f64>()
            };
            let (new_rho, _) = brent_minimize(neg_ll_rho, -0.9999, 0.9999, 1e-8);
            rho = new_rho.clamp(-0.9999, 0.9999);

            // Optimise ν with fixed ρ.
            let rho_c = rho;
            let neg_ll_nu = |n: f64| {
                if n <= 2.0 {
                    return f64::INFINITY;
                }
                let c = StudentTCopula { rho: rho_c, nu: n };
                -u.iter()
                    .zip(v.iter())
                    .map(|(&ui, &vi)| c.log_density(ui, vi))
                    .sum::<f64>()
            };
            let (new_nu, _) = brent_minimize(neg_ll_nu, 2.001, 100.0, 1e-6);
            nu = new_nu.max(2.001);
        }
        Ok(Self::new(rho, nu))
    }
}

impl BivariateCopula for StudentTCopula {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let nu = self.nu;
        let rho = self.rho;
        let x = student_t_cdf_inv(u, nu);
        let y = student_t_cdf_inv(v, nu);
        let r2 = 1.0 - rho * rho;

        // log c = log f_{bivariate-t}(x,y;ρ,ν) − log f_ν(x) − log f_ν(y)
        let log_bivariate = lgamma((nu + 2.0) / 2.0) + lgamma(nu / 2.0)
            - 2.0 * lgamma((nu + 1.0) / 2.0)
            - 0.5 * r2.ln()
            - ((nu + 2.0) / 2.0) * (1.0 + (x * x + y * y - 2.0 * rho * x * y) / (nu * r2)).ln()
            + ((nu + 1.0) / 2.0) * (1.0 + x * x / nu).ln()
            + ((nu + 1.0) / 2.0) * (1.0 + y * y / nu).ln();

        log_bivariate
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        let nu = self.nu;
        let rho = self.rho;
        let x = student_t_cdf_inv(u, nu);
        let y = student_t_cdf_inv(v, nu);
        let r2 = 1.0 - rho * rho;
        // h(u|v; ρ, ν) = T_{ν+1}( (x − ρy) / √((ν + y²)(1−ρ²)/(ν+1)) )
        let scale = ((nu + y * y) * r2 / (nu + 1.0)).sqrt();
        student_t_cdf((x - rho * y) / scale, nu + 1.0)
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        let nu = self.nu;
        let rho = self.rho;
        let y = student_t_cdf_inv(v, nu);
        let r2 = 1.0 - rho * rho;
        let scale = ((nu + y * y) * r2 / (nu + 1.0)).sqrt();
        let t_val = student_t_cdf_inv(p, nu + 1.0);
        student_t_cdf(t_val * scale + rho * y, nu)
    }
}

// ── Clayton copula ────────────────────────────────────────────────────────────

/// Bivariate Clayton copula with lower-tail dependence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClaytonCopula {
    /// Shape parameter θ > 0. Higher values → stronger lower-tail dependence.
    pub theta: f64,
}

impl ClaytonCopula {
    pub fn new(theta: f64) -> Self {
        assert!(theta > 0.0, "Clayton copula requires theta > 0");
        Self { theta }
    }

    pub fn fit(u: &[f64], v: &[f64]) -> Result<Self, String> {
        if u.len() < 2 || u.len() != v.len() {
            return Err("Clayton copula fit requires ≥ 2 matched pairs".into());
        }
        let neg_ll = |theta: f64| {
            if theta <= 0.0 {
                return f64::INFINITY;
            }
            let c = ClaytonCopula { theta };
            -u.iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| c.log_density(ui, vi))
                .sum::<f64>()
        };
        let (theta_hat, _) = brent_minimize(neg_ll, 1e-6, 20.0, 1e-8);
        Ok(Self::new(theta_hat.max(1e-6)))
    }
}

impl BivariateCopula for ClaytonCopula {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let t = self.theta;
        let sum = u.powf(-t) + v.powf(-t) - 1.0;
        if sum <= 0.0 {
            return f64::NEG_INFINITY;
        }
        (1.0 + t).ln() - (t + 1.0) * (u.ln() + v.ln()) - (2.0 + 1.0 / t) * sum.ln()
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        let t = self.theta;
        let sum = u.powf(-t) + v.powf(-t) - 1.0;
        if sum <= 0.0 {
            return 0.5;
        }
        v.powf(-(t + 1.0)) * sum.powf(-(t + 1.0) / t)
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        let t = self.theta;
        // h_inv: solve h(u|v) = p for u
        // p = v^{-(t+1)} (u^{-t} + v^{-t} - 1)^{-(t+1)/t}
        // Rearrange: (u^{-t} + v^{-t} - 1) = (p * v^{t+1})^{-t/(t+1)}
        // => u^{-t} = (p * v^{t+1})^{-t/(t+1)} - v^{-t} + 1
        let inner = (p * v.powf(t + 1.0)).powf(-t / (t + 1.0)) - v.powf(-t) + 1.0;
        if inner <= 0.0 {
            return 1e-10;
        }
        inner.powf(-1.0 / t)
    }
}

// ── Gumbel copula ─────────────────────────────────────────────────────────────

/// Bivariate Gumbel copula with upper-tail dependence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GumbelCopula {
    /// Shape parameter θ ≥ 1. θ=1 → independence; θ→∞ → comonotone.
    pub theta: f64,
}

impl GumbelCopula {
    pub fn new(theta: f64) -> Self {
        assert!(theta >= 1.0, "Gumbel copula requires theta >= 1");
        Self { theta }
    }

    pub fn fit(u: &[f64], v: &[f64]) -> Result<Self, String> {
        if u.len() < 2 || u.len() != v.len() {
            return Err("Gumbel copula fit requires ≥ 2 matched pairs".into());
        }
        let neg_ll = |theta: f64| {
            if theta < 1.0 {
                return f64::INFINITY;
            }
            let c = GumbelCopula { theta };
            -u.iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| c.log_density(ui, vi))
                .sum::<f64>()
        };
        let (theta_hat, _) = brent_minimize(neg_ll, 1.0, 20.0, 1e-8);
        Ok(Self::new(theta_hat.max(1.0)))
    }

    /// log C(u,v;θ) = −(a_u^θ + a_v^θ)^{1/θ} where a = −log u, b = −log v.
    fn log_cdf(&self, u: f64, v: f64) -> f64 {
        let t = self.theta;
        let a = -u.ln();
        let b = -v.ln();
        let s = (a.powf(t) + b.powf(t)).powf(1.0 / t);
        -s
    }
}

impl BivariateCopula for GumbelCopula {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let t = self.theta;
        let a = -u.ln(); // −log u
        let b = -v.ln(); // −log v
        let at = a.powf(t);
        let bt = b.powf(t);
        let s = at + bt; // (−ln u)^θ + (−ln v)^θ

        // log c = log_cdf + log_cdf_factor
        // Using the analytical formula:
        // c(u,v;θ) = C(u,v;θ) * (uv)^{-1} * s^{1/θ-2} *
        //            [s^{1/θ} + θ−1] * (at * bt)^{1−1/θ} / (at * bt)
        // i.e., log c = log C + (1/θ-2)*log(s) + log(s^{1/θ} + θ-1)
        //               - log(u) - log(v) + (1-1/θ)*(log(at)+log(bt))
        //               - log(at) - log(bt)
        // Simplify to a clean form:
        let log_c_val = self.log_cdf(u, v);
        let s_to_inv_t = s.powf(1.0 / t);

        log_c_val - u.ln() - v.ln()
            + (1.0 / t - 2.0) * s.ln()
            + (s_to_inv_t + t - 1.0).ln()
            + (1.0 - 1.0 / t) * (at.ln() + bt.ln())
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return 0.5;
        }
        let t = self.theta;
        let a = -u.ln();
        let b = -v.ln();
        let at = a.powf(t);
        let bt = b.powf(t);
        let s = at + bt;

        // h(u|v) = ∂C/∂v = C(u,v) * (1/v) * (-log v)^{θ-1} * s^{1/θ-1}
        let log_c = self.log_cdf(u, v);
        let log_h = log_c - v.ln() + (t - 1.0) * b.ln() + (1.0 / t - 1.0) * s.ln();
        log_h.exp().clamp(0.0, 1.0)
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        // No simple closed form; invert h via bisection on (0, 1).
        let (mut lo, mut hi) = (1e-12_f64, 1.0 - 1e-12);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.h(mid, v) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

// ── Frank copula ──────────────────────────────────────────────────────────────

/// Bivariate Frank copula — symmetric, with no tail dependence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrankCopula {
    /// Dependence parameter θ ≠ 0. θ > 0 → positive association; θ < 0 →
    /// negative. Kendall's τ = 1 − 4/θ (1 − D₁(θ)/θ) where D₁ is the
    /// Debye function.
    pub theta: f64,
}

impl FrankCopula {
    pub fn new(theta: f64) -> Self {
        assert!(theta != 0.0, "Frank copula requires theta ≠ 0");
        Self { theta }
    }

    pub fn fit(u: &[f64], v: &[f64]) -> Result<Self, String> {
        if u.len() < 2 || u.len() != v.len() {
            return Err("Frank copula fit requires ≥ 2 matched pairs".into());
        }
        // Avoid exact zero; search on positive then negative half.
        let neg_ll = |theta: f64| {
            if theta == 0.0 {
                return f64::INFINITY;
            }
            let c = FrankCopula { theta };
            -u.iter()
                .zip(v.iter())
                .map(|(&ui, &vi)| c.log_density(ui, vi))
                .sum::<f64>()
        };
        let (t_pos, ll_pos) = brent_minimize(&neg_ll, 1e-6, 40.0, 1e-8);
        let (t_neg, ll_neg) = brent_minimize(&neg_ll, -40.0, -1e-6, 1e-8);
        let theta_hat = if ll_pos <= ll_neg { t_pos } else { t_neg };
        let theta_hat = if theta_hat.abs() < 1e-6 {
            1e-6
        } else {
            theta_hat
        };
        Ok(Self::new(theta_hat))
    }
}

impl BivariateCopula for FrankCopula {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || u >= 1.0 || v <= 0.0 || v >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let t = self.theta;
        let a = (-t * u).exp() - 1.0;
        let b = (-t * v).exp() - 1.0;
        let d = (-t).exp() - 1.0;
        if d.abs() < 1e-15 || a.abs() < 1e-15 || b.abs() < 1e-15 {
            return f64::NEG_INFINITY;
        }
        // log c = log(−θ) + log(d) + log(−θu) + log(−θv)
        //       − 2 log(a*b + d)
        // where a = e^{-θu}−1, b = e^{-θv}−1, d = e^{-θ}−1
        // c(u,v;θ) = −θ d (e^{-θ(u+v)}) / (d + a*b)²
        let num = -t * d * (-t * (u + v)).exp();
        let den = d + a * b;
        if den.abs() < 1e-300 || num <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // den² is always positive; den itself can be negative.
        num.ln() - 2.0 * den.abs().ln()
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        let t = self.theta;
        let a = (-t * u).exp() - 1.0;
        let b = (-t * v).exp() - 1.0;
        let d = (-t).exp() - 1.0;
        if d.abs() < 1e-15 {
            return u;
        }
        // h(u|v) = ∂C/∂v = e^{-θv} * a / (d + a * b)
        let num = (-t * v).exp() * a;
        let den = d + a * b;
        if den.abs() < 1e-300 {
            return 0.5;
        }
        (num / den).clamp(0.0, 1.0)
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        let t = self.theta;
        let b = (-t * v).exp() - 1.0;
        let d = (-t).exp() - 1.0;
        if d.abs() < 1e-15 {
            return p;
        }
        // Invert h: p = e^{-θv} * a / (d + a*b)
        // => a = p*d / (e^{-θv} - p*b)
        // => e^{-θu} - 1 = a  => u = -log(1+a)/θ
        let ev = (-t * v).exp();
        let denom = ev - p * b;
        if denom.abs() < 1e-300 {
            return p;
        }
        let a = p * d / denom;
        let inner = 1.0 + a;
        if inner <= 0.0 {
            return 1e-12;
        }
        (-inner.ln() / t).clamp(1e-12, 1.0 - 1e-12)
    }
}

// ── CopulaFamily enum ─────────────────────────────────────────────────────────

/// Unified dispatch over all supported bivariate copula families.
#[derive(Debug, Clone, PartialEq)]
pub enum CopulaFamily {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Clayton(ClaytonCopula),
    Gumbel(GumbelCopula),
    Frank(FrankCopula),
}

impl CopulaFamily {
    /// AIC = −2 log L + 2 k, where k is the number of parameters.
    pub fn aic(&self, u: &[f64], v: &[f64]) -> f64 {
        let k = match self {
            CopulaFamily::StudentT(_) => 2.0,
            _ => 1.0,
        };
        let log_l: f64 = u
            .iter()
            .zip(v.iter())
            .map(|(&ui, &vi)| self.log_density(ui, vi))
            .sum();
        -2.0 * log_l + 2.0 * k
    }

    /// Select the best family for (u, v) by AIC. Starts from a fitted Gaussian
    /// and accepts a competing family only when its AIC is strictly lower.
    pub fn select(u: &[f64], v: &[f64]) -> Result<Self, String> {
        let candidates: [CopulaFamily; 5] = [
            CopulaFamily::Gaussian(GaussianCopula::fit(u, v)?),
            CopulaFamily::Clayton(ClaytonCopula::fit(u, v)?),
            CopulaFamily::Gumbel(GumbelCopula::fit(u, v)?),
            CopulaFamily::Frank(FrankCopula::fit(u, v)?),
            CopulaFamily::StudentT(StudentTCopula::fit(u, v)?),
        ];
        let best = candidates
            .into_iter()
            .min_by(|a, b| {
                a.aic(u, v)
                    .partial_cmp(&b.aic(u, v))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty candidate array");
        Ok(best)
    }
}

impl BivariateCopula for CopulaFamily {
    fn log_density(&self, u: f64, v: f64) -> f64 {
        match self {
            CopulaFamily::Gaussian(c) => c.log_density(u, v),
            CopulaFamily::StudentT(c) => c.log_density(u, v),
            CopulaFamily::Clayton(c) => c.log_density(u, v),
            CopulaFamily::Gumbel(c) => c.log_density(u, v),
            CopulaFamily::Frank(c) => c.log_density(u, v),
        }
    }

    fn h(&self, u: f64, v: f64) -> f64 {
        match self {
            CopulaFamily::Gaussian(c) => c.h(u, v),
            CopulaFamily::StudentT(c) => c.h(u, v),
            CopulaFamily::Clayton(c) => c.h(u, v),
            CopulaFamily::Gumbel(c) => c.h(u, v),
            CopulaFamily::Frank(c) => c.h(u, v),
        }
    }

    fn h_inv(&self, p: f64, v: f64) -> f64 {
        match self {
            CopulaFamily::Gaussian(c) => c.h_inv(p, v),
            CopulaFamily::StudentT(c) => c.h_inv(p, v),
            CopulaFamily::Clayton(c) => c.h_inv(p, v),
            CopulaFamily::Gumbel(c) => c.h_inv(p, v),
            CopulaFamily::Frank(c) => c.h_inv(p, v),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;

    // ── Special functions ────────────────────────────────────────────────────

    #[test]
    fn normal_cdf_inv_roundtrips() {
        // The erf approximation (A&S 7.1.26) has ~1.5e-7 max error in erf,
        // translating to ~1.5e-5 relative CDF error at the tails.
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = normal_cdf_inv(p);
            let p2 = normal_cdf(x);
            assert_relative_eq!(p2, p, epsilon = 1e-5);
        }
    }

    #[test]
    fn student_t_cdf_at_known_values() {
        // t_4(0) = 0.5 by symmetry.
        assert_relative_eq!(student_t_cdf(0.0, 4.0), 0.5, epsilon = 1e-9);
        // For large nu, t_cdf → normal_cdf; allow 2e-3 for betai rounding
        // at a=500 (large parameter regime of the continued fraction).
        let z = 1.96;
        assert_relative_eq!(student_t_cdf(z, 1000.0), normal_cdf(z), epsilon = 2e-3);
    }

    #[test]
    fn student_t_cdf_inv_roundtrips() {
        for &nu in &[3.0, 5.0, 10.0, 30.0] {
            for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                let x = student_t_cdf_inv(p, nu);
                let p2 = student_t_cdf(x, nu);
                assert_relative_eq!(p2, p, max_relative = 1e-6, epsilon = 1e-9);
            }
        }
    }

    // ── Helper: finite-difference log-density gradient check ────────────────

    fn fd_log_density_du(c: &impl BivariateCopula, u: f64, v: f64) -> f64 {
        let h = 1e-7;
        (c.log_density(u + h, v) - c.log_density(u - h, v)) / (2.0 * h)
    }

    // ── h-function is the inverse of h_inv ──────────────────────────────────

    fn check_h_inv_roundtrip(c: &impl BivariateCopula, label: &str) {
        let test_pairs: &[(f64, f64)] =
            &[(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.5)];
        for &(u, v) in test_pairs {
            let p = c.h(u, v);
            let u2 = c.h_inv(p, v);
            assert!(
                (u2 - u).abs() < 1e-6,
                "{label}: h_inv(h({u}, {v})) = {u2} ≠ {u} (p={p})"
            );
        }
    }

    // ── Gaussian copula ──────────────────────────────────────────────────────

    #[test]
    fn gaussian_copula_independence_at_rho_zero() {
        let c = GaussianCopula::new(0.0);
        // At rho=0 the copula density should be 1 (log = 0) everywhere.
        for &(u, v) in &[(0.3, 0.4), (0.5, 0.5), (0.8, 0.2)] {
            assert_relative_eq!(c.log_density(u, v), 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn gaussian_copula_h_inv_roundtrip() {
        check_h_inv_roundtrip(&GaussianCopula::new(0.5), "Gaussian rho=0.5");
        check_h_inv_roundtrip(&GaussianCopula::new(-0.3), "Gaussian rho=-0.3");
    }

    #[test]
    fn gaussian_copula_fit_recovers_rho() {
        // Simulate positively-correlated pseudo-observations, fit, check ρ.
        let rho_true = 0.6;
        let n = 500;
        let mut rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(1)
        };
        use rand_distr::StandardNormal;
        let mut u = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            let x = z1;
            let y = rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * z2;
            u.push(normal_cdf(x));
            v.push(normal_cdf(y));
        }
        let fitted = GaussianCopula::fit(&u, &v).unwrap();
        assert!(
            (fitted.rho - rho_true).abs() < 0.08,
            "Gaussian fit: expected ρ ≈ {rho_true}, got {:.4}",
            fitted.rho
        );
    }

    // ── Clayton copula ───────────────────────────────────────────────────────

    #[test]
    fn clayton_copula_h_inv_roundtrip() {
        check_h_inv_roundtrip(&ClaytonCopula::new(2.0), "Clayton theta=2");
        check_h_inv_roundtrip(&ClaytonCopula::new(0.5), "Clayton theta=0.5");
    }

    #[test]
    fn clayton_copula_density_positive() {
        let c = ClaytonCopula::new(1.5);
        for &(u, v) in &[(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)] {
            assert!(
                c.log_density(u, v).is_finite(),
                "Clayton log_density finite at ({u},{v})"
            );
        }
    }

    #[test]
    fn clayton_copula_fit_recovers_theta() {
        use rand::SeedableRng;
        let theta_true = 2.0_f64;
        // Sample from Clayton: inverse Rosenblatt (u uniform, v = h_inv(w|u))
        let c = ClaytonCopula::new(theta_true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let n = 500;
        let mut u_vec = Vec::with_capacity(n);
        let mut v_vec = Vec::with_capacity(n);
        use rand::Rng;
        for _ in 0..n {
            let u: f64 = rng.gen();
            let w: f64 = rng.gen();
            let v = c.h_inv(w, u);
            u_vec.push(u);
            v_vec.push(v);
        }
        let fitted = ClaytonCopula::fit(&u_vec, &v_vec).unwrap();
        assert!(
            (fitted.theta - theta_true).abs() < 0.4,
            "Clayton fit: expected θ ≈ {theta_true}, got {:.3}",
            fitted.theta
        );
    }

    // ── Gumbel copula ────────────────────────────────────────────────────────

    #[test]
    fn gumbel_theta1_is_independence() {
        // At theta=1, Gumbel degenerates to the independence copula: density = 1.
        let c = GumbelCopula::new(1.0);
        for &(u, v) in &[(0.3, 0.4), (0.5, 0.5), (0.8, 0.2)] {
            let ld = c.log_density(u, v);
            assert!(
                ld.abs() < 0.01,
                "Gumbel theta=1 log_density at ({u},{v}) = {ld} (expected ≈0)"
            );
        }
    }

    #[test]
    fn gumbel_copula_h_inv_roundtrip() {
        check_h_inv_roundtrip(&GumbelCopula::new(2.0), "Gumbel theta=2");
        check_h_inv_roundtrip(&GumbelCopula::new(1.5), "Gumbel theta=1.5");
    }

    // ── Frank copula ─────────────────────────────────────────────────────────

    #[test]
    fn frank_copula_h_inv_roundtrip() {
        check_h_inv_roundtrip(&FrankCopula::new(3.0), "Frank theta=3");
        check_h_inv_roundtrip(&FrankCopula::new(-2.0), "Frank theta=-2");
    }

    #[test]
    fn frank_copula_density_positive() {
        let c = FrankCopula::new(4.0);
        for &(u, v) in &[(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)] {
            assert!(
                c.log_density(u, v).is_finite(),
                "Frank log_density finite at ({u},{v})"
            );
        }
    }

    // ── Student-t copula ─────────────────────────────────────────────────────

    #[test]
    fn student_t_copula_h_inv_roundtrip() {
        check_h_inv_roundtrip(&StudentTCopula::new(0.5, 4.0), "StudentT rho=0.5 nu=4");
    }

    #[test]
    fn student_t_copula_large_nu_matches_gaussian() {
        // For large ν, the Student-t copula should be close to Gaussian.
        let g = GaussianCopula::new(0.5);
        let t = StudentTCopula::new(0.5, 200.0);
        for &(u, v) in &[(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)] {
            let diff = (g.log_density(u, v) - t.log_density(u, v)).abs();
            assert!(
                diff < 0.01,
                "Student-t (nu=200) should approximate Gaussian at ({u},{v}), diff={diff}"
            );
        }
    }

    // ── CopulaFamily::select ─────────────────────────────────────────────────

    #[test]
    fn copula_family_select_returns_valid_family() {
        // With independence pseudo-observations, select should return something
        // without panicking.
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let u: Vec<f64> = (0..100).map(|_| rng.gen::<f64>()).collect();
        let v: Vec<f64> = (0..100).map(|_| rng.gen::<f64>()).collect();
        CopulaFamily::select(&u, &v).expect("select must not error on valid pseudo-obs");
    }
}
