//! Bivariate pair-copula catalogue for the regular-vine SAEM estimator.
//!
//! Supported pair-copula families:
//! - [`CopulaFamily::Independence`]: `c(u, v) = 1` (`k = 0` free parameters).
//! - [`CopulaFamily::Gaussian`]: elliptical Gaussian copula with `rho in (-1, 1)` (`k = 1`).
//! - [`CopulaFamily::StudentT`]: elliptical Student-t copula with `rho in (-1, 1)` and `nu > 2.10` (`k = 2`).
//! - [`CopulaFamily::Frank`]: symmetric Archimedean copula with `theta != 0` (`k = 1`).
//! - [`CopulaFamily::Clayton`]: Archimedean Clayton copula (`theta > 0`) at `0`, `90`, `180`,
//!   or `270` degree counter-clockwise rotation (`k = 1`).
//! - [`CopulaFamily::Gumbel`]: extreme-value Gumbel copula (`theta >= 1`) at `0`, `90`, `180`,
//!   or `270` degree counter-clockwise rotation (`k = 1`).
//!
//! Because `90` and `270` degree rotations are asymmetric (`C(u, v) != C(v, u)`),
//! every copula exposes *both* directed conditional CDFs:
//! - `h1(u, v) = dC(u, v)/dv = P(U <= u | V = v)` and `h1_inv(p, v)`
//! - `h2(v, u) = dC(u, v)/du = P(V <= v | U = u)` and `h2_inv(p, u)`

use crate::stats::marginals::{
    invert_small, make_spd_small, probit_exact, solve_small, student_t_cdf, student_t_quantile,
};
use crate::stats::special::{ln_gamma, pnorm};
use crate::stats::util::log_sum_exp2;
use serde::{Deserialize, Serialize};

const U_EPS: f64 = 1e-12;
const MAX_RHO: f64 = 0.9995;
const CLAYTON_MIN: f64 = 1e-4;
const CLAYTON_MAX: f64 = 28.0;
const GUMBEL_MIN: f64 = 1.0 + 1e-5;
const GUMBEL_MAX: f64 = 28.0;
const FRANK_MAX: f64 = 40.0;
const T_NU_SHIFT: f64 = 2.10;
const T_NU_MAX: f64 = 100.0;

/// Counter-clockwise rotation on `[0, 1]^2` for asymmetric Archimedean pair-copulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CopulaRotation {
    /// Unrotated (`0` degrees): positive dependence (`tau >= 0`).
    #[default]
    R0,
    /// `90` degrees (`(u, v) -> (1 - u, v)`): negative dependence (`tau <= 0`).
    R90,
    /// `180` degrees / survival copula (`(u, v) -> (1 - u, 1 - v)`): positive dependence, flipped tail.
    R180,
    /// `270` degrees (`(u, v) -> (u, 1 - v)`): negative dependence (`tau <= 0`).
    R270,
}

impl CopulaRotation {
    /// Degrees in `{0, 90, 180, 270}` for reporting and serialization.
    pub fn degrees(self) -> u16 {
        match self {
            Self::R0 => 0,
            Self::R90 => 90,
            Self::R180 => 180,
            Self::R270 => 270,
        }
    }

    /// Construct from integer degrees (`0`, `90`, `180`, `270`).
    pub fn from_degrees(deg: u16) -> Self {
        match deg {
            90 => Self::R90,
            180 => Self::R180,
            270 => Self::R270,
            _ => Self::R0,
        }
    }

    #[inline]
    fn map_uv(self, u: f64, v: f64) -> (f64, f64) {
        match self {
            Self::R0 => (u, v),
            Self::R90 => (1.0 - u, v),
            Self::R180 => (1.0 - u, 1.0 - v),
            Self::R270 => (u, 1.0 - v),
        }
    }

    #[inline]
    fn apply_h1<F: Fn(f64, f64) -> f64>(self, base_h: F, u: f64, v: f64) -> f64 {
        match self {
            Self::R0 => base_h(u, v),
            Self::R90 => (1.0 - base_h(1.0 - u, v)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R180 => (1.0 - base_h(1.0 - u, 1.0 - v)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R270 => base_h(u, 1.0 - v).clamp(U_EPS, 1.0 - U_EPS),
        }
    }

    #[inline]
    fn apply_h2<F: Fn(f64, f64) -> f64>(self, base_h: F, v: f64, u: f64) -> f64 {
        match self {
            Self::R0 => base_h(v, u),
            Self::R90 => base_h(v, 1.0 - u).clamp(U_EPS, 1.0 - U_EPS),
            Self::R180 => (1.0 - base_h(1.0 - v, 1.0 - u)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R270 => (1.0 - base_h(1.0 - v, u)).clamp(U_EPS, 1.0 - U_EPS),
        }
    }

    #[inline]
    fn apply_h1_inv<F: Fn(f64, f64) -> f64>(self, base_h_inv: F, p: f64, v: f64) -> f64 {
        match self {
            Self::R0 => base_h_inv(p, v),
            Self::R90 => (1.0 - base_h_inv(1.0 - p, v)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R180 => (1.0 - base_h_inv(1.0 - p, 1.0 - v)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R270 => base_h_inv(p, 1.0 - v).clamp(U_EPS, 1.0 - U_EPS),
        }
    }

    #[inline]
    fn apply_h2_inv<F: Fn(f64, f64) -> f64>(self, base_h_inv: F, p: f64, u: f64) -> f64 {
        match self {
            Self::R0 => base_h_inv(p, u),
            Self::R90 => base_h_inv(p, 1.0 - u).clamp(U_EPS, 1.0 - U_EPS),
            Self::R180 => (1.0 - base_h_inv(1.0 - p, 1.0 - u)).clamp(U_EPS, 1.0 - U_EPS),
            Self::R270 => (1.0 - base_h_inv(1.0 - p, u)).clamp(U_EPS, 1.0 - U_EPS),
        }
    }
}

/// Bounded 1-D Brent minimizer on `[lo, hi]`.
fn brent_minimize<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, tol: f64) -> (f64, f64) {
    const GOLD: f64 = 0.381_966_011_250_105;
    const ZEPS: f64 = 1e-11;
    const ITMAX: usize = 250;

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
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return (x, fx);
        }
        let mut use_parabola = false;
        let mut p_step = 0.0_f64;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q2 = 2.0 * (q - r);
            let (p2, q3) = if q2 > 0.0 { (-p, q2) } else { (p, -q2) };
            if p2.abs() < (0.5 * q3 * e).abs() && p2 > q3 * (a - x) && p2 < q3 * (b - x) {
                p_step = p2 / q3;
                if (x + p_step - a) < tol2 || (b - x - p_step) < tol2 {
                    p_step = if xm >= x { tol1 } else { -tol1 };
                }
                use_parabola = true;
            }
        }
        let u = if use_parabola {
            e = d;
            d = p_step;
            x + d
        } else {
            e = if x >= xm { a - x } else { b - x };
            d = GOLD * e;
            if d.abs() >= tol1 {
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

// ── Unrotated base Archimedean formulas (log-space overflow-safe) ─────────────

#[inline]
fn clayton_base_log_density(a: f64, b: f64, theta: f64) -> f64 {
    let la = a.clamp(U_EPS, 1.0 - U_EPS).ln();
    let lb = b.clamp(U_EPS, 1.0 - U_EPS).ln();
    // log(a^-t + b^-t - 1) = log(exp(-t*la) + exp(-t*lb) - 1).
    let x = -theta * la; // > 0
    let y = -theta * lb; // > 0
    let log_ab = log_sum_exp2(x, y);
    let log_sum = if log_ab > 35.0 { log_ab } else { (log_ab.exp() - 1.0).max(1e-300).ln() };
    theta.ln_1p() - (theta + 1.0) * (la + lb) - (2.0 + 1.0 / theta) * log_sum
}

#[inline]
fn clayton_base_h(a: f64, b: f64, theta: f64) -> f64 {
    let la = a.clamp(U_EPS, 1.0 - U_EPS).ln();
    let lb = b.clamp(U_EPS, 1.0 - U_EPS).ln();
    let log_ab = log_sum_exp2(-theta * la, -theta * lb);
    let log_sum = if log_ab > 35.0 { log_ab } else { (log_ab.exp() - 1.0).max(1e-300).ln() };
    let log_h = -(theta + 1.0) * lb - ((theta + 1.0) / theta) * log_sum;
    log_h.exp().clamp(U_EPS, 1.0 - U_EPS)
}

#[inline]
fn clayton_base_h_inv(p: f64, b: f64, theta: f64) -> f64 {
    let lp = p.clamp(U_EPS, 1.0 - U_EPS).ln();
    let lb = b.clamp(U_EPS, 1.0 - U_EPS).ln();
    // a^-t = exp(-t/(t+1) * (lp + (t+1)*lb)) - exp(-t*lb) + 1
    //      = exp(-t*lb) * (exp(-t/(t+1) * lp) - 1) + 1.
    // Notice p in (0,1) => lp < 0 => -t/(t+1)*lp > 0 => exp(...) - 1 > 0 ALWAYS!
    // Writing it in this factored form makes negativity mathematically impossible!
    let pos_term = (-theta / (theta + 1.0) * lp).exp_m1().max(0.0);
    let log_factor = -theta * lb + pos_term.max(1e-300).ln();
    let log_a_neg_t = if pos_term == 0.0 {
        0.0
    } else if log_factor > 35.0 {
        log_factor
    } else {
        log_factor.exp().ln_1p()
    };
    (-log_a_neg_t / theta).exp().clamp(U_EPS, 1.0 - U_EPS)
}

#[inline]
fn gumbel_base_log_density(a: f64, b: f64, theta: f64) -> f64 {
    let a_c = a.clamp(U_EPS, 1.0 - U_EPS);
    let b_c = b.clamp(U_EPS, 1.0 - U_EPS);
    let na = (-a_c.ln()).max(1e-300);
    let nb = (-b_c.ln()).max(1e-300);
    let lna = na.ln();
    let lnb = nb.ln();
    let log_s = log_sum_exp2(theta * lna, theta * lnb);
    let s_inv_t = (log_s / theta).exp();
    -s_inv_t - a_c.ln() - b_c.ln()
        + (1.0 / theta - 2.0) * log_s
        + (s_inv_t + theta - 1.0).ln()
        + (theta - 1.0) * (lna + lnb)
}

#[inline]
fn gumbel_base_h(a: f64, b: f64, theta: f64) -> f64 {
    let a_c = a.clamp(U_EPS, 1.0 - U_EPS);
    let b_c = b.clamp(U_EPS, 1.0 - U_EPS);
    let na = (-a_c.ln()).max(1e-300);
    let nb = (-b_c.ln()).max(1e-300);
    let log_s = log_sum_exp2(theta * na.ln(), theta * nb.ln());
    let s_inv_t = (log_s / theta).exp();
    let log_h = -s_inv_t - b_c.ln() + (theta - 1.0) * nb.ln() + (1.0 / theta - 1.0) * log_s;
    log_h.exp().clamp(U_EPS, 1.0 - U_EPS)
}

#[inline]
fn gumbel_base_h_inv(p: f64, b: f64, theta: f64) -> f64 {
    let p_c = p.clamp(U_EPS, 1.0 - U_EPS);
    let b_c = b.clamp(U_EPS, 1.0 - U_EPS);
    if (theta - 1.0).abs() < 1e-9 {
        return p_c;
    }
    // Solve g(z) = z + (theta - 1) ln z - [nb + (theta - 1) ln nb - ln p] = 0
    // for z = s^{1/theta} > nb, where nb = -ln b > 0 (Joe 2014 §4.10).
    let nb = (-b_c.ln()).max(1e-300);
    let tm1 = theta - 1.0;
    let g_nb = nb + tm1 * nb.ln();
    let rhs = g_nb - p_c.ln(); // strictly > g(nb) since ln p < 0, ensuring unique root z > nb.
    let mut z = (nb - p_c.ln()).max(nb + 1e-15);
    if tm1 > 0.0 && rhs > 1.5 {
        let cand = rhs - tm1 * rhs.max(1e-300).ln();
        if cand > nb {
            z = cand;
        }
    }
    for _ in 0..20 {
        let f = z + tm1 * z.ln() - rhs;
        if f.abs() <= 1e-14 * (1.0 + rhs) {
            break;
        }
        let fp = 1.0 + tm1 / z;
        let fpp = -tm1 / (z * z);
        let step = f / (fp - 0.5 * f * fpp / fp);
        let next = (z - step).max(nb);
        if (next - z).abs() <= 1e-14 * (1.0 + z) {
            z = next;
            break;
        }
        z = next;
    }
    // z^theta = na^theta + nb^theta => na = (z^theta - nb^theta)^{1/theta}
    //        = z * (1 - (nb / z)^theta)^{1/theta}.
    let ratio = (nb / z).clamp(0.0, 1.0);
    let one_minus = if ratio >= 1.0 { 0.0 } else { (1.0 - ratio.powf(theta)).max(0.0) };
    let na = z * one_minus.powf(1.0 / theta);
    (-na).exp().clamp(U_EPS, 1.0 - U_EPS)
}

#[inline]
fn frank_log_density(u: f64, v: f64, theta: f64) -> f64 {
    let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
    let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
    if theta.abs() < 1e-7 {
        return 0.0;
    }
    // Symmetry: c(u, v; -theta) == c(1 - u, v; +theta). Evaluate with t = |theta| > 0!
    let (uu, vv, t) = if theta > 0.0 { (u_c, v_c, theta) } else { (1.0 - u_c, v_c, -theta) };
    // For t > 0: d = 1 - e^-t = -expm1(-t) in (0, 1);
    // a = 1 - e^{-t*uu}, b = 1 - e^{-t*vv} in (0, 1);
    // denom = d - a*b = e^{-t*uu} + e^{-t*vv} - e^{-t} - e^{-t*(uu+vv)} > 0.
    // Wait: in original variables (-theta*u):
    // c(uu, vv; t) = t * (1 - e^-t) * e^{-t*(uu+vv)} / [ (1 - e^-t) - (1 - e^{-t*uu})(1 - e^{-t*vv}) ]^2!
    let d = -(-t).exp_m1();
    let a = -(-t * uu).exp_m1();
    let b = -(-t * vv).exp_m1();
    let denom = (d - a * b).max(1e-300);
    t.ln() + d.ln() - t * (uu + vv) - 2.0 * denom.ln()
}

#[inline]
fn frank_h(u: f64, v: f64, theta: f64) -> f64 {
    let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
    let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
    if theta.abs() < 1e-7 {
        return u_c;
    }
    if theta < 0.0 {
        return (1.0 - frank_h(1.0 - u_c, v_c, -theta)).clamp(U_EPS, 1.0 - U_EPS);
    }
    let t = theta;
    let d = -(-t).exp_m1();
    let a = -(-t * u_c).exp_m1();
    let b = -(-t * v_c).exp_m1();
    let num = (-t * v_c).exp() * a;
    let den = (d - a * b).max(1e-300);
    (num / den).clamp(U_EPS, 1.0 - U_EPS)
}

#[inline]
fn frank_h_inv(p: f64, v: f64, theta: f64) -> f64 {
    let p_c = p.clamp(U_EPS, 1.0 - U_EPS);
    let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
    if theta.abs() < 1e-7 {
        return p_c;
    }
    if theta < 0.0 {
        return (1.0 - frank_h_inv(1.0 - p_c, v_c, -theta)).clamp(U_EPS, 1.0 - U_EPS);
    }
    let t = theta;
    // Solve p = e^{-t*v} * a / (d - a*b) for a = 1 - e^{-t*u}:
    // p*d - p*a*b = e^{-t*v}*a => a = p*d / (e^{-t*v} + p*b).
    let d = -(-t).exp_m1();
    let b = -(-t * v_c).exp_m1();
    let ev = (-t * v_c).exp();
    let a = (p_c * d / (ev + p_c * b)).clamp(0.0, 1.0 - 1e-16);
    let u = -(-a).ln_1p() / t;
    u.clamp(U_EPS, 1.0 - U_EPS)
}

/// Unified bivariate pair-copula family used across all R-vine trees.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CopulaFamily {
    /// Independence copula `C(u, v) = u * v` (`0` free parameters).
    Independence,
    /// Elliptical Gaussian copula (`rho in (-1, 1)`).
    Gaussian {
        /// Linear correlation `rho`.
        rho: f64,
    },
    /// Elliptical Student-t copula (`rho in (-1, 1)`, `nu > 2.10`).
    StudentT {
        /// Linear correlation `rho`.
        rho: f64,
        /// Degrees of freedom `nu`.
        nu: f64,
    },
    /// Symmetric Frank copula (`theta != 0`).
    Frank {
        /// Dependence parameter `theta`.
        theta: f64,
    },
    /// Rotated Clayton copula (`theta > 0`, `rotation in {0, 90, 180, 270}`).
    Clayton {
        /// Shape parameter `theta > 0`.
        theta: f64,
        /// Counter-clockwise rotation.
        rotation: CopulaRotation,
    },
    /// Rotated Gumbel copula (`theta >= 1`, `rotation in {0, 90, 180, 270}`).
    Gumbel {
        /// Shape parameter `theta >= 1`.
        theta: f64,
        /// Counter-clockwise rotation.
        rotation: CopulaRotation,
    },
}

impl CopulaFamily {
    /// Construct a clamped Gaussian copula.
    pub fn gaussian(rho: f64) -> Self {
        Self::Gaussian { rho: rho.clamp(-MAX_RHO, MAX_RHO) }
    }

    /// Construct a clamped Student-t copula.
    pub fn student_t(rho: f64, nu: f64) -> Self {
        Self::StudentT {
            rho: rho.clamp(-MAX_RHO, MAX_RHO),
            nu: nu.clamp(T_NU_SHIFT + 0.05, T_NU_MAX),
        }
    }

    /// Construct a clamped Frank copula.
    pub fn frank(theta: f64) -> Self {
        let t = theta.clamp(-FRANK_MAX, FRANK_MAX);
        Self::Frank {
            theta: if t.abs() < 1e-5 {
                if t >= 0.0 {
                    1e-5
                } else {
                    -1e-5
                }
            } else {
                t
            },
        }
    }

    /// Construct a clamped rotated Clayton copula.
    pub fn clayton(theta: f64, rotation: CopulaRotation) -> Self {
        Self::Clayton { theta: theta.clamp(CLAYTON_MIN, CLAYTON_MAX), rotation }
    }

    /// Construct a clamped rotated Gumbel copula.
    pub fn gumbel(theta: f64, rotation: CopulaRotation) -> Self {
        Self::Gumbel { theta: theta.clamp(GUMBEL_MIN, GUMBEL_MAX), rotation }
    }

    /// Canonical family label (`"independence"`, `"gaussian"`, `"student_t"`, `"frank"`,
    /// `"clayton"`, `"clayton_90"`, `"clayton_180"`, `"clayton_270"`, `"gumbel"`, etc.).
    pub fn family_name(&self) -> &'static str {
        match *self {
            Self::Independence => "independence",
            Self::Gaussian { .. } => "gaussian",
            Self::StudentT { .. } => "student_t",
            Self::Frank { .. } => "frank",
            Self::Clayton { rotation, .. } => match rotation {
                CopulaRotation::R0 => "clayton",
                CopulaRotation::R90 => "clayton_90",
                CopulaRotation::R180 => "clayton_180",
                CopulaRotation::R270 => "clayton_270",
            },
            Self::Gumbel { rotation, .. } => match rotation {
                CopulaRotation::R0 => "gumbel",
                CopulaRotation::R90 => "gumbel_90",
                CopulaRotation::R180 => "gumbel_180",
                CopulaRotation::R270 => "gumbel_270",
            },
        }
    }

    /// Base family name without rotation suffix.
    pub fn base_family_name(&self) -> &'static str {
        match self {
            Self::Independence => "independence",
            Self::Gaussian { .. } => "gaussian",
            Self::StudentT { .. } => "student_t",
            Self::Frank { .. } => "frank",
            Self::Clayton { .. } => "clayton",
            Self::Gumbel { .. } => "gumbel",
        }
    }

    /// Rotation degrees (`0` for symmetric/elliptical families).
    pub fn rotation_degrees(&self) -> u16 {
        match *self {
            Self::Clayton { rotation, .. } | Self::Gumbel { rotation, .. } => rotation.degrees(),
            _ => 0,
        }
    }

    /// Number of free estimated scalar parameters (`0` for `Independence`, `2` for `StudentT`, `1` otherwise).
    pub fn n_params(&self) -> usize {
        match self {
            Self::Independence => 0,
            Self::StudentT { .. } => 2,
            _ => 1,
        }
    }

    /// Log copula density `ln c(u, v)` on `(0, 1)^2`.
    pub fn log_density(&self, u: f64, v: f64) -> f64 {
        if !(u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0) {
            return f64::NEG_INFINITY;
        }
        let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
        let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
        match *self {
            Self::Independence => 0.0,
            Self::Gaussian { rho } => {
                let r2 = 1.0 - rho * rho;
                let x = probit_exact(u_c);
                let y = probit_exact(v_c);
                -0.5 * (r2.ln() + (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / r2)
            }
            Self::StudentT { rho, nu } => {
                let r2 = 1.0 - rho * rho;
                let x = student_t_quantile(u_c, nu);
                let y = student_t_quantile(v_c, nu);
                let quad = (x * x + y * y - 2.0 * rho * x * y) / (nu * r2);
                ln_gamma(0.5 * (nu + 2.0)) + ln_gamma(0.5 * nu)
                    - 2.0 * ln_gamma(0.5 * (nu + 1.0))
                    - 0.5 * r2.ln()
                    - 0.5 * (nu + 2.0) * quad.ln_1p()
                    + 0.5 * (nu + 1.0) * ((x * x / nu).ln_1p() + (y * y / nu).ln_1p())
            }
            Self::Frank { theta } => frank_log_density(u_c, v_c, theta),
            Self::Clayton { theta, rotation } => {
                let (a, b) = rotation.map_uv(u_c, v_c);
                clayton_base_log_density(a, b, theta)
            }
            Self::Gumbel { theta, rotation } => {
                let (a, b) = rotation.map_uv(u_c, v_c);
                gumbel_base_log_density(a, b, theta)
            }
        }
    }

    /// First directed conditional CDF `h1(u | v) = dC(u, v)/dv = P(U <= u | V = v)`.
    pub fn h1(&self, u: f64, v: f64) -> f64 {
        let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
        let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
        match *self {
            Self::Independence => u_c,
            Self::Gaussian { rho } => {
                let x = probit_exact(u_c);
                let y = probit_exact(v_c);
                pnorm((x - rho * y) / (1.0 - rho * rho).sqrt()).clamp(U_EPS, 1.0 - U_EPS)
            }
            Self::StudentT { rho, nu } => {
                let x = student_t_quantile(u_c, nu);
                let y = student_t_quantile(v_c, nu);
                let scale = ((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0)).sqrt();
                student_t_cdf((x - rho * y) / scale, nu + 1.0).clamp(U_EPS, 1.0 - U_EPS)
            }
            Self::Frank { theta } => frank_h(u_c, v_c, theta),
            Self::Clayton { theta, rotation } => {
                rotation.apply_h1(|a, b| clayton_base_h(a, b, theta), u_c, v_c)
            }
            Self::Gumbel { theta, rotation } => {
                rotation.apply_h1(|a, b| gumbel_base_h(a, b, theta), u_c, v_c)
            }
        }
    }

    /// Second directed conditional CDF `h2(v | u) = dC(u, v)/du = P(V <= v | U = u)`.
    pub fn h2(&self, v: f64, u: f64) -> f64 {
        let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
        let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
        match *self {
            Self::Independence
            | Self::Gaussian { .. }
            | Self::StudentT { .. }
            | Self::Frank { .. } => self.h1(v_c, u_c),
            Self::Clayton { theta, rotation } => {
                rotation.apply_h2(|b, a| clayton_base_h(b, a, theta), v_c, u_c)
            }
            Self::Gumbel { theta, rotation } => {
                rotation.apply_h2(|b, a| gumbel_base_h(b, a, theta), v_c, u_c)
            }
        }
    }

    /// Inverse of `h1(., v)`: unique `u in (0, 1)` such that `h1(u, v) == p`.
    pub fn h1_inv(&self, p: f64, v: f64) -> f64 {
        let p_c = p.clamp(U_EPS, 1.0 - U_EPS);
        let v_c = v.clamp(U_EPS, 1.0 - U_EPS);
        match *self {
            Self::Independence => p_c,
            Self::Gaussian { rho } => {
                let y = probit_exact(v_c);
                pnorm(probit_exact(p_c) * (1.0 - rho * rho).sqrt() + rho * y)
                    .clamp(U_EPS, 1.0 - U_EPS)
            }
            Self::StudentT { rho, nu } => {
                let y = student_t_quantile(v_c, nu);
                let scale = ((nu + y * y) * (1.0 - rho * rho) / (nu + 1.0)).sqrt();
                let t_val = student_t_quantile(p_c, nu + 1.0);
                student_t_cdf(t_val * scale + rho * y, nu).clamp(U_EPS, 1.0 - U_EPS)
            }
            Self::Frank { theta } => frank_h_inv(p_c, v_c, theta),
            Self::Clayton { theta, rotation } => {
                rotation.apply_h1_inv(|q, b| clayton_base_h_inv(q, b, theta), p_c, v_c)
            }
            Self::Gumbel { theta, rotation } => {
                rotation.apply_h1_inv(|q, b| gumbel_base_h_inv(q, b, theta), p_c, v_c)
            }
        }
    }

    /// Inverse of `h2(., u)`: unique `v in (0, 1)` such that `h2(v, u) == p`.
    pub fn h2_inv(&self, p: f64, u: f64) -> f64 {
        let p_c = p.clamp(U_EPS, 1.0 - U_EPS);
        let u_c = u.clamp(U_EPS, 1.0 - U_EPS);
        match *self {
            Self::Independence
            | Self::Gaussian { .. }
            | Self::StudentT { .. }
            | Self::Frank { .. } => self.h1_inv(p_c, u_c),
            Self::Clayton { theta, rotation } => {
                rotation.apply_h2_inv(|q, a| clayton_base_h_inv(q, a, theta), p_c, u_c)
            }
            Self::Gumbel { theta, rotation } => {
                rotation.apply_h2_inv(|q, a| gumbel_base_h_inv(q, a, theta), p_c, u_c)
            }
        }
    }

    /// Analytical Kendall's rank correlation `tau in [-1, 1]`.
    pub fn kendall_tau(&self) -> f64 {
        match *self {
            Self::Independence => 0.0,
            Self::Gaussian { rho } | Self::StudentT { rho, .. } => {
                (2.0 * std::f64::consts::FRAC_1_PI) * rho.asin()
            }
            Self::Frank { theta } => kendall_tau_frank(theta),
            Self::Clayton { theta, rotation } => {
                let t0 = theta / (theta + 2.0);
                match rotation {
                    CopulaRotation::R0 | CopulaRotation::R180 => t0,
                    CopulaRotation::R90 | CopulaRotation::R270 => -t0,
                }
            }
            Self::Gumbel { theta, rotation } => {
                let t0 = 1.0 - 1.0 / theta;
                match rotation {
                    CopulaRotation::R0 | CopulaRotation::R180 => t0,
                    CopulaRotation::R90 | CopulaRotation::R270 => -t0,
                }
            }
        }
    }

    /// Lower and upper diagonal tail-dependence coefficients `(lambda_L, lambda_U)`.
    pub fn tail_dependence(&self) -> (f64, f64) {
        match *self {
            Self::Independence | Self::Gaussian { .. } | Self::Frank { .. } => (0.0, 0.0),
            Self::StudentT { rho, nu } => {
                if rho <= -1.0 + 1e-8 {
                    (0.0, 0.0)
                } else {
                    let arg = -((nu + 1.0) * (1.0 - rho) / (1.0 + rho)).sqrt();
                    let td = 2.0 * student_t_cdf(arg, nu + 1.0);
                    (td, td)
                }
            }
            Self::Clayton { theta, rotation } => {
                let td = 2.0_f64.powf(-1.0 / theta);
                match rotation {
                    CopulaRotation::R0 => (td, 0.0),
                    CopulaRotation::R180 => (0.0, td),
                    CopulaRotation::R90 | CopulaRotation::R270 => (0.0, 0.0),
                }
            }
            Self::Gumbel { theta, rotation } => {
                let td = (2.0 - 2.0_f64.powf(1.0 / theta)).clamp(0.0, 1.0);
                match rotation {
                    CopulaRotation::R0 => (0.0, td),
                    CopulaRotation::R180 => (td, 0.0),
                    CopulaRotation::R90 | CopulaRotation::R270 => (0.0, 0.0),
                }
            }
        }
    }

    /// Named natural parameters in canonical order.
    pub fn named_params(&self) -> Vec<(String, f64)> {
        match *self {
            Self::Independence => Vec::new(),
            Self::Gaussian { rho } => vec![("rho".into(), rho)],
            Self::StudentT { rho, nu } => vec![("rho".into(), rho), ("nu".into(), nu)],
            Self::Frank { theta } | Self::Clayton { theta, .. } | Self::Gumbel { theta, .. } => {
                vec![("theta".into(), theta)]
            }
        }
    }

    /// Pack parameters onto unconstrained coordinates `psi in R^p`.
    pub fn pack_psi(&self) -> Vec<f64> {
        match *self {
            Self::Independence => Vec::new(),
            Self::Gaussian { rho } => vec![rho.clamp(-MAX_RHO, MAX_RHO).atanh()],
            Self::StudentT { rho, nu } => {
                vec![rho.clamp(-MAX_RHO, MAX_RHO).atanh(), (nu - T_NU_SHIFT).max(1e-3).ln()]
            }
            Self::Frank { theta } => vec![theta],
            Self::Clayton { theta, .. } => vec![theta.max(CLAYTON_MIN).ln()],
            Self::Gumbel { theta, .. } => vec![(theta - 1.0).max(1e-5).ln()],
        }
    }

    /// Unpack unconstrained coordinates `psi` into the same copula variant.
    pub fn from_psi(&self, psi: &[f64]) -> Self {
        match *self {
            Self::Independence => Self::Independence,
            Self::Gaussian { .. } => Self::gaussian(psi[0].tanh()),
            Self::StudentT { .. } => {
                let rho = psi[0].tanh();
                let nu = T_NU_SHIFT + psi[1].clamp(-3.0, (T_NU_MAX - T_NU_SHIFT).ln()).exp();
                Self::student_t(rho, nu)
            }
            Self::Frank { .. } => Self::frank(psi[0]),
            Self::Clayton { rotation, .. } => {
                Self::clayton(psi[0].clamp(CLAYTON_MIN.ln(), CLAYTON_MAX.ln()).exp(), rotation)
            }
            Self::Gumbel { rotation, .. } => {
                let d = psi[0].clamp(-12.0, (GUMBEL_MAX - 1.0).ln()).exp();
                Self::gumbel(1.0 + d, rotation)
            }
        }
    }

    /// Weighted copula log-likelihood `weight * sum_i ln c(u_i, v_i)`.
    pub fn weighted_log_lik(&self, u: &[f64], v: &[f64], weight: f64) -> f64 {
        if matches!(self, Self::Independence) {
            return 0.0;
        }
        weight * u.iter().zip(v).map(|(&ui, &vi)| self.log_density(ui, vi)).sum::<f64>()
    }

    /// Score and positive-definite observed Fisher information on unconstrained `psi`.
    pub fn score_and_obs_info(
        &self, u: &[f64], v: &[f64], weight: f64,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let psi = self.pack_psi();
        let p = psi.len();
        if p == 0 {
            return (Vec::new(), Vec::new());
        }
        let h = 1e-4_f64;
        let ll0 = self.weighted_log_lik(u, v, weight);
        let mut grad = vec![0.0; p];
        let mut neg_hess = vec![vec![0.0; p]; p];

        for j in 0..p {
            let mut p_up = psi.clone();
            let mut p_dn = psi.clone();
            p_up[j] += h;
            p_dn[j] -= h;
            let ll_p = self.from_psi(&p_up).weighted_log_lik(u, v, weight);
            let ll_m = self.from_psi(&p_dn).weighted_log_lik(u, v, weight);
            grad[j] = (ll_p - ll_m) / (2.0 * h);
            let diag = -(ll_p - 2.0 * ll0 + ll_m) / (h * h);
            neg_hess[j][j] = if diag.is_finite() { diag.max(1e-4) } else { 1.0 };
        }
        if p == 2 {
            let v_pp = self.from_psi(&[psi[0] + h, psi[1] + h]).weighted_log_lik(u, v, weight);
            let v_pm = self.from_psi(&[psi[0] + h, psi[1] - h]).weighted_log_lik(u, v, weight);
            let v_mp = self.from_psi(&[psi[0] - h, psi[1] + h]).weighted_log_lik(u, v, weight);
            let v_mm = self.from_psi(&[psi[0] - h, psi[1] - h]).weighted_log_lik(u, v, weight);
            let cross = -(v_pp - v_pm - v_mp + v_mm) / (4.0 * h * h);
            let cross = if cross.is_finite() { cross } else { 0.0 };
            neg_hess[0][1] = cross;
            neg_hess[1][0] = cross;
        }
        make_spd_small(&mut neg_hess);
        (grad, neg_hess)
    }

    /// Full `p x p` inverse-observed-information delta-method standard errors.
    pub fn observed_se(&self, u: &[f64], v: &[f64], weight: f64) -> Vec<(String, f64)> {
        let names: Vec<String> = self.named_params().into_iter().map(|(n, _)| n).collect();
        if names.is_empty() {
            return Vec::new();
        }
        if u.len() < 4 || u.len() != v.len() || weight <= 0.0 {
            return names.into_iter().map(|n| (n, f64::NAN)).collect();
        }
        let (_, info) = self.score_and_obs_info(u, v, weight);
        let Some(cov_psi) = invert_small(&info) else {
            return names.into_iter().map(|n| (n, f64::NAN)).collect();
        };
        let jac_diag: Vec<f64> = match *self {
            Self::Independence => Vec::new(),
            Self::Gaussian { rho } => vec![1.0 - rho * rho],
            Self::StudentT { rho, nu } => vec![1.0 - rho * rho, (nu - T_NU_SHIFT).max(1e-3)],
            Self::Frank { .. } => vec![1.0],
            Self::Clayton { theta, .. } => vec![theta],
            Self::Gumbel { theta, .. } => vec![(theta - 1.0).max(1e-5)],
        };
        names
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                let var = cov_psi[i][i];
                let se = if var.is_finite() && var > 0.0 {
                    (jac_diag[i] * var.sqrt()).abs()
                } else {
                    f64::NAN
                };
                (name, se)
            })
            .collect()
    }

    /// Fit the parameters of this copula variant by MLE on `(u, v)`.
    pub fn fit_mle(&self, u: &[f64], v: &[f64]) -> Self {
        if u.len() < 3 || u.len() != v.len() {
            return *self;
        }
        match *self {
            Self::Independence => Self::Independence,
            Self::Gaussian { .. } => {
                let (rho, _) = brent_minimize(
                    |r| -Self::gaussian(r).weighted_log_lik(u, v, 1.0),
                    -MAX_RHO,
                    MAX_RHO,
                    1e-7,
                );
                Self::gaussian(rho)
            }
            Self::StudentT { .. } => {
                let (mut rho, _) = brent_minimize(
                    |r| -Self::student_t(r, 6.0).weighted_log_lik(u, v, 1.0),
                    -MAX_RHO,
                    MAX_RHO,
                    1e-6,
                );
                let mut nu = 6.0_f64;
                for _ in 0..8 {
                    let (new_nu, _) = brent_minimize(
                        |n| -Self::student_t(rho, n).weighted_log_lik(u, v, 1.0),
                        T_NU_SHIFT + 0.05,
                        T_NU_MAX,
                        1e-5,
                    );
                    nu = new_nu;
                    let (new_rho, _) = brent_minimize(
                        |r| -Self::student_t(r, nu).weighted_log_lik(u, v, 1.0),
                        -MAX_RHO,
                        MAX_RHO,
                        1e-6,
                    );
                    if (new_rho - rho).abs() < 1e-5 {
                        rho = new_rho;
                        break;
                    }
                    rho = new_rho;
                }
                Self::student_t(rho, nu)
            }
            Self::Frank { .. } => {
                let (t_pos, f_pos) = brent_minimize(
                    |t| -Self::frank(t).weighted_log_lik(u, v, 1.0),
                    1e-4,
                    FRANK_MAX,
                    1e-6,
                );
                let (t_neg, f_neg) = brent_minimize(
                    |t| -Self::frank(t).weighted_log_lik(u, v, 1.0),
                    -FRANK_MAX,
                    -1e-4,
                    1e-6,
                );
                if f_pos <= f_neg {
                    Self::frank(t_pos)
                } else {
                    Self::frank(t_neg)
                }
            }
            Self::Clayton { rotation, .. } => {
                let (t, _) = brent_minimize(
                    |t| -Self::clayton(t, rotation).weighted_log_lik(u, v, 1.0),
                    CLAYTON_MIN,
                    CLAYTON_MAX,
                    1e-6,
                );
                Self::clayton(t, rotation)
            }
            Self::Gumbel { rotation, .. } => {
                let (t, _) = brent_minimize(
                    |t| -Self::gumbel(t, rotation).weighted_log_lik(u, v, 1.0),
                    GUMBEL_MIN,
                    GUMBEL_MAX,
                    1e-6,
                );
                Self::gumbel(t, rotation)
            }
        }
    }

    /// Select the best bivariate copula family (including [`CopulaFamily::Independence`] and
    /// all rotations of Clayton/Gumbel) by BIC on `(u, v)` weighted by `weight`, with penalty
    /// `k * ln(n_subjects)`.
    pub fn select_bic(u: &[f64], v: &[f64], weight: f64, n_subjects: usize) -> Self {
        if u.len() < 6 || u.len() != v.len() || n_subjects < 6 || weight <= 0.0 {
            return Self::Independence;
        }
        let tau_emp = empirical_kendall_tau(u, v);
        let ln_n = (n_subjects as f64).ln();
        let bic = |cop: &Self| -> f64 {
            let ll = cop.weighted_log_lik(u, v, weight);
            if ll.is_finite() {
                -2.0 * ll + (cop.n_params() as f64) * ln_n
            } else {
                f64::INFINITY
            }
        };

        // Always test Independence, Gaussian, Student-t, Frank, plus the sign-consistent
        // (and borderline) rotations of Clayton and Gumbel.
        let mut templates: Vec<Self> = vec![
            Self::Gaussian { rho: 0.0 },
            Self::StudentT { rho: 0.0, nu: 6.0 },
            Self::Frank { theta: 0.5 },
        ];
        if tau_emp >= -0.05 {
            templates.push(Self::Clayton { theta: 0.5, rotation: CopulaRotation::R0 });
            templates.push(Self::Clayton { theta: 0.5, rotation: CopulaRotation::R180 });
            templates.push(Self::Gumbel { theta: 1.3, rotation: CopulaRotation::R0 });
            templates.push(Self::Gumbel { theta: 1.3, rotation: CopulaRotation::R180 });
        }
        if tau_emp <= 0.05 {
            templates.push(Self::Clayton { theta: 0.5, rotation: CopulaRotation::R90 });
            templates.push(Self::Clayton { theta: 0.5, rotation: CopulaRotation::R270 });
            templates.push(Self::Gumbel { theta: 1.3, rotation: CopulaRotation::R90 });
            templates.push(Self::Gumbel { theta: 1.3, rotation: CopulaRotation::R270 });
        }

        let mut best = Self::Independence;
        let mut best_bic = 0.0_f64; // BIC(Independence) == 0.0!
        for tmpl in templates {
            let fitted = tmpl.fit_mle(u, v);
            let b = bic(&fitted);
            if b + 1e-6 < best_bic {
                best = fitted;
                best_bic = b;
            }
        }
        best
    }
}

/// Robbins-Monro score & observed-information stochastic approximation state for one pair-copula.
#[derive(Debug, Clone, Default)]
pub struct CopulaSaState {
    avg_score: Vec<f64>,
    avg_info: Vec<Vec<f64>>,
    initialized: bool,
}

impl CopulaSaState {
    /// Reset state when pair-copula families are selected at `k = K_select`.
    pub fn reset(&mut self) {
        self.avg_score.clear();
        self.avg_info.clear();
        self.initialized = false;
    }

    /// Apply one Robbins-Monro score & Fisher-information stochastic approximation step.
    pub fn step(&mut self, cop: &mut CopulaFamily, u: &[f64], v: &[f64], gamma: f64) {
        if matches!(cop, CopulaFamily::Independence) || u.len() < 4 || gamma <= 0.0 {
            return;
        }
        let (g, info) = cop.score_and_obs_info(u, v, 1.0);
        let p = g.len();
        if p == 0 {
            return;
        }
        if !self.initialized || self.avg_score.len() != p {
            self.avg_score = g.iter().map(|&x| gamma * x).collect();
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
            let max_step = 0.25_f64;
            let norm_inf = delta.iter().fold(0.0_f64, |acc, &d| acc.max(d.abs()));
            if norm_inf > max_step {
                let scale = max_step / norm_inf;
                for d in &mut delta {
                    *d *= scale;
                }
            }
            let mut psi = cop.pack_psi();
            for i in 0..p {
                psi[i] += delta[i];
            }
            *cop = cop.from_psi(&psi);
        }
    }
}

/// Empirical Kendall's `tau` computed on `(u, v)` (subsampled deterministically if `n > 600`
/// so warm-up pools of `M * N` pairs remain fast).
pub fn empirical_kendall_tau(u: &[f64], v: &[f64]) -> f64 {
    let n_raw = u.len().min(v.len());
    if n_raw < 2 {
        return 0.0;
    }
    let stride = (n_raw / 500).max(1);
    let idx: Vec<usize> = (0..n_raw).step_by(stride).collect();
    let m = idx.len();
    if m < 2 {
        return 0.0;
    }
    let mut conc = 0_i64;
    let mut disc = 0_i64;
    for a in 0..m {
        let ia = idx[a];
        for &ib in &idx[(a + 1)..] {
            let du = u[ia] - u[ib];
            let dv = v[ia] - v[ib];
            let prod = du * dv;
            if prod > 0.0 {
                conc += 1;
            } else if prod < 0.0 {
                disc += 1;
            }
        }
    }
    let pairs = (m * (m - 1) / 2) as f64;
    (conc - disc) as f64 / pairs
}

fn kendall_tau_frank(theta: f64) -> f64 {
    if theta.abs() < 1e-8 {
        return 0.0;
    }
    let n = 200_usize;
    let dt = theta / n as f64;
    let integrand = |t: f64| -> f64 {
        if t.abs() < 1e-8 {
            1.0 - 0.5 * t + t * t / 12.0
        } else {
            t / t.exp_m1()
        }
    };
    let mut integral = 0.5 * (integrand(0.0) + integrand(theta));
    for i in 1..n {
        integral += integrand(i as f64 * dt);
    }
    integral *= dt;
    let d1 = integral / theta;
    1.0 - 4.0 * (1.0 - d1) / theta
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn all_family_instances() -> Vec<CopulaFamily> {
        vec![
            CopulaFamily::Independence,
            CopulaFamily::gaussian(0.55),
            CopulaFamily::gaussian(-0.45),
            CopulaFamily::student_t(0.50, 4.0),
            CopulaFamily::student_t(-0.35, 6.0),
            CopulaFamily::frank(4.5),
            CopulaFamily::frank(-3.8),
            CopulaFamily::clayton(2.2, CopulaRotation::R0),
            CopulaFamily::clayton(2.2, CopulaRotation::R90),
            CopulaFamily::clayton(2.2, CopulaRotation::R180),
            CopulaFamily::clayton(2.2, CopulaRotation::R270),
            CopulaFamily::gumbel(2.1, CopulaRotation::R0),
            CopulaFamily::gumbel(2.1, CopulaRotation::R90),
            CopulaFamily::gumbel(2.1, CopulaRotation::R180),
            CopulaFamily::gumbel(2.1, CopulaRotation::R270),
        ]
    }

    #[test]
    fn h1_and_h2_invert_cleanly_and_differentiate_to_copula_density() {
        let eps = 1e-6;
        let pts = [(0.15, 0.25), (0.25, 0.70), (0.50, 0.50), (0.75, 0.20), (0.85, 0.80)];
        for cop in all_family_instances() {
            for &(u, v) in &pts {
                let p1 = cop.h1(u, v);
                let u_back = cop.h1_inv(p1, v);
                assert!((u_back - u).abs() < 1e-8, "{cop:?} h1_inv(h1({u}, {v})) = {u_back}");

                let p2 = cop.h2(v, u);
                let v_back = cop.h2_inv(p2, u);
                assert!((v_back - v).abs() < 1e-8, "{cop:?} h2_inv(h2({v}, {u})) = {v_back}");

                // Fundamental copula calculus identities:
                //   dh1(u, v)/du == c(u, v)  AND  dh2(v, u)/dv == c(u, v)!
                let c = cop.log_density(u, v).exp();
                let dh1_du = (cop.h1(u + eps, v) - cop.h1(u - eps, v)) / (2.0 * eps);
                let dh2_dv = (cop.h2(v + eps, u) - cop.h2(v - eps, u)) / (2.0 * eps);
                assert!(
                    (dh1_du - c).abs() < 1e-5 * (1.0 + c),
                    "{cop:?} at ({u},{v}): dh1/du={dh1_du} vs c={c}"
                );
                assert!(
                    (dh2_dv - c).abs() < 1e-5 * (1.0 + c),
                    "{cop:?} at ({u},{v}): dh2/dv={dh2_dv} vs c={c}"
                );
            }
        }
    }

    #[test]
    fn every_copula_integrates_to_one_on_unit_square() {
        let m = 80_usize;
        let h = 1.0 / m as f64;
        for cop in all_family_instances() {
            let mut sum = 0.0;
            for i in 0..m {
                let u = (i as f64 + 0.5) * h;
                for j in 0..m {
                    let v = (j as f64 + 0.5) * h;
                    sum += cop.log_density(u, v).exp();
                }
            }
            let integral = sum * h * h;
            assert!((integral - 1.0).abs() < 2e-2, "{cop:?}: unit-square integral = {integral}");
        }
    }

    #[test]
    fn bic_selection_recovers_independence_clayton_and_rotated_clayton90() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 400;

        // 1. Independence -> must select Independence (0 params)!
        let u_ind: Vec<f64> = (0..n).map(|_| rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4)).collect();
        let v_ind: Vec<f64> = (0..n).map(|_| rng.random::<f64>().clamp(1e-4, 1.0 - 1e-4)).collect();
        let sel_ind = CopulaFamily::select_bic(&u_ind, &v_ind, 1.0, n);
        assert_eq!(sel_ind, CopulaFamily::Independence);

        // 2. Positive Clayton(2.5)
        let truth_pos = CopulaFamily::clayton(2.5, CopulaRotation::R0);
        let mut u_pos = Vec::with_capacity(n);
        let mut v_pos = Vec::with_capacity(n);
        for _ in 0..n {
            let u = rng.random::<f64>().clamp(1e-5, 1.0 - 1e-5);
            let w = rng.random::<f64>().clamp(1e-5, 1.0 - 1e-5);
            let v = truth_pos.h2_inv(w, u);
            u_pos.push(u);
            v_pos.push(v);
        }
        let sel_pos = CopulaFamily::select_bic(&u_pos, &v_pos, 1.0, n);
        assert_eq!(sel_pos.family_name(), "clayton");
        assert!((sel_pos.kendall_tau() - truth_pos.kendall_tau()).abs() < 0.08);

        // 3. Negative rotated Clayton90(2.5)
        let truth_r90 = CopulaFamily::clayton(2.5, CopulaRotation::R90);
        let mut u_neg = Vec::with_capacity(n);
        let mut v_neg = Vec::with_capacity(n);
        for _ in 0..n {
            let u = rng.random::<f64>().clamp(1e-5, 1.0 - 1e-5);
            let w = rng.random::<f64>().clamp(1e-5, 1.0 - 1e-5);
            let v = truth_r90.h2_inv(w, u);
            u_neg.push(u);
            v_neg.push(v);
        }
        let sel_neg = CopulaFamily::select_bic(&u_neg, &v_neg, 1.0, n);
        assert_eq!(sel_neg.family_name(), "clayton_90");
        assert!(sel_neg.kendall_tau() < -0.40);
    }
}
