//! Every random-effect distribution's latent log-density, over [`PkNum`] (#1716).
//!
//! [`latent_neg2_logpdf`] is `−2 ln p_z(z)` for a univariate kind at latent `z`, variance
//! (or scale) `ω` and shape θs `s`. `p_z(z) = p_η(m(z))·|m′(z)|` is the density of the
//! latent the estimators hold, with `m` the kind's domain map ([`DensityKind::map`]). It is
//! `+∞` where `m(z)` leaves the kind's support (a bounded kernel, Pareto's lower bound, a
//! GEV or GPD endpoint) or where `ω` and the shape admit no distribution. Every derivative
//! is the exact derivative of this one implementation.
//!
//! The parameterisations, so that `ω` keeps one meaning wherever it can:
//!
//! - On ℝ: mean 0 and variance `ω` (`logistic`, `laplace`, `gumbel`, `nig`, `pgg`, `gev`,
//!   `arcsine`, `uniform` and the kernels), scale² `ω` where no variance exists (`t`,
//!   `cauchy`).
//! - On (0, ∞): mean 1 and variance `ω` (`gamma`, `invgamma`, `lognormal`, `chisq`, `chi`,
//!   `weibull`, `frechet`, `inversegaussian`, `betaprime`, `noncentralchisq`, `pareto`,
//!   `gpd`); scale `ω` for `levy`, which has no mean. `exponential`, `rayleigh`,
//!   `erlang(k)`, `kolmogorov` and `loguniform(a, b)` have no free spread, so their `ω` is
//!   unused and must be `FIX`.
//! - On (0, 1): mean `μ` and variance `ω` (`beta`); `logit η ~ N(logit μ, ω)`
//!   (`logitnormal`).
//!
//! A shape a "mean 1, variance ω" family implies without a closed form (Weibull's `k`,
//! Fréchet's `α`, the scaled chi's `m`) is solved in `f64` and then refined by Newton steps
//! over `T` ([`implied_shape`]), so its jets are those of the exact root.

use super::DensityKind;
use crate::sens::jet::DualT;
use crate::sens::num::{BranchCmp as B, HostFn, JetFn, PkNum};

/// Euler–Mascheroni γ.
#[allow(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const PI: f64 = std::f64::consts::PI;

fn c<T: PkNum>(v: f64) -> T {
    T::from_f64(v)
}

/// `ln(1 + x)`, exact to rounding near 0.
pub(super) fn ln1p_g<T: PkNum>(x: T) -> T {
    x.jet_unary(JetFn::Ln1p)
}

/// `ln(1 + eˣ)`, without overflow.
pub(super) fn softplus_g<T: PkNum>(x: T) -> T {
    x.jet_unary(JetFn::Softplus)
}

/// `ln(1 + λu)/λ`, exact through λ = 0 (where it is `u`).
fn ln1p_over<T: PkNum>(u: T, lambda: T) -> T {
    u * (lambda * u).jet_unary(JetFn::Lnrel)
}

/// `expm1(λu)/λ`, exact through λ = 0 (where it is `u`).
fn expm1_over<T: PkNum>(u: T, lambda: T) -> T {
    u * (lambda * u).jet_unary(JetFn::Exprel)
}

/// `atan(x)`.
pub(super) fn atan_g<T: PkNum>(x: T) -> T {
    x.jet_unary(JetFn::Atan)
}

/// Whether `x` is a finite number, decided through the branch API.
pub(super) fn finite<T: PkNum>(x: T) -> bool {
    x.br_cmp_c(B::Gt, f64::NEG_INFINITY) && x.br_cmp_c(B::Lt, f64::INFINITY)
}

/// `ln K₁(x)` for `x > 0`.
pub(super) fn ln_bessel_k1<T: PkNum>(x: T) -> T {
    x.jet_unary(JetFn::LnBesselK1)
}

/// `ln B(a, b)`.
fn ln_beta<T: PkNum>(a: T, b: T) -> T {
    a.ln_gamma() + b.ln_gamma() - (a + b).ln_gamma()
}

/// The root `k` of `h(k, ω) = 0` over `T`: the `f64` root ([`implied_shape_root`], run on the
/// values as a [`HostFn`], so a recording re-solves it on every evaluation), then four Newton
/// steps over `T`, each on `h` and `∂h/∂k` from one first-order dual pass in `k`. A Newton
/// step from the exact root keeps its value and doubles the order to which its jets are
/// those of the implicit function, so four steps are exact through order 15. `None` when
/// the root does not exist.
fn implied_shape<T: PkNum>(
    f: HostFn, w: T, h: impl Fn(DualT<1, T>, T) -> DualT<1, T>,
) -> Option<T> {
    let mut k = T::host(f, &[w])[0];
    if k.br_is_nan() {
        return None;
    }
    for _ in 0..4 {
        let r = h(DualT { v: k, g: [c(1.0)] }, w);
        k = k - r.v / r.g[0];
    }
    finite(k).then_some(k)
}

/// `h(k, ω)` of Weibull's shape: `Γ(1 + 2/k)/Γ(1 + 1/k)² − 1 − ω`.
fn weibull_h<T: PkNum>(k: DualT<1, T>, w: T) -> DualT<1, T> {
    let one = DualT::<1, T>::constant(1.0);
    let two = DualT::<1, T>::constant(2.0);
    ((one + two / k).ln_gamma() - two * (one + one / k).ln_gamma()).exp()
        - one
        - DualT { v: w, g: [c(0.0)] }
}

/// `h(α, ω)` of Fréchet's shape (`α > 2`): `Γ(1 − 2/α)/Γ(1 − 1/α)² − 1 − ω`.
fn frechet_h<T: PkNum>(a: DualT<1, T>, w: T) -> DualT<1, T> {
    let one = DualT::<1, T>::constant(1.0);
    let two = DualT::<1, T>::constant(2.0);
    ((one - two / a).ln_gamma() - two * (one - one / a).ln_gamma()).exp()
        - one
        - DualT { v: w, g: [c(0.0)] }
}

/// `h(m, ω)` of the scaled chi's (Nakagami) shape, mean 1 and `E[η²] = 1 + ω`:
/// `lnΓ(m + ½) − lnΓ(m) + ½ ln((1 + ω)/m)`.
fn nakagami_h<T: PkNum>(m: DualT<1, T>, w: T) -> DualT<1, T> {
    let half = DualT::<1, T>::constant(0.5);
    let om = DualT { v: c::<T>(1.0) + w, g: [c(0.0)] };
    (m + half).ln_gamma() - m.ln_gamma() + half * (om / m).ln()
}

/// The `f64` root behind [`implied_shape`]: Newton from the shape's classic first guess,
/// clamped to its bracket, to convergence or 200 steps. `NaN` when the iteration leaves
/// the domain.
pub(crate) fn implied_shape_root(f: HostFn, w: f64) -> f64 {
    let (h, k0, lo, hi): (fn(DualT<1, f64>, f64) -> DualT<1, f64>, f64, f64, f64) = match f {
        // CV ≈ 1.2/k is the classic first guess.
        HostFn::WeibullShape => (weibull_h, (1.2 / w.sqrt()).clamp(0.05, 500.0), 0.02, 1e4),
        HostFn::FrechetShape => (frechet_h, 2.0 + 1.3 / w.sqrt(), 2.0 + 1e-9, 1e5),
        // Var ≈ 1/(4m) for large m.
        HostFn::NakagamiShape => (nakagami_h, (0.25 / w).max(0.5), 0.5 + 1e-12, 1e7),
        _ => return f64::NAN,
    };
    if !(w > 0.0 && w.is_finite()) {
        return f64::NAN;
    }
    let mut k = k0;
    for _ in 0..200 {
        let r = h(DualT { v: k, g: [1.0] }, w);
        if !(r.v.is_finite() && r.g[0].is_finite()) || r.g[0] == 0.0 {
            return f64::NAN;
        }
        let next = (k - r.v / r.g[0]).clamp(lo, hi);
        if (next - k).abs() <= 1e-15 * k.abs().max(1.0) {
            return next;
        }
        k = next;
    }
    k
}

/// Weibull's shape `k` with mean 1 and variance `ω`.
fn weibull_k<T: PkNum>(w: T) -> Option<T> {
    implied_shape(HostFn::WeibullShape, w, weibull_h)
}

/// Fréchet's shape `α > 2` with mean 1 and variance `ω`.
fn frechet_alpha<T: PkNum>(w: T) -> Option<T> {
    implied_shape(HostFn::FrechetShape, w, frechet_h)
}

/// The scaled chi's (Nakagami) shape `m` with mean 1 and `E[η²] = 1 + ω`.
fn nakagami_m<T: PkNum>(w: T) -> Option<T> {
    implied_shape(HostFn::NakagamiShape, w, nakagami_h)
}

/// `(Γ(1 − 2ξ) − Γ(1 − ξ)²)/ξ²` and `(Γ(1 − ξ) − 1)/ξ`, exact through ξ = 0, for the GEV.
/// Away from 0 the closed forms; within `|ξ| < 0.05` their series in
/// `L(x) = lnΓ(1 − x) = γx + Σ_{k≥2} ζ(k)xᵏ/k`.
fn gev_moments<T: PkNum>(xi: T) -> (T, T) {
    if !xi.br_abs_lt(0.05) {
        let one = c::<T>(1.0);
        let g1 = (one - xi).ln_gamma().exp();
        let g2 = (one - c::<T>(2.0) * xi).ln_gamma().exp();
        return ((g2 - g1 * g1) / (xi * xi), (g1 - one) / xi);
    }
    // ζ(2..=16).
    const ZETA: [f64; 15] = [
        1.644_934_066_848_226_4,
        1.202_056_903_159_594_3,
        1.082_323_233_711_138_2,
        1.036_927_755_143_369_9,
        1.017_343_061_984_449_1,
        1.008_349_277_381_922_8,
        1.004_077_356_197_944_3,
        1.002_008_392_826_082_2,
        1.000_994_575_127_818_1,
        1.000_494_188_604_119_5,
        1.000_246_086_553_308_1,
        1.000_122_713_347_578_5,
        1.000_061_248_135_058_7,
        1.000_030_588_236_307_0,
        1.000_015_282_259_408_7,
    ];
    // L(x)/x and (L(2x) − 2L(x))/x² as polynomials in x.
    let l_over = |x: T| {
        let mut acc = c::<T>(0.0);
        for (i, z) in ZETA.iter().enumerate().rev() {
            let k = (i + 2) as f64;
            acc = acc * x + c(z / k);
        }
        c::<T>(EULER_GAMMA) + x * acc
    };
    let mut d = c::<T>(0.0);
    for (i, z) in ZETA.iter().enumerate().rev() {
        let k = (i + 2) as f64;
        d = d * xi + c(z * (2f64.powi(k as i32) - 2.0) / k);
    }
    // (g1 − 1)/ξ = expm1(L(ξ))/ξ = expm1_over(ξ·(L/ξ)/ξ…): write L(ξ) = ξ·l, so
    // expm1(ξ l)/ξ = expm1_over(l, ξ).
    let l = l_over(xi);
    let m1 = expm1_over(l, xi);
    // g2 − g1² = g1²·expm1(L(2ξ) − 2L(ξ)), and L(2ξ) − 2L(ξ) = ξ²·d.
    let g1 = (xi * l).exp();
    let var = g1 * g1 * expm1_over(d, xi * xi);
    (var, m1)
}

/// `−2 ln p_z(z)` of a univariate `kind` at latent `z`, variance or scale `w`, shapes `s`
/// (see the module docs). `+∞` outside the support or the parameter domain.
pub(super) fn latent_neg2_logpdf<T: PkNum>(kind: DensityKind, z: T, w: T, s: &[T]) -> T {
    use DensityKind as K;
    let inf = c::<T>(f64::INFINITY);
    let ln2pi = c::<T>(std::f64::consts::TAU.ln());
    let one = c::<T>(1.0);
    let two = c::<T>(2.0);
    if !w.br_cmp_c(B::Gt, 0.0) {
        return inf;
    }
    // ln p_η at η, for the exp-mapped kinds; the latent adds the Jacobian `z`.
    let pos = |lp: T| -> T { c::<T>(-2.0) * (lp + z) };
    match kind {
        K::StudentT => unreachable!("the t is scored over its block"),
        K::Copula => unreachable!("a copula is scored over its block"),
        K::Normal => z * z / w + w.ln() + ln2pi,
        K::Logistic => {
            let sc = (c::<T>(3.0) * w).sqrt() * c(1.0 / PI);
            let u = z / sc;
            two * u + c::<T>(4.0) * softplus_g(-u) + two * sc.ln()
        }
        K::Laplace => {
            let b = (w * c(0.5)).sqrt();
            two * (two * b).ln() + two * z.abs() / b
        }
        K::Cauchy => {
            let g = w.sqrt();
            let u = z / g;
            two * (c::<T>(PI) * g).ln() + two * ln1p_g(u * u)
        }
        K::Gumbel => {
            let beta = (c::<T>(6.0) * w).sqrt() * c(1.0 / PI);
            let u = (z + c::<T>(EULER_GAMMA) * beta) / beta;
            two * (beta.ln() + u + (-u).exp())
        }
        K::NormalInverseGaussian => {
            // ζ = δγ > 0, ρ = β/α ∈ (−1, 1); mean 0, variance ω.
            let (zeta, rho) = (s[0], s[1]);
            if !(zeta.br_cmp_c(B::Gt, 0.0) && rho.br_abs_lt(1.0)) {
                return inf;
            }
            let om = one - rho * rho;
            let alpha = (zeta / w).sqrt() / om;
            let beta = rho * alpha;
            let gamma = alpha * om.sqrt();
            let delta = zeta / gamma;
            let mu = -delta * beta / gamma;
            let x = z - mu;
            let q = (delta * delta + x * x).sqrt();
            let lp = alpha.ln() + delta.ln() - c::<T>(PI).ln() + ln_bessel_k1(alpha * q) - q.ln()
                + delta * gamma
                + beta * x;
            c::<T>(-2.0) * lp
        }
        K::PGeneralizedGaussian => {
            let p = s[0];
            if !p.br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            let ip = one / p;
            let a = (w * ((ip).ln_gamma() - (c::<T>(3.0) * ip).ln_gamma()).exp()).sqrt();
            let u = (z / a).abs();
            // |u|^p as exp(p ln|u|), with u = 0 exactly the limit 0.
            let up = if u.br_cmp_c(B::Eq, 0.0) { c::<T>(0.0) } else { (p * u.ln()).exp() };
            two * ((two * a).ln() + (one + ip).ln_gamma() + up)
        }
        K::GeneralizedExtremeValue => {
            let xi = s[0];
            if !xi.br_cmp_c(B::Lt, 0.5) {
                return inf;
            }
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let mu = -sigma * m1;
            let u = (z - mu) / sigma;
            if !(c::<T>(1.0) + xi * u).br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            // ln t = −ln(1 + ξu)/ξ; ln p = −ln σ + (ξ + 1) ln t − t.
            let ln_t = -ln1p_over(u, xi);
            two * (sigma.ln() - (xi + one) * ln_t + ln_t.exp())
        }
        K::Arcsine => {
            let a2 = two * w;
            if !(z * z).br_cmp(B::Lt, a2) {
                return inf;
            }
            two * c::<T>(PI).ln() + (a2 - z * z).ln()
        }
        K::Uniform => {
            let a = (c::<T>(3.0) * w).sqrt();
            if !z.abs().br_cmp(B::Lt, a) {
                return inf;
            }
            two * (two * a).ln()
        }
        K::Biweight
        | K::Triweight
        | K::Epanechnikov
        | K::Cosine
        | K::Semicircle
        | K::SymTriangular => {
            // Variance of the unit kernel on (−1, 1), and its log-density there.
            let v1 = match kind {
                K::Biweight => 1.0 / 7.0,
                K::Triweight => 1.0 / 9.0,
                K::Epanechnikov => 1.0 / 5.0,
                K::Cosine => 1.0 - 8.0 / (PI * PI),
                K::Semicircle => 0.25,
                _ => 1.0 / 6.0,
            };
            let sig = (w / c(v1)).sqrt();
            let u = z / sig;
            if !u.br_abs_lt(1.0) {
                return inf;
            }
            let one_m = one - u * u;
            let lf = match kind {
                K::Biweight => c::<T>((15.0f64 / 16.0).ln()) + two * one_m.ln(),
                K::Triweight => c::<T>((35.0f64 / 32.0).ln()) + c::<T>(3.0) * one_m.ln(),
                K::Epanechnikov => c::<T>(0.75f64.ln()) + one_m.ln(),
                K::Cosine => c::<T>((PI / 4.0).ln()) + (c::<T>(PI / 2.0) * u).cos().ln(),
                K::Semicircle => c::<T>((2.0 / PI).ln()) + c::<T>(0.5) * one_m.ln(),
                _ => (one - u.abs()).ln(),
            };
            c::<T>(-2.0) * (lf - sig.ln())
        }
        K::Gamma | K::ChiSq => {
            let k = one / w;
            c::<T>(-2.0) * (k * z - z.exp() * k - k.ln_gamma() - k * w.ln())
        }
        K::Exponential => {
            // Gamma with k = 1, mean 1: ln p = −η.
            pos(-z.exp())
        }
        K::Erlang => {
            let k = s[0];
            if !k.br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            // Gamma(k, 1/k): ln p = k ln k + (k − 1) ln η − kη − lnΓ(k).
            let eta = z.exp();
            pos(k * k.ln() + (k - one) * z - k * eta - k.ln_gamma())
        }
        K::InvGamma => {
            let alpha = two + one / w;
            let beta = alpha - one;
            c::<T>(-2.0) * (alpha * beta.ln() - alpha.ln_gamma() - alpha * z - beta * (-z).exp())
        }
        K::LogNormal => {
            let s2 = ln1p_g(w);
            let r = z + s2 * c(0.5);
            r * r / s2 + s2.ln() + ln2pi
        }
        K::Chi => {
            let Some(m) = nakagami_m(w) else {
                return inf;
            };
            let om = one + w;
            // Nakagami: ln p = ln 2 + m ln m − lnΓ(m) − m ln Ω + (2m − 1) ln η − mη²/Ω.
            let eta = z.exp();
            pos(c::<T>(2f64.ln()) + m * m.ln() - m.ln_gamma() - m * om.ln() + (two * m - one) * z
                - m * eta * eta / om)
        }
        K::Rayleigh => {
            // Weibull k = 2 with mean 1: σ = √(2/π).
            let s2 = 2.0 / PI;
            let eta = z.exp();
            pos(z - c::<T>(s2.ln()) - eta * eta * c(1.0 / (2.0 * s2)))
        }
        K::Weibull => {
            let Some(k) = weibull_k(w) else {
                return inf;
            };
            let lam = -(one + one / k).ln_gamma(); // ln λ, λ = 1/Γ(1 + 1/k)
            let lx = z - lam; // ln(η/λ)
            pos(k.ln() - lam + (k - one) * lx - (k * lx).exp())
        }
        K::Frechet => {
            let Some(a) = frechet_alpha(w) else {
                return inf;
            };
            let ls = -(one - one / a).ln_gamma(); // ln s, s = 1/Γ(1 − 1/α)
            let lx = z - ls;
            pos(a.ln() - ls - (one + a) * lx - (-a * lx).exp())
        }
        K::InverseGaussian => {
            let lam = one / w;
            let eta = z.exp();
            let r = eta - one;
            pos(c::<T>(0.5) * (lam.ln() - ln2pi) - c::<T>(1.5) * z - lam * r * r / (two * eta))
        }
        K::BetaPrime => {
            let b = two + two / w;
            let a = b - one;
            pos((a - one) * z - (a + b) * softplus_g(z) - ln_beta(a, b))
        }
        K::NoncentralChiSq => {
            let k = s[0];
            if !k.br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            // Mean 1: X = Y/(k + λ), Y ~ χ'²(k, λ), with 2(k + 2λ)/(k + λ)² = ω.
            let qa = w;
            let qb = two * w * k - c(4.0);
            let qc = w * k * k - two * k;
            let disc = qb * qb - c::<T>(4.0) * qa * qc;
            if !disc.br_cmp_c(B::Ge, 0.0) {
                return inf;
            }
            let lam = (-qb + disc.sqrt()) / (two * qa);
            if !lam.br_cmp_c(B::Ge, 0.0) {
                return inf;
            }
            let sc = one / (k + lam);
            let y = z.exp() / sc;
            pos(ln_ncx2_pdf(y, k, lam) - sc.ln())
        }
        K::Kolmogorov => {
            let x = z.exp();
            // f(x) = 8x Σ (−1)^{k−1} k² e^{−2k²x²} for x ≥ 1, the theta-transformed series
            // below, each to rounding.
            let lf = if x.br_cmp_c(B::Ge, 1.0) {
                // The leading `e^{−2x²}` factored out, as for x < 1: summed as is, it is
                // subnormal by x ≈ 19.
                let mut s = c::<T>(0.0);
                for k in 1..=20 {
                    let kf = k as f64;
                    let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
                    s = s + c::<T>(sign * kf * kf) * (c::<T>(-2.0 * (kf * kf - 1.0)) * x * x).exp();
                }
                (c::<T>(8.0) * x).ln() - c::<T>(2.0) * x * x + s.ln()
            } else {
                // In log space, the leading `e^{−π²/(8x²)}` factored out: summed as is, it
                // is subnormal by x ≈ 0.04 (its derivative overflowing) and 0 below.
                let a1 = PI * PI;
                let mut s = c::<T>(0.0);
                for k in 1..=20 {
                    let a = ((2 * k - 1) as f64).powi(2) * PI * PI;
                    s = s
                        + (c::<T>(a / 4.0) / (x * x) - one)
                            * (c::<T>(-(a - a1) / 8.0) / (x * x)).exp();
                }
                c::<T>((2.0 * PI).sqrt()).ln() - two * x.ln() - c::<T>(a1 / 8.0) / (x * x) + s.ln()
            };
            pos(lf)
        }
        K::Levy => {
            let eta = z.exp();
            pos(c::<T>(0.5) * (w.ln() - ln2pi) - w / (two * eta) - c::<T>(1.5) * z)
        }
        K::Pareto => {
            let a = one + (one + one / w).sqrt();
            let lxm = ((a - one) / a).ln();
            if !z.br_cmp(B::Ge, lxm) {
                return inf;
            }
            pos(a.ln() + a * lxm - (a + one) * z)
        }
        K::GeneralizedPareto => {
            // Mean 1, variance ω: ξ = (1 − 1/ω)/2, σ = 1 − ξ.
            let xi = (one - one / w) * c(0.5);
            let sigma = one - xi;
            let u = z.exp() / sigma;
            if !(c::<T>(1.0) + xi * u).br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            // ln p = −ln σ − (1/ξ + 1) ln(1 + ξu).
            pos(-sigma.ln() - ln1p_over(u, xi) - ln1p_g(xi * u))
        }
        K::LogUniform => {
            let (a, b) = (s[0], s[1]);
            if !(a.br_cmp_c(B::Gt, 0.0) && b.br_cmp(B::Gt, a))
                || !(z.br_cmp(B::Gt, a.ln()) && z.br_cmp(B::Lt, b.ln()))
            {
                return inf;
            }
            pos(-z - (b.ln() - a.ln()).ln())
        }
        K::Beta => {
            let mu = s[0];
            let n = mu * (one - mu) / w - one;
            if !n.br_cmp_c(B::Gt, 0.0) {
                return inf;
            }
            let (pa, pb) = (mu * n, (one - mu) * n);
            two * (pa * softplus_g(-z) + pb * softplus_g(z) + ln_beta(pa, pb))
        }
        K::LogitNormal => {
            let mu = s[0];
            let m = mu.ln() - (one - mu).ln();
            let r = z - m;
            r * r / w + w.ln() + ln2pi
        }
    }
}

/// A draw of the random effect `η` itself from a univariate `kind` (the latent is
/// `m⁻¹(η)`), at variance or scale `w` and shapes `s`. `NaN` when the parameters admit no
/// distribution.
pub(super) fn sample_eta(kind: DensityKind, w: f64, s: &[f64], rng: &mut impl rand::Rng) -> f64 {
    use rand_distr::{
        Beta, Cauchy, ChiSquared, Distribution, Gamma, Gumbel, InverseGaussian,
        NormalInverseGaussian, Poisson, StandardNormal, Uniform,
    };
    use DensityKind as K;
    fn draw_u(rng: &mut impl rand::Rng) -> f64 {
        Uniform::new(0.0f64, 1.0)
            .map_or(0.5, |d| d.sample(rng))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }
    match kind {
        K::StudentT => unreachable!("the t is drawn over its block"),
        K::Copula => unreachable!("a copula is drawn over its block"),
        K::Normal => {
            let n: f64 = StandardNormal.sample(rng);
            w.sqrt() * n
        }
        K::Logistic => {
            let u = draw_u(rng);
            (3.0 * w).sqrt() / PI * (u / (1.0 - u)).ln()
        }
        K::Laplace => {
            let b = (w / 2.0).sqrt();
            let u = draw_u(rng) - 0.5;
            -b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
        }
        K::Cauchy => Cauchy::new(0.0, w.sqrt()).map_or(f64::NAN, |d| d.sample(rng)),
        K::Gumbel => {
            let beta = (6.0 * w).sqrt() / PI;
            Gumbel::new(-EULER_GAMMA * beta, beta).map_or(f64::NAN, |d| d.sample(rng))
        }
        K::NormalInverseGaussian => {
            let (zeta, rho) = (s[0], s[1]);
            let om = 1.0 - rho * rho;
            let alpha = (zeta / w).sqrt() / om;
            let beta = rho * alpha;
            let gamma = alpha * om.sqrt();
            let delta = zeta / gamma;
            let mu = -delta * beta / gamma;
            // Y ~ NIG(αδ, βδ) (δ = 1, μ = 0), then η = μ + δY.
            NormalInverseGaussian::new(alpha * delta, beta * delta)
                .map_or(f64::NAN, |d| mu + delta * d.sample(rng))
        }
        K::PGeneralizedGaussian => {
            let p = s[0];
            let a = (w * (special::ln_gamma(1.0 / p) - special::ln_gamma(3.0 / p)).exp()).sqrt();
            let g = Gamma::new(1.0 / p, 1.0).map_or(f64::NAN, |d| d.sample(rng));
            let sign = if draw_u(rng) < 0.5 { -1.0 } else { 1.0 };
            sign * a * g.powf(1.0 / p)
        }
        K::GeneralizedExtremeValue => {
            let xi = s[0];
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let mu = -sigma * m1;
            // x = μ + σ((−ln U)^{−ξ} − 1)/ξ = μ + σ·expm1(−ξ ln(−ln U))/ξ.
            let l = -(-draw_u(rng).ln()).ln();
            mu + sigma * expm1_over(l, xi)
        }
        K::Arcsine => (2.0 * w).sqrt() * (PI * draw_u(rng)).cos(),
        K::Uniform => (3.0 * w).sqrt() * (2.0 * draw_u(rng) - 1.0),
        K::Biweight | K::Triweight | K::Epanechnikov | K::Semicircle => {
            let (shape, v1) = match kind {
                K::Biweight => (3.0, 1.0 / 7.0),
                K::Triweight => (4.0, 1.0 / 9.0),
                K::Epanechnikov => (2.0, 1.0 / 5.0),
                _ => (1.5, 0.25),
            };
            let b = Beta::new(shape, shape).map_or(f64::NAN, |d| d.sample(rng));
            (w / v1).sqrt() * (2.0 * b - 1.0)
        }
        K::Cosine => {
            let sig = (w / (1.0 - 8.0 / (PI * PI))).sqrt();
            // F(u) = (1 + sin(πu/2))/2 ⇒ u = (2/π) asin(2U − 1).
            sig * 2.0 / PI * (2.0 * draw_u(rng) - 1.0).asin()
        }
        K::SymTriangular => (6.0 * w).sqrt() * (draw_u(rng) + draw_u(rng) - 1.0),
        K::Gamma | K::ChiSq => Gamma::new(1.0 / w, w).map_or(f64::NAN, |d| d.sample(rng)),
        K::Exponential => -draw_u(rng).ln(),
        K::Erlang => Gamma::new(s[0], 1.0 / s[0]).map_or(f64::NAN, |d| d.sample(rng)),
        K::InvGamma => {
            let alpha = 2.0 + 1.0 / w;
            Gamma::new(alpha, 1.0 / (alpha - 1.0)).map_or(f64::NAN, |d| 1.0 / d.sample(rng))
        }
        K::LogNormal => {
            let s2 = w.ln_1p();
            let n: f64 = StandardNormal.sample(rng);
            (-0.5 * s2 + s2.sqrt() * n).exp()
        }
        K::Chi => {
            let Some(m) = nakagami_m(w) else {
                return f64::NAN;
            };
            Gamma::new(m, (1.0 + w) / m).map_or(f64::NAN, |d| d.sample(rng).sqrt())
        }
        K::Rayleigh => (2.0f64 / PI).sqrt() * (-2.0 * draw_u(rng).ln()).sqrt(),
        K::Weibull => {
            let Some(k) = weibull_k(w) else {
                return f64::NAN;
            };
            let lam = (-special::ln_gamma(1.0 + 1.0 / k)).exp();
            lam * (-draw_u(rng).ln()).powf(1.0 / k)
        }
        K::Frechet => {
            let Some(a) = frechet_alpha(w) else {
                return f64::NAN;
            };
            let sc = (-special::ln_gamma(1.0 - 1.0 / a)).exp();
            sc * (-draw_u(rng).ln()).powf(-1.0 / a)
        }
        K::InverseGaussian => {
            InverseGaussian::new(1.0, 1.0 / w).map_or(f64::NAN, |d| d.sample(rng))
        }
        K::BetaPrime => {
            let b = 2.0 + 2.0 / w;
            let a = b - 1.0;
            let g1 = Gamma::new(a, 1.0).map_or(f64::NAN, |d| d.sample(rng));
            let g2 = Gamma::new(b, 1.0).map_or(f64::NAN, |d| d.sample(rng));
            g1 / g2
        }
        K::NoncentralChiSq => {
            let k = s[0];
            let (qa, qb, qc) = (w, 2.0 * w * k - 4.0, w * k * k - 2.0 * k);
            let lam = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
            let j = if lam > 0.0 {
                Poisson::new(lam / 2.0).map_or(0.0, |d| d.sample(rng))
            } else {
                0.0
            };
            ChiSquared::new(k + 2.0 * j).map_or(f64::NAN, |d| d.sample(rng)) / (k + lam)
        }
        K::Kolmogorov => {
            // Inverse CDF by bisection on F(x) = 1 − 2 Σ (−1)^{k−1} e^{−2k²x²}.
            let u = draw_u(rng);
            let cdf = |x: f64| -> f64 {
                if x <= 0.0 {
                    return 0.0;
                }
                if x < 1.0 {
                    let mut s = 0.0;
                    for k in 1..=20 {
                        let a = ((2 * k - 1) as f64).powi(2) * PI * PI;
                        s += (-a / (8.0 * x * x)).exp();
                    }
                    (2.0 * PI).sqrt() / x * s
                } else {
                    let mut s = 0.0;
                    for k in 1..=20 {
                        let kf = k as f64;
                        s += if k % 2 == 1 { 1.0 } else { -1.0 } * (-2.0 * kf * kf * x * x).exp();
                    }
                    1.0 - 2.0 * s
                }
            };
            let (mut lo, mut hi) = (1e-6, 10.0);
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if cdf(mid) < u {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        }
        K::Levy => {
            let n: f64 = StandardNormal.sample(rng);
            w / (n * n)
        }
        K::Pareto => {
            let a = 1.0 + (1.0 + 1.0 / w).sqrt();
            let xm = (a - 1.0) / a;
            xm * draw_u(rng).powf(-1.0 / a)
        }
        K::GeneralizedPareto => {
            let xi = (1.0 - 1.0 / w) / 2.0;
            let sigma = 1.0 - xi;
            // x = σ·((1 − U)^{−ξ} − 1)/ξ = σ·expm1(−ξ ln(1 − U))/ξ.
            sigma * expm1_over(-(1.0 - draw_u(rng)).ln(), xi)
        }
        K::LogUniform => {
            let (a, b) = (s[0].ln(), s[1].ln());
            (a + (b - a) * draw_u(rng)).exp()
        }
        K::Beta => {
            let mu = s[0];
            let n = mu * (1.0 - mu) / w - 1.0;
            Beta::new(mu * n, (1.0 - mu) * n).map_or(f64::NAN, |d| d.sample(rng))
        }
        K::LogitNormal => {
            let mu = s[0];
            let n: f64 = StandardNormal.sample(rng);
            let z = (mu / (1.0 - mu)).ln() + w.sqrt() * n;
            1.0 / (1.0 + (-z).exp())
        }
    }
}

/// Terms of [`inc_beta`]'s continued fraction. On the side of `(a + 1)/(a + b + 2)` it takes,
/// the fraction converges in `O(√max(a, b))` terms, so 300 cover the shapes a density meets
/// (the kinds tests pin the value against the adaptive stop it replaced). A fixed count keeps
/// a recording's plan independent of where the fraction converged; terms past convergence
/// multiply `f` by 1.
const INC_BETA_TERMS: usize = 300;

/// The regularized incomplete beta `I_x(a, b)` over `T`, by its continued fraction
/// (modified Lentz), on the side of `x = (a + 1)/(a + b + 2)` where it converges fast.
pub(super) fn inc_beta<T: PkNum>(a: T, b: T, x: T) -> T {
    let one = c::<T>(1.0);
    if x.br_cmp_c(B::Le, 0.0) {
        return c(0.0);
    }
    if x.br_cmp_c(B::Ge, 1.0) {
        return one;
    }
    if x.br_cmp(B::Gt, (a + one) / (a + b + c(2.0))) {
        return one - inc_beta(b, a, one - x);
    }
    let front = (a * x.ln() + b * ln1p_g(-x) - ln_beta(a, b)).exp() / a;
    // Lentz's floor on a vanishing denominator, as a selection a recording re-makes.
    let floor = |v: T| T::select_if(B::Lt, v.abs(), c(1e-300), c(1e-300), v);
    let mut f = one;
    let mut cc = one;
    let mut d = c::<T>(0.0);
    for i in 0..INC_BETA_TERMS {
        let m = (i / 2) as f64;
        let num = if i == 0 {
            one
        } else if i % 2 == 0 {
            c::<T>(m) * (b - c(m)) * x / ((a + c(2.0 * m - 1.0)) * (a + c(2.0 * m)))
        } else {
            -(a + c(m)) * (a + b + c(m)) * x / ((a + c(2.0 * m)) * (a + c(2.0 * m + 1.0)))
        };
        d = one / floor(one + num * d);
        cc = floor(one + num / cc);
        f = f * (cc * d);
    }
    front * (f - one)
}

/// `ln p_η(x)` of a univariate kind at the random effect `x` itself (not the latent):
/// the latent density at `m⁻¹(x)` less the map's log-Jacobian. A t here is the univariate
/// t, scale² `w`.
pub(super) fn eta_log_pdf<T: PkNum>(kind: DensityKind, x: T, w: T, s: &[T]) -> T {
    use super::DomainMap;
    let one = c::<T>(1.0);
    if kind == DensityKind::StudentT {
        let nu = s[0];
        let q = x * x / w;
        return c::<T>(-0.5)
            * ((nu + one) * ln1p_g(q / nu)
                + w.ln()
                + (nu * c(PI)).ln()
                + c::<T>(2.0) * (nu * c(0.5)).ln_gamma()
                - c::<T>(2.0) * ((nu + one) * c(0.5)).ln_gamma());
    }
    let (z, log_jac) = match kind.map() {
        DomainMap::Identity => (x, c::<T>(0.0)),
        DomainMap::Exp => {
            if !x.br_cmp_c(B::Gt, 0.0) {
                return c(f64::NEG_INFINITY);
            }
            let z = x.ln();
            (z, z)
        }
        DomainMap::InvLogit => {
            if !(x.br_cmp_c(B::Gt, 0.0) && x.br_cmp_c(B::Lt, 1.0)) {
                return c(f64::NEG_INFINITY);
            }
            (x.ln() - (one - x).ln(), x.ln() + (one - x).ln())
        }
    };
    c::<T>(-0.5) * latent_neg2_logpdf(kind, z, w, s) - log_jac
}

/// `ln f(y)` of the noncentral χ²(k, λ) density, over `T`. The Poisson-mixture series
/// `Σ_j Pois(j; λ/2) f_{χ²_{k+2j}}(y)` costs ~√j* terms around its peak `j* ≈ √(λy)/2`,
/// so far in the tail (`√(λy)` past [`NCX2_ASYMPTOTIC_FROM`]) it uses the Bessel form
/// `f = ½ e^{−(y+λ)/2} (y/λ)^{k/4−½} I_{k/2−1}(√(λy))` with `I`'s large-argument
/// expansion instead, which is exact to rounding there; the two are pinned against each
/// other across the switch.
pub(super) fn ln_ncx2_pdf<T: PkNum>(y: T, k: T, lam: T) -> T {
    let x = (lam * y).sqrt();
    if x.br_cmp_c(B::Gt, NCX2_ASYMPTOTIC_FROM) {
        ln_ncx2_asymptotic(y, k, lam)
    } else {
        ln_ncx2_series(y, k, lam)
    }
}

/// Where [`ln_ncx2_pdf`] switches to the Bessel asymptotic. Its m-th term shrinks the
/// last by `|4ν² − (2m − 1)²| / (8mx)`, about `m/(2x)` for the small ν (`k/2 − 1`) the
/// density meets, so at `x = 1000` sixteen terms leave a remainder far below rounding.
pub(super) const NCX2_ASYMPTOTIC_FROM: f64 = 1000.0;

/// The terms `j` a noncentral χ² series sums around its peak `center`: they fall like
/// `e^{−(j − j*)²/(2s²)}` with `s² ≤ j* + 1`, so `10·√(center + 1) + 15` each side reach
/// `e^{−50}` below the peak. The centre is a guarded floor (`br_floor_div`), so a recording's
/// window follows the value on every evaluation.
fn ncx2_window(center: f64) -> std::ops::Range<usize> {
    let half = (10.0 * (center + 1.0).sqrt()).ceil() as usize + 15;
    let c = center as usize;
    c.saturating_sub(half)..c + half + 1
}

/// The series form of [`ln_ncx2_pdf`]: `Σ_j Pois(j; λ/2)·f_{χ²_{k+2j}}(y)`, whose terms are
/// unimodal in `j`, peaking where `(j + 1)(j + k/2) = λy/4`; summed over [`ncx2_window`].
pub(super) fn ln_ncx2_series<T: PkNum>(y: T, k: T, lam: T) -> T {
    let one = c::<T>(1.0);
    let hl = lam * c(0.5);
    let lchi = |kk: T| {
        (kk * c(0.5) - one) * y.ln()
            - y * c(0.5)
            - kk * c(0.5 * 2f64.ln())
            - (kk * c(0.5)).ln_gamma()
    };
    if hl.br_cmp_c(B::Eq, 0.0) {
        return lchi(k);
    }
    let term = |j: usize| -> T {
        let jf = j as f64;
        -hl + c::<T>(jf) * hl.ln() - c::<T>(jf + 1.0).ln_gamma() + lchi(k + c(2.0 * jf))
    };
    let hk = k * c(0.5);
    let peak = (((hk - one) * (hk - one) + c::<T>(2.0) * hl * y).sqrt() - (one + hk)) * c(0.5);
    let center = if peak.br_cmp_c(B::Gt, 0.0) { peak.br_floor_div(1.0) } else { 0.0 };
    let terms: Vec<T> = ncx2_window(center).map(term).collect();
    crate::sens::num::log_sum_exp(&terms)
}

/// The Bessel asymptotic form of [`ln_ncx2_pdf`]:
/// `ln I_ν(x) = x − ½ ln(2πx) + ln Σ_m (−1)^m a_m(ν) x^{−m}`,
/// `a_m = Π_{i ≤ m} (4ν² − (2i − 1)²) / (m! 8^m)`, to 16 terms.
pub(super) fn ln_ncx2_asymptotic<T: PkNum>(y: T, k: T, lam: T) -> T {
    let half = c::<T>(0.5);
    let nu = k * half - c(1.0);
    let x = (lam * y).sqrt();
    let mu4 = c::<T>(4.0) * nu * nu;
    let mut a = c::<T>(1.0);
    let mut sum = c::<T>(1.0);
    for m in 1..=16 {
        let odd = (2 * m - 1) as f64;
        a = a * (mu4 - c(odd * odd)) / (c::<T>(8.0 * m as f64) * x);
        sum = if m % 2 == 1 { sum - a } else { sum + a };
    }
    let ln_i = x - half * (c::<T>(std::f64::consts::TAU) * x).ln() + sum.ln();
    half.ln() - (y + lam) * half + (k * c(0.25) - half) * (y.ln() - lam.ln()) + ln_i
}

/// The mean and variance of the random effect a univariate `kind` describes, at
/// variance or scale `w` and shapes `s`: the module's parameterisation (mean 0 / 1 / μ and
/// variance ω where the family has them), stated per kind. `None` where a moment does not
/// exist (Cauchy, Lévy, log-uniform's is not ω, a t with ν ≤ 2) or has no closed form
/// (logit-normal, Kolmogorov). The kinds tests check every entry against quadrature of the
/// density itself. Consumers: the exact CV% of a reported effect and effect-scale
/// shrinkage.
pub(super) fn effect_moments(kind: DensityKind, w: f64, s: &[f64]) -> Option<(f64, f64)> {
    use DensityKind as K;
    match kind {
        K::Cauchy | K::Levy | K::LogitNormal | K::Kolmogorov | K::LogUniform | K::Copula => None,
        // Scale² ω: Var = ω ν / (ν − 2), for ν > 2.
        K::StudentT => {
            let nu = *s.first()?;
            (nu > 2.0).then(|| (0.0, w * nu / (nu - 2.0)))
        }
        K::Normal => Some((0.0, w)),
        K::Exponential => Some((1.0, 1.0)),
        K::Erlang => Some((1.0, 1.0 / s.first()?)),
        K::Rayleigh => Some((1.0, 4.0 / PI - 1.0)),
        K::Beta => Some((*s.first()?, w)),
        K::Logistic
        | K::Laplace
        | K::Gumbel
        | K::NormalInverseGaussian
        | K::PGeneralizedGaussian
        | K::GeneralizedExtremeValue
        | K::Arcsine
        | K::Uniform
        | K::Biweight
        | K::Triweight
        | K::Epanechnikov
        | K::Cosine
        | K::Semicircle
        | K::SymTriangular => Some((0.0, w)),
        K::Gamma
        | K::InvGamma
        | K::LogNormal
        | K::ChiSq
        | K::Chi
        | K::Weibull
        | K::Frechet
        | K::InverseGaussian
        | K::BetaPrime
        | K::NoncentralChiSq
        | K::Pareto
        | K::GeneralizedPareto => Some((1.0, w)),
    }
}

/// `F_η(x)`, the CDF of a univariate kind at the random effect `x`, over `T`; `None` for a
/// kind with no CDF here (none at present). The NIG has no closed form and is integrated
/// numerically (the trapezoid rule in `u = ln(x − t)` over its exponential tails).
pub(super) fn eta_cdf<T: PkNum>(kind: DensityKind, x: T, w: T, s: &[T]) -> Option<T> {
    use special::{regularized_gamma_p, regularized_gamma_q};
    // `Φ` as one recorded primitive: `normal_cdf_g`'s tail switches are branches on the
    // value, which a latent crosses during an EBE search (a recording would re-record).
    let normal_cdf_g = |z: T| z.jet_unary(JetFn::Pnorm);
    // `if x ≥ 0 { a } else { b }` as a selection a recording re-makes, not a guard.
    let by_sign = |a: T, b: T| T::select_if(B::Ge, x, c(0.0), a, b);
    use DensityKind as K;
    let one = c::<T>(1.0);
    let half = c::<T>(0.5);
    let zero = c::<T>(0.0);
    let pos = x.br_cmp_c(B::Gt, 0.0);
    Some(match kind {
        K::Copula => return None,
        K::Normal => normal_cdf_g(x / w.sqrt()),
        K::StudentT => {
            let nu = s[0];
            let t2 = x * x / w;
            let tail = half * inc_beta(nu * half, half, nu / (nu + t2));
            by_sign(one - tail, tail)
        }
        K::Logistic => (x / ((c::<T>(3.0) * w).sqrt() * c(1.0 / PI))).inv_logit(),
        K::Laplace => {
            let b = (w * half).sqrt();
            // `e^{−|x|/b}` on both arms, so the unselected one cannot overflow.
            let e = half * (-x.abs() / b).exp();
            by_sign(one - e, e)
        }
        K::Cauchy => half + atan_g(x / w.sqrt()) * c(1.0 / PI),
        K::Gumbel => {
            let beta = (c::<T>(6.0) * w).sqrt() * c(1.0 / PI);
            let u = (x + c::<T>(EULER_GAMMA) * beta) / beta;
            (-(-u).exp()).exp()
        }
        K::NormalInverseGaussian => {
            // F(x) = ∫₀^∞ p(x − e^v·σ₀)·e^v·σ₀ dv over v ∈ [−40, 6] ∪ the tail.
            let sd = w.sqrt();
            let h = 0.02;
            let mut acc = zero;
            let mut v = -40.0;
            while v <= 6.0 {
                let du = sd * c(v.exp());
                let wt =
                    if (v + 40.0).abs() < 1e-12 || (v - 6.0).abs() < 1e-12 { 0.5 } else { 1.0 };
                acc = acc + c::<T>(wt * h) * eta_log_pdf(kind, x - du, w, s).exp() * du;
                v += h;
            }
            acc
        }
        K::PGeneralizedGaussian => {
            let p = s[0];
            let ip = one / p;
            let a = (w * (ip.ln_gamma() - (c::<T>(3.0) * ip).ln_gamma()).exp()).sqrt();
            let u = (x / a).abs();
            let up = if u.br_cmp_c(B::Eq, 0.0) { zero } else { (p * u.ln()).exp() };
            let g = half * regularized_gamma_p(ip, up);
            by_sign(half + g, half - g)
        }
        K::GeneralizedExtremeValue => {
            let xi = s[0];
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let u = (x + sigma * m1) / sigma;
            if !(c::<T>(1.0) + xi * u).br_cmp_c(B::Gt, 0.0) {
                return Some(if xi.br_cmp_c(B::Gt, 0.0) { zero } else { one });
            }
            (-(-ln1p_over(u, xi)).exp()).exp()
        }
        K::Arcsine => {
            let r = x / (c::<T>(2.0) * w).sqrt();
            if r.br_cmp_c(B::Le, -1.0) {
                return Some(zero);
            }
            if r.br_cmp_c(B::Ge, 1.0) {
                return Some(one);
            }
            // ½ + asin(r)/π = 1 − acos(r)/π.
            one - r.acos() * c(1.0 / PI)
        }
        K::Uniform => {
            let a = (c::<T>(3.0) * w).sqrt();
            let u = (x + a) / (c::<T>(2.0) * a);
            T::select_if(B::Le, u, zero, zero, T::select_if(B::Ge, u, one, one, u))
        }
        K::Biweight
        | K::Triweight
        | K::Epanechnikov
        | K::Cosine
        | K::Semicircle
        | K::SymTriangular => {
            let v1 = match kind {
                K::Biweight => 1.0 / 7.0,
                K::Triweight => 1.0 / 9.0,
                K::Epanechnikov => 1.0 / 5.0,
                K::Cosine => 1.0 - 8.0 / (PI * PI),
                K::Semicircle => 0.25,
                _ => 1.0 / 6.0,
            };
            let u = x / (w / c(v1)).sqrt();
            if u.br_cmp_c(B::Le, -1.0) {
                return Some(zero);
            }
            if u.br_cmp_c(B::Ge, 1.0) {
                return Some(one);
            }
            let u2 = u * u;
            match kind {
                K::Biweight => {
                    half + c::<T>(15.0 / 16.0)
                        * u
                        * (one - c::<T>(2.0 / 3.0) * u2 + c::<T>(0.2) * u2 * u2)
                }
                K::Triweight => {
                    half + c::<T>(35.0 / 32.0)
                        * u
                        * (one - u2 + c::<T>(0.6) * u2 * u2 - c::<T>(1.0 / 7.0) * u2 * u2 * u2)
                }
                K::Epanechnikov => half + c::<T>(0.75) * u * (one - u2 * c(1.0 / 3.0)),
                K::Cosine => half * (one + (c::<T>(PI / 2.0) * (one - u)).cos()),
                K::Semicircle => {
                    half + (u * (one - u2).sqrt() + (c::<T>(PI / 2.0) - u.acos())) * c(1.0 / PI)
                }
                _ => by_sign(one - half * (one - u) * (one - u), half * (one + u) * (one + u)),
            }
        }
        K::Gamma | K::ChiSq => {
            if !pos {
                return Some(zero);
            }
            let k = one / w;
            regularized_gamma_p(k, k * x)
        }
        K::Exponential => {
            if !pos {
                return Some(zero);
            }
            one - (-x).exp()
        }
        K::Erlang => {
            if !pos {
                return Some(zero);
            }
            regularized_gamma_p(s[0], s[0] * x)
        }
        K::InvGamma => {
            if !pos {
                return Some(zero);
            }
            let alpha = c::<T>(2.0) + one / w;
            regularized_gamma_q(alpha, (alpha - one) / x)
        }
        K::LogNormal => {
            if !pos {
                return Some(zero);
            }
            let s2 = ln1p_g(w);
            normal_cdf_g((x.ln() + s2 * half) / s2.sqrt())
        }
        K::Chi => {
            if !pos {
                return Some(zero);
            }
            let m = nakagami_m(w)?;
            regularized_gamma_p(m, m * x * x / (one + w))
        }
        K::Rayleigh => {
            if !pos {
                return Some(zero);
            }
            one - (-x * x * c(PI / 4.0)).exp()
        }
        K::Weibull => {
            if !pos {
                return Some(zero);
            }
            let k = weibull_k(w)?;
            let lam = -(one + one / k).ln_gamma();
            one - (-(k * (x.ln() - lam)).exp()).exp()
        }
        K::Frechet => {
            if !pos {
                return Some(zero);
            }
            let a = frechet_alpha(w)?;
            let ls = -(one - one / a).ln_gamma();
            (-(-a * (x.ln() - ls)).exp()).exp()
        }
        K::InverseGaussian => {
            if !pos {
                return Some(zero);
            }
            let lam = one / w;
            let r = (lam / x).sqrt();
            normal_cdf_g(r * (x - one)) + (c::<T>(2.0) * lam).exp() * normal_cdf_g(-r * (x + one))
        }
        K::BetaPrime => {
            if !pos {
                return Some(zero);
            }
            let b = c::<T>(2.0) + c::<T>(2.0) / w;
            inc_beta(b - one, b, x / (one + x))
        }
        K::NoncentralChiSq => {
            if !pos {
                return Some(zero);
            }
            let k = s[0];
            let (qa, qb, qc) = (w, c::<T>(2.0) * w * k - c(4.0), w * k * k - c::<T>(2.0) * k);
            let lam = (-qb + (qb * qb - c::<T>(4.0) * qa * qc).sqrt()) / (c::<T>(2.0) * qa);
            let y = x * (k + lam);
            let hl = lam * half;
            if hl.br_cmp_c(B::Eq, 0.0) {
                return Some(regularized_gamma_p(k * half, y * half));
            }
            // Pois(j; λ/2)·P((k + 2j)/2, y/2), peaked at j = λ/2 (see `ncx2_window`).
            let mut acc = zero;
            for j in ncx2_window(hl.br_floor_div(1.0)) {
                let jf = j as f64;
                let lp = -hl + c::<T>(jf) * hl.ln() - c::<T>(jf + 1.0).ln_gamma();
                acc = acc + lp.exp() * regularized_gamma_p((k + c(2.0 * jf)) * half, y * half);
            }
            acc
        }
        K::Kolmogorov => {
            if !pos {
                return Some(zero);
            }
            if x.br_cmp_c(B::Lt, 1.0) {
                let mut acc = zero;
                for k in 1..=20 {
                    let a = ((2 * k - 1) as f64).powi(2) * PI * PI;
                    acc = acc + (c::<T>(-a / 8.0) / (x * x)).exp();
                }
                c::<T>((2.0 * PI).sqrt()) / x * acc
            } else {
                let mut acc = zero;
                for k in 1..=20 {
                    let kf = k as f64;
                    let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
                    acc = acc + c::<T>(sign) * (c::<T>(-2.0 * kf * kf) * x * x).exp();
                }
                one - c::<T>(2.0) * acc
            }
        }
        K::Levy => {
            if !pos {
                return Some(zero);
            }
            special::erfc_g((w / (c::<T>(2.0) * x)).sqrt())
        }
        K::Pareto => {
            let a = one + (one + one / w).sqrt();
            let xm = (a - one) / a;
            if !x.br_cmp(B::Gt, xm) {
                return Some(zero);
            }
            one - (a * (xm.ln() - x.ln())).exp()
        }
        K::GeneralizedPareto => {
            if !pos {
                return Some(zero);
            }
            let xi = (one - one / w) * half;
            let sigma = one - xi;
            let u = x / sigma;
            if !(c::<T>(1.0) + xi * u).br_cmp_c(B::Gt, 0.0) {
                return Some(one);
            }
            one - (-ln1p_over(u, xi)).exp()
        }
        K::LogUniform => {
            let (a, b) = (s[0], s[1]);
            if !x.br_cmp(B::Gt, a) {
                return Some(zero);
            }
            if !x.br_cmp(B::Lt, b) {
                return Some(one);
            }
            (x.ln() - a.ln()) / (b.ln() - a.ln())
        }
        K::Beta => {
            let mu = s[0];
            let n = mu * (one - mu) / w - one;
            inc_beta(mu * n, (one - mu) * n, x)
        }
        K::LogitNormal => {
            if !(x.br_cmp_c(B::Gt, 0.0) && x.br_cmp_c(B::Lt, 1.0)) {
                return Some(if x.br_cmp_c(B::Le, 0.0) { zero } else { one });
            }
            let mu = s[0];
            let m = mu.ln() - (one - mu).ln();
            normal_cdf_g((x.ln() - (one - x).ln() - m) / w.sqrt())
        }
    })
}

use crate::stats::special;

#[cfg(test)]
#[path = "eta_density_kinds_tests.rs"]
mod tests;
