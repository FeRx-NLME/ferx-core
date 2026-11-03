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
use crate::sens::num::PkNum;
use crate::sens::rtape::DualT;

/// Euler–Mascheroni γ.
#[allow(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const PI: f64 = std::f64::consts::PI;

fn c<T: PkNum>(v: f64) -> T {
    T::from_f64(v)
}

/// `ln(1 + x)`: the value from libm's `ln_1p`, the jets from `ln(1 + x)`.
pub(super) fn ln1p_g<T: PkNum>(x: T) -> T {
    let l = (c::<T>(1.0) + x).ln();
    (l - c(l.val())) + c(x.val().ln_1p())
}

/// `expm1(x)`: the value from libm's `exp_m1`, the jets from `exp`.
fn expm1_g<T: PkNum>(x: T) -> T {
    let e = x.exp();
    if !e.val().is_finite() {
        return e;
    }
    (e - c(e.val())) + c(x.val().exp_m1())
}

/// `ln(1 + eˣ)`, without overflow.
pub(super) fn softplus_g<T: PkNum>(x: T) -> T {
    if x.val() > 0.0 {
        x + ln1p_g((-x).exp())
    } else {
        ln1p_g(x.exp())
    }
}

/// `ln(1 + λu)/λ`, exact through λ = 0 (where it is `u`), by its series below `|λu| = 0.1`.
fn ln1p_over<T: PkNum>(u: T, lambda: T) -> T {
    let x = lambda * u;
    if x.val().abs() < 0.1 {
        let mut acc = c::<T>(1.0 / 16.0);
        for k in (0..15).rev() {
            acc = c::<T>(1.0 / (k + 1) as f64) - x * acc;
        }
        u * acc
    } else {
        ln1p_g(x) * (c::<T>(1.0) / lambda)
    }
}

/// `expm1(λu)/λ`, exact through λ = 0 (where it is `u`).
fn expm1_over<T: PkNum>(u: T, lambda: T) -> T {
    let x = lambda * u;
    if x.val().abs() < 0.1 {
        let mut coef = [0.0f64; 12];
        let mut fact = 1.0;
        for (k, cf) in coef.iter_mut().enumerate() {
            fact *= (k + 1) as f64;
            *cf = 1.0 / fact;
        }
        let mut acc = c::<T>(coef[11]);
        for cf in coef[..11].iter().rev() {
            acc = c::<T>(*cf) + x * acc;
        }
        u * acc
    } else {
        expm1_g(x) * (c::<T>(1.0) / lambda)
    }
}

/// `atan(x)`: two half-angle reductions (`atan x = 2 atan(x / (1 + √(1 + x²)))`) bring
/// `|x| ≤ 1` below `tan(π/16)`, where a 14-term Taylor series is exact to rounding; past
/// `|x| = 1`, `atan x = ±π/2 − atan(1/x)`.
pub(super) fn atan_g<T: PkNum>(x: T) -> T {
    let one = c::<T>(1.0);
    if x.val().abs() > 1.0 {
        let half_pi = c::<T>(if x.val() > 0.0 { PI / 2.0 } else { -PI / 2.0 });
        return half_pi - atan_g(one / x);
    }
    let mut y = x;
    for _ in 0..2 {
        y = y / (one + (one + y * y).sqrt());
    }
    // atan y = y·Σ (−y²)^k/(2k + 1), by Horner from k = 13.
    let y2 = y * y;
    let mut acc = c::<T>(1.0 / 27.0);
    for k in (0..13).rev() {
        acc = c::<T>(1.0 / (2 * k + 1) as f64) - y2 * acc;
    }
    c::<T>(4.0) * y * acc
}

/// `ln Σ exp(a_j)`, stable.
fn log_sum_exp<T: PkNum>(a: &[T]) -> T {
    let m = a.iter().map(|v| v.val()).fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return c(m);
    }
    let mut s = c::<T>(0.0);
    for &v in a {
        s = s + (v - c(m)).exp();
    }
    c::<T>(m) + s.ln()
}

/// `ln K₁(x)` for `x > 0`, by the trapezoid rule on `K₁(x) = ∫₀^∞ e^{−x cosh t} cosh t dt`
/// at step `1/8`. The integrand is analytic in a strip, so the rule's error is
/// `O(e^{−π²·8})`, far below rounding; nodes run until the integrand falls `e^{−745}`
/// below its largest term.
pub(super) fn ln_bessel_k1<T: PkNum>(x: T) -> T {
    let h = 0.125;
    let mut terms = Vec::new();
    let mut j = 0usize;
    loop {
        let t = h * j as f64;
        let ch = t.cosh();
        let a = -x * c(ch) + c(ch.ln() + if j == 0 { 0.5f64.ln() } else { 0.0 });
        let done = j > 0
            && a.val()
                - terms
                    .iter()
                    .map(|v: &T| v.val())
                    .fold(f64::NEG_INFINITY, f64::max)
                < -745.0;
        terms.push(a);
        if done || j > 20_000 {
            break;
        }
        j += 1;
    }
    c::<T>(h.ln()) + log_sum_exp(&terms)
}

/// `ln B(a, b)`.
fn ln_beta<T: PkNum>(a: T, b: T) -> T {
    a.ln_gamma() + b.ln_gamma() - (a + b).ln_gamma()
}

/// The root `k` of `h(k) = 0` near `k0`, over `T`: Newton in `f64` from `k0` to convergence,
/// then four Newton steps over `T` from that root, each on `h` and `h′` from one
/// first-order dual pass in `k`. A Newton step from the exact root leaves its value and
/// doubles the order to which its jets are those of the implicit function, so four steps
/// are exact through order 15. `None` when the `f64` iteration does not converge inside
/// `[lo, hi]`.
pub(super) fn implied_shape<T: PkNum>(
    h: impl Fn(DualT<1, T>) -> DualT<1, T>,
    h64: impl Fn(DualT<1, f64>) -> DualT<1, f64>,
    k0: f64,
    lo: f64,
    hi: f64,
) -> Option<T> {
    let mut k = k0;
    for _ in 0..200 {
        let r = h64(DualT { v: k, g: [1.0] });
        if !(r.v.is_finite() && r.g[0].is_finite()) || r.g[0] == 0.0 {
            return None;
        }
        let next = (k - r.v / r.g[0]).clamp(lo, hi);
        if (next - k).abs() <= 1e-15 * k.abs().max(1.0) {
            k = next;
            break;
        }
        k = next;
    }
    let mut kt = c::<T>(k);
    for _ in 0..4 {
        let r = h(DualT { v: kt, g: [c(1.0)] });
        kt = kt - r.v / r.g[0];
    }
    kt.val().is_finite().then_some(kt)
}

/// Weibull's shape `k` with mean 1 and variance `ω`: `Γ(1 + 2/k)/Γ(1 + 1/k)² − 1 = ω`.
fn weibull_k<T: PkNum>(w: T) -> Option<T> {
    let w64 = w.val();
    let h = |k: DualT<1, T>| {
        let one = DualT::<1, T>::constant(1.0);
        let two = DualT::<1, T>::constant(2.0);
        let wl = DualT { v: w, g: [c(0.0)] };
        ((one + two / k).ln_gamma() - two * (one + one / k).ln_gamma()).exp() - one - wl
    };
    let h64 = |k: DualT<1, f64>| {
        let one = DualT::<1, f64>::constant(1.0);
        let two = DualT::<1, f64>::constant(2.0);
        ((one + two / k).ln_gamma() - two * (one + one / k).ln_gamma()).exp()
            - one
            - DualT::constant(w64)
    };
    // CV ≈ 1.2/k is the classic first guess.
    implied_shape(h, h64, (1.2 / w64.sqrt()).clamp(0.05, 500.0), 0.02, 1e4)
}

/// Fréchet's shape `α > 2` with mean 1 and variance `ω`: `Γ(1 − 2/α)/Γ(1 − 1/α)² − 1 = ω`.
fn frechet_alpha<T: PkNum>(w: T) -> Option<T> {
    let w64 = w.val();
    let h = |a: DualT<1, T>| {
        let one = DualT::<1, T>::constant(1.0);
        let two = DualT::<1, T>::constant(2.0);
        let wl = DualT { v: w, g: [c(0.0)] };
        ((one - two / a).ln_gamma() - two * (one - one / a).ln_gamma()).exp() - one - wl
    };
    let h64 = |a: DualT<1, f64>| {
        let one = DualT::<1, f64>::constant(1.0);
        let two = DualT::<1, f64>::constant(2.0);
        ((one - two / a).ln_gamma() - two * (one - one / a).ln_gamma()).exp()
            - one
            - DualT::constant(w64)
    };
    implied_shape(h, h64, 2.0 + 1.3 / w64.sqrt(), 2.0 + 1e-9, 1e5)
}

/// The scaled chi's (Nakagami) shape `m` with mean 1 and `E[η²] = 1 + ω`:
/// `lnΓ(m + ½) − lnΓ(m) + ½ ln((1 + ω)/m) = 0`.
fn nakagami_m<T: PkNum>(w: T) -> Option<T> {
    let w64 = w.val();
    let h = |m: DualT<1, T>| {
        let half = DualT::<1, T>::constant(0.5);
        let om = DualT {
            v: c::<T>(1.0) + w,
            g: [c(0.0)],
        };
        (m + half).ln_gamma() - m.ln_gamma() + half * (om / m).ln()
    };
    let h64 = |m: DualT<1, f64>| {
        let half = DualT::<1, f64>::constant(0.5);
        (m + half).ln_gamma() - m.ln_gamma() + half * (DualT::constant(1.0 + w64) / m).ln()
    };
    // Var ≈ 1/(4m) for large m.
    implied_shape(h, h64, (0.25 / w64).max(0.5), 0.5 + 1e-12, 1e7)
}

/// `(Γ(1 − 2ξ) − Γ(1 − ξ)²)/ξ²` and `(Γ(1 − ξ) − 1)/ξ`, exact through ξ = 0, for the GEV.
/// Away from 0 the closed forms; within `|ξ| < 0.05` their series in
/// `L(x) = lnΓ(1 − x) = γx + Σ_{k≥2} ζ(k)xᵏ/k`.
fn gev_moments<T: PkNum>(xi: T) -> (T, T) {
    if xi.val().abs() >= 0.05 {
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
    if !(w.val() > 0.0) {
        return inf;
    }
    // ln p_η at η, for the exp-mapped kinds; the latent adds the Jacobian `z`.
    let pos = |lp: T| -> T { c::<T>(-2.0) * (lp + z) };
    match kind {
        K::StudentT => unreachable!("the t is scored over its block"),
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
            if !(zeta.val() > 0.0 && rho.val().abs() < 1.0) {
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
            if !(p.val() > 0.0) {
                return inf;
            }
            let ip = one / p;
            let a = (w * ((ip).ln_gamma() - (c::<T>(3.0) * ip).ln_gamma()).exp()).sqrt();
            let u = (z / a).abs();
            // |u|^p as exp(p ln|u|), with u = 0 exactly the limit 0.
            let up = if u.val() == 0.0 {
                c::<T>(0.0)
            } else {
                (p * u.ln()).exp()
            };
            two * ((two * a).ln() + (one + ip).ln_gamma() + up)
        }
        K::GeneralizedExtremeValue => {
            let xi = s[0];
            if !(xi.val() < 0.5) {
                return inf;
            }
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let mu = -sigma * m1;
            let u = (z - mu) / sigma;
            if !(1.0 + xi.val() * u.val() > 0.0) {
                return inf;
            }
            // ln t = −ln(1 + ξu)/ξ; ln p = −ln σ + (ξ + 1) ln t − t.
            let ln_t = -ln1p_over(u, xi);
            two * (sigma.ln() - (xi + one) * ln_t + ln_t.exp())
        }
        K::Arcsine => {
            let a2 = two * w;
            if !(z.val() * z.val() < a2.val()) {
                return inf;
            }
            two * c::<T>(PI).ln() + (a2 - z * z).ln()
        }
        K::Uniform => {
            let a = (c::<T>(3.0) * w).sqrt();
            if !(z.val().abs() < a.val()) {
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
            if !(u.val().abs() < 1.0) {
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
            if !(k.val() > 0.0) {
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
            pos(
                c::<T>(2f64.ln()) + m * m.ln() - m.ln_gamma() - m * om.ln() + (two * m - one) * z
                    - m * eta * eta / om,
            )
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
            if !(k.val() > 0.0) {
                return inf;
            }
            // Mean 1: X = Y/(k + λ), Y ~ χ'²(k, λ), with 2(k + 2λ)/(k + λ)² = ω.
            let qa = w;
            let qb = two * w * k - c(4.0);
            let qc = w * k * k - two * k;
            let disc = qb * qb - c::<T>(4.0) * qa * qc;
            if !(disc.val() >= 0.0) {
                return inf;
            }
            let lam = (-qb + disc.sqrt()) / (two * qa);
            if !(lam.val() >= 0.0) {
                return inf;
            }
            let sc = one / (k + lam);
            let y = z.exp() / sc;
            let hl = lam * c(0.5);
            // ln f_Y(y) = ln Σ_j Pois(j; λ/2) f_{χ²_{k+2j}}(y), to `e^{−40}` of its peak.
            let mut terms = Vec::new();
            let peak_j = hl.val().floor() as usize;
            let mut j = 0usize;
            loop {
                let jf = j as f64;
                let kk = k + c(2.0 * jf);
                let lpois = -hl
                    + c::<T>(jf)
                        * if hl.val() > 0.0 {
                            hl.ln()
                        } else {
                            c(f64::NEG_INFINITY)
                        }
                    - c::<T>(jf + 1.0).ln_gamma();
                let lchi = (kk * c(0.5) - one) * y.ln()
                    - y * c(0.5)
                    - kk * c(0.5 * 2f64.ln())
                    - (kk * c(0.5)).ln_gamma();
                let t = if j == 0 && hl.val() == 0.0 {
                    lchi
                } else {
                    lpois + lchi
                };
                let small = j > peak_j + 5
                    && t.val()
                        < terms
                            .iter()
                            .map(|v: &T| v.val())
                            .fold(f64::NEG_INFINITY, f64::max)
                            - 40.0;
                terms.push(t);
                if small || j > 100_000 || hl.val() == 0.0 {
                    break;
                }
                j += 1;
            }
            pos(log_sum_exp(&terms) - sc.ln())
        }
        K::Kolmogorov => {
            let x = z.exp();
            // f(x) = 8x Σ (−1)^{k−1} k² e^{−2k²x²} for x ≥ 1, the theta-transformed series
            // below, each to rounding.
            let lf = if x.val() >= 1.0 {
                let mut s = c::<T>(0.0);
                for k in 1..=20 {
                    let kf = k as f64;
                    let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
                    s = s + c::<T>(sign * kf * kf) * (c::<T>(-2.0 * kf * kf) * x * x).exp();
                }
                (c::<T>(8.0) * x * s).ln()
            } else {
                let mut s = c::<T>(0.0);
                for k in 1..=20 {
                    let a = ((2 * k - 1) as f64).powi(2) * PI * PI;
                    s = s + (c::<T>(a / 4.0) / (x * x) - one) * (c::<T>(-a / 8.0) / (x * x)).exp();
                }
                c::<T>((2.0 * PI).sqrt()).ln() - two * x.ln() + s.ln()
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
            if !(z.val() >= lxm.val()) {
                return inf;
            }
            pos(a.ln() + a * lxm - (a + one) * z)
        }
        K::GeneralizedPareto => {
            // Mean 1, variance ω: ξ = (1 − 1/ω)/2, σ = 1 − ξ.
            let xi = (one - one / w) * c(0.5);
            let sigma = one - xi;
            let u = z.exp() / sigma;
            if !(1.0 + xi.val() * u.val() > 0.0) {
                return inf;
            }
            // ln p = −ln σ − (1/ξ + 1) ln(1 + ξu).
            pos(-sigma.ln() - ln1p_over(u, xi) - ln1p_g(xi * u))
        }
        K::LogUniform => {
            let (a, b) = (s[0], s[1]);
            if !(a.val() > 0.0 && b.val() > a.val())
                || !(z.val() > a.val().ln() && z.val() < b.val().ln())
            {
                return inf;
            }
            pos(-z - (b.ln() - a.ln()).ln())
        }
        K::Beta => {
            let mu = s[0];
            let n = mu * (one - mu) / w - one;
            if !(n.val() > 0.0) {
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

/// The regularized incomplete beta `I_x(a, b)` over `T`, by its continued fraction
/// (modified Lentz), on the side of `x = (a + 1)/(a + b + 2)` where it converges fast.
pub(super) fn inc_beta<T: PkNum>(a: T, b: T, x: T) -> T {
    let one = c::<T>(1.0);
    if x.val() <= 0.0 {
        return c(0.0);
    }
    if x.val() >= 1.0 {
        return one;
    }
    if x.val() > (a.val() + 1.0) / (a.val() + b.val() + 2.0) {
        return one - inc_beta(b, a, one - x);
    }
    let front = (a * x.ln() + b * ln1p_g(-x) - ln_beta(a, b)).exp() / a;
    let tiny = 1e-300;
    let mut f = one;
    let mut cc = one;
    let mut d = c::<T>(0.0);
    for i in 0..400 {
        let m = (i / 2) as f64;
        let num = if i == 0 {
            one
        } else if i % 2 == 0 {
            c::<T>(m) * (b - c(m)) * x / ((a + c(2.0 * m - 1.0)) * (a + c(2.0 * m)))
        } else {
            -(a + c(m)) * (a + b + c(m)) * x / ((a + c(2.0 * m)) * (a + c(2.0 * m + 1.0)))
        };
        d = one + num * d;
        if d.val().abs() < tiny {
            d = c(tiny);
        }
        d = one / d;
        cc = one + num / cc;
        if cc.val().abs() < tiny {
            cc = c(tiny);
        }
        let delta = cc * d;
        f = f * delta;
        if (delta.val() - 1.0).abs() < 1e-16 {
            break;
        }
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
            if !(x.val() > 0.0) {
                return c(f64::NEG_INFINITY);
            }
            let z = x.ln();
            (z, z)
        }
        DomainMap::InvLogit => {
            if !(x.val() > 0.0 && x.val() < 1.0) {
                return c(f64::NEG_INFINITY);
            }
            (x.ln() - (one - x).ln(), x.ln() + (one - x).ln())
        }
    };
    c::<T>(-0.5) * latent_neg2_logpdf(kind, z, w, s) - log_jac
}

/// `F_η(x)`, the CDF of a univariate kind at the random effect `x`, over `T`; `None` for a
/// kind with no CDF here (none at present). The NIG has no closed form and is integrated
/// numerically (the trapezoid rule in `u = ln(x − t)` over its exponential tails).
pub(super) fn eta_cdf<T: PkNum>(kind: DensityKind, x: T, w: T, s: &[T]) -> Option<T> {
    use special::{normal_cdf_g, regularized_gamma_p, regularized_gamma_q};
    use DensityKind as K;
    let one = c::<T>(1.0);
    let half = c::<T>(0.5);
    let zero = c::<T>(0.0);
    let pos = x.val() > 0.0;
    Some(match kind {
        K::StudentT => {
            let nu = s[0];
            let t2 = x * x / w;
            let tail = half * inc_beta(nu * half, half, nu / (nu + t2));
            if x.val() >= 0.0 {
                one - tail
            } else {
                tail
            }
        }
        K::Logistic => (x / ((c::<T>(3.0) * w).sqrt() * c(1.0 / PI))).inv_logit(),
        K::Laplace => {
            let b = (w * half).sqrt();
            if x.val() < 0.0 {
                half * (x / b).exp()
            } else {
                one - half * (-x / b).exp()
            }
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
                let wt = if (v + 40.0).abs() < 1e-12 || (v - 6.0).abs() < 1e-12 {
                    0.5
                } else {
                    1.0
                };
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
            let up = if u.val() == 0.0 {
                zero
            } else {
                (p * u.ln()).exp()
            };
            let g = half * regularized_gamma_p(ip, up);
            if x.val() >= 0.0 {
                half + g
            } else {
                half - g
            }
        }
        K::GeneralizedExtremeValue => {
            let xi = s[0];
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let u = (x + sigma * m1) / sigma;
            if !(1.0 + xi.val() * u.val() > 0.0) {
                return Some(if xi.val() > 0.0 { zero } else { one });
            }
            (-(-ln1p_over(u, xi)).exp()).exp()
        }
        K::Arcsine => {
            let r = x / (c::<T>(2.0) * w).sqrt();
            if r.val() <= -1.0 {
                return Some(zero);
            }
            if r.val() >= 1.0 {
                return Some(one);
            }
            // ½ + asin(r)/π = 1 − acos(r)/π.
            one - r.acos() * c(1.0 / PI)
        }
        K::Uniform => {
            let a = (c::<T>(3.0) * w).sqrt();
            let u = (x + a) / (c::<T>(2.0) * a);
            if u.val() <= 0.0 {
                zero
            } else if u.val() >= 1.0 {
                one
            } else {
                u
            }
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
            if u.val() <= -1.0 {
                return Some(zero);
            }
            if u.val() >= 1.0 {
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
                _ => {
                    if u.val() < 0.0 {
                        half * (one + u) * (one + u)
                    } else {
                        one - half * (one - u) * (one - u)
                    }
                }
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
            let mut acc = zero;
            let peak = hl.val().floor() as usize;
            for j in 0..100_000usize {
                let jf = j as f64;
                let lp = if hl.val() > 0.0 {
                    -hl + c::<T>(jf) * hl.ln() - c::<T>(jf + 1.0).ln_gamma()
                } else {
                    zero
                };
                let term = lp.exp() * regularized_gamma_p((k + c(2.0 * jf)) * half, y * half);
                acc = acc + term;
                if hl.val() == 0.0 || (j > peak + 5 && term.val() < 1e-17 * acc.val()) {
                    break;
                }
            }
            acc
        }
        K::Kolmogorov => {
            if !pos {
                return Some(zero);
            }
            if x.val() < 1.0 {
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
            if !(x.val() > xm.val()) {
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
            if !(1.0 + xi.val() * u.val() > 0.0) {
                return Some(one);
            }
            one - (-ln1p_over(u, xi)).exp()
        }
        K::LogUniform => {
            let (a, b) = (s[0], s[1]);
            if !(x.val() > a.val()) {
                return Some(zero);
            }
            if !(x.val() < b.val()) {
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
            if !(x.val() > 0.0 && x.val() < 1.0) {
                return Some(if x.val() <= 0.0 { zero } else { one });
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
