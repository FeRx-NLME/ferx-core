use super::*;
use crate::sens::rtape::DualT;
use rand::SeedableRng;

const OMEGA: f64 = 0.12;

/// Every univariate kind with the shapes the tests use, and its support in η (`None`
/// for unbounded) at variance `OMEGA`.
fn cases() -> Vec<(DensityKind, Vec<f64>)> {
    use DensityKind::*;
    vec![
        (Logistic, vec![]),
        (Laplace, vec![]),
        (Cauchy, vec![]),
        (Gumbel, vec![]),
        (NormalInverseGaussian, vec![1.5, 0.3]),
        (PGeneralizedGaussian, vec![2.5]),
        (GeneralizedExtremeValue, vec![0.15]),
        (GeneralizedExtremeValue, vec![0.01]),
        (GeneralizedExtremeValue, vec![-0.2]),
        (Arcsine, vec![]),
        (Uniform, vec![]),
        (Biweight, vec![]),
        (Triweight, vec![]),
        (Epanechnikov, vec![]),
        (Cosine, vec![]),
        (Semicircle, vec![]),
        (SymTriangular, vec![]),
        (Gamma, vec![]),
        (InvGamma, vec![]),
        (LogNormal, vec![]),
        (Exponential, vec![]),
        (Erlang, vec![3.0]),
        (ChiSq, vec![]),
        (Chi, vec![]),
        (Rayleigh, vec![]),
        (Weibull, vec![]),
        (Frechet, vec![]),
        (InverseGaussian, vec![]),
        (BetaPrime, vec![]),
        (NoncentralChiSq, vec![3.0]),
        (Kolmogorov, vec![]),
        (Levy, vec![]),
        (Pareto, vec![]),
        (GeneralizedPareto, vec![]),
        (LogUniform, vec![0.5, 3.0]),
        (Beta, vec![0.3]),
        (LogitNormal, vec![0.3]),
    ]
}

/// The latent interval a kind's mass lives on, for quadrature: its exact support where
/// it is bounded (so Simpson never straddles an edge), `±40` otherwise.
fn latent_range(kind: DensityKind, w: f64, s: &[f64]) -> (f64, f64) {
    use DensityKind::*;
    let pi = std::f64::consts::PI;
    match kind {
        Arcsine => {
            let a = (2.0 * w).sqrt();
            (-a, a)
        }
        Uniform => {
            let a = (3.0 * w).sqrt();
            (-a, a)
        }
        Biweight => ((-(7.0 * w).sqrt()), (7.0 * w).sqrt()),
        Triweight => ((-(9.0 * w).sqrt()), (9.0 * w).sqrt()),
        Epanechnikov => ((-(5.0 * w).sqrt()), (5.0 * w).sqrt()),
        Semicircle => ((-(4.0 * w).sqrt()), (4.0 * w).sqrt()),
        SymTriangular => ((-(6.0 * w).sqrt()), (6.0 * w).sqrt()),
        Cosine => {
            let sig = (w / (1.0 - 8.0 / (pi * pi))).sqrt();
            (-sig, sig)
        }
        Pareto => {
            let a = 1.0 + (1.0 + 1.0 / w).sqrt();
            (((a - 1.0) / a).ln(), 40.0)
        }
        LogUniform => (s[0].ln(), s[1].ln()),
        GeneralizedPareto => {
            let xi = (1.0 - 1.0 / w) / 2.0;
            let sigma = 1.0 - xi;
            if xi < 0.0 {
                (-40.0, (-sigma / xi).ln())
            } else {
                (-40.0, 40.0)
            }
        }
        GeneralizedExtremeValue => {
            let xi = s[0];
            let (vr, m1) = gev_moments(xi);
            let sigma = (w / vr).sqrt();
            let mu = -sigma * m1;
            let edge = mu - sigma / xi;
            if xi > 0.0 {
                (edge, 40.0)
            } else {
                (-40.0, edge)
            }
        }
        // The latent `logit x` has tails `e^{a z}` and `e^{−b z}`: reach `e^{−40}` on each.
        Beta => {
            let mu = s[0];
            let n = mu * (1.0 - mu) / w - 1.0;
            (-40.0 / (mu * n).min(1.0), 40.0 / ((1.0 - mu) * n).min(1.0))
        }
        _ => (-40.0, 40.0),
    }
}

/// Composite Simpson's rule on `[a, b]`, `n` even.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        s += f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    s * h / 3.0
}

/// `p_z(z)`, `0` outside the support.
fn pdf(kind: DensityKind, z: f64, w: f64, s: &[f64]) -> f64 {
    let v = latent_neg2_logpdf(kind, z, w, s);
    if v.is_finite() {
        (-0.5 * v).exp()
    } else {
        0.0
    }
}

/// `E[g(z)]` under the latent density by Simpson, with the t-substitution for the
/// polynomial-tailed Cauchy (`z = tan u`).
fn expect(kind: DensityKind, w: f64, s: &[f64], g: impl Fn(f64) -> f64) -> f64 {
    if kind == DensityKind::Cauchy {
        let h = std::f64::consts::FRAC_PI_2 - 1e-10;
        return simpson(
            |u| {
                let cu = u.cos();
                let z = u.tan();
                g(z) * pdf(kind, z, w, s) / (cu * cu)
            },
            -h,
            h,
            40_000,
        );
    }
    if kind == DensityKind::Arcsine {
        // z = a·sin(π(u − ½)) takes the arcsine's integrable edge singularities out and
        // leaves a flat integrand; the midpoint rule never evaluates at the edges, where
        // `z²` rounds to `a²` and the density reads 0 (Simpson's end points lost h/3 each).
        let a = (2.0 * w).sqrt();
        let pi = std::f64::consts::PI;
        let n = 40_000;
        return (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                let t = pi * (u - 0.5);
                let z = a * t.sin();
                g(z) * pdf(kind, z, w, s) * a * pi * t.cos()
            })
            .sum::<f64>()
            / n as f64;
    }
    if kind == DensityKind::PGeneralizedGaussian {
        // `ln p ∝ −|z|^p`: its curvature carries `|z|^{p−2}`, a √-kink at 0 for p = 2.5,
        // where Simpson converges only as h^{1.5}. `z = s|s|` makes every integrand the
        // identities read smooth through 0 (`dz = 2|s| ds`).
        let r = 40f64.sqrt();
        return simpson(
            |t| {
                let z = t * t.abs();
                g(z) * pdf(kind, z, w, s) * 2.0 * t.abs()
            },
            -r,
            r,
            8_000,
        );
    }
    if kind == DensityKind::GeneralizedPareto {
        let xi = (1.0 - 1.0 / w) / 2.0;
        if xi < -1.0 {
            // Below ξ = −1 the density is infinite (integrably) at its upper end
            // `x_max = σ/|ξ|`, as `(x_max − x)^{−1/ξ − 1}`: `x = x_max·(1 − t^{−ξ})` makes
            // the transformed integrand exactly flat there.
            let x_max = -(1.0 - xi) / xi;
            let m = -xi;
            return simpson(
                |t| {
                    let x = x_max * (1.0 - t.powf(m));
                    if x <= 0.0 {
                        return 0.0;
                    }
                    let z = x.ln();
                    g(z) * pdf(kind, z, w, s) * x_max * m * t.powf(m - 1.0) / x
                },
                1e-9,
                1.0 - 1e-12,
                8_000,
            );
        }
    }
    let (a, b) = latent_range(kind, w, s);
    // Pull the end points a hair inside a bounded support, where a density that is
    // infinite or a kink sits.
    let eps = (b - a) * 1e-12;
    simpson(|z| g(z) * pdf(kind, z, w, s), a + eps, b - eps, 8_000)
}

/// The η mean and variance the module documents, `None` where none exists or the kind
/// is not parameterised by them.
fn documented_moments(kind: DensityKind, w: f64, s: &[f64]) -> Option<(f64, f64)> {
    use DensityKind::*;
    match kind {
        Cauchy | Levy | StudentT | LogitNormal | Kolmogorov | LogUniform => None,
        Exponential => Some((1.0, 1.0)),
        Erlang => Some((1.0, 1.0 / s[0])),
        Rayleigh => Some((1.0, 4.0 / std::f64::consts::PI - 1.0)),
        Beta => Some((s[0], w)),
        Logistic
        | Laplace
        | Gumbel
        | NormalInverseGaussian
        | PGeneralizedGaussian
        | GeneralizedExtremeValue
        | Arcsine
        | Uniform
        | Biweight
        | Triweight
        | Epanechnikov
        | Cosine
        | Semicircle
        | SymTriangular => Some((0.0, w)),
        _ => Some((1.0, w)),
    }
}

/// The mass an `f64` latent cannot hold: below ξ = −1 a GPD piles its mass at
/// `x_max`, and the slice within one ulp of it, `(x_max − x)/x_max < ε`, carries
/// `ε^{−1/ξ}` — 5.6e-5 at the tests' ω — where `x` rounds to `x_max` and the density reads
/// 0. Derived, not fitted: the quadrature may miss exactly this much, and no more.
fn unresolvable_mass(kind: DensityKind, w: f64) -> f64 {
    let xi = (1.0 - 1.0 / w) / 2.0;
    if kind == DensityKind::GeneralizedPareto && xi < -1.0 {
        f64::EPSILON.powf(-1.0 / xi)
    } else {
        0.0
    }
}

/// Every kind's latent density integrates to 1, on its exact support. Simpson, an oracle
/// that shares nothing with the density.
/// Dies under: any wrong normalising constant, or a domain-map Jacobian left out.
#[test]
fn every_kind_integrates_to_one() {
    let mut worst = (0.0f64, "");
    for (kind, s) in cases() {
        let mass = expect(kind, OMEGA, &s, |_| 1.0);
        eprintln!("{:?} {s:?}: mass − 1 = {:.2e}", kind, mass - 1.0);
        assert!(mass.is_finite(), "{kind:?}: {mass}");
        let slack = unresolvable_mass(kind, OMEGA);
        let g = ((mass - 1.0).abs() - slack).max(0.0);
        if g > worst.0 {
            worst = (g, kind.name());
        }
    }
    eprintln!(
        "mass: worst beyond the unresolvable {:.2e} ({})",
        worst.0, worst.1
    );
    // Measured 5.78e-7 (the semicircle, its √ edges under Simpson), beyond the GPD's
    // unresolvable 5.6e-5; 5× headroom.
    assert!(
        worst.0 < 3e-6,
        "worst |mass − 1| = {:.2e} ({})",
        worst.0,
        worst.1
    );
}

/// The random effect `η = m(z)` has the mean and variance the module documents: mean 0
/// and variance ω on ℝ, mean 1 and variance ω on (0, ∞), mean μ and variance ω for the
/// beta, and the fixed moments of the kinds whose ω is unused.
/// Dies under: a swapped shape and scale, a mis-solved implied shape (Weibull, Fréchet,
/// chi), or a variance constant off for a kernel.
#[test]
fn every_kind_has_its_documented_moments() {
    let mut worst = (0.0f64, "");
    for (kind, s) in cases() {
        let Some((m, v)) = documented_moments(kind, OMEGA, &s) else {
            continue;
        };
        let map = kind.map();
        let m1 = expect(kind, OMEGA, &s, |z| map.apply(z));
        let m2 = expect(kind, OMEGA, &s, |z| map.apply(z).powi(2));
        let (dm, dv) = (m1 - m, m2 - m1 * m1 - v);
        eprintln!("{:?} {s:?}: Δmean {dm:.2e}, Δvar {dv:.2e}", kind);
        // The unresolvable mass sits at x_max (`≤ 1.3` here): its first two moments are
        // at most `x_max²` times it.
        let slack = unresolvable_mass(kind, OMEGA) * 4.0;
        let g = (dm.abs().max(dv.abs()) - slack).max(0.0);
        assert!(g.is_finite(), "{kind:?}");
        if g > worst.0 {
            worst = (g, kind.name());
        }
    }
    eprintln!("moments: worst {:.2e} ({})", worst.0, worst.1);
    // Measured 3.68e-7 (GEV ξ = 0.15, its x^{−1/ξ} tail cut at the latent's 40); 5×.
    assert!(
        worst.0 < 2e-6,
        "worst moment gap {:.2e} ({})",
        worst.0,
        worst.1
    );
}

/// The kinds smooth enough for the score identities: their density and its derivative
/// vanish at the edges of the support and no kink sits inside it.
fn smooth(kind: DensityKind) -> bool {
    use DensityKind::*;
    !matches!(
        kind,
        Laplace
            | Arcsine
            | Uniform
            | Epanechnikov
            | Cosine
            | Semicircle
            | SymTriangular
            | Pareto
            | GeneralizedPareto
            | LogUniform
    )
}

/// The score identities, exact for a smooth density: `E[∂ ln p/∂θ] = 0` for θ = z, ω and
/// every shape, and `E[∂² ln p/∂z²] + E[(∂ ln p/∂z)²] = 0`. They hold for the true
/// derivatives of a normalised density, so they check every jet the estimators read
/// (η-gradient, η-Hessian, ∂/∂ω, ∂/∂shape) against the density's own normalisation:
/// no finite difference.
/// Dies under: a jet dropped or mis-signed in any kind (a shape lifted as a constant
/// fails its `E[∂/∂shape] = 0`).
#[test]
fn the_score_identities_hold_for_every_smooth_kind() {
    type H = DualT<1, DualT<1, f64>>;
    let mut worst = (0.0f64, String::new());
    for (kind, s) in cases().into_iter().filter(|(k, _)| smooth(*k)) {
        let n = 2 + s.len();
        // Direction 0: z; 1: ω; 2..: the shapes.
        let jets = |z: f64, dir: usize| -> H {
            let lift = |v: f64, d: usize| -> H {
                DualT {
                    v: DualT {
                        v,
                        g: [if d == dir { 1.0 } else { 0.0 }],
                    },
                    g: [DualT {
                        v: if d == 0 { 1.0 } else { 0.0 },
                        g: [0.0],
                    }],
                }
            };
            let sh: Vec<H> = s.iter().enumerate().map(|(i, &v)| lift(v, 2 + i)).collect();
            latent_neg2_logpdf(kind, lift(z, 0), lift(OMEGA, 1), &sh)
        };
        for dir in 0..n {
            let e = expect(kind, OMEGA, &s, |z| {
                let r = jets(z, dir);
                if !r.v.v.is_finite() {
                    return 0.0;
                }
                // At a GEV's support edge `t = (1 + ξu)^{−1/ξ}` reaches e^700: `−2 ln p`
                // is still finite but its derivative overflows, where `p` itself is
                // exactly 0 in `f64`. Only there is a non-finite jet allowed; anywhere
                // with mass it fails the assertion below.
                if !r.v.g[0].is_finite() {
                    assert!((-0.5 * r.v.v).exp() == 0.0, "{kind:?} jet {dir} at z = {z}");
                    return 0.0;
                }
                -0.5 * r.v.g[0]
            });
            let label = format!("{kind:?} E[∂/∂x{dir}]");
            assert!(e.is_finite(), "{label}");
            if e.abs() > worst.0 {
                worst = (e.abs(), label);
            }
        }
        let info = expect(kind, OMEGA, &s, |z| {
            let r = jets(z, 0);
            if !r.v.v.is_finite() {
                return 0.0;
            }
            let (d1, d2) = (-0.5 * r.v.g[0], -0.5 * r.g[0].g[0]);
            let v = d2 + d1 * d1;
            // As above: an overflowed jet only where `p` is exactly 0.
            if !v.is_finite() {
                assert!(
                    (-0.5 * r.v.v).exp() == 0.0,
                    "{kind:?} information at z = {z}"
                );
                return 0.0;
            }
            v
        });
        let label = format!("{kind:?} information identity");
        assert!(info.is_finite(), "{label}");
        if info.abs() > worst.0 {
            worst = (info.abs(), label);
        }
    }
    eprintln!("worst {:.2e} ({})", worst.0, worst.1);
    // Measured 1.65e-7 (the biweight's information identity, its edge curvature under
    // Simpson); 6× headroom.
    assert!(worst.0 < 1e-6, "worst {:.2e} ({})", worst.0, worst.1);
}

/// The simulator draws the distribution the density scores: 200 000 draws of each kind
/// match its documented mean and variance to a few Monte Carlo SEs. The draws come from
/// `rand_distr` or inverse CDFs, written independently of the densities.
/// Dies under: a sampler of another distribution (a scale and a rate swapped, the
/// wrong implied shape).
#[test]
fn every_sampler_draws_its_density() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1716);
    let n = 100_000;
    let mut worst = (0.0f64, "");
    for (kind, s) in cases() {
        let Some((m, v)) = documented_moments(kind, OMEGA, &s) else {
            continue;
        };
        let xs: Vec<f64> = (0..n)
            .map(|_| sample_eta(kind, OMEGA, &s, &mut rng))
            .collect();
        assert!(
            xs.iter().all(|x| x.is_finite()),
            "{kind:?}: a non-finite draw"
        );
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        // In units of the standard errors (√(v/n) for the mean; the variance's SE
        // ~ v√(2/n) is inflated by heavy tails, so it is checked relatively, loosely).
        let zm = (mean - m).abs() / (v / n as f64).sqrt();
        let rv = (var - v).abs() / v;
        eprintln!("{:?}: mean z {zm:.2}, var rel {rv:.3}", kind);
        let g = zm.max(rv * 100.0);
        if g > worst.0 {
            worst = (g, kind.name());
        }
    }
    eprintln!("sampler: worst {:.2} ({})", worst.0, worst.1);
    // Measured 3.27 (Pareto's mean, in SEs; seed 1716) over ~40 kinds × 2 statistics, as
    // the maximum of that many draws should be; 6 SE is still a sampler of the wrong
    // distribution's floor.
    assert!(worst.0 < 6.0, "worst {:.2} ({})", worst.0, worst.1);
}

/// `atan_g` is libm's `atan` to rounding across both reduction branches, and `ln K₁` is
/// the reference at three arguments (Abramowitz & Stegun table 9.8 values,
/// `K₁(0.1) = 9.853844780870606`, `K₁(1) = 0.6019072301972346`,
/// `K₁(10) = 1.864877345382558e-5`).
#[test]
fn the_special_functions_are_the_references() {
    let mut worst = 0.0f64;
    for x in [
        -37.0, -1.5, -1.0, -0.3, 0.0, 1e-8, 0.2, 0.999, 1.0, 4.0, 1e6,
    ] {
        worst = worst.max((atan_g::<f64>(x) - x.atan()).abs());
    }
    eprintln!("atan: worst {worst:.2e}");
    // Measured 2.22e-16, one ulp at 1; bound 4 ulp.
    assert!(worst < 1e-15, "atan {worst:.2e}");
    for (x, k1) in [
        (0.1, 9.853_844_780_870_606),
        (1.0, 0.601_907_230_197_234_6),
        (10.0, 1.864_877_345_382_558e-5),
    ] {
        let got = ln_bessel_k1::<f64>(x).exp();
        assert!((got - k1).abs() / k1 < 1e-13, "K1({x}) = {got}, want {k1}");
    }
}

/// The noncentral χ² density's two forms agree where `ln_ncx2_pdf` switches between
/// them: the Poisson-mixture series and the Bessel asymptotic, over `√(λy)` from just
/// below to 3× the switch, for several `k` and λ (`k = 3`, ν = ½, makes the asymptotic
/// exact; the others exercise its terms). Values and their `k`/λ derivatives: a wrong
/// sign or power in either form shows here.
/// Dies under: a dropped term, a wrong `(y/λ)` exponent, or a series that stops short.
#[test]
fn the_noncentral_chisq_series_and_asymptotic_agree() {
    type D = DualT<2, f64>;
    let mut worst = 0.0f64;
    for (k, lam) in [(1.0, 2.0), (3.0, 50.0), (7.5, 10.0), (0.6, 400.0)] {
        for x in [800.0, NCX2_ASYMPTOTIC_FROM, 1500.0, 3000.0] {
            let y = x * x / lam;
            let kd = D {
                v: k,
                g: [1.0, 0.0],
            };
            let ld = D {
                v: lam,
                g: [0.0, 1.0],
            };
            let yd = D {
                v: y,
                g: [0.0, 0.0],
            };
            let a = ln_ncx2_series(yd, kd, ld);
            let b = ln_ncx2_asymptotic(yd, kd, ld);
            for (p, q) in [(a.v, b.v), (a.g[0], b.g[0]), (a.g[1], b.g[1])] {
                assert!(p.is_finite() && q.is_finite(), "k {k} λ {lam} x {x}");
                worst = worst.max((p - q).abs() / q.abs().max(1.0));
            }
        }
    }
    eprintln!("ncx2 series vs asymptotic: worst relative {worst:.2e}");
    // Measured 2.53e-13 (the series' rounding over ~10³ terms); 40× headroom.
    assert!(worst < 1e-11, "{worst:e}");
}

/// The Kolmogorov density far in either tail. Left: for x ≪ 1 its theta series is one term
/// to rounding (the next is `e^{−π²/x²}` smaller), so `ln f = ln √(2π) − 2 ln x −
/// π²/(8x²) + ln(π²/(4x²) − 1)`. Summed in linear space the term was subnormal by
/// x ≈ 0.04, its η-gradient overflowing, and 0 below, the latent reading `+∞`.
/// Dies under: the leading exponent left inside the sum.
#[test]
fn the_kolmogorov_left_tail_stays_finite() {
    type D = DualT<1, f64>;
    let pi = std::f64::consts::PI;
    let mut worst = 0.0f64;
    for x in [0.04f64, 0.02, 0.01] {
        let z = x.ln();
        let want_lf = (2.0 * pi).sqrt().ln() - 2.0 * x.ln() - pi * pi / (8.0 * x * x)
            + (pi * pi / (4.0 * x * x) - 1.0).ln();
        // −2 ln p_z = −2 (ln f + z).
        let want = -2.0 * (want_lf + z);
        let got = latent_neg2_logpdf::<D>(
            DensityKind::Kolmogorov,
            D { v: z, g: [1.0] },
            D { v: OMEGA, g: [0.0] },
            &[],
        );
        assert!(got.v.is_finite() && got.g[0].is_finite(), "x {x}: {got:?}");
        worst = worst.max((got.v - want).abs() / want.abs());
    }
    // The right tail, the same way: `ln f = ln 8x − 2x²` to rounding for x ≫ 1.
    for x in [10.0f64, 19.0, 30.0] {
        let z = x.ln();
        let want = -2.0 * ((8.0 * x).ln() - 2.0 * x * x + z);
        let got = latent_neg2_logpdf::<D>(
            DensityKind::Kolmogorov,
            D { v: z, g: [1.0] },
            D { v: OMEGA, g: [0.0] },
            &[],
        );
        assert!(got.v.is_finite() && got.g[0].is_finite(), "x {x}: {got:?}");
        worst = worst.max((got.v - want).abs() / want.abs());
    }
    eprintln!("kolmogorov tail: worst relative {worst:.2e}");
    // Measured 5.91e-16 on the left tail (rounding); 100× headroom.
    assert!(worst < 1e-13, "{worst:e}");
}
