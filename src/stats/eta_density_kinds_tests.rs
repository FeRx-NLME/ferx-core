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
        let h = std::f64::consts::FRAC_PI_2 - 1e-7;
        return simpson(
            |u| {
                let cu = u.cos();
                let z = u.tan();
                g(z) * pdf(kind, z, w, s) / (cu * cu)
            },
            -h,
            h,
            400_000,
        );
    }
    let (a, b) = latent_range(kind, w, s);
    // Pull the end points a hair inside a bounded support, where a density that is
    // infinite (arcsine) or a kink sits.
    let eps = (b - a) * 1e-12;
    simpson(|z| g(z) * pdf(kind, z, w, s), a + eps, b - eps, 400_000)
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
        if (mass - 1.0).abs() > worst.0 {
            worst = ((mass - 1.0).abs(), kind.name());
        }
    }
    // Placeholder until measured.
    assert!(
        worst.0 < 1e-6,
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
        let g = dm.abs().max(dv.abs());
        assert!(g.is_finite(), "{kind:?}");
        if g > worst.0 {
            worst = (g, kind.name());
        }
    }
    // Placeholder until measured.
    assert!(
        worst.0 < 1e-6,
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
                if r.v.v.is_finite() {
                    -0.5 * r.v.g[0]
                } else {
                    0.0
                }
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
            d2 + d1 * d1
        });
        let label = format!("{kind:?} information identity");
        assert!(info.is_finite(), "{label}");
        if info.abs() > worst.0 {
            worst = (info.abs(), label);
        }
    }
    eprintln!("worst {:.2e} ({})", worst.0, worst.1);
    // Placeholder until measured.
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
    let n = 200_000;
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
    // Placeholder until measured.
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
    // Placeholder until measured.
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
