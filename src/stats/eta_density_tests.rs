use super::*;
use crate::sens::jet::DualT;

/// Euler–Mascheroni γ, for the closed-form digamma values. Written out: stable `std`
/// has no `EULER_GAMMA` constant yet (`f64::consts::EGAMMA` is unstable).
#[allow(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const LN_2PI: f64 = 1.837_877_066_409_345_5;

fn group(kind: DensityKind, etas: Vec<usize>, shape: Vec<usize>) -> EtaDensity {
    EtaDensity {
        kind,
        etas,
        shape_thetas: shape,
        trunc: None,
        margins: vec![],
        log_anchors: Vec::new(),
    }
}

/// `−2 ln p_z(z)` of one univariate group: the Gaussian part plus the correction, with the
/// `ln 2π` the ferx convention drops added back, so that `exp(−½·this)` is a density.
fn neg2_log_pdf(d: &EtaDensity, z: f64, omega: f64, theta: &[f64]) -> f64 {
    let oi = DMatrix::from_element(1, 1, 1.0 / omega);
    z * z / omega + omega.ln() + LN_2PI + correction(std::slice::from_ref(d), &[z], &oi, theta)
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

/// `∫ g(m(z)) p_z(z) dz` for one univariate kind by Simpson on a wide latent interval
/// (every kind but the t decays at least exponentially in `z`).
fn latent_expectation(
    d: &EtaDensity, omega: f64, theta: &[f64], g: impl Fn(f64, f64) -> f64,
) -> f64 {
    let m = |z: f64| match d.kind.map() {
        DomainMap::Identity => z,
        DomainMap::Exp => z.exp(),
        DomainMap::InvLogit => 1.0 / (1.0 + (-z).exp()),
    };
    simpson(
        |z| {
            let v = neg2_log_pdf(d, z, omega, theta);
            if v.is_finite() {
                g(m(z), z) * (-0.5 * v).exp()
            } else {
                0.0
            }
        },
        -40.0,
        40.0,
        40_000,
    )
}

/// The univariate kinds, each with its shape θ (`[μ]` or none) and a variance ω. The
/// beta's ω is small enough (`n = 9.5`) that its latent tails (`∝ e^{−a|z|}`,
/// `a = μn`) sit far inside the ±40 integration range; at ω = 0.12 its `a` is 0.23 and
/// ±40 leaves 1e-4 of the mass out.
fn univariate_kinds() -> Vec<(EtaDensity, Vec<f64>, f64)> {
    vec![
        (group(DensityKind::Logistic, vec![0], vec![]), vec![], 0.12),
        (group(DensityKind::Gamma, vec![0], vec![]), vec![], 0.12),
        (group(DensityKind::InvGamma, vec![0], vec![]), vec![], 0.12),
        (group(DensityKind::LogNormalMean, vec![0], vec![]), vec![], 0.12),
        (group(DensityKind::Beta, vec![0], vec![0]), vec![0.3], 0.02),
        (group(DensityKind::LogitNormal, vec![0], vec![0]), vec![0.3], 0.12),
    ]
}

/// Each latent density integrates to 1, the t over the `η = s·tan u` substitution so its
/// polynomial tails sit on a finite interval. Simpson, an oracle that shares nothing with
/// the correction.
/// Dies under: any wrong normalising constant (a lnΓ term, `n ln(ν/2)`, `ln s`, the
/// Jacobian of the domain map).
#[test]
fn each_density_integrates_to_one() {
    let omega = 0.12;
    let t = group(DensityKind::StudentT, vec![0], vec![0]);
    let s = omega.sqrt();
    let mut worst = (simpson(
        |u| {
            let c = u.cos();
            if c.abs() < 1e-300 {
                return 0.0;
            }
            (-0.5 * neg2_log_pdf(&t, s * u.tan(), omega, &[5.0])).exp() * s / (c * c)
        },
        // Stop short of ±π/2: past `|u| = π/2 − 1e-4` (|η| > 3.5e3) the t's tail mass is
        // ~1e-17 at ν = 5, while `z²/ω − q` cancels catastrophically in f64 there.
        -std::f64::consts::FRAC_PI_2 + 1e-4,
        std::f64::consts::FRAC_PI_2 - 1e-4,
        20_000,
    ) - 1.0)
        .abs();
    for (d, th, omega) in univariate_kinds() {
        let mass = latent_expectation(&d, omega, &th, |_, _| 1.0);
        eprintln!("{:?}: mass − 1 = {:.3e}", d.kind, mass - 1.0);
        assert!(mass.is_finite(), "{:?}: mass {mass}", d.kind);
        worst = worst.max((mass - 1.0).abs());
    }
    // Measured 4.5e-14 (Simpson on 2e4 / 4e5 panels); 10× headroom.
    assert!(worst < 5e-13, "worst |mass − 1| = {worst:.3e}");
}

/// The random effect the model reads, `η = m(z)`, has the mean and variance the module
/// documents: mean 1 and variance ω for `gamma`, `invgamma` and `lognormal_mean`, mean μ and
/// variance ω for `beta`, variance ω for `logistic` (on ℝ, mean 0). The logit-normal is
/// parameterised on the latent scale (`z ~ N(logit μ, ω)`), so its latent moments are
/// checked instead.
/// Dies under: a swapped shape/scale, `k = ω` instead of `1/ω`, the beta's `n` off by one,
/// or a logistic scale missing its `√3/π`.
#[test]
fn each_density_has_its_documented_moments() {
    let mut worst = 0.0f64;
    for (d, th, omega) in univariate_kinds() {
        let (dmean, dvar) = if d.kind == DensityKind::LogitNormal {
            // Latent moments, read off `z` itself.
            let m1 = latent_expectation(&d, omega, &th, |_, z| z);
            let m2 = latent_expectation(&d, omega, &th, |_, z| z * z);
            (m1 - (th[0] / (1.0 - th[0])).ln(), m2 - m1 * m1 - omega)
        } else {
            let m1 = latent_expectation(&d, omega, &th, |e, _| e);
            let m2 = latent_expectation(&d, omega, &th, |e, _| e * e);
            let want_mean = match d.kind {
                DensityKind::Logistic => 0.0,
                DensityKind::Beta => th[0],
                _ => 1.0,
            };
            (m1 - want_mean, m2 - m1 * m1 - omega)
        };
        eprintln!("{:?}: Δmean {dmean:.3e}, Δvar {dvar:.3e}", d.kind);
        assert!(dmean.is_finite() && dvar.is_finite(), "{:?}", d.kind);
        worst = worst.max(dmean.abs()).max(dvar.abs());
    }
    // Measured 3.0e-14 (Simpson on 4e5 panels); 10× headroom.
    assert!(worst < 3e-13, "worst moment gap {worst:.3e}");
}

/// The Gaussian reductions: a logit-normal at μ = ½ (location 0) is the Gaussian bit for
/// bit, and a large ν approaches it at the `O(1/ν)` rate.
/// Dies under: a wrong t normalising constant (which would not vanish as ν grows).
#[test]
fn the_densities_reduce_to_the_gaussian() {
    let oi = DMatrix::from_row_slice(2, 2, &[12.0, -3.0, -3.0, 20.0]);
    let eta = [0.31, -0.22];
    let ln = group(DensityKind::LogitNormal, vec![0], vec![0]);
    let oi1 = DMatrix::from_element(1, 1, 8.0);
    assert_eq!(correction(&[ln], &[0.4], &oi1, &[0.5]), 0.0);
    let mut last = f64::INFINITY;
    for nu in [1e3, 1e5, 1e7] {
        let t = group(DensityKind::StudentT, vec![0, 1], vec![0]);
        let c = correction(&[t], &eta, &oi, &[nu]).abs();
        assert!(c < last, "ν = {nu}: |c| = {c:.3e} shrinks");
        last = c;
    }
    // O(1/ν) with q ≈ 1.6: about 4e-7 at ν = 1e7; 10× headroom.
    assert!(last < 4e-6, "ν = 1e7: |c| = {last:.3e}");
}

/// The bivariate t's `−2 ln p` matches the textbook multivariate-t log-pdf written out
/// independently: ν = 4, n = 2, where `lnΓ(2) = 0` and `lnΓ(3) = ln 2` are exact, so the
/// reference uses no lnΓ approximation.
/// Dies under: the `(ν + n)` exponent or `n ln(ν/2)` term wrong for n > 1.
#[test]
fn the_block_t_is_the_multivariate_t() {
    let omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.05]);
    let oi = omega.clone().try_inverse().unwrap();
    let ld = omega.determinant().ln();
    let eta = [0.4, -0.25];
    let q = oi[(0, 0)] * eta[0] * eta[0]
        + 2.0 * oi[(0, 1)] * eta[0] * eta[1]
        + oi[(1, 1)] * eta[1] * eta[1];
    let nu: f64 = 4.0;
    let n = 2.0;
    // −2 ln p = (ν+n) ln(1+q/ν) + ln|Ω| + n ln(νπ) + 2 lnΓ(ν/2) − 2 lnΓ((ν+n)/2).
    let want = (nu + n) * (1.0 + q / nu).ln() + ld + n * (nu * std::f64::consts::PI).ln()
        - 2.0 * std::f64::consts::LN_2;
    let t = group(DensityKind::StudentT, vec![0, 1], vec![0]);
    let got = q + ld + n * LN_2PI + correction(&[t], &eta, &oi, &[nu]);
    // Measured 1.8e-15 (the Lanczos lnΓ at 2 and 3); 10× headroom.
    assert!((got - want).abs() < 2e-14, "got {got}, want {want}");
}

/// The η-gradient and η-Hessian from the hyper-dual seeds match closed forms, on one
/// 4-η model holding a t, a Gaussian η, a gamma and a beta:
/// - t, with `w = Ω⁻¹η`: `∂c/∂η = 2(n − q)/(ν + q)·w`,
///   `∂²c/∂η² = 2(n − q)/(ν + q)·Ω⁻¹ − 4(ν + n)/(ν + q)²·w²`;
/// - gamma: `∂c/∂z = −2(k − e^z/ω) − 2z/ω`, `∂²c/∂z² = 2e^z/ω − 2/ω`;
/// - beta: `∂c/∂z = 2(bσ(z) − aσ(−z)) − 2z/ω`, `∂²c/∂z² = 2nσ(z)σ(−z) − 2/ω`.
///
/// The Gaussian η must carry no correction, and no two univariate groups couple.
/// Dies under: a seed on the wrong lane, a Hessian entry left unsymmetrised, or a group's
/// gradient written outside its own ETAs.
#[test]
fn the_eta_jets_are_the_closed_forms() {
    let om = [0.09, 0.04, 0.15, 0.03];
    let omega = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&om));
    let a = omega.clone().try_inverse().unwrap();
    let eta = [0.35, -0.2, 0.25, -0.3];
    let theta = [6.0, 0.4];
    let ds = [
        group(DensityKind::StudentT, vec![0], vec![0]),
        group(DensityKind::Gamma, vec![2], vec![]),
        group(DensityKind::Beta, vec![3], vec![1]),
    ];
    let (v, g, h) = correction_jets(&ds, &eta, &a, &theta, true);
    let h = h.unwrap();
    assert_eq!(v, correction(&ds, &eta, &a, &theta), "value is the f64 one");
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());

    let mut want_g = [0.0; 4];
    let mut want_h = [0.0; 4];
    let (nu, n) = (theta[0], 1.0);
    let q = a[(0, 0)] * eta[0] * eta[0];
    let w = a[(0, 0)] * eta[0];
    want_g[0] = 2.0 * (n - q) / (nu + q) * w;
    want_h[0] = 2.0 * (n - q) / (nu + q) * a[(0, 0)] - 4.0 * (nu + n) / (nu + q).powi(2) * w * w;
    let (z, w2) = (eta[2], om[2]);
    let k = 1.0 / w2;
    want_g[2] = -2.0 * (k - z.exp() / w2) - 2.0 * z / w2;
    want_h[2] = 2.0 * z.exp() / w2 - 2.0 / w2;
    let (z, w3, mu) = (eta[3], om[3], theta[1]);
    let nb = mu * (1.0 - mu) / w3 - 1.0;
    let (pa, pb) = (mu * nb, (1.0 - mu) * nb);
    want_g[3] = 2.0 * (pb * sig(z) - pa * sig(-z)) - 2.0 * z / w3;
    want_h[3] = 2.0 * nb * sig(z) * sig(-z) - 2.0 / w3;

    let mut worst = 0.0f64;
    for i in 0..4 {
        assert!(
            g[i].is_finite()
                && h[(i, i)].is_finite()
                && want_g[i].is_finite()
                && want_h[i].is_finite(),
            "η{i}: {} {}",
            g[i],
            h[(i, i)]
        );
        worst = worst.max((g[i] - want_g[i]).abs() / (1.0 + want_g[i].abs()));
        worst = worst.max((h[(i, i)] - want_h[i]).abs() / (1.0 + want_h[i].abs()));
        for j in 0..4 {
            if i != j {
                assert_eq!(h[(i, j)], 0.0, "univariate groups do not couple: h[{i},{j}]");
            }
        }
    }
    assert_eq!(g[1], 0.0, "the Gaussian η has no correction");
    // Measured 9.2e-16; 10× headroom.
    assert!(worst < 1e-14, "worst relative gap {worst:.3e}");
}

/// `∂c/∂ν` through the dual lnΓ matches `ln(1 + q/ν) − (ν + n)q/(ν(ν + q)) + n/ν +
/// ψ(ν/2) − ψ((ν + n)/2)` at ν = 4, n = 1, where `ψ(2) = 1 − γ` and
/// `ψ(5/2) = 8/3 − γ − 2 ln 2` are exact.
/// Dies under: the lnΓ pair dropped from the jets (constants lifted as `f64`).
#[test]
fn the_shape_derivative_is_the_digamma_closed_form() {
    let oi = DMatrix::from_element(1, 1, 1.0 / 0.09);
    let eta = 0.3;
    let q = oi[(0, 0)] * eta * eta;
    let (nu, n) = (4.0f64, 1.0);
    let d = DualT::<1, f64> { v: nu, g: [1.0] };
    let c = correction_g(
        &[group(DensityKind::StudentT, vec![0], vec![0])],
        &[DualT::<1, f64>::constant(eta)],
        |a, b| DualT::constant(oi[(a, b)]),
        |_| d,
    );
    let psi2 = 1.0 - EULER_GAMMA;
    let psi52 = 8.0 / 3.0 - EULER_GAMMA - 2.0 * std::f64::consts::LN_2;
    let want = (1.0 + q / nu).ln() - (nu + n) * q / (nu * (nu + q)) + n / nu + psi2 - psi52;
    // Measured 1.1e-15 (the Lanczos digamma); 10× headroom.
    assert!((c.g[0] - want).abs() < 1e-14, "∂c/∂ν = {}, want {want}", c.g[0]);
}

/// Over a dual `T` the jets carry `∂/∂θ` of the gradient: `∂(∂c/∂η)/∂ν` from the
/// kernel's type matches the closed form `∂/∂ν [2(n − q)/(ν + q)·w] = −2(n − q)/(ν + q)²·w`.
/// Dies under: `theta` read as a constant in the seeds (`lift(θ)` dropping its jet).
#[test]
fn the_kernel_jets_carry_the_shape_derivative() {
    let oi = 1.0 / 0.09;
    let eta = 0.3;
    let nu = 6.0;
    let d = DualT::<1, f64> { v: nu, g: [1.0] };
    let (_, g, _) = correction_jets_g(
        &[group(DensityKind::StudentT, vec![0], vec![0])],
        &[DualT::<1, f64>::constant(eta)],
        |_, _| DualT::constant(oi),
        |_| d,
        false,
    );
    let (q, w, n) = (oi * eta * eta, oi * eta, 1.0);
    let want = -2.0 * (n - q) / (nu + q).powi(2) * w;
    assert!((g[0].g[0] - want).abs() < 1e-13, "{} vs {want}", g[0].g[0]);
}

/// A beta variance past `μ(1 − μ)` has no beta distribution: the correction is
/// `+∞`, so every consumer rejects the point; just below it is finite.
#[test]
fn a_beta_variance_past_its_bound_is_infinite() {
    let b = [group(DensityKind::Beta, vec![0], vec![0])];
    let at = |omega: f64| correction(&b, &[0.1], &DMatrix::from_element(1, 1, 1.0 / omega), &[0.3]);
    assert_eq!(at(0.22), f64::INFINITY, "ω just past μ(1 − μ) = 0.21");
    assert_eq!(at(0.3), f64::INFINITY);
    assert!(at(0.2).is_finite());
}

/// `truncated(KIND, lo, hi)`: the truncated latent density integrates to 1 over the
/// unconstrained latent, which checks the CDF mass and the map's Jacobian. The random
/// effect's mean is the base density's conditional mean on `(lo, hi)`, computed by
/// direct quadrature of the η-density, an oracle that shares nothing with the CDFs.
/// Covers both bounds, a lower bound only and an upper bound only, on ℝ, (0, ∞) and
/// (0, 1) kinds, the t and the Cauchy (incomplete beta, atan).
/// Dies under: a CDF wrong in any covered kind, the mass left out, or a map Jacobian
/// off.
#[test]
fn a_truncated_density_is_the_renormalised_kind() {
    use super::kinds::eta_log_pdf;
    let omega = 0.12;
    let oi = DMatrix::from_element(1, 1, 1.0 / omega);
    let cases: Vec<(DensityKind, Vec<f64>, Bound, Bound)> = vec![
        (DensityKind::Gamma, vec![], Bound::Value(0.5), Bound::Value(2.0)),
        (DensityKind::StudentT, vec![5.0], Bound::Value(-0.2), Bound::Open),
        (DensityKind::Cauchy, vec![], Bound::Value(-1.0), Bound::Value(1.0)),
        (DensityKind::Beta, vec![0.3], Bound::Value(0.1), Bound::Value(0.6)),
        (DensityKind::Weibull, vec![], Bound::Open, Bound::Value(1.5)),
        (DensityKind::Logistic, vec![], Bound::Value(-0.3), Bound::Value(0.8)),
    ];
    let mut worst = 0.0f64;
    for (kind, sh, lo, hi) in cases {
        let d = EtaDensity {
            kind,
            etas: vec![0],
            shape_thetas: (0..sh.len()).collect(),
            trunc: Some(Truncation { lo, hi }),
            margins: vec![],
            log_anchors: Vec::new(),
        };
        let val = |b: Bound, open: f64| match b {
            Bound::Value(v) => v,
            _ => open,
        };
        let (a, b) = (val(lo, f64::NAN), val(hi, f64::NAN));
        let read = |z: f64| match (lo, hi) {
            (Bound::Value(a), Bound::Value(b)) => a + (b - a) / (1.0 + (-z).exp()),
            (Bound::Value(a), _) => a + z.exp(),
            (_, Bound::Value(b)) => b - z.exp(),
            _ => z,
        };
        let lat = |z: f64| {
            let v = z * z / omega
                + omega.ln()
                + LN_2PI
                + correction(std::slice::from_ref(&d), &[z], &oi, &sh);
            if v.is_finite() {
                (-0.5 * v).exp()
            } else {
                0.0
            }
        };
        let mass = simpson(lat, -40.0, 40.0, 400_000);
        let mean = simpson(|z| read(z) * lat(z), -40.0, 40.0, 400_000);
        // The oracle: ∫ x p(x) / ∫ p(x) on (lo, hi), in η, by Simpson.
        let p = |x: f64| eta_log_pdf::<f64>(kind, x, omega, &sh).exp();
        let (xa, xb) = (
            if a.is_nan() { -60.0 } else { a },
            if b.is_nan() {
                if kind == DensityKind::StudentT {
                    400.0
                } else {
                    60.0
                }
            } else {
                b
            },
        );
        let (xa, xb) = (xa.max(if kind.map() == DomainMap::Identity { -1e9 } else { 1e-12 }), xb);
        let den = simpson(p, xa, xb, 200_000);
        let want = simpson(|x| x * p(x), xa, xb, 200_000) / den;
        eprintln!("{kind:?}: mass − 1 {:.2e}, mean {mean:.6} vs {want:.6}", mass - 1.0);
        assert!(mass.is_finite() && mean.is_finite() && want.is_finite(), "{kind:?}");
        worst = worst.max((mass - 1.0).abs()).max((mean - want).abs());
    }
    eprintln!("truncated: worst {worst:.2e}");
    // Measured 3.99e-12 (the 200 000-panel Simpson reference); 25× headroom.
    assert!(worst < 1e-10, "worst {worst:.2e}");
}

/// A Gaussian copula whose margins are all `normal` is the multivariate normal, exactly:
/// its correction vanishes for every latent and correlation.
/// Dies under: a wrong copula term (`ln|R|`, `R⁻¹ − I`) or the margins' variances not
/// read off Ω's diagonal.
#[test]
fn a_gaussian_copula_is_the_multivariate_normal() {
    let omega =
        DMatrix::from_row_slice(3, 3, &[0.09, 0.02, -0.01, 0.02, 0.05, 0.01, -0.01, 0.01, 0.04]);
    let oi = omega.clone().try_inverse().unwrap();
    let d = EtaDensity {
        kind: DensityKind::Copula,
        etas: vec![0, 1, 2],
        shape_thetas: vec![],
        trunc: None,
        margins: vec![DensityKind::Normal; 3],
        log_anchors: Vec::new(),
    };
    let mut worst = 0.0f64;
    for z in [[0.1, -0.2, 0.3], [-0.5, 0.4, 0.05], [0.0, 0.0, 0.0]] {
        let c = correction(std::slice::from_ref(&d), &z, &oi, &[]);
        assert!(c.is_finite(), "{z:?}: {c}");
        worst = worst.max(c.abs());
    }
    eprintln!("copula of normals vs MVN: worst {worst:.2e}");
    // Measured 4.44e-15 (rounding: the correction is identically 0); 20× headroom.
    assert!(worst < 1e-13, "worst {worst:.2e}");
}

/// A copula of a gamma and a beta margin integrates to 1 over the latent plane (2-D
/// Simpson), and each margin keeps its own documented mean: the copula couples the
/// margins without changing them.
/// Dies under: the copula density's Jacobian (`Σ −2 ln pₖ`) or its normaliser wrong.
#[test]
fn a_copula_integrates_to_one_and_keeps_its_margins() {
    let omega = DMatrix::from_row_slice(2, 2, &[0.12, 0.02, 0.02, 0.02]);
    let oi = omega.clone().try_inverse().unwrap();
    let ld = omega.determinant().ln();
    let d = EtaDensity {
        kind: DensityKind::Copula,
        etas: vec![0, 1],
        shape_thetas: vec![0],
        trunc: None,
        margins: vec![DensityKind::Gamma, DensityKind::Beta],
        log_anchors: Vec::new(),
    };
    let theta = [0.3];
    let p = |z0: f64, z1: f64| {
        let z = [z0, z1];
        let q = oi[(0, 0)] * z0 * z0 + 2.0 * oi[(0, 1)] * z0 * z1 + oi[(1, 1)] * z1 * z1;
        let v = q + ld + 2.0 * LN_2PI + correction(std::slice::from_ref(&d), &z, &oi, &theta);
        if v.is_finite() {
            (-0.5 * v).exp()
        } else {
            0.0
        }
    };
    let n = 800;
    let (a, b) = (-12.0, 12.0);
    let h = (b - a) / n as f64;
    let wgt = |i: usize| {
        if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        }
    };
    let (mut mass, mut m0, mut m1) = (0.0, 0.0, 0.0);
    for i in 0..=n {
        for j in 0..=n {
            let (z0, z1) = (a + i as f64 * h, a + j as f64 * h);
            let w = wgt(i) * wgt(j) * p(z0, z1);
            mass += w;
            m0 += w * z0.exp();
            m1 += w * 1.0 / (1.0 + (-z1).exp());
        }
    }
    let k = h * h / 9.0;
    let (mass, m0, m1) = (mass * k, m0 * k / (mass * k), m1 * k / (mass * k));
    eprintln!("copula: mass − 1 {:.2e}, gamma mean {m0:.8}, beta mean {m1:.8}", mass - 1.0);
    // Measured: mass − 1 = −1.44e-13, both margins' means at 1e-8 printed precision;
    // bounds with ≥ 100× headroom on the mass.
    assert!((mass - 1.0).abs() < 1e-11 && (m0 - 1.0).abs() < 1e-8 && (m1 - 0.3).abs() < 1e-8);
}

/// `lognormal_mean` on a block is the multivariate log-normal of mean 1 and covariance Ω:
/// the sampler's draws have that mean and covariance, and its latent density at a point
/// is the textbook `N(ln η; −diag(S)/2, S)` with `S = ln(1 + Ω)`.
/// Dies under: `S` taken as Ω, or the mean shift dropped.
#[test]
fn the_block_lognormal_is_the_multivariate_lognormal() {
    use rand::SeedableRng;
    let omega: DMatrix<f64> = DMatrix::from_row_slice(2, 2, &[0.09, 0.03, 0.03, 0.05]);
    let oi = omega.clone().try_inverse().unwrap();
    let d = EtaDensity {
        kind: DensityKind::LogNormalMean,
        etas: vec![0, 1],
        shape_thetas: vec![],
        trunc: None,
        margins: vec![],
        log_anchors: Vec::new(),
    };
    let s = DMatrix::from_fn(2, 2, |i, j| f64::ln_1p(omega[(i, j)]));
    let si = s.clone().try_inverse().unwrap();
    let z = [0.1, -0.2];
    let r = nalgebra::DVector::from_vec(vec![z[0] + s[(0, 0)] / 2.0, z[1] + s[(1, 1)] / 2.0]);
    let want = (r.transpose() * &si * &r)[(0, 0)] + s.determinant().ln() + 2.0 * LN_2PI;
    let q = oi[(0, 0)] * z[0] * z[0] + 2.0 * oi[(0, 1)] * z[0] * z[1] + oi[(1, 1)] * z[1] * z[1];
    let got = q
        + omega.determinant().ln()
        + 2.0 * LN_2PI
        + correction(std::slice::from_ref(&d), &z, &oi, &[]);
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    let mut rng = rand::rngs::StdRng::seed_from_u64(1716);
    let chol = omega.clone().cholesky().unwrap().l();
    let n = 100_000;
    let (mut m, mut c) = ([0.0f64; 2], [[0.0f64; 2]; 2]);
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        let zz = nalgebra::DVector::from_fn(2, |_, _| {
            rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng)
        });
        let mut e: Vec<f64> = (&chol * zz).iter().copied().collect();
        redraw_latents(std::slice::from_ref(&d), &omega, &[], &mut e, &mut rng);
        let x = [e[0].exp(), e[1].exp()];
        m[0] += x[0];
        m[1] += x[1];
        xs.push(x);
    }
    m = [m[0] / n as f64, m[1] / n as f64];
    for x in &xs {
        for i in 0..2 {
            for j in 0..2 {
                c[i][j] += (x[i] - m[i]) * (x[j] - m[j]) / (n - 1) as f64;
            }
        }
    }
    eprintln!("mvlognormal: mean {m:?}, cov {c:?}");
    // 100 000 draws: the mean's SE is ≈ 1e-3, the covariance's ≈ 5e-4.
    assert!((m[0] - 1.0).abs() < 5e-3 && (m[1] - 1.0).abs() < 5e-3);
    assert!((c[0][1] - 0.03).abs() < 3e-3 && (c[0][0] - 0.09).abs() < 5e-3);
}

/// The reported percentiles of a density's effect (#1716) against closed forms: a
/// normal, a log-normal (mean 1, variance ω: `ln η ~ N(−s/2, s)`, `s = ln(1 + ω)`), a
/// Cauchy, whose 95th percentile at `ω = 4` (`2·tan(0.45π) = 12.6`) lies outside the
/// first ±40·√ω bracket only after the CDF's slow tail is searched — the widening is
/// what reaches it — and a normal truncated to `[0, ∞)`, the half-normal.
#[test]
fn effect_quantiles_are_the_closed_forms() {
    use crate::stats::special::normal_inv_cdf as zq;
    let dist = |kind: &str, trunc| crate::types::EtaDistribution {
        form: "density".into(),
        kind: kind.into(),
        shape_thetas: vec![],
        truncation: trunc,
        truncation_thetas: (None, None),
    };
    let p = [0.05, 0.5, 0.95];
    let mut worst = 0.0f64;
    let mut check = |got: Option<[f64; 3]>, want: [f64; 3]| {
        let got = got.expect("a distribution");
        for (g, w) in got.iter().zip(want) {
            assert!(g.is_finite() && w.is_finite(), "{got:?} vs {want:?}");
            worst = worst.max((g - w).abs() / w.abs().max(1.0));
        }
    };
    let w = 0.09;
    check(effect_quantiles(&dist("normal", None), w, &[]), p.map(|p| w.sqrt() * zq(p)));
    check(effect_quantiles(&dist("lognormal", None), w, &[]), p.map(|p| (w.sqrt() * zq(p)).exp()));
    let s = (1.0f64 + w).ln();
    check(
        effect_quantiles(&dist("lognormal_mean", None), w, &[]),
        p.map(|p| (-s / 2.0 + s.sqrt() * zq(p)).exp()),
    );
    check(
        effect_quantiles(&dist("cauchy", None), 4.0, &[]),
        p.map(|p| 2.0 * (std::f64::consts::PI * (p - 0.5)).tan()),
    );
    check(
        effect_quantiles(&dist("normal", Some((Some(0.0), None))), w, &[]),
        p.map(|p| w.sqrt() * zq(0.5 + 0.5 * p)),
    );
    // Measured: 4.6e-9 worst (normal_inv_cdf's own accuracy); bound with 20× headroom.
    eprintln!("effect_quantiles worst relative error {worst:e}");
    assert!(worst < 1e-7, "{worst:e}");
    // No distribution: an unknown kind, a non-positive ω.
    assert!(effect_quantiles(&dist("nope", None), w, &[]).is_none());
    assert!(effect_quantiles(&dist("normal", None), 0.0, &[]).is_none());
}

/// Every kind, as a parsed model, against every method (#2669 review, findings 1 and 12):
/// `fit_refusal` refuses exactly the kinds with a kink or an edge, and exactly under the
/// methods whose reported objective reads the prior's curvature: FOCE, FOCEI, the GN pair,
/// Laplace, VI's `vi_final_ofv = laplace` and SAEM's `saem_ofv = focei`. IMP, SAEM's own
/// objectives, Bayes and VI's default run every kind. The refusal names the ETA.
/// Dies under: any kind dropped from `has_kink_or_edge`, or a method added to or dropped
/// from the gate.
#[test]
fn every_kink_kind_is_refused_exactly_where_curvature_is_read() {
    use crate::types::{EstimationMethod as M, FitOptions, SaemOfv, ViFinalOfv};
    let mut opts_for: Vec<(String, FitOptions, bool)> = Vec::new();
    for (m, reads) in [
        (M::Foce, true),
        (M::FoceI, true),
        (M::FoceGn, true),
        (M::FoceGnHybrid, true),
        (M::Laplace, true),
        (M::Imp, false),
        (M::Impmap, false),
        (M::Saem, false),
        (M::Bayes, false),
        (M::Vi, false),
    ] {
        let o = FitOptions { method: m, ..FitOptions::default() };
        opts_for.push((format!("{m:?}"), o, reads));
    }
    let o = FitOptions { method: M::Saem, saem_ofv: SaemOfv::Focei, ..FitOptions::default() };
    opts_for.push(("saem_ofv = focei".into(), o, true));
    for ofv in [SaemOfv::Imp, SaemOfv::Agq] {
        let o = FitOptions { method: M::Saem, saem_ofv: ofv, ..FitOptions::default() };
        opts_for.push((format!("saem_ofv = {ofv:?}"), o, false));
    }
    let o =
        FitOptions { method: M::Vi, vi_final_ofv: ViFinalOfv::Laplace, ..FitOptions::default() };
    opts_for.push(("vi_final_ofv = laplace".into(), o, true));
    let mut n_kink = 0;
    for kind in DensityKind::ALL {
        if matches!(kind, DensityKind::Normal | DensityKind::Copula) {
            continue;
        }
        let read = match kind.map() {
            DomainMap::Identity => "TVCL * exp(ETA_CL)",
            DomainMap::Exp => "TVCL * ETA_CL",
            DomainMap::InvLogit => "TVCL * ETA_CL",
        };
        let fix = if kind.omega_unused() { "1 FIX" } else { "0.09" };
        let shapes = if kind.shapes().is_empty() { String::new() } else { "()".to_string() };
        let src = format!(
            "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
             omega ETA_CL ~ {fix} density {}{shapes}\n  omega ETA_V ~ 0.04\n  sigma EPS ~ 0.01\n\n\
             [individual_parameters]\n  CL = {read}\n  V = TVV * exp(ETA_V)\n\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\n\
             [error_model]\n  DV ~ proportional(EPS)\n",
            kind.name()
        );
        let model = crate::parser::model_parser::parse_model_string(&src)
            .unwrap_or_else(|e| panic!("{}: {e}", kind.name()));
        let kink = kind.has_kink_or_edge();
        n_kink += usize::from(kink);
        assert_eq!(
            kink_kind(&model).map(|(k, e)| (k, e)),
            kink.then(|| (kind, "ETA_CL".to_string())),
            "{}",
            kind.name()
        );
        for (label, o, reads) in &opts_for {
            let r = fit_refusal(&model, o, &model.default_params);
            assert_eq!(r.is_some(), kink && *reads, "{} under {label}: {r:?}", kind.name());
            if let Some(msg) = r {
                assert!(msg.contains("`ETA_CL` follows") && msg.contains("method = imp"), "{msg}");
            }
        }
    }
    // The seven kinds the review lists.
    assert_eq!(n_kink, 7);
}

/// Measurement (#2669 review, finding 11): the cost of `correction_jets` (value, gradient
/// and η-Hessian) on a Gaussian copula of `p` margins, the shape FREM gives a density block
/// (its covariate ETAs join on `normal` margins), against the plain correction. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn measure_copula_correction_jets_cost() {
    for p in [2usize, 4, 8, 12, 16] {
        let mut margins = vec![DensityKind::Gamma, DensityKind::Logistic];
        margins.resize(p, DensityKind::Normal);
        let d = EtaDensity {
            kind: DensityKind::Copula,
            etas: (0..p).collect(),
            shape_thetas: vec![],
            trunc: None,
            margins: margins[..p].to_vec(),
            log_anchors: Vec::new(),
        };
        let omega = DMatrix::from_fn(p, p, |i, j| if i == j { 0.09 } else { 0.01 });
        let oi = omega.clone().try_inverse().unwrap();
        let eta: Vec<f64> = (0..p).map(|i| 0.05 * (i as f64 - 1.5)).collect();
        let reps = 2000 / p.max(1);
        let t0 = std::time::Instant::now();
        let mut sink = 0.0;
        for _ in 0..reps {
            sink += correction(std::slice::from_ref(&d), &eta, &oi, &[]);
        }
        let plain = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let (v, g, h) = correction_jets(std::slice::from_ref(&d), &eta, &oi, &[], true);
            sink += v + g[0] + h.unwrap()[(0, 0)];
        }
        let jets = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let (v, g, h) = correction_jets_impl(
                std::slice::from_ref(&d),
                &eta,
                |a, b| oi[(a, b)],
                |_| 0.0,
                true,
                false,
            );
            sink += v + g[0] + h.unwrap()[0][0];
        }
        let generic = t0.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "p = {p:2}: correction {:8.2} µs, generic jets {:10.2} µs ({:.0}×), separable {:8.2} µs ({:.0}×)  [{sink:.1}]",
            plain * 1e6,
            generic * 1e6,
            generic / plain,
            jets * 1e6,
            jets / plain
        );
    }
}

/// A model with the given `[parameters]` extra lines and `[individual_parameters]`, on a
/// 1-cpt oral structure reading CL, V, KA and F.
fn tv_model(params: &str, indiv: &str) -> crate::types::CompiledModel {
    crate::parser::model_parser::parse_model_string(&format!(
        "[parameters]\n  theta TVCL(1.0, 0.001, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta TVKA(1.0, 0.01, 10.0)\n  theta TVF(0.6, 0.01, 0.99)\n{params}\n  sigma EPS ~ 0.01\n\n\
         [individual_parameters]\n{indiv}\n\n\
         [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)\n\n\
         [error_model]\n  DV ~ proportional(EPS)\n"
    ))
    .unwrap_or_else(|e| panic!("{e}"))
}

/// Which summary each typical-value θ is (#2669): `TV·exp(η)` a median, `TV + η` a mean,
/// a mean-1 positive density's `TV·η` a mean, a symmetric ℝ density under `exp` a median,
/// an asymmetric one a geometric mean, `beta(μ)`'s μ a mean and `logitnormal(μ)`'s a
/// median; `density lognormal` is the classic median; a θ shared by two parameters, none.
/// Dies under: any rule swapped, or the shared θ labelled.
#[test]
fn typical_values_name_the_summary_each_theta_is() {
    use crate::types::TypicalStatistic::{GeometricMean, Mean, Median};
    let get = |m: &crate::types::CompiledModel| {
        let mut v: Vec<(String, String, crate::types::TypicalStatistic)> =
            typical_values(m).into_iter().map(|t| (t.theta, t.parameter, t.statistic)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    let s = |a: &str, b: &str, c| (a.to_string(), b.to_string(), c);
    let m = tv_model(
        "  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.1 density t(NU)\n  \
         theta NU(10, 3, 100)\n  omega ETA_F ~ 0.02 density beta(TVF)",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV + ETA_V\n  KA = TVKA * exp(ETA_KA)\n  F = ETA_F",
    );
    assert_eq!(
        get(&m),
        vec![
            s("TVCL", "CL", Median),
            s("TVF", "F", Mean),
            s("TVKA", "KA", Median),
            s("TVV", "V", Mean)
        ]
    );
    let m = tv_model(
        "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V ~ 0.04 density lognormal\n  \
         omega ETA_KA ~ 0.1 density gumbel\n  omega ETA_F ~ 0.3 density logitnormal(TVF)",
        "  CL = TVCL * ETA_CL\n  V = TVV * ETA_V\n  KA = TVKA * exp(ETA_KA)\n  F = ETA_F",
    );
    assert_eq!(
        get(&m),
        vec![
            s("TVCL", "CL", Mean),
            s("TVF", "F", Median),
            s("TVKA", "KA", GeometricMean),
            s("TVV", "V", Median)
        ]
    );
    // One θ, two parameters: no label.
    let m = tv_model(
        "  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVCL * exp(ETA_V)\n  KA = TVKA\n  F = TVF",
    );
    assert!(get(&m).iter().all(|v| v.0 != "TVCL"), "{:?}", get(&m));
}

/// `latent_of_effect` inverts `effect_of_latent` for every map, a truncation's included,
/// refuses an effect outside the support, and a θ-valued truncation bound is read at the θ
/// it is resolved at (#2669: it held the initial value).
/// Dies under: a wrong inverse, or a bound left at its initial value.
#[test]
fn the_effect_map_inverts_and_reads_fitted_bounds() {
    let dist = |kind: &str, trunc, tt| crate::types::EtaDistribution {
        form: "density".into(),
        kind: kind.into(),
        shape_thetas: vec![],
        truncation: trunc,
        truncation_thetas: tt,
    };
    for d in [
        dist("gamma", None, (None, None)),
        dist("logistic", None, (None, None)),
        dist("beta", None, (None, None)),
        dist("gamma", Some((Some(0.2), Some(5.0))), (None, None)),
        dist("t", Some((Some(-1.0), None)), (None, None)),
        dist("gamma", Some((None, Some(3.0))), (None, None)),
    ] {
        for z in [-2.0, -0.3, 0.0, 0.7, 1.9] {
            let x = effect_of_latent(&d, z);
            let back = latent_of_effect(&d, x);
            assert!((back - z).abs() < 1e-12, "{} {:?}: {z} → {x} → {back}", d.kind, d.truncation);
        }
    }
    assert!(latent_of_effect(&dist("gamma", None, (None, None)), -0.1).is_nan());
    assert!(latent_of_effect(&dist("beta", None, (None, None)), 1.2).is_nan());
    assert!(
        latent_of_effect(&dist("gamma", Some((Some(0.2), Some(5.0))), (None, None)), 6.0).is_nan()
    );
    // A θ bound: stored at its initial 5.0, resolved at the fitted 4.0.
    let d = dist("gamma", Some((Some(0.2), Some(5.0))), (None, Some("HI".into())));
    let r = resolve_distribution(&d, |n| (n == "HI").then_some(4.0));
    assert_eq!(r.truncation, Some((Some(0.2), Some(4.0))));
    assert!((effect_of_latent(&r, 50.0) - 4.0).abs() < 1e-12);
}

/// Effect shrinkage (#2669): with closed-form moments it is `1 − RMS(m(ẑ) − μ)/σ`; where
/// there are none (Cauchy) the quantile form `1 − IQR(m(ẑ))/IQR(prior)`, which is 0 for
/// effects spread as the prior is and 1 for effects all at one value.
/// Dies under: the fallback missing (NaN), or its ratio inverted.
#[test]
fn effect_shrinkage_falls_back_to_quantiles_without_moments() {
    let m = tv_model(
        "  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.04 density cauchy\n  \
         omega ETA_F ~ 0.02",
        "  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n  \
         F = inv_logit(logit(TVF) + ETA_F)",
    );
    let d = crate::types::EtaDistribution {
        form: "density".into(),
        kind: "cauchy".into(),
        shape_thetas: vec![],
        truncation: None,
        truncation_thetas: (None, None),
    };
    let th = m.default_params.theta.clone();
    // The prior's own quantiles, evenly spaced in probability: the sample IQR is the prior's.
    let n = 999;
    let probs: Vec<f64> = (1..=n).map(|i| i as f64 / (n + 1) as f64).collect();
    let spread = effect_quantiles_at(&d, 0.04, &[], &probs).unwrap();
    let s0 = effect_shrinkage_of(&d, 0.04, &m, &th, &spread);
    assert!(s0.abs() < 0.01, "spread as the prior: {s0}");
    let s1 = effect_shrinkage_of(&d, 0.04, &m, &th, &vec![0.1; 50]);
    assert!((s1 - 1.0).abs() < 1e-12, "all at one value: {s1}");
    // A gamma has moments: the RMS form.
    let g = crate::types::EtaDistribution { kind: "gamma".into(), ..d.clone() };
    let e = [0.7, 1.0, 1.3, 1.0];
    let want = 1.0 - ((0.09 + 0.0 + 0.09 + 0.0) / 4.0f64).sqrt() / 0.3;
    assert!((effect_shrinkage_of(&g, 0.09, &m, &th, &e) - want).abs() < 1e-12);
}

/// The separable copula path of `correction_jets` (#2669 review, finding 11) gives the
/// value, gradient and Hessian of the generic pair passes, on copulas of 2 to 6 margins
/// mixing every margin family (gamma, beta, t, logistic, normal) at points across the
/// support.
/// Dies under: a wrong term in the assembly (the `2(Aq)q″` diagonal, the `−2Ω⁻¹` part, a
/// margin's `A` scaling), or a margin read on the wrong scale.
#[test]
fn the_separable_copula_jets_are_the_generic_ones() {
    let margins = [
        DensityKind::Gamma,
        DensityKind::Beta,
        DensityKind::StudentT,
        DensityKind::Logistic,
        DensityKind::Normal,
        DensityKind::Gamma,
    ];
    let mut worst = 0.0f64;
    for p in 2..=6 {
        let mk = margins[..p].to_vec();
        // Shape θs: β's μ at θ[0], t's ν at θ[1].
        let shape_thetas: Vec<usize> = mk
            .iter()
            .filter_map(|k| match k {
                DensityKind::Beta => Some(0),
                DensityKind::StudentT => Some(1),
                _ => None,
            })
            .collect();
        let d = EtaDensity {
            kind: DensityKind::Copula,
            etas: (0..p).collect(),
            shape_thetas,
            trunc: None,
            margins: mk.clone(),
            log_anchors: Vec::new(),
        };
        let omega = DMatrix::from_fn(p, p, |i, j| {
            let w = |k: usize| if mk[k] == DensityKind::Beta { 0.02 } else { 0.09 };
            if i == j {
                w(i)
            } else {
                0.3 * (w(i) * w(j)).sqrt() * if (i + j) % 2 == 0 { 1.0 } else { -1.0 }
            }
        });
        let oi = omega.clone().try_inverse().unwrap();
        let theta = [0.3, 7.0];
        for t in [-0.8, 0.0, 0.6] {
            let eta: Vec<f64> = (0..p).map(|i| t * (1.0 - 0.3 * i as f64)).collect();
            let fast = correction_jets(std::slice::from_ref(&d), &eta, &oi, &theta, true);
            let (v, g, h) = correction_jets_impl(
                std::slice::from_ref(&d),
                &eta,
                |a, b| oi[(a, b)],
                |k| theta[k],
                true,
                false,
            );
            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            let (fh, h) = (fast.2.as_ref().unwrap(), h.as_ref().unwrap());
            assert!(fast.0.is_finite() && v.is_finite(), "value {} vs {v}", fast.0);
            assert!(fast.1.iter().chain(&g).all(|x| x.is_finite()), "{:?} vs {g:?}", fast.1);
            assert!(fh.iter().chain(h.iter().flatten()).all(|x| x.is_finite()), "Hessians");
            let mut gaps = vec![rel(fast.0, v)];
            for i in 0..p {
                gaps.push(rel(fast.1[i], g[i]));
                for j in 0..p {
                    gaps.push(rel(fh[(i, j)], h[i][j]));
                }
            }
            for gap in gaps {
                assert!(gap.is_finite(), "a non-finite gap");
                worst = worst.max(gap);
            }
        }
    }
    eprintln!("separable vs generic copula jets: worst {worst:.2e}");
    assert!(worst < 1e-10, "worst {worst:.2e}");
}
