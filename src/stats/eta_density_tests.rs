use super::*;
use crate::sens::rtape::DualT;

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
    d: &EtaDensity,
    omega: f64,
    theta: &[f64],
    g: impl Fn(f64, f64) -> f64,
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
        400_000,
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
        (group(DensityKind::LogNormal, vec![0], vec![]), vec![], 0.12),
        (group(DensityKind::Beta, vec![0], vec![0]), vec![0.3], 0.02),
        (
            group(DensityKind::LogitNormal, vec![0], vec![0]),
            vec![0.3],
            0.12,
        ),
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
        worst = worst.max((mass - 1.0).abs());
    }
    // Measured 4.5e-14 (Simpson on 2e4 / 4e5 panels); 10× headroom.
    assert!(worst < 5e-13, "worst |mass − 1| = {worst:.3e}");
}

/// The random effect the model reads, `η = m(z)`, has the mean and variance the module
/// documents: mean 1 and variance ω for `gamma`, `invgamma` and `lognormal`, mean μ and
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
            g[i].is_finite() && h[(i, i)].is_finite(),
            "η{i}: {} {}",
            g[i],
            h[(i, i)]
        );
        worst = worst.max((g[i] - want_g[i]).abs() / (1.0 + want_g[i].abs()));
        worst = worst.max((h[(i, i)] - want_h[i]).abs() / (1.0 + want_h[i].abs()));
        for j in 0..4 {
            if i != j {
                assert_eq!(
                    h[(i, j)],
                    0.0,
                    "univariate groups do not couple: h[{i},{j}]"
                );
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
    assert!(
        (c.g[0] - want).abs() < 1e-14,
        "∂c/∂ν = {}, want {want}",
        c.g[0]
    );
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
    let at = |omega: f64| {
        correction(
            &b,
            &[0.1],
            &DMatrix::from_element(1, 1, 1.0 / omega),
            &[0.3],
        )
    };
    assert_eq!(at(0.22), f64::INFINITY, "ω just past μ(1 − μ) = 0.21");
    assert_eq!(at(0.3), f64::INFINITY);
    assert!(at(0.2).is_finite());
}
