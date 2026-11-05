use super::*;
use crate::sens::rtape::DualT;

include!("eta_density_copula_refs.rs");

/// `Mlin`'s mixed partials against closed forms: for `f = exp(u₀u₁u₂)`, every subset's
/// coefficient; for `ln`, `powf`, `recip`, `ln1p`, `expm1` of `u₀ + 2u₁`, the coefficient
/// of `{0, 1}` is `2 f″`.
/// Dies under: a wrong subset convolution, a wrong derivative table, or `apply` dropping
/// an order.
#[test]
fn the_multilinear_dual_has_the_closed_form_mixed_partials() {
    let (a, b, c) = (0.7f64, 1.3f64, 0.4f64);
    let x: Vec<Mlin<f64>> = [a, b, c]
        .iter()
        .enumerate()
        .map(|(i, &v)| Mlin::var(3, i, v))
        .collect();
    let f = x[0].mul(&x[1]).mul(&x[2]).exp();
    let e = (a * b * c).exp();
    // ∂³/∂a∂b∂c exp(abc) = e·(1 + 3abc + (abc)²).
    let p = a * b * c;
    let want_top = e * (1.0 + 3.0 * p + p * p);
    // ∂²/∂a∂b = e·(c + abc²) = e·c(1 + p).
    let want_ab = e * c * (1.0 + p);
    let mut worst = 0.0f64;
    for (got, want) in [(f.val(), e), (f.coeff(0b011), want_ab), (f.top(), want_top)] {
        assert!(got.is_finite() && want.is_finite());
        worst = worst.max((got - want).abs() / want.abs().max(1.0));
    }
    // g(u₀ + 2u₁): the {0,1} coefficient is 2·g″(s).
    let s = 0.6f64;
    let y = Mlin::var(2, 0, 0.2).add(&Mlin::var(2, 1, 0.2).scale(2.0));
    let cases: [(Mlin<f64>, f64); 6] = [
        (y.ln(), -1.0 / (s * s)),
        (y.powf(1.7), 1.7 * 0.7 * s.powf(-0.3)),
        (y.recip(), 2.0 / (s * s * s)),
        (y.ln1p(), -1.0 / ((1.0 + s) * (1.0 + s))),
        (y.expm1(), s.exp()),
        (y.exp(), s.exp()),
    ];
    for (g, g2) in cases {
        let got = g.top();
        assert!(got.is_finite() && g2.is_finite());
        worst = worst.max((got - 2.0 * g2).abs() / g2.abs().max(1.0));
    }
    eprintln!("Mlin mixed partials: worst relative {worst:.2e}");
    assert!(worst < 1e-12, "{worst:e}");
}

/// Every Archimedean family's log-density and CDF, at three parameters and in two to
/// four dimensions, against Copulas.jl 1.0.1 (`copula_anchor/refs.jl`), and the
/// negative-θ Frank and AMH (bivariate only).
/// Dies under: a wrong generator or inverse, a sign, or the density read off the wrong
/// coefficient.
#[test]
fn every_archimedean_family_is_copulas_jl() {
    let mut worst = (0.0f64, String::new());
    let scalar = ARCH_REFS
        .iter()
        .chain(ARCH_NEG_REFS)
        .map(|&(f, th, u, lp, c)| (f, vec![th], u, lp, c));
    let vector = ARCH2_REFS
        .iter()
        .map(|&(f, p, u, lp, c)| (f, p.to_vec(), u, lp, c));
    let mut skipped = 0;
    for (fam, pv, u, lp, cdf) in scalar.chain(vector) {
        let th = pv[0];
        // Copulas.jl evaluates far into the tail in log space; ferx's density comes off
        // its CDF and reads a density below `e^{−700}` as a wall (`log_density`'s docs).
        // ... and a density far below its CDF's terms too (cancellation): a wall is
        // accepted only where Copulas.jl's own density is below e^-30, and counted.
        if lp < -30.0 && fam.log_density::<f64>(u, &pv) == f64::NEG_INFINITY {
            skipped += 1;
            continue;
        }
        let got_lp = fam.log_density::<f64>(u, &pv);
        let xs: Vec<Mlin<f64>> = u.iter().map(|&v| Mlin::cst(u.len(), v)).collect();
        let got_cdf = fam.cdf(&xs, &pv).val();
        assert!(
            got_lp.is_finite() && got_cdf.is_finite() && lp.is_finite() && cdf.is_finite(),
            "{fam:?} θ {th} at {u:?}: {got_lp} {got_cdf}"
        );
        for (g, w, what) in [(got_lp, lp, "log c"), (got_cdf, cdf, "C")] {
            let e = (g - w).abs() / w.abs().max(1.0);
            if e > worst.0 {
                worst = (e, format!("{fam:?} {pv:?} (θ {th}) {what} at {u:?}"));
            }
        }
    }
    eprintln!(
        "non-elliptical vs Copulas.jl: worst relative {:.2e} ({}); {skipped} wall(s) below e^-30",
        worst.0, worst.1
    );
    assert!(worst.0 < ARCH_TOL, "{:e} ({})", worst.0, worst.1);
    // Measured: 2 walls (Hüsler–Reiss θ = 4 far off the diagonal, BB2 deep in a tail);
    // allow none beyond them.
    assert!(skipped <= 2, "{skipped} walls");
}
// Measured 6.05e-7 (Galambos θ = 4, d = 3: its alternating subset sum costs digits under
// strong dependence); Archimedean families alone 1.4e-13. Bound 8× the worst.
const ARCH_TOL: f64 = 5e-6;

/// The Student-t copula's log-density against Copulas.jl, bivariate and trivariate.
/// Dies under: a wrong quantile, a dropped Γ term, or `R` read where `R⁻¹` belongs.
#[test]
fn the_t_copula_is_copulas_jl() {
    let mut worst = 0.0f64;
    for &(nu, tri, u, lp) in T_REFS {
        let d = u.len();
        let mut r = nalgebra::DMatrix::<f64>::zeros(d, d);
        let mut at = 0;
        for i in 0..d {
            for j in 0..=i {
                r[(i, j)] = tri[at];
                r[(j, i)] = tri[at];
                at += 1;
            }
        }
        let ri = r.clone().try_inverse().expect("R inverts");
        let r_inv: Vec<Vec<f64>> = (0..d)
            .map(|i| (0..d).map(|j| ri[(i, j)]).collect())
            .collect();
        let got = t_copula_log_density::<f64>(u, nu, &r_inv, r.determinant().ln());
        assert!(got.is_finite() && lp.is_finite(), "ν {nu} at {u:?}: {got}");
        worst = worst.max((got - lp).abs() / lp.abs().max(1.0));
    }
    eprintln!("t copula vs Copulas.jl: worst relative {worst:.2e}");
    assert!(worst < T_TOL, "{worst:e}");
}
// Measured 7.55e-15; bound 1e-12.
const T_TOL: f64 = 1e-12;

/// Each bivariate copula's conditional margin is uniform for every `u` and every parameter,
/// so `∫ c(u, v) dv = 1` and `∫ ∂c(u, v)/∂pⱼ dv = 0` exactly, at each fixed `u`. Integrated
/// on the probit scale (`v = Φ(y)`, where the integrand is smooth and decays) at
/// `u = 0.1, 0.5, 0.9`, for every family at each reference parameter vector. The
/// parameter score is read off a `DualT` through the whole `Mlin` evaluation, so this pins
/// the exact parameter gradient the estimators use, with no finite difference; a 1-D
/// integral at fixed `u` resolves a strongly tail-dependent family's ridge, which a 2-D
/// grid does not.
/// Dies under: a density off by a factor, or a parameter's jet dropped or mis-signed
/// anywhere in a generator or its inverse.
#[test]
fn each_bivariate_conditional_margin_is_uniform_with_zero_score() {
    type D = DualT<2, f64>;
    let n = 4000usize;
    let (lo, hi) = (-8.5f64, 8.5f64);
    let h = (hi - lo) / n as f64;
    let w = |i: usize| -> f64 {
        if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        }
    };
    let phi = |x: f64| (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    let mut worst = (0.0f64, String::new());
    let mut seen = std::collections::HashSet::new();
    let scalar = ARCH_REFS
        .iter()
        .chain(ARCH_NEG_REFS)
        .map(|&(f, th, ..)| (f, vec![th]));
    let vector = ARCH2_REFS.iter().map(|&(f, p, ..)| (f, p.to_vec()));
    for (fam, pv) in scalar.chain(vector) {
        let key: Vec<u64> = pv.iter().map(|x| x.to_bits()).collect();
        if !seen.insert((fam.name(), key)) {
            continue;
        }
        let pd: Vec<D> = pv
            .iter()
            .enumerate()
            .map(|(j, &v)| D {
                v,
                g: [
                    if j == 0 { 1.0 } else { 0.0 },
                    if j == 1 { 1.0 } else { 0.0 },
                ],
            })
            .collect();
        for u in [0.1f64, 0.5, 0.9] {
            let (mut mass, mut score) = (0.0f64, [0.0f64; 2]);
            for i in 0..=n {
                let y = lo + i as f64 * h;
                let v = crate::stats::special::normal_cdf(y);
                let lc =
                    fam.log_density::<D>(&[D { v: u, g: [0.0; 2] }, D { v, g: [0.0; 2] }], &pd);
                if !lc.v.is_finite() {
                    continue;
                }
                assert!(
                    lc.g.iter().all(|g| g.is_finite()),
                    "{fam:?} {pv:?} at ({u}, {v})"
                );
                let wt = w(i) * phi(y) * lc.v.exp();
                mass += wt;
                score[0] += wt * lc.g[0];
                score[1] += wt * lc.g[1];
            }
            let k1 = h / 3.0;
            let mass = mass * k1;
            let score = [score[0] * k1, score[1] * k1];
            assert!(
                mass.is_finite() && score[0].is_finite() && score[1].is_finite(),
                "{fam:?} {pv:?} u {u}: mass {mass}, score {score:?}"
            );
            for (g, what) in [
                ((mass - 1.0).abs(), "mass − 1"),
                (score[0].abs(), "score 1"),
                (score[1].abs(), "score 2"),
            ] {
                if g > worst.0 {
                    worst = (g, format!("{fam:?} {pv:?} u {u} {what}"));
                }
            }
        }
    }
    eprintln!("conditional margins: worst {:.2e} ({})", worst.0, worst.1);
    assert!(worst.0 < MASS_TOL, "{:e} ({})", worst.0, worst.1);
}
// Measured 4.17e-5 (BB2 θ = 2, part of whose v-range reads as a wall); 2.4×.
const MASS_TOL: f64 = 1e-4;

/// The sampler draws its family: Kendall's τ of 3 000 bivariate draws against Copulas.jl's
/// τ, within 4 of its standard errors (`√(2(2n + 5)/(9n(n − 1)))` is the null SE; the
/// dependent one is smaller); and the trivariate Clayton's pairwise τ is the bivariate
/// one (an Archimedean copula's 2-margins are its own family).
/// Dies under: a conditional CDF read off the wrong coefficient, or a bisection on the
/// wrong side.
#[test]
fn each_sampler_has_its_kendall_tau() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1716);
    let n = 3000usize;
    let tau = |x: &[Vec<f64>], a: usize, b: usize| -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                s += ((x[i][a] - x[j][a]) * (x[i][b] - x[j][b])).signum();
            }
        }
        s / (x.len() * (x.len() - 1) / 2) as f64
    };
    let nf = n as f64;
    let se = (2.0 * (2.0 * nf + 5.0) / (9.0 * nf * (nf - 1.0))).sqrt();
    let mut worst = (0.0f64, String::new());
    let taus = TAU_REFS
        .iter()
        .map(|&(f, th, t)| (f, vec![th], t))
        .chain(TAU2_REFS.iter().map(|&(f, p, t)| (f, p.to_vec(), t)));
    for (fam, pv, want) in taus {
        let th = pv[0];
        let draws: Vec<Vec<f64>> = (0..n).map(|_| fam.sample(2, &pv, &mut rng)).collect();
        assert!(draws
            .iter()
            .flatten()
            .all(|v| v.is_finite() && *v > 0.0 && *v < 1.0));
        let got = tau(&draws, 0, 1);
        let z = (got - want).abs() / se;
        eprintln!("{fam:?} {pv:?}: τ {got:.4} vs {want:.4} ({z:.2} SE)");
        if z > worst.0 {
            worst = (z, format!("{fam:?} {pv:?} (θ {th})"));
        }
    }
    let clayton3: Vec<Vec<f64>> = (0..n)
        .map(|_| CopulaFamily::Clayton.sample(3, &[2.0], &mut rng))
        .collect();
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let z = (tau(&clayton3, a, b) - 0.5).abs() / se;
        if z > worst.0 {
            worst = (z, format!("trivariate Clayton θ 2, pair ({a}, {b})"));
        }
    }
    eprintln!("sampler Kendall τ: worst {:.2} SE ({})", worst.0, worst.1);
    assert!(worst.0 < 4.0, "{:.2} SE ({})", worst.0, worst.1);
}
