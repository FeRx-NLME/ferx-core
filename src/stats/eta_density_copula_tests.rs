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

/// The `CopulaSpec` each `COMPOSITE_REFS` label stands for (`copula_anchor/refs.jl`).
fn composite_spec(label: &str) -> CopulaSpec {
    use CopulaFamily as F;
    let fam = |f| Box::new(CopulaSpec::Family(f));
    match label {
        "fgm2" | "fgm3" => CopulaSpec::Family(F::Fgm),
        "plackett" | "plackett_neg" => CopulaSpec::Family(F::Plackett),
        "survival_clayton3" => CopulaSpec::Reflect {
            base: fam(F::Clayton),
            flip: vec![true; 3],
        },
        "survival_clayton3_flip2" => CopulaSpec::Reflect {
            base: fam(F::Clayton),
            flip: vec![false, true, false],
        },
        "rot90_gumbel" => CopulaSpec::Reflect {
            base: fam(F::Gumbel),
            flip: vec![true, false],
        },
        "rot270_gumbel" => CopulaSpec::Reflect {
            base: fam(F::Gumbel),
            flip: vec![false, true],
        },
        "khoudraji_gumbel" => CopulaSpec::Liebscher {
            parts: vec![
                CopulaSpec::Family(F::Independent),
                CopulaSpec::Family(F::Gumbel),
            ],
        },
        "khoudraji_clayton_gumbel" => CopulaSpec::Liebscher {
            parts: vec![
                CopulaSpec::Family(F::Clayton),
                CopulaSpec::Family(F::Gumbel),
            ],
        },
        "liebscher3" => CopulaSpec::Liebscher {
            parts: vec![
                CopulaSpec::Family(F::Clayton),
                CopulaSpec::Family(F::Gumbel),
                CopulaSpec::Family(F::Frank),
            ],
        },
        "nested_clayton" => CopulaSpec::Nested {
            family: F::Clayton,
            root: NestNode::Node(vec![
                NestNode::Node(vec![NestNode::Leaf(0), NestNode::Leaf(1)]),
                NestNode::Leaf(2),
            ]),
        },
        _ => panic!("no spec for {label}"),
    }
}

/// FGM, Plackett, and Copulas.jl's constructions over families — survival copulas (all
/// and some coordinates reflected), the 90°/270° rotations, Khoudraji (one and two
/// components), a three-part Liebscher, and a nested Clayton tree — each against
/// Copulas.jl 1.0.1's log-density and CDF, through the same `Mlin` CDF route.
/// Dies under: a reflection's inclusion–exclusion sign, a Liebscher weight row
/// misplaced, a nested tree's θ order, or FGM's subset order.
#[test]
fn every_composite_copula_is_copulas_jl() {
    let mut worst = (0.0f64, String::new());
    for &(label, pv, u, lp, cdf) in COMPOSITE_REFS {
        let spec = composite_spec(label);
        assert_eq!(spec.n_params(u.len()), pv.len(), "{label}: parameter count");
        let got_lp = spec.log_density::<f64>(u, pv);
        let xs: Vec<Mlin<f64>> = u.iter().map(|&v| Mlin::cst(u.len(), v)).collect();
        let got_cdf = spec.cdf(&xs, pv).val();
        assert!(
            got_lp.is_finite() && got_cdf.is_finite() && lp.is_finite() && cdf.is_finite(),
            "{label} at {u:?}: {got_lp} {got_cdf}"
        );
        for (g, w, what) in [(got_lp, lp, "log c"), (got_cdf, cdf, "C")] {
            let e = (g - w).abs() / w.abs().max(1.0);
            if e > worst.0 {
                worst = (e, format!("{label} {what} at {u:?}"));
            }
        }
    }
    eprintln!(
        "composites vs Copulas.jl: worst relative {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 < COMPOSITE_TOL, "{:e} ({})", worst.0, worst.1);
}
// Measured 4.05e-15 (the nested Clayton); bound 1e-12.
const COMPOSITE_TOL: f64 = 1e-12;

/// The generic sampler on constructions: a survival Clayton has the Clayton's τ (a
/// reflection of every coordinate preserves concordance), a 90° rotation negates it, and
/// a nested Clayton's within-child pair has the child's τ and its cross pairs the root's.
/// Dies under: a reflection applied to the wrong coordinate, or a tree's θs swapped.
#[test]
fn composite_samplers_have_their_kendall_tau() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(1716);
    let n = 2000usize;
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
    let clayton_tau = |t: f64| t / (t + 2.0);
    let draws =
        |spec: &CopulaSpec, d: usize, p: &[f64], rng: &mut rand::rngs::StdRng| -> Vec<Vec<f64>> {
            (0..n).map(|_| spec.sample(d, p, rng)).collect()
        };
    let mut worst = (0.0f64, String::new());
    let mut check = |got: f64, want: f64, what: &str| {
        let z = (got - want).abs() / se;
        eprintln!("{what}: τ {got:.4} vs {want:.4} ({z:.2} SE)");
        if z > worst.0 {
            worst = (z, what.to_string());
        }
    };
    let surv = composite_spec("survival_clayton3");
    let x = draws(&surv, 3, &[1.8], &mut rng);
    check(tau(&x, 0, 1), clayton_tau(1.8), "survival Clayton (0, 1)");
    let rot = composite_spec("rot90_gumbel");
    let x = draws(&rot, 2, &[1.7], &mut rng);
    check(tau(&x, 0, 1), -(1.0 - 1.0 / 1.7), "rotated90 Gumbel");
    let nest = composite_spec("nested_clayton");
    let x = draws(&nest, 3, &[0.8, 2.5], &mut rng);
    check(tau(&x, 0, 1), clayton_tau(2.5), "nested child (0, 1)");
    check(tau(&x, 0, 2), clayton_tau(0.8), "nested root (0, 2)");
    eprintln!("composite sampler τ: worst {:.2} SE ({})", worst.0, worst.1);
    assert!(worst.0 < 4.0, "{:.2} SE ({})", worst.0, worst.1);
}
