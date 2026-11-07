use super::*;
use crate::sens::num::PkNum;
use crate::sens::rtape::DualT;

/// `log N(φ; μ, Ω)` for a 2×2 Ω given by its entries `(Ω₁₁, Ω₂₁, Ω₂₂)`, generic so that
/// nested forward duals give its exact gradient and Hessian.
fn log_normal_2<T: PkNum>(mu: [T; 2], om: [T; 3], phi: [f64; 2]) -> T {
    let c = T::from_f64;
    let e0 = c(phi[0]) - mu[0];
    let e1 = c(phi[1]) - mu[1];
    let (a, b, d) = (om[0], om[1], om[2]);
    let det = a * d - b * b;
    let quad = (d * e0 * e0 - c(2.0) * b * e0 * e1 + a * e1 * e1) / det;
    c(-0.5) * det.ln() - c(0.5) * quad
}

type D1 = DualT<5, f64>;
type D2 = DualT<5, D1>;

/// Gradient and Hessian of [`log_normal_2`] in `(μ₁, μ₂, Ω₁₁, Ω₂₁, Ω₂₂)` at `φ`.
fn dual_derivatives(
    mu: [f64; 2],
    omega: &DMatrix<f64>,
    phi: [f64; 2],
) -> ([f64; 5], [[f64; 5]; 5]) {
    let x = [mu[0], mu[1], omega[(0, 0)], omega[(1, 0)], omega[(1, 1)]];
    let var = |i: usize| -> D2 {
        let mut g = [D1 {
            v: 0.0,
            g: [0.0; 5],
        }; 5];
        g[i] = D1 {
            v: 1.0,
            g: [0.0; 5],
        };
        let mut inner = [0.0; 5];
        inner[i] = 1.0;
        D2 {
            v: D1 { v: x[i], g: inner },
            g,
        }
    };
    let v: Vec<D2> = (0..5).map(var).collect();
    let f = log_normal_2([v[0], v[1]], [v[2], v[3], v[4]], phi);
    let mut grad = [0.0; 5];
    let mut hess = [[0.0; 5]; 5];
    for i in 0..5 {
        grad[i] = f.g[i].v;
        for j in 0..5 {
            hess[i][j] = f.g[i].g[j];
        }
    }
    (grad, hess)
}

fn scope_2x2() -> AccumScope {
    AccumScope {
        mu: vec![
            MuCoord {
                theta: 0,
                eta: 0,
                additive: false,
            },
            MuCoord {
                theta: 1,
                eta: 1,
                additive: false,
            },
        ],
        data: Vec::new(),
        om: vec![(0, 0), (1, 0), (1, 1)],
        n_theta: 2,
        n_sigma: 0,
        n_eta: 2,
    }
}

/// [`prior_score`] is the exact gradient of `log N(φ; μ, Ω)` in `(μ, Ω₁₁, Ω₂₁, Ω₂₂)`, by
/// nested forward duals (an exact oracle, not finite differences). Catches a slip in the
/// off-diagonal factor 2 or in the sign of the μ score, either of which moves the fixed
/// point of the recursion.
#[test]
fn prior_score_is_the_exact_gradient_of_the_normal_log_density() {
    let omega = DMatrix::from_row_slice(2, 2, &[0.3, 0.08, 0.08, 0.5]);
    let oi = omega.clone().try_inverse().unwrap();
    let mu = [0.4, -0.2];
    let phi = [0.9, -0.75];
    let eta = [phi[0] - mu[0], phi[1] - mu[1]];
    let s = prior_score(&scope_2x2(), &eta, &oi);
    let (grad, _) = dual_derivatives(mu, &omega, phi);
    let mut worst = 0.0f64;
    for i in 0..5 {
        assert!(s[i].is_finite() && grad[i].is_finite());
        worst = worst.max((s[i] - grad[i]).abs());
    }
    eprintln!("prior score vs nested duals: worst {worst:.2e}");
    // Measured 8.9e-16; 1e-12 leaves room for the 2×2 inverse's rounding.
    assert!(worst < 1e-12, "{worst:e}");
}

/// [`prior_info`] is the prior expectation of the exact negative Hessian. The Hessian of
/// `log N` is affine in `ηηᵀ` (and linear in η across μ–Ω), so its expectation under
/// `N(0, Ω)` is its average over the four sigma points `±√2·L eⱼ` (`L Lᵀ = Ω`), exactly.
/// Catches a gain built on the wrong scale (a missing ½, or the Ω block's trace taken on
/// the wrong product), which would change the speed of the recursion silently.
#[test]
fn prior_info_is_the_expected_negative_hessian() {
    let omega = DMatrix::from_row_slice(2, 2, &[0.3, 0.08, 0.08, 0.5]);
    let oi = omega.clone().try_inverse().unwrap();
    let l = omega.clone().cholesky().unwrap().l();
    let mu = [0.4, -0.2];
    let mut avg = [[0.0; 5]; 5];
    let mut pts = 0.0;
    for j in 0..2 {
        for sgn in [-1.0, 1.0] {
            let eta = [sgn * 2f64.sqrt() * l[(0, j)], sgn * 2f64.sqrt() * l[(1, j)]];
            let phi = [mu[0] + eta[0], mu[1] + eta[1]];
            let (_, h) = dual_derivatives(mu, &omega, phi);
            for a in 0..5 {
                for b in 0..5 {
                    avg[a][b] -= h[a][b];
                }
            }
            pts += 1.0;
        }
    }
    let info = prior_info(&scope_2x2(), &oi);
    let mut worst = 0.0f64;
    for a in 0..5 {
        for b in 0..5 {
            let (got, want) = (info[(a, b)], avg[a][b] / pts);
            assert!(got.is_finite() && want.is_finite());
            worst = worst.max((got - want).abs());
        }
    }
    eprintln!("prior information vs E[−Hessian]: worst {worst:.2e}");
    // Measured 4.4e-15 (rounding in the sigma points); 1e-12 is 225× that.
    assert!(worst < 1e-12, "{worst:e}");
}

/// `γₖ = k^{−2/3}`: a full first step, and a sequence in the Polyak–Ruppert range.
#[test]
fn polyak_gamma_is_in_the_averaging_range() {
    assert_eq!(polyak_gamma(1), 1.0);
    assert!((polyak_gamma(8) - 0.25).abs() < 1e-15);
    assert!((polyak_gamma(1000) - 0.01).abs() < 1e-12);
}

/// The scope moves exactly the free parameters: a FIXed θ, a FIXed σ, a mu-referenced θ
/// (it moves on its mu scale instead), a structurally absent Ω entry and a FIXed ω are all
/// left out of the data block or the Ω block.
#[test]
fn the_scope_moves_only_free_parameters() {
    let parsed = crate::parser::model_parser::parse_full_model(
        r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0, FIX)
  theta SHIFT(0.5, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04 FIX
  sigma PROP_ERR ~ 0.02 (sd)
  sigma ADD_ERR ~ 0.1 (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V) + SHIFT
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#,
    )
    .expect("model parses");
    let p = &parsed.model.default_params;
    let mu = vec![MuCoord {
        theta: 0,
        eta: 0,
        additive: false,
    }];
    let sc = AccumScope::new(p, mu, 2);
    // TVV and SHIFT are data-scored; TVKA is FIX; TVCL moves as μ. PROP_ERR is free
    // (packed index 4), ADD_ERR FIX.
    assert_eq!(sc.data, vec![1, 3, 4]);
    // Ω: only ETA_CL's variance; (2,1) is structurally absent and ETA_V FIX.
    assert_eq!(sc.om, vec![(0, 0)]);
    assert_eq!(sc.n(), 1 + 3 + 1);
}
