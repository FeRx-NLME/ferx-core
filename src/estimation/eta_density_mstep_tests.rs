use super::*;
use crate::stats::eta_density::{redraw_latents, DensityKind};
use rand::SeedableRng;

/// A model shell carrying only density groups, enough for the M-step.
fn model_with(densities: Vec<EtaDensity>) -> CompiledModel {
    let mut m = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Auto);
    m.eta_densities = densities;
    m
}

/// `n` latent draws of each group at `omega`/`theta`, the simulation's own sampler.
fn draws(model: &CompiledModel, omega: &DMatrix<f64>, theta: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1716);
    let chol = omega.clone().cholesky().unwrap().l();
    (0..n)
        .map(|_| {
            let z: Vec<f64> = (0..omega.nrows())
                .map(|_| rand_distr::Distribution::sample(&rand_distr::StandardNormal, &mut rng))
                .collect();
            let mut e: Vec<f64> = (&chol * DVector::from_vec(z)).iter().copied().collect();
            redraw_latents(&model.eta_densities, omega, theta, &mut e, &mut rng);
            e
        })
        .collect()
}

/// The MCEM M-step recovers the parameters its draws were simulated at: ω of a gamma, ω
/// and μ of a beta, and ω and ν of a t, each from a perturbed start, on 40 000 draws. The
/// oracle is the simulation truth; the bound is set from the sampling error measured
/// below, not chosen.
/// Dies under: a wrong score (the fixed point moves), or a write-back to the wrong θ / Ω
/// entry.
#[test]
fn the_mcem_mstep_recovers_the_simulation_truth() {
    let model = model_with(vec![
        EtaDensity {
            kind: DensityKind::Gamma,
            etas: vec![0],
            shape_thetas: vec![],
            trunc: None,
            margins: vec![],
        },
        EtaDensity {
            kind: DensityKind::Beta,
            etas: vec![1],
            shape_thetas: vec![0],
            trunc: None,
            margins: vec![],
        },
        EtaDensity {
            kind: DensityKind::StudentT,
            etas: vec![2],
            shape_thetas: vec![1],
            trunc: None,
            margins: vec![],
        },
    ]);
    let truth_omega = DMatrix::from_diagonal(&DVector::from_vec(vec![0.15, 0.02, 0.09]));
    let truth_theta = [0.3, 6.0];
    let zs = draws(&model, &truth_omega, &truth_theta, 40_000);
    let w = 1.0 / zs.len() as f64;
    let weighted: Vec<(&[f64], f64)> = zs.iter().map(|z| (z.as_slice(), w)).collect();
    let ms = DensityMstep::new(
        &model,
        &[false; 3],
        &[false; 2],
        &[(0.001, 0.999), (2.1, 100.0)],
    )
    .expect("density groups");
    let mut omega = DMatrix::from_diagonal(&DVector::from_vec(vec![0.3, 0.04, 0.2]));
    let mut theta = vec![0.5, 20.0];
    ms.maximise(&model, &weighted, &mut omega, &mut theta);
    let rel = |got: f64, want: f64| (got - want).abs() / want;
    let gaps = [
        rel(omega[(0, 0)], 0.15),
        rel(omega[(1, 1)], 0.02),
        rel(theta[0], 0.3),
        rel(omega[(2, 2)], 0.09),
        rel(theta[1], 6.0),
    ];
    eprintln!("relative gaps ω_gamma, ω_beta, μ, ω_t, ν: {gaps:?}");
    for g in gaps {
        assert!(g.is_finite());
    }
    // Measured (seed 1716, 40 000 draws): ω 0.6–0.7%, μ 0.01%, ν 2.3%, i.e. about one
    // sampling SE each (`√(2/N)` = 0.7% for a variance); bounds about 4× and 3×.
    let bounds = [0.03, 0.03, 0.03, 0.03, 0.07];
    for (g, b) in gaps.iter().zip(bounds) {
        assert!(*g < b, "{gaps:?}");
    }
}

/// A FIXed ω and a FIXed shape θ are left where they are.
#[test]
fn a_fixed_parameter_is_not_moved() {
    let model = model_with(vec![EtaDensity {
        kind: DensityKind::Beta,
        etas: vec![0],
        shape_thetas: vec![0],
        trunc: None,
        margins: vec![],
    }]);
    let omega0 = DMatrix::from_element(1, 1, 0.03);
    let zs = draws(&model, &DMatrix::from_element(1, 1, 0.02), &[0.3], 2_000);
    let weighted: Vec<(&[f64], f64)> = zs.iter().map(|z| (z.as_slice(), 1.0 / 2_000.0)).collect();
    let ms = DensityMstep::new(&model, &[true], &[true], &[(0.001, 0.999)]).unwrap();
    let (mut omega, mut theta) = (omega0.clone(), vec![0.5]);
    ms.maximise(&model, &weighted, &mut omega, &mut theta);
    assert_eq!(omega, omega0);
    assert_eq!(theta, vec![0.5]);
    assert!(ms.owned_thetas().is_empty(), "a FIXed shape θ is not owned");
}

/// Each Ω prior's log-density, up to its constant, matches its textbook form: the
/// inverse Wishart and Wishart on a 2×2 block, and LKJ(η) with inverse-gamma variances,
/// whose 1×1 case is the inverse gamma alone. Differences between two blocks, so the
/// dropped constants cancel.
/// Dies under: a wrong exponent on `ln|Ω|`, the trace term transposed, or LKJ's
/// `(ω, R) → Ω` Jacobian dropped.
#[test]
fn every_omega_prior_is_its_textbook_density() {
    let lam = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.05]);
    let (o1, o2) = (
        DMatrix::from_row_slice(2, 2, &[0.11, 0.01, 0.01, 0.04]),
        DMatrix::from_row_slice(2, 2, &[0.07, 0.025, 0.025, 0.06]),
    );
    let nu = 4.0;
    let n = 2.0;
    let iw = |o: &DMatrix<f64>| {
        -0.5 * (nu + n + 1.0) * o.determinant().ln()
            - 0.5 * (&lam * o.clone().try_inverse().unwrap()).trace()
    };
    let wi = |o: &DMatrix<f64>| {
        0.5 * (nu - n - 1.0) * o.determinant().ln()
            - 0.5 * nu * (lam.clone().try_inverse().unwrap() * o).trace()
    };
    let lkj = |o: &DMatrix<f64>, eta: f64, a: f64, b: &[f64]| {
        let d = [o[(0, 0)], o[(1, 1)]];
        let r = o[(0, 1)] / (d[0] * d[1]).sqrt();
        (eta - 1.0) * (1.0 - r * r).ln()
            + d.iter()
                .zip(b)
                .map(|(&w, &bi)| -(a + 1.0) * w.ln() - bi / w)
                .sum::<f64>()
            - 0.5 * (d[0].ln() + d[1].ln())
    };
    let pri = [
        (
            BlockPrior::InverseWishart {
                nu,
                lambda: lam.clone(),
            },
            iw(&o1) - iw(&o2),
        ),
        (
            BlockPrior::Wishart {
                nu,
                lambda: lam.clone(),
            },
            wi(&o1) - wi(&o2),
        ),
        (
            BlockPrior::Lkj {
                eta: 2.5,
                a: 1.5,
                b: vec![0.045, 0.025],
            },
            lkj(&o1, 2.5, 1.5, &[0.045, 0.025]) - lkj(&o2, 2.5, 1.5, &[0.045, 0.025]),
        ),
    ];
    for (p, want) in pri {
        let got = p.log_density(&o1) - p.log_density(&o2);
        assert!((got - want).abs() < 1e-12, "{p:?}: {got} vs {want}");
    }
    // LKJ on one variance is its inverse gamma.
    let one = DMatrix::from_element(1, 1, 0.08);
    let ig = BlockPrior::InverseWishart {
        nu: 1.0,
        lambda: DMatrix::from_element(1, 1, 0.1),
    };
    let lk = BlockPrior::Lkj {
        eta: 3.0,
        a: 0.5,
        b: vec![0.05],
    };
    let two = DMatrix::from_element(1, 1, 0.15);
    let d_ig = ig.log_density(&one) - ig.log_density(&two);
    let d_lk = lk.log_density(&one) - lk.log_density(&two);
    assert!((d_ig - d_lk).abs() < 1e-12, "{d_ig} vs {d_lk}");
}
