//! Unit tests for the variational families.
//!
//! Every derivative here is checked against central finite differences of the
//! quantity it claims to differentiate. These are the same class of check the
//! `Dual2`-vs-FD rule imposes on the sensitivity kernels, and for the same
//! reason: each formula is written once, so nothing else would disagree with a
//! wrong one.

use super::*;
use nalgebra::{DMatrix, DVector};

use crate::types::OmegaMatrix;

/// A correlated 2×2 `Ω` — off-diagonal entries are what separate `FullRank` from
/// `MeanField`, so the default fixture must have them.
fn omega_2x2() -> OmegaMatrix {
    let m = DMatrix::from_row_slice(2, 2, &[0.037, 0.0113, 0.0113, 0.017]);
    OmegaMatrix::from_matrix_with_mask(
        m,
        vec!["ETA_CL".into(), "ETA_V".into()],
        false,
        DMatrix::from_element(2, 2, true),
    )
}

fn omega_3x3() -> OmegaMatrix {
    let m = DMatrix::from_row_slice(
        3,
        3,
        &[0.09, 0.02, 0.01, 0.02, 0.06, 0.015, 0.01, 0.015, 0.04],
    );
    OmegaMatrix::from_matrix_with_mask(
        m,
        vec!["A".into(), "B".into(), "C".into()],
        false,
        DMatrix::from_element(3, 3, true),
    )
}

/// A deterministic, non-trivial `φ` — perturbed off the prior so no derivative is
/// accidentally evaluated at a stationary point.
fn perturbed_phi<F: VariationalFamily>(fam: &F, omega: &OmegaMatrix) -> Vec<f64> {
    let mut phi = fam.init(omega);
    for (i, p) in phi.iter_mut().enumerate() {
        *p += 0.15 * ((i * 7 + 3) as f64).sin();
    }
    phi
}

fn central_fd<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], i: usize) -> f64 {
    let h = 1e-6 * (1.0 + x[i].abs());
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[i] += h;
    xm[i] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() / scale < tol,
        "{what}: got {got:.9e}, want {want:.9e}, rel {:.2e}",
        (got - want).abs() / scale
    );
}

// ---------------------------------------------------------------------------
// Shapes and initialization
// ---------------------------------------------------------------------------

#[test]
fn param_counts_match_layout() {
    assert_eq!(MeanField::new(1).n_params(), 2);
    assert_eq!(MeanField::new(4).n_params(), 8);
    // d means + d(d+1)/2 lower-triangular entries.
    assert_eq!(FullRank::new(1).n_params(), 2);
    assert_eq!(FullRank::new(2).n_params(), 5);
    assert_eq!(FullRank::new(3).n_params(), 9);
}

#[test]
fn tril_index_is_row_major_lower_triangle() {
    assert_eq!(tril_index(0, 0), 0);
    assert_eq!(tril_index(1, 0), 1);
    assert_eq!(tril_index(1, 1), 2);
    assert_eq!(tril_index(2, 0), 3);
    assert_eq!(tril_index(2, 2), 5);
    assert_eq!(n_tril(3), 6);
}

/// `init` must put `q` exactly at the prior: mean zero, covariance `Ω`. For
/// `FullRank` this is exact even with correlation; for `MeanField` only the
/// diagonal can be matched, which is the family's known limitation.
#[test]
fn init_places_q_at_the_prior() {
    let omega = omega_2x2();

    let fr = FullRank::new(2);
    let (mean, cov) = fr.moments(&fr.init(&omega));
    for k in 0..2 {
        assert_close(mean[k], 0.0, 1e-12, "full-rank init mean");
        for j in 0..2 {
            assert_close(
                cov[(k, j)],
                omega.matrix[(k, j)],
                1e-10,
                "full-rank init cov",
            );
        }
    }

    let mf = MeanField::new(2);
    let (mean, cov) = mf.moments(&mf.init(&omega));
    for k in 0..2 {
        assert_close(mean[k], 0.0, 1e-12, "mean-field init mean");
        assert_close(
            cov[(k, k)],
            omega.matrix[(k, k)],
            1e-10,
            "mean-field init var",
        );
    }
    assert_eq!(cov[(0, 1)], 0.0, "mean-field cov must stay diagonal");
}

/// At the prior the KL is exactly zero — `q == p(η)`. The full-rank family must
/// hit this on a *correlated* `Ω`, which is the check that the Cholesky layout
/// and the `log`-diagonal convention agree with `omega.chol`.
#[test]
fn kl_is_zero_at_the_prior() {
    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let kl = fr
        .kl_to_normal(&fr.init(&omega), &omega)
        .expect("closed form");
    assert_close(kl.value, 0.0, 1e-9, "full-rank KL at prior");

    // Mean-field cannot represent the correlation, so its KL at "its own prior"
    // is strictly positive — that gap IS the mean-field approximation error.
    let mf = MeanField::new(2);
    let kl_mf = mf
        .kl_to_normal(&mf.init(&omega), &omega)
        .expect("closed form");
    assert!(
        kl_mf.value > 0.0,
        "mean-field cannot match a correlated prior exactly, got KL = {}",
        kl_mf.value
    );

    // On an uncorrelated Ω it can, and must.
    let diag = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["A".into(), "B".into()]);
    let kl_diag = mf
        .kl_to_normal(&mf.init(&diag), &diag)
        .expect("closed form");
    assert_close(kl_diag.value, 0.0, 1e-9, "mean-field KL at diagonal prior");
}

// ---------------------------------------------------------------------------
// Sampling and the reparameterization path
// ---------------------------------------------------------------------------

/// `sample` must implement `η = μ + Lε` exactly, and reproduce the family's own
/// claimed moments when averaged over `ε`.
#[test]
fn sample_matches_mu_plus_l_eps() {
    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let phi = perturbed_phi(&fr, &omega);
    let (mean, cov) = fr.moments(&phi);

    // ε = 0 must return the mean exactly.
    let eta0 = fr.sample(&phi, &[0.0, 0.0]);
    for k in 0..2 {
        assert_close(eta0[k], mean[k], 1e-12, "sample at eps=0");
    }

    // A unit ε in coordinate j must move η by column j of L, whose outer
    // products reconstruct the covariance.
    let mut reconstructed = DMatrix::<f64>::zeros(2, 2);
    for j in 0..2 {
        let mut eps = vec![0.0; 2];
        eps[j] = 1.0;
        let col: Vec<f64> = fr
            .sample(&phi, &eps)
            .iter()
            .zip(mean.iter())
            .map(|(e, m)| e - m)
            .collect();
        for a in 0..2 {
            for b in 0..2 {
                reconstructed[(a, b)] += col[a] * col[b];
            }
        }
    }
    for a in 0..2 {
        for b in 0..2 {
            assert_close(reconstructed[(a, b)], cov[(a, b)], 1e-10, "LLᵀ vs moments");
        }
    }
}

/// `chain_to_phi` must be the transposed Jacobian of `sample`: for every basis
/// direction of `η`, the accumulated `∂η/∂φ` must match central FD of `sample`.
#[test]
fn chain_to_phi_matches_fd_of_sample() {
    let omega = omega_3x3();

    for (label, fam) in [
        (
            "full_rank",
            Box::new(FullRank::new(3)) as Box<dyn VariationalFamily>,
        ),
        ("mean_field", Box::new(MeanField::new(3))),
    ] {
        let phi = {
            let mut p = fam.init(&omega);
            for (i, v) in p.iter_mut().enumerate() {
                *v += 0.15 * ((i * 7 + 3) as f64).sin();
            }
            p
        };
        let eps = [0.7_f64, -1.3, 0.4];

        // One basis direction at a time isolates a row of ∂η/∂φ.
        for out_k in 0..3 {
            let mut g_eta = vec![0.0; 3];
            g_eta[out_k] = 1.0;
            let mut got = vec![0.0; fam.n_params()];
            fam.chain_to_phi(&phi, &eps, &g_eta, &mut got);

            for (i, &g) in got.iter().enumerate() {
                let want = central_fd(|p| fam.sample(p, &eps)[out_k], &phi, i);
                assert_close(g, want, 1e-6, &format!("{label} ∂η[{out_k}]/∂φ[{i}]"));
            }
        }
    }
}

/// `chain_to_phi` accumulates rather than overwrites — the ELBO assembly relies
/// on this to average over Monte-Carlo draws in one buffer.
#[test]
fn chain_to_phi_accumulates() {
    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let phi = perturbed_phi(&fr, &omega);
    let eps = [0.5_f64, -0.25];
    let g_eta = [1.0_f64, 2.0];

    let mut once = vec![0.0; fr.n_params()];
    fr.chain_to_phi(&phi, &eps, &g_eta, &mut once);

    let mut twice = vec![0.0; fr.n_params()];
    fr.chain_to_phi(&phi, &eps, &g_eta, &mut twice);
    fr.chain_to_phi(&phi, &eps, &g_eta, &mut twice);

    for i in 0..fr.n_params() {
        assert_close(twice[i], 2.0 * once[i], 1e-12, "accumulation");
    }
}

// ---------------------------------------------------------------------------
// KL: value, and gradients wrt φ and Ω
// ---------------------------------------------------------------------------

/// The closed-form KL must equal a Monte-Carlo estimate of
/// `E_q[log q(η) − log p(η|Ω)]`. This cross-checks the analytic KL against the
/// family's own `log_density` and the prior — the two halves of the `vi_kl` routes
/// must agree, or the switch would change the objective rather than only its
/// variance. The same identity is checked at the ELBO level, with derivatives, by
/// `elbo_tests::mc_kl_kernel_converges_to_the_closed_form`.
#[test]
fn closed_form_kl_matches_monte_carlo() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let phi = perturbed_phi(&fr, &omega);
    let closed = fr.kl_to_normal(&phi, &omega).expect("closed form").value;

    let mut rng = StdRng::seed_from_u64(20240704);
    let n = 400_000;
    let mut acc = 0.0;
    for _ in 0..n {
        let eps: Vec<f64> = (0..2).map(|_| StandardNormal.sample(&mut rng)).collect();
        let eta = fr.sample(&phi, &eps);
        let (log_q, _) = fr.log_density(&phi, &eta);
        // log p(η | Ω) = −½(ηᵀΩ⁻¹η + log|Ω| + d·log 2π)
        let ev = DVector::from_vec(eta.clone());
        let quad = ev.dot(&(&omega.inv * &ev));
        let log_p = -0.5 * (quad + omega.log_det + 2.0 * std::f64::consts::TAU.ln());
        acc += log_q - log_p;
    }
    let mc = acc / n as f64;
    // Monte-Carlo error at this sample size; the point is agreement, not precision.
    assert!(
        (mc - closed).abs() < 5e-3,
        "closed-form KL {closed:.6} vs Monte-Carlo {mc:.6}"
    );
}

/// `∂KL/∂φ` against central FD of the KL value, for both families.
#[test]
fn kl_d_phi_matches_fd() {
    let omega = omega_3x3();

    for (label, fam) in [
        (
            "full_rank",
            Box::new(FullRank::new(3)) as Box<dyn VariationalFamily>,
        ),
        ("mean_field", Box::new(MeanField::new(3))),
    ] {
        let phi = {
            let mut p = fam.init(&omega);
            for (i, v) in p.iter_mut().enumerate() {
                *v += 0.15 * ((i * 7 + 3) as f64).sin();
            }
            p
        };
        let kl = fam.kl_to_normal(&phi, &omega).expect("closed form");

        for i in 0..fam.n_params() {
            let want = central_fd(
                |p| fam.kl_to_normal(p, &omega).expect("closed form").value,
                &phi,
                i,
            );
            assert_close(kl.d_phi[i], want, 1e-6, &format!("{label} ∂KL/∂φ[{i}]"));
        }
    }
}

/// `∂KL/∂Ω` against central FD, perturbing `Ω` symmetrically.
///
/// `d_omega` is the derivative wrt `Ω` treated as a free symmetric matrix, so an
/// off-diagonal FD must perturb both `(i,j)` and `(j,i)` and the analytic entry
/// is compared to half the resulting slope (each of the two entries carries half
/// the symmetric perturbation).
#[test]
fn kl_d_omega_matches_fd() {
    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let phi = perturbed_phi(&fr, &omega);
    let kl = fr.kl_to_normal(&phi, &omega).expect("closed form");

    let rebuild = |m: DMatrix<f64>| {
        OmegaMatrix::from_matrix_with_mask(
            m,
            vec!["ETA_CL".into(), "ETA_V".into()],
            false,
            DMatrix::from_element(2, 2, true),
        )
    };

    let h = 1e-7;
    for i in 0..2 {
        for j in 0..=i {
            let mut mp = omega.matrix.clone();
            let mut mm = omega.matrix.clone();
            mp[(i, j)] += h;
            mm[(i, j)] -= h;
            if i != j {
                mp[(j, i)] += h;
                mm[(j, i)] -= h;
            }
            let fp = fr.kl_to_normal(&phi, &rebuild(mp)).expect("cf").value;
            let fm = fr.kl_to_normal(&phi, &rebuild(mm)).expect("cf").value;
            let slope = (fp - fm) / (2.0 * h);
            // Diagonal: one entry moved, so the analytic entry is the slope.
            // Off-diagonal: two entries moved, each contributing d_omega[(i,j)].
            let want = if i == j { slope } else { slope / 2.0 };
            assert_close(kl.d_omega[(i, j)], want, 1e-5, &format!("∂KL/∂Ω[{i},{j}]"));
        }
    }
}

/// The closed-form `Ω` maximizer claimed in the module docs — `Ω* = (1/N)Σ(Sᵢ +
/// μᵢμᵢᵀ)` — must be a stationary point of the summed KL. This is the property
/// that lets `Ω` leave the stochastic optimization entirely, so it needs its own
/// check rather than being taken on faith.
#[test]
fn closed_form_omega_is_a_stationary_point_of_the_kl() {
    let omega_start = omega_3x3();
    let fr = FullRank::new(3);

    // Three subjects with distinct posteriors.
    let phis: Vec<Vec<f64>> = (0..3)
        .map(|s| {
            let mut p = fr.init(&omega_start);
            for (i, v) in p.iter_mut().enumerate() {
                *v += 0.2 * (((i + s * 5) * 7 + 3) as f64).sin();
            }
            p
        })
        .collect();

    // Ω* = (1/N) Σ (Sᵢ + μᵢ μᵢᵀ)
    let mut acc = DMatrix::<f64>::zeros(3, 3);
    for phi in &phis {
        let (mu, s) = fr.moments(phi);
        acc += s + &mu * mu.transpose();
    }
    let omega_star = OmegaMatrix::from_matrix_with_mask(
        acc / phis.len() as f64,
        vec!["A".into(), "B".into(), "C".into()],
        false,
        DMatrix::from_element(3, 3, true),
    );

    // Σᵢ ∂KLᵢ/∂Ω must vanish there.
    let mut total = DMatrix::<f64>::zeros(3, 3);
    for phi in &phis {
        total += fr.kl_to_normal(phi, &omega_star).expect("cf").d_omega;
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                total[(i, j)].abs() < 1e-9,
                "∂ΣKL/∂Ω[{i},{j}] = {:.3e} at the closed-form Ω*, expected 0",
                total[(i, j)]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// log q
// ---------------------------------------------------------------------------

/// `log_density` must be a normalized density: at the mean it equals
/// `−½(log|S| + d·log 2π)`, and it must integrate to 1 (checked by Monte Carlo
/// through the identity `E_q[−log q] = entropy = ½log|2πeS|`).
#[test]
fn log_density_is_a_normalized_gaussian() {
    let omega = omega_2x2();
    let fr = FullRank::new(2);
    let phi = perturbed_phi(&fr, &omega);
    let (mean, cov) = fr.moments(&phi);

    let (at_mean, g_at_mean) = fr.log_density(&phi, mean.as_slice());
    let log_det = cov.determinant().ln();
    let want = -0.5 * (log_det + 2.0 * std::f64::consts::TAU.ln());
    assert_close(at_mean, want, 1e-10, "log q at the mean");
    for &gk in &g_at_mean {
        assert_close(gk, 0.0, 1e-9, "∂log q/∂η at the mean");
    }

    // Entropy identity: ½·log|2πe·S|.
    let entropy_want = 0.5 * (log_det + 2.0 * (std::f64::consts::TAU.ln() + 1.0));
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = StdRng::seed_from_u64(7);
    let n = 200_000;
    let mut acc = 0.0;
    for _ in 0..n {
        let eps: Vec<f64> = (0..2).map(|_| StandardNormal.sample(&mut rng)).collect();
        let eta = fr.sample(&phi, &eps);
        acc -= fr.log_density(&phi, &eta).0;
    }
    assert!(
        (acc / n as f64 - entropy_want).abs() < 5e-3,
        "entropy: MC {:.6} vs analytic {entropy_want:.6}",
        acc / n as f64
    );
}

/// `∂log q/∂η` against central FD, for both families.
#[test]
fn log_density_gradient_matches_fd() {
    let omega = omega_3x3();

    for (label, fam) in [
        (
            "full_rank",
            Box::new(FullRank::new(3)) as Box<dyn VariationalFamily>,
        ),
        ("mean_field", Box::new(MeanField::new(3))),
    ] {
        let phi = {
            let mut p = fam.init(&omega);
            for (i, v) in p.iter_mut().enumerate() {
                *v += 0.15 * ((i * 7 + 3) as f64).sin();
            }
            p
        };
        let eta = vec![0.12_f64, -0.31, 0.07];
        let (_, g) = fam.log_density(&phi, &eta);
        for (k, &gk) in g.iter().enumerate() {
            let want = central_fd(|e| fam.log_density(&phi, e).0, &eta, k);
            assert_close(gk, want, 1e-6, &format!("{label} ∂log q/∂η[{k}]"));
        }
    }
}

/// `MeanField` and `FullRank` must agree exactly whenever the full-rank factor is
/// diagonal — the two families are the same distribution there, so any
/// disagreement is a layout or convention bug in one of them.
#[test]
fn families_agree_when_full_rank_factor_is_diagonal() {
    let omega = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["A".into(), "B".into()]);
    let mf = MeanField::new(2);
    let fr = FullRank::new(2);

    let phi_mf = perturbed_phi(&mf, &omega);
    // Build the matching full-rank φ: same means, same log-SDs, zero off-diagonal.
    let mut phi_fr = vec![0.0; fr.n_params()];
    phi_fr[0] = phi_mf[0];
    phi_fr[1] = phi_mf[1];
    phi_fr[2 + tril_index(0, 0)] = phi_mf[2];
    phi_fr[2 + tril_index(1, 1)] = phi_mf[3];

    let eps = [0.6_f64, -0.9];
    let a = mf.sample(&phi_mf, &eps);
    let b = fr.sample(&phi_fr, &eps);
    for k in 0..2 {
        assert_close(a[k], b[k], 1e-12, "sample agreement");
    }

    let kl_a = mf.kl_to_normal(&phi_mf, &omega).expect("cf");
    let kl_b = fr.kl_to_normal(&phi_fr, &omega).expect("cf");
    assert_close(kl_a.value, kl_b.value, 1e-12, "KL agreement");

    let eta = [0.05_f64, -0.2];
    assert_close(
        mf.log_density(&phi_mf, &eta).0,
        fr.log_density(&phi_fr, &eta).0,
        1e-12,
        "log q agreement",
    );
}

/// One random effect is the degenerate case most likely to break an index
/// calculation, and single-η models are common. Exercise it end to end.
#[test]
fn single_eta_models_work_for_both_families() {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);

    for (label, fam) in [
        (
            "full_rank",
            Box::new(FullRank::new(1)) as Box<dyn VariationalFamily>,
        ),
        ("mean_field", Box::new(MeanField::new(1))),
    ] {
        let phi = fam.init(&omega);
        let (mean, cov) = fam.moments(&phi);
        assert_close(mean[0], 0.0, 1e-12, &format!("{label} init mean"));
        assert_close(cov[(0, 0)], 0.09, 1e-10, &format!("{label} init var"));

        let kl = fam.kl_to_normal(&phi, &omega).expect("cf");
        assert_close(kl.value, 0.0, 1e-10, &format!("{label} KL at prior"));

        // η = μ + σ·ε with σ = 0.3.
        let eta = fam.sample(&phi, &[2.0]);
        assert_close(eta[0], 0.6, 1e-10, &format!("{label} sample"));
    }
}
