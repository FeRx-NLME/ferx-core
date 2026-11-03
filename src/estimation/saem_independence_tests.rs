use super::*;

/// A 3×3 SPD covariance with off-diagonal structure, and an off-zero mean.
fn cov() -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[0.09, 0.02, -0.01, 0.02, 0.04, 0.005, -0.01, 0.005, 0.25],
    )
}

fn proposal() -> GaussianProposal {
    GaussianProposal::from_covariance(vec![0.3, -0.2, 0.1], cov()).expect("SPD")
}

/// A draw is `m + L z`, so its quadratic form is exactly `−½‖z‖²`: `quad` and `draw_into`
/// are one Gaussian, the precondition of the acceptance ratio. The draw at `z = 0` is the
/// mean, where the form is 0.
#[test]
fn a_draw_scores_its_own_standard_normal() {
    let p = proposal();
    let mut out = vec![0.0; 3];
    p.draw_into(&[0.0, 0.0, 0.0], &mut out);
    assert_eq!(out, p.mean);
    assert_eq!(p.quad(&p.mean), 0.0);
    let mut worst = 0.0f64;
    for z in [[1.0, 0.0, 0.0], [0.3, -1.7, 0.9], [-2.0, 0.5, 1.1]] {
        p.draw_into(&z, &mut out);
        let want = -0.5 * z.iter().map(|v| v * v).sum::<f64>();
        let got = p.quad(&out);
        assert!(got.is_finite());
        worst = worst.max((got - want).abs());
    }
    eprintln!("quad of a draw vs −½‖z‖²: {worst:.3e}");
    assert!(worst < 1e-13, "quad of a draw vs −½‖z‖²: {worst:.3e}");
}

/// The draws have the requested covariance: with `z = e_j` the offsets from the mean are
/// the columns `L e_j`, whose outer-product sum is `L Lᵀ`.
#[test]
fn the_draws_have_the_requested_covariance() {
    let p = proposal();
    let mut sum = DMatrix::<f64>::zeros(3, 3);
    let mut out = vec![0.0; 3];
    for j in 0..3 {
        let mut z = [0.0; 3];
        z[j] = 1.0;
        p.draw_into(&z, &mut out);
        let d: Vec<f64> = out.iter().zip(&p.mean).map(|(a, m)| a - m).collect();
        for a in 0..3 {
            for b in 0..3 {
                sum[(a, b)] += d[a] * d[b];
            }
        }
    }
    let worst = (sum - cov()).abs().max();
    eprintln!("draw covariance vs requested: {worst:.3e}");
    assert!(
        worst < 1e-15,
        "Σ L e_j e_jᵀ Lᵀ vs the covariance: {worst:.3e}"
    );
}

/// No proposal where the covariance is not positive definite, not finite, or the wrong
/// size.
#[test]
fn a_non_spd_or_misfit_covariance_has_no_proposal() {
    let indefinite = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    assert!(GaussianProposal::from_covariance(vec![0.0, 0.0], indefinite).is_none());
    let nan = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, f64::NAN, 1.0]);
    assert!(GaussianProposal::from_covariance(vec![0.0, 0.0], nan).is_none());
    assert!(GaussianProposal::from_covariance(vec![0.0], DMatrix::identity(2, 2)).is_none());
}

/// The running moments are the two-pass sample mean and covariance of the draws pushed,
/// and mode 1B waits for more than `2·n_η` of them.
#[test]
fn the_conditional_moments_are_the_sample_moments() {
    let draws: Vec<[f64; 2]> = (0..9)
        .map(|k| {
            let t = k as f64;
            [0.1 * t.sin() + 0.02 * t, 0.3 * (0.7 * t).cos() - 0.05]
        })
        .collect();
    let mut m = ConditionalMoments::new(2);
    for (k, d) in draws.iter().enumerate() {
        assert_eq!(m.proposal().is_some(), k > 4, "{k} draws");
        m.push(d);
    }
    let n = draws.len() as f64;
    let mean: Vec<f64> = (0..2)
        .map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / n)
        .collect();
    let mut cov = DMatrix::<f64>::zeros(2, 2);
    for d in &draws {
        for a in 0..2 {
            for b in 0..2 {
                cov[(a, b)] += (d[a] - mean[a]) * (d[b] - mean[b]) / (n - 1.0);
            }
        }
    }
    let p = m.proposal().expect("enough draws");
    let mut worst = 0.0f64;
    for j in 0..2 {
        worst = worst.max((p.mean[j] - mean[j]).abs());
    }
    worst = worst.max((&p.l * p.l.transpose() - cov).abs().max());
    eprintln!("running vs two-pass moments: {worst:.3e}");
    assert!(worst < 1e-14, "running vs two-pass moments: {worst:.3e}");
}

/// The independence kernel samples the subject's conditional. It runs
/// composed with the random walk, as in the SAEM E-step: per iteration one walk step and
/// one independence step, against two walk steps. Over long chains on warfarin subjects
/// (three η) the composed chain's conditional mean and variance of every η agree with a
/// reference computed outside any chain: self-normalised importance sampling from 200 000
/// iid proposal draws. The random walk itself is not the reference — at this step size it
/// mixes slowly enough to sit 0.42 posterior SD off on subject 0's third η, measured. A kernel whose
/// ratio or proposal density were off would bias the composed chain wherever it accepts,
/// and the test requires it to accept a measurable share. The proposal is mode 1B's
/// kind, deliberately not the posterior (the walk's moments, broadened and shifted).
/// Mode 1 runs the same kernel on another proposal (`mode_ones_proposal_is_the_prior`);
/// on these informative subjects its prior proposals are almost never accepted, so a chain
/// test could not see its ratio.
#[test]
fn the_independence_kernel_samples_the_conditional() {
    use super::super::{mh_steps, MhScratch};
    use rand::SeedableRng;
    let root = env!("CARGO_MANIFEST_DIR");
    let prep = crate::api::prepare_run(
        &format!("{root}/examples/warfarin.ferx"),
        Some(&format!("{root}/data/warfarin.csv")),
    )
    .unwrap();
    let model = &prep.parsed.model;
    let params = &model.default_params;
    let n_eta = model.n_eta;
    let mut worst_mean = 0.0f64;
    let mut worst_var = 0.0f64;
    for (si, subject) in prep.population.subjects.iter().take(3).enumerate() {
        let start = vec![0.0; n_eta];
        let nll0 = crate::stats::likelihood::individual_nll(
            model,
            subject,
            &params.theta,
            &start,
            &params.omega,
            &params.sigma.values,
        );
        // Moments of every η, and the independence kernel's acceptance rate.
        let chain = |prop: Option<&GaussianProposal>| -> (Vec<f64>, Vec<f64>, f64) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(11 + si as u64);
            let mut scratch = MhScratch::default();
            scratch.begin_subject(model, subject, &params.theta, n_eta);
            let mut eta = start.clone();
            let mut nll = nll0;
            let mut s1 = vec![0.0; n_eta];
            let mut s2 = vec![0.0; n_eta];
            let mut acc = 0usize;
            let (burn, n) = (2_000, 60_000);
            for k in 0..burn + n {
                let walk = |eta: &mut Vec<f64>, nll: f64, rng: &mut _, scratch: &mut _| {
                    mh_steps(
                        eta,
                        nll,
                        subject,
                        model,
                        &params.theta,
                        &params.omega,
                        &params.sigma.values,
                        0.5,
                        None,
                        rng,
                        1,
                        scratch,
                        None,
                        None,
                    )
                };
                nll = walk(&mut eta, nll, &mut rng, &mut scratch).1;
                let (a, l) = match prop {
                    Some(p) => independence_steps(
                        &mut eta,
                        nll,
                        subject,
                        model,
                        &params.theta,
                        &params.omega,
                        &params.sigma.values,
                        p,
                        &mut rng,
                        1,
                        &mut scratch,
                        None,
                    ),
                    None => walk(&mut eta, nll, &mut rng, &mut scratch),
                };
                nll = l;
                if k >= burn {
                    acc += a;
                    for j in 0..n_eta {
                        s1[j] += eta[j];
                        s2[j] += eta[j] * eta[j];
                    }
                }
            }
            let nf = n as f64;
            let m: Vec<f64> = s1.iter().map(|v| v / nf).collect();
            let v: Vec<f64> = s2.iter().zip(&m).map(|(q, mm)| q / nf - mm * mm).collect();
            (m, v, acc as f64 / nf)
        };
        let (m_r, v_r, _) = chain(None);
        let mean_shift: Vec<f64> = m_r.iter().zip(&v_r).map(|(m, v)| m + v.sqrt()).collect();
        let broad = GaussianProposal::from_covariance(
            mean_shift,
            DMatrix::from_fn(n_eta, n_eta, |a, b| if a == b { 4.0 * v_r[a] } else { 0.0 }),
        )
        .unwrap();
        // A third estimate, independent of both chains: self-normalised importance
        // sampling from iid draws of the broad proposal, `w = π/q` (no autocorrelation).
        let oracle = {
            let mut rng = rand::rngs::StdRng::seed_from_u64(99 + si as u64);
            let mut scratch = MhScratch::default();
            scratch.begin_subject(model, subject, &params.theta, n_eta);
            let mut z = vec![0.0; n_eta];
            let mut x = vec![0.0; n_eta];
            let mut lw = Vec::new();
            let mut xs = Vec::new();
            for _ in 0..200_000 {
                for v in z.iter_mut() {
                    *v = rng.sample(StandardNormal);
                }
                broad.draw_into(&z, &mut x);
                let nll = crate::stats::likelihood::individual_nll(
                    model,
                    subject,
                    &params.theta,
                    &x,
                    &params.omega,
                    &params.sigma.values,
                );
                lw.push(-nll - broad.quad(&x));
                xs.push(x.clone());
            }
            let mx = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = lw.iter().map(|l| (l - mx).exp()).collect();
            let sw: f64 = w.iter().sum();
            let m: Vec<f64> = (0..n_eta)
                .map(|j| xs.iter().zip(&w).map(|(x, w)| w * x[j]).sum::<f64>() / sw)
                .collect();
            let v: Vec<f64> = (0..n_eta)
                .map(|j| {
                    xs.iter()
                        .zip(&w)
                        .map(|(x, w)| w * (x[j] - m[j]).powi(2))
                        .sum::<f64>()
                        / sw
                })
                .collect();
            (m, v)
        };
        let (m_o, v_o) = oracle;
        eprintln!("subject {si} importance-sampled means {m_o:.4?}, walk {m_r:.4?}");
        for (name, prop) in [("mode 1B", &broad)] {
            let (m, v, a) = chain(Some(prop));
            eprintln!("subject {si} {name}: acc {a:.3}, means {m:.4?}");
            assert!(
                a > 0.02,
                "{name}: the kernel accepted {a:.4}, too rarely to be tested"
            );
            for j in 0..n_eta {
                let sd = v_o[j].sqrt();
                assert!(m[j].is_finite() && v[j].is_finite());
                worst_mean = worst_mean.max((m[j] - m_o[j]).abs() / sd);
                worst_var = worst_var.max((v[j] / v_o[j] - 1.0).abs());
            }
        }
    }
    eprintln!("worst |Δmean|/sd {worst_mean:.4}, worst |Δvar|/var {worst_var:.4}");
    // Measured 0.029 SD and 1.8% (Linux x86_64, these seeds): Monte-Carlo error of a
    // 60 000-step chain and a 200 000-draw estimate. The bounds keep 3.5× and 11× headroom
    // and are far inside what a wrong ratio produces: a proposal one SD off, accepted on
    // the target ratio alone, drags the mean toward it.
    assert!(
        worst_mean < 0.1,
        "conditional means differ by {worst_mean:.3} SD"
    );
    assert!(
        worst_var < 0.2,
        "conditional variances differ by {:.1}%",
        100.0 * worst_var
    );
}

/// Mode 1's proposal is the population distribution: its quadratic form is the prior's,
/// `−½ ηᵀΩ⁻¹η`, by an independent inverse.
#[test]
fn mode_ones_proposal_is_the_prior() {
    let omega = cov();
    let chol = omega.clone().cholesky().unwrap().l();
    let p = GaussianProposal::prior(&chol);
    let oinv = omega.try_inverse().unwrap();
    let mut worst = 0.0f64;
    for x in [[0.3, -0.2, 0.1], [1.0, -0.6, 0.4], [-0.4, 0.8, -1.2]] {
        let xv = DVector::from_column_slice(&x);
        let want = -0.5 * (xv.transpose() * &oinv * &xv)[0];
        let got = p.quad(&x);
        assert!(got.is_finite() && want.is_finite());
        worst = worst.max((got - want).abs() / (1.0 + want.abs()));
    }
    eprintln!("prior quad vs −½ηᵀΩ⁻¹η: {worst:.3e}");
    assert!(worst < 1e-13, "prior quad vs −½ηᵀΩ⁻¹η: {worst:.3e}");
}
