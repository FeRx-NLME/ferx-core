//! The gradient E-step kernels (`saem_kernel = hmc | nuts | mala`, #2986) sample the right
//! distribution: each subject's conditional `p(η | y)` — and under IOV the joint
//! `p(η, κ₁..κ_K | y)` — whatever the metric.
//!
//! Every other test of these kernels checks that a fit runs, is served, or recovers a
//! simulated truth; none of those sees a kernel that converges to the wrong distribution.
//! Here the target's moments are computed exactly, by quadrature on a dense grid over the
//! random effects of `EtaPotential::value` (the NLL the kernels' accept/reject reads, prior
//! included), and a long chain at fixed parameters must reproduce them: every mean within
//! four batch-means standard errors, every covariance entry within 10 % of its scale.
//! The fixtures are nonlinear (one-compartment IV, two or four observations), so the
//! posteriors are not Gaussian, and each kernel runs under the exact posterior's metric and
//! under a deliberately mis-scaled one (Ω's own factor): a valid kernel's target does not
//! depend on its metric.

use super::{
    nuts_step, preconditioned_hmc_step, EtaPotential, GradStep, MhScratch, HMC_EPS_GAIN,
    HMC_EPS_HI, HMC_EPS_INIT, HMC_EPS_LO, HMC_MAX_LEAPFROG, HMC_TARGET_ACCEPT, MALA_TARGET_ACCEPT,
};
use crate::parser::model_parser::parse_model_string;
use crate::types::*;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

const MODEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

const IOV_MODEL: &str = r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

/// Two observations after one dose: a posterior far from Gaussian in (η_CL, η_V).
fn subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 8.0],
        observations: vec![4.1, 1.6],
        obs_cmts: vec![1; 2],
        cens: vec![0; 2],
        ..Default::default()
    }
}

/// Two occasions of two observations each, one dose per occasion.
fn iov_subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(48.0, 100.0, 1, 0.0, false, 0.0),
        ],
        dose_occasions: vec![1, 2],
        obs_times: vec![1.0, 8.0, 49.0, 56.0],
        observations: vec![4.4, 2.0, 4.6, 1.2],
        obs_cmts: vec![1; 4],
        occasions: vec![1, 1, 2, 2],
        cens: vec![0; 4],
        ..Default::default()
    }
}

/// Exact mean and covariance of `exp(−U(x))` by quadrature: a coarse grid over the prior's
/// ±6 sd finds the bulk, then a fine grid over its mean ± 7 sd integrates it.
fn grid_moments(
    dim: usize, prior_sd: &[f64], u: &mut dyn FnMut(&[f64]) -> f64,
) -> (Vec<f64>, DMatrix<f64>) {
    let pass = |centre: &[f64], half: &[f64], n: usize, u: &mut dyn FnMut(&[f64]) -> f64| {
        let total = n.pow(dim as u32);
        let mut pts = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        for idx in 0..total {
            let mut r = idx;
            let x: Vec<f64> = (0..dim)
                .map(|d| {
                    let i = r % n;
                    r /= n;
                    centre[d] - half[d] + 2.0 * half[d] * i as f64 / (n - 1) as f64
                })
                .collect();
            vals.push(u(&x));
            pts.push(x);
        }
        assert!(vals.iter().all(|v| v.is_finite()), "the NLL is finite on the grid");
        let umin = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let w: Vec<f64> = vals.iter().map(|v| (umin - v).exp()).collect();
        let sw: f64 = w.iter().sum();
        let mean: Vec<f64> = (0..dim)
            .map(|d| pts.iter().zip(&w).map(|(p, wi)| wi * p[d]).sum::<f64>() / sw)
            .collect();
        let mut cov = DMatrix::<f64>::zeros(dim, dim);
        for (p, wi) in pts.iter().zip(&w) {
            for a in 0..dim {
                for b in 0..dim {
                    cov[(a, b)] += wi * (p[a] - mean[a]) * (p[b] - mean[b]) / sw;
                }
            }
        }
        (mean, cov)
    };
    let zero = vec![0.0; dim];
    let wide: Vec<f64> = prior_sd.iter().map(|s| 6.0 * s).collect();
    let (m0, c0) = pass(&zero, &wide, if dim > 2 { 31 } else { 81 }, u);
    let half: Vec<f64> = (0..dim).map(|d| 7.0 * c0[(d, d)].sqrt()).collect();
    pass(&m0, &half, if dim > 2 { 41 } else { 161 }, u)
}

/// A chain of `n` draws from `x0` under `step` (one transition), the step size adapted by
/// the production rule for `warmup` transitions and frozen after.
#[allow(clippy::type_complexity)]
fn chain(
    x0: &[f64], n: usize, warmup: usize, target: f64, pot: &mut EtaPotential<'_>,
    step: &mut dyn FnMut(&mut [f64], f64, f64, &mut EtaPotential<'_>) -> Option<GradStep>,
) -> Vec<Vec<f64>> {
    let mut x = x0.to_vec();
    let mut nll = pot.value(&x);
    let mut eps = HMC_EPS_INIT;
    let mut out = Vec::with_capacity(n);
    for t in 0..warmup + n {
        let g = step(&mut x, nll, eps, pot).expect("the AD engine serves the fixture");
        let (alpha, nll_new) = (g.alpha, g.nll);
        nll = nll_new;
        if t < warmup {
            eps = (eps * (HMC_EPS_GAIN * (alpha - target)).exp()).clamp(HMC_EPS_LO, HMC_EPS_HI);
        } else {
            out.push(x.clone());
        }
    }
    out
}

/// The chain's moments against the exact ones: means within four batch-means standard
/// errors (40 batches), covariances within 10 % of `sqrt(Σ_aa Σ_bb)`.
fn assert_moments(label: &str, draws: &[Vec<f64>], mean: &[f64], cov: &DMatrix<f64>) {
    let dim = mean.len();
    let n = draws.len();
    let m: Vec<f64> =
        (0..dim).map(|d| draws.iter().map(|x| x[d]).sum::<f64>() / n as f64).collect();
    let n_batch = 40;
    let bs = n / n_batch;
    for d in 0..dim {
        let bm: Vec<f64> = (0..n_batch)
            .map(|b| draws[b * bs..(b + 1) * bs].iter().map(|x| x[d]).sum::<f64>() / bs as f64)
            .collect();
        let mb = bm.iter().sum::<f64>() / n_batch as f64;
        let se = (bm.iter().map(|v| (v - mb).powi(2)).sum::<f64>()
            / (n_batch * (n_batch - 1)) as f64)
            .sqrt();
        assert!(
            (m[d] - mean[d]).abs() < 4.0 * se + 1e-3 * cov[(d, d)].sqrt(),
            "{label}: mean[{d}] {} vs exact {} (se {se:.2e})",
            m[d],
            mean[d]
        );
    }
    for a in 0..dim {
        for b in 0..dim {
            let c = draws.iter().map(|x| (x[a] - m[a]) * (x[b] - m[b])).sum::<f64>() / n as f64;
            let scale = (cov[(a, a)] * cov[(b, b)]).sqrt();
            assert!(
                (c - cov[(a, b)]).abs() < 0.1 * scale,
                "{label}: cov[{a},{b}] {c:.4e} vs exact {:.4e}",
                cov[(a, b)]
            );
        }
    }
}

fn lower_chol(c: &DMatrix<f64>) -> DMatrix<f64> {
    c.clone().cholesky().expect("SPD").l()
}

/// HMC, NUTS and MALA over (η_CL, η_V) without IOV, under the exact posterior's metric and under
/// Ω's (mis-scaled: the prior is ~3× wider than the posterior here).
#[test]
fn gradient_kernels_sample_the_exact_conditional_posterior() {
    let model = parse_model_string(MODEL).expect("parses");
    let s = subject();
    let p = &model.default_params;
    let (theta, omega, sigma) = (&p.theta, &p.omega, &p.sigma.values);
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &s, theta, 2);
    let mut pot =
        EtaPotential::new(&model, &s, theta, omega, sigma, None, None, &mut scratch, 2, false);
    let sd: Vec<f64> = (0..2).map(|d| omega.matrix[(d, d)].sqrt()).collect();
    let (mean, cov) = grid_moments(2, &sd, &mut |x| pot.value(x));
    for (metric, l) in [("posterior", lower_chol(&cov)), ("omega", omega.chol.clone())] {
        // The gradient carries from one transition to the next, as in the E-step.
        let mut rng = StdRng::seed_from_u64(2986);
        let mut grad_at = None;
        let draws =
            chain(&mean, 20_000, 1_000, HMC_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                preconditioned_hmc_step(
                    x,
                    nll,
                    &l,
                    eps,
                    HMC_MAX_LEAPFROG,
                    None,
                    &mut grad_at,
                    pot,
                    &mut rng,
                )
            });
        assert_moments(&format!("hmc, {metric} metric"), &draws, &mean, &cov);
        let mut rng = StdRng::seed_from_u64(2987);
        let draws =
            chain(&mean, 10_000, 1_000, HMC_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                nuts_step(x, nll, &l, eps, pot, &mut rng)
            });
        assert_moments(&format!("nuts, {metric} metric"), &draws, &mean, &cov);
        let mut rng = StdRng::seed_from_u64(29870);
        let mut grad_at = None;
        let draws =
            chain(&mean, 40_000, 1_000, MALA_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                preconditioned_hmc_step(x, nll, &l, eps, 1, Some(1), &mut grad_at, pot, &mut rng)
            });
        assert_moments(&format!("mala, {metric} metric"), &draws, &mean, &cov);
    }
}

/// The joint IOV move over `[η_CL, κ₁, κ₂]`: HMC, NUTS and MALA reproduce the exact joint
/// posterior, η–κ correlation included, under its metric and under the block prior one.
#[test]
fn the_joint_iov_move_samples_the_exact_joint_posterior() {
    let model = parse_model_string(IOV_MODEL).expect("parses");
    let s = iov_subject();
    let p = &model.default_params;
    let (theta, omega, sigma) = (&p.theta, &p.omega, &p.sigma.values);
    let omega_iov = p.omega_iov.as_ref().expect("a κ");
    let kappas = vec![vec![0.0]; 2];
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &s, theta, 1);
    let mut pot = EtaPotential::new(
        &model,
        &s,
        theta,
        omega,
        sigma,
        None,
        Some((kappas.as_slice(), omega_iov)),
        &mut scratch,
        1,
        true,
    );
    let w = omega_iov.matrix[(0, 0)].sqrt();
    let (mean, cov) = grid_moments(3, &[omega.matrix[(0, 0)].sqrt(), w, w], &mut |x| pot.value(x));
    assert!(
        cov[(0, 1)] / (cov[(0, 0)] * cov[(1, 1)]).sqrt() < -0.2,
        "premise: η and κ are correlated a posteriori"
    );
    let mut prior = DMatrix::<f64>::zeros(3, 3);
    prior[(0, 0)] = omega.chol[(0, 0)];
    prior[(1, 1)] = omega_iov.chol[(0, 0)];
    prior[(2, 2)] = omega_iov.chol[(0, 0)];
    for (metric, l) in [("posterior", lower_chol(&cov)), ("block prior", prior)] {
        // The gradient carries from one transition to the next, as in the E-step.
        let mut rng = StdRng::seed_from_u64(2988);
        let mut grad_at = None;
        let draws =
            chain(&mean, 20_000, 1_000, HMC_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                preconditioned_hmc_step(
                    x,
                    nll,
                    &l,
                    eps,
                    HMC_MAX_LEAPFROG,
                    None,
                    &mut grad_at,
                    pot,
                    &mut rng,
                )
            });
        assert_moments(&format!("joint hmc, {metric} metric"), &draws, &mean, &cov);
        let mut rng = StdRng::seed_from_u64(2989);
        let draws =
            chain(&mean, 10_000, 1_000, HMC_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                nuts_step(x, nll, &l, eps, pot, &mut rng)
            });
        assert_moments(&format!("joint nuts, {metric} metric"), &draws, &mean, &cov);
        let mut rng = StdRng::seed_from_u64(29890);
        let mut grad_at = None;
        let draws =
            chain(&mean, 40_000, 1_000, MALA_TARGET_ACCEPT, &mut pot, &mut |x, nll, eps, pot| {
                preconditioned_hmc_step(x, nll, &l, eps, 1, Some(1), &mut grad_at, pot, &mut rng)
            });
        assert_moments(&format!("joint mala, {metric} metric"), &draws, &mean, &cov);
    }
}

/// The kernel schedule (#2986): each preset's counts, an `isample_*` the caller set overriding
/// its preset entry and leaving the others, the f-SAEM count running as `m1` outside its
/// scope, the legacy presets declining, and the order and wording the fit reports.
#[test]
fn the_kernel_schedule_resolves_presets_and_overrides() {
    use super::KernelSchedule;
    use crate::types::{FitOptions, SaemKernel};
    let with = |kernel: SaemKernel, edit: &dyn Fn(&mut FitOptions)| {
        let mut o = FitOptions { saem_kernel: kernel, ..FitOptions::default() };
        edit(&mut o);
        o
    };
    let nm = KernelSchedule::resolve(&with(SaemKernel::Nonmem, &|_| {}), true).expect("counted");
    assert_eq!(nm.describe(), "m1 2, m1b 2, m2 2, m3 2");
    let hmc = KernelSchedule::resolve(&with(SaemKernel::Hmc, &|_| {}), true).expect("counted");
    assert_eq!(hmc.describe(), "imh 1, hmc 1");
    assert!(hmc.gradient());
    let nuts = KernelSchedule::resolve(&with(SaemKernel::Nuts, &|_| {}), true).expect("counted");
    assert_eq!(nuts.describe(), "imh 1, nuts 1");
    let mala = KernelSchedule::resolve(&with(SaemKernel::Mala, &|_| {}), true).expect("counted");
    assert_eq!(mala.describe(), "imh 1, mala 3");
    assert!(mala.gradient());
    // MALA beside the HMC preset, in the order the E-step runs them.
    let both = with(SaemKernel::Hmc, &|o| o.saem_isample_mala = Some(2));
    assert_eq!(
        KernelSchedule::resolve(&both, true).expect("counted").describe(),
        "imh 1, hmc 1, mala 2"
    );
    // Outside the f-SAEM proposal's scope the preset's jump runs as M1.
    let out = KernelSchedule::resolve(&with(SaemKernel::Hmc, &|_| {}), false).expect("counted");
    assert_eq!(out.describe(), "m1 1, hmc 1");
    // An override changes its own entry only.
    let mix = with(SaemKernel::Hmc, &|o| {
        o.saem_isample_m3 = Some(1);
    });
    assert_eq!(
        KernelSchedule::resolve(&mix, true).expect("counted").describe(),
        "imh 1, hmc 1, m3 1"
    );
    // Outside the f-SAEM proposal's scope its count runs as M1.
    assert_eq!(
        KernelSchedule::resolve(&mix, false).expect("counted").describe(),
        "m1 1, hmc 1, m3 1"
    );
    // A preset entry can be switched off; all of them off is empty.
    let off = with(SaemKernel::Nonmem, &|o| {
        o.saem_isample_m1 = Some(0);
        o.saem_isample_m1b = Some(0);
        o.saem_isample_m2 = Some(0);
        o.saem_isample_m3 = Some(0);
    });
    assert!(KernelSchedule::resolve(&off, true).expect("counted").is_empty());
    // The legacy kernels keep their own path.
    assert!(KernelSchedule::resolve(&with(SaemKernel::RandomWalk, &|_| {}), true).is_none());
    assert!(KernelSchedule::resolve(&with(SaemKernel::Imh, &|_| {}), true).is_none());
}

/// Value/gradient fusion under IOV (#2986): the value the joint `[η, κ]` potential takes from
/// the AD pass is the native IOV NLL (`individual_nll_iov…`) every other kernel scores with,
/// at points across the posterior — the gradient kernels' accept/reject reads one, the
/// random-walk kernels after them the other.
#[test]
fn the_fused_iov_value_is_the_native_iov_nll() {
    let model = parse_model_string(IOV_MODEL).expect("parses");
    let s = iov_subject();
    let p = &model.default_params;
    let (theta, omega, sigma) = (&p.theta, &p.omega, &p.sigma.values);
    let omega_iov = p.omega_iov.as_ref().expect("a κ");
    let kappas = vec![vec![0.0]; 2];
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &s, theta, 1);
    let mut pot = EtaPotential::new(
        &model,
        &s,
        theta,
        omega,
        sigma,
        None,
        Some((kappas.as_slice(), omega_iov)),
        &mut scratch,
        1,
        true,
    );
    for x in [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [-0.5, 0.4, -0.3], [0.8, 0.1, 0.6]] {
        let native = pot.value(&x);
        let (_, fused) = pot.grad(&x, true).expect("served");
        // The AD pass itself scores it (no silent fallback to the native objective).
        let mut direct = None;
        crate::estimation::inner_optimizer::analytic_eta_nll_value_gradient_iov(
            &model,
            &s,
            theta,
            &x,
            omega,
            omega_iov,
            sigma,
            None,
            Some(&mut direct),
        )
        .expect("served");
        assert_eq!(direct, Some(fused), "at {x:?}: the potential reads the fused value");
        assert!(
            (fused - native).abs() <= 1e-12 * native.abs().max(1.0),
            "at {x:?}: fused {fused} vs native {native}"
        );
    }
}
