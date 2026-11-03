//! Unit tests for the f-SAEM independent-MH E-step kernel (#1449).
//!
//! The load-bearing test is [`imh_chain_matches_a_long_random_walk`]: the IMH
//! proposal is **not** symmetric, so the acceptance ratio carries a `q`-ratio the
//! random-walk kernels do not have, and getting that term wrong yields a chain
//! that still runs, still accepts and still looks plausible — it just targets the
//! wrong distribution. That test pins the sampled conditional against the random
//! walk (which is known-correct), and its sibling
//! [`dropping_the_q_ratio_visibly_biases_the_chain`] feeds the assertion the exact
//! defect it exists to catch, so a green result is evidence and not a tautology.

use super::*;
use crate::estimation::saem::{
    imh_decline_warning, imh_kernel_decision, imh_knobs, imh_supported_model, mh_steps,
    obs_per_subject_per_eta, run_saem, rw_scale_has_evidence, saem_sampler_summary_for,
    ImhDecision, ImhKnobs, MhScratch, LAST_IMH_SNAPSHOT, LAST_RW_BLOCK_CALLS,
};
use crate::parser::model_parser::parse_model_string;
use crate::stats::likelihood::individual_nll;
use crate::types::{
    DoseEvent, EstimationMethod, FitOptions, OmegaMatrix, Population, SaemKernel, Subject,
};
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;

const IV_MODEL: &str = "\
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The same structure under an additive error — a residual variance that does
/// not depend on η, so the interaction term vanishes and `H̃` is the exact
/// Hessian of the individual objective at a zero-residual point.
const IV_MODEL_ADDITIVE: &str = "\
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(ADD_ERR)
";

const OBS_TIMES: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

fn iv_subject(observations: Vec<f64>) -> Subject {
    let n = OBS_TIMES.len();
    assert_eq!(n, observations.len());
    Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: OBS_TIMES.to_vec(),
        observations,
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Default::default()
    }
}

fn omega2(v1: f64, v2: f64) -> OmegaMatrix {
    OmegaMatrix::from_diagonal(&[v1, v2], vec!["ETA_CL".into(), "ETA_V".into()])
}

fn predictions(model: &CompiledModel, subject: &Subject, theta: &[f64], eta: &[f64]) -> Vec<f64> {
    let mut scratch = crate::pk::EventPkParams::with_capacity_for(subject);
    crate::pk::compute_predictions_with_tv_into(model, subject, theta, eta, &mut scratch)
}

/// The stress fixture: observations that put the conditional mode at
/// `η_CL ≈ 1.5`, on a surface far flatter than its own curvature at the mode.
fn stress_subject() -> Subject {
    iv_subject(vec![4.2, 3.6, 2.9, 1.8, 0.75, 0.33])
}

/// A subject whose observations are the model's own predictions at a modest
/// `η = (0.25, −0.15)` with a ±4 % ripple, so the conditional is informative,
/// centred away from zero and close enough to Gaussian for two samplers to be
/// compared on it. (The stress fixture is a bad invariance oracle: two correct
/// samplers genuinely disagree about it at any affordable chain length.)
fn well_conditioned_subject() -> Subject {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let mut subject = iv_subject(vec![0.0; 6]);
    let preds = predictions(&model, &subject, &theta, &[0.25, -0.15]);
    assert!(preds.iter().all(|p| p.is_finite() && *p > 0.0));
    let ripple = [1.04, 0.97, 1.02, 0.96, 1.03, 0.98];
    subject.observations = preds.iter().zip(ripple).map(|(p, r)| p * r).collect();
    subject
}

fn population_of(subject: &Subject, n: usize) -> Population {
    Population {
        subjects: (0..n)
            .map(|i| {
                let mut s = subject.clone();
                s.id = format!("{}", i + 1);
                s
            })
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn build(
    model: &CompiledModel,
    subject: &Subject,
    warm: &[f64],
    steps: usize,
    defensive: f64,
    counters: &ImhCounters,
) -> ImhProposal {
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    refresh_proposal(
        model,
        subject,
        &theta,
        &omega2(0.09, 0.09),
        &sigma,
        warm,
        steps,
        defensive,
        counters,
    )
    .expect("analytic 1-cpt IV subject must be in IMH scope")
}

/// Running mean / variance of a scalar stream.
#[derive(Default, Clone, Copy)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn push(&mut self, x: f64) {
        assert!(x.is_finite(), "non-finite sample fed to Moments");
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }
    fn sd(&self) -> f64 {
        (self.m2 / self.n).sqrt()
    }
}

// ---------------------------------------------------------------------------
// The proposal's linear algebra
// ---------------------------------------------------------------------------

/// `scale_matrix()` must be `c · P⁻¹` for the precision `P` it was built from —
/// the stored Cholesky factorises the *precision*, not the covariance. Inverting
/// the two gives a proposal with reciprocal variances.
#[test]
fn scale_matrix_is_the_inflated_inverse_precision() {
    let p = DMatrix::from_row_slice(2, 2, &[40.0, 6.0, 6.0, 11.0]);
    let prop = ImhProposal::from_precision(vec![0.1, -0.2], &p, 1.5, 0.0, None).unwrap();
    let got = prop.scale_matrix();
    let want = p.clone().try_inverse().unwrap() * 1.5;
    for i in 0..2 {
        for j in 0..2 {
            let d = (got[(i, j)] - want[(i, j)]).abs();
            // Two 2×2 inversions; realised worst |Δ| 3.5e-18.
            assert!(d.is_finite() && d < 1e-14, "({i},{j}): |Δ| = {d:e}");
        }
    }
}

/// The Mahalanobis form in the acceptance ratio must be the quadratic form of the
/// *inflated* proposal precision, `(η−η̂)ᵀ(P/c)(η−η̂)`. An inflation applied to
/// the draw but not to `maha` leaves a sampler whose `q` is not what it draws.
#[test]
fn maha_is_the_inflated_quadratic_form() {
    let p = DMatrix::from_row_slice(2, 2, &[40.0, 6.0, 6.0, 11.0]);
    let eta_hat = vec![0.1, -0.2];
    let c = 1.3;
    let prop = ImhProposal::from_precision(eta_hat.clone(), &p, c, 0.0, None).unwrap();
    for eta in [[0.0, 0.0], [0.5, 0.4], [-1.0, 2.0], [0.1, -0.2]] {
        let d = DVector::from_vec(vec![eta[0] - eta_hat[0], eta[1] - eta_hat[1]]);
        let want = d.dot(&(&p * &d)) / c;
        let got = prop.maha_for_test(&eta);
        assert!(
            (got - want).abs() < 1e-12,
            "maha({eta:?}) = {got}, want {want}"
        );
    }
}

/// Draws must realise the advertised scale matrix — the only check that the
/// triangular solve in `draw_into` solves `Lᵀx = z` and not `Lx = z`; both run,
/// both produce a Gaussian, and only one has covariance `P⁻¹`.
#[test]
fn draws_realise_the_scale_matrix() {
    let p = DMatrix::from_row_slice(2, 2, &[40.0, 6.0, 6.0, 11.0]);
    let eta_hat = vec![0.1, -0.2];
    let prop = ImhProposal::from_precision(eta_hat.clone(), &p, IMH_INFLATE, 0.0, None).unwrap();
    let want = prop.scale_matrix();

    let mut rng = StdRng::seed_from_u64(7);
    let n = 400_000;
    let (mut s0, mut s1, mut s00, mut s11, mut s01) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for _ in 0..n {
        let x = prop.draw_for_test(&mut rng);
        assert!(x[0].is_finite() && x[1].is_finite(), "non-finite draw");
        let (d0, d1) = (x[0] - eta_hat[0], x[1] - eta_hat[1]);
        s0 += d0;
        s1 += d1;
        s00 += d0 * d0;
        s11 += d1 * d1;
        s01 += d0 * d1;
    }
    let nf = n as f64;
    let (m0, m1) = (s0 / nf, s1 / nf);
    let cov = [s00 / nf - m0 * m0, s01 / nf - m0 * m1, s11 / nf - m1 * m1];
    let want3 = [want[(0, 0)], want[(0, 1)], want[(1, 1)]];
    // Realised worst relative error 2.5e-3 (seed 7), at the Monte-Carlo floor of
    // ~√(2/n) = 2.2e-3 for a variance at n = 400k. Bound 3e-2 (12×), while a
    // transposed solve swaps a 0.028 variance for a 0.10 one (a 260 % error).
    for (got, w) in cov.iter().zip(want3.iter()) {
        let rel = (got - w).abs() / w.abs().max(1e-12);
        assert!(
            rel < 3e-2,
            "draw covariance {got:.6} vs {w:.6} (rel {rel:.3e})"
        );
    }
}

/// The mixture `log q` must be a normalised density. With the defensive
/// component on, the two components' normalisers no longer cancel, so a wrong
/// `log_norm_main` or `log_norm_prior` gives a self-consistent-looking but wrong
/// acceptance ratio. Checked by importance sampling: `E_q[p_ref(η)/q(η)] = 1`
/// for a normalised reference `p_ref` — here the prior component itself.
#[test]
fn defensive_mixture_density_is_normalised() {
    let p = DMatrix::from_row_slice(2, 2, &[40.0, 6.0, 6.0, 11.0]);
    let omega = OmegaMatrix::from_diagonal(&[0.2, 0.5], vec!["A".into(), "B".into()]);
    let mix =
        ImhProposal::from_precision(vec![0.1, -0.2], &p, IMH_INFLATE, 0.3, Some(&omega)).unwrap();
    // Reference: the closed-form prior density, written out here rather than
    // read from the proposal, so it is not a second copy of what is under test.
    let log_prior_density = |eta: &[f64]| -> f64 {
        // log N(η; 0, Ω) without the shared −(p/2)·log 2π, the same convention
        // `log_q` uses.
        -0.5 * omega.log_det - 0.5 * quad_form(&omega.inv, eta)
    };
    let mut rng = StdRng::seed_from_u64(23);
    let mut acc = Moments::default();
    for _ in 0..300_000 {
        let x = mix.draw_for_test(&mut rng);
        let w = (log_prior_density(&x) - mix.log_q_for_test(&x)).exp();
        assert!(w.is_finite(), "non-finite importance weight");
        acc.push(w);
    }
    // w ≤ 1/α = 3.33, so the estimator is bounded; realised 0.9979 at 300k draws
    // (seed 23). Bound 0.02 (~10× the realised error).
    assert!(
        (acc.mean - 1.0).abs() < 0.02,
        "mixture density not normalised: E_q[p/q] = {:.4}, want 1",
        acc.mean
    );
}

/// A precision the Cholesky rejects must be repaired, not dropped: `H̃ ⪰ Ω⁻¹` in
/// exact arithmetic, so an indefinite matrix here is numerical, and `None` would
/// silently demote the subject to the random walk forever.
#[test]
fn eigen_floor_repairs_an_indefinite_precision() {
    let bad = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]); // eigenvalues 3, −1
    assert!(
        bad.clone().cholesky().is_none(),
        "fixture is not indefinite"
    );
    let l = floored_cholesky(&bad).expect("eigen floor must repair an indefinite precision");
    let mut vals: Vec<f64> = (&l * l.transpose())
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .collect();
    vals.sort_by(f64::total_cmp);
    assert!(
        vals[0] > 0.0,
        "repaired precision is still not PD: {vals:?}"
    );
    assert!(
        (vals[1] - 3.0).abs() < 1e-9,
        "repair moved the healthy eigenvalue: {vals:?}"
    );
}

// ---------------------------------------------------------------------------
// The curvature: Dual2 Jacobian + assembly against finite differences
// ---------------------------------------------------------------------------

/// Central finite-difference Jacobian `∂f_j/∂η_a` of the production predictor.
fn fd_jacobian(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> DMatrix<f64> {
    let n_obs = subject.obs_times.len();
    let h = 1e-6;
    let mut jac = DMatrix::zeros(n_obs, eta.len());
    for a in 0..eta.len() {
        let (mut up, mut dn) = (eta.to_vec(), eta.to_vec());
        up[a] += h;
        dn[a] -= h;
        let (fu, fd) = (
            predictions(model, subject, theta, &up),
            predictions(model, subject, theta, &dn),
        );
        for j in 0..n_obs {
            jac[(j, a)] = (fu[j] - fd[j]) / (2.0 * h);
        }
    }
    jac
}

/// Under an additive error, at a point where every residual is zero, `H̃` is the
/// **exact** Hessian of the individual objective: `Σ f'f'ᵀ/V + Ω⁻¹` (the `r·f''`
/// term vanishes with `r`, and `V` does not depend on η). So the curvature the
/// proposal is built on — the `Dual2` Jacobian fed through
/// `conditional_eta_precision` — must match a central-FD Hessian of
/// `individual_nll`, the `f64` value path. Pins the derivative against the value
/// path, per the sensitivity-parity rule.
#[test]
fn conditional_precision_matches_the_fd_hessian_at_zero_residual() {
    let model = parse_model_string(IV_MODEL_ADDITIVE).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let eta_star = [0.25, -0.15];
    let mut subject = iv_subject(vec![0.0; 6]);
    subject.observations = predictions(&model, &subject, &theta, &eta_star);

    // The production pair: `J` and the `f` it rides with, from one dual sweep.
    let (jac, f) = super::eta_jacobian_and_preds(&model, &subject, &theta, &eta_star)
        .expect("analytic Jacobian in scope");
    let h_tilde = crate::stats::likelihood::conditional_eta_precision(
        &model, &subject, &theta, &eta_star, &f, &jac, &omega, &sigma, true,
    )
    .expect("curvature in scope");

    let nll = |e: &[f64]| individual_nll(&model, &subject, &theta, e, &omega, &sigma);
    let h = 1e-4;
    let mut fd = DMatrix::zeros(2, 2);
    for a in 0..2 {
        for b in 0..2 {
            let at = |da: f64, db: f64| {
                let mut e = eta_star.to_vec();
                e[a] += da;
                e[b] += db;
                nll(&e)
            };
            fd[(a, b)] = (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h);
        }
    }
    // Realised worst relative error 1.1e-8 (FD truncation at h = 1e-4). Bound
    // 1e-6: ~90× headroom, and far below what a dropped Ω⁻¹ (11.1 on each
    // diagonal entry) or a mis-scaled HᵀR⁻¹H would produce.
    for a in 0..2 {
        for b in 0..2 {
            let (g, w) = (h_tilde[(a, b)], fd[(a, b)]);
            assert!(g.is_finite() && w.is_finite(), "non-finite curvature entry");
            let rel = (g - w).abs() / w.abs();
            assert!(
                rel < 1e-6,
                "H̃[{a},{b}] = {g} vs FD Hessian {w} (rel {rel:e})"
            );
        }
    }
}

/// Under a proportional error the interaction term is live, and both halves of
/// `H̃` have a closed form in the FD Jacobian `J` and the predictions `f`:
/// `V_j = σ²f_j²`, so `HᵀR⁻¹H = Σ J_ja J_jb / (σ² f_j²)` and the interaction half
/// `½ Σ (∂V/∂η_a)(∂V/∂η_b)/V² = 2 Σ J_ja J_jb / f_j²`. Each half is asserted
/// separately (`interaction = false` isolates the first), so a wrong ½ on
/// `C̃ᵀC̃`, a dropped interaction term or a σ read as an SD each reddens its own
/// line.
#[test]
fn conditional_precision_halves_match_their_closed_forms_under_proportional_error() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let subject = well_conditioned_subject();
    let eta = [0.2, -0.1];
    let n_obs = subject.obs_times.len();
    let (jac_dual, f_dual) = super::eta_jacobian_and_preds(&model, &subject, &theta, &eta)
        .expect("analytic Jacobian in scope");
    let jac_fd = fd_jacobian(&model, &subject, &theta, &eta);
    // The closed forms below use the value path's `f`; the curvature gets the
    // dual sweep's. Equal to the last bit is not required, only to the bound.
    let f = predictions(&model, &subject, &theta, &eta);
    // `sigma_values` carries SDs: `PROP_ERR ~ 0.04` is a variance, stored as 0.2.
    let sigma2 = sigma[0] * sigma[0];

    let with = crate::stats::likelihood::conditional_eta_precision(
        &model, &subject, &theta, &eta, &f_dual, &jac_dual, &omega, &sigma, true,
    )
    .unwrap();
    let without = crate::stats::likelihood::conditional_eta_precision(
        &model, &subject, &theta, &eta, &f_dual, &jac_dual, &omega, &sigma, false,
    )
    .unwrap();

    for a in 0..2 {
        for b in 0..2 {
            let hrh: f64 = (0..n_obs)
                .map(|j| jac_fd[(j, a)] * jac_fd[(j, b)] / (sigma2 * f[j] * f[j]))
                .sum();
            let inter: f64 = (0..n_obs)
                .map(|j| 2.0 * jac_fd[(j, a)] * jac_fd[(j, b)] / (f[j] * f[j]))
                .sum();
            let want_without = hrh + omega.inv[(a, b)];
            let want_inter = inter;
            let got_inter = with[(a, b)] - without[(a, b)];
            // Realised worst relative errors 9.4e-11 for both halves — the FD
            // Jacobian's truncation. Bound 1e-8 (~100×).
            let r1 = (without[(a, b)] - want_without).abs() / want_without.abs();
            let r2 = (got_inter - want_inter).abs() / want_inter.abs();
            assert!(
                r1.is_finite() && r1 < 1e-8,
                "HᵀR⁻¹H+Ω⁻¹ [{a},{b}] rel {r1:e}"
            );
            assert!(r2.is_finite() && r2 < 1e-8, "½C̃ᵀC̃ [{a},{b}] rel {r2:e}");
        }
    }
}

// ---------------------------------------------------------------------------
// The kernel targets the right distribution
// ---------------------------------------------------------------------------

fn chain_moments_imh(n_steps: usize, seed: u64, drop_q_ratio: bool) -> (Moments, Moments) {
    let model = parse_model_string(IV_MODEL).expect("parse");
    chain_moments_imh_on(
        &model,
        &well_conditioned_subject(),
        n_steps,
        seed,
        drop_q_ratio,
    )
}

fn chain_moments_imh_on(
    model: &CompiledModel,
    subject: &Subject,
    n_steps: usize,
    seed: u64,
    drop_q_ratio: bool,
) -> (Moments, Moments) {
    let (model, subject) = (model.clone(), subject.clone());
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let counters = ImhCounters::default();
    let prop = build(&model, &subject, &[0.0, 0.0], 30, 0.05, &counters);

    let mut eta = vec![0.0, 0.0];
    let mut nll = individual_nll(&model, &subject, &theta, &eta, &omega, &sigma);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &subject, &theta, 2);
    let (mut m0, mut m1) = (Moments::default(), Moments::default());

    for step in 0..n_steps {
        if drop_q_ratio {
            // The exact defect the correct kernel exists to avoid: treat the
            // independent proposal as if it were symmetric.
            let eta_prop = prop.draw_for_test(&mut rng);
            let nll_prop = scratch.score(&model, &subject, &theta, &eta_prop, &omega, &sigma, None);
            let log_u: f64 = rng.random::<f64>().ln();
            if log_u < nll - nll_prop {
                eta = eta_prop;
                nll = nll_prop;
            }
        } else {
            let (_acc, nll_new) = imh_steps(
                &mut eta,
                nll,
                &prop,
                &subject,
                &model,
                &theta,
                &omega,
                &sigma,
                &mut rng,
                1,
                &mut scratch,
                None,
                &counters,
            );
            nll = nll_new;
        }
        if step >= n_steps / 10 {
            m0.push(eta[0]);
            m1.push(eta[1]);
        }
    }
    (m0, m1)
}

fn chain_moments_rw(n_steps: usize, seed: u64) -> (Moments, Moments) {
    let model = parse_model_string(IV_MODEL).expect("parse");
    chain_moments_rw_on(&model, &well_conditioned_subject(), n_steps, seed, 0.25)
}

fn chain_moments_rw_on(
    model: &CompiledModel,
    subject: &Subject,
    n_steps: usize,
    seed: u64,
    rw_scale: f64,
) -> (Moments, Moments) {
    let (model, subject) = (model.clone(), subject.clone());
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);

    let mut eta = vec![0.0, 0.0];
    let mut nll = individual_nll(&model, &subject, &theta, &eta, &omega, &sigma);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &subject, &theta, 2);
    let (mut m0, mut m1) = (Moments::default(), Moments::default());
    for step in 0..n_steps {
        // Small scale: the prior-shaped move is too wide for this conditional,
        // which is exactly the phenomenon f-SAEM addresses.
        let (_acc, nll_new) = mh_steps(
            &mut eta,
            nll,
            &subject,
            &model,
            &theta,
            &omega,
            &sigma,
            rw_scale,
            None,
            &mut rng,
            1,
            &mut scratch,
            None,
            None,
        );
        nll = nll_new;
        if step >= n_steps / 10 {
            m0.push(eta[0]);
            m1.push(eta[1]);
        }
    }
    (m0, m1)
}

/// The IMH kernel must leave `p(η | y)` invariant. Pinned against a long
/// random-walk chain on the *same* subject and Ω — the random walk is the
/// already-validated sampler, so agreement here is agreement with the
/// conditional and not with a second copy of the IMH formula.
#[test]
fn imh_chain_matches_a_long_random_walk() {
    let (r0, r1) = chain_moments_rw(400_000, 11);
    let (i0, i1) = chain_moments_imh(40_000, 11, false);
    // Conditional SDs here: η_CL ≈ 0.200, η_V ≈ 0.104. Realised (seed 11):
    // |Δmean| 6.2e-3 / 2.4e-3, |Δsd|/sd 3.5e-3 / 4.4e-3; with the q-ratio
    // dropped the worst |Δsd|/sd is 0.33 (the control below). Bounds: mean 2e-2,
    // sd 4e-2 — ≥3× the realised error and 8× below the defect.
    // The SD check is the load-bearing one: the biased chain's η_V *mean* barely
    // moves, while both coordinates' SDs collapse by a quarter.
    for (imh, rw, name) in [(i0, r0, "ETA_CL"), (i1, r1, "ETA_V")] {
        assert!(imh.mean.is_finite() && rw.mean.is_finite() && rw.sd() > 0.0);
        assert!(
            (imh.mean - rw.mean).abs() < 2e-2,
            "{name}: IMH mean {:.6} vs RW mean {:.6}",
            imh.mean,
            rw.mean
        );
        let rel_sd = (imh.sd() - rw.sd()).abs() / rw.sd();
        assert!(
            rel_sd < 4e-2,
            "{name}: IMH sd {:.6} vs RW sd {:.6}",
            imh.sd(),
            rw.sd()
        );
    }
}

/// Feeds the assertion above the defect it exists to catch: with the `q`-ratio
/// dropped the chain converges to `p(η|y)·q(η)` — an over-concentrated
/// conditional. If this ever goes green, the test above has stopped testing.
#[test]
fn dropping_the_q_ratio_visibly_biases_the_chain() {
    let (r0, r1) = chain_moments_rw(400_000, 11);
    let (b0, b1) = chain_moments_imh(40_000, 11, true);
    let worst = [(b0, r0), (b1, r1)]
        .iter()
        .map(|(b, r)| {
            assert!(b.sd().is_finite() && r.sd() > 0.0);
            (b.sd() - r.sd()).abs() / r.sd()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst > 0.15,
        "dropping the q-ratio did not move the sampled conditional (worst sd rel {worst:.3}) \
         — `imh_chain_matches_a_long_random_walk` is not testing the acceptance ratio"
    );
}

/// The premise, asserted rather than assumed: on an informative subject the
/// independent proposal accepts far more often than the prior-shaped block move
/// at the same number of proposals.
#[test]
fn imh_accepts_far_more_often_than_the_prior_shaped_block_move() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let subject = stress_subject();
    let counters = ImhCounters::default();
    let prop = build(&model, &subject, &[0.0, 0.0], 30, 0.05, &counters);
    let mut scratch = MhScratch::default();
    scratch.begin_subject(&model, &subject, &theta, 2);

    let n = 4_000;
    let mut eta = prop.eta_hat.clone();
    let nll0 = individual_nll(&model, &subject, &theta, &eta, &omega, &sigma);
    let mut rng = StdRng::seed_from_u64(3);
    let (acc_imh, _) = imh_steps(
        &mut eta,
        nll0,
        &prop,
        &subject,
        &model,
        &theta,
        &omega,
        &sigma,
        &mut rng,
        n,
        &mut scratch,
        None,
        &counters,
    );
    let mut eta_rw = prop.eta_hat.clone();
    let mut rng = StdRng::seed_from_u64(3);
    let (acc_rw, _) = mh_steps(
        &mut eta_rw,
        nll0,
        &subject,
        &model,
        &theta,
        &omega,
        &sigma,
        1.0,
        None,
        &mut rng,
        n,
        &mut scratch,
        None,
        None,
    );
    let (r_imh, r_rw) = (acc_imh as f64 / n as f64, acc_rw as f64 / n as f64);
    // Realised (seed 3): IMH 0.695, RW 0.178. Not ~1 because this conditional is
    // far flatter than its curvature at the mode and 5 % of proposals are prior
    // draws. Bounds: 0.55 absolute, and 3× the random walk (realised 3.9×).
    assert!(r_imh > 0.55, "IMH acceptance {r_imh:.3}");
    assert!(
        r_imh > 3.0 * r_rw,
        "IMH {r_imh:.3} not materially above RW {r_rw:.3}"
    );
    // And the counters saw exactly those proposals.
    let snap = counters.snapshot();
    assert_eq!((snap.proposals, snap.accepts), (n as u64, acc_imh as u64));
}

/// The stress fixture as its own test: a pure `N(η̂, c·Γ)` independence sampler
/// is *correctly* stuck at `η = 0` (`log α ≈ −278`), and the defensive prior
/// component is what unsticks it. Pins the failure mode and the guard together.
#[test]
fn imh_survives_a_conditional_far_flatter_than_its_mode_curvature() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let subject = stress_subject();
    let counters = ImhCounters::default();

    let probe = build(&model, &subject, &[0.0, 0.0], 30, 0.0, &counters);
    // Assert the straddle so this cannot quietly become an easy conditional.
    assert!(
        probe.eta_hat[0] > 1.0,
        "mode no longer far from zero: {:?}",
        probe.eta_hat
    );
    assert!(
        probe.maha_for_test(&[0.0, 0.0]) > 100.0,
        "start no longer far out"
    );

    let run = |defensive: f64| {
        let prop = build(&model, &subject, &[0.0, 0.0], 30, defensive, &counters);
        let mut eta = vec![0.0, 0.0];
        let nll = individual_nll(&model, &subject, &theta, &eta, &omega, &sigma);
        let mut rng = StdRng::seed_from_u64(5);
        let mut scratch = MhScratch::default();
        scratch.begin_subject(&model, &subject, &theta, 2);
        let (acc, _) = imh_steps(
            &mut eta,
            nll,
            &prop,
            &subject,
            &model,
            &theta,
            &omega,
            &sigma,
            &mut rng,
            200,
            &mut scratch,
            None,
            &counters,
        );
        (acc, eta)
    };
    let (acc_off, eta_off) = run(0.0);
    let (acc_on, eta_on) = run(0.05);
    assert_eq!(
        acc_off, 0,
        "a pure Laplace sampler is no longer stuck here ({acc_off}/200)"
    );
    assert_eq!(eta_off, vec![0.0, 0.0]);
    assert!(
        acc_on > 100,
        "the defensive mixture did not unstick the chain: {acc_on}/200"
    );
    assert!(
        (eta_on[0] - probe.eta_hat[0]).abs() < 0.5,
        "the defensive chain did not reach the mode: η = {eta_on:?}, η̂ = {:?}",
        probe.eta_hat
    );
}

// ---------------------------------------------------------------------------
// The Gauss-Newton MAP refresh
// ---------------------------------------------------------------------------

/// From a cold start, 5 GN steps must reach a stationary point of the objective
/// — checked against the objective's own gradient, not against another search.
#[test]
fn gn_map_reaches_a_stationary_point() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let subject = well_conditioned_subject();
    let grad = |e: &[f64]| {
        let g = analytic_eta_nll_gradient(&model, &subject, &theta, e, &omega, &sigma).unwrap();
        g.iter().map(|x| x * x).sum::<f64>().sqrt()
    };
    // Guard the fixture: the cold start must not already be the mode.
    let g0 = grad(&[0.0, 0.0]);
    assert!(g0 > 1.0, "cold start already stationary (|g| = {g0:e})");
    let gn = build(
        &model,
        &subject,
        &[0.0, 0.0],
        IMH_GN_MAX_STEPS,
        0.05,
        &ImhCounters::default(),
    );
    let g = grad(&gn.eta_hat);
    // Realised: 3.1e-3 from a cold-start norm of 6.2. Bound 2e-2 (~6×).
    assert!(
        g.is_finite() && g < 2e-2,
        "GN mode is not stationary: |g| = {g:e}"
    );
}

/// The cost claims the table is built on: a cold refresh spends gradients and
/// Jacobians only (the GN step needs no NLL evaluation), and once η̂ tracks the
/// mode a refresh costs **one** gradient and **one** Jacobian.
#[test]
fn a_warm_gn_refresh_costs_one_gradient_and_one_jacobian() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let subject = well_conditioned_subject();
    let cold = ImhCounters::default();
    let mode = build(&model, &subject, &[0.0, 0.0], IMH_GN_MAX_STEPS, 0.05, &cold).eta_hat;
    let c = cold.snapshot();
    assert!(
        c.map_grad_evals > 1 && c.map_jac_evals >= 1,
        "cold refresh did no search: {c:?}"
    );
    assert_eq!((c.refreshes, c.proposals), (1, 0));

    let warm = ImhCounters::default();
    let _ = build(&model, &subject, &mode, IMH_GN_MAX_STEPS, 0.05, &warm);
    let w = warm.snapshot();
    assert_eq!(
        (w.map_grad_evals, w.map_jac_evals, w.refreshes),
        (1, 1, 1),
        "a warm GN refresh cost {w:?}, not 1 gradient + 1 Jacobian"
    );
}

/// Re-curvature is its own decision, not a side effect of taking a step: a
/// refresh started a short distance from the mode takes Newton steps under the
/// curvature it already holds, and rebuilds it only once those steps have moved
/// it [`GN_RECURVE_MAHA`] in that metric. Started just off the mode it should
/// cost more gradients than Jacobians.
#[test]
fn a_near_mode_refresh_steps_without_re_curving() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let subject = well_conditioned_subject();
    let mode = build(
        &model,
        &subject,
        &[0.0, 0.0],
        30,
        0.05,
        &ImhCounters::default(),
    )
    .eta_hat;
    let near = [mode[0] + 0.01, mode[1] - 0.005];
    let c = ImhCounters::default();
    let _ = build(&model, &subject, &near, IMH_GN_MAX_STEPS, 0.05, &c);
    let s = c.snapshot();
    // Realised: 4 gradients (three steps under the held curvature, then the
    // converged check) and 1 Jacobian. Re-curving after every step would make
    // that 4 Jacobians.
    assert!(s.map_grad_evals >= 2, "the refresh took no step: {s:?}");
    assert_eq!(
        s.map_jac_evals, 1,
        "a near-mode refresh cost {s:?}: it re-curved after steps shorter than the threshold"
    );
}

/// The trust region caps one Newton step at [`GN_STEP_MAHA_CAP`] in the metric of
/// the curvature it was taken under. On the stress fixture, started on the far
/// side of the prior, the uncapped first step is longer than that, so a one-step
/// refresh must land on the cap exactly.
#[test]
fn one_gn_step_is_capped_by_the_trust_region() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let subject = stress_subject();
    // Far on the other side of the prior from the stress fixture's mode
    // (η_CL ≈ 1.5), where the first Newton step is long.
    const START: [f64; 2] = [-1.0, 0.0];
    let (jac, f) = super::eta_jacobian_and_preds(&model, &subject, &theta, &START).unwrap();
    let h0 = crate::stats::likelihood::conditional_eta_precision(
        &model, &subject, &theta, &START, &f, &jac, &omega, &sigma, true,
    )
    .unwrap();
    let g0 = analytic_eta_nll_gradient(&model, &subject, &theta, &START, &omega, &sigma).unwrap();
    let newton = h0
        .clone()
        .cholesky()
        .unwrap()
        .solve(&DVector::from_vec(g0.clone()));
    let uncapped = DVector::from_vec(g0).dot(&newton).sqrt();
    // Realised 3.97 against the cap of 3: an uncapped step would land 0.97 past
    // where the capped one does, far outside the 1e-9 the assertion allows.
    assert!(
        uncapped > 1.2 * GN_STEP_MAHA_CAP,
        "fixture's first step is not capped: {uncapped}"
    );

    let one = build(&model, &subject, &START, 1, 0.05, &ImhCounters::default());
    let d = DVector::from_vec(vec![one.eta_hat[0] - START[0], one.eta_hat[1] - START[1]]);
    let taken = d.dot(&(&h0 * &d)).sqrt();
    assert!(
        (taken - GN_STEP_MAHA_CAP).abs() < 1e-9,
        "first step length {taken} in the H̃ metric, cap {GN_STEP_MAHA_CAP} (uncapped {uncapped})"
    );
}

/// The one pairing measured to lose (+5.2 IS −2LL on busulfan with a no-η Q):
/// `imh` with a θ the numerical M-step estimates, under `bobyqa`. The warning
/// fires there and on no neighbouring configuration.
#[test]
fn imh_with_a_no_eta_theta_under_bobyqa_warns() {
    use crate::estimation::saem::imh_noeta_theta_warning;
    let w = imh_noeta_theta_warning(true, true, false).expect("the measured losing pairing");
    for needle in [
        "saem_kernel = imh",
        "no eta",
        "BOBYQA",
        "+5.2",
        "`mstep_solver = score_sa`",
    ] {
        assert!(w.contains(needle), "warning missing {needle:?}: {w}");
    }
    assert_eq!(imh_noeta_theta_warning(false, true, false), None, "rw");
    assert_eq!(
        imh_noeta_theta_warning(true, false, false),
        None,
        "every theta mu-referenced"
    );
    assert_eq!(imh_noeta_theta_warning(true, true, true), None, "score_sa");

    // End to end: V without an η is estimated by the numerical M-step.
    let noeta = parse_model_string(
        &IV_MODEL
            .replace("omega ETA_V  ~ 0.09\n", "")
            .replace("V  = TVV  * exp(ETA_V)", "V  = TVV"),
    )
    .expect("parse");
    let pop = population_of(&well_conditioned_subject(), 3);
    let warned = |model: &CompiledModel, opts: &FitOptions| {
        run_saem(model, &pop, &model.default_params, opts)
            .unwrap()
            .warnings
            .iter()
            .any(|w| w.contains("single-draw BOBYQA"))
    };
    assert!(
        warned(&noeta, &imh_opts(1)),
        "no warning on the losing pairing"
    );
    let sa = FitOptions {
        saem_mstep_solver: crate::types::SaemMstepSolver::ScoreSa,
        ..imh_opts(1)
    };
    assert!(!warned(&noeta, &sa), "warned under score_sa");
    let full = parse_model_string(IV_MODEL).expect("parse");
    assert!(
        !warned(&full, &imh_opts(1)),
        "warned with every theta mu-referenced"
    );
}

/// A Newton step on a badly non-quadratic objective can overshoot; the trust
/// region and the prior-ball rejection must keep the centre usable — and must
/// not pin it at the start either.
#[test]
fn gn_map_stays_in_the_prior_ball_on_the_stress_fixture() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let subject = stress_subject();
    // 20 steps, not 5: this mode is ~20 conditional SDs from the cold start and
    // each step is capped at 3. In a fit the cached η̂ carries forward instead.
    let prop = build(
        &model,
        &subject,
        &[0.0, 0.0],
        20,
        0.05,
        &ImhCounters::default(),
    );
    let maha = quad_form(&omega2(0.09, 0.09).inv, &prop.eta_hat);
    assert!(
        maha.is_finite() && maha < GN_RUNAWAY_MAHA_PER_ETA * 2.0,
        "GN left the prior ball: η̂ = {:?}, ηᵀΩ⁻¹η = {maha}",
        prop.eta_hat
    );
    assert!(
        prop.eta_hat[0] > 1.0,
        "GN did not reach the mode: η̂ = {:?}",
        prop.eta_hat
    );
}

// ---------------------------------------------------------------------------
// The cost report
// ---------------------------------------------------------------------------

/// One iteration's counters as population totals with per-call nanosecond costs
/// dialled in directly.
fn snap(
    props: u64,
    grads: u64,
    jacs: u64,
    n_subj: u64,
    ns_nll: u64,
    ns_grad: u64,
    ns_jac: u64,
) -> ImhSnapshot {
    ImhSnapshot {
        proposals: props,
        accepts: props,
        map_grad_evals: grads,
        map_jac_evals: jacs,
        refreshes: n_subj,
        declines: 0,
        imh_nll_ns: props * ns_nll,
        map_grad_ns: grads * ns_grad,
        map_jac_ns: jacs * ns_jac,
    }
}

/// The cost model reproduces the numbers measured on the #1449 benches
/// (per-refresh rates and ns costs from the E2 report):
///
/// | bench | grad/refresh | jac/refresh | ns/NLL | ns/grad | ns/jac |
/// |---|---|---|---|---|---|
/// | B1 warfarin | 2.40 | 1.50 | 385 | 2316 | 3235 |
/// | B5 pembro   | 4.45 | 2.50 | 4926 | 110068 | 117379 |
///
/// and the warning fires on the second and not the first.
#[test]
fn cost_estimate_and_warning_separate_the_measured_benches() {
    let b1 = estimate_imh_cost(&snap(200, 240, 150, 100, 385, 2316, 3235), 2, 5, 6).unwrap();
    assert!((b1.grad_ratio - 6.016).abs() < 0.01 && (b1.jac_ratio - 8.403).abs() < 0.01);
    // 2 + (2.40 × 6.016 + 1.50 × 8.403) / 5 = 7.41 → 1.23× the 6-step walk.
    assert!((b1.imh_cost - 7.410).abs() < 0.01, "B1: {b1:?}");
    assert_eq!(
        imh_cost_warning(&b1),
        None,
        "B1 is within budget: {:.2}x",
        b1.ratio()
    );

    let b5 = estimate_imh_cost(&snap(200, 445, 250, 100, 4926, 110068, 117379), 2, 5, 6).unwrap();
    // 2 + (4.45 × 22.344 + 2.50 × 23.828) / 5 = 33.80 → 5.63×.
    assert!((b5.imh_cost - 33.804).abs() < 0.01, "B5: {b5:?}");
    let w = imh_cost_warning(&b5).expect("B5 is 5.6x the walk");
    // Every number and both levers the sentence names.
    for needle in [
        "33.8",
        "6 the block",
        "5.6x",
        "22.3",
        "23.8",
        "saem_imh_refresh",
        "saem_kernel = rw",
    ] {
        assert!(w.contains(needle), "warning missing {needle:?}: {w}");
    }

    // The boundary itself: exactly 1.5× is silent, just above is not.
    let at = ImhCostEstimate {
        grad_ratio: 1.0,
        jac_ratio: 1.0,
        imh_cost: 9.0,
        rw_cost: 6.0,
    };
    assert_eq!(imh_cost_warning(&at), None);
    let over = ImhCostEstimate {
        imh_cost: 9.01,
        ..at
    };
    assert!(imh_cost_warning(&over).is_some());
}

/// A run with no usable measurement must never produce a cost claim.
#[test]
fn the_cost_estimate_needs_a_complete_measurement() {
    assert!(estimate_imh_cost(&snap(200, 0, 0, 100, 385, 2316, 3235), 2, 5, 6).is_none());
    assert!(estimate_imh_cost(&snap(0, 240, 150, 100, 385, 2316, 3235), 2, 5, 6).is_none());
    assert!(estimate_imh_cost(&snap(200, 240, 150, 0, 385, 2316, 3235), 2, 5, 6).is_none());
    // And amortisation is real: a longer refresh interval prices lower.
    let at = |r| {
        estimate_imh_cost(&snap(200, 445, 250, 100, 4926, 110068, 117379), 2, r, 6)
            .unwrap()
            .imh_cost
    };
    assert!(at(1) > at(5) && at(5) > at(10));
}

/// The eval counters include declined refresh attempts, so they are priced per
/// *attempt*: moving half the refreshes to declines (same evals, same attempts)
/// must not change the cost. Dividing by successful refreshes alone would price
/// it at double the refresh cost and could fire the warning on a cheap run.
#[test]
fn the_cost_estimate_prices_declined_attempts_too() {
    let all_ok = snap(200, 445, 250, 100, 385, 2316, 3235);
    let half_declined = ImhSnapshot {
        refreshes: 50,
        declines: 50,
        ..all_ok
    };
    let a = estimate_imh_cost(&all_ok, 2, 5, 6).unwrap().imh_cost;
    let b = estimate_imh_cost(&half_declined, 2, 5, 6).unwrap().imh_cost;
    assert!(a.is_finite() && b.is_finite());
    assert_eq!(a, b, "declined attempts priced differently: {a} vs {b}");
}

// ---------------------------------------------------------------------------
// The gates, and the kernel inside a fit
// ---------------------------------------------------------------------------

/// Shapes the curvature does not score keep the random walk; a plain analytic
/// model is in scope.
#[test]
fn imh_model_gate_rejects_the_shapes_the_curvature_does_not_score() {
    let ok = parse_model_string(IV_MODEL).expect("parse");
    assert!(
        imh_supported_model(&ok),
        "a plain analytic 1-cpt IV model must be in scope"
    );

    let mut ltbs = parse_model_string(IV_MODEL).expect("parse");
    ltbs.log_transform = true;
    assert!(!imh_supported_model(&ltbs), "LTBS must be out of scope");

    let mut iov = parse_model_string(IV_MODEL).expect("parse");
    iov.n_kappa = 1;
    assert!(!imh_supported_model(&iov), "IOV must be out of scope");

    let mut corr = parse_model_string(IV_MODEL).expect("parse");
    corr.residual_correlations
        .push(crate::types::ResidualCorrelation {
            sigma_i: 0,
            sigma_j: 0,
            rho: 0.5,
        });
    assert!(
        !imh_supported_model(&corr),
        "block_sigma must be out of scope"
    );

    let mut ruv = parse_model_string(IV_MODEL).expect("parse");
    ruv.residual_error_eta = Some(1);
    assert!(
        !imh_supported_model(&ruv),
        "iiv_on_ruv must be out of scope"
    );

    let mut m3 = parse_model_string(IV_MODEL).expect("parse");
    m3.bloq_method = crate::types::BloqMethod::M3;
    assert!(!imh_supported_model(&m3), "M3 must be out of scope");
}

fn imh_opts(seed: u64) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 6,
        saem_n_convergence: 6,
        saem_seed: Some(seed),
        run_covariance_step: false,
        saem_kernel: SaemKernel::Imh,
        saem_imh_refresh: 2,
        ..Default::default()
    }
}

fn last_imh() -> ImhSnapshot {
    LAST_IMH_SNAPSHOT
        .with(|c| c.get())
        .expect("run_saem records its counters")
}

/// Every arm of the decision, in gate order, and the warning each one produces.
/// Each sentence of a decline warning is asserted, and both sides of the
/// information gate are exercised in one test so a gate stuck on either branch
/// reddens it.
#[test]
fn the_kernel_decision_and_its_warnings() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let rich = population_of(&well_conditioned_subject(), 3); // 6 obs / 2 η = 3.0
    let mut sparse = rich.clone();
    for s in sparse.subjects.iter_mut() {
        s.obs_times.truncate(2);
        s.observations.truncate(2);
        s.obs_cmts.truncate(2);
        s.cens.truncate(2);
        s.occasions.truncate(2);
    } // 2 obs / 2 η = 1.0
    assert_eq!(obs_per_subject_per_eta(&rich, 2), Some(3.0));
    assert_eq!(obs_per_subject_per_eta(&sparse, 2), Some(1.0));
    assert_eq!(obs_per_subject_per_eta(&sparse, 0), None);

    let on = imh_opts(1);
    let rw = FitOptions {
        saem_kernel: SaemKernel::RandomWalk,
        ..on.clone()
    };
    assert_eq!(
        imh_kernel_decision(&model, &rich, &rw, false),
        ImhDecision::NotRequested
    );
    assert_eq!(
        imh_kernel_decision(&model, &rich, &on, false),
        ImhDecision::On
    );
    assert_eq!(
        imh_kernel_decision(&model, &rich, &on, true),
        ImhDecision::DeclinedHmc
    );
    assert_eq!(
        imh_kernel_decision(&model, &sparse, &on, false),
        ImhDecision::DeclinedInformation { ratio: 1.0 }
    );
    // `0` disables the information gate.
    let forced = FitOptions {
        saem_imh_min_obs_per_eta: 0.0,
        ..on.clone()
    };
    assert_eq!(
        imh_kernel_decision(&model, &sparse, &forced, false),
        ImhDecision::On
    );
    let mut ltbs = parse_model_string(IV_MODEL).expect("parse");
    ltbs.log_transform = true;
    assert_eq!(
        imh_kernel_decision(&ltbs, &rich, &on, false),
        ImhDecision::DeclinedScope
    );

    assert_eq!(imh_decline_warning(ImhDecision::On, 2.0, 2), None);
    assert_eq!(imh_decline_warning(ImhDecision::NotRequested, 2.0, 2), None);
    let info =
        imh_decline_warning(ImhDecision::DeclinedInformation { ratio: 1.0 }, 2.0, 2).unwrap();
    for needle in [
        "saem_kernel = imh requested but",
        "1.00 observations per subject per random effect (2 random effects)",
        "`saem_imh_min_obs_per_eta = 2.00`",
        "barely narrower than the prior",
        "Set `saem_imh_min_obs_per_eta = 0` to force the kernel on",
        "using the Metropolis-Hastings random walk",
    ] {
        assert!(
            info.contains(needle),
            "info warning missing {needle:?}: {info}"
        );
    }
    let scope = imh_decline_warning(ImhDecision::DeclinedScope, 2.0, 2).unwrap();
    assert!(
        scope.contains("outside the scope") && scope.contains("LTBS"),
        "{scope}"
    );
    let hmc = imh_decline_warning(ImhDecision::DeclinedHmc, 2.0, 2).unwrap();
    assert!(hmc.contains("HMC, which owns the primary kernel"), "{hmc}");

    // The banner reports the same decision the run takes.
    assert!(saem_sampler_summary_for(&model, &rich, &on).starts_with("f-SAEM"));
    assert!(saem_sampler_summary_for(&model, &sparse, &on).contains("1.00 obs/subject/eta < 2.00"));
    assert!(saem_sampler_summary_for(&model, &rich, &rw).starts_with("Metropolis-Hastings"));
}

/// The block step-scale controller only reads random-walk evidence.
#[test]
fn rw_scale_reads_only_random_walk_evidence() {
    assert!(rw_scale_has_evidence(false, 6));
    assert!(
        !rw_scale_has_evidence(true, 6),
        "an IMH-served iteration is not RW evidence"
    );
    assert!(
        !rw_scale_has_evidence(false, 0),
        "no proposals is no evidence"
    );
}

/// End to end: inside a fit the kernel runs, refreshes on the stated interval,
/// accepts at an independence-sampler's rate — and a declined fit runs none of
/// it, with the warning.
#[test]
fn the_kernel_runs_inside_a_fit_and_a_declined_fit_runs_none_of_it() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 6);
    let r = run_saem(&model, &pop, &model.default_params, &imh_opts(7)).expect("SAEM runs");
    let used = last_imh();
    // 12 iterations × 6 subjects × 2 proposals, every subject in scope.
    assert_eq!(used.proposals, 12 * 6 * 2, "{used:?}");
    // Refresh every 2 iterations → 6 refreshes per subject.
    assert_eq!(used.refreshes + used.declines, 6 * 6, "{used:?}");
    // The stuck warning reads the convergence phase only: 6 of the 12 iterations.
    let conv = crate::estimation::saem::LAST_IMH_CONV.with(|c| c.get());
    assert_eq!(conv.1, 6 * 6 * 2, "stuck-warning window: {conv:?}");
    assert!(conv.0 > 0 && conv.0 <= conv.1, "{conv:?}");
    assert_eq!(used.declines, 0, "{used:?}");
    // The per-call NLL clock is folded into the counters (once per subject call).
    assert!(used.imh_nll_ns > 0, "IMH NLL time never recorded: {used:?}");
    let rate = used.accepts as f64 / used.proposals as f64;
    // Realised 0.80 (seed 7); the random walk targets 0.40.
    assert!(rate > 0.6, "IMH acceptance {rate:.3} inside a fit");
    assert!(r.ofv.is_finite());
    assert!(!r
        .warnings
        .iter()
        .any(|w| w.contains("saem_kernel = imh requested")));
    // Every subject-iteration was served by the IMH kernel, so the block random
    // walk never ran — not even after it, as a second primary move.
    assert_eq!(
        LAST_RW_BLOCK_CALLS.with(|c| c.get()),
        0,
        "the random walk ran under IMH"
    );

    let mut sparse = pop.clone();
    for s in sparse.subjects.iter_mut() {
        s.obs_times.truncate(2);
        s.observations.truncate(2);
        s.obs_cmts.truncate(2);
        s.cens.truncate(2);
        s.occasions.truncate(2);
    }
    let r = run_saem(&model, &sparse, &model.default_params, &imh_opts(7)).expect("SAEM runs");
    assert_eq!(
        last_imh(),
        ImhSnapshot::default(),
        "a declined fit touched the kernel"
    );
    assert_eq!(
        LAST_RW_BLOCK_CALLS.with(|c| c.get()),
        12 * 6,
        "the declined fit is the random walk"
    );
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("saem_imh_min_obs_per_eta")),
        "no decline warning: {:?}",
        r.warnings
    );
}

/// The IMH kernel's acceptance must never steer the block random walk's step
/// scale, under either adaptation rule: a subject the kernel served every
/// iteration ends the fit with the scale it started with (0.3). The random-walk
/// control on the same fit must move it, or the assertion is of a constant.
#[test]
fn imh_acceptance_never_steers_the_random_walk_scale() {
    use crate::estimation::saem::LAST_BLOCK_STEP_SCALES;
    use crate::types::ScaleAdaptation;
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 4);
    for rule in [ScaleAdaptation::RobbinsMonro, ScaleAdaptation::Interval] {
        let base = FitOptions {
            saem_n_exploration: 30,
            saem_n_convergence: 30,
            saem_adapt_interval: 10,
            saem_scale_adaptation: rule,
            ..imh_opts(5)
        };
        run_saem(&model, &pop, &model.default_params, &base).expect("SAEM runs");
        assert_eq!(
            last_imh().declines,
            0,
            "{rule:?}: a refresh declined, so the RW ran"
        );
        let imh = LAST_BLOCK_STEP_SCALES.with(|c| c.borrow().clone());
        assert_eq!(
            imh,
            vec![0.3; 4],
            "{rule:?}: IMH acceptance moved the RW scale"
        );

        let rw = FitOptions {
            saem_kernel: SaemKernel::RandomWalk,
            ..base
        };
        run_saem(&model, &pop, &model.default_params, &rw).expect("SAEM runs");
        let moved = LAST_BLOCK_STEP_SCALES.with(|c| c.borrow().clone());
        assert!(
            moved.iter().all(|&s| s != 0.3),
            "{rule:?}: the random walk never adapted its scale ({moved:?}) — the IMH \
             assertion above is not testing the controller"
        );
    }
}

/// No clock reaches the estimates: two seeded IMH fits are bit-identical, and a
/// different seed is not (so the comparison is not of two constants).
#[test]
fn a_seeded_imh_fit_is_reproducible() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 6);
    let fit = |seed| {
        let r = run_saem(&model, &pop, &model.default_params, &imh_opts(seed)).unwrap();
        let mut v: Vec<u64> = r.params.theta.iter().map(|x| x.to_bits()).collect();
        v.extend(r.params.omega.matrix.iter().map(|x| x.to_bits()));
        v.extend(r.params.sigma.values.iter().map(|x| x.to_bits()));
        v
    };
    let (a, b, c) = (fit(3), fit(3), fit(4));
    assert_eq!(a, b, "same seed, different estimates");
    assert_ne!(
        a, c,
        "the seed does not reach the estimates — the comparison above is vacuous"
    );
}

/// The default `saem_kernel` must not build a single proposal or spend a single
/// IMH evaluation.
#[test]
fn default_fit_options_leave_the_imh_kernel_untouched() {
    assert_eq!(FitOptions::default().saem_kernel, SaemKernel::RandomWalk);
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 3);
    let opts = FitOptions {
        saem_kernel: SaemKernel::RandomWalk,
        ..imh_opts(1)
    };
    let _ = run_saem(&model, &pop, &model.default_params, &opts).expect("SAEM runs");
    assert_eq!(
        last_imh(),
        ImhSnapshot::default(),
        "a default SAEM run touched the IMH kernel"
    );
}

// ---------------------------------------------------------------------------
// Review round 1 (PR #2613): the proposal's state-independence, the decline
// retry interval, the trust-region rejection, the options backstop, the full
// scope gate, and the default path frozen against `main`.
// ---------------------------------------------------------------------------

/// A sentinel proposal whose centre records the warm start it was built from.
fn sentinel_prop(warm: &[f64]) -> ImhProposal {
    let n = warm.len();
    ImhProposal::from_precision(warm.to_vec(), &DMatrix::identity(n, n), 1.0, 0.05, None).unwrap()
}

/// The proposal is never built from the chain's current state: the first build
/// starts at the prior mean `0`, later ones at the previous centre — and the
/// rule is not *handed* the current η, so no caller can thread it in. A
/// Gauss-Newton search capped short of the mode would otherwise make `q` a
/// function of the state it is about to be compared against (review round 1).
#[test]
fn a_refresh_starts_from_the_previous_centre_or_the_prior_mean() {
    let mut slot = ImhSlot::default();
    let mut seen: Vec<Vec<f64>> = Vec::new();
    slot.refresh_if_due_with(0, 3, 2, |warm| {
        seen.push(warm.to_vec());
        Some(sentinel_prop(&[0.7, -0.4]))
    });
    assert_eq!(seen, vec![vec![0.0, 0.0]], "a cold build must start at 0");
    slot.refresh_if_due_with(3, 3, 2, |warm| {
        seen.push(warm.to_vec());
        Some(sentinel_prop(&[0.9, -0.2]))
    });
    assert_eq!(
        seen[1],
        vec![0.7, -0.4],
        "a warm build must start at the previous centre"
    );
    // A decline drops the centre: the next build is cold again.
    slot.refresh_if_due_with(6, 3, 2, |_| None);
    assert!(slot.prop.is_none());
    slot.refresh_if_due_with(9, 3, 2, |warm| {
        seen.push(warm.to_vec());
        None
    });
    assert_eq!(seen[2], vec![0.0, 0.0], "after a decline the build is cold");
}

/// A declined refresh waits the same interval as a successful one, instead of
/// re-running the full refresh every iteration on top of the random walk; a
/// successful one is reused until it is due.
#[test]
fn a_declined_refresh_waits_its_interval() {
    let mut slot = ImhSlot::default();
    let mut calls: Vec<usize> = Vec::new();
    for k in 0..11 {
        slot.refresh_if_due_with(k, 5, 2, |_| {
            calls.push(k);
            None
        });
    }
    assert_eq!(calls, vec![0, 5, 10], "declines retried off-interval");

    let mut slot = ImhSlot::default();
    let mut calls: Vec<usize> = Vec::new();
    for k in 0..11 {
        slot.refresh_if_due_with(k, 5, 2, |w| {
            calls.push(k);
            Some(sentinel_prop(w))
        });
        assert!(slot.prop.is_some(), "a built proposal was dropped at {k}");
    }
    assert_eq!(calls, vec![0, 5, 10], "successful builds off-interval");
}

/// The prior-ball rejection, reached directly: a quadratic objective whose mode
/// sits far outside the ball, started just inside its edge, so the very first
/// (trust-region-capped) step crosses it. The step must be rejected — the
/// centre stays at the start — while the same objective started further in
/// does move, so the assertion is not of a search that never steps.
#[test]
fn the_prior_ball_rejects_the_step_that_crosses_it() {
    let omega = OmegaMatrix::from_diagonal(&[1.0], vec!["E".into()]);
    let mode = 50.0_f64; // ηᵀΩ⁻¹η = 2500 ≫ the ball's 100
    let gradient_at = |e: &[f64]| Some(vec![e[0] - mode]);
    let curvature_at = |_: &[f64]| Some(DMatrix::from_element(1, 1, 1.0));
    let edge = GN_RUNAWAY_MAHA_PER_ETA.sqrt() - 1.0; // 9: one capped step lands at 12
    let (eta, _) = gn_map_refine(&gradient_at, &curvature_at, &[edge], &omega, 5).unwrap();
    assert_eq!(eta, vec![edge], "the step out of the prior ball was taken");

    let (eta, _) = gn_map_refine(&gradient_at, &curvature_at, &[0.0], &omega, 5).unwrap();
    assert!(
        eta[0] > 0.0 && quad_form(&omega.inv, &eta) <= GN_RUNAWAY_MAHA_PER_ETA,
        "from inside the ball the search must step and stay inside: {eta:?}"
    );
}

/// `FitOptions` built in code bypass the parser: the run's backstop replaces an
/// unusable knob with its default and says so, in a sentence naming the value.
#[test]
fn unusable_imh_knobs_fall_back_to_their_defaults_with_a_warning() {
    let ok = imh_opts(1);
    let (k, w) = imh_knobs(&ok);
    assert_eq!(
        k,
        ImhKnobs {
            defensive: 0.05,
            min_obs_per_eta: 2.0
        }
    );
    assert!(w.is_empty(), "{w:?}");
    let edge = FitOptions {
        saem_imh_defensive: 0.5,
        saem_imh_min_obs_per_eta: 0.0,
        ..ok.clone()
    };
    let (k, w) = imh_knobs(&edge);
    assert_eq!((k.defensive, k.min_obs_per_eta), (0.5, 0.0));
    assert!(w.is_empty(), "{w:?}");

    for bad in [f64::NAN, -0.1, 0.6] {
        let o = FitOptions {
            saem_imh_defensive: bad,
            ..ok.clone()
        };
        let (k, w) = imh_knobs(&o);
        assert_eq!(k.defensive, 0.05, "{bad}");
        assert_eq!(
            w,
            vec![format!(
                "saem_imh_defensive = {bad} is outside [0, 0.5]; using 0.05"
            )]
        );
    }
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let o = FitOptions {
            saem_imh_min_obs_per_eta: bad,
            ..ok.clone()
        };
        let (k, w) = imh_knobs(&o);
        assert_eq!(k.min_obs_per_eta, 2.0, "{bad}");
        assert_eq!(
            w,
            vec![format!(
                "saem_imh_min_obs_per_eta = {bad} is not a finite value >= 0; using 2 \
                 (set it to 0 to switch the information gate off)"
            )]
        );
    }
}

/// End to end: a NaN threshold no longer switches the information gate off (the
/// sparse population is declined, as under the default), a NaN weight no longer
/// rejects every proposal (the kernel accepts), each with its warning — and a
/// random-walk fit carrying the same bad knobs says nothing about them.
#[test]
fn a_fit_with_unusable_imh_knobs_runs_on_the_defaults() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let rich = population_of(&well_conditioned_subject(), 3);
    let mut sparse = rich.clone();
    for s in sparse.subjects.iter_mut() {
        s.obs_times.truncate(2);
        s.observations.truncate(2);
        s.obs_cmts.truncate(2);
        s.cens.truncate(2);
        s.occasions.truncate(2);
    }
    let bad = FitOptions {
        saem_imh_defensive: f64::NAN,
        saem_imh_min_obs_per_eta: f64::NAN,
        ..imh_opts(3)
    };
    assert_eq!(
        imh_kernel_decision(&model, &sparse, &bad, false),
        ImhDecision::DeclinedInformation { ratio: 1.0 },
        "a NaN threshold switched the information gate off"
    );

    let r = run_saem(&model, &rich, &model.default_params, &bad).expect("SAEM runs");
    let used = last_imh();
    assert!(
        used.proposals > 0 && used.accepts > 0,
        "a NaN weight rejected every proposal: {used:?}"
    );
    for key in ["saem_imh_defensive = NaN", "saem_imh_min_obs_per_eta = NaN"] {
        assert!(
            r.warnings.iter().any(|w| w.starts_with(key)),
            "no `{key}` warning: {:?}",
            r.warnings
        );
    }
    let rw = FitOptions {
        saem_kernel: SaemKernel::RandomWalk,
        ..bad
    };
    let r = run_saem(&model, &rich, &model.default_params, &rw).expect("SAEM runs");
    assert!(
        !r.warnings.iter().any(|w| w.contains("saem_imh_")),
        "a random-walk fit warned about IMH knobs: {:?}",
        r.warnings
    );
}

/// Every conjunct of the scope gate on its own: each fixture differs from the
/// in-scope model in exactly the one property its line names.
#[test]
fn every_scope_condition_declines_on_its_own() {
    let base = parse_model_string(IV_MODEL).expect("parse");
    assert!(imh_supported_model(&base));

    // An ODE model is in scope since #2639: its η-gradient and η-Jacobian come
    // from the adjoint's dual ODE walk, as the analytical model's come from the
    // closed form.
    let ode_src = "\
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -CL / V * central
[error_model]
  DV ~ proportional(PROP_ERR)
";
    let ode = parse_model_string(ode_src).expect("ODE twin parses");
    assert!(ode.ode_spec.is_some() && ode.tv_fn.is_none());
    assert!(imh_supported_model(&ode), "an ODE model must be in scope");

    // The same ODE with a `[diffusion]` block — an SDE, whose EKF likelihood the
    // dual ODE walk does not record — differs in that one block.
    let sde = parse_model_string(&format!("{ode_src}[diffusion]\n  central ~ 0.05 FIX\n"))
        .expect("SDE twin parses");
    assert!(sde.is_sde());
    assert!(
        !imh_supported_model(&sde),
        "an SDE model must be out of scope"
    );

    // An ODE the recorder declines (no compiled RHS program to walk over duals),
    // differing from the in-scope ODE in that one field.
    let mut no_rhs = parse_model_string(ode_src).expect("parse");
    no_rhs.ode_spec.as_mut().unwrap().rhs_program = None;
    assert!(
        !imh_supported_model(&no_rhs),
        "an ODE the dual walk declines must be out of scope"
    );

    let mixture = parse_model_string(
        "\
[parameters]
  theta TVCL1(1.0, 0.01, 50.0)
  theta TVCL2(3.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("mixture twin parses");
    assert!(mixture.mixture.is_some());
    assert!(
        !imh_supported_model(&mixture),
        "a mixture must be out of scope"
    );

    let mut frem = parse_model_string(IV_MODEL).expect("parse");
    frem.frem_config = Some(crate::types::FremConfig {
        fremtype_to_indices: std::collections::HashMap::from([(100u16, (0usize, 0usize))]),
        covariate_sigma_index: 0,
    });
    assert!(!imh_supported_model(&frem), "FREM must be out of scope");

    let mut fd = parse_model_string(IV_MODEL).expect("parse");
    fd.gradient_method = crate::types::GradientMethod::Fd;
    assert!(
        !imh_supported_model(&fd),
        "a model without the analytic η-gradient must be out of scope"
    );
}

/// The scope decline names every shape it covers, so a user whose model is one
/// of them finds it in the sentence.
#[test]
fn the_scope_decline_warning_names_every_shape() {
    assert_eq!(
        imh_decline_warning(ImhDecision::DeclinedScope, 2.0, 2).unwrap(),
        "saem_kernel = imh requested but the model is outside the scope the IMH proposal \
         curvature covers (it needs an analytical PK or ODE model whose eta-gradient and \
         eta-Jacobian are analytic, and no IOV, iiv_on_ruv, mixture, FREM, SDE, block_sigma, \
         M3 censoring, non-Gaussian endpoint or LTBS); using the Metropolis-Hastings random \
         walk"
    );
    assert_eq!(
        imh_decline_warning(ImhDecision::DeclinedHmc, 2.0, 2).unwrap(),
        "saem_kernel = imh requested but saem_n_leapfrog > 0 selected HMC, which owns the \
         primary kernel; using the Metropolis-Hastings random walk"
    );
}

/// The banner and the run ask one HMC predicate: a mixture model with
/// `saem_n_leapfrog > 0` runs the MH kernels, and the banner says so (it used to
/// announce HMC).
#[test]
fn the_banner_and_the_run_agree_on_hmc_for_a_mixture() {
    let mixture = parse_model_string(
        "\
[parameters]
  theta TVCL1(1.0, 0.01, 50.0)
  theta TVCL2(3.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("mixture twin parses");
    let plain = parse_model_string(IV_MODEL).expect("parse");
    let opts = FitOptions {
        saem_n_leapfrog: 5,
        ..imh_opts(1)
    };
    assert!(crate::estimation::saem::saem_uses_hmc(&plain, &opts));
    assert!(!crate::estimation::saem::saem_uses_hmc(&mixture, &opts));
    let banner = crate::estimation::saem::saem_sampler_summary(&mixture, &opts);
    assert_eq!(
        banner,
        "Metropolis-Hastings random walk (HMC requested but unavailable — needs an \
         analytical PK model, no IOV, no iiv_on_ruv, no mixture)"
    );
    // And the IMH banner reads the same predicate: HMC displaces the kernel on
    // the plain model, not on the mixture (which IMH's scope declines instead).
    let pop = population_of(&well_conditioned_subject(), 3);
    assert_eq!(
        saem_sampler_summary_for(&plain, &pop, &opts),
        "HMC (5 leapfrog steps, Dual2 analytic gradients) (saem_kernel = imh: HMC owns the kernel)"
    );
    assert!(saem_sampler_summary_for(&mixture, &pop, &opts)
        .ends_with("(saem_kernel = imh: model out of scope)"));
}

/// Under `saem_kernel = imh` the end-of-run mixing tally holds only the random
/// walk's kernels, read against their own targets: with every primary move
/// served by IMH, the tallied proposals are exactly the componentwise sweep's.
#[test]
fn the_mixing_tally_leaves_the_imh_move_out() {
    use crate::estimation::saem::LAST_MH_TALLY;
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 3);
    let opts = FitOptions {
        saem_n_exploration: 10,
        saem_n_convergence: 10,
        saem_omega_burnin: 0,
        ..imh_opts(9)
    };
    run_saem(&model, &pop, &model.default_params, &opts).expect("SAEM runs");
    assert_eq!(last_imh().declines, 0);
    let (prop, cw_prop) = LAST_MH_TALLY.with(|c| c.get());
    assert!(cw_prop > 0, "the sweep never ran");
    assert_eq!(prop, cw_prop, "IMH proposals reached the mixing tally");
}

/// Captured on `main` f8955c24 (macOS arm64) by running this test's body there:
/// θ (2), Ω (2×2, column-major), σ (1), the SAEM objective.
const FROZEN_LEN: usize = 8;
const FROZEN_MAIN: [f64; FROZEN_LEN] = [
    f64::from_bits(0x4005b9e068362eda),
    f64::from_bits(0x403308f6f0f08f9c),
    f64::from_bits(0x3fba2eb62a8bc775),
    0.0,
    0.0,
    f64::from_bits(0x3f6e0970b0407fdd),
    f64::from_bits(0x3fa543cf2f825db8),
    f64::from_bits(0xc049b5cf65530c04),
];

/// The default path frozen against `main` (f8955c24): a seeded random-walk fit's
/// estimates and objective, captured there with this exact fixture. Any change
/// to the default E-step — an extra RNG draw, a reordered proposal, a scale
/// controller fed differently — moves these far past the tolerance (one extra
/// draw moves θ in the third figure), while the tolerance absorbs a last-bit
/// libm difference between platforms.
#[test]
fn the_default_random_walk_fit_matches_main() {
    use crate::types::DoseEvent;
    let model = parse_model_string(IV_MODEL).expect("parse");
    let obs = [
        [4.6, 3.9, 3.1, 2.0, 0.9, 0.4],
        [5.4, 4.8, 4.2, 3.3, 2.1, 1.3],
        [4.9, 4.5, 3.6, 2.6, 1.4, 0.8],
        [5.9, 5.1, 4.6, 3.6, 2.4, 1.6],
    ];
    let pop = Population {
        subjects: obs
            .iter()
            .enumerate()
            .map(|(i, o)| Subject {
                id: format!("{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
                observations: o.to_vec(),
                obs_cmts: vec![1; 6],
                cens: vec![0; 6],
                occasions: vec![1; 6],
                ..Default::default()
            })
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let opts = FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 8,
        saem_n_convergence: 8,
        saem_seed: Some(11),
        run_covariance_step: false,
        ..Default::default()
    };
    let r = run_saem(&model, &pop, &model.default_params, &opts).expect("SAEM runs");
    let mut got: Vec<f64> = r.params.theta.clone();
    got.extend(r.params.omega.matrix.iter().copied());
    got.extend(r.params.sigma.values.iter().copied());
    got.push(r.ofv);
    let want: [f64; FROZEN_LEN] = FROZEN_MAIN;
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "entry {i} non-finite: {g}");
        let rel = (g - w).abs() / w.abs().max(1e-12);
        assert!(
            rel < 1e-9,
            "default SAEM drifted from main at entry {i}: {g:e} vs {w:e} (rel {rel:e})"
        );
    }
}

/// The non-Gaussian conjunct on its own: the in-scope PK model plus a
/// time-to-event endpoint (only parseable with `survival`).
#[cfg(feature = "survival")]
#[test]
fn a_non_gaussian_endpoint_declines_the_kernel() {
    let src = format!(
        "{IV_MODEL}[event_model]\n  cmt    = 2\n  family = exponential\n  scale  = TVCL * exp(ETA_CL)\n"
    );
    let tte = parse_model_string(&src).expect("joint PK + TTE parses");
    assert!(tte.has_non_gaussian());
    assert!(
        !imh_supported_model(&tte),
        "a non-Gaussian endpoint must be out of scope"
    );
}

/// A stuck independence sampler is reported on its own scale, in a sentence
/// that holds for a one-η model too (no componentwise sweep to fall back on);
/// a healthy one, and a phase that never proposed, say nothing.
#[test]
fn a_stuck_imh_kernel_is_reported() {
    assert_eq!(
        imh_stuck_warning(5, 1000, 0.01).unwrap(),
        "saem_kernel = imh accepted 0.50% of its 1000 proposals over the convergence phase \
         — the independent proposal sits where the conditional has almost no mass, so its \
         draws barely moved the chain and the estimates rest on whatever mixing the rest of \
         the E-step managed. `saem_kernel = rw` is the safer kernel for this model."
    );
    assert_eq!(imh_stuck_warning(800, 1000, 0.01), None);
    assert_eq!(imh_stuck_warning(0, 0, 0.01), None);
}

/// The production wrapper seeds the refresh where the rule says: a cold
/// `refresh_if_due` on the stress subject builds exactly the proposal a
/// `refresh_proposal` from `0` builds (5 capped GN steps stop far short of the
/// mode at `η_CL ≈ 1.5`, so any other seed lands elsewhere), and a warm one
/// exactly the proposal from the previous centre.
#[test]
fn the_production_refresh_seeds_from_zero_then_the_previous_centre() {
    let model = parse_model_string(IV_MODEL).expect("parse");
    let subject = stress_subject();
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let c = ImhCounters::default();
    let from = |warm: &[f64]| {
        refresh_proposal(
            &model,
            &subject,
            &theta,
            &omega,
            &sigma,
            warm,
            IMH_GN_MAX_STEPS,
            0.05,
            &c,
        )
        .unwrap()
        .eta_hat
    };
    let mut slot = ImhSlot::default();
    slot.refresh_if_due(0, 2, &model, &subject, &theta, &omega, &sigma, 0.05, &c);
    let cold = slot.prop.as_ref().unwrap().eta_hat.clone();
    assert_eq!(cold, from(&[0.0, 0.0]), "the cold build did not start at 0");
    assert!(
        cold[0] < 1.4,
        "five steps reached the mode, so the seed is not observable: {cold:?}"
    );
    slot.refresh_if_due(2, 2, &model, &subject, &theta, &omega, &sigma, 0.05, &c);
    let warm = slot.prop.as_ref().unwrap().eta_hat.clone();
    assert_eq!(
        warm,
        from(&cold),
        "the warm build did not start at the previous centre"
    );
    assert_ne!(
        warm, cold,
        "the warm build did not step, so it is not observable"
    );
}

/// The stuck warning is wired into the run: with the threshold raised above
/// any attainable rate the same mixing fit reports it, and at the production
/// threshold it does not.
#[test]
fn the_stuck_warning_reaches_the_fit_result() {
    use crate::estimation::saem::IMH_STUCK_OVERRIDE;
    let model = parse_model_string(IV_MODEL).expect("parse");
    let pop = population_of(&well_conditioned_subject(), 3);
    let has = |r: &crate::estimation::outer_optimizer::OuterResult| {
        r.warnings
            .iter()
            .any(|w| w.starts_with("saem_kernel = imh accepted"))
    };
    let r = run_saem(&model, &pop, &model.default_params, &imh_opts(6)).expect("SAEM runs");
    assert!(last_imh().accepts > 0);
    assert!(!has(&r), "a mixing fit warned stuck: {:?}", r.warnings);
    IMH_STUCK_OVERRIDE.with(|c| c.set(Some(1.01)));
    let r = run_saem(&model, &pop, &model.default_params, &imh_opts(6));
    IMH_STUCK_OVERRIDE.with(|c| c.set(None));
    let r = r.expect("SAEM runs");
    assert!(
        has(&r),
        "the stuck warning never reached the result: {:?}",
        r.warnings
    );
    // And it reports the convergence-phase window, not the run: 3 subjects × 2
    // proposals × the 6 convergence iterations, against the run's 12.
    let (conv_acc, conv_prop) = crate::estimation::saem::LAST_IMH_CONV.with(|c| c.get());
    let run_prop = last_imh().proposals;
    assert_eq!((conv_prop, run_prop), (36, 72));
    let w = r
        .warnings
        .iter()
        .find(|w| w.starts_with("saem_kernel = imh accepted"))
        .unwrap();
    let want = format!(
        "saem_kernel = imh accepted {:.2}% of its {conv_prop} proposals over the convergence phase",
        conv_acc as f64 / conv_prop as f64 * 100.0
    );
    assert!(
        w.starts_with(&want),
        "{w}\nwanted the convergence window: {want}"
    );
}

// ---------------------------------------------------------------------------
// ODE models (#2639)
// ---------------------------------------------------------------------------

/// A one-state Michaelis-Menten ODE: genuinely nonlinear in `η_VM` (the
/// concentration sits near `KM`, so elimination is neither first- nor
/// zero-order), and with no closed form, so every derivative below comes from
/// the adjoint's dual ODE walk.
const MM_ODE_MODEL: &str = "\
[parameters]
  theta TVVM(10.0, 0.1, 500.0)
  theta TVKM(2.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_VM ~ 0.09
  omega ETA_V  ~ 0.09
  sigma PROP_ERR ~ 0.04
[individual_parameters]
  VM = TVVM * exp(ETA_VM)
  KM = TVKM
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -VM * (central / V) / (KM + central / V)
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// [`MM_ODE_MODEL`] under an additive error, for the zero-residual Hessian.
const MM_ODE_MODEL_ADDITIVE: &str = "\
[parameters]
  theta TVVM(10.0, 0.1, 500.0)
  theta TVKM(2.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_VM ~ 0.09
  omega ETA_V  ~ 0.09
  sigma ADD_ERR ~ 0.04
[individual_parameters]
  VM = TVVM * exp(ETA_VM)
  KM = TVKM
  V  = TVV  * exp(ETA_V)
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -VM * (central / V) / (KM + central / V)
[scaling]
  y = central / V
[error_model]
  DV ~ additive(ADD_ERR)
";

/// Two bolus doses, the second at `t = 12` while drug from the first is still
/// present, with observations on both sides of it — so the derivative through
/// the second dose event carries a non-zero incoming state.
const MM_OBS_TIMES: [f64; 10] = [0.5, 1.0, 2.0, 4.0, 8.0, 11.5, 12.5, 14.0, 18.0, 24.0];
const MM_SECOND_DOSE: f64 = 12.0;

/// Parse an MM ODE fixture with the solver tightened to `reltol = 1e-10`,
/// `abstol = 1e-12`. At the default `1e-4` the `f64` value path's adaptive step
/// sizes move with η, and a finite difference of it picks up that motion: the
/// realised dual-vs-FD Jacobian gap on the last observation was 8.0e-5 relative —
/// the solver tolerance, not a derivative defect. The oracle has to be more
/// accurate than the difference it adjudicates, so the parity fixtures run tight.
fn mm_model(src: &str) -> CompiledModel {
    let mut model = parse_model_string(src).expect("parse");
    let ode = model.ode_spec.as_mut().expect("MM fixture is an ODE model");
    ode.solver_opts.reltol = 1e-10;
    ode.solver_opts.abstol = 1e-12;
    model
}

fn mm_subject(observations: Vec<f64>) -> Subject {
    let n = MM_OBS_TIMES.len();
    assert_eq!(n, observations.len());
    Subject {
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(MM_SECOND_DOSE, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: MM_OBS_TIMES.to_vec(),
        observations,
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1; n],
        ..Default::default()
    }
}

/// The ODE twin of [`well_conditioned_subject`]: the model's own predictions at
/// `η = (0.25, −0.15)` with a ±4 % ripple.
fn mm_well_conditioned_subject() -> Subject {
    let model = mm_model(MM_ODE_MODEL);
    let theta = model.default_params.theta.clone();
    let mut subject = mm_subject(vec![0.0; MM_OBS_TIMES.len()]);
    let preds = predictions(&model, &subject, &theta, &[0.25, -0.15]);
    assert!(preds.iter().all(|p| p.is_finite() && *p > 0.0));
    let ripple = [1.04, 0.97, 1.02, 0.96, 1.03, 0.98, 1.02, 0.97, 1.03, 0.99];
    subject.observations = preds.iter().zip(ripple).map(|(p, r)| p * r).collect();
    subject
}

/// The fixture is what the parity tests below need it to be: an ODE model (so
/// no closed form can supply the answer), in the IMH scope, nonlinear, and with
/// drug on board when the second dose lands.
#[test]
fn the_ode_fixture_is_multi_dose_nonlinear_and_in_scope() {
    let model = parse_model_string(MM_ODE_MODEL).expect("parse");
    assert!(model.ode_spec.is_some() && model.tv_fn.is_none());
    assert!(
        imh_supported_model(&model),
        "the MM ODE model must be in scope"
    );
    let theta = model.default_params.theta.clone();
    let subject = mm_subject(vec![0.0; MM_OBS_TIMES.len()]);
    let f = predictions(&model, &subject, &theta, &[0.0, 0.0]);
    // Just before the second dose the concentration is still a sizeable share of
    // the first dose's peak (100 / 20 = 5): the second dose's incoming side is live.
    let before = f[MM_OBS_TIMES.iter().position(|&t| t == 11.5).unwrap()];
    assert!(
        before > 0.5,
        "no residual drug at the second dose: C(11.5) = {before}"
    );
    // ... and near KM = 2, where MM elimination is far from linear.
    assert!(f[0] > 2.0, "C(0.5) = {} never reaches KM", f[0]);
    // The second dose raises the concentration above the residual.
    let after = f[MM_OBS_TIMES.iter().position(|&t| t == 12.5).unwrap()];
    assert!(
        after > before + 3.0,
        "second dose did not land: {before} → {after}"
    );
}

/// Dual-vs-FD parity of the η-Jacobian on the ODE fixture: the production pair
/// `(H, f)` from one dual sweep of the ODE walk against a central-FD Jacobian of
/// the `f64` ODE predictor, and `f` against the predictor itself.
#[test]
fn ode_eta_jacobian_matches_fd_across_a_second_dose() {
    let model = mm_model(MM_ODE_MODEL);
    let theta = model.default_params.theta.clone();
    let subject = mm_well_conditioned_subject();
    let eta = [0.2, -0.1];
    let (jac, f) = super::eta_jacobian_and_preds(&model, &subject, &theta, &eta)
        .expect("ODE η-Jacobian in scope");
    let jac_fd = fd_jacobian(&model, &subject, &theta, &eta);
    let f_val = predictions(&model, &subject, &theta, &eta);
    let mut worst_f = 0.0_f64;
    let mut worst_j = 0.0_f64;
    for j in 0..MM_OBS_TIMES.len() {
        assert!(f[j].is_finite() && f_val[j].is_finite());
        worst_f = worst_f.max((f[j] - f_val[j]).abs() / f_val[j].abs());
        for a in 0..2 {
            let (g, w) = (jac[(j, a)], jac_fd[(j, a)]);
            assert!(g.is_finite() && w.is_finite(), "non-finite J[{j},{a}]");
            worst_j = worst_j.max((g - w).abs() / w.abs().max(1e-3));
        }
    }
    // Both sides of the second dose carry η-sensitivity in both coordinates.
    for t in [11.5, 12.5] {
        let j = MM_OBS_TIMES.iter().position(|&x| x == t).unwrap();
        assert!(
            jac[(j, 0)].abs() > 1e-2 && jac[(j, 1)].abs() > 1e-2,
            "J row at t={t} dead"
        );
    }
    // Realised: f bit-identical (0), worst J rel 1.2e-8 (FD truncation at
    // h = 1e-6 over a 1e-10-tolerance solve). Bounds 1e-9 / 1e-6 (~80×); at the
    // default `reltol = 1e-4` the same gap was 8e-5, which is what a
    // tolerance-sized derivative defect would look like here.
    assert!(worst_f < 1e-9, "dual f vs predictor rel {worst_f:e}");
    assert!(worst_j < 1e-6, "dual J vs FD rel {worst_j:e}");
}

/// The ODE twin of [`conditional_precision_matches_the_fd_hessian_at_zero_residual`]:
/// at a zero-residual point under additive error, `H̃` built from the dual ODE
/// Jacobian must equal the central-FD Hessian of `individual_nll`.
#[test]
fn ode_conditional_precision_matches_the_fd_hessian_at_zero_residual() {
    let model = mm_model(MM_ODE_MODEL_ADDITIVE);
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let eta_star = [0.25, -0.15];
    let mut subject = mm_subject(vec![0.0; MM_OBS_TIMES.len()]);
    subject.observations = predictions(&model, &subject, &theta, &eta_star);

    let (jac, f) = super::eta_jacobian_and_preds(&model, &subject, &theta, &eta_star)
        .expect("ODE η-Jacobian in scope");
    let h_tilde = crate::stats::likelihood::conditional_eta_precision(
        &model, &subject, &theta, &eta_star, &f, &jac, &omega, &sigma, true,
    )
    .expect("curvature in scope");

    let nll = |e: &[f64]| individual_nll(&model, &subject, &theta, e, &omega, &sigma);
    let h = 1e-4;
    let mut fd = DMatrix::zeros(2, 2);
    for a in 0..2 {
        for b in 0..2 {
            let at = |da: f64, db: f64| {
                let mut e = eta_star.to_vec();
                e[a] += da;
                e[b] += db;
                nll(&e)
            };
            fd[(a, b)] = (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4.0 * h * h);
        }
    }
    let mut worst = 0.0_f64;
    for a in 0..2 {
        for b in 0..2 {
            let (g, w) = (h_tilde[(a, b)], fd[(a, b)]);
            assert!(g.is_finite() && w.is_finite(), "non-finite curvature entry");
            worst = worst.max((g - w).abs() / w.abs());
        }
    }
    // The data term must dominate the prior, or the check is of Ω⁻¹ alone.
    assert!(
        h_tilde[(0, 0)] > 2.0 * omega.inv[(0, 0)] && h_tilde[(1, 1)] > 2.0 * omega.inv[(1, 1)],
        "curvature is mostly prior: {h_tilde}"
    );
    // Realised worst relative error 2.9e-8 (FD truncation at h = 1e-4). Bound
    // 1e-6 (~34×); a dropped Ω⁻¹ is 11.1 on a ~2200 diagonal (5e-3).
    assert!(worst < 1e-6, "H̃ vs FD Hessian rel {worst:e}");
}

/// Posterior moments of `p(η | y) ∝ exp(−individual_nll)` on a 2-D grid —
/// the reference for the ODE chain test. Deterministic and independent of both
/// MH kernels, so it adjudicates the IMH chain without a second sampler's
/// Monte-Carlo error. Returns `(mean, sd)` per coordinate and the largest
/// edge-to-peak density ratio, which must be negligible for the box to hold
/// the mass.
fn grid_posterior_moments(
    model: &CompiledModel,
    subject: &Subject,
    lo: [f64; 2],
    hi: [f64; 2],
    n: [usize; 2],
) -> ([(f64, f64); 2], f64) {
    let theta = model.default_params.theta.clone();
    let sigma = model.default_params.sigma.values.clone();
    let omega = omega2(0.09, 0.09);
    let step = |d: usize| (hi[d] - lo[d]) / (n[d] - 1) as f64;
    let (na, nb) = (n[0], n[1]);
    let mut nll = vec![0.0_f64; na * nb];
    for a in 0..na {
        for b in 0..nb {
            let e = [lo[0] + a as f64 * step(0), lo[1] + b as f64 * step(1)];
            let v = individual_nll(model, subject, &theta, &e, &omega, &sigma);
            assert!(v.is_finite(), "non-finite NLL at {e:?}");
            nll[a * nb + b] = v;
        }
    }
    let min = nll.iter().copied().fold(f64::INFINITY, f64::min);
    let w: Vec<f64> = nll.iter().map(|v| (min - v).exp()).collect();
    let mut edge = 0.0_f64;
    for a in 0..na {
        for b in 0..nb {
            if a == 0 || b == 0 || a == na - 1 || b == nb - 1 {
                edge = edge.max(w[a * nb + b]);
            }
        }
    }
    let z: f64 = w.iter().sum();
    let mut out = [(0.0, 0.0); 2];
    for d in 0..2 {
        let coord = |a: usize, b: usize| lo[d] + [a, b][d] as f64 * step(d);
        let (mut m1, mut m2) = (0.0, 0.0);
        for a in 0..na {
            for b in 0..nb {
                let x = coord(a, b);
                m1 += w[a * nb + b] * x;
                m2 += w[a * nb + b] * x * x;
            }
        }
        let mean = m1 / z;
        out[d] = (mean, (m2 / z - mean * mean).sqrt());
    }
    (out, edge)
}

/// The quadrature reference for [`mm_well_conditioned_subject`], computed once
/// per test binary. The box is wide in `η_VM` because the MM conditional has an
/// exponential, not Gaussian, left tail (low `VM` saturates elimination, so the
/// profile NLL rises ~20 per unit of `η_VM` there rather than quadratically):
/// it is still 1e-5 of the peak at `η_VM = −0.1`, eight conditional SDs out.
/// A 0.02 step (~2 points per conditional SD) suffices: the trapezoid rule on
/// this smooth integrand agrees with a 0.01 grid to 6e-7 in every moment.
fn mm_quadrature_reference() -> [(f64, f64); 2] {
    static REF: std::sync::OnceLock<[(f64, f64); 2]> = std::sync::OnceLock::new();
    *REF.get_or_init(|| {
        let model = parse_model_string(MM_ODE_MODEL).expect("parse");
        let subject = mm_well_conditioned_subject();
        let (want, edge) =
            grid_posterior_moments(&model, &subject, [-1.6, -0.8], [0.6, 0.6], [111, 71]);
        assert!(
            edge < 1e-9,
            "grid box truncates the posterior (edge/peak {edge:e})"
        );
        want
    })
}

/// The IMH chain-invariance oracle on an ODE subject (#2639): the kernel built
/// from the dual ODE walk's curvature must sample the multi-dose MM conditional,
/// pinned against grid quadrature of `exp(−individual_nll)` — the same value
/// path the acceptance ratio reads, integrated without any sampler.
#[test]
fn ode_imh_chain_matches_the_quadrature_posterior() {
    let model = parse_model_string(MM_ODE_MODEL).expect("parse");
    assert!(model.ode_spec.is_some());
    let subject = mm_well_conditioned_subject();
    let want = mm_quadrature_reference();
    let (i0, i1) = chain_moments_imh_on(&model, &subject, 20_000, 11, false);
    // Quadrature: η_VM 0.2496 ± 0.0409, η_V −0.1064 ± 0.0784. Realised over
    // seeds 11–16: |Δmean| ≤ 0.025 conditional SDs, |Δsd|/sd ≤ 0.017; with the
    // q-ratio dropped |Δsd|/sd = 0.34 (the control below). Bounds: mean 0.1 SD,
    // sd 0.08 — ≥ 4× the realised worst and 4× below the defect.
    for (imh, (mean, sd), name) in [(i0, want[0], "ETA_VM"), (i1, want[1], "ETA_V")] {
        assert!(imh.mean.is_finite() && mean.is_finite() && sd > 0.0);
        let dm = (imh.mean - mean).abs() / sd;
        let ds = (imh.sd() - sd).abs() / sd;
        assert!(
            dm < 0.1,
            "{name}: IMH mean {:.6} vs grid {mean:.6} ({dm:.3} sd)",
            imh.mean
        );
        assert!(
            ds < 0.08,
            "{name}: IMH sd {:.6} vs grid {sd:.6} (rel {ds:.3})",
            imh.sd()
        );
    }
}

/// [`ode_imh_chain_matches_the_quadrature_posterior`]'s control: the same chain
/// with the `q`-ratio dropped must miss the quadrature SDs by far more than the
/// bound there, or that test has stopped testing the acceptance ratio.
#[test]
fn ode_dropping_the_q_ratio_visibly_biases_the_chain() {
    let model = parse_model_string(MM_ODE_MODEL).expect("parse");
    let subject = mm_well_conditioned_subject();
    let want = mm_quadrature_reference();
    let (b0, b1) = chain_moments_imh_on(&model, &subject, 20_000, 11, true);
    let worst = [(b0, want[0]), (b1, want[1])]
        .iter()
        .map(|(b, (_, sd))| {
            assert!(b.sd().is_finite() && *sd > 0.0);
            (b.sd() - sd).abs() / sd
        })
        .fold(0.0_f64, f64::max);
    // Realised 0.34 (seed 11); the bound in the test above is 0.08.
    assert!(
        worst > 0.2,
        "dropping the q-ratio did not move the chain (worst sd rel {worst:.3})"
    );
}

/// An ODE fit takes the kernel end to end (#2639): every refresh is served by
/// the dual ODE walk (no declines, so no subject silently falls back to the
/// random walk), and the block random walk never runs.
#[test]
fn an_ode_fit_runs_the_kernel_without_declines() {
    let model = parse_model_string(MM_ODE_MODEL).expect("parse");
    let pop = population_of(&mm_well_conditioned_subject(), 4);
    let r = run_saem(&model, &pop, &model.default_params, &imh_opts(7)).expect("SAEM runs");
    let used = last_imh();
    assert_eq!(used.proposals, 12 * 4 * 2, "{used:?}");
    assert_eq!(used.refreshes, 6 * 4, "{used:?}");
    assert_eq!(
        used.declines, 0,
        "an ODE subject declined its refresh: {used:?}"
    );
    assert!(used.accepts > 0, "{used:?}");
    assert!(r.ofv.is_finite());
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("saem_kernel = imh requested")),
        "{:?}",
        r.warnings
    );
    assert_eq!(
        LAST_RW_BLOCK_CALLS.with(|c| c.get()),
        0,
        "the random walk ran under IMH"
    );
}
