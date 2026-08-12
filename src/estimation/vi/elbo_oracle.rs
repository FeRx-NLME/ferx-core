//! Exact-case oracle for VI (VI_PLAN §6 test 2, and its IOV extension §10.5 test 2).
//!
//! Every other VI test checks *internal consistency* — the analytic gradient against
//! finite differences of the same objective, the MC KL against the closed-form KL.
//! Those all pass if the implementation is self-consistently wrong. This module is the
//! one check with an **external** truth: a model on which the correct answer is known
//! in closed form, independent of anything in `vi/`.
//!
//! # The construction
//!
//! Take a 1-cpt IV bolus with a *constant* volume and an **additive** `η` on clearance,
//! observed with `log_additive` residual error:
//!
//! ```text
//!   CL = TVCL + ETA_CL      V = V0 (constant)
//!   log f(t) = log(D/V0) − (t/V0)·(TVCL + ETA_CL)
//!            = c_t + a_t·η          c_t = log(D/V0) − (TVCL/V0)·t,  a_t = −t/V0
//! ```
//!
//! The log-prediction is **exactly affine in `η`**, and `log_additive` puts Gaussian
//! error on that same log scale. So the model is linear-Gaussian, and therefore:
//!
//! * the true posterior `p(η|y)` is Gaussian — a full-rank Gaussian `q` can represent
//!   it *exactly*, so the ELBO's bound gap `KL(q ‖ p(η|y))` is zero at the optimum;
//! * `−2·ELBO` at that `q` equals the exact `−2 log p(y)`;
//! * both have closed forms this module computes independently.
//!
//! This is the anchor a NONMEM run cannot provide: NONMEM would give another
//! *estimate*, whereas here the exact answer is available in arithmetic.
//!
//! # Why the tolerances are not `1e-8`
//!
//! Under the analytic-KL route the KL term is exact, but the data term
//! `E_q[log p(y|η)]` is still estimated by Monte Carlo. For this model the integrand is
//! quadratic in `η`, so the MC error is driven entirely by how far the drawn `ε`'s
//! sample moments sit from `(0, 1)` — deterministic for a fixed seed, but not zero.
//! The tolerances below are therefore MC tolerances, tightened by raising
//! `n_mc_samples`, not exactness tolerances. The *moment* assertions (`μ`, `S`) are
//! unaffected by this and are checked tightly.

use super::*;
use crate::estimation::parameterization::pack_params;
use crate::estimation::vi::family::FullRank;
use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;

/// Dose and (constant) volume baked into the fixture below.
const DOSE: f64 = 100.0;
const V0: f64 = 10.0;
const TVCL: f64 = 1.0;
/// Prior variance of `η` declared by the fixture.
const OMEGA: f64 = 0.09;
/// Residual SD on the log scale declared by the fixture.
const SIGMA: f64 = 0.2;

/// Observation times shared by every subject in the fixture.
///
/// Deliberately early, so `log f(t)` stays comfortably **positive** over the whole
/// range of `η` the sampler visits. That is not cosmetic: the data term stops being
/// quadratic in `η` once the log-scale prediction crosses zero (see
/// `ltbs_data_term_is_quadratic_below_unit_predictions`), which would make this
/// oracle fail for a reason that has nothing to do with VI.
const TIMES: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

/// `Σ aₜ²` for [`TIMES`] — the design quantity every closed form below needs.
fn sum_a_squared() -> f64 {
    TIMES.iter().map(|&t| slope(t) * slope(t)).sum()
}

/// The linear-Gaussian model described in the module docs.
///
/// `V` is a bare constant so no second random effect enters, and the `η` on `CL` is
/// **additive** — a lognormal `exp(η)` would make the log-prediction nonlinear in `η`
/// and destroy the exactness this whole module rests on.
fn linear_gaussian_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma ADD_LOG ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL + ETA_CL
  V  = 10.0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ log_additive(ADD_LOG)
",
    )
    .expect("linear-Gaussian oracle fixture parses")
}

/// One subject. `observations` are already on the **log** scale: `log_additive`
/// sets `dv_pre_logged`, so `fit()` must not transform them again.
fn oracle_subject(id: &str, obs: &[f64]) -> Subject {
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)],
        obs_times: TIMES.to_vec(),
        obs_raw_times: Vec::new(),
        observations: obs.to_vec(),
        obs_cmts: vec![1; TIMES.len()],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; TIMES.len()],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Two subjects whose log-scale observations sit deliberately off the `η = 0`
/// prediction, so the exact posterior mean is nonzero in both directions.
fn oracle_population() -> Population {
    let c: Vec<f64> = TIMES.iter().map(|&t| intercept(t)).collect();
    let up: Vec<f64> = c.iter().map(|v| v + 0.18).collect();
    let dn: Vec<f64> = c.iter().map(|v| v - 0.11).collect();
    Population {
        subjects: vec![oracle_subject("1", &up), oracle_subject("2", &dn)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// `c_t` — the log-prediction at `η = 0`.
fn intercept(t: f64) -> f64 {
    (DOSE / V0).ln() - (TVCL / V0) * t
}

/// `a_t` — the exact `∂ log f(t) / ∂η`.
fn slope(t: f64) -> f64 {
    -t / V0
}

/// Exact Gaussian posterior `p(η | y)` for one subject: `(mean, variance)`.
///
/// `prec = 1/ω + Σ aₜ²/σ²`, `mean = (Σ aₜ·rₜ/σ²) / prec`, with `rₜ = yₜ − cₜ`.
fn exact_posterior(obs: &[f64]) -> (f64, f64) {
    let s2 = SIGMA * SIGMA;
    let mut ata = 0.0;
    let mut atr = 0.0;
    for (k, &t) in TIMES.iter().enumerate() {
        let a = slope(t);
        let r = obs[k] - intercept(t);
        ata += a * a;
        atr += a * r;
    }
    let prec = 1.0 / OMEGA + ata / s2;
    let var = 1.0 / prec;
    (var * atr / s2, var)
}

/// Exact `−2 log p(y)` for one subject.
///
/// `y ~ N(c, σ²I + ω·aaᵀ)`. The rank-1 covariance is inverted with the
/// Sherman–Morrison / matrix-determinant identities rather than by forming the
/// matrix, so this stays independent of any linear algebra in `vi/`.
fn exact_neg_two_log_marginal(obs: &[f64]) -> f64 {
    let s2 = SIGMA * SIGMA;
    let n = TIMES.len() as f64;
    let mut ata = 0.0;
    let mut atr = 0.0;
    let mut rtr = 0.0;
    for (k, &t) in TIMES.iter().enumerate() {
        let a = slope(t);
        let r = obs[k] - intercept(t);
        ata += a * a;
        atr += a * r;
        rtr += r * r;
    }
    // |Σ| = σ^{2n}·(1 + ω·aᵀa/σ²)
    let log_det = n * s2.ln() + (1.0 + OMEGA * ata / s2).ln();
    // Σ⁻¹ = (1/σ²)·(I − ω·aaᵀ/(σ² + ω·aᵀa))
    let quad = (rtr - OMEGA * atr * atr / (s2 + OMEGA * ata)) / s2;
    // NOTE: the `n·log(2π)` constant is deliberately **omitted**, because ferx's
    // observation NLL omits it too — the same convention NONMEM prints as "REPORTED
    // OBJECTIVE FUNCTION DOES NOT CONTAIN CONSTANT". Comparing a full `−2 log p(y)`
    // against ferx's objective would be off by exactly `n_obs·log(2π)` and say nothing
    // about correctness. Verified against `obs_nll_subject_into` to 1e-8.
    log_det + quad
}

/// `φ` for [`FullRank`] encoding exactly `N(mean, var)` in one dimension:
/// `φ = [μ, log L]` with `L = sqrt(var)`.
fn full_rank_phi(mean: f64, var: f64) -> Vec<f64> {
    vec![mean, 0.5 * var.ln()]
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

/// **Self-check on the construction itself.** Everything downstream assumes the
/// log-prediction is affine in `η`; if the parser, the additive-`η` path, or the
/// `log_additive` wrapping ever stopped delivering that, the oracle would silently
/// become an oracle for the wrong model. Second differences of an affine function
/// vanish, so that is what is asserted — on the *production* predictor, not a
/// reimplementation.
#[test]
fn oracle_model_is_affine_in_eta() {
    let model = linear_gaussian_model();
    let pop = oracle_population();
    let params = model.default_params.clone();
    let subject = &pop.subjects[0];

    let mut scratch = crate::pk::EventPkParams::default();
    // The production predictor — the same call `obs_nll_subject_into` makes — so this
    // asserts affineness of what VI actually evaluates, not of a stand-in.
    let pred_at = |eta: f64, scratch: &mut crate::pk::EventPkParams| -> Vec<f64> {
        crate::pk::compute_predictions_with_tv_into(&model, subject, &params.theta, &[eta], scratch)
    };

    let h = 0.25;
    let lo = pred_at(-h, &mut scratch);
    let mid = pred_at(0.0, &mut scratch);
    let hi = pred_at(h, &mut scratch);

    for k in 0..TIMES.len() {
        // Affine ⇒ f(-h) - 2f(0) + f(h) == 0.
        let second_diff = lo[k] - 2.0 * mid[k] + hi[k];
        assert!(
            second_diff.abs() < 1e-9,
            "log-prediction must be affine in eta at t={}: second difference {:.3e}",
            TIMES[k],
            second_diff
        );
        // And the slope must be the `a_t` the closed forms use.
        let fd_slope = (hi[k] - lo[k]) / (2.0 * h);
        assert!(
            (fd_slope - slope(TIMES[k])).abs() < 1e-9,
            "d log f/d eta at t={} is {:.9}, closed form says {:.9}",
            TIMES[k],
            fd_slope,
            slope(TIMES[k])
        );
    }
}

/// The oracle proper: at the **exact** posterior, full-rank VI's `−2·ELBO` equals the
/// exact `−2 log p(y)`.
///
/// A bound that is loose here is a bug, not an approximation: the variational family
/// contains the true posterior, so the gap must vanish. This catches sign errors in the
/// KL, a mis-scaled data term, and a prior that is being counted twice — none of which
/// the FD-parity tests can see, because FD parity holds for a consistently wrong
/// objective.
#[test]
fn full_rank_vi_is_exact_on_a_linear_gaussian_model() {
    let model = linear_gaussian_model();
    let pop = oracle_population();
    let template = model.default_params.clone();

    let family = FullRank::new(1);
    let phis: Vec<Vec<f64>> = pop
        .subjects
        .iter()
        .map(|s| {
            let (mean, var) = exact_posterior(&s.observations);
            full_rank_phi(mean, var)
        })
        .collect();

    let exact: f64 = pop
        .subjects
        .iter()
        .map(|s| exact_neg_two_log_marginal(&s.observations))
        .sum();

    let cfg = ElboConfig {
        // Large enough that the MC error on the quadratic data term is well under the
        // 1e-3 relative tolerance below. See the module docs on why this is not exact.
        n_mc_samples: 4000,
        eta_grad: EtaGradMode::Auto,
        kl: KlMode::Analytic,
        seed: 20250811,
    };

    let x = pack_params(&template);
    let eval = population_neg_elbo(&model, &pop, &template, &x, &family, &phis, &cfg, 0)
        .expect("linear-Gaussian model is inside VI's support scope");

    let neg_two_elbo = 2.0 * eval.neg_elbo;
    let rel = (neg_two_elbo - exact).abs() / exact.abs();
    assert!(
        rel < 1e-3,
        "-2*ELBO at the exact posterior = {neg_two_elbo:.6}, exact -2 log p(y) = {exact:.6} \
         (relative gap {rel:.3e}); the variational family CONTAINS this posterior, so the \
         bound must be tight"
    );

    // The ELBO is a lower bound on log p(y), so -2*ELBO can only sit at or *above*
    // the exact value. Landing below it means the objective is not the ELBO.
    assert!(
        neg_two_elbo > exact - 1e-2,
        "-2*ELBO ({neg_two_elbo:.6}) fell below the exact -2 log p(y) ({exact:.6}); a lower \
         bound on log p(y) cannot exceed it"
    );
}

/// The exact posterior is a **stationary point** of the ELBO in `φ`.
///
/// Complements the value check above: a gradient that does not vanish at the known
/// optimum means the `φ` route (reparameterization chain + KL derivative) is wrong even
/// if the value happens to come out right.
#[test]
fn phi_gradient_vanishes_at_the_exact_posterior() {
    let model = linear_gaussian_model();
    let pop = oracle_population();
    let template = model.default_params.clone();

    let family = FullRank::new(1);
    let phis: Vec<Vec<f64>> = pop
        .subjects
        .iter()
        .map(|s| {
            let (mean, var) = exact_posterior(&s.observations);
            full_rank_phi(mean, var)
        })
        .collect();

    let cfg = ElboConfig {
        n_mc_samples: 4000,
        eta_grad: EtaGradMode::Auto,
        kl: KlMode::Analytic,
        seed: 20250811,
    };

    let x = pack_params(&template);
    let eval = population_neg_elbo(&model, &pop, &template, &x, &family, &phis, &cfg, 0)
        .expect("linear-Gaussian model is inside VI's support scope");

    // The KL half of this gradient is exact, but the data half is a Monte-Carlo mean
    // whose error does *not* vanish at the optimum, so the bar is an MC tolerance
    // (~3e-2 at 4096 draws), not an exactness one. It is still far tighter than any
    // structural error would produce: double-counting the prior, for instance, leaves
    // `∂/∂μ = μ/ω ≈ 1.3` here — two orders of magnitude above this threshold.
    for (i, g) in eval.grad_phi.iter().enumerate() {
        for (j, gj) in g.iter().enumerate() {
            assert!(
                gj.abs() < 5e-2,
                "subject {i} phi-gradient component {j} = {gj:.3e} at the exact posterior; \
                 the known optimum must be stationary"
            );
        }
    }
}

/// `closed_form_omega` must reproduce the `Ω` that the exact posteriors imply:
/// `Ω* = (1/N)·Σᵢ(Sᵢ + μᵢ²)`.
///
/// Checked against arithmetic done here rather than against the implementation's own
/// helper, so a change to the update rule cannot silently move the target.
#[test]
fn closed_form_omega_matches_the_exact_posterior_moments() {
    let model = linear_gaussian_model();
    let pop = oracle_population();
    let template = model.default_params.clone();

    let family = FullRank::new(1);
    let mut expected = 0.0;
    let phis: Vec<Vec<f64>> = pop
        .subjects
        .iter()
        .map(|s| {
            let (mean, var) = exact_posterior(&s.observations);
            expected += var + mean * mean;
            full_rank_phi(mean, var)
        })
        .collect();
    expected /= pop.subjects.len() as f64;

    let omega = closed_form_omega(&family, &phis, &template);
    let got = omega.matrix[(0, 0)];
    assert!(
        (got - expected).abs() < 1e-12,
        "closed_form_omega gave {got:.12}, exact posterior moments give {expected:.12}"
    );
}

// ---------------------------------------------------------------------------
// LTBS regression: the data term must stay quadratic below unit predictions
// ---------------------------------------------------------------------------

/// Under LTBS the log-scale prediction is legitimately **negative** whenever the
/// natural-scale concentration falls below one unit — routine for ng/mL data, late
/// samples, or a high-clearance subject. The residual `y − log f` must stay a plain
/// affine function of `η` there, so the observation NLL stays quadratic in `η`.
///
/// It does not. Once `log f(t)` crosses zero the NLL behaves exactly as though the
/// *log-scale* prediction were floored at `0` (equivalently, the natural-scale
/// prediction floored at `1.0`) — even though `compute_predictions_with_tv_into`
/// returns the correct negative log-prediction to machine precision (verified to
/// 4e-16). The discrepancy matches `½[(y − max(log f, 0))² − (y − log f)²]/σ²` to six
/// decimals, which is how it was identified.
///
/// This is **not** VI-specific: it lives in the observation NLL, so every estimator
/// that scores an LTBS model with sub-unit predictions inherits it. And no
/// `Dual2`-vs-FD parity test can see it, because finite differences of a consistently
/// floored objective are themselves consistently floored — which is precisely why the
/// exact-case oracle in this module was needed to surface it.
///
/// Ignored rather than deleted: it reproduces a real defect and should be un-ignored by
/// the fix, not rewritten to match today's behaviour.
#[test]
#[ignore = "reproduces an open defect: LTBS residual floors at log f = 0 (see doc comment)"]
fn ltbs_data_term_is_quadratic_below_unit_predictions() {
    let model = linear_gaussian_model();
    let template = model.default_params.clone();
    let s2 = SIGMA * SIGMA;

    // Observations sit on the eta = 0 log-prediction, so residuals are driven purely by
    // the eta excursion below.
    let obs: Vec<f64> = TIMES.iter().map(|&t| intercept(t)).collect();
    let subj = oracle_subject("ltbs", &obs);

    let mut scratch = crate::pk::EventPkParams::default();
    let h = 0.1;
    // Second difference of a quadratic is constant and equal to (sum a^2 / sigma^2)*h^2.
    let expected = (sum_a_squared() / s2) * h * h;

    // `log f(5) = log(10) - 0.5*(1 + eta)`, so it crosses zero near eta = 3.6. Centers
    // straddle that crossing: the first two are safe, the last two are not.
    for &center in &[0.0f64, 2.0, 5.0, 8.0] {
        let mut ev = |e: f64| {
            crate::stats::likelihood::obs_nll_subject_into(
                &model,
                &subj,
                &template.theta,
                &template.sigma.values,
                &[e],
                &mut scratch,
            )
        };
        let second_diff = ev(center - h) - 2.0 * ev(center) + ev(center + h);
        assert!(
            (second_diff - expected).abs() < 1e-9,
            "LTBS data term must stay quadratic in eta at center={center}: second \
             difference {second_diff:.8}, expected {expected:.8}. A departure means the \
             residual is being floored where log f goes negative."
        );
    }
}

// ---------------------------------------------------------------------------
// IOV extension (VI_PLAN §10.5 test 2)
// ---------------------------------------------------------------------------
//
// Same idea as the oracle above, over the **stacked** random-effects vector
// `z = [η, κ₁ … κ_K]`. `CL_g = TVCL + η + κ_g` is per-occasion, so under LTBS the
// log-prediction stays affine in `z` — even though the concentration decays piecewise,
// with a different clearance in each occasion. The prior is block diagonal,
// `Σ_b = Ω ⊕ Ω_iov^{⊗K}`, so the true posterior over `z` is again Gaussian and a
// full-rank `q` over the stacked vector is exact.
//
// The design matrix is **measured** from the production IOV predictor rather than
// derived here. Assuming a formula for how clearance switches at an occasion boundary
// would make this an oracle for my assumption instead of for ferx: the affineness check
// below is a genuine test, and everything downstream is then anchored on what the engine
// actually computes.

use crate::estimation::vi::family::{n_tril, tril_index};

/// Prior variance of each `κ` in the IOV fixture.
const OMEGA_IOV: f64 = 0.04;
/// Observation times for the IOV fixture, three per occasion.
const IOV_TIMES: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// Occasion index per observation in [`IOV_TIMES`].
const IOV_OCCASIONS: [u32; 6] = [1, 1, 1, 2, 2, 2];
/// Number of occasions, i.e. the number of `κ` blocks in the stacked vector.
const K_OCC: usize = 2;

fn linear_gaussian_iov_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma ADD_LOG ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL + ETA_CL + KAPPA_CL
  V  = 10.0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ log_additive(ADD_LOG)

[fit_options]
  iov_column = OCC
",
    )
    .expect("linear-Gaussian IOV oracle fixture parses")
}

/// One IOV subject: a dose at the start of each occasion, three log-scale observations
/// per occasion.
fn iov_subject(id: &str, obs: &[f64]) -> Subject {
    Subject {
        id: id.into(),
        // A **single** bolus, deliberately. A second dose would make the concentration a
        // sum of two exponentials, and `log(Σ exp)` is not affine in `z` — the affineness
        // check catches exactly that. With one dose the amount is a single exponential
        // whose exponent accumulates `CL_g` over each occasion's elapsed time, so the
        // log-prediction stays affine even though clearance switches partway through.
        doses: vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)],
        obs_times: IOV_TIMES.to_vec(),
        obs_raw_times: Vec::new(),
        observations: obs.to_vec(),
        obs_cmts: vec![1; IOV_TIMES.len()],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; IOV_TIMES.len()],
        occasions: IOV_OCCASIONS.to_vec(),
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Split a stacked `z = [η, κ₁ … κ_K]` and evaluate the production IOV predictor.
fn iov_predict(model: &CompiledModel, subject: &Subject, theta: &[f64], z: &[f64]) -> Vec<f64> {
    let kappas: Vec<Vec<f64>> = (0..K_OCC).map(|g| vec![z[1 + g]]).collect();
    crate::pk::predict_iov(model, subject, theta, &z[..1], &kappas)
}

/// Measure the affine decomposition `log f(t) = c_t + Σ_j A[t][j]·z_j` of the production
/// predictor, and assert it really is affine.
///
/// Central differences are *exact* for an affine function, so the returned `A` is the
/// true design matrix, not an approximation — which is what lets the closed forms below
/// be exact rather than FD-accurate.
fn measure_iov_design(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    d: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = subject.obs_times.len();
    let zero = vec![0.0; d];
    let c = iov_predict(model, subject, theta, &zero);

    let h = 0.25;
    let mut a = vec![vec![0.0; d]; n];
    for j in 0..d {
        let mut zp = zero.clone();
        let mut zm = zero.clone();
        zp[j] += h;
        zm[j] -= h;
        let up = iov_predict(model, subject, theta, &zp);
        let dn = iov_predict(model, subject, theta, &zm);
        for k in 0..n {
            let second = up[k] - 2.0 * c[k] + dn[k];
            assert!(
                second.abs() < 1e-9,
                "IOV log-prediction must be affine in stacked coordinate {j} at obs {k}: \
                 second difference {second:.3e}"
            );
            a[k][j] = (up[k] - dn[k]) / (2.0 * h);
        }
    }
    (c, a)
}

/// Block-diagonal stacked prior `Σ_b = Ω ⊕ Ω_iov^{⊗K}`.
fn stacked_prior(d: usize) -> DMatrix<f64> {
    let mut s = DMatrix::zeros(d, d);
    s[(0, 0)] = OMEGA;
    for g in 0..K_OCC {
        s[(1 + g, 1 + g)] = OMEGA_IOV;
    }
    s
}

/// Exact Gaussian posterior over the stacked vector: `S = (AᵀA/σ² + Σ_b⁻¹)⁻¹`,
/// `μ = S·Aᵀr/σ²`.
fn exact_iov_posterior(
    c: &[f64],
    a: &[Vec<f64>],
    obs: &[f64],
    d: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let s2 = SIGMA * SIGMA;
    let n = obs.len();
    let amat = DMatrix::from_fn(n, d, |i, j| a[i][j]);
    let r = DVector::from_fn(n, |i, _| obs[i] - c[i]);

    let prec = amat.transpose() * &amat / s2
        + stacked_prior(d)
            .try_inverse()
            .expect("block-diagonal prior is invertible");
    let cov = prec
        .try_inverse()
        .expect("posterior precision is invertible");
    let mean = &cov * (amat.transpose() * r) / s2;
    (mean, cov)
}

/// Exact `−2 log p(y)` for one IOV subject, on ferx's constant-free convention
/// (see [`exact_neg_two_log_marginal`]). `y ~ N(c, σ²I + A·Σ_b·Aᵀ)`.
fn exact_iov_neg_two_log_marginal(c: &[f64], a: &[Vec<f64>], obs: &[f64], d: usize) -> f64 {
    let s2 = SIGMA * SIGMA;
    let n = obs.len();
    let amat = DMatrix::from_fn(n, d, |i, j| a[i][j]);
    let r = DVector::from_fn(n, |i, _| obs[i] - c[i]);
    let sigma = DMatrix::identity(n, n) * s2 + &amat * stacked_prior(d) * amat.transpose();
    let chol = sigma.clone().cholesky().expect("marginal covariance is PD");
    let log_det = 2.0 * chol.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
    let quad = r.dot(&chol.solve(&r));
    log_det + quad
}

/// `φ` for a `d`-dimensional [`FullRank`]: `[μ | vech(L)]`, diagonal stored as `log`.
fn full_rank_phi_nd(mean: &DVector<f64>, cov: &DMatrix<f64>) -> Vec<f64> {
    let d = mean.len();
    let l = cov
        .clone()
        .cholesky()
        .expect("posterior covariance is PD")
        .l();
    let mut phi = vec![0.0; d + n_tril(d)];
    phi[..d].copy_from_slice(mean.as_slice());
    for i in 0..d {
        for j in 0..=i {
            phi[d + tril_index(i, j)] = if i == j { l[(i, i)].ln() } else { l[(i, j)] };
        }
    }
    phi
}

fn iov_population() -> Population {
    let model = linear_gaussian_iov_model();
    let params = model.default_params.clone();
    let probe = iov_subject("probe", &[0.0; 6]);
    let (c, _) = measure_iov_design(&model, &probe, &params.theta, 1 + K_OCC);
    let up: Vec<f64> = c.iter().map(|v| v + 0.16).collect();
    let dn: Vec<f64> = c.iter().map(|v| v - 0.09).collect();
    Population {
        subjects: vec![iov_subject("1", &up), iov_subject("2", &dn)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// The fixture really is linear-Gaussian in the stacked vector — and the measured design
/// matrix has the structure IOV implies.
///
/// Runs today: it exercises only the IOV predictor, so the scaffolding the ignored oracle
/// depends on is validated now rather than when IOV support lands.
#[test]
fn iov_oracle_model_is_affine_in_the_stacked_random_effects() {
    let model = linear_gaussian_iov_model();
    let params = model.default_params.clone();
    let subject = iov_subject("1", &[0.0; 6]);
    let d = 1 + K_OCC;

    // `measure_iov_design` asserts affineness in every stacked coordinate.
    let (_c, a) = measure_iov_design(&model, &subject, &params.theta, d);

    // η acts on every observation; each κ acts only from its own occasion onward, so a
    // first-occasion observation must be completely insensitive to κ₂.
    for (k, &occ) in IOV_OCCASIONS.iter().enumerate() {
        assert!(
            a[k][0].abs() > 1e-9,
            "obs {k} must respond to the BSV eta, got {:.3e}",
            a[k][0]
        );
        if occ == 1 {
            assert!(
                a[k][2].abs() < 1e-12,
                "obs {k} is in occasion 1 and must not respond to kappa_2, got {:.3e}",
                a[k][2]
            );
        }
    }
    // A second-occasion observation must respond to both kappas: drug carried over from
    // occasion 1 was cleared at occasion 1's clearance.
    let last = IOV_TIMES.len() - 1;
    assert!(
        a[last][1].abs() > 1e-9 && a[last][2].abs() > 1e-9,
        "a late observation must respond to both kappas, got k1={:.3e} k2={:.3e}",
        a[last][1],
        a[last][2]
    );
}

/// The IOV oracle: with `q` set to the exact stacked posterior, `−2·ELBO` must equal the
/// exact `−2 log p(y)`.
///
/// This is the gate VI_PLAN §10.6 asks for. VI's `Ω_iov` update is a mean of `S + μμᵀ`
/// over occasions, so a variational posterior that understates `S` biases `Ω_iov`
/// **downward** — and with only a handful of occasions per subject there is far less
/// averaging to wash that out than `Ω_bsv` gets from N subjects. On a linear-Gaussian
/// model the variational family contains the truth, so any gap here is implementation
/// error rather than approximation error, which is what makes it a usable gate.
///
/// Ignored until IOV support lands (`vi/elbo.rs` currently refuses `n_kappa > 0`). It is
/// written against the stacked-family contract of §10.1: one `FullRank` of dimension
/// `n_eta + K·n_kappa`, evaluated against a block-diagonal stacked prior. Every subject
/// here has the same `K`, so this compiles against today's single-family signature; a
/// per-subject-`K` fixture will need the `Vec<Box<dyn VariationalFamily>>` of §10.3.
///
/// Run with `--ignored` today it panics inside nalgebra with `Gemv: dimensions mismatch`
/// rather than failing an assertion: the stacked `φ` has dimension `n_eta + K·n_kappa`
/// while the data term still assumes `n_eta`. That mismatch *is* the §10.2 work — the
/// draw has to be split into `η` plus per-occasion `κ` and routed through
/// `obs_nll_subject_grad_iov`. A clean `Err` from `unsupported_data_term_reason` would be
/// preferable to a panic, and is worth adding to `population_neg_elbo` alongside the
/// existing `run_vi` check.
#[test]
#[ignore = "enable with IOV support (VI_PLAN §10): vi/elbo.rs refuses n_kappa > 0 today"]
fn full_rank_vi_is_exact_on_a_linear_gaussian_iov_model() {
    let model = linear_gaussian_iov_model();
    let pop = iov_population();
    let template = model.default_params.clone();
    let d = 1 + K_OCC;

    let family = FullRank::new(d);
    let mut phis = Vec::with_capacity(pop.subjects.len());
    let mut exact = 0.0;
    for subj in &pop.subjects {
        let (c, a) = measure_iov_design(&model, subj, &template.theta, d);
        let (mean, cov) = exact_iov_posterior(&c, &a, &subj.observations, d);
        phis.push(full_rank_phi_nd(&mean, &cov));
        exact += exact_iov_neg_two_log_marginal(&c, &a, &subj.observations, d);
    }

    let cfg = ElboConfig {
        n_mc_samples: 4000,
        eta_grad: EtaGradMode::Auto,
        kl: KlMode::Analytic,
        seed: 20250812,
    };
    let x = pack_params(&template);
    let eval = population_neg_elbo(&model, &pop, &template, &x, &family, &phis, &cfg, 0)
        .expect("IOV model must be inside VI's support scope once §10 lands");

    let neg_two_elbo = 2.0 * eval.neg_elbo;
    let rel = (neg_two_elbo - exact).abs() / exact.abs();
    assert!(
        rel < 1e-3,
        "-2*ELBO at the exact stacked posterior = {neg_two_elbo:.6}, exact -2 log p(y) = \
         {exact:.6} (relative gap {rel:.3e}); the family contains this posterior, so the \
         bound must be tight"
    );
    assert!(
        neg_two_elbo > exact - 1e-2,
        "-2*ELBO ({neg_two_elbo:.6}) fell below the exact -2 log p(y) ({exact:.6})"
    );
}
