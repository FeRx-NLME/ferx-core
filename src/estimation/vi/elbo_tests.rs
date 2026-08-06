//! Tests for the ELBO value, its gradients, and the support routing.
//!
//! The central check is `neg_elbo_gradient_matches_central_fd`: with the seed and
//! iteration index fixed, `−ELBO` is a *deterministic* function of `(x, φ)`, so
//! central finite differences of it are exact rather than noisy. That is the
//! `Dual2`-vs-FD parity rule applied to the one new gradient path this module
//! introduces.

use super::*;
use crate::estimation::parameterization::pack_params;
use crate::estimation::vi::family::{FullRank, MeanField};
use crate::types::test_helpers::analytical_model;
use crate::types::{DoseEvent, GradientMethod, Population, Subject};
use std::collections::HashMap;

/// A two-subject population on a real closed-form 1-cpt IV model with two random
/// effects.
///
/// Deliberately **not** `types::test_helpers::analytical_model`: that fixture's
/// `pk_param_fn` returns `PkParams::default()`, so its predictions do not depend
/// on `η` at all. Every gradient through the reparameterization path would be
/// identically zero and the parity tests would pass vacuously. Two `η`s rather
/// than one also means `FullRank` has a genuine off-diagonal to exercise.
fn fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("closed-form 1-cpt IV fixture parses");
    let params = model.default_params.clone();

    let make = |id: &str, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0 * scale, 5.0 * scale, 2.5 * scale],
        obs_cmts: vec![1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 3],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let population = Population {
        subjects: vec![make("1", 1.0), make("2", 1.2)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    (model, population, params)
}

/// Deterministic, non-trivial `φ` per subject — off the prior so nothing is
/// evaluated at a stationary point.
fn perturbed_phis(family: &dyn VariationalFamily, omega: &OmegaMatrix, n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|s| {
            let mut p = family.init(omega);
            for (i, v) in p.iter_mut().enumerate() {
                *v += 0.2 * (((i + s * 3) * 7 + 1) as f64).sin();
            }
            p
        })
        .collect()
}

fn cfg_seeded() -> ElboConfig {
    ElboConfig {
        n_mc_samples: 3,
        eta_grad: EtaGradMode::Auto,
        seed: 12345,
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

#[test]
fn packed_layout_matches_pack_params() {
    let (_, _, params) = fixture();
    let layout = PackedLayout::new(&params);
    assert_eq!(
        layout.total(),
        pack_params(&params).len(),
        "layout must account for every packed coordinate"
    );
    assert_eq!(layout.omega_start(), layout.n_theta);
    assert_eq!(layout.sigma_start(), layout.n_theta + layout.n_omega);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

/// Same seed and iteration ⇒ bit-identical result. This is what makes the FD
/// tests below meaningful, and what makes a VI fit reproducible.
#[test]
fn evaluation_is_deterministic_given_seed_and_iteration() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let x = pack_params(&params);
    let cfg = cfg_seeded();

    let a = population_neg_elbo(&model, &population, &params, &x, &family, &phis, &cfg, 7).unwrap();
    let b = population_neg_elbo(&model, &population, &params, &x, &family, &phis, &cfg, 7).unwrap();
    assert_eq!(a.neg_elbo.to_bits(), b.neg_elbo.to_bits());
    for (ga, gb) in a.grad_x.iter().zip(b.grad_x.iter()) {
        assert_eq!(ga.to_bits(), gb.to_bits());
    }

    // A different iteration draws different ε, so the MC term must move.
    let c = population_neg_elbo(&model, &population, &params, &x, &family, &phis, &cfg, 8).unwrap();
    assert!(
        (a.neg_elbo - c.neg_elbo).abs() > 0.0,
        "different iterations must draw different common random numbers"
    );
}

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// At the prior (`φ = init(Ω)`), the KL is zero for the full-rank family, so
/// `−ELBO` is purely the expected data term.
#[test]
fn kl_half_vanishes_at_the_prior() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis: Vec<Vec<f64>> = (0..2).map(|_| family.init(&params.omega)).collect();
    let x = pack_params(&params);

    let e = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        &family,
        &phis,
        &cfg_seeded(),
        1,
    )
    .unwrap();
    assert!(
        e.kl_term.abs() < 1e-9,
        "KL at the prior should be 0, got {}",
        e.kl_term
    );
    assert!((e.neg_elbo - e.data_term).abs() < 1e-9);
    assert!(e.data_term.is_finite() && e.data_term > 0.0);
}

/// A zero-variance `q` collapses the expectation onto its mean, so the data term
/// must equal the ordinary fixed-η observation NLL evaluated at `μ`. This pins
/// the data half against machinery that knows nothing about VI.
#[test]
fn data_term_collapses_to_fixed_eta_nll_as_q_degenerates() {
    use crate::stats::likelihood::obs_nll_subject_into;

    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);

    // μ = (0.3, 0.3) with the covariance driven to ~0. Only the *diagonal* of the
    // Cholesky factor is stored as a log, so the off-diagonal must be set to 0
    // rather than to a large negative — writing -60 everywhere would give a huge
    // off-diagonal and a wildly non-degenerate q.
    let d = model.n_eta;
    let mut phi = family.init(&params.omega);
    for m in phi.iter_mut().take(d) {
        *m = 0.3;
    }
    for i in 0..d {
        for j in 0..=i {
            phi[d + crate::estimation::vi::family::tril_index(i, j)] =
                if i == j { -60.0 } else { 0.0 };
        }
    }
    let phis = vec![phi.clone(), phi];
    let x = pack_params(&params);

    let e = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        &family,
        &phis,
        &cfg_seeded(),
        3,
    )
    .unwrap();

    let mut scratch = crate::pk::EventPkParams::default();
    let mu = vec![0.3; d];
    let want: f64 = population
        .subjects
        .iter()
        .map(|s| {
            obs_nll_subject_into(
                &model,
                s,
                &params.theta,
                &params.sigma.values,
                &mu,
                &mut scratch,
            )
        })
        .sum();

    assert!(
        (e.data_term - want).abs() / want.abs().max(1.0) < 1e-9,
        "degenerate q: data term {} vs fixed-η NLL {want}",
        e.data_term
    );
}

// ---------------------------------------------------------------------------
// Gradient parity — the central check
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn fd_of_neg_elbo_wrt_x(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    x: &[f64],
    family: &dyn VariationalFamily,
    phis: &[Vec<f64>],
    cfg: &ElboConfig,
    iter: u64,
    i: usize,
) -> f64 {
    let h = 1e-6 * (1.0 + x[i].abs());
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[i] += h;
    xm[i] -= h;
    let fp = population_neg_elbo(model, population, params, &xp, family, phis, cfg, iter)
        .unwrap()
        .neg_elbo;
    let fm = population_neg_elbo(model, population, params, &xm, family, phis, cfg, iter)
        .unwrap()
        .neg_elbo;
    (fp - fm) / (2.0 * h)
}

#[allow(clippy::too_many_arguments)]
fn fd_of_neg_elbo_wrt_phi(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    x: &[f64],
    family: &dyn VariationalFamily,
    phis: &[Vec<f64>],
    cfg: &ElboConfig,
    iter: u64,
    subj: usize,
    i: usize,
) -> f64 {
    let h = 1e-6 * (1.0 + phis[subj][i].abs());
    let mut pp = phis.to_vec();
    let mut pm = phis.to_vec();
    pp[subj][i] += h;
    pm[subj][i] -= h;
    let fp = population_neg_elbo(model, population, params, x, family, &pp, cfg, iter)
        .unwrap()
        .neg_elbo;
    let fm = population_neg_elbo(model, population, params, x, family, &pm, cfg, iter)
        .unwrap()
        .neg_elbo;
    (fp - fm) / (2.0 * h)
}

/// Every coordinate of `∂(−ELBO)/∂x` and `∂(−ELBO)/∂φ` against central finite
/// differences of `−ELBO`, for both variational families.
///
/// The `θ` and `σ` blocks flow from `obs_nll_subject_grad`, the `Ω` block from
/// the analytic KL chained through the Cholesky packing, and the `φ` gradients
/// from the reparameterization path plus the KL — four independent assemblies
/// that this single test covers end to end.
///
/// # Why the `θ` block gets a looser tolerance
///
/// `obs_nll_subject_grad`'s `θ` gradient is itself a **forward** difference of
/// the predictions (`h = 1e-5·(1+|θ|)`), inherited from the SAEM M-step. So for
/// `θ` this is forward-FD versus central-FD at a different step size, and the
/// residual is the forward rule's `O(h)` truncation error — not a defect in the
/// assembly. The `σ`, `Ω` and `φ` blocks are genuinely analytic and are held to a
/// tolerance three orders tighter, which is what actually pins them.
#[test]
fn neg_elbo_gradient_matches_central_fd() {
    let (model, population, params) = fixture();
    let x = pack_params(&params);
    let cfg = cfg_seeded();
    let iter = 5;
    let layout = PackedLayout::new(&params);
    // Forward-FD truncation inside the θ block; everything else is analytic.
    let tol_for = |i: usize| {
        if i < layout.n_theta {
            2e-3
        } else {
            1e-6
        }
    };

    for (label, family) in [
        (
            "full_rank",
            Box::new(FullRank::new(model.n_eta)) as Box<dyn VariationalFamily>,
        ),
        ("mean_field", Box::new(MeanField::new(model.n_eta))),
    ] {
        let phis = perturbed_phis(family.as_ref(), &params.omega, 2);
        let e = population_neg_elbo(
            &model,
            &population,
            &params,
            &x,
            family.as_ref(),
            &phis,
            &cfg,
            iter,
        )
        .unwrap();

        for (i, &got) in e.grad_x.iter().enumerate() {
            let want = fd_of_neg_elbo_wrt_x(
                &model,
                &population,
                &params,
                &x,
                family.as_ref(),
                &phis,
                &cfg,
                iter,
                i,
            );
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() / scale < tol_for(i),
                "{label} ∂(−ELBO)/∂x[{i}]: got {got:.9e}, fd {want:.9e}"
            );
        }

        for (s, gphi) in e.grad_phi.iter().enumerate() {
            for (i, &got) in gphi.iter().enumerate() {
                let want = fd_of_neg_elbo_wrt_phi(
                    &model,
                    &population,
                    &params,
                    &x,
                    family.as_ref(),
                    &phis,
                    &cfg,
                    iter,
                    s,
                    i,
                );
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < 1e-5,
                    "{label} ∂(−ELBO)/∂φ[{s}][{i}]: got {got:.9e}, fd {want:.9e}"
                );
            }
        }
    }
}

/// The finite-difference `∂/∂η` path must reproduce the analytic one. Both feed
/// the same `φ` gradient, so if they disagreed, `vi_eta_grad` would silently
/// change the fit rather than only its speed.
#[test]
fn fd_and_analytic_eta_gradients_agree() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let x = pack_params(&params);

    let base = cfg_seeded();
    let analytic = ElboConfig {
        eta_grad: EtaGradMode::Auto,
        ..base.clone()
    };
    let fd = ElboConfig {
        eta_grad: EtaGradMode::Fd,
        ..base
    };

    let a = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        &family,
        &phis,
        &analytic,
        2,
    )
    .unwrap();
    let b = population_neg_elbo(&model, &population, &params, &x, &family, &phis, &fd, 2).unwrap();

    assert_eq!(
        a.n_fd_subjects, 0,
        "analytic provider should serve this model"
    );
    assert_eq!(b.n_fd_subjects, 2, "forced FD should report every subject");
    assert!(
        (a.neg_elbo - b.neg_elbo).abs() < 1e-12,
        "value must not depend on the gradient route"
    );

    for (s, (ga, gb)) in a.grad_phi.iter().zip(b.grad_phi.iter()).enumerate() {
        for (i, (x1, x2)) in ga.iter().zip(gb.iter()).enumerate() {
            let scale = x1.abs().max(1.0);
            assert!(
                (x1 - x2).abs() / scale < 1e-6,
                "subject {s} φ[{i}]: analytic {x1:.9e} vs fd {x2:.9e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Closed-form Ω
// ---------------------------------------------------------------------------

/// The closed-form `Ω` must zero the `Ω` block of the gradient — that is exactly
/// the claim that licenses taking it instead of stepping `Ω` with Adam.
#[test]
fn closed_form_omega_zeroes_the_omega_gradient() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);

    let mut updated = params.clone();
    updated.omega = closed_form_omega(&family, &phis, &params);
    let x = pack_params(&updated);

    let e = population_neg_elbo(
        &model,
        &population,
        &updated,
        &x,
        &family,
        &phis,
        &cfg_seeded(),
        1,
    )
    .unwrap();

    let layout = PackedLayout::new(&updated);
    for i in layout.omega_start()..layout.sigma_start() {
        assert!(
            e.grad_x[i].abs() < 1e-8,
            "∂(−ELBO)/∂x[{i}] = {:.3e} at the closed-form Ω, expected 0",
            e.grad_x[i]
        );
    }
}

/// Structural zeros and FIXed diagonals survive the closed-form update.
#[test]
fn closed_form_omega_respects_structure() {
    let names = vec!["A".into(), "B".into()];
    let omega = OmegaMatrix::from_diagonal(&[0.04, 0.09], names.clone());
    let mut template = analytical_model(GradientMethod::Auto)
        .default_params
        .clone();
    template.omega = omega.clone();
    template.omega_fixed = vec![false, true];

    let family = FullRank::new(2);
    let phis = perturbed_phis(&family, &omega, 4);
    let out = closed_form_omega(&family, &phis, &template);

    assert_eq!(
        out.matrix[(0, 1)],
        0.0,
        "a declared-diagonal Ω must not pick up sampling correlation"
    );
    assert!(
        (out.matrix[(1, 1)] - 0.09).abs() < 1e-15,
        "a FIXed diagonal must keep its declared value, got {}",
        out.matrix[(1, 1)]
    );
    assert!(
        (out.matrix[(0, 0)] - 0.04).abs() > 1e-9,
        "the free diagonal should have moved off its initial value"
    );
}

// ---------------------------------------------------------------------------
// Support routing
// ---------------------------------------------------------------------------

/// The support probe must agree with what the evaluation actually does, and an
/// out-of-scope model must fall back loudly rather than silently succeed.
///
/// # Why the probe is a probe
///
/// [`analytic_eta_grad_available`] asks the provider rather than re-deriving its
/// scope, and that scope turns out to be wide: closed-form *and* ODE models,
/// `iiv_on_ruv`, steady-state doses, reset events and `gradient = fd` models are
/// all served analytically (verified by direct probing while writing this).
/// A hand-maintained list of exclusions would have been wrong on the day it was
/// written and wrong again whenever the provider grew. The invariant worth
/// pinning is not *which* models decline but that the probe and the evaluation
/// never disagree about it.
#[test]
fn eta_grad_probe_agrees_with_evaluation_and_falls_back_loudly() {
    // In scope: a real closed-form model. Probe says yes, evaluation uses no FD.
    let (model, population, params) = fixture();
    assert!(
        analytic_eta_grad_available(
            &model,
            &population.subjects[0],
            &params.theta,
            &params.omega,
            &params.sigma.values
        ),
        "a closed-form model should be analytically differentiable"
    );
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let in_scope = population_neg_elbo(
        &model,
        &population,
        &params,
        &pack_params(&params),
        &family,
        &phis,
        &cfg_seeded(),
        1,
    )
    .unwrap();
    assert_eq!(
        in_scope.n_fd_subjects, 0,
        "probe said analytic, so the evaluation must not have used FD"
    );

    // Out of scope: the hand-built `analytical_model` shell, whose `pk_param_fn`
    // returns `PkParams::default()` and whose `pk_indices`/`eta_map` are empty —
    // there is no closed form for the provider to differentiate.
    let shell = analytical_model(GradientMethod::Auto);
    let shell_params = shell.default_params.clone();
    assert!(
        !analytic_eta_grad_available(
            &shell,
            &population.subjects[0],
            &shell_params.theta,
            &shell_params.omega,
            &shell_params.sigma.values
        ),
        "a model with no differentiable closed form must report unavailable"
    );

    let shell_family = FullRank::new(shell.n_eta);
    let shell_phis = perturbed_phis(&shell_family, &shell_params.omega, 2);
    let shell_x = pack_params(&shell_params);
    let out_of_scope = population_neg_elbo(
        &shell,
        &population,
        &shell_params,
        &shell_x,
        &shell_family,
        &shell_phis,
        &cfg_seeded(),
        1,
    )
    .unwrap();
    assert_eq!(
        out_of_scope.n_fd_subjects, 2,
        "probe said unavailable, so every subject must be reported as FD"
    );

    // `analytic` mode must refuse rather than degrade silently.
    let strict = ElboConfig {
        eta_grad: EtaGradMode::Analytic,
        ..cfg_seeded()
    };
    let err = population_neg_elbo(
        &shell,
        &population,
        &shell_params,
        &shell_x,
        &shell_family,
        &shell_phis,
        &strict,
        1,
    )
    .expect_err("vi_eta_grad = analytic must error on an out-of-scope model");
    assert!(err.contains("analytic"), "unhelpful error: {err}");
}

/// IOV is rejected outright — the variational family would have to cover
/// `(η, κ)` jointly, which v1 does not implement.
#[test]
fn iov_models_are_rejected_by_the_data_term_predicate() {
    let mut model = analytical_model(GradientMethod::Auto);
    assert!(unsupported_data_term_reason(&model).is_none());

    model.n_kappa = 1;
    let reason = unsupported_data_term_reason(&model).expect("IOV must be rejected");
    assert!(reason.contains("IOV"), "unhelpful reason: {reason}");
}
