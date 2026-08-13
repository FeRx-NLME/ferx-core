//! Tests for the ELBO value, its gradients, and the support routing.
//!
//! The central check is `neg_elbo_gradient_matches_central_fd`: with the seed and
//! iteration index fixed, `−ELBO` is a *deterministic* function of `(x, φ)`, so
//! central finite differences of it are exact rather than noisy. That is the
//! `Dual2`-vs-FD parity rule applied to the one new gradient path this module
//! introduces.

use super::*;
use crate::estimation::parameterization::pack_params;
use crate::estimation::vi::family::{FullRank, KlTerm, MeanField};
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
    (model, two_subject_population(1.0), params)
}

/// Three concentration observations after a single bolus, scaled per subject.
fn subject(id: &str, scale: f64) -> Subject {
    Subject {
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
    }
}

fn two_subject_population(scale: f64) -> Population {
    Population {
        subjects: vec![subject("1", scale), subject("2", 1.2 * scale)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// As [`fixture`], but with a **mixed** `block_omega` + standalone-eta `Ω` — the
/// shape `examples/warfarin_block_omega.ferx` uses.
///
/// `Ω` is `3 × 3` and declared non-diagonal, so `pack_params` carries the full
/// lower triangle: `(0,0) (1,0) (2,0) (1,1) (2,1) (2,2)` in column-major order.
/// The `ETA_KA` row is *not* in the `(ETA_CL, ETA_V)` block, so slots 2 and 4 —
/// `(2,0)` and `(2,1)` — are **structural zeros**: they occupy packed coordinates
/// but are not parameters. Nothing else in the VI test suite exercises them.
fn mixed_omega_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("mixed block + diagonal omega fixture parses");
    let params = model.default_params.clone();
    (model, two_subject_population(1.0), params)
}

/// Packed `Ω`-block slots of a mixed 3-eta `Ω` that are structural zeros, i.e.
/// `(2,0)` and `(2,1)` in `pack_params`' column-major lower-triangle order.
fn structural_zero_slots(template: &ModelParameters) -> Vec<usize> {
    let n_eta = template.omega.dim();
    crate::estimation::parameterization::lower_tri_iter(n_eta, template.omega.diagonal)
        .enumerate()
        .filter(|&(_, (i, j))| !template.omega.free_mask[(i, j)])
        .map(|(slot, _)| slot)
        .collect()
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
        kl: KlMode::Analytic,
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

    let a = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
        &phis,
        &cfg,
        7,
    )
    .unwrap();
    let b = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
        &phis,
        &cfg,
        7,
    )
    .unwrap();
    assert_eq!(a.neg_elbo.to_bits(), b.neg_elbo.to_bits());
    for (ga, gb) in a.grad_x.iter().zip(b.grad_x.iter()) {
        assert_eq!(ga.to_bits(), gb.to_bits());
    }

    // A different iteration draws different ε, so the MC term must move.
    let c = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
        &phis,
        &cfg,
        8,
    )
    .unwrap();
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
        Families::Uniform(&family),
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
        Families::Uniform(&family),
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
    family: Families<'_>,
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
    family: Families<'_>,
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
            Families::Uniform(family.as_ref()),
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
                Families::Uniform(family.as_ref()),
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
                    Families::Uniform(family.as_ref()),
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
        Families::Uniform(&family),
        &phis,
        &analytic,
        2,
    )
    .unwrap();
    let b = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
        &phis,
        &fd,
        2,
    )
    .unwrap();

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
    updated.omega = closed_form_omega(
        Families::Uniform(&family),
        &phis,
        &vec![0usize; phis.len()],
        &params,
    )
    .0;
    let x = pack_params(&updated);

    let e = population_neg_elbo(
        &model,
        &population,
        &updated,
        &x,
        Families::Uniform(&family),
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

/// A mixed `block_omega` + standalone-eta `Ω` must get **no gradient** in its
/// structural-zero slots, and correct gradients everywhere else.
///
/// The two halves matter equally. `∂KL/∂Ω` at a cross-block entry is emphatically
/// *not* zero — perturbing `Ω` there does change the KL, which the FD assertion
/// below pins — so the zero has to come from `chain_omega_grad` deliberately
/// skipping a non-parameter. Without that skip `vi_omega_update = adam` estimates a
/// covariance the model declared absent, silently and with no error anywhere.
#[test]
fn structural_zero_omega_slots_get_no_gradient() {
    let (model, population, params) = mixed_omega_fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let x = pack_params(&params);
    let cfg = cfg_seeded();
    let layout = PackedLayout::new(&params);
    let zeros = structural_zero_slots(&params);
    assert_eq!(
        zeros,
        vec![2, 4],
        "fixture must have (2,0) and (2,1) absent"
    );

    let e = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
        &phis,
        &cfg,
        4,
    )
    .unwrap();

    for &slot in &zeros {
        let i = layout.omega_start() + slot;
        assert_eq!(
            e.grad_x[i], 0.0,
            "structural-zero Ω slot {slot} must get no gradient"
        );
        // ... and it is a deliberate skip, not an accident of this Ω: the objective
        // really does respond to that coordinate.
        let fd = fd_of_neg_elbo_wrt_x(
            &model,
            &population,
            &params,
            &x,
            Families::Uniform(&family),
            &phis,
            &cfg,
            4,
            i,
        );
        assert!(
            fd.abs() > 1e-6,
            "test is vacuous: FD at structural-zero slot {slot} is already ~0 ({fd:.3e})"
        );
    }

    // Every *free* Ω coordinate still matches FD, so the skip is surgical.
    for (slot, (i, j)) in
        crate::estimation::parameterization::lower_tri_iter(model.n_eta, params.omega.diagonal)
            .enumerate()
    {
        if zeros.contains(&slot) {
            continue;
        }
        let idx = layout.omega_start() + slot;
        let want = fd_of_neg_elbo_wrt_x(
            &model,
            &population,
            &params,
            &x,
            Families::Uniform(&family),
            &phis,
            &cfg,
            4,
            idx,
        );
        let got = e.grad_x[idx];
        assert!(
            (got - want).abs() / want.abs().max(1.0) < 1e-6,
            "free Ω slot {slot} = ({i},{j}): got {got:.9e}, fd {want:.9e}"
        );
    }
}

/// A FIXed eta pins its whole row and column of `Ω`, not just its own variance —
/// the `fi || fj` rule [`crate::estimation::parameterization::packed_fixed_mask`]
/// uses. So an off-diagonal inside a block whose partner eta is FIXed gets no
/// gradient either.
#[test]
fn fixed_eta_zeroes_its_whole_omega_row_and_column() {
    let (model, population, mut params) = mixed_omega_fixture();
    // FIX ETA_V, the *second* member of the (ETA_CL, ETA_V) block: the (1,0)
    // off-diagonal is then fixed through its column index, which the previous
    // row-index-only test would have missed.
    params.omega_fixed = vec![false, true, false];

    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let e = population_neg_elbo(
        &model,
        &population,
        &params,
        &pack_params(&params),
        Families::Uniform(&family),
        &phis,
        &cfg_seeded(),
        6,
    )
    .unwrap();

    let layout = PackedLayout::new(&params);
    for (slot, (i, j)) in
        crate::estimation::parameterization::lower_tri_iter(model.n_eta, params.omega.diagonal)
            .enumerate()
    {
        let touches_fixed = i == 1 || j == 1;
        let got = e.grad_x[layout.omega_start() + slot];
        if touches_fixed {
            assert_eq!(
                got, 0.0,
                "Ω({i},{j}) touches FIXed ETA_V but got a gradient"
            );
        } else if params.omega.free_mask[(i, j)] {
            assert!(got != 0.0, "free Ω({i},{j}) lost its gradient");
        }
    }
}

/// The closed-form `Ω` keeps a mixed `Ω`'s structure and restores a FIXed eta's
/// whole declared row and column.
#[test]
fn closed_form_omega_respects_block_structure_and_fixed_rows() {
    let (_, _, mut params) = mixed_omega_fixture();
    params.omega_fixed = vec![false, true, false];
    let declared = params.omega.matrix.clone();

    let family = FullRank::new(3);
    let phis = perturbed_phis(&family, &params.omega, 4);
    let out = closed_form_omega(
        Families::Uniform(&family),
        &phis,
        &vec![0usize; phis.len()],
        &params,
    )
    .0;

    for i in 0..3 {
        for j in 0..3 {
            if !params.omega.free_mask[(i, j)] {
                assert_eq!(
                    out.matrix[(i, j)],
                    0.0,
                    "cross-block Ω({i},{j}) must stay a structural zero"
                );
            } else if i == 1 || j == 1 {
                assert_eq!(
                    out.matrix[(i, j)].to_bits(),
                    declared[(i, j)].to_bits(),
                    "Ω({i},{j}) touches FIXed ETA_V and must keep its declared value"
                );
            }
        }
    }
    // The free block entries did move.
    assert!(
        (out.matrix[(0, 0)] - declared[(0, 0)]).abs() > 1e-9,
        "the free ETA_CL variance should have been updated"
    );
    assert!(
        (out.matrix[(2, 2)] - declared[(2, 2)]).abs() > 1e-9,
        "the free ETA_KA variance should have been updated"
    );
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
    let out = closed_form_omega(
        Families::Uniform(&family),
        &phis,
        &vec![0usize; phis.len()],
        &template,
    )
    .0;

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
        Families::Uniform(&family),
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
        Families::Uniform(&shell_family),
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
        Families::Uniform(&shell_family),
        &shell_phis,
        &strict,
        1,
    )
    .expect_err("vi_eta_grad = analytic must error on an out-of-scope model");
    assert!(err.contains("analytic"), "unhelpful error: {err}");
}

// ---------------------------------------------------------------------------
// Monte-Carlo KL (`vi_kl = mc`)
// ---------------------------------------------------------------------------

/// `FullRank` with its closed-form KL hidden, standing in for a family that has
/// none — a Gaussian mixture, a normalizing flow. Every family shipped today *does*
/// have one, so this stub is the only way to exercise the `vi_kl = analytic`
/// fallback, which is the extension point the Monte-Carlo path exists to serve.
struct NoClosedFormKl(FullRank);

impl VariationalFamily for NoClosedFormKl {
    fn n_eta(&self) -> usize {
        self.0.n_eta()
    }
    fn n_params(&self) -> usize {
        self.0.n_params()
    }
    fn label(&self) -> &'static str {
        "no_closed_form"
    }
    fn init(&self, omega: &OmegaMatrix) -> Vec<f64> {
        self.0.init(omega)
    }
    fn sample(&self, phi: &[f64], eps: &[f64]) -> Vec<f64> {
        self.0.sample(phi, eps)
    }
    fn chain_to_phi(&self, phi: &[f64], eps: &[f64], g_eta: &[f64], out: &mut [f64]) {
        self.0.chain_to_phi(phi, eps, g_eta, out)
    }
    fn kl_to_normal(&self, _phi: &[f64], _omega: &OmegaMatrix) -> Option<KlTerm> {
        None
    }
    fn log_density(&self, phi: &[f64], eta: &[f64]) -> (f64, Vec<f64>) {
        self.0.log_density(phi, eta)
    }
    fn moments(&self, phi: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        self.0.moments(phi)
    }
}

/// All three halves of the Monte-Carlo KL — value, `∂/∂φ` and `∂/∂Ω` — must converge
/// to the closed form they replace.
///
/// Driven at the **kernel** level rather than through `population_neg_elbo`, for two
/// reasons. It isolates the new algebra from the data term entirely, so a failure
/// localizes; and pure density algebra is cheap enough to average 200k draws, which
/// a Monte-Carlo agreement test needs and a full ELBO evaluation could not afford in
/// Tier 1.
///
/// The `∂/∂φ` half is the load-bearing assertion: it is the path-derivative
/// estimator, which is unbiased but *not* the derivative of any single draw's value,
/// so convergence-in-mean to the analytic gradient is the only statement available —
/// and the right one, since the analytic gradient is itself pinned against FD by
/// `kl_d_phi_matches_fd`.
#[test]
fn mc_kl_kernel_converges_to_the_closed_form() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    let (_, _, params) = fixture();
    let d = params.omega.dim();
    let family = FullRank::new(d);
    // Off the prior, so nothing is evaluated at a stationary point where the KL and
    // its derivatives are all trivially zero.
    let phi = perturbed_phis(&family, &params.omega, 1).remove(0);
    let closed = family
        .kl_to_normal(&phi, &params.omega)
        .expect("FullRank has a closed form");

    let n = 200_000;
    let mut rng = StdRng::seed_from_u64(4242);
    let mut value = 0.0;
    let mut d_phi = vec![0.0; family.n_params()];
    let mut d_omega = DMatrix::<f64>::zeros(d, d);
    let inv_n = 1.0 / n as f64;

    for _ in 0..n {
        let eps: Vec<f64> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
        let eta = family.sample(&phi, &eps);
        let draw = mc_kl_draw(&family, &phi, &eta, &params.omega);

        value += inv_n * draw.integrand;
        d_omega += draw.d_omega * inv_n;
        // Same chaining the ELBO assembly does: the path-derivative gradient is the
        // reparameterization Jacobian applied to `∂/∂η` of the integrand.
        let scaled: Vec<f64> = draw.d_eta.iter().map(|g| g * inv_n).collect();
        family.chain_to_phi(&phi, &eps, &scaled, &mut d_phi);
    }

    // Monte-Carlo error at 200k draws; the point is agreement, not precision.
    assert!(
        (value - closed.value).abs() < 5e-3,
        "MC KL value {value:.6} vs closed form {:.6}",
        closed.value
    );
    for (i, (&got, &want)) in d_phi.iter().zip(closed.d_phi.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.02 * (1.0 + want.abs()),
            "path-derivative ∂KL/∂φ[{i}]: mc {got:.6}, closed form {want:.6}"
        );
    }
    for i in 0..d {
        for j in 0..d {
            let (got, want) = (d_omega[(i, j)], closed.d_omega[(i, j)]);
            assert!(
                (got - want).abs() < 0.02 * (1.0 + want.abs()),
                "MC ∂KL/∂Ω[{i},{j}]: {got:.6} vs closed form {want:.6}"
            );
        }
    }
}

/// The "sticking the landing" property, exactly rather than in expectation: at
/// `φ = init(Ω)` the variational posterior *is* the prior, so
/// `∇_η log q + Ω⁻¹η ≡ 0` for every draw and the KL contributes nothing to the
/// gradient.
///
/// So at that point the two routes must produce **identical** `φ` gradients — the
/// analytic KL's `d_phi` is also zero there — and identical values. This pins the
/// MC kernel's sign and scale without any Monte-Carlo tolerance, which the
/// convergence test above cannot.
#[test]
fn mc_and_analytic_agree_exactly_at_the_prior() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis: Vec<Vec<f64>> = (0..2).map(|_| family.init(&params.omega)).collect();
    let x = pack_params(&params);

    let eval = |kl| {
        population_neg_elbo(
            &model,
            &population,
            &params,
            &x,
            Families::Uniform(&family),
            &phis,
            &ElboConfig { kl, ..cfg_seeded() },
            3,
        )
        .unwrap()
    };
    let analytic = eval(KlMode::Analytic);
    let mc = eval(KlMode::Mc);

    assert!(
        analytic.kl_term.abs() < 1e-9 && mc.kl_term.abs() < 1e-9,
        "both routes should report a zero KL at the prior: analytic {}, mc {}",
        analytic.kl_term,
        mc.kl_term
    );
    for (s, (ga, gm)) in analytic.grad_phi.iter().zip(mc.grad_phi.iter()).enumerate() {
        for (i, (a, m)) in ga.iter().zip(gm.iter()).enumerate() {
            assert!(
                (a - m).abs() < 1e-9 * (1.0 + a.abs()),
                "at the prior the KL contributes nothing, so φ[{s}][{i}] must agree: \
                 analytic {a:.9e}, mc {m:.9e}"
            );
        }
    }
    // The data term is driven by the same common random numbers either way.
    assert_eq!(analytic.data_term.to_bits(), mc.data_term.to_bits());
}

/// Under `vi_kl = mc` the **`x`** gradient stays exactly FD-checkable — including the
/// `Ω` block, whose MC derivative has nothing dropped — while the **`φ`** gradient
/// deliberately does not match FD, because it is a path derivative.
///
/// Both halves are asserted. The first is the ordinary parity requirement. The second
/// guards the documentation: if someone "fixed" `mc_kl_draw` by adding the score term
/// back, `φ` would start matching FD and this test would fail, which is the point —
/// that change would silently raise the gradient variance the estimator exists to
/// avoid.
#[test]
fn mc_kl_grad_x_matches_fd_but_phi_is_path_derivative() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let x = pack_params(&params);
    let cfg = ElboConfig {
        kl: KlMode::Mc,
        ..cfg_seeded()
    };
    let iter = 11;
    let layout = PackedLayout::new(&params);

    let e = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&family),
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
            Families::Uniform(&family),
            &phis,
            &cfg,
            iter,
            i,
        );
        // The θ block inherits `obs_nll_subject_grad`'s forward-FD truncation, as in
        // `neg_elbo_gradient_matches_central_fd`; σ and Ω are analytic.
        let tol = if i < layout.n_theta { 2e-3 } else { 1e-6 };
        assert!(
            (got - want).abs() / want.abs().max(1.0) < tol,
            "mc-KL ∂(−ELBO)/∂x[{i}]: got {got:.9e}, fd {want:.9e}"
        );
    }

    // And the φ gradient is a path derivative: it must differ from FD somewhere.
    let mut max_rel_gap = 0.0f64;
    for (s, gphi) in e.grad_phi.iter().enumerate() {
        for (i, &got) in gphi.iter().enumerate() {
            let want = fd_of_neg_elbo_wrt_phi(
                &model,
                &population,
                &params,
                &x,
                Families::Uniform(&family),
                &phis,
                &cfg,
                iter,
                s,
                i,
            );
            max_rel_gap = max_rel_gap.max((got - want).abs() / want.abs().max(1.0));
        }
    }
    assert!(
        max_rel_gap > 1e-4,
        "the path-derivative estimator drops the score term, so φ must NOT match FD \
         (max relative gap {max_rel_gap:.3e}); if this fires, mc_kl_draw has started \
         computing a total derivative and the variance argument in its docs no longer holds"
    );
}

/// A family with no closed-form KL falls back to sampling instead of failing — the
/// behaviour [`VariationalFamily::kl_to_normal`]'s contract promises — and the
/// fallback is reported rather than silent.
///
/// The fallback must also be the *same* code path as the explicit option, so the stub
/// under `vi_kl = analytic` is asserted bit-identical to real `FullRank` under
/// `vi_kl = mc`: same draws, same kernel, same numbers.
#[test]
fn family_without_a_closed_form_kl_falls_back_to_sampling() {
    let (model, population, params) = fixture();
    let stub = NoClosedFormKl(FullRank::new(model.n_eta));
    let real = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&real, &params.omega, 2);
    let x = pack_params(&params);

    let fell_back = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&stub),
        &phis,
        &ElboConfig {
            kl: KlMode::Analytic,
            ..cfg_seeded()
        },
        5,
    )
    .expect("a family without a closed-form KL must fall back, not fail");
    assert_eq!(
        fell_back.n_kl_fallback_subjects, 2,
        "every subject's fallback must be reported"
    );

    let explicit = population_neg_elbo(
        &model,
        &population,
        &params,
        &x,
        Families::Uniform(&real),
        &phis,
        &ElboConfig {
            kl: KlMode::Mc,
            ..cfg_seeded()
        },
        5,
    )
    .unwrap();
    assert_eq!(
        explicit.n_kl_fallback_subjects, 0,
        "asking for mc outright is not a fallback"
    );

    assert_eq!(fell_back.neg_elbo.to_bits(), explicit.neg_elbo.to_bits());
    assert_eq!(fell_back.kl_term.to_bits(), explicit.kl_term.to_bits());
    for (a, b) in fell_back.grad_x.iter().zip(explicit.grad_x.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (ga, gb) in fell_back.grad_phi.iter().zip(explicit.grad_phi.iter()) {
        for (a, b) in ga.iter().zip(gb.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

/// With enough draws the two routes agree on the objective itself, and the data half
/// is bit-identical either way — the switch moves only the KL.
#[test]
fn mc_and_analytic_kl_agree_on_the_objective() {
    let (model, population, params) = fixture();
    let family = FullRank::new(model.n_eta);
    let phis = perturbed_phis(&family, &params.omega, 2);
    let x = pack_params(&params);

    let eval = |kl, s| {
        population_neg_elbo(
            &model,
            &population,
            &params,
            &x,
            Families::Uniform(&family),
            &phis,
            &ElboConfig {
                kl,
                n_mc_samples: s,
                ..cfg_seeded()
            },
            9,
        )
        .unwrap()
    };

    let analytic = eval(KlMode::Analytic, 4000);
    let mc = eval(KlMode::Mc, 4000);

    assert_eq!(
        analytic.data_term.to_bits(),
        mc.data_term.to_bits(),
        "the data half must not depend on the KL route"
    );
    assert!(
        (analytic.kl_term - mc.kl_term).abs() < 0.02 * (1.0 + analytic.kl_term.abs()),
        "kl_term: analytic {:.6}, mc {:.6}",
        analytic.kl_term,
        mc.kl_term
    );
    assert!(
        analytic.kl_term > 0.0,
        "a KL away from the prior is positive"
    );
}

/// IOV is no longer a data-term refusal: the variational family covers `(η, κ)` jointly
/// via the stacked prior (VI_PLAN §10).
///
/// Kept as a test rather than deleted, so that re-introducing a blanket `n_kappa > 0`
/// refusal — the easy thing to do while debugging something else — fails loudly.
#[test]
fn iov_models_are_not_refused_by_the_data_term_predicate() {
    let mut model = analytical_model(GradientMethod::Auto);
    assert!(unsupported_data_term_reason(&model).is_none());

    model.n_kappa = 1;
    assert!(
        unsupported_data_term_reason(&model).is_none(),
        "IOV must be servable: the data term routes through the _iov siblings"
    );
}

// ---------------------------------------------------------------------------
// IOV foundations (VI_PLAN §10.2a)
// ---------------------------------------------------------------------------

/// An IOV template's packed layout must account for the trailing `chol(Ω_iov)` block,
/// and must still describe `pack_params` exactly.
///
/// `pack_params` appends `Ω_iov` *after* `σ`, so a layout that stops at `σ` reports a
/// short `total()` and would leave the IOV block of the gradient silently untouched.
#[test]
fn packed_layout_covers_the_omega_iov_block() {
    let (_, _, mut template) = fixture();
    // Two kappas, diagonal: two packed coordinates after sigma.
    template.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[0.04, 0.02],
        vec!["KAPPA_CL".into(), "KAPPA_V".into()],
    ));
    template.kappa_fixed = vec![false, false];

    let layout = PackedLayout::new(&template);
    let packed = pack_params(&template);

    assert_eq!(layout.n_omega_iov, 2);
    assert_eq!(
        layout.total(),
        packed.len(),
        "layout must describe pack_params exactly"
    );
    assert_eq!(
        layout.omega_iov_start(),
        layout.sigma_start() + layout.n_sigma
    );
    assert_eq!(layout.total() - layout.omega_iov_start(), 2);

    // Without IOV the layout is unchanged and the IOV range is empty.
    let (_, _, plain) = fixture();
    let plain_layout = PackedLayout::new(&plain);
    assert_eq!(plain_layout.n_omega_iov, 0);
    assert_eq!(plain_layout.omega_iov_start(), plain_layout.total());
    assert_eq!(plain_layout.total(), pack_params(&plain).len());
}

/// `stacked_prior` must build `Σ_b = Ω ⊕ Ω_iov^{⊗K}` — block diagonal, with cross-block
/// entries marked as structural zeros.
///
/// The mask matters as much as the values: `η` and `κ` are independent by construction,
/// as are two distinct occasions, so those entries are not parameters and must never
/// acquire a covariance from the closed-form update or the gradient.
#[test]
fn stacked_prior_is_the_block_diagonal_of_omega_and_omega_iov() {
    let omega = OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
    let iov = OmegaMatrix::from_diagonal(&[0.02], vec!["KAPPA_CL".into()]);

    let k = 3;
    let sb = stacked_prior(&omega, Some(&iov), k);
    assert_eq!(sb.dim(), 2 + k, "d = n_eta + K*n_kappa");

    // Diagonal carries the two etas then one kappa variance per occasion.
    assert!((sb.matrix[(0, 0)] - 0.09).abs() < 1e-15);
    assert!((sb.matrix[(1, 1)] - 0.04).abs() < 1e-15);
    for g in 0..k {
        assert!((sb.matrix[(2 + g, 2 + g)] - 0.02).abs() < 1e-15);
    }
    // Everything off the diagonal is zero, and not a free parameter.
    for i in 0..sb.dim() {
        for j in 0..sb.dim() {
            if i != j {
                assert_eq!(sb.matrix[(i, j)], 0.0, "({i},{j}) must be zero");
                assert!(!sb.free_mask[(i, j)], "({i},{j}) must be a structural zero");
            }
        }
    }

    // log|Σ_b| = log|Ω| + K·log|Ω_iov| — the identity the closed-form KL relies on.
    let expected_log_det = omega.log_det + (k as f64) * iov.log_det;
    assert!(
        (sb.log_det - expected_log_det).abs() < 1e-12,
        "stacked log-determinant must decompose: {} vs {}",
        sb.log_det,
        expected_log_det
    );

    // No IOV, or no occasions: Ω is returned untouched, so the non-IOV path is unchanged.
    let same = stacked_prior(&omega, None, 3);
    assert_eq!(same.dim(), omega.dim());
    assert_eq!(stacked_prior(&omega, Some(&iov), 0).dim(), omega.dim());
}

/// A `block_omega` on either side must leave the stacked prior non-diagonal, so
/// `lower_tri_iter` walks the full triangle and the within-block covariance survives.
#[test]
fn stacked_prior_keeps_block_structure_and_marks_cross_block_zeros() {
    let mut m = DMatrix::zeros(2, 2);
    m[(0, 0)] = 0.09;
    m[(1, 1)] = 0.04;
    m[(0, 1)] = 0.02;
    m[(1, 0)] = 0.02;
    let mask = DMatrix::from_element(2, 2, true);
    let omega =
        OmegaMatrix::from_matrix_with_mask(m, vec!["ETA_CL".into(), "ETA_V".into()], false, mask);
    let iov = OmegaMatrix::from_diagonal(&[0.02], vec!["KAPPA_CL".into()]);

    let sb = stacked_prior(&omega, Some(&iov), 2);
    assert!(
        !sb.diagonal,
        "a block Omega must leave the stack non-diagonal"
    );
    // The declared BSV covariance survives...
    assert!((sb.matrix[(0, 1)] - 0.02).abs() < 1e-15);
    assert!(sb.free_mask[(0, 1)]);
    // ...while eta-to-kappa stays a structural zero.
    assert!(!sb.free_mask[(0, 2)]);
    assert!(
        !sb.free_mask[(2, 3)],
        "distinct occasions must not correlate"
    );
}

/// A two-subject IOV population on a closed-form 1-cpt IV model, with a differing
/// occasion count per subject.
///
/// The differing `K` is the point: it is what makes the stacked dimension a per-subject
/// quantity, so a fixture where every subject agreed would not exercise
/// [`Families::PerSubject`] or catch an offset computed from the wrong subject's `K`.
fn iov_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
",
    )
    .expect("IOV fixture parses");

    let subject = |id: &str, occ: Vec<u32>, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: (0..occ.len()).map(|k| 1.0 + 2.0 * k as f64).collect(),
        obs_raw_times: Vec::new(),
        observations: (0..occ.len())
            .map(|k| scale * (8.0 - 0.9 * k as f64))
            .collect(),
        obs_cmts: vec![1; occ.len()],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; occ.len()],
        occasions: occ,
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let population = Population {
        // Two occasions for the first subject, three for the second.
        subjects: vec![
            subject("1", vec![1, 1, 2, 2], 1.0),
            subject("2", vec![1, 2, 2, 3, 3], 1.15),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let params = model.default_params.clone();
    (model, population, params)
}

/// Per-subject families sized to each subject's stacked vector.
fn iov_families(
    model: &CompiledModel,
    population: &Population,
    full_rank: bool,
) -> Vec<Box<dyn VariationalFamily>> {
    population
        .subjects
        .iter()
        .map(|s| {
            let d = model.n_eta + subject_k_occasions(model, s) * model.n_kappa;
            if full_rank {
                Box::new(FullRank::new(d)) as Box<dyn VariationalFamily>
            } else {
                Box::new(MeanField::new(d))
            }
        })
        .collect()
}

/// `Dual2`-vs-FD parity for the **IOV** ELBO gradient (CLAUDE.md's rule applied to the
/// gradient path §10.2b introduced).
///
/// The oracle in `elbo_oracle.rs` checks the ELBO's *value* against a known answer; this
/// checks its *derivatives*. Three things here exist only on the IOV path and are not
/// covered by the non-IOV parity test: the stacked η-gradient from
/// `analytic_eta_nll_gradient_iov`, the subtraction of the **block-diagonal** prior
/// `Σ_b⁻¹z` rather than `Ω⁻¹η`, and the `Ω_iov` block of `grad_x`, which is reached by
/// summing the per-occasion blocks of `∂KL/∂Σ_b` — a sign or offset error there would
/// leave the value correct and the gradient wrong.
///
/// With the seed and iteration fixed, `−ELBO` is deterministic in `(x, φ)`, so central
/// differences are exact rather than noisy.
#[test]
fn iov_neg_elbo_gradient_matches_central_fd() {
    let (model, population, params) = iov_fixture();
    assert!(model.n_kappa > 0, "fixture must carry IOV");
    let x = pack_params(&params);
    let layout = PackedLayout::new(&params);
    assert!(
        layout.n_omega_iov > 0,
        "packed vector must carry an Omega_iov block"
    );
    let cfg = cfg_seeded();
    let iter = 5;
    // Forward-FD truncation inside the θ block; everything else is analytic.
    let tol_for = |i: usize| if i < layout.n_theta { 2e-3 } else { 1e-6 };

    for (label, full_rank) in [("full_rank", true), ("mean_field", false)] {
        let families = iov_families(&model, &population, full_rank);
        let fams = Families::PerSubject(&families);

        // Subjects differ in K, so their φ differ in length — the case the enum exists for.
        let phis: Vec<Vec<f64>> = families
            .iter()
            .enumerate()
            .map(|(s, f)| {
                let prior = stacked_prior(
                    &params.omega,
                    params.omega_iov.as_ref(),
                    subject_k_occasions(&model, &population.subjects[s]),
                );
                let mut p = f.init(&prior);
                for (i, v) in p.iter_mut().enumerate() {
                    *v += 0.15 * (((i + s * 5) * 7 + 1) as f64).sin();
                }
                p
            })
            .collect();
        assert_ne!(
            phis[0].len(),
            phis[1].len(),
            "fixture must present differing stacked dimensions"
        );

        let e =
            population_neg_elbo(&model, &population, &params, &x, fams, &phis, &cfg, iter).unwrap();
        assert_eq!(
            e.n_fd_subjects, 0,
            "{label}: IOV eta-gradient must be analytic"
        );

        for (i, &got) in e.grad_x.iter().enumerate() {
            let want =
                fd_of_neg_elbo_wrt_x(&model, &population, &params, &x, fams, &phis, &cfg, iter, i);
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
                    fams,
                    &phis,
                    &cfg,
                    iter,
                    s,
                    i,
                );
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < 1e-6,
                    "{label} ∂(−ELBO)/∂φ[{s}][{i}]: got {got:.9e}, fd {want:.9e}"
                );
            }
        }
    }
}

/// The IOV analogue of the shell [`eta_grad_probe_agrees_with_evaluation_and_falls_back_loudly`]
/// uses: [`analytical_model`] with IOV bolted on.
///
/// It is a *structural* decline — `pk_param_fn` returns `PkParams::default()` and
/// `pk_indices`/`eta_map` are empty, so there is no closed form for the provider to
/// differentiate — and that is the point. The analytic IOV scope is very wide: probing it
/// while writing this test, closed-form 1-cpt, **ODE**, `iiv_on_ruv`, and ODE +
/// `iiv_on_ruv` IOV models are *all* served analytically, despite comments in
/// `iov_analytical_supported` suggesting some are not (that predicate gates the outer
/// assembly; the inner gradient reaches ODE subjects through
/// `ode_subject_eta_grad_iov`). Picking any of those as the "out of scope" fixture would
/// have produced a test that passed for the wrong reason today and silently stopped
/// testing anything tomorrow. A model with nothing to differentiate cannot drift into
/// scope.
fn iov_shell_fixture() -> (CompiledModel, Population, ModelParameters) {
    let (_, population, _) = iov_fixture();
    let mut shell = analytical_model(GradientMethod::Auto);
    shell.n_kappa = 1;
    shell.kappa_names = vec!["KAPPA_CL".into()];
    let mut params = shell.default_params.clone();
    params.omega_iov = Some(crate::types::OmegaMatrix::from_diagonal(
        &[0.02],
        vec!["KAPPA_CL".into()],
    ));
    (shell, population, params)
}

/// VI_PLAN §10.5(4): an IOV model the analytic provider declines must fall back to
/// finite differences **loudly**, and `vi_eta_grad = analytic` must refuse it.
///
/// [`eta_grad_probe_agrees_with_evaluation_and_falls_back_loudly`] pins this for the
/// non-IOV path, but the IOV path reaches its gradient through a different function
/// (`analytic_eta_nll_gradient_iov` → `subject_eta_grad_iov`) with its own `None` branch.
/// Without this test the IOV coverage is entirely positive-direction — every other IOV
/// test asserts `n_fd_subjects == 0` — so a scope gap that silently returned a *wrong*
/// stacked gradient rather than declining would go unnoticed. That failure mode is worse
/// on the IOV path than elsewhere: a stacked gradient has the right length whether or not
/// its κ blocks mean anything, so nothing downstream would notice.
#[test]
fn iov_out_of_scope_model_falls_back_to_fd_loudly() {
    let (shell, population, params) = iov_shell_fixture();
    assert!(shell.n_kappa > 0, "fixture must carry IOV");

    let families: Vec<Box<dyn VariationalFamily>> = population
        .subjects
        .iter()
        .map(|s| {
            let d = shell.n_eta + subject_k_occasions(&shell, s) * shell.n_kappa;
            Box::new(FullRank::new(d)) as Box<dyn VariationalFamily>
        })
        .collect();
    let phis: Vec<Vec<f64>> = families
        .iter()
        .enumerate()
        .map(|(s, f)| {
            let prior = stacked_prior(
                &params.omega,
                params.omega_iov.as_ref(),
                subject_k_occasions(&shell, &population.subjects[s]),
            );
            f.init(&prior)
        })
        .collect();
    let x = pack_params(&params);

    let eval = population_neg_elbo(
        &shell,
        &population,
        &params,
        &x,
        Families::PerSubject(&families),
        &phis,
        &cfg_seeded(),
        1,
    )
    .expect("auto mode must fall back, not fail");
    assert_eq!(
        eval.n_fd_subjects,
        population.subjects.len(),
        "every subject of an out-of-scope IOV model must be counted as FD"
    );
    // The fallback has to be *usable*, not merely reported: a decline that returned NaN
    // would satisfy the count above while being useless.
    assert!(
        eval.neg_elbo.is_finite(),
        "the FD fallback must still produce a finite objective"
    );
    assert!(
        eval.grad_x.iter().all(|g| g.is_finite()),
        "the FD fallback's packed gradient must be finite"
    );

    // And `analytic` mode must refuse rather than degrade silently.
    let strict = ElboConfig {
        eta_grad: EtaGradMode::Analytic,
        ..cfg_seeded()
    };
    let err = population_neg_elbo(
        &shell,
        &population,
        &params,
        &x,
        Families::PerSubject(&families),
        &phis,
        &strict,
        1,
    )
    .expect_err("vi_eta_grad = analytic must error on an out-of-scope IOV model");
    assert!(
        err.contains("analytic"),
        "the error should name the mode that refused: {err}"
    );
}

/// The IOV gradient under **`vi_kl = mc`** — the other half of VI_PLAN §10.5(1)'s
/// "both families × both KL modes" matrix.
///
/// Analytic KL and MC KL reach `Ω_iov` by different routes, which is why covering one
/// does not cover the other. Under analytic KL the `Ω_iov` gradient comes from
/// `chain_omega_grad` contracting `∂KL/∂Σ_b` over the K κ-blocks in closed form. Under
/// MC KL there is no closed-form KL at all: the prior enters through `mc_kl_draw`
/// scoring each *sampled* stacked vector against `N(0, Σ_b)`, so the same `Ω_iov` entry
/// is reached through the sampled η/κ rather than through the family's covariance. A
/// block-offset or per-occasion summation error could be right in one path and wrong in
/// the other.
///
/// The φ carve-out from [`mc_kl_grad_x_matches_fd_but_phi_is_path_derivative`] applies
/// unchanged and for the same reason: under MC KL the φ gradient is deliberately a path
/// derivative (Roeder's "sticking the landing"), which drops the score term and is
/// therefore *not* the total derivative FD measures. Asserting φ against FD here would
/// be asserting the estimator is the thing it exists not to be. θ/σ/Ω stay checkable,
/// and the stacked dimension is what this test is actually about.
#[test]
fn iov_mc_kl_neg_elbo_gradient_matches_central_fd() {
    let (model, population, params) = iov_fixture();
    let x = pack_params(&params);
    let layout = PackedLayout::new(&params);
    assert!(
        layout.n_omega_iov > 0,
        "packed vector must carry an Omega_iov block"
    );
    let cfg = ElboConfig {
        kl: KlMode::Mc,
        ..cfg_seeded()
    };
    let iter = 7;
    // Same split as the analytic-KL IOV test: forward-FD truncation inside the θ block,
    // everything downstream analytic.
    let tol_for = |i: usize| if i < layout.n_theta { 2e-3 } else { 1e-6 };

    for (label, full_rank) in [("full_rank", true), ("mean_field", false)] {
        let families = iov_families(&model, &population, full_rank);
        let fams = Families::PerSubject(&families);

        let phis: Vec<Vec<f64>> = families
            .iter()
            .enumerate()
            .map(|(s, f)| {
                let prior = stacked_prior(
                    &params.omega,
                    params.omega_iov.as_ref(),
                    subject_k_occasions(&model, &population.subjects[s]),
                );
                let mut p = f.init(&prior);
                for (i, v) in p.iter_mut().enumerate() {
                    *v += 0.15 * (((i + s * 5) * 7 + 1) as f64).sin();
                }
                p
            })
            .collect();
        assert_ne!(
            phis[0].len(),
            phis[1].len(),
            "fixture must present differing stacked dimensions"
        );

        let e =
            population_neg_elbo(&model, &population, &params, &x, fams, &phis, &cfg, iter).unwrap();
        assert_eq!(
            e.n_fd_subjects, 0,
            "{label}: IOV eta-gradient must be analytic"
        );
        assert_eq!(
            e.n_kl_fallback_subjects, 0,
            "{label}: this family has a closed-form KL, so `mc` must be the requested \
             mode rather than a fallback — otherwise the test is not exercising MC KL"
        );

        // Guard that the Omega_iov block actually carries signal here; a silently-zero
        // block would satisfy the comparison below without testing anything.
        let iov_start = layout.omega_iov_start();
        let iov_peak = (iov_start..iov_start + layout.n_omega_iov)
            .map(|i| e.grad_x[i].abs())
            .fold(0.0f64, f64::max);
        assert!(
            iov_peak > 1e-6,
            "{label}: Omega_iov gradient block is ~zero ({iov_peak:.3e})"
        );

        for (i, &got) in e.grad_x.iter().enumerate() {
            let want =
                fd_of_neg_elbo_wrt_x(&model, &population, &params, &x, fams, &phis, &cfg, iter, i);
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() / scale < tol_for(i),
                "{label} mc-KL ∂(−ELBO)/∂x[{i}]: got {got:.9e}, fd {want:.9e}"
            );
        }

        // φ must remain a path derivative on the IOV path too — the stacked vector does
        // not change which estimator `mc_kl_draw` uses.
        let mut max_rel_gap = 0.0f64;
        for (s, gphi) in e.grad_phi.iter().enumerate() {
            for (i, &got) in gphi.iter().enumerate() {
                let want = fd_of_neg_elbo_wrt_phi(
                    &model,
                    &population,
                    &params,
                    &x,
                    fams,
                    &phis,
                    &cfg,
                    iter,
                    s,
                    i,
                );
                max_rel_gap = max_rel_gap.max((got - want).abs() / want.abs().max(1.0));
            }
        }
        assert!(
            max_rel_gap > 1e-4,
            "{label}: the path-derivative estimator drops the score term, so φ must NOT \
             match FD (max relative gap {max_rel_gap:.3e})"
        );
    }
}

// ---------------------------------------------------------------------------
// Time-varying clearance × IOV ("busulfan-shaped")
// ---------------------------------------------------------------------------

/// A busulfan-shaped model: clearance **declines with time** on top of per-occasion IOV.
///
/// Busulfan's autoinduction-like decline in CL over a multi-day course is the canonical
/// case where a time-varying structural parameter and IOV appear together, and it is a
/// genuinely different code path from either alone. `TIME` in `[individual_parameters]`
/// sets `uses_time_builtin`, which forces the per-event PK-parameter route
/// (`compute_event_pk_params_into`) — so the sensitivity walk must seed each occasion's
/// `CombinedDerivs` at *that event's* time, not once per subject. Get that wrong and
/// `KDEC` and `KAPPA_CL` trade against each other silently: both make clearance differ
/// between early and late records, so a fit still converges, to the wrong split.
///
/// `dose_occasions` / `occasions` give subject 1 three daily occasions and subject 2
/// four, preserving the differing-`K` property the stacked layout needs exercised.
fn busulfan_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta KDEC(0.02, 0.0001, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(-KDEC * TIME) * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
",
    )
    .expect("busulfan fixture parses");

    // One dose per day, two samples per day: the sampling design that makes a decline in
    // CL identifiable at all. A single trough per occasion would leave KDEC and the
    // per-occasion kappa aliased.
    let subject = |id: &str, n_days: usize, scale: f64| {
        let mut doses = Vec::new();
        let mut dose_occasions = Vec::new();
        let mut obs_times = Vec::new();
        let mut occasions = Vec::new();
        let mut observations = Vec::new();
        for d in 0..n_days {
            let t0 = 24.0 * d as f64;
            doses.push(DoseEvent::new(t0, 100.0, 1, 0.0, false, 0.0));
            dose_occasions.push(d as u32 + 1);
            for (j, dt) in [1.0f64, 6.0].iter().enumerate() {
                obs_times.push(t0 + dt);
                occasions.push(d as u32 + 1);
                // Declining exposure across days, decaying within each day.
                observations.push(scale * (8.0 - 0.35 * d as f64) * (0.75f64).powi(j as i32));
            }
        }
        let n_obs = obs_times.len();
        Subject {
            id: id.into(),
            doses,
            obs_times,
            obs_raw_times: Vec::new(),
            observations,
            obs_cmts: vec![1; n_obs],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; n_obs],
            occasions,
            obs_l2: Vec::new(),
            dose_occasions,
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    };

    let population = Population {
        subjects: vec![subject("1", 3, 1.0), subject("2", 4, 1.12)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let params = model.default_params.clone();
    (model, population, params)
}

/// The same declining clearance driven by a **time-varying covariate** rather than the
/// `TIME` built-in.
///
/// This is the other per-event route, and it was entirely uncovered: before this test
/// every subject in every VI test carried `obs_covariates: Vec::new()` and
/// `dose_covariates: Vec::new()`, so no VI test had ever evaluated a subject whose
/// covariates move. The two routes reach `pk_param_fn` through different branches of
/// `subject_needs_per_event_pk` (`uses_time` vs `has_tv_covariates`) and read different
/// snapshots, so passing on one says nothing about the other.
fn tvcov_iov_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta CLSLOPE(0.3, 0.001, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * (SCR / 1.0)^(-CLSLOPE) * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
",
    )
    .expect("tv-covariate IOV fixture parses");

    let (_, base_pop, _) = busulfan_fixture();
    let subjects: Vec<Subject> = base_pop
        .subjects
        .into_iter()
        .enumerate()
        .map(|(s, mut subj)| {
            // SCR drifts upward across the course — clearance falls as it rises, the same
            // shape the TIME-driven fixture produces by a different mechanism.
            let scr_at = |t: f64| 1.0 + 0.004 * t + 0.02 * s as f64;
            subj.obs_covariates = subj
                .obs_times
                .iter()
                .map(|&t| HashMap::from([("SCR".to_string(), scr_at(t))]))
                .collect();
            subj.dose_covariates = subj
                .doses
                .iter()
                .map(|d| HashMap::from([("SCR".to_string(), scr_at(d.time))]))
                .collect();
            subj.covariates = HashMap::from([("SCR".to_string(), scr_at(0.0))]);
            subj
        })
        .collect();

    let population = Population {
        subjects,
        covariate_names: vec!["SCR".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let params = model.default_params.clone();
    (model, population, params)
}

/// Shared parity body for the two time-varying × IOV fixtures.
///
/// Both are checked against central differences of `−ELBO` at a fixed seed and iteration,
/// which makes the objective deterministic in `(x, φ)` and the comparison exact rather
/// than noisy.
fn assert_tv_iov_gradient_matches_fd(
    what: &str,
    tv_theta: &str,
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) {
    assert!(model.n_kappa > 0, "{what}: fixture must carry IOV");
    let x = pack_params(params);
    let layout = PackedLayout::new(params);
    assert!(
        layout.n_omega_iov > 0,
        "{what}: packed vector must carry an Omega_iov block"
    );
    let cfg = cfg_seeded();
    let iter = 3;
    // Same split as the other IOV parity tests: forward-FD truncation inside the θ block
    // (inherited from `obs_nll_subject_grad`), everything downstream analytic.
    let tol_for = |i: usize| if i < layout.n_theta { 2e-3 } else { 1e-6 };

    for (label, full_rank) in [("full_rank", true), ("mean_field", false)] {
        let families = iov_families(model, population, full_rank);
        let fams = Families::PerSubject(&families);
        let phis: Vec<Vec<f64>> = families
            .iter()
            .enumerate()
            .map(|(s, f)| {
                let prior = stacked_prior(
                    &params.omega,
                    params.omega_iov.as_ref(),
                    subject_k_occasions(model, &population.subjects[s]),
                );
                let mut p = f.init(&prior);
                for (i, v) in p.iter_mut().enumerate() {
                    *v += 0.12 * (((i + s * 3) * 5 + 1) as f64).sin();
                }
                p
            })
            .collect();
        assert_ne!(
            phis[0].len(),
            phis[1].len(),
            "{what}: fixture must present differing stacked dimensions"
        );

        let e =
            population_neg_elbo(model, population, params, &x, fams, &phis, &cfg, iter).unwrap();
        assert_eq!(
            e.n_fd_subjects, 0,
            "{what}/{label}: a time-varying IOV model is inside the analytic scope; a \
             fallback here means the provider quietly stopped serving this model class"
        );

        // The guard that makes this a test *about* time-variation. If `TIME` (or the
        // moving covariate) were silently read as a constant, the parameter governing
        // the decline would simply have no gradient — and every comparison below would
        // still pass, agreeing precisely about a model that had stopped varying.
        let tv_idx = params
            .theta_names
            .iter()
            .position(|n| n == tv_theta)
            .unwrap_or_else(|| panic!("{what}: theta {tv_theta} not found"));
        assert!(
            e.grad_x[tv_idx].abs() > 1e-6,
            "{what}/{label}: the time-varying parameter {tv_theta} has no gradient \
             ({:.3e}) — the time dependence is not reaching the objective",
            e.grad_x[tv_idx]
        );
        // Likewise the kappa block: a zero Omega_iov gradient would mean the per-occasion
        // effect had collapsed into the time trend.
        let iov_start = layout.omega_iov_start();
        let iov_peak = (iov_start..iov_start + layout.n_omega_iov)
            .map(|i| e.grad_x[i].abs())
            .fold(0.0f64, f64::max);
        assert!(
            iov_peak > 1e-6,
            "{what}/{label}: Omega_iov gradient block is ~zero ({iov_peak:.3e})"
        );

        for (i, &got) in e.grad_x.iter().enumerate() {
            let want =
                fd_of_neg_elbo_wrt_x(model, population, params, &x, fams, &phis, &cfg, iter, i);
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() / scale < tol_for(i),
                "{what}/{label} ∂(−ELBO)/∂x[{i}]: got {got:.9e}, fd {want:.9e}"
            );
        }

        for (s, gphi) in e.grad_phi.iter().enumerate() {
            for (i, &got) in gphi.iter().enumerate() {
                let want = fd_of_neg_elbo_wrt_phi(
                    model, population, params, &x, fams, &phis, &cfg, iter, s, i,
                );
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < 1e-6,
                    "{what}/{label} ∂(−ELBO)/∂φ[{s}][{i}]: got {got:.9e}, fd {want:.9e}"
                );
            }
        }
    }
}

/// Time-varying clearance via the `TIME` built-in, combined with IOV.
#[test]
fn busulfan_shaped_iov_gradient_matches_central_fd() {
    let (model, population, params) = busulfan_fixture();
    assert_tv_iov_gradient_matches_fd("busulfan", "KDEC", &model, &population, &params);
}

/// Time-varying clearance via a time-varying **covariate**, combined with IOV.
#[test]
fn time_varying_covariate_iov_gradient_matches_central_fd() {
    let (model, population, params) = tvcov_iov_fixture();
    assert!(
        population
            .subjects
            .iter()
            .all(|s| !s.obs_covariates.is_empty() && !s.time_varying_covariate_names().is_empty()),
        "fixture must actually carry within-subject covariate variation"
    );
    assert_tv_iov_gradient_matches_fd("tv-covariate", "CLSLOPE", &model, &population, &params);
}
