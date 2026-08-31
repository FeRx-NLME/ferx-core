//! **The bound property**: `−2·ELBO ≥ −2 log p(y)`, checked against adaptive
//! Gauss–Hermite quadrature (`VI_VALIDATION.md` Anchor A).
//!
//! # What this establishes that nothing else does
//!
//! `elbo_oracle.rs` anchors the ELBO on a model where `q` can represent the posterior
//! *exactly*, so the bound gap is zero by construction. That is a sharp check on the
//! objective's algebra, and it is blind to everything that only happens when `q`
//! **cannot** match the posterior — which is every real model, and is where
//! `elbo_tightness_ratio` and the `Ω`-collapse behaviour live.
//!
//! This module closes that gap. On a genuinely nonlinear model the marginal has no
//! closed form, so the truth comes from adaptive GH quadrature
//! ([`crate::estimation::agq`]) instead. The anchoring chain is
//!
//! ```text
//!   NONMEM LAPLACIAN  →  AGQ  →  the ELBO bound
//! ```
//!
//! AGQ reproduces NONMEM's `LAPLACIAN` to six significant figures (see
//! [`crate::types::EstimationMethod::Laplace`]), and
//! `oracle_tests::agq_marginal_matches_the_exact_linear_gaussian_marginal` pins it to a
//! *closed form* in the same additive-constant convention the ELBO uses. So the
//! quadrature is not another estimate to be argued with — it is a transferred anchor.
//!
//! # Why a violation cannot be explained away
//!
//! Every estimate-to-estimate comparison in this space has an escape hatch: FOCEI, SAEM
//! and VI target the same parameters by different approximations, so a disagreement is
//! always arguably bias rather than a bug. An ELBO exceeding the marginal it bounds has
//! no such defence. It is a theorem, and the failures it catches are exactly the ones
//! finite-difference parity cannot see, because FD parity holds just as well for a
//! consistently wrong objective: a mis-scaled KL, a prior counted twice, a dropped
//! entropy term, a wrong Jacobian in the reparameterization.
//!
//! # What it does *not* isolate
//!
//! Both sides evaluate the same conditional likelihood — VI's data term and AGQ's
//! integrand are both `obs_nll_subject_into` underneath. So this does not re-validate
//! the likelihood; it validates the **marginalization** built on top of it. The
//! likelihood itself is anchored by the NONMEM comparisons elsewhere in the repo, which
//! is precisely why the chain above starts there.
//!
//! More importantly, **the bound is one-sided**, and that shapes what the tests here can
//! and cannot see. An objective that is too *small* violates `ELBO ≤ log p(y)` and is
//! caught; an objective that is too *large* satisfies it more comfortably and is not. So
//! the bound alone is not a complete check on the ELBO, and the module pairs it with
//! [`monte_carlo_elbo_matches_the_quadrature_elbo`], which is two-sided.
//!
//! Measured, by mutation:
//!
//! | Mutation | bound | bridge | KL |
//! |---|---|---|---|
//! | `neg_elbo = data − kl` (KL sign flipped) | **fails** | **fails** | passes |
//! | `neg_elbo = data + 0.5·kl` (KL halved) | **fails** | **fails** | passes |
//! | `neg_elbo = 0.98·data + kl` (too optimistic) | **fails** | **fails** | passes |
//! | `neg_elbo = 1.02·data + kl` (too pessimistic) | passes | **fails** | passes |
//! | `tr(Ω⁻¹S)` → diagonal only (a no-op in 1-D) | passes | passes | **fails** |
//!
//! where *bound* is [`production_elbo_never_exceeds_the_agq_marginal`], *bridge* is
//! [`monte_carlo_elbo_matches_the_quadrature_elbo`], and *KL* is
//! [`production_kl_matches_the_textbook_kl`].
//!
//! The last row is why that third test exists, and it is the most instructive one here. A
//! diagonal-only trace is exactly correct when `n_eta = 1` and silently discards posterior
//! correlation in 2-D, and **neither the bound nor the bridge could see it** — the dropped
//! term is only `0.003`–`0.012` on the `−2` scale against a `4·SEM` band of `0.047`–`0.099`.
//! Both sides of that comparison have a closed form, so comparing them through a
//! Monte-Carlo ELBO was discarding two orders of magnitude of precision for nothing. When
//! both sides have a closed form, compare the closed forms.
//!
//! Note also that [`elbo_never_exceeds_the_agq_marginal`] survives *every* row: it
//! evaluates the ELBO through this module's own quadrature and its own
//! [`kl_to_prior_1d`], so it tests the inequality as mathematics and says nothing about
//! production. [`production_elbo_never_exceeds_the_agq_marginal`] is the one that makes
//! the claim about the number `FitResult::vi::neg_two_elbo` reports. Both are kept: the
//! first localizes a failure to the theory or to AGQ, the second to the implementation.
//!
//! # Why the bound is testable without a converged fit
//!
//! `ELBO(q) ≤ log p(y)` holds for **every** `q`, not just the optimum — the gap is
//! `KL(q ‖ p(η|y)) ≥ 0`. So the strongest fast test is not one converged point but many
//! deliberately-wrong `q`s: displaced means, inflated and collapsed variances, `q` at the
//! prior. Each must satisfy the bound, and the gap must grow as `q` is moved away from
//! the posterior. A converged VI fit is checked too, under `slow-tests`, but it is the
//! weaker of the two tests because it probes one point.
//!
//! # Two fixtures
//!
//! `n_eta = 1` — lognormal `η` on `CL` — is the fixture that makes every quantity here
//! auditable by hand. `n_eta = 2` — lognormal `η` on both `CL` and `V` under a
//! `block_omega` with correlation `0.3` — is the one that actually exercises the code
//! paths a scalar fixture cannot reach: the full-rank Cholesky `φ` layout, the
//! multivariate KL, and a posterior whose *shape* (not just width) `q` has to match.
//!
//! The `q` sweep in two dimensions displaces the mean along four directions in whitened
//! posterior space — each axis, and both diagonals — because a bug in the off-diagonal of
//! `L` is invisible to a displacement along an axis. AGQ costs `n_nodes^n_eta`, so the 2-D
//! fixture uses fewer nodes per dimension; the tolerances are unchanged.
//!
//! # The fixtures are lognormal on purpose
//!
//! `CL = TVCL·exp(η)` stays positive for every `η`, so a wide quadrature grid never
//! leaves the model's domain. The additive `CL = TVCL + η` construction that makes
//! `elbo_oracle.rs` exact would drive `CL` negative in the tails — that is the documented
//! reason its own calibration test stops at five nodes, and it would put a floor on the
//! accuracy reachable here. Lognormality also makes the true posterior genuinely skewed,
//! so a Gaussian `q` *cannot* be exact and the gap under test is real rather than
//! numerical noise.
//!
//! # The error model has to be additive, and that is not a detail
//!
//! Quadrature over `q` evaluates `−log p(y|η)` at nodes several `σ_q` out, which for a
//! lognormal `η` on a rate constant means predictions ranging over many orders of
//! magnitude. Under **proportional** error the residual variance `(f·σ)²` collapses with
//! the prediction, so the `res²/(f·σ)²` term explodes as `f → 0`: measured on this
//! fixture, the data term runs `5.2` at `η = 0` to `9.5e11` at `η = +3`, flattening at
//! `8.4e13` once the variance floor clamps it. `E_q` is then dominated entirely by tail
//! nodes whose weights are `~1e−30` but whose integrand is `~1e13`, and the quadrature
//! ELBO becomes meaningless (it came out at `1e11`).
//!
//! Under **additive** error the same sweep stays bounded — `11.5` at `η = 0`, `331` at
//! the extremes — because `(y − f)²/σ²` is bounded whenever `f` is. That boundedness is
//! what makes an expectation over a full-support Gaussian `q` well behaved, so the
//! fixture uses additive error.
//!
//! This is a property of the objective, not of this test: the ELBO's data term genuinely
//! is tail-dominated for a proportional error model whose prediction can approach zero.
//! VI's Monte-Carlo estimator does not notice, because with `vi_mc_samples = 8` it never
//! draws a `4σ` tail — which is worth knowing as a variance hazard, and is a separate
//! question from the bound.

use super::*;
use crate::estimation::agq::{agq_population_nll, gauss_hermite};
use crate::estimation::inner_optimizer::find_ebe;
use crate::estimation::parameterization::pack_params;
use crate::estimation::vi::family::{FullRank, MeanField, VariationalFamily};
use crate::stats::likelihood::{individual_nll, obs_nll_subject_into};
use crate::types::{DoseEvent, HessianAnchor, ModelParameters, Population, Subject};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

const DOSE: f64 = 100.0;
/// Observation times. Sparse and late enough that the likelihood is informative about
/// `CL` without pinning it, which is what leaves the posterior visibly non-Gaussian.
const TIMES: [f64; 4] = [0.5, 2.0, 6.0, 12.0];

/// A **nonlinear** 1-cpt IV model: lognormal `η` on `CL`, additive residual error.
///
/// `ω² = 0.25` is deliberately large. The bound gap scales with how badly a Gaussian `q`
/// approximates the posterior, so a small `ω` would make this test pass on a
/// near-linear-Gaussian problem and prove much less.
fn lognormal_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.25
  sigma ADD ~ 0.5 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = 10.0

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
",
    )
    .expect("lognormal bound fixture parses")
}

/// The same model with a **correlated two-dimensional** `η`.
///
/// `corr(ETA_CL, ETA_V) = 0.06/√(0.25·0.16) = 0.3` — large enough that a `q` ignoring it
/// is measurably worse, small enough that `Ω` stays comfortably positive-definite. This is
/// the fixture that reaches the full-rank Cholesky `φ` layout and the multivariate KL.
fn two_eta_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  block_omega (ETA_CL, ETA_V) = [0.25, 0.06, 0.16]
  sigma ADD ~ 0.5 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD)
",
    )
    .expect("two-eta bound fixture parses")
}

fn bound_subject(id: &str, obs: &[f64]) -> Subject {
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
        reset_covariates: Vec::new(),
        cens: vec![0; TIMES.len()],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// Noiseless predictions at a chosen `η`, so a subject can be built whose posterior is
/// centred somewhere known-nonzero.
fn preds_at_eta(model: &CompiledModel, eta: &[f64]) -> Vec<f64> {
    let subject = bound_subject("probe", &[0.0; 4]);
    let mut scratch = crate::pk::EventPkParams::with_capacity_for(&subject);
    crate::pk::compute_predictions_with_tv_into(
        model,
        &subject,
        &model.default_params.theta,
        eta,
        &mut scratch,
    )
}

/// Three subjects whose posteriors sit at clearly different places, built by evaluating
/// the production predictor at a chosen `η` and perturbing.
///
/// Generated from the model rather than hard-coded so the fixture cannot drift out of sync
/// with the predictor it is meant to exercise.
fn bound_population(model: &CompiledModel, centres: &[(&'static str, &[f64], f64)]) -> Population {
    let subjects = centres
        .iter()
        .map(|(id, eta, bump)| {
            let obs: Vec<f64> = preds_at_eta(model, eta)
                .iter()
                .enumerate()
                .map(|(k, p)| p * (1.0 + bump * if k % 2 == 0 { 1.0 } else { -1.0 }))
                .collect();
            bound_subject(id, &obs)
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// A model + population + the node counts appropriate to its `η` dimension.
struct Fixture {
    label: &'static str,
    model: CompiledModel,
    pop: Population,
    /// GH nodes per dimension for the AGQ marginal (`agq_nodes^n_eta` evaluations).
    agq_nodes: usize,
    /// GH nodes per dimension for this module's own quadrature ELBO.
    quad_nodes: usize,
}

/// Both fixtures, in ascending `n_eta`.
///
/// Node counts fall as the dimension rises because both quadratures are tensor products.
/// They are still far above what either integrand needs: the 1-D calibration in
/// `elbo_oracle.rs` reaches machine precision at three.
fn fixtures() -> Vec<Fixture> {
    let one = lognormal_model();
    let one_pop = bound_population(
        &one,
        &[
            ("1", &[0.45], 0.10),
            ("2", &[-0.55], 0.06),
            ("3", &[0.05], 0.18),
        ],
    );
    let two = two_eta_model();
    let two_pop = bound_population(
        &two,
        &[
            ("1", &[0.40, 0.25], 0.10),
            ("2", &[-0.50, 0.30], 0.06),
            ("3", &[0.10, -0.35], 0.15),
        ],
    );
    vec![
        Fixture {
            label: "1 eta: lognormal CL",
            model: one,
            pop: one_pop,
            agq_nodes: 21,
            quad_nodes: 41,
        },
        Fixture {
            label: "2 eta: lognormal CL+V, block omega",
            model: two,
            pop: two_pop,
            agq_nodes: 15,
            quad_nodes: 21,
        },
    ]
}

/// A one-subject `Population`, so population-level entry points yield per-subject values.
///
/// The bound is a *per-subject* statement — `−2·ELBOᵢ ≥ −2 log p(yᵢ)` for every `i` — and a
/// population total can hide a violation on one subject behind slack on another. Both
/// `agq_population_nll` and [`population_neg_elbo`] fold over subjects, so slicing the
/// population is how to read one term out of either without widening any production API.
fn singleton(pop: &Population, i: usize) -> Population {
    Population {
        subjects: vec![pop.subjects[i].clone()],
        covariate_names: pop.covariate_names.clone(),
        dv_column: pop.dv_column.clone(),
        input_columns: pop.input_columns.clone(),
        exclusions: None,
        warnings: vec![],
    }
}

/// `−2 log p(yᵢ)` by adaptive GH quadrature, with the grid laid at the subject's EBE.
fn agq_neg_two_log_marginal(
    model: &CompiledModel,
    pop: &Population,
    i: usize,
    params: &ModelParameters,
    n_nodes: usize,
) -> f64 {
    let one = singleton(pop, i);
    let ebe = find_ebe(model, &one.subjects[0], params, 200, 1e-10, None, None, 0);
    2.0 * agq_population_nll(
        model,
        &one,
        params,
        &[ebe.eta],
        &[Vec::new()],
        n_nodes,
        HessianAnchor::Exact,
    )
}

/// `KL(N(μ, S) ‖ N(0, Ω))` from the textbook formula, written out here rather than taken
/// from [`family::VariationalFamily::kl_to_normal`].
///
/// The point of the duplication is that a wrong KL in the production family is exactly one
/// of the bugs this module exists to catch, so the comparison must not route through it.
fn kl_to_prior(mu: &DVector<f64>, s: &DMatrix<f64>, omega: &DMatrix<f64>) -> f64 {
    let d = mu.len() as f64;
    let omega_inv = omega
        .clone()
        .try_inverse()
        .expect("prior Omega is invertible");
    let trace = (&omega_inv * s).trace();
    let quad = (mu.transpose() * &omega_inv * mu)[(0, 0)];
    let log_det_ratio = omega.clone().determinant().ln() - s.clone().determinant().ln();
    0.5 * (trace + quad - d + log_det_ratio)
}

/// The Laplace posterior `(mode, covariance)` for one subject, by finite-differencing the
/// **joint** objective at the mode.
///
/// [`individual_nll`] is `−log p(y, η)` up to a constant, so its Hessian at the mode is the
/// posterior precision — prior curvature included. Deliberately *not* taken from
/// `EbeResult::h_matrix`, which is FOCE's `∂f/∂η` Gauss-Newton artefact rather than the
/// conditional Hessian; using it as a variance silently mis-scaled every `q` in this module
/// by an order of magnitude, which is the sort of thing a bound test scaled in absolute
/// units would have hidden.
fn laplace_posterior(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
) -> (DVector<f64>, DMatrix<f64>) {
    let ebe = find_ebe(model, subject, params, 200, 1e-10, None, None, 0);
    let mode = ebe.eta.clone();
    let d = mode.len();
    let nll = |e: &DVector<f64>| {
        individual_nll(
            model,
            subject,
            &params.theta,
            e.as_slice(),
            &params.omega,
            &params.sigma.values,
        )
    };
    let h = 1e-4;
    let f0 = nll(&mode);
    let mut hess = DMatrix::zeros(d, d);
    let shifted = |signs: &[(usize, f64)]| {
        let mut e = mode.clone();
        for (k, s) in signs {
            e[*k] += s * h;
        }
        nll(&e)
    };
    for i in 0..d {
        for j in i..d {
            let v = if i == j {
                (shifted(&[(i, 1.0)]) - 2.0 * f0 + shifted(&[(i, -1.0)])) / (h * h)
            } else {
                (shifted(&[(i, 1.0), (j, 1.0)])
                    - shifted(&[(i, 1.0), (j, -1.0)])
                    - shifted(&[(i, -1.0), (j, 1.0)])
                    + shifted(&[(i, -1.0), (j, -1.0)]))
                    / (4.0 * h * h)
            };
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
    let cov = hess
        .clone()
        .try_inverse()
        .expect("conditional Hessian at the mode is invertible");
    for i in 0..d {
        assert!(
            cov[(i, i)] > 0.0 && cov[(i, i)].is_finite(),
            "subject {}: Laplace variance {} on coordinate {i} is not positive, so the mode \
             is not a minimum and nothing downstream is meaningful",
            subject.id,
            cov[(i, i)]
        );
    }
    (mode, cov)
}

/// The tensor-product Gauss–Hermite grid over `d` dimensions: `(z, Πw)` per node.
fn gh_grid(n_nodes: usize, d: usize) -> Vec<(Vec<f64>, f64)> {
    let (nodes, weights) = gauss_hermite(n_nodes);
    let mut grid = vec![(Vec::new(), 1.0f64)];
    for _ in 0..d {
        let mut next = Vec::with_capacity(grid.len() * n_nodes);
        for (z, w) in &grid {
            for (zi, wi) in nodes.iter().zip(weights.iter()) {
                let mut zz = z.clone();
                zz.push(*zi);
                next.push((zz, w * wi));
            }
        }
        grid = next;
    }
    grid
}

/// `E_q[−log p(yᵢ | η)]` by Gauss–Hermite quadrature over `q = N(μ, S)`.
///
/// `∫ f(η) N(η; μ, S) dη = π^{−d/2} Σⱼ wⱼ f(μ + √2·L·zⱼ)` with `S = LLᵀ`, in the physicists'
/// convention [`gauss_hermite`] returns. Deterministic — which is the whole reason to have
/// it alongside VI's Monte-Carlo data term: the bound test needs both sides free of sampling
/// noise, or an ELBO that overshoots by MC error looks like a violated bound.
fn data_term_quad(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    mu: &DVector<f64>,
    s: &DMatrix<f64>,
    n_nodes: usize,
) -> f64 {
    let d = mu.len();
    let l = s
        .clone()
        .cholesky()
        .expect("q covariance is positive-definite")
        .l();
    let mut scratch = crate::pk::EventPkParams::with_capacity_for(subject);
    let mut acc = 0.0;
    for (z, w) in gh_grid(n_nodes, d) {
        let eta = mu + std::f64::consts::SQRT_2 * (&l * DVector::from_vec(z));
        acc += w * obs_nll_subject_into(
            model,
            subject,
            &params.theta,
            &params.sigma.values,
            eta.as_slice(),
            &mut scratch,
        );
    }
    acc / std::f64::consts::PI.powf(d as f64 / 2.0)
}

/// `−2·ELBO` for one subject, entirely by quadrature: no Monte Carlo on either half.
fn neg_two_elbo_quad(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    mu: &DVector<f64>,
    s: &DMatrix<f64>,
    n_nodes: usize,
) -> f64 {
    2.0 * (data_term_quad(model, subject, params, mu, s, n_nodes)
        + kl_to_prior(mu, s, &params.omega.matrix))
}

/// [`FullRank`]'s `φ` for a given `(μ, S)`: `[μ | vech(L)]` row-major over the lower
/// triangle, with the diagonal stored as its logarithm.
fn full_rank_phi(mu: &DVector<f64>, s: &DMatrix<f64>) -> Vec<f64> {
    let d = mu.len();
    let l = s
        .clone()
        .cholesky()
        .expect("q covariance is positive-definite")
        .l();
    let mut phi: Vec<f64> = mu.iter().copied().collect();
    for i in 0..d {
        for j in 0..=i {
            phi.push(if i == j { l[(i, i)].ln() } else { l[(i, j)] });
        }
    }
    phi
}

/// Seeds per `(subject, q)` cell for the Monte-Carlo tests. Small on purpose — the bands
/// below are multiples of the measured SEM, so a loose SEM estimate costs breadth rather
/// than correctness.
const N_SEEDS: usize = 3;
const N_MC: usize = 600;

/// `(mean, SEM)` of production's `−2·ELBO` at one `(subject, q)`, over [`N_SEEDS`] seeds.
///
/// Replicating over seeds rather than trusting one is what lets the assertions below
/// separate *bias* from *sampling noise*. `population_neg_elbo` derives its draws from
/// `(seed, iteration, subject, sample)`, so changing the seed is the only way to get an
/// independent replicate of the same quantity.
fn mc_neg_two_elbo_stats(
    model: &CompiledModel,
    one: &Population,
    params: &ModelParameters,
    x: &[f64],
    family: &FullRank,
    phi: &[f64],
) -> (f64, f64) {
    let phis = vec![phi.to_vec()];
    let draws: Vec<f64> = (0..N_SEEDS)
        .map(|r| {
            let cfg = ElboConfig {
                n_mc_samples: N_MC,
                eta_grad: EtaGradMode::Auto,
                kl: KlMode::Analytic,
                seed: 20260819 + r as u64 * 7_919,
            };
            let eval = population_neg_elbo(
                model,
                one,
                params,
                x,
                Families::Uniform(family),
                &phis,
                &cfg,
                0,
            )
            .expect("bound fixtures are inside VI's support scope");
            2.0 * eval.neg_elbo
        })
        .collect();

    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

/// Directions in whitened posterior space along which `q`'s mean is displaced.
///
/// One direction is enough in 1-D. In 2-D the diagonals are the point: a bug in the
/// off-diagonal of `L`, or a KL that drops the cross term, leaves an axis-aligned
/// displacement looking perfectly healthy.
fn sweep_directions(d: usize) -> Vec<(&'static str, DVector<f64>)> {
    match d {
        1 => vec![("", DVector::from_vec(vec![1.0]))],
        2 => {
            let r = 1.0 / 2.0f64.sqrt();
            vec![
                (" along CL", DVector::from_vec(vec![1.0, 0.0])),
                (" along V", DVector::from_vec(vec![0.0, 1.0])),
                (" along (+,+)", DVector::from_vec(vec![r, r])),
                (" along (+,-)", DVector::from_vec(vec![r, -r])),
            ]
        }
        _ => unreachable!("fixtures here are 1- or 2-dimensional"),
    }
}

/// The `q`s every bound assertion is swept over: `(label, mean shift in posterior SDs,
/// variance multiplier)`.
///
/// Shifts are in units of the **posterior** scale — `μ = mode + shift · L_post · u` for a
/// unit direction `u` — so the sweep means the same thing whether a subject's data is
/// sparse or rich, and in either dimension. The displaced and mis-scaled entries are the
/// ones with teeth: a bug that inverts the KL's sign or mis-scales the entropy can still
/// satisfy the bound at a near-optimal `q`, where the gap is small and both halves sit near
/// their stationary values, and fail once `q` is pushed away.
const Q_SWEEP: [(&str, f64, f64); 7] = [
    ("the Laplace q", 0.0, 1.0),
    ("mean +1.5 SD", 1.5, 1.0),
    ("mean -1.5 SD", -1.5, 1.0),
    ("variance inflated 4x", 0.0, 4.0),
    ("variance collapsed 16x", 0.0, 1.0 / 16.0),
    ("mean +1 SD, inflated 3x", 1.0, 3.0),
    ("mean -1 SD, collapsed 10x", -1.0, 0.1),
];

/// `(μ, S)` for one sweep entry: displace the mode along `u` in whitened space, then scale.
fn sweep_q(
    mode: &DVector<f64>,
    lap_cov: &DMatrix<f64>,
    u: &DVector<f64>,
    shift: f64,
    s_mult: f64,
) -> (DVector<f64>, DMatrix<f64>) {
    let l = lap_cov
        .clone()
        .cholesky()
        .expect("Laplace covariance is positive-definite")
        .l();
    (mode + shift * (&l * u), lap_cov * s_mult)
}

/// **The bound property.** `−2·ELBO ≥ −2 log p(y)` for every fixture, subject and `q`.
///
/// Both sides are deterministic: AGQ on the right, quadrature-ELBO on the left. The margin
/// is `-1e-7` rather than `0.0` because the two integrate the same function on different
/// grids, so a *zero* gap — which the collapsed-variance entries approach, since a narrow
/// `q` at the mode is close to the Laplace approximation AGQ is anchored at — can land
/// either side of exact equality by roundoff. Anything beyond that is a real violation.
#[test]
fn elbo_never_exceeds_the_agq_marginal() {
    for fx in fixtures() {
        let params = fx.model.default_params.clone();
        for i in 0..fx.pop.subjects.len() {
            let subject = &fx.pop.subjects[i];
            let marginal = agq_neg_two_log_marginal(&fx.model, &fx.pop, i, &params, fx.agq_nodes);
            let (mode, lap_cov) = laplace_posterior(&fx.model, subject, &params);

            for (dir_label, u) in sweep_directions(mode.len()) {
                for (label, shift, s_mult) in Q_SWEEP {
                    let (mu, s) = sweep_q(&mode, &lap_cov, &u, shift, s_mult);
                    let neg_two_elbo =
                        neg_two_elbo_quad(&fx.model, subject, &params, &mu, &s, fx.quad_nodes);
                    assert!(
                        neg_two_elbo >= marginal - 1e-7,
                        "[{}] subject {} q={label}{dir_label}: -2*ELBO = {neg_two_elbo:.10} \
                         fell BELOW -2 log p(y) = {marginal:.10} (by {:.3e}). The ELBO is a \
                         lower bound on log p(y); -2*ELBO cannot go under -2 log p(y) for \
                         any q.",
                        fx.label,
                        subject.id,
                        marginal - neg_two_elbo
                    );
                }
            }
        }
    }
}

/// **The bound property, on the number VI itself reports.**
///
/// [`elbo_never_exceeds_the_agq_marginal`] establishes the inequality for the ELBO *as a
/// formula*, evaluated by this module's own quadrature. That is the sharper numerical
/// statement, but it routes around production: it builds its own KL ([`kl_to_prior`]) and
/// its own expectation, so a bug inside [`population_neg_elbo`] does not show up there.
/// Verified by mutation — flipping the sign of `neg_elbo = data_term + kl_term` leaves that
/// test green.
///
/// This test closes the loop by asserting the bound directly on [`population_neg_elbo`]'s
/// output, which is what `FitResult::vi::neg_two_elbo` reports and what Adam descends.
///
/// The comparison is one-sided and statistical, in that order for a reason. The MC data
/// term is *unbiased*, so a single noisy draw can land below the true ELBO and manufacture
/// a violation that is not one. Averaging over seeds and allowing `4·SEM` of slack on the
/// side where noise could fake a failure keeps the test sensitive to a real bias while
/// immune to the noise — the same reasoning as `mc_kl` versus the closed-form KL elsewhere
/// in this file.
///
/// One sweep direction only, unlike the deterministic test above: each cell here costs
/// `N_SEEDS · N_MC` predictor evaluations, and directions are what the cheap test covers.
#[test]
fn production_elbo_never_exceeds_the_agq_marginal() {
    for fx in fixtures() {
        let params = fx.model.default_params.clone();
        let family = FullRank::new(params.omega.dim());
        let x = pack_params(&params);

        for i in 0..fx.pop.subjects.len() {
            let one = singleton(&fx.pop, i);
            let subject = &fx.pop.subjects[i];
            let marginal = agq_neg_two_log_marginal(&fx.model, &fx.pop, i, &params, fx.agq_nodes);
            let (mode, lap_cov) = laplace_posterior(&fx.model, subject, &params);
            let (dir_label, u) = sweep_directions(mode.len()).remove(0);

            for (label, shift, s_mult) in Q_SWEEP {
                let (mu, s) = sweep_q(&mode, &lap_cov, &u, shift, s_mult);
                let phi = full_rank_phi(&mu, &s);
                let (mean, sem) =
                    mc_neg_two_elbo_stats(&fx.model, &one, &params, &x, &family, &phi);

                assert!(
                    mean >= marginal - 4.0 * sem - 1e-9,
                    "[{}] subject {} q={label}{dir_label}: VI's reported -2*ELBO averaged \
                     {mean:.8} over {N_SEEDS} seeds (SEM {sem:.3e}) but -2 log p(y) = \
                     {marginal:.8}. The shortfall {:.3e} exceeds the 4*SEM allowance, so \
                     this is bias, not sampling noise: the objective VI optimizes is not a \
                     lower bound on log p(y).",
                    fx.label,
                    subject.id,
                    marginal - mean
                );
            }
        }
    }
}

/// The bound is **not vacuous**: on these fixtures the gap at a well-chosen `q` is small
/// but strictly positive, and moving `q` away from the posterior makes it grow.
///
/// Without this, [`elbo_never_exceeds_the_agq_marginal`] would also pass for an
/// implementation whose ELBO was uselessly loose, or for a fixture so near-Gaussian that
/// the inequality holds trivially. The monotonicity is the part that says the *size* of the
/// gap tracks `KL(q ‖ p(η|y))` rather than merely being non-negative.
#[test]
fn the_bound_gap_grows_as_q_leaves_the_posterior() {
    for fx in fixtures() {
        let params = fx.model.default_params.clone();
        for i in 0..fx.pop.subjects.len() {
            let subject = &fx.pop.subjects[i];
            let marginal = agq_neg_two_log_marginal(&fx.model, &fx.pop, i, &params, fx.agq_nodes);
            let (mode, lap_cov) = laplace_posterior(&fx.model, subject, &params);

            for (dir_label, u) in sweep_directions(mode.len()) {
                // The gap is KL(q || p(eta|y)) in nats: half the difference of the -2 scale.
                let gap = |shift: f64| {
                    let (mu, s) = sweep_q(&mode, &lap_cov, &u, shift, 1.0);
                    0.5 * (neg_two_elbo_quad(&fx.model, subject, &params, &mu, &s, fx.quad_nodes)
                        - marginal)
                };

                // At the Laplace q the gap is a genuine KL: positive, because the true
                // posterior is skewed by the lognormal eta, but small.
                let g0 = gap(0.0);
                assert!(
                    g0 > 1e-6,
                    "[{}] subject {}: gap at the Laplace q is {g0:.3e}, indistinguishable \
                     from zero — this fixture is not exercising a nonlinear posterior at all",
                    fx.label,
                    subject.id
                );
                assert!(
                    g0 < 0.5,
                    "[{}] subject {}: gap at the Laplace q is {g0:.4}, far larger than \
                     expected for a near-optimal q; either the ELBO or the marginal is wrong",
                    fx.label,
                    subject.id
                );

                // KL(q||p) is minimized near the posterior, so displacing the mean must cost.
                for d in [0.5f64, 1.0, 2.0] {
                    assert!(
                        gap(d) > g0 && gap(-d) > g0,
                        "[{}] subject {}: displacing q's mean by +/-{d} posterior SDs\
                         {dir_label} did not increase the bound gap (g(0)={g0:.6}, \
                         g(+{d})={:.6}, g(-{d})={:.6}); the gap must behave like a \
                         divergence from the posterior",
                        fx.label,
                        subject.id,
                        gap(d),
                        gap(-d)
                    );
                }
            }
        }
    }
}

/// VI's **Monte-Carlo** ELBO agrees with the quadrature ELBO at the same `φ`.
///
/// This is what licenses the two bound tests to use the deterministic quadrature ELBO as a
/// stand-in for the objective VI actually optimizes. It is also an integration-scheme
/// cross-check in its own right: `population_neg_elbo` estimates the data term by sampling
/// `q`, this module integrates it on a GH grid, and the two share no code beyond the
/// integrand itself. Unlike the bound, it is **two-sided** — which is what catches an
/// objective that is too *large*, a direction a lower bound cannot see.
///
/// # The tolerance is measured, not chosen
///
/// The obvious form — one seed and a relative tolerance — does not work here, because the
/// ELBO omits the `½·n_obs·log 2π` constant (see this module's parent) and so can land
/// arbitrarily close to zero, at which point a *relative* tolerance means nothing. The
/// natural scale for the disagreement is instead the Monte-Carlo standard error, so the
/// test estimates it: several seeds per `q`, and the quadrature value must sit inside a
/// `4·SEM` band of their mean. That adapts to whichever `q` is under test — a wide `q`
/// samples a more variable integrand and earns a wider band automatically — and it fails on
/// bias rather than on noise, which is the distinction that matters.
///
/// Three representative `q`s rather than the whole [`Q_SWEEP`]: the point here is that two
/// integration schemes agree on the same objective, which does not need the full sweep. The
/// bound itself is checked over all of `Q_SWEEP` above, where both sides are deterministic
/// and the evaluation is cheap.
#[test]
fn monte_carlo_elbo_matches_the_quadrature_elbo() {
    for fx in fixtures() {
        let params = fx.model.default_params.clone();
        let family = FullRank::new(params.omega.dim());
        let x = pack_params(&params);

        for i in 0..fx.pop.subjects.len() {
            let one = singleton(&fx.pop, i);
            let subject = &fx.pop.subjects[i];
            let (mode, lap_cov) = laplace_posterior(&fx.model, subject, &params);
            let (dir_label, u) = sweep_directions(mode.len()).remove(0);

            for (label, shift, s_mult) in [
                ("the Laplace q", 0.0, 1.0),
                ("mean +1.5 SD", 1.5, 1.0),
                ("variance inflated 4x", 0.0, 4.0),
            ] {
                let (mu, s) = sweep_q(&mode, &lap_cov, &u, shift, s_mult);
                let phi = full_rank_phi(&mu, &s);
                let (mean, sem) =
                    mc_neg_two_elbo_stats(&fx.model, &one, &params, &x, &family, &phi);
                let quad = neg_two_elbo_quad(&fx.model, subject, &params, &mu, &s, fx.quad_nodes);
                let band = 4.0 * sem + 1e-6;

                assert!(
                    (mean - quad).abs() < band,
                    "[{}] subject {} q={label}{dir_label}: MC -2*ELBO averaged {mean:.8} \
                     over {N_SEEDS} seeds (SEM {sem:.3e}) against quadrature {quad:.8}; the \
                     gap {:.3e} exceeds the 4*SEM band {band:.3e}, so the two integration \
                     schemes disagree by more than sampling noise",
                    fx.label,
                    subject.id,
                    (mean - quad).abs()
                );
            }
        }
    }
}

/// Self-check on this module's own quadrature transform, before the tests above lean on it.
///
/// Three independent things are asserted, all against arithmetic rather than against
/// production code: that the GH change of variables reproduces the mean and *covariance* of
/// `q` — including its off-diagonal, which is what the 2-D fixture depends on — and that
/// `E_q[log q − log p(η|Ω)]` by quadrature equals the closed-form Gaussian KL. A mistake in
/// the `√2`, the `π^{−d/2}`, or the Cholesky orientation would break the moments; a mistake
/// in [`kl_to_prior`] would break the KL.
#[test]
fn the_quadrature_transform_reproduces_gaussian_moments_and_kl() {
    let cases: Vec<(DVector<f64>, DMatrix<f64>, DMatrix<f64>)> = vec![
        (
            DVector::from_vec(vec![0.0]),
            DMatrix::from_vec(1, 1, vec![1.0]),
            DMatrix::from_vec(1, 1, vec![0.25]),
        ),
        (
            DVector::from_vec(vec![0.4]),
            DMatrix::from_vec(1, 1, vec![0.25]),
            DMatrix::from_vec(1, 1, vec![0.09]),
        ),
        (
            DVector::from_vec(vec![0.3, -0.6]),
            DMatrix::from_row_slice(2, 2, &[0.20, 0.05, 0.05, 0.12]),
            DMatrix::from_row_slice(2, 2, &[0.25, 0.06, 0.06, 0.16]),
        ),
        (
            DVector::from_vec(vec![-0.2, 0.5]),
            DMatrix::from_row_slice(2, 2, &[0.05, -0.02, -0.02, 0.30]),
            DMatrix::from_row_slice(2, 2, &[0.25, 0.06, 0.06, 0.16]),
        ),
    ];

    for (mu, s, omega) in cases {
        let d = mu.len();
        let l = s.clone().cholesky().expect("S is PD").l();
        let grid = gh_grid(31, d);
        let norm = std::f64::consts::PI.powf(d as f64 / 2.0);
        let expect = |f: &dyn Fn(&DVector<f64>) -> f64| -> f64 {
            grid.iter()
                .map(|(z, w)| {
                    let eta = &mu + std::f64::consts::SQRT_2 * (&l * DVector::from_vec(z.clone()));
                    w * f(&eta)
                })
                .sum::<f64>()
                / norm
        };

        for k in 0..d {
            let m1 = expect(&|e| e[k]);
            assert!(
                (m1 - mu[k]).abs() < 1e-12,
                "E_q[eta_{k}] = {m1:.14}, expected {:.14}",
                mu[k]
            );
            for j in 0..d {
                let m2 = expect(&|e| e[k] * e[j]);
                let want = mu[k] * mu[j] + s[(k, j)];
                assert!(
                    (m2 - want).abs() < 1e-12,
                    "E_q[eta_{k}*eta_{j}] = {m2:.14}, expected {want:.14}"
                );
            }
        }

        // E_q[log q - log p(eta|Omega)], the KL by definition.
        let log_gauss = |e: &DVector<f64>, m: &DVector<f64>, c: &DMatrix<f64>| -> f64 {
            let ci = c.clone().try_inverse().expect("covariance is invertible");
            let dv = e - m;
            -0.5 * ((2.0 * std::f64::consts::PI).ln() * d as f64
                + c.clone().determinant().ln()
                + (dv.transpose() * ci * &dv)[(0, 0)])
        };
        let zero = DVector::zeros(d);
        let by_quad = expect(&|e| log_gauss(e, &mu, &s) - log_gauss(e, &zero, &omega));
        let closed = kl_to_prior(&mu, &s, &omega);
        assert!(
            (by_quad - closed).abs() < 1e-12,
            "KL by quadrature {by_quad:.14} vs closed form {closed:.14}"
        );
    }
}

/// The bound at the point a **real VI fit** actually converges to.
///
/// The tests above are stronger as tests — they sweep many `q`s — but they all evaluate the
/// ELBO through this module's quadrature, and they choose `φ` themselves. This one runs
/// `fit()` end to end and checks the bound on the `(θ, Ω, σ)` and `q` that `run_vi`'s Adam
/// loop, projection, Polyak averaging and closed-form `Ω` update jointly produced. It is
/// the only test here that can catch a bug living in that assembly rather than in the
/// objective.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bound_holds_at_the_converged_vi_point() {
    for fx in fixtures() {
        let opts = crate::types::FitOptions {
            method: crate::types::EstimationMethod::Vi,
            vi_iters: 4000,
            vi_mc_samples: 32,
            run_covariance_step: false,
            ..Default::default()
        };

        let fit = crate::api::fit(&fx.model, &fx.pop, &fx.model.default_params, &opts)
            .unwrap_or_else(|e| panic!("[{}] VI fit failed: {e}", fx.label));
        let vi = fit
            .vi
            .as_ref()
            .expect("method = vi populates FitResult::vi");

        // Rebuild the parameters VI converged to, so the marginal is evaluated at the same
        // point the ELBO was. `FitResult` reports theta/omega/sigma separately rather than
        // as a `ModelParameters`, so the reassembly is explicit.
        let mut params = fx.model.default_params.clone();
        params.theta = fit.theta.clone();
        params.sigma.values = fit.sigma.clone();
        params.omega = crate::types::OmegaMatrix::from_matrix(
            fit.omega.clone(),
            fit.eta_names.clone(),
            fx.model.default_params.omega.diagonal,
        );

        let mut total_elbo = 0.0;
        let mut total_marginal = 0.0;
        for i in 0..fx.pop.subjects.len() {
            let subject = &fx.pop.subjects[i];
            let mu = DVector::from_vec(vi.eta_means[i].clone());
            let d = mu.len();
            let s = DMatrix::from_fn(d, d, |r, c| vi.eta_covs[i][r][c]);
            assert!(
                s.clone().cholesky().is_some(),
                "[{}] subject {}: VI reported a non-PD posterior covariance {s}",
                fx.label,
                subject.id
            );

            let neg_two_elbo =
                neg_two_elbo_quad(&fx.model, subject, &params, &mu, &s, fx.quad_nodes);
            let marginal = agq_neg_two_log_marginal(&fx.model, &fx.pop, i, &params, fx.agq_nodes);
            total_elbo += neg_two_elbo;
            total_marginal += marginal;

            assert!(
                neg_two_elbo >= marginal - 1e-7,
                "[{}] subject {}: at the converged VI point, -2*ELBO = {neg_two_elbo:.8} \
                 fell below -2 log p(y) = {marginal:.8}",
                fx.label,
                subject.id
            );
        }

        // And the converged q should be a *good* q: the population gap is the sum of the
        // per-subject KL(q||p(eta|y)), which a working fit keeps to a fraction of a nat each.
        let gap_per_subject = 0.5 * (total_elbo - total_marginal) / fx.pop.subjects.len() as f64;
        assert!(
            gap_per_subject < 0.25,
            "[{}] mean per-subject KL(q || p(eta|y)) at convergence is {gap_per_subject:.4} \
             nats; the bound holds but q is a poor approximation, which is what \
             elbo_tightness_ratio is meant to surface",
            fx.label
        );
    }
}

/// One `production_kl_matches_the_textbook_kl` case: `(label, Ω, μ, S)`.
type KlCase = (&'static str, DMatrix<f64>, DVector<f64>, DMatrix<f64>);

/// Production's **closed-form KL** against the textbook formula, deterministically.
///
/// # Why this exists separately from the bound
///
/// Both `q` and the prior are Gaussian, so `KL(q ‖ N(0, Ω))` is available in closed form on
/// *both* sides — production's [`family::VariationalFamily::kl_to_normal`] and this module's
/// [`kl_to_prior`]. Comparing them through a Monte-Carlo ELBO, as
/// [`monte_carlo_elbo_matches_the_quadrature_elbo`] does, throws away roughly two orders of
/// magnitude of precision for nothing.
///
/// That is not hypothetical. Mutating `FullRank::kl_to_normal`'s `tr(Ω⁻¹S)` to a
/// **diagonal-only** trace — an exact no-op when `n_eta = 1`, and a real bug in 2-D that
/// silently ignores posterior correlation — passed every other test in this module. Measured
/// on the 2-D fixture, the dropped term `2·Ω⁻¹₀₁·S₀₁` is only `0.003`–`0.012` on the `−2`
/// scale, because the posterior there is genuinely correlated (`−0.30` to `−0.42`) but tight
/// (`S₀₀ ≈ 0.014`). The MC bridge test's `4·SEM` band is `0.047`–`0.099`, so it cannot
/// resolve it, and the bound test cannot either: a too-small KL makes `−2·ELBO` *smaller* by
/// `0.007`, nowhere near enough to cross a gap of that size.
///
/// A `1e-12` deterministic comparison catches it immediately. The lesson generalizes — when
/// both sides of a check have a closed form, compare the closed forms.
///
/// # The cases
///
/// `S` is not merely a scaled Laplace covariance here, because the whole point is to reach
/// off-diagonals that a near-diagonal posterior never produces: the 2-D cases include
/// correlations of both signs up to `±0.8`, against a prior `Ω` that is itself correlated.
/// [`MeanField`] is included too — its diagonal-only trace is *correct* for a diagonal `S`,
/// since `Ω⁻¹ₖⱼSⱼₖ` vanishes off the diagonal, so the same textbook formula must reproduce it.
#[test]
fn production_kl_matches_the_textbook_kl() {
    let omega_1d = DMatrix::from_vec(1, 1, vec![0.25]);
    let omega_2d = DMatrix::from_row_slice(2, 2, &[0.25, 0.06, 0.06, 0.16]);

    let cases: Vec<KlCase> = vec![
        (
            "1d: q at the prior",
            omega_1d.clone(),
            DVector::from_vec(vec![0.0]),
            omega_1d.clone(),
        ),
        (
            "1d: displaced, narrow",
            omega_1d.clone(),
            DVector::from_vec(vec![0.6]),
            DMatrix::from_vec(1, 1, vec![0.01]),
        ),
        (
            "1d: displaced, wide",
            omega_1d.clone(),
            DVector::from_vec(vec![-0.4]),
            DMatrix::from_vec(1, 1, vec![1.5]),
        ),
        (
            "2d: q at the prior",
            omega_2d.clone(),
            DVector::zeros(2),
            omega_2d.clone(),
        ),
        (
            "2d: S nearly diagonal",
            omega_2d.clone(),
            DVector::from_vec(vec![0.2, -0.3]),
            DMatrix::from_row_slice(2, 2, &[0.014, -0.002, -0.002, 0.0035]),
        ),
        (
            "2d: S correlated +0.8",
            omega_2d.clone(),
            DVector::from_vec(vec![0.5, 0.4]),
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    0.20,
                    0.8 * 0.20_f64.sqrt() * 0.12_f64.sqrt(),
                    0.8 * 0.20_f64.sqrt() * 0.12_f64.sqrt(),
                    0.12,
                ],
            ),
        ),
        (
            "2d: S correlated -0.8",
            omega_2d.clone(),
            DVector::from_vec(vec![-0.7, 0.6]),
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    0.20,
                    -0.8 * 0.20_f64.sqrt() * 0.12_f64.sqrt(),
                    -0.8 * 0.20_f64.sqrt() * 0.12_f64.sqrt(),
                    0.12,
                ],
            ),
        ),
        (
            "2d: collapsed and displaced",
            omega_2d.clone(),
            DVector::from_vec(vec![1.2, -0.9]),
            DMatrix::from_row_slice(2, 2, &[0.002, 0.0005, 0.0005, 0.001]),
        ),
    ];

    for (label, omega_m, mu, s) in cases {
        let d = mu.len();
        let names: Vec<String> = (0..d).map(|k| format!("ETA{k}")).collect();
        let is_diag = omega_m.iter().enumerate().all(|(idx, v)| {
            let (r, c) = (idx % d, idx / d);
            r == c || *v == 0.0
        });
        let omega = crate::types::OmegaMatrix::from_matrix(omega_m.clone(), names, is_diag);
        let expected = kl_to_prior(&mu, &s, &omega_m);

        // Full rank: the general case, off-diagonals and all.
        let full = FullRank::new(d);
        let phi = full_rank_phi(&mu, &s);
        let got = full
            .kl_to_normal(&phi, &omega)
            .expect("FullRank has a closed-form KL")
            .value;
        assert!(
            (got - expected).abs() < 1e-12,
            "[{label}] FullRank::kl_to_normal = {got:.14}, textbook = {expected:.14} \
             (difference {:.3e}); a closed form must match a closed form",
            (got - expected).abs()
        );

        // Mean field: only meaningful against a diagonal S, which is all it can represent.
        let s_diag = DMatrix::from_fn(d, d, |r, c| if r == c { s[(r, r)] } else { 0.0 });
        let mf = MeanField::new(d);
        let mut phi_mf: Vec<f64> = mu.iter().copied().collect();
        phi_mf.extend((0..d).map(|k| 0.5 * s[(k, k)].ln()));
        let got_mf = mf
            .kl_to_normal(&phi_mf, &omega)
            .expect("MeanField has a closed-form KL")
            .value;
        let expected_mf = kl_to_prior(&mu, &s_diag, &omega_m);
        assert!(
            (got_mf - expected_mf).abs() < 1e-12,
            "[{label}] MeanField::kl_to_normal = {got_mf:.14}, textbook = {expected_mf:.14} \
             (difference {:.3e})",
            (got_mf - expected_mf).abs()
        );
    }
}

/// **Claim (b): a `mean_field` `q` gives a looser bound than `full_rank` — measured at
/// *one* parameter vector, with a closed form to check the size against.**
///
/// `docs/estimation/vi.qmd` tells users a diagonal `q` costs "a looser bound whenever the
/// true posterior is correlated". `VI_VALIDATION.md` §4.13 could only get a *direction* for
/// that from the `emvi` harness, and explains why: each family arm runs its own M-step, so
/// the two ELBOs are evaluated at different `(θ, Ω, σ)` — and an ELBO bounds `−2 log L` at
/// the parameter vector where it was evaluated, so a gap between two different vectors
/// attributes nothing. On the `emvi` side the arms still drift 5.7% apart.
///
/// This test is the route §4.13 names. Both families are evaluated at the **same** fixed
/// parameter vector and the **same** posterior mode, by the deterministic quadrature in
/// this module, so nothing is left for a parameter difference or Monte-Carlo noise to
/// explain.
///
/// # Which `q` represents each family
///
/// For a Gaussian posterior `N(m, Σ)` with precision `H = Σ⁻¹`:
///
/// * the best **full-rank** `q` is the posterior itself, `S = Σ`, giving a zero gap;
/// * the best **mean-field** `q` matches the diagonal of the **precision**, not of the
///   covariance: `Sₖₖ = 1/Hₖₖ`. This is the textbook result behind the whole
///   variance-understatement story — `1/Hₖₖ ≤ Σₖₖ`, with equality only when the posterior
///   is uncorrelated.
///
/// Substituting the second into `2·KL(q ‖ p)` makes the trace term collapse to `d`
/// exactly, so in 2-D the bound gap has the closed form
///
/// ```text
///   −2·ELBO_meanfield − (−2 log p(y)) = −log(1 − r²),    r = H₁₂ / √(H₁₁H₂₂)
/// ```
///
/// the **precision** correlation. That is what makes this a measurement rather than a
/// direction: the test asserts not just that `mean_field` is looser but that it is looser
/// by the predicted amount.
///
/// # Why the inequality is safe despite `S = Σ` being only approximately optimal
///
/// The posterior here is not exactly Gaussian, so the Laplace `q` is not exactly the
/// full-rank optimum. It does not need to be. The mean-field family is *contained* in the
/// full-rank family, so
///
/// ```text
///   best mean-field  ≥  this particular full-rank q  ≥  best full-rank
/// ```
///
/// and showing the mean-field optimum is looser than one specific full-rank `q` settles the
/// claim in the direction that matters. The mean-field side *is* asserted to be its own
/// family's optimum, by perturbation: scaling any single variance up or down worsens the
/// bound.
///
/// One-`η` fixtures are skipped — with `d = 1` the two families are the same family, and
/// `−log(1 − r²)` is zero, so there is nothing to measure.
#[test]
fn the_mean_field_bound_is_looser_than_full_rank_at_one_parameter_vector() {
    for fx in fixtures() {
        let params = fx.model.default_params.clone();
        let d = params.omega.dim();
        if d < 2 {
            continue;
        }

        for i in 0..fx.pop.subjects.len() {
            let subject = &fx.pop.subjects[i];
            let marginal = agq_neg_two_log_marginal(&fx.model, &fx.pop, i, &params, fx.agq_nodes);
            let (mode, lap_cov) = laplace_posterior(&fx.model, subject, &params);
            let precision = lap_cov
                .clone()
                .try_inverse()
                .expect("the Laplace covariance is invertible");

            // The mean-field optimum: S_kk = 1/H_kk.
            let mf_cov = DMatrix::from_diagonal(&DVector::from_iterator(
                d,
                (0..d).map(|k| 1.0 / precision[(k, k)]),
            ));

            let full_rank_bound =
                neg_two_elbo_quad(&fx.model, subject, &params, &mode, &lap_cov, fx.quad_nodes);
            let mean_field_bound =
                neg_two_elbo_quad(&fx.model, subject, &params, &mode, &mf_cov, fx.quad_nodes);

            // `-log(1 - r^2)` on the precision correlation, for d = 2.
            let r = precision[(0, 1)] / (precision[(0, 0)] * precision[(1, 1)]).sqrt();
            let predicted_gap = -(1.0 - r * r).ln();

            assert!(
                mean_field_bound > full_rank_bound,
                "[{}] subject {}: mean_field -2*ELBO = {mean_field_bound:.6} is not looser \
                 than full_rank's {full_rank_bound:.6}. The posterior's precision \
                 correlation is {r:.4}, so a diagonal q must pay for it.",
                fx.label,
                subject.id
            );

            // Both are still bounds — a looser bound is still a bound.
            for (label, value) in [
                ("full_rank", full_rank_bound),
                ("mean_field", mean_field_bound),
            ] {
                assert!(
                    value >= marginal - 1e-7,
                    "[{}] subject {} {label}: -2*ELBO = {value:.10} fell BELOW \
                     -2 log p(y) = {marginal:.10}",
                    fx.label,
                    subject.id
                );
            }

            // The measured gap against its closed form. The closed form assumes an exactly
            // Gaussian posterior; this one is not, so 12% of slack covers that plus the
            // quadrature's own error. It is a threshold in an empty gap, not a guess:
            // measured error is 4-8% across the three subjects, and substituting the
            // *covariance* diagonal for the precision diagonal -- the natural wrong q, and
            // the one the variance-understatement story is about -- reads 17.1%.
            let measured_gap = mean_field_bound - full_rank_bound;
            let gap_err = (measured_gap - predicted_gap).abs() / predicted_gap.abs().max(1e-12);
            assert!(
                gap_err < 0.12,
                "[{}] subject {}: the family cost measured {measured_gap:.6} against the \
                 closed form -log(1 - r^2) = {predicted_gap:.6} (r = {r:.4}, {:.1}% off). \
                 A disagreement here means the mean-field optimum is not where the theory \
                 puts it, or the KL is wrong for a diagonal S.",
                fx.label,
                subject.id,
                gap_err * 100.0
            );

            // The mean-field q really is its family's optimum: any single-coordinate rescale
            // makes the bound worse. This is what licenses reading `measured_gap` as the cost
            // of the *family* rather than of one arbitrary diagonal q.
            for k in 0..d {
                for scale in [0.75_f64, 1.33] {
                    let mut perturbed = mf_cov.clone();
                    perturbed[(k, k)] *= scale;
                    let worse = neg_two_elbo_quad(
                        &fx.model,
                        subject,
                        &params,
                        &mode,
                        &perturbed,
                        fx.quad_nodes,
                    );
                    assert!(
                        worse > mean_field_bound,
                        "[{}] subject {}: scaling S[{k},{k}] by {scale} IMPROVED the \
                         mean-field bound ({worse:.6} against {mean_field_bound:.6}), so \
                         1/H_kk is not the diagonal optimum and the comparison above is \
                         against the wrong q.",
                        fx.label,
                        subject.id
                    );
                }
            }
        }
    }
}
