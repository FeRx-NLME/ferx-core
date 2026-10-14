use super::*;
use crate::types::{CompiledModel, OmegaMatrix, Population};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::Path;

/// `examples/two_cpt_oral_cov.ferx`: CL reads TVCL, THETA_WT, THETA_CRCL; V1
/// reads TVV1 and the same THETA_WT (#1620's joint group); Q, V2, KA are plain
/// log anchors.
fn cov_model() -> (CompiledModel, Population) {
    let model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/two_cpt_oral_cov.ferx"))
            .expect("two_cpt_oral_cov parses");
    let pop = crate::read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
        .expect("two_cpt_oral_cov data loads");
    (model, pop)
}

fn theta_idx(model: &CompiledModel, name: &str) -> usize {
    model
        .theta_names
        .iter()
        .position(|n| n == name)
        .expect(name)
}

/// Deterministic, subject- and eta-varying η values.
fn fixture_etas(n_subjects: usize, n_eta: usize) -> Vec<Vec<f64>> {
    (0..n_subjects)
        .map(|i| {
            (0..n_eta)
                .map(|j| 0.3 * (((i * 7 + j * 3) % 11) as f64 / 10.0 - 0.5) + 0.02 * j as f64)
                .collect()
        })
        .collect()
}

/// The production log target the sampler hands `draw`: `−Σᵢ individual_nll`
/// at the model's default σ.
fn nll_target<'a>(
    model: &'a CompiledModel,
    pop: &'a Population,
    omega: &'a OmegaMatrix,
) -> impl FnMut(&[f64], &[Vec<f64>]) -> Vec<f64> + 'a {
    move |theta: &[f64], etas: &[Vec<f64>]| {
        pop.subjects
            .iter()
            .zip(etas)
            .map(|(s, e)| {
                crate::stats::likelihood::individual_nll(
                    model,
                    s,
                    theta,
                    e,
                    omega,
                    &model.default_params.sigma.values,
                )
            })
            .collect()
    }
}

/// The per-subject NLL baseline [`MuDesign::draw`] takes in, at the state the
/// draw starts from.
fn nll_baseline(
    model: &CompiledModel,
    pop: &Population,
    omega: &OmegaMatrix,
    theta: &[f64],
    etas: &[Vec<f64>],
) -> Vec<f64> {
    nll_target(model, pop, omega)(theta, etas)
}

/// A dense, positive-definite Ω, so the cross terms of Ω⁻¹ between member
/// and non-member etas are live in the conditional.
fn dense_omega(n: usize, names: Vec<String>) -> OmegaMatrix {
    let mut m = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = if i == j { 0.1 + 0.02 * i as f64 } else { 0.015 };
        }
    }
    OmegaMatrix::from_matrix(m, names, false)
}

#[test]
fn covariate_exponents_join_the_conjugate_design_on_their_natural_scale() {
    let (model, pop) = cov_model();
    let d =
        MuDesign::build(&model, &pop, &model.default_params, true).expect("every eta is covered");
    let coord = |n: &str| {
        let t = theta_idx(&model, n);
        *d.coords.iter().find(|c| c.theta_idx == t).expect(n)
    };
    for tv in ["TVCL", "TVV1", "TVQ", "TVV2", "TVKA"] {
        assert!(
            coord(tv).log,
            "{tv} is a typical value: drawn on the log scale"
        );
    }
    for ex in ["THETA_WT", "THETA_CRCL"] {
        assert!(
            !coord(ex).log,
            "{ex} is an exponent: linear only on its own scale"
        );
    }
    assert_eq!(d.coords.len(), 7);
}

#[test]
fn without_groups_the_design_is_the_anchor_only_move() {
    let (model, pop) = cov_model();
    let d = MuDesign::build(&model, &pop, &model.default_params, false)
        .expect("every eta has a log anchor");
    let mut got = d.theta_indices();
    got.sort_unstable();
    let mut want: Vec<usize> = ["TVCL", "TVV1", "TVQ", "TVV2", "TVKA"]
        .iter()
        .map(|n| theta_idx(&model, n))
        .collect();
    want.sort_unstable();
    assert_eq!(got, want, "the exponents stay with the random walk");
}

/// The conditional against the regression written out by hand from the data's
/// covariates — `log CL_i = ln TVCL + θ_WT·ln(WT_i/70) + θ_CRCL·ln(CRCL_i/100)`,
/// `log V1_i = ln TVV1 + θ_WT·ln(WT_i/70)` — not from the typical-value
/// evaluator the design uses: `P = Σ XᵢᵀΩ⁻¹Xᵢ`, mean `β + P⁻¹ Σ XᵢᵀΩ⁻¹ηᵢ`.
#[test]
fn conjugate_conditional_matches_the_closed_form_regression() {
    let (model, pop) = cov_model();
    let d = MuDesign::build(&model, &pop, &model.default_params, true).expect("design");
    let n_eta = model.n_eta;
    let theta = model.default_params.theta.clone();
    let etas = fixture_etas(pop.subjects.len(), n_eta);
    let omega = dense_omega(n_eta, model.eta_names.clone());
    let (mean, prec) = d.conditional(&theta, &etas, &omega).expect("conditional");

    let p = d.coords.len();
    let col = |n: &str| {
        let t = theta_idx(&model, n);
        d.coords.iter().position(|c| c.theta_idx == t).expect(n)
    };
    let eta_row = |n: &str| model.eta_names.iter().position(|e| e == n).expect(n);
    let w = omega.matrix.clone().try_inverse().expect("Ω invertible");
    let mut p_hand = DMatrix::<f64>::zeros(p, p);
    let mut g_hand = DVector::<f64>::zeros(p);
    for (s, eta) in pop.subjects.iter().zip(&etas) {
        let lwt = (s.covariates["WT"] / 70.0).ln();
        let lcr = (s.covariates["CRCL"] / 100.0).ln();
        let mut x = DMatrix::<f64>::zeros(n_eta, p);
        x[(eta_row("ETA_CL"), col("TVCL"))] = 1.0;
        x[(eta_row("ETA_CL"), col("THETA_WT"))] = lwt;
        x[(eta_row("ETA_CL"), col("THETA_CRCL"))] = lcr;
        x[(eta_row("ETA_V1"), col("TVV1"))] = 1.0;
        x[(eta_row("ETA_V1"), col("THETA_WT"))] = lwt;
        x[(eta_row("ETA_Q"), col("TVQ"))] = 1.0;
        x[(eta_row("ETA_V2"), col("TVV2"))] = 1.0;
        x[(eta_row("ETA_KA"), col("TVKA"))] = 1.0;
        let xtw = x.transpose() * &w;
        p_hand += &xtw * &x;
        g_hand += &xtw * DVector::from_column_slice(eta);
    }
    let beta: DVector<f64> = DVector::from_iterator(
        p,
        d.coords.iter().map(|c| {
            let v = theta[c.theta_idx];
            if c.log {
                v.ln()
            } else {
                v
            }
        }),
    );
    let mean_hand = &beta + p_hand.clone().try_inverse().expect("P invertible") * g_hand;
    let mut worst_p = 0.0_f64;
    for a in 0..p {
        for b in 0..p {
            let rel = (prec[(a, b)] - p_hand[(a, b)]).abs() / p_hand[(a, b)].abs().max(1.0);
            assert!(rel.is_finite(), "precision ({a},{b}) not finite");
            worst_p = worst_p.max(rel);
        }
    }
    let mut worst_m = 0.0_f64;
    for k in 0..p {
        let err = (mean[k] - mean_hand[k]).abs();
        assert!(err.is_finite(), "mean[{k}] not finite");
        worst_m = worst_m.max(err);
    }
    // Measured (Linux): precision 5.6e-15 relative, mean 5.6e-16 absolute — the
    // central difference of an affine function is exact up to rounding. A
    // wrong design (θ_WT dropped from V1, a missed CRCL column) moves both by
    // O(0.1–1); the bounds keep ~1000× headroom over the measurement.
    assert!(worst_p < 1e-11, "precision differs by {worst_p:e}");
    assert!(worst_m < 1e-12, "mean differs by {worst_m:e}");
}

/// The draws themselves have the conditional's mean and covariance.
#[test]
fn conjugate_draws_have_the_conditional_moments() {
    let (model, pop) = cov_model();
    let d = MuDesign::build(&model, &pop, &model.default_params, true).expect("design");
    let theta = model.default_params.theta.clone();
    let etas = fixture_etas(pop.subjects.len(), model.n_eta);
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let (mean, prec) = d.conditional(&theta, &etas, &omega).expect("conditional");
    let cov = prec.try_inverse().expect("P invertible");
    let p = d.coords.len();
    let n = 5_000usize;
    let mut rng = StdRng::seed_from_u64(1626);
    let mut sum = DVector::<f64>::zeros(p);
    let mut sq = DMatrix::<f64>::zeros(p, p);
    for _ in 0..n {
        let t = d
            .sample_theta(&theta, &etas, &omega, &mut rng)
            .expect("draw");
        let b = DVector::from_iterator(
            p,
            d.coords.iter().map(|c| {
                let v = t[c.theta_idx];
                if c.log {
                    v.ln()
                } else {
                    v
                }
            }),
        );
        let dv = &b - &mean;
        sum += &dv;
        sq += &dv * dv.transpose();
    }
    for k in 0..p {
        let sd = cov[(k, k)].sqrt();
        let m = sum[k] / n as f64;
        // 4.5 Monte Carlo SEs of the mean.
        assert!(
            m.abs() < 4.5 * sd / (n as f64).sqrt(),
            "coord {k}: mean offset {m:e} vs sd {sd:e}"
        );
        let v = sq[(k, k)] / n as f64;
        // Var of a sample variance ≈ 2σ⁴/n → relative SE √(2/n) = 2%; allow
        // 4 SEs. Drawing with P instead of P⁻¹ is off by orders of magnitude.
        assert!(
            (v / cov[(k, k)] - 1.0).abs() < 0.08,
            "coord {k}: variance {v:e} vs {:e}",
            cov[(k, k)]
        );
    }
}

/// A moved draw holds every individual parameter: `φ = μ(θ) + η` is unchanged,
/// so the data likelihood is untouched.
#[test]
fn a_moved_draw_holds_every_individual_parameter() {
    let (model, pop) = cov_model();
    let params = &model.default_params;
    let d = MuDesign::build(&model, &pop, params, true).expect("design");
    let mut theta = params.theta.clone();
    let mut etas = fixture_etas(pop.subjects.len(), model.n_eta);
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let phi_before: Vec<Vec<f64>> = pop
        .subjects
        .iter()
        .zip(&etas)
        .map(|(s, e)| d.mus(&theta, s).iter().zip(e).map(|(m, x)| m + x).collect())
        .collect();
    let lower = vec![f64::NEG_INFINITY; theta.len()];
    let upper = vec![f64::INFINITY; theta.len()];
    let mut rng = StdRng::seed_from_u64(7);
    let theta0 = theta.clone();
    let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
    let out = d
        .draw(
            &mut theta,
            &mut etas,
            &omega,
            &lower,
            &upper,
            &pop,
            &mut rng,
            &mut nll,
            &mut nll_target(&model, &pop, &omega),
        )
        .0;
    assert_eq!(out, DrawOutcome::Moved);
    let wt = theta_idx(&model, "THETA_WT");
    assert!(theta[wt] != theta0[wt], "the exponent was drawn");
    let mut worst = 0.0_f64;
    for ((s, e), before) in pop.subjects.iter().zip(&etas).zip(&phi_before) {
        for ((m, x), b) in d.mus(&theta, s).iter().zip(e).zip(before) {
            let err = (m + x - b).abs();
            assert!(err.is_finite());
            worst = worst.max(err);
        }
    }
    // Measured (Linux): exactly 0. A missed or partial re-centring (only the
    // first group member, say) moves φ by the drawn shift, O(0.01–0.1).
    assert!(worst < 1e-12, "φ moved by {worst:e}");

    // The move owns the caller's NLL cache: `draw` evaluates the proposal and,
    // on a move, hands that vector back instead of leaving the caller to
    // recompute it. Bit-equality, not a tolerance — it is the same function at
    // the same state, and anything else is a stale baseline silently biasing
    // the (θ, σ) block's Metropolis ratio for the rest of the sweep.
    assert_eq!(
        nll,
        nll_baseline(&model, &pop, &omega, &theta, &etas),
        "the returned NLL baseline is the proposed state's"
    );
}

/// `draw` scores its Metropolis ratio against the per-subject NLL it is
/// **handed**, not one it recomputes: perturbing the baseline by δ must move
/// `log α` by exactly δ.
///
/// The regression this exists to catch is the caller and the move disagreeing
/// about which state the cached baseline describes — a refresh moved to the
/// wrong side of the block, or a gate that skips it. A `draw` that quietly
/// recomputed the current state would be immune to the perturbation and is
/// what this kills; it is also the only unit-level hold on the contract, since
/// the placement of that refresh inside `run_bayes` is an argument, not an
/// assertion.
#[test]
fn the_draw_scores_against_the_baseline_it_is_given() {
    let (model, pop) = cov_model();
    let params = &model.default_params;
    let d = MuDesign::build(&model, &pop, params, true).expect("design");
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let etas0 = fixture_etas(pop.subjects.len(), model.n_eta);
    let lo = vec![f64::NEG_INFINITY; params.theta.len()];
    let hi = vec![f64::INFINITY; params.theta.len()];
    const DELTA: f64 = 0.125; // exact in binary, so the comparison is exact.

    let run = |bump: f64| {
        let mut theta = params.theta.clone();
        let mut etas = etas0.clone();
        let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
        nll[0] += bump;
        d.draw(
            &mut theta,
            &mut etas,
            &omega,
            &lo,
            &hi,
            &pop,
            &mut StdRng::seed_from_u64(21),
            &mut nll,
            &mut nll_target(&model, &pop, &omega),
        )
        .1
    };

    let base = run(0.0);
    let bumped = run(DELTA);
    assert!(base.is_finite() && bumped.is_finite(), "{base} / {bumped}");
    // log α = Σ nll_given − Σ nll(proposal) + log q-ratio, so +δ on the given
    // baseline is +δ on log α. Measured: exactly DELTA.
    assert!(
        (bumped - base - DELTA).abs() < 1e-12,
        "log α moved by {} for a baseline bump of {DELTA}",
        bumped - base
    );
}

/// A draw outside a bound is rejected and changes nothing — and the same
/// draw inside wide bounds moves (both sides of the gate, one seed).
#[test]
fn an_out_of_bounds_draw_is_rejected_and_changes_nothing() {
    let (model, pop) = cov_model();
    let params = &model.default_params;
    let d = MuDesign::build(&model, &pop, params, true).expect("design");
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let wt = theta_idx(&model, "THETA_WT");
    let etas0 = fixture_etas(pop.subjects.len(), model.n_eta);
    let open_lower = vec![f64::NEG_INFINITY; params.theta.len()];
    let open_upper = vec![f64::INFINITY; params.theta.len()];
    // An upper bound on THETA_WT far below its conditional (the current value
    // 0.6 ± ~0.2): every draw lands above it.
    let mut tight_upper = open_upper.clone();
    tight_upper[wt] = -10.0;

    let mut theta = params.theta.clone();
    let mut etas = etas0.clone();
    let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
    let nll0 = nll.clone();
    let out = d
        .draw(
            &mut theta,
            &mut etas,
            &omega,
            &open_lower,
            &tight_upper,
            &pop,
            &mut StdRng::seed_from_u64(3),
            &mut nll,
            &mut nll_target(&model, &pop, &omega),
        )
        .0;
    assert_eq!(out, DrawOutcome::RejectedOutOfBounds);
    assert_eq!(theta, params.theta, "θ unchanged on reject");
    assert_eq!(etas, etas0, "η unchanged on reject");
    assert_eq!(
        nll, nll0,
        "the caller's NLL baseline is unchanged on reject"
    );

    let mut theta = params.theta.clone();
    let mut etas = etas0.clone();
    let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
    let out = d
        .draw(
            &mut theta,
            &mut etas,
            &omega,
            &open_lower,
            &open_upper,
            &pop,
            &mut StdRng::seed_from_u64(3),
            &mut nll,
            &mut nll_target(&model, &pop, &omega),
        )
        .0;
    assert_eq!(
        out,
        DrawOutcome::Moved,
        "the same draw inside the bounds moves"
    );
}

#[test]
fn within_bounds_is_inclusive_and_rejects_non_finite() {
    let lo = [0.0, f64::NEG_INFINITY];
    let hi = [1.0, f64::INFINITY];
    assert!(within_bounds(0.0, 0, &lo, &hi));
    assert!(within_bounds(1.0, 0, &lo, &hi));
    assert!(!within_bounds(1.0 + 1e-12, 0, &lo, &hi));
    assert!(!within_bounds(-1e-12, 0, &lo, &hi));
    assert!(within_bounds(-1e300, 1, &lo, &hi));
    assert!(!within_bounds(f64::NAN, 1, &lo, &hi));
    // A theta with no recorded bound is unbounded.
    assert!(within_bounds(5.0, 7, &lo, &hi));
}

/// A covariate entering additively inside the typical value is not affine in
/// any coordinates; the design must not claim it. Its eta has no plain anchor
/// either, so the whole move is off (the random walk takes every theta).
#[test]
fn a_non_affine_typical_value_is_left_to_the_random_walk() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TH_CRCL(0.02, -1.0, 1.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  omega ETA_V ~ 0.1
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = (TVCL + (CRCL - 100) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
    let (_m, pop) = cov_model();
    assert!(
        model
            .covariate_mu_refs
            .iter()
            .any(|c| c.eta_name == "ETA_CL"),
        "precondition: ETA_CL is a recorded covariate mu-reference"
    );
    assert!(
        MuDesign::build(&model, &pop, &model.default_params, true).is_none(),
        "an additive covariate inside the typical value is not a linear regression"
    );
}

/// Set covariate `name` to `value` on every record of every subject, so it
/// is subject-constant (the group keeps its exact, data-free engine).
fn set_covariate(pop: &mut Population, name: &str, value: impl Fn(usize) -> f64) {
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        let v = value(i);
        let maps = std::iter::once(&mut s.covariates)
            .chain(s.dose_covariates.iter_mut())
            .chain(s.obs_covariates.iter_mut())
            .chain(s.pk_only_covariates.iter_mut())
            .chain(s.reset_covariates.iter_mut());
        for m in maps {
            if let Some(slot) = m.get_mut(name) {
                *slot = v;
            }
        }
    }
}

fn has_coord(d: &MuDesign<'_>, model: &CompiledModel, name: &str) -> bool {
    d.theta_indices().contains(&theta_idx(model, name))
}

/// An odd nonlinear term cancels from every symmetric second difference about
/// the start, so a numerical probe admitted `(WT/70)^(θ + (θ − 0.75)³)` as
/// affine and drew θ_WT from a wrong conditional (codex review of #1951). The
/// structural check refuses it; the affine control in the same test is
/// admitted, so a gate stuck on either answer fails here.
#[test]
fn an_odd_nonlinear_exponent_is_not_admitted_but_the_affine_one_is() {
    let affine = std::fs::read_to_string("examples/two_cpt_oral_cov.ferx").expect("model text");
    let odd = affine.replace(
        "(WT / 70)^THETA_WT",
        "(WT / 70)^(THETA_WT + (THETA_WT - 0.6)^3)",
    );
    assert_ne!(odd, affine, "precondition: the substitution happened");
    let (_m, pop) = cov_model();

    let m_aff = crate::parser::model_parser::parse_model_string(&affine).expect("parses");
    let d_aff = MuDesign::build(&m_aff, &pop, &m_aff.default_params, true).expect("design");
    assert!(
        has_coord(&d_aff, &m_aff, "THETA_WT"),
        "control: the affine exponent is drawn"
    );

    let m_odd = crate::parser::model_parser::parse_model_string(&odd).expect("parses");
    assert!(
        m_odd
            .covariate_mu_refs
            .iter()
            .any(|c| c.eta_name == "ETA_CL"),
        "precondition: the odd form is still a recorded covariate mu-reference"
    );
    let d_odd = MuDesign::build(&m_odd, &pop, &m_odd.default_params, true)
        .expect("the anchors still cover every eta");
    assert!(
        !has_coord(&d_odd, &m_odd, "THETA_WT"),
        "a cubic in THETA_WT is not a linear regression coefficient"
    );
    // The group is refused as a whole: its other exponent stays on the
    // random walk too, while the anchors keep the conjugate move.
    assert!(!has_coord(&d_odd, &m_odd, "THETA_CRCL"));
    assert!(has_coord(&d_odd, &m_odd, "TVCL"));
}

/// Every subject at the covariates' reference values makes the exponent
/// columns zero, so their conditional precision is singular. They must go
/// back to the random walk — not freeze — and the typical values must keep
/// moving (codex review of #1951: the whole move used to be skipped).
#[test]
fn reference_valued_covariates_send_their_exponents_back_to_the_random_walk() {
    let (model, mut pop) = cov_model();
    set_covariate(&mut pop, "WT", |_| 70.0);
    set_covariate(&mut pop, "CRCL", |_| 100.0);
    let d = MuDesign::build(&model, &pop, &model.default_params, true).expect("design");
    assert!(!has_coord(&d, &model, "THETA_WT"));
    assert!(!has_coord(&d, &model, "THETA_CRCL"));
    for tv in ["TVCL", "TVV1", "TVQ", "TVV2", "TVKA"] {
        assert!(has_coord(&d, &model, tv), "{tv} keeps the conjugate move");
    }
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let mut theta = model.default_params.theta.clone();
    let mut etas = fixture_etas(pop.subjects.len(), model.n_eta);
    let open = vec![f64::INFINITY; theta.len()];
    let neg = vec![f64::NEG_INFINITY; theta.len()];
    let before = theta.clone();
    let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
    let out = d
        .draw(
            &mut theta,
            &mut etas,
            &omega,
            &neg,
            &open,
            &pop,
            &mut StdRng::seed_from_u64(11),
            &mut nll,
            &mut nll_target(&model, &pop, &omega),
        )
        .0;
    assert_eq!(out, DrawOutcome::Moved);
    let tvcl = theta_idx(&model, "TVCL");
    assert_ne!(theta[tvcl], before[tvcl], "TVCL moved");
}

/// A covariate constant across subjects but not at its reference value makes
/// its exponent's column a multiple of the typical value's: collinear, not
/// zero. Only that exponent is dropped.
#[test]
fn a_constant_covariate_drops_only_its_collinear_exponent() {
    let (model, mut pop) = cov_model();
    set_covariate(&mut pop, "CRCL", |_| 50.0);
    let d = MuDesign::build(&model, &pop, &model.default_params, true).expect("design");
    assert!(!has_coord(&d, &model, "THETA_CRCL"), "collinear with TVCL");
    assert!(has_coord(&d, &model, "THETA_WT"), "WT still varies");
    assert!(has_coord(&d, &model, "TVCL"));
    let omega = dense_omega(model.n_eta, model.eta_names.clone());
    let etas = fixture_etas(pop.subjects.len(), model.n_eta);
    assert!(
        d.conditional(&model.default_params.theta, &etas, &omega)
            .is_some(),
        "the reduced design's conditional can be formed"
    );
}

/// The structural walker on the forms it must and must not accept.
#[test]
fn affinity_is_proven_from_the_expression_form() {
    use crate::parser::model_parser::{BinOp, Expression as E};
    let free = |_t: usize| true;
    let th = |t| Box::new(E::Theta(t));
    let cov = |n: &str| Box::new(E::Covariate(n.to_string()));
    let lit = |v| Box::new(E::Literal(v));
    // θ0 · (WT/70)^θ1 · exp(θ2·SEX) → log θ0, θ1, θ2.
    let e = E::BinOp(
        Box::new(E::BinOp(
            th(0),
            BinOp::Mul,
            Box::new(E::Power(
                Box::new(E::BinOp(cov("WT"), BinOp::Div, lit(70.0))),
                th(1),
            )),
        )),
        BinOp::Mul,
        Box::new(E::UnaryFn(
            "exp".into(),
            Box::new(E::BinOp(th(2), BinOp::Mul, cov("SEX"))),
        )),
    );
    let mut got = log_form(&e, &free).expect("affine");
    got.sort_unstable();
    assert_eq!(got, vec![(0, true), (1, false), (2, false)]);
    // sqrt(θ0) is refused — the evaluator clamps its argument at 0 — while
    // θ0·θ0 is log-affine.
    assert_eq!(log_form(&E::UnaryFn("sqrt".into(), th(0)), &free), None);
    assert_eq!(
        log_form(&E::BinOp(th(0), BinOp::Mul, th(0)), &free),
        Some(vec![(0, true)])
    );
    // θ0 + c·θ1 under the log link: ln of a sum is not affine.
    let sum = E::BinOp(
        th(0),
        BinOp::Add,
        Box::new(E::BinOp(cov("CRCL"), BinOp::Mul, th(1))),
    );
    assert_eq!(log_form(&sum, &free), None);
    // ... but under the identity/logit link it is.
    let mut got = lin_form(&sum, &free).expect("affine");
    got.sort_unstable();
    assert_eq!(got, vec![(0, false), (1, false)]);
    // A product of two thetas is not linear; θ on two scales is refused.
    assert_eq!(lin_form(&E::BinOp(th(0), BinOp::Mul, th(1)), &free), None);
    let both = E::BinOp(
        th(0),
        BinOp::Mul,
        Box::new(E::UnaryFn(
            "exp".into(),
            Box::new(E::BinOp(th(0), BinOp::Mul, cov("WT"))),
        )),
    );
    assert_eq!(log_form(&both, &free), None);
    // A FIX-ed theta is a constant.
    let fixed_1 = |t: usize| t != 1;
    assert_eq!(
        log_form(&E::BinOp(th(0), BinOp::Mul, th(1)), &fixed_1),
        Some(vec![(0, true)])
    );
}

/// End to end: with the exponents' columns zero, every free θ still moves —
/// the typical values on the conjugate move, the exponents on the random walk
/// under their declared bounds.
#[test]
fn run_bayes_samples_every_theta_when_the_covariates_are_at_reference() {
    let (model, mut pop) = cov_model();
    set_covariate(&mut pop, "WT", |_| 70.0);
    set_covariate(&mut pop, "CRCL", |_| 100.0);
    let mut opts = crate::types::FitOptions::default();
    opts.bayes_warmup = 60;
    opts.bayes_iters = 60;
    opts.bayes_chains = 2;
    opts.bayes_seed = Some(1);
    opts.saem_n_mh_steps = 2;
    let res = crate::estimation::bayes::run_bayes(&model, &pop, &model.default_params, &opts)
        .expect("bayes runs");
    let b = res.bayes.as_ref().expect("BayesResult");
    for name in ["TVCL", "TVV1", "THETA_WT", "THETA_CRCL"] {
        let s = b.summaries.iter().find(|s| s.name == name).expect(name);
        assert!(
            s.sd.is_finite() && s.sd > 0.0,
            "{name} never moved (sd {})",
            s.sd
        );
    }
    for name in ["THETA_WT", "THETA_CRCL"] {
        let s = b.summaries.iter().find(|s| s.name == name).expect(name);
        assert!(s.q025 >= 0.01 && s.q975 <= 5.0, "{name} left its bounds");
    }
}

/// The DSL `log` is `ln(max(x, 1e-30))`, so `log(exp(θ·c))` is flat below the
/// floor, not linear (codex review of #1951, round 2). The walker refuses any
/// theta-dependent `log`; the unfloored control in the same test is admitted.
#[test]
fn a_floored_log_of_theta_is_not_admitted_but_the_unfloored_form_is() {
    let base = std::fs::read_to_string("examples/two_cpt_oral_cov.ferx")
        .expect("model text")
        .replace("THETA_WT(0.6, 0.01, 5.0)", "THETA_WT(0.6, -200.0, 200.0)");
    assert!(
        base.contains("THETA_WT(0.6, -200.0, 200.0)"),
        "precondition"
    );
    let control = base.replace("(WT / 70)^THETA_WT", "exp(THETA_WT * (WT / 70))");
    let floored = base.replace("(WT / 70)^THETA_WT", "exp(log(exp(THETA_WT * (WT / 70))))");
    let (_m, pop) = cov_model();
    let m_c = crate::parser::model_parser::parse_model_string(&control).expect("parses");
    let d_c = MuDesign::build(&m_c, &pop, &m_c.default_params, true).expect("design");
    assert!(
        has_coord(&d_c, &m_c, "THETA_WT"),
        "control: exp(θ·c) is linear in θ"
    );
    let m_f = crate::parser::model_parser::parse_model_string(&floored).expect("parses");
    assert!(
        m_f.covariate_mu_refs.iter().any(|c| c.eta_name == "ETA_CL"),
        "precondition: still a recorded covariate mu-reference"
    );
    let d_f = MuDesign::build(&m_f, &pop, &m_f.default_params, true).expect("anchors cover");
    assert!(
        !has_coord(&d_f, &m_f, "THETA_WT"),
        "log is clamped in the evaluator: not a linear coefficient"
    );
}

/// The move is corrected by Metropolis–Hastings on the real likelihood, so a
/// wrong belief about a typical value costs acceptance, never exactness. With
/// `CL = TVCL·(WT/70)^TVCL·…` the ETA_CL group is refused (TVCL on two
/// scales) but TVCL stays a log anchor, and re-centring only `ln TVCL` moves
/// the real CL (codex review of #1951, round 2: by 0.086 on seed 11). The
/// correction sees it: `log α` is far from 0. On the affine control in the
/// same test `log α` is 0 to rounding — the move is the exact Gibbs draw.
#[test]
fn the_conjugate_move_is_corrected_on_the_real_likelihood() {
    let affine = std::fs::read_to_string("examples/two_cpt_oral_cov.ferx").expect("model text");
    let a = "CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)";
    assert!(affine.contains(a), "precondition: the CL line");
    let bad = affine.replace(
        a,
        "CL = TVCL * (WT / 70)^TVCL * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)",
    );
    let (_m, pop) = cov_model();
    let worst_log_alpha = |src: &str| -> (f64, usize) {
        let model = crate::parser::model_parser::parse_model_string(src).expect("parses");
        let d = MuDesign::build(&model, &pop, &model.default_params, true).expect("design");
        assert!(
            has_coord(&d, &model, "TVCL"),
            "precondition: TVCL is conjugate"
        );
        let omega = dense_omega(model.n_eta, model.eta_names.clone());
        let theta0 = model.default_params.theta.clone();
        let etas0 = fixture_etas(pop.subjects.len(), model.n_eta);
        let open_lo = vec![f64::NEG_INFINITY; theta0.len()];
        let open_hi = vec![f64::INFINITY; theta0.len()];
        let mut worst = 0.0_f64;
        let mut rejected = 0;
        for seed in 0..20u64 {
            let mut theta = theta0.clone();
            let mut etas = etas0.clone();
            let mut nll = nll_baseline(&model, &pop, &omega, &theta, &etas);
            let (out, la) = d.draw(
                &mut theta,
                &mut etas,
                &omega,
                &open_lo,
                &open_hi,
                &pop,
                &mut StdRng::seed_from_u64(seed),
                &mut nll,
                &mut nll_target(&model, &pop, &omega),
            );
            assert!(la.is_finite(), "seed {seed}: log α {la}");
            worst = worst.max(la.abs());
            if out == DrawOutcome::RejectedMetropolis {
                rejected += 1;
                assert_eq!(theta, theta0, "a rejected draw changes nothing");
                assert_eq!(etas, etas0);
            }
        }
        (worst, rejected)
    };
    let (ctrl, ctrl_rej) = worst_log_alpha(&affine);
    let (bad_la, bad_rej) = worst_log_alpha(&bad);
    // Measured (Linux): control 5.4e-12, 0 of 20 rejected; the TVCL-exponent
    // model up to 5.3e4, 11 of 20 rejected. Without the correction all 11
    // would have moved the real CL with nothing to account for it.
    assert!(ctrl < 1e-8, "affine design: |log α| {ctrl:e} should be ~0");
    assert_eq!(ctrl_rej, 0);
    assert!(
        bad_la > 0.05,
        "the correction must see the moved CL: |log α| {bad_la}"
    );
    assert!(bad_rej > 0, "and reject some draws");
}
