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
    let out = d.draw(
        &mut theta, &mut etas, &omega, &lower, &upper, &pop, &mut rng,
    );
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
    let out = d.draw(
        &mut theta,
        &mut etas,
        &omega,
        &open_lower,
        &tight_upper,
        &pop,
        &mut StdRng::seed_from_u64(3),
    );
    assert_eq!(out, DrawOutcome::RejectedOutOfBounds);
    assert_eq!(theta, params.theta, "θ unchanged on reject");
    assert_eq!(etas, etas0, "η unchanged on reject");

    let mut theta = params.theta.clone();
    let mut etas = etas0.clone();
    let out = d.draw(
        &mut theta,
        &mut etas,
        &omega,
        &open_lower,
        &open_upper,
        &pop,
        &mut StdRng::seed_from_u64(3),
    );
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
    let out = d.draw(
        &mut theta,
        &mut etas,
        &omega,
        &neg,
        &open,
        &pop,
        &mut StdRng::seed_from_u64(11),
    );
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
    // sqrt(θ0) and θ0·θ0 are log-affine.
    assert_eq!(
        log_form(&E::UnaryFn("sqrt".into(), th(0)), &free),
        Some(vec![(0, true)])
    );
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
