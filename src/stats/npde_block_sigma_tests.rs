//! #1733: the NPDE reference simulation draws its residuals at the **parameter
//! set's** `block_sigma` correlation, and draws paired rows jointly from the
//! dense `R` the way `simulate()` does.
//!
//! The oracle is a closed form outside both engines. With ω ≈ 0 every replicate
//! shares one IPRED `f`, so the simulated residual vector is exactly
//! `N(0, R(ρ_sim))` and the scores reduce to standardised residuals: one row's
//! NPD is `(y − f) / √R₁₁`, and the second row of a pair, after the Cholesky
//! decorrelation, has NPDE `(z₂ − ρ·z₁) / √(1 − ρ²)` with `z` the residuals in
//! σ units. Both are computed here from `ρ` alone, never from ferx.

use super::*;
use crate::types::{DoseEvent, ResidualCorrelation, Subject};
use std::collections::HashMap;

fn population(subject: Subject, covariate_names: Vec<String>) -> Population {
    Population {
        subjects: vec![subject],
        covariate_names,
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn with_rho(model: &CompiledModel, rho: f64) -> Vec<ResidualCorrelation> {
    assert_eq!(model.residual_correlations.len(), 1);
    vec![ResidualCorrelation {
        rho,
        ..model.residual_correlations[0].clone()
    }]
}

/// Arm A: one observation, `combined(PROP, ADD)`, σ = (0.2, 1.0), declared
/// ρ = 0.5 (`0.10 / (0.2·1.0)`), ω ≈ 0.
const ARM_A: &str =
    "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  \
    omega ETA_CL ~ 1e-10 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00]\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
    [error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n";

/// Arm B: a total (`FREE = 0`) / unbound (`FREE = 1`) pair at t = 1, the two
/// proportional sigmas σ = (0.05, 0.30) in one `block_sigma` with declared
/// ρ = 0.5 (`0.0075 / (0.05·0.30)`), ω ≈ 0.
const ARM_B: &str =
    "[parameters]\n  theta TVCL(1.0, 0.1, 10.0) FIX\n  theta TVV(10.0, 1.0, 100.0) FIX\n  \
    omega ETA_CL ~ 1e-10 FIX\n  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.0025, 0.0075, 0.09]\n\
    [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
    [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
    [error_model]\n  if (FREE == 0) {\n    DV ~ proportional(PROP_TOTAL)\n  } else {\n    \
    DV ~ proportional(PROP_UNBOUND)\n  }\n\
    [covariates]\n  FREE continuous\n";

fn parse(text: &str) -> CompiledModel {
    crate::parser::model_parser::parse_full_model(text)
        .expect("parse")
        .model
}

fn arm_a_subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0],
        observations: vec![0.0],
        obs_cmts: vec![1],
        cens: vec![0],
        fremtype: vec![0],
        ..Default::default()
    }
}

fn arm_b_subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0],
        observations: vec![0.0, 0.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        fremtype: vec![0, 0],
        obs_covariates: [0.0, 1.0]
            .iter()
            .map(|&f| HashMap::from([("FREE".to_string(), f)]))
            .collect(),
        ..Default::default()
    }
}

fn ipreds(model: &CompiledModel, subject: &Subject, params: &ModelParameters) -> Vec<f64> {
    crate::pk::compute_predictions_with_tv(model, subject, &params.theta, &vec![0.0; model.n_eta])
}

/// Arm A. The fit's ρ = −0.9 against a declared 0.5; the observation sits one
/// fitted SD above `f`, so NPD = 1 at the fitted ρ and `sd_fit / sd_decl` at the
/// declared one. Before #1733 the draw read the declaration.
#[test]
fn npd_scores_a_free_block_sigma_at_the_fitted_correlation() {
    let model = parse(ARM_A);
    let mut params = model.default_params.clone();
    params.residual_correlations = with_rho(&model, -0.9);
    let mut subject = arm_a_subject();
    let f = ipreds(&model, &subject, &params)[0];
    assert!(f.is_finite() && f > 1.0, "f = {f}");
    let var = |rho: f64| 0.04 * f * f + 1.0 + 2.0 * rho * 0.2 * f;
    let sd_fit = var(-0.9).sqrt();
    subject.observations = vec![f + sd_fit];
    let cf_fit = 1.0;
    let cf_decl = sd_fit / var(0.5).sqrt();
    let pop = population(subject, vec![]);
    let mut worst = 0.0f64;
    for seed in 1..=SEEDS {
        let out = compute_npde_npd(&model, &pop, &params, NSIM, Some(seed)).expect("npde");
        let npd = out[0].npd[0];
        assert!(npd.is_finite(), "seed {seed}: npd = {npd}");
        eprintln!("arm A seed {seed}: npd = {npd}");
        worst = worst.max((npd - cf_fit).abs());
    }
    eprintln!("arm A: cf_fit {cf_fit} cf_decl {cf_decl} worst |npd - cf_fit| = {worst}");
    // The straddle: the two closed forms are farther apart than the bound.
    assert!((cf_fit - cf_decl).abs() > 3.0 * TOL_A, "degenerate fixture");
    assert!(
        worst < TOL_A,
        "NPD not at the fitted ρ: worst |npd − {cf_fit}| = {worst} (declared ρ gives {cf_decl})"
    );
}

/// Arm B. The pair sits at `z = (1.0, 0.5)` in σ units; the fit's ρ = −0.8
/// against a declared 0.5. Row 2's NPDE is `(0.5 + 0.8) / 0.6 = 2.167` at the
/// fitted ρ, 0.5 under independent per-row draws (ρ = 0), and 0 at the declared
/// ρ. Row 2's NPD is the marginal `z₂ = 0.5` whatever ρ is.
#[test]
fn npde_decorrelates_a_paired_block_sigma_at_the_fitted_correlation() {
    let model = parse(ARM_B);
    let mut params = model.default_params.clone();
    params.residual_correlations = with_rho(&model, -0.8);
    let mut subject = arm_b_subject();
    let f = ipreds(&model, &subject, &params);
    assert!(f.iter().all(|v| v.is_finite() && *v > 1.0), "f = {f:?}");
    // The fixture must reach the dense-R draw, not its diagonal fast path.
    let err_keys = model.error_spec.obs_keys(&subject);
    let r = crate::stats::residual_error::compute_r_matrix_with_correlations(
        &model.error_spec,
        &f,
        err_keys.as_ref(),
        &subject.obs_times,
        &subject.obs_raw_times,
        &subject.occasions,
        &subject.obs_l2,
        &params.sigma.values,
        &params.residual_correlations,
    );
    assert!(r[(0, 1)] != 0.0, "the pair is not paired in R: {r}");
    let (z1, z2) = (1.0, 0.5);
    subject.observations = vec![f[0] * (1.0 + z1 * 0.05), f[1] * (1.0 + z2 * 0.30)];
    let cf = |rho: f64| (z2 - rho * z1) / (1.0 - rho * rho).sqrt();
    let (cf_fit, cf_indep, cf_decl) = (cf(-0.8), cf(0.0), cf(0.5));
    let pop = population(subject, vec!["FREE".into()]);
    let mut worst_npde = 0.0f64;
    let mut worst_npd = 0.0f64;
    for seed in 1..=SEEDS {
        let out = compute_npde_npd(&model, &pop, &params, NSIM, Some(seed)).expect("npde");
        let (npde, npd) = (out[0].npde[1], out[0].npd[1]);
        assert!(
            npde.is_finite() && npd.is_finite(),
            "seed {seed}: {npde} {npd}"
        );
        eprintln!("arm B seed {seed}: npde2 = {npde}, npd2 = {npd}");
        worst_npde = worst_npde.max((npde - cf_fit).abs());
        worst_npd = worst_npd.max((npd - z2).abs());
    }
    eprintln!(
        "arm B: cf_fit {cf_fit} cf_indep {cf_indep} cf_decl {cf_decl} \
         worst |npde2 - cf_fit| = {worst_npde}, worst |npd2 - {z2}| = {worst_npd}"
    );
    assert!(
        (cf_fit - cf_indep).abs() > 3.0 * TOL_B && (cf_fit - cf_decl).abs() > 3.0 * TOL_B,
        "degenerate fixture"
    );
    assert!(
        worst_npde < TOL_B,
        "row 2 NPDE not decorrelated at the fitted ρ: worst |npde − {cf_fit}| = {worst_npde} \
         (independent rows give {cf_indep}, the declared ρ gives {cf_decl})"
    );
    assert!(
        worst_npd < TOL_A,
        "row 2 NPD moved off its marginal {z2}: worst {worst_npd}"
    );
}

/// The unchanged half (#1733): a `FIX` `block_sigma` whose parameter set carries
/// the declared ρ (or none, falling back to it), on rows that are never paired.
/// Its `R` is diagonal, so the shared correlated draw takes its per-row fast
/// path, which consumes the RNG stream in the same order as the old scalar loop.
/// The scores are the ones `d43afca9` produced before the draw moved, pinned to
/// the bit. A draw that perturbed the stream (an extra normal, or rows drawn in
/// another order) would move them. Turning the fast path off does not: the
/// eigen square root of a diagonal `R` was measured to give the same bits, so
/// the fast path is a cost saving, not a numerical one.
#[test]
fn a_fixed_block_sigma_scores_exactly_as_before_the_shared_draw() {
    let root = env!("CARGO_MANIFEST_DIR");
    let model = crate::parser::model_parser::parse_model_file(std::path::Path::new(&format!(
        "{root}/examples/correlated_residual_combined.ferx"
    )))
    .expect("parse");
    let pop = crate::io::datareader::read_nonmem_csv(
        std::path::Path::new(&format!("{root}/data/correlated_residual_combined.csv")),
        None,
        None,
    )
    .expect("read data");
    let mut params = model.default_params.clone();
    assert_eq!(model.residual_correlations.len(), 1);
    let pinned: [f64; 4] = [
        0.15096921547331393,
        0.12566134687610309,
        -0.2793190341322942,
        0.2793190341322942,
    ];
    for corr in [model.residual_correlations.clone(), vec![]] {
        params.residual_correlations = corr;
        let out = compute_npde_npd(&model, &pop, &params, 200, Some(1)).expect("npde");
        let got = &out[0].npde;
        eprintln!("subject 1 npde = {got:?}");
        assert_eq!(got.len(), pinned.len());
        for (j, (&g, &p)) in got.iter().zip(pinned.iter()).enumerate() {
            assert_eq!(g.to_bits(), p.to_bits(), "row {j}: {g} vs pinned {p}");
        }
    }
}

/// A FREM subject keeps the per-row draw, because its covariate pseudo-rows sit
/// outside `R` (the same gate `simulate()` uses). That per-row draw must still
/// read the parameter set's ρ: arm A's PK row, plus one FREMTYPE row, scored at
/// the fitted ρ = −0.9 against a declared 0.5. Both consumers of the per-row
/// fallback are checked, NPDE and `simulate()`, each naming itself on failure.
#[test]
fn a_frem_subject_draws_its_pk_rows_at_the_fitted_correlation() {
    let mut model = parse(ARM_A);
    // FREMTYPE 100 → (θ TVCL, η ETA_CL); the covariate σ reuses slot 1 (ADD).
    model.frem_config = Some(crate::types::FremConfig {
        fremtype_to_indices: HashMap::from([(100u16, (0usize, 0usize))]),
        covariate_sigma_index: 1,
    });
    let mut params = model.default_params.clone();
    params.residual_correlations = with_rho(&model, -0.9);
    let mut subject = arm_a_subject();
    subject.obs_times.push(0.0);
    subject.observations.push(0.0);
    subject.obs_cmts.push(1);
    subject.cens.push(0);
    subject.fremtype.push(100);
    let f = ipreds(&model, &subject, &params);
    assert!(f[0].is_finite() && f[0] > 1.0, "f = {f:?}");
    let var = |rho: f64| 0.04 * f[0] * f[0] + 1.0 + 2.0 * rho * 0.2 * f[0];
    let (sd_fit, sd_decl) = (var(-0.9).sqrt(), var(0.5).sqrt());
    subject.observations = vec![f[0] + sd_fit, f[1]];
    let cf_decl = sd_fit / sd_decl;
    assert!((1.0 - cf_decl).abs() > 3.0 * TOL_A, "degenerate fixture");
    let pop = population(subject, vec![]);

    let mut worst = 0.0f64;
    for seed in 1..=SEEDS {
        let out = compute_npde_npd(&model, &pop, &params, NSIM, Some(seed)).expect("npde");
        let npd = out[0].npd[0];
        assert!(npd.is_finite(), "seed {seed}: npd = {npd}");
        worst = worst.max((npd - 1.0).abs());
    }
    assert!(
        worst < TOL_A,
        "npde: the FREM subject's PK row is not drawn at the fitted ρ: worst |npd − 1| = \
         {worst} (declared ρ gives {cf_decl})"
    );

    let sims = crate::api::simulate_with_seed(&model, &pop, &params, NSIM, 1733).expect("simulate");
    let pk: Vec<f64> = sims
        .iter()
        .filter(|r| r.time == 1.0)
        .map(|r| r.outcome.continuous_value() - r.ipred)
        .collect();
    assert_eq!(pk.len(), NSIM);
    assert!(
        pk.iter().all(|e| e.is_finite()),
        "simulate: non-finite residual"
    );
    let sd = (pk.iter().map(|e| e * e).sum::<f64>() / NSIM as f64).sqrt();
    // sd of 2000 normal draws is within ~1.6% of σ at 1 se; 8% is 5 se, while the
    // declared ρ is 124% away (sd_decl / sd_fit = 2.24).
    assert!(
        (sd / sd_fit - 1.0).abs() < 0.08,
        "simulate: the FREM subject's PK residual sd {sd} is not the fitted {sd_fit} \
         (the declared ρ gives {sd_decl})"
    );
}

const NSIM: usize = 2000;
const SEEDS: u64 = 10;
const TOL_A: f64 = 0.15;
const TOL_B: f64 = 0.25;
