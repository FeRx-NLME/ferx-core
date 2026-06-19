//! Integration tests for IIV on residual error (`iiv_on_ruv`, NONMEM
//! `Y = IPRED + EPS*EXP(ETA)`; issue #409).
//!
//! Tier-2 tests call the public `fit()` boundary but return immediately — either
//! with an `Err` (the validation reject) or after a couple of outer iterations
//! (no convergence loop) — so they stay fast and are compile-checked on every
//! PR / run nightly. One Tier-3 round-trip recovery test (gated behind
//! `slow-tests`) runs a full FOCEI fit to convergence on simulated data.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::Population;
use ferx_core::{
    fit, read_nonmem_csv, simulate_with_seed, EstimationMethod, FitOptions, SimulationResult,
};
use std::path::Path;

/// Warfarin oral 1-cpt model with a dedicated residual-error eta wired via
/// `iiv_on_ruv`. The `ETA_RUV` omega is the 4th declared (index 3) and is NOT
/// referenced by any individual parameter.
fn iiv_on_ruv_model() -> ferx_core::types::CompiledModel {
    let src = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.05

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV

[fit_options]
  method = focei
";
    parse_model_string(src).expect("iiv_on_ruv model must parse")
}

#[test]
fn iiv_on_ruv_parses_and_wires_eta() {
    let model = iiv_on_ruv_model();
    // ETA_RUV is the 4th declared omega → eta index 3.
    assert_eq!(model.residual_error_eta, Some(3));
    assert_eq!(model.n_eta, 4);
    assert!(model.eta_names.contains(&"ETA_RUV".to_string()));
    // It is not a structural/individual-parameter eta.
    assert!(!model.eta_param_info.iter().any(|e| e.eta_name == "ETA_RUV"));
}

#[test]
fn iiv_on_ruv_rejects_non_interaction_foce() {
    let model = iiv_on_ruv_model();
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Foce; // non-interaction
    opts.methods = vec![];
    let err = fit(&model, &population, &model.default_params, &opts)
        .expect_err("non-interaction FOCE with iiv_on_ruv must be rejected");
    assert!(
        err.contains("iiv_on_ruv") && err.to_lowercase().contains("interaction"),
        "unexpected error: {err}"
    );
}

#[test]
fn iiv_on_ruv_focei_runs_and_reports_extra_omega() {
    let model = iiv_on_ruv_model();
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.interaction = true;
    opts.outer_maxiter = 2; // a couple of outer iterations — Tier-2, no convergence loop
    opts.run_covariance_step = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("FOCEI fit with iiv_on_ruv must run");

    // The residual-error eta is reported like any other random effect.
    assert!(result.eta_names.contains(&"ETA_RUV".to_string()));
    assert_eq!(result.omega.nrows(), 4);
    assert_eq!(result.omega.ncols(), 4);
    // Its variance stays finite and positive across the (short) run.
    let idx = result
        .eta_names
        .iter()
        .position(|n| n == "ETA_RUV")
        .unwrap();
    assert!(
        result.omega[(idx, idx)].is_finite() && result.omega[(idx, idx)] > 0.0,
        "ETA_RUV variance must be finite/positive, got {}",
        result.omega[(idx, idx)]
    );
    assert!(result.ofv.is_finite(), "OFV must be finite");
}

#[test]
fn iiv_on_ruv_impmap_runs_and_reports_is_marginal() {
    // IMPMAP is the acceptance-target estimator (#409): it integrates the
    // residual eta out by Monte Carlo through `obs_nll_subject_into`, which
    // applies the exp(2·η_ruv) scaling per draw.
    let model = iiv_on_ruv_model();
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Impmap;
    opts.methods = vec![];
    opts.impmap_iterations = 2; // a couple of MCEM iterations — Tier-2
    opts.impmap_samples = 50;
    opts.impmap_averaging = 1;
    opts.impmap_seed = Some(7);
    opts.run_covariance_step = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("IMPMAP fit with iiv_on_ruv must run");

    assert_eq!(result.method, EstimationMethod::Impmap);
    assert!(result.eta_names.contains(&"ETA_RUV".to_string()));
    assert!(result.ofv.is_finite(), "IMPMAP OFV must be finite");
    let idx = result
        .eta_names
        .iter()
        .position(|n| n == "ETA_RUV")
        .unwrap();
    assert!(
        result.omega[(idx, idx)].is_finite() && result.omega[(idx, idx)] > 0.0,
        "ETA_RUV variance must stay finite/positive under IMPMAP, got {}",
        result.omega[(idx, idx)]
    );
    // If an IS −2logL is reported, it must be finite.
    if let Some(is) = result.importance_sampling.as_ref() {
        assert!(is.minus2_log_likelihood.is_finite());
    }
}

/// Build a refit population by overwriting each subject's observations with the
/// single-replicate simulated DVs (Gaussian rows, in observation order).
fn population_from_sim(template: &Population, sims: &[SimulationResult]) -> Population {
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        let vals: Vec<f64> = sims
            .iter()
            .filter(|r| r.id == subj.id)
            .map(|r| r.outcome.continuous_value())
            .collect();
        assert_eq!(
            vals.len(),
            subj.observations.len(),
            "simulated obs count must match template for subject {}",
            subj.id
        );
        subj.observations = vals;
    }
    pop
}

/// Warfarin model with a *strong* residual-error IIV (ETA_RUV ~ 0.30), used for
/// the recovery test. A large variance keeps the random effect identifiable on a
/// modest design (a small `iiv_on_ruv` variance is genuinely weakly identified —
/// the marginal correctly shrinks it toward zero when the data carries little
/// per-subject residual-scale signal).
fn iiv_on_ruv_strong_model() -> ferx_core::types::CompiledModel {
    let src = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  omega ETA_RUV ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
  iiv_on_ruv = ETA_RUV

[fit_options]
  method = focei
";
    parse_model_string(src).expect("strong iiv_on_ruv model must parse")
}

/// Replicate every subject `copies` times with distinct IDs, for more subjects
/// (tighter identification of the residual-error IIV).
fn replicate_population(base: &Population, copies: usize) -> Population {
    let mut pop = base.clone();
    pop.subjects.clear();
    for c in 0..copies {
        for subj in &base.subjects {
            let mut s = subj.clone();
            s.id = format!("{}_{c}", subj.id);
            pop.subjects.push(s);
        }
    }
    pop
}

/// Slow round-trip self-consistency check (Tier-3): simulate from a model with a
/// known, strong IIV-on-RUV variance (0.30) over a replicated design, and confirm
/// FOCEI recovers it to within a factor of ~2. This is the strongest in-repo
/// numerical validation of the estimation path — it demonstrates the FOCEI
/// marginal (with the residual-eta `c̃` curvature) genuinely identifies the
/// random effect rather than collapsing it. A direct NONMEM 7.5.1 cross-check on
/// this exact setup lives in `nonmem_anchor/` (ferx vs NONMEM FOCEI agree to
/// ΔOFV = 0.017; ETA_RUV variance within 2.4%).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iiv_on_ruv_focei_recovers_simulated_variance() {
    const TRUE_RUV_VAR: f64 = 0.30;

    let model = iiv_on_ruv_strong_model();
    let design = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin design must load");
    let design = replicate_population(&design, 5); // ~50 subjects

    // Simulate one replicate at the model's declared params (ETA_RUV ~ 0.30).
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, 20240619);
    let population = population_from_sim(&design, &sims);

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.interaction = true;
    opts.run_covariance_step = false;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("FOCEI fit on simulated iiv_on_ruv data must converge");

    let idx = result
        .eta_names
        .iter()
        .position(|n| n == "ETA_RUV")
        .expect("ETA_RUV reported");
    let est = result.omega[(idx, idx)];
    // Recover within a factor of ~2 (residual-error IIV shrinks somewhat under
    // FOCEI); must be clearly non-zero (not collapsed).
    assert!(
        est.is_finite() && est > TRUE_RUV_VAR / 2.0 && est < TRUE_RUV_VAR * 2.0,
        "recovered ETA_RUV variance {est} should be near the true {TRUE_RUV_VAR}"
    );
}
