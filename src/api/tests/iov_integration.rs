use super::fit;
use super::simulate_with_seed;
use super::{simulate_with_options, SimOutcome};
use crate::types::*;

use std::collections::HashMap;

// ── Model ────────────────────────────────────────────────────────────────
fn make_iov_model() -> CompiledModel {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
    let default_params = ModelParameters {
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: Some(omega_iov),
        kappa_fixed: vec![false],
        mixture: None,
    };
    CompiledModel {
        covariate_model: None,
        name: "iov_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        // pk_param_fn: eta[0]=BSV for CL, eta[1]=KAPPA_CL (appended by IOV path)
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL = TVCL * exp(ETA_CL + KAPPA_CL)
                p.values[1] = theta[1]; // V
                p
            },
        ),
        n_theta: 2,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 1,
        kappa_names: vec!["KAPPA_CL".into()],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: vec!["ETA_CL".into()],
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params,
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: vec![false],
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: None,
        pk_indices: vec![0, 1],
        eta_map: vec![0],
        pk_idx_f64: vec![0.0, 1.0],
        sel_flat: vec![1.0, 0.0],
        ode_spec: None,
        dose_attr_map: Default::default(),
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        referenced_covariates: Vec::new(),
        gradient_method: GradientMethod::Fd,
        parse_warnings: Vec::new(),
        has_conditional_eta_params: false,
        eta_param_info: Vec::new(),
        theta_transform: Vec::new(),
        #[cfg(feature = "nn")]
        covariate_nns: Vec::new(),
        scaling: ScalingSpec::None,
        log_transform: false,
        dv_pre_logged: false,
        derived_exprs: vec![],
        output_columns: vec![],
        #[cfg(feature = "survival")]
        endpoints: std::collections::HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    }
}

// ── Population ───────────────────────────────────────────────────────────
// 4 subjects, each with 2 occasions.  Times 1–3 = occasion 1, times 4–6 = occ 2.
// Observations are plausible IV-bolus concentrations (dose=100).
fn make_iov_population() -> Population {
    let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let occasions = vec![1u32, 1, 1, 2, 2, 2];
    let dose_occ = vec![1u32, 2]; // two dose rows: one per occasion
    let subject_data: &[(&str, Vec<f64>)] = &[
        ("S1", vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0]),
        ("S2", vec![40.0, 32.0, 24.0, 38.0, 29.0, 22.0]),
        ("S3", vec![33.0, 25.0, 19.0, 31.0, 24.0, 18.0]),
        ("S4", vec![42.0, 33.0, 25.0, 39.0, 30.0, 23.0]),
    ];

    let subjects: Vec<Subject> = subject_data
        .iter()
        .map(|(id, obs)| Subject {
            id: id.to_string(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: obs.clone(),
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 6],
            occasions: occasions.clone(),
            obs_l2: Vec::new(),
            dose_occasions: dose_occ.clone(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        })
        .collect();
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn fast_opts(method: EstimationMethod, optimizer: Optimizer, mu_referencing: bool) -> FitOptions {
    FitOptions {
        method,
        methods: Vec::new(),
        outer_maxiter: 60,
        outer_gtol: 1e-3,
        inner_maxiter: 50,
        inner_tol: 1e-4,
        run_covariance_step: false,
        interaction: method == EstimationMethod::FoceI,
        mu_referencing,
        optimizer,
        lbfgs_memory: 5,
        verbose: false,
        ..FitOptions::default()
    }
}

// ── Helper ───────────────────────────────────────────────────────────────
fn assert_iov_fit_ok(result: &FitResult) {
    assert!(result.ofv.is_finite(), "OFV must be finite");
    assert!(result.omega_iov.is_some(), "omega_iov must be populated");
    let iov_diag = result.omega_iov.as_ref().unwrap()[(0, 0)];
    assert!(
        iov_diag > 0.0,
        "omega_iov diagonal must be positive, got {iov_diag}"
    );
    assert_eq!(result.kappa_names, vec!["KAPPA_CL"], "kappa name mismatch");
    assert_eq!(result.ebe_kappas.len(), 4, "expected kappas for 4 subjects");
    for (i, subj_kappas) in result.ebe_kappas.iter().enumerate() {
        assert_eq!(subj_kappas.len(), 2, "subject {i} should have 2 occasions");
    }
}

// ── Tests: FOCE + all outer optimizers ───────────────────────────────────

#[test]
fn test_iov_foce_bobyqa() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
fn test_iov_foce_auto_reports_resolved_optimizer() {
    // `optimizer = auto` (the default) must resolve per-model and the
    // FitResult must report the concrete pick so the output records what
    // actually ran (#490). This IOV model is outside the analytic-gradient
    // scope, so `auto` lands on bobyqa.
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Auto, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
    let resolved = Optimizer::Auto.resolve_auto(&model, false);
    assert_eq!(result.optimizer, format!("auto ({})", resolved.label()));
    assert_eq!(resolved, Optimizer::Bobyqa);
}

#[test]
fn test_iov_foce_slsqp() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Slsqp, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
fn test_iov_foce_lbfgs() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Lbfgs, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
fn test_iov_foce_nlopt_lbfgs() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::NloptLbfgs, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
fn test_iov_foce_mma() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Mma, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
fn test_iov_foce_bfgs() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

/// End-to-end: the **same IOV model written two ways** — closed-form `one_cpt_iv`
/// vs a user `[odes]` twin (`d/dt central = -(CL/V)·central`, `y = central/V`) —
/// must converge to the **same optimum** (OFV + θ) on the same data. This exercises
/// the full ODE IOV pipeline (analytic inner stacked-η gradient + analytic outer
/// block-Ω gradient) through a real fit, not just per-evaluation gradient parity
/// (#439 ODE IOV).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow (~1 min: two full IOV fits, one over RK45); opt in with --features slow-tests"
)]
fn test_iov_ode_fit_matches_analytical_twin() {
    use crate::parser::model_parser::parse_model_string;
    const ANALYTICAL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
"#;
    const ODE: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  ode(states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method     = foce
  iov_column = OCC
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;
    let ana = parse_model_string(ANALYTICAL).expect("parse analytical IOV");
    let ode = parse_model_string(ODE).expect("parse ODE IOV");
    assert!(
        crate::sens::ode_provider::ode_iov_supported(&ode),
        "ODE twin must be ODE-IOV-provider supported (analytic inner+outer)"
    );
    let pop = make_iov_population();
    // Use the **default** optimizer (`Auto`) — exactly what a real user gets.
    // For this model `Auto` resolves to the gradient-based NLopt L-BFGS
    // (analytic outer gradient is available on both the closed-form and ODE
    // IOV paths). The previous explicit built-in `Bfgs` was the problem: from
    // the far default start (TVCL=5.0, true ≈0.28) its line-search overshoots
    // TVCL to the lower bound on the *ODE* path and stalls in a worse basin
    // (OFV ~165 vs ~148), while the closed-form twin survives the same
    // trajectory. The objective and analytic sensitivities are correct —
    // seeding built-in BFGS at the optimum converges fine — so it was
    // optimizer basin-capture, not an ODE-IOV gradient defect. See #439/#486.
    assert_eq!(
        Optimizer::Auto.resolve_auto(&ode, false),
        Optimizer::NloptLbfgs,
        "ODE IOV twin should resolve `auto` to the analytic-gradient L-BFGS"
    );
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Auto, false);
    opts.outer_maxiter = 200;

    let ra = fit(&ana, &pop, &ana.default_params, &opts).expect("analytical fit");
    let ro = fit(&ode, &pop, &ode.default_params, &opts).expect("ODE fit");
    assert_iov_fit_ok(&ra);
    assert_iov_fit_ok(&ro);

    // Same structural model → same minimum (small slack for ODE integration vs
    // closed form, and the differing inner gradient routes).
    approx::assert_relative_eq!(ra.ofv, ro.ofv, max_relative = 1e-3, epsilon = 0.5);
    for k in 0..2 {
        approx::assert_relative_eq!(
            ra.theta[k],
            ro.theta[k],
            max_relative = 0.02,
            epsilon = 1e-2
        );
    }
    let iov_a = ra.omega_iov.as_ref().unwrap()[(0, 0)];
    let iov_o = ro.omega_iov.as_ref().unwrap()[(0, 0)];
    approx::assert_relative_eq!(iov_a, iov_o, max_relative = 0.10, epsilon = 1e-3);
}

// ── Tests: FOCEI ─────────────────────────────────────────────────────────

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_iov_focei_bobyqa() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert_iov_fit_ok(&result);
}

// ── Tests: mu-referencing ─────────────────────────────────────────────────

#[test]
fn test_iov_foce_mu_referencing_on() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, true);
    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fit with mu_referencing should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_iov_focei_mu_referencing_on() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fit with mu_referencing should succeed");
    assert_iov_fit_ok(&result);
}

// ── Tests: GN and GN_Hybrid ───────────────────────────────────────────────

#[test]
fn test_iov_gn() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params, &opts).expect("GN fit should succeed");
    assert_iov_fit_ok(&result);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_iov_gn_hybrid() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("GN hybrid fit should succeed");
    assert_iov_fit_ok(&result);
}

// ── Test: SAEM + IOV now supported (Step 11) ─────────────────────────────

#[test]
fn test_iov_saem_succeeds() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params, &opts);
    assert!(
        result.is_ok(),
        "SAEM with IOV must succeed, got: {:?}",
        result.err()
    );
    let fr = result.unwrap();
    assert!(
        fr.ofv.is_finite(),
        "SAEM IOV OFV must be finite, got {}",
        fr.ofv
    );
    assert!(
        fr.omega_iov.is_some(),
        "omega_iov must be present in result"
    );
}

// ── Test: SAEM in a chained methods sequence + IOV succeeds ──────────────
#[test]
fn test_iov_saem_in_methods_chain_succeeds() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Foce];
    let result = fit(&model, &pop, &model.default_params, &opts);
    assert!(
        result.is_ok(),
        "SAEM in methods chain with IOV must succeed, got: {:?}",
        result.err()
    );
    let fr = result.unwrap();
    assert!(
        fr.ofv.is_finite(),
        "chained SAEM+FOCE IOV OFV must be finite"
    );
    assert!(
        fr.omega_iov.is_some(),
        "omega_iov must survive the FOCE polishing step in a chained run"
    );
}

// ── Test: IMP in a chained methods sequence + IOV exercises the IS IOV path ─
// `methods = [foce, imp]` on a kappa-bearing model drives the importance-
// sampling marginal-likelihood step through its IOV branch
// (`obs_nll_iov_fixed_kappa`, `compute_posterior_hessian`,
// `subject_is_estimate`, `build_proposals`). κ is held at its EBE, so the
// reported −2LL is a partial marginal (see `KappaTreatment::FixedAtMode`).
#[test]
fn test_iov_imp_chain_runs_importance_sampling() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    opts.methods = vec![EstimationMethod::Foce, EstimationMethod::Imp];
    opts.imp_samples = 200; // keep the per-subject sampling cheap
    opts.imp_seed = Some(42); // deterministic proposal draws
                              // The IS IOV branch is the evaluation-only path; the estimating IMP
                              // M-step does not yet support IOV (refused up front).
    opts.imp_eval_only = true;
    let result = fit(&model, &pop, &model.default_params, &opts);
    assert!(
        result.is_ok(),
        "FOCE→IMP chain with IOV must succeed, got: {:?}",
        result.err()
    );
    let fr = result.unwrap();
    let is = fr
        .importance_sampling
        .as_ref()
        .expect("importance_sampling result must be populated by the IMP stage");
    assert!(
        is.minus2_log_likelihood.is_finite(),
        "IS marginal −2LL must be finite, got {}",
        is.minus2_log_likelihood
    );
    assert!(
        is.mc_standard_error.is_finite() && is.mc_standard_error >= 0.0,
        "IS Monte-Carlo SE must be finite and non-negative, got {}",
        is.mc_standard_error
    );
    assert_eq!(is.n_samples, 200, "n_samples should echo the IS budget");
    assert!(
        fr.omega_iov.is_some(),
        "omega_iov must survive into the IS-augmented result"
    );
}

// ── Test: trust-region optimizer + IOV must return Err ────────────────────
// trust_region.rs currently passes `&[]` for kappas to pop_nll, which would
// silently route the OFV through the non-IOV path. Guard at api.rs blocks
// that before any wrong number escapes.
#[test]
fn test_iov_trust_region_returns_err() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
    let result = fit(&model, &pop, &model.default_params, &opts);
    assert!(
        result.is_err(),
        "trust_region with IOV must return an error"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("trust_region") && msg.contains("IOV"),
        "error message should mention trust_region and IOV, got: {msg}"
    );
}

// When a model has kappa declarations but the dataset carries no occasion
// labels, `fit()` must return an error and `check_model_data` must surface
// E_IOV_MISSING_OCC — rather than silently ignoring the kappas.
#[test]
fn test_iov_missing_occ_returns_err() {
    let model = make_iov_model();
    // Population built without occasion labels (empty `occasions` vectors).
    let mut pop = make_iov_population();
    for subj in &mut pop.subjects {
        subj.occasions.clear();
    }
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params, &opts);
    assert!(
        result.is_err(),
        "IOV model without occasion labels must error"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("iov_column") || msg.contains("OCC") || msg.contains("occasion"),
        "error message should mention the missing occasion column, got: {msg}"
    );
}

#[test]
fn test_check_model_data_flags_missing_occ() {
    use crate::diagnostics::Severity;
    let model = make_iov_model();
    let mut pop = make_iov_population();
    for subj in &mut pop.subjects {
        subj.occasions.clear();
    }
    let diags = super::check_model_data(&model, &pop);
    let d = diags
        .iter()
        .find(|d| d.code == "E_IOV_MISSING_OCC")
        .expect("expected E_IOV_MISSING_OCC diagnostic");
    assert_eq!(d.severity, Severity::Error);
    assert!(d.message.contains("iov_column") || d.message.contains("kappa"));
    assert_eq!(d.block.as_deref(), Some("fit_options"));
}

// A model-side occasion rule (`iov_occasion`) supplies the labels at fit time,
// so the E_IOV_MISSING_OCC data check must NOT fire even with empty occasions.
#[test]
fn test_iov_occasion_rule_suppresses_missing_occ_check() {
    let model = make_iov_model();
    let mut pop = make_iov_population();
    for subj in &mut pop.subjects {
        subj.occasions.clear();
        subj.dose_occasions.clear();
    }
    let rule = IovOccasionRule::TimeWindows(vec![3.5]);
    let diags = super::check_model_data_rule(&model, &pop, &rule);
    assert!(
        !diags.iter().any(|d| d.code == "E_IOV_MISSING_OCC"),
        "iov_occasion rule should suppress the missing-occasion error"
    );
    // Column rule (default) still flags the empty occasions.
    let diags = super::check_model_data_rule(&model, &pop, &IovOccasionRule::Column);
    assert!(diags.iter().any(|d| d.code == "E_IOV_MISSING_OCC"));
}

// TimeWindows breakpoints bucket obs and doses by (raw) time: `time(3.5)`
// splits the make_iov_population timeline (obs 1..6, doses at 0.0 and 3.5)
// into occasion 0 (t < 3.5) and occasion 1 (t ≥ 3.5).
#[test]
fn test_apply_iov_occasion_time_windows() {
    let mut pop = make_iov_population();
    let mut warnings = Vec::new();
    super::apply_iov_occasion_rule(
        &mut pop,
        &IovOccasionRule::TimeWindows(vec![3.5]),
        false,
        &mut warnings,
    );
    assert!(warnings.is_empty(), "no column set → no override warning");
    for s in &pop.subjects {
        assert_eq!(s.occasions, vec![0u32, 0, 0, 1, 1, 1]);
        assert_eq!(s.dose_occasions, vec![0u32, 1]);
    }
}

// PerDose opens a new occasion at each administration; an observation takes
// the occasion of the most recent dose (dose-before-obs at equal TIME). For
// make_iov_population (doses at 0.0 and 3.5) this yields the same partition
// as time(3.5): obs 1..3 → occasion 0, obs 4..6 → occasion 1.
#[test]
fn test_apply_iov_occasion_per_dose() {
    let mut pop = make_iov_population();
    let mut warnings = Vec::new();
    super::apply_iov_occasion_rule(&mut pop, &IovOccasionRule::PerDose, false, &mut warnings);
    for s in &pop.subjects {
        assert_eq!(s.occasions, vec![0u32, 0, 0, 1, 1, 1]);
        assert_eq!(s.dose_occasions, vec![0u32, 1]);
    }
}

// When both iov_column and a DSL rule are set, the DSL rule wins and a single
// override warning is emitted.
#[test]
fn test_apply_iov_occasion_coexistence_warns() {
    let mut pop = make_iov_population();
    let mut warnings = Vec::new();
    super::apply_iov_occasion_rule(
        &mut pop,
        &IovOccasionRule::PerDose,
        /* had_column = */ true,
        &mut warnings,
    );
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("iov_column"));
}

// NONMEM-equivalence by proxy: the column path is already NONMEM-validated
// (tests/warfarin_iov_nonmem.rs). A DSL `time(3.5)` rule that reproduces the
// same occasion partition must produce a bit-identical fit (same OFV, same
// per-occasion kappa structure) as reading the occasions from the column.
#[test]
fn test_iov_occasion_dsl_matches_column_fit() {
    let model = make_iov_model();
    let pop = make_iov_population(); // occasions [1,1,1,2,2,2] from the column

    let opts_col = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let r_col =
        fit(&model, &pop, &model.default_params, &opts_col).expect("column fit should succeed");

    // Same population, but derive occasions from the model instead. time(3.5)
    // reproduces the column partition (relabeled 0/1 vs 1/2 — grouping is by
    // label identity, so the estimation is identical).
    let mut opts_dsl = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    opts_dsl.iov_occasion = IovOccasionRule::TimeWindows(vec![3.5]);
    let r_dsl = fit(&model, &pop, &model.default_params, &opts_dsl)
        .expect("DSL-occasion fit should succeed");

    // The two fits share model + population and differ only in how each
    // subject's occasion partition is derived; both produce the identical
    // partition (relabelled 0/1 vs 1/2 — see above), and the FOCE pipeline is
    // bit-reproducible regardless of the thread schedule: the per-subject
    // objective reduces in a fixed subject order (collect-then-serial-sum,
    // #703) and the inner EBE loop is order-preserving. So the two OFVs are
    // *bit-identical*, not merely close.
    //
    // Assert that exactly. col and dsl run in the same process on the same CPU,
    // so this compares two values that shared every FP op sequence — it is a
    // 0-vs-0 check that cannot be perturbed by parallel-reduction order or CPU
    // load (the class of flakiness a loose tolerance would only paper over).
    // If it ever fails, do NOT loosen it: a real regression has broken
    // partition equivalence (independently pinned by the `apply_iov_occasion_*`
    // derivation tests above) or leaked nondeterminism into estimation. The
    // sibling `test_iov_ode_fit_matches_analytical_twin` compares two *different*
    // numeric engines (ODE vs closed form) and rightly uses a basin tolerance;
    // this test does not.
    assert!(
        r_col.ofv.is_finite() && r_dsl.ofv.is_finite(),
        "both OFVs must be finite (col={}, dsl={})",
        r_col.ofv,
        r_dsl.ofv,
    );
    assert_eq!(
        r_col.ofv.to_bits(),
        r_dsl.ofv.to_bits(),
        "DSL-occasion OFV {} must be bit-identical to column OFV {} \
         (same partition + deterministic pipeline)",
        r_dsl.ofv,
        r_col.ofv,
    );
    // The same bit-identity thesis extends to the per-occasion kappa EBEs: same
    // subject data, same theta/Omega trajectory, same first-occ->g0/second-occ->g1
    // group map, so every kappa component must match bit-for-bit. This is a
    // *stricter* guard than the OFV check — a summed OFV can round-trip to the
    // same bits while a per-kappa ULP diverges — so pin it exactly too.
    assert_eq!(r_col.ebe_kappas.len(), r_dsl.ebe_kappas.len());
    for (subj, (a, b)) in r_col.ebe_kappas.iter().zip(&r_dsl.ebe_kappas).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "subject {} must have the same number of occasions",
            subj,
        );
        for (k, (ka, kb)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                ka.len(),
                kb.len(),
                "subject {} occasion {} must have the same kappa dimension",
                subj,
                k,
            );
            for (j, (va, vb)) in ka.iter().zip(kb.iter()).enumerate() {
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "subject {} occasion {} kappa[{}]: DSL EBE {} must be \
                     bit-identical to column EBE {}",
                    subj,
                    k,
                    j,
                    vb,
                    va,
                );
            }
        }
    }
}

// #757: TimeWindows must bucket observations on the internal (monotonic)
// event clock, not the raw user clock. A reset-stacked crossover restarts
// the user TIME each period; if the derivation read `obs_raw_times` the
// second period's early samples would fold back into occasion 0 and split
// from their (internal-clock) doses.
#[test]
fn test_apply_iov_occasion_time_windows_uses_internal_clock() {
    let mut pop = make_iov_population();
    // Raw times that restart mid-series, as a period-2 reset would; the
    // internal obs_times stay [1..6]. The rule must ignore these.
    for s in &mut pop.subjects {
        s.obs_raw_times = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    }
    let mut warnings = Vec::new();
    super::apply_iov_occasion_rule(
        &mut pop,
        &IovOccasionRule::TimeWindows(vec![3.5]),
        false,
        &mut warnings,
    );
    for s in &pop.subjects {
        // Internal clock [1..6] with break at 3.5 → [0,0,0,1,1,1]. Had the raw
        // times been used, the last three (all < 3.5) would be occasion 0.
        assert_eq!(s.occasions, vec![0u32, 0, 0, 1, 1, 1]);
    }
}

// #757: two doses administered at the same time (e.g. a simultaneous
// depot + central bolus) must share one occasion — the earlier one must not
// be stranded in an empty, observation-less occasion.
#[test]
fn test_apply_iov_occasion_per_dose_cotimed_doses_share_occasion() {
    let mut pop = make_iov_population();
    for s in &mut pop.subjects {
        s.doses = vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(0.0, 50.0, 2, 0.0, false, 0.0), // co-timed
            DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
        ];
    }
    let mut warnings = Vec::new();
    super::apply_iov_occasion_rule(&mut pop, &IovOccasionRule::PerDose, false, &mut warnings);
    for s in &pop.subjects {
        // Two co-timed doses at t=0 → occasion 0; dose at 3.5 → occasion 1.
        assert_eq!(s.dose_occasions, vec![0u32, 0, 1]);
        // Observation partition unchanged: 1..3 → occ 0, 4..6 → occ 1.
        assert_eq!(s.occasions, vec![0u32, 0, 0, 1, 1, 1]);
    }
}

// #757: a derived rule that collapses every subject to a single occasion
// leaves kappa confounded with BSV — the same unidentifiability the column
// path guards with E_IOV_MISSING_OCC. fit() must error rather than run.
#[test]
fn test_iov_occasion_degenerate_single_occasion_errors() {
    let model = make_iov_model();
    let pop = make_iov_population();
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    // Breakpoint past the last observation (max 6.0) → one occasion for all.
    opts.iov_occasion = IovOccasionRule::TimeWindows(vec![100.0]);
    let res = fit(&model, &pop, &model.default_params, &opts);
    let msg = res.expect_err("single-occasion derivation must error");
    assert!(
        msg.contains("single occasion") || msg.contains("unidentifiable"),
        "unexpected error: {msg}"
    );
}

// #757: `iov_occasion` with no kappa in the model is a no-op — it must warn
// (so the user knows it had no effect) and skip the derivation entirely.
#[test]
fn test_iov_occasion_without_kappa_warns() {
    let mut model = make_iov_model();
    // Strip IOV so the rule has nothing to feed.
    model.n_kappa = 0;
    model.kappa_names.clear();
    model.default_params.omega_iov = None;
    model.default_params.kappa_fixed.clear();
    let pop = make_iov_population();
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    opts.iov_occasion = IovOccasionRule::PerDose;
    let res = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
    assert!(
        res.warnings
            .iter()
            .any(|w| w.contains("iov_occasion") && w.contains("no effect")),
        "expected a no-op warning, got: {:?}",
        res.warnings
    );
}

// #757: `run_model_*` mirrors the model-side occasion derivation onto the
// returned population so sdtab keeps its OCC column. Exercises the helper the
// CLI wrapper calls without a full fit.
#[test]
fn test_derive_output_occasions_matches_rule() {
    let model = make_iov_model();
    let empty = |p: &mut Population| {
        for s in &mut p.subjects {
            s.occasions.clear();
            s.dose_occasions.clear();
        }
    };

    // Derived rule → occasions written for output.
    let mut pop = make_iov_population();
    empty(&mut pop);
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    opts.iov_occasion = IovOccasionRule::PerDose;
    super::derive_output_occasions(&model, &opts, &mut pop);
    for s in &pop.subjects {
        assert_eq!(s.occasions, vec![0u32, 0, 0, 1, 1, 1]);
    }

    // Column rule (default) → no-op, occasions stay empty.
    let mut pop_col = make_iov_population();
    empty(&mut pop_col);
    let opts_col = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    super::derive_output_occasions(&model, &opts_col, &mut pop_col);
    assert!(pop_col.subjects.iter().all(|s| s.occasions.is_empty()));

    // Rule set but no kappa → no-op.
    let mut model0 = make_iov_model();
    model0.n_kappa = 0;
    let mut pop0 = make_iov_population();
    empty(&mut pop0);
    super::derive_output_occasions(&model0, &opts, &mut pop0);
    assert!(pop0.subjects.iter().all(|s| s.occasions.is_empty()));
}

// `ferx check` must surface the same trust_region+IOV incompatibility that
// `fit()` rejects — without it, a model could report `valid: true` and then
// fail at fit time. `check_model_options` is the shared source of truth.
#[test]
fn test_check_model_options_flags_trust_region_iov() {
    let model = make_iov_model();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
    let diags = super::check_model_options(&model, &opts);
    let d = diags
        .iter()
        .find(|d| d.code == "E_OPTIMIZER_IOV")
        .expect("expected E_OPTIMIZER_IOV diagnostic");
    // Same wording fit() produces (regression against the extracted guard).
    assert!(d.message.contains("trust_region") && d.message.contains("IOV"));

    // A compatible optimizer produces no compatibility diagnostics.
    let ok_opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    assert!(super::check_model_options(&model, &ok_opts).is_empty());
}

// block_sigma correlated residual errors are wired into the Gaussian FOCE,
// FOCEI, SAEM, IMP, AGQ, and Laplace paths (AGQ/Laplace via `score_core`'s
// `corr_diag` branch, #251); the Gauss-Newton (gn / gn_hybrid) methods still
// use diagonal residual-error derivatives and must be rejected up front.
#[test]
fn test_check_model_options_block_sigma_rejects_unsupported_methods() {
    let mut model = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    }];

    // gn (Gauss-Newton) is rejected.
    let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, true);
    let diags = super::check_model_options(&model, &opts);
    let d = diags
        .iter()
        .find(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED")
        .expect("expected E_BLOCK_SIGMA_METHOD_UNSUPPORTED for gn");
    assert!(d.is_error() && d.message.contains("focei") && d.message.contains("gn"));

    // gn_hybrid is likewise rejected.
    let hybrid = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
    assert!(super::check_model_options(&model, &hybrid)
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"));

    // FOCE, FOCEI, SAEM, IMP, AGQ, Laplace, and VI are all accepted (no method
    // diagnostic). VI belongs here because its data term is `obs_nll_subject_grad`,
    // whose non-IOV path routes a dense `R` through the same full-FD fallback the
    // others use; only the closed-form σ maximizer is unavailable there, and that
    // falls back to Adam with a recorded reason rather than failing the fit.
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Laplace,
        EstimationMethod::Vi,
    ] {
        let opts = fast_opts(method, Optimizer::Bobyqa, false);
        assert!(
            !super::check_model_options(&model, &opts)
                .iter()
                .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"),
            "block_sigma must be accepted for {method:?}"
        );
    }

    model.n_kappa = 1;
    let saem = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
    assert!(super::check_model_options(&model, &saem)
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_IOV_UNSUPPORTED"));
}

// The block_sigma + SDE rejection is scoped to SAEM: the FOCE dense path adds
// the EKF process-noise `p_obs` into R and worked before SAEM support was
// added, so a FOCE-only chain must stay accepted. Regression guard against
// the over-broad method-agnostic rejection (PR #557 review).
#[test]
fn test_check_model_options_block_sigma_sde_rejected_for_saem_not_foce() {
    let mut model = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    }];
    // is_sde() reads diffusion_theta_start; flag the model as an SDE/EKF model.
    model.diffusion_theta_start = Some(0);

    let saem = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
    assert!(super::check_model_options(&model, &saem)
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_SDE_UNSUPPORTED"));

    let foce = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    assert!(!super::check_model_options(&model, &foce)
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_SDE_UNSUPPORTED"));
}

// The block_sigma + TTE rejection is blanket (every method): the TTE Laplace
// path accumulates R diagonally and cannot carry the cross-endpoint
// covariance, so it must be rejected for FOCE as well as SAEM (PR #557).
#[cfg(feature = "survival")]
#[test]
fn test_check_model_options_block_sigma_tte_rejected_all_methods() {
    let mut model = crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Fd);
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.5,
    }];
    // has_tte() scans endpoints for a Tte variant; attach a trivial one.
    model.endpoints.insert(
        2,
        crate::types::EndpointLikelihood::Tte {
            hazard: crate::types::HazardSpec::Analytic {
                family: crate::types::HazardFamily::Exponential,
                param_fn: Box::new(|_theta, _eta, _cov| vec![0.1]),
            },
            recurrence: crate::types::TteRecurrence::Single,
            hazard_covariates: Vec::new(),
        },
    );

    for method in [EstimationMethod::Saem, EstimationMethod::Foce] {
        let opts = fast_opts(method, Optimizer::Bobyqa, false);
        assert!(
            super::check_model_options(&model, &opts)
                .iter()
                .any(|d| d.code == "E_BLOCK_SIGMA_TTE_UNSUPPORTED"),
            "TTE + block_sigma must be rejected for {method:?}"
        );
    }
}

#[test]
fn test_saem_accepts_block_sigma_cross_endpoint_fit() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
    )
    .expect("cross-endpoint block_sigma ODE model parses");

    let mut subjects = Vec::new();
    for (id, dose_amt, obs) in [
        ("1", 100.0, vec![17.0, 8.0, 15.0, 7.0]),
        ("2", 80.0, vec![14.0, 6.8, 12.0, 6.0]),
    ] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, dose_amt, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 1.0, 2.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 2, 1, 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mut opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
    opts.saem_n_exploration = 1;
    opts.saem_n_convergence = 0;
    opts.saem_n_mh_steps = 1;
    opts.saem_adapt_interval = 10;
    opts.saem_omega_burnin = 0;
    opts.saem_seed = Some(548);

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("SAEM fit with cross-endpoint block_sigma should succeed");
    assert!(result.ofv.is_finite(), "SAEM OFV must be finite");
}

/// A VI block that a later *estimating* stage has overtaken says so; one followed only by
/// an evaluator does not.
///
/// `FitResult.vi` survives the rest of a chain on purpose — `methods = [vi, laplace]` with
/// `agq_eval_only` is the recommended way to turn VI's lower bound into a real `−2 log L`,
/// and the variational covariance is the only per-subject covariance in the result. But on
/// `methods = [vi, focei]` the reported θ/Ω/σ and subject diagnostics come from FOCEI while
/// every number under `vi` still describes VI's parameter point. That is worth reporting,
/// not worth reporting *silently*.
#[test]
fn test_vi_block_is_marked_when_a_later_stage_re_estimates() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("1-cpt IV fixture parses");

    let mut subjects = Vec::new();
    for (id, dose_amt, obs) in [("1", 100.0, vec![8.0, 6.0]), ("2", 80.0, vec![6.5, 4.9])] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, dose_amt, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 2],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let base = |methods: Vec<EstimationMethod>| {
        let mut o = fast_opts(EstimationMethod::Vi, Optimizer::Bobyqa, false);
        o.methods = methods;
        o.outer_maxiter = 2;
        o.vi_iters = 5;
        o.vi_mc_samples = 2;
        o
    };

    // VI alone: nothing supersedes it.
    let solo = fit(&model, &population, &model.default_params, &base(vec![]))
        .expect("VI-only fit succeeds");
    assert_eq!(solo.vi.as_ref().expect("VI block").superseded_by, None);

    // A trailing evaluator reads the fit out without moving it, so the VI block still
    // describes the reported parameters.
    let mut eval_opts = base(vec![EstimationMethod::Vi, EstimationMethod::Laplace]);
    eval_opts.agq_eval_only = true;
    let evaluated = fit(&model, &population, &model.default_params, &eval_opts)
        .expect("vi + agq_eval_only laplace fit succeeds");
    assert_eq!(
        evaluated.vi.as_ref().expect("VI block").superseded_by,
        None,
        "an evaluation-only stage must not count as superseding VI"
    );

    // A later estimating stage does move the parameters, and is named.
    let chained = fit(
        &model,
        &population,
        &model.default_params,
        &base(vec![EstimationMethod::Vi, EstimationMethod::FoceI]),
    )
    .expect("vi + focei fit succeeds");
    assert_eq!(
        chained
            .vi
            .as_ref()
            .expect("VI block")
            .superseded_by
            .as_deref(),
        Some("FOCEI")
    );
    assert!(
        chained
            .warnings
            .iter()
            .any(|w| w.contains("still describes the parameter point VI ended on")),
        "the staleness must be stated in prose too, got {:?}",
        chained.warnings
    );
}

/// VI must accept a correlated `$SIGMA` block too, and fall back to Adam on `σ` rather
/// than failing.
///
/// Its data term is `obs_nll_subject_grad`, whose non-IOV path routes a dense `R` through
/// the same full-FD fallback FOCE and SAEM use, so the objective and its θ/σ gradient are
/// correct here. What a dense `R` does remove is the *closed-form* `σ` maximizer — `σ` is
/// no longer inside a scalar variance with a stationary point — and `closed_form_sigma_support`
/// declines it with a reason and steps `σ` by Adam instead. The option docs promise exactly
/// that fallback, so rejecting the configuration at check time contradicted them.
#[test]
fn test_vi_accepts_block_sigma_cross_endpoint_fit() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
    )
    .expect("cross-endpoint block_sigma ODE model parses");

    // Check time first: this is the guard the docs contradicted.
    let check_opts = fast_opts(EstimationMethod::Vi, Optimizer::Bobyqa, false);
    assert!(
        !super::check_model_options(&model, &check_opts)
            .iter()
            .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"),
        "block_sigma + vi must not be rejected at check time"
    );

    let mut subjects = Vec::new();
    for (id, dose_amt, obs) in [
        ("1", 100.0, vec![17.0, 8.0, 15.0, 7.0]),
        ("2", 80.0, vec![14.0, 6.8, 12.0, 6.0]),
    ] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, dose_amt, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 1.0, 2.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 2, 1, 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mut opts = fast_opts(EstimationMethod::Vi, Optimizer::Bobyqa, false);
    opts.vi_iters = 5;
    opts.vi_mc_samples = 2;

    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("VI fit with cross-endpoint block_sigma should succeed");
    let vi = result.vi.as_ref().expect("VI block present");
    assert!(
        vi.neg_two_elbo.is_finite(),
        "dense-R data term must produce a finite -2*ELBO"
    );
    // And the promised fallback actually fired, with its reason recorded.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("vi_sigma_update") && w.contains("Adam")),
        "expected the closed-form sigma fallback warning, got {:?}",
        result.warnings
    );
}

// Importance sampling must accept a correlated $SIGMA block: the weights use
// the dense residual covariance (obs_nll_subject_into) and the Student-t
// proposal precision is built from R⁻¹ (compute_posterior_hessian dense
// branch). A short evaluation-only run must produce a finite marginal.
#[test]
fn test_imp_accepts_block_sigma_cross_endpoint() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0) FIX
  theta TVV(10.0, 1.0, 100.0) FIX
  omega ETA_CL ~ 0.04 FIX
  block_sigma (PROP_ERR_UNBOUND, PROP_ERR_TOTAL) = [
    0.04,
    0.01, 0.09
  ]

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -CL/V * central

[scaling]
  y[CMT=1] = 2.0 * central / V
  y[CMT=2] = central / V

[error_model]
  CMT=1: DV ~ proportional(PROP_ERR_TOTAL)
  CMT=2: DV ~ proportional(PROP_ERR_UNBOUND)
",
    )
    .expect("cross-endpoint block_sigma ODE model parses");

    let mut subjects = Vec::new();
    for (id, dose_amt, obs) in [
        ("1", 100.0, vec![17.0, 8.0, 15.0, 7.0]),
        ("2", 80.0, vec![14.0, 6.8, 12.0, 6.0]),
    ] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, dose_amt, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 1.0, 2.0, 2.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 2, 1, 2],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // No method diagnostic for IMP + block_sigma.
    let mut opts = fast_opts(EstimationMethod::Imp, Optimizer::Bobyqa, false);
    assert!(!super::check_model_options(&model, &opts)
        .iter()
        .any(|d| d.code == "E_BLOCK_SIGMA_METHOD_UNSUPPORTED"));

    // Evaluation-only marginal at the initial params (fast + deterministic).
    opts.imp_eval_only = true;
    opts.imp_samples = 200;
    opts.imp_iterations = 1;
    opts.imp_seed = Some(7);
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("IMP eval with cross-endpoint block_sigma should succeed");
    assert!(result.ofv.is_finite(), "IMP block_sigma OFV must be finite");
}

// Regression: the SAEM final OFV (computed via `2·pop_nll` with
// `interaction = true`) must include the block_sigma cross-endpoint
// covariance. The interaction Laplace accumulator builds R diagonally, so
// before #557 SAEM silently dropped the correlation and reported the
// diagonal OFV (~4 units low here) instead of the FOCE/NONMEM value. At
// fully-fixed parameters both estimators evaluate the same conditional
// likelihood, so their OFVs must agree.
#[test]
fn test_saem_block_sigma_ofv_matches_foce() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    // Single-endpoint combined error with a fixed correlated $SIGMA block —
    // the docs validation model (NONMEM FOCE OFV 18.7158, ferx FOCE 18.7274).
    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0) FIX
  theta TVV(10.0, 0.1, 100.0) FIX
  omega ETA_CL ~ 0.04 FIX
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
",
    )
    .expect("fixed-param block_sigma combined model parses");

    let mut subjects = Vec::new();
    for (id, obs) in [
        ("1", vec![9.5, 8.0, 6.2]),
        ("2", vec![10.5, 7.4, 5.6]),
        ("3", vec![8.8, 7.9, 6.7]),
    ] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 3],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    // FOCE evaluation (all params fixed → maxiter 0 just evaluates the OFV).
    let mut foce_opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    foce_opts.outer_maxiter = 0;
    let foce_ofv = fit(&model, &population, &model.default_params, &foce_opts)
        .expect("FOCE block_sigma evaluation succeeds")
        .ofv;

    // SAEM evaluation at the same fixed params.
    let mut saem_opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
    saem_opts.saem_n_exploration = 2;
    saem_opts.saem_n_convergence = 0;
    saem_opts.saem_n_mh_steps = 1;
    saem_opts.saem_adapt_interval = 10;
    saem_opts.saem_omega_burnin = 0;
    saem_opts.saem_seed = Some(12345);
    let saem_ofv = fit(&model, &population, &model.default_params, &saem_opts)
        .expect("SAEM block_sigma evaluation succeeds")
        .ofv;

    // Params are fixed, so the EBEs (hence the conditional OFV) are identical
    // regardless of estimator; the dense covariance must be carried by both.
    assert!(
        (saem_ofv - foce_ofv).abs() < 1e-3,
        "SAEM OFV {saem_ofv} must match FOCE OFV {foce_ofv} (block_sigma correlation dropped?)"
    );
}

// FOCEI with a correlated $SIGMA block must (a) evaluate to a finite OFV and
// (b) genuinely apply the interaction correction — i.e. differ from the FOCE
// (non-interaction) OFV on an f-dependent (combined) error model. Before #616
// the only correlated objective was the FOCE marginal, so the interaction term
// was unavailable (FOCEI was rejected up front). Fixed params → maxiter 0 just
// evaluates the OFV at the typical values, so this stays fast.
#[test]
fn test_focei_block_sigma_ofv_finite_and_applies_interaction() {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 10.0) FIX
  theta TVV(10.0, 0.1, 100.0) FIX
  omega ETA_CL ~ 0.04 FIX
  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
",
    )
    .expect("fixed-param block_sigma combined model parses");

    let mut subjects = Vec::new();
    for (id, obs) in [
        ("1", vec![9.5, 8.0, 6.2]),
        ("2", vec![10.5, 7.4, 5.6]),
        ("3", vec![8.8, 7.9, 6.7]),
    ] {
        subjects.push(Subject {
            id: id.into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 4.0],
            obs_raw_times: Vec::new(),
            observations: obs,
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 3],
            occasions: Vec::new(),
            obs_l2: Vec::new(),
            dose_occasions: Vec::new(),
            reset_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        });
    }
    let population = Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let eval = |method, interaction| {
        let mut opts = fast_opts(method, Optimizer::Bobyqa, false);
        opts.outer_maxiter = 0;
        opts.interaction = interaction;
        fit(&model, &population, &model.default_params, &opts)
            .expect("block_sigma evaluation succeeds")
            .ofv
    };

    let focei_ofv = eval(EstimationMethod::FoceI, true);
    let foce_ofv = eval(EstimationMethod::Foce, false);

    assert!(
        focei_ofv.is_finite(),
        "FOCEI block_sigma OFV must be finite"
    );
    // The interaction term (½·log|H̃| with the dense ∂R/∂η curvature) shifts
    // the OFV away from the non-interaction marginal on a combined model.
    assert!(
        (focei_ofv - foce_ofv).abs() > 1e-4,
        "FOCEI OFV {focei_ofv} must differ from FOCE OFV {foce_ofv} (interaction dropped?)"
    );
}

/// Shared covariate-selected free/total `block_sigma` model + paired-row
/// subject for the `simulate()` cross-branch correlation tests (#672).
/// Mirrors `SELECTED_BLOCK_SIGMA_MODEL` in `parser/model_parser.rs`, which
/// pins the same off-diagonal on the estimation side.
fn block_sigma_selected_model_and_population() -> (CompiledModel, Population) {
    use crate::parser::model_parser::parse_model_string;
    use std::collections::HashMap;

    let model = parse_model_string(
        r"
[parameters]
  theta TVCL(5.0, 0.1, 50.0) FIX
  theta TVV(50.0, 5.0, 500.0) FIX
  omega ETA_CL ~ 0.09 FIX
  block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.01, 0.005, 0.09] FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (FREE == 0) {
    DV ~ proportional(PROP_TOTAL)
  } else {
    DV ~ proportional(PROP_UNBOUND)
  }

[covariates]
  FREE continuous
",
    )
    .expect("selected-error block_sigma model parses");

    // Two paired rows at the same subject time (total vs. unbound), the
    // exact fixture `test_selected_error_block_sigma_cross_branch_covariance`
    // uses to pin the estimation-side dense `R`.
    let subject = Subject {
        id: "S1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 1.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0, 0.0],
        obs_cmts: vec![1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: vec![
            HashMap::from([("FREE".to_string(), 0.0)]),
            HashMap::from([("FREE".to_string(), 1.0)]),
        ],
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0, 0],
        occasions: vec![1, 1],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![subject],
        covariate_names: vec!["FREE".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    (model, population)
}

// #672: `simulate()` must draw the two paired rows (same subject time,
// different `FREE` branch) from the dense correlated `R`, not independent
// per-row normals — otherwise a VPC of the correlated endpoints understates
// the residual covariance. Large-N: the empirical correlation of the two
// rows' residuals (value − ipred) must recover the fixed `rho` implied by
// `block_sigma`'s off-diagonal.
#[test]
fn test_simulate_recovers_block_sigma_cross_branch_correlation() {
    let (model, population) = block_sigma_selected_model_and_population();

    let n_sim = 20_000;
    let results = simulate_with_seed(&model, &population, &model.default_params, n_sim, 42);
    assert_eq!(results.len(), 2 * n_sim);

    let mut resid_total = Vec::with_capacity(n_sim);
    let mut resid_unbound = Vec::with_capacity(n_sim);
    for pair in results.chunks(2) {
        resid_total.push(pair[0].outcome.continuous_value() - pair[0].ipred);
        resid_unbound.push(pair[1].outcome.continuous_value() - pair[1].ipred);
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (m0, m1) = (mean(&resid_total), mean(&resid_unbound));
    let n = n_sim as f64;
    let cov = resid_total
        .iter()
        .zip(&resid_unbound)
        .map(|(a, b)| (a - m0) * (b - m1))
        .sum::<f64>()
        / n;
    let var0 = resid_total.iter().map(|a| (a - m0).powi(2)).sum::<f64>() / n;
    let var1 = resid_unbound.iter().map(|a| (a - m1).powi(2)).sum::<f64>() / n;
    let empirical_rho = cov / (var0.sqrt() * var1.sqrt());

    let sd_i = 0.01f64.sqrt();
    let sd_j = 0.09f64.sqrt();
    let expected_rho = 0.005 / (sd_i * sd_j);

    assert!(
        (empirical_rho - expected_rho).abs() < 0.03,
        "empirical rho {empirical_rho} should recover the specified block_sigma \
             rho {expected_rho} (simulate() ignoring the correlation would give ~0)"
    );
}

// Degenerate case (#672): a `residual_correlations` entry with `rho == 0.0`
// makes `R` exactly diagonal, so the new dense Cholesky draw
// (`emit_correlated_residual_rows`) must reproduce the untouched
// independent per-row draw (`emit_subject_rows`'s scalar branch, taken when
// `model.residual_correlations` is empty) — same seed, same per-row RNG
// draw order in both paths. Not bit-for-bit: `compute_r_matrix_with_correlations`'s
// diagonal goes through `variance_at_scaled`'s `((f·f)·σ)·σ` association
// whenever `correlations` is non-empty (even at rho=0), versus `variance_at`'s
// `(f·σ)·(f·σ)` when it's empty — a pre-existing, documented ~1-ULP
// reassociation difference (see `residual_error::compute_r_matrix_with_correlations`),
// not a regression from this draw path.
#[test]
fn test_simulate_zero_rho_matches_diagonal_draw_path() {
    let (mut model, population) = block_sigma_selected_model_and_population();
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 0.0,
    }];
    let dense_path = simulate_with_seed(&model, &population, &model.default_params, 200, 7);

    model.residual_correlations.clear();
    let scalar_path = simulate_with_seed(&model, &population, &model.default_params, 200, 7);

    assert_eq!(dense_path.len(), scalar_path.len());
    for (a, b) in dense_path.iter().zip(scalar_path.iter()) {
        let (va, vb) = (a.outcome.continuous_value(), b.outcome.continuous_value());
        assert!(
            (va - vb).abs() < 1e-9 * va.abs().max(1.0),
            "rho=0 dense-R draw {va} must match the independent scalar draw {vb} \
                 to within floating-point rounding"
        );
    }
}

// #672 regression: a fixed `block_sigma` with a perfect cross-endpoint
// correlation (`rho = 1`) makes each paired subject's R singular. The
// original Cholesky draw panicked the whole simulation; the PSD
// symmetric-eigen square root must instead produce a valid,
// perfectly-correlated draw (empirical rho ≈ 1).
#[test]
fn test_simulate_singular_rho_one_does_not_panic() {
    let (mut model, population) = block_sigma_selected_model_and_population();
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 1,
        rho: 1.0,
    }];

    let n_sim = 20_000;
    let results = simulate_with_seed(&model, &population, &model.default_params, n_sim, 11);
    assert_eq!(results.len(), 2 * n_sim);

    let mut resid_total = Vec::with_capacity(n_sim);
    let mut resid_unbound = Vec::with_capacity(n_sim);
    for pair in results.chunks(2) {
        resid_total.push(pair[0].outcome.continuous_value() - pair[0].ipred);
        resid_unbound.push(pair[1].outcome.continuous_value() - pair[1].ipred);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (m0, m1) = (mean(&resid_total), mean(&resid_unbound));
    let n = n_sim as f64;
    let cov = resid_total
        .iter()
        .zip(&resid_unbound)
        .map(|(a, b)| (a - m0) * (b - m1))
        .sum::<f64>()
        / n;
    let var0 = resid_total.iter().map(|a| (a - m0).powi(2)).sum::<f64>() / n;
    let var1 = resid_unbound.iter().map(|a| (a - m1).powi(2)).sum::<f64>() / n;
    let empirical_rho = cov / (var0.sqrt() * var1.sqrt());
    assert!(
        (empirical_rho - 1.0).abs() < 0.01,
        "rho=1 singular block_sigma must draw perfectly-correlated residuals \
             (no panic), got empirical rho {empirical_rho}"
    );
}

// Cover the custom-magnitude (`Some(mult)`) R-build branch and the
// `ruv_scale != 1.0` scaling inside `emit_correlated_residual_rows`. The
// model-driven simulate() path can't reach these for a supported
// `block_sigma` model (block_sigma + iiv_on_ruv is rejected at fit), so call
// the helper directly with a hand-built per-observation multiplier and a
// non-unit scale and assert it draws both rows without panicking.
#[test]
fn test_emit_correlated_residual_rows_magnitude_and_scale_paths() {
    use rand::SeedableRng;
    let (model, population) = block_sigma_selected_model_and_population();
    let subject = &population.subjects[0];
    let ipreds = vec![10.0_f64, 10.0];
    // [obs][sigma-slot] multiplier (all ones here — the point is to exercise
    // the `_scaled` builder and the `ruv_scale` multiply, not a specific
    // magnitude value).
    let mult = vec![vec![1.0_f64, 1.0], vec![1.0, 1.0]];
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let mut results = Vec::new();
    super::emit_correlated_residual_rows(
        &model,
        subject,
        &model.default_params,
        &ipreds,
        2.0, // ruv_scale != 1.0
        Some(&mult),
        0,
        0,
        normal,
        &mut rng,
        &mut results,
    );
    assert_eq!(results.len(), 2);
    assert!(results
        .iter()
        .all(|r| r.outcome.continuous_value().is_finite()));
}

#[test]
fn test_check_model_options_block_sigma_rejects_iov() {
    let mut model = make_iov_model();
    model.residual_correlations = vec![crate::types::ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 0,
        rho: 0.5,
    }];

    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let diags = super::check_model_options(&model, &opts);
    let d = diags
        .iter()
        .find(|d| d.code == "E_BLOCK_SIGMA_IOV_UNSUPPORTED")
        .expect("expected E_BLOCK_SIGMA_IOV_UNSUPPORTED for block_sigma + IOV");

    assert!(d.is_error());
    assert!(d.message.contains("inter-occasion"));
    assert_eq!(d.block.as_deref(), Some("fit_options"));
}

// IMPMAP does not yet support IOV; `ferx check` must flag it up front rather
// than letting the fit fail at runtime (review finding #3).
#[test]
fn test_check_model_options_flags_impmap_iov() {
    let model = make_iov_model();
    let opts = fast_opts(EstimationMethod::Impmap, Optimizer::Bobyqa, false);
    let diags = super::check_model_options(&model, &opts);
    let d = diags
        .iter()
        .find(|d| d.code == "E_IMPMAP_IOV_UNSUPPORTED")
        .expect("expected E_IMPMAP_IOV_UNSUPPORTED diagnostic");
    assert!(d.is_error() && d.message.contains("inter-occasion"));
}

// The Enzyme AD path was retired, so explicitly requesting AD must error
// rather than silently running a different method. `auto`/`fd` must still pass.
#[test]
fn ad_requested_errors_now_that_ad_is_retired() {
    let model = make_iov_model();
    let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);

    opts.gradient_method = crate::types::GradientMethod::Ad;
    let diags = super::check_model_options(&model, &opts);
    assert!(
        diags
            .iter()
            .any(|d| d.code == "E_AD_RETIRED" && d.is_error()),
        "explicit gradient_method=ad must error now that AD is retired, got: {diags:?}"
    );

    for gm in [
        crate::types::GradientMethod::Auto,
        crate::types::GradientMethod::Fd,
    ] {
        opts.gradient_method = gm;
        assert!(
            !super::check_model_options(&model, &opts)
                .iter()
                .any(|d| d.code == "E_AD_RETIRED"),
            "gradient_method={gm:?} must not trigger E_AD_RETIRED"
        );
    }
}

/// Regression for review finding #5 (IOV + compartments[i]).
///
/// When the model has a [derived] expression that references `compartments[i]`
/// and the fit has IOV subjects, `W_DERIVED_CMT_IOV_UNSUPPORTED` must be
/// emitted. The `predict_iov` path does not compute compartment states; the
/// per-subject `compartment_states` vec stays empty (`vec![]`), so any
/// `compartments[i]` reference evaluates to NaN. The warning makes this
/// explicit rather than silent.
#[test]
fn iov_with_compartments_derived_emits_unsupported_warning() {
    let mut model = make_iov_model();
    // Inject a derived expression that sets uses_compartments = true,
    // just like a parsed `[derived] cmt0 = compartments[0]` would.
    model.derived_exprs.push(DerivedExprSpec {
        name: "cmt0".into(),
        kind: DerivedKind::PerRow {
            eval: Box::new(|ctx| ctx.compartments.first().copied().unwrap_or(f64::NAN)),
        },
        uses_compartments: true,
    });
    let pop = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params.clone(), &opts).expect("fit must succeed");

    // Warning must be present.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_IOV_UNSUPPORTED")),
        "expected W_DERIVED_CMT_IOV_UNSUPPORTED warning; got: {:?}",
        result.warnings
    );
    // Compartment states for IOV subjects must be entirely empty (outer vec
    // is vec![], not vec![vec![]; n_obs]) — the predict_iov path never
    // populates them.
    for sr in &result.subjects {
        assert!(
            sr.compartment_states.is_empty(),
            "IOV subject {} must have empty compartment_states (len={}), \
                 got {}",
            sr.id,
            sr.ipred.len(),
            sr.compartment_states.len()
        );
    }
}

/// #400: an analytical oral model with a zero-order input into the depot
/// (an infusion into cmt 1) cannot have its compartment states expressed by
/// the superposition state helper (which models an oral infusion as a depot
/// bypass). So `compartment_states` is left empty (→ NaN compartments) and
/// `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL` makes that explicit, just
/// like the reset/IOV/TV cases. Predictions themselves stay exact.
#[test]
fn analytical_oral_depot_infusion_with_compartments_derived_emits_warning() {
    use crate::parser::model_parser::parse_full_model;
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#;
    let mut model = parse_full_model(src).expect("model parses").model;
    assert!(model.ode_spec.is_none(), "model must be analytical");
    // Inject a derived expression that references compartments[0] so the
    // warning is gated on (mirrors a parsed `[derived] cmt0 = compartments[0]`).
    model.derived_exprs.push(DerivedExprSpec {
        name: "cmt0".into(),
        kind: DerivedKind::PerRow {
            eval: Box::new(|ctx| ctx.compartments.first().copied().unwrap_or(f64::NAN)),
        },
        uses_compartments: true,
    });

    // One subject with an explicit zero-order infusion into the depot (cmt 1):
    // rate 25 over AMT/rate = 4 h, then first-order KA absorption.
    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 4.0, 8.0, 12.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.8, 1.4, 1.6, 0.9, 0.4],
        obs_cmts: vec![2; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let pop = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let result = fit(&model, &pop, &model.default_params.clone(), &opts).expect("fit must succeed");

    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL")),
        "expected W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL warning; got: {:?}",
        result.warnings
    );
    // Predictions must still be finite (the event-driven path computed them);
    // only the compartment states degrade.
    for sr in &result.subjects {
        assert!(
            sr.compartment_states.is_empty(),
            "depot-infusion subject must have empty compartment_states (got {})",
            sr.compartment_states.len()
        );
        assert!(
            sr.ipred.iter().all(|p| p.is_finite()),
            "predictions must be finite"
        );
    }
}

// ── #1019: simulating an IOV model from caller-rebuilt parameters ────────
//
// The R bridge rebuilds `ModelParameters` from a fit object; before #1019 it
// dropped `omega_iov`, and the per-occasion κ draw in `emit_subject_rows`
// unwrapped that `None` — a panic across the FFI boundary that took the R
// session's error handling with it. The contract is now enforced at every
// simulate entry point: a clean `Err` where one can be returned, a loud panic
// on the Vec-returning chokepoint.

/// `params` with the IOV block stripped — exactly what a fit-rebuilt
/// `ModelParameters` looked like before the fix.
fn iov_params_without_omega_iov(model: &CompiledModel) -> ModelParameters {
    let mut params = model.default_params.clone();
    params.omega_iov = None;
    params
}

#[test]
fn test_simulate_with_options_errs_when_omega_iov_missing() {
    let model = make_iov_model();
    let population = make_iov_population();
    let params = iov_params_without_omega_iov(&model);

    let err = simulate_with_options(
        &model,
        &population,
        &params,
        2,
        &crate::SimulateOptions {
            seed: Some(1),
            ..Default::default()
        },
    )
    .expect_err("an IOV model with no omega_iov must not simulate");
    assert!(
        err.contains("omega_iov") && err.contains("kappa"),
        "error must name the missing IOV covariance; got: {err}"
    );
}

#[test]
#[should_panic(expected = "carry no omega_iov")]
fn test_simulate_with_seed_panics_when_omega_iov_missing() {
    let model = make_iov_model();
    let population = make_iov_population();
    let params = iov_params_without_omega_iov(&model);
    // No Err channel on this entry point: fail loud rather than emit rows with
    // zero inter-occasion variability.
    let _ = simulate_with_seed(&model, &population, &params, 1, 1);
}

#[test]
fn test_simulate_from_fitted_params_carries_omega_iov() {
    let model = make_iov_model();
    let population = make_iov_population();
    let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
    let fit_result = fit(&model, &population, &model.default_params.clone(), &opts)
        .expect("IOV fit must succeed");

    // The supported way to rebuild parameters from a fit — the fix the R bridge
    // mirrors. It must carry the IOV covariance through, and simulate cleanly.
    let params =
        crate::estimation::uncertainty_samples::fitted_params_from_result(&fit_result, &model);
    assert!(
        params.omega_iov.is_some(),
        "fitted_params_from_result must thread omega_iov through for a kappa model"
    );

    let results = simulate_with_options(
        &model,
        &population,
        &params,
        2,
        &crate::SimulateOptions {
            seed: Some(7),
            ..Default::default()
        },
    )
    .expect("simulating from fitted params must succeed");
    assert_eq!(
        results.len(),
        2 * population.subjects.len() * population.subjects[0].obs_times.len()
    );
    assert!(results.iter().all(|r| {
        r.ipred.is_finite()
            && matches!(&r.outcome, SimOutcome::Continuous { value } if value.is_finite())
    }));
}
