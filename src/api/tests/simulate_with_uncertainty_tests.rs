//! End-to-end smoke tests for `simulate_with_uncertainty`. The parameter
//! sampler itself is exercised in `estimation::uncertainty_samples::tests`;
//! these tests verify the wiring: row count, draw index range, and SIR
//! pool reuse.

use super::*;
use crate::estimation::uncertainty_samples::UncertaintyMethod;
use nalgebra::DMatrix;
use std::collections::HashMap;

fn tiny_model() -> CompiledModel {
    let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
    let default_params = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };
    CompiledModel {
        residual_serial: None,
        priors: Vec::new(),
        prior_from_fit: None,
        covariate_model: None,
        name: "uncertainty_smoke".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            },
        ),
        n_theta: 2,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: Vec::new(),
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: vec!["ETA_CL".into()],
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: crate::types::IndivParamPartials::empty(),
        default_params,
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        covariate_mu_refs: Vec::new(),
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
        kappa_param_types: Vec::new(),
        theta_transform: Vec::new(),
        theta_eta_linked: Vec::new(),
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

/// #735: a `one_cpt_transit` + `TIME` model carrying a `lagtime=` mapping now AUTO-ROUTES
/// to its ODE twin instead of being rejected. Before #735 the lagtime form declined the
/// twin, so the `TIME` switch had nothing to route to and was rejected up front; now the
/// twin is built (carrying the lagtime), so the `TIME` / lagtime subjects are served by it.
#[test]
fn transit_with_time_and_lagtime_autoroutes() {
    use crate::parser::model_parser::parse_model_string;
    const TRANSIT_TIME_LAG: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVCL_LATE(7.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMTT(1.0, 0.05, 24.0)
  theta TVN(3.0, 0.0, 30.0)
  theta TVLAG(0.3, 0.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  if (TIME > 12.0) {
    CL = TVCL_LATE * exp(ETA_CL)
  } else {
    CL = TVCL * exp(ETA_CL)
  }
  V   = TVV
  MTT = TVMTT
  NTR = TVN
  LAGTIME = TVLAG
[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, lagtime=LAGTIME)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"#;
    let model = parse_model_string(TRANSIT_TIME_LAG).expect("parse transit+TIME+lag");
    assert_eq!(
        model.pk_model,
        crate::types::PkModel::OneCptTransit,
        "the closed-form shorthand stays OneCptTransit; the twin serves TIME/lagtime subjects"
    );
    assert!(
        crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
        "fixture must use the TIME built-in"
    );
    assert!(
        model.absorption_ode_equivalent.is_some(),
        "#735: a lagtime= transit model now carries an ODE twin (auto-routes)"
    );
    // No longer rejected — the twin handles both the TIME switch and the lagtime.
    assert!(
        check_absorption_closed_form_support(&model, &tiny_population()).is_none(),
        "transit + TIME + lagtime is now served via the twin, not rejected up front"
    );
}

// ---- #741: time-varying covariate on a survival hazard is rejected ----

/// A standalone TTE model with a covariate (`CRCL`) on the hazard via the PH
/// `loghr` term, so the parser records `CRCL` in the endpoint's
/// `hazard_covariates`. Used by the `check_survival_tv_covariates` tests.
#[cfg(feature = "survival")]
fn parse_tte_hazard_cov_model() -> CompiledModel {
    use crate::parser::model_parser::parse_model_string;
    const M: &str = r#"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta TVBETA(0.1, -5.0, 5.0)
  omega ETA_LAMBDA ~ 0.09
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
  loghr  = TVBETA * CRCL
[fit_options]
  method = focei
"#;
    parse_model_string(M).expect("TTE + covariate-on-hazard model parses")
}

/// One-subject population whose per-observation covariate snapshots are `snaps`
/// (a covariate that differs across them is time-varying). `obs_records` are left
/// empty — `check_survival_tv_covariates` inspects only covariate snapshots and
/// the model endpoints.
#[cfg(feature = "survival")]
fn tte_cov_snapshot_pop(id: &str, snaps: Vec<HashMap<String, f64>>) -> Population {
    let n = snaps.len();
    let base = snaps.first().cloned().unwrap_or_default();
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: id.to_string(),
        doses: Vec::new(),
        obs_times: vec![0.0; n],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; n],
        obs_cmts: vec![2; n],
        covariates: base,
        dose_covariates: Vec::new(),
        obs_covariates: snaps,
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: Vec::new(),
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["CRCL".to_string(), "WT".to_string()],
        dv_column: "DV".to_string(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Binary model whose `logit` references covariate `X`, so the parser records `X`
/// in the endpoint's `lp_covariates` (the categorical analogue of `hazard_covariates`).
#[cfg(feature = "survival")]
fn parse_binary_lp_cov_model() -> CompiledModel {
    use crate::parser::model_parser::parse_model_string;
    const M: &str = r#"
[parameters]
  theta TH0(0.0, -10.0, 10.0)
  theta THX(0.0, -10.0, 10.0)
[binary_model]
  cmt   = 3
  logit = TH0 + THX * X
[fit_options]
  method = focei
"#;
    parse_model_string(M).expect("binary + covariate-in-logit model parses")
}

/// The one refusal `check_survival_tv_covariates` returns for this input, asserted to carry
/// `E_ENDPOINT_TV_COVARIATE` on the endpoint's own `block` (#1774); returns its message.
/// Mutation — key the code or the block on anything but the endpoint arm (e.g. `event_model`
/// for all four) and the binary / CTMM callers die naming the block.
#[cfg(feature = "survival")]
pub(super) fn tv_refusal(model: &CompiledModel, pop: &Population, block: &str) -> String {
    let diags = check_survival_tv_covariates(model, pop);
    assert_eq!(diags.len(), 1, "one refusal: {diags:?}");
    let d = &diags[0];
    assert!(d.is_error(), "{d:?}");
    assert_eq!(d.code, "E_ENDPOINT_TV_COVARIATE", "{d:?}");
    assert_eq!(d.block.as_deref(), Some(block), "{d:?}");
    d.message.clone()
}

/// #741 analogue for the binary endpoint: a time-varying covariate in the `logit`
/// predictor is rejected at fit setup — it would otherwise be silently frozen at the
/// baseline snapshot (the bug the PR-807 review caught; the `types.rs` doc had claimed
/// this was guarded when it was not).
#[cfg(feature = "survival")]
#[test]
fn binary_logit_time_varying_covariate_is_rejected() {
    let model = parse_binary_lp_cov_model();
    let pop = tte_cov_snapshot_pop(
        "S1",
        vec![
            HashMap::from([("X".to_string(), 1.0)]),
            HashMap::from([("X".to_string(), 2.0)]),
        ],
    );
    let msg = tv_refusal(&model, &pop, "binary_model");
    assert_eq!(
        msg,
        "[binary_model] on CMT=3: the logit predictor references covariate `X`, which is \
         time-varying for subject `S1`. Time-varying covariates on the predictor are not yet \
         supported — it is evaluated at the baseline covariate value, so the fit would silently \
         use the wrong log-odds (#741). Use a time-constant (baseline) covariate for now.",
        "the whole text (#1774)"
    );
    assert!(msg.contains("[binary_model]"), "must name the block: {msg}");
    assert!(msg.contains('X'), "must name the covariate: {msg}");
    assert!(msg.contains("#741"), "must cite the issue: {msg}");
    // A time-constant X is accepted.
    let pop_ok = tte_cov_snapshot_pop(
        "S1",
        vec![
            HashMap::from([("X".to_string(), 1.0)]),
            HashMap::from([("X".to_string(), 1.0)]),
        ],
    );
    assert!(
        check_survival_tv_covariates(&model, &pop_ok).is_empty(),
        "a constant logit covariate must not be rejected"
    );
}

#[cfg(feature = "survival")]
#[test]
fn analytic_hazard_time_varying_covariate_is_rejected() {
    let model = parse_tte_hazard_cov_model();
    let pop = tte_cov_snapshot_pop(
        "S1",
        vec![
            HashMap::from([("CRCL".to_string(), 100.0)]),
            HashMap::from([("CRCL".to_string(), 60.0)]),
        ],
    );
    let msg = tv_refusal(&model, &pop, "event_model");
    assert_eq!(
        msg,
        "[event_model] on CMT=2: the survival hazard references covariate `CRCL`, which is \
         time-varying for subject `S1`. Time-varying covariates on a hazard are not yet \
         supported — the analytic hazard is evaluated at the baseline covariate value, so the \
         fit would silently use the wrong hazard (#741). Use a time-constant (baseline) \
         covariate on the hazard for now.",
        "the whole text (#1774)"
    );
    assert!(msg.contains("CRCL"), "must name the covariate: {msg}");
    assert!(msg.contains("S1"), "must name the subject: {msg}");
    assert!(msg.contains("#741"), "must cite the issue: {msg}");
}

#[cfg(feature = "survival")]
#[test]
fn analytic_hazard_constant_covariate_is_accepted() {
    let model = parse_tte_hazard_cov_model();
    let pop = tte_cov_snapshot_pop(
        "S1",
        vec![
            HashMap::from([("CRCL".to_string(), 100.0)]),
            HashMap::from([("CRCL".to_string(), 100.0)]),
        ],
    );
    assert!(
        check_survival_tv_covariates(&model, &pop).is_empty(),
        "a constant hazard covariate must not be rejected"
    );
}

#[cfg(feature = "survival")]
#[test]
fn analytic_hazard_ignores_tv_covariate_it_does_not_reference() {
    // WT is time-varying but the hazard references only CRCL (held constant). A
    // time-varying covariate used solely by an unrelated (e.g. PK) part of a joint
    // model must not trigger a false rejection — the analytic hazard never reads it.
    let model = parse_tte_hazard_cov_model();
    let pop = tte_cov_snapshot_pop(
        "S1",
        vec![
            HashMap::from([("CRCL".to_string(), 100.0), ("WT".to_string(), 70.0)]),
            HashMap::from([("CRCL".to_string(), 100.0), ("WT".to_string(), 80.0)]),
        ],
    );
    assert!(
        check_survival_tv_covariates(&model, &pop).is_empty(),
        "a time-varying covariate the hazard does not reference must not be rejected"
    );
}

#[cfg(feature = "survival")]
#[test]
fn ode_accumulated_hazard_rejects_any_time_varying_covariate() {
    // The survival ODE solve freezes the whole PK-parameter vector at t=0, so ANY
    // time-varying covariate corrupts the drug-driven hazard — even one not listed in
    // `hazard_covariates` (empty for the ODE variant).
    let mut model = parse_tte_hazard_cov_model();
    model.endpoints.insert(
        2,
        crate::types::EndpointLikelihood::Tte {
            hazard: crate::types::HazardSpec::OdeAccumulated { chz_state: 0 },
            recurrence: crate::types::TteRecurrence::Single,
            hazard_covariates: Vec::new(),
        },
    );
    let pop = tte_cov_snapshot_pop(
        "S9",
        vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 90.0)]),
        ],
    );
    let msg = tv_refusal(&model, &pop, "event_model");
    assert_eq!(
        msg,
        "[event_model] on CMT=2: the ODE-accumulated (joint PK-TTE) hazard is integrated with \
         covariates frozen at their t=0 baseline, but covariate `WT` is time-varying for \
         subject `S9`. Time-varying covariates on a drug-driven hazard are not yet supported \
         (#741). Hold the covariate constant within each subject for now.",
        "the whole text (#1774)"
    );
    assert!(msg.contains("WT"), "must name the covariate: {msg}");
    assert!(msg.contains("#741"), "must cite the issue: {msg}");
}

/// The CTMM text (#759): a time-varying covariate on a transition intensity is refused on
/// `markov_model` with the same code as the three survival texts above (#1774), and a
/// constant one is accepted. No test pinned this text before; mutation — delete any of
/// its sentences and the `assert_eq!` dies.
#[cfg(feature = "markov")]
#[test]
fn ctmm_intensity_time_varying_covariate_is_rejected() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
        "[parameters]\n  theta LQ01(-0.7, -6.0, 3.0)\n  theta LQ10(-1.2, -6.0, 3.0)\n\
         [markov_model]\n  type   = ctmm\n  cmt    = 2\n  states = [awake=0, asleep=1]\n  \
         transition awake  -> asleep = exp(LQ01) * WT / 70\n  \
         transition asleep -> awake  = exp(LQ10)\n",
    )
    .expect("CTMM + covariate-on-intensity model parses");
    let snaps = |wt2: f64| {
        vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), wt2)]),
        ]
    };
    let msg = tv_refusal(
        &model,
        &tte_cov_snapshot_pop("S3", snaps(90.0)),
        "markov_model",
    );
    assert_eq!(
        msg,
        "[markov_model] on CMT=2: a transition intensity references covariate `WT`, which is \
         time-varying for subject `S3`. Time-varying covariates on the generator are not yet \
         supported — the generator is evaluated at the baseline covariate value \
         (time-homogeneous CTMM), so the fit would silently use the wrong intensity (#741). A \
         drug-driven Q(t) is Phase 6; use a time-constant (baseline) covariate for now."
    );
    assert!(
        check_survival_tv_covariates(&model, &tte_cov_snapshot_pop("S3", snaps(70.0))).is_empty(),
        "a constant intensity covariate must not be rejected"
    );
}

fn tiny_population() -> Population {
    let obs_times = vec![1.0, 2.0, 3.0];
    let subjects: Vec<Subject> = (0..2)
        .map(|i| Subject {
            obs_rgroup: Vec::new(),
            id: format!("S{}", i + 1),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: obs_times.clone(),
            obs_raw_times: Vec::new(),
            observations: vec![30.0, 22.0, 16.0],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            obs_l2: Vec::new(),
            dose_occasions: vec![1],
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
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

/// Build a synthetic `FitResult` carrying the fitted theta/Omega/Sigma
/// from `template` plus a small identity covariance in packed log-space.
/// Avoids invoking `fit()` (slow) while still exercising the full
/// `simulate_with_uncertainty` wiring.
pub(super) fn synthetic_fit(template: &ModelParameters) -> FitResult {
    let n_packed = crate::estimation::parameterization::packed_len(template);
    let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
    FitResult {
        ofv_data: 0.0,
        ofv_prior: 0.0,
        prior_summary: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        se_residual_correlations: None,
        covariate_relations: Vec::new(),
        restored_from_checkpoint: false,
        method: EstimationMethod::FoceI,
        method_chain: vec![EstimationMethod::FoceI],
        method_wall_times_secs: vec![0.0],
        covariance_wall_time_secs: 0.0,
        converged: true,
        ofv: 0.0,
        aic: 0.0,
        bic: 0.0,
        theta: template.theta.clone(),
        theta_names: template.theta_names.clone(),
        eta_names: template.omega.eta_names.clone(),
        omega: template.omega.matrix.clone(),
        sigma: template.sigma.values.clone(),
        sigma_names: template.sigma.names.clone(),
        residual_correlations: Vec::new(),
        error_model: ErrorModel::Proportional,
        covariance_matrix: Some(cov),
        se_theta: None,
        se_omega: None,
        se_sigma: None,
        theta_fixed: template.theta_fixed.clone(),
        omega_fixed: template.omega_fixed.clone(),
        sigma_fixed: template.sigma_fixed.clone(),
        omega_init_as_sd: vec![false; template.omega.matrix.nrows()],
        sigma_init_as_sd: vec![false; template.sigma.values.len()],
        subjects: vec![],
        n_obs: 6,
        n_subjects: 2,
        n_parameters: n_packed,
        n_iterations: 0,
        interaction: true,
        warnings: vec![],
        warnings_structured: vec![],
        sir_ci_theta: None,
        sir_ci_omega: None,
        sir_ci_sigma: None,
        sir_ci_kappa: None,
        sir_ess: None,
        sir_resamples_packed: None,
        importance_sampling: None,
        impmap_trace: None,
        bayes: None,
        vi: None,
        omega_iov: None,
        kappa_names: vec![],
        kappa_fixed: vec![],
        kappa_init_as_sd: vec![],
        kappa_weights: Vec::new(),
        kappa_weight_typical: Vec::new(),
        se_kappa: None,
        shrinkage_kappa: vec![],
        shrinkage_kappa_by_occ: vec![],
        ebe_kappas: vec![],
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        gradient_method_inner: String::new(),
        gradient_method_outer: String::new(),
        uses_ode_solver: false,
        ode_linear_exact: false,
        uses_sde: false,
        n_threads_used: 1,
        nlopt_missing_algorithms: vec![],
        covariance_n_evals_estimated: None,
        trace_path: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        covariance_status: CovarianceStatus::Computed,
        covariance_method: Some(crate::types::CovarianceMethod::Hessian),
        shrinkage_eta: vec![],
        cond_dist: None,
        shrinkage_eps: f64::NAN,
        iwres_lag1_r: f64::NAN,
        dw_statistic: f64::NAN,
        wall_time_secs: 0.0,
        model_name: String::new(),
        ferx_version: String::new(),
        environment: crate::environment::EnvironmentInfo::default(),
        eta_param_info: vec![],
        kappa_param_types: Vec::new(),
        data_bindings: Default::default(),
        theta_transform: vec![],
        sigma_types: vec![],
        cov_eigenvalues: None,
        cov_condition_number: None,
        bic_inputs: Default::default(),
        eta_log_transformed: vec![],
        omega_param_corr: None,
        omega_iov_param_corr: None,
        model_path: None,
        data_path: None,
        model_hash: None,
        data_hash: None,
        model_text: None,
        reader_settings: None,
        population_fingerprint: None,
        iov_occasion: None,
        theta_init: template.theta.clone(),
        omega_init: template.omega.matrix.clone(),
        sigma_init: template.sigma.values.clone(),
        obs_time_range: None,
        final_gradient: None,
        final_gradient_source: None,
        optimizer: "bobyqa".to_string(),
        n_starts: 1,
        multi_start_seed: None,
        saem_seed: None,
        sir_seed: None,
        sir_settings: None,
        scoring_settings: None,
        imp_seed: None,
        npde_seed: None,
        bloq_method: "drop".to_string(),
        outer_maxiter: 0,
        outer_gtol: 0.0,
        inits_from_nca: None,
        covariate_names: Vec::new(),
        input_columns: vec![],
        #[cfg(feature = "nn")]
        neural_networks: Vec::new(),
        covariate_table: None,
        exclusions: None,
        packed_estimate: None,
        left_init: None,
        omega_is_diagonal: None,
        kappa_is_diagonal: None,
    }
}

// ── #785 / #786 flip-flop follow-up fixtures & tests ─────────────────────────
// A twin-*less* transit / IG closed form whose *typical* parameters are IN-DOMAIN
// (`ke = CL/V` below the tilting abscissa) — so it passes the η=0 fit-start reject —
// but which a large positive `ETA_CL` drives into the flip-flop regime. ke0 = 0.5/4 =
// 0.125 < KTR = (3+1)/20 = 0.2.
// Genuinely twin-less: a user `[scaling]` block declines the ODE-twin desugar. (A
// `lagtime=`/`f=` mapping no longer declines it — those auto-route now, #735 — so a
// `[scaling]` form is what keeps this the twin-less case the #785 EBE warning targets.)
// `obs_scale = 1` is an inert identity divisor — it declines the twin without perturbing the
// `pk` block's built-in concentration output. (Using the volume `V` would divide by V a
// second time and emit the #712 double-scale warning; the flip-flop detection keys off
// `CL/V`, so the scale value is otherwise immaterial here.)
const INDOMAIN_TWINLESS_TRANSIT_SRC: &str = "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[scaling]
  obs_scale = 1

[error_model]
  DV ~ proportional(PROP)
";

// IG analogue: ke0 = 5/50 = 0.1 < 1/(2·MAT·CV²) = 1/(2·2·0.3) = 0.833 (in-domain);
// the inert `[scaling] obs_scale = 1` block declines the twin (twin-less; see above).
const INDOMAIN_TWINLESS_IG_SRC: &str = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.15 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  MAT = TVMAT
  CV2 = TVCV2

[structural_model]
  pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)

[scaling]
  obs_scale = 1

[error_model]
  DV ~ proportional(PROP)
";

// Plain transit — no lagtime, so the ODE-twin desugar succeeds (twin-*carrying*):
// a flip-flop subject reroutes per-eval, so the EBE warning must NOT fire.
const PLAIN_TWIN_TRANSIT_SRC: &str = "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP)
";

fn parse_fixture(src: &str) -> CompiledModel {
    crate::parser::model_parser::parse_full_model(src)
        .expect("fixture model parses")
        .model
}

fn eta_vec(values: &[f64]) -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_column_slice(values)
}

/// #785: a twin-less transit closed form in-domain at η=0 but whose second
/// subject's fitted EBE (`ETA_CL = 0.6` → CL/V = 0.911/4 = 0.228 ≥ KTR = 0.2)
/// crosses the flip-flop boundary must surface a typed `FlipFlop` warning naming
/// that subject — the silent-degeneration case the η=0 reject cannot catch.
#[test]
fn flip_flop_ebe_warning_fires_for_twinless_transit_crosser() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    assert!(
        model.absorption_ode_equivalent.is_none(),
        "a [scaling] transit model is twin-less"
    );
    // The inert `obs_scale = 1` must not double-scale the `pk` block's built-in concentration
    // output: no #712 volume double-scale warning (reverting to `obs_scale = V` would
    // reintroduce it and this pins that regression).
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("divides by")),
        "twin-less fixture must not emit the obs_scale double-scale warning: {:?}",
        model.parse_warnings
    );
    let pop = tiny_population();
    // Subject S1 in-domain (η=0), subject S2 crosses.
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[0.6])];
    let (msg, entry) =
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .expect("a crossing EBE must warn");
    assert_eq!(entry.category, crate::types::WarningCode::FlipFlop);
    assert_eq!(entry.message, msg);
    assert!(msg.contains("flip-flop regime"), "wording: {msg}");
    assert!(msg.contains("transit()"), "transit ODE hint: {msg}");
    assert!(msg.contains("S2"), "must name the crossing subject: {msg}");
    assert!(
        !msg.contains("S1"),
        "must not name the in-domain subject: {msg}"
    );
    // The real emitter's message must itself classify to `flip_flop` — pins the
    // classify branch against wording drift (not a hand-copied proxy string).
    assert_eq!(
        crate::types::classify_warning(&msg).category,
        crate::types::WarningCode::FlipFlop
    );
    let details = entry.details.expect("native entry carries details");
    assert_eq!(details["phase"], "ebe");
    assert_eq!(details["subjects"], serde_json::json!(["S2"]));
}

/// #785: the IG closed form takes the same path, with IG-specific wording.
/// ke crosses when CL/V ≥ 1/(2·MAT·CV²) = 0.833; `ETA_CL = 2.3` → CL/V ≈ 0.998.
#[test]
fn flip_flop_ebe_warning_fires_for_twinless_ig_crosser() {
    let model = parse_fixture(INDOMAIN_TWINLESS_IG_SRC);
    assert!(model.absorption_ode_equivalent.is_none());
    // The inert `obs_scale = 1` must not double-scale the IG closed form (no #712 warning).
    assert!(
        !model
            .parse_warnings
            .iter()
            .any(|w| w.contains("divides by")),
        "twin-less IG fixture must not emit the obs_scale double-scale warning: {:?}",
        model.parse_warnings
    );
    let pop = tiny_population();
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[2.3])];
    let (msg, entry) =
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .expect("a crossing IG EBE must warn");
    assert_eq!(entry.category, crate::types::WarningCode::FlipFlop);
    assert!(msg.contains("igd()"), "IG ODE hint: {msg}");
    assert!(msg.contains("S2"), "must name the crossing subject: {msg}");
}

/// #785: every subject in-domain at its EBE → no warning.
#[test]
fn flip_flop_ebe_warning_silent_when_all_in_domain() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    // CL/V = 0.5·e^{0.1}/4 = 0.138 < 0.2 for both.
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[0.1])];
    assert!(
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .is_none()
    );
}

/// #785: a twin-*carrying* model reroutes per-eval, so even a crossing EBE must
/// not warn (the fit is correct for it).
#[test]
fn flip_flop_ebe_warning_silent_for_twin_carrying() {
    let model = parse_fixture(PLAIN_TWIN_TRANSIT_SRC);
    assert!(
        model.absorption_ode_equivalent.is_some(),
        "a plain transit model carries an ODE twin"
    );
    let pop = tiny_population();
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[0.6])];
    assert!(
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .is_none()
    );
}

/// #785: a non-absorption model has no flip-flop regime → no warning.
#[test]
fn flip_flop_ebe_warning_silent_for_non_absorption() {
    let model = tiny_model();
    assert!(!matches!(
        model.pk_model,
        PkModel::OneCptTransit | PkModel::TwoCptTransit | PkModel::OneCptIg | PkModel::TwoCptIg
    ));
    let pop = tiny_population();
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[3.0])];
    assert!(
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .is_none()
    );
}

/// #785: an EBE whose length ≠ `n_eta + n_kappa` is defensively skipped rather
/// than fed to `pk_params_at_time` (which would panic reading `eta[0]` of a
/// zero-length vector). Even a would-be crosser given the wrong shape yields no
/// warning — proving the guard, not a panic.
#[test]
fn flip_flop_ebe_warning_skips_wrong_length_eta() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    // S1 valid (in-domain, len 1); S2 has a zero-length eta (≠ n_eta = 1).
    let eta_hats = [eta_vec(&[0.0]), eta_vec(&[])];
    assert!(
        absorption_flip_flop_ebe_warning(&model, &pop, &model.default_params.theta, &eta_hats)
            .is_none()
    );
}

/// #786: `simulate_with_uncertainty` on a twin-less transit model whose point
/// estimate is in-domain but whose covariance spreads some draws into the
/// flip-flop regime must NOT fail (pre-fix: the per-draw
/// chokepoint flip-flop check aborted the whole run). The crossing
/// draws are skipped; the rest still yield rows.
#[test]
fn uncertainty_skips_flip_flop_draws_without_panicking() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let mut fit = synthetic_fit(&model.default_params);
    // Inflate the log-TVCL variance (packed index 0) so a good fraction of draws
    // push ke = CL/V past KTR = 0.2 (needs TVCL ≥ 0.8, a +0.47 log-shift).
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = 4.0;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 30,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(7),
    };
    // Must return Ok rather than panicking (the #786 regression).
    let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts)
        .expect("flip-flop draws must be skipped, not panic the run");
    assert!(
        !rows.is_empty(),
        "surviving in-domain draws must still yield rows"
    );
    let mut draws: Vec<usize> = rows.iter().map(|r| r.draw).collect();
    draws.sort();
    draws.dedup();
    assert!(
        draws.len() < 30,
        "some flip-flop draws must be skipped (got {} of 30)",
        draws.len()
    );

    // #1485: the `_diag` form reports the skips as ONE warning carrying the count and the
    // 1-based indices of exactly the draws missing from the rows — no per-draw lines.
    let diag = simulate_with_uncertainty_diag(&model, &pop, &fit, &opts).expect("diag");
    let skipped: Vec<usize> = (1..=30).filter(|k| !draws.contains(k)).collect();
    assert_eq!(skipped, vec![7, 16, 19, 22, 23, 24], "measured at d43afca9");
    let msg = only_skip_warning(&diag.warnings);
    assert!(msg.contains("6 of 30 uncertainty draws"), "{msg}");
    assert!(msg.contains("(draws 7, 16, 19, 22, 23, 24)"), "{msg}");
    assert!(
        !diag
            .warnings
            .iter()
            .any(|w| w.starts_with("uncertainty draw ")),
        "the per-draw lines are gone: {:#?}",
        diag.warnings
    );
    // The wrapper's rows are the `_diag` rows, and every surviving draw ran.
    assert_eq!(rows.len(), 144);
    assert_eq!(rows.len(), diag.results.len());
    for (a, b) in rows.iter().zip(&diag.results) {
        assert_eq!((a.draw, a.sim, &a.id), (b.draw, b.sim, &b.id));
        assert_eq!(a.ipred.to_bits(), b.ipred.to_bits());
        assert_eq!(format!("{:?}", a.outcome), format!("{:?}", b.outcome));
    }
    let expected_draws: Vec<usize> = (1..=30).filter(|k| !skipped.contains(k)).collect();
    assert_eq!(draws, expected_draws);

    // Pin the skip *predicate* directly too: a flip-flop-theta draw is flagged,
    // the in-domain point estimate is not.
    let mut crossing = model.default_params.theta.clone();
    crossing[0] = 1.0; // TVCL = 1.0 → ke = 1.0/4 = 0.25 ≥ KTR = 0.2 (flip-flop)
    assert!(
        check_absorption_flip_flop_no_twin(&model, &pop, &crossing).is_some(),
        "a flip-flop-theta draw must be flagged for skipping"
    );
    assert!(
        check_absorption_flip_flop_no_twin(&model, &pop, &model.default_params.theta).is_none(),
        "the in-domain point estimate must not be flagged"
    );
}

/// The single `W_UNCERTAINTY_DRAWS_SKIPPED` warning in `warnings` (#1485).
fn only_skip_warning(warnings: &[String]) -> &str {
    let hits: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("W_UNCERTAINTY_DRAWS_SKIPPED"))
        .collect();
    assert_eq!(hits.len(), 1, "exactly one skip warning: {warnings:#?}");
    hits[0]
}

/// Run `simulate_with_uncertainty_diag` with cov(log TVCL) = `var` and return the skip
/// warning plus the skipped draw indices read off the rows.
fn skip_run(model: &CompiledModel, var: f64, n: usize, seed: u64) -> (String, Vec<usize>) {
    let pop = tiny_population();
    let mut fit = synthetic_fit(&model.default_params);
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = var;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: n,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(seed),
    };
    let out = simulate_with_uncertainty_diag(model, &pop, &fit, &opts).expect("some draws run");
    let skipped: Vec<usize> = (1..=n)
        .filter(|k| !out.results.iter().any(|r| r.draw == *k))
        .collect();
    (only_skip_warning(&out.warnings).to_string(), skipped)
}

// 2-cpt transit, in-domain at the point estimate: k10 = k12 = 0.125, k21 = 0.0625 →
// α ≈ 0.285 < KTR = (3+1)/10 = 0.4. Twin-less through the inert `[scaling]` block.
const INDOMAIN_TWINLESS_TRANSIT_2CPT_SRC: &str = "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV1(4.0, 0.1, 500.0)
  theta TVQ(0.5, 0.001, 50.0)
  theta TVV2(8.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(10.0, 0.05, 200.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  NTR = TVNTR
  MTT = TVMTT

[structural_model]
  pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT)

[scaling]
  obs_scale = 1

[error_model]
  DV ~ proportional(PROP)
";

// 2-cpt IG, in-domain at the point estimate: k10 = k12 = 0.1, k21 = 0.05 → α ≈ 0.228 <
// 1/(2·MAT·CV²) = 0.833. Twin-less through the inert `[scaling]` block.
//
// This is the **cross-cell** of the two wording predicates — IG *and* 2-cpt — and the only
// fixture covering `PkModel::TwoCptIg` in `twinless_absorption_closed_form` (#1823 review
// r6). Without it, dropping `TwoCptIg` from either that family list or the IG arm of
// `absorption_flip_flop_wording` left the whole suite green: measured, both mutations
// survived. The round-2 argument that three fixtures suffice because the two wording
// predicates are independent was right about the *wording* and wrong about the family
// predicate, which enumerates all four variants.
const INDOMAIN_TWINLESS_IG_2CPT_SRC: &str = "\
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 5.0, 500.0)
  theta TVQ(5.0, 0.1, 100.0)
  theta TVV2(100.0, 5.0, 1000.0)
  theta TVMAT(2.0, 0.05, 24.0)
  theta TVCV2(0.3, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.15 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q = TVQ
  V2 = TVV2
  MAT = TVMAT
  CV2 = TVCV2

[structural_model]
  pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2)

[scaling]
  obs_scale = 1

[error_model]
  DV ~ proportional(PROP)
";

/// #1485: every sentence of the skip warning, on all three families, with both sides of
/// each wording gate asserted in the same test — transit vs IG (domain + ODE rewrite) and
/// 1-cpt vs 2-cpt (the confluent-eigenvalue clause).
#[test]
fn uncertainty_skip_warning_wording_per_family() {
    let transit = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let ig = parse_fixture(INDOMAIN_TWINLESS_IG_SRC);
    let transit2 = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_2CPT_SRC);
    assert!(transit2.absorption_ode_equivalent.is_none(), "twin-less");
    assert!(
        check_absorption_flip_flop_no_twin(
            &transit2,
            &tiny_population(),
            &transit2.default_params.theta
        )
        .is_none(),
        "the 2-cpt point estimate is in-domain"
    );

    let (t1, t1_skipped) = skip_run(&transit, 4.0, 30, 7);
    let (ig1, ig1_skipped) = skip_run(&ig, 4.0, 30, 7);
    let (t2, t2_skipped) = skip_run(&transit2, 4.0, 30, 7);

    for (msg, skipped, n_name) in [
        (&t1, &t1_skipped, "one_cpt_transit"),
        (&ig1, &ig1_skipped, "one_cpt_ig"),
        (&t2, &t2_skipped, "two_cpt_transit"),
    ] {
        // Count + indices sentence: exactly the draws missing from the rows.
        let list = skipped
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        assert!(
            !skipped.is_empty() && skipped.len() < 30,
            "{n_name}: {skipped:?}"
        );
        assert!(
            msg.contains(&format!(
                "{} of 30 uncertainty draws were skipped (draws {list})",
                skipped.len()
            )),
            "{n_name}: {msg}"
        );
        // Cause sentence.
        assert!(msg.contains("flip-flop regime"), "{n_name}: {msg}");
        assert!(
            msg.contains(&format!("twin-less {n_name} closed form")),
            "{msg}"
        );
        // Bias sentence.
        assert!(
            msg.contains("exclude the tail of the parameter uncertainty"),
            "{n_name}: {msg}"
        );
        assert!(msg.contains("biased, not just noisier"), "{n_name}: {msg}");
        // The remaining rows are not claimed correct, and CL is not named as the cause.
        assert!(!msg.contains("high CL"), "{n_name}: {msg}");
        // A token-carrying message classifies on the token, not the "flip-flop" prose.
        let entry = crate::types::classify_warning(msg);
        assert_eq!(
            entry.category,
            crate::types::WarningCode::UncertaintyDrawsSkipped
        );
        assert_eq!(entry.severity, crate::types::WarningSeverity::Warning);
    }
    // Transit vs IG: the domain and the ODE rewrite.
    for m in [&t1, &t2] {
        assert!(m.contains("transit rate KTR = (n+1)/mtt"), "{m}");
        assert!(m.contains("explicit ODE `transit()` model"), "{m}");
        assert!(!m.contains("igd()") && !m.contains("MAT"), "{m}");
    }
    assert!(ig1.contains("disposition rate ≥ 1/(2·MAT·CV²)"), "{ig1}");
    assert!(ig1.contains("explicit ODE `igd()` model"), "{ig1}");
    assert!(!ig1.contains("KTR") && !ig1.contains("transit()"), "{ig1}");
    // 1-cpt vs 2-cpt: only a 2-cpt model has a confluent-eigenvalue edge.
    assert!(
        t2.contains("or with coincident disposition eigenvalues"),
        "{t2}"
    );
    assert!(!t1.contains("coincident"), "{t1}");
    assert!(!ig1.contains("coincident"), "{ig1}");
}

/// The M1 transit fixture with `[scaling]` swapped for an individual parameter named after
/// the twin's own `central` state: the twin is built and its parse rejects the collision
/// (#1008), so the model is twin-less *and* carries
/// [`crate::types::absorption_twin_decline_reason`]. One builder for the two tests that need
/// that distinction (#1485's skip warning and #1823's tally).
fn declined_twin_transit_src() -> String {
    INDOMAIN_TWINLESS_TRANSIT_SRC
        .replace("  MTT = TVMTT\n", "  MTT = TVMTT\n  CENTRAL = TVX\n")
        .replace(
            "  theta TVMTT(20.0, 0.05, 200.0)\n",
            "  theta TVMTT(20.0, 0.05, 200.0)\n  theta TVX(1.0, 0.0, 10.0)\n",
        )
        .replace("[scaling]\n  obs_scale = 1\n\n", "")
}

/// #1485 review finding 3: on a model whose ODE twin was built and then declined (#1008),
/// the aggregate keeps the decline clause the per-draw lines used to carry — it names the
/// easier fix. The `[scaling]` fixture, whose twin was never built, must not carry it.
#[test]
fn uncertainty_skip_warning_carries_the_twin_decline_clause() {
    let declined = parse_fixture(&declined_twin_transit_src());
    assert!(declined.absorption_ode_equivalent.is_none(), "twin-less");
    assert!(
        crate::types::absorption_twin_decline_reason(&declined).is_some(),
        "the twin was built and declined"
    );
    let scaling = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    assert!(crate::types::absorption_twin_decline_reason(&scaling).is_none());

    let clause = "its twin was built and then rejected by its own parse";
    let (with, _) = skip_run(&declined, 4.0, 30, 7);
    let (without, _) = skip_run(&scaling, 4.0, 30, 7);
    assert!(with.contains(clause), "{with}");
    assert!(
        with.contains("CENTRAL"),
        "the twin parser's reason is quoted: {with}"
    );
    assert!(!without.contains(clause), "{without}");
}

/// #1485: an in-domain point estimate whose every draw is skipped is refused rather than
/// returning an empty `Ok`. Fixture chosen by measurement: point estimate TVCL = 0.75 (ke =
/// 0.1875 < KTR = 0.2) with a log-variance wide enough that, at this seed, all draws cross.
#[test]
fn uncertainty_refuses_when_every_draw_is_skipped() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let mut params = model.default_params.clone();
    params.theta[0] = 0.75;
    assert!(check_absorption_flip_flop_no_twin(&model, &pop, &params.theta).is_none());
    let mut fit = synthetic_fit(&params);
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = ALL_SKIP_VAR;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: ALL_SKIP_N,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(ALL_SKIP_SEED),
    };
    let err = simulate_with_uncertainty_diag(&model, &pop, &fit, &opts)
        .expect_err("every draw skipped must be refused");
    assert_eq!(err.code(), Some("E_TRANSIT_FLIP_FLOP"), "{err}");
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("all {ALL_SKIP_N} uncertainty draws")),
        "{msg}"
    );
    assert!(
        msg.contains("although the point estimate is in-domain"),
        "{msg}"
    );
    assert!(msg.contains("First draw: one_cpt_transit"), "{msg}");
    assert!(!msg.contains("W_UNCERTAINTY_DRAWS_SKIPPED"), "{msg}");
    // Straddle on the `k = n` boundary: the same fixture at a seed that skips n − 1 draws
    // still runs, and says so — a refusal at `k ≥ n − 1` reddens this half.
    let partial = SimulateUncertaintyOptions {
        seed: Some(ALL_BUT_ONE_SKIP_SEED),
        ..opts
    };
    let out = simulate_with_uncertainty_diag(&model, &pop, &fit, &partial)
        .expect("a run with a surviving draw is not refused");
    let warn = only_skip_warning(&out.warnings);
    assert!(
        warn.contains(&format!(
            "{} of {ALL_SKIP_N} uncertainty draws",
            ALL_SKIP_N - 1
        )),
        "{warn}"
    );
}

/// #1485: no skipped draw, no warning — on the twin-carrying twin of the M1 fixture (same
/// wide covariance, every draw reroutes to the ODE `transit()`), and on a non-absorption
/// model. And a zero-draw run stays `Ok` (k = n = 0 is not "every draw skipped").
#[test]
fn uncertainty_without_skips_carries_no_skip_warning() {
    let pop = tiny_population();
    for (name, model) in [
        ("twin", parse_fixture(PLAIN_TWIN_TRANSIT_SRC)),
        ("tiny", tiny_model()),
    ] {
        let mut fit = synthetic_fit(&model.default_params);
        if name == "twin" {
            if let Some(cov) = fit.covariance_matrix.as_mut() {
                cov[(0, 0)] = 4.0;
            }
        }
        for n in [0usize, 30] {
            let opts = SimulateUncertaintyOptions {
                n_uncertainty_draws: n,
                n_sim_per_draw: 1,
                method: UncertaintyMethod::Asymptotic,
                seed: Some(7),
            };
            let out = simulate_with_uncertainty_diag(&model, &pop, &fit, &opts)
                .unwrap_or_else(|e| panic!("{name} n={n}: {e}"));
            let mut draws: Vec<usize> = out.results.iter().map(|r| r.draw).collect();
            draws.dedup();
            assert_eq!(draws, (1..=n).collect::<Vec<_>>(), "{name} n={n}");
            assert!(
                !out.warnings
                    .iter()
                    .any(|w| w.contains("W_UNCERTAINTY_DRAWS_SKIPPED")),
                "{name} n={n}: {:#?}",
                out.warnings
            );
        }
    }
}

// Measured by seed scans at this var / n: over 0..300, 7 seeds skip every draw (9 the
// first); over 0..60, seed 2 skips 4 of 5 (also 45, 50, 52) — the straddle partner.
const ALL_BUT_ONE_SKIP_SEED: u64 = 2;
const ALL_SKIP_VAR: f64 = 0.01;
const ALL_SKIP_N: usize = 5;
const ALL_SKIP_SEED: u64 = 9;

/// #1485: a point estimate that is itself in the flip-flop regime of a twin-less transit
/// closed form is refused with the code `simulate()` gives at that θ, before any draw —
/// pre-fix it returned `Ok` with 28 of 30 draws skipped.
#[test]
fn uncertainty_refuses_a_flip_flop_point_estimate() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let mut params = model.default_params.clone();
    params.theta[0] = 1.0; // TVCL = 1.0 → ke = 0.25 ≥ KTR = 0.2
    let mut fit = synthetic_fit(&params);
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = 0.01;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 30,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(7),
    };
    let err = simulate_with_uncertainty_diag(&model, &pop, &fit, &opts)
        .expect_err("a flip-flop point estimate must be refused");
    assert_eq!(err.code(), Some("E_TRANSIT_FLIP_FLOP"), "{err}");
    // The same refusal `simulate()` gives at that θ — not the all-draws-skipped one.
    let point = simulate(&model, &pop, &params, 1).expect_err("simulate refuses too");
    assert_eq!(err.to_string(), point.to_string());
    assert!(!err.to_string().contains("uncertainty draws"), "{err}");
}

// ── #1823 simulated-η flip-flop tally ────────────────────────────────────────
// The entry-point refusal samples η = 0, the #785 warning samples the fitted EBEs and the
// #786/#1485 skip samples each draw's η = 0 θ. None of them sees an individual whose own
// *sampled* η crosses the tilting abscissa, which on a twin-less closed form comes back as
// `ipred = 0` for the whole profile with nothing said on any channel.

/// `pop` with every dose amount set to zero: the **dose-free twin**, which is the oracle
/// for "the entire dose-driven concentration was clamped".
///
/// A clamped individual's prediction must equal its dose-free twin's bit for bit — its dose
/// contributed nothing — and a non-clamped one must not. Nothing else changes: the subject
/// count and the observation grid are identical, so `simulate*` consumes the RNG in the
/// same order and the two runs draw the same etas and epsilons at the same seed.
///
/// This is the oracle `ipred == 0.0` cannot be, because a clamped individual is only
/// literally zero when nothing is layered on the closed form's output afterwards
/// (`validation::clamp_shows_as_literal_zero`) — measured on `TWINLESS_TRANSIT_INIT_SRC`,
/// where 27 individuals are clamped and not one row reads 0.
fn dose_free(pop: &Population) -> Population {
    let mut out = pop.clone();
    for s in &mut out.subjects {
        for d in &mut s.doses {
            d.amt = 0.0;
        }
    }
    out
}

/// Individuals whose prediction is bit-identical to the dose-free twin's, and how many were
/// simulated in all, keyed by `(draw, sim, id)`. `a` and `b` must be two runs of the same
/// model at the same seed, differing only in `dose_free`.
fn dose_free_matches(a: &[SimulationResult], b: &[SimulationResult]) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "the two runs must emit the same rows");
    let mut per_individual: HashMap<(usize, usize, String), bool> = HashMap::new();
    for (x, y) in a.iter().zip(b) {
        assert_eq!(
            (x.draw, x.sim, &x.id, x.time),
            (y.draw, y.sim, &y.id, y.time),
            "the dose-free twin must emit the same grid in the same order"
        );
        let same = per_individual
            .entry((x.draw, x.sim, x.id.clone()))
            .or_insert(true);
        *same &= x.ipred.to_bits() == y.ipred.to_bits();
    }
    let matched = per_individual.values().filter(|m| **m).count();
    // The oracle has to be able to say "no" (#1823 review r3). If a fixture's dose never
    // reached an observation — dosed after the last one, a `lagtime` past it, `f = 0` —
    // then *every* individual matches its dose-free twin, crossing or not, and the
    // comparison stops discriminating. The callers assert an exact count, which already
    // catches that; this makes the requirement explicit so relaxing one of them to
    // `> 0` cannot quietly re-open it.
    assert!(
        matched < per_individual.len(),
        "the dose-free oracle is degenerate on this fixture: all {} individuals match it, \
         so the comparison cannot distinguish a clamped individual from a live one",
        per_individual.len()
    );
    (matched, per_individual.len())
}

/// #1823 review r4: the `W_SIMULATED_FLIP_FLOP` example quoted in
/// `docs/api/simulation.qmd` is the text the implementation actually emits.
///
/// Review r4 found the example stale — it had kept the pre-r2 wording and omitted "that
/// received a dose". A hand-copied message in prose cannot be kept honest by reading it, so
/// this reads the fenced block out of the page and asserts the real warning contains it
/// (whitespace-normalised, since the page hard-wraps). Any future reword of the message now
/// reddens here until the page is updated, in either direction.
#[test]
fn the_documented_warning_example_is_the_message_the_code_emits() {
    // `docs/*` is excluded from the published crate (`Cargo.toml`), and `src/` unit tests
    // *are* shipped, so a `cargo test --lib` from an unpacked tarball has no `docs/` to
    // read. Skip only when the whole directory is absent — that is the packaged case and
    // there are no docs there to protect. In any repo checkout, where this gate is the
    // point, a missing page or a missing quote is still a failure.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if !root.join("docs").is_dir() {
        return;
    }
    let page = std::fs::read_to_string(root.join("docs/api/simulation.qmd"))
        .expect("docs/ is present, so the simulation page must be too");
    // The one fenced block that quotes this warning.
    let start = page
        .find("W_SIMULATED_FLIP_FLOP: 27 of 600")
        .expect("the page quotes the warning");
    let end = start
        + page[start..]
            .find("\n```")
            .expect("the quote sits in a fenced block");
    let quoted = normalize_ws(&page[start..end]);

    // The real thing, at the fixture and seed the page names.
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let (msg, clamped, total) = clamped_run(&model, 0.09, 300, 7);
    assert_eq!((clamped, total), (27, 600), "the page's 27 of 600");
    let emitted = normalize_ws(&msg.expect("the fixture reports"));
    // **Equality**, not containment (#1823 review r5): a `contains` check with a length
    // floor passes on a truncated quote — dropping the last three sentences still leaves a
    // 200-character prefix — so it would not notice the page quoting half the message. This
    // fixture carries no twin-decline clause, so the page's block is the whole thing.
    assert_eq!(
        emitted, quoted,
        "the documented example has drifted from the emitted message"
    );
}

/// Collapse every run of whitespace to one space, so a hard-wrapped quote in a `.qmd` page
/// compares against the single-line message the code builds.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// #1823 review r7: an individual whose parameters overflow to non-finite is **not** counted.
///
/// `absorption_flip_flop_at` guards its inputs with positivity only (`cl > 0.0 && v1 > 0.0`),
/// and `f64::INFINITY > 0.0` is `true`, so an overflowed draw reaches the domain comparison
/// and `!(x < KTR)` is `true` for both `x = inf` and `x = NaN`. The individual was therefore
/// reported as a domain crossing, with the message attributing it to "disposition rate ≥ the
/// tilting abscissa" and recommending an ODE "valid in both regimes" — which would receive
/// the same non-finite parameters. That is the predicate's own stated contract being broken
/// ("invalid parameters … a deliberate optimiser penalty, not a flip-flop"), so the tally
/// enforces it.
///
/// Measured at η = 710, where `exp(η)` overflows:
///
/// | fixture | `cl` | `v` | `cl / v` | pre-fix |
/// |---|---|---|---|---|
/// | η on `CL` only | `inf` | 4 | `inf` | counted |
/// | η on `CL` and `V` | `inf` | `inf` | `NaN` | counted |
///
/// Both sides of the gate in one test: the same two fixtures at a *finite* crossing η are
/// still counted, and the η-on-both fixture at η = 0.6 is correctly in-domain (`CL/V` is
/// unchanged at 0.125 because both scale together), which is what makes it a real
/// discriminator rather than a second copy of the first.
#[test]
fn simulated_flip_flop_tally_skips_a_nonfinite_individual() {
    let cl_only = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    // Same fixture with η on V as well, so a huge draw overflows both and the ratio is NaN.
    let cl_and_v = parse_fixture(
        &INDOMAIN_TWINLESS_TRANSIT_SRC.replace("  V = TVV\n", "  V = TVV * exp(ETA_CL)\n"),
    );
    let pop = tiny_population();
    let subject = &pop.subjects[0];

    for (name, model, crossing_eta) in [
        ("η on CL only", &cl_only, 0.6_f64),
        // With η on both, CL/V is η-independent, so no finite η crosses — the overflow is
        // the only way in, which is exactly the case under test.
        ("η on CL and V", &cl_and_v, f64::NAN),
    ] {
        let theta = model.default_params.theta.clone();
        for eta in [710.0_f64, 1e4] {
            let pk = (model.pk_param_fn)(&theta, &[eta], &subject.covariates, 0.0);
            assert!(
                !pk.values.iter().all(|v| v.is_finite()),
                "{name}: η = {eta} must overflow for this test to mean anything (cl = {}, v = {})",
                pk.cl(),
                pk.v()
            );
            // The predicate still says "crossing" — that is the pre-existing behaviour this
            // guard compensates for, asserted so the guard's reason cannot go stale.
            assert!(
                crate::pk::absorption_flip_flop_at(model, subject, &theta, &[eta]),
                "{name}: the predicate flags a non-finite individual"
            );
            let mut tally = SimulatedFlipFlopTally::default();
            tally.observe(model, subject, &theta, &[eta]);
            assert_eq!(
                tally.warning(model),
                None,
                "{name}: η = {eta} overflows, so it is not a domain crossing"
            );
        }
        // The other side: a finite crossing η is still counted (where one exists).
        if crossing_eta.is_finite() {
            let mut tally = SimulatedFlipFlopTally::default();
            tally.observe(model, subject, &theta, &[crossing_eta]);
            assert!(
                tally.warning(model).is_some(),
                "{name}: a finite crossing η must still be counted"
            );
        }
    }

    // And the η-on-both fixture is genuinely in-domain at a finite η, which is why it is a
    // different fixture and not a restatement of the first.
    let theta = cl_and_v.default_params.theta.clone();
    let pk = (cl_and_v.pk_param_fn)(&theta, &[0.6], &subject.covariates, 0.0);
    assert!(
        (pk.cl() / pk.v() - 0.125).abs() < 1e-12,
        "CL/V is η-independent here"
    );
    let mut tally = SimulatedFlipFlopTally::default();
    tally.observe(&cl_and_v, subject, &theta, &[0.6]);
    assert_eq!(tally.warning(&cl_and_v), None);

    // **Mixed, which is the case the guard's placement decides** (#1823 review r8). An
    // overflowed individual is still *eligible* — it is dosed before an observation, which
    // is what the message's denominator names — so it belongs in that denominator even
    // though it is not a crossing. One finite crosser beside one overflowed draw is
    // therefore "1 of 2", not "1 of 1": dropping it from both sides would report a larger
    // share than the population the count was measured over.
    //
    // Asserting the non-finite individual *alone* cannot see this — there is no crossing,
    // so no warning is emitted at all — which is why this case exists separately.
    let theta = cl_only.default_params.theta.clone();
    let mut mixed = SimulatedFlipFlopTally::default();
    mixed.observe(&cl_only, subject, &theta, &[0.6]); // finite crosser
    mixed.observe(&cl_only, subject, &theta, &[710.0]); // eligible, overflowed
    let msg = mixed
        .warning(&cl_only)
        .expect("the finite crosser is reported");
    assert!(
        msg.contains("1 of 2 simulated individuals dosed before one of their observations"),
        "the overflowed individual stays in the denominator: {msg}"
    );
}

/// #1823 review r9: the message stays true for a counted individual the clamp did **not**
/// act on, and the docs do not claim its rows are finite.
///
/// Two cases, both measured, both reached through the same count:
///
/// 1. **A `lagtime` past every observation.** `[scaling]` declines the twin while a
///    `lagtime=` binding is carried in the closed form (a bare `lagtime=` auto-routes since
///    #735, so it takes both to build this), and the gate compares the *raw* dose time, so
///    the individual is counted. But the absorption never arrives, the domain guard is never
///    reached, and the rows are identical at a crossing η and at η = 0 — `[0, 0, 0]` either
///    way. So "**so** their entire dose-driven concentration was clamped to zero" asserted a
///    cause that did not occur. The message now states the clamp as a property of the closed
///    form and the undosed rows as the observation, without linking them causally.
///
/// 2. **An η-dependent `obs_scale` that goes non-positive.** `apply_scaling` divides, and
///    replaces the row with `NaN` when the evaluated scale is not finite and positive
///    (`pk/mod.rs`). The tally's finiteness guard checks the **PK parameters**, not the
///    evaluated scale, so a counted individual's rows can be `NaN`. Measured on
///    `obs_scale = 1 - ETA_CL`: η = 0.6 gives `[0, 0, 0]`, η = 2 gives `[NaN, NaN, NaN]`,
///    both counted. The docs no longer claim clamped rows stay finite; this pins the
///    behaviour behind that wording.
#[test]
fn simulated_flip_flop_message_holds_where_the_clamp_was_not_the_cause() {
    let pop = tiny_population();
    let subject = &pop.subjects[0];

    // ── 1. lagtime past every observation ───────────────────────────────────────────
    let lagged = parse_fixture(
        &INDOMAIN_TWINLESS_TRANSIT_SRC
            .replace("  MTT = TVMTT\n", "  MTT = TVMTT\n  LAG = TVLAG\n")
            .replace(
                "  theta TVMTT(20.0, 0.05, 200.0)\n",
                "  theta TVMTT(20.0, 0.05, 200.0)\n  theta TVLAG(50.0, 0.0, 500.0)\n",
            )
            .replace(
                "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)",
                "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT, lagtime=LAG)",
            ),
    );
    assert!(
        lagged.absorption_ode_equivalent.is_none(),
        "the `[scaling]` block declines the twin even with a `lagtime=` binding (#735)"
    );
    let theta = lagged.default_params.theta.clone();
    let in_domain = crate::pk::compute_predictions_with_tv(&lagged, subject, &theta, &[0.0]);
    let crossing = crate::pk::compute_predictions_with_tv(&lagged, subject, &theta, &[0.6]);
    assert_eq!(
        in_domain, crossing,
        "the lag (50) is past every observation (1, 2, 3), so the crossing changes nothing"
    );
    assert_eq!(in_domain, vec![0.0, 0.0, 0.0]);
    let mut tally = SimulatedFlipFlopTally::default();
    tally.observe(&lagged, subject, &theta, &[0.6]);
    let msg = tally
        .warning(&lagged)
        .expect("the gate reads the raw dose time, so this individual is counted");
    // The row-level claim is true here — nothing from the dose reaches these rows — while
    // the causal one would not be.
    assert!(
        msg.contains("None of their rows carries any contribution from the doses the subject"),
        "{msg}"
    );
    assert!(
        msg.contains("because the dose never arrived"),
        "the message names this alternative cause: {msg}"
    );
    assert!(
        !msg.contains("so their entire dose-driven concentration was clamped"),
        "{msg}"
    );

    // ── 2. an η-dependent obs_scale going non-positive ──────────────────────────────
    let scaled = parse_fixture(
        &INDOMAIN_TWINLESS_TRANSIT_SRC.replace("  obs_scale = 1\n", "  obs_scale = 1 - ETA_CL\n"),
    );
    let theta = scaled.default_params.theta.clone();
    for (eta, want_finite) in [(0.0_f64, true), (0.6, true), (2.0, false)] {
        let preds = crate::pk::compute_predictions_with_tv(&scaled, subject, &theta, &[eta]);
        assert_eq!(
            preds.iter().all(|p| p.is_finite()),
            want_finite,
            "η = {eta}: `apply_scaling` divides and writes NaN for a non-positive scale              (scale = {}), so a counted individual's rows are not guaranteed finite: {preds:?}",
            1.0 - eta
        );
        let mut tally = SimulatedFlipFlopTally::default();
        tally.observe(&scaled, subject, &theta, &[eta]);
        // η = 0 is in-domain; both crossing draws are counted, including the NaN one — the
        // PK parameters are finite, which is all the tally's guard asks.
        assert_eq!(tally.warning(&scaled).is_some(), eta > 0.0, "η = {eta}");
    }
}

/// #1823 review r3: [`dose_free_matches`]'s own degeneracy guard fires. No fixture in this
/// file is degenerate, so nothing else can reach it — and a guard whose job is to reject a
/// *future* fixture is worth nothing if it cannot be shown to fire at all.
///
/// Two identical row sets are exactly the degenerate case: every individual matches, so the
/// comparison cannot tell a clamped individual from a live one.
#[test]
#[should_panic(expected = "the dose-free oracle is degenerate on this fixture")]
fn dose_free_oracle_rejects_a_degenerate_fixture() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let rows = flip_flop_rows(&model, 1e-4, 2, 7);
    assert!(!rows.is_empty(), "the fixture emits rows");
    // `rows` against itself: matched == total, which must be refused.
    dose_free_matches(&rows, &rows);
}

/// Simulated individuals whose whole profile came back identically zero, and how many were
/// simulated in all, keyed by `(draw, sim, id)`.
///
/// Read off the **rows**, never from `absorption_flip_flop_at` — the predicate the fix
/// counts on. Only a valid count of the *clamped* individuals on a model where the clamp
/// shows as a literal zero; `dose_free_matches` is the oracle that holds on every model.
fn zeroed_individuals(results: &[SimulationResult]) -> (usize, usize) {
    let mut per_individual: HashMap<(usize, usize, String), bool> = HashMap::new();
    for r in results {
        let all_zero = per_individual
            .entry((r.draw, r.sim, r.id.clone()))
            .or_insert(true);
        *all_zero &= r.ipred == 0.0;
    }
    (
        per_individual.values().filter(|z| **z).count(),
        per_individual.len(),
    )
}

/// The single `W_SIMULATED_FLIP_FLOP` warning in `warnings`, or `None`.
fn flip_flop_warning(warnings: &[String]) -> Option<String> {
    let hits: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("W_SIMULATED_FLIP_FLOP"))
        .collect();
    assert!(hits.len() <= 1, "at most one tally warning: {warnings:#?}");
    hits.first().map(|w| (*w).clone())
}

/// `simulate_with_options_diag` with ω²(ETA_CL) widened to `var`: the tally warning (if
/// any), the zeroed-individual count read off the rows, and the individuals simulated.
///
/// Widening ω is how a crosser is produced at all — the fixture's point estimate is
/// in-domain by construction (the entry points refuse it otherwise), so the crossing has
/// to come from the η draw, which is exactly the gap under test.
fn flip_flop_run(
    model: &CompiledModel,
    var: f64,
    n_sim: usize,
    seed: u64,
    match_method: Option<crate::propensity_match::MatchMethod>,
) -> (Option<String>, usize, usize) {
    let pop = tiny_population();
    assert_eq!(model.n_eta, 1, "every fixture here carries one η");
    let mut params = model.default_params.clone();
    params.omega = OmegaMatrix::from_diagonal(&[var], model.eta_names.clone());
    let opts = SimulateOptions {
        seed: Some(seed),
        match_method,
        ..Default::default()
    };
    let out = simulate_with_options_diag(model, &pop, &params, n_sim, &opts)
        .expect("an in-domain point estimate simulates");
    // The row-only wrapper is the same rows, so the tally cannot have perturbed the draw
    // order (it consumes no randomness).
    let rows = simulate_with_options(model, &pop, &params, n_sim, &opts).expect("rows");
    assert_eq!(rows.len(), out.results.len());
    for (a, b) in rows.iter().zip(&out.results) {
        assert_eq!(a.ipred.to_bits(), b.ipred.to_bits());
    }
    let (zeroed, total) = zeroed_individuals(&out.results);
    (flip_flop_warning(&out.warnings), zeroed, total)
}

/// `simulate_with_options_diag`'s rows at ω² = `var`, for an oracle that needs them twice.
fn flip_flop_rows(
    model: &CompiledModel,
    var: f64,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    let mut params = model.default_params.clone();
    params.omega = OmegaMatrix::from_diagonal(&[var], model.eta_names.clone());
    let opts = SimulateOptions {
        seed: Some(seed),
        ..Default::default()
    };
    simulate_with_options_diag(model, &tiny_population(), &params, n_sim, &opts)
        .expect("an in-domain point estimate simulates")
        .results
}

/// The tally warning, the individuals the **dose-free twin** says were clamped, and the
/// individuals simulated. Unlike [`flip_flop_run`] this oracle holds on a model that layers
/// a baseline / readout / log transform on the closed form's output, where a clamped
/// individual is not zero at all (#1823 review r1).
fn clamped_run(
    model: &CompiledModel,
    var: f64,
    n_sim: usize,
    seed: u64,
) -> (Option<String>, usize, usize) {
    assert_eq!(model.n_eta, 1, "every fixture here carries one η");
    let mut params = model.default_params.clone();
    params.omega = OmegaMatrix::from_diagonal(&[var], model.eta_names.clone());
    let opts = SimulateOptions {
        seed: Some(seed),
        ..Default::default()
    };
    let dosed = simulate_with_options_diag(model, &tiny_population(), &params, n_sim, &opts)
        .expect("an in-domain point estimate simulates");
    let undosed =
        simulate_with_options_diag(model, &dose_free(&tiny_population()), &params, n_sim, &opts)
            .expect("the dose-free twin simulates");
    let (clamped, total) = dose_free_matches(&dosed.results, &undosed.results);
    (flip_flop_warning(&dosed.warnings), clamped, total)
}

// A twin-less transit form declined by `[initial_conditions]` rather than by `[scaling]`,
// in-domain at η = 0 (same ke = 0.125 < KTR = 0.2 as the M1 fixture). The init baseline
// `BASE / V = 10` is layered on the closed form's output *after* the flip-flop clamp, so a
// clamped individual's rows are the baseline's decay, not zeros (#1823 review r1).
const TWINLESS_TRANSIT_INIT_SRC: &str = "\
[parameters]
  theta TVCL(0.5, 0.001, 50.0)
  theta TVV(4.0, 0.1, 500.0)
  theta TVNTR(3.0, 0.0, 20.0)
  theta TVMTT(20.0, 0.05, 200.0)
  theta TVBASE(40.0, 0.0, 1000.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV
  NTR = TVNTR
  MTT = TVMTT
  BASE = TVBASE

[initial_conditions]
  init(central) = BASE

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)

[error_model]
  DV ~ proportional(PROP)
";

/// The M1 fixture's `[scaling] obs_scale = 1` divisor replaced by a **Form-C** readout that
/// adds a constant: `apply_analytic_readout` replaces the output after the clamp, so a
/// clamped individual reads 2.5, not 0.
fn twinless_transit_form_c_src() -> String {
    INDOMAIN_TWINLESS_TRANSIT_SRC.replace(
        "[scaling]\n  obs_scale = 1\n",
        "[scaling]\n  y = central/V + 2.5\n",
    )
}

/// The same fixture with a Form-C readout that **preserves** zero: the readout sees the
/// central *amount*, so `central / V` recovers the concentration exactly and a clamped
/// individual still reads `0`.
///
/// Structurally indistinguishable from [`twinless_transit_form_c_src`] —
/// `analytic_readout.is_some()` for both — and the opposite row value. This pair is why the
/// message makes no claim about what a clamped row contains (#1823 review r2).
fn twinless_transit_form_c_zero_preserving_src() -> String {
    INDOMAIN_TWINLESS_TRANSIT_SRC.replace(
        "[scaling]\n  obs_scale = 1\n",
        "[scaling]\n  y = central/V\n",
    )
}

/// The M1 fixture under log-transform-both-sides: `apply_log_transform` maps the clamped
/// `0` to `ln(LTBS_FLOOR)`, a large negative number, so again not a zero.
fn twinless_transit_ltbs_src() -> String {
    INDOMAIN_TWINLESS_TRANSIT_SRC.replace("  DV ~ proportional(PROP)", "  log(DV) ~ additive(PROP)")
}

/// #1823 review r2: the message **claims nothing about what a clamped individual's rows
/// contain**, because no structural predicate can know.
///
/// Round 1 replaced an unconditional "identically-zero profile" with a gate on
/// `analytical_init` / `analytic_readout` / `log_transform`. Round 2 showed that gate is
/// unfixable in principle: the two Form-C readouts `y = central/V` and
/// `y = central/V + 2.5` are the *same* gate input (`analytic_readout.is_some()`) and give
/// opposite row values — 27 zero rows and 0 zero rows on the same seed. A fourth
/// post-clamp layer, `apply_frem_prediction_override` (`pk/mod.rs:3156`), was outside the
/// gate entirely. So the message now states the clamp (certain, structural) and tells the
/// reader that searching for a zero row will not find these individuals.
///
/// This test pins that: across five fixtures whose zero-row counts span 0 and 27, the
/// count is right on every one (verified by the **dose-free twin**, not by looking for
/// zeros) and the message says the same thing on every one. The spread in the last column
/// is the evidence that no value claim would be safe.
///
/// | fixture | post-clamp layer | clamped | rows reading 0 |
/// |---|---|---|---|
/// | `[scaling] obs_scale = 1` | none (a divisor; zero stays zero) | 27 | 27 |
/// | `y = central/V` | Form-C readout, zero-preserving | 27 | 27 |
/// | `y = central/V + 2.5` | Form-C readout, additive | 27 | 0 |
/// | `init(central) = BASE` | `[initial_conditions]` baseline | 27 | 0 |
/// | `log(DV) ~ additive(PROP)` | LTBS log transform | 27 | 0 |
#[test]
fn simulated_flip_flop_warning_claims_nothing_about_the_row_values() {
    let plain = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let form_c_zero = parse_fixture(&twinless_transit_form_c_zero_preserving_src());
    let form_c_add = parse_fixture(&twinless_transit_form_c_src());
    let init = parse_fixture(TWINLESS_TRANSIT_INIT_SRC);
    let ltbs = parse_fixture(&twinless_transit_ltbs_src());

    let cases = [
        ("obs_scale", &plain, 27usize),
        ("form_c_zero_preserving", &form_c_zero, 27),
        ("form_c_additive", &form_c_add, 0),
        ("init_baseline", &init, 0),
        ("ltbs", &ltbs, 0),
    ];
    for (name, m, want_zero_rows) in cases {
        assert!(m.absorption_ode_equivalent.is_none(), "{name}: twin-less");
        assert!(
            check_absorption_flip_flop_no_twin(m, &tiny_population(), &m.default_params.theta)
                .is_none(),
            "{name}: the point estimate is in-domain"
        );
        // The count, verified against the dose-free twin — the oracle that holds whatever
        // the rows contain.
        let (msg, clamped, total) = clamped_run(m, 0.09, 300, 7);
        assert_eq!((clamped, total), (27, 600), "{name}: measured at afe7f52b");
        // …and the rows themselves, which is where the fixtures disagree.
        let (zeroed, _) = zeroed_individuals(&flip_flop_rows(m, 0.09, 300, 7));
        assert_eq!(
            zeroed, want_zero_rows,
            "{name}: measured at afe7f52b — the spread across fixtures is the point"
        );

        let msg = msg.expect("a clamped individual must be reported");
        // The four claims the message makes, all of them true on every fixture above.
        assert!(
            msg.contains(&format!(
                "{clamped} of {total} simulated individuals dosed before one of their \
                 observations"
            )),
            "{name}: {msg}"
        );
        assert!(
            msg.contains(
                "closed form has no solution and clamps its dose-driven concentration to zero"
            ),
            "{name}: {msg}"
        );
        assert!(
            msg.contains(
                "None of their rows carries any contribution from the doses the subject was \
                 given"
            ),
            "{name}: {msg}"
        );
        // …and it must not link the two with a causal "so": for an individual whose
        // absorption never reaches an observation the rows are undosed because the dose
        // never arrived, not because anything was clamped (#1823 review r9).
        assert!(
            !msg.contains("so their entire dose-driven concentration was clamped"),
            "{name}: the clamp is not the cause for every counted individual: {msg}"
        );
        assert!(
            msg.contains(
                "every VPC band, prediction interval or mean computed over them describes an \
                 undosed subject"
            ),
            "{name}: {msg}"
        );
        // …and it must not call those rows drug-free: on the `[initial_conditions]`
        // fixture the baseline `init(central) = BASE` *is* drug, and it survives the
        // clamp. "Undosed" is the claim the dose-free oracle actually supports
        // (#1823 review r5).
        for drug_free in ["no drug in it", "drug-free", "contributes nothing to any"] {
            assert!(
                !msg.contains(drug_free),
                "{name}: the rows are undosed, not drug-free ({drug_free:?}): {msg}"
            );
        }
        assert!(
            msg.contains("searching the output for a zero row will not reliably find them"),
            "{name}: {msg}"
        );
        // And nothing that predicts the row values, in either direction.
        for banned in [
            "identically zero",
            "identically-zero",
            "they are zeros",
            "`ipred = 0`.",
            "cannot be found by looking for one",
            "simulated DV is 0",
            // …and nothing asserting the clamp *moved* anything: false for a `lagtime`
            // past the last observation or `f = 0` (#1823 review r3).
            "pulled toward",
        ] {
            assert!(
                !msg.contains(banned),
                "{name}: the message must not predict the row value ({banned:?}): {msg}"
            );
        }
    }
    // The decisive pair, stated as its own assertion so the structural
    // indistinguishability cannot quietly stop being true.
    assert_eq!(
        (
            form_c_zero.analytic_readout.is_some(),
            form_c_add.analytic_readout.is_some()
        ),
        (true, true),
        "both Form-C fixtures present identically to any structural predicate"
    );
}

/// #1823 review r2: a subject carrying no positive dose amount is **not** counted — on
/// either side of the ratio.
///
/// Its profile is drug-free with the clamp and without it, so nothing was lost and an ODE
/// rewrite would produce the same rows; counting it would attach "the drug contributes
/// nothing to any of their rows" to a placebo arm and inflate the denominator. Measured
/// before the fix: an `AMT = 0` second subject pushed the count to 27 of 600 while all 900
/// of its own rows were already zero.
///
/// Both sides in one test: the dosed subject's crossers are still counted, the placebo
/// subject's are not, and a population of placebo subjects alone reports nothing at all.
#[test]
fn simulated_flip_flop_tally_skips_an_undosed_subject() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let theta = model.default_params.theta.clone();
    let dosed = tiny_population();
    let placebo = dose_free(&dosed);
    let crosser = [0.6_f64]; // CL/V = 0.5·e^0.6/4 = 0.228 ≥ KTR = 0.2

    // One dosed crosser and one placebo crosser, same η: only the dosed one counts.
    let mut mixed = SimulatedFlipFlopTally::default();
    mixed.observe(&model, &dosed.subjects[0], &theta, &crosser);
    mixed.observe(&model, &placebo.subjects[0], &theta, &crosser);
    let msg = mixed.warning(&model).expect("the dosed crosser is counted");
    assert!(
        msg.contains("1 of 1 simulated individuals dosed before one of their observations"),
        "the placebo individual counts on neither side: {msg}"
    );

    // Placebo subjects only — nothing to report, however far their η crosses.
    let mut placebo_only = SimulatedFlipFlopTally::default();
    for s in &placebo.subjects {
        placebo_only.observe(&model, s, &theta, &crosser);
    }
    assert_eq!(placebo_only.warning(&model), None);

    // End to end, through `simulate_with_options_diag`: a population whose second subject
    // is a placebo reports a denominator of 300 rather than 600 — only S1's 300 replicates
    // are dosed individuals.
    let mut half = tiny_population();
    half.subjects[1].doses[0].amt = 0.0;
    let mut params = model.default_params.clone();
    params.omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let opts = SimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let out = simulate_with_options_diag(&model, &half, &params, 300, &opts).expect("sim");
    let reported = flip_flop_warning(&out.warnings).expect("the dosed subject still crosses");
    // S1 is dosed and S2 is not, so the denominator is 300, not 600 — and the placebo
    // subject's own 900 rows are all zero whether or not it crossed, which is exactly why
    // counting it would have been meaningless.
    assert!(
        reported.contains("of 300 simulated individuals dosed before one of their observations"),
        "{reported}"
    );
    assert_eq!(
        out.results
            .iter()
            .filter(|r| r.id == "S2" && r.ipred == 0.0)
            .count(),
        900,
        "every placebo row is zero, crossing or not"
    );
}

/// #1823 review r3 / r4: a dose that cannot reach any returned row is not counted either —
/// and for a transit / IG absorption "cannot reach" includes landing **on** the last
/// observation, not only after it.
///
/// Measured (r3): with observations at t = 1, 2, 3 and the dose at t = 100 the prediction is
/// `[0, 0, 0]` with the dose and `[0, 0, 0]` without it, at a crossing η *and* at η = 0 — so
/// the clamp changed nothing, and before this gate the individual was counted anyway.
///
/// Measured (r4): the same holds for a dose exactly **at** t = 3, because a transit or IG
/// absorption contributes exactly zero central concentration at elapsed time zero — nothing
/// has transited yet. Moving it `1e-9` earlier gives `1.67e-39`: tiny, but not zero, so
/// `d.time < last_obs` sits on the right side of the boundary. (`<=` would be right for an
/// IV bolus; this family is oral.)
///
/// The oracle is applied **at η = 0**, deliberately. At a crossing η the with-dose and
/// dose-free predictions are identical for *every* dose time — that is the clamp itself — so
/// an identity asserted there says nothing about whether the dose could have contributed.
#[test]
fn simulated_flip_flop_tally_skips_a_dose_that_reaches_no_observation() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let ig = parse_fixture(INDOMAIN_TWINLESS_IG_SRC);
    let base = tiny_population();
    let last_obs = *base.subjects[0].obs_times.last().expect("an obs grid");
    let at = |t: f64| {
        let mut s = base.subjects[0].clone();
        s.doses[0].time = t;
        s
    };
    let undosed = |s: &Subject| {
        let mut s = s.clone();
        s.doses[0].amt = 0.0;
        s
    };

    // The oracle, at η = 0, on both families: a dose at or after the last observation
    // contributes nothing; one just before it does.
    for (fam, m, crosser) in [("transit", &model, 0.6_f64), ("ig", &ig, 2.3)] {
        let theta = m.default_params.theta.clone();
        for (name, t, reaches) in [
            ("long after the last obs", 100.0, false),
            ("exactly on the last obs", last_obs, false),
            ("just before the last obs", last_obs - 1e-9, true),
            ("before the first obs", 0.0, true),
        ] {
            let s = at(t);
            let with = crate::pk::compute_predictions_with_tv(m, &s, &theta, &[0.0]);
            let without = crate::pk::compute_predictions_with_tv(m, &undosed(&s), &theta, &[0.0]);
            assert_eq!(
                with != without,
                reaches,
                "{fam}, dose {name} (t = {t}): expected the dose to reach a row = {reaches} \
                 (with = {with:?}, without = {without:?})"
            );
            // …and at the crossing η the two are identical whatever the dose time, which is
            // why the oracle above must not be read there.
            assert_eq!(
                crate::pk::compute_predictions_with_tv(m, &s, &theta, &[crosser]),
                crate::pk::compute_predictions_with_tv(m, &undosed(&s), &theta, &[crosser]),
                "{fam}, dose {name}: the clamp makes these identical for any dose time"
            );
            // The gate agrees with the oracle, case for case.
            let mut tally = SimulatedFlipFlopTally::default();
            tally.observe(m, &s, &theta, &[crosser]);
            assert_eq!(
                tally.warning(m).is_some(),
                reaches,
                "{fam}, dose {name}: the gate must follow the oracle"
            );
        }
    }

    // `any`, not `all`: one qualifying dose among several is enough. Every other fixture
    // carries exactly one dose, which makes the two spellings indistinguishable — measured
    // in the #1823 r3 sweep, where `.any` → `.all` survived the whole suite until this
    // case was added. A two-dose subject discriminates in both directions.
    let theta = model.default_params.theta.clone();
    let crosser = [0.6_f64];
    let two_doses = |a: f64, b: f64| {
        let mut s = base.subjects[0].clone();
        s.doses = vec![
            DoseEvent::new(a, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(b, 100.0, 1, 0.0, false, 0.0),
        ];
        s.dose_occasions = vec![1, 1];
        s
    };
    for (name, a, b, want) in [
        ("one in range, one late", 0.0, 100.0, true),
        ("one late, one in range", 100.0, last_obs - 0.5, true),
        ("both late", 100.0, 200.0, false),
        ("one on the boundary, one after", last_obs, 100.0, false),
    ] {
        let mut tally = SimulatedFlipFlopTally::default();
        tally.observe(&model, &two_doses(a, b), &theta, &crosser);
        assert_eq!(
            tally.warning(&model).is_some(),
            want,
            "two doses, {name}: expected counted = {want}"
        );
    }

    // A subject with no observations at all is likewise not counted: there is no row for
    // the clamp to have affected.
    let mut no_obs = base.subjects[0].clone();
    no_obs.obs_times.clear();
    no_obs.obs_raw_times.clear();
    no_obs.observations.clear();
    no_obs.obs_cmts.clear();
    no_obs.cens.clear();
    no_obs.occasions.clear();
    let mut tally = SimulatedFlipFlopTally::default();
    tally.observe(&model, &no_obs, &theta, &crosser);
    assert_eq!(tally.warning(&model), None);
}

/// #1823: the count the warning reports is exactly the number of simulated individuals
/// whose profile came back identically zero — on the measured fixture from the issue, and
/// on its twin-carrying control, where both are zero.
///
/// Measured on this branch at `afe7f52b`: 27 of 600 individuals (4.5%) at ω² = 0.09 /
/// n_sim = 300 / seed 7, and 260 of 4000 (6.5%) at n_sim = 2000 / seed 42. Pre-fix every
/// one of those individuals contributed three finite `ipred = 0.0` rows and no warning on
/// any channel.
#[test]
fn simulated_flip_flop_count_equals_the_zeroed_individuals() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    assert!(model.absorption_ode_equivalent.is_none(), "twin-less");
    // The point estimate is in-domain, so nothing refuses this run and the η = 0 gates are
    // all silent — the crossing is the draw's, not θ's.
    assert!(check_absorption_flip_flop_no_twin(
        &model,
        &tiny_population(),
        &model.default_params.theta
    )
    .is_none());

    for (n_sim, seed, want) in [(300usize, 7u64, 27usize), (2000, 42, 260)] {
        let (msg, zeroed, total) = flip_flop_run(&model, 0.09, n_sim, seed, None);
        assert_eq!(total, 2 * n_sim, "two subjects per replicate");
        assert_eq!(zeroed, want, "measured at afe7f52b (n_sim={n_sim})");
        assert!(zeroed > 0, "the fixture must produce crossers to be a test");
        let msg = msg.expect("a crossing individual must be reported");
        assert!(
            msg.contains(&format!(
                "{zeroed} of {total} simulated individuals dosed before one of their \
                 observations"
            )),
            "count must equal the zeroed individuals: {msg}"
        );
        // …and the dose-free twin agrees, which is the oracle that also holds on a
        // model where the clamp does not read as a zero (#1823 review r1).
        let (_, clamped, clamped_total) = clamped_run(&model, 0.09, n_sim, seed);
        assert_eq!((clamped, clamped_total), (zeroed, total));
    }

    // The twin-carrying control: the same model without the inert `[scaling]`, at ω² = 4 —
    // 44× the variance above — reroutes every crosser to its ODE `transit()` twin, so no
    // individual is zeroed and nothing is reported.
    let twin = parse_fixture(PLAIN_TWIN_TRANSIT_SRC);
    assert!(twin.absorption_ode_equivalent.is_some(), "twin-carrying");
    let (msg, zeroed, total) = flip_flop_run(&twin, 4.0, 300, 7, None);
    assert_eq!((zeroed, total), (0, 600), "the twin serves both regimes");
    assert_eq!(msg, None, "a twin-carrying model has nothing to report");
}

/// #1823: the propensity-matched branch of `simulate_with_options_diag` builds its own
/// `eta_slice` (a pool draw reassigned by Mahalanobis matching), so it carries its own
/// `observe` call. Deleting *that* call must redden a test — this one — and deleting the
/// unmatched branch's call must not: the two sides of the twin are mutated separately.
#[test]
fn simulated_flip_flop_count_covers_the_propensity_matched_branch() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let (msg, zeroed, total) = flip_flop_run(
        &model,
        0.09,
        300,
        7,
        Some(crate::propensity_match::MatchMethod::Optimal),
    );
    assert!(
        zeroed > 0,
        "the matched run must produce crossers or this proves nothing"
    );
    assert_eq!(total, 600);
    let msg = msg.expect("a matched run reports its crossers too");
    assert!(
        msg.contains(&format!(
            "{zeroed} of {total} simulated individuals dosed before one of their observations"
        )),
        "{msg}"
    );
    // Not compared against the unmatched run's count: the matched branch draws a whole
    // replicate's η pool before emitting any rows, so the two branches consume the RNG in a
    // different order and their pools differ from the second subject on.
}

/// #1823: every sentence of the warning, on all three absorption families, with both sides
/// of each wording gate asserted in the same test — transit vs IG (domain + ODE rewrite)
/// and 1-cpt vs 2-cpt (the confluent-eigenvalue clause). Mirrors
/// `uncertainty_skip_warning_wording_per_family`, which gates on the same two helpers, so
/// three fixtures kill both gates in both directions (the two predicates are independent:
/// one keys on IG-vs-transit, the other on 1-cpt-vs-2-cpt).
///
/// Deleting any one of the three sentences must redden this test; that is the point of
/// asserting each separately rather than matching the whole string.
#[test]
fn simulated_flip_flop_warning_wording_per_family() {
    let transit = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let ig = parse_fixture(INDOMAIN_TWINLESS_IG_SRC);
    let transit2 = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_2CPT_SRC);
    let ig2 = parse_fixture(INDOMAIN_TWINLESS_IG_2CPT_SRC);
    // All four `PkModel` variants the family predicate enumerates, so dropping any one of
    // them from it reddens here (#1823 review r6 — before `ig2` existed, dropping
    // `TwoCptIg` left the suite green).
    assert_eq!(
        [
            transit.pk_model,
            ig.pk_model,
            transit2.pk_model,
            ig2.pk_model
        ],
        [
            PkModel::OneCptTransit,
            PkModel::OneCptIg,
            PkModel::TwoCptTransit,
            PkModel::TwoCptIg
        ]
    );

    // ω² chosen per family so each produces crossers: the IG abscissa sits at 0.833 (vs
    // ke = 0.1) and the 2-cpt α at 0.4 (vs 0.285) / 0.228 (vs 0.833), so they need a wider
    // draw than the 1-cpt transit fixture's 0.2-vs-0.125.
    let (t1, t1_zeroed, _) = flip_flop_run(&transit, 0.09, 300, 7, None);
    let (ig1, ig1_zeroed, _) = flip_flop_run(&ig, 4.0, 300, 7, None);
    let (t2, t2_zeroed, _) = flip_flop_run(&transit2, 1.0, 300, 7, None);
    let (ig2m, ig2_zeroed, _) = flip_flop_run(&ig2, 4.0, 300, 7, None);
    let t1 = t1.expect("transit crossers");
    let ig1 = ig1.expect("IG crossers");
    let t2 = t2.expect("2-cpt transit crossers");
    let ig2m = ig2m.expect("2-cpt IG crossers");

    for (msg, zeroed, name) in [
        (&t1, t1_zeroed, "one_cpt_transit"),
        (&ig1, ig1_zeroed, "one_cpt_ig"),
        (&t2, t2_zeroed, "two_cpt_transit"),
        (&ig2m, ig2_zeroed, "two_cpt_ig"),
    ] {
        // Sentence 1 — the count, the cause, and the consequence for `ipred`.
        assert!(
            msg.contains(&format!(
                "{zeroed} of 600 simulated individuals dosed before one of their observations"
            )),
            "{name}: {msg}"
        );
        assert!(msg.contains("flip-flop regime"), "{name}: {msg}");
        assert!(
            msg.contains(&format!("twin-less {name} closed form has no solution")),
            "{name}: {msg}"
        );
        assert!(
            msg.contains(
                "closed form has no solution and clamps its dose-driven concentration to zero"
            ),
            "{name}: {msg}"
        );
        // Sentence 2 — the consequence for anything computed over those rows.
        assert!(
            msg.contains("Those rows are still returned and still summarised"),
            "{name}: {msg}"
        );
        assert!(
            msg.contains("describes an undosed subject"),
            "{name}: {msg}"
        );
        // Sentence 3 — why a zero row is not the way to find them
        // (`..._claims_nothing_about_the_row_values` is what makes this sentence necessary).
        assert!(
            msg.contains("searching the output for a zero row will not reliably find them"),
            "{name}: {msg}"
        );
        // Sentence 4 — the way out, and that it is per individual.
        assert!(
            msg.contains("which reroutes per individual and is valid in both regimes"),
            "{name}: {msg}"
        );
        // It must not claim the starting estimates are the problem (the η = 0 refusal's
        // advice, which is false on a fixture whose point estimate is in-domain).
        assert!(!msg.contains("starting estimates"), "{name}: {msg}");
        // The token carries the category, not the "flip-flop regime" prose.
        let entry = crate::types::classify_warning(msg);
        assert_eq!(entry.category, crate::types::WarningCode::FlipFlop);
        assert_eq!(entry.severity, crate::types::WarningSeverity::Warning);
    }
    // Transit vs IG: the domain bound and the ODE rewrite, on both compartment counts.
    for m in [&t1, &t2] {
        assert!(m.contains("transit rate KTR = (n+1)/mtt"), "{m}");
        assert!(m.contains("explicit ODE `transit()` model"), "{m}");
        assert!(!m.contains("igd()") && !m.contains("MAT"), "{m}");
    }
    for m in [&ig1, &ig2m] {
        assert!(m.contains("disposition rate ≥ 1/(2·MAT·CV²)"), "{m}");
        assert!(m.contains("explicit ODE `igd()` model"), "{m}");
        assert!(!m.contains("KTR") && !m.contains("transit()"), "{m}");
    }
    // 1-cpt vs 2-cpt: only a 2-cpt model has a confluent-eigenvalue edge — on both families.
    for m in [&t2, &ig2m] {
        assert!(
            m.contains("or with coincident disposition eigenvalues"),
            "{m}"
        );
    }
    for m in [&t1, &ig1] {
        assert!(!m.contains("coincident"), "{m}");
    }
}

/// #1823: on a model whose ODE twin was built and then declined (#1008) the tally carries
/// the decline clause — the easier fix than the rewrite it just suggested. The `[scaling]`
/// fixture, whose twin was never built, must not carry it. Both sides in one test, so a
/// clause stuck on either branch reddens it.
#[test]
fn simulated_flip_flop_warning_carries_the_twin_decline_clause() {
    let declined = parse_fixture(&declined_twin_transit_src());
    assert!(declined.absorption_ode_equivalent.is_none(), "twin-less");
    assert!(
        crate::types::absorption_twin_decline_reason(&declined).is_some(),
        "the twin was built and declined"
    );
    let scaling = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    assert!(crate::types::absorption_twin_decline_reason(&scaling).is_none());

    let clause = "its twin was built and then rejected by its own parse";
    let (with, with_zeroed, _) = flip_flop_run(&declined, 0.09, 300, 7, None);
    let (without, without_zeroed, _) = flip_flop_run(&scaling, 0.09, 300, 7, None);
    assert!(with_zeroed > 0 && without_zeroed > 0, "both must cross");
    let with = with.expect("declined-twin crossers");
    let without = without.expect("scaling crossers");
    assert!(with.contains(clause), "{with}");
    assert!(
        with.contains("CENTRAL"),
        "the twin parser's own reason is quoted: {with}"
    );
    assert!(!without.contains(clause), "{without}");
}

/// #1823: the `W_SIMULATED_FLIP_FLOP` token arm sits ahead of every prose arm because the
/// message quotes the twin parser's own error text verbatim (#1008), and a decline reason
/// that happens to say "ill-conditioned" would otherwise be claimed by the
/// `condition_number` arm and come back `Critical` — a VPC consumer branching on that would
/// read an estimation failure where the finding is a degenerate simulated profile. Mirrors
/// `classify_warning_twin_decline_beats_prose_arms_in_the_reason`, and is what makes the
/// token arm more than a duplicate of the `flip-flop regime` prose arm (which classifies
/// the plain message correctly on its own).
#[test]
fn simulated_flip_flop_warning_classifies_on_its_token_not_the_reason() {
    let mut model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    model
        .parse_warnings
        .push(crate::types::absorption_twin_declined_warning(
            "[odes]: the Jacobian is ill-conditioned at the initial state",
        ));
    let pop = tiny_population();
    let theta = model.default_params.theta.clone();
    let mut tally = SimulatedFlipFlopTally::default();
    // CL/V = 0.5·e^0.6/4 = 0.228 ≥ KTR = 0.2.
    tally.observe(&model, &pop.subjects[0], &theta, &[0.6]);
    let msg = tally.warning(&model).expect("a crossing individual");
    assert!(
        msg.contains("ill-conditioned"),
        "the reason is quoted: {msg}"
    );
    let entry = crate::types::classify_warning(&msg);
    assert_eq!(entry.category, crate::types::WarningCode::FlipFlop, "{msg}");
    assert_eq!(entry.severity, crate::types::WarningSeverity::Warning);
}

/// #1823: no crossing individual, no warning — on a narrow ω where every draw stays
/// in-domain (the straddle partner of the wide-ω runs above), on a twin-carrying model at a
/// ω 44× wider, and on a non-absorption model, which has no flip-flop regime at all.
#[test]
fn simulated_flip_flop_warning_silent_without_a_crosser() {
    for (name, model, var) in [
        (
            "narrow ω",
            parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC),
            1e-4,
        ),
        ("twin", parse_fixture(PLAIN_TWIN_TRANSIT_SRC), 4.0),
        ("non-absorption", tiny_model(), 4.0),
    ] {
        let (msg, zeroed, total) = flip_flop_run(&model, var, 300, 7, None);
        assert_eq!((zeroed, total), (0, 600), "{name}");
        assert_eq!(msg, None, "{name}");
    }
}

/// #1823: a wrong-length η is skipped rather than read past the end inside `pk_param_fn`.
/// Production cannot deliver one (`simulate_inner_with_draw` resizes every `eta_slice` to
/// `n_eta + n_kappa`), so the guard is pinned on the helper directly — a would-be crosser
/// at the wrong shape must not be counted on either side of the ratio.
#[test]
fn simulated_flip_flop_tally_skips_a_wrong_length_eta() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let theta = &model.default_params.theta;
    let crosser = [0.6_f64]; // CL/V = 0.5·e^0.6/4 = 0.228 ≥ KTR = 0.2
    let mut tally = SimulatedFlipFlopTally::default();
    tally.observe(&model, &pop.subjects[0], theta, &crosser);
    tally.observe(&model, &pop.subjects[1], theta, &[]);
    tally.observe(&model, &pop.subjects[1], theta, &[0.6, 0.6]);
    let msg = tally
        .warning(&model)
        .expect("the well-shaped crosser is counted");
    assert!(
        msg.contains("1 of 1 simulated individuals dosed before"),
        "the two mis-shaped etas count on neither side: {msg}"
    );
}

/// #1823: on `simulate_with_uncertainty_diag` the per-individual tally and the #1485
/// per-draw skip are two different findings and both are reported on one run. The skip
/// counts θ draws the closed form cannot serve at all; the tally counts individuals inside
/// the draws that *did* run, so a skipped draw contributes to neither side of its ratio —
/// which is why the denominator is the surviving draws' individuals, not 2 × n_draws.
///
/// Measured at `afe7f52b`: 172 of 2000 draws skipped, and 365 of 3656 individuals zeroed
/// across the 1828 draws that ran (1095 of the 10968 returned rows).
#[test]
fn uncertainty_reports_the_tally_alongside_the_skipped_draws() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let mut fit = synthetic_fit(&model.default_params);
    if let Some(cov) = fit.covariance_matrix.as_mut() {
        cov[(0, 0)] = 0.09;
    }
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 2000,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(42),
    };
    let out = simulate_with_uncertainty_diag(&model, &pop, &fit, &opts).expect("some draws run");
    let ran: std::collections::BTreeSet<usize> = out.results.iter().map(|r| r.draw).collect();
    assert_eq!(ran.len(), 1828, "measured at afe7f52b");
    // Both findings, in this order: the coarser per-draw skip first.
    let skip = only_skip_warning(&out.warnings);
    assert!(skip.contains("172 of 2000 uncertainty draws"), "{skip}");
    let (zeroed, total) = zeroed_individuals(&out.results);
    assert_eq!((zeroed, total), (365, 3656), "measured at afe7f52b");
    assert_eq!(out.results.len(), 10968, "3 rows per individual");
    let msg = flip_flop_warning(&out.warnings).expect("the tally is reported here too");
    assert!(
        msg.contains("365 of 3656 simulated individuals"),
        "the denominator is the individuals that ran, not 2 × n_draws: {msg}"
    );
    let skip_at = out.warnings.iter().position(|w| w == skip).unwrap();
    let tally_at = out.warnings.iter().position(|w| *w == msg).unwrap();
    assert!(skip_at < tally_at, "{:#?}", out.warnings);
}

/// #1823: `simulate` / `simulate_with_seed` have no diagnostics channel, so the tally is
/// counted at the shared chokepoint and dropped — their rows are the `_diag` form's rows,
/// bit for bit, on the same seed. Pins that the fix did not quietly grow a second row
/// implementation, and that the row-only contract
/// (`the_row_only_entry_points_stay_silent_by_contract`) still holds for a run that *does*
/// have something to report.
#[test]
fn row_only_simulate_still_returns_the_diag_rows_when_individuals_cross() {
    let model = parse_fixture(INDOMAIN_TWINLESS_TRANSIT_SRC);
    let pop = tiny_population();
    let mut params = model.default_params.clone();
    params.omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let opts = SimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let diag = simulate_with_options_diag(&model, &pop, &params, 300, &opts).expect("diag");
    assert!(
        flip_flop_warning(&diag.warnings).is_some(),
        "the fixture must report something, or 'the rows are the same' is vacuous"
    );
    let rows = simulate_with_seed(&model, &pop, &params, 300, 7).expect("rows");
    assert_eq!(rows.len(), diag.results.len());
    for (a, b) in rows.iter().zip(&diag.results) {
        assert_eq!((a.draw, a.sim, &a.id), (b.draw, b.sim, &b.id));
        assert_eq!(a.ipred.to_bits(), b.ipred.to_bits());
        assert_eq!(format!("{:?}", a.outcome), format!("{:?}", b.outcome));
    }
    let (zeroed, _) = zeroed_individuals(&rows);
    assert!(zeroed > 0, "the row-only form returns the zeroed rows too");
}

#[test]
fn asymptotic_row_count_and_draw_range() {
    let model = tiny_model();
    let pop = tiny_population();
    let fit = synthetic_fit(&model.default_params);

    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 2,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(7),
    };
    let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
    // 3 draws * 2 sims * 2 subjects * 3 obs = 36 rows
    assert_eq!(rows.len(), 36);

    let mut draws: Vec<usize> = rows.iter().map(|r| r.draw).collect();
    draws.sort();
    draws.dedup();
    assert_eq!(draws, vec![1, 2, 3]);

    let mut sims: Vec<usize> = rows.iter().map(|r| r.sim).collect();
    sims.sort();
    sims.dedup();
    assert_eq!(sims, vec![1, 2]);
}

#[test]
fn legacy_simulate_emits_draw_one() {
    // The original simulate() path should tag every row with draw = 1,
    // preserving a sensible default for callers that don't propagate
    // parameter uncertainty.
    let model = tiny_model();
    let pop = tiny_population();
    let rows = simulate_with_seed(&model, &pop, &model.default_params, 2, 42).unwrap();
    assert!(rows.iter().all(|r| r.draw == 1));
}

#[test]
fn compute_npde_npd_shapes_finite_and_reproducible() {
    // Drives the full post-fit NPDE/NPD simulation path on a real (model,
    // population, params). nsim > n_obs (3) so the decorrelated NPDE is
    // non-NaN, and a fixed seed must reproduce bit-for-bit.
    let model = tiny_model();
    let pop = tiny_population();
    let nsim = 200;

    let a = crate::stats::npde::compute_npde_npd(
        &model,
        &pop,
        &model.default_params,
        nsim,
        Some(7),
        None,
    )
    .expect("npde");
    let b = crate::stats::npde::compute_npde_npd(
        &model,
        &pop,
        &model.default_params,
        nsim,
        Some(7),
        None,
    )
    .expect("npde");

    assert_eq!(a.len(), pop.subjects.len());
    for (sn, subj) in a.iter().zip(pop.subjects.iter()) {
        assert_eq!(sn.npd.len(), subj.observations.len());
        assert_eq!(sn.npde.len(), subj.observations.len());
        assert!(sn.npd.iter().all(|v| v.is_finite()), "NPD finite");
        assert!(sn.npde.iter().all(|v| v.is_finite()), "NPDE finite");
    }
    // Reproducible across calls with the same seed.
    for (sa, sb) in a.iter().zip(b.iter()) {
        assert_eq!(sa.npd, sb.npd);
        assert_eq!(sa.npde, sb.npde);
    }
}

#[test]
fn fit_rejects_ltbs_with_proportional_error() {
    // Defensive guard against hand-built `CompiledModel`s: the parser already
    // rejects this combination, but a Rust caller flipping `log_transform = true`
    // on a model with proportional/combined error would silently mis-fit (the
    // prediction is log-wrapped but the variance still expects natural-scale f).
    let mut model = tiny_model(); // tiny_model uses Proportional error
    model.log_transform = true;
    let pop = tiny_population();
    let opts = FitOptions::default();
    let err = fit(&model, &pop, &model.default_params, &opts)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("LTBS") && err.contains("Additive"),
        "expected LTBS+proportional rejection, got: {err}"
    );
}

#[test]
fn diagnostic_details_covers_numeric_codes() {
    use crate::types::WarningCode;
    // Shim for the scalar codes (no ETA/eigenvalue inputs).
    fn dd(
        code: &crate::types::WarningCode,
        dw: f64,
        lag1: f64,
        eps: f64,
        cond: Option<f64>,
    ) -> Option<serde_json::Value> {
        super::diagnostic_details(
            code,
            &super::DiagStats {
                dw_statistic: dw,
                iwres_lag1_r: lag1,
                shrinkage_eps: eps,
                cov_condition_number: cond,
                ..Default::default()
            },
        )
    }
    // Durbin–Watson: both stored stats surface.
    let d = dd(&WarningCode::DwAutocorrelation, 1.2, 0.4, f64::NAN, None).expect("DW details");
    assert_eq!(d["durbin_watson"], 1.2);
    assert_eq!(d["iwres_lag1_autocorr"], 0.4);
    // EPS shrinkage: fraction + percent.
    let d = dd(&WarningCode::EpsShrinkage, f64::NAN, 0.0, -0.083, None).expect("eps details");
    assert_eq!(d["eps_shrinkage"], -0.083);
    assert!((d["eps_shrinkage_pct"].as_f64().unwrap() + 8.3).abs() < 1e-9);
    // Condition number: from the Option.
    let d = dd(
        &WarningCode::ConditionNumber,
        f64::NAN,
        0.0,
        f64::NAN,
        Some(1.4e6),
    )
    .expect("cond details");
    assert_eq!(d["condition_number"], 1.4e6);
    // None cases: missing stat, non-finite stat, and non-numeric code.
    assert!(dd(&WarningCode::ConditionNumber, 0.0, 0.0, 0.0, None).is_none());
    assert!(dd(
        &WarningCode::DwAutocorrelation,
        f64::INFINITY,
        0.0,
        0.0,
        None
    )
    .is_none());
    assert!(dd(&WarningCode::Convergence, 1.0, 0.0, 0.0, Some(1.0)).is_none());

    // Non-finite *contributing* stats omit details rather than emit a
    // null-valued payload (serde_json maps non-finite floats to null).
    // cov_condition_number = +Inf (documented for near-singular spaces):
    assert!(dd(
        &WarningCode::ConditionNumber,
        f64::NAN,
        0.0,
        f64::NAN,
        Some(f64::INFINITY)
    )
    .is_none());
    // finite Durbin-Watson but non-finite lag-1 autocorrelation:
    assert!(dd(
        &WarningCode::DwAutocorrelation,
        1.20,
        f64::NAN,
        f64::NAN,
        None
    )
    .is_none());
}

#[test]
fn eta_shrinkage_warning_fires_above_threshold_only() {
    let names = vec![
        "eta_CL".to_string(),
        "eta_V".to_string(),
        "eta_KA".to_string(),
    ];
    // All low → no warning.
    assert!(super::eta_shrinkage_warning(&[0.01, 0.05, 0.10], &names).is_none());
    // NaN is ignored (not treated as high).
    assert!(super::eta_shrinkage_warning(&[0.01, f64::NAN, 0.20], &names).is_none());
    // One above the 30% threshold → warning names it with its percent.
    let w = super::eta_shrinkage_warning(&[0.02, 0.42, 0.10], &names).expect("warning");
    assert!(w.contains("eta_V (42%)"), "got: {w}");
    assert!(w.to_lowercase().contains("eta shrinkage"));
    // Boundary: exactly at threshold counts.
    assert!(super::eta_shrinkage_warning(&[0.30], &["eta_CL".into()]).is_some());
}

#[test]
fn eta_shrinkage_details_lists_high_etas() {
    use crate::types::WarningCode;
    let ed = |shrink: &[f64], names: &[String]| {
        super::diagnostic_details(
            &WarningCode::EtaShrinkage,
            &super::DiagStats {
                shrinkage_eta: shrink,
                eta_names: names,
                ..Default::default()
            },
        )
    };
    let names = vec!["eta_CL".to_string(), "eta_V".to_string()];
    let d = ed(&[0.05, 0.42], &names).expect("eta details");
    assert_eq!(d["threshold_pct"], 30.0);
    let high = d["high_shrinkage_etas"].as_array().unwrap();
    assert_eq!(high.len(), 1);
    assert_eq!(high[0]["eta"], "eta_V");
    assert!((high[0]["shrinkage_pct"].as_f64().unwrap() - 42.0).abs() < 1e-9);
    // No high eta → no details payload.
    assert!(ed(&[0.05, 0.10], &names).is_none());
    // Missing name → non-empty `eta_{i}` placeholder, not "".
    let d = ed(&[0.42], &[]).expect("details");
    assert_eq!(d["high_shrinkage_etas"][0]["eta"], "eta_0");
}

#[test]
fn covariance_details_reports_eigenvalues_and_condition() {
    use crate::types::WarningCode;
    let cd = |cond: Option<f64>, eigs: Option<&[f64]>| {
        super::diagnostic_details(
            &WarningCode::CovarianceRegularized,
            &super::DiagStats {
                cov_condition_number: cond,
                cov_eigenvalues: eigs,
                ..Default::default()
            },
        )
    };
    // Regularized (or failed) with eigenvalues present: min + negative count.
    let d = cd(Some(1.4e6), Some(&[0.5, -1.0e-3, 2.1])).expect("cov details");
    assert_eq!(d["condition_number"], 1.4e6);
    assert!((d["min_eigenvalue"].as_f64().unwrap() + 1.0e-3).abs() < 1e-12);
    assert_eq!(d["n_negative_eigenvalues"], 1);
    // CovarianceFailed shares the same payload builder.
    assert!(super::diagnostic_details(
        &WarningCode::CovarianceFailed,
        &super::DiagStats {
            cov_condition_number: Some(9.9),
            ..Default::default()
        },
    )
    .is_some());
    // Nothing available (hard failure: no matrix → no eigenvalues, no cond)
    // → details omitted.
    assert!(cd(None, None).is_none());
    // Non-finite condition number is skipped.
    assert!(cd(Some(f64::INFINITY), None).is_none());
}

#[test]
fn theta_boundary_side_detects_bounds() {
    let side = |e, l, u| super::theta_boundary_side(e, l, u).map(|(s, _)| s);
    // Identity-packed (lower < 0): interior, at-lower, at-upper.
    assert_eq!(side(0.0, -1.0, 1.0), None);
    assert_eq!(side(-1.0, -1.0, 1.0), Some("lower"));
    assert_eq!(side(1.0, -1.0, 1.0), Some("upper"));
    // Identity-packed with PsN's scm bounds for a power / exponential
    // covariate θ, (-100, 1e6): an estimate near 1 is interior. Judged by
    // its distance to each bound on that bound's scale — as a fraction of
    // the range it would sit at 1e-4 and read as "at the lower bound", which
    // excluded every candidate of the first covsearch run (#1180).
    assert_eq!(side(1.05, -100.0, 1e6), None);
    assert_eq!(side(0.7, -100.0, 1e6), None);
    assert_eq!(side(-50.0, -100.0, 1e6), None);
    assert_eq!(side(-99.95, -100.0, 1e6), Some("lower"));
    assert_eq!(side(999_500.0, -100.0, 1e6), Some("upper"));
    // A bound of zero on the identity path uses an absolute tolerance.
    assert_eq!(side(-0.0005, -5.0, 0.0), Some("upper"));
    assert_eq!(side(-0.5, -5.0, 0.0), None);
    // …capped by the range: on a narrow identity interval the midpoint is
    // interior, and only the ends themselves are at a bound (review of #1238).
    assert_eq!(side(0.0, -1e-4, 1e-4), None);
    assert_eq!(side(-1e-4, -1e-4, 1e-4), Some("lower"));
    assert_eq!(side(1e-4 - 1e-8, -1e-4, 1e-4), Some("upper"));
    assert_eq!(side(0.5e-4, -1e-4, 1e-4), None);
    // Log-packed (lower >= 0): "at bound" is within a constant factor.
    assert_eq!(side(1.0, 0.001, 1000.0), None);
    assert_eq!(side(0.001, 0.001, 1000.0), Some("lower"));
    // Degenerate bounds → None.
    assert_eq!(side(1.0, 5.0, 5.0), None);
    // Effective bound: a user lower bound of 0 (log-packed) is reported as
    // the optimizer's 1e-10 floor, agreeing with the pinned estimate.
    let (s, bound) = super::theta_boundary_side(1e-10, 0.0, 100.0).unwrap();
    assert_eq!(s, "lower");
    assert_eq!(bound, 1e-10);
}

#[test]
fn high_correlation_pairs_flags_collinear_estimates() {
    let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    // 3x3 covariance: A~B correlation 0.99 (0.0099/0.01), A~C and B~C ~0.
    let mut cov = DMatrix::identity(3, 3) * 0.01;
    cov[(0, 1)] = 0.0099;
    cov[(1, 0)] = 0.0099;
    let pairs = super::high_correlation_pairs(Some(&cov), &names);
    assert_eq!(pairs.len(), 1);
    assert_eq!((pairs[0].0.as_str(), pairs[0].1.as_str()), ("A", "B"));
    assert!((pairs[0].2 - 0.99).abs() < 1e-9);
    // A fixed / zero-variance parameter (diag == 0) is excluded.
    let mut cov2 = DMatrix::identity(2, 2) * 0.01;
    cov2[(1, 1)] = 0.0;
    assert!(super::high_correlation_pairs(Some(&cov2), &names).is_empty());
    // No covariance matrix → nothing.
    assert!(super::high_correlation_pairs(None, &names).is_empty());
    // Theta-only scope: a strong correlation involving a non-theta packed
    // entry (index >= n_theta) is NOT flagged. With 2 theta names but a 3x3
    // covariance, the (0,2) correlation lies outside the theta block.
    let mut cov3 = DMatrix::identity(3, 3) * 0.01;
    cov3[(0, 2)] = 0.0099;
    cov3[(2, 0)] = 0.0099;
    let theta2 = vec!["A".to_string(), "B".to_string()];
    assert!(super::high_correlation_pairs(Some(&cov3), &theta2).is_empty());
}

#[test]
fn high_correlation_warning_emits_typed_entry_with_details() {
    use crate::types::WarningCode;
    let model = tiny_model();
    let mut fit = synthetic_fit(&model.default_params);
    // synthetic_fit's covariance is a scaled identity (no correlation) →
    // no warning.
    assert!(super::high_correlation_warning(&fit).is_none());
    // Inject a strong theta0~theta1 correlation into the packed covariance.
    let n = fit.covariance_matrix.as_ref().unwrap().nrows();
    assert!(n >= 2);
    let mut cov = DMatrix::identity(n, n) * 0.01;
    cov[(0, 1)] = 0.0098; // r = 0.98
    cov[(1, 0)] = 0.0098;
    fit.covariance_matrix = Some(cov);
    let (msg, entry) = super::high_correlation_warning(&fit).expect("expected a pair");
    assert!(msg.contains("highly correlated") || msg.contains("Highly correlated"));
    assert_eq!(entry.category, WarningCode::HighCorrelation);
    let d = entry.details.as_ref().expect("details");
    assert_eq!(d["threshold"], 0.95);
    assert!((d["pairs"][0]["correlation"].as_f64().unwrap() - 0.98).abs() < 1e-9);
    // The labels are the packed parameter names (first two thetas here).
    assert_eq!(d["pairs"][0]["parameter_a"], fit.theta_names[0].as_str());
}

#[test]
fn rebuild_native_key_reconstructs_chain_prefix() {
    let model = tiny_model();
    let mut fit = synthetic_fit(&model.default_params);
    // A native entry produced inside a `[FOCEI]` chain: message is the
    // stripped body, source_method carries the tag; the raw warning is the
    // full prefixed string.
    fit.warnings = vec!["[FOCEI] pinned to an optimizer bound: X".to_string()];
    fit.warnings_structured = vec![crate::types::WarningEntry {
        severity: crate::types::WarningSeverity::Warning,
        category: crate::types::WarningCode::BoundaryEstimate,
        message: "pinned to an optimizer bound: X".to_string(),
        source_method: Some("FOCEI".to_string()),
        details: Some(serde_json::json!({ "k": 1 })),
    }];
    super::rebuild_warnings_structured(&mut fit);
    // Native entry (with details) is preferred despite the prefix.
    assert_eq!(
        fit.warnings_structured[0].category,
        crate::types::WarningCode::BoundaryEstimate
    );
    assert_eq!(fit.warnings_structured[0].details.as_ref().unwrap()["k"], 1);
}

#[test]
fn boundary_estimate_warning_emits_typed_entry_with_details() {
    use crate::types::WarningCode;
    let mut params = tiny_model().default_params;
    // No hit initially (estimates interior) — or at least assert whatever
    // tiny_model starts with, then force a hit on theta 0 at its lower bound.
    params.theta_fixed = vec![false; params.theta.len()];
    params.theta[0] = params.theta_lower[0];
    let (msg, entry) = super::boundary_estimate_warning(&params).expect("expected a boundary hit");
    assert!(msg.contains("optimizer bound"));
    assert_eq!(entry.category, WarningCode::BoundaryEstimate);
    let d = entry.details.as_ref().expect("details");
    assert_eq!(d["parameters"][0]["side"], "lower");
    assert_eq!(
        d["parameters"][0]["parameter"],
        params.theta_names[0].as_str()
    );

    // A fixed theta at its bound is not reported.
    params.theta_fixed[0] = true;
    assert!(super::boundary_estimate_warning(&params).is_none());
}

#[test]
fn packed_guard_side_requires_an_exact_non_degenerate_hit() {
    assert_eq!(
        super::packed_guard_side(-8.0, -8.0, 5.0),
        Some(("lower", -8.0))
    );
    assert_eq!(
        super::packed_guard_side(5.0, -8.0, 5.0),
        Some(("upper", 5.0))
    );
    assert_eq!(super::packed_guard_side(5.0 - 1e-12, -8.0, 5.0), None);
    // An optimizer can hit the rail exactly in scaled space but recover a
    // packed value one ULP away through `(bound / scale) * scale`.
    let scale = 10.0_f64.ln();
    let recovered = (-8.0 / scale) * scale;
    assert_ne!(recovered, -8.0);
    assert_eq!(
        super::packed_guard_side(recovered, -8.0, 5.0),
        Some(("lower", -8.0))
    );
    assert_eq!(super::packed_guard_side(1.0, 1.0, 1.0), None);
    assert_eq!(super::packed_guard_side(f64::NAN, -8.0, 5.0), None);
}

#[test]
fn runaway_guard_warning_reports_sigma_and_skips_fixed_or_nearby_values() {
    use crate::types::WarningCode;
    let mut params = tiny_model().default_params;
    params.sigma.values.fill(1.0);
    params.sigma_fixed = vec![false; params.sigma.values.len()];

    params.sigma.values[0] = 5.0_f64.exp();
    let mut converged = true;
    let (msg, entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("sigma guard hit");
    assert!(msg.contains("implementation rail"));
    // #1118: an implementation rail cannot be an interior optimum.
    assert!(!converged, "a runaway hit is not a converged fit");
    assert!(msg.contains("Reported converged: false"));
    // The Σ ceiling is reachable by a sound model on unscaled data, so the
    // message names the remediation the generic advice does not (#1205 review).
    assert!(msg.contains("rescale DV"));
    assert_eq!(entry.severity, crate::types::WarningSeverity::Critical);
    assert_eq!(entry.category, WarningCode::ParameterAtRunawayGuard);
    let details = entry.details.as_ref().expect("details");
    assert_eq!(details["guard_space"], "packed");
    assert_eq!(details["parameters"][0]["side"], "upper");
    assert_eq!(details["parameters"][0]["verdict"], "runaway");
    assert_eq!(details["parameters"][0]["packed_estimate"], 5.0);
    assert_eq!(details["parameters"][0]["packed_guard"], 5.0);

    // FIX creates lower == upper at this value, but is intentional and excluded.
    params.sigma_fixed[0] = true;
    let mut converged = true;
    assert!(super::runaway_guard_warning(&mut converged, &params).is_none());
    assert!(converged, "no hit must leave the flag alone");

    // A free value arbitrarily near the safety rail is still an interior result.
    params.sigma_fixed[0] = false;
    params.sigma.values[0] = (5.0_f64 - 1e-12).exp();
    assert!(super::runaway_guard_warning(&mut converged, &params).is_none());
    assert!(converged);

    // The lower rail has opposite meaning and remediation: collapse to zero.
    params.sigma.values[0] = (-8.0_f64).exp();
    let (msg, entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("sigma lower guard hit");
    assert!(msg.contains("collapsed toward zero"));
    assert!(msg.contains("removing or simplifying"));
    // #1118: a collapsed component is a modelling decision, not a runaway — the
    // fit still converged and the entry stays a plain warning. The Σ-ceiling
    // remediation is for the ceiling only and must not ride along.
    assert!(converged, "a collapse hit must not demote convergence");
    assert!(!msg.contains("converged: false"));
    assert!(!msg.contains("rescale DV"));
    assert_eq!(entry.severity, crate::types::WarningSeverity::Warning);
    let hit = &entry.details.as_ref().unwrap()["parameters"][0];
    assert_eq!(hit["side"], "lower");
    assert_eq!(hit["verdict"], "collapse");
}

#[test]
fn runaway_guard_warning_reports_omega_cholesky_guard_and_skips_fixed() {
    use crate::types::{OmegaMatrix, WarningCode};
    let mut params = tiny_model().default_params;
    params.sigma.values.fill(1.0);
    params.sigma_fixed = vec![false; params.sigma.values.len()];
    params.omega_fixed = vec![false; params.omega.dim()];

    let mut chol = params.omega.chol.clone();
    chol[(0, 0)] = 6.0_f64.exp();
    params.omega = OmegaMatrix::from_chol_factor(
        chol,
        params.omega.eta_names.clone(),
        params.omega.diagonal,
        params.omega.free_mask.clone(),
    );

    let mut converged = true;
    let (_msg, entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("omega guard hit");
    assert!(!converged);
    assert_eq!(entry.category, WarningCode::ParameterAtRunawayGuard);
    let hit = &entry.details.as_ref().unwrap()["parameters"][0];
    assert_eq!(hit["parameter"], params.omega.eta_names[0].as_str());
    assert_eq!(hit["packed_estimate"], 6.0);
    assert_eq!(hit["packed_guard"], 6.0);
    assert_eq!(hit["side"], "upper");
    assert!((hit["estimate"].as_f64().unwrap() - 12.0_f64.exp()).abs() < 1e-6);

    // A simultaneous lower collapse and upper runaway gets advice for both.
    params.sigma.values[0] = (-8.0_f64).exp();
    let mut converged = true;
    let (msg, entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("mixed guard hits");
    assert!(msg.contains("Collapse hits"));
    assert!(msg.contains("Runaway hits"));
    // A mixed set still contains a runaway hit, so the fit is demoted.
    assert!(!converged);
    assert_eq!(entry.severity, crate::types::WarningSeverity::Critical);

    params.sigma.values[0] = 1.0;
    params.omega_fixed[0] = true;
    let mut converged = true;
    assert!(super::runaway_guard_warning(&mut converged, &params).is_none());
    assert!(converged);
}

/// #1205 review: an Ω off-diagonal is the raw `L[i,j]` bounded symmetrically at
/// ±10, so the *lower* rail there is a correlation runaway, not a collapse
/// toward zero. Keying the verdict on the side alone reported that runaway as
/// `converged: true` with advice to remove the component.
#[test]
fn omega_off_diagonal_lower_rail_is_a_runaway_not_a_collapse() {
    use crate::types::{OmegaMatrix, WarningSeverity};
    use nalgebra::DMatrix;

    let mut params = tiny_model().default_params;
    params.sigma.values.fill(1.0);
    params.sigma_fixed = vec![false; params.sigma.values.len()];

    // A 2×2 block Ω whose off-diagonal ran to the negative rail.
    let mut chol = DMatrix::<f64>::identity(2, 2);
    chol[(1, 0)] = -10.0;
    params.omega = OmegaMatrix::from_chol_factor(
        chol,
        vec!["ETA_CL".into(), "ETA_V".into()],
        false,
        DMatrix::from_element(2, 2, true),
    );
    params.omega_fixed = vec![false; 2];

    let mut converged = true;
    let (msg, entry) = super::runaway_guard_warning(&mut converged, &params)
        .expect("omega off-diagonal at its lower rail");
    assert_eq!(
        entry.details.as_ref().unwrap()["parameters"][0]["side"],
        "lower",
        "the hit is factually on the lower rail"
    );
    assert_eq!(
        entry.details.as_ref().unwrap()["parameters"][0]["verdict"],
        "runaway",
        "but a symmetric rail means it ran away, not collapsed"
    );
    assert!(
        !converged,
        "a runaway must demote convergence on either rail"
    );
    assert_eq!(entry.severity, WarningSeverity::Critical);
    assert!(msg.contains("implementation rail"));
    assert!(!msg.contains("collapsed toward zero"));

    // The classifier must reach the same severity from the flat message alone,
    // which is all a spliced multi-start warning carries.
    assert_eq!(
        crate::types::classify_warning(&msg).severity,
        WarningSeverity::Critical
    );
}

/// #1328 (PR #1406 review): the fit-end guard walk does **not** read the
/// `ModelParameters` a caller handed `fit()`. On every estimating path it reads
/// what `unpack_params` rebuilt from the optimizer's packed best point, so for a
/// block Ω the property that matters is a composition — `pack_with_bounds` →
/// `clamp_to_bounds` → `unpack_params` → the guard walk must still see the rail.
/// The two tests either side of this one each cover one end of it and neither
/// crosses the middle: they build an `OmegaMatrix` and re-pack it in the same
/// breath.
///
/// Measured, not assumed: replacing the off-diagonal arm of `unpack_params`'
/// reconstruction with `0.0` leaves the whole of
/// `tests/runaway_guard_demotes_convergence.rs` green, because the
/// evaluation-only IMP stage that file uses to reach the fit-end check hands the
/// inner loop its `stage_params` directly and never round-trips them. This test
/// is the one that reddens on that mutation.
///
/// Both rails, because the verdict is not the side here (#1205).
#[test]
fn a_block_omega_off_diagonal_at_the_rail_survives_pack_clamp_unpack() {
    use crate::estimation::parameterization::{
        clamp_to_bounds, coordinate_kinds, pack_with_bounds, unpack_params, PackedCoordKind,
        PackedStart,
    };
    use crate::types::OmegaMatrix;
    use nalgebra::DMatrix;

    for rail in [10.0_f64, -10.0_f64] {
        let mut params = tiny_model().default_params;
        params.sigma.values.fill(1.0);
        params.sigma_fixed = vec![false; params.sigma.values.len()];

        let mut chol = DMatrix::<f64>::identity(2, 2);
        chol[(1, 0)] = rail;
        params.omega = OmegaMatrix::from_chol_factor(
            chol,
            vec!["ETA_CL".into(), "ETA_V".into()],
            false,
            DMatrix::from_element(2, 2, true),
        );
        params.omega_fixed = vec![false; 2];

        // Premise: the coordinate under test is the Ω off-diagonal, it packs to
        // the literal rail, and the rail is a bound of its own box — otherwise
        // the clamp below would be doing the work instead of the round trip.
        let kinds = coordinate_kinds(&params);
        let k = kinds
            .iter()
            .position(|k| *k == PackedCoordKind::OmegaOffDiagonal)
            .expect("the fixture must contain an Ω off-diagonal coordinate");
        let PackedStart {
            mut packed, bounds, ..
        } = pack_with_bounds(&params);
        assert_eq!(packed[k], rail, "rail {rail}: the packed start is the rail");
        assert_eq!(
            if rail > 0.0 {
                bounds.upper[k]
            } else {
                bounds.lower[k]
            },
            rail,
            "rail {rail}: and the rail is that coordinate's own bound"
        );

        // The production sequence: the optimizer works on a clamped packed
        // vector and `optimize_population` publishes `unpack_params` of it.
        clamp_to_bounds(&mut packed, &bounds);
        let round_tripped = unpack_params(&packed, &params);

        // The mechanism, named: the reconstruction puts the rail back on the
        // Cholesky element. `unpack_params` writing `0.0` there fails here.
        assert_eq!(
            round_tripped.omega.chol[(1, 0)],
            rail,
            "rail {rail}: the off-diagonal must survive pack → clamp → unpack"
        );

        // And the consequence the user sees.
        let mut converged = true;
        let (_msg, entry) = super::runaway_guard_warning(&mut converged, &round_tripped)
            .unwrap_or_else(|| panic!("rail {rail}: the round-tripped point is still at the rail"));
        assert!(
            !converged,
            "rail {rail}: a rail that survives the round trip must still demote"
        );
        let hit = &entry.details.as_ref().unwrap()["parameters"][0];
        assert_eq!(hit["parameter"], "ETA_V~ETA_CL", "rail {rail}");
        assert_eq!(hit["verdict"], "runaway", "rail {rail}");
        assert_eq!(
            hit["packed_estimate"].as_f64().unwrap(),
            rail,
            "rail {rail}"
        );
    }
}

#[test]
fn hidden_theta_caps_use_internal_guard_warning_not_user_boundary_warning() {
    use crate::types::WarningCode;
    let mut params = tiny_model().default_params;
    params.sigma.values.fill(1.0);
    params.sigma_fixed = vec![false; params.sigma.values.len()];
    params.theta_fixed = vec![false; params.theta.len()];
    params.theta_lower[0] = 0.0;
    params.theta_upper[0] = f64::INFINITY;

    params.theta[0] = 1e9;
    assert!(super::boundary_estimate_warning(&params).is_none());
    let mut converged = true;
    let (upper_msg, upper_entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("hidden theta upper cap");
    assert!(upper_msg.contains("implementation rail"));
    // The Σ remediation is Σ-specific; a THETA cap must not pick it up.
    assert!(!upper_msg.contains("rescale DV"));
    assert!(!converged);
    assert_eq!(upper_entry.category, WarningCode::ParameterAtRunawayGuard);
    assert_eq!(
        upper_entry.details.as_ref().unwrap()["parameters"][0]["parameter"],
        params.theta_names[0].as_str()
    );

    params.theta[0] = 1e-10;
    assert!(super::boundary_estimate_warning(&params).is_none());
    let mut converged = true;
    let (lower_msg, lower_entry) =
        super::runaway_guard_warning(&mut converged, &params).expect("hidden theta lower cap");
    assert!(lower_msg.contains("collapsed toward zero"));
    assert!(converged);
    assert_eq!(
        lower_entry.details.as_ref().unwrap()["parameters"][0]["side"],
        "lower"
    );

    // A genuine finite user bound keeps the existing warning and remediation.
    params.theta_lower[0] = 0.1;
    params.theta_upper[0] = 100.0;
    params.theta[0] = 0.1;
    assert!(super::boundary_estimate_warning(&params).is_some());
    assert!(super::runaway_guard_warning(&mut converged, &params).is_none());

    // FIX'd hidden-cap coordinates remain intentional and excluded.
    params.theta_lower[0] = 0.0;
    params.theta_upper[0] = f64::INFINITY;
    params.theta[0] = 1e9;
    params.theta_fixed[0] = true;
    assert!(super::runaway_guard_warning(&mut converged, &params).is_none());
    assert!(converged, "a FIX'd cap must not demote convergence");
}

#[test]
fn inflated_rse_warning_emits_typed_entry_with_details() {
    use crate::types::WarningCode;
    let mut params = tiny_model().default_params;
    params.theta_fixed = vec![false; params.theta.len()];
    // No SEs (covariance step didn't run) → no warning.
    assert!(super::inflated_rse_warning(&None, &params).is_none());
    // theta[0] with SE = |est| → RSE 100% (> 50%); theta[1] SE tiny → fine.
    let mut se = vec![0.0; params.theta.len()];
    se[0] = params.theta[0].abs();
    let (msg, entry) =
        super::inflated_rse_warning(&Some(se.clone()), &params).expect("expected an RSE hit");
    assert!(msg.contains("relative standard error"));
    assert_eq!(entry.category, WarningCode::InflatedRse);
    let d = entry.details.as_ref().expect("details");
    assert_eq!(d["threshold_pct"], 50.0);
    assert_eq!(
        d["parameters"][0]["parameter"],
        params.theta_names[0].as_str()
    );
    assert!((d["parameters"][0]["rse_pct"].as_f64().unwrap() - 100.0).abs() < 1e-6);
    // A fixed theta with a huge SE is not reported.
    params.theta_fixed[0] = true;
    assert!(super::inflated_rse_warning(&Some(se), &params).is_none());
    // A near-zero estimate is skipped (avoids divide-by-~0 blowups).
    params.theta_fixed[0] = false;
    params.theta[0] = 0.0;
    assert!(super::inflated_rse_warning(&Some(vec![1.0; params.theta.len()]), &params).is_none());

    // Borderline value renders with a decimal (not rounded to the
    // threshold): RSE = 50.4% must show "50.4%", not "50%".
    let mut p2 = tiny_model().default_params;
    p2.theta_fixed = vec![false; p2.theta.len()];
    p2.theta[0] = 1.0;
    let mut se2 = vec![0.0; p2.theta.len()];
    se2[0] = 0.504; // RSE = 50.4%
    let (msg2, _) = super::inflated_rse_warning(&Some(se2), &p2).expect("borderline hit");
    assert!(msg2.contains("50.4%"), "got: {msg2}");
}

#[test]
fn rebuild_prefers_native_entry_and_is_idempotent() {
    let model = tiny_model();
    let mut fit = synthetic_fit(&model.default_params);
    let msg = "Parameter estimate(s) pinned to an optimizer bound: CL (0.0010 at lower bound)."
        .to_string();
    fit.warnings = vec![msg.clone()];
    // Native entry (with details) seeded as the fit pipeline would.
    fit.warnings_structured = vec![crate::types::WarningEntry {
        severity: crate::types::WarningSeverity::Warning,
        category: crate::types::WarningCode::BoundaryEstimate,
        message: msg.clone(),
        source_method: None,
        details: Some(serde_json::json!({ "parameters": [{ "parameter": "CL" }] })),
    }];
    super::rebuild_warnings_structured(&mut fit);
    // Native entry preferred over re-classifying the string (details kept).
    assert_eq!(fit.warnings_structured.len(), 1);
    assert_eq!(
        fit.warnings_structured[0].category,
        crate::types::WarningCode::BoundaryEstimate
    );
    assert_eq!(
        fit.warnings_structured[0].details.as_ref().unwrap()["parameters"][0]["parameter"],
        "CL"
    );
    // Idempotent: a second rebuild re-captures the native entry, details intact.
    super::rebuild_warnings_structured(&mut fit);
    assert_eq!(
        fit.warnings_structured[0].details.as_ref().unwrap()["parameters"][0]["parameter"],
        "CL"
    );
}

#[test]
fn rebuild_attaches_details_from_typed_fields() {
    let model = tiny_model();
    let mut fit = synthetic_fit(&model.default_params);
    fit.warnings = vec![
        "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20).".to_string(),
        "Outer optimization did not converge".to_string(),
        "High ETA shrinkage (\u{2265} 30%): eta_V (42%). EBE-based diagnostics \
             for these random effects are unreliable."
            .to_string(),
    ];
    fit.dw_statistic = 1.20;
    fit.iwres_lag1_r = 0.4;
    fit.shrinkage_eta = vec![0.05, 0.42];
    fit.eta_names = vec!["eta_CL".to_string(), "eta_V".to_string()];
    super::rebuild_warnings_structured(&mut fit);

    let dw = &fit.warnings_structured[0];
    assert_eq!(dw.category, crate::types::WarningCode::DwAutocorrelation);
    let details = dw.details.as_ref().expect("DW warning carries details");
    assert_eq!(details["durbin_watson"], 1.20);
    assert_eq!(details["iwres_lag1_autocorr"], 0.4);
    // A non-numeric code keeps details omitted.
    assert!(fit.warnings_structured[1].details.is_none());
    // ETA shrinkage classifies + attaches its high-eta details.
    let eta = &fit.warnings_structured[2];
    assert_eq!(eta.category, crate::types::WarningCode::EtaShrinkage);
    let ed = eta.details.as_ref().expect("ETA shrinkage details");
    assert_eq!(ed["high_shrinkage_etas"][0]["eta"], "eta_V");
}

#[test]
fn sir_path_reuses_pool() {
    let model = tiny_model();
    let pop = tiny_population();
    let mut fit = synthetic_fit(&model.default_params);
    // Build a 4-element resample pool: small perturbations of x_hat in
    // packed log-space. Tests will sample with replacement from this pool.
    let x_hat = crate::estimation::parameterization::pack_params(&model.default_params);
    let pool: Vec<Vec<f64>> = (0..4)
        .map(|k| {
            let mut xk = x_hat.clone();
            xk[0] += 0.005 * (k as f64);
            xk
        })
        .collect();
    fit.sir_resamples_packed = Some(pool);

    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 5,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Sir,
        seed: Some(11),
    };
    let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
    // 5 draws * 1 sim * 2 subjects * 3 obs = 30 rows
    assert_eq!(rows.len(), 30);
}

#[test]
fn asymptotic_errors_without_covariance_step() {
    let model = tiny_model();
    let pop = tiny_population();
    let mut fit = synthetic_fit(&model.default_params);
    fit.covariance_matrix = None;
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 2,
        n_sim_per_draw: 1,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(0),
    };
    let err = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap_err();
    assert!(err.to_string().contains("covariance"));
}

// ---------------------------------------------------------------------------
// #529 — `Default` at the R boundary
// ---------------------------------------------------------------------------

/// Pin every field of the derived `Default`. The R wrapper builds this struct
/// with `..Default::default()`, so these values are the ones a field it does
/// *not* name silently takes.
#[test]
fn simulate_uncertainty_options_default_values() {
    let d = SimulateUncertaintyOptions::default();
    assert_eq!(d.n_uncertainty_draws, 0);
    assert_eq!(d.n_sim_per_draw, 0);
    assert_eq!(d.method, UncertaintyMethod::Asymptotic);
    assert_eq!(d.seed, None);
}

/// The point of #529: a spread construction that names only the fields the
/// caller cares about must behave identically to the exhaustive literal it
/// replaces. Run both through `simulate_with_uncertainty` with the same seed
/// and require bit-identical predictions — an oracle that fails if a future
/// field's default silently changes the simulation rather than merely
/// compiling.
#[test]
fn spread_default_matches_exhaustive_literal() {
    let model = tiny_model();
    let pop = tiny_population();
    let fit = synthetic_fit(&model.default_params);

    let exhaustive = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 2,
        method: UncertaintyMethod::Asymptotic,
        seed: Some(42),
    };
    let spread = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 2,
        seed: Some(42),
        ..Default::default()
    };

    let a = simulate_with_uncertainty(&model, &pop, &fit, &exhaustive).unwrap();
    let b = simulate_with_uncertainty(&model, &pop, &fit, &spread).unwrap();

    // Non-degenerate: the comparison is worthless on an empty or all-zero run.
    assert_eq!(a.len(), 3 * 2 * 2 * 3);
    assert!(a
        .iter()
        .all(|r| r.ipred.is_finite() && r.outcome.continuous_value().is_finite()));
    assert!(a.iter().any(|r| r.ipred != 0.0));

    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.id, y.id);
        assert_eq!(x.time, y.time);
        assert_eq!(x.draw, y.draw);
        assert_eq!(x.sim, y.sim);
        assert_eq!(x.ipred.to_bits(), y.ipred.to_bits());
        assert_eq!(
            x.outcome.continuous_value().to_bits(),
            y.outcome.continuous_value().to_bits()
        );
    }
}

/// #529, trait-level half: the options types `ferx-core` exposes at the R
/// boundary really do resolve `Default`, so the wrapper can spread them.
///
/// This is a **named spot check, not the inventory guard** — the list is
/// hand-written, so adding a new options struct does not change it and this
/// test would stay green. Discovery is
/// `every_public_options_type_implements_default` in
/// `tests/public_api_boundary.rs` (A4), which scans `src/` and `crates/` for
/// every `*Options` declaration and fails on one lacking `Default`. The two
/// are complementary: the scan is textual and sees types nobody registered,
/// this one is a real trait bound and would catch a `Default` that exists in
/// source but does not apply to the type (a `cfg`-ed out impl, say).
#[test]
fn every_public_options_struct_implements_default() {
    fn assert_default<T: Default>() -> T {
        T::default()
    }
    let _ = assert_default::<crate::types::FitOptions>();
    let _ = assert_default::<crate::io::fitrx::SaveFitOptions>();
    let _ = assert_default::<crate::ode::solver::OdeSolverOptions>();
    let _ = assert_default::<crate::api::simulate::SimulateOptions>();
    let _ = assert_default::<SimulateUncertaintyOptions>();
    let _ = assert_default::<crate::AdaptiveSimulateOptions>();
}
