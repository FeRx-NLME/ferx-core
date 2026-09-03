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
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
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
    let msg = check_survival_tv_covariates(&model, &pop)
        .expect("a time-varying X in the logit must be rejected up front");
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
        check_survival_tv_covariates(&model, &pop_ok).is_none(),
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
    let msg = check_survival_tv_covariates(&model, &pop)
        .expect("a time-varying CRCL on the hazard must be rejected up front");
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
        check_survival_tv_covariates(&model, &pop).is_none(),
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
        check_survival_tv_covariates(&model, &pop).is_none(),
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
    let msg = check_survival_tv_covariates(&model, &pop)
        .expect("ODE-accumulated hazard + any TV covariate must be rejected");
    assert!(msg.contains("WT"), "must name the covariate: {msg}");
    assert!(msg.contains("#741"), "must cite the issue: {msg}");
}

fn tiny_population() -> Population {
    let obs_times = vec![1.0, 2.0, 3.0];
    let subjects: Vec<Subject> = (0..2)
        .map(|i| Subject {
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
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            obs_l2: Vec::new(),
            dose_occasions: vec![1],
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

/// Build a synthetic `FitResult` carrying the fitted theta/Omega/Sigma
/// from `template` plus a small identity covariance in packed log-space.
/// Avoids invoking `fit()` (slow) while still exercising the full
/// `simulate_with_uncertainty` wiring.
fn synthetic_fit(template: &ModelParameters) -> FitResult {
    let n_packed = crate::estimation::parameterization::packed_len(template);
    let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
    FitResult {
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
        gradient_method_inner: String::new(),
        gradient_method_outer: String::new(),
        uses_ode_solver: false,
        uses_sde: false,
        n_threads_used: 1,
        nlopt_missing_algorithms: vec![],
        covariance_n_evals_estimated: None,
        trace_path: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
        covariance_status: CovarianceStatus::Computed,
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
        theta_transform: vec![],
        sigma_types: vec![],
        cov_eigenvalues: None,
        cov_condition_number: None,
        eta_log_transformed: vec![],
        omega_param_corr: None,
        omega_iov_param_corr: None,
        model_path: None,
        data_path: None,
        model_hash: None,
        data_hash: None,
        model_text: None,
        theta_init: template.theta.clone(),
        omega_init: template.omega.matrix.clone(),
        sigma_init: template.sigma.values.clone(),
        obs_time_range: None,
        final_gradient: None,
        optimizer: "bobyqa".to_string(),
        n_starts: 1,
        multi_start_seed: None,
        saem_seed: None,
        sir_seed: None,
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
/// flip-flop regime must NOT panic (pre-fix: the per-draw
/// `assert_absorption_flip_flop_no_twin` aborted the whole run). The crossing
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

    // The per-draw skip warning is pushed into `sim_warnings`, which this
    // aggregated-uncertainty entry point discards (it returns only the row vec),
    // so the message is not observable here. Pin the skip *predicate* directly:
    // a flip-flop-theta draw is flagged, the in-domain point estimate is not.
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
    let rows = simulate_with_seed(&model, &pop, &model.default_params, 2, 42);
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

    let a =
        crate::stats::npde::compute_npde_npd(&model, &pop, &model.default_params, nsim, Some(7));
    let b =
        crate::stats::npde::compute_npde_npd(&model, &pop, &model.default_params, nsim, Some(7));

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
    let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
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
    assert!(err.contains("covariance"));
}
