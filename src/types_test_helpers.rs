use super::*;
use std::collections::HashMap;

/// Build an analytical-PK model (`tv_fn = Some`, `ode_spec = None`).
pub(crate) fn analytical_model(gradient_method: GradientMethod) -> CompiledModel {
    make_compiled_model(false, gradient_method)
}

/// Build an ODE-backed model (`tv_fn = None`, `ode_spec = Some`).
pub(crate) fn ode_model(gradient_method: GradientMethod) -> CompiledModel {
    make_compiled_model(true, gradient_method)
}

fn make_compiled_model(with_ode: bool, gradient_method: GradientMethod) -> CompiledModel {
    CompiledModel {
        covariate_model: None,
        name: "test".into(),
        pk_model: PkModel::OneCptOral,
        error_model: ErrorModel::Additive,
        error_spec: ErrorSpec::Single(ErrorModel::Additive),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(|_, _, _, _| PkParams::default()),
        n_theta: 1,
        n_eta: 1,
        n_epsilon: 1,
        n_kappa: 0,
        theta_names: vec!["CL".into()],
        eta_names: vec!["ETA_CL".into()],
        kappa_names: Vec::new(),
        indiv_param_names: vec!["CL".into()],
        indiv_param_partials: IndivParamPartials::empty(),
        default_params: ModelParameters {
            residual_correlations: Vec::new(),
            residual_correlation_fixed: Vec::new(),
            theta: vec![1.0],
            theta_names: vec!["CL".into()],
            theta_lower: vec![0.0],
            theta_upper: vec![f64::INFINITY],
            theta_fixed: vec![false],
            omega: OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["EPS".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
            mixture: None,
        },
        omega_init_as_sd: vec![false],
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        // Analytical models populate tv_fn; ODE models leave it None.
        tv_fn: if with_ode {
            None
        } else {
            Some(Box::new(|_t, _c| vec![1.0]))
        },
        pk_indices: vec![],
        eta_map: vec![],
        pk_idx_f64: vec![],
        sel_flat: vec![],
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        ode_spec: if with_ode {
            Some(crate::ode::OdeSpec {
                chz_state_slots: Vec::new(),
                rhs: Box::new(|_y, _p, _t, _dy| {}),
                n_states: 2,
                state_names: vec!["depot".into(), "central".into()],
                readout: crate::ode::OdeReadout::ObsCmt(0),
                diffusion_var: Vec::new(),
                init_fn: None,
                solver_opts: crate::ode::OdeSolverOptions::default(),
                input_rate: Vec::new(),
                rhs_program: None,
                readout_program: None,
                indiv_param_program: None,
                dose_attr_map: Default::default(),
            })
        } else {
            None
        },
        dose_attr_map: Default::default(),
        bloq_method: BloqMethod::Drop,
        referenced_covariates: vec![],
        gradient_method,
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
        endpoints: HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    }
}

/// A 1-cpt IV model with **proportional** error whose CL scales with the
/// `WT` covariate (`CL = TVCL·WT/70`), paired with a subject whose `WT`
/// rises across observation rows (70 → 140 → 210). Because `WT` is
/// time-varying, the TV-aware predictor and the baseline-only
/// `pk_param_fn(subject.covariates)` disagree on every obs row after the
/// first — the fixture shared by the simulation / NPDE / NCA-sweep
/// TV-covariate regression tests (#506). `n_eta = 0` so the only spread in
/// simulated replicates is residual error, which keeps NPD deterministic.
pub(crate) fn tv_cov_iv_model_and_subject() -> (CompiledModel, Subject) {
    let params = ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.1, 5.0],
        theta_upper: vec![50.0, 500.0],
        theta_fixed: vec![false; 2],
        omega: OmegaMatrix::from_diagonal(&[], vec![]),
        omega_fixed: Vec::new(),
        sigma: SigmaVector {
            values: vec![0.05],
            names: vec!["PROP_ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
        mixture: None,
    };
    let model = CompiledModel {
        covariate_model: None,
        name: "tv_cov_iv".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], _eta: &[f64], cov: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * (wt / 70.0);
                p.values[1] = theta[1];
                p
            },
        ),
        n_theta: 2,
        n_eta: 0,
        n_epsilon: 1,
        n_kappa: 0,
        kappa_names: Vec::new(),
        theta_names: vec!["TVCL".into(), "TVV".into()],
        eta_names: Vec::new(),
        indiv_param_names: vec!["CL".into(), "V".into()],
        indiv_param_partials: IndivParamPartials::empty(),
        default_params: params.clone(),
        omega_init_as_sd: Vec::new(),
        sigma_init_as_sd: vec![false],
        kappa_init_as_sd: Vec::new(),
        kappa_weights: Vec::new(),
        mu_refs: HashMap::new(),
        kappa_mu_refs: HashMap::new(),
        tv_fn: None,
        pk_indices: vec![0, 1],
        eta_map: Vec::new(),
        pk_idx_f64: vec![0.0, 1.0],
        sel_flat: Vec::new(),
        ode_spec: None,
        dose_attr_map: Default::default(),
        diffusion_theta_start: None,
        diffusion_state_indices: Vec::new(),
        bloq_method: BloqMethod::Drop,
        referenced_covariates: vec!["WT".into()],
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
        endpoints: HashMap::new(),
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
    };

    let mut baseline_cov = HashMap::new();
    baseline_cov.insert("WT".to_string(), 70.0);
    let subject = Subject {
        id: "TVCOV".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0, 0.0, 0.0],
        obs_cmts: vec![1, 1, 1],
        covariates: baseline_cov,
        dose_covariates: vec![HashMap::from([("WT".to_string(), 70.0)])],
        obs_covariates: vec![
            HashMap::from([("WT".to_string(), 70.0)]),
            HashMap::from([("WT".to_string(), 140.0)]),
            HashMap::from([("WT".to_string(), 210.0)]),
        ],
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
    };
    (model, subject)
}
