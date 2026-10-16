use super::*;
use crate::stats::likelihood::iov_occasion_groups;
use crate::types::{
    BloqMethod, DoseEvent, ErrorModel, GradientMethod, OmegaMatrix, PkModel, PkParams, SigmaVector,
};
use std::collections::HashMap;

#[test]
fn iov_joint_warm_start_preserves_kappas() {
    let mu = [0.5, -0.25];
    let bsv_only = iov_initial_vector(2, 6, &mu, Some(&[0.25, 0.5]));
    assert_eq!(bsv_only, vec![0.75, 0.25, 0.0, 0.0, 0.0, 0.0]);

    let joint = [0.25, 0.5, -0.7, 0.8, 0.9, -1.0];
    let warm = iov_initial_vector(2, 6, &mu, Some(&joint));
    assert_eq!(
        warm,
        vec![0.75, 0.25, -0.7, 0.8, 0.9, -1.0],
        "the BSV block is shifted to psi-space while every fitted kappa is retained"
    );
}

#[test]
fn gradient_route_summary_reports_route_taken_not_requested() {
    // make_iov_model has `tv_fn: None` and the default `gradient_method:
    // Auto`. With no `tv_fn`, AD is unavailable, so the route resolves to
    // FD in every build — even though the *requested* method is `auto`.
    // The banner must report the route actually taken (FD) and surface the
    // request, so a silent AD→FD fallback is visible.
    let model = make_iov_model();
    let population = Population {
        subjects: vec![make_iov_subject()],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    // `requested` is the user's FitOptions value, passed independently of
    // model.gradient_method (which compatibility rules may have mutated).
    let summary = gradient_route_summary(&model, &population, GradientMethod::Auto);
    assert!(
        summary.starts_with("FD"),
        "tv_fn=None must resolve to FD, got: {summary}"
    );
    // The bracket echoes the requested method, e.g. "[requested: auto]".
    assert!(
        summary.contains("[requested: auto"),
        "summary must surface the requested method, got: {summary}"
    );
    // The bracket reflects the passed `requested`, not model.gradient_method
    // — guards against regressing to the SDE-mislabel Copilot flagged on #117.
    let fd_summary = gradient_route_summary(&model, &population, GradientMethod::Fd);
    assert!(
        fd_summary.contains("[requested: FD"),
        "bracket must echo the requested arg, got: {fd_summary}"
    );
}

#[test]
fn gradient_route_summary_reports_ode_iov_analytic_route() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    let population = Population {
        subjects: vec![Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 25.0, 30.0],
            obs_raw_times: Vec::new(),
            observations: vec![8.0, 6.0, 7.0, 5.0],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 4],
            occasions: vec![1, 1, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        }],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let summary = gradient_route_summary(&model, &population, GradientMethod::Auto);
    assert!(
        summary.starts_with("analytic (Dual2)"),
        "ODE IOV provider should be reported as analytic, got: {summary}"
    );
}

/// Regression: `gradient = fd` must force the FD inner route on an
/// analytic-supported model (previously the executor ignored
/// `model.gradient_method`, so the option silently ran the Dual2 path while
/// `build_info` reported FD). Uses the bundled warfarin model, which is in the
/// analytic provider's scope (1-cpt oral, no LTBS / TV-cov / SDE).
#[test]
fn gradient_fd_forces_fd_inner_route() {
    use std::path::Path;
    let mut model =
        crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
            .expect("warfarin parses");
    let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let subj = &pop.subjects[0];

    model.gradient_method = GradientMethod::Auto;
    assert_eq!(
        resolve_gradient_method(&model, subj),
        InnerGradientMethod::Analytic,
        "auto must resolve to the analytic route for the warfarin model"
    );
    model.gradient_method = GradientMethod::Fd;
    assert_eq!(
        resolve_gradient_method(&model, subj),
        InnerGradientMethod::Fd,
        "gradient = fd must force the FD inner route"
    );
}

/// `fd_fallback_warning` fires only for a *mixed* population — some subjects
/// analytic, some on FD (here a modeled-duration `RATE=-2` subject, which the
/// provider declines per-point). Uniform populations return `None`.
#[test]
fn fd_fallback_warning_fires_only_for_mixed_population() {
    use std::path::Path;
    let model = crate::parser::model_parser::parse_model_file(Path::new("examples/warfarin.ferx"))
        .expect("warfarin parses");
    let pop = crate::read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let theta = &model.default_params.theta;
    let analytic = pop.subjects[0].clone();
    let mut fd_subj = pop.subjects[0].clone();
    let mut d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    d.rate_mode = crate::types::RateMode::ModeledDuration;
    fd_subj.doses.push(d);
    let mk_pop = |subjects| Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let mixed = mk_pop(vec![analytic.clone(), fd_subj]);
    let w = fd_fallback_warning(&model, &mixed, theta).expect("mixed population warns");
    assert!(w.contains("1 of 2"), "got: {w}");

    // Uniform analytic → no warning.
    assert!(fd_fallback_warning(&model, &mk_pop(vec![analytic]), theta).is_none());
}

/// Tier-1 follow-up to #665: a TV-cov + LTBS subject now takes the **analytic** inner
/// gradient (the event-driven inner walk applies the `ln f` jet LAST), so the reported
/// route is Analytic, `subject_eta_grad` returns `Some`, and no FD-fallback warning fires.
#[test]
fn tvcov_ltbs_reports_analytic_inner() {
    use crate::parser::model_parser::parse_model_string;
    let src = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma ADD_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[covariates]
  WT continuous
[error_model]
  log(DV) ~ additive(ADD_ERR)
"#;
    let model = parse_model_string(src).expect("parse TV-cov LTBS");
    assert!(model.log_transform);
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let tvcov_subj = || Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 4.0, 8.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0, 3.0],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        // Time-varying WT → routes to the event-driven walk, which now serves LTBS.
        obs_covariates: vec![wt(70.0), wt(72.0), wt(80.0), wt(85.0), wt(90.0)],
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![1; 5],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let subject = tvcov_subj();
    assert!(subject.has_tv_covariates(), "fixture must be TV-cov");
    // TV-cov + LTBS now takes the analytic inner gradient.
    assert_eq!(
        resolve_gradient_method(&model, &subject),
        InnerGradientMethod::Analytic,
        "TV-cov + LTBS inner gradient is analytic now"
    );
    assert!(
        crate::sens::provider::subject_eta_grad(
            &model,
            &subject,
            &model.default_params.theta,
            &[0.0; 3]
        )
        .is_some(),
        "TV-cov + LTBS inner provider must serve the subject"
    );
    // Whole population analytic → no FD-fallback warning.
    let population = Population {
        subjects: vec![tvcov_subj(), tvcov_subj()],
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    assert!(
        fd_fallback_warning(&model, &population, &model.default_params.theta).is_none(),
        "analytic TV-cov + LTBS population must not warn"
    );
}

#[test]
fn iov_fd_fallback_warning_reports_subject_reason() {
    // Covariate-free ODE IOV model the provider serves analytically, so the FD
    // subject below is the *only* one out of scope — a genuinely mixed
    // population (the all-FD case is suppressed, mirroring the non-IOV
    // contract, #590 review).
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    // Analytic subject: doses and observations in the same occasions.
    let analytic_subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    // FD subject: more occasion groups than the widened ODE IOV dispatch serves.
    let n_wide = crate::sens::ode_provider::MAX_ODE_IOV_AXES;
    let fd_subject = Subject {
        obs_rgroup: Vec::new(),
        id: "2".into(),
        doses: (0..n_wide)
            .map(|i| DoseEvent::new(i as f64 * 24.0, 100.0, 1, 0.0, false, 0.0))
            .collect(),
        obs_times: (0..n_wide).map(|i| i as f64 * 24.0 + 1.0).collect(),
        obs_raw_times: Vec::new(),
        observations: vec![8.0; n_wide],
        obs_cmts: vec![1; n_wide],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; n_wide],
        occasions: (1..=n_wide as u32).collect(),
        obs_l2: Vec::new(),
        dose_occasions: (1..=n_wide as u32).collect(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![analytic_subject, fd_subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let warning = fd_fallback_warning(&model, &population, &model.default_params.theta)
        .expect("mixed IOV population should warn with a reason");
    assert!(warning.contains("1 of 2"), "got: {warning}");
    assert!(
        warning.contains("ODE IOV stacked axis cap"),
        "got: {warning}"
    );
}

/// Uniform all-FD IOV populations are silent, matching the non-IOV contract:
/// the `finite-difference` banner already makes a model-level fallback obvious.
#[test]
fn iov_fd_fallback_warning_silent_for_uniform_all_fd() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  gradient = fd\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + gradient = fd");
    let mk_subject = |id: &str| Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let population = Population {
        subjects: vec![mk_subject("1"), mk_subject("2")],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    // gradient = fd routes every subject to FD (uniform) → no warning.
    assert!(
        fd_fallback_warning(&model, &population, &model.default_params.theta).is_none(),
        "uniform all-FD population must not warn"
    );
}

/// Regression (#486): steady-state combined with an estimated lagtime is now analytic
/// under IOV (the `K_SS_SEED` pre-arrival seed, shared with the non-IOV walk). Before
/// #486 this subject declined via the `SS + lagtime` gate (#590 review); pins that the
/// inner IOV route now admits it instead.
#[test]
fn iov_inner_subject_route_admits_steady_state_lagtime() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVLAG(0.5,0.01,5.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_LAG ~ 0.09\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  LAGTIME = TVLAG * exp(ETA_LAG)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  obs_scale = V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + lagtime");
    assert!(
        model.has_lagtime(),
        "model should carry a LAGTIME individual parameter"
    );
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        // Steady-state bolus (ss, ii > 0) under an estimated lagtime.
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 24.0)],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_some(),
        "SS + lagtime subject must be analytic now (#486)"
    );
}

/// Regression (#486): modeled-`RATE`/duration doses combined with steady-state are now
/// analytic under IOV too (`equilibrate_ss_state_g` threads the same per-occasion
/// `inf_eff` jet into its per-cycle split). Before #486 this subject declined via the
/// modeled+SS screen; pins that the inner IOV route now admits it instead.
#[test]
fn iov_inner_subject_route_admits_modeled_dose_steady_state() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVD1(5.0,0.1,24.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_D1 ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  D1 = TVD1 * exp(ETA_D1)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + modeled D1");
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            true,
            24.0,
            crate::types::RateMode::ModeledDuration,
        )],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_some(),
        "modeled + SS subject must be analytic now (#486)"
    );
}

/// Regression (#486): a modeled dose whose `D{cmt}`/`R{cmt}` slot is undeclared (normally
/// rejected by `check_model_data`, but defended here) routes to FD and is attributed to the
/// missing slot — not silently mis-resolved. The base model declares no `D1`, so the
/// `ModeledDuration` dose finds no duration slot.
#[test]
fn iov_fd_reason_attributes_modeled_dose_missing_slot() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV without D1");
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_none(),
        "modeled dose with missing slot must route to FD"
    );
    assert_eq!(
        iov_fd_reason(&model, &subject),
        "modeled RATE/DURATION dose with missing D/R slot"
    );
}

/// #1613 G5: the IOV FD-fallback reason reads the **call's** `gradient = fd`
/// (`GradientMethod::forced_fd`), so a direct `fit()` with `FitOptions { gradient_method: Fd }`
/// on an unstamped model attributes its FD subjects to the setting the user chose, not to
/// whatever scope clause happens to come next. The straddle is on one subject of the fixture
/// above: unarmed it names its own scope reason, armed it names `gradient = fd`.
///
/// Mutation: revert `iov_fd_reason`'s first clause to `model.gradient_method` → the armed
/// assert reads the modeled-dose reason.
#[test]
fn iov_fd_reason_names_the_calls_gradient_fd() {
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV without D1");
    assert_eq!(
        model.gradient_method,
        GradientMethod::Auto,
        "premise: unstamped"
    );
    let subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::modeled(
            0.0,
            100.0,
            1,
            false,
            0.0,
            crate::types::RateMode::ModeledDuration,
        )],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        dose_occasions: vec![1],
        ..Default::default()
    };
    assert_eq!(
        iov_fd_reason(&model, &subject),
        "modeled RATE/DURATION dose with missing D/R slot",
        "control: unarmed, the subject's own scope reason"
    );
    let _fd = crate::types::arm_forced_fd(true);
    assert_eq!(
        iov_fd_reason(&model, &subject),
        "gradient = fd",
        "armed: the call's gradient = fd must be the reason"
    );
}

#[test]
fn iov_fd_reason_attributes_ss_input_rate() {
    // #486: a steady-state dose + a built-in absorption forcing (`zero_order`) declines
    // to FD under IOV; `iov_fd_reason` must name that combination (not the generic
    // "outside IOV analytic scope"), mirroring `ode_iov_subject_supported`'s bail.
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVDUR(5.0,0.1,24.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_DUR ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  DUR = TVDUR * exp(ETA_DUR)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse zero_order ODE IOV");
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_none(),
        "SS + zero_order must route to FD under IOV"
    );
    assert_eq!(
        iov_fd_reason(&model, &subject),
        "steady-state dose + built-in absorption forcing"
    );
}

#[test]
fn iov_fd_reason_attributes_infusion_into_absorption() {
    // #719 gap 2 / #835 review: a finite infusion into a built-in absorption compartment
    // declines to FD under IOV (the dual `+rate` would double-count the convolved `R_in_inf`
    // mass). `iov_fd_reason` must name the infusion — mirroring `ode_iov_subject_supported`'s
    // bail, which fires *before* its SS-absorption gate — not fall through to the generic
    // "subject outside IOV analytic scope".
    let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVDUR(5.0,0.1,24.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_DUR ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  DUR = TVDUR * exp(ETA_DUR)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = zero_order(dur=DUR) - (CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse zero_order ODE IOV");
    // Finite infusion (rate > 0, not steady state) into the absorption compartment.
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 10.0, false, 0.0)],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_none(),
        "infusion into absorption must route to FD under IOV"
    );
    assert_eq!(
        iov_fd_reason(&model, &subject),
        "infusion into built-in absorption compartment"
    );
}

/// #835: an SS dose into a built-in absorption compartment is now analytic for the smooth
/// density kernels, *including through a closed-form primary's ODE twin*. A closed-form
/// `one_cpt_transit` + IOV subject with a steady-state dose reroutes to its ODE twin
/// (`effective_for`, #719/#814); the twin carries a `transit()` forcing, which #835 admits, so
/// the subject now routes to the analytic ODE-IOV inner gradient rather than FD. (Pre-#835 it
/// declined with "steady-state dose + built-in absorption forcing"; the still-FD combinations —
/// SS into a `zero_order` window, SS + absorption lagtime — remain covered by
/// `iov_fd_reason_attributes_ss_input_rate`.)
#[test]
fn transit_twin_ss_forcing_is_analytic_under_iov() {
    let model = crate::parser::model_parser::parse_model_string(TRANSIT_IOV_MODEL)
        .expect("parse transit IOV");
    assert!(
        model.ode_spec.is_none(),
        "the closed-form transit primary is analytic"
    );
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        iov_inner_subject_route(&model, &subject, &model.default_params.theta).is_some(),
        "#835: SS + transit-twin absorption forcing is now analytic under IOV (via the twin)"
    );
}

/// A closed-form `one_cpt_transit` + IOV model (#814): analytic primary
/// (`ode_spec == None`) that carries an ODE twin and reroutes every subject to it
/// (`n_kappa > 0`, #719). Used to assert the inner loop treats it as ODE+IOV.
const TRANSIT_IOV_MODEL: &str = "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVMTT(1.0,0.01,50.0)\n  theta TVN(3.0,0.5,20.0)\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  MTT = TVMTT\n  NTR = TVN\n[structural_model]\n  pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n";

fn make_iov_model() -> CompiledModel {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
    let default_params = crate::types::ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.01, 1.0],
        theta_upper: vec![100.0, 500.0],
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
        residual_serial: None,
        priors: Vec::new(),
        prior_from_fit: None,
        covariate_model: None,
        name: "iov_test".into(),
        pk_model: PkModel::OneCptIv,
        error_model: ErrorModel::Proportional,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
        residual_correlations: Vec::new(),
        pk_param_fn: Box::new(
            |theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
                let mut p = PkParams::default();
                // eta[0] = bsv, eta[1] = kappa (combined)
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
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
        gradient_method: GradientMethod::default(),
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

fn make_iov_subject() -> Subject {
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        obs_raw_times: Vec::new(),
        observations: vec![40.0, 32.0, 25.0, 38.0, 30.0, 22.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

#[test]
fn test_find_ebe_iov_two_occasions_returns_two_kappas() {
    let model = make_iov_model();
    let subject = make_iov_subject();
    let params = model.default_params.clone();
    let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None, 0);
    assert_eq!(result.kappas.len(), 2, "Expected 2 kappas for 2 occasions");
    assert_eq!(result.kappas[0].len(), 1);
    assert_eq!(result.kappas[1].len(), 1);
    assert!(result.converged || result.nll.is_finite());
}

#[test]
fn test_find_ebe_iov_h_matrix_dimensions() {
    let model = make_iov_model();
    let subject = make_iov_subject();
    let params = model.default_params.clone();
    let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None, 0);
    // H-matrix: n_obs × n_eta (BSV only, kappas fixed)
    assert_eq!(result.h_matrix.nrows(), subject.obs_times.len());
    assert_eq!(result.h_matrix.ncols(), model.n_eta);
}

/// The analytic ODE IOV inner gradient (`analytic_eta_nll_gradient_iov`) must match
/// central finite differences of the inner objective `individual_nll_iov` over the
/// stacked `[η_bsv, κ₁..κ_K]` vector — the gradient that now drives `find_ebe_iov`
/// for ODE IOV models (#439 ODE IOV inner).
#[test]
fn analytic_iov_inner_grad_matches_fd_of_nll() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV");
    assert!(crate::sens::ode_provider::ode_iov_supported(&model));
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 4.0, 7.0, 5.0, 3.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    let stacked = vec![0.10, -0.05, 0.08, -0.12];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV inner gradient");

    // Central FD of the inner objective (same NLL `find_ebe_iov` minimises).
    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    for p in 0..n_stacked {
        let h = 1e-6 * (1.0 + stacked[p].abs());
        let mut sp = stacked.clone();
        sp[p] += h;
        let mut sm = stacked.clone();
        sm[p] -= h;
        let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
    }
}

/// Closed-form twin of [`analytic_iov_inner_grad_matches_fd_of_nll`] with an η-dependent
/// `ExpressionScale` `obs_scale = V` divisor (#486): the analytic IOV inner gradient
/// (`analytic_eta_nll_gradient_iov`, now fed the scaled `subject_eta_grad_iov`) must match
/// central FD of the same objective `individual_nll_iov` (which applies `obs_scale`) over
/// the stacked `[η_bsv, κ]` vector — the gradient that drives `find_ebe_iov` for a scaled
/// closed-form IOV model.
#[test]
fn analytic_iov_inner_grad_matches_fd_of_nll_closed_form_expr_scale() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + obs_scale");
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.8, 0.6, 0.4, 0.7, 0.5, 0.3],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    let stacked = vec![0.10, -0.05, 0.08, -0.12];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV inner gradient");

    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    for p in 0..n_stacked {
        let h = 1e-6 * (1.0 + stacked[p].abs());
        let mut sp = stacked.clone();
        sp[p] += h;
        let mut sm = stacked.clone();
        sm[p] -= h;
        let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
    }
}

/// LTBS × IOV closed-form **inner** EBE gradient vs central FD of its own objective (#486).
///
/// This is the cell that closed the last inner/outer scope split: the outer IOV gradient has
/// served LTBS since #677, but the `Dual1` inner walk had no `ln` jet, so `find_ebe_iov`
/// reconverged on finite differences. `run_obs_iov_eta` now applies the shared
/// `apply_ltbs_transform_inner` last — after the per-occasion scale quotient — so the EBE
/// gradient differentiates the same `ln(f/s)` the objective scores.
///
/// Run over BOTH fixtures because the ordering is the part that can silently go wrong:
///   * plain LTBS × IOV — the bare `∂ln f/∂x = (∂f/∂x)/f` jet over the stacked `[η_bsv, κ]`;
///   * LTBS × IOV × `ExpressionScale` — pins **scale-then-log** (`ln(f/s)`, production's
///     order). Applying the jet before the quotient instead still yields a plausible
///     gradient — it is just the gradient of `ln(f)/s`, a different function — and FD of the
///     real objective is what catches it.
///
/// The κ columns need no special treatment (the transform is one per-observation scalar
/// quotient, identical on every stacked axis), and that is exactly what this asserts: FD
/// parity is checked on every stacked coordinate, κ blocks included.
#[test]
fn analytic_iov_inner_grad_matches_fd_of_nll_closed_form_ltbs() {
    use crate::parser::model_parser::parse_model_string;
    let fixture = |scaling_block: &str| {
        format!(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  \
             omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  \
             sigma ADD_ERR ~ 0.2 (sd)\n[individual_parameters]\n  \
             CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n\
             [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n{scaling_block}\
             [error_model]\n  log(DV) ~ additive(ADD_ERR)\n[fit_options]\n  method = focei\n  \
             iov_column = OCC\n"
        )
    };
    for (label, scaling_block) in [
        ("plain LTBS", ""),
        ("LTBS + obs_scale", "[scaling]\n  obs_scale = 1000 / V\n"),
    ] {
        let model = parse_model_string(&fixture(scaling_block)).expect("parse LTBS IOV fixture");
        assert!(model.log_transform, "{label}: fixture must be LTBS");
        // Routing first: a scope gap must fail loudly here rather than quietly FD-ing and
        // leaving the parity check below to pass against a gradient nobody ran.
        assert!(
            crate::sens::provider::iov_analytical_supported(&model),
            "{label}: must be on the analytic closed-form IOV path"
        );
        assert!(
            !analytic_inner_common_bail(&model),
            "{label}: the analytic INNER must not be bailed (#486)"
        );
        // The tolerance predicate must move WITH the bail, not independently (PR #950
        // review #1). `uses_closed_form_ltbs_inner` is the sole input to
        // `effective_inner_tol` / `effective_cov_inner_tol`; while it carried an
        // `n_kappa == 0` carve-out — justified only by the bail that just disappeared —
        // this model ran the analytic ln-jet inner (with its ~1e-9
        // provider-vs-`compute_predictions` gap) while the fit AND covariance steps stayed
        // at the loose default, quietly inflating SEs with unchanged point estimates.
        assert!(
            model.uses_closed_form_ltbs_inner(),
            "{label}: a model whose inner runs the analytic ln-jet must also take the \
             tightened LTBS fit/covariance tolerances"
        );

        let subject = Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.8, 0.6, 0.4, 0.7, 0.5, 0.3],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0; 6],
            occasions: vec![1, 1, 1, 2, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        // Off-zero on every coordinate, and distinct per κ block, so a transform applied to
        // the wrong occasion's jet cannot cancel.
        let stacked = vec![0.10, -0.05, 0.08, -0.12];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic IOV inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-6 * (1.0 + stacked[p].abs());
            let mut sp = stacked.clone();
            sp[p] += h;
            let mut sm = stacked.clone();
            sm[p] -= h;
            let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
        }
    }
}

// ----- #555 shared fixtures (two_cpt_oral_cov, η+covariate `obs_scale = V1`) -----

/// Analytical + ODE twin of the ferx-r `two_cpt_oral_cov` model (5 η, `obs_scale = V1`
/// with `V1 = TVV1·(WT/70)^θ·exp(ETA_V1)` — both covariate- and η-dependent). The ODE
/// solver block is caller-supplied so a cheap fixed-η check can run tight and the inner
/// EBE check can run loose.
fn repro555_model_pair(ode_solver_block: &str) -> (CompiledModel, CompiledModel) {
    use crate::parser::model_parser::parse_model_string;
    let header = "[parameters]\n  theta TVCL(5.0,0.1,100.0)\n  theta TVV1(50.0,1.0,500.0)\n  theta TVQ(10.0,0.1,100.0)\n  theta TVV2(100.0,1.0,500.0)\n  theta TVKA(1.2,0.01,10.0)\n  theta THETA_WT(0.75,0.01,5.0)\n  theta THETA_CRCL(0.50,0.01,5.0)\n  omega ETA_CL ~ 0.10\n  omega ETA_V1 ~ 0.10\n  omega ETA_Q ~ 0.05\n  omega ETA_V2 ~ 0.05\n  omega ETA_KA ~ 0.15\n  sigma PROP_ERR ~ 0.02 (sd)\n[individual_parameters]\n  CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)\n  V1 = TVV1 * (WT/70)^THETA_WT * exp(ETA_V1)\n  Q = TVQ * exp(ETA_Q)\n  V2 = TVV2 * exp(ETA_V2)\n  KA = TVKA * exp(ETA_KA)\n";
    let an = parse_model_string(&format!(
            "{header}[structural_model]\n  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)\n[covariates]\n  WT continuous\n  CRCL continuous\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n"
        )).expect("parse analytical");
    let ode = parse_model_string(&format!(
            "{header}[structural_model]\n  ode(obs_cmt=central, states=[depot, central, periph])\n[odes]\n  d/dt(depot) = -KA * depot\n  d/dt(central) = KA*depot - (CL/V1 + Q/V1)*central + (Q/V2)*periph\n  d/dt(periph) = (Q/V1)*central - (Q/V2)*periph\n[scaling]\n  obs_scale = V1\n[covariates]\n  WT continuous\n  CRCL continuous\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n{ode_solver_block}"
        )).expect("parse ode");
    (an, ode)
}

/// Subject 22 of the ferx-r `two_cpt_oral_cov` dataset — the subject that diverged in
/// #555 — inlined verbatim so the regression is self-contained (no external data file).
fn repro555_subject22() -> Subject {
    let mut covariates = HashMap::new();
    covariates.insert("WT".to_string(), 72.1);
    covariates.insert("CRCL".to_string(), 76.7);
    let obs_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0];
    let observations = vec![
        2.0190, 2.4021, 2.2985, 1.8141, 1.4699, 1.2589, 1.0804, 0.8993, 0.7054, 0.5966,
    ];
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: "22".into(),
        doses: vec![DoseEvent::new(0.0, 250.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![2; n],
        covariates,
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
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
        obs_records: vec![],
    }
}

/// #555 guard: at a *fixed* η the ODE `obs_scale = V1` form and its analytical twin
/// must agree on the assembled FOCEI marginal (incl. log|H̃|) to integrator tolerance —
/// the forward IPRED, the ∂f/∂η Jacobian, and the assembly are all path-independent
/// here. (The #555 divergence lives in EBE *convergence*, not this fixed-η objective —
/// see `repro555_ode_exprscale_ebe_finds_global_min`.) Tight ODE tol is cheap: no inner
/// optimisation runs, only a handful of `predict`/Jacobian evaluations.
#[test]
fn repro555_ode_exprscale_marginal_vs_analytical() {
    use crate::stats::likelihood::foce_subject_nll_interaction;
    let (an, ode) = repro555_model_pair("  ode_reltol = 1e-11\n  ode_abstol = 1e-13\n");
    let subject = repro555_subject22();

    let marginal = |m: &CompiledModel, eta: &[f64]| -> f64 {
        let p = &m.default_params;
        let eta_v = nalgebra::DVector::from_column_slice(eta);
        let ipreds = crate::pk::compute_predictions_with_tv(m, &subject, &p.theta, eta);
        let jac = crate::sens::provider::subject_eta_jacobian(m, &subject, &p.theta, eta)
            .expect("analytic jac");
        let h = nalgebra::DMatrix::from_row_slice(subject.obs_times.len(), m.n_eta, &jac);
        foce_subject_nll_interaction(
            &subject,
            &ipreds,
            &eta_v,
            &h,
            &p.omega,
            &p.sigma.values,
            &m.error_spec,
            m.bloq_method,
            &[],
            None,
            m.residual_error_eta,
            None, // no custom residual magnitude (#484) in this model
        )
    };

    for eta in [vec![0.0; 5], vec![0.12, -0.08, 0.05, 0.04, -0.10]] {
        let ma = marginal(&an, &eta);
        let mo = marginal(&ode, &eta);
        approx::assert_relative_eq!(ma, mo, max_relative = 1e-5, epsilon = 1e-4);
    }
}

/// #555 regression: on an ODE model with an η-dependent `[scaling] obs_scale = V1`,
/// the inner EBE must reach the correct posterior mode.
///
/// Root cause: the inner BFGS *reaches* the mode but its gradient norm floors above
/// `tol` (the adaptive ODE solver's non-smoothness caps `gnorm` above `tol`), so it
/// spun to `max_iter` and reported failure; `find_ebe` then discarded that correct η̂
/// and overwrote it with a cold Nelder–Mead restart from η=0, which on this multimodal
/// inner objective settled ~20 NLL units worse, inflating the FOCEI OFV by ~370 on the
/// full dataset (the analytical twin, whose smooth objective lets BFGS satisfy
/// `gnorm < tol`, was the correct reference). The fix is twofold: `find_ebe` now keeps
/// the lower-objective of the BFGS partial and the NM restart (so a `false`-on-a-
/// converged search can never regress the EBE), and the inner BFGS gained a gated
/// objective-stall (`ftol`) stop so it converges at the mode instead of spinning. This
/// test runs at a *moderate* ODE tolerance (not the 1e-10 of the original repro) to
/// stay Tier-1-fast and to exercise the realistic-tolerance path where the stall may
/// not fire and the argmin fallback is what guarantees correctness.
#[test]
fn repro555_ode_exprscale_ebe_finds_global_min() {
    use crate::stats::likelihood::individual_nll;
    let subject = repro555_subject22();

    // Run at the DEFAULT ODE tolerances (`ode_reltol = 1e-4`, no override) — the
    // realistic case a user actually hits, and the one where the gradient-noise floor
    // is high enough that the BFGS objective-stall may never fire, so the argmin
    // fallback is what guarantees the correct EBE. The analytical twin (whose smooth
    // objective lets BFGS satisfy `gnorm < tol`) gives the reference global minimum.
    let (an, ode) = repro555_model_pair("");

    let nll = |m: &CompiledModel, e: &[f64]| {
        let p = &m.default_params;
        individual_nll(m, &subject, &p.theta, e, &p.omega, &p.sigma.values)
    };

    let e_an = find_ebe(&an, &subject, &an.default_params, 300, 1e-7, None, None, 0);
    let eta_an: Vec<f64> = e_an.eta.iter().copied().collect();
    let ref_nll = nll(&an, &eta_an);

    // The ODE form must reach the same global minimum (objectives are identical), and
    // must report it as a converged EBE (not a fallback that left it non-stationary).
    let e_od = find_ebe(
        &ode,
        &subject,
        &ode.default_params,
        300,
        1e-7,
        None,
        None,
        0,
    );
    let eta_od: Vec<f64> = e_od.eta.iter().copied().collect();
    let ode_nll = nll(&ode, &eta_od);

    // Pre-fix the ODE EBE stalled ~20 NLL units high in a spurious basin; the global
    // min is now reached (to integrator tolerance) at the default ODE tolerance too.
    assert!(
            ode_nll <= ref_nll + 0.5,
            "ODE EBE stuck in a spurious basin: inner NLL {ode_nll:.4} vs analytical global min {ref_nll:.4} \
             (eta_ode={eta_od:?}, eta_an={eta_an:?})"
        );
    assert!(
        e_od.converged,
        "ODE EBE should be reported converged at the mode"
    );
}

/// #587 review: the shared inner-EBE fallback keeps the lower-objective **value** of the
/// BFGS partial and the Nelder–Mead restart (the substantive #555 fix — the η̂ value fed
/// to the FOCEI gradient), seeds NM from the partial under `ebe_warm_start`, and discards
/// a non-finite-objective partial. Uses a bimodal 1-D objective (deep well at x=-2,
/// f=-10; shallow well at x=+2, f=-1).
/// The bimodal 1-D objective the `argmin_inner_fallback` tests share — deep well at x = −2
/// (f = −10), shallow well at x = +2 (f = −1) — and its exact gradient.
fn bimodal_obj(x: &[f64]) -> f64 {
    let v = x[0];
    if v < 0.0 {
        (v + 2.0).powi(2) - 10.0
    } else {
        (v - 2.0).powi(2) - 1.0
    }
}

fn bimodal_grad(x: &[f64]) -> Vec<f64> {
    let v = x[0];
    vec![if v < 0.0 {
        2.0 * (v + 2.0)
    } else {
        2.0 * (v - 2.0)
    }]
}

fn never_stall(_: &[f64]) -> bool {
    false
}

fn always_stall(_: &[f64]) -> bool {
    true
}

/// A [`FallbackCertifier`] for an exact objective (no stall stop, no preconditioner).
fn exact_cert<'a>(
    grad: &'a dyn Fn(&[f64]) -> Vec<f64>,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> FallbackCertifier<'a> {
    FallbackCertifier {
        grad,
        precond: None,
        stop_precond: None,
        stall_at: &never_stall,
        n,
        max_iter,
        tol,
    }
}

#[test]
fn argmin_inner_fallback_keeps_better_basin() {
    let obj = bimodal_obj;
    let cert = exact_cert(&bimodal_grad, 1, 200, 1e-8);
    let cold = arm_inner_settings(InnerSettings::default());

    // Partial in the deep (global) well, cold NM seed in the shallow well: the fallback
    // keeps the lower-objective partial rather than overwriting with the shallow NM
    // result (the old behaviour, which on this multimodal objective inflated the OFV).
    let (eta, _) = argmin_inner_fallback(&obj, &[-2.0], &[2.0], &cert);
    assert!((eta[0] + 2.0).abs() < 1e-2, "kept deep well, got {eta:?}");

    // Partial in the shallow well, cold NM seed reaches the deep well: NM wins.
    let (eta2, _) = argmin_inner_fallback(&obj, &[2.0], &[-2.0], &cert);
    assert!(
        (eta2[0] + 2.0).abs() < 1e-2,
        "NM found deeper well, got {eta2:?}"
    );

    // Non-finite partial objective → unusable → NM result is taken.
    let (eta3, _) = argmin_inner_fallback(&obj, &[f64::NAN], &[-2.0], &cert);
    assert!(
        eta3[0].is_finite(),
        "NaN partial must be discarded, got {eta3:?}"
    );

    // `ebe_warm_start` seeds the single NM from the partial (covers the warm branch):
    // from the deep well it stays there even though the cold seed is far away.
    let warm = arm_inner_settings(InnerSettings {
        warm: true,
        ..Default::default()
    });
    let (eta4, _) = argmin_inner_fallback(&obj, &[-2.0], &[5.0], &cert);
    assert!(
        (eta4[0] + 2.0).abs() < 1e-2,
        "warm seed held the deep well, got {eta4:?}"
    );
    drop(warm);
    drop(cold);
}

#[test]
fn ode_iov_skips_nelder_mead_inner_fallback() {
    use crate::parser::model_parser::parse_model_string;
    let ode_iov = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    assert!(skip_ode_iov_nm_fallback(&ode_iov));

    let closed_form_iov = make_iov_model();
    assert!(!skip_ode_iov_nm_fallback(&closed_form_iov));

    // #814: a closed-form transit + IOV model reroutes every subject to its ODE twin, so it
    // carries the same per-vertex ODE cost — it must skip the impractical NM inner fallback
    // like a hand-written [odes] + IOV model, even though its own `ode_spec` is `None`.
    let transit_iov = parse_model_string(TRANSIT_IOV_MODEL).expect("parse transit IOV");
    assert!(
        transit_iov.ode_spec.is_none(),
        "transit primary is analytic"
    );
    assert!(
        transit_iov.absorption_ode_equivalent.is_some(),
        "transit+IOV carries an ODE twin"
    );
    assert!(skip_ode_iov_nm_fallback(&transit_iov));
}

#[test]
fn ode_iov_start_rejects_only_pathological_ode_iov_nll() {
    use crate::parser::model_parser::parse_model_string;
    let ode_iov = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    let closed_form_iov = make_iov_model();

    assert!(!reject_ode_iov_inner_start(&ode_iov, 4, 999.0));
    assert!(reject_ode_iov_inner_start(&ode_iov, 4, 1_001.0));
    assert!(!reject_ode_iov_inner_start(&ode_iov, 20, 4_999.0));
    assert!(reject_ode_iov_inner_start(&ode_iov, 20, 5_001.0));
    assert!(reject_ode_iov_inner_start(&ode_iov, 20, f64::NAN));
    assert!(!reject_ode_iov_inner_start(
        &closed_form_iov,
        4,
        1_000_000.0
    ));

    // #814: the transit + IOV twin puts the model in the ODE+IOV cost class, so the
    // degenerate-warm-start rejection guard applies to it too (inactive before the fix).
    let transit_iov = parse_model_string(TRANSIT_IOV_MODEL).expect("parse transit IOV");
    assert!(!reject_ode_iov_inner_start(&transit_iov, 4, 999.0));
    assert!(reject_ode_iov_inner_start(&transit_iov, 4, 1_001.0));
    assert!(reject_ode_iov_inner_start(&transit_iov, 20, f64::NAN));
}

#[test]
fn inner_stall_enabled_at_iov_tracks_the_effective_model() {
    // #814: the inner objective-stall stop (#555) must key off the *effective* model, not the
    // raw one. #1560: a closed-form transit/IG + IOV subject is no longer on its ODE twin
    // subject-statically — the absorption walk serves it exactly, so it must keep the exact
    // `gnorm < tol` stop — unless κ puts a (dose, interval) pair in flip-flop, where the twin
    // takes it and it needs the stall. Both sides are asserted here, at the same subject.
    use crate::parser::model_parser::parse_model_string;
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 25.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 7.0, 5.0],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    // Plain closed-form IOV (no twin): objective is exact → no stall (bit-identical path).
    let plain = make_iov_model();
    assert!(!inner_stall_enabled_at_iov(
        &plain,
        &subject,
        &plain.default_params.theta,
        &[0.0; 3]
    ));

    // Hand-written [odes] + IOV: RK45 objective → stall (unchanged pre-existing behaviour).
    let ode_iov = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    assert!(inner_stall_enabled_at_iov(
        &ode_iov,
        &subject,
        &ode_iov.default_params.theta,
        &[0.0; 4]
    ));

    // Closed-form transit + IOV. Stacked layout [η_V, κ_occ1, κ_occ2].
    let transit_iov = parse_model_string(TRANSIT_IOV_MODEL).expect("parse transit IOV");
    let theta = transit_iov.default_params.theta.clone();
    assert!(
        transit_iov.ode_spec.is_none(),
        "transit primary is analytic"
    );
    // κ = 0: ke = 0.2/10 = 0.02 against KTR = (3+1)/1 = 4 — in domain on both occasions.
    let in_domain = [0.0, 0.0, 0.0];
    // κ_occ2 = 6: CL = 0.2·e⁶ = 80.7, ke = 8.07 ≥ KTR = 4 — the second dose's pair flips.
    let flipped = [0.0, 0.0, 6.0];

    // The straddle, on the routing the gate is built from.
    assert!(
        std::ptr::eq(
            crate::pk::effective_model_for_eval_iov(&transit_iov, &subject, &theta, &in_domain),
            &transit_iov
        ),
        "fixture drift: κ = 0 must be served by the absorption walk"
    );
    assert!(
        !std::ptr::eq(
            crate::pk::effective_model_for_eval_iov(&transit_iov, &subject, &theta, &flipped),
            &transit_iov
        ),
        "fixture drift: κ_occ2 = 6 must send the subject to the twin"
    );

    // The subject-static half is blind to both: it still names the twin for every κ.
    assert!(inner_stall_enabled(&transit_iov, &subject));

    // Both sides of the gate under test.
    assert!(
        !inner_stall_enabled_at_iov(&transit_iov, &subject, &theta, &in_domain),
        "#1560: a walk-served transit+IOV subject has an exact objective and must keep `gnorm < tol`"
    );
    assert!(
        inner_stall_enabled_at_iov(&transit_iov, &subject, &theta, &flipped),
        "#814: a twin-served transit+IOV subject integrates with RK45 and needs the stall stop"
    );
}

/// Closed-form IOV + `iiv_on_ruv` (#4b): the analytic stacked-η inner gradient
/// (`analytic_eta_nll_gradient_iov`) must match central FD of the inner objective
/// `individual_nll_iov` over `[η_bsv, η_ruv, κ₁..κ_K]` — including the `η_ruv` column
/// (`Σ_j 1 − ε²/v`) and the `exp(2·η_ruv)` residual-variance scaling now woven into
/// the IOV inner gradient. Proves the gate flip (`iov_analytical_supported`) ships a
/// *correct* gradient, not just an enabled one.
#[test]
fn iov_iiv_on_ruv_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + iiv_on_ruv");
    // Gate flip: the closed-form IOV + iiv_on_ruv path is now analytic on both loops.
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    assert!(crate::sens::provider::iov_sens_supported(&model));
    assert!(!analytic_inner_common_bail(&model));

    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![8.0, 6.0, 4.0, 7.0, 5.0, 3.0],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    // Non-zero η_ruv (index 3) so the residual-variance scaling is genuinely exercised.
    let stacked = vec![0.10, -0.05, 0.08, 0.15, 0.05, -0.07];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV + iiv_on_ruv inner gradient");

    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    for p in 0..n_stacked {
        let h = 1e-6 * (1.0 + stacked[p].abs());
        let mut sp = stacked.clone();
        sp[p] += h;
        let mut sm = stacked.clone();
        sm[p] -= h;
        let fd = (nll(&sp) - nll(&sm)) / (2.0 * h);
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-6);
    }
}

/// M3 BLOQ + IOV (#580): the analytic stacked-η inner gradient
/// (`analytic_eta_nll_gradient_iov`) must match central FD of the inner objective
/// `individual_nll_iov` over `[η_bsv, κ₁..κ_K]` when the subject carries M3-censored
/// rows (data term `−logΦ(z)`, matching `individual_nll_iov`'s `−2·m3_logcdf`). The
/// censored `h·m` f-coefficient rides the stacked Jacobian (κ columns included), so
/// the EBE minimises the same censored objective. Proves the gate flip
/// (`iov_analytical_supported` now admits M3) ships a *correct* censored gradient over
/// the stacked layout, not just an enabled one.
#[test]
fn iov_m3_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3");
    model.bloq_method = crate::types::BloqMethod::M3;
    // Gate: M3 + IOV is now analytic on both loops (no `iiv_on_ruv`, so not the
    // FD-only triple). `residual_error_eta` is `None`, so `iov_analytical_supported`
    // does not early-return on M3.
    assert_eq!(model.residual_error_eta, None);
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    assert!(crate::sens::provider::iov_sens_supported(&model));
    assert!(!analytic_inner_common_bail(&model));

    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 6],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        // The last two rows (occasion 2 tail) are M3 left-censored; the `−logΦ`
        // term differentiates them.
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    // Synthesize observations from the model at a reference (η, κ), scaled to 0.85·f
    // — including the censored rows, so the carried LLOQ sits just below the
    // prediction (z ≈ −0.75, the moderate regime where the A&S `log_normal_cdf` and
    // the exact φ in `inv_mills` agree to FD precision; a deep-tail LLOQ would expose
    // only the CDF-approximation floor, not the gradient's correctness).
    let preds = crate::pk::predict_iov(
        &model,
        &subject,
        &params.theta,
        &[0.12, -0.08, 0.2],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV + M3 inner gradient");

    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    // Richardson-extrapolated central FD: the censored `−logΦ` term has sharp
    // curvature on the occasion-2 κ axis, so plain central FD is truncation-limited
    // (~2e-4) there — Richardson removes it and validates the analytic to ~1e-7.
    for p in 0..n_stacked {
        let h = 1e-5 * (1.0 + stacked[p].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut sp = stacked.clone();
            sp[p] += hh;
            let mut sm = stacked.clone();
            sm[p] -= hh;
            (nll(&sp) - nll(&sm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// Right-censored (above-ULOQ, `CENS = -1`) regression for the analytic IOV+M3
/// inner gradient. The objective `individual_nll_iov` scores these rows with the
/// **upper** tail (`m3_logcdf`, `z = (f − ULOQ)/√v`); the analytic gradient must use
/// the same tail. Before the signed `m3_censored_kernel` (review of #591) the kernel
/// always took the lower tail, so this gradient was wrong-signed for `CENS = -1` and
/// `find_ebe_iov` pushed the EBE the wrong way. Same model/fixture as
/// `iov_m3_inner_grad_matches_fd` with the occasion-2 tail flipped to `CENS = -1`.
#[test]
fn iov_m3_right_censored_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3");
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(crate::sens::provider::iov_analytical_supported(&model));

    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 6],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        // Occasion-2 tail is M3 *right*-censored (above ULOQ): upper tail.
        cens: vec![0, 0, 0, 0, -1, -1],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    // Carry ULOQ at 0.85·f for the censored rows, so z = (f − ULOQ)/√v ≈ +0.15·f/√v
    // sits in the moderate upper-tail regime (Φ(z) well away from 0/1), where the
    // A&S `log_normal_cdf` and the exact φ in `inv_mills` agree to FD precision.
    let preds = crate::pk::predict_iov(
        &model,
        &subject,
        &params.theta,
        &[0.12, -0.08, 0.2],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV + M3 inner gradient (right-censored)");

    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    for p in 0..n_stacked {
        let h = 1e-5 * (1.0 + stacked[p].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut sp = stacked.clone();
            sp[p] += hh;
            let mut sm = stacked.clone();
            sm[p] -= hh;
            (nll(&sp) - nll(&sm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// **ODE** M3 BLOQ + IOV (#486): the ODE counterpart of
/// [`iov_m3_inner_grad_matches_fd`]. The analytic stacked-η inner gradient produced
/// via the **event-driven ODE sensitivity walk** (`ode_subject_eta_grad_iov`, not the
/// closed-form Dual1 walk) must match Richardson central FD of `individual_nll_iov`
/// over `[η_bsv, κ₁, κ₂]` on a censored subject. Censoring is provider-agnostic —
/// the `−logΦ` coefficient rides the same `residual_inner_obs` path keyed on
/// `subject.cens[j]` whether the walk was closed-form or ODE — so removing the gate
/// clause is all that was needed. Both tails (`CENS = 1` left, `CENS = -1` right).
#[test]
fn analytic_iov_inner_gradient_m3_matches_fd_on_ode_bloq() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  bloq_method = m3\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV + M3");
    assert!(
        matches!(model.bloq_method, crate::types::BloqMethod::M3),
        "model must be M3"
    );
    assert!(model.is_ode_based(), "must be on the ODE path");
    // After the #486 gate flip, the ODE IOV walk serves M3 analytically on the inner
    // loop (single gate — no separate M3 bail).
    assert!(crate::sens::provider::iov_sens_supported(&model));
    assert!(!analytic_inner_common_bail(&model));

    // Both tails: occasion-2 tail left-censored (CENS=1), then right-censored (CENS=-1).
    for cens_sign in [1i8, -1] {
        let mut subject = Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0, 0, 0, cens_sign, cens_sign],
            occasions: vec![1, 1, 1, 2, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        // Carry the censoring limit at 0.85·f so z = (f − LIMIT)/√v sits in the
        // moderate regime where the A&S `log_normal_cdf` and the exact φ in `inv_mills`
        // agree to FD precision (a deep-tail limit would expose only the CDF floor).
        let preds = crate::pk::predict_iov(
            &model,
            &subject,
            &params.theta,
            &[0.12, -0.08, 0.2],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        let stacked = vec![0.10, -0.05, 0.08, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic ODE IOV + M3 inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-5 * (1.0 + stacked[p].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut sp = stacked.clone();
                sp[p] += hh;
                let mut sm = stacked.clone();
                sm[p] -= hh;
                (nll(&sp) - nll(&sm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-5);
        }
    }
}

/// **ODE** triple M3 + IOV + `iiv_on_ruv` (#486): the ODE counterpart of
/// [`iov_m3_iiv_on_ruv_inner_grad_matches_fd`]. The analytic stacked-η inner gradient
/// from the **event-driven ODE walk** must match Richardson FD of `individual_nll_iov`
/// over `[η_bsv, η_ruv, κ₁, κ₂]` when censored rows co-occur with the `exp(2·η_ruv)`
/// residual-variance scaling. The ODE walk emits a zero `∂f/∂η_ruv` column (η_ruv is
/// absent from CL/V/KA), so the residual-eta column comes entirely from the
/// provider-agnostic `residual_inner_obs` term — exactly as on the closed-form path.
/// Both tails.
#[test]
fn analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot / V - (CL/V) * central\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  bloq_method = m3\n  iov_column = OCC\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE IOV + M3 + iiv_on_ruv");
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(model.is_ode_based(), "must be on the ODE path");
    // The ODE triple is analytic on both loops as of #486.
    assert!(crate::sens::provider::iov_sens_supported(&model));
    assert!(!analytic_inner_common_bail(&model));

    for cens_sign in [1i8, -1] {
        let mut subject = Subject {
            obs_rgroup: Vec::new(),
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
            ],
            obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
            obs_raw_times: Vec::new(),
            observations: vec![0.0; 6],
            obs_cmts: vec![1; 6],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_raw_times: Vec::new(),
            reset_period_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            reset_covariates: Vec::new(),
            cens: vec![0, 0, 0, 0, cens_sign, cens_sign],
            occasions: vec![1, 1, 1, 2, 2, 2],
            obs_l2: Vec::new(),
            dose_occasions: vec![1, 2],
            reset_occasions: Vec::new(),
            pk_only_occasions: Vec::new(),
            fremtype: Vec::new(),
            obs_records: vec![],
        };
        let params = model.default_params.clone();
        let preds = crate::pk::predict_iov(
            &model,
            &subject,
            &params.theta,
            &[0.12, -0.08, 0.2, 0.10],
            &[vec![0.05], vec![-0.07]],
        );
        subject.observations = preds.iter().map(|p| p * 0.85).collect();
        let n_eta = model.n_eta;
        let n_kappa = model.n_kappa;
        let k = iov_occasion_groups(&subject).len();
        let n_stacked = n_eta + k * n_kappa;
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
        // Non-zero η_ruv (index 3) so the residual-variance scaling is exercised.
        let stacked = vec![0.10, -0.05, 0.08, 0.12, 0.05, -0.07];
        assert_eq!(stacked.len(), n_stacked);

        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            omega_iov,
            &params.sigma.values,
            n_eta,
            n_kappa,
            k,
            None,
        )
        .expect("analytic ODE IOV + M3 + iiv_on_ruv inner gradient");

        let nll = |s: &[f64]| -> f64 {
            let eta_t = &s[..n_eta];
            let kappas: Vec<Vec<f64>> = (0..k)
                .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
                .collect();
            individual_nll_iov(
                &model,
                &subject,
                &params.theta,
                eta_t,
                &kappas,
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        for p in 0..n_stacked {
            let h = 1e-5 * (1.0 + stacked[p].abs());
            let fd_at = |hh: f64| -> f64 {
                let mut sp = stacked.clone();
                sp[p] += hh;
                let mut sm = stacked.clone();
                sm[p] -= hh;
                (nll(&sp) - nll(&sm)) / (2.0 * hh)
            };
            let f1 = fd_at(h);
            let f2 = fd_at(h / 2.0);
            let fd = (4.0 * f2 - f1) / 3.0;
            approx::assert_relative_eq!(g[p], fd, max_relative = 1e-4, epsilon = 1e-5);
        }
    }
}

/// The triple **M3 + IOV + `iiv_on_ruv`** (#591): the analytic stacked-η inner
/// gradient (`analytic_eta_nll_gradient_iov`) must match Richardson FD of
/// `individual_nll_iov` over `[η_bsv, η_ruv, κ₁..κ_K]` when censored rows co-occur with
/// the `exp(2·η_ruv)` residual-variance scaling. `residual_inner_obs` returns the
/// censored `(h·m, h·z)` pair (f-coefficient + residual-eta column) on a censored row
/// under `iiv_on_ruv`, and the residual variance carries the `η_ruv` scale on every
/// row. Proves the gate flip ships a correct *triple* inner gradient.
#[test]
fn iov_m3_iiv_on_ruv_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.05\n  kappa KAPPA_CL ~ 0.02\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + M3 + iiv_on_ruv");
    model.bloq_method = crate::types::BloqMethod::M3;
    // The closed-form triple is analytic on both loops as of #591; the ODE IOV triple
    // is analytic as of #486 (see `analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode`).
    assert_eq!(model.residual_error_eta, Some(3));
    assert!(crate::sens::provider::iov_analytical_supported(&model));
    assert!(crate::sens::provider::iov_sens_supported(&model));
    assert!(!analytic_inner_common_bail(&model));

    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 6.0, 12.0, 25.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 6],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        // Occasion-2 tail rows M3 left-censored, co-occurring with iiv_on_ruv.
        cens: vec![0, 0, 0, 0, 1, 1],
        occasions: vec![1, 1, 1, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    // Shallow censoring (≈ 0.85·f → z ≈ −0.75), the regime where the A&S CDF and the
    // exact φ in `inv_mills` agree to FD precision.
    let preds = crate::pk::predict_iov(
        &model,
        &subject,
        &params.theta,
        &[0.12, -0.08, 0.2, 0.10],
        &[vec![0.05], vec![-0.07]],
    );
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;
    let k = iov_occasion_groups(&subject).len();
    let n_stacked = n_eta + k * n_kappa;
    let omega_iov = params.omega_iov.as_ref().expect("omega_iov present");
    // Non-zero η_ruv (index 3) so the residual-variance scaling is exercised.
    let stacked = vec![0.10, -0.05, 0.08, 0.12, 0.05, -0.07];
    assert_eq!(stacked.len(), n_stacked);

    let g = analytic_eta_nll_gradient_iov(
        &model,
        &subject,
        &params.theta,
        &stacked,
        &params.omega,
        omega_iov,
        &params.sigma.values,
        n_eta,
        n_kappa,
        k,
        None,
    )
    .expect("analytic IOV + M3 + iiv_on_ruv inner gradient");

    let nll = |s: &[f64]| -> f64 {
        let eta_t = &s[..n_eta];
        let kappas: Vec<Vec<f64>> = (0..k)
            .map(|kk| s[n_eta + kk * n_kappa..n_eta + (kk + 1) * n_kappa].to_vec())
            .collect();
        individual_nll_iov(
            &model,
            &subject,
            &params.theta,
            eta_t,
            &kappas,
            &params.omega,
            Some(omega_iov),
            &params.sigma.values,
        )
    };
    for p in 0..n_stacked {
        let h = 1e-5 * (1.0 + stacked[p].abs());
        let fd_at = |hh: f64| -> f64 {
            let mut sp = stacked.clone();
            sp[p] += hh;
            let mut sm = stacked.clone();
            sm[p] -= hh;
            (nll(&sp) - nll(&sm)) / (2.0 * hh)
        };
        let f1 = fd_at(h);
        let f2 = fd_at(h / 2.0);
        let fd = (4.0 * f2 - f1) / 3.0;
        approx::assert_relative_eq!(g[p], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// The IOV inner loop must honour the same model-level FD bails as the non-IOV inner
/// (#466 review #1/#3): `gradient = fd` / escape hatch, IIV-on-residual-error
/// (`iiv_on_ruv`), and LTBS all force the FD inner gradient — the shared
/// `analytic_inner_common_bail` gate `find_ebe_iov` now consults. Without it an
/// IOV + `iiv_on_ruv` fit would build the inner gradient on an unscaled residual
/// variance, and `gradient = fd` would silently fail to disable the analytic inner.
/// (`ExpressionScale` is no longer a common bail — the non-IOV analytical inner serves
/// it via the quotient rule. The **ODE** IOV path now serves an η-dependent `obs_scale`
/// too via the post-walk quotient (#575/#590); the **closed-form** IOV
/// path still declines it (`iov_analytical_supported` requires `ScalingSpec::None`) —
/// both pinned by the `iov_sens_supported` assertions below.)
#[test]
fn iov_inner_honours_common_bails() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV");
    // Clean IOV model: no common bail → analytic inner runs.
    assert!(!analytic_inner_common_bail(&model));
    // `gradient = fd` (and the escape hatch) force FD (#466 review #3).
    model.gradient_method = GradientMethod::Fd;
    assert!(analytic_inner_common_bail(&model));
    model.gradient_method = GradientMethod::default();
    assert!(!analytic_inner_common_bail(&model));
    // IIV on residual error is NO LONGER a blanket common bail (#4b): the inner
    // gradient now carries the `exp(2·η_ruv)` scaling and the `η_ruv` variance column.
    // For this **IOV** model (`n_kappa > 0`), even the M3 triple is analytic as of #486;
    // the non-IOV ODE M3 + `iiv_on_ruv` combo is analytic as well (#623).
    model.residual_error_eta = Some(0);
    assert!(!analytic_inner_common_bail(&model));
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(!analytic_inner_common_bail(&model));
    model.bloq_method = crate::types::BloqMethod::Drop;
    model.residual_error_eta = None;
    // LTBS is no longer a common bail at any κ count (#486): the closed-form IOV inner walk
    // applies the `ln` jet last (`run_obs_iov_eta`), matching its outer twin. This model is
    // **ODE**-based, though, and the ODE IOV path is a different story — `ode_iov_supported`
    // declines `log_transform` outright (its walk applies LTBS in PK-param space *before* the
    // η/θ/κ chain, so the production scale-then-log order can't be rebuilt post-walk). So the
    // decline for an ODE IOV LTBS model must come from the support gate, NOT from the common
    // bail. Pin both halves: dropping the bail clause must not have opened an analytic inner
    // for a model whose outer still declines (that split is exactly what the scope rule forbids).
    model.log_transform = true;
    assert!(
        !analytic_inner_common_bail(&model),
        "LTBS is no longer a common bail — the closed-form IOV inner serves it (#486)"
    );
    assert!(
        !crate::sens::provider::iov_sens_supported(&model),
        "ODE IOV + LTBS still routes to FD via the ODE IOV support gate — the sole decline now"
    );
    model.log_transform = false;

    // The *outer* IOV gate (`iov_sens_supported`) for this **ODE** model now admits
    // `iiv_on_ruv` and the M3 triple (#486): the ODE walk emits a zero `∂f/∂η_ruv`
    // column and the shared assembly applies the variance scaling — proven by the
    // dedicated FD-comparison tests (`analytic_iov_inner_gradient_m3_iiv_on_ruv_matches_fd_on_ode`
    // here and `iov_iiv_on_ruv_ode_packed_gradient_matches_reconverged_fd` in
    // `sens_outer_gradient`). FREM still routes to FD.
    assert!(crate::sens::provider::iov_sens_supported(&model));
    model.residual_error_eta = Some(0);
    assert!(
        crate::sens::provider::iov_sens_supported(&model),
        "ODE IOV + iiv_on_ruv is analytic as of #486"
    );
    model.residual_error_eta = None;
    assert!(crate::sens::provider::iov_sens_supported(&model));

    // ODE IOV + η-dependent `ExpressionScale` `obs_scale` is analytic (#575/#590):
    // the post-walk quotient carries `d(obs_scale)/d(stacked-η)`, so `ode_iov_supported`
    // (and hence `iov_sens_supported`) admits it. LTBS still declines — pinned in
    // `sens::provider::tests::ode_iov_expr_scale_supported_and_gated`.
    let iov_scaled = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse ODE IOV + obs_scale");
    assert!(
        matches!(
            iov_scaled.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ),
        "obs_scale = 1000/V must parse as an η-dependent ExpressionScale"
    );
    assert!(
        crate::sens::provider::iov_sens_supported(&iov_scaled),
        "ODE IOV + η-dependent obs_scale is analytic via the post-walk quotient (#575/#590)"
    );

    // The CLOSED-FORM IOV path (`iov_analytical_supported`) now admits an η-dependent
    // `ExpressionScale` `obs_scale` too (#486): the closed-form event-driven walk applies
    // the same per-occasion-group post-walk quotient as the ODE path. LTBS is served on
    // the OUTER gradient now (#486) — pinned in
    // `sens::provider::tests::iov_analytical_expr_scale_supported_and_gated`.
    let iov_scaled_cf = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
        )
        .expect("parse closed-form IOV + obs_scale");
    assert!(matches!(
        iov_scaled_cf.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    assert!(
        crate::sens::provider::iov_sens_supported(&iov_scaled_cf),
        "closed-form IOV + η-dependent obs_scale is analytic via the post-walk quotient (#486)"
    );
    let mut iov_scaled_cf_ltbs = iov_scaled_cf;
    iov_scaled_cf_ltbs.log_transform = true;
    // Closed-form IOV + obs_scale + LTBS is served on BOTH loops now (#486): each applies the
    // `ln(f)` jet after its own per-occasion scale quotient, so the EBE and the θ/Ω/σ gradient
    // differentiate the same `ln(f/s)`. Pin the pair together — an analytic outer against an FD
    // inner is the scope split the provider forbids, so these two must move as one.
    assert!(
        crate::sens::provider::iov_sens_supported(&iov_scaled_cf_ltbs),
        "closed-form IOV + obs_scale + LTBS is served on the OUTER gradient (#486)"
    );
    assert!(
        !analytic_inner_common_bail(&iov_scaled_cf_ltbs),
        "closed-form IOV + obs_scale + LTBS is served on the INNER gradient too (#486)"
    );
}

/// Pinning `inner_optimizer` to dense BFGS vs L-BFGS must reach the *same* EBE
/// — both are gradient-based solvers of the same convex inner objective, so the
/// explicit choice only changes the path, not the stationary point. Guards the
/// `inner_optimizer` dispatch (and that pinning bypasses the size threshold).
#[test]
fn inner_optimizer_pin_reaches_same_ebe() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::InnerOptimizer;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse");
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
        obs_raw_times: Vec::new(),
        observations: vec![18.0, 16.0, 13.0, 9.0, 4.5, 2.2],
        obs_cmts: vec![1; 6],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 6],
        occasions: vec![1; 6],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();

    let solve = |mode| {
        let _mode = arm_inner_settings(InnerSettings {
            mode,
            ..Default::default()
        });
        find_ebe(&model, &subject, &params, 200, 1e-8, None, None, 0)
    };
    let bfgs = solve(InnerOptimizer::Bfgs);
    let lbfgs = solve(InnerOptimizer::Lbfgs);

    assert!(bfgs.converged && lbfgs.converged, "both must converge");
    for k in 0..model.n_eta {
        approx::assert_relative_eq!(
            bfgs.eta[k],
            lbfgs.eta[k],
            max_relative = 1e-5,
            epsilon = 1e-7
        );
    }
}

/// IIV on residual error (#474): the closed-form inner η-gradient must match a
/// The inner-gradient model gate accepts a closed-form `iiv_on_ruv` model AND
/// closed-form **M3 BLOQ + `iiv_on_ruv`** (#4c — the censored × residual-eta
/// cross-terms are assembled). The **non-IOV ODE** M3 + `iiv_on_ruv` combo is analytic
/// as well (#623), as is the ODE *IOV* triple (#486).
#[test]
fn analytic_inner_grad_gate_iiv_on_ruv() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = parse_model_string(
            "[parameters]\n  theta TVCL(0.13,0.001,10.0)\n  theta TVV(8.0,0.1,500.0)\n  theta TVKA(1.0,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n",
        )
        .expect("parse");
    assert!(analytic_inner_grad_supported_model(&model));
    // #4c: closed-form M3 + iiv_on_ruv is now analytic (was FD).
    model.bloq_method = crate::types::BloqMethod::M3;
    assert!(analytic_inner_grad_supported_model(&model));
}

/// central finite difference of the production `individual_nll` (which applies
/// the `exp(2·η_ruv)` variance scaling) at a non-zero η — including the `η_ruv`
/// column, which the shared `coef·∂f/∂η` loop never touches (`∂f/∂η_ruv = 0`).
#[test]
fn analytic_eta_gradient_matches_fd_iiv_on_ruv() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.13,0.001,10.0)\n  theta TVV(8.0,0.1,500.0)\n  theta TVKA(1.0,0.01,50.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.1 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n",
        )
        .expect("parse");
    assert_eq!(model.residual_error_eta, Some(3));
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![2.1, 3.4, 4.0, 3.1, 1.8, 1.1, 0.4],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    // A genuinely non-zero η, including the residual-error component.
    check_inner_ruv_grad(&model, &subject, &[0.20, -0.15, 0.30, 0.25]);
}

/// ODE counterpart of [`analytic_eta_gradient_matches_fd_iiv_on_ruv`]: the
/// residual-variance scaling and `η_ruv` column live in the shared, provider-
/// agnostic `analytic_eta_nll_gradient`, so the light ODE `Dual1` walk serves
/// `iiv_on_ruv` too (#474). Verified against FD of the production `individual_nll`.
#[test]
fn analytic_eta_gradient_matches_fd_iiv_on_ruv_ode() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
    assert_eq!(model.residual_error_eta, Some(2));
    assert!(model.ode_spec.is_some());
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![28.0, 25.0, 20.0, 13.0, 5.5, 2.4, 0.5],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    check_inner_ruv_grad(&model, &subject, &[0.15, -0.10, 0.25]);
}

/// Compare the analytic inner η-gradient to a central FD of the production
/// `individual_nll` (which scales the residual variance by `exp(2·η_ruv)`) at a
/// non-zero η — including the `η_ruv` column that `∂f/∂η = 0` leaves to the
/// variance term.
fn check_inner_ruv_grad(model: &CompiledModel, subject: &Subject, eta: &[f64]) {
    let params = model.default_params.clone();
    let analytic = analytic_eta_nll_gradient(
        model,
        subject,
        &params.theta,
        eta,
        &params.omega,
        &params.sigma.values,
    )
    .expect("ruv model is in analytic inner scope");

    let nll = |e: &[f64]| {
        crate::stats::likelihood::individual_nll(
            model,
            subject,
            &params.theta,
            e,
            &params.omega,
            &params.sigma.values,
        )
    };
    for k in 0..model.n_eta {
        let h = 1e-6 * (1.0 + eta[k].abs());
        let mut ep = eta.to_vec();
        ep[k] += h;
        let mut em = eta.to_vec();
        em[k] -= h;
        let fd = (nll(&ep) - nll(&em)) / (2.0 * h);
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// Dense-`R` (`block_sigma`, #627) analytic inner η-gradient must match a central
/// FD of the production `individual_nll`, which routes a correlated-residual model
/// through the dense data term. The `combined(PROP,ADD)` + correlated `block_sigma`
/// modifies the diagonal residual variance and its `∂/∂f` (the within-observation
/// `2ρσ_iσ_j c_i c_j` cross term) — exactly the term the plain scalar `dvar_df` omits
/// and the dense kernel (`compute_dr_df_matrices`) carries. Pins that the dense inner
/// branch reduces correctly and stays consistent with the marginal it optimises.
#[test]
fn dense_residual_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse correlated block_sigma model");
    assert!(
        !model.residual_correlations.is_empty(),
        "fixture must carry a residual correlation"
    );
    assert!(
        !analytic_inner_common_bail(&model),
        "block_sigma must now take the analytic inner gradient (#627)"
    );

    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 5],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![1; 5],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    // Nonzero residuals: predictions at η=0.1 nudged, scored at a different η.
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1],
    );
    subject.observations = preds.iter().map(|p| p * 1.15 + 0.3).collect();
    check_inner_ruv_grad(&model, &subject, &[-0.2]);
}

/// ODE variant of [`dense_residual_inner_grad_matches_fd`]: the dense-R inner branch
/// reuses the light `Dual1` ODE walk's per-obs `∂f/∂η`, so the same correlated
/// `block_sigma` model on an ODE structural model must also match FD of `individual_nll`.
/// Exercises the ODE-branch gate flip (`analytic_inner_grad_supported`, #627).
#[test]
fn dense_residual_ode_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse ODE correlated block_sigma model");
    assert!(!model.residual_correlations.is_empty());
    assert!(model.ode_spec.is_some());
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 5],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![1; 5],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    assert!(
        analytic_inner_grad_supported(&model, &subject),
        "ODE block_sigma subject should take the analytic (Dual1) inner gradient (#627)"
    );
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1],
    );
    subject.observations = preds.iter().map(|p| p * 1.15 + 0.3).collect();
    check_inner_ruv_grad(&model, &subject, &[-0.2]);
}

/// `block_sigma` + η-dependent `ExpressionScale` `obs_scale` (#627 × #486): the dense
/// inner gradient must still match FD of `individual_nll`. The scale enters only
/// through `∂f/∂η` (quotient rule, from the provider) and the scaled prediction that
/// `R` is built on, so the dense branch composes with it. Pins the numerical side of
/// `expression_scale_with_correlated_residual_is_analytic_both_loops`.
#[test]
fn dense_expression_scale_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(1.0, 0.01, 10.0) FIX\n  theta TVV(10.0, 0.1, 100.0) FIX\n  omega ETA_CL ~ 0.09 FIX\n  omega ETA_V ~ 0.04 FIX\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.05, 1.00] FIX\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse block_sigma + ExpressionScale");
    assert!(matches!(
        model.scaling,
        crate::types::ScalingSpec::ExpressionScale { .. }
    ));
    assert!(!analytic_inner_common_bail(&model));
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 5],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![1; 5],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1, -0.05],
    );
    subject.observations = preds.iter().map(|p| p * 1.1 + 5.0).collect();
    check_inner_ruv_grad(&model, &subject, &[-0.15, 0.12]);
}

/// Parse an ODE + LTBS (`log_additive`) + `iiv_on_ruv` model and build a
/// subject with log-scale observations, shared by the two ODE-LTBS inner tests.
fn ode_ltbs_ruv_model_and_subject() -> (CompiledModel, Subject) {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
    assert!(model.log_transform, "log_additive must set LTBS");
    assert_eq!(model.residual_error_eta, Some(2));
    // Predictions for an LTBS model are on the log scale; perturb them so the
    // residual is nonzero.
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 7],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1, -0.1, 0.0],
    );
    subject.observations = preds.iter().map(|p| p + 0.2).collect();
    (model, subject)
}

/// ODE + LTBS + `iiv_on_ruv`: the analytic inner η-gradient must match a central
/// FD of the production `individual_nll` (which applies the `g = ln(f)` wrap and
/// the `exp(2·η_ruv)` scale). Confirms the residual-eta column and the log chain
/// compose correctly (#474).
#[test]
fn ode_ltbs_inner_grad_matches_fd() {
    let (model, subject) = ode_ltbs_ruv_model_and_subject();
    check_inner_ruv_grad(&model, &subject, &[0.15, -0.10, 0.20]);
}

/// The covariance concern that kept LTBS on the FD inner (#438): the analytic
/// EBE must coincide with the FD (objective's-own) EBE. For ODE the Dual1 walk
/// shares `solve_ode_g` with `individual_nll`, so they agree to integrator
/// tolerance — leaving the covariance Hessian clean (#474).
#[test]
fn ode_ltbs_inner_ebe_matches_fd() {
    let (mut model, subject) = ode_ltbs_ruv_model_and_subject();
    let params = model.default_params.clone();

    model.gradient_method = GradientMethod::Auto; // analytic inner
    assert!(
        analytic_inner_grad_supported(&model, &subject),
        "ODE-LTBS subject should now take the analytic inner gradient"
    );
    let analytic = find_ebe(&model, &subject, &params, 200, 1e-10, None, None, 0);

    model.gradient_method = GradientMethod::Fd; // force FD inner
    assert!(!analytic_inner_grad_supported(&model, &subject));
    let fd = find_ebe(&model, &subject, &params, 200, 1e-10, None, None, 0);

    assert!(
        analytic.converged && fd.converged,
        "both EBE solves converge"
    );
    for k in 0..model.n_eta {
        approx::assert_relative_eq!(
            analytic.eta[k],
            fd.eta[k],
            max_relative = 1e-5,
            epsilon = 1e-7
        );
    }
}

/// #576/#486: an ODE model carrying a custom residual-error magnitude now
/// takes the analytic inner EBE gradient too — `residual_inner_obs` (shared by
/// the closed-form and ODE inner paths) threads the η-independent
/// per-observation multiplier into the variance/its `f`-derivative, so the
/// gradient stays magnitude-aware without falling back to FD. A control model
/// without the magnitude is unaffected (same analytic route as before).
#[test]
fn ode_custom_magnitude_takes_analytic_inner_gradient() {
    use crate::parser::model_parser::parse_model_string;
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![5.0, 4.0, 3.0, 2.0, 1.0],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![1; 5],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    // Control: plain proportional ODE → analytic inner gradient.
    let mut plain = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n",
        )
        .expect("parse plain ODE");
    plain.gradient_method = GradientMethod::Auto;
    assert!(!plain.has_custom_ruv_magnitude());
    assert!(
        analytic_inner_grad_supported(&plain, &subject),
        "plain ODE model should take the analytic inner gradient"
    );

    // Same model with a TIME-varying residual magnitude → must bail to FD.
    let mut mag = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  theta RUV_LATE(1.5,0.0,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR * (if (TIME > 4.0) RUV_LATE else 1.0))\n[fit_options]\n  method = focei\n",
        )
        .expect("parse ODE + custom magnitude");
    mag.gradient_method = GradientMethod::Auto;
    assert!(
        mag.has_custom_ruv_magnitude(),
        "fixture must carry a custom residual magnitude"
    );
    assert!(
        analytic_inner_grad_supported(&mag, &subject),
        "ODE + custom magnitude should take the analytic inner gradient (#576/#486)"
    );
}

/// #576/#486: the closed-form analytic inner η-gradient of a custom / time-
/// varying residual-magnitude model must match FD of the (already magnitude-
/// aware) `individual_nll` — the magnitude is η-independent, so
/// `residual_inner_obs` only needs the per-observation multiplier threaded
/// into the variance/its `f`-derivative, no new η term.
#[test]
fn magnitude_inner_eta_gradient_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  theta RUV_LATE(1.5,0.1,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n",
        )
        .expect("parse magnitude model");
    assert!(model.has_custom_ruv_magnitude());
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 7],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let theta = vec![0.22, 11.0, 1.4, 1.6];
    let preds =
        crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &[0.1, -0.1, 0.05]);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let eta = [0.15_f64, -0.10, 0.20];
    let analytic = analytic_eta_nll_gradient(
        &model,
        &subject,
        &params.theta,
        &eta,
        &params.omega,
        &params.sigma.values,
    )
    .expect("magnitude model is in the analytic inner scope");
    for k in 0..model.n_eta {
        let h = 1e-6 * (1.0 + eta[k].abs());
        let mut ep = eta;
        ep[k] += h;
        let mut em = eta;
        em[k] -= h;
        let nllp = crate::stats::likelihood::individual_nll(
            &model,
            &subject,
            &params.theta,
            &ep,
            &params.omega,
            &params.sigma.values,
        );
        let nllm = crate::stats::likelihood::individual_nll(
            &model,
            &subject,
            &params.theta,
            &em,
            &params.omega,
            &params.sigma.values,
        );
        let fd = (nllp - nllm) / (2.0 * h);
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// Gate test: `SDE` diffusion combined with a custom residual magnitude must
/// still route the inner gradient to FD — #576/#486 relaxes the plain
/// magnitude bail in `analytic_inner_common_bail`, but SDE stays its own,
/// independent reason to decline (`model.is_sde()`).
#[test]
fn magnitude_with_sde_still_routes_inner_to_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  theta RUV_LATE(1.5,0.1,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(obs_cmt=central, states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[diffusion]\n  central ~ 0.05 FIX\n[error_model]\n  DV ~ proportional(PROP_ERR * (1.0 + RUV_LATE * TIME / 48.0))\n",
        )
        .expect("parse SDE + magnitude model");
    assert!(model.is_sde(), "fixture must be an SDE model");
    assert!(model.has_custom_ruv_magnitude());
    assert!(
        analytic_inner_common_bail(&model),
        "SDE + custom magnitude must still bail the inner gradient to FD"
    );
}

/// Parse a plain ODE + LTBS (`log_additive`) model with **no** `iiv_on_ruv`
/// — the exact #438 case — and a subject with log-scale observations.
fn ode_ltbs_no_ruv_model_and_subject() -> (CompiledModel, Subject) {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n[fit_options]\n  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n",
        )
        .expect("parse");
    assert!(model.log_transform, "log_additive must set LTBS");
    assert_eq!(model.residual_error_eta, None, "no iiv_on_ruv");
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 7],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1, -0.1],
    );
    subject.observations = preds.iter().map(|p| p + 0.2).collect();
    (model, subject)
}

/// #438 regression: PR #474 flipped plain ODE + LTBS (no `iiv_on_ruv`) onto the
/// analytic inner. The #438 concern was that the analytic EBE could drift off the
/// objective's own EBE and inflate the covariance Hessian (~5× SEs). For the ODE
/// `Dual1` walk that shares `solve_ode_g` with `individual_nll` this must NOT
/// happen — the analytic and FD EBEs must coincide to integrator tolerance.
#[test]
fn ode_ltbs_no_ruv_inner_ebe_matches_fd() {
    let (mut model, subject) = ode_ltbs_no_ruv_model_and_subject();
    let params = model.default_params.clone();

    model.gradient_method = GradientMethod::Auto; // analytic inner
    assert!(
        analytic_inner_grad_supported(&model, &subject),
        "plain ODE-LTBS subject should take the analytic inner gradient"
    );
    let analytic = find_ebe(&model, &subject, &params, 200, 1e-10, None, None, 0);

    model.gradient_method = GradientMethod::Fd; // force FD inner
    assert!(!analytic_inner_grad_supported(&model, &subject));
    let fd = find_ebe(&model, &subject, &params, 200, 1e-10, None, None, 0);

    assert!(
        analytic.converged && fd.converged,
        "both EBE solves converge"
    );
    for k in 0..model.n_eta {
        // Two independently-converged EBE solves; agreement to ~1e-4 confirms
        // no #438-style drift (which inflated SEs ~5×, i.e. ~400% off).
        approx::assert_relative_eq!(
            analytic.eta[k],
            fd.eta[k],
            max_relative = 1e-4,
            epsilon = 1e-6
        );
    }
}

/// ODE + LTBS + `iiv_on_ruv` with an **eta-dependent initial condition**
/// (`init(central) = C0·V`, as in the thioguanine `run14` model). The analytic
/// inner gradient must still match FD of `individual_nll` — confirms the init-
/// condition η-derivative composes with the log wrap and residual-eta column.
#[test]
fn ode_ltbs_init_cond_inner_grad_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let src = "[parameters]\n  theta TVCL(4.0,0.1,100.0)\n  theta TVV(30.0,1.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_RUV ~ 0.10\n  sigma ADD_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  C0 = 5.0\n[structural_model]\n  ode(states=[central])\n[odes]\n  init(central) = C0 * V\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ log_additive(ADD_ERR)\n  iiv_on_ruv = ETA_RUV\n[fit_options]\n  ode_reltol = 1e-11\n  ode_abstol = 1e-13\n";
    let model = parse_model_string(src).expect("parse");
    let mut subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
        obs_raw_times: Vec::new(),
        observations: vec![0.0; 7],
        obs_cmts: vec![1; 7],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 7],
        occasions: vec![1; 7],
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let preds = crate::pk::compute_predictions_with_tv(
        &model,
        &subject,
        &model.default_params.theta,
        &[0.1, -0.1, 0.0],
    );
    subject.observations = preds.iter().map(|p| p + 0.2).collect();
    let params = model.default_params.clone();
    let eta = [0.15_f64, -0.10, 0.20];
    let analytic = analytic_eta_nll_gradient(
        &model,
        &subject,
        &params.theta,
        &eta,
        &params.omega,
        &params.sigma.values,
    )
    .expect("scope");
    for k in 0..model.n_eta {
        let h = 1e-6 * (1.0 + eta[k].abs());
        let mut ep = eta;
        ep[k] += h;
        let mut em = eta;
        em[k] -= h;
        let nllp = crate::stats::likelihood::individual_nll(
            &model,
            &subject,
            &params.theta,
            &ep,
            &params.omega,
            &params.sigma.values,
        );
        let nllm = crate::stats::likelihood::individual_nll(
            &model,
            &subject,
            &params.theta,
            &em,
            &params.omega,
            &params.sigma.values,
        );
        let fd = (nllp - nllm) / (2.0 * h);
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// The inner settings the EBE solves read are armed per thread and restored LIFO, a panic
/// included, and an unarmed thread reads the defaults (`Auto`, warm start off — matching
/// `FitOptions::default`). #426: they were process globals that every `fit()` wrote and none
/// restored. Mutation: make the guard's `drop` a no-op → the post-drop reads keep the armed
/// value.
#[test]
fn inner_settings_guard_round_trips() {
    let read = || (inner_optimizer_mode(), ebe_warm_start_enabled());
    assert_eq!(read(), (InnerOptimizer::Auto, false), "default");
    let outer = arm_inner_settings(InnerSettings {
        mode: InnerOptimizer::Lbfgs,
        warm: true,
    });
    assert_eq!(read(), (InnerOptimizer::Lbfgs, true), "armed");
    let nested = arm_inner_settings(InnerSettings::default());
    assert_eq!(
        read(),
        (InnerOptimizer::Auto, false),
        "a nested default arm reads the default, not the enclosing call's settings"
    );
    drop(nested);
    assert_eq!(read(), (InnerOptimizer::Lbfgs, true), "nested restored");
    drop(outer);
    assert_eq!(read(), (InnerOptimizer::Auto, false), "outer restored");

    let unwound = std::panic::catch_unwind(|| {
        let _armed = arm_inner_settings(InnerSettings {
            mode: InnerOptimizer::NelderMead,
            warm: true,
        });
        panic!("inside the scope");
    });
    assert!(unwound.is_err());
    assert_eq!(read(), (InnerOptimizer::Auto, false), "restored on unwind");
}

/// A minimal non-IOV 1-cpt IV model (`CL = TVCL·exp(η)`, `V = TVV`) shared by
/// the plain-EBE and inner-restart tests.
fn no_iov_1cpt_model() -> CompiledModel {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let default_params = crate::types::ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.01, 1.0],
        theta_upper: vec![100.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.05],
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
        name: "no_iov".into(),
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
        gradient_method: GradientMethod::default(),
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

fn no_iov_subject(reset_times: Vec<f64>) -> Subject {
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 4.0],
        obs_raw_times: Vec::new(),
        observations: vec![40.0, 32.0, 20.0],
        obs_cmts: vec![1; 3],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times,
        reset_covariates: Vec::new(),
        cens: vec![0; 3],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

#[test]
fn test_find_ebe_no_iov_kappas_empty() {
    // A model without IOV should return empty kappas.
    let model = no_iov_1cpt_model();
    let subject = no_iov_subject(Vec::new());
    let params = model.default_params.clone();
    let result = find_ebe(&model, &subject, &params, 200, 1e-5, None, None, 0);
    assert!(result.kappas.is_empty());
}

/// The guarded inner multi-start (`inner_restarts > 0`) fires on a
/// reset-bearing subject (cold start), exercises the Ω-scaled seed scan, and
/// — because this 1-cpt objective is unimodal — reconverges to the SAME EBE
/// as `inner_restarts = 0`. This pins the "unimodal subjects stay
/// bit-identical" guarantee while covering the restart loop.
#[test]
fn test_inner_restart_unimodal_is_bit_identical() {
    let model = no_iov_1cpt_model();
    // reset_times non-empty ⇒ `has_resets()` ⇒ the restart trigger is armed.
    let subject = no_iov_subject(vec![0.0]);
    assert!(subject.has_resets());
    let params = model.default_params.clone();
    // Cold start (`eta_init = None`) is required for the restart to fire.
    let base = find_ebe(&model, &subject, &params, 200, 1e-6, None, None, 0);
    let restarted = find_ebe(&model, &subject, &params, 200, 1e-6, None, None, 2);
    assert!(restarted.eta[0].is_finite());
    assert!(
        (restarted.eta[0] - base.eta[0]).abs() < 1e-7,
        "unimodal EBE must be unchanged by the restart: base={} restarted={}",
        base.eta[0],
        restarted.eta[0]
    );
}

/// Regression guard for #302: the non-IOV inner EBE must be invariant to the
/// mu-reference shift — it is a pure reparametrization of the search frame.
/// The bug was searching the offset psi-space (`psi = eta + mu`), which
/// mis-scaled the FD gradient step (`~|psi|`) for a LARGE mu (additive
/// mu-refs, `mu = TVx`), driving the EBE to a wrong point. A large `mu_k`
/// must yield the same `eta_true` as `mu_k = None`.
#[test]
fn find_ebe_noniov_invariant_to_large_mu_shift() {
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let default_params = crate::types::ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
        theta: vec![5.0, 50.0],
        theta_names: vec!["TVCL".into(), "TVV".into()],
        theta_lower: vec![0.01, 1.0],
        theta_upper: vec![100.0, 500.0],
        theta_fixed: vec![false; 2],
        omega,
        omega_fixed: vec![false],
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
        residual_serial: None,
        priors: Vec::new(),
        prior_from_fit: None,
        covariate_model: None,
        frem_config: None,
        residual_error_eta: None,
        analytical_init: Vec::new(),
        analytic_readout: None,
        ruv_magnitude: None,
        absorption_ode_equivalent: None,
        mixture: None,
        name: "noniov_mu".into(),
        has_conditional_eta_params: false,
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
        gradient_method: GradientMethod::default(),
        parse_warnings: Vec::new(),
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
    };
    let subject = Subject {
        obs_rgroup: Vec::new(),
        fremtype: Vec::new(),
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 4.0],
        obs_raw_times: Vec::new(),
        observations: vec![40.0, 32.0, 20.0],
        obs_cmts: vec![1; 3],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 3],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let r_none = find_ebe(&model, &subject, &params, 200, 1e-6, None, None, 0);
    // A large mu (e.g. an additive mu-ref's typical value) is the case that
    // mis-converged in psi-space; the EBE must be unchanged.
    let r_mu = find_ebe(&model, &subject, &params, 200, 1e-6, None, Some(&[8.0]), 0);
    assert!(
        (r_none.eta[0] - r_mu.eta[0]).abs() < 1e-9,
        "non-IOV EBE must be mu-shift invariant: none={}, mu=8 -> {}",
        r_none.eta[0],
        r_mu.eta[0]
    );
}

#[test]
fn test_find_ebe_iov_honors_mu_shift() {
    // With mu-referencing, the IOV inner loop must shift its BSV optimization
    // variable by mu so the returned EBE is mean-zero (psi - mu), matching
    // the non-IOV path's NONMEM-compatible convention. Two equivalent fits
    // — same data, same params, but expressed with vs. without a mu shift —
    // should yield essentially the same returned BSV eta.
    let model = make_iov_model();
    let subject = make_iov_subject();
    let params = model.default_params.clone();

    // Fit without mu_k.
    let r1 = find_ebe(&model, &subject, &params, 200, 1e-5, None, None, 0);

    // Fit with a non-zero mu_k. If mu were dropped, BSV eta would shift by
    // -mu; with the fix, BSV eta is recovered as psi - mu and matches r1.
    let mu = vec![0.1];
    let r2 = find_ebe(&model, &subject, &params, 200, 1e-5, None, Some(&mu), 0);

    // No `converged` assertion (#1527): this subject's objective is ~2.68e5 at the mode, where
    // the FD inner gradient's rounding floor is ε·|f|/h ≈ 2.2e-16·2.68e5/8.6e-7 ≈ 7e-5 — seven
    // times `tol` — so whether a run is *certified* there is luck, not a property of the
    // mu-shift. Measured: r1 stops on BFGS `gnorm < tol` (`true`); r2 reaches the Nelder–Mead
    // fallback and lands 2.2e-7 from r1, which the stationarity certificate cannot certify
    // (`false`) and Nelder–Mead's simplex spread used to (`true`). The invariant is the η.
    assert!(r1.eta[0].is_finite() && r2.eta[0].is_finite());
    // This fixture's BSV mode sits far out (η̂ ≈ −7.6) where the FD-gradient
    // inner objective is very flat: the gnorm < 1e-5 stop is satisfied across
    // an η basin wider than 1e-4, so the exact landing point is line-search
    // path dependent. The invariant under test is that mu-referencing is
    // *honored* — if it were dropped the two runs would differ by ~mu (0.1).
    // The realised gap (~2e-4) is two orders of magnitude smaller, so a 1e-3
    // bound robustly distinguishes "applied" from "dropped".
    assert!(
        (r1.eta[0] - r2.eta[0]).abs() < 1e-3,
        "mu shift not applied: r1.eta={}, r2.eta={}",
        r1.eta[0],
        r2.eta[0],
    );
}

/// Interaction of the #486 analytic `ExpressionScale` path with correlated residual
/// error (`block_sigma`): as of #627 the dense-R gradient serves correlated residuals
/// on BOTH loops, and the η-dependent `obs_scale` composes with it (the scale enters
/// only through `∂f/∂η` and the scaled prediction, both of which the dense assembly
/// consumes). So a model with BOTH features is now analytic on both loops —
/// `analytic_inner_common_bail` false, `analytic_outer_gradient_available` true. The
/// diagonal control stays analytic too. Inverts the previous FD-pinning assertion.
#[test]
fn expression_scale_with_correlated_residual_is_analytic_both_loops() {
    use crate::parser::model_parser::parse_model_string;
    let corr = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.10, 1.00]\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n",
        )
        .expect("parse ExpressionScale + correlated residual");
    assert!(
        matches!(
            corr.scaling,
            crate::types::ScalingSpec::ExpressionScale { .. }
        ) && !corr.residual_correlations.is_empty(),
        "fixture must carry both an ExpressionScale obs_scale and a residual correlation"
    );
    assert!(
        !analytic_inner_common_bail(&corr),
        "block_sigma + ExpressionScale is now analytic inner (#627)"
    );
    assert!(
        crate::sens::provider::analytic_outer_gradient_available(&corr),
        "block_sigma + ExpressionScale is now analytic outer (#627)"
    );

    // Control: same obs_scale, diagonal (uncorrelated) residual → analytic on both loops.
    let diag = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.04\n  sigma ADD_ERR ~ 0.10\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ combined(PROP_ERR, ADD_ERR)\n",
        )
        .expect("parse ExpressionScale + diagonal residual");
    assert!(diag.residual_correlations.is_empty());
    assert!(!analytic_inner_common_bail(&diag));
    assert!(crate::sens::provider::analytic_outer_gradient_available(
        &diag
    ));
}

// `analytical_ad_unsupported` is the VESTIGIAL retired-AD classifier (not consulted by
// live routing; the live gate is `analytic_inner_grad_supported[_model]`). It still flags
// four genuinely out-of-scope classes (non-log-normal ETA, LTBS, conditional params, TTE)
// and — historically — any `ExpressionScale`. The final case below pins the deliberate
// DIVERGENCE for a differentiable `ExpressionScale`: the classifier flags it, but the live
// gate serves it analytically (#486). This guards against a regression that re-wires the
// classifier into routing. Build-independent; runs in the FD-only `ci` build (#278).
#[test]
fn analytical_ad_unsupported_flags_each_class() {
    use crate::parser::model_parser::parse_model_string;
    let mut model = make_iov_model();
    // Plain log-normal fixture -> supported.
    assert!(analytical_ad_unsupported(&model).is_none());

    // Non-log-normal ETA.
    model.eta_param_info = vec![crate::types::EtaParamInfo {
        eta_name: "ETA_CL".into(),
        param_type: crate::types::EtaParamType::Additive,
        linked_theta: None,
        individual_param_name: "CL".into(),
    }];
    assert!(analytical_ad_unsupported(&model).is_some());
    model.eta_param_info.clear();
    assert!(analytical_ad_unsupported(&model).is_none());

    // LTBS.
    model.log_transform = true;
    assert!(analytical_ad_unsupported(&model).is_some());
    model.log_transform = false;
    assert!(analytical_ad_unsupported(&model).is_none());

    // Conditional parameter (structured flag).
    model.has_conditional_eta_params = true;
    assert!(analytical_ad_unsupported(&model).is_some());
    model.has_conditional_eta_params = false;
    assert!(analytical_ad_unsupported(&model).is_none());

    // Expression-scale obs_scale: the vestigial classifier flags any `ExpressionScale`.
    model.scaling = crate::types::ScalingSpec::ExpressionScale {
        scale_fn: Box::new(|_, _, _, _| 1.0),
        deriv: None,
    };
    assert!(analytical_ad_unsupported(&model).is_some());
    model.scaling = crate::types::ScalingSpec::ScalarScale(1000.0);
    assert!(analytical_ad_unsupported(&model).is_none());

    // DIVERGENCE pin (#486 / #534 audit): a *differentiable* η-dependent `ExpressionScale`
    // is still flagged by the vestigial classifier, but the LIVE inner gate serves it
    // analytically. If a future change re-wired `analytical_ad_unsupported` into routing,
    // it would silently send analytic ExpressionScale fits back to FD — assert both here so
    // that regression is caught.
    let scaled = parse_model_string(
            "[parameters]\n  theta TVCL(5.0,0.5,50.0)\n  theta TVV(50.0,5.0,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.09\n  sigma PROP_ERR ~ 0.05\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n  obs_scale = 1000 / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n",
        )
        .expect("parse differentiable ExpressionScale");
    assert!(matches!(
        scaled.scaling,
        crate::types::ScalingSpec::ExpressionScale { deriv: Some(_), .. }
    ));
    assert!(
        analytical_ad_unsupported(&scaled).is_some(),
        "vestigial classifier still flags any ExpressionScale"
    );
    assert!(
        analytic_inner_grad_supported_model(&scaled),
        "but the LIVE inner gate serves a differentiable ExpressionScale analytically (#486)"
    );
}

/// #1182: the `power(σ, P)` form's analytic inner η-gradient must match FD of
/// the (exponent-aware) `individual_nll` — the exponent rides the magnitude
/// channel, so `residual_inner_obs` sees `|f|^{p}` through `dvar_df_scaled`.
#[test]
fn power_exponent_inner_eta_gradient_matches_fd() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
            "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  theta TVKA(1.5,0.01,50.0)\n  theta RUV_POW(1.3,0.01,10.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  omega ETA_KA ~ 0.30\n  sigma PROP_ERR ~ 0.04\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ power(PROP_ERR, RUV_POW)\n",
        )
        .expect("parse power model");
    assert!(model.has_ruv_exponent());
    let mut subject = Subject {
        id: "1".into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0],
        observations: vec![0.0; 7],
        obs_cmts: vec![1; 7],
        cens: vec![0; 7],
        occasions: vec![1; 7],
        ..Default::default()
    };
    let theta = vec![0.22, 11.0, 1.4, 1.3];
    let preds =
        crate::pk::compute_predictions_with_tv(&model, &subject, &theta, &[0.1, -0.1, 0.05]);
    subject.observations = preds.iter().map(|p| p * 0.85).collect();
    let mut params = model.default_params.clone();
    params.theta = theta.clone();
    let eta = [0.15_f64, -0.10, 0.20];
    let analytic = analytic_eta_nll_gradient(
        &model,
        &subject,
        &params.theta,
        &eta,
        &params.omega,
        &params.sigma.values,
    )
    .expect("power model is in the analytic inner scope");
    for k in 0..model.n_eta {
        let h = 1e-6 * (1.0 + eta[k].abs());
        let mut ep = eta;
        ep[k] += h;
        let mut em = eta;
        em[k] -= h;
        let nll = |e: &[f64]| {
            crate::stats::likelihood::individual_nll(
                &model,
                &subject,
                &params.theta,
                e,
                &params.omega,
                &params.sigma.values,
            )
        };
        let fd = (nll(&ep) - nll(&em)) / (2.0 * h);
        approx::assert_relative_eq!(analytic[k], fd, max_relative = 1e-5, epsilon = 1e-6);
    }
}

/// #1327: on an **exact** (closed-form) IOV objective, a BFGS "failure" must keep the
/// lower-objective of {BFGS partial, Nelder–Mead restart} — the `argmin_inner_fallback`
/// policy the non-IOV `find_ebe` has used since #378 — rather than adopt the NM result
/// unconditionally.
///
/// What it catches: a cold-started closed-form BFGS that arrives at the mode and stalls at a
/// gradient norm just above `tol` (busulfan DCM+IOV subject 261: partial at NLL 94.6,
/// |g| ≈ 7e-5 > 1e-5) was replaced by an NM from the cold seed that stops at NLL ≈ 5030.
/// Every cold-started evaluation — the final inner loop, an `outer_maxiter = 0`
/// re-evaluation — then scored such subjects thousands of −2LL units too high, while the
/// warm-started trajectory (BFGS converging within `tol` from the previous mode) never
/// tripped the fallback: the reported OFV sat 9 300 above the value the optimizer minimised.
///
/// The fixture forces the fallback deterministically: `max_iter = 1` makes BFGS report
/// non-convergence after a single exact-gradient line search, which already moves the
/// objective well below the cold seed's; the NM restart it triggers runs `5 · max_iter`
/// iterations from the zero seed with NM's `0.00025` initial simplex step, so it barely
/// leaves the seed (NLL 504.9 → 501.0). Under the pre-#1327 policy that NM point is
/// returned; under the fix the partial (NLL −1.1, against a converged mode of −4.1) is
/// kept. Verified by mutation: restoring the unconditional-NM arm turns this red.
#[test]
fn iov_inner_fallback_keeps_bfgs_partial_over_worse_cold_nm() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
        r#"
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  iov_column = OCC
"#,
    )
    .expect("closed-form IOV model parses");
    assert!(
        !skip_ode_iov_nm_fallback(&model),
        "fixture must take the closed-form NM fallback arm"
    );
    let _cold = arm_inner_settings(InnerSettings::default());

    // Two occasions, one 100 mg IV bolus each; observations simulated at
    // CL = 5·exp(−1.5), V = 50 (see the test doc) so the mode is far from the zero seed.
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![2.0, 6.0, 26.0, 30.0],
        obs_raw_times: Vec::new(),
        observations: vec![1.913, 1.749, 3.032, 2.773],
        obs_cmts: vec![1; 4],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let cold_nll = individual_nll_iov(
        &model,
        &subject,
        &params.theta,
        &[0.0],
        &[vec![0.0], vec![0.0]],
        &params.omega,
        params.omega_iov.as_ref(),
        &params.sigma.values,
    );

    // `max_iter = 1`: BFGS cannot certify convergence, so the fallback fires.
    let ebe = find_ebe_iov(
        &model,
        &subject,
        &params,
        1,
        1e-5,
        None,
        None,
        EbeVariance::Conditional,
    );
    assert!(
        !ebe.converged && ebe.used_fallback,
        "fixture must trip the NM fallback (converged={}, fallback={})",
        ebe.converged,
        ebe.used_fallback
    );
    assert!(ebe.nll.is_finite(), "fallback returned a non-finite NLL");

    // The converged solve is the floor and the yardstick. Measured on this fixture:
    // cold-seed NLL 504.88; converged mode −4.13; the one-line-search BFGS partial −1.11
    // (gap 3.0); the unconditionally-adopted 5-iteration NM from the zero seed 500.98
    // (gap 505). The bound is ~3× the realised gap of the kept partial and 50× below the
    // old policy's, so it fails on exactly the mutation it exists to catch and nothing else.
    let full = find_ebe_iov(
        &model,
        &subject,
        &params,
        200,
        1e-5,
        None,
        None,
        EbeVariance::Conditional,
    );
    assert!(full.converged, "reference solve must converge");
    assert!(
        ebe.nll >= full.nll - 1e-9,
        "kept partial ({:.4}) cannot undercut the converged mode ({:.4})",
        ebe.nll,
        full.nll
    );
    assert!(
        ebe.nll < full.nll + 10.0,
        "the BFGS partial must survive the fallback: returned NLL {:.4}, converged mode {:.4}, \
         cold-seed NLL {:.4} (the pre-#1327 unconditional NM restart returns ≈ 501)",
        ebe.nll,
        full.nll,
        cold_nll
    );
}

/// PR #1337 review (P1): the convergence flag must belong to the point that is returned.
///
/// Scenario: the BFGS partial sits on the slope of the deep well (not a mode) and wins on
/// objective; the cold NM restart, seeded exactly at the shallow well's minimum, converges
/// there. The old helper returned the partial with the *restart's* `true`, reporting a
/// non-stationary point as a converged EBE — which bypasses `max_unconverged_frac` and
/// AGQ's per-subject acceptance. Two assertions, one per half of the fix:
///
/// * with a budget too small to polish the partial to the mode, the returned point is at
///   least as good as the partial and the flag is `false` (the restart's `true` must not
///   leak through);
/// * with a normal budget the polish reaches the deep mode and reports `true` — the flag is
///   earned by the returned point, not inherited.
///
/// Objective: deep well at x = −2 (f = −10), shallow well at x = +2 (f = −1). Budgets:
/// `max_iter = 1` gives each NM 5 iterations — enough for a restart seeded on the shallow
/// minimum to collapse its 0.1-wide simplex under `tol = 1e-3`, not enough for a polish
/// from x = −1.5 to reach x = −2 within `tol` — and gives the stationarity certifier's
/// gradient polish (#1527) one BFGS iteration, which ends before its next `gnorm < tol`
/// test. Verified by mutation: returning the restart's flag turns the first half red.
#[test]
fn argmin_inner_fallback_flag_belongs_to_the_returned_point() {
    let obj = bimodal_obj;
    let _cold = arm_inner_settings(InnerSettings::default());
    let partial = [-1.5];
    let f_partial = obj(&partial);

    // Tiny budget: the cold restart converges in the shallow well, the partial wins, the
    // polish cannot certify it.
    let (eta, ok) = argmin_inner_fallback(
        &obj,
        &partial,
        &[2.0],
        &exact_cert(&bimodal_grad, 1, 1, 1e-3),
    );
    assert!(eta[0] < 0.0, "must stay in the deep well, got {eta:?}");
    assert!(
        obj(&eta) <= f_partial,
        "returned point ({:.4}) must not be worse than the partial ({f_partial:.4})",
        obj(&eta)
    );
    assert!(
        !ok,
        "a partial that was neither BFGS- nor NM-certified must not report converged \
         (the restart converged in the shallow well; its flag must not leak)"
    );

    // Normal budget: the polish reaches the deep mode and earns the flag.
    let (eta2, ok2) = argmin_inner_fallback(
        &obj,
        &partial,
        &[2.0],
        &exact_cert(&bimodal_grad, 1, 200, 1e-8),
    );
    assert!(
        (eta2[0] + 2.0).abs() < 1e-2,
        "polish must reach the deep mode, got {eta2:?}"
    );
    assert!(ok2, "a stationary returned point is certified");
    assert!(
        bimodal_grad(&eta2)[0].abs() < 1e-8,
        "`converged` must mean stationary at the returned point (#1527), got |g| = {:.3e}",
        bimodal_grad(&eta2)[0].abs()
    );
}

/// #1527: the fallback's `converged` is **stationarity at the returned point**, not
/// Nelder–Mead's `frange < tol` simplex spread. Three cases, both sides of the gate in one
/// test so forcing the certificate either way reddens it:
///
/// 1. **No stationary point exists.** `f(x) = 0.01·x` is a plane: NM's first simplex
///    (`x = 0` and `x = 0.00025`) already spans `frange = 2.5e-6 < tol = 1e-5`, so NM returns
///    `true` on its very first check — at a point whose gradient is 1e-2, 1000× `tol`, and on
///    an objective that has *no* mode. Must report `converged == false`, and keep the point
///    (the polish cannot certify anything, so it must not move the result either).
///    **Red on `main`**, which returned NM's `true`.
/// 2. **NM collapses short of a genuine mode.** `f(x) = 50·(x − 1)²` from `x = 0.9`: NM stops
///    once its spread is under `tol`, i.e. at `|x − 1| ≈ √(tol/50)` and a gradient of ~1e-2 —
///    measured below. The certificate must not take that point; the gradient polish carries it
///    to the mode, and the returned point must be stationary under `tol`. **Red on `main`**,
///    which returned the NM point (gradient ≫ `tol`) as converged.
/// 3. **The control:** the same bowl handed its exact mode is certified unchanged by the
///    gradient test alone.
#[test]
fn argmin_inner_fallback_certifies_on_stationarity_not_simplex_spread() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;

    // 1. A plane: NM's simplex spread is under `tol` at once, and nothing is stationary.
    let plane = |x: &[f64]| 0.01 * x[0];
    let plane_grad = |_: &[f64]| vec![0.01];
    let mut nm_point = vec![0.0];
    assert!(
        nelder_mead_minimize(&plane, &mut nm_point, 1, 250, tol),
        "fixture: NM must report its own `frange < tol` success on the plane, or this case \
         does not reproduce #1527"
    );
    let (eta, ok) =
        argmin_inner_fallback(&plane, &[0.0], &[0.0], &exact_cert(&plane_grad, 1, 50, tol));
    assert!(
        !ok,
        "NM's simplex collapsed on a plane with |∇f| = 1e-2 = 1000·tol; that is not a \
         converged EBE (#1527), got converged = true at {eta:?}"
    );
    assert_eq!(
        eta, nm_point,
        "an uncertified polish must not move the returned point"
    );

    // 2. A bowl NM approaches but does not reach under a gradient `tol`.
    let bowl = |x: &[f64]| 50.0 * (x[0] - 1.0).powi(2);
    let bowl_grad = |x: &[f64]| vec![100.0 * (x[0] - 1.0)];
    let mut nm_end = vec![0.9];
    nelder_mead_minimize(&bowl, &mut nm_end, 1, 250, tol);
    let nm_gnorm = bowl_grad(&nm_end)[0].abs();
    assert!(
        nm_gnorm.is_finite() && nm_gnorm > 10.0 * tol,
        "fixture: NM must stop short of stationarity here, got |g| = {nm_gnorm:.3e}"
    );
    let (eta2, ok2) =
        argmin_inner_fallback(&bowl, &[0.9], &[0.9], &exact_cert(&bowl_grad, 1, 50, tol));
    let g2 = bowl_grad(&eta2)[0].abs();
    assert!(g2.is_finite(), "returned a non-finite gradient at {eta2:?}");
    assert!(
        ok2 && g2 < tol,
        "the gradient polish must certify the bowl's mode: converged = {ok2}, |g| = {g2:.3e} \
         (NM alone stopped at |g| = {nm_gnorm:.3e})"
    );

    // 3. Control: already at the mode → certified by the gradient test, point unchanged.
    let mut at_mode = vec![1.0];
    assert!(exact_cert(&bowl_grad, 1, 50, tol).certify(&bowl, &mut at_mode));
    assert_eq!(at_mode, vec![1.0]);
}

/// #1527's Newton-step test (test 2 of [`FallbackCertifier`]): a point within `tol` of a
/// **positive-definite** mode is certified in place even when its gradient norm is far above
/// `tol` — the stiff-direction case no line search can resolve, which is what sends such
/// subjects to the fallback in the first place — and a point the same distance from a
/// **saddle** is not. Both sides of the PD gate in one test:
///
/// * stiff bowl `f = ½·1e10·(x − 1)²` at `x = 1 + 1e-8`: `‖∇‖ = 100 = 1e7·tol`, Newton step
///   `1e-8`, and a remaining decrease of `5e-7` — far above the resolution arm's
///   `INNER_FTOL_REL·(1 + |f|) ≈ 1e-11`, so only the **step** arm can certify it. Certified
///   **and returned bit-unchanged**: the gradient polish (test 3) would also certify it, but
///   only after moving it, so `assert_eq!` on the point is what kills a mutation that deletes
///   test 2 or its step arm.
/// * saddle `f = 5e3·(x² − y²)` at `(1e-8, 1e-8)`: `‖∇‖ = 1.4e-4 > tol`, Newton step
///   `‖H⁻¹∇‖ = 1.4e-8 < tol`, but `H = diag(1e4, −1e4)` is indefinite — not a mode. Must not
///   certify (the polish runs off down `y` and does not certify either), and the point must
///   come back unchanged. Mutation: solving `H⁻¹∇` without the Cholesky (PD) requirement
///   certifies the saddle and reddens this half.
#[test]
fn fallback_certifier_newton_step_needs_a_positive_definite_mode() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;

    let stiff = |x: &[f64]| 0.5e10 * (x[0] - 1.0).powi(2);
    let stiff_grad = |x: &[f64]| vec![1e10 * (x[0] - 1.0)];
    let start = vec![1.0 + 1e-8];
    let g0 = stiff_grad(&start)[0].abs();
    assert!(
        g0 > 10.0 * tol,
        "fixture: the gradient test alone must not pass here, got |g| = {g0:.3e}"
    );
    let mut x = start.clone();
    assert!(
        exact_cert(&stiff_grad, 1, 50, tol).certify(&stiff, &mut x),
        "a point 1e-8 from a PD mode is within `tol` of it and must be certified"
    );
    assert_eq!(
        x, start,
        "the Newton-step test certifies in place, without moving the point"
    );

    let saddle = |x: &[f64]| 5e3 * (x[0] * x[0] - x[1] * x[1]);
    let saddle_grad = |x: &[f64]| vec![1e4 * x[0], -1e4 * x[1]];
    let at = vec![1e-8, 1e-8];
    let gs = saddle_grad(&at);
    let gs_norm = (gs[0] * gs[0] + gs[1] * gs[1]).sqrt();
    assert!(
        gs_norm > tol,
        "fixture: the saddle point must fail the gradient test, got |g| = {gs_norm:.3e}"
    );
    let mut y = at.clone();
    assert!(
        !exact_cert(&saddle_grad, 2, 50, tol).certify(&saddle, &mut y),
        "a saddle is not a mode, however short its Newton step; certified at {y:?}"
    );
    assert_eq!(y, at, "an uncertified point is returned unchanged");
}

/// The certified Nelder–Mead recovery the FREM arm and both runaway-EBE guards use
/// (`FallbackCertifier::nelder_mead_recover`, #1527), and its admissibility gate:
///
/// * on a plane NM's simplex collapses at once and nothing is stationary → `false`;
/// * on a bowl NM stops short of the mode and the certificate's polish carries it there →
///   `true`, stationary under `tol`;
/// * the same bowl with an `admissible` predicate that excludes the mode's neighbourhood —
///   what the runaway guards pass, so a gradient polish cannot march back past the runaway
///   limit — must refuse the polished point: `false`, and the NM end point returned as is.
///   Mutation: ignoring `admissible` certifies the excluded mode and reddens this case.
#[test]
fn nelder_mead_recover_is_certified_and_honours_admissibility() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;

    let plane = |x: &[f64]| 0.01 * x[0];
    let plane_grad = |_: &[f64]| vec![0.01];
    let (_, ok) =
        exact_cert(&plane_grad, 1, 50, tol).nelder_mead_recover(&plane, vec![0.0], &|_| true);
    assert!(
        !ok,
        "a plane has no mode; NM's collapse must not certify one"
    );

    let bowl = |x: &[f64]| 50.0 * (x[0] - 1.0).powi(2);
    let bowl_grad = |x: &[f64]| vec![100.0 * (x[0] - 1.0)];
    let (x, ok) =
        exact_cert(&bowl_grad, 1, 50, tol).nelder_mead_recover(&bowl, vec![0.9], &|_| true);
    let g = bowl_grad(&x)[0].abs();
    assert!(
        ok && g < tol,
        "the bowl's mode is certified: converged = {ok}, |g| = {g:.3e}"
    );

    let mut nm_end = vec![0.9];
    nelder_mead_minimize(&bowl, &mut nm_end, 1, 250, tol);
    let (y, ok) =
        exact_cert(&bowl_grad, 1, 50, tol)
            .nelder_mead_recover(&bowl, vec![0.9], &|e: &[f64]| (e[0] - 1.0).abs() > 1e-5);
    assert!(
        !ok,
        "a polish into the excluded region must not certify, ended at {y:?}"
    );
    assert_eq!(y, nm_end, "the NM end point is returned unchanged");
}

/// The Newton test's **resolution arm** (#1527): a point whose remaining Newton decrease
/// `½∇ᵀH⁻¹∇` is at or below `INNER_FTOL_REL·(1 + |f|)` is a mode to the objective's precision
/// and is certified in place, whatever `tol` says. It is what keeps tight `inner_tol` fits —
/// Schnider at 1e-8, the AGQ score fixtures at 1e-11 — from reporting their converged
/// subjects unconverged. Straddle, one fixture, `f = 100 + ½(x − 1)²`, `tol = 1e-5`,
/// threshold `1e-11·101 ≈ 1.0e-9`:
///
/// * `x = 1 + 3e-5`: `‖∇‖ = 3e-5` and Newton step `3e-5`, both above `tol`, but the decrease
///   left is `4.5e-10` — certified, returned **unmoved**;
/// * `x = 1 + 1e-4`: decrease `5e-9`, above the threshold — not certified in place; the
///   polish certifies it at the mode instead, so the point **moves**.
///
/// Mutation: deleting the resolution arm sends the first case to the polish, which moves
/// it, and the `assert_eq!` reddens.
#[test]
fn fallback_certifier_certifies_a_mode_resolved_to_the_objectives_precision() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;
    let f = |x: &[f64]| 100.0 + 0.5 * (x[0] - 1.0).powi(2);
    let g = |x: &[f64]| vec![x[0] - 1.0];

    let inside = vec![1.0 + 3e-5];
    let mut x = inside.clone();
    assert!(exact_cert(&g, 1, 50, tol).certify(&f, &mut x));
    assert_eq!(
        x, inside,
        "a decrease below the objective's resolution is certified in place"
    );

    let outside = vec![1.0 + 1e-4];
    let mut y = outside.clone();
    assert!(exact_cert(&g, 1, 50, tol).certify(&f, &mut y));
    assert_ne!(
        y, outside,
        "a resolvable decrease is not certified in place — the polish takes it to the mode"
    );
    assert!((y[0] - 1.0).abs() < tol, "polished to the mode, got {y:?}");
}

/// On a numerical objective (`stall_at` true) the certificate skips the Newton-step test —
/// a finite-difference Hessian of a noise-floored gradient is not evidence — and lets the
/// stall-armed polish decide (#1527). Observable on an objective of magnitude ~1e12, where
/// the Newton test's resolution arm (`½∇ᵀH⁻¹∇ ≤ INNER_FTOL_REL·(1 + |f|)`, here ≤ 10)
/// accepts a point 0.1 from the mode (remaining decrease 0.5): run on a numerical objective
/// it would certify `x = 1.1` in place. The polish instead carries the point to the mode.
/// Mutation: applying the Newton test whatever `stall_at` says returns `x = 1.1` and
/// reddens this test.
#[test]
fn fallback_certifier_leaves_numerical_objectives_to_the_stall_polish() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;
    let big = |x: &[f64]| 1e12 + 50.0 * (x[0] - 1.0).powi(2);
    let big_grad = |x: &[f64]| vec![100.0 * (x[0] - 1.0)];
    let cert = FallbackCertifier {
        stall_at: &always_stall,
        ..exact_cert(&big_grad, 1, 200, tol)
    };
    let mut x = vec![1.1];
    assert!(
        cert.certify(&big, &mut x),
        "the stall polish certifies, ended at {x:?}"
    );
    assert!(
        (x[0] - 1.0).abs() < 1e-2,
        "a numerical objective's point must come from the polish, not be certified in place \
         0.1 from the mode by a Hessian it cannot trust: got {x:?}"
    );
}

/// The certificate's polish is a gradient driver even under `inner_optimizer = nelder_mead`
/// (#1527 review round 1): the fit-scoped dispatch would hand it back to Nelder–Mead, whose
/// simplex-spread exit would then certify exactly what tests 1 and 2 just refused. On the
/// plane NM collapses at once, so with the polish routed through the configured driver this
/// returns `true`; with a gradient driver it is `false`. The `auto` control in the same test
/// pins that the fixture refuses the plane on the ordinary path too.
#[test]
fn fallback_certifier_polish_is_gradient_based_under_nelder_mead_mode() {
    let plane = |x: &[f64]| 0.01 * x[0];
    let plane_grad = |_: &[f64]| vec![0.01];
    for mode in [
        crate::types::InnerOptimizer::Auto,
        crate::types::InnerOptimizer::NelderMead,
    ] {
        let _armed = arm_inner_settings(InnerSettings {
            mode,
            ..Default::default()
        });
        let mut x = vec![0.0];
        assert!(
            !exact_cert(&plane_grad, 1, 50, 1e-5).certify(&plane, &mut x),
            "{mode:?}: a plane has no mode; the polish must not certify one on simplex spread"
        );
        assert_eq!(
            x,
            vec![0.0],
            "{mode:?}: an uncertified point is returned unchanged"
        );
    }
}

/// Test 1 of the certificate is the BFGS criterion itself, `‖∇‖ < tol`, and it is what makes
/// the common case cheap: a fallback point that is already stationary costs **one** gradient
/// evaluation. Without it every such point would buy the Newton test's `2n` extra gradients
/// (the polish would still certify it, unmoved — so this cost is the only observable).
/// Mutation: deleting test 1 makes the count 1 + 2n = 5 here.
#[test]
fn fallback_certifier_certifies_a_stationary_point_with_one_gradient() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let calls = std::cell::Cell::new(0usize);
    let bowl = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
    let counted = |x: &[f64]| {
        calls.set(calls.get() + 1);
        vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)]
    };
    let mut x = vec![1.0, -2.0];
    assert!(exact_cert(&counted, 2, 50, 1e-5).certify(&bowl, &mut x));
    assert_eq!(
        calls.get(),
        1,
        "a stationary point is certified on its first gradient"
    );
}

/// Deterministic pseudo-noise in `[−1, 1)` — a hash of `x`'s bits, so a fixture can carry a
/// solver-like noise floor and still be reproducible.
fn hash_noise(x: f64) -> f64 {
    let mut h = x.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// #1527's stall arm: on a numerical objective the certificate's gradient polish runs the
/// inner driver **with the subject's objective-stall predicate** (#555 / #1521), so a mode
/// whose gradient is floored above `tol` by solver noise is certified rather than reported
/// unconverged. The fixture models an RK45 objective the way #1521 measured one: a bowl at
/// `f ≈ 1000` whose value carries `±1e-10` of deterministic noise (far under the stall's
/// `1e-11·(1 + |f|) ≈ 1e-8` flatness threshold) and whose gradient carries `±1e-4` (ten
/// times `tol`), so no point can pass `gnorm < tol` and the Newton-step test is skipped.
///
/// Both sides of the gate, one fixture: with `stall_at = true` the certificate holds and the
/// returned point is the mode to the floor's resolution; with `stall_at = false` (the
/// exact-objective criterion) the same start is **not** certified. Mutation: handing the
/// polish a constant-`false` stall predicate instead of the certifier's reddens the first
/// half; deleting the "skip test 2 on a numerical objective" guard does not (the Newton step
/// from this start is ~3e-3, far above `tol`), which is stated rather than claimed.
#[test]
fn fallback_certifier_honours_the_stall_stop_on_a_noise_floored_objective() {
    let _cold = arm_inner_settings(InnerSettings::default());
    let tol = 1e-5;
    let noisy = |x: &[f64]| 1000.0 + 50.0 * (x[0] - 1.0).powi(2) + 1e-10 * hash_noise(x[0]);
    let floored = |x: &[f64]| {
        let d = x[0] - 1.0;
        let floor = 1e-4 * (1.5 + 0.5 * hash_noise(x[0] + 0.5));
        vec![100.0 * d + if d >= 0.0 { floor } else { -floor }]
    };
    let start = vec![1.0 + 3e-3];

    let stall_cert = FallbackCertifier {
        stall_at: &always_stall,
        ..exact_cert(&floored, 1, 200, tol)
    };
    let mut x = start.clone();
    assert!(
        stall_cert.certify(&noisy, &mut x),
        "a noise-floored mode must be certified by the stall stop, ended at {x:?}"
    );
    assert!(
        (x[0] - 1.0).abs() < 1e-5,
        "the stall-certified point must be the mode to the floor's resolution, got {x:?}"
    );

    let mut y = start.clone();
    assert!(
        !exact_cert(&floored, 1, 200, tol).certify(&noisy, &mut y),
        "without the stall stop a floor above `tol` cannot be certified — if this passes, the \
         fixture no longer models a noise floor and the first half proves nothing"
    );
    assert_eq!(y, start, "an uncertified point is returned unchanged");
}

/// The behavioural side of `inner_stall_enabled_at_iov_tracks_the_effective_model`: a
/// transit + IOV subject the absorption walk serves has an **exact** objective, so its inner
/// solve must stop on `gnorm < tol`, not on the objective-stall plateau (#555) that exists for
/// RK45 noise. Keyed on `effective_for` — which still names the twin for every `n_kappa > 0`
/// transit subject — the stall fired on this exact objective and reported convergence early.
///
/// Measured on this fixture (the analytic stacked gradient at the returned `[η, κ₁, κ₂]`):
///
/// | `tol` | | `converged` | max \|∂l/∂(η,κ)\| |
/// |---|---|---|---|
/// | 1e-8 | fix | `true` | **1.49e-9** |
/// | 1e-8 | stall keyed on `effective_for` | `true` | **2.03e-8** (above `tol`) |
/// | 1e-10 | fix | **`false`** | 1.49e-9 (the exact objective's floor) |
/// | 1e-10 | stall keyed on `effective_for` | **`true`** | 2.03e-8 |
///
/// So at `tol = 1e-8` a converged solve must have a gradient below `tol` — 6.7× headroom over
/// the realised 1.49e-9, and the mutation lands 2× above it — and at `tol = 1e-10`, below the
/// floor, it must not claim convergence at all. Both are asserted; either catches the mutation.
#[test]
fn a_walk_served_iov_subject_stops_on_the_gradient_not_the_stall() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(TRANSIT_IOV_MODEL).expect("parse transit IOV");
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![0.5, 1.0, 2.0, 6.0, 12.0, 24.5, 25.0, 26.0, 30.0, 36.0],
        obs_raw_times: Vec::new(),
        observations: vec![3.0, 6.0, 8.0, 9.0, 8.5, 12.0, 14.0, 16.0, 15.0, 14.0],
        obs_cmts: vec![1; 10],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 10],
        occasions: vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };
    let params = model.default_params.clone();
    let solve = |tol: f64| {
        let r = find_ebe_iov(
            &model,
            &subject,
            &params,
            200,
            tol,
            None,
            None,
            EbeVariance::Conditional,
        );
        let mut stacked: Vec<f64> = r.eta.as_slice().to_vec();
        for k in &r.kappas {
            stacked.extend_from_slice(k.as_slice());
        }
        assert!(
            std::ptr::eq(
                crate::pk::effective_model_for_eval_iov(&model, &subject, &params.theta, &stacked),
                &model
            ),
            "fixture drift: the returned EBE must be served by the absorption walk"
        );
        let g = analytic_eta_nll_gradient_iov(
            &model,
            &subject,
            &params.theta,
            &stacked,
            &params.omega,
            params.omega_iov.as_ref().expect("IOV model"),
            &params.sigma.values,
            model.n_eta,
            model.n_kappa,
            r.kappas.len(),
            None,
        )
        .expect("the analytic IOV gradient serves a walk subject");
        assert!(g.iter().all(|v| v.is_finite()), "gradient {g:?}");
        let gmax = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        (r.converged, gmax)
    };

    let (conv, gmax) = solve(1e-8);
    assert!(
        conv && gmax < 1e-8,
        "an exact objective must stop on gnorm < tol: converged {conv}, max |g| {gmax:e}"
    );
    let (conv, gmax) = solve(1e-10);
    assert!(
        !conv,
        "below the exact objective's floor the solve must not claim convergence on a plateau \
         (max |g| {gmax:e})"
    );
}

/// #1082 PR3: the one nested model the analytic walks decline — a closed-form transit
/// one, whose IOV subjects run on an ODE twin — names its levels as the finite-difference
/// reason, ahead of any per-subject cause. The same model with one level, and a nested
/// model without a twin, do not: PR2 named the levels for every nested model.
#[test]
fn a_nested_transit_model_names_its_levels_as_the_fd_reason() {
    const REASON: &str = "nested variability levels on a closed-form transit/IG model";
    let parse = crate::parser::model_parser::parse_model_string;
    let one_level = |src: &str| {
        parse(
            &src.replace("  level B by ARM in A\n", "")
                .replace("at B", "at A"),
        )
    };
    let algebraic = "[parameters]\n  theta T(1.0)\n  omega E ~ 0.1\n  level A by STUDY in ID\n  \
                     level B by ARM in A\n  kappa KA ~ 0.2 at A\n  kappa KB ~ 0.3 at B\n  \
                     sigma S ~ 0.1\n[individual_parameters]\n  P = T * exp(E + KA + KB)\n\
                     [structural_model]\n  y = P\n[error_model]\n  DV ~ additive(S)\n";
    let transit = "[parameters]\n  theta TVCL(5.0, 0.1, 100.0)\n  theta TVV(50.0, 5.0, 500.0)\n  \
                   theta TVMTT(3.0, 0.05, 24.0)\n  theta TVN(3.0, 0.0, 30.0)\n  \
                   omega ETA_V ~ 0.09\n  level A by STUDY in ID\n  level B by ARM in A\n  \
                   kappa KA ~ 0.04 at A\n  kappa KB ~ 0.09 at B\n  sigma PROP ~ 0.15 (sd)\n\
                   [individual_parameters]\n  CL = TVCL * exp(KA)\n  V = TVV * exp(ETA_V)\n  \
                   MTT = TVMTT * exp(KB)\n  NTR = TVN\n[structural_model]\n  \
                   pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)\n[error_model]\n  \
                   DV ~ proportional(PROP)\n";
    let subject = Subject {
        obs_times: vec![1.0],
        observations: vec![1.0],
        obs_cmts: vec![1],
        cens: vec![0],
        occasions: vec![1],
        ..Default::default()
    };
    let nested_transit = parse(transit).unwrap();
    assert!(nested_transit.has_nested_levels());
    assert_eq!(iov_fd_reason(&nested_transit, &subject), REASON);
    for (label, m) in [
        ("transit, one level", one_level(transit).unwrap()),
        ("algebraic, nested", parse(algebraic).unwrap()),
        ("algebraic, one level", one_level(algebraic).unwrap()),
    ] {
        assert_ne!(iov_fd_reason(&m, &subject), REASON, "{label}");
    }
}

/// #1082 PR3: the analytic IOV inner gradient of a nested-level model is over the compact
/// vector the EBE search moves — data term from the walk's compact jet, prior one block per
/// group of each level — and matches central FD of the inner objective over that vector,
/// on each walk. A prior laid out per cell (`κ_cellᵀΩ⁻¹κ_cell` for every arm, counting a
/// study κ once per arm) or a study block read without its off-diagonal fails here, named
/// by axis.
#[test]
fn nested_inner_gradient_matches_fd_of_the_inner_objective_over_the_compact_vector() {
    use crate::stats::nested_levels::test_helpers as fx;
    use crate::stats::nested_levels::NestedLayout;
    for (label, src) in [
        ("algebraic", fx::algebraic()),
        ("closed form", fx::closed_form()),
        ("ode", fx::ode()),
    ] {
        let (model, pop) = fx::prepared(&src, 1);
        let subject = &pop.subjects[0];
        let params = model.default_params.clone();
        let omega_iov = params.omega_iov.as_ref().expect("omega_iov");
        assert!(
            crate::sens::provider::iov_sens_eta_supported(&model),
            "{label}: the inner gradient must be analytic"
        );
        let layout = NestedLayout::of(&model, subject).expect("prepared");
        let b = fx::point(&model, &layout, 1.0);
        let k = iov_occasion_groups(subject).len();
        let mult = model.ruv_obs_mult(subject, &params.theta);
        let g = analytic_eta_nll_gradient_iov(
            &model,
            subject,
            &params.theta,
            &fx::stacked(&layout, &b),
            &params.omega,
            omega_iov,
            &params.sigma.values,
            model.n_eta,
            model.n_kappa,
            k,
            mult.as_deref(),
        )
        .unwrap_or_else(|| panic!("{label}: the analytic inner gradient declined"));
        assert_eq!(g.len(), layout.n_compact(), "{label}");
        let nll = |bb: &[f64]| -> f64 {
            individual_nll_iov(
                &model,
                subject,
                &params.theta,
                &bb[..model.n_eta],
                &layout.expand(bb),
                &params.omega,
                Some(omega_iov),
                &params.sigma.values,
            )
        };
        let mut worst = 0.0_f64;
        for p in 0..b.len() {
            let h = 1e-6 * (1.0 + b[p].abs());
            let mut bp = b.clone();
            bp[p] += h;
            let mut bm = b.clone();
            bm[p] -= h;
            let fd = (nll(&bp) - nll(&bm)) / (2.0 * h);
            assert!(g[p].is_finite() && fd.is_finite(), "{label}");
            let e = (g[p] - fd).abs() / (1e-6 + 1e-4 * fd.abs());
            worst = worst.max(e);
            assert!(
                e < 1.0,
                "{label}: ∂NLL/∂[{}] analytic {} vs FD {fd}",
                fx::axis_label(&model, &layout, p),
                g[p]
            );
        }
        eprintln!("MEASURE nested inner NLL gradient {label}: worst margin {worst:.3e}");
    }
}

/// #1082 PR3: the two nested cases the analytic walks decline do so at the walks' own
/// entry, not only in the model-level predicates — so a caller that skips the predicate
/// still gets `None` (→ FD) rather than a jet laid out per cell — and the FD-fallback
/// reason names each: a closed-form transit model (its subjects run on an ODE twin) and a
/// subject never given its level cells (its compact axes are unknown). The prepared
/// closed-form / ODE twins of each fixture are served, so the declines are the cases' own.
#[test]
fn the_nested_walks_decline_a_transit_twin_and_an_unprepared_subject() {
    use crate::sens::provider::{subject_eta_grad_iov, subject_sensitivities_iov};
    use crate::stats::nested_levels::test_helpers as fx;
    let walks = |m: &CompiledModel, s: &Subject| {
        let k = iov_occasion_groups(s).len();
        let stacked = vec![0.01; m.n_eta + k * m.n_kappa];
        let th = &m.default_params.theta;
        (
            subject_sensitivities_iov(m, s, th, &stacked).is_some(),
            subject_eta_grad_iov(m, s, th, &stacked).is_some(),
        )
    };
    let (transit, pop) = fx::prepared(&fx::transit(), 1);
    assert!(transit.absorption_ode_equivalent.is_some() && transit.has_nested_levels());
    assert_eq!(
        walks(&transit, &pop.subjects[0]),
        (false, false),
        "transit twin"
    );
    assert_eq!(
        iov_fd_reason(&transit, &pop.subjects[0]),
        "nested variability levels on a closed-form transit/IG model"
    );
    for src in [fx::closed_form(), fx::ode()] {
        let (m, pop) = fx::prepared(&src, 1);
        let prepared = &pop.subjects[0];
        assert_eq!(
            walks(&m, prepared),
            (true, true),
            "the prepared twin is served"
        );
        // Unprepared: occasion labels no longer parallel to the records.
        let mut bare = prepared.clone();
        bare.occasions.pop();
        assert!(crate::stats::stacked_layout::StackedLayout::sens_of(&m, &bare).is_none());
        assert_eq!(walks(&m, &bare), (false, false), "unprepared subject");
        if m.ode_spec.is_some() {
            assert_eq!(
                iov_fd_reason(&m, &bare),
                "nested-level subject without level cells"
            );
        }
    }
}
